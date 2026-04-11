# Router Module Documentation

> Source code: `router/`

---

## Module Overview

`router/` is the **dynamic sensing and scheduling** layer of nanoPD. It continuously observes the live system load at runtime and decides, for each incoming request, whether to route it to the Collocated or Disaggregated path. The module is composed of three classes in a layered stack:

```
OutputLengthPredictor          ← online learning: predict output length from prompt_len
        ↓
Router                         ← per-request routing decision (calls AnalyticalCostModel)
        ↓
CentralScheduler               ← global orchestration: live load sensing, concurrent execution
```

---

## OutputLengthPredictor

```python
class OutputLengthPredictor:
    def __init__(self, buckets=[64,128,256,512,1024,2048], window=50, default=256, min_samples=5)
```

An online sliding-window predictor that estimates output length based on which bucket `prompt_len` falls into.

### Data Structures

- `_bucket_data`: one `deque` of capacity `window` per bucket, storing the most recent N **actual** output lengths seen in that bucket.
- `_global_data`: a global sliding window (capacity = `window × num_buckets`) aggregating all buckets, used as a cold-start fallback.

### `predict(prompt_len) -> int`

Three-level fallback strategy:

```
1. Find the bucket for prompt_len (bisect_left; values beyond the largest bucket map to the last bucket)
2. If the bucket has >= min_samples entries → return the bucket mean
3. Else if the global window is non-empty → return the global mean
4. Else → return default (256)
```

During cold start all requests return `default`; as each bucket accumulates enough samples, prediction accuracy converges toward the true distribution for that length range.

### `update(prompt_len, actual_output_len)`

Called back by `CentralScheduler.run_until_done()` after each request completes. Writes the true output length into the matching bucket and the global window, implementing **online adaptive learning**.

### `stats() -> dict`

Returns the sample count and bucket-specific mean for each bucket, useful for monitoring per-range prediction quality.

---

## Router

```python
class Router:
    def __init__(self, cost_model: AnalyticalCostModel, predictor: OutputLengthPredictor = None)
    @classmethod
    def from_params(cls, params_path: str, **predictor_kwargs) -> "Router"
```

A thin façade that chains `OutputLengthPredictor` and `AnalyticalCostModel` into a single routing call.

### `route(prompt_len, system_load, decode_batch_size=1) -> str`

```python
predicted_output_len = self.predictor.predict(prompt_len)
decision, t_c, t_d   = self.cost_model.route(
    prompt_len, predicted_output_len, system_load,
    decode_batch_size=decode_batch_size
)
self._history.append((prompt_len, predicted_output_len, decision))
return decision   # "collocated" | "disaggregated"
```

The routing logic itself lives entirely in `AnalyticalCostModel.route()` (see [04-cost_model_en.md](04-cost_model_en.md)): compare `T_collocated` and `T_disaggregated`, return the lower-cost path.

Two key **runtime dynamic quantities** are passed into the decision:

- **`system_load`**: the total number of sequences currently executing in the system, which scales the interference penalty `γ × system_load`.
- **`decode_batch_size`**: the current batch size of the Decode Worker, which affects the amortised benefit of the KV transfer during the decode phase.

This means the same `prompt_len` may be routed differently depending on whether the system is idle (load=0) or under high load (load=8).

### `update(prompt_len, actual_output_len)`

Forwards to `predictor.update()`, feeding the true output length of a completed request back to the predictor.

### `decision_stats() -> dict`

Returns a summary of historical routing decisions: total requests, per-path counts, and the disaggregated ratio.

---

## CentralScheduler

```python
class CentralScheduler:
    def __init__(self, collocated_worker, prefill_workers, decode_worker, router, block_size=16)
    @classmethod
    def build(cls, model_path, params_path, collocated_gpu=0,
              prefill_gpus=[1], decode_gpu=2, block_size=16, max_blocks=512) -> "CentralScheduler"
```

The global orchestrator of the system. It holds all workers and the router, and drives the full P/D inference loop.

### Members

| Field | Type | Description |
|---|---|---|
| `cw` | `CollocatedWorker` | Collocated-path execution unit |
| `pw_list` | `List[PrefillWorker]` | Disagg-path PrefillWorker pool (multi-GPU support) |
| `dw` | `DecodeWorker` | Disagg-path DecodeWorker |
| `router` | `Router` | Routing decision engine |
| `_waiting` | `List[(rid, prompt)]` | Queue of requests awaiting dispatch |
| `_states` | `Dict[str, _RequestState]` | Runtime state for every request |
| `_prefill_threads` | `Dict[int, Thread]` | Background thread per PrefillWorker |
| `_prefill_done` | `List[...]` | Results written by prefill threads (protected by `_prefill_lock`) |

`_RequestState` tracks each request's `group`, `prompt_len`, `path` (routing decision), `output_token_ids`, and `finished` flag.

### `build()`

Factory method that creates all dependencies in one call:

- A single `BlockSpaceManager` is shared by the Disagg path's `PrefillWorker`(s) and `DecodeWorker`.
- The `prefill_gpus` list enables multiple `PrefillWorker` instances, each bound to a different GPU.
- `CollocatedWorker` holds its own `Engine` with an independent block manager.

### Core Loop: `step()`

```python
def step(self):
    self._dispatch_waiting()     # 1. route waiting requests, dispatch to workers
    self._flush_prefill_done()   # 2. push completed prefill results to DecodeWorker

    t_coll   = Thread(target=self._step_collocated)
    t_disagg = Thread(target=self._step_disaggregated)
    t_coll.start(); t_disagg.start()
    t_coll.join();  t_disagg.join()   # 3. advance both paths concurrently by one step
```

The Collocated side (`_step_collocated`) and Disaggregated side (`_step_disaggregated`) run in two independent threads simultaneously. The wall-clock cost of a `step()` equals the **maximum** of the two.

### `_dispatch_waiting()`: the live load-sensing core

At the start of every `step()`, the current system state is **read fresh**:

```python
system_load = (
    len(cw.engine.scheduler.running)   # sequences running on the Collocated path
  + len(dw.running)                    # sequences being decoded by the Decode Worker
  + len(dw._pending)                   # sequences waiting for KV transfer to complete
)
decode_batch_size = len(dw.running) + len(dw._pending)
```

These two runtime quantities are passed to `router.route()`, making routing decisions **aware of the actual live load** rather than any static configuration.

Each waiting request is handled as follows:

```
if decode_batch_size >= MAX_DECODE_BATCH (20):
    force "collocated"  (guard against DecodeWorker overload)
else:
    path = router.route(prompt_len, system_load, decode_batch_size)

if path == "collocated":
    cw.engine.add_request(prompt)
    system_load += 1
else:
    append to disaggregated_groups; optimistically decode_batch_size += 1
```

`system_load` and `decode_batch_size` are **incremented per request** inside the loop, so later requests in the same batch see load estimates that already include earlier dispatches — preventing a burst of requests from all being sent to an already-saturated path.

After routing, an idle PrefillWorker is located via `_pick_idle_worker()`:

- **All PrefillWorkers busy**: calls `_requeue()` to push the Disagg batch back into `_waiting`; retried on the next step.
- **Insufficient free blocks**: also requeued, waiting for Decode to finish and release blocks.
- **Idle worker found**: starts a background thread running `prefill_batch_and_extract(groups)`; results are written to `_prefill_done`.

### `_flush_prefill_done()`

Atomically drains `_prefill_done` under `_prefill_lock`. For each completed prefill:
1. Appends the first token to `state.output_token_ids`.
2. Calls `dw.receive_kv_async(group, block_table, kv_buf, src_k, src_v)` to initiate asynchronous KV transfer.

The request then sits in `DecodeWorker._pending` until its KV transfer completes, at which point it is promoted to `running` on the next `step()`.

### `_step_collocated()` and `_step_disaggregated()`

- `_step_collocated`: calls `cw.step()`, then scans `cw.engine.scheduler.finished` to mark completed requests and write their output back into `_states`.
- `_step_disaggregated`: calls `dw.step()`, collects the `(group, tok_id)` pairs, appends tokens to the corresponding state, and handles EOS. Also includes a basic OOM recovery: on `MemoryError`, the oldest sequence in `dw.running` is evicted and its blocks freed, allowing subsequent requests to proceed.

### `run_until_done(max_new_tokens=200) -> Dict[str, str]`

Loops over `step()` until all requests are complete, checking `_enforce_max_tokens()` each step to cap output length. On completion, calls `router.update(prompt_len, actual_output_len)` for every request, feeding true output lengths back to `OutputLengthPredictor` and **closing the online learning loop**.

Returns a `{rid: decoded_text}` dict.

### `stats() -> dict`

Returns a live snapshot:
- Historical routing statistics (`router.decision_stats()`)
- Per-bucket predictor status (`predictor.stats()`)
- Current pending / running counts in the Decode Worker
- Number of PrefillWorkers and currently busy prefill threads

---

## Overall Data Flow

```
add_request(prompt)
    → _waiting

step()
├── _dispatch_waiting()
│   ├── read live system_load / decode_batch_size
│   ├── router.route(prompt_len, system_load, decode_batch_size)
│   │     └── predictor.predict()  +  cost_model.route()
│   ├── [collocated]     → cw.engine.add_request()
│   └── [disaggregated]  → Thread(_prefill_task)
│                              └── pw.prefill_batch_and_extract()
│                                  → _prefill_done
│
├── _flush_prefill_done()
│   └── dw.receive_kv_async()  → dw._pending
│
├── Thread(_step_collocated)    ─┐
│   └── cw.step()               ├─ concurrent
└── Thread(_step_disaggregated) ─┘
    └── dw.step()
          ├── _promote_ready()  (_pending → running)
          └── decode forward pass

run_until_done()
    └── loop step() until all done
        → router.update(actual_output_len)  ← online feedback
```

---

## Design Highlights

**1. Per-step live load sensing**

`_dispatch_waiting()` never relies on static configuration. Every `step()` re-reads the current lengths of `running` and `pending` queues, so routing decisions shift in real time with the system state. Under high load, the interference penalty pushes more requests toward Disagg; under low load, the transfer overhead may make Collocated consistently cheaper.

**2. Hard overload guard**

When `decode_batch_size >= MAX_DECODE_BATCH`, the router is bypassed and all requests are forced to Collocated, regardless of what the cost model would decide. This prevents block exhaustion and latency spikes in the Decode Worker.

**3. PrefillWorker pool**

Multiple `PrefillWorker` instances can be registered (one per GPU). `_pick_idle_worker()` assigns each disaggregated batch to a free worker and runs it in a background thread, allowing multiple Disagg requests to be prefilled on different GPUs concurrently without blocking one another.

**4. Prefill–Decode pipeline overlap**

Prefill runs in a background thread; KV transfer runs asynchronously on `transfer_stream`; decode advances independently on `compute_stream`. All three stages proceed concurrently on the same timeline, hiding new-request latency behind ongoing decode computation.

**5. Online learning loop**

After `run_until_done()`, actual output lengths are fed back to `OutputLengthPredictor`. Future routing decisions are made on increasingly accurate output-length predictions, making the system self-adapting to the workload distribution.

---

## Known Limitations

**1. Collocated step can stall the main loop**

`_step_collocated` calls `cw.step()`, which contains a GPU-synchronised forward pass. If the Collocated batch is large, the overall `step()` latency is dominated by it (since both threads are joined). The timing instrumentation in the code already calls this out: `coll占step: X% ← 这个高说明 GPU0 在阻塞主循环`.

**2. No Chunked Prefill on the Prefill side**

As noted in [05-workers_en.md](05-workers_en.md), `PrefillWorker` performs a full-sequence prefill in one shot. Very long prompts cause the prefill thread to be occupied for extended periods, delaying new requests from entering the Decode queue.

**3. `system_load` estimate is optimistic under high concurrency**

`_dispatch_waiting()` processes all waiting requests in a single pass, incrementing `system_load` for each dispatched request before any prefill thread has actually started or any KV transfer has occurred. This is an optimistic estimate; under extreme concurrency it can cause routing decisions to underestimate true load.

**4. `_requeue` incurs redundant tokenisation**

When a request is requeued because all PrefillWorkers are busy or there are insufficient blocks, `_requeue()` decodes the token IDs back to a string, only for `_dispatch_waiting()` to tokenise it again on the next step. Preserving the token ID list directly would be cleaner.
