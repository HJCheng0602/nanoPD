# Workers Module Documentation

> Source code: `workers/`

---

## Module Overview

`workers/` provides the top-level execution units for two deployment paths:

| Class | File | Path |
|---|---|---|
| `CollocatedWorker` | `collocated_worker.py` | Collocated (P/D mixed, single GPU) |
| `PrefillWorker` | `prefill_worker.py` | Disaggregated — Prefill side |
| `DecodeWorker` | `decode_worker.py` | Disaggregated — Decode side |
| `PinnedKVBuffer` / transfer functions | `kv_transfer.py` | KV Cache cross-device migration infrastructure |

Both paths share the same `BlockSpaceManager` instance and `ModelRunner` design; the difference lies only in the scheduling logic and the KV transfer layer.

---

## kv_transfer.py

Low-level infrastructure for cross-device KV Cache transfer, shared by `PrefillWorker` and `DecodeWorker`.

### `PinnedKVBuffer`

```python
class PinnedKVBuffer:
    def __init__(self, num_layers, num_block, num_kv_heads, block_size, head_dim, dtype=torch.float16)
    @staticmethod
    def from_runner(runner, num_blocks) -> "PinnedKVBuffer"
```

A pair of pinned-memory tensors (`self.k` / `self.v`) with shape `(num_layers, num_block, num_kv_heads, block_size, head_dim)`.

**Pinned memory (page-locked memory)** is a prerequisite for CUDA asynchronous H2D/D2H transfers: ordinary CPU memory can be paged out by the OS during a transfer, causing interruption or requiring an extra temporary copy. Pinned memory locks the physical pages so DMA can operate on them directly, enabling `non_blocking=True` asynchronous copies.

`from_runner` is a convenience factory that reads `num_layers`, `num_kv_heads`, `block_size`, and `head_dim` from a `ModelRunner` instance, avoiding manual parameter passing.

### `extract_kv_to_pinned`

```python
def extract_kv_to_pinned(k_cache, v_cache, block_table, buf: PinnedKVBuffer)
```

Copies blocks from the Prefill GPU's `k_cache / v_cache` (shape `[num_layers, max_blocks, ...]`) to `buf`'s pinned memory, using the physical block indices in `block_table`:

```python
for i, bid in enumerate(block_table):
    buf.k[:, i].copy_(k_cache[:, bid])   # all layers, logical slot i ← physical block bid
    buf.v[:, i].copy_(v_cache[:, bid])
```

This is a **synchronous** operation (no `non_blocking`). A `torch.cuda.synchronize()` is required afterwards to guarantee the GPU→CPU copy has completed.

### `load_kv_from_pinned`

```python
def load_kv_from_pinned(k_cache, v_cache, block_table, buf, stream=None)
```

The inverse of `extract_kv_to_pinned`: writes KV from pinned memory back into the Decode GPU's `k_cache / v_cache` according to `block_table`. When a `stream` is provided, copies are submitted with `non_blocking=True` on that stream, enabling concurrency with the compute stream.

### `transfer_kv`

```python
def transfer_kv(src_k, src_v, dst_k, dst_v, block_table, stream=None, buf=None) -> str
```

The unified transfer entry point, which automatically selects the best path:

```
if P2P available (both CUDA devices, peer access enabled):
    direct GPU→GPU copy (src_k[:, bid] → dst_k[:, bid])
    return "p2p"
else:
    relay through pinned memory (load_kv_from_pinned)
    return "pinned_relay"
```

The P2P path eliminates the CPU relay and has lower latency. However, consumer GPUs (RTX series on PCIe) typically cannot enable P2P, so `pinned_relay` is the common case in practice.

### `_check_p2p`

```python
def _check_p2p(src_device, dst_device) -> bool
```

Probes P2P support via `torch.cuda.can_device_access_peer()`, returning `False` silently on failure.

---

## CollocatedWorker

```python
class CollocatedWorker:
    def __init__(self, model_path, gpu_id=0, block_size=16, max_blocks=512)
    def add_request(self, prompt) -> str
    def step(self) -> List[Tuple[SequenceGroup, int]]
    def run_until_done(self, max_tokens_per_seq=500) -> dict
    def run_until_done_single(self, prompt, max_new_tokens=500) -> str
```

`CollocatedWorker` is an **extremely thin wrapper** around `Engine`; every interface delegates directly to `self.engine`:

```python
self.engine = Engine(model_path, block_size, max_blocks, device=f"cuda:{gpu_id}")
self.finished = self.engine.scheduler.finished
```

### Continuous Batching and Chunked Prefill

The core advantage of `CollocatedWorker` is that it fully inherits the scheduling capabilities of `Engine`:

- **Continuous Batching**: Each `step()` is driven by the `Scheduler`, which mixes new requests' Prefill tokens with existing requests' Decode tokens in a **single forward pass**. The GPU never idles waiting for new arrivals, and long prefills do not block ongoing decodes.
- **Chunked Prefill**: Long prompts are not computed in one shot; instead, each step processes up to `budget` tokens, allowing decode requests to interleave continuously and avoiding long-tail latency.

These two mechanisms jointly determine the throughput advantage of the Collocated path under high concurrency.

---

## PrefillWorker

```python
class PrefillWorker:
    def __init__(self, model_path, gpu_id, block_manager, block_size=16, max_blocks=512)
```

`PrefillWorker` holds a `ModelRunner` directly — **it does not use `Engine` or `Scheduler`**. Its sole responsibility is to run full-sequence prefill for a batch of requests and extract the resulting KV Cache for transfer.

### `prefill_batch`

```python
def prefill_batch(self, groups: List[SequenceGroup]) -> Tuple[List[int], List[List[int]]]
```

The core batch prefill method, in five steps:

1. **Allocate blocks per request**: calls `block_manager.allocate(group)` to assign physical blocks and build a `block_table` for each sequence.
2. **Concatenate input tensors**: flattens all token IDs and positions into `[1, total_tokens]`-shaped tensors (the same mixed-batch format as `Engine`).
3. **Set `_current_context`**: injects `num_prefill_tokens / num_decode_tokens / prefills / decodes` into the `ModelRunner`, so that monkey-patched attention layers can locate each sequence's `block_table` correctly.
4. **Forward pass**: calls `runner.model(input_ids, position_ids, use_cache=False)`; all layers write K/V into the KV Cache via the `paged_forward` closure.
5. **Extract first token**: for each sequence, samples from the logits at its last token position using `top_k_sample`; updates `seq.num_computed_tokens` and `seq.output_token_ids`.

**Note**: `PrefillWorker` has no Chunked Prefill logic — it runs a **full-sequence prefill in a single shot**. For very long prompts, this means one long forward pass during which the Decode Worker can only wait.

### `prefill_batch_and_extract`

```python
def prefill_batch_and_extract(self, groups) -> List[Tuple[int, List[int], PinnedKVBuffer, Tensor, Tensor]]
```

Calls `prefill_batch` and then extracts the KV Cache to pinned memory for each request, returning a 5-tuple per request:
`(first_token, block_table, pinned_buf, k_cache, v_cache)`

`k_cache / v_cache` are the full cache tensors on the Prefill GPU (used by `transfer_kv` for direct P2P reads when available). The method ends with `torch.cuda.synchronize()` to ensure all D2H copies are complete before returning.

**Known limitation**: `extract_kv_to_pinned` is called unconditionally regardless of whether P2P is available. On server GPUs with NVLink/NVSwitch where P2P is supported, the pinned buffer allocation and D2H copy are wasted work — the actual transfer will use the direct GPU→GPU path and never read the buffer. On those platforms the extra D2H copy (constrained by PCIe bandwidth, ~64 GB/s) adds latency comparable to the P2P transfer itself before the function even returns. The fix is to probe `_check_p2p` before extracting and skip the pinned copy entirely on the P2P path.

### `prefill` / `prefill_and_extract`

Single-request convenience wrappers around `prefill_batch([group])` and `prefill_batch_and_extract([group])` respectively.

### `extract_kv`

```python
def extract_kv(self, block_table: List[int]) -> PinnedKVBuffer
```

Standalone KV extraction for a sequence that has already been prefilled. Useful in scenarios where prefill and extraction need to be decoupled.

---

## DecodeWorker

```python
class DecodeWorker:
    def __init__(self, model_path, gpu_id, block_manager, block_size=16, max_blocks=512)
```

`DecodeWorker` likewise holds a `ModelRunner` directly, without `Engine`. It manages two queues:

- `self.running`: `SequenceGroup` list currently being decoded
- `self._pending`: `_PendingTransfer` list waiting for KV transfer to complete
- `self.finished`: completed sequences

Two CUDA streams:
- `compute_stream`: runs the decode forward pass
- `transfer_stream`: asynchronously receives incoming KV Cache

### `_PendingTransfer`

```python
class _PendingTransfer:
    group: SequenceGroup
    event: torch.cuda.Event
```

A lightweight dataclass binding a `SequenceGroup` to a `CUDA Event` that records the completion of its KV transfer.

### `receive_kv_async`

```python
def receive_kv_async(self, group, block_table, buf, src_k=None, src_v=None)
```

Submits the KV transfer asynchronously on `transfer_stream` (via `transfer_kv`), then records an `Event` on `transfer_stream` and enqueues a `_PendingTransfer(group, event)` into `self._pending`.

The caller returns **immediately without blocking**. The actual copy proceeds on the background stream.

### `_promote_ready`

```python
def _promote_ready(self)
```

Iterates over `_pending` and calls `event.query()` (a non-blocking GPU poll) on each entry:

- If the event is complete → move `group` to `self.running`
- Otherwise → leave it in `_pending`

This is called at the start of every `step()`, implementing **transfer–decode pipeline overlap**: while the Decode Worker is processing existing sequences, incoming KV transfers proceed concurrently on `transfer_stream`; when the next `step()` arrives, any newly completed transfers are automatically promoted into the decode batch.

### `step`

```python
def step(self) -> List[Tuple[SequenceGroup, int]]
```

One decode step, end-to-end:

1. **`_promote_ready()`**: promote transfer-complete requests into `running`.
2. **Build inputs**: for each sequence in `running`, call `block_manager.append_slot(seq)` to claim the next KV slot, take `seq.output_token_ids[-1]` as the current input token, record `position = seq.num_computed_tokens`.
3. **Set `_current_context`**: `num_decode_tokens = len(running)`, `prefills=[]`, inject each sequence's `block_table` and `position`.
4. **Forward pass** (on `compute_stream`): `runner.model(input_ids, position_ids)`, producing logits of shape `[1, B, vocab_size]`.
5. **Sampling**: for each sequence, `top_k_sample` the logits at its index; append the next token to `seq.output_token_ids`, increment `num_computed_tokens`.
6. **EOS detection**: if EOS is sampled, set status to `FINISHED_STOPPED`, remove from `running`, call `block_manager.free(seq)` to release physical blocks, move to `self.finished`. Immediately follow with another `_promote_ready()` so queued requests can fill the vacated slot.

---

## Overall Data Flow

### Collocated Path

```
add_request(prompt)
    → Scheduler.waiting
    → step(): Scheduler mixes Prefill + Decode tokens
    → ModelRunner.run_batch()
        → paged_kv_store kernel      writes KV Cache
        → paged_attention_forward kernel  reads KV Cache
    → sample → return token
```

### Disaggregated Path

```
PrefillWorker                        DecodeWorker
─────────────────────────────────    ─────────────────────────────────
prefill_and_extract(group)
  ├─ block_manager.allocate(group)
  ├─ ModelRunner.model(...)
  │    └─ paged_kv_store kernel
  ├─ extract_kv_to_pinned(...)       
  └─ return (first_token,            receive_kv_async(group, ...)
             block_table,               ├─ transfer_kv(..., transfer_stream)
             pinned_buf,                │    └─ P2P  or  Pinned Relay
             k_cache, v_cache)  ──►     └─ record Event → _pending

                                     step()
                                       ├─ _promote_ready()  [event.query()]
                                       ├─ append_slot(seq)
                                       ├─ ModelRunner.model(...)
                                       │    └─ paged_attention_forward kernel
                                       └─ top_k_sample → token
```

**Shared `BlockSpaceManager`**: both workers operate on the same `block_manager` instance. `PrefillWorker` calls `allocate`; `DecodeWorker` calls `append_slot` and `free`. Block indices are consistent across both sides, but the KV Cache tensors on each GPU are independent — the transfer exists precisely to synchronize block contents across devices.

---

## Throughput Limitations of the Disaggregated Path

The current disaggregated implementation has several structural limitations that keep its throughput ceiling well below the Collocated path:

**1. No Continuous Batching**

`DecodeWorker.step()` performs pure decode; `PrefillWorker` performs pure prefill — **the two are never mixed into a single forward pass**. The core advantage of the Collocated path — sharing computation between prefill tokens and decode tokens in the same forward pass — does not exist in the Disagg path.

Concrete impact: each `DecodeWorker` forward pass contains only `B` decode tokens (typically a small number), with very low arithmetic intensity. Matrix-multiply units are heavily underutilised compared to a mixed prefill+decode batch.

**2. No Chunked Prefill**

`PrefillWorker.prefill_batch` runs a **full-sequence prefill in a single shot**. A long prompt (e.g. 4096 tokens) occupies an entire step or more, during which the Decode Worker receives no new sequences and its queue stalls.

**3. KV Transfer Overhead**

Every completed prefill must transfer its KV Cache from the Prefill GPU to the Decode GPU (via P2P or pinned relay). At the RTX 4090's measured cross-card bandwidth of ~12.9 GB/s, transferring a 2048-token KV takes ~23 ms — time during which the Decode Worker cannot produce any tokens for that request.

The routing analysis in the cost model shows that even under interference load (load=8), the Disagg path's advantage over Collocated is modest for short sequences (L≈128, ~0.032 ms/token gain) and only grows meaningfully at longer sequence lengths. In other words, the practical benefit of the Disagg path is mainly that the Decode stage is isolated from Prefill interference — but raw throughput is structurally limited by the absence of Continuous Batching.

**4. Synchronisation Barrier in `prefill_batch_and_extract`**

`prefill_batch_and_extract` ends with `torch.cuda.synchronize()`, blocking the CPU until all D2H copies complete before returning. The caller cannot submit the next batch to `PrefillWorker` until this returns, creating a serial bottleneck. On server GPUs where P2P is available, this is compounded by the fact that the D2H copy is done unconditionally and then never used — see the note in the `prefill_batch_and_extract` section above.

**Summary**: to improve Disagg throughput, the primary directions are (a) introducing a scheduler to drive `DecodeWorker` with a properly batched decode queue, and (b) adding Chunked Prefill support to `PrefillWorker`. The current implementation is a complete and correct prototype for functional validation, but is not suited for production-grade disaggregated serving.

---

## Test: test_disaggregated.py

`workers/test/test_disaggregated.py` provides three integration tests that operate directly on the worker classes:

| Test | What it verifies |
|---|---|
| `test_single_request` | Full disaggregated pipeline for one request: Prefill → KV Extract → Transfer → N decode steps, with per-step timing |
| `test_overlap` | Overlap test: send req-1's KV transfer while req-0 is decoding; verifies `_promote_ready` picks up req-1 at the correct step |
| `test_collocated` | Collocated baseline: runs the same prompt through `CollocatedWorker` for output sanity comparison |

Usage:
```bash
python workers/test/test_disaggregated.py \
    --model Qwen/Qwen3-8B \
    --prefill-gpu 1 --decode-gpu 2
```
