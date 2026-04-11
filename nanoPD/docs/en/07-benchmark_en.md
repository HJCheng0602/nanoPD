# Benchmark Module Documentation

> Source code: `benchmark/`

---

## Module Overview

`benchmark/` contains two sets of experiments:

| Script | Type | Purpose |
|---|---|---|
| `benchmark.py` | Static batch benchmark | Serial/batch comparison of latency and throughput across three strategies |
| `benchmark_poisson.py` | Poisson arrival benchmark | Simulates realistic concurrent request streams; measures saturation behaviour |
| `sweep.py` | Automated sweep + plotting | Drives `benchmark_poisson.py` across multiple arrival rates and aggregates figures |
| `plot_benchmark.py` | Visualisation | Converts `benchmark.py` JSON results into charts |

---

## benchmark.py: Static Batch Test

### Workloads

```python
short  : prompt_len ∈ [64, 256],   output ≈ 128 tokens
long   : prompt_len ∈ [1024, 2048], output ≈ 128 tokens
mixed  : 50% short + 50% long
```

`_rand_prompt` samples from the first 5000 vocabulary tokens to generate prompts, ensuring identical inputs across all strategies (fixed seed).

### Three Execution Modes

**`run_collocated`**: Processes requests one at a time — `add_request` → loop `step()` until done → next request. Each request must reach EOS or the target output length before the next is submitted. **The GPU is idle between requests.**

**`run_disaggregated`**: Likewise serial; each request completes the full `prefill_and_extract` → `receive_kv_async` → `step()` loop before the next begins.

**`run_adaptive`**: Submits all requests at once via `scheduler.add_request()`, then calls `run_until_done()` to let `CentralScheduler` schedule them concurrently. **Unlike the other two, Adaptive is fully concurrent.**

### Methodology Note

The three strategies are driven fundamentally differently — Collocated and Disaggregated are serial; Adaptive is fully batched — so **the throughput numbers in Fig 2 are not directly comparable**. Adaptive's throughput advantage stems primarily from concurrent execution across multiple GPUs, not from the routing strategy itself. For a fair throughput comparison, refer to the Poisson benchmark results below.

### `BenchResult` Data Structure

```python
@dataclass
class BenchResult:
    strategy: str
    workload: str
    n_requests: int
    total_ms: float
    throughput_tokens_per_s: float
    p50_e2e_ms: float
    p99_e2e_ms: float
    requests: List[RequestResult]    # per-request detail
```

`RequestResult` contains `prompt_len`, `output_len`, `ttft_ms` (time to first token), `e2e_ms`, and `path`.

**Note**: The Adaptive path currently has no TTFT instrumentation; `ttft_ms` is always recorded as 0.0.

---

## benchmark_poisson.py: Poisson Arrival Test

### Motivation

Requests in a real inference service arrive at **random intervals**, not as a serial queue. The Poisson arrival process (inter-arrival time ∼ Exp(1/λ)) is a classical load model for serving systems and can measure throughput saturation, latency distributions, and request-drop behaviour at different load intensities.

### Procedure

1. **Warmup phase**: run for `warmup_s` seconds (default 10 s) to bring the GPU to steady state and warm up CUDA kernels and memory.
2. **Benchmark phase**: run for `duration` seconds (default 60 s); requests arrive according to the Poisson process and are submitted in real time.
3. **Drain phase**: after `duration` expires, wait up to `drain_timeout` seconds to harvest remaining in-flight requests; anything still pending after the timeout is counted as `n_dropped`.

### Three Execution Modes

- **`run_poisson_collocated`**: maintains a `pending` dict; arriving requests are submitted via `cw.engine.add_request()`, `cw.step()` is called each iteration, and completed requests are harvested from the `finished` queue. Uses `CollocatedWorker`'s Continuous Batching so newly arrived requests are scheduled alongside ongoing ones.

- **`run_poisson_disaggregated`**: maintains a `req_queue` list. **Only one request is processed at a time** — the next request is dequeued only when both `dw.running` and `dw._pending` are empty, running the full prefill → transfer → decode pipeline. This makes the Disagg path purely serial under Poisson load, causing rapid queue build-up at higher arrival rates.

- **`run_poisson_adaptive`**: maintains an `arrive_times` dict; requests are submitted via `scheduler.add_request()` according to Poisson timing, `scheduler.step()` is called each iteration, and completed requests are harvested from `_finish_time`.

### `PoissonBenchResult` Data Structure

```python
@dataclass
class PoissonBenchResult:
    strategy: str
    workload: str
    arrival_rate: float
    duration_s: float
    n_completed: int
    n_dropped: int
    throughput_rps: float
    throughput_tokens_per_s: float
    p50_e2e_ms: float
    p95_e2e_ms: float
    p99_e2e_ms: float
    p50_queue_ms: float
    p99_queue_ms: float
    requests: List[ReqResult]
```

`ReqResult` additionally records `arrive_time`, `finish_time`, and `queue_wait_ms` to support queue-buildup analysis.

---

## sweep.py: Automated Sweep

`sweep.py` drives `benchmark_poisson.py` sequentially across multiple arrival rates (default 0.1 → 3.0 rps, 10 points), appending results to JSON, then calls `plot_all()` to generate all figures in one shot.

Already-completed `(strategy, rate)` combinations are skipped automatically (by checking whether the JSON key exists), supporting resumable runs.

Supports multiple Prefill GPUs (`PREFILL_GPUS = [1, 3]`); the Adaptive path uses two PrefillWorkers concurrently.

---

## Results

Experiments were run on two devices: **RTX 4090** (`figures/`, `results.json`) and **H20** (`figures_h20/`, `results_h20.json`), each using the corresponding cost model parameters (`params.json` / `params_h20.json`). Results for both devices are presented side by side below.

---

### Static Serial Benchmark (benchmark.py)

#### Fig 1a — End-to-End Latency (RTX 4090)

![Fig 1 Latency Bar 4090](../../benchmark/figures/fig1_latency_bar.png)

#### Fig 1b — End-to-End Latency (H20)

![Fig 1 Latency Bar H20](../../benchmark/figures_h20/fig1_latency_bar.png)

| Workload | Strategy | 4090 p50 (s) | 4090 p99 (s) | H20 p50 (s) | H20 p99 (s) |
|---|---|---|---|---|---|
| short | Collocated | 6.4 | 6.4 | 4.9 | 7.2 |
| short | Disaggregated | 9.2 | 9.2 | **4.9** | 3.4 |
| short | Adaptive | 8.9 | 9.2 | 11.4 | 11.4 |
| long | Collocated | 7.2 | 7.3 | 6.1 | 10.2 |
| long | Disaggregated | 7.3 | ~7 | 8.4 | 10.4 |
| long | Adaptive | 11.8 | 22.2 | 43.1 | 43.7 |
| mixed | Collocated | 6.7 | 7.3 | 6.7 | 9.8 |
| mixed | Disaggregated | 6.9 | 6.9 | 6.6 | 9.0 |
| mixed | Adaptive | 8.1 | 11.7 | 20.2 | 20.1 |

**Key observations**:

- **On H20, Disagg short-workload latency matches Collocated (both 4.9 s)**, whereas on the 4090 Disagg is ~44% slower (9.2 s vs 6.4 s). H20's P2P bandwidth (~392 GB/s) makes KV Transfer overhead nearly negligible, while the 4090's cross-card bandwidth (~12.9 GB/s) makes it a visible bottleneck — exactly as the cost model predicts (γ/bw ratio: H20 346×, 4090 7.6×).
- **Adaptive on H20 reaches 43 s for the long workload**, far above the 4090's 11.8 s. This is not because H20 is slower, but because this serial test causes `CentralScheduler` to queue multiple concurrent requests, and tail requests wait for a Prefill thread to become free. This is a concrete manifestation of the **fundamental unfairness of a serial benchmark against a batch-concurrent design**.

#### Fig 2a — Throughput (RTX 4090)

![Fig 2 Throughput 4090](../../benchmark/figures/fig2_throughput.png)

#### Fig 2b — Throughput (H20)

![Fig 2 Throughput H20](../../benchmark/figures_h20/fig2_throughput.png)

| Workload | 4090 Coll | 4090 Disagg | 4090 Adaptive | H20 Coll | H20 Disagg | H20 Adaptive |
|---|---|---|---|---|---|---|
| short | 20.1 | 20.2 | **134.4** | 26.1 | 25.9 | **112.7** |
| long | 17.0 | 16.7 | **57.7** | 15.3 | 14.9 | **29.6** |
| mixed | 18.4 | 18.3 | **104.9** | 19.4 | 19.1 | **63.4** |

H20 Collocated is slightly faster than 4090 on the short workload (26.1 vs 20.1 tok/s), consistent with H20's shorter decode step (β=33 ms vs 51 ms). Adaptive's throughput advantage follows the same mechanism as above (concurrent vs serial) and is not a fair comparison.

#### Fig 3a — Disaggregated TTFT vs Prompt Length (RTX 4090)

![Fig 4 TTFT 4090](../../benchmark/figures/fig4_ttft.png)

**Fit**: TTFT ≈ 0.1375 × L + 43.5 ms — data points closely follow the line, R² ≈ 1.

#### Fig 3b — Disaggregated TTFT vs Prompt Length (H20)

![Fig 4 TTFT H20](../../benchmark/figures_h20/fig4_ttft.png)

**Fit**: TTFT ≈ 0.1103 × L + 590.8 ms — noticeably higher scatter than the 4090.

| Device | Slope α (ms/token) | Intercept (ms) | Data quality |
|---|---|---|---|
| RTX 4090 | 0.1375 | 43.5 | Excellent, tight linear |
| H20 | 0.1103 | 590.8 | High scatter |

The H20 intercept is anomalously high (590 ms vs 43 ms) and scatter is much larger, indicating greater prefill latency variability on H20. Likely causes: heavier OS/scheduler intervention on a server GPU, or insufficient warmup. The slope 0.1103 ms/token deviates somewhat from the cost model's α=0.1452 ms/token, possibly due to actual batch-size effects during profiling.

---

### Poisson Arrival Benchmark (sweep.py)

#### Fig 4 — P50 End-to-End Latency vs Arrival Rate

| | 4090 | H20 |
|---|---|---|
| | ![P50 4090](../../benchmark/figures/fig_p50_e2e.png) | ![P50 H20](../../benchmark/figures_h20/fig_p50_e2e.png) |

On both devices, Disaggregated diverges past 40 s after rate > 0.5 rps for the same reason: `run_poisson_disaggregated` is purely serial and queue build-up is inevitable.

**H20 vs 4090**: On H20, Adaptive (green) starts rising noticeably after rate=1.5 rps, whereas on the 4090 it stays lower for longer. This may be because the H20 run used two Prefill Workers (GPU-1 and GPU-3), increasing scheduling contention at high load; the higher prefill latency variability seen in the TTFT plot may also contribute.

#### Fig 5 — P99 End-to-End Latency vs Arrival Rate

| | 4090 | H20 |
|---|---|---|
| | ![P99 4090](../../benchmark/figures/fig_p99_e2e.png) | ![P99 H20](../../benchmark/figures_h20/fig_p99_e2e.png) |

On H20, Collocated p99 overtakes Adaptive at high rates (rate > 2 rps), a behaviour not seen on the 4090. This suggests that under extreme concurrency on H20, the combination of KV Transfer overhead (even at high bandwidth) and CentralScheduler scheduling delay pushes tail latency above what a single-GPU Collocated path incurs.

#### Fig 6 — Throughput vs Arrival Rate

| | 4090 | H20 |
|---|---|---|
| | ![Throughput 4090](../../benchmark/figures/fig_throughput.png) | ![Throughput H20](../../benchmark/figures_h20/fig_throughput.png) |

Adaptive peaks at ~175 tok/s on H20 (rate ≈ 1.75 rps) vs ~240 tok/s on the 4090 (rate ≈ 2 rps). The 4090 peaks higher because its slower decode step (β=51 ms) gives the Disagg path more work to do, distributing load more evenly across GPUs. On H20 (β=33 ms), more requests complete quickly on the single Collocated GPU, reducing utilisation of the Disagg path.

Disagg plateaus at ~25–30 tok/s on both devices; the bottleneck is identical (serial processing).

#### Fig 7 — Completed and Dropped Requests vs Arrival Rate

| | 4090 | H20 |
|---|---|---|
| | ![Completed/Dropped 4090](../../benchmark/figures/fig_completed_dropped.png) | ![Completed/Dropped H20](../../benchmark/figures_h20/fig_completed_dropped.png) |

On H20, Adaptive completes more requests at peak rate (~80 at rate=2 rps vs ~70 on the 4090), reflecting H20's faster decode. Disagg on both devices completes only ~10 requests regardless of arrival rate, entirely throttled by its serial bottleneck.

---

## Known Limitations

**1. benchmark.py methodology is not a fair comparison**

Collocated/Disaggregated run serially while Adaptive runs fully concurrently, making Fig 1 (latency) unfair to Adaptive and Fig 2 (throughput) overly favourable to it. Neither metric is suitable for a direct cross-strategy comparison. A fair comparison requires all three strategies to be driven with equivalent concurrency.

**2. Serial bottleneck of Disaggregated in benchmark_poisson.py**

`run_poisson_disaggregated` dequeues only one request at a time, regardless of how many are waiting. This reflects the lack of concurrent batching in `DecodeWorker` (see [05-workers_en.md](05-workers_en.md)), but it also makes the Disagg gap in the Poisson benchmark more extreme than a well-implemented disaggregated system would show.

**3. No TTFT instrumentation for Adaptive**

`run_adaptive` hardcodes `ttft_ms = 0.0`; time-to-first-token distribution cannot be analysed for the Adaptive path.

**4. Inconsistent warmup across strategies**

Collocated and Disaggregated warmup by running full inference, while Adaptive warmup manually constructs `_current_context` and calls the forward pass directly. The quality of GPU warm-up may differ between the two approaches.
