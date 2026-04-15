<p align="center">
  <img width="260" src="assets/logo.png">
</p>

<h1 align="center">nanoPD</h1>

<p align="center">
  A from-scratch <strong>Prefill/Decode disaggregation inference engine</strong> for LLMs
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-76b900?logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Hardware-H20-blueviolet">
  <img src="https://img.shields.io/badge/Model-Qwen3--8B-orange">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

---

Disaggregated inference separates the two phases of LLM generation — the compute-intensive **prefill** (processing the prompt) and the memory-bandwidth-bound **decode** (generating tokens one at a time) — onto dedicated GPUs. This avoids the mutual interference between the two phases that limits throughput on collocated deployments.

nanoPD implements the full stack: a custom paged KV cache, chunked prefill, a custom CUDA paged attention kernel, multi-GPU KV transfer, an adaptive router driven by an analytical cost model, and a Poisson-arrival benchmark suite. All three serving strategies — **Collocated**, **Disaggregated**, and **Adaptive** — are implemented and benchmarked.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CentralScheduler                       │
│   (dispatch, KV transfer coordination, path accounting)     │
└────────────┬─────────────────┬───────────────┬─────────────┘
             │                 │               │
     ┌───────▼──────┐  ┌───────▼──────┐  ┌────▼──────────┐
     │  Collocated  │  │   Prefill    │  │    Decode     │
     │   Worker     │  │   Worker     │  │    Worker     │
     │  (GPU 0)     │  │ (GPU 1 / 3)  │  │   (GPU 2)    │
     └──────────────┘  └──────┬───────┘  └────▲──────────┘
                              │    KV Transfer │
                              └────────────────┘
             ↑
      ┌──────┴──────┐
      │   Router    │  ← analytical cost model decides path per request
      └─────────────┘
```

Each incoming request is routed by an **analytical cost model** that estimates the end-to-end latency of both strategies on the current hardware and picks the cheaper one. All parameters (prefill speed, decode speed, interference coefficient, inter-GPU bandwidth) are measured live on the actual device at startup.

---

## Key Features

- **Paged KV Cache** — block-granular memory management with copy-on-write for beam search / speculative decoding forks
- **Chunked Prefill** — long prompts are split into configurable chunks and interleaved with decode steps, keeping GPU utilisation high
- **Custom CUDA Paged Attention Kernel** — hand-written CUDA kernel for gather-scatter attention over non-contiguous KV blocks
- **Async KV Transfer** — prefill→decode KV cache migration over a dedicated CUDA stream via pinned memory relay or P2P direct (NVLink), with overlap against the compute stream
- **Adaptive Router** — per-request routing decision from a hardware-fitted analytical cost model; no oracle, no offline training
- **Output Length Predictor** — online Bayesian predictor for output length, used by the router to estimate decode cost before generation starts
- **Multi-Worker CentralScheduler** — concurrent Collocated and Disaggregated pipelines on separate threads, with dynamic batch management
- **Poisson Arrival Benchmark** — realistic open-loop load test with configurable arrival rate, workload distribution, warmup, and drain phases

---

## Modules

| Module | Description |
|---|---|
| `block_manager/` | Sequence + SequenceGroup data structures, `BlockSpaceManager` (paged KV allocation, CoW fork) |
| `engine/` | `ModelRunner` (custom `paged_forward` hook), `Engine` (scheduler loop, chunked prefill), `Scheduler` |
| `paged_attention/` | CUDA C++ extension: paged KV store ops + paged multi-head attention kernel |
| `workers/` | `CollocatedWorker`, `PrefillWorker`, `DecodeWorker`, `kv_transfer` (pinned relay + P2P) |
| `router/` | `Router` (wraps cost model), `OutputLengthPredictor` (online Bayesian), `CentralScheduler` |
| `cost_model/` | `profiler.py` (device micro-benchmarks), `analytical.py` (curve fitting + latency formulas) |
| `benchmark/` | Static batch benchmark, Poisson arrival benchmark, automated sweep, plotting |
| `examples/` | `demo_collocated.py` (single GPU), `demo_multiGPU.py` (full pipeline on 8× GPU) |
| `docs/` | Per-module deep-dive documentation in English and Chinese |

---

## Installation

**1. Clone the repo**

```bash
git clone https://github.com/your-username/nanoPD.git
cd nanoPD
```

**2. Build the CUDA extension** (compiles for the GPU on the current machine)

```bash
cd nanoPD/paged_attention
pip install -e . --no-build-isolation
cd ../..
```

> Requires: Python ≥ 3.10, PyTorch ≥ 2.1, CUDA ≥ 11.8, and NVCC in `PATH`.  
> The extension uses `-arch=native` and auto-detects the installed GPU's compute capability.

**3. Install Python dependencies**

```bash
pip install transformers scipy numpy matplotlib
```

---

## Quick Start

### Single-GPU collocated inference

Runs 5 prompts through `Engine.generate()` on a single GPU. Suitable for RTX 4060/4070/4080 with Qwen2-1.5B.

```bash
python examples/demo_collocated.py
# or specify a local path:
python examples/demo_collocated.py --model /path/to/Qwen2-1.5B --gpu 0 --max-new-tokens 300
```

```
Loading Qwen/Qwen2-1.5B on cuda:0 ...
Model loaded in 7.1s

[1/5] Prompt: What is the capital of France?
  Response (7 tokens, 1.04s, 6.7 tok/s):
    The capital of France is Paris.
...
```

### Multi-GPU adaptive inference (full pipeline)

Runs the three-step demo on 8× GPUs — profile → fit cost model → 60 s Poisson adaptive inference. Results are written to `output/output.txt`.

```bash
python examples/demo_multiGPU.py --model /path/to/Qwen3-8B

# Skip re-profiling if output/data/profile_data.pt already exists
python examples/demo_multiGPU.py --model /path/to/Qwen3-8B --skip-profile

# Tune load
python examples/demo_multiGPU.py --model /path/to/Qwen3-8B \
    --skip-profile --arrival-rate 0.3 --workload long
```

Default GPU assignment:

| Role | GPU | Flag |
|---|---|---|
| Collocated worker | 0 | `--collocated-gpu` |
| Prefill workers | 1, 3 | `--prefill-gpus 1 3` |
| Decode worker | 2 | `--decode-gpu` |

Output files:

| File | Content |
|---|---|
| `output/data/profile_data.pt` | Raw micro-benchmark measurements |
| `output/data/params.json` | Fitted cost model parameters |
| `output/data/results.json` | Full per-request benchmark results |
| `output/output.txt` | Human-readable summary |

---

## Cost Model & Routing

The router estimates end-to-end latency for both strategies using four hardware-measured parameters:

| Parameter | Meaning | RTX 4090 × 8 | H20 |
|---|---|---|---|
| α | Prefill latency (ms/token) | 0.1247 | 0.1452 |
| β | Decode step latency at batch=1 (ms) | 51.56 | 33.10 |
| batch_thresh | Memory→compute crossover batch size | 16 | 16 |
| γ | Prefill interference on decode (ms/token) | 0.0869 | 0.1302 |
| bandwidth | Inter-GPU transfer bandwidth (GB/s) | 12.9 (pinned relay) | 392 (P2P) |

**Key insight** — The routing decision reduces to comparing two costs that are both linear in prompt length:

```
Extra cost of disaggregated : transfer_rate × L          (pay to move KV cache)
Extra cost of collocated     : γ × L × (load/batch_thresh)  (pay for prefill interference)

Disaggregated wins when:  γ / transfer_rate > batch_thresh / system_load
```

On RTX 4090: `γ / transfer_rate ≈ 7.6` → Disaggregated wins from **system_load ≥ 3**  
On H20: `γ / transfer_rate ≈ 346` → Disaggregated wins at **virtually any non-zero load**

The full formula and per-hardware analysis is in [`docs/en/04-cost_model_en.md`](nanoPD/docs/en/04-cost_model_en.md).

---

## Benchmark Results

Tested on **Qwen3-8B** with two hardware configurations.

### Static Serial Benchmark

| Workload | Strategy | 4090 p50 | 4090 p99 | H20 p50 | H20 p99 |
|---|---|---|---|---|---|
| short | Collocated | 6.4 s | 6.4 s | 4.9 s | 7.2 s |
| short | Disaggregated | 9.2 s | 9.2 s | 4.9 s | 3.4 s |
| long | Collocated | 7.2 s | 7.3 s | 6.1 s | 10.2 s |
| long | Disaggregated | 7.3 s | ~7 s | 8.4 s | 10.4 s |

On H20, Disaggregated matches Collocated on short prompts (both 4.9 s) because 392 GB/s P2P bandwidth makes KV transfer nearly free. On the 4090, the 12.9 GB/s pinned-relay bandwidth adds a visible overhead — exactly as the cost model predicts.

### Poisson Arrival Benchmark (60 s window, mixed workload)

**RTX 4090 × 8:**

![Throughput 4090](nanoPD/benchmark/figures/fig_throughput.png)

**H20:**

![Throughput H20](nanoPD/benchmark/figures_h20/fig_throughput.png)

- **Adaptive** saturates at ~240 tok/s on the 4090 and ~175 tok/s on H20 at moderate arrival rates
- **Collocated** is competitive at low load but p99 tail latency degrades quickly as concurrency grows
- **Disaggregated** (serial implementation) plateaus at ~25–30 tok/s regardless of device — the bottleneck is the lack of concurrent decode batching in the serial benchmark path

More plots and analysis: [`docs/en/07-benchmark_en.md`](nanoPD/docs/en/07-benchmark_en.md)

---

## Project Structure

```
nanoPD/                            ← repo root
├── .gitignore
├── README.md
├── disaggregated_inference_engine.md   ← high-level design notes
├── examples/
│   ├── demo_collocated.py         ← single-GPU demo (Qwen2-1.5B)
│   └── demo_multiGPU.py           ← 8× GPU full pipeline demo
└── nanoPD/                        ← source package
    ├── block_manager/
    │   ├── block_manager.py       ← BlockSpaceManager, PhysicalBlock
    │   └── sequence.py            ← Sequence, SequenceGroup, SequenceStatus
    ├── engine/
    │   ├── engine.py              ← Engine (scheduler loop + chunked prefill)
    │   ├── model_runner.py        ← ModelRunner with paged_forward hook
    │   └── scheduler.py           ← Scheduler (prefill/decode batching)
    ├── paged_attention/
    │   └── csrc/                  ← CUDA kernels (paged attention + KV store)
    ├── workers/
    │   ├── collocated_worker.py
    │   ├── prefill_worker.py
    │   ├── decode_worker.py
    │   └── kv_transfer.py         ← async KV migration (pinned relay + P2P)
    ├── router/
    │   ├── central_scheduler.py   ← CentralScheduler (multi-worker dispatch)
    │   ├── router.py              ← Router (wraps cost model + predictor)
    │   └── output_lenth_predictor.py
    ├── cost_model/
    │   ├── profiler.py            ← device micro-benchmarks
    │   ├── analytical.py          ← curve fitting + routing decision
    │   ├── params.json            ← RTX 4090 fitted parameters
    │   └── params_h20.json        ← H20 fitted parameters
    ├── benchmark/
    │   ├── benchmark.py           ← static serial benchmark
    │   ├── benchmark_poisson.py   ← Poisson arrival benchmark
    │   ├── sweep.py               ← automated sweep across arrival rates
    │   └── plot_benchmark.py      ← result visualisation
    └── docs/
        ├── en/                    ← English documentation (7 modules)
        └── zh/                    ← Chinese documentation (7 modules)
```

---

## Documentation

Each module has a dedicated deep-dive doc covering design rationale, data structures, algorithms, and worked examples.

| # | English | Chinese |
|---|---|---|
| 1 | [Block Manager](nanoPD/docs/en/01-block_manager_en.md) | [块管理器](nanoPD/docs/zh/01-block_manager_cn.md) |
| 2 | [Engine](nanoPD/docs/en/02-engine_en.md) | [推理引擎](nanoPD/docs/zh/02-engine_cn.md) |
| 3 | [CUDA Kernels](nanoPD/docs/en/03-cuda_kernels_en.md) | [CUDA 内核](nanoPD/docs/zh/03-cuda_kernels_cn.md) |
| 4 | [Cost Model](nanoPD/docs/en/04-cost_model_en.md) | [代价模型](nanoPD/docs/zh/04-cost_model_cn.md) |
| 5 | [Workers](nanoPD/docs/en/05-workers_en.md) | [Worker 层](nanoPD/docs/zh/05-workers_cn.md) |
| 6 | [Router](nanoPD/docs/en/06-router_en.md) | [路由器](nanoPD/docs/zh/06-router_cn.md) |
| 7 | [Benchmark](nanoPD/docs/en/07-benchmark_en.md) | [基准测试](nanoPD/docs/zh/07-benchmark_cn.md) |
