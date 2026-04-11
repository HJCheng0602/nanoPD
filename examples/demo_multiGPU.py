"""
demo_multiGPU.py — Multi-GPU adaptive P/D disaggregation demo

Hardware target: 8× RTX 4090
  GPU 0      : collocated worker  (prefill + decode on same device)
  GPU 1, 3   : prefill workers    (disaggregated path)
  GPU 2      : decode worker      (disaggregated path)

Pipeline (fully automated):
  Step 1 — Profile all kernels and P2P bandwidth  →  output/data/profile_data.pt
  Step 2 — Fit analytical cost model              →  output/data/params.json
           Print routing threshold analysis
  Step 3 — Poisson adaptive inference (60 s)      →  output/output.txt
           Full per-request JSON saved to           output/data/results.json

Usage:
    # Full run (profile takes ~5 min)
    python examples/demo_multiGPU.py --model /path/to/Qwen3-8B

    # Skip profiling if output/data/profile_data.pt already exists
    python examples/demo_multiGPU.py --model /path/to/Qwen3-8B --skip-profile

    # Tune arrival rate / workload
    python examples/demo_multiGPU.py --model /path/to/Qwen3-8B \\
        --skip-profile --arrival-rate 0.3 --workload long
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../nanoPD'))

import argparse
import json
import time

from cost_model.profiler  import run_full_profile
from cost_model.analytical import AnalyticalCostModel
from benchmark.benchmark_poisson import run_poisson_adaptive, print_summary

_ROOT       = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR  = os.path.join(_ROOT, "output")
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.txt")


# ── Step 1: profile ───────────────────────────────────────────────────────────

def step1_profile(args) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    profile_path = os.path.join(DATA_DIR, "profile_data.pt")

    print("\n" + "=" * 60)
    print("STEP 1 / 3  —  Kernel profiling")
    print(f"  src GPU (prefill): {args.prefill_gpus[0]}  "
          f"dst GPU (decode): {args.decode_gpu}")
    print("=" * 60)

    run_full_profile(
        model_path=args.model,
        output_path=profile_path,
        src_gpu=args.prefill_gpus[0],
        dst_gpu=args.decode_gpu,
    )
    return profile_path


# ── Step 2: fit cost model + routing analysis ─────────────────────────────────

def step2_fit_and_analyse(profile_path: str) -> str:
    params_path = os.path.join(DATA_DIR, "params.json")

    print("\n" + "=" * 60)
    print("STEP 2 / 3  —  Cost model fitting + routing analysis")
    print("=" * 60)

    model = AnalyticalCostModel.fit_from_profile(
        profile_data_path=profile_path,
        save_params_path=params_path,
    )
    p = model.p

    print("\n[routing analysis]  estimated latency at system_load=4, output_len=128")
    print(f"  {'prompt_len':>10}  {'T_coll (ms)':>12}  {'T_disagg (ms)':>14}  {'decision':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*10}")
    for L in [64, 128, 256, 512, 1024, 2048]:
        tc = model.t_collocated(L, output_len=128, system_load=4)
        td = model.t_disaggregated(L, output_len=128)
        decision = "disagg" if td < tc else "coll"
        marker = " <" if td < tc else ""
        print(f"  {L:>10}  {tc:>12.1f}  {td:>14.1f}  {decision:>10}{marker}")

    print(f"\n[params summary]")
    print(f"  alpha (prefill ms/tok) : {p.alpha:.5f}")
    print(f"  beta  (decode ms/step) : {p.beta:.2f}")
    print(f"  batch_thresh           : {p.batch_thresh:.0f}")
    print(f"  gamma (interference)   : {p.gamma:.5f}")
    print(f"  P2P bandwidth          : {p.bandwidth_GBps:.2f} GB/s")

    return params_path


# ── Step 3: Poisson adaptive inference ────────────────────────────────────────

def step3_benchmark(args, params_path: str) -> dict:
    print("\n" + "=" * 60)
    print("STEP 3 / 3  —  Poisson adaptive inference  (60 s)")
    print(f"  workload={args.workload}  arrival_rate={args.arrival_rate} rps")
    print(f"  collocated=GPU{args.collocated_gpu}  "
          f"prefill=GPU{args.prefill_gpus}  decode=GPU{args.decode_gpu}")
    print("=" * 60)

    result = run_poisson_adaptive(
        model_path=args.model,
        params_path=params_path,
        collocated_gpu=args.collocated_gpu,
        prefill_gpus=args.prefill_gpus,
        decode_gpu=args.decode_gpu,
        arrival_rate=args.arrival_rate,
        duration=60.0,
        workload=args.workload,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        warmup_s=args.warmup,
        drain_timeout=30.0,
        seed=42,
    )
    print_summary(result)
    return _to_serialisable(result)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(i) for i in obj]
    if hasattr(obj, '__dataclass_fields__'):
        import dataclasses
        return _to_serialisable(dataclasses.asdict(obj))
    return obj


def _write_output(args, result: dict, t_total: float):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # path breakdown
    path_counts: dict = {}
    for req in result.get("requests", []):
        path_counts[req["path"]] = path_counts.get(req["path"], 0) + 1
    n = result["n_completed"] or 1

    lines = [
        "=" * 60,
        "nanoPD  —  Multi-GPU Adaptive Inference Demo",
        f"  model       : {args.model}",
        f"  hardware    : GPU{args.collocated_gpu} (coll)  "
            f"GPU{args.prefill_gpus} (prefill)  GPU{args.decode_gpu} (decode)",
        f"  workload    : {result['workload']}   "
            f"arrival_rate={result['arrival_rate']} rps",
        "=" * 60,
        "",
        "[throughput]",
        f"  completed   : {result['n_completed']}  dropped: {result['n_dropped']}",
        f"  throughput  : {result['throughput_rps']:.3f} rps  "
            f"({result['throughput_tokens_per_s']:.1f} tok/s)",
        "",
        "[latency]",
        f"  e2e  p50 = {result['p50_e2e_ms']:.0f} ms",
        f"       p95 = {result['p95_e2e_ms']:.0f} ms",
        f"       p99 = {result['p99_e2e_ms']:.0f} ms",
        f"  queue wait  p50 = {result['p50_queue_ms']:.0f} ms  "
            f"p99 = {result['p99_queue_ms']:.0f} ms",
        "",
        "[routing breakdown]",
    ]
    for path, cnt in sorted(path_counts.items()):
        lines.append(f"  {path:>14} : {cnt:4d} requests  ({cnt/n*100:.1f}%)")

    lines += [
        "",
        f"total wall time : {t_total:.1f} s",
        f"profile data    : {os.path.join(DATA_DIR, 'profile_data.pt')}",
        f"cost model      : {os.path.join(DATA_DIR, 'params.json')}",
        f"per-request JSON: {os.path.join(DATA_DIR, 'results.json')}",
        "=" * 60,
    ]

    text = "\n".join(lines)
    with open(OUTPUT_FILE, "w") as f:
        f.write(text + "\n")

    # full JSON
    results_json = os.path.join(DATA_DIR, "results.json")
    with open(results_json, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + text)
    print(f"\nSummary written to  {OUTPUT_FILE}")
    print(f"Full JSON written to {results_json}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--collocated-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpus",   type=int, nargs="+", default=[1, 3],
                        help="One or more GPUs for the prefill worker(s)")
    parser.add_argument("--decode-gpu",     type=int, default=2)
    parser.add_argument("--block-size",     type=int, default=16)
    parser.add_argument("--max-blocks",     type=int, default=512,
                        help="KV cache blocks per GPU (512×16 = 8192 token slots)")
    parser.add_argument("--arrival-rate",   type=float, default=0.2,
                        help="Poisson request arrival rate (req/s); "
                             "try 0.1–0.4 for Qwen3-8B on 4090")
    parser.add_argument("--workload",       default="mixed",
                        choices=["short", "long", "mixed"],
                        help="short=64-256 tok prompts, long=1024-2048, mixed=50/50")
    parser.add_argument("--warmup",         type=float, default=10.0,
                        help="Warmup time (s) before benchmark window starts")
    parser.add_argument("--skip-profile",   action="store_true",
                        help="Reuse output/data/profile_data.pt if it already exists")
    args = parser.parse_args()

    t_total_start = time.perf_counter()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    profile_path = os.path.join(DATA_DIR, "profile_data.pt")
    if args.skip_profile and os.path.exists(profile_path):
        print(f"[skip-profile] reusing {profile_path}")
    else:
        profile_path = step1_profile(args)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    params_path = step2_fit_and_analyse(profile_path)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    result = step3_benchmark(args, params_path)

    # ── Write output ──────────────────────────────────────────────────────────
    _write_output(args, result, time.perf_counter() - t_total_start)


if __name__ == "__main__":
    main()
