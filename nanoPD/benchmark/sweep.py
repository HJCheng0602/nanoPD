# benchmark/sweep.py
"""
python benchmark/sweep.py
Runs collocated / disaggregated / adaptive across multiple arrival rates,
appends results to benchmark/results_poisson.json, then saves all figures to benchmark/figures/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Configuration ──────────────────────────────────────────────────────────────────────

RATES       = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 2.50, 3.00]
STRATEGIES  = ["collocated", "disaggregated", "adaptive"]
WORKLOAD    = "mixed"
DURATION    = 60
WARMUP      = 10
DRAIN       = 30
MAX_BLOCKS  = 4096

MODEL       = "/root/autodl-tmp/models/Qwen3-8B"
PARAMS      = "cost_model/params_h20.json"
COLL_GPU    = 0
PREFILL_GPUS = [1, 3]
DECODE_GPU  = 2
PREFILL_GPU = 1          # for disaggregated single worker
SEED        = 42

OUTPUT_JSON = "benchmark/results_poisson_h20.json"
FIG_DIR     = "benchmark/figures_h20"


# ── Run benchmark_poisson.py ─────────────────────────────────────────────────────

def run_one(strategy: str, rate: float) -> bool:
    key = f"{strategy}_{WORKLOAD}_{rate}"
    # skip if this (strategy, rate) combination already exists
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            existing = json.load(f)
        if key in existing:
            print(f"  [skip] {key} already in results")
            return True

    cmd = [
        "python", "benchmark/benchmark_poisson.py",
        "--model",          MODEL,
        "--params",         PARAMS,
        "--strategy",       strategy,
        "--workload",       WORKLOAD,
        "--arrival-rate",   str(rate),
        "--duration",       str(DURATION),
        "--warmup",         str(WARMUP),
        "--drain-timeout",  str(DRAIN),
        "--max-blocks",     str(MAX_BLOCKS),
        "--seed",           str(SEED),
        "--output",         OUTPUT_JSON,
        "--collocated-gpu", str(COLL_GPU),
        "--decode-gpu",     str(DECODE_GPU),
    ]
    if strategy == "adaptive":
        cmd += ["--prefill-gpus"] + [str(g) for g in PREFILL_GPUS]
    else:
        cmd += ["--prefill-gpu", str(PREFILL_GPU)]

    print(f"\n{'='*60}")
    print(f"  strategy={strategy}  rate={rate}rps")
    print(f"{'='*60}")
    ret = subprocess.run(cmd)
    return ret.returncode == 0


# ── Plotting ───────────────────────────────────────────────────────────────────────

COLORS = {
    "collocated":    "#3498DB",
    "disaggregated": "#E74C3C",
    "adaptive":      "#2ECC71",
}
LABELS = {
    "collocated":    "Collocated",
    "disaggregated": "Disaggregated",
    "adaptive":      "Adaptive (P/D)",
}


def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def extract_series(results: dict, strategy: str, metric: str):
    xs, ys = [], []
    for rate in RATES:
        key = f"{strategy}_{WORKLOAD}_{rate}"
        if key not in results:
            continue
        r = results[key]
        val = r.get(metric)
        if val is not None and val > 0:
            xs.append(rate)
            ys.append(val)
    return xs, ys


def plot_all(results: dict, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)

    # ── Figure 1: p50 e2e latency vs arrival rate ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for strat in STRATEGIES:
        xs, ys = extract_series(results, strat, "p50_e2e_ms")
        if xs:
            ax.plot(xs, [y/1000 for y in ys], marker="o", linewidth=2,
                    color=COLORS[strat], label=LABELS[strat])
    ax.set_xlabel("Arrival Rate (req/s)", fontsize=13)
    ax.set_ylabel("P50 E2E Latency (s)", fontsize=13)
    ax.set_title("P50 End-to-End Latency vs Arrival Rate", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_p50_e2e.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[plot] {p}")

    # ── Figure 2: p99 e2e latency ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for strat in STRATEGIES:
        xs, ys = extract_series(results, strat, "p99_e2e_ms")
        if xs:
            ax.plot(xs, [y/1000 for y in ys], marker="s", linewidth=2,
                    color=COLORS[strat], label=LABELS[strat])
    ax.set_xlabel("Arrival Rate (req/s)", fontsize=13)
    ax.set_ylabel("P99 E2E Latency (s)", fontsize=13)
    ax.set_title("P99 End-to-End Latency vs Arrival Rate", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_p99_e2e.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[plot] {p}")

    # ── Figure 3: throughput (tok/s) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for strat in STRATEGIES:
        xs, ys = extract_series(results, strat, "throughput_tokens_per_s")
        if xs:
            ax.plot(xs, ys, marker="^", linewidth=2,
                    color=COLORS[strat], label=LABELS[strat])
    ax.set_xlabel("Arrival Rate (req/s)", fontsize=13)
    ax.set_ylabel("Throughput (tok/s)", fontsize=13)
    ax.set_title("Throughput vs Arrival Rate", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_throughput.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[plot] {p}")

    # ── Figure 4: completed / dropped ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for strat in STRATEGIES:
        xs_c, ys_c = extract_series(results, strat, "n_completed")
        xs_d, ys_d = extract_series(results, strat, "n_dropped")
        if xs_c:
            axes[0].plot(xs_c, ys_c, marker="o", linewidth=2,
                         color=COLORS[strat], label=LABELS[strat])
        if xs_d:
            axes[1].plot(xs_d, ys_d, marker="x", linewidth=2,
                         color=COLORS[strat], label=LABELS[strat])
    for ax, title, ylabel in zip(axes,
            ["Completed Requests", "Dropped Requests"],
            ["Count", "Count"]):
        ax.set_xlabel("Arrival Rate (req/s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_completed_dropped.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[plot] {p}")

    # ── Figure 5: latency distribution violin (per-request e2e_ms)────────────────
    # plot only low / mid / high rate points
    target_rates = [0.10, 1, 2]
    fig, axes = plt.subplots(1, len(target_rates), figsize=(15, 5), sharey=True)
    for ax, rate in zip(axes, target_rates):
        data, labels, colors = [], [], []
        for strat in STRATEGIES:
            key = f"{strat}_{WORKLOAD}_{rate}"
            if key not in results:
                continue
            reqs = results[key].get("requests", [])
            e2es = [r["e2e_ms"] / 1000 for r in reqs]
            if e2es:
                data.append(e2es)
                labels.append(LABELS[strat])
                colors.append(COLORS[strat])
        if data:
            parts = ax.violinplot(data, showmedians=True)
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f"rate={rate} rps", fontsize=12)
        ax.set_ylabel("E2E Latency (s)" if ax == axes[0] else "", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("E2E Latency Distribution by Strategy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_latency_violin.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[plot] {p}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="skip benchmark runs and only regenerate plots")
    parser.add_argument("--rates", type=float, nargs="+", default=RATES)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    args = parser.parse_args()

    if not args.plot_only:
        failed = []
        for strategy in args.strategies:
            for rate in args.rates:
                ok = run_one(strategy, rate)
                if not ok:
                    failed.append((strategy, rate))
                    print(f"  [WARN] {strategy} rate={rate} failed, continuing...")

        if failed:
            print(f"\n[WARN] {len(failed)} runs failed: {failed}")

    print("\n\nDrawing figures...")
    results = load_results(OUTPUT_JSON)
    plot_all(results, FIG_DIR)
    print(f"\nAll figures saved to {FIG_DIR}/")