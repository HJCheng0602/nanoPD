"""
plot_benchmark.py — Visualise results from benchmark/results.json

Usage:
    python benchmark/plot_benchmark.py \
        --input benchmark/results.json \
        --output benchmark/figures/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STRATEGIES = ["collocated", "disaggregated", "adaptive"]
WORKLOADS  = ["short", "long", "mixed"]

COLORS = {
    "collocated":    "#3498DB",
    "disaggregated": "#2ECC71",
    "adaptive":      "#E67E22",
}
LABELS = {
    "collocated":    "Collocated (GPU 0)",
    "disaggregated": "Disaggregated (GPU 1+2)",
    "adaptive":      "Adaptive (Router)",
}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── fig 1: p50 / p99 e2e latency grouped bar ─────────────────────────────────

def plot_latency_bar(data: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("End-to-End Latency: Collocated vs Disaggregated vs Adaptive",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, wl in zip(axes, WORKLOADS):
        strategies_present = [s for s in STRATEGIES if s in data.get(wl, {})]
        x = np.arange(len(strategies_present))
        w = 0.35

        p50s = [data[wl][s]["p50_e2e_ms"] / 1000 for s in strategies_present]
        p99s = [data[wl][s]["p99_e2e_ms"] / 1000 for s in strategies_present]
        colors = [COLORS[s] for s in strategies_present]

        bars50 = ax.bar(x - w/2, p50s, w, label="p50",
                        color=colors, alpha=0.9, edgecolor="white")
        bars99 = ax.bar(x + w/2, p99s, w, label="p99",
                        color=colors, alpha=0.45, edgecolor="white",
                        hatch="//")

        for bar, val in zip(bars50, p50s):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars99, p99s):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s].split()[0] for s in strategies_present],
                           fontsize=9)
        ax.set_title(f"Workload: {wl}", fontsize=12)
        ax.set_ylabel("Latency (s)" if wl == "short" else "")
        ax.grid(axis="y", alpha=0.3)

        # legend for hatch
        solid = mpatches.Patch(facecolor="gray", alpha=0.9, label="p50")
        hatch = mpatches.Patch(facecolor="gray", alpha=0.45,
                               hatch="//", label="p99")
        ax.legend(handles=[solid, hatch], fontsize=8, loc="upper left")

    plt.tight_layout()
    p = os.path.join(out_dir, "fig1_latency_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p}")


# ── fig 2: throughput comparison ──────────────────────────────────────────────

def plot_throughput(data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(WORKLOADS))
    n = len(STRATEGIES)
    w = 0.22
    offsets = np.linspace(-(n-1)/2 * w, (n-1)/2 * w, n)

    for offset, strategy in zip(offsets, STRATEGIES):
        vals = []
        for wl in WORKLOADS:
            if wl in data and strategy in data[wl]:
                vals.append(data[wl][strategy]["throughput_tokens_per_s"])
            else:
                vals.append(0)
        bars = ax.bar(x + offset, vals, w,
                      color=COLORS[strategy], alpha=0.85,
                      label=LABELS[strategy], edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS, fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax.set_title("Throughput: Collocated vs Disaggregated vs Adaptive",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig2_throughput.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p}")


# ── fig 3: per-request e2e scatter (adaptive path breakdown) ──────────────────

def plot_adaptive_scatter(data: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Adaptive: Per-Request E2E Latency vs Prompt Length",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, wl in zip(axes, WORKLOADS):
        if wl not in data or "adaptive" not in data[wl]:
            ax.set_visible(False)
            continue
        reqs = data[wl]["adaptive"].get("requests", [])
        if not reqs:
            continue

        for path, color in [("collocated", COLORS["collocated"]),
                             ("disaggregated", COLORS["disaggregated"])]:
            xs = [r["prompt_len"] for r in reqs if r["path"] == path]
            ys = [r["e2e_ms"] / 1000 for r in reqs if r["path"] == path]
            if xs:
                ax.scatter(xs, ys, c=color, alpha=0.8, s=60,
                           label=LABELS[path].split()[0], zorder=3)

        ax.set_xlabel("Prompt Length (tokens)", fontsize=10)
        ax.set_ylabel("E2E Latency (s)" if wl == "short" else "")
        ax.set_title(f"Workload: {wl}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig3_adaptive_scatter.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p}")


# ── fig 4: ttft for disaggregated ─────────────────────────────────────────────

def plot_ttft(data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_prompts, all_ttfts = [], []
    for wl in WORKLOADS:
        if wl not in data or "disaggregated" not in data[wl]:
            continue
        reqs = data[wl]["disaggregated"].get("requests", [])
        for r in reqs:
            if r.get("ttft_ms", 0) > 0:
                all_prompts.append(r["prompt_len"])
                all_ttfts.append(r["ttft_ms"])

    if all_prompts:
        ax.scatter(all_prompts, all_ttfts,
                   c=COLORS["disaggregated"], alpha=0.7, s=50, zorder=3)

        # fit line
        z = np.polyfit(all_prompts, all_ttfts, 1)
        xs = np.linspace(min(all_prompts), max(all_prompts), 200)
        ax.plot(xs, np.polyval(z, xs), "--",
                color="#E74C3C", linewidth=1.5,
                label=f"fit: {z[0]:.4f}·L + {z[1]:.1f} ms")

    ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
    ax.set_ylabel("TTFT (ms)", fontsize=12)
    ax.set_title("Disaggregated: Time to First Token vs Prompt Length\n"
                 "(prefill time only, excludes KV transfer)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "fig4_ttft.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="benchmark/results_h20.json")
    parser.add_argument("--output", default="benchmark/figures_h20")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load(args.input)

    plot_latency_bar(data, args.output)
    plot_throughput(data, args.output)
    plot_adaptive_scatter(data, args.output)
    plot_ttft(data, args.output)

    print(f"\nAll figures saved to {args.output}/")