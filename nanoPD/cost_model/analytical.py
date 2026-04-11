"""
Analytical Cost Model for Adaptive P/D Routing
Qwen3-8B + 8x RTX 4090 (pinned memory relay, no P2P direct)
"""

import os
import json
import torch
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, Optional

import importlib.util, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from profiler import ProfileResult
sys.modules['__main__'].ProfileResult = ProfileResult
DISAGG_FIXED_OVERHEAD_MS = 0

@dataclass
class CostModelParams:
    alpha: float = 0.0          # T_prefill(L) ≈ alpha * L  (ms/token)
    beta: float = 0.0           # T_decode single step, batch=1 (ms)
    batch_thresh: float = 16.0  # batch size where decode leaves memory-bound region
    gamma: float = 0.0          # T_interference ≈ gamma * prompt_tokens (ms/token)
    bandwidth_GBps: float = 12.90   # pinned relay measured bandwidth
    bytes_per_token: int = 147456   # Qwen3-8B: 36*8*128*2*2


class AnalyticalCostModel:
    def __init__(self, params: CostModelParams):
        self.p = params

    # ------------------------------------------------------------------ #
    #  Curve fitting                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def fit_from_profile(cls, profile_data_path: str,
                         save_params_path: Optional[str] = None) -> "AnalyticalCostModel":
        data = torch.load(profile_data_path, weights_only=False)
        params = CostModelParams()
        params.bandwidth_GBps = data.p2p_bandwidth_GBps

        def _linear(x, a):
            return a * np.array(x, dtype=float)

        # alpha: T_prefill(L) = alpha * L
        Ls = sorted(data.T_prefill.keys())
        ts = [data.T_prefill[L] for L in Ls]
        (alpha,), _ = curve_fit(_linear, Ls, ts)
        params.alpha = float(alpha)
        print(f"[fit] alpha = {params.alpha:.5f} ms/token")

        # beta: decode step time at batch=1
        key = (512, 1)
        if key in data.T_decode:
            params.beta = data.T_decode[key]
        else:
            b1 = [v for (kv, b), v in data.T_decode.items() if b == 1]
            params.beta = float(np.median(b1)) if b1 else 50.0
        print(f"[fit] beta  = {params.beta:.2f} ms/step")

        # batch_thresh: where decode time starts growing with batch
        kv_fixed = 512
        Bs_kv = sorted(set(b for (kv, b) in data.T_decode if kv == kv_fixed))
        ts_B = [data.T_decode.get((kv_fixed, B), 0) for B in Bs_kv]

        best_thresh, best_mse = 1.0, float('inf')
        for thresh_candidate in [float(b) for b in Bs_kv]:
            predicted = [params.beta * max(1.0, B / thresh_candidate) for B in Bs_kv]
            mse = np.mean([(p - t) ** 2 for p, t in zip(predicted, ts_B)])
            if mse < best_mse:
                best_mse = mse
                best_thresh = thresh_candidate
        params.batch_thresh = best_thresh
        print(f"[fit] batch_thresh = {params.batch_thresh:.0f}")

        # gamma: interference per prefill token (use batch=8 data)
        cs_list, infer_list = [], []
        for (chunk_size, batch), val in data.T_interference.items():
            if batch == 8 and val > 0:
                cs_list.append(chunk_size)
                infer_list.append(val)
        if len(cs_list) >= 2:
            (gamma,), _ = curve_fit(_linear, cs_list, infer_list)
            params.gamma = float(max(gamma, 0.0))
        print(f"[fit] gamma = {params.gamma:.5f} ms/token")

        transfer_rate = params.bytes_per_token / (params.bandwidth_GBps * 1e6)
        print(f"[fit] transfer_rate = {transfer_rate:.5f} ms/token")
        print(f"[fit] gamma/transfer_rate ratio = {params.gamma/transfer_rate:.1f}x")

        if save_params_path:
            os.makedirs(os.path.dirname(save_params_path), exist_ok=True)
            with open(save_params_path, "w") as f:
                json.dump(params.__dict__, f, indent=2)
            print(f"[fit] params saved → {save_params_path}")

        return cls(params)

    @classmethod
    def load_params(cls, params_path: str) -> "AnalyticalCostModel":
        with open(params_path) as f:
            d = json.load(f)
        return cls(CostModelParams(**d))

    # ------------------------------------------------------------------ #
    #  Latency estimation                                                 #
    # ------------------------------------------------------------------ #

    def t_prefill(self, prompt_len: int) -> float:
        return self.p.alpha * prompt_len

    def t_transfer(self, prompt_len: int) -> float:
        """KV cache pinned-relay transfer latency (ms)"""
        return (prompt_len * self.p.bytes_per_token / 1e9) / self.p.bandwidth_GBps * 1000.0

    def t_decode_step(self, batch_size: int = 1) -> float:
        if batch_size <= self.p.batch_thresh:
            return self.p.beta
        return self.p.beta * (batch_size / self.p.batch_thresh)

    def t_decode_total(self, output_len: int, batch_size: int = 1) -> float:
        return self.t_decode_step(batch_size) * output_len

    def t_collocated(self, prompt_len: int, output_len: int,
                    system_load: int, chunk_size: int = 256) -> float:
        t_p = self.t_prefill(prompt_len)
        t_d = self.t_decode_total(output_len, batch_size=max(system_load, 1))
        # interference grows linearly with load, capped at batch_thresh (compute-bound regime)
        if system_load <= 0:
            t_i = 0.0
        else:
            load_factor = min(system_load, self.p.batch_thresh) / self.p.batch_thresh
            t_i = self.p.gamma * prompt_len * load_factor
        return t_p + t_d + t_i

    def t_disaggregated(self, prompt_len: int, output_len: int) -> float:
        return (self.t_prefill(prompt_len)
                + self.t_transfer(prompt_len)
                + self.t_decode_total(output_len, batch_size=1))

    def route(self, prompt_len: int, predicted_output_len: int,
            system_load: int, decode_batch_size: int = 1,
            chunk_size: int = 256) -> Tuple[str, float, float]:
        t_c = self.t_collocated(prompt_len, predicted_output_len, system_load, chunk_size)
        # disaggregated decode uses the actual decode batch size
        t_d = (self.t_prefill(prompt_len)
            + self.t_transfer(prompt_len)
            + self.t_decode_total(predicted_output_len, batch_size=decode_batch_size + 1))
        decision = "disaggregated" if t_d < t_c else "collocated"
        return decision, t_c, t_d

    # ------------------------------------------------------------------ #
    #  Visualisation — generate 4 plots                                   #
    # ------------------------------------------------------------------ #

    def plot_all(self, output_dir: str = "cost_model/figures",
                 output_len: int = 200, chunk_size: int = 256):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        os.makedirs(output_dir, exist_ok=True)
        saved = []

        COLORS = {
            "red":    "#E74C3C",
            "green":  "#2ECC71",
            "blue":   "#3498DB",
            "orange": "#E67E22",
            "purple": "#9B59B6",
            "gray":   "#95A5A6",
        }

        prompt_lens  = [64, 128, 256, 512, 1024, 2048]
        system_loads = [0, 1, 2, 4, 8, 16, 32]

        # ── Figure 1: Routing Decision Heatmap ────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        grid = np.zeros((len(system_loads), len(prompt_lens)))
        for i, load in enumerate(system_loads):
            for j, L in enumerate(prompt_lens):
                dec, _, _ = self.route(L, output_len, load, decode_batch_size=1, chunk_size=chunk_size)
                grid[i, j] = 1.0 if dec == "disaggregated" else 0.0

        ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1,
                  aspect="auto", origin="lower")
        ax.set_xticks(range(len(prompt_lens)))
        ax.set_xticklabels(prompt_lens, fontsize=11)
        ax.set_yticks(range(len(system_loads)))
        ax.set_yticklabels(system_loads, fontsize=11)
        ax.set_xlabel("Prompt Length (tokens)", fontsize=13)
        ax.set_ylabel("System Load (concurrent decode requests)", fontsize=13)
        ax.set_title("Adaptive Router Decision\n"
                     "Green (D) = Disaggregated  |  Red (C) = Collocated",
                     fontsize=13)
        for i in range(len(system_loads)):
            for j in range(len(prompt_lens)):
                label = "D" if grid[i, j] == 1 else "C"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(output_dir, "fig1_routing_heatmap.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)
        print(f"[plot] {p}")

        # ── Figure 2: Latency Breakdown (load=0 vs load=8) ───────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, load in zip(axes, [0, 8]):
            t_cs, t_ds = [], []
            for L in prompt_lens:
                _, tc, td = self.route(L, output_len, load, decode_batch_size=1, chunk_size=chunk_size)
                t_cs.append(tc)
                t_ds.append(td)
            x = np.arange(len(prompt_lens))
            w = 0.35
            ax.bar(x - w/2, t_cs, w, label="Collocated",
                   color=COLORS["red"], alpha=0.85, edgecolor="white")
            ax.bar(x + w/2, t_ds, w, label="Disaggregated",
                   color=COLORS["green"], alpha=0.85, edgecolor="white")
            for j, (tc, td) in enumerate(zip(t_cs, t_ds)):
                winner = "D" if td < tc else "C"
                c = COLORS["green"] if winner == "D" else COLORS["red"]
                ax.text(j, max(tc, td) * 1.03, winner,
                        ha="center", fontsize=10, color=c, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(prompt_lens, fontsize=10)
            ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
            ax.set_ylabel("Estimated E2E Latency (ms)", fontsize=12)
            ax.set_title(f"system_load={load}  |  output_len={output_len}",
                         fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Collocated vs Disaggregated: End-to-End Latency",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(output_dir, "fig2_latency_comparison.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)
        print(f"[plot] {p}")

        # ── Figure 3: Transfer Overhead vs Interference ───────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        L_range = np.linspace(64, 2048, 200)
        transfer_ms    = [self.t_transfer(L) for L in L_range]
        interference_0 = [self.p.gamma * L for L in L_range]   # load=0 → no interference
        interference_8 = [self.p.gamma * L for L in L_range]   # same formula, shown for load=8

        ax.plot(L_range, transfer_ms, color=COLORS["blue"],
                linewidth=2.5, label="KV Transfer cost (pinned relay, 12.9 GB/s)")
        ax.plot(L_range, interference_8, color=COLORS["orange"],
                linewidth=2.5, linestyle="--",
                label="Prefill interference on decode (any load ≥ 1)")
        ax.fill_between(L_range, transfer_ms, interference_8,
                        where=[t < i for t, i in zip(transfer_ms, interference_8)],
                        alpha=0.12, color=COLORS["green"],
                        label="Region where Disagg wins (transfer < interference)")
        ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
        ax.set_ylabel("Overhead (ms)", fontsize=12)
        ax.set_title("KV Transfer Overhead vs Prefill→Decode Interference\n"
                     "γ/transfer_rate ≈ {:.0f}× → Disagg almost always wins at load≥1".format(
                         self.p.gamma / (self.p.bytes_per_token /
                                         (self.p.bandwidth_GBps * 1e6))),
                     fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(output_dir, "fig3_transfer_vs_interference.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)
        print(f"[plot] {p}")

        # ── Figure 4: Decode Step Time vs Batch Size ──────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        Bs_fine = np.arange(1, 33)
        decode_ms = [self.t_decode_step(int(B)) for B in Bs_fine]
        ax.plot(Bs_fine, decode_ms, color=COLORS["purple"],
                linewidth=2.5, marker="o", markersize=4)
        ax.axvline(x=self.p.batch_thresh, color=COLORS["gray"],
                   linestyle="--", linewidth=1.5,
                   label=f"batch_thresh = {self.p.batch_thresh:.0f} "
                         f"(memory→compute boundary)")
        ax.set_xlabel("Decode Batch Size", fontsize=12)
        ax.set_ylabel("Decode Step Latency (ms)", fontsize=12)
        ax.set_title("Decode Step Latency vs Batch Size\n"
                     "(flat = memory-bound, rising = compute-bound)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(output_dir, "fig4_decode_batch_scaling.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)
        print(f"[plot] {p}")

        print(f"\n[plot] all {len(saved)} figures saved to {output_dir}/")
        return saved

    def print_summary(self, output_len: int = 50, chunk_size: int = 256):
        p = self.p
        tr = p.bytes_per_token / (p.bandwidth_GBps * 1e6)
        print("\n" + "=" * 65)
        print("Cost Model Summary  (Qwen3-8B + 8x RTX 4090, pinned relay)")
        print("=" * 65)
        print(f"  alpha         = {p.alpha:.5f} ms/token   (prefill rate)")
        print(f"  beta          = {p.beta:.2f} ms/step    (decode, batch=1)")
        print(f"  gamma         = {p.gamma:.5f} ms/token   (interference rate)")
        print(f"  bandwidth     = {p.bandwidth_GBps:.2f} GB/s     (pinned relay)")
        print(f"  transfer_rate = {tr:.5f} ms/token")
        print(f"  gamma/transfer= {p.gamma/tr:.1f}x  ← at load≥1, disagg almost always wins")
        print()
        print(f"  {'L_p':>5}  {'load':>5}  {'T_colloc':>10}  {'T_disagg':>10}  decision")
        print(f"  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  --------")
        for load in [0, 1, 4, 8, 16]:
            for L in [128, 256, 512, 1024, 2048]:
                dec, tc, td = self.route(L, output_len, load, decode_batch_size=1, chunk_size=chunk_size)
                marker = " ◀" if load == 0 and dec == "collocated" else ""
                print(f"  {L:>5}  {load:>5}  {tc:>9.1f}ms  {td:>9.1f}ms  {dec}{marker}")
            print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="cost_model/profile_data_h20.pt")
    parser.add_argument("--save-params", default="cost_model/params_h20.json")
    parser.add_argument("--figures", default="cost_model/figures_h20")
    parser.add_argument("--output-len", type=int, default=200)
    args = parser.parse_args()

    model = AnalyticalCostModel.fit_from_profile(args.profile, args.save_params)
    model.print_summary(output_len=args.output_len)
    model.plot_all(output_dir=args.figures, output_len=args.output_len)