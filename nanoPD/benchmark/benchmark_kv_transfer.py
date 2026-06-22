"""
benchmark_kv_transfer.py — Microbenchmark for extract_kv_to_pinned and load_kv_from_pinned

Usage:
    python nanoPD/benchmark/benchmark_kv_transfer.py
    python nanoPD/benchmark/benchmark_kv_transfer.py --num-layers 32 --num-kv-heads 8 --head-dim 128
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import argparse
import torch
from workers.kv_transfer import PinnedKVBuffer, extract_kv_to_pinned, load_kv_from_pinned


def make_cache(num_layers, max_blocks, num_kv_heads, block_size, head_dim, device):
    shape = (num_layers, max_blocks, num_kv_heads, block_size, head_dim)
    k = torch.randn(shape, dtype=torch.float16, device=device)
    v = torch.randn(shape, dtype=torch.float16, device=device)
    return k, v


def time_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def run(args):
    device = torch.device(f"cuda:{args.gpu}")
    max_blocks = 512

    k_cache, v_cache = make_cache(
        args.num_layers, max_blocks, args.num_kv_heads,
        args.block_size, args.head_dim, device,
    )

    print(f"\nKV transfer microbenchmark  |  "
          f"layers={args.num_layers}  kv_heads={args.num_kv_heads}  "
          f"block_size={args.block_size}  head_dim={args.head_dim}  "
          f"dtype=float16  GPU={args.gpu}\n")
    print(f"{'n_blocks':>10}  {'extract ms':>12}  {'load ms':>10}  {'total ms':>10}")
    print("-" * 50)

    for n in args.n_blocks:
        block_table = list(range(n))
        buf = PinnedKVBuffer(
            num_layers=args.num_layers,
            num_block=n,
            num_kv_heads=args.num_kv_heads,
            block_size=args.block_size,
            head_dim=args.head_dim,
        )

        t_extract = time_fn(lambda: extract_kv_to_pinned(k_cache, v_cache, block_table, buf))
        torch.cuda.synchronize()
        t_load    = time_fn(lambda: load_kv_from_pinned(k_cache, v_cache, block_table, buf))
        torch.cuda.synchronize()

        print(f"{n:>10}  {t_extract:>12.3f}  {t_load:>10.3f}  {t_extract+t_load:>10.3f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",          type=int,         default=0)
    parser.add_argument("--num-layers",   type=int,         default=28)
    parser.add_argument("--num-kv-heads", type=int,         default=4)
    parser.add_argument("--block-size",   type=int,         default=16)
    parser.add_argument("--head-dim",     type=int,         default=128)
    parser.add_argument("--n-blocks",     type=int, nargs="+", default=[8, 16, 32, 64, 128])
    args = parser.parse_args()
    run(args)