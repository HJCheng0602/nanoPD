"""
Benchmark & profiling for paged_attention_forward.

Usage:
    # Quick benchmark table
    python benchmark_paged_attention.py

    # With torch.profiler (generates Chrome-trace JSON)
    python benchmark_paged_attention.py --profile

    # With NVTX ranges (use nsys to capture)
    #   nsys profile --trace=cuda,nvtx python benchmark_paged_attention.py --nvtx
    python benchmark_paged_attention.py --nvtx

Build extension first:
    cd paged_attention && pip install -e .
"""

import argparse
import math
import sys
import time

import torch
import torch.utils.benchmark as tb

try:
    import paged_kernels as paged_attn
    HAS_PAGED_ATTN = True
except ImportError:
    print("ERROR: paged_kernels not built. Run: python setup.py build_ext --inplace")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------

def make_inputs(
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    seq_len: int,           # same for all seqs (decoding step)
    block_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    assert num_heads % num_kv_heads == 0
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq

    query       = torch.randn(num_seqs, num_heads, head_size, dtype=dtype, device=device)
    key_cache   = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    value_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)

    block_tables = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_tables[i, b] = i * num_blocks_per_seq + b

    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    scale    = 1.0 / math.sqrt(head_size)

    out = torch.zeros_like(query)
    return out, query, key_cache, value_cache, block_tables, seq_lens, scale, num_blocks_per_seq


def call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size):
    paged_attn.paged_attention_forward(
        out, query, key_cache, value_cache,
        block_tables, seq_lens,
        scale, block_size, max_blocks,
    )


# ---------------------------------------------------------------------------
# torch.utils.benchmark grid
# ---------------------------------------------------------------------------

CONFIGS = [
    # (num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype)
    (1,   32, 32, 128,   128, 16, torch.float16),
    (1,   32, 32, 128,   512, 16, torch.float16),
    (1,   32, 32, 128,  1024, 16, torch.float16),
    (8,   32, 32, 128,   256, 16, torch.float16),
    (16,  32, 32, 128,   256, 16, torch.float16),
    (32,  32, 32, 128,   256, 16, torch.float16),
    # GQA (Llama-3 style: 32 Q heads, 8 KV heads)
    (1,   32,  8, 128,   512, 16, torch.float16),
    (8,   32,  8, 128,   512, 16, torch.float16),
    (16,  32,  8, 128,   512, 16, torch.float16),
    # bfloat16
    (8,   32, 32, 128,   256, 16, torch.bfloat16),
]


def run_benchmark():
    print(f"{'num_seqs':>10} {'num_heads':>10} {'kv_heads':>9} {'head_size':>10} "
          f"{'seq_len':>8} {'blk_sz':>7} {'dtype':>8}  {'lat(us)':>10}  {'tflops':>8}")
    print("-" * 95)

    for (num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype) in CONFIGS:
        inputs = make_inputs(num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype)
        out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs

        # Warm-up
        for _ in range(5):
            call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
        torch.cuda.synchronize()

        t = tb.Timer(
            stmt="call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)",
            globals={**locals(), "call_kernel": call_kernel},
        )
        result = t.blocked_autorange(min_run_time=1.0)
        lat_us = result.mean * 1e6

        # Rough FLOPs: for each seq, 2 * num_heads * seq_len * head_size (QK^T) + same (AV)
        flops = num_seqs * num_heads * 2 * 2 * seq_len * head_size
        tflops = flops / (result.mean * 1e12)

        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        print(f"{num_seqs:>10} {num_heads:>10} {num_kv_heads:>9} {head_size:>10} "
              f"{seq_len:>8} {block_size:>7} {dtype_str:>8}  {lat_us:>10.2f}  {tflops:>8.4f}")


# ---------------------------------------------------------------------------
# torch.profiler (generates a Chrome JSON trace)
# ---------------------------------------------------------------------------

def run_profiler(trace_file: str = "paged_attn_trace.json"):
    from torch.profiler import profile, record_function, ProfilerActivity

    inputs = make_inputs(8, 32, 8, 128, 512, 16, torch.float16)
    out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs
    block_size = 16

    # Warm-up
    for _ in range(3):
        call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_log"),
    ) as prof:
        for _ in range(7):
            with record_function("paged_attention"):
                call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
            torch.cuda.synchronize()
            prof.step()

    # Also export Chrome trace
    prof.export_chrome_trace(trace_file)
    print(f"\n[Profiler] Chrome trace saved to: {trace_file}")
    print(f"[Profiler] TensorBoard log saved to: ./tb_log")
    print("\nTop CUDA kernels:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


# ---------------------------------------------------------------------------
# NVTX ranges (for nsys / nvprof)
# ---------------------------------------------------------------------------

def run_nvtx():
    try:
        import nvtx
        has_nvtx = True
    except ImportError:
        has_nvtx = False
        print("nvtx package not found – falling back to torch.cuda.nvtx")

    inputs = make_inputs(8, 32, 8, 128, 512, 16, torch.float16)
    out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs
    block_size = 16

    for _ in range(3):
        call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    torch.cuda.synchronize()

    for i in range(10):
        if has_nvtx:
            with nvtx.annotate(f"paged_attn_iter_{i}", color="green"):
                call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
        else:
            torch.cuda.nvtx.range_push(f"paged_attn_iter_{i}")
            call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
            torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print("NVTX ranges emitted. Capture with:")
    print("  nsys profile --trace=cuda,nvtx -o paged_attn_report python benchmark_paged_attention.py --nvtx")


# ---------------------------------------------------------------------------
# Roofline helpers (manual CUDA event timing)
# ---------------------------------------------------------------------------

def cuda_event_timing(num_seqs=8, num_heads=32, num_kv_heads=8,
                      head_size=128, seq_len=512, block_size=16,
                      dtype=torch.float16, n_iters=100):
    """More accurate latency using CUDA events (no Python overhead)."""
    inputs = make_inputs(num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype)
    out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs

    # warm-up
    for _ in range(10):
        call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    torch.cuda.synchronize()

    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)

    start_e.record()
    for _ in range(n_iters):
        call_kernel(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    end_e.record()
    torch.cuda.synchronize()

    avg_ms = start_e.elapsed_time(end_e) / n_iters
    print(f"\n[CUDA event timing] avg latency: {avg_ms*1000:.2f} us  ({n_iters} iters)")
    return avg_ms


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Run torch.profiler and save trace")
    parser.add_argument("--nvtx",    action="store_true", help="Emit NVTX ranges (pair with nsys)")
    parser.add_argument("--events",  action="store_true", help="Single-config CUDA event timing")
    args = parser.parse_args()

    if args.profile:
        run_profiler()
    elif args.nvtx:
        run_nvtx()
    elif args.events:
        cuda_event_timing()
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print()
        run_benchmark()
        print()
        cuda_event_timing()
