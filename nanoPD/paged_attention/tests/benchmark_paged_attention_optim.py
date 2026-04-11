"""
Benchmark & profiling for paged_kernels (paged_attention_forward + paged_kv_store).

Usage:
    # Quick benchmark table (default)
    python benchmark_paged_kernels.py

    # With torch.profiler (generates Chrome-trace JSON)
    python benchmark_paged_kernels.py --profile

    # With NVTX ranges (use nsys to capture)
    python benchmark_paged_kernels.py --nvtx

    # Attention only / kv_store only
    python benchmark_paged_kernels.py --attn-only
    python benchmark_paged_kernels.py --kv-only

Build extension first:
    cd paged_attention && pip install -e . --no-build-isolation
"""

import argparse
import math
import sys

import torch
import torch.utils.benchmark as tb

try:
    import paged_kernels
    HAS_PAGED_KERNELS = True
except ImportError:
    print("ERROR: paged_kernels not built. Run: pip install -e . --no-build-isolation")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Layout helpers (same as test file)
# ---------------------------------------------------------------------------

def _x_factor(dtype: torch.dtype) -> int:
    return 16 // torch.tensor([], dtype=dtype).element_size()


def to_k_cache_layout(k_std: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    x = _x_factor(dtype)
    blocks, kv_heads, block_size, head_size = k_std.shape
    return (k_std.view(blocks, kv_heads, block_size, head_size // x, x)
                 .permute(0, 1, 3, 2, 4)
                 .contiguous())


def to_v_cache_layout(v_std: torch.Tensor) -> torch.Tensor:
    return v_std.permute(0, 1, 3, 2).contiguous()


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def make_attn_inputs(num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype,
                     device="cuda"):
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks       = num_seqs * num_blocks_per_seq

    query = torch.randn(num_seqs, num_heads, head_size, dtype=dtype, device=device)
    k_std = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    v_std = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    key_cache   = to_k_cache_layout(k_std, dtype)
    value_cache = to_v_cache_layout(v_std)

    block_tables = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_tables[i, b] = i * num_blocks_per_seq + b

    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    scale    = 1.0 / math.sqrt(head_size)
    out      = torch.zeros_like(query)

    return out, query, key_cache, value_cache, block_tables, seq_lens, scale, num_blocks_per_seq


def call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size):
    paged_kernels.paged_attention_forward(
        out, query, key_cache, value_cache,
        block_tables, seq_lens,
        scale, block_size, max_blocks,
    )


def make_kv_store_inputs(num_kv_heads, seq_len, head_dim, block_size, dtype, device="cuda"):
    num_blocks = (seq_len + block_size - 1) // block_size
    k_src   = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v_src   = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return k_src, v_src, k_cache, v_cache, block_table


def call_kv_store(k_cache, v_cache, k_src, v_src, block_table):
    paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_table, 0)


# ---------------------------------------------------------------------------
# Benchmark configs
# Attention configs mirror benchmark_paged_attention.py exactly.
# ---------------------------------------------------------------------------

ATTN_CONFIGS = [
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

KV_STORE_CONFIGS = [
    # (num_kv_heads, seq_len, head_dim, block_size, dtype)
    (8,   128, 128, 16, torch.float16),
    (8,   512, 128, 16, torch.float16),
    (8,  1024, 128, 16, torch.float16),
    (8,  2048, 128, 16, torch.float16),
    (8,   512, 128, 32, torch.float16),
    (32,  512, 128, 16, torch.float16),
    (8,   512, 128, 16, torch.bfloat16),
]


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------

def _dtype_str(dtype):
    return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(dtype, str(dtype))


def run_attn_benchmark():
    print("=== paged_attention_forward ===")
    print(f"{'num_seqs':>10} {'num_heads':>10} {'kv_heads':>9} {'head_size':>10} "
          f"{'seq_len':>8} {'blk_sz':>7} {'dtype':>6}  {'lat(us)':>10}  {'tflops':>8}")
    print("-" * 98)

    for (num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype) in ATTN_CONFIGS:
        inputs = make_attn_inputs(num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype)
        out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs

        for _ in range(5):
            call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
        torch.cuda.synchronize()

        t = tb.Timer(
            stmt="call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)",
            globals={**locals(), "call_attn": call_attn},
        )
        result  = t.blocked_autorange(min_run_time=1.0)
        lat_us  = result.mean * 1e6
        # Approx FLOPs: QK^T + AV, each 2 * heads * seq_len * head_size MAC
        flops  = num_seqs * num_heads * 2 * 2 * seq_len * head_size
        tflops = flops / (result.mean * 1e12)

        print(f"{num_seqs:>10} {num_heads:>10} {num_kv_heads:>9} {head_size:>10} "
              f"{seq_len:>8} {block_size:>7} {_dtype_str(dtype):>6}  {lat_us:>10.2f}  {tflops:>8.4f}")


def run_kv_store_benchmark():
    print("\n=== paged_kv_store ===")
    print(f"{'kv_heads':>9} {'seq_len':>8} {'head_dim':>9} {'blk_sz':>7} {'dtype':>6}  {'lat(us)':>10}  {'GB/s':>8}")
    print("-" * 68)

    for (num_kv_heads, seq_len, head_dim, block_size, dtype) in KV_STORE_CONFIGS:
        k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(
            num_kv_heads, seq_len, head_dim, block_size, dtype
        )

        for _ in range(5):
            call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
        torch.cuda.synchronize()

        t = tb.Timer(
            stmt="call_kv_store(k_cache, v_cache, k_src, v_src, block_table)",
            globals={**locals(), "call_kv_store": call_kv_store},
        )
        result = t.blocked_autorange(min_run_time=1.0)
        lat_us = result.mean * 1e6
        # Bytes written: 2 tensors (K+V) * seq_len * kv_heads * head_dim
        bytes_written = 2 * seq_len * num_kv_heads * head_dim * torch.tensor([], dtype=dtype).element_size()
        gb_s = bytes_written / (result.mean * 1e9)

        print(f"{num_kv_heads:>9} {seq_len:>8} {head_dim:>9} {block_size:>7} "
              f"{_dtype_str(dtype):>6}  {lat_us:>10.2f}  {gb_s:>8.2f}")


# ---------------------------------------------------------------------------
# CUDA event timing (single config, low Python overhead)
# ---------------------------------------------------------------------------

def cuda_event_timing_attn(num_seqs=8, num_heads=32, num_kv_heads=8,
                            head_size=128, seq_len=512, block_size=16,
                            dtype=torch.float16, n_iters=100):
    inputs = make_attn_inputs(num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype)
    out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs

    for _ in range(10):
        call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    torch.cuda.synchronize()

    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)
    start_e.record()
    for _ in range(n_iters):
        call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    end_e.record()
    torch.cuda.synchronize()

    avg_us = start_e.elapsed_time(end_e) / n_iters * 1000
    print(f"\n[CUDA event] paged_attention_forward  "
          f"(seqs={num_seqs}, heads={num_heads}, kv={num_kv_heads}, h={head_size}, "
          f"seq_len={seq_len}, blk={block_size}, {_dtype_str(dtype)})  "
          f"avg={avg_us:.2f} us  ({n_iters} iters)")


def cuda_event_timing_kv(num_kv_heads=8, seq_len=512, head_dim=128, block_size=16,
                          dtype=torch.float16, n_iters=100):
    k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(
        num_kv_heads, seq_len, head_dim, block_size, dtype
    )

    for _ in range(10):
        call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
    torch.cuda.synchronize()

    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)
    start_e.record()
    for _ in range(n_iters):
        call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
    end_e.record()
    torch.cuda.synchronize()

    avg_us = start_e.elapsed_time(end_e) / n_iters * 1000
    print(f"[CUDA event] paged_kv_store  "
          f"(kv_heads={num_kv_heads}, seq_len={seq_len}, head_dim={head_dim}, "
          f"blk={block_size}, {_dtype_str(dtype)})  "
          f"avg={avg_us:.2f} us  ({n_iters} iters)")


# ---------------------------------------------------------------------------
# torch.profiler
# ---------------------------------------------------------------------------

def run_profiler(trace_file="paged_kernels_trace.json"):
    from torch.profiler import profile, record_function, ProfilerActivity

    # attention
    inputs = make_attn_inputs(8, 32, 8, 128, 512, 16, torch.float16)
    out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs
    block_size = 16
    for _ in range(3):
        call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
    torch.cuda.synchronize()

    # kv_store
    k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(8, 512, 128, 16, torch.float16)
    for _ in range(3):
        call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_log_kernels"),
    ) as prof:
        for _ in range(7):
            with record_function("paged_attention"):
                call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
            with record_function("paged_kv_store"):
                call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(trace_file)
    print(f"\n[Profiler] Chrome trace saved to: {trace_file}")
    print(f"[Profiler] TensorBoard log saved to: ./tb_log_kernels")
    print("\nTop CUDA kernels:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# ---------------------------------------------------------------------------
# NVTX
# ---------------------------------------------------------------------------

def run_nvtx():
    try:
        import nvtx
        has_nvtx = True
    except ImportError:
        has_nvtx = False
        print("nvtx package not found – falling back to torch.cuda.nvtx")

    inputs = make_attn_inputs(8, 32, 8, 128, 512, 16, torch.float16)
    out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = inputs
    block_size = 16

    k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(8, 512, 128, 16, torch.float16)

    for _ in range(3):
        call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
        call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
    torch.cuda.synchronize()

    for i in range(10):
        if has_nvtx:
            with nvtx.annotate(f"paged_attn_iter_{i}", color="green"):
                call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
            with nvtx.annotate(f"paged_kv_store_iter_{i}", color="blue"):
                call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
        else:
            torch.cuda.nvtx.range_push(f"paged_attn_iter_{i}")
            call_attn(out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"paged_kv_store_iter_{i}")
            call_kv_store(k_cache, v_cache, k_src, v_src, block_table)
            torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print("NVTX ranges emitted. Capture with:")
    print("  nsys profile --trace=cuda,nvtx -o paged_kernels_report python benchmark_paged_kernels.py --nvtx")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile",   action="store_true", help="Run torch.profiler and save trace")
    parser.add_argument("--nvtx",      action="store_true", help="Emit NVTX ranges (pair with nsys)")
    parser.add_argument("--events",    action="store_true", help="Single-config CUDA event timing")
    parser.add_argument("--attn-only", action="store_true", help="Benchmark paged_attention_forward only")
    parser.add_argument("--kv-only",   action="store_true", help="Benchmark paged_kv_store only")
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    if args.profile:
        run_profiler()
    elif args.nvtx:
        run_nvtx()
    elif args.events:
        cuda_event_timing_attn()
        cuda_event_timing_kv()
    else:
        if not args.kv_only:
            run_attn_benchmark()
        if not args.attn_only:
            run_kv_store_benchmark()
        print()
        cuda_event_timing_attn()
        cuda_event_timing_kv()
