"""
Benchmark: FlashInfer vs hand-written CUDA paged attention.
"""

import math
import sys
import time

import torch

sys.path.insert(0, '/home/zhouyuhan01/nanoPD/nanoPD/paged_attention')
import paged_kernels as paged_attn

import flashinfer

# ---------------------------------------------------------------------------
# Input builders (same layout as existing benchmark)
# ---------------------------------------------------------------------------

def make_inputs_paged(num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size, dtype=torch.float16, device='cuda'):
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq
    query = torch.randn(num_seqs, num_heads, head_size, dtype=dtype, device=device)
    key_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    value_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    block_tables = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_tables[i, b] = i * num_blocks_per_seq + b
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(head_size)
    out = torch.zeros_like(query)
    return out, query, key_cache, value_cache, block_tables, seq_lens, scale, num_blocks_per_seq


# ---------------------------------------------------------------------------
# FlashInfer wrapper
# ---------------------------------------------------------------------------

def run_flashinfer(query, key_cache, value_cache, block_tables, seq_lens, scale, block_size):
    num_seqs = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    num_kv_heads = key_cache.size(1)
    max_num_pages = key_cache.size(0)
    seq_len = int(seq_lens[0].item())
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    # FlashInfer HND layout: [max_num_pages, num_kv_heads, page_size, head_dim]
    # This matches our key_cache/value_cache shape exactly

    # Build indptr, indices, last_page_len for FlashInfer
    # indptr: [batch_size + 1], cumulative number of pages
    indptr = torch.arange(0, (num_seqs + 1) * num_blocks_per_seq, num_blocks_per_seq, dtype=torch.int32, device='cuda')

    # indices: flat list of physical page indices for all sequences
    indices = block_tables.view(-1).contiguous()

    # last_page_len: number of valid tokens in the last page of each sequence
    last_page_len = torch.full((num_seqs,), seq_len % block_size if seq_len % block_size != 0 else block_size, dtype=torch.int32, device='cuda')

    # Workspace buffer
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device='cuda')

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout='HND')
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        page_size=block_size,
        pos_encoding_mode='NONE',
        sm_scale=scale,
    )

    out = wrapper.run(query, (key_cache, value_cache))
    return out


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_cuda_event(fn, warmup=10, iters=100):
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
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA not available')
        sys.exit(1)

    print(f'Device: {torch.cuda.get_device_name(0)}')
    print()

    CONFIGS = [
        # (num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size)
        (1,  32, 32, 128, 128,  16),
        (1,  32, 32, 128, 512,  16),
        (1,  32, 32, 128, 1024, 16),
        (8,  32, 32, 128, 256,  16),
        (16, 32, 32, 128, 256,  16),
        (32, 32, 32, 128, 256,  16),
        (1,  32, 8,  128, 512,  16),
        (8,  32, 8,  128, 512,  16),
        (16, 32, 8,  128, 512,  16),
    ]

    for num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size in CONFIGS:
        print(f'\n--- Config: batch={num_seqs}, heads={num_heads}, kv_heads={num_kv_heads}, head_size={head_size}, seq_len={seq_len} ---')

        out, query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = make_inputs_paged(
            num_seqs, num_heads, num_kv_heads, head_size, seq_len, block_size
        )

        # Hand-written CUDA benchmark
        cuda_lat = bench_cuda_event(
            lambda: paged_attn.paged_attention_forward(
                out, query, key_cache, value_cache, block_tables, seq_lens, scale, block_size, max_blocks
            )
        )
        print(f'Hand-written CUDA latency: {cuda_lat * 1000:.2f} us')

        # FlashInfer benchmark
        try:
            # Correctness check first
            fi_out = run_flashinfer(query, key_cache, value_cache, block_tables, seq_lens, scale, block_size)
            # Compare with CUDA output
            cuda_out = out.clone()
            paged_attn.paged_attention_forward(
                cuda_out, query, key_cache, value_cache, block_tables, seq_lens, scale, block_size, max_blocks
            )
            print(f'FlashInfer correct: {torch.allclose(fi_out, cuda_out, atol=5e-2, rtol=5e-2)}')

            # Pre-create wrapper to benchmark run() only (excluding plan overhead)
            num_blocks_per_seq = (seq_len + block_size - 1) // block_size
            indptr = torch.arange(0, (num_seqs + 1) * num_blocks_per_seq, num_blocks_per_seq, dtype=torch.int32, device='cuda')
            indices = block_tables.view(-1).contiguous()
            last_page_len = torch.full((num_seqs,), seq_len % block_size if seq_len % block_size != 0 else block_size, dtype=torch.int32, device='cuda')
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device='cuda')
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout='HND')
            wrapper.plan(
                indptr=indptr, indices=indices, last_page_len=last_page_len,
                num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_size,
                page_size=block_size, pos_encoding_mode='NONE', sm_scale=scale,
            )

            fi_lat = bench_cuda_event(lambda: wrapper.run(query, (key_cache, value_cache)))
            print(f'FlashInfer latency:      {fi_lat * 1000:.2f} us')
            print(f'  => FlashInfer / CUDA:  {fi_lat / cuda_lat:.2f}x')
        except Exception as e:
            print(f'FlashInfer FAILED: {e}')
            import traceback
            traceback.print_exc()

