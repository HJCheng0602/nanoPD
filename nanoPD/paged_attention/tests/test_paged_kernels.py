"""
Correctness tests for paged_kernels:
  - paged_attention_forward (5-D key cache, 4-D value cache)
  - paged_kv_store

Build first:
    cd paged_attention && pip install -e . --no-build-isolation

Run:
    pytest tests/test_paged_kernels.py -v
    python tests/test_paged_kernels.py
"""

import math
import pytest
import torch
import torch.nn.functional as F

try:
    import paged_kernels
    HAS_PAGED_KERNELS = True
except ImportError:
    HAS_PAGED_KERNELS = False
    print("WARNING: paged_kernels not found. Build with: pip install -e . --no-build-isolation")

requires_kernels = pytest.mark.skipif(not HAS_PAGED_KERNELS, reason="paged_kernels not built")
requires_cuda    = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Layout conversion helpers
# ---------------------------------------------------------------------------

def _x_factor(dtype: torch.dtype) -> int:
    """Vectorisation factor: number of scalars per 16-byte load."""
    return 16 // torch.tensor([], dtype=dtype).element_size()


def to_k_cache_layout(k_std: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert key cache from standard layout [blocks, kv_heads, block_size, head_size]
    to the kernel's interleaved layout    [blocks, kv_heads, head_size//x, block_size, x].
    """
    x = _x_factor(dtype)
    blocks, kv_heads, block_size, head_size = k_std.shape
    return (k_std.view(blocks, kv_heads, block_size, head_size // x, x)
                 .permute(0, 1, 3, 2, 4)
                 .contiguous())


def to_v_cache_layout(v_std: torch.Tensor) -> torch.Tensor:
    """
    Convert value cache from standard layout [blocks, kv_heads, block_size, head_size]
    to the kernel's layout                   [blocks, kv_heads, head_size, block_size].
    """
    return v_std.permute(0, 1, 3, 2).contiguous()


# ---------------------------------------------------------------------------
# Reference attention (operates on standard-layout caches)
# ---------------------------------------------------------------------------

def ref_paged_attention(
    query:       torch.Tensor,   # [num_seqs, num_heads, head_size]
    key_cache:   torch.Tensor,   # [num_blocks, num_kv_heads, block_size, head_size]  (standard)
    value_cache: torch.Tensor,   # [num_blocks, num_kv_heads, block_size, head_size]  (standard)
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_seq]  int32
    seq_lens:    torch.Tensor,   # [num_seqs]  int32
    scale:       float,
) -> torch.Tensor:
    num_seqs, num_heads, head_size = query.shape
    num_kv_heads = key_cache.size(1)
    block_size   = key_cache.size(2)
    head_group   = num_heads // num_kv_heads

    outputs = []
    for s in range(num_seqs):
        slen    = seq_lens[s].item()
        nblocks = (slen + block_size - 1) // block_size

        k_blocks = [key_cache[block_tables[s, b].item()]   for b in range(nblocks)]
        v_blocks = [value_cache[block_tables[s, b].item()] for b in range(nblocks)]
        keys   = torch.cat(k_blocks, dim=1)[:, :slen, :].float()  # [kv_heads, slen, head_size]
        values = torch.cat(v_blocks, dim=1)[:, :slen, :].float()

        keys   = keys.repeat_interleave(head_group, dim=0)    # [num_heads, slen, head_size]
        values = values.repeat_interleave(head_group, dim=0)

        q    = query[s].unsqueeze(1).float()                  # [num_heads, 1, head_size]
        attn = torch.bmm(q, keys.transpose(1, 2)) * scale     # [num_heads, 1, slen]
        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, values).squeeze(1)             # [num_heads, head_size]
        outputs.append(out)

    return torch.stack(outputs).to(query.dtype)


# ---------------------------------------------------------------------------
# Input builder for paged_attention_forward
# ---------------------------------------------------------------------------

def make_attn_inputs(
    num_seqs:    int,
    num_heads:   int,
    num_kv_heads: int,
    head_size:   int,
    seq_lens:    list,
    block_size:  int,
    dtype:       torch.dtype,
    device:      str = "cuda",
    seed:        int = 42,
):
    torch.manual_seed(seed)
    assert num_heads % num_kv_heads == 0
    assert head_size % _x_factor(dtype) == 0

    max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens)
    total_blocks       = sum((s + block_size - 1) // block_size for s in seq_lens)

    query     = torch.randn(num_seqs, num_heads, head_size, dtype=dtype, device=device)
    # Standard-layout caches (for reference)
    k_std = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    v_std = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    # Kernel-layout caches
    key_cache   = to_k_cache_layout(k_std, dtype)
    value_cache = to_v_cache_layout(v_std)

    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    blk = 0
    for i, slen in enumerate(seq_lens):
        n = (slen + block_size - 1) // block_size
        for b in range(n):
            block_tables[i, b] = blk
            blk += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    scale      = 1.0 / math.sqrt(head_size)

    return query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks_per_seq


def run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size):
    out = torch.zeros_like(query)
    paged_kernels.paged_attention_forward(
        out, query, key_cache, value_cache,
        block_tables, seq_lens,
        scale, block_size, max_blocks,
    )
    return out


# ---------------------------------------------------------------------------
# Tests: paged_attention_forward
# ---------------------------------------------------------------------------

@requires_cuda
@requires_kernels
class TestPagedAttention:

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_single_seq_single_head(self, dtype):
        args = make_attn_inputs(1, 1, 1, 128, [8], 16, dtype)
        query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args

        ref = ref_paged_attention(query, k_std, v_std, block_tables, seq_lens, scale)
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, 16)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_multi_seq(self, dtype):
        args = make_attn_inputs(3, 4, 4, 128, [32, 48, 64], 16, dtype)
        query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args

        ref = ref_paged_attention(query, k_std, v_std, block_tables, seq_lens, scale)
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, 16)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gqa(self, dtype):
        """Grouped Query Attention: num_heads=32, num_kv_heads=8 (Llama-3 style)."""
        args = make_attn_inputs(2, 32, 8, 128, [64, 48], 16, dtype)
        query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args

        ref = ref_paged_attention(query, k_std, v_std, block_tables, seq_lens, scale)
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, 16)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 32, 64, 128, 512])
    def test_various_seq_lens(self, seq_len):
        args = make_attn_inputs(1, 4, 4, 128, [seq_len], 16, torch.float16)
        query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args

        ref = ref_paged_attention(query, k_std, v_std, block_tables, seq_lens, scale)
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, 16)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("block_size", [16, 32])
    def test_various_block_sizes(self, block_size):
        args = make_attn_inputs(2, 4, 4, 128, [48, 32], block_size, torch.float16)
        query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args

        ref = ref_paged_attention(query, k_std, v_std, block_tables, seq_lens, scale)
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_output_shape(self):
        num_seqs, num_heads, head_size = 4, 8, 128
        args = make_attn_inputs(num_seqs, num_heads, num_heads, head_size, [32]*num_seqs, 16, torch.float16)
        query, _, _, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, 16)
        assert out.shape == (num_seqs, num_heads, head_size)

    def test_bad_block_tables_dtype_raises(self):
        args = make_attn_inputs(1, 4, 4, 128, [16], 16, torch.float16)
        query, _, _, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args
        out = torch.zeros_like(query)
        with pytest.raises(RuntimeError):
            paged_kernels.paged_attention_forward(
                out, query, key_cache, value_cache,
                block_tables.to(torch.int64), seq_lens,
                scale, 16, max_blocks,
            )


# ---------------------------------------------------------------------------
# Input builder for paged_kv_store
# ---------------------------------------------------------------------------

def make_kv_store_inputs(
    num_kv_heads: int,
    seq_len:      int,
    head_dim:     int,
    block_size:   int,
    dtype:        torch.dtype,
    total_blocks: int = None,
    start_position: int = 0,
    device:       str = "cuda",
    seed:         int = 0,
):
    torch.manual_seed(seed)
    num_blocks_needed = (start_position + seq_len + block_size - 1) // block_size
    if total_blocks is None:
        total_blocks = num_blocks_needed

    # k_src / v_src shape: [1, num_kv_heads, seq_len, head_dim]
    k_src = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v_src = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)

    # k_cache / v_cache: [total_blocks, num_kv_heads, block_size, head_dim]
    k_cache = torch.zeros(total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)

    # Sequential block table
    block_table = torch.arange(num_blocks_needed, dtype=torch.int32, device=device)

    return k_src, v_src, k_cache, v_cache, block_table


# ---------------------------------------------------------------------------
# Tests: paged_kv_store
# ---------------------------------------------------------------------------

@requires_cuda
@requires_kernels
class TestPagedKVStore:

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_basic_store_and_readback(self, dtype):
        """Values written by paged_kv_store should match k_src / v_src."""
        num_kv_heads, seq_len, head_dim, block_size = 4, 16, 128, 16
        k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(
            num_kv_heads, seq_len, head_dim, block_size, dtype
        )

        paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_table, 0)
        torch.cuda.synchronize()

        # k_cache layout: [blocks, kv_heads, block_size, head_dim]
        # k_src layout:   [1, kv_heads, seq_len, head_dim]  ->  [kv_heads, seq_len, head_dim]
        k_src_3d = k_src.squeeze(0)  # [kv_heads, seq_len, head_dim]
        v_src_3d = v_src.squeeze(0)

        for tok in range(seq_len):
            blk_idx    = tok // block_size
            blk_offset = tok % block_size
            phys_blk   = block_table[blk_idx].item()

            for h in range(num_kv_heads):
                expected_k = k_src_3d[h, tok]
                stored_k   = k_cache[phys_blk, h, blk_offset]
                torch.testing.assert_close(stored_k, expected_k, atol=0, rtol=0)

                expected_v = v_src_3d[h, tok]
                stored_v   = v_cache[phys_blk, h, blk_offset]
                torch.testing.assert_close(stored_v, expected_v, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_start_position_nonzero(self, dtype):
        """With start_position > 0, tokens should land in the correct blocks."""
        num_kv_heads, seq_len, head_dim, block_size = 2, 8, 128, 16
        start_pos = 8  # tokens go to offsets 8..15 within block 0
        k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(
            num_kv_heads, seq_len, head_dim, block_size, dtype, start_position=start_pos
        )

        paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_table, start_pos)
        torch.cuda.synchronize()

        k_src_3d = k_src.squeeze(0)
        for tok in range(seq_len):
            abs_pos    = start_pos + tok
            blk_idx    = abs_pos // block_size
            blk_offset = abs_pos % block_size
            phys_blk   = block_table[blk_idx].item()
            for h in range(num_kv_heads):
                torch.testing.assert_close(
                    k_cache[phys_blk, h, blk_offset], k_src_3d[h, tok], atol=0, rtol=0
                )

    @pytest.mark.parametrize("seq_len", [1, 8, 16, 32, 64])
    def test_various_seq_lens(self, seq_len):
        k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(
            2, seq_len, 128, 16, torch.float16
        )
        paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_table, 0)
        torch.cuda.synchronize()

        k_src_3d = k_src.squeeze(0)
        for tok in range(seq_len):
            phys_blk   = block_table[tok // 16].item()
            blk_offset = tok % 16
            for h in range(2):
                torch.testing.assert_close(
                    k_cache[phys_blk, h, blk_offset], k_src_3d[h, tok], atol=0, rtol=0
                )

    def test_multi_head(self):
        """All kv_heads are written independently and correctly."""
        k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(
            8, 16, 128, 16, torch.float16
        )
        paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_table, 0)
        torch.cuda.synchronize()

        k_src_3d = k_src.squeeze(0)
        for h in range(8):
            for tok in range(16):
                torch.testing.assert_close(
                    k_cache[0, h, tok], k_src_3d[h, tok], atol=0, rtol=0
                )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(1)
    if not HAS_PAGED_KERNELS:
        print("paged_kernels not built. Run: pip install -e . --no-build-isolation")
        exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # --- paged_attention_forward ---
    print("=== paged_attention_forward ===")
    attn_cases = [
        dict(num_seqs=1,  num_heads=1,  num_kv_heads=1, head_size=128, seq_lens=[8],     block_size=16, dtype=torch.float16,  label="single seq/head fp16"),
        dict(num_seqs=3,  num_heads=4,  num_kv_heads=4, head_size=128, seq_lens=[32,48,64], block_size=16, dtype=torch.float16, label="multi-seq fp16"),
        dict(num_seqs=2,  num_heads=32, num_kv_heads=8, head_size=128, seq_lens=[64,48], block_size=16, dtype=torch.float16,  label="GQA fp16"),
        dict(num_seqs=2,  num_heads=4,  num_kv_heads=4, head_size=128, seq_lens=[48,32], block_size=16, dtype=torch.bfloat16, label="bf16"),
        dict(num_seqs=1,  num_heads=4,  num_kv_heads=4, head_size=128, seq_lens=[1],     block_size=16, dtype=torch.float16,  label="seq_len=1"),
        dict(num_seqs=1,  num_heads=4,  num_kv_heads=4, head_size=128, seq_lens=[512],   block_size=16, dtype=torch.float16,  label="seq_len=512"),
    ]
    passed = failed = 0
    for c in attn_cases:
        label = c.pop("label")
        bs    = c["block_size"]
        args  = make_attn_inputs(**c)
        query, k_std, v_std, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args
        ref = ref_paged_attention(query, k_std, v_std, block_tables, seq_lens, scale)
        out = run_attn_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, bs)
        err = (out.float() - ref.float()).abs().max().item()
        ok  = err < 0.02
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:45s}  max_err={err:.6f}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\n  {passed}/{passed+failed} attention tests passed.\n")

    # --- paged_kv_store ---
    print("=== paged_kv_store ===")
    kv_cases = [
        dict(num_kv_heads=4, seq_len=16,  head_dim=128, block_size=16, dtype=torch.float16,  label="basic fp16"),
        dict(num_kv_heads=4, seq_len=16,  head_dim=128, block_size=16, dtype=torch.bfloat16, label="basic bf16"),
        dict(num_kv_heads=8, seq_len=64,  head_dim=128, block_size=16, dtype=torch.float16,  label="multi-head seq64"),
        dict(num_kv_heads=2, seq_len=1,   head_dim=128, block_size=16, dtype=torch.float16,  label="seq_len=1"),
        dict(num_kv_heads=2, seq_len=32,  head_dim=128, block_size=32, dtype=torch.float16,  label="block_size=32"),
    ]
    passed = failed = 0
    for c in kv_cases:
        label = c.pop("label")
        k_src, v_src, k_cache, v_cache, block_table = make_kv_store_inputs(**c)
        bs = c["block_size"]
        paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_table, 0)
        torch.cuda.synchronize()
        k_src_3d = k_src.squeeze(0)
        ok = True
        for tok in range(c["seq_len"]):
            phys_blk   = block_table[tok // bs].item()
            blk_offset = tok % bs
            for h in range(c["num_kv_heads"]):
                if not torch.allclose(k_cache[phys_blk, h, blk_offset], k_src_3d[h, tok]):
                    ok = False
                    break
            if not ok:
                break
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\n  {passed}/{passed+failed} kv_store tests passed.")
