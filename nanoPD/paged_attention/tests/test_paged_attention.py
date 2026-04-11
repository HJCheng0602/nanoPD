"""
Correctness tests for paged_attention_forward.

Build first:
    cd paged_attention && pip install -e .

Run:
    pytest test_paged_attention.py -v
    # or directly:
    python test_paged_attention.py
"""

import pytest
import torch
import torch.nn.functional as F
import math

try:
    import paged_attn
    HAS_PAGED_ATTN = True
except ImportError:
    HAS_PAGED_ATTN = False
    print("WARNING: paged_attn not found. Build with: pip install -e .")

requires_paged_attn = pytest.mark.skipif(not HAS_PAGED_ATTN, reason="paged_attn not built")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def ref_paged_attention(
    query: torch.Tensor,        # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    block_tables: torch.Tensor, # [num_seqs, max_num_blocks_per_seq]  int32
    seq_lens: torch.Tensor,     # [num_seqs]  int32
    scale: float,
) -> torch.Tensor:
    """Pure-PyTorch reference: gather KV from paged cache, run standard attention."""
    num_seqs, num_heads, head_size = query.shape
    num_kv_heads = key_cache.size(1)
    block_size = key_cache.size(2)
    head_group = num_heads // num_kv_heads

    outputs = []
    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        # Gather K/V blocks  →  [num_kv_heads, seq_len, head_size]
        k_blocks = [key_cache[block_tables[seq_idx, b].item()]   for b in range(num_blocks_for_seq)]
        v_blocks = [value_cache[block_tables[seq_idx, b].item()] for b in range(num_blocks_for_seq)]
        keys   = torch.cat(k_blocks, dim=1)[:, :seq_len, :]   # [kv_heads, seq_len, head]
        values = torch.cat(v_blocks, dim=1)[:, :seq_len, :]

        # GQA: expand kv heads  →  [num_heads, seq_len, head_size]
        keys   = keys.repeat_interleave(head_group, dim=0)
        values = values.repeat_interleave(head_group, dim=0)

        q = query[seq_idx].unsqueeze(1).float()   # [num_heads, 1, head_size]
        k = keys.float()
        v = values.float()

        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [num_heads, 1, seq_len]
        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, v).squeeze(1)             # [num_heads, head_size]
        outputs.append(out)

    return torch.stack(outputs).to(query.dtype)


# ---------------------------------------------------------------------------
# Helper: build random paged KV cache + block tables
# ---------------------------------------------------------------------------

def make_inputs(
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    seq_lens: list,
    block_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
):
    torch.manual_seed(seed)
    assert num_heads % num_kv_heads == 0

    max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens)
    total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)

    query       = torch.randn(num_seqs, num_heads, head_size, dtype=dtype, device=device)
    key_cache   = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)
    value_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device)

    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    block_counter = 0
    for i, seq_len in enumerate(seq_lens):
        n = (seq_len + block_size - 1) // block_size
        for b in range(n):
            block_tables[i, b] = block_counter
            block_counter += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(head_size)

    return query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks_per_seq


def run_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks_per_seq, block_size):
    out = torch.zeros_like(query)
    paged_attn.paged_attention_forward(
        out, query, key_cache, value_cache,
        block_tables, seq_lens,
        scale, block_size, max_blocks_per_seq,
    )
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_cuda
@requires_paged_attn
class TestCorrectness:

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_single_seq_single_head(self, dtype):
        """Simplest case: 1 sequence, 1 head, seq_len fits in one block."""
        args = make_inputs(
            num_seqs=1, num_heads=1, num_kv_heads=1, head_size=64,
            seq_lens=[8], block_size=16, dtype=dtype,
        )
        query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks = args
        block_size = 16

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, max_blocks, block_size)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_multi_seq(self, dtype):
        """Multiple sequences with different lengths."""
        seq_lens = [4, 8, 12]
        args = make_inputs(
            num_seqs=3, num_heads=4, num_kv_heads=4, head_size=64,
            seq_lens=seq_lens, block_size=8, dtype=dtype,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args
        block_size = 8

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens_t, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, block_size)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gqa(self, dtype):
        """Grouped Query Attention: num_heads > num_kv_heads."""
        args = make_inputs(
            num_seqs=2, num_heads=8, num_kv_heads=2, head_size=64,
            seq_lens=[16, 8], block_size=8, dtype=dtype,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args
        block_size = 8

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens_t, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, block_size)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", [1, 7, 8, 9, 16, 32])
    def test_various_seq_lens(self, seq_len):
        """Seq lengths that are below / equal / above block boundaries."""
        block_size = 8
        args = make_inputs(
            num_seqs=1, num_heads=4, num_kv_heads=4, head_size=64,
            seq_lens=[seq_len], block_size=block_size, dtype=torch.float16,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens_t, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, block_size)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("block_size", [4, 8, 16, 32])
    def test_various_block_sizes(self, block_size):
        args = make_inputs(
            num_seqs=2, num_heads=4, num_kv_heads=4, head_size=64,
            seq_lens=[24, 16], block_size=block_size, dtype=torch.float16,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens_t, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, block_size)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_paged_attn
class TestEdgeCases:

    def test_seq_len_one(self):
        """Single token – attention over itself."""
        args = make_inputs(
            num_seqs=1, num_heads=4, num_kv_heads=4, head_size=64,
            seq_lens=[1], block_size=16, dtype=torch.float16,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens_t, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, 16)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_output_shape(self):
        num_seqs, num_heads, head_size = 4, 8, 128
        args = make_inputs(
            num_seqs=num_seqs, num_heads=num_heads, num_kv_heads=num_heads, head_size=head_size,
            seq_lens=[10] * num_seqs, block_size=16, dtype=torch.float16,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, 16)
        assert out.shape == (num_seqs, num_heads, head_size)

    def test_dtype_mismatch_raises(self):
        """Kernel should reject mismatched dtypes (block_tables must be int32)."""
        args = make_inputs(
            num_seqs=1, num_heads=4, num_kv_heads=4, head_size=64,
            seq_lens=[8], block_size=8, dtype=torch.float16,
        )
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args
        bad_block_tables = block_tables.to(torch.int64)
        out = torch.zeros_like(query)
        with pytest.raises(RuntimeError):
            paged_attn.paged_attention_forward(
                out, query, key_cache, value_cache,
                bad_block_tables, seq_lens_t,
                scale, 8, max_blocks,
            )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        exit(1)
    if not HAS_PAGED_ATTN:
        print("paged_attn not built. Run: pip install -e .")
        exit(1)

    cases = [
        dict(num_seqs=1,  num_heads=1,  num_kv_heads=1, head_size=64,  seq_lens=[8],         block_size=16, dtype=torch.float16, label="single seq/head fp16"),
        dict(num_seqs=3,  num_heads=4,  num_kv_heads=4, head_size=64,  seq_lens=[4,8,12],    block_size=8,  dtype=torch.float16, label="multi-seq fp16"),
        dict(num_seqs=2,  num_heads=8,  num_kv_heads=2, head_size=64,  seq_lens=[16,8],      block_size=8,  dtype=torch.float16, label="GQA fp16"),
        dict(num_seqs=2,  num_heads=4,  num_kv_heads=4, head_size=128, seq_lens=[32,24],     block_size=16, dtype=torch.bfloat16,label="bf16 h128"),
        dict(num_seqs=1,  num_heads=4,  num_kv_heads=4, head_size=64,  seq_lens=[1],         block_size=8,  dtype=torch.float16, label="seq_len=1"),
    ]

    passed = failed = 0
    for c in cases:
        label = c.pop("label")
        block_size = c["block_size"]
        args = make_inputs(**c)
        query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks = args

        ref = ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens_t, scale)
        out = run_kernel(query, key_cache, value_cache, block_tables, seq_lens_t, scale, max_blocks, block_size)

        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err < 0.02
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label:40s}  max_err={max_err:.6f}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
