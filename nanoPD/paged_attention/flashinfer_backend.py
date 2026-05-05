"""
FlashInfer-backed paged attention kernel, drop-in replacement for
paged_kernels.paged_attention_forward.

Usage in model_runner.py:
    from paged_attention.flashinfer_backend import run_kernel as paged_attn_run_kernel
    # then use paged_attn_run_kernel(...) with the same signature as before.
"""

import math
import torch
import flashinfer

# ---------------------------------------------------------------------------
# Per-device workspace cache – required for multi-GPU deployments
# ---------------------------------------------------------------------------
_WORKSPACE_MAP: dict[str, torch.Tensor] = {}
_WRAPPER_MAP: dict[str, flashinfer.BatchDecodeWithPagedKVCacheWrapper] = {}


def _ensure_workspace(device: torch.device):
    """Create (or reuse) a FlashInfer workspace + wrapper bound to *device*."""
    key = str(device)
    if key not in _WORKSPACE_MAP:
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, kv_layout="HND", use_tensor_cores=True
        )
        _WORKSPACE_MAP[key] = ws
        _WRAPPER_MAP[key] = wrapper


# ---------------------------------------------------------------------------
# Core: convert block_tables + seq_lens -> FlashInfer page metadata
# ---------------------------------------------------------------------------

def _build_page_metadata(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
):
    """
    Build indptr / indices / last_page_len from block_tables and seq_lens.

    Parameters
    ----------
    block_tables : [batch_size, max_blocks_per_seq]
    seq_lens     : [batch_size]
    block_size   : int

    Returns
    -------
    indptr        : [batch_size + 1]
    indices       : [total_num_pages]
    last_page_len : [batch_size]
    """
    batch_size = block_tables.size(0)
    device = block_tables.device

    # Number of pages actually used by each sequence
    num_blocks = (seq_lens + block_size - 1) // block_size

    # indptr: cumulative sum of num_blocks
    indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    indptr[0] = 0
    torch.cumsum(num_blocks, dim=0, out=indptr[1:])

    # indices: flatten the valid portion of block_tables
    # Use boolean mask to select only the used blocks, then flatten
    # max_blocks_per_seq may be larger than actual blocks
    max_blocks = block_tables.size(1)
    # Build a mask [batch_size, max_blocks] where first num_blocks[i] entries are True
    arange = torch.arange(max_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    mask = arange < num_blocks.unsqueeze(1)  # [batch_size, max_blocks]
    indices = block_tables[mask].contiguous()

    # last_page_len: tokens in the last physical page of each sequence
    last_page_len = seq_lens % block_size
    last_page_len[last_page_len == 0] = block_size
    last_page_len = last_page_len.to(torch.int32)

    return indptr, indices, last_page_len


# ---------------------------------------------------------------------------
# Drop-in replacement for run_kernel in model_runner.py
# ---------------------------------------------------------------------------

def run_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    block_size: int,
    max_blocks_per_seq: int,
) -> torch.Tensor:
    """
    Compute paged attention for decoding step.

    Signature is identical to the existing CUDA kernel wrapper so that
    model_runner.py can swap backends with a single import change.

    Parameters
    ----------
    query       : [batch_size, num_heads, head_dim]
    key_cache   : [max_num_pages, num_kv_heads, page_size, head_dim]
    value_cache : [max_num_pages, num_kv_heads, page_size, head_dim]
    block_tables: [batch_size, max_blocks_per_seq]
    seq_lens    : [batch_size]
    scale       : float
    block_size  : int
    max_blocks_per_seq : int (unused, kept for API compatibility)

    Returns
    -------
    out : [batch_size, num_heads, head_dim]
    """
    device = query.device
    _ensure_workspace(device)

    batch_size = query.size(0)
    num_heads = query.size(1)
    head_dim = query.size(2)
    num_kv_heads = key_cache.size(1)

    indptr, indices, last_page_len = _build_page_metadata(
        block_tables, seq_lens, block_size
    )

    # Plan is lightweight (~few us) but required before each run.
    # FlashInfer caches internal tile configs keyed by problem shape.
    wrapper = _WRAPPER_MAP[str(device)]
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=block_size,
        pos_encoding_mode="NONE",
        sm_scale=scale,
    )

    out = wrapper.run(query, (key_cache, value_cache))
    return out
