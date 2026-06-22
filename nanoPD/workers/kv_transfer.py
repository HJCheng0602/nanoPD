import torch
from contextlib import nullcontext
from typing import List


def _check_p2p(src_device: torch.device, dst_device: torch.device) -> bool:
    if src_device == dst_device:
        return False
    if src_device.type != "cuda" or dst_device.type != "cuda":
        return False
    try:
        return torch.cuda.can_device_access_peer(dst_device.index, src_device.index)
    except Exception:
        return False


class PinnedKVBuffer:

    def __init__(
        self,
        num_layers: int,
        num_block: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype=torch.float16,
    ):
        shape = (num_layers, num_block, num_kv_heads, block_size, head_dim)
        self.k = torch.empty(shape, dtype=dtype, pin_memory=True)
        self.v = torch.empty(shape, dtype=dtype, pin_memory=True)

    @staticmethod
    def from_runner(runner, num_blocks: int) -> "PinnedKVBuffer":
        num_layers = runner.k_cache.shape[0]
        return PinnedKVBuffer(
            num_layers=num_layers,
            num_block=num_blocks,
            num_kv_heads=runner.num_kv_heads,
            block_size=runner.block_size,
            head_dim=runner.head_dim,
        )


def extract_kv_to_pinned(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: List[int],
    buf: PinnedKVBuffer,
):
    indices = torch.tensor(block_table, dtype=torch.long, device=k_cache.device)
    n = len(block_table)
    buf.k[:, :n].copy_(k_cache[:, indices])
    buf.v[:, :n].copy_(v_cache[:, indices])


def load_kv_from_pinned(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: List[int],
    buf: PinnedKVBuffer,
    stream: torch.cuda.Stream = None,
):
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        nb = stream is not None
        indices = torch.tensor(block_table, dtype=torch.long, device=k_cache.device)
        n = len(block_table)

        tmp_k = buf.k[:, :n].to(k_cache.device, non_blocking=nb)
        tmp_v = buf.v[:, :n].to(v_cache.device, non_blocking=nb)
        k_cache[:, indices] = tmp_k
        v_cache[:, indices] = tmp_v


def transfer_kv(src_k, src_v, dst_k, dst_v, block_table, stream=None, buf=None) -> str:
    src_device = src_k.device
    dst_device = dst_k.device

    if _check_p2p(src_device, dst_device) and src_k is not None:
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            nb = stream is not None
            for bid in block_table:
                dst_k[:, bid].copy_(src_k[:, bid], non_blocking=nb)
                dst_v[:, bid].copy_(src_v[:, bid], non_blocking=nb)
        return "p2p"
    else:
        assert buf is not None
        load_kv_from_pinned(dst_k, dst_v, block_table, buf, stream=stream)
        return "pinned_relay"