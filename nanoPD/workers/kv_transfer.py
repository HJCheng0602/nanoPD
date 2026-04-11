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
    for i, bid in enumerate(block_table):
        buf.k[:, i].copy_(k_cache[:, bid])
        buf.v[:, i].copy_(v_cache[:, bid])


def load_kv_from_pinned(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: List[int],
    buf: PinnedKVBuffer,
    stream: torch.cuda.Stream = None,
):
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        for i, bid in enumerate(block_table):
            k_cache[:, bid].copy_(buf.k[:, i], non_blocking=(stream is not None))
            v_cache[:, bid].copy_(buf.v[:, i], non_blocking=(stream is not None))


def transfer_kv(src_k, src_v, dst_k, dst_v, block_table, stream=None, buf=None) -> str:
    src_device = src_k.device
    dst_device = dst_k.device

    if _check_p2p(src_device, dst_device) and src_k is not None:
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            for i, bid in enumerate(block_table):
                dst_k[:, bid].copy_(src_k[:, bid], non_blocking=(stream is not None))
                dst_v[:, bid].copy_(src_v[:, bid], non_blocking=(stream is not None))
        return "p2p"
    else:
        assert buf is not None
        load_kv_from_pinned(dst_k, dst_v, block_table, buf, stream=stream)
        return "pinned_relay"