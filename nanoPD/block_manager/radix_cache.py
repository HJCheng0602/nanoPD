from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PrefixCacheMatch:
    """Result returned by a block-aligned prefix cache lookup."""

    token_ids: Tuple[int, ...]
    blocks: Tuple[Any, ...]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def block_table(self) -> List[int]:
        return [block.block_num for block in self.blocks]


@dataclass(frozen=True)
class PrefixCacheInsertResult:
    num_blocks: int
    num_new_blocks: int
    num_existing_blocks: int
    num_tokens: int


@dataclass
class _RadixNode:
    token_ids: Tuple[int, ...] = ()
    block: Any = None
    children: Dict[Tuple[int, ...], "_RadixNode"] = field(default_factory=dict)


class RadixKVCache:
    """Block-granular prefix cache for paged KV blocks."""

    def __init__(
        self,
        block_size: int,
        release_block: Callable[[Any], None],
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self._release_block = release_block
        self._root = _RadixNode()
        self._num_cached_blocks = 0

    @property
    def num_cached_blocks(self) -> int:
        return self._num_cached_blocks

    def match(self, token_ids: Sequence[int]) -> Optional[PrefixCacheMatch]:
        """Return the longest cached prefix made of complete token blocks."""
        node = self._root
        matched_tokens: List[int] = []
        matched_blocks: List[Any] = []

        for chunk in self._iter_full_blocks(token_ids):
            child = node.children.get(chunk)
            if child is None:
                break
            matched_tokens.extend(chunk)
            matched_blocks.append(child.block)
            node = child

        if not matched_blocks:
            return None

        return PrefixCacheMatch(
            token_ids=tuple(matched_tokens),
            blocks=tuple(matched_blocks),
        )

    def insert(
        self,
        token_ids: Sequence[int],
        blocks: Sequence[Any],
    ) -> PrefixCacheInsertResult:
        """Insert full token blocks and retain newly cached physical blocks."""
        chunks = self._iter_full_blocks(token_ids)
        if len(blocks) < len(chunks):
            raise ValueError(
                "blocks must cover every full token block in token_ids"
            )

        node = self._root
        num_new = 0
        num_existing = 0

        for chunk, block in zip(chunks, blocks):
            child = node.children.get(chunk)
            if child is None:
                self._retain_block(block)
                child = _RadixNode(token_ids=chunk, block=block)
                node.children[chunk] = child
                self._num_cached_blocks += 1
                num_new += 1
            else:
                num_existing += 1
            node = child

        num_blocks = num_new + num_existing
        return PrefixCacheInsertResult(
            num_blocks=num_blocks,
            num_new_blocks=num_new,
            num_existing_blocks=num_existing,
            num_tokens=num_blocks * self.block_size,
        )

    def clear(self):
        for node in list(self._iter_nodes_postorder(self._root)):
            if node is not self._root:
                self._release_block(node.block)
        self._root.children.clear()
        self._num_cached_blocks = 0

    def _iter_full_blocks(
        self,
        token_ids: Sequence[int],
    ) -> Tuple[Tuple[int, ...], ...]:
        full_len = (len(token_ids) // self.block_size) * self.block_size
        return tuple(
            tuple(token_ids[start:start + self.block_size])
            for start in range(0, full_len, self.block_size)
        )

    def _retain_block(self, block: Any):
        assert block.ref_cout > 0
        block.ref_cout += 1

    def _iter_nodes_postorder(self, node: _RadixNode):
        for child in list(node.children.values()):
            yield from self._iter_nodes_postorder(child)
        yield node
