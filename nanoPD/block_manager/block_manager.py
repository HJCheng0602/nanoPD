from dataclasses import dataclass
from typing import Dict, List, Optional, Union

try:
    from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
except ImportError:
    from sequence import Sequence, SequenceGroup, SequenceStatus

try:
    from block_manager.radix_cache import PrefixCacheMatch, RadixKVCache
except ImportError:
    from radix_cache import PrefixCacheMatch, RadixKVCache

@dataclass
class PhysicalBlock:
    block_num: int
    ref_cout: int = 0


class BlockAllocator:
    """
    allocate a GPU block pool, with stack order to utilizing the cache
    """
    def __init__(self, num_blocks:int):
        self._free_blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_num=i) for i in range(num_blocks)
        ]
        self._num_blocks = num_blocks
    
    def allocate(self) -> PhysicalBlock:
        if not self._free_blocks:
            raise MemoryError("OOM : no free physical blocks")
        block = self._free_blocks.pop()
        block.ref_cout = 1
        return block

    def free(self, block: PhysicalBlock):
        assert block.ref_cout > 0
        block.ref_cout -= 1
        if block.ref_cout == 0:
            self._free_blocks.append(block)
    
    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)
    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks
    
class BlockSpaceManager:
    """
    maintain the blocktable for each sequence: logic_idx -> physical_idx

    can_allocate / allocate :   allocate new block when prefill
    can_append_slot / append_slot : check and extend block when decode
    fork                          : beam search / prefix sharing
    free                          : free after the sequence finished
    get_block_table               : set kernel input for model runner
    """
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks:int,
        enable_prefix_cache: bool = False,
    ):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_gpu_blocks)
        self._block_table: Dict[int, List[PhysicalBlock]] = {}
        self.prefix_cache: Optional[RadixKVCache] = None
        if enable_prefix_cache:
            self.prefix_cache = RadixKVCache(
                block_size=block_size,
                release_block=self.allocator.free,
            )

    def can_allocate(self, seq_group:SequenceGroup) -> bool:
        seq = seq_group.get_seqs()[0]           # usually there is only one seq in a group when prefill
        return self.allocator.num_free_blocks >= seq.num_logic_blocks

    def can_allocate_with_prefix_cache(self, seq_group: SequenceGroup) -> bool:
        if self.prefix_cache is None:
            return self.can_allocate(seq_group)

        needed_blocks = 0
        for seq in seq_group.get_seqs(SequenceStatus.WAITING):
            match = self.match_cached_prefix(seq)
            cached_blocks = match.num_blocks if match is not None else 0
            needed_blocks += seq.num_logic_blocks - cached_blocks
        return self.allocator.num_free_blocks >= needed_blocks

    def allocate(self, seq_group:SequenceGroup):
        for seq in seq_group.get_seqs(SequenceStatus.WAITING):
            block_table:List[PhysicalBlock] = []
            for _ in range(seq.num_logic_blocks):
                block_table.append(self.allocator.allocate())
            self._block_table[seq.seq_id] = block_table
            seq.status = SequenceStatus.RUNNING

    def allocate_with_prefix_cache(
        self,
        seq_group: SequenceGroup,
    ) -> Dict[int, Optional[PrefixCacheMatch]]:
        if self.prefix_cache is None:
            self.allocate(seq_group)
            return {}

        waiting = seq_group.get_seqs(SequenceStatus.WAITING)
        planned_matches = {
            seq.seq_id: self.match_cached_prefix(seq)
            for seq in waiting
        }
        needed_blocks = 0
        for seq in waiting:
            match = planned_matches[seq.seq_id]
            cached_blocks = match.num_blocks if match is not None else 0
            needed_blocks += seq.num_logic_blocks - cached_blocks

        if self.allocator.num_free_blocks < needed_blocks:
            raise MemoryError("OOM : no free physical blocks")

        matches: Dict[int, Optional[PrefixCacheMatch]] = {}
        for seq in waiting:
            match = planned_matches[seq.seq_id]
            cached_blocks = list(match.blocks) if match is not None else []
            suffix_blocks = seq.num_logic_blocks - len(cached_blocks)

            block_table: List[PhysicalBlock] = []
            for block in cached_blocks:
                self._retain_block(block)
                block_table.append(block)
            for _ in range(suffix_blocks):
                block_table.append(self.allocator.allocate())

            self._block_table[seq.seq_id] = block_table
            seq.num_computed_tokens = len(cached_blocks) * self.block_size
            seq.status = SequenceStatus.RUNNING
            matches[seq.seq_id] = match
        return matches

    
    def can_append_slot(self, seq_group:SequenceGroup) -> bool:
        num_running = len(seq_group.get_seqs(SequenceStatus.RUNNING))
        return self.allocator.num_free_blocks >= num_running
    
    def append_slot(self, seq:Sequence):
        '''
        use when decode for each sequence:
        return:
            None            : no CoW, no extra
            (old_num, new_num) : CoW, copy the old block context to the new context
        
        if seq append logic block, use for it
        if the last physical block's ref count is not 1, then CoW
        '''
        block_table = self._block_table[seq.seq_id]
        next_position = max(seq.num_computed_tokens, seq.total_len - 1)
        needed_blocks = next_position // self.block_size + 1

        if len(block_table) < needed_blocks:
            new_block = self.allocator.allocate()
            block_table.append(new_block)
            return None
        
        last_block = block_table[-1]
        if last_block.ref_cout > 1:
            new_block = self.allocator.allocate()
            self.allocator.free(last_block)
            block_table[-1] = new_block
            return (last_block.block_num, new_block.block_num)
        
        return None
    
    def fork(self, parent:Sequence, child:Sequence):
        parent_table = self._block_table[parent.seq_id]
        child_table = list(parent_table)
        for block in child_table:
            block.ref_cout += 1
        self._block_table[child.seq_id] = child_table

    def cache_sequence(
        self,
        seq: Sequence,
        token_limit: Optional[int] = None,
    ):
        if self.prefix_cache is None or seq.seq_id not in self._block_table:
            return None

        token_ids = self._token_ids(seq)
        if token_limit is not None:
            token_ids = token_ids[:token_limit]
        return self.prefix_cache.insert(
            token_ids=token_ids,
            blocks=self._block_table[seq.seq_id],
        )

    def match_cached_prefix(
        self,
        seq_or_token_ids: Union[Sequence, List[int]],
    ) -> Optional[PrefixCacheMatch]:
        if self.prefix_cache is None:
            return None

        token_ids = (
            self._token_ids(seq_or_token_ids)
            if isinstance(seq_or_token_ids, Sequence)
            else seq_or_token_ids
        )
        return self.prefix_cache.match(token_ids)

    def clear_prefix_cache(self):
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def free(
        self,
        seq:Sequence,
        cache: bool = False,
        cache_token_limit: Optional[int] = None,
    ):
        if seq.seq_id not in self._block_table:
            return
        if cache:
            self.cache_sequence(seq, token_limit=cache_token_limit)
        for block in self._block_table.pop(seq.seq_id):
            self.allocator.free(block)
        seq.status = SequenceStatus.FINISHED_ABORTED
    
    def get_block_table(self, seq:Sequence):
        return [b.block_num for b in self._block_table[seq.seq_id]]

    def _token_ids(self, seq: Sequence) -> List[int]:
        return [
            token_id
            for block in seq.logical_token_blocks
            for token_id in block.token_ids
        ]

    @staticmethod
    def _retain_block(block: PhysicalBlock):
        assert block.ref_cout > 0
        block.ref_cout += 1

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    @property
    def num_cached_blocks(self) -> int:
        if self.prefix_cache is None:
            return 0
        return self.prefix_cache.num_cached_blocks
