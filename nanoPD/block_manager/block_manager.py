from dataclasses import field, dataclass
from typing import Dict, List, Union, Optional, Tuple
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus

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
    def __init__(self, block_size: int, num_gpu_blocks:int):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_gpu_blocks)
        self._block_table: Dict[int, List[PhysicalBlock]] = {}

    def can_allocate(self, seq_group:SequenceGroup) -> bool:
        seq = seq_group.get_seqs()[0]           # usually there is only one seq in a group when prefill
        return self.allocator.num_free_blocks >= seq.num_logic_blocks
    def allocate(self, seq_group:SequenceGroup):
        for seq in seq_group.get_seqs(SequenceStatus.WAITING):
            block_table:List[PhysicalBlock] = []
            for _ in range(seq.num_logic_blocks):
                block_table.append(self.allocator.allocate())
            self._block_table[seq.seq_id] = block_table
            seq.status = SequenceStatus.RUNNING

    
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
        next_position = seq.num_computed_tokens
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

    def free(self, seq:Sequence):
        if seq.seq_id not in self._block_table:
            return
        for block in self._block_table.pop(seq.seq_id):
            self.allocator.free(block)
        seq.status = SequenceStatus.FINISHED_ABORTED
    
    def get_block_table(self, seq:Sequence):
        return [b.block_num for b in self._block_table[seq.seq_id]]
    
    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks
    


        