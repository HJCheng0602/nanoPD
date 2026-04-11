import torch
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional

class SequenceStatus(Enum):
    WAITING = auto()    
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_ABORTED = auto()

@dataclass
class LogicalTokenBlock:
    block_num : int
    block_size : int
    token_ids : List[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)
    
    @property
    def is_full(self) -> bool:
        return len(self.token_ids) == self.block_size
    
    @property
    def num_empty_slots(self) -> int:
        return self.block_size - len(self.token_ids)
    
    def append_token(self, token_id:int):
        assert not self.is_full, "block is full"
        self.token_ids.append(token_id)

class Sequence:
    def __init__(self, seq_id:int, prompt_token_ids: List[int], block_size:int):
        self.status = SequenceStatus.WAITING
        self.block_size = block_size
        self.seq_id = seq_id

        self.prompt_len = len(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.logical_token_blocks: List[LogicalTokenBlock] = []

        self.num_computed_tokens = 0

        for token_id in prompt_token_ids:
            self._append_token_id_to_blocks(token_id)

    # here are inner methods
    def _append_new_logical_block(self):
        block = LogicalTokenBlock(
            block_num=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        
        self.logical_token_blocks.append(block)
    
    def _append_token_id_to_blocks(self, token_id : int):
        if not self.logical_token_blocks or self.logical_token_blocks[-1].is_full:
            self._append_new_logical_block()
        self.logical_token_blocks[-1].append_token(token_id)
    # out methods
    def append_token_id(self, token_id: int):
        self._append_token_id_to_blocks(token_id)
        self.output_token_ids.append(token_id)

    @property
    def num_logic_blocks(self) -> int:
        return len(self.logical_token_blocks)
    
    @property
    def last_token_id(self) -> int:
        return self.output_token_ids[-1]
    @property
    def total_len(self) -> int:
        return self.prompt_len + len(self.output_token_ids)
    @property
    def is_finished(self) -> bool:
        return self.status in (
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_STOPPED
        )
    @property
    def is_prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.prompt_len
    
class SequenceGroup:
    """
    usually one request is regarded as a group of sequence when beam search,
    meanwhile we also can employ prefix sharing
    """
    def __init__(self, request_id:str, seqs:List[Sequence]):
        self.request_id = request_id
        self.seqs : List[Sequence] = seqs
    
    def get_seqs(self, status:Optional[SequenceStatus] = None) -> List[Sequence]:
        if status is None:
            return self.seqs
        return [s for s in self.seqs if s.status == status]
    
    @property
    def num_seqs(self) -> int:
        return len(self.seqs)
    @property
    def is_finished(self) -> bool:
        return all(s.is_finished for s in self.seqs)
    

        

