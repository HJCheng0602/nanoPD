import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from typing import List, Optional
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from block_manager.block_manager import BlockAllocator, BlockSpaceManager
from dataclasses import dataclass

@dataclass
class SchedulerOutput:
    prefill_group: Optional[SequenceGroup]
    prefill_chunk_tokens: Optional[torch.tensor]
    prefill_start_position: int
    prefill_is_last: bool
    decode_groups:List[SequenceGroup]


class Scheduler:
    def __init__(self, block_manager:BlockSpaceManager, max_batch_size:int=8, budget=512):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.BUDGET = budget

        self.waiting: List[SequenceGroup] = []
        self.prefilling: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []
        self.finished: List[SequenceGroup] = []
    
    def schedule(self) -> SchedulerOutput:
        # first clean the finished in running
        running = []
        if self.running:
            for seq_group in self.running:
                if seq_group.is_finished:
                    self.finished.append(seq_group)
                    for seq in seq_group.get_seqs():
                        self.block_manager.free(seq)
                else:
                    running.append(seq_group)
        self.running = running

        # second decide who to decode

        decode_tokens = len(self.running)
        prefill_budget = self.BUDGET - decode_tokens

        prefill_group = None
        prefill_chunk_tokens = None
        prefill_start_position = 0
        prefill_is_last = False

        decode_groups = list(self.running)

        # third we decide a sequence whether to prefill

        if prefill_budget <= 0:
            return SchedulerOutput(
                prefill_group=None,
                prefill_chunk_tokens=None,
                prefill_start_position=0,
                prefill_is_last=False,
                decode_groups=decode_groups
            )



        
        if self.prefilling:
            prefill_group = self.prefilling[0]
            seq = prefill_group.get_seqs()[0]
            start = seq.num_computed_tokens
            end = min(start + prefill_budget, seq.prompt_len)
            # print(f"[scheduler] chunked prefill req={prefill_group.request_id} "
            #   f"chunk=[{start}:{end}] / {seq.prompt_len} tokens")
            all_tokens = [t for b in seq.logical_token_blocks for t in b.token_ids]
            prefill_chunk_tokens = all_tokens[start:end]
            prefill_start_position = start
            prefill_is_last = (end >= seq.prompt_len)
            if prefill_is_last:
                self.prefilling.pop(0)

        elif self.waiting:
            candidate = self.waiting[0]
            if (len(self.running) < self.max_batch_size and
                self.block_manager.can_allocate(candidate)):
                prefill_group = self.waiting.pop(0)
                self.block_manager.allocate(prefill_group)
                for seq in prefill_group.get_seqs():
                    seq.status = SequenceStatus.RUNNING
                seq = prefill_group.get_seqs()[0]
                end = min(prefill_budget, seq.prompt_len)
                all_tokens = [t for b in seq.logical_token_blocks for t in b.token_ids]
                prefill_chunk_tokens = all_tokens[:end]
                prefill_start_position = 0
                prefill_is_last = (end >= seq.prompt_len)
            #     print(f"[scheduler] new prefill req={prefill_group.request_id} "
            #   f"prompt_len={seq.prompt_len} chunk=[0:{end}]")
                if not prefill_is_last:
                    self.prefilling.append(prefill_group)
        
        # print(f"[scheduler] schedule result: prefill_group={prefill_group.request_id if prefill_group else None} "
        #   f"decode_groups={[g.request_id for g in decode_groups]}")

        return SchedulerOutput(
            prefill_group=prefill_group,
            prefill_chunk_tokens=prefill_chunk_tokens,
            prefill_start_position=prefill_start_position,
            prefill_is_last=prefill_is_last,
            decode_groups=decode_groups
        )
        
    

