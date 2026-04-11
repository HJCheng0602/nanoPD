import torch
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from block_manager.block_manager import BlockSpaceManager
from engine.model_runner import ModelRunner, top_k_sample
from workers.kv_transfer import PinnedKVBuffer, extract_kv_to_pinned, _check_p2p

class PrefillWorker:
    def __init__(self, model_path:str, gpu_id:int, block_manager:BlockSpaceManager, block_size:int=16, max_blocks:int=512):
        self.device = f"cuda:{gpu_id}"
        self.block_manager = block_manager
        self.block_size = block_size
        self.runner = ModelRunner(model_path=model_path, device=self.device, max_blocks=max_blocks, block_size=block_size)
        self._busy = False

    def prefill(self, group: SequenceGroup) -> Tuple[int, List[int]]:
        first_tokens, block_tables = self.prefill_batch([group])
        return first_tokens[0], block_tables[0]

    def prefill_and_extract(self, group: SequenceGroup):
        results = self.prefill_batch_and_extract([group])
        return results[0]   # (first_token, block_table, buf, k_cache, v_cache)
    

    def prefill_batch(
        self, groups: List[SequenceGroup]
    ) -> Tuple[List[int], List[List[int]]]:
        self._busy = True
        try:
            all_token_ids: List[int] = []
            all_positions: List[int] = []
            prefills_ctx: List[dict] = []

            for group in groups:
                assert self.block_manager.can_allocate(group), "no free blocks for prefill"
                self.block_manager.allocate(group)

                seq = group.get_seqs(SequenceStatus.RUNNING)[0]
                token_ids = [t for b in seq.logical_token_blocks for t in b.token_ids]
                n = len(token_ids)
                block_table = self.block_manager.get_block_table(seq)

                prefills_ctx.append({
                    "block_table":    block_table,
                    "start_position": 0,
                    "num_tokens":     n,
                })
                all_token_ids.extend(token_ids)
                all_positions.extend(range(n))   

            total = len(all_token_ids)
            input_ids  = torch.tensor([all_token_ids], dtype=torch.long,  device=self.device)
            position_ids = torch.tensor([all_positions], dtype=torch.long, device=self.device)

            self.runner._current_context = {
                "num_prefill_tokens": total,
                "num_decode_tokens":  0,
                "prefills":           prefills_ctx,
                "decodes":            [],
            }

            with torch.no_grad():
                logits = self.runner.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits  

            first_tokens: List[int] = []
            block_tables: List[List[int]] = []
            offset = 0
            for group, pinfo in zip(groups, prefills_ctx):
                seq = group.get_seqs(SequenceStatus.RUNNING)[0]
                n = pinfo['num_tokens']
                next_token = top_k_sample(logits[0, offset + n - 1, :], top_k=10)
                seq.num_computed_tokens = n
                seq.output_token_ids.append(next_token.item())
                first_tokens.append(next_token.item())
                block_tables.append(pinfo['block_table'])
                offset += n

            return first_tokens, block_tables

        finally:
            self._busy = False


    def prefill_batch_and_extract(
        self, groups: List[SequenceGroup]
    ) -> List[Tuple[int, List[int], "PinnedKVBuffer", torch.Tensor, torch.Tensor]]:
        """
        Prefill a batch of requests, then extract KV into pinned buffers one by one.

        Returns a list with one entry per request:
            (first_token, block_table, pinned_buf, k_cache, v_cache)
        """
        first_tokens, block_tables = self.prefill_batch(groups)

        results = []
        for first_token, block_table in zip(first_tokens, block_tables):
            buf = PinnedKVBuffer.from_runner(self.runner, num_blocks=len(block_table))
            extract_kv_to_pinned(
                self.runner.k_cache, self.runner.v_cache, block_table, buf
            )
            results.append((first_token, block_table, buf, self.runner.k_cache, self.runner.v_cache))

        torch.cuda.synchronize(self.device)
        return results
    
    def extract_kv(self, block_table:List[int]) -> PinnedKVBuffer:
        buf = PinnedKVBuffer.from_runner(self.runner, num_blocks=len(block_table))
        extract_kv_to_pinned(self.runner.k_cache, self.runner.v_cache, block_table, buf)
        torch.cuda.synchronize(self.device)
        return buf
    
    
