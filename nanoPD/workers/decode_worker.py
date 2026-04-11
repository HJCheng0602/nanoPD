import torch
from typing import List, Tuple, Optional

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from block_manager.sequence import SequenceGroup, SequenceStatus
from block_manager.block_manager import BlockSpaceManager
from engine.model_runner import ModelRunner, top_k_sample
from workers.kv_transfer import PinnedKVBuffer, load_kv_from_pinned, transfer_kv

class _PendingTransfer:
    __slots__ = ("group", "event")
    def __init__(self, group:SequenceGroup, event:torch.cuda.Event):
        self.group = group
        self.event = event


class DecodeWorker:
    def __init__(self, model_path:str, gpu_id:int, block_manager:BlockSpaceManager, block_size:int=16, max_blocks:int=512):
        self.device = f"cuda:{gpu_id}"
        self.block_manager = block_manager
        self.block_size = block_size
        self.runner = ModelRunner(
            model_path=model_path,
            device=self.device,
            block_size=block_size,
            max_blocks=max_blocks
        )

        self.running: List[SequenceGroup] = []
        self.finished: List[SequenceGroup] = []

        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._pending: List[_PendingTransfer] = []

    def receive_kv_async(
        self,
        group: SequenceGroup,
        block_table: List[int],
        buf: PinnedKVBuffer,
        src_k: torch.Tensor = None,   # k_cache from the prefill worker
        src_v: torch.Tensor = None,
    ):
        path = transfer_kv(
            src_k=src_k,
            src_v=src_v,
            dst_k=self.runner.k_cache,
            dst_v=self.runner.v_cache,
            block_table=block_table,
            stream=self.transfer_stream,
            buf=buf,
        )
        event = torch.cuda.Event()
        self.transfer_stream.record_event(event=event)
        self._pending.append(_PendingTransfer(group, event))
        return path   # returned for caller logging
    
    def _promote_ready(self):
        still_pending = []
        for pt in self._pending:
            if pt.event.query():
                self.running.append(pt.group)
            else:
                still_pending.append(pt)
        self._pending = still_pending

    def step(self) -> List[Tuple[SequenceGroup, int]]:
        self._promote_ready()
        if not self.running:
            return []

        input_ids_list, position_list, decodes_ctx = [], [], []

        for group in self.running:
            seq = group.get_seqs(SequenceStatus.RUNNING)[0]
            self.block_manager.append_slot(seq)
            block_table = self.block_manager.get_block_table(seq)
            pos = seq.num_computed_tokens
            input_ids_list.append(seq.output_token_ids[-1])
            position_list.append(pos)
            decodes_ctx.append({"block_table":block_table, "position":pos})

        B = len(self.running)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([position_list], dtype=torch.long, device=self.device)

        self.runner._current_context = {
            "num_prefill_tokens":0,
            "num_decode_tokens":B,
            "prefills":[],
            "decodes":decodes_ctx
        }

        with torch.cuda.stream(self.compute_stream):
            with torch.no_grad():
                logits = self.runner.model(
                    input_ids = input_ids, position_ids = position_ids
                ).logits

        torch.cuda.current_stream(self.device).wait_stream(self.compute_stream)

        results = []
        newly_finished = []

        for i, group in enumerate(self.running):
            seq = group.get_seqs(SequenceStatus.RUNNING)[0]
            next_tok = top_k_sample(logits[0, i, :], top_k=10)
            tok_id = next_tok.item()              # single .item() call to sync once
            seq.output_token_ids.append(tok_id)
            seq.num_computed_tokens += 1
            results.append((group, tok_id))

            if tok_id == self.runner.tokenizer.eos_token_id:
                seq.status = SequenceStatus.FINISHED_STOPPED
                newly_finished.append(group)
                self.finished.append(group)
        for g in newly_finished:
            self.running.remove(g)
            seq = g.get_seqs()[0]
            self.block_manager.free(seq)
        if newly_finished:
            self._promote_ready()
        return results
        
        