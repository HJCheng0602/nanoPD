import sys
import os

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from block_manager.block_manager import BlockSpaceManager
from engine.model_runner import ModelRunner, top_k_sample
from engine.scheduler import Scheduler, SchedulerOutput


class Engine:
    def __init__(self,model_path:str ,block_size:int=16, max_blocks:int=512, device:str="cuda"):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.runner = ModelRunner(model_path, block_size=block_size, max_blocks=max_blocks, device=device)
        self.block_manager = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)
        self.seq_counter = 0
        self.scheduler = Scheduler(self.block_manager, max_batch_size=16, budget=1024)
    
    def add_request(self, prompt: str, request_id: str = None):
        token_ids = self.runner.tokenizer(prompt).input_ids
        seq = Sequence(
            seq_id=self.seq_counter,
            prompt_token_ids=token_ids,
            block_size=self.block_size
        )
        self.seq_counter += 1
        actual_rid = request_id if request_id is not None else str(seq.seq_id)
        group = SequenceGroup(request_id=actual_rid, seqs=[seq])
        self.scheduler.waiting.append(group)
        return group
    
    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        group = self.add_request(prompt)
        rid = group.request_id
        results = self.run_until_done(max_tokens_per_seq=max_new_tokens)
        return results[rid]

    def step(self):
        sched = self.scheduler.schedule()

        input_ids_list = []
        position_list = []
        num_prefill = 0

        if sched.prefill_group:
            tokens = sched.prefill_chunk_tokens
            start = sched.prefill_start_position
            input_ids_list.extend(tokens)
            position_list.extend(range(start, start + len(tokens)))
            num_prefill = len(tokens)

        for group in sched.decode_groups:
            seq = group.get_seqs(SequenceStatus.RUNNING)[0]
            self.block_manager.append_slot(seq)
            input_ids_list.append(seq.output_token_ids[-1])
            position_list.append(seq.num_computed_tokens)

        if num_prefill == 0 and len(sched.decode_groups) == 0:
            return []

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.runner.device)
        position_ids = torch.tensor([position_list], dtype=torch.long, device=self.runner.device)

        ctx = {
            "num_prefill_tokens": num_prefill,
            "num_decode_tokens": len(sched.decode_groups),
            "prefills": [],
            "decodes": [],
        }

        if sched.prefill_group:
            seq = sched.prefill_group.get_seqs()[0]
            ctx["prefills"] = [{
                "block_table":    self.block_manager.get_block_table(seq),
                "start_position": sched.prefill_start_position,
                "num_tokens":     num_prefill,
            }]

        for group in sched.decode_groups:
            seq = group.get_seqs(SequenceStatus.RUNNING)[0]
            ctx["decodes"].append({
                "block_table": self.block_manager.get_block_table(seq),
                "position":    seq.num_computed_tokens,
            })

        self.runner._current_context = ctx

        with torch.no_grad():
            logits = self.runner.model(input_ids=input_ids, position_ids=position_ids).logits

        results = []
        if sched.prefill_group:
            seq = sched.prefill_group.get_seqs()[0]
            seq.num_computed_tokens += num_prefill
            if sched.prefill_is_last:
                tok_id = top_k_sample(logits[0, num_prefill - 1, :]).item()
                seq.append_token_id(tok_id)
                self.scheduler.running.append(sched.prefill_group)

        for i, group in enumerate(sched.decode_groups):
            seq = group.get_seqs(SequenceStatus.RUNNING)[0]
            tok_id = top_k_sample(logits[0, num_prefill + i, :]).item()
            seq.append_token_id(tok_id)
            seq.num_computed_tokens += 1
            results.append((group, tok_id))
            if tok_id == self.runner.tokenizer.eos_token_id:
                seq.status = SequenceStatus.FINISHED_STOPPED

        return results

    def run_until_done(self, max_tokens_per_seq: int = 500) -> dict:
        while self.scheduler.running or self.scheduler.waiting or self.scheduler.prefilling:
            self.step()
            for group in self.scheduler.running:
                seqs = group.get_seqs(SequenceStatus.RUNNING)
                if seqs and len(seqs[0].output_token_ids) >= max_tokens_per_seq:
                    seqs[0].status = SequenceStatus.FINISHED_STOPPED

        results = {}
        for group in self.scheduler.finished:
            seq = group.get_seqs()[0]
            results[group.request_id] = self.runner.tokenizer.decode(
                seq.output_token_ids, skip_special_tokens=True
            )
        return results

