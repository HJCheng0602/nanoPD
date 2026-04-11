# scheduler = CentralScheduler.build(
#         model_path="Qwen/Qwen3-8B",
#         params_path="cost_model/params.json",
#         collocated_gpu=0,
#         prefill_gpu=1,
#         decode_gpu=2,
#     )
#     scheduler.add_request("介绍一下北京大学。")
#     scheduler.add_request("What is the capital of France?")
 
#     results = scheduler.run_until_done(max_new_tokens=200)
#     for rid, text in results.items():
#         print(rid, text)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from block_manager.block_manager import BlockSpaceManager
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from workers.collocated_worker import CollocatedWorker
from workers.prefill_worker import PrefillWorker
from workers.decode_worker import DecodeWorker
from router.router import Router
import time

MAX_DECODE_BATCH = 20
DEBUG = False
_t0 = time.perf_counter()

def _dbg(tag: str, msg: str):
    if DEBUG:
        t = time.perf_counter() - _t0
        thread = threading.current_thread().name
        print(f"[{t:7.3f}s][{thread:20s}][{tag}] {msg}")

@dataclass
class _RequestState:
    group:SequenceGroup
    prompt_len:int
    path:str            # "collocated" | "disaggregated"
    output_token_ids:List[int] = field(default_factory=list)
    finished:bool = False


class CentralScheduler:
    def __init__(
            self,
            collocated_worker:CollocatedWorker,
            prefill_workers:List[PrefillWorker],
            decode_worker:DecodeWorker,
            router:Router,
            block_size:int=16):
        self.cw = collocated_worker
        self.pw_list = prefill_workers
        self.dw = decode_worker
        self.router = router
        self.block_size = block_size

        self._req_counter = 0
        self._states:Dict[str, _RequestState] = {}
        self._waiting:List[Tuple[str, str]] = []
        self._finish_time: Dict[str, float] = {}

        self._prefill_threads: Dict[int, Optional[threading.Thread]] = {
            i: None for i in range(len(self.pw_list))
        }
        self._prefill_done: List[Tuple[str, int, List[int], object, object, object]] = []
        self._prefill_lock = threading.Lock()


        self._step_count = 0
        self._step_times: List[float] = []   # wall-clock time per step() in ms
        self._collocated_times: List[float] = []
        self._disaggregated_times: List[float] = []
        self._flush_times: List[float] = []

        self._state_lock = threading.Lock()
        

    @classmethod
    def build(
        cls,
        model_path: str,
        params_path: str,
        collocated_gpu: int = 0,
        prefill_gpus: List[int] = None,   # ← supports multiple prefill GPUs
        decode_gpu: int = 2,
        block_size: int = 16,
        max_blocks: int = 512,
    ) -> "CentralScheduler":
        if prefill_gpus is None:
            prefill_gpus = [1]

        shared_bm = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)

        cw = CollocatedWorker(
            model_path=model_path, gpu_id=collocated_gpu,
            block_size=block_size, max_blocks=max_blocks,
        )
        pw_list = [
            PrefillWorker(
                model_path=model_path, gpu_id=g,
                block_manager=shared_bm,
                block_size=block_size, max_blocks=max_blocks,
            )
            for g in prefill_gpus
        ]
        dw = DecodeWorker(
            model_path=model_path, gpu_id=decode_gpu,
            block_manager=shared_bm,
            block_size=block_size, max_blocks=max_blocks,
        )
        router = Router.from_params(params_path=params_path)
        return cls(cw, pw_list, dw, router, block_size=block_size)


    def add_request(self, prompt: str) -> str:
        rid = str(self._req_counter)
        self._req_counter += 1
        self._waiting.append((rid, prompt))
        return rid

    def step(self):
        self._dispatch_waiting()
        self._flush_prefill_done()

        t_coll   = threading.Thread(target=self._step_collocated,   daemon=True)
        t_disagg = threading.Thread(target=self._step_disaggregated, daemon=True)
        t_coll.start();   t_disagg.start()
        t_coll.join();    t_disagg.join()

        nothing_running = (
            not self._waiting
            and len(self.cw.engine.scheduler.running) == 0
            and len(self.cw.engine.scheduler.waiting) == 0
            and len(self.dw.running) == 0
            and len(self.dw._pending) == 0
            and not any(t and t.is_alive() for t in self._prefill_threads.values())
        )
        if nothing_running:
            time.sleep(0.005)
    def run_until_done(self, max_new_tokens: int = 200) -> Dict[str, str]:
        while not self._all_done():
            self.step()
            self._enforce_max_tokens(max_new_tokens)

        for rid, state in self._states.items():
            self.router.update(state.prompt_len, len(state.output_token_ids))

        tokenizer = self.pw_list[0].runner.tokenizer
        return {
            rid: tokenizer.decode(state.output_token_ids, skip_special_tokens=True)
            for rid, state in self._states.items()
        }
    
    def _print_timing_summary(self):
        if not self._step_times:
            return
        import statistics
        def fmt(lst):
            if not lst:
                return "n/a"
            return (f"mean={statistics.mean(lst):.1f}ms "
                    f"p50={sorted(lst)[len(lst)//2]:.1f}ms "
                    f"max={max(lst):.1f}ms")
        print("\n" + "="*60)
        print(f"[TIMING SUMMARY] total steps={self._step_count}")
        print(f"  step total : {fmt(self._step_times)}")
        print(f"  coll step  : {fmt(self._collocated_times)}")
        print(f"  disagg step: {fmt(self._disaggregated_times)}")
        print(f"  flush      : {fmt(self._flush_times)}")
        # compute collocated fraction of total step time
        mean_step = statistics.mean(self._step_times)
        mean_coll = statistics.mean(self._collocated_times)
        print(f"  coll/step  : {mean_coll/mean_step*100:.1f}%  ← high value means GPU0 is blocking the main loop")
        print("="*60 + "\n")


    def _pick_idle_worker(self) -> Optional[Tuple[int, PrefillWorker]]:
        for i, pw in enumerate(self.pw_list):
            t = self._prefill_threads[i]
            if t is None or not t.is_alive():
                return i, pw
        return None

    def _requeue(self, disaggregated_groups: List[Tuple[str, SequenceGroup]]):
        for rid, group in disaggregated_groups:
            seq = group.get_seqs()[0]
            prompt_str = self.pw_list[0].runner.tokenizer.decode(
                [t for b in seq.logical_token_blocks for t in b.token_ids],
                skip_special_tokens=False,
            )
            self._waiting.append((rid, prompt_str))
            with self._state_lock:
                self._states.pop(rid, None)

    def _dispatch_waiting(self):
        if not self._waiting:
            return

        system_load = (
            len(self.cw.engine.scheduler.running)
            + len(self.dw.running)
            + len(self.dw._pending)
        )
        decode_batch_size = len(self.dw.running) + len(self.dw._pending)

        disaggregated_groups: List[Tuple[str, SequenceGroup]] = []

        for rid, prompt in self._waiting:
            token_ids = self.pw_list[0].runner.tokenizer(prompt).input_ids
            prompt_len = len(token_ids)

            # force collocated when decode worker is overloaded
            if decode_batch_size >= MAX_DECODE_BATCH:
                path = "collocated"
            else:
                path = self.router.route(prompt_len, system_load,
                                        decode_batch_size=decode_batch_size)

            seq = Sequence(seq_id=int(rid), prompt_token_ids=token_ids, block_size=self.block_size)
            group = SequenceGroup(rid, [seq])
            state = _RequestState(group=group, prompt_len=prompt_len, path=path)
            self._states[rid] = state

            if path == "collocated":
                self.cw.engine.add_request(prompt=prompt, request_id=rid)
                system_load += 1
            else:
                disaggregated_groups.append((rid, group))
                system_load += 1
                decode_batch_size += 1   # optimistic estimate: this request will join decode

        self._waiting.clear()

        if not disaggregated_groups:
            return

        slot = self._pick_idle_worker()
        if slot is None:
            _dbg("DISPATCH", f"all prefill workers busy, re-queuing {len(disaggregated_groups)} requests")
            self._requeue(disaggregated_groups)
            return

        # not enough blocks: requeue and wait for decode to release blocks
        total_blocks_needed = sum(
            len(g.get_seqs()[0].logical_token_blocks)
            for _, g in disaggregated_groups
        )
        if self.dw.block_manager.num_free_blocks < total_blocks_needed + 16:
            _dbg("DISPATCH", f"not enough blocks (free={self.dw.block_manager.num_free_blocks} "
                            f"needed={total_blocks_needed}), re-queuing {len(disaggregated_groups)} requests")
            self._requeue(disaggregated_groups)
            return

        worker_idx, pw = slot
        groups_only = [g for _, g in disaggregated_groups]
        rids = [rid for rid, _ in disaggregated_groups]
        _dbg("DISPATCH", f"firing prefill worker[{worker_idx}] for rids={rids} batch_size={len(rids)}")

        def _prefill_task(pw=pw, groups=groups_only, rids=rids, widx=worker_idx):
            try:
                batch_results = pw.prefill_batch_and_extract(groups)
                with self._prefill_lock:
                    for rid, (first_token, block_table, kv_buf, src_k, src_v) in zip(rids, batch_results):
                        self._prefill_done.append(
                            (rid, first_token, block_table, kv_buf, src_k, src_v)
                        )
            except AssertionError:
                print(f"[PrefillWorker{widx}] OOM, re-queuing rids={rids}")
                self._requeue(list(zip(rids, groups)))
            except Exception as e:
                print(f"[PrefillWorker{widx}] ERROR: {e}")
                import traceback; traceback.print_exc()

        t = threading.Thread(target=_prefill_task, daemon=True, name=f"PrefillW{worker_idx}")
        self._prefill_threads[worker_idx] = t
        t.start()

    def _flush_prefill_done(self):
        with self._prefill_lock:
            done, self._prefill_done = self._prefill_done, []

        if done:
            _dbg("FLUSH", f"flushing {len(done)} prefill results into decode worker")

        for rid, first_token, block_table, kv_buf, src_k, src_v in done:
            if rid not in self._states:
                continue
            state = self._states[rid]
            state.output_token_ids.append(first_token)
            self.dw.receive_kv_async(
                state.group, block_table, kv_buf, src_k=src_k, src_v=src_v
            )

    def _step_collocated(self):
        t0 = time.perf_counter()
        self.cw.step()
        dt = (time.perf_counter() - t0) * 1000
        if dt > 200:
            _dbg("COLL", f"cw.step() took {dt:.0f}ms running={len(self.cw.engine.scheduler.running)}")

        for group in list(self.cw.engine.scheduler.finished):
            rid = group.request_id
            with self._state_lock:
                if rid not in self._states:
                    continue
                state = self._states[rid]
                if state.finished:
                    continue
                seq = group.get_seqs()[0]
                state.output_token_ids = list(seq.output_token_ids)
                state.finished = True
                self._finish_time[rid] = time.perf_counter()
            _dbg("COLL", f"rid={rid} finished output_len={len(state.output_token_ids)}")

    def _step_disaggregated(self):
        try:
            results = self.dw.step()
        except MemoryError:
            # OOM: evict the oldest sequence from dw.running to free its blocks
            if self.dw.running:
                victim = self.dw.running[0]
                seq = victim.get_seqs(SequenceStatus.RUNNING)[0]
                seq.status = SequenceStatus.FINISHED_STOPPED
                self.dw.running.remove(victim)
                self.dw.finished.append(victim)
                self.dw.block_manager.free(seq)
                rid = victim.request_id
                with self._state_lock:
                    if rid in self._states:
                        self._states[rid].finished = True
                        self._finish_time[rid] = time.perf_counter()
                print(f"[OOM] evicted rid={rid} to free blocks")
            return

        eos = self.dw.runner.tokenizer.eos_token_id
        for group, tok_id in results:
            rid = group.request_id
            with self._state_lock:
                if rid not in self._states:
                    continue
                state = self._states[rid]
                state.output_token_ids.append(tok_id)
                if tok_id == eos:
                    state.finished = True
                    self._finish_time[rid] = time.perf_counter()
    def _enforce_max_tokens(self, max_new_tokens: int):
        for rid, state in self._states.items():
            if state.finished:
                continue
            if state.path == "collocated":
                for group in self.cw.engine.scheduler.running:
                    if group.request_id == rid:
                        seqs = group.get_seqs(SequenceStatus.RUNNING)
                        if not seqs:
                            break
                        seq = seqs[0]
                        if len(seq.output_token_ids) >= max_new_tokens:
                            state.output_token_ids = list(seq.output_token_ids)
                            seq.status = SequenceStatus.FINISHED_STOPPED
                            state.finished = True
                            self._finish_time[rid] = time.perf_counter()
                        break
            else:
                if len(state.output_token_ids) >= max_new_tokens:
                    state.finished = True
                    self._finish_time[rid] = time.perf_counter()
                    for group in self.dw.running:
                        if group.request_id == rid:
                            seq = group.get_seqs(SequenceStatus.RUNNING)[0]
                            seq.status = SequenceStatus.FINISHED_STOPPED
                            self.dw.running.remove(group)
                            self.dw.finished.append(group)
                            self.dw.block_manager.free(seq)
                            break

    def _all_done(self) -> bool:
        if self._waiting:
            return False
        if not self._states:
            return False
        if any(t is not None and t.is_alive() for t in self._prefill_threads.values()):
            return False
        with self._prefill_lock:
            if self._prefill_done:
                return False
        return all(s.finished for s in self._states.values())

    def stats(self) -> dict:
        return {
            "router":           self.router.decision_stats(),
            "predictor":        self.router.predictor.stats(),
            "pending_transfer": len(self.dw._pending),
            "decode_running":   len(self.dw.running),
            "prefill_workers":  len(self.pw_list),
            "prefill_threads_busy": sum(
                1 for t in self._prefill_threads.values()
                if t is not None and t.is_alive()
            ),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="Qwen/Qwen3-8B")
    parser.add_argument("--params",         default="cost_model/params.json")
    parser.add_argument("--collocated-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpus",   type=int, nargs="+", default=[1, 3])
    parser.add_argument("--decode-gpu",     type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    args = parser.parse_args()

    scheduler = CentralScheduler.build(
        model_path=args.model,
        params_path=args.params,
        collocated_gpu=args.collocated_gpu,
        prefill_gpus=args.prefill_gpus,
        decode_gpu=args.decode_gpu,
    )

    prompts = [
        "介绍一下北京大学。",
        "What is the capital of France?",
        "用一句话解释量子纠缠。",
        "Tell me a short joke.",
    ]
    for p in prompts:
        scheduler.add_request(p)

    print("Running...")
    results = scheduler.run_until_done(max_new_tokens=args.max_new_tokens)

    print(f"\n{'='*60}")
    for rid, text in results.items():
        state = scheduler._states[rid]
        print(f"[{rid}] path={state.path} prompt_len={state.prompt_len}")
        print(f"      {text}\n")

    print(scheduler.stats())
