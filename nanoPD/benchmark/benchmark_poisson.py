"""
benchmark_poisson.py — Poisson arrival process benchmark

Usage:
    python benchmark/benchmark_poisson.py --strategy collocated    --arrival-rate 0.1 --duration 60
    python benchmark/benchmark_poisson.py --strategy disaggregated --arrival-rate 0.1 --duration 60
    python benchmark/benchmark_poisson.py --strategy adaptive      --arrival-rate 0.1 --duration 60

Suggested: start arrival-rate at 0.1 and increase until saturation (each request ~6-7s; 0.1 rps ≈ 70% utilisation)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import json
import time
import random
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from block_manager.block_manager import BlockSpaceManager
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from workers.collocated_worker import CollocatedWorker
from workers.prefill_worker import PrefillWorker
from workers.decode_worker import DecodeWorker
from router.central_scheduler import CentralScheduler


# ── workload ──────────────────────────────────────────────────────────────────

def _rand_prompt(tokenizer, target_len: int, vocab_size: int = 5000) -> str:
    ids = [random.randint(100, vocab_size) for _ in range(target_len)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def make_request(tokenizer, kind: str, rng: random.Random) -> Tuple[str, int]:
    if kind == "short":
        p_len = rng.randint(64, 256)
    elif kind == "long":
        p_len = rng.randint(1024, 2048)
    else:
        p_len = rng.randint(64, 256) if rng.random() < 0.5 else rng.randint(1024, 2048)
    return _rand_prompt(tokenizer, p_len), 128


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class ReqResult:
    request_id: int
    arrive_time: float
    finish_time: float
    prompt_len: int
    output_len: int
    path: str
    e2e_ms: float
    queue_wait_ms: float


@dataclass
class PoissonBenchResult:
    strategy: str
    workload: str
    arrival_rate: float
    duration_s: float
    n_completed: int
    n_dropped: int
    throughput_rps: float
    throughput_tokens_per_s: float
    p50_e2e_ms: float
    p95_e2e_ms: float
    p99_e2e_ms: float
    p50_queue_ms: float
    p99_queue_ms: float
    requests: List[ReqResult] = field(default_factory=list)


def _make_result(strategy, workload, arrival_rate, duration, results, n_dropped=0):
    if not results:
        return PoissonBenchResult(
            strategy=strategy, workload=workload,
            arrival_rate=arrival_rate, duration_s=duration,
            n_completed=0, n_dropped=n_dropped,
            throughput_rps=0, throughput_tokens_per_s=0,
            p50_e2e_ms=0, p95_e2e_ms=0, p99_e2e_ms=0,
            p50_queue_ms=0, p99_queue_ms=0,
        )
    e2e = [r.e2e_ms for r in results]
    q   = [r.queue_wait_ms for r in results]
    total_tokens = sum(r.output_len for r in results)
    return PoissonBenchResult(
        strategy=strategy, workload=workload,
        arrival_rate=arrival_rate, duration_s=duration,
        n_completed=len(results), n_dropped=n_dropped,
        throughput_rps=len(results) / duration,
        throughput_tokens_per_s=total_tokens / duration,
        p50_e2e_ms=float(np.percentile(e2e, 50)),
        p95_e2e_ms=float(np.percentile(e2e, 95)),
        p99_e2e_ms=float(np.percentile(e2e, 99)),
        p50_queue_ms=float(np.percentile(q, 50)),
        p99_queue_ms=float(np.percentile(q, 99)),
        requests=results,
    )


# ── collocated ────────────────────────────────────────────────────────────────

def run_poisson_collocated(
    model_path, gpu_id, arrival_rate, duration, workload,
    block_size, max_blocks, warmup_s=10.0, drain_timeout=30.0, seed=42,
):
    print(f"\n[poisson/collocated] GPU={gpu_id} rate={arrival_rate}rps duration={duration}s")
    rng = random.Random(seed)
    cw = CollocatedWorker(model_path, gpu_id=gpu_id,
                          block_size=block_size, max_blocks=max_blocks)
    tokenizer = cw.engine.runner.tokenizer

    print(f"  warming up {warmup_s}s ...")
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warmup_s:
        prompt, o_len = make_request(tokenizer, workload, rng)
        cw.engine.generate(prompt, max_new_tokens=o_len)
    cw.engine.scheduler.finished.clear()
    cw.engine.seq_counter = 0

    pending: Dict[str, Tuple[float, int, int]] = {}
    results: List[ReqResult] = []
    n_dropped = 0

    t_start = time.perf_counter()
    t_next = t_start + rng.expovariate(arrival_rate)

    print(f"  benchmark started ...")
    while True:
        now = time.perf_counter()
        elapsed = now - t_start

        if elapsed >= duration:
            if not pending and not cw.engine.scheduler.running and not cw.engine.scheduler.waiting:
                break
            if elapsed >= duration + drain_timeout:
                n_dropped += len(pending)
                print(f"  drain timeout, dropping {len(pending)} requests")
                break

        while now >= t_next and elapsed < duration:
            prompt, o_len = make_request(tokenizer, workload, rng)
            prompt_len = len(tokenizer(prompt).input_ids)
            engine_rid = str(cw.engine.seq_counter)
            cw.engine.add_request(prompt)
            pending[engine_rid] = (now, prompt_len, o_len)
            t_next += rng.expovariate(arrival_rate)

        for group in cw.engine.scheduler.running:
            rid = group.request_id
            if rid not in pending:
                continue
            _, _, o_len = pending[rid]
            seq = group.get_seqs(SequenceStatus.RUNNING)
            if seq and len(seq[0].output_token_ids) >= o_len:
                seq[0].status = SequenceStatus.FINISHED_STOPPED

        cw.step()

        for group in list(cw.engine.scheduler.finished):
            rid = group.request_id
            if rid not in pending:
                continue
            arrive_t, prompt_len, _ = pending.pop(rid)
            finish_t = time.perf_counter()
            seq = group.get_seqs()[0]
            results.append(ReqResult(
                request_id=int(rid),
                arrive_time=arrive_t - t_start,
                finish_time=finish_t - t_start,
                prompt_len=prompt_len,
                output_len=len(seq.output_token_ids),
                path="collocated",
                e2e_ms=(finish_t - arrive_t) * 1000,
                queue_wait_ms=0.0,
            ))
            print(f"  completed rid={rid} prompt={prompt_len} "
                  f"e2e={(finish_t-arrive_t)*1000:.0f}ms  "
                  f"[{len(results)} done, {len(pending)} pending]")

    print(f"  total completed={len(results)} dropped={n_dropped}")
    return _make_result("collocated", workload, arrival_rate, duration, results, n_dropped)


# ── disaggregated ─────────────────────────────────────────────────────────────

def run_poisson_disaggregated(
    model_path, prefill_gpu, decode_gpu, arrival_rate, duration, workload,
    block_size, max_blocks, warmup_s=10.0, drain_timeout=30.0, seed=42,
):
    print(f"\n[poisson/disaggregated] p={prefill_gpu} d={decode_gpu} "
          f"rate={arrival_rate}rps duration={duration}s")
    rng = random.Random(seed)
    shared_bm = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)
    pw = PrefillWorker(model_path, gpu_id=prefill_gpu, block_manager=shared_bm,
                       block_size=block_size, max_blocks=max_blocks)
    dw = DecodeWorker(model_path, gpu_id=decode_gpu, block_manager=shared_bm,
                      block_size=block_size, max_blocks=max_blocks)
    tokenizer = pw.runner.tokenizer
    eos = tokenizer.eos_token_id
    sid_counter = [0]

    def _run_one(prompt, o_len):
        sid = sid_counter[0]; sid_counter[0] += 1
        token_ids = tokenizer(prompt).input_ids
        seq = Sequence(seq_id=sid, prompt_token_ids=token_ids, block_size=block_size)
        group = SequenceGroup(str(sid), [seq])
        # prefill_and_extract delegates to prefill_batch internally
        first_tok, bt, kv_buf, src_k, src_v = pw.prefill_and_extract(group)
        dw.receive_kv_async(group, bt, kv_buf, src_k=src_k, src_v=src_v)
        torch.cuda.synchronize(dw.device)
        generated = [first_tok]
        for _ in range(o_len - 1):
            res = dw.step()
            if not res: break
            _, tok = res[0]
            generated.append(tok)
            if tok == eos: break
        dw.running.clear(); dw.finished.clear()
        shared_bm.free(group.get_seqs()[0])
        return len(generated)

    print(f"  warming up {warmup_s}s ...")
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warmup_s:
        prompt, o_len = make_request(tokenizer, workload, rng)
        _run_one(prompt, o_len)

    req_queue: List[Tuple[float, str, int]] = []
    results: List[ReqResult] = []
    n_dropped = 0
    t_start = time.perf_counter()
    t_next = t_start + rng.expovariate(arrival_rate)

    print(f"  benchmark started ...")
    while True:
        now = time.perf_counter()
        elapsed = now - t_start

        if elapsed >= duration:
            if not req_queue and not dw.running and not dw._pending:
                break
            if elapsed >= duration + drain_timeout:
                n_dropped += len(req_queue)
                print(f"  drain timeout, dropping {len(req_queue)} requests")
                break

        while now >= t_next and elapsed < duration:
            prompt, o_len = make_request(tokenizer, workload, rng)
            req_queue.append((now, prompt, o_len))
            t_next += rng.expovariate(arrival_rate)

        if req_queue and not dw.running and not dw._pending:
            arrive_t, prompt, o_len = req_queue.pop(0)
            prompt_len = len(tokenizer(prompt).input_ids)
            start_t = time.perf_counter()
            output_len = _run_one(prompt, o_len)
            finish_t = time.perf_counter()
            results.append(ReqResult(
                request_id=sid_counter[0] - 1,
                arrive_time=arrive_t - t_start,
                finish_time=finish_t - t_start,
                prompt_len=prompt_len,
                output_len=output_len,
                path="disaggregated",
                e2e_ms=(finish_t - arrive_t) * 1000,
                queue_wait_ms=(start_t - arrive_t) * 1000,
            ))
            print(f"  completed prompt={prompt_len} "
                  f"e2e={(finish_t-arrive_t)*1000:.0f}ms "
                  f"queue={(start_t-arrive_t)*1000:.0f}ms "
                  f"[{len(results)} done, {len(req_queue)} queued]")

    print(f"  total completed={len(results)} dropped={n_dropped}")
    return _make_result("disaggregated", workload, arrival_rate, duration, results, n_dropped)


# ── adaptive ──────────────────────────────────────────────────────────────────

def run_poisson_adaptive(
    model_path, params_path, collocated_gpu, prefill_gpus, decode_gpu,
    arrival_rate, duration, workload, block_size, max_blocks,
    warmup_s=10.0, drain_timeout=30.0, seed=42,
):
    print(f"\n[poisson/adaptive] c={collocated_gpu} p={prefill_gpus} d={decode_gpu} "
          f"rate={arrival_rate}rps duration={duration}s")
    rng = random.Random(seed)
    scheduler = CentralScheduler.build(
        model_path=model_path, params_path=params_path,
        collocated_gpu=collocated_gpu,
        prefill_gpus=prefill_gpus,
        decode_gpu=decode_gpu,
        block_size=block_size, max_blocks=max_blocks,
    )
    tokenizer = scheduler.pw_list[0].runner.tokenizer

    # ── warmup: bring all GPUs to steady state ──────────────────────────────
    print(f"  warming up {warmup_s}s ...")
    t_warm = time.perf_counter()

    # GPU 0 (collocated): run forward passes until warmup_s/2
    device_cw = scheduler.cw.engine.runner.device
    while time.perf_counter() - t_warm < warmup_s / 2:
        prompt, _ = make_request(tokenizer, workload, rng)
        token_ids = tokenizer(prompt).input_ids
        L = len(token_ids)
        input_ids    = torch.tensor([token_ids], dtype=torch.long, device=device_cw)
        position_ids = torch.arange(L, device=device_cw).unsqueeze(0)
        num_blocks   = (L + block_size - 1) // block_size
        scheduler.cw.engine.runner._current_context = {
            "num_prefill_tokens": L,
            "num_decode_tokens":  0,
            "prefills": [{"block_table": list(range(num_blocks)),
                          "start_position": 0, "num_tokens": L}],
            "decodes": [],
        }
        with torch.no_grad():
            scheduler.cw.engine.runner.model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False)
    torch.cuda.synchronize(device_cw)

    # GPU 1/3 (prefill) + GPU 2 (decode): run one full prefill_and_extract + decode
    warmup_sid = 900000
    for pw in scheduler.pw_list:
        prompt, _ = make_request(tokenizer, workload, rng)
        token_ids  = tokenizer(prompt).input_ids
        seq   = Sequence(seq_id=warmup_sid, prompt_token_ids=token_ids, block_size=block_size)
        group = SequenceGroup(str(warmup_sid), [seq])
        warmup_sid += 1
        first_tok, bt, kv_buf, src_k, src_v = pw.prefill_and_extract(group)
        scheduler.dw.receive_kv_async(group, bt, kv_buf, src_k=src_k, src_v=src_v)
        torch.cuda.synchronize(pw.device)
        for _ in range(5):
            scheduler.dw.step()
        torch.cuda.synchronize(scheduler.dw.device)
        # clean up warmup state so it does not pollute the block manager
        scheduler.dw.running.clear()
        scheduler.dw.finished.clear()
        scheduler.dw.block_manager.free(seq)

    torch.cuda.synchronize()

    # ensure scheduler state is clean before benchmark starts
    scheduler._states.clear()
    scheduler._waiting.clear()
    scheduler._finish_time.clear()
    scheduler._req_counter = 0
    print(f"  warmup done, all workers hot")
    # ─────────────────────────────────────────────────────────────────

    arrive_times: Dict[str, float] = {}
    results: List[ReqResult] = []
    n_dropped = 0
    t_start = time.perf_counter()
    t_next  = t_start + rng.expovariate(arrival_rate)

    print(f"  benchmark started ...")
    while True:
        now     = time.perf_counter()
        elapsed = now - t_start

        if elapsed >= duration:
            if scheduler._all_done() and not scheduler._waiting:
                break
            if elapsed >= duration + drain_timeout:
                n_dropped += len([s for s in scheduler._states.values() if not s.finished])
                print(f"  drain timeout, dropping {n_dropped} requests")
                break

        while now >= t_next and elapsed < duration:
            prompt, _ = make_request(tokenizer, workload, rng)
            rid = scheduler.add_request(prompt)
            arrive_times[rid] = now
            t_next += rng.expovariate(arrival_rate)

        scheduler.step()
        scheduler._enforce_max_tokens(128)
        for rid, state in list(scheduler._states.items()):
            if state.finished and rid in arrive_times and rid in scheduler._finish_time:
                arrive_t = arrive_times.pop(rid)
                finish_t = scheduler._finish_time[rid]
                results.append(ReqResult(
                    request_id=int(rid),
                    arrive_time=arrive_t - t_start,
                    finish_time=finish_t - t_start,
                    prompt_len=state.prompt_len,
                    output_len=len(state.output_token_ids),
                    path=state.path,
                    e2e_ms=(finish_t - arrive_t) * 1000,
                    queue_wait_ms=0.0,
                ))
                print(f"  completed rid={rid} path={state.path} "
                      f"prompt={state.prompt_len} "
                      f"e2e={(finish_t-arrive_t)*1000:.0f}ms "
                      f"[{len(results)} done]")

    print(f"  total completed={len(results)} dropped={n_dropped}")
    return _make_result("adaptive", workload, arrival_rate, duration, results, n_dropped)


# ── summary + main ────────────────────────────────────────────────────────────

def print_summary(r: PoissonBenchResult):
    print(f"\n  strategy={r.strategy} workload={r.workload} "
          f"rate={r.arrival_rate}rps duration={r.duration_s}s")
    print(f"  completed={r.n_completed} dropped={r.n_dropped} "
          f"throughput={r.throughput_rps:.2f}rps ({r.throughput_tokens_per_s:.1f} tok/s)")
    print(f"  e2e  p50={r.p50_e2e_ms:.0f}ms  p95={r.p95_e2e_ms:.0f}ms  "
          f"p99={r.p99_e2e_ms:.0f}ms")
    print(f"  queue_wait p50={r.p50_queue_ms:.0f}ms  p99={r.p99_queue_ms:.0f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="Qwen/Qwen3-8B")
    parser.add_argument("--params",          default="cost_model/params_h20.json")
    parser.add_argument("--collocated-gpu",  type=int, default=0)
    parser.add_argument("--prefill-gpus",    type=int, nargs="+", default=[1, 3])  # ← supports multiple
    parser.add_argument("--prefill-gpu",     type=int, default=1)   # for disaggregated path only
    parser.add_argument("--decode-gpu",      type=int, default=2)
    parser.add_argument("--block-size",      type=int, default=16)
    parser.add_argument("--max-blocks",      type=int, default=512)
    parser.add_argument("--strategy",        default="collocated",
                        choices=["collocated", "disaggregated", "adaptive"])
    parser.add_argument("--workload",        default="mixed",
                        choices=["short", "long", "mixed"])
    parser.add_argument("--arrival-rate",    type=float, default=0.1)
    parser.add_argument("--duration",        type=float, default=60.0)
    parser.add_argument("--warmup",          type=float, default=10.0)
    parser.add_argument("--drain-timeout",   type=float, default=30.0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--output",          default="benchmark/results_poisson_h20.json")
    args = parser.parse_args()

    kwargs = dict(
        arrival_rate=args.arrival_rate,
        duration=args.duration,
        workload=args.workload,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        warmup_s=args.warmup,
        drain_timeout=args.drain_timeout,
        seed=args.seed,
    )

    if args.strategy == "collocated":
        result = run_poisson_collocated(args.model, args.collocated_gpu, **kwargs)
    elif args.strategy == "disaggregated":
        # disaggregated path uses a single prefill worker via --prefill-gpu
        result = run_poisson_disaggregated(
            args.model, args.prefill_gpu, args.decode_gpu, **kwargs)
    else:
        result = run_poisson_adaptive(
            args.model, args.params,
            args.collocated_gpu, args.prefill_gpus, args.decode_gpu, **kwargs)

    print_summary(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)

    def to_dict(obj):
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_dict(i) for i in obj]
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        return obj

    key = f"{args.strategy}_{args.workload}_{args.arrival_rate}"
    existing[key] = to_dict(result)

    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {args.output}  (key={key})")