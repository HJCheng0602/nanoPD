"""
Benchmark: Collocated vs Disaggregated vs Adaptive
=====================================================

Usage (from project root):
    python benchmark/run_benchmark.py \
        --model Qwen/Qwen3-8B \
        --params cost_model/params.json \
        --collocated-gpu 0 \
        --prefill-gpu 1 \
        --decode-gpu 2 \
        --output benchmark/results.json

Workloads:
    short   : prompt_len in [64, 256],  output ~128 tokens
    long    : prompt_len in [1024, 4096], output ~128 tokens
    mixed   : 50% short + 50% long  (adaptive should perform best here)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import json
import time
import random
import argparse
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from block_manager.block_manager import BlockSpaceManager
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from workers.collocated_worker import CollocatedWorker
from workers.prefill_worker import PrefillWorker
from workers.decode_worker import DecodeWorker
from router.central_scheduler import CentralScheduler


# ── workload generation ───────────────────────────────────────────────────────

def _rand_prompt(tokenizer, target_len: int, vocab_size: int = 5000) -> str:
    """Generate a random prompt of approximately target_len tokens."""
    ids = [random.randint(100, vocab_size) for _ in range(target_len)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def make_workload(
    tokenizer,
    kind: str,
    n_requests: int,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """
    Returns [(prompt, target_output_len), ...]
    kind: "short" | "long" | "mixed"
    """
    random.seed(seed)
    prompts = []
    for i in range(n_requests):
        if kind == "short":
            p_len = random.randint(64, 256)
            o_len = 128
        elif kind == "long":
            p_len = random.randint(1024, 2048)   # keep within block capacity
            o_len = 128
        else:  # mixed
            if i % 2 == 0:
                p_len = random.randint(64, 256)
            else:
                p_len = random.randint(1024, 2048)
            o_len = 128
        prompt = _rand_prompt(tokenizer, p_len)
        prompts.append((prompt, o_len))
    return prompts


# ── single run ────────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_id: str
    prompt_len: int
    output_len: int
    ttft_ms: float          # time to first token (measured when prefill completes)
    e2e_ms: float           # end-to-end latency
    path: str               # "collocated" | "disaggregated"


@dataclass
class BenchResult:
    strategy: str
    workload: str
    n_requests: int
    total_ms: float
    throughput_tokens_per_s: float
    p50_e2e_ms: float
    p99_e2e_ms: float
    requests: List[RequestResult] = field(default_factory=list)


def run_collocated(
    model_path: str,
    gpu_id: int,
    workload: List[Tuple[str, int]],
    block_size: int,
    max_blocks: int,
    warmup: int = 2,
) -> BenchResult:
    print(f"\n[collocated] GPU={gpu_id}  n={len(workload)}")
    cw = CollocatedWorker(model_path, gpu_id=gpu_id,
                          block_size=block_size, max_blocks=max_blocks)
    tokenizer = cw.engine.runner.tokenizer

    # warmup
    print(f"  warming up ({warmup} requests)...")
    for prompt, o_len in workload[:warmup]:
        cw.engine.generate(prompt, max_new_tokens=o_len)

    results = []
    t_wall_start = time.perf_counter()

    for i, (prompt, o_len) in enumerate(workload):
        prompt_len = len(tokenizer(prompt).input_ids)
        t0 = time.perf_counter()
        cw.engine.add_request(prompt)

        # run until this request finishes
        output_tokens = 0
        ttft_ms = None
        step_count = 0

        while cw.engine.scheduler.waiting or cw.engine.scheduler.running or cw.engine.scheduler.prefilling:
            step_results = cw.step()
            step_count += 1

            if step_results and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000

            for group in cw.engine.scheduler.running:
                seq = group.get_seqs(SequenceStatus.RUNNING)
                if seq and len(seq[0].output_token_ids) >= o_len:
                    seq[0].status = SequenceStatus.FINISHED_STOPPED

        e2e_ms = (time.perf_counter() - t0) * 1000

        # collect output from finished queue
        for group in cw.engine.scheduler.finished:
            seq = group.get_seqs()[0]
            output_tokens = len(seq.output_token_ids)

        results.append(RequestResult(
            request_id=str(i),
            prompt_len=prompt_len,
            output_len=output_tokens,
            ttft_ms=ttft_ms or e2e_ms,
            e2e_ms=e2e_ms,
            path="collocated",
        ))
        print(f"  [{i+1}/{len(workload)}] prompt={prompt_len} output={output_tokens} "
              f"e2e={e2e_ms:.0f}ms")

    total_ms = (time.perf_counter() - t_wall_start) * 1000
    total_tokens = sum(r.output_len for r in results)
    e2e_list = [r.e2e_ms for r in results]

    return BenchResult(
        strategy="collocated",
        workload="",
        n_requests=len(workload),
        total_ms=total_ms,
        throughput_tokens_per_s=total_tokens / (total_ms / 1000),
        p50_e2e_ms=float(np.percentile(e2e_list, 50)),
        p99_e2e_ms=float(np.percentile(e2e_list, 99)),
        requests=results,
    )


def run_disaggregated(
    model_path: str,
    prefill_gpu: int,
    decode_gpu: int,
    workload: List[Tuple[str, int]],
    block_size: int,
    max_blocks: int,
    warmup: int = 2,
) -> BenchResult:
    print(f"\n[disaggregated] prefill={prefill_gpu} decode={decode_gpu}  n={len(workload)}")
    shared_bm = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)
    pw = PrefillWorker(model_path, gpu_id=prefill_gpu, block_manager=shared_bm,
                       block_size=block_size, max_blocks=max_blocks)
    dw = DecodeWorker(model_path, gpu_id=decode_gpu, block_manager=shared_bm,
                      block_size=block_size, max_blocks=max_blocks)
    tokenizer = pw.runner.tokenizer
    eos = tokenizer.eos_token_id

    def _run_one(prompt: str, o_len: int) -> RequestResult:
        prompt_len = len(tokenizer(prompt).input_ids)
        token_ids = tokenizer(prompt).input_ids
        seq = Sequence(seq_id=0, prompt_token_ids=token_ids, block_size=block_size)
        group = SequenceGroup("0", [seq])

        t0 = time.perf_counter()
        first_token, block_table, kv_buf, src_k, src_v = pw.prefill_and_extract(group)
        ttft_ms = (time.perf_counter() - t0) * 1000

        dw.receive_kv_async(group, block_table, kv_buf, src_k=src_k, src_v=src_v)
        torch.cuda.synchronize(dw.device)

        generated = [first_token]
        for _ in range(o_len - 1):
            results = dw.step()
            if not results:
                break
            _, tok = results[0]
            generated.append(tok)
            if tok == eos:
                break

        # reset dw.running / finished between requests
        dw.running.clear()
        dw.finished.clear()  # clear dw state between requests
        shared_bm.free(group.get_seqs()[0])

        e2e_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            request_id="",
            prompt_len=prompt_len,
            output_len=len(generated),
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            path="disaggregated",
        )

    # warmup
    print(f"  warming up ({warmup} requests)...")
    for prompt, o_len in workload[:warmup]:
        _run_one(prompt, o_len)

    results = []
    t_wall_start = time.perf_counter()

    for i, (prompt, o_len) in enumerate(workload):
        r = _run_one(prompt, o_len)
        r.request_id = str(i)
        results.append(r)
        print(f"  [{i+1}/{len(workload)}] prompt={r.prompt_len} output={r.output_len} "
              f"ttft={r.ttft_ms:.0f}ms e2e={r.e2e_ms:.0f}ms")

    total_ms = (time.perf_counter() - t_wall_start) * 1000
    total_tokens = sum(r.output_len for r in results)
    e2e_list = [r.e2e_ms for r in results]

    return BenchResult(
        strategy="disaggregated",
        workload="",
        n_requests=len(workload),
        total_ms=total_ms,
        throughput_tokens_per_s=total_tokens / (total_ms / 1000),
        p50_e2e_ms=float(np.percentile(e2e_list, 50)),
        p99_e2e_ms=float(np.percentile(e2e_list, 99)),
        requests=results,
    )


def run_adaptive(
    model_path: str,
    params_path: str,
    collocated_gpu: int,
    prefill_gpu: int,
    decode_gpu: int,
    workload: List[Tuple[str, int]],
    block_size: int,
    max_blocks: int,
    warmup: int = 2,
) -> BenchResult:
    print(f"\n[adaptive] c={collocated_gpu} p={prefill_gpu} d={decode_gpu}  n={len(workload)}")
    scheduler = CentralScheduler.build(
        model_path=model_path,
        params_path=params_path,
        collocated_gpu=collocated_gpu,
        prefill_gpus=[prefill_gpu],
        decode_gpu=decode_gpu,
        block_size=block_size,
        max_blocks=max_blocks,
    )
    tokenizer = scheduler.pw_list[0].runner.tokenizer

    print(f"  warming up ({warmup} requests)...")
    for prompt, o_len in workload[:warmup]:
        token_ids = tokenizer(prompt).input_ids
        L = len(token_ids)
        device = scheduler.cw.engine.runner.device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        position_ids = torch.arange(L, device=device).unsqueeze(0)
        num_blocks = (L + block_size - 1) // block_size
        scheduler.cw.engine.runner._current_context = {
            "num_prefill_tokens": L,
            "num_decode_tokens": 0,
            "prefills": [{
                "block_table":    list(range(num_blocks)),
                "start_position": 0,
                "num_tokens":     L,
            }],
            "decodes": []
        }
        with torch.no_grad():
            scheduler.cw.engine.runner.model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            )
    torch.cuda.synchronize()
    results = []
    t_wall_start = time.perf_counter()

    req_t0 = {}
    for i, (prompt, _) in enumerate(workload):
        rid = scheduler.add_request(prompt)
        req_t0[rid] = time.perf_counter()

    max_o_len = max(o for _, o in workload)
    final_texts = scheduler.run_until_done(max_new_tokens=max_o_len)
    # for rid in scheduler._states:
    #     finish_t = scheduler._finish_time.get(rid, None)
    #     start_t = req_t0.get(rid, None)
    #     diff = (finish_t - start_t) * 1000 if finish_t and start_t else "N/A"
    #     print(f"  debug rid={rid} finish={finish_t:.3f} start={start_t:.3f} diff={diff}")

    total_ms = (time.perf_counter() - t_wall_start) * 1000

    for rid, state in scheduler._states.items():
        finish_t = scheduler._finish_time.get(rid, time.perf_counter())
        e2e_ms = (finish_t - req_t0[rid]) * 1000
        results.append(RequestResult(
            request_id=rid,
            prompt_len=state.prompt_len,
            output_len=len(state.output_token_ids),
            ttft_ms=0.0,   # TTFT not instrumented for the adaptive path
            e2e_ms=e2e_ms,
            path=state.path,
        ))
        print(f"  [{rid}] path={state.path} prompt={state.prompt_len} "
              f"output={len(state.output_token_ids)} e2e={e2e_ms:.0f}ms")

    total_tokens = sum(r.output_len for r in results)
    e2e_list = [r.e2e_ms for r in results]

    return BenchResult(
        strategy="adaptive",
        workload="",
        n_requests=len(workload),
        total_ms=total_ms,
        throughput_tokens_per_s=total_tokens / (total_ms / 1000),
        p50_e2e_ms=float(np.percentile(e2e_list, 50)),
        p99_e2e_ms=float(np.percentile(e2e_list, 99)),
        requests=results,
    )


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(results: Dict[str, Dict[str, BenchResult]]):
    print(f"\n{'='*70}")
    print(f"{'BENCHMARK SUMMARY':^70}")
    print(f"{'='*70}")
    header = f"{'workload':<10} {'strategy':<16} {'p50_e2e':>10} {'p99_e2e':>10} {'tok/s':>10}"
    print(header)
    print('-' * 70)
    for wl in ["short", "long", "mixed"]:
        for strategy in ["collocated", "disaggregated", "adaptive"]:
            r = results.get(wl, {}).get(strategy)
            if r is None:
                continue
            print(f"{wl:<10} {strategy:<16} "
                  f"{r.p50_e2e_ms:>9.0f}ms "
                  f"{r.p99_e2e_ms:>9.0f}ms "
                  f"{r.throughput_tokens_per_s:>9.1f}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--params", default="cost_model/params_h20.json")
    parser.add_argument("--collocated-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpu", type=int, default=1)
    parser.add_argument("--decode-gpu", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-blocks", type=int, default=512)
    parser.add_argument("--n-requests", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workloads", nargs="+",
                        default=["short", "long", "mixed"],
                        choices=["short", "long", "mixed"])
    parser.add_argument("--strategies", nargs="+",
                        default=["collocated", "disaggregated", "adaptive"],
                        choices=["collocated", "disaggregated", "adaptive"])
    parser.add_argument("--output", default="benchmark/results_h20.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # use the collocated worker's tokenizer to build workloads (avoids loading an extra model)
    print("Building workloads...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    workloads = {
        wl: make_workload(tokenizer, wl, args.n_requests, seed=args.seed)
        for wl in args.workloads
    }

    all_results: Dict[str, Dict[str, BenchResult]] = {wl: {} for wl in args.workloads}

    for wl_name, wl_data in workloads.items():
        print(f"\n{'='*60}")
        print(f"Workload: {wl_name}  ({len(wl_data)} requests)")
        print(f"{'='*60}")

        if "collocated" in args.strategies:
            r = run_collocated(
                args.model, args.collocated_gpu, wl_data,
                args.block_size, args.max_blocks, args.warmup,
            )
            r.workload = wl_name
            all_results[wl_name]["collocated"] = r

        if "disaggregated" in args.strategies:
            r = run_disaggregated(
                args.model, args.prefill_gpu, args.decode_gpu, wl_data,
                args.block_size, args.max_blocks, args.warmup,
            )
            r.workload = wl_name
            all_results[wl_name]["disaggregated"] = r

        if "adaptive" in args.strategies:
            r = run_adaptive(
                args.model, args.params,
                args.collocated_gpu, args.prefill_gpu, args.decode_gpu,
                wl_data, args.block_size, args.max_blocks, args.warmup,
            )
            r.workload = wl_name
            all_results[wl_name]["adaptive"] = r

    print_summary(all_results)

    # save results to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def to_dict(obj):
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_dict(i) for i in obj]
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        return obj

    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)

    new_data = to_dict(all_results)
    for wl, strategies in new_data.items():
        if wl not in existing:
            existing[wl] = {}
        existing[wl].update(strategies)

    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {args.output}")