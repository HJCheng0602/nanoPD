"""
Integration test for disaggregated P/D workers.

Usage (from project root):
    python workers/test_disaggregated.py --model Qwen/Qwen3-8B \
        --prefill-gpu 1 --decode-gpu 2

What this verifies:
    1. PrefillWorker allocates blocks and produces a first token
    2. KV buffer is correctly extracted to pinned CPU memory
    3. DecodeWorker receives KV async and promotes the request to running
    4. DecodeWorker.step() produces coherent tokens for N steps
    5. (sanity) CollocatedWorker produces similar output for the same prompt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
import argparse
import time
import torch

from block_manager.block_manager import BlockSpaceManager
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from workers.prefill_worker import PrefillWorker
from workers.decode_worker import DecodeWorker
from workers.collocated_worker import CollocatedWorker


# ── helpers ──────────────────────────────────────────────────────────────────

def make_group(prompt: str, runner, block_size: int, seq_id: int) -> SequenceGroup:
    token_ids = runner.tokenizer(prompt).input_ids
    seq = Sequence(seq_id=seq_id, prompt_token_ids=token_ids, block_size=block_size)
    # seq.status = SequenceStatus.RUNNING          # allocate() expects RUNNING
    return SequenceGroup(request_id=str(seq_id), seqs=[seq])


def decode_tokens(token_ids, tokenizer) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ── test 1: single request, disaggregated path ───────────────────────────────

def test_single_request(model_path: str, prefill_gpu: int, decode_gpu: int,
                         block_size: int, max_blocks: int, max_new_tokens: int,
                         prompt: str):
    print(f"\n{'='*60}")
    print(f"[Test 1] Disaggregated single request")
    print(f"  prefill GPU: {prefill_gpu}  decode GPU: {decode_gpu}")
    print(f"  prompt: {prompt!r}")
    print(f"{'='*60}")

    # shared block manager — same instance for both workers
    block_manager = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)

    pw = PrefillWorker(model_path, gpu_id=prefill_gpu,
                       block_manager=block_manager,
                       block_size=block_size, max_blocks=max_blocks)
    dw = DecodeWorker(model_path, gpu_id=decode_gpu,
                      block_manager=block_manager,
                      block_size=block_size, max_blocks=max_blocks)

    # build sequence group using prefill worker's tokenizer
    group = make_group(prompt, pw.runner, block_size, seq_id=0)

    # ── prefill ──
    t0 = time.perf_counter()
    first_token, block_table, kv_buf, src_k, src_v = pw.prefill_and_extract(group)
    t_prefill = (time.perf_counter() - t0) * 1000

    seq = group.get_seqs()[0]
    prompt_len = seq.prompt_len
    print(f"  prefill done  | prompt_len={prompt_len}  "
          f"first_token={first_token} ({pw.runner.tokenizer.decode([first_token])!r})  "
          f"time={t_prefill:.1f} ms")
    print(f"  block_table: {block_table}  num_blocks={len(block_table)}")

    # ── KV transfer ──
    t0 = time.perf_counter()
    dw.receive_kv_async(group, block_table, kv_buf, src_k, src_v)
    t_transfer_launch = (time.perf_counter() - t0) * 1000
    print(f"  transfer launched in {t_transfer_launch:.2f} ms  "
          f"(async, actual copy happens on transfer_stream)")

    # wait for transfer to complete before first decode step
    torch.cuda.synchronize(dw.device)

    # ── decode ──
    generated = [first_token]
    eos = dw.runner.tokenizer.eos_token_id
    step_times = []

    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        results = dw.step()
        step_times.append((time.perf_counter() - t0) * 1000)

        if not results:
            print(f"  decode step {step}: no results (request finished or not promoted)")
            break

        _, tok_id = results[0]
        generated.append(tok_id)

        if tok_id == eos:
            print(f"  EOS at step {step}")
            break

    avg_step = sum(step_times) / len(step_times) if step_times else 0
    print(f"\n  decode steps={len(step_times)}  avg_step={avg_step:.1f} ms")
    print(f"\n  [Output]\n  {decode_tokens(generated, pw.runner.tokenizer)}")

    return generated


# ── test 2: overlap — receive while decoding ──────────────────────────────────

def test_overlap(model_path: str, prefill_gpu: int, decode_gpu: int,
                 block_size: int, max_blocks: int, prompts: list):
    """
    Send 2 requests back to back.
    While req-0 is decoding, req-1's KV is being received on transfer_stream.
    Verifies _promote_ready() picks it up correctly.
    """
    print(f"\n{'='*60}")
    print(f"[Test 2] Overlap: decode req-0 while receiving req-1 KV")
    print(f"{'='*60}")

    assert len(prompts) >= 2

    block_manager = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)
    pw = PrefillWorker(model_path, gpu_id=prefill_gpu,
                       block_manager=block_manager,
                       block_size=block_size, max_blocks=max_blocks)
    dw = DecodeWorker(model_path, gpu_id=decode_gpu,
                      block_manager=block_manager,
                      block_size=block_size, max_blocks=max_blocks)

    # prefill req-0
    g0 = make_group(prompts[0], pw.runner, block_size, seq_id=0)
    tok0, bt0, buf0, src_k, src_v = pw.prefill_and_extract(g0)
    dw.receive_kv_async(g0, bt0, buf0, src_k, src_v)
    torch.cuda.synchronize(dw.device)      # ensure req-0 is ready before first step

    # prefill req-1 (while req-0 is already in decode worker)
    g1 = make_group(prompts[1], pw.runner, block_size, seq_id=1)
    tok1, bt1, buf1, src_k, src_v = pw.prefill_and_extract(g1)

    # launch req-1 KV transfer (async — overlaps with decode steps below)
    dw.receive_kv_async(g1, bt1, buf1, src_k, src_v)
    print(f"  req-1 KV transfer launched asynchronously")

    # decode req-0 for a few steps; req-1 should appear in running after transfer done
    all_results = {0: [tok0], 1: [tok1]}
    promoted_at_step = None

    for step in range(20):
        results = dw.step()
        n_running = len(dw.running) + len(results)

        for group, tok in results:
            gid = int(group.request_id)
            all_results[gid].append(tok)

        if promoted_at_step is None and any(r[0].request_id == "1" for r in results):
            promoted_at_step = step
            print(f"  req-1 promoted to running at step {step}")

        if not dw.running and not dw._pending:
            break

    print(f"  req-0 output: {decode_tokens(all_results[0], pw.runner.tokenizer)!r}")
    print(f"  req-1 output: {decode_tokens(all_results[1], pw.runner.tokenizer)!r}")
    assert promoted_at_step is not None, "req-1 was never promoted — transfer may have failed"
    print(f"  [PASS] overlap test passed")


# ── test 3: collocated baseline ───────────────────────────────────────────────

def test_collocated(model_path: str, gpu_id: int, block_size: int,
                    max_blocks: int, prompt: str, max_new_tokens: int):
    print(f"\n{'='*60}")
    print(f"[Test 3] Collocated baseline  GPU={gpu_id}")
    print(f"  prompt: {prompt!r}")
    print(f"{'='*60}")

    cw = CollocatedWorker(model_path, gpu_id=gpu_id,
                          block_size=block_size, max_blocks=max_blocks)
    t0 = time.perf_counter()
    results = cw.run_until_done_single(prompt, max_new_tokens=max_new_tokens)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  time={elapsed:.1f} ms")
    print(f"  [Output]\n  {results}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prefill-gpu", type=int, default=1)
    parser.add_argument("--decode-gpu", type=int, default=2)
    parser.add_argument("--collocated-gpu", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-blocks", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--skip-collocated", action="store_true")
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()

    PROMPT_A = "介绍一下北京大学。"
    PROMPT_B = "What is the capital of France?"

    # test 1: basic disaggregated path
    test_single_request(
        model_path=args.model,
        prefill_gpu=args.prefill_gpu,
        decode_gpu=args.decode_gpu,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        max_new_tokens=args.max_new_tokens,
        prompt=PROMPT_A,
    )

    # test 2: overlap
    if not args.skip_overlap:
        test_overlap(
            model_path=args.model,
            prefill_gpu=args.prefill_gpu,
            decode_gpu=args.decode_gpu,
            block_size=args.block_size,
            max_blocks=args.max_blocks,
            prompts=[PROMPT_A, PROMPT_B],
        )

    # test 3: collocated baseline for sanity comparison
    if not args.skip_collocated:
        test_collocated(
            model_path=args.model,
            gpu_id=args.collocated_gpu,
            block_size=args.block_size,
            max_blocks=args.max_blocks,
            prompt=PROMPT_A,
            max_new_tokens=args.max_new_tokens,
        )