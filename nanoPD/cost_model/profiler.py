import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from engine.engine import Engine

@dataclass
class ProfileResult:
    T_prefill: Dict[int, float] = field(default_factory=dict)           # prompt_len -> ms
    T_decode:Dict[Tuple, float] = field(default_factory=dict)           # (kv_len, batch) -> ms
    T_interference: Dict[Tuple, float] = field(default_factory=dict)    # (chunk_size, decode_batch) -> ms
    p2p_bandwidth_GBps: float = 0.0

def _cuda_time(fn, warm_up=3, repeat=10)->float:
    # warm up
    for _ in range(warm_up):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))

def profile_prefill(engine:Engine, prompt_lens: List[int]) -> Dict[int, float]:
    results = {}
    device = engine.runner.device

    for L in prompt_lens:
        print(f"      profilling prefill L = {L}...")

        input_ids = torch.randint(0, 1000, (1, L), device=device)
        position_ids = torch.arange(L, device=device).unsqueeze(0)

        num_blocks_needed = (L + engine.block_size - 1) // engine.block_size
        block_table = list(range(num_blocks_needed))

        engine.runner._current_context = {
            "num_prefill_tokens": L,
            "num_decode_tokens": 0,
            "prefills": [{
                "block_table": block_table,
                "start_position": 0,
                "num_tokens": L,
            }],
            "decodes": []
        }

        def _forward(input_ids=input_ids, position_ids=position_ids):
            with torch.no_grad():
                engine.runner.model(
                    input_ids = input_ids,
                    position_ids=position_ids
                )
        ms = _cuda_time(_forward)
        results[L] = ms
        print(f"   T_prefill({L}) = {ms:.2f} ms")
    
    return results

def profile_decode(engine:Engine, kv_lens:List[int], batch_sizes:List[int]) -> Dict[Tuple, float]:
    results = {}
    runner = engine.runner
    device = runner.device

    for B in batch_sizes:
        for kv_len in kv_lens:
            print(f"    profiling decode batch={B} kv_len={kv_len}...")
            
            input_ids = torch.randint(0, 1000, (1, B), device=device)
            position_ids = torch.tensor(
                [[kv_len] * B], dtype=torch.long, device=device
            )

            num_blocks = (kv_len + 1 + engine.block_size - 1) // engine.block_size

            engine.runner._current_context = {
                "num_prefill_tokens": 0,
                "num_decode_tokens": B,
                "prefills": [], 
                "decodes":[
                    {
                        "block_table": list(range(num_blocks)),
                        "position": kv_len
                    }
                    for _ in range(B)
                ]
            }

            def _forward(input_ids=input_ids, position_ids=position_ids):
                with torch.no_grad():
                    engine.runner.model(
                        input_ids = input_ids,
                        position_ids = position_ids
                    )

            ms = _cuda_time(_forward)
            results[(kv_len, B)] = ms

            print(f"    T_decode(kv={kv_len}, B={B}) = {ms:.2f} ms")
        
    return results


def profile_interference(engine: Engine, chunk_sizes: List[int], decode_batches: List[int]) -> Dict[Tuple, float]:
    results = {}
    device = engine.runner.device
    kv_len = 512

    for B in decode_batches:
        input_ids = torch.randint(0, 1000, (1, B), device=device)
        position_ids = torch.tensor([[kv_len] * B], dtype=torch.long, device=device)
        num_blocks = (kv_len + 1 + engine.block_size - 1) // engine.block_size

        # baseline: pure decode without prefill
        engine.runner._current_context = {
            "num_prefill_tokens": 0,
            "num_decode_tokens": B,
            "prefills": [],
            "decodes": [{"block_table": list(range(num_blocks)), "position": kv_len}] * B
        }

        def _baseline(input_ids=input_ids, position_ids=position_ids):
            with torch.no_grad():
                engine.runner.model(input_ids=input_ids, position_ids=position_ids)

        t_baseline = _cuda_time(_baseline)
        print(f"  baseline decode B={B}: {t_baseline:.2f} ms")

        for chunk_size in chunk_sizes:
            print(f"  profiling interference chunk={chunk_size} decode_batch={B}...")

            chunk_blocks = (chunk_size + engine.block_size - 1) // engine.block_size
            total_tokens = chunk_size + B
            mixed_input = torch.randint(0, 1000, (1, total_tokens), device=device)
            mixed_pos = torch.cat([
                torch.arange(chunk_size, device=device),
                torch.tensor([kv_len] * B, device=device)
            ]).unsqueeze(0)

            engine.runner._current_context = {
                "num_prefill_tokens": chunk_size,
                "num_decode_tokens": B,
                "prefills": [{
                    "block_table": list(range(chunk_blocks)),
                    "start_position": 0,
                    "num_tokens": chunk_size,
                }],
                "decodes": [{"block_table": list(range(num_blocks)), "position": kv_len}] * B
            }

            def _mixed(mixed_input=mixed_input, mixed_pos=mixed_pos):
                with torch.no_grad():
                    engine.runner.model(input_ids=mixed_input, position_ids=mixed_pos)

            t_mixed = _cuda_time(_mixed)
            interference = t_mixed - t_baseline
            results[(chunk_size, B)] = interference
            print(f"    interference(chunk={chunk_size}, B={B}) = {interference:.2f} ms "
                  f"(baseline={t_baseline:.2f}, mixed={t_mixed:.2f})")

    return results

def profile_p2p_bandwidth(src_gpu: int = 0, dst_gpu: int = 1) -> float:
    size_bytes = 512 * 1024 * 1024
    src = torch.zeros(size_bytes // 4, dtype=torch.float32, device=f'cuda:{src_gpu}')
    cpu_buf = torch.empty(size_bytes // 4, dtype=torch.float32, pin_memory=True)

    from workers.kv_transfer import _check_p2p
    src_dev = torch.device(f'cuda:{src_gpu}')
    dst_dev = torch.device(f'cuda:{dst_gpu}')
    has_p2p = _check_p2p(src_dev, dst_dev)

    # warmup
    if has_p2p:
        dst = torch.empty_like(src, device=f'cuda:{dst_gpu}')
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
    else:
        for _ in range(3):
            cpu_buf.copy_(src)
            cpu_buf.to(f'cuda:{dst_gpu}')
        torch.cuda.synchronize()

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        if has_p2p:
            dst.copy_(src)
            torch.cuda.synchronize()
        else:
            cpu_buf.copy_(src)
            _ = cpu_buf.to(f'cuda:{dst_gpu}')
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    bw = (size_bytes / 1e9) / float(np.median(times))
    path = "P2P direct" if has_p2p else "pinned relay"
    print(f"  [{path}] GPU{src_gpu}->GPU{dst_gpu}: {bw:.2f} GB/s")
    return bw


def run_full_profile(
    model_path: str,
    output_path: str = "cost_model/profile_data_h20.pt",
    src_gpu: int = 0,
    dst_gpu: int = 2,
):
    print(f"Loading engine with model: {model_path}")
    engine = Engine(model_path, block_size=16, max_blocks=512)
    result = ProfileResult()
 
    print("\n=== [1/4] Profiling T_prefill ===")
    result.T_prefill = profile_prefill(
        engine,
        prompt_lens=[64, 128, 256, 512, 1024, 2048],
    )
 
    print("\n=== [2/4] Profiling T_decode ===")
    result.T_decode = profile_decode(
        engine,
        kv_lens=[128, 512, 1024, 2048],
        batch_sizes=[1, 4, 8, 16, 32],
    )
 
    print("\n=== [3/4] Profiling T_interference ===")
    result.T_interference = profile_interference(
        engine,
        chunk_sizes=[64, 128, 256, 512],
        decode_batches=[4, 8, 16],
    )
 
    print("\n=== [4/4] Profiling P2P Bandwidth ===")
    result.p2p_bandwidth_GBps = profile_p2p_bandwidth(src_gpu=src_gpu, dst_gpu=dst_gpu)
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(result, output_path)
    print(f"\nProfile data saved to: {output_path}")
    return result
 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--output", default="cost_model/profile_data_h20.pt")
    parser.add_argument("--src-gpu", type=int, default=0)
    parser.add_argument("--dst-gpu", type=int, default=2)
    args = parser.parse_args()
 
    run_full_profile(args.model, args.output, args.src_gpu, args.dst_gpu)