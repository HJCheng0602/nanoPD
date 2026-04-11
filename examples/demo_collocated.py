"""
demo_collocated.py — Single-GPU collocated inference demo

Model : Qwen/Qwen2-1.5B  (fits in ~4 GB VRAM)
Target: NVIDIA RTX 4060 (or any single GPU)

Usage:
    python examples/demo_collocated.py
    python examples/demo_collocated.py --model /path/to/Qwen2-1.5B --gpu 0 --max-new-tokens 300
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../nanoPD'))

import argparse
import time

from engine.engine import Engine

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum entanglement in one sentence.",
    "Write a short poem about the ocean.",
    "What are the main advantages of transformer architectures?",
    "Describe the difference between prefill and decode phases in LLM inference.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="Qwen/Qwen2-1.5B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--gpu",            type=int, default=0)
    parser.add_argument("--block-size",     type=int, default=16)
    parser.add_argument("--max-blocks",     type=int, default=256,
                        help="KV cache capacity (256 blocks × 16 tokens = 4096 token slots)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading {args.model} on cuda:{args.gpu} ...")
    t0 = time.perf_counter()
    engine = Engine(
        model_path=args.model,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        device=f"cuda:{args.gpu}",
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    total_tokens = 0
    total_time   = 0.0

    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] Prompt: {prompt}")
        t0 = time.perf_counter()
        response = engine.generate(prompt, max_new_tokens=args.max_new_tokens)
        elapsed  = time.perf_counter() - t0

        tok_ids   = engine.runner.tokenizer(response).input_ids
        tok_count = len(tok_ids)
        total_tokens += tok_count
        total_time   += elapsed

        print(f"  Response ({tok_count} tokens, {elapsed:.2f}s, {tok_count/elapsed:.1f} tok/s):")
        # indent multi-line responses
        for line in response.splitlines():
            print(f"    {line}")
        print()

    print("=" * 50)
    print(f"Total: {len(PROMPTS)} requests  {total_tokens} tokens  "
          f"{total_time:.1f}s  avg {total_tokens/total_time:.1f} tok/s")


if __name__ == "__main__":
    main()
