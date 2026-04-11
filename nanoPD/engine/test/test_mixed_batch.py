
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from engine.engine import Engine
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

engine_a = Engine(model_path)
engine_a.add_request("1+1=")
results_a = engine_a.run_until_done(max_tokens_per_seq=30)
seq_a = engine_a.scheduler.finished[0].get_seqs()[0]
tokens_a = seq_a.output_token_ids
print("Decode-only:", tokens_a)

engine_b = Engine(model_path)
engine_b.add_request("介绍一下北京大学。" * 20)  
engine_b.add_request("1+1=")
results_b = engine_b.run_until_done(max_tokens_per_seq=30)

for group in engine_b.scheduler.finished:
    seq = group.get_seqs()[0]
    if seq.prompt_len < 10:  
        tokens_b = seq.output_token_ids
        break
print("Mixed batch :", tokens_b)

assert tokens_a == tokens_b, f"mixed batch affected decode output!\nA: {tokens_a}\nB: {tokens_b}"
print("mixed batch test passed")