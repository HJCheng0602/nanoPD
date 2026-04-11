# test_equivalence.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from engine.engine import Engine
from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from block_manager.block_manager import BlockSpaceManager
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner

prompt = "北京是"
model_path = "Qwen/Qwen3-8B"

# ── HuggingFace baseline ──────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_path)
# hf_model  = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).cuda()
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
# generated = input_ids.clone()
# hf_tokens = []
# prompt_len = input_ids.shape[1]
# for step in range(200):
#     with torch.no_grad():
#         out = hf_model(generated)
#     next_tok = out.logits[0, -1, :].argmax().item()
#     # print(f"[HF] pos={prompt_len + step} tok={next_tok}")
#     hf_tokens.append(next_tok)
#     generated = torch.cat([generated, torch.tensor([[next_tok]]).cuda()], dim=1)
#     if next_tok == tokenizer.eos_token_id or len(hf_tokens) >= 200:
#         break
# print("HF :", hf_tokens)
# print("HF text:", tokenizer.decode(hf_tokens))

# ── Your engine ───────────────────────────────────────────────────────
engine = Engine(model_path)
engine.add_request(prompt)
results = engine.run_until_done(max_tokens_per_seq=200)
seq = engine.scheduler.finished[0].get_seqs()[0]
your_tokens = seq.output_token_ids
print("Yours:", your_tokens)
print("Yours text:", tokenizer.decode(your_tokens))

# assert hf_tokens == your_tokens, f"mismatch!\nHF:    {hf_tokens}\nYours: {your_tokens}"