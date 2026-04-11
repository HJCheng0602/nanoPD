from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch

class ModelRunner:
    def __init__(self, model_path:str, device:str="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype = torch.float16,
            device_map=device
        )
        self.model.eval()
        
    @torch.inference_mode()
    def prefill(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits
        past_kv = outputs.past_key_values
        next_token = top_k_sample(logits[0, -1, :])
        return next_token, past_kv, attention_mask


    @torch.inference_mode()
    def decode_step(self, token_id:torch.Tensor, past_kv, attention_mask:torch.Tensor):
        # token_id : scalar tensor
        x = token_id.view(1, 1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, device=self.device)], dim=1
        )
        outputs = self.model(x, past_key_values=past_kv, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits     # (1, 1, vocab_size)
        past_kv = outputs.past_key_values

        next_token = top_k_sample(logits[0, -1, :])
        return next_token, past_kv, attention_mask
    
    def generate(self, prompt:str, max_new_tokens: int=200):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        token, past_kv, mask = self.prefill(input_ids)
        generated = [token.item()]

        for _ in range(max_new_tokens - 1):
            if token.item() == self.tokenizer.eos_token_id:
                break
            token, past_kv, mask = self.decode_step(token, past_kv, mask)
            generated.append(token.item())
        
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def top_k_sample(logit:torch.Tensor) -> torch.Tensor:
    top_k = 10
    top_k_logits, top_k_ids = torch.topk(logit, top_k)
    top_k_softmax = torch.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(top_k_softmax, num_samples=1).squeeze(0)
    return top_k_ids[sampled_idx]


if __name__ == "__main__":
    qwen2run = ModelRunner("Qwen/Qwen3-8B")
    print(qwen2run.generate("I am a stupid student."))
