from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Auto-select attention backend: FlashInfer > hand-written CUDA fallback
# ---------------------------------------------------------------------------
try:
    from paged_attention.flashinfer_backend import run_kernel as _paged_attn_run_kernel
    _HAS_FLASHINFER = True
except Exception:
    _HAS_FLASHINFER = False

class ModelRunner:
    def __init__(self, model_path:str, device:str="cuda", max_blocks:int=512,block_size:int=16 ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype = torch.float16,
            device_map=device
        )
        self.model.eval()
        configs = self.model.config
        num_layers = configs.num_hidden_layers
        num_kv_heads = getattr(configs, "num_key_value_heads", configs.num_attention_heads)

        head_dim = getattr(configs, "head_dim", configs.hidden_size // configs.num_attention_heads)
        self.num_heads = configs.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.k_cache = torch.zeros(
            num_layers, max_blocks, num_kv_heads, block_size, head_dim,
            dtype=torch.float16, device=device
        )
        self.v_cache = torch.zeros(
            num_layers, max_blocks, num_kv_heads, block_size, head_dim,
            dtype=torch.float16, device=device
        )
        # self._current_context = {
        #     "num_prefill_tokens":4,
        #     "num_decode_tokens":3,

        #     "prefill":{
        #         "blocktable":[...],
        #         "start_position":12,
        #     },

        #     "decodes":[
        #         {"block_table":[...], "position":42},
        #         {"block_table":[...], "position":4},
        #         {"block_table":[...], "position":114514},
        #     ],
        # }
        self._current_context = None
        self._patch_attention_layers()

    
    def _patch_attention_layers(self):
        model_type = self.model.config.model_type
        if model_type == "qwen2":
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        elif model_type == "llama":
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        elif model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        else:
            raise NotImplementedError(f"unsupported model type: {model_type}")
        first_attn = self.model.model.layers[0].self_attn
        if hasattr(first_attn, "rotary_emb"):
            self._rotary_mode = "per_layer"
        elif hasattr(self.model.model, "rotary_emb"):
            self._rotary_mode = "top_level"
        else:
            raise RuntimeError("cannot find rotary embedding")

        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        for layer_idx, layer in enumerate(self.model.model.layers):
            self._patch_single_layer(layer.self_attn, layer_idx)

        

    def _patch_single_layer(self, attn_module, layer_idx):
        runner = self
        def paged_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                **kwargs,
        ):
            # when we don't use batch like forward
            bsz, q_len, _ = hidden_states.shape
            q = attn_module.q_proj(hidden_states)
            k = attn_module.k_proj(hidden_states)
            v = attn_module.v_proj(hidden_states)
            num_heads    = runner.num_heads
            num_kv_heads = runner.num_kv_heads
            head_dim     = runner.head_dim   
            num_groups   = num_heads // num_kv_heads
            scale = head_dim ** -0.5
            import paged_kernels


            q = q.view(bsz, q_len, num_heads, head_dim)         # (1, H, total, D)
            k = k.view(bsz, q_len, num_kv_heads, head_dim)      # (1, kvh, total, D)
            v = v.view(bsz, q_len, num_kv_heads, head_dim)

            if hasattr(attn_module, 'q_norm'):
                q = attn_module.q_norm(q)
            if hasattr(attn_module, 'k_norm'):
                k = attn_module.k_norm(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if runner._rotary_mode == "per_layer":
                cos, sin = attn_module.rotary_emb(v, position_ids)
                q, k = runner._apply_rotary_pos_emb(q, k, cos, sin)
            else:
                cos, sin = runner.model.model.rotary_emb(v, position_ids)
                q, k = runner._apply_rotary_pos_emb(q, k, cos, sin)
    
            ctx = runner._current_context
            # print(f"[paged_forward] setting ctx: num_prefill={ctx['num_prefill_tokens']} num_decode={ctx['num_decode_tokens']}")
            num_prefill = ctx['num_prefill_tokens']
            num_decode = ctx['num_decode_tokens']
            outputs = []

            # prefill
            if num_prefill > 0:
                q_p = q[:, :, :num_prefill, :]
                k_p = k[:, :, :num_prefill, :]
                v_p = v[:, :, :num_prefill, :]

                prefills = ctx['prefills']

                token_offset = 0
                for pinfo in prefills:
                    n = pinfo['num_tokens']
                    positions = torch.arange(
                        pinfo['start_position'],
                        pinfo['start_position'] + n,
                        dtype=torch.int32, device=runner.device
                    )
                    bt = torch.tensor(pinfo['block_table'], dtype=torch.int32, device=runner.device)
                    bt_2d = bt.unsqueeze(0).expand(n, -1).contiguous()

                    k_src = k_p[:, :, token_offset:token_offset + n, :].squeeze(0).contiguous()
                    v_src = v_p[:, :, token_offset:token_offset + n, :].squeeze(0).contiguous()

                    with torch.cuda.device(runner.device):
                        paged_kernels.paged_kv_store(
                            runner.k_cache[layer_idx], runner.v_cache[layer_idx],
                            k_src, v_src, bt_2d, positions
                        )
                    token_offset += n

                mask = torch.full(
                    (num_prefill, num_prefill), float('-inf'),
                    dtype=q_p.dtype, device=runner.device
                )
                blk_offset = 0
                for pinfo in prefills:
                    L = pinfo['num_tokens']
                    causal_blk = torch.triu(
                        torch.full((L, L), float('-inf'), dtype=q_p.dtype, device=runner.device),
                        diagonal=1
                    )
                    mask[blk_offset:blk_offset + L, blk_offset:blk_offset + L] = causal_blk
                    blk_offset += L

                k_p_ex = k_p.repeat_interleave(num_groups, dim=1)
                v_p_ex = v_p.repeat_interleave(num_groups, dim=1)
                out_p = F.scaled_dot_product_attention(q_p, k_p_ex, v_p_ex, attn_mask=mask, scale=scale)
                outputs.append(out_p.transpose(1, 2).reshape(1, num_prefill, -1))
            
            if num_decode > 0:
                q_d = q[:,:, num_prefill:, :]
                k_d = k[:,:, num_prefill:, :]
                v_d = v[:,:, num_prefill:, :]

                decodes = ctx['decodes']
                max_blocks = max(len(d['block_table']) for d in decodes)
                block_tables = torch.zeros(num_decode, max_blocks, dtype=torch.int32, device=runner.device)
                seq_lens = torch.zeros(num_decode, dtype=torch.int32, device=runner.device)
                for i, d in enumerate(decodes):
                    bt = d['block_table']
                    block_tables[i, :len(bt)] = torch.tensor(bt, dtype=torch.int32, device=runner.device)
                    seq_lens[i] = d['position'] + 1
                positions_d = torch.tensor(
                    [d['position'] for d in decodes],
                    dtype=torch.int32, device=runner.device
                )
                with torch.cuda.device(runner.device):
                    paged_kernels.paged_kv_store(
                        runner.k_cache[layer_idx], runner.v_cache[layer_idx],
                        k_d.squeeze(0).contiguous(),
                        v_d.squeeze(0).contiguous(),
                        block_tables, positions_d   
                    )
                    # (1, H, num_decode, D) -> (num_decode, H, D)
                    q_kernel = q_d.squeeze(0).transpose(0, 1).contiguous()

                    out_d = run_kernel(
                        query=q_kernel,
                        key_cache=runner.k_cache[layer_idx],
                        value_cache=runner.v_cache[layer_idx],
                        block_tables=block_tables,
                        seq_lens=seq_lens,
                        scale=scale,
                        block_size=runner.block_size,
                        max_blocks_per_seq=max_blocks
                    )

                outputs.append(out_d.view(1, num_decode, -1))
            attn_out = torch.cat(outputs, dim=1)

            return attn_module.o_proj(attn_out), None
        attn_module.forward = paged_forward



        
    @torch.inference_mode()
    def prefill_chunk(self, input_ids_chunk, block_table: list, start_position: int, is_last_chunk: bool):
        chunk_len = input_ids_chunk.shape[1]
        position_ids = torch.arange(
            start_position, start_position + chunk_len,
            device=self.device
        ).unsqueeze(0)

        self._current_context = {
            "num_prefill_tokens": chunk_len,
            "num_decode_tokens": 0,
            "prefills": [{
                "block_table": block_table,
                "start_position": start_position,
                "num_tokens": chunk_len,
            }],
            "decodes": [],
        }

        outputs = self.model(input_ids_chunk, position_ids=position_ids, use_cache=False)
        if is_last_chunk:
            return top_k_sample(outputs.logits[0, -1, :])
        return None

    @torch.inference_mode()
    def decode_step(self, token_id: torch.Tensor, block_table: list, position: int):
        self._current_context = {
            "num_prefill_tokens": 0,
            "num_decode_tokens": 1,
            "prefills": [],
            "decodes": [{"block_table": block_table, "position": position}],
        }
        x = token_id.view(1, 1)
        position_ids = torch.tensor([[position]], device=self.device)
        outputs = self.model(x, position_ids=position_ids, use_cache=False)
        return top_k_sample(outputs.logits[0, 0, :])

    def generate(self, prompt: str, block_table: list, max_new_tokens: int = 200):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        token = self.prefill_chunk(input_ids, block_table, start_position=0, is_last_chunk=True)
        generated = [token.item()]

        for step in range(max_new_tokens - 1):
            if token.item() == self.tokenizer.eos_token_id:
                break
            token = self.decode_step(token, block_table, prompt_len + step)
            generated.append(token.item())

        return self.tokenizer.decode(generated, skip_special_tokens=True)

def run_kernel(query:torch.Tensor, key_cache, value_cache, block_tables, seq_lens, scale, block_size, max_blocks_per_seq)-> torch.Tensor:
    if _HAS_FLASHINFER:
        return _paged_attn_run_kernel(
            query, key_cache, value_cache,
            block_tables, seq_lens,
            scale, block_size, max_blocks_per_seq
        )
    import paged_kernels
    out = torch.zeros_like(query)
    paged_kernels.paged_attention_forward(
        out, query, key_cache, value_cache,
        block_tables, seq_lens,
        scale, block_size, max_blocks_per_seq
    )
    return out

def top_k_sample(logit: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    logit = torch.nan_to_num(logit, nan=0.0, posinf=1e4, neginf=-1e4)  # guard against NaN/Inf logits
    top_k_logits, top_k_ids = torch.topk(logit, top_k)
    top_k_softmax = torch.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(top_k_softmax, num_samples=1).squeeze(0)
    return top_k_ids[sampled_idx]

# old test not suitable for qwen3
if __name__ == "__main__":
    dummy_block_table = list(range(64))
    runner = ModelRunner("Qwen/Qwen3-8B")
    print(runner.generate("介绍一下北京大学。", block_table=dummy_block_table))
