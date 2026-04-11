# Engine Module

> Source files: `engine/scheduler.py`, `engine/model_runner.py`, `engine/model_runner_huggingface.py`, `engine/engine.py`

---

## Overview

The Engine module is the core of the single-GPU inference engine. It connects the physical memory management provided by the Block Manager with HuggingFace Transformers models, and fully implements the following techniques:

| Technique | Where and How |
|---|---|
| **Continuous Batching** | `Scheduler.schedule()` + `Engine.step()` |
| **Chunked Prefill** | `prefilling` queue logic in `Scheduler.schedule()` |
| **PagedAttention KV Cache** | Pre-allocated `k_cache / v_cache` in `ModelRunner` |
| **Monkey Patching** (runtime Attention replacement) | `ModelRunner._patch_attention_layers()` |
| **Paged Kernel calls** | `paged_kernels.paged_kv_store` / `paged_kernels.paged_attention_forward` |
| **GQA (Grouped Query Attention)** | `repeat_interleave` to expand KV heads |

The module consists of four files:

| File | Responsibility |
|---|---|
| `scheduler.py` | Scheduler: manages four queues, decides batch composition each step |
| `model_runner.py` | Paged ModelRunner: manages KV Cache, applies Monkey Patch, drives forward computation |
| `model_runner_huggingface.py` | HuggingFace baseline ModelRunner: uses traditional `use_cache=True` KV Cache for correctness comparison |
| `engine.py` | Top-level engine: wires scheduler and ModelRunner together, manages the full request lifecycle |

---

## scheduler.py

### SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    prefill_group         : Optional[SequenceGroup]  # Group being prefilled this step (at most one)
    prefill_chunk_tokens  : Optional[List[int]]       # Token id slice for this prefill chunk
    prefill_start_position: int                       # Start position of the slice in the original prompt
    prefill_is_last       : bool                      # Whether this chunk is the last one for this sequence
    decode_groups         : List[SequenceGroup]       # All groups executing decode this step
```

`schedule()` returns one `SchedulerOutput` per step, fully describing the batch composition for the upcoming forward pass. `Engine.step()` uses this to construct input tensors and assemble `_current_context`.

---

### Scheduler

```python
class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager, max_batch_size: int = 8, budget: int = 512)
```

The scheduler maintains four queues. Every request flows through them in order during its lifetime:

```
waiting → prefilling → running → finished
```

#### The four queues

| Queue | Type | Meaning |
|---|---|---|
| `waiting` | `List[SequenceGroup]` | Newly submitted requests; no physical blocks allocated yet |
| `prefilling` | `List[SequenceGroup]` | Physical blocks allocated; chunked prefill in progress (prompt too long for a single step) |
| `running` | `List[SequenceGroup]` | Prefill complete; actively decoding |
| `finished` | `List[SequenceGroup]` | All sequences ended; results ready to be consumed |

#### Key parameters

| Parameter | Meaning |
|---|---|
| `max_batch_size` | Maximum concurrent groups in `running`; limits parallel decode requests |
| `BUDGET` | Maximum token count per forward pass (prefill tokens + decode tokens combined) |

#### schedule() — core scheduling logic

```python
def schedule(self) -> SchedulerOutput
```

Called once per inference step. Determines the batch composition for that step. Executes three phases:

**Phase 1: Clean up the running queue**

Iterates over `running`. Groups where `is_finished` is true are moved to `finished` and `block_manager.free()` is called to reclaim physical blocks. The remaining groups stay in `running`.

> **Note**: `block_manager.free()` unconditionally sets the sequence status to `FINISHED_ABORTED`, which overwrites the original termination reason (`FINISHED_STOPPED`). See the [Known Issues](#known-issues) section.

**Phase 2: Compute the prefill budget**

```python
decode_tokens  = len(self.running)          # each running group contributes 1 decode token
prefill_budget = self.BUDGET - decode_tokens
```

`BUDGET` is the per-step token count ceiling. Decode slots are reserved first; the remainder goes to prefill. If `prefill_budget <= 0`, prefill is skipped this step and only decoding runs.

**This is the essence of Continuous Batching**: prefill and decode share one forward pass, neither monopolising it; `BUDGET` dynamically allocates compute between them.

**Phase 3: Select the prefill task (Chunked Prefill)**

The `prefilling` queue (in-progress sequences) takes priority over `waiting` (new requests).

For a sequence already in `prefilling`:

```python
start = seq.num_computed_tokens                      # position processed up to last step
end   = min(start + prefill_budget, seq.prompt_len)  # furthest we can reach this step
prefill_chunk_tokens = all_tokens[start:end]         # token slice for this step
prefill_is_last = (end >= seq.prompt_len)            # last chunk?
if prefill_is_last:
    self.prefilling.pop(0)                           # remove from prefilling queue
```

For a new request pulled from `waiting`:

1. Check that `running` has not exceeded `max_batch_size` and that `block_manager.can_allocate()` passes;
2. If both pass, call `block_manager.allocate()` to assign physical blocks and set status to `RUNNING`;
3. Cut the first chunk using `prefill_budget`;
4. If the prompt is too long to finish in one chunk, append the group to `prefilling` for continuation.

**Why Chunked Prefill matters**: a very long prompt no longer stalls the system. A 2048-token prompt with `budget=512` is spread across at least 4 steps; other requests keep decoding throughout, dramatically reducing tail latency.

---

## model_runner.py (Paged version)

### ModelRunner

```python
class ModelRunner:
    def __init__(self, model_path: str, device: str = "cuda", max_blocks: int = 512, block_size: int = 16)
```

The Paged ModelRunner is the most complex component in the engine. After loading the model it immediately applies a Monkey Patch, replacing the `forward` method of every Attention layer with a custom implementation that supports the Paged KV Cache.

#### Initialization

**1. Load model and tokenizer**

```python
self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map=device)
self.model.eval()
```

Loaded in `float16` precision and switched to eval mode (disabling dropout and other training-only layers).

**2. Read model config**

Key hyperparameters extracted from `model.config`:

| Field | Source | Meaning |
|---|---|---|
| `num_heads` | `configs.num_attention_heads` | Number of Q heads |
| `num_kv_heads` | `configs.num_key_value_heads` (GQA) or `num_attention_heads` | Number of KV heads |
| `head_dim` | `configs.head_dim` or `hidden_size // num_attention_heads` | Dimension per head |

**3. Pre-allocate KV Cache**

```python
self.k_cache = torch.zeros(
    num_layers, max_blocks, num_kv_heads, block_size, head_dim,
    dtype=torch.float16, device=device
)
self.v_cache = torch.zeros(...)  # same shape
```

KV Cache is a five-dimensional tensor **allocated once at model load time**. All subsequent reads and writes index it directly by `block_num`; no dynamic allocation ever happens again:

| Dimension | Meaning |
|---|---|
| `num_layers` | One independent KV Cache per transformer layer |
| `max_blocks` | Total physical block count, in 1-to-1 correspondence with `BlockAllocator` |
| `num_kv_heads` | KV head count (less than Q heads in GQA models) |
| `block_size` | Tokens that fit in one block |
| `head_dim` | Feature dimension per head |

A sequence's KV Cache is accessed via its `block_table` (list of physical block numbers) indexing directly into `k_cache[layer_idx][block_num]`, giving O(1) paged addressing.

**4. Apply Monkey Patch**

```python
self._patch_attention_layers()
```

See below.

---

#### _patch_attention_layers() — Monkey Patch entry point

```python
def _patch_attention_layers(self)
```

**Replaces the Attention `forward` method of every transformer layer at runtime.** This is the central hook that makes Paged Attention work.

Steps:

1. **Detect model type** and import the corresponding `apply_rotary_pos_emb` function from transformers. Currently supports `qwen2`, `llama`, `qwen3`;
2. **Detect RoPE location**:
   - `per_layer` mode: each `self_attn` has its own `rotary_emb` (e.g. Llama, Qwen2);
   - `top_level` mode: `rotary_emb` is on the top-level `model.model` object;
3. Store the `apply_rotary_pos_emb` reference in `self._apply_rotary_pos_emb`;
4. Iterate over all layers and call `_patch_single_layer()` on each `self_attn`.

---

#### _patch_single_layer() — per-layer Attention replacement

```python
def _patch_single_layer(self, attn_module, layer_idx: int)
```

Defines a closure `paged_forward` and assigns it with `attn_module.forward = paged_forward`. From this point on, every call to `model(input_ids=...)` routes each layer's Attention through `paged_forward` instead of the original transformers implementation.

**`paged_forward` execution walkthrough:**

**① QKV projection**

```python
q = attn_module.q_proj(hidden_states)   # (1, total_tokens, num_heads * head_dim)
k = attn_module.k_proj(hidden_states)
v = attn_module.v_proj(hidden_states)
```

Uses the original module's projection weights, preserving numerical equivalence with the base model.

**② QK Norm (optional)**

```python
if hasattr(attn_module, 'q_norm'):
    q = attn_module.q_norm(q)
if hasattr(attn_module, 'k_norm'):
    k = attn_module.k_norm(k)
```

Models such as Qwen3 apply RMSNorm to Q and K before RoPE; this handles that case.

**③ Apply RoPE (Rotary Position Embedding)**

```python
# per_layer mode
cos, sin = attn_module.rotary_emb(v, position_ids)
q, k = runner._apply_rotary_pos_emb(q, k, cos, sin)
```

`position_ids` is constructed precisely by `Engine.step()` — prefill tokens and decode tokens each receive their correct absolute positions, preserving RoPE semantics.

**④ Read the scheduling context**

```python
ctx         = runner._current_context
num_prefill = ctx['num_prefill_tokens']
num_decode  = ctx['num_decode_tokens']
```

`_current_context` is injected by `Engine.step()` via `runner._current_context = ctx` immediately before each forward call. It is the sole runtime communication channel between the Scheduler, Engine, and ModelRunner.

**⑤ Prefill branch**

```python
q_p = q[:, :, :num_prefill, :]   # Q for the first num_prefill tokens
k_p = k[:, :, :num_prefill, :]
v_p = v[:, :, :num_prefill, :]
```

For each prefill sequence:

- Call `paged_kernels.paged_kv_store()` to write this chunk's K/V into the corresponding physical blocks of the global KV Cache;
- Construct a causal mask (upper triangle set to `-inf`);
- For GQA models, expand KV heads to match Q heads with `repeat_interleave`;
- Run `F.scaled_dot_product_attention()` for standard prefill Attention.

Prefill uses PyTorch's `scaled_dot_product_attention` (which can dispatch to FlashAttention) because all Q, K, V are available in the current step — no historical cache lookup needed.

**⑥ Decode branch**

```python
q_d = q[:, :, num_prefill:, :]   # Q for the num_decode tokens
```

For all decode sequences:

- Build `block_tables` (2-D tensor, shape `[num_decode, max_blocks]`) and `seq_lens` (current length of each sequence);
- Call `paged_kernels.paged_kv_store()` to write the newly computed K/V into the cache;
- Call `run_kernel()` → `paged_kernels.paged_attention_forward()` for Paged Attention: for each sequence, the kernel follows its `block_table` to read historical K/V from non-contiguous physical blocks and compute Attention.

Decode must use a custom kernel rather than standard SDPA because each sequence's historical KV is scattered across non-contiguous physical blocks, requiring kernel-level indirect addressing.

**⑦ Merge outputs**

```python
attn_out = torch.cat(outputs, dim=1)    # concatenate prefill and decode outputs along the token dim
return attn_module.o_proj(attn_out), None
```

Prefill and decode outputs are concatenated along the sequence dimension and passed through the original `o_proj` output projection.

---

#### GQA handling

When `num_kv_heads < num_heads` (e.g. Qwen2-7B with `num_heads=28, num_kv_heads=4`), the prefill branch expands KV heads via:

```python
num_groups = num_heads // num_kv_heads
k_p_ex = k_p.repeat_interleave(num_groups, dim=1)
v_p_ex = v_p.repeat_interleave(num_groups, dim=1)
```

This replicates the KV tensors to match the Q head count before running standard SDPA. The decode branch's GQA is handled internally by `paged_attention_forward`.

---

#### Kernel interface summary

> For a detailed breakdown of both kernels' implementation, see [03-cuda_kernels_en.md](03-cuda_kernels_en.md).

Both kernels come from the `paged_kernels` extension:

| Kernel | Call site | Purpose |
|---|---|---|
| `paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_tables, positions)` | Prefill and decode branches | Write the K/V computed this step into the correct slots of the correct physical blocks, as specified by `block_table` |
| `paged_kernels.paged_attention_forward(out, query, k_cache, v_cache, block_tables, seq_lens, scale, block_size, max_blocks_per_seq)` | Decode branch | Read historical KV from non-contiguous physical blocks via `block_table` and compute Attention for each decode sequence |

---

#### Standalone interface methods (for single-sequence debugging)

```python
def prefill_chunk(self, input_ids_chunk, block_table, start_position, is_last_chunk) -> Optional[Tensor]
```

Runs a prefill forward pass on a single chunk, constructing `_current_context` internally. Returns the sampled next token id if `is_last_chunk=True`, otherwise `None`.

```python
def decode_step(self, token_id, block_table, position) -> Tensor
```

Runs a single decode forward pass, returning the sampled next token id.

```python
def generate(self, prompt, block_table, max_new_tokens=200) -> str
```

Single-sequence generation interface that loops over `prefill_chunk` and `decode_step`. Useful for debugging. **Note**: `Engine` does not use these methods — it sets `_current_context` directly and calls `model()`.

---

#### run_kernel()

```python
def run_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, block_size, max_blocks_per_seq) -> Tensor
```

A thin wrapper around `paged_kernels.paged_attention_forward`. Handles output tensor creation (`torch.zeros_like(query)`) and argument forwarding, keeping the `paged_forward` closure cleaner.

---

#### top_k_sample()

```python
def top_k_sample(logit: torch.Tensor, top_k: int = 1) -> torch.Tensor
```

Samples a token id from a logit vector:

1. `torch.nan_to_num()` — cleans up NaN and Inf (Qwen3 and similar models occasionally produce numerical anomalies);
2. `torch.topk()` — selects the top-k candidates;
3. `softmax` + `multinomial` — samples from the top-k distribution.

Default `top_k=1` is equivalent to greedy decoding (argmax).

---

## model_runner_huggingface.py (HuggingFace baseline version)

A **simplified ModelRunner that does not use Paged Attention**, operating with HuggingFace's native `use_cache=True` KV Cache. Its purpose is functional comparison and correctness verification.

```python
class ModelRunner:
    def __init__(self, model_path: str, device: str = "cuda")
```

No `block_size` or `max_blocks` parameters. No Monkey Patch. No pre-allocated KV Cache.

### Methods

```python
def prefill(self, input_ids) -> (next_token, past_kv, attention_mask)
```

Runs one forward pass over the full prompt, returning the first generated token, HuggingFace-format `past_key_values`, and the current `attention_mask`.

```python
def decode_step(self, token_id, past_kv, attention_mask) -> (next_token, past_kv, attention_mask)
```

Relies on HuggingFace's `past_key_values` to carry the KV cache between steps, appending to `attention_mask` each step. Memory grows linearly with sequence length; no cross-sequence sharing; Continuous Batching is not supported.

```python
def generate(self, prompt, max_new_tokens=200) -> str
```

Sequential single-sequence generation loop over `prefill` and `decode_step`.

### Core differences from the paged version

| Feature | HuggingFace version | Paged version |
|---|---|---|
| KV Cache storage | `past_key_values` (Python object, grows dynamically) | Pre-allocated `k_cache / v_cache` tensors (fixed size, block-indexed) |
| Continuous Batching | Not supported | Supported |
| Chunked Prefill | Not supported | Supported |
| Cross-sequence memory sharing (CoW) | Not supported | Supported (via `fork`) |
| Monkey Patch | None | All Attention layers replaced |
| `top_k_sample` default `top_k` | 10 (stochastic) | 1 (greedy) |

---

## engine.py

### Engine

```python
class Engine:
    def __init__(self, model_path: str, block_size: int = 16, max_blocks: int = 512, device: str = "cuda")
```

The top-level engine. Assembles `ModelRunner`, `BlockSpaceManager`, and `Scheduler` and exposes a clean request submission and generation interface.

#### Initialization

```python
self.runner        = ModelRunner(model_path, block_size=block_size, max_blocks=max_blocks, device=device)
self.block_manager = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)
self.scheduler     = Scheduler(self.block_manager, max_batch_size=16, budget=1024)
self.seq_counter   = 0
```

`block_size` and `max_blocks` must be consistent between `ModelRunner` (KV Cache tensor shape) and `BlockSpaceManager` (physical block pool size). `Engine` is responsible for enforcing this.

---

#### add_request()

```python
def add_request(self, prompt: str, request_id: str = None) -> SequenceGroup
```

Converts a natural-language prompt into a `SequenceGroup` and enqueues it in `waiting`:

1. Tokenize the prompt into a token id list;
2. Construct a `Sequence` (assigning `seq_id` from the auto-incrementing `seq_counter`);
3. Wrap it in a `SequenceGroup` (`request_id` defaults to the string form of `seq_id`);
4. Append to `self.scheduler.waiting`.

---

#### step() — single inference step

```python
def step(self) -> List[Tuple[SequenceGroup, int]]
```

Executes one complete inference step:

**1. Schedule**

```python
sched = self.scheduler.schedule()
```

Obtain the scheduling decision for this step.

**2. Build inputs**

Concatenate the prefill chunk and all decode tokens into a single flat token id list:

```python
# Prefill: the token slice produced by the scheduler
input_ids_list.extend(sched.prefill_chunk_tokens)
position_list.extend(range(start, start + len(tokens)))

# Decode: each running sequence contributes its last generated token
for group in sched.decode_groups:
    seq = group.get_seqs(SequenceStatus.RUNNING)[0]
    self.block_manager.append_slot(seq)             # reserve a slot for the new token
    input_ids_list.append(seq.output_token_ids[-1]) # last generated token
    position_list.append(seq.num_computed_tokens)   # current sequence length as position
```

The final `input_ids` has shape `(1, total_tokens)` where `total_tokens = num_prefill + num_decode`. All sequences are processed in a **single forward pass** — this is **Continuous Batching** in practice: prefill tokens and decode tokens are concatenated along the token dimension and share the QKV projection computation.

**3. Assemble _current_context**

```python
ctx = {
    "num_prefill_tokens": num_prefill,
    "num_decode_tokens": len(sched.decode_groups),
    "prefills": [{"block_table": ..., "start_position": ..., "num_tokens": ...}],
    "decodes":  [{"block_table": ..., "position": ...}, ...],
}
runner._current_context = ctx
```

Scheduling information is injected into the ModelRunner before the forward call. The `paged_forward` closure reads each sequence's block table and position from here.

**4. Forward pass**

```python
with torch.no_grad():
    logits = self.runner.model(input_ids=input_ids, position_ids=position_ids).logits
```

Shape: `(1, total_tokens, vocab_size)`.

**5. Sample and post-process**

- **Prefill sequences**: increment `num_computed_tokens`; if this is the last chunk (`prefill_is_last`), sample from `logits[0, num_prefill-1, :]` to get the first generated token, append it to `output_token_ids`, and move the group to `running`;
- **Decode sequences**: sample from `logits[0, num_prefill + i, :]`, append to `output_token_ids`, increment `num_computed_tokens`; if EOS is sampled, set status to `FINISHED_STOPPED`.

---

#### run_until_done()

```python
def run_until_done(self, max_tokens_per_seq: int = 500) -> dict
```

Loops over `step()` until `waiting`, `prefilling`, and `running` are all empty:

```python
while self.scheduler.running or self.scheduler.waiting or self.scheduler.prefilling:
    self.step()
    for group in self.scheduler.running:
        seqs = group.get_seqs(SequenceStatus.RUNNING)
        if seqs and len(seqs[0].output_token_ids) >= max_tokens_per_seq:
            seqs[0].status = SequenceStatus.FINISHED_STOPPED
```

After the loop, collects results from `scheduler.finished`, decodes token ids to strings with the tokenizer, and returns a `{request_id: text}` dict.

#### generate()

```python
def generate(self, prompt: str, max_new_tokens: int = 500) -> str
```

Single-request convenience wrapper: `add_request()` + `run_until_done()`. Returns the generated text for the one submitted request.

---

## Full data flow

```
User calls engine.add_request(prompt)
  │
  ▼
Tokenizer encodes → Sequence → SequenceGroup
  │ appended to scheduler.waiting
  ▼
engine.step() loop
  │
  ├─ scheduler.schedule()
  │     ├─ remove finished groups from running (free physical blocks)
  │     ├─ compute prefill_budget = BUDGET - len(running)
  │     └─ return SchedulerOutput
  │           ├─ prefill_group + prefill_chunk_tokens  (Chunked Prefill)
  │           └─ decode_groups                         (Continuous Batching)
  │
  ├─ build input_ids (1, num_prefill + num_decode)
  │     ├─ Prefill: token slice from scheduler
  │     └─ Decode:  last generated token per sequence
  │
  ├─ block_manager.append_slot(seq) per decode sequence
  │
  ├─ runner._current_context = ctx  (inject block tables and positions)
  │
  ├─ runner.model(input_ids, position_ids)
  │     every self_attn.forward = paged_forward  (Monkey Patch)
  │     │
  │     ├─ QKV projection + QK Norm + RoPE
  │     │
  │     ├─ Prefill branch
  │     │     ├─ paged_kv_store(): write K/V to physical blocks
  │     │     └─ scaled_dot_product_attention(): causal Attention
  │     │
  │     └─ Decode branch
  │           ├─ paged_kv_store(): write current K/V to physical blocks
  │           └─ paged_attention_forward(): read historical KV from
  │                non-contiguous blocks and compute Attention
  │
  ├─ logits (1, total_tokens, vocab_size)
  │
  ├─ top_k_sample()
  │     ├─ Prefill last chunk: sample first output token, group → running
  │     └─ Decode: sample next token, check for EOS
  │
  └─ all three queues empty → run_until_done returns result dict
```

---

## Class relationship overview

```
Engine
  ├── ModelRunner          ← holds model weights, KV Cache, applies Monkey Patch
  │     ├── k_cache / v_cache   (num_layers, max_blocks, num_kv_heads, block_size, head_dim)
  │     └── _current_context    ← injected by Engine each step, read by paged_forward
  ├── BlockSpaceManager    ← manages physical block allocation, maintains block tables
  └── Scheduler            ← manages four queues, outputs batch composition per step
        └── SchedulerOutput
              ├── prefill_group / prefill_chunk_tokens / prefill_start_position / prefill_is_last
              └── decode_groups
```

---

## Test coverage summary

| Test file | Scenarios covered |
|---|---|
| `test/test_scheduler.py` | Queue transitions, multi-step chunked prefill progress, cleanup after sequence ends |
| `test/test_mixed_batch.py` | Mixed-batch consistency: the same request should produce identical output whether run alone or alongside other requests |

---

## Known issues

| # | Location | Issue |
|---|---|---|
| 1 | `BlockSpaceManager.free()` called in `schedule()` | Unconditionally sets status to `FINISHED_ABORTED`, overwriting the `FINISHED_STOPPED` state for normally-completed sequences |
| 2 | `run_until_done` max-length check | The length check runs after `step()` completes, so sequences may produce `max_tokens_per_seq + 1` tokens before being truncated |
