# Engine 模块文档

> 对应源码：`engine/scheduler.py`、`engine/model_runner.py`、`engine/model_runner_huggingface.py`、`engine/engine.py`

---

## 模块概述

Engine 模块是单卡推理引擎的核心部分，将 Block Manager 提供的物理内存管理能力与 HuggingFace Transformers 模型串联起来，完整实现了以下技术：

| 技术 | Where and How |
|---|---|
| **Continuous Batching**（持续批处理） | `Scheduler.schedule()` + `Engine.step()` |
| **Chunked Prefill**（分块预填充） | `Scheduler.schedule()` 的 `prefilling` 队列逻辑 |
| **PagedAttention KV Cache** | `ModelRunner` 预分配的 `k_cache / v_cache` |
| **Monkey Patching**（运行时替换 Attention） | `ModelRunner._patch_attention_layers()` |
| **Paged Kernel 调用** | `paged_kernels.paged_kv_store` / `paged_kernels.paged_attention_forward` |
| **GQA（Grouped Query Attention）** | `repeat_interleave` 扩展 KV head |

模块由四个文件组成：

| 文件 | 职责 |
|---|---|
| `scheduler.py` | 调度器：管理四条队列，决定每步 batch 的组成 |
| `model_runner.py` | Paged 版 ModelRunner：管理 KV Cache、执行 Monkey Patch、驱动前向计算 |
| `model_runner_huggingface.py` | HuggingFace 原版 ModelRunner：用于对比基线，使用 `use_cache=True` 的传统 KV Cache |
| `engine.py` | 顶层引擎：串联调度器与 ModelRunner，实现完整的请求生命周期管理 |

---

## scheduler.py

### SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    prefill_group        : Optional[SequenceGroup]  # 本轮执行 Prefill 的组（至多一个）
    prefill_chunk_tokens : Optional[List[int]]       # 本轮 Prefill 的 token id 切片
    prefill_start_position: int                      # 切片在原始 prompt 中的起始位置
    prefill_is_last      : bool                      # 本切片是否是该序列 Prefill 的最后一块
    decode_groups        : List[SequenceGroup]       # 本轮执行 Decode 的所有 Group
```

调度器每次调用 `schedule()` 返回一个 `SchedulerOutput`，完整描述本步 forward pass 的组成。`Engine.step()` 根据这个数据包构造输入张量并组装 `_current_context`。

---

### Scheduler

```python
class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager, max_batch_size: int = 8, budget: int = 512)
```

调度器维护四条队列，所有请求在生命周期中依次流经这四条队列：

```
waiting → prefilling → running → finished
```

#### 四条队列详解

| 队列 | 类型 | 含义 |
|---|---|---|
| `waiting` | `List[SequenceGroup]` | 刚加入的请求，物理块尚未分配 |
| `prefilling` | `List[SequenceGroup]` | 已分配物理块，正在进行 Chunked Prefill（prompt 过长，需要跨多步完成） |
| `running` | `List[SequenceGroup]` | Prefill 完成，正在执行 Decode |
| `finished` | `List[SequenceGroup]` | 所有序列已结束，等待结果被读取 |

#### 关键参数

| 参数 | 含义 |
|---|---|
| `max_batch_size` | `running` 队列的最大并发 Group 数，限制同时 Decode 的请求数 |
| `BUDGET` | 每步 forward pass 允许处理的最大 token 数（Prefill token 数 + Decode token 数之和） |

#### schedule() —— 核心调度逻辑

```python
def schedule(self) -> SchedulerOutput
```

**每个推理步调用一次**，决定本步 batch 由哪些序列组成。完整执行三个阶段：

**阶段一：清理 running 队列**

遍历 `running`，将 `is_finished` 的 Group 移入 `finished` 并调用 `block_manager.free()` 归还物理块 **（这里似乎有点问题，结束状态的标定会被覆盖）**，其余保留在 `running`。

**阶段二：计算 Prefill 预算**

```python
decode_tokens = len(self.running)         # 每个 running Group 贡献 1 个 decode token
prefill_budget = self.BUDGET - decode_tokens
```

`BUDGET` 是每步 forward 允许的 token 总数上限。优先保证所有 Decode 序列的 slot，剩余预算留给 Prefill。若 `prefill_budget <= 0`，本步跳过 Prefill，只执行 Decode。

**这正是 Continuous Batching 的核心思想**：Prefill 和 Decode 共享同一个 forward pass，互不独占，通过 Budget 动态分配算力。

**阶段三：选取 Prefill 任务（Chunked Prefill）**

优先处理 `prefilling` 队列中已在进行的序列，其次从 `waiting` 拉取新请求。

对于正在 Prefill 的序列：

```python
start = seq.num_computed_tokens                     # 上一步已处理到的位置
end   = min(start + prefill_budget, seq.prompt_len) # 本步最多能处理到的位置
prefill_chunk_tokens = all_tokens[start:end]        # 切出本步的 token 切片
prefill_is_last = (end >= seq.prompt_len)           # 是否是最后一块
if prefill_is_last:
    self.prefilling.pop(0)                          # 处理完毕，移出 prefilling 队列
```

对于从 `waiting` 拉取的新请求：

1. 检查 `running` 是否未超过 `max_batch_size`，以及 `block_manager.can_allocate()` 是否允许；
2. 若满足，调用 `block_manager.allocate()` 分配物理块，设置序列状态为 `RUNNING`；
3. 同样按 `prefill_budget` 切出第一块；
4. 若 prompt 过长（第一块没覆盖完），将该 Group 追加到 `prefilling` 队列，等待后续步骤继续处理。

**Chunked Prefill 的意义**：超长 prompt 不再阻塞整个系统。一个 2048 token 的 prompt 在 `budget=512` 下会被分成至少 4 步处理，期间其他请求仍可正常 Decode，大幅降低尾延迟。

---

## model_runner.py（Paged version）

### ModelRunner

```python
class ModelRunner:
    def __init__(self, model_path: str, device: str = "cuda", max_blocks: int = 512, block_size: int = 16)
```

Paged  ModelRunner 是整个引擎最复杂的组件。它在加载模型后对模型执行 Monkey Patch，将每一层的 Attention 替换为支持 Paged KV Cache 的自定义实现。

#### 初始化过程

**1. 加载模型和 Tokenizer**

```python
self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map=device)
self.model.eval()
```

以 `float16` 精度加载，并切换为 eval 模式（关闭 dropout 等训练专用层）。

**2. 读取模型配置**

从 `model.config` 中提取推理所需的关键超参数：

| 字段 | 来源 | 含义 |
|---|---|---|
| `num_heads` | `configs.num_attention_heads` | Q head 数 |
| `num_kv_heads` | `configs.num_key_value_heads`（GQA 模型）或 `num_attention_heads` | KV head 数 |
| `head_dim` | `configs.head_dim` 或 `hidden_size // num_attention_heads` | 每个 head 的维度 |

**3. 预分配 KV Cache**

```python
self.k_cache = torch.zeros(
    num_layers, max_blocks, num_kv_heads, block_size, head_dim,
    dtype=torch.float16, device=device
)
self.v_cache = torch.zeros(...)  # 同形状
```

KV Cache 是一个五维张量，在模型加载时**一次性分配所有显存**，后续通过 `block_num` 索引直接读写，不再动态申请：

| 维度 | 含义 |
|---|---|
| `num_layers` | 模型层数，每层有独立的 KV Cache |
| `max_blocks` | 物理块总数，与 `BlockAllocator` 一一对应 |
| `num_kv_heads` | KV head 数（GQA 场景下小于 Q head 数） |
| `block_size` | 每块能存放的 token 数 |
| `head_dim` | 每个 head 的特征维度 |

访问某个序列的 KV Cache 时，通过其 `block_table`（物理块号列表）直接索引 `k_cache[layer_idx][block_num]`，实现 O(1) 的页式寻址。

**4. 执行 Monkey Patch**

```python
self._patch_attention_layers()
```

见下文。

---

#### _patch_attention_layers() —— Monkey Patch 入口

```python
def _patch_attention_layers(self)
```

**运行时替换所有 Transformer 层的 Attention forward 方法**，是整个 Paged Attention 实现的关键入口。

执行流程：

1. **检测模型类型**，从对应的 transformers 模块中导入 `apply_rotary_pos_emb` 函数。目前支持 `qwen2`、`llama`、`qwen3`；
2. **检测 RoPE 位置**：
   - `per_layer` 模式：每层 `self_attn` 有自己的 `rotary_emb`（如 Llama、Qwen2）；
   - `top_level` 模式：`rotary_emb` 挂在顶层 `model.model` 上（部分模型架构）；
3. 保存 `apply_rotary_pos_emb` 引用到 `self._apply_rotary_pos_emb`；
4. 遍历所有层，对每层的 `self_attn` 调用 `_patch_single_layer()`。

---

#### _patch_single_layer() —— 单层 Attention 替换

```python
def _patch_single_layer(self, attn_module, layer_idx: int)
```

定义闭包函数 `paged_forward`，用 `attn_module.forward = paged_forward` 直接覆盖原始 forward 方法。此后调用 `model(input_ids=...)` 时，每一层 Attention 都会执行 `paged_forward` 而不是 transformers 原版实现。

**`paged_forward` 的完整执行流程：**

**① QKV 投影**

```python
q = attn_module.q_proj(hidden_states)   # (1, total_tokens, num_heads * head_dim)
k = attn_module.k_proj(hidden_states)
v = attn_module.v_proj(hidden_states)
```

沿用原始模块的投影权重，保持与原模型数值等价。

**② QK Norm（可选）**

```python
if hasattr(attn_module, 'q_norm'):
    q = attn_module.q_norm(q)
if hasattr(attn_module, 'k_norm'):
    k = attn_module.k_norm(k)
```

Qwen3 等模型在 RoPE 之前对 Q、K 做 RMSNorm，需要在此处应用。

**③ 应用 RoPE（旋转位置编码）**

```python
# per_layer 模式
cos, sin = attn_module.rotary_emb(v, position_ids)
q, k = runner._apply_rotary_pos_emb(q, k, cos, sin)
```

`position_ids` 由 `Engine.step()` 精确构造，对 Prefill token 和 Decode token 分别填入正确的绝对位置，保证 RoPE 的语义正确。

**④ 读取调度上下文**

```python
ctx = runner._current_context
num_prefill = ctx['num_prefill_tokens']
num_decode  = ctx['num_decode_tokens']
```

`_current_context` 是 `Engine.step()` 在每次 forward 前通过 `runner._current_context = ctx` 注入的字典，包含本步 batch 的完整描述。这是 Scheduler、Engine 与 ModelRunner 之间唯一的运行时信息传递通道。

**⑤ Prefill 分支**

```python
q_p = q[:, :, :num_prefill, :]  # 取前 num_prefill 个 token 的 Q
k_p = k[:, :, :num_prefill, :]
v_p = v[:, :, :num_prefill, :]
```

对每个 prefill 序列：

- 调用 `paged_kernels.paged_kv_store()` 将本块的 K/V 写入全局 KV Cache 的对应物理块；
- 构造因果掩码（上三角为 `-inf`）；
- 对 GQA 模型，用 `repeat_interleave` 将 KV head 扩展至与 Q head 相同数量；
- 调用 `F.scaled_dot_product_attention()` 完成标准 Prefill Attention 计算。

Prefill 阶段使用 PyTorch 的 `scaled_dot_product_attention`（可后端到 FlashAttention），因为 Prefill 的 Q、K、V 都在当前步中，无需读取历史缓存。

**⑥ Decode 分支**

```python
q_d = q[:, :, num_prefill:, :]  # 取后 num_decode 个 token 的 Q
```

对所有 Decode 序列：

- 构造 `block_tables`（二维张量，shape `[num_decode, max_blocks]`）和 `seq_lens`（每条序列当前长度）；
- 调用 `paged_kernels.paged_kv_store()` 将本步新生成的 K/V 写入 KV Cache；
- 调用 `run_kernel()` → `paged_kernels.paged_attention_forward()` 执行 Paged Attention：对每条序列，根据其 `block_table` 从不连续的物理块中读取历史 K/V，完成 Attention 计算。

Decode 阶段必须使用自定义 Kernel 而非标准 SDPA，因为每条序列的历史 KV 分散存放在非连续的物理块中，需要 Kernel 层面的间接寻址。

**⑦ 合并输出**

```python
attn_out = torch.cat(outputs, dim=1)     # 拼接 Prefill 和 Decode 的输出
return attn_module.o_proj(attn_out), None
```

Prefill 和 Decode 的输出在序列维度上拼接后，统一经过原始的 `o_proj` 输出投影。

---

#### GQA 处理

当 `num_kv_heads < num_heads` 时（如 Qwen2-7B 的 `num_heads=28, num_kv_heads=4`），Prefill 分支通过：

```python
k_p_ex = k_p.repeat_interleave(num_groups, dim=1)  # num_groups = num_heads // num_kv_heads
v_p_ex = v_p.repeat_interleave(num_groups, dim=1)
```

将 KV 复制扩展，使维度与 Q 匹配，然后再执行标准 SDPA。Decode 分支由 `paged_attention_forward` kernel 在内部处理 GQA。

---

#### Kernel 接口概览

> 两个 Kernel 的详细实现原理见 [03-cuda_kernels_cn.md](03-cuda_kernels_cn.md)。

本模块使用的两个 Kernel 均来自 `paged_kernels` 扩展：

| Kernel | 调用位置 | 作用 |
|---|---|---|
| `paged_kernels.paged_kv_store(k_cache, v_cache, k_src, v_src, block_tables, positions)` | Prefill 和 Decode 分支 | 将当前步计算出的 K/V 按 `block_table` 写入 KV Cache 对应物理块的对应 slot |
| `paged_kernels.paged_attention_forward(out, query, k_cache, v_cache, block_tables, seq_lens, scale, block_size, max_blocks_per_seq)` | Decode 分支 | 按 `block_table` 从非连续物理块中读取历史 KV，为每条 Decode 序列计算 Attention |

---

#### 独立接口方法（供单序列调试使用）

```python
def prefill_chunk(self, input_ids_chunk, block_table, start_position, is_last_chunk) -> Optional[Tensor]
```

单独对一个 chunk 执行 Prefill 前向计算，内部自行构造 `_current_context`。若 `is_last_chunk=True`，返回采样到的下一个 token id，否则返回 `None`。

```python
def decode_step(self, token_id, block_table, position) -> Tensor
```

单独执行一步 Decode 前向计算，返回采样到的下一个 token id。

```python
def generate(self, prompt, block_table, max_new_tokens=200) -> str
```

单序列独立生成接口，内部循环调用 `prefill_chunk` 和 `decode_step`，主要用于调试。**注意**：`Engine` 不使用这些方法，而是直接操作 `_current_context` 后调用 `model()`。

---

#### run_kernel()

```python
def run_kernel(query, key_cache, value_cache, block_tables, seq_lens, scale, block_size, max_blocks_per_seq) -> Tensor
```

对 `paged_kernels.paged_attention_forward` 的薄封装，主要处理输出张量的创建（`torch.zeros_like(query)`）和参数传递，减少 `paged_forward` 闭包内的代码量。

---

#### top_k_sample()

```python
def top_k_sample(logit: torch.Tensor, top_k: int = 1) -> torch.Tensor
```

从 logit 向量中采样一个 token id。执行步骤：

1. `torch.nan_to_num()` 清理 NaN 和 Inf（Qwen3 等模型偶尔产生数值异常）；
2. `torch.topk()` 取 top-k 个候选；
3. `softmax` + `multinomial` 在 top-k 内按概率采样。

默认 `top_k=1` 等价于贪心解码（argmax）。

---

## model_runner_huggingface.py（HuggingFace baseline version）

这是一个**不使用 Paged Attention 的简化版 ModelRunner**，以 HuggingFace 原生的 `use_cache=True` KV Cache 机制工作，主要用于功能对比和正确性验证。

```python
class ModelRunner:
    def __init__(self, model_path: str, device: str = "cuda")
```

无 `block_size`、`max_blocks` 参数，不做 Monkey Patch，不预分配 KV Cache。

### 方法

```python
def prefill(self, input_ids) -> (next_token, past_kv, attention_mask)
```

对完整 prompt 执行一次前向计算，返回第一个生成的 token、HuggingFace 格式的 `past_key_values`，以及当前的 `attention_mask`。

```python
def decode_step(self, token_id, past_kv, attention_mask) -> (next_token, past_kv, attention_mask)
```

依赖 HuggingFace 的 `past_key_values` 传递 KV 缓存，每步追加 attention_mask。显存消耗随序列长度线性增长，不支持多序列共享内存，无法实现 Continuous Batching。

```python
def generate(self, prompt, max_new_tokens=200) -> str
```

串行单序列生成，循环调用 `prefill` 和 `decode_step`。

### 与 Paged 版的核心差异

| 特性 | HuggingFace 版 | Paged 版 |
|---|---|---|
| KV Cache 存储 | `past_key_values`（Python 对象，动态增长） | 预分配的 `k_cache / v_cache` 张量（固定大小，按块寻址） |
| Continuous Batching | 不支持 | 支持 |
| Chunked Prefill | 不支持 | 支持 |
| 多序列共享内存（CoW） | 不支持 | 支持（通过 `fork`） |
| Monkey Patch | 无 | 全部 Attention 层替换 |
| `top_k_sample` top_k 默认值 | 10（有随机性） | 1（贪心） |

---

## engine.py

### Engine

```python
class Engine:
    def __init__(self, model_path: str, block_size: int = 16, max_blocks: int = 512, device: str = "cuda")
```

顶层引擎，将 `ModelRunner`、`BlockSpaceManager`、`Scheduler` 三者组装起来，对外暴露简洁的请求提交和生成接口。

#### 初始化

```python
self.runner       = ModelRunner(model_path, block_size=block_size, max_blocks=max_blocks, device=device)
self.block_manager = BlockSpaceManager(block_size=block_size, num_gpu_blocks=max_blocks)
self.scheduler    = Scheduler(self.block_manager, max_batch_size=16, budget=1024)
self.seq_counter  = 0
```

`block_size` 和 `max_blocks` 必须在 `ModelRunner`（KV Cache 形状）和 `BlockSpaceManager`（物理块池大小）之间保持一致，`Engine` 负责确保这一点。

---

#### add_request()

```python
def add_request(self, prompt: str, request_id: str = None) -> SequenceGroup
```

将一个自然语言 prompt 转化为 `SequenceGroup` 并放入调度器的 `waiting` 队列：

1. 用 tokenizer 将 prompt 编码为 token id 列表；
2. 构造 `Sequence`（`seq_id` 由自增的 `seq_counter` 分配）；
3. 包装成 `SequenceGroup`，`request_id` 默认为 `seq_id` 的字符串形式；
4. 追加到 `self.scheduler.waiting`。

---

#### step() —— 单步推理

```python
def step(self) -> List[Tuple[SequenceGroup, int]]
```

执行一个完整的推理步骤，包含以下子阶段：

**1. 调度**

```python
sched = self.scheduler.schedule()
```

获得本步的调度结果。

**2. 构造输入**

将 Prefill chunk 和所有 Decode token 拼接成一个一维 token id 列表：

```python
# Prefill 部分：直接使用调度器切出的 token 切片
input_ids_list.extend(sched.prefill_chunk_tokens)
position_list.extend(range(start, start + len(tokens)))

# Decode 部分：每条序列贡献最后一个生成的 token
for group in sched.decode_groups:
    seq = group.get_seqs(SequenceStatus.RUNNING)[0]
    self.block_manager.append_slot(seq)        # 为新 token 预留物理 slot
    input_ids_list.append(seq.output_token_ids[-1])   # 上一步生成的 token
    position_list.append(seq.num_computed_tokens)      # 当前序列长度作为位置
```

最终 `input_ids` 是形状为 `(1, total_tokens)` 的张量，`total_tokens = num_prefill + num_decode`，整批序列在一次 forward 中并行计算。这正是 **Continuous Batching** 的实现：Prefill token 和 Decode token 在同一个序列维度上拼接，共享 QKV 投影计算。

**3. 组装 _current_context**

```python
ctx = {
    "num_prefill_tokens": num_prefill,
    "num_decode_tokens": len(sched.decode_groups),
    "prefills": [{"block_table": ..., "start_position": ..., "num_tokens": ...}],
    "decodes":  [{"block_table": ..., "position": ...}, ...],
}
runner._current_context = ctx
```

在执行 `model()` 之前，将调度信息注入 ModelRunner。`paged_forward` 闭包从这里读取每条序列的块表和位置信息。

**4. 前向计算**

```python
with torch.no_grad():
    logits = self.runner.model(input_ids=input_ids, position_ids=position_ids).logits
```

Shape: `(1, total_tokens, vocab_size)`。

**5. 采样与后处理**

- **Prefill 序列**：更新 `num_computed_tokens`；若是最后一块（`prefill_is_last`），从 `logits[0, num_prefill-1, :]` 采样第一个生成 token，追加到 `output_token_ids`，并将 Group 移入 `running`；
- **Decode 序列**：从 `logits[0, num_prefill + i, :]` 采样，追加到 `output_token_ids`，`num_computed_tokens += 1`；若采样到 EOS，设置状态为 `FINISHED_STOPPED`。

---

#### run_until_done()

```python
def run_until_done(self, max_tokens_per_seq: int = 500) -> dict
```

循环调用 `step()` 直到 `waiting`、`prefilling`、`running` 三条队列全部清空：

```python
while self.scheduler.running or self.scheduler.waiting or self.scheduler.prefilling:
    self.step()
    # 检查是否超过最大生成长度
    for group in self.scheduler.running:
        seqs = group.get_seqs(SequenceStatus.RUNNING)
        if seqs and len(seqs[0].output_token_ids) >= max_tokens_per_seq:
            seqs[0].status = SequenceStatus.FINISHED_STOPPED
```

完成后从 `scheduler.finished` 收集结果，用 tokenizer decode 成字符串，以 `{request_id: text}` 字典返回。

#### generate()

```python
def generate(self, prompt: str, max_new_tokens: int = 500) -> str
```

单请求接口：`add_request()` + `run_until_done()` 的组合，返回第一个（也是唯一一个）请求的生成文本。

---

## 完整数据流

```
用户调用 engine.add_request(prompt)
  │
  ▼
Tokenizer 编码 → Sequence → SequenceGroup
  │ 追加到 scheduler.waiting
  ▼
engine.step() 循环
  │
  ├─ scheduler.schedule()
  │     ├─ 清理 running 中已完成的序列（free 物理块）
  │     ├─ 计算 prefill_budget = BUDGET - len(running)
  │     └─ 返回 SchedulerOutput
  │           ├─ prefill_group + prefill_chunk_tokens（Chunked Prefill）
  │           └─ decode_groups（Continuous Batching）
  │
  ├─ 构造 input_ids (1, num_prefill + num_decode)
  │     ├─ Prefill: 调度器切出的 token 片段
  │     └─ Decode:  每条序列的上一个生成 token
  │
  ├─ block_manager.append_slot(seq) 为每条 decode 序列预留 slot
  │
  ├─ runner._current_context = ctx（注入块表和位置信息）
  │
  ├─ runner.model(input_ids, position_ids)
  │     每层 self_attn.forward = paged_forward（Monkey Patch）
  │     │
  │     ├─ QKV 投影 + QK Norm + RoPE
  │     │
  │     ├─ Prefill 分支
  │     │     ├─ paged_kv_store(): 写 K/V 到物理块
  │     │     └─ scaled_dot_product_attention(): 因果 Attention
  │     │
  │     └─ Decode 分支
  │           ├─ paged_kv_store(): 写当前 K/V 到物理块
  │           └─ paged_attention_forward(): 从非连续物理块读取历史 KV 并计算 Attention
  │
  ├─ logits (1, total_tokens, vocab_size)
  │
  ├─ top_k_sample() 采样
  │     ├─ Prefill 最后一块：采样第一个输出 token，Group → running
  │     └─ Decode：采样下一个 token，检查 EOS
  │
  └─ 若三条队列清空 → run_until_done 返回结果字典
```

---

## 各类关系总览

```
Engine
  ├── ModelRunner          ← 持有模型权重、KV Cache、执行 Monkey Patch
  │     ├── k_cache / v_cache   (num_layers, max_blocks, num_kv_heads, block_size, head_dim)
  │     └── _current_context    ← Engine 每步注入，paged_forward 读取
  ├── BlockSpaceManager    ← 管理物理块分配，维护 block_table
  └── Scheduler            ← 管理四条队列，输出每步 batch 的组成
        └── SchedulerOutput
              ├── prefill_group / prefill_chunk_tokens / prefill_start_position / prefill_is_last
              └── decode_groups
```

---

## 测试覆盖概览

| 测试文件 | 覆盖场景 |
|---|---|
| `test/test_scheduler.py` | Scheduler 队列流转、Chunked Prefill 多步进度、序列结束后的清理 |
| `test/test_engine.py` | Engine 端到端生成，与 HuggingFace 原版输出进行等价性对比（对比部分已注释） |
| `test/test_mixed_batch.py` | 混合 batch 一致性：同一请求在独立 Engine 和 Mixed Batch Engine 中的输出应完全相同 |

---

## 已知问题

| # | 位置 | 问题 |
|---|---|---|
| 1 | `schedule()` 中调用的 `BlockSpaceManager.free()` | 强制将状态设为 `FINISHED_ABORTED`，覆盖了正常结束序列的 `FINISHED_STOPPED` 状态 |
| 2 | `run_until_done` 长度检查时机 | 长度检查在 `step()` 完成后才执行，序列最多可能生成 `max_tokens_per_seq + 1` 个 token 才被截断 |
