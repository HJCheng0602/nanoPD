# Workers 模块文档

> 对应源码：`workers/`

---

## 模块概述

`workers/` 提供三种部署路径的顶层执行单元：

| 类 | 文件 | 路径 |
|---|---|---|
| `CollocatedWorker` | `collocated_worker.py` | Collocated（P/D 混合，单 GPU） |
| `PrefillWorker` | `prefill_worker.py` | Disaggregated 的 Prefill 侧 |
| `DecodeWorker` | `decode_worker.py` | Disaggregated 的 Decode 侧 |
| `PinnedKVBuffer` / 传输函数 | `kv_transfer.py` | KV Cache 跨设备迁移基础设施 |

两种路径共用同一套 `BlockSpaceManager` 实例和 `ModelRunner`，差异仅在调度逻辑和 KV 传输层。

---

## kv_transfer.py

KV Cache 跨设备传输的底层基础设施，供 `PrefillWorker` 和 `DecodeWorker` 共同使用。

### `PinnedKVBuffer`

```python
class PinnedKVBuffer:
    def __init__(self, num_layers, num_block, num_kv_heads, block_size, head_dim, dtype=torch.float16)
    @staticmethod
    def from_runner(runner, num_blocks) -> "PinnedKVBuffer"
```

形状为 `(num_layers, num_block, num_kv_heads, block_size, head_dim)` 的一对 Pinned Memory 张量（`self.k` / `self.v`）。

**Pinned Memory（页锁定内存）** 是 CUDA 异步 H2D/D2H 传输的前提：普通 CPU 内存在传输期间可能被 OS 换页，导致传输中断或需要先临时拷贝一次；Pinned Memory 锁定物理页，DMA 可以直接操作，允许 `non_blocking=True` 的异步拷贝。

`from_runner` 是便捷方法，从 `ModelRunner` 实例中读取 `num_layers`、`num_kv_heads`、`block_size`、`head_dim`，避免手动传参。

### `extract_kv_to_pinned`

```python
def extract_kv_to_pinned(k_cache, v_cache, block_table, buf: PinnedKVBuffer)
```

从 Prefill GPU 的 `k_cache / v_cache`（形状 `[num_layers, max_blocks, ...]`）按 `block_table` 中的物理块号，将对应块逐层拷贝到 `buf` 的 Pinned Memory：

```python
for i, bid in enumerate(block_table):
    buf.k[:, i].copy_(k_cache[:, bid])   # 所有层，第 i 个逻辑块 ← 物理块 bid
    buf.v[:, i].copy_(v_cache[:, bid])
```

此操作是**同步**的（无 `non_blocking`），调用后需 `torch.cuda.synchronize()` 确保 GPU→CPU 完成。

### `load_kv_from_pinned`

```python
def load_kv_from_pinned(k_cache, v_cache, block_table, buf, stream=None)
```

`extract_kv_to_pinned` 的逆操作：将 Pinned Memory 中的 KV 按 `block_table` 写入 Decode GPU 的 `k_cache / v_cache`。当传入 `stream` 时，拷贝以 `non_blocking=True` 提交到该 Stream，实现与 Compute Stream 的并发。

### `transfer_kv`

```python
def transfer_kv(src_k, src_v, dst_k, dst_v, block_table, stream=None, buf=None) -> str
```

统一传输入口，自动选择最优路径：

```
if P2P 可用（两卡均为 CUDA，且互相可访问）:
    直接 GPU→GPU 拷贝（src_k[:, bid] → dst_k[:, bid]）
    return "p2p"
else:
    通过 Pinned Memory 中转（load_kv_from_pinned）
    return "pinned_relay"
```

P2P 路径省去 CPU 中转，延迟更低；但消费级显卡（如 RTX 系列）在 PCIe 拓扑下往往无法开启 P2P，实际通常走 `pinned_relay`。

### `_check_p2p`

```python
def _check_p2p(src_device, dst_device) -> bool
```

用 `torch.cuda.can_device_access_peer()` 检测两卡是否支持 P2P 访问，失败时静默返回 `False`。

---

## CollocatedWorker

```python
class CollocatedWorker:
    def __init__(self, model_path, gpu_id=0, block_size=16, max_blocks=512)
    def add_request(self, prompt) -> str
    def step(self) -> List[Tuple[SequenceGroup, int]]
    def run_until_done(self, max_tokens_per_seq=500) -> dict
    def run_until_done_single(self, prompt, max_new_tokens=500) -> str
```

`CollocatedWorker` 是对 `Engine` 的**极薄包装**，所有接口直接委托给内部的 `self.engine`：

```python
self.engine = Engine(model_path, block_size, max_blocks, device=f"cuda:{gpu_id}")
self.finished = self.engine.scheduler.finished
```

### Continuous Batching 与 Chunked Prefill

`CollocatedWorker` 的核心优势在于它完整继承了 `Engine` 的调度能力：

- **Continuous Batching**：每个 `step()` 由 `Scheduler` 混合调度——新请求的 Prefill token 与已有请求的 Decode token 在**同一次前向传播**中一起计算。GPU 不会因为等待新请求而空转，也不会因为有大批 Prefill 而阻塞 Decode。
- **Chunked Prefill**：长 Prompt 不会独占一整步，而是以 `budget` 为上限逐步推进，让 Decode 请求始终能穿插进来，避免长尾延迟。

这两点共同决定了 Collocated 路径在高并发下的吞吐量优势。

---

## PrefillWorker

```python
class PrefillWorker:
    def __init__(self, model_path, gpu_id, block_manager, block_size=16, max_blocks=512)
```

`PrefillWorker` 直接持有 `ModelRunner`，**不使用 `Engine` 或 `Scheduler`**。它的职责单一：为一批请求完成全序列 Prefill，并将 KV Cache 提取出来供传输。

### `prefill_batch`

```python
def prefill_batch(self, groups: List[SequenceGroup]) -> Tuple[List[int], List[List[int]]]
```

批量 Prefill 的核心方法：

1. **逐请求分配 Block**：调用 `block_manager.allocate(group)`，为每条序列分配物理块并建立 `block_table`。
2. **拼接输入张量**：将所有请求的 token_ids 和 positions 拼接为形状 `[1, total_tokens]` 的张量（与 `Engine` 的 mixed batch 格式相同）。
3. **设置 `_current_context`**：向 `ModelRunner` 注入 `num_prefill_tokens / num_decode_tokens / prefills / decodes` 等上下文，使 Monkey Patching 后的 Attention 层能正确定位每条序列的 `block_table`。
4. **前向传播**：调用 `runner.model(input_ids, position_ids, use_cache=False)`，所有层通过 `paged_forward` 闭包将 K/V 写入 KV Cache，最终输出所有 token 的 logits。
5. **提取首 token**：对每条序列取其最后一个 token 的 logits，经 `top_k_sample` 采样得到第一个输出 token；更新 `seq.num_computed_tokens` 和 `seq.output_token_ids`。

**注意**：`PrefillWorker` 没有 Chunked Prefill 逻辑——它对每条序列做**全序列一次性 Prefill**。对于极长 Prompt，这会导致单次前向传播耗时很长，期间 Decode Worker 只能等待。

### `prefill_batch_and_extract`

```python
def prefill_batch_and_extract(self, groups) -> List[Tuple[int, List[int], PinnedKVBuffer, Tensor, Tensor]]
```

在 `prefill_batch` 基础上，逐条提取 KV Cache 到 Pinned Memory，返回五元组：
`(first_token, block_table, pinned_buf, k_cache, v_cache)`

其中 `k_cache / v_cache` 是 Prefill GPU 上的完整缓存张量（`transfer_kv` 在 P2P 时直接从这里读取）。最后调用 `torch.cuda.synchronize()` 确保所有 D2H 拷贝完成。

**已知问题**：无论 P2P 是否可用，`extract_kv_to_pinned` 都会被**无条件执行**。在 NVLink/NVSwitch 互联的服务器卡上，P2P 可用时实际传输会走直接 GPU→GPU 路径，pinned buffer 根本不会被读取——但 D2H 拷贝已经白做了一遍。受 PCIe 带宽限制（~64 GB/s），这次多余的 D2H 拷贝引入的延迟与 P2P 传输本身相当，且在函数返回之前就已阻塞。修法是在 extract 之前先探测 `_check_p2p`，P2P 路径直接跳过 pinned 提取和 `synchronize`。

### `prefill` / `prefill_and_extract`

单请求的便捷接口，分别封装 `prefill_batch([group])` 和 `prefill_batch_and_extract([group])`。

### `extract_kv`

```python
def extract_kv(self, block_table: List[int]) -> PinnedKVBuffer
```

对已完成 Prefill 的序列独立提取 KV，适合需要将 Prefill 和 Extract 解耦的场景。

---

## DecodeWorker

```python
class DecodeWorker:
    def __init__(self, model_path, gpu_id, block_manager, block_size=16, max_blocks=512)
```

`DecodeWorker` 同样直接持有 `ModelRunner`，不使用 `Engine`。它管理两个队列：

- `self.running`：当前正在 Decode 的 `SequenceGroup` 列表
- `self._pending`：KV Transfer 尚未完成的 `_PendingTransfer` 列表
- `self.finished`：已完成的序列

两个 CUDA Stream：
- `compute_stream`：执行 Decode 前向传播
- `transfer_stream`：异步接收 KV Cache

### `_PendingTransfer`

```python
class _PendingTransfer:
    group: SequenceGroup
    event: torch.cuda.Event
```

轻量 dataclass，将一个 `SequenceGroup` 和记录 Transfer 完成时刻的 `CUDA Event` 绑定在一起。

### `receive_kv_async`

```python
def receive_kv_async(self, group, block_table, buf, src_k=None, src_v=None)
```

在 `transfer_stream` 上异步发起 KV 传输（调用 `transfer_kv`），然后在 `transfer_stream` 上 record 一个 `Event`，将 `(group, event)` 包装为 `_PendingTransfer` 放入 `self._pending`。

调用方在调用此函数后立刻返回，**不会阻塞**。实际拷贝在后台 Stream 上进行。

### `_promote_ready`

```python
def _promote_ready(self)
```

遍历 `_pending`，对每个 `_PendingTransfer` 调用 `event.query()`（非阻塞 GPU 轮询）：

- 若 Event 已完成 → 将 `group` 移入 `self.running`
- 否则 → 保留在 `_pending`

此方法在每次 `step()` 开始时调用，实现 **Transfer 与 Decode 的流水线重叠**：当 Decode Worker 正在处理现有序列时，新请求的 KV Transfer 在 `transfer_stream` 上并发进行；下一个 `step()` 到来时，已完成传输的请求会被自动提升进入 Decode 批次。

### `step`

```python
def step(self) -> List[Tuple[SequenceGroup, int]]
```

一次 Decode 步骤，完整流程：

1. **`_promote_ready()`**：将传输完成的请求提升到 `running`。
2. **构建输入**：对 `running` 中每条序列，调用 `block_manager.append_slot(seq)` 申请下一个 KV 槽位，取 `seq.output_token_ids[-1]` 作为当前输入 token，记录 `position = seq.num_computed_tokens`。
3. **设置 `_current_context`**：`num_decode_tokens = len(running)`，`prefills=[]`，注入所有 Decode 序列的 `block_table` 和 `position`。
4. **前向传播**（在 `compute_stream` 上）：`runner.model(input_ids, position_ids)`，输出形状 `[1, B, vocab_size]` 的 logits。
5. **采样**：对每条序列取对应位置的 logits，`top_k_sample` 得到下一个 token，追加到 `seq.output_token_ids`，`num_computed_tokens += 1`。
6. **EOS 检测**：若采到 EOS，将序列状态改为 `FINISHED_STOPPED`，从 `running` 移除，调用 `block_manager.free(seq)` 释放物理块，移入 `self.finished`。完成后再调一次 `_promote_ready()`，让等待中的请求立即填充空出的位置。

---

## 整体数据流

### Collocated 路径

```
add_request(prompt)
    → Scheduler.waiting
    → step(): Scheduler 混排 Prefill + Decode tokens
    → ModelRunner.run_batch()
        → paged_kv_store kernel  写入 KV Cache
        → paged_attention_forward kernel  读取 KV Cache
    → 采样 → 返回 token
```

### Disaggregated 路径

```
PrefillWorker                        DecodeWorker
─────────────────────────────────    ─────────────────────────────────
prefill_and_extract(group)
  ├─ block_manager.allocate(group)
  ├─ ModelRunner.model(...)           
  │    └─ paged_kv_store kernel      
  ├─ extract_kv_to_pinned(...)       
  └─ return (first_token,            
             block_table,            
             pinned_buf,             receive_kv_async(group, ...)
             k_cache, v_cache)  ──►     ├─ transfer_kv(..., transfer_stream)
                                        │    └─ P2P 或 Pinned Relay
                                        └─ record Event → _pending

                                     step()
                                       ├─ _promote_ready()  [event.query()]
                                       ├─ append_slot(seq)
                                       ├─ ModelRunner.model(...)
                                       │    └─ paged_attention_forward kernel
                                       └─ top_k_sample → token
```

**共享 `BlockSpaceManager`**：两个 Worker 操作同一个 `block_manager` 实例。`PrefillWorker` 调用 `allocate`，`DecodeWorker` 调用 `append_slot` 和 `free`，块号在两侧含义一致（但各自 GPU 上的 KV Cache 张量是独立的——传输就是为了将块内容同步过去）。

---

## Disaggregated 路径的吞吐局限

Disaggregated 路径在当前实现中存在若干结构性限制，导致其吞吐量上限远低于 Collocated 路径：

**1. 无 Continuous Batching**

`DecodeWorker.step()` 只做纯 Decode，`PrefillWorker` 只做纯 Prefill，两者**不混批**。Collocated 路径的核心优势——Prefill token 和 Decode token 在同一次前向传播中共享计算——在 Disagg 路径中不存在。

具体影响：`DecodeWorker` 的每一步 forward pass 只含 `B` 个 decode token（通常 B 很小），计算量极低，GPU 的矩阵乘法单元大量空转（decode-only batch 的计算强度远低于 prefill+decode 混合 batch）。

**2. 无 Chunked Prefill**

`PrefillWorker.prefill_batch` 对每条序列做**全序列一次性 Prefill**。长 Prompt（如 4096 token）会占用一整步甚至更多时间，期间 Decode Worker 无法立即获得新序列的 KV，等待队列累积。

**3. KV Transfer 额外开销**

每条请求完成 Prefill 后，需要将 KV Cache 从 Prefill GPU 传输到 Decode GPU（P2P 或经 Pinned Memory 中转）。以 RTX 4090 实测约 12.9 GB/s 的跨卡带宽，传输 2048-token 的 KV 约需 ~23 ms，这段时间 Decode Worker 对该请求无法产出任何 token。

cost_model 中的路由分析表明，即使在有干扰负载（load=8）的场景下，对于较短序列（L≈128），Disagg 相对 Collocated 的优势也十分有限（~0.032 ms/token），而当序列较长时差距才会拉开。换言之，当前 Disagg 路径的实际增益主要体现在 Decode 阶段不受 Prefill 干扰这一点上，但吞吐量本身因缺乏 Continuous Batching 而受限。

**4. `prefill_batch_and_extract` 中的同步屏障**

`prefill_batch_and_extract` 最后调用了 `torch.cuda.synchronize(self.device)`，确保所有 D2H 拷贝完成后才返回。这意味着 Prefill 侧在一批请求完全提取完 KV 之前无法处理下一批，存在串行瓶颈。

**小结**：若要提升 Disagg 路径的吞吐量，核心方向是引入一个独立的调度器驱动 `DecodeWorker` 批量处理多条 Decode 序列，同时在 `PrefillWorker` 侧加入 Chunked Prefill 支持。当前实现作为正确性验证的原型是完整的，但不适合作为高吞吐量 Disagg 系统的生产路径。

---

## 测试：test_disaggregated.py

`workers/test/test_disaggregated.py` 提供三个集成测试，均直接操作 Worker 类：

| 测试 | 内容 |
|---|---|
| `test_single_request` | 单请求全链路：Prefill → KV Extract → Transfer → Decode N 步，打印每步耗时 |
| `test_overlap` | 重叠测试：Req-0 Decode 时异步传输 Req-1 KV，验证 `_promote_ready` 在正确步骤提升 Req-1 |
| `test_collocated` | Collocated 基线对比：用同一个 prompt 跑 `CollocatedWorker`，验证输出一致性 |

运行方式：
```bash
python workers/test/test_disaggregated.py \
    --model Qwen/Qwen3-8B \
    --prefill-gpu 1 --decode-gpu 2
```
