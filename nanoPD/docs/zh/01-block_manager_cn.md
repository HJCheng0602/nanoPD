# Block Manager 模块文档

> 对应源码：`block_manager/sequence.py`、`block_manager/block_manager.py`

---

## 模块概述

Block Manager 是 **nanoPD** 中负责 **KV-Cache 物理内存管理** 的核心模块。其核心思想来自论文 *PagedAttention*：将 GPU 显存切分为固定大小的Block，通过逻辑地址到物理地址的映射表（Block Table）来管理每条序列的 KV-Cache，从而实现：

1. **消除内存碎片**：不再为每条序列预留连续的最大长度显存；
2. **Copy-on-Write (CoW)**：Beam Search / Prefix Sharing 场景下多条序列可共享同一物理块，写时才拷贝；（本实现中未包含该功能）
3. **动态扩展**：Decode 阶段按需追加新块，无需提前分配。

模块由两个文件组成：

| 文件 | 职责 |
|---|---|
| `sequence.py` | 序列的逻辑层抽象：Token、逻辑块、序列、序列组 |
| `block_manager.py` | 物理层管理：物理块分配器、块空间管理器 |

---

## sequence.py

### SequenceStatus

```python
class SequenceStatus(Enum):
    WAITING          # 等待被调度，尚未分配物理块
    RUNNING          # 已分配物理块，正在推理
    FINISHED_STOPPED # 正常结束（命中 EOS）
    FINISHED_ABORTED # 异常结束（被抢占/显存不足后丢弃）
```

描述一条序列在其生命周期中所处的阶段。`BlockSpaceManager.free()` 会将序列强制设置为 `FINISHED_ABORTED`；正常停止需要调用方在外部将状态置为 `FINISHED_STOPPED`。

---

### LogicalTokenBlock

```python
@dataclass
class LogicalTokenBlock:
    block_num  : int        # 逻辑块编号，从 0 递增
    block_size : int        # 每块能容纳的最大 token 数
    token_ids  : List[int]  # 当前块中已存放的 token id 列表
```

**代表序列的一个逻辑内存页**，只存在于 CPU 侧，不直接对应 GPU 显存。

#### 属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `num_tokens` | `int` | 已存放的 token 数量，即 `len(token_ids)` |
| `is_full` | `bool` | 当前块是否已满（`num_tokens == block_size`） |
| `num_empty_slots` | `int` | 还能追加多少个 token |

#### 方法

```python
def append_token(self, token_id: int)
```

向块中追加一个 token id。若块已满，会触发 `assert` 异常。调用前应先检查 `is_full`。

#### 实现原理

`LogicalTokenBlock` 是一个纯数据容器，不负责任何分配逻辑。它由 `Sequence` 统一管理——当最后一个块满了，`Sequence` 会新建一个 `LogicalTokenBlock` 并追加到列表中。

---

### Sequence

```python
class Sequence:
    def __init__(self, seq_id: int, prompt_token_ids: List[int], block_size: int)
```

**代表一次完整的推理序列**，包含 Prompt 阶段的输入 token 以及 Decode 阶段逐步生成的 token，同时维护其在逻辑块层面的视图。

#### 初始化过程

1. 将 `prompt_token_ids` 中的每个 token 逐一调用 `_append_token_id_to_blocks()`，在此过程中按需创建 `LogicalTokenBlock`；
2. 此时 `output_token_ids` 为空，`num_computed_tokens` 为 0（表示尚未进行任何一步前向计算）。

#### 内部字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `seq_id` | `int` | 序列唯一标识符 |
| `status` | `SequenceStatus` | 当前状态，初始为 `WAITING` |
| `block_size` | `int` | 逻辑块大小，与全局配置保持一致 |
| `prompt_len` | `int` | Prompt 的 token 数量 |
| `output_token_ids` | `List[int]` | Decode 阶段生成的 token id 列表 |
| `logical_token_blocks` | `List[LogicalTokenBlock]` | 逻辑块列表 |
| `num_computed_tokens` | `int` | 已经完成前向计算的 token 数（由调度器在外部更新） |

#### 私有方法

```python
def _append_new_logical_block(self)
```

新建一个 `LogicalTokenBlock`（编号为当前块列表长度），追加到 `logical_token_blocks`。

```python
def _append_token_id_to_blocks(self, token_id: int)
```

核心的 token 追加逻辑：若块列表为空或最后一个块已满，则先调用 `_append_new_logical_block()` 创建新块，再将 token 写入最后一个块。

#### 公开方法

```python
def append_token_id(self, token_id: int)
```

Decode 阶段的接口，在调用 `_append_token_id_to_blocks()` 的同时，也将 token id 追加到 `output_token_ids`，维护完整的输出记录。

#### 公开属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `num_logic_blocks` | `int` | 当前逻辑块总数，即 `⌈(prompt_len + output_len) / block_size⌉` |
| `last_token_id` | `int` | 最后一个生成的 token id（仅 Decode 阶段有效，否则会 IndexError） |
| `total_len` | `int` | 已有 token 总数（`prompt_len + len(output_token_ids)`） |
| `is_finished` | `bool` | 序列是否已结束（包含正常停止和异常中断两种状态） |
| `is_prefill_done` | `bool` | Prefill 是否完成（`num_computed_tokens >= prompt_len`） |

---

### SequenceGroup

```python
class SequenceGroup:
    def __init__(self, request_id: str, seqs: List[Sequence])
```

**将同一请求产生的多条序列聚合为一个组**。在普通贪心解码场景下，一个 Group 只有一条序列；在 Beam Search 场景下，同一 Group 包含 `beam_width` 条并行序列，它们共享相同的 Prompt，可以通过 `fork()` 复用物理块。

#### 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `request_id` | `str` | 请求的唯一标识符 |
| `seqs` | `List[Sequence]` | 属于该请求的所有序列 |

#### 方法

```python
def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]
```

按状态过滤序列。不传参数时返回全部序列；传入 `SequenceStatus.RUNNING` 时只返回运行中的序列，常用于 `BlockSpaceManager` 中判断需要多少空闲块。

#### 属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `num_seqs` | `int` | 组内序列总数 |
| `is_finished` | `bool` | 组内所有序列是否均已结束 |

---

## block_manager.py

### PhysicalBlock

```python
@dataclass
class PhysicalBlock:
    block_num : int      # 物理块编号，对应 GPU KV-Cache 显存中的页索引
    ref_cout  : int = 0  # 引用计数（注：字段名拼写为 ref_cout，疑为笔误）
```

**GPU 物理内存页的抽象**。`block_num` 是 Attention Kernel 在访问 KV-Cache 时使用的实际下标。`ref_cout` 用于实现引用计数式的 CoW 语义：

- 当 `ref_cout == 1` 时，该块只被一条序列持有，可以直接写入；
- 当 `ref_cout > 1` 时，该块被多条序列共享（通过 `fork`），写入前必须先复制（CoW）。

---

### BlockAllocator

```python
class BlockAllocator:
    def __init__(self, num_blocks: int)
```

**管理一个 GPU 块池的底层分配器**，内部使用一个 `List[PhysicalBlock]` 作为空闲块栈（LIFO）。

#### 初始化

预先创建 `num_blocks` 个 `PhysicalBlock` 对象（编号 0 到 `num_blocks-1`），全部放入 `_free_blocks` 列表。使用栈（`list.pop()`）而非队列，有助于局部性：最近释放的块优先被再次分配，减少 GPU 显存的冷访问。

#### 方法

```python
def allocate(self) -> PhysicalBlock
```

从栈顶弹出一个空闲块，将其 `ref_cout` 设为 1 后返回。若空闲块为空，抛出 `MemoryError("OOM : no free physical blocks")`。

```python
def free(self, block: PhysicalBlock)
```

将 `ref_cout` 减 1。当 `ref_cout` 降至 0 时，将块重新压回 `_free_blocks` 栈（归还到池中）。若 `ref_cout` 已为 0 时调用，会触发 `assert` 异常（防止 double-free）。

#### 属性

| 属性 | 类型 | 含义 |
|---|---|---|
| `num_free_blocks` | `int` | 当前空闲块数量 |
| `num_total_blocks` | `int` | 总块数（含已分配和空闲） |

---

### BlockSpaceManager

```python
class BlockSpaceManager:
    def __init__(self, block_size: int, num_gpu_blocks: int)
```

**整个模块最核心的类**，维护每条序列的块表（Block Table），对外暴露调度器所需的全部内存管理接口。

#### 初始化

| 字段 | 含义 |
|---|---|
| `block_size` | 每个物理/逻辑块能容纳的 token 数 |
| `allocator` | 内部持有的 `BlockAllocator` 实例 |
| `_block_table` | `Dict[seq_id, List[PhysicalBlock]]`，从序列 id 到物理块列表的映射 |

---

#### can_allocate / allocate —— Prefill 阶段分配

```python
def can_allocate(self, seq_group: SequenceGroup) -> bool
```

**检查是否有足够的空闲块来为该 Group 做 Prefill 分配。**

实现：取 Group 中第一条序列（Prefill 时通常只有一条），比较其 `num_logic_blocks` 与 `allocator.num_free_blocks`。

> **注意**：当前实现只检查 Group 中第一条序列的需求，在多序列 Group（Beam Search）的初始化场景下可能低估实际需求。

```python
def allocate(self, seq_group: SequenceGroup)
```

**为 Group 中所有 `WAITING` 状态的序列分配物理块，并将其状态切换为 `RUNNING`。**

实现流程：
1. 遍历所有 `WAITING` 序列；
2. 对每条序列，根据其 `num_logic_blocks` 调用 `allocator.allocate()` 相应次数，构建物理块列表；
3. 将物理块列表写入 `_block_table[seq.seq_id]`；
4. 将序列状态更新为 `RUNNING`。

---

#### can_append_slot / append_slot —— Decode 阶段扩块

```python
def can_append_slot(self, seq_group: SequenceGroup) -> bool
```

**检查是否有足够的空闲块，为 Group 中所有 `RUNNING` 序列各预留一个可能需要的新块。**

实现：在最坏情况下，每条 `RUNNING` 序列都需要新分配一个块（上一块恰好写满），因此需要 `num_running` 个空闲块。

```python
def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]
```

**Decode 步骤中，为单条序列写入下一个 token 前确保有可用 slot，并在必要时触发 CoW。**

参数为单条序列（调用方需遍历所有 `RUNNING` 序列逐一调用）。

实现逻辑：

```
next_position = seq.num_computed_tokens
needed_blocks = next_position // block_size + 1

if 当前物理块数 < needed_blocks:
    # 需要新块（上一块已写满）
    allocate 一个新块追加到 block_table
    return None   # 无 CoW

last_block = block_table[-1]
if last_block.ref_cout > 1:
    # 最后一块被多个序列共享 → CoW
    new_block = allocate()
    free(last_block)          # 旧块引用计数 -1
    block_table[-1] = new_block
    return (last_block.block_num, new_block.block_num)  # 通知 kernel 做数据拷贝

return None  # 最后一块未满且未共享，直接写入即可
```

返回值语义：
- `None`：无需额外操作，正常写入；
- `(old_num, new_num)`：发生了 CoW，调用方（通常是 Model Runner）需要在 GPU 端将 `old_num` 块的内容拷贝到 `new_num` 块。

> **重要说明**：`next_position` 使用的是 `seq.num_computed_tokens` 而非 `seq.total_len - 1`。这意味着 `num_computed_tokens` 必须在每步 Decode 的前向计算完成后由调度器及时更新，否则 `needed_blocks` 的计算会不准确。

---

#### fork —— Beam Search / Prefix Sharing

```python
def fork(self, parent: Sequence, child: Sequence)
```

**从父序列派生出子序列，两者共享相同的物理块（零拷贝）。**

实现：
1. 浅拷贝父序列的块列表 `list(parent_table)`（列表结构独立，但 `PhysicalBlock` 对象共享）；
2. 对列表中每个 `PhysicalBlock` 的 `ref_cout` 加 1；
3. 将新列表注册到 `_block_table[child.seq_id]`。

`fork` 本身不消耗任何新的物理块。共享块在有任意一方写入时，由 `append_slot` 触发 CoW，完成懒拷贝。

---

#### free —— 释放序列资源

```python
def free(self, seq: Sequence)
```

**释放一条序列持有的所有物理块，并将序列状态标记为 `FINISHED_ABORTED`。**

实现：
1. 若 `seq.seq_id` 不在 `_block_table` 中，直接返回（幂等设计，防止重复释放崩溃）；
2. 弹出并遍历该序列的块列表，对每个 `PhysicalBlock` 调用 `allocator.free()`；
3. 在引用计数语义下，共享块（`ref_cout > 1`）不会立即归还池，只有最后一个持有者释放时才真正回池；
4. 强制将序列状态设为 `FINISHED_ABORTED`。

---

#### get_block_table —— 向 Kernel 传递块表

```python
def get_block_table(self, seq: Sequence) -> List[int]
```

将序列的物理块列表转为纯整数列表（只取 `block_num`），供 Attention Kernel 使用。Kernel 通过这张表将逻辑块索引转换为 GPU 显存中的实际地址。

---

#### 属性

```python
@property
def num_free_blocks(self) -> int
```

代理到内部 `allocator.num_free_blocks`，供调度器判断全局剩余显存。

---

## 数据流与生命周期

```
请求到来
  │
  ▼
SequenceGroup 创建（含一或多条 WAITING Sequence）
  │
  ▼
调度器调用 can_allocate() → 检查空闲块是否足够
  │
  ├─ 不足 → 请求继续等待（或抢占其他序列）
  │
  └─ 足够 → allocate()
              为每条 WAITING Sequence 分配物理块
              Sequence.status → RUNNING
  │
  ▼
Prefill 前向计算
  num_computed_tokens += prompt_len（调度器更新）
  │
  ▼
Decode 循环：
  ┌───────────────────────────────────────────────── ┐
  │  can_append_slot() → 检查是否有块可追加             │
  │  append_slot(seq)                                │
  │    → None：直接写入                               │
  │    → (old, new)：GPU 端执行 CoW 拷贝              │
  │  模型前向计算，生成下一个 token                    │
  │  seq.append_token_id(token_id)                  │
  │  num_computed_tokens += 1（调度器更新）           │
  └─────────────────────────────────────────────────┘
  │
  ▼
序列结束（EOS 或长度上限）
  free(seq) → 归还物理块，ref_cout 计数驱动实际回池
```

---

## 各类关系总览

```
SequenceGroup
  └── List[Sequence]
        ├── status: SequenceStatus
        ├── logical_token_blocks: List[LogicalTokenBlock]
        ├── output_token_ids: List[int]
        └── num_computed_tokens: int

BlockSpaceManager
  ├── allocator: BlockAllocator
  │     └── _free_blocks: List[PhysicalBlock]  ← 栈
  └── _block_table: Dict[seq_id, List[PhysicalBlock]]
                                  ↑
                         可被多条序列共享（fork CoW）
```

---

## 测试覆盖概览（test_block_manager.py）

| 测试类 | 覆盖场景 |
|---|---|
| `TestAllocateFree` | 基础分配/释放，块计数，状态变更，块表长度 |
| `TestOOM` | `can_allocate` 边界，强制分配触发 OOM，`can_append_slot` 满池检测 |
| `TestAppendSlot` | 块内追加无需新块，块满后新分配，连续 decode 多步 |
| `TestForkAndCoW` | fork 共享不消耗块，共享写触发 CoW，free 后引用计数递减 |
| `TestFullLifecycle` | 完整 prefill→decode→free 后块池完全归还，块号合法性 |
