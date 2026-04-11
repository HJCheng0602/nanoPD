# CUDA Kernels 模块文档

> 对应源码：`paged_attention/csrc/`

---

## 模块概述

本模块提供两个 CUDA Kernel，构成 Paged Attention 在 GPU 侧的核心计算：

| Kernel | 源文件 | 职责 |
|---|---|---|
| `paged_kv_store` | `csrc/kvstore/` | 将当前步新计算的 K/V 写入 KV Cache 的对应物理块 |
| `paged_attention_forward` | `csrc/attention/` | Decode 阶段，从非连续物理块中读取历史 KV，计算 Attention 输出 |

两个 Kernel 均通过 `torch::extension` 绑定为 Python 模块 `paged_kernels`，以 `AT_DISPATCH_FLOATING_TYPES_AND2` 分发，支持 `float16` 和 `bfloat16`。

---

## paged_kv_store

### 功能

每个推理步结束后，将本步为 Prefill 或 Decode 序列新计算出的 K/V 向量，按各自的 `position` 写入全局 KV Cache 张量中对应的物理块槽位。

### 张量形状约定

```
k_cache / v_cache : (max_blocks, num_kv_heads, block_size, head_dim)
k_src   / v_src   : (num_kv_heads, num_tokens, head_dim)   ← 本步所有 token 的新 KV
block_tables      : (num_tokens, max_blocks_per_seq)
positions         : (num_tokens,)                           ← 每个 token 在其序列中的绝对位置
```

### 启动配置

```
grid  = (num_tokens, num_kv_heads)
block = (head_dim / 4)
```

一个 CUDA Block 负责一个 `(token, kv_head)` 对的写入。

### 实现思路

每个线程负责 4 个 fp16 元素（通过将 `half*` 重新解释为 `float2*` 来向量化读写，`float2` = 8 字节 = 4 个 fp16）：

```
position     = positions[seq_id]
block_idx    = position / block_size          # 逻辑块号
block_offset = position % block_size          # 块内偏移

physical_block = block_tables[seq_id * max_blocks_per_seq + block_idx]

src_idx = head_id * num_tokens * (head_dim/4) + seq_id * (head_dim/4) + tid
dst_idx = physical_block * num_kv_heads * block_size * (head_dim/4)
        + head_id        * block_size * (head_dim/4)
        + block_offset   * (head_dim/4)
        + tid

k_cache_f2[dst_idx] = k_src_f2[src_idx]
v_cache_f2[dst_idx] = v_src_f2[src_idx]
```

逻辑非常直接：查表定位物理地址，向量化写入。K 和 V 在同一个线程内串行完成，无需额外同步。

### 为什么需要独立的 Kernel？

如果不用 CUDA Kernel，在 Python 侧用 PyTorch 循环实现同等功能，伪代码如下：

```python
for i, (position, block_table) in enumerate(zip(positions, block_tables)):
    block_idx    = position // block_size
    block_offset = position % block_size
    phys_block   = block_table[block_idx]
    k_cache[phys_block, :, block_offset, :] = k_src[:, i, :]
    v_cache[phys_block, :, block_offset, :] = v_src[:, i, :]
```

这个循环在 Python 层串行执行，每次 index 赋值都会触发一次独立的 GPU kernel launch，launch overhead 累积到 batch 内所有 token 上，实测耗时约 **200 ms/step**。

改用本 CUDA Kernel 后，所有 token 的写入在一次 kernel launch 中并行完成，实测降至约 **50 ms/step**，**提速约 4×**。

---

## paged_attention_forward

### 功能

Decode 阶段的核心计算。对每条 Decode 序列的每个 Q head，从分散在非连续物理块中的 KV Cache 里读取完整历史 KV，完成 Scaled Dot-Product Attention，输出该 head 的 `head_size` 维向量。

### 张量形状约定

```
out          : (num_seqs, num_heads, head_size)
query        : (num_seqs, num_heads, head_size)
k_cache      : (max_blocks, num_kv_heads, block_size, head_size)
v_cache      : (max_blocks, num_kv_heads, block_size, head_size)
block_tables : (num_seqs, max_blocks_per_seq)
seq_lens     : (num_seqs,)
```

### 启动配置

```
grid  = (num_seqs, num_heads)
block = (NUM_WARPS * WARP_SIZE) = (4 * 32) = 128 threads
shared memory = block_size * max_blocks_per_seq * sizeof(scalar_t)
              ≈ 整条序列的所有 Attention 分数
```

一个 CUDA Block 负责一个 `(seq, head)` 对，输出 `out[seq, head, :]`。

### 实现思路（三阶段）

当前版本是一个naive实现，将计算分为三个独立阶段。

**阶段一：计算所有 QK 点积，存入 Shared Memory**

Block 内的 4 个 warp 并行处理多个 token 的 K：

```
对每个物理块 block_id:
    K_cache_ptr = k_cache[physical_block][kv_head]
    对块内 token 以 NUM_WARPS 为步长滚动:
        warp_in_dotmul(sdata[abs_token_pos], Q, K[token], head_size, lane_id, scale)
```

`warp_in_dotmul` 用一个 Warp（32 线程）计算一个 `head_size` 维的点积：
- 每线程负责 `head_size / 32` 个元素的乘累加；
- 用 `__shfl_down_sync` 做 Warp 内规约；
- lane 0 将结果（已乘 scale）写入 Shared Memory。

本阶段结束后，`sdata[0..seq_len-1]` 存储了所有历史 token 的 QK 分数。

**阶段二：在线 Softmax 归一化**

```
for token in range(seq_len):          # 串行
    val       = sdata[token]
    l_max_new = max(l_max, val)
    m_sum     = m_sum * exp(l_max - l_max_new) + exp(val - l_max_new)
    l_max     = l_max_new

# 归一化（所有线程并行）
for each token:
    sdata[token] = exp(sdata[token] - l_max) / m_sum
```

在线 Softmax 避免了两遍扫描，但当前实现中 running max / sum 的更新是**单线程串行执行**的，128 个线程中只有 1 个在工作。

**阶段三：加权累加 V**

```
for i in range(seq_len):              # 串行
    p = sdata[i]                      # softmax 权重
    physical_block = block_tables[i // block_size]
    V_ptr = v_cache[physical_block][kv_head][i % block_size]
    for h in 0..head_size/blockDim.x: # 线程并行分担 head 维度
        v_acc[h] += p * V_ptr[tid + h * blockDim.x]

out[seq, head, :] = v_acc
```

所有线程并行分担 `head_size` 维度，但 `seq_len` 维度是串行遍历的，每个 token 都需通过 `block_table` 做一次 Global Memory 随机访问。

---

## 性能局限（Naive）

当前实现能够正确运行，但在性能上有若干明显不足：

**1. Shared Memory 用量随序列长度线性增长**

大小为 `block_size × max_blocks_per_seq × sizeof(scalar_t)`，等于整条序列所有 Attention 分数的存储量。对长序列（如 block_size=16、max_blocks=512，即最长 8192 token），这是 16 KB，直接压低 SM 上的 Block 占用率（occupancy）。

**2. 在线 Softmax 的归约是串行的**

阶段二的 running max / sum 更新由单线程顺序执行，128 个线程中只有 1 个在干活，利用率极低。

**3. V 累加的全局内存访问无法合并（uncoalesced）**

阶段三按 token 顺序逐一访问 V，每次通过 `block_table` 间接寻址，访问的物理地址不连续，无法触发 L2 缓存的 coalescing 优化。

**4. NUM_WARPS 硬编码为 4**

未根据 `head_size` 或设备的 SM 资源做任何调优，对 128/256 维 head 并非最优。

**5. 三阶段分离，无法 fuse**

QK 计算、Softmax、V 累加三轮完成，中间结果必须落回 Shared Memory，增加了额外的读写开销。

---

## 废弃的优化版本（paged_attention_optimized）

`csrc/paged_attention_optimized/` 包含一个已被废弃的尝试性优化版本，未集成进主流程，仅供参考。相较于 naive 版，该版本有如下结构改进：

- **分区并行（Partition）**：`grid` 增加 `blockIdx.z = partition_id` 维度，将长序列切分为若干 `PARTITION_SIZE` 大小的分区，各分区独立并行计算局部 softmax；最后由 `paged_attention_reduce_kernel` 合并各分区的 `(partial_out, exp_sum, max_logit)` 为最终输出。这与 vLLM v2 Paged Attention Kernel 的思路一致，可显著缓解长序列下单 Block 的计算和 Shared Memory 压力。
- **Q 预加载到寄存器**：`q_reg[HEAD_SIZE / WARP_SIZE]` 在 kernel 入口一次性加载，避免每次 token 循环重复访问 Global Memory。
- **单遍 Online Softmax + V 融合**：QK 点积、Softmax 更新、V 累加三步在同一个 token 循环中完成，无需将所有 QK 分数暂存在 Shared Memory 后再做第二遍。
- **HEAD_SIZE 编译期模板化**：`template<int HEAD_SIZE>` 让编译器展开内层循环，减少循环控制开销。

该版本因集成复杂度较高（需额外的 reduce kernel 和中间 buffer 分配）而暂时搁置，未来可作为性能优化的起点。
