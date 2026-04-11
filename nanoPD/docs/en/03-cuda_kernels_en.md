# CUDA Kernels Module

> Source files: `paged_attention/csrc/`

---

## Overview

This module provides two CUDA kernels that form the GPU-side core of Paged Attention:

| Kernel | Source directory | Responsibility |
|---|---|---|
| `paged_kv_store` | `csrc/kvstore/` | Write the newly computed K/V for the current step into the corresponding physical blocks of the KV Cache |
| `paged_attention_forward` | `csrc/attention/` | Decode phase: read historical KV from non-contiguous physical blocks and compute the Attention output |

Both kernels are bound as the Python module `paged_kernels` via `torch::extension`, dispatched with `AT_DISPATCH_FLOATING_TYPES_AND2` to support `float16` and `bfloat16`.

---

## paged_kv_store

### Purpose

After each inference step, write the newly computed K/V vectors — for both prefill and decode sequences — into the correct slots of the global KV Cache tensor, indexed by each token's absolute `position`.

### Tensor shapes

```
k_cache / v_cache : (max_blocks, num_kv_heads, block_size, head_dim)
k_src   / v_src   : (num_kv_heads, num_tokens, head_dim)   ← new KV for all tokens this step
block_tables      : (num_tokens, max_blocks_per_seq)
positions         : (num_tokens,)                           ← absolute position of each token
```

### Launch configuration

```
grid  = (num_tokens, num_kv_heads)
block = (head_dim / 4)
```

One CUDA block handles the write for one `(token, kv_head)` pair.

### Implementation

Each thread handles 4 fp16 elements by reinterpreting the `half*` pointers as `float2*` for vectorized loads and stores (`float2` = 8 bytes = 4 × fp16):

```
position     = positions[seq_id]
block_idx    = position / block_size          # logical block index
block_offset = position % block_size          # offset within block

physical_block = block_tables[seq_id * max_blocks_per_seq + block_idx]

src_idx = head_id * num_tokens * (head_dim/4) + seq_id * (head_dim/4) + tid
dst_idx = physical_block * num_kv_heads * block_size * (head_dim/4)
        + head_id        * block_size * (head_dim/4)
        + block_offset   * (head_dim/4)
        + tid

k_cache_f2[dst_idx] = k_src_f2[src_idx]
v_cache_f2[dst_idx] = v_src_f2[src_idx]
```

The logic is straightforward: look up the physical address via the block table, then write K and V with a single vectorized store per thread. K and V are written sequentially within the same thread; no synchronisation is needed.

### Why a dedicated kernel?

Without this kernel, the equivalent operation in Python using a PyTorch loop would look like:

```python
for i, (position, block_table) in enumerate(zip(positions, block_tables)):
    block_idx    = position // block_size
    block_offset = position % block_size
    phys_block   = block_table[block_idx]
    k_cache[phys_block, :, block_offset, :] = k_src[:, i, :]
    v_cache[phys_block, :, block_offset, :] = v_src[:, i, :]
```

This loop executes serially on the Python side. Each indexed assignment compiles down to a separate GPU kernel launch, and the accumulated launch overhead across all tokens in a batch makes this extremely slow — around **200 ms/step** in practice.

With this kernel, all tokens' writes happen in a single launch in parallel, dropping the cost to around **50 ms/step** — a **~4× speedup**.

---

## paged_attention_forward

### Purpose

The core compute of the decode phase. For every decode sequence and every Q head, reads the complete historical KV from physical blocks that may be scattered non-contiguously in the KV Cache, computes Scaled Dot-Product Attention, and outputs the `head_size`-dimensional attention vector for that head.

### Tensor shapes

```
out          : (num_seqs, num_heads, head_size)
query        : (num_seqs, num_heads, head_size)
k_cache      : (max_blocks, num_kv_heads, block_size, head_size)
v_cache      : (max_blocks, num_kv_heads, block_size, head_size)
block_tables : (num_seqs, max_blocks_per_seq)
seq_lens     : (num_seqs,)
```

### Launch configuration

```
grid  = (num_seqs, num_heads)
block = (NUM_WARPS * WARP_SIZE) = (4 * 32) = 128 threads
shared memory = block_size * max_blocks_per_seq * sizeof(scalar_t)
              ≈ all attention scores for one full sequence
```

One CUDA block is responsible for one `(seq, head)` pair and produces `out[seq, head, :]`.

### Implementation (three phases)

The current version is a naive implementation that separates the computation into three distinct phases.

**Phase 1: Compute all QK dot products, store in Shared Memory**

The 4 warps in the block work through K tokens in parallel:

```
for each physical block block_id:
    K_cache_ptr = k_cache[physical_block][kv_head]
    for tokens in block, stepping by NUM_WARPS:
        warp_in_dotmul(sdata[abs_token_pos], Q, K[token], head_size, lane_id, scale)
```

`warp_in_dotmul` computes a `head_size`-dimensional dot product using one warp (32 threads):
- Each thread accumulates `head_size / 32` elements;
- Warp-level reduction via `__shfl_down_sync`;
- Lane 0 writes the result (multiplied by `scale`) to Shared Memory.

After this phase, `sdata[0..seq_len-1]` holds the scaled QK scores for all historical tokens.

**Phase 2: Online Softmax normalisation**

```
for token in range(seq_len):          # serial
    val       = sdata[token]
    l_max_new = max(l_max, val)
    m_sum     = m_sum * exp(l_max - l_max_new) + exp(val - l_max_new)
    l_max     = l_max_new

# normalise (all threads in parallel)
for each token:
    sdata[token] = exp(sdata[token] - l_max) / m_sum
```

Online softmax avoids a two-pass scan by maintaining a running max and denominator, but in this implementation the running update is **executed serially by a single thread** — only 1 of the 128 threads does useful work here.

**Phase 3: Weighted V accumulation**

```
for i in range(seq_len):              # serial
    p = sdata[i]                      # softmax weight
    physical_block = block_tables[i // block_size]
    V_ptr = v_cache[physical_block][kv_head][i % block_size]
    for h in 0..head_size/blockDim.x:  # threads split head dimension
        v_acc[h] += p * V_ptr[tid + h * blockDim.x]

out[seq, head, :] = v_acc
```

All threads share the `head_size` dimension in parallel, but the `seq_len` dimension is traversed serially. Each token requires one indirect Global Memory access via `block_table`.

---

## Performance limitations (naive version)

The current implementation is correct but has several notable inefficiencies:

**1. Shared Memory grows linearly with sequence length**

The shared memory allocation is `block_size × max_blocks_per_seq × sizeof(scalar_t)` — equal to storing every attention score for the full sequence. For long sequences (e.g. block_size=16, max_blocks=512, giving 8192 tokens), this is 16 KB, which directly reduces SM occupancy.

**2. The online softmax reduction is serial**

The running max / sum update in phase 2 is executed sequentially by a single thread. 127 of the 128 threads sit idle.

**3. V accumulation has uncoalesced global memory access**

Phase 3 accesses V one token at a time via indirect `block_table` addressing. The resulting physical addresses are non-contiguous, preventing L2 coalescing.

**4. NUM_WARPS is hardcoded to 4**

No tuning is done based on `head_size` or SM resources. 4 warps is not optimal for head sizes of 128 or 256.

**5. Three separate phases prevent fusion**

QK computation, softmax, and V accumulation are three separate passes. Intermediate results must be written back to and re-read from Shared Memory between passes.

---

## Deprecated optimised version (paged_attention_optimized)

`csrc/paged_attention_optimized/` contains an experimental optimised version that has been deprecated and is not integrated into the main code path. It is kept for reference. Key structural improvements over the naive version:

- **Partition parallelism**: a `blockIdx.z = partition_id` dimension is added to the grid, splitting each long sequence into `PARTITION_SIZE`-sized chunks that are processed independently and in parallel. A separate `paged_attention_reduce_kernel` merges the per-partition `(partial_out, exp_sum, max_logit)` outputs into the final result. This follows the same approach as vLLM's v2 Paged Attention kernel and significantly reduces per-block compute and Shared Memory pressure for long sequences.
- **Q preloaded into registers**: `q_reg[HEAD_SIZE / WARP_SIZE]` is loaded once at kernel entry, avoiding repeated Global Memory reads in the token loop.
- **Single-pass fused online softmax + V accumulation**: QK dot product, softmax update, and V accumulation are fused into a single token loop — no need to store all QK scores in Shared Memory for a second pass.
- **Compile-time HEAD_SIZE template**: `template<int HEAD_SIZE>` allows the compiler to unroll inner loops and eliminate loop-control overhead.

This version was shelved due to integration complexity (it requires an extra reduce kernel and intermediate buffer allocation), but it serves as a natural starting point for future performance work.
