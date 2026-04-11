# Block Manager Module

> Source files: `block_manager/sequence.py`, `block_manager/block_manager.py`

---

## Overview

The Block Manager is the core KV-Cache memory management module of an LLM inference engine (vLLM-style). Its design is rooted in the *PagedAttention* paper: GPU memory is divided into fixed-size pages (blocks), and a per-sequence mapping table (Block Table) translates logical block indices to physical ones. This enables:

1. **Fragmentation elimination** — no need to pre-reserve contiguous maximum-length memory per sequence;
2. **Copy-on-Write (CoW)** — multiple sequences (Beam Search / Prefix Sharing) can share the same physical block and copy lazily on write;
3. **Dynamic extension** — new blocks are appended on demand during the decode phase.

The module consists of two files:

| File | Responsibility |
|---|---|
| `sequence.py` | Logical layer: tokens, logical blocks, sequences, sequence groups |
| `block_manager.py` | Physical layer: block allocator, block space manager |

---

## sequence.py

### SequenceStatus

```python
class SequenceStatus(Enum):
    WAITING          # Waiting to be scheduled; no physical blocks allocated yet
    RUNNING          # Physical blocks allocated; actively being processed
    FINISHED_STOPPED # Ended normally (EOS token reached)
    FINISHED_ABORTED # Ended abnormally (preempted or OOM)
```

Describes the lifecycle stage of a sequence. `BlockSpaceManager.free()` unconditionally sets the status to `FINISHED_ABORTED`; normal termination must be set by the caller to `FINISHED_STOPPED`.

---

### LogicalTokenBlock

```python
@dataclass
class LogicalTokenBlock:
    block_num  : int        # Logical block index, increments from 0
    block_size : int        # Maximum number of tokens per block
    token_ids  : List[int]  # Token ids stored in this block
```

Represents a **logical memory page** of a sequence. It exists only on the CPU side and does not directly correspond to GPU memory.

#### Properties

| Property | Type | Description |
|---|---|---|
| `num_tokens` | `int` | Number of tokens currently stored (`len(token_ids)`) |
| `is_full` | `bool` | Whether the block has reached capacity (`num_tokens == block_size`) |
| `num_empty_slots` | `int` | Remaining capacity (`block_size - num_tokens`) |

#### Methods

```python
def append_token(self, token_id: int)
```

Appends a single token id to the block. Raises `AssertionError` if the block is already full. Callers should check `is_full` before calling.

#### Design notes

`LogicalTokenBlock` is a pure data container with no allocation logic. `Sequence` owns and drives the lifecycle of all its logical blocks — when the last block fills up, `Sequence` creates a new one and appends it to its list.

---

### Sequence

```python
class Sequence:
    def __init__(self, seq_id: int, prompt_token_ids: List[int], block_size: int)
```

Represents a **complete inference sequence**, holding both the prompt tokens and the tokens generated during decoding, along with a logical-block view of all stored tokens.

#### Initialization

1. Each token in `prompt_token_ids` is fed through `_append_token_id_to_blocks()`, which creates `LogicalTokenBlock` instances on demand;
2. After construction, `output_token_ids` is empty and `num_computed_tokens` is 0 (no forward pass has run yet).

#### Internal fields

| Field | Type | Description |
|---|---|---|
| `seq_id` | `int` | Unique sequence identifier |
| `status` | `SequenceStatus` | Current status; initialized to `WAITING` |
| `block_size` | `int` | Block size, must match the global configuration |
| `prompt_len` | `int` | Number of tokens in the prompt |
| `output_token_ids` | `List[int]` | Token ids generated during decoding |
| `logical_token_blocks` | `List[LogicalTokenBlock]` | Ordered list of logical blocks |
| `num_computed_tokens` | `int` | Tokens for which a forward pass has completed (updated externally by the scheduler) |

#### Private methods

```python
def _append_new_logical_block(self)
```

Creates a new `LogicalTokenBlock` (numbered by the current list length) and appends it to `logical_token_blocks`.

```python
def _append_token_id_to_blocks(self, token_id: int)
```

Core append logic: if the block list is empty or the last block is full, calls `_append_new_logical_block()` first, then writes the token into the last block.

#### Public methods

```python
def append_token_id(self, token_id: int)
```

The decode-phase interface. Calls `_append_token_id_to_blocks()` and additionally appends the token id to `output_token_ids`, maintaining a complete output record.

#### Public properties

| Property | Type | Description |
|---|---|---|
| `num_logic_blocks` | `int` | Total logical block count, equals `⌈(prompt_len + output_len) / block_size⌉` |
| `last_token_id` | `int` | The most recently generated token id (valid only after at least one decode step) |
| `total_len` | `int` | Total token count (`prompt_len + len(output_token_ids)`) |
| `is_finished` | `bool` | Whether the sequence has ended (either stopped or aborted) |
| `is_prefill_done` | `bool` | Whether prefill is complete (`num_computed_tokens >= prompt_len`) |

---

### SequenceGroup

```python
class SequenceGroup:
    def __init__(self, request_id: str, seqs: List[Sequence])
```

**Groups all sequences belonging to the same request.** In greedy decoding there is exactly one sequence per group; in Beam Search a group holds `beam_width` sequences that share the same prompt and can reuse physical blocks via `fork()`.

#### Fields

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Unique identifier for the originating request |
| `seqs` | `List[Sequence]` | All sequences in this group |

#### Methods

```python
def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]
```

Returns sequences filtered by status. When called without arguments, returns all sequences. Passing `SequenceStatus.RUNNING` returns only running sequences, which `BlockSpaceManager` uses to count how many free blocks are needed.

#### Properties

| Property | Type | Description |
|---|---|---|
| `num_seqs` | `int` | Total number of sequences in the group |
| `is_finished` | `bool` | Whether every sequence in the group has finished |

---

## block_manager.py

### PhysicalBlock

```python
@dataclass
class PhysicalBlock:
    block_num : int      # Physical block index; used directly as the GPU KV-Cache page index
    ref_cout  : int = 0  # Reference count (note: field name is a typo for ref_count)
```

Represents a **GPU physical memory page**. `block_num` is the actual index the Attention kernel uses when accessing KV-Cache. `ref_cout` implements CoW reference-counting semantics:

- `ref_cout == 1` — the block is exclusively owned by one sequence; writes are safe in-place;
- `ref_cout > 1` — the block is shared (via `fork`); a write must copy the block first (CoW).

---

### BlockAllocator

```python
class BlockAllocator:
    def __init__(self, num_blocks: int)
```

A **low-level allocator for a pool of GPU blocks**, backed by a `List[PhysicalBlock]` used as a LIFO stack.

#### Initialization

Pre-allocates `num_blocks` `PhysicalBlock` objects (numbered 0 to `num_blocks-1`) and places them all into `_free_blocks`. The stack ordering (`list.pop()`) promotes locality: recently freed blocks are reused first, reducing cold GPU memory accesses.

#### Methods

```python
def allocate(self) -> PhysicalBlock
```

Pops the top block from the free stack, sets its `ref_cout` to 1, and returns it. Raises `MemoryError("OOM : no free physical blocks")` if the pool is empty.

```python
def free(self, block: PhysicalBlock)
```

Decrements `ref_cout` by 1. When it reaches 0, the block is pushed back onto `_free_blocks`. Calling `free` on a block with `ref_cout == 0` triggers an `AssertionError` (double-free guard).

#### Properties

| Property | Type | Description |
|---|---|---|
| `num_free_blocks` | `int` | Number of currently free blocks |
| `num_total_blocks` | `int` | Total block count (free + allocated) |

---

### BlockSpaceManager

```python
class BlockSpaceManager:
    def __init__(self, block_size: int, num_gpu_blocks: int)
```

**The central class of the module.** Maintains a Block Table for each sequence and exposes all memory management interfaces the scheduler needs.

#### Initialization

| Field | Description |
|---|---|
| `block_size` | Tokens per physical/logical block |
| `allocator` | The internal `BlockAllocator` instance |
| `_block_table` | `Dict[seq_id, List[PhysicalBlock]]` — maps sequence id to its physical block list |

---

#### can_allocate / allocate — Prefill allocation

```python
def can_allocate(self, seq_group: SequenceGroup) -> bool
```

**Checks whether there are enough free blocks to allocate for the group's prefill.**

Implementation: takes the first sequence in the group (there is typically only one during prefill) and compares its `num_logic_blocks` against `allocator.num_free_blocks`.

> **Known issue**: only the first sequence's requirement is checked. In multi-sequence groups (Beam Search initialization), this may underestimate the actual demand. See the [Issues](#known-issues) section.

```python
def allocate(self, seq_group: SequenceGroup)
```

**Allocates physical blocks for all `WAITING` sequences in the group and transitions them to `RUNNING`.**

Steps:
1. Iterate over all `WAITING` sequences;
2. For each sequence, call `allocator.allocate()` once per logical block, building a physical block list;
3. Register the list in `_block_table[seq.seq_id]`;
4. Set `seq.status = SequenceStatus.RUNNING`.

---

#### can_append_slot / append_slot — Decode extension

```python
def can_append_slot(self, seq_group: SequenceGroup) -> bool
```

**Checks whether there are enough free blocks to extend every running sequence by one block in the worst case.**

Implementation: in the worst case every `RUNNING` sequence needs a new block (its last block just filled up), so the required count equals the number of running sequences.

```python
def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]
```

**Before writing the next token in a decode step, ensures the sequence has a writable slot — and triggers CoW if necessary.**

Takes a single sequence; the caller iterates over all running sequences.

Logic:

```
next_position = seq.num_computed_tokens
needed_blocks = next_position // block_size + 1

if len(block_table) < needed_blocks:
    # The last block is full; allocate a new one
    allocate a new block and append to block_table
    return None            # no CoW

last_block = block_table[-1]
if last_block.ref_cout > 1:
    # Last block is shared → CoW
    new_block = allocate()
    free(last_block)       # decrement old block's ref count
    block_table[-1] = new_block
    return (last_block.block_num, new_block.block_num)  # signal kernel to copy

return None  # last block has space and is not shared; write in-place
```

Return value semantics:
- `None` — no extra action needed; write proceeds in-place;
- `(old_num, new_num)` — CoW occurred; the caller (typically the Model Runner) must copy the contents of `old_num` to `new_num` on the GPU before the kernel runs.

> **Important**: `next_position` uses `seq.num_computed_tokens`, which must be updated by the scheduler after each forward pass. If it is stale, `needed_blocks` will be miscalculated.

---

#### fork — Beam Search / Prefix Sharing

```python
def fork(self, parent: Sequence, child: Sequence)
```

**Derives a child sequence from a parent, sharing all physical blocks zero-copy.**

Steps:
1. Shallow-copy the parent's block list (`list(parent_table)`) — the list structure is independent, but the `PhysicalBlock` objects are shared;
2. Increment `ref_cout` on every block in the copied list;
3. Register the new list under `_block_table[child.seq_id]`.

`fork` consumes no new physical blocks. Shared blocks are lazily copied by `append_slot` when either sequence writes to them.

---

#### free — Release sequence resources

```python
def free(self, seq: Sequence)
```

**Releases all physical blocks held by a sequence and marks it as `FINISHED_ABORTED`.**

Steps:
1. If `seq.seq_id` is not in `_block_table`, return immediately (idempotent; prevents double-free crashes);
2. Pop and iterate over the block list, calling `allocator.free()` on each block;
3. Under reference-counting semantics, shared blocks (still referenced by other sequences) are not returned to the pool until the last holder calls `free`;
4. Forces `seq.status = SequenceStatus.FINISHED_ABORTED`.

---

#### get_block_table — Kernel input

```python
def get_block_table(self, seq: Sequence) -> List[int]
```

Converts the sequence's physical block list to a plain integer list (extracting only `block_num`) for the Attention kernel. The kernel uses this table to translate logical block indices into actual GPU memory addresses.

---

#### Properties

```python
@property
def num_free_blocks(self) -> int
```

Delegates to `allocator.num_free_blocks`; used by the scheduler to gauge remaining GPU memory.

---

## Data flow and lifecycle

```
Request arrives
  │
  ▼
SequenceGroup created (one or more WAITING Sequences)
  │
  ▼
Scheduler calls can_allocate() → checks free block count
  │
  ├─ Insufficient → request stays in waiting queue (or preempt another sequence)
  │
  └─ Sufficient → allocate()
                   Assign physical blocks to all WAITING sequences
                   Sequence.status → RUNNING
  │
  ▼
Prefill forward pass
  num_computed_tokens += prompt_len  (scheduler updates)
  │
  ▼
Decode loop:
  ┌──────────────────────────────────────────────────────┐
  │  can_append_slot() → check block availability        │
  │  append_slot(seq)                                    │
  │    → None        : write in-place                   │
  │    → (old, new)  : GPU copies old block to new      │
  │  Model forward pass; next token produced             │
  │  seq.append_token_id(token_id)                       │
  │  num_computed_tokens += 1  (scheduler updates)       │
  └──────────────────────────────────────────────────────┘
  │
  ▼
Sequence ends (EOS or length limit)
  free(seq) → blocks returned to pool driven by ref_cout
```

---

## Class relationship overview

```
SequenceGroup
  └── List[Sequence]
        ├── status: SequenceStatus
        ├── logical_token_blocks: List[LogicalTokenBlock]
        ├── output_token_ids: List[int]
        └── num_computed_tokens: int

BlockSpaceManager
  ├── allocator: BlockAllocator
  │     └── _free_blocks: List[PhysicalBlock]  ← LIFO stack
  └── _block_table: Dict[seq_id, List[PhysicalBlock]]
                                  ↑
                    may be shared across sequences (fork / CoW)
```

---

## Test coverage summary (`test_block_manager.py`)

| Test class | Scenarios covered |
|---|---|
| `TestAllocateFree` | Basic allocate/free, block count, status transition, block table length |
| `TestOOM` | `can_allocate` boundary, forced allocate triggers OOM, `can_append_slot` on full pool |
| `TestAppendSlot` | In-block append (no new block), block-full triggers new allocation, multi-step decode |
| `TestForkAndCoW` | Fork shares blocks without consuming new ones, shared write triggers CoW, free decrements ref count |
| `TestFullLifecycle` | Full prefill→decode→free restores pool completely, block indices stay in valid range |

---

## Known issues

| # | Location | Issue |
|---|---|---|
| 1 | `PhysicalBlock.ref_cout` | Typo — should be `ref_count`. Cosmetic but misleading. |
| 2 | `BlockSpaceManager.can_allocate` L57 | Only checks the first sequence; underestimates demand for multi-sequence groups (Beam Search init). |
| 3 | `BlockSpaceManager.append_slot` L83 | `next_position` depends on `num_computed_tokens` being kept up to date externally; no in-class enforcement. |
| 4 | `BlockSpaceManager.free` L112 | Unconditionally sets status to `FINISHED_ABORTED`, overwriting the `FINISHED_STOPPED` state for normally-completed sequences. |
| 5 | `Sequence.is_finished` L78 | Return type annotation is `int` instead of `bool`. |
