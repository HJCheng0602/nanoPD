# Router 模块文档

> 对应源码：`router/`

---

## 模块概述

`router/` 是 nanoPD 的**动态感知与调度**层，负责在运行时持续观测系统负载，并为每条新请求实时决定走 Collocated 还是 Disaggregated 路径。整个模块由三个类构成，层层组合：

```
OutputLengthPredictor          ← 在线学习：根据 prompt_len 预测输出长度
        ↓
Router                         ← 单次路由决策（调用 AnalyticalCostModel）
        ↓
CentralScheduler               ← 全局编排：动态感知负载，多线程并发执行
```

---

## OutputLengthPredictor

```python
class OutputLengthPredictor:
    def __init__(self, buckets=[64,128,256,512,1024,2048], window=50, default=256, min_samples=5)
```

在线滑动窗口预测器，根据 `prompt_len` 所落入的桶来预测输出长度。

### 数据结构

- `_bucket_data`：每个桶对应一个长度为 `window` 的 `deque`，存储该桶最近 N 条请求的**实际**输出长度。
- `_global_data`：全局滑动窗口（容量 = `window × num_buckets`），汇总所有桶的历史数据，用于冷启动回退。

### `predict(prompt_len) -> int`

三级回退策略：

```
1. 找到 prompt_len 对应的桶（bisect_left，超出最大桶则归入最大桶）
2. 若该桶样本数 >= min_samples → 返回桶内均值
3. 否则，若全局窗口非空 → 返回全局均值
4. 否则 → 返回 default（默认 256）
```

系统启动时（冷启动阶段）所有请求都返回 `default`，待每个桶积累足够样本后，预测精度逐渐逼近该长度区间的真实分布。

### `update(prompt_len, actual_output_len)`

每条请求完成后由 `CentralScheduler.run_until_done()` 回调，将真实输出长度写入对应桶和全局窗口，实现**在线自适应学习**。

### `stats() -> dict`

返回每个桶的样本数和该桶自身的均值，便于监控各长度段的预测质量。



---

## Router

```python
class Router:
    def __init__(self, cost_model: AnalyticalCostModel, predictor: OutputLengthPredictor = None)
    @classmethod
    def from_params(cls, params_path: str, **predictor_kwargs) -> "Router"
```

单次路由决策的封装，将 `OutputLengthPredictor` 与 `AnalyticalCostModel` 串联。

### `route(prompt_len, system_load, decode_batch_size=1) -> str`

```python
predicted_output_len = self.predictor.predict(prompt_len)
decision, t_c, t_d   = self.cost_model.route(
    prompt_len, predicted_output_len, system_load,
    decode_batch_size=decode_batch_size
)
self._history.append((prompt_len, predicted_output_len, decision))
return decision   # "collocated" | "disaggregated"
```

路由的核心逻辑完全在 `AnalyticalCostModel.route()` 中（详见 [04-cost_model_cn.md](04-cost_model_cn.md)）：比较 `T_collocated` 和 `T_disaggregated`，返回耗时更低的路径。

这里有两个关键的**运行时动态量**传入决策：

- **`system_load`**：当前系统中正在执行的序列总数，影响干扰代价 `γ × system_load`
- **`decode_batch_size`**：Decode Worker 当前的批次大小，影响 Decode 阶段的 KV Transfer 均摊收益

也就是说，同样的 `prompt_len`，在系统空闲（load=0）和高负载（load=8）时可能得到完全不同的路由决策。

### `update(prompt_len, actual_output_len)`

转发给 `predictor.update()`，将已完成请求的真实输出长度反馈给预测器。

### `decision_stats() -> dict`

返回历史决策的统计汇总：总请求数、各路径计数、Disagg 占比。

---

## CentralScheduler

```python
class CentralScheduler:
    def __init__(self, collocated_worker, prefill_workers, decode_worker, router, block_size=16)
    @classmethod
    def build(cls, model_path, params_path, collocated_gpu=0,
              prefill_gpus=[1], decode_gpu=2, block_size=16, max_blocks=512) -> "CentralScheduler"
```

系统的全局编排器，持有所有 Worker 和 Router，驱动完整的 P/D 推理循环。

### 成员

| 字段 | 类型 | 说明 |
|---|---|---|
| `cw` | `CollocatedWorker` | Collocated 路径执行单元 |
| `pw_list` | `List[PrefillWorker]` | Disagg 路径 Prefill Worker 池（支持多卡） |
| `dw` | `DecodeWorker` | Disagg 路径 Decode Worker |
| `router` | `Router` | 路由决策器 |
| `_waiting` | `List[(rid, prompt)]` | 等待分发的请求队列 |
| `_states` | `Dict[str, _RequestState]` | 所有请求的运行时状态 |
| `_prefill_threads` | `Dict[int, Thread]` | 各 Prefill Worker 对应的后台线程 |
| `_prefill_done` | `List[...]` | Prefill 线程完成后写入的结果队列（受 `_prefill_lock` 保护） |

`_RequestState` 记录每条请求的 `group`、`prompt_len`、`path`（路由决策）、`output_token_ids` 和 `finished` 状态。

### `build()`

factory method，统一创建所有依赖：

- `BlockSpaceManager`：Disagg 路径的 `PrefillWorker` 和 `DecodeWorker` 共享**同一个** `block_manager` 实例。
- 支持 `prefill_gpus` 列表，可同时启动多个 `PrefillWorker`，每个绑定不同 GPU。
- `CollocatedWorker` 持有独立的 `Engine`，内部有自己的 block manager。

### 核心循环：`step()`

```python
def step(self):
    self._dispatch_waiting()     # 1. 为等待队列中的请求做路由，分发到各 Worker
    self._flush_prefill_done()   # 2. 将已完成 Prefill 的结果推送给 Decode Worker

    t_coll   = Thread(target=self._step_collocated)
    t_disagg = Thread(target=self._step_disaggregated)
    t_coll.start(); t_disagg.start()
    t_coll.join();  t_disagg.join()   # 3. Collocated 和 Disaggregated 并发推进一步
```

Collocated 侧（`_step_collocated`）和 Disaggregated 侧（`_step_disaggregated`）在两个独立线程中同时执行，整个 `step()` 的耗时等于两者的**较大值**。

### `_dispatch_waiting()`：动态感知负载的核心

每次 `step()` 开始时，这里会**实时读取**当前系统状态：

```python
system_load = (
    len(cw.engine.scheduler.running)   # Collocated 路径正在运行的序列数
  + len(dw.running)                    # Disagg Decode Worker 正在 Decode 的序列数
  + len(dw._pending)                   # 等待 KV Transfer 完成的序列数
)
decode_batch_size = len(dw.running) + len(dw._pending)
```

这两个运行时量被传入 `router.route()`，使路由决策能**感知当前的真实负载**，而不是依赖静态配置。

每条等待请求按以下逻辑处理：

```
if decode_batch_size >= MAX_DECODE_BATCH (20):
    强制走 Collocated（防止 Decode Worker 过载）
else:
    path = router.route(prompt_len, system_load, decode_batch_size)

if path == "collocated":
    cw.engine.add_request(prompt)
    system_load += 1
else:
    加入 disaggregated_groups，乐观地 decode_batch_size += 1
```

注意 `system_load` 和 `decode_batch_size` 在循环内**逐请求递增**，后续请求看到的是已包含前序请求的预估负载，避免同一批请求被连续分配到已经过载的路径。

处理完路由后，从 Prefill Worker 池中找一个空闲 worker（`_pick_idle_worker()`）：

- **所有 Prefill Worker 都在忙**：调用 `_requeue()` 将这批 Disagg 请求退回 `_waiting`，等下一步再试。
- **空闲 blocks 不足**：同样退回 `_waiting`，等 Decode 完成释放 block 后再重试。
- **找到空闲 Worker**：在后台线程中启动 `prefill_batch_and_extract(groups)`，结果写入 `_prefill_done`。

### `_flush_prefill_done()`

用 `_prefill_lock` 安全地取走 `_prefill_done` 中的所有结果，为每条请求：
1. 将首 token 追加到 `state.output_token_ids`
2. 调用 `dw.receive_kv_async(group, block_table, kv_buf, src_k, src_v)` 发起异步 KV Transfer

此后该请求进入 `DecodeWorker._pending` 队列，直到 KV Transfer 完成后在下一个 `step()` 被提升为 `running`。

### `_step_collocated()` 和 `_step_disaggregated()`

- `_step_collocated`：调用 `cw.step()`，完成后扫描 `cw.engine.scheduler.finished`，将完成的请求标记并写回 `_states`。
- `_step_disaggregated`：调用 `dw.step()`，收集返回的 `(group, tok_id)` 对，追加 token 到对应 `state`，处理 EOS。内置了简单的 OOM 恢复：若捕获到 `MemoryError`，驱逐 `dw.running` 中最老的序列并释放其 block，让后续请求有机会继续。

### `run_until_done(max_new_tokens=200) -> Dict[str, str]`

循环调用 `step()` 直到所有请求完成，期间每步检查 `_enforce_max_tokens()` 防止超长输出。完成后对每条请求调用 `router.update(prompt_len, actual_output_len)`，将真实输出长度反馈给 `OutputLengthPredictor`，**形成在线学习闭环**。

最终返回 `{rid: decoded_text}` 字典。

### `stats() -> dict`

返回当前运行时快照：
- Router 历史决策统计（`router.decision_stats()`）
- 预测器各桶状态（`predictor.stats()`）
- Decode Worker 当前 pending / running 数
- Prefill Worker 数量及当前忙碌线程数

---

## 整体数据流

```
add_request(prompt)
    → _waiting

step()
├── _dispatch_waiting()
│   ├── 实时采集 system_load / decode_batch_size
│   ├── router.route(prompt_len, system_load, decode_batch_size)
│   │     └── predictor.predict()  +  cost_model.route()
│   ├── [collocated]  → cw.engine.add_request()
│   └── [disaggregated] → Thread(_prefill_task)
│                             └── pw.prefill_batch_and_extract()
│                                 → _prefill_done
│
├── _flush_prefill_done()
│   └── dw.receive_kv_async()  → dw._pending
│
├── Thread(_step_collocated)    ─┐
│   └── cw.step()               ├─ 并发执行
└── Thread(_step_disaggregated) ─┘
    └── dw.step()
          ├── _promote_ready()  （_pending → running）
          └── decode forward pass

run_until_done()
    └── 循环 step() 直到全部完成
        → router.update(actual_output_len)  ← 在线反馈
```

---

## 设计亮点

**1. 每步实时感知负载**

`_dispatch_waiting()` 不依赖任何静态配置，每次 `step()` 都重新读取 `running` / `pending` 队列的当前长度，使路由决策随系统状态实时变化。高负载时自动偏向 Disagg（干扰代价上升），低负载时可能全走 Collocated（传输开销不合算）。

**2. 过载硬保护**

`decode_batch_size >= MAX_DECODE_BATCH` 时，无论 Router 怎么算，都强制走 Collocated。防止大量请求涌入 Decode Worker 导致 block 耗尽或步延迟飙升。

**3. Prefill Worker 池**

支持多个 `PrefillWorker`（不同 GPU），通过 `_pick_idle_worker()` 在空闲 worker 上启动后台线程，多条 Disagg 请求的 Prefill 可以在不同 GPU 上并发进行，互不阻塞。

**4. Prefill–Decode 流水线重叠**

Prefill 在后台线程异步执行，KV Transfer 在 `transfer_stream` 上异步进行，Decode 在 `compute_stream` 上独立推进。三个阶段在同一时间轴上并发，新请求的延迟被 Decode 步骤的计算时间所掩盖。

**5. 在线学习闭环**

`run_until_done()` 完成后将真实输出长度反馈给 `OutputLengthPredictor`，下一批请求的路由决策将基于更准确的输出长度预测，使系统对工作负载特征自适应。

---

## 已知局限

**1. Collocated Step 阻塞主循环**

`_step_collocated` 调用 `cw.step()`，此调用内含 GPU 同步（forward pass），若 Collocated 批次较大，整个 `step()` 的耗时会被它拉长（两个线程 join 取最大值）。代码中的 timing 注释已点出这一问题：`coll占step: X% ← 这个高说明 GPU0 在阻塞主循环`。

**2. Prefill 侧无 Chunked Prefill**

如 [05-workers_cn.md](05-workers_cn.md) 所述，`PrefillWorker` 做全序列一次性 Prefill，极长 Prompt 会导致 Prefill 线程长时间占用，新请求无法及时进入 Decode 队列。

**3. `system_load` 估算在高并发时偏乐观**

`_dispatch_waiting()` 在同一个 `step()` 内一次性处理所有等待请求，每分发一条就将 `system_load += 1`，但此时 Prefill 线程还未启动、KV Transfer 还未发生，实际负载可能并没有真正上升——这是一种乐观估算，极端情况下可能导致路由决策偏差。

**4. `_requeue` 需要重新分词**

当请求因 Prefill Worker 全忙或 block 不足而退回时，`_requeue()` 将 token IDs 重新解码为字符串，下次 `_dispatch_waiting()` 再重新分词。直接保留 token ID 列表退回似乎会更好一些？晕晕晕晕该准备算分期中了呜呜呜
