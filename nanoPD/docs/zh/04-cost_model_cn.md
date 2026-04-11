# Cost Model 模块文档

> 对应源码：`cost_model/profiler.py`、`cost_model/analytical.py`

---

## 模块概述

Cost Model 模块解决的核心问题是：**对于一个到来的请求，应该在同一张 GPU 上同时执行 Prefill 和 Decode（co-located），还是将 Prefill 和 Decode 分配到不同 GPU 上并在完成后传输 KV Cache（disaggregated）？**

该模块采用"先对真实设备进行微基准测试，再拟合解析公式，最后基于公式做路由决策"的三步流程，**所有参数均来自对当前设备的实际量测，而非理论预设**，因此能够自适应不同的硬件环境。

| 文件 | 职责 |
|---|---|
| `profiler.py` | 动态 profiling：在真实设备上测量 Prefill 延迟、Decode 延迟、混合干扰、P2P 带宽 |
| `analytical.py` | 解析模型：拟合 profiling 数据，提供延迟估算和路由决策接口 |
| `params.json` | 已拟合的参数文件（RTX 4090 × 8，pinned relay，12.9 GB/s） |
| `params_h20.json` | 已拟合的参数文件（H20，P2P direct，392 GB/s） |

---

## profiler.py — 动态设备感知

### 设计思路

不同硬件（RTX 4090 vs H20）、不同互联方式（NVLink P2P vs pinned memory relay）的性能差异极大。`profiler.py` 在首次部署时对当前设备执行一次系统性的微基准测试，将测量结果序列化为 `profile_data.pt`，供 `analytical.py` 拟合使用。整套流程完全在真实推理路径（`Engine` + `paged_forward`）上运行，保证测量值与实际推理行为高度吻合。

### ProfileResult

```python
@dataclass
class ProfileResult:
    T_prefill       : Dict[int, float]         # prompt_len → 延迟(ms)
    T_decode        : Dict[Tuple, float]        # (kv_len, batch) → 延迟(ms)
    T_interference  : Dict[Tuple, float]        # (chunk_size, decode_batch) → 干扰增量(ms)
    p2p_bandwidth_GBps : float                 # GPU 间实测带宽(GB/s)
```

四项测量数据对应后续解析模型的四个参数（α、β、γ、bandwidth）。

### 核心测量函数

#### profile_prefill

测量不同 prompt 长度下单次 Prefill 前向的延迟，用于拟合 α。

```
prompt_lens = [64, 128, 256, 512, 1024, 2048]
```

每个长度热身 3 次 + 正式测量 10 次，取中位数，使用 `torch.cuda.Event` 计时（GPU 端精确计时，避免 CPU/GPU 同步误差）。

#### profile_decode

测量不同 kv 长度 × 不同 batch size 下单步 Decode 的延迟，用于拟合 β 和 batch_thresh。

```
kv_lens     = [128, 512, 1024, 2048]
batch_sizes = [1, 4, 8, 16, 32]
```

构造 B 条虚拟 Decode 序列（共享同一 block_table），注入 `_current_context` 后执行前向计算。

#### profile_interference

测量 Prefill chunk 对同步执行的 Decode 序列的额外延迟，用于拟合 γ。

测量方式：

```
先测纯 Decode baseline（B 条序列，kv_len=512）
再测混合 batch（chunk_size 个 Prefill token + B 条 Decode）
interference = t_mixed - t_baseline
```

固定 kv_len=512，在 `chunk_sizes=[64, 128, 256, 512]` × `decode_batches=[4, 8, 16]` 上网格搜索，捕捉干扰随 chunk 大小线性增长的规律。

#### profile_p2p_bandwidth

检测 GPU 间实际可用带宽：

- 若支持 P2P direct（`_check_p2p()` 返回 True）：直接 `dst.copy_(src)` 测量 NVLink/PCIe P2P 带宽；
- 否则（如多卡服务器无 NVLink）：经 pinned memory relay（`src → cpu_buf → dst`）测量，反映真实传输路径。

### run_full_profile

```python
def run_full_profile(model_path, output_path, src_gpu, dst_gpu) -> ProfileResult
```

依次执行上述四项测量，将 `ProfileResult` 以 `torch.save()` 序列化到磁盘，供后续 `AnalyticalCostModel.fit_from_profile()` 使用。

---

## analytical.py — 解析模型与路由决策

### CostModelParams

```python
@dataclass
class CostModelParams:
    alpha            : float  # T_prefill(L) ≈ α × L         (ms/token)
    beta             : float  # Decode 单步延迟，batch=1 时    (ms/step)
    batch_thresh     : float  # Decode 从 memory-bound 转 compute-bound 的 batch size 临界点
    gamma            : float  # 干扰系数 T_interference ≈ γ × chunk_tokens (ms/token)
    bandwidth_GBps   : float  # GPU 间实测传输带宽             (GB/s)
    bytes_per_token  : int    # 每个 token 的 KV Cache 字节数
                              # Qwen3-8B: 36层 × 8个KV head × 128维 × 2(K+V) × 2字节 = 147456
```

#### 实测参数对比

| 参数 | RTX 4090 × 8（pinned relay） | H20（P2P direct） |
|---|---|---|
| α | 0.1247 ms/token | 0.1452 ms/token |
| β | 51.56 ms/step | 33.10 ms/step |
| batch_thresh | 16 | 16 |
| γ | 0.0869 ms/token | 0.1302 ms/token |
| bandwidth | 12.9 GB/s | 392 GB/s |
| transfer_rate | 0.01143 ms/token | 0.000376 ms/token |
| **γ / transfer_rate** | **~7.6×** | **~346×** |

`γ / transfer_rate` 是路由决策的核心比值（详见下文）。

---

### AnalyticalCostModel

#### 构造方式

```python
# 从 profiling 数据拟合（首次部署）
model = AnalyticalCostModel.fit_from_profile("profile_data.pt", save_params_path="params.json")

# 直接加载已有参数（日常使用）
model = AnalyticalCostModel.load_params("params.json")
```

#### 拟合逻辑（fit_from_profile）

| 参数 | 拟合方法 |
|---|---|
| α | 对 `T_prefill` 数据做线性回归 `T = α × L`，用 `scipy.optimize.curve_fit` |
| β | 取 batch=1 时各 kv_len 下的 Decode 延迟中位数 |
| batch_thresh | 枚举所有 batch 采样点，找使 `β × max(1, B/thresh)` 拟合 MSE 最小的临界值 |
| γ | 对 batch=8 的干扰数据做线性回归 `T_interf = γ × chunk_size` |

---

### 延迟估算方法

```python
def t_prefill(self, prompt_len) -> float
    # α × prompt_len
```

```python
def t_transfer(self, prompt_len) -> float
    # prompt_len × bytes_per_token / (bandwidth × 1e6)  (ms)
    # 即 KV Cache 通过 pinned relay 或 P2P 传输的时间
```

```python
def t_decode_step(self, batch_size) -> float
    # batch ≤ batch_thresh : β          （memory-bound，带宽墙，延迟与 batch 无关）
    # batch > batch_thresh : β × (batch / batch_thresh) （compute-bound，线性增长）
```

```python
def t_decode_total(self, output_len, batch_size) -> float
    # t_decode_step(batch_size) × output_len
```

```python
def t_collocated(self, prompt_len, output_len, system_load, chunk_size=256) -> float
    # t_prefill + t_decode_total(batch=system_load) + t_interference
    # t_interference = γ × prompt_len × min(system_load, batch_thresh) / batch_thresh
    #                  （system_load=0 时为 0；超过 batch_thresh 后干扰不再增加）
```

```python
def t_disaggregated(self, prompt_len, output_len) -> float
    # t_prefill + t_transfer + t_decode_total(batch=1)
```

---

## 路由决策（route）—— 核心逻辑详解

```python
def route(self, prompt_len, predicted_output_len, system_load,
          decode_batch_size=1, chunk_size=256) -> Tuple[str, float, float]
```

计算两种路由方案的估算端到端延迟，选择较小者：

```python
t_c = t_collocated(prompt_len, output_len, system_load, chunk_size)
t_d = t_prefill(prompt_len) + t_transfer(prompt_len)
    + t_decode_total(output_len, batch_size=decode_batch_size + 1)

decision = "disaggregated" if t_d < t_c else "collocated"
```

返回值：`(decision, t_collocated, t_disaggregated)`。

### 两种方案的成本构成

**Collocated（同卡）：**

```
T_collocated = T_prefill(L)
             + T_decode_total(output_len, batch = system_load)
             + T_interference(L, system_load)
```

- Prefill 和 Decode 争抢同一张卡的算力，Decode 的 batch size 等于整个系统当前的运行请求数；
- 同卡运行时，Prefill 的 QKV 矩阵乘和 Decode 的 Paged Attention 共享 SM 资源，Prefill 对 Decode 造成干扰，延迟增量约为 `γ × prompt_len × load_factor`。

**Disaggregated（分卡）：**

```
T_disaggregated = T_prefill(L)         [P 卡]
                + T_transfer(L)         [KV Cache 传输]
                + T_decode_total(output_len, batch = decode_batch_size + 1)  [D 卡]
```

- Prefill 在专用 P 卡执行，Decode 在专用 D 卡执行，互不干扰；
- 额外代价是将 Prefill 完成的 KV Cache 通过 P2P 或 pinned relay 传输到 D 卡；
- D 卡上 Decode 的 batch size 是 `decode_batch_size + 1`（+1 是因为加入了本条新请求）。

### 路由决策的直觉分析

#### 情形一：系统负载为 0（system_load = 0）

无其他 Decode 请求在运行，因此：
- `T_interference = 0`（无干扰）
- 同卡 Decode 的 batch=1 与分卡 Decode 的 batch=2 延迟相同（均在 memory-bound 区）

此时 Collocated 的代价等于 Disaggregated 代价**减去传输开销**，Collocated 几乎总是胜出。

#### 情形二：中等负载（1 ≤ system_load < batch_thresh）

引入两个线性于 prompt_len 的代价项，比较谁更大：

- Disaggregated 额外付出的传输代价：`transfer_rate × L`（其中 `transfer_rate = bytes_per_token / bandwidth`）
- Collocated 额外付出的干扰代价：`γ × L × (system_load / batch_thresh)`

Disaggregated 胜出的条件（仅考虑干扰 vs 传输）：

```
γ × (system_load / batch_thresh) > transfer_rate
⟹ system_load > batch_thresh × (transfer_rate / γ)
```

代入 RTX 4090 参数（`γ/transfer_rate ≈ 7.6`）：

```
system_load > 16 × (1/7.6) ≈ 2.1
```

**即从 system_load ≥ 3 开始，仅干扰开销已超过传输开销，Disaggregated 开始在干扰项上胜出。**

代入 H20 参数（`γ/transfer_rate ≈ 346`）：

```
system_load > 16 × (1/346) ≈ 0.05
```

**H20 上传输极快，几乎在任何非零负载下 Disaggregated 都优于 Collocated。**

#### 情形三：高负载（system_load > batch_thresh）

Decode 从 memory-bound 进入 compute-bound 区域，`t_decode_step` 随 batch 线性增长：

- Collocated Decode：`β × (system_load / batch_thresh)` 每步 → 总延迟翻倍甚至更多；
- Disaggregated Decode：batch 仅为 `decode_batch_size + 1`（通常远小于 batch_thresh）→ 延迟仍为 β。

在这一区域，Collocated 的 Decode 总延迟可能比 Disaggregated 高出数倍，路由几乎总是选择 Disaggregated。

#### 路由决策总结

| 系统负载 | RTX 4090 倾向 | H20 倾向 | 主导因素 |
|---|---|---|---|
| load = 0 | Collocated | Collocated | 无传输 vs 无干扰，Collocated 零额外开销 |
| 1 ≤ load ≤ 2 | Collocated | **Disaggregated** | H20 传输极快，干扰已超传输 |
| 3 ≤ load ≤ batch_thresh | **Disaggregated** | **Disaggregated** | 干扰开销 > 传输开销 |
| load > batch_thresh | **Disaggregated** | **Disaggregated** | Collocated Decode 进入 compute-bound，延迟倍增 |

#### 具体数值示例（RTX 4090，L=512，output_len=200）

| system_load | T_collocated | T_disaggregated | 决策 |
|---|---|---|---|
| 0 | 10376 ms | 10382 ms | Collocated |
| 4 | 10387 ms | 10382 ms | Disaggregated |
| 16 | 10420 ms | 10382 ms | Disaggregated |
| 32 | 20732 ms | 10382 ms | **Disaggregated（差距 2×）** |

load=32 时 Decode 进入 compute-bound，Collocated 总延迟翻倍，Disaggregated 优势显著。

---

## 可视化方法（plot_all）

```python
def plot_all(self, output_dir, output_len=200, chunk_size=256) -> List[str]
```

生成四张分析图，保存到 `output_dir/`。以下是 RTX 4090 × 8（pinned relay）上的实测结果及分析。

---

### Fig 1 — 路由决策热力图

![Fig 1: Routing Heatmap](../../cost_model/figures/fig1_routing_heatmap.png)

**解读**：热力图的边界极其清晰——整张图只有 load=0 这一行是红色（Collocated），其余所有格子均为绿色（Disaggregated）。

这意味着：**在 RTX 4090 上，只要系统中存在哪怕一条并发的 Decode 请求，路由器就会选择 Disaggregated。** 横轴（prompt_len）对决策几乎没有影响，prompt 长短不改变这一结论。

原因：4090 间通过 pinned memory relay 传输 KV Cache 的带宽仅约 12.9 GB/s，transfer_rate ≈ 0.0114 ms/token；而 Prefill 对 Decode 的干扰系数 γ ≈ 0.0869 ms/token，**γ/transfer_rate ≈ 7.6×**。即便 load=1 时干扰被 batch_thresh=16 除以后只有 γ/16 ≈ 0.0054 ms/token，但 t_collocated 和 t_disaggregated 的差距已足以让后者胜出。

---

### Fig 2 — 端到端延迟对比

![Fig 2: Latency Comparison](../../cost_model/figures/fig2_latency_comparison.png)

**左图（load=0）**：两组柱子几乎等高，Collocated 略低（差距 < 0.1%），决策标注均为 C。这印证了 load=0 时 Collocated 的微弱优势——省去了传输开销，同时没有任何干扰。

**右图（load=8）**：Disaggregated（绿）明显低于 Collocated（红），且差距随 prompt_len 增大而扩大。这是因为两项随 L 线性增长的代价之差在 load=8 时已相当可观：

```
额外干扰代价：γ × L × (8/16) = 0.0869 × L × 0.5 ≈ 0.043 × L  ms
额外传输代价：transfer_rate × L = 0.0114 × L ms
差值 ≈ 0.032 × L ms（每 token 干扰比传输多 0.032ms）
```

L=2048 时差距约 65ms，在 output_len=50 的短生成场景下已是相当显著的比例。

---

### Fig 3 — 传输开销 vs 干扰开销

![Fig 3: Transfer vs Interference](../../cost_model/figures/fig3_transfer_vs_interference.png)

这是理解 4090 路由行为比较直观的一张图。

- **蓝线（传输开销）**：斜率极低，2048 token 时约 23ms。这是 pinned relay 带宽限制的直接体现——12.9 GB/s 对于每 token 147KB 的 KV Cache 已经很慢了。
- **橙色虚线（干扰开销）**：斜率约为蓝线的 7.6 倍，2048 token 时约 175ms。Prefill 对 Decode 的 SM 资源争抢比传输 KV Cache 贵得多。
- **绿色填充区域**：传输开销 < 干扰开销的区间，覆盖了几乎整张图（从约 0 token 开始就已经绿色填满）。

结论：**在 RTX 4090 上，"把 KV Cache 搬走"比"留在原地干扰 Decode"便宜 7.6 倍**，Disaggregated 策略的经济性显而易见。

---

### Fig 4 — Decode 单步延迟 vs Batch Size

![Fig 4: Decode Batch Scaling](../../cost_model/figures/fig4_decode_batch_scaling.png)

图中呈现出典型的"两段式"延迟曲线：

- **平坦段（batch 1→16，约 51ms/step）**：Decode 受内存带宽限制（memory-bound），增加 batch 不增加延迟，GPU 的计算单元大量闲置；
- **线性上升段（batch > 16）**：进入 compute-bound，延迟随 batch 线性增长，batch=32 时约 105ms，恰好是 β 的 2 倍（32/16=2）；
- **虚线（batch_thresh=16）**：拟合得到的临界点，与直观观察完全吻合。

这个曲线解释了为何高负载下 Collocated 的劣势会急剧扩大：system_load 一旦超过 16，Collocated 的 Decode 每步延迟就会线性增加，而 Disaggregated 的 D 卡 Decode batch 通常只有个位数，始终在平坦区。

---

## 调用方式

```python
from cost_model.analytical import AnalyticalCostModel

# 首次部署：profiling + 拟合（需要模型在跑）
from cost_model.profiler import run_full_profile
run_full_profile(model_path, output_path="cost_model/profile_data.pt")
model = AnalyticalCostModel.fit_from_profile(
    "cost_model/profile_data.pt",
    save_params_path="cost_model/params.json"
)

# 日常使用：直接加载
model = AnalyticalCostModel.load_params("cost_model/params.json")

# 路由决策
decision, t_c, t_d = model.route(
    prompt_len=512,
    predicted_output_len=200,
    system_load=8,          # 当前正在 Decode 的请求数
    decode_batch_size=1,    # 目标 D 卡当前的 Decode batch
    chunk_size=256
)
# decision: "disaggregated" or "collocated"
```
