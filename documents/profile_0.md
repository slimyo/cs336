# 性能分析工作流：预测 → benchmark → nsys → ncu → 回填解释

本文把 CS336 Assignment 2（Part 1）里做过的 profiling 整理成可重复使用的流程。参考配置：small（L=12），ctx 512，batch 4，fp32（TF32 关），RTX 5070 Ti（70 个 SM，CC 12.0）。换模型或换配置时，流程不变，数字重算。

---

## 0. 总原则

**一个问题，一个工具。** 每个工具只回答它擅长的问题，越界使用会得到看似合理的错误结论。

| 工具 | 回答什么 | 可以信 | 不能信 |
|---|---|---|---|
| `benchmark.py`（不加 profiler） | **多快**：各阶段真实耗时 | 绝对时间 | 不告诉你时间花在哪 |
| nsys | **时间花在哪**：kernel 种类、次数、占比、空隙 | 构成、次数、占比、利用率 | 绝对时间（有采集开销，每阶段同步破坏重叠） |
| ncu | **单个 kernel 为什么慢**：算力还是带宽、占用率、stall | 利用率百分比与比例 | duration（重放、锁频、清缓存），不能和 nsys、benchmark 比 |
| `torch.profiler` | 算子（`aten::`）级的 CUDA 时间 | 与 nsys 互补 | 同 nsys |
| `max_memory_allocated` / 内存快照 | 显存峰值与构成 | 峰值 | — |

**顺序**：先预测，再 benchmark，再 nsys，最后 ncu。

```
预测（第 11 章）
   │
   ▼
benchmark ──► 真实耗时、TFLOP/s、显存峰值
   │
   ▼
nsys ──► 标注自检 → GPU 利用率 → 各阶段构成 → 类别表 → 待深挖清单
   │         │
   │         └─ 利用率低（发射受限）→ 不上 ncu，回到减少 kernel 数
   ▼
ncu ──► 先整体扫一个 block（basic）→ 再深挖少数 kernel（full）
   │
   ▼
回填：预测 vs 实测 vs 差异原因（带证据）→ 最出乎意料的三项 → 下一步优化
```

---

## 1. 阶段 0：先写下预测

在测任何东西之前，写出这些预测，否则不知道什么算"出乎意料"：

| 预测项 | 来源 |
|---|---|
| 各阶段 FLOPs、理想时间（FLOPs ÷ 峰值）、访存部分 | 11.3 / 11.5(1) |
| 注意力、FFN、其余的时间占比（与 FLOPs 占比并排） | 11.5(2) |
| 各类 kernel 数与时间占比 | 11.5(3) |
| 各区间 kernel 数与时间（按 block） | 11.5(4) |
| 显存峰值 | 11.6 |

**预测里的常数要先校准**，否则差异分不清是"模型错"还是"常数错"：

| 常数 | 校准方法 |
|---|---|
| 算力峰值 P（本项目取 33.4 TFLOP/s） | 单独跑一个大 fp32 GEMM 的 benchmark |
| 显存带宽 BW（本项目取 769 GB/s） | 单独跑一个大逐元素 kernel，或用 ncu 的 `DRAM Throughput %` 反推。本项目实测约 882 GB/s，比 769 高约 13% |
| 词表大小 V、参数个数 | 直接读配置 |

---

## 2. 阶段 1：benchmark

### 2.1 目的

只回答"多快"。**不加 profiler，不加细粒度 NVTX，不在阶段间额外同步。**

### 2.2 运行

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py \
    --model small --ctx 512 --batch 4 --dtype float32 --mode all
```

### 2.3 输出与检查

| 检查 | 标准 |
|---|---|
| 抖动（jitter） | 小（本项目 0～1%）；大说明卡不空闲、频率波动或预热不足 |
| 预热 | 预热步数足够，不计入统计 |
| 中位数 | 用中位数，不用平均值 |
| 阶段拆分 | fwd、fwdbwd、full 三个 mode 的**中位数相减**，得到 backward、optimizer |
| 显存 | 同时读 `max_memory_allocated` |
| 卡是否空闲 | 运行前 `nvidia-smi` |

### 2.4 本项目参考结果

| mode | 中位数 (ms) | TFLOP/s | 峰值显存 |
|---|---|---|---|
| fwd | 37.72 | 14.16 | 0.71 GiB |
| fwdbwd | 114.90 | 13.94 | 4.06 GiB |
| full | 127.85 | 12.53 | 5.04 GiB |

相减得到：backward 77.18 ms，optimizer 12.95 ms。

---

## 3. 阶段 2：nsys

### 3.1 先理解三个概念（决定你会不会读错）

**① GPU 是异步的**：CPU 只发射 kernel，GPU 之后才执行。NVTX 区间记录的是 **CPU 侧**的起止。

**② kernel 属于哪个 NVTX 区间，看的是"哪个线程在哪个区间里发射了它"**，不是"它在 GPU 上什么时候跑"。

**③ 同一个区间有三种不同的"时间"：**

| 名称 | 含义 | 报告 |
|---|---|---|
| CPU 时间 | 区间在 CPU 上持续多久 | `nvtx_sum` |
| GPU 投影跨度 | 第一个 kernel 开始到最后一个结束（含空隙） | `nvtx_gpu_proj_sum` |
| kernel 耗时之和 | 区间内每个 kernel 的 duration 相加（不含空隙） | `nvtx_kern_sum` |

跨度 − 耗时之和 = 空隙。

### 3.2 标注方案

| 区间 | 位置 | 说明 |
|---|---|---|
| `step` / `forward` / `backward` / `optimizer` | 主线程 | 每阶段末尾 `synchronize`，仅 profile 模式 |
| `embed`、`block`、`ln_final`、`lm_head` 等模块名 | forward hook | 同类用同一个名字 |
| `attn.scores` / `attn.softmax` / `attn.av` | 替换函数 | 不改 staff 源码 |

命名规则：**静态字符串**；同类层同名；层级用点号；需要看单层时才临时加编号。

```python
def phase(name, fn):
    with nvtx.range(name):
        out = fn()
        torch.cuda.synchronize()      # 仅 profile 模式：让阶段边界干净
    return out

for _ in range(args.warmup): train_step()     # 预热不被抓
torch.cuda.synchronize()
torch.cuda.profiler.start()                    # 只从这里开始抓
for _ in range(3):                             # 3 步足够
    with nvtx.range("step"):
        loss = phase("forward", fwd_and_loss)
        phase("backward", loss.backward)
        phase("optimizer", opt_step_and_zero)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
```

### 3.3 采集

```bash
mkdir -p results/nsys
CUDA_VISIBLE_DEVICES=0 uv run nsys profile \
  -o results/nsys/small_ctx512_fp32 --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python scripts/profile_run.py --model small --ctx 512 --batch 4 --mode full \
         --warmup 5 --steps 3 --nvtx-fine
```

### 3.4 生成报告

```bash
cd /data2/c220/hz/cs336
uv run nsys stats --report cuda_gpu_kern_sum --report nvtx_sum --format csv \
    --output results/nsys/small_ctx512 results/nsys/small_ctx512_fp32.nsys-rep
uv run nsys stats --report nvtx_kern_sum --format csv \
    --output results/nsys/xkern results/nsys/small_ctx512_fp32.nsys-rep
uv run nsys stats --report nvtx_gpu_proj_sum --format csv \
    --output results/nsys/xproj results/nsys/small_ctx512_fp32.nsys-rep
```

报告名随 nsys 版本变化，先 `nsys stats --help-reports`。文件名规律是"前缀_报告名.csv"。

### 3.5 分析流程（按顺序，每步有通过标准）

#### 步骤 1：标注自检（不通过，后面全部别解读）

读 `nvtx_sum` 的 Instances 列：

| 区间 | 应为（steps=3） |
|---|---|
| step / forward / backward / optimizer | 3 |
| `block`（TransformerBlock） | 3L |
| `RMSNorm` | 3(2L+1) |
| `Linear` | 3(7L+1) |
| `Embedding` | 3 |
| `attn.scores` / `attn.softmax` / `attn.av` | 3L |

本项目（L=12）：36 / 75 / 255 / 3，全部对上。

#### 步骤 2：检查 backward 有没有被漏

**这是最容易踩的坑。** 反向由 autograd 引擎的工作线程发射，主线程里的 `backward` 区间收不到这些 kernel。

| 检查 | 异常表现 |
|---|---|
| `nvtx_kern_sum` 里 `backward` 的 kernel 数 | 远小于 forward 的约 2 倍（本项目只有 1 个/步） |
| `step` 的 kernel 数 | = forward + optimizer + backward 的少数几个，漏掉了反向 |
| TID 列 | 各区间在同一个线程；反向发射在另一个线程（本项目只在 `CCCL:cub::DeviceRadixSort` 上看到另一个 TID） |

注意几件事：

- `nvtx_kern_sum` 中嵌套区间是**累计计入**的：kernel 同时算进它所在的每一层区间，所以不同行不能相加。
- nsys 给默认 domain 的区间名加了前缀 `:`（如 `:forward`）。
- `nvtx_gpu_proj_sum` 里的 `Total Range Time` 是 **CPU 时间**，GPU 投影在 `Total Proj Time`。

#### 步骤 3：用"差分法"恢复 backward（不用 SQL）

`cuda_gpu_kern_sum` 不分区间，包含全部 kernel。总表减去 forward 和 optimizer，剩下就是 backward：

```python
import pandas as pd, re
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)

def load(pattern):
    import glob
    f = glob.glob(pattern)
    if not f: raise FileNotFoundError(f"没匹配到 {pattern}；有：{sorted(glob.glob('results/nsys/*'))}")
    d = pd.read_csv(f[0]); d.columns = [c.strip() for c in d.columns]; return d

ks   = load("results/nsys/small_ctx512_cuda_gpu_kern_sum.csv")
kern = load("results/nsys/xkern*.csv")           # 列：NVTX Range, TID, Kern Inst, Total Time (ns), Kernel Name
STEPS = 3

tot = ks.set_index("Name")[["Instances", "Total Time (ns)"]].rename(columns={"Instances": "n", "Total Time (ns)": "t"})
def per_kernel(r):
    d = kern[kern["NVTX Range"] == r]
    return d.groupby("Kernel Name").agg(n=("Kern Inst", "sum"), t=("Total Time (ns)", "sum"))

fwd, opt = per_kernel(":forward"), per_kernel(":optimizer")
bwd = tot.sub(fwd, fill_value=0).sub(opt, fill_value=0)
assert not (bwd < 0).any().any(), "两张报告里 kernel 名没对上"
print("backward kernel 数/步:", bwd.n.sum() / STEPS)
```

自检：`forward + backward + optimizer` 的 kernel 数之和应等于总表的 kernel 数；`backward ÷ forward` 约 2 倍量级。

差分法的假设：抓取范围内所有 kernel 都属于这三个阶段。

要看逐次的构成或者按时间窗归因时，再用 sqlite 的时间窗法（`NVTX_EVENTS` 与 `CUPTI_ACTIVITY_KIND_KERNEL`，表名与列名以 `PRAGMA table_info` 为准）。

#### 步骤 4：GPU 利用率自检（决定要不要上 ncu）

```
利用率 = kernel 耗时之和 ÷ 该阶段的 CPU 窗口
```

分子用 `cuda_gpu_kern_sum` 或差分法的结果，**不要用 NVTX 区间汇总**（会漏掉 backward）。

| 比值 | 含义 | 下一步 |
|---|---|---|
| 接近 1 | GPU 受限 | 优化 kernel 本身，值得上 ncu |
| 明显低于 1（如 < 0.6） | **发射受限**，GPU 在等 CPU | 减少 kernel 数（融合、`torch.compile`、CUDA graph、foreach）；ncu 帮不上 |

本项目：

| 阶段 | kernel 数/步 | kernel 耗时 | CPU 窗口 | 利用率 |
|---|---|---|---|---|
| forward | 641 | 37.4 ms | 39.0 ms | 0.96 |
| backward | 1238 | 73.8 ms | 76.8 ms | 0.96 |
| optimizer | 1776 | 10.4 ms | 21.1 ms | **0.49** |

#### 步骤 5：类别表

```python
CATS = [("gemm", r"gemm|xmma|cutlass|cublas"), ("softmax", r"softmax"), ("reduce", r"reduce"),
        ("elementwise", r"elementwise|vectorized"), ("copy/cat", r"copy|cat|memcpy"),
        ("index/embed", r"index|embedding|scatter|gather|sort"), ("optimizer", r"multi_tensor|adam")]
def cat(n):
    for c, pat in CATS:
        if re.search(pat, n, re.I): return c
    return "other"

def table(d):
    d = d[d.n > 0].copy(); d["cat"] = [cat(x) for x in d.index]
    g = d.groupby("cat")[["n", "t"]].sum()
    g["n/step"] = g.n / STEPS; g["ms/step"] = g.t / 1e6 / STEPS; g["share_%"] = g.t / g.t.sum() * 100
    return g[["n/step", "ms/step", "share_%"]].sort_values("ms/step", ascending=False).round(2)

for name, d in [("forward", fwd), ("backward", bwd), ("optimizer", opt)]:
    print(name); print(table(d))
```

先把归为 `other` 的最大几项看一遍，补全规则。

#### 步骤 6：把 kernel 对回代码

| 方法 | 可靠度 |
|---|---|
| 区间内的 kernel 数与名字（如 `attn.softmax` 里是哪几个） | 可靠 |
| 区间总量往回推（用于 GEMM 的对应） | 推断，要标明 |
| 单算子隔离：写个小循环只跑一个算子，看产生的 kernel 名 | 最可靠 |

示例：softmax 一次是 5 个 kernel（max、sub、exp、sum、div）；Linear 里 73 个是 `magma_sgemmEx`、12 个 w2 是 `cutlass 128x256_tn`；QKᵀ 与 PV 的对应是从区间总量推出的。

#### 步骤 7：产出"待深挖清单"

挑 3 到 6 个 kernel：总时间占比最大的 GEMM、softmax 相关、最大的逐元素、reduce。记下完整名字和所在区间（forward 里的可以用 NVTX 过滤；backward 里的只能按名字）。

### 3.6 GUI（可选，10 分钟）

`nsys-ui` 打开 `.nsys-rep`：

| 看什么 | 目的 |
|---|---|
| NVTX 行 | 3 个 step 长得一样；嵌套层级正确 |
| kernel 行 | 是否落在对应阶段条内；空隙 |
| 悬停 kernel | 看 NVTX 父链，抽查归属 |

---

## 4. 阶段 3：ncu

### 4.1 前提

- nsys 的利用率自检通过（接近 1）；
- 已有待深挖清单；
- 权限：`ERR_NVGPUCTRPERM` 要联系管理员，**最早试一次 `-c 1`**；
- 版本要支持 sm_120；指标名以 `ncu --query-metrics`、`--list-sections` 为准；
- 形状、dtype、TF32 设置与真实运行一致。

### 4.2 过滤

ncu 把每个 kernel 重放多遍，**必须过滤**，先 `-c 1` 试通。

| 方式 | 选项 | 注意 |
|---|---|---|
| NVTX 区间 | `--nvtx --nvtx-include "TransformerBlock/"` | 只对主线程发射的 kernel 有效（forward、optimizer）；**对 backward 失效** |
| 跳过加数量 | `-s 6 -c 1` | 在过滤之后计数（本项目验证有效） |
| 按名字 | `-k regex:xxx`，模板内的名字要加 `--kernel-name-base demangled` | backward 用这个 |
| 其他 | `--profile-from-start off`（配合 `profiler.start()`）、`--cache-control none`、`--clock-control none`、`--target-processes all` | |

先 `mkdir -p results/ncu`，否则会报"Unable to write to file"。

### 4.3 两轮策略

**第一轮：整体扫一个 block（`basic`，便宜）。** 一次拿到 52 个 kernel 各自的 Compute / Memory / DRAM 百分比：

```bash
CUDA_VISIBLE_DEVICES=0 uv run ncu -o results/ncu/block0 -f \
  --profile-from-start off --nvtx --nvtx-include "TransformerBlock/" \
  -c 52 --set basic \
  python scripts/profile_run.py --model small --ctx 512 --batch 4 \
        --mode fwd --warmup 2 --steps 1 --nvtx-fine
```

```bash
ncu --import results/ncu/block0.ncu-rep --page raw --csv > results/ncu/block0.csv
```

```python
d = pd.read_csv("results/ncu/block0.csv")
d = d[pd.to_numeric(d["ID"], errors="coerce").notna()].copy()     # 去掉单位行
print("抓到的 kernel 数:", len(d))                                  # 应等于 -c
def col(sub): return next(c for c in d.columns if sub in c)
def num(s): return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
t = pd.DataFrame({
  "kernel": d["Kernel Name"].map(lambda n: re.sub(r"<.*", "<…>", re.sub(r"^void\s+", "", str(n)))[:60]),
  "grid": d[col("launch__grid_size")],
  "compute%": num(d[col("sm__throughput.avg.pct_of_peak_sustained_elapsed")]),
  "memory%":  num(d[col("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")]),
  "dram%":    num(d[col("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")])}).round(1)
print(t.to_string())
```

列名找不到时，先 `print([c for c in d.columns if "throughput" in c])`。

**第二轮：深挖 1～3 个有代表性的 kernel（`full`，`-c 1`）。**

```bash
# 代表性 GEMM（block 内第 7 个）
ncu -o results/ncu/gemm_qkv_full -f --profile-from-start off \
  --nvtx --nvtx-include "TransformerBlock/" -s 6 -c 1 --set full \
  python scripts/profile_run.py ... --mode fwd --warmup 2 --steps 1 --nvtx-fine

# softmax 的 5 个 kernel
ncu -o results/ncu/softmax -f --profile-from-start off \
  --nvtx --nvtx-include "attn.softmax/" -c 5 --set full python scripts/profile_run.py ...
```

`--set basic` 只含 Speed Of Light、Launch Statistics、Occupancy 等；想要 Compute Workload、Warp State、Memory Workload、Scheduler 这些，必须 `--set full` 或指定 `--section`。

打印要看的 section：

```bash
ncu --import results/ncu/gemm_qkv_full.ncu-rep --page details \
 | awk '/^ *Section: /{p = ($0 ~ /Compute Workload|Scheduler|Warp State|Memory Workload|Instruction Stat/)} p'
```

### 4.4 怎么读：判读规则

**Speed Of Light 的两个百分比：**

| Compute (SM) | Memory | 结论 |
|---|---|---|
| 低 | 高（且 DRAM 高） | **带宽受限**：要减少访存（融合、降精度） |
| 高 | 低 | 算力受限 |
| 都低 | 都低 | 并行度不足或 kernel 太小：延迟受限，看 grid、occupancy |
| 都高 | 都高 | 接近硬件极限 |

**重要修正：`Memory Throughput` 取的是"最繁忙的那一级"。** GEMM 的 L1/共享内存很忙，Memory 会高达 76～92%，但 DRAM 只有 5～8%。判断是否 DRAM 受限，看 **DRAM Throughput** 这一列。

**本项目的实测分类：**

| 类型 | Compute | Memory / DRAM | 结论 |
|---|---|---|---|
| 大张量 vectorized 逐元素、softmax 的 5 个 kernel | 3～43% | DRAM 80～87% | 带宽受限，接近上限 |
| 中等张量逐元素、RMSNorm 的 reduce | 2～43% | DRAM 58～78% | 带宽受限，没那么满 |
| 极小 kernel（grid ≤ 8） | ≈ 0 | 6～9% | 延迟受限，代价是发射开销 |
| 大 GEMM | 51～69% | Memory 63～92%，DRAM 5～8% | 不是访存受限，也未到峰值 |

**ncu 自带的 OPT 提示要自己核算量级**，不要直接信"可提速 N%"。例如 qkv 的共享内存 bank conflict 提示"可提速 42%"，按全部 SM 的活跃周期折算上限只有约 6%。

### 4.5 三个验算技巧（把 ncu 的数字和你的预测连起来）

**① 搬运量自检**（带宽受限的 kernel）：

```
带宽 × duration ≈ 张量字节数
```

softmax 的 max reduce：734.6 GB/s × 68.6 µs ≈ 50.4 MB = 4×12×512×512×4 B，对上。

**② 用 nsys 时间反推有效带宽**（ncu 的 duration 不可信，这里用 nsys 的）：

```
有效带宽 = 读写字节数 ÷ nsys 耗时；相对峰值带宽的比例
```

softmax 的 5 个 kernel 合计搬运 8 遍 T² 张量，平均约 889 GB/s，约等于 DRAM 峰值。超过峰值的 kernel（max、sum）说明读到了上一个 kernel 刚写出的数据，命中了 L2，ncu 单 kernel 重放复现不出这一点。

**③ 指令数拆分 GEMM 的 FMA 利用率**：

```
FMA 利用率 ≈ (FMA 指令数 ÷ 总指令数) × Issue Slots Busy
FMA 指令数 = M×N×K ÷ 32
```

| | qkv（64×64 tile） | w2（128×256 tile） |
|---|---|---|
| FMA 占指令比例 | 71% | 87% |
| Issue Slots Busy | 66% | 55% |
| 乘积 | ≈ 47% | ≈ 48% |
| ncu 的 FMA 利用率 | 50.0% | 48.8% |
| 每调度器活跃 / 可发射 warp | 4.2 / 2.1 | 2.0 / 0.9 |
| 限制 | 指令占比加发射效率 | 寄存器与共享内存限占用，延迟受限 |

### 4.6 ncu 的坑

| 坑 | 处理 |
|---|---|
| 默认清缓存、锁频 | 想接近真实时用 `--cache-control none`、`--clock-control none`，并在报告里注明；本项目 softmax 暖缓存对比冷缓存几乎一样（张量略大于 L2） |
| SM 频率在 kernel 之间不同（本项目 1.96 与 2.27 GHz） | Compute% 不宜横向比较；算 TFLOP/s 用 nsys 的时间 |
| duration 与 nsys 对不上（softmax max：68.6 µs 对 51.5 µs） | 不比，只用比例 |
| 大 GEMM 的低占用率（16.7%）被提示"可提速 83%" | 对大 tile 的 GEMM 是常态，不要追 |
| NVTX 过滤对 backward 失效 | backward 用 `-k` 名字加 `-s/-c` |
| TF32 没关 | GEMM 走不同路径，与预测不符；看 kernel 名（本项目是 `cutlass_80_simt_sgemm`，SIMT） |

---

## 5. 阶段 4：回填与解释

### 5.1 回填模板

| 项 | 预测 | 实测 | 实测 ÷ 预测 | 差异原因 | ncu 证据 |
|---|---|---|---|---|---|

"ncu 证据"只对需要解释的差异填，**写具体指标**，不写"推断为"：

| 好 | 不好 |
|---|---|
| `softmax：DRAM 83.3%（734.6 GB/s），Compute 37.8%，带宽受限` | `推断为访存受限` |
| `w2 GEMM：FMA 48.8%，每调度器仅 2 个 warp，44.7% 周期无可发射 warp` | `没到峰值` |

判定：实测 ÷ 预测在 **0.8～1.25** 内算预测成立，超出就要找原因。

### 5.2 差异原因速查

| 现象 | 常见原因 | 用什么确认 |
|---|---|---|
| 实测远高于预测，GPU 利用率低 | 发射受限（小模型、短 ctx、逐参数 optimizer） | nsys 的利用率自检、`cuda_api_sum` |
| kernel 数多于预测 | 漏估类型转换、`contiguous`、`cat`；一个算子被拆成多个 kernel（如手写 softmax = 5 个） | nsys 的 kernel 清单、区间内 kernel 数 |
| 注意力时间占比高于 FLOPs 占比 | 访存密集，$T^2$ 大小的中间张量读写 | ncu：Memory/DRAM 高、Compute 低 |
| GEMM 比理想慢 | tile 形状、占用率、没走 Tensor Core（TF32 没开或 dtype 没统一） | ncu：FMA 利用率、指令占比、warp 数 |
| 逐元素占比大 | 每个 kernel 读写整个张量，带宽受限 | ncu：DRAM 接近峰值 |
| backward 不是 forward 的 2 倍 | 各类别倍数不同（GEMM、elementwise 方向相反）；归因窗口没对齐 | 差分法按类别分别算 |
| 显存实测与预估不符 | 份数估错、临时 buffer、allocator 碎片 | `max_memory_reserved`、`saved_tensors_hooks`、内存快照 |

### 5.3 误差抵消要拆开看

总时间吻合不等于模型对。**把总量拆成分量再比。** 本项目 forward 总时间预测 1.02×，但：

| 分量 | 预测 | 实测 | 比值 |
|---|---|---|---|
| GEMM | 16.0 ms | 24.3 ms | 1.52 |
| 访存类 | 约 20 ms | 13.0 ms | 0.65 |

两个方向相反的误差抵消了。

### 5.4 最后写"最出乎意料的三项"

每项写清：哪项预测错了、实测是什么、原因（带证据）、下一步优化什么。**这三条比整张表更重要。**

---

## 6. 本项目参考结果（small，ctx 512，batch 4，fp32，5070 Ti）

**阶段**：forward 37.7 ms，backward 77.2 ms，optimizer 12.95 ms；整步 127.85 ms。

**整步 kernel 构成**（nsys）：GEMM 52.2%，elementwise 44.7%，reduce 2.7%，copy/index 0.4%。

**预测对照**：

| 项 | 预测 | 实测 | 比值 |
|---|---|---|---|
| forward | 约 37 ms | 37.72 ms | 1.02 |
| 注意力占 forward 时间 | 约 35% | 29.4% | 0.84 |
| block × 12 的 kernel 数 | 约 600 | 624 | 1.04 |
| forward GEMM 的 kernel 数 | 约 109 | 109 | 1.00 |
| 显存峰值 | 6.16 GiB | 5.04 GiB | 0.82 |

**最出乎意料的三项：**

1. **softmax**：预测 5%，实测约 14.6%，且是 5 个 kernel，每次搬运 8 遍 T² 张量，有效带宽约等于 DRAM 峰值。融合版理论上约 4 倍（上限估计，不是承诺）。
2. **大 GEMM**：不是"接近峰值"，约 43～48% fp32 峰值。SIMT 的 `cutlass_80_simt_sgemm`，没走 Tensor Core。
3. **optimizer**：是带宽下限的 2.76 倍，且发射受限（1776 个 kernel，nsys 里利用率 0.49，benchmark 里约 0.80）。backward 的 elementwise 是 forward 的 2.89 倍。

**下一步优化（按证据排序）**：融合 optimizer（foreach 或融合 AdamW）→ `torch.compile` 融合 elementwise → TF32 或 BF16 走 Tensor Core → 融合 softmax（随 ctx 增大收益增大）。

---

## 7. 命令速查

```bash
# benchmark
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model small --ctx 512 --batch 4 --dtype float32 --mode all

# nsys 采集
mkdir -p results/nsys
CUDA_VISIBLE_DEVICES=0 uv run nsys profile -o results/nsys/<名> --force-overwrite=true --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop python scripts/profile_run.py <参数> --nvtx-fine

# nsys 报告
uv run nsys stats --report cuda_gpu_kern_sum --report nvtx_sum --format csv --output results/nsys/<前缀> results/nsys/<名>.nsys-rep
uv run nsys stats --report nvtx_kern_sum     --format csv --output results/nsys/xkern results/nsys/<名>.nsys-rep
uv run nsys stats --report nvtx_gpu_proj_sum --format csv --output results/nsys/xproj results/nsys/<名>.nsys-rep

# ncu：整体扫一个 block，再深挖
mkdir -p results/ncu
ncu -o results/ncu/block0 -f --profile-from-start off --nvtx --nvtx-include "TransformerBlock/" -c 52 --set basic python scripts/profile_run.py <参数> --mode fwd --warmup 2 --steps 1 --nvtx-fine
ncu -o results/ncu/<名>_full -f --profile-from-start off --nvtx --nvtx-include "TransformerBlock/" -s <跳过数> -c 1 --set full python scripts/profile_run.py <参数>
ncu --import <文件>.ncu-rep --page details            # 文本
ncu --import <文件>.ncu-rep --page raw --csv > <文件>.csv
```

---

## 8. 检查清单

**benchmark**
- [ ] 不加 profiler、不加细标注；预热足够；用中位数；jitter 小
- [ ] 运行前 `nvidia-smi` 确认卡空闲
- [ ] fwd / fwdbwd / full 相减得到各阶段；同时记显存峰值

**nsys**
- [ ] 预热在 `profiler.start()` 之前；capture-range；只抓 3 步
- [ ] 标注自检通过（次数对得上 L、2L+1、7L+1）
- [ ] 检查 backward 有没有被漏；用 TID 验证线程；用差分法恢复
- [ ] 利用率用 `cuda_gpu_kern_sum` 总时间做分子，不用 NVTX 区间汇总
- [ ] 嵌套区间的行不能相加（累计计入）；区间名带 `:` 前缀；`Total Range Time` 是 CPU 时间
- [ ] 只看构成，不看绝对时间

**ncu**
- [ ] 利用率接近 1 才上 ncu；`mkdir -p results/ncu`；先 `-c 1`
- [ ] 过滤：forward 用 NVTX，backward 用名字；形状、dtype、TF32 与真实运行一致
- [ ] 先 `basic` 整体扫，再 `full` 深挖少数几个
- [ ] 看 **DRAM Throughput** 判断带宽受限，不是 `Memory Throughput`
- [ ] 不比较 ncu 的 duration；用搬运量、有效带宽、指令数做验算
- [ ] 提示里的"可提速 N%"要自己折算量级

**回填**
- [ ] 总量对了也要拆成分量再比
- [ ] "ncu 证据"写具体指标
- [ ] 区分"实测"与"推断"，推断要标出来
- [ ] 最后写"最出乎意料的三项"

---

## 9. 版本相关，使用前核对

| 项 | 核对方式 |
|---|---|
| nsys 报告名 | `nsys stats --help-reports` |
| CSV 文件名和列名 | `glob` 找文件、`df.columns` 看列（本项目的列名见 3.5） |
| sqlite 的表名和列名 | `PRAGMA table_info(表名)` |
| ncu 的 section 与指标名 | `ncu --list-sections`、`ncu --query-metrics` |
| ncu 对 sm_120 的支持 | `ncu --version`，以能否识别你的卡为准 |
| kernel 名 | 以 nsys 实际清单为准（依库版本与架构变化） |
| NVTX 区间名前缀 | 本项目是 `:`，换版本可能不同 |