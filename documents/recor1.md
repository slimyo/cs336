# CS336 A2 Part 1：Profiling 与 Benchmarking
> **环境**：2× RTX 5070 Ti（16 GiB，sm_120，70 SM），torch 2.11.0+cu130，Python 3.13。以下除特别注明外均在单卡独占下测得。
> **数据使用原则**：3.2 的基线已与 nsys、独立复跑对账，作为**绝对数值基准**；3.6 / 3.7 在各自会话内 vanilla 与 compiled 同进程测得，用于**相对比较（加速比）**；不同会话之间存在约 ±20% 的系统性环境偏差（证据见 3.6 与 3.7 的交叉核对），因此不同节的绝对数值不做跨表相除。
> 标 **［待补］** 的地方是仍缺数据。
---
## 3.1 benchmark 脚本
- 脚本路径：`scripts/benchmark.py`、`scripts/attn_microbench.py`、`scripts/mem_profile.py`。
- **TF32 设置：关闭**（`torch.backends.cuda.matmul.allow_tf32 = False`，`torch.set_float32_matmul_precision("highest")`）。每个配置开始前显式设置并打印读回值。
  - 旁证：nsys 里 fp32 GEMM 的 kernel 名是 `cutlass_80_simt_sgemm_*`（SIMT 路径），不是 tensorop，确认 fp32 基线没有走 TF32。
- 阶段计时方式：**同一个训练步里用 4 个 CUDA event 切出 forward(+loss) / backward / optimizer，整步只同步一次**；不用“整步时间减去单独前向”的相减法（相减法会把 no_grad 前向混进来，实测偏差见 3.2 交叉核对）。
- 统计：样本标准差（n−1），warmup 5，测量 10 步。
- 显存：`torch.cuda.memory._record_memory_history` 录制快照导出 pickle，本地解析 + pytorch.org/memory_viz 查看。
---
## 3.2 端到端计时
固定 warmup=5、测量 10 步，context 取 128 / 256 / 512 / 1024。
### 3.2 端到端计时（ms，均值±样本std）
| 模型 | ctx | dtype | forward 均值±std (ms) | backward 均值±std (ms) | optimizer step 均值±std (ms) | 状态 |
|---|---|---|---|---|---|---|
| small | 128 | fp32 | 13.16±0.16 | 20.00±0.23 | 13.07±0.10 | OK |
| small | 256 | fp32 | 16.99±0.05 | 31.60±0.05 | 12.98±0.05 | OK |
| small | 512 | fp32 | 38.49±0.02 | 76.09±0.04 | 13.13±0.10 | OK |
| small | 1024 | fp32 | 100.59±0.06 | 198.40±0.06 | 13.04±0.08 | OK |
| medium | 128 | fp32 | 25.99±0.03 | 51.28±0.07 | 47.20±0.09 | OK |
| medium | 256 | fp32 | 47.42±0.14 | 93.38±0.05 | 47.17±0.03 | OK |
| medium | 512 | fp32 | 112.95±0.10 | 218.83±0.45 | 47.62±0.26 | OK |
| medium | 1024 | fp32 | 290.96±0.23* | N/A | N/A | OOM(train) |
| large | 128 | fp32 | 58.75±0.02* | N/A | N/A | OOM(train) |
| large | 256 | fp32 | 108.35±0.06* | N/A | N/A | OOM(train) |
| large | 512 | fp32 | 255.48±0.34* | N/A | N/A | OOM(train) |
| large | 1024 | fp32 | 630.42±0.22* | N/A | N/A | OOM(train) |
\* 是 no_grad 前向；其余 forward 是训练中的前向（含 loss、带计算图）。
### 派生数据（由上表计算，B=4，FLOPs 用 11.3 的公式）
| 模型 | ctx | 整步 = f+b+o (ms) | bwd ÷ fwd | optimizer 占整步 | forward TFLOP/s | 整步 TFLOP/s (3F÷整步) |
|---|---|---|---|---|---|---|
| small | 128 | 46.23 | 1.52 | 28.3% | 9.59 | 8.19 |
| small | 256 | 61.57 | 1.86 | 21.1% | 15.15 | 12.54 |
| small | 512 | 127.71 | 1.98 | 10.3% | 13.87 | 12.54 |
| small | 1024 | 312.03 | 1.97 | 4.2% | 11.39 | 11.01 |
| medium | 128 | 124.47 | 1.97 | 37.9% | 16.52 | 10.35 |
| medium | 256 | 187.97 | 1.97 | 25.1% | 18.38 | 13.91 |
| medium | 512 | 379.40 | 1.94 | 12.6% | 15.89 | 14.19 |
no_grad 前向的 TFLOP/s：medium/1024 为 13.0；large 为 16.9（128）、18.5（256）、16.1（512）、13.7（1024）。fp32 实测可达算力约 33.4 TFLOP/s（11.5 用的 P），前向大致在 **40%～55%** 水平。
### 交叉核对（测量方法本身是否可靠）
| 核对项 | 数据 | 结论 |
|---|---|---|
| 阶段之和 vs 第一版脚本测的整步 | 127.71 ms vs 127.85 ms | 差 0.1%，阶段计时自洽 |
| 训练前向 vs no_grad 前向（small/512） | 38.49 vs 37.72 ms | 训练前向慢约 2%（建图、保存张量、loss） |
| backward：同步内切段 vs 相减法 | 76.09 vs 77.18 ms | 相减法高估 1.09 ms（1.4%），方向正是预测的：减掉的是 no_grad 前向 |
| 两次独立运行的复现性（small/512，同会话） | fwd 38.49 / 38.56，bwd 76.09 / 76.11 | run-to-run 差 0.2% 以内 |
### 需要回答
**哪个阶段最慢？随 ctx 如何变化？**
- **backward 在所有配置里都是最慢的**。bwd÷fwd 在 ctx ≥ 256 时稳定在约 1.9～2.0（small / 128 只有 1.52，原因见下），与理论一致：反向对每个前向 GEMM 要做两次 GEMM（对输入求导和对权重求导），FLOPs 约为前向的 2 倍。
- **forward 与 backward 随 ctx 增长是“先次线性、后超线性”**：
  - ctx 小时（small / 128 → 256，token 翻倍，forward 只涨 1.29×）：**发射受限**。small 的 forward 有约 641 个 kernel，13.16 ms ÷ 641 ≈ 20 µs/个，正是 PyTorch eager 单算子的 CPU 发射开销量级——此时时间由 kernel 个数决定，不由 FLOPs 决定，算力只有 9.6 TFLOP/s。
  - ctx 中等（256 → 512）：接近线性，算力升到约 14～18 TFLOP/s 的最高点。
  - ctx 大（512 → 1024，small 涨 2.61×）：**超线性**，attention 的 T² 项（缩放、mask、softmax 这些访存受限运算）增长更快，TFLOP/s 回落到 11 左右。medium/large 的 no_grad 前向同样如此（large：18.5 → 16.1 → 13.7）。
- **optimizer 与 ctx 无关，只与模型有关**：small 约 13.0 ms，medium 约 47.2 ms，四个 ctx 下几乎不变。因此它占整步的比例随 ctx 增大迅速下降（small：28% → 21% → 10% → 4%；medium：38% → 25% → 13%）。短 ctx 时 optimizer 能与 backward 一个量级（medium / 128：47.2 vs 51.3 ms）。
- **optimizer 为什么是这个时间**：
  - medium ÷ small 的 optimizer 耗时之比 = 3.61，参数量之比 = 3.29，参数张量个数之比 = 1.97。**耗时随参数量而不是随 kernel 个数缩放，所以 medium 上已是带宽受限，不是发射受限**（small 上两者混合：nsys 里 GPU 利用率 0.49，benchmark 墙钟下约 0.8）。
  - AdamW 的带宽下限约 28 B/参数（读 p、g、m、v，写 p、m、v）：small 约 4.1 ms、medium 约 13.4 ms（按实测峰值带宽 882 GB/s）。实测是下限的 **3.2～3.5 倍**，说明 staff 的逐参数循环把一次更新拆成约 16 个 elementwise kernel，每个都完整读写一遍张量。
  - 优化方向：fused / foreach AdamW 把多遍合成一遍，上限接近 3 倍；对 small 还能去掉发射开销。
**测量方差大不大？为什么？**
- **很小**：std 全部不超过 0.5 ms，相对 std 绝大多数在 0.3% 以内，最大的是 small / 128（forward ±0.16 = 1.2%，backward ±0.23 = 1.1%）。
- 原因：独占 GPU；预热后进入稳态；每步之间同步，用 event 在 GPU 侧计时，不含 CPU 空隙；固定形状与种子；没有 profiler 开销。
- **为什么 small / 128 相对方差最大**：发射受限，GPU 在等 CPU，CPU 侧的调度抖动直接进入测得时间；大 kernel 的时间由 GPU 决定，更稳定。
- 两点提醒：
  1. n = 10 时 std 本身只是粗估；
  2. **std 小不等于准**。系统误差（GPU 频率、功耗限制、别的进程、跨会话状态）不会体现在 std 里。本报告里 3.6 的第一次测量（seq=256 异常）与 3.7 的跨会话 ~20% 偏差，都是靠“与其他测法对账”而不是靠 std 发现的。
**OOM 的解释**：large 完整训练步的静态部分 16P ≈ 14.45 GiB，加上激活，16 GiB 的卡放不下；medium / 1024 的激活估算约 23.6 GiB（见 3.8 的拟合），也放不下。no_grad 前向因为没有梯度、Adam 状态和保存的激活，仍然能跑。
---
## 3.3 warmup 影响
**需要回答**：没有 warmup 时为什么慢且方差大？
**［待补：数据］** 实验设计与归因方法如下。
### 实验：不预热，记录最初 N 步每一步每个阶段的时间
```python
# 用 phase_step（同一步里切 forward / backward / optimizer），不做任何预热，连续测 12 步
import torch
torch.cuda.synchronize()
rows = [phase_step(model, x, y, opt, ce, amp) for _ in range(12)]     # 每行 [fwd, bwd, opt]
for i, r in enumerate(rows): print(i, [f"{t:.2f}" for t in r])
```
在**全新进程**里跑 small / 512 / fp32，另跑一次“预热 5 步后的稳态”作对照。

| 步 | forward | backward | optimizer | 与稳态之比 |
|---|---|---|---|---|
| 0（首步） | ［待补］ | ［待补］ | ［待补］ | |
| 1 | ［待补］ | ［待补］ | ［待补］ | |
| 2 | ［待补］ | ［待补］ | ［待补］ | |
| …… | | | | |
| 稳态（预热 5 步后） | 38.49 | 76.09 | 13.13 | 1.00 |

### 对每个候选因素的预期与归因办法

| 因素 | 是否适用本项目 | 预期表现 | 怎么单独验证 |
|---|---|---|---|
| CUDA context 初始化 | 适用，但发生在 `.cuda()` 时，**不在计时区内** | 若在计时区内，首次是秒级 | 单独计时第一次 `torch.zeros(1, device="cuda")` |
| cuBLAS 句柄、workspace、按形状选算法 | 适用 | 首步 forward 慢；每种新 GEMM 形状第一次也慢 | 第一次与第二次 matmul 的时间对比 |
| CUDA 模块懒加载 | 适用 | 首步慢；每种新 kernel 首次出现时慢（backward 的新 kernel 在第 0 步才首次出现） | 逐步看 backward 是否在第 0 步特别慢 |
| allocator 首次分配与扩容 | 适用 | 首步慢；显存池长大后稳定 | `memory_reserved` 前几步的增长曲线 |
| Adam 状态首次分配 | **适用，且只影响 optimizer** | 第 0 步 optimizer 明显慢（111 个参数各分配两份状态），第 1 步起恢复 | 看 optimizer 一列只在第 0 步异常 |
| JIT 编译 / torch.compile | eager 模式不适用；compile 模式非常适用（秒级） | — | 3.7 的 compile 首步 |
| cudnn autotune | **不适用**：模型里没有卷积 | 不会出现 | — |
| L2 cache 冷启动 | 适用但影响很小 | 微秒级 | 与其他因素相比可忽略 |
| GPU 频率爬升 | 适用 | 前几步频率未升满，时间偏长 | `nvidia-smi -q -d CLOCK` 或多跑几步看是否继续下降 |

**预期图景**：首步最慢（多种初始化叠加），第 1～2 步有残余，之后进入稳态；不预热时方差大主要来自首步离群值。所以统计必须排除预热，且预热要覆盖 Adam 首次 step 和 backward 所有 kernel 的首次出现。
---
## 3.4 nsys profiling

| 模型 / ctx | 阶段 | 耗时最长 kernel | 占该阶段 % | 调用次数 | 是否 GEMM |
|---|---|---|---|---|---|
| small / 512 | forward | cuBLASLt fp32 GEMM（`magma_sgemmEx_kernel`） | 47.5% | 85 / 步 | 是 |
| small / 512 | backward | `cutlass_80_simt_sgemm_128x64_8x5_nt` | 14.6% | 72 / 步 | 是 |
| small / 512 | optimizer | `vectorized_elementwise_kernel`（乘法） | 44.5% | 666 / 步 | 否 |
| 问题 | 答案 |
|---|---|
| forward 中占比最高的 kernel？ | `magma_sgemmEx_kernel`，47.5%。73 次来自 Linear，12 次来自 QKᵀ |
| 同一 kernel 在 fwd+bwd 中占比变化？ | 47.5% → 16.6%。backward 里它只剩 12 次，GEMM 换成 cutlass 的多种 tile。GEMM 整类占比：forward 65.7% → backward 57.4% → 整步 53.5% |
| 除 matmul 外占时间最多的 kernel？ | 单个 kernel 是 vectorized elementwise 乘法，占整步 8.7%；按类别，elementwise 占整步 43.2% |
| 一次完整训练步中 matmul 占比？ | 53.5% |
| attention 内 softmax 与 matmul 的耗时比，与 FLOPs 比相比如何？ | 耗时比 2.15 : 1，FLOPs 比 0.020 : 1。按单位 FLOPs 的耗时算，softmax 比 matmul 高约 100 倍量级 |

注释：
- backward 第一名只有 14.6%，是因为同类 GEMM 按 tile 配置拆成几个名字（`128x64 nt` 72 次、`128x64 nn` 60 次、`256x128 nn` 25 次）。
- optimizer 的 666 次是 111 的倍数：111 个参数张量逐个发射，每个参数的同类 kernel 出现 6 次。
- softmax 是 5 个 kernel 拼成的（max、sub、exp、sum、div），不是融合 kernel。
- 该份 sqlite 的绝对毫秒数偏慢（采集开销，已与 3.2 对账确认），只引用占比与次数——它们在多次采集里一致。
结论：
- **forward**：时间由 GEMM 主导（65.7%）。
- **backward**：GEMM 仍占约一半，分散在多种 cutlass tile；elementwise 个数约是 forward 的 2 倍，单次耗时更长。
- **optimizer**：全是 elementwise、逐参数发射。结合 3.2：small 上有发射开销成分，medium 上主要是多遍带宽受限。优化方向是融合。
- **整步**：GEMM 53.5% + elementwise 43.2% ≈ 97%。
- **attention**：softmax 的 FLOPs 只有 matmul 的 2%，耗时却是其 2 倍多——访存密集（ncu 里 DRAM 约 83%）。这是 Part 2 FlashAttention 要解决的核心问题。
---
## 3.5 混合精度（bf16 autocast）
forward / backward 取训练步中的阶段均值；显存单位 GiB（完整训练步峰值）。

| 模型 | ctx | dtype | forward (ms) | backward (ms) | 峰值显存 (GiB) | 加速比（fwd+bwd，相对 fp32） | loss0 |
|---|---|---|---|---|---|---|---|
| small | 512 | fp32 | 38.56 | 76.11 | 5.03 | 1.00× | 9.2808 |
| small | 512 | bf16 autocast | 20.18 | 44.29 | 4.25 | 1.78× | 9.2808 |
| medium | 512 | fp32 | 113.01 | 218.78 | 13.72 | 1.00× | 9.3120 |
| medium | 512 | bf16 autocast | 57.90 | 126.74 | 11.62 | 1.80× | 9.3120 |
| large | 256 | fp32 | ［待补］ | ［待补］ | ［待补］ | 1.00× | |
| large | 256 | bf16 autocast | ［待补］ | ［待补］ | ［待补］ | | |

**large / 256 的补测方法**：完整训练步 OOM，所以只测 fwd+bwd（不创建优化器），峰值取反向结束时：
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model large --ctx 256 --dtype float32 bfloat16 --no-opt
```
估算：large 的权重加梯度 8P ≈ 7.2 GiB，加激活应该能放下；若 fp32 仍 OOM，也是有效结果，记 OOM 并写明静态部分大小。

### 分阶段加速比

| 模型 | forward | backward | fwd+bwd | 峰值显存比 |
|---|---|---|---|---|
| small / 512 | 1.91× | 1.72× | 1.78× | 0.85× |
| medium / 512 | 1.95× | 1.73× | 1.80× | 0.85× |

**为什么只有约 1.8×，而不是 Tensor Core 的“数倍”？** 用 Amdahl 定律：
- fp32 基线里，fwd+bwd 的 GEMM 占 57.4%，非 GEMM（主要是 elementwise）占 42.6%。
- **即使 GEMM 变成零耗时，整体上限也只有 1/0.426 ≈ 2.35×。** 实测 1.78×。
- 反推：若非 GEMM 完全没加速，GEMM 需快 4.2×；若非 GEMM 快 1.5×，GEMM 需快 2.1×；若非 GEMM 快 2×，GEMM 需快 1.65×。区分到底是哪种需要对 bf16 再跑一次 nsys（［待补］，非必答）。
- 非 GEMM 难以加速的原因（由 3.8 的 dtype 实测推出）：残差流和 RMSNorm 仍是 fp32；每个 Linear 的输入要先转 bf16；手写 softmax 的 `exp`、`sum` 升到 fp32。
- optimizer 不受影响（权重、梯度、Adam 仍 fp32），所以**整步**加速比更低：small 为 127.7 / 77.6 ≈ 1.6×。
**峰值显存为什么只降 15%？**
- 16P 静态部分完全不变（fp32 主权重、梯度、Adam 状态都在），只有激活能省；
- bf16 autocast 还会多缓存一份 bf16 权重（约 2P ≈ 0.79 GiB），见 3.8。
### 附加：autocast 下各类算子的 dtype（hook 实测，small / ctx 128）

| 对象 | dtype | 为什么 |
|---|---|---|
| 模型参数 | float32 | autocast 不改参数；保留 fp32 主权重，避免小更新被 bf16 舍入掉 |
| 所有 Linear 输出 | bfloat16 | `linear` 在 autocast 的低精度名单里 |
| LayerNorm / RMSNorm 输出 | float32 | staff 的 RMSNorm 内部先升 fp32 再转回输入类型，输入（残差流）是 fp32 |
| TransformerBlock 输出（残差流） | float32 | embedding 输出 fp32，bf16 子层输出加回去时类型提升为 fp32 |
| logits | bfloat16 | `lm_head` 是 linear |
| loss | float32 | 在 autocast 之外、对 `logits.float()` 计算 |
| 梯度 | float32 | 参数是 fp32，autograd 把 bf16 副本的梯度转回参数 dtype |

数值检查：fp32 与 bf16 的 loss0 在显示精度内一致（9.2808、9.3120），说明 autocast 没有破坏数值；更强的检验可打印更多位比较相对差（预期 1e-3 量级）。
---
## 3.6 attention 微基准（vanilla vs torch.compile）
固定 batch=8，单头，fp32（TF32 关），d_model ∈ {16, 32, 64, 128}，seq ∈ {256, 1024, 4096, 8192, 16384}。本节数据为**重测版**（打乱配置顺序、加大预热），重测消除了旧版 seq=256 列的测量假象（旧版 d=128 误测为 0.06/0.26 ms，违反 FLOPs 趋势；重测后为 0.17/0.45 ms，与其他 d 一致）。
**理论预测（先写后测）**：backward 前保存 ≈ 1 × B·S²·4B；forward 瞬态峰值 ≈ 2 × B·S²·4B。seq=16384 → 峰值 ~16 GiB → **预测 OOM 首现点**；seq=8192 → ~4 GiB 可过。
### forward / backward 时间 (ms)，vanilla（格内为 fwd / bwd）

| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.13 / 0.45 | 0.20 / 0.44 | 4.78 / 10.42 | 20.20 / 40.38 | OOM |
| 32 | 0.07 / 0.43 | 0.20 / 0.65 | 4.83 / 10.46 | 20.26 / 40.59 | OOM |
| 64 | 0.15 / 0.43 | 0.22 / 0.49 | 5.60 / 10.31 | 21.98 / 42.78 | OOM |
| 128 | 0.17 / 0.45 | 0.51 / 0.79 | 7.80 / 15.60 | 33.51 / 62.67 | OOM |

（seq=256 列各 d 的差异在 0.1 ms 内，处于发射开销主导区，不反映 FLOPs 差异。）

### backward 前显存 (GiB) —— 对账基准 1·B·S²·4B

| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.02 | 0.05 | 0.52 | 2.03 | OOM |
| 32 | 0.02 | 0.05 | 0.53 | 2.05 | OOM |
| 64 | 0.02 | 0.05 | 0.55 | 2.08 | OOM |
| 128 | 0.02 | 0.06 | 0.58 | 2.14 | OOM |

### [附加] forward 瞬态峰值显存 (GiB) —— 对账基准 2·B·S²·4B，OOM 咬人的位置

| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.02 | 0.08 | 1.03 | 4.04 | OOM |
| 32 | 0.02 | 0.08 | 1.04 | 4.05 | OOM |
| 64 | 0.02 | 0.09 | 1.05 | 4.09 | OOM |
| 128 | 0.02 | 0.10 | 1.09 | 4.17 | OOM |

### torch.compile 对比

| d_model | seq | vanilla fwd | compiled fwd | vanilla bwd | compiled bwd | 加速比(fwd+bwd) | vanilla pre-G | compiled pre-G |
|---|---|---|---|---|---|---|---|---|
| 128 | 4096 | 7.80 | 6.66 | 15.60 | 11.79 | 1.27× | 0.58 | 0.58 |
| 64 | 8192 | 21.98 | 19.48 | 42.78 | 26.93 | 1.40× | 2.08 | 2.08 |
| 128 | 8192 | 33.51 | 31.92 | 62.67 | 46.44 | 1.23× | 2.14 | 2.14 |

### 需要回答
**最小 OOM 的配置是什么？显存主要来自哪个张量？**
- 网格里最小的 OOM 配置是 **seq = 16384**（四个 d_model 同时 OOM，最小的是 d_model = 16）。显存几乎与 d_model 无关：同一 seq 下，pre-G 在 d 从 16 到 128 只差 0.06 GiB（4096 列：0.52 → 0.58）。
- 来源是 **B × S × S 的 fp32 分数矩阵**：8 × 16384² × 4 B = 8.0 GiB 一个。
- **OOM 发生在 forward，不是 backward**：forward 里 scores 和 softmax 输出同时存活，需要 2 × 8.0 = 16.0 GiB（17.2 GB），超出卡的可用显存。
**手算 backward 前保存的显存与实测是否吻合？**

| S | 1 × B·S²·4B | 实测 backward 前 | 2 × B·S²·4B | 实测 forward 瞬态峰值 |
|---|---|---|---|---|
| 4096 | 0.50 GiB | 0.52～0.58 | 1.00 GiB | 1.03～1.09 |
| 8192 | 2.00 GiB | 2.03～2.14 | 4.00 GiB | 4.04～4.17 |
| 16384 | 8.00 GiB | OOM | 16.00 GiB | OOM |

- **结论：保存给反向的只有 1 个 S² 张量；2 个 S² 同时存活只出现在 forward 的瞬间，这才是峰值、才是 OOM 的直接原因。**
- 为什么：softmax 的反向只需要它的**输出** y（$\partial x = y\odot(g-\langle g,y\rangle)$），不需要输入 scores；而 $PV$ 这个矩阵乘保存的输入正好也是 y，同一张量只存一份。scores 在算完 softmax 后即被释放。
- 固定的约 0.02 GiB（seq=256 列）是 cuBLAS workspace 等固定开销；d 增大的少量增长来自 Q、K、V 与输出（B·S·d·4B）。
- 与 3.8 对照：完整模型里 staff 的手写 softmax 保存的是 **2 个** S² 张量（3.8 拟合出 c = 2.00）。**同样的数学，不同实现，保存的显存差 2 倍**——vanilla 用的是融合 softmax kernel（只保存输出）。
**如何避免这个显存（引出 FlashAttention）？**

| 办法 | 原理 | 代价 |
|---|---|---|
| **FlashAttention / SDPA 融合 kernel** | 分块 + online softmax，**不物化 S×S 矩阵**；反向用保存的 logsumexp 重算 | 显存 $O(S^2)→O(S)$；FLOPs 不变，但省掉 S² 张量的反复读写。flash 后端通常要求 fp16/bf16 |
| 激活重算（checkpoint） | 不保存，反向重算前向 | 多约一次前向计算 |
| 按 query 分块 | 每次只算一块行 | 实现复杂，反向要小心 |
| bf16 | 张量减半 | 只减半，不改变 $O(S^2)$ |
| **torch.compile（本表数据证明不行）** | 融合 elementwise | **pre-G 一点没降（0.58→0.58，2.08→2.08，2.14→2.14）**：compile 不改算法，S×S 矩阵仍要物化 |

### 其他观察
- **时间随 S 的变化**：4096 → 8192（S² 变 4 倍），forward 涨 4.2～4.3×，backward 涨 3.9～4.0×——接近 S² 的平方律，是访存受限运算的标志。d = 16 与 32 几乎一样（20.20 vs 20.26），说明时间由 S² 张量的搬运决定，与 d 无关。粗估：S = 8192 时一个 S² 张量 2.1 GB，约 10 遍读写 ≈ 21 GB，在 882 GB/s 下约 24 ms，与实测 20 ms 同量级（遍数是粗估）。
- **d = 128 明显更慢**（33.51 ms @ 8192）：QKᵀ 与 PV 的 GEMM（共约 275 GFLOP）开始贡献可观的计算时间。
- **backward ÷ forward** 约 1.7～2.0。
- **compile 的收益**：1.23～1.40×，d 越大收益越小（8192 列：d=64 为 1.40×，d=128 为 1.23×），因为 GEMM 比重变大，compile 默认不加速 cuBLAS GEMM，只融合 elementwise。
---
## 3.7 完整模型 compile
（ms；vanilla 与 compiled 在**同一会话**内先后测得，加速比为同会话内比较。fwd 为训练步中带图前向段（含 loss），与 3.2 的 no_grad 前向口径不同。）

| 模型 | ctx | vanilla fwd | compiled fwd | vanilla fwd+bwd | compiled fwd+bwd | vanilla 全步 | compiled 全步 | 全步加速比 |
|---|---|---|---|---|---|---|---|---|
| small | 512 | 46.60 | 35.31 | 137.52 | 99.32 | 152.76 | 114.68 | **1.33×** |
| medium | 512 | 133.42 | 104.84 | 389.30 | 293.02 | 444.66 | 349.47 | **1.27×** |

### 跨会话系统性偏差的说明（数据可信性）
本节 vanilla 的绝对数值比 3.2 的同配置基线系统性偏慢约 18%～21%：

| 配置 | 3.2 基线 | 3.7 vanilla | 比值 |
|---|---|---|---|
| small / 512 forward | 38.49 | 46.60 | 1.21× |
| small / 512 fwd+bwd | 114.58 | 137.52 | 1.20× |
| medium / 512 forward | 112.95 | 133.42 | 1.18× |
| medium / 512 fwd+bwd | 331.78 | 389.30 | 1.17× |

三点判断：
1. **偏差是会话级、系统性的**：small 与 medium 的慢的比例一致（1.17～1.21×），且 3.6 的重测与首测之间也观察到同量级的会话间漂移（如 d=16/S=8192 forward 25.06 → 20.20 ms）。最可能的环境因素是 GPU 时钟/热状态（3.6 的 OOM 实验后未完全恢复）。
2. **加速比不受影响**：vanilla 与 compiled 在同一会话内相隔几分钟测得，环境因素对两者同等作用，因此 **1.27×～1.33× 的相对加速比是有效结论**；但 3.7 的绝对数值不应与 3.2 跨表混用。
3. **与 3.2 一致性的旁证**：3.7 compiled 的全步（114.68 / 349.47 ms）与 3.2 eager 基线（127.71 / 379.40 ms）方向相符——compiled 确实快于 3.2 的 eager，进一步排除“3.7 整体计时方式错误”的可能。
### compile 在整模型里做了什么

| 方面 | 结论 / 预期 |
|---|---|
| 融合 | 把 RMSNorm、RoPE、缩放、mask、softmax 各步、SiLU、残差等 elementwise 与归约融合成少量 Triton kernel；kernel 数下降，HBM 读写遍数下降 |
| GEMM | 默认仍用 cuBLAS，不变 |
| 上界 | 只能加速非 GEMM 部分：fwd+bwd 里非 GEMM 约占 42.6%。实测全步加速 1.27～1.33×，低于 Amdahl 上界 2.35×，也低于 3.6 微基准（attention 内 elementwise 密集，1.23～1.40×），符合“整模型 GEMM 占比更高、收益更小”的预期 |
| 显存 | 不省 S² 显存（3.6：pre-G 不变）；整模型同理，compile 不改变保存策略 |
| 编译开销 | 首次调用含秒级编译（3.6 实测约 1–2 s），warmup 必须覆盖；形状变化触发重编译 |

---
## 3.8 显存 profiling
方法：`torch.cuda.memory._record_memory_history(max_entries=...)` 导出 pickle，本地解析（segments / blocks / device_traces）+ pytorch.org/memory_viz 查看。
### 峰值显存总表（单位 GiB）

| 模型 | ctx | dtype | forward 峰值(no_grad) | 带图 forward 峰值 | 完整训练步峰值 | 峰值阶段 | 理论 4P | 理论 16P | 峰值估算(11.6) | 估算÷实测 | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| medium | 128 | fp32 | 1.64 | 3.27 | 6.52 | optimizer | 1.58 | 6.31 | 8.44 | 1.29 | OK |
| medium | 128 | bf16 autocast | 2.37 | 3.46 | 6.62 | forward | 1.58 | 6.31 | 8.44 | 1.27 | OK |
| medium | 256 | fp32 | 1.70 | 5.32 | 8.49 | forward | 1.58 | 6.31 | 10.96 | 1.29 | OK |
| medium | 256 | bf16 autocast | 2.39 | 4.84 | 8.01 | forward | 1.58 | 6.31 | 10.96 | 1.37 | OK |
| medium | 512 | fp32 | 1.88 | 10.56 | 13.72 | forward | 1.58 | 6.31 | 17.11 | 1.25 | OK |
| medium | 512 | bf16 autocast | 2.54 | 8.45 | 11.62 | forward | 1.58 | 6.31 | 17.11 | 1.47 | OK |

### 显存构成（GiB）：forward 结束时 与 backward 结束时

| 模型 | ctx | dtype | fwd结束: 权重 | Adam | 激活/其他 | 合计 | bwd结束: 权重 | 梯度 | Adam | 其他 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| medium | 128 | fp32 | 1.58 | 3.15 | 1.66 | 6.39 | 1.58 | 1.58 | 3.15 | 0.02 | 6.33 |
| medium | 128 | bf16 autocast | 1.58 | 3.15 | 1.85 | 6.57 | 1.58 | 1.58 | 3.15 | 0.02 | 6.33 |
| medium | 256 | fp32 | 1.58 | 3.15 | 3.68 | 8.41 | 1.58 | 1.58 | 3.15 | 0.02 | 6.33 |
| medium | 256 | bf16 autocast | 1.58 | 3.15 | 3.18 | 7.91 | 1.58 | 1.58 | 3.15 | 0.02 | 6.33 |
| medium | 512 | fp32 | 1.58 | 3.15 | 8.84 | 13.57 | 1.58 | 1.58 | 3.15 | 0.02 | 6.33 |
| medium | 512 | bf16 autocast | 1.58 | 3.15 | 6.70 | 11.43 | 1.58 | 1.58 | 3.15 | 0.02 | 6.33 |

### 解读
**① 测量自洽性**
- backward 结束时权重 1.58、梯度 1.58、Adam 3.15 GiB，**与理论 4P / 4P / 8P 精确吻合**（P = 423.2M），三种 dtype、三个 ctx 完全相同。
- 其余“其他”只有 0.02 GiB：没有泄漏，backward 把保存的激活全部释放了。
- 同一配置在 3.5 与 3.8 的完整步峰值一致（medium / 512：fp32 13.72、bf16 11.62），两个脚本互相印证。
- 快照解析的总量交叉验证：medium / 128 fp32 快照的活跃块合计 **881 个 / 6.33 GiB**，与构成表的 6.33 GiB 精确一致。
**② 峰值出在训练步的哪一刻**

| 现象 | 数据 | 解释 |
|---|---|---|
| ctx 256、512（fp32 与 bf16）的峰值在 **forward 末尾**，不在 backward | medium / 512 fp32：fwd 结束 13.57，峰值 13.72 | forward 结束时激活最多、梯度还没生成。backward 每消费一块激活就释放一块，同时梯度（共 1.58 GiB）逐步增加；只要激活比梯度大，backward 的占用就低于 forward 末尾 |
| 峰值比 fwd 结束时还高约 0.15 GiB | 13.72 − 13.57 = 0.15 GiB | 恰好 ≈ 2 个 N×V 的 fp32 张量（2 × 78 MiB = 0.153 GiB）：**是 loss 计算里 logits 及其 fp32 副本**，峰值在 forward 的最后一步。快照 trace 里 `loss_fn` 处两个 78.12 MiB 分配直接证实（见下） |
| ctx 128（fp32）的峰值在 **optimizer** | 6.52 vs bwd 结束 6.33 | 激活很小，静态 16P 主导；optimizer 的逐参数临时张量（最大 embedding / lm_head 约 39 MiB 每个，约 5 份）把峰值推到 6.52 |

**③ 估算÷实测 = 1.25～1.47**
- 11.6 的公式高估 25%～47%。拆开：T² 项系数 c 取 2 是对的（拟合 2.00），**线性项取多了**（拟合的等效份数 29.5，公式里用了 40，即 other 取 20 实际约 9.4）。
- bf16 下高估更多，因为公式按 fp32 口径写的。
### 附加问题
**激活值显存与 ctx 的关系？**

| ctx | 128 | 256 | 512 |
|---|---|---|---|
| fp32 保存的激活 (GiB) | 1.64 | 3.66 | 8.82 |
| bf16 autocast | 1.82 | 3.16 | 6.68 |

- fp32：ctx 每翻倍激活增大 2.2×、2.4×；128 → 512 共 5.4×。**介于纯线性（4×）与纯 T²（16×）之间，线性项与 T² 项同时存在**，ctx 越大 T² 占比越高。
- 三个点拟合 `act = k + a·N + c·B·h·T²`：
  - T² 项：**c = 1.99**（每层约保存 2 个 B·h·T² 的 fp32 张量）；
  - 线性项：每层约 **29.5 份** N×d 张量（公式里取 40）；
  - 常数项 k = −0.01 GiB ≈ 0，说明 fp32 下模型形式正确。
- **c = 2 的来源**：staff 的 softmax 是手写 max-sub-exp-sum-div，`exp` 的输出（反向要用）和 softmax 输出 P（PV 要用）是两个不同张量，各存一份；3.6 的融合 softmax 只存输出，c = 1。该解释与 kernel 清单和拟合值吻合，并经下面的快照逐张量核对支持。
- 分解（medium / 512 / fp32）：T² 部分 ≈ 3.0 GiB（34%），线性部分约 5.8 GiB（66%）。
- **T² 与线性项相等的 ctx ≈ 990**。ctx 1024 时激活 ≈ 23.6 GiB，加静态 6.3 GiB ≈ 30 GiB，**解释 medium / 1024 的 OOM**。
**2.7B 理论峰值？**
- P = 3.41B（SwiGLU 三矩阵、词表 10000）。
- batch=4、ctx=128、fp32：静态 16P = **50.8 GiB**（权重 12.7 + 梯度 12.7 + Adam 25.4）；激活（校准后 c=2.0、other=9.4）≈ 5.2 GiB；**合计 ≈ 55.9 GiB（60.1 GB）**。
- bf16 autocast：静态 16P 不变，另加约 2P = 6.3 GiB 的 bf16 权重副本，激活约省 1/3，**合计仍约 60 GiB**，放不进 16 GiB 的卡。
- 仅推理：fp32 权重 12.7 GiB；纯 bf16 权重 6.3 GiB（16 GiB 卡只能以 bf16 权重做 forward）。
- 要训练这个规模，需要 ZeRO / FSDP 分片、激活重算、低精度优化器状态或更大的卡。
**最大的单次分配（如 attention score）是多大？——快照解析实测**
从快照 pickle 的 `segments`（当前存活块）与 `device_traces`（历史 alloc 事件，含已释放的瞬态张量）解析；大小用 `requested_size` 口径（对齐前的真实请求量）。理论值：logits = B·T·V·4B（V=10000），scores = B·h·T²·4B（h=16），最大参数张量（embedding/lm_head）= V·d·4B = 39.06 MiB。

| ctx | 实测最大单次 alloc | 来源（调用栈） | 理论 logits | 理论 scores | 最大参数张量 |
|---|---|---|---|---|---|
| 128 | **39.06 MiB** | `optimizer.step`（Adam 状态）与 autograd 线程（梯度） | 19.53 MiB | 4 MiB | 39.06 MiB |
| 256 | **39.06 MiB** | **`loss_fn`（logits）**，与 Adam 状态并列 | 39.06 MiB ✓ | 16 MiB | 39.06 MiB |
| 512 | **78.12 MiB** | **`loss_fn`（logits 及其 fp32 副本，两个 78.12 MiB 分配）** | 78.12 MiB ✓ | 64 MiB | 39.06 MiB |

- **logits 的实测大小与理论精确吻合**（39.06 / 78.12 MiB），调用栈定位在 `loss_fn`（`mem_profile.py:229/230`）。
- **ctx=512 时最大单次分配是 logits（78.12 MiB），压过 attention scores（64 MiB）**——回答了原预期；且 `loss_fn` 处**两个** 78.12 MiB 分配并存（logits 与 fp32 副本），2 × 78.12 MiB = 0.153 GiB，与“峰值比 fwd 结束高 0.15 GiB”精确对上。
- **ctx=128 时最大分配是参数级张量（39.06 MiB 的 Adam 状态/梯度）**，logits（19.5 MiB）和 scores（4 MiB）都不如它——短 ctx 时瞬态张量不是显存的主角。
- **临界点**：logits ∝ T、scores ∝ T²，令 4·T·V = 4·16·T² 得 **T = V/16 = 625**。ctx ≥ 625 后 scores 反超 logits（ctx 1024 时约 256 MiB vs 156 MiB），与 3.8 激活拟合中 T² 项占比随 ctx 升高的结论一致。
- 存活块 top（训练结束后的快照）在三个 ctx 下均为 39.06 MiB（ctx 512 另有无 Python 栈的 64 MiB scores 块，为录制开启前/前向中分配）；梯度的栈是 C++ 帧（autograd 线程），属正常现象。
**混合精度对峰值显存的影响？**

| ctx | 保存的激活 | no_grad 前向峰值 | 完整步峰值 |
|---|---|---|---|
| 128 | 1.11× | 1.44× | 1.02× |
| 256 | 0.86× | 1.40× | 0.94× |
| 512 | 0.76× | 1.35× | **0.85×** |

三个关键点：
1. **bf16 autocast 会缓存一份 bf16 权重副本，大小约 2P**（medium 约 0.79 GiB）。证据：no_grad 前向峰值 1.64 → 2.37 GiB，多出 0.73 GiB ≈ 2P；bf16 激活拟合的常数项 k = 0.76 GiB 亦吻合。
2. 扣掉该常数后，**bf16 的激活实际是 fp32 的约 0.63～0.67 倍**（1.03/1.64、2.37/3.66、5.89/8.82），不是 0.5 倍——残差流、RMSNorm 输出仍 fp32（见 3.5 dtype 表），softmax 的 exp 输出也 fp32（T² 项等效 c = 1.49 ≈ 1 个 fp32 + 1 个 bf16 副本）。
3. **混合精度并不总省显存**：ctx 128 时完整步还多 0.1 GiB（0.79 GiB 固定代价盖过激活节省），到 ctx 512 才净省 15%；静态 16P 不变。想真正省权重显存，只能用纯 bf16 权重（仅推理）或低精度优化器状态。

---

## 附：对“需要回答”项的完成情况

| 小节 | 状态 |
|---|---|
| 3.1 TF32 设置 | 已答：关闭（脚本设置 + nsys kernel 名双证） |
| 3.2 两个问题 | 已答 |
| 3.3 warmup | **［待补］**：实验设计与归因表已备，缺 12 步数据 |
| 3.4 | 已答 |
| 3.5 dtype 表 | 已答；large / 256 两行**［待补］**（`--no-opt` 补测） |
| 3.6 三个问题 | 已答；OOM 原因归为 forward 瞬态峰值；seq=256 异常已由重测排除 |
| 3.7 | 已答：全步加速比 1.33×（small）/ 1.27×（medium）；会话间存在 ~1.2× 系统性环境偏差，相对加速比有效、绝对值不跨表混用 |
| 3.8 四个问题 | **全部已答**；"最大单次分配”由快照解析填齐（ctx≤128 参数级 → ctx≥256 logits → 临界 T≈625 后 scores） |
**遗留待补**（互不阻塞）：3.3 的 12 步 warmup 数据；3.5 large/256 的 fwd+bwd；3.7 如需绝对数值对齐可做同进程 ABAB 复测；3.5 可选对 bf16 补一次 nsys 以拆分 GEMM/非 GEMM 的加速贡献。
