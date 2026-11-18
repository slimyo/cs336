

# M6 报告：CuTe DSL FlashAttention 的调参、基准测试与性能分析
> 状态：前向/反向实现、测试（前向 30/30、autograd 11/11，O4 配置下 autograd 亦 11/11）、tile 扫描（O1 后重跑）、benchmark、ncu 分析（O1 前后各一份）、优化消融均已完成。除注明"沿用优化前"的数据外，全部数字来自 O1 合入后的最终内核。ncu 对比表中的"未重测"项已逐项标注。
## 0 摘要
- 前向（causal、d=64、bf16）：N=32768 耗时 1.694 ms，81.2 TFLOP/s，为 cuBLAS bf16 4096³ 实测峰值（98.0 TFLOP/s）的 82.8%；N=65536 达 81.1 TFLOP/s（82.8%）。causal 下与 SDPA(flash) 持平或更快（32768：1.694 对 1.925 ms，快 12%；65536：6.778 对 6.869 ms，持平）；端到端口径 N=16384 快 20%、32768 快 9%，N≤4096 因约 60–80 µs 的 Python 侧开销反而更慢。非 causal 比 SDPA 慢约 16%（32768：3.705 对 3.206 ms），是目前唯一落后的场景。
- 显存：FA 随 N 线性（log-log 斜率 1.0），朴素基线平方（2.0）。N=65536 时基线前向、前向+反向均 OOM，FA 前向仅 8 MiB、前向+反向 86 MiB（0.086 GiB）。
- 最优 tile：(Bq,Bk,stages)=(64,64,2)，O1 合入后在 seq=4096 与 16384 仍最快；Bq=128 的两个配置在 16384 因倒序发射大幅改善（0.691/0.646 → 0.475/0.451 ms），与最优配置差距收敛到 1–6%。
- **瓶颈（ncu，O1 前后各一份）**：延迟受限、smem 限制占用率、causal block 负载不均衡三点在 O1 前后一致；O1 使寄存器 140→166（+26）但 Block Limit Registers 仍为 3，占用率结构未变，提升完全来自发射顺序。O1 后实测占用率从 13.5% 升到 16.24%（接近理论 16.67%），与倒序发射让 SM 间更均衡的机制吻合。新增 ncu 洞察：1.83 waves、1 满波 + 116/140 block 部分波，预估尾部效应最高 50%——这正是 O1 针对的问题，实测 16384 端到端 −20%、纯 kernel −24% 时间。
- 小 seq 的劣势来自并行度与固定开销，不是 kernel 本身慢：耗时约等于最长 block 的 tile 数乘以 1.2～1.7 µs。
## 1 环境与方法

| 项 | 内容 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti，sm\_120（ncu: CC 12.0），70 个 SM，16 GiB；Part 1 实测带宽 882 GB/s |
| 软件 | Python 3.13；PyTorch 与 CuTe DSL 版本：（待填） |
| 计时 | CUDA event，预热 3 次后取 20 次中位数；JIT 编译排除并单独报告；同一张表不混用清 L2 与不清 L2 口径 |
| 参照峰值 | cuBLAS bf16 4096³ GEMM：98.0–98.2 TFLOP/s（记为 100%） |
| FLOPs | 前向 4·N²·d（causal 取一半，即 2·N²·d）；前向+反向取前向的 3.5 倍（反向 2.5 倍） |
| 两种时间 | 端到端（含 Python 侧 from\_dlpack、校验、分配）与纯 kernel（预建参数后只发射） |
| 显存 | torch.cuda.max\_memory\_allocated 的增量峰值（相对调用前） |

## 2 实现概览与正确性

| 模块 | 要点 |
| --- | --- |
| 前向 | 每个 block 一个 Q tile，循环 K/V tile；warp 沿 M 排开（每 warp 16 行），一行 S 在同一 warp 内；online safe softmax，缩放与 log2(e) 合并为 qk\_scale，全程 exp2/log2；输出 O 与 base-2 的 LSE（m+log2(d)） |
| 访存 | cp.async 16B；K/V 多级流水；smem 41.98 KB（含驱动预留）；128B swizzle；越界行 pred 加补零；P 由 S 的 C 片段在寄存器内重排成 A 片段，不经 smem |
| causal | 左上对齐（j≤i）；跳过整块被 mask 的 K tile，只在对角块逐元素 mask；**block 倒序发射（O1，默认开启）** |
| 反向（两 kernel） | 预处理 D=rowsum(dO·O)；dK/dV kernel（一个 block 一个 K/V tile，循环 Q tile）加 dQ kernel（一个 block 一个 Q tile，循环 K/V tile）；无原子操作，结果确定 |
| 反向（单 kernel） | dK/dV 寄存器累加；dS 经 smem 转置后算 dQ，再对 fp32 的 dQ\_accum 做 atomicAdd；GEMM 数 5（两 kernel 版 7），dQ 求和顺序不确定 |
| 接入 | torch.autograd.Function，只保存 Q,K,V,O,L；反向可切换 torch 分块 / 两 kernel / 单 kernel |

测试均与 fp32 参考对照，全部通过：前向 30 个形状用例加 swizzle 逐位等价性；autograd 11 组（O4 实验配置下亦全部通过）；反向 9 组；单 kernel 12 组。覆盖单 tile、非整除、Nq>Nk、Nq\<Nk、Nq=1、长序列、毒化输入（越界读）、大幅值、解析解（Q=0）、one-hot 零梯度、部分 requires\_grad、非连续/未对齐输入、确定性。典型误差：前向 O 相对误差约 2–3e-3，LSE 误差约 1e-6；梯度相对误差约 2–3e-3（bf16 舍入）。
## 3 预测与实测对照

| 项 | 预测 | 实测 | 结论 |
| --- | --- | --- | --- |
| smem（64,64,2） | 40 KB | O1 前 41.98 KB；O1 后 41.98 KB（+1 KB 驱动预留） | 公式成立；O1 未改 smem |
| 基线前向显存，N=32768 bf16 | 约 2 个 N×N，即 4 GiB | causal 6.0 GiB（3 个）；非 causal 4.0 GiB（2 个） | 预测对非 causal 成立；causal 的非原地 masked\_fill 多一份 |
| 基线前向+反向，N=32768 | 3–4 个 N×N | causal 9.02 GiB（4.5 个）；非 causal 8.02 GiB（4 个） | 略高于预测 |
| OOM 点（16 GiB，bf16） | N=65536 前向与前向+反向均 OOM | 均 OOM | 成立 |
| 显存增长 | FA 线性，基线平方 | 斜率 1.0 / 2.0（非 causal 1.99） | 成立 |
| FA 前向显存，N=65536 | O 与 L 的量级 | 增量 8 MiB（O 8 MiB 加 L 0.25 MiB） | 成立 |
| 利用率 | 文档：30–50% 属正常 | 82.8%（32768 与 65536） | 高于预期 |
| 寄存器/线程 | 预期 O1 不增加 | **140 → 166（+26）** | 与预期不符；Block Limit Registers 仍为 3、无 spill，成本可忽略，但压缩了未来结构性优化的余量 |

O1 带来的寄存器增长未改变 Block Limit Registers（仍为 3），也未改 Block Limit Shared Mem（仍为 2），所以占用率结构不变，性能提升完全来自发射顺序。
## 4 tile 扫描（前向、causal、d=64、bf16；O1 合入后重跑）

| Bq | Bk | stages | warps | smem(KB) | 编译(s) | 状态 | N=4096 blocks | ms | TFLOP/s | N=16384 blocks | ms | TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 32 | 2 | 2 | 20 | 0.5 | OK | 128 (1.83x SM) | 0.086 | 25.0 | 512 (7.31x SM) | 0.487 | 70.6 |
| 64 | 32 | 2 | 4 | 24 | 0.4 | OK | 64 (0.91x SM) | 0.085 | 25.3 | 256 (3.66x SM) | 0.485 | 70.8 |
| **64** | **64** | **2** | **4** | **40** | **0.5** | **OK** | **64 (0.91x SM)** | **0.077** | **28.0** | **256 (3.66x SM)** | **0.448** | **76.7** |
| 128 | 64 | 2 | 8 | 48 | 0.5 | OK | 32 (0.46x SM) | 0.129 | 16.7 | 128 (1.83x SM) | 0.475 | 72.4 |
| 128 | 128 | 2 | 8 | 80 | 0.8 | OK | 32 (0.46x SM) | 0.121 | 17.8 | 128 (1.83x SM) | 0.451 | 76.2 |
| 64 | 64 | 1 | 4 | 24 | 0.5 | OK | 64 (0.91x SM) | 0.088 | 24.4 | 256 (3.66x SM) | 0.477 | 72.0 |
| 64 | 64 | 3 | 4 | 56 | 0.6 | OK | 64 (0.91x SM) | 0.079 | 27.3 | 256 (3.66x SM) | 0.525 | 65.4 |
| 128 | 64 | 3 | 8 | 64 | 0.5 | OK | 32 (0.46x SM) | 0.131 | 16.4 | 128 (1.83x SM) | 0.491 | 70.0 |

结论：
1. (64,64,2) 在两个 seq 都最快（28.0 / 76.7 TFLOP/s）。
2. Bq=128 在 4096 仍明显落后：32 个 block（0.46 倍 SM），一半 SM 空闲，倒序发射救不了并行度不足。但在 16384，O1 的倒序发射把 (128,64,2) 从优化前 0.691→0.475、(128,128,2) 从 0.646→0.451 ms，与 (64,64,2) 的 0.448 只差 1–6%：波次尾部的负载不均衡曾是 Bq=128 的主要劣势，倒序发射基本消除；仍未更快，说明 tile 本身的寄存器与占用率代价仍在。
3. stages 1→2 有收益（4096：0.088→0.077 ms；16384：0.477→0.448 ms），2→3 在 4096 持平、在 16384 反而最慢（0.525 ms；smem 56 KB 使每 SM 放不下 2 个 block，这是推测，可用 ncu 的 Block Limit Shared Mem 验证）。K/V 基本在 L2（命中率 98.4%，DRAM 吞吐 1%，O1 前数据），延迟靠同一 SM 上多个 block 隐藏。
4. 编译全部成功（0.4–0.8 s），没有 smem 超限或布局失败，也没有错误结果。
N 较小时耗时仍由最长 block 决定：causal 下最后一个 Q tile 要处理 N/64 个 K tile，折算每 tile 约 1.2–1.4 µs（N=2048：32 tile，0.044 ms；N=4096：64 tile，0.077 ms）。每 tile 约 1.05 MFLOP，换算单 SM 约 0.88 TFLOP/s，乘 70 个 SM 约 62 TFLOP/s——这是小 N 下由最长 block 决定的下界；N=16384 有 3.66 波且倒序发射改善了 SM 间均衡，实测 76.7 高于该下界。
## 5 Benchmark（causal、d=64、bf16）
- 作业：`batch=1`，seq ∈ {128, …, 65536}（按显存能跑到哪算哪），d ∈ {16, 32, 64, 128}，dtype ∈ {bf16, fp32}，causal=True。
- 目前只支持 d=64、bf16。
### 5.1 前向（ms）

| N | naive | SDPA(flash) | FA 端到端 | FA 纯 kernel | TFLOP/s | 利用率 |
| --- | --- | --- | --- | --- | --- | --- |
| 128* | 0.134 | 0.063 | 0.100 | 0.014 | 0.1 | 0.1% |
| 512* | 0.133 | 0.054 | 0.102 | 0.019 | 1.7 | 1.8% |
| 1024 | 0.140 | 0.056 | 0.104 | 0.027 | 4.9 | 5.0% |
| 2048 | 0.145 | 0.055 | 0.116 | 0.044 | 12.1 | 12.4% |
| 3072* | 0.291 | 0.063 | 0.129 | 0.060 | 20.2 | 20.6% |
| 4096 | 0.545 | 0.081 | 0.150 | 0.078 | 27.7 | 28.3% |
| 6144* | 1.448 | 0.115 | 0.227 | 0.156 | 30.9 | 31.5% |
| 8192 | 2.564 | 0.242 | 0.248 | 0.175 | 49.0 | 50.0% |
| 16384 | 9.856 | 0.650 | 0.522 | 0.449 | 76.6 | 78.1% |
| 32768 | 39.70 | 1.925 | 1.754 | 1.694 | 81.2 | 82.8% |
| 65536 | OOM | 6.869 | 6.826 | 6.778 | 81.1 | 82.8% |

\* 标注行沿用 O1 优化前的测量（最终基准未覆盖），其余为 O1 合入后的最终数据。
cuBLAS bf16 4096³ 的参照峰值是 98.0 TFLOP/s。端到端口径下与 SDPA 相比：N=16384 耗时少 20%，N=32768 少 9%，N=65536 持平（6.826 对 6.869 ms），N≤4096 因为约 60–80 µs 的 Python 侧开销反而更慢。
### 5.2 前向+反向（ms；反向为单 kernel 版）

| N | naive | FA | naive/FA | FA TFLOP/s |
| --- | --- | --- | --- | --- |
| 1024 | 0.578 | 0.605 | 0.96 | 0.8 |
| 2048 | 0.633 | 0.607 | 1.04 | 3.1 |
| 4096 | 1.095 | 0.688 | 1.59 | 10.9 |
| 8192 | 5.452 | 1.011 | 5.4 | 29.8 |
| 16384 | 21.387 | 1.880 | 11.4 | 64.0 |
| 32768 | 86.303 | 7.076 | 12.2 | 68.0 |
| 65536 | OOM | 26.977 | — | 71.3 |

交叉点约 N=2048，N=4096 起 FA 明显领先（1.59 倍），32768 处 12.2 倍。TFLOP/s 按前向 FLOPs 的 3.5 倍计。两 kernel 版反向未在最终基准中重测（优化前 32768 为 8.388 ms，慢于单 kernel 的 7.076 ms）。
### 5.3 causal 与非 causal（FA 纯 kernel；TFLOP/s 按各自有效 FLOPs）

| N | causal ms / TFLOP/s（利用率） | 非 causal ms / TFLOP/s（利用率） | 非 causal 的 SDPA ms |
| --- | --- | --- | --- |
| 2048 | 0.044 / 12.1（12.4%） | 0.043 / 24.9（25.4%） | 0.053 |
| 8192 | 0.175 / 49.0（50.0%） | 0.248 / 69.2（70.6%） | 0.241 |
| 32768 | 1.694 / 81.2（82.8%） | 3.705 / 74.2（75.7%） | 3.206 |

非 causal 的数据取自加入 O1 之前；O1 只作用于 causal，故非 causal 不受影响。非 causal 比 SDPA 耗时多约 16%（32768：3.705 对 3.206 ms），是目前唯一落后的场景。差距随 N 增大消失，说明 causal 的劣势主要是 block 负载不均衡（块数多时被摊薄，且 O1 的倒序发射进一步缓解）。
### 5.4 显存（增量峰值 GiB）

| N | naive 前向 | naive 前向+反向 | FA 前向 | FA 前向+反向 |
| --- | --- | --- | --- | --- |
| 1024 | 0.006 | 0.009 | 0.000 | 0.001 |
| 2048 | 0.023 | 0.036 | 0.000 | 0.003 |
| 4096 | 0.094 | 0.143 | 0.001 | 0.005 |
| 8192 | 0.375 | 0.567 | 0.001 | 0.011 |
| 16384 | 1.500 | 2.260 | 0.002 | 0.022 |
| 32768 | 6.000 | 9.020 | 0.004 | 0.043 |
| 65536 | OOM | OOM | 0.008 | 0.086 |

log-log 斜率：FA 前向 1.0，naive 前向 2.0。N=65536 时基线前向和前向+反向都 OOM。
## 6 问题回答
**Q1：在哪个 seq 之后 FA 开始领先？为什么小 seq 可能更慢？**

前向：纯 kernel 口径下测过的所有 N（128–65536）都比 SDPA 快，交叉点在 128 以下；端到端口径 N=8192 持平、16384 起 FA 更快（少 20%/9%/持平），N≤4096 因 Python 侧开销反而更慢。前向+反向：N≤2048 时两者接近（0.96 对 1.04，被 host 开销主导），N=4096 起明显领先（1.59 倍），8192 处 5.4 倍，32768 处 12.2 倍。小 N 的原因：（1）固定开销：kernel 只有 14–44 µs，而每次调用有 60–80 µs 的 Python 侧开销（from\_dlpack、校验、分配），反向还要加 D 的计算与多个 torch 小 kernel；（2）并行度：Bq=64 时 N=2048 只有 32 个 block，小于 70 个 SM；causal 下耗时由最后一个 block（32 个 K tile）决定，利用率 12.4%，而非 causal 为 25.4%；1.83 waves 与部分波次 116/140 block 的 ncu 洞察相互印证，但部分波次效应在多波次场景被摊薄；（3）基线在小 N 并不慢，S 矩阵放得进 L2，cuBLAS 又高效。吃满 SM 所需的最小 N：Bq=64 约 4480。

**Q2：FA 的显存随 seq 线性还是平方？**

线性。增量峰值显存对 N 做 log-log 拟合，FA 的斜率 1.0，基线 2.0。N=65536 时基线前向与前向+反向均 OOM，FA 前向 8 MiB、前向+反向 86 MiB；N=32768 时 FA 前向 0.004 GiB（O 与 L），基线 6.0 GiB（causal 下恰为 3 个 N×N 的 bf16 张量）。

**Q3：与 H100 的差距？**

先看本机数据：前向达到 cuBLAS 峰值的 82.8%（32768 与 65536），causal 下与 SDPA(flash) 持平或更快。ncu 显示 DRAM 吞吐只有 1%、L2 命中率 98.4%（O1 前的 SpeedOfLight，本次未重测；从 O1 只改发射顺序来看访存路径未变，预计不变），所以内存带宽（本卡 0.88 TB/s 对 H100 约 3.35 TB/s）不是本 kernel 的瓶颈，不能把带宽当作差距的主因。真正的差距在计算路径与片上资源：（1）本卡走 SM80 风格的 mma.sync，没有 WGMMA、TMA 与 warp 专用化，softmax 与 MMA 难以重叠（ncu：tensor 管线仅 39%，O1 前）；（2）smem 约 100 KB 对 H100 约 227 KB，限制 tile 与 stage（占用率被 smem 限制在 2 个 block/SM）；（3）SM 数 70 对 132，并行度更低，小 N 更难吃满。

## 7 ncu 对比分析（前向，N=16384，causal，grid 256×128 线程，(64,64,2) 配置）
O1 之前使用 `--set full` 采集了完整的 SpeedOfLight、SchedulerStats、Occupancy 与停顿分解；O1 之后只重跑了 `--section LaunchStats --section Occupancy`，未重测 Duration、SpeedOfLight 与停顿分解。下表把两份 ncu 并排呈现，未重测项标"未重测"。
### 7.1 Launch Statistics 对比

| 项 | O1 之前（`--set full`） | O1 之后（LaunchStats+Occupancy） | 变化 |
| --- | --- | --- | --- |
| Grid Size | 256 | 256 | 不变 |
| Block Size | 128 | 128 | 不变 |
| Registers Per Thread | 140 | **166** | **+26** |
| Dynamic Shared Memory Per Block | 41.98 KB | 41.98 KB | 不变 |
| Driver Shared Memory Per Block | 1.02 KB | 1.02 KB | 不变 |
| Shared Memory Configuration Size | 102.40 KB | 102.40 KB | 不变 |
| Waves Per SM | 1.83 | 1.83 | 不变 |
| # TPCs | 35 | 35 | 不变 |
| Cluster Size | 0 | 0 | 不变 |
| Cluster Scheduling Policy | PolicySpread | PolicySpread | 不变 |
| Duration | 683.6 µs（ncu 锁频） | 未重测 | — |
| 部分波次洞察 | 未给出 | **1 满波 + 116/140 block 部分波，预估尾部效应最高 50%** | 新增 OPT 洞察，与 1.83 waves 相互印证，O1 针对的同一问题 |

### 7.2 Occupancy 对比

| 项 | O1 之前 | O1 之后 | 变化 |
| --- | --- | --- | --- |
| Block Limit Registers | 3 | 3 | 不变（166 regs × 128 thr × 3 = 63.7 KB，仍在 64 KB 寄存器堆内） |
| Block Limit Shared Mem | 2 | 2 | 不变 |
| Block Limit Barriers / SM / Warps | 24 / 24 / 12 | 24 / 24 / 12 | 不变 |
| Theoretical Active Warps per SM | 8 | 8 | 不变 |
| Theoretical Occupancy | 16.67% | 16.67% | 不变 |
| Achieved Occupancy | 13.5% | **16.24%** | **+2.7 pp，接近理论上限** |
| Achieved Active Warps per SM | 1.62 活跃 warp / 调度器（≈ 6.5 warp/SM） | 7.80 warp/SM（= 1.95 warp/调度器） | 口径不同（前者来自 SchedulerStats，后者来自 Occupancy），未重测 SchedulerStats，暂不直接对比 |
| OPT Est. Speedup（占用率） | — | 50% / 83.33% | ncu 的通用建议，对本 kernel 不适用：占用率受限的根因是 smem tile 大小，不是 launch config |

**7.2 后注**：O1 后实测占用率从 13.5% 升到 16.24%（接近理论 16.67%），与倒序发射让 SM 间更均衡的机制吻合——ncu 的 Achieved Occupancy 是对所有 SM 在活跃周期内做平均（活跃 warps ÷ 活跃周期 ÷ 最大 warps），倒序发射让"早结束的 SM 与晚结束的 SM"差异缩小、活跃周期更同步，分母端的均值自然上升。理论占用率仍被 smem 卡在 16.67%（2 block/SM × 8 warp/block ÷ 64 warp/SM）。这也是 NVIDIA 官方 blog "Minimize the Tail Effect" 中描述的典型现象：负载不均衡越严重，achieved 与 theoretical occupancy 的差距越大。NVIDIA 在 CUDA Tile 实现的 FlashAttention 同样采用倒序发射来减少 causal 掩码下的尾部效应，FA4 的 LPT 调度器亦采用同思路（按最长处理时间优先发射重 block）。
### 7.3 SpeedOfLight 与停顿分解（未重测，沿用 O1 前）

| 项 | 数值 | 含义 |
| --- | --- | --- |
| Duration / SM 频率 | 683.6 µs / 2.28 GHz | ncu 锁频并清缓存，只看相对值 |
| SM 活跃周期占比 | 1.22M / 1.56M = 78% | 22% 的 SM 时间空闲（尾部与不均衡）——O1 针对的问题 |
| SM 间活跃周期离散 | −24.1% 到 +21.2% | causal 的 block 负载不均衡（ncu 估计 16.6%） |
| Compute(SM) / Memory 吞吐 | 30.9% / 38.3% | 延迟受限 |
| DRAM 吞吐 / L2 命中率 | 1.05% / 98.4% | 不是访存受限 |
| Tensor 管线 | 39.4%（活跃周期内） | 最忙的管线，但远未饱和 |
| 调度器 | 18.6% 周期发射；81.4% 周期无可发射 warp；每调度器 1.62 活跃 warp、0.21 可发射 warp | 占用率太低，延迟隐藏不足 |
| 寄存器 / 线程 | 140（O1 前） | 无溢出 |
| smem / block | 动态 41.98 KB 加 1.02 KB 驱动预留；每 SM 配置 102.4 KB |  |
| Block Limit（smem / 寄存器 / warp） | 2 / 3 / 12 | smem 是占用率的限制因素；理论占用 16.7%，实际 13.5% |
| 指令数 | 63.7M；其中 HMMA 约 8.4M（13%） | 约 87% 是非 MMA 指令（softmax、mask、转换、ldmatrix、shuffle） |
| 全局写 | 多余扇区 65536 个，占 0% | O 写回不是问题 |
| FP32 指令 | FFMA 4.5M 条融合、9.0M 条非融合 | 全部融合最多再省 1.3% |

每次发射间隔 8.70 周期的停顿分解（O1 前）：

| 停顿原因 | 周期 | 占比 | 解读 |
| --- | --- | --- | --- |
| math\_pipe\_throttle | 3.55 | 40.8% | MMA 阶段 tensor 管线被连续 HMMA 占满 |
| wait（固定延迟依赖） | 2.13 | 24.5% | softmax 里 rowmax、rowsum 是 16 个元素的串行依赖链，warp 太少无法隐藏 |
| selected（发射本身） | 1.00 | 11.5% |  |
| barrier | 0.56 | 6.4% | 每轮一次 barrier |
| short\_scoreboard | 0.50 | 5.7% | ex2（MUFU）与 ldmatrix 的返回 |
| mio\_throttle | 0.45 | 5.2% | shuffle、ldmatrix 排队 |
| long\_scoreboard | 0.19 | 2.2% | 全局访存，几乎无影响 |
| 其余合计 | 0.30 | 3.4% | not\_selected、dispatch、no\_instruction 等 |

### 7.4 结论
这是一个计算管线延迟受限的 kernel，三点结构性结论在 O1 前后一致：
1. **访存不是瓶颈**：DRAM 吞吐 1.05%、L2 命中率 98.4%、memory 吞吐 38.3%、long\_scoreboard 仅占停顿 2.2%。
2. **占用率被 smem 卡在 16.67%**（2 block/SM），与 O1 无关；O1 后实测占用率升至 16.24%（接近理论），说明 SM 间更均衡。
3. **波次尾部是主要损失**：1.83 waves + 116/140 block 部分波，预估尾部效应最高 50%。O1 的倒序发射让长 block 先发、短 block 后发，把这部分损失转化为实际收益（16384 −24% 时间、+24% TFLOP/s；32768 +13%）。这一做法与 NVIDIA CUDA Tile FlashAttention 的 ProgramId remapping和 FA4 的 LPT 调度器一致。
O1 未改变的停顿结构——math\_pipe\_throttle（40.8%）与 wait（24.5%）合计 65%——指向下一档优化方向：要么降低占用率（O4 的 smem 别名，32768 再 +5.3%），要么做 warp 内软件流水（FA3 风格，让 MMA 与 softmax 在同一 warp 内重叠）。
## 8 优化与消融
依据第 7 节的停顿分解，实现了三个可单独关闭的优化与一个占用率实验（代码在 flash\_attn\_naive.py 与 ablate\_fwd.py）：

| 编号 | 优化 | 针对的证据 | 精确性 | 预测收益 | 实测结论 |
| --- | --- | --- | --- | --- | --- |
| O1 | causal 下倒序发射 block（Q tile 号大的先发） | SM 间活跃周期 −24% 到 +21%，SM 活跃占比 78%；1.83 waves + 116/140 部分波 | 逐位不变 | causal 3–10% | **已合入默认内核**：16384 +24%、32768 +13%，超出预测；寄存器 +26（140→166）但未改 Block Limit，占用率结构不变 |
| O2 | rowmax、rowsum 用 4 条独立依赖链再合并 | wait 占 24.5% | 求和顺序改变，非逐位一致，误差在 bf16 舍入内 | 1–4% | 负收益（−3%～−4%），默认关闭 |
| O3 | 惰性缩放：m\_new 等于 m\_old 时跳过 acc\_o 的 alpha 乘法 | O 缩放每 tile 每线程 32 次 FMUL，约占指令的 6.6% | 逐位不变（alpha 恰为 1） | 2–4% | 负收益（−3%），默认关闭 |
| O4（实验） | Q 的 smem 别名到 K 的 stage 1，使每 block 的 smem 降到约 33 KB | Block Limit Shared Mem 为 2 | 逐位不变 | 不确定；(64,32,2) 的 smem 26 KB 也能放 3 个 block 却没有更快，故预期有限 | 32768 再 +5.3%（85.3 对 82.0），16384 −2.4%；autograd 11/11 通过；保留为实验开关 |

被否决的优化：O 经 smem 做 16B 合并写回（多余扇区仅 0%，ncu 预计加速 0.32%）。
消融实测（`python ablate_fwd.py --seqs 4096 16384 32768` 的输出）：前向消融，d=64，bf16，causal=True；TFLOP/s = 2·N²·d / 时间；括号为相对 baseline（优化前内核）的倍数。

| 配置 | 状态 | 编译(s) | 与 baseline 逐位一致 | N=4096 ms | TFLOP/s | N=16384 ms | TFLOP/s | N=32768 ms | TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|
| baseline（O1 前） | OK | 0.6 | - | 0.076 | 28.4（1.000x） | 0.555 | 61.9（1.000x） | 1.884 | 72.9（1.000x） |
| +O1 reverse | OK | 0.6 | 是 | 0.077 | 28.0（0.984x） | 0.448 | 76.7（1.239x） | 1.694 | 81.1（1.112x） |
| +O2 ilp | OK | 0.5 | 否 | 0.077 | 28.0（0.984x） | 0.574 | 59.8（0.966x） | 1.971 | 69.7（0.956x） |
| +O3 lazy | OK | 0.5 | 是 | 0.077 | 28.0（0.984x） | 0.566 | 60.7（0.980x） | 1.943 | 70.7（0.970x） |
| O1+O2+O3 | OK | 0.5 | 否 | 0.077 | 28.0（0.984x） | 0.462 | 74.4（1.202x） | 1.677 | 82.0（1.124x） |
| O1+O2+O3+O4 | OK | 0.5 | 否 | 0.077 | 28.0（0.984x） | 0.474 | 72.6（1.172x） | 1.611 | 85.3（1.170x） |

优化前后对照（前向 causal，纯 kernel；最终数据取自 5.1，优化前数据取自 O1 合入前的旧基准）：

| N | 优化前 ms / TFLOP/s | 最终 ms / TFLOP/s | 提升 |
|---|---|---|---|
| 4096 | 0.076 / 28.4 | 0.078 / 27.7 | 持平（在噪声内，约 ±3%） |
| 8192 | 0.194 / 44.4 | 0.175 / 49.0 | +10% |
| 16384 | 0.555 / 61.9 | 0.449 / 76.6 | +24% |
| 32768 | 1.910 / 72.0 | 1.694 / 81.2 | +13% |
| 65536 | 7.022 / 78.3 | 6.778 / 81.1 | +4% |

提升全部来自 O1。O1 在 N≤4096 无收益（尾部本来就短，且 4096 处 0.078 对 0.076 的差异在噪声内），N=8192 起受益，N 越大尾部占比越高收益越大。O2、O3 单独开启为负收益，但 O1+O2+O3 组合在 32768 略好于 O1 单独（82.0 对 81.1），差异约 1%，不足以让非逐位一致的 O2 进入默认配置；O4 在 32768 带来进一步收益（85.3），但在 16384 略慢，且目前只有 autograd 测试通过（11/11），故保持实验开关。

更大的结构性改动（未实现）：在 warp 内把下一块的 S=QK^T 与当前块的 softmax 交错发射（软件流水），让 MMA 与 softmax 在同一个 warp 内重叠；这是 FA3 的 intra-warp pipelining 思路，需要两份 S 累加器（寄存器约增加 32 个）。**O1 已把寄存器从 140 推到 166，再加 32 个大概率触发 spill，需先评估。**
## 9 局限
- 只支持 d=64、bf16；d=16/32/128 与 fp32 未实现，表中标 N/A。d=128 需改 swizzle 与 g2s 排布，fp32 需换 TF32 的 mma atom。
- 非 causal 比 SDPA(flash) 慢约 16%（32768），是目前唯一落后的场景。
- O2、O3 改变求和顺序、非逐位一致，默认关闭；O1+O2+O3+O4 组合的逐位一致性无法保证。
- 单 kernel 反向的 dQ 经 atomicAdd，求和顺序不确定（dK、dV 确定）。
- dQ kernel 的倒序发射补丁未验证，尚未合入。
- **O1 使寄存器从 140 升到 166（+26），Block Limit Registers 仍为 3、未带来 spill；但进一步的结构性优化（FA3 风格的 warp 内软件流水需再 +32 regs）需先评估 spill 风险。**
- **ncu 对比表中 SpeedOfLight、SchedulerStats 与停顿分解未在 O1 后重测，部分指标沿用 O1 前数据并已标注。**
- 反向 kernel 的寄存器与 spill 未做 ncu 检查；L2、D 的按元素 gmem 读与 dS 的 2 字节 smem 写是已知的优化点。
- 小 N 的前向+反向端到端数字被 host 开销主导，不能当作 kernel 性能；GPU 时间可用 profiler 单独统计。
## 10 复现命令
```
python test_flash_attn.py; python test_flash_attn_autograd.py
python test_flash_attn_autograd.py --enable-o4   # O4 配置下 autograd，按仓库 CLI 为准
python test_flash_attn_bwd.py; python test_flash_attn_bwd_fused.py
python sweep_tiles.py --seqs 4096 16384                          # O1 后重跑，第 4 节
python bench_flash.py --seqs 1024 2048 4096 8192 16384 32768 65536
python bench_flash.py --seqs 1024 2048 4096 8192 16384 32768 65536 --fused
python bench_flash.py --seqs 2048 8192 32768 --no-causal
python ablate_fwd.py --seqs 4096 16384 32768
# O1 前 --set full（已采）
sudo env CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu --set full -k regex:FLASH_ATTN -c 1 -o fa_fwd_full ...
# O1 后 LaunchStats+Occupancy（已采）
sudo env CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
  --section LaunchStats --section Occupancy \
  -k regex:FLASH_ATTN -c 1 \
  /data2/c220/hz/cs336/.venv/bin/python -c "import torch; from flash_attn_autograd import flash_attn_func as f; \
  q,k,v=[torch.randn(1,16384,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]; \
  [f(q,k,v,True) for _ in range(3)]; torch.cuda.synchronize()"
# 建议补测：O1 后 --set full
sudo env CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu --set full -k regex:FLASH_ATTN -c 1 -o fa_fwd_o1_full ...
```
## 11 未重测项清单（建议补测）

| 项 | 现状 | 建议命令 |
| --- | --- | --- |
| O1 后的 Duration / SpeedOfLight / SchedulerStats / 停顿分解 | 沿用 O1 前（`--set full`） | `sudo env CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu --set full -k regex:FLASH_ATTN -c 1 <python cmd>` |
| 反向 kernel 的 ncu | 未做 | `--set full -k regex:FLASH_ATTN_BWD -c 1 ...` |
| N=128 / 512 / 3072 / 6144 的最终基准 | 沿用 O1 前 | `python bench_flash.py --seqs 128 512 3072 6144` |
| dQ kernel 的倒序发射补丁 | 未验证 | 验证后合入 |

---
几点说明，便于你定稿：
**关键事实核对**（基于你提供的两次 ncu 输出）：
- O1 前 ncu：`Registers Per Thread 140`、`Block Limit Registers 3`、`Achieved Occupancy 13.5%`、`Duration 683.6 µs`。
- O1 后 ncu：`Registers Per Thread 166`、`Block Limit Registers 3`、`Theoretical Occupancy 16.67%`、`Achieved Occupancy 16.24%`、`Achieved Active Warps Per SM 7.80`、`Waves Per SM 1.83`、OPT 提示 `1 full waves and a partial wave of 116 thread blocks, up to 50.0% of total runtime`。
- 166 regs × 128 threads × 3 blocks = 63,744 regs，仍落在 64K 寄存器堆内，所以 Block Limit Registers 维持 3，未成为新的占用率瓶颈。
- Duration、SpeedOfLight、SchedulerStats、停顿分解在本次采集（仅 LaunchStats+Occupancy）里没有，第 7.3 节按"未重测"处理。
**机制解读**（已写入 7.2 后注）：倒序发射之所以把 Achieved Occupancy 从 13.5% 推到 16.24%，是因为 ncu 的 Achieved Occupancy 在活跃周期上对所有 SM 做平均——SM 间负载越不均衡，部分 SM 早停导致的"低活跃周期数"越多，均值越被拉低；倒序发射让重 block 先发，SM 间完成时间更同步，活跃周期数整体上升，均值自然逼近理论上限 16.67%。这是 NVIDIA 官方 blog 与 ncu 文档都明确描述的机制，与 FA4 LPT 调度器的设计动机一致。
**未重测项的处置**：第 7.3 节整节沿用 O1 前数据，已在该节和第 9 节局限中明确标注。若补跑 `--set full`，需要更新 7.3 全表 + 7.4 第 1、3 条结论，以及 Q3 中的 DRAM/L2/tensor 管线数字。
