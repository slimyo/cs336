# CS336 A2 Part 1 实验后总结：可回查的 Takehome 知识（面试向）

> 用法：先读第 0 节"一页速记"，面试前再读第 3 节"概念专题"和第 5 节"问答"。
> 凡是带数字的例子，都来自 small / medium、fp32、RTX 5070 Ti（70 个 SM，16 GiB）、batch 4 的实测，用来说明方法，不是通用结论。

---

## 0. 一页速记

1. **GPU 是异步的**：CPU 只发射 kernel。计时必须同步，用 CUDA event 在 GPU 侧计时，预热后再测，报中位数、均值和 std。
2. **测什么就要建什么图**：测推理用 `no_grad`；测训练的 forward 要带计算图。带图的 forward 更慢，更占显存。
3. **backward 要这样测**：计算图只能用一次，所以每次迭代重新 forward。**在同一步里用 event 切出 forward / backward / optimizer**，不要用不同运行相减。
4. **backward ≈ 2 × forward**：每个前向 GEMM 对应两个反向 GEMM（对输入求导、对权重求导）。
5. **optimizer 与 batch、ctx 无关，只与参数量有关**，而且是访存受限。AdamW 下限约 28 B/参数。
6. **显存 = 静态 + 动态**。静态（权重 + 梯度 + Adam）fp32 下是 16 B/参数，与 batch、ctx 无关；动态（激活）随 B·T 线性增长，attention 部分随 T² 增长。
7. **峰值在 forward 末尾**（激活最多、梯度还没出现），不一定在 backward。
8. **混合精度**：参数保持 fp32，autocast 只包 forward，loss 在外面算。加速受 Amdahl 限制，显存节省很有限。
9. **attention 的显存瓶颈是 S×S 矩阵**。`torch.compile` 不能消除它，FlashAttention 靠分块与 online softmax 才能。
10. **一个工具回答一个问题**：benchmark 回答"多快"，nsys 回答"花在哪"，ncu 回答"为什么"，快照回答"显存被什么占着"。
11. **总量吻合不等于模型对**：要把总量拆成分量对账，并让不同测法互相印证。

---

## 1. 测量基础（所有表共用）

### 1.1 为什么不能 `time.time()`

| 问题 | 原因 |
|---|---|
| 量到的是发射时间 | Python 调用把 kernel 放进队列就返回 |
| 首次迭代不是稳态 | cuBLAS 初始化、kernel 懒加载、allocator 首次分配、Adam 状态首次分配、`torch.compile` 编译、GPU 频率爬升 |
| 其他进程 / 温度 / 功耗 | 抖动与系统误差 |

### 1.2 标准做法

1. 被测单元只含要测的内容，数据生成、H2D、`.item()` 放在计时区外。
2. 固定条件：独占 GPU、dtype、形状、种子、TF32 开关、`train/eval`、是否 `no_grad`。
3. 预热，让所有首次开销发生在计时之前。
4. 用 `torch.cuda.Event(enable_timing=True)`，每步结束同步后再读。
5. 重复并统计：中位数为主，同时报均值±样本 std、最小、最大、抖动。
6. 与预测和其他测法对账。

### 1.3 两种计时方式

| 方式 | 回答 | 特点 |
|---|---|---|
| 逐次 event（每步同步） | 一步本身的 GPU 耗时（延迟） | 不含步间 CPU 空隙，稳定，适合与预测比 |
| 连续墙钟（只在头尾同步） | 真实吞吐 | 含 CPU 发射与流水重叠 |

### 1.4 测量方法本身的检验（比数据本身更重要）

| 检验 | 例子（本项目） |
|---|---|
| 阶段之和 = 整步 | 38.49 + 76.09 + 13.13 = 127.71 ms vs 另一脚本测得的 127.85 ms |
| 两次独立运行一致 | 同配置 fwd 38.49 / 38.56，bwd 76.09 / 76.11 |
| 不同工具互相印证 | benchmark 的阶段时间 ≈ nsys 的 kernel 耗时之和（forward 0.99、backward 0.96） |
| 理论账目闭合 | backward 结束时显存 = 4P + 4P + 8P 精确吻合 |
| 与旧测法对账 | 相减法比同步内切段高估 backward 1.4%，方向与理论一致 |

**教训**：3.7 里 vanilla 基线比 3.2 同配置慢 36%～42%，就是靠这种对账才发现的。std 小不等于准，系统误差不进 std。

---

## 2. 各表：为什么测、解决什么问题、怎么测、注意什么

### 3.1 / 3.2 端到端计时（forward / backward / optimizer）

| 项 | 内容 |
|---|---|
| **为什么测** | 建立后面所有优化（混合精度、compile、融合）都要对比的**基线**；回答"哪个阶段最慢、随 ctx 怎么变" |
| **解决的问题** | 知道瓶颈在哪个阶段，以及它是计算受限、访存受限还是发射受限 |
| **测量思路** | 同一个训练步里放 4 个 event：`zero_grad` → ev0 → forward+loss → ev1 → backward → ev2 → optimizer → ev3 → 同步一次。三段之和就是整步 |
| **关键点** | ① forward 要带图、含 loss（与 backward 配套）；② 每次迭代重新 forward（图只能用一次）；③ 预热要覆盖 Adam 首次 step；④ 记录 TF32 开关；⑤ OOM 要记录而不是中断 |
| **注意** | 不能用 `fwdbwd − fwd`、`full − fwdbwd` 相减（见 3.1 节） |
| **本项目结论** | ① backward 最慢，bwd÷fwd 在 ctx ≥ 256 时约 1.9～2.0；② optimizer 与 ctx 无关（small 13 ms，medium 47 ms），占比随 ctx 增大迅速下降；③ 小 ctx 是发射受限（约 20 µs/kernel），ctx 增大后先线性、再因 T² 项超线性 |
| **面试问法** | "为什么 backward 约是 forward 的 2 倍？""为什么小 ctx 时 GPU 利用率低？""optimizer 为什么占这么多？" |

### 3.3 warmup 影响

| 项 | 内容 |
|---|---|
| **为什么测** | 说明为什么预热是必须的，并且知道预热要覆盖哪些东西 |
| **解决的问题** | 避免把一次性开销算进稳态耗时，降低方差 |
| **测量思路** | 在全新进程里不预热，逐步记录最初 N 步的每个阶段，与稳态比 |
| **候选因素** | CUDA context 初始化；cuBLAS 句柄、workspace、按形状选算法；kernel 懒加载（backward 的新 kernel 第一次出现）；allocator 首次分配与扩容；**Adam 状态首次分配（只影响 optimizer）**；GPU 频率爬升；`torch.compile` 编译；L2 冷启动（影响很小） |
| **不适用的** | cudnn autotune（本模型没有卷积）；eager 模式下没有 JIT |
| **注意** | 预热步数要覆盖所有"首次出现"的东西，最典型的遗漏是 Adam 首次 step；compile 的预热要包含编译 |

### 3.4 nsys：时间花在哪

| 项 | 内容 |
|---|---|
| **为什么测** | benchmark 只给总时间；nsys 给 kernel 种类、次数、占比、空隙，回答**时间花在哪** |
| **解决的问题** | 区分计算受限与发射受限；找出占比最大的 kernel 类别；为 ncu 挑选目标 |
| **测量思路** | NVTX 给 forward / backward / optimizer 与 attention 内部打标；`profiler.start/stop` 只抓测量步；用 `nvtx_kern_sum` 等报告汇总 |
| **关键点** | ① NVTX 区间记录的是 **CPU 侧**，kernel 靠"发射线程 + 发射时刻"投影回区间；② 每阶段末尾同步只用于归因，会破坏重叠，所以**只看构成，不看绝对时间**；③ 先算 GPU 利用率再决定要不要用 ncu |
| **最大的坑** | **backward 被漏归因**：反向在 autograd 引擎的工作线程里发射，主线程里的 `backward` 区间收不到这些 kernel。解决：差分法（总表 − forward − optimizer）、时间窗归因、`emit_nvtx` |
| **本项目结论** | GEMM 占整步 53.5%，elementwise 43.2%；softmax 是 5 个 kernel 拼成的；optimizer 的 kernel 数（1776 = 111 × 16）说明逐参数循环 |
| **面试问法** | "怎么判断是发射受限？""NVTX 为什么会漏掉 backward？""怎么得到各类 kernel 的占比？" |

### 3.5 混合精度

| 项 | 内容 |
|---|---|
| **为什么测** | 量化 bf16 能快多少、省多少，并理解为什么达不到理论值 |
| **解决的问题** | 判断混合精度的实际收益，识别它的开销来源 |
| **测量思路** | 只改 dtype，其余全部固定；每档用同样的阶段计时；加速比 = 同一阶段 fp32 时间 ÷ bf16 时间；用 hook 实测各类张量的 dtype |
| **关键点** | 参数保持 fp32；autocast 只包 forward；loss 在外面、用 fp32 logits；backward 在外面；bf16 不需要 GradScaler；fp32 基线要显式关闭 TF32 |
| **本项目结论** | fwd+bwd 加速约 1.8×；峰值显存只降 15%；dtype 表见 4.4 |
| **要理解** | **Amdahl**：GEMM 占 fwd+bwd 的 57.4%，即使它变成零耗时，整体上限也只有 2.35×；optimizer 不受影响，整步加速更低 |
| **面试问法** | "autocast 里哪些算子用 bf16、哪些用 fp32、为什么？""为什么混合精度没有省一半显存？""为什么 loss 要用 fp32？" |

### 3.6 attention 微基准

| 项 | 内容 |
|---|---|
| **为什么测** | 把 attention 的 $O(S^2)$ 显存与访存开销**单独隔离出来**，在不受模型其余部分干扰的条件下量化它 |
| **解决的问题** | ① 找到 OOM 的临界点并验证显存公式；② 理解为什么需要 FlashAttention；③ 看 compile 对纯 attention 的作用 |
| **测量思路** | 固定 batch，扫 seq 与 d_model；分别记 forward / backward 时间、backward 前显存（保存的激活）、forward 瞬态峰值；OOM 用 try/except 记录 |
| **关键点** | **区分"保存给反向的"与"瞬态峰值"**：保存的是 1 个 S² 张量（softmax 输出，PV 矩阵乘与 softmax 反向共用同一份），forward 里 scores 与输出同时存在的瞬间是 2 个 |
| **本项目结论** | OOM 首现于 seq = 16384（2 × 8 GiB = 16 GiB），与 d 无关；pre-G 实测 ≈ 1 × B·S²·4B；compile 加速 1.28～1.51×，**但显存纹丝不动**；d 越大 compile 收益越小（GEMM 比重上升） |
| **面试问法** | "attention 为什么是 $O(S^2)$ 显存？""FlashAttention 为什么能省显存而 compile 不能？" |

### 3.7 整模型 compile

| 项 | 内容 |
|---|---|
| **为什么测** | 看 compile 在真实模型上的收益，而不只是微基准 |
| **解决的问题** | 判断 compile 值不值得用；理解它的收益来自哪里、上界在哪里 |
| **测量思路** | **同一进程、交替（ABAB）测 eager 与 compiled**，预热含编译，用同一套阶段计时 |
| **关键点** | ① 基线必须与已验证的基线一致；② compile 首次调用含编译；③ 形状变化会重编译 |
| **本项目的教训** | 3.7 的 vanilla 基线比 3.2 慢 36%～42%，若用它算会得到 1.38×，而对照 3.2 的基线只有 2%～5%。**先对账基线，再报加速比** |

### 3.8 显存 profiling

| 项 | 内容 |
|---|---|
| **为什么测** | 3.5 只给一个峰值标量；3.8 要回答**峰值在哪一刻、被什么占着、与理论差多少** |
| **解决的问题** | 验证显存理论、校准公式常数、找最大分配、决定省显存的手段（重算、分片、低精度） |
| **测量思路** | ① 先写理论（4P / 16P / 激活估算）；② 刚初始化、还没有优化器时测 no_grad 前向峰值；③ 带图前向后读保存的激活；④ 预热 3 步让 Adam 状态存在，只对一步开快照，并**按阶段 reset 峰值**；⑤ 解析快照；⑥ 用多个 ctx 拟合 `act = k + a·N + b·B·h·T²` |
| **关键点** | 先测无优化器的前向，否则 Adam 状态混进来；只录一步；用 GiB 并写明；OOM 时导出快照 |
| **本项目结论** | 静态精确等于 4P + 4P + 8P；峰值在 forward 末尾（loss 的临时张量）；T² 项 c = 2（手写 softmax 保存 2 个）；线性项约 29.5 份 N×d；bf16 缓存 2P 的权重副本 |
| **面试问法** | "训练一个 X 参数的模型需要多少显存？""峰值出现在哪个阶段？""怎么省激活显存？" |

---

## 3. 概念专题

### 3.1 forward：带 grad 与否

| | no_grad | 带 grad（训练） |
|---|---|---|
| 计算图 | 不建 | 每个算子记录一个节点，保存反向所需的张量 |
| 中间张量 | 用完即释放 | 保存到 backward 用完才释放 |
| CPU 开销 | 小 | 多了建节点（本项目：训练前向慢约 2%） |
| 显存 | 低（medium / 512：1.88 GiB） | 高（同配置：10.56 GiB，其中保存的激活 8.8 GiB） |
| 用途 | 推理、与 FLOPs 预测对比 | 与 backward 配套的训练前向 |

**经验**：`no_grad` 的 forward 不能与 backward 配对。拿 `fwdbwd − fwd` 当 backward，会把"建图 + 保存张量 + loss"的成本错记到 backward。

### 3.2 backward 的测量

**事实**：

- 前向建图，`backward()` 沿图反向走并**释放保存的张量**；默认图只能用一次，再 backward 会报错。
- `retain_graph=True` 可以绕过，但保存的张量一直占着显存，行为与真实训练不同，**不要用来测速**。

**所以**：每次计时迭代都必须重新 forward，再计时 backward。

**相减法的问题**（`backward ≈ fwdbwd − fwd`，`optimizer ≈ full − fwdbwd`）：

| 问题 | 机理 |
|---|---|
| forward 口径不同 | `fwd` 是 no_grad，`fwdbwd` 里是带图 + loss，backward 被高估 |
| 大数相减放大误差 | 127.85 − 114.90 = 12.95，两个 ±1 ms 的波动就是 ±8% |
| 得不到各阶段自己的 std | 三档是独立运行，没有逐步配对 |
| 运行状态不同 | fwdbwd 档没有 Adam 状态；GPU 频率与温度不同 |

**正确做法**：同一步里 4 个 event，只在最后同步一次：

```python
ev[0].record(); loss = loss_fn()
ev[1].record(); loss.backward()
ev[2].record(); opt.step()
ev[3].record(); torch.cuda.synchronize()      # forward = 0→1, backward = 1→2, optimizer = 2→3
```

**backward ≈ 2 × forward 的原因**：线性层 $Y = XW$，反向要算 $dX = dY\,W^\top$ 和 $dW = X^\top dY$，各与前向同样大小，所以 GEMM 的 FLOPs 是 2 倍。**但时间不一定正好 2 倍**：本项目 GEMM 的时间比是 1.61、elementwise 是 2.89，两个相反的偏差抵消成约 2。

**profile 的坑**：反向在 autograd 引擎的工作线程里发射 kernel，NVTX 区间是主线程的，所以 `backward` 区间会收不到 kernel。

### 3.3 optimizer

- AdamW 每步对每个参数：读 p、g、m、v，写 p、m、v，约 **28 B/参数**。
- **与 batch、ctx 无关，只与参数量有关**；因此在整步里的占比随 ctx 下降（small：28% → 4%）。
- **访存受限**：medium 上耗时与参数量成正比（3.61 对 3.29），不与 kernel 个数成正比（1.97）。实测是带宽下限的 3.2～3.5 倍，因为逐参数循环把一次更新拆成约 16 个 elementwise kernel，每个读写一遍。
- small 上还有**发射开销**成分（kernel 很小，每个约 6 µs）。
- 优化：`foreach=True` / `fused=True` 的 AdamW，或 `torch.compile` 融合。
- **Adam 状态在第一次 `step()` 时才分配**（8 B/参数），所以预热要包含 step，显存测量要在预热之后。

### 3.4 激活与权重的区别

| | 权重（静态） | 激活（动态） |
|---|---|---|
| 大小 | 与参数量成正比 | 与 batch、ctx 成正比，attention 部分与 T² 成正比 |
| 是否随 batch、ctx 变 | 否 | 是 |
| 何时占用 | 始终 | forward 时产生，backward 时消费并释放 |
| 包含 | 权重 4P + 梯度 4P + Adam 8P = **16P**（fp32） | 保存给反向用的张量 |
| 省显存的手段 | 分片（ZeRO / FSDP）、低精度状态、offload | 重算（checkpoint）、混合精度、FlashAttention、减小 batch |

**激活的构成**（本项目拟合）：

```
act ≈ 4·L·[ (线性份数·d)·N + c·B·h·T² ] + 16·N·V        字节，fp32 口径
N = B·T；线性份数 ≈ 29.5（FFN 中间量、残差、投影输入等）；c = 2（手写 softmax）
```

- **线性项与 T² 项相等的 ctx**：medium 约 990 个 token。ctx 小时激活主要是线性项，ctx 大了 T² 项主导，这就是长上下文 OOM 的来源。

**哪些张量会被保存给反向**（理解激活的钥匙）：

| 算子 | 反向需要 | 所以保存 |
|---|---|---|
| 线性层 | 输入和权重 | 输入 X |
| 逐元素乘 | 另一个操作数 | 两个输入之一 |
| exp | 它自己的输出 | 输出 |
| softmax（融合版） | 它自己的输出 | 输出（1 份） |
| 手写 softmax（max-sub-exp-sum-div） | `exp` 的输出、除法要用的 sum；而 PV 矩阵乘还要保存 P | **2 个** S² 张量 |
| 加法、乘以标量 | 不需要输入 | 不保存 |

### 3.5 显存峰值出现在哪

```
              forward 过程中          loss          backward           optimizer
显存      ──────上升────────▶ ┐峰值  ┌──▶ 激活逐块释放、梯度逐块增加 ──▶  临时张量
                              └──────┘
```

- forward 末尾：激活最多、**梯度还没有**，且 loss 里有 logits 的临时张量 → **常常是峰值**。本项目：medium / 512 的峰值比 forward 结束时高 0.15 GiB，恰好 ≈ 2 个 N×V 的 fp32 张量。
- backward：每消费一块激活就释放，同时梯度增加；只要激活比梯度（4P）大，这里就低于 forward 末尾。
- optimizer：静态 16P 加上逐参数的临时张量。**激活很小（ctx 短）时，峰值在这里**。

**4 个阶段的读法**：对每个阶段单独 `reset_peak_memory_stats()` 再读，才能知道峰值在哪个阶段。

**allocated 与 reserved**：`memory_allocated` 是张量实际占用；`memory_reserved` 是 allocator 持有的，二者之差是碎片与缓存。`nvidia-smi` 看到的是 reserved 加 CUDA 上下文，比 allocated 大，不要拿它填表。

### 3.6 混合精度（autocast）

**设置指南**：

| 设置 | 原因 |
|---|---|
| 参数保持 fp32（fp32 主权重） | bf16 只有约 8 位有效数字，`lr × grad` 这种小更新会被舍入掉 |
| autocast 只包 forward | autocast 对"算子调用"生效 |
| loss 在 autocast 外、用 fp32 logits | 对数、指数、求和在低精度里不够准 |
| backward 在 autocast 外调用 | 反向算子沿用对应前向算子的 dtype |
| bf16 不需要 GradScaler | bf16 指数范围与 fp32 相同，不下溢；fp16 才需要 |
| fp32 对照组显式关闭 TF32 | 否则对照组可能悄悄走 TF32 |

**算子分类**（以官方文档为准）：

| 类别 | 例子 | 行为 |
|---|---|---|
| 降到 bf16 | linear、matmul、bmm（einsum 内部分解到 bmm） | 输入转 bf16，输出 bf16 |
| 升到 fp32 | exp、log、softmax、sum、layer_norm、cross_entropy | 输入转 fp32 |
| 按输入 dtype 运行 | add、mul、cat、where、relu | 不转换；二元运算遵循类型提升（bf16 + fp32 → fp32） |

**本项目的 dtype 实测**（hook 得到）：

| 对象 | dtype |
|---|---|
| 参数 | fp32 |
| Linear 输出 | bf16 |
| RMSNorm 输出 | **fp32** |
| 残差流 | **fp32** |
| logits | bf16 |
| loss | fp32 |
| 梯度 | fp32 |

**后果**：残差流是 fp32，所以每个 Linear 的输入要先转成 bf16（额外的转换 kernel）；手写 softmax 的 `exp` 被升到 fp32，其输出以 fp32 保存。

**为什么加速只有约 1.8×**（Amdahl）：

```
加速比 = 1 / ( GEMM占比/GEMM加速 + 非GEMM占比/非GEMM加速 )
GEMM 占 fwd+bwd 的 57.4%，即使 GEMM 变成零：上限 1/0.426 = 2.35×
实测 1.78× ⇒ 非 GEMM 部分几乎没有同步提速（残差流 fp32、类型转换抵消了读写减半）
整步更低，因为 optimizer（13 ms）不受影响
```

**为什么显存只降 15%**：

- 静态 16P 不变；
- autocast **缓存一份 bf16 权重副本，约 2P**（medium：0.79 GiB）。证据：no_grad 前向峰值多出 0.73 GiB；
- 激活只降到约 0.63～0.67 倍（不是 0.5），因为残差流等仍是 fp32；
- 所以**短 ctx 时混合精度反而多占显存**（ctx 128：1.02×），ctx 512 才净省 15%。

### 3.7 torch.compile

**它做什么**：

| 阶段 | 内容 |
|---|---|
| Dynamo | 捕获 Python 代码为 FX 图，插入 guard 保证复用条件 |
| Inductor | 把相邻的逐元素与归约算子**融合**成少量 Triton kernel；对 GEMM 默认调用 cuBLAS |
| 效果 | kernel 个数下降（发射开销下降）；融合后中间张量不再写回显存（HBM 读写遍数下降） |

**它不做什么**：

- 不改变算法：S×S 的 attention 矩阵仍然要物化（本项目数据：pre-G 完全不变）。这是它与 FlashAttention 的本质区别。
- 默认不加速 GEMM（`max-autotune` 才会尝试其他实现）。
- 不能消除编译时间，形状变化会触发重编译。

**收益上界**：只能压缩非 GEMM 部分，而且只能减少读写遍数。GEMM 比重越大，整体收益越小（3.6：d = 64 → 1.49×，d = 128 → 1.28×）。

**测量要点**：预热包含编译；基线用同一进程、交替测；编译耗时单独报告。

### 3.8 attention 的显存与 FlashAttention

```
scores = Q Kᵀ / √d        # B×h×S×S
P      = softmax(scores)  # B×h×S×S
out    = P V
```

| 量 | 大小 | 备注 |
|---|---|---|
| 一个 S² 张量 | B·h·S²·4 字节 | 8 × 8192² × 4 = 2.0 GiB |
| forward 瞬态峰值 | **2 个**（scores 与 P 同时存在） | seq = 16384 时 16 GiB → OOM |
| 保存给反向的（融合 softmax） | **1 个**（P） | softmax 反向只需要输出；PV 保存的也是同一个 P |
| 保存给反向的（手写 softmax，fp32） | **2 个**（exp 输出 + P） | 本项目拟合 c = 2.00 |
| 保存给反向的（手写 softmax，bf16 autocast） | 约 1.5 个 | exp 输出 fp32（1 个）+ P 的 bf16 副本（0.5 个） |

**FlashAttention 为什么能省**：分块（tiling）加 **online softmax**，在 SRAM 里分块算 $QK^\top$、softmax、乘 V，**从不把 S×S 写回显存**；反向时用保存的 logsumexp **重算**所需的块。显存 $O(S^2) \to O(S)$，FLOPs 不变（反向多一点重算），但去掉了 S² 张量的反复读写，所以也更快。

**与其他手段的对比**：

| 手段 | 能否降低 $O(S^2)$ 显存 |
|---|---|
| bf16 | 只减半 |
| `torch.compile` | **不能**（数据证明） |
| 激活重算 | 能，代价是多一次前向 |
| FlashAttention / SDPA 融合 kernel | **能**，且更快（注意 flash 后端通常要求 fp16/bf16） |

### 3.9 Roofline 与 Amdahl

```
理想时间 = max( FLOPs / 峰值算力 , 字节数 / 峰值带宽 )          测量值不应低于它
算术强度 I = FLOPs / 字节数；I 低于 ridge point → 带宽受限
```

| 类型 | 判据 | 本项目例子 |
|---|---|---|
| 计算受限 | FMA 等计算单元接近饱和 | 大 GEMM（但本项目 SIMT fp32 只到约 50%） |
| 带宽受限 | DRAM 吞吐接近峰值 | softmax、逐元素（DRAM 80%～87%） |
| 延迟受限 | 两者都低，kernel 太小 | 小 ctx 的小 kernel |
| **发射受限** | GPU 在等 CPU，出现在 kernel 之间 | small / 128：约 20 µs/kernel |

**Amdahl**：整体加速比由"未被加速的部分"决定。优化前先看占比。

### 3.10 工具分工

| 工具 | 回答 | 不能信 |
|---|---|---|
| benchmark（不加 profiler） | 多快 | — |
| nsys | 时间花在哪、GPU 利用率、kernel 构成 | 绝对时间 |
| ncu | 单个 kernel 为什么慢 | duration |
| 显存快照 | 显存被什么占着、峰值时刻 | 不用于计时 |

**顺序**：先预测 → benchmark → nsys（算 GPU 利用率）→ ncu（只在利用率高时）→ 回填并解释差异。

---

## 4. 公式速查

### 4.1 FLOPs（forward，不含逐元素）

```
F = L·8·N·d² + L·6·N·d·d_ff + L·4·N·T·d + 2·N·d·V        N = B·T
训练一步 ≈ 3F（forward F + backward 2F）；不含 optimizer
```

### 4.2 参数个数（本项目的 SwiGLU 结构）

```
P = L·(4d² + 3·d·d_ff + 2d) + 2·V·d + d
small 128.6M；medium 423.2M；large 969.4M；"2.7B" 配置 3.41B
```

### 4.3 显存

| 项 | 公式 |
|---|---|
| 权重 / 梯度 / Adam | 4P / 4P / 8P（fp32） |
| 完整训练步静态 | **16P**（fp32） |
| autocast 额外 | 约 2P 的 bf16 权重副本 |
| 激活（粗估） | `L·4·(5·N·d_ff + c·B·h·T² + other·N·d) + 16·N·V`，本项目拟合 c = 2、other ≈ 9.4 |
| 一个 attention 分数张量 | `B·h·T²·4` 字节 |
| logits | `N·V·4` 字节 |
| GiB 与 GB | 1 GiB = 1.0737 GB，相差约 7%，写表要统一 |

### 4.4 其他

```
AdamW 访存下限 ≈ 28 B/参数  →  时间 ≥ 28·P / 带宽
Amdahl：  S = 1 / ( p/s + (1−p) )           p = 被加速部分的占比，s = 其加速比
达到算力 = FLOPs / 时间；利用率 = 达到值 / 峰值
```

---

## 5. 面试问答

**Q1：为什么 GPU 计时要同步？怎么计时才对？**
CPU 只是把 kernel 放进队列就返回，不同步量到的是发射时间。用 CUDA event 在 GPU 侧计时，读之前同步；预热后再测；报中位数和 std；固定 TF32、dtype、形状；卡要独占。

**Q2：预热为什么重要？预热要覆盖什么？**
CUDA/cuBLAS 初始化、kernel 懒加载、allocator 首次分配、Adam 状态首次分配、GPU 频率爬升、compile 编译，都是一次性开销。预热要覆盖所有"首次出现"的东西，最容易漏的是 Adam 的第一次 `step()` 与 backward 里新出现的 kernel。

**Q3：怎么测 backward 的时间？**
计算图只能用一次，所以每次迭代重新 forward，在 `loss.backward()` 前后各放一个 event。最好在同一步里同时切出 forward、backward、optimizer，只在最后同步一次。不要用 `fwdbwd − fwd`：no_grad 的前向与带图的前向不是一回事，会高估 backward，而且得不到各阶段自己的方差。

**Q4：backward 为什么大约是 forward 的 2 倍？**
线性层的反向要算对输入的梯度和对权重的梯度，各与前向 GEMM 同样大，所以 GEMM 是 2 倍。实际时间不一定正好 2 倍：访存受限的运算倍数不同。

**Q5：optimizer 的耗时与什么有关？为什么？**
只与参数量有关，与 batch、ctx 无关。AdamW 每个参数读写约 28 字节，是访存受限；朴素实现把一次更新拆成十多个 elementwise kernel，每个读写一遍张量，实测是下限的约 3 倍。用 fused 或 foreach 版本可以合并。

**Q6：训练 X 参数的模型要多少显存？**
静态 fp32 每参数 16 字节（权重 4、梯度 4、Adam 两份各 4），再加激活：随 batch 与 ctx 线性增长，attention 部分随 ctx 平方增长。例：3.4B 参数静态约 51 GiB，一张 16 GiB 的卡训练不了。

**Q7：显存峰值通常在训练的哪个阶段？**
forward 末尾附近：激活最多，梯度还没出现，loss 里还有 logits 的临时张量。backward 过程中边消费激活边释放，同时梯度增加，只要激活大于梯度就比 forward 末尾低。ctx 很短、激活很小时，峰值反而在 optimizer。验证方法：每个阶段前 reset 峰值，再读。

**Q8：激活显存和权重显存的区别？怎么降低激活？**
权重（含梯度、优化器状态）与 batch、ctx 无关；激活是为反向保存的中间张量，随 B·T 增长，attention 的分数矩阵随 T² 增长。降激活：激活重算、更小 batch、混合精度、FlashAttention；降权重：ZeRO / FSDP 分片、低精度优化器状态、offload。

**Q9：autocast 是什么？为什么参数留 fp32、loss 用 fp32？**
它按算子分类自动选精度：matmul/linear 用 bf16，exp/softmax/sum/归一化/损失升到 fp32。参数留 fp32 是为了保留小更新（bf16 只有约 8 位有效数字）；loss 里有对数、指数、求和，低精度不够准。bf16 不需要 GradScaler，fp16 才需要。

**Q10：混合精度为什么只快 1.8× 而不是几倍？**
Amdahl：GEMM 只占 fwd+bwd 的约 57%，即使它零耗时，上限也只有 2.35×；残差流和归一化仍是 fp32，额外的类型转换抵消了 elementwise 读写减半的收益；optimizer 不受影响。

**Q11：混合精度能省多少显存？**
没有想象的多。静态 16P 不变（fp32 主权重仍在），autocast 还缓存一份 bf16 权重（约 2P）；激活只降到 fp32 的约 2/3，因为残差流仍是 fp32。短 ctx 时甚至更高，ctx 512 时才净省约 15%。

**Q12：torch.compile 做了什么？能减少 attention 的显存吗？**
Dynamo 捕获图，Inductor 把逐元素与归约算子融合成少量 Triton kernel，减少 kernel 个数和显存读写遍数；GEMM 默认仍用 cuBLAS。它**不改变算法**，S×S 的矩阵仍要物化，所以不能降低 $O(S^2)$ 显存（我的数据：显存完全没变）。FlashAttention 靠分块和 online softmax 才能。

**Q13：FlashAttention 为什么省显存、还更快？**
分块在片上存储里完成 $QK^\top$、softmax、乘 V，不把 S×S 矩阵写回显存；反向用保存的 logsumexp 重算。显存 $O(S^2) \to O(S)$，FLOPs 基本不变，但去掉了 S² 张量的大量读写，所以更快。

**Q14：softmax 在 FLOPs 里占 2%，为什么时间占 2 倍以上？**
它是访存受限运算，时间取决于搬运的字节数而不是 FLOPs。手写版本由五个 kernel 组成，每个都完整读写一遍 S² 张量，ncu 里 DRAM 吞吐约 83%。

**Q15：怎么判断瓶颈是 kernel 本身还是 CPU 发射？**
算 GPU 利用率 = kernel 耗时之和 ÷ 阶段墙钟。接近 1 是 GPU 受限；明显低于 1（如 < 0.6）是发射受限，kernel 之间有空隙。例：small / 128 里每个 kernel 约 20 µs，是 eager 的发射开销量级。发射受限时 ncu 帮不上忙，要减少 kernel 个数（融合、compile、CUDA graph）。

**Q16：nsys 里 backward 的 kernel 为什么会丢？**
NVTX 区间是 CPU 侧、按线程记录的，kernel 按"发射线程 + 发射时刻"投影回区间；反向在 autograd 引擎的工作线程里发射，主线程里的 `backward` 区间收不到。解决：用总表减去 forward 和 optimizer（差分法）、按时间窗归因、`emit_nvtx`。

**Q17：怎么在 ncu 里判断一个 kernel 是计算受限还是带宽受限？**
看 Speed of Light：DRAM 吞吐高、计算低是带宽受限；计算高、DRAM 低是计算受限；两个都低是延迟受限（并行度不足）。注意 `Memory Throughput` 是"最繁忙的那一级"，GEMM 的 L1 和共享内存很忙但 DRAM 很低，要看 DRAM 一项。

**Q18：怎么验证你的测量是对的？**
让不同测法互相印证：阶段之和等于整步；两次独立运行一致；benchmark 与 nsys 的 kernel 耗时之和接近；显存的构成与理论账目闭合（权重、梯度、Adam 精确等于 4P、4P、8P）。我就是靠这种对账发现 compile 实验的基线有问题的。

**Q19：同样的 attention 数学，为什么保存的显存可以差 2 倍？**
取决于实现保存了哪些张量。融合的 softmax 反向只需要输出，且与 PV 矩阵乘共用同一份，所以只存 1 个 S² 张量；手写的 max-sub-exp-sum-div 里，`exp` 的输出和 softmax 输出是两个不同的张量，各存一份，共 2 个。我从完整模型拟合出 c = 2.00，微基准实测是 1×，两者对应这两种实现。

**Q20：如果让你优化这个训练步，按什么顺序？**
先按占比和上界排序：① 融合 optimizer（改动最小，对小模型和短 ctx 收益最大）；② compile 融合 elementwise（上界受限于非 GEMM 占比约 43%）；③ bf16/TF32 走 Tensor Core（GEMM 占 53%，是最大杠杆，要验证数值）；④ 长 ctx 时用 FlashAttention / 融合 softmax（收益随 ctx 平方增长）；⑤ 显存紧张时激活重算。

---

## 6. 坑清单

| 类别 | 坑 | 对策 |
|---|---|---|
| 计时 | 没同步就读时间 | event + synchronize |
| 计时 | 预热不够，或没覆盖 Adam 首次 step | 预热 ≥ 5 步并含 `opt.step()` |
| 计时 | 计时区里有 `.item()`、print | 挪出去 |
| 计时 | 用不同运行相减得阶段时间 | 同一步里 event 切段 |
| 计时 | 对同一个 loss 重复 backward 或用 retain_graph | 每次迭代重新 forward |
| 计时 | 基线与其他实验不一致却继续比较 | 先对账（同进程、交替测） |
| 计时 | 只报均值，或 std 小就当准 | 中位数 + 均值 + std，并与其他测法对账 |
| 精度 | fp32 对照组悄悄走了 TF32 | 显式关闭并记录读回值 |
| 精度 | autocast 包住了 backward 或 loss | 只包 forward |
| 精度 | 把参数转成 bf16 再叫 autocast | 参数保持 fp32 |
| 显存 | 把 Adam 状态混进 forward 峰值 | 先测无优化器的前向 |
| 显存 | 对整个预热过程录快照 | 只录一步 |
| 显存 | 用 `nvidia-smi` 的数填表 | 用 `max_memory_allocated`，写明 GiB |
| 显存 | OOM 后不清理就测下一个 | 释放引用、`empty_cache`，子进程隔离 |
| 显存 | 以为 bf16 一定省一半 | 实测，注意 2P 的权重副本 |
| 显存 | 以为 compile 能省 attention 显存 | 不能 |
| profile | NVTX 漏 backward | 差分法、时间窗、emit_nvtx |
| profile | 拿 nsys / ncu 的绝对时间当性能 | 只看构成与比例 |
| profile | 发射受限时去用 ncu | 先算 GPU 利用率 |
| 公式 | 11.6 的显存公式常数没校准 | 用多个 ctx 拟合 c 与线性项份数，并检验常数项是否 ≈ 0 |

---

## 7. 本项目关键数字（用作例子）

**配置**：small（L=12, d=768, d_ff=3072）128.6M 参数；medium（L=24, d=1024）423.2M；batch 4；词表 10000；fp32，TF32 关；RTX 5070 Ti（70 SM，16 GiB）。

| 类别 | 数字 |
|---|---|
| small / 512 阶段耗时 | forward 38.49、backward 76.09、optimizer 13.13 ms；整步 127.7 ms |
| medium / 512 阶段耗时 | 112.95、218.83、47.62 ms |
| optimizer 与 ctx | small 约 13.0 ms、medium 约 47.2 ms，四个 ctx 下不变 |
| bwd ÷ fwd | ctx ≥ 256 时约 1.9～2.0 |
| 前向达到的算力 | 约 40%～55% 的 33.4 TFLOP/s |
| GEMM / elementwise 占整步 | 53.5% / 43.2% |
| softmax 对 matmul | 耗时 2.15 : 1，FLOPs 0.02 : 1 |
| bf16 autocast | fwd+bwd 加速 1.78×（small）、1.80×（medium）；峰值显存 0.85× |
| 显存静态（medium） | 权重 1.58、梯度 1.58、Adam 3.15 GiB，合计 6.31 GiB = 16P |
| 保存的激活（medium fp32） | ctx 128 / 256 / 512：1.64 / 3.66 / 8.82 GiB |
| 激活拟合 | T² 项 c = 1.99，线性项约 29.5 份 N×d，常数项 ≈ 0 |
| T² 与线性项相等的 ctx | 约 990 |
| bf16 缓存的权重副本 | 约 2P = 0.79 GiB（medium） |
| attention 微基准 | seq = 16384 OOM；保存 1×、瞬态 2× B·S²·4B；compile 加速 1.28～1.51×，显存不变 |
| 3.4B 参数的配置，batch 4、ctx 128 | 静态 16P ≈ 50.8 GiB，合计约 56 GiB |

**尚未完成 / 有疑点**：3.3 warmup（无数据）；3.7 compile 基线与 3.2 不一致，加速比待重测；3.5 large / 256 与 3.8 的"最大单次分配"待补。