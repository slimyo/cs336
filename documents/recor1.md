---
## 3. Part 1：Profiling 与 Benchmarking
**目标**：建立可靠的测量方法，并解释时间和显存花在哪里。
### 3.1 实现 benchmark 脚本
要求：
- 参数：**batch size**、模型规模、context length、warmup 步数、测量步数、dtype
- 分别测 forward、forward+backward、optimizer step
- 每次计时前后 `torch.cuda.synchronize()`
- 输出均值与标准差
- **记录 TF32 设置**（`torch.backends.cuda.matmul.allow_tf32`；50 系 fp32
  matmul 开/关 TF32 差异巨大，所有 fp32 基线必须注明用哪种）
脚本路径：`scripts/benchmark.py`　TF32 设置：＿＿＿
### 3.2 端到端计时
固定 warmup=5、测量 10 步。context 取 128 / 256 / 512 / 1024。

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

说明：带 * 的是 no_grad 前向；其余 forward 是训练中的前向（含 loss、带计算图）。

saved -> results/bench/bench2_small_medium_large_ctx128-256-512-1024_b4_float32.json

**需要回答**：
- 哪个阶段最慢？随 ctx 如何变化？
- 测量方差大不大？为什么？
### 3.3 warmup 影响

**需要回答**：没有 warmup 时为什么慢且方差大？
（候选因素：CUDA context 初始化、kernel/JIT 编译、cudnn autotune、
allocator 首次分配与扩容、L2 cache 冷启动——按本机实测归因）

### 3.4 nsys profiling


| 模型 / ctx | 阶段 | 耗时最长 kernel | 占该阶段 % | 调用次数 | 是否 GEMM |
|---|---|---|---|---|---|
| small / 512 | forward | cuBLASLt fp32 GEMM（`magma_sgemmEx_kernel`） | 47.5% | 85 / 步 | 是 |
| small / 512 | backward | `cutlass_80_simt_sgemm_128x64_8x5_nt` | 14.6% | 72 / 步 | 是 |
| small / 512 | optimizer | `vectorized_elementwise_kernel`（乘法） | 44.5% | 666 / 步 | 否 |


| 问题 | 答案 |
|---|---|
| forward 中占比最高的 kernel？ | `magma_sgemmEx_kernel`，47.5%。73 次是 Linear，12 次是 QKᵀ |
| 同一 kernel 在 fwd+bwd 中占比变化？ | 47.5% → 16.6%。backward 里它只剩 12 次，GEMM 换成了 cutlass 的多种 tile。GEMM 整类：65.7% → 57.4% → 整步 53.5% |
| 除 matmul 外占时间最多的 kernel？ | 单个 kernel 是 vectorized elementwise 乘法，占整步 8.7%；按类别，elementwise 占整步 43.2% |
| 一次完整训练步中 matmul 占比？ | 53.5% |
| attention 内 softmax 与 matmul 的耗时比，与 FLOPs 比相比如何？ | 耗时比 2.15 : 1，FLOPs 比 0.020 : 1。按单位 FLOPs 的耗时算，softmax 比 matmul 高约 100 倍量级 |

三点注释：

- backward 第一名只有 14.6%，是因为同类 GEMM 按 tile 配置被拆成几个名字（`128x64 nt` 72 次、`128x64 nn` 60 次、`256x128 nn` 25 次）。我用上一次输出和这次的差值推算，backward 的 GEMM 整类约占 53%。
- optimizer 的 666 次是 111 的倍数，说明 111 个参数张量逐个发射，每个参数的同类 kernel 出现 6 次。
- Q5 的 matmul 指 QKᵀ 加 PV；FLOPs 里 softmax 按每元素 5 次运算估计，matmul 没有按因果 mask 减半。结论对这两个常数不敏感。

结论：
- **forward**：时间由 GEMM 主导（65.7%），最大的单项是 cuBLASLt 的 fp32 GEMM，覆盖 Linear 和 QKᵀ。
- **backward**：GEMM 仍占约一半，但分散在多种 cutlass tile 里；elementwise 个数是 forward 的约 2 倍，单次耗时更长。
- **optimizer**：全是 elementwise，且是逐参数发射（111 的倍数），瓶颈在发射开销而非计算，应换融合或 foreach 版本。
- **整步**：GEMM 53.5%，elementwise 43.2%，两类合计约 97%。
- **attention**：softmax 的 FLOPs 只有 matmul 的 2%，耗时却是 2 倍多，原因是 softmax 是由 5 个 kernel 拼成的访存密集运算（ncu 里 DRAM 约 83%）。

### 3.5 混合精度（forward / backward 取训练步中的阶段均值；显存单位 GiB）

| 模型 | ctx | dtype | forward (ms) | backward (ms) | 峰值显存 (GiB) | 加速比（fwd+bwd，相对 fp32） | loss0 |
|---|---|---|---|---|---|---|---|
| small | 512 | fp32 | 38.56 | 76.11 | 5.03 | 1.00× | 9.2808 |
| small | 512 | bf16 autocast | 20.18 | 44.29 | 4.25 | 1.78× | 9.2808 |
| medium | 512 | fp32 | 113.01 | 218.78 | 13.72 | 1.00× | 9.3120 |
| medium | 512 | bf16 autocast | 57.90 | 126.74 | 11.62 | 1.80× | 9.3120 |

峰值显存：有优化器时取完整训练步，--no-opt 时取反向结束时。loss0 为刚初始化时的 loss，fp32 与 bf16 应非常接近。

saved -> results/bench/bench2_small_medium_ctx512_b4_float32-bfloat16.json

附加：dtype 实测：small ctx=128 dtype=bfloat16

| 对象 | dtype |
|---|---|
| 模型参数 | float32 |
| 第一层 linear 输出 | bfloat16 |
| 所有 Linear 输出 | bfloat16 |
| LayerNorm / RMSNorm 输出 | float32 |
| TransformerBlock 输出（残差流） | float32 |
| logits | bfloat16 |
| loss | float32 |
| 梯度 | float32 |

### 3.6 attention 微基准（vanilla vs torch.compile）
固定 batch=8，单头，d_model ∈ {16, 32, 64, 128}，seq ∈ {256, 1024, 4096, 8192, 16384}。
> **理论预测（先写后测）**：backward 前需保存 scores + softmax 输出
> ≈ 2 × batch × S² × 4B。seq=16384 → ~17.2 GB → **预测 OOM 首现点**；
**forward / backward 时间 (ms)，vanilla attention（格内为 fwd / bwd）**
| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.16 / 0.34 | 0.20 / 0.44 | 4.76 / 11.41 | 25.06 / 41.53 | OOM |
| 32 | 0.17 / 0.34 | 0.20 / 0.44 | 4.85 / 9.61 | 25.23 / 41.63 | OOM |
| 64 | 0.16 / 0.33 | 0.34 / 0.49 | 6.46 / 10.77 | 29.41 / 41.12 | OOM |
| 128 | 0.06 / 0.26 | 0.52 / 0.79 | 9.57 / 16.05 | 37.56 / 68.40 | OOM |

**backward 前显存 (GiB)，vanilla —— 对账基准 1·B·S²·4B**
| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.02 | 0.05 | 0.52 | 2.03 | OOM |
| 32 | 0.02 | 0.05 | 0.53 | 2.05 | OOM |
| 64 | 0.02 | 0.05 | 0.55 | 2.08 | OOM |
| 128 | 0.02 | 0.06 | 0.58 | 2.14 | OOM |

**[附加] forward 瞬态峰值显存 (GiB) —— 对账基准 2·B·S²·4B，OOM 咬人的位置**
| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 | 0.02 | 0.08 | 1.03 | 4.04 | OOM |
| 32 | 0.02 | 0.08 | 1.04 | 4.05 | OOM |
| 64 | 0.02 | 0.09 | 1.05 | 4.09 | OOM |
| 128 | 0.02 | 0.10 | 1.09 | 4.17 | OOM |

**torch.compile 对比**
| d_model | seq | vanilla fwd | compiled fwd | vanilla bwd | compiled bwd | 加速比(fwd+bwd) | vanilla pre-G | compiled pre-G |
|---|---|---|---|---|---|---|---|---|
| 128 | 4096 | 9.57 | 6.30 | 16.05 | 10.70 | 1.51× | 0.58 | 0.58 |
| 64 | 8192 | 29.41 | 21.07 | 41.12 | 26.21 | 1.49× | 2.08 | 2.08 |
| 128 | 8192 | 37.56 | 31.30 | 68.40 | 51.55 | 1.28× | 2.14 | 2.14 |

**需要回答**：
- 最小 OOM 的配置是什么？显存主要来自哪个张量（N×N 的 score 矩阵）？
- 手算 backward 前保存的显存与实测是否吻合？
- 如何避免这个显存（引出 FlashAttention）？
### 3.7 完整模型 compile（ms，均值口径；加速比为全步）

| 模型 | ctx | vanilla fwd | compiled fwd | vanilla fwd+bwd | compiled fwd+bwd | vanilla 全步 | compiled 全步 | 全步加速比 |
|---|---|---|---|---|---|---|---|---|
| small | 512 | 54.55 | 38.98 | 157.84 | 108.68 | 173.32 | 125.41 | 1.38× |

注意：fwd 为训练步中带图前向段（含 loss），与 fwd(no_grad) 口径不同，勿混填。
    compile 首次调用含编译时间，warmup 必须排除。

### 3.8 显存 profiling
方法：`torch.cuda.memory._record_memory_history(max_entries=...)`，跑完导出
pickle，用 pytorch.org/memory_viz 查看。

（单位 GiB）
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


### 附加问题 1：激活显存与 ctx 的关系（带图前向后保存的激活，GiB）
- medium fp32: ctx128=1.64  ctx256=3.66  ctx512=8.82
  ctx 增大 4×：激活增大 5.4×（纯线性项应为 4×，纯 T² 项为 16×）
  拟合 act = a·N + b·B·h·T²：每层线性项等效 29.4 份 N×d 张量（11.6 取约 40），T² 项等效 c = 2.00（11.6 取 2）
- medium bf16 autocast: ctx128=1.82  ctx256=3.16  ctx512=6.68
  ctx 增大 4×：激活增大 3.7×（纯线性项应为 4×，纯 T² 项为 16×）
  拟合 act = a·N + b·B·h·T²：每层线性项等效 33.2 份 N×d 张量（11.6 取约 40），T² 项等效 c = 0.08（11.6 取 2）

### 附加问题 2：2.7B 理论峰值（用本配置的参数个数）
- P = 3.41B（SwiGLU 三矩阵、词表 10000，不是名字里的 2.7B）
- batch=4, ctx=128：静态 16P = 50.8 GiB；激活估算（c=2.00, other=9.4）= 5.2 GiB；合计 ≈ 55.9 GiB（60.1 GB）
- 仅推理：fp32 权重 4P = 12.7 GiB；纯 bf16 权重 2P = 6.3 GiB

### 附加问题 4：混合精度对峰值显存的影响（bf16 autocast ÷ fp32）
- medium ctx=128: 保存的激活 1.11×，no_grad 前向峰值 1.44×，完整步峰值 1.02×
- medium ctx=256: 保存的激活 0.86×，no_grad 前向峰值 1.40×，完整步峰值 0.94×
- medium ctx=512: 保存的激活 0.76×，no_grad 前向峰值 1.35×，完整步峰值 0.85×

saved -> results/mem/summary.json

说明：forward 结束时的『激活/其他』≈ 保存给反向用的激活（autocast 下含缓存的 bf16 权重副本）。
附加：
| 问题 | 答案 |
|---|---|
| 激活值显存与 ctx 的关系？ |  |
| 2.7B 理论峰值（参数+梯度+Adam+激活估算）？ |  |
| 最大的单次分配（如 attention score）是多大？ |  |
| 混合精度对峰值显存的影响？ |  |