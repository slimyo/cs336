
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
| 模型 | ctx | forward 均值±std (ms) | backward 均值±std (ms) | optimizer step 均值±std (ms) | 状态 |
|---|---|---|---|---|---|
| small | 128 |  |  |  |  |
| small | 256 |  |  |  |  |
| small | 512 |  |  |  |  |
| small | 1024 |  |  |  |  |
| medium | 128 |  |  |  |  |
| medium | 256 |  |  |  |  |
| medium | 512 |  |  |  |  |
| medium | 1024 |  |  |  |  |
| large | 128 |  |  |  |  |
| large | 256 |  |  |  |  |
| large | 512 |  |  |  |  |
| large | 1024 |  |  |  |  |
| xl（仅 fwd） | 128 |  | N/A | N/A |  |
| 2.7B（仅 fwd, bf16） | 128 |  | N/A | N/A |  |
状态列填：OK / OOM / 降配（写明降成什么）。
预填预期：xl 仅 fwd fp32 约 4×4B=16GB 超显存 → 需降配或 bf16；2.7B bf16
参数 ~5.4GB，ctx=128 激活可控，预期 OK。实测后修正。
**需要回答**：
- 哪个阶段最慢？随 ctx 如何变化？
- 测量方差大不大？为什么？
### 3.3 warmup 影响
模型固定为 small，ctx=256。
| warmup 步数 | forward 均值±std (ms) | forward+backward 均值±std (ms) |
|---|---|---|
| 0 |  |  |
| 1 |  |  |
| 2 |  |  |
| 5 |  |  |
| 10 |  |  |
**需要回答**：没有 warmup 时为什么慢且方差大？
（候选因素：CUDA context 初始化、kernel/JIT 编译、cudnn autotune、
allocator 首次分配与扩容、L2 cache 冷启动——按本机实测归因）
### 3.4 nsys profiling
方法：`torch.cuda.nvtx.range` 给 forward / backward / optimizer 以及
attention 内部（scores、softmax、weighted sum）打标。
```bash
nsys profile -o <name> --trace=cuda,nvtx,osrt uv run python scripts/benchmark.py --model small --ctx 512
```
| 模型 / ctx | 阶段 | 耗时最长 kernel | 占该阶段 % | 调用次数 | 是否 GEMM |
|---|---|---|---|---|---|
| small / 512 | forward |  |  |  |  |
| small / 512 | backward |  |  |  |  |
| small / 512 | optimizer |  |  |  |  |
附加记录：
| 问题 | 答案 |
|---|---|
| forward 中占比最高的 kernel？ |  |
| 同一 kernel 在 fwd+bwd 中占比变化？ |  |
| 除 matmul 外占时间最多的 kernel（如 elementwise）？ |  |
| 一次完整训练步中 matmul 占比？ |  |
| attention 内 softmax 与 matmul 的耗时比，与 FLOPs 比相比如何？ |  |
### 3.5 混合精度（bf16 autocast）
| 模型 | ctx | dtype | forward (ms) | backward (ms) | 峰值显存 (GB) | 加速比（相对 fp32） |
|---|---|---|---|---|---|---|
| small | 512 | fp32 |  |  |  | 1.00× |
| small | 512 | bf16 autocast |  |  |  |  |
| medium | 512 | fp32 |  |  |  | 1.00× |
| medium | 512 | bf16 autocast |  |  |  |  |
| large | 256 | fp32 |  |  |  | 1.00× |
| large | 256 | bf16 autocast |  |  |  |  |
附加：autocast 下各类算子的 dtype（作业必答）。
| 对象 | dtype |
|---|---|
| 模型参数 |  |
| 第一层 linear 输出 |  |
| LayerNorm / RMSNorm 输出 |  |
| logits |  |
| loss |  |
| 梯度 |  |
### 3.6 attention 微基准（vanilla vs torch.compile）
固定 batch=8，单头，d_model ∈ {16, 32, 64, 128}，seq ∈ {256, 1024, 4096, 8192, 16384}。
> **理论预测（先写后测）**：backward 前需保存 scores + softmax 输出
> ≈ 2 × batch × S² × 4B。seq=16384 → ~17.2 GB → **预测 OOM 首现点**；
> seq=8192 → ~4.3 GB 可过。实测对照此预测。
**forward / backward 时间 (ms)，vanilla attention**

| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 |  |  |  |  |  |
| 32 |  |  |  |  |  |
| 64 |  |  |  |  |  |
| 128 |  |  |  |  |  |

**backward 前显存 (GB)**（记录 OOM 的最小配置）

| d_model \ seq | 256 | 1024 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 16 |  |  |  |  |  |
| 32 |  |  |  |  |  |
| 64 |  |  |  |  |  |
| 128 |  |  |  |  |  |

**torch.compile 对比（选 3 个代表配置）**

| d_model | seq | vanilla fwd | compiled fwd | vanilla bwd | compiled bwd | 加速比 |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

**需要回答**：
- 最小 OOM 的配置是什么？显存主要来自哪个张量（N×N 的 score 矩阵）？
- 手算 backward 前保存的显存与实测是否吻合？
- 如何避免这个显存（引出 FlashAttention）？
### 3.7 完整模型 compile

| 模型 | ctx | vanilla fwd | compiled fwd | vanilla fwd+bwd | compiled fwd+bwd | vanilla 全步 | compiled 全步 |
|---|---|---|---|---|---|---|---|
| small | 512 |  |  |  |  |  |  |
| medium | 512 |  |  |  |  |  |  |

注意：compile 首次调用含编译时间，warmup 必须排除。
### 3.8 显存 profiling
方法：`torch.cuda.memory._record_memory_history(max_entries=...)`，跑完导出
pickle，用 pytorch.org/memory_viz 查看。
| 模型 | ctx | dtype | forward 峰值 (GB) | 完整训练步峰值 (GB) | 理论静态显存 (GB) | 主要占用来源 |
|---|---|---|---|---|---|---|
| medium | 128 | fp32 |  |  |  |  |
| medium | 512 | fp32 |  |  |  |  |
| medium | 128 | bf16 |  |  |  |  |
| medium | 512 | bf16 |  |  |  |  |
| large | 256 | fp32 |  |  |  |  |
| large | 256 | bf16 |  |  |  |  |
附加：
| 问题 | 答案 |
|---|---|
| 激活值显存与 ctx 的关系？ |  |
| 2.7B 理论峰值（参数+梯度+Adam+激活估算）？ |  |
| 最大的单次分配（如 attention score）是多大？ |  |
| 混合精度对峰值显存的影响？ |  |
---
## 4. Part 2：FlashAttention-2
**目标**：实现 FA2 的 forward 和 backward，通过测试，并与 PyTorch 基线对比。
### 4.1 里程碑
- [ ] **M0：朴素 attention + autograd.Function 管道验证**（迷你脚本对 SDPA
      验证前向+梯度，先隔离 plumbing 错误）← M1 前置
- [ ] M1：纯 PyTorch tiled forward（`autograd.Function`），保存 logsumexp L，通过测试
- [ ] M2：PyTorch backward（用保存的 LSE 重算 P；D = rowsum(dO⊙O) 技巧），通过梯度测试
- [ ] M3：Triton forward kernel，通过测试
- [ ] M4：causal masking，并跳过完全被 mask 的 tile
- [ ] M5：Triton backward（加分项）
- [ ] M6：benchmark 与调参
### 4.2 正确性记录

| 版本 | 测试 | 是否通过 | 最大绝对误差 | 备注 |
|---|---|---|---|---|
| PyTorch tiled fwd | test_flash_forward_pass_pytorch |  |  |  |
| PyTorch bwd | test_flash_backward_pytorch |  |  |  |
| Triton fwd（非 causal） | test_flash_forward_pass_triton[False] |  |  |  |
| Triton fwd（causal） | test_flash_forward_pass_triton[True] |  |  |  |
| Triton bwd | test_flash_backward_triton[False/True] |  |  |  |

### 4.3 Tile 大小调参（Blackwell 消费级共享内存较小，需要自己扫）
固定 seq=4096, d=64, bf16, causal。
约束主项：`num_stages × (Bq + Bk) × d × 2B` ≲ smem（sm_120 ≈128KB/SM）；
num_stages 通常是最敏感参数，编译报错为准试探。

| Bq | Bk | num_warps | num_stages | fwd 延迟 (ms) | 是否因共享内存失败 |
|---|---|---|---|---|---|
| 32 | 32 |  |  |  |  |
| 64 | 32 |  |  |  |  |
| 64 | 64 |  |  |  |  |
| 128 | 64 |  |  |  |  |
| 128 | 128 |  |  |  |  |

### 4.4 与 PyTorch 对比
用 `triton.testing.do_bench`，**batch=1**，seq ∈ {128, 256, …, 65536}
（按显存能跑到哪算哪），d ∈ {16, 32, 64, 128}，dtype ∈ {bf16, fp32}，causal=True。
> 理论预填：PyTorch 基线 fp32 scores @ 65536 = 17.2 GB 必 OOM；bf16 的
> scores+softmax 亦 ~17 GB OOM。FA 显存 O(N·d) 线性，65536 应可跑——
> 这是对比的核心卖点。
**forward**

| seq | d | dtype | PyTorch (ms) | Triton FA (ms) | 加速比 | 显存 PyTorch (GB) | 显存 FA (GB) |
|---|---|---|---|---|---|---|---|
| 1024 | 64 | bf16 |  |  |  |  |  |
| 4096 | 64 | bf16 |  |  |  |  |  |
| 16384 | 64 | bf16 |  |  |  |  |  |
| 65536 | 64 | bf16 |  |  |  |  |  |

**backward 与 fwd+bwd**
| seq | d | dtype | PyTorch bwd (ms) | FA bwd (ms) | PyTorch fwd+bwd (ms) | FA fwd+bwd (ms) |
|---|---|---|---|---|---|---|
| 1024 | 64 | bf16 |  |  |  |  |
| 4096 | 64 | bf16 |  |  |  |  |
| 16384 | 64 | bf16 |  |  |  |  |
**吞吐（TFLOPs/s）**：attention forward FLOPs ≈ 4·N²·d（causal 约减半），
backward 约 2.5× forward。
| seq | d | FA fwd TFLOPs/s | FA bwd TFLOPs/s | 备注 |
|---|---|---|---|---|
|  |  |  |  |  |
> **吞吐参照法（本机专用）**：先实测 cuBLAS bf16 大 GEMM（如 4096³）峰值
> TFLOPs 作为 100%，FA2 达其 30–50% 属正常。与 H100（~3.35TB/s 带宽、
> TMA/WGMMA）差距来源：带宽 ~0.9TB/s、sm_120 无 TMA/tcgen05、smem 较小。
**需要回答**：
- 在哪个 seq 之后 FA 开始领先？为什么小 seq 反而可能更慢？
- FA 的显存随 seq 是线性还是平方增长？
- 与 H100 上的典型数值差距的原因？
### 4.5（可选）与已有实现对比
| 实现 | fwd (ms) | bwd (ms) | 备注 |
|---|---|---|---|
| 本作业 Triton FA |  |  |  |
| PyTorch SDPA（FlashAttention 后端） |  |  | 上界参考；注明本卡选中哪个 backend |
---
## 5. Part 3：分布式数据并行（DDP）
**目标**：从零实现数据并行，量化通信开销，并验证 overlap 的收益。
### 5.1 里程碑
- [ ] M1：单卡训练脚本作为"真值"
- [ ] M2：naive DDP（逐参数 all_reduce），与单卡逐步对齐权重
- [ ] M3：flat DDP（梯度打平，一次 all_reduce）
- [ ] M4：overlap（`register_post_accumulate_grad_hook` 异步 all_reduce +
      **pinned staging buffer**）
- [ ] M5：bucketed DDP（按 MB 分桶 + overlap）
- [ ] M6：benchmark 与 nsys 验证 overlap
### 5.2 正确性记录
> ⚠️ tied weights（ToyModelWithTiedWeights）：共享参数产生两份梯度，
> 合并顺序必须全 rank 一致，否则 rank divergence 挂死
> （nccl_sweep 事故教训：集合通信绝不放进 `if rank==0`）。
| 版本 | 与单卡 loss 对齐（N 步） | 权重最大误差 | pytest |
|---|---|---|---|
| naive |  |  |  |
| flat |  |  |  |
| overlap |  |  |  |
| bucketed |  |  |  |
### 5.3 实验配置
| 项目 | 取值 |
|---|---|
| world size | 2（NCCL，传输已知 SHM） |
| 模型 | medium / large（二选一或都做） |
| global batch | 8（每卡 4） |
| ctx | 256 |
| dtype | bf16 autocast / fp32 |
| warmup / 测量步数 | 5 / 10 |
### 5.4 各版本对比
| 版本 | 每步总时间 (ms) | 通信时间 (ms) | 通信占比 | 相对 naive 加速 | 峰值显存 (GB) |
|---|---|---|---|---|---|
| 单卡（batch=4，无通信，作为参考） |  | 0 | 0 |  |  |
| naive |  |  |  | 1.00× |  |
| flat |  |  |  |  |  |
| overlap（逐参数） |  |  |  |  |  |
| bucketed（best bucket） |  |  |  |  |  |
通信时间的获取：nsys timeline 统计 NCCL kernel 总时长，或 CUDA event
夹 all_reduce 区间。
### 5.5 Bucket 大小扫描
> **1.3 实测修正预期**：≥1MB 区间 NCCL 带宽平坦无地板 → bucket 下限不是
> NCCL 效率，而是 **CPU launch 开销 + overlap 粒度**。如需 sub-MB 拐点
> 数据支撑下限，跑 `scripts/nccl_knee.py`。
| bucket size (MB) | bucket 数 | 每步时间 (ms) | 通信占比 | overlap 程度（nsys 观察） |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 10 |  |  |  |  |
| 100 |  |  |  |  |
| 1000（≈单桶） |  |  |  |  |
### 5.6 用基线带宽做理论对照
| 项目 | 值 |
|---|---|
| 模型梯度总量（字节） | 参数量 × 4（fp32）或 × 2（bf16） |
| 实测 busbw（来自 1.3） | **5.2 GB/s** |
| 理论最短通信时间 | 梯度量 ÷ busbw（medium 30M→23ms，large 124M→95ms） |
| 实测通信时间 |  |
| 差距原因（实测后勾选） | ① sub-MB 小消息（knee 补测）② launch 开销 ③ SHM 经主机 staging（无 P2P）④ 同步等待 |
**需要回答**：
- naive 与 flat 差距主要来自什么？（逐参数消息是 KB 级，在 1.3 平坦区以下——结合 knee 数据判断是延迟还是 launch 主导）
- 如果没有 P2P / NVLink，overlap 的收益会比 H100 上更大还是更小？
  → **更大**：本机通信占比高一个量级（5.2 vs ~450 GB/s），可隐藏的绝对时间
  更长；上限受无 P2P 限制（staging 经 pinned 主存两段拷贝）。
- 最优 bucket size 受什么影响？（下限：每桶 ≥1MB 保带宽（本机实测平坦区起点）；
  上限：桶够多以保留 overlap 机会）
- 如何在 nsys 上直接看到 overlap？（backward kernel 与 NCCL kernel 在
  不同 stream 上并行，截图标注）
---
## 6. Part 4：优化器状态分片
**目标**：实现类似 ZeRO-1 的分片，量化显存收益与时间代价。
### 6.1 里程碑
- [ ] M1：包装 `torch.optim.Optimizer`，每个 rank 只持有部分参数的 optimizer state
- [ ] M2：step 后将更新后的参数 broadcast 给所有 rank
- [ ] M3：通过测试，并与非分片版本对齐
- [ ] M4：显存和速度 benchmark
- [ ] M5：完成作业中的 memory accounting 题
### 6.2 峰值显存（单卡，GB）
测量点用 `torch.cuda.memory_allocated()` / `max_memory_allocated()` 打点。
| 配置 | 模型初始化后 | optimizer.step 前 | optimizer.step 后 | 全程峰值 |
|---|---|---|---|---|
| DDP，无分片 |  |  |  |  |
| DDP + 优化器分片 |  |  |  |  |
| 差值 |  |  |  |  |
> 理论节省 = **4P 字节/卡**（Adam m+v 8P→4P）：medium 30M→0.12GB，
> large 124M→0.5GB。实测对齐看 allocator 碎片。
### 6.3 速度
| 配置 | 每步时间 (ms) | optimizer step 时间 (ms) | 额外通信 (ms) |
|---|---|---|---|
| 无分片 |  |  | 0 |
| 分片 |  |  |  |
> 额外通信理论值 ≈ 4P ÷ 5.2 GB/s（medium→23ms，large→95ms/step），
> 可能占步时 10–20%；NVLink 下同操作 <2ms——"互联决定分片性价比"
> 的量化论据。
### 6.4 理论显存核算（fp32 + AdamW）
设参数量为 P，world size = 2。
| 组成 | 无分片（字节） | 分片后（字节/卡） |
|---|---|---|
| 参数 | 4P | 4P |
| 梯度 | 4P | 4P |
| Adam m + v | 8P | 4P |
| 激活 | 与 batch、ctx 有关 | 不变 |
| 合计（不含激活） | 16P | 12P |
| 模型 | P | 无分片（GB） | 分片（GB/卡） | 实测差值（GB） | 与理论是否吻合 |
|---|---|---|---|---|---|
| medium |  |  |  |  |  |
| large |  |  |  |  |  |
**需要回答**：
- 实测节省是否等于理论的 4P？差异来自哪里（allocator 碎片、临时 buffer）？
- 分片对每步时间的代价有多大？这个代价在 PCIe+SHM（本机）与 NVLink 下有何不同？
- 与 ZeRO-2 / ZeRO-3 相比，还有哪些状态可以分片？（梯度 4P、参数 4P）
---
## 7. 报告整理清单
- [ ] 每个 problem 的答案都有"数字 + 一句因果解释"
- [ ] 所有降配处都有说明
- [ ] 环境节直接引用第 0 节档案（含 P2P=CNS、NCCL=SHM、busbw 5.2、Gen4）
- [ ] TF32 设置写明（3.1）
- [ ] 所有图表标注环境（GPU 型号、PyTorch、Triton、NCCL 版本）
- [ ] nsys 截图标出关键区间（backward、NCCL overlap）
- [ ] OOM 配置与理论估算并列（3.6 / 4.4 已有预填预测可对照）
- [ ] 代码通过全部 pytest 后再做最终 benchmark
- [ ] 保存原始数据（csv：sweep / knee / benchmark）与 nsys / memory snapshot
- [ ] 附录（可选加分）：rank divergence 排查记录（nccl_sweep 事故 →
      分布式调试方法论）
## 8. 排期（约 2 周）
| 天数 | 内容 | 状态 |
|---|---|---|
| D1 | 环境、通信基线（1.3）、smoke | ✅ 2026-10-26 完成 |
| D2–D3 | Part 1 全部实验（3.1 ~ 3.8） |  |
| D4–D5 | Part 2：M0–M4，正确性优先 |  |
| D6 | Part 2：调参与 benchmark（4.3 ~ 4.5） |  |
| D7–D9 | Part 3：M1–M6 |  |
| D10–D11 | Part 4 与 memory accounting |  |
| D12–D14 | 补实验、整理报告 |  |
> 提示：Triton kernel 编译/调试等待间隙可穿插 Part 1 的 benchmark 脚本，
> 两者互不依赖。
## 9. 服务器使用注意事项
- 用 `CUDA_VISIBLE_DEVICES` 明确指定卡；Part 1 / 2 单卡，Part 3 / 4 用 2 卡。
- 做计时实验前 `nvidia-smi` 确认无残留进程/他人任务；卡死 `pkill -f torchrun`
  后再清点显存。
- **分布式三纪律**（nccl_sweep 事故入库）：
  ① 集合通信绝不放进 `if rank == 0`（只包输出）
  ② 起跑前清场（显存与进程）
  ③ 无声挂死 = rank divergence：`py-spy dump --pid <pid>` 看各 rank 卡点；
     NCCL watchdog 默认 10 分钟超时。
- P2P = CNS 是芯片组级，**不要试图修**；`NCCL_SHM_DISABLE=1` 仅作对照组用。
- 自定义通信 / overlap 的 staging buffer **必须 pinned**，否则经主机拷贝带宽减半。
- shell 里 `(vllm)` conda 环境与作业无关：一律 `uv run`（走项目 .venv），
  勿把依赖 pip 装错环境。
- 长任务用 `tmux` 或 `screen`；nsys 输出的 `.nsys-rep` 下载到本地用 GUI 查看。
- 实测前后记录 `nvidia-smi -q -d CLOCK,POWER`，必要时 `-lgc` 固定频率以降低方差。
- 先用小规模配置跑通全流程，再扩大规模，避免长时间任务才发现 bug。

