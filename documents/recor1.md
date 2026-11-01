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