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