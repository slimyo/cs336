# Step 1（按服务器规范重写）：测出这块卡的"天花板"

**目标**：得到四个参照数，后面所有模型数字都拿它们对照。

| 符号 | 含义 | 用在哪 |
|---|---|---|
| P_fp32 | fp32（TF32 关）matmul 平台吞吐 | fp32 基线的预期时间 |
| P_tf32 | fp32（TF32 开）matmul 平台吞吐 | 可选参考行 |
| P_bf16 | bf16 matmul 平台吞吐 | 混合精度的预期时间 |
| BW | 显存拷贝带宽 | optimizer step 等访存密集算子的预期 |

## 0. 前置检查（共享纪律）

```bash
cd /data2/c220/hz/cs336
nvidia-smi          # 两卡显存应接近 0，且没有 vllm server
mkdir -p results
```

## 1. 脚本 `scripts/matmul_peak.py`


## 2. 运行

用 tmux 开两个窗格，一个监控，一个跑实验：

```bash
# 窗格 A：监控（-i 用的是物理卡号）
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -l 1 > results/clock_gpu0.csv

# 窗格 B：跑实验
CUDA_VISIBLE_DEVICES=0 uv run python scripts/matmul_peak.py --tag gpu0
```

跑完后停掉监控，再对 1 号卡重复一遍（监控里 `-i 1`，实验里 `CUDA_VISIBLE_DEVICES=1 ... --tag gpu1`）。

## 3. 怎么读结果

- 看每种模式的 **n=8192 和 16384 的 median**。小 n 吞吐低是因为没喂满 GPU，平台值才是天花板。
- 预期顺序：bf16 > fp32(TF32 开) > fp32(TF32 关)。如果 TF32 开和关几乎一样，检查输出里 `float32_matmul_precision` 那行是否真的变了。
- 两张卡的平台值应接近。差距明显：检查有没有别的任务，或者 `clock_gpu*.csv` 里 SM 频率有没有掉。
- 平台值与 median 的差异：`max` 与 `median` 差距大，说明测量抖动或降频，以 median 为准并在表里注明。
- memcopy 的参照：规格书标称显存带宽约 896 GB/s，实测通常是它的一部分，以实测为准。

## 4. 填表

| 项目 | 卡 0 | 卡 1 | 采用值 |
|---|---|---|---|
| P_fp32（TF32 关，n=8192 / 16384 median，TFLOPs/s） | 32.84/33.04 | 33.69/33.93 |  |
| P_tf32（TF32 开） | 48.84/49.74 | 48.72/49.57 |  |
| P_bf16 |92.36/98.41  | 95.03/98.74 |  |
| BW（memcopy，GB/s） | 768.0 | 769.6 |  |
| 持续 SM 频率 / 功耗（来自 clock csv） | - | -  |  |

## 5. 做两个决定并写进 3.1

1. **fp32 基线用 TF32 关**（PyTorch 默认，也是最干净的"纯 fp32"）。TF32 开作为参考行。在 3.1 的"TF32 设置"栏写：`torch.set_float32_matmul_precision("highest")`，并在 `benchmark.py` 里显式设置，不依赖默认值。
2. **后续所有"预期时间"都用本表的采用值**，不要用规格书峰值。

---

# Step 2：用手算建立预期

## 0. 先更正我之前的三处数字

我之前给的参数量（medium 0.35B、large 0.76B、xl 1.6B）是按 GPT-2 命名的名义规模，没有按作业实际结构算。作业的 FFN 是 SwiGLU，有 **3 个**矩阵，且 d_ff = 4·d_model，所以每层的权重是 16d²，比 GPT-2 的 12d² 大。重算后：

| 我之前说的 | 实际应为 |
|---|---|
| small 非 embedding 约 85M，一步约 1 TFLOP | 约 113M，一步约 **1.6 TFLOP**（ctx=512，batch=4） |
| large 完整训练约 12GB，勉强能放 | 约 **14.4 GiB** 静态，几乎必然 OOM |
| xl 约 1.6B | 约 **2.0B**；2.7B 约 **3.4B** |

这会改变降配策略，见第 7 节。**以代码实测的参数量为准**，下面第 1 节教你怎么核对。

## 1. 参数量

**每层**（无 bias）：
- 注意力：Wq、Wk、Wv、Wo 各 d×d，共 4d²
- FFN（SwiGLU）：W1、W3 为 d×d_ff，W2 为 d_ff×d，共 3·d·d_ff
- 两个 RMSNorm：2d（可忽略）

**全模型**：

```
P = L·(4d² + 3·d·d_ff)  +  2·V·d  + d
         层内权重         embedding + lm_head（不绑定）
```

**小模型手算**（d=768，d_ff=3072，L=12，V=10000）：
- 每层：4×768² = 2,359,296；3×768×3072 = 7,077,888；合计 9,437,184
- 12 层：113,246,208 ≈ 113.2M
- embedding + lm_head：2×10000×768 = 15,360,000
- 总计约 128.6M

**用 meta 设备核对，不占显存**（`scripts/count_params.py`）：

```python
import torch
from cs336_basics.model import BasicsTransformerLM
with torch.device("meta"):
    m = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=768,
        num_layers=12, num_heads=12, d_ff=3072, rope_theta=10000.0)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M")
```

```bash
uv run python scripts/count_params.py
```

手算和实测对不上，先查两点：lm_head 是否与 embedding 绑定（绑定则减去 V·d），构造函数参数名是否与你仓库一致。

## 2. 训练一步的 FLOPs

只有矩阵乘法才算 FLOPs（RMSNorm、softmax、RoPE、逐元素运算是访存密集，FLOPs 可忽略）。一个 (m×k)·(k×n) 的 matmul 是 2mkn FLOPs。

设 B 为 batch，ctx 为序列长度，T = B·ctx 为 token 数。

**forward 的三项**：

| 项 | 公式 | 说明 |
|---|---|---|
| 层内线性层 | 2 · T · L·(4d² + 3·d·d_ff) | 每个 token 对每个权重做一次乘加 |
| lm_head | 2 · T · d · V | embedding 查表不是 matmul，不算 |
| 注意力 QKᵀ 与 AV | L · 4 · B · ctx² · d | 两次 matmul，各 2·B·ctx²·d，vanilla 实现不跳过因果 mask 的一半 |

**backward ≈ 2 × forward**：对输入求梯度和对权重求梯度，各是一次与 forward 同规模的 matmul。所以：

```
训练一步 FLOPs = 3 × forward FLOPs
forward : backward ≈ 1 : 2
```

你会在文献里看到的 "6·N·T" 就是这个公式略去注意力项的版本，可以用来交叉验证。

## 3. 完整的小模型算例（B=4，ctx=512，T=2048）

| 项 | 计算 | 结果 |
|---|---|---|
| 层内线性层 | 2 × 2048 × 113,246,208 | 463.9 GFLOP |
| lm_head | 2 × 2048 × 768 × 10000 | 31.5 GFLOP |
| 注意力 | 12 × 4 × 4 × 512² × 768 | 38.7 GFLOP |
| **forward** |  | **534.0 GFLOP** |
| **训练一步（×3）** |  | **1.60 TFLOP** |

交叉验证：6 × (113.2M + 7.7M) × 2048 ≈ 1.49 TFLOP，加上注意力的 3×38.7 = 0.12 TFLOP，约 1.60 TFLOP，吻合。

## 4. 各模型的预期 FLOPs（B=4，ctx=512）

| 模型 | 参数量 | forward (TFLOP) | 一步 = 3×fwd (TFLOP) |
|---|---|---|---|
| small | 128.6M | 0.534 | 1.60 |
| medium | 423.1M | 1.794 | 5.38 |
| large | 969.3M | 4.111 | 12.33 |
| xl | 1998M | 8.441 | 仅 forward |
| 2.7B | 3407M | 14.19 | 仅 forward |

其他 ctx 的换算：线性层和 lm_head 两项与 T 成正比，注意力项与 ctx² 成正比。例如 ctx=128（T=512）：前两项变成 1/4，注意力项变成 1/16。

你也可以用脚本，避免手算出错（`scripts/flops_estimate.py`）：

```python
CFG = {"small": (768, 3072, 12), "medium": (1024, 4096, 24), "large": (1280, 5120, 36),
       "xl": (1600, 6400, 48), "2.7B": (2560, 10240, 32)}
V = 10000

def est(name, B, ctx):
    d, ff, L = CFG[name]
    T = B * ctx
    n_layer = L * (4 * d * d + 3 * d * ff)
    fwd = 2 * T * n_layer + 2 * T * d * V + L * 4 * B * ctx * ctx * d
    return n_layer + 2 * V * d, fwd, 3 * fwd

for name in CFG:
    P, fwd, step = est(name, 4, 512)
    print(f"{name:7} params {P/1e6:8.1f}M  fwd {fwd/1e12:6.3f}T  step {step/1e12:7.3f}T")
```

## 5. 从 FLOPs 到预期时间

```
t_pred = FLOPs / P
```

P 取 Step 1 测出的平台值：纯 fp32 基线用 P_fp32，bf16 autocast 用 P_bf16。

**示意**（数字是假设的，你要换成自己测的）：设 P_fp32 = 25 TFLOPs/s，则 small 一步预期约 1.60 / 25 = 64 ms，其中 forward 约 21 ms，backward 约 43 ms。

**MFU（利用率）= FLOPs / t_meas / P**，是你实测数据最该报告的一个派生指标。

| 比值 t_meas / t_pred | 含义与下一步 |
|---|---|
| < 1 | 不可能。通常是漏了 `synchronize`，或 FLOPs 算少了 |
| 1 到 2 | 正常。大模型、长 ctx 应落在这一档 |
| 2 到 4 | 非 matmul 算子（norm、softmax、elementwise）占了显著时间，值得在 nsys 里看 |
| > 4 | 多半是 kernel 发射开销（小模型、短 ctx）或脚本有问题 |

小模型加短 ctx（如 small、ctx=128）的比值偏高是正常的：计算量太小，kernel 数量固定，launch 开销占比上升。这会让"MFU 随模型和 ctx 增大而上升"，是 3.2 要回答的现象。

## 6. optimizer step 的预期（访存密集，不用 FLOPs）

AdamW 对每个参数要读 p、g、m、v（16 B），写 p、m、v（12 B），约 28 B/参数。

```
t_opt_下限 ≈ 28 × P_params / BW
```

例：small，128.6M × 28 B ≈ 3.6 GB，除以你实测的 BW（假设 600 GB/s）约 6 ms。

这只是下限。如果 `cs336_basics` 的 AdamW 是逐参数的 Python 循环（A1 要求自己写的那种），实测会明显更高，差距主要来自 kernel 发射开销，而不是带宽。这是个值得写进报告的发现。

## 7. 显存预算（决定降配）

fp32 权重 + 梯度 + AdamW 两个状态 = 16 B/参数，单位用 GiB（与 `torch.cuda.max_memory_allocated` 一致）：

| 模型 | 参数量 | 训练静态 (GiB) | fp32 仅 forward (GiB) | bf16 仅 forward (GiB) | 在 16GB 卡上 |
|---|---|---|---|---|---|
| small | 128.6M | 1.9 | 0.5 | 0.2 | 完整训练 OK |
| medium | 423.1M | 6.3 | 1.6 | 0.8 | 完整训练 OK |
| large | 969.3M | 14.4 | 3.6 | 1.8 | 完整一步 OOM；forward 与 forward+backward（不含 optimizer）可以 |
| xl | 1998M | 29.8 | 7.4 | 3.7 | 仅 forward |
| 2.7B | 3407M | 50.8 | 12.7 | 6.3 | 仅 forward，且建议 bf16 |

对 3.2 表的影响：
- large 的 optimizer step 预期 OOM，这本身是结果，附上这张表的估算。可以补测"仅 forward+backward、不做 optimizer"，仍能得到 backward 时间。
- xl 和 2.7B 的 forward 必须用 `torch.no_grad()`，否则激活值会撑爆显存。

对后面 Part 3、4 的影响：
- DDP 实验用 **medium**，一定放得下。
- Part 4 优化器分片有个很好的演示机会：large 无分片约 14.4 GiB 静态（OOM 或贴边），分片后每卡约 11 GiB。这要等你实测验证，但值得作为实验设计的一个亮点。

## 8. 预期 vs 实测记录表（加进 3.2）

| 模型 | ctx | 阶段 | FLOPs (TFLOP) | P (TFLOPs/s) | t_pred (ms) | t_meas (ms) | t_meas / t_pred | MFU |
|---|---|---|---|---|---|---|---|---|
| small | 128 | fwd |  | P_fp32 |  |  |  |  |
| small | 128 | bwd |  | P_fp32 |  |  |  |  |
| small | 512 | fwd |  | P_fp32 |  |  |  |  |
| small | 512 | bwd |  | P_fp32 |  |  |  |  |
| medium | 512 | fwd |  | P_fp32 |  |  |  |  |
| medium | 512 | bwd |  | P_fp32 |  |  |  |  |

## 9. 现在的执行顺序

1. 跑 `matmul_peak.py`，填 Step 1 的表。
2. 跑 `count_params.py`，核对第 1 节的参数量（尤其 large、xl、2.7B）。
3. 用 `flops_estimate.py` 或手算，填第 4 节的 FLOPs。
4. 写 `benchmark.py`，先只跑 small、ctx=128，把 t_pred 与 t_meas 的对比贴给我。