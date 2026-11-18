# 含GEMM的各layout设计
下面按**从后往前**推导：先由计算（MMA）定出寄存器里每个线程要拿什么，再反推 s→r 怎么读、smem 怎么放、g→s 怎么搬。数字都用同一个例子：bf16，d = 64，nq = nk = 64，nt = 128。具体 API 名称我按印象写，以 4.8.1 文档为准；

## 0. 总原则

- **消费者决定生产者**：MMA 要什么寄存器布局，就决定 s→r 用哪种 `ldmatrix`；`ldmatrix` 的访问模式决定 smem 怎么排（swizzle）；smem 的行宽决定 g→s 的 16B 向量怎么分给线程。
- **同一套读法**：每个 partition 视图的形状是（一条指令里本线程的值，沿 M 重复，沿 K 重复）。

```
gmem [b,n,d] ──cp.async 16B──▶ smem (swizzled) ──ldmatrix──▶ 寄存器 ──mma──▶ 累加器
 ② 由 ④ 反推              ③ 由 ④ 反推            ① 由计算定
```

## 1. 计算与 MMA 布局

两个 GEMM（每个 batch、每个 Q tile）：

| GEMM | M | N | K | A | B |
|---|---|---|---|---|---|
| S = Q Kᵀ | nq | nk | d | Q（smem） | K（smem，K 方向 = d 方向连续） |
| O = P V | nq | d | nk | P（**寄存器**） | V（smem，N 方向 = d 方向连续） |

**MMA 选择**：SM80 的 `m16n8k16`，bf16 输入，fp32 累加。设 `lane = 4g + t`（g = lane/4 ∈ 0..7，t = lane%4）。每线程持有的值：

| 操作数 | 值 | 位置 |
|---|---|---|
| A（16×16） | 8 个，(2,2,2) | 行 g 与 g+8；列 2t、2t+1，以及 2t+8、2t+9 |
| B（16×8，K×N） | 4 个，(2,2) | k = 2t、2t+1 与 2t+8、2t+9；列 n = g |
| C（16×8） | 4 个，(2,2) | 行 g 与 g+8；列 2t、2t+1 |

**线程如何铺到 MMA 上**：`atom_layout = (nq/16, 1, 1)`，nt = 32·nq/16。

- 这里就是 nq 与 nt 的绑定：nq = 64 → 4 个 warp → nt = 128。
- **只沿 M 切**是刻意的：每个 warp 拥有完整的 16 行 Q，行 softmax 只需要同一个 quad（4 个线程）内归约，不需要跨 warp。
- 想让 nt 小于 32·nq/16，就让每个 warp 管多个 m16 块，代价是累加器寄存器翻倍。

每个 warp 的形状（nq = nk = d = 64）：

| 视图 | 形状 | 每线程元素 |
|---|---|---|
| S 累加器 | `((2,2), MMA_M=1, MMA_N=nk/8=8)` | 32 个 fp32 |
| Q 的 A 片段 | `((2,2,2), 1, MMA_K=d/16=4)` | 32 个 bf16 |
| K 的 B 片段 | `((2,2), 8, 4)` | 128 个 bf16（逐块使用，不会全放寄存器） |
| P（转成 bf16 的 A 片段） | `((2,2,2), 1, MMA_K=nk/16=4)` | 32 个 bf16 |
| V 的 B 片段 | `((2,2), MMA_N=d/8=8, 4)` | 同上 |
| O 累加器 | `((2,2), 1, 8)` | 32 个 fp32 |

## 2. 为什么 P 不经过 smem

第一个 MMA 的 C 片段里，**相邻两个 n8 块**（键 0–7、8–15）拼起来，恰好是第二个 MMA 的 A 片段的一个 k16 块：

- 寄存器对应：`a0a1 = (c0,c1)[块 2j]`，`a2a3 = (c2,c3)[块 2j]`，`a4a5 = (c0,c1)[块 2j+1]`，`a6a7 = (c2,c3)[块 2j+1]`。
- 只需要 fp32→bf16 的转换和打包，所以 S→P 全程在寄存器里。
- 这是第一版以后的优化：第一版可以先写 smem 再 `ldmatrix`，通过测试再改。

## 3. s→r：由 MMA 反推 ldmatrix

`ldmatrix` 读 8×8 的 16 位矩阵，每个线程得到**同一行里连续的 2 个元素**：`row = lane/4`，`col = 2·(lane%4)`——恰好就是上面 A、B、C 的片段布局。

| 操作数 | 读法 | 原因 |
|---|---|---|
| Q（A，16×16） | `ldmatrix.x4`，**不转置** | A 要求同一行连续的 K 对，而 Q 的 d 方向连续 |
| K（B，K×N） | `ldmatrix.x4`，**不转置** | B[k, n] = K[n, k]，线程要 K 的某一行里连续的 d 对 |
| V（B，K×N，K = 键，N = d） | `ldmatrix.x4.trans` | 线程要的是**沿键方向**连续的对，而 V 在 smem 里是 d 方向连续，所以要转置 |

构造：用 `tiled_mma` 去构造 `make_tiled_copy_A/B`，这样 copy 的分区自动与 MMA 的寄存器布局匹配；再用 `retile` 把 MMA 的片段重排成 copy 要的形状（`tXrA = retile(tCrA)`，同一块寄存器）。

## 4. smem 布局：行主序 + swizzle

**问题**：`ldmatrix` 一个 8×8 矩阵由 8 个线程各给一个 16B 行地址。d = 64 的 bf16 行宽 = 128B，**每行的起始 bank 都相同**，8 个地址落在同一组 bank，发生 8 路冲突。

**解法**：对 16B 块号做 XOR：`chunk' = chunk ^ (row % 8)`。8 行就落到 8 个不同的块位置上，无冲突。

| d（bf16） | 每行 16B 块数 | Swizzle |
|---|---|---|
| 64 | 8 | `Swizzle<3,3,3>`（把比特 6–8 异或进比特 3–5） |
| 128 | 16 | `Swizzle<3,3,4>` |
| 32 | 4 | `Swizzle<2,3,2>` |

通式：`B = min(3, log2(d/8))`，`M = 3`（8 个元素 = 16B），`S = log2(d) − 3`。smem tensor = `composed_layout(swizzle, row-major (nq, d):(d,1))`，并保证 16B 对齐。我上一轮写的 d = 32 的参数不对，以这一版为准。验证方法：ncu 看 shared memory 的 bank conflict 计数应当为 0 或接近 0。

## 5. g→s：由 smem 反推 cp.async 的线程布局

约束：

- 每线程 16B（128 位）= V = 8 个 bf16，沿 d 连续；
- 一行 d = 64 → 每行 8 个向量，所以线程沿 d 排 8 列：`thr_n = d/V = 8`；
- 行方向：`thr_m = nt / 8 = 16`。

得到：

```
thr_layout = (16, 8) : (8, 1)       # tid = r·8 + c，相邻线程读相邻 16B → 一行 128B 合并访问
val_layout = (1, 8)
tiler_mn   = (16, 64)               # nq = 64 → REP_M = 4
tQgQ : ((8,1), 4, 1)   元素 (e, m)：row = m·16 + tid//8，col = (tid%8)·8 + e
```

要点：

- **swizzle 不影响 g→s 的线程划分**。它只是置换同一行内 16B 块的存放位置，`partition_D(sQ)` 会通过 composed layout 自动处理地址。
- gmem 端：`local_tile(mQ[b], (nq, d), (i, 0))`，用 `pred` 保护 N 方向越界；K、V 的越界行要补零（否则 `P·V` 里会出现 `0 × NaN`）。
- Q 的 tile 加载一次；K、V 每次循环加载一个 (nk, d) tile，用同样的拷贝布局和同样的 swizzle。

## 6. 完整链路（本例，每个 K/V tile）

| 阶段 | 布局 / 指令 | 形状（每线程） |
|---|---|---|
| gmem | `[b,n,d]`，stride `(n·d, d, 1)` | — |
| g→s | cp.async，thr `(16,8)`，val `(1,8)` | `((8,1),4,1)` |
| smem | `Swizzle<3,3,3>` ∘ row-major | sQ 8 KB，sK、sV 各 8 KB / stage |
| s→r（Q、K） | `ldmatrix.x4`，由 `tiled_mma` 构造 | Q：`((2,2,2),1,4)`；K：`((2,2),8,4)` |
| MMA 1 | `m16n8k16`，S 累加器 | `((2,2),1,8)` = 32 个 fp32 |
| S→P | 寄存器内 fp32→bf16 重排 | `((2,2,2),1,4)` |
| s→r（V） | `ldmatrix.x4.trans` | `((2,2),8,4)` |
| MMA 2 | O 累加器 | `((2,2),1,8)` = 32 个 fp32 |

smem 总量：sQ 8 KB + 2 个 stage 的 sK、sV 共 32 KB = 40 KB，低于你的 99 KB 上限。

## 7. 写回（r→g）

C 片段每线程只持有 2 个连续 bf16（4B），直接写 gmem 的合并度差。做法：O 先写进 smem（复用 sQ，注意 swizzle），再用与 g→s 同样的二维 tile 布局做 s→g，用 `row < N` 的谓词。`L` 每行只由同一个 quad 里 t = 0 的线程写。

## 8. 设计清单

1. 定 `nq`，得 warp 数 = nq/16，`nt = 32·warp 数`。
2. 算出各片段形状和每线程寄存器数（S、O 累加器 + P + Q 片段），确认没有溢出。
3. 对每个操作数判断 ldmatrix 是否需要 `.trans`：看 smem 里连续的方向是不是 MMA 要的 K 方向。
4. 选 swizzle：`B = min(3, log2(d/8))`，`M = 3`，`S = log2(d) − 3`。
5. 推 g→s 的 thr/val 布局：`val = (1, 8)`，`thr_n = d/8`，`thr_m = nt/thr_n`，要求 `nq % thr_m == 0`。
6. 每一步都 `print` 视图形状，核对元素总数：`每线程元素 × 线程数 = tile 元素数`。

