下面重新从 Tensor Core 讲起，一路到 g→s、s→r 的设计。布局事实（寄存器片段、ldmatrix 的元素归属）是我对 PTX 的记忆，API 名字按印象写，请以你的 4.8.1 文档和实测为准；我没有在你的环境里运行过。上一条里我对 C 片段的 `(2,2)` 说得含糊，以本文为准。

## 1. Tensor Core：一条 mma 做什么

`mma.sync.m16n8k16`（bf16 输入、fp32 累加）由**一个 warp 的 32 个线程协作**完成：

```
D[m,n] = Σ_k A[m,k]·B[n,k] + C[m,n]        m∈16, n∈8, k∈16
```

- B 按 (n,k) 看待，也就是 PTX 里说的 "col" 排布。所以 `S = Q·Kᵀ` 里 K（nk×d，d 连续）正好就是 B。
- 操作数不是放在 smem 里算的，而是**必须先在寄存器里**，并且每个线程只能拿到固定的那几个元素。这就是"片段（fragment）"。

记 `lane = 4g + t`，`g = lane/4 ∈ 0..7`，`t = lane%4 ∈ 0..3`。

| 操作数 | 寄存器 | 每线程元素 | 位置 |
|---|---|---|---|
| A (16×16) | 4 个 32 位 | 8 个 bf16 | reg0：(行 g, k 2t..2t+1)；reg1：(行 g+8, k 2t..)；reg2：(行 g, k 2t+8..)；reg3：(行 g+8, k 2t+8..) |
| B (k16 × n8) | 2 个 32 位 | 4 个 bf16 | reg0：(k 2t..2t+1, n g)；reg1：(k 2t+8..2t+9, n g) |
| C/D (16×8) | 4 个 32 位 | 4 个 fp32 | c0,c1：(行 g, 列 2t, 2t+1)；c2,c3：(行 g+8, 列 2t, 2t+1) |

**三个要记住的规律**：

1. 一个 32 位寄存器装 2 个 bf16，是**同一行（B 则是同一个 n）里沿 k 相邻的 2 个元素**。
2. 32 线程 × 1 个寄存器 × 2 个元素 = 64 个元素 = 一个 **8×8 小块**。A 由 4 个这样的块组成，B 由 2 个，C 是 2 个 8×8 的 fp32 块（每块 2 个寄存器）。
3. 所以 CuTe 里 A 的每线程值 mode 是 `(2,2,2)`：寄存器内第几个元素 × 行 +8 × k +8；B 是 `(2,2)`；C 是 `(2,2)`（列相邻 × 行 +8）。

## 2. ldmatrix：专门产出这种寄存器的指令

`ldmatrix.m8n8.x4` 一次从 smem 读 **4 个 8×8 的 16 位矩阵**，直接分发成 MMA 要的寄存器：

- **地址**：线程 0–7 给第 0 个矩阵的 8 行地址，8–15 给第 1 个，16–23 给第 2 个，24–31 给第 3 个。每个地址指向 **16B 连续**的一行（8 个 bf16）。
- **结果**：线程 `l` 对每个矩阵得到一个寄存器。
  - 不转置：该矩阵第 `l/4` 行的第 `2(l%4), 2(l%4)+1` 个元素，也就是 `(g, 2t..2t+1)`。
  - `.trans`：该矩阵第 `2(l%4), 2(l%4)+1` 行的第 `l/4` 列，也就是 `(2t..2t+1, g)`。

用在三种操作数上（smem 里一律是"一行 16B 连续"）：

| 操作数 | smem 里什么连续 | MMA 要的“对”方向 | 变种 | 4 个矩阵怎么取 |
|---|---|---|---|---|
| Q（A） | d = k | k | 不转置 x4 | m0=(行0–7,k0–7)→reg0；m1=(行8–15,k0–7)→reg1；m2=(行0–7,k8–15)→reg2；m3=(行8–15,k8–15)→reg3。线程 l 的地址：行 `l%16`，k 起点 `(l/16)·8` |
| K（B） | d = k | k | 不转置 x4 | 两个 n8 块：m0=(n0–7,k0–7)，m1=(n0–7,k8–15)，m2=(n8–15,k0–7)，m3=(n8–15,k8–15) |
| V（B） | d = n | k（键方向） | **`.trans`** x4 | smem 里行 = 键 k，列 = n；矩阵 = (k0–7, n0–7) 等；转置后线程得到 (k 2t..2t+1, n g) |

**判据只有一句**：寄存器要的“对”方向是 k；smem 里 16B 连续的方向是不是 k？是就不转置，不是就 `.trans`。V 要转置不是因为 MMA 特殊，而是因为它的 k（键）方向在 smem 里不连续。

## 3. GEMM：把一条 mma 铺成 tile

一条 mma 只算 16×8×16。大 GEMM 就是**在 M、N、K 三个方向重复**。CuTe 里这个重复写成 mode：

```
partition_A(sQ) : ((2,2,2), MMA_M, MMA_K)     MMA_M=行方向 m16 块数, MMA_K=k 方向 k16 块数
partition_B(sK) : ((2,2),   MMA_N, MMA_K)     MMA_N = n8 块数
partition_C     : ((2,2),   MMA_M, MMA_N)
```

第 0 个 mode 是**一条指令里本线程的值**，后面的 mode 是**沿该方向重复几条指令**。

**例：S = Q·Kᵀ**（bf16，nq = nk = d = 64，nt = 128 → 4 个 warp）

- `atom_layout = (4,1,1)`：每个 warp 拥有 16 行 Q，所有 warp 共用同一个 K tile，所以 `MMA_M = 1`。
- 形状：A `((2,2,2),1,4)`（32 个 bf16），B `((2,2),8,4)`（128 个 bf16，逐块使用），C `((2,2),1,8)`（32 个 fp32）。
- 计算循环：对 `k = 0..3`：读 A 的第 k 块（1 条 ldmatrix.x4）→ 读 B 的第 k 块（4 条 x4，每条覆盖两个 n8 块）→ 发 8 条 mma（对 8 个 n8 块累加到 C）。共 32 条 mma。
- 校验：每线程 C 元素 32 × 128 线程 = 4096 = nq·nk ✓。

**例：O = P·V**：M = nq，N = d，K = nk。同样 `((2,2,2),1,nk/16)` 的 A、`((2,2), d/8, nk/16)` 的 B（V 用 `.trans`）、`((2,2),1,d/8)` 的 C。P 作为 A，既可以来自 smem，也可以直接由 S 的 C 片段重排得到（相邻两个 n8 块拼成一个 k16 块）。第一版先走 smem，第二版再用寄存器重排。

## 4. 往上游反推：smem 与 g→s

**约束从下往上传**：

```
mma 的寄存器片段（固定）
   ↓ 决定
ldmatrix 变种与 4 个 8×8 的取法（上面第 2 节）
   ↓ 对 smem 提出三个要求
smem 布局：① 每行 16B 连续且 16B 对齐  ② tile 是 MMA tile 的整数倍  ③ 8 个行地址无 bank 冲突
   ↑ 同时被它的写入者约束
g→s：cp.async 每线程一次 16B，沿 gmem 连续方向（d）
```

smem 同时有**两个使用者**：写它的 cp.async（沿 d 每次 16B），读它的 ldmatrix（一次 8 行 × 16B）。设计任务就是找一个布局让两者都高效。

### 4.1 为什么要 swizzle（以 d = 64 为例）

- smem 有 32 个 bank，每个 4B，一个 bank 周期 = 128B。d = 64 的 bf16 一行正好 128B，**每一行都从 bank 0 开始**。
- 一次 `ldmatrix` 的 8 个行地址取的是同一个 16B 列块，每个 16B 占 4 个 bank → 全部落在同样的 4 个 bank 里 → **8 路冲突**。
- 解法：把行内的 16B 块号与行号异或：`块号' = 块号 XOR (行号 % 8)`。8 行取到 8 个不同的块位置，恰好铺满 32 个 bank，无冲突。
- CuTe 写法：`Swizzle<B,M,S>`，作用在以元素为单位的偏移上：把第 `[M+S, M+S+B)` 位异或进第 `[M, M+B)` 位。
  - `M = 3`：8 个元素 = 16B，16B 块内部保持原样；
  - 行号在偏移的第 `log2(d)` 位以上，所以 `S = log2(d) − 3`；
  - `B = min(3, log2(d/8))`：行内 16B 块数 = d/8，最多对 3 位异或。
  - d = 64 → `Swizzle<3,3,3>`；d = 128 → `<3,3,4>`；d = 32 → `<2,3,2>`。
- smem tensor = `composed_layout(swizzle, 行主序 (rows, d):(d,1))`，16B 对齐。

### 4.2 g→s 的线程分工（cp.async）

- 每线程 16B = 8 个 bf16，沿 d 连续；一行 64 个元素 = 8 个向量 → 线程沿 d 排 8 列，其余线程排行：
  `thr_layout = (nt/8, 8) : (8, 1)`，`val_layout = (1, 8)`。nt = 128 时 `thr_layout = (16, 8)`。
- 相邻线程读相邻 16B，8 个线程合并读完一行 128B，gmem 访问合并。
- `tQgQ : ((8,1), nq/16, 1)`，nq = 64 时 `((8,1),4,1)`。
- **swizzle 不改变线程划分**：它只置换同一行内的 16B 块（`M=3` 保证 cp.async 的 16B 单位不被打散），`partition_D(sQ)` 会通过 composed layout 自动算出每个线程该写的 smem 地址。
- 越界行用 `pred`，形状 `(REP_V, REP_M, REP_K)`；K、V 的越界行补零。

### 4.3 s→r 的 copy（用 tiled_mma 派生）

```python
# 下面的 API 名称按印象写，以你的 4.8.1 文档为准
tiled_mma = cute.make_tiled_mma(mma_op, atom_layout)           # 4 个 warp 沿 M
thr_mma   = tiled_mma.get_slice(tidx)
tCrQ = tiled_mma.make_fragment_A(thr_mma.partition_A(sQ))      # 寄存器片段 ((2,2,2),1,4)

# ldmatrix 的 copy：用 tiled_mma 构造，线程—值布局自动与 MMA 的寄存器布局匹配
copy_Q  = cute.make_tiled_copy_A(ldmatrix_atom_N, tiled_mma)   # 不转置
thr_cpy = copy_Q.get_slice(tidx)
tXsQ = thr_cpy.partition_S(sQ)        # smem 视图（含 swizzle）
tXrQ = thr_cpy.retile(tCrQ)           # 同一块寄存器，换成 copy 要的 mode 标签
cute.copy(copy_Q, tXsQ[None, None, k], tXrQ[None, None, k])
# V 用 .trans 版本的 atom；K 用 make_tiled_copy_B
```

这一步在**寄存器一侧没有自由度**：copy 的线程—值布局由 `tiled_mma` 导出，`retile` 只换标签、不搬数据。**真正的设计空间只有 smem 的布局（swizzle）和 g→s 的线程分工**。

## 5. 设计顺序

1. 把每个 GEMM 写成 `D[m,n] = Σ_k A[m,k]·B[n,k]`，找出每个操作数的 **k 方向**。
2. MMA 片段固定：每线程拿 k 方向连续的 2 个元素；A、B、C 的值 mode 是 `(2,2,2)`、`(2,2)`、`(2,2)`。
3. 看 smem 里该操作数 16B 连续的方向：是 k → ldmatrix 不转置；否则 `.trans`。
4. 选 `atom_layout`，算出每个片段的形状和寄存器数，确认无溢出。
5. 按 ldmatrix 的访问模式选 swizzle（`M=3`，`S=log2(d)−3`，`B=min(3,log2(d/8))`）。
6. 由 d 和 16B 推 g→s 的 thr/val 布局。
7. 每一步 `print` 形状，核对"每线程元素数 × 线程数 = tile 元素数"。

## 6. 一个值得先做的验证

在 3b 里让 `sQ[i][j] = 16·i + j`（16×16，先不加 swizzle），用 `ldmatrix.x4` 读到寄存器，打印线程 (g,t) 的 8 个值。应当依次是 `(g,2t),(g,2t+1)`、`(g+8,2t),(g+8,2t+1)`、`(g,2t+8),(g,2t+9)`、`(g+8,2t+8),(g+8,2t+9)` 对应的 `16·行 + 列`。如果顺序与此不同，以实测为准，告诉我，我来改上面的表。之后再加 swizzle，用 ncu 看 shared memory bank conflict 是否降到 0 或接近 0。