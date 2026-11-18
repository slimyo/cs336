# CuTe DSL：Copy Partition、线程映射与 Predication 完整解析

> 核心问题：`partition_S` 到底做了什么？它为什么会产生 `((8,1),2,1)` 这样的嵌套 Shape？每个 mode 分别对应什么？又为什么 `pred` 的形状与数据 Tensor 不一样？
>
> 本文以一个具体的 Global → Shared 向量化 Copy 为例，从命名体系、CTA 线程层级、TiledCopy 构造到逐 mode 推导，建立完整的数据映射模型。

**阅读路线**

```text
CuTe 命名体系
      ↓
CTA → TiledCopy → Thr
      ↓
Copy Atom + ThrLayout + ValLayout
      ↓
partition_S / partition_D
      ↓
Thread-Value → Instruction-Repeat
      ↓
pred 生成与越界处理
      ↓
cute.copy 执行
      ↓
Shape / Layout / Stride 排错
```

---

# Part I. CuTe DSL 命名体系

## 0. 先学会读变量名

在阅读 CuTe DSL Kernel 之前，先建立一个认识：

> CuTe 的变量命名通常不是随意的缩写，而是通过前缀、内存空间标记、数据角色和维度后缀，描述一个 Tensor 在计算流程中的身份。

NVIDIA 官方将这套命名风格称为 **Hungarian-style naming conventions（匈牙利式命名约定）**。

官方文档的定位是 lookup reference：帮助读者查阅示例代码，而不是强制新代码遵循的命名规范。

官方文档：[CuTe DSL Naming Conventions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/naming_conventions.html)

### 0.1 三个层次的命名约定

以下三层划分是本文为了教学而整理的框架，并非官方文档本身的章节结构。

| 层次 | 主要内容 | 典型例子 |
|---|---|---|
| GPU 通用约定 | 描述数据所在的内存空间 | gmem、smem、rmem、tmem |
| CUTLASS / CuTe 官方约定 | 描述 Tensor 角色、分区方式和数据流 | tAgA、tAsA、thr_mma |
| Kernel 局部命名 | 根据当前算法定义数据对象 | gTile、sTile、tSgS |

三者不能混为一谈。

特别注意：

**变量名是语义标签，不是 CuTe 类型系统的一部分。**

真正决定数据如何分布的是：

- Shape：Tensor 有哪些维度，各维大小是多少。
- Layout：逻辑坐标如何映射到物理地址。
- Stride：各维度在地址空间中的步长。
- TiledCopy：线程和元素如何组合成整体 Copy 映射。

### 0.2 内存空间：先识别数据在哪里

CuTe DSL 经常使用以下字母表示数据所在的内存空间：

| 标记 | 英文 | 含义 | 常见对象 |
|---|---|---|---|
| g | Global Memory | 全局显存 | gA、gB、gC |
| s | Shared Memory | CTA 内共享内存 | sA、sB |
| r | Register Memory | 线程私有寄存器 | rA、rAcc |
| t | Tensor Memory | Tensor Memory | tAcc |

典型数据流：

```text
GMEM → SMEM → RMEM → Compute
                    ↓
                   RMEM → GMEM
```

这只是常见的数据流，并不是每个 Kernel 都必须经过所有内存空间。

**容易混淆的两个 t：**

- `tAgA` 开头的 `t`：表示 per-thread partitioned view。
- `tCtAcc` 中间的 `t`：表示 Tensor Memory。

必须结合变量名中的位置和后缀判断含义。

### 0.3 官方复合变量名：从左到右拆解

官方给出的复合命名模板为：

```text
[t|b][FamilyPrefix]_[g|s|r|t][Operand/Role][AxisSuffix?]
```

其中：

| 部分 | 含义 |
|---|---|
| t / b | per-thread / block-threadblock partition context |
| FamilyPrefix | 操作数家族或数据移动方向 |
| g / s / r / t | 内存空间 |
| Operand / Role | 数据角色 |
| AxisSuffix | Tensor 视图的轴顺序 |

**下划线规则非常重要：**

- 单字母家族：通常不加下划线，例如 `tAgA`。
- 双字母方向家族：保留下划线，例如 `tTR_gC`。

家族前缀主要分成两组：

| 类型 | 家族前缀 | 含义 |
|---|---|---|
| GEMM 操作数 | A、B、C | GEMM 的输入和输出 |
| Attention 操作数 | Q、K、V、P、O | Attention 的 Query、Key、Value、Probability、Output |
| 数据移动方向 | TR | TMEM → Register |
| 数据移动方向 | RS | Register → Shared |
| 数据移动方向 | SG | Shared → Global |
| 数据移动方向 | GS | Global → Shared |
| TMEM Copy | TM | TMEM tiled-copy partition |
| 便捷别名 | R2G、S2R、T2R、R2T | 对应特定数据移动路径 |

典型例子：

| 变量 | 拆解 | 含义 |
|---|---|---|
| tAgA | t + A + gA | A 路径中的线程级 Global Tensor 分区 |
| tAsA | t + A + sA | A 路径中的线程级 Shared Tensor 分区 |
| tCrA | t + C + rA | 计算路径中的 A 寄存器视图 |
| tCtAcc | t + C + tAcc | 计算路径中的 TMEM 累加器视图 |
| tTR_tAcc | t + TR + tAcc | TMEM → Register 路径中的累加器源 |
| tTR_rAcc | t + TR + rAcc | TMEM → Register 路径中的寄存器目的 |
| bSG_sC | b + SG + sC | Shared → Global 的 Shared 源 |
| bSG_gC | b + SG + gC | Shared → Global 的 Global 目的 |
| tTMrO | t + TM + rO | TMEM tiled-copy 的寄存器侧视图 |

这里的 `A/B/C/Q/K/V/P/O` 是操作数家族；`TR/RS/SG/GS/TM` 是数据移动或访问家族。

### 0.4 thr_*：线程级分区器

另一个非常常见的命名系列是 `thr_*`。

它通常表示从整体 Tiled 操作中取得的线程级分区器。

| 命名 | 含义 |
|---|---|
| thr_mma | MMA 线程分区器 |
| thr_copy_g2s | Global → Shared Copy |
| thr_copy_s2r | Shared → Register Copy |
| thr_copy_r2s | Register → Shared Copy |
| thr_copy_t2r | TMEM → Register Copy |
| thr_copy_r2t | Register → TMEM Copy |
| thr_copy_s2t | Shared → TMEM Copy |
| thr_copy_sfa / sfb | Scale Factor 相关 Copy |
| thr_copy_load | Load 路径 |
| thr_copy_beta_g2s | Beta 向量的 Global → Shared 路径 |
| thr_tmem_load | TMEM Load 分区器 |
| thr_tmem_store | TMEM Store 分区器 |

复杂 Kernel 还会出现：

```text
thr_mma_qk
thr_mma_pv
thr_mma_kv
thr_mma_qkv
thr_mma_intra1
thr_mma_intra2
```

用于区分不同矩阵乘法阶段或 CTA 内部计算角色。

基本使用形式：

```python
thr_copy = tiled_copy.get_slice(tidx)

tSrcG = thr_copy.partition_S(gA)
tDstS = thr_copy.partition_D(sA)
```

这里：

- `tiled_copy`：整体线程块级 Copy 配置。
- `thr_copy`：当前线程对应的分区器。
- `partition_S()`：获取源 Tensor 的线程局部视图。
- `partition_D()`：获取目的 Tensor 的线程局部视图。

`tSrcG`、`tDstS` 是本文为避免混淆而采用的局部变量名，并非官方固定名称。

官方文档中的 `tAgA`、`tAsA` 也可以出现在普通 TiledCopy 的线程分区示例中，例如 `sgemm_2.cu`。它们不是 TMA 专属的 API。

### 0.5 Copy Atom：用数据流方向命名

常见形式：

```text
<src>2<dst>_atom
```

| 命名 | 数据方向 |
|---|---|
| s2r_atom | Shared → Register |
| r2s_atom | Register → Shared |
| t2r_atom | Tensor Memory → Register |
| r2t_atom | Register → Tensor Memory |
| s2s_atom | Shared → Shared |
| s2t | Shared → Tensor Memory |
| simt_atom | 特定 SIMT Copy 路径，官方示例用于 TMA Store 不启用时的 Register → Global |

其中：

- `Copy_Atom`：描述一次具体的 Copy 操作及其约束。
- `TiledCopy`：将 Copy Atom 与线程布局、值布局组合，形成线程块级搬运方案。
- `ThrCopy`：从 TiledCopy 中选取一个线程的视图。

注意：`g2s`、`g2r`、`r2g` 等也符合方向命名的逻辑，但具体是否以相应的 Atom 名称暴露，要看 DSL / CUTLASS 实际提供的实现，不能仅凭命名规律推断 API 存在。

### 0.6 轴序后缀：不是提示，而是视图声明

官方命名约定：

> Suffix encodes axis order of the view.

即后缀编码的是 Tensor 视图的轴顺序。

GEMM 常见后缀：

| 后缀 | 轴顺序 |
|---|---|
| _mn | (M, N) |
| _mk | (M, K) |
| _nk | (N, K) |
| _mnl | (M, N, L) |
| _nkl | (N, K, L) |
| _mkl | (M, K, L) |
| _mkl | M、K、L 的三维视图 |

例如：

```python
gB_nkl
gC_mnl
```

表示：

- `gB_nkl`：B 的视图按 (N, K, L) 排列。
- `gC_mnl`：C 的视图按 (M, N, L) 排列。

Attention 常见后缀：

| 后缀 | 轴顺序 |
|---|---|
| _qdl | (SeqQ, HeadDim, Batch) |
| _kdl | (SeqK, HeadDim, Batch) |
| _dkl | (HeadDim, SeqK, Batch) |

例如：

```python
mQ_qdl
mK_kdl
mV_dkl
```

其中 `mV_dkl` 的 D 维放在最前面，是为了适应 P·V 第二次矩阵乘法的布局需求。

需要区分两个概念：

- 轴序后缀：声明视图的逻辑轴顺序。
- Layout / Stride：描述逻辑坐标如何映射到实际存储位置。

所以 `_nkl` 不只是提示，但它也不意味着固定的行主序或列主序。

CuTe 内部还可能出现：

```text
MMA_M / MMA_N / MMA_K
EPI_M / EPI_N
RestM / RestN / RestK / RestL
STAGE
```

这些通常是 Layout 内部的分组 mode，更多属于实现细节。

### 0.7 官方命名与本文局部命名的区别

本文后面使用：

```python
gTile
sTile
cTile
tSgS
tSsS
tScS
tDgD
```

这些变量是为了说明当前 Copy 的源、目的和坐标关系而定义的局部命名。

特别注意：

官方家族字母表中没有 S = Source、D = Destination 这种标准家族定义。

本文：

- `S` 表示 Source。
- `D` 表示 Destination。
- `c` 表示 coordinate。

这与官方 Attention 命名中 `P` 表示 softmax probability / score matrix 的约定不同。

因此，不要把 `tSgS` 误认为官方标准命名。

**阅读原则：命名帮助建立直觉，Shape 和 Layout 决定实际的数据分布。**

### 0.8 官方代码阅读索引

| 研究对象 | 官方代码位置 |
|---|---|
| tAgA、tAsA、tBgB、tBsB | `CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py` |
| tCtAcc、tTR_tAcc、tTR_rAcc、tTR_gC | 同上 |
| tRS_rC、tRS_sC | `CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/mixed_input_gemm.py` |
| bSG_sC、bSG_gC | `CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py` |
| tQsQ、mQ_qdl、mK_kdl、mV_dkl | `CuTeDSL/cute/hopper/kernel/attention/fmha.py` |
| tOtO、tOrO、tPrP | `CuTeDSL/cute/blackwell/tutorial/tutorial_fmha/fmha_0.py` |
| tR2G_rO_src、tR2G_rO_dst | `CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py` |
| s2r_atom_* | `CuTeDSL/cute/blackwell/kernel/attention/mamba2_ssd/mamba2_ssd.py` |
| bGS_gC、bGS_sC | `CuTeDSL/cute/blackwell/efc/common_efc.py` |

官方代码库：[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)

---

# Part II. CTA → TiledCopy → Thr：先理解线程分工

## 1. 整体层级

理解 `partition_S` 之前，先弄清楚 Copy 是在哪个层级定义和执行的。

CuTe 的 TiledCopy 可以理解为一个 CTA 级的整体搬运方案。

```text
GPU Kernel
│
├── Grid
│   │
│   ├── CTA 0
│   │   ├── 线程 0
│   │   ├── 线程 1
│   │   ├── ...
│   │   └── 线程 255
│   │
│   ├── CTA 1
│   │   └── ...
│   │
│   └── CTA N
│
└── 每个 CTA 使用相同的 TiledCopy 规则
```

三个核心层次：

| 层次 | 对象 | 负责什么 |
|---|---|---|
| CTA | 一个线程块 | 负责当前 Tile 的整体计算和搬运 |
| TiledCopy | 整体 Copy 映射 | 决定线程如何分工、每个线程搬哪些元素 |
| ThrCopy | 单线程分区器 | 从整体映射中提取当前线程的工作视图 |

注意：

TiledCopy 不是 CTA 本身，而是描述 CTA 内部线程如何执行 Copy 的映射方案。

## 2. CTA：确定当前处理哪一块数据

假设输入矩阵为：

```text
G : (N, D)
```

CTA 通过自己的 block index 确定当前负责的数据块。

例如：

```python
cta_coord = cute.make_coord(blockIdx.x, blockIdx.y)

gTile = cute.local_tile(
    gInput,
    tiler_mn,
    cta_coord
)
```

这里：

- `gInput`：整个 Global Memory Tensor。
- `cta_coord`：当前 CTA 对应的 Tile 坐标。
- `tiler_mn`：每次处理的 Tile 大小。
- `gTile`：当前 CTA 实际负责的 Global Memory 子区域。

可以理解为：

```text
完整矩阵
┌──────────────────────────┐
│ CTA 0      │ CTA 1       │
│            │             │
├────────────┼─────────────┤
│ CTA 2      │ CTA 3       │
│            │             │
└──────────────────────────┘

每个 CTA 只处理自己的 Tile
```

## 3. TiledCopy：确定 CTA 内部如何分工

TiledCopy 主要由三个部分构成：

```text
TiledCopy
│
├── Copy Atom
│     一条 Copy 指令及其数据宽度
│
├── ThrLayout
│     CTA 中线程如何排列
│
└── ValLayout
      每个线程负责哪些元素
```

对应 API：

```python
tiled_copy = cute.make_tiled_copy(
    copy_atom,
    thr_layout,
    val_layout
)
```

三个部分的作用：

| 参数 | 作用 | 例子 |
|---|---|---|
| copy_atom | 定义搬运操作 | 128-bit Copy |
| thr_layout | 定义线程映射 | (32, 8) |
| val_layout | 定义每线程的元素分布 | (1, 8) |

可以把它理解为：

**Copy Atom 决定一次搬多少，ThrLayout 决定谁来搬，ValLayout 决定每个线程负责哪些值。**

## 4. ThrCopy：从整体分工中提取一个线程

```python
tidx = cute.arch.thread_idx()

thr = tiled_copy.get_slice(tidx)
```

这里的 `tidx` 是当前线程在 CTA 内的线性线程索引。

例如：

```text
CTA 内共 256 个线程

tidx = 0
tidx = 1
...
tidx = 255
```

每个线程执行相同的 Kernel 代码，但 `get_slice(tidx)` 会根据当前线程索引得到不同的分区器。

```python
thr0 = tiled_copy.get_slice(0)
thr1 = tiled_copy.get_slice(1)
thr2 = tiled_copy.get_slice(2)
```

三个分区器对应不同线程的工作。

因此：

```python
thr = tiled_copy.get_slice(tidx)
```

并不是复制数据，而是选择当前线程在整体 Copy 映射中的位置。

## 5. 三层关系总结

| 对象 | 作用 | 是否直接搬数据 |
|---|---|---|
| CTA | 确定当前处理的数据块 | 否 |
| TiledCopy | 定义线程块级搬运规则 | 否 |
| ThrCopy | 取得当前线程的映射视图 | 否 |
| partition_S / partition_D | 取得源 / 目的 Tensor 的线程局部视图 | 否 |
| cute.copy | 真正执行数据搬运 | 是 |

这就是后面所有 Shape 推导的基础。

---

# Part III. partition_S：从整体 Tile 到单线程工作视图

## 6. 例子设定

我们构造一个具体的 Global → Shared Copy。

| 项目 | 值 |
|---|---|
| tile | `gTile : (64, 64)` |
| 内存 | Global Memory |
| 数据类型 | bf16 |
| Atom | 128-bit `cp.async` |
| 每个 Atom 搬运元素 | `V = 128 / 16 = 8` |
| thr_layout | `(32, 8) : (8, 1)` |
| val_layout | `(1, 8)` |
| tiler_mn | `(32×1, 8×8) = (32, 64)` |

其中：

```text
bf16 = 16 bits

128-bit / 16 bits = 8 elements
```

所以一条 128-bit Copy 指令对应 8 个 bf16 元素。

注意：这里假设 Atom、源地址对齐和目标布局满足 128-bit 向量化访问要求。`V=8` 是当前例子的指令宽度，不是所有 Copy 的固定值。

## 7. 首先理解 ThrLayout

线程布局：

```text
thr_layout = (32, 8) : (8, 1)
```

可以理解为：

```text
线程二维坐标：(r, c)

r = 0 ... 31
c = 0 ... 7

tidx = r × 8 + c
```

例如：

| tidx | r | c |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 7 | 0 | 7 |
| 8 | 1 | 0 |
| 9 | 1 | 1 |
| 255 | 31 | 7 |

每一行有 8 个线程，一共有 32 行线程。

## 8. 再理解 ValLayout

```text
val_layout = (1, 8)
```

表示每个线程在 Tile 中负责：

```text
1 行 × 8 列
```

也就是：

```text
线程 (r,c)

row = r
col = c × 8 + v

v = 0 ... 7
```

例如：

```text
线程 (0,0) → row 0, col 0~7
线程 (0,1) → row 0, col 8~15
线程 (1,0) → row 1, col 0~7
```

一个线程负责 8 个连续的 bf16 元素。

## 9. 推导 Tiler

线程布局覆盖：

```text
32 × 8 个线程
```

每个线程负责：

```text
1 × 8 个元素
```

所以：

```text
Tiler_M = 32 × 1 = 32
Tiler_N =  8 × 8 = 64
```

最终：

```python
tiler_mn = (32, 64)
```

一个 CTA 的一次完整 Copy 覆盖：

```text
32 × 64 = 2048 elements
```

而原始 Tile 为：

```text
64 × 64 = 4096 elements
```

所以需要在 M 方向重复两次：

```text
Rest = (2, 1)
```

即：

```text
(64,64)
    ↓
((32,64), (2,1))
```

可以理解为：

```text
┌──────────────────────────────┐
│          REP_M = 0           │
│          (32,64)             │
├──────────────────────────────┤
│          REP_M = 1           │
│          (32,64)             │
└──────────────────────────────┘
```

## 10. partition_S 究竟做了什么？

先看核心代码：

```python
thr = tiled_copy.get_slice(tidx)

tSgS = thr.partition_S(gTile)
```

这里最重要的理解是：

> `partition_S` 不会复制或修改数据，也不会重新决定哪些元素属于哪个线程。它根据 TiledCopy 已经定义好的线程—值映射，把源 Tensor 重新组织成当前线程的局部工作视图。

可以将其理解为重新编目：

```text
原始 Tile
    ↓
TiledCopy 的 TV 映射
    ↓
固定当前线程 tidx
    ↓
按照 Copy Atom 的指令宽度组织元素
    ↓
得到当前线程的源 Tensor 视图
```

这里的变化是 Tensor 的视图 Shape 和索引方式，而不是底层数据本身。

## 11. 四步推导 partition_S

### Step 1：按 Tiler 切块

```text
gTile : (64,64)

tiler_mn = (32,64)
```

得到：

```text
Tile = (32,64)
Rest = (2,1)
```

### Step 2：用 Thread-Value Layout 重解释 Tile

线程布局和元素布局组合：

```text
Thread = (32,8)
Value  = (1,8)
```

得到：

```text
Tile = (32,64)
```

其中：

```text
row = tidx // 8

col = (tidx % 8) × 8 + v
```

### Step 3：固定线程号 tidx

固定某一个线程后，线程 mode 被消掉。

原来的：

```text
(thread, value)
```

变成：

```text
(value, Rest)
```

因此结果中不再保留完整线程维。

### Step 4：把值 mode 按 Atom 分组

当前线程有 8 个值：

```text
V = 8
```

一条 128-bit Copy 指令正好搬运 8 个 bf16：

```text
V = 8
REP_V = 1
```

因此值 mode 可以理解为：

```text
(V, REP_V) = (8,1)
```

最终得到：

```text
tSgS : ((V, REP_V), REP_M, REP_K)

     = ((8,1), 2, 1)
```

这里的 Shape 结构是依据本文的布局和预期的分区结果推导的，具体打印时可能因 DSL 版本、Layout 组合和静态 mode 简化而呈现不同的嵌套形式。

---

# Part IV. partition_S 的每个 mode

## 12. 各 mode 含义

| mode | 名称 | 本例值 | 含义 |
|---|---|---:|---|
| shape[0][0] | V | 8 | 一条 Copy Atom 搬运的元素数 |
| shape[0][1] | REP_V | 1 | 当前线程的值方向需要重复多少条 Atom |
| shape[1] | REP_M | 2 | M 方向 Tile 重复次数 |
| shape[2] | REP_K | 1 | K/N 方向 Tile 重复次数 |

注意：本文把第二个矩阵维称为 K，是沿用 Copy / GEMM 中的常见维度符号。当前例子实际是二维矩阵的列方向 N/D。

公式：

```text
V = bits_per_copy / bits(dtype)

REP_V = 每线程值元素数 / V

REP_M = Rest_M

REP_K = Rest_K
```

在本例中：

```text
V     = 128 / 16 = 8

REP_V = 8 / 8 = 1

REP_M = 64 / 32 = 2

REP_K = 64 / 64 = 1
```

因此：

```text
((8,1), 2, 1)
```

### 12.1 元素数量自检

每线程元素数：

```text
8 × 1 × 2 × 1 = 16
```

CTA 总线程数：

```text
32 × 8 = 256
```

CTA 总搬运元素数：

```text
256 × 16 = 4096
```

刚好等于：

```text
64 × 64 = 4096
```

因此：

```text
线程总数 × 每线程元素数 = Tile 元素数
```

这是检查线程布局和 Tiler 是否匹配的一个重要方法。

注意：这个乘积检查只能验证数量关系，不能证明线程访问连续、无冲突或满足向量化对齐。

## 13. 1D 情形

如果处理的是一个一维向量，可能得到：

```text
((8,1), ((1)))
```

其中：

- `(8,1)`：Atom 内元素数和指令重复数。
- `((1))`：剩余维度只有一个元素，没有额外重复。

因此，不能把 1D 和 2D 的 Shape 结构机械地混用。

尤其在构造 `pred` 时，应以实际打印出的 mode 为准。

---

# Part V. 具体到线程：数据到底怎么分

## 14. 线程数据映射

因为：

```text
REP_M = 2
```

所以每个线程在 M 方向有两个向量。

| 线程 | r,c | REP_M=0 | REP_M=1 |
|---|---|---|---|
| 0 | 0,0 | 行0，列0–7 | 行32，列0–7 |
| 9 | 1,1 | 行1，列8–15 | 行33，列8–15 |
| 255 | 31,7 | 行31，列56–63 | 行63，列56–63 |

因此：

```text
tSgS[(e,rv),m,k]
```

对应：

```text
row = m × 32 + r

col = k × 64 + c × 8 + e
```

其中：

```text
e = 0 ... 7
rv = 0
m = 0 ... 1
k = 0
```

## 15. 为什么访问是合并的？

观察同一行线程：

```text
线程 0：列  0~7
线程 1：列  8~15
线程 2：列 16~23
...
线程 7：列 56~63
```

每个线程访问 16B：

```text
8 × 2 bytes = 16 bytes
```

相邻线程访问相邻的 16B。

因此，同一行的 8 个线程正好覆盖：

```text
8 × 16B = 128B
```

这是一个连续的 128B 区域，有利于全局内存访问合并。

注意：这里描述的是逻辑访问连续性；实际生成的访存事务还受到硬件事务粒度、对齐和具体指令的影响。

## 16. Stride 的含义

在当前例子中：

值方向：

```text
stride_V = 1 element
```

REP_M 方向：

```text
stride_REP_M = 32 × 64 = 2048 elements
```

因此：

```text
一个 REP_M 的跳转
= 32 行 × 每行 64 个元素
= 2048 个 bf16
```

如果换算成字节：

```text
2048 × 2 = 4096 bytes
```

---

# Part VI. 同一个 TiledCopy 切出的四个视图

## 17. 四种视图

| 视图 | 来源 | 形状 | 元素 | 用途 |
|---|---|---|---|---|
| tSgS | thr.partition_S(gTile) | ((8,1),2,1) | bf16 数据 | Copy 源 |
| tSsS | thr.partition_D(sTile) | ((8,1),2,1) | bf16 数据 | Copy 目的 |
| tScS | thr.partition_S(cTile) | ((8,1),2,1) | (row,col) 坐标 | 计算谓词 |
| tDgD | thr.partition_D(gDst) | ((8,1),2,1) | bf16 数据 | 写回目的 |

这里所有变量名都是本文的局部命名。

### 17.1 S 和 D 的区别

```python
tSgS = thr.partition_S(gTile)

tSsS = thr.partition_D(sTile)
```

- `partition_S`：使用 TiledCopy 的源端映射。
- `partition_D`：使用 TiledCopy 的目的端映射。

对于当前 G → S Copy，源和目的布局匹配，因此预期得到相同的逻辑 Shape。

但这并不是所有 Copy 的必然规律：某些指令要求源和目的采用不同的分区布局。

### 17.2 坐标张量为什么能用于谓词？

构造 Identity Tensor：

```python
cTile = cute.local_tile(
    identity_tensor,
    tiler_mn,
    cta_coord
)

tScS = thr.partition_S(cTile)
```

Identity Tensor 不保存实际 bf16 数据，而是保存逻辑坐标：

```text
(0,0), (0,1), (0,2), ...
(1,0), (1,1), (1,2), ...
```

由于坐标张量与数据张量经过相同的 Tile 和线程映射：

```text
tScS[i] 与 tSgS[i]
```

对应同一个逻辑元素位置。

因此，可以通过 `tScS` 获取 `tSgS` 中对应元素的行列坐标。

---

# Part VII. Pred：为什么形状与数据 Tensor 不一样？

## 18. 核心原因：一个谓词保护一条指令

在当前例子中：

```text
V = 8
```

一条 128-bit Copy 指令搬运 8 个 bf16。

假设向量中的 8 个元素属于同一行，那么判断这一行是否有效，只需要一个谓词。

因此：

```text
数据 Tensor 的第一层：

(V, REP_V)
```

而谓词不需要保护每一个元素，只需要保护每条指令：

```text
Pred 的第一层：

REP_V
```

最终：

```text
Data Shape = ((8,1),2,1)

Pred Shape = (1,2,1)
```

也就是：

```text
(V, REP_V, REP_M, REP_K)

          ↓ 去掉 V

(REP_V, REP_M, REP_K)
```

这是本例中的预期逻辑结构。实际 DSL 对嵌套 mode 的打印和接受形式仍应以运行结果为准。

## 19. 为什么 make_fragment_like 不适合直接创建 Pred？

如果写：

```python
pred = cute.make_fragment_like(tSgS)
```

那么它会按照数据 Tensor 的 Shape 创建 Fragment：

```text
((8,1),2,1)
```

但当前 Copy 的 pred 需要：

```text
(1,2,1)
```

因此，数据 Tensor 与谓词 Tensor 的 Shape 不匹配。

原先的报错：

```text
expects ... (1,((1)))
```

可以理解为：当前 Copy 的接口期望的是指令级谓词布局，而不是完整元素级数据布局。

需要强调：报错中的嵌套括号是具体 Layout / Shape 表示，不能只凭一个错误字符串推断所有版本的完整模式规则。

## 20. 构造 Pred

按照本文的 2D 逻辑结构：

```python
n_rep = tSgS.shape[0][1]

n_m = cute.size(tSgS, mode=[1])
n_k = cute.size(tSgS, mode=[2])

pred = cute.make_rmem_tensor(
    (n_rep, n_m, n_k),
    cutlass.Boolean
)
```

逐个遍历：

```python
for rv in range(n_rep):
    for m in range(n_m):
        for k in range(n_k):

            row = tScS[(0, rv), m, k][0]

            pred[rv, m, k] = row < N
```

这里：

```python
tScS[(0, rv), m, k]
```

取的是当前向量的第一个元素对应的坐标。

因为本例每个向量的 8 个元素都在同一行，所以只检查第一个元素的行坐标即可。

注意：以上代码是基于当前 Shape 假设的推导代码，`shape` 索引、静态循环和坐标提取方式需要根据你实际使用的 CuTe DSL 版本核对。

## 21. N=100 时的例子

假设：

```text
N = 100

当前 gTile 对应全局行 64~127
```

局部 Tile 的两个 REP_M：

```text
REP_M = 0：全局行 64~95

REP_M = 1：全局行 96~127
```

因此：

| REP_M | 线程行 r | 全局行 | Pred |
|---|---|---|---|
| 0 | 0–31 | 64–95 | 全部 True |
| 1 | 0–3 | 96–99 | True |
| 1 | 4–31 | 100–127 | False |

所以第二个向量只有前 4 行有效。

### 21.1 如果还要判断列越界

如果 D 不是向量宽度的整数倍，还要判断：

```python
col < D
```

但要注意：

如果一个 16B 向量跨越了有效边界，不能简单地使用一个代表整条向量的谓词。

常见解决方式：

1. 让 D 满足 8 个 bf16 的倍数要求。
2. 将尾部元素单独处理。
3. 使用支持相应粒度谓词的其他 Copy 路径。

---

# Part VIII. Pred 如何参与 Copy

## 22. 执行 Copy

```python
cute.copy(
    tiled_copy,
    tSgS,
    tSsS,
    pred=pred
)
```

逻辑上：

```text
遍历所有 (REP_V, REP_M, REP_K)
             ↓
         检查 Pred
             ↓
       ┌─────┴─────┐
       │           │
     True        False
       │           │
   执行 Atom     不执行 Atom
       │           │
   搬运 V 个值   目的位置不更新
```

这里有一个非常重要的细节：

**谓词为 False，并不意味着目的 Shared Memory 自动变成 0。**

它意味着对应的 Copy 操作不执行，原来的 Shared Memory 内容保持不变。

如果后续计算需要补零，必须显式处理无效元素。

## 23. 越界数据补零

例如：

```python
if row >= N:
    for e in range(8):
        tSsS[(e, rv), m, k] = cutlass.BFloat16(0.0)
```

需要注意：

- 补零必须发生在后续消费者读取之前。
- 对应的 Shared Memory 位置必须已经正确初始化。
- 如果使用异步 Copy，还必须正确处理异步完成和同步问题。

为什么要补零？

在矩阵乘法中：

```text
P × V
```

如果越界位置没有初始化，可能读取到旧数据或未定义数据。

特别是：

```text
0 × NaN = NaN
```

不能依赖乘法中的零系数自动消除无效数据。

---

# Part IX. 进一步优化：谓词广播

## 24. Pred 是否一定需要三维？

当前谓词只依赖行号：

```text
pred = row < N
```

因此，它与列方向 K 无关。

如果：

```text
pred[rv,m,k] = pred[rv,m]
```

那么 K 方向实际上是重复的。

理论上可以使用 stride 为 0 的 Layout 广播：

```text
stride = (stride_rep, stride_m, 0)
```

这样可以减少重复的谓词存储。

但需要注意：

- 这是 Layout 广播优化，不代表 Copy 接口允许任意 Shape。
- 需要满足当前 DSL 的 pred Layout 约束。
- 应先验证基本版本，再优化广播。

建议当前阶段先使用显式三维 Pred，确保正确性后再做压缩。

---

# Part X. 排错与核对

## 25. 打印 Shape 与 Layout

建议首先打印：

```python
print(tiled_copy.tiler_mn)

print(tiled_copy.layout_src_tv_tiled)

print(tSgS)
print(tSsS)
print(tScS)
```

同时打印：

```python
print(tSgS.shape[0])
print(cute.size(tSgS, mode=[1]))
print(cute.size(tSgS, mode=[2]))
```

注意：不同 CuTe DSL 版本中属性名、打印格式和 Shape 的嵌套方式可能不同，不能假设所有版本都完全一致。

## 26. 常见问题

| 现象 | 可能原因 |
|---|---|
| tSgS 和 tSsS Shape 不同 | Source / Destination 布局不同，或使用了不同的 TiledCopy |
| 元素总数对不上 | ThrLayout、ValLayout 和 Tiler 不匹配 |
| Mode 数量与预期不同 | 一维、二维或嵌套 Shape 的表示不同 |
| Pred 维度不匹配 | 把元素级 Shape 误当成指令级 Pred Shape |
| Pred 全为 False | 坐标 Tensor 的 Tile 起点或 N 使用错误 |
| Pred 全为 True | 没有考虑当前 CTA 的全局偏移 |
| Copy 后 Shared Memory 数据异常 | 未处理无效位置，或者异步 Copy 尚未完成 |
| 向量化 Copy 编译失败 | 地址对齐、源/目的 Layout 或向量宽度不满足 Atom 要求 |
| 访问不合并 | 线程映射与连续内存方向不匹配 |

## 27. 最后建立一个统一的心智模型

```text
                  CTA
                   │
            确定当前 Tile
                   │
                   ▼
               TiledCopy
                   │
          ┌────────┼────────┐
          │        │        │
        Atom   ThrLayout  ValLayout
          │        │        │
          └────────┼────────┘
                   │
                   ▼
             get_slice(tidx)
                   │
                   ▼
                ThrCopy
                   │
          ┌────────┴────────┐
          │                 │
     partition_S      partition_D
          │                 │
          ▼                 ▼
       Source View       Dest View
          │                 │
          └────────┬────────┘
                   │
             Predication
                   │
                   ▼
               cute.copy
                   │
                   ▼
              数据搬运完成
```

**最重要的五句话：**

1. CTA 决定当前处理哪一个 Tile。
2. TiledCopy 决定整个 CTA 内线程如何搬运数据。
3. ThrCopy 是当前线程在整体 Copy 映射中的局部视图。
4. partition_S / partition_D 将整体映射投影到当前线程的源 / 目的 Tensor。
5. Pred 保护的是 Copy Atom 的执行，而不一定是每一个独立元素。

最终要记住：

```text
TiledCopy = 整体搬运规则

ThrCopy = 单线程分工

partition_S / partition_D = 线程局部数据视图

Pred = 指令级有效性控制

cute.copy = 实际执行搬运
```

---

## 官方参考资料

1. [CuTe DSL Naming Conventions — NVIDIA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/naming_conventions.html)
2. [NVIDIA CUTLASS 官方代码库](https://github.com/NVIDIA/cutlass)
3. [tiled_copy.cu — 官方 TiledCopy 教程](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu)
4. [sgemm_2.cu — TiledCopy / ThrCopy 实例](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_2.cu)
5. [tiled_copy_if.cu — Predication 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy_if.cu)
6. [CuTe Tensor Documentation — Thread-Value Partitioning](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/03_tensor.md)

**验证说明：** 本文的 `((8,1),2,1)`、Pred Shape 和具体索引代码，是基于给定布局、官方分区语义及此前报错信息所作的推导，并非在目标机器上实际编译运行的结果。应以当前 CUDA / CUTLASS / CuTe DSL 版本的 Trace 输出和编译结果为最终依据。