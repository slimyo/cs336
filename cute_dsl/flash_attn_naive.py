"""flash_attn_naive.py：CuTe DSL FlashAttention 前向（bf16、d=64），含 M6 的优化开关。

功能：
  - online safe softmax；缩放与 log2(e) 合并为 qk_scale，全程 exp2/log2
  - 输出 O，以及 base-2 的 LSE：lse2 = m + log2(sum 2^(s'-m))（m 已含 qk_scale）
  - causal=True：第 i 行只看 j<=i（左上对齐），跳过全被 mask 的 K tile，只在对角块做逐元素 mask
  - K/V 的 cp.async 多级流水（stages>=1；stages=2 即双缓冲）
  - smem 128B swizzle（use_swizzle=False 仅用于对照调试）
  - tile 可配：bq、bk、stages（用于 tile 扫描）

M6 的优化开关（用于消融；ablate_fwd.py 会逐个测）。
实测结论（RTX 5070 Ti，causal，d=64，bf16，相对基线）：O1 在 N=16384/32768 分别 +24%/+12%，默认开启；
O2、O3 单独开启反而慢 1–4%，默认关闭（保留开关只为复现消融）；O4 结果不一致，默认关闭，待单独测。
  - reverse_blocks（O1）：causal 下倒序发射 block，Q tile 号大（工作量大）的先发，
                          缓解 SM 间负载不均衡与尾部空转。逐位不变。
  - ilp_reduce   （O2）：rowmax / rowsum 用 4 条独立依赖链再合并，缩短 softmax 的串行依赖
                          （ncu：wait 停顿占 24.5%）。求和顺序改变，结果与基线不再逐位一致，
                          但误差在 bf16 舍入之内。
  - lazy_rescale （O3）：m_new == m_old 时 alpha 恰为 1，跳过 acc_o 的缩放乘法。逐位不变。
  - alias_q      （O4，实验）：Q 的 smem 别名到 K 的 stage 1（Q 在序言里读进寄存器后就不再使用，
                          第一次预取才会覆盖它），使每 block 的 smem 减少 BQ*D*2 字节。
                          仅 bq==bk 且 stages==2 时可用。若想让每 SM 放下 3 个 block，需要
                          smem_slack=0 并依赖动态 smem 基址本身 1024B 对齐，务必跑测试确认。

形状约定：Q/K/V/O 为 (B, N, 64) bf16；M(=LSE) 为 (B, Nq) fp32。
"""
import math
import torch
import cutlass
import cutlass.cute as cute

D = 64        # 特征维度（swizzle 的参数按 128B 行写死，D 只能是 64）
BQ = 64       # 默认 Q tile 行数
BK = 64       # 默认 K/V tile 行数
STG = 2       # 默认 K/V 流水级数
LOG2E = math.log2(math.e)
HAS_CUDA = torch.cuda.is_available()


class FLASH_ATTN_naive:

    def __init__(self, causal: bool = False, use_swizzle: bool = True,
                 use_pred: bool = True, zero_fill: bool = True, debug: bool = False,
                 bq: int = BQ, bk: int = BK, stages: int = STG,
                 reverse_blocks: bool = True, ilp_reduce: bool = False,
                 lazy_rescale: bool = False, alias_q: bool = False, smem_slack: int = 1024):
        self.causal, self.use_swizzle = causal, use_swizzle
        self.use_pred, self.zero_fill, self.debug = use_pred, zero_fill, debug
        self.BQ, self.BK, self.D, self.VEC = bq, bk, D, 8   # 8 个 bf16 = 16B = cp.async 单条搬运量
        self.STG = stages
        assert D == 64, "Swizzle<3,3,3> 对应 128B 行（64 个 bf16）；其他 D 需要改 swizzle 参数"
        assert bq % 16 == 0 and bk % 16 == 0 and stages >= 1

        # M6 优化开关
        self.reverse_blocks = bool(reverse_blocks) and bool(causal)   # 只有 causal 才有负载不均衡
        self.ilp_reduce = bool(ilp_reduce)
        self.lazy_rescale = bool(lazy_rescale)
        self.alias_q = bool(alias_q)
        self.smem_slack = int(smem_slack)
        if self.alias_q:
            assert bq == bk and stages == 2, "alias_q 要求 bq == bk 且 stages == 2"

        # softmax 缩放与换底合并：exp(x/√d) = 2^(x * qk_scale)
        self.qk_scale = LOG2E / math.sqrt(D)

        # warp 划分 atom_layout = (WM,1,1)：每 warp 独占 16 行完整的 S（行内 softmax 不跨 warp）；
        # 不切 N、不切 K。线程数由 bq 决定：bq=32/64/128 → 2/4/8 个 warp。
        self.WM = bq // 16
        self.NT = self.WM * 32

        # g2s 线程排布：每行 D/8 个线程，每线程一个 16B 向量；Q、K、V tile 共用
        self.thr_n = D // self.VEC
        self.thr_m = self.NT // self.thr_n                  # 一轮 g2s 覆盖的行数
        assert bq % self.thr_m == 0 and bk % self.thr_m == 0, \
            f"g2s 每轮覆盖 {self.thr_m} 行，BQ={bq}、BK={bk} 都必须是它的倍数"

    # ------------------------------------------------------------ smem layout
    def _smem_layout(self, rows, stages=None):
        """(rows, D[, stages])，行主序；可选套 128B swizzle。"""
        if stages is None:
            base = cute.make_layout((rows, self.D), stride=(self.D, 1))
        else:
            base = cute.make_layout((rows, self.D, stages),
                                    stride=(self.D, 1, rows * self.D))
        if self.use_swizzle:
            # Swizzle<B=3, M=3, S=3>：以元素为单位，bit[3:6) ^= bit[6:9)，
            # 即 16B 块号 ^= 行号%8。stage stride 是 4096 的倍数，远大于 bit 9，不受影响。
            return cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, base)
        return base

    def _vt(self, sV_s):
        """sV_s: (BK, D) 的（可能带 swizzle 的）smem tensor → 转置视图 (D, BK)。
        用 composition 从 sV 自己的 layout 派生，保证 swizzle 与写入时一致：
          Vt(n,k) = V(k,n)，把 (n,k) 看成 V 的列优先线性下标 n*BK + k。"""
        return cute.make_tensor(
            sV_s.iterator,
            cute.composition(sV_s.layout, cute.make_layout((self.D, self.BK), stride=(self.BK, 1))))

    def _alloc_sq(self, smem, dtype, sK):
        """sQ 的分配。普通 Python 辅助函数（内部是 Python 层面的分支，不会被 DSL 当作受控制流改写）。
        alias_q：别名到 sK 的 stage 1；否则单独分配。"""
        if self.alias_q:
            return sK[(None, None, 1)]
        return smem.allocate_tensor(dtype, self._smem_layout(self.BQ), 1024)

    # ------------------------------------------------------------ softmax 的行内规约（普通 Python 辅助函数）
    # 注意：下面两个是【普通 Python 函数】，不会被 DSL 预处理，所以循环必须用 Python 的 range，
    # 不能用 cutlass.range_constexpr（后者只在 @cute.kernel / @cute.jit 函数里有效，
    # 否则报 "range_constexpr should be preprocessed by preprocessor"）。
    def _row_max(self, acc_s, h, NS, neg_inf):
        """线程内第 h 行的最大值（本线程持有该行的 2*NS 个值）。
        ilp_reduce：4 条独立的 fmax 依赖链（按 n 的奇偶与 e 分配），最后再两两合并。"""
        if self.ilp_reduce:
            parts = [neg_inf, neg_inf, neg_inf, neg_inf]
            for n in range(NS):
                for e in range(2):
                    c = (n % 2) * 2 + e
                    parts[c] = cute.arch.fmax(parts[c], acc_s[(e, h), 0, n])
            return cute.arch.fmax(cute.arch.fmax(parts[0], parts[1]),
                                  cute.arch.fmax(parts[2], parts[3]))
        m = neg_inf
        for n in range(NS):
            for e in range(2):
                m = cute.arch.fmax(m, acc_s[(e, h), 0, n])
        return m

    def _exp_and_sum(self, acc_s, h, NS, qk_scale, m_new):
        """p = 2^(s*qk_scale - m_new) 写回 acc_s，并返回线程内的行和。
        ilp_reduce：4 条独立的加法链。"""
        zero = cutlass.Float32(0.0)
        if self.ilp_reduce:
            sums = [zero, zero, zero, zero]
            for n in range(NS):
                for e in range(2):
                    c = (n % 2) * 2 + e
                    p = cute.math.exp2(acc_s[(e, h), 0, n] * qk_scale - m_new, fastmath=True)
                    acc_s[(e, h), 0, n] = p
                    sums[c] = sums[c] + p
            return (sums[0] + sums[1]) + (sums[2] + sums[3])
        total = zero
        for n in range(NS):
            for e in range(2):
                p = cute.math.exp2(acc_s[(e, h), 0, n] * qk_scale - m_new, fastmath=True)
                acc_s[(e, h), 0, n] = p
                total = total + p
        return total

    # ------------------------------------------------------------ gmem → smem
    @cute.jit
    def _g2s(self, tiled_copy, tidx, g_tile, s_tile, c_tile, n_valid):
        """cp.async：gmem tile → smem tile。c_tile 是同形状的 identity tile，
        用它取全局行号，行号 >= n_valid 的向量不搬（pred=False）。
        zero_fill 时就地给这些越界向量写 0：它们从不被 cp.async 触碰，所以发射时写 0
        不会与异步拷贝冲突；对其他线程的可见性由之后的 barrier 保证。"""
        thr = tiled_copy.get_slice(tidx)
        tg = thr.partition_S(g_tile)
        ts = thr.partition_D(s_tile)
        tc = thr.partition_S(c_tile)
        n_rep = tg.shape[0][1]
        n_m = cute.size(tg, mode=[1])
        n_k = cute.size(tg, mode=[2])
        if self.use_pred:
            pred = cute.make_rmem_tensor((n_rep, n_m, n_k), cutlass.Boolean)
            for rv in cutlass.range_constexpr(n_rep):
                for m in cutlass.range_constexpr(n_m):
                    for k in cutlass.range_constexpr(n_k):
                        pred[rv, m, k] = tc[(0, rv), m, k][0] < n_valid
            cute.copy(tiled_copy, tg, ts, pred=pred)
        else:
            cute.copy(tiled_copy, tg, ts)
        if self.zero_fill:
            zero = cutlass.BFloat16(0.0)
            for rv in cutlass.range_constexpr(n_rep):
                for m in cutlass.range_constexpr(n_m):
                    for k in cutlass.range_constexpr(n_k):
                        if tc[(0, rv), m, k][0] >= n_valid:
                            for e in cutlass.range_constexpr(self.VEC):
                                ts[(e, rv), m, k] = zero

    # ------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor,
               O: cute.Tensor, M: cute.Tensor,
               tiled_copy_g2s: cute.TiledCopy, tiled_mma: cute.TiledMma):

        tidx, _, _ = cute.arch.thread_idx()
        bx, bidy, _ = cute.arch.block_idx()        # bx: 发射序号；bidy: 第几个 batch
        bidx = bx                                  # bidx: 第几个 Q tile
        if self.reverse_blocks:
            # O1：倒序。causal 下 Q tile 号越大要处理的 K tile 越多，硬件按 block 序号递增派发，
            # 倒过来就是“最重的先发、短的去填尾部”（LPT 调度）。只改映射，不改计算。
            gx, _, _ = cute.arch.grid_dim()
            bidx = gx - 1 - bx
        BQ_, BK_, D_ = self.BQ, self.BK, self.D
        NEG_INF = cutlass.Float32(-float("inf"))
        qk_scale = cutlass.Float32(self.qk_scale)

        # ================= 选出本 batch =================
        gsrcQ = Q[(bidy, None, None)]              # (Nq, D)
        gsrcK = K[(bidy, None, None)]              # (Nk, D)
        gsrcV = V[(bidy, None, None)]              # (Nk, D)，假定 Nv == Nk
        gsrcO = O[(bidy, None, None)]
        gM = M[(bidy, None)]                       # (Nq,) fp32，base-2 LSE
        Nq = gsrcQ.shape[0]                        # 运行时真实行数
        Nk = gsrcK.shape[0]

        # ================= smem：sQ 一份；sK、sV 各 stages 份 =================
        # 1024B 对齐：swizzle 的一个周期是 8 行×128B，基址对齐才能让 bank 分布与理论一致
        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(K.element_type, self._smem_layout(BK_, self.STG), 1024)   # (BK, D, STG)
        sV = smem.allocate_tensor(V.element_type, self._smem_layout(BK_, self.STG), 1024)
        sQ = self._alloc_sq(smem, Q.element_type, sK)          # O4：可别名到 sK 的 stage 1
        if self.debug:
            print("sK layout:", sK.layout)

        # ================= gmem tile 与 identity（坐标）tile =================
        gQ = cute.local_tile(gsrcQ, (BQ_, D_), (bidx, 0))
        gO = cute.local_tile(gsrcO, (BQ_, D_), (bidx, 0))
        idQ = cute.make_identity_tensor(gsrcQ.shape)
        idK = cute.make_identity_tensor(gsrcK.shape)
        cQ = cute.local_tile(idQ, (BQ_, D_), (bidx, 0))

        # ================= MMA 划分（用 stage 0 的静态切片定形状）=================
        thr_mma = tiled_mma.get_slice(tidx)
        sK0 = sK[(None, None, 0)]
        sVt0 = self._vt(sV[(None, None, 0)])

        # S = Q K^T：M=BQ, N=BK, K=D
        tCsA = thr_mma.partition_A(sQ)
        tCrA = tiled_mma.make_fragment_A(tCsA)                     # Q fragment：常驻寄存器
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sK0))  # K fragment：每 tile 重新读
        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, BK_)), cutlass.Float32)

        # O += P V：M=BQ, N=D, K=BK
        tOrP = cute.make_rmem_tensor(thr_mma.partition_shape_A((BQ_, BK_)), Q.element_type)
        tOrV = tiled_mma.make_fragment_B(thr_mma.partition_B(sVt0))
        acc_o = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, D_)), cutlass.Float32)
        acc_o.fill(0.0)

        # 坐标视图：与 acc 同形状，元素是 (tile 内行, tile 内列)
        tCcS = thr_mma.partition_C(cute.make_identity_tensor((BQ_, BK_)))
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((BQ_, D_)))

        NS = cute.size(acc_s, mode=[2])        # S 的 n8 块数 = BK/8
        NO = cute.size(acc_o, mode=[2])        # O 的 n8 块数 = D/8
        KS = cute.size(tCrA, mode=[2])         # S-GEMM 的 k16 块数 = D/16
        KO = cute.size(tOrP, mode=[2])         # O-GEMM 的 k16 块数 = BK/16

        # ================= s2r：ldmatrix（x4：一条读 4 个 8x8） =================
        atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), Q.element_type)
        atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), K.element_type)
        atom_Bt = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), V.element_type)
        s2r_A = cute.make_tiled_copy_A(atom_A, tiled_mma)
        s2r_B = cute.make_tiled_copy_B(atom_B, tiled_mma)     # K：k 连续，不转置
        s2r_Bt = cute.make_tiled_copy_B(atom_Bt, tiled_mma)   # V：n 连续，需要转置
        thr_A = s2r_A.get_slice(tidx)
        thr_B = s2r_B.get_slice(tidx)
        thr_Bt = s2r_Bt.get_slice(tidx)

        tAsA = thr_A.partition_S(sQ)
        tArA = thr_A.retile(tCrA)
        tBrB = thr_B.retile(tCrB)          # 寄存器侧只需建一次；smem 侧每轮按 stage 重新 partition
        tVrV = thr_Bt.retile(tOrV)

        # ================= 行统计量：每线程 2 行（g 与 g+8），放 rmem =================
        m_i = cute.make_rmem_tensor((2,), cutlass.Float32)    # 当前行最大值（已乘 qk_scale，base-2 域）
        l_i = cute.make_rmem_tensor((2,), cutlass.Float32)    # 线程局部的行和（收尾再跨线程规约）
        m_i.fill(NEG_INF)
        l_i.fill(0.0)

        # ================= 本 block 需要处理的 K tile 数 =================
        n_tiles = (Nk + BK_ - 1) // BK_
        if self.causal:
            # 本 Q tile 最后一行的行号 = bidx*BQ+BQ-1，最多需要 key 列 0..bidx*BQ+BQ-1
            n_tiles = cutlass.min(n_tiles, ((bidx + 1) * BQ_ + BK_ - 1) // BK_)

        # ================= 序言：Q，以及前 STG-1 个 K/V tile =================
        # 每个 tile 一个 commit group（超出 n_tiles 的发空组），这样 wait_group 的计数规则统一
        self._g2s(tiled_copy_g2s, tidx, gQ, sQ, cQ, Nq)
        cute.arch.cp_async_commit_group()                                   # 组：Q
        for s in cutlass.range_constexpr(self.STG - 1):
            if s < n_tiles:
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcK, (BK_, D_), (s, 0)),
                          sK[(None, None, s)], cute.local_tile(idK, (BK_, D_), (s, 0)), Nk)
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcV, (BK_, D_), (s, 0)),
                          sV[(None, None, s)], cute.local_tile(idK, (BK_, D_), (s, 0)), Nk)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.STG - 1)                         # 只等 Q
        cute.arch.barrier()
        for k in cutlass.range_constexpr(KS):                               # Q → 寄存器，之后不再碰 sQ
            cute.copy(s2r_A, tAsA[None, None, k], tArA[None, None, k])
        # alias_q 时：第 0 轮循环开头的 barrier 保证所有 warp 都读完 Q，之后才会预取覆盖 sK 的 stage 1

        # ================= 主循环：沿 K/V 的 tile =================
        for j in cutlass.range(n_tiles, unroll=1):
            stage = j % self.STG

            # ---- (0) 等第 j 块到齐 + 预取 ----
            if self.STG == 1:
                # 无流水：同步加载第 j 块到唯一的 stage
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcK, (BK_, D_), (j, 0)),
                          sK[(None, None, 0)], cute.local_tile(idK, (BK_, D_), (j, 0)), Nk)
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcV, (BK_, D_), (j, 0)),
                          sV[(None, None, 0)], cute.local_tile(idK, (BK_, D_), (j, 0)), Nk)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
            else:
                # 此时已提交到第 j+STG-2 块，wait_group(STG-2) 保证第 j 块已完成
                cute.arch.cp_async_wait_group(self.STG - 2)
                # barrier：① 所有线程的 cp.async 数据可见；② 所有 warp 已算完第 j-1 块，
                # 所以 stage (j-1)%STG 可以被覆盖
                cute.arch.barrier()
                nj = j + self.STG - 1                       # 预取第 j+STG-1 块，进入刚用完的 stage
                if nj < n_tiles:
                    self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcK, (BK_, D_), (nj, 0)),
                              sK[(None, None, nj % self.STG)],
                              cute.local_tile(idK, (BK_, D_), (nj, 0)), Nk)
                    self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcV, (BK_, D_), (nj, 0)),
                              sV[(None, None, nj % self.STG)],
                              cute.local_tile(idK, (BK_, D_), (nj, 0)), Nk)
                cute.arch.cp_async_commit_group()           # 无论是否发射都提交（空组立即完成）

            # ---- (1) 当前 stage 的 smem 视图 ----
            sK_s = sK[(None, None, stage)]
            sVt_s = self._vt(sV[(None, None, stage)])
            tBsB = thr_B.partition_S(sK_s)
            tVsV = thr_Bt.partition_S(sVt_s)

            # ---- (2) S = Q K_j^T ----
            acc_s.fill(0.0)
            for k in cutlass.range_constexpr(KS):
                cute.copy(s2r_B, tBsB[None, None, k], tBrB[None, None, k])
                cute.gemm(tiled_mma, acc_s, tCrA[None, None, k], tCrB[None, None, k], acc_s)

            # ---- (3) mask：越界列 +（causal）对角线上方，置 -inf（仅边界/对角 tile 进入） ----
            need_mask = (j * BK_ + BK_) > Nk
            if self.causal:
                need_mask = need_mask | ((j * BK_ + BK_ - 1) > (bidx * BQ_))
            if need_mask:
                for h in cutlass.range_constexpr(2):
                    for n in cutlass.range_constexpr(NS):
                        for e in cutlass.range_constexpr(2):
                            col_g = j * BK_ + tCcS[(e, h), 0, n][1]
                            masked = col_g >= Nk
                            if self.causal:
                                row_g = bidx * BQ_ + tCcS[(e, h), 0, n][0]
                                masked = masked | (col_g > row_g)
                            if masked:
                                acc_s[(e, h), 0, n] = NEG_INF

            # ---- (4) online safe softmax（base-2） ----
            # 线程持有行 g (h=0) 与 g+8 (h=1)；每行本线程有 NS*2 个值，
            # 同一行的另外 3 个线程是 lane 的 xor 1、xor 2 伙伴。
            for h in cutlass.range_constexpr(2):
                # 4a. 行最大值：线程内（O2：4 条独立链）→ 跨 4 线程
                local_max = self._row_max(acc_s, h, self.BK // 8, NEG_INF)
                local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, offset=2))
                local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, offset=1))

                # 4b. 新最大值（乘 qk_scale 后取 max；qk_scale>0 不改变 argmax）
                m_old = m_i[h]
                m_new = cute.arch.fmax(m_old, local_max * qk_scale)
                alpha = cute.math.exp2(m_old - m_new, fastmath=True)   # 首个 tile: 2^(-inf)=0
                m_i[h] = m_new

                # 4c. p = 2^(s*qk_scale - m_new)，累加线程局部行和（O2：4 条独立链）
                row_sum = self._exp_and_sum(acc_s, h, self.BK // 8, qk_scale, m_new)
                l_i[h] = l_i[h] * alpha + row_sum    # 线程局部和；跨线程规约推迟到收尾

                # 4d. 旧的 O 累加器按 alpha 缩放
                if self.lazy_rescale:
                    # O3：最大值没变时 alpha 恰为 1，乘 1 是空操作，整个 warp 都没变就整段跳过（逐位不变）
                    if m_new > m_old:
                        for n in cutlass.range_constexpr(NO):
                            for e in cutlass.range_constexpr(2):
                                acc_o[(e, h), 0, n] = acc_o[(e, h), 0, n] * alpha
                else:
                    for n in cutlass.range_constexpr(NO):
                        for e in cutlass.range_constexpr(2):
                            acc_o[(e, h), 0, n] = acc_o[(e, h), 0, n] * alpha

            # ---- (5) P：S 的 C 片段 → A 片段（寄存器内重排，fp32→bf16） ----
            # C: v=(e,h)，n8 块号 n；A: v=(e,h,kh)，k16 块号 kb；n = 2*kb + kh
            for kb in cutlass.range_constexpr(KO):
                for kh in cutlass.range_constexpr(2):
                    for h in cutlass.range_constexpr(2):
                        for e in cutlass.range_constexpr(2):
                            tOrP[(e, h, kh), 0, kb] = acc_s[(e, h), 0, 2 * kb + kh].to(Q.element_type)

            # ---- (6) O += P V_j ----
            for k in cutlass.range_constexpr(KO):
                cute.copy(s2r_Bt, tVsV[None, None, k], tVrV[None, None, k])     # ldmatrix.trans
                cute.gemm(tiled_mma, acc_o, tOrP[None, None, k], tOrV[None, None, k], acc_o)

            # STG>=2 时不需要循环末尾的 barrier：下一轮开头的 barrier 已覆盖
            if self.STG == 1:
                cute.arch.barrier()          # 下一轮同步加载会覆盖 sK/sV，须等所有 warp 算完

        # ================= 收尾：归一化、写 O 与 LSE =================
        for h in cutlass.range_constexpr(2):
            l = l_i[h]
            l = l + cute.arch.shuffle_sync_bfly(l, offset=2)     # 跨 4 线程求完整行和
            l = l + cute.arch.shuffle_sync_bfly(l, offset=1)
            inv_l = cutlass.Float32(1.0) / l                     # l >= 1（最大元素的 p = 1），安全
            for n in cutlass.range_constexpr(NO):
                for e in cutlass.range_constexpr(2):
                    acc_o[(e, h), 0, n] = acc_o[(e, h), 0, n] * inv_l

            # base-2 LSE：lse2 = m + log2(d)。每行只让 lane%4==0 的线程写一次。
            row = bidx * BQ_ + tCcO[(0, h), 0, 0][0]
            if (tidx % 4 == 0) & (row < Nq):
                gM[row] = m_i[h] + cute.math.log2(l)

        # 写 O：每线程写若干个 2 元素对。ncu 显示这里的多余扇区仅占 0%，不值得优化。
        tCgO = thr_mma.partition_C(gO)
        for h in cutlass.range_constexpr(2):
            row = bidx * BQ_ + tCcO[(0, h), 0, 0][0]
            if row < Nq:
                for n in cutlass.range_constexpr(NO):
                    for e in cutlass.range_constexpr(2):
                        tCgO[(e, h), 0, n] = acc_o[(e, h), 0, n].to(O.element_type)

    # ------------------------------------------------------------ host
    @cute.jit
    def __call__(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor,
                 O: cute.Tensor, M: cute.Tensor):
        assert Q.element_type == cutlass.BFloat16
        B = Q.shape[0]                                   # Q/K/V/O: [B, N, D]；M: [B, Nm] fp32
        Nq, Nk, Nv, No, Nm = Q.shape[1], K.shape[1], V.shape[1], O.shape[1], M.shape[1]

        # 用静态 D 重建 layout：行 stride 是静态常数，编译器才能证明 16B 对齐
        src3_Q = cute.make_tensor(Q.iterator, cute.make_layout((B, Nq, self.D), stride=(Nq * self.D, self.D, 1)))
        src3_K = cute.make_tensor(K.iterator, cute.make_layout((B, Nk, self.D), stride=(Nk * self.D, self.D, 1)))
        src3_V = cute.make_tensor(V.iterator, cute.make_layout((B, Nv, self.D), stride=(Nv * self.D, self.D, 1)))
        dst3_O = cute.make_tensor(O.iterator, cute.make_layout((B, No, self.D), stride=(No * self.D, self.D, 1)))
        dst3_M = cute.make_tensor(M.iterator, cute.make_layout((B, Nm), stride=(Nm, 1)))   # 二维

        # ---- g2s：cp.async 16B ----
        copy_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), Q.element_type, num_bits_per_copy=128)
        tiled_copy_g2s = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout((self.thr_m, self.thr_n), stride=(self.thr_n, 1)),   # 沿 d 相邻 → 合并访问
            cute.make_layout((1, self.VEC)))                                      # 每线程 1 行 × 8 元素

        # ---- mma：m16n8k16，bf16 输入、fp32 累加 ----
        mma_atom = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))
        # atom_layout = (WM,1,1)：warp 沿 M 排开，每 warp 独占 16 行完整的 S。
        # 每 warp 的重复次数：S-GEMM: MMA_M=1, MMA_N=BK/8, MMA_K=D/16；O-GEMM: MMA_M=1, MMA_N=D/8, MMA_K=BK/16
        tiled_mma = cute.make_tiled_mma(mma_atom, cute.make_layout((self.WM, 1, 1)))

        # smem = Q 一份（alias_q 时为 0）+ K、V 各 stages 份，再加 slack（默认 1KB，给 1024B 对齐的余量）
        # （即作业 §10.1 的公式 Bq·d·2 + stages·2·Bk·d·2）
        q_rows = 0 if self.alias_q else self.BQ
        smem_bytes = (q_rows + 2 * self.STG * self.BK) * self.D * 2 + self.smem_slack
        self.kernel(src3_Q, src3_K, src3_V, dst3_O, dst3_M, tiled_copy_g2s, tiled_mma).launch(
            grid=((Nq + self.BQ - 1) // self.BQ, B, 1),
            block=(self.NT, 1, 1),
            smem=smem_bytes,
        )