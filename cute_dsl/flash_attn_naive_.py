## 原版flash_attn, 无stage、swizzle、causal
import math
import torch
import cutlass
import cutlass.cute as cute

D = 64    # 特征维度
BQ = 64   # Q tile 行数
BK = 64   # K/V tile 行数（内层循环每次处理的列数）
LOG2E = math.log2(math.e)
HAS_CUDA = torch.cuda.is_available()


class FLASH_ATTN_naive:
    """每个 block：一个 batch 的一个 Q tile (BQ×D)，沿 K/V 循环做 online softmax。
       O = softmax(Q K^T / sqrt(d)) V ；同时输出 base-2 的 LSE：lse2 = m + log2(sum 2^(s'-m))"""

    def __init__(self, use_pred: bool = True, zero_fill: bool = True,
                 debug: bool = False, num_threads: int = 128):
        self.use_pred, self.zero_fill, self.debug = use_pred, zero_fill, debug
        self.BQ, self.BK, self.D, self.VEC = BQ, BK, D, 8   # 8 个 bf16 = 16B = cp.async 单条搬运量
        assert D % 16 == 0 and BK % 16 == 0

        # softmax 缩放与换底合并：exp(x/√d) = 2^(x * qk_scale)
        self.qk_scale = LOG2E / math.sqrt(D)

        # warp 排布 (WM,1,1)：每 warp 16 行，不切 N（保证一行 S 在同一 warp），不切 K
        self.WM = BQ // 16
        self.NT = num_threads
        assert self.NT == self.WM * 32, f"BQ={BQ} 需要恰好 {self.WM * 32} 个线程"

        # g2s 线程排布：每行 D/8 个线程，每线程搬一个 16B 向量；Q、K、V tile 共用
        self.thr_n = D // self.VEC
        self.thr_m = self.NT // self.thr_n          # 一轮覆盖行数（128 线程 → 16 行）
        assert BQ % self.thr_m == 0 and BK % self.thr_m == 0

    # ---------------------------------------------------------------- 搬运小工具
    @cute.jit
    def _g2s(self, tiled_copy, tidx, g_tile, s_tile, c_tile, n_valid):
        """cp.async：gmem tile → smem tile。c_tile 是同形状的 identity tile，
        用它取全局行号，行号 >= n_valid 的向量不搬（pred=False）。"""
        thr = tiled_copy.get_slice(tidx)
        tg = thr.partition_S(g_tile)
        ts = thr.partition_D(s_tile)
        tc = thr.partition_S(c_tile)
        if self.use_pred:
            n_rep = tg.shape[0][1]
            n_m = cute.size(tg, mode=[1])
            n_k = cute.size(tg, mode=[2])
            pred = cute.make_rmem_tensor((n_rep, n_m, n_k), cutlass.Boolean)
            for rv in cutlass.range_constexpr(n_rep):
                for m in cutlass.range_constexpr(n_m):
                    for k in cutlass.range_constexpr(n_k):
                        pred[rv, m, k] = tc[(0, rv), m, k][0] < n_valid
            cute.copy(tiled_copy, tg, ts, pred=pred)
        else:
            cute.copy(tiled_copy, tg, ts)

    @cute.jit
    def _zero_oob(self, tiled_copy, tidx, s_tile, c_tile, n_valid):
        """pred=False 的行 smem 里是残留垃圾，补 0（写 smem）。V 必须清零：否则 0×NaN=NaN。"""
        thr = tiled_copy.get_slice(tidx)
        ts = thr.partition_D(s_tile)
        tc = thr.partition_S(c_tile)
        n_rep = ts.shape[0][1]
        n_m = cute.size(ts, mode=[1])
        n_k = cute.size(ts, mode=[2])
        zero = cutlass.BFloat16(0.0)
        for rv in cutlass.range_constexpr(n_rep):
            for m in cutlass.range_constexpr(n_m):
                for k in cutlass.range_constexpr(n_k):
                    if tc[(0, rv), m, k][0] >= n_valid:
                        for e in cutlass.range_constexpr(self.VEC):
                            ts[(e, rv), m, k] = zero

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor,
               O: cute.Tensor, M: cute.Tensor,
               tiled_copy_g2s: cute.TiledCopy, tiled_mma: cute.TiledMma):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()      # bidx: 第几个 Q tile；bidy: 第几个 batch
        BQ_, BK_, D_ = self.BQ, self.BK, self.D
        NEG_INF = cutlass.Float32(-float("inf"))
        qk_scale = cutlass.Float32(self.qk_scale)

        # ================= 选出本 batch =================
        gsrcQ = Q[(bidy, None, None)]              # (Nq, D)
        gsrcK = K[(bidy, None, None)]              # (Nk, D)
        gsrcV = V[(bidy, None, None)]              # (Nk, D)，假定 Nv == Nk
        gsrcO = O[(bidy, None, None)]              # (Nq 或 pad, D)
        gM = M[(bidy, None)]                       # (Nq 或 pad,)  fp32，base-2 LSE
        Nq = gsrcQ.shape[0]                        # 运行时真实行数
        Nk = gsrcK.shape[0]

        # ================= smem：三块静态 tile =================
        # 无 swizzle，ldmatrix 会有 bank conflict；先保证正确，之后再套 swizzle。
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(Q.element_type, cute.make_layout((BQ_, D_), stride=(D_, 1)), 16)
        sK = smem.allocate_tensor(K.element_type, cute.make_layout((BK_, D_), stride=(D_, 1)), 16)
        sV = smem.allocate_tensor(V.element_type, cute.make_layout((BK_, D_), stride=(D_, 1)), 16)
        # V 的转置视图 (D, BK)：同一块内存，只是把 (k,n) 看成 (n,k)。
        # mma 的 B 操作数是 (N=D, K=BK)，配合 ldmatrix.trans 使用。
        sVt = cute.make_tensor(sV.iterator, cute.make_layout((D_, BK_), stride=(1, D_)))

        # ================= gmem tile 与 identity（坐标）tile =================
        gQ = cute.local_tile(gsrcQ, (BQ_, D_), (bidx, 0))
        gO = cute.local_tile(gsrcO, (BQ_, D_), (bidx, 0))
        idQ = cute.make_identity_tensor(gsrcQ.shape)
        idK = cute.make_identity_tensor(gsrcK.shape)
        cQ = cute.local_tile(idQ, (BQ_, D_), (bidx, 0))

        # ================= MMA 划分 =================
        thr_mma = tiled_mma.get_slice(tidx)

        # ---- S = Q K^T：M=BQ, N=BK, K=D ----
        tCsA = thr_mma.partition_A(sQ)                        # ((2,2,2), 1, D/16)
        tCsB = thr_mma.partition_B(sK)                        # ((2,2),  BK/8, D/16)
        tCrA = tiled_mma.make_fragment_A(tCsA)                # Q fragment：常驻寄存器
        tCrB = tiled_mma.make_fragment_B(tCsB)                # K fragment：每 tile 重新读
        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, BK_)), cutlass.Float32)  # ((2,2),1,BK/8)

        # ---- O += P V：M=BQ, N=D, K=BK ----
        tOrP = cute.make_rmem_tensor(thr_mma.partition_shape_A((BQ_, BK_)), Q.element_type)    # ((2,2,2),1,BK/16)
        tOsVt = thr_mma.partition_B(sVt)                      # ((2,2), D/8, BK/16)
        tOrV = tiled_mma.make_fragment_B(tOsVt)
        acc_o = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, D_)), cutlass.Float32)   # ((2,2),1,D/8)
        acc_o.fill(0.0)

        # 坐标视图：与 acc 同形状，元素是 (tile 内行, tile 内列)
        tCcS = thr_mma.partition_C(cute.make_identity_tensor((BQ_, BK_)))
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((BQ_, D_)))

        NS = cute.size(acc_s, mode=[2])        # S 的 n8 块数 = BK/8
        NO = cute.size(acc_o, mode=[2])        # O 的 n8 块数 = D/8
        KS = cute.size(tCrA, mode=[2])         # S-GEMM 的 k16 块数 = D/16
        KO = cute.size(tOrP, mode=[2])         # O-GEMM 的 k16 块数 = BK/16

        # ================= s2r：ldmatrix 拷贝（x4：一条读 4 个 8x8） =================
        atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), Q.element_type)
        atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), K.element_type)
        atom_Bt = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), V.element_type)

        s2r_A = cute.make_tiled_copy_A(atom_A, tiled_mma)
        s2r_B = cute.make_tiled_copy_B(atom_B, tiled_mma)     # K：k 连续，不转置
        s2r_Bt = cute.make_tiled_copy_B(atom_Bt, tiled_mma)   # V：n 连续，需要转置

        tAsA = s2r_A.get_slice(tidx).partition_S(sQ)
        tArA = s2r_A.get_slice(tidx).retile(tCrA)
        tBsB = s2r_B.get_slice(tidx).partition_S(sK)
        tBrB = s2r_B.get_slice(tidx).retile(tCrB)
        tVsV = s2r_Bt.get_slice(tidx).partition_S(sVt)
        tVrV = s2r_Bt.get_slice(tidx).retile(tOrV)

        # ================= 行统计量：每线程 2 行（g 与 g+8），放 rmem =================
        m_i = cute.make_rmem_tensor((2,), cutlass.Float32)    # 当前行最大值（已乘 qk_scale，base-2 域）
        l_i = cute.make_rmem_tensor((2,), cutlass.Float32)    # 线程局部的行和（收尾再跨线程规约）
        m_i.fill(NEG_INF)
        l_i.fill(0.0)

        # ================= 加载 Q，并常驻寄存器 =================
        self._g2s(tiled_copy_g2s, tidx, gQ, sQ, cQ, Nq)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if self.zero_fill:
            self._zero_oob(tiled_copy_g2s, tidx, sQ, cQ, Nq)
            cute.arch.barrier()
        for k in cutlass.range_constexpr(KS):                 # Q 的 fragment 只读一次
            cute.copy(s2r_A, tAsA[None, None, k], tArA[None, None, k])

        # ================= 主循环：沿 K/V 的 tile =================
        n_tiles = (Nk + BK_ - 1) // BK_
        for j in cutlass.range(n_tiles, unroll=1):
            gK = cute.local_tile(gsrcK, (BK_, D_), (j, 0))
            gV = cute.local_tile(gsrcV, (BK_, D_), (j, 0))
            cK = cute.local_tile(idK, (BK_, D_), (j, 0))      # 全局行号 = j*BK + 行

            # ---- (1) g2s：K_j、V_j（朴素版：同步等待；之后可做双缓冲） ----
            self._g2s(tiled_copy_g2s, tidx, gK, sK, cK, Nk)
            self._g2s(tiled_copy_g2s, tidx, gV, sV, cK, Nk)   # V 与 K 的行坐标相同
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if self.zero_fill:
                self._zero_oob(tiled_copy_g2s, tidx, sK, cK, Nk)
                self._zero_oob(tiled_copy_g2s, tidx, sV, cK, Nk)
                cute.arch.barrier()

            # ---- (2) S = Q K_j^T ----
            acc_s.fill(0.0)
            for k in cutlass.range_constexpr(KS):
                cute.copy(s2r_B, tBsB[None, None, k], tBrB[None, None, k])      # 4 条 x4
                cute.gemm(tiled_mma, acc_s, tCrA[None, None, k], tCrB[None, None, k], acc_s)  # 8 条 mma

            # ---- (3) 越界列 mask 成 -inf（只有最后一个 tile 会触发） ----
            if j * BK_ + BK_ > Nk:
                for h in cutlass.range_constexpr(2):
                    for n in cutlass.range_constexpr(NS):
                        for e in cutlass.range_constexpr(2):
                            if j * BK_ + tCcS[(e, h), 0, n][1] >= Nk:
                                acc_s[(e, h), 0, n] = NEG_INF

            # ---- (4) online safe softmax（base-2） ----
            # 线程持有行 g (h=0) 与 g+8 (h=1)；每行本线程有 NS*2 个值，
            # 同一行的另外 3 个线程是 lane 的 xor 1、xor 2 伙伴。
            for h in cutlass.range_constexpr(2):
                # 4a. 行最大值：线程内 → 跨 4 线程
                local_max = NEG_INF
                for n in cutlass.range_constexpr(NS):
                    for e in cutlass.range_constexpr(2):
                        local_max = cute.arch.fmax(local_max, acc_s[(e, h), 0, n])
                local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, offset=2))
                local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, offset=1))

                # 4b. 新最大值（乘 qk_scale 后取 max；qk_scale>0 不改变 argmax）
                m_old = m_i[h]
                m_new = cute.arch.fmax(m_old, local_max * qk_scale)
                alpha = cute.math.exp2(m_old - m_new, fastmath=True)   # 首个 tile: 2^(-inf)=0
                m_i[h] = m_new

                # 4c. p = 2^(s*qk_scale - m_new)，累加线程局部行和
                row_sum = cutlass.Float32(0.0)
                for n in cutlass.range_constexpr(NS):
                    for e in cutlass.range_constexpr(2):
                        p = cute.math.exp2(acc_s[(e, h), 0, n] * qk_scale - m_new, fastmath=True)
                        acc_s[(e, h), 0, n] = p
                        row_sum = row_sum + p
                l_i[h] = l_i[h] * alpha + row_sum    # 线程局部和；跨线程规约推迟到收尾

                # 4d. 旧的 O 累加器按 alpha 缩放
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

            cute.arch.barrier()      # 下一轮 g2s 会覆盖 sK/sV，须等所有 warp 读完

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
            if (tidx % 4 == 0) and (row < Nq):
                gM[row] = m_i[h] + cute.math.log2(l)

        # 写 O：每线程写若干个 2 元素对（朴素版；之后可经 smem 整理成 16B 合并写）
        tCgO = thr_mma.partition_C(gO)
        for h in cutlass.range_constexpr(2):
            row = bidx * BQ_ + tCcO[(0, h), 0, 0][0]
            if row < Nq:
                for n in cutlass.range_constexpr(NO):
                    for e in cutlass.range_constexpr(2):
                        tCgO[(e, h), 0, n] = acc_o[(e, h), 0, n].to(O.element_type)

    # ---------------------------------------------------------------- host
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
        dst3_M = cute.make_tensor(M.iterator, cute.make_layout((B, Nm), stride=(Nm, 1)))   # 二维！

        copy_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), Q.element_type, num_bits_per_copy=128)
        tiled_copy_g2s = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout((self.thr_m, self.thr_n), stride=(self.thr_n, 1)),   # 沿 d 相邻 → 合并访问
            cute.make_layout((1, self.VEC)))                                      # 每线程 1 行 × 8 元素

        mma_atom = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))
        # atom_layout = (WM,1,1)：warp 沿 M；S-GEMM: MMA_M=1, MMA_N=BK/8, MMA_K=D/16
        #                              O-GEMM: MMA_M=1, MMA_N=D/8,  MMA_K=BK/16
        tiled_mma = cute.make_tiled_mma(mma_atom, cute.make_layout((self.WM, 1, 1)))

        self.kernel(src3_Q, src3_K, src3_V, dst3_O, dst3_M, tiled_copy_g2s, tiled_mma).launch(
            grid=((Nq + self.BQ - 1) // self.BQ, B, 1),
            block=(self.NT, 1, 1),
            smem=3 * self.BQ * self.D * 2,               # sQ + sK + sV，各 64*64 个 bf16 = 24KB
        )