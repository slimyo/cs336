"""CuTe DSL backward：两个 kernel（dK/dV、dQ），无原子操作，结果确定。
约定：Q,K,V,O,dO: (B,N,64) bf16；L2: (B,Nq) fp32，base-2 LSE（与前向保存的一致）；
      Dvec = rowsum(dO*O): (B,Nq) fp32。causal：左上对齐（保留 j <= i）。"""
import math
import torch
import cutlass
import cutlass.cute as cute

D = 64
BQ = 64
BK = 64
STG = 2
LOG2E = math.log2(math.e)


# =============================================================================
# 公共部分：smem layout / 转置视图 / gmem→smem
# =============================================================================
class _BwdBase:
    def __init__(self, causal: bool = False, use_swizzle: bool = True, num_threads: int = 128):
        self.causal, self.use_swizzle = causal, use_swizzle
        self.use_pred, self.zero_fill = True, True
        self.BQ, self.BK, self.D, self.VEC = BQ, BK, D, 8
        assert D == 64
        assert BQ == BK, "K/V（或 Q/dO）别名到另一个 stage 的缓冲上，需要 BQ == BK"
        self.qk_scale = LOG2E / math.sqrt(D)
        self.inv_sqrt = 1.0 / math.sqrt(D)
        self.WM = BQ // 16                      # warp 沿 M；M 在两个 kernel 里分别是 K 行 / Q 行，都是 64
        self.NT = num_threads
        assert self.NT == self.WM * 32
        self.thr_n = D // self.VEC
        self.thr_m = self.NT // self.thr_n
        assert BQ % self.thr_m == 0

    def _smem_layout(self, rows, stages=None):
        if stages is None:
            base = cute.make_layout((rows, self.D), stride=(self.D, 1))
        else:
            base = cute.make_layout((rows, self.D, stages), stride=(self.D, 1, rows * self.D))
        if self.use_swizzle:
            return cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, base)
        return base

    def _vt(self, s_tile):
        """(rows, D) → 转置视图 (D, rows)，从自身 layout 用 composition 派生以保持 swizzle 一致。"""
        return cute.make_tensor(
            s_tile.iterator,
            cute.composition(s_tile.layout, cute.make_layout((self.D, self.BK), stride=(self.BK, 1))))

    @cute.jit
    def _g2s(self, tiled_copy, tidx, g_tile, s_tile, c_tile, n_valid):
        """cp.async 搬一个 (rows, D) tile；越界行不搬并就地补零（写 smem）。"""
        thr = tiled_copy.get_slice(tidx)
        tg = thr.partition_S(g_tile)
        ts = thr.partition_D(s_tile)
        tc = thr.partition_S(c_tile)
        n_rep = tg.shape[0][1]
        n_m = cute.size(tg, mode=[1])
        n_k = cute.size(tg, mode=[2])
        pred = cute.make_rmem_tensor((n_rep, n_m, n_k), cutlass.Boolean)
        for rv in cutlass.range_constexpr(n_rep):
            for m in cutlass.range_constexpr(n_m):
                for k in cutlass.range_constexpr(n_k):
                    pred[rv, m, k] = tc[(0, rv), m, k][0] < n_valid
        cute.copy(tiled_copy, tg, ts, pred=pred)
        zero = cutlass.BFloat16(0.0)
        for rv in cutlass.range_constexpr(n_rep):
            for m in cutlass.range_constexpr(n_m):
                for k in cutlass.range_constexpr(n_k):
                    if tc[(0, rv), m, k][0] >= n_valid:
                        for e in cutlass.range_constexpr(self.VEC):
                            ts[(e, rv), m, k] = zero

    def _make_g2s(self, dtype):
        atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), dtype, num_bits_per_copy=128)
        return cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((self.thr_m, self.thr_n), stride=(self.thr_n, 1)),
            cute.make_layout((1, self.VEC)))

    def _make_mma(self):
        atom = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))
        return cute.make_tiled_mma(atom, cute.make_layout((self.WM, 1, 1)))

    def _ldmatrix_atoms(self, dtype):
        a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dtype)
        at = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), dtype)
        return a, at

    def _tensors3(self, T, N):
        B = T.shape[0]
        return cute.make_tensor(T.iterator, cute.make_layout((B, N, self.D), stride=(N * self.D, self.D, 1)))


# =============================================================================
# dK / dV kernel：一个 block 负责一个 K/V tile，内层循环 Q tile
# =============================================================================
class FlashBwdDKDV(_BwdBase):

    @cute.kernel
    def kernel(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, dO: cute.Tensor,
               L: cute.Tensor, Dv: cute.Tensor, dK: cute.Tensor, dV: cute.Tensor,
               tiled_copy_g2s: cute.TiledCopy, tiled_mma: cute.TiledMma):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()          # bidx: K/V tile 号；bidy: batch
        BQ_, BK_, D_ = self.BQ, self.BK, self.D
        qk_scale = cutlass.Float32(self.qk_scale)
        inv_sqrt = cutlass.Float32(self.inv_sqrt)

        gsrcQ = Q[(bidy, None, None)]
        gsrcK = K[(bidy, None, None)]
        gsrcV = V[(bidy, None, None)]
        gsrcdO = dO[(bidy, None, None)]
        gsrcdK = dK[(bidy, None, None)]
        gsrcdV = dV[(bidy, None, None)]
        gL = L[(bidy, None)]
        gD = Dv[(bidy, None)]
        Nq = gsrcQ.shape[0]
        Nk = gsrcK.shape[0]

        # ---------------- smem：Q、dO 各 2 stage；K、V 别名到 stage 1 ----------------
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(Q.element_type, self._smem_layout(BQ_, STG), 1024)
        sdO = smem.allocate_tensor(dO.element_type, self._smem_layout(BQ_, STG), 1024)
        sK = sQ[(None, None, 1)]       # 序言里临时放 K；读进寄存器后，第一次预取才会覆盖它
        sV = sdO[(None, None, 1)]

        idQ = cute.make_identity_tensor(gsrcQ.shape)
        idK = cute.make_identity_tensor(gsrcK.shape)
        gK = cute.local_tile(gsrcK, (BK_, D_), (bidx, 0))
        gV = cute.local_tile(gsrcV, (BK_, D_), (bidx, 0))
        cK = cute.local_tile(idK, (BK_, D_), (bidx, 0))

        # ---------------- MMA 划分：所有 GEMM 的 M 都是 K 行（=64） ----------------
        thr_mma = tiled_mma.get_slice(tidx)
        # S^T = K Q^T、dP^T = V dO^T：A = K/V（常驻寄存器），B = Q/dO（k=d 连续，不转置）
        tKrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sK))
        tVrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sV))
        tBrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ[(None, None, 0)]))
        # dV += P^T dO、dK += dS^T Q：B = dO/Q 的 (n=d, k=q) 视图（需要 ldmatrix.trans）
        tTrB = tiled_mma.make_fragment_B(thr_mma.partition_B(self._vt(sdO[(None, None, 0)])))

        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, BQ_)), cutlass.Float32)   # ((2,2),1,BQ/8)
        acc_dp = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, BQ_)), cutlass.Float32)
        acc_dk = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, D_)), cutlass.Float32)   # ((2,2),1,D/8)
        acc_dv = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, D_)), cutlass.Float32)
        acc_dk.fill(0.0)
        acc_dv.fill(0.0)
        tPrA = cute.make_rmem_tensor(thr_mma.partition_shape_A((BK_, BQ_)), Q.element_type)     # P^T 的 A fragment
        tdSrA = cute.make_rmem_tensor(thr_mma.partition_shape_A((BK_, BQ_)), Q.element_type)    # dS^T 的 A fragment

        tCcS = thr_mma.partition_C(cute.make_identity_tensor((BK_, BQ_)))   # (K 行, Q 列) 的 tile 内坐标
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((BK_, D_)))

        NS = cute.size(acc_s, mode=[2])        # BQ/8
        NO = cute.size(acc_dk, mode=[2])       # D/8
        KD = cute.size(tKrA, mode=[2])         # D/16
        KQ = cute.size(tPrA, mode=[2])         # BQ/16

        # ---------------- ldmatrix ----------------
        atom_A, atom_At = self._ldmatrix_atoms(Q.element_type)
        s2r_A = cute.make_tiled_copy_A(atom_A, tiled_mma)
        s2r_B = cute.make_tiled_copy_B(atom_A, tiled_mma)
        s2r_Bt = cute.make_tiled_copy_B(atom_At, tiled_mma)
        thr_A, thr_B, thr_Bt = s2r_A.get_slice(tidx), s2r_B.get_slice(tidx), s2r_Bt.get_slice(tidx)
        tKsA, tKrA_v = thr_A.partition_S(sK), thr_A.retile(tKrA)
        tVsA, tVrA_v = thr_A.partition_S(sV), thr_A.retile(tVrA)
        tBrB_v = thr_B.retile(tBrB)
        tTrB_v = thr_Bt.retile(tTrB)

        # ---------------- 本 block 要处理的 Q tile 范围 ----------------
        n_total = (Nq + BQ_ - 1) // BQ_
        i_start = 0
        if self.causal:
            i_start = (bidx * BK_) // BQ_          # 行号 < bidx*BK 的 Q 都看不到这个 K tile
        n_iter = cutlass.max(n_total - i_start, cutlass.Int32(0))   # Nk>Nq 时可能为 0

        # ---------------- 序言：K、V（临时放在 stage 1），Q、dO 的第一块放 stage 0 ----------------
        self._g2s(tiled_copy_g2s, tidx, gK, sK, cK, Nk)
        self._g2s(tiled_copy_g2s, tidx, gV, sV, cK, Nk)
        if n_iter > 0:
            self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcQ, (BQ_, D_), (i_start, 0)),
                      sQ[(None, None, 0)], cute.local_tile(idQ, (BQ_, D_), (i_start, 0)), Nq)
            self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcdO, (BQ_, D_), (i_start, 0)),
                      sdO[(None, None, 0)], cute.local_tile(idQ, (BQ_, D_), (i_start, 0)), Nq)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        for k in cutlass.range_constexpr(KD):           # K、V → 寄存器，之后常驻
            cute.copy(s2r_A, tKsA[None, None, k], tKrA_v[None, None, k])
            cute.copy(s2r_A, tVsA[None, None, k], tVrA_v[None, None, k])
        # 不需要再加 barrier：循环第一轮开头的 barrier 保证所有 warp 读完 stage 1，才会有人预取覆盖它

        # ---------------- 主循环：沿 Q tile ----------------
        for t in cutlass.range(n_iter, unroll=1):
            qi = i_start + t
            stage = t % STG

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if t + 1 < n_iter:
                nxt = (t + 1) % STG
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcQ, (BQ_, D_), (qi + 1, 0)),
                          sQ[(None, None, nxt)], cute.local_tile(idQ, (BQ_, D_), (qi + 1, 0)), Nq)
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcdO, (BQ_, D_), (qi + 1, 0)),
                          sdO[(None, None, nxt)], cute.local_tile(idQ, (BQ_, D_), (qi + 1, 0)), Nq)
            cute.arch.cp_async_commit_group()

            sQ_s = sQ[(None, None, stage)]
            sdO_s = sdO[(None, None, stage)]
            tBsQ = thr_B.partition_S(sQ_s)
            tBsdO = thr_B.partition_S(sdO_s)
            tTsQ = thr_Bt.partition_S(self._vt(sQ_s))
            tTsdO = thr_Bt.partition_S(self._vt(sdO_s))

            # (1) S^T = K Q^T， dP^T = V dO^T（B 的寄存器 fragment 在两次 GEMM 间复用）
            acc_s.fill(0.0)
            for k in cutlass.range_constexpr(KD):
                cute.copy(s2r_B, tBsQ[None, None, k], tBrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_s, tKrA[None, None, k], tBrB[None, None, k], acc_s)
            acc_dp.fill(0.0)
            for k in cutlass.range_constexpr(KD):
                cute.copy(s2r_B, tBsdO[None, None, k], tBrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dp, tVrA[None, None, k], tBrB[None, None, k], acc_dp)

            # (2) P^T = exp2(S^T*scale - L2[q])，dS^T = P^T * (dP^T - D[q])；L2、D 沿“列”变化
            q0 = qi * BQ_
            for n in cutlass.range_constexpr(NS):
                for e in cutlass.range_constexpr(2):
                    q_g = q0 + tCcS[(e, 0), 0, n][1]
                    l_val = cutlass.Float32(0.0)
                    d_val = cutlass.Float32(0.0)
                    if q_g < Nq:
                        l_val = gL[q_g]
                        d_val = gD[q_g]
                    for h in cutlass.range_constexpr(2):
                        k_g = bidx * BK_ + tCcS[(e, h), 0, n][0]
                        p = cute.math.exp2(acc_s[(e, h), 0, n] * qk_scale - l_val, fastmath=True)
                        masked = q_g >= Nq                      # 越界的 Q 行（L2 无意义）
                        if self.causal:
                            masked = masked | (k_g > q_g)       # key 在 query 之后
                        if masked:
                            p = cutlass.Float32(0.0)
                        acc_s[(e, h), 0, n] = p
                        acc_dp[(e, h), 0, n] = p * (acc_dp[(e, h), 0, n] - d_val)

            # (3) P^T、dS^T：C 片段 → A 片段（寄存器内重排，fp32→bf16）
            for kb in cutlass.range_constexpr(KQ):
                for kh in cutlass.range_constexpr(2):
                    for h in cutlass.range_constexpr(2):
                        for e in cutlass.range_constexpr(2):
                            tPrA[(e, h, kh), 0, kb] = acc_s[(e, h), 0, 2 * kb + kh].to(Q.element_type)
                            tdSrA[(e, h, kh), 0, kb] = acc_dp[(e, h), 0, 2 * kb + kh].to(Q.element_type)

            # (4) dV += P^T dO；dK += dS^T Q
            for k in cutlass.range_constexpr(KQ):
                cute.copy(s2r_Bt, tTsdO[None, None, k], tTrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dv, tPrA[None, None, k], tTrB[None, None, k], acc_dv)
            for k in cutlass.range_constexpr(KQ):
                cute.copy(s2r_Bt, tTsQ[None, None, k], tTrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dk, tdSrA[None, None, k], tTrB[None, None, k], acc_dk)

        # ---------------- 收尾：dK *= 1/√d，转 bf16，按行写回 ----------------
        tCgdK = thr_mma.partition_C(cute.local_tile(gsrcdK, (BK_, D_), (bidx, 0)))
        tCgdV = thr_mma.partition_C(cute.local_tile(gsrcdV, (BK_, D_), (bidx, 0)))
        for h in cutlass.range_constexpr(2):
            row = bidx * BK_ + tCcO[(0, h), 0, 0][0]
            if row < Nk:
                for n in cutlass.range_constexpr(NO):
                    for e in cutlass.range_constexpr(2):
                        tCgdK[(e, h), 0, n] = (acc_dk[(e, h), 0, n] * inv_sqrt).to(dK.element_type)
                        tCgdV[(e, h), 0, n] = acc_dv[(e, h), 0, n].to(dV.element_type)

    @cute.jit
    def __call__(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, dO: cute.Tensor,
                 L: cute.Tensor, Dv: cute.Tensor, dK: cute.Tensor, dV: cute.Tensor):
        assert Q.element_type == cutlass.BFloat16
        B, Nq, Nk = Q.shape[0], Q.shape[1], K.shape[1]
        q3, k3, v3 = self._tensors3(Q, Nq), self._tensors3(K, Nk), self._tensors3(V, Nk)
        do3, dk3, dv3 = self._tensors3(dO, Nq), self._tensors3(dK, Nk), self._tensors3(dV, Nk)
        l2 = cute.make_tensor(L.iterator, cute.make_layout((B, Nq), stride=(Nq, 1)))
        d2 = cute.make_tensor(Dv.iterator, cute.make_layout((B, Nq), stride=(Nq, 1)))

        smem_bytes = 2 * STG * self.BQ * self.D * 2 + 1024      # Q、dO 各 2 stage = 32KB，+1KB 对齐余量
        self.kernel(q3, k3, v3, do3, l2, d2, dk3, dv3, self._make_g2s(Q.element_type), self._make_mma()).launch(
            grid=((Nk + self.BK - 1) // self.BK, B, 1),
            block=(self.NT, 1, 1),
            smem=smem_bytes)


# =============================================================================
# dQ kernel：一个 block 负责一个 Q tile，内层循环 K/V tile
# =============================================================================
class FlashBwdDQ(_BwdBase):

    @cute.kernel
    def kernel(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, dO: cute.Tensor,
               L: cute.Tensor, Dv: cute.Tensor, dQ: cute.Tensor,
               tiled_copy_g2s: cute.TiledCopy, tiled_mma: cute.TiledMma):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()          # bidx: Q tile 号
        BQ_, BK_, D_ = self.BQ, self.BK, self.D
        qk_scale = cutlass.Float32(self.qk_scale)
        inv_sqrt = cutlass.Float32(self.inv_sqrt)
        NEG = cutlass.Float32(0.0)

        gsrcQ = Q[(bidy, None, None)]
        gsrcK = K[(bidy, None, None)]
        gsrcV = V[(bidy, None, None)]
        gsrcdO = dO[(bidy, None, None)]
        gsrcdQ = dQ[(bidy, None, None)]
        gL = L[(bidy, None)]
        gD = Dv[(bidy, None)]
        Nq = gsrcQ.shape[0]
        Nk = gsrcK.shape[0]

        # ---------------- smem：K、V 各 2 stage；Q、dO 别名到 stage 1 ----------------
        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(K.element_type, self._smem_layout(BK_, STG), 1024)
        sV = smem.allocate_tensor(V.element_type, self._smem_layout(BK_, STG), 1024)
        sQ = sK[(None, None, 1)]
        sdO = sV[(None, None, 1)]

        idQ = cute.make_identity_tensor(gsrcQ.shape)
        idK = cute.make_identity_tensor(gsrcK.shape)
        gQ = cute.local_tile(gsrcQ, (BQ_, D_), (bidx, 0))
        gdO = cute.local_tile(gsrcdO, (BQ_, D_), (bidx, 0))
        cQ = cute.local_tile(idQ, (BQ_, D_), (bidx, 0))

        # ---------------- MMA 划分：M = Q 行 ----------------
        thr_mma = tiled_mma.get_slice(tidx)
        # S = Q K^T、dP = dO V^T：A = Q/dO（常驻寄存器），B = K/V（不转置）
        tQrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tdOrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sdO))
        tBrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sK[(None, None, 0)]))
        # dQ += dS K：B = K 的 (n=d, k=kv) 视图（ldmatrix.trans）
        tTrB = tiled_mma.make_fragment_B(thr_mma.partition_B(self._vt(sK[(None, None, 0)])))

        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, BK_)), cutlass.Float32)
        acc_dp = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, BK_)), cutlass.Float32)
        acc_dq = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, D_)), cutlass.Float32)
        acc_dq.fill(0.0)
        tdSrA = cute.make_rmem_tensor(thr_mma.partition_shape_A((BQ_, BK_)), Q.element_type)

        tCcS = thr_mma.partition_C(cute.make_identity_tensor((BQ_, BK_)))   # (Q 行, K 列)
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((BQ_, D_)))
        NS = cute.size(acc_s, mode=[2])
        NO = cute.size(acc_dq, mode=[2])
        KD = cute.size(tQrA, mode=[2])
        KK = cute.size(tdSrA, mode=[2])

        atom_A, atom_At = self._ldmatrix_atoms(Q.element_type)
        s2r_A = cute.make_tiled_copy_A(atom_A, tiled_mma)
        s2r_B = cute.make_tiled_copy_B(atom_A, tiled_mma)
        s2r_Bt = cute.make_tiled_copy_B(atom_At, tiled_mma)
        thr_A, thr_B, thr_Bt = s2r_A.get_slice(tidx), s2r_B.get_slice(tidx), s2r_Bt.get_slice(tidx)
        tQsA, tQrA_v = thr_A.partition_S(sQ), thr_A.retile(tQrA)
        tdOsA, tdOrA_v = thr_A.partition_S(sdO), thr_A.retile(tdOrA)
        tBrB_v = thr_B.retile(tBrB)
        tTrB_v = thr_Bt.retile(tTrB)

        # ---------------- 每个线程负责的两行：提前取 L2、D ----------------
        l_row = cute.make_rmem_tensor((2,), cutlass.Float32)
        d_row = cute.make_rmem_tensor((2,), cutlass.Float32)
        for h in cutlass.range_constexpr(2):
            r = bidx * BQ_ + tCcS[(0, h), 0, 0][0]
            l_row[h] = cutlass.Float32(0.0)
            d_row[h] = cutlass.Float32(0.0)
            if r < Nq:
                l_row[h] = gL[r]
                d_row[h] = gD[r]

        n_tiles = (Nk + BK_ - 1) // BK_
        if self.causal:
            n_tiles = cutlass.min(n_tiles, ((bidx + 1) * BQ_ + BK_ - 1) // BK_)

        # ---------------- 序言：Q、dO（临时放 stage 1），K0、V0 放 stage 0 ----------------
        self._g2s(tiled_copy_g2s, tidx, gQ, sQ, cQ, Nq)
        self._g2s(tiled_copy_g2s, tidx, gdO, sdO, cQ, Nq)
        self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcK, (BK_, D_), (0, 0)),
                  sK[(None, None, 0)], cute.local_tile(idK, (BK_, D_), (0, 0)), Nk)
        self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcV, (BK_, D_), (0, 0)),
                  sV[(None, None, 0)], cute.local_tile(idK, (BK_, D_), (0, 0)), Nk)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        for k in cutlass.range_constexpr(KD):
            cute.copy(s2r_A, tQsA[None, None, k], tQrA_v[None, None, k])
            cute.copy(s2r_A, tdOsA[None, None, k], tdOrA_v[None, None, k])

        # ---------------- 主循环：沿 K/V tile ----------------
        for j in cutlass.range(n_tiles, unroll=1):
            stage = j % STG
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if j + 1 < n_tiles:
                nxt = (j + 1) % STG
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcK, (BK_, D_), (j + 1, 0)),
                          sK[(None, None, nxt)], cute.local_tile(idK, (BK_, D_), (j + 1, 0)), Nk)
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcV, (BK_, D_), (j + 1, 0)),
                          sV[(None, None, nxt)], cute.local_tile(idK, (BK_, D_), (j + 1, 0)), Nk)
            cute.arch.cp_async_commit_group()

            sK_s = sK[(None, None, stage)]
            sV_s = sV[(None, None, stage)]
            tBsK = thr_B.partition_S(sK_s)
            tBsV = thr_B.partition_S(sV_s)
            tTsK = thr_Bt.partition_S(self._vt(sK_s))

            # (1) S = Q K^T，dP = dO V^T
            acc_s.fill(0.0)
            for k in cutlass.range_constexpr(KD):
                cute.copy(s2r_B, tBsK[None, None, k], tBrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_s, tQrA[None, None, k], tBrB[None, None, k], acc_s)
            acc_dp.fill(0.0)
            for k in cutlass.range_constexpr(KD):
                cute.copy(s2r_B, tBsV[None, None, k], tBrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dp, tdOrA[None, None, k], tBrB[None, None, k], acc_dp)

            # (2) P = exp2(S*scale - L2[row])，dS = P * (dP - D[row])；L2、D 沿“行”变化
            for h in cutlass.range_constexpr(2):
                row_g = bidx * BQ_ + tCcS[(0, h), 0, 0][0]
                for n in cutlass.range_constexpr(NS):
                    for e in cutlass.range_constexpr(2):
                        col_g = j * BK_ + tCcS[(e, h), 0, n][1]
                        p = cute.math.exp2(acc_s[(e, h), 0, n] * qk_scale - l_row[h], fastmath=True)
                        masked = col_g >= Nk
                        if self.causal:
                            masked = masked | (col_g > row_g)
                        if masked:
                            p = cutlass.Float32(0.0)
                        acc_dp[(e, h), 0, n] = p * (acc_dp[(e, h), 0, n] - d_row[h])

            # (3) dS：C 片段 → A 片段
            for kb in cutlass.range_constexpr(KK):
                for kh in cutlass.range_constexpr(2):
                    for h in cutlass.range_constexpr(2):
                        for e in cutlass.range_constexpr(2):
                            tdSrA[(e, h, kh), 0, kb] = acc_dp[(e, h), 0, 2 * kb + kh].to(Q.element_type)

            # (4) dQ += dS K
            for k in cutlass.range_constexpr(KK):
                cute.copy(s2r_Bt, tTsK[None, None, k], tTrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dq, tdSrA[None, None, k], tTrB[None, None, k], acc_dq)

        # ---------------- 收尾：dQ *= 1/√d，转 bf16，按行写回 ----------------
        tCgdQ = thr_mma.partition_C(cute.local_tile(gsrcdQ, (BQ_, D_), (bidx, 0)))
        for h in cutlass.range_constexpr(2):
            row = bidx * BQ_ + tCcO[(0, h), 0, 0][0]
            if row < Nq:
                for n in cutlass.range_constexpr(NO):
                    for e in cutlass.range_constexpr(2):
                        tCgdQ[(e, h), 0, n] = (acc_dq[(e, h), 0, n] * inv_sqrt).to(dQ.element_type)

    @cute.jit
    def __call__(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, dO: cute.Tensor,
                 L: cute.Tensor, Dv: cute.Tensor, dQ: cute.Tensor):
        assert Q.element_type == cutlass.BFloat16
        B, Nq, Nk = Q.shape[0], Q.shape[1], K.shape[1]
        q3, k3, v3 = self._tensors3(Q, Nq), self._tensors3(K, Nk), self._tensors3(V, Nk)
        do3, dq3 = self._tensors3(dO, Nq), self._tensors3(dQ, Nq)
        l2 = cute.make_tensor(L.iterator, cute.make_layout((B, Nq), stride=(Nq, 1)))
        d2 = cute.make_tensor(Dv.iterator, cute.make_layout((B, Nq), stride=(Nq, 1)))

        smem_bytes = 2 * STG * self.BK * self.D * 2 + 1024
        self.kernel(q3, k3, v3, do3, l2, d2, dq3, self._make_g2s(Q.element_type), self._make_mma()).launch(
            grid=((Nq + self.BQ - 1) // self.BQ, B, 1),
            block=(self.NT, 1, 1),
            smem=smem_bytes)


# =============================================================================
# host 入口（与 flash_bwd_tiled_torch 同签名，可直接 set_backward_impl）
# =============================================================================
_bwd_cache = {}


def _prep(t):
    t = t.detach().contiguous()
    if t.data_ptr() % 16 != 0:
        t = t.clone()
    return t


def flash_bwd_cute(Q, K, V, O, L2, dO, is_causal=False, *, need=(True, True, True)):
    """返回 (dQ, dK, dV)；need 为 False 的项返回 None。L2 是前向保存的 base-2 LSE。"""
    import cutlass.cute as cute_
    from cutlass.cute.runtime import from_dlpack

    need_q, need_k, need_v = need
    Q, K, V, O, dO = (_prep(x) for x in (Q, K, V, O, dO))
    L2 = _prep(L2)
    B, Nq, _ = Q.shape
    Nk = K.shape[1]

    # 5a：D = rowsum(dO ∘ O)，先用 torch 算（fp32）
    Dvec = (dO.float() * O.float()).sum(-1).contiguous()           # (B, Nq)

    def c(t, leading_dim):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)

    dQ = dK = dV = None
    causal = bool(is_causal)

    if need_k or need_v:                      # dK/dV kernel 同时算 dK、dV；两者之一不需要也无额外代价
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        args = (c(Q, 2), c(K, 2), c(V, 2), c(dO, 2), c(L2, 1), c(Dvec, 1), c(dK, 2), c(dV, 2))
        key = ("dkdv", causal)
        fn = _bwd_cache.get(key)
        if fn is None:
            fn = cute_.compile(FlashBwdDKDV(causal=causal), *args)
            _bwd_cache[key] = fn
        fn(*args)
        if not need_k:
            dK = None
        if not need_v:
            dV = None

    if need_q:
        dQ = torch.empty_like(Q)
        args = (c(Q, 2), c(K, 2), c(V, 2), c(dO, 2), c(L2, 1), c(Dvec, 1), c(dQ, 2))
        key = ("dq", causal)
        fn = _bwd_cache.get(key)
        if fn is None:
            fn = cute_.compile(FlashBwdDQ(causal=causal), *args)
            _bwd_cache[key] = fn
        fn(*args)

    return dQ, dK, dV


# =============================================================================
# 单 kernel 版（FA2 风格）：block 负责一个 K/V tile，循环 Q tile
#   dK、dV：寄存器累加；dQ：dS 经 smem 做 GEMM，再对 fp32 的 dQ_accum 做 atomicAdd
# =============================================================================
class FlashBwdFused(_BwdBase):

    def __init__(self, causal: bool = False, use_swizzle: bool = True,
                 with_dq: bool = True, num_threads: int = 128):
        super().__init__(causal, use_swizzle, num_threads)
        self.with_dq = with_dq                    # False：只算 dK、dV（need 里不要 dQ 时）
        assert BQ == D, "sdS 复用 (rows, D) 的 swizzle layout，要求 BQ == D == 64"

    @cute.kernel
    def kernel(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, dO: cute.Tensor,
               L: cute.Tensor, Dv: cute.Tensor, dK: cute.Tensor, dV: cute.Tensor,
               dQacc: cute.Tensor, tiled_copy_g2s: cute.TiledCopy, tiled_mma: cute.TiledMma):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()          # bidx: K/V tile 号；bidy: batch
        BQ_, BK_, D_ = self.BQ, self.BK, self.D
        qk_scale = cutlass.Float32(self.qk_scale)
        inv_sqrt = cutlass.Float32(self.inv_sqrt)

        gsrcQ = Q[(bidy, None, None)]
        gsrcK = K[(bidy, None, None)]
        gsrcV = V[(bidy, None, None)]
        gsrcdO = dO[(bidy, None, None)]
        gsrcdK = dK[(bidy, None, None)]
        gsrcdV = dV[(bidy, None, None)]
        gdQ = dQacc[(bidy, None, None)]                # (Nq, D) fp32
        gL = L[(bidy, None)]
        gD = Dv[(bidy, None)]
        Nq = gsrcQ.shape[0]
        Nk = gsrcK.shape[0]

        # ---------------- smem ----------------
        # Q、dO 各 2 stage；sK 常驻（dQ 的 B 操作数每轮要读）；
        # sdS：存 dS^T (kv, q)；序言里临时借给 V（读进寄存器之后才会被 dS 覆盖）
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(Q.element_type, self._smem_layout(BQ_, STG), 1024)
        sdO = smem.allocate_tensor(dO.element_type, self._smem_layout(BQ_, STG), 1024)
        sK = smem.allocate_tensor(K.element_type, self._smem_layout(BK_), 1024)
        sdS = smem.allocate_tensor(Q.element_type, self._smem_layout(BK_), 1024)
        sV = sdS

        idQ = cute.make_identity_tensor(gsrcQ.shape)
        idK = cute.make_identity_tensor(gsrcK.shape)
        gK = cute.local_tile(gsrcK, (BK_, D_), (bidx, 0))
        gV = cute.local_tile(gsrcV, (BK_, D_), (bidx, 0))
        cK = cute.local_tile(idK, (BK_, D_), (bidx, 0))

        # ---------------- MMA 划分：dK/dV 部分的 M 是 K 行 ----------------
        thr_mma = tiled_mma.get_slice(tidx)
        tKrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sK))
        tVrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sV))
        tBrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ[(None, None, 0)]))
        tTrB = tiled_mma.make_fragment_B(thr_mma.partition_B(self._vt(sdO[(None, None, 0)])))

        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, BQ_)), cutlass.Float32)
        acc_dp = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, BQ_)), cutlass.Float32)
        acc_dk = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, D_)), cutlass.Float32)
        acc_dv = cute.make_rmem_tensor(thr_mma.partition_shape_C((BK_, D_)), cutlass.Float32)
        acc_dk.fill(0.0)
        acc_dv.fill(0.0)
        tPrA = cute.make_rmem_tensor(thr_mma.partition_shape_A((BK_, BQ_)), Q.element_type)
        tdSrA = cute.make_rmem_tensor(thr_mma.partition_shape_A((BK_, BQ_)), Q.element_type)

        tCcS = thr_mma.partition_C(cute.make_identity_tensor((BK_, BQ_)))   # (kv 行, q 列)
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((BK_, D_)))    # BQ==BK：也用作 dQ 的 (q 行, d 列)

        NS = cute.size(acc_s, mode=[2])
        NO = cute.size(acc_dk, mode=[2])
        KD = cute.size(tKrA, mode=[2])
        KQ = cute.size(tPrA, mode=[2])

        atom_A, atom_At = self._ldmatrix_atoms(Q.element_type)
        s2r_A = cute.make_tiled_copy_A(atom_A, tiled_mma)
        s2r_B = cute.make_tiled_copy_B(atom_A, tiled_mma)
        s2r_Bt = cute.make_tiled_copy_B(atom_At, tiled_mma)
        thr_A, thr_B, thr_Bt = s2r_A.get_slice(tidx), s2r_B.get_slice(tidx), s2r_Bt.get_slice(tidx)
        tKsA, tKrA_v = thr_A.partition_S(sK), thr_A.retile(tKrA)
        tVsA, tVrA_v = thr_A.partition_S(sV), thr_A.retile(tVrA)
        tBrB_v = thr_B.retile(tBrB)
        tTrB_v = thr_Bt.retile(tTrB)

        # ---------------- dQ 部分的 fragment：M = Q 行，N = D，K = BK（kv） ----------------
        # 注意：必须无条件定义。动态循环里要用的变量不能在 if 分支里才第一次出现。
        sdSq = self._vt(sdS)          # (q, kv)：A 操作数 (m=q, k=kv)，m 连续 → ldmatrix.trans
        sKt = self._vt(sK)            # (d, kv)：B 操作数 (n=d, k=kv)，n 连续 → ldmatrix.trans
        tdQrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sdSq))      # ((2,2,2), 1, BK/16)
        tdQrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sKt))       # ((2,2), D/8, BK/16)
        acc_dq = cute.make_rmem_tensor(thr_mma.partition_shape_C((BQ_, D_)), cutlass.Float32)
        s2r_At = cute.make_tiled_copy_A(atom_At, tiled_mma)
        thr_At = s2r_At.get_slice(tidx)
        tdSsA, tdQrA_v = thr_At.partition_S(sdSq), thr_At.retile(tdQrA)
        tKsB, tdQrB_v = thr_Bt.partition_S(sKt), thr_Bt.retile(tdQrB)
        KK = cute.size(tdQrA, mode=[2])                                    # BK/16

        # ---------------- 本 block 的 Q tile 范围 ----------------
        n_total = (Nq + BQ_ - 1) // BQ_
        i_start = 0
        if self.causal:
            i_start = (bidx * BK_) // BQ_
        n_iter = cutlass.max(n_total - i_start, cutlass.Int32(0))

        # ---------------- 序言 ----------------
        self._g2s(tiled_copy_g2s, tidx, gK, sK, cK, Nk)
        self._g2s(tiled_copy_g2s, tidx, gV, sV, cK, Nk)
        if n_iter > 0:
            self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcQ, (BQ_, D_), (i_start, 0)),
                      sQ[(None, None, 0)], cute.local_tile(idQ, (BQ_, D_), (i_start, 0)), Nq)
            self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcdO, (BQ_, D_), (i_start, 0)),
                      sdO[(None, None, 0)], cute.local_tile(idQ, (BQ_, D_), (i_start, 0)), Nq)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        for k in cutlass.range_constexpr(KD):
            cute.copy(s2r_A, tKsA[None, None, k], tKrA_v[None, None, k])
            cute.copy(s2r_A, tVsA[None, None, k], tVrA_v[None, None, k])   # V 读完后 sdS 才会被 dS 覆盖

        # ---------------- 主循环 ----------------
        for t in cutlass.range(n_iter, unroll=1):
            qi = i_start + t
            stage = t % STG

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()      # ① 本 stage 数据可见；② 所有 warp 已用完上一轮的 sdS
            if t + 1 < n_iter:
                nxt = (t + 1) % STG
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcQ, (BQ_, D_), (qi + 1, 0)),
                          sQ[(None, None, nxt)], cute.local_tile(idQ, (BQ_, D_), (qi + 1, 0)), Nq)
                self._g2s(tiled_copy_g2s, tidx, cute.local_tile(gsrcdO, (BQ_, D_), (qi + 1, 0)),
                          sdO[(None, None, nxt)], cute.local_tile(idQ, (BQ_, D_), (qi + 1, 0)), Nq)
            cute.arch.cp_async_commit_group()

            sQ_s = sQ[(None, None, stage)]
            sdO_s = sdO[(None, None, stage)]
            tBsQ = thr_B.partition_S(sQ_s)
            tBsdO = thr_B.partition_S(sdO_s)
            tTsQ = thr_Bt.partition_S(self._vt(sQ_s))
            tTsdO = thr_Bt.partition_S(self._vt(sdO_s))

            # (1) S^T = K Q^T，dP^T = V dO^T
            acc_s.fill(0.0)
            for k in cutlass.range_constexpr(KD):
                cute.copy(s2r_B, tBsQ[None, None, k], tBrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_s, tKrA[None, None, k], tBrB[None, None, k], acc_s)
            acc_dp.fill(0.0)
            for k in cutlass.range_constexpr(KD):
                cute.copy(s2r_B, tBsdO[None, None, k], tBrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dp, tVrA[None, None, k], tBrB[None, None, k], acc_dp)

            # (2) P^T、dS^T（L2、D 沿列）
            q0 = qi * BQ_
            for n in cutlass.range_constexpr(NS):
                for e in cutlass.range_constexpr(2):
                    q_g = q0 + tCcS[(e, 0), 0, n][1]
                    l_val = cutlass.Float32(0.0)
                    d_val = cutlass.Float32(0.0)
                    if q_g < Nq:
                        l_val = gL[q_g]
                        d_val = gD[q_g]
                    for h in cutlass.range_constexpr(2):
                        k_g = bidx * BK_ + tCcS[(e, h), 0, n][0]
                        p = cute.math.exp2(acc_s[(e, h), 0, n] * qk_scale - l_val, fastmath=True)
                        masked = (q_g >= Nq) | (k_g >= Nk)     # 越界的 Q 行 / K 行
                        if self.causal:
                            masked = masked | (k_g > q_g)
                        if masked:
                            p = cutlass.Float32(0.0)
                        acc_s[(e, h), 0, n] = p
                        acc_dp[(e, h), 0, n] = p * (acc_dp[(e, h), 0, n] - d_val)

            # (3) P^T、dS^T：C 片段 → A 片段
            for kb in cutlass.range_constexpr(KQ):
                for kh in cutlass.range_constexpr(2):
                    for h in cutlass.range_constexpr(2):
                        for e in cutlass.range_constexpr(2):
                            tPrA[(e, h, kh), 0, kb] = acc_s[(e, h), 0, 2 * kb + kh].to(Q.element_type)
                            tdSrA[(e, h, kh), 0, kb] = acc_dp[(e, h), 0, 2 * kb + kh].to(Q.element_type)

            # (4) dV += P^T dO；dK += dS^T Q
            for k in cutlass.range_constexpr(KQ):
                cute.copy(s2r_Bt, tTsdO[None, None, k], tTrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dv, tPrA[None, None, k], tTrB[None, None, k], acc_dv)
            for k in cutlass.range_constexpr(KQ):
                cute.copy(s2r_Bt, tTsQ[None, None, k], tTrB_v[None, None, k])
                cute.gemm(tiled_mma, acc_dk, tdSrA[None, None, k], tTrB[None, None, k], acc_dk)

            # (5) dQ += dS K（经 smem 转置），再 atomicAdd 到 fp32 的 dQ_accum
            if self.with_dq:
                for h in cutlass.range_constexpr(2):
                    for n in cutlass.range_constexpr(NS):
                        for e in cutlass.range_constexpr(2):
                            r = tCcS[(e, h), 0, n][0]
                            c = tCcS[(e, h), 0, n][1]
                            sdS[(r, c)] = acc_dp[(e, h), 0, n].to(Q.element_type)   # 写 dS^T (kv, q)
                cute.arch.barrier()                                                  # dS 对所有 warp 可见

                acc_dq.fill(0.0)
                for k in cutlass.range_constexpr(KK):
                    cute.copy(s2r_At, tdSsA[None, None, k], tdQrA_v[None, None, k])   # A：ldmatrix.trans
                    cute.copy(s2r_Bt, tKsB[None, None, k], tdQrB_v[None, None, k])    # B：ldmatrix.trans
                    cute.gemm(tiled_mma, acc_dq, tdQrA[None, None, k], tdQrB[None, None, k], acc_dq)

                for h in cutlass.range_constexpr(2):
                    q_g = q0 + tCcO[(0, h), 0, 0][0]
                    if q_g < Nq:
                        for n in cutlass.range_constexpr(NO):
                            for e in cutlass.range_constexpr(2):
                                d_g = tCcO[(e, h), 0, n][1]
                                ptr = gdQ.iterator + cute.crd2idx((q_g, d_g), gdQ.layout)
                                cute.arch.atomic_add(ptr, acc_dq[(e, h), 0, n])

        # ---------------- 收尾：dK *= 1/√d；写 dK、dV（dQ 已在 dQ_accum 里，缩放在 host 做） ----------------
        tCgdK = thr_mma.partition_C(cute.local_tile(gsrcdK, (BK_, D_), (bidx, 0)))
        tCgdV = thr_mma.partition_C(cute.local_tile(gsrcdV, (BK_, D_), (bidx, 0)))
        for h in cutlass.range_constexpr(2):
            row = bidx * BK_ + tCcO[(0, h), 0, 0][0]
            if row < Nk:
                for n in cutlass.range_constexpr(NO):
                    for e in cutlass.range_constexpr(2):
                        tCgdK[(e, h), 0, n] = (acc_dk[(e, h), 0, n] * inv_sqrt).to(dK.element_type)
                        tCgdV[(e, h), 0, n] = acc_dv[(e, h), 0, n].to(dV.element_type)

    @cute.jit
    def __call__(self, Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, dO: cute.Tensor,
                 L: cute.Tensor, Dv: cute.Tensor, dK: cute.Tensor, dV: cute.Tensor,
                 dQacc: cute.Tensor):
        assert Q.element_type == cutlass.BFloat16
        B, Nq, Nk, Nacc = Q.shape[0], Q.shape[1], K.shape[1], dQacc.shape[1]
        q3, k3, v3 = self._tensors3(Q, Nq), self._tensors3(K, Nk), self._tensors3(V, Nk)
        do3, dk3, dv3 = self._tensors3(dO, Nq), self._tensors3(dK, Nk), self._tensors3(dV, Nk)
        acc3 = self._tensors3(dQacc, Nacc)                       # fp32，(B, Nq, D)；with_dq=False 时是 (B,1,D) 的占位
        l2 = cute.make_tensor(L.iterator, cute.make_layout((B, Nq), stride=(Nq, 1)))
        d2 = cute.make_tensor(Dv.iterator, cute.make_layout((B, Nq), stride=(Nq, 1)))

        # Q(2 stage) + dO(2 stage) + sK + sdS，再加 1KB 对齐余量
        smem_bytes = (2 * STG * self.BQ + 2 * self.BK) * self.D * 2 + 1024
        self.kernel(q3, k3, v3, do3, l2, d2, dk3, dv3, acc3,
                    self._make_g2s(Q.element_type), self._make_mma()).launch(
            grid=((Nk + self.BK - 1) // self.BK, B, 1),
            block=(self.NT, 1, 1),
            smem=smem_bytes)


def flash_bwd_cute_fused(Q, K, V, O, L2, dO, is_causal=False, *, need=(True, True, True)):
    """单 kernel 版；签名与 flash_bwd_cute 一致，可直接 set_backward_impl。"""
    import cutlass.cute as cute_
    from cutlass.cute.runtime import from_dlpack

    need_q, need_k, need_v = need
    Q, K, V, O, dO = (_prep(x) for x in (Q, K, V, O, dO))
    L2 = _prep(L2)
    B, Nq, _ = Q.shape
    Nk = K.shape[1]

    Dvec = (dO.float() * O.float()).sum(-1).contiguous()            # D = rowsum(dO ∘ O)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    if need_q:
        dQacc = torch.zeros(B, Nq, D, dtype=torch.float32, device=Q.device)   # 必须清零：kernel 只做 atomicAdd
    else:
        dQacc = torch.zeros(B, 1, D, dtype=torch.float32, device=Q.device)    # 占位，kernel 不会访问

    def c(t, leading_dim):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)

    args = (c(Q, 2), c(K, 2), c(V, 2), c(dO, 2), c(L2, 1), c(Dvec, 1), c(dK, 2), c(dV, 2), c(dQacc, 2))
    causal = bool(is_causal)
    key = ("fused", causal, bool(need_q))
    fn = _bwd_cache.get(key)
    if fn is None:
        fn = cute_.compile(FlashBwdFused(causal=causal, with_dq=bool(need_q)), *args)
        _bwd_cache[key] = fn
    fn(*args)

    dQ = (dQacc * (1.0 / math.sqrt(D))).to(Q.dtype) if need_q else None
    return dQ, (dK if need_k else None), (dV if need_v else None)