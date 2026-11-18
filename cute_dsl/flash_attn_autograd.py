"""CuTe FlashAttention 前向 + 可切换的反向，封装成 torch.autograd.Function。

约定：
  - Q/K/V: (B, N, 64) bf16、CUDA；缩放固定 1/sqrt(64)；
  - L（保存给反向的 LSE）是 base-2：L2 = logsumexp(QK^T/√d) * log2(e)，shape (B, Nq) fp32；
  - 反向用 P = exp2(S*qk_scale - L2) 重算，不存 P。
CuTe 相关的 import 都放在 flash_fwd_cute 里（惰性），所以反向相关测试在没有 cutlass 的机器上也能跑。

前向 kernel 的构造参数（reverse_blocks / ilp_reduce / lazy_rescale / alias_q / smem_slack / bq / bk / stages）
可以不改代码就切换，两种方式：
  1) 环境变量：FA_FWD_KW="alias_q=1,smem_slack=0"（布尔项用 0/1，其余是整数），用于 ncu 等命令行场景；
  2) 代码里调用 set_fwd_config(alias_q=True, smem_slack=0)（会清空编译缓存）。
"""
import math
import os

import torch
from torch.autograd.function import once_differentiable

HEAD_DIM = 64
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)

_fwd_cache = {}       # {(is_causal, 构造参数): 已编译的 kernel}

# 这几个构造参数是布尔型，其余（smem_slack、bq、bk、stages）是整数
_BOOL_KEYS = {"reverse_blocks", "ilp_reduce", "lazy_rescale", "alias_q", "use_swizzle", "use_pred", "zero_fill"}


def _parse_env_kw():
    """解析 FA_FWD_KW，例如 'alias_q=1,smem_slack=0'。"""
    kw = {}
    for item in filter(None, os.environ.get("FA_FWD_KW", "").split(",")):
        k, v = (x.strip() for x in item.split("="))
        kw[k] = bool(int(v)) if k in _BOOL_KEYS else int(v)
    return kw


_FWD_KW = _parse_env_kw()     # 传给 FLASH_ATTN_naive 的额外构造参数


def set_fwd_config(**kw):
    """设置前向 kernel 的构造参数；会清空编译缓存，下次调用时重新编译。set_fwd_config() 恢复默认。"""
    global _FWD_KW
    _FWD_KW = dict(kw)
    _fwd_cache.clear()


def _fwd_fn(is_causal, args):
    """取（必要时编译）当前配置对应的前向 kernel。"""
    key = (bool(is_causal), tuple(sorted(_FWD_KW.items())))
    fn = _fwd_cache.get(key)
    if fn is None:
        import cutlass.cute as cute
        from flash_attn_naive import FLASH_ATTN_naive
        fn = cute.compile(FLASH_ATTN_naive(causal=bool(is_causal), **_FWD_KW), *args)
        _fwd_cache[key] = fn
    return fn


# =============================================================================
# 输入检查 / 准备
# =============================================================================
def _check_inputs(Q, K, V):
    for name, t in (("Q", Q), ("K", K), ("V", V)):
        if t.dim() != 3:
            raise ValueError(f"{name} 必须是 (B, N, D)，实际 {tuple(t.shape)}")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} 必须是 bfloat16，实际 {t.dtype}")
        if not t.is_cuda:
            raise ValueError(f"{name} 必须在 CUDA 上")
        if t.shape[-1] != HEAD_DIM:
            raise ValueError(f"{name} 的 head_dim 必须是 {HEAD_DIM}，实际 {t.shape[-1]}")
    if not (Q.shape[0] == K.shape[0] == V.shape[0]):
        raise ValueError("Q/K/V 的 batch 不一致")
    if K.shape[1] != V.shape[1]:
        raise ValueError("K 与 V 的序列长度必须相同")
    if Q.shape[1] == 0 or K.shape[1] == 0:
        raise ValueError("序列长度必须 > 0")


def _prep(t):
    """detach（dlpack 不能导出 requires_grad 的张量）→ 连续 → 16B 对齐。
    非连续或起始地址未 16B 对齐的视图会被拷贝。"""
    t = t.detach().contiguous()
    if t.data_ptr() % 16 != 0:
        t = t.clone()
    return t


def _fwd_args(Q, K, V):
    """准备前向的输入、输出缓冲与 cute 张量。返回 (args, O, L)。"""
    from cutlass.cute.runtime import from_dlpack
    Q, K, V = _prep(Q), _prep(K), _prep(V)
    B, Nq, _ = Q.shape
    O = torch.empty_like(Q)
    L = torch.empty(B, Nq, dtype=torch.float32, device=Q.device)

    def c(t, leading_dim):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)

    return (c(Q, 2), c(K, 2), c(V, 2), c(O, 2), c(L, 1)), O, L


# =============================================================================
# 前向：CuTe kernel
# =============================================================================
def flash_fwd_cute(Q, K, V, is_causal=False):
    """返回 (O, L2)。O: (B,Nq,D) bf16；L2: (B,Nq) fp32，base-2 LSE。"""
    _check_inputs(Q, K, V)
    args, O, L = _fwd_args(Q, K, V)
    fn = _fwd_fn(is_causal, args)
    # 注意：这里按默认流启动。若你在自定义 CUDA stream 里使用，需要给 kernel 的 __call__ 增加 stream 参数。
    fn(*args)
    return O, L


def make_fwd_runner(Q, K, V, is_causal=False):
    """返回 (run, O, L)。run() 只发射 kernel，不含 from_dlpack / 校验 / 分配，用来测纯 kernel 时间。"""
    _check_inputs(Q, K, V)
    args, O, L = _fwd_args(Q, K, V)
    fn = _fwd_fn(is_causal, args)
    return (lambda: fn(*args)), O, L


# =============================================================================
# 反向：torch 版 FA2 分块重算（不存 P）
# =============================================================================
def flash_bwd_tiled_torch(Q, K, V, O, L2, dO, is_causal=False, *,
                          block_q=128, block_k=128, need=(True, True, True)):
    """
    S = QK^T/√d，P = softmax(S)，P 用 L2 重算：P = exp2(S*(log2e/√d) - L2)
    dV = P^T dO；dP = dO V^T；dS = P ∘ (dP - rowsum(dO∘O))；dQ = dS K/√d；dK = dS^T Q/√d
    外层沿 K 块、内层沿 Q 块；dQ 在 fp32 中累加。float64 输入时全程 float64（用于算法验证）。
    causal：左上对齐，保留 j <= i。
    """
    cdt = torch.float64 if Q.dtype == torch.float64 else torch.float32
    B, Nq, Dh = Q.shape
    Nk = K.shape[1]
    inv_sqrt = 1.0 / math.sqrt(Dh)
    qk_scale = inv_sqrt * LOG2E
    need_q, need_k, need_v = need

    Qf, Kf, Vf, dOf = (x.to(cdt) for x in (Q, K, V, dO))
    L2 = L2.to(cdt)
    Dvec = (dOf * O.to(cdt)).sum(-1)                      # (B, Nq)：rowsum(dO ∘ O)

    dQ = torch.zeros_like(Qf) if need_q else None
    dK = torch.zeros_like(Kf) if need_k else None
    dV = torch.zeros_like(Vf) if need_v else None
    neg_inf = float("-inf")

    for j0 in range(0, Nk, block_k):
        j1 = min(j0 + block_k, Nk)
        Kj, Vj = Kf[:, j0:j1], Vf[:, j0:j1]
        dKj = torch.zeros_like(Kj) if need_k else None
        dVj = torch.zeros_like(Vj) if need_v else None

        # causal：行 i 只看 j<=i，所以行号 < j0 的 Q 块整块被 mask，直接跳过
        i_start = (j0 // block_q) * block_q if is_causal else 0
        for i0 in range(i_start, Nq, block_q):
            i1 = min(i0 + block_q, Nq)
            if is_causal and i1 - 1 < j0:
                continue
            Qi, dOi = Qf[:, i0:i1], dOf[:, i0:i1]
            Li, Di = L2[:, i0:i1], Dvec[:, i0:i1]

            S2 = (Qi @ Kj.transpose(-1, -2)) * qk_scale   # (B, bq, bk)，base-2 域
            if is_causal:
                rows = torch.arange(i0, i1, device=Q.device)[:, None]
                cols = torch.arange(j0, j1, device=Q.device)[None, :]
                S2 = S2.masked_fill(cols > rows, neg_inf)
            P = torch.exp2(S2 - Li[..., None])            # 被 mask 的位置 → 0

            if need_v:
                dVj += P.transpose(-1, -2) @ dOi
            if need_q or need_k:
                dP = dOi @ Vj.transpose(-1, -2)
                dS = P * (dP - Di[..., None])
                if need_q:
                    dQ[:, i0:i1] += (dS @ Kj) * inv_sqrt
                if need_k:
                    dKj += (dS.transpose(-1, -2) @ Qi) * inv_sqrt

        if need_k:
            dK[:, j0:j1] = dKj
        if need_v:
            dV[:, j0:j1] = dVj

    return (dQ.to(Q.dtype) if need_q else None,
            dK.to(K.dtype) if need_k else None,
            dV.to(V.dtype) if need_v else None)


# 反向实现可替换：签名 (Q,K,V,O,L2,dO,is_causal=False,*,need) -> (dQ,dK,dV)
_BWD_IMPL = flash_bwd_tiled_torch


def set_backward_impl(fn):
    global _BWD_IMPL
    _BWD_IMPL = fn


def use_reference_bwd(flash_bwd_tiled):
    """接入你自己的 flash_bwd_tiled(Q,K,V,O,L,dO,is_causal)。
    假定它的 L 是自然对数 LSE（L_nat = L2 * ln2）；若你的约定不同请自行调整。
    自定义实现的约定：签名 (Q,K,V,O,L2,dO,is_causal=False,*,need=(True,True,True)) -> (dQ,dK,dV)。"""
    def impl(Q, K, V, O, L2, dO, is_causal=False, *, need=(True, True, True)):
        return flash_bwd_tiled(Q, K, V, O, L2 * LN2, dO, is_causal)   # 你的实现不支持 need，三个梯度都会算
    set_backward_impl(impl)


def use_cute_bwd(fused: bool = False):
    """反向切换到 CuTe kernel：fused=False 两 kernel（确定性），True 单 kernel（dQ 用 atomicAdd）。"""
    from flash_attn_bwd import flash_bwd_cute, flash_bwd_cute_fused
    set_backward_impl(flash_bwd_cute_fused if fused else flash_bwd_cute)


def use_torch_bwd():
    set_backward_impl(flash_bwd_tiled_torch)


# =============================================================================
# autograd.Function
# =============================================================================
class FlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = flash_fwd_cute(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, O, L)         # 只存这五个，不存 P
        ctx.is_causal = bool(is_causal)
        return O

    @staticmethod
    @once_differentiable
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        need = tuple(ctx.needs_input_grad[:3])
        dQ, dK, dV = _BWD_IMPL(Q, K, V, O, L, dO.contiguous(), ctx.is_causal, need=need)
        return dQ, dK, dV, None


def flash_attn_func(q, k, v, causal=False):
    return FlashAttnFunction.apply(q, k, v, causal)