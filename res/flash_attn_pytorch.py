"""FlashAttention 纯 PyTorch 参考实现：M0 / M1 / M2。

约定（与 tests/adapters.py 对齐）：
  Q, K, V : (B, N, d)
  O       : 与输入同 dtype
  L       : (B, N) fp32（官方契约：logsumexp 存 fp32；自然对数、含 1/√d、causal 时只对 j <= i 求和）
  内部累加精度 compute_dtype = promote_types(Q.dtype, fp32)：
      bf16/fp16/fp32 输入 → fp32；float64 输入 → float64（gradcheck 的前提）

自检：
  python res/flash_attn_pytorch.py
"""
import math

import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor


# ================================================================ M0：朴素实现（金标准之一）

def naive_attention(
    Q: Float[Tensor, "batch seq d"],
    K: Float[Tensor, "batch seq d"],
    V: Float[Tensor, "batch seq d"],
    is_causal: bool = False,
) -> tuple[Float[Tensor, "batch seq d"], Float[Tensor, "batch seq seq"]]:
    """朴素前向：物化全部 N×N 中间量。返回 (O, P)。"""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    S = scale * einsum(Q, K, "b i d, b k d -> b i k")
    if is_causal:
        N = Q.shape[-2]
        mask: Bool[Tensor, "seq seq"] = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=Q.device), diagonal=1
        )
        S = S.masked_fill(mask, float("-inf"))

    P = torch.softmax(S, dim=-1)
    O = einsum(P, V, "b i k, b k d -> b i d")
    return O, P


class NaiveAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, P = naive_attention(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, P)               # M0 特例：存 P；M1 起换成 (O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, P = ctx.saved_tensors
        scale = 1.0 / math.sqrt(Q.shape[-1])

        dV = einsum(P, dO, "b i k, b i d -> b k d")              # Pᵀ dO：沿 i 求和
        dP = einsum(dO, V, "b i d, b k d -> b i k")              # dO Vᵀ：沿 d 求和
        dS = P * (dP - reduce(P * dP, "b i k -> b i 1", "sum"))  # softmax 雅可比
        dQ = scale * einsum(dS, K, "b i k, b k d -> b i d")
        dK = scale * einsum(dS, Q, "b i k, b i d -> b k d")      # dSᵀ Q：沿 i 求和
        return dQ, dK, dV, None


# ================================================================ M1：tiled forward

def flash_fwd_tiled(
    Q: Float[Tensor, "batch seq d"],
    K: Float[Tensor, "batch seq d"],
    V: Float[Tensor, "batch seq d"],
    is_causal: bool = False,
    Bq: int = 64,
    Bk: int = 64,
) -> tuple[Float[Tensor, "batch seq d"], Float[Tensor, "batch seq"]]:
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    compute_dtype = torch.promote_types(Q.dtype, torch.float32)
    dev = Q.device
    O = torch.empty_like(Q)
    L = torch.empty(B, N, dtype=torch.float32, device=dev)

    for qs in range(0, N, Bq):                          # 外层：Q 的 tile（M3 里对应并行的 block）
        qe = min(qs + Bq, N)
        Qt  = Q[:, qs:qe].to(compute_dtype)
        m   = torch.full((B, qe - qs), float("-inf"), dtype=compute_dtype, device=dev)
        l   = torch.zeros((B, qe - qs), dtype=compute_dtype, device=dev)
        acc = torch.zeros((B, qe - qs, d), dtype=compute_dtype, device=dev)

        for ks in range(0, N, Bk):                      # 内层：K/V 的 tile（M3 里是 kernel 内循环）
            if is_causal and ks > qe - 1:               # 整个 tile 都在对角线之上：全被 mask，跳过
                break
            ke = min(ks + Bk, N)
            Kj = K[:, ks:ke].to(compute_dtype)
            Vj = V[:, ks:ke].to(compute_dtype)

            S = scale * einsum(Qt, Kj, "b q d, b k d -> b q k")
            if is_causal:                               # mask 必须用【全局】下标
                q_idx: Int[Tensor, "q 1"] = torch.arange(qs, qe, device=dev)[:, None]
                k_idx: Int[Tensor, "1 k"] = torch.arange(ks, ke, device=dev)[None, :]
                S = S.masked_fill(q_idx < k_idx, float("-inf"))

            m_new = torch.maximum(m, reduce(S, "b q k -> b q", "max"))   # 运行最大值：不能丢 maximum！
            # 全 mask 的 tile 会让 m_new = -inf；守护减法参考值，避免 exp(-inf - (-inf)) = nan：
            m_ref = torch.where(m_new == float("-inf"), torch.zeros_like(m_new), m_new)
            P     = torch.exp(S - rearrange(m_ref, "b q -> b q 1"))
            alpha = torch.exp(m - m_ref)
            l     = alpha * l + reduce(P, "b q k -> b q", "sum")
            acc   = rearrange(alpha, "b q -> b q 1") * acc + einsum(P, Vj, "b q k, b k d -> b q d")
            m     = m_new


        O[:, qs:qe] = (acc / rearrange(l, "b q -> b q 1")).to(Q.dtype)
        L[:, qs:qe] = m + torch.log(l)                  # logsumexp
    return O, L


# ================================================================ M2：tiled backward

def flash_bwd_tiled(
    Q: Float[Tensor, "batch seq d"],
    K: Float[Tensor, "batch seq d"],
    V: Float[Tensor, "batch seq d"],
    O: Float[Tensor, "batch seq d"],
    L: Float[Tensor, "batch seq"],
    dO: Float[Tensor, "batch seq d"],
    is_causal: bool = False,
    Bq: int = 64,
    Bk: int = 64,
) -> tuple[
    Float[Tensor, "batch seq d"],
    Float[Tensor, "batch seq d"],
    Float[Tensor, "batch seq d"],
]:
    """Tiled FlashAttention-2 backward。

    输入：
      Q, K, V : (B, N, d)   M1 forward 保存的原始输入
      O       : (B, N, d)   M1 forward 的输出
      L       : (B, N)      M1 forward 的 logsumexp（fp32，含 1/√d，自然对数）
      dO      : (B, N, d)   输出梯度
    输出：
      dQ, dK, dV : 各 (B, N, d)，dtype 与输入一致
    """
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    dev = Q.device
    compute_dtype = torch.promote_types(Q.dtype, torch.float32)

    D = einsum(dO.to(compute_dtype), O.to(compute_dtype), "b q d, b q d -> b q")  # rowsum(dO⊙O)

    dQ = torch.zeros(B, N, d, dtype=compute_dtype, device=dev)
    dK = torch.zeros(B, N, d, dtype=compute_dtype, device=dev)
    dV = torch.zeros(B, N, d, dtype=compute_dtype, device=dev)

    for qs in range(0, N, Bq):                          # 外层：Q tile
        qe = min(qs + Bq, N)
        Qt  = Q[:, qs:qe].to(compute_dtype)
        dOt = dO[:, qs:qe].to(compute_dtype)
        Lt  = L[:, qs:qe].to(compute_dtype)             # fp32 的 L 升到 compute_dtype 再参与运算
        Dt  = D[:, qs:qe]

        for ks in range(0, N, Bk):                      # 内层：K/V tile
            if is_causal and ks >= qe:                  # k 列全部 > q 行：整块 mask，跳过
                break
            ke = min(ks + Bk, N)
            Kj = K[:, ks:ke].to(compute_dtype)
            Vj = V[:, ks:ke].to(compute_dtype)

            S = scale * einsum(Qt, Kj, "b q d, b k d -> b q k")
            if is_causal:
                q_idx: Int[Tensor, "q 1"] = torch.arange(qs, qe, device=dev)[:, None]
                k_idx: Int[Tensor, "1 k"] = torch.arange(ks, ke, device=dev)[None, :]
                S = S.masked_fill(q_idx < k_idx, float("-inf"))

            P = torch.exp(S - rearrange(Lt, "b q -> b q 1"))     # 用保存的 L 重算 P

            dV[:, ks:ke] += einsum(P, dOt, "b q k, b q d -> b k d")
            dP = einsum(dOt, Vj, "b q d, b k d -> b q k")
            dS = P * (dP - rearrange(Dt, "b q -> b q 1"))
            dK[:, ks:ke] += scale * einsum(dS, Qt, "b q k, b q d -> b k d")
            dQ[:, qs:qe] += scale * einsum(dS, Kj, "b q k, b k d -> b q d")

    return dQ.to(Q.dtype), dK.to(Q.dtype), dV.to(Q.dtype)


# ================================================================ autograd 接口（两轮测试共用）

class FlashAttnPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, Bq=64, Bk=64):
        O, L = flash_fwd_tiled(Q, K, V, is_causal, Bq, Bk)
        ctx.save_for_backward(Q, K, V, O, L)            # 只存这五个，不存 P
        ctx.is_causal = is_causal
        ctx.tile = (Bq, Bk)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        Bq, Bk = ctx.tile
        dQ, dK, dV = flash_bwd_tiled(Q, K, V, O, L, dO, ctx.is_causal, Bq, Bk)
        return dQ, dK, dV, None, None, None             # 个数 = forward 除 ctx 外的参数个数


# ================================================================ 自检

def _reference(Q, K, V, causal):
    """float64 全量计算：O 与 L。"""
    N, d = Q.shape[1], Q.shape[2]
    S = (Q.double() @ K.double().transpose(-2, -1)) / math.sqrt(d)
    if causal:
        S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=Q.device), 1), float("-inf"))
    return torch.softmax(S, -1) @ V.double(), torch.logsumexp(S, -1)


def run_checks(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # fp32：全 tile 组合 × 边界形状，与 fp64 参考对齐；换 tile 不改变结果
    tiles = [(16, 16), (32, 16), (16, 64), (64, 32), (128, 128), (7, 5)]
    worst, count = 0.0, 0
    for N in (1, 16, 17, 100, 128, 1000):
        for d in (16, 64):
            for causal in (False, True):
                Q, K, V = (torch.randn(2, N, d, device=device) for _ in range(3))
                O_ref, L_ref = _reference(Q, K, V, causal)
                outs = []
                for Bq, Bk in tiles:
                    O, L = flash_fwd_tiled(Q, K, V, causal, Bq, Bk)
                    outs.append(O)
                    err = max((O.double() - O_ref).abs().max().item(),
                              (L.double() - L_ref).abs().max().item())
                    assert err < 1e-4, f"N={N} d={d} causal={causal} tile=({Bq},{Bk}) err={err:.2e}"
                    worst, count = max(worst, err), count + 1
                spread = max((o - outs[0]).abs().max().item() for o in outs)
                assert spread < 1e-4, f"tile 改变了结果：N={N} d={d} causal={causal} spread={spread:.2e}"
    print(f"[fp32] {count} 个组合通过，最大误差 {worst:.2e}；换 tile 不改变结果")

    # fp64：计算不得降精度（gradcheck 的前提）；L 按契约仍存 fp32
    Q, K, V = (torch.randn(2, 100, 16, device=device, dtype=torch.float64) for _ in range(3))
    for causal in (False, True):
        O_ref, L_ref = _reference(Q, K, V, causal)
        O, L = flash_fwd_tiled(Q, K, V, causal, 16, 32)
        assert O.dtype == torch.float64, f"fp64 输入时 O 不得降精度，实际 {O.dtype}"
        e = (O - O_ref).abs().max().item()
        eL = (L.double() - L_ref).abs().max().item()
        assert e < 1e-10 and eL < 1e-6, (causal, e, eL)   # L 是 fp32，容差放宽到 1e-6
    print("[fp64] 前向保持 float64（L 按契约 fp32），误差达标")

    # M0 vs M1+M2：两套独立实现的 O 与梯度必须一致
    for causal in (False, True):
        Q0, K0, V0 = (torch.randn(2, 100, 32, device=device, requires_grad=True) for _ in range(3))
        Q2, K2, V2 = (t.detach().clone().requires_grad_() for t in (Q0, K0, V0))
        dO = torch.randn(2, 100, 32, device=device)
        NaiveAttentionFn.apply(Q0, K0, V0, causal).backward(dO)
        FlashAttnPytorch.apply(Q2, K2, V2, causal).backward(dO)
        for a, b, n in ((Q0.grad, Q2.grad, "dQ"), (K0.grad, K2.grad, "dK"), (V0.grad, V2.grad, "dV")):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4, msg=lambda m, x=n: f"{x}: {m}")
    print("[M0 vs M1+M2] O 与 dQ/dK/dV 一致")

    # 协议：dtype 透传
    Q, K, V = (torch.randn(1, 64, 16, device=device, requires_grad=True) for _ in range(3))
    O = FlashAttnPytorch.apply(Q, K, V, True)
    assert O.dtype == Q.dtype and O.shape == Q.shape
    print("全部通过")


if __name__ == "__main__":
    run_checks()
