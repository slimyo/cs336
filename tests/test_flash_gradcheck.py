"""tests/test_flash_all.py — M0（朴素）/ M1+M2（FlashAttention PyTorch 版）的完整验收。

覆盖（按依赖顺序）：
  ① gradcheck（fp64）：
     - 小形状单 tile（N=8）—— 抓 autograd 协议与数学错误
     - 小 tile 强制多 tile（N=9，tile 不整除、Bq != Bk）—— 抓 online softmax 的
       rescale 与 backward 跨 tile 累加（Bq != Bk 时还有全 mask tile 的 -inf 陷阱）
  ② float32 vs SDPA：O 与 dQ/dK/dV，容差 1e-4；causal 两态、batch>1、N=100 非 2 的幂
  ③ tile 不变性 + 边界 N：换 (Bq,Bk) 不改变任何输出与梯度；N ∈ {1, 2, 17, 100, 129}
  ④ 保存的张量契约：恰好一个 (B,N) 的 L 且值正确；不保存 N×N（P/S）
  ⑤ 低精度 dtype：bf16/fp16，梯度 dtype 必须等于输入 dtype
  ⑥ 显存 O(N)：flash 线性、naive 平方

使用前提：FlashAttnPytorch.forward 支持 (ctx, Q, K, V, is_causal=False, Bq=64, Bk=64)，
backward 返回 (dQ, dK, dV, None, None, None)。接口没有 Bq/Bk 时，tile 相关测试会被
跳过（跳过 ≠ 通过：跑完确认输出里没有 tile 项被 skip）。

跑法：
  uv run pytest tests/test_flash_all.py -v                  # 全量
  uv run pytest tests/test_flash_all.py -v -k "not memory"  # 跳过慢的显存测试
"""
import inspect
import math

import pytest
import torch

from res.flash_attn_pytorch import FlashAttnPytorch, NaiveAttentionFn

# ---------------------------------------------------------------- 跳过条件与工具

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
HAS_TILES = {"Bq", "Bk"} <= set(inspect.signature(FlashAttnPytorch.forward).parameters)
needs_tiles = pytest.mark.skipif(not HAS_TILES, reason="forward 没有 Bq/Bk 参数")


def flash(Q, K, V, causal, tiles=None):
    """统一入口：tiles=None 用默认 tile，否则 FlashAttnPytorch.apply(Q, K, V, causal, Bq, Bk)。"""
    if tiles is None:
        return FlashAttnPytorch.apply(Q, K, V, causal)
    return FlashAttnPytorch.apply(Q, K, V, causal, *tiles)


def ref_attention(Q, K, V, causal):
    """float64 全量参考，返回 (O, L)，可微。"""
    N, d = Q.shape[1], Q.shape[2]
    S = (Q.double() @ K.double().transpose(-2, -1)) / math.sqrt(d)
    if causal:
        S = S.masked_fill(
            torch.triu(torch.ones(N, N, dtype=torch.bool, device=Q.device), 1),
            float("-inf"),
        )
    return torch.softmax(S, -1) @ V.double(), torch.logsumexp(S, -1)


# ================================================================ ① gradcheck（fp64，最强）

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("impl", ["naive", "flash"])
def test_gradcheck_single_tile(impl, causal):
    """小形状 (2, 8, 4)：默认 tile ≥ 8 时只走一个 tile——协议与数学的第一道关。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    torch.manual_seed(0)
    fn = NaiveAttentionFn if impl == "naive" else FlashAttnPytorch
    Q, K, V = (torch.randn(2, 8, 4, dtype=torch.float64, device="cuda", requires_grad=True)
               for _ in range(3))
    assert torch.autograd.gradcheck(
        lambda q, k, v: fn.apply(q, k, v, causal), (Q, K, V),
        eps=1e-6, atol=1e-6, rtol=1e-4,
    ), f"{impl} causal={causal} gradcheck 失败"


@needs_cuda
@needs_tiles
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tiles", [(2, 2), (4, 2), (2, 4), (3, 5)])   # N=9：全不整除；含 Bq != Bk
def test_gradcheck_multi_tile(causal, tiles):
    """小 tile 强制多 tile：online rescale、跨 tile 梯度累加、错位 tile 的全 mask 行。"""
    torch.manual_seed(0)
    Q, K, V = (torch.randn(2, 9, 4, dtype=torch.float64, device="cuda", requires_grad=True)
               for _ in range(3))
    assert torch.autograd.gradcheck(
        lambda q, k, v: flash(q, k, v, causal, tiles), (Q, K, V),
        eps=1e-6, atol=1e-6, rtol=1e-4,
    ), f"causal={causal} tiles={tiles} gradcheck 失败"


# ================================================================ ② float32 vs SDPA

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("impl", ["naive", "flash"])
@pytest.mark.parametrize("B,N,d", [(4, 100, 32), (2, 128, 64)])   # batch>1；N=100 非 2 的幂
def test_vs_sdpa_f32(impl, causal, B, N, d):
    """O 与 dQ/dK/dV 对比 SDPA + autograd，容差 1e-4。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    torch.manual_seed(42)
    fn = NaiveAttentionFn if impl == "naive" else FlashAttnPytorch

    Q, K, V = (torch.randn(B, N, d, device="cuda", requires_grad=True) for _ in range(3))
    Qr, Kr, Vr = (t.detach().clone().requires_grad_() for t in (Q, K, V))

    dO = torch.randn(B, N, d, device="cuda")
    O = fn.apply(Q, K, V, causal)
    O.backward(dO)

    O_ref = torch.nn.functional.scaled_dot_product_attention(Qr, Kr, Vr, is_causal=causal)
    O_ref.backward(dO)

    torch.testing.assert_close(O, O_ref, atol=1e-4, rtol=1e-4,
                               msg=lambda m: f"{impl} causal={causal} O: {m}")
    for g, ref, name in ((Q.grad, Qr.grad, "dQ"), (K.grad, Kr.grad, "dK"), (V.grad, Vr.grad, "dV")):
        assert g is not None, f"{name} 为 None"
        torch.testing.assert_close(g, ref, atol=1e-4, rtol=1e-4,
                                   msg=lambda m, n=name: f"{impl} causal={causal} {n}: {m}")


# ================================================================ ③ tile 不变性 + 边界 N

@needs_cuda
@needs_tiles
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", [1, 2, 17, 100, 129])
def test_tile_invariance_and_edges(causal, N):
    """换 (Bq,Bk)（含 Bq != Bk、不整除、单 tile 基准 (128,128)）不改变 O 与三个梯度。

    第一组 (16,16) 作为基准时先对 fp64 参考验证一次，之后各组只与基准对拍。
    """
    d = 16
    base = None
    for tiles in [(16, 16), (32, 16), (16, 64), (7, 5), (128, 128)]:
        torch.manual_seed(1)                                 # 每个 tile 组合用同一组输入
        Q, K, V = (torch.randn(2, N, d, device="cuda", requires_grad=True) for _ in range(3))
        dO = torch.randn(2, N, d, device="cuda")
        O = flash(Q, K, V, causal, tiles)
        O.backward(dO)
        cur = (O.detach(), Q.grad, K.grad, V.grad)
        if base is None:
            base = cur
            Or, _ = ref_attention(Q.detach(), K.detach(), V.detach(), causal)
            torch.testing.assert_close(O.double(), Or, atol=1e-4, rtol=1e-4)
        else:
            for a, b, name in zip(cur, base, ("O", "dQ", "dK", "dV")):
                torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4,
                                           msg=lambda m, n=name, t=tiles: f"tiles={t} 改变了 {n}: {m}")


# ================================================================ ④ 保存的张量契约

@needs_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_saved_tensors(causal):
    """恰好一个 (B,N) 的 L（值 = 自然对数 logsumexp、含 1/√d、causal 只对 j<=i 求和）；
    不保存 N×N 的 P/S——这是 FlashAttention 的核心内存契约。"""
    torch.manual_seed(2)
    B, N, d = 2, 100, 16
    Q, K, V = (torch.randn(B, N, d, device="cuda", requires_grad=True) for _ in range(3))
    O = flash(Q, K, V, causal)
    saved = O.grad_fn.saved_tensors

    assert all(t.numel() <= B * N * d for t in saved), \
        f"保存了超过 O(N·d) 的张量：{[tuple(t.shape) for t in saved]}（有没有偷偷存 P？）"
    Ls = [t for t in saved if t.shape == (B, N)]
    assert len(Ls) == 1, f"找不到形状为 (B, N) 的 L：{[tuple(t.shape) for t in saved]}"
    _, L_ref = ref_attention(Q.detach(), K.detach(), V.detach(), causal)
    torch.testing.assert_close(Ls[0].double(), L_ref, atol=1e-4, rtol=1e-4,
                               msg=lambda m: f"L 的约定不对（自然对数、含 1/√d、causal 只对 j<=i 求和）: {m}")


# ================================================================ ⑤ 低精度 dtype

@needs_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision(causal, dtype):
    """bf16/fp16：内部 fp32 累加；O 与梯度的 dtype 必须等于输入 dtype。"""
    torch.manual_seed(3)
    B, N, d = 2, 200, 32
    Q, K, V = (torch.randn(B, N, d, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
    dO = torch.randn(B, N, d, device="cuda", dtype=dtype)
    O = flash(Q, K, V, causal)
    O.backward(dO)
    assert O.dtype == dtype
    for t, name in ((Q.grad, "dQ"), (K.grad, "dK"), (V.grad, "dV")):
        assert t.dtype == dtype, f"{name} 的 dtype 应与输入一致，实际 {t.dtype}"

    # 参考：用同样被舍入过的输入，在 float64 上算
    Qr, Kr, Vr = (t.detach().double().requires_grad_() for t in (Q, K, V))
    Or, _ = ref_attention(Qr, Kr, Vr, causal)
    Or.backward(dO.double())
    tol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(O.double(), Or.detach(), atol=tol, rtol=tol)
    for g, r, name in ((Q.grad, Qr.grad, "dQ"), (K.grad, Kr.grad, "dK"), (V.grad, Vr.grad, "dV")):
        torch.testing.assert_close(g.double(), r, atol=tol * 3, rtol=tol * 3,
                                   msg=lambda m, n=name: f"{n}: {m}")


# ================================================================ ⑥ 显存是 O(N)

@needs_cuda
def test_memory_is_linear():
    """前向+反向的峰值增量：flash 应接近线性（比值≈2），朴素版应接近平方（比值≈4）。"""
    def peak_delta(fn, N, d=64):
        torch.manual_seed(0)
        Q, K, V = (torch.randn(1, N, d, device="cuda", requires_grad=True) for _ in range(3))
        dO = torch.randn(1, N, d, device="cuda")
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn(Q, K, V, True).backward(dO)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    f1, f2 = peak_delta(flash, 4096), peak_delta(flash, 8192)
    n1, n2 = peak_delta(NaiveAttentionFn.apply, 4096), peak_delta(NaiveAttentionFn.apply, 8192)
    print(f"flash 峰值增量 {f1 / 2**20:.0f} -> {f2 / 2**20:.0f} MiB（比值 {f2 / f1:.2f}）；"
          f"naive {n1 / 2**20:.0f} -> {n2 / 2**20:.0f} MiB（比值 {n2 / n1:.2f}）")
    assert f2 < 0.25 * 8192 * 8192 * 4, "flash 的峰值增量接近一个 N×N 张量，说明某处物化了 P"
    assert f2 / f1 < 3.0, f"flash 显存增长比 {f2 / f1:.2f}，应接近 2（线性），而不是 4（平方）"
    assert n2 / n1 > 3.0, f"naive 的增长比 {n2 / n1:.2f}，应接近 4（平方）——如果不是，测量方法有问题"
