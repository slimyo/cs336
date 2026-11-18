"""FlashAttnFunction 的测试：反向算法（float64 精确验证）、分块无关性、前向、端到端梯度、
部分 requires_grad、非连续/未对齐输入、no_grad、输入校验、确定性、训练图集成。

用法：
    python test_flash_attn_autograd.py
    python test_flash_attn_autograd.py --filter 端到端
没有 CUDA/cutlass 时，不依赖 kernel 的测试（反向算法、分块无关性、输入校验）仍会运行，其余标记 SKIP。
退出码：全部通过为 0，否则为 1。
"""
import argparse
import math
import sys

import torch

from flash_attn_autograd import (FlashAttnFunction, flash_attn_func, flash_fwd_cute,
                                 flash_bwd_tiled_torch, LOG2E)

HAS_CUDA = torch.cuda.is_available()
DEV = "cuda" if HAS_CUDA else "cpu"
D = 64
BF16 = torch.bfloat16

# (B, Nq, Nk, causal)：覆盖单 tile、非整除、Nq>Nk、Nq<Nk、Nq=1、长序列、多 tile 方阵
CASES = [
    (1,  64,   64, False), (2, 130,  257, False), (2, 130,  257, True),
    (1, 200,   80, True),  (1,  70,  300, True),  (1,   1,  128, False),
    (1,   1,  128, True),  (3,  64, 1000, False), (2, 256,  256, True),
    (1,  17,    5, False), (1,  17,    5, True),
]


# =============================================================================
# 参考实现（fp32/fp64 autograd）与指标
# =============================================================================
def ref_forward(q, k, v, causal):
    S = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:                                   # 左上对齐：保留 j <= i
        keep = torch.ones(S.shape[-2:], dtype=torch.bool, device=S.device).tril()
        S = S.masked_fill(~keep, float("-inf"))
    lse2 = torch.logsumexp(S, dim=-1) * LOG2E    # base-2 LSE，与 kernel 的 L 同一量
    O = torch.softmax(S, dim=-1) @ v
    return O, lse2


def ref_grads(Q, K, V, dO, causal, dtype=torch.float32):
    q, k, v = [x.detach().to(dtype).requires_grad_(True) for x in (Q, K, V)]
    o, _ = ref_forward(q, k, v, causal)
    o.backward(dO.to(dtype))
    return o.detach(), q.grad, k.grad, v.grad


def rel_err(a, b, floor=1e-3):
    """max|a-b| / (max|b| + floor)。floor 防止参考值≈0（如 one-hot softmax 的精确梯度为 0）时被噪声放大。
    等价于 allclose 的 atol/rtol 复合判据；在 float64 下计算。"""
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / (b.abs().max() + floor)).item()


def rand(B, N, dtype, scale=1.0):
    return (torch.randn(B, N, D, device=DEV) * scale).to(dtype)


def tag(c):
    B, Nq, Nk, causal = c
    return f"B={B} Nq={Nq:<4} Nk={Nk:<5} causal={causal!s:<5}"


# =============================================================================
# 不依赖 kernel 的测试
# =============================================================================
def t_bwd_algebra_fp64():
    """float64 下，分块反向 vs autograd，应达到 ~1e-12：验证 dS/dQ/dK/dV 的公式和 causal 跳块。"""
    ok = True
    for c in CASES:
        B, Nq, Nk, causal = c
        Q, K, V, dO = (torch.randn(B, n, D, device=DEV, dtype=torch.float64) for n in (Nq, Nk, Nk, Nq))
        with torch.no_grad():
            O, lse2 = ref_forward(Q, K, V, causal)
        dq, dk, dv = flash_bwd_tiled_torch(Q, K, V, O, lse2, dO, causal, block_q=32, block_k=48)
        _, rq, rk, rv = ref_grads(Q, K, V, dO, causal, torch.float64)
        errs = [rel_err(a, b) for a, b in ((dq, rq), (dk, rk), (dv, rv))]
        good = max(errs) < 1e-9
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tag(c)} dQ/dK/dV 相对误差="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    return ok


def t_bwd_block_invariance():
    """不同块大小（含不整除的 7×5）结果应一致。"""
    ok = True
    for causal in (False, True):
        B, Nq, Nk = 2, 130, 257
        Q, K, V, dO = (torch.randn(B, n, D, device=DEV, dtype=torch.float64) for n in (Nq, Nk, Nk, Nq))
        with torch.no_grad():
            O, lse2 = ref_forward(Q, K, V, causal)
        base = flash_bwd_tiled_torch(Q, K, V, O, lse2, dO, causal, block_q=128, block_k=128)
        for bq, bk in ((16, 16), (64, 32), (7, 5)):
            got = flash_bwd_tiled_torch(Q, K, V, O, lse2, dO, causal, block_q=bq, block_k=bk)
            e = max(rel_err(g, b) for g, b in zip(got, base))
            good = e < 1e-10
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} 块({bq},{bk}) vs (128,128) 误差={e:.1e}")
    return ok


def t_bwd_need_flags():
    """need 只让需要的梯度被计算，其余返回 None，且已算出的与全量一致。"""
    B, Nq, Nk = 2, 100, 150
    Q, K, V, dO = (torch.randn(B, n, D, device=DEV, dtype=torch.float64) for n in (Nq, Nk, Nk, Nq))
    with torch.no_grad():
        O, lse2 = ref_forward(Q, K, V, True)
    full = flash_bwd_tiled_torch(Q, K, V, O, lse2, dO, True)
    ok = True
    for need in ((True, False, False), (False, True, True), (False, False, True)):
        got = flash_bwd_tiled_torch(Q, K, V, O, lse2, dO, True, need=need)
        for i in range(3):
            if need[i]:
                ok &= got[i] is not None and rel_err(got[i], full[i]) < 1e-12
            else:
                ok &= got[i] is None
        print(f"  [{'PASS' if ok else 'FAIL'}] need={need}")
    return ok


def t_input_validation():
    """非法输入应抛 ValueError（校验在碰 CUDA 之前，CPU 上也能跑）。"""
    ok = True

    def expect(name, fn):
        try:
            fn()
        except ValueError:
            print(f"  [PASS] {name}")
            return True
        except Exception as e:
            print(f"  [FAIL] {name}：抛了 {type(e).__name__} 而不是 ValueError：{e}")
            return False
        print(f"  [FAIL] {name}：没有抛异常")
        return False

    mk = lambda n, d=D, dt=BF16, dev="cpu": torch.zeros(1, n, d, dtype=dt, device=dev)
    ok &= expect("fp32 输入", lambda: flash_attn_func(mk(8, dt=torch.float32), mk(8), mk(8)))
    ok &= expect("CPU 输入", lambda: flash_attn_func(mk(8), mk(8), mk(8)))
    ok &= expect("2 维输入", lambda: flash_attn_func(torch.zeros(8, D, dtype=BF16), mk(8), mk(8)))
    if HAS_CUDA:
        g = lambda n, d=D: mk(n, d, dev="cuda")
        ok &= expect("head_dim=32", lambda: flash_attn_func(g(8, 32), g(8, 32), g(8, 32)))
        ok &= expect("K/V 长度不同", lambda: flash_attn_func(g(8), g(8), g(9)))
        ok &= expect("batch 不一致", lambda: flash_attn_func(g(8), torch.zeros(2, 8, D, dtype=BF16, device="cuda"), g(8)))
    return ok


# =============================================================================
# 依赖 kernel 的测试（需要 CUDA + cutlass）
# =============================================================================
def t_forward():
    ok = True
    for c in CASES:
        B, Nq, Nk, causal = c
        Q, K, V = rand(B, Nq, BF16), rand(B, Nk, BF16), rand(B, Nk, BF16)
        with torch.no_grad():
            out = flash_attn_func(Q, K, V, causal)
            _, L = flash_fwd_cute(Q, K, V, causal)
        o_ref, lse_ref = ref_forward(Q.float(), K.float(), V.float(), causal)
        finite = bool(torch.isfinite(out.float()).all() and torch.isfinite(L).all())
        e_o = rel_err(out, o_ref)
        e_l = (L - lse_ref).abs().max().item()
        good = finite and out.shape == Q.shape and out.dtype == BF16 and e_o < 2e-2 and e_l < 5e-3
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tag(c)} O相对误差={e_o:.2e} LSE误差={e_l:.2e} 有限={finite}")
    return ok


def t_end_to_end():
    ok = True
    for c in CASES:
        B, Nq, Nk, causal = c
        Q, K, V, dO = rand(B, Nq, BF16), rand(B, Nk, BF16), rand(B, Nk, BF16), rand(B, Nq, BF16)
        q, k, v = (x.clone().requires_grad_(True) for x in (Q, K, V))
        out = flash_attn_func(q, k, v, causal)
        out.backward(dO)

        o_ref, rq, rk, rv = ref_grads(Q, K, V, dO, causal)
        errs = [rel_err(out, o_ref)] + [rel_err(g, r) for g, r in ((q.grad, rq), (k.grad, rk), (v.grad, rv))]
        meta = (all(g is not None and g.dtype == BF16 and g.shape == x.shape and torch.isfinite(g.float()).all()
                    for g, x in ((q.grad, Q), (k.grad, K), (v.grad, V))))
        good = meta and errs[0] < 2e-2 and max(errs[1:]) < 5e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tag(c)} O/dQ/dK/dV 相对误差="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}/{errs[3]:.1e} 形状dtype有限={meta}")
    return ok


def t_partial_requires_grad():
    ok = True
    B, Nq, Nk = 2, 130, 257
    for causal in (False, True):
        Q, K, V, dO = rand(B, Nq, BF16), rand(B, Nk, BF16), rand(B, Nk, BF16), rand(B, Nq, BF16)
        _, rq, rk, rv = ref_grads(Q, K, V, dO, causal)
        for name, flags in (("只 Q", (1, 0, 0)), ("只 K,V", (0, 1, 1)), ("只 V", (0, 0, 1))):
            ts = [x.clone().requires_grad_(bool(f)) for x, f in zip((Q, K, V), flags)]
            flash_attn_func(*ts, causal).backward(dO)
            refs = (rq, rk, rv)
            good = True
            for t, f, r in zip(ts, flags, refs):
                good &= (t.grad is not None and rel_err(t.grad, r) < 5e-2) if f else (t.grad is None)
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} {name}")
    return ok


def t_noncontiguous_and_misaligned():
    """非连续（batch 方向切片）、起始地址未 16B 对齐（偏移 1 个 bf16）的输入。"""
    ok = True
    B, Nq, Nk = 2, 100, 130
    for causal in (False, True):
        q = torch.randn(B, Nq + 1, D, device=DEV).to(BF16)[:, 1:].detach().requires_grad_(True)  # 非连续
        flat = torch.randn(B * Nk * D + 1, device=DEV).to(BF16)
        k = flat[1:].view(B, Nk, D).detach().requires_grad_(True)                                # 连续但未对齐
        v = rand(B, Nk, BF16).requires_grad_(True)
        dO = rand(B, Nq, BF16)
        assert not q.is_contiguous() and k.data_ptr() % 16 != 0

        out = flash_attn_func(q, k, v, causal)
        out.backward(dO)
        o_ref, rq, rk, rv = ref_grads(q, k, v, dO, causal)
        errs = [rel_err(out, o_ref), rel_err(q.grad, rq), rel_err(k.grad, rk), rel_err(v.grad, rv)]
        good = errs[0] < 2e-2 and max(errs[1:]) < 5e-2 and q.grad.shape == q.shape
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} O/dQ/dK/dV 相对误差="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}/{errs[3]:.1e}")
    return ok


def t_no_grad():
    Q, K, V = rand(1, 100, BF16), rand(1, 100, BF16), rand(1, 100, BF16)
    q, k, v = (x.clone().requires_grad_(True) for x in (Q, K, V))
    with torch.no_grad():
        out = flash_attn_func(q, k, v, True)
    o_ref, _ = ref_forward(Q.float(), K.float(), V.float(), True)
    good = (not out.requires_grad) and out.grad_fn is None and rel_err(out, o_ref) < 2e-2
    print(f"  [{'PASS' if good else 'FAIL'}] no_grad 下不建图且结果正确")
    # 有梯度时应有 grad_fn
    out2 = flash_attn_func(q, k, v, True)
    good2 = out2.requires_grad and out2.grad_fn is not None
    print(f"  [{'PASS' if good2 else 'FAIL'}] 有梯度时输出带 grad_fn")
    return good and good2


def t_determinism():
    """无 atomics：同一输入两次前向、两次反向应逐位相同。"""
    B, Nq, Nk = 2, 130, 257
    q, k, v = (rand(B, n, BF16).requires_grad_(True) for n in (Nq, Nk, Nk))
    dO = rand(B, Nq, BF16)
    out1 = flash_attn_func(q, k, v, True)
    out2 = flash_attn_func(q, k, v, True)
    g1 = torch.autograd.grad(out1, (q, k, v), dO, retain_graph=True)
    g2 = torch.autograd.grad(out1, (q, k, v), dO, retain_graph=True)
    good_f = torch.equal(out1, out2)
    good_b = all(torch.equal(a, b) for a, b in zip(g1, g2))
    print(f"  [{'PASS' if good_f else 'FAIL'}] 前向两次逐位一致")
    print(f"  [{'PASS' if good_b else 'FAIL'}] 反向两次逐位一致（retain_graph）")
    return good_f and good_b


def t_training_graph():
    """集成：fp32 参数 → cast bf16 → 注意力 → float → 加权求和损失；梯度穿过 cast 回到 fp32 参数。"""
    ok = True
    B, Nq, Nk = 2, 100, 100
    for causal in (False, True):
        P = [torch.randn(B, n, D, device=DEV).requires_grad_(True) for n in (Nq, Nk, Nk)]
        w = torch.randn(B, Nq, D, device=DEV)
        out = flash_attn_func(*(p.bfloat16() for p in P), causal)
        loss = (out.float() * w).sum()
        loss.backward()

        Qb, Kb, Vb = (p.detach().bfloat16() for p in P)
        _, rq, rk, rv = ref_grads(Qb, Kb, Vb, w.bfloat16(), causal)   # out.float() 的反向把 w 转成 bf16 送回
        errs = [rel_err(p.grad, r) for p, r in zip(P, (rq, rk, rv))]
        good = all(p.grad is not None and p.grad.dtype == torch.float32 for p in P) and max(errs) < 5e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} 参数梯度相对误差="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    return ok


# =============================================================================
# 入口
# =============================================================================
TESTS = [
    # (名称, 函数, 是否需要 kernel)
    ("反向算法-float64 精确",    t_bwd_algebra_fp64,             False),
    ("反向分块无关性",           t_bwd_block_invariance,         False),
    ("反向 need 标志",           t_bwd_need_flags,               False),
    ("输入校验",                 t_input_validation,             False),
    ("前向 O 与 LSE",            t_forward,                      True),
    ("端到端梯度",               t_end_to_end,                   True),
    ("部分 requires_grad",       t_partial_requires_grad,        True),
    ("非连续/未对齐输入",        t_noncontiguous_and_misaligned, True),
    ("no_grad 与 grad_fn",       t_no_grad,                      True),
    ("确定性",                   t_determinism,                  True),
    ("训练图集成",               t_training_graph,               True),
]


def main():
    if HAS_CUDA:
        torch.empty(1, device="cuda")   # 提前建立 CUDA context
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", type=str, default=None, help="只跑名称包含该子串的测试")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    have_kernel = HAS_CUDA and torch.cuda.get_device_capability()[0] >= 8
    if have_kernel:
        try:
            import cutlass.cute  # noqa: F401
        except Exception:
            have_kernel = False
    if not have_kernel:
        print("提示：无 sm80+ CUDA 或无 cutlass，依赖 kernel 的测试将 SKIP\n")

    passed = failed = skipped = 0
    for name, fn, needs_kernel in TESTS:
        if args.filter and args.filter not in name:
            continue
        print(f"== {name}")
        if needs_kernel and not have_kernel:
            print("  [SKIP]")
            skipped += 1
            continue
        torch.manual_seed(args.seed)
        try:
            ok = fn()
        except Exception as e:
            print(f"  [FAIL] 运行出错: {type(e).__name__}: {e}")
            ok = False
        passed += ok
        failed += (not ok)

    print(f"\n{passed} 组通过，{failed} 组失败，{skipped} 组跳过")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()