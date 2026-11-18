"""CuTe backward 测试：对 fp32 autograd、对 torch 分块实现、need 标志、causal 空 tile、毒化输入、
确定性、极端数值、经 autograd.Function 的端到端。
用法：python test_flash_attn_bwd.py [--filter 关键字]
退出码：全部通过为 0。"""
import argparse
import math
import sys

import torch

from flash_attn_autograd import (flash_attn_func, flash_fwd_cute, flash_bwd_tiled_torch,
                                 set_backward_impl, LOG2E)
from flash_attn_bwd import flash_bwd_cute

DEV = "cuda"
D = 64
BF16 = torch.bfloat16

CASES = [  # (B, Nq, Nk, causal)
    (1,  64,   64, False), (1,  64,   64, True),
    (2, 130,  257, False), (2, 130,  257, True),
    (1, 200,   80, False), (1, 200,   80, True),     # Nq>Nk
    (1,  70,  300, False), (1,  70,  300, True),     # Nq<Nk：causal 下后面的 K tile 无人可见
    (1,   1,  128, False), (1,   1,  128, True),
    (3,  64, 1000, False), (2, 256,  256, True),
    (1,  17,    5, False), (1,  17,    5, True),
    (1, 512,  512, False), (1, 512,  512, True),     # 多 tile、stage 回绕
    (2, 333,  129, False), (2, 333,  129, True),
]


# ---------------------------------------------------------------- 参考与指标
def ref_forward(q, k, v, causal):
    S = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        keep = torch.ones(S.shape[-2:], dtype=torch.bool, device=S.device).tril()
        S = S.masked_fill(~keep, float("-inf"))
    return torch.softmax(S, dim=-1) @ v


def ref_grads(Q, K, V, dO, causal):
    q, k, v = [x.detach().float().requires_grad_(True) for x in (Q, K, V)]
    ref_forward(q, k, v, causal).backward(dO.float())
    return q.grad, k.grad, v.grad


def rel_err(a, b, floor=1e-3):
    """max|a-b| / (max|b| + floor)；floor 防止参考值≈0（one-hot softmax 的精确梯度为 0）时被噪声放大。"""
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / (b.abs().max() + floor)).item()


def rand(B, N, scale=1.0):
    return (torch.randn(B, N, D, device=DEV) * scale).to(BF16)


def make_poisoned(B, N, scale=1.0):
    """放进更大的缓冲区，尾部填 NaN；若 kernel 越界读，NaN 会传进结果。"""
    x = rand(B, N, scale)
    n = B * N * D
    buf = torch.full((n + 4096,), float("nan"), dtype=BF16, device=DEV)
    v = buf[:n].view(B, N, D)
    v.copy_(x)
    return v


def tag(c):
    B, Nq, Nk, causal = c
    return f"B={B} Nq={Nq:<4} Nk={Nk:<5} causal={causal!s:<5}"


def fwd(Q, K, V, causal):
    with torch.no_grad():
        return flash_fwd_cute(Q, K, V, causal)           # (O, L2)


def finite(*ts):
    return all(t is None or bool(torch.isfinite(t.float()).all()) for t in ts)


# ---------------------------------------------------------------- 测试
def t_vs_autograd():
    ok = True
    for c in CASES:
        B, Nq, Nk, causal = c
        Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
        O, L = fwd(Q, K, V, causal)
        dq, dk, dv = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        rq, rk, rv = ref_grads(Q, K, V, dO, causal)
        meta = all(g.shape == x.shape and g.dtype == BF16 for g, x in ((dq, Q), (dk, K), (dv, V)))
        errs = [rel_err(a, b) for a, b in ((dq, rq), (dk, rk), (dv, rv))]
        good = meta and finite(dq, dk, dv) and max(errs) < 5e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tag(c)} dQ/dK/dV 相对误差="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e} 形状dtype={meta}")
    return ok


def t_vs_torch_tiled():
    """与 torch 分块实现（同样的 O、L2 输入）对比：隔离出 kernel 与公式的差异。"""
    ok = True
    for c in CASES[::2] + CASES[1::4]:
        B, Nq, Nk, causal = c
        Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
        O, L = fwd(Q, K, V, causal)
        got = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        ref = flash_bwd_tiled_torch(Q, K, V, O, L, dO, causal)
        errs = [rel_err(g, r) for g, r in zip(got, ref)]
        good = max(errs) < 3e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tag(c)} 与 torch 分块 dQ/dK/dV="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    return ok


def t_need_flags():
    B, Nq, Nk = 2, 130, 257
    Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
    O, L = fwd(Q, K, V, True)
    full = flash_bwd_cute(Q, K, V, O, L, dO, True)
    ok = True
    for need in ((True, False, False), (False, True, True), (False, True, False), (False, False, True)):
        got = flash_bwd_cute(Q, K, V, O, L, dO, True, need=need)
        for i in range(3):
            if need[i]:
                ok &= got[i] is not None and torch.equal(got[i], full[i])    # 与全量逐位一致
            else:
                ok &= got[i] is None
        print(f"  [{'PASS' if ok else 'FAIL'}] need={need}")
    return ok


def t_causal_unseen_keys_zero():
    """causal 且 Nk>Nq：第 i 行只看 j<=i，所以 j>=Nq 的 key 无人可见，dK、dV 的这些行必须恰好为 0（而不是 NaN 或噪声）。"""
    ok = True
    for Nq, Nk in ((70, 300), (64, 256), (1, 128)):
        Q, K, V, dO = rand(1, Nq), rand(1, Nk), rand(1, Nk), rand(1, Nq)
        O, L = fwd(Q, K, V, True)
        dq, dk, dv = flash_bwd_cute(Q, K, V, O, L, dO, True)
        tail_k, tail_v = dk[:, Nq:], dv[:, Nq:]
        good = finite(tail_k, tail_v) and (tail_k.float() == 0).all().item() and (tail_v.float() == 0).all().item()
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] Nq={Nq} Nk={Nk}：dK/dV 的第 {Nq} 行及之后全为 0")
    return ok


def t_poisoned_inputs():
    """Q/K/V/dO 尾部是 NaN：pred/补零/L2-D 的越界读取有问题时，结果会出现 NaN。"""
    ok = True
    for (Nq, Nk, causal) in ((100, 130, False), (100, 130, True), (33, 77, True)):
        Q, K, V, dO = (make_poisoned(1, n) for n in (Nq, Nk, Nk, Nq))
        O, L = fwd(Q, K, V, causal)
        dq, dk, dv = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        rq, rk, rv = ref_grads(Q, K, V, dO, causal)
        errs = [rel_err(a, b) for a, b in ((dq, rq), (dk, rk), (dv, rv))]
        good = finite(dq, dk, dv) and max(errs) < 5e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] Nq={Nq} Nk={Nk} causal={causal!s:<5} "
              f"有限={finite(dq, dk, dv)} 误差={errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    return ok


def t_determinism():
    """无原子操作：同一输入两次运行逐位一致。"""
    ok = True
    for causal in (False, True):
        Q, K, V, dO = rand(2, 300), rand(2, 400), rand(2, 400), rand(2, 300)
        O, L = fwd(Q, K, V, causal)
        a = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        b = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        good = all(torch.equal(x, y) for x, y in zip(a, b))
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} 两次运行逐位一致")
    return ok


def t_extremes():
    """大幅值（softmax 很尖）与 dO=0。"""
    ok = True
    for causal in (False, True):
        Q, K, V, dO = rand(1, 100, 8.0), rand(1, 300), rand(1, 300), rand(1, 100)
        O, L = fwd(Q, K, V, causal)
        got = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        ref = ref_grads(Q, K, V, dO, causal)
        errs = [rel_err(a, b) for a, b in zip(got, ref)]
        good = finite(*got) and max(errs) < 1e-1          # 尖锐 softmax 下 bf16 的舍入更敏感，放宽到 1e-1
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} 大幅值 dQ/dK/dV={errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")

        zero = torch.zeros_like(dO)
        z = flash_bwd_cute(Q, K, V, O, L, zero, causal)
        good = all((g.float() == 0).all().item() for g in z)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} dO=0 时三个梯度恰好为 0")
    return ok


def t_one_hot():
    """Nq=1 且 causal：softmax 为 one-hot，dQ、dK 的精确值为 0，dV 只有第 0 行且等于 dO。"""
    Q, K, V, dO = rand(1, 1), rand(1, 128), rand(1, 128), rand(1, 1)
    O, L = fwd(Q, K, V, True)
    dq, dk, dv = flash_bwd_cute(Q, K, V, O, L, dO, True)
    na = max(dq.float().abs().max().item(), dk.float().abs().max().item())
    dv_ok = torch.allclose(dv[:, 0].float(), dO[:, 0].float(), atol=1e-2) and dv[:, 1:].float().abs().max().item() == 0.0
    good = na < 1e-3 and dv_ok
    print(f"  [{'PASS' if good else 'FAIL'}] max|dQ|,|dK|={na:.1e}（应≈0），dV 仅第 0 行=dO：{dv_ok}")
    return good


def t_through_autograd():
    """切换到 CuTe 反向，走完整 autograd.Function：fp32 参数 → bf16 → 注意力 → 损失。"""
    ok = True
    set_backward_impl(flash_bwd_cute)
    try:
        for causal in (False, True):
            B, Nq, Nk = 2, 130, 257
            P = [torch.randn(B, n, D, device=DEV).requires_grad_(True) for n in (Nq, Nk, Nk)]
            w = torch.randn(B, Nq, D, device=DEV)
            out = flash_attn_func(*(p.bfloat16() for p in P), causal)
            (out.float() * w).sum().backward()
            Qb, Kb, Vb = (p.detach().bfloat16() for p in P)
            refs = ref_grads(Qb, Kb, Vb, w.bfloat16(), causal)
            errs = [rel_err(p.grad, r) for p, r in zip(P, refs)]
            good = all(p.grad is not None and p.grad.dtype == torch.float32 for p in P) and max(errs) < 5e-2
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} 参数梯度={errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    finally:
        set_backward_impl(flash_bwd_tiled_torch)         # 恢复默认，避免影响其他测试
    return ok


TESTS = [
    ("对 fp32 autograd",        t_vs_autograd),
    ("对 torch 分块实现",       t_vs_torch_tiled),
    ("need 标志",               t_need_flags),
    ("causal 无人可见的 K 行",  t_causal_unseen_keys_zero),
    ("毒化输入",                t_poisoned_inputs),
    ("确定性",                  t_determinism),
    ("极端数值",                t_extremes),
    ("one-hot 零梯度",          t_one_hot),
    ("经 autograd 端到端",      t_through_autograd),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        print("需要 sm80+ 的 CUDA GPU")
        sys.exit(1)
    torch.empty(1, device=DEV)                           # 提前建立 CUDA context

    passed = failed = 0
    for name, fn in TESTS:
        if args.filter and args.filter not in name:
            continue
        print(f"== {name}")
        torch.manual_seed(args.seed)
        try:
            ok = fn()
        except Exception as e:
            print(f"  [FAIL] 运行出错: {type(e).__name__}: {e}")
            ok = False
        passed += ok
        failed += (not ok)
    print(f"\n{passed} 组通过，{failed} 组失败")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()