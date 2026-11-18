"""单 kernel 版 backward 测试。
复用 test_flash_attn_bwd 里的正确性测试（把被测函数换成 fused 版），再加：
与两 kernel 版互相印证、atomics 压力、dQ_accum 清零、need 标志、运行间一致性。
用法：python test_flash_attn_bwd_fused.py [--filter 关键字]"""
import argparse
import sys

import torch

import test_flash_attn_bwd as tb
from test_flash_attn_bwd import (rand, fwd, rel_err, finite, ref_grads, DEV, D)
from flash_attn_bwd import flash_bwd_cute, flash_bwd_cute_fused


# ---------------------------------------------------------------- 新增测试
def t_matches_two_kernel():
    """与两 kernel 版互相印证：dK、dV 的数学与运算顺序相同，应几乎一致；dQ 只差 fp32 求和顺序。"""
    ok = True
    for (B, Nq, Nk, causal) in tb.CASES[::2] + tb.CASES[1::4]:
        Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
        O, L = fwd(Q, K, V, causal)
        a = flash_bwd_cute_fused(Q, K, V, O, L, dO, causal)
        b = flash_bwd_cute(Q, K, V, O, L, dO, causal)
        errs = [rel_err(x, y) for x, y in zip(a, b)]
        good = finite(*a) and max(errs) < 1e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tb.tag((B, Nq, Nk, causal))} 与两 kernel 版 dQ/dK/dV="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    return ok


def t_atomic_stress():
    """很多 block 同时往同一批 dQ 行做 atomicAdd（Nq 小、Nk 大；以及 causal 的长方阵）。"""
    ok = True
    for (B, Nq, Nk, causal) in ((1, 64, 4096, False), (2, 100, 2048, False),
                                (1, 1024, 1024, True), (1, 1024, 1024, False)):
        Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
        O, L = fwd(Q, K, V, causal)
        dq, dk, dv = flash_bwd_cute_fused(Q, K, V, O, L, dO, causal)
        rq, rk, rv = ref_grads(Q, K, V, dO, causal)
        errs = [rel_err(a, b) for a, b in ((dq, rq), (dk, rk), (dv, rv))]
        good = finite(dq, dk, dv) and max(errs) < 5e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tb.tag((B, Nq, Nk, causal))} dQ/dK/dV="
              f"{errs[0]:.1e}/{errs[1]:.1e}/{errs[2]:.1e}")
    return ok


def t_accum_fresh():
    """dQ_accum 每次调用都要从 0 开始：先用随机 dO 跑一次，再用 dO=0 跑，dQ 必须恰好为 0。"""
    B, Nq, Nk = 2, 130, 257
    Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
    O, L = fwd(Q, K, V, True)
    flash_bwd_cute_fused(Q, K, V, O, L, dO, True)
    dq0, dk0, dv0 = flash_bwd_cute_fused(Q, K, V, O, L, torch.zeros_like(dO), True)
    good = all((g.float() == 0).all().item() for g in (dq0, dk0, dv0))
    print(f"  [{'PASS' if good else 'FAIL'}] dO=0 的第二次调用，三个梯度恰好为 0（无上次残留）")
    return good


def t_need_flags():
    B, Nq, Nk = 2, 130, 257
    Q, K, V, dO = rand(B, Nq), rand(B, Nk), rand(B, Nk), rand(B, Nq)
    O, L = fwd(Q, K, V, True)
    full = flash_bwd_cute_fused(Q, K, V, O, L, dO, True)
    ok = True
    for need in ((True, False, False), (False, True, True), (False, True, False), (False, False, True)):
        got = flash_bwd_cute_fused(Q, K, V, O, L, dO, True, need=need)
        good = True
        for i in range(3):
            if need[i]:
                good &= got[i] is not None and rel_err(got[i], full[i]) < 1e-2
            else:
                good &= got[i] is None
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] need={need}")
    return ok


def t_run_to_run():
    """dK、dV 不涉及 atomics，应逐位一致；dQ 只差 fp32 求和顺序，应非常接近（统计一下差别）。"""
    ok = True
    for causal in (False, True):
        Q, K, V, dO = rand(2, 300), rand(2, 400), rand(2, 400), rand(2, 300)
        O, L = fwd(Q, K, V, causal)
        a = flash_bwd_cute_fused(Q, K, V, O, L, dO, causal)
        b = flash_bwd_cute_fused(Q, K, V, O, L, dO, causal)
        kv_same = torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        dq_diff = rel_err(a[0], b[0], floor=1e-3)
        good = kv_same and dq_diff < 1e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] causal={causal!s:<5} dK/dV 逐位一致={kv_same}，"
              f"dQ 两次差别={dq_diff:.1e}（atomics 求和顺序；允许非零）")
    return ok


# ---------------------------------------------------------------- 入口
REUSED = [  # 复用上一版测试，被测函数已替换成 fused
    ("对 fp32 autograd",        tb.t_vs_autograd),
    ("对 torch 分块实现",       tb.t_vs_torch_tiled),
    ("causal 无人可见的 K 行",  tb.t_causal_unseen_keys_zero),
    ("毒化输入",                tb.t_poisoned_inputs),
    ("极端数值",                tb.t_extremes),
    ("one-hot 零梯度",          tb.t_one_hot),
    ("经 autograd 端到端",      tb.t_through_autograd),
]
NEW = [
    ("与两 kernel 版互相印证",  t_matches_two_kernel),
    ("atomics 压力",            t_atomic_stress),
    ("dQ_accum 清零",           t_accum_fresh),
    ("need 标志",               t_need_flags),
    ("运行间一致性",            t_run_to_run),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        print("需要 sm80+ 的 CUDA GPU")
        sys.exit(1)
    torch.empty(1, device=DEV)

    tb.flash_bwd_cute = flash_bwd_cute_fused          # 让复用的测试都去测 fused 版
    passed = failed = 0
    try:
        for name, fn in REUSED + NEW:
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
    finally:
        tb.flash_bwd_cute = flash_bwd_cute
    print(f"\n{passed} 组通过，{failed} 组失败")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()