"""ablate_fwd.py：前向优化（O1 倒序发射 / O2 ILP 规约 / O3 惰性缩放 / O4 Q 别名）的消融实验。

对每个配置：编译 → 与 fp32 参考对照（两个形状）→ 与 baseline 比较是否逐位一致 → 计时。
用法：
    python ablate_fwd.py                                  # causal，seq = 4096 16384 32768
    python ablate_fwd.py --seqs 2048 8192 --iters 50
    python ablate_fwd.py --no-causal                      # 非 causal 时 O1 不起作用（预期与 baseline 持平）
逐位一致的预期：+O1、+O3、+O4 应当与 baseline 逐位一致；+O2（以及包含它的配置）因求和顺序改变而不一致，
只要对参考的误差在容差内即可。
"""
import argparse
import math
import statistics
import time

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn_naive import FLASH_ATTN_naive

DEV, D, BF16 = "cuda", 64, torch.bfloat16
OFF = dict(reverse_blocks=False, ilp_reduce=False, lazy_rescale=False, alias_q=False)

CONFIGS = [
    ("baseline",       dict(OFF)),
    ("+O1 reverse",    dict(OFF, reverse_blocks=True)),
    ("+O2 ilp",        dict(OFF, ilp_reduce=True)),
    ("+O3 lazy",       dict(OFF, lazy_rescale=True)),
    ("O1+O2+O3",       dict(OFF, reverse_blocks=True, ilp_reduce=True, lazy_rescale=True)),
    ("O1+O2+O3+O4",    dict(OFF, reverse_blocks=True, ilp_reduce=True, lazy_rescale=True,
                            alias_q=True, smem_slack=0)),
    # 单独评估 O4（不混入 O2/O3）：
    #   slack=0    ：每 block 32768B 动态 smem，若动态 smem 基址 1024B 对齐，每 SM 可放 3 个 block
    #   slack=1024 ：同样的代码、smem 只多 1KB，每 SM 仍是 2 个 block —— 用来分离“占用率”与“别名本身”的影响
    ("O1+O4 slack=0",    dict(OFF, reverse_blocks=True, alias_q=True, smem_slack=0)),
    ("O1+O4 slack=1024", dict(OFF, reverse_blocks=True, alias_q=True, smem_slack=1024)),
    ("O1 only (新默认)", dict(OFF, reverse_blocks=True)),
]


def event_ms(fn, warmup=3, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def make_args(N, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    Q, K, V = (torch.randn(1, N, D, device=DEV, generator=g).to(BF16) for _ in range(3))
    O = torch.empty_like(Q)
    L = torch.empty(1, N, dtype=torch.float32, device=DEV)
    c = lambda t, ld: from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=ld)
    return (Q, K, V, O, L), (c(Q, 2), c(K, 2), c(V, 2), c(O, 2), c(L, 1))


def reference(Q, K, V, causal):
    S = Q.float() @ K.float().transpose(-1, -2) / math.sqrt(D)
    if causal:
        keep = torch.ones(S.shape[-2:], dtype=torch.bool, device=S.device).tril()
        S = S.masked_fill(~keep, float("-inf"))
    return torch.softmax(S, -1) @ V.float(), torch.logsumexp(S, -1) * math.log2(math.e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-causal", action="store_true")
    args = ap.parse_args()
    causal = not args.no_causal

    # ---------- 编译 + 正确性 ----------
    fns, status, bitwise, compile_s = {}, {}, {}, {}
    base_out = {}
    for name, kw in CONFIGS:
        status[name], bitwise[name] = "OK", "-"
        try:
            kern = FLASH_ATTN_naive(causal=causal, **kw)
            tens, a = make_args(1000)
            t0 = time.perf_counter()
            fn = cute.compile(kern, *a)
            compile_s[name] = time.perf_counter() - t0
            outs = []
            for N in (1000, 257):                       # 一个多 tile 的、一个非整除的小形状
                tens, a = make_args(N, seed=N)
                fn(*a)
                torch.cuda.synchronize()
                o_ref, l_ref = reference(tens[0], tens[1], tens[2], causal)
                e_o = (tens[3].float() - o_ref).abs().max().item()
                e_l = (tens[4] - l_ref).abs().max().item()
                if not (e_o < 2e-2 and e_l < 5e-3):
                    status[name] = f"WRONG(O={e_o:.1e},L={e_l:.1e})"
                outs.append((tens[3].clone(), tens[4].clone()))
            if name == "baseline":
                base_out = outs
            elif base_out:
                same = all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(outs, base_out))
                bitwise[name] = "是" if same else "否"
            fns[name] = fn
        except Exception as e:                           # 编译失败也是数据
            first = str(e).strip().splitlines()[0] if str(e).strip() else ""
            status[name] = f"FAIL({type(e).__name__}: {first[:80]})"

    # ---------- 计时 ----------
    res = {name: {} for name, _ in CONFIGS}
    for N in args.seqs:
        tens, a = make_args(N)
        flops = (2.0 if causal else 4.0) * N * N * D
        for name, _ in CONFIGS:
            if name in fns and status[name].startswith("OK"):
                ms = event_ms(lambda: fns[name](*a), iters=args.iters)
                res[name][N] = (ms, flops / ms / 1e9)
        del tens, a
        torch.cuda.empty_cache()

    # ---------- 输出 ----------
    print(f"\n前向消融：d=64，bf16，causal={causal}；TFLOP/s = {'2' if causal else '4'}·N²·d / 时间；括号为相对 baseline 的倍数\n")
    head = "| 配置 | 状态 | 编译(s) | 与 baseline 逐位一致 |"
    for N in args.seqs:
        head += f" N={N} ms | TFLOP/s |"
    print(head)
    print("|" + "---|" * (4 + 2 * len(args.seqs)))
    for name, _ in CONFIGS:
        cs = f"{compile_s[name]:.1f}" if name in compile_s else "-"
        line = f"| {name} | {status[name]} | {cs} | {bitwise[name]} |"
        for N in args.seqs:
            if N in res[name]:
                ms, tf = res[name][N]
                base = res["baseline"].get(N)
                ratio = f"（{tf / base[1]:.3f}x）" if base else ""
                line += f" {ms:.3f} | {tf:.1f}{ratio} |"
            else:
                line += " - | - |"
        print(line)


if __name__ == "__main__":
    main()