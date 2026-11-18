"""bench_flash.py：对比 朴素 PyTorch attention / SDPA / 本 CuTe FA（前向、前向+反向、显存）。
用法：python bench_flash.py --seqs 128 512 2048 8192 32768 --ds 64 --dtypes bf16 [--fused] [--flush-l2]
CuTe FA 只支持 d=64、bf16，其余组合标 N/A。"""
import argparse
import math
import statistics
import time

import torch
import torch.nn.functional as F

DEV = "cuda"
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


def timeit(fn, warmup=3, iters=20, flush=False):
    """event 计时，取中位数。flush=True 每次先清 L2（与 triton do_bench 的默认口径一致）。同一张表里别混用口径。"""
    buf = torch.empty(256 * 2**20 // 4, device=DEV) if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if flush:
            buf.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def peak_mem_gib(fn):
    """返回 (相对调用前的增量峰值 GiB, 绝对峰值 GiB)。"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / 2**30, peak / 2**30


def guard(fn):
    """OOM 记为 None，释放后继续。"""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def cublas_peak_tflops(dtype, n=4096):
    a = torch.randn(n, n, device=DEV, dtype=dtype)
    b = torch.randn(n, n, device=DEV, dtype=dtype)
    ms = timeit(lambda: a @ b, warmup=5, iters=30)
    return 2.0 * n**3 / ms / 1e9


def naive_attention(q, k, v, causal):
    S = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        keep = torch.ones(S.shape[-2:], dtype=torch.bool, device=S.device).tril()
        S = S.masked_fill(~keep, float("-inf"))
    return torch.softmax(S, dim=-1) @ v


def pick_sdpa_backend(q4, k4, v4, causal):
    """依次试 flash / efficient / math，返回第一个能跑的 (名字, 调用)。"""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("math", SDPBackend.MATH)):
        try:
            def call(be=be):
                with sdpa_kernel([be]):
                    return F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal)
            call()
            torch.cuda.synchronize()
            return name, call
        except Exception:
            continue
    return None, None


def loglog_slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y and y > 0]
    if len(pts) < 3:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def fmt(x, nd=3):
    return "OOM" if x is None else ("N/A" if x == "NA" else f"{x:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 512, 2048, 8192, 32768])
    ap.add_argument("--ds", type=int, nargs="+", default=[64])
    ap.add_argument("--dtypes", nargs="+", default=["bf16"])
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--fused", action="store_true", help="反向用单 kernel 版")
    ap.add_argument("--flush-l2", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    causal = not args.no_causal
    torch.manual_seed(0)

    from flash_attn_autograd import flash_attn_func, make_fwd_runner, use_cute_bwd
    use_cute_bwd(fused=args.fused)

    # 编译时间：单独报告，且在任何计时之前
    q, k, v = (torch.randn(1, 256, 64, device=DEV, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    t0 = time.perf_counter()
    flash_attn_func(q, k, v, causal).backward(torch.randn(1, 256, 64, device=DEV, dtype=torch.bfloat16))
    torch.cuda.synchronize()
    print(f"[compile] FA fwd+bwd 首次调用（含 JIT）：{time.perf_counter() - t0:.1f} s\n")

    for dtname in args.dtypes:
        dt = DT[dtname]
        peak = cublas_peak_tflops(dt)
        print(f"## dtype={dtname}：cuBLAS 4096³ 峰值 = {peak:.1f} TFLOP/s（= 100%）")
        for d in args.ds:
            supported = (d == 64 and dt == torch.bfloat16)
            print(f"\n### d={d}，causal={causal}，CuTe FA 支持：{supported}\n")
            print("| N | naive fwd ms | SDPA fwd ms | FA fwd e2e ms | FA fwd kernel ms | FA fwd TFLOP/s | 利用率 | "
                  "naive f+b ms | FA f+b ms | FA f+b TFLOP/s |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            mem_rows, fa_mem = [], []
            sdpa_name = None
            for N in args.seqs:
                q, k, v, dO = (torch.randn(1, N, d, device=DEV, dtype=dt) for _ in range(4))
                f_fwd = (2.0 if causal else 4.0) * N * N * d      # 前向 FLOPs
                f_fb = 3.5 * f_fwd                                # fwd + bwd(2.5×)

                def naive_f():
                    with torch.no_grad():
                        naive_attention(q, k, v, causal)

                def naive_fb():
                    qg, kg, vg = (x.detach().clone().requires_grad_() for x in (q, k, v))
                    naive_attention(qg, kg, vg, causal).backward(dO)

                r = {}
                r["naive_f"] = guard(lambda: timeit(naive_f, iters=args.iters, flush=args.flush_l2))
                r["naive_fb"] = guard(lambda: timeit(naive_fb, iters=args.iters, flush=args.flush_l2))
                r["m_naive_f"] = guard(lambda: peak_mem_gib(naive_f))
                r["m_naive_fb"] = guard(lambda: peak_mem_gib(naive_fb))

                # SDPA（注明后端）
                name, call = pick_sdpa_backend(q[:, None], k[:, None], v[:, None], causal)
                sdpa_name = sdpa_name or name
                r["sdpa"] = guard(lambda: timeit(call, iters=args.iters, flush=args.flush_l2)) if call else "NA"

                if supported:
                    def fa_f():
                        with torch.no_grad():
                            flash_attn_func(q, k, v, causal)

                    run, _, _ = make_fwd_runner(q, k, v, causal)

                    def fa_fb():
                        qg, kg, vg = (x.detach().clone().requires_grad_() for x in (q, k, v))
                        flash_attn_func(qg, kg, vg, causal).backward(dO)

                    r["fa_e2e"] = guard(lambda: timeit(fa_f, iters=args.iters, flush=args.flush_l2))
                    r["fa_k"] = guard(lambda: timeit(run, iters=args.iters, flush=args.flush_l2))
                    r["fa_fb"] = guard(lambda: timeit(fa_fb, iters=args.iters, flush=args.flush_l2))
                    r["m_fa_f"] = guard(lambda: peak_mem_gib(fa_f))
                    r["m_fa_fb"] = guard(lambda: peak_mem_gib(fa_fb))
                    tf = f_fwd / r["fa_k"] / 1e9 if r["fa_k"] else None
                    tf_fb = f_fb / r["fa_fb"] / 1e9 if r["fa_fb"] else None
                else:
                    r["fa_e2e"] = r["fa_k"] = r["fa_fb"] = "NA"
                    tf = tf_fb = "NA"
                    r["m_fa_f"] = r["m_fa_fb"] = "NA"

                util = (tf / peak * 100) if isinstance(tf, float) else tf
                print(f"| {N} | {fmt(r['naive_f'])} | {fmt(r['sdpa'])} | {fmt(r['fa_e2e'])} | {fmt(r['fa_k'])} | "
                      f"{fmt(tf, 1)} | {fmt(util, 1) if isinstance(util, float) else fmt(util)}% | "
                      f"{fmt(r['naive_fb'])} | {fmt(r['fa_fb'])} | {fmt(tf_fb, 1)} |")
                mem_rows.append((N, r))
                if isinstance(r["m_fa_f"], tuple):
                    fa_mem.append((N, r["m_fa_f"][0]))
                del q, k, v, dO
                torch.cuda.empty_cache()

            print(f"\nSDPA 实际使用的后端：{sdpa_name}")
            print("\n显存（增量峰值 GiB / 绝对峰值 GiB）：\n")
            print("| N | naive fwd | naive f+b | FA fwd | FA f+b |")
            print("|---|---|---|---|---|")
            f2 = lambda m: "OOM" if m is None else ("N/A" if m == "NA" else f"{m[0]:.3f} / {m[1]:.3f}")
            for N, r in mem_rows:
                print(f"| {N} | {f2(r['m_naive_f'])} | {f2(r['m_naive_fb'])} | {f2(r['m_fa_f'])} | {f2(r['m_fa_fb'])} |")
            if supported:
                sl = loglog_slope([n for n, _ in fa_mem], [m for _, m in fa_mem])
                sn = loglog_slope([n for n, r in mem_rows if isinstance(r["m_naive_f"], tuple)],
                                  [r["m_naive_f"][0] for n, r in mem_rows if isinstance(r["m_naive_f"], tuple)])
                print(f"\nlog-log 斜率（增量峰值显存 vs N）：FA fwd = {sl if sl is None else round(sl, 2)}，"
                      f"naive fwd = {sn if sn is None else round(sn, 2)}（预测 FA≈1，基线≈2）")


if __name__ == "__main__":
    main()