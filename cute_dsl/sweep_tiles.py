"""tile 扫描：固定 d=64、bf16、causal，对 (Bq, Bk, stages) 逐个配置编译、校验、计时。
用法：python sweep_tiles.py [--seqs 4096 16384] [--iters 20]"""
import argparse
import math
import statistics
import time

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn_naive import FLASH_ATTN_naive

DEV, D, BF16 = "cuda", 64, torch.bfloat16
NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count
CONFIGS = [  # (Bq, Bk, stages)：作业要求 5 组；这里多放几组 stages 的对照
    (32, 32, 2), (64, 32, 2), (64, 64, 2), (128, 64, 2), (128, 128, 2),
    (64, 64, 1), (64, 64, 3), (128, 64, 3),
]


def event_ms(fn, warmup=3, iters=20):
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


def cargs(Q, K, V, O, L):
    c = lambda t, ld: from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=ld)
    return (c(Q, 2), c(K, 2), c(V, 2), c(O, 2), c(L, 1))


def ref_causal(q, k, v):
    S = q.float() @ k.float().transpose(-1, -2) / math.sqrt(D)
    keep = torch.ones(S.shape[-2:], dtype=torch.bool, device=S.device).tril()
    return torch.softmax(S.masked_fill(~keep, float("-inf")), -1) @ v.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0)

    rows = []
    for (bq, bk, st) in CONFIGS:
        smem_kb = (bq + 2 * st * bk) * D * 2 / 1024
        rec = dict(bq=bq, bk=bk, st=st, warps=bq // 16, smem_kb=smem_kb, status="OK", compile_s=None, ms={}, tf={})
        try:
            kern = FLASH_ATTN_naive(causal=True, bq=bq, bk=bk, stages=st)
            # 先用校验用的小形状编译（动态形状：同一份编译可复用到所有 N）
            Nc = 1000
            Qc, Kc, Vc = (torch.randn(1, Nc, D, device=DEV).to(BF16) for _ in range(3))
            Oc = torch.empty_like(Qc)
            Lc = torch.empty(1, Nc, dtype=torch.float32, device=DEV)
            a = cargs(Qc, Kc, Vc, Oc, Lc)
            t0 = time.perf_counter()
            fn = cute.compile(kern, *a)
            rec["compile_s"] = time.perf_counter() - t0
            fn(*a)
            torch.cuda.synchronize()
            err = ((Oc.float() - ref_causal(Qc, Kc, Vc)).abs().max() / 1.0).item()
            if not (err < 2e-2):
                rec["status"] = f"WRONG(err={err:.2e})"
            else:
                for N in args.seqs:
                    Q, K, V = (torch.randn(1, N, D, device=DEV).to(BF16) for _ in range(3))
                    O = torch.empty_like(Q)
                    L = torch.empty(1, N, dtype=torch.float32, device=DEV)
                    a = cargs(Q, K, V, O, L)
                    ms = event_ms(lambda: fn(*a), iters=args.iters)
                    rec["ms"][N] = ms
                    rec["tf"][N] = 2.0 * N * N * D / ms / 1e9           # causal forward：2·N²·d
        except Exception as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else ""
            rec["status"] = f"FAIL({type(e).__name__}: {first[:90]})"      # 编译失败也是数据
        rows.append(rec)
        torch.cuda.empty_cache()

    print(f"\nSM 数 = {NUM_SM}；forward、causal、d=64、bf16；TFLOP/s = 2·N²·d / 时间\n")
    head = "| Bq | Bk | stages | warps | smem(KB) | 编译(s) | 状态 |"
    for N in args.seqs:
        head += f" N={N} blocks | ms | TFLOP/s |"
    print(head)
    print("|" + "---|" * (7 + 3 * len(args.seqs)))
    for r in rows:
        line = f"| {r['bq']} | {r['bk']} | {r['st']} | {r['warps']} | {r['smem_kb']:.0f} | " \
               f"{r['compile_s'] if r['compile_s'] is None else format(r['compile_s'], '.1f')} | {r['status']} |"
        for N in args.seqs:
            blocks = math.ceil(N / r["bq"])
            if N in r["ms"]:
                line += f" {blocks} ({blocks / NUM_SM:.2f}x SM) | {r['ms'][N]:.3f} | {r['tf'][N]:.1f} |"
            else:
                line += f" {blocks} | - | - |"
        print(line)


if __name__ == "__main__":
    main()