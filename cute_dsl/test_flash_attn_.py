"""FLASH_ATTN_naive 的完整测试：正确性（O 与 LSE）、边界、数值稳定、越界保护，可选性能对比。

运行：
    python test_flash_attn.py              # 全部正确性用例
    python test_flash_attn.py --bench      # 额外跑性能（对比 torch SDPA）
    python test_flash_attn.py --no-dynamic # 每个形状单独编译（动态形状编译有问题时用）
"""
import argparse
import math
import sys

import torch
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn_naive_ import FLASH_ATTN_naive, BQ, BK, D

LOG2E = math.log2(math.e)
DEV = "cuda"
BF16 = torch.bfloat16
SENTINEL = -12345.0          # M 的 pad 行哨兵：kernel 不应写这些行

# 编译缓存：动态形状模式只编译一次；静态模式按形状各编译一次
_compiled = {}


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def to_cute(t: torch.Tensor, dynamic: bool, leading_dim: int):
    """torch 张量 → cute 张量。assumed_align=16 让编译器知道可以做 16B 向量访问；
    mark_layout_dynamic 让形状/stride 变成运行时值，这样一份编译可复用到所有形状。"""
    ct = from_dlpack(t, assumed_align=16)
    if dynamic:
        ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def make_input(B, N, scale, gen, poison):
    """生成 (B,N,D) 的 bf16 输入。
    poison=True：把它放进一块更大的缓冲区，尾部填满 NaN，再取前面的连续视图。
    若 kernel 的 pred / 补零有 bug，读到尾部的 NaN 会污染结果（只会作用于最后一个 batch 的尾部）。"""
    x = (torch.randn(B, N, D, device=DEV, generator=gen) * scale).to(BF16)
    if not poison:
        return x
    n = B * N * D
    buf = torch.full((n + 2 * BK * D,), float("nan"), dtype=BF16, device=DEV)
    view = buf[:n].view(B, N, D)
    view.copy_(x)
    return view                       # 视图持有 buf 的引用，不会被释放


def reference(Q, K, V):
    """fp32 参考：O = softmax(QK^T/√d) V，lse2 = logsumexp(S) * log2(e)。
    lse2 与 kernel 的 m + log2(d)（base-2、m 已含 qk_scale）是同一个量。"""
    S = Q.float() @ K.float().transpose(-1, -2) / math.sqrt(D)
    O = torch.softmax(S, dim=-1) @ V.float()
    lse2 = torch.logsumexp(S, dim=-1) * LOG2E
    return O, lse2


def run_kernel(Q, K, V, O_full, M_full, dynamic):
    """调用 kernel（首次调用时编译）。"""
    args = (to_cute(Q, dynamic, 2), to_cute(K, dynamic, 2), to_cute(V, dynamic, 2),
            to_cute(O_full, dynamic, 2), to_cute(M_full, dynamic, 1))
    key = "dyn" if dynamic else (tuple(Q.shape), tuple(K.shape), tuple(O_full.shape))
    if key not in _compiled:
        _compiled[key] = cute.compile(FLASH_ATTN_naive(), *args)
    _compiled[key](*args)
    torch.cuda.synchronize()          # 保证 kernel 的错误在这里暴露，而不是之后某处


def diagnose(o_ker, o_ref, Nq, Nk):
    """失败时打印误差分布，帮助定位问题类型。"""
    err = (o_ker.float() - o_ref).abs()
    flat = err.reshape(-1).nan_to_num(nan=float("inf")).argmax().item()
    b, r, c = flat // (err.shape[1] * D), (flat // D) % err.shape[1], flat % D
    print(f"      最大误差位置 batch={b} row={r} col={c}，"
          f"kernel={o_ker[b, r, c].item():.4f} ref={o_ref[b, r, c].item():.4f}")
    n_bad_rows = (~torch.isfinite(o_ker.float()).all(-1)).sum().item()
    print(f"      含 NaN/Inf 的行数: {n_bad_rows}")
    groups = [err[:, i:i + 16].max().item() for i in range(0, min(Nq, 128), 16)]
    print("      按 16 行（=1 个 warp）分组的最大误差:", [f"{g:.3f}" for g in groups])


# ----------------------------------------------------------------------------
# 单个用例
# ----------------------------------------------------------------------------
def run_case(name, B, Nq, Nk, *, q_scale=1.0, zero_q=False, poison=True,
             dynamic=True, seed=0, o_atol=2e-2, lse_atol=5e-3):
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    Q = make_input(B, Nq, q_scale, gen, poison)
    K = make_input(B, Nk, 1.0, gen, poison)
    V = make_input(B, Nk, 1.0, gen, poison)
    if zero_q:
        Q.zero_()                     # Q=0 → S 全 0 → 均匀 softmax：O=mean(V)，lse2=log2(Nk)

    # 输出按 Q tile 向上 pad，pad 行预填 NaN / 哨兵，用来检查 kernel 没越界写
    Np = (Nq + BQ - 1) // BQ * BQ
    O_full = torch.full((B, Np, D), float("nan"), dtype=BF16, device=DEV)
    M_full = torch.full((B, Np), SENTINEL, dtype=torch.float32, device=DEV)

    try:
        run_kernel(Q, K, V, O_full, M_full, dynamic)
    except Exception as e:             # 编译或运行报错：直接判失败并给出原因
        print(f"[FAIL] {name:<28} B={B} Nq={Nq} Nk={Nk}  运行出错: {type(e).__name__}: {e}")
        return False

    o_ref, lse_ref = reference(Q, K, V)
    o_ker = O_full[:, :Nq]
    lse_ker = M_full[:, :Nq]

    # ---- 检查 1：数值正确性 ----
    finite = torch.isfinite(o_ker.float()).all().item() and torch.isfinite(lse_ker).all().item()
    o_err = (o_ker.float() - o_ref).abs().max().item() if finite else float("inf")
    lse_err = (lse_ker - lse_ref).abs().max().item() if finite else float("inf")
    ok_o, ok_lse = o_err <= o_atol, lse_err <= lse_atol

    # ---- 检查 2：pad 行没被写（写回 predicate） ----
    if Np > Nq:
        ok_pad = bool(torch.isnan(O_full[:, Nq:].float()).all().item()
                      and (M_full[:, Nq:] == SENTINEL).all().item())
    else:
        ok_pad = True

    ok = finite and ok_o and ok_lse and ok_pad
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name:<28} B={B} Nq={Nq:<4} Nk={Nk:<5} "
          f"O误差={o_err:.2e} LSE误差={lse_err:.2e} 有限={finite} pad完好={ok_pad}")
    if not ok:
        if finite:
            diagnose(o_ker, o_ref, Nq, Nk)
        elif not finite:
            print("      输出含 NaN/Inf：优先怀疑 mask（-inf 行）、V 补零、毒化数据被读入")
    return ok


# ----------------------------------------------------------------------------
# 性能对比（可选）
# ----------------------------------------------------------------------------
def bench(B=8, N=2048, iters=50, dynamic=True):
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    Q, K, V = (make_input(B, N, 1.0, gen, poison=False) for _ in range(3))
    O = torch.empty(B, N, D, dtype=BF16, device=DEV)
    M = torch.empty(B, N, dtype=torch.float32, device=DEV)

    def t_kernel():
        run_kernel(Q, K, V, O, M, dynamic)

    def t_sdpa():
        F.scaled_dot_product_attention(Q[:, None], K[:, None], V[:, None])

    flops = 4.0 * B * N * N * D            # QK^T 与 PV 各 2*N*N*D
    for name, fn in (("本 kernel", t_kernel), ("torch SDPA", t_sdpa)):
        for _ in range(5):
            fn()                           # 预热（含首次编译）
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        print(f"[bench] {name:<10} B={B} N={N} D={D}: {ms:.3f} ms, {flops / ms / 1e9:.1f} TFLOPs")


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--no-dynamic", action="store_true", help="每个形状单独编译")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dyn = not args.no_dynamic
    if not torch.cuda.is_available():
        print("需要 CUDA GPU（sm80+，bf16 mma）")
        sys.exit(1)

    cases = [
        # (名称, B, Nq, Nk, 额外参数)
        ("单tile",                1,   64,   64, {}),
        ("多batch",               2,   64,   64, {}),
        ("多Q-tile",              1,  128,   64, {}),
        ("多K-tile(整除)",        1,   64,  256, {}),
        ("Nk非BK倍数",            1,   64,  100, {}),
        ("Nq非BQ倍数",            1,  100,   64, {}),
        ("双向非整除",            2,  130,  257, {}),
        ("长序列(online递推)",    3,   64, 1000, {}),
        ("Nq=1",                  1,    1,  128, {}),
        ("Nk=1",                  1,   64,    1, {}),
        ("极小",                  1,   17,    5, {}),
        ("大幅值(safe softmax)",  1,  100,  300, {"q_scale": 8.0}),
        ("Q=0 解析解",            1,   64,  200, {"zero_q": True}),
        ("无毒化基线",            1,   64,  100, {"poison": False}),
    ]

    results = [run_case(n, B, nq, nk, dynamic=dyn, seed=args.seed, **kw)
               for (n, B, nq, nk, kw) in cases]

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} 通过")
    if args.bench:
        bench(dynamic=dyn)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()