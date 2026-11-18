"""FLASH_ATTN_naive 终版测试：O 与 LSE 的正确性、边界、causal、2-stage 流水、swizzle 等价性、越界保护，可选性能对比。

用法：
    python test_flash_attn.py                  # 全部用例
    python test_flash_attn.py --filter causal  # 只跑名称含 "causal" 的用例
    python test_flash_attn.py --bench          # 额外跑性能（对比 torch SDPA，swizzle/causal 各开关）
    python test_flash_attn.py --no-dynamic     # 每个形状单独编译（动态形状编译有问题时用）
退出码：全部通过为 0，否则为 1。
"""
import argparse
import math
import sys

import torch
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn_naive import FLASH_ATTN_naive, BQ, BK, D

LOG2E = math.log2(math.e)
DEV = "cuda"
BF16 = torch.bfloat16
SENTINEL = -12345.0           # M 的 pad 行哨兵：kernel 不应写这些行

_compiled = {}                # 编译缓存


# =============================================================================
# 工具函数
# =============================================================================
def to_cute(t: torch.Tensor, dynamic: bool, leading_dim: int):
    """torch → cute 张量。assumed_align=16 让编译器知道可用 16B 向量访问；
    mark_layout_dynamic 把形状/stride 变成运行时值，一份编译可复用到所有形状。"""
    ct = from_dlpack(t, assumed_align=16)
    if dynamic:
        ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def make_input(B, N, scale, gen, poison):
    """生成 (B,N,D) 的 bf16 输入。
    poison=True：放进更大的缓冲区，缓冲区尾部填 NaN，再取前面的连续视图。
    kernel 的 pred/补零有 bug 时会读到尾部 NaN 而污染结果（只作用于最后一个 batch 的尾部，
    所以单 batch 用例对越界读更敏感）。"""
    x = (torch.randn(B, N, D, device=DEV, generator=gen) * scale).to(BF16)
    if not poison:
        return x
    n = B * N * D
    buf = torch.full((n + 2 * BK * D,), float("nan"), dtype=BF16, device=DEV)
    view = buf[:n].view(B, N, D)
    view.copy_(x)
    return view               # 视图持有 buf 的引用，不会被释放


def reference(Q, K, V, causal=False):
    """fp32 参考。lse2 = logsumexp(S) * log2(e)，与 kernel 的 m + log2(d)（base-2，m 已含 qk_scale）同一个量。
    causal：左上对齐，保留 j <= i（与 torch SDPA is_causal 一致）；Nq>Nk 时 i>=Nk 的行看到全部 key。"""
    S = Q.float() @ K.float().transpose(-1, -2) / math.sqrt(D)
    if causal:
        Nq, Nk = S.shape[-2:]
        keep = torch.ones(Nq, Nk, dtype=torch.bool, device=S.device).tril()
        S = S.masked_fill(~keep, float("-inf"))
    O = torch.softmax(S, dim=-1) @ V.float()
    lse2 = torch.logsumexp(S, dim=-1) * LOG2E
    return O, lse2


def run_kernel(Q, K, V, O_full, M_full, dynamic, causal=False, swizzle=True):
    """调用 kernel（首次调用时编译）。编译缓存键包含 causal / swizzle。"""
    args = (to_cute(Q, dynamic, 2), to_cute(K, dynamic, 2), to_cute(V, dynamic, 2),
            to_cute(O_full, dynamic, 2), to_cute(M_full, dynamic, 1))
    flags = (causal, swizzle)
    if dynamic:
        key = ("dyn",) + flags
    else:
        key = (tuple(Q.shape), tuple(K.shape), tuple(O_full.shape)) + flags
    if key not in _compiled:
        _compiled[key] = cute.compile(FLASH_ATTN_naive(causal=causal, use_swizzle=swizzle), *args)
    _compiled[key](*args)
    torch.cuda.synchronize()  # 让 kernel 内的错误在此处暴露


def diagnose(o_ker, o_ref, Nq):
    """失败时打印误差分布，帮助判断问题类型。"""
    err = (o_ker.float() - o_ref).abs()
    flat = err.reshape(-1).nan_to_num(nan=float("inf")).argmax().item()
    b, r, c = flat // (err.shape[1] * D), (flat // D) % err.shape[1], flat % D
    print(f"      最大误差位置 batch={b} row={r} col={c}，"
          f"kernel={o_ker[b, r, c].item():.4f} ref={o_ref[b, r, c].item():.4f}")
    n_bad_rows = (~torch.isfinite(o_ker.float()).all(-1)).sum().item()
    print(f"      含 NaN/Inf 的行数: {n_bad_rows}")
    groups = [err[:, i:i + 16].max().item() for i in range(0, min(Nq, 128), 16)]
    print("      按 16 行（=1 个 warp）分组的最大误差:", [f"{g:.3f}" for g in groups])


# =============================================================================
# 单个用例
# =============================================================================
def run_case(name, B, Nq, Nk, *, causal=False, swizzle=True, q_scale=1.0, zero_q=False,
             poison=True, dynamic=True, seed=0, o_atol=2e-2, lse_atol=5e-3):
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    Q = make_input(B, Nq, q_scale, gen, poison)
    K = make_input(B, Nk, 1.0, gen, poison)
    V = make_input(B, Nk, 1.0, gen, poison)
    if zero_q:
        Q.zero_()             # Q=0 → S 全 0 → 均匀 softmax（causal 时为前缀均匀），有解析解

    # 输出按 Q tile 向上 pad；pad 行预填 NaN/哨兵，用来检查 kernel 没越界写
    Np = (Nq + BQ - 1) // BQ * BQ
    O_full = torch.full((B, Np, D), float("nan"), dtype=BF16, device=DEV)
    M_full = torch.full((B, Np), SENTINEL, dtype=torch.float32, device=DEV)

    tag = f"{'C' if causal else '-'}{'S' if swizzle else '-'}"     # C=causal, S=swizzle
    head = f"{name:<26} [{tag}] B={B} Nq={Nq:<4} Nk={Nk:<5}"
    try:
        run_kernel(Q, K, V, O_full, M_full, dynamic, causal, swizzle)
    except Exception as e:
        print(f"[FAIL] {head} 运行出错: {type(e).__name__}: {e}")
        return False

    o_ref, lse_ref = reference(Q, K, V, causal)
    o_ker = O_full[:, :Nq]
    lse_ker = M_full[:, :Nq]

    # 检查 1：数值正确性
    finite = torch.isfinite(o_ker.float()).all().item() and torch.isfinite(lse_ker).all().item()
    o_err = (o_ker.float() - o_ref).abs().max().item() if finite else float("inf")
    lse_err = (lse_ker - lse_ref).abs().max().item() if finite else float("inf")
    ok_val = finite and o_err <= o_atol and lse_err <= lse_atol

    # 检查 2：pad 行没被写（写回 predicate）
    if Np > Nq:
        ok_pad = bool(torch.isnan(O_full[:, Nq:].float()).all().item()
                      and (M_full[:, Nq:] == SENTINEL).all().item())
    else:
        ok_pad = True

    ok = ok_val and ok_pad
    print(f"[{'PASS' if ok else 'FAIL'}] {head} "
          f"O误差={o_err:.2e} LSE误差={lse_err:.2e} 有限={finite} pad完好={ok_pad}")
    if not ok:
        if finite:
            diagnose(o_ker, o_ref, Nq)
        else:
            print("      输出含 NaN/Inf：优先怀疑 mask、V 补零、毒化数据被读入、整行被 mask")
    return ok


# =============================================================================
# swizzle 等价性：swizzle 只改 smem 地址、不改任何计算顺序，结果应当逐位相同
# =============================================================================
def test_swizzle_equiv(B=2, Nq=130, Nk=257, causal=True, dynamic=True, seed=1):
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    Q = make_input(B, Nq, 1.0, gen, poison=False)
    K = make_input(B, Nk, 1.0, gen, poison=False)
    V = make_input(B, Nk, 1.0, gen, poison=False)
    Np = (Nq + BQ - 1) // BQ * BQ
    outs = []
    try:
        for sw in (False, True):
            O = torch.zeros(B, Np, D, dtype=BF16, device=DEV)
            M = torch.zeros(B, Np, dtype=torch.float32, device=DEV)
            run_kernel(Q, K, V, O, M, dynamic, causal, sw)
            outs.append((O[:, :Nq].clone(), M[:, :Nq].clone()))
    except Exception as e:
        print(f"[FAIL] swizzle 等价性 (causal={causal}) 运行出错: {type(e).__name__}: {e}")
        return False
    same = torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    print(f"[{'PASS' if same else 'FAIL'}] swizzle 开/关 逐位一致 (causal={causal})")
    if not same:
        d = (outs[0][0].float() - outs[1][0].float()).abs().max().item()
        print(f"      O 最大差异 {d:.3e}：sVt 的 swizzle 与 sV 不一致，或 smem 基址未按 1024B 对齐")
    return same


# =============================================================================
# 性能对比（可选）
# =============================================================================
def bench(B=8, N=2048, iters=50, dynamic=True):
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    Q, K, V = (make_input(B, N, 1.0, gen, poison=False) for _ in range(3))
    O = torch.empty(B, N, D, dtype=BF16, device=DEV)
    M = torch.empty(B, N, dtype=torch.float32, device=DEV)

    def timeit(fn):
        for _ in range(5):
            fn()                                   # 预热（含首次编译）
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    print(f"\n[bench] B={B} N={N} D={D}（causal 的 TFLOPs 按实际有效计算量折半）")
    for causal in (False, True):
        flops = 4.0 * B * N * N * D * (0.5 if causal else 1.0)
        for sw in (False, True):
            ms = timeit(lambda: run_kernel(Q, K, V, O, M, dynamic, causal, sw))
            print(f"  本 kernel  causal={causal!s:<5} swizzle={sw!s:<5}: {ms:7.3f} ms, {flops / ms / 1e9:6.1f} TFLOPs")
        ms = timeit(lambda: F.scaled_dot_product_attention(Q[:, None], K[:, None], V[:, None], is_causal=causal))
        print(f"  torch SDPA causal={causal!s:<5}                : {ms:7.3f} ms, {flops / ms / 1e9:6.1f} TFLOPs")


# =============================================================================
# 用例表与入口
# =============================================================================
def build_cases():
    # (名称, B, Nq, Nk, 额外参数)
    return [
        # ---- 基础正确性 ----
        ("单tile",                1,   64,   64, {}),
        ("多batch",               2,   64,   64, {}),
        ("多Q-tile",              1,  128,   64, {}),
        ("多K-tile(整除)",        1,   64,  256, {}),
        # ---- 边界 ----
        ("Nk非BK倍数",            1,   64,  100, {}),
        ("Nq非BQ倍数",            1,  100,   64, {}),
        ("双向非整除",            2,  130,  257, {}),
        ("长序列(online递推)",    3,   64, 1000, {}),
        ("Nq=1",                  1,    1,  128, {}),
        ("Nk=1",                  1,   64,    1, {}),
        ("极小",                  1,   17,    5, {}),
        # ---- 数值稳定 / 解析解 ----
        ("大幅值(safe softmax)",  1,  100,  300, {"q_scale": 8.0}),
        ("Q=0 解析解",            1,   64,  200, {"zero_q": True}),
        ("无毒化基线",            1,   64,  100, {"poison": False}),
        # ---- 2-stage 流水：tile 数 1/2/3/5，覆盖“无预取”“stage 回绕”“尾块” ----
        ("流水:1 tile",           1,   64,   64, {}),
        ("流水:2 tile",           1,   64,  128, {}),
        ("流水:3 tile(回绕)",     1,   64,  192, {}),
        ("流水:5 tile+尾巴",      2,   64,  300, {}),
        # ---- causal ----
        ("causal 单tile",         1,   64,   64, {"causal": True}),
        ("causal 多tile 方阵",    2,  256,  256, {"causal": True}),
        ("causal 非整除",         2,  130,  257, {"causal": True}),
        ("causal Nq>Nk",          1,  200,   80, {"causal": True}),   # 行 i>=Nk 看到全部 key
        ("causal Nq<Nk",          1,   70,  300, {"causal": True}),
        ("causal Nq=1",           1,    1,  128, {"causal": True}),
        ("causal 大幅值",         1,  100,  300, {"causal": True, "q_scale": 8.0}),
        ("causal Q=0",            1,   64,  200, {"causal": True, "zero_q": True}),
        # ---- 无 swizzle 对照路径 ----
        ("无swizzle",             1,  130,  257, {"swizzle": False}),
        ("无swizzle causal",      1,  130,  257, {"swizzle": False, "causal": True}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--no-dynamic", action="store_true", help="每个形状单独编译")
    ap.add_argument("--filter", type=str, default=None, help="只跑名称包含该子串的用例")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dyn = not args.no_dynamic

    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        sys.exit(1)
    if torch.cuda.get_device_capability()[0] < 8:
        print("需要 sm80+（bf16 mma + cp.async）")
        sys.exit(1)

    cases = [c for c in build_cases() if args.filter is None or args.filter in c[0]]
    results = [run_case(n, B, nq, nk, dynamic=dyn, seed=args.seed, **kw)
               for (n, B, nq, nk, kw) in cases]

    if args.filter is None:                       # 等价性测试：逐位一致，不受 --seed 影响
        results.append(test_swizzle_equiv(causal=True, dynamic=dyn))
        results.append(test_swizzle_equiv(causal=False, dynamic=dyn))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} 通过")
    if args.bench:
        bench(dynamic=dyn)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()