#!/usr/bin/env python
"""3.6 attention 微基准（vanilla vs torch.compile）。

测量对象：
  - vanilla = eager 三段式 attention（scores→softmax→av），与模型内 kernel 序列同构。
    刻意手写而不是调 sdpa：sdpa 内部已是融合实现，当基线就测不出"未融合"的差距。
  - compile = torch.compile(attention)，dynamo 抓图 + inductor 融合 softmax 的
    sub/exp/div 逐元素链，再执行。compile 能省时间，但【不消除 S² 张量的
    materialize】——scores 照样真分配显存，因为反传仍需要它（实测验证这点）。

显存机理（本实验的核心，先读再跑）：
  scores = (Q@K^T)/sqrt(d)   除法是 out-of-place：除法瞬间，QK^T 临时量与 scores
                             【同时各占一份 B·S²】→ 2·B·S²·4B 的瞬态并存
  p = softmax(scores)        softmax 前向只把【输出 p】存入 autograd（反传公式
                             dL/dx_i = p_i·(g_i − Σ_j g_j·p_j) 只需输出），
                             输入 scores 不被保存，函数返回后引用归零、释放回 allocator
  p @ V                      matmul 反传需保存 p 和 V——p 与 softmax 输出共享同一存储
  ⇒ 推论：反传真正【长期保存】的 S² 张量只有 1 份（p）；
    作业的 2·B·S²·4B 预测描述的是 forward 中的【瞬态峰值】——
    OOM 恰恰咬在这一时刻（div 或 softmax 中途），而不是反传长期占用。
  ⇒ 所以脚本对每个配置记录两个显存数：
       mem_pre_bwd  （backward 前稳态占用 ≈ 1×S² + QKV）→ 填作业"backward 前显存"表
       peak_fwd     （forward 期间瞬态峰值 ≈ 2×S² + QKV）→ 与 2·B·S²·4B 手算对账
    两个数各有各的对账意义，混用会得出"实测只有理论一半"或"理论错了"的错误结论。

OOM 预测（B=8, fp32, 16GB 卡）：
  S=16384: 2×S² = 8×16384²×4B×2 ≈ 17.2 GB(≈16.0 GiB) > 可用显存 → forward 中途 OOM，
           且与 d_model 无关（QKV 只有 B·S·d 线性量，scores 的 S² 项独大）→ 整列 OOM
  S=8192 : forward 瞬态 ≈ 8.6 GB 可过；但 backward 中 softmax 反传要现场开 dP（S²）
           和中间量（S²），叠加已保存的 p → 步内峰值 ≈ 3×S² ≈ 12.9 GB，贴着上限，
           若此处 OOM 是正常现象，报告里注明"OOM 点前移到 backward"即可。
"""
import argparse
import gc
import statistics
import time

import torch

# 基线口径与 3.2/3.5/12 章一致：TF32 关，纯 CUDA-core fp32（compile 版 inductor 也尊重此设置）
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


# ----------------------------------------------------------------------------- 被测函数
def attention(Q, K, V):
    """教科书三段式。每一行的显存行为见文件头 docstring 的机理表。
    不加 causal mask：作业的 2·B·S²·4B 预测对应无 mask 版；
    若加 masked_fill 会多出一份 S² 的 masked 副本，实测对账时要额外说明。"""
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / d_k ** 0.5   # (B,S,S)：瞬态并存点 1（div 时两份 S²）
    p = torch.softmax(scores, dim=-1)                 # (B,S,S)：瞬态并存点 2；p 被 autograd 长期保存
    return p @ V                                      # 反传需保存 p、V


# ----------------------------------------------------------------------------- 单配置测量
def measure(Q, K, V, fn, warmup, iters):
    """返回 dict 或 None(OOM)。
    计时设计：3 个 event 切两段（fwd / bwd），每步末尾一次 sync——与 12.2 的
    phase_step 同一套纪律，只是这里 loss 换成 out.sum()（标量损失，保证反向跑全图）。
    显存读数全部是 host 侧计数器（allocator 记账），不需要 GPU sync，
    穿插在计时循环里不扰动 event 时间线。"""
    t0 = torch.cuda.Event(enable_timing=True)   # event 建一次复用；创建开销在计时段之外
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    try:
        for _ in range(warmup):
            # compile 时：首次调用触发 dynamo 抓图 + inductor 编译（fwd 图），
            # 首次 backward 触发 bwd 图编译——两次编译都被 warmup 吸收，绝不进测量区
            fn(Q, K, V).sum().backward()
        torch.cuda.synchronize()

        f_ts, b_ts = [], []
        mem_pre, peak_f, peak_step = [], [], []
        for _ in range(iters):
            torch.cuda.reset_peak_memory_stats()
            t0.record()
            out = fn(Q, K, V)
            loss = out.sum()
            t1.record()                                    # 此刻即"backward 前"
            mem_pre.append(torch.cuda.memory_allocated() / 2**30)      # backward 前稳态占用
            peak_f.append(torch.cuda.max_memory_allocated() / 2**30)   # forward 瞬态峰值
            loss.backward()
            t2.record()
            torch.cuda.synchronize()          # sync 之后 elapsed_time 才可读
            peak_step.append(torch.cuda.max_memory_allocated() / 2**30)
            Q.grad = K.grad = V.grad = None
            # ↓↓↓ 缺的就是这两行：分段耗时必须在 sync 之后、从 event 读出并存下
            f_ts.append(t0.elapsed_time(t1))  # forward 段 ms
            b_ts.append(t1.elapsed_time(t2))  # backward 段 ms
        return dict(fwd=statistics.median(f_ts), bwd=statistics.median(b_ts),
                    mem_pre=statistics.median(mem_pre),
                    peak_fwd=statistics.median(peak_f),
                    peak_step=statistics.median(peak_step))

    except torch.cuda.OutOfMemoryError:
        # OOM 是预测的确认、是数据，不是崩溃：清场后继续下一个配置。
        # OOM 可能发生在 forward 中途（div/softmax 瞬态超限）——正是机理预测的位置
        torch.cuda.empty_cache()
        return None


def run_cfg(d, S, args, compiled):
    """构建 → 测量 → 彻底清场。返回结果 dict 或 None。
    清场三连（del→gc→empty_cache）缺一不可：上一个配置的残留缓存会让
    下一个配置虚假 OOM，直接污染 OOM 首现点的判定。"""
    torch.manual_seed(args.seed)
    try:                                            # QKV 构建本身也可能 OOM（极大配置）
        Q = torch.randn(args.batch, S, d, device="cuda", requires_grad=True)
        K = torch.randn(args.batch, S, d, device="cuda", requires_grad=True)
        V = torch.randn(args.batch, S, d, device="cuda", requires_grad=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    if compiled:
        fn = torch.compile(attention)               # 只编译 attention 本体；.sum() 不进图
    else:
        fn = attention
    r = measure(Q, K, V, fn, args.warmup, args.iters)
    # 清场：先删张量与函数引用，gc 断开 autograd 图与 dynamo 缓存引用，再还显存
    del Q, K, V, fn
    gc.collect()
    torch.cuda.empty_cache()
    return r


# ----------------------------------------------------------------------------- 表格输出
def grid_table(rows, seqs, ds, cell, title):
    print(f"\n**{title}**")
    print("| d_model \\ seq | " + " | ".join(str(s) for s in seqs) + " |")
    print("|---" * (len(seqs) + 1) + "|")
    for d in ds:
        cells = [cell(rows.get((d, s))) for s in seqs]
        print(f"| {d} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--seq", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    p.add_argument("--batch", type=int, default=8)       # 作业固定单头、batch=8
    p.add_argument("--warmup", type=int, default=3)      # compile 首调用含编译（数十秒~分钟级），被此吸收
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--compile-cfgs", nargs="*", default=[],
                   help='额外对哪些 (d,S) 跑 compiled，格式 "d,S"，如 128,4096 64,8192 128,8192')
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  b={args.batch} 单头 fp32(TF32 off)")
    print("理论 forward 瞬态峰值 2·B·S²·4B：" + "  ".join(
        f"S={s}:{2 * args.batch * s * s * 4 / 2**30:.2f}GiB" for s in args.seq))
    print("理论 backward 前保存  1·B·S²·4B：" + "  ".join(
        f"S={s}:{args.batch * s * s * 4 / 2**30:.2f}GiB" for s in args.seq))

    rows = {}                                   # (d,S) -> {"van":...,"comp":...}
    # 先跑 vanilla 全网格（含 OOM 探测）
    for S in args.seq:
        for d in args.d_model:
            r = run_cfg(d, S, args, compiled=False)
            rows.setdefault((d, S), {})["van"] = r
            tag = (f"fwd={r['fwd']:8.2f} bwd={r['bwd']:8.2f} "
                   f"pre={r['mem_pre']:6.2f}G peakF={r['peak_fwd']:6.2f}G") if r else "OOM"
            print(f"  vanilla  d={d:4d} S={S:6d}: {tag}")
    # 再对指定配置跑 compiled
    for spec in args.compile_cfgs:
        d, S = (int(x) for x in spec.split(","))
        t0 = time.perf_counter()
        r = run_cfg(d, S, args, compiled=True)
        wall = time.perf_counter() - t0
        rows.setdefault((d, S), {})["comp"] = r
        tag = (f"fwd={r['fwd']:8.2f} bwd={r['bwd']:8.2f} "
               f"pre={r['mem_pre']:6.2f}G peakF={r['peak_fwd']:6.2f}G") if r else "OOM"
        # 首次调用含编译的耗时在此打出（已被 warmup 吸收，不进测量数）
        print(f"  compiled d={d:4d} S={S:6d}: {tag}   (含编译共 {wall:.0f}s)")

    # ---- 作业表 1：forward / backward 时间（vanilla）----
    grid_table(rows, args.seq, args.d_model,
               lambda r: f"{r['van']['fwd']:.2f} / {r['van']['bwd']:.2f}" if r and r.get("van") else "OOM",
               "forward / backward 时间 (ms)，vanilla attention（格内为 fwd / bwd）")

    # ---- 作业表 2：backward 前显存 ----
    grid_table(rows, args.seq, args.d_model,
               lambda r: f"{r['van']['mem_pre']:.2f}" if r and r.get("van") else "OOM",
               "backward 前显存 (GiB)，vanilla —— 对账基准 1·B·S²·4B")
    # 附加表：forward 瞬态峰值 —— 对账基准 2·B·S²·4B（OOM 首现点的位置）
    grid_table(rows, args.seq, args.d_model,
               lambda r: f"{r['van']['peak_fwd']:.2f}" if r and r.get("van") else "OOM",
               "[附加] forward 瞬态峰值显存 (GiB) —— 对账基准 2·B·S²·4B，OOM 咬人的位置")

    # ---- 作业表 3：compile 对比 ----
    have = [(d, s) for (d, s) in rows if rows[(d, s)].get("comp")]
    if have:
        print("\n**torch.compile 对比**")
        print("| d_model | seq | vanilla fwd | compiled fwd | vanilla bwd | compiled bwd | 加速比(fwd+bwd) |"
              " vanilla pre-G | compiled pre-G |")
        print("|---|---|---|---|---|---|---|---|---|")
        for d, s in have:
            v, c = rows[(d, s)]["van"], rows[(d, s)]["comp"]
            sp = (v["fwd"] + v["bwd"]) / (c["fwd"] + c["bwd"])
            # 最后两列是附加验证：compile 是否改变了 S² 显存占用
            # （预期：基本不变——fusion 省访存，不消除 materialize）
            print(f"| {d} | {s} | {v['fwd']:.2f} | {c['fwd']:.2f} | {v['bwd']:.2f} | {c['bwd']:.2f} "
                  f"| {sp:.2f}× | {v['mem_pre']:.2f} | {c['mem_pre']:.2f} |")


if __name__ == "__main__":
    main()


