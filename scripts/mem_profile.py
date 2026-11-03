"""3.8 显存 profiling：整个训练步的显存时间线、构成、峰值位置，并与理论对照。

每个配置在【独立子进程】里测（cuBLAS workspace、allocator、Adam 状态互不污染），流程：
  理论  先算 4P / 16P / 激活估算（11.6 的公式），预测是否会 OOM
  A    刚初始化、还没有优化器：no_grad 前向峰值                       -> 表里 "forward 峰值"
  B    带图的前向：保存下来的激活、带图前向峰值                         -> 激活与 ctx 的关系
  C    创建 AdamW、预热 3 步让 Adam 状态存在，然后只对【一步】记录显存快照，并按阶段分别读峰值
       forward / backward / optimizer 各自的峰值与结束时占用            -> 峰值出现在哪个阶段
       结束 forward 与结束 backward 时的构成：权重 / 梯度 / Adam / 其他(激活)
  D    解析快照事件流：重建峰值时刻的存活块，按调用栈归类；列出最大的几次分配
然后汇总成 3.8 的表，并给出附加问题所需的数（激活拟合、2.7B 理论、混合精度对比）。

用法：
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/mem_profile.py --model medium --ctx 128 512 --dtype float32 bfloat16
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/mem_profile.py --model large  --ctx 256     --dtype float32 bfloat16
  # 先用小配置试通：
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/mem_profile.py --model small --ctx 128 256 --dtype float32

快照 pickle 在 results/mem/，拖到 https://pytorch.org/memory_viz 查看（浏览器本地解析）。
注意：开着记录的那一步有额外开销，这个脚本不用于计时。
"""
import argparse
import contextlib
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

G = 2 ** 30
VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
MODEL_CONFIGS = {
    "small":  dict(num_layers=12, num_heads=12, d_model=768,  d_ff=3072),
    "medium": dict(num_layers=24, num_heads=16, d_model=1024, d_ff=4096),
    "large":  dict(num_layers=36, num_heads=20, d_model=1280, d_ff=5120),
    "xl":     dict(num_layers=48, num_heads=25, d_model=1600, d_ff=6400),
    "2.7b":   dict(num_layers=32, num_heads=32, d_model=2560, d_ff=10240),
}
DTYPES = ("float32", "tf32", "bfloat16")
LABEL = {"float32": "fp32", "tf32": "tf32", "bfloat16": "bf16 autocast"}


# ============================================================================= 理论（纯 Python）
def n_params(cfg, V=VOCAB_SIZE):
    """每层 4d²(q,k,v,o) + 3·d·d_ff(SwiGLU) + 2d(两个 RMSNorm)；加 embedding、lm_head、ln_final。"""
    d, L, dff = cfg["d_model"], cfg["num_layers"], cfg["d_ff"]
    return L * (4 * d * d + 3 * d * dff + 2 * d) + 2 * V * d + d


def theory(cfg, B, T, c=2.0, other=20.0, V=VOCAB_SIZE):
    """11.6 的估算。激活 = L·4·(5·N·d_ff + c·B·h·T² + other·N·d) + 16·N·V 字节（fp32 口径）。"""
    P = n_params(cfg, V)
    d, L, h, dff = cfg["d_model"], cfg["num_layers"], cfg["num_heads"], cfg["d_ff"]
    N = B * T
    act = L * 4 * (5 * N * dff + c * B * h * T * T + other * N * d) + 16 * N * V
    return dict(P=P, static_fwd=4 * P, static_train=16 * P, act_est=act, peak_est=16 * P + act)


def fit_activation(rs, cfg, V=VOCAB_SIZE):
    """用 ≥2 个 ctx 的 saved_act 拟合 act = a·N + b·B·h·T²，换算成 11.6 的等效份数。"""
    import numpy as np
    pts = [(r["ctx"], r["probes"]["saved_act_gib"] * G) for r in rs
           if "saved_act_gib" in r.get("probes", {})]
    if len({c for c, _ in pts}) < 2:
        return None
    B, h = rs[0]["batch"], cfg["num_heads"]
    A = np.array([[B * c, B * h * c * c] for c, _ in pts], float)
    y = np.array([v for _, v in pts], float)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    L, d = cfg["num_layers"], cfg["d_model"]
    return dict(a=float(a), b=float(b),
                c_eff=float(b / (4 * L)),                         # 每层等效 B·h·T² 张量份数（11.6 的 c）
                lin_eff=float((a - 16 * V) / (4 * L * d)))        # 每层线性项等效 N×d 张量份数（11.6 约 40）


# ============================================================================= 快照事件流解析（纯 Python）
def short_stack(frames, k=3):
    keep = [f for f in frames if "/torch/" not in f.get("filename", "")]
    use = keep or frames
    return " <- ".join(f"{Path(f.get('filename', '?')).name}:{f.get('line', '?')} {f.get('name', '')}"
                       for f in use[:k])


def classify(frames):
    """启发式：按调用栈里的文件/函数名归类。反向分配发生在 autograd 线程，常常没有 Python 帧。"""
    s = " ".join(f"{f.get('filename', '')} {f.get('name', '')}" for f in frames).lower()
    if "optimizer" in s:
        return "optimizer 临时"
    if "autograd" in s or "engine" in s:
        return "backward(梯度/临时)"
    if "cross_entropy" in s or "nn_utils" in s:
        return "loss"
    if "model.py" in s or "cs336_basics" in s:
        return "forward(激活)"
    return "other"


def analyze_trace(snap, baseline_bytes, topn=8):
    """重建峰值时刻的存活块。baseline_bytes = 开始记录前已占用的字节（权重 + Adam 状态）。"""
    traces = snap.get("device_traces") or []
    tr = traces[0] if traces else []
    free_act = "free_completed" if any(e.get("action") == "free_completed" for e in tr) else "free_requested"

    def replay(upto):
        live, cur, peak, peak_i = {}, baseline_bytes, baseline_bytes, -1
        for i, e in enumerate(tr[: upto if upto is not None else len(tr)]):
            a = e.get("action")
            if a == "alloc":
                live[e["addr"]] = (e["size"], e.get("frames", []))
                cur += e["size"]
                if cur > peak:
                    peak, peak_i = cur, i
            elif a == free_act:
                v = live.pop(e.get("addr"), None)
                cur -= v[0] if v is not None else e.get("size", 0)
        return live, peak, peak_i

    _, peak, peak_i = replay(None)
    live, _, _ = replay(peak_i + 1) if peak_i >= 0 else ({}, 0, -1)
    tags = defaultdict(int)
    tags["静态(权重+Adam)"] += baseline_bytes
    for sz, fr in live.values():
        tags[classify(fr)] += sz
    top_live = sorted(live.values(), key=lambda v: -v[0])[:topn]
    allocs = sorted(((e["size"], e.get("frames", [])) for e in tr if e.get("action") == "alloc"),
                    key=lambda v: -v[0])[:5]
    return dict(n_events=len(tr), peak_gib=peak / G, peak_event=peak_i,
                tags_gib={k: v / G for k, v in sorted(tags.items(), key=lambda kv: -kv[1])},
                top_live=[dict(mib=sz / 2**20, stack=short_stack(fr)) for sz, fr in top_live],
                largest=[dict(mib=sz / 2**20, stack=short_stack(fr)) for sz, fr in allocs])


# ============================================================================= 子进程：测一个配置
def setup_precision(dtype):
    tf32 = dtype == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def autocast_ctx(dtype):
    if dtype == "bfloat16":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return lambda: contextlib.nullcontext()


def build(model_name, ctx, batch, seed):
    import cs336_basics.model as M
    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW
    torch.manual_seed(seed)
    cfg = MODEL_CONFIGS[model_name]
    model = M.BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=ctx, d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=ROPE_THETA).cuda().train()
    x = torch.randint(0, VOCAB_SIZE, (batch, ctx), device="cuda")
    y = torch.randint(0, VOCAB_SIZE, (batch, ctx), device="cuda")
    return cfg, model, x, y, cross_entropy, AdamW


def composition(model, opt):
    """此刻的显存构成：权重 / 梯度 / Adam 状态 / 其他（激活、临时、workspace 等），单位 GiB。"""
    w = sum(p.numel() * p.element_size() for p in model.parameters())
    g = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    a, seen = 0, set()
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v) and v.is_cuda and v.data_ptr() not in seen:
                seen.add(v.data_ptr())
                a += v.numel() * v.element_size()
    total = torch.cuda.memory_allocated()
    return dict(total=total / G, weights=w / G, grads=g / G, adam=a / G, other=(total - w - g - a) / G)


def phase_memory(loss_fn, model, opt, out):
    """一个训练步，按阶段分别 reset 并读峰值，直接写进 out（OOM 时已测到的部分不丢）。"""
    def read():
        torch.cuda.synchronize()
        return dict(end=torch.cuda.memory_allocated() / G, peak=torch.cuda.max_memory_allocated() / G,
                    reserved_peak=torch.cuda.max_memory_reserved() / G)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    out["base_gib"] = torch.cuda.memory_allocated() / G            # 权重 + Adam 状态，梯度已清
    torch.cuda.reset_peak_memory_stats()
    loss = loss_fn()
    out["fwd"] = read()
    out["comp_after_fwd"] = composition(model, opt)
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    out["bwd"] = read()
    out["comp_after_bwd"] = composition(model, opt)
    torch.cuda.reset_peak_memory_stats()
    opt.step()
    out["opt"] = read()
    out["train_peak_gib"] = max(out[k]["peak"] for k in ("fwd", "bwd", "opt"))
    out["reserved_peak_gib"] = max(out[k]["reserved_peak"] for k in ("fwd", "bwd", "opt"))
    del loss


def run_single(a):
    dtype, model_name, ctx = a.dtype[0], a.model[0], a.ctx[0]
    setup_precision(dtype)
    cfg = MODEL_CONFIGS[model_name]
    th = theory(cfg, a.batch, ctx)
    total_mem = torch.cuda.mem_get_info()[1]
    res = dict(model=model_name, ctx=ctx, batch=a.batch, dtype=dtype, status="OK", stage="build",
               matmul_precision=torch.get_float32_matmul_precision(),
               theory={k: (v / G if k != "P" else v) for k, v in th.items()}, probes={}, train={})
    fits = "放得下" if th["peak_est"] < 0.95 * total_mem else "预计放不下（OOM 是预期结果）"
    print(f"\n=== {model_name} ctx={ctx} b={a.batch} {dtype} ===")
    print(f"理论：P={th['P'] / 1e6:.1f}M  4P={th['static_fwd'] / G:.2f}  16P={th['static_train'] / G:.2f}  "
          f"激活估算={th['act_est'] / G:.2f}  峰值估算={th['peak_est'] / G:.2f} GiB  "
          f"(显卡 {total_mem / G:.1f} GiB，{fits})")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_ctx{ctx}_{dtype}"
    try:
        _, model, x, y, ce, AdamW = build(model_name, ctx, a.batch, 0)
        res["P"] = sum(p.numel() for p in model.parameters())
        amp = autocast_ctx(dtype)

        def loss_fn():
            with amp():                                           # autocast 只包 forward
                logits = model(x)
            return ce(logits.float().view(-1, logits.shape[-1]), y.view(-1))

        # A. 无优化器时的 no_grad 前向峰值
        res["stage"] = "A(no_grad 前向)"
        for _ in range(2):
            with torch.no_grad():
                loss_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        w = torch.cuda.memory_allocated()
        with torch.no_grad():
            loss_fn()
        torch.cuda.synchronize()
        res["probes"]["weights_gib"] = w / G
        res["probes"]["fwd_nograd_peak_gib"] = torch.cuda.max_memory_allocated() / G

        # B. 带图前向：保存下来的激活
        res["stage"] = "B(带图前向)"
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = loss_fn()
        torch.cuda.synchronize()
        res["probes"]["saved_act_gib"] = (torch.cuda.memory_allocated() - base) / G
        res["probes"]["fwd_grad_peak_gib"] = torch.cuda.max_memory_allocated() / G
        loss.backward()
        model.zero_grad(set_to_none=True)
        del loss

        # C. 完整训练步：先预热出 Adam 状态，再只对一步记录
        res["stage"] = "C(预热)"
        opt = AdamW(model.parameters(), lr=1e-4)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        res["probes"]["static_with_adam_gib"] = baseline / G

        res["stage"] = "C(记录的训练步)"
        torch.cuda.memory._record_memory_history(max_entries=a.max_entries)
        try:
            phase_memory(loss_fn, model, opt, res["train"])
        finally:                                                  # 即使 OOM 也导出快照（OOM 时的快照最有用）
            snap = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            path = out_dir / f"mem_{tag}.pickle"
            with open(path, "wb") as f:
                pickle.dump(snap, f)
            res["snapshot"] = str(path)
            try:
                res["trace"] = analyze_trace(snap, baseline)
                if res["trace"]["n_events"] >= a.max_entries:
                    print("警告：事件数达到 max_entries，缓冲区可能溢出，请调大 --max-entries")
            except Exception as e:                                # 快照字段随版本变化，解析失败不影响其余结果
                res["trace_error"] = repr(e)
        res["stage"] = "done"
    except torch.cuda.OutOfMemoryError:
        res["status"] = f"OOM({res['stage']})"
        print(f"OOM，发生在阶段：{res['stage']}")
    finally:
        (out_dir / f"res_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    pr = res["probes"]
    print(f"A no_grad 前向峰值 {pr.get('fwd_nograd_peak_gib', float('nan')):.2f} GiB | "
          f"B 保存的激活 {pr.get('saved_act_gib', float('nan')):.2f} GiB | "
          f"C 完整步峰值 {res['train'].get('train_peak_gib', float('nan')):.2f} GiB | status={res['status']}")


# ============================================================================= 父进程：调度与汇总
def fg(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "N/A"


def peak_phase(tr):
    ph = {k: tr[k]["peak"] for k in ("fwd", "bwd", "opt") if k in tr}
    if not ph:
        return "N/A"
    k = max(ph, key=ph.get)
    return {"fwd": "forward", "bwd": "backward", "opt": "optimizer"}[k]


def report(results, a):
    print("\n\n" + "=" * 30 + " 汇总 " + "=" * 30)
    print("\n### 3.8 显存 profiling（单位 GiB）")
    print("| 模型 | ctx | dtype | forward 峰值(no_grad) | 带图 forward 峰值 | 完整训练步峰值 | 峰值阶段 "
          "| 理论 4P | 理论 16P | 峰值估算(11.6) | 估算÷实测 | 状态 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        pr, tr, th = r["probes"], r["train"], r["theory"]
        real = tr.get("train_peak_gib")
        ratio = f"{th['peak_est'] / real:.2f}" if real else "N/A"
        print(f"| {r['model']} | {r['ctx']} | {LABEL[r['dtype']]} | {fg(pr.get('fwd_nograd_peak_gib'))} "
              f"| {fg(pr.get('fwd_grad_peak_gib'))} | {fg(real)} | {peak_phase(tr)} "
              f"| {fg(th['static_fwd'])} | {fg(th['static_train'])} | {fg(th['peak_est'])} | {ratio} | {r['status']} |")

    print("\n### 显存构成（GiB）：forward 结束时 与 backward 结束时")
    print("| 模型 | ctx | dtype | fwd结束: 权重 | Adam | 激活/其他 | 合计 | bwd结束: 权重 | 梯度 | Adam | 其他 | 合计 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        f, b = r["train"].get("comp_after_fwd"), r["train"].get("comp_after_bwd")
        if not f or not b:
            continue
        print(f"| {r['model']} | {r['ctx']} | {LABEL[r['dtype']]} | {fg(f['weights'])} | {fg(f['adam'])} | "
              f"{fg(f['other'])} | {fg(f['total'])} | {fg(b['weights'])} | {fg(b['grads'])} | {fg(b['adam'])} | "
              f"{fg(b['other'])} | {fg(b['total'])} |")
    print("说明：forward 结束时的『激活/其他』≈ 保存给反向用的激活（autocast 下含缓存的 bf16 权重副本）。")

    print("\n### 快照解析：峰值时刻的存活块归类（启发式，GiB）与最大单次分配")
    for r in results:
        t = r.get("trace")
        if not t:
            print(f"- {r['model']} ctx={r['ctx']} {LABEL[r['dtype']]}: 无快照解析"
                  + (f"（{r['trace_error']}）" if "trace_error" in r else ""))
            continue
        peak_np = r["train"].get("train_peak_gib")
        print(f"\n- {r['model']} ctx={r['ctx']} {LABEL[r['dtype']]}  事件数 {t['n_events']}  "
              f"事件流峰值 {t['peak_gib']:.2f} GiB（allocator 口径 {fg(peak_np)}，二者应接近）")
        print("  归类：" + "；".join(f"{k} {v:.2f}" for k, v in t["tags_gib"].items()))
        for x in t["largest"][:3]:
            print(f"  最大分配 {x['mib']:8.1f} MiB  {x['stack']}")

    print("\n### 附加问题 1：激活显存与 ctx 的关系（带图前向后保存的激活，GiB）")
    groups = defaultdict(list)
    for r in results:
        groups[(r["model"], r["dtype"])].append(r)
    for (m, dt), rs in groups.items():
        rs.sort(key=lambda r: r["ctx"])
        line = "  ".join(f"ctx{r['ctx']}={fg(r['probes'].get('saved_act_gib'))}" for r in rs)
        print(f"- {m} {LABEL[dt]}: {line}")
        if len(rs) >= 2 and all("saved_act_gib" in r["probes"] for r in rs[:2]):
            r0, r1 = rs[0], rs[-1]
            print(f"  ctx 增大 {r1['ctx'] / r0['ctx']:.0f}×：激活增大 "
                  f"{r1['probes']['saved_act_gib'] / r0['probes']['saved_act_gib']:.1f}×"
                  f"（纯线性项应为 {r1['ctx'] / r0['ctx']:.0f}×，纯 T² 项为 {(r1['ctx'] / r0['ctx']) ** 2:.0f}×）")
            fit = fit_activation(rs, MODEL_CONFIGS[m])
            if fit:
                print(f"  拟合 act = a·N + b·B·h·T²：每层线性项等效 {fit['lin_eff']:.1f} 份 N×d 张量"
                      f"（11.6 取约 40），T² 项等效 c = {fit['c_eff']:.2f}（11.6 取 2）")

    print("\n### 附加问题 2：2.7B 理论峰值（用本配置的参数个数）")
    cfg = MODEL_CONFIGS["2.7b"]
    c, o = 2.0, 20.0
    for (m, dt), rs in groups.items():
        fit = fit_activation(rs, MODEL_CONFIGS[m]) if len(rs) >= 2 else None
        if fit and dt == "float32":
            cm = MODEL_CONFIGS[m]                                     # lin_eff = 5·d_ff/d + other
            c = max(fit["c_eff"], 0.0)
            o = max(fit["lin_eff"] - 5 * cm["d_ff"] / cm["d_model"], 0.0)
    t = theory(cfg, a.theory_batch, a.theory_ctx, c=c, other=o)
    print(f"- P = {t['P'] / 1e9:.2f}B（SwiGLU 三矩阵、词表 10000，不是名字里的 2.7B）")
    print(f"- batch={a.theory_batch}, ctx={a.theory_ctx}：静态 16P = {t['static_train'] / G:.1f} GiB；"
          f"激活估算（c={c:.2f}, other={o:.1f}）= {t['act_est'] / G:.1f} GiB；"
          f"合计 ≈ {t['peak_est'] / G:.1f} GiB（{t['peak_est'] / 1e9:.1f} GB）")
    print(f"- 仅推理：fp32 权重 4P = {t['static_fwd'] / G:.1f} GiB；纯 bf16 权重 2P = {t['static_fwd'] / 2 / G:.1f} GiB")

    print("\n### 附加问题 4：混合精度对峰值显存的影响（bf16 autocast ÷ fp32）")
    by = {(r["model"], r["ctx"], r["dtype"]): r for r in results}
    for (m, c_, dt), r in by.items():
        if dt != "bfloat16" or (m, c_, "float32") not in by:
            continue
        f32 = by[(m, c_, "float32")]
        def ratio(k, src):
            x, y_ = r[src].get(k), f32[src].get(k)
            return f"{x / y_:.2f}×" if x and y_ else "N/A"
        print(f"- {m} ctx={c_}: 保存的激活 {ratio('saved_act_gib', 'probes')}，"
              f"no_grad 前向峰值 {ratio('fwd_nograd_peak_gib', 'probes')}，完整步峰值 {ratio('train_peak_gib', 'train')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", choices=list(MODEL_CONFIGS), default=["medium"])
    p.add_argument("--ctx", nargs="+", type=int, default=[128, 512])
    p.add_argument("--dtype", nargs="+", choices=DTYPES, default=["float32"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-entries", type=int, default=200_000, help="快照事件环形缓冲大小，太小会丢早期事件")
    p.add_argument("--out", default="results/mem")
    p.add_argument("--theory-ctx", type=int, default=128)
    p.add_argument("--theory-batch", type=int, default=4)
    p.add_argument("--single", action="store_true", help="内部使用：在当前进程里测一个配置")
    a = p.parse_args()

    if a.single:
        run_single(a)
        return

    results = []
    for m in a.model:
        for c in a.ctx:
            for dt in a.dtype:
                cmd = [sys.executable, os.path.abspath(__file__), "--single", "--model", m, "--ctx", str(c),
                       "--dtype", dt, "--batch", str(a.batch), "--max-entries", str(a.max_entries), "--out", a.out]
                rc = subprocess.call(cmd)
                f = Path(a.out) / f"res_{m}_ctx{c}_{dt}.json"
                if f.exists():
                    r = json.loads(f.read_text())
                    if rc != 0 and r["status"] == "OK":
                        r["status"] = f"FAIL(rc={rc})"
                    results.append(r)
                else:
                    print(f"子进程失败且无结果：{m} ctx={c} {dt} (rc={rc})")
    report(results, a)
    (Path(a.out) / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {Path(a.out) / 'summary.json'}")


if __name__ == "__main__":
    main()