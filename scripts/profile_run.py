#!/usr/bin/env python
"""12.5 profile 运行脚本：NVTX 标注 + capture-range，专供 nsys / ncu。

本脚本不测绝对时间（那是 benchmark.py 的职责）：每个 phase 末尾
torch.cuda.synchronize()，把该阶段发射的 GPU 工作全部收进对应 NVTX 时间窗，
nsys 按窗口归因（12.8 方案 A：backward 的 kernel 用时间窗选取）。

模型配置、dtype 语义（bfloat16 = fp32 权重 + autocast；tf32 = fp32 权重 +
matmul.allow_tf32）、optimizer 选择与 benchmark.py 保持一致，nsys/ncu 看到
的构成才能和 benchmark 的时间、第 11 章的预测对上。

nsys（12.6）：
  CUDA_VISIBLE_DEVICES=0 uv run nsys profile \
    -o results/nsys/small_ctx512_fp32 --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python scripts/profile_run.py --model small --ctx 512 --batch 4 \
        --mode full --warmup 5 --steps 3 --nvtx-fine

ncu（12.10）：同一脚本。--profile-from-start off 对应本文件的
profiler.start()/stop()；-k / -c / --nvtx-include 过滤，先 -c 1 试通。
ncu 建议 --warmup 2 --steps 1（每个 kernel 会被重放多次）。

注意：--dtype tf32 会让 GEMM 走 tensor-core 路径，kernel 序列与 float32
不同，ncu 的 -s 序号需重新数（先 --steps 1 不带过滤全采一遍核对）。
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

# 第 10 章的前提：TF32 默认关闭（--dtype tf32 时在 make_fns 里打开）
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

import cs336_basics.model as M
from cs336_basics.nn_utils import softmax

try:
    from cs336_basics.nn_utils import cross_entropy
except ImportError:
    cross_entropy = None

try:
    from cs336_basics.optimizer import AdamW as StaffAdamW
except ImportError:
    StaffAdamW = None

# 与 benchmark.py、第 11 章同一份配置（small: 12 层、12 头、d=768、d_ff=4d）
MODEL_CONFIGS = {
    "small":  dict(num_layers=12, num_heads=12, d_model=768,  d_ff=3072),
    "medium": dict(num_layers=24, num_heads=16, d_model=1024, d_ff=4096),
    "large":  dict(num_layers=36, num_heads=20, d_model=1280, d_ff=5120),
    "xl":     dict(num_layers=48, num_heads=25, d_model=1600, d_ff=6400),
    "tiny":   dict(num_layers=2,  num_heads=2,  d_model=64,   d_ff=128),   # 冒烟测试用
}
VOCAB_SIZE = {"tiny": 1000}          # 其余模型 10000
DEFAULT_VOCAB = 10000
ROPE_THETA = 10000.0
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "tf32": "tf32"}

# 自检计数：每个被标注的模块类、以及替换后的 sdpa 被调用了多少次
COUNTS: collections.Counter = collections.Counter()


def annotated_sdpa(Q, K, V, mask=None):
    """带 NVTX 三段标注的 sdpa。staff 的 CausalMultiHeadSelfAttention.forward
    按模块全局名调用 scaled_dot_product_attention（以 Q=/K=/V=/mask= 关键字传参），
    所以替换 M 模块属性即生效，不改 staff 源码。只在 profile 运行里替换。"""
    COUNTS["sdpa"] += 1
    d_k = K.shape[-1]
    with nvtx.range("attn.scores"):
        s = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
        if mask is not None:
            s = torch.where(mask, s, float("-inf"))
    with nvtx.range("attn.softmax"):
        w = softmax(s, dim=-1)
    with nvtx.range("attn.av"):
        return einsum(w, V, "... q k, ... k d -> ... q d")


def add_nvtx_hooks(model, names=("RMSNorm", "CausalMultiHeadSelfAttention", "SwiGLU",
                                 "TransformerBlock", "Embedding", "Linear")):
    """模块级 push/pop hook（只覆盖 forward）。区间名=类名。

    hook 必须返回 None：pre-hook 返回非 None 会被当作"替换后的输入"，
    forward hook 返回非 None 会被当作"替换后的输出"，而 range_push/range_pop
    本身返回 int。所以下面两个函数都不写 return。"""
    hs = []
    for mod in model.modules():
        k = type(mod).__name__
        if k in names:
            def pre(m, a, _k=k):
                COUNTS[_k] += 1
                nvtx.range_push(_k)
            def post(m, a, o):
                nvtx.range_pop()
            hs.append(mod.register_forward_pre_hook(pre))
            hs.append(mod.register_forward_hook(post))
    return hs


def expected_counts(L):
    """每个 forward 中各区间应出现的次数（用于自检）。"""
    return {"RMSNorm": 2 * L + 1, "Linear": 7 * L + 1,
            "CausalMultiHeadSelfAttention": L, "SwiGLU": L,
            "TransformerBlock": L, "Embedding": 1, "sdpa": L}


def phase(name, fn):
    """一个训练阶段：NVTX 区间 + 末尾同步。代价是破坏阶段间重叠（12.5），
    所以本脚本只看构成、不做性能结论。"""
    with nvtx.range(name):
        out = fn()
        torch.cuda.synchronize()
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="small")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dtype", choices=list(DTYPES), default="float32",
                   help="bfloat16 = fp32 权重 + autocast；tf32 = fp32 权重 + "
                        "matmul.allow_tf32（语义与 benchmark.py 一致）")
    p.add_argument("--mode", choices=["fwd", "fwdbwd", "full"], default="full")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nvtx-fine", action="store_true",
                   help="细粒度标注：sdpa 三段替换 + 模块 hook（有 CPU 开销）")
    p.add_argument("--emit-nvtx", action="store_true",
                   help="12.8 方案 B：emit_nvtx() 为反向算子自动打 NVTX（开销大）")
    p.add_argument("--optimizer", choices=["auto", "staff", "torch"], default="auto",
                   help="auto：有 staff AdamW 就用它，与 benchmark.py 一致")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build(args):
    """staff 构造签名没有 device/dtype：先建再 .cuda()（RoPE buffer 一并搬移）。
    权重始终 fp32；bf16 走 autocast、tf32 走 allow_tf32，与 benchmark.py 同一语义。"""
    torch.manual_seed(args.seed)
    cfg = dict(MODEL_CONFIGS[args.model])
    V = VOCAB_SIZE.get(args.model, DEFAULT_VOCAB)
    model = M.BasicsTransformerLM(
        vocab_size=V, context_length=args.ctx,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        rope_theta=ROPE_THETA,
    ).cuda().train()
    # batch 一次生成、常驻 GPU：生成/H2D 不进任何测量区间
    x = torch.randint(0, V, (args.batch, args.ctx), device="cuda")
    y = torch.randint(0, V, (args.batch, args.ctx), device="cuda")
    use_staff = args.optimizer == "staff" or (args.optimizer == "auto" and StaffAdamW is not None)
    if use_staff:
        assert StaffAdamW is not None, "找不到 cs336_basics.optimizer.AdamW"
        opt, impl = StaffAdamW(model.parameters(), lr=args.lr), "staff AdamW"
    else:
        opt, impl = torch.optim.AdamW(model.parameters(), lr=args.lr), "torch.optim.AdamW"
    return model, x, y, opt, impl, cfg, V


def make_fns(model, x, y, opt, V, dtype):
    """与 benchmark.py 对齐。"""
    last = {}

    if dtype == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp = lambda: contextlib.nullcontext()
    elif dtype == torch.bfloat16:
        torch.backends.cuda.matmul.allow_tf32 = False   # 防串味
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:  # float32
        torch.backends.cuda.matmul.allow_tf32 = False
        amp = lambda: contextlib.nullcontext()

    def loss_of():
        with amp():
            logits = model(x)                                # (B, T, V)
        flat_logits = logits.view(-1, V).float()             # loss 在 fp32 里算
        if cross_entropy is not None:
            loss = cross_entropy(flat_logits, y.view(-1))
        else:
            loss = torch.nn.functional.cross_entropy(flat_logits, y.view(-1))
        last["loss"] = loss.detach()
        return loss

    def forward_only():
        with torch.no_grad():              # fwd：与 11.5 forward 预测同前提
            with amp():
                model(x)

    def fwd():
        with nvtx.range("step"):
            phase("forward", forward_only)

    def fwdbwd():
        with nvtx.range("step"):
            # set_to_none 不发射 kernel；保证每步反向从"无 grad"开始，
            # kernel 序列逐 step 一致（否则第 2 步起多出梯度累加的 add kernel）
            opt.zero_grad(set_to_none=True)
            loss = phase("forward", loss_of)
            phase("backward", loss.backward)

    def full():
        with nvtx.range("step"):
            opt.zero_grad(set_to_none=True)
            loss = phase("forward", loss_of)
            phase("backward", loss.backward)
            phase("optimizer", opt.step)

    return {"fwd": fwd, "fwdbwd": fwdbwd, "full": full}, last


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "profile_run 必须在 GPU 上跑"
    model, x, y, opt, opt_impl, cfg, V = build(args)
    fns, last = make_fns(model, x, y, opt, V, DTYPES[args.dtype])
    step_once = fns[args.mode]           # 本进程只跑一档：capture range 只开一次

    handles = []
    if args.nvtx_fine:
        assert hasattr(M, "scaled_dot_product_attention"), \
            "cs336_basics.model 里没有 scaled_dot_product_attention，替换不会生效"
        M.scaled_dot_product_attention = annotated_sdpa   # 一次打好，nsys/ncu 共用
        handles = add_nvtx_hooks(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"model={args.model} params={n_params / 1e6:.2f}M optimizer={opt_impl} "
          f"mode={args.mode} ctx={args.ctx} batch={args.batch} dtype={args.dtype}")
    print(f"warmup={args.warmup} steps={args.steps} nvtx_fine={args.nvtx_fine} "
          f"emit_nvtx={args.emit_nvtx} "
          f"(matmul_tf32={torch.backends.cuda.matmul.allow_tf32})")

    for _ in range(args.warmup):           # 预热：cuBLAS/懒加载/优化器状态，不在 capture range 内
        step_once()
    torch.cuda.synchronize()
    COUNTS.clear()                         # 只统计测量步

    torch.cuda.profiler.start()            # nsys --capture-range=cudaProfilerApi 从此开始抓
    meas_ctx = (torch.autograd.profiler.emit_nvtx()
                if args.emit_nvtx else contextlib.nullcontext())
    with meas_ctx:
        for _ in range(args.steps):        # nsys 3 步足够，多了文件很大
            step_once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    if args.nvtx_fine:                     # 自检：标注次数是否符合结构
        exp = expected_counts(cfg["num_layers"])
        bad = False
        print("[自检] 每步各区间次数（实测 / 预期）：")
        for k, e in exp.items():
            got = COUNTS[k] / args.steps
            flag = "" if got == e else "   <-- 不符"
            bad |= got != e
            print(f"  {k:30s} {got:8.1f} / {e}{flag}")
        if bad:
            print("[自检] 有不符：检查 staff 里的类名、sdpa 是否被替换，"
                  "再解读 nsys 的 NVTX 汇总")
        for h in handles:
            h.remove()

    print(f"done: {args.steps} steps profiled (capture range closed)")


if __name__ == "__main__":
    main()
