"""12.2 真实耗时基准：不加任何 profiler，测各阶段稳态耗时。

模式（对应 12.14 回填 (1)）：
  fwd     no_grad 下的 forward            -> 对比 11.5 的 forward 预测
  fwdbwd  forward + loss + backward
  full    完整训练步（zero_grad + fwd + loss + bwd + optimizer.step）
  all     三档依次各测一遍

输出列：
  median/min/max  每步 GPU 耗时(ms)，每步之间同步（无跨步重叠）
  jitter          (max-min)/median，大说明卡不空闲或有其他进程
  wall            连续跑 iters 步只同步一次的墙钟 ms/步（含流水）
  tok/s           按 wall 算的吞吐
  TFLOP/s         按 median 和 11.3 的 FLOPs 公式算的达到算力（fwdbwd/full 取 3x forward）
  peak GiB        该模式下 max_memory_allocated（对照 11.6 的 est_peak_gib）

用法示例：
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model small --ctx 512 --batch 4
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --dtype bfloat16 --mode fwd
"""
import argparse, contextlib, json, statistics, time
from pathlib import Path
import torch

# 第 10 章的前提：TF32 关闭，否则 GEMM 走不同路径，与第 11 章预测对不上
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

# 与第 11 章预测同一份配置（small: 12 层、12 头、d=768、d_ff=4d）
MODEL_CONFIGS = {
    "small":  dict(num_layers=12, num_heads=12, d_model=768,  d_ff=3072),
    "medium": dict(num_layers=24, num_heads=16, d_model=1024, d_ff=4096),
    "large":  dict(num_layers=36, num_heads=20, d_model=1280, d_ff=5120),
    "xl":     dict(num_layers=48, num_heads=25, d_model=1600, d_ff=6400),
}
VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "tf32":'tf32'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="small")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dtype", choices=list(DTYPES), default="float32",
                   help="bfloat16 = fp32 权重 + autocast")
    p.add_argument("--mode", choices=["fwd", "fwdbwd", "full", "all"], default="all")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--compile", action="store_true", help="torch.compile(model)，预热会包含编译")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/bench")
    return p.parse_args()


def flops_fwd(cfg, V, B, T):
    """11.3 的 forward FLOPs（不含逐元素运算）。"""
    d, L, d_ff, N = cfg["d_model"], cfg["num_layers"], cfg["d_ff"], B * T
    return (L * 8 * N * d * d            # q,k,v,o 投影
            + L * 6 * N * d * d_ff       # SwiGLU 三个 GEMM
            + L * 4 * N * T * d          # QK^T 与 PV
            + 2 * N * d * V)             # lm_head


def bench(fn, warmup, iters):
    """每次迭代的 GPU 耗时(ms)列表。预热 -> 同步 -> event 逐轮计时（每轮之间同步）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def wall_ms_per_step(fn, warmup, iters):
    """连续 iters 步只同步一次：墙钟 ms/步（含步与步之间的流水）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def build(args):
    import cs336_basics.model as M
    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW

    torch.manual_seed(args.seed)
    cfg = MODEL_CONFIGS[args.model]
    model = M.BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=args.ctx,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        rope_theta=ROPE_THETA,
    ).cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    if args.compile:
        model = torch.compile(model)
    # 计时只关心形状：batch 一次生成、常驻 GPU，不把生成与 H2D 计进去
    x = torch.randint(0, VOCAB_SIZE, (args.batch, args.ctx), device="cuda")
    y = torch.randint(0, VOCAB_SIZE, (args.batch, args.ctx), device="cuda")
    opt = AdamW(model.parameters(), lr=args.lr)
    return cfg, n_params, model, x, y, opt, cross_entropy


def make_fns(model, x, y, opt, cross_entropy, dtype):
    last = {}                                     # 记录最后一次 loss（张量，不在计时里 .item()）

    # def amp():
    #     if dtype == torch.bfloat16:
    #         return torch.autocast("cuda", dtype=torch.bfloat16)
    #     return contextlib.nullcontext()
    if dtype == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        amp = lambda: contextlib.nullcontext()
    elif dtype == torch.bfloat16:
        torch.backends.cuda.matmul.allow_tf32 = False   # 防串味
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:  # fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        amp = lambda: contextlib.nullcontext()

    def loss_of():
        with amp():
            logits = model(x)
        loss = cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        last["loss"] = loss.detach()
        return loss

    def fwd():                                    # no_grad：才能与 11.5 forward 预测对比
        with torch.no_grad():
            with amp():
                model(x)

    def fwdbwd():
        opt.zero_grad(set_to_none=True)
        loss_of().backward()

    def full():                                   # 一个真实训练步
        opt.zero_grad(set_to_none=True)
        loss_of().backward()
        opt.step()

    return {"fwd": fwd, "fwdbwd": fwdbwd, "full": full}, last


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "benchmark 必须在 GPU 上跑"
    cfg, n_params, model, x, y, opt, ce = build(args)
    fns, last = make_fns(model, x, y, opt, ce, DTYPES[args.dtype])
    modes = ["fwd", "fwdbwd", "full"] if args.mode == "all" else [args.mode]

    F = flops_fwd(cfg, VOCAB_SIZE, args.batch, args.ctx)
    flops_of = {"fwd": F, "fwdbwd": 3 * F, "full": 3 * F}

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"model={args.model} params={n_params/1e6:.1f}M ctx={args.ctx} batch={args.batch} "
          f"dtype={args.dtype} compile={args.compile}")
    print(f"forward FLOPs = {F/1e9:.1f} GFLOP  (step = 3x = {3*F/1e12:.2f} TFLOP)")
    print(f"{'mode':8s} {'median':>9s} {'min':>8s} {'max':>8s} {'jitter':>7s} "
          f"{'wall':>8s} {'tok/s':>9s} {'TFLOP/s':>8s} {'peak GiB':>9s}   (ms)")

    rows = {}
    for mode in modes:
        torch.cuda.reset_peak_memory_stats()
        ts = bench(fns[mode], args.warmup, args.iters)
        med, lo, hi = statistics.median(ts), min(ts), max(ts)
        wall = wall_ms_per_step(fns[mode], args.warmup, args.iters)
        tps = args.batch * args.ctx / (wall / 1e3)
        tflops = flops_of[mode] / (med / 1e3) / 1e12
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{mode:8s} {med:9.2f} {lo:8.2f} {hi:8.2f} {(hi - lo) / med:6.0%} "
              f"{wall:8.2f} {tps:9.0f} {tflops:8.2f} {peak:9.2f}")
        rows[mode] = dict(median_ms=med, min_ms=lo, max_ms=hi, jitter=(hi - lo) / med,
                          wall_ms_per_step=wall, tokens_per_s=tps,
                          achieved_tflops=tflops, peak_mem_gib=peak,
                          raw_ms=[round(t, 3) for t in ts])
        if mode != "fwd":
            ok = bool(torch.isfinite(last["loss"]).item())
            rows[mode]["final_loss"] = float(last["loss"].item())
            if not ok:
                print(f"  WARNING: {mode} 的 loss 非有限值，请降低 --lr 后重测")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_ctx{args.ctx}_b{args.batch}_{args.dtype}" + ("_compile" if args.compile else "")
    p = Path(args.out) / f"bench_{name}.json"
    p.write_text(json.dumps(dict(config=vars(args), model_cfg=cfg, n_params=n_params,
                                 fwd_flops=F, device=torch.cuda.get_device_name(0),
                                 torch=torch.__version__, results=rows),
                            indent=2, ensure_ascii=False))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()