"""12.2 真实耗时基准 v2：不加任何 profiler，测各阶段稳态耗时。

与 v1 的区别：
  1. 阶段时间不再靠 fwdbwd−fwd、full−fwdbwd 相减，而是在【同一个训练步里】用 4 个 CUDA event
     分别量出 forward(+loss) / backward / optimizer，每步只同步一次，因此能给出每个阶段自己的 均值±std。
       # why 不能相减：fwd 模式是 no_grad 前向，训练步里的前向是带图的（存激活+算 loss），
       #   fwdbwd−fwd 混入了"带图前向的额外开销"，口径是脏的；
       #   事件切分在同一步内打时间戳，三段之和严格等于整步（分解性质自带 self-check）。
       # why 每步只同步一次：sync 会打断 CPU 发射与 GPU 执行的流水；只在整个训练步末尾
       #   同步，段内三阶段保持异步发射，各段时长才是真实占比。
  2. 一次命令可扫多个模型 × ctx × dtype，自动记录 OOM（状态列），并直接打印作业的 3.2、3.5 表。
       # why 逐配置 try/except：一个配置 OOM 不中断整个扫描（跑批过夜的关键）。
  3. dtype 支持 float32 / tf32 / bfloat16(autocast) / bf16_weights(权重直接转 bf16，用于 2.7B 只测 fwd)。
       # bf16_weights 与 autocast 是两种不同的混合精度：前者真的省参数/优化器状态的显存，
       #   后者只省激活和 GEMM 时间、权重主本仍是 fp32。3.5 表测的是后者。
  4. --check-dtypes：用 hook 实测 autocast 下各类张量的 dtype（3.5 附加表）。
       # why 实测而不是背规则：autocast 的白名单/fp32名单/透传三分法容易记错，
       #   hook 一次 forward+backward 把"招供"全部抓下来。
  5. 峰值显存分三档记录：fwd(no_grad)、反向结束时、完整训练步（含 Adam 状态）。
       # 三档分别对应 11.6 显存模型的三个组成部分，可与预测逐项对账。

模式：
  fwd    no_grad 前向（不含 loss）          -> 对比 11.5 的 forward 预测、推理场景
  train  训练步分阶段计时 + 端到端 + 墙钟 + 显存探针
  all    先 fwd 再 train

用法示例（作业 3.2 / 3.5）：
  # 3.2：small/medium/large × 4 个 ctx（OOM 会记在状态列里，不会中断）
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py \
      --model small medium large --ctx 128 256 512 1024 --warmup 5 --iters 10
  # large 完整训练步放不下时，只测 fwd+bwd（不创建优化器）
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model large --ctx 128 256 512 1024 --no-opt
  # xl 仅 fwd；2.7B 仅 fwd + bf16 权重
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model xl --ctx 128 --mode fwd
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model 2.7b --ctx 128 --mode fwd --dtype bf16_weights
  # 3.5：fp32 与 bf16 autocast 对比（加 tf32 可拆出 Tensor Core 与类型转换各自的贡献）
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model small medium --ctx 512 --dtype float32 bfloat16
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model large --ctx 256 --dtype float32 bfloat16 --no-opt
  # 3.5 附加表：autocast 下各类张量的 dtype
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --check-dtypes --model small --ctx 128 --dtype bfloat16
"""
import argparse
import contextlib
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

# 与第 11 章预测同一份配置（d_ff = 4d；2.7b 为作业给定的结构）
# 2.7b 验算：SwiGLU 每层 16d² → 32·16·2560² ≈ 3.36B + embed/lm_head 2·10000·2560 ≈ 0.05B ≈ 3.4B
#（若作业表指 2.7B 总量，以作业为准调 num_layers/d_ff）
MODEL_CONFIGS = {
    "small":  dict(num_layers=12, num_heads=12, d_model=768,  d_ff=3072),
    "medium": dict(num_layers=24, num_heads=16, d_model=1024, d_ff=4096),
    "large":  dict(num_layers=36, num_heads=20, d_model=1280, d_ff=5120),
    "xl":     dict(num_layers=48, num_heads=25, d_model=1600, d_ff=6400),
    "2.7b":   dict(num_layers=32, num_heads=32, d_model=2560, d_ff=10240),
}
FWD_ONLY_MODELS = {"xl", "2.7b"}          # 默认只测 forward，--force-train 可覆盖
# why 默认 fwd-only：fp32 完整训练步 ≈ 16 B/参数（w4+g4+m4+v4）+ 激活，
#   xl≈1.5B → ~24 GB + 激活，消费卡放不下；fwd no_grad 只要 4 B/参数 ≈ 6 GB，安全。
VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
DTYPES = ("float32", "tf32", "bfloat16", "bf16_weights")
DTYPE_LABEL = {"float32": "fp32", "tf32": "tf32", "bfloat16": "bf16 autocast", "bf16_weights": "bf16 weights"}


# ----------------------------------------------------------------------------- 参数
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", choices=list(MODEL_CONFIGS), default=["small"])
    p.add_argument("--ctx", nargs="+", type=int, default=[512])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dtype", nargs="+", choices=DTYPES, default=["float32"],
                   help="bfloat16 = fp32 权重 + autocast；bf16_weights = 权重直接转 bf16（不用 autocast）")
    p.add_argument("--mode", choices=["fwd", "train", "all"], default="all")
    p.add_argument("--warmup", type=int, default=5)
    # warmup 的三重作用：cuBLAS 惰性初始化 + allocator 稳态 + AdamW m/v 首步懒创建。
    # 缺了最后一条，第一个测量步的 optimizer 段会混入状态初始化 kernel，std 被污染。
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-opt", action="store_true", help="不创建优化器，只测 fwd+bwd（optimizer 记为 N/A）")
    # why --no-opt：省下 m+v 的 8 B/参数，large 在小卡上就能测 fwd+bwd 两段（3.5 表够用）
    p.add_argument("--force-train", action="store_true", help="xl / 2.7b 也测训练步")
    p.add_argument("--compile", action="store_true", help="torch.compile(model)；warmup 要够长，包含编译")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="该 dtype 的硬件峰值 TFLOP/s；给出则额外打印利用率（用于检查测量值不应超过峰值）")
    # 用法：--peak-tflops 44（5070 Ti fp32 CUDA-core 估算值）。测出值若超过峰值，
    # 说明 FLOPs 模型或计时有问题——这是一个"测量不可能超过物理上限"的 sanity check。
    p.add_argument("--check-dtypes", action="store_true", help="只做 dtype 实测并打印 3.5 附加表")
    p.add_argument("--out", default="results/bench")
    return p.parse_args()


# ----------------------------------------------------------------------------- 工具
def flops_fwd(cfg, V, B, T):
    """11.3 的 forward FLOPs（不含逐元素运算）。
    # 口径约定：只算 GEMM 家族（matmul/bmm），softmax/RoPE/残差逐元素计 0——
    # 与第 11 章预测同一口径，两边才能相除得"达到算力"。
    # backward 记 2×（dgrad+wgrad 各约一遍），optimizer 记 0（纯逐元素，FLOPs 可忽略）——
    # 所以 full 模式的 TFLOP/s 天然低于 fwdbwd：分母多了 optimizer 时间，分子没变。"""
    d, L, d_ff, N = cfg["d_model"], cfg["num_layers"], cfg["d_ff"], B * T
    return (L * 8 * N * d * d            # q,k,v,o 投影
            + L * 6 * N * d * d_ff       # SwiGLU 三个 GEMM
            + L * 4 * N * T * d          # QK^T 与 PV
            + 2 * N * d * V)             # lm_head


def stats(xs):
    """对一组原始时长做统计。mean±std 直接对应作业表格；
    median 另行保留（抗毛刺，v1 的口径）；raw 留档，表格后续可离线重算。"""
    xs = [float(v) for v in xs]
    med = statistics.median(xs)
    return dict(mean=statistics.fmean(xs),
                std=statistics.stdev(xs) if len(xs) > 1 else 0.0,      # 样本标准差（n-1）
                median=med, min=min(xs), max=max(xs),
                jitter=(max(xs) - min(xs)) / med if med else float("nan"),
                # jitter = (max-min)/median：大说明卡被抢/降频/有其他进程，先查环境再信数据
                raw=[round(v, 3) for v in xs])


def fmt(s):
    return f"{s['mean']:.2f}±{s['std']:.2f}" if s else "N/A"


def sd(dtype):
    return str(dtype).replace("torch.", "")


def git_commit():
    # 记录代码版本：benchmark 数字只在"同一份代码 + 同一台卡"下可复现/可比
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def setup_precision(dtype):
    """每个配置开始前显式设置 TF32，避免上一个配置串味。
    # 关键：allow_tf32 是进程级全局状态，扫描循环里上个配置设了 True、
    # 这个配置不显式设回 False，fp32 基线就悄悄变成 tf32 路径了。"""
    tf32 = dtype == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def autocast_ctx(dtype):
    # 返回"返回上下文管理器的 lambda"：使 phase_step 里能写成 with amp():，每次进入现开现关
    if dtype == "bfloat16":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return lambda: contextlib.nullcontext()


# ----------------------------------------------------------------------------- 构建
def build(model_name, ctx, args, dtype):
    import cs336_basics.model as M
    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW

    torch.manual_seed(args.seed)
    cfg = MODEL_CONFIGS[model_name]
    kw = dict(vocab_size=VOCAB_SIZE, context_length=ctx, d_model=cfg["d_model"],
              num_layers=cfg["num_layers"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
              rope_theta=ROPE_THETA)
    if dtype == "bf16_weights":
        # 先在 CPU 上直接以 bf16 构建，避免 2.7B 的 fp32 权重先占满显存
        # （否则路径是 fp32 建(11GB) → cuda 搬运(11GB) → 再转 bf16，峰值双倍）
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            raw = M.BasicsTransformerLM(**kw)
        finally:
            torch.set_default_dtype(prev)
        raw = raw.to(torch.bfloat16)                  # 若构建时没吃到默认 dtype，这里兜底
    else:
        raw = M.BasicsTransformerLM(**kw)
    raw = raw.cuda().train()
    n_params = sum(p.numel() for p in raw.parameters())
    model = torch.compile(raw) if args.compile else raw
    # 注意：优化器必须拿【原始模型的参数引用】，不是 compiled 包装的。
    # OptimizedModule 与原模型共享同一批参数张量，所以两份引用指向同一片显存，
    # opt.step() 的更新对 compiled 前向立即可见——这是 compile 不破坏训练的根基。
    # 计时只关心形状：batch 一次生成、常驻 GPU，不把生成与 H2D 计进去
    x = torch.randint(0, VOCAB_SIZE, (args.batch, ctx), device="cuda")
    y = torch.randint(0, VOCAB_SIZE, (args.batch, ctx), device="cuda")
    opt = None if args.no_opt else AdamW(raw.parameters(), lr=args.lr)
    return dict(cfg=cfg, n_params=n_params, raw=raw, model=model, x=x, y=y, opt=opt, ce=cross_entropy)


def make_fns(B, amp):
    model, x, y, opt, ce = B["model"], B["x"], B["y"], B["opt"], B["ce"]
    last = {}                                      # 记录最后一次 loss（张量，不在计时里 .item()）
    # why 存张量不存数值：.item() 隐含一次 device→host 同步，放进计时循环会把
    # 异步流水打断。数值化推迟到所有计时结束之后（bench_train 末尾才 .item()）。

    def zero():
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        # set_to_none=True 的两个作用：① 不发射任何 GPU kernel（不写零）；
        # ② 下一步 backward 重新分配 grad，保证每步 kernel 序列逐 step 一致。

    def loss_fn():
        with amp():                                # autocast 只包 forward
            logits = model(x)
        # why 只包 forward：autocast 是前向的 dtype 分派机制；backward 会自动
        # 沿用前向各算子保存的 dtype，不需要（也不应该）再包一层。
        # loss 在 autocast 之外、用 fp32 logits 计算（fp32 模式下 .float() 是空操作）
        # why fp32 算 loss：cross_entropy 对数值敏感（log-sum-exp），fp32 是正确性要求
        loss = ce(logits.float().view(-1, logits.shape[-1]), y.view(-1))
        last["loss"] = loss.detach()
        return loss

    def fwd_nograd():                              # no_grad：不建图，才能与 11.5 的 forward 预测对比
        with torch.no_grad(), amp():
            model(x)
        # 11.5 的预测只算了"算一遍输出"的 FLOPs/流量，不含存激活的写流量——
        # no_grad 保证实测与预测同前提。也正因此，fwd 的数字≠训练步里的前向段。

    def opt_step():
        if opt is not None:
            opt.step()
        # AdamW 状态在首次 step() 懒创建；bench_train 的 warmup 会吸收掉这一次

    def phase_step():
        """一个真实训练步，4 个 event 把它切成 3 段；整步只在最后同步一次。
        # event 创建是 CPU 侧 µs 级开销，且发生在 ev[0].record() 之前，不进任何计时段。
        # 时间戳记在当前 stream 上，GPU 走到该点才打戳——所以各段时长是纯 GPU 时间线，
        # 且三段之和 = ev[0]→ev[3]，与逐段累加严格一致（分解性质）。"""
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        zero()                                     # 只是把梯度置 None，没有 GPU 工作，放在 event 之外
        ev[0].record()
        loss = loss_fn()                           # forward + loss（带计算图）
        ev[1].record()
        loss.backward()                            # 在 autocast 之外；反向算子沿用前向的 dtype
        ev[2].record()
        opt_step()
        ev[3].record()
        torch.cuda.synchronize()                   # 同步后 event 才可读；每步之间无重叠
        # why 全步只 sync 这一次：段内三阶段保持异步发射；若每段都 sync，
        # 会破坏发射流水、且量到的是"同步版"耗时，偏离真实训练节奏。
        return [ev[i].elapsed_time(ev[i + 1]) for i in range(3)]

    def full_step():                               # 用于连续墙钟：不放 event
        # why 不放 event：wall 测的是"零插桩 + 跨步流水"的极限吞吐，
        # event 虽小也是插桩；这里一粒插桩都不加。
        zero()
        loss_fn().backward()
        opt_step()

    return dict(zero=zero, loss_fn=loss_fn, fwd_nograd=fwd_nograd, opt_step=opt_step,
                phase_step=phase_step, full_step=full_step, last=last)


# ----------------------------------------------------------------------------- 计时
def bench_fwd(fns, warmup, iters):
    """逐次 event 计时（每次之间同步），返回 (ms 列表, 峰值 GiB)。"""
    for _ in range(warmup):
        fns["fwd_nograd"]()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()            # 预热之后再重置，排除一次性分配
    # why 预热后重置：cuBLAS workspace、首次 kernel 加载等一次性分配都发生在预热期，
    # 不清掉的话 peak 里混进"非稳态"部分，无法与 11.6 的稳态模型对账。
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fns["fwd_nograd"]()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts, torch.cuda.max_memory_allocated() / 2**30


def wall_ms_per_step(fn, warmup, iters):
    """连续 iters 步只同步一次：墙钟 ms/步（含步与步之间的流水）。
    # 与 bench 的孤立步口径互补：
    #   孤立步 median ≈ 每步本身多重（无流水）
    #   wall        ≈ 真实吞吐节奏（CPU 提前发射下一步，GPU 无空隙）
    # wall ≈ median  → CPU 发射不是瓶颈，GPU 满负荷；
    # wall ≫ median  → launch-bound（典型于 tiny 模型），tok/s 必须按 wall 算才真实。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def probe_memory(fns, has_opt):
    """稳态下的显存探针（不计时）。注意：预热后 Adam 状态已存在，所以两个数都含 8 B/参数的优化器状态。
    # why 与计时分开：显存是"状态量"不是"时间量"，混在计时循环里既测不准也测不稳。
    # reset_peak 只清峰值计数器、不清 allocator——所以已驻留的 m/v 仍被计入，
    # 这里量到的是稳态峰值（含优化器状态），与 11.6 的 est_peak_gib 同口径。"""
    fns["zero"]()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fns["loss_fn"]().backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated() / 2**30   # 反向结束 = 激活+grad 的峰值点
    if has_opt:
        fns["opt_step"]()
        torch.cuda.synchronize()
    peak_step = torch.cuda.max_memory_allocated() / 2**30  # 完整步 = 再加 m/v 状态（若尚不存在则懒创建时计入）
    return peak_bwd, peak_step


def bench_train(fns, B, args):
    for _ in range(args.warmup):                    # 预热：cuBLAS 初始化、allocator、Adam 状态首次分配、编译
        fns["phase_step"]()
    rows = [fns["phase_step"]() for _ in range(args.iters)]
    finite = bool(torch.isfinite(fns["last"]["loss"]).item())
    # 健康检查：lr 大了会 NaN，NaN 不影响 kernel 时长但使结果无意义，必须显式暴露
    has_opt = B["opt"] is not None
    peak_bwd, peak_step = probe_memory(fns, has_opt)
    wall = wall_ms_per_step(fns["full_step"], 2, args.iters)   # wall 的 warmup 给 2 就够：模型已热
    f, b, o = ([r[j] for r in rows] for j in range(3))
    step = [x + y + z for x, y, z in rows]
    # step = 三段逐次累加（分解性质），与 v1 的"整步孤立计时"同口径，可互为对账
    return dict(forward=stats(f), backward=stats(b), optimizer=stats(o) if has_opt else None,
                step=stats(step), wall_ms=wall, tokens_per_s=B["x"].numel() / (wall / 1e3),
                # tok/s 按 wall 算：吞吐是流水概念，用孤立步 median 会高估
                peak_gib_after_bwd=peak_bwd, peak_gib_step=peak_step, loss_finite=finite,
                final_loss=float(fns["last"]["loss"].item()))   # 数值化推迟到计时全部结束


# ----------------------------------------------------------------------------- 单个配置
def run_one(model_name, ctx, args, dtype):
    res = dict(model=model_name, ctx=ctx, batch=args.batch, dtype=dtype, no_opt=args.no_opt,
               status={"build": "OK", "fwd": "skip", "train": "skip"})
    setup_precision(dtype)                          # 每个配置先钉死精度全局状态，防上个配置串味
    res["matmul_precision"] = torch.get_float32_matmul_precision()   # 记录实际生效值，而非假设
    do_fwd = args.mode in ("fwd", "all")
    do_train = args.mode in ("train", "all")
    if do_train and model_name in FWD_ONLY_MODELS and not args.force_train:
        do_train = False
        res["status"]["train"] = "skip(fwd-only)"
    B = fns = None
    try:
        B = build(model_name, ctx, args, dtype)
        fns = make_fns(B, autocast_ctx(dtype))
        res["n_params"] = B["n_params"]
        F = flops_fwd(B["cfg"], VOCAB_SIZE, args.batch, ctx)
        res["fwd_flops"] = F
        with torch.no_grad():                       # 刚初始化时的 loss，用来粗查不同 dtype 的数值是否一致
            res["loss0"] = float(fns["loss_fn"]().item())
        # loss0 的用法：初始化模型 loss ≈ ln(V) = 9.21；fp32 与 bf16 的 loss0 应非常接近
        # （差得多说明 autocast 路径有数值问题），这也是 3.5 表的一列。
        # 顺带完成首次前向：cuBLAS workspace 等一次性分配在此吸收。
        if do_fwd:
            try:
                ts, peak = bench_fwd(fns, args.warmup, args.iters)
                res["fwd"] = stats(ts)
                res["fwd_tflops"] = F / (res["fwd"]["median"] / 1e3) / 1e12
                res["peak_gib_fwd"] = peak
                res["status"]["fwd"] = "OK"
            except torch.cuda.OutOfMemoryError:     # OOM 是"结果"不是"崩溃"：记录后继续
                res["status"]["fwd"] = "OOM"
                torch.cuda.empty_cache()
        if do_train and res["status"]["fwd"] != "OOM":
            # fwd 都 OOM 的话 train 只会更大，不必再试
            try:
                res["train"] = bench_train(fns, B, args)
                res["train"]["tflops"] = 3 * F / (res["train"]["step"]["median"] / 1e3) / 1e12
                # optimizer 计 0 FLOPs → 这是达到算力的下界；对照 fwd_tflops 可拆出
                # "backward 占比"与"optimizer 拖累"两块
                res["status"]["train"] = "OK"
            except torch.cuda.OutOfMemoryError:
                res["status"]["train"] = "OOM"
                torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        res["status"]["build"] = "OOM"              # 连模型都建不起来（2.7b fp32 场景）
    finally:
        B = fns = None                              # 释放引用，下一个配置才有显存
        gc.collect()
        torch.cuda.empty_cache()
        # why 三连：引用清空 → gc 断开 Python 环 → empty_cache 归还 CUDA 缓存块。
        # 缺任何一步，下个配置可能虚假 OOM（显存被上一配置的缓存占着）。
    return res


def status_str(r):
    st = r["status"]
    bad = [f"OOM({k})" for k, v in st.items() if v == "OOM"]
    if bad:
        return ",".join(bad)
    if st.get("train") == "skip(fwd-only)":
        return "仅fwd"
    return "OK"


def print_result(r, peak_tflops=None):
    print(f"\n[{r['model']} ctx={r['ctx']} b={r['batch']} {r['dtype']}] "
          f"params={r.get('n_params', 0) / 1e6:.1f}M  matmul_precision={r['matmul_precision']}  "
          f"status={status_str(r)}  loss0={r.get('loss0', float('nan')):.4f}")
    head = f"  {'phase':14s} {'mean±std':>14s} {'median':>9s} {'min':>8s} {'max':>8s} {'jitter':>7s}"
    if "fwd" in r:
        print(head)
        s = r["fwd"]
        print(f"  {'fwd(no_grad)':14s} {fmt(s):>14s} {s['median']:9.2f} {s['min']:8.2f} {s['max']:8.2f} "
              f"{s['jitter']:6.0%}   {r['fwd_tflops']:.2f} TFLOP/s  peak {r['peak_gib_fwd']:.2f} GiB")
        if peak_tflops:
            print(f"  {'':14s} 相对峰值 {r['fwd_tflops'] / peak_tflops:.0%}")
    if "train" in r:
        t = r["train"]
        print(head)
        # 三段逐行打印；optimizer 为 None（--no-opt）时打 N/A——
        # 表格语义：N/A=没测（没创建优化器），与 0.00（测了但趋近 0）是两回事
        for k, name in (("forward", "forward(+loss)"), ("backward", "backward"),
                        ("optimizer", "optimizer"), ("step", "step=sum")):
            s = t.get(k)
            if s:
                print(f"  {name:14s} {fmt(s):>14s} {s['median']:9.2f} {s['min']:8.2f} {s['max']:8.2f} {s['jitter']:6.0%}")
            else:
                print(f"  {name:14s} {'N/A':>14s}")
        print(f"  e2e wall {t['wall_ms']:.2f} ms/步  {t['tokens_per_s']:.0f} tok/s  "
              f"{t['tflops']:.2f} TFLOP/s(3xF, 按 step 中位数)  "
              f"peak 反向后 {t['peak_gib_after_bwd']:.2f} / 整步 {t['peak_gib_step']:.2f} GiB  "
              f"loss有限={t['loss_finite']}")
        # wall 与 step.median 的关系是免费的瓶颈诊断：相近→GPU-bound；wall 明显更大→launch-bound
        if peak_tflops:
            print(f"  相对峰值 {t['tflops'] / peak_tflops:.0%}")


# ----------------------------------------------------------------------------- 作业表格
def table_32(results):
    print("\n### 3.2 端到端计时（ms，均值±样本std）")
    print("| 模型 | ctx | dtype | forward 均值±std (ms) | backward 均值±std (ms) | optimizer step 均值±std (ms) | 状态 |")
    print("|---|---|---|---|---|---|---|")
    star = False
    for r in results:
        t = r.get("train")
        if t:
            # 训练步口径：forward 段含 loss 且带计算图（与 no_grad 前向不可混填）
            fwd, bwd, opt = fmt(t["forward"]), fmt(t["backward"]), fmt(t["optimizer"])
        elif "fwd" in r:
            # 兜底：只有 fwd 数据时填 no_grad 数字并加 * 标注口径差异
            fwd, bwd, opt, star = fmt(r["fwd"]) + "*", "N/A", "N/A", True
        else:
            fwd = bwd = opt = "N/A"
        print(f"| {r['model']} | {r['ctx']} | {DTYPE_LABEL[r['dtype']]} | {fwd} | {bwd} | {opt} | {status_str(r)} |")
    print("\n说明：带 * 的是 no_grad 前向；其余 forward 是训练中的前向（含 loss、带计算图）。")
    if any(r.get("no_opt") for r in results):
        print("--no-opt：未创建优化器，optimizer 一列为 N/A。")


def table_35(results):
    # 按 (model, ctx) 分组，组内按 dtype 对齐——A/B 对比的分母是同组的 fp32 行
    groups = {}
    for r in results:
        if "train" in r:
            groups.setdefault((r["model"], r["ctx"]), {})[r["dtype"]] = r
    if not any(len(g) > 1 for g in groups.values()):
        return
    print("\n### 3.5 混合精度（forward / backward 取训练步中的阶段均值；显存单位 GiB）")
    print("| 模型 | ctx | dtype | forward (ms) | backward (ms) | 峰值显存 (GiB) | 加速比（fwd+bwd，相对 fp32） | loss0 |")
    print("|---|---|---|---|---|---|---|---|")
    for (m, c), g in groups.items():
        base = g.get("float32")
        # 加速比只用 fwd+bwd：optimizer 段两种 dtype 下逐元素 op 相同、贡献相同，
        # 放进比值只会稀释 GEMM 提速的可读性
        base_fb = (base["train"]["forward"]["mean"] + base["train"]["backward"]["mean"]) if base else None
        for dt, r in g.items():
            t = r["train"]
            fb = t["forward"]["mean"] + t["backward"]["mean"]
            # 峰值口径：有优化器取整步（含 m/v，两种 dtype 相同）；无优化器取反向结束
            peak = t["peak_gib_step"] if t.get("optimizer") else t["peak_gib_after_bwd"]
            sp = f"{base_fb / fb:.2f}×" if base_fb else "N/A"
            print(f"| {m} | {c} | {DTYPE_LABEL[dt]} | {t['forward']['mean']:.2f} | {t['backward']['mean']:.2f} "
                  f"| {peak:.2f} | {sp} | {r.get('loss0', float('nan')):.4f} |")
    print("\n峰值显存：有优化器时取完整训练步，--no-opt 时取反向结束时。loss0 为刚初始化时的 loss，fp32 与 bf16 应非常接近。")


# ----------------------------------------------------------------------------- dtype 实测
def check_dtypes(model_name, ctx, args, dtype):
    """用 hook 实测 autocast 下各类张量的 dtype（3.5 附加表）。
    # 方法：forward hook 抓【输出 dtype】，backward 后读 p.grad 抓【梯度 dtype】。
    # 不依赖属性名、只按类名挂 hook（staff 代码改名也不影响）；
    # uniq() 把同类多实例的 dtype 去重——若同一类出现两种 dtype，本身就是发现。"""
    setup_precision(dtype)
    B = build(model_name, ctx, args, dtype)
    raw, x, y, ce = B["raw"], B["x"], B["y"], B["ce"]
    amp = autocast_ctx(dtype)
    seen = {"Linear": [], "RMSNorm": [], "TransformerBlock": []}   # 按类名挂 hook，不依赖属性名
    hooks = []

    def mk(k):
        def h(m, i, o):
            if torch.is_tensor(o):
                seen[k].append(o.dtype)
        return h

    for m in raw.modules():
        k = type(m).__name__
        if k in seen:
            hooks.append(m.register_forward_hook(mk(k)))
    raw.zero_grad(set_to_none=True)
    with amp():
        logits = raw(x)
    loss = ce(logits.float().view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    for h in hooks:
        h.remove()

    def uniq(ds):
        return "/".join(sorted({sd(d) for d in ds})) or "未捕获（类名不匹配？）"

    rows = [("模型参数", uniq(p.dtype for p in raw.parameters())),          # 预期 fp32（autocast 不动权重主本）
            ("第一层 linear 输出", sd(seen["Linear"][0]) if seen["Linear"] else "未捕获"),   # 预期 bf16（白名单现场 cast）
            ("所有 Linear 输出", uniq(seen["Linear"])),
            ("LayerNorm / RMSNorm 输出", uniq(seen["RMSNorm"])),            # 预期 fp32（数值敏感 + 残差流 fp32）
            ("TransformerBlock 输出（残差流）", uniq(seen["TransformerBlock"])),  # 预期 fp32（embedding 透传 fp32 起步）
            ("logits", sd(logits.dtype)),                                    # 预期 bf16（lm_head 是 Linear）
            ("loss", sd(loss.dtype)),                                        # 预期 fp32（.float() + CE 名单双保险）
            ("梯度", uniq(p.grad.dtype for p in raw.parameters() if p.grad is not None))]  # 预期 fp32（与参数同 dtype）
    print(f"\n### dtype 实测：{model_name} ctx={ctx} dtype={dtype}")
    print("| 对象 | dtype |\n|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    return rows


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    assert torch.cuda.is_available(), "benchmark 必须在 GPU 上跑"
    # meta 随 JSON 存档：benchmark 数字离开（device, torch, git, args）四元组没有意义
    meta = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                git=git_commit(), args=vars(args))
    print(f"device: {meta['device']}  torch {meta['torch']}  cuda {meta['cuda']}  git {meta['git']}")
    print("提示：运行前请用 nvidia-smi 确认卡空闲；抖动(jitter)过大先查是否有别的进程。")

    if args.check_dtypes:
        for dt in args.dtype:
            check_dtypes(args.model[0], args.ctx[0], args, dt)
        return

    # 扫描顺序 model → ctx → dtype：同一 (model, ctx) 的各 dtype 连续跑完，
    # table_35 的 A/B 对比在时间上邻近，环境漂移（降频等）影响最小
    results = []
    for model_name in args.model:
        for ctx in args.ctx:
            for dt in args.dtype:
                r = run_one(model_name, ctx, args, dt)
                print_result(r, args.peak_tflops)
                results.append(r)

    table_32(results)
    table_35(results)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    name = ("_".join(args.model) + "_ctx" + "-".join(map(str, args.ctx)) + f"_b{args.batch}_" + "-".join(args.dtype)
            + ("_noopt" if args.no_opt else "") + ("_compile" if args.compile else ""))
    p = Path(args.out) / f"bench2_{name}.json"
    p.write_text(json.dumps(dict(meta=meta, results=results), indent=2, ensure_ascii=False))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
