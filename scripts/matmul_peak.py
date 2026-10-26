import argparse, csv, statistics, torch

MODES = {  # name: (dtype, float32_matmul_precision)
    "fp32_tf32_off": (torch.float32, "highest"),
    "fp32_tf32_on":  (torch.float32, "high"),
    "bf16":          (torch.bfloat16, "highest"),
}

def time_matmul(n, dtype, iters, warmup=10):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):          # 连续发射，摊掉 launch 开销
        a @ b
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms

def tflops(n, ms):
    return 2 * n**3 / (ms * 1e9)       # 2n^3 FLOPs / (ms*1e-3 s) / 1e12

def bandwidth(gib=1, reps=20):
    x = torch.empty(gib * 2**30 // 4, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3): y.copy_(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): y.copy_(x)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / reps
    return 2 * x.numel() * 4 / (ms * 1e6)   # 读+写，GB/s

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--tag", default="gpu")
    args = p.parse_args()

    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(),
          "torch", torch.__version__, "cuda", torch.version.cuda)
    rows = []
    for name, (dt, prec) in MODES.items():
        torch.set_float32_matmul_precision(prec)
        print(f"[{name}] float32_matmul_precision = {torch.get_float32_matmul_precision()}")
        for n in args.sizes:
            iters = max(10, int(4e12 / (2 * n**3)))
            tf = [tflops(n, time_matmul(n, dt, iters)) for _ in range(args.reps)]
            med, mx = statistics.median(tf), max(tf)
            print(f"  n={n:>6}  median {med:7.1f}  max {mx:7.1f}  TFLOPs/s")
            rows.append([name, n, f"{med:.2f}", f"{mx:.2f}"])
    bw = bandwidth()
    print(f"[memcopy] {bw:.0f} GB/s (read+write)")
    rows.append(["memcopy", "1GiB", f"{bw:.1f}", ""])
    with open(f"results/matmul_peak_{args.tag}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["mode", "n", "median", "max"]] + rows)
