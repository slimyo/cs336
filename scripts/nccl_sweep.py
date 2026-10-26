import os, sys, torch, torch.distributed as dist

SIZES_MB = [1, 4, 16, 64, 256, 1024]
ITERS    = {1: 200, 4: 100, 16: 50, 64: 20, 256: 10, 1024: 5}

def bench(x, iters):
    for _ in range(5): dist.all_reduce(x)          # warmup：所有 rank 都要执行
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): dist.all_reduce(x)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters               # ms

def main():
    dist.init_process_group("nccl")
    r = dist.get_rank(); torch.cuda.set_device(r)
    sizes = [int(a) for a in sys.argv[1:]] or SIZES_MB

    if r == 0:   # 只有"打印"在 if 里
        shm_off = os.environ.get("NCCL_SHM_DISABLE") == "1"
        print(f"<!-- transport: {'NET/socket (SHM off)' if shm_off else 'SHM/direct (default)'} -->", flush=True)

    for mb in sizes:                               # ★ 集合通信在 if 外：每个 rank 都跑
        x = torch.empty(mb << 18, dtype=torch.float32, device="cuda")
        t = bench(x, ITERS.get(mb, 10))
        if r == 0:
            algbw = x.numel() * 4 / (t / 1e3) / 1e9
            print(f"| {mb} MB | {ITERS.get(mb, 10)} | {t:.3f} | {algbw:.2f} | {algbw:.2f} | |", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
