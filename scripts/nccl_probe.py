import torch, torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    r = dist.get_rank()
    torch.cuda.set_device(r)
    x = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="cuda")  # 512 MB
    for _ in range(5): dist.all_reduce(x)          # warmup
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): dist.all_reduce(x)
    e.record(); torch.cuda.synchronize()
    t = s.elapsed_time(e) / 10 / 1000              # 秒
    S = x.numel() * 4
    if r == 0:
        print(f"size={S/1e9:.2f}GB  algbw={S/t/1e9:.2f}GB/s  busbw={2*(1)/2*S/t/1e9:.2f}GB/s")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
