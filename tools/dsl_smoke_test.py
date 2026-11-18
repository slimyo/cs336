"""tools/dsl_smoke_test.py — CuTe DSL 工具链冒烟测试（M3 的门槛）。

验证三件事：
  1. nvidia-cutlass-dsl 能 import，版本对得上（你的环境：4.8.1）
  2. 一个最小的 @cute.kernel 能通过 cute.compile 编译
  3. 编译产物能 launch 并算出正确结果（用 torch 对照）

跑法：
  uv run python tools/dsl_smoke_test.py
"""
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


print(f"cutlass DSL 版本: {cutlass.__version__}")
print(f"torch: {torch.__version__}，GPU: {torch.cuda.get_device_name(0)}")
major, minor = torch.cuda.get_device_capability(0)
print(f"compute capability: sm_{major}{minor}")


# ---------------------------------------------------------------- 最小 kernel：c = a + b
# 故意选向量加法：与 attention 无关，任何失败都只能是工具链问题，不会和算法混淆。

@cute.kernel
def vadd_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < n:
        mC[i] = mA[i] + mB[i]


@cute.jit
def vadd(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = cute.size(mA)
    threads = 256
    blocks = (n + threads - 1) // threads
    vadd_kernel(mA, mB, mC, n).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    n = 1 << 20   # 1M 元素，跨多个 block，顺便验证 grid 索引

    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty(n, device="cuda")

    # torch tensor → cute tensor（DSL 的运行时桥）
    ta, tb, tc = from_dlpack(a), from_dlpack(b), from_dlpack(c)

    compiled = cute.compile(vadd, ta, tb, tc)
    print("cute.compile 通过")

    compiled(ta, tb, tc)
    torch.cuda.synchronize()
    print("launch 通过")

    torch.testing.assert_close(c, a + b, atol=0, rtol=0)   # 逐元素加法应当精确
    print("结果正确：CuTe DSL 工具链冒烟测试全部通过 ✓")


if __name__ == "__main__":
    main()
