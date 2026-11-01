# ncu（Nsight Compute）

读完你应该能回答：

1. ncu 到底测了什么，和 nsys 有什么本质区别
2. 什么时候该用，什么时候用了也没用
3. 怎么只抓你关心的 kernel，抓完怎么读，怎么得出"算力受限还是带宽受限"的结论
4. 有哪些坑会让你得出错误结论

---

## 第一部分：是什么

### 1.1 一句话

**ncu 是单个 kernel 的显微镜。** 它深入一个 kernel 的内部，测量它对 GPU 硬件的使用情况：算力用了多少、显存带宽用了多少、有多少线程在同时跑、线程都在等什么。

### 1.2 和 nsys 的分工

| | nsys | ncu |
|---|---|---|
| 视角 | 整个程序的**时间线** | **单个 kernel** 的内部 |
| 回答 | 哪些 kernel 占时间？有没有空隙？CPU 发射是不是瓶颈？ | 这个 kernel **为什么**慢？离硬件极限还有多远？ |
| 数据 | 起止时间、次数 | 硬件性能计数器（算力利用率、带宽利用率、占用率、stall 原因……） |
| 开销 | 较小 | **很大**（每个 kernel 被重放多次） |
| 绝对时间 | 只看构成 | **完全不可信** |

类比：nsys 是"体检报告里的各项指标排名"，告诉你哪个器官占的比重最大；ncu 是"对那个器官做 CT"，告诉你它为什么有问题。

### 1.3 ncu 是怎么测量的：重放（replay）

GPU 硬件在**一次运行**里只能同时收集有限的几个性能计数器。而 ncu 想收集几十上百个指标，所以它的做法是：

```
同一个 kernel：
  第 1 遍：保存显存状态 → 运行 → 收集一批计数器
  第 2 遍：恢复显存状态 → 重新运行 → 收集下一批计数器
  ...
  第 N 遍：……
把各遍的结果拼起来，得到完整的指标表
```

这个机制带来三个直接后果，后面会反复用到：

| 后果 | 说明 |
|---|---|
| **很慢** | 一个 kernel 可能被重放几十遍（`--set full` 尤其多），所以必须只抓少数几个 |
| **duration 不可信** | 重放、锁频、清缓存，使 ncu 报告的时间和真实运行不一样，**不能和 nsys、benchmark 比** |
| **kernel 被串行化** | 抓取时 kernel 一个一个单独跑，不反映真实的并发与重叠 |

所以：**ncu 只看"利用率和比例"，不看绝对时间。**

---

## 第二部分：为什么用

### 2.1 它解决的问题

nsys 只能告诉你"softmax 占了 15% 的 GPU 时间"。至于为什么，比如"因为它是访存密集的"，用时间占比只能**推断**。

ncu 能**直接验证**这个推断：看到 softmax kernel 的 Memory Throughput 是 80%、Compute 只有 10%，才能写"带宽受限"，而不是"推断为带宽受限"。

### 2.2 什么时候该用

| 情况 | 用什么 |
|---|---|
| 某个 kernel 占时间很多，想知道它离硬件极限多远 | **ncu** |
| 想确认某个 kernel 是算力受限还是带宽受限 | **ncu** |
| 想比较融合/优化前后同一个 kernel 的表现 | **ncu**（比较指标，不比较 duration） |
| 想验证 roofline 预测（如"逐元素访存受限""大 GEMM 接近峰值"） | **ncu** |

### 2.3 什么时候用了也没用

| 情况 | 原因 | 该用什么 |
|---|---|---|
| GPU 利用率低、kernel 之间有大空隙（**发射受限**） | 每个 kernel 单看可能很健康，问题出在 kernel 太多太碎、CPU 追不上 | nsys 时间线；减少 kernel 数（融合、`torch.compile`、CUDA graph） |
| 逐参数 AdamW 这种一大批极小 kernel | 每个 kernel 极短，瓶颈是发射开销 | nsys 的 `cuda_api_sum` 和空隙分析 |
| 想知道绝对速度 | ncu 的时间不可信 | benchmark |

**顺序原则：先 nsys 算 GPU 利用率。** 接近 1 才值得上 ncu；明显低于 1 先回去解决发射问题。

### 2.4 在你这个项目里的位置

ncu 为 11 章的"预测"提供**硬件证据**。例如：

| 11 章预测 | ncu 要看到什么才算验证 |
|---|---|
| 缩放、mask、softmax、SiLU、残差等逐元素 kernel 访存受限 | Memory Throughput 高，Compute 很低 |
| FFN 的大 GEMM 计算受限，接近峰值 | Compute Throughput 高 |
| QKᵀ、PV 的 GEMM 强度偏低 | Compute 利用率达不到大 GEMM 水平，看 occupancy 和形状 |
| RMSNorm 的 reduce 访存受限且可能并行度低 | Memory 偏高，或两者都低，看 occupancy |

---

## 第三部分：核心概念（读懂 ncu 输出必需）

### 3.1 三种"受限"

一个 kernel 的耗时，通常被下面某一种因素卡住：

| 类型 | 含义 | 典型 kernel |
|---|---|---|
| **算力受限**（compute-bound） | 计算单元（FMA、Tensor Core）接近满负荷 | 大矩阵乘 |
| **带宽受限**（memory-bound） | 显存（DRAM）带宽接近满负荷，计算单元在等数据 | 逐元素运算、softmax、归一化 |
| **延迟受限**（latency-bound） | 两边都没用满：并行度不够或 kernel 太小，线程大量时间在等 | 很小的 kernel、grid 太小的 kernel |

还有第四种，**发射受限**：GPU 在等 CPU 发射下一个 kernel。它发生在 kernel **之间**，ncu 看不到，要用 nsys。

### 3.2 Speed Of Light（SOL）：最重要的一个 section

ncu 里最先看的东西。它给出两个核心百分比：

| 指标 | 含义 |
|---|---|
| **Compute (SM) Throughput** | 计算单元的利用率，相对**理论峰值**的百分比 |
| **Memory Throughput** | 存储系统（DRAM、L2、L1 等里最繁忙的那个）的利用率，相对理论峰值的百分比 |

百分比的分母是**理论峰值**（在 ncu 锁定的时钟下），所以 80% 的含义是"达到了硬件上限的 80%"。

读法（粗略判据）：

| Compute (SM) | Memory | 结论 |
|---|---|---|
| 高 | 低 | 算力受限。已接近峰值就没有多少优化空间 |
| 低 | 高 | **带宽受限**。要减少访存（融合、降精度），而不是算得更快 |
| 都低 | 都低 | 并行度不足或 kernel 太小，通常是**延迟受限**。看 grid 大小、occupancy |
| 都高 | 都高 | 已接近硬件极限 |

细化：Memory Throughput 是"最繁忙的那一级"。要分清是 DRAM、L2 还是 L1：

- DRAM 高：真的在吃显存带宽（带宽受限的典型）；
- DRAM 低但 L1/L2 高：数据复用好，瓶颈在缓存层；
- 想看 Compute 具体是哪个功能单元繁忙（FMA、Tensor、ALU……），看 Compute Workload Analysis。

### 3.3 Roofline（屋顶线）

把 kernel 画在一张图上：

- 横轴：**算术强度** = 总 FLOPs ÷ 总访存字节数（FLOP/Byte）
- 纵轴：达到的性能（FLOP/s）
- 图上有两条线：一条斜线（带宽屋顶）和一条水平线（算力屋顶），交点叫 ridge point。

```
性能上限 = min( 峰值算力 P ,  算术强度 I × 峰值带宽 BW )
```

- 点落在**斜线区**（强度低）：带宽受限；
- 点落在**平线区**（强度高）：算力受限。

举例：fp32 的 `y = x + 1`，每个元素读 4 字节、写 4 字节，只做 1 次加法，强度约 0.125 FLOP/Byte，远低于任何 GPU 的 ridge point，必然带宽受限。

ncu 的 `SpeedOfLight_RooflineChart` section 直接给出这张图和你的 kernel 的位置。

### 3.4 Occupancy（占用率）

GPU 的每个 SM 可以同时驻留很多个 warp（32 线程一组）。**Occupancy = 实际驻留的 warp 数 ÷ 理论最大值。**

高占用率有助于隐藏访存延迟（一个 warp 在等数据时，调度器切换到另一个 warp）。ncu 会同时给出：

- **理论占用率**：由 block 大小、每线程寄存器数、每 block 共享内存决定的上限；
- **实际占用率**：运行中真实达到的。

两者差很多，说明 grid 太小或负载不均；理论占用率本身低，说明被寄存器或共享内存限制。

### 3.5 Launch Statistics 与 wave

- **Grid / Block 大小**：kernel 起了多少 block、每个 block 多少线程。
- **Waves Per SM**：block 总数能铺满 SM 几轮。如果 block 数远少于"SM 数 × 每 SM 可驻留 block 数"，GPU 没被填满（**kernel 太小**）；整数轮次的"尾部不足一轮"也会造成浪费。

### 3.6 Warp State Statistics（stall 原因）

每个 warp 不能发射指令时，ncu 记录原因。常见的几类（名字以 ncu 实际输出为准）：

| 现象 | 含义 |
|---|---|
| 等全局内存（如 long scoreboard） | 在等 DRAM/L2 的数据，带宽或延迟问题 |
| 等同步（barrier） | 线程块内同步等待 |
| 等数学流水线（math pipe throttle） | 计算单元忙，算力受限的信号 |
| 等指令获取/依赖 | 指令级的依赖或取指问题 |

它回答"线程具体停在哪"，在 Compute/Memory 判断之后用于细究。

---

## 第四部分：怎么做

### 4.1 环境与权限

**版本**：你的卡是 sm_120（Blackwell 消费级），需要较新的 ncu。

```bash
ncu --version          # 看版本
ncu --list-sections    # 看本机支持的 section
ncu --list-sets        # 看有哪些预设集合
```

我的印象是对 sm_120 的支持对应 CUDA 12.8 及以后的工具包，**这一点请以本机能否识别你的卡为准**。

**权限**：读取 GPU 性能计数器可能需要管理员放开，共享服务器上常见报错：

```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

处理：联系管理员放开计数器权限（驱动模块的相关选项），或用有权限的账号运行。你在共享服务器上自己改不了的话，只能走管理员。这是最常见的第一道坎，**建议最早去试一次 `-c 1`**，别等到最后。

**指标名随版本和架构变化**：不要照抄旧教程里的指标名，以 `ncu --query-metrics` 和 section 为准。

### 4.2 只抓你关心的 kernel（必须）

ncu 会把每个被抓取的 kernel 重放多遍。**不加限制会慢到跑不完。** 三种过滤方式，可以组合。

| 方式 | 选项 | 说明 |
|---|---|---|
| 按名字 | `-k regex:softmax` | 名字来自 nsys 的"待深挖清单" |
| 按次数 | `-s 20 -c 3` | 在**匹配的** kernel 里，跳过前 20 次，只抓接下来的 3 次 |
| 按 NVTX 区间 | `--nvtx --nvtx-include "attn.softmax/"` | 复用 NVTX 标注，需要脚本里已经打了这些区间 |

**关于 `-k`**：

- 默认匹配的是 kernel 的**函数名**（不含模板参数的基础名）。PyTorch 的 kernel 名往往是很长的模板，想匹配模板参数里的内容，需要 `--kernel-name-base demangled`。
- 多个关键字用正则的 `|`：`-k regex:"gemm|cutlass|xmma"`。

**关于 NVTX 过滤的重要限制**：ncu 的 NVTX 过滤同样按**发射线程**匹配区间。所以它对 backward 的区间**很可能失效**（反向 kernel 由 autograd 线程发射，和 nsys 里的归因问题同源）。对反向 kernel，用 `-k` 名字加 `-s/-c`。（这是基于机制的判断，请用 `-c 1` 验证过滤是否抓到了你要的 kernel。）

### 4.3 只在测量区间抓取

和 nsys 一样，预热不应被抓。脚本里已经有 `torch.cuda.profiler.start()` / `stop()`，ncu 用：

```bash
--profile-from-start off
```

表示"程序启动时不抓，等 `profiler.start()` 之后才开始"。

### 4.4 完整命令示例

```bash
mkdir -p results/ncu
CUDA_VISIBLE_DEVICES=0 ncu \
  -o results/ncu/softmax --force-overwrite \
  --profile-from-start off \
  -k regex:softmax -c 3 \
  --set full \
  python scripts/profile_run.py --model small --ctx 512 --batch 4 \
        --mode fwd --warmup 3 --steps 1 --nvtx-fine
```

注意：

- 如果你的启动方式是 `uv run python ...` 作为被测程序，ncu 默认只分析它直接启动的进程，需要加 `--target-processes all`。或者直接把 `ncu` 放在 `uv run` 里面、让它启动 `python`。
- `--mode fwd --steps 1`：ncu 下尽量缩小运行量。
- 形状要与真实运行一致（见 4.8）。

### 4.5 常用选项速查

| 选项 | 作用 |
|---|---|
| `-o 文件名` / `-f` | 输出 `.ncu-rep`；允许覆盖 |
| `--set basic/detailed/full` | 采集的指标集合：basic 快，full 全但慢；先 basic 看一眼 |
| `--section 名字` | 只采指定 section，比 `--set full` 快很多 |
| `-k` / `-s` / `-c` | 按名字、跳过次数、抓取次数过滤 |
| `--profile-from-start off` | 配合 `profiler.start()`，预热不被抓 |
| `--nvtx --nvtx-include "区间/"` | 按 NVTX 区间过滤 |
| `--clock-control none` | 默认会锁频使结果稳定；想看接近真实运行的频率时关掉 |
| `--cache-control none` | 默认每遍重放前会清空缓存；想保留真实运行中的缓存状态时关掉 |
| `--target-processes all` | 被测程序会派生子进程时用 |
| `--query-metrics` | 查本机支持的指标名 |

### 4.6 推荐工作流

```
① 先在 nsys 里算 GPU 利用率，确认值得用 ncu（接近 1）
   │
② 从 nsys 的 cuda_gpu_kern_sum 里挑 3～6 个 kernel，记下完整名字
   （总时间占比最大的 GEMM、softmax、最大的逐元素、reduce）
   │
③ 先 -c 1 试通流程（权限、过滤是否抓到了目标、能不能出报告）
   │
④ 用 --set basic 或指定 --section 快速看一遍
   │
⑤ 需要细究的 kernel 再用 --set full，-c 取 1～3
   │
⑥ 读 SOL → 判断受限类型；读 occupancy / stall → 找原因
   │
⑦ 把具体指标写进回填表的"ncu 证据"列
```

### 4.7 读结果

**命令行**：

```bash
ncu --import results/ncu/softmax.ncu-rep --page details
```

`details` 页会按 section 打印每个指标，并附带 ncu 内置的规则给出的提示（通常是"某项偏低/偏高"和优化建议）。这些提示很有参考价值，但仍要用 3.2 的判据自己判断。

**图形界面**：把 `.ncu-rep` 下载到本地用 Nsight Compute GUI 打开（GUI 版本不低于生成报告的 ncu）。优点是 roofline 图、各 section 可视化直观，kernel 之间可以对比。

**先看哪几个 section**：

| 想回答 | 看什么 |
|---|---|
| 算力还是带宽受限 | `Speed Of Light`：Compute (SM) Throughput 与 Memory Throughput |
| 在 roofline 上的位置 | roofline 图 |
| 访存是否高效 | `Memory Workload Analysis`：DRAM 吞吐、L2 命中率、访存合并 |
| 并行度够不够 | `Launch Statistics`（grid/block、waves）、`Occupancy` |
| 线程停在哪 | `Warp State Statistics`（stall 原因） |

### 4.8 一个最小练习：亲眼看三种"受限"

先用一个玩具脚本，同时造出带宽受限、算力受限、延迟受限的 kernel，看清 ncu 对它们的区别，再去分析真实模型：

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = False      # 保持 fp32 路径，不走 TF32

x     = torch.randn(1 << 26, device="cuda")         # 约 256 MB，足够大
a     = torch.randn(8192, 8192, device="cuda")
b     = torch.randn(8192, 8192, device="cuda")
small = torch.randn(256, device="cuda")

def run():
    y = x + 1            # 大逐元素：预期带宽受限
    c = a @ b            # 大 GEMM：预期算力受限
    s = small + 1        # 极小 kernel：预期延迟受限（并行度不足）
    torch.cuda.synchronize()

for _ in range(3): run()                             # 预热
torch.cuda.synchronize()
torch.cuda.profiler.start()
run()
torch.cuda.profiler.stop()
```

```bash
ncu -o results/ncu/toy -f --profile-from-start off -c 5 --set basic python toy.py
ncu --import results/ncu/toy.ncu-rep --page details --print-summary per-kernel
```

预期（以实测为准）：

| kernel | Compute (SM) | Memory | 结论 |
|---|---|---|---|
| `x + 1`（大逐元素） | 低 | 高 | 带宽受限 |
| 大 GEMM | 高 | 中或低 | 算力受限 |
| `small + 1` | 都很低 | 都很低 | 延迟受限（grid 太小，看 Launch Statistics） |

`-c 5` 是为了留余量：实际抓到哪些 kernel 取决于 PyTorch 与 cuBLAS 的实现，先用 summary 看清楚抓到了什么，再收窄。

### 4.9 把结果写进回填表

11/12 章的回填模板里有"ncu 证据"列。写**具体指标**，不写结论性的话：

| 好 | 不好 |
|---|---|
| `softmax：Memory 约 80%，Compute 约 10%，带宽受限` | `推断为访存受限` |
| `FFN 大 GEMM：Compute 约 85%，Memory 约 40%` | `接近峰值` |
| `RMSNorm reduce：Memory 约 30%，Compute 约 5%，achieved occupancy 低` | `并行度不够` |

（上面的百分比只是示意，请填你实测的数字。）

只对需要解释的差异填；没抓 ncu 的行留空。

---

## 第五部分：坑

| 坑 | 说明 | 处理 |
|---|---|---|
| **拿 ncu 的 duration 与 nsys 或 benchmark 比** | 锁频、清缓存、重放，数字对不上，容易误以为有 bug | 不比。ncu 只用利用率与比例 |
| **不加过滤** | 全程序重放，慢到跑不完 | `-k`、`-s`、`-c`，先 `-c 1` |
| **形状与真实运行不一致** | 想验证 small、T=512 的结论，却用缩小的玩具形状，结论不适用 | 用和真实运行完全相同的配置 |
| **TF32 没关** | GEMM 走了不同的路径（Tensor Core），与预测不符 | 确认第 10 章的 TF32 设置；看 kernel 名里是否有 tf32 |
| **指标名与版本** | sm_120 上旧教程的指标名可能失效 | 用 section 和 `--query-metrics` |
| **默认清缓存** | 真实运行中输入可能还在 L2 里，ncu 默认每遍清缓存，结果偏"冷" | 需要时 `--cache-control none`，并在报告里注明 |
| **默认锁频** | 锁在基础频率，不反映真实 boost 下的频率 | 想接近真实时 `--clock-control none` |
| **发射受限的问题来用 ncu** | 单个 kernel 很健康，却找不到问题 | 先在 nsys 里做 GPU 利用率自检 |
| **NVTX 过滤用在 backward 上** | 按发射线程匹配，反向 kernel 的线程没有你的区间 | 反向 kernel 用 `-k` 名字加 `-s/-c` |
| **kernel 名对不上** | PyTorch 版本/架构变化导致名字不同 | 以 nsys 实际清单为准；或单算子隔离后用 summary 看名字 |
| **权限报错** | `ERR_NVGPUCTRPERM` | 联系管理员，建议最早试 |
| **卡不空闲** | 别的进程占用，结果不可信 | 运行前 `nvidia-smi`，用 `CUDA_VISIBLE_DEVICES` 隔离 |

---

## 速查清单

- [ ] 先用 nsys 做 GPU 利用率自检，接近 1 再上 ncu
- [ ] 待深挖的 kernel 名字来自 nsys 的清单
- [ ] 一定过滤：`-k` / `-s` / `-c`，先 `-c 1` 试通
- [ ] `--profile-from-start off`，预热不被抓
- [ ] 形状、dtype、TF32 设置与真实运行一致
- [ ] 先看 Speed Of Light 的 Compute 与 Memory 两个百分比，再看 occupancy 和 stall
- [ ] 不比较 ncu 的 duration，只用利用率与比例
- [ ] 回填表里写具体指标，不写"推断"
- [ ] 指标名、section 名、kernel 名都以本机实际输出为准