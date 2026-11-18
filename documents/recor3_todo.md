# Part 3：分布式数据并行（DDP）实验设计与学习路线

> 每个阶段按「干什么 / 为什么 / 验收 / takeaway / 与推理框架的联系」展开；最后一节列出欠缺项和补法。

## 1 全局视图

**你的条件**：2 张 RTX 5070 Ti（sm\_120，每卡 16 GiB），无 P2P、无 NVLink，NCCL 走 SHM（GPU→主机共享内存→GPU，两段拷贝），实测 all\_reduce busbw 5.2 GB/s；已有 CuTe FlashAttention 与 nsys/ncu 的分析经验。

**这一部分在你目标里的位置**：DDP 是训练侧的东西，但它练的四种能力在推理优化里一样要用：（1）用 α-β 模型预测通信时间；（2）把小消息聚合成大消息；（3）用 stream 与 event 让通信和计算重叠；（4）用 nsys 把「是否真的重叠」量化。推理里张量并行的 all-reduce、KV cache 的传输与换入换出、CUDA graph 里的集合通信，都是同一套思路。

**两条要先写下的判断**：

1. 本机链路慢，通信时间大于反向时间（模板里 medium 约 23 ms、large 约 95 ms；每卡每步只有 1024 个 token，反向很短）。因此 overlap 最多省下约一个 T\_bwd，其余通信仍然暴露。报告里用「暴露通信」而不是总通信来衡量。
2. bf16 autocast 时参数与梯度仍是 fp32，通信量按参数量×4 字节；只有主动把梯度转成 bf16 才是×2。5.6 的理论值要区分这两种情形。

| 阶段 | 核心动作 | 核心 takeaway | 推理侧的对应物 |
| --- | --- | --- | --- |
| P0 | α-β 测量、防挂死、写预测 | 通信时间模型 | 张量并行 decode 时消息很小，α 主导 |
| M1 | 单卡真值 | 数值对齐方法 | 换并行度或换 kernel 后的 logprob 回归 |
| M2 | naive DDP | 集合通信语义 | 所有 rank 必须同序调用 |
| M3 | flat DDP | 小消息聚合、内存布局 | KV 块合并传输、CUDA graph 的静态 buffer |
| M4 | overlap | stream/event、重叠上限 | TP 通信与计算重叠、KV offload 的 pinned 内存加独立拷贝 stream |
| M5 | bucketed | 粒度权衡 | chunked prefill 的 chunk、graph 的 batch 档位 |
| M6 | benchmark 与 nsys | 暴露与隐藏的归因 | TTFT/TPOT 分解、launch-bound 诊断 |

## 2 P0：度量地基（约 0.5 天）

**干什么**

1. 确认两卡空闲（nvidia-smi），fp32 路径关 TF32；每个版本交错跑 3 次取中位数（功耗墙与温度会让单次结果漂移）。
2. 跑 scripts/nccl\_knee.py，对 all\_reduce 的时间做 T = α + size/β 的拟合，得到每次调用的固定延迟 α 和带宽 β。
3. 用 NCCL\_DEBUG=INFO 跑一次，把传输类型（SHM）、通道数、算法（Ring/Tree）、协议（LL/Simple）记进报告。
4. 防挂死：init\_process\_group 设 60 秒超时；调试期开 TORCH\_DISTRIBUTED\_DEBUG=DETAIL；所有命令外套 timeout；集合通信永远不放进 if rank == 0。
5. 写预测表（naive、flat、overlap 的通信与步时间公式），测之前定稿。

**为什么**：α 和 β 是后面所有预测的输入；没有预测，实测数字只是数字，没法判断「符合预期还是出了 bug」。超时保护是因为 nccl\_sweep 那次事故说明挂死会吃掉一整个下午。

**验收**：knee 曲线与 α、β 拟合值；NCCL 日志截图；预测表。

**takeaway**

- all\_reduce 的 busbw = algbw × 2(n−1)/n，n=2 时两者相等，所以通信时间 ≈ 字节数 ÷ 5.2 GB/s。
- 通信时间由两项构成：与大小无关的启动延迟 α，以及与大小成正比的 n/β；小消息看 α，大消息看 β。
- NCCL 会按消息大小和拓扑选算法与协议；拓扑（NVLink、P2P、SHM、PCIe、网络）决定 β。

**与推理框架的联系**：张量并行每层有两次 all-reduce（注意力输出后、MLP 输出后）。decode 阶段消息只有 batch×hidden×2 字节（例如 8×4096×2 = 64 KB），完全落在 α 主导区，这就是推理框架会写自定义 all-reduce、并偏好低延迟协议的原因；这类自定义实现通常对 P2P 或 NVLink 有要求，具体条件以对应框架的文档为准。你的机器恰好没有 P2P，这是值得单独记一笔的差异。

## 3 M1：单卡真值（约 0.5 天）

**干什么**

1. 用固定 seed 预先生成 N 个 global batch（批 8、ctx 256 的随机 token），所有版本消费同样的数据；DDP 时 rank r 取第 r×4 到 r×4+3 行。
2. 单卡跑 global batch 8，记录每步 loss，保存第 1 步梯度和最终权重（fp32，放 CPU）。
3. 开确定性：CUBLAS\_WORKSPACE\_CONFIG=:4096:8、固定初始化 seed、AdamW。
4. 同时测单卡 batch=4（无通信参照），并把一步拆成前向、反向、优化器三段计时（后面 overlap 的预测要用 T\_bwd）。

**为什么**：没有真值，通信的 bug（重复同步、错位平均）很难发现；分段时间给出 overlap 的上限。

**验收**：同一脚本跑两遍，loss 与权重逐位一致，否则后面的「对齐」没有意义。

**takeaway**

- 数据并行的数学等价：等分的 batch 加上均值 loss，各 rank 梯度的平均就等于全局 batch 的梯度。
- 浮点加法不满足结合律，所以 DDP 与单卡只能近似对齐；容差的定法（逐参数相对误差，例如 1e-4）要在报告里写清，并写出实际观察到的值。
- 确定性的来源：cuBLAS workspace、原子操作、算法选择。

**与推理框架的联系**：推理框架在换并行度、换 kernel 或量化之后，都用「对齐参考实现的 logprob」做回归。TP 切分后 all-reduce 的累加顺序变化，带来的就是同一类数值差异；你做 FlashAttention 时已经有 fp32 参考加容差的习惯，这里是把它推广到多卡。

## 4 M2：naive DDP（约 0.5 天）

**干什么**

1. 构造时把 module 的参数和 buffers 从 rank 0 广播。
2. forward 直通。
3. finish\_gradient\_synchronization 里对每个 requires\_grad 的参数做阻塞的 all\_reduce（SUM），再除以 world size；用 SUM 加除法而不是 ReduceOp.AVG，因为 gloo 不支持 AVG（CPU 版 pytest 会用到）。
4. 遍历用 module.parameters()（按对象去重），共享参数只同步一次；某个 rank 上 grad 为 None 的参数当作零张量参与通信，保证各 rank 的调用序列一致。
5. 跑 ToyModel 与 ToyModelWithTiedWeights；每步后检查两个 rank 的参数逐位相同；与真值比较；跑 pytest。

**为什么**：这是最慢最简单的基线，同时把共享参数和顺序一致性这两个最容易出事的点先暴露出来。

**验收**：N 步 loss 曲线与真值重合，权重最大误差在容差内，两 rank 参数一致，pytest 通过。

**takeaway**

- 集合通信的语义：所有 rank 必须以相同顺序调用，且形状、dtype 一致，否则挂死或结果错。
- async\_op 的 stream 语义：NCCL 在自己的 stream 上运行，Work.wait() 让当前 stream 等它，而不阻塞 CPU。
- 共享参数的两份梯度，在 AccumulateGrad 处就已被 autograd 合并，只需对最终的 p.grad 同步一次；用 named\_parameters(remove\_duplicate=False) 或遍历 state\_dict 会重复同步。
- 初始化广播为什么必须有。

**与推理框架的联系**：张量并行里同样是「所有 rank 同序调用」的硬约束。推理框架里 scheduler 要把本步的 batch 元数据广播给所有 worker，让它们执行完全相同的算子序列；任何 rank 间的调度不一致（例如各自按不同请求形状走了不同分支）都会死锁，这与你在 nccl\_sweep 里踩过的坑同源。

## 5 M3：flat DDP（约 0.25 天）

**干什么**

1. 版本 A：torch.\_utils.\_flatten\_dense\_tensors 把所有梯度拼成一个张量，一次 all\_reduce，再 unflatten 拷回。
2. 版本 B：预分配一块平坦 buffer，把每个 p.grad 设为它的视图，反向时 autograd 就地累加进去，通信前后都没有拷贝。注意 zero\_grad(set\_to\_none=True) 会丢掉视图，要改用 set\_to\_none=False 或自己清零 buffer。
3. 记录通信时间，与 naive 对比，并用 α-β 预测对照：naive ≈ 张量个数×α + 字节数/β，flat ≈ α + 字节数/β。
4. 记录峰值显存，flat 多一份梯度大小。

**为什么**：把 n 次 α 变成 1 次；版本 B 的零拷贝视图是 M5 的前置。

**验收**：对齐同 M2；naive 与 flat 的差距能用「张量个数×α」解释到数量级。

**takeaway**

- 小消息由延迟主导，把许多小消息合并成大消息是第一性的优化。
- 内存布局：view、stride、contiguous 的关系，以及 set\_to\_none 如何改变 .grad 的身份。
- 以显存换通信效率的取舍。

**与推理框架的联系**：「把许多小块合成一次传输」在推理里到处出现：KV cache 分页块的合并拷贝、MoE all-to-all 前的 token 打包、权重加载时的参数打平。CUDA graph 要求地址固定，所以集合通信的 buffer 必须预分配，这与版本 B 的预分配视图思想一致。

## 6 M4：overlap（约 0.5 到 0.75 天）

**干什么**

1. 对每个 requires\_grad 的参数注册 register\_post\_accumulate\_grad\_hook；hook 里先 p.grad.div\_(world size)，再发 all\_reduce(async\_op=True)，handle 存进列表；finish\_gradient\_synchronization 里逐个 wait，然后才能 optimizer.step()。
2. 用 nsys 抓一段，确认 backward 的 kernel 与 NCCL kernel 在不同 stream 上时间重叠；没有重叠说明哪里串行了，先查清再往下做。
3. 估计 hook 的 CPU 开销：看 hook 触发到 NCCL kernel 启动之间的间隙。
4. 可选 M4b：手动 pinned staging。梯度 copy\_ 到预分配的 pinned 主机 buffer（non\_blocking=True），在 gloo 上做 all\_reduce，再拷回。预期比 NCCL 直通更慢，用来量化「经主存」的代价。主线是 NCCL 直接作用在 GPU 梯度上，因为 SHM 传输自己已经做了主存 staging。

**为什么**：把通信藏进反向。你的机器上通信大于反向，所以收益上限约是 T\_bwd，这个事实要用数据确认。

**验收**：对齐同 M2；nsys 上看到重叠；overlap 步时间与预测公式 T\_fwd + max(T\_bwd, T\_comm) + T\_tail + T\_opt 相符。

**takeaway**

- autograd hook 机制与梯度就绪顺序（大致是层的逆序）。
- stream 与 event 模型：计算 stream 与 NCCL stream 如何靠 event 建立依赖。
- 重叠的上限公式，以及重叠效率小于 1 的原因：NCCL kernel 占用 SM、内存带宽和 PCIe，与计算互相抢资源。
- pinned 内存的语义：pageable 内存上的 non\_blocking 拷贝实际是同步的，所以要异步拷贝就必须 pinned。

**与推理框架的联系**

- 张量并行里把 all-reduce 与计算重叠（通信 kernel 与计算 kernel 分流，或融合 GEMM 加 all-reduce）。
- KV cache 的 CPU 换入换出，以及 prefill/decode 分离部署里的 KV 传输，都依赖 pinned（或已注册）内存加独立拷贝 stream，与计算重叠。
- hook 触发的动态通信与 CUDA graph 不兼容，推理侧要用静态图加预分配 buffer，所以这里学到的是机制，不是照搬的方案。

## 7 M5：bucketed DDP（约 1 天）

**干什么**

1. 分桶：按 module.parameters() 的逆序遍历，累加字节数到 ≥ bucket\_mb×2^20 就封桶；单个超过桶大小的参数独占一桶。
2. 每桶：一块平坦 buffer（沿用 M3 的零拷贝视图）、pending 计数、handle。
3. 每步开始（on\_train\_batch\_start）重置 pending、清零 buffer、清空 handles；hook 里 pending 减到 0 就对整个桶发 all\_reduce(async\_op=True)；finish 里等所有 handle 并除以 world size。
4. 某参数从不产生梯度会让它的桶永不发出：在 finish 里把没发出的桶补发（各 rank 一致，因为没有梯度的参数集合一致）。
5. 扫桶大小：1、10、100、1000 MB，再补 0.25、4、25 MB；每个大小都要先对齐真值再计时；记录实际桶数与桶大小分布。

**为什么**：折中 α 的开销（桶太小）与 overlap 的粒度（桶太大，尾部那个桶无法隐藏）。

**验收**：所有桶大小都与真值对齐；每步时间随桶大小呈 U 形曲线，或能解释为什么不是。

**takeaway**

- 粒度权衡与尾桶暴露：最后一个桶的通信时间 T\_tail 无法被隐藏。
- 为什么逆序：反向按层的逆序产生梯度。
- 对照 PyTorch DDP 的 bucket\_cap\_mb（默认 25 MB）与 Reducer 的实现思路；ZeRO 与 FSDP 的 reduce-scatter、all-gather 也是同样的分桶。

**与推理框架的联系**：同一类「粒度权衡」：chunked prefill 的 chunk 大小（太小开销大、太大延迟高），CUDA graph 只为若干 batch 档位捕获并对请求 padding，continuous batching 的 token 预算。你亲手测出的 U 形曲线，是理解这些参数的直观基础。

## 8 M6：benchmark、nsys 与报告（约 1 到 1.5 天）

**干什么**

1. 测量协议：5 步预热加 10 步测量；预热前先做一次空 all\_reduce 建立通信器；测量前 dist.barrier() 一次，步内不加 barrier；每步结束 cuda.synchronize() 后计时；取各 rank 的最大值；每个配置跑 3 次取中位数。
2. 指标定义（报告里逐个写清）：每步总时间；总通信时间（NCCL kernel 忙时并集，来自 nsys）；暴露通信时间（finish 里真正等待的时间，naive 与 flat 则是整个 all\_reduce 区间）；通信占比 = 暴露通信 ÷ 总时间；峰值显存（max\_memory\_allocated，NCCL 自己的 buffer 另用 nvidia-smi 看）。建议在 5.4 表里加「暴露通信」和「重叠比例」两列。
3. nsys：脚本里在测量步前后调用 cudaProfilerStart/Stop，并用 NVTX 标出前向、反向、优化器、finish；nsys profile 套在 torchrun 外面，再导出 sqlite。
4. 写 scripts/overlap\_from\_nsys.py：读 kernel 表，把名字含 nccl 的区间并集记为通信、其余 kernel 的并集记为计算，输出 NCCL 总时长、与计算重叠的时长、暴露时长、重叠比例。
5. 填 5.2 到 5.6 的表，用 5.6 的理论对照解释差距，回答模板里的四个问题。

**为什么**：把「感觉有 overlap」变成数字。

**验收**：5.2 到 5.6 全部有数据；每个问题至少有一条数据支撑；预测与实测的差距有解释。

**takeaway**

- 性能归因：区分总通信与暴露通信；用 timeline 加数据化（sqlite）的方式，而不是只看截图。
- 误差来源：NCCL kernel 会在 kernel 内等待对端，rank 之间步调偏差会虚增其时长，所以只统计稳态步，并比较两个 rank；热与功耗漂移。
- 报告写法：预测与实测并列，差距归因到具体机制。

**与推理框架的联系**：推理优化的日常就是把一个 decode step 拆开看：CPU launch 间隙多就是 launch-bound，要用 CUDA graph；通信是否暴露决定张量并行是否划算。你现在做的「总时间、暴露时间、重叠比例」三分法，可以原样用于分析 TP 的 all-reduce、KV 传输、专家并行的 all-to-all。

## 9 欠缺项与补强路线

| 欠缺 | 为什么重要 | 怎么补（可操作的小实验） |
| --- | --- | --- |
| 张量并行与推理 all-reduce | 推理多卡的主要并行方式，decode 通信是小消息 | 2 卡实现 Megatron 式列并行加行并行的 MLP 与注意力前向，扫 batch 看 all-reduce 占比；用本机 α、β 建模 TPOT ≈ 权重字节/带宽 + 2L×(α + 消息/β)，判断本机上 TP 是否划算 |
| 自定义 all-reduce 与低延迟通信 | 小消息下 NCCL 的 α 太高 | 读 vLLM 的 custom all-reduce 源码，理解 one-shot 与 two-shot；本机没有 P2P，只能读源码并用 host staging 写一个对照，需要 P2P 或 NVLink 的验证可以短租一台带 NVLink 的机器 |
| CUDA graph 加集合通信 | 推理消除 launch 开销的标配 | 用 torch.cuda.graph 捕获含 all\_reduce 的 decode step，体会静态 buffer 和固定地址的要求 |
| KV cache、paged attention、continuous batching | 推理框架的核心 | 读 vLLM 的 scheduler 与 block manager；写一个最小的分页 KV，并把你现有的 CuTe FlashAttention 改成 decode 的单 query 版本，把 kernel 经验接到推理 |
| FSDP 与 ZeRO | 训练大模型必备，通信模式不同（reduce-scatter 与 all-gather） | 用本阶段的分桶框架实现最小的优化器状态分片，对比通信量 |
| 多机、RDMA、KV 传输（prefill/decode 分离） | 大规模推理部署 | 读 Mooncake 与相关论文；本机没有硬件，用 NCCL 的 socket 传输做慢链路模拟 |
| 梯度或激活的低精度通信 | fp8 all-reduce 是推理与训练的趋势 | 在 M3 的 flat 版本上加 bf16 通信，测数值误差与时间收益 |

**你的机器的特殊性**：消费级双卡没有 P2P 和 NVLink，通信比数据中心慢一个量级（5.2 对约 450 GB/s）。这使它成为一个很好的「慢链路」实验台：你可以用实测的 α、β 回答「在这种机器上，张量并行是否值得」，通常答案是不太值得，更适合单卡加量化或流水并行。这个结论加上数据，是面试时很好的素材。

## 10 交付清单

- 代码：ddp/ 下的 naive、flat、overlap、bucketed，train\_single.py，benchmark\_ddp.py，overlap\_from\_nsys.py。
- 数据：truth 文件、knee 曲线、5.2 到 5.6 的表、nsys 截图与数值。
- 报告：预测与实测对照、四个问题的回答、一页「推理联系与后续补强」。
