
## 5. Part 3：分布式数据并行（DDP）
**目标**：从零实现数据并行，量化通信开销，并验证 overlap 的收益。
### 5.1 里程碑
- [ ] M1：单卡训练脚本作为"真值"
- [ ] M2：naive DDP（逐参数 all_reduce），与单卡逐步对齐权重
- [ ] M3：flat DDP（梯度打平，一次 all_reduce）
- [ ] M4：overlap（`register_post_accumulate_grad_hook` 异步 all_reduce +
      **pinned staging buffer**）
- [ ] M5：bucketed DDP（按 MB 分桶 + overlap）
- [ ] M6：benchmark 与 nsys 验证 overlap
### 5.2 正确性记录
> ⚠️ tied weights（ToyModelWithTiedWeights）：共享参数产生两份梯度，
> 合并顺序必须全 rank 一致，否则 rank divergence 挂死
> （nccl_sweep 事故教训：集合通信绝不放进 `if rank==0`）。
| 版本 | 与单卡 loss 对齐（N 步） | 权重最大误差 | pytest |
|---|---|---|---|
| naive |  |  |  |
| flat |  |  |  |
| overlap |  |  |  |
| bucketed |  |  |  |
### 5.3 实验配置
| 项目 | 取值 |
|---|---|
| world size | 2（NCCL，传输已知 SHM） |
| 模型 | medium / large（二选一或都做） |
| global batch | 8（每卡 4） |
| ctx | 256 |
| dtype | bf16 autocast / fp32 |
| warmup / 测量步数 | 5 / 10 |
### 5.4 各版本对比
| 版本 | 每步总时间 (ms) | 通信时间 (ms) | 通信占比 | 相对 naive 加速 | 峰值显存 (GB) |
|---|---|---|---|---|---|
| 单卡（batch=4，无通信，作为参考） |  | 0 | 0 |  |  |
| naive |  |  |  | 1.00× |  |
| flat |  |  |  |  |  |
| overlap（逐参数） |  |  |  |  |  |
| bucketed（best bucket） |  |  |  |  |  |
通信时间的获取：nsys timeline 统计 NCCL kernel 总时长，或 CUDA event
夹 all_reduce 区间。
### 5.5 Bucket 大小扫描
> **1.3 实测修正预期**：≥1MB 区间 NCCL 带宽平坦无地板 → bucket 下限不是
> NCCL 效率，而是 **CPU launch 开销 + overlap 粒度**。如需 sub-MB 拐点
> 数据支撑下限，跑 `scripts/nccl_knee.py`。
| bucket size (MB) | bucket 数 | 每步时间 (ms) | 通信占比 | overlap 程度（nsys 观察） |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 10 |  |  |  |  |
| 100 |  |  |  |  |
| 1000（≈单桶） |  |  |  |  |
### 5.6 用基线带宽做理论对照
| 项目 | 值 |
|---|---|
| 模型梯度总量（字节） | 参数量 × 4（fp32）或 × 2（bf16） |
| 实测 busbw（来自 1.3） | **5.2 GB/s** |
| 理论最短通信时间 | 梯度量 ÷ busbw（medium 30M→23ms，large 124M→95ms） |
| 实测通信时间 |  |
| 差距原因（实测后勾选） | ① sub-MB 小消息（knee 补测）② launch 开销 ③ SHM 经主机 staging（无 P2P）④ 同步等待 |
**需要回答**：
- naive 与 flat 差距主要来自什么？（逐参数消息是 KB 级，在 1.3 平坦区以下——结合 knee 数据判断是延迟还是 launch 主导）
- 如果没有 P2P / NVLink，overlap 的收益会比 H100 上更大还是更小？
  → **更大**：本机通信占比高一个量级（5.2 vs ~450 GB/s），可隐藏的绝对时间
  更长；上限受无 P2P 限制（staging 经 pinned 主存两段拷贝）。
- 最优 bucket size 受什么影响？（下限：每桶 ≥1MB 保带宽（本机实测平坦区起点）；
  上限：桶够多以保留 overlap 机会）
- 如何在 nsys 上直接看到 overlap？（backward kernel 与 NCCL kernel 在
  不同 stream 上并行，截图标注）