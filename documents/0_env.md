
# 服务器环境使用指南（c220 · CS336 实验机）
> 面向对象：将在本服务器上跑 CS336 Assignment 2 实验的同学
> 机器：`c220`，2× RTX 5070 Ti (16GB)，代码仓库：`/data2/c220/hz/cs336`
---
## TL;DR（只记三条）
1. **作业相关的一切命令都以 `uv run` 开头**，且必须在 `/data2/c220/hz/cs336` 目录下执行
2. 终端提示符里的 `(vllm)` 是另一个 conda 环境（vLLM 推理用），**与作业无关——看到它不用管，更不要在里面装作业依赖**
3. 跑计时/分布式实验前先 `nvidia-smi` 确认卡是空的；程序卡死用 `pkill -f torchrun` 清场
---
## 1. 硬件概况与限制
| 项 | 值 | 对使用的影响 |
|---|---|---|
| GPU | 2× RTX 5070 Ti 16GB（Blackwell, sm_120） | 显存仅 16GB，大模型需降配；需 cu12.8+ 构建的 torch |
| 互联 | PCIe Gen4 x16，两卡经 CPU（PHB），无 NVLink | 卡间通信必须经主机内存中转 |
| P2P | **不可用**（`nvidia-smi topo -p2p` → CNS，芯片组级封锁） | 不要尝试修复，与驱动版本无关 |
| NCCL | 默认走 SHM（主机内存）传输；2 进程 all_reduce 实测 busbw ≈ **5.2 GB/s** | 通信开销比 NVLink 机器大一个量级，DDP 实验设计时要考虑 |
| 共享 | 可能与他人共用 | 计时数据只有在卡空闲时才可信 |
---
## 2. 两套 Python 环境：谁是谁
服务器上有**两套互不相干**的 Python 环境：
| | conda 环境 `vllm` | uv 项目环境（作业用） |
|---|---|---|
| 位置 | conda 默认目录 | `/data2/c220/hz/cs336/.venv` |
| Python | 独立 | 3.13.17（uv 自动管理，无需系统装） |
| 内容 | vLLM 推理框架及依赖 | torch (cu13)、triton、pytest、作业包（editable 安装） |
| 用途 | 与作业无关的推理/服务任务 | CS336 A2 全部实验 |
| 进入方式 | `conda activate vllm` | **无需手动激活**，`uv run` 自动使用 |
**黄金法则**：
- 做作业 → 只用 `uv run ...`，**永远不用 pip install**
- 用 vLLM → `conda activate vllm` 后正常操作，装包就在那个环境里装
- 两者**不要交叉装包**
> ⚠️ 最常见的坑：新开终端提示符自带 `(vllm)`，新手顺手在这里 `pip install` 作业依赖
> → 装错环境。作业不受影响（uv 有自己的 lock），但会污染 vllm 环境。
自查"我现在用的是哪个 python"：
```bash
which python                                      # (vllm) 激活时 → conda 的
cd /data2/c220/hz/cs336 && uv run which python    # → .../cs336/.venv/bin/python
```
---
## 3. uv 速成（只需四个命令）
uv 是基于锁文件（`uv.lock`）的包管理器：环境完全由 `pyproject.toml + uv.lock`
决定，任何人 pull 之后一条命令即可还原一致环境。
| 命令 | 干什么 | 什么时候用 |
|---|---|---|
| `uv run <命令>` | 在项目 .venv 里执行任意命令；执行前自动核对环境与 lock 一致 | 日常 90% 场景 |
| `uv add <包>` | 添加依赖（写 pyproject + lock + 装进 venv） | 需要新库时；**替代 pip install** |
| `uv sync` | 把 .venv 同步到 lock 状态 | `git pull` 拉到依赖变更后 |
| `uvx <工具>` | 一次性运行工具，不污染任何环境 | 调试小工具（如 py-spy） |
要点：
- `uv run` 会把项目 .venv 放到 PATH 最前面，所以链式调用也正确：
  ```bash
  uv run nsys profile -o out --trace=cuda,nvtx,osrt python scripts/benchmark.py
  # 这里的 python 就是 .venv 的 python
  ```
- 一律从**仓库根目录**执行（uv 向上查找 pyproject.toml，别处执行会解析错项目）
- 不必也不建议 `source .venv/bin/activate`（容易忘记自己激活过；`uv run` 已覆盖全部需求）
- 若 `uv` 不在 PATH：它在 `~/.local/bin/uv`
---
## 4. vLLM 环境（其他工作用）
- 用途：与作业无关的推理/服务任务（如本地起 OpenAI 兼容 server）
- 进入/退出：`conda activate vllm` / `conda deactivate`
- 验证：`python -c "import vllm; print(vllm.__version__)"`
- 典型用法：`vllm serve <模型路径> --port 8000`（按实际任务替换）
> ⚠️ **与实验的最大冲突是显存**：vLLM 服务会预分配大量显存（KV cache）。
> 做任何 benchmark / 训练前，`nvidia-smi` 确认没有 vLLM server 在跑。
---
## 5. 作业日常命令（复制即用）
### 5.1 上手自检（新登录 2 分钟）
```bash
ssh <user>@c220
cd /data2/c220/hz/cs336
# ① 栈是否完整
uv run python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.device_count())"
# 预期：2.x cu13 3.x 2
# ② 测试能否收集
uv run pytest --collect-only -q | tail -2
# 预期：共 14 个（attention 6 / ddp 2 / fsdp 4 / sharded-optimizer 2）
```
> adapters 未实现前，测试失败并抛 `NotImplementedError` 是**正常状态**，不是环境问题。
### 5.2 跑测试
```bash
uv run pytest -q                                      # 全部
uv run pytest tests/test_attention.py -q              # 只跑 attention
uv run pytest tests/test_attention.py -k pytorch -x   # 关键字过滤 + 首错即停
```
### 5.3 单卡脚本
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark.py --model small --ctx 512
```
### 5.4 双卡分布式（torchrun）
```bash
# 常规跑法（NCCL 默认 SHM 传输）
uv run torchrun --nproc-per-node 2 scripts/nccl_sweep.py
# 对照组：关 SHM 强制回落 socket
NCCL_SHM_DISABLE=1 uv run torchrun --nproc-per-node 2 scripts/nccl_sweep.py 64 256
# 排查 NCCL 传输选择（日志多，仅排查时开）
NCCL_DEBUG=INFO uv run torchrun --nproc-per-node 2 scripts/nccl_probe.py 2>&1 | grep -E "via|algbw|busbw"
```
环境变量速查：
| 变量 | 作用 | 默认建议 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 限定可见 GPU | 单卡任务设 `0` |
| `NCCL_DEBUG=INFO` | 打印 NCCL 传输细节 | 平时不开 |
| `NCCL_SHM_DISABLE=1` | 强制 NCCL 回落 socket（对照实验） | 默认不设 |
| `NCCL_SOCKET_IFNAME=lo` | socket 回落时绑 loopback | 仅 SHM off 且卡住时用 |
> torchrun 打印的 `OMP_NUM_THREADS=1` warning 是正常提示，忽略。
### 5.5 性能分析
```bash
nsys --version   # 先确认可用
uv run nsys profile -o small512 --trace=cuda,nvtx,osrt python scripts/benchmark.py --model small --ctx 512
# 产物 .nsys-rep 建议下载到本地 GUI 打开；服务器上也可用 nsys stats 查汇总表
```
---
## 6. 排错与共享纪律（本机实测验证过的教训）
1. **无声挂死**（打印几行后不再动）：九成是各进程执行路径不一致
   （rank divergence，典型：集合通信被放进 `if rank == 0`）。排查：
   ```bash
   uvx py-spy dump --pid <某个rank的PID>   # 看每个进程各卡在哪一行（报权限错就加 sudo）
   ```
2. **清理残留进程**（重跑前必做）：
   ```bash
   pkill -f torchrun; sleep 2; nvidia-smi   # 确认两卡显存归零
   ```
   NCCL 自带 10 分钟超时 watchdog，等它报错也行，但手动清场更快。
3. **计时实验前**：`nvidia-smi` 确认无他人任务、无 vLLM 服务占显存；
   严格实验前后各记一次 `nvidia-smi -q -d CLOCK,POWER`。
4. **长任务**：用 tmux——`tmux new -s exp`；断线后 `tmux attach -t exp`。
5. **拉代码后**：`git pull && uv sync`，再跑一遍 5.1 自检。
6. **nvidia-smi 之外**：单卡任务一律 `CUDA_VISIBLE_DEVICES=0`（或 1）明确占卡，
   避免与他人任务挤在同一张卡上。
---
## 7. 快速参考卡
| 想做什么 | 命令 |
|---|---|
| 验证环境 | `cd /data2/c220/hz/cs336 && uv run pytest --collect-only -q \| tail -2` |
| 跑全部测试 | `uv run pytest -q` |
| 跑单个测试 | `uv run pytest tests/test_attention.py -k <名字> -x` |
| 单卡脚本 | `CUDA_VISIBLE_DEVICES=0 uv run python scripts/<x>.py` |
| 双卡分布式 | `uv run torchrun --nproc-per-node 2 scripts/<x>.py` |
| NCCL 排查 | `NCCL_DEBUG=INFO uv run torchrun ... 2>&1 \| grep ...` |
| 加依赖 | `uv add <包>`（不要 pip） |
| 同步环境 | `uv sync`（pull 之后） |
| 临时工具 | `uvx py-spy dump --pid <pid>` |
| 杀残留 | `pkill -f torchrun; nvidia-smi` |
| 看卡 | `nvidia-smi`（或 `watch -n1 nvidia-smi`） |
| vLLM 工作 | `conda activate vllm` → 正常用 → 用完 deactivate |

