# nsys + NVTX
目标：搞懂三件事
1. NVTX 标注到底标的是什么
2. nsys 拿到标注后怎么分析，三种"时间"有什么区别
3. 用的时候有哪些坑

---

## 1. 先建立一个心智模型：CPU 发射，GPU 执行

PyTorch 里写 `y = x @ w`，实际发生的是：

```
CPU 线程:  [发射 matmul kernel] → 立刻返回 → [发射下一个] → ...
GPU 流 :            [matmul 执行........] [下一个执行....]
```

- CPU 只是把 kernel "投递"到 GPU 的队列里，**不等它跑完**（异步）。
- GPU 按队列顺序执行，所以 GPU 上的工作通常**晚于**CPU 的调用发生。

这是理解 NVTX 的全部前提。

---

## 2. NVTX 是什么

NVTX = 在 **CPU 侧**给一段代码打上"带名字的区间"（range）。nsys 把它画在时间线上。

关键点：**NVTX 区间本身只记录 CPU 上的起止时间**。它不知道 GPU 干了什么。

那 nsys 怎么知道"这个区间对应哪些 GPU 工作"？靠**关联**：

1. 每个 kernel 都是被某个 CUDA API 调用（如 `cudaLaunchKernel`）发射的；
2. nsys 记录了这次调用发生在**哪个线程、哪个时刻**，那一刻该线程处在哪些 NVTX 区间里；
3. 于是 nsys 把 kernel **"投影"回发射它时所在的 NVTX 区间**。

> 一句话：**kernel 属于哪个区间，看的是"谁在哪个区间里发射了它"，不是"它在 GPU 上什么时候跑"。**

这个规则决定了后面所有注意事项。

---

## 3. 一个具体例子：三种"时间"

假设你标注了区间 `A`，里面发射了 3 个 kernel：

```
CPU 线程:   |--- 区间 A (CPU) ---|
             launch1 launch2 launch3          ← 很快发完，CPU 区间很短
GPU 流 :        [k1]   [k2]        [k3]       ← GPU 慢慢跑，k2 和 k3 之间有空隙
                ^                  ^
                A 的 GPU 投影起点   A 的 GPU 投影终点
```

于是同一个区间 A 有**三个不同的数**：

| 名称 | 含义 | 例子里是什么 | nsys 报告 |
|---|---|---|---|
| **CPU 时间** | 区间在 CPU 上持续多久 | 很短（只是发射） | `nvtx_sum` |
| **GPU 投影时间跨度** | 从第一个 kernel 开始到最后一个 kernel 结束（**含空隙**） | k1 开始 → k3 结束 | `nvtx_gpu_proj_sum` |
| **kernel 耗时之和** | 区间内每个 kernel 的 duration 相加（**不含空隙**） | k1+k2+k3 | `nvtx_kern_sum` |

怎么选：

- 想知道"这块代码占 GPU 多少**工作量**/占比" → 用 **kernel 耗时之和**（`nvtx_kern_sum`）。
- 想知道"这块代码在 GPU 上**占了多长的墙钟时间**" → 用**投影跨度**。
- 跨度 − 耗时之和 = 区间内 GPU 的**空闲空隙**，大说明发射受限。
- CPU 时间**不要**当作 GPU 耗时，除非你在区间末尾做了同步（见 §6）。

---

## 4. NVTX API

```python
import torch.cuda.nvtx as nvtx
```

### 4.1 三种用法

```python
# (1) 上下文管理器：推荐，自动配对，不会忘记 pop
with nvtx.range("forward"):
    out = model(x)

# (2) 手动 push/pop：跨函数/跨 hook 时用，必须成对
nvtx.range_push("attn")
...
nvtx.range_pop()

# (3) 瞬时标记：时间线上一个点，没有时长
nvtx.mark("step_begin")
```

### 4.2 push/pop 的规则

- 区间是**每个线程一个栈**：`push` 压栈，`pop` 弹出**最近一次**push 的区间。
- push 和 pop 必须在**同一个线程**、**顺序嵌套**，否则区间错乱。
- 嵌套就是层级：

```python
with nvtx.range("step"):
    with nvtx.range("forward"):
        ...
    with nvtx.range("backward"):
        ...
```

时间线上 `forward`、`backward` 会画在 `step` 下面一层。

### 4.3 用 hook 给模块自动打标（不改模型源码）

```python
def add_nvtx_hooks(model, names):
    handles = []
    for mod in model.modules():
        k = type(mod).__name__
        if k in names:
            handles.append(mod.register_forward_pre_hook(
                lambda m, a, k=k: nvtx.range_push(k)))     # forward 开始前 push
            handles.append(mod.register_forward_hook(
                lambda m, a, o: nvtx.range_pop()))         # forward 结束后 pop
    return handles
```

注意：
- lambda 里 `k=k` 是为了**固定当前循环的值**，否则所有 hook 都用最后一个 k（经典闭包坑）。
- forward hook **只管 forward**，反向不会被覆盖。
- 用完可以 `for h in handles: h.remove()` 移除。

### 4.4 替换函数来标注内部段落

想标注 attention 内部的 `scores / softmax / av`，没法用 hook（它们是函数里的几行，不是模块）。办法是写一个带标注的版本，替换原函数：

```python
def annotated_sdpa(Q, K, V, mask=None):
    with nvtx.range("attn.scores"):
        ...
    with nvtx.range("attn.softmax"):
        ...
    with nvtx.range("attn.av"):
        ...
```

---

## 5. 命名规范（影响汇总结果是否可读）

nsys **按名字汇总**。所以：

| 做法 | 原因 |
|---|---|
| 用**静态字符串** | 名字里带 step 编号（`"step_17"`）会让每个 step 变成独立一行，无法汇总 |
| **同类对象用同一个名字**（所有层都叫 `block`） | 汇总后"调用次数"= 层数，可自检（每个 forward 应有 L 次） |
| 层级用点号：`attn.scores` | 便于阅读与过滤 |
| 需要看单层时才临时加编号 | 不作为默认 |

**调用次数自检是最实用的验证手段**：
- 设模型有 L 层，则 `block` 每个 forward 应出现 L 次；
- 每层有 2 个 RMSNorm，加最后 1 个，则 `RMSNorm` 应出现 2L+1 次；
- 次数对不上，说明标注漏了或 push/pop 不配对。

---

## 6. 为什么要 `torch.cuda.synchronize()`

回到 §3 的例子：区间 A 的 CPU 时间很短，GPU 还在后面跑。

如果在区间末尾加同步：

```python
with nvtx.range("forward"):
    out = model(x)
    torch.cuda.synchronize()   # CPU 在这里等 GPU 全部做完
```

效果：
- CPU 区间被**拉长到 GPU 真正做完**，CPU 时间 ≈ 该阶段的 GPU 耗时；
- GPU 上的工作被**限制在这个区间的时间窗内**，阶段之间不再重叠，边界清晰；
- 代价：**破坏了真实的 CPU/GPU 流水线重叠**，测出来的时间不等于真实训练速度。

所以：
- 同步**只在 profile 模式里**用，用来做"构成分析"；
- **测速度**用不带 profiler 的 benchmark。

> 注意：投影本身（§2）靠的是"发射时刻"，**不依赖**同步。同步的作用是让 CPU 区间有意义、让阶段边界在时间轴上分开（对"时间窗归因"尤其重要）。

---

## 7. nsys 的工作流程

### 7.1 采集

```bash
nsys profile \
  -o out_name --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python script.py
```

| 选项 | 作用 |
|---|---|
| `--trace=cuda,nvtx,osrt` | 抓 CUDA API 与 kernel、NVTX 区间、OS 运行时 |
| `--capture-range=cudaProfilerApi` | **只在脚本调用 `profiler.start()` 之后才开始抓** |
| `--capture-range-end=stop` | 在 `profiler.stop()` 时结束抓取 |

脚本里配合：

```python
for _ in range(warmup): step()          # 预热：不被抓
torch.cuda.synchronize()

torch.cuda.profiler.start()             # 从这里开始抓
for _ in range(3): step()               # 3 步足够，多了文件很大
torch.cuda.synchronize()
torch.cuda.profiler.stop()
```

**预热为什么不能被抓**：第一次运行有 cuBLAS 初始化、kernel 懒加载，不是稳态。

### 7.2 分析：命令行（nsys stats）

```bash
nsys stats --report cuda_gpu_kern_sum --format csv --output prefix file.nsys-rep
```

报告名随 nsys 版本变化，先 `nsys stats --help-reports` 看本机有哪些。常用的：

| 想看什么 | 报告 |
|---|---|
| 所有 kernel 的汇总（次数、总时间、平均） | `cuda_gpu_kern_sum` |
| 每个 NVTX 区间的 **CPU 时间** | `nvtx_sum` |
| 每个 NVTX 区间的 **GPU 投影跨度** | `nvtx_gpu_proj_sum` |
| **按 NVTX 区间汇总 kernel**（耗时之和） | `nvtx_kern_sum` |
| CPU 侧 CUDA API 耗时（发射开销） | `cuda_api_sum` |
| 逐次 kernel 明细 | `cuda_gpu_trace` |

CSV 的文件名和列名以实际输出为准：用 `glob` 找文件，用 `df.columns` 看列，别硬编码。

### 7.3 分析：图形界面（Nsight Systems GUI）

把 `.nsys-rep` 下载到本地打开（GUI 版本不低于生成它的 nsys）。

| 行 | 看什么 |
|---|---|
| NVTX | 区间层级与宽度，先定位 `forward`/`backward` |
| CUDA HW → Kernels | kernel 是否首尾相接；**空隙 = 发射受限** |
| 主线程 → CUDA API | kernel 对应的发射调用；点 kernel 能跳转 |
| 其他线程 | autograd 反向线程（见 §8） |

---

## 8. backward 的归因

这是 nsys + NVTX 里最容易踩、也最容易"看起来没问题"的一个坑。先说结论，再讲原因、验证和解决办法。

**结论**：你在主线程里写 `with nvtx.range("backward"): loss.backward()`，nsys 很可能**不会**把反向的 kernel 算到 `backward` 区间里。反向 kernel 在报告里要么缺失，要么不属于任何区间。

（这是基于机制的判断，不是对某个版本的实测，所以 8.3 先教你验证。）

---

### 8.1 你会看到什么现象

| 报告 | 异常表现 |
|---|---|
| `nvtx_kern_sum` | `backward` 区间的 kernel 数很少，甚至为 0；`forward` 却有几百个 |
| `nvtx_gpu_proj_sum` | `backward` 的 GPU 投影跨度很短或为空 |
| `nvtx_sum` | `backward` 的 **CPU 时间**却很长（主线程在等） |
| 各区间 kernel 数相加 | 明显**小于** `cuda_gpu_kern_sum` 里的 kernel 总数 |
| GUI | NVTX 行里 `backward` 区间很宽，但它下面对应的 GPU kernel 行是空的；而 GPU 上明明有大段反向 kernel |

最典型的"错觉"：CPU 时间那一栏看起来 backward 很耗时，但 GPU 投影里没有它，两个数对不上，让人怀疑是不是自己标错了。其实没标错，是线程问题。

---

### 8.2 为什么：三个事实叠加

**事实 1：NVTX 的区间栈是"每个线程一个"。**
`range_push` 只压到**当前线程**的栈上。主线程 push 了 `backward`，别的线程不知道。

**事实 2：PyTorch 的反向在另一个线程上发射 kernel。**
`loss.backward()` 把任务交给 autograd 引擎。引擎为每个设备（比如一张 GPU）有一个**工作线程**，真正去调用 CUDA 发射反向 kernel 的是这个工作线程。主线程只是阻塞，等它完成。

**事实 3：nsys 的 NVTX→GPU 投影，按"发射线程"来匹配区间。**
nsys 找到每个 kernel 的发射 API 调用，看这次调用**发生在哪个线程、那一刻该线程在哪些区间里**。

三者合起来：

```
主线程:        |------------------------ step -------------------------|
               |-- forward --|--- backward (阻塞等待) ---|-- optimizer --|
                launch fwd kernels   不发射任何东西        launch opt kernels

autograd 线程:                  [launch bwd kernel][launch bwd kernel]...
                                 ↑ 这个线程的 NVTX 栈是空的

GPU 流:         [fwd kernels]    [bwd kernels ..................]  [opt kernels]
```

反向 kernel 的发射线程是 autograd 线程，那个线程里**没有任何 NVTX 区间**。所以：

- 它们不属于 `backward`；
- 也不属于 `step`（`step` 也是主线程打的）；
- 它们成了"区间外的 kernel"，只会出现在不分区间的汇总（`cuda_gpu_kern_sum`）里。

**连带影响**：如果你用"各区间 kernel 之和 ÷ step 时间"去算 GPU 利用率，反向的部分被漏掉了，利用率会被**严重低估**，从而误判成"发射受限"。算利用率时，分子请用 `cuda_gpu_kern_sum` 的**总时间**（不分区间），不要用 NVTX 区间汇总。

---

### 8.3 先验证：三个检查

不要想当然，先确认你的环境里确实有这个问题：

1. **看 kernel 数**：`nvtx_kern_sum` 里 `backward` 的 kernel 数，对比 `forward`。一次正常的反向，kernel 数通常是前向的 **1.5～2.5 倍**（量级判断，不是精确值）。远小于这个，就是被漏掉了。
2. **看总数是否对得上**：同一个 step 内，
   `forward + backward + optimizer 区间的 kernel 数之和` 应当 ≈ `cuda_gpu_kern_sum` 里该 step 的 kernel 总数。差的那部分就是没被归到任何区间的反向 kernel。
3. **看 GUI**：点一个反向的 kernel（比如名字里有 `dgrad`/`wgrad`，或者与前向 GEMM 对称的那个），在"主线程 → CUDA API"里看不到它的发射调用；它的发射调用在**另一个线程**的行里。

如果三项都对得上（backward 有足够多的 kernel），说明你的版本/用法没问题，不用处理。

---

### 8.4 方案 A：时间窗归因（最稳妥，推荐）

**思路**：不依赖"线程投影"，改用**时间**。

- `backward` 区间的 CPU 起止时间 `[t0, t1]` 是可靠的（主线程确实在这段时间里等反向）；
- 只要保证"这段时间里 GPU 只在做反向"，就能把"**GPU 上起始时间落在 `[t0, t1]` 内的所有 kernel**"归给 backward。

**前提：每个阶段末尾必须 `torch.cuda.synchronize()`。** 为什么：

```
没有同步：  forward 区间结束时，GPU 还在跑 forward 的尾巴
            → 这些 kernel 的开始时间可能落进 backward 的窗口，被错归

有同步：    forward 区间在 GPU 做完后才结束
            → 窗口之间 GPU 是空闲的，边界干净
```

**步骤**：

1. 采集（照 12.6 / 第 7 节，3 步，每阶段末尾同步）。
2. 导出 sqlite（`nsys stats` 本身也会在同目录生成 `.sqlite`，二选一）：

```bash
nsys export --type sqlite --force-overwrite true \
    -o results/nsys/small_ctx512_fp32.sqlite results/nsys/small_ctx512_fp32.nsys-rep
```

3. 用 pandas 取"区间"与"kernel"两张表，按时间窗匹配：

```python
import sqlite3, pandas as pd
con = sqlite3.connect("results/nsys/small_ctx512_fp32.sqlite")

# 先确认表名（不同版本可能有差异）
# print([r[0] for r in con.execute("select name from sqlite_master where type='table'")])

# NVTX 区间：起止时间与名字（名字可能在 text 列，也可能在 StringIds 表里）
rng = pd.read_sql("""
    SELECT e.start AS t0, e.end AS t1, COALESCE(e.text, s.value) AS name
    FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
    WHERE e.end IS NOT NULL
""", con)

# GPU kernel：起止时间与名字
ker = pd.read_sql("""
    SELECT k.start AS ks, k.end AS ke, s.value AS name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
""", con)
ker["dur"] = ker.ke - ker.ks

def kernels_in(phase):
    """返回 phase 区间时间窗内（按 kernel 起始时间）的所有 kernel，标注第几个 step"""
    parts = []
    for i, r in enumerate(rng[rng.name == phase].sort_values("t0").itertuples()):
        sub = ker[(ker.ks >= r.t0) & (ker.ks < r.t1)].copy()
        sub["step"] = i
        parts.append(sub)
    return pd.concat(parts)

fwd, bwd, opt = (kernels_in(p) for p in ("forward", "backward", "optimizer"))
print(len(fwd), len(bwd), len(opt))        # 每个都应是 3 个 step 的总和
```

4. 自检：`len(fwd)+len(bwd)+len(opt)` 应 ≈ 捕获范围内的 kernel 总数（`len(ker)`）；`len(bwd)` 约为 `len(fwd)` 的 2 倍量级。
5. 之后套用第 12.7 的类别表（`cat()` 函数）对 `bwd` 算占比，即可回答"某 kernel 在 forward 与 forward+backward 中的占比"。

**注意**：

- `NVTX_EVENTS`、`CUPTI_ACTIVITY_KIND_KERNEL`、`StringIds` 以及列名是按我对 nsys 导出 schema 的印象写的，**以你本机为准**（`.schema 表名` 或 `pd.read_sql("select * from ... limit 1")` 看列）。
- 时间戳：NVTX 是 CPU 时钟、kernel 是 GPU 时钟，nsys 导出时已对齐到同一时间线，可以直接比较。
- 如果脚本里还有梯度裁剪、`zero_grad` 等，要么放进 `optimizer` 阶段，要么给它单独一个阶段，别让它们的 kernel 落进 backward 窗口。
- 窗口按 **kernel 起始时间** 判定，不要用结束时间。

---

### 8.5 方案 B：`emit_nvtx`（让 PyTorch 自己给反向打区间）

**思路**：让 autograd 引擎在**它自己的线程里**，为每个算子自动打 NVTX 区间。区间和 kernel 在同一线程，投影自然正确。

```python
with torch.autograd.profiler.emit_nvtx():
    # 整个要 profile 的区域，放在 profiler.start() 之后
    loss = fwd_and_loss()
    loss.backward()
    opt_step_and_zero()
```

效果：

- 前向算子：区间名形如 `aten::mm`、`aten::softmax`；
- 反向算子：区间名形如 `MmBackward0`、`SoftmaxBackward0`，位于 autograd 线程；
- `nvtx_kern_sum` 里能直接看到每个反向算子对应多少 kernel、多少时间。

**优点**：

- 不用自己算时间窗；
- 能得到**算子级**的反向拆解（哪个 Backward 节点最贵），这是方案 A 做不到的。

**缺点与注意**：

- 区间名可能带序号或形状（取决于版本和 `record_shapes` 参数），会让"同类算子"散成很多行，违背命名规范。汇总时要用正则去掉序号再聚合。
- 得到的是**算子级**区间，**不是**你的 `backward` 整体区间。要得到"backward 总和"，需要把所有名字含 `Backward` 的区间加起来（以实测名字为准）。
- 额外开销比普通 NVTX 大，只在专门的 profile 运行里开。
- 它与你自己打的 `forward`/`backward`/`step` 区间可以并存（自己的区间在主线程，它们在各自线程）。

---

### 8.6 方案 C：在 autograd 线程里自己打区间（hook，需自行验证）

**思路**：用 autograd 的 hook 在**反向开始时 push、结束时 pop**，这两个 hook 本身就是在 autograd 线程里执行的，所以区间栈就在正确的线程上。

```python
def attach_backward_range(logits, first_param):
    # logits 的梯度是反向最先算出来的：在它的 hook 里 push
    logits.register_hook(lambda g: nvtx.range_push("backward"))
    # 最先用到的参数（如 embedding 权重）梯度是最后累加的：在它累加完成后 pop
    first_param.register_post_accumulate_grad_hook(lambda p: nvtx.range_pop())
```

要点：

- 每个 step 要重新对新的 `logits` 注册（它每次前向都是新张量）；参数上的 hook 只需注册一次，但每个 step 都会触发一次 pop，所以每个 step 必须有对应的 push，**数量要配对**。
- "最后一个梯度累加"的假设成立与否取决于模型结构（如是否共享权重）。**必须用 8.3 的计数检查来验证**，不要直接相信。
- `register_post_accumulate_grad_hook` 需要较新的 PyTorch（约 2.1 以上）。

这个方案比较 hacky，适合你既想要一个干净的 `backward` 整体区间、又不想自己算时间窗时用。拿不准就用方案 A。

---

### 8.7 方案 D：nsys 新版本的 PyTorch 选项

较新的 nsys 提供 PyTorch 相关的采集选项（形如 `--pytorch`，具体取值因版本而异）。**先 `nsys profile --help` 看你的版本有没有、取值是什么**，不确定就不要依赖。

---

### 8.8 怎么选

| 你要回答的问题 | 推荐 |
|---|---|
| 完整训练步里各类 kernel 的占比（如 matmul 占多少） | **不需要归因**：直接用 `cuda_gpu_kern_sum` 的总表，它不分区间，包含 backward |
| 同一个 kernel 在 forward 与 forward+backward 中的占比 | 方案 A；或者**更简单**：分别用 `--mode fwd` 和 `--mode fwdbwd` 各跑一次 profile，两份总表直接对比，完全不需要归因 |
| backward 整体花多久、占多少 kernel | 方案 A |
| 反向里哪个算子最贵 | 方案 B（`emit_nvtx`） |
| 想要时间线上一个干净的 `backward` 区间 | 方案 C，并用计数验证 |

对这次作业，**方案 A（或两次分开 profile）已经足够**。

---

### 8.9 常见错误

| 错误 | 后果 |
|---|---|
| 没验证就默认 `backward` 区间里的 kernel 是完整的 | 反向被漏算，占比和利用率全部失真 |
| 用区间汇总的 kernel 之和去算 GPU 利用率 | 漏掉反向，误判成发射受限；分子应用 `cuda_gpu_kern_sum` 总时间 |
| 方案 A 里阶段末尾没有同步 | 前一阶段的尾巴被归到下一阶段 |
| 方案 A 用 kernel 的**结束时间**判断归属 | 跨窗口的 kernel 被错归 |
| 方案 B 里把区间名直接当静态名汇总 | 带序号/形状，散成大量行 |
| 方案 C 里 push 与 pop 数量不配对 | 栈错乱，后面的区间层级全错 |
| optimizer 的 kernel（含梯度裁剪）落进 backward 窗口 | backward 占比偏高 |

---

## 9. 开销与精度

- **NVTX 本身**：每个区间约微秒级 CPU 开销。标得越细 CPU 越慢，对"小模型、短序列"（本来就发射受限）影响最大。
- **nsys 本身**：有采集开销。
- **每阶段加同步**：破坏重叠。

结论：**nsys 的绝对时间不可信，只看构成（占比、次数、空隙）。** 绝对时间用不带 profiler 的 benchmark。粗粒度标注用于计时，细粒度标注只在专门的 profile 运行里开。

---

## 10. 怎么读结果：两个核心判断

### 10.1 占比：时间花在哪

- 按 kernel 类别汇总 `cuda_gpu_kern_sum` 的 Total Time：看 GEMM / softmax / 逐元素 / 归约各占多少。
- 按区间汇总 `nvtx_kern_sum`：看 attention / FFN / norm 各占多少。
- 为避免 `other` 类吞掉占比：先看最大的几个 `other` kernel 名，补全分类规则。

### 10.2 GPU 利用率：瓶颈在 GPU 还是 CPU

```
利用率 = kernel 总时间 ÷ 步的实际时间
```

| 比值 | 含义 | 下一步 |
|---|---|---|
| 接近 1 | GPU 受限：GPU 一直在忙 | 优化 kernel 本身（此时才值得用 ncu） |
| 明显低于 1（如 < 0.6） | **发射受限**：GPU 在等 CPU，时间线上 kernel 之间有空隙 | 减少 kernel 数：融合、`torch.compile`、CUDA graph；ncu 帮不上忙 |

这就是为什么 nsys 先于 ncu：它先告诉你"该不该去看单个 kernel"。

---

## 11. 一个可以马上跑的最小例子（先用它搞懂再上大模型）

```python
import torch, torch.cuda.nvtx as nvtx

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")

def step():
    with nvtx.range("matmul"):
        y = x @ w
    with nvtx.range("softmax"):
        y = torch.softmax(y, dim=-1)
    with nvtx.range("relu"):
        y = torch.relu(y)
    torch.cuda.synchronize()

for _ in range(5): step()                  # 预热
torch.cuda.synchronize()
torch.cuda.profiler.start()
for _ in range(3): step()
torch.cuda.profiler.stop()
```

```bash
nsys profile -o toy --force-overwrite=true --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop python toy.py
nsys stats --report nvtx_sum --report nvtx_kern_sum --report cuda_gpu_kern_sum toy.nsys-rep
```

应该看到（数值以实测为准）：
- 每个区间名（`matmul`/`softmax`/`relu`）的**实例数 = 3**（3 步）；
- `nvtx_kern_sum` 里 `matmul` 区间对应 gemm 类 kernel，`softmax` 区间对应 softmax kernel；
- 然后试着**去掉区间内的同步/改变区间位置**，对比 `nvtx_sum` 的 CPU 时间与 `nvtx_kern_sum` 的 kernel 时间，直观感受 §3 的三种时间。

建议先在这个玩具上确认你看懂了每份报告，再放到真实训练脚本里。

---

## 12. 检查清单

- [ ] 区间名是静态字符串，同类用同名
- [ ] push/pop 成对、同线程；优先用 `with`
- [ ] 预热在 `profiler.start()` 之前，nsys 使用 capture-range
- [ ] profile 模式下阶段末尾 `synchronize`；benchmark 里不加
- [ ] 调用次数自检（`block`=L、`RMSNorm`=2L+1）
- [ ] 区分三种时间：CPU 时间 / GPU 投影跨度 / kernel 耗时之和
- [ ] 检查 `backward` 是否归因为空，必要时用时间窗法或 `emit_nvtx`
- [ ] 只用 nsys 看构成，不看绝对时间；先算 GPU 利用率再决定要不要用 ncu
- [ ] 报告名、CSV 列名以本机实际输出为准