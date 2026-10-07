# SimAI 构建选项

> [English Version](../../configuration/build-options.md)

## 构建模式

| 模式 | 命令 | 输出二进制 | 说明 |
|---|---|---|---|
| analytical | `./scripts/build.sh -c analytical` | `bin/SimAI_analytical` | 快速 busbw 仿真（无网络建模） |
| ns3 | `./scripts/build.sh -c ns3` | `bin/SimAI_simulator` | 完整 ns3 网络仿真（高保真度） |
| phy | `./scripts/build.sh -c phy` | `bin/SimAI_phynet` | 物理 RDMA 流量生成（需要 IB 硬件） |

## 清理构建

```bash
./scripts/build.sh -l ns3         # 清理 ns3 构建产物
./scripts/build.sh -l analytical  # 清理 analytical 构建
./scripts/build.sh -l phy         # 清理 physical 构建
```

## SimAI_simulator 参数（ns3 模式）

| 参数 | 长格式 | 说明 | 默认值 |
|---|---|---|---|
| `-t` | `--thread` | 多线程加速的线程数 | 1 |
| `-w` | `--workload` | Workload 文件路径 | 必填 |
| `-n` | `--network-topo` | 网络拓扑文件路径 | 必填 |
| `-c` | `--config` | 仿真配置文件路径 | 必填 |
| `-j` | `--jobs` | 多任务 CSV manifest | 可选 |
| `-p` | `--policy` | 多任务策略：`fcfs`、`first-fit`、`best-fit`、`fgd` 或 `crux` | `first-fit` |
| `-m` | `--gpu-memory` | 每张 GPU 显存容量（字节），`0` 表示未指定 | `0` |
| `-H` | `--host-memory` | 每张 GPU 对应主机内存容量（字节），`0` 表示未指定 | `0` |
| `-R` | `--gpu-resources` | 每张 GPU 资源 CSV：`gpu_id,gpu_memory_bytes,host_memory_bytes` | 可选 |
| `-L` | `--scheduler-log` | 调度器 CSV 日志路径 | `./SimAI_scheduler.csv` |

### 多任务调度

使用 `-j` 传入 CSV job manifest，在同一 ns3 仿真中运行独立的 AICB/SimAI workload。每行格式为 `job_id,workload,gpus,submit_time_ns[,name[,gpu_memory_bytes[,host_memory_bytes[,communication_bytes]]]]`，资源字段按每张 GPU 计算；设置 `communication_bytes` 后，ns-3 会在任务启动时生成对应的 trace workload。服务器规模的任务优先使用完整物理 GPU 服务器，但服务器编号可以不连续。

#### v2026 GPU trace 适配

`example/cluster-trace-v2026-spot-gpu` 可通过调度层适配器转换为 FCFS manifest。适配器只读取 GPU 字段，忽略 CPU；每个 trace 节点固定按 8 张 GPU 计算，但任务 GPU 数严格保留为 `ceil(gpu_request * worker_num)`，不向上取整到 8 的倍数，`duration` 直接写入任务 workload 的 `ALLREDUCE` 通信量。提交时间默认按 trace 秒转换为 ns，`--arrival-scale` 可整体压缩或放大。

```bash
sh ./scripts/generate_cluster_trace_jobs.sh \
  --max-jobs 10000 \
  --arrival-scale 0.001 \
  --duration-scale 1.0

POLICY=fcfs \
JOBS_FILE=./example/cluster-trace-v2026-spot-gpu/simai_fcfs/multi_jobs.csv \
./scripts/run_multi_jobs.sh
```

当前 `Spectrum-X_128g_8gps_100Gbps_A100` 拓扑按 128 张 GPU 运行，超过容量的任务默认跳过，统计写入 `trace_adapter_summary.json`。可用 `--total-gpus` 指定其他 GPU 总容量；`gpu_capacity_num` 仍固定按 8 使用。

```bash
./bin/SimAI_simulator -t 16 -j ./example/multi_jobs.csv -p first-fit \
  -n ./Spectrum-X_128g_8gps_100Gbps_A100 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

`-p best-fit` 优先选择满足需求且剩余显存较小的 GPU。GPU 容量不同时使用 `-R` 覆盖单卡资源。`-L` 输出资源快照以及任务完成时的 `e2e_latency_ns` 端到端时延。每个任务拥有独立的 `Sys`、workload 和 SimCCL group，网络回调使用 `(job_id, tag, src, dst)` 隔离。

`-p fgd` 启用碎片梯度下降放置启发式：评估候选连续 GPU 区间释放后留下的空闲碎片，选择碎片最小的放置。`-p crux` 根据通信量、GPU 数量和排队时间提升通信关键任务优先级，并选择跨服务器范围最小的物理 GPU 集合。

这两个策略是论文思想在调度层的实现。FGD 当前优化放置后剩余 GPU 的碎片度；Crux 使用通信量作为关键度信号，并结合 GPU 数量和排队时间排序，同时最小化所选物理 GPU 的服务器跨度。任务内部逻辑 collective rank 与物理 GPU ID 分离，因此两种策略都可以把任务放到不连续的物理 GPU 上，同时保持 SimCCL 的任务内 rank 组正确。

#### 泊松到达推理任务

SimAI 调度层提供 `scripts/generate_poisson_inference_jobs.sh`，它在指定仿真时间窗口内生成泊松到达任务，并调用已有的 `aicb/scripts/inference_workload_with_aiob.sh` 生成各类推理 workload。该脚本不修改 AICB 源码。

```bash
DURATION_SECONDS=60 ARRIVAL_RATE=0.2 GPU_CHOICES=8,16,32 \
  ./scripts/generate_poisson_inference_jobs.sh \
  --gpu-weights 5,3,2 \
  --models deepseek-671B,qwen3-235B \
  --model-weights 1,1 \
  --phases decode,prefill \
  --phase-weights 4,1 \
  --seed 2026
```

默认输出 `example/poisson_inference_jobs/multi_jobs.csv` 和每个任务独立的 workload 文件。生成后可直接运行：

```bash
JOBS_FILE=./example/poisson_inference_jobs/multi_jobs.csv \
  ./scripts/run_multi_jobs.sh
```

#### 泊松到达训练任务

训练任务使用 `scripts/generate_poisson_training_jobs.sh`。它从调度层调用已有的 `aicb/scripts/megatron_workload_with_aiob.sh`，AICB 的执行目录和所有产物都放在 SimAI 输出目录中，不修改 AICB 源码。

```bash
DURATION_SECONDS=60 ARRIVAL_RATE=0.2 GPU_CHOICES=8,16,32 \
  sh ./scripts/generate_poisson_training_jobs.sh \
  --gpu-weights 5,3,2 \
  --models 13,7,deepseek16 \
  --model-weights 4,3,1 \
  --tp-size 1 \
  --pp-size 1 \
  --ep-size 1 \
  --global-batch 1024 \
  --micro-batch 1 \
  --seq-length 4096 \
  --epoch-num 1 \
  --seed 2026
```

默认输出为 `example/poisson_training_jobs/multi_jobs.csv`。AICB 内置模型可能覆盖部分结构参数，例如 175B/22B 使用 TP=8、7B 使用 TP=4、DeepSeek 预设使用 TP=1；生成器会按照实际值检查 GPU 并行度。运行生成的任务：

```bash
JOBS_FILE=./example/poisson_training_jobs/multi_jobs.csv \
  ./scripts/run_multi_jobs.sh
```

#### 调度器可视化

完成多任务仿真后，使用 `scripts/plot_scheduler_metrics.sh` 离线读取 `SimAI_scheduler.csv`。该程序不会启动 SimAI、ns-3 或 AICB。

```bash
SCHEDULER_LOG=./SimAI_scheduler.csv \
PLOT_DIR=./scheduler_plots \
  sh ./scripts/plot_scheduler_metrics.sh
```

输出目录还包含资源占用与碎片化图：`resource_statistics.png`（分布统计）、`resource_occupancy_timeseries.png`（占用/碎片化时序）和 `resource_allocation_bubbles.png`（按 GPU 的资源时序气泡图）。`resource_timeseries.csv` 保存每个快照的资源指标，`resource_summary.csv` 保存均值、P50、P95 和最大值；`job_metrics.csv` 提供作业时延结构化数据。碎片化比例定义为空闲 GPU 中不属于最大连续空闲段的比例。绘图需要 Python `matplotlib`。

显存容量配置为非零值时，气泡大小还会反映显存占用比例；容量为零时，气泡表示 GPU 忙碌样本。大日志可通过 `--max-bubbles 20000` 限制气泡采样点数。

典型用法：
```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

> **线程建议**：多线程模式推荐使用 8-16 个线程。更高的线程数可减少大规模拓扑的仿真时间。

## SimAI_analytical 参数

| 参数 | 长格式 | 说明 | 默认值 |
|---|---|---|---|
| `-w` | `--workload` | Workload 文件路径 | 必填 |
| `-g` | `--gpus` | GPU 总数 | 必填 |
| `-g_p_s` | `--gpus-per-server` | 每台服务器 GPU 数（纵向扩展大小） | 必填 |
| `-r` | `--result` | 输出文件路径/前缀 | `./results/` |
| `-busbw` | `--bus-bandwidth` | busbw.yaml 文件路径（用户自定义 busbw）。**当前开源二进制未接线** — 见下方说明 | 不适用 |
| `-v` | `--visual` | 生成可视化文件 | 关闭 |
| `-nv` | - | NVLink 带宽（GB/s），用于自动 busbw（默认方法） | 推荐 |
| `-nic` | - | NIC 带宽（GB/s），用于自动 busbw（默认方法） | 推荐 |
| `-n_p_s` | - | 每台服务器 NIC 数，用于自动 busbw（默认方法） | 推荐 |

典型用法（自动 busbw —— 默认方式，当前二进制支持）：
```bash
./bin/SimAI_analytical \
  -w ./example/workload_analytical.txt \
  -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 \
  -r example-
```

> 注意：下面的 `-busbw example/busbw.yaml`（用户自定义 busbw）用法在当前开源 analytical 二进制中**不受支持**——`-busbw` 参数不会被解析，命令会打印用法并退出。此处仅作保留参考；手动 `busbw.yaml` 用法请参考早期 SimAI 版本。
```bash
./bin/SimAI_analytical \
  -w example/workload_analytical.txt \
  -g 9216 -g_p_s 8 \
  -r test- \
  -busbw example/busbw.yaml
```

## 拓扑生成器参数

详见 [快速入门 - 生成拓扑](../getting_started/quickstart.md#2-生成拓扑)。

---

> 最后编辑：2026-06-25
