# SimAI Build Options

> [中文版](../CN/configuration/build-options.md)

## Build Modes

| Mode | Command | Output Binary | Description |
|---|---|---|---|
| analytical | `./scripts/build.sh -c analytical` | `bin/SimAI_analytical` | Fast busbw-based simulation (no network modeling) |
| ns3 | `./scripts/build.sh -c ns3` | `bin/SimAI_simulator` | Full ns3 network simulation (high fidelity) |
| phy | `./scripts/build.sh -c phy` | `bin/SimAI_phynet` | Physical RDMA traffic generation (requires IB hardware) |

## Clean Build

```bash
./scripts/build.sh -l ns3         # Clean ns3 build artifacts
./scripts/build.sh -l analytical  # Clean analytical build
./scripts/build.sh -l phy         # Clean physical build
```

## SimAI_simulator Parameters (ns3 mode)

| Parameter | Long Form | Description | Default |
|---|---|---|---|
| `-t` | `--thread` | Number of threads for multi-threading acceleration | 1 |
| `-w` | `--workload` | Path to workload file | Required |
| `-n` | `--network-topo` | Path to network topology file | Required |
| `-c` | `--config` | Path to simulation config file | Required |
| `-j` | `--jobs` | CSV manifest for multi-job simulation | Optional |
| `-p` | `--policy` | Multi-job policy: `fcfs`, `first-fit`, `best-fit`, `fgd`, or `crux` | `first-fit` |
| `-m` | `--gpu-memory` | Per-GPU memory capacity in bytes; `0` leaves it unspecified | `0` |
| `-H` | `--host-memory` | Per-GPU host-memory capacity in bytes; `0` leaves it unspecified | `0` |
| `-R` | `--gpu-resources` | Per-GPU resource CSV: `gpu_id,gpu_memory_bytes,host_memory_bytes` | Optional |
| `-L` | `--scheduler-log` | Scheduler CSV log path | `./SimAI_scheduler.csv` |

### Multi-job scheduling

Pass a CSV job manifest with `-j` to run independent AICB/SimAI workloads in one ns3 simulation. Each row is `job_id,workload,gpus,submit_time_ns[,name[,gpu_memory_bytes[,host_memory_bytes[,communication_bytes]]]]`; resource values are per GPU. When `communication_bytes` is set, ns-3 creates the duration-specific trace workload when the job starts. Server-sized jobs use complete physical servers, while server IDs may be non-contiguous.

#### v2026 GPU trace adapter

Convert `example/cluster-trace-v2026-spot-gpu` at the scheduling layer with the FCFS adapter. It reads GPU fields only and ignores CPU; every trace node is treated as having 8 GPUs, while each job keeps the exact `ceil(gpu_request * worker_num)` GPU request instead of rounding to a multiple of 8. `duration` is written into the task workload's `ALLREDUCE` communication amount. Trace submit seconds are converted to ns; use `--arrival-scale` to compress or expand the arrival timeline. `--max-jobs` randomly selects up to that many data rows from `job_info_df.csv` (the header is excluded); `0` processes all rows.

```bash
sh ./scripts/generate_cluster_trace_jobs.sh \
  --max-jobs 10000 \
  --arrival-scale 0.001 \
  --duration-scale 1.0

POLICY=fcfs \
JOBS_FILE=./example/cluster-trace-v2026-spot-gpu/simai_fcfs/multi_jobs.csv \
./scripts/run_multi_jobs.sh
```

The current `Spectrum-X_128g_8gps_100Gbps_A100` topology has 128 GPUs; jobs above this capacity are skipped by default and counted in `trace_adapter_summary.json`. Use `--total-gpus` for a different topology; `gpu_capacity_num` remains fixed at 8.

```bash
./bin/SimAI_simulator -t 16 -j ./example/multi_jobs.csv -p first-fit \
  -n ./Spectrum-X_128g_8gps_100Gbps_A100 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

Use `-p best-fit` to prefer GPUs with the least remaining memory that satisfy the request. Use `-R` when GPU capacities differ. `-L` writes resource snapshots and per-job `e2e_latency_ns` completion latency. Each job has its own `Sys`, workload and SimCCL group; network callbacks are keyed by `(job_id, tag, src, dst)`.

`-p fgd` enables a fragmentation gradient descent placement heuristic. It scores candidate contiguous windows by the free GPU fragmentation they leave behind and chooses the lowest score. `-p crux` prioritizes waiting jobs by communication volume, GPU count, and queue age, then places them on the lowest server-span physical GPU set available.

These are scheduler-layer implementations inspired by the two papers. FGD currently optimizes the post-placement free-GPU fragmentation score. Crux currently uses communication bytes as the criticality signal, combines it with requested GPU count and queue age, and minimizes the physical server span of the selected GPUs. The simulator keeps the task's logical collective ranks separate from physical GPU IDs, so both policies can place a task on non-contiguous physical GPUs while preserving its SimCCL rank group.

#### Poisson inference arrivals

The scheduling layer provides `scripts/generate_poisson_inference_jobs.sh`. It samples Poisson arrivals over a simulation-time window and invokes the existing `aicb/scripts/inference_workload_with_aiob.sh` for the required inference workloads without changing AICB source files.

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

By default it writes `example/poisson_inference_jobs/multi_jobs.csv` and one workload file per job. Run the result with:

```bash
JOBS_FILE=./example/poisson_inference_jobs/multi_jobs.csv \
  ./scripts/run_multi_jobs.sh
```

#### Poisson training arrivals

Use `scripts/generate_poisson_training_jobs.sh` for training jobs. It invokes the existing `aicb/scripts/megatron_workload_with_aiob.sh` from the scheduling layer. AICB runs from a SimAI-owned staging directory, so its source tree is not modified.

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

The default manifest is `example/poisson_training_jobs/multi_jobs.csv`. AICB model presets can override architecture options: 175B/22B use TP=8, 7B uses TP=4, and the DeepSeek presets use TP=1. The generator validates GPU parallelism against these effective values. Run the jobs with:

```bash
JOBS_FILE=./example/poisson_training_jobs/multi_jobs.csv \
  ./scripts/run_multi_jobs.sh
```

#### Scheduler visualization

After a multi-job simulation, use `scripts/plot_scheduler_metrics.sh` to read `SimAI_scheduler.csv` offline. It does not launch SimAI, ns-3, or AICB.

```bash
SCHEDULER_LOG=./SimAI_scheduler.csv \
PLOT_DIR=./scheduler_plots \
  sh ./scripts/plot_scheduler_metrics.sh
```

Plots start after warmup by default: the first scheduler snapshot reaching the peak observed GPU occupancy is used as the warmup boundary. Use `--warmup-fraction 0.95` to start at the first snapshot reaching 95% of peak occupancy, or pass an exact boundary with `--warmup-time-ns`. The original complete `job_metrics.csv` is preserved; post-warmup metrics are written under `post_warmup/`.

The output directory also contains resource occupancy and fragmentation plots: `resource_statistics.png` (distributions), `resource_occupancy_timeseries.png` (occupancy/fragmentation over time), and `resource_allocation_bubbles.png` (per-GPU resource bubbles over time). `resource_timeseries.csv` stores per-snapshot resource metrics, while `resource_summary.csv` stores mean, P50, P95, and maximum values. `job_metrics.csv` provides structured job latency data. Fragmentation is defined as the fraction of free GPUs outside the largest contiguous free-GPU run. Plotting requires Python `matplotlib`.

When non-zero GPU memory capacities are configured, bubble size also reflects memory occupancy; otherwise bubbles represent busy-GPU samples. For large logs, limit bubble samples with `--max-bubbles 20000`.

#### Policy comparison

Use `scripts/compare_scheduler_policies.sh` to replay one job manifest with several policies. Each policy receives a private copy of the manifest and workloads and writes its simulator output, scheduler log, per-job output, and plots to a separate directory.

```bash
JOBS_FILE=./example/cluster-trace-v2026-spot-gpu/multi_jobs.csv \
POLICIES=fcfs,first-fit,best-fit,fgd,crux \
COMPARISON_ROOT=./scheduler_comparisons \
./scripts/run_multi_jobs.sh
```

Each run creates a timestamped directory. `comparison_metadata.json` records the source manifest SHA-256 and shared topology/config. `comparison_summary.csv` reports status, post-warmup completed-job count, mean/P50/P95 end-to-end latency, queue/service latency, GPU allocation, and fragmentation. The final comparison figures are `comparison_metrics.png` (combined dashboard), `comparison_latency.png` (E2E and queue/service latency), `comparison_utilization.png` (GPU allocation curve and mean), and `comparison_fragmentation.png` (fragmentation curve and mean). All policies use the latest detected warmup completion time as a common measurement boundary. A failed policy is recorded while the remaining policies continue; pass `--stop-on-error` to stop at the first failure.

Typical usage:
```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

> **Threading**: Recommended 8-16 threads for multi-threaded mode. Higher thread count reduces simulation time for large topologies.

## SimAI_analytical Parameters

| Parameter | Long Form | Description | Default |
|---|---|---|---|
| `-w` | `--workload` | Path to workload file | Required |
| `-g` | `--gpus` | Total GPU count | Required |
| `-g_p_s` | `--gpus-per-server` | GPUs per server (scale-up size) | Required |
| `-r` | `--result` | Output file path/prefix | `./results/` |
| `-busbw` | `--bus-bandwidth` | Path to busbw.yaml (user-defined busbw). **Not wired in the current open-source binary** — see note below | n/a |
| `-v` | `--visual` | Generate visualization files | Off |
| `-nv` | - | NVLink bandwidth (GB/s) for auto busbw (default method) | Recommended |
| `-nic` | - | NIC bandwidth (GB/s) for auto busbw (default method) | Recommended |
| `-n_p_s` | - | NICs per server for auto busbw (default method) | Recommended |

Typical usage (automatic busbw — the default, supported by the current binary):
```bash
./bin/SimAI_analytical \
  -w ./example/workload_analytical.txt \
  -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 \
  -r example-
```

> Note: The `-busbw example/busbw.yaml` form below (user-defined busbw) is **not supported** by the current open-source analytical binary — the `-busbw` flag is not parsed, so the command prints usage and exits. It is kept for reference; see earlier SimAI versions for the manual `busbw.yaml` workflow.
```bash
./bin/SimAI_analytical \
  -w example/workload_analytical.txt \
  -g 9216 -g_p_s 8 \
  -r test- \
  -busbw example/busbw.yaml
```

## Topology Generator Parameters

See [Quick Start - Generate Topology](../getting_started/quickstart.md#2-generate-topology) for full parameter reference.

---

> Last edited: 2026-06-25
