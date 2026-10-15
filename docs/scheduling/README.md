# SimAI Multi-Job Scheduling

SimAI's ns-3 frontend can run multiple independent workloads in one simulation. The scheduler reserves GPU resources, starts a workload only after placement succeeds, and releases the placement when all ranks in that workload finish.

## Quick Start

```bash
cd /data4/nyf/workspace/SimAI

./scripts/build.sh -c ns3

POLICY=fcfs \
JOBS_FILE=./example/cluster-trace-v2026-spot-gpu/simai_fcfs/multi_jobs.csv \
./scripts/run_multi_jobs.sh
```

The input manifest uses this schema:

```text
job_id,workload,gpus,submit_time_ns,name,gpu_memory_bytes,host_memory_bytes,communication_bytes
```

The last column is optional. When it is nonzero, the ns-3 frontend creates a small duration-specific workload for that job at start time.

## Policies

| Policy | Behavior |
|---|---|
| `fcfs` | Strict first-come-first-served. A blocked head job prevents later jobs from overtaking it. |
| `first-fit` | Scans pending jobs in arrival order and chooses the first eligible placement. |
| `best-fit` | Prefers eligible GPUs with the least remaining memory. |
| `fgd` | Fragmentation Gradient Descent heuristic. Chooses the placement that leaves the fewest free GPUs outside the largest free interval. |
| `crux` | Communication-aware priority. Scores waiting jobs with communication bytes, GPU count, and queue age, then minimizes the physical server span of the placement. |

Example:

```bash
POLICY=fgd JOBS_FILE=./example/multi_jobs.csv ./scripts/run_multi_jobs.sh
POLICY=crux JOBS_FILE=./example/multi_jobs.csv ./scripts/run_multi_jobs.sh
```

The FGD and Crux implementations are scheduler-layer approximations of the corresponding paper ideas. FGD uses the current free-GPU layout as its fragmentation state. Crux uses the communication amount available in the manifest as the criticality signal and uses server span as the topology cost.

## Non-Contiguous GPUs and Collectives

Each job has a local rank space `0..N-1`. The collective implementation and MockNCCL group are built entirely in this local space. The network frontend stores a separate mapping:

```text
local rank 0 -> physical GPU 12
local rank 1 -> physical GPU 13
local rank 2 -> physical GPU 28
local rank 3 -> physical GPU 29
```

Communication requests keep local `sender_node` and `receiver_node` values. `ASTRASimNetwork::sim_send` and `sim_recv` translate them to physical ns-3 nodes immediately before creating a network flow. Callback maps include `job_id`, so equal tags and rank pairs from different jobs cannot collide.

This separation is required for non-contiguous placement. Passing physical GPU IDs into SimCCL would change ring/tree rank semantics and produce incorrect collective groups.

## Resource Accounting

Every GPU has an independent reservation bit and optional GPU and host-memory capacity. A job reserves all selected GPUs exclusively. A placement is eligible only when every selected GPU satisfies the requested memory. Use a default capacity for all GPUs:

```bash
GPU_MEMORY_BYTES=85899345920 \
HOST_MEMORY_BYTES=17179869184 \
POLICY=crux \
./scripts/run_multi_jobs.sh
```

For heterogeneous capacity, pass `GPU_RESOURCE_FILE` pointing to:

```text
gpu_id,gpu_memory_bytes,host_memory_bytes
0,85899345920,17179869184
1,85899345920,17179869184
```

## Trace Adapters

### GPU trace

The v2026 trace adapter reads only `gpu_request`, `worker_num`, `submit_time`, and `duration`. CPU fields are ignored. `--max-jobs` randomly selects up to that many data rows from `job_info_df.csv` (excluding the header); `0` processes all rows. The exact job request is:

```text
ceil(gpu_request * worker_num)
```

`gpu_capacity_num=8` describes the trace node capacity; it does not round job requests. Generate a manifest with:

```bash
sh ./scripts/generate_cluster_trace_jobs.sh \
  --max-jobs 10000 \
  --arrival-scale 0.001
```

The adapter writes per-job workloads and `trace_adapter_summary.json` below `simai_fcfs/`. Jobs above the selected SimAI topology capacity are counted and skipped by default.

Large communication amounts are split into consecutive `ALLREDUCE` layers in
the same job. `--max-allreduce-bytes` (or `MAX_ALLREDUCE_BYTES` in the shell
wrapper) defaults to 1048576. The effective bound is
`min(max_allreduce_bytes, 8192 * job_gpu_count)`: with the current 9000-byte
packet payload setting, this conservatively bounds each Ring flow to 8192
bytes even with one channel. This avoids the known final-packet size check
that loses send completion notifications for payloads near the MTU. This
generation workaround assumes the current Ring model and MTU; it does not
fix the packet-size check in ns-3.

Chunks differ by at most one byte and sum exactly to the job's communication
amount. New chunks from splitting are at least 4096 bytes, so they do not
introduce extra padding from the frontend's existing minimum. Original jobs
below 4096 bytes retain that existing minimum behavior. Each chunk is a
blocking forward collective named `trace_job_part_N`; the scheduler still
sees one job with its original GPU demand and arrival time. Extra collectives
add startup overhead, so resulting service latency can change. Summary JSON
records the split job count, collective count, and total communication bytes.

Both CSV modes now reference a separate workload per job. In `--compact-csv`
mode the total remains in `communication_bytes` for Crux, and a rebuilt
frontend validates and reuses split workloads instead of overwriting them
with one collective. The default seven-column mode needs no frontend rebuild
for splitting. Existing workload files must be regenerated; generation does
not change previous experiment outputs.

```bash
OUTPUT_DIR=./example/cluster-trace-v2026-spot-gpu/simai_split \
MAX_ALLREDUCE_BYTES=1048576 \
sh ./scripts/generate_cluster_trace_jobs.sh --max-jobs 300
```

### Synthetic inference and training

The Poisson generators call the existing AICB scripts from a SimAI-owned staging directory:

```bash
./scripts/generate_poisson_inference_jobs.sh --help
./scripts/generate_poisson_training_jobs.sh --help
```

They do not modify AICB source files.

## Logs and Plots

The scheduler log is selected with `SCHEDULER_LOG` or `-L`:

```bash
SCHEDULER_LOG=./SimAI_scheduler.csv ./scripts/run_multi_jobs.sh
```

The CSV records submit, schedule, and finish snapshots for every GPU. Completion rows include:

- `queue_latency_ns`
- `service_latency_ns`
- `e2e_latency_ns`

Generate offline plots after a simulation:

```bash
SCHEDULER_LOG=./SimAI_scheduler.csv \
PLOT_DIR=./scheduler_plots \
sh ./scripts/plot_scheduler_metrics.sh
```

The plotting script does not start SimAI, ns-3, or AICB.

The default output contains three figures: `gpu_schedule_timeline.png`,
`job_latencies.png`, and `resource_occupancy_timeseries.png`. Latency bars
show the 15 slowest completed jobs, with values above each bar; the histogram
and CSV metrics retain all jobs. Resource panels show GPU allocation,
fragmentation ratio, and active jobs. Use `--details` to additionally generate
the job timeline, utilization, distributions, and optional heatmap/bubble plots;
the existing skip flags still apply to those optional figures. Old PNGs in a
reused output directory are retained.

`--curve-bins 60` is the default: dense resource curves use equal-duration
bins, averaging each step state by how long it lasted. Shading shows the
minimum and maximum within each bin. Duplicate timestamps retain the final
snapshot. Fewer bins produce a coarser trend; `--curve-bins 0` shows raw event
states. Sparse curves keep their raw states. CSV and summary statistics are
computed from the original snapshots, independently of these display settings.
`--max-latency-jobs` adjusts the number of jobs in the latency bars.

```bash
SCHEDULER_LOG=./SimAI_scheduler.csv PLOT_DIR=./scheduler_plots_readable \
PLOT_CURVE_BINS=40 PLOT_MAX_LATENCY_JOBS=15 \
sh ./scripts/plot_scheduler_metrics.sh --warmup-time-ns 0
```

## Policy Comparison

`scripts/compare_scheduler_policies.sh` replays one manifest with each policy in an isolated directory. The script copies the manifest and all referenced workloads before each run, then saves simulator output, scheduler CSV, per-job logs, plots, and metrics separately.

```bash
JOBS_FILE=./example/cluster-trace-v2026-spot-gpu/multi_jobs.csv \
POLICIES=fcfs,first-fit,best-fit,fgd,crux \
COMPARISON_ROOT=./scheduler_comparisons \
./scripts/run_multi_jobs.sh
```

Each comparison gets a timestamped directory. `comparison_metadata.json` contains the source manifest SHA-256 and shared topology/config settings. `comparison_summary.csv` compares exit status, post-warmup completed jobs, mean/P50/P95 end-to-end latency, queue/service latency, GPU allocation, and fragmentation. The final comparison figures are `comparison_metrics.png` (combined dashboard), `comparison_latency.png` (E2E and queue/service latency), `comparison_utilization.png` (GPU allocation curve and mean), and `comparison_fragmentation.png` (fragmentation curve and mean). All policies use the latest detected warmup completion time as a common plotting boundary. A failing policy is recorded and the remaining policies continue; pass `--stop-on-error` to stop after the first failure.

Comparison bars include numeric values, with latency expressed in ms. The
overview contains summary bars; the focused utilization and fragmentation
figures contain the trends and mean bars. `PLOT_CURVE_BINS`,
`PLOT_MAX_LATENCY_JOBS`, and `PLOT_DETAILS=1` on `run_multi_jobs.sh` forward the
same display settings to every policy; the trend bin count is also used for
the final cross-policy plots. Direct comparison CLI options are `--curve-bins`,
`--max-latency-jobs`, and `--details`.

All shell entries forward the same display settings. `plot_scheduler_metrics.sh`
also accepts `PLOT_DPI` and `PLOT_DETAILS=1`; memory, heatmap and bubble plots
are omitted by default, and can be enabled in details mode with
`PLOT_NO_MEMORY=0`, `PLOT_SKIP_HEATMAP=0`, and `PLOT_SKIP_BUBBLES=0`.
Warmup is disabled unless `WARMUP_TIME_NS` or `WARMUP_FRACTION` is set.

Replot an existing comparison without starting any simulator:

```bash
COMPARISON_DIR=./scheduler_comparisons/test \
POLICIES=fcfs,fgd,crux PLOT_CURVE_BINS=40 PLOT_MAX_LATENCY_JOBS=15 \
sh ./scripts/plot_scheduler_comparison.sh
```

This regenerates each policy's `scheduler_plots/` and the four comparison
figures in the run directory. Source logs, simulation metadata, workloads and
simulation summaries are preserved. Display settings and recomputed metrics
are recorded separately in `replot_summary.json`. Logs from unfinished runs
may be plotted, but their metrics describe only the currently recorded jobs.
Omit `POLICIES` to discover available policy logs, or pass
`--run-dir ./scheduler_comparisons/test` directly. Explicit CLI values override
environment defaults. Multiple runs can be plotted with:

```bash
for run in exp exp1 exp2; do
  COMPARISON_DIR="./scheduler_comparisons/$run" PLOT_CURVE_BINS=40 \
    sh ./scripts/plot_scheduler_comparison.sh || break
done
```

## Verification

RDMA completion callbacks use a hash lookup on physical `(src, dst, port)`.
Each entry retains its job tag until both send and receive completion arrive,
then is erased. Repeated completions do not advance the collective counters.
Port allocation skips live entries when the 16-bit counter wraps. The frontend
wraps MTP's manual critical section in a scope guard, so early returns and
exceptions release it; callbacks run after that lock is released.

The standalone regression test covers both completion orders, duplicate and
missing completions, job isolation on nonconsecutive GPU IDs, port reuse and
wrap, cleanup after 200,000 flows, and lock release on return and exhaustion.
Run it from the SimAI root when compiling the changes:

```bash
rdma_test_dir=$(mktemp -d)
g++ -std=c++17 -Wall -Wextra -pedantic \
  -Iastra-sim-alibabacloud/astra-sim/network_frontend/ns3 \
  tests/rdma_flow_registry_test.cc -o "$rdma_test_dir/rdma_flow_registry_test"
"$rdma_test_dir/rdma_flow_registry_test"
```

Useful local checks:

```bash
g++ -std=c++11 -fsyntax-only \
  -Iastra-sim-alibabacloud/astra-sim/network_frontend/ns3 \
  -x c++ - <<'CPP'
#include "MultiJobScheduler.h"
int main() {
  SimAiMultiJob::Scheduler scheduler(
      16, 8, SimAiMultiJob::Policy::FragmentationGradient);
  return 0;
}
CPP

g++ -std=c++17 -fsyntax-only \
  -Iastra-sim-alibabacloud \
  -Iastra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/src/applications \
  -Iastra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/build/include \
  astra-sim-alibabacloud/astra-sim/network_frontend/ns3/AstraSimNetwork.cc
```

The full ns-3 build may require writable ownership of the generated ns-3 build directory.
