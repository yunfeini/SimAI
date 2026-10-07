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

The v2026 trace adapter reads only `gpu_request`, `worker_num`, `submit_time`, and `duration`. CPU fields are ignored. The exact job request is:

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

## Verification

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
