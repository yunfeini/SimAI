#!/usr/bin/env python3
"""Adapt the v2026 spot-GPU trace to the SimAI multi-job scheduler.

Only GPU resources are modeled.  Each trace row becomes one scheduler job:
effective GPUs = ceil(gpu_request * worker_num), submit_time is scaled into
SimAI nanoseconds, and duration is written as the job communication amount.
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


DEFAULT_MAX_ALLREDUCE_BYTES = 1024 * 1024
# With the current 9000-byte MTU, a Ring flow of at most 8192 bytes fits
# inside a single packet including its headers. Ring divides each collective
# by ranks and channels; assuming one channel is the conservative bound.
MAX_RING_FLOW_BYTES = 8192
MIN_RUNTIME_ALLREDUCE_BYTES = 4096


def allreduce_limit(gpu_count: int, max_allreduce_bytes: int) -> int:
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    if max_allreduce_bytes < 2 * MIN_RUNTIME_ALLREDUCE_BYTES:
        raise ValueError("--max-allreduce-bytes must be at least 8192")
    return min(max_allreduce_bytes, gpu_count * MAX_RING_FLOW_BYTES)


def split_communication(communication: int, gpu_count: int,
                        max_allreduce_bytes: int = DEFAULT_MAX_ALLREDUCE_BYTES):
    """Yield bounded collective sizes without changing the total byte count."""
    if communication <= 0:
        raise ValueError("communication must be positive")
    limit = allreduce_limit(gpu_count, max_allreduce_bytes)
    count = (communication + limit - 1) // limit
    size, larger_parts = divmod(communication, count)
    # Balance all parts, rather than emitting a small remainder that the
    # frontend would silently raise to its 4096-byte minimum.
    for index in range(count):
        yield size + (index < larger_parts)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_node_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def trace_workload_path(root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return "./" + str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_workload(path: Path, gpu_count: int, communication: int,
                   max_allreduce_bytes: int = DEFAULT_MAX_ALLREDUCE_BYTES) -> int:
    limit = allreduce_limit(gpu_count, max_allreduce_bytes)
    count = (communication + limit - 1) // limit
    if communication <= 0:
        raise ValueError("communication must be positive")
    with path.open("w", encoding="utf-8") as output:
        output.write(
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            "model_parallel_NPU_group: {0} ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: {0} "
            "checkpoints: 0 checkpoint_initiates: 0 pp_comm: 0\n{1}\n".format(
                gpu_count, count
            )
        )
        for index, amount in enumerate(split_communication(
                communication, gpu_count, max_allreduce_bytes)):
            name = "trace_job" if count == 1 else "trace_job_part_{}".format(index)
            output.write(
                "{} -1 1 ALLREDUCE {} 1 NONE 0 1 NONE 0 1\n".format(name, amount)
            )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert cluster-trace-v2026-spot-gpu into a SimAI FCFS job CSV."
    )
    root = Path(__file__).resolve().parents[1]
    default_trace = root / "example" / "cluster-trace-v2026-spot-gpu"
    parser.add_argument("--trace-dir", type=Path, default=default_trace)
    parser.add_argument("--jobs-file", type=Path, default=None)
    parser.add_argument("--nodes-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=default_trace / "simai_fcfs")
    parser.add_argument("--total-gpus", type=int, default=128,
                        help="GPU capacity of the selected SimAI topology")
    parser.add_argument("--gpu-capacity-num", type=int, default=8,
                        help="fixed GPUs per trace node; CPU is ignored")
    parser.add_argument("--arrival-scale", type=positive_float, default=1.0,
                        help="multiply trace submit seconds before converting to ns")
    parser.add_argument("--duration-scale", type=positive_float, default=1.0,
                        help="multiply duration before writing communication amount")
    parser.add_argument("--max-allreduce-bytes", type=int,
                        default=DEFAULT_MAX_ALLREDUCE_BYTES,
                        help="maximum bytes per ALLREDUCE (default: 1048576; "
                             "also capped at 8192 * job GPUs for the 9000-byte MTU)")
    parser.add_argument("--max-jobs", type=int, default=0,
                        help="randomly select up to this many input jobs; 0 keeps all")
    parser.add_argument("--overflow-policy", choices=("drop", "error"), default="drop",
                        help="handling for jobs requiring more than --total-gpus")
    parser.add_argument("--keep-trace-start", action="store_true",
                        help="do not normalize the first trace submit time to zero")
    parser.add_argument("--compact-csv", action="store_true",
                        help="use the 8-column CSV with total communication bytes; "
                             "requires a rebuilt SimAI binary to preserve split workloads")
    args = parser.parse_args()

    if args.total_gpus <= 0 or args.gpu_capacity_num <= 0:
        parser.error("--total-gpus and --gpu-capacity-num must be positive")
    if args.max_jobs < 0:
        parser.error("--max-jobs cannot be negative")
    if args.max_allreduce_bytes < 2 * MIN_RUNTIME_ALLREDUCE_BYTES:
        parser.error("--max-allreduce-bytes must be at least 8192")
    trace_dir = args.trace_dir.resolve()
    jobs_file = (args.jobs_file or trace_dir / "job_info_df.csv").resolve()
    nodes_file = (args.nodes_file or trace_dir / "node_info_df.csv").resolve()
    output_dir = args.output_dir.resolve()
    if not jobs_file.is_file():
        parser.error("job trace not found: {}".format(jobs_file))
    if not nodes_file.is_file():
        parser.error("node trace not found: {}".format(nodes_file))

    trace_node_count = load_node_count(nodes_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = {}
    rows = []
    skipped = 0
    first_submit = None
    with jobs_file.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"job_name", "gpu_request", "worker_num", "submit_time", "duration"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error("job trace misses fields: " + ", ".join(sorted(missing)))
        selected_row_indices = None
        if args.max_jobs:
            total_jobs = sum(1 for _ in reader)
            if args.max_jobs < total_jobs:
                selected_row_indices = set(
                    random.sample(range(1, total_jobs + 1), args.max_jobs)
                )
            source.seek(0)
            reader = csv.DictReader(source)
        for row_index, row in enumerate(reader, start=1):
            if selected_row_indices is not None and row_index not in selected_row_indices:
                continue
            try:
                requested = float(row["gpu_request"])
                workers = float(row["worker_num"])
                submit_seconds = float(row["submit_time"])/1e10
                requested_gpus = int(2*math.ceil(requested))
                # The trace duration is intentionally used as the simulated
                # communication amount.  Do not normalize it by GPU count or
                # an arbitrary byte scale: that made all values round to 1.
                duration = float(row["duration"])/requested_gpus
                # gpu_capacity_num describes node capacity only; it must not
                # round a job request to a multiple of eight GPUs.
                gpu_count = requested_gpus
            except (TypeError, ValueError) as error:
                raise ValueError("invalid numeric value at trace row {}: {}".format(row_index, error))
            if gpu_count <= 0 or not math.isfinite(submit_seconds) or not math.isfinite(duration):
                skipped += 1
                continue
            if gpu_count > args.total_gpus:
                if args.overflow_policy == "error":
                    raise ValueError(
                        "job {} requires {} GPUs, above topology capacity {}".format(
                            row["job_name"], gpu_count, args.total_gpus
                        )
                    )
                skipped += 1
                continue
            if first_submit is None:
                first_submit = submit_seconds
            communication = max(1, int(round(duration * args.duration_scale)))
            relative_submit = submit_seconds if args.keep_trace_start else submit_seconds - first_submit
            submit_ns = max(0, int(round(relative_submit * args.arrival_scale * 1_000_000_000)))
            job_id = len(rows) + 1
            # Every job needs its own workload even in the 8-column manifest:
            # jobs with equal GPU counts can have different collective splits.
            workload_path = output_dir / "workloads" / (
                "job_{}_{}gpu.txt".format(job_id, gpu_count)
            )
            workload = trace_workload_path(root, workload_path)
            workloads[workload_path] = (gpu_count, communication)
            job_row = {
                "job_id": job_id,
                "workload": workload,
                "gpus": gpu_count,
                "submit_time_ns": submit_ns,
                "name": "trace-{}-req{}g".format(row["job_name"], requested_gpus),
                "gpu_memory_bytes": 0,
                "host_memory_bytes": 0,
            }
            if args.compact_csv:
                job_row["communication_bytes"] = communication
            rows.append(job_row)

    rows.sort(key=lambda row: (row["submit_time_ns"], row["job_id"]))
    output_csv = output_dir / "multi_jobs.csv"
    fields = ("job_id", "workload", "gpus", "submit_time_ns", "name",
              "gpu_memory_bytes", "host_memory_bytes")
    if args.compact_csv:
        fields += ("communication_bytes",)
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    collective_count = 0
    split_jobs = 0
    communication_total = 0
    for workload_path, (gpu_count, communication) in workloads.items():
        workload_path.parent.mkdir(parents=True, exist_ok=True)
        count = write_workload(workload_path, gpu_count, communication,
                               args.max_allreduce_bytes)
        collective_count += count
        split_jobs += count > 1
        communication_total += communication
    summary = {
        "source_jobs_file": str(jobs_file),
        "source_nodes_file": str(nodes_file),
        "trace_node_count": trace_node_count,
        "fixed_gpu_capacity_num": args.gpu_capacity_num,
        "round_gpu_requests_to_node_capacity": False,
        "trace_gpu_capacity": trace_node_count * args.gpu_capacity_num,
        "simai_total_gpus": args.total_gpus,
        "jobs_written": len(rows),
        "jobs_skipped": skipped,
        "arrival_scale": args.arrival_scale,
        "duration_scale": args.duration_scale,
        "max_allreduce_bytes": args.max_allreduce_bytes,
        "max_ring_flow_bytes": MAX_RING_FLOW_BYTES,
        "allreduce_limit_rule": "min(max_allreduce_bytes, 8192 * job GPUs)",
        "jobs_with_split_allreduces": split_jobs,
        "allreduce_count": collective_count,
        "total_workload_communication_bytes": communication_total,
        "policy": "fcfs",
        "cpu_modeled": False,
    }
    (output_dir / "trace_adapter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote {} jobs to {}".format(len(rows), output_csv))
    print("Skipped {} jobs above/without topology capacity {}".format(skipped, args.total_gpus))
    print("Wrote {} ALLREDUCE operations; {} jobs split; total communication {} bytes".format(
        collective_count, split_jobs, communication_total
    ))
    print("Trace nodes: {}, fixed GPUs/node: {}, trace GPU capacity: {}".format(
        trace_node_count, args.gpu_capacity_num, trace_node_count * args.gpu_capacity_num
    ))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(1)
