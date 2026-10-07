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
import sys
from pathlib import Path


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_node_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as source:
        return sum(1 for _ in csv.DictReader(source))


def trace_workload_path(root: Path, output_dir: Path, gpu_count: int) -> str:
    # The ns-3 frontend creates the duration-specific workload at job start.
    # This compact placeholder is retained in the manifest for compatibility.
    path = (output_dir / "workloads" /
            "trace_workload_{}gpu.txt".format(gpu_count)).resolve()
    try:
        return "./" + str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_workload(path: Path, gpu_count: int, communication: int) -> None:
    path.write_text(
        "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
        "model_parallel_NPU_group: {0} ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: {0} "
        "checkpoints: 0 checkpoint_initiates: 0 pp_comm: 0\n"
        "1\ntrace_job -1 1 ALLREDUCE {1} 1 NONE 0 1 NONE 0 1\n".format(
            gpu_count, communication
        ),
        encoding="utf-8",
    )


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
    parser.add_argument("--max-jobs", type=int, default=0,
                        help="0 keeps all jobs")
    parser.add_argument("--overflow-policy", choices=("drop", "error"), default="drop",
                        help="handling for jobs requiring more than --total-gpus")
    parser.add_argument("--keep-trace-start", action="store_true",
                        help="do not normalize the first trace submit time to zero")
    parser.add_argument("--compact-csv", action="store_true",
                        help="use the 8-column CSV; requires a rebuilt SimAI binary")
    args = parser.parse_args()

    if args.total_gpus <= 0 or args.gpu_capacity_num <= 0:
        parser.error("--total-gpus and --gpu-capacity-num must be positive")
    if args.max_jobs < 0:
        parser.error("--max-jobs cannot be negative")
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
    placeholders = {}
    rows = []
    skipped = 0
    first_submit = None
    with jobs_file.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"job_name", "gpu_request", "worker_num", "submit_time", "duration"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error("job trace misses fields: " + ", ".join(sorted(missing)))
        for row_index, row in enumerate(reader, start=1):
            if args.max_jobs and len(rows) >= args.max_jobs:
                break
            try:
                requested = float(row["gpu_request"])
                workers = float(row["worker_num"])
                submit_seconds = float(row["submit_time"])
                duration = float(row["duration"])
                requested_gpus = int(math.ceil(requested * workers))
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
            workload = trace_workload_path(root, output_dir, gpu_count)
            if not args.compact_csv:
                workload_path = output_dir / "workloads" / (
                    "job_{}_{}gpu.txt".format(len(rows) + 1, gpu_count)
                )
                workload = trace_workload_path(root, output_dir, gpu_count)
                try:
                    workload = "./" + str(workload_path.resolve().relative_to(root.resolve()))
                except ValueError:
                    workload = str(workload_path.resolve())
                placeholders[workload_path] = (gpu_count, communication)
            else:
                placeholders.setdefault(output_dir / "workloads" /
                                        ("trace_workload_{}gpu.txt".format(gpu_count)),
                                        (gpu_count, communication))
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

    placeholder_dir = output_dir / "workloads"
    placeholder_dir.mkdir(parents=True, exist_ok=True)
    for workload_path, (gpu_count, communication) in placeholders.items():
        workload_path.parent.mkdir(parents=True, exist_ok=True)
        write_workload(workload_path, gpu_count, communication)
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
        "policy": "fcfs",
        "cpu_modeled": False,
    }
    (output_dir / "trace_adapter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote {} jobs to {}".format(len(rows), output_csv))
    print("Skipped {} jobs above/without topology capacity {}".format(skipped, args.total_gpus))
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
