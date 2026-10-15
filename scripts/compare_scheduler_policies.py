#!/usr/bin/env python3
"""Run one SimAI job trace with several scheduling policies.

Each policy runs in a separate working directory. The source manifest and
workload files are copied, so relative output files from the simulator cannot
overwrite another policy's results.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from scheduler_plot_style import (
    DEFAULT_CURVE_BINS, DEFAULT_MAX_LATENCY_JOBS,
    curve_note, label_bars, plot_curve,
)

DEFAULT_POLICIES = ("fcfs", "first-fit", "best-fit", "fgd", "crux")


def resolve_source_path(value: str, manifest: Path, root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = [
        (root / value.lstrip("./")).resolve(),
        (manifest.parent / value).resolve(),
        (Path.cwd() / value).resolve(),
    ]
    for resolved in candidates:
        if resolved.is_file():
            return resolved
    return candidates[0]


def read_manifest(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise RuntimeError("job manifest contains no jobs")
    reader = csv.reader(lines)
    parsed = list(reader)
    header = parsed[0]
    if "job_id" in header and "workload" in header:
        fields = header
        data_rows = parsed[1:]
    else:
        fields = ["job_id", "workload", "gpus", "submit_time_ns", "name",
                  "gpu_memory_bytes", "host_memory_bytes", "communication_bytes"][:len(header)]
        data_rows = parsed
    if "workload" not in fields:
        raise RuntimeError("job manifest must contain a workload column")
    rows = [dict(zip(fields, row)) for row in data_rows]
    return fields, rows


def read_numeric_column(path: Path, column: str) -> List[float]:
    if not path.is_file():
        return []
    values = []
    with path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            value = row.get(column, "")
            if value:
                try:
                    values.append(float(value))
                except ValueError:
                    pass
    return values


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def result_metrics(policy_dir: Path) -> Dict[str, object]:
    jobs_csv = policy_dir / "scheduler_plots" / "post_warmup" / "job_metrics.csv"
    e2e = read_numeric_column(jobs_csv, "e2e_latency_ns")
    queue = read_numeric_column(jobs_csv, "queue_latency_ns")
    service = read_numeric_column(jobs_csv, "service_latency_ns")
    summary = {}
    resource_csv = policy_dir / "scheduler_plots" / "resource_summary.csv"
    if resource_csv.is_file():
        with resource_csv.open("r", newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                try:
                    summary[row["metric"]] = float(row["mean"])
                except (KeyError, ValueError):
                    pass
    return {
        "completed_jobs_post_warmup": len(e2e),
        "e2e_mean_ns": sum(e2e) / len(e2e) if e2e else "",
        "e2e_p50_ns": percentile(e2e, 0.50) if e2e else "",
        "e2e_p95_ns": percentile(e2e, 0.95) if e2e else "",
        "queue_mean_ns": sum(queue) / len(queue) if queue else "",
        "service_mean_ns": sum(service) / len(service) if service else "",
        "gpu_allocation_mean": summary.get("gpu_allocation_ratio", ""),
        "fragmentation_mean": summary.get("free_fragmentation_ratio", ""),
    }


def read_resource_curve(policy_dir: Path, warmup_ns: int) -> Tuple[List[float], List[float], List[float]]:
    path = policy_dir / "scheduler_plots" / "resource_timeseries.csv"
    if not path.is_file():
        return [], [], []
    times, allocation, fragmentation = [], [], []
    with path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            time_ns = float(row["time_ns"])
            if time_ns < warmup_ns:
                continue
            times.append((time_ns - warmup_ns) / 1_000_000.0)
            allocation.append(float(row["gpu_allocation_ratio"]) * 100.0)
            fragmentation.append(float(row["free_fragmentation_ratio"]) * 100.0)
    return times, allocation, fragmentation


def plot_comparison(results: List[Dict[str, object]], run_dir: Path,
                    warmup_ns: int, curve_bins: int = DEFAULT_CURVE_BINS) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib unavailable"
    valid = [result for result in results if result["plot_status"] == "ok"]
    if not valid:
        return "skipped: no plotted policies"
    labels = [str(result["policy"]) for result in valid]
    positions = list(range(len(valid)))
    policy_colors = {policy: plt.get_cmap("tab10")(index % 10)
                     for index, policy in enumerate(labels)}
    curves = {}
    for result in valid:
        policy = str(result["policy"])
        curves[policy] = read_resource_curve(run_dir / policy, warmup_ns)

    # The overview contains summary bars only; curves live in focused figures.
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = (
        ("e2e_mean_ns", "Mean end-to-end latency (ms)", 1e-6),
        ("gpu_allocation_mean", "Mean GPU allocation (%)", 100.0),
        ("fragmentation_mean", "Mean free-GPU fragmentation (%)", 100.0),
        ("completed_jobs_post_warmup", "Completed jobs", 1.0),
    )
    for axis, (field, title, scale) in zip(axes.flat, fields):
        values = [float(result.get(field) or 0.0) * scale for result in valid]
        label_bars(axis, axis.bar(positions, values,
                                  color=[policy_colors[policy] for policy in labels]))
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_dir / "comparison_metrics.png", dpi=160)
    plt.close(figure)

    latency_figure, latency_axes = plt.subplots(1, 2, figsize=(15, 5.5))
    latency_fields = (
        ("e2e_mean_ns", "Mean E2E"),
        ("e2e_p50_ns", "E2E P50"),
        ("e2e_p95_ns", "E2E P95"),
    )
    width = 0.8 / len(latency_fields)
    for index, (field, label) in enumerate(latency_fields):
        values = [float(result.get(field) or 0.0) / 1_000_000 for result in valid]
        offset = (index - (len(latency_fields) - 1) / 2) * width
        label_bars(latency_axes[0], latency_axes[0].bar(
            [position + offset for position in positions], values,
            width=width, label=label))
    latency_axes[0].set_title("End-to-End Latency")
    latency_axes[0].set_ylabel("Latency (ms)")
    latency_axes[0].set_xticks(positions)
    latency_axes[0].set_xticklabels(labels, rotation=30, ha="right")
    latency_axes[0].grid(axis="y", alpha=0.25)
    latency_axes[0].legend(fontsize=8)
    queue_fields = (("queue_mean_ns", "Queue"), ("service_mean_ns", "Service"))
    width = 0.7 / len(queue_fields)
    for index, (field, label) in enumerate(queue_fields):
        values = [float(result.get(field) or 0.0) / 1_000_000 for result in valid]
        offset = (index - (len(queue_fields) - 1) / 2) * width
        label_bars(latency_axes[1], latency_axes[1].bar(
            [position + offset for position in positions], values,
            width=width, label=label))
    latency_axes[1].set_title("Latency Breakdown")
    latency_axes[1].set_ylabel("Latency (ms)")
    latency_axes[1].set_xticks(positions)
    latency_axes[1].set_xticklabels(labels, rotation=30, ha="right")
    latency_axes[1].grid(axis="y", alpha=0.25)
    latency_axes[1].legend(fontsize=8)
    latency_figure.tight_layout()
    latency_figure.savefig(run_dir / "comparison_latency.png", dpi=160)
    plt.close(latency_figure)

    for name, curve_index, field, title, ylabel in (
            ("utilization", 1, "gpu_allocation_mean", "GPU Allocation", "Allocated GPUs (%)"),
            ("fragmentation", 2, "fragmentation_mean", "GPU Fragmentation", "Free GPU fragmentation (%)")):
        figure, (trend, bars) = plt.subplots(1, 2, figsize=(14, 5))
        for policy, curve in curves.items():
            plot_curve(trend, curve[0], curve[curve_index], policy,
                       color=policy_colors[policy], bins=curve_bins)
        trend.set_title(title + " Over Time")
        trend.set_xlabel("Time after common warmup (ms)")
        trend.set_ylabel(ylabel)
        trend.set_ylim(0, 100)
        trend.grid(alpha=0.2)
        if trend.lines:
            trend.legend(fontsize=9)
        values = [float(result.get(field) or 0.0) * 100 for result in valid]
        label_bars(bars, bars.bar(positions, values,
                                  color=[policy_colors[policy] for policy in labels]))
        bars.set_title("Mean " + title)
        bars.set_ylabel(ylabel)
        bars.set_ylim(0, 110)
        bars.set_yticks(range(0, 101, 20))
        bars.set_xticks(positions)
        bars.set_xticklabels(labels)
        bars.grid(axis="y", alpha=0.2)
        figure.text(0.5, 0.015, curve_note(curve_bins), ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.055, 1, 1))
        figure.savefig(run_dir / ("comparison_" + name + ".png"), dpi=160)
        plt.close(figure)
    return "ok"


def first_peak_occupancy_time(log_path: Path, fraction: float) -> int:
    """Find the first time reaching the selected fraction of peak occupancy."""
    snapshots: List[Tuple[int, int]] = []
    key = None
    busy_count = 0
    snapshot_time = 0
    with log_path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            current_key = (row["event"], row["time_ns"], row["job_id"])
            if key is not None and current_key != key:
                snapshots.append((snapshot_time, busy_count))
                busy_count = 0
            key = current_key
            snapshot_time = int(row["time_ns"])
            busy_count += row["gpu_state"] == "busy"
    if key is not None:
        snapshots.append((snapshot_time, busy_count))
    if not snapshots:
        return 0
    threshold = max(count for _, count in snapshots) * fraction
    return next(time_ns for time_ns, count in snapshots if count >= threshold)


def prepare_policy_manifest(
    source_manifest: Path,
    policy_dir: Path,
    root: Path,
) -> Path:
    fields, rows = read_manifest(source_manifest)
    workload_dir = policy_dir / "workloads"
    workload_dir.mkdir(parents=True, exist_ok=True)
    rewritten: List[Dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        source_workload = resolve_source_path(row["workload"], source_manifest, root)
        if not source_workload.is_file():
            raise RuntimeError(
                "workload for job {} not found: {}".format(
                    row.get("job_id", index), source_workload
                )
            )
        job_id = row.get("job_id", str(index))
        destination = workload_dir / "job_{}_{}".format(job_id, source_workload.name)
        shutil.copy2(source_workload, destination)
        rewritten_row = dict(row)
        rewritten_row["workload"] = str(destination.resolve())
        rewritten.append(rewritten_row)

    output_manifest = policy_dir / "multi_jobs.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rewritten)
    return output_manifest


def run_policy(args: argparse.Namespace, policy: str, run_dir: Path) -> Dict[str, object]:
    policy_dir = run_dir / policy.replace("/", "_")
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "results").mkdir(exist_ok=True)
    manifest = prepare_policy_manifest(args.jobs_file, policy_dir, args.root)
    scheduler_log = policy_dir / "SimAI_scheduler.csv"
    stdout_log = policy_dir / "simai.stdout.log"
    stderr_log = policy_dir / "simai.stderr.log"
    plot_dir = policy_dir / "scheduler_plots"

    command = [
        str(args.simai_bin),
        "-t", str(args.threads),
        "-j", str(manifest),
        "-p", policy,
        "-m", str(args.gpu_memory_bytes),
        "-H", str(args.host_memory_bytes),
        "-L", str(scheduler_log),
        "-n", str(args.network_topology),
        "-c", str(args.simai_config),
    ]
    if args.gpu_resource_file:
        command.extend(["-R", str(args.gpu_resource_file)])

    environment = dict(os.environ)
    environment["AS_SEND_LAT"] = str(args.send_latency)
    environment["AS_NVLS_ENABLE"] = str(args.nvls_enable)
    started = time.time()
    status = "ok"
    exit_code = 0
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=policy_dir,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            exit_code = completed.returncode
            if exit_code != 0:
                status = "failed"
        except OSError as error:
            status = "launch_failed"
            exit_code = 127
            stderr.write(str(error) + "\n")

    result = {
        "policy": policy,
        "status": status,
        "exit_code": exit_code,
        "plot_status": "pending",
        "manifest": str(manifest),
        "scheduler_log": str(scheduler_log),
        "plot_dir": str(plot_dir),
        "duration_seconds": round(time.time() - started, 3),
    }
    return result


def plot_policy(args: argparse.Namespace, policy_dir: Path, warmup_ns: int) -> str:
    scheduler_log = policy_dir / "SimAI_scheduler.csv"
    if not scheduler_log.is_file() or scheduler_log.stat().st_size == 0:
        return "skipped"
    command = [
        sys.executable, str(args.plotter),
        "--log", str(scheduler_log),
        "--output-dir", str(policy_dir / "scheduler_plots"),
        "--warmup-time-ns", str(warmup_ns),
        "--skip-bubbles",
        "--skip-heatmap",
        "--no-memory",
        "--curve-bins", str(getattr(args, "curve_bins", DEFAULT_CURVE_BINS)),
        "--max-latency-jobs", str(getattr(args, "max_latency_jobs", DEFAULT_MAX_LATENCY_JOBS)),
    ]
    if getattr(args, "details", False):
        command.append("--details")
    plot = subprocess.run(command, cwd=policy_dir, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=False)
    (policy_dir / "plot.stdout.log").write_text(plot.stdout, encoding="utf-8")
    return "ok" if plot.returncode == 0 else "failed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SimAI scheduling policies on one identical job trace."
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                        help="comma-separated policies")
    parser.add_argument("--output-root", type=Path,
                        default=root / "scheduler_comparisons")
    parser.add_argument("--run-id", default=None,
                        help="output run name; default is a timestamp")
    parser.add_argument("--simai-bin", type=Path,
                        default=root / "bin" / "SimAI_simulator")
    parser.add_argument("--network-topology", type=Path,
                        default=root / "Spectrum-X_128g_8gps_400Gbps_A100")
    parser.add_argument("--simai-config", type=Path,
                        default=root / "astra-sim-alibabacloud" / "inputs" / "config" / "SimAI.conf")
    parser.add_argument("--gpu-resource-file", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-memory-bytes", type=int, default=0)
    parser.add_argument("--host-memory-bytes", type=int, default=0)
    parser.add_argument("--send-latency", default="3")
    parser.add_argument("--nvls-enable", default="1")
    parser.add_argument("--warmup-fraction", type=float, default=None,
                        help="metadata warmup fraction; policy plots start at time zero")
    parser.add_argument("--warmup-time-ns", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--curve-bins", type=int, default=DEFAULT_CURVE_BINS,
                        help="resource trend time bins; 0 shows raw states (default: 60)")
    parser.add_argument("--max-latency-jobs", type=int, default=DEFAULT_MAX_LATENCY_JOBS,
                        help="slowest jobs shown per policy in latency bars (default: 15)")
    parser.add_argument("--details", action="store_true",
                        help="generate additional per-policy diagnostic plots")
    args = parser.parse_args()

    args.root = root.resolve()
    args.jobs_file = args.jobs_file.resolve()
    args.simai_bin = args.simai_bin.resolve()
    args.network_topology = args.network_topology.resolve()
    args.simai_config = args.simai_config.resolve()
    if args.gpu_resource_file:
        args.gpu_resource_file = args.gpu_resource_file.resolve()
    args.plotter = root / "scripts" / "plot_scheduler_metrics.py"
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    if not policies:
        parser.error("--policies cannot be empty")
    invalid_policies = sorted(set(policies) - set(DEFAULT_POLICIES))
    if invalid_policies:
        parser.error("unsupported policies: " + ", ".join(invalid_policies))
    if not args.jobs_file.is_file():
        parser.error("job manifest not found: {}".format(args.jobs_file))
    for required in (args.simai_bin, args.network_topology, args.simai_config):
        if not required.is_file():
            parser.error("required file not found: {}".format(required))
    if args.threads <= 0 or args.gpu_memory_bytes < 0 or args.host_memory_bytes < 0:
        parser.error("invalid thread or resource argument")
    if args.warmup_fraction is not None and not 0 < args.warmup_fraction <= 1:
        parser.error("--warmup-fraction must be in (0, 1]")
    if args.warmup_time_ns is not None and args.warmup_time_ns < 0:
        parser.error("--warmup-time-ns cannot be negative")
    if args.curve_bins < 0 or args.max_latency_jobs <= 0:
        parser.error("--curve-bins must be nonnegative and --max-latency-jobs positive")

    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with args.jobs_file.open("rb") as source:
        trace_sha256 = hashlib.sha256(source.read()).hexdigest()
    run_metadata = {
        "source_manifest": str(args.jobs_file),
        "source_manifest_sha256": trace_sha256,
        "policies": policies,
        "simai_bin": str(args.simai_bin),
        "network_topology": str(args.network_topology),
        "simai_config": str(args.simai_config),
        "warmup_fraction": args.warmup_fraction,
        "warmup_time_ns": args.warmup_time_ns,
        "curve_bins": args.curve_bins,
        "max_latency_jobs": args.max_latency_jobs,
        "detailed_plots": args.details,
    }
    (run_dir / "comparison_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for policy in policies:
        print("Running policy {} in {}".format(policy, run_dir / policy))
        result = run_policy(args, policy, run_dir)
        if result["status"] == "ok":
            # Generate this policy's plots immediately after its simulation.
            result["plot_status"] = plot_policy(
                args, run_dir / policy.replace("/", "_"),
                args.warmup_time_ns if args.warmup_time_ns is not None else 0,
            )
        results.append(result)
        print("  status={} exit_code={}".format(result["status"], result["exit_code"]))
        if args.stop_on_error and result["status"] != "ok":
            break

    if args.warmup_time_ns is not None:
        common_warmup_ns = args.warmup_time_ns
    elif args.warmup_fraction is not None:
        warmup_times = [
            first_peak_occupancy_time(Path(result["scheduler_log"]),
                                      args.warmup_fraction)
            for result in results
            if result["status"] == "ok" and Path(result["scheduler_log"]).is_file()
        ]
        common_warmup_ns = max(warmup_times, default=0)
    else:
        common_warmup_ns = 0
    for result in results:
        policy_dir = run_dir / str(result["policy"])
        # Policy plots were generated immediately after each simulation. Keep
        # their per-policy full-range metrics and use the common boundary only
        # for the cross-policy comparison metadata/summary.
        result.update(result_metrics(policy_dir))
        (policy_dir / "run_summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    run_metadata["common_warmup_end_ns"] = common_warmup_ns
    (run_dir / "comparison_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    summary = run_dir / "comparison_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as output:
        fields = ["policy", "status", "exit_code", "plot_status", "manifest",
                  "scheduler_log", "plot_dir", "duration_seconds",
                  "completed_jobs_post_warmup", "e2e_mean_ns", "e2e_p50_ns",
                  "e2e_p95_ns", "queue_mean_ns", "service_mean_ns",
                  "gpu_allocation_mean", "fragmentation_mean"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    comparison_plot_status = plot_comparison(results, run_dir, common_warmup_ns,
                                             args.curve_bins)
    run_metadata["comparison_plot_status"] = comparison_plot_status
    (run_dir / "comparison_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    failed = sum(result["status"] != "ok" for result in results)
    print("Comparison finished: {} policies, {} failed".format(len(results), failed))
    print("Results: {}".format(run_dir))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
