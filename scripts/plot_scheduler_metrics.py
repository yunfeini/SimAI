#!/usr/bin/env python3
"""Create scheduler and GPU utilization plots from SimAI_scheduler.csv.

This program is intentionally offline: it reads a completed scheduler log and
does not launch SimAI, AICB, or ns-3.
"""

from __future__ import print_function

import argparse
import csv
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Snapshot:
    event: str
    time_ns: int
    job_id: int
    job_name: str
    rows: Dict[int, Dict[str, str]] = field(default_factory=dict)


@dataclass
class JobMetric:
    job_id: int
    job_name: str
    submit_ns: Optional[int] = None
    start_ns: Optional[int] = None
    finish_ns: Optional[int] = None
    gpu_ids: List[int] = field(default_factory=list)
    queue_ns: Optional[int] = None
    service_ns: Optional[int] = None
    e2e_ns: Optional[int] = None


def as_int(value: Optional[str], default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: "
            "python3 -m pip install matplotlib"
        ) from error
    return plt, Patch


def load_snapshots(log_path: Path) -> List[Snapshot]:
    snapshots: "OrderedDict[Tuple[str, int, int, str], Snapshot]" = OrderedDict()
    with log_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"event", "time_ns", "job_id", "gpu_id", "gpu_state"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError("scheduler log misses fields: " + ", ".join(sorted(missing)))
        for row in reader:
            event = row["event"].strip()
            key = (event, as_int(row["time_ns"]), as_int(row["job_id"]), row.get("job_name", ""))
            snapshot = snapshots.get(key)
            if snapshot is None:
                snapshot = Snapshot(key[0], key[1], key[2], key[3])
                snapshots[key] = snapshot
            snapshot.rows[as_int(row["gpu_id"])] = row
    result = list(snapshots.values())
    if not result:
        raise RuntimeError("scheduler log contains no snapshots")
    return result


def first_row(snapshot: Snapshot) -> Dict[str, str]:
    return next(iter(snapshot.rows.values()))


def latency_value(row: Dict[str, str], name: str) -> Optional[int]:
    value = row.get(name, "")
    return as_int(value) if value not in (None, "") else None


def build_job_metrics(snapshots: Iterable[Snapshot]) -> Dict[int, JobMetric]:
    jobs: Dict[int, JobMetric] = {}
    for snapshot in snapshots:
        if snapshot.job_id == 0:
            continue
        metric = jobs.setdefault(snapshot.job_id, JobMetric(snapshot.job_id, snapshot.job_name))
        if snapshot.job_name:
            metric.job_name = snapshot.job_name
        if snapshot.event == "submit":
            metric.submit_ns = snapshot.time_ns
        elif snapshot.event == "schedule":
            metric.start_ns = snapshot.time_ns
            metric.gpu_ids = sorted(
                gpu_id
                for gpu_id, row in snapshot.rows.items()
                if as_int(row.get("gpu_job_id", ""), -1) == snapshot.job_id
            )
        elif snapshot.event == "finish":
            metric.finish_ns = snapshot.time_ns
            row = first_row(snapshot)
            metric.queue_ns = latency_value(row, "queue_latency_ns")
            metric.service_ns = latency_value(row, "service_latency_ns")
            metric.e2e_ns = latency_value(row, "e2e_latency_ns")

    for metric in jobs.values():
        if metric.submit_ns is not None and metric.start_ns is not None and metric.queue_ns is None:
            metric.queue_ns = max(0, metric.start_ns - metric.submit_ns)
        if metric.start_ns is not None and metric.finish_ns is not None and metric.service_ns is None:
            metric.service_ns = max(0, metric.finish_ns - metric.start_ns)
        if metric.submit_ns is not None and metric.finish_ns is not None and metric.e2e_ns is None:
            metric.e2e_ns = max(0, metric.finish_ns - metric.submit_ns)
    return jobs


def time_scale(maximum_ns: int) -> Tuple[float, str]:
    if maximum_ns >= 1_000_000_000:
        return 1_000_000_000.0, "s"
    if maximum_ns >= 1_000_000:
        return 1_000_000.0, "ms"
    if maximum_ns >= 1_000:
        return 1_000.0, "us"
    return 1.0, "ns"


def job_color(plt, job_id: int):
    return plt.get_cmap("tab20")((job_id - 1) % 20)


def save_figure(figure, output_path: Path, dpi: int) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")


def plot_gpu_schedule_timeline(plt, Patch, metrics, gpu_ids, first_ns, last_ns, output_dir, dpi):
    scale, unit = time_scale(max(1, last_ns - first_ns))
    figure, axis = plt.subplots(figsize=(13, max(5, len(gpu_ids) * 0.24)))
    for metric in metrics.values():
        if metric.start_ns is None or not metric.gpu_ids:
            continue
        end_ns = metric.finish_ns if metric.finish_ns is not None else last_ns
        width = max(0, end_ns - metric.start_ns) / scale
        for gpu_id in metric.gpu_ids:
            axis.barh(
                gpu_id,
                width,
                left=(metric.start_ns - first_ns) / scale,
                height=0.72,
                color=job_color(plt, metric.job_id),
                edgecolor="white",
                linewidth=0.3,
            )
    axis.set_title("GPU Allocation Timeline")
    axis.set_xlabel("Simulation time ({})".format(unit))
    axis.set_ylabel("Physical GPU")
    axis.set_yticks(gpu_ids)
    axis.grid(axis="x", alpha=0.25)
    scheduled = [metric for metric in metrics.values() if metric.start_ns is not None]
    if len(scheduled) <= 12:
        legend = [
            Patch(color=job_color(plt, metric.job_id), label="job {}".format(metric.job_id))
            for metric in scheduled
        ]
        axis.legend(handles=legend, loc="upper right", ncol=2, fontsize=8)
    save_figure(figure, output_dir / "gpu_schedule_timeline.png", dpi)
    plt.close(figure)


def plot_job_timeline(plt, Patch, metrics, first_ns, last_ns, output_dir, dpi):
    ordered = sorted(metrics.values(), key=lambda item: (item.submit_ns or 0, item.job_id))
    scale, unit = time_scale(max(1, last_ns - first_ns))
    figure, axis = plt.subplots(figsize=(13, max(5, len(ordered) * 0.32)))
    for index, metric in enumerate(ordered):
        submit_ns = metric.submit_ns if metric.submit_ns is not None else first_ns
        start_ns = metric.start_ns if metric.start_ns is not None else last_ns
        if start_ns > submit_ns:
            axis.barh(index, (start_ns - submit_ns) / scale,
                      left=(submit_ns - first_ns) / scale, height=0.7,
                      color="#aeb7c2", edgecolor="white", linewidth=0.3)
        if metric.start_ns is not None:
            end_ns = metric.finish_ns if metric.finish_ns is not None else last_ns
            axis.barh(index, max(0, end_ns - metric.start_ns) / scale,
                      left=(metric.start_ns - first_ns) / scale, height=0.7,
                      color=job_color(plt, metric.job_id), edgecolor="white", linewidth=0.3)
    axis.set_title("Job Queue and Execution Timeline")
    axis.set_xlabel("Simulation time ({})".format(unit))
    axis.set_ylabel("Job")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels(["{}".format(metric.job_id) for metric in ordered])
    axis.grid(axis="x", alpha=0.25)
    axis.legend(
        handles=[Patch(color="#aeb7c2", label="queued"), Patch(color="#4c78a8", label="running")],
        loc="upper right",
    )
    save_figure(figure, output_dir / "job_queue_execution_timeline.png", dpi)
    plt.close(figure)


def plot_job_latencies(plt, metrics, output_dir, dpi):
    finished = [metric for metric in metrics.values() if metric.finish_ns is not None]
    finished.sort(key=lambda item: item.job_id)
    figure, (axis, histogram) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    if not finished:
        axis.text(0.5, 0.5, "No completed jobs in scheduler log", ha="center", va="center")
        histogram.axis("off")
    else:
        maximum = max(metric.e2e_ns or 0 for metric in finished)
        scale, unit = time_scale(max(1, maximum))
        positions = list(range(len(finished)))
        width = 0.25
        queue = [(metric.queue_ns or 0) / scale for metric in finished]
        service = [(metric.service_ns or 0) / scale for metric in finished]
        e2e = [(metric.e2e_ns or 0) / scale for metric in finished]
        axis.bar([position - width for position in positions], queue, width, label="queue")
        axis.bar(positions, service, width, label="service")
        axis.bar([position + width for position in positions], e2e, width, label="end-to-end")
        axis.set_title("Per-job Latency")
        axis.set_ylabel("Latency ({})".format(unit))
        axis.set_xticks(positions)
        axis.set_xticklabels(["job {}".format(metric.job_id) for metric in finished], rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        histogram.hist(e2e, bins=min(20, max(1, len(e2e))), color="#4c78a8", edgecolor="white")
        histogram.set_title("End-to-end Latency Distribution")
        histogram.set_xlabel("End-to-end latency ({})".format(unit))
        histogram.set_ylabel("Completed jobs")
        histogram.grid(axis="y", alpha=0.25)
    save_figure(figure, output_dir / "job_latencies.png", dpi)
    plt.close(figure)


def snapshot_utilization(snapshot: Snapshot) -> Tuple[int, int, Optional[float], Optional[float]]:
    rows = list(snapshot.rows.values())
    busy = sum(row.get("gpu_state") == "busy" for row in rows)
    gpu_capacity = sum(as_int(row.get("gpu_memory_capacity_bytes")) for row in rows)
    gpu_used = sum(as_int(row.get("gpu_memory_used_bytes")) for row in rows)
    host_capacity = sum(as_int(row.get("host_memory_capacity_bytes")) for row in rows)
    host_used = sum(as_int(row.get("host_memory_used_bytes")) for row in rows)
    gpu_ratio = float(gpu_used) / gpu_capacity if gpu_capacity else None
    host_ratio = float(host_used) / host_capacity if host_capacity else None
    return busy, len(rows), gpu_ratio, host_ratio


def contiguous_runs(values: List[int]) -> List[int]:
    if not values:
        return []
    runs = []
    run_length = 1
    for previous, current in zip(values, values[1:]):
        if current == previous + 1:
            run_length += 1
        else:
            runs.append(run_length)
            run_length = 1
    runs.append(run_length)
    return runs


def snapshot_resource_metrics(snapshot: Snapshot) -> Dict[str, Optional[float]]:
    rows = list(snapshot.rows.items())
    gpu_ids = sorted(gpu_id for gpu_id, _ in rows)
    busy_gpu_ids = sorted(
        gpu_id for gpu_id, row in rows if row.get("gpu_state") == "busy"
    )
    free_gpu_ids = sorted(set(gpu_ids) - set(busy_gpu_ids))
    free_runs = contiguous_runs(free_gpu_ids)
    busy_count = len(busy_gpu_ids)
    total_count = len(gpu_ids)
    free_count = len(free_gpu_ids)
    largest_free_run = max(free_runs, default=0)
    gpu_capacity = sum(as_int(row.get("gpu_memory_capacity_bytes")) for _, row in rows)
    gpu_used = sum(as_int(row.get("gpu_memory_used_bytes")) for _, row in rows)
    host_capacity = sum(as_int(row.get("host_memory_capacity_bytes")) for _, row in rows)
    host_used = sum(as_int(row.get("host_memory_used_bytes")) for _, row in rows)
    active_jobs = len({
        as_int(row.get("gpu_job_id"), -1)
        for _, row in rows
        if row.get("gpu_state") == "busy" and row.get("gpu_job_id", "") != ""
    })
    return {
        "busy_gpu_count": float(busy_count),
        "total_gpu_count": float(total_count),
        "gpu_allocation_ratio": busy_count / total_count if total_count else None,
        "active_job_count": float(active_jobs),
        "free_gpu_count": float(free_count),
        "free_fragment_count": float(max(0, len(free_runs) - 1)),
        "largest_free_run": float(largest_free_run),
        "free_fragmentation_ratio": (
            (free_count - largest_free_run) / free_count if free_count else 0.0
        ),
        "gpu_memory_ratio": gpu_used / gpu_capacity if gpu_capacity else None,
        "host_memory_ratio": host_used / host_capacity if host_capacity else None,
    }


def plot_resource_statistics(plt, snapshots, output_dir, dpi):
    metrics = [snapshot_resource_metrics(snapshot) for snapshot in snapshots]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    chart_data = [
        ([item["gpu_allocation_ratio"] * 100 for item in metrics if item["gpu_allocation_ratio"] is not None],
         "GPU allocation ratio", "Snapshots", "%"),
        ([item["free_fragmentation_ratio"] * 100 for item in metrics],
         "Free-GPU fragmentation ratio", "Snapshots", "%"),
        ([item["free_fragment_count"] for item in metrics],
         "Number of free-GPU fragments", "Snapshots", "Fragments"),
        ([item["active_job_count"] for item in metrics],
         "Active jobs", "Snapshots", "Jobs"),
    ]
    for axis, (values, title, x_label, y_label) in zip(axes.flat, chart_data):
        if values:
            axis.hist(values, bins=min(30, max(5, len(set(values)))), color="#4c78a8", edgecolor="white")
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output_dir / "resource_statistics.png", dpi)
    plt.close(figure)


def plot_resource_timeseries(plt, snapshots, first_ns, last_ns, output_dir, dpi):
    scale, unit = time_scale(max(1, last_ns - first_ns))
    times = [(snapshot.time_ns - first_ns) / scale for snapshot in snapshots]
    metrics = [snapshot_resource_metrics(snapshot) for snapshot in snapshots]
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].step(times, [item["gpu_allocation_ratio"] * 100 for item in metrics], where="post",
                 color="#4c78a8", label="GPU allocation")
    axes[0].set_ylabel("Allocated (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Resource Occupancy and Fragmentation Over Time")
    axes[1].step(times, [item["free_fragment_count"] for item in metrics], where="post",
                 color="#e45756", label="free fragments")
    axes[1].step(times, [item["largest_free_run"] for item in metrics], where="post",
                 color="#54a24b", label="largest contiguous free run")
    axes[1].set_ylabel("GPU count")
    axes[2].step(times, [item["active_job_count"] for item in metrics], where="post",
                 color="#f58518", label="active jobs")
    axes[2].set_ylabel("Jobs")
    axes[2].set_xlabel("Simulation time ({})".format(unit))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    save_figure(figure, output_dir / "resource_occupancy_timeseries.png", dpi)
    plt.close(figure)


def plot_resource_bubbles(plt, snapshots, first_ns, last_ns, output_dir, dpi, max_bubbles):
    scale, unit = time_scale(max(1, last_ns - first_ns))
    points = []
    for snapshot in snapshots:
        time_value = (snapshot.time_ns - first_ns) / scale
        for gpu_id, row in snapshot.rows.items():
            if row.get("gpu_state") != "busy":
                continue
            capacity = as_int(row.get("gpu_memory_capacity_bytes"))
            used = as_int(row.get("gpu_memory_used_bytes"))
            job_memory = as_int(row.get("job_gpu_memory_bytes"))
            occupancy = used / capacity if capacity else None
            size_value = max(1, used, job_memory)
            points.append((time_value, gpu_id, size_value, occupancy, as_int(row.get("gpu_job_id"), -1)))
    if len(points) > max_bubbles:
        stride = math.ceil(len(points) / max_bubbles)
        points = points[::stride]
    figure, axis = plt.subplots(figsize=(14, max(6, len({point[1] for point in points}) * 0.25)))
    if points:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        sizes = [35 if point[3] is None else 35 + 180 * point[3] for point in points]
        colors = [point[4] for point in points]
        scatter = axis.scatter(x_values, y_values, s=sizes, c=colors, cmap="tab20", alpha=0.55,
                               linewidths=0)
        figure.colorbar(scatter, ax=axis, label="GPU job id")
    else:
        axis.text(0.5, 0.5, "No busy GPU samples", ha="center", va="center")
    axis.set_title("Resource Allocation Bubbles by GPU")
    axis.set_xlabel("Simulation time ({})".format(unit))
    axis.set_ylabel("Physical GPU")
    axis.grid(alpha=0.2)
    save_figure(figure, output_dir / "resource_allocation_bubbles.png", dpi)
    plt.close(figure)


def write_resource_timeseries(snapshots, output_dir: Path) -> None:
    fieldnames = [
        "event", "time_ns", "job_id", "job_name", "busy_gpu_count", "total_gpu_count",
        "gpu_allocation_ratio", "active_job_count", "free_gpu_count", "free_fragment_count",
        "largest_free_run", "free_fragmentation_ratio", "gpu_memory_ratio", "host_memory_ratio",
    ]
    with (output_dir / "resource_timeseries.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in snapshots:
            row = {"event": snapshot.event, "time_ns": snapshot.time_ns,
                   "job_id": snapshot.job_id, "job_name": snapshot.job_name}
            row.update(snapshot_resource_metrics(snapshot))
            writer.writerow(row)


def write_resource_summary(snapshots, output_dir: Path) -> None:
    metrics = [snapshot_resource_metrics(snapshot) for snapshot in snapshots]
    fields = [
        "gpu_allocation_ratio", "active_job_count", "free_gpu_count", "free_fragment_count",
        "largest_free_run", "free_fragmentation_ratio", "gpu_memory_ratio", "host_memory_ratio",
    ]
    with (output_dir / "resource_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["metric", "mean", "p50", "p95", "max", "samples"])
        for field_name in fields:
            values = sorted(
                float(item[field_name]) for item in metrics if item[field_name] is not None
            )
            if not values:
                continue
            def percentile(rank: float) -> float:
                index = min(len(values) - 1, int(math.ceil(rank * len(values))) - 1)
                return values[index]
            writer.writerow([
                field_name, sum(values) / len(values), percentile(0.50), percentile(0.95),
                values[-1], len(values),
            ])


def plot_gpu_utilization(plt, snapshots, first_ns, last_ns, output_dir, dpi):
    scale, unit = time_scale(max(1, last_ns - first_ns))
    times = [(snapshot.time_ns - first_ns) / scale for snapshot in snapshots]
    usage = [snapshot_utilization(snapshot) for snapshot in snapshots]
    busy = [item[0] for item in usage]
    total = [item[1] for item in usage]
    gpu_memory = [item[2] for item in usage]
    host_memory = [item[3] for item in usage]
    figure, (allocation_axis, memory_axis) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    allocation_axis.step(times, busy, where="post", color="#4c78a8", label="busy GPUs")
    allocation_axis.step(times, total, where="post", color="#9aa5b1", linestyle="--", label="total GPUs")
    allocation_axis.set_title("GPU Allocation Utilization")
    allocation_axis.set_ylabel("GPU count")
    allocation_axis.set_ylim(bottom=0)
    allocation_axis.grid(alpha=0.25)
    allocation_axis.legend()
    valid_gpu_memory = [(time, ratio) for time, ratio in zip(times, gpu_memory) if ratio is not None]
    valid_host_memory = [(time, ratio) for time, ratio in zip(times, host_memory) if ratio is not None]
    if valid_gpu_memory or valid_host_memory:
        if valid_gpu_memory:
            memory_axis.step([item[0] for item in valid_gpu_memory], [item[1] * 100 for item in valid_gpu_memory],
                             where="post", label="GPU memory")
        if valid_host_memory:
            memory_axis.step([item[0] for item in valid_host_memory], [item[1] * 100 for item in valid_host_memory],
                             where="post", label="host memory")
        memory_axis.set_ylim(0, 100)
        memory_axis.set_ylabel("Resource used (%)")
        memory_axis.legend()
        memory_axis.grid(alpha=0.25)
    else:
        memory_axis.text(0.5, 0.5, "GPU/host memory capacities were not configured", ha="center", va="center")
        memory_axis.set_yticks([])
    memory_axis.set_xlabel("Simulation time ({})".format(unit))
    save_figure(figure, output_dir / "gpu_utilization.png", dpi)
    plt.close(figure)


def plot_gpu_resource_heatmap(plt, snapshots, gpu_ids, output_dir, dpi):
    if not gpu_ids:
        return
    import numpy as np

    busy = np.zeros((len(gpu_ids), len(snapshots)))
    memory = np.full((len(gpu_ids), len(snapshots)), np.nan)
    gpu_index = {gpu_id: index for index, gpu_id in enumerate(gpu_ids)}
    for column, snapshot in enumerate(snapshots):
        for gpu_id, row in snapshot.rows.items():
            index = gpu_index[gpu_id]
            busy[index, column] = 1 if row.get("gpu_state") == "busy" else 0
            capacity = as_int(row.get("gpu_memory_capacity_bytes"))
            if capacity:
                memory[index, column] = as_int(row.get("gpu_memory_used_bytes")) * 100.0 / capacity
    figure, (busy_axis, memory_axis) = plt.subplots(2, 1, figsize=(14, max(7, len(gpu_ids) * 0.28 + 3)))
    busy_image = busy_axis.imshow(busy, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=1)
    busy_axis.set_title("Per-GPU Allocation State by Scheduler Snapshot")
    busy_axis.set_ylabel("Physical GPU")
    busy_axis.set_yticks(range(len(gpu_ids)))
    busy_axis.set_yticklabels(gpu_ids)
    figure.colorbar(busy_image, ax=busy_axis, label="busy")
    if np.isfinite(memory).any():
        memory_image = memory_axis.imshow(memory, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=100)
        figure.colorbar(memory_image, ax=memory_axis, label="GPU memory used (%)")
    else:
        memory_axis.text(0.5, 0.5, "GPU memory capacities were not configured", ha="center", va="center")
    memory_axis.set_title("Per-GPU Memory Use by Scheduler Snapshot")
    memory_axis.set_ylabel("Physical GPU")
    memory_axis.set_yticks(range(len(gpu_ids)))
    memory_axis.set_yticklabels(gpu_ids)
    tick_count = min(10, len(snapshots))
    if tick_count:
        step = max(1, len(snapshots) // tick_count)
        ticks = list(range(0, len(snapshots), step))
        labels = ["{} ns".format(snapshots[index].time_ns) for index in ticks]
        memory_axis.set_xticks(ticks)
        memory_axis.set_xticklabels(labels, rotation=30, ha="right")
    memory_axis.set_xlabel("Scheduler snapshot time")
    save_figure(figure, output_dir / "gpu_resource_heatmap.png", dpi)
    plt.close(figure)


def write_job_metrics(metrics: Dict[int, JobMetric], output_dir: Path) -> None:
    with (output_dir / "job_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["job_id", "job_name", "submit_time_ns", "start_time_ns", "finish_time_ns",
             "gpu_ids", "queue_latency_ns", "service_latency_ns", "e2e_latency_ns"]
        )
        for metric in sorted(metrics.values(), key=lambda item: item.job_id):
            writer.writerow(
                [metric.job_id, metric.job_name, metric.submit_ns, metric.start_ns, metric.finish_ns,
                 ";".join(str(gpu_id) for gpu_id in metric.gpu_ids), metric.queue_ns,
                 metric.service_ns, metric.e2e_ns]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SimAI multi-job scheduler metrics from a CSV log.")
    parser.add_argument("--log", type=Path, default=Path("SimAI_scheduler.csv"), help="scheduler CSV log")
    parser.add_argument("--output-dir", type=Path, default=Path("scheduler_plots"), help="directory for PNG files")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max-bubbles", type=int, default=120000,
                        help="maximum busy-GPU samples in the bubble chart")
    parser.add_argument("--title", default="SimAI Scheduler", help="reserved chart title prefix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        print("error: --dpi must be positive", file=sys.stderr)
        return 2
    if args.max_bubbles <= 0:
        print("error: --max-bubbles must be positive", file=sys.stderr)
        return 2
    log_path = args.log.resolve()
    if not log_path.is_file():
        print("error: scheduler log not found: {}".format(log_path), file=sys.stderr)
        return 2
    try:
        plt, Patch = require_matplotlib()
        snapshots = load_snapshots(log_path)
        snapshots.sort(key=lambda snapshot: snapshot.time_ns)
        metrics = build_job_metrics(snapshots)
        gpu_ids = sorted({gpu_id for snapshot in snapshots for gpu_id in snapshot.rows})
        first_ns = snapshots[0].time_ns
        last_ns = snapshots[-1].time_ns
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_gpu_schedule_timeline(plt, Patch, metrics, gpu_ids, first_ns, last_ns, output_dir, args.dpi)
        plot_job_timeline(plt, Patch, metrics, first_ns, last_ns, output_dir, args.dpi)
        plot_job_latencies(plt, metrics, output_dir, args.dpi)
        plot_gpu_utilization(plt, snapshots, first_ns, last_ns, output_dir, args.dpi)
        plot_gpu_resource_heatmap(plt, snapshots, gpu_ids, output_dir, args.dpi)
        plot_resource_statistics(plt, snapshots, output_dir, args.dpi)
        plot_resource_timeseries(plt, snapshots, first_ns, last_ns, output_dir, args.dpi)
        plot_resource_bubbles(plt, snapshots, first_ns, last_ns, output_dir, args.dpi, args.max_bubbles)
        write_resource_timeseries(snapshots, output_dir)
        write_resource_summary(snapshots, output_dir)
        write_job_metrics(metrics, output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    print("Wrote scheduler plots to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
