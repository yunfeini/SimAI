#!/usr/bin/env python3
"""Regenerate existing policy plots and comparison figures without simulation."""

import argparse
import json
from pathlib import Path
import sys

from compare_scheduler_policies import (
    DEFAULT_POLICIES, first_peak_occupancy_time, plot_comparison,
    plot_policy, result_metrics,
)
from scheduler_plot_style import DEFAULT_CURVE_BINS, DEFAULT_MAX_LATENCY_JOBS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="existing comparison directory, e.g. scheduler_comparisons/test")
    parser.add_argument("--policies", default=None,
                        help="comma-separated subset; default discovers logs in the run")
    parser.add_argument("--curve-bins", type=int, default=DEFAULT_CURVE_BINS)
    parser.add_argument("--max-latency-jobs", type=int, default=DEFAULT_MAX_LATENCY_JOBS)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--warmup-time-ns", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=None)
    args = parser.parse_args()
    if args.curve_bins < 0 or args.max_latency_jobs <= 0:
        parser.error("--curve-bins must be nonnegative and --max-latency-jobs positive")
    if args.warmup_time_ns is not None and args.warmup_time_ns < 0:
        parser.error("--warmup-time-ns cannot be negative")
    if args.warmup_fraction is not None and not 0 < args.warmup_fraction <= 1:
        parser.error("--warmup-fraction must be in (0, 1]")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error("comparison directory not found: {}".format(run_dir))
    policies = (list(dict.fromkeys(p.strip() for p in args.policies.split(",") if p.strip()))
                if args.policies is not None else
                [p for p in DEFAULT_POLICIES if (run_dir / p / "SimAI_scheduler.csv").is_file()])
    if not policies or any(p not in DEFAULT_POLICIES for p in policies):
        parser.error("select supported policies with existing scheduler logs")
    for policy in policies:
        path = run_dir / policy / "SimAI_scheduler.csv"
        if not path.is_file() or not path.stat().st_size:
            parser.error("scheduler log missing or empty: {}".format(path))
    args.plotter = Path(__file__).resolve().with_name("plot_scheduler_metrics.py")
    warmup = args.warmup_time_ns or 0
    if args.warmup_time_ns is None and args.warmup_fraction is not None:
        warmup = max(first_peak_occupancy_time(
            run_dir / policy / "SimAI_scheduler.csv", args.warmup_fraction)
            for policy in policies)
    results = []
    for policy in policies:
        policy_dir = run_dir / policy
        print("Replotting {} from existing scheduler log".format(policy), flush=True)
        status = plot_policy(args, policy_dir, warmup)
        result = {"policy": policy, "plot_status": status}
        if status == "ok":
            result.update(result_metrics(policy_dir))
        results.append(result)
        print("  plot_status={}".format(status), flush=True)
    comparison_status = plot_comparison(results, run_dir, warmup, args.curve_bins)
    metadata = {"mode": "offline_replot", "warmup_time_ns": warmup,
                "curve_bins": args.curve_bins, "max_latency_jobs": args.max_latency_jobs,
                "details": args.details, "policies": results,
                "comparison_plot_status": comparison_status}
    (run_dir / "replot_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Comparison plots: {} ({})".format(run_dir, comparison_status))
    return int(comparison_status != "ok" or any(r["plot_status"] != "ok" for r in results))


if __name__ == "__main__":
    sys.exit(main())
