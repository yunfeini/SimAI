#!/usr/bin/env python3
"""Generate AICB inference workloads and a SimAI multi-job CSV manifest."""

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


MODEL_OUTPUT_NAMES = {
    "deepseek-671B": "DeepSeek-671B",
    "qwen3-235B": "Qwen3-Moe-235B",
    "qwen3-next-80B": "Qwen3-Next-80B",
}


@dataclass(frozen=True)
class WorkloadConfig:
    model: str
    phase: str
    gpus: int
    tp_size: int
    ep_size: int
    pp_size: int
    seq_length: int
    micro_batch: int


@dataclass(frozen=True)
class Job:
    job_id: int
    submit_time_ns: int
    config: WorkloadConfig


def comma_values(value: str, converter=str) -> List:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    try:
        return [converter(item) for item in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def weighted_choice(rng: random.Random, values: Sequence, weights: Sequence[float]):
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += weight
        if threshold < cumulative:
            return value
    return values[-1]


def parse_weights(text: str, count: int, label: str) -> List[float]:
    if not text:
        return [1.0] * count
    weights = comma_values(text, float)
    if len(weights) != count:
        raise ValueError(f"{label} must contain {count} values")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"{label} must be non-negative and contain a positive value")
    return weights


def generate_jobs(args: argparse.Namespace) -> List[Job]:
    rng = random.Random(args.seed)
    gpu_weights = parse_weights(args.gpu_weights, len(args.gpu_choices), "--gpu-weights")
    model_weights = parse_weights(args.model_weights, len(args.models), "--model-weights")
    phase_weights = parse_weights(args.phase_weights, len(args.phases), "--phase-weights")
    duration_ns = int(args.duration_seconds * 1_000_000_000)
    now_seconds = 0.0
    jobs: List[Job] = []

    while True:
        now_seconds += rng.expovariate(args.arrival_rate)
        submit_time_ns = int(round(now_seconds * 1_000_000_000))
        if jobs:
            submit_time_ns = max(submit_time_ns, jobs[-1].submit_time_ns + 1)
        if submit_time_ns > duration_ns:
            break
        gpus = weighted_choice(rng, args.gpu_choices, gpu_weights)
        model = weighted_choice(rng, args.models, model_weights)
        phase = weighted_choice(rng, args.phases, phase_weights)
        if gpus % args.gpus_per_server != 0:
            raise ValueError(
                f"GPU request {gpus} is not a multiple of --gpus-per-server "
                f"({args.gpus_per_server})"
            )
        if gpus % (args.tp_size * args.pp_size) != 0:
            raise ValueError(
                f"GPU request {gpus} is not divisible by tp_size * pp_size "
                f"({args.tp_size * args.pp_size})"
            )
        ep_size = args.ep_size or gpus // (args.tp_size * args.pp_size)
        config = WorkloadConfig(
            model=model,
            phase=phase,
            gpus=gpus,
            tp_size=args.tp_size,
            ep_size=ep_size,
            pp_size=args.pp_size,
            seq_length=args.seq_length,
            micro_batch=args.micro_batch,
        )
        jobs.append(Job(args.first_job_id + len(jobs), submit_time_ns, config))
    return jobs


def aicb_output_name(config: WorkloadConfig) -> str:
    model_name = MODEL_OUTPUT_NAMES[config.model]
    return (
        f"{model_name}-world_size{config.gpus}-tp{config.tp_size}"
        f"-pp{config.pp_size}-ep{config.ep_size}-bs{config.micro_batch}"
        f"-seq{config.seq_length}-{config.phase}.txt"
    )


def workload_command(args: argparse.Namespace, config: WorkloadConfig) -> List[str]:
    command = [
        "sh",
        str(args.aicb_script),
        "-m",
        config.model,
        "-p",
        config.phase,
        "-s",
        str(config.seq_length),
        "-b",
        str(config.micro_batch),
        "-w",
        str(config.gpus),
        "-t",
        str(config.tp_size),
        "-e",
        str(config.ep_size),
        "-l",
        str(config.pp_size),
        "-r",
        str(args.template_dir),
    ]
    if args.config_file:
        command.extend(["-c", str(args.config_file)])
    if args.aiob_enable:
        command.extend(["-a", "-f", str(args.aiob_loops)])
    return command


def display_command(command: Sequence[str]) -> str:
    return " ".join(f"'{item}'" if " " in item else item for item in command)


def generate_workloads(
    args: argparse.Namespace, jobs: Sequence[Job]
) -> Dict[WorkloadConfig, Path]:
    templates: Dict[WorkloadConfig, Path] = {}
    for config in dict.fromkeys(job.config for job in jobs):
        output_path = args.template_dir / aicb_output_name(config)
        templates[config] = output_path
        if output_path.is_file() and not args.force:
            print(f"[reuse] {output_path}")
            continue
        command = workload_command(args, config)
        if args.dry_run:
            print(f"[dry-run] {display_command(command)}")
            continue
        print(f"[generate] {display_command(command)}")
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(args.aicb_dir) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        # Run from the SimAI-owned output directory.  AICB is imported through
        # PYTHONPATH, so even normal caches/AIOB artifacts are not written into
        # the AICB checkout.
        subprocess.run(command, cwd=args.output_dir, env=environment, check=True)
        if not output_path.is_file():
            raise RuntimeError(f"AICB did not create the expected workload: {output_path}")
    return templates


def manifest_path(path: Path, root: Path) -> str:
    try:
        return "./" + str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_manifest(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    templates: Dict[WorkloadConfig, Path],
) -> None:
    if args.dry_run:
        print(f"[dry-run] would write {len(jobs)} jobs to {args.output_csv}")
        return
    args.workload_dir.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "job_id",
                "workload",
                "gpus",
                "submit_time_ns",
                "name",
                "gpu_memory_bytes",
                "host_memory_bytes",
            ]
        )
        for job in jobs:
            config = job.config
            workload = args.workload_dir / f"job_{job.job_id}_{aicb_output_name(config)}"
            shutil.copyfile(templates[config], workload)
            name = f"{config.model}-{config.phase}-{config.gpus}g-job-{job.job_id}"
            writer.writerow(
                [
                    job.job_id,
                    manifest_path(workload, args.simai_root),
                    config.gpus,
                    job.submit_time_ns,
                    name,
                    args.gpu_memory_bytes,
                    args.host_memory_bytes,
                ]
            )


def build_parser(simai_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Poisson-arrival inference jobs, call the existing AICB "
            "generator, and write a SimAI multi-job CSV."
        )
    )
    parser.add_argument("--duration-seconds", type=float, required=True,
                        help="arrival time window in simulated seconds")
    parser.add_argument("--arrival-rate", type=float, required=True,
                        help="mean arrival rate in jobs per simulated second")
    parser.add_argument("--gpu-choices", type=lambda x: comma_values(x, int),
                        default=[8, 16, 32], help="comma-separated GPU requests")
    parser.add_argument("--gpu-weights", default="",
                        help="optional weights aligned with --gpu-choices")
    parser.add_argument("--models", type=lambda x: comma_values(x, str),
                        default=["deepseek-671B"], help="comma-separated AICB models")
    parser.add_argument("--model-weights", default="",
                        help="optional weights aligned with --models")
    parser.add_argument("--phases", type=lambda x: comma_values(x, str),
                        default=["decode"], help="comma-separated prefill/decode phases")
    parser.add_argument("--phase-weights", default="",
                        help="optional weights aligned with --phases")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=0,
                        help="fixed EP size; 0 chooses world_size/(TP*PP)")
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--gpus-per-server", type=int, default=8)
    parser.add_argument("--gpu-memory-bytes", type=int, default=0,
                        help="per-GPU job memory request written to the manifest")
    parser.add_argument("--host-memory-bytes", type=int, default=0,
                        help="per-GPU host-memory request written to the manifest")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--first-job-id", type=int, default=1)
    parser.add_argument("--aiob-enable", action="store_true")
    parser.add_argument("--aiob-loops", type=int, default=10)
    parser.add_argument("--config", dest="config_file", type=Path, default=None,
                        help="optional AICB inference JSON configuration")
    parser.add_argument("--force", action="store_true",
                        help="regenerate existing AICB workload templates")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned AICB commands without writing files")
    parser.add_argument(
        "--output-dir", type=Path,
        default=simai_root / "example" / "poisson_inference_jobs",
        help="SimAI-side directory for the manifest and workloads",
    )
    parser.add_argument(
        "--aicb-dir", type=Path, default=simai_root / "aicb",
        help="existing AICB checkout; the generator does not write into it",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if not math.isfinite(args.arrival_rate) or args.arrival_rate <= 0:
        parser.error("--arrival-rate must be positive")
    for name in ("tp_size", "pp_size", "gpus_per_server", "seq_length", "micro_batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ep_size < 0:
        parser.error("--ep-size cannot be negative")
    if args.gpu_memory_bytes < 0 or args.host_memory_bytes < 0:
        parser.error("memory requests cannot be negative")
    if any(gpus <= 0 for gpus in args.gpu_choices):
        parser.error("--gpu-choices values must be positive")
    invalid_models = sorted(set(args.models) - set(MODEL_OUTPUT_NAMES))
    if invalid_models:
        parser.error("unsupported models: " + ", ".join(invalid_models))
    invalid_phases = sorted(set(args.phases) - {"prefill", "decode"})
    if invalid_phases:
        parser.error("unsupported phases: " + ", ".join(invalid_phases))
    try:
        parse_weights(args.gpu_weights, len(args.gpu_choices), "--gpu-weights")
        parse_weights(args.model_weights, len(args.models), "--model-weights")
        parse_weights(args.phase_weights, len(args.phases), "--phase-weights")
    except ValueError as error:
        parser.error(str(error))


def main() -> int:
    simai_root = Path(__file__).resolve().parents[1]
    parser = build_parser(simai_root)
    args = parser.parse_args()
    validate_args(parser, args)
    args.simai_root = simai_root
    args.aicb_dir = args.aicb_dir.resolve()
    args.aicb_script = args.aicb_dir / "scripts" / "inference_workload_with_aiob.sh"
    if args.config_file:
        args.config_file = args.config_file.resolve()
    args.output_dir = args.output_dir.resolve()
    args.template_dir = args.output_dir / "templates"
    args.workload_dir = args.output_dir / "workloads"
    args.output_csv = args.output_dir / "multi_jobs.csv"
    if not args.aicb_script.is_file():
        parser.error(f"AICB generator not found: {args.aicb_script}")
    if not args.dry_run:
        args.template_dir.mkdir(parents=True, exist_ok=True)

    try:
        jobs = generate_jobs(args)
        if not jobs:
            print(
                "No arrivals were sampled in the requested time window. "
                "Increase --duration-seconds or --arrival-rate.",
                file=sys.stderr,
            )
            return 2
        templates = generate_workloads(args, jobs)
        write_manifest(args, jobs, templates)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    action = "Planned" if args.dry_run else "Generated"
    print(f"{action} {len(jobs)} jobs in {args.output_csv}")
    print(f"First arrival: {jobs[0].submit_time_ns} ns")
    print(f"Last arrival:  {jobs[-1].submit_time_ns} ns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
