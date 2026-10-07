#!/usr/bin/env python3
"""Generate Poisson-arrival AICB training workloads for SimAI multi-job."""

import argparse
import csv
import hashlib
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


SUPPORTED_MODELS = {
    "175",
    "22",
    "13",
    "7",
    "405",
    "moe",
    "deepseek671",
    "deepseek236",
    "deepseek16",
    "deepseek",
}

MODEL_TP_OVERRIDES = {
    "175": 8,
    "22": 8,
    "7": 4,
    "moe": 4,
    "deepseek671": 1,
    "deepseek236": 1,
    "deepseek16": 1,
}

MODEL_EXPERT_DEFAULTS = {
    "moe": 8,
    "deepseek671": 256,
    "deepseek236": 160,
    "deepseek16": 64,
}


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    gpus: int
    tp_size: int
    pp_size: int
    ep_size: int
    num_experts: int
    moe_enable: bool


@dataclass(frozen=True)
class Job:
    job_id: int
    submit_time_ns: int
    config: TrainingConfig


def comma_values(value: str, converter=str) -> List:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    try:
        return [converter(item) for item in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_weights(text: str, count: int, label: str) -> List[float]:
    if not text:
        return [1.0] * count
    weights = comma_values(text, float)
    if len(weights) != count:
        raise ValueError(f"{label} must contain {count} values")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"{label} must be non-negative and contain a positive value")
    return weights


def weighted_choice(rng: random.Random, values: Sequence, weights: Sequence[float]):
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += weight
        if threshold < cumulative:
            return value
    return values[-1]


def effective_tp(model: str, requested_tp: int) -> int:
    return MODEL_TP_OVERRIDES.get(model, requested_tp)


def effective_experts(model: str, requested_experts: int) -> int:
    if requested_experts > 0:
        return requested_experts
    return MODEL_EXPERT_DEFAULTS.get(model, 1)


def is_moe(model: str, force_moe: bool) -> bool:
    return force_moe or model == "moe" or model.startswith("deepseek")


def generate_jobs(args: argparse.Namespace) -> List[Job]:
    rng = random.Random(args.seed)
    gpu_weights = parse_weights(args.gpu_weights, len(args.gpu_choices), "--gpu-weights")
    model_weights = parse_weights(args.model_weights, len(args.models), "--model-weights")
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
        tp_size = effective_tp(model, args.tp_size)
        moe_enable = is_moe(model, args.moe_enable)
        num_experts = effective_experts(model, args.num_experts)
        parallel_product = tp_size * args.pp_size
        if gpus % args.gpus_per_server != 0:
            raise ValueError(
                f"GPU request {gpus} is not a multiple of --gpus-per-server "
                f"({args.gpus_per_server})"
            )
        if gpus % parallel_product != 0:
            raise ValueError(
                f"GPU request {gpus} is not divisible by effective TP * PP "
                f"({tp_size} * {args.pp_size}) for model {model}"
            )
        dp_size = gpus // parallel_product
        if dp_size % args.ep_size != 0:
            raise ValueError(
                f"data-parallel size {dp_size} is not divisible by EP size "
                f"{args.ep_size} for model {model} on {gpus} GPUs"
            )
        if args.ep_size > 1 and num_experts % args.ep_size != 0:
            raise ValueError(
                f"num_experts {num_experts} is not divisible by EP size {args.ep_size}"
            )
        if args.global_batch < dp_size * args.micro_batch:
            raise ValueError(
                f"global batch {args.global_batch} is smaller than DP * micro batch "
                f"({dp_size} * {args.micro_batch})"
            )

        config = TrainingConfig(
            model=model,
            gpus=gpus,
            tp_size=tp_size,
            pp_size=args.pp_size,
            ep_size=args.ep_size,
            num_experts=num_experts,
            moe_enable=moe_enable,
        )
        jobs.append(Job(args.first_job_id + len(jobs), submit_time_ns, config))
    return jobs


def workload_slug(args: argparse.Namespace, config: TrainingConfig) -> str:
    signature = (
        config,
        args.frame,
        args.global_batch,
        args.micro_batch,
        args.num_layers,
        args.seq_length,
        args.hidden_size,
        args.epoch_num,
        args.num_attention_heads,
        args.max_position_embeddings,
        args.ffn_hidden_size,
        args.gpu_type,
        args.aiob_enable,
        args.use_flash_attn,
        args.swiglu,
        args.sp_enable,
        str(args.comp_filepath) if args.comp_filepath else "",
        args.moe_router_topk,
        args.moe_grouped_gemm,
        args.recompute_activations,
    )
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return (
        f"{config.model}-world{config.gpus}-tp{config.tp_size}-pp{config.pp_size}"
        f"-ep{config.ep_size}-gbs{args.global_batch}-mbs{args.micro_batch}"
        f"-seq{args.seq_length}-epochs{args.epoch_num}-{digest}"
    )


def append_value(command: List[str], option: str, value) -> None:
    if value is not None:
        command.extend([option, str(value)])


def workload_command(
    args: argparse.Namespace, config: TrainingConfig
) -> List[str]:
    command = [
        "sh",
        str(args.aicb_script),
        "-m",
        config.model,
        "--frame",
        args.frame,
        "--world_size",
        str(config.gpus),
        "--tensor_model_parallel_size",
        str(config.tp_size),
        "--pipeline_model_parallel",
        str(config.pp_size),
        "--expert_model_parallel_size",
        str(config.ep_size),
        "--global_batch",
        str(args.global_batch),
        "--micro_batch",
        str(args.micro_batch),
        "--num_layers",
        str(args.num_layers),
        "--seq_length",
        str(args.seq_length),
        "--hidden_size",
        str(args.hidden_size),
        "--epoch_num",
        str(args.epoch_num),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--max_position_embeddings",
        str(args.max_position_embeddings),
        "--num_experts",
        str(config.num_experts),
        "--gpu_type",
        args.gpu_type,
    ]
    append_value(command, "--ffn_hidden_size", args.ffn_hidden_size)
    append_value(command, "--comp_filepath", args.comp_filepath)
    append_value(command, "--moe_router_topk", args.moe_router_topk)
    if args.aiob_enable:
        command.append("--aiob_enable")
    if args.use_flash_attn:
        command.append("--use_flash_attn")
    if args.swiglu:
        command.append("--swiglu")
    if args.sp_enable or config.moe_enable:
        command.append("--sp")
    if config.moe_enable:
        command.append("--moe_enable")
    if args.moe_grouped_gemm:
        command.append("--moe_grouped_gemm")
    if args.recompute_activations:
        command.append("--recompute_activations")
    return command


def display_command(command: Sequence[str]) -> str:
    return " ".join(f"'{item}'" if " " in item else item for item in command)


def generate_workloads(
    args: argparse.Namespace, jobs: Sequence[Job]
) -> Dict[TrainingConfig, Path]:
    templates: Dict[TrainingConfig, Path] = {}
    for config in dict.fromkeys(job.config for job in jobs):
        slug = workload_slug(args, config)
        template = args.template_dir / f"{slug}.txt"
        templates[config] = template
        if template.is_file() and not args.force:
            print(f"[reuse] {template}")
            continue

        staging_dir = args.staging_dir / slug
        generated_dir = staging_dir / "results" / "workload"
        command = workload_command(args, config)
        if args.dry_run:
            print(f"[dry-run] cwd={staging_dir} {display_command(command)}")
            continue

        generated_dir.mkdir(parents=True, exist_ok=True)
        for previous_workload in generated_dir.glob("*.txt"):
            previous_workload.unlink()
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(args.aicb_dir) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        print(f"[generate] {display_command(command)}")
        subprocess.run(command, cwd=staging_dir, env=environment, check=True)
        generated = sorted(generated_dir.glob("*.txt"))
        if len(generated) != 1:
            raise RuntimeError(
                f"expected one AICB workload in {generated_dir}, found {len(generated)}"
            )
        args.template_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated[0], template)
    return templates


def manifest_path(path: Path, root: Path) -> str:
    try:
        return "./" + str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_manifest(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    templates: Dict[TrainingConfig, Path],
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
            slug = workload_slug(args, job.config)
            workload = args.workload_dir / f"job_{job.job_id}_{slug}.txt"
            shutil.copyfile(templates[job.config], workload)
            writer.writerow(
                [
                    job.job_id,
                    manifest_path(workload, args.simai_root),
                    job.config.gpus,
                    job.submit_time_ns,
                    f"training-{slug}-job-{job.job_id}",
                    args.gpu_memory_bytes,
                    args.host_memory_bytes,
                ]
            )


def build_parser(simai_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Poisson-arrival training jobs, invoke the existing AICB "
            "Megatron workload script, and write a SimAI multi-job CSV."
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
                        default=["13"], help="comma-separated AICB model sizes")
    parser.add_argument("--model-weights", default="",
                        help="optional weights aligned with --models")
    parser.add_argument("--frame", default="Megatron")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="TP size unless the selected AICB model overrides it")
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--epoch-num", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=40)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=None)
    parser.add_argument("--gpu-type", default="None")
    parser.add_argument("--aiob-enable", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--swiglu", action="store_true")
    parser.add_argument("--sp-enable", action="store_true")
    parser.add_argument("--comp-filepath", type=Path, default=None)
    parser.add_argument("--moe-enable", action="store_true")
    parser.add_argument("--moe-router-topk", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=0,
                        help="0 selects the model default")
    parser.add_argument("--moe-grouped-gemm", action="store_true")
    parser.add_argument("--recompute-activations", action="store_true")
    parser.add_argument("--gpus-per-server", type=int, default=8)
    parser.add_argument("--gpu-memory-bytes", type=int, default=0)
    parser.add_argument("--host-memory-bytes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--first-job-id", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=simai_root / "example" / "poisson_training_jobs",
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
    positive_fields = (
        "tp_size", "pp_size", "ep_size", "global_batch", "micro_batch",
        "num_layers", "seq_length", "hidden_size", "epoch_num",
        "num_attention_heads", "max_position_embeddings", "gpus_per_server",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_experts < 0:
        parser.error("--num-experts cannot be negative")
    if args.gpu_memory_bytes < 0 or args.host_memory_bytes < 0:
        parser.error("memory requests cannot be negative")
    if any(gpus <= 0 for gpus in args.gpu_choices):
        parser.error("--gpu-choices values must be positive")
    invalid_models = sorted(set(args.models) - SUPPORTED_MODELS)
    if invalid_models:
        parser.error("unsupported models: " + ", ".join(invalid_models))
    try:
        parse_weights(args.gpu_weights, len(args.gpu_choices), "--gpu-weights")
        parse_weights(args.model_weights, len(args.models), "--model-weights")
    except ValueError as error:
        parser.error(str(error))


def main() -> int:
    simai_root = Path(__file__).resolve().parents[1]
    parser = build_parser(simai_root)
    args = parser.parse_args()
    validate_args(parser, args)
    args.simai_root = simai_root
    args.aicb_dir = args.aicb_dir.resolve()
    args.aicb_script = args.aicb_dir / "scripts" / "megatron_workload_with_aiob.sh"
    args.output_dir = args.output_dir.resolve()
    args.template_dir = args.output_dir / "templates"
    args.workload_dir = args.output_dir / "workloads"
    args.staging_dir = args.output_dir / ".aicb_staging"
    args.output_csv = args.output_dir / "multi_jobs.csv"
    if args.comp_filepath:
        args.comp_filepath = args.comp_filepath.resolve()
    if not args.aicb_script.is_file():
        parser.error(f"AICB generator not found: {args.aicb_script}")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"{action} {len(jobs)} training jobs in {args.output_csv}")
    print(f"First arrival: {jobs[0].submit_time_ns} ns")
    print(f"Last arrival:  {jobs[-1].submit_time_ns} ns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
