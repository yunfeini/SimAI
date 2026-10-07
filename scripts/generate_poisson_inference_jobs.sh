#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python3 "${SCRIPT_DIR}/generate_poisson_inference_jobs.py" \
  --duration-seconds "${DURATION_SECONDS:-60}" \
  --arrival-rate "${ARRIVAL_RATE:-0.1}" \
  --gpu-choices "${GPU_CHOICES:-8,16,32}" \
  --models "${MODELS:-deepseek-671B}" \
  --phases "${PHASES:-decode}" \
  --gpus-per-server "${GPUS_PER_SERVER:-8}" \
  --output-dir "${OUTPUT_DIR:-${ROOT_DIR}/example/poisson_inference_jobs}" \
  "$@"
