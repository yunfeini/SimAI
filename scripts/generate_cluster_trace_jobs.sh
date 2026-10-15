#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRACE_DIR="${TRACE_DIR:-${ROOT_DIR}/example/cluster-trace-v2026-spot-gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRACE_DIR}/}"

exec python3 "${SCRIPT_DIR}/generate_cluster_trace_jobs.py" \
  --trace-dir "${TRACE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --total-gpus "${TOTAL_GPUS:-128}" \
  --gpu-capacity-num 8 \
  --arrival-scale "${ARRIVAL_SCALE:-1.0}" \
  --duration-scale "${DURATION_SCALE:-1.0}" \
  --max-allreduce-bytes "${MAX_ALLREDUCE_BYTES:-1048576}" \
  --overflow-policy "${OVERFLOW_POLICY:-drop}" \
  "$@"
