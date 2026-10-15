#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "${PLOT_DETAILS:-0}" = "1" ]; then
  set -- --details "$@"
fi
if [ -n "${COMPARISON_RUN_ID:-}" ]; then
  set -- --run-id "${COMPARISON_RUN_ID}" "$@"
fi
if [ -n "${GPU_RESOURCE_FILE:-}" ]; then
  set -- --gpu-resource-file "${GPU_RESOURCE_FILE}" "$@"
fi
if [ -n "${WARMUP_FRACTION:-}" ]; then
  set -- --warmup-fraction "${WARMUP_FRACTION}" "$@"
fi
if [ -n "${WARMUP_TIME_NS:-}" ]; then
  set -- --warmup-time-ns "${WARMUP_TIME_NS}" "$@"
fi

exec python3 "${SCRIPT_DIR}/compare_scheduler_policies.py" \
  --jobs-file "${JOBS_FILE:-${ROOT_DIR}/example/multi_jobs.csv}" \
  --policies "${POLICIES:-fcfs,first-fit,best-fit,fgd,crux}" \
  --output-root "${COMPARISON_ROOT:-${ROOT_DIR}/scheduler_comparisons}" \
  --simai-bin "${SIMAI_BIN:-${ROOT_DIR}/bin/SimAI_simulator}" \
  --network-topology "${NETWORK_TOPO:-${ROOT_DIR}/Spectrum-X_128g_8gps_400Gbps_A100}" \
  --simai-config "${SIMAI_CONF:-${ROOT_DIR}/astra-sim-alibabacloud/inputs/config/SimAI.conf}" \
  --threads "${THREADS:-16}" \
  --gpu-memory-bytes "${GPU_MEMORY_BYTES:-0}" \
  --host-memory-bytes "${HOST_MEMORY_BYTES:-0}" \
  --send-latency "${AS_SEND_LAT:-3}" \
  --nvls-enable "${AS_NVLS_ENABLE:-1}" \
  --curve-bins "${PLOT_CURVE_BINS:-60}" \
  --max-latency-jobs "${PLOT_MAX_LATENCY_JOBS:-15}" \
  "$@"
