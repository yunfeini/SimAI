#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIMAI_BIN="${SIMAI_BIN:-${ROOT_DIR}/bin/SimAI_simulator}"
JOBS_FILE="${JOBS_FILE:-${ROOT_DIR}/example/multi_jobs.csv}"
NETWORK_TOPO="${NETWORK_TOPO:-${ROOT_DIR}/Spectrum-X_128g_8gps_400Gbps_A100}"
SIMAI_CONF="${SIMAI_CONF:-${ROOT_DIR}/astra-sim-alibabacloud/inputs/config/SimAI.conf}"
THREADS="${THREADS:-16}"
POLICY="${POLICY:-first-fit}"
GPU_MEMORY_BYTES="${GPU_MEMORY_BYTES:-0}"
HOST_MEMORY_BYTES="${HOST_MEMORY_BYTES:-0}"
GPU_RESOURCE_FILE="${GPU_RESOURCE_FILE:-}"
SCHEDULER_LOG="${SCHEDULER_LOG:-${ROOT_DIR}/SimAI_scheduler.csv}"

if [[ -n "${POLICIES:-}" ]]; then
  exec sh "${SCRIPT_DIR}/compare_scheduler_policies.sh" "$@"
fi

if [[ ! -x "${SIMAI_BIN}" ]]; then
  echo "SimAI binary not found or not executable: ${SIMAI_BIN}" >&2
  echo "Build it first or set SIMAI_BIN=/path/to/SimAI_simulator." >&2
  exit 1
fi

args=(
  -t "${THREADS}"
  -j "${JOBS_FILE}"
  -p "${POLICY}"
  -m "${GPU_MEMORY_BYTES}"
  -H "${HOST_MEMORY_BYTES}"
  -L "${SCHEDULER_LOG}"
  -n "${NETWORK_TOPO}"
  -c "${SIMAI_CONF}"
)

if [[ -n "${GPU_RESOURCE_FILE}" ]]; then
  args+=( -R "${GPU_RESOURCE_FILE}" )
fi

cd "${ROOT_DIR}"
echo "Running scheduled SimAI jobs from ${JOBS_FILE}"

AS_SEND_LAT="${AS_SEND_LAT:-3}" \
AS_NVLS_ENABLE="${AS_NVLS_ENABLE:-1}" \
  "${SIMAI_BIN}" "${args[@]}"
