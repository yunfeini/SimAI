#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "${PLOT_DETAILS:-0}" = "1" ]; then
  set -- --details "$@"
fi
if [ -n "${POLICIES:-}" ]; then
  set -- --policies "${POLICIES}" "$@"
fi
if [ -n "${WARMUP_FRACTION:-}" ]; then
  set -- --warmup-fraction "${WARMUP_FRACTION}" "$@"
fi
if [ -n "${WARMUP_TIME_NS:-}" ]; then
  set -- --warmup-time-ns "${WARMUP_TIME_NS}" "$@"
fi
if [ -n "${COMPARISON_DIR:-}" ]; then
  set -- --run-dir "${COMPARISON_DIR}" "$@"
elif [ -n "${COMPARISON_RUN_ID:-}" ]; then
  set -- --run-dir "${COMPARISON_ROOT:-${ROOT_DIR}/scheduler_comparisons}/${COMPARISON_RUN_ID}" "$@"
fi

exec python3 "${SCRIPT_DIR}/plot_scheduler_comparison.py" \
  --curve-bins "${PLOT_CURVE_BINS:-60}" \
  --max-latency-jobs "${PLOT_MAX_LATENCY_JOBS:-15}" \
  "$@"
