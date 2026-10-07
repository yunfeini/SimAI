#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python3 "${SCRIPT_DIR}/plot_scheduler_metrics.py" \
  --log "${SCHEDULER_LOG:-${ROOT_DIR}/SimAI_scheduler.csv}" \
  --output-dir "${PLOT_DIR:-${ROOT_DIR}/scheduler_plots}" \
  "$@"
