#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-../fetal_dataset}"
GROUP_ARGS=()
if [[ -n "${GROUPS_CSV:-}" ]]; then GROUP_ARGS=(--groups-csv "$GROUPS_CSV"); fi
python -m curvrevision.splits --data-root "$DATA_ROOT" --output outputs/splits --outer-folds 5 --inner-val-fraction 0.20 --split-seed 2026 "${GROUP_ARGS[@]}"
