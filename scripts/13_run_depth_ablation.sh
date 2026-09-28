#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib_run.sh"
read -ra SEED_LIST <<< "${SEEDS:-42 123 2026}"
DEPTHS=(shallow middle deep all)
for seed in "${SEED_LIST[@]}"; do for fold in 0 1 2 3 4; do for depth in "${DEPTHS[@]}"; do run_one depth depth_ablation stage1 log "$depth" "$seed" "$fold"; done; done; done
