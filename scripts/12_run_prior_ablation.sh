#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib_run.sh"
read -ra SEED_LIST <<< "${SEEDS:-42 123 2026}"
PRIORS=(none sobel laplacian gaussian_gradient log)
for seed in "${SEED_LIST[@]}"; do for fold in 0 1 2 3 4; do for prior in "${PRIORS[@]}"; do run_one prior prior_ablation stage1 "$prior" shallow "$seed" "$fold"; done; done; done
