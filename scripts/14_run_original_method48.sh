#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib_run.sh"
read -ra SEED_LIST <<< "${SEEDS:-42 123 2026}"
for seed in "${SEED_LIST[@]}"; do for fold in 0 1 2 3 4; do reuse_stage1_development core full_log_shallow log shallow "$seed" "$fold" original48; run_one core full_log_shallow original48 log shallow "$seed" "$fold"; done; done
