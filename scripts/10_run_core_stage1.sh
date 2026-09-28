#!/usr/bin/env bash
set -euo pipefail
export KEEP_DEVELOP=1
source "$(dirname "$0")/lib_run.sh"
read -ra SEED_LIST <<< "${SEEDS:-42 123 2026}"
MODELS=(unet attention_unet resunet attention_resunet full_standard full_log_attention_only full_log_shallow)
for seed in "${SEED_LIST[@]}"; do for fold in 0 1 2 3 4; do for model in "${MODELS[@]}"; do run_one core "$model" stage1 log shallow "$seed" "$fold"; done; done; done
