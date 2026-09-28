#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib_run.sh"
read -ra SEED_LIST <<< "${SEEDS:-42 123 2026}"
MODELS=(full_standard full_log_attention_only full_log_shallow)
for seed in "${SEED_LIST[@]}"; do
  for fold in 0 1 2 3 4; do
    for model in "${MODELS[@]}"; do
      src="${RUN_ROOT}/core/${model}/budget_stage1/selection_best_dice/seed_${seed}/fold_${fold}"
      dst="${RUN_ROOT}/core/${model}/budget_stage1/selection_constrained_hd/seed_${seed}/fold_${fold}"
      [[ -f "$src/DEVELOP_COMPLETE" && -f "$src/selection.json" ]] || { echo "Run scripts/10_run_core_stage1.sh first: $src" >&2; exit 2; }
      mkdir -p "$dst"
      if [[ ! -f "$dst/DEVELOP_COMPLETE" ]]; then
        cp "$src/selection.json" "$dst/selection.json"
        cp "$src/develop_stage1_history.csv" "$dst/develop_stage1_history.csv"
        printf 'selection/history reused from %s\n' "$src" > "$dst/SELECTION_REUSED_FROM.txt"
        printf 'inner validation only; reused deterministic development history\n' > "$dst/DEVELOP_COMPLETE"
      fi
      run_one core "$model" stage1 log shallow "$seed" "$fold" constrained_hd
    done
  done
done
