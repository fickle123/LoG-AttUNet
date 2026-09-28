#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../fetal_dataset}"
SPLIT_ROOT="${SPLIT_ROOT:-outputs/splits}"
RUN_ROOT="${RUN_ROOT:-outputs/runs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-40}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
AMP_FLAG="${AMP_FLAG:---amp}"
KEEP_DEVELOP="${KEEP_DEVELOP:-0}"

run_one() {
  local experiment="$1" model="$2" budget="$3" prior="$4" depth="$5" seed="$6" fold="$7" selection="${8:-best_dice}"
  local label="$model"
  [[ "$model" == "prior_ablation" ]] && label="prior_${prior}"
  [[ "$model" == "depth_ablation" ]] && label="depth_${depth}"
  local run_dir="${RUN_ROOT}/${experiment}/${label}/budget_${budget}/selection_${selection}/seed_${seed}/fold_${fold}"
  local log_dir="outputs/logs/${experiment}/${label}/budget_${budget}/selection_${selection}/seed_${seed}"
  mkdir -p "$log_dir"
  local log="${log_dir}/fold_${fold}.log"
  local common=(
    --data-root "$DATA_ROOT" --split-json "${SPLIT_ROOT}/fold_${fold}.json" --output-root "$RUN_ROOT"
    --experiment "$experiment" --model "$model" --budget "$budget" --selection-rule "$selection" --prior-kind "$prior" --prior-depth "$depth"
    --seed "$seed" --image-size "$IMAGE_SIZE" --base-channels "$BASE_CHANNELS" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
  )
  if [[ ! -f "${run_dir}/DEVELOP_COMPLETE" ]]; then
    echo "[$(date -Is)] DEVELOP $experiment $label $budget seed=$seed fold=$fold" | tee -a "$log"
    python -m curvrevision.train_cv "${common[@]}" --phase develop $AMP_FLAG 2>&1 | tee -a "$log"
  else echo "[SKIP] develop complete: $run_dir" | tee -a "$log"; fi
  if [[ ! -f "${run_dir}/REFIT_COMPLETE" ]]; then
    echo "[$(date -Is)] REFIT $experiment $label $budget seed=$seed fold=$fold" | tee -a "$log"
    local keep_flag=()
    [[ "$KEEP_DEVELOP" == "1" ]] && keep_flag=(--keep-develop-checkpoints)
    python -m curvrevision.train_cv "${common[@]}" --phase refit $AMP_FLAG "${keep_flag[@]}" 2>&1 | tee -a "$log"
  else echo "[SKIP] refit complete: $run_dir" | tee -a "$log"; fi
  echo "[HOLD] outer-test is intentionally NOT run here. Complete all formal refits, freeze the protocol, then run scripts/19_run_all_outer_tests.sh" | tee -a "$log"
}


reuse_stage1_development() {
  local experiment="$1" model="$2" prior="$3" depth="$4" seed="$5" fold="$6" target_budget="$7"
  local label="$model"
  [[ "$model" == "prior_ablation" ]] && label="prior_${prior}"
  [[ "$model" == "depth_ablation" ]] && label="depth_${depth}"
  local src="${RUN_ROOT}/${experiment}/${label}/budget_stage1/selection_best_dice/seed_${seed}/fold_${fold}"
  local dst="${RUN_ROOT}/${experiment}/${label}/budget_${target_budget}/selection_best_dice/seed_${seed}/fold_${fold}"
  [[ -f "$src/DEVELOP_COMPLETE" ]] || { echo "Missing source development: $src" >&2; return 2; }
  [[ -f "$src/develop_stage1_history.csv" && -f "$src/develop_stage1_best_dice.pth" ]] || { echo "Source was not retained with KEEP_DEVELOP=1: $src" >&2; return 2; }
  mkdir -p "$dst"
  if [[ ! -f "$dst/develop_stage1_history.csv" ]]; then
    cp "$src/develop_stage1_history.csv" "$dst/develop_stage1_history.csv"
    cp "$src/develop_stage1_best_dice.pth" "$dst/develop_stage1_best_dice.pth"
    {
      echo "source=$src"
      echo "history_sha256=$(sha256sum "$src/develop_stage1_history.csv" | awk '{print $1}')"
      echo "checkpoint_sha256=$(sha256sum "$src/develop_stage1_best_dice.pth" | awk '{print $1}')"
    } > "$dst/STAGE1_DEVELOPMENT_REUSED_FROM.txt"
    echo "[REUSE] stage1 inner development $src -> $dst"
  fi
}
