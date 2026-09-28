#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/statistics outputs/figures
# A. Final submitted architecture, architecture-only (no extra stage-2 budget)
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment core --budget stage1 --selection-rule best_dice --ours full_log_shallow --baselines full_standard attention_resunet resunet attention_unet unet --bootstrap 10000 --permutations 10000 | tee outputs/statistics/core_stage1_final.log
# B. Clean attention isolation: only shallow attention guidance differs
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment core --budget stage1 --selection-rule best_dice --ours full_log_attention_only --baselines full_standard --bootstrap 10000 --permutations 10000 | tee outputs/statistics/core_stage1_attention_isolation.log
# C. Fair matched-budget final comparison (all comparators receive stage-2 opportunity)
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment core --budget matched48 --selection-rule best_dice --ours full_log_shallow --baselines full_standard attention_resunet resunet attention_unet unet --bootstrap 10000 --permutations 10000 | tee outputs/statistics/core_matched48_final.log
# D. Matched-budget attention isolation
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment core --budget matched48 --selection-rule best_dice --ours full_log_attention_only --baselines full_standard --bootstrap 10000 --permutations 10000 | tee outputs/statistics/core_matched48_attention_isolation.log
python -m curvrevision.plot_comparison --ci outputs/statistics/bootstrap_ci_core_stage1_best_dice_full_log_shallow.csv --output outputs/figures
