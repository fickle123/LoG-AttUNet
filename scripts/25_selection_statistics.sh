#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/statistics
# Repeat the clean LoG-vs-standard attention comparison under the historical HD-aware selector.
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment core --budget stage1 --selection-rule constrained_hd --ours full_log_attention_only --baselines full_standard --bootstrap 10000 --permutations 10000 | tee outputs/statistics/core_stage1_constrained_hd_attention_isolation.log
# Also check the submitted final architecture under the same selector.
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment core --budget stage1 --selection-rule constrained_hd --ours full_log_shallow --baselines full_standard --bootstrap 10000 --permutations 10000 | tee outputs/statistics/core_stage1_constrained_hd_final.log
