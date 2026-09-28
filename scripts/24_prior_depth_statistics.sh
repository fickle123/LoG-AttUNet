#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/statistics
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment prior --budget stage1 --selection-rule best_dice --ours prior_log --baselines prior_none prior_sobel prior_laplacian prior_gaussian_gradient --bootstrap 10000 --permutations 10000 | tee outputs/statistics/prior_stage1.log
python -m curvrevision.statistics --input outputs/aggregate/all_per_image_metrics.csv --output outputs/statistics --experiment depth --budget stage1 --selection-rule best_dice --ours depth_shallow --baselines depth_middle depth_deep depth_all --bootstrap 10000 --permutations 10000 | tee outputs/statistics/depth_stage1.log
