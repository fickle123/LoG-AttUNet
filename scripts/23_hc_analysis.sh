#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-../fetal_dataset}"
python -m curvrevision.hc_analysis --metrics outputs/aggregate/all_per_image_metrics.csv --data-root "$DATA_ROOT" --output outputs/hc_analysis --experiment core --budget original48 --models full_log_shallow | tee outputs/hc_analysis.log
