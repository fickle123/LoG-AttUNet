#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-../fetal_dataset}"
python -m curvrevision.annotation_audit --data-root "$DATA_ROOT" --output outputs/annotation_audit | tee outputs/annotation_audit.log
