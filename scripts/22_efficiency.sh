#!/usr/bin/env bash
set -euo pipefail
python -m curvrevision.efficiency --output outputs/efficiency --image-size 256 --base-channels 40 --warmup 50 --iters 200 --throughput-batch 8 --amp | tee outputs/efficiency.log
