#!/usr/bin/env bash
set -euo pipefail
python -m curvrevision.collect --runs-root outputs/runs --output outputs/aggregate | tee outputs/collect.log
