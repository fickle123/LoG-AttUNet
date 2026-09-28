#!/usr/bin/env bash
set -euo pipefail
python -m curvrevision.test_all --runs-root outputs/runs --lock outputs/PROTOCOL_FROZEN.json | tee outputs/outer_test_master.log
