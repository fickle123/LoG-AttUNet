#!/usr/bin/env bash
set -euo pipefail
python -m curvrevision.freeze_protocol --runs-root outputs/runs --split-root outputs/splits --matrix-report outputs/formal_matrix_check.json --lock outputs/PROTOCOL_FROZEN.json | tee outputs/protocol_freeze.log
