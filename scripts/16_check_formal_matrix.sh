#!/usr/bin/env bash
set -euo pipefail
read -ra SEED_LIST <<< "${SEEDS:-42 123 2026}"
python -m curvrevision.matrix_check --runs-root outputs/runs --seeds "${SEED_LIST[@]}" --report outputs/formal_matrix_check.json
