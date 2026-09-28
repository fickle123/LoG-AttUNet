#!/usr/bin/env bash
set -euo pipefail
# Safe only after matched48/original48 branches have reused the core stage-1 development checkpoints.
find outputs/runs -type f \( -name 'develop_stage1_best_dice.pth' -o -name 'develop_stage2_best_dice.pth' -o -name 'develop_stage1_last.pth' -o -name 'develop_stage2_last.pth' \) -print -delete
find outputs/runs -type f -name 'refit_last.pth' -print -delete
echo "Temporary development checkpoints cleaned. Final outer-train checkpoints are untouched."
