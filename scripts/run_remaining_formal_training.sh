#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
export DATA_ROOT="../fetal_dataset"
export SPLIT_ROOT="outputs/splits"
export RUN_ROOT="outputs/runs"
export SEEDS="42 123 2026"
export IMAGE_SIZE=256
export BASE_CHANNELS=40
export BATCH_SIZE=8
export EVAL_BATCH_SIZE=8
export AMP_FLAG="--amp"

mkdir -p outputs/driver_logs outputs/pids

cuda_check() {
python - <<'PY'
import torch, sys

print("cuda_available =", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable.")
    sys.exit(1)

print("GPU =", torch.cuda.get_device_name(0))

x = torch.randn(1024,1024,device="cuda")
y = x @ x
torch.cuda.synchronize()

print("CUDA COMPUTE TEST = PASS")
PY
}

outer_test_check() {
    local n
    n=$(find outputs/runs \
        -type f -name OUTER_TEST_COMPLETE \
        2>/dev/null | wc -l)

    echo "OUTER_TEST_COMPLETE = $n"

    if [ "$n" -ne 0 ]; then
        echo "[FATAL] Outer test has already been touched."
        exit 1
    fi
}

total_refits() {
    find outputs/runs \
        -type f -name REFIT_COMPLETE \
        2>/dev/null | wc -l
}

run_stage() {
    local stage="$1"
    local script="$2"
    local target="$3"
    local name="$4"

    echo
    echo "============================================================"
    echo "PREPARING STAGE $stage"
    echo "SCRIPT = $script"
    echo "TARGET TOTAL = $target / 405"
    echo "============================================================"

    outer_test_check
    cuda_check

    local log
    log="outputs/driver_logs/${name}_AUTO_$(date -u +%Y%m%dT%H%M%SZ).log"

    echo "LOG = $log"

    # Create a separate process group so a CPU fallback can stop
    # the whole current stage safely.
    setsid env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        DATA_ROOT="$DATA_ROOT" \
        SPLIT_ROOT="$SPLIT_ROOT" \
        RUN_ROOT="$RUN_ROOT" \
        SEEDS="$SEEDS" \
        IMAGE_SIZE="$IMAGE_SIZE" \
        BASE_CHANNELS="$BASE_CHANNELS" \
        BATCH_SIZE="$BATCH_SIZE" \
        EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        AMP_FLAG="$AMP_FLAG" \
        bash "$script" \
        > "$log" 2>&1 &

    local pid=$!

    echo "$pid" > outputs/pids/current_formal.pid
    echo "$log" > outputs/pids/current_formal.logpath
    echo "$stage" > outputs/pids/current_formal.stage

    echo "PID = $pid"

    # Monitor the stage until it finishes.
    while kill -0 "$pid" 2>/dev/null; do

        # Abort immediately if CPU fallback is detected.
        if grep -q '^device=cpu' "$log" 2>/dev/null; then
            echo
            echo "============================================================"
            echo "[FATAL] CPU FALLBACK DETECTED IN STAGE $stage"
            echo "Stopping the entire current stage process group."
            echo "LOG = $log"
            echo "============================================================"

            kill -TERM -- "-$pid" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$pid" 2>/dev/null || true
            exit 2
        fi

        sleep 10
    done

    set +e
    wait "$pid"
    rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        echo "[FATAL] Stage $stage exited with code $rc"
        echo "Inspect:"
        echo "tail -n 200 $log"
        exit "$rc"
    fi

    # Final CPU audit.
    if grep -q '^device=cpu' "$log"; then
        echo "[FATAL] CPU fallback found after Stage $stage."
        exit 2
    fi

    local total
    total=$(total_refits)

    echo
    echo "Stage $stage finished."
    echo "TOTAL REFIT = $total / 405"

    if [ "$total" -ne "$target" ]; then
        echo "[FATAL] Unexpected formal matrix count."
        echo "Expected $target, got $total."
        exit 3
    fi

    outer_test_check

    echo "[OK] Stage $stage verified."
}

echo "============================================================"
echo "REMAINING FORMAL TRAINING DRIVER"
echo "Stages 13 -> 14 -> 15"
echo "Outer test is LOCKED."
echo "============================================================"

outer_test_check
cuda_check

TOTAL=$(total_refits)

echo
echo "Current TOTAL REFIT = $TOTAL / 405"

# Stage 13 target: 345
if [ "$TOTAL" -lt 345 ]; then
    run_stage \
        13 \
        scripts/13_run_depth_ablation.sh \
        345 \
        13_depth_ablation
else
    echo "[SKIP] Stage 13 already complete."
fi

TOTAL=$(total_refits)

# Stage 14 target: 360
if [ "$TOTAL" -lt 360 ]; then
    run_stage \
        14 \
        scripts/14_run_original_method48.sh \
        360 \
        14_original_method48
else
    echo "[SKIP] Stage 14 already complete."
fi

TOTAL=$(total_refits)

# Stage 15 target: 405
if [ "$TOTAL" -lt 405 ]; then
    run_stage \
        15 \
        scripts/15_run_selection_sensitivity.sh \
        405 \
        15_selection_sensitivity
else
    echo "[SKIP] Stage 15 already complete."
fi

TOTAL=$(total_refits)
TEST=$(find outputs/runs \
    -type f -name OUTER_TEST_COMPLETE \
    2>/dev/null | wc -l)

echo
echo "============================================================"
echo "ALL FORMAL TRAINING FINISHED"
echo "TOTAL REFIT = $TOTAL / 405"
echo "OUTER TEST  = $TEST"
echo "============================================================"

if [ "$TOTAL" -eq 405 ] && [ "$TEST" -eq 0 ]; then
    echo
    echo "[SUCCESS] Formal training matrix is complete."
    echo
    echo "STOPPING HERE BY DESIGN."
    echo "DO NOT automatically run outer test."
    echo
    echo "Next required steps:"
    echo "  16_check_formal_matrix.sh"
    echo "  17_cleanup_development_checkpoints.sh"
    echo "  18_freeze_protocol.sh"
    echo "  19_run_all_outer_tests.sh"
else
    echo "[FATAL] Final matrix validation failed."
    exit 4
fi
