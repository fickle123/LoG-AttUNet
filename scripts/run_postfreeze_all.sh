#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
export DATA_ROOT="../fetal_dataset"
export SPLIT_ROOT="outputs/splits"
export RUN_ROOT="outputs/runs"
export BATCH_SIZE=8
export EVAL_BATCH_SIZE=8
export AMP_FLAG="--amp"

STATE_DIR="outputs/postfreeze_pipeline"
LOG_DIR="outputs/driver_logs"
PID_DIR="outputs/pids"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$PID_DIR"

timestamp() {
    date -u +%Y%m%dT%H%M%SZ
}

count_refit() {
    find outputs/runs \
        -type f -name REFIT_COMPLETE \
        2>/dev/null | wc -l
}

count_final() {
    find outputs/runs \
        -type f -name final_outer_train_model.pth \
        2>/dev/null | wc -l
}

count_outer() {
    find outputs/runs \
        -type f -name OUTER_TEST_COMPLETE \
        2>/dev/null | wc -l
}

cuda_check() {
python - <<'PY'
import torch
import sys

print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable.")
    sys.exit(1)

print("GPU =", torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("CUDA COMPUTE TEST = PASS")
PY
}

echo "============================================================"
echo "POST-FREEZE FINAL PIPELINE"
echo "Stage 19 -> 20 -> 21 -> 22 -> 23 -> 24 -> 25"
echo "============================================================"

#
# ----- GLOBAL SAFETY GATE -----
#

if [ ! -f outputs/PROTOCOL_FROZEN.json ]; then
    echo "[FATAL] outputs/PROTOCOL_FROZEN.json not found."
    echo "Protocol has not been frozen."
    exit 10
fi

REFIT=$(count_refit)
FINAL=$(count_final)
OUTER=$(count_outer)

echo
echo "REFIT_COMPLETE          = $REFIT"
echo "final_outer_train_model = $FINAL"
echo "OUTER_TEST_COMPLETE     = $OUTER"

if [ "$REFIT" -ne 405 ]; then
    echo "[FATAL] Expected 405 REFIT_COMPLETE markers."
    exit 11
fi

if [ "$FINAL" -ne 405 ]; then
    echo "[FATAL] Expected 405 final_outer_train_model.pth files."
    exit 12
fi

if [ "$OUTER" -gt 405 ]; then
    echo "[FATAL] Unexpected OUTER_TEST_COMPLETE count: $OUTER"
    exit 13
fi

echo
echo "Rechecking formal matrix..."

bash scripts/16_check_formal_matrix.sh

echo
echo "[OK] Formal matrix verified."
echo "[OK] Protocol frozen."

#
# ----- STAGE 19 -----
#

OUTER=$(count_outer)

if [ "$OUTER" -eq 405 ]; then

    echo
    echo "============================================================"
    echo "[SKIP] STAGE 19 already complete: 405 / 405"
    echo "============================================================"

    touch "$STATE_DIR/19.done"

else

    echo
    echo "============================================================"
    echo "STAGE 19 — LOCKED OUTER TEST"
    echo "Current: $OUTER / 405"
    echo "============================================================"

    cuda_check

    LOG="$LOG_DIR/19_outer_tests_AUTO_$(timestamp).log"

    echo "$LOG" > "$PID_DIR/postfreeze.current.logpath"
    echo "19" > "$PID_DIR/postfreeze.current.stage"

    echo "LOG=$LOG"

    setsid env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        DATA_ROOT="$DATA_ROOT" \
        SPLIT_ROOT="$SPLIT_ROOT" \
        RUN_ROOT="$RUN_ROOT" \
        BATCH_SIZE="$BATCH_SIZE" \
        EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        AMP_FLAG="$AMP_FLAG" \
        bash scripts/19_run_all_outer_tests.sh \
        > "$LOG" 2>&1 &

    PID=$!

    echo "$PID" > "$PID_DIR/postfreeze.current.pid"

    echo "PID=$PID"

    while kill -0 "$PID" 2>/dev/null; do

        if grep -q '^device=cpu' "$LOG" 2>/dev/null; then

            echo
            echo "============================================================"
            echo "[FATAL] CPU FALLBACK DETECTED DURING STAGE 19"
            echo "Stopping Stage 19."
            echo "============================================================"

            kill -TERM -- "-$PID" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$PID" 2>/dev/null || true

            exit 19
        fi

        sleep 10
    done

    set +e
    wait "$PID"
    RC=$?
    set -e

    if [ "$RC" -ne 0 ]; then
        echo "[FATAL] Stage 19 exited with code $RC"
        echo "LOG=$LOG"
        exit "$RC"
    fi

    OUTER=$(count_outer)

    echo
    echo "Stage19 OUTER_TEST_COMPLETE = $OUTER / 405"

    if [ "$OUTER" -ne 405 ]; then
        echo "[FATAL] Stage19 finished but outer-test matrix is incomplete."
        exit 20
    fi

    touch "$STATE_DIR/19.done"

    echo "[OK] Stage 19 complete."
fi


run_post_stage() {

    local STAGE="$1"
    local SCRIPT="$2"
    local NAME="$3"

    if [ -f "$STATE_DIR/${STAGE}.done" ]; then

        echo
        echo "============================================================"
        echo "[SKIP] Stage $STAGE already completed"
        echo "============================================================"

        return
    fi

    echo
    echo "============================================================"
    echo "STAGE $STAGE — $NAME"
    echo "SCRIPT = $SCRIPT"
    echo "============================================================"

    #
    # All analytical stages must use a complete outer-test matrix.
    #
    OUTER=$(count_outer)

    if [ "$OUTER" -ne 405 ]; then
        echo "[FATAL] Stage $STAGE requires 405/405 outer tests."
        echo "Current OUTER_TEST_COMPLETE=$OUTER"
        exit 30
    fi

    LOG="$LOG_DIR/${STAGE}_${NAME}_AUTO_$(timestamp).log"

    echo "$LOG" > "$PID_DIR/postfreeze.current.logpath"
    echo "$STAGE" > "$PID_DIR/postfreeze.current.stage"

    echo "LOG=$LOG"

    set +e
    bash "$SCRIPT" > "$LOG" 2>&1
    RC=$?
    set -e

    if [ "$RC" -ne 0 ]; then
        echo "[FATAL] Stage $STAGE failed."
        echo "EXIT_CODE=$RC"
        echo "LOG=$LOG"
        exit "$RC"
    fi

    touch "$STATE_DIR/${STAGE}.done"

    echo "[OK] Stage $STAGE complete."
}

#
# ----- STAGES 20–21 -----
#

run_post_stage \
    20 \
    scripts/20_collect_oof.sh \
    collect_oof

run_post_stage \
    21 \
    scripts/21_statistics.sh \
    statistics

#
# ----- STAGE 22 -----
#
# Efficiency measurements may use CUDA,
# so verify GPU health again immediately beforehand.
#

if [ ! -f "$STATE_DIR/22.done" ]; then
    echo
    echo "Checking CUDA before Stage 22 efficiency..."
    cuda_check
fi

run_post_stage \
    22 \
    scripts/22_efficiency.sh \
    efficiency

#
# ----- STAGES 23–25 -----
#

run_post_stage \
    23 \
    scripts/23_hc_analysis.sh \
    hc_analysis

run_post_stage \
    24 \
    scripts/24_prior_depth_statistics.sh \
    prior_depth_statistics

run_post_stage \
    25 \
    scripts/25_selection_statistics.sh \
    selection_statistics


#
# ----- FINAL AUDIT -----
#

REFIT=$(count_refit)
FINAL=$(count_final)
OUTER=$(count_outer)

echo
echo "============================================================"
echo "FINAL PIPELINE STATUS"
echo "============================================================"
echo "REFIT_COMPLETE          = $REFIT / 405"
echo "final_outer_train_model = $FINAL / 405"
echo "OUTER_TEST_COMPLETE     = $OUTER / 405"
echo

for STAGE in 19 20 21 22 23 24 25
do
    if [ -f "$STATE_DIR/${STAGE}.done" ]; then
        echo "Stage $STAGE = COMPLETE"
    else
        echo "Stage $STAGE = INCOMPLETE"
    fi
done

if [ "$REFIT" -eq 405 ] && \
   [ "$FINAL" -eq 405 ] && \
   [ "$OUTER" -eq 405 ] && \
   [ -f "$STATE_DIR/19.done" ] && \
   [ -f "$STATE_DIR/20.done" ] && \
   [ -f "$STATE_DIR/21.done" ] && \
   [ -f "$STATE_DIR/22.done" ] && \
   [ -f "$STATE_DIR/23.done" ] && \
   [ -f "$STATE_DIR/24.done" ] && \
   [ -f "$STATE_DIR/25.done" ]; then

    echo
    echo "============================================================"
    echo "[SUCCESS]"
    echo "ALL STAGES 19–25 COMPLETED SUCCESSFULLY"
    echo "============================================================"

    touch "$STATE_DIR/ALL_POSTFREEZE_COMPLETE"

else

    echo
    echo "[FATAL] Final pipeline validation failed."
    exit 99

fi
