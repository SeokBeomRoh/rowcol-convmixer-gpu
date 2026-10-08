#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

DEPTH="${DEPTH:-8}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
LR_MAX="${LR_MAX:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GPU_IDS="${GPU_IDS:-0,1}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Run first: bash scripts/setup_wsl.sh" >&2
  exit 1
fi

COMMON_ARGS=(
  --depth "$DEPTH"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --lr-max "$LR_MAX"
  --wd "$WEIGHT_DECAY"
  --gpu-ids "$GPU_IDS"
  --workers "$WORKERS"
  --batch-size "$BATCH_SIZE"
  --data-dir "$DATA_DIR"
  --results-dir "$RESULTS_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
)

if [[ "$SAVE_CHECKPOINT" == "1" ]]; then
  COMMON_ARGS+=(--save-checkpoint)
fi

run_condition() {
  local label="$1"
  shift
  echo
  echo "===== Running condition $label ====="
  "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" "$@"
}

echo "Verifying A-E structures before training"
"$PYTHON_BIN" scripts/check_gpu.py
"$PYTHON_BIN" scripts/verify_m3_m5_controls.py

run_condition A \
  --model rowcol_weighted \
  --name A_original_m3 \
  --rowcol-combine learnable

run_condition B \
  --model rowcol_weighted \
  --name B_m3_prenorm \
  --rowcol-combine learnable \
  --rowcol-pre-norm

run_condition C \
  --model rowcol_weighted \
  --name C_add_control \
  --rowcol-combine sum \
  --rowcol-pre-norm

run_condition D \
  --model rowcol_mul_weighted \
  --name D_mul_control \
  --no-mul-pos-prior

run_condition E \
  --model rowcol_mul_weighted \
  --name E_original_m5

echo
echo "All five controlled experiments completed."
