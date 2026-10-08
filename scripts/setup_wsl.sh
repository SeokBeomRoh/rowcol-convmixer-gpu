#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if [[ ! -r /proc/version ]] || ! grep -qi microsoft /proc/version; then
  echo "Warning: this script is intended for WSL." >&2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "Failed to create the virtual environment." >&2
    echo "On Ubuntu/WSL, install it with: sudo apt install python3-venv" >&2
    exit 1
  fi
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Installing CUDA-enabled PyTorch from $TORCH_INDEX_URL"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
"$VENV_PYTHON" -m pip install -r requirements.txt

echo "Checking WSL GPU visibility"
"$VENV_PYTHON" scripts/check_gpu.py

echo "Running controlled-ablation smoke checks"
"$VENV_PYTHON" scripts/verify_m3_m5_controls.py

echo "WSL setup and smoke verification completed."
echo "Run the experiments with: bash scripts/run_m3_m5_controls_wsl.sh"
