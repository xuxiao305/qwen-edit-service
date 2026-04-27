#!/usr/bin/env bash
# Start qwen_edit_server (dual-GPU mode).
#
# Usage:
#   bash run_server.sh           # foreground (Ctrl-C to stop)
#   bash run_server.sh --bg      # background, logs to $LOG
set -u

# --- adjust if your conda lives elsewhere ---
CONDA_SH="${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate qwen_edit

# --- service config ---
export QWEN_EDIT_MODEL_PATH="${QWEN_EDIT_MODEL_PATH:-/project/qwen_edit/models/Qwen-Image-Edit-2511}"
export QWEN_EDIT_HOST="${QWEN_EDIT_HOST:-127.0.0.1}"
export QWEN_EDIT_PORT="${QWEN_EDIT_PORT:-8765}"
export QWEN_EDIT_GPU_MEM_GIB="${QWEN_EDIT_GPU_MEM_GIB:-22}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$HERE/qwen_edit_server.py"
LOG="${QWEN_EDIT_LOG:-/project/qwen_edit/logs/server.out}"

if [ "${1:-}" = "--bg" ]; then
  mkdir -p "$(dirname "$LOG")"
  nohup python "$SERVER" > "$LOG" 2>&1 &
  echo "server PID $!  log: $LOG"
else
  python "$SERVER"
fi
