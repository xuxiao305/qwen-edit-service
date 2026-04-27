#!/usr/bin/env bash
# Create qwen_edit conda env and install all deps.
#
# Strategy:
#   1. Download torch / torchvision wheels DIRECTLY from SJTU mirror
#      (88-99 MB/s). We avoid `pip install torch` against
#      download.pytorch.org which throttles the nvidia-cuda-* deps to
#      ~20 KB/s.
#   2. Install those local wheels with deps coming from Tsinghua PyPI
#      mirror (the nvidia-cuda-nvrtc-cu12 etc. wheels live on PyPI,
#      Tsinghua mirrors them at full bandwidth).
#   3. Install the rest of the stack (diffusers / fastapi / ...) from
#      Tsinghua too. We pin transformers <5 because 5.x broke diffusers
#      Qwen Image-Edit pipeline integration as of 2026-04.
set -u
LOG="${SETUP_LOG:-/project/qwen_edit/setup.log}"
WHL_DIR="${WHL_DIR:-/project/qwen_edit/wheels}"
mkdir -p "$(dirname "$LOG")" "$WHL_DIR"

CONDA_SH="${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"

PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PT_CDN="https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
TORCH_WHL="torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl"
TV_WHL="torchvision-0.20.1%2Bcu121-cp311-cp311-linux_x86_64.whl"

if conda env list | grep -q "/qwen_edit\b"; then
  echo "[$(date)] env qwen_edit already exists" >> "$LOG"
else
  echo "[$(date)] creating env qwen_edit (python=3.11)" >> "$LOG"
  conda create -y -n qwen_edit python=3.11 >> "$LOG" 2>&1
fi

conda activate qwen_edit

pip config set global.index-url "$PIP_INDEX" >> "$LOG" 2>&1

echo "[$(date)] === downloading torch wheels from SJTU mirror ===" >> "$LOG"
wget -c -nv -P "$WHL_DIR" "$PT_CDN/$TORCH_WHL"  >> "$LOG" 2>&1
wget -c -nv -P "$WHL_DIR" "$PT_CDN/$TV_WHL"     >> "$LOG" 2>&1
ls -lh "$WHL_DIR" >> "$LOG"

echo "[$(date)] === installing torch wheels (deps from Tsinghua) ===" >> "$LOG"
pip install --no-cache-dir --index-url "$PIP_INDEX" \
  "$WHL_DIR/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl" \
  "$WHL_DIR/torchvision-0.20.1+cu121-cp311-cp311-linux_x86_64.whl" \
  >> "$LOG" 2>&1

echo "[$(date)] === installing diffusers / fastapi stack (Tsinghua PyPI) ===" >> "$LOG"
pip install --no-cache-dir --index-url "$PIP_INDEX" \
  "diffusers>=0.37.0,<0.40" \
  "transformers>=4.49,<5" \
  "accelerate>=1.0.0" \
  "safetensors>=0.4.5" \
  sentencepiece \
  protobuf \
  pillow \
  fastapi \
  "uvicorn[standard]" \
  python-multipart \
  >> "$LOG" 2>&1

echo "[$(date)] === DONE. Versions: ===" >> "$LOG"
python -c "import torch, diffusers, transformers, fastapi, accelerate; print('torch', torch.__version__, 'cuda_avail=', torch.cuda.is_available(), 'cuda_ver=', torch.version.cuda, 'gpu_count=', torch.cuda.device_count()); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('fastapi', fastapi.__version__)" >> "$LOG" 2>&1

echo "[$(date)] === pipeline class probe ===" >> "$LOG"
python -c "import diffusers; print('QwenImageEditPlusPipeline:', hasattr(diffusers, 'QwenImageEditPlusPipeline')); print('QwenImageEditPipeline:', hasattr(diffusers, 'QwenImageEditPipeline'))" >> "$LOG" 2>&1
