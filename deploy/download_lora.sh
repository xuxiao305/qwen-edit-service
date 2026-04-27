#!/usr/bin/env bash
# Download LoRA weights for Qwen-Image-Edit to /project/qwen_edit/loras/
#
# Files downloaded:
#   For QwenEditService (diffusers, 2511 base):
#     1. qwen-image-edit-2511-multiple-angles-lora.safetensors  (281 MB)
#        Source: fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
#        Trigger: "<sks> right side view eye-level shot medium shot"
#        Strength: 0.8-1.0
#
#     2. Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors  (850 MB)
#        Source: lightx2v/Qwen-Image-Lightning
#        Use: QWEN_EDIT_DEFAULT_STEPS=4  QWEN_EDIT_LORA_PATHS=...,<this>
#
#   For ComfyUI workflow (2509 base):
#     3. Qwen-Edit-2509-Multiple-angles.safetensors  (~300 MB)
#        Source: dx8152/Qwen-Edit-2509-Multiple-angles
#
# Usage:
#   bash deploy/download_lora.sh
#   HF_TOKEN=<token> bash deploy/download_lora.sh   # if rate-limited
set -eu

LORA_DIR="${QWEN_EDIT_LORA_DIR:-/project/qwen_edit/loras}"
mkdir -p "$LORA_DIR"

CONDA_SH="${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate qwen_edit

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "Using HF endpoint: $HF_ENDPOINT"

echo "=== [1/3] downloading multiple-angles LoRA (2511, for QwenEditService) ==="
huggingface-cli download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA \
  qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --local-dir "$LORA_DIR"

echo "=== [2/3] downloading Lightning 4-step LoRA (for QwenEditService) ==="
huggingface-cli download lightx2v/Qwen-Image-Lightning \
  Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors \
  --local-dir "$LORA_DIR"

echo "=== [3/3] downloading multiple-angles LoRA (2509, for ComfyUI workflow) ==="
huggingface-cli download dx8152/Qwen-Edit-2509-Multiple-angles \
  Qwen-Edit-2509-Multiple-angles.safetensors \
  --local-dir "$LORA_DIR"

echo ""
echo "Done. Files in $LORA_DIR:"
ls -lh "$LORA_DIR"/*.safetensors
