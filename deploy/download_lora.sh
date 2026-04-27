#!/usr/bin/env bash
# Download LoRA weights for Qwen-Image-Edit-2511 to /project/qwen_edit/loras/
#
# LoRAs downloaded:
#   1. multiple-angles  fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
#      Trigger: <sks> [azimuth] [elevation] [distance]
#      Example: "<sks> right side view eye-level shot medium shot"
#      Size: 281 MB; recommended strength: 0.8-1.0
#
#   2. lightning-4step  lightx2v/Qwen-Image-Lightning
#      File: Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors
#      Use with: num_inference_steps=4, true_cfg_scale=1.0
#      Size: 850 MB
#
# Usage:
#   bash deploy/download_lora.sh
#   HF_TOKEN=<your_token> bash deploy/download_lora.sh   # if rate-limited
set -eu

LORA_DIR="${QWEN_EDIT_LORA_DIR:-/project/qwen_edit/loras}"
mkdir -p "$LORA_DIR"

CONDA_SH="${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate qwen_edit

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

echo "=== downloading multiple-angles LoRA (2511) ==="
huggingface-cli download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA \
  qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --local-dir "$LORA_DIR"

echo "=== downloading Lightning 4-step LoRA ==="
huggingface-cli download lightx2v/Qwen-Image-Lightning \
  Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors \
  --local-dir "$LORA_DIR"

echo ""
echo "Done. Files in $LORA_DIR:"
ls -lh "$LORA_DIR"/*.safetensors
