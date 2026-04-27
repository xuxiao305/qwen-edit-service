#!/usr/bin/env bash
# Download ComfyUI-format FP8 model files for Qwen-Image-Edit-2509 workflow.
#
# These files are NOT used by QwenEditService (which uses diffusers format).
# They are for running the Qwen_MultiView.json ComfyUI workflow directly.
#
# Files downloaded (~30 GB total):
#   /project/qwen_edit/comfyui_models/
#   ├── diffusion_models/
#   │   └── qwen_image_edit_2509_fp8_e4m3fn.safetensors   (20.4 GB) ← unet
#   ├── text_encoders/
#   │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors          (9.4 GB) ← clip
#   └── vae/
#       └── qwen_image_vae.safetensors                      (~0.5 GB) ← vae
#
# ComfyUI workflow node mapping:
#   Load Diffusion Model  → diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors
#   CLIPLoader            → text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
#   VAELoader             → vae/qwen_image_vae.safetensors
#   LoRAs                 → see /project/qwen_edit/loras/ (run download_lora.sh first)
#
# Usage:
#   bash deploy/download_fp8_comfyui.sh
#   nohup bash deploy/download_fp8_comfyui.sh > /project/qwen_edit/logs/dl_fp8.log 2>&1 &
set -eu

BASE_DIR="${QWEN_COMFYUI_MODEL_DIR:-/project/qwen_edit/comfyui_models}"
UNET_DIR="$BASE_DIR/diffusion_models"
CLIP_DIR="$BASE_DIR/text_encoders"
VAE_DIR="$BASE_DIR/vae"

mkdir -p "$UNET_DIR" "$CLIP_DIR" "$VAE_DIR"
mkdir -p /project/qwen_edit/logs

CONDA_SH="${CONDA_SH:-/data/miniconda3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate qwen_edit

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "=== [1/3] UNet FP8 (20.4 GB) — qwen_image_edit_2509_fp8_e4m3fn.safetensors ==="
# Source file is named with "_scaled" suffix; rename to match ComfyUI workflow
TMP_UNET="$UNET_DIR/qwen_image_edit_2509_fp8_e4m3fn_scaled.safetensors"
huggingface-cli download lightx2v/Qwen-Image-Lightning \
  Qwen-Image-Edit-2509/qwen_image_edit_2509_fp8_e4m3fn_scaled.safetensors \
  --local-dir "$UNET_DIR" --local-dir-use-symlinks False
# Move out of the subfolder created by huggingface-cli
if [ -f "$UNET_DIR/Qwen-Image-Edit-2509/qwen_image_edit_2509_fp8_e4m3fn_scaled.safetensors" ]; then
  mv "$UNET_DIR/Qwen-Image-Edit-2509/qwen_image_edit_2509_fp8_e4m3fn_scaled.safetensors" \
     "$UNET_DIR/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
  rmdir "$UNET_DIR/Qwen-Image-Edit-2509" 2>/dev/null || true
elif [ -f "$TMP_UNET" ]; then
  mv "$TMP_UNET" "$UNET_DIR/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
fi
echo "UNet done: $(ls -lh $UNET_DIR/qwen_image_edit_2509_fp8_e4m3fn.safetensors 2>/dev/null | awk '{print $5}')"

echo "=== [2/3] Text Encoder FP8 (9.4 GB) — qwen_2.5_vl_7b_fp8_scaled.safetensors ==="
huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
  --local-dir "$CLIP_DIR" --local-dir-use-symlinks False
# Flatten subfolders
NESTED="$CLIP_DIR/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
if [ -f "$NESTED" ]; then
  mv "$NESTED" "$CLIP_DIR/qwen_2.5_vl_7b_fp8_scaled.safetensors"
  rm -rf "$CLIP_DIR/split_files"
fi
echo "CLIP done: $(ls -lh $CLIP_DIR/qwen_2.5_vl_7b_fp8_scaled.safetensors 2>/dev/null | awk '{print $5}')"

echo "=== [3/3] VAE — qwen_image_vae.safetensors ==="
huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir "$VAE_DIR" --local-dir-use-symlinks False
NESTED_VAE="$VAE_DIR/split_files/vae/qwen_image_vae.safetensors"
if [ -f "$NESTED_VAE" ]; then
  mv "$NESTED_VAE" "$VAE_DIR/qwen_image_vae.safetensors"
  rm -rf "$VAE_DIR/split_files"
fi
echo "VAE done: $(ls -lh $VAE_DIR/qwen_image_vae.safetensors 2>/dev/null | awk '{print $5}')"

echo ""
echo "=== All done. Directory layout ==="
find "$BASE_DIR" -name "*.safetensors" -exec ls -lh {} \;
