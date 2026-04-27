"""Dual-GPU loader for Qwen-Image-Edit-2511 (bf16, no quantization).

Memory budget on 2× A30 24GB (48 GB total)
-----------------------------------------
The bf16 weights alone exceed total VRAM:
    transformer  ~41 GB   (20.4 B params × 2 bytes)
    text_encoder ~14 GB   (Qwen2.5-VL 7 B × 2 bytes)
    vae           ~1 GB
    -------- subtotal  ~56 GB > 48 GB

So one component must be CPU-resident. Strategy:
    * `text_encoder`  -> CPU-pinned, **auto-streamed to cuda:0 on forward**
                        via `accelerate.cpu_offload`. Encoding cost per call
                        ~10 s (PCIe 4.0 x16 ≈ 20 GB/s + Qwen2.5-VL forward).
    * `transformer`   -> sharded across cuda:0 + cuda:1 via `device_map="auto"`,
                        using ~22 GB on each card. The diffusion loop runs
                        cross-GPU but never touches CPU.
    * `vae`           -> cuda:0 (small).
    * tokenizer / processor / scheduler -> CPU (negligible).

The processor for `QwenImageEditPlusPipeline` is `Qwen2VLProcessor`; we load
it explicitly because `AutoProcessor` can't infer the type from the bare
processor subfolder.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import torch

LOG = logging.getLogger(__name__)


def select_pipeline_class(model_path: str) -> tuple[Any, str]:
    """Pick the diffusers pipeline class declared by `model_index.json`."""
    import diffusers

    idx_path = os.path.join(model_path, "model_index.json")
    cls_name = "QwenImageEditPipeline"
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                cls_name = json.load(f).get("_class_name", cls_name)
        except Exception as exc:  # pragma: no cover
            LOG.warning("failed to read model_index.json: %s", exc)

    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        for fallback in (
            "QwenImageEditPlusPipeline",
            "QwenImageEditPipeline",
            "DiffusionPipeline",
        ):
            cls = getattr(diffusers, fallback, None)
            if cls is not None:
                cls_name = fallback
                break
    if cls is None:
        raise RuntimeError("no suitable diffusers pipeline class found")
    return cls, cls_name


def load_dual_gpu_pipeline(
    model_path: str,
    *,
    per_gpu_mem_gib: int = 22,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the Qwen-Image-Edit pipeline for dual-GPU bf16 inference.

    See module docstring for the memory strategy.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"dual-GPU mode requires >= 2 CUDA devices, found {n_gpus}"
        )

    from accelerate import dispatch_model
    from accelerate.utils import infer_auto_device_map
    from transformers import (
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLProcessor,
    )
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageTransformer2DModel,
    )

    cls, cls_name = select_pipeline_class(model_path)
    LOG.info("pipeline class: %s", cls_name)

    text_encoder_exec_device = "cuda:0"
    vae_device = "cuda:1"  # off cuda:0 to leave headroom for text_encoder streaming

    t0 = time.time()

    # 1) text_encoder: load to CPU then dispatch with PER-LAYER offloading.
    #    Whole-model `cpu_offload(...)` would try to pull all ~14 GB onto
    #    execution_device at every forward and OOMs because the transformer
    #    leaves <1 GB free on cuda:0. dispatch_model with a tight max_memory
    #    keeps only ~2 GiB resident at a time and streams the rest from CPU.
    LOG.info("loading text_encoder fully to CPU RAM ...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()
    te_max_memory = {0: "2GiB", "cpu": "64GiB"}
    te_no_split = getattr(text_encoder, "_no_split_modules", None) or []
    te_device_map = infer_auto_device_map(
        text_encoder,
        max_memory=te_max_memory,
        no_split_module_classes=te_no_split,
        dtype=dtype,
    )
    on_gpu = sum(1 for v in te_device_map.values() if v != "cpu")
    on_cpu = sum(1 for v in te_device_map.values() if v == "cpu")
    LOG.info("text_encoder dispatch: %d submodules on GPU, %d on CPU (max %s GPU resident)",
             on_gpu, on_cpu, te_max_memory[0])
    text_encoder = dispatch_model(
        text_encoder,
        device_map=te_device_map,
        main_device=text_encoder_exec_device,
    )

    # 2) vae on cuda:1 (small, keeps cuda:0 headroom free)
    LOG.info("loading vae on %s ...", vae_device)
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(vae_device)

    # 3) transformer sharded across both GPUs. cuda:0 cap is reduced to leave
    #    ~6 GiB free for text_encoder layer streaming + activations.
    transformer_mem = {
        0: "16GiB",
        1: f"{per_gpu_mem_gib}GiB",
    }
    LOG.info("loading transformer (sharded) max_memory=%s ...", transformer_mem)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="auto",
        max_memory=transformer_mem,
        low_cpu_mem_usage=True,
    )
    LOG.info("transformer params per device (B): %s",
             _summarize_device_map(transformer))

    LOG.info("loading tokenizer / processor / scheduler ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(model_path, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )

    LOG.info("assembling pipeline ...")
    pipe = cls(
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        processor=processor,
        scheduler=scheduler,
    )
    LOG.info("pipeline assembled in %.1fs total", time.time() - t0)

    # Defensive activation savers (only enable if attribute exists).
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, fn)()
            LOG.info("enabled: %s", fn)
        except Exception:
            pass

    return pipe, cls_name


def _summarize_device_map(model: torch.nn.Module) -> dict[str, float]:
    """Count parameters per device (in billions) for diagnostic logging."""
    counts: dict[str, int] = {}
    for p in model.parameters():
        d = str(p.device)
        counts[d] = counts.get(d, 0) + p.numel()
    return {k: round(v / 1e9, 2) for k, v in counts.items()}


def gpu_status() -> list[dict[str, Any]]:
    """Snapshot GPU memory for /health endpoint."""
    if not torch.cuda.is_available():
        return []
    out: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        allocated = torch.cuda.memory_allocated(i)
        out.append(
            {
                "idx": i,
                "name": torch.cuda.get_device_name(i),
                "total_mb": round(total / 1024 / 1024),
                "free_mb": round(free / 1024 / 1024),
                "allocated_mb": round(allocated / 1024 / 1024),
            }
        )
    return out
