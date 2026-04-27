"""Dual-GPU loader for Qwen-Image-Edit-2511 (component-level sharding).

Why component-level loading?
---------------------------
diffusers' pipeline-level `from_pretrained(..., device_map="balanced")`
treats whole sub-modules (transformer, text_encoder, vae) as atomic units
and places each on a single device. On 2× A30 24GB this puts the entire
~16 GB transformer on cuda:0 (alongside the 17 GB text_encoder if it
fits, otherwise it spills to CPU) while cuda:1 sits empty. The result
is effectively single-GPU inference with possible CPU offload.

We instead:
  1. Load the heavy `transformer` *separately* with `device_map="auto"`
     so accelerate shards its layers across **both** GPUs.
  2. Load the much-smaller `text_encoder` (Qwen2.5-VL, ~17 GB bf16) onto
     cuda:1 — keeping it off cuda:0 leaves room there for the
     transformer's first half + activations.
  3. Load `vae` to cuda:0 (small).
  4. Assemble the pipeline by passing components as kwargs, skipping
     `from_pretrained`'s default placement.
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
    """Load the Qwen-Image-Edit pipeline with components on different GPUs.

    Component placement:
      - text_encoder: cuda:1 (~17 GB Qwen2.5-VL)
      - transformer:  sharded across cuda:0 + cuda:1 via device_map="auto"
      - vae:          cuda:0 (small)
      - tokenizer/processor/scheduler: CPU (negligible)

    cuda:1 holds text_encoder + roughly half of transformer.
    cuda:0 holds vae + the other half of transformer + activations.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"dual-GPU mode requires >= 2 CUDA devices, found {n_gpus}"
        )

    from transformers import (
        AutoTokenizer,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageTransformer2DModel,
    )

    cls, cls_name = select_pipeline_class(model_path)
    LOG.info("pipeline class: %s", cls_name)

    text_encoder_device = "cuda:1"
    vae_device = "cuda:0"

    t0 = time.time()

    # 1) text_encoder first — pin it to cuda:1 so accelerate sees what's
    #    occupied when it shards the transformer.
    LOG.info("loading text_encoder on %s ...", text_encoder_device)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(text_encoder_device)

    # 2) vae onto cuda:0
    LOG.info("loading vae on %s ...", vae_device)
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(vae_device)

    # 3) transformer with cross-GPU sharding. After step 1+2 cuda:0 has
    #    vae (~1 GB) and cuda:1 has text_encoder (~17 GB), so we cap
    #    transformer to leave headroom for activations + KV.
    transformer_mem = {
        0: f"{max(per_gpu_mem_gib - 2, 8)}GiB",  # vae small, lots of room
        1: f"{max(per_gpu_mem_gib - 18, 3)}GiB",  # text_encoder takes most
        "cpu": "64GiB",
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
    processor = AutoProcessor.from_pretrained(model_path, subfolder="processor")
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
