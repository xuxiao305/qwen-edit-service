"""Dual-GPU loader for Qwen-Image-Edit-2511.

Strategy
--------
The transformer is the only module that doesn't fit on a single 24 GB card,
so we shard it across both GPUs with `device_map="balanced"` (this uses
accelerate's `infer_auto_device_map` under the hood). The text encoder and
VAE are tiny in comparison, so they're pinned to `cuda:0`.

Why not `enable_*_cpu_offload`?
  - `enable_model_cpu_offload`  — still requires the entire transformer to
    fit on one card during forward (it doesn't, ~22 GB transformer + ~3 GB
    activations on a 23.5 GiB card -> OOM).
  - `enable_sequential_cpu_offload` — works but ~12 minutes per image due
    to constant PCIe traffic.

Two cards eliminate both problems at the cost of minor cross-GPU activation
transfer (negligible vs PCIe-to-CPU offload).
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
    """Pick the diffusers pipeline class declared by `model_index.json`.

    Newer (2509/2511) checkpoints use `QwenImageEditPlusPipeline`;
    the original release uses `QwenImageEditPipeline`.
    """
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
    """Load the Qwen-Image-Edit pipeline sharded across two GPUs.

    Returns the assembled diffusers pipeline (no `.to(device)` call needed —
    components already live on their target devices).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"dual-GPU mode requires >= 2 CUDA devices, found {n_gpus}"
        )

    cls, cls_name = select_pipeline_class(model_path)
    LOG.info("pipeline class: %s", cls_name)

    # accelerate's max_memory dict expects per-device caps. Reserve room
    # for activations and KV caches on each card.
    max_memory = {i: f"{per_gpu_mem_gib}GiB" for i in range(n_gpus)}
    LOG.info("dual-GPU loader: per_gpu_mem=%sGiB n_gpus=%d", per_gpu_mem_gib, n_gpus)

    t0 = time.time()
    # Load the heavy transformer with auto-balanced sharding across all GPUs.
    pipe = cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    LOG.info("from_pretrained finished in %.1fs", time.time() - t0)

    # Activation memory savers (cheap, defensive).
    for fn in ("enable_vae_tiling", "enable_attention_slicing"):
        try:
            getattr(pipe, fn)()
            LOG.info("enabled: %s", fn)
        except Exception as exc:  # pragma: no cover
            LOG.warning("could not enable %s: %s", fn, exc)

    return pipe, cls_name


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
