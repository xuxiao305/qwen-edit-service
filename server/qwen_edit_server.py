"""FastAPI server exposing Qwen-Image-Edit-2511 with dual-GPU bf16 inference.

Endpoints
---------
GET  /health        -> service info + per-GPU memory
POST /warmup        -> force model load (returns when ready)
POST /edit          -> multipart: `image` file + `payload` JSON, returns image/png
POST /edit_b64      -> JSON {image_b64, prompt, ...}, returns
                       {image_b64, seed, steps, elapsed_sec}

The model is loaded lazily on the first request to `/warmup` or `/edit*`
unless `QWEN_EDIT_EAGER_LOAD=1` is set.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from loader import gpu_status, load_dual_gpu_pipeline

LOG = logging.getLogger("qwen_edit_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_PATH = os.environ.get(
    "QWEN_EDIT_MODEL_PATH",
    "/project/qwen_edit/models/Qwen-Image-Edit-2511",
)
PER_GPU_MEM_GIB = int(os.environ.get("QWEN_EDIT_GPU_MEM_GIB", "22"))
DEFAULT_STEPS = int(os.environ.get("QWEN_EDIT_DEFAULT_STEPS", "40"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_STRATEGY = "dual_balanced"

# Lazy-loaded singletons
_pipe = None
_pipe_class_name: str | None = None


def load_pipeline():
    global _pipe, _pipe_class_name
    if _pipe is not None:
        return _pipe
    LOG.info("loading Qwen-Image-Edit from %s ...", MODEL_PATH)
    t0 = time.time()
    pipe, cls_name = load_dual_gpu_pipeline(
        MODEL_PATH, per_gpu_mem_gib=PER_GPU_MEM_GIB
    )
    _pipe = pipe
    _pipe_class_name = cls_name
    LOG.info("pipeline ready in %.1fs", time.time() - t0)
    return _pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info(
        "server start; MODEL_PATH=%s DEVICE=%s strategy=%s",
        MODEL_PATH, DEVICE, LOAD_STRATEGY,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        LOG.warning(
            "dual-GPU strategy expects >=2 CUDA devices; found %d. "
            "/warmup will fail.", torch.cuda.device_count(),
        )
    if os.environ.get("QWEN_EDIT_EAGER_LOAD", "0") == "1":
        try:
            load_pipeline()
        except Exception as exc:
            LOG.error("eager load failed: %s", exc)
    yield


app = FastAPI(title="Qwen-Image-Edit Service", version="0.2.0", lifespan=lifespan)


class EditRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 PNG/JPEG of input image")
    prompt: str = Field(..., description="Edit instruction in natural language")
    negative_prompt: str = Field(default=" ", description="Negative prompt")
    num_inference_steps: int | None = Field(default=None, ge=1, le=100,
        description="Inference steps; defaults to QWEN_EDIT_DEFAULT_STEPS env var (40 normal, 4 Lightning)")
    true_cfg_scale: float = Field(default=4.0, ge=0.0, le=20.0)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0,
        description="LoRA adapter weight; 0.0 disables LoRA effect")
    seed: int | None = Field(default=None)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)


def _decode_b64_image(b64: str) -> Image.Image:
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _active_lora_names() -> list[str]:
    """Return names of currently loaded LoRA adapters (empty list if none)."""
    if _pipe is None:
        return []
    try:
        adapters = _pipe.get_list_adapters()
        return list(adapters.keys()) if adapters else []
    except Exception:
        return []


def _run_edit(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    true_cfg_scale: float,
    lora_scale: float,
    seed: int | None,
    width: int | None,
    height: int | None,
) -> tuple[Image.Image, dict[str, Any]]:
    pipe = load_pipeline()
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    gen = torch.Generator(device="cuda").manual_seed(seed)

    # Adjust LoRA strength per-request if adapters are loaded.
    active_loras = _active_lora_names()
    if active_loras:
        try:
            pipe.set_adapters(active_loras, adapter_weights=[lora_scale] * len(active_loras))
        except Exception as exc:
            LOG.warning("set_adapters failed (non-fatal): %s", exc)

    kwargs: dict[str, Any] = dict(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        generator=gen,
    )
    if width is not None and height is not None:
        kwargs["width"] = width
        kwargs["height"] = height

    t0 = time.time()
    result = pipe(**kwargs)
    elapsed = time.time() - t0
    out_img = result.images[0]
    meta = {
        "seed": seed,
        "steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "lora_scale": lora_scale,
        "active_loras": active_loras,
        "elapsed_sec": round(elapsed, 2),
        "pipeline_class": _pipe_class_name,
        "load_strategy": LOAD_STRATEGY,
    }
    return out_img, meta


@app.get("/health")
def health():
    info = {
        "status": "ok",
        "model_loaded": _pipe is not None,
        "pipeline_class": _pipe_class_name,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "load_strategy": LOAD_STRATEGY,
        "per_gpu_mem_gib": PER_GPU_MEM_GIB,
        "default_steps": DEFAULT_STEPS,
        "active_loras": _active_lora_names(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = gpu_status()
    return info


@app.post("/warmup")
def warmup():
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        raise HTTPException(
            status_code=503,
            detail=(
                "dual-GPU mode requires >= 2 CUDA devices, found "
                f"{torch.cuda.device_count()}"
            ),
        )
    try:
        load_pipeline()
    except Exception as exc:
        LOG.exception("warmup failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ready",
        "pipeline_class": _pipe_class_name,
        "load_strategy": LOAD_STRATEGY,
    }


@app.post("/edit_b64")
def edit_b64(req: EditRequest):
    try:
        img = _decode_b64_image(req.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad image_b64: {exc}") from exc

    try:
        out, meta = _run_edit(
            image=img,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps if req.num_inference_steps is not None else DEFAULT_STEPS,
            true_cfg_scale=req.true_cfg_scale,
            lora_scale=req.lora_scale,
            seed=req.seed,
            width=req.width,
            height=req.height,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail=f"OOM: {exc}") from exc
    except Exception as exc:
        LOG.exception("inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"image_b64": _encode_image_b64(out), **meta})


@app.post("/edit")
async def edit_multipart(
    image: UploadFile = File(...),
    payload: str = Form(default="{}"),
):
    try:
        params = json.loads(payload) if payload else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"bad payload JSON: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad image upload: {exc}") from exc

    prompt = params.get("prompt") or ""
    if not prompt:
        raise HTTPException(status_code=400, detail="payload.prompt is required")

    try:
        out, meta = _run_edit(
            image=img,
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", " "),
            num_inference_steps=int(params.get("num_inference_steps", DEFAULT_STEPS)),
            true_cfg_scale=float(params.get("true_cfg_scale", 4.0)),
            lora_scale=float(params.get("lora_scale", 1.0)),
            seed=params.get("seed"),
            width=params.get("width"),
            height=params.get("height"),
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail=f"OOM: {exc}") from exc
    except Exception as exc:
        LOG.exception("inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    headers = {f"x-meta-{k}": str(v) for k, v in meta.items()}
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("QWEN_EDIT_HOST", "127.0.0.1")
    port = int(os.environ.get("QWEN_EDIT_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")
