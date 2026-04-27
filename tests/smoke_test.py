"""Smoke test: probe /health and /warmup, optionally do one edit.

Usage:
    python tests/smoke_test.py
    python tests/smoke_test.py --warmup
    python tests/smoke_test.py --image fixtures/concept_sample.png \
        --prompt "Make it night" --out output.png
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--prompt", default="Convert to a watercolor sketch")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("tests/output/smoke.png"))
    args = p.parse_args()

    print(f"GET {args.server}/health")
    h = requests.get(f"{args.server}/health", timeout=10).json()
    print("  ", h)
    if not h.get("model_loaded") and args.warmup:
        print(f"POST {args.server}/warmup ...")
        t0 = time.time()
        r = requests.post(f"{args.server}/warmup", timeout=1800)
        print(f"  -> {r.status_code} in {time.time() - t0:.1f}s  {r.text[:200]}")
        r.raise_for_status()
        h = requests.get(f"{args.server}/health", timeout=10).json()
        print("  health after warmup:", h)

    if args.image:
        print(f"POST {args.server}/edit_b64  src={args.image}  steps={args.steps}")
        body = {
            "image_b64": base64.b64encode(args.image.read_bytes()).decode("ascii"),
            "prompt": args.prompt,
            "num_inference_steps": args.steps,
        }
        t0 = time.time()
        r = requests.post(f"{args.server}/edit_b64", json=body, timeout=1800)
        print(f"  -> {r.status_code} in {time.time() - t0:.1f}s")
        r.raise_for_status()
        data = r.json()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(base64.b64decode(data["image_b64"]))
        meta = {k: v for k, v in data.items() if k != "image_b64"}
        print(f"saved: {args.out}  meta={meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
