"""Local CLI: call the QwenEditService through an SSH tunnel.

Assumes you have started a tunnel:
    deploy/ssh_tunnel.ps1
or manually:
    ssh -i C:/path/key -p <port> -L 8765:127.0.0.1:8765 user@host

Usage:
    python clients/python/qwen_edit_client.py \
        --image input.png --prompt "Make it night" --out output.png
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import requests


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=" ")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--warmup", action="store_true",
                   help="Hit /warmup first (recommended on a cold server)")
    args = p.parse_args()

    if args.warmup:
        print("warming up server (this may take several minutes on cold start)...")
        r = requests.post(f"{args.server}/warmup", timeout=1800)
        r.raise_for_status()
        print("  ", r.json())

    body = {
        "image_b64": encode(args.image),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.cfg,
    }
    if args.seed is not None:
        body["seed"] = args.seed

    print(f"POST {args.server}/edit_b64  steps={args.steps} cfg={args.cfg}")
    t0 = time.time()
    r = requests.post(f"{args.server}/edit_b64", json=body, timeout=1800)
    print(f"  -> {r.status_code} in {time.time() - t0:.1f}s")
    if r.status_code != 200:
        print(r.text, file=sys.stderr)
        return 1

    data = r.json()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(base64.b64decode(data["image_b64"]))
    meta = {k: v for k, v in data.items() if k != "image_b64"}
    print(f"saved: {args.out}  meta={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
