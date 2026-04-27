# QwenEditService

> Self-contained microservice that runs **Qwen-Image-Edit-2511** on a
> dual-GPU server (designed for 2× NVIDIA A30 24GB) and exposes it via a
> small FastAPI HTTP API.

## Why this exists

Qwen-Image-Edit-2511 in `bf16` is ~50 GB on disk and ~22-24 GB of VRAM per
shard at runtime. A single 24 GB card cannot hold the transformer plus the
activations without `cpu_offload` (which is ~12 minutes per image). This
project splits the transformer across two cards using
`accelerate`'s `device_map="balanced"` so the full bf16 model runs natively
in ~30-60 s per image.

The project is intentionally tiny:

```
QwenEditService/
├── server/        FastAPI server + dual-GPU loader (the actual service)
├── deploy/        bash scripts for env setup / model download
├── clients/
│   ├── python/    `qwen_edit_client.py` CLI
│   └── typescript/ `qwenEdit.ts` (importable from any TS app via Vite proxy)
└── tests/         smoke tests + pipeline demo
```

Consumers (e.g. ConceptToHighresModel) only need `clients/typescript/qwenEdit.ts`
or any HTTP client of their choice.

## API

| Method | Path        | Body                                                 | Returns |
|--------|-------------|------------------------------------------------------|---------|
| GET    | `/health`   | -                                                    | service + GPU status |
| POST   | `/warmup`   | -                                                    | `{status:"ready", pipeline_class}` |
| POST   | `/edit_b64` | `{image_b64, prompt, num_inference_steps, ...}`     | `{image_b64, seed, steps, elapsed_sec}` |
| POST   | `/edit`     | multipart `image=<file>&payload=<json>`             | `image/png` + `x-meta-*` headers |

## Server deployment

```bash
# 1. one-time: create conda env on the GPU host
bash deploy/setup_env.sh

# 2. one-time: download model weights (~50 GB) from hf-mirror.com
bash deploy/download_model.sh

# 3. start the service (foreground)
bash server/run_server.sh
# or background
bash server/run_server.sh --bg
```

The service listens on `127.0.0.1:8765` by default. **Two GPUs are required**;
the server refuses to start with fewer (single-GPU mode is intentionally not
supported anymore — see git history for the old `cpu_offload` path).

### Environment variables

| Variable                   | Default                                              | Meaning |
|----------------------------|------------------------------------------------------|---------|
| `QWEN_EDIT_MODEL_PATH`     | `/project/qwen_edit/models/Qwen-Image-Edit-2511`     | local diffusers checkpoint dir |
| `QWEN_EDIT_HOST`           | `127.0.0.1`                                          | bind address |
| `QWEN_EDIT_PORT`           | `8765`                                               | bind port |
| `QWEN_EDIT_GPU_MEM_GIB`    | `22`                                                 | per-GPU max memory for transformer sharding |
| `QWEN_EDIT_EAGER_LOAD`     | `0`                                                  | `1` = load model on startup, else lazy via `/warmup` |

## Local access via SSH tunnel

The service binds to `127.0.0.1` only. Open a tunnel from your workstation:

```powershell
# Windows / PowerShell
deploy\ssh_tunnel.ps1
# or manually:
ssh -i C:\path\to\key -p 44304 -L 8765:127.0.0.1:8765 user@gpu-host
```

Then the CLI / TS client just talks to `http://127.0.0.1:8765`.

## CLI

```powershell
python clients\python\qwen_edit_client.py `
  --image input.png `
  --prompt "Convert to cyberpunk neon" `
  --steps 30 --cfg 4.0 --warmup `
  --out output.png
```

## TypeScript client

Drop `clients/typescript/qwenEdit.ts` into your project (or publish as an
internal npm package). Functions:

```ts
const health = await getHealth();
if (!health.modelLoaded) await warmup();

const result = await editImage(file, {
  prompt: 'Convert to cyberpunk neon',
  steps: 30,
  cfg: 4.0,
});
imgEl.src = result.imageUrl;
```

For browser dev behind a Vite proxy, route `/qwen` -> `http://127.0.0.1:8765`.

## License

MIT
