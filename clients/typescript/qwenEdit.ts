/**
 * Qwen-Image-Edit HTTP client.
 *
 * The actual server runs on the DanLu GPU instance and is reached through a
 * local SSH tunnel:
 *
 *   ssh -i C:/tmp/DanLu_key -p 44304 -L 8765:127.0.0.1:8765 \
 *     root@apps-sl.danlu.netease.com
 *
 * In dev mode the Vite proxy maps `/qwen` → `http://127.0.0.1:8765` (see
 * vite.config.ts), so frontend code only needs the relative path.
 */

const BASE = '/qwen';

export interface QwenEditParams {
  /** Edit instruction in natural language. */
  prompt: string;
  /** Anti-prompt; default empty. */
  negativePrompt?: string;
  /** Inference steps; 30-50 is a good range. Default 40. */
  steps?: number;
  /** True classifier-free-guidance scale. Default 4. */
  cfg?: number;
  /** Optional seed for deterministic results. */
  seed?: number;
  /** Optional output dimensions. If omitted the server keeps input aspect ratio. */
  width?: number;
  height?: number;
}

export interface QwenEditResult {
  /** PNG image as a blob URL ready to bind to <img src>. */
  imageUrl: string;
  /** Raw PNG bytes for further processing / saving. */
  blob: Blob;
  meta: {
    seed: number;
    steps: number;
    trueCfgScale: number;
    elapsedSec: number;
    pipelineClass: string;
  };
}

export interface QwenHealth {
  status: string;
  modelLoaded: boolean;
  pipelineClass: string | null;
  modelPath: string;
  device: string;
  gpuName?: string;
  gpuCount?: number;
}

async function fileToBase64(file: File | Blob): Promise<string> {
  const buf = await file.arrayBuffer();
  // Avoid spreading huge Uint8Array into String.fromCharCode (stack overflow on large images).
  let binary = '';
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(
      null,
      bytes.subarray(i, i + chunk) as unknown as number[],
    );
  }
  return btoa(binary);
}

export async function getHealth(): Promise<QwenHealth> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Qwen health failed: ${res.status}`);
  const j = await res.json();
  return {
    status: j.status,
    modelLoaded: !!j.model_loaded,
    pipelineClass: j.pipeline_class ?? null,
    modelPath: j.model_path,
    device: j.device,
    gpuName: j.gpu_name,
    gpuCount: j.gpu_count,
  };
}

/** Trigger model load on the server. Resolves once the pipeline is ready. */
export async function warmup(): Promise<void> {
  const res = await fetch(`${BASE}/warmup`, { method: 'POST' });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Qwen warmup failed: ${res.status} ${text}`);
  }
}

/**
 * Run an image edit. Uses base64 JSON transport so it works behind the SSH
 * tunnel without multipart quirks.
 */
export async function editImage(
  image: File | Blob,
  params: QwenEditParams,
): Promise<QwenEditResult> {
  const body = {
    image_b64: await fileToBase64(image),
    prompt: params.prompt,
    negative_prompt: params.negativePrompt ?? ' ',
    num_inference_steps: params.steps ?? 40,
    true_cfg_scale: params.cfg ?? 4.0,
    seed: params.seed ?? null,
    width: params.width ?? null,
    height: params.height ?? null,
  };

  const res = await fetch(`${BASE}/edit_b64`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Qwen edit failed: ${res.status} ${text}`);
  }
  const j = await res.json();
  const bin = atob(j.image_b64 as string);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'image/png' });
  return {
    imageUrl: URL.createObjectURL(blob),
    blob,
    meta: {
      seed: j.seed,
      steps: j.steps,
      trueCfgScale: j.true_cfg_scale,
      elapsedSec: j.elapsed_sec,
      pipelineClass: j.pipeline_class,
    },
  };
}
