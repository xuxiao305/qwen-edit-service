# TypeScript client

Single-file zero-dependency client. Drop [qwenEdit.ts](qwenEdit.ts) into your
project, or install via the (currently optional) npm package
`@xuxiao305/qwen-edit-client`.

## API surface

```ts
getHealth(): Promise<QwenHealth>
warmup(): Promise<void>
editImage(file: File | Blob, params: QwenEditParams): Promise<QwenEditResult>
```

The base URL is `/qwen` — front-end code is expected to be served by a dev
proxy (e.g. Vite) that forwards `/qwen/*` to `http://127.0.0.1:8765/*`.

For non-browser use, copy [qwenEdit.ts](qwenEdit.ts) and edit the `BASE`
constant, or wrap with your own fetch.

## Vite proxy snippet

```ts
// vite.config.ts
const QWEN_URL = env.VITE_QWEN_URL ?? 'http://127.0.0.1:8765';
proxy: {
  '/qwen': {
    target: QWEN_URL,
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/qwen/, ''),
    timeout: 30 * 60 * 1000,
  },
}
```
