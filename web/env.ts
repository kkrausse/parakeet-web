// Where things are, for the build and every driver. Nothing else in web/ names a machine path or a port.
//   PK_CACHE    models (gguf/), test audio (audio/), native reference texts (out/); default ~/devfs/cache/parakeet-ggml-webgpu
//   PK_PORT     the local server of web/dist (bun serve.ts); default 8796
//   PK_DISPLAY  X display the browsers open on (Xvfb); default :99
import { join } from "node:path";

export const ROOT = join(import.meta.dir, "..");
export const CACHE = process.env.PK_CACHE ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu");
export const PORT = Number(process.env.PK_PORT ?? 8796);
export const BASE = `http://127.0.0.1:${PORT}/`;
export const DISPLAY = process.env.PK_DISPLAY ?? ":99";
