// What the layers share: the worker RPC, the types of what the worker reports (adapter probe, GPU accounting),
// the environment the WASM backend is loaded with, and the step-trail lines for the adapter probe.
// The runtime itself is src/worker.ts, shipped as assets/worker.js and started by URL.

/** One line of the step trail: what the library is doing, for a log a person reads. Never transcript text unless `sensitive`. */
export interface Step { name: string; ms?: number; detail?: string; /** May hold transcript text (WASM stdout with `verbose`): keep it on the device. */ sensitive?: boolean }
export type StepSink = (name: string, ms?: number, detail?: string, sensitive?: boolean) => void;

/** The worker's GPU accounting after a load or a pass. MiB to one decimal; counts are cumulative since the worker started. */
export interface GpuStats {
  weightsMb: number; workingMb: number; computeMb: number; stagingMb: number; liveMb: number;
  /** Live GPU buffers. */
  buffers: number;
  /** Weight MiB by tensor type. */
  byType: Record<string, number>;
  created: number; destroyed: number; createdMb: number; poolHits: number; poolMisses: number;
  pipelines: number; layouts: number; bindGroups: number; encoders: number; submits: number; writes: number; maps: number;
  /** The same counted from the JS side (GPUDevice methods wrapped), unless `gpuWatch` is off. */
  js?: { registry: boolean; objects: number; collected: number; buffers: number; buffersDestroyed: number; buffersCollected: number; bufferLiveMb: number;
    bindGroups: number; bindGroupsCollected: number; layouts: number; encoders: number; commandBuffers: number; commandBuffersCollected: number; pipelines: number; shaderModules: number };
}
/** The worker's WebGPU adapter probe. Only `available` (and `reason` when it is false) is always there. */
export interface AdapterInfo {
  available: boolean; reason?: string;
  vendor?: string; architecture?: string; device?: string; description?: string; isFallbackAdapter?: boolean | null;
  shaderF16?: boolean; features?: string[]; limits?: Record<string, number>; belowSpecDefault?: string[];
  /** The device request that worked: limits "default" (spec defaults) or "" (the two byte limits raised), with or without shader-f16. null: none did. */
  plan?: { limits: string; f16: boolean } | null;
  [key: string]: unknown;
}
export type Variant = "jspi" | "asyncify";
export type ModelStorage = "opfs" | "opfs-blob" | "opfs-writable" | "blob" | "memfs";

/** `url` as an absolute directory URL (trailing slash), relative ones against the document. */
export function directoryUrl(url: string | URL) {
  const u = new URL(String(url), (globalThis as any).document?.baseURI ?? location.href);
  if (!u.pathname.endsWith("/")) u.pathname += "/";
  return u.href;
}
export const absoluteUrl = (url: string | URL) => new URL(String(url), (globalThis as any).document?.baseURI ?? location.href).href;

/** iOS / iPadOS (every browser there is WebKit) and phones in general, from the user agent. */
export interface Device { ios: boolean; phone: boolean; why: string }
export function detectDevice(): Device {
  const ua = navigator.userAgent;
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const phone = ios || /Android/.test(ua);
  return { ios, phone, why: ios ? "iOS/iPadOS user agent" : phone ? "Android user agent" : "desktop user agent" };
}

/** The GPU numbers as one line. `prev`: the numbers one pass earlier, for the "last pass" part. */
export function formatGpuStats(g: GpuStats | null | undefined, prev?: GpuStats | null) {
  if (!g) return "GPU: none (CPU backend)";
  const types = Object.entries(g.byType ?? {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(", ");
  let s = `GPU: weights ${g.weightsMb} MiB${types ? ` (${types})` : ""}, working ${g.workingMb} MiB, ${g.buffers} buffers`;
  if (prev) s += `, last pass +${g.created - prev.created} buffers +${g.bindGroups - prev.bindGroups} bind groups +${g.encoders - prev.encoders} encoders +${g.writes - prev.writes} writes`;
  s += ` · since load ${g.created} buffers made, ${g.destroyed} destroyed`;
  if (g.js) s += ` · JS objects ${g.js.objects} made, ${g.js.registry ? `${g.js.collected} collected` : "collection not observable"}; JS buffers ${g.js.buffers - g.js.buffersDestroyed} live (${g.js.bufferLiveMb} MiB)`;
  return s;
}

export interface Rpc {
  call<T = any>(type: string, body?: Record<string, unknown>, transfer?: Transferable[]): Promise<T>;
  terminate(): void;
}
/** Starts the worker at `url`. `onEvent` gets the messages that are not replies (progress, stages, stdout, stderr) and failed replies' stacks. */
export function createRpc(url: string, onEvent: (m: any) => void): Rpc {
  const worker = new Worker(url, { type: "module" });
  let nextId = 1;
  const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  const failAll = (why: string) => { for (const p of pending.values()) p.reject(new Error(why)); pending.clear(); };
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type !== "result") return onEvent(m);
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.error && m.stack) onEvent({ type: "stack", stack: m.stack });
    m.error ? p.reject(new Error(m.error)) : p.resolve(m.out);
  };
  // A worker script that cannot be fetched or parsed gives an error event with no message.
  worker.onerror = (e) => failAll(e.message ? `worker error: ${e.message}` : `the worker could not be started from ${url} (is assetsUrl the folder holding worker.js, on this origin?)`);
  return {
    call<T>(type: string, body: Record<string, unknown> = {}, transfer: Transferable[] = []) {
      const id = nextId++;
      return new Promise<T>((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ type, id, ...body }, transfer); });
    },
    terminate() { worker.terminate(); failAll("worker terminated"); },
  };
}

/** The adapter probe from the worker's init, as steps. Throws when WebGPU is there but no device can be made. */
export function adapterSteps(a: any, step: StepSink) {
  step(a.available ? `adapter: ${a.vendor} ${a.architecture} ${a.description || a.device || ""}; shader-f16 ${a.shaderF16 ? "present" : "ABSENT"}; fallback=${a.isFallbackAdapter}` : `no WebGPU adapter: ${a.reason}`);
  if (!a.available) return step("WebGPU is unavailable: the model would run on one WASM thread (very slow)");
  const L = a.limits, mib = (x: number) => Math.round(x / 2 ** 20);
  step(`adapter limits: binding ${mib(L.maxStorageBufferBindingSize)} MiB, buffer ${mib(L.maxBufferSize)} MiB, ${L.maxComputeInvocationsPerWorkgroup} invocations/workgroup, workgroup storage ${L.maxComputeWorkgroupStorageSize} B, ${L.maxStorageBuffersPerShaderStage} storage buffers/stage`,
    undefined, a.belowSpecDefault.length ? `BELOW THE WEBGPU SPEC DEFAULT: ${a.belowSpecDefault.join("; ")}` : "all at or above the WebGPU spec defaults, which is all this runtime needs");
  for (const e of a.probeErrors ?? []) step(`device request FAILED: ${e}`);
  if (!a.plan) throw new Error(`WebGPU is present but no device could be created, even with the spec-default limits and no features: ${a.deviceError}`);
  step(`device probe ${a.deviceProbe}`, undefined, a.plan.limits === "default" ? "spec-default limits" : `requested ${JSON.stringify(a.requested.requiredLimits)} ${a.requested.requiredFeatures.join(",") || "no features"}`);
  if (L.maxStorageBufferBindingSize < 2 ** 28) step(`note: tensors above ${mib(L.maxStorageBufferBindingSize)} MiB cannot be bound on this adapter; long clips may run partly on the CPU`);
}

/** Which shader path the adapter gets and the environment the model is loaded with (`extra` wins). */
export function backendEnv(a: any, o: { f16: boolean; flash: boolean; extra: Record<string, string> }) {
  // Without shader-f16 the backend compiles f32-only shaders; the two pointwise convs must then use an F32 im2col
  // (ggml_conv_2d's F16 one would bounce to the CPU). Flash attention needs F16 masks, so it is off on that path.
  const useF16 = !!a.shaderF16 && o.f16 && a.plan?.f16 !== false;
  const flash = o.flash && useF16;
  const pathLine = useF16 ? "shader path: f16" : `shader path: f32 only (${!a.shaderF16 ? "adapter has no shader-f16" : o.f16 ? "the device was refused with shader-f16" : "f16 turned off"})`;
  const env: Record<string, string> = { TRANSCRIBE_NO_FLASH: flash ? "" : "1", TRANSCRIBE_F32_MASK_CONCAT: flash ? "1" : "",
    GGML_WEBGPU_NO_F16: useF16 ? "" : "1", TRANSCRIBE_F32_POINTWISE: useF16 ? "" : "1",
    TRANSCRIBE_PRE_ENCODE_TILE: "128", // clips over 15 s: subsampling convs in 10 s time tiles (exact): their activations no longer grow with the clip
    TRANSCRIBE_ENC_PROJ_GPU: "1", // the joint's encoder projection as the last encoder node; only it is read back
    TRANSCRIBE_MEL_REAL_FFT: "1", // STFT through a half-size complex FFT (the frame is real); same text on every clip checked
    // clips over 60 s: encoder in 30 s windows with 4 s of audio either side, stitched and decoded once, so GPU memory is that of one window
    TRANSCRIBE_PARAKEET_CHUNK_S: "30", TRANSCRIBE_PARAKEET_CHUNK_HALO_S: "4", TRANSCRIBE_PARAKEET_CHUNK_MIN_S: "60",
    ...(a.plan?.limits === "default" ? { GGML_WEBGPU_LIMITS: "default" } : {}), ...o.extra };
  return { useF16, flash, pathLine, env };
}

/** Faint deterministic noise: what the warm-up and sizing passes run over. */
export function noise(samples: number, seed = 1) {
  const out = new Float32Array(samples);
  for (let i = 0; i < out.length; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; out[i] = (seed / 2 ** 32 - 0.5) * 2e-3; }
  return { pcm: out, seed };
}
