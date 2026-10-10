// Shared by the benchmark page (main.ts) and the live page (live.ts): the worker RPC, the persisted
// step trail, what the adapter probe is turned into (steps, shader path, the environment the WASM
// backend is loaded with) and small formatting helpers. The runtime itself is src/worker.ts for both.

export interface Step { t: number; name: string; ms?: number; detail?: string }
/** `sid`: the diagnostics session id of the page load that made it (src/diag.ts). */
export interface RunLog<C = unknown> { id: string; sid?: string; started: string; config: C; ua: string; steps: Step[]; done: boolean; error?: string }

export const r1 = (x: number) => Math.round(x * 10) / 10;
export function fmtMs(ms: number) { return ms >= 10000 ? `${(ms / 1000).toFixed(1)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${r1(ms)} ms`; }
export function stepLine(s: Step) { return `${(s.t / 1000).toFixed(1).padStart(6)}s  ${s.name}${s.ms !== undefined ? `: ${fmtMs(s.ms)}` : ""}${s.detail ? `  (${s.detail})` : ""}`; }
export function esc(s: string) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!); }

/** Run logs kept in localStorage under `key` (the last `keep`), so a killed tab leaves its trail. */
export function historyStore<C>(key: string, keep = 8) {
  const load = (): RunLog<C>[] => { try { return JSON.parse(localStorage.getItem(key) ?? "[]"); } catch { return []; } };
  return {
    load,
    save(runs: RunLog<C>[]) { try { localStorage.setItem(key, JSON.stringify(runs.slice(-keep))); } catch { /* private mode or full */ } },
    clear() { try { localStorage.removeItem(key); } catch { /* nothing */ } },
  };
}

/** Words of `a` and `b` outside their longest common subsequence (insertions + deletions). */
export function wordDiff(a: string, b: string) {
  const x = a.split(/\s+/).filter(Boolean), y = b.split(/\s+/).filter(Boolean);
  let prev = new Uint16Array(y.length + 1);
  for (let i = 1; i <= x.length; i++) {
    const cur = new Uint16Array(y.length + 1);
    for (let j = 1; j <= y.length; j++) cur[j] = x[i - 1] === y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    prev = cur;
  }
  return { differing: x.length + y.length - 2 * prev[y.length], words: x.length };
}

export interface Rpc {
  call<T = any>(type: string, body?: Record<string, unknown>, transfer?: Transferable[]): Promise<T>;
  terminate(): void;
}
/** Starts worker.js. `onEvent` gets the messages that are not replies (progress, stdout, stderr) and failed replies' stacks. */
export function createRpc(onEvent: (m: any) => void): Rpc {
  const worker = new Worker(new URL("worker.js", location.href), { type: "module" });
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
  worker.onerror = (e) => failAll(`worker error: ${e.message}`);
  return {
    call<T>(type: string, body: Record<string, unknown> = {}, transfer: Transferable[] = []) {
      const id = nextId++;
      return new Promise<T>((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ type, id, ...body }, transfer); });
    },
    terminate() { worker.terminate(); failAll("worker terminated"); },
  };
}

/** The adapter probe from the worker's init, as steps. Throws when WebGPU is there but no device can be made. */
export function adapterSteps(a: any, step: (name: string, ms?: number, detail?: string) => void) {
  step(a.available ? `adapter: ${a.vendor} ${a.architecture} ${a.description || a.device || ""}; shader-f16 ${a.shaderF16 ? "present" : "ABSENT"}; fallback=${a.isFallbackAdapter}` : `no WebGPU adapter: ${a.reason}`);
  if (!a.available) return step("WebGPU is unavailable: the model would run on one WASM thread (very slow)");
  const L = a.limits, mib = (x: number) => Math.round(x / 2 ** 20);
  step(`adapter limits: binding ${mib(L.maxStorageBufferBindingSize)} MiB, buffer ${mib(L.maxBufferSize)} MiB, ${L.maxComputeInvocationsPerWorkgroup} invocations/workgroup, workgroup storage ${L.maxComputeWorkgroupStorageSize} B, ${L.maxStorageBuffersPerShaderStage} storage buffers/stage`,
    undefined, a.belowSpecDefault.length ? `BELOW THE WEBGPU SPEC DEFAULT: ${a.belowSpecDefault.join("; ")}` : "all at or above the WebGPU spec defaults, which is all this page needs");
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
  const pathLine = useF16 ? "shader path: f16" : `shader path: f32 only (${!a.shaderF16 ? "adapter has no shader-f16" : o.f16 ? "the device was refused with shader-f16" : "f16=0 requested"})`;
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
