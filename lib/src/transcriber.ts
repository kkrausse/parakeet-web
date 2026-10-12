// Layer 1, offline transcription: one worker holding the WASM module (transcribe.cpp + ggml WebGPU), the model
// file (kept in OPFS between visits) and the GPU device. createTranscriber picks the build (JSPI where the browser
// has it, ASYNCIFY otherwise) and the backend (WebGPU where a device can be made, one WASM thread otherwise), loads
// the model, runs a warm-up pass so shaders are compiled, and a sizing pass when a longest clip is given.
import { absoluteUrl, adapterSteps, backendEnv, createRpc, detectDevice, directoryUrl, formatGpuStats, noise, type AdapterInfo, type GpuStats, type ModelStorage, type Rpc, type Step, type Variant } from "./runtime";
import { SAMPLE_RATE, type Transcription } from "./stream";

export interface LoadProgress {
  /** runtime: the WASM module; model: the model file (download, then into the backend); warmup: first passes; ready. */
  phase: "runtime" | "model" | "warmup" | "ready";
  /** Overall, 0 to 1, for a progress bar. */
  fraction: number;
  /** While the model file downloads (not when it is already stored in this browser). */
  loadedBytes?: number; totalBytes?: number;
}
/**
 * What createTranscriber rejects with and what onError gets. kind "gpu": the GPU device reported an error (a shader
 * the browser rejected, validation, out of memory, device lost); nothing after that is to be trusted, so the
 * transcriber stops. `deviceLost` says the device is gone (make a new transcriber). Other kinds: the load, a pass.
 */
export interface TranscriberError extends Error { kind: "gpu" | "load" | "run" | "disposed"; deviceLost: boolean; gpuErrors: string[]; /** load, warm-up, sizing pass or pass. */ where: string }
export function isTranscriberError(e: unknown): e is TranscriberError { return e instanceof Error && typeof (e as any).kind === "string" && Array.isArray((e as any).gpuErrors); }
function fail(e: unknown, kind: TranscriberError["kind"], where: string, gpuErrors: string[] = []): TranscriberError {
  const err = e instanceof Error ? e : new Error(String(e));
  return Object.assign(err, { kind, where, gpuErrors, deviceLost: gpuErrors.some((x) => /device lost/i.test(x)) });
}

/** Everything a caller rarely needs. The defaults are what was measured. */
export interface TranscriberAdvanced {
  /** Which WASM build: default JSPI when the browser has WebAssembly.Suspending, else ASYNCIFY. */
  variant?: Variant;
  /** false skips the warm-up and sizing passes (the first real pass then compiles the shaders). */
  warmup?: boolean;
  /** false: f32-only shaders even where the adapter has shader-f16. */
  f16?: boolean;
  flashAttention?: boolean;
  /** WebGPU device limits: "default" asks for nothing above the spec defaults (the default on iOS), "raised" lets the backend raise the two byte limits. */
  gpuLimits?: "default" | "raised";
  /** Where the model file lives while loaded. Default "opfs" (stored once, read in place); falls back to a fetch Blob without OPFS. */
  storage?: ModelStorage;
  /** Environment for the backend, set before load (wins over everything above). */
  env?: Record<string, string>;
  threads?: number;
  /** WASM stdout / stderr and the library log as steps. */
  verbose?: boolean;
  /** Count WebGPU objects from the JS side as well (GpuStats.js). Default true. */
  gpuWatch?: boolean;
  /** MiB of throw-away ArrayBuffers after each pass so the JS collector runs. Default 0 (off). */
  gcMb?: number;
  /** The step trail: one human-readable line per thing done. */
  onStep?(step: Step): void;
  /** Structured events: "adapter", "loaded", "load-heap", "warmup-pass". No audio, no text. */
  onDiagnostic?(kind: string, data: any): void;
}
export interface TranscriberOptions {
  /** The folder the package's dist/assets was copied to (worker.js, worklet.js, pk-web*.js, pk-web*.wasm). Same origin as the page. */
  assetsUrl: string | URL;
  /** The GGUF model file. Downloaded once per browser (kept in OPFS). */
  modelUrl: string | URL;
  /** Longest clip, in seconds, later given to transcribe() in one call. On WebGPU a pass that long is run at load so the working buffer is made once. */
  maxSeconds?: number;
  /** "cpu": one WASM thread even where WebGPU works. Default "auto". */
  backend?: "auto" | "cpu";
  onProgress?(p: LoadProgress): void;
  /** After load: a pass failed or the GPU device reported an error. The transcriber has stopped; dispose it and make a new one. */
  onError?(e: TranscriberError): void;
  /** Abort a load in progress (createTranscriber rejects with an AbortError). */
  signal?: AbortSignal;
  advanced?: TranscriberAdvanced;
}
export interface TranscriberInfo {
  /** What the library reports, e.g. "WebGPU: WebGPU" or "CPU". */
  backend: string;
  webgpu: boolean;
  /** f16 shaders in use (false: f32-only, or not on WebGPU). */
  shaderF16: boolean;
  variant: Variant;
  /** WebAssembly JSPI present in this browser. */
  jspi: boolean;
  adapter: AdapterInfo;
  /** "vendor architecture description" of the adapter; null without WebGPU. */
  adapterName: string | null;
  /** "default" or "raised"; null without a device. */
  gpuLimits: "default" | "raised" | null;
  /** Why the model is not on WebGPU; null when it is. */
  cpuReason: string | null;
  modelUrl: string; modelFile: string; modelBytes: number;
  /** e.g. "OPFS (already stored)" or "network, streamed to OPFS". */
  modelFrom: string; storagePath: string;
  fetchMs: number; loadMs: number; heapMb: number; heapUsedMb: number;
  gpuAtLoad: GpuStats | null;
}
export interface TranscribeResult extends Transcription {
  /** Wall time of the pass in the worker, and its parts as the library reports them. */
  ms: number; melMs: number; encodeMs: number; decodeMs: number;
  tokens: number;
  /** WASM heap size and bytes in use, MB. */
  heapMb: number; heapUsedMb: number;
  gpu: GpuStats | null;
  log: string;
}
export interface Transcriber {
  readonly info: TranscriberInfo;
  /** ready until a pass fails or a GPU error arrives (failed), or dispose() (disposed). */
  readonly state: "ready" | "failed" | "disposed";
  /**
   * 16 kHz mono float PCM in, text and word times out. Calls are run one at a time, in order.
   * `transfer`: hand the buffer to the worker instead of copying it (the caller's array is empty afterwards).
   */
  transcribe(pcm: Float32Array, options?: { transfer?: boolean }): Promise<TranscribeResult>;
  /** The GPU accounting after the most recent pass (null on the CPU backend). */
  gpuStats(): GpuStats | null;
  /** Releases the GPU working buffers kept between passes (they are made again by the next pass). */
  releaseWorkingMemory(): Promise<GpuStats | null>;
  /** Frees the model and the GPU device and stops the worker. Resolves with the accounting after the free (null: CPU, or the worker was busy or failed and was just stopped). */
  dispose(): Promise<GpuStats | null>;
}

const abortError = () => { try { return new DOMException("the load was aborted", "AbortError"); } catch { return Object.assign(new Error("the load was aborted"), { name: "AbortError" }); } };

export async function createTranscriber(options: TranscriberOptions): Promise<Transcriber> {
  const adv = options.advanced ?? {};
  const assets = directoryUrl(options.assetsUrl), modelUrl = absoluteUrl(options.modelUrl);
  const modelFile = decodeURIComponent(new URL(modelUrl).pathname.split("/").pop() || "model.gguf");
  const variant: Variant = adv.variant ?? ("Suspending" in WebAssembly ? "jspi" : "asyncify");
  const cpu = options.backend === "cpu", verbose = !!adv.verbose, wantF16 = adv.f16 !== false;
  const device = detectDevice();
  // "default" requests nothing above the spec defaults (all the 110M models need: the same text was verified that
  // way); "" lets the backend raise the two byte limits to min(adapter, 1 GiB).
  const limits = adv.env?.GGML_WEBGPU_LIMITS ?? (adv.gpuLimits ? (adv.gpuLimits === "default" ? "default" : "") : device.ios ? "default" : "");
  const signal = options.signal;
  let state: "loading" | "ready" | "failed" | "disposed" = "loading";
  let silent = false; // set once the worker is gone
  const quiet = () => silent || !!signal?.aborted;
  const step = (name: string, ms?: number, detail?: string, sensitive?: boolean) => { if (!quiet()) adv.onStep?.({ name, ms: ms === undefined ? undefined : Math.round(ms * 10) / 10, detail, ...(sensitive ? { sensitive } : {}) }); };
  const diag = (kind: string, data: any) => { if (!quiet()) adv.onDiagnostic?.(kind, data); };
  const progress = (p: LoadProgress) => { if (!quiet()) options.onProgress?.(p); };
  if (signal?.aborted) throw abortError();

  progress({ phase: "runtime", fraction: 0.02 });
  const rpc: Rpc = createRpc(new URL("worker.js", assets).href, (m) => {
    if (quiet()) return;
    if (m.type === "progress") return progress({ phase: "model", loadedBytes: m.got, totalBytes: m.total, fraction: 0.05 + 0.8 * (m.got / Math.max(1, m.total)) });
    if (m.type === "stage") return step(m.name, undefined, m.detail);
    if (m.type === "diag") return diag(m.kind, m.data);
    if (m.type === "stdout") { if (verbose) step(`wasm: ${String(m.line).slice(0, 400)}`, undefined, undefined, true); return; }
    if (m.type === "stderr") { if (verbose || /error|abort|fail/i.test(m.line)) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
    if (m.type === "stack" && verbose) step(`stack: ${String(m.stack).slice(0, 2500)}`);
  });
  const onAbort = () => rpc.terminate();
  signal?.addEventListener("abort", onAbort, { once: true });
  let gpu: GpuStats | null = null;
  /** GPU errors of a load or a pass: steps, then the error to throw (null: none). */
  const gpuFailure = (where: string, errs: string[] | undefined): TranscriberError | null => {
    if (!errs?.length) return null;
    for (const e of errs.slice(0, 12)) step(`GPU ERROR (${where}): ${e.slice(0, 700)}`);
    return fail(new Error(`GPU error (${where}): ${errs.slice(0, 4).join("\n").slice(0, 900)}`), "gpu", where, errs);
  };

  let info: TranscriberInfo;
  try {
    const init = await rpc.call("init", { base: assets, variant, wantF16, limits, gpuwatch: adv.gpuWatch !== false, gcMb: adv.gcMb ?? 0 });
    const a = init.adapter;
    diag("adapter", { adapter: a, jspi: init.jspi, heapMb: init.heapMb, variant });
    step("WASM module ready", init.ms, `heap ${init.heapMb} MB; JSPI ${init.jspi ? "available" : "absent"}`);
    adapterSteps(a, step);
    const { useF16, flash, pathLine, env } = backendEnv(a, { f16: wantF16, flash: !!adv.flashAttention, extra: { ...(cpu ? { TRANSCRIBE_BACKENDS: "cpu" } : {}), ...(adv.env ?? {}) } });
    if (adv.flashAttention && !flash) step("flash attention needs shader-f16: using matmul + softmax attention instead");
    if (a.available && !cpu) step(pathLine);
    if (a.available && !cpu) step(a.plan?.limits === "default" ? `WebGPU limits: spec defaults, nothing raised${limits === "default" ? (device.ios && !adv.gpuLimits && !adv.env?.GGML_WEBGPU_LIMITS ? " (the default on iOS)" : " (requested)") : " (the raised request was refused)"}` : `WebGPU limits: the two byte limits raised to ${JSON.stringify(a.requested?.requiredLimits)}`);
    progress({ phase: "model", fraction: 0.05 });
    let store: ModelStorage = adv.storage ?? "opfs";
    if (store.startsWith("opfs")) { // private windows and some embedded browsers have no origin-private file system: keep the file as a Blob for this visit
      try { await navigator.storage.getDirectory(); } catch (e: any) { store = "blob"; step(`no origin-private file system (${e?.name ?? e}): the model file is fetched again on every visit`); }
    }
    const ld = await rpc.call("load", { url: modelUrl, name: modelFile, store, env, threads: adv.threads ?? 1, verbose });
    step(`model file ready (${ld.from})`, ld.fetchMs, `${ld.mb.toFixed(0)} MB from ${modelUrl}; storage path ${ld.storagePath}`);
    step(`model loaded on ${ld.backend}`, ld.loadMs, `WASM heap ${ld.heapMb} MB, ${ld.heapUsedMb} MB in use`);
    gpu = ld.gpu ?? null;
    if (ld.gpu) step(`after load: ${formatGpuStats(ld.gpu)}`, undefined, `JS object counting ${init.gpuwatch ? (init.finalizationRegistry ? "on" : "on, no FinalizationRegistry") : "off"}; collector nudge ${init.gcMb ? `${init.gcMb} MiB per pass` : "off"}`);
    if (ld.log && verbose) step(`library log: ${ld.log.trim().slice(0, 600)}`);
    const webgpu = /webgpu/i.test(ld.backend);
    const cpuReason = webgpu ? null : cpu ? "the CPU backend was requested" : a.available ? `the backend reported "${ld.backend}"` : String(a.reason);
    info = {
      backend: ld.backend, webgpu, shaderF16: webgpu && useF16, variant, jspi: !!init.jspi, adapter: a,
      adapterName: a.available ? `${a.vendor} ${a.architecture} ${a.description || a.device || ""}`.trim() : null,
      gpuLimits: a.plan ? (a.plan.limits === "default" ? "default" : "raised") : null, cpuReason,
      modelUrl, modelFile, modelBytes: Math.round(ld.mb * 2 ** 20), modelFrom: ld.from, storagePath: ld.storagePath,
      fetchMs: ld.fetchMs, loadMs: ld.loadMs, heapMb: ld.heapMb, heapUsedMb: ld.heapUsedMb, gpuAtLoad: ld.gpu ?? null,
    };
    if (!webgpu) step(`WARNING: not on WebGPU (backend "${ld.backend}"): ${cpuReason}`);
    diag("loaded", { info, gpuWatch: init.gpuwatch, gcMb: init.gcMb, cpuRequested: cpu });
    let bad = gpuFailure("load", ld.gpuErrors);
    if (bad) throw bad;
    if (adv.warmup !== false) {
      progress({ phase: "warmup", fraction: 0.92 });
      // One pass over 1.5 s of faint noise, so shader pipelines are compiled before the first words arrive.
      step("warm-up pass starting (1.5 s of noise: shaders compile here)");
      const n1 = noise(SAMPLE_RATE * 1.5);
      const w = await rpc.call("run", { pcm: n1.pcm }, [n1.pcm.buffer]);
      step("warm-up pass (1.5 s of noise)", w.wallMs, `WASM heap ${w.heapMb} MB, ${w.heapUsedMb} MB in use`);
      gpu = w.gpu ?? null;
      diag("warmup-pass", { kind: "warm-up", seconds: 1.5, ms: w.wallMs, heapMb: w.heapMb, gpu });
      if ((bad = gpuFailure("warm-up", w.gpuErrors))) throw bad;
      if (webgpu && options.maxSeconds) {
        // A second pass as long as the longest clip can get: the backend sizes its working buffer for it once,
        // and no later pass has to make a larger one.
        const longS = options.maxSeconds + 1;
        step(`sizing pass starting (${longS} s of noise: the working buffer is made for the longest utterance)`);
        const n2 = noise(Math.round(SAMPLE_RATE * longS), n1.seed);
        const z = await rpc.call("run", { pcm: n2.pcm }, [n2.pcm.buffer]);
        gpu = z.gpu ?? null;
        step(`sizing pass (${longS} s of noise)`, z.wallMs, z.gpu ? formatGpuStats(z.gpu) : `WASM heap ${z.heapMb} MB`);
        diag("warmup-pass", { kind: "sizing", seconds: longS, ms: z.wallMs, heapMb: z.heapMb, gpu });
        if ((bad = gpuFailure("sizing pass", z.gpuErrors))) throw bad;
      }
    }
  } catch (e) {
    signal?.removeEventListener("abort", onAbort);
    const aborted = !!signal?.aborted;
    state = "disposed"; silent = true;
    rpc.terminate();
    if (aborted) throw abortError();
    throw isTranscriberError(e) ? e : fail(e, "load", "load");
  }
  signal?.removeEventListener("abort", onAbort);
  if (signal?.aborted) { state = "disposed"; silent = true; rpc.terminate(); throw abortError(); }
  state = "ready";
  progress({ phase: "ready", fraction: 1 });

  let chain: Promise<unknown> = Promise.resolve(), queued = 0;
  const failed = (e: TranscriberError) => { if (state === "ready") { state = "failed"; options.onError?.(e); } return e; };
  async function pass(pcm: Float32Array, transfer: boolean): Promise<TranscribeResult> {
    if (state !== "ready") throw fail(new Error(state === "disposed" ? "the transcriber was disposed" : "the transcriber has stopped after an error"), state === "disposed" ? "disposed" : "run", "pass");
    let r: any;
    try { r = await rpc.call("run", { pcm }, transfer ? [pcm.buffer] : []); }
    catch (e) {
      if ((state as string) === "disposed") throw fail(e, "disposed", "pass");
      throw failed(fail(e, "run", "pass"));
    }
    gpu = r.gpu ?? null;
    const bad = gpuFailure("pass", r.gpuErrors);
    if (bad) throw failed(bad);
    return { text: r.text, words: r.words ?? [], ms: r.wallMs, melMs: r.mel_ms, encodeMs: r.encode_ms, decodeMs: r.decode_ms, tokens: r.n_tokens, heapMb: r.heapMb, heapUsedMb: r.heapUsedMb, gpu, log: r.log ?? "" };
  }
  return {
    info,
    get state() { return state as "ready" | "failed" | "disposed"; },
    transcribe(pcm, o) {
      queued++;
      const p = chain.then(() => pass(pcm, !!o?.transfer)).finally(() => { queued--; });
      chain = p.catch(() => {});
      return p;
    },
    gpuStats: () => gpu,
    async releaseWorkingMemory() {
      if (state !== "ready") return null;
      const p = chain.then(() => rpc.call("trim"));
      chain = p.catch(() => {});
      return (gpu = (await p) ?? null);
    },
    async dispose() {
      if (state === "disposed") return null;
      const clean = state === "ready" && queued === 0;
      state = "disposed";
      let after: GpuStats | null = null;
      // Idle and healthy: free the model and the device first, so the accounting can show nothing is left. Otherwise
      // (a pass in flight, a failed device) the worker is stopped where it stands, which also drops everything it holds.
      if (clean) try { after = await Promise.race([rpc.call("free").then(() => rpc.call("gpu")), new Promise<null>((res) => setTimeout(() => res(null), 3000))]); } catch { /* stopped below */ }
      silent = true;
      rpc.terminate();
      return after;
    },
  };
}
