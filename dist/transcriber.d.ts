import { type AdapterInfo, type GpuStats, type ModelStorage, type Step, type Variant } from "./runtime";
import { type Transcription } from "./stream";
export interface LoadProgress {
    /** runtime: the WASM module; model: the model file (download, then into the backend); warmup: first passes; ready. */
    phase: "runtime" | "model" | "warmup" | "ready";
    /** Overall, 0 to 1, for a progress bar. */
    fraction: number;
    /** While the model file downloads (not when it is already stored in this browser). */
    loadedBytes?: number;
    totalBytes?: number;
}
/**
 * What createTranscriber rejects with and what onError gets. kind "gpu": the GPU device reported an error (a shader
 * the browser rejected, validation, out of memory, device lost); nothing after that is to be trusted, so the
 * transcriber stops. `deviceLost` says the device is gone (make a new transcriber). Other kinds: the load, a pass.
 */
export interface TranscriberError extends Error {
    kind: "gpu" | "load" | "run" | "disposed";
    deviceLost: boolean;
    gpuErrors: string[]; /** load, warm-up, sizing pass or pass. */
    where: string;
}
export declare function isTranscriberError(e: unknown): e is TranscriberError;
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
    modelUrl: string;
    modelFile: string;
    modelBytes: number;
    /** e.g. "OPFS (already stored)" or "network, streamed to OPFS". */
    modelFrom: string;
    storagePath: string;
    fetchMs: number;
    loadMs: number;
    heapMb: number;
    heapUsedMb: number;
    gpuAtLoad: GpuStats | null;
}
export interface TranscribeResult extends Transcription {
    /** Wall time of the pass in the worker, and its parts as the library reports them. */
    ms: number;
    melMs: number;
    encodeMs: number;
    decodeMs: number;
    tokens: number;
    /** WASM heap size and bytes in use, MB. */
    heapMb: number;
    heapUsedMb: number;
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
    transcribe(pcm: Float32Array, options?: {
        transfer?: boolean;
    }): Promise<TranscribeResult>;
    /** The GPU accounting after the most recent pass (all zeros on the CPU backend). */
    gpuStats(): GpuStats | null;
    /** Releases the GPU working buffers kept between passes (they are made again by the next pass). */
    releaseWorkingMemory(): Promise<GpuStats | null>;
    /** Frees the model and the GPU device and stops the worker. Resolves with the accounting after the free (null: the worker was busy or failed and was just stopped). */
    dispose(): Promise<GpuStats | null>;
}
export declare function createTranscriber(options: TranscriberOptions): Promise<Transcriber>;
