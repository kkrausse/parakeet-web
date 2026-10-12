/** One line of the step trail: what the library is doing, for a log a person reads. Never transcript text unless `sensitive`. */
export interface Step {
    name: string;
    ms?: number;
    detail?: string; /** May hold transcript text (WASM stdout with `verbose`): keep it on the device. */
    sensitive?: boolean;
}
export type StepSink = (name: string, ms?: number, detail?: string, sensitive?: boolean) => void;
/** The worker's GPU accounting after a load or a pass. MiB to one decimal; counts are cumulative since the worker started. */
export interface GpuStats {
    weightsMb: number;
    workingMb: number;
    computeMb: number;
    stagingMb: number;
    liveMb: number;
    /** Live GPU buffers. */
    buffers: number;
    /** Weight MiB by tensor type. */
    byType: Record<string, number>;
    created: number;
    destroyed: number;
    createdMb: number;
    poolHits: number;
    poolMisses: number;
    pipelines: number;
    layouts: number;
    bindGroups: number;
    encoders: number;
    submits: number;
    writes: number;
    maps: number;
    /** The same counted from the JS side (GPUDevice methods wrapped), unless `gpuWatch` is off. */
    js?: {
        registry: boolean;
        objects: number;
        collected: number;
        buffers: number;
        buffersDestroyed: number;
        buffersCollected: number;
        bufferLiveMb: number;
        bindGroups: number;
        bindGroupsCollected: number;
        layouts: number;
        encoders: number;
        commandBuffers: number;
        commandBuffersCollected: number;
        pipelines: number;
        shaderModules: number;
    };
}
/** The worker's WebGPU adapter probe. Only `available` (and `reason` when it is false) is always there. */
export interface AdapterInfo {
    available: boolean;
    reason?: string;
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
    isFallbackAdapter?: boolean | null;
    shaderF16?: boolean;
    features?: string[];
    limits?: Record<string, number>;
    belowSpecDefault?: string[];
    /** The device request that worked: limits "default" (spec defaults) or "" (the two byte limits raised), with or without shader-f16. null: none did. */
    plan?: {
        limits: string;
        f16: boolean;
    } | null;
    [key: string]: unknown;
}
export type Variant = "jspi" | "asyncify";
export type ModelStorage = "opfs" | "opfs-blob" | "opfs-writable" | "blob" | "memfs";
/** `url` as an absolute directory URL (trailing slash), relative ones against the document. */
export declare function directoryUrl(url: string | URL): string;
export declare const absoluteUrl: (url: string | URL) => string;
/** iOS / iPadOS (every browser there is WebKit) and phones in general, from the user agent. */
export interface Device {
    ios: boolean;
    phone: boolean;
    why: string;
}
export declare function detectDevice(): Device;
/** The GPU numbers as one line. `prev`: the numbers one pass earlier, for the "last pass" part. */
export declare function formatGpuStats(g: GpuStats | null | undefined, prev?: GpuStats | null): string;
export interface Rpc {
    call<T = any>(type: string, body?: Record<string, unknown>, transfer?: Transferable[]): Promise<T>;
    terminate(): void;
}
/** Starts the worker at `url`. `onEvent` gets the messages that are not replies (progress, stages, stdout, stderr) and failed replies' stacks. */
export declare function createRpc(url: string, onEvent: (m: any) => void): Rpc;
/** The adapter probe from the worker's init, as steps. Throws when WebGPU is there but no device can be made. */
export declare function adapterSteps(a: any, step: StepSink): void;
/** Which shader path the adapter gets and the environment the model is loaded with (`extra` wins). */
export declare function backendEnv(a: any, o: {
    f16: boolean;
    flash: boolean;
    extra: Record<string, string>;
}): {
    useF16: boolean;
    flash: boolean;
    pathLine: string;
    env: Record<string, string>;
};
/** Faint deterministic noise: what the warm-up and sizing passes run over. */
export declare function noise(samples: number, seed?: number): {
    pcm: Float32Array<ArrayBuffer>;
    seed: number;
};
