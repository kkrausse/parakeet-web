// Replay page (replay.html): the real model in this browser, fed a recording instead of a microphone, through either
// the streaming engine (src/stream.ts) or the endpointer the live page had before it (replay/old.ts). No UI: the
// driver (drive-replay.ts) calls window.__pkr.run(...) and reads the log. Audio is fed on a virtual clock, so a run
// takes as long as its passes do, not as long as the recording.
import { backendEnv, createRpc, type Rpc } from "./common";
import { applyCondition, replay, type Condition, type Model, type ReplayOptions } from "../replay/lib";
import type { StreamParams, Transcription } from "./stream";
declare const MODELS: Record<string, { label: string; file: string; mb: number; family: string }>;

const params = new URLSearchParams(location.search);
const SR = 16000;
const out = document.getElementById("out")!;
const say = (s: string) => { out.textContent += s + "\n"; };
const pk: any = ((window as any).__pkr = { phase: "loading", error: null, backend: "", webgpu: false, model: params.get("model") ?? "s4", gpu: null, heapMb: 0 });
let rpc: Rpc;
let lastGpu: any = null, lastHeap = 0;
const model: Model = async (pcm) => {
  pk.inPass = pcm.length / SR;
  const r = await rpc.call("run", { pcm }, [pcm.buffer]);
  if (r.gpuErrors?.length) throw new Error(`GPU error: ${r.gpuErrors.slice(0, 3).join("; ")}`);
  lastGpu = r.gpu; lastHeap = r.heapMb; pk.lastPass = { audioS: pcm.length / SR, ms: r.wallMs, words: r.words?.length, tail: r.words?.slice(-6), n: (pk.lastPass?.n ?? 0) + 1 };
  return { text: r.text, words: r.words, ms: r.wallMs };
};
const clips = new Map<string, Float32Array>();
async function clip(url: string) {
  if (!clips.has(url)) { const b = await (await fetch(url)).arrayBuffer(); clips.set(url, new Float32Array(b, 0, b.byteLength >> 2)); }
  return clips.get(url)!;
}
interface Spec { clip: string; condition: Condition; seconds?: number; engine: "old" | "new"; capS: number; params?: Partial<StreamParams>; latencyMs?: number | "measured"; slow?: number }
async function conditioned(s: { clip: string; condition: Condition; seconds?: number }) {
  let pcm = await clip(s.clip);
  if (s.seconds) pcm = pcm.subarray(0, Math.round(s.seconds * SR));
  return applyCondition(pcm, s.condition);
}
/** One whole-file pass over the conditioned recording: the reference the streamed text is compared with. */
pk.offline = async (s: { clip: string; condition: Condition; seconds?: number }): Promise<Transcription & { ms: number }> => {
  const r = await model((await conditioned(s)).slice(), { kind: "final", startSample: 0, endSample: 0 });
  return { text: r.text, words: r.words, ms: r.ms! };
};
pk.run = async (s: Spec) => {
  const pcm = await conditioned(s);
  const memory: any[] = [];
  const o: ReplayOptions = { engine: s.engine, capS: s.capS, params: s.params, latencyMs: s.slow ? (audioS) => audioS * 1000 * s.slow! : s.latencyMs ?? "measured",
    onProgress: (t) => { pk.progress = t; memory.push({ t, jsHeapMb: Math.round(((performance as any).memory?.usedJSHeapSize ?? 0) / 2 ** 20 * 10) / 10, wasmHeapMb: lastHeap, gpuWorkMb: lastGpu?.workingMb ?? 0, gpuBuffers: lastGpu?.buffers ?? 0, gpuMade: lastGpu?.created ?? 0 }); } };
  const t0 = performance.now();
  const log = await replay(pcm, model, o);
  const frames = log.frames ? { floor: Array.from(log.frames.floor, (x) => Math.round(x * 1e5) / 1e5), speech: Array.from(log.frames.speech) } : undefined;
  return { ...log, frames, memory, wallS: (performance.now() - t0) / 1000 };
};

async function load() {
  const m = MODELS[pk.model];
  if (!m) throw new Error(`no model ${pk.model}`);
  const capS = Number(params.get("cap") ?? (params.get("phone") === "1" ? 12 : 24));
  rpc = createRpc((e) => { if (e.type === "stage") say(e.name); if (e.type === "stderr" && /error|abort|fail/i.test(e.line)) say(`wasm: ${e.line}`); });
  const variant = params.get("variant") ?? ("Suspending" in WebAssembly ? "jspi" : "asyncify");
  const init = await rpc.call("init", { base: location.href, variant, wantF16: params.get("f16") !== "0", limits: params.get("limits") === "default" ? "default" : "", gpuwatch: false, gcMb: 0 });
  const a = init.adapter;
  const { useF16, env } = backendEnv(a, { f16: params.get("f16") !== "0", flash: false, extra: params.get("cpu") === "1" ? { TRANSCRIBE_BACKENDS: "cpu" } : {} });
  const ld = await rpc.call("load", { url: new URL("models/" + m.file, location.href).href, name: m.file, store: "opfs", env, threads: 1, verbose: false });
  if (ld.gpuErrors?.length) throw new Error(`GPU error at load: ${ld.gpuErrors.join("; ")}`);
  Object.assign(pk, { variant, jspi: init.jspi, adapterReason: a.available ? null : a.reason, backend: ld.backend, webgpu: /webgpu/i.test(ld.backend), shaderF16: useF16, adapter: a.available ? `${a.vendor} ${a.architecture} ${a.description || ""}`.trim() : null, capS });
  // warm-up and sizing as on the live page: pipelines compiled, the working buffer made for the longest pass
  let seed = 1;
  for (const s of [1.5, capS + 1]) {
    const noise = new Float32Array(SR * s);
    for (let i = 0; i < noise.length; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; noise[i] = (seed / 2 ** 32 - 0.5) * 2e-3; }
    const r = await model(noise, { kind: "final", startSample: 0, endSample: 0 });
    say(`pass over ${s} s of noise: ${Math.round(r.ms!)} ms`);
  }
  pk.gpuAfterWarmup = lastGpu;
  pk.memoryNow = () => ({ wasmHeapMb: lastHeap, gpu: lastGpu });
  pk.phase = "ready"; say(`ready: ${m.label} on ${ld.backend}, cap ${capS} s`);
}
load().catch((e) => { pk.phase = "failed"; pk.error = String(e?.message ?? e); say(`ERROR: ${pk.error}`); });
