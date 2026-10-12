// Parakeet in the browser on onnxruntime-web: TDT-CTC 110m (the small one, default) and TDT 0.6b v2.
// One page, no framework.
// The pipeline mirrors ../ort-bench/src/main.rs, except that the mel front end is JS (src/mel.ts)
// because onnxruntime-web cannot run nemo128.onnx: encoder on the chosen provider, greedy TDT decode loop over decoder_joint. Every session.run() resolves to CPU
// tensors, so every timing includes the GPU finishing and the output being read back.
import type * as OrtNS from "onnxruntime-web";
import { logMel, type MelStyle } from "./mel";

declare const ORT_VERSION: string;
type Ort = typeof OrtNS;

const MAX_SYMBOLS_PER_FRAME = 10;
const TDT_DURATIONS = 5; // durations 0..4 follow the vocabulary logits, in both models
const HIDDEN = 640;

const EXPECTED_06B: Record<string, string> = {
  a07: "going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight.",
  a14: "going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight. He'll have to put in an appearance at some place of worship on Sunday morning, and he can come to us immediately afterwards.",
  a56: "Welcome to QuirkQuidQuill Inc., where finance meets innovation. Explore diverse offerings from the P3 Quattro, a unique investment portfolio quadrant, to the O3 Omni, a platform for intricate derivative trading strategies. Delve into unconventional bond markets with our B3 Bond X and experience non-standard equity trading with E3 Equity. Personalize your wealth management with W3 RAPZ and anticipate market trends with the O2 Outlier, our forward-thinking financial forecasting tool. Explore venture capital world with U3Unifund or move your money with the M3 Mover, our sophisticated monetary transfer module. At QuirkQuidQuill Inc., we turn complex finance into creative solutions. Join us in redefining financial services.",
};
// Native onnxruntime on the CPU, fp32 weights (ort-bench and onnx-asr 0.12 agree): results/110m/.
const EXPECTED_110M: Record<string, string> = {
  a07: "going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight.",
  a14: "going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after day for a fortnight. He'll have to put in an appearance at some place of worship on Sunday morning, and he can come to us immediately afterwards",
  a56: "Welcome to Quirk Quid Quill Inc. where finance meets innovation, explore diverse offerings from the P three Quatro a unique investment portfolio quadrant to the O three Omni, a platform for intricate derivative trading strategies. Delve into unconventional bond markets with our B three Bond X and our and experience non standard equity trading with E three equity, personalize your wealth management with W three Wrap Z and anticipate market trends with the O two outlier, our forward thinking financial forecasting tool. Explore venture capital world with U three unifund or move your money with the M three mover, our sophisticated monetary transfer module. At Cork Quid Quill Inc. we turn complex finance into creative solutions. Join us in redefining financial services.",
};
const CLIPS: Record<string, string> = { a07: "7.0 s", a14: "13.7 s", a56: "56.1 s" };

/** What differs between the two models besides the weights. Vocabulary size and blank id come from vocab.txt. */
interface Family {
  title: string;
  vocab: string;
  filterbank: string; // 257 x mels, raw f32; the number of mel bins is its width
  melStyle: MelStyle;
  stateLayers: number; // LSTM layers of the prediction network: state tensors are [layers, 1, 640]
  expected: Record<string, string>; // native fp32 transcript per clip
}
const FAMILIES: Record<string, Family> = {
  // 80 mels, encoder width 512, 1024 tokens + blank, one LSTM layer.
  "110m": { title: "TDT-CTC 110m", vocab: "110m/vocab.txt", filterbank: "110m/melfb-257x80.f32", melStyle: "zero", stateLayers: 1, expected: EXPECTED_110M },
  // 128 mels, encoder width 1024, 1024 tokens + blank, two LSTM layers.
  "0.6b": { title: "TDT 0.6b v2", vocab: "vocab.txt", filterbank: "melfb-257x128.f32", melStyle: "reflect", stateLayers: 2, expected: EXPECTED_06B },
};
interface ModelFiles {
  family: string;
  label: string;
  encoder: string;
  encoderData?: string; // external data file; its name inside the model is the basename
  decoder: string;
  decoderData?: string;
  fp16?: boolean;
  approxMb: number;
}
// Order is the order of the menu: smallest first. Keys without a prefix are the 0.6b ones (older links).
const MODELS: Record<string, ModelFiles> = {
  "110m-int8": { family: "110m", label: "110m int8 (OpenVoiceOS export)", encoder: "110m/encoder-model.int8.onnx", decoder: "110m/decoder_joint-model.int8.onnx", approxMb: 130 },
  "110m-fp16": { family: "110m", label: "110m fp16 encoder (own conversion)", encoder: "110m/encoder-model.fp16.onnx", decoder: "110m/decoder_joint-model.onnx", decoderData: "110m/decoder_joint-model.onnx_data", fp16: true, approxMb: 238 },
  "110m-fp32": { family: "110m", label: "110m fp32 (OpenVoiceOS export)", encoder: "110m/encoder-model.onnx", encoderData: "110m/encoder-model.onnx_data", decoder: "110m/decoder_joint-model.onnx", decoderData: "110m/decoder_joint-model.onnx_data", approxMb: 455 },
  int8: { family: "0.6b", label: "0.6b int8 (istupakov export)", encoder: "int8/encoder-model.int8.onnx", decoder: "int8/decoder_joint-model.int8.onnx", approxMb: 631 },
  fp16: { family: "0.6b", label: "0.6b fp16 encoder (own conversion)", encoder: "fp16/encoder-model.onnx", encoderData: "fp16/encoder-model.onnx.data", decoder: "decoder_joint-model.onnx", fp16: true, approxMb: 1216 },
  fp32: { family: "0.6b", label: "0.6b fp32 encoder (istupakov export)", encoder: "fp32/encoder-model.onnx", encoderData: "fp32/encoder-model.onnx.data", decoder: "decoder_joint-model.onnx", approxMb: 2397 },
};

interface Config {
  model: string;
  clip: string; // a07 | a14 | a56 | all
  enc: "webgpu" | "wasm";
  dec: "webgpu" | "wasm";
  rt: "webgpu" | "jsep"; // which onnxruntime-web build: native WebGPU EP (asyncify) or the older JS one
  runs: number;
  threads: number; // 0 = onnxruntime-web's default
  cache: boolean; // keep model files in the browser's Cache Storage
  base: string;
}

const params = new URLSearchParams(location.search);
function readConfig(): Config {
  const pick = <T extends string>(k: string, allowed: T[], d: T): T => (allowed.includes(params.get(k) as T) ? (params.get(k) as T) : d);
  const model = pick("model", Object.keys(MODELS), "110m-int8");
  const encDefault = model.endsWith("int8") ? "wasm" : "webgpu";
  return {
    model,
    clip: pick("clip", [...Object.keys(CLIPS), "all"], "a07"),
    enc: pick("enc", ["webgpu", "wasm"], encDefault),
    dec: pick("dec", ["webgpu", "wasm"], "wasm"),
    rt: pick("rt", ["webgpu", "jsep"], "webgpu"),
    runs: Math.max(0, Number(params.get("runs") ?? 10) | 0),
    threads: Math.max(0, Number(params.get("threads") ?? 0) | 0),
    cache: params.get("cache") !== "0",
    base: params.get("base") ?? "models/",
  };
}
const cfg = readConfig();

// ---------- persistent step log ----------
interface Step { t: number; name: string; ms?: number; detail?: string }
interface RunLog { id: string; started: string; config: Config; ua: string; steps: Step[]; done: boolean; error?: string }
const HISTORY_KEY = "pkb:runs";
function loadHistory(): RunLog[] {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]"); } catch { return []; }
}
let history = loadHistory();
let current: RunLog | null = null;
function persist() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-8))); } catch { /* private mode or full */ }
}
const $ = (id: string) => document.getElementById(id)!;
const r1 = (x: number) => Math.round(x * 10) / 10;
function fmtMs(ms: number) { return ms >= 10000 ? `${(ms / 1000).toFixed(1)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${r1(ms)} ms`; }
function stepLine(s: Step) {
  return `${(s.t / 1000).toFixed(1).padStart(6)}s  ${s.name}${s.ms !== undefined ? `: ${fmtMs(s.ms)}` : ""}${s.detail ? `  (${s.detail})` : ""}`;
}
let runStart = 0;
/** Record a completed step: on screen and in localStorage, before anything else happens. */
function step(name: string, ms?: number, detail?: string) {
  const s: Step = { t: Math.round(performance.now() - runStart), name, ms: ms === undefined ? undefined : r1(ms), detail };
  current?.steps.push(s);
  persist();
  $("steps").textContent += stepLine(s) + "\n";
  (window as any).__pkb.steps = current?.steps;
}
function status(text: string) { $("status").textContent = text; }
/** Give the browser a moment to draw the status line before work that blocks the main thread. */
const paint = () => new Promise((r) => setTimeout(r, 30));

// ---------- environment ----------
async function adapterInfo() {
  const gpu = (navigator as any).gpu as GPU | undefined;
  if (!gpu) return { available: false, reason: "navigator.gpu is undefined" };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { available: false, reason: "requestAdapter() returned null" };
    const i = adapter.info;
    return {
      available: true,
      vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description,
      isFallbackAdapter: (i as any).isFallbackAdapter ?? (adapter as any).isFallbackAdapter ?? null,
      shaderF16: adapter.features.has("shader-f16"),
      features: [...adapter.features].sort(),
      maxBufferSizeMb: Math.round(adapter.limits.maxBufferSize / 2 ** 20),
      maxStorageBufferBindingSizeMb: Math.round(adapter.limits.maxStorageBufferBindingSize / 2 ** 20),
    };
  } catch (e) {
    return { available: false, reason: String(e) };
  }
}
function memory() {
  const out: Record<string, unknown> = {};
  const pm = (performance as any).memory;
  if (pm) out.jsHeapUsedMb = Math.round(pm.usedJSHeapSize / 2 ** 20), out.jsHeapTotalMb = Math.round(pm.totalJSHeapSize / 2 ** 20);
  else out.jsHeap = "performance.memory not supported";
  return out;
}
/** measureUserAgentSpecificMemory resolves at some later GC, so it is started here and never awaited inline. */
function uaMemory(): Promise<string> {
  const measure = (performance as any).measureUserAgentSpecificMemory;
  if (!measure) return Promise.resolve("not supported");
  if (!crossOriginIsolated) return Promise.resolve("needs cross-origin isolation");
  return measure.call(performance).then((m: any) => `${Math.round(m.bytes / 2 ** 20)} MB`, (e: any) => `failed: ${e}`);
}
function memText(m: Record<string, unknown>) {
  return Object.entries(m).map(([k, v]) => `${k}=${v}`).join(" ");
}

// ---------- fetching ----------
interface Fetched { bytes: Uint8Array; ms: number; from: string }
async function readBody(resp: Response, label: string): Promise<Uint8Array> {
  const total = Number(resp.headers.get("content-length") ?? 0);
  const reader = resp.body!.getReader();
  let got = 0, last = 0;
  const tick = () => {
    const now = performance.now();
    if (now - last > 200) last = now, status(`${label}: ${(got / 2 ** 20).toFixed(0)}${total ? ` / ${(total / 2 ** 20).toFixed(0)}` : ""} MB`);
  };
  if (total) {
    // One allocation of the final size: no second copy of a multi-gigabyte file.
    // Chrome refuses a plain ArrayBuffer above 2 GB ("Array buffer allocation failed"); a WebAssembly
    // memory of the same size is allowed (onnxruntime-web's own loader does the same).
    const pages = Math.ceil(total / 65536);
    const out = total < 2 ** 31 - 65536 ? new Uint8Array(total) : new Uint8Array(new WebAssembly.Memory({ initial: pages, maximum: pages }).buffer, 0, total);
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.set(value, got), got += value.length, tick();
    }
    if (got !== total) throw new Error(`${label}: got ${got} of ${total} bytes`);
    return out;
  }
  const chunks: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value), got += value.length, tick();
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) out.set(c, o), o += c.length;
  return out;
}
async function fetchBytes(path: string, useCache: boolean): Promise<Fetched> {
  const url = new URL(path, new URL(cfg.base, location.href)).href;
  const name = path.split("/").slice(-2).join("/");
  const t = performance.now();
  if (useCache && "caches" in globalThis) {
    try {
      const store = await caches.open("pkb-models-v1");
      let hit = await store.match(url);
      let from = "browser cache";
      if (!hit) {
        from = "network, via browser cache";
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
        const total = Number(resp.headers.get("content-length") ?? 0);
        const est = await navigator.storage?.estimate?.().catch(() => undefined);
        const before = est?.usage ?? 0;
        if (est?.quota !== undefined && est.quota - before < total * 1.05) {
          // Not enough storage quota (private windows, small devices): keep this one download and skip the cache.
          const bytes = await readBody(resp, `downloading ${name}`);
          return { bytes, ms: performance.now() - t, from: `network (not cached: ${Math.round((est.quota - before) / 2 ** 20)} MB of storage quota left)` };
        }
        // Stream network -> disk without holding the file in JS memory; show storage growth as progress.
        const poll = setInterval(async () => {
          const used = ((await navigator.storage?.estimate?.())?.usage ?? 0) - before;
          status(`downloading ${name}: about ${(used / 2 ** 20).toFixed(0)}${total ? ` / ${(total / 2 ** 20).toFixed(0)}` : ""} MB stored`);
        }, 500);
        try { await store.put(url, resp); } finally { clearInterval(poll); }
        hit = await store.match(url);
        if (!hit) throw new Error("cache.put succeeded but cache.match found nothing");
      }
      const bytes = await readBody(hit, `reading ${name}`);
      return { bytes, ms: performance.now() - t, from };
    } catch (e) {
      step(`cache unusable for ${name}, fetching directly`, undefined, String(e));
    }
  }
  const t2 = performance.now();
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  return { bytes: await readBody(resp, `downloading ${name}`), ms: performance.now() - t2, from: "network" };
}

// ---------- model ----------
interface Model { ort: Ort; family: Family; filterbank: Float32Array; enc: OrtNS.InferenceSession; dec: OrtNS.InferenceSession; vocab: string[]; blank: number }
interface Timing { pre: number; enc: number; dec: number; total: number; frames: number; steps: number; text: string }

async function transcribe(m: Model, audio: Float32Array): Promise<Timing> {
  const { ort } = m;
  const start = performance.now();
  let t = start;
  const mel = logMel(audio, m.filterbank, m.family.melStyle);
  const feats = new ort.Tensor("float32", mel.features, [1, mel.mels, mel.frames]);
  const nFeat = BigInt(mel.validFrames);
  const pre = performance.now() - t;

  t = performance.now();
  const encOut = await m.enc.run({ audio_signal: feats, length: new ort.Tensor("int64", BigInt64Array.from([nFeat]), [1]) });
  const encData = encOut.outputs.data as Float32Array; // already on the CPU: run() downloads outputs
  const [, dim, maxFrames] = encOut.outputs.dims;
  const frames = Math.min(Number((encOut.encoded_lengths.data as BigInt64Array)[0]), maxFrames);
  // [1, D, T] -> T rows of D
  const rows = new Float32Array(frames * dim);
  for (let d = 0; d < dim; d++) for (let f = 0; f < frames; f++) rows[f * dim + d] = encData[d * maxFrames + f];
  const enc = performance.now() - t;

  t = performance.now();
  const layers = m.family.stateLayers;
  let s1: Float32Array = new Float32Array(layers * HIDDEN), s2: Float32Array = new Float32Array(layers * HIDDEN);
  const tokens: number[] = [];
  const targetLength = new ort.Tensor("int32", Int32Array.from([1]), [1]);
  let frame = 0, emitted = 0, steps = 0;
  const argmax = (a: Float32Array, from: number, to: number) => {
    let best = from;
    for (let i = from + 1; i < to; i++) if (a[i] > a[best]) best = i;
    return best - from;
  };
  while (frame < frames) {
    const last = tokens.length ? tokens[tokens.length - 1] : m.blank;
    const out = await m.dec.run({
      encoder_outputs: new ort.Tensor("float32", rows.subarray(frame * dim, (frame + 1) * dim), [1, dim, 1]),
      targets: new ort.Tensor("int32", Int32Array.from([last]), [1, 1]),
      target_length: targetLength,
      input_states_1: new ort.Tensor("float32", s1, [layers, 1, HIDDEN]),
      input_states_2: new ort.Tensor("float32", s2, [layers, 1, HIDDEN]),
    });
    steps++;
    const logits = out.outputs.data as Float32Array;
    const nVocab = logits.length - TDT_DURATIONS;
    const token = argmax(logits, 0, nVocab), skip = argmax(logits, nVocab, logits.length);
    if (token !== m.blank) {
      s1 = out.output_states_1.data as Float32Array;
      s2 = out.output_states_2.data as Float32Array;
      tokens.push(token), emitted++;
    }
    if (skip > 0) frame += skip, emitted = 0;
    else if (token === m.blank || emitted === MAX_SYMBOLS_PER_FRAME) frame++, emitted = 0;
  }
  const dec = performance.now() - t;
  const text = tokens.map((k) => m.vocab[k]).join("").replaceAll("▁", " ").trim();
  return { pre, enc, dec, total: performance.now() - start, frames, steps, text };
}

function stats(v: number[]) {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  return { median: r1(s[s.length >> 1]), worst: r1(s[s.length - 1]), best: r1(s[0]) };
}

async function run() {
  runStart = performance.now();
  const result: any = { config: cfg, ortVersion: ORT_VERSION, ua: navigator.userAgent, started: new Date().toISOString(), clips: [] };
  const w = window as any;
  w.__pkb = { done: false, phase: "start", result, steps: [] };
  current = { id: Date.now().toString(36), started: result.started, config: cfg, ua: navigator.userAgent, steps: [], done: false };
  history.push(current), persist();
  $("steps").textContent = "", $("result").textContent = "", $("summary").innerHTML = "";

  result.crossOriginIsolated = crossOriginIsolated;
  result.hardwareConcurrency = navigator.hardwareConcurrency;
  result.deviceMemoryGb = (navigator as any).deviceMemory ?? null;
  step("page", undefined, `crossOriginIsolated=${crossOriginIsolated} cores=${navigator.hardwareConcurrency} deviceMemory=${result.deviceMemoryGb ?? "n/a"} GB`);

  result.adapter = await adapterInfo();
  const a = result.adapter;
  step("WebGPU adapter", undefined, a.available ? `${a.vendor} / ${a.architecture} / ${a.device || "-"} / ${a.description || "-"}; shader-f16=${a.shaderF16}; fallback=${a.isFallbackAdapter}; maxBuffer=${a.maxBufferSizeMb} MB` : `none: ${a.reason}`);
  const needsGpu = cfg.enc === "webgpu" || cfg.dec === "webgpu";
  if (needsGpu && !a.available) throw new Error(`this configuration needs WebGPU and there is no adapter (${a.reason})`);
  const files = MODELS[cfg.model], family = FAMILIES[files.family];
  if (files.fp16 && cfg.enc === "webgpu" && !a.shaderF16) step("warning: fp16 encoder on an adapter without shader-f16");

  // ---- runtime ----
  let t = performance.now();
  const ort: Ort = await import(new URL(cfg.rt === "jsep" ? "ort/ort.all.min.mjs" : "ort/ort.webgpu.min.mjs", location.href).href);
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
  if (cfg.threads) ort.env.wasm.numThreads = cfg.threads;
  ort.env.webgpu.powerPreference = "high-performance";
  step(`onnxruntime-web ${ORT_VERSION} script (${cfg.rt === "jsep" ? "JSEP build" : "native WebGPU EP build"})`, performance.now() - t);

  // ---- fetch ----
  w.__pkb.phase = "fetch";
  result.model = { family: files.family, label: files.label };
  const fetched: Record<string, Fetched> = {};
  const fetchMs: Record<string, unknown> = {};
  const get = async (key: string, path: string, useCache: boolean) => {
    fetched[key] = await fetchBytes(path, useCache);
    fetchMs[key] = { ms: Math.round(fetched[key].ms), mb: r1(fetched[key].bytes.length / 2 ** 20), from: fetched[key].from };
    step(`fetched ${path}`, fetched[key].ms, `${(fetched[key].bytes.length / 2 ** 20).toFixed(1)} MB, ${fetched[key].from}`);
  };
  const fetchStart = performance.now();
  await get("vocab", family.vocab, false);
  await get("fb", family.filterbank, false);
  await get("dec", files.decoder, cfg.cache);
  if (files.decoderData) await get("decData", files.decoderData, cfg.cache);
  await get("enc", files.encoder, cfg.cache);
  if (files.encoderData) await get("encData", files.encoderData, cfg.cache);
  result.fetch = { totalMs: Math.round(performance.now() - fetchStart), files: fetchMs };
  step("all model files in memory", performance.now() - fetchStart, memText(memory()));

  const vocab = new TextDecoder().decode(fetched.vocab.bytes).split("\n").filter(Boolean).map((l) => l.slice(0, l.lastIndexOf(" ")));
  const blank = vocab.indexOf("<blk>");
  if (blank < 0) throw new Error("no <blk> in vocab");

  // ---- sessions ----
  w.__pkb.phase = "sessions";
  const opts = (ep: "webgpu" | "wasm"): OrtNS.InferenceSession.SessionOptions => ({ executionProviders: [ep], graphOptimizationLevel: "all" });
  const sessionMs: Record<string, number> = {};
  const create = async (key: string, bytes: Uint8Array, o: OrtNS.InferenceSession.SessionOptions) => {
    status(`creating ${key} session`);
    await paint();
    const t0 = performance.now();
    const s = await ort.InferenceSession.create(bytes, o);
    sessionMs[key] = Math.round(performance.now() - t0);
    step(`${key} session created on ${(o.executionProviders as string[])[0]}`, performance.now() - t0);
    return s;
  };
  const sessionsStart = performance.now();
  const fbBytes = fetched.fb.bytes;
  const filterbank = new Float32Array(fbBytes.buffer, fbBytes.byteOffset, fbBytes.length / 4);
  const encOpts = opts(cfg.enc);
  if (files.encoderData) encOpts.externalData = [{ path: files.encoderData.split("/").pop()!, data: fetched.encData.bytes }];
  const enc = await create("encoder", fetched.enc.bytes, encOpts);
  const decOpts = opts(cfg.dec);
  if (files.decoderData) decOpts.externalData = [{ path: files.decoderData.split("/").pop()!, data: fetched.decData.bytes }];
  const dec = await create("decoder", fetched.dec.bytes, decOpts);
  for (const k of Object.keys(fetched)) delete fetched[k]; // let the downloaded copies go
  result.session = { ms: sessionMs, totalMs: Math.round(performance.now() - sessionsStart) };
  result.wasm = { numThreads: ort.env.wasm.numThreads ?? null, simd: ort.env.wasm.simd ?? null, proxy: ort.env.wasm.proxy ?? false };
  const dev = (ort.env.webgpu as any).device as GPUDevice | undefined;
  if (dev) {
    result.ortDevice = { features: [...dev.features].sort(), shaderF16: dev.features.has("shader-f16") };
    dev.lost.then((info) => step(`WebGPU device lost: ${info.reason} ${info.message}`));
  }
  result.startToLoadedMs = Math.round(performance.now() - runStart);
  result.memoryAfterLoad = memory();
  step("ready", result.startToLoadedMs, `wasm threads=${result.wasm.numThreads ?? "default"}; ${memText(result.memoryAfterLoad)}`);
  const ua = uaMemory().then((v) => (step(`measureUserAgentSpecificMemory (asked after load): ${v}`), (result.userAgentSpecificMemory = v)));
  await new Promise((r) => setTimeout(r, 700)); // let the driver sample memory in a settled state
  w.__pkb.phase = "loaded";
  await new Promise((r) => setTimeout(r, 700));

  // ---- inference ----
  const m: Model = { ort, family, filterbank, enc, dec, vocab, blank };
  const clips = cfg.clip === "all" ? Object.keys(CLIPS) : [cfg.clip];
  for (const [i, clip] of clips.entries()) {
    w.__pkb.phase = `run:${clip}`;
    const audio = new Float32Array(await (await fetch(`audio/${clip}.f32`)).arrayBuffer());
    const seconds = audio.length / 16000;
    status(`${clip}: first run`);
    await paint();
    const first = await transcribe(m, audio);
    step(`${clip} (${CLIPS[clip]}) first run${i === 0 ? " (cold)" : ""}`, first.total, `mel ${fmtMs(first.pre)}, encoder ${fmtMs(first.enc)}, decode ${fmtMs(first.dec)}`);
    const rec: any = {
      clip, seconds: Math.round(seconds * 100) / 100, firstInPage: i === 0,
      firstRunMs: { pre: r1(first.pre), enc: r1(first.enc), dec: r1(first.dec), total: r1(first.total) },
      startToFirstTranscriptMs: i === 0 ? Math.round(performance.now() - runStart) : null,
      encFrames: first.frames, decSteps: first.steps, text: first.text,
      matchesNative: first.text === family.expected[clip],
    };
    step(`${clip} transcript ${rec.matchesNative ? "matches" : "DIFFERS from"} the native fp32 one`, undefined, first.text.length > 90 ? first.text.slice(0, 90) + "..." : first.text);
    result.clips.push(rec);
    const all: Timing[] = [];
    if (cfg.runs > 0) {
      status(`${clip}: warm-up`);
      await paint();
      await transcribe(m, audio); // discarded warm-up
      for (let k = 0; k < cfg.runs; k++) {
        status(`${clip}: warm run ${k + 1} / ${cfg.runs}`);
        all.push(await transcribe(m, audio));
        await paint();
      }
      const col = (f: (x: Timing) => number) => stats(all.map(f));
      Object.assign(rec, { warmRuns: all.length, preMs: col((x) => x.pre), encMs: col((x) => x.enc), decMs: col((x) => x.dec), totalMs: col((x) => x.total) });
      rec.xRealTime = r1((seconds * 1000) / rec.totalMs.median);
      rec.textStable = all.every((x) => x.text === first.text);
      step(`${clip} warm x${all.length}`, rec.totalMs.median, `total median/worst ${rec.totalMs.median} / ${rec.totalMs.worst} ms; encoder ${rec.encMs.median} / ${rec.encMs.worst} ms; decode ${rec.decMs.median} ms; ${rec.xRealTime}x real time; text stable=${rec.textStable}`);
    }
    render(result);
  }
  result.memoryAtEnd = memory();
  await Promise.race([ua, new Promise((r) => setTimeout(r, 4000))]);
  result.userAgentSpecificMemory ??= "no answer by the end of the run";
  step("done", performance.now() - runStart, memText(result.memoryAtEnd));
  current.done = true, persist();
  status("done");
  render(result);
  w.__pkb.phase = "done";
  w.__pkb.done = true;
}

// ---------- page ----------
function esc(s: string) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!); }
function render(result: any) {
  $("result").textContent = JSON.stringify(result, null, 1);
  const rows = result.clips.map((c: any) => `<tr><td>${c.clip} ${c.seconds} s</td><td>${fmtMs(c.firstRunMs.total)}</td>
    <td>${c.encMs ? `${c.encMs.median} / ${c.encMs.worst}` : "-"}</td><td>${c.totalMs ? `${c.totalMs.median} / ${c.totalMs.worst}` : "-"}</td>
    <td>${c.xRealTime ?? "-"}</td><td>${c.matchesNative ? "matches" : "differs"}</td></tr>`).join("");
  $("summary").innerHTML = `<table><tr><th>clip</th><th>first run</th><th>encoder ms<br>median / worst</th><th>total ms<br>median / worst</th><th>x real time</th><th>transcript vs native fp32</th></tr>${rows}</table>`
    + result.clips.map((c: any) => `<p class="text"><b>${c.clip}</b> ${esc(c.text)}</p>`).join("");
}
function link(over: Record<string, string | number>) {
  const p = new URLSearchParams(location.search);
  for (const [k, v] of Object.entries(over)) p.set(k, String(v));
  return `?${p}`;
}
function renderHistory() {
  const prev = history.filter((h) => h !== current).reverse();
  $("history").innerHTML = prev.length
    ? prev.map((h) => `<details${h.done ? "" : " open"}><summary class="${h.done ? "" : "bad"}">${esc(h.started)} · ${h.config.model} enc=${h.config.enc} dec=${h.config.dec} clip=${h.config.clip} · ${h.done ? "finished" : h.error ? "failed" : `did not finish; last step: ${esc(h.steps.at(-1)?.name ?? "none")}`}</summary><pre>${esc(h.steps.map(stepLine).join("\n"))}${h.error ? "\nERROR: " + esc(h.error) : ""}</pre></details>`).join("")
    : "<p>No earlier runs stored in this browser.</p>";
}
async function coi(on: boolean) {
  if (!("serviceWorker" in navigator)) return alert("no service worker support here");
  if (on) await navigator.serviceWorker.register("sw.js");
  else for (const r of await navigator.serviceWorker.getRegistrations()) await r.unregister();
  if (on) await navigator.serviceWorker.ready;
  location.reload();
}
function start() {
  ($("go") as HTMLButtonElement).disabled = true;
  run().catch((e) => {
    const msg = e?.stack && String(e.stack).includes(String(e.message)) ? String(e.stack) : `${e?.name ?? "Error"}: ${e?.message ?? e}`;
    if (current) current.error = msg;
    step(`ERROR: ${msg}`);
    status("failed");
    (window as any).__pkb.error = msg;
    (window as any).__pkb.done = true;
  }).finally(() => (($("go") as HTMLButtonElement).disabled = false));
}

function init() {
  (window as any).__pkb = { done: false, phase: "idle" };
  const preset = (label: string, note: string, over: Record<string, string | number>) => `<a class="preset" href="${link({ ...over, runs: 3, auto: 1 })}"><b>${label}</b><span>${note}</span></a>`;
  const group = (title: string, body: string) => `<p class="group">${title}</p>${body}`;
  $("presets").innerHTML =
    group("Small model: Parakeet TDT-CTC 110m",
      preset("1. 110m int8, all WASM", "130 MB download, no WebGPU needed. Smallest. 7 s clip, 3 warm runs.", { model: "110m-int8", enc: "wasm", dec: "wasm" })
      + preset("2. 110m fp16, encoder on WebGPU", "238 MB download, needs shader-f16", { model: "110m-fp16", enc: "webgpu", dec: "wasm" })
      + preset("3. 110m fp32, encoder on WebGPU", "455 MB download", { model: "110m-fp32", enc: "webgpu", dec: "wasm" })
      + preset("4. 110m fp32, all WASM", "455 MB download, no WebGPU needed", { model: "110m-fp32", enc: "wasm", dec: "wasm" }))
    + group("Large model: Parakeet TDT 0.6b v2",
      preset("5. 0.6b int8, all WASM", "631 MB download, no WebGPU needed", { model: "int8", enc: "wasm", dec: "wasm" })
      + preset("6. 0.6b fp16, encoder on WebGPU", "1.25 GB download, needs shader-f16", { model: "fp16", enc: "webgpu", dec: "wasm" })
      + preset("7. 0.6b fp32, encoder on WebGPU", "2.4 GB download", { model: "fp32", enc: "webgpu", dec: "wasm" }));
  const sel = (id: string, key: keyof Config, options: [string, string][]) => {
    $(id).innerHTML = options.map(([v, l]) => `<option value="${v}"${String(cfg[key]) === v ? " selected" : ""}>${l}</option>`).join("");
    $(id).addEventListener("change", (e) => (location.href = link({ [key]: (e.target as HTMLSelectElement).value, auto: 0 })));
  };
  sel("model", "model", Object.entries(MODELS).map(([k, v]) => [k, `${v.label}, ${v.approxMb} MB`]));
  sel("clip", "clip", [...Object.entries(CLIPS).map(([k, v]) => [k, `${k} (${v})`] as [string, string]), ["all", "all three"]]);
  sel("enc", "enc", [["webgpu", "encoder: WebGPU"], ["wasm", "encoder: WASM"]]);
  sel("dec", "dec", [["wasm", "decoder: WASM"], ["webgpu", "decoder: WebGPU"]]);
  sel("rt", "rt", [["webgpu", "ORT native WebGPU EP build"], ["jsep", "ORT JSEP build"]]);
  sel("runs", "runs", [["10", "10 warm runs"], ["3", "3 warm runs"], ["0", "first run only"]]);
  sel("threads", "threads", [["0", "threads: default"], ["1", "1 thread"], ["2", "2 threads"], ["4", "4 threads"], ["8", "8 threads"]]);
  $("env").textContent = `onnxruntime-web ${ORT_VERSION} · crossOriginIsolated=${crossOriginIsolated} (WASM threads ${crossOriginIsolated ? "available" : "unavailable: single-threaded"}) · navigator.gpu ${"gpu" in navigator ? "present" : "absent"}`;
  $("coi").textContent = crossOriginIsolated ? "Turn WASM threads off (remove service worker)" : "Turn WASM threads on (installs a service worker, reloads)";
  $("coi").addEventListener("click", () => coi(!crossOriginIsolated));
  $("go").addEventListener("click", start);
  $("clear").addEventListener("click", async () => {
    localStorage.removeItem(HISTORY_KEY), history = [];
    if ("caches" in globalThis) await caches.delete("pkb-models-v1");
    renderHistory();
    status("cleared stored runs and cached model files");
  });
  $("copy").addEventListener("click", () => navigator.clipboard?.writeText(`${$("steps").textContent}\n${$("result").textContent}`));
  renderHistory();
  addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => step(`unhandled rejection: ${e.reason?.message ?? e.reason}`));
  addEventListener("error", (e: ErrorEvent) => step(`uncaught error: ${e.message}`));
  if (params.get("auto") === "1") start();
}
init();
