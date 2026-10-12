// Live page: microphone -> 16 kHz PCM (src/live-worklet.ts) -> the streaming engine (src/stream.ts) -> the same
// worker and WASM runtime as the benchmark page. The engine owns the audio buffer, the gate and the commit rule;
// this file is the page around it: model loading, the microphone, the transcript, statistics, diagnostics, soak test.
import { adapterSteps, backendEnv, createRpc, esc, fmtMs, gpuDiag, gpuLine, historyStore, r1, stepLine, type Rpc, type RunLog, type Step } from "./common";
import { createDiag, detectDevice, lifeStore, sendEnvironment, watchErrors } from "./diag";
import { createStream, type StreamParams, type StreamStats, type Transcribe } from "./stream";
declare const MODELS: Record<string, { label: string; file: string; mb: number; family: "0.6b" | "110m" }>; // build.ts; smallest verified first

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => { const v = Number(params.get(k)); return params.has(k) && Number.isFinite(v) ? v : d; };
const SR = 16000;
const device = detectDevice(params); // phone=1|0 forces it
const diag = createDiag("live", params);
// The engine's numbers that can be set from the query string (tuning): gap, hang, cap, hold, stable, keep, ratio, thr, probe.
const opt = (k: string): number | undefined => (params.has(k) && Number.isFinite(Number(params.get(k))) ? Number(params.get(k)) : undefined);
const engineOptions: Partial<StreamParams> = Object.fromEntries(Object.entries({
  capS: num("cap", device.phone ? 12 : 24), // longest audio in one pass (phones: half, so a pass needs less GPU memory)
  gapMs: opt("gap"), hangS: opt("hang") === undefined ? undefined : opt("hang")! / 1000, holdS: opt("hold"), stableS: opt("stable"), keepS: opt("keep"),
  gateRatio: opt("ratio"), gateAbsRms: opt("thr"), probeEveryS: opt("probe"),
}).filter(([, v]) => v !== undefined));
const LAG_WARN_S = 2; // "lagging" is shown above this

interface LiveConfig { model: string; variant: "jspi" | "asyncify"; f16: boolean; cpu: boolean; store: string; verbose: boolean; base: string | null; limits: string; env: Record<string, string>; gpuwatch: boolean; gcMb: number }
const MODEL_KEY = "pkggml:live-model", PROC_KEY = "pkggml:live-proc";
const lsGet = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };
function readConfig(): LiveConfig {
  const names = Object.keys(MODELS);
  const pick = <T extends string>(v: string | null, allowed: T[], d: T): T => (allowed.includes(v as T) ? (v as T) : d);
  const env: Record<string, string> = {};
  for (const kv of params.getAll("env")) { const i = kv.indexOf("="); if (i > 0) env[kv.slice(0, i)] = kv.slice(i + 1); }
  return {
    model: pick(params.get("model") ?? lsGet(MODEL_KEY), names, names.includes("s4") ? "s4" : names[0]), // 110M Q4_0 unless chosen otherwise
    variant: pick(params.get("variant"), ["jspi", "asyncify"], "Suspending" in WebAssembly ? "jspi" : "asyncify"),
    f16: params.get("f16") !== "0",
    cpu: params.get("cpu") === "1", // run on the CPU backend even where WebGPU works
    store: pick(params.get("store"), ["opfs", "opfs-blob", "opfs-writable", "blob", "memfs"], "opfs"),
    verbose: params.get("verbose") === "1",
    base: params.get("base"),
    // WebGPU device limits: "default" requests nothing above the spec defaults (all the 110M models need: the same
    // text was verified that way); "" lets the backend raise the two byte limits to min(adapter, 1 GiB). On iOS the
    // default is "default"; limits=raised|default and env=GGML_WEBGPU_LIMITS=... override.
    limits: env.GGML_WEBGPU_LIMITS ?? (params.get("limits") === "raised" ? "" : params.get("limits") === "default" || device.ios ? "default" : ""),
    env,
    gpuwatch: params.get("gpuwatch") !== "0", // count WebGPU objects from the JS side as well (created, destroyed, collected)
    gcMb: Math.max(0, num("gc", 0)), // MiB of throw-away ArrayBuffers after each pass, so the JS collector runs; off unless asked for (it cost 300 MiB of peak in simulator Safari)
  };
}
const cfg = readConfig();
const $ = (id: string) => document.getElementById(id)!;

// ---------- persisted step trail ----------
const store = historyStore<LiveConfig>("pkggml:live-runs", 6);
const life = lifeStore("pkggml:live-life", diag);
let runs = store.load();
let current: RunLog<LiveConfig> | null = null;
let trailStart = performance.now();
const MAX_STEPS = 400;
/** One line of the trail: on screen, in localStorage, and to the collector. `local`: never sent (could hold text). */
function step(name: string, ms?: number, detail?: string, local = false) {
  if (!local) diag.send("step", { name, ms: ms === undefined ? undefined : r1(ms), detail });
  if (!current) return;
  if (current.steps.length >= MAX_STEPS) { if (current.steps.length === MAX_STEPS) current.steps.push({ t: Math.round(performance.now() - trailStart), name: "(further steps not recorded)" }); return; }
  const s: Step = { t: Math.round(performance.now() - trailStart), name, ms: ms === undefined ? undefined : r1(ms), detail };
  current.steps.push(s);
  store.save(runs);
  $("steps").textContent += stepLine(s) + "\n";
}
function newTrail() {
  trailStart = performance.now();
  current = { id: String(Date.now()), sid: diag.sid, started: new Date().toISOString(), config: { ...cfg }, ua: navigator.userAgent, steps: [], done: false };
  runs.push(current);
  $("steps").textContent = "";
  pk.steps = current.steps;
}
function renderHistory() {
  const prev = runs.filter((h) => h !== current).reverse();
  $("history").innerHTML = prev.length
    ? prev.map((h) => `<details><summary class="${h.error ? "bad" : ""}">${esc(h.started)} · ${h.sid ? `session ${esc(h.sid)} · ` : ""}${esc(String(h.config?.model))} · ${h.error ? "failed" : h.done ? "model loaded" : `did not finish loading; last step: ${esc(h.steps.at(-1)?.name ?? "none")}`}</summary><pre>${esc(h.steps.map(stepLine).join("\n"))}${h.error ? "\nERROR: " + esc(h.error) : ""}</pre></details>`).join("")
    : "<p>No earlier sessions stored in this browser.</p>";
}

// ---------- what the driver (and a curious person in the console) can read ----------
interface Segment { startS: number; endS: number; reason: string; opened: string; words: number } // one utterance: gate open to utterance end
const pk: any = ((window as any).__pkl = {
  phase: "loading", recording: false, idle: true, committed: "", provisional: "", backend: "", webgpu: false, shaderF16: false, variant: cfg.variant,
  segments: [] as Segment[], passes: [] as number[][], // [ms, audio seconds, 1 = final / 2 = probe, mel ms, decode ms]
  passCount: 0, finalCount: 0, reusedCount: 0, staleCount: 0, maxLagS: 0, lagS: 0, bufferS: 0, droppedS: 0, inputRate: 0, heapMb: 0, errors: [] as string[], steps: [] as Step[], params: null as StreamParams | null, load: null,
  engine: null as StreamStats | null, gateEvents: [] as any[],
  gpu: null as any, gpuAtLoad: null as any, gpuAfterWarmup: null as any, soak: null as any,
});
let gpuPrev: any = null; // the GPU numbers one pass earlier
function noteGpu(g: any) { gpuPrev = pk.gpu; pk.gpu = g ?? null; }

// ---------- notices ----------
function showError(msg: string, reload = false) {
  if (pk.errors.length >= 8) return;
  pk.errors.push(msg);
  const el = $("errors");
  el.hidden = false;
  el.textContent = (el.textContent ? el.textContent + "\n" : "") + msg.slice(0, 1200);
  if (reload && !el.querySelector("button")) {
    const b = document.createElement("button");
    b.textContent = "Reload the model";
    b.onclick = () => { el.hidden = true; el.textContent = ""; void loadModel(cfg.model); };
    el.append(document.createElement("br"), b);
  }
}
function warn(msg: string | null) { const el = $("warn"); el.hidden = !msg; el.textContent = msg ?? ""; }
function micMessage(msg: string | null) { $("reclabel").textContent = msg ?? (recording ? "Recording: tap to stop" : "Record"); $("reclabel").classList.toggle("bad", !!msg); }

// ---------- model ----------
let rpc: Rpc | null = null;
let loadGen = 0, soakAuto = false, guarded = false;
let modelState: "loading" | "ready" | "failed" = "loading";
let busy = false;
let backendLabel = "backend: not loaded yet";

async function modelBase(file: string) {
  if (cfg.base) return cfg.base;
  // Next to the page when built locally; the published live page reuses the benchmark deployment's files.
  const local = "models/", shared = "../parakeet-ggml-browser/models/";
  for (const base of /live\.html$/.test(location.pathname) ? [local, shared] : [shared, local]) {
    try { const r = await fetch(base + file, { method: "HEAD" }); if (r.ok && Number(r.headers.get("content-length") ?? 0) > 1e6) return base; } catch { /* next */ }
  }
  return "models/";
}
function loadStatus(text: string, frac: number | null, ready = false) {
  $("mstatus").textContent = text;
  if (frac !== null) ($("mbar") as HTMLElement).style.width = `${Math.round(frac * 100)}%`;
  $("load").classList.toggle("ready", ready);
}
function modelFailed(msg: string) {
  modelState = "failed"; pk.phase = "failed"; life.phase("failed");
  if (current) current.error = msg, store.save(runs);
  step(`ERROR: ${msg.slice(0, 1500)}`);
  loadStatus("model not running", 0);
  for (const w of readyWaiters.splice(0)) w(); // a pass waiting for the model fails now instead of waiting for ever
  showError(msg, true);
  setStats();
}
function gpuErrors(where: string, errs: string[] | undefined) {
  if (!errs?.length) return;
  for (const e of errs.slice(0, 12)) step(`GPU ERROR (${where}): ${e.slice(0, 700)}`);
  // A shader the browser rejected, a validation error or a lost device: what comes out after that is not to be trusted, so stop.
  const lost = errs.some((e) => /device lost/i.test(e));
  showError(`GPU error (${where}): ${errs.slice(0, 4).join("\n").slice(0, 900)}`, true);
  modelState = "failed"; pk.phase = "failed"; life.phase("failed"); loadStatus(lost ? "GPU device lost" : "stopped after a GPU error", 0);
  if (current) current.error = `GPU error (${where})`, store.save(runs);
}
async function loadModel(key: string) {
  const gen = ++loadGen;
  rpc?.terminate();
  modelState = "loading"; pk.phase = "loading"; busy = false; realPasses = 0; maxPassSamples = 0;
  if (!recording || engine.stats().failed) engine.reset(); // a pass of the old worker never comes back; while recording, audio keeps arriving and waits for this load
  cfg.model = key;
  life.phase("loading", { model: key, cpu: cfg.cpu });
  const model = MODELS[key];
  newTrail(); renderHistory();
  warn(null);
  backendLabel = "backend: loading"; setStats();
  loadStatus(`${model.label}: starting`, 0.02);
  step(`start: ${model.label}, ${cfg.variant}${cfg.cpu ? ", cpu=1" : ""}`, undefined, `session ${diag.sid}; ${device.why}; cap ${engine.params.capS} s; limits ${cfg.limits || "raised"}; store ${cfg.store}`);
  try {
    const mine = (rpc = createRpc((m) => {
      if (gen !== loadGen) return;
      if (m.type === "progress") return loadStatus(`downloading ${Math.round(m.got / 2 ** 20)} / ${Math.round(m.total / 2 ** 20)} MB`, 0.05 + 0.8 * (m.got / Math.max(1, m.total)));
      if (m.type === "stage") return step(m.name, undefined, m.detail);
      if (m.type === "diag") return diag.send(m.kind, m.data);
      if (m.type === "stdout") { if (cfg.verbose) step(`wasm: ${String(m.line).slice(0, 400)}`, undefined, undefined, true); return; }
      if (m.type === "stderr") { if (cfg.verbose || /error|abort|fail/i.test(m.line)) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
      if (m.type === "stack" && cfg.verbose) step(`stack: ${String(m.stack).slice(0, 2500)}`);
    }));
    const init = await mine.call("init", { base: location.href, variant: cfg.variant, wantF16: cfg.f16, limits: cfg.limits, gpuwatch: cfg.gpuwatch, gcMb: cfg.gcMb });
    if (gen !== loadGen) return;
    const a = init.adapter;
    diag.send("adapter", { adapter: a, jspi: init.jspi, heapMb: init.heapMb });
    step("WASM module ready", init.ms, `heap ${init.heapMb} MB; JSPI ${init.jspi ? "available" : "absent"}`);
    adapterSteps(a, step);
    $("env").textContent = `${a.available ? `${a.vendor} ${a.architecture} ${a.description || ""} · shader-f16 ${a.shaderF16 ? "present" : "absent"}` : `no WebGPU adapter (${a.reason})`} · JSPI ${init.jspi ? "available" : "absent (ASYNCIFY build)"} · ${cfg.variant} build`;
    const { useF16, pathLine, env } = backendEnv(a, { f16: cfg.f16, flash: false, extra: { ...(cfg.cpu ? { TRANSCRIBE_BACKENDS: "cpu" } : {}), ...cfg.env } });
    if (a.available && !cfg.cpu) step(pathLine);
    if (a.available && !cfg.cpu) step(a.plan?.limits === "default" ? `WebGPU limits: spec defaults, nothing raised${cfg.limits === "default" ? (device.ios && !params.has("limits") && !cfg.env.GGML_WEBGPU_LIMITS ? " (the default on iOS; limits=raised changes it)" : " (requested)") : " (the raised request was refused)"}` : `WebGPU limits: the two byte limits raised to ${JSON.stringify(a.requested?.requiredLimits)} (limits=default requests none)`);
    loadStatus(`${model.label}: fetching the model file (${model.mb} MB, once)`, 0.05);
    const base = await modelBase(model.file);
    let where = cfg.store;
    if (where.startsWith("opfs")) { // private windows and some embedded browsers have no origin-private file system: keep the file as a Blob for this visit
      try { await navigator.storage.getDirectory(); } catch (e: any) { where = "blob"; step(`no origin-private file system (${e?.name ?? e}): the model file is fetched again on every visit`); }
    }
    const ld = await mine.call("load", { url: new URL(base + model.file, location.href).href, name: model.file, store: where, env, threads: 1, verbose: cfg.verbose });
    if (gen !== loadGen) return;
    step(`model file ready (${ld.from})`, ld.fetchMs, `${ld.mb.toFixed(0)} MB from ${base}; storage path ${ld.storagePath}`);
    step(`model loaded on ${ld.backend}`, ld.loadMs, `WASM heap ${ld.heapMb} MB, ${ld.heapUsedMb} MB in use`);
    pk.gpu = gpuPrev = null; noteGpu(ld.gpu); pk.gpuAtLoad = ld.gpu ?? null;
    if (ld.gpu) step(`after load: ${gpuLine(ld.gpu)}`, undefined, `JS object counting ${init.gpuwatch ? (init.finalizationRegistry ? "on" : "on, no FinalizationRegistry") : "off"}; collector nudge ${init.gcMb ? `${init.gcMb} MiB per pass (gc=0 turns it off)` : "off (gc=<MiB> turns it on)"}`);
    if (ld.log && cfg.verbose) step(`library log: ${ld.log.trim().slice(0, 600)}`);
    const webgpu = /webgpu/i.test(ld.backend);
    Object.assign(pk, { backend: ld.backend, webgpu, shaderF16: webgpu && useF16, load: { fetchMs: Math.round(ld.fetchMs), loadMs: Math.round(ld.loadMs), from: ld.from, base, adapter: a.available ? `${a.vendor} ${a.architecture} ${a.description || ""}`.trim() : null } });
    backendLabel = webgpu
      ? `WebGPU: ${`${a.vendor} ${a.architecture} ${a.description || a.device || ""}`.trim()} · ${useF16 ? "f16" : "f32-only"} shaders · ${cfg.variant.toUpperCase()} · ${model.label}`
      : `CPU fallback, one WASM thread (slow) · ${cfg.variant.toUpperCase()} · ${model.label}`;
    if (!webgpu) {
      const why = cfg.cpu ? "cpu=1 was requested" : a.available ? `the backend reported "${ld.backend}"` : a.reason;
      warn(`Not on WebGPU: ${why}. The model runs on one WASM thread here, far slower than a GPU: text will trail behind speech and the lag figure below shows by how much.`);
      step(`WARNING: not on WebGPU (backend "${ld.backend}"): ${why}`);
    }
    diag.send("chosen", { model: key, file: model.file, build: cfg.variant, backend: ld.backend, webgpu, shaderPath: webgpu ? (useF16 ? "f16" : "f32-only") : null, limits: a.plan?.limits ?? null, cpuRequested: cfg.cpu,
      storagePath: ld.storagePath, from: ld.from, fileBytes: Math.round(ld.mb * 2 ** 20), fetchMs: Math.round(ld.fetchMs), loadMs: Math.round(ld.loadMs), heapMb: ld.heapMb, heapUsedMb: ld.heapUsedMb, capS: engine.params.capS, phone: device.phone, ios: device.ios,
      gpu: ld.gpu ?? null, gpuwatch: init.gpuwatch, gcMb: init.gcMb });
    gpuErrors("load", ld.gpuErrors);
    if (pk.phase === "failed") return;
    loadStatus(`${model.label}: warming up`, 0.92);
    life.phase("warmup");
    step("warm-up pass starting (1.5 s of noise: shaders compile here)");
    // One pass over 1.5 s of faint noise, so shader pipelines are compiled before the first words arrive.
    const noise = new Float32Array(SR * 1.5);
    let seed = 1;
    for (let i = 0; i < noise.length; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; noise[i] = (seed / 2 ** 32 - 0.5) * 2e-3; }
    const w = await mine.call("run", { pcm: noise });
    if (gen !== loadGen) return;
    step("warm-up pass (1.5 s of noise)", w.wallMs, `WASM heap ${w.heapMb} MB, ${w.heapUsedMb} MB in use`);
    noteGpu(w.gpu);
    gpuErrors("warm-up", w.gpuErrors);
    if (pk.phase === "failed") return;
    if (webgpu && params.get("sizing") !== "0") {
      // A second pass as long as the longest utterance can get: the backend sizes its working buffer for it once,
      // and no later pass has to make a larger one. (sizing=0 skips it.)
      const longS = engine.params.capS + 1;
      step(`sizing pass starting (${longS} s of noise: the working buffer is made for the longest utterance)`);
      const long = new Float32Array(SR * longS);
      for (let i = 0; i < long.length; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; long[i] = (seed / 2 ** 32 - 0.5) * 2e-3; }
      const z = await mine.call("run", { pcm: long });
      if (gen !== loadGen) return;
      noteGpu(z.gpu);
      step(`sizing pass (${longS} s of noise)`, z.wallMs, z.gpu ? gpuLine(z.gpu) : `WASM heap ${z.heapMb} MB`);
      gpuErrors("sizing pass", z.gpuErrors);
      if (pk.phase === "failed") return;
    }
    pk.gpuAfterWarmup = pk.gpu;
    current!.done = true; store.save(runs);
    modelState = "ready"; pk.phase = "ready"; life.phase("ready");
    loadStatus(`${model.label} ready`, 1, true);
    setStats();
    for (const w of readyWaiters.splice(0)) w();
    // soak=N: once per page load, and not after a visit that died (the guard's "Try again" loads without it)
    if (params.has("soak") && !soakAuto && !guarded) { soakAuto = true; void runSoak(Math.max(1, num("soak", 300) | 0 || 300)); return; }
  } catch (e: any) {
    if (gen !== loadGen) return;
    modelFailed(`${e?.name ?? "Error"}: ${e?.message ?? e}`);
  }
}

// ---------- the engine and its passes ----------
let captured = 0; // 16 kHz samples received since Record
let provText = "";
let realPasses = 0, maxPassSamples = 0; // since the model was loaded
let last: { ms: number; audioS: number } | null = null;
const readyWaiters: (() => void)[] = [];
// Microphone only (infer=0 or the checkbox): capture, resampling and the level meter run, nothing is buffered or transcribed.
let micOnly = false;

/** One pass for the engine: through the worker, with the page's bookkeeping around it. Rejects when the model is gone. */
const transcribe: Transcribe = async (pcm, req) => {
  while (modelState === "loading" || soaking || busy) await new Promise<void>((res) => readyWaiters.push(res)); // audio recorded before the model is ready waits in the engine
  if (modelState !== "ready" || !rpc) throw new Error("the model is not running");
  busy = true;
  const audioS = pcm.length / SR;
  // The pass that kills a tab never reports back: the first passes, and every pass over a longer buffer than any before, are announced first.
  if (realPasses === 0) { life.phase("first-pass"); step("first pass starting", undefined, `${audioS.toFixed(1)} s of audio`); }
  else if (realPasses < 5 || pcm.length > maxPassSamples + SR) diag.send("pass-begin", { n: pk.passCount + 1, audioS: r1(audioS), kind: req.kind, longest: pcm.length > maxPassSamples });
  if (pcm.length > maxPassSamples) maxPassSamples = pcm.length;
  const gen = loadGen, t0 = performance.now();
  let r: any;
  try { r = await rpc.call("run", { pcm }, [pcm.buffer]); }
  catch (e: any) {
    if (gen === loadGen) { busy = false; modelFailed(`transcription failed: ${e?.message ?? e}`); }
    throw e;
  }
  if (gen !== loadGen) throw new Error("the model was reloaded"); // the engine was reset with it
  busy = false;
  for (const w of readyWaiters.splice(0)) w();
  const ms = performance.now() - t0;
  pk.passCount++;
  if (++realPasses === 1) { life.phase("running"); step("first pass done", ms, `${audioS.toFixed(1)} s of audio, WASM heap ${r.heapMb} MB`); }
  noteGpu(r.gpu);
  const e = engine.stats();
  // No text and no audio here: counts, times and the gate's numbers.
  diag.send("pass", { n: pk.passCount, ms: r1(ms), audioS: r1(audioS), kind: req.kind, final: req.kind === "final", atS: r1(req.endSample / SR), words: r.words?.length ?? 0, melMs: r1(r.mel_ms), decodeMs: r1(r.decode_ms), heapMb: r.heapMb, heapUsedMb: r.heapUsedMb, lagS: r1(e.lagS),
    gate: { open: e.open, floor: +e.noiseFloor.toFixed(5), thr: +e.threshold.toFixed(5), rms: +e.lastRms.toFixed(5), misses: e.gateMisses, holds: e.modelHolds }, gpu: gpuDiag(r.gpu) }, false);
  if (pk.passes.length >= 4000) pk.passes.splice(0, 2000);
  pk.passes.push([r1(ms), r1(audioS), req.kind === "final" ? 1 : req.kind === "probe" ? 2 : 0, r1(r.mel_ms), r1(r.decode_ms)]);
  pk.heapMb = r.heapMb;
  last = { ms, audioS };
  gpuErrors("pass", r.gpuErrors);
  if (r.gpuErrors?.length) throw new Error("GPU error"); // what comes out after that is not to be trusted
  return { text: r.text, words: r.words ?? [] };
};
let gateSteps = 0;
const reasonLine = (e: StreamStats) => Object.entries(e.byReason).filter(([, n]) => n).map(([k, n]) => `${k} ${n}`).join(", ") || "none";
const engine = createStream(transcribe, {
  onCommit(c) {
    const add = (pk.committed ? " " : "") + c.text;
    pk.committed += add;
    $("committed").append(add); // append-only: earlier text nodes are never touched
    render();
  },
  onTentative(text) { provText = text; render(); },
  onGate(g) {
    // The gate as the next phone log should show it: when it opened and closed, by what, and its floor and threshold then.
    const ev = { atS: r1(g.atSample / SR), open: g.open, by: g.by, floor: +g.noiseFloor.toFixed(5), thr: +g.threshold.toFixed(5) };
    if (pk.gateEvents.length < 2000) pk.gateEvents.push(ev);
    diag.send("gate", ev, false);
    if (g.open && ++gateSteps <= 150) step(`gate open at ${ev.atS.toFixed(1)} s: ${g.by === "probe" ? "a probe pass found words the gate had not opened for (gate miss)" : "energy"}`, undefined, `noise floor ${ev.floor}, threshold ${ev.thr}`);
  },
  onUtteranceEnd(u) {
    const e = engine.stats();
    const seg: Segment = { startS: r1(u.startSample / SR), endS: r1(u.endSample / SR), reason: u.reason, opened: u.opened, words: u.words };
    if (pk.segments.length < 1000) pk.segments.push(seg);
    pk.finalCount = e.utterances; pk.reusedCount = e.reusedFinals;
    diag.send("utterance", { ...seg, commits: e.byReason, gateMisses: e.gateMisses, modelHolds: e.modelHolds, floor: +e.noiseFloor.toFixed(5), thr: +e.threshold.toFixed(5) }, false);
    if (e.utterances <= 150) step(`utterance ${e.utterances}: ${(seg.endS - seg.startS).toFixed(1)} s of audio, ${u.words} words, ended by ${u.reason}, opened by ${u.opened}`, undefined, `words committed so far by reason: ${reasonLine(e)}; noise floor ${e.noiseFloor.toFixed(5)}, threshold ${e.threshold.toFixed(5)}`);
    render();
  },
  onSkip(k) {
    const n = Math.round((k.toSample - k.fromSample) / SR);
    const add = `${pk.committed ? " " : ""}[${n} s of audio skipped: transcription is slower than real time here]`;
    pk.committed += add; $("committed").append(add);
    pk.droppedS = r1(pk.droppedS + (k.toSample - k.fromSample) / SR);
    step(`audio was skipped: ${n} s (transcription is slower than real time)`);
  },
  onError(e: any) { if (modelState === "ready") modelFailed(`transcription failed: ${e?.message ?? e}`); },
}, engineOptions);
pk.params = engine.params;

function onPcm(pcm: Float32Array) {
  captured += pcm.length;
  if (micOnly) { speechNow = false; return; }
  engine.push(pcm);
  speechNow = engine.stats().speech;
}

// ---------- rendering ----------
let speechNow = false, levelPeak = 0, lagS = 0;
function render() {
  const box = $("transcript");
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  pk.provisional = provText;
  $("prov").textContent = provText ? (pk.committed ? " " : "") + provText : "";
  $("hint").hidden = !!(pk.committed || provText || recording);
  if (atEnd) box.scrollTop = box.scrollHeight;
  setStats();
}
function setStats() {
  const e = engine.stats();
  pk.engine = e; pk.idle = !busy && engine.idle() && !recording;
  // Lag: audio of the open utterance that no displayed text has seen yet (silence with the gate closed does not count).
  lagS = e.lagS;
  pk.lagS = r1(lagS); pk.bufferS = r1(e.bufferS); pk.maxLagS = r1(e.maxLagS); pk.staleCount = e.stale;
  $("s-pass").textContent = last ? fmtMs(last.ms) : "-";
  $("s-buf").textContent = recording && micOnly ? `mic only ${(captured / SR).toFixed(0)} s` : recording || e.open ? `${e.bufferS.toFixed(1)} s${e.open ? "" : " idle"}` : "-";
  $("s-rtf").textContent = last ? `${last.audioS * 1000 / last.ms >= 10 ? Math.round((last.audioS * 1000) / last.ms) : ((last.audioS * 1000) / last.ms).toFixed(1)}x` : "-";
  const lagging = lagS > LAG_WARN_S || e.skipping;
  $("s-lag").textContent = recording || e.open ? `${lagS.toFixed(1)} s${e.skipping ? ", skipping audio" : lagging ? ", lagging" : ""}` : "-";
  $("lagcell").classList.toggle("lagging", lagging);
  $("s-backend").textContent = backendLabel + (pk.inputRate ? ` · mic ${pk.inputRate} Hz` : "");
  $("s-gpu").textContent = pk.gpu ? gpuLine(pk.gpu, gpuPrev) : modelState === "ready" ? "GPU: none (CPU backend)" : "GPU: -";
  // The gate, for whoever reads the screen or a screenshot of it: state, level, floor and threshold (RMS), and how often the model overruled it.
  $("s-gate").textContent = recording && !micOnly ? `gate ${e.open ? "open" : "closed"} · level ${e.lastRms.toFixed(4)} · floor ${e.noiseFloor.toFixed(4)} · threshold ${e.threshold.toFixed(4)} · opened ${e.gateOpens}x, ${e.gateMisses} by probe · ${e.wordsCommitted} words (${reasonLine(e)})`
    : e.passes ? `gate: opened ${e.gateOpens}x, ${e.gateMisses} by probe (gate misses), ${e.modelHolds} held open by the model · ${e.wordsCommitted} words (${reasonLine(e)})` : "gate: -";
}
function setLevel(peak: number) {
  levelPeak = Math.max(peak, levelPeak * 0.7); // quick attack, short decay
  const db = 20 * Math.log10(Math.max(levelPeak, 1e-5));
  ($("level") as HTMLElement).style.transform = `scaleX(${Math.min(1, Math.max(0, (db + 60) / 60)).toFixed(3)})`;
  $("meter").classList.toggle("speech", speechNow);
}

// ---------- microphone ----------
let recording = false, starting = false;
let ctx: AudioContext | null = null, stream: MediaStream | null = null, node: AudioWorkletNode | null = null, wake: any = null;
let recStart = 0, sawSignal = false;

function micErrorText(e: any) {
  const n = e?.name ?? "";
  if (n === "NotAllowedError" || n === "SecurityError") return "Microphone access was refused. Allow the microphone for this site in the browser's settings, then tap Record again.";
  if (n === "NotFoundError" || n === "OverconstrainedError") return "No microphone was found on this device.";
  if (n === "NotReadableError" || n === "AbortError") return "The microphone could not be opened (another app may be using it).";
  return `The microphone could not be started: ${n} ${e?.message ?? e}`.trim();
}
async function acquireWake() { try { wake = await (navigator as any).wakeLock?.request("screen"); } catch { wake = null; } }
async function startRecording() {
  if (recording || starting) return;
  if (soaking) return micMessage("A soak test is running (details below); stop it first.");
  if (!navigator.mediaDevices?.getUserMedia) return micMessage(isSecureContext ? "This browser has no microphone capture (getUserMedia)." : "Microphone capture needs https (or localhost); this page was opened over plain http.");
  const AC: typeof AudioContext | undefined = (window as any).AudioContext ?? (window as any).webkitAudioContext;
  if (!AC) return micMessage("This browser has no Web Audio (AudioContext).");
  starting = true; micMessage("Starting the microphone");
  // Created and resumed inside the tap, before any await: Safari only starts audio from a user gesture.
  // The context runs at the device rate (usually 48 kHz); the worklet resamples to 16 kHz.
  const c = (ctx = new AC({ latencyHint: "interactive" }));
  void c.resume().catch(() => {});
  try {
    if (!c.audioWorklet) throw Object.assign(new Error("AudioWorklet is not available in this browser"), { name: "NotSupportedError" });
    const proc = ($("proc") as HTMLInputElement).checked;
    const s = (stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: proc, noiseSuppression: proc, autoGainControl: true }, video: false }));
    await c.audioWorklet.addModule(new URL("live-worklet.js", location.href).href);
    if (c.state !== "running") await c.resume().catch(() => {});
    const src = c.createMediaStreamSource(s);
    const n = (node = new AudioWorkletNode(c, "pk-resample16k", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers" }));
    n.port.onmessage = (e) => {
      if (node !== n || !recording) return;
      if (e.data.rate) return;
      if (e.data.peak > 0) sawSignal = true;
      setLevel(e.data.peak);
      onPcm(e.data.pcm);
      if (!sawSignal && performance.now() - recStart > 4000) micMessage("The microphone is delivering only silence (muted, or blocked by the system?)");
      else if (sawSignal && $("reclabel").classList.contains("bad")) micMessage(null);
    };
    // Through a muted gain to the output so every browser keeps pulling the graph; nothing is played.
    const mute = c.createGain(); mute.gain.value = 0;
    src.connect(n); n.connect(mute); mute.connect(c.destination);
    s.getAudioTracks()[0]?.addEventListener("ended", () => { if (stream === s) stopRecording("The microphone went away; recording stopped."); });
    const track = s.getAudioTracks()[0];
    for (const evName of ["mute", "unmute", "ended"]) track?.addEventListener(evName, () => diag.send("mic-track", { event: evName, readyState: track.readyState, muted: track.muted }));
    c.onstatechange = () => { diag.send("audio-context", { state: c.state, sampleRate: c.sampleRate, recording }); if (ctx === c && recording && c.state !== "running") micMessage(`Audio capture is ${c.state} (paused by the browser); it resumes when this tab is in front`); else if (ctx === c && recording) micMessage(null); };
    const set = s.getAudioTracks()[0]?.getSettings?.() ?? {};
    // a fresh engine (positions start at 0 again); the committed transcript is kept
    if (!engine.idle() && modelState === "ready") await engine.stop(); // the last words of the previous recording first
    engine.reset(); captured = 0; provText = ""; gateSteps = 0; pk.gateEvents.length = 0; pk.segments.length = 0;
    micOnly = ($("miconly") as HTMLInputElement).checked;
    recording = true; starting = false; recStart = performance.now(); sawSignal = false;
    pk.recording = true; pk.inputRate = c.sampleRate;
    $("rec").classList.add("on"); $("rec").setAttribute("aria-pressed", "true"); $("rec").setAttribute("aria-label", "Stop");
    ($("model") as HTMLSelectElement).disabled = true;
    micMessage(null);
    diag.send("recording", { contextState: c.state, contextSampleRate: c.sampleRate, baseLatency: c.baseLatency ?? null, trackSettings: { ...set, deviceId: undefined, groupId: undefined }, trackLabelPresent: !!track?.label, processing: proc });
    step(`recording: context ${c.state} at ${c.sampleRate} Hz -> 16000 Hz, track ${set.sampleRate ?? "?"} Hz, ${set.channelCount ?? "?"} ch, echoCancellation=${set.echoCancellation} noiseSuppression=${set.noiseSuppression} autoGainControl=${set.autoGainControl}`);
    if (micOnly) step("microphone only: transcription is off for this recording (infer=0 / the checkbox under details)");
    void acquireWake();
    render();
  } catch (e: any) {
    starting = false;
    releaseMic();
    micMessage(micErrorText(e));
    step(`microphone: ${e?.name ?? "Error"}: ${e?.message ?? e}`, undefined, `AudioContext ${c.state}`);
  }
}
function releaseMic() {
  try { node?.port.postMessage("stop"); node?.disconnect(); } catch { /* gone */ }
  stream?.getTracks().forEach((t) => t.stop()); // the browser's recording indicator goes off
  void ctx?.close().catch(() => {});
  try { void wake?.release(); } catch { /* gone */ }
  node = null; stream = null; ctx = null; wake = null;
}
function stopRecording(message: string | null = null) {
  if (!recording) return;
  recording = false; pk.recording = false;
  releaseMic();
  speechNow = false; levelPeak = 0; setLevel(0);
  const capturedS = captured / SR;
  // what was said since the last commit still gets its final pass
  void engine.stop().then(() => {
    const e = engine.stats();
    step(`stopped after ${capturedS.toFixed(1)} s: ${e.passes} passes (${e.probes} probes), ${e.utterances} utterances, ${e.wordsCommitted} words committed (${reasonLine(e)}), gate opened ${e.gateOpens}x by energy and ${e.gateMisses}x by a probe, ${e.modelHolds} held open by the model, max lag ${e.maxLagS.toFixed(1)} s, max buffer ${e.maxBufferS.toFixed(1)} s${e.skippedS ? `, ${e.skippedS.toFixed(0)} s skipped` : ""}`);
    diag.send("stopped", { capturedS: r1(capturedS), engine: { ...e, noiseFloor: +e.noiseFloor.toFixed(5), threshold: +e.threshold.toFixed(5), lastRms: +e.lastRms.toFixed(5) } });
    render();
  });
  $("rec").classList.remove("on"); $("rec").setAttribute("aria-pressed", "false"); $("rec").setAttribute("aria-label", "Record");
  ($("model") as HTMLSelectElement).disabled = false;
  micMessage(message);
  render();
}

// ---------- soak test ----------
// N passes back to back over a fixture clip cut to a different length each time, no microphone: what the GPU
// backend holds after every pass, on the page and to the collector. soak=N in the query starts it when the model
// is ready; soakgap=<ms> waits between passes (0); the visit is marked so that a tab killed in a soak says so next time.
let soaking = false, soakStop = false;
async function soakAudio(): Promise<{ pcm: Float32Array; from: string }> {
  for (const u of /live\.html$/.test(location.pathname) ? ["audio/a56.f32", "../parakeet-ggml-browser/audio/a56.f32"] : ["../parakeet-ggml-browser/audio/a56.f32", "audio/a56.f32"]) {
    try { const r = await fetch(u); if (r.ok) { const b = await r.arrayBuffer(); if (b.byteLength > SR * 4 * 20) return { pcm: new Float32Array(b, 0, b.byteLength >> 2), from: u }; } } catch { /* next */ }
  }
  // No fixture reachable: tones under noise. The text is nonsense, the graph and its sizes are the same.
  const pcm = new Float32Array(SR * 56);
  let seed = 7;
  for (let i = 0; i < pcm.length; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; pcm[i] = (seed / 2 ** 32 - 0.5) * 0.02 + 0.1 * Math.sin(i * 0.05 * (1 + ((i >> 13) % 5))) * (((i >> 12) & 3) ? 1 : 0); }
  return { pcm, from: "synthetic (no fixture clip reachable)" };
}
async function runSoak(n: number) {
  if (soaking || recording || busy || modelState !== "ready" || !rpc) return;
  soaking = true; soakStop = false; busy = true;
  const out = $("soak"), gapMs = Math.max(0, num("soakgap", 0)), gen = loadGen;
  out.hidden = false; $("soakstop").hidden = false; ($("soakgo") as HTMLButtonElement).disabled = true; ($("model") as HTMLSelectElement).disabled = true;
  const head: string[] = [], show = (line: string) => { out.textContent = [...head, line].join("\n"); };
  const { pcm: clip, from } = await soakAudio();
  const g0 = pk.gpu, t0 = performance.now();
  const S: any = (pk.soak = { n, done: 0, running: true, from, gapMs, gcMb: cfg.gcMb, start: gpuDiag(g0), series: [] as number[][], madeAfterWarm: 0, missesAfterWarm: 0, workMin: Infinity, workMax: 0, sameText: null as boolean | null, error: null as string | null, ms: [] as number[] });
  const capS = engine.params.capS;
  head.push(`soak: ${n} passes over ${from}, lengths 0.3-${capS} s, ${gapMs} ms between passes, collector nudge ${cfg.gcMb ? cfg.gcMb + " MiB/pass" : "off"}`, `start: ${gpuLine(g0)}`);
  step(`soak starting: ${n} passes, lengths 0.3-${engine.params.capS} s`, undefined, `${from}; gap ${gapMs} ms; collector nudge ${cfg.gcMb} MiB; ${gpuLine(g0)}`);
  life.phase("soak", { at: `pass 0 of ${n}` });
  const WARM = 10; // passes after which nothing new should be made
  let seed = 12345, refText: string | null = null, warm: any = null;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  try {
    for (let i = 1; i <= n && !soakStop; i++) {
      // Every 50th pass is the same 7 s: its text must not change. The others: a random length from a random place.
      const fixed = i % 50 === 1;
      const len = fixed ? 7 * SR : Math.round((0.3 + rnd() * (engine.params.capS - 0.3)) * SR);
      const off = fixed ? 0 : Math.floor(rnd() * (clip.length - len));
      const r = await rpc.call("run", { pcm: clip.slice(off, off + len) });
      if (gen !== loadGen) throw new Error("the model was reloaded");
      const before = pk.gpu;
      noteGpu(r.gpu);
      pk.passCount++; pk.heapMb = r.heapMb; last = { ms: r.wallMs, audioS: len / SR };
      S.done = i; S.ms.push(r1(r.wallMs));
      if (fixed) { const t = String(r.text).trim(); if (refText === null) refText = t; else if (t !== refText) S.sameText = false; if (S.sameText === null && i > 1) S.sameText = true; }
      const g = r.gpu;
      if (g) {
        if (i === WARM) warm = g;
        if (i > WARM) { S.madeAfterWarm = g.created - warm.created; S.missesAfterWarm = g.poolMisses - warm.poolMisses; S.workMin = Math.min(S.workMin, g.workingMb); S.workMax = Math.max(S.workMax, g.workingMb); }
        if (S.series.length < 5000) S.series.push([i, r1(len / SR), r1(r.wallMs), g.workingMb, g.buffers, g.created - (before?.created ?? g.created), g.bindGroups - (before?.bindGroups ?? g.bindGroups), g.js ? g.js.objects - g.js.collected : -1, g.js ? g.js.buffers - g.js.buffersDestroyed : -1]);
      }
      diag.send("pass", { n: i, soak: n, ms: r1(r.wallMs), audioS: r1(len / SR), final: false, melMs: r1(r.mel_ms), decodeMs: r1(r.decode_ms), heapMb: r.heapMb, heapUsedMb: r.heapUsedMb, gpu: gpuDiag(g) }, false);
      const line = `pass ${i} of ${n}: ${fmtMs(r.wallMs)} for ${(len / SR).toFixed(1)} s · WASM heap ${r.heapMb} MB\n${gpuLine(g, before)}${i > WARM && g ? `\nsince pass ${WARM}: ${S.madeAfterWarm} buffers made, working ${S.workMin}-${S.workMax} MiB` : ""}`;
      show(line);
      if (i % 10 === 0) life.phase("soak", { at: `pass ${i} of ${n}` });
      if (i % 25 === 0 || i === 1 || i === WARM) step(`soak pass ${i} of ${n}`, r.wallMs, `${(len / SR).toFixed(1)} s; heap ${r.heapMb} MB; ${gpuLine(g, before)}`);
      if (i % 5 === 0) setStats();
      if (r.gpuErrors?.length) { gpuErrors("soak", r.gpuErrors); throw new Error(`GPU error at pass ${i}`); }
      if (gapMs) await new Promise((res) => setTimeout(res, gapMs));
    }
  } catch (e: any) { S.error = `${e?.message ?? e}`.slice(0, 400); }
  S.running = false;
  const g = pk.gpu, sorted = [...S.ms].sort((a: number, b: number) => a - b), med = sorted.length ? sorted[sorted.length >> 1] : 0;
  const flat = g && S.done > WARM ? S.madeAfterWarm === 0 && S.workMin === S.workMax : null;
  S.flat = flat; S.end = gpuDiag(g); S.medianMs = med; S.seconds = r1((performance.now() - t0) / 1000);
  const verdict = S.error ? `STOPPED: ${S.error}` : flat === null ? "no GPU numbers (CPU backend, or fewer than 11 passes)" : flat ? `FLAT after pass ${WARM}: no buffer made, working memory ${S.workMax} MiB throughout` : `NOT FLAT after pass ${WARM}: ${S.madeAfterWarm} buffers made, working memory ${S.workMin}-${S.workMax} MiB`;
  const summary = `soak ${soakStop ? "stopped" : "done"}: ${S.done} of ${n} passes in ${S.seconds} s, median ${fmtMs(med)} · ${verdict}${S.sameText === false ? " · THE 7 s CHECK CLIP'S TEXT CHANGED" : S.sameText ? " · the 7 s check clip gave the same text every time" : ""}`;
  step(summary, undefined, g ? gpuLine(g) : undefined);
  diag.send("soak", { done: S.done, n, seconds: S.seconds, medianMs: med, flat, madeAfterWarm: S.madeAfterWarm, missesAfterWarm: S.missesAfterWarm, workMin: S.workMin === Infinity ? null : S.workMin, workMax: S.workMax, sameText: S.sameText, error: S.error, start: S.start, end: S.end, stopped: soakStop });
  head.push(summary);
  show(g ? `end: ${gpuLine(g)}` : "");
  soaking = false; busy = false;
  for (const w of readyWaiters.splice(0)) w();
  life.phase("ready", { at: undefined });
  $("soakstop").hidden = true; ($("soakgo") as HTMLButtonElement).disabled = false; ($("model") as HTMLSelectElement).disabled = false;
  if (gen !== loadGen) return;
  setStats();
}

// ---------- page ----------
function init() {
  const sel = $("model") as HTMLSelectElement;
  sel.innerHTML = Object.entries(MODELS).map(([k, v]) => `<option value="${k}"${k === cfg.model ? " selected" : ""}>${esc(v.label)}, ${v.mb} MB</option>`).join("");
  sel.addEventListener("change", () => { lsSet(MODEL_KEY, sel.value); void loadModel(sel.value); });
  ($("bench") as HTMLAnchorElement).href = /live\.html$/.test(location.pathname) ? "./" : "../parakeet-ggml-browser/";
  const proc = $("proc") as HTMLInputElement;
  proc.checked = lsGet(PROC_KEY) === "1";
  proc.addEventListener("change", () => lsSet(PROC_KEY, proc.checked ? "1" : "0"));
  ($("miconly") as HTMLInputElement).checked = params.get("infer") === "0";
  $("rec").addEventListener("click", () => (recording ? stopRecording() : void startRecording()));
  $("soakgo").addEventListener("click", () => void runSoak(Math.max(1, Number(($("soakn") as HTMLInputElement).value) | 0 || 300)));
  $("soakstop").addEventListener("click", () => { soakStop = true; });
  if (params.has("soak")) ($("soakn") as HTMLInputElement).value = String(Math.max(1, num("soak", 300) | 0 || 300));
  pk.runSoak = runSoak;
  $("copy").addEventListener("click", async () => {
    const text = [pk.committed, pk.provisional].filter(Boolean).join(" ");
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch {
      const r = document.createRange(); r.selectNodeContents($("transcript")); const s = getSelection()!; s.removeAllRanges(); s.addRange(r);
      try { ok = document.execCommand("copy"); } catch { /* leave it selected */ }
    }
    $("copy").textContent = ok ? "Copied" : "Selected";
    setTimeout(() => ($("copy").textContent = "Copy"), 1200);
  });
  $("clear").addEventListener("click", () => {
    pk.committed = ""; pk.segments.length = 0; $("committed").textContent = ""; // the transcript only: the engine keeps listening
    render();
  });
  $("forget").addEventListener("click", async () => {
    store.clear(); runs = current ? [current] : [];
    try { await (await navigator.storage.getDirectory()).removeEntry("pk-models", { recursive: true }); } catch { /* nothing stored, or in use */ }
    renderHistory();
  });
  addEventListener("keydown", (e: KeyboardEvent) => { if (e.code === "Space" && !(e.target instanceof HTMLSelectElement) && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); recording ? stopRecording() : void startRecording(); } });
  document.addEventListener("visibilitychange", () => { step(`page ${document.visibilityState}${recording ? " while recording" : ""}`, undefined, ctx ? `AudioContext ${ctx.state}` : undefined); if (document.visibilityState === "visible" && recording) { void ctx?.resume().catch(() => {}); if (!wake || wake.released) void acquireWake(); } });
  // Only errors from this page's own scripts are shown. A wallet, extension or in-app browser injects scripts whose
  // errors also arrive here ("Script error.", "window.ethereum ..."): those are counted and sent to the collector.
  watchErrors(diag, (label, text) => { step(`${label}: ${text}`); showError(`Unexpected error: ${text}`); }, () => { pk.foreign = diag.foreign; setNote(); });
  // For the driver: one offline pass over a whole 16 kHz float32 file, through the same worker.
  pk.offline = async (url: string) => {
    if (busy || recording || modelState !== "ready") throw new Error("busy");
    busy = true;
    try { const pcm = new Float32Array(await (await fetch(url)).arrayBuffer()); const r = await rpc!.call("run", { pcm }); return { text: r.text as string, words: r.words, ms: r.wallMs as number, seconds: pcm.length / SR }; } finally { busy = false; for (const w of readyWaiters.splice(0)) w(); }
  };
  pk.run = async (pcm: Float32Array) => { if (busy || recording || modelState !== "ready") throw new Error("busy"); busy = true; try { return await rpc!.call("run", { pcm }); } finally { busy = false; for (const w of readyWaiters.splice(0)) w(); } };
  pk.vad = () => { const e = engine.stats(); return { noiseFloor: +e.noiseFloor.toFixed(5), threshold: +e.threshold.toFixed(5), hasSpeech: e.open, lastRms: +e.lastRms.toFixed(5), gateMisses: e.gateMisses }; };
  setInterval(() => { if (recording || !engine.idle()) setStats(); }, 250);
  // Every 5 s while something is going on: the numbers that would show a tab growing before it is killed.
  setInterval(() => {
    if (!recording && !busy && modelState !== "loading" && !soaking) return;
    diag.send("heartbeat", { phase: pk.phase, recording, heapMb: pk.heapMb, jsHeapMb: (performance as any).memory ? Math.round((performance as any).memory.usedJSHeapSize / 2 ** 20) : null, passes: pk.passCount, lagS: pk.lagS, bufferS: pk.bufferS,
      gate: gateDiag(), audioContext: ctx?.state ?? null, visibility: document.visibilityState, gpu: gpuDiag(pk.gpu), micOnly: recording && micOnly, capturedS: recording ? r1(captured / SR) : undefined });
  }, 5000);
  setStats();
  setNote();
  sendEnvironment(diag, { device, capS: engine.params.capS, engine: engine.params, limits: cfg.limits || "raised", config: cfg });
  // How the last visit ended. pagehide marks a visit closed; a tab the browser killed never gets there.
  const prev = life.prev, prevRun = runs.at(-1);
  if (prev) {
    const lastStep = prevRun && (!prevRun.sid || prevRun.sid === prev.sid) ? prevRun.steps.at(-1)?.name ?? "none" : "none recorded";
    const line = `previous session ${prev.sid} ended at step "${lastStep.slice(0, 200)}" (phase ${prev.phase}${prev.phase === "soak" ? `: IN A SOAK TEST, last recorded ${prev.at ?? "before its first pass"}` : ""}, ${prev.closed ? "page closed normally" : prev.hidden ? "NOT closed normally, in the background" : "NOT closed normally, in front"}, ${Math.round((Date.now() - prev.wall) / 1000)} s ago)`;
    diag.send("previous-session", { line, prev, lastStep, steps: prevRun?.steps.length ?? 0, error: prevRun?.error ?? null });
    if (!prev.closed) { $("prevnote").hidden = false; $("prevnote").textContent = `This page was not closed normally last time (the browser may have killed or reloaded the tab): ${line}.`; }
    // Died while loading or in its first pass, in front: loading the same thing again would most likely die the same way
    // (and a browser that reloads a killed tab would loop). Stop and let the person choose. guard=0 turns this off.
    if (!prev.closed && !prev.hidden && ["loading", "warmup", "first-pass", "soak"].includes(prev.phase) && params.get("guard") !== "0") return showGuard(prev, lastStep);
  }
  void loadModel(cfg.model);
}
/** The engine's counters for a heartbeat: gate state and numbers, misses, commit reasons. No text. */
function gateDiag() {
  const e = engine.stats();
  return { open: e.open, floor: +e.noiseFloor.toFixed(5), thr: +e.threshold.toFixed(5), rms: +e.lastRms.toFixed(5), opens: e.gateOpens, misses: e.gateMisses, holds: e.modelHolds, probes: e.probes, probesSilent: e.probesSilent,
    utterances: e.utterances, words: e.wordsCommitted, byReason: e.byReason, stale: e.stale, maxBufferS: r1(e.maxBufferS), maxLagS: r1(e.maxLagS), skippedS: r1(e.skippedS), failed: e.failed };
}
function setNote() { $("diagnote").textContent = diag.note(); $("diagshort").textContent = diag.url ? `diagnostics session ${diag.sid}` : "diagnostics off"; }
function showGuard(prev: { sid: string; phase: string; model?: string; cpu?: boolean; at?: string }, lastStep: string) {
  modelState = "failed"; pk.phase = "guard"; guarded = true;
  const what = prev.phase === "soak" ? `in a soak test (last recorded: ${prev.at ?? "before its first pass"})` : prev.phase === "first-pass" ? "during its first transcription pass" : prev.phase === "warmup" ? "during the warm-up pass" : "while loading the model";
  const prevModel = MODELS[prev.model ?? ""]?.label ?? MODELS[cfg.model].label;
  loadStatus("model not loaded", 0);
  diag.send("guard", { prev, lastStep });
  const el = $("errors");
  el.hidden = false;
  el.textContent = `The last time this page was open (session ${prev.sid}) the tab went away ${what} (${prevModel}${prev.cpu ? ", on the CPU" : ""}); its last step was "${lastStep.slice(0, 200)}". The model was not loaded again automatically.`;
  const go = (over: Record<string, string>) => { const p = new URLSearchParams(location.search); for (const [k, v] of Object.entries(over)) p.set(k, v); life.closing(); location.search = `?${p}`; };
  const button = (label: string, act: () => void) => { const b = document.createElement("button"); b.textContent = label; b.onclick = act; b.style.marginRight = "8px"; el.append(b); };
  el.append(document.createElement("br"));
  button("Try again", () => { el.hidden = true; el.textContent = ""; void loadModel(cfg.model); });
  if (!cfg.cpu) button("Try on the CPU (slow)", () => go({ cpu: "1" }));
  if (MODELS.s4 && cfg.model !== "s4") button(`Try the smaller model (${MODELS.s4.label}, ${MODELS.s4.mb} MB)`, () => { lsSet(MODEL_KEY, "s4"); go({ model: "s4" }); });
  setStats();
}
init();
