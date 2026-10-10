// Live page: microphone -> 16 kHz PCM (src/live-worklet.ts) -> the same worker and WASM runtime as the
// benchmark page. The models are offline ones, so "live" is pseudo-streaming: the current utterance
// buffer is re-transcribed as soon as the previous pass is done (never faster than P.gapMs) and shown
// as provisional text; on a pause, or when the buffer reaches the cap, the utterance is transcribed one
// last time, its text is appended to the committed transcript and its audio is dropped. Committed
// text is append-only. Passes never queue: a provisional pass always takes the newest whole buffer.
import { adapterSteps, backendEnv, createRpc, esc, fmtMs, historyStore, r1, stepLine, type Rpc, type RunLog, type Step } from "./common";
declare const MODELS: Record<string, { label: string; file: string; mb: number; family: "0.6b" | "110m" }>; // build.ts; smallest verified first

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => { const v = Number(params.get(k)); return params.has(k) && Number.isFinite(v) ? v : d; };
const SR = 16000, FRAME = 320; // endpointing works on 20 ms frames
// Decisions (all overridable from the query string for tuning: gap, hang, cap, thr, ratio).
const P = {
  gapMs: num("gap", 250), // floor between the starts of two passes
  hangMs: num("hang", 700), // silence after speech that ends an utterance
  preRollMs: 300, // audio kept before the first speech frame
  tailMs: 300, // silence kept after the last speech frame
  capS: num("cap", 24), // an utterance longer than this is cut ...
  cutLookbackS: 10, // ... in the longest gap (or failing that the quietest 200 ms) of its last 10 s
  minSpeechMs: 120, // shorter bursts are dropped without a pass (clicks)
  onsetFrames: 2, // consecutive frames over the threshold that start an utterance
  absRms: num("thr", 0.004), // about -48 dBFS; the threshold is max(this, noise floor * ratio)
  noiseRatio: num("ratio", 3),
  lagWarnS: 2, // "lagging" is shown above this
  backlogS: 120, // audio waiting for its final pass; above this new audio is skipped until half is cleared
};

interface LiveConfig { model: string; variant: "jspi" | "asyncify"; f16: boolean; cpu: boolean; store: string; verbose: boolean; base: string | null; env: Record<string, string> }
const MODEL_KEY = "pkggml:live-model", PROC_KEY = "pkggml:live-proc";
const lsGet = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };
function readConfig(): LiveConfig {
  const names = Object.keys(MODELS);
  const pick = <T extends string>(v: string | null, allowed: T[], d: T): T => (allowed.includes(v as T) ? (v as T) : d);
  const env: Record<string, string> = {};
  for (const kv of params.getAll("env")) { const i = kv.indexOf("="); if (i > 0) env[kv.slice(0, i)] = kv.slice(i + 1); }
  return {
    model: pick(params.get("model") ?? lsGet(MODEL_KEY), names, names[0]),
    variant: pick(params.get("variant"), ["jspi", "asyncify"], "Suspending" in WebAssembly ? "jspi" : "asyncify"),
    f16: params.get("f16") !== "0",
    cpu: params.get("cpu") === "1", // run on the CPU backend even where WebGPU works
    store: pick(params.get("store"), ["opfs", "opfs-blob", "blob", "memfs"], "opfs"),
    verbose: params.get("verbose") === "1",
    base: params.get("base"),
    env,
  };
}
const cfg = readConfig();
const $ = (id: string) => document.getElementById(id)!;

// ---------- persisted step trail ----------
const store = historyStore<LiveConfig>("pkggml:live-runs", 6);
let runs = store.load();
let current: RunLog<LiveConfig> | null = null;
let trailStart = performance.now();
const MAX_STEPS = 400;
function step(name: string, ms?: number, detail?: string) {
  if (!current) return;
  if (current.steps.length >= MAX_STEPS) { if (current.steps.length === MAX_STEPS) current.steps.push({ t: Math.round(performance.now() - trailStart), name: "(further steps not recorded)" }); return; }
  const s: Step = { t: Math.round(performance.now() - trailStart), name, ms: ms === undefined ? undefined : r1(ms), detail };
  current.steps.push(s);
  store.save(runs);
  $("steps").textContent += stepLine(s) + "\n";
}
function newTrail() {
  trailStart = performance.now();
  current = { id: String(Date.now()), started: new Date().toISOString(), config: { ...cfg }, ua: navigator.userAgent, steps: [], done: false };
  runs.push(current);
  $("steps").textContent = "";
  pk.steps = current.steps;
}
function renderHistory() {
  const prev = runs.filter((h) => h !== current).reverse();
  $("history").innerHTML = prev.length
    ? prev.map((h) => `<details><summary class="${h.error ? "bad" : ""}">${esc(h.started)} · ${esc(String(h.config?.model))} · ${h.error ? "failed" : h.done ? "model loaded" : `did not finish loading; last step: ${esc(h.steps.at(-1)?.name ?? "none")}`}</summary><pre>${esc(h.steps.map(stepLine).join("\n"))}${h.error ? "\nERROR: " + esc(h.error) : ""}</pre></details>`).join("")
    : "<p>No earlier sessions stored in this browser.</p>";
}

// ---------- what the driver (and a curious person in the console) can read ----------
interface Segment { startS: number; endS: number; reason: string; ms: number; reused: boolean; text: string }
const pk: any = ((window as any).__pkl = {
  phase: "loading", recording: false, idle: true, committed: "", provisional: "", backend: "", webgpu: false, shaderF16: false, variant: cfg.variant,
  segments: [] as Segment[], passes: [] as number[][], // [ms, audio seconds, 1 = final]
  passCount: 0, finalCount: 0, reusedCount: 0, staleCount: 0, maxLagS: 0, lagS: 0, bufferS: 0, droppedS: 0, inputRate: 0, heapMb: 0, errors: [] as string[], steps: [] as Step[], params: P, load: null,
});

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
let loadGen = 0;
let modelState: "loading" | "ready" | "failed" = "loading";
let busy = false;
let backendLabel = "backend: not loaded yet";

async function modelBase(file: string) {
  if (cfg.base) return cfg.base;
  // Next to the page when built locally; the published live page reuses the benchmark deployment's files.
  for (const base of ["models/", "../parakeet-ggml-browser/models/"]) {
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
  modelState = "failed"; pk.phase = "failed";
  if (current) current.error = msg, store.save(runs);
  step(`ERROR: ${msg.slice(0, 1500)}`);
  loadStatus("model not running", 0);
  showError(msg, true);
  setStats();
}
function gpuErrors(where: string, errs: string[] | undefined) {
  if (!errs?.length) return;
  for (const e of errs.slice(0, 12)) step(`GPU ERROR (${where}): ${e.slice(0, 700)}`);
  // A shader the browser rejected, a validation error or a lost device: what comes out after that is not to be trusted, so stop.
  const lost = errs.some((e) => /device lost/i.test(e));
  showError(`GPU error (${where}): ${errs.slice(0, 4).join("\n").slice(0, 900)}`, true);
  modelState = "failed"; pk.phase = "failed"; loadStatus(lost ? "GPU device lost" : "stopped after a GPU error", 0);
  if (current) current.error = `GPU error (${where})`, store.save(runs);
}
async function loadModel(key: string) {
  const gen = ++loadGen;
  rpc?.terminate();
  modelState = "loading"; pk.phase = "loading"; busy = false;
  cfg.model = key;
  const model = MODELS[key];
  newTrail(); renderHistory();
  warn(null);
  backendLabel = "backend: loading"; setStats();
  loadStatus(`${model.label}: starting`, 0.02);
  step(`start: ${model.label}, ${cfg.variant}${cfg.cpu ? ", cpu=1" : ""}`);
  try {
    const mine = (rpc = createRpc((m) => {
      if (gen !== loadGen) return;
      if (m.type === "progress") return loadStatus(`downloading ${Math.round(m.got / 2 ** 20)} / ${Math.round(m.total / 2 ** 20)} MB`, 0.05 + 0.8 * (m.got / Math.max(1, m.total)));
      if (m.type === "stdout") { if (cfg.verbose) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
      if (m.type === "stderr") { if (cfg.verbose || /error|abort|fail/i.test(m.line)) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
      if (m.type === "stack" && cfg.verbose) step(`stack: ${String(m.stack).slice(0, 2500)}`);
    }));
    const init = await mine.call("init", { base: location.href, variant: cfg.variant, wantF16: cfg.f16, limits: cfg.env.GGML_WEBGPU_LIMITS ?? "" });
    if (gen !== loadGen) return;
    const a = init.adapter;
    step("WASM module ready", init.ms, `heap ${init.heapMb} MB; JSPI ${init.jspi ? "available" : "absent"}`);
    adapterSteps(a, step);
    $("env").textContent = `${a.available ? `${a.vendor} ${a.architecture} ${a.description || ""} · shader-f16 ${a.shaderF16 ? "present" : "absent"}` : `no WebGPU adapter (${a.reason})`} · JSPI ${init.jspi ? "available" : "absent (ASYNCIFY build)"} · ${cfg.variant} build`;
    const { useF16, pathLine, env } = backendEnv(a, { f16: cfg.f16, flash: false, extra: { ...(cfg.cpu ? { TRANSCRIBE_BACKENDS: "cpu" } : {}), ...cfg.env } });
    if (a.available && !cfg.cpu) step(pathLine);
    loadStatus(`${model.label}: fetching the model file (${model.mb} MB, once)`, 0.05);
    const base = await modelBase(model.file);
    let where = cfg.store;
    if (where === "opfs") { // private windows and some embedded browsers have no origin-private file system: keep the file as a Blob for this visit
      try { await navigator.storage.getDirectory(); } catch (e: any) { where = "blob"; step(`no origin-private file system (${e?.name ?? e}): the model file is fetched again on every visit`); }
    }
    const ld = await mine.call("load", { url: new URL(base + model.file, location.href).href, name: model.file, store: where, env, threads: 1, verbose: cfg.verbose });
    if (gen !== loadGen) return;
    step(`model file ready (${ld.from})`, ld.fetchMs, `${ld.mb.toFixed(0)} MB from ${base}`);
    step(`model loaded on ${ld.backend}`, ld.loadMs, `WASM heap ${ld.heapMb} MB, ${ld.heapUsedMb} MB in use`);
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
    gpuErrors("load", ld.gpuErrors);
    loadStatus(`${model.label}: warming up`, 0.92);
    // One pass over 1.5 s of faint noise, so shader pipelines are compiled before the first words arrive.
    const noise = new Float32Array(SR * 1.5);
    let seed = 1;
    for (let i = 0; i < noise.length; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; noise[i] = (seed / 2 ** 32 - 0.5) * 2e-3; }
    const w = await mine.call("run", { pcm: noise });
    if (gen !== loadGen) return;
    step("warm-up pass (1.5 s of noise)", w.wallMs);
    gpuErrors("warm-up", w.gpuErrors);
    if (pk.phase === "failed") return;
    current!.done = true; store.save(runs);
    modelState = "ready"; pk.phase = "ready";
    loadStatus(`${model.label} ready`, 1, true);
    setStats();
    pump();
  } catch (e: any) {
    if (gen !== loadGen) return;
    modelFailed(`${e?.name ?? "Error"}: ${e?.message ?? e}`);
  }
}

// ---------- utterance buffer and endpointing ----------
const MAX_FRAMES = Math.ceil(((P.capS + 2) * SR) / FRAME);
let buf = new Float32Array(MAX_FRAMES * FRAME); // the current utterance, from absolute sample bufAbs
let rms = new Float32Array(MAX_FRAMES), flag = new Uint8Array(MAX_FRAMES);
let nFrames = 0, bufAbs = 0, captured = 0; // captured: 16 kHz samples received since Record
let hasSpeech = false, lastSpeech = -1, onsetRun = 0, noiseFloor = 0.001;
let epoch = 0; // bumped whenever the buffer's start moves past speech, so a provisional result for the old buffer is dropped
let provText = "", provTextEnd = 0, provEndAbs = 0, coveredAbs = 0, lastPassStart = 0; // provTextEnd: how far the pass behind provText reached
// A cut utterance waiting for its text. `text` is already there when no pass is needed; `wait` holds a skip marker until its length is known.
interface Final { pcm: Float32Array | null; startAbs: number; endAbs: number; reason: string; placeholder: string; text?: string; wait?: boolean }
const finals: Final[] = [];
let dropping = false, droppedFrom = 0;
const backlogS = () => finals.reduce((s, f) => s + (f.endAbs - f.startAbs), 0) / SR;

function shiftFrames(n: number) {
  buf.copyWithin(0, n * FRAME, nFrames * FRAME); rms.copyWithin(0, n, nFrames); flag.copyWithin(0, n, nFrames);
  nFrames -= n; bufAbs += n * FRAME; lastSpeech -= n;
}
function resetUtterance() { hasSpeech = false; lastSpeech = -1; onsetRun = 0; provText = ""; provTextEnd = 0; epoch++; }
/** Hand the first `n` frames over for a final pass (or drop them when they hold no real speech). */
function cut(n: number, reason: string) {
  let speech = 0;
  for (let i = 0; i < n; i++) speech += flag[i];
  if (speech * 20 >= P.minSpeechMs) {
    // After a pause the newest provisional pass has usually seen this whole utterance already (and a little more
    // silence): then its text, the one on screen, is the final text and no further pass is run.
    const seen = !reason.startsWith("cap") && provText !== "" && provTextEnd >= bufAbs + n * FRAME;
    finals.push({ pcm: seen ? null : buf.slice(0, n * FRAME), startAbs: bufAbs, endAbs: bufAbs + n * FRAME, reason, placeholder: provText, text: seen ? provText : undefined });
  }
  shiftFrames(n);
  provText = ""; provTextEnd = 0; epoch++;
  let left = 0; lastSpeech = -1;
  for (let i = 0; i < nFrames; i++) if (flag[i]) left++, lastSpeech = i;
  hasSpeech = reason.startsWith("cap") && left > 0; // after a pause or Stop what is left is silence by construction
  if (!hasSpeech) { lastSpeech = -1; onsetRun = 0; }
  render();
}
function capCut() {
  // Where to cut an utterance that has no pause long enough to end it: the middle of the longest run of
  // non-speech frames in the last cutLookbackS (a breath between sentences), not closer than 0.5 s to the end ...
  const lo = Math.max(10, nFrames - Math.round(P.cutLookbackS * 50)), hi = nFrames - 25;
  let runStart = -1, bestStart = -1, bestLen = 0;
  for (let i = lo; i <= hi; i++) {
    if (i < hi && !flag[i]) { if (runStart < 0) runStart = i; continue; }
    if (runStart >= 0 && i - runStart >= bestLen) bestLen = i - runStart, bestStart = runStart; // ties: the later one
    runStart = -1;
  }
  if (bestLen >= 8) return cut(bestStart + (bestLen >> 1), "cap");
  // ... and without one of at least 160 ms, the quietest 200 ms (10 frames) there.
  let best = lo, bestE = Infinity, e = 0;
  for (let i = lo; i < lo + 10; i++) e += rms[i] * rms[i];
  for (let i = lo; ; i++) {
    if (e < bestE) bestE = e, best = i;
    if (i + 10 >= hi) break;
    e += rms[i + 10] * rms[i + 10] - rms[i] * rms[i];
  }
  // Even the quietest stretch of a whole cap of audio counts as speech: the room got louder. Take that as the floor.
  const q = Math.sqrt(bestE / 10);
  if (q > Math.max(P.absRms, noiseFloor * P.noiseRatio)) noiseFloor = q / 2;
  cut(best + 5, "cap-quiet");
}
function endDrop(at: number) {
  dropping = false; bufAbs = at; nFrames = 0; resetUtterance();
  const m = finals.find((f) => f.wait);
  if (m) m.wait = false, m.text = `[${Math.round((at - droppedFrom) / SR)} s of audio skipped: transcription is slower than real time here]`;
  pk.droppedS += (at - droppedFrom) / SR;
}
function onPcm(pcm: Float32Array) {
  captured += pcm.length;
  if (dropping) {
    if (backlogS() > P.backlogS / 2) return;
    endDrop(captured - pcm.length);
  }
  buf.set(pcm, nFrames * FRAME);
  for (let o = 0; o + FRAME <= pcm.length; o += FRAME) {
    let s = 0;
    for (let i = o; i < o + FRAME; i++) s += pcm[i] * pcm[i];
    const r = Math.sqrt(s / FRAME), sp = r > Math.max(P.absRms, noiseFloor * P.noiseRatio);
    if (!sp || r < noiseFloor) noiseFloor += (r - noiseFloor) * (r < noiseFloor ? 0.3 : 0.05); // tracks non-speech frames only: falls fast, rises slowly
    rms[nFrames] = r; flag[nFrames] = sp ? 1 : 0;
    onsetRun = sp ? onsetRun + 1 : 0;
    if (sp && (hasSpeech || onsetRun >= P.onsetFrames)) hasSpeech = true, lastSpeech = nFrames;
    nFrames++;
  }
  speechNow = flag[nFrames - 1] === 1;
  if (!hasSpeech) {
    const keep = Math.round(P.preRollMs / 20);
    if (nFrames > keep) shiftFrames(nFrames - keep); // silence is not kept and not transcribed
  } else if ((nFrames - 1 - lastSpeech) * 20 >= P.hangMs) {
    cut(Math.min(nFrames, lastSpeech + 1 + Math.round(P.tailMs / 20)), "pause");
    const keep = Math.round(P.preRollMs / 20);
    if (nFrames > keep) shiftFrames(nFrames - keep);
  } else if (nFrames * FRAME >= P.capS * SR) capCut();
  if (!dropping && backlogS() + (nFrames * FRAME) / SR > P.backlogS) { // far slower than real time: stop taking audio rather than grow without bound
    if (hasSpeech) cut(nFrames, "overflow");
    finals.push({ pcm: null, startAbs: captured, endAbs: captured, reason: "skip", placeholder: "", text: "", wait: true });
    dropping = true; droppedFrom = captured;
    step(`audio is being skipped: ${Math.round(backlogS())} s waits for transcription`);
  }
  pump();
}

// ---------- passes ----------
function commit(text: string, seg: Final, ms: number) {
  const t = text.trim();
  if (t) {
    const add = (pk.committed ? " " : "") + t;
    pk.committed += add;
    $("committed").append(add); // append-only: earlier text nodes are never touched
  }
  if (pk.segments.length < 1000) pk.segments.push({ startS: r1(seg.startAbs / SR), endS: r1(seg.endAbs / SR), reason: seg.reason, ms: r1(ms), reused: seg.text !== undefined, text: t });
  if (seg.reason !== "skip") {
    pk.finalCount++; if (seg.text !== undefined) pk.reusedCount++;
    if (pk.finalCount <= 150) step(`commit ${pk.finalCount}: ${((seg.endAbs - seg.startAbs) / SR).toFixed(1)} s of audio, ${seg.reason}`, seg.text !== undefined ? undefined : ms, `${t.split(/\s+/).filter(Boolean).length} words${seg.text !== undefined ? "; text of the last provisional pass" : ""}`);
  }
}
function notePass(ms: number, audioS: number, final: boolean, r: any) {
  pk.passCount++;
  if (pk.passes.length >= 4000) pk.passes.splice(0, 2000);
  pk.passes.push([r1(ms), r1(audioS), final ? 1 : 0, r1(r.mel_ms), r1(r.decode_ms)]);
  pk.heapMb = r.heapMb;
  last = { ms, audioS };
  gpuErrors("pass", r.gpuErrors);
}
let last: { ms: number; audioS: number } | null = null;
async function pump() {
  if (busy || modelState !== "ready" || !rpc) return;
  const seg = finals[0];
  if (seg?.text !== undefined) { // text already known: an utterance its last provisional pass covered, or a skip marker
    if (seg.wait) return;
    finals.shift(); commit(seg.text, seg, 0); coveredAbs = Math.max(coveredAbs, seg.endAbs); render(); return pump();
  }
  let pcm: Float32Array, jobEpoch = epoch, endAbs: number;
  if (seg) pcm = seg.pcm!, endAbs = seg.endAbs;
  else {
    endAbs = bufAbs + nFrames * FRAME;
    if (!hasSpeech || endAbs - provEndAbs < SR / 10) return; // nothing new to look at
    if (performance.now() - lastPassStart < P.gapMs) return; // the next audio block calls again
    pcm = buf.slice(0, nFrames * FRAME);
    provEndAbs = endAbs;
  }
  busy = true; lastPassStart = performance.now();
  const gen = loadGen, t0 = performance.now();
  let r: any;
  try { r = await rpc.call("run", { pcm }); } // a copy is sent: a final segment stays queued until its text is in
  catch (e: any) {
    if (gen !== loadGen) return;
    busy = false;
    return modelFailed(`transcription failed: ${e?.message ?? e}`);
  }
  if (gen !== loadGen) return;
  busy = false;
  const ms = performance.now() - t0;
  notePass(ms, pcm.length / SR, !!seg, r);
  if (seg) {
    if (finals[0] === seg) { finals.shift(); commit(r.text, seg, ms); coveredAbs = Math.max(coveredAbs, endAbs); }
  } else if (jobEpoch === epoch) { provText = String(r.text).trim(); provTextEnd = endAbs; coveredAbs = Math.max(coveredAbs, endAbs); }
  else pk.staleCount++; // the utterance was cut while this pass ran; its final pass supplies the text
  render();
  pump();
}

// ---------- rendering ----------
let speechNow = false, levelPeak = 0, lagS = 0;
function render() {
  const pending = [...finals.map((f) => f.placeholder), provText].filter(Boolean).join(" ");
  const box = $("transcript");
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  pk.provisional = pending;
  $("prov").textContent = pending ? (pk.committed ? " " : "") + pending : "";
  $("hint").hidden = !!(pk.committed || pending || recording);
  if (atEnd) box.scrollTop = box.scrollHeight;
  pk.idle = !busy && finals.length === 0 && !recording;
  setStats();
}
function setStats() {
  // Lag: audio of pending utterances that no displayed text has seen yet (silence after an utterance does not count).
  const oldest = finals.length ? finals[0].startAbs : hasSpeech ? bufAbs : null;
  const newest = hasSpeech ? bufAbs + nFrames * FRAME : finals.length ? finals[finals.length - 1].endAbs : 0;
  lagS = oldest === null ? 0 : Math.max(0, (newest - Math.max(coveredAbs, oldest)) / SR);
  pk.lagS = r1(lagS); pk.bufferS = r1((nFrames * FRAME) / SR);
  if (recording && lagS > pk.maxLagS) pk.maxLagS = r1(lagS);
  $("s-pass").textContent = last ? fmtMs(last.ms) : "-";
  $("s-buf").textContent = recording || nFrames ? `${((hasSpeech ? nFrames * FRAME : 0) / SR).toFixed(1)} s${finals.length ? ` +${finals.length}` : ""}` : "-";
  $("s-rtf").textContent = last ? `${last.audioS * 1000 / last.ms >= 10 ? Math.round((last.audioS * 1000) / last.ms) : ((last.audioS * 1000) / last.ms).toFixed(1)}x` : "-";
  const lagging = lagS > P.lagWarnS || dropping;
  $("s-lag").textContent = recording || finals.length ? `${lagS.toFixed(1)} s${dropping ? ", skipping audio" : lagging ? ", lagging" : ""}` : "-";
  $("lagcell").classList.toggle("lagging", lagging);
  $("s-backend").textContent = backendLabel + (pk.inputRate ? ` · mic ${pk.inputRate} Hz` : "");
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
    c.onstatechange = () => { if (ctx === c && recording && c.state !== "running") micMessage(`Audio capture is ${c.state} (paused by the browser); it resumes when this tab is in front`); else if (ctx === c && recording) micMessage(null); };
    const set = s.getAudioTracks()[0]?.getSettings?.() ?? {};
    // fresh utterance state; the committed transcript is kept
    nFrames = 0; bufAbs = 0; captured = 0; coveredAbs = 0; provEndAbs = 0; noiseFloor = 0.001; dropping = false; resetUtterance();
    recording = true; starting = false; recStart = performance.now(); sawSignal = false;
    pk.recording = true; pk.inputRate = c.sampleRate;
    $("rec").classList.add("on"); $("rec").setAttribute("aria-pressed", "true"); $("rec").setAttribute("aria-label", "Stop");
    ($("model") as HTMLSelectElement).disabled = true;
    micMessage(null);
    step(`recording: context ${c.sampleRate} Hz -> 16000 Hz, track ${set.sampleRate ?? "?"} Hz, ${set.channelCount ?? "?"} ch, echoCancellation=${set.echoCancellation} noiseSuppression=${set.noiseSuppression} autoGainControl=${set.autoGainControl}`);
    void acquireWake();
    render();
  } catch (e: any) {
    starting = false;
    releaseMic();
    micMessage(micErrorText(e));
    step(`microphone: ${e?.name ?? "Error"}: ${e?.message ?? e}`);
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
  if (dropping) endDrop(captured);
  if (hasSpeech) cut(nFrames, "stop"); // what was said since the last pause still gets its final pass
  nFrames = 0; resetUtterance(); speechNow = false; levelPeak = 0; setLevel(0);
  $("rec").classList.remove("on"); $("rec").setAttribute("aria-pressed", "false"); $("rec").setAttribute("aria-label", "Record");
  ($("model") as HTMLSelectElement).disabled = false;
  micMessage(message);
  step(`stopped after ${(captured / SR).toFixed(1)} s: ${pk.passCount} passes, ${pk.finalCount} commits, max lag ${pk.maxLagS} s`);
  render();
  pump();
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
  $("rec").addEventListener("click", () => (recording ? stopRecording() : void startRecording()));
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
    finals.length = 0; nFrames = 0; bufAbs = captured; resetUtterance();
    pk.committed = ""; pk.segments.length = 0; $("committed").textContent = "";
    render();
  });
  $("forget").addEventListener("click", async () => {
    store.clear(); runs = current ? [current] : [];
    try { await (await navigator.storage.getDirectory()).removeEntry("pk-models", { recursive: true }); } catch { /* nothing stored, or in use */ }
    renderHistory();
  });
  addEventListener("keydown", (e: KeyboardEvent) => { if (e.code === "Space" && !(e.target instanceof HTMLSelectElement) && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); recording ? stopRecording() : void startRecording(); } });
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && recording) { void ctx?.resume().catch(() => {}); if (!wake || wake.released) void acquireWake(); } });
  addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => { step(`unhandled rejection: ${e.reason?.message ?? e.reason}`); showError(`Unexpected error: ${e.reason?.message ?? e.reason}`); });
  addEventListener("error", (e: ErrorEvent) => { step(`uncaught error: ${e.message}`); showError(`Unexpected error: ${e.message}`); });
  // For the driver: one offline pass over a whole 16 kHz float32 file, through the same worker.
  pk.offline = async (url: string) => {
    if (busy || recording || modelState !== "ready") throw new Error("busy");
    busy = true;
    try { const pcm = new Float32Array(await (await fetch(url)).arrayBuffer()); const r = await rpc!.call("run", { pcm }); return { text: r.text as string, ms: r.wallMs as number, seconds: pcm.length / SR }; } finally { busy = false; }
  };
  pk.run = async (pcm: Float32Array) => { if (busy || recording || modelState !== "ready") throw new Error("busy"); busy = true; try { return await rpc!.call("run", { pcm }); } finally { busy = false; } };
  pk.vad = () => ({ noiseFloor: +noiseFloor.toFixed(5), hasSpeech, lastRms: +(nFrames ? rms[nFrames - 1] : 0).toFixed(5) });
  setInterval(() => { if (recording || finals.length) setStats(); }, 250);
  setStats();
  void loadModel(cfg.model);
}
init();
