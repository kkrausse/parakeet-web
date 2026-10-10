// Parakeet (TDT 0.6b v2 and tdt_ctc-110m) in the browser on transcribe.cpp + ggml's WebGPU backend (WASM). One page,
// no framework. Everything heavy happens in src/worker.ts; this file is configuration, the step
// trail (screen + localStorage, so a killed tab leaves one) and the benchmark loop.
import { adapterSteps, backendEnv, createRpc, esc, fmtMs, gpuDiag, gpuLine, historyStore, r1, stepLine, wordDiff, type Rpc, type RunLog as RunLogOf, type Step } from "./common";
import { createDiag, detectDevice, lifeStore, sendEnvironment, watchErrors } from "./diag";
// Reference text per model family: 0.6b = native transcribe.cpp F16; 110m = onnx-asr / ONNX Runtime fp32
// (../parakeet-webgpu-bench/results/110m/reference-ort-cpu-fp32.jsonl).
const EXPECTED_06B: Record<string, string> = {
  a07: "going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight.",
  a14: "going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight. He'll have to put in an appearance at some place of worship on Sunday morning, and he can come to us immediately afterwards.",
  a56: "Welcome to QuirkQuidQuill Inc., where finance meets innovation. Explore diverse offerings from the P3 Quattro, a unique investment portfolio quadrant, to the O3 Omni, a platform for intricate derivative trading strategies. Delve into unconventional bond markets with our B3 Bond X and experience non-standard equity trading with E3 Equity. Personalize your wealth management with W3 RAPZ and anticipate market trends with the O2 Outlier, our forward-thinking financial forecasting tool. Explore venture capital world with U3Unifund or move your money with the M3 Mover, our sophisticated monetary transfer module. At QuirkQuidQuill Inc., we turn complex finance into creative solutions. Join us in redefining financial services.",
};
const EXPECTED_110M: Record<string, string> = {
  "a07": "going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight.",
  "a14": "going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after day for a fortnight. He'll have to put in an appearance at some place of worship on Sunday morning, and he can come to us immediately afterwards",
  "a56": "Welcome to Quirk Quid Quill Inc. where finance meets innovation, explore diverse offerings from the P three Quatro a unique investment portfolio quadrant to the O three Omni, a platform for intricate derivative trading strategies. Delve into unconventional bond markets with our B three Bond X and our and experience non standard equity trading with E three equity, personalize your wealth management with W three Wrap Z and anticipate market trends with the O two outlier, our forward thinking financial forecasting tool. Explore venture capital world with U three unifund or move your money with the M three mover, our sophisticated monetary transfer module. At Cork Quid Quill Inc. we turn complex finance into creative solutions. Join us in redefining financial services."
};
const CLIPS: Record<string, string> = { a07: "7.0 s", a14: "13.7 s", a56: "56.1 s", l2: "2.1 min", l5: "4.4 min", l10: "8.9 min" };
const SHORT_CLIPS = ["a07", "a14", "a56"]; // clip=all; the long ones are the fixtures concatenated (chunked encoding)
// Long clips have no independent reference: the expected text is what the native build gives with the same
// chunking (30 s windows + 4 s halo), per model. Filled in by build.ts from out/lt/<model>-<clip>-chunk.txt.
declare const EXPECTED_LONG: Record<string, Record<string, string>>;
declare const MODELS: Record<string, { label: string; file: string; mb: number; family: "0.6b" | "110m" }>; // filled in by build.ts from what is on disk

interface Config {
  model: string; clip: string; runs: number; threads: number;
  store: "opfs" | "opfs-blob" | "opfs-writable" | "blob" | "memfs"; // where the model file lives while it is loaded
  f16: boolean; // false (f16=0): do not use shader-f16 even where the adapter has it (the f32-only shader path)
  flash: boolean; // ggml FLASH_ATTN_EXT in the encoder (off: plain matmul + softmax attention)
  variant: "jspi" | "asyncify" | "prof"; // prof: JSPI build with ggml CPU-side profiling (use with verbose=1)
  verbose: boolean; base: string; env: Record<string, string>;
  limits: string; // "default": request no WebGPU limit above the spec defaults (the default on iOS); "": the backend raises the two byte limits
}
const params = new URLSearchParams(location.search);
const device = detectDevice(params); // phone=1|0 forces it
const diag = createDiag("bench", params);
function readConfig(): Config {
  const pick = <T extends string>(k: string, allowed: T[], d: T): T => (allowed.includes(params.get(k) as T) ? (params.get(k) as T) : d);
  const env: Record<string, string> = {};
  for (const kv of params.getAll("env")) { const i = kv.indexOf("="); if (i > 0) env[kv.slice(0, i)] = kv.slice(i + 1); }
  const names = Object.keys(MODELS);
  return {
    model: pick("model", names, names[0]), // build.ts lists the smallest verified configuration first
    clip: pick("clip", [...Object.keys(CLIPS), "all"], "a07"),
    runs: Math.max(0, Number(params.get("runs") ?? 10) | 0),
    threads: Math.max(1, Number(params.get("threads") ?? 1) | 0),
    store: pick("store", ["opfs", "opfs-blob", "opfs-writable", "blob", "memfs"], "opfs"),
    f16: params.get("f16") !== "0",
    flash: params.get("flash") === "1",
    variant: pick("variant", ["jspi", "asyncify", "prof"], "Suspending" in WebAssembly ? "jspi" : "asyncify"),
    verbose: params.get("verbose") === "1",
    base: params.get("base") ?? "models/",
    env,
    limits: env.GGML_WEBGPU_LIMITS ?? (params.get("limits") === "raised" ? "" : params.get("limits") === "default" || device.ios ? "default" : ""),
  };
}
const cfg = readConfig();

// ---------- persistent step log ----------
type RunLog = RunLogOf<Config>;
const HISTORY_KEY = "pkggml:runs";
const store = historyStore<Config>(HISTORY_KEY);
const life = lifeStore("pkggml:bench-life", diag);
let history = store.load();
let current: RunLog | null = null;
function persist() { store.save(history); }
const $ = (id: string) => document.getElementById(id)!;
let runStart = 0;
/** Record a completed step: on screen and in localStorage, before anything else happens. */
function step(name: string, ms?: number, detail?: string, local = false) { // local: the detail is not sent to the collector (transcript text)
  diag.send("step", { name, ms: ms === undefined ? undefined : r1(ms), detail: local ? undefined : detail });
  const s: Step = { t: Math.round(performance.now() - runStart), name, ms: ms === undefined ? undefined : r1(ms), detail };
  current?.steps.push(s);
  persist();
  $("steps").textContent += stepLine(s) + "\n";
  (window as any).__pkb.steps = current?.steps;
}
function status(text: string) { $("status").textContent = text; }
function stats(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  return { median: r1(s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2), worst: r1(s[s.length - 1]), best: r1(s[0]) };
}

// ---------- worker RPC ----------
let rpc: Rpc | null = null;
const call = <T = any>(type: string, body: Record<string, unknown> = {}, transfer: Transferable[] = []) => rpc!.call<T>(type, body, transfer);
function startWorker() {
  rpc = createRpc((m) => {
    if (m.type === "progress") return status(`downloading model: ${Math.round(m.got / 2 ** 20)} / ${Math.round(m.total / 2 ** 20)} MB`);
    if (m.type === "stage") return step(m.name, undefined, m.detail);
    if (m.type === "diag") return diag.send(m.kind, m.data);
    if (m.type === "stdout") { if (cfg.verbose) step("wasm stdout", undefined, String(m.line).slice(0, 400), true); return; }
    if (m.type === "stderr") { if (cfg.verbose || /error|abort|fail/i.test(m.line)) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
    if (m.type === "stack" && cfg.verbose) step(`stack: ${String(m.stack).slice(0, 2500)}`);
  });
}

// ---------- the run ----------
async function run() {
  const w = window as any;
  runStart = performance.now();
  life.phase("init", { model: cfg.model });
  current = { id: String(Date.now()), sid: diag.sid, started: new Date().toISOString(), config: cfg, ua: navigator.userAgent, steps: [], done: false };
  history.push(current);
  $("steps").textContent = "";
  const model = MODELS[cfg.model];
  const EXPECTED: Record<string, string> = { ...(model.family === "110m" ? EXPECTED_110M : EXPECTED_06B), ...(EXPECTED_LONG[cfg.model] ?? {}) };
  const result: any = { config: cfg, model, ua: navigator.userAgent, crossOriginIsolated, clips: [] };
  w.__pkb.result = result;
  step(`start: ${model.label}, flash=${cfg.flash ? "on" : "off"}, store=${cfg.store}, ${cfg.variant}`, undefined, `session ${diag.sid}; ${device.why}; limits ${cfg.limits || "raised"}; clip ${cfg.clip}`);

  w.__pkb.phase = "init";
  startWorker();
  // gpuwatch=0: no JS-side counting of WebGPU objects; gc=<MiB>: nudge the JS collector after each pass (src/worker.ts)
  const init = await call("init", { base: location.href, variant: cfg.variant, wantF16: cfg.f16, limits: cfg.limits, gpuwatch: params.get("gpuwatch") !== "0", gcMb: Number(params.get("gc") ?? 0) || 0 });
  let gpuNow: any = null, gpuPrev: any = null;
  result.adapter = init.adapter; result.jspi = init.jspi;
  const a = init.adapter;
  diag.send("adapter", { adapter: a, jspi: init.jspi, heapMb: init.heapMb });
  step("WASM module ready", init.ms, `heap ${init.heapMb} MB; JSPI ${init.jspi ? "available" : "absent"}`);
  $("env").textContent = envLine(a);
  adapterSteps(a, step);

  w.__pkb.phase = "sessions";
  status("fetching and loading the model");
  const { useF16, flash, pathLine, env } = backendEnv(a, { f16: cfg.f16, flash: cfg.flash, extra: cfg.env });
  result.shaderF16Used = useF16;
  if (cfg.flash && !flash) step("flash attention needs shader-f16: using matmul + softmax attention instead");
  step(pathLine);
  if (a.available) step(a.plan?.limits === "default" ? `WebGPU limits: spec defaults, nothing raised${cfg.limits === "default" ? "" : " (the raised request was refused)"}` : `WebGPU limits: the two byte limits raised to ${JSON.stringify(a.requested?.requiredLimits)} (limits=default requests none)`);
  life.phase("loading");
  const ld = await call("load", { url: new URL(cfg.base + model.file, location.href).href, name: model.file, store: cfg.store, env, threads: cfg.threads, verbose: cfg.verbose });
  result.load = { fetchMs: Math.round(ld.fetchMs), loadMs: Math.round(ld.loadMs), from: ld.from, fileMb: r1(ld.mb), wasmHeapMb: ld.heapMb, wasmHeapUsedMb: ld.heapUsedMb, backend: ld.backend };
  result.session = { totalMs: Math.round(ld.loadMs) };
  step(`model file ready (${ld.from})`, ld.fetchMs, `${ld.mb.toFixed(0)} MB; storage path ${ld.storagePath}`);
  diag.send("chosen", { model: cfg.model, file: model.file, build: cfg.variant, backend: ld.backend, webgpu: /webgpu/i.test(ld.backend), shaderPath: useF16 ? "f16" : "f32-only", limits: a.plan?.limits ?? null,
    storagePath: ld.storagePath, from: ld.from, fileBytes: Math.round(ld.mb * 2 ** 20), fetchMs: Math.round(ld.fetchMs), loadMs: Math.round(ld.loadMs), heapMb: ld.heapMb, heapUsedMb: ld.heapUsedMb, phone: device.phone, ios: device.ios });
  step(`model loaded on ${ld.backend}`, ld.loadMs, `WASM heap ${ld.heapMb} MB, ${ld.heapUsedMb} MB in use`);
  if (ld.gpu) { step(`after load: ${gpuLine(ld.gpu)}`); result.gpuAtLoad = ld.gpu; }
  if (!/webgpu/i.test(ld.backend)) step(`WARNING: not on WebGPU (backend "${ld.backend}"). Everything runs on one WASM thread and the weights sit in the WASM heap.`);
  if (ld.log) step(`library log: ${ld.log.trim().slice(0, 600)}`);
  const gpuFail = (where: string, errs: string[] | undefined) => {
    if (!errs?.length) return;
    for (const e of errs.slice(0, 12)) step(`GPU ERROR (${where}): ${e.slice(0, 700)}`);
    result.gpuErrors = [...(result.gpuErrors ?? []), ...errs];
  };
  gpuFail("load", ld.gpuErrors);
  result.startToLoadedMs = Math.round(performance.now() - runStart);
  await new Promise((r) => setTimeout(r, 700)); // let the driver sample memory in a settled state
  w.__pkb.phase = "loaded";
  await new Promise((r) => setTimeout(r, 700));

  const clips = cfg.clip === "all" ? SHORT_CLIPS : [cfg.clip];
  for (const [i, clip] of clips.entries()) {
    w.__pkb.phase = `run:${clip}`;
    const audio = new Float32Array(await (await fetch(`audio/${clip}.f32`)).arrayBuffer());
    const seconds = audio.length / 16000;
    let nRun = 0;
    const once = async () => {
      const r = await call("run", { pcm: audio });
      diag.send("pass", { clip, n: ++nRun, ms: r1(r.wallMs), audioS: r1(seconds), melMs: r1(r.mel_ms), decodeMs: r1(r.decode_ms), heapMb: r.heapMb, heapUsedMb: r.heapUsedMb, gpu: gpuDiag(r.gpu) }, false);
      gpuPrev = gpuNow; gpuNow = r.gpu ?? null;
      gpuFail(clip, r.gpuErrors);
      if (cfg.verbose && r.log) for (const l of String(r.log).split("\n")) if (/decoder:|mel:/.test(l)) step(`library: ${l.trim().slice(0, 300)}`);
      // encoder = until its output is on the CPU: with lazy synchronize the library's own encode_ms stops at submit
      return { pre: r.mel_ms as number, enc: (r.wallMs - r.mel_ms - r.decode_ms) as number, encSubmit: r.encode_ms as number, dec: r.decode_ms as number, total: r.wallMs as number, text: r.text as string, tokens: r.n_tokens as number, heapMb: r.heapMb as number, heapUsedMb: r.heapUsedMb as number };
    };
    status(`${clip}: first run`);
    life.phase("first-pass");
    step(`${clip} first run starting`, undefined, `${seconds.toFixed(1)} s of audio`);
    const first = await once();
    life.phase("running");
    step(`${clip} (${CLIPS[clip]}) first run${i === 0 ? " (cold)" : ""}`, first.total, `mel ${fmtMs(first.pre)}, encoder ${fmtMs(first.enc)}, decode ${fmtMs(first.dec)}, heap ${first.heapMb} MB (${first.heapUsedMb} in use)`);
    const rec: any = {
      clip, seconds: Math.round(seconds * 100) / 100, firstInPage: i === 0,
      firstRunMs: { pre: r1(first.pre), enc: r1(first.enc), dec: r1(first.dec), total: r1(first.total) },
      startToFirstTranscriptMs: i === 0 ? Math.round(performance.now() - runStart) : null,
      decSteps: first.tokens, text: first.text, matchesNative: first.text === EXPECTED[clip], hasReference: clip in EXPECTED,
    };
    const long = !SHORT_CLIPS.includes(clip);
    if (long && rec.hasReference) rec.wordDiff = wordDiff(EXPECTED[clip], first.text);
    step(long ? (rec.hasReference ? `${clip} transcript (chunked encoding): ${rec.wordDiff.differing} of ${rec.wordDiff.words} words differ from the native build's chunked transcript` : `${clip} transcript: no stored reference for this model`)
      : `${clip} transcript ${rec.matchesNative ? "matches" : "DIFFERS from"} the reference`, undefined, first.text.length > 90 ? first.text.slice(0, 90) + "..." : first.text, true);
    result.clips.push(rec);
    if (cfg.runs > 0) {
      status(`${clip}: warm-up`);
      await once(); // discarded
      const all: Awaited<ReturnType<typeof once>>[] = [];
      for (let k = 0; k < cfg.runs; k++) { status(`${clip}: warm run ${k + 1} / ${cfg.runs}`); all.push(await once()); }
      const col = (f: (x: (typeof all)[0]) => number) => stats(all.map(f));
      Object.assign(rec, { warmRuns: all.length, preMs: col((x) => x.pre), encMs: col((x) => x.enc), decMs: col((x) => x.dec), totalMs: col((x) => x.total) });
      rec.xRealTime = r1((seconds * 1000) / rec.totalMs.median);
      rec.textStable = all.every((x) => x.text === first.text);
      rec.wasmHeapMb = all.at(-1)!.heapMb; rec.wasmHeapUsedMb = all.at(-1)!.heapUsedMb;
      step(`${clip} warm x${all.length}`, rec.totalMs.median, `total median/worst ${rec.totalMs.median} / ${rec.totalMs.worst} ms; encoder ${rec.encMs.median} / ${rec.encMs.worst} ms; mel ${rec.preMs.median} ms; decode ${rec.decMs.median} ms; ${rec.xRealTime}x real time; text stable=${rec.textStable}`);
    }
    if (gpuNow) { rec.gpu = gpuNow; step(`${clip}: ${gpuLine(gpuNow, gpuPrev)}`); }
    render(result);
  }
  if (params.get("trim") === "1") { const t = await call("trim"); step("released the GPU compute buffers kept between runs (trim=1)", undefined, t ? gpuLine(t) : undefined); result.gpuAfterTrim = t; await new Promise((r) => setTimeout(r, 1500)); w.__pkb.phase = "trimmed"; await new Promise((r) => setTimeout(r, 600)); }
  if (cfg.verbose) await call("free"); // a profiling build prints its summary when the backend is freed
  if (result.gpuErrors?.length) throw new Error(`the GPU device reported ${result.gpuErrors.length} error line(s) (see the GPU ERROR steps); results above are not trustworthy`);
  step("done", performance.now() - runStart);
  current.done = true, persist();
  life.phase("done");
  status("done");
  render(result);
  w.__pkb.phase = "done";
  w.__pkb.done = true;
}

// ---------- page ----------
function envLine(a: any) {
  const gpu = !a ? ("gpu" in navigator ? "navigator.gpu present" : "navigator.gpu ABSENT") : a.available ? `${a.vendor} ${a.architecture} ${a.description || ""} · shader-f16 ${a.shaderF16 ? (cfg.f16 ? "present" : "present, not used (f16=0)") : "absent (f32-only shaders)"}` : `no adapter (${a.reason})`;
  return `${gpu} · JSPI ${"Suspending" in WebAssembly ? "available" : "absent (ASYNCIFY build)"} · crossOriginIsolated=${crossOriginIsolated}`;
}
function render(result: any) {
  $("result").textContent = JSON.stringify(result, null, 1);
  const rows = result.clips.map((c: any) => `<tr><td>${c.clip} ${c.seconds} s</td><td>${fmtMs(c.firstRunMs.total)}</td>
    <td>${c.encMs ? `${c.encMs.median} / ${c.encMs.worst}` : "-"}</td><td>${c.totalMs ? `${c.totalMs.median} / ${c.totalMs.worst}` : "-"}</td>
    <td>${c.xRealTime ?? "-"}</td><td>${c.matchesNative ? "matches" : c.wordDiff ? `${c.wordDiff.differing} of ${c.wordDiff.words} words differ (chunked)` : c.hasReference ? "differs" : "no reference"}</td></tr>`).join("");
  $("summary").innerHTML = `<table><tr><th>clip</th><th>first run</th><th>encoder ms<br>median / worst</th><th>total ms<br>median / worst</th><th>x real time</th><th>transcript vs reference</th></tr>${rows}</table>`
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
    ? prev.map((h) => `<details${h.done ? "" : " open"}><summary class="${h.done ? "" : "bad"}">${esc(h.started)} · ${esc(h.config.model)} flash=${h.config.flash ? 1 : 0} store=${esc(h.config.store)} clip=${esc(h.config.clip)} · ${h.done ? "finished" : h.error ? "failed" : `did not finish; last step: ${esc(h.steps.at(-1)?.name ?? "none")}`}</summary><pre>${esc(h.steps.map(stepLine).join("\n"))}${h.error ? "\nERROR: " + esc(h.error) : ""}</pre></details>`).join("")
    : "<p>No earlier runs stored in this browser.</p>";
}
function start() {
  ($("go") as HTMLButtonElement).disabled = true;
  run().catch((e) => {
    const msg = `${e?.name ?? "Error"}: ${e?.message ?? e}`;
    if (current) current.error = msg;
    life.phase("failed");
    step(`ERROR: ${msg}`);
    status("failed");
    (window as any).__pkb.error = msg;
    (window as any).__pkb.done = true;
  }).finally(() => (($("go") as HTMLButtonElement).disabled = false));
}
function init() {
  (window as any).__pkb = { done: false, phase: "idle" };
  const preset = (key: string, note: string) => MODELS[key] ? `<a class="preset" href="${link({ model: key, clip: "a07", runs: 3, auto: 1 })}"><b>${esc(MODELS[key].label)}</b><span>${MODELS[key].mb} MB download. ${note}</span></a>` : "";
  $("presets").innerHTML = preset("s8", "Smallest verified: the 110M model with 8-bit weights packed on the GPU. 7 s clip, 3 warm runs.")
    + preset("s4", "Smaller still, 4-bit weights; punctuation differs from the reference.")
    + preset("q4", "The 0.6b model, 4-bit weights: more accurate, about 0.5 GB of GPU memory.") + preset("q8", "0.6b, 8-bit weights.") + preset("f16", "0.6b, half-float weights.");
  const sel = (id: string, key: keyof Config, options: [string, string][]) => {
    $(id).innerHTML = options.map(([v, l]) => `<option value="${v}"${String(cfg[key] === true ? 1 : cfg[key] === false ? 0 : cfg[key]) === v ? " selected" : ""}>${l}</option>`).join("");
    $(id).addEventListener("change", (e) => (location.href = link({ [key]: (e.target as HTMLSelectElement).value, auto: 0 })));
  };
  sel("model", "model", Object.entries(MODELS).map(([k, v]) => [k, `${v.label}, ${v.mb} MB`]));
  sel("clip", "clip", [...Object.entries(CLIPS).map(([k, v]) => [k, `${k} (${v})`] as [string, string]), ["all", "the three short clips"]]);
  sel("runs", "runs", [["10", "10 warm runs"], ["3", "3 warm runs"], ["0", "first run only"]]);
  sel("store", "store", [["opfs", "model file: OPFS, read in place"], ["opfs-blob", "model file: OPFS as a File"], ["opfs-writable", "model file: OPFS without a sync handle"], ["blob", "model file: fetch Blob"], ["memfs", "model file: copied into WASM heap"]]);
  sel("flash", "flash", [["0", "attention: matmul + softmax"], ["1", "attention: flash kernel"]]);
  sel("variant", "variant", [["jspi", "JSPI build"], ["asyncify", "ASYNCIFY build"]]);
  sel("f16", "f16", [["1", "shader-f16: use if present"], ["0", "shader-f16: never (f32-only shaders)"]]);
  $("env").textContent = envLine(null);
  $("go").addEventListener("click", start);
  $("clear").addEventListener("click", async () => {
    store.clear(), history = [];
    try { await (await navigator.storage.getDirectory()).removeEntry("pk-models", { recursive: true }); } catch { /* nothing stored */ }
    renderHistory();
    status("cleared stored runs and stored model files");
  });
  $("copy").addEventListener("click", () => navigator.clipboard?.writeText(`${$("steps").textContent}\n${$("result").textContent}`));
  renderHistory();
  // Only this page's own scripts' errors become steps; injected scripts' errors (wallets, extensions) are counted and sent to the collector.
  const setNote = () => { $("diagnote").textContent = diag.note(); };
  watchErrors(diag, (label, text) => step(`${label}: ${text}`), () => { (window as any).__pkb.foreign = diag.foreign; setNote(); });
  setNote();
  sendEnvironment(diag, { device, limits: cfg.limits || "raised", config: cfg });
  const prev = life.prev, prevRun = history.at(-1);
  if (prev) {
    const lastStep = prevRun && (!prevRun.sid || prevRun.sid === prev.sid) ? prevRun.steps.at(-1)?.name ?? "none" : "none recorded";
    const line = `previous session ${prev.sid} ended at step "${lastStep.slice(0, 200)}" (phase ${prev.phase}, ${prev.closed ? "page closed normally" : prev.hidden ? "NOT closed normally, in the background" : "NOT closed normally, in front"}, ${Math.round((Date.now() - prev.wall) / 1000)} s ago)`;
    diag.send("previous-session", { line, prev, lastStep, steps: prevRun?.steps.length ?? 0, error: prevRun?.error ?? null });
    if (!prev.closed) $("prevnote").textContent = `This page was not closed normally last time (the browser may have killed or reloaded the tab): ${line}.`;
    // auto=1 after a visit that died mid-run would repeat the same death on every reload: wait for a tap instead (guard=0 turns this off).
    if (!prev.closed && !prev.hidden && ["init", "loading", "first-pass"].includes(prev.phase) && params.get("auto") === "1" && params.get("guard") !== "0") {
      diag.send("guard", { prev, lastStep });
      return status("not started automatically: the last visit died mid-run (see the line above). Tap \"Run this configuration\" to try again, or pick a smaller one.");
    }
  }
  if (params.get("auto") === "1") start();
}
init();
