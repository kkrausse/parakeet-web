// Parakeet (TDT 0.6b v2 and tdt_ctc-110m) in the browser on transcribe.cpp + ggml's WebGPU backend (WASM). One page,
// no framework. Everything heavy happens in src/worker.ts; this file is configuration, the step
// trail (screen + localStorage, so a killed tab leaves one) and the benchmark loop.
export {};
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
const CLIPS: Record<string, string> = { a07: "7.0 s", a14: "13.7 s", a56: "56.1 s" };
declare const MODELS: Record<string, { label: string; file: string; mb: number; family: "0.6b" | "110m" }>; // filled in by build.ts from what is on disk

interface Config {
  model: string; clip: string; runs: number; threads: number;
  store: "opfs" | "opfs-blob" | "blob" | "memfs"; // where the model file lives while it is loaded
  f16: boolean; // false (f16=0): do not use shader-f16 even where the adapter has it (the f32-only shader path)
  flash: boolean; // ggml FLASH_ATTN_EXT in the encoder (off: plain matmul + softmax attention)
  variant: "jspi" | "asyncify" | "prof"; // prof: JSPI build with ggml CPU-side profiling (use with verbose=1)
  verbose: boolean; base: string; env: Record<string, string>;
}
const params = new URLSearchParams(location.search);
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
    store: pick("store", ["opfs", "opfs-blob", "blob", "memfs"], "opfs"),
    f16: params.get("f16") !== "0",
    flash: params.get("flash") === "1",
    variant: pick("variant", ["jspi", "asyncify", "prof"], "Suspending" in WebAssembly ? "jspi" : "asyncify"),
    verbose: params.get("verbose") === "1",
    base: params.get("base") ?? "models/",
    env,
  };
}
const cfg = readConfig();

// ---------- persistent step log ----------
interface Step { t: number; name: string; ms?: number; detail?: string }
interface RunLog { id: string; started: string; config: Config; ua: string; steps: Step[]; done: boolean; error?: string }
const HISTORY_KEY = "pkggml:runs";
function loadHistory(): RunLog[] { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]"); } catch { return []; } }
let history = loadHistory();
let current: RunLog | null = null;
function persist() { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-8))); } catch { /* private mode or full */ } }
const $ = (id: string) => document.getElementById(id)!;
const r1 = (x: number) => Math.round(x * 10) / 10;
function fmtMs(ms: number) { return ms >= 10000 ? `${(ms / 1000).toFixed(1)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${r1(ms)} ms`; }
function stepLine(s: Step) { return `${(s.t / 1000).toFixed(1).padStart(6)}s  ${s.name}${s.ms !== undefined ? `: ${fmtMs(s.ms)}` : ""}${s.detail ? `  (${s.detail})` : ""}`; }
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
function stats(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  return { median: r1(s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2), worst: r1(s[s.length - 1]), best: r1(s[0]) };
}

// ---------- worker RPC ----------
let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
function call<T = any>(type: string, body: Record<string, unknown> = {}, transfer: Transferable[] = []): Promise<T> {
  const id = nextId++;
  return new Promise<T>((resolve, reject) => { pending.set(id, { resolve, reject }); worker!.postMessage({ type, id, ...body }, transfer); });
}
function startWorker() {
  worker = new Worker(new URL("worker.js", location.href), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") return status(`downloading model: ${Math.round(m.got / 2 ** 20)} / ${Math.round(m.total / 2 ** 20)} MB`);
    if (m.type === "stdout") { if (cfg.verbose) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
    if (m.type === "stderr") { if (cfg.verbose || /error|abort|fail/i.test(m.line)) step(`wasm: ${String(m.line).slice(0, 400)}`); return; }
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.error && cfg.verbose && m.stack) step(`stack: ${String(m.stack).slice(0, 2500)}`);
    m.error ? p.reject(new Error(m.error)) : p.resolve(m.out);
  };
  worker.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(`worker error: ${e.message}`)); pending.clear(); };
}

// ---------- the run ----------
async function run() {
  const w = window as any;
  runStart = performance.now();
  current = { id: String(Date.now()), started: new Date().toISOString(), config: cfg, ua: navigator.userAgent, steps: [], done: false };
  history.push(current);
  $("steps").textContent = "";
  const model = MODELS[cfg.model];
  const EXPECTED = model.family === "110m" ? EXPECTED_110M : EXPECTED_06B;
  const result: any = { config: cfg, model, ua: navigator.userAgent, crossOriginIsolated, clips: [] };
  w.__pkb.result = result;
  step(`start: ${model.label}, flash=${cfg.flash ? "on" : "off"}, store=${cfg.store}, ${cfg.variant}`);

  w.__pkb.phase = "init";
  startWorker();
  const init = await call("init", { base: location.href, variant: cfg.variant, wantF16: cfg.f16, limits: cfg.env.GGML_WEBGPU_LIMITS ?? "" });
  result.adapter = init.adapter; result.jspi = init.jspi;
  const a = init.adapter;
  step("WASM module ready", init.ms, `heap ${init.heapMb} MB; JSPI ${init.jspi ? "available" : "absent"}`);
  step(a.available ? `adapter: ${a.vendor} ${a.architecture} ${a.description || a.device || ""}; shader-f16 ${a.shaderF16 ? "present" : "ABSENT"}; fallback=${a.isFallbackAdapter}` : `no WebGPU adapter: ${a.reason}`);
  $("env").textContent = envLine(a);
  if (a.available) {
    const L = a.limits, mib = (x: number) => Math.round(x / 2 ** 20);
    step(`adapter limits: binding ${mib(L.maxStorageBufferBindingSize)} MiB, buffer ${mib(L.maxBufferSize)} MiB, ${L.maxComputeInvocationsPerWorkgroup} invocations/workgroup, workgroup storage ${L.maxComputeWorkgroupStorageSize} B, ${L.maxStorageBuffersPerShaderStage} storage buffers/stage`,
      undefined, a.belowSpecDefault.length ? `BELOW THE WEBGPU SPEC DEFAULT: ${a.belowSpecDefault.join("; ")}` : "all at or above the WebGPU spec defaults, which is all this page needs");
    for (const e of a.probeErrors ?? []) step(`device request FAILED: ${e}`);
    if (!a.plan) throw new Error(`WebGPU is present but no device could be created, even with the spec-default limits and no features: ${a.deviceError}`);
    step(`device probe ${a.deviceProbe}`, undefined, a.plan.limits === "default" ? "spec-default limits" : `requested ${JSON.stringify(a.requested.requiredLimits)} ${a.requested.requiredFeatures.join(",") || "no features"}`);
    if (L.maxStorageBufferBindingSize < 2 ** 28) step(`note: tensors above ${mib(L.maxStorageBufferBindingSize)} MiB cannot be bound on this adapter; long clips may run partly on the CPU`);
  } else step("WebGPU is unavailable: the model would run on one WASM thread (very slow)");

  w.__pkb.phase = "sessions";
  status("fetching and loading the model");
  // Without shader-f16 the backend compiles f32-only shaders; the two pointwise convs must then use an F32 im2col
  // (ggml_conv_2d's F16 one would bounce to the CPU). Flash attention needs F16 masks, so it is off on that path.
  const useF16 = !!a.shaderF16 && cfg.f16 && a.plan?.f16 !== false;
  result.shaderF16Used = useF16;
  const flash = cfg.flash && useF16;
  if (cfg.flash && !flash) step("flash attention needs shader-f16: using matmul + softmax attention instead");
  step(useF16 ? "shader path: f16" : `shader path: f32 only (${!a.shaderF16 ? "adapter has no shader-f16" : cfg.f16 ? "the device was refused with shader-f16" : "f16=0 requested"})`);
  const env: Record<string, string> = { TRANSCRIBE_NO_FLASH: flash ? "" : "1", TRANSCRIBE_F32_MASK_CONCAT: flash ? "1" : "",
    GGML_WEBGPU_NO_F16: useF16 ? "" : "1", TRANSCRIBE_F32_POINTWISE: useF16 ? "" : "1",
    TRANSCRIBE_PRE_ENCODE_TILE: "128", // clips over 15 s: subsampling convs in 10 s time tiles (exact): their activations no longer grow with the clip
    ...(a.plan?.limits === "default" ? { GGML_WEBGPU_LIMITS: "default" } : {}), ...cfg.env };
  const ld = await call("load", { url: new URL(cfg.base + model.file, location.href).href, name: model.file, store: cfg.store, env, threads: cfg.threads, verbose: cfg.verbose });
  result.load = { fetchMs: Math.round(ld.fetchMs), loadMs: Math.round(ld.loadMs), from: ld.from, fileMb: r1(ld.mb), wasmHeapMb: ld.heapMb, wasmHeapUsedMb: ld.heapUsedMb, backend: ld.backend };
  result.session = { totalMs: Math.round(ld.loadMs) };
  step(`model file ready (${ld.from})`, ld.fetchMs, `${ld.mb.toFixed(0)} MB`);
  step(`model loaded on ${ld.backend}`, ld.loadMs, `WASM heap ${ld.heapMb} MB, ${ld.heapUsedMb} MB in use`);
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

  const clips = cfg.clip === "all" ? Object.keys(CLIPS) : [cfg.clip];
  for (const [i, clip] of clips.entries()) {
    w.__pkb.phase = `run:${clip}`;
    const audio = new Float32Array(await (await fetch(`audio/${clip}.f32`)).arrayBuffer());
    const seconds = audio.length / 16000;
    const once = async () => {
      const r = await call("run", { pcm: audio });
      gpuFail(clip, r.gpuErrors);
      if (cfg.verbose && r.log) for (const l of String(r.log).split("\n")) if (/decoder:/.test(l)) step(`library: ${l.trim().slice(0, 300)}`);
      // encoder = until its output is on the CPU: with lazy synchronize the library's own encode_ms stops at submit
      return { pre: r.mel_ms as number, enc: (r.wallMs - r.mel_ms - r.decode_ms) as number, encSubmit: r.encode_ms as number, dec: r.decode_ms as number, total: r.wallMs as number, text: r.text as string, tokens: r.n_tokens as number, heapMb: r.heapMb as number, heapUsedMb: r.heapUsedMb as number };
    };
    status(`${clip}: first run`);
    const first = await once();
    step(`${clip} (${CLIPS[clip]}) first run${i === 0 ? " (cold)" : ""}`, first.total, `mel ${fmtMs(first.pre)}, encoder ${fmtMs(first.enc)}, decode ${fmtMs(first.dec)}, heap ${first.heapMb} MB (${first.heapUsedMb} in use)`);
    const rec: any = {
      clip, seconds: Math.round(seconds * 100) / 100, firstInPage: i === 0,
      firstRunMs: { pre: r1(first.pre), enc: r1(first.enc), dec: r1(first.dec), total: r1(first.total) },
      startToFirstTranscriptMs: i === 0 ? Math.round(performance.now() - runStart) : null,
      decSteps: first.tokens, text: first.text, matchesNative: first.text === EXPECTED[clip],
    };
    step(`${clip} transcript ${rec.matchesNative ? "matches" : "DIFFERS from"} the reference`, undefined, first.text.length > 90 ? first.text.slice(0, 90) + "..." : first.text);
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
    render(result);
  }
  if (params.get("trim") === "1") { await call("trim"); step("released the GPU compute buffer kept between runs (trim=1)"); await new Promise((r) => setTimeout(r, 1500)); w.__pkb.phase = "trimmed"; await new Promise((r) => setTimeout(r, 600)); }
  if (cfg.verbose) await call("free"); // a profiling build prints its summary when the backend is freed
  if (result.gpuErrors?.length) throw new Error(`the GPU device reported ${result.gpuErrors.length} error line(s) (see the GPU ERROR steps); results above are not trustworthy`);
  step("done", performance.now() - runStart);
  current.done = true, persist();
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
function esc(s: string) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!); }
function render(result: any) {
  $("result").textContent = JSON.stringify(result, null, 1);
  const rows = result.clips.map((c: any) => `<tr><td>${c.clip} ${c.seconds} s</td><td>${fmtMs(c.firstRunMs.total)}</td>
    <td>${c.encMs ? `${c.encMs.median} / ${c.encMs.worst}` : "-"}</td><td>${c.totalMs ? `${c.totalMs.median} / ${c.totalMs.worst}` : "-"}</td>
    <td>${c.xRealTime ?? "-"}</td><td>${c.matchesNative ? "matches" : "differs"}</td></tr>`).join("");
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
  sel("clip", "clip", [...Object.entries(CLIPS).map(([k, v]) => [k, `${k} (${v})`] as [string, string]), ["all", "all three"]]);
  sel("runs", "runs", [["10", "10 warm runs"], ["3", "3 warm runs"], ["0", "first run only"]]);
  sel("store", "store", [["opfs", "model file: OPFS, read in place"], ["opfs-blob", "model file: OPFS as a File"], ["blob", "model file: fetch Blob"], ["memfs", "model file: copied into WASM heap"]]);
  sel("flash", "flash", [["0", "attention: matmul + softmax"], ["1", "attention: flash kernel"]]);
  sel("variant", "variant", [["jspi", "JSPI build"], ["asyncify", "ASYNCIFY build"]]);
  sel("f16", "f16", [["1", "shader-f16: use if present"], ["0", "shader-f16: never (f32-only shaders)"]]);
  $("env").textContent = envLine(null);
  $("go").addEventListener("click", start);
  $("clear").addEventListener("click", async () => {
    localStorage.removeItem(HISTORY_KEY), history = [];
    try { await (await navigator.storage.getDirectory()).removeEntry("pk-models", { recursive: true }); } catch { /* nothing stored */ }
    renderHistory();
    status("cleared stored runs and stored model files");
  });
  $("copy").addEventListener("click", () => navigator.clipboard?.writeText(`${$("steps").textContent}\n${$("result").textContent}`));
  renderHistory();
  addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => step(`unhandled rejection: ${e.reason?.message ?? e.reason}`));
  addEventListener("error", (e: ErrorEvent) => step(`uncaught error: ${e.message}`));
  if (params.get("auto") === "1") start();
}
init();
