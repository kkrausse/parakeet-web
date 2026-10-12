// Exercises the library's public API (libtest.html, window.__pklib) from outside the page.
//   bun drive-lib.ts webkit NAME [--clip t14]     Playwright's Linux WebKit: createTranscriber + createStream fed from a PCM file (no
//                                                 microphone there), then createMicrophone / createDictation.start() where capture is unavailable
//   bun drive-lib.ts lifecycle NAME [--cycles 5]  Chrome, fake microphone: model switch while recording, then dispose() and a fresh
//                                                 createDictation again and again; the worker's GPU accounting and nvidia-smi after each
// Writes ../results/lib/<name>.json.   [--url http://127.0.0.1:8796/libtest.html] [--browser chrome|webkit]
import { chromium, webkit } from "playwright-core";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { BASE, CACHE, DISPLAY } from "./env";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const url = flag("--url") ?? BASE + "libtest.html", clipName = flag("--clip") ?? "t14", cycles = Number(flag("--cycles") ?? 5);
const [mode, name] = args;
if (!["webkit", "lifecycle"].includes(mode) || !name) throw new Error("usage: bun drive-lib.ts webkit|lifecycle NAME");
const browser = flag("--browser") ?? (mode === "webkit" ? "webkit" : "chrome");
const R = CACHE;
const here = import.meta.dir, outDir = join(here, "..", "results", "lib"), profile = join(here, "..", "cache", "chrome-profiles", `lib-${name}`);
mkdirSync(outDir, { recursive: true });
rmSync(profile, { recursive: true, force: true });
const display = DISPLAY;
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);
const wav = join(R, "audio/live/t56.wav");
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
function differing(a: string[], b: string[]) {
  let prev = new Uint16Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) { const cur = new Uint16Array(b.length + 1); for (let j = 1; j <= b.length; j++) cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]); prev = cur; }
  return a.length + b.length - 2 * prev[b.length];
}
const sh = (cmd: string[]) => new TextDecoder().decode(Bun.spawnSync(cmd).stdout);
const gpuMib = (pid: number) => {
  let sum = 0;
  for (const l of sh(["nvidia-smi"]).split("\n")) { const m = l.match(/^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+.*?(\d+)MiB\s+\|$/); if (m && Number(m[1]) === pid) sum += Number(m[2]); }
  return sum;
};
// A persistent context in both: OPFS sync access handles (the model file) fail in WebKit's ephemeral one.
const ctx = browser === "webkit"
  ? await webkit.launchPersistentContext(profile, { headless: false, viewport: null, env: { ...process.env, DISPLAY: display } })
  : await chromium.launchPersistentContext(profile, {
      executablePath: "/usr/bin/google-chrome", headless: false, viewport: null, env: { ...process.env, DISPLAY: display },
      ignoreDefaultArgs: ["--enable-automation", "--disable-background-timer-throttling"],
      args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--enable-webgpu-developer-features", "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--window-size=1200,860",
        "--enable-dawn-features=vulkan_enable_f16_on_nvidia", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${wav}`],
    });
const result: any = { name, mode, browser, browserVersion: ctx.browser()?.version(), url, loadavg: readFileSync("/proc/loadavg", "utf8").trim() };
const S4 = "models/parakeet-tdt_ctc-110m-Q4_0.gguf", S8 = "models/parakeet-tdt_ctc-110m-Q8_0.gguf";
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") logs.push(m.text().slice(0, 400)); });
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  await page.goto(url);
  await page.waitForFunction(() => !!(window as any).__pklib);

  if (mode === "webkit") {
    const clipFile = join(R, "audio/live", clipName + ".f32"), expectFile = join(R, "audio/live", clipName + ".txt");
    await page.route("**/__clip.f32", (r) => r.fulfill({ body: readFileSync(clipFile), contentType: "application/octet-stream" }));
    // Layer 1 and 2: transcriber, then the stream fed 40 ms blocks from the file. The clock is the audio's (a pass costs
    // no virtual time), so the run takes as long as its passes do; what is checked is the API and the text, not the lag.
    const r: any = await page.evaluate(async (modelUrl) => {
      const lib = (window as any).__pklib, steps: string[] = [], progress: string[] = [];
      const t0 = performance.now();
      const t = await lib.createTranscriber({ assetsUrl: "pk/", modelUrl, maxSeconds: 12,
        onProgress: (p: any) => { const s = `${p.phase} ${Math.round(p.fraction * 100)}%`; if (progress.at(-1) !== s) progress.push(s); },
        advanced: { onStep: (s: any) => { if (!/^download:/.test(s.name)) steps.push(s.ms === undefined ? s.name : `${s.name}: ${Math.round(s.ms)} ms`); } } });
      const loadMs = Math.round(performance.now() - t0);
      const pcm = new Float32Array(await (await fetch("__clip.f32")).arrayBuffer());
      const off = await t.transcribe(pcm);
      let clock = 0, committed = "", tentativeUpdates = 0;
      const passMs: number[] = [];
      const s = lib.createStream(async (p: Float32Array) => { const x = await t.transcribe(p, { transfer: true }); passMs.push(Math.round(x.ms)); return x; },
        { onCommit: (c: any) => { committed += (committed ? " " : "") + c.text; }, onTentative: () => { tentativeUpdates++; } }, { capS: 12, now: () => clock });
      const s0 = performance.now();
      for (let i = 0; i < pcm.length; i += 640) { s.push(pcm.subarray(i, i + 640)); clock += 40; while (s.stats().busy) await new Promise((res) => setTimeout(res, 2)); }
      await s.stop();
      const st = s.stats(), info = t.info;
      const after = await t.dispose();
      let afterDispose = "";
      try { await t.transcribe(new Float32Array(1600)); } catch (e: any) { afterDispose = `${e.kind}: ${e.message}`; }
      passMs.sort((a, b) => a - b);
      return { loadMs, steps, progress, info: { backend: info.backend, webgpu: info.webgpu, variant: info.variant, jspi: info.jspi, adapter: info.adapter.available ? info.adapterName : `none: ${info.adapter.reason}`, cpuReason: info.cpuReason, modelFrom: info.modelFrom, storagePath: info.storagePath },
        clipSeconds: pcm.length / 16000, offline: { text: off.text, ms: Math.round(off.ms), words: off.words.length }, stream: { committed, tentativeUpdates, wallS: Math.round((performance.now() - s0) / 100) / 10, passes: st.passes, passMsMedian: passMs[passMs.length >> 1], passMsMax: passMs.at(-1), byReason: st.byReason, utterances: st.utterances, maxBufferS: st.maxBufferS },
        disposeReturned: after, transcribeAfterDispose: afterDispose, state: t.state };
    }, S4);
    const exp = existsSync(expectFile) ? readFileSync(expectFile, "utf8").trim() : "";
    r.expected = exp;
    r.offlineVsExpected = `${differing(norm(exp), norm(r.offline.text))} of ${norm(exp).length} words`;
    r.streamVsExpected = `${differing(norm(exp), norm(r.stream.committed))} of ${norm(exp).length} words`;
    r.streamVsOffline = `${differing(norm(r.offline.text), norm(r.stream.committed))} of ${norm(r.offline.text).length} words`;
    result.transcriberAndStream = r;
    // Layer 3 where capture is unavailable: how start() ends and how long that takes.
    result.microphone = await page.evaluate(async () => {
      const lib = (window as any).__pklib, out: any = { secureContext: isSecureContext, hasGetUserMedia: !!navigator.mediaDevices?.getUserMedia, hasAudioWorklet: typeof AudioWorkletNode === "function" };
      const statuses: string[] = [];
      const mic = lib.createMicrophone({ assetsUrl: "pk/", onAudio() {}, onStatus: (s: string) => statuses.push(s), startTimeoutMs: 8000 });
      const t0 = performance.now();
      try { const info = await mic.start(); out.started = info; mic.stop(); } catch (e: any) { out.error = { isMicrophoneError: lib.isMicrophoneError(e), code: e.code, message: e.message, cause: e.cause ? `${e.cause.name}: ${e.cause.message}` : null }; }
      out.ms = Math.round(performance.now() - t0); out.statuses = statuses; out.statusAfter = mic.status;
      return out;
    });
    // The same through createDictation: start() resolves false and onError has the reason.
    result.dictationStart = await page.evaluate(async (modelUrl) => {
      const lib = (window as any).__pklib, errors: any[] = [], states: string[] = [];
      const d = lib.createDictation({ assetsUrl: "pk/", modelUrl, onError: (e: any) => errors.push({ code: e.code, kind: e.kind, message: e.message }), onState: (s: any) => states.push(`${s.model}/${s.recording}/${s.microphone}`) });
      const t0 = performance.now();
      const ok = await Promise.race([d.start(), new Promise((res) => setTimeout(() => res("NO ANSWER IN 90 s"), 90000))]);
      const ms = Math.round(performance.now() - t0);
      const info = await d.ready.then((i: any) => i.backend, (e: any) => `load failed: ${e.message}`);
      await d.dispose();
      return { startResolved: ok, ms, errors, states, modelAfterStart: info };
    }, S4);
  } else {
    const { processInfo } = await (await ctx.browser()!.newBrowserCDPSession()).send("SystemInfo.getProcessInfo");
    const gpuPid = processInfo.find((p) => p.type === "GPU")?.id ?? 0;
    const tap = async (code: string) => { await page.evaluate((c) => { (window as any).__ontap = new Function(`return (${c})`)(); }, code); await page.click("#tap"); return page.evaluate(() => (window as any).__tapResult); };
    const brief = "(g) => g && ({ weightsMb: g.weightsMb, workingMb: g.workingMb, buffers: g.buffers, created: g.created, destroyed: g.destroyed, jsBuffersLive: g.js ? g.js.buffers - g.js.buffersDestroyed : null, jsBufferLiveMb: g.js?.bufferLiveMb ?? null })";
    result.gpuMibBefore = gpuMib(gpuPid);
    // 1. One dictation, the model switched while it records.
    await page.evaluate(async ([s4, briefSrc]) => {
      const lib = (window as any).__pklib, w = window as any, brief = new Function(`return (${briefSrc})`)();
      const T: any = (w.__t = { committed: "", commits: [] as any[], states: [] as string[], errors: [] as string[], loads: [] as any[], t0: performance.now() });
      const at = () => Math.round((performance.now() - T.t0) / 100) / 10;
      w.__d = lib.createDictation({ assetsUrl: "pk/", modelUrl: s4, maxUtteranceSeconds: 12,
        onCommit: (text: string) => { T.committed += (T.committed ? " " : "") + text; T.commits.push([at(), text.split(/\s+/).length, w.__d.transcriber?.info.modelFile ?? "(none)"]); },
        onState: (s: any) => T.states.push(`${at()} ${s.model}/${s.recording ? "rec" : "idle"}/${s.microphone}`), onError: (e: any) => T.errors.push(`${e.kind ?? e.code}: ${e.message}`) });
      const i = await w.__d.ready;
      T.loads.push({ at: at(), file: i.modelFile, backend: i.backend, gpu: brief(w.__d.transcriber.gpuStats()) });
    }, [S4, brief]);
    result.gpuMibS4Ready = gpuMib(gpuPid);
    await tap("() => window.__d.start()");
    await Bun.sleep(20000);
    const sw = await page.evaluate(async ([s8, briefSrc]) => {
      const w = window as any, T = w.__t, brief = new Function(`return (${briefSrc})`)(), at = () => Math.round((performance.now() - T.t0) / 100) / 10;
      const wordsBefore = T.committed.split(/\s+/).filter(Boolean).length, t0 = at();
      const i = await w.__d.load(s8);
      T.loads.push({ at: at(), file: i.modelFile, backend: i.backend, gpu: brief(w.__d.transcriber.gpuStats()) });
      return { switchStartedAt: t0, switchReadyAt: at(), wordsBefore, recordingThroughout: w.__d.state.recording };
    }, [S8, brief]);
    await Bun.sleep(20000);
    result.gpuMibS8Recording = gpuMib(gpuPid);
    result.modelSwitch = { ...sw, ...(await page.evaluate(async () => { const w = window as any, T = w.__t; await w.__d.stop(); const e = w.__d.stats(); return { committed: T.committed, words: T.committed.split(/\s+/).filter(Boolean).length, commits: T.commits, states: T.states, errors: T.errors, loads: T.loads, engine: { passes: e.passes, utterances: e.utterances, failed: e.failed, byReason: e.byReason } }; })) };
    await page.evaluate(() => (window as any).__d.dispose());
    await Bun.sleep(2500);
    result.gpuMibAfterDispose = gpuMib(gpuPid);
    // 2. dispose() and a fresh createDictation, again and again: 6 s of recording each time.
    result.cycles = [];
    for (let c = 1; c <= cycles; c++) {
      await page.evaluate(async (s4) => {
        const lib = (window as any).__pklib, w = window as any;
        const T: any = (w.__t = { committed: "", errors: [] as string[] });
        w.__d = lib.createDictation({ assetsUrl: "pk/", modelUrl: s4, maxUtteranceSeconds: 12, onCommit: (text: string) => { T.committed += (T.committed ? " " : "") + text; }, onError: (e: any) => T.errors.push(`${e.kind ?? e.code}: ${e.message}`) });
        await w.__d.ready;
      }, S4);
      const ready = gpuMib(gpuPid);
      const started = await tap("() => window.__d.start()");
      await Bun.sleep(6000);
      const row = await page.evaluate(async (briefSrc) => {
        const w = window as any, T = w.__t, brief = new Function(`return (${briefSrc})`)();
        await w.__d.stop();
        const t = w.__d.transcriber, live = brief(t.gpuStats()), passes = w.__d.stats().passes;
        // the transcriber's own dispose() frees the model and the device first and reports what the accounting says then
        const freed = brief(await t.dispose());
        await w.__d.dispose();
        return { words: T.committed.split(/\s+/).filter(Boolean).length, passes, errors: T.errors, live, afterFree: freed, stateAfter: w.__d.state.model };
      }, brief);
      await Bun.sleep(2500);
      result.cycles.push({ cycle: c, started, gpuMibReady: ready, ...row, gpuMibAfterDispose: gpuMib(gpuPid) });
      console.log(JSON.stringify(result.cycles.at(-1)));
    }
  }
  result.console = logs;
} catch (e: any) { result.error = String(e?.stack ?? e); console.error(result.error); }
finally { await ctx.close().catch(() => {}); xvfb.kill(); }
writeFileSync(join(outDir, `${name}.json`), JSON.stringify(result, null, 1));
console.log(JSON.stringify({ ...result, cycles: undefined }, null, 1));
process.exit(result.error ? 1 : 0);
