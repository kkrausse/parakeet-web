// Drives the live page in real Chrome with Chrome's fake microphone fed from a wav file, records while
// the file plays, stops, and writes ../results/live/<name>.json: committed text, segments, pass times,
// lag, memory over time (GPU process from nvidia-smi, renderer RSS, WASM heap), and word diffs against
// an offline pass over the same audio (through the same worker) and against a reference text.
// usage: bun drive-live.ts <name> "<query>" --wav FILE.wav [--f32 FILE.f32] [--expect FILE.txt] [--seconds N]
//        [--url http://127.0.0.1:8791/live.html] [--viewport 390x844] [--shot SECONDS] [--no-webgpu] [--chrome-arg ARG]...
import { chromium } from "playwright-core";
import { mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const bool = (k: string) => { const i = args.indexOf(k); if (i < 0) return false; args.splice(i, 1); return true; };
const url0 = flag("--url") ?? "http://127.0.0.1:8791/live.html";
const wav = flag("--wav"), f32 = flag("--f32"), expectFile = flag("--expect");
const viewport = flag("--viewport"), shotAt = Number(flag("--shot") ?? 0), tail = Number(flag("--tail") ?? 2);
const noWebgpu = bool("--no-webgpu");
const css = flag("--css"); // extra CSS injected before recording (experiments)
const sampleMs = Number(flag("--sample-ms") ?? 2000);
const chromeArgs: string[] = [];
for (let a = flag("--chrome-arg"); a; a = flag("--chrome-arg")) chromeArgs.push(a);
if (!wav) throw new Error("--wav is required");
const wavSeconds = (() => { const b = readFileSync(wav); return (b.length - 44) / (b.readUInt32LE(28)); })();
const seconds = Number(flag("--seconds") ?? Math.floor(wavSeconds - tail)); // stop inside the trailing silence: after a %noloop file ends Chrome repeats its last buffer
const [name, query = ""] = args;
if (!name) throw new Error("usage: bun drive-live.ts <name> <query> --wav FILE");

const here = import.meta.dir;
const outDir = join(here, "..", "results", "live");
const profile = join(here, "..", "cache", "chrome-profiles", `live-${name}`);
mkdirSync(outDir, { recursive: true });
rmSync(profile, { recursive: true, force: true });
const display = process.env.PK_DISPLAY ?? ":196";
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);

const sh = (cmd: string[]) => new TextDecoder().decode(Bun.spawnSync(cmd).stdout);
const gpuMib = (pid: number) => {
  let sum = 0;
  for (const l of sh(["nvidia-smi"]).split("\n")) {
    const m = l.match(/^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+.*?(\d+)MiB\s+\|$/);
    if (m && Number(m[1]) === pid) sum += Number(m[2]);
  }
  return sum;
};
const procMib = (pid: number, key: string) => { try { return Math.round(Number(readFileSync(`/proc/${pid}/status`, "utf8").match(new RegExp(`${key}:\\s+(\\d+)`))![1]) / 1024); } catch { return 0; } };
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
/** Word-level diff (LCS): count of inserted + deleted words and the changed spans with one word of context. */
function diff(a: string[], b: string[]) {
  const n = a.length, m = b.length, w = m + 1;
  const t = new Uint16Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) t[i * w + j] = a[i] === b[j] ? t[(i + 1) * w + j + 1] + 1 : Math.max(t[(i + 1) * w + j], t[i * w + j + 1]);
  const spans: string[] = [];
  let i = 0, j = 0, del: string[] = [], ins: string[] = [], ctx = "";
  const flush = () => { if (del.length || ins.length) spans.push(`${ctx} [${del.join(" ")} -> ${ins.join(" ")}]`), (del = []), (ins = []); };
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { flush(); ctx = a[i]; i++; j++; }
    else if (j < m && (i === n || t[i * w + j + 1] >= t[(i + 1) * w + j])) ins.push(b[j++]);
    else del.push(a[i++]);
  }
  flush();
  return { differing: n + m - 2 * t[0], words: n, spans };
}
const stat = (xs: number[]) => { const s = [...xs].sort((p, q) => p - q); return s.length ? { n: s.length, median: s[s.length >> 1], p95: s[Math.floor(s.length * 0.95)], max: s[s.length - 1] } : null; };

const [vw, vh] = (viewport ?? "").split("x").map(Number);
const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: "/usr/bin/google-chrome", headless: false,
  viewport: viewport ? { width: vw, height: vh } : null, ...(viewport ? { deviceScaleFactor: 2, isMobile: true, hasTouch: true } : {}),
  env: { ...process.env, DISPLAY: display },
  ignoreDefaultArgs: ["--enable-automation", "--disable-background-timer-throttling"],
  args: [...(noWebgpu ? ["--disable-features=WebGPU"] : ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--enable-webgpu-developer-features"]),
    "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--window-size=1200,860",
    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${wav}%noloop`,
    ...chromeArgs],
});
const result: any = { name, wav, wavSeconds, recordSeconds: seconds, chromeArgs, noWebgpu, viewport, loadavgStart: readFileSync("/proc/loadavg", "utf8").trim(), chrome: ctx.browser()?.version() };
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") logs.push(`${m.type()}: ${m.text()}`.slice(0, 600)); });
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  page.on("crash", () => logs.push("PAGE CRASHED"));
  if (f32) await page.route("**/__test.f32", (r) => r.fulfill({ body: readFileSync(f32), contentType: "application/octet-stream", headers: { "Access-Control-Allow-Origin": "*" } }));
  const pids = async () => {
    const { processInfo } = await (await ctx.browser()!.newBrowserCDPSession()).send("SystemInfo.getProcessInfo");
    return { gpu: processInfo.find((p) => p.type === "GPU")?.id ?? 0, renderers: processInfo.filter((p) => p.type === "renderer").map((p) => p.id) };
  };
  result.url = `${url0}${query ? (url0.includes("?") ? "&" : "?") + query : ""}`;
  const t0 = performance.now();
  await page.goto(result.url);
  const state = () => page.evaluate(() => { const s = (window as any).__pkl; return { phase: s.phase, idle: s.idle, lagS: s.lagS, bufferS: s.bufferS, heapMb: s.heapMb, passCount: s.passCount, js: Math.round(((performance as any).memory?.usedJSHeapSize ?? 0) / 2 ** 20), words: s.committed.split(/\s+/).filter(Boolean).length, vad: s.vad() }; });
  for (;;) {
    const s = await state();
    if (s.phase !== "loading") { if (s.phase !== "ready") throw new Error(`model did not load: ${JSON.stringify(await page.evaluate(() => (window as any).__pkl.errors))}`); break; }
    if (performance.now() - t0 > 15 * 60e3) throw new Error("timeout loading the model");
    await Bun.sleep(200);
  }
  result.readyMs = Math.round(performance.now() - t0);
  const p = await pids();
  const sample = async (t: number) => { const s = await state(); return { t: Math.round(t), gpuMib: gpuMib(p.gpu), rendererRssMib: Math.max(0, ...p.renderers.map((r) => procMib(r, "VmRSS"))), gpuProcRssMib: procMib(p.gpu, "VmRSS"), wasmHeapMb: s.heapMb, jsHeapMb: s.js, lagS: s.lagS, bufferS: s.bufferS, passes: s.passCount, words: s.words, vad: s.vad }; };
  result.beforeRecording = await sample(0);
  if (css) await page.addStyleTag({ content: css });
  await page.click("#rec");
  const r0 = performance.now();
  const series: any[] = [];
  let shot = false;
  while ((performance.now() - r0) / 1000 < seconds) {
    const t = (performance.now() - r0) / 1000;
    if (shotAt && !shot && t >= shotAt) { shot = true; await page.screenshot({ path: join(outDir, `${name}.png`) }); }
    series.push(await sample(t));
    await Bun.sleep(sampleMs);
  }
  await page.click("#rec");
  for (const s0 = performance.now(); !(await state()).idle; await Bun.sleep(100)) if (performance.now() - s0 > 10 * 60e3) throw new Error("timeout waiting for the last passes");
  result.drainMs = Math.round(performance.now() - r0 - seconds * 1000);
  const live = await page.evaluate(() => { const s = (window as any).__pkl; return { committed: s.committed, provisional: s.provisional, segments: s.segments, passes: s.passes, passCount: s.passCount, finalCount: s.finalCount, reusedCount: s.reusedCount, staleCount: s.staleCount, maxLagS: s.maxLagS, droppedS: s.droppedS, inputRate: s.inputRate, backend: s.backend, webgpu: s.webgpu, shaderF16: s.shaderF16, variant: s.variant, errors: s.errors, steps: s.steps, params: s.params, load: s.load }; });
  series.push(await sample(seconds + result.drainMs / 1000));
  const prov = live.passes.filter((x: number[]) => !x[2]), fin = live.passes.filter((x: number[]) => x[2]);
  const settled = series.filter((s) => s.t >= Math.min(30, seconds / 3));
  const range = (k: string) => ({ first: settled[0]?.[k], last: settled.at(-1)?.[k], min: Math.min(...settled.map((s) => s[k])), max: Math.max(...settled.map((s) => s[k])) });
  Object.assign(result, {
    live: { ...live, passes: undefined, steps: undefined }, steps: live.steps, console: logs,
    provisionalPassMs: stat(prov.map((x: number[]) => x[0])), finalPassMs: stat(fin.map((x: number[]) => x[0])),
    provisionalAudioS: stat(prov.map((x: number[]) => x[1])), finalAudioS: stat(fin.map((x: number[]) => x[1])),
    lagS: stat(series.map((s) => s.lagS)), maxLagS: live.maxLagS,
    memoryAfterWarmup: { gpuMib: range("gpuMib"), rendererRssMib: range("rendererRssMib"), gpuProcRssMib: range("gpuProcRssMib"), wasmHeapMb: range("wasmHeapMb"), jsHeapMb: range("jsHeapMb") },
    series, passes: live.passes,
  });
  if (f32) {
    const off = await page.evaluate(() => (window as any).__pkl.offline("__test.f32"));
    result.offline = off;
    result.vsOffline = { exact: off.text.trim() === live.committed, raw: diff(off.text.trim().split(/\s+/), live.committed.split(/\s+/)), normalized: diff(norm(off.text), norm(live.committed)) };
  }
  if (expectFile && existsSync(expectFile)) {
    const exp = readFileSync(expectFile, "utf8").trim();
    result.vsExpected = { exact: exp === live.committed, raw: diff(exp.split(/\s+/), live.committed.split(/\s+/)), normalized: diff(norm(exp), norm(live.committed)) };
  }
  result.loadavgEnd = readFileSync("/proc/loadavg", "utf8").trim();
} catch (e: any) { result.error = String(e?.message ?? e); }
finally { await ctx.close().catch(() => {}); xvfb.kill(); }
writeFileSync(join(outDir, `${name}.json`), JSON.stringify(result, null, 1));
const d = (x: any) => (x ? `${x.raw.differing}/${x.raw.words} raw, ${x.normalized.differing}/${x.normalized.words} words` : "-");
console.log(JSON.stringify({ name, error: result.error, backend: result.live?.backend, f16: result.live?.shaderF16, variant: result.live?.variant, adapter: result.live?.load?.adapter, inputRate: result.live?.inputRate,
  commits: result.live?.finalCount, reused: result.live?.reusedCount, reasons: result.live?.segments?.map((s: any) => `${s.reason}:${(s.endS - s.startS).toFixed(1)}`).join(" "), passes: result.live?.passCount, stale: result.live?.staleCount,
  provMs: result.provisionalPassMs, finalMs: result.finalPassMs, provAudioS: result.provisionalAudioS, lag: result.lagS, maxLagS: result.maxLagS, drainMs: result.drainMs, droppedS: result.live?.droppedS,
  vsOffline: d(result.vsOffline), offlineSpans: result.vsOffline?.normalized.spans.slice(0, 12), vsExpected: d(result.vsExpected), expectedSpans: result.vsExpected?.normalized.spans.slice(0, 12),
  memory: result.memoryAfterWarmup, pageErrors: result.live?.errors, console: result.console?.slice(0, 5) }, null, 1));
process.exit(0);
