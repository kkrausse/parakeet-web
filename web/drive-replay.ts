// Replays recordings through the streaming engine (and the old endpointer) with the real model in Chrome, and
// compares each run's committed text with a whole-file pass over the same audio. Audio is made phone-like in the
// page (replay/lib.ts CONDITIONS). Writes ../results/replay/<name>.json (measurements) and ../cache/replay/<name>/
// (full logs, not in git); prints one line per run.
// usage: bun drive-replay.ts <name> [--clip iv90] [--conditions clean,low,phone,noisy,far] [--engines old,new] [--model s4] [--cap 12]
//        [--params '{"holdS":0.8}'] [--seconds N] [--latency measured|<ms>] [--slow <x real time>] [--url http://127.0.0.1:8793/replay.html] [--stock]
import { chromium } from "playwright-core";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { CONDITIONS, coverage, measure, type Condition } from "./replay/lib";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const bool = (k: string) => { const i = args.indexOf(k); if (i < 0) return false; args.splice(i, 1); return true; };
const url0 = flag("--url") ?? "http://127.0.0.1:8793/replay.html";
const clipName = flag("--clip") ?? "iv90", model = flag("--model") ?? "s4", cap = Number(flag("--cap") ?? 12);
const conds = (flag("--conditions") ?? "clean,low,phone,noisy,far").split(","), engines = (flag("--engines") ?? "old,new").split(",") as ("old" | "new")[];
const paramsJson = flag("--params"), seconds = Number(flag("--seconds") ?? 0) || undefined, lat = flag("--latency"), slow = Number(flag("--slow") ?? 0) || undefined;
const stock = bool("--stock"), saveRef = bool("--save-ref");
const [name] = args;
if (!name) throw new Error("usage: bun drive-replay.ts <name> [--clip iv90] ...");
const R = process.env.PK_CACHE ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu");
const clipFile = [join(R, "audio/replay", clipName + ".f32"), join(R, "audio/live", clipName + ".f32"), join(R, "audio", clipName + ".f32")].find(existsSync);
if (!clipFile) throw new Error(`no ${clipName}.f32 under ${R}/audio (scripts/replay-clips.sh makes them)`);
const here = import.meta.dir, outDir = join(here, "..", "results", "replay"), rawDir = join(here, "..", "cache", "replay", name);
mkdirSync(outDir, { recursive: true }); mkdirSync(rawDir, { recursive: true });
const profile = join(here, "..", "cache", "chrome-profiles", `replay-${name}`);
rmSync(profile, { recursive: true, force: true });
const display = process.env.PK_DISPLAY ?? ":197";
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);
const procMib = (pid: number) => { try { return Math.round(Number(readFileSync(`/proc/${pid}/status`, "utf8").match(/VmRSS:\s+(\d+)/)![1]) / 1024); } catch { return 0; } };

const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: "/usr/bin/google-chrome", headless: false, viewport: null, env: { ...process.env, DISPLAY: display },
  ignoreDefaultArgs: ["--enable-automation", "--disable-background-timer-throttling"],
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--enable-webgpu-developer-features", "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--window-size=1200,860",
    ...(stock ? [] : ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"])],
});
const clean = new Float32Array(readFileSync(clipFile).buffer.slice(0));
const result: any = { name, clip: clipName, clipSeconds: clean.length / 16000, model, capS: cap, params: paramsJson ? JSON.parse(paramsJson) : {}, latency: slow ? `${slow} x audio` : lat ?? "measured", chrome: ctx.browser()?.version(), runs: [] };
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") logs.push(m.text().slice(0, 400)); });
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  await page.route("**/__clip.f32", (r) => r.fulfill({ body: readFileSync(clipFile), contentType: "application/octet-stream" }));
  await page.goto(`${url0}?model=${model}&cap=${cap}`);
  for (const t0 = performance.now(); ; await Bun.sleep(200)) {
    const s = await page.evaluate(() => { const p = (window as any).__pkr; return { phase: p.phase, error: p.error }; });
    if (s.phase === "ready") break;
    if (s.phase === "failed") throw new Error(`model did not load: ${s.error}`);
    if (performance.now() - t0 > 10 * 60e3) throw new Error("timeout loading the model");
  }
  Object.assign(result, await page.evaluate(() => { const p = (window as any).__pkr; return { backend: p.backend, webgpu: p.webgpu, shaderF16: p.shaderF16, adapter: p.adapter }; }));
  const { processInfo } = await (await ctx.browser()!.newBrowserCDPSession()).send("SystemInfo.getProcessInfo");
  const renderers = processInfo.filter((p) => p.type === "renderer").map((p) => p.id), gpuPid = processInfo.find((p) => p.type === "GPU")?.id ?? 0;
  const rss = () => ({ rendererRssMib: Math.max(0, ...renderers.map(procMib)), gpuProcRssMib: procMib(gpuPid) });
  for (const c of conds) {
    const condition: Condition = CONDITIONS[c] ?? JSON.parse(c);
    const spec = { clip: "__clip.f32", condition, seconds };
    const logsOf: Record<string, any> = {};
    for (const engine of engines) {
      const before = rss();
      const rssSeries: any[] = [];
      const timer = setInterval(() => rssSeries.push({ wall: Math.round(performance.now()), ...rss() }), 2000);
      let log: any;
      try { log = await page.evaluate((s) => (window as any).__pkr.run(s), { ...spec, engine, capS: cap, params: result.params, latencyMs: lat && lat !== "measured" ? Number(lat) : "measured", slow }); }
      finally { clearInterval(timer); }
      logsOf[engine] = { log, before, after: rss(), rssSeries };
    }
    // the reference pass comes after the runs: it makes the GPU working buffer grow for the whole file
    const ref = await page.evaluate((s) => (window as any).__pkr.offline(s), spec);
    if (saveRef && c === "clean") writeFileSync(join(outDir, `ref-${clipName}-${model}.json`), JSON.stringify({ clip: clipName, model, seconds: clean.length / 16000, text: ref.text, words: ref.words }));
    writeFileSync(join(rawDir, `${c}-ref.json`), JSON.stringify(ref));
    for (const engine of engines) {
      const { log, before, after, rssSeries } = logsOf[engine];
      writeFileSync(join(rawDir, `${c}-${engine}.json`), JSON.stringify({ ...log, rssSeries }));
      const sub = seconds ? clean.subarray(0, seconds * 16000) : clean;
      const m = measure(log, ref), cov = coverage(log, sub);
      const mem = log.memory as any[], half = mem.filter((x) => x.t >= Math.min(30, log.seconds / 3));
      const range = (k: string) => (half.length ? { first: half[0][k], last: half.at(-1)[k], min: Math.min(...half.map((x) => x[k])), max: Math.max(...half.map((x) => x[k])) } : null);
      const bufs = (log.series as any[]).map((x) => x.bufferS), passS = (log.passes as any[]).map((p) => (p.endSample - p.startSample) / 16000);
      const run = { condition: c, engine, ...m, coverage: cov, stats: log.stats, wallS: Math.round(log.wallS * 10) / 10, passes: log.passes.length || log.stats.passes, passMsMedian: [...log.passMs].sort((a: number, b: number) => a - b)[log.passMs.length >> 1],
        maxBufferS: log.stats.maxBufferS ?? Math.max(0, ...bufs), maxPassAudioS: Math.max(0, ...passS), skipped: log.skipped,
        memory: { jsHeapMb: range("jsHeapMb"), wasmHeapMb: range("wasmHeapMb"), gpuWorkMb: range("gpuWorkMb"), gpuBuffers: range("gpuBuffers"), gpuMade: range("gpuMade"), rssBefore: before, rssAfter: after },
        refText: ref.text, committed: log.committed };
      result.runs.push(run);
      console.log(`${c.padEnd(6)} ${engine.padEnd(3)} ref ${String(m.refWords).padStart(4)} w | committed ${String(m.committedWords).padStart(4)} | WER ${String(m.wer).padStart(5)}% | contradicted ${String(m.contradicted).padStart(3)} missed ${String(m.missed).padStart(3)} | no text ${String(m.longestUnshownS).padStart(5)} s (${m.unshownWords} w) | commit lat ${m.commitLatency.median}/${m.commitLatency.p95} s | show ${m.showLatency.median}/${m.showLatency.p95} s | gate lost ${cov.lostPct}% longest ${cov.longestLostS} s | segs ${cov.segments} dropped ${cov.dropped} | maxbuf ${Math.round((run.maxBufferS ?? 0) * 10) / 10} s | ${engine === "new" ? `opens ${log.stats.gateOpens} misses ${log.stats.gateMisses} probes ${log.stats.probes} ${JSON.stringify(log.stats.byReason)}` : `maxfloor ${log.stats.maxNoiseFloor?.toFixed(4)}`} | ${run.wallS} s`);
    }
  }
  result.console = logs;
} catch (e: any) { result.error = String(e?.stack ?? e); console.error(result.error); }
finally { await ctx.close().catch(() => {}); xvfb.kill(); }
writeFileSync(join(outDir, `${name}.json`), JSON.stringify(result, null, 1));
process.exit(result.error ? 1 : 0);
