// Drives the live page's soak test in real Chrome (no microphone): N passes back to back over a fixture
// clip cut to a different length each time, and writes ../results/soak/<name>.json with the page's own
// counters per pass (GPU buffers live / made, bind groups, JS objects made / collected) next to what the
// machine saw (GPU process memory from nvidia-smi, renderer and GPU process RSS).
// usage: bun drive-soak.ts <name> "<query>" [--passes 300] [--url http://127.0.0.1:8791/live.html] [--chrome-arg ARG]...
import { chromium } from "playwright-core";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const url0 = flag("--url") ?? "http://127.0.0.1:8791/live.html";
const passes = Number(flag("--passes") ?? 300);
const chromeArgs: string[] = [];
for (let a = flag("--chrome-arg"); a; a = flag("--chrome-arg")) chromeArgs.push(a);
const [name, query = ""] = args;
if (!name) throw new Error("usage: bun drive-soak.ts <name> <query> [--passes N]");

const here = import.meta.dir;
const outDir = join(here, "..", "results", "soak");
const profile = join(here, "..", "cache", "chrome-profiles", `soak-${name}`);
mkdirSync(outDir, { recursive: true });
rmSync(profile, { recursive: true, force: true });
const display = process.env.PK_DISPLAY ?? ":197";
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

const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: "/usr/bin/google-chrome", headless: false, viewport: null,
  env: { ...process.env, DISPLAY: display },
  ignoreDefaultArgs: ["--enable-automation", "--disable-background-timer-throttling"],
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--enable-webgpu-developer-features",
    "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--window-size=1200,860", ...chromeArgs],
});
const result: any = { name, passes, chromeArgs, loadavgStart: readFileSync("/proc/loadavg", "utf8").trim(), chrome: ctx.browser()?.version() };
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") logs.push(`${m.type()}: ${m.text()}`.slice(0, 600)); });
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  page.on("crash", () => logs.push("PAGE CRASHED"));
  result.url = `${url0}${url0.includes("?") ? "&" : "?"}soak=${passes}${query ? "&" + query : ""}`;
  const t0 = performance.now();
  await page.goto(result.url);
  const pids = async () => {
    const { processInfo } = await (await ctx.browser()!.newBrowserCDPSession()).send("SystemInfo.getProcessInfo");
    return { gpu: processInfo.find((p) => p.type === "GPU")?.id ?? 0, renderers: processInfo.filter((p) => p.type === "renderer").map((p) => p.id) };
  };
  const state = () => page.evaluate(() => { const s = (window as any).__pkl; return { phase: s.phase, errors: s.errors, done: s.soak?.done ?? 0, running: s.soak?.running ?? null, heapMb: s.heapMb, gpu: s.gpu }; });
  const series: any[] = [];
  let p: Awaited<ReturnType<typeof pids>> | null = null;
  for (;;) {
    const s = await state();
    if (s.phase === "failed" || s.phase === "guard") throw new Error(`page ${s.phase}: ${JSON.stringify(s.errors)}`);
    if (s.running !== null) {
      p ??= await pids();
      series.push({ t: Math.round(performance.now() - t0), pass: s.done, gpuMib: gpuMib(p.gpu), rendererRssMib: Math.max(0, ...p.renderers.map((r) => procMib(r, "VmRSS"))), gpuProcRssMib: procMib(p.gpu, "VmRSS"), wasmHeapMb: s.heapMb,
        working: s.gpu?.workingMb, buffers: s.gpu?.buffers, made: s.gpu?.created, jsObjects: s.gpu?.js?.objects, jsCollected: s.gpu?.js?.collected });
      if (s.running === false) break;
    }
    if (performance.now() - t0 > 40 * 60e3) throw new Error("timeout");
    await Bun.sleep(1000);
  }
  const out = await page.evaluate(() => { const s = (window as any).__pkl; return { soak: s.soak, gpu: s.gpu, gpuAtLoad: s.gpuAtLoad, gpuAfterWarmup: s.gpuAfterWarmup, backend: s.backend, variant: s.variant, steps: s.steps, errors: s.errors, text: document.getElementById("soak")!.textContent }; });
  Object.assign(result, out, { series, console: logs });
  result.loadavgEnd = readFileSync("/proc/loadavg", "utf8").trim();
} catch (e: any) { result.error = String(e?.message ?? e); }
finally { await ctx.close().catch(() => {}); xvfb.kill(); }
writeFileSync(join(outDir, `${name}.json`), JSON.stringify(result, null, 1));
const S = result.soak ?? {}, ser: any[] = result.series ?? [], late = ser.filter((s) => s.pass > 20);
const range = (k: string) => (late.length ? `${Math.min(...late.map((s) => s[k]))}-${Math.max(...late.map((s) => s[k]))}` : "-");
const rows: number[][] = S.series ?? [];
const after = rows.filter((r) => r[0] > 10);
console.log(JSON.stringify({ name, error: result.error, backend: result.backend, variant: result.variant, done: S.done, seconds: S.seconds, medianMs: S.medianMs, flat: S.flat, sameText: S.sameText, soakError: S.error,
  atLoad: result.gpuAtLoad && { weightsMb: result.gpuAtLoad.weightsMb, byType: result.gpuAtLoad.byType, workingMb: result.gpuAtLoad.workingMb, buffers: result.gpuAtLoad.buffers },
  afterWarmup: result.gpuAfterWarmup && { workingMb: result.gpuAfterWarmup.workingMb, buffers: result.gpuAfterWarmup.buffers, made: result.gpuAfterWarmup.created },
  end: result.gpu && { workingMb: result.gpu.workingMb, buffers: result.gpu.buffers, made: result.gpu.created, destroyed: result.gpu.destroyed, poolMisses: result.gpu.poolMisses, js: result.gpu.js },
  first10: { buffersMade: rows.filter((r) => r[0] <= 10).reduce((s, r) => s + r[5], 0) },
  afterPass10: { passes: after.length, buffersMade: after.reduce((s, r) => s + r[5], 0), passesThatMadeABuffer: after.filter((r) => r[5] > 0).length, workingMb: after.length ? `${Math.min(...after.map((r) => r[3]))}-${Math.max(...after.map((r) => r[3]))}` : "-",
    bindGroupsPerPass: after.length ? `${Math.min(...after.map((r) => r[6]))}-${Math.max(...after.map((r) => r[6]))}` : "-", jsUncollected: after.length ? `${Math.min(...after.map((r) => r[7]))}-${Math.max(...after.map((r) => r[7]))}` : "-" },
  machineAfterPass20: { gpuMib: range("gpuMib"), rendererRssMib: range("rendererRssMib"), gpuProcRssMib: range("gpuProcRssMib"), wasmHeapMb: range("wasmHeapMb") },
  console: result.console?.slice(0, 5) }, null, 1));
process.exit(0);
