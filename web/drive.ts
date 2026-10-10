// Runs one configuration of the page in real Chrome (headed, on its own Xvfb display, WebGPU on
// Vulkan) and writes ../results/browser/<name>.json: the page's result plus what only the outside
// can see (GPU memory of Chrome's GPU process from nvidia-smi, renderer RSS, chrome://gpu facts).
// usage: bun drive.ts <name> "<query string>" [--url http://127.0.0.1:8787/] [--keep-profile] [--timeout-min 30] [--chrome-arg ARG]...
import { chromium } from "playwright-core";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const base = flag("--url") ?? "http://127.0.0.1:8787/";
const timeoutMin = Number(flag("--timeout-min") ?? 30);
const chromeArgs: string[] = [];
for (let a = flag("--chrome-arg"); a; a = flag("--chrome-arg")) chromeArgs.push(a);
const keep = args.includes("--keep-profile") && !!args.splice(args.indexOf("--keep-profile"), 1);
const [name, query = ""] = args;
if (!name) throw new Error("usage: bun drive.ts <name> <query>");

const here = import.meta.dir;
const outDir = join(here, "..", "results", "browser");
const profile = join(here, "..", "cache", "chrome-profiles", keep ? "kept" : name);
mkdirSync(outDir, { recursive: true });
if (!keep) rmSync(profile, { recursive: true, force: true });

const display = process.env.PK_DISPLAY ?? ":196"; // our own; :197 is the ORT bench, :97-:99 belong to other things
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);

const sh = (cmd: string[]) => new TextDecoder().decode(Bun.spawnSync(cmd).stdout);
const gpuMib = (pid: number) => {
  // "|    0   N/A  N/A   1234    C+G   name    158MiB |"
  let sum = 0;
  for (const l of sh(["nvidia-smi"]).split("\n")) {
    const m = l.match(/^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\S+\s+.*?(\d+)MiB\s+\|$/);
    if (m && Number(m[1]) === pid) sum += Number(m[2]);
  }
  return sum;
};
const procMib = (pid: number, key: string) => {
  try { return Math.round(Number(readFileSync(`/proc/${pid}/status`, "utf8").match(new RegExp(`${key}:\\s+(\\d+)`))![1]) / 1024); } catch { return 0; }
};
const load = () => ({ loadavg: readFileSync("/proc/loadavg", "utf8").trim(), cpuPressure: readFileSync("/proc/pressure/cpu", "utf8").split("\n")[0] });

const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: "/usr/bin/google-chrome", headless: false, viewport: null,
  env: { ...process.env, DISPLAY: display },
  ignoreDefaultArgs: ["--enable-automation", "--disable-background-timer-throttling"],
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan",
    "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--window-size=1200,860",
    "--enable-webgpu-developer-features", // adapter.info then names the device instead of leaving it blank
    ...chromeArgs],
});
const result: any = { name, url: "", chromeArgs, machineAtStart: load(), chrome: ctx.browser()?.version() };
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") logs.push(`${m.type()}: ${m.text()}`.slice(0, 600)); });
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  page.on("crash", () => logs.push("PAGE CRASHED"));

  const cdp = await ctx.newCDPSession(page);
  const pids = async () => {
    const { processInfo } = await (ctx.browser() ? await ctx.browser()!.newBrowserCDPSession() : cdp).send("SystemInfo.getProcessInfo");
    return { gpu: processInfo.find((p) => p.type === "GPU")?.id ?? 0, renderers: processInfo.filter((p) => p.type === "renderer").map((p) => p.id) };
  };

  const sep = query && !query.startsWith("?") ? "?" : "";
  result.url = `${base}${sep}${query}${query ? "&" : "?"}auto=1`;
  const t0 = performance.now();
  await page.goto(result.url);
  let p = await pids();
  const mem = { gpuProcessGpuMibAfterLoad: 0, gpuProcessGpuMibPeak: 0, rendererRssMibAfterLoad: 0, rendererRssMibPeak: 0, gpuProcessRssMibAfterLoad: 0, gpuProcessRssMibPeak: 0 };
  let phase = "", sawLoaded = false, state: any = {};
  for (;;) {
    state = await page.evaluate(() => { const s = (window as any).__pkb; return { done: s.done, phase: s.phase, error: s.error ?? null }; }).catch((e) => ({ done: true, error: `page gone: ${e.message}` }));
    const g = gpuMib(p.gpu), rss = Math.max(0, ...p.renderers.map((r) => procMib(r, "VmRSS"))), grss = procMib(p.gpu, "VmRSS");
    mem.gpuProcessGpuMibPeak = Math.max(mem.gpuProcessGpuMibPeak, g);
    mem.rendererRssMibPeak = Math.max(mem.rendererRssMibPeak, rss);
    mem.gpuProcessRssMibPeak = Math.max(mem.gpuProcessRssMibPeak, grss);
    if (state.phase !== phase) {
      phase = state.phase;
      console.error(`[${((performance.now() - t0) / 1000).toFixed(1)}s] ${phase}  gpu=${g} MiB  renderer rss=${rss} MiB  gpu-process rss=${grss} MiB`);
      if (phase === "sessions") p = await pids();
    }
    if (state.phase === "loaded" && !sawLoaded) sawLoaded = true, mem.gpuProcessGpuMibAfterLoad = g, mem.rendererRssMibAfterLoad = rss, mem.gpuProcessRssMibAfterLoad = grss;
    if (state.done) break;
    if (performance.now() - t0 > timeoutMin * 60e3) { state.error = `driver timeout after ${timeoutMin} min in phase ${phase}`; break; }
    await Bun.sleep(150);
  }
  const page_ = await page.evaluate(() => { const s = (window as any).__pkb; return { result: s.result, steps: s.steps, error: s.error ?? null }; }).catch(() => ({ result: null, steps: null, error: null }));
  Object.assign(result, { error: state.error ?? page_.error, memory: mem, pids: p, page: page_.result, steps: page_.steps, console: logs, machineAtEnd: load() });

  // What chrome://gpu says about WebGPU and the GL/Vulkan backend.
  try {
    const { gpu } = await (await ctx.browser()!.newBrowserCDPSession()).send("SystemInfo.getInfo");
    result.chromeGpu = { devices: gpu.devices, driverBugWorkarounds: undefined, featureStatus: (gpu as any).featureStatus, auxAttributes: Object.fromEntries(Object.entries(gpu.auxAttributes ?? {}).filter(([k]) => /vendor|renderer|version|vulkan|display|angle|passthrough/i.test(k))) };
  } catch (e) { result.chromeGpu = { error: String(e) }; }
} finally {
  await ctx.close().catch(() => {});
  xvfb.kill();
}
writeFileSync(join(outDir, `${name}.json`), JSON.stringify(result, null, 1));
const clips = (result.page?.clips ?? []).map((c: any) => `${c.clip}: first ${c.firstRunMs.total} enc ${c.encMs?.median}/${c.encMs?.worst} total ${c.totalMs?.median}/${c.totalMs?.worst} x${c.xRealTime} match=${c.matchesNative}`);
console.log(JSON.stringify({ name, error: result.error, adapter: result.page?.adapter && `${result.page.adapter.vendor}/${result.page.adapter.architecture}/${result.page.adapter.description}`, load: result.page?.load, memory: result.memory, clips }, null, 1));
process.exit(0);
