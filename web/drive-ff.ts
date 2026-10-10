// Second WebGPU implementation: Playwright's Firefox (wgpu + naga instead of Dawn + Tint) on Xvfb.
// usage: PLAYWRIGHT_BROWSERS_PATH=~/devfs/cache/parakeet-ggml-webgpu/pw-browsers bun drive-ff.ts <name> "<query>" [--url http://127.0.0.1:8791/]
// Prints the page's step trail; writes ../results/browser/<name>.json. WebGPU in Firefox on Linux is behind prefs.
import { firefox } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const urlAt = args.indexOf("--url");
const base = urlAt >= 0 ? args.splice(urlAt, 2)[1] : "http://127.0.0.1:8791/";
const [name, query = ""] = args;
const display = process.env.PK_DISPLAY ?? ":195";
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);
const browser = await firefox.launch({
  headless: false, env: { ...process.env, DISPLAY: display },
  firefoxUserPrefs: { "dom.webgpu.enabled": true, "gfx.webgpu.ignore-blocklist": true, "gfx.webgpu.force-enabled": true, "dom.webgpu.workers.enabled": true,
    "gfx.webrender.all": true, "layers.acceleration.force-enabled": true, "dom.fs.enabled": true },
});
const out: any = { name, version: browser.version(), console: [] as string[] };
try {
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") out.console.push(`${m.type()}: ${m.text()}`.slice(0, 800)); });
  page.on("pageerror", (e) => out.console.push(`pageerror: ${e.message}`));
  out.url = `${base}?${query}&auto=1`;
  await page.goto(out.url);
  const t0 = performance.now();
  for (;;) {
    const s = await page.evaluate(() => { const s = (window as any).__pkb; return { done: s?.done, phase: s?.phase }; }).catch(() => ({ done: true, phase: "gone" }));
    if (s.done || performance.now() - t0 > 8 * 60e3) break;
    await Bun.sleep(300);
  }
  Object.assign(out, await page.evaluate(() => { const s = (window as any).__pkb; return { result: s.result, steps: s.steps, error: s.error ?? null }; }));
  out.gpuPage = await page.evaluate(async () => { const g = (navigator as any).gpu; if (!g) return "navigator.gpu undefined on the main thread"; const a = await g.requestAdapter(); return a ? `adapter: ${JSON.stringify({ vendor: a.info?.vendor, arch: a.info?.architecture, desc: a.info?.description })}` : "requestAdapter null"; }).catch((e) => String(e));
} finally { await browser.close().catch(() => {}); xvfb.kill(); }
mkdirSync(join(import.meta.dir, "..", "results", "browser"), { recursive: true });
writeFileSync(join(import.meta.dir, "..", "results", "browser", `${name}.json`), JSON.stringify(out, null, 1));
console.log(`firefox ${out.version}; main thread: ${out.gpuPage}; error: ${out.error}`);
for (const s of out.steps ?? []) console.log(`  ${s.name}${s.ms !== undefined ? `: ${s.ms} ms` : ""}${s.detail ? `  (${s.detail})` : ""}`.slice(0, 600));
for (const l of out.console.slice(0, 12)) console.log("  console:", l.slice(0, 400));
process.exit(0);
