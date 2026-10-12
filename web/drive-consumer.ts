// Drives an app built on the library from outside, knowing nothing of its internals: Chrome with the fake
// microphone playing a wav, a click on the record button, a click to stop, then the textarea against a reference
// text. Writes ../results/consumer/<name>.json.
// usage: bun drive-consumer.ts <name> --wav FILE.wav [--expect FILE.txt] [--url http://127.0.0.1:8795/] [--button "#rec"] [--text "#text"]
//        [--status "#status"] [--ready "model ready"] [--seconds N] [--stock] [--no-webgpu]
import { chromium } from "playwright-core";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { DISPLAY } from "./env";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const bool = (k: string) => { const i = args.indexOf(k); if (i < 0) return false; args.splice(i, 1); return true; };
const url = flag("--url") ?? "http://127.0.0.1:8795/", wav = flag("--wav"), expectFile = flag("--expect");
const button = flag("--button") ?? "#rec", text = flag("--text") ?? "#text", status = flag("--status") ?? "#status", readyText = flag("--ready") ?? "model ready";
const stock = bool("--stock"), noWebgpu = bool("--no-webgpu");
if (!wav) throw new Error("--wav is required");
const wavSeconds = (() => { const b = readFileSync(wav); return (b.length - 44) / b.readUInt32LE(28); })();
const seconds = Number(flag("--seconds") ?? Math.floor(wavSeconds - 2)); // stop inside the trailing silence (see drive-live.ts)
const [name] = args;
if (!name) throw new Error("usage: bun drive-consumer.ts <name> --wav FILE");
const here = import.meta.dir, outDir = join(here, "..", "results", "consumer"), profile = join(here, "..", "cache", "chrome-profiles", `consumer-${name}`);
mkdirSync(outDir, { recursive: true });
rmSync(profile, { recursive: true, force: true });
const display = DISPLAY;
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean);
/** Words outside the longest common subsequence (insertions + deletions). */
function differing(a: string[], b: string[]) {
  let prev = new Uint16Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) { const cur = new Uint16Array(b.length + 1); for (let j = 1; j <= b.length; j++) cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]); prev = cur; }
  return a.length + b.length - 2 * prev[b.length];
}
const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: "/usr/bin/google-chrome", headless: false, viewport: null, env: { ...process.env, DISPLAY: display },
  ignoreDefaultArgs: ["--enable-automation", "--disable-background-timer-throttling"],
  args: [...(noWebgpu ? ["--disable-features=WebGPU"] : ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--enable-webgpu-developer-features"]),
    "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--window-size=1200,860",
    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${wav}%noloop`,
    ...(stock ? [] : ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"])],
});
const result: any = { name, url, wav, wavSeconds, recordSeconds: seconds, chrome: ctx.browser()?.version(), loadavg: readFileSync("/proc/loadavg", "utf8").trim() };
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [], requests: string[] = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") logs.push(`${m.type()}: ${m.text()}`.slice(0, 600)); });
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  ctx.on("request", (r) => requests.push(`${r.method()} ${new URL(r.url()).pathname}`));
  ctx.on("response", (r) => { if (r.status() >= 400) logs.push(`${r.status()} ${new URL(r.url()).pathname}`); });
  const t0 = performance.now();
  await page.goto(url);
  const read = (sel: string) => page.evaluate((s) => { const el: any = document.querySelector(s); return String(el?.value ?? el?.textContent ?? ""); }, sel);
  for (;;) {
    const s = await read(status);
    if (s.includes(readyText)) break;
    if (performance.now() - t0 > 10 * 60e3) throw new Error(`the app did not say "${readyText}" in 10 min; status: ${s}`);
    await Bun.sleep(200);
  }
  result.readyMs = Math.round(performance.now() - t0);
  await page.click(button);
  const r0 = performance.now();
  const growth: [number, number][] = []; // seconds, words on screen
  while ((performance.now() - r0) / 1000 < seconds) { growth.push([Math.round((performance.now() - r0) / 100) / 10, (await read(text)).split(/\s+/).filter(Boolean).length]); await Bun.sleep(2000); }
  result.statusWhileRecording = await read(status);
  await page.click(button);
  let last = "", same = 0;
  for (const s0 = performance.now(); same < 15 && performance.now() - s0 < 60e3; await Bun.sleep(200)) { const v = await read(text); same = v === last ? same + 1 : 0; last = v; } // unchanged for 3 s
  result.text = last; result.statusAfter = await read(status); result.wordsOverTime = growth;
  result.requests = [...new Set(requests)];
  result.console = logs;
  if (expectFile && existsSync(expectFile)) {
    const exp = readFileSync(expectFile, "utf8").trim();
    result.expected = exp;
    result.vsExpected = { exact: exp === last, rawDiffering: differing(exp.split(/\s+/), last.split(/\s+/)), normalizedDiffering: differing(norm(exp), norm(last)), expectedWords: norm(exp).length, gotWords: norm(last).length };
  }
} catch (e: any) { result.error = String(e?.message ?? e); }
finally { await ctx.close().catch(() => {}); xvfb.kill(); }
writeFileSync(join(outDir, `${name}.json`), JSON.stringify(result, null, 1));
console.log(JSON.stringify({ name, error: result.error, readyMs: result.readyMs, vsExpected: result.vsExpected, statusWhileRecording: result.statusWhileRecording, statusAfter: result.statusAfter, requests: result.requests, console: result.console, text: result.text }, null, 1));
process.exit(result.error ? 1 : 0);
