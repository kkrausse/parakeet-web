// Does a copy of the live page reach "ready", and on which backend? Opens it, waits for the model, prints the
// status line, the backend line and any page error. No microphone, no audio.
// usage: bun drive-ready.ts URL [--browser webkit|chrome] [--timeout-s 180]
import { chromium, webkit } from "playwright-core";
import { rmSync } from "node:fs";
import { join } from "node:path";
import { DISPLAY } from "./env";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const browser = flag("--browser") ?? "webkit", timeoutS = Number(flag("--timeout-s") ?? 180), url = args[0];
if (!url) throw new Error("usage: bun drive-ready.ts URL [--browser webkit|chrome]");
const profile = join(import.meta.dir, "..", "cache", "chrome-profiles", `ready-${browser}`);
rmSync(profile, { recursive: true, force: true });
const env = { ...process.env, DISPLAY };
const ctx = browser === "webkit"
  ? await webkit.launchPersistentContext(profile, { headless: false, viewport: null, env })
  : await chromium.launchPersistentContext(profile, { executablePath: "/usr/bin/google-chrome", headless: false, viewport: null, env, args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--no-first-run"] });
const page = ctx.pages()[0] ?? (await ctx.newPage());
const errors: string[] = [], requests: string[] = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("response", (r) => { if (/\.(gguf|wasm)$/.test(new URL(r.url()).pathname)) requests.push(`${r.request().method()} ${r.status()} ${r.url()}`); });
const t0 = Date.now();
await page.goto(url);
let status = "";
while (Date.now() - t0 < timeoutS * 1000) {
  status = (await page.textContent("#mstatus")) ?? "";
  if (/ ready$|not running|not loaded|stopped|lost/.test(status)) break;
  await new Promise((r) => setTimeout(r, 500));
}
const out = { url, browser, status, seconds: Math.round((Date.now() - t0) / 100) / 10, backend: await page.textContent("#s-backend"), notice: await page.textContent("#errors"), diagnostics: await page.textContent("#diagnote"), modelRequests: requests, pageErrors: errors };
console.log(JSON.stringify(out, null, 1));
await ctx.close();
process.exit(/ ready$/.test(status) ? 0 : 1);
