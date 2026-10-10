// Checks of the diagnostics and error handling of the live page in real Chrome (no microphone needed):
//   foreign: once the model is ready, a cross-origin script that throws, an injected inline script with a
//            window.ethereum-style TypeError and a foreign rejection must be ignored (counted, sent to the
//            collector); an error and a rejection thrown from live.js itself must still be shown; a pass still runs.
//   kill:    the renderer is killed (SIGKILL) at the first step matching --at while the model loads; the reload
//            must show the "not closed normally" line and the guard instead of loading again; "Try again" loads.
// usage: bun drive-diag.ts foreign|kill [--url http://127.0.0.1:8791/live.html] [--query "phone=1"] [--at "download:"] [--any-phase]
//        kill inside a soak test: --query "soak=2000" --at "soak pass 25" --any-phase
import { chromium } from "playwright-core";
import { rmSync } from "node:fs";
import { join } from "node:path";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const url0 = flag("--url") ?? "http://127.0.0.1:8791/live.html", query = flag("--query") ?? "", at = flag("--at") ?? "download:";
const anyPhase = (() => { const i = args.indexOf("--any-phase"); if (i < 0) return false; args.splice(i, 1); return true; })(); // kill: wait for the --at step even after the model is ready (e.g. inside a soak)
const mode = args[0];
const url = `${url0}${query ? (url0.includes("?") ? "&" : "?") + query : ""}`;
const profile = join(import.meta.dir, "..", "cache", "chrome-profiles", `diag-${mode}`);
rmSync(profile, { recursive: true, force: true });
const display = process.env.PK_DISPLAY ?? ":197";
const xvfb = Bun.spawn(["Xvfb", display, "-screen", "0", "1280x900x24", "-nolisten", "tcp", "-ac"], { stderr: "ignore" });
await Bun.sleep(700);
const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: "/usr/bin/google-chrome", headless: false, viewport: null, env: { ...process.env, DISPLAY: display },
  ignoreDefaultArgs: ["--enable-automation"],
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan", "--no-first-run", "--no-default-browser-check", "--password-store=basic",
    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
});
const out: any = { mode, url };
const pk = (page: any) => page.evaluate(() => { const s = (window as any).__pkl; return { phase: s.phase, errors: s.errors, foreign: s.foreign ?? [], steps: s.steps.map((x: any) => x.name), sid: document.getElementById("diagshort")!.textContent, errorsBox: document.getElementById("errors")!.hidden ? null : document.getElementById("errors")!.textContent, prevnote: document.getElementById("prevnote")!.hidden ? null : document.getElementById("prevnote")!.textContent, note: document.getElementById("diagnote")!.textContent, status: document.getElementById("mstatus")!.textContent }; });
const until = async (page: any, ok: (s: any) => boolean, what: string, ms = 120000) => { for (const t0 = Date.now(); ; await Bun.sleep(15)) { const s = await pk(page); if (ok(s)) return s; if (Date.now() - t0 > ms) throw new Error(`timeout waiting for ${what}: ${JSON.stringify(s)}`); } };
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  if (mode === "foreign") {
    // hooks inside the page's own script, so "own" errors really come from live.js
    await page.route("**/live.js", async (r) => { const resp = await r.fetch(); await r.fulfill({ response: resp, body: (await resp.text()) + `\n;window.__throwOwn = () => { throw new Error("own test error"); }; window.__rejectOwn = () => { Promise.reject(new Error("own test rejection")); };\n` }); });
    await ctx.route("https://foreign.example/throw.js", (r) => r.fulfill({ contentType: "text/javascript", body: `throw new Error("boom from another origin");` }));
    await page.goto(url);
    await until(page, (s) => s.phase === "ready", "ready");
    await page.addScriptTag({ url: "https://foreign.example/throw.js" }); // -> "Script error." with no filename
    await page.addScriptTag({ content: "window.ethereum.selectedAddress = undefined" }); // -> TypeError from an inline script
    await page.addScriptTag({ content: `Promise.reject(new TypeError("undefined is not an object (evaluating 'window.ethereum.selectedAddress = undefined')"))` });
    await page.addScriptTag({ content: `Promise.reject("plain string from nowhere")` });
    await Bun.sleep(500);
    out.afterForeign = await pk(page);
    const pass = await page.evaluate(async () => { const r = await (window as any).__pkl.run(new Float32Array(16000)); return { ms: r.wallMs, heapMb: r.heapMb }; });
    out.passAfterForeign = pass;
    await page.evaluate(() => { setTimeout((window as any).__throwOwn, 0); setTimeout((window as any).__rejectOwn, 0); });
    await Bun.sleep(500);
    out.afterOwn = await pk(page);
    const f = out.afterForeign, o = out.afterOwn;
    out.ok = f.phase === "ready" && f.errors.length === 0 && f.errorsBox === null && f.foreign.length === 4 && o.errors.length === 2 && /own test error/.test(o.errorsBox ?? "") && /own test rejection/.test(o.errorsBox ?? "") && o.foreign.length === 4;
  } else if (mode === "kill") {
    const cdp = await ctx.browser()!.newBrowserCDPSession();
    await page.goto(url);
    const before = await until(page, (s) => s.steps.some((n: string) => n.includes(at)) || (!anyPhase && s.phase !== "loading"), `a step matching "${at}"`);
    out.killedAt = before.steps.at(-1); out.killedSid = before.sid; out.phaseAtKill = before.phase;
    const { processInfo } = await cdp.send("SystemInfo.getProcessInfo");
    const crashed = new Promise((r) => page.once("crash", r));
    for (const p of processInfo.filter((p: any) => p.type === "renderer")) process.kill(p.id, "SIGKILL");
    await Promise.race([crashed, Bun.sleep(5000)]);
    await Bun.sleep(300);
    const page2 = await ctx.newPage(); // the crashed tab cannot navigate under Playwright; same profile, so the same localStorage
    await page2.goto(url);
    await Bun.sleep(1500);
    out.afterReload = await pk(page2);
    await page2.getByRole("button", { name: "Try again" }).click();
    out.afterRetry = await until(page2, (s) => s.phase === "ready" || s.phase === "failed", "ready after Try again");
    await page2.reload(); // a normal reload: no guard, no "not closed normally" line
    out.afterNormalReload = await until(page2, (s) => s.phase === "ready" || s.phase === "failed" || s.phase === "guard", "ready after a normal reload");
    out.ok = out.afterReload.phase === "guard" && /not closed normally/.test(out.afterReload.prevnote ?? "") && out.afterRetry.phase === "ready" && out.afterNormalReload.phase === "ready" && out.afterNormalReload.prevnote === null;
  } else throw new Error("mode: foreign | kill");
} catch (e: any) { out.error = String(e?.stack ?? e); }
finally { await ctx.close().catch(() => {}); xvfb.kill(); }
const brief = (s: any) => s && { ...s, steps: `${s.steps.length} steps, last: ${s.steps.at(-1)}` };
for (const k of ["afterForeign", "afterOwn", "afterReload", "afterRetry", "afterNormalReload"]) if (out[k]) out[k] = brief(out[k]);
console.log(JSON.stringify(out, null, 1));
process.exit(out.ok ? 0 : 1);
