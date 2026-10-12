// Markdown tables for the browser rows in results/browser/ (chrome-*.json from web/drive.ts and the
// native-*.jsonl reruns taken in the same window). usage: bun scripts/browser-table.ts [110m]
// With 110m: only the parakeet-tdt_ctc-110m rows (names containing -110m-); without: only the 0.6b rows.
import { readdirSync, readFileSync } from "node:fs";
const dir = `${import.meta.dir}/../results/browser`;
const mw = (s?: { median: number; worst: number }) => (s ? `${Math.round(s.median)} / ${Math.round(s.worst)}` : "");
const small = Bun.argv.includes("110m");
const timing: string[] = [], load: string[] = [], failed: string[] = [];
for (const f of readdirSync(dir).sort()) {
  if (f.startsWith("published-")) continue; // single-clip checks of the published copy
  if (f.includes("-110m-") !== small) continue;
  if (f.endsWith(".jsonl")) {
    const recs = readFileSync(`${dir}/${f}`, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    const l = recs.find((r) => r.record === "load"), a = recs.filter((r) => r.record === "audio");
    if (!l || a.length < 3) { failed.push(`${f}: incomplete`); continue; }
    timing.push(`| ${f.replace(".jsonl", "")} | ${a.map((c) => `${mw(c.enc_ms)} | ${mw(c.total_ms)}`).join(" | ")} | ${Math.round(a[2].rtf_x_realtime)} | ${a.every((c) => c.text_stable) ? "same" : "UNSTABLE"} |`);
    load.push(`| ${f.replace(".jsonl", "")} | ${(l.load_ms.encoder / 1000).toFixed(1)} s | ${a[0].first_run_ms.total} ms (${a[0].first_run_ms.enc}) | ${l.gpu_mib_after_load} / ${a[2].gpu_mib_peak} MiB | ${a[2].rss_mib_peak} MiB | |`);
  } else if (f.endsWith(".json")) {
    const r = JSON.parse(readFileSync(`${dir}/${f}`, "utf8"));
    const p = r.page, name = f.replace(".json", "");
    if (r.error || !p || p.clips.length < 3 || !p.clips[2].totalMs) { failed.push(`${name}: ${String(r.error).split("\n")[0]}${p?.clips?.length ? ` (after ${p.clips.length} clip(s))` : ""}`); if (!p?.clips?.[0]?.totalMs) continue; }
    const c = p.clips;
    const cell = (i: number) => (c[i]?.totalMs ? `${mw(c[i].encMs)} | ${mw(c[i].totalMs)}` : " | ");
    timing.push(`| ${name} | ${cell(0)} | ${cell(1)} | ${cell(2)} | ${c[2]?.xRealTime ? Math.round(c[2].xRealTime) : ""} | ${c.every((x: any) => x.matchesNative && x.textStable !== false) ? "same" : c.map((x: any) => (x.matchesNative ? "same" : "differs")).join(", ")} |`);
    const m = r.memory;
    load.push(`| ${name} | ${(p.session.ms.encoder / 1000).toFixed(1)} s (fetch ${(p.fetch.totalMs / 1000).toFixed(1)} s) | ${Math.round(c[0].firstRunMs.total)} ms (${Math.round(c[0].firstRunMs.enc)}) | ${m.gpuProcessGpuMibAfterLoad} / ${m.gpuProcessGpuMibPeak} MiB | ${m.rendererRssMibPeak} MiB (after load ${m.rendererRssMibAfterLoad}) | ${m.gpuProcessRssMibPeak} MiB |`);
  }
}
console.log("| Row | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s | Transcripts vs native fp32 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
console.log(timing.join("\n"));
console.log("\n| Row | Encoder session creation | First 7 s transcribe (encoder part) | GPU memory after load / peak | Renderer (or process) RSS peak | Chrome GPU process RSS peak |\n| --- | ---: | ---: | ---: | ---: | ---: |");
console.log(load.join("\n"));
if (failed.length) console.log("\nFailed or incomplete:\n" + failed.map((f) => `- ${f}`).join("\n"));
