// Prints the README tables from results/browser/*.json.   usage: bun scripts/table.ts [name ...]
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
const dir = join(import.meta.dir, "..", "results", "browser");
const names = Bun.argv.slice(2).length ? Bun.argv.slice(2) : readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)).sort();
const rows = names.map((n) => ({ n, d: JSON.parse(readFileSync(join(dir, `${n}.json`), "utf8")) }));
const c = (d: any, clip: string, k: string) => { const x = d.page?.clips?.find((y: any) => y.clip === clip); return x?.[k] ? `${Math.round(x[k].median)} / ${Math.round(x[k].worst)}` : "-"; };
console.log("| Row | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const { n, d } of rows) console.log(`| ${n} | ${["a07", "a14", "a56"].map((k) => `${c(d, k, "encMs")} | ${c(d, k, "totalMs")}`).join(" | ")} | ${d.page?.clips?.find((y: any) => y.clip === "a56")?.xRealTime ?? "-"} |`);
console.log("\n| Row | Model load | Fetch (localhost) | First 7 s transcribe (encoder part) | Page start to first transcript | GPU memory after load / peak | Renderer RSS after load / peak | GPU process RSS peak | WASM heap (in use) | decode 7 s | text 7/14/56 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
for (const { n, d } of rows) {
  const p = d.page ?? {}, m = d.memory ?? {}, f = p.clips?.[0], last = p.clips?.at(-1);
  console.log(`| ${n} | ${p.load?.loadMs ?? "-"} ms | ${p.load?.fetchMs ?? "-"} ms | ${f ? `${Math.round(f.firstRunMs.total)} ms (${Math.round(f.firstRunMs.enc)})` : "-"} | ${f?.startToFirstTranscriptMs ?? "-"} ms | ${m.gpuProcessGpuMibAfterLoad} / ${m.gpuProcessGpuMibPeak} MiB | ${m.rendererRssMibAfterLoad} / ${m.rendererRssMibPeak} MiB | ${m.gpuProcessRssMibPeak} MiB | ${last?.wasmHeapMb ?? "-"} (${last?.wasmHeapUsedMb ?? "-"}) MB | ${f?.decMs?.median ?? "-"} ms | ${(p.clips ?? []).map((x: any) => (x.matchesNative ? "same" : "differs")).join(" / ")} |`);
}
