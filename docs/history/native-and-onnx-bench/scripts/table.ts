// Markdown tables from results/*.jsonl. Run: bun scripts/table.ts
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const dir = join(import.meta.dir, "../results");
const read = (name: string) => {
  const path = join(dir, `${name}.jsonl`);
  return existsSync(path) ? readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line)) : [];
};
const rows: [string, string][] = [
  ["ort-webgpu-fp32", "ORT 1.28 WebGPU EP (Dawn/Vulkan), fp32 encoder, decoder CPU"],
  ["ort-webgpu-fp16", "same, fp16 encoder"],
  ["ort-webgpu-fp32-dec-webgpu", "same, fp32, decoder on WebGPU too"],
  ["ort-cpu-fp32", "ORT 1.28 CPU EP, fp32, 12 cores"],
  ["burn-wgpu-fp32", "Burn 0.22 wgpu, WGSL shaders, fp32 (encoder only)"],
  ["burn-vulkan-spirv-fp32", "Burn 0.22 wgpu on Vulkan, SPIR-V shaders, fp32 (encoder only)"],
  ["tcpp-vulkan-tdt-f16", "transcribe.cpp 0.3.1 Vulkan, TDT v2 F16"],
  ["tcpp-vulkan-tdt-f32", "transcribe.cpp 0.3.1 Vulkan, TDT v2 F32"],
  ["tcpp-vulkan-unified-f16", "transcribe.cpp 0.3.1 Vulkan, unified F16 offline (the service's weights)"],
];
const mw = (s?: { median: number; worst: number }) => s ? `${Math.round(s.median)} / ${Math.round(s.worst)}` : "";
const clip = (records: any[], name: string) => records.find(r => r.record === "audio" && r.file.includes(name));

console.log("Warm, median / worst ms over the warm runs. enc = encoder, total = mel + encoder + decode.\n");
console.log("| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time at 56 s |");
console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const [name, label] of rows) {
  const records = read(name);
  if (!records.length) continue;
  const cells = ["a07", "a14", "a56"].flatMap(c => { const r = clip(records, c); return [mw(r?.enc_ms), mw(r?.total_ms)]; });
  console.log(`| ${label} | ${cells.join(" | ")} | ${clip(records, "a56")?.rtf_x_realtime ?? ""} |`);
}

console.log("\n| Runtime | Load (warm disk) | First 7 s transcribe in process | of which encoder | GPU memory after load / peak | RSS peak |");
console.log("| --- | ---: | ---: | ---: | ---: | ---: |");
for (const [name, label] of rows) {
  const records = read(name);
  if (!records.length) continue;
  const load = records.find(r => r.record === "load");
  const first = clip(records, "a07");
  const last = records.at(-1);
  const loadMs = load.load_ms.process_start_to_loaded;
  console.log(`| ${label} | ${(loadMs / 1000).toFixed(2)} s | ${first.first_run_ms.total ?? "n/a"} ms | ${first.first_run_ms.enc} ms | ${load.gpu_mib_after_load} / ${last.gpu_mib_peak} MiB | ${last.rss_mib_peak} MiB |`);
}

console.log("\nProcess start to first transcript of the 7 s clip (ms): empty NVIDIA pipeline cache, then three more processes.\n");
for (const name of ["cold-ort-webgpu-fp32", "cold-ort-webgpu-fp16", "cold-tcpp-vulkan-tdt-f16"]) {
  const records = read(name);
  if (records.length) console.log(`- ${name}: ${records.map(r => `${r.process_start_to_first_transcript_ms} (first run ${r.first_run_ms.total}, encoder ${r.first_run_ms.enc})`).join("; ")}`);
}

const stream = read("tcpp-vulkan-unified-f16-stream");
if (stream.length) {
  console.log("\nService unit of work (transcribe.cpp Vulkan, unified F16, buffered streaming 5.6 s + 560 ms + 560 ms, fed back to back):\n");
  for (const r of stream.filter(r => r.record === "audio"))
    console.log(`- ${r.seconds} s clip: per chunk with full left context median ${r.full_window_chunk_ms.median} ms, worst ${r.full_window_chunk_ms.worst} ms (${r.full_window_chunks_per_run} chunks x ${r.warm_runs} runs); whole clip median ${r.total_ms.median} ms`);
}

const live = read("live-service");
if (live.length) {
  console.log(`\nLive service on :9876 (${live[0].modelId}), timed from the client, ms median / worst:\n`);
  for (const r of live.filter(r => r.record === "audio"))
    console.log(`- ${r.seconds} s clip: per full-window chunk back to back ${mw(r.back_to_back.full_window_chunk_ms)}, in real time ${mw(r.real_time.full_window_chunk_ms)}; whole clip back to back ${mw(r.back_to_back.total_ms)}`);
}

const burn = ["burn-wgpu-fp32", "burn-vulkan-spirv-fp32"].flatMap(name => read(name).filter(r => r.record === "audio").map(r => `${name} ${r.file.slice(-3)}: ${r.max_abs_diff_vs_ort_cpu.toExponential(1)}`));
if (burn.length) console.log(`\nBurn encoder output, largest absolute difference from ORT CPU (values up to ~0.8): ${burn.join("; ")}`);

console.log("\nTranscripts (first run):\n");
for (const [name, label] of rows) {
  const records = read(name);
  for (const c of ["a14", "a56"]) { const r = clip(records, c); if (r?.text) console.log(`- ${label}, ${r.seconds} s${r.text_stable ? "" : " (NOT stable across runs)"}: ${r.text}`); }
}
