// Builds the static site into dist/: page, onnxruntime-web runtime files, test clips, and hard links
// to the model files in ../cache (so dist/ costs no disk and rsync sees ordinary files).
import { copyFileSync, existsSync, linkSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";

const here = import.meta.dir;
const dist = join(here, "dist");
const cache = join(here, "..", "cache");
const ortDist = join(here, "node_modules/onnxruntime-web/dist");
const ortVersion = (await Bun.file(join(here, "node_modules/onnxruntime-web/package.json")).json()).version as string;

rmSync(dist, { recursive: true, force: true });
const out = await Bun.build({
  entrypoints: [join(here, "src/main.ts")], outdir: dist, target: "browser", format: "esm", minify: false,
  define: { ORT_VERSION: JSON.stringify(ortVersion) }, external: ["onnxruntime-web"],
});
if (!out.success) throw new AggregateError(out.logs, "build failed");

const place = (from: string, to: string, link = false) => {
  if (!existsSync(from)) return console.warn(`missing, skipped: ${from}`);
  const dest = join(dist, to);
  mkdirSync(dirname(dest), { recursive: true });
  if (link) try { return linkSync(from, dest); } catch { /* other filesystem: copy */ }
  copyFileSync(from, dest);
};
place(join(here, "index.html"), "index.html");
place(join(here, "sw.js"), "sw.js");
for (const f of ["ort.webgpu.min.mjs", "ort.all.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) place(join(ortDist, f), `ort/${f}`);
for (const c of ["a07", "a14", "a56"]) place(join(cache, `audio/${c}.f32`), `audio/${c}.f32`);
const models: [string, string][] = [
  ["onnx-tdt-v2/vocab.txt", "vocab.txt"], ["onnx-tdt-v2/melfb-257x128.f32", "melfb-257x128.f32"],
  ["onnx-tdt-v2/decoder_joint-model.onnx", "decoder_joint-model.onnx"],
  ["onnx-tdt-v2/encoder-model.onnx", "fp32/encoder-model.onnx"], ["onnx-tdt-v2/encoder-model.onnx.data", "fp32/encoder-model.onnx.data"],
  ["onnx-tdt-v2-fp16/encoder-model.onnx", "fp16/encoder-model.onnx"], ["onnx-tdt-v2-fp16/encoder-model.onnx.data", "fp16/encoder-model.onnx.data"],
  ["onnx-tdt-v2-int8/encoder-model.int8.onnx", "int8/encoder-model.int8.onnx"], ["onnx-tdt-v2-int8/decoder_joint-model.int8.onnx", "int8/decoder_joint-model.int8.onnx"],
  // parakeet-tdt_ctc-110m (OpenVoiceOS export; fp16 encoder is scripts/onnx-fp16.py --inline)
  ...["vocab.txt", "melfb-257x80.f32", "decoder_joint-model.onnx", "decoder_joint-model.onnx_data", "decoder_joint-model.int8.onnx",
    "encoder-model.onnx", "encoder-model.onnx_data", "encoder-model.int8.onnx"].map((f): [string, string] => [`onnx-tdt-ctc-110m/${f}`, `110m/${f}`]),
  ["onnx-tdt-ctc-110m-fp16/encoder-model.onnx", "110m/encoder-model.fp16.onnx"],
];
for (const [from, to] of models) place(join(cache, from), `models/${to}`, true);
console.log(`built ${dist} (onnxruntime-web ${ortVersion})`);
