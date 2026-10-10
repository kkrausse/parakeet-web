// Builds the static site into dist/: page, worker, the WASM module(s) from the transcribe.cpp build
// dirs, test clips, and hard links to the GGUF files (so dist/ costs no disk and rsync sees files).
// usage: bun build.ts            env: PK_CACHE (default ~/devfs/cache/parakeet-ggml-webgpu)
import { copyFileSync, existsSync, linkSync, mkdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";

const here = import.meta.dir;
const dist = join(here, "dist");
const cache = process.env.PK_CACHE ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu");
const f16 = join(process.env.HOME!, "devfs/repos/kkrausse/random/.claude/worktrees/parakeet-webgpu-bench/parakeet-webgpu-bench/cache/gguf/parakeet-tdt-0.6b-v2-F16.gguf");
const candidates: [string, string, string][] = [
  ["q4", "Q4_0 (4-bit weights)", join(cache, "gguf/parakeet-tdt-0.6b-v2-Q4_0.gguf")],
  ["q4km", "Q4_K_M", join(cache, "gguf/parakeet-tdt-0.6b-v2-Q4_K_M.gguf")],
  ["q8", "Q8_0 (8-bit weights)", join(cache, "gguf/parakeet-tdt-0.6b-v2-Q8_0.gguf")],
  ["f16", "F16", existsSync(join(cache, "gguf/parakeet-tdt-0.6b-v2-F16.gguf")) ? join(cache, "gguf/parakeet-tdt-0.6b-v2-F16.gguf") : f16],
];
for (const extra of (process.env.PK_EXTRA_MODELS ?? "").split(",").filter(Boolean)) { // key=label=path
  const [k, l, p] = extra.split("=");
  candidates.push([k, l, p]);
}
const models: Record<string, { label: string; file: string; mb: number }> = {};
for (const [k, label, path] of candidates) if (existsSync(path)) models[k] = { label, file: path.split("/").pop()!, mb: Math.round(statSync(path).size / 2 ** 20) };

rmSync(dist, { recursive: true, force: true });
const out = await Bun.build({
  entrypoints: [join(here, "src/main.ts"), join(here, "src/worker.ts")], outdir: dist, target: "browser", format: "esm", minify: false,
  define: { MODELS: JSON.stringify(models) },
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
for (const [dir, suffix] of [["build-web", ""], ["build-web-asyncify", "-asyncify"]]) {
  // the asyncify build is linked as pk-web too; it is renamed here, and so is the .wasm it asks for
  const js = join(cache, dir, "bin/pk-web.js");
  if (!existsSync(js)) { console.warn(`no ${dir} build`); continue; }
  await Bun.write(join(dist, `pk-web${suffix}.js`), (await Bun.file(js).text()).replaceAll("pk-web.wasm", `pk-web${suffix}.wasm`));
  place(join(cache, dir, "bin/pk-web.wasm"), `pk-web${suffix}.wasm`);
}
for (const c of ["a07", "a14", "a56"]) place(join(cache, `audio/${c}.f32`), `audio/${c}.f32`);
for (const [k, , path] of candidates) if (models[k]) place(path, `models/${models[k].file}`, true);
console.log(`built ${dist}: models ${Object.keys(models).join(", ")}`);
