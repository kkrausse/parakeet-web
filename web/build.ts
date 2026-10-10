// Builds the static site into dist/: page, worker, the WASM module(s) from the transcribe.cpp build
// dirs, test clips, and hard links to the GGUF files (so dist/ costs no disk and rsync sees files).
// usage: bun build.ts            env: PK_CACHE (default ~/devfs/cache/parakeet-ggml-webgpu)
import { copyFileSync, existsSync, linkSync, mkdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";

const here = import.meta.dir;
const dist = join(here, "dist");
const cache = process.env.PK_CACHE ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu");
const f16 = join(process.env.HOME!, "devfs/repos/kkrausse/random/.claude/worktrees/parakeet-webgpu-bench/parakeet-webgpu-bench/cache/gguf/parakeet-tdt-0.6b-v2-F16.gguf");
// key, label, path, family. Order = order on the page; the first one present is the page default.
const candidates: [string, string, string, string][] = [
  ["s8", "110M, Q8_0 (8-bit weights)", join(cache, "gguf/parakeet-tdt_ctc-110m-Q8_0.gguf"), "110m"],
  ["s4", "110M, Q4_0 (4-bit weights)", join(cache, "gguf/parakeet-tdt_ctc-110m-Q4_0.gguf"), "110m"],
  ["q4", "0.6b v2, Q4_0 (4-bit weights)", join(cache, "gguf/parakeet-tdt-0.6b-v2-Q4_0.gguf"), "0.6b"],
  ["q8", "0.6b v2, Q8_0 (8-bit weights)", join(cache, "gguf/parakeet-tdt-0.6b-v2-Q8_0.gguf"), "0.6b"],
  ["f16", "0.6b v2, F16", existsSync(join(cache, "gguf/parakeet-tdt-0.6b-v2-F16.gguf")) ? join(cache, "gguf/parakeet-tdt-0.6b-v2-F16.gguf") : f16, "0.6b"],
];
// Q4_K_M is left out: Q4_K blocks have no direct mul_mat kernel yet (2x slower encoder). PK_EXTRA_MODELS adds it back.
for (const extra of (process.env.PK_EXTRA_MODELS ?? "").split(",").filter(Boolean)) { // key=label=path[=family]
  const [k, l, p, fam] = extra.split("=");
  candidates.push([k, l, p, fam ?? "0.6b"]);
}
const models: Record<string, { label: string; file: string; mb: number; family: string }> = {};
for (const [k, label, path, family] of candidates) if (existsSync(path)) models[k] = { label, file: path.split("/").pop()!, mb: Math.round(statSync(path).size / 2 ** 20), family };

// Long clips: expected text = the native build's chunked transcript (scripts/lt.sh NAME=<model>-<clip>-chunk).
const LONG = ["l2", "l5", "l10"];
const expectedLong: Record<string, Record<string, string>> = {};
for (const k of Object.keys(models)) for (const c of LONG) {
  const f = join(cache, `out/lt/${k}-${c}-chunk.txt`);
  if (existsSync(f)) (expectedLong[k] ??= {})[c] = await Bun.file(f).text();
}

rmSync(dist, { recursive: true, force: true });
const out = await Bun.build({
  entrypoints: [join(here, "src/main.ts"), join(here, "src/worker.ts"), join(here, "src/live.ts"), join(here, "src/live-worklet.ts")], outdir: dist, target: "browser", format: "esm", minify: false,
  // DIAG_URL: where both pages POST their step trail (scripts/diag-collector.ts on diesel2, tailnet https). PK_DIAG_URL= (empty) builds without it.
  define: { MODELS: JSON.stringify(models), EXPECTED_LONG: JSON.stringify(expectedLong), DIAG_URL: JSON.stringify(process.env.PK_DIAG_URL ?? "https://diesel2.guineafowl-truck.ts.net:9445/log") },
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
place(join(here, "live.html"), "live.html");
for (const [dir, suffix] of [["build-web", ""], ["build-web-asyncify", "-asyncify"], ["build-web-prof", "-prof"]]) {
  // the asyncify build is linked as pk-web too; it is renamed here, and so is the .wasm it asks for
  const js = join(cache, dir, "bin/pk-web.js");
  if (!existsSync(js)) continue;
  await Bun.write(join(dist, `pk-web${suffix}.js`), (await Bun.file(js).text()).replaceAll("pk-web.wasm", `pk-web${suffix}.wasm`));
  place(join(cache, dir, "bin/pk-web.wasm"), `pk-web${suffix}.wasm`);
}
for (const c of ["a07", "a14", "a56", ...LONG]) place(join(cache, `audio/${c}.f32`), `audio/${c}.f32`);
for (const [k, , path] of candidates) if (models[k]) place(path, `models/${models[k].file}`, true);

// dist-live/: the live page alone as its own site (short link). It has no models/ of its own: when published it
// reads them from the benchmark deployment on the same origin (../parakeet-ggml-browser/models/).
const live = join(here, "dist-live");
rmSync(live, { recursive: true, force: true });
mkdirSync(live);
copyFileSync(join(dist, "live.html"), join(live, "index.html"));
for (const f of ["live.js", "live-worklet.js", "worker.js", "pk-web.js", "pk-web.wasm", "pk-web-asyncify.js", "pk-web-asyncify.wasm"]) if (existsSync(join(dist, f))) copyFileSync(join(dist, f), join(live, f));
console.log(`built ${dist}: models ${Object.keys(models).join(", ")}`);
