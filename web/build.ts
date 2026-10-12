// One command for everything: the library (../lib: dist/, its assets and the installable tarball), then the static
// site in dist/: the pages (which import the library's source), the library's assets under pk/ as an app would
// serve them, test clips, and hard links to the GGUF files (so dist/ costs no disk and rsync sees files).
// usage: bun build.ts            env: PK_CACHE (default ~/devfs/cache/parakeet-ggml-webgpu)
import { copyFileSync, cpSync, existsSync, linkSync, mkdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { buildLib, copyWasm } from "../lib/build";

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

const lib = await buildLib();
rmSync(dist, { recursive: true, force: true });
const out = await Bun.build({
  entrypoints: [join(here, "src/main.ts"), join(here, "src/live.ts"), join(here, "src/replay-page.ts")], outdir: dist, target: "browser", format: "esm", minify: false,
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
place(join(here, "replay.html"), "replay.html"); // the replay harness (drive-replay.ts); not part of dist-live
// The library's runtime files, served from pk/ (the pages' assetsUrl), plus the CPU-profile build the benchmark page can ask for (variant=prof).
cpSync(lib.assets, join(dist, "pk"), { recursive: true });
await copyWasm(cache, "build-web-prof", "-prof", join(dist, "pk"));
for (const c of ["a07", "a14", "a56", ...LONG]) place(join(cache, `audio/${c}.f32`), `audio/${c}.f32`);
for (const [k, , path] of candidates) if (models[k]) place(path, `models/${models[k].file}`, true);

// dist-live/: the live page alone as its own site (short link). It has no models/ of its own: when published it
// reads them from the benchmark deployment on the same origin (../parakeet-ggml-browser/models/).
const live = join(here, "dist-live");
rmSync(live, { recursive: true, force: true });
mkdirSync(live);
copyFileSync(join(dist, "live.html"), join(live, "index.html"));
copyFileSync(join(dist, "live.js"), join(live, "live.js"));
cpSync(lib.assets, join(live, "pk"), { recursive: true });
console.log(`built ${dist}: models ${Object.keys(models).join(", ")}`);
