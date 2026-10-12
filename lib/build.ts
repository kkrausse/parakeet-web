// Builds the package into dist/ and packs it: dist/index.js + .d.ts (the API), dist/assets/ (what an app serves:
// worker.js, worklet.js and the two Emscripten builds, copied from the runtime's build dirs; prebuilt WASM is
// not in git), then parakeet-web-<version>.tgz beside this file (`bun add <path to it>` installs it;
// scripts/release.ts puts the same files on the `release` branch, which is what apps install from).
// usage: bun build.ts [--no-pack]
// env: PK_WASM_BUILDS = the directory holding build-web/ and build-web-asyncify/ (default ../vendor, where
//      `bun scripts/runtime.ts build` puts them)
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const here = import.meta.dir;
/** Emscripten builds shipped: [build dir under the cache, file name suffix]. The asyncify build is linked as pk-web too; it and the .wasm it asks for are renamed. */
export const WASM_BUILDS: [string, string][] = [["build-web", ""], ["build-web-asyncify", "-asyncify"]];

/** Where the Emscripten build dirs are. */
export const wasmBuilds = () => process.env.PK_WASM_BUILDS ?? join(here, "..", "vendor");

export async function copyWasm(cache: string, dir: string, suffix: string, to: string) {
  const js = join(cache, dir, "bin/pk-web.js"), wasm = join(cache, dir, "bin/pk-web.wasm");
  if (!existsSync(js) || !existsSync(wasm)) return false;
  await Bun.write(join(to, `pk-web${suffix}.js`), (await Bun.file(js).text()).replaceAll("pk-web.wasm", `pk-web${suffix}.wasm`));
  copyFileSync(wasm, join(to, `pk-web${suffix}.wasm`));
  return true;
}

export async function buildLib(o: { pack?: boolean } = {}) {
  const cache = wasmBuilds();
  const dist = join(here, "dist"), assets = join(dist, "assets");
  rmSync(dist, { recursive: true, force: true });
  mkdirSync(assets, { recursive: true });
  const bundle = async (entry: string, outdir: string) => {
    const out = await Bun.build({ entrypoints: [join(here, "src", entry)], outdir, target: "browser", format: "esm", minify: false });
    if (!out.success) throw new AggregateError(out.logs, `build of ${entry} failed`);
  };
  await bundle("index.ts", dist);
  await bundle("worker.ts", assets);
  await bundle("worklet.ts", assets);
  for (const [dir, suffix] of WASM_BUILDS) if (!(await copyWasm(cache, dir, suffix, assets))) throw new Error(`no WASM build in ${join(cache, dir, "bin")} (run "bun scripts/runtime.ts source" and "bun scripts/runtime.ts build" in the repository root, or set PK_WASM_BUILDS)`);
  // The licences of what the WASM is compiled from, as upstream ships them, in one file that travels with the binaries.
  const src = join(cache, "transcribe.cpp");
  let notices = "The WebAssembly files in this folder (pk-web*.wasm, pk-web*.js) are compiled from transcribe.cpp and ggml, modified.\nModels are not part of this folder; NVIDIA's Parakeet models are licensed CC BY 4.0.\n";
  for (const [from, title] of [["LICENSE", "transcribe.cpp"], ["ggml/LICENSE", "ggml"], ["THIRD-PARTY-LICENSES.md", "transcribe.cpp: third-party licences"]]) {
    if (!existsSync(join(src, from))) throw new Error(`no ${join(src, from)}: the runtime source is expected beside the build dirs (bun scripts/runtime.ts source)`);
    notices += `\n${"=".repeat(78)}\n${title}\n${"=".repeat(78)}\n\n${await Bun.file(join(src, from)).text()}`;
  }
  const extra = join(cache, "licenses"); // the toolchain's (Emscripten runtime, Dawn's WebGPU bindings), put there by scripts/runtime.ts build
  for (const f of existsSync(extra) ? readdirSync(extra).sort() : []) notices += `\n${"=".repeat(78)}\n${f}\n${"=".repeat(78)}\n\n${await Bun.file(join(extra, f)).text()}`;
  await Bun.write(join(assets, "LICENSES.txt"), notices);
  // Declarations only for what index.ts reaches; the worker and the worklet are assets, not API.
  const tsc = Bun.spawnSync(["bun", "x", "tsc", "-p", "tsconfig.build.json"], { cwd: here, stdout: "inherit", stderr: "inherit" });
  if (tsc.exitCode !== 0) throw new Error("tsc failed (run `bun install` in lib/ first)");
  let tarball: string | null = null;
  if (o.pack !== false) {
    for (const f of readdirSync(here)) if (/^parakeet-web-.*\.tgz$/.test(f)) rmSync(join(here, f));
    const pack = Bun.spawnSync(["bun", "pm", "pack", "--destination", here, "--quiet"], { cwd: here, stdout: "pipe", stderr: "inherit" });
    if (pack.exitCode !== 0) throw new Error("bun pm pack failed");
    tarball = readdirSync(here).find((f) => /^parakeet-web-.*\.tgz$/.test(f)) ?? null;
    if (tarball) tarball = join(here, tarball);
  }
  console.log(`built ${dist}: index.js, ${readdirSync(dist).filter((f) => f.endsWith(".d.ts")).length} .d.ts, assets ${readdirSync(assets).join(" ")}${tarball ? `\npacked ${tarball}` : ""}`);
  return { dist, assets, tarball };
}

if (import.meta.main) await buildLib({ pack: !Bun.argv.includes("--no-pack") });
