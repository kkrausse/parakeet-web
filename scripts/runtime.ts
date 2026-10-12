// The runtime: upstream transcribe.cpp at the commit in ../runtime.json plus the patches in ../patches, and the
// two Emscripten builds of it that lib/build.ts copies into the package. Nothing of it is in git but the patches.
//
//   bun scripts/runtime.ts source [--dir DIR]             clone upstream at the pinned commit into DIR, `git am` the patches,
//                                                         check the result against the pinned tree hash
//   bun scripts/runtime.ts build [--dir DIR] [--out OUT] [jspi] [asyncify] [prof]
//                                                         emcmake + make into OUT/build-web, OUT/build-web-asyncify
//                                                         (and OUT/build-web-prof when asked); default: jspi asyncify
//   bun scripts/runtime.ts export [--dir DIR]             after committing in DIR: rewrite ../patches from its commits
//                                                         on top of the pinned base and update the tree hash
//   bun scripts/runtime.ts verify [--dir DIR]             does DIR hold exactly base + patches?
//
// DIR defaults to vendor/transcribe.cpp, OUT to vendor/ (both git-ignored). The Emscripten SDK and the emdawnwebgpu
// package are not fetched: they are read from $PK_CACHE/emsdk and $PK_CACHE/emdawn/emdawnwebgpu_pkg
// (PK_CACHE defaults to ~/devfs/cache/parakeet-ggml-webgpu).
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

interface Pin { upstream: string; base: string; tree: string; patches: number }

const root = join(import.meta.dir, "..");
const pinFile = join(root, "runtime.json"), patchDir = join(root, "patches");
const pin: Pin = JSON.parse(readFileSync(pinFile, "utf8"));
const cache = process.env.PK_CACHE ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu");

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const dir = resolve(flag("--dir") ?? join(root, "vendor/transcribe.cpp"));
const out = resolve(flag("--out") ?? join(root, "vendor"));
const [command, ...rest] = args;

function run(cmd: string[], cwd?: string, quiet = false): string {
  const p = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: quiet ? "pipe" : "inherit" });
  if (p.exitCode !== 0) throw new Error(`failed (${p.exitCode}): ${cmd.join(" ")}${quiet ? "\n" + p.stderr.toString() : ""}`);
  return p.stdout.toString().trim();
}
const git = (...a: string[]) => run(["git", "-C", dir, ...a]);
const patchFiles = () => readdirSync(patchDir).filter((f) => f.endsWith(".patch")).sort();
const treeOf = () => git("rev-parse", "HEAD^{tree}");

function verify(): boolean {
  const tree = treeOf(), dirty = git("status", "--porcelain");
  const ok = tree === pin.tree && !dirty;
  console.log(`${dir}\n  tree ${tree}${tree === pin.tree ? " = runtime.json" : ` != runtime.json ${pin.tree}`}${dirty ? "\n  uncommitted changes:\n" + dirty : ""}`);
  return ok;
}

function source() {
  if (existsSync(join(dir, ".git"))) {
    if (verify()) return console.log("already base + patches; nothing to do");
    throw new Error(`${dir} exists and is not base + patches. Commit and run \`export\`, or remove the directory.`);
  }
  mkdirSync(dir, { recursive: true });
  git("init", "-q", "-b", "parakeet-web");
  git("remote", "add", "origin", pin.upstream);
  git("fetch", "-q", "--no-tags", "origin", pin.base); // one commit's history; no branches, no tags
  git("checkout", "-q", "-B", "parakeet-web", pin.base);
  const files = patchFiles();
  if (files.length !== pin.patches) throw new Error(`${files.length} patches in ${patchDir}, runtime.json says ${pin.patches}`);
  // Committer = author and committer date = author date, so the same patches give the same commit hashes everywhere.
  run(["git", "-C", dir, "-c", "user.name=parakeet-web", "-c", "user.email=parakeet-web@localhost", "am", "-q", "--committer-date-is-author-date", ...files.map((f) => join(patchDir, f))]);
  console.log(`${pin.upstream} at ${pin.base.slice(0, 12)} + ${files.length} patches -> ${git("rev-parse", "--short", "HEAD")}`);
  if (!verify()) throw new Error("the patched tree is not the one runtime.json names");
}

function exportPatches() {
  if (git("status", "--porcelain")) throw new Error(`${dir} has uncommitted changes; commit them first`);
  if (git("merge-base", pin.base, "HEAD") !== pin.base) throw new Error(`HEAD of ${dir} is not on top of the pinned base ${pin.base}`);
  for (const f of patchFiles()) rmSync(join(patchDir, f));
  // --zero-commit, --no-signature: the files depend on the commits' content only, not on who exported them or with which git.
  git("format-patch", "-q", "--zero-commit", "--no-signature", "-o", patchDir, `${pin.base}..HEAD`);
  const next: Pin = { ...pin, tree: treeOf(), patches: patchFiles().length };
  writeFileSync(pinFile, JSON.stringify(next, null, 2) + "\n");
  console.log(`${next.patches} patches in ${patchDir}; tree ${next.tree}`);
}

/** Build dir under OUT, and what distinguishes it. The flags are the ones the measured builds were made with. */
const BUILDS: Record<string, [string, string[]]> = {
  jspi: ["build-web", ["-DGGML_WEBGPU_JSPI=ON"]],
  asyncify: ["build-web-asyncify", ["-DGGML_WEBGPU_JSPI=OFF"]],
  prof: ["build-web-prof", ["-DGGML_WEBGPU_JSPI=ON", "-DGGML_WEBGPU_CPU_PROFILE=ON"]],
};

function build(which: string[]) {
  if (!verify()) console.warn("warning: building a tree that is not base + patches");
  const emsdk = join(cache, "emsdk/emsdk_env.sh"), emdawn = join(cache, "emdawn/emdawnwebgpu_pkg");
  for (const need of [emsdk, emdawn]) if (!existsSync(need)) throw new Error(`missing ${need} (Emscripten SDK and emdawnwebgpu are expected under PK_CACHE=${cache})`);
  // The toolchain's own licences (its runtime and the WebGPU bindings are linked into the WASM): kept beside the
  // builds, where lib/build.ts picks them up for the LICENSES.txt that ships with the binaries.
  mkdirSync(join(out, "licenses"), { recursive: true });
  for (const [from, to] of [[join(cache, "emsdk/upstream/emscripten/LICENSE"), "Emscripten"], [join(emdawn, "webgpu_cpp/LICENSE"), "Dawn (emdawnwebgpu)"]])
    if (existsSync(from)) writeFileSync(join(out, "licenses", to), readFileSync(from)); else console.warn(`no ${from}`);
  for (const name of which.length ? which : ["jspi", "asyncify"]) {
    const b = BUILDS[name];
    if (!b) throw new Error(`unknown build ${name}: ${Object.keys(BUILDS).join(" ")}`);
    const bdir = join(out, b[0]);
    const cmake = ["emcmake", "cmake", "-S", dir, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_WEBGPU=ON", ...b[1],
      `-DEMDAWNWEBGPU_DIR=${emdawn}`, "-DTRANSCRIBE_BUILD_EXAMPLES=ON", "-DTRANSCRIBE_BUILD_TESTS=OFF", "-DTRANSCRIBE_BUILD_TOOLS=OFF",
      // Source paths in assert messages read "transcribe.cpp/...", not this machine's directory: the binaries are published.
      `-DCMAKE_C_FLAGS=-ffile-prefix-map=${dir}/=transcribe.cpp/`, `-DCMAKE_CXX_FLAGS=-ffile-prefix-map=${dir}/=transcribe.cpp/`,
      "-DBUILD_SHARED_LIBS=OFF", "-DPK_WEB_MEMORY_FLAGS=-sALLOW_MEMORY_GROWTH=1;-sINITIAL_MEMORY=32MB;-sMAXIMUM_MEMORY=4GB"];
    const quote = (s: string) => `'${s.replaceAll("'", `'\\''`)}'`;
    const jobs = process.env.PK_JOBS ?? "8";
    mkdirSync(bdir, { recursive: true });
    const log = join(bdir, "build.log");
    console.log(`${name}: ${bdir} (log: ${log})`);
    const sh = `source ${quote(emsdk)} >/dev/null 2>&1 && ${cmake.map(quote).join(" ")} > ${quote(log)} 2>&1 && nice make -C ${quote(bdir)} -j${jobs} pk-web >> ${quote(log)} 2>&1`;
    const p = Bun.spawnSync(["bash", "-c", sh], { cwd: bdir, stdout: "inherit", stderr: "inherit" }) /* cwd: emsdk_env leaves a.out.* where it runs */;
    if (p.exitCode !== 0) { console.error(run(["bash", "-c", `grep -n -E 'error|Error' -A4 ${quote(log)} | head -40`])); throw new Error(`${name} build failed, see ${log}`); }
    for (const f of ["pk-web.js", "pk-web.wasm"]) console.log(`  ${join(bdir, "bin", f)}  ${Bun.file(join(bdir, "bin", f)).size} bytes`);
  }
}

if (command === "source") source();
else if (command === "export") exportPatches();
else if (command === "verify") process.exit(verify() ? 0 : 1);
else if (command === "build") build(rest);
else { console.error("usage: bun scripts/runtime.ts source|build|export|verify [--dir DIR] [--out OUT] [jspi asyncify prof]"); process.exit(2); }
