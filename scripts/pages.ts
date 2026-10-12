// The public demo: the live dictation page alone, as a static site for GitHub Pages (served under /parakeet-web/).
//   bun scripts/pages.ts build      -> dist-pages/
//   bun scripts/pages.ts publish    -> dist-pages/ becomes the single commit of the `gh-pages` branch, pushed to origin
// What makes it fit to be public, and is checked at the end of `build`:
//   - no diagnostics collector (DIAG_URL is empty: the page posts nothing anywhere), no sibling deployment (SHARED empty);
//   - one model, beside the page (models/, same origin: no CORS, HEAD with Content-Length): 110m Q4_0, 81 MB;
//   - no benchmark page, no test audio, no replay or API test pages;
//   - only relative URLs, so it works under any path;
//   - none of the strings in FORBIDDEN in any shipped text file.
// The repository stays private; a Pages site is public whatever the repository is.
import { copyFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildLib } from "../lib/build";
import { CACHE } from "../web/env";
import { commitDir, git, ROOT } from "./branch";

const out = join(ROOT, "dist-pages");
const MODEL = { key: "s4", label: "110M, Q4_0 (4-bit weights)", file: "parakeet-tdt_ctc-110m-Q4_0.gguf", family: "110m" };
const FORBIDDEN = ["ts.net", "guineafowl", "diesel2", "kkrausse97", "raspberrypi", "/home/", "devfs", "127.0.0.1"];

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]));
}

async function build() {
  const gguf = join(CACHE, "gguf", MODEL.file);
  if (!existsSync(gguf)) throw new Error(`missing ${gguf}`);
  const lib = await buildLib({ pack: false });
  rmSync(out, { recursive: true, force: true });
  mkdirSync(join(out, "models"), { recursive: true });
  const models = { [MODEL.key]: { label: MODEL.label, file: MODEL.file, mb: Math.round(statSync(gguf).size / 2 ** 20), family: MODEL.family } };
  const b = await Bun.build({
    entrypoints: [join(ROOT, "web/src/live.ts")], outdir: out, target: "browser", format: "esm", minify: true,
    define: { MODELS: JSON.stringify(models), SHARED: '""', DIAG_URL: '""' },
  });
  if (!b.success) throw new AggregateError(b.logs, "build failed");
  copyFileSync(join(ROOT, "web/live.html"), join(out, "index.html"));
  cpSync(lib.assets, join(out, "pk"), { recursive: true });
  copyFileSync(gguf, join(out, "models", MODEL.file));
  writeFileSync(join(out, ".nojekyll"), ""); // serve the files as they are
  const hits: string[] = [];
  for (const f of walk(out)) {
    if (/\.(gguf|wasm)$/.test(f)) continue;
    const text = readFileSync(f, "utf8");
    for (const s of FORBIDDEN) if (text.includes(s)) hits.push(`${f}: ${s}`);
    for (const m of text.matchAll(/(?:src|href)="(\/[^"/][^"]*)"/g)) hits.push(`${f}: root-relative URL ${m[1]}`);
  }
  // The WASM files hold C++ source paths only if the build embedded them.
  for (const f of walk(out).filter((f) => f.endsWith(".wasm"))) { const bin = readFileSync(f).toString("latin1"); for (const s of FORBIDDEN) if (bin.includes(s)) hits.push(`${f}: ${s}`); }
  if (hits.length) throw new Error("not fit to publish:\n  " + hits.join("\n  "));
  const files = walk(out);
  console.log(`built ${out}: ${files.length} files, ${Math.round(files.reduce((n, f) => n + statSync(f).size, 0) / 2 ** 20)} MB; checked for ${FORBIDDEN.join(", ")}: none`);
}

function publish() {
  if (!existsSync(join(out, "index.html"))) throw new Error("no dist-pages/: run `bun scripts/pages.ts build` first");
  const commit = commitDir(out, "gh-pages", `Demo page built from main ${git(["rev-parse", "--short", "HEAD"])}`, false);
  git(["push", "-f", "origin", "gh-pages"]);
  console.log(`gh-pages ${commit} pushed`);
}

const command = Bun.argv[2];
if (command === "build") await build();
else if (command === "publish") publish();
else { console.error("usage: bun scripts/pages.ts build|publish"); process.exit(2); }
