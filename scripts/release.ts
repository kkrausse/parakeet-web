// A release = the built package as the whole tree of a commit on the `release` branch, tagged v<version>.
// Apps install that tag straight from the (private) repository; nothing is published to a registry:
//   bun add git+ssh://git@github.com/kkrausse/parakeet-web.git#v0.1.0
// The tree is exactly what `bun pm pack` puts in the tarball (package.json, README.md, bin/, dist/ with index.js,
// the .d.ts files, licenses/ and assets/ with both WASM builds), with the package.json reduced to what a consumer needs.
//
// usage: bun scripts/release.ts [--push] [--retag]
//   Bump "version" in lib/package.json first. Builds lib/ (the WASM builds must exist: bun scripts/runtime.ts build),
//   commits the result on `release` on top of the previous release, tags it. --push sends the branch and the tag
//   to origin. An existing tag is refused unless --retag (which moves it; only for a release nobody installed).
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildLib } from "../lib/build";
import { commitDir, git, ROOT } from "./branch";

const push = Bun.argv.includes("--push"), retag = Bun.argv.includes("--retag");
const version: string = JSON.parse(readFileSync(join(ROOT, "lib/package.json"), "utf8")).version;
const tag = `v${version}`;
if (git(["rev-parse", "-q", "--verify", `refs/tags/${tag}`], {}, true) && !retag) throw new Error(`tag ${tag} exists: bump "version" in lib/package.json (or --retag)`);
if (git(["status", "--porcelain", "--", "lib", "patches", "runtime.json"])) throw new Error("uncommitted changes under lib/, patches/ or runtime.json: commit first, so the release names a commit of main");

const { tarball } = await buildLib();
if (!tarball) throw new Error("no tarball");
const stage = join(ROOT, "cache", "release");
rmSync(stage, { recursive: true, force: true });
mkdirSync(stage, { recursive: true });
const tar = Bun.spawnSync(["tar", "-xzf", tarball, "-C", stage, "--strip-components=1"], { stderr: "inherit" });
if (tar.exitCode !== 0) throw new Error("tar failed");
const pkgFile = join(stage, "package.json"), pkg = JSON.parse(readFileSync(pkgFile, "utf8"));
delete pkg.devDependencies; delete pkg.scripts; delete pkg.files; // nothing to build or to leave out in the installed tree
writeFileSync(pkgFile, JSON.stringify(pkg, null, 2) + "\n");

const source = git(["rev-parse", "HEAD"]), pin = JSON.parse(readFileSync(join(ROOT, "runtime.json"), "utf8"));
const commit = commitDir(stage, "release", `parakeet-web ${version}\n\nBuilt from main ${source}.\nRuntime: ${pin.upstream} at ${pin.base} + ${pin.patches} patches (tree ${pin.tree}).`, true);
git(["tag", ...(retag ? ["-f"] : []), "-a", tag, commit, "-m", `parakeet-web ${version}`]);
console.log(`release ${commit} tagged ${tag}\n${git(["ls-tree", "-r", "--name-only", commit])}`);
if (push) { git(["push", "origin", "release"]); git(["push", ...(retag ? ["-f"] : []), "origin", tag]); console.log("pushed release and " + tag); }
else console.log(`not pushed: git push origin release ${tag}`);
