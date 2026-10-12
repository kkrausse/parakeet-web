// A directory becomes the whole tree of a commit on a branch, without touching the checkout or its index
// (git plumbing with a throwaway index). Used for the two built-output branches: `release` (the package apps
// install) and `gh-pages` (the public demo). Neither shares history with `main`.
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

export const ROOT = join(import.meta.dir, "..");

export function git(args: string[], env: Record<string, string> = {}, allowFail = false): string {
  const p = Bun.spawnSync(["git", "-C", ROOT, ...args], { stdout: "pipe", stderr: "pipe", env: { ...process.env, ...env } });
  if (p.exitCode !== 0 && !allowFail) throw new Error(`git ${args.join(" ")}: ${p.stderr.toString().trim()}`);
  return p.exitCode === 0 ? p.stdout.toString().trim() : "";
}

/** Commits `dir` as the tree of `branch`. `chain`: on top of the branch's previous commit (else a commit with no parent). Returns the commit. */
export function commitDir(dir: string, branch: string, message: string, chain: boolean): string {
  mkdirSync(join(ROOT, "cache"), { recursive: true });
  const index = join(ROOT, "cache", `index-${branch}`);
  rmSync(index, { force: true });
  const env = { GIT_INDEX_FILE: index, GIT_WORK_TREE: dir };
  git(["add", "-A", "-f", "."], env); // -f: the checkout's ignore rules (dist, *.tgz) do not apply to a built tree
  const tree = git(["write-tree"], env);
  rmSync(index, { force: true });
  const parent = chain ? git(["rev-parse", "-q", "--verify", `refs/heads/${branch}`], {}, true) || git(["rev-parse", "-q", "--verify", `refs/remotes/origin/${branch}`], {}, true) : "";
  if (parent && git(["rev-parse", `${parent}^{tree}`]) === tree) return parent; // nothing changed
  const commit = git(["commit-tree", tree, ...(parent ? ["-p", parent] : []), "-m", message]);
  git(["update-ref", `refs/heads/${branch}`, commit]);
  return commit;
}
