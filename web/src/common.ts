// Shared by the benchmark page (main.ts) and the live page (live.ts): the persisted step trail and small
// formatting helpers. The runtime (worker, adapter probe, backend environment) is the library in ../../lib.
export { formatGpuStats as gpuLine } from "../../lib/src/index";

export interface Step { t: number; name: string; ms?: number; detail?: string }
/** `sid`: the diagnostics session id of the page load that made it (src/diag.ts). */
export interface RunLog<C = unknown> { id: string; sid?: string; started: string; config: C; ua: string; steps: Step[]; done: boolean; error?: string }

export const r1 = (x: number) => Math.round(x * 10) / 10;
export function fmtMs(ms: number) { return ms >= 10000 ? `${(ms / 1000).toFixed(1)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${r1(ms)} ms`; }
export function stepLine(s: Step) { return `${(s.t / 1000).toFixed(1).padStart(6)}s  ${s.name}${s.ms !== undefined ? `: ${fmtMs(s.ms)}` : ""}${s.detail ? `  (${s.detail})` : ""}`; }
export function esc(s: string) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!); }

/** Run logs kept in localStorage under `key` (the last `keep`), so a killed tab leaves its trail. */
export function historyStore<C>(key: string, keep = 8) {
  const load = (): RunLog<C>[] => { try { return JSON.parse(localStorage.getItem(key) ?? "[]"); } catch { return []; } };
  return {
    load,
    save(runs: RunLog<C>[]) { try { localStorage.setItem(key, JSON.stringify(runs.slice(-keep))); } catch { /* private mode or full */ } },
    clear() { try { localStorage.removeItem(key); } catch { /* nothing */ } },
  };
}

/** Words of `a` and `b` outside their longest common subsequence (insertions + deletions). */
export function wordDiff(a: string, b: string) {
  const x = a.split(/\s+/).filter(Boolean), y = b.split(/\s+/).filter(Boolean);
  let prev = new Uint16Array(y.length + 1);
  for (let i = 1; i <= x.length; i++) {
    const cur = new Uint16Array(y.length + 1);
    for (let j = 1; j <= y.length; j++) cur[j] = x[i - 1] === y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    prev = cur;
  }
  return { differing: x.length + y.length - 2 * prev[y.length], words: x.length };
}

/** The same numbers cut down for a diagnostics event (one per pass). */
export function gpuDiag(g: any) {
  if (!g) return null;
  return { w: g.weightsMb, work: g.workingMb, n: g.buffers, made: g.created, gone: g.destroyed, miss: g.poolMisses, bg: g.bindGroups, enc: g.encoders, wr: g.writes, lay: g.layouts, pipe: g.pipelines,
    ...(g.js ? { jsObj: g.js.objects, jsGone: g.js.collected, jsBuf: g.js.buffers - g.js.buffersDestroyed, jsBufMb: g.js.bufferLiveMb, jsBg: g.js.bindGroups, jsBgGone: g.js.bindGroupsCollected } : {}) };
}
