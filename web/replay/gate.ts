// The reproduction and the guard, without a browser or a model: a recording made phone-like (replay/lib.ts
// CONDITIONS) is fed to the endpointer the live page used to have (replay/old.ts) and to the streaming engine
// (src/stream.ts). The model is stood in for by the whole-file pass saved in results/replay/ref-<clip>-<model>.json:
// a pass over any stretch of audio returns the reference words that lie in it (the last one cut short when the
// stretch ends inside it), whatever the level or the noise. So what is measured is only what each engine's gate and
// commit logic do with speech the model could have read: audio never given to a pass, words never committed.
// Exit status 1 when the streaming engine loses words (the guard), 0 otherwise.
// usage: bun replay/gate.ts [--clip iv90] [--model s4] [--cap 12] [--conditions phone,noisy,...] [--verbose]
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { applyCondition, CONDITIONS, coverage, frameRms, measure, replay, speechFrames, type Model, type ReplayOptions } from "./lib";
import type { Transcription, Word } from "../../lib/src/stream";

const args = Bun.argv.slice(2);
const flag = (k: string) => { const i = args.indexOf(k); return i < 0 ? undefined : args.splice(i, 2)[1]; };
const verbose = args.includes("--verbose");
const clipName = flag("--clip") ?? "iv90", model = flag("--model") ?? "s4", cap = Number(flag("--cap") ?? 12);
const conds = (flag("--conditions") ?? Object.keys(CONDITIONS).join(",")).split(",");
const R = process.env.PK_CACHE ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu");
const clipFile = join(R, "audio/replay", clipName + ".f32"), refFile = join(import.meta.dir, "..", "..", "results", "replay", `ref-${clipName}-${model}.json`);
if (!existsSync(clipFile)) { console.error(`missing ${clipFile}: run scripts/replay-clips.sh`); process.exit(2); }
if (!existsSync(refFile)) { console.error(`missing ${refFile}: run bun drive-replay.ts NAME --clip ${clipName} --conditions clean --save-ref`); process.exit(2); }
const clean = new Float32Array(readFileSync(clipFile).buffer.slice(0));
const ref: Transcription = JSON.parse(readFileSync(refFile, "utf8"));
const SR = 16000;

/** The stand-in model: the reference words inside the stretch; a word the stretch ends in comes out cut short. */
const oracle: Model = async (_pcm, req) => {
  const a = req.startSample / SR, b = req.endSample / SR, words: Word[] = [];
  for (const w of ref.words) {
    if ((w.s + w.e) / 2 < a || w.s >= b - 0.08) continue;
    if (w.e > b) words.push({ w: w.w.slice(0, Math.max(1, Math.ceil(w.w.length / 2))), s: w.s - a, e: b - a, p: 0.4 });
    else words.push({ w: w.w, s: w.s - a, e: w.e - a, p: w.p });
  }
  return { text: words.map((w) => w.w).join(" "), words, ms: 60 };
};
const speech = speechFrames(clean);
const dbfs = (x: number) => (20 * Math.log10(Math.max(x, 1e-9))).toFixed(1);
let bad = 0;
console.log(`${clipName}: ${(clean.length / SR).toFixed(0)} s, ${ref.words.length} reference words (${model}), cap ${cap} s; stand-in model, 60 ms a pass`);
for (const c of conds) {
  const pcm = applyCondition(clean, CONDITIONS[c]);
  // levels of this condition: speech frames and the rest
  const r = frameRms(pcm), sp: number[] = [], no: number[] = [];
  for (let f = 0; f < r.length; f++) (speech[f] ? sp : no).push(r[f]);
  const med = (xs: number[]) => [...xs].sort((x, y) => x - y)[xs.length >> 1] ?? 0;
  console.log(`\n${c}: speech frames median ${dbfs(med(sp))} dBFS (rms ${med(sp).toFixed(4)}), other frames median ${dbfs(med(no))} dBFS (rms ${med(no).toFixed(4)})`);
  const runs: [string, ReplayOptions][] = [
    ["old", { engine: "old", capS: cap }],
    ["old, floor creep off", { engine: "old", capS: cap, floorUp: 0 }],
    ["new", { engine: "new", capS: cap }],
    ["new, probe off", { engine: "new", capS: cap, params: { probeEveryS: 0 } }],
  ];
  for (const [label, o] of runs) {
    const log = await replay(pcm, oracle, { ...o, latencyMs: 60 });
    const m = measure(log, ref), cov = coverage(log, clean);
    let extra = "";
    if (log.frames) {
      // the old gate from the inside: its floor while people speak, and how many plain speech frames it called silence
      const fl: number[] = []; let deaf = 0, n = 0;
      for (let f = 0; f < speech.length; f++) if (speech[f]) { n++; fl.push(log.frames.floor[f]); if (!log.frames.speech[f]) deaf++; }
      extra = ` | floor during speech median ${med(fl).toFixed(4)} max ${Math.max(...fl).toFixed(4)} (threshold 3x) | speech frames called silence ${((100 * deaf) / n).toFixed(0)}% | utterances dropped ${log.dropped.length}, ${log.dropped.filter((d) => d.hadText).length} with text on screen`;
    } else extra = ` | opens ${log.stats.gateOpens}, probe hits ${log.stats.gateMisses}, probes ${log.stats.probes}, model holds ${log.stats.modelHolds} | max buffer ${log.stats.maxBufferS.toFixed(1)} s | ${JSON.stringify(log.stats.byReason)}`;
    console.log(`  ${label.padEnd(21)} committed ${String(m.committedWords).padStart(4)}/${m.refWords} words, missed ${String(m.missed).padStart(3)}, wrong ${String(m.contradicted).padStart(2)} | speech never given to a pass ${String(cov.lostPct).padStart(5)}%, longest ${String(cov.longestLostS).padStart(5)} s at ${cov.longestLostAtS} s | longest speech with no text ${m.longestUnshownS} s | commit ${m.commitLatency.median}/${m.commitLatency.p95} s | ${cov.segments} segments${extra}`);
    if (verbose) console.log("     " + m.spans.slice(0, 8).join(" · "));
    if (label === "new" && (m.missed > Math.max(2, 0.02 * m.refWords) || m.contradicted > Math.max(2, 0.02 * m.refWords) || m.longestUnshownS > 2.5 || log.stats.maxBufferS > cap + 1)) { bad++; console.log("     ^ GUARD: the streaming engine lost or invented words here"); }
  }
}
process.exit(bad ? 1 : 0);
