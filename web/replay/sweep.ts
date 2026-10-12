// Level x noise sweep of the old endpointer (replay/old.ts) with and without its noise-floor creep, stand-in model as in gate.ts:
// which of its faults costs what at which level. Columns: % of speech never given to a pass, reference words missed,
// utterances dropped with text on screen, highest noise floor.   usage: bun replay/sweep.ts
import { readFileSync } from "node:fs";
import { applyCondition, coverage, measure, replay, type Model } from "./lib";
import { CACHE } from "../env";
const clean = new Float32Array(readFileSync(CACHE + "/audio/replay/iv90.f32").buffer.slice(0));
const ref = JSON.parse(readFileSync(import.meta.dir + "/../../results/replay/ref-iv90-s4.json", "utf8"));
const SR = 16000;
const oracle: Model = async (_p, req) => { const a = req.startSample / SR, b = req.endSample / SR; const words = ref.words.filter((w: any) => (w.s + w.e) / 2 >= a && w.e <= b).map((w: any) => ({ ...w, s: w.s - a, e: w.e - a })); return { text: words.map((w: any) => w.w).join(" "), words, ms: 60 }; };
console.log("gain  noise  drift soft |  old lost% missed dropTxt maxfloor | nocreep lost% missed");
for (const gainDb of [0, -12, -18, -24]) for (const snr of [30, 20, 12]) for (const shape of [0, 1]) {
  const speechDb = -23.5 + gainDb, noiseDb = speechDb - snr;
  const c = { gainDb, noiseDb, ...(shape ? { driftDb: 6, driftPeriodS: 23, softEndDb: 10 } : {}) };
  const pcm = applyCondition(clean, c);
  const row: string[] = [];
  for (const floorUp of [0.05, 0]) {
    const log = await replay(pcm, oracle, { engine: "old", capS: 12, floorUp, latencyMs: 60 });
    const m = measure(log, ref), cov = coverage(log, clean);
    row.push(`${String(cov.lostPct).padStart(6)} ${String(m.missed).padStart(4)} ${log.dropped.filter((d) => d.hadText).length} ${log.stats.maxNoiseFloor.toFixed(4)}`);
  }
  console.log(`${String(gainDb).padStart(4)} ${noiseDb.toFixed(0).padStart(5)} (snr ${snr}) ${shape ? "drift+soft" : "flat      "} | ${row.join(" | ")}`);
}
