// Replay harness, shared by the bun script (replay/gate.ts, no model) and the Chrome page (src/replay-page.ts,
// the real model): phone-like conditions made from a clean recording, a runner that feeds an engine on a virtual
// clock, and the measurements. Nothing here touches the DOM.
import { createStream, type StreamParams, type StreamStats, type Transcribe, type Transcription, type Word } from "../src/stream";
import { createOldLive } from "./old";

export const SR = 16000;
const FRAME = 320;

// ---------- conditions ----------
export interface Condition {
  /** Level change of the whole recording, dB. */
  gainDb?: number;
  /** Slow level drift: +-driftDb as a sine of period driftPeriodS (the speaker moving about). */
  driftDb?: number; driftPeriodS?: number;
  /** Each phrase fades by this many dB over its second half (soft sentence endings). */
  softEndDb?: number;
  /** Synthetic room noise added after the above, RMS in dBFS: pink noise, 120 Hz hum, slow swell, a thud now and then. */
  noiseDb?: number;
  seed?: number;
}
export const CONDITIONS: Record<string, Condition> = {
  clean: {},
  low: { gainDb: -26 },
  phone: { gainDb: -24, driftDb: 6, driftPeriodS: 23, softEndDb: 10, noiseDb: -62 },
  noisy: { gainDb: -24, driftDb: 6, driftPeriodS: 23, softEndDb: 10, noiseDb: -54 },
  far: { gainDb: -34, driftDb: 8, driftPeriodS: 31, softEndDb: 12, noiseDb: -68 },
};
const db = (x: number) => 10 ** (x / 20);
function lcg(seed: number) { let s = seed >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); }
export function frameRms(pcm: Float32Array): Float32Array {
  const n = Math.floor(pcm.length / FRAME), out = new Float32Array(n);
  for (let f = 0; f < n; f++) { let s = 0; for (let i = f * FRAME; i < (f + 1) * FRAME; i++) s += pcm[i] * pcm[i]; out[f] = Math.sqrt(s / FRAME); }
  return out;
}
/** Frames of the clean recording that are plainly speech: within 30 dB of its loud frames (95th percentile). */
export function speechFrames(clean: Float32Array): Uint8Array {
  const r = frameRms(clean), sorted = Float32Array.from(r).sort(), ref = sorted[Math.floor(sorted.length * 0.95)], thr = ref * db(-30);
  return Uint8Array.from(r, (x) => (x > thr ? 1 : 0));
}
export function applyCondition(clean: Float32Array, c: Condition): Float32Array {
  const out = new Float32Array(clean.length), g0 = db(c.gainDb ?? 0);
  // phrases: runs of speech frames separated by at least 300 ms without one
  const sp = speechFrames(clean), nF = sp.length, fade = new Float32Array(nF).fill(1);
  if (c.softEndDb) {
    let a = -1, last = -1;
    const close = (b: number) => { const mid = (a + b) >> 1; for (let f = mid; f <= b; f++) fade[f] = db((-c.softEndDb! * (f - mid)) / Math.max(1, b - mid)); for (let f = b + 1; f < Math.min(nF, b + 15); f++) fade[f] = db(-c.softEndDb!); };
    for (let f = 0; f < nF; f++) {
      if (!sp[f]) continue;
      if (a >= 0 && f - last > 15) { close(last); a = -1; }
      if (a < 0) a = f;
      last = f;
    }
    if (a >= 0) close(last);
  }
  for (let i = 0; i < clean.length; i++) {
    const f = Math.min(nF - 1, (i / FRAME) | 0);
    const drift = c.driftDb ? db(c.driftDb * Math.sin((2 * Math.PI * i) / SR / (c.driftPeriodS ?? 20))) : 1;
    out[i] = clean[i] * g0 * drift * fade[f];
  }
  if (c.noiseDb !== undefined) {
    const rnd = lcg(c.seed ?? 7), noise = new Float32Array(clean.length);
    let b0 = 0, b1 = 0, b2 = 0, e = 0, thud = 0, thudPhase = 0;
    for (let i = 0; i < noise.length; i++) {
      const w = rnd() * 2 - 1;
      b0 = 0.99765 * b0 + w * 0.099046; b1 = 0.963 * b1 + w * 0.2965164; b2 = 0.57 * b2 + w * 1.0526913; // Paul Kellet's pink filter
      const swell = 1 + 0.25 * Math.sin((2 * Math.PI * i) / SR / 7.3) + 0.15 * Math.sin((2 * Math.PI * i) / SR / 1.9);
      if (thud < 1e-4 && rnd() < 1 / (SR * 6)) thud = 6, thudPhase = 0; // every 6 s or so, a knock: 90 Hz, 80 ms decay
      const knock = thud * Math.sin((2 * Math.PI * 90 * thudPhase++) / SR); thud *= 0.9992;
      noise[i] = (b0 + b1 + b2 + w * 0.1848) * swell + 0.6 * Math.sin((2 * Math.PI * 120 * i) / SR) + knock;
      e += noise[i] * noise[i];
    }
    const k = db(c.noiseDb) / Math.sqrt(e / noise.length);
    for (let i = 0; i < out.length; i++) out[i] += noise[i] * k;
  }
  return out;
}

// ---------- running an engine over a recording ----------
export interface Shown { atSample: number; words: Word[] }
export interface ReplayLog {
  engine: "old" | "new"; seconds: number;
  commits: { atSample: number; reason: string; text: string; words: Word[] }[];
  tentative: Shown[]; // every tentative text that was on screen, with the audio position at which it appeared
  dropped: { startSample: number; endSample: number; speechMs: number; hadText: boolean }[]; // old engine only
  utterances: { startSample: number; endSample: number; reason: string; opened?: string }[];
  passes: { kind: string; startSample: number; endSample: number; ms: number; atSample: number }[];
  gate: { atSample: number; open: boolean; by: string }[];
  frames?: { floor: Float32Array; speech: Uint8Array }; // old engine: its noise floor and speech flag per 20 ms frame
  series: { t: number; bufferS: number; lagS: number; floor: number; thr: number }[]; // every 2 s of audio
  stats: any; committed: string; passMs: number[]; skipped: { fromSample: number; toSample: number }[];
}
export interface ReplayOptions {
  engine: "old" | "new"; capS: number; params?: Partial<StreamParams>;
  /** Old engine only: its noise-floor creep rate (0.05 as shipped, 0 to switch the creep off). */
  floorUp?: number;
  /** Virtual time one pass takes: a number of ms, or "measured" for the `ms` the model function reports. */
  latencyMs?: number | "measured" | ((audioS: number) => number);
  blockN?: number;
  /** Called every 2 s of audio (the page samples memory here). */
  onProgress?(tS: number): void | Promise<void>;
}
export type Model = (pcm: Float32Array, req: { kind: string; startSample: number; endSample: number }) => Promise<Transcription & { ms?: number }>;

/** Feeds `pcm` to an engine in 40 ms blocks on a virtual clock: a pass takes `latencyMs` of it, however long the model really took. */
export async function replay(pcm: Float32Array, model: Model, o: ReplayOptions): Promise<ReplayLog> {
  const blockN = o.blockN ?? 640;
  let nowV = 0;
  let flight: { p: Promise<Transcription & { ms?: number }>; startV: number; audioS: number; resolve: (r: Transcription) => void; reject: (e: unknown) => void } | null = null;
  const transcribe: Transcribe = (x, req) => new Promise((resolve, reject) => { flight = { p: model(x, req), startV: nowV, audioS: x.length / SR, resolve, reject }; });
  const log: ReplayLog = { engine: o.engine, seconds: pcm.length / SR, commits: [], tentative: [], dropped: [], utterances: [], passes: [], gate: [], series: [], stats: null, committed: "", passMs: [], skipped: [] };
  const tick = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); };
  const onCommit = (c: { atSample: number; reason: string; text: string; words: Word[] }) => { log.commits.push({ atSample: c.atSample, reason: c.reason, text: c.text, words: c.words }); if (c.text) log.committed += (log.committed ? " " : "") + c.text; };
  const onTentative = (_t: string, words: Word[], atSample: number) => { if (words.length) log.tentative.push({ atSample, words }); };
  let floors: Float32Array | null = null, flags: Uint8Array | null = null;
  const eng = o.engine === "new"
    ? createStream(transcribe, { onCommit, onTentative, onUtteranceEnd: (u) => log.utterances.push(u), onGate: (g) => log.gate.push({ atSample: g.atSample, open: g.open, by: g.by }),
        onPass: (p) => { log.passes.push({ kind: p.kind, startSample: p.startSample, endSample: p.endSample, ms: p.ms, atSample: p.atSample }); }, onSkip: (s) => log.skipped.push(s) }, { capS: o.capS, ...o.params, now: () => nowV })
    : (() => {
        floors = new Float32Array(Math.ceil(pcm.length / FRAME)); flags = new Uint8Array(floors.length);
        let f = 0;
        return createOldLive(transcribe, { onCommit: (c) => { onCommit(c); log.utterances.push({ startSample: c.startSample, endSample: c.endSample, reason: c.reason }); }, onTentative, onDropped: (d) => log.dropped.push(d),
          onFrame: (x) => { floors![f] = x.noiseFloor; flags![f++] = x.speech ? 1 : 0; } }, { capS: o.capS, now: () => nowV, floorUp: o.floorUp });
      })();
  const latency = (r: { ms?: number }, audioS: number) => (typeof o.latencyMs === "function" ? o.latencyMs(audioS) : o.latencyMs === "measured" || o.latencyMs === undefined ? r.ms ?? 60 : o.latencyMs);
  /** Deliver the pass in flight if its (virtual) time is up by `untilV`. */
  const deliver = async (untilV: number) => {
    while (flight) {
      const f: NonNullable<typeof flight> = flight;
      let r: Transcription & { ms?: number };
      try { r = await f.p; } catch (e) { flight = null; f.reject(e); await tick(); return; }
      const lat = latency(r, f.audioS);
      if (f.startV + lat > untilV) return;
      flight = null; nowV = Math.max(nowV, f.startV + lat);
      log.passMs.push(lat);
      f.resolve(r);
      await tick();
    }
  };
  let nextSeries = 0;
  for (let at = 0; at + blockN <= pcm.length; at += blockN) {
    const tV = ((at + blockN) / SR) * 1000;
    await deliver(tV);
    nowV = tV;
    eng.push(pcm.subarray(at, at + blockN));
    await tick();
    if ((at + blockN) / SR >= nextSeries) {
      nextSeries += 2;
      const s: any = eng.stats();
      log.series.push({ t: (at + blockN) / SR, bufferS: s.bufferS ?? 0, lagS: s.lagS ?? 0, floor: s.noiseFloor, thr: s.threshold ?? 0 });
      await o.onProgress?.((at + blockN) / SR);
    }
  }
  let done = false;
  const stopped = eng.stop().then(() => { done = true; });
  await tick();
  for (let guard = 0; !done && guard < 10000; guard++) { await deliver(Infinity); await tick(); }
  await stopped;
  log.stats = eng.stats();
  if (floors && flags) log.frames = { floor: floors, speech: flags };
  return log;
}

// ---------- measurements ----------
export const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}' ]+/gu, " ").split(/\s+/).filter(Boolean);
/** Levenshtein alignment of two word lists. `hypOps[j]`: "=" matched, "s" substituted, "i" inserted (not in ref). */
export function align(ref: string[], hyp: string[]) {
  const n = ref.length, m = hyp.length, w = m + 1, d = new Int32Array((n + 1) * w);
  for (let j = 0; j <= m; j++) d[j] = j;
  for (let i = 1; i <= n; i++) {
    d[i * w] = i;
    for (let j = 1; j <= m; j++) d[i * w + j] = Math.min(d[(i - 1) * w + j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1), d[(i - 1) * w + j] + 1, d[i * w + j - 1] + 1);
  }
  const hypOps: string[] = new Array(m), refOps: string[] = new Array(n);
  let i = n, j = m, sub = 0, del = 0, ins = 0;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i * w + j] === d[(i - 1) * w + j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)) { const eq = ref[i - 1] === hyp[j - 1]; if (!eq) sub++; hypOps[--j] = eq ? "=" : "s"; refOps[--i] = eq ? "=" : "s"; }
    else if (j > 0 && d[i * w + j] === d[i * w + j - 1] + 1) { ins++; hypOps[--j] = "i"; }
    else { del++; refOps[--i] = "d"; }
  }
  return { sub, del, ins, errors: sub + del + ins, refWords: n, hypWords: m, hypOps, refOps };
}
const pct = (xs: number[], p: number) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * p))] : NaN; };
const r2 = (x: number) => Math.round(x * 100) / 100;
export interface Measured {
  refWords: number; committedWords: number; wer: number; errors: number;
  /** Committed words the reference does not have at that place (substituted or extra). */
  contradicted: number;
  /** Reference words missing from the committed text. */
  missed: number;
  /** Longest run of reference words that never had any text (tentative or committed) over them, in seconds and words; and how many such words in all. */
  longestUnshownS: number; longestUnshownWords: number; unshownWords: number;
  /** The same for committed text only. */
  longestUncommittedS: number;
  /** Seconds from the end of a word to its commit. */
  commitLatency: { median: number; p95: number; max: number };
  /** Seconds from the end of a reference word to the first text (tentative or committed) over it. */
  showLatency: { median: number; p95: number; max: number };
  spans: string[]; // the first differences, with one word of context
}
/** `ref`: the whole-file pass over the same audio (words with times). */
export function measure(log: ReplayLog, ref: Transcription): Measured {
  const refN = ref.words.map((w) => norm(w.w).join("")).filter(Boolean), refW = ref.words.filter((w) => norm(w.w).join(""));
  const hyp = norm(log.committed);
  const a = align(refN, hyp);
  const spans: string[] = [];
  { // readable differences
    let i = 0, j = 0, ctx = "", delW: string[] = [], insW: string[] = [];
    const flush = () => { if (delW.length || insW.length) { if (spans.length < 40) spans.push(`${ctx} [${delW.join(" ")} -> ${insW.join(" ")}]`); delW = []; insW = []; } };
    while (i < refN.length || j < hyp.length) {
      if (i < refN.length && a.refOps[i] === "d") { delW.push(refN[i++]); continue; }
      if (j < hyp.length && a.hypOps[j] === "i") { insW.push(hyp[j++]); continue; }
      if (a.refOps[i] === "s") { delW.push(refN[i++]); insW.push(hyp[j++]); continue; }
      flush(); ctx = refN[i]; i++; j++;
    }
    flush();
  }
  // coverage in time: which reference words ever had text over them
  const cover = (spansIn: { s: number; e: number; at: number }[]) => {
    const sorted = [...spansIn].sort((x, y) => x.s - y.s);
    const first = new Float64Array(refW.length).fill(Infinity);
    let lo = 0;
    for (let k = 0; k < refW.length; k++) {
      const w = refW[k];
      while (lo < sorted.length && sorted[lo].e < w.s - 2) lo++;
      for (let q = lo; q < sorted.length && sorted[q].s <= w.e + 0.3; q++) if (sorted[q].e >= w.s - 0.3 && sorted[q].at < first[k]) first[k] = sorted[q].at;
    }
    let longS = 0, longW = 0, runStart = -1, count = 0;
    for (let k = 0; k <= refW.length; k++) {
      if (k < refW.length && first[k] === Infinity) { count++; if (runStart < 0) runStart = k; continue; }
      if (runStart >= 0) { const s = refW[k - 1].e - refW[runStart].s; if (s > longS) longS = s, longW = k - runStart; runStart = -1; }
    }
    return { first, longS, longW, count };
  };
  const committedSpans = log.commits.flatMap((c) => c.words.map((w) => ({ s: w.s, e: w.e, at: c.atSample / SR })));
  const shownSpans = [...committedSpans, ...log.tentative.flatMap((t) => t.words.map((w) => ({ s: w.s, e: w.e, at: t.atSample / SR })))];
  const shown = cover(shownSpans), com = cover(committedSpans);
  const showLat: number[] = [];
  for (let k = 0; k < refW.length; k++) if (shown.first[k] !== Infinity) showLat.push(shown.first[k] - refW[k].e);
  const comLat = committedSpans.map((c) => c.at - c.e);
  const st = (xs: number[]) => ({ median: r2(pct(xs, 0.5)), p95: r2(pct(xs, 0.95)), max: r2(Math.max(...xs, NaN)) });
  return { refWords: a.refWords, committedWords: a.hypWords, wer: r2((100 * a.errors) / Math.max(1, a.refWords)), errors: a.errors, contradicted: a.sub + a.ins, missed: a.del,
    longestUnshownS: r2(shown.longS), longestUnshownWords: shown.longW, unshownWords: shown.count, longestUncommittedS: r2(com.longS), commitLatency: st(comLat), showLatency: st(showLat), spans };
}
export type { StreamStats, Transcription, Word };

/** How much plain speech (speechFrames of the clean recording) no pass was ever given: the gate's doing, whatever the model is. */
export function coverage(log: ReplayLog, clean: Float32Array) {
  const sp = speechFrames(clean), seen = new Uint8Array(sp.length);
  const ranges = log.engine === "old" ? log.utterances : log.passes;
  for (const r of ranges) for (let f = Math.max(0, Math.floor(r.startSample / FRAME)); f < Math.min(sp.length, Math.ceil(r.endSample / FRAME)); f++) seen[f] = 1;
  let speech = 0, lost = 0, longest = 0, longestAt = 0, runStart = -1, lastLost = -1;
  for (let f = 0; f < sp.length; f++) {
    if (!sp[f]) continue;
    speech++;
    if (!seen[f]) { lost++; if (runStart < 0) runStart = f; lastLost = f; if (lastLost - runStart + 1 > longest) longest = lastLost - runStart + 1, longestAt = runStart; }
    else runStart = -1;
  }
  return { speechS: r2(speech / 50), lostS: r2(lost / 50), lostPct: r2((100 * lost) / Math.max(1, speech)), longestLostS: r2(longest / 50), longestLostAtS: r2(longestAt / 50),
    segments: log.utterances.length, dropped: log.dropped.length, droppedWithText: log.dropped.filter((d) => d.hadText).length };
}
