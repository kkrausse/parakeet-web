// The live page's endpointer and pass scheduling as they were before src/stream.ts (commit 8c7907c,
// src/live.ts lines 259-430), lifted out of the page so a replay can run them: same arithmetic, same
// constants, the DOM and the diagnostics removed. Kept only to reproduce what the phone showed.
import type { Transcribe, Word } from "../../lib/src/stream";

const SR = 16000, FRAME = 320;
export interface OldEvents {
  onCommit?(c: { text: string; words: Word[]; reason: string; atSample: number; startSample: number; endSample: number }): void;
  onTentative?(text: string, words: Word[], atSample: number): void;
  /** An utterance the endpointer threw away without a final pass (under minSpeechMs of flagged frames). */
  onDropped?(d: { startSample: number; endSample: number; speechMs: number; hadText: boolean; atSample: number }): void;
  onFrame?(f: { atSample: number; rms: number; noiseFloor: number; speech: boolean; hasSpeech: boolean }): void;
}
/** `floorUp`: how fast a non-speech frame pulls the noise floor up (0.05 as shipped; 0 is the ablation that shows what the creep costs). */
export function createOldLive(transcribe: Transcribe, events: OldEvents, o: { capS: number; now: () => number; floorUp?: number }) {
  const floorUp = o.floorUp ?? 0.05;
  const P = { gapMs: 250, hangMs: 700, preRollMs: 300, tailMs: 300, capS: o.capS, cutLookbackS: 10, minSpeechMs: 120, onsetFrames: 2, absRms: 0.004, noiseRatio: 3, backlogS: 120 };
  const MAX_FRAMES = Math.ceil(((P.capS + 2) * SR) / FRAME);
  const buf = new Float32Array(MAX_FRAMES * FRAME), rms = new Float32Array(MAX_FRAMES), flag = new Uint8Array(MAX_FRAMES);
  let nFrames = 0, bufAbs = 0, captured = 0;
  let hasSpeech = false, lastSpeech = -1, onsetRun = 0, noiseFloor = 0.001;
  let epoch = 0;
  let provText = "", provTextEnd = 0, provEndAbs = 0, lastPassStart = -Infinity;
  interface Final { pcm: Float32Array | null; startAbs: number; endAbs: number; reason: string; text?: string; words?: Word[] }
  const finals: Final[] = [];
  let provWords: Word[] = [];
  let busy = false, stopped = false, waiters: (() => void)[] = [];
  const stats = { passes: 0, commits: 0, dropped: 0, droppedWithText: 0, stale: 0, maxNoiseFloor: 0 };

  function shiftFrames(n: number) {
    buf.copyWithin(0, n * FRAME, nFrames * FRAME); rms.copyWithin(0, n, nFrames); flag.copyWithin(0, n, nFrames);
    nFrames -= n; bufAbs += n * FRAME; lastSpeech -= n;
  }
  function cut(n: number, reason: string) {
    let speech = 0;
    for (let i = 0; i < n; i++) speech += flag[i];
    if (speech * 20 >= P.minSpeechMs) {
      const seen = !reason.startsWith("cap") && provText !== "" && provTextEnd >= bufAbs + n * FRAME;
      finals.push({ pcm: seen ? null : buf.slice(0, n * FRAME), startAbs: bufAbs, endAbs: bufAbs + n * FRAME, reason, text: seen ? provText : undefined, words: seen ? provWords : undefined });
    } else {
      stats.dropped++; if (provText) stats.droppedWithText++;
      events.onDropped?.({ startSample: bufAbs, endSample: bufAbs + n * FRAME, speechMs: speech * 20, hadText: provText !== "", atSample: captured });
    }
    shiftFrames(n);
    provText = ""; provTextEnd = 0; provWords = []; epoch++;
    events.onTentative?.("", [], captured);
    let left = 0; lastSpeech = -1;
    for (let i = 0; i < nFrames; i++) if (flag[i]) left++, lastSpeech = i;
    hasSpeech = reason.startsWith("cap") && left > 0;
    if (!hasSpeech) { lastSpeech = -1; onsetRun = 0; }
  }
  function capCut() {
    const lo = Math.max(10, nFrames - Math.round(P.cutLookbackS * 50)), hi = nFrames - 25;
    let runStart = -1, bestStart = -1, bestLen = 0;
    for (let i = lo; i <= hi; i++) {
      if (i < hi && !flag[i]) { if (runStart < 0) runStart = i; continue; }
      if (runStart >= 0 && i - runStart >= bestLen) bestLen = i - runStart, bestStart = runStart;
      runStart = -1;
    }
    if (bestLen >= 8) return cut(bestStart + (bestLen >> 1), "cap");
    let best = lo, bestE = Infinity, e = 0;
    for (let i = lo; i < lo + 10; i++) e += rms[i] * rms[i];
    for (let i = lo; ; i++) {
      if (e < bestE) bestE = e, best = i;
      if (i + 10 >= hi) break;
      e += rms[i + 10] * rms[i + 10] - rms[i] * rms[i];
    }
    const q = Math.sqrt(bestE / 10);
    if (q > Math.max(P.absRms, noiseFloor * P.noiseRatio)) noiseFloor = q / 2;
    cut(best + 5, "cap-quiet");
  }
  function push(pcm: Float32Array) { // whole 20 ms frames, as the worklet delivers (640 samples)
    captured += pcm.length;
    buf.set(pcm, nFrames * FRAME);
    for (let o = 0; o + FRAME <= pcm.length; o += FRAME) {
      let s = 0;
      for (let i = o; i < o + FRAME; i++) s += pcm[i] * pcm[i];
      const r = Math.sqrt(s / FRAME), sp = r > Math.max(P.absRms, noiseFloor * P.noiseRatio);
      if (!sp || r < noiseFloor) noiseFloor += (r - noiseFloor) * (r < noiseFloor ? 0.3 : floorUp);
      if (noiseFloor > stats.maxNoiseFloor) stats.maxNoiseFloor = noiseFloor;
      rms[nFrames] = r; flag[nFrames] = sp ? 1 : 0;
      onsetRun = sp ? onsetRun + 1 : 0;
      if (sp && (hasSpeech || onsetRun >= P.onsetFrames)) hasSpeech = true, lastSpeech = nFrames;
      nFrames++;
      events.onFrame?.({ atSample: captured - pcm.length + o + FRAME, rms: r, noiseFloor, speech: sp, hasSpeech });
    }
    if (!hasSpeech) {
      const keep = Math.round(P.preRollMs / 20);
      if (nFrames > keep) shiftFrames(nFrames - keep);
    } else if ((nFrames - 1 - lastSpeech) * 20 >= P.hangMs) {
      cut(Math.min(nFrames, lastSpeech + 1 + Math.round(P.tailMs / 20)), "pause");
      const keep = Math.round(P.preRollMs / 20);
      if (nFrames > keep) shiftFrames(nFrames - keep);
    } else if (nFrames * FRAME >= P.capS * SR) capCut();
    void pump();
  }
  const abs = (ws: Word[] | undefined, startAbs: number): Word[] => (ws ?? []).map((w) => ({ ...w, s: w.s + startAbs / SR, e: w.e + startAbs / SR }));
  function commit(text: string, words: Word[], seg: Final) {
    stats.commits++;
    events.onCommit?.({ text: text.trim(), words, reason: seg.reason, atSample: captured, startSample: seg.startAbs, endSample: seg.endAbs });
  }
  async function pump(): Promise<void> {
    if (busy) return;
    const seg = finals[0];
    if (seg?.text !== undefined) { finals.shift(); commit(seg.text, seg.words ?? [], seg); settle(); return pump(); }
    let pcm: Float32Array, startAbs: number, endAbs: number;
    const jobEpoch = epoch;
    if (seg) pcm = seg.pcm!, startAbs = seg.startAbs, endAbs = seg.endAbs;
    else {
      startAbs = bufAbs; endAbs = bufAbs + nFrames * FRAME;
      if (!hasSpeech || endAbs - provEndAbs < SR / 10) return settle();
      if (o.now() - lastPassStart < P.gapMs) return;
      pcm = buf.slice(0, nFrames * FRAME);
      provEndAbs = endAbs;
    }
    busy = true; lastPassStart = o.now();
    const r = await transcribe(pcm, { kind: seg ? "final" : "live", startSample: startAbs, endSample: endAbs });
    busy = false; stats.passes++;
    if (seg) { if (finals[0] === seg) { finals.shift(); commit(r.text, abs(r.words, startAbs), seg); } }
    else if (jobEpoch === epoch) { provText = String(r.text).trim(); provTextEnd = endAbs; provWords = abs(r.words, startAbs); events.onTentative?.(provText, provWords, captured); }
    else stats.stale++;
    settle();
    return pump();
  }
  function settle() { if (stopped && !busy && !finals.length) { const w = waiters; waiters = []; for (const f of w) f(); } }
  function stop(): Promise<void> {
    stopped = true;
    if (hasSpeech) cut(nFrames, "stop");
    nFrames = 0; hasSpeech = false;
    const p = new Promise<void>((res) => waiters.push(res));
    void pump().then(settle);
    return p;
  }
  return { push, stop, stats: () => ({ ...stats, noiseFloor }), params: P };
}
