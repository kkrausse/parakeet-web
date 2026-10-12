// Streaming engine for an offline speech model: 16 kHz PCM in, committed and tentative text out.
// Pure: no DOM, no worker, no microphone, no timers and no clock other than the injected now().
// Every position is a count of 16 kHz samples since the stream began (reset() starts again at 0), so
// a replay of the same audio with the same transcribe() results gives the same events.
//
// The model decides what is speech. Recent audio is kept in one buffer and transcribed again and
// again; each pass returns words with times. A word is committed (append-only) once successive
// passes agree on it and enough audio follows it; the buffer then drops all but a little of the
// committed audio, which stays as left context. The energy gate only decides when passes are worth
// running: it never removes audio a pass has put words on, and while it is closed a probe pass runs
// every few seconds over the recent audio, so a gate that misses speech costs seconds, not sentences.

export const SAMPLE_RATE = 16000;
const SR = SAMPLE_RATE;
const FRAME = 320; // the gate works on 20 ms frames
const GRID = 1280; // the encoder's 80 ms frame: the buffer start only moves by multiples of it, so word times keep their grid

/** One word. `s` / `e` are seconds: from the start of the PCM in a Transcription, from the start of the stream in events. */
export interface Word { w: string; s: number; e: number; p?: number }
export interface Transcription { text: string; words: Word[] }
export type PassKind = "live" | "final" | "probe";
export interface PassRequest { kind: PassKind; startSample: number; endSample: number }
/** Runs the model over `pcm` (16 kHz mono, owned by the callee from here on). A rejection stops the engine (onError). */
export type Transcribe = (pcm: Float32Array, req: PassRequest) => Promise<Transcription>;

export interface StreamParams {
  /** Longest audio one pass is given, seconds. The GPU working buffer is sized for it. */
  capS: number;
  /** Least time between the starts of two passes, ms of now(). */
  gapMs: number;
  /** A pass is only run when this much audio arrived since the last one, seconds. */
  minNewS: number;
  /** Commit: a word must have been in every pass for this long (audio time) ... */
  stableS: number;
  /** ... in at least this many passes ... */
  minPasses: number;
  /** ... and have this much audio after its end. */
  holdS: number;
  /** Committed audio kept in front of the buffer as left context, seconds. */
  keepS: number;
  /** Audio with no words in it is given up (as context only) this long after a pass saw it empty, seconds. */
  emptyHoldS: number;
  /** A pass over a full buffer (capS) commits every word that has this much audio after it, agreed or not. */
  forceHoldS: number;
  /** When passes fall this far behind the microphone, words are committed without waiting for agreement. */
  lagS: number;
  /** Audio that may wait behind a full buffer before new audio is skipped, seconds. */
  backlogS: number;
  /** The utterance ends after this long with the gate closed and no new word from the model, seconds. */
  hangS: number;
  /** Audio kept in front of the frame that opened the gate, seconds. */
  preRollS: number;
  /** Gate: a frame is speech when its RMS is over max(gateAbsRms, noise floor * gateRatio). */
  gateRatio: number;
  gateAbsRms: number;
  /** Gate: the noise floor is the lowest 60 ms RMS of the last this many seconds. */
  floorWindowS: number;
  /** Gate: consecutive speech frames that open it. */
  onsetFrames: number;
  /** With the gate closed, a probe pass every probeEveryS over the last probeS (0 turns the probe off). */
  probeEveryS: number;
  probeS: number;
  /** A probe window whose peak is under this is digital silence and is not probed. */
  silencePeak: number;
}
export const DEFAULT_PARAMS: StreamParams = {
  capS: 24, gapMs: 250, minNewS: 0.1, stableS: 0.5, minPasses: 2, holdS: 0.8, keepS: 2, emptyHoldS: 1, forceHoldS: 1, lagS: 3, backlogS: 30,
  hangS: 1.5, preRollS: 0.4, gateRatio: 2.5, gateAbsRms: 0.0004, floorWindowS: 4, onsetFrames: 2, probeEveryS: 2, probeS: 3, silencePeak: 1e-4,
};

/** agree: passes agreed; idle / stop: the utterance ended; cap: full buffer; lag: passes behind the microphone; skip: draining before skipped audio. */
export type CommitReason = "agree" | "idle" | "stop" | "cap" | "lag" | "skip";
export interface CommitEvent { text: string; words: Word[]; reason: CommitReason; atSample: number }
export interface UtteranceEvent { startSample: number; endSample: number; reason: "idle" | "stop" | "skip"; words: number; opened: "gate" | "probe" }
export interface PassEvent { kind: PassKind; startSample: number; endSample: number; ms: number; words: number; committed: number; tentative: number; stale: boolean; atSample: number }
export interface GateEvent { open: boolean; atSample: number; by: "energy" | "probe" | "idle" | "stop" | "skip"; noiseFloor: number; threshold: number }
export interface StreamEvents {
  /** Words appended to the committed text; never revised. `text` is the words joined by spaces. */
  onCommit?(c: CommitEvent): void;
  /** The current guess for what follows the committed text; replaces the previous one ("" clears it). */
  onTentative?(text: string, words: Word[], atSample: number): void;
  onUtteranceEnd?(u: UtteranceEvent): void;
  onPass?(p: PassEvent): void;
  onGate?(g: GateEvent): void;
  /** Audio that was not transcribed because passes are slower than real time. */
  onSkip?(s: { fromSample: number; toSample: number }): void;
  /** transcribe() rejected: the engine stops running passes until reset(). */
  onError?(e: unknown): void;
}
export interface StreamStats {
  capturedS: number; open: boolean; busy: boolean; failed: boolean; skipping: boolean;
  bufferS: number; maxBufferS: number; lagS: number; maxLagS: number;
  noiseFloor: number; threshold: number; lastRms: number; speech: boolean;
  gateOpens: number; gateMisses: number; modelHolds: number; probes: number; probesSilent: number;
  passes: number; livePasses: number; finalPasses: number; stale: number; reusedFinals: number;
  utterances: number; commits: number; wordsCommitted: number; byReason: Record<CommitReason, number>;
  skippedS: number; trims: number; anchorShifts: number;
}
export interface Stream {
  /** A block of 16 kHz mono PCM, any length. Copied. */
  push(pcm: Float32Array): void;
  /** No more audio for now: what is buffered gets its last pass and is committed. Resolves when that is done. */
  stop(): Promise<void>;
  /** Forget everything (buffer, positions, counters, a failure); the next push is sample 0. */
  reset(): void;
  /** True when nothing is running or waiting. */
  idle(): boolean;
  stats(): StreamStats;
  readonly params: StreamParams;
}

interface W { w: string; n: string; s: number; e: number; p: number; since: number; seen: number } // s, e, since: absolute samples
/** Case and punctuation do not count when two passes are compared. */
export const normalizeWord = (w: string) => w.toLowerCase().replace(/[^\p{L}\p{N}']+/gu, "");
const outWord = (x: W): Word => ({ w: x.w, s: x.s / SR, e: x.e / SR, p: x.p });

export function createStream(transcribe: Transcribe, events: StreamEvents = {}, options: Partial<StreamParams> & { now?: () => number } = {}): Stream {
  const { now = () => performance.now(), ...over } = options;
  const P: StreamParams = { ...DEFAULT_PARAMS, ...over };
  const capN = Math.round(P.capS * SR), capacity = capN + Math.round(P.backlogS * SR);
  const closedKeepN = Math.round(Math.max(P.probeS, P.preRollS + 0.2) * SR);
  const buf = new Float32Array(capacity + 4 * FRAME);
  let start = 0, len = 0, captured = 0; // buf[0] is absolute sample `start`
  // gate
  const subN = 25, subs = new Float32Array(Math.max(2, Math.round((P.floorWindowS * 50) / subN))); // minima of 0.5 s stretches
  let subFill = 0, subCur = Infinity, subAt = 0, floor = 0, lastRms = 0, speech = false, run = 0, r1 = 0, r2 = 0, frames = 0;
  const frame = new Float32Array(FRAME);
  let frameFill = 0;
  let lastSpeech = -1; // absolute sample after the last speech frame
  // utterance
  let open = false, openedBy: "gate" | "probe" = "gate", uttStart = 0, uttWords = 0;
  let boundary = 0; // words whose middle lies before this are behind the committed text
  let tail: W[] = [], tent: W[] = []; // last committed words (for anchoring); the uncommitted words of the newest pass
  let hypEnd = 0; // how far the newest pass saw
  let drain: null | "idle" | "stop" | "skip" = null;
  let skipFrom = -1, holding = false;
  let busy = false, failed = false, gen = 0, lastPassStart = -Infinity, lastProbeAt = 0, closedSince = 0;
  let stopWaiters: (() => void)[] = [];
  const zero = (): StreamStats => ({ capturedS: 0, open: false, busy: false, failed: false, skipping: false, bufferS: 0, maxBufferS: 0, lagS: 0, maxLagS: 0, noiseFloor: 0, threshold: 0, lastRms: 0, speech: false,
    gateOpens: 0, gateMisses: 0, modelHolds: 0, probes: 0, probesSilent: 0, passes: 0, livePasses: 0, finalPasses: 0, stale: 0, reusedFinals: 0, utterances: 0, commits: 0, wordsCommitted: 0,
    byReason: { agree: 0, idle: 0, stop: 0, cap: 0, lag: 0, skip: 0 }, skippedS: 0, trims: 0, anchorShifts: 0 });
  let S = zero();

  const threshold = () => Math.max(P.gateAbsRms, floor * P.gateRatio);
  const end = () => start + len;
  function shiftTo(abs: number) { // drop audio before `abs`
    const n = Math.min(len, Math.max(0, abs - start));
    if (n <= 0) return;
    buf.copyWithin(0, n, len); len -= n; start += n;
  }

  // ---------- gate ----------
  function gateFrame(endAbs: number) {
    let s = 0;
    for (let i = 0; i < FRAME; i++) s += frame[i] * frame[i];
    const r = Math.sqrt(s / FRAME);
    // Noise floor by minimum statistics: the lowest 60 ms average of the last floorWindowS. Speech cannot raise it
    // above its own quietest 60 ms, and it falls at once when the room gets quieter.
    const sm = frames >= 2 ? (r + r1 + r2) / 3 : r;
    r2 = r1; r1 = r; frames++;
    if (sm < subCur) subCur = sm;
    if (++subAt >= subN) { subs[subFill++ % subs.length] = subCur; subCur = Infinity; subAt = 0; }
    let f = subCur;
    for (let i = 0; i < Math.min(subFill, subs.length); i++) if (subs[i] < f) f = subs[i];
    floor = f;
    lastRms = r;
    speech = r > threshold();
    run = speech ? run + 1 : 0;
    if (run >= P.onsetFrames) {
      lastSpeech = endAbs;
      if (!open && !drain) openUtterance(Math.max(start, closedSince, endAbs - run * FRAME - Math.round(P.preRollS * SR)), "gate");
    }
  }
  function openUtterance(at: number, by: "gate" | "probe") {
    at = start + Math.floor(Math.max(0, at - start) / GRID) * GRID;
    shiftTo(at);
    open = true; openedBy = by; uttStart = start; uttWords = 0; boundary = start; tail = []; tent = []; hypEnd = start; gen++;
    if (by === "gate") S.gateOpens++; else S.gateMisses++;
    events.onGate?.({ open: true, atSample: captured, by: by === "gate" ? "energy" : "probe", noiseFloor: floor, threshold: threshold() });
  }
  function closeUtterance(endAbs: number, reason: "idle" | "stop" | "skip") {
    const u: UtteranceEvent = { startSample: uttStart, endSample: endAbs, reason, words: uttWords, opened: openedBy };
    open = false; drain = null; tent = []; tail = []; gen++;
    shiftTo(endAbs);
    closedSince = endAbs; lastProbeAt = captured; boundary = endAbs; hypEnd = endAbs;
    S.utterances++;
    events.onTentative?.("", [], captured);
    events.onGate?.({ open: false, atSample: captured, by: reason, noiseFloor: floor, threshold: threshold() });
    events.onUtteranceEnd?.(u);
  }

  // ---------- audio in ----------
  function push(pcm: Float32Array) {
    if (failed) { captured += pcm.length; return; }
    const at = captured;
    captured += pcm.length;
    if (skipFrom >= 0) return void pump(); // skipping: what is buffered is being drained
    if (len + pcm.length > capacity) { // far slower than real time: bounded memory comes first
      skipFrom = at;
      if (open) drain = "skip"; else { shiftTo(end()); endSkip(); }
      return void pump();
    }
    buf.set(pcm, len); len += pcm.length;
    for (let i = 0; i < pcm.length; i++) {
      frame[frameFill++] = pcm[i];
      if (frameFill === FRAME) { frameFill = 0; gateFrame(at + i + 1); }
    }
    if (!open) {
      if (len > closedKeepN + SR / 2) shiftTo(end() - closedKeepN); // closed: only what a probe or a pre-roll needs is kept
    } else if (!drain) {
      // The utterance ends when the gate has been closed for hangS and the model's last word is that old as well.
      if (quietNow() && hypEnd > uttStart) drain = "idle";
    }
    if (len / SR > S.maxBufferS) S.maxBufferS = len / SR;
    const lag = open ? (end() - hypEnd) / SR : 0;
    if (lag > S.maxLagS) S.maxLagS = lag;
    void pump();
  }
  function endSkip() {
    if (skipFrom < 0) return;
    S.skippedS += (captured - skipFrom) / SR;
    events.onSkip?.({ fromSample: skipFrom, toSample: captured });
    skipFrom = -1; start = captured; len = 0; closedSince = captured; lastProbeAt = captured; frameFill = 0; run = 0;
  }

  // ---------- passes ----------
  async function pump(): Promise<void> {
    if (busy || failed) return;
    let kind: PassKind, from = 0, n = 0;
    if (open) {
      if (drain) {
        // The newest pass has usually seen the whole utterance already (and some silence after it): its words are the last words.
        if (drain === "idle" && hypEnd >= Math.max(lastSpeech + 0.3 * SR, end() - 0.5 * SR)) {
          S.reusedFinals++;
          commit(tent.length, "idle"); finishDrain(hypEnd);
          return pump();
        }
        kind = "final";
      } else {
        if (end() - hypEnd < P.minNewS * SR || now() - lastPassStart < P.gapMs) return; // the next block calls again
        kind = "live";
      }
      n = Math.min(len, capN);
      if (n < FRAME) { if (drain) finishDrain(end()); return; }
    } else {
      if (!P.probeEveryS || captured - lastProbeAt < P.probeEveryS * SR) return;
      from = Math.max(0, len - Math.round(P.probeS * SR), closedSince - start);
      from = Math.ceil(from / GRID) * GRID; n = len - from;
      if (n < SR) return;
      lastProbeAt = captured;
      let peak = 0;
      for (let i = from; i < len && peak < P.silencePeak; i += 2) { const a = Math.abs(buf[i]); if (a > peak) peak = a; }
      if (peak < P.silencePeak) { S.probesSilent++; return; }
      kind = "probe"; S.probes++;
    }
    const req: PassRequest = { kind, startSample: start + from, endSample: start + from + n };
    const myGen = gen, t0 = now();
    busy = true; lastPassStart = t0;
    let r: Transcription;
    try { r = await transcribe(buf.slice(from, from + n), req); }
    catch (e) { busy = false; failed = true; events.onError?.(e); settle(); return; }
    busy = false;
    S.passes++; if (kind === "final") S.finalPasses++; else if (kind === "live") S.livePasses++;
    const ev: PassEvent = { ...req, ms: now() - t0, words: r.words?.length ?? 0, committed: 0, tentative: 0, stale: myGen !== gen, atSample: captured };
    if (myGen !== gen) S.stale++; // reset() while it ran: the result belongs to nothing
    else {
      const before = S.wordsCommitted;
      if (kind === "probe") onProbe(req, r); else onPass(req, r, kind === "final");
      ev.committed = S.wordsCommitted - before; ev.tentative = tent.length;
    }
    events.onPass?.(ev);
    settle();
    return pump();
  }
  function settle() {
    if (!idle()) return;
    const w = stopWaiters; stopWaiters = [];
    for (const f of w) f();
  }

  function toAbs(r: Transcription, startSample: number): W[] {
    const out: W[] = [];
    for (const x of r.words ?? []) {
      const n = normalizeWord(x.w);
      if (!n) continue; // punctuation on its own
      out.push({ w: x.w, n, s: startSample + Math.round(x.s * SR), e: startSample + Math.round(x.e * SR), p: x.p ?? 1, since: 0, seen: 1 });
    }
    return out;
  }
  /** Index of the first word of `h` that is not committed yet: by time, corrected by the committed words themselves. */
  function firstFresh(h: W[]): number {
    let k0 = 0;
    while (k0 < h.length && (h[k0].s + h[k0].e) / 2 < boundary) k0++;
    if (!tail.length || tail[tail.length - 1].e < boundary - GRID) return k0; // the boundary moved on over empty audio
    const lastE = tail[tail.length - 1].e;
    // The same word can sit one encoder frame earlier or later from pass to pass: when the words just before the
    // split are not the last committed ones, try one or two words to either side.
    const match = (k: number) => { let m = 0; while (m < 4 && m < k && m < tail.length && h[k - 1 - m].n === tail[tail.length - 1 - m].n) m++; return m; };
    let best = k0, bestM = match(k0);
    for (const d of [-1, 1, -2, 2]) {
      const k = k0 + d;
      if (k < 1 || k > h.length || Math.abs(h[k - 1].e - lastE) > 0.4 * SR) continue;
      const m = match(k);
      if (m > bestM) best = k, bestM = m;
    }
    if (best !== k0) S.anchorShifts++;
    return best;
  }
  function commit(n: number, reason: CommitReason) {
    if (n <= 0) return;
    const ws = tent.splice(0, n);
    const lastE = ws[ws.length - 1].e;
    boundary = Math.max(boundary, tent.length ? Math.min(lastE, tent[0].s) : lastE);
    tail.push(...ws); if (tail.length > 48) tail = tail.slice(-24);
    uttWords += n; S.commits++; S.wordsCommitted += n; S.byReason[reason] += n;
    events.onCommit?.({ text: ws.map((x) => x.w).join(" "), words: ws.map(outWord), reason, atSample: captured });
  }
  function onPass(req: PassRequest, r: Transcription, final: boolean) {
    const h = toAbs(r, req.startSample), passEnd = req.endSample;
    const fresh = h.slice(firstFresh(h));
    // Agreement: a word keeps its age when the previous pass had the same word at about the same time.
    let j = 0;
    for (const x of fresh) {
      x.since = passEnd; x.seen = 1;
      for (let k = j; k < tent.length && tent[k].s <= x.s + 3 * GRID; k++) {
        if (tent[k].n === x.n && Math.abs(tent[k].s - x.s) <= 3 * GRID) { x.since = tent[k].since; x.seen = tent[k].seen + 1; j = k + 1; break; }
      }
    }
    tent = fresh; hypEnd = passEnd;
    const full = req.endSample - req.startSample >= capN - SR / 2, behind = end() - passEnd;
    if (drain === "idle" && !quietNow()) drain = null; // speech went on (frames, or a new word from this pass): carry on
    if (final && drain && behind <= 0) { // the whole utterance, seen to its end
      commit(tent.length, drain);
      return finishDrain(passEnd);
    }
    // Agreement is not waited for when the utterance is being drained or passes are behind the microphone.
    const hurry: CommitReason | null = drain === "stop" || drain === "skip" ? drain : behind >= P.lagS * SR ? "lag" : null;
    let n = 0;
    while (n < tent.length && passEnd - tent[n].e >= P.holdS * SR && (hurry || (tent[n].seen >= P.minPasses && passEnd - tent[n].since >= P.stableS * SR))) n++;
    commit(n, hurry ?? "agree");
    if (full) { // nothing may wait behind a full buffer
      let m = 0;
      while (m < tent.length && passEnd - tent[m].e >= P.forceHoldS * SR) m++;
      commit(m, "cap");
      boundary = Math.max(boundary, Math.min(tent.length ? tent[0].s : passEnd, passEnd - Math.round(P.keepS * SR)));
    }
    // Audio no pass put a word on stops being a reason to keep the buffer long.
    boundary = Math.max(boundary, Math.min(passEnd, tent.length ? tent[0].s : passEnd) - Math.round(P.emptyHoldS * SR));
    trim();
    events.onTentative?.(tent.map((x) => x.w).join(" "), tent.map(outWord), captured);
  }
  /** The gate has been closed for hangS and the newest pass has no word younger than that. */
  function quietNow() {
    const lastWord = tent.length ? tent[tent.length - 1].e : tail.length ? tail[tail.length - 1].e : uttStart;
    const gateQuiet = captured - Math.max(lastSpeech, 0) >= P.hangS * SR, modelQuiet = captured - lastWord >= P.hangS * SR;
    if (gateQuiet && !modelQuiet && lastSpeech < lastWord - 0.3 * SR) { if (!holding) holding = true, S.modelHolds++; } else holding = false;
    return gateQuiet && modelQuiet;
  }
  function finishDrain(endAbs: number) {
    const reason = drain ?? "idle";
    closeUtterance(endAbs, reason);
    if (reason === "skip") { shiftTo(end()); endSkip(); }
    settle();
  }
  function trim() {
    const target = boundary - Math.round(P.keepS * SR);
    if (target - start < SR / 2) return;
    // Cut in the widest gap between committed words near the target, else at the target.
    let cut = target, gap = -1;
    for (let i = tail.length - 1; i >= 0; i--) {
      const w = tail[i];
      if (w.s > target + SR / 2) continue;
      if (w.s < target - SR / 2) break;
      const prevE = i > 0 ? tail[i - 1].e : w.s, g = w.s - prevE;
      if (g > gap) gap = g, cut = Math.round((prevE + w.s) / 2);
    }
    cut = start + Math.floor((Math.min(cut, boundary) - start) / GRID) * GRID;
    if (cut <= start) return;
    shiftTo(cut); S.trims++;
    let drop = 0;
    while (drop < tail.length && tail[drop].e <= start) drop++;
    if (drop) tail.splice(0, drop);
  }
  function onProbe(req: PassRequest, r: Transcription) {
    if (open || drain || skipFrom >= 0) return; // the gate opened by itself meanwhile
    const h = toAbs(r, req.startSample);
    if (!h.length) return;
    // The model hears words where the gate heard nothing: the utterance starts where the probe window did.
    openUtterance(req.startSample, "probe");
    lastSpeech = Math.max(lastSpeech, h[h.length - 1].e);
    for (const x of h) x.since = req.endSample;
    tent = h; hypEnd = req.endSample;
    events.onTentative?.(tent.map((x) => x.w).join(" "), tent.map(outWord), captured);
  }

  function idle() { return !busy && !open && skipFrom < 0; }
  function stop(): Promise<void> {
    if (failed) return Promise.resolve();
    if (open && drain !== "skip") drain = "stop";
    if (!open) { gen++; if (skipFrom >= 0) endSkip(); } // a probe still running is not acted on
    if (idle()) return Promise.resolve();
    const p = new Promise<void>((res) => stopWaiters.push(res));
    void pump();
    return p;
  }
  function reset() {
    gen++; busy = false; failed = false;
    start = 0; len = 0; captured = 0; open = false; drain = null; skipFrom = -1; holding = false; tent = []; tail = []; boundary = 0; hypEnd = 0; uttStart = 0; uttWords = 0;
    subFill = 0; subCur = Infinity; subAt = 0; floor = 0; lastRms = 0; speech = false; run = 0; r1 = r2 = 0; frames = 0; frameFill = 0; lastSpeech = -1;
    lastPassStart = -Infinity; lastProbeAt = 0; closedSince = 0;
    S = zero();
    settle();
  }
  function stats(): StreamStats {
    return { ...S, byReason: { ...S.byReason }, capturedS: captured / SR, open, busy, failed, skipping: skipFrom >= 0, bufferS: len / SR, lagS: open ? Math.max(0, (end() - hypEnd) / SR) : 0,
      noiseFloor: floor, threshold: threshold(), lastRms, speech };
  }
  return { push, stop, reset, idle, stats, params: P };
}
