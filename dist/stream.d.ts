export declare const SAMPLE_RATE = 16000;
/** One word. `s` / `e` are seconds: from the start of the PCM in a Transcription, from the start of the stream in events. */
export interface Word {
    w: string;
    s: number;
    e: number;
    p?: number;
}
export interface Transcription {
    text: string;
    words: Word[];
}
export type PassKind = "live" | "final" | "probe";
export interface PassRequest {
    kind: PassKind;
    startSample: number;
    endSample: number;
}
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
    /** A word's end time is taken to be no later than this after its start, seconds (see toAbs). */
    maxWordS: number;
}
export declare const DEFAULT_PARAMS: StreamParams;
/** agree: passes agreed; idle / stop: the utterance ended; cap: full buffer; lag: passes behind the microphone; skip: draining before skipped audio. */
export type CommitReason = "agree" | "idle" | "stop" | "cap" | "lag" | "skip";
export interface CommitEvent {
    text: string;
    words: Word[];
    reason: CommitReason;
    atSample: number;
}
export interface UtteranceEvent {
    startSample: number;
    endSample: number;
    reason: "idle" | "stop" | "skip";
    words: number;
    opened: "gate" | "probe";
}
export interface PassEvent {
    kind: PassKind;
    startSample: number;
    endSample: number;
    ms: number;
    words: number;
    committed: number;
    tentative: number;
    stale: boolean;
    atSample: number;
}
export interface GateEvent {
    open: boolean;
    atSample: number;
    by: "energy" | "probe" | "idle" | "stop" | "skip";
    noiseFloor: number;
    threshold: number;
}
export interface StreamEvents {
    /** Words appended to the committed text; never revised. `text` is the words joined by spaces. */
    onCommit?(c: CommitEvent): void;
    /** The current guess for what follows the committed text; replaces the previous one ("" clears it). */
    onTentative?(text: string, words: Word[], atSample: number): void;
    onUtteranceEnd?(u: UtteranceEvent): void;
    onPass?(p: PassEvent): void;
    onGate?(g: GateEvent): void;
    /** Audio that was not transcribed because passes are slower than real time. */
    onSkip?(s: {
        fromSample: number;
        toSample: number;
    }): void;
    /** transcribe() rejected: the engine stops running passes until reset(). */
    onError?(e: unknown): void;
}
export interface StreamStats {
    capturedS: number;
    open: boolean;
    busy: boolean;
    failed: boolean;
    skipping: boolean;
    bufferS: number;
    maxBufferS: number;
    lagS: number;
    maxLagS: number;
    noiseFloor: number;
    threshold: number;
    lastRms: number;
    speech: boolean;
    gateOpens: number;
    gateMisses: number;
    modelHolds: number;
    probes: number;
    probesSilent: number;
    passes: number;
    livePasses: number;
    finalPasses: number;
    stale: number;
    reusedFinals: number;
    utterances: number;
    commits: number;
    wordsCommitted: number;
    byReason: Record<CommitReason, number>;
    skippedS: number;
    trims: number;
    anchorShifts: number;
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
/** Case and punctuation do not count when two passes are compared. */
export declare const normalizeWord: (w: string) => string;
export declare function createStream(transcribe: Transcribe, events?: StreamEvents, options?: Partial<StreamParams> & {
    now?: () => number;
}): Stream;
