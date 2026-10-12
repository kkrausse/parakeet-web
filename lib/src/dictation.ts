// The common case in one call: microphone -> streaming engine -> transcriber. The model starts loading when
// createDictation is called; start() may be called before it is ready (audio waits in the engine), and must be
// called directly from the tap handler. Committed text only ever grows; tentative text replaces itself.
import { createMicrophone, isMicrophoneError, type MicrophoneError, type MicrophoneInfo, type MicrophoneStatus } from "./microphone";
import { detectDevice, type Step } from "./runtime";
import { createStream, SAMPLE_RATE, type CommitEvent, type PassKind, type StreamEvents, type StreamParams, type StreamStats, type Transcribe } from "./stream";
import { createTranscriber, type LoadProgress, type Transcriber, type TranscriberAdvanced, type TranscriberError, type TranscriberInfo, type TranscribeResult } from "./transcriber";

export interface DictationState {
  /** loading (also while a new model replaces the old one); ready; failed (onError said why; load() tries again). */
  model: "loading" | "ready" | "failed";
  recording: boolean;
  microphone: MicrophoneStatus;
}
export type DictationError = TranscriberError | MicrophoneError;
/** A pass about to run. `sinceLoad`: passes finished since this model was loaded; `longestBefore`: the longest of them, seconds. */
export interface PassStart { n: number; sinceLoad: number; kind: PassKind; audioSeconds: number; longestBefore: number }
/** A pass that came back. `ms`: wall time as the page saw it, queueing included. `result` holds the text: keep it on the device. */
export interface PassDone { n: number; sinceLoad: number; kind: PassKind; audioSeconds: number; ms: number; endSample: number; result: TranscribeResult }
export interface DictationOptions {
  /** The folder the package's dist/assets was copied to. Same origin as the page. */
  assetsUrl: string | URL;
  /** The GGUF model file (recommended: parakeet-tdt_ctc-110m-Q4_0.gguf). Downloaded once per browser. */
  modelUrl: string | URL;
  /** Words to append to the transcript; never revised. Put a space between successive texts. */
  onCommit?(text: string, event: CommitEvent): void;
  /** The current guess for what follows the committed text; replaces the previous one ("" clears it). */
  onTentative?(text: string): void;
  onState?(state: DictationState): void;
  /** A model failure (state.model is "failed") or a microphone failure (recording has stopped). The message can be shown as it is. */
  onError?(error: DictationError): void;
  onLoadProgress?(p: LoadProgress): void;
  /** Per 40 ms block while recording: peak 0 to 1, whether the gate calls it speech, seconds captured since start(). */
  onLevel?(level: { peak: number; speech: boolean; capturedSeconds: number }): void;
  /** Longest audio one pass is given, seconds. Default 12 on phones, 24 elsewhere. GPU memory grows with it. */
  maxUtteranceSeconds?: number;
  /** "cpu": never use WebGPU. Default "auto". */
  backend?: "auto" | "cpu";
  /** Browser echo cancellation and noise suppression. Default false. */
  voiceProcessing?: boolean;
  /** Keep the screen awake while recording. Default true. */
  wakeLock?: boolean;
  /** The escape hatch: tuning and diagnostics. Nothing here is needed for dictation to work. */
  advanced?: {
    transcriber?: TranscriberAdvanced;
    /** The engine's numbers (see StreamParams); capS is maxUtteranceSeconds. */
    stream?: Partial<StreamParams>;
    /** Every engine event (gate, passes, utterance ends, skipped audio), beside what the options above get. */
    streamEvents?: StreamEvents;
    /** false: no sizing pass at load (the GPU working buffer then grows with the first long utterances). */
    sizingPass?: boolean;
    /** The step trail of the load and of each recording. */
    onStep?(step: Step): void;
    /** Structured events of the load ("adapter", "loaded", "load-heap", "warmup-pass") and the microphone ("recording", "mic-track", "audio-context"). */
    onDiagnostic?(kind: string, data: any): void;
    onPassStart?(p: PassStart): void;
    onPass?(p: PassDone): void;
  };
}
export interface Dictation {
  /**
   * Starts recording. Call it directly from the tap or click handler with nothing awaited before it (iOS).
   * Resolves true once recording, false when the microphone could not be started (onError has the reason).
   * `transcribe: false` records and reports levels only. Never rejects.
   */
  start(overrides?: { voiceProcessing?: boolean; transcribe?: boolean }): Promise<boolean>;
  /** Stops recording and releases the microphone. Resolves when the last words have been committed (also when called again while that is going on). */
  stop(): Promise<void>;
  /**
   * Loads another model (or the same one again after a failure), replacing the current one. Recording may go on
   * meanwhile: new audio waits for the model; words not yet committed when the old model is dropped may be lost.
   */
  load(modelUrl?: string | URL): Promise<TranscriberInfo>;
  /** Resolves when the model requested last is ready. */
  readonly ready: Promise<TranscriberInfo>;
  readonly state: DictationState;
  /** The engine's counters: lag, buffer, gate, passes, commit reasons. */
  stats(): StreamStats;
  /** True when not recording and nothing is left to transcribe. */
  idle(): boolean;
  readonly params: StreamParams;
  /** The loaded transcriber (null while loading or failed), for offline passes and GPU accounting. Its calls queue with the dictation's own. */
  readonly transcriber: Transcriber | null;
  /** Stops everything and frees the model, the GPU device and the worker. The object is dead afterwards. */
  dispose(): Promise<void>;
}

export function createDictation(options: DictationOptions): Dictation {
  const adv = options.advanced ?? {};
  const ev = adv.streamEvents ?? {};
  const capS = options.maxUtteranceSeconds ?? adv.stream?.capS ?? (detectDevice().phone ? 12 : 24);
  let modelUrl = options.modelUrl;
  let model: DictationState["model"] = "loading";
  let transcriber: Transcriber | null = null;
  let loadGen = 0, abort: AbortController | null = null;
  let recording = false, starting: Promise<boolean> | null = null, transcribing = true, captured = 0, disposed = false;
  let micStatus: MicrophoneStatus = "off";
  let passes = 0, sinceLoad = 0, longest = 0; // passes ever; since the model was loaded; the longest of those, samples
  const waiters: (() => void)[] = [];
  const wake = () => { for (const w of waiters.splice(0)) w(); };
  const state = (): DictationState => ({ model, recording, microphone: micStatus });
  let told = "";
  const changed = () => { // only real changes are reported
    const s = state(), key = `${s.model} ${s.recording} ${s.microphone}`;
    if (disposed || key === told) return;
    told = key; options.onState?.(s);
  };
  const step = (name: string, ms?: number, detail?: string) => { if (!disposed) adv.onStep?.({ name, ms, detail }); };
  const report = (e: unknown) => { if (!disposed) options.onError?.(e as DictationError); };

  /** One pass for the engine. Audio recorded before the model is ready waits here; rejects when the model is gone. */
  const transcribe: Transcribe = async (pcm, req) => {
    while (model === "loading") await new Promise<void>((res) => waiters.push(res));
    const t = transcriber;
    if (model !== "ready" || !t) throw new Error("the model is not running");
    const gen = loadGen, n = ++passes, audioSeconds = pcm.length / SAMPLE_RATE;
    adv.onPassStart?.({ n, sinceLoad, kind: req.kind, audioSeconds, longestBefore: longest / SAMPLE_RATE });
    if (pcm.length > longest) longest = pcm.length;
    const t0 = performance.now();
    let result: TranscribeResult;
    try { result = await t.transcribe(pcm, { transfer: true }); } // a failure has gone to onError through the transcriber
    catch (e) {
      // load() replaced the model under this pass: its audio went with the old worker. The engine stops on the
      // rejection; it is started afresh so a recording in progress goes on with the new model.
      if (gen !== loadGen && !disposed) queueMicrotask(() => { if (stream.stats().failed) stream.reset(); });
      throw e;
    }
    if (gen !== loadGen) throw new Error("the model was reloaded"); // the engine was reset with it
    adv.onPass?.({ n, sinceLoad: ++sinceLoad, kind: req.kind, audioSeconds, ms: performance.now() - t0, endSample: req.endSample, result });
    return result;
  };
  const stream = createStream(transcribe, {
    onCommit(c) { ev.onCommit?.(c); options.onCommit?.(c.text, c); },
    onTentative(text, words, at) { ev.onTentative?.(text, words, at); options.onTentative?.(text); },
    onUtteranceEnd(u) { ev.onUtteranceEnd?.(u); },
    onPass(p) { ev.onPass?.(p); },
    onGate(g) { ev.onGate?.(g); },
    onSkip(s) { ev.onSkip?.(s); },
    onError(e) { ev.onError?.(e); },
  }, { ...adv.stream, capS });

  const microphone = createMicrophone({
    assetsUrl: options.assetsUrl, voiceProcessing: options.voiceProcessing, wakeLock: options.wakeLock,
    onAudio(pcm, peak) {
      if (!recording) return;
      captured += pcm.length;
      let speech = false;
      if (transcribing) { stream.push(pcm); speech = stream.stats().speech; }
      options.onLevel?.({ peak, speech, capturedSeconds: captured / SAMPLE_RATE });
    },
    onStatus(s) { micStatus = s; changed(); },
    onEnded(e) { void stop(); report(e); },
    onDiagnostic: (kind, data) => { if (!disposed) adv.onDiagnostic?.(kind, data); },
  });

  let ready: Promise<TranscriberInfo>;
  function load(url: string | URL = modelUrl): Promise<TranscriberInfo> {
    if (disposed) return Promise.reject(new Error("the dictation was disposed"));
    const gen = ++loadGen;
    abort?.abort();
    const old = transcriber;
    transcriber = null; modelUrl = url; model = "loading"; sinceLoad = 0; longest = 0;
    void old?.dispose(); // a pass in flight never comes back: the worker is stopped where it stands
    if (!recording || stream.stats().failed) stream.reset(); // while recording, audio keeps arriving and waits for this load
    changed();
    const mine = (abort = new AbortController());
    const p = createTranscriber({
      assetsUrl: options.assetsUrl, modelUrl: url, backend: options.backend, signal: mine.signal,
      maxSeconds: adv.sizingPass === false ? undefined : stream.params.capS,
      onProgress: (pr) => { if (gen === loadGen && !disposed) options.onLoadProgress?.(pr); },
      onError(e) { if (gen !== loadGen || disposed) return; model = "failed"; changed(); report(e); },
      advanced: { ...adv.transcriber,
        onStep: (s) => { if (gen === loadGen && !disposed) { adv.transcriber?.onStep?.(s); adv.onStep?.(s); } },
        onDiagnostic: (kind, data) => { if (gen === loadGen && !disposed) { adv.transcriber?.onDiagnostic?.(kind, data); adv.onDiagnostic?.(kind, data); } } },
    }).then((t) => {
      if (gen !== loadGen || disposed) { void t.dispose(); throw Object.assign(new Error("the load was replaced by a newer one"), { name: "AbortError" }); }
      transcriber = t; model = "ready";
      changed(); wake();
      return t.info;
    }, (e) => {
      if (gen === loadGen && !disposed) { model = "failed"; changed(); wake(); report(e); } // a pass waiting for the model fails now instead of waiting for ever
      throw e;
    });
    p.catch(() => {}); // failures go to onError; nobody has to hold the promise
    return (ready = p);
  }

  function start(overrides: { voiceProcessing?: boolean; transcribe?: boolean } = {}): Promise<boolean> {
    if (disposed) return Promise.resolve(false);
    if (recording) return Promise.resolve(true);
    if (starting) return starting;
    // microphone.start() runs up to its first await right here, inside the caller's tap: the AudioContext is made and resumed now.
    const mic = microphone.start({ voiceProcessing: overrides?.voiceProcessing });
    const p: Promise<boolean> = mic.then(async (info: MicrophoneInfo) => {
      // a fresh engine (positions start at 0 again); what the caller has of the transcript is kept
      if (!stream.idle() && model === "ready") await stream.stop(); // the last words of the previous recording first
      if (starting !== p || disposed) return false; // stop() or dispose() meanwhile
      stream.reset(); captured = 0;
      transcribing = overrides?.transcribe !== false;
      recording = true; starting = null;
      adv.onDiagnostic?.("recording", { ...info, transcribing });
      const set = info.track;
      step(`recording: context ${info.contextState} at ${info.sampleRate} Hz -> 16000 Hz, track ${set.sampleRate ?? "?"} Hz, ${set.channelCount ?? "?"} ch, echoCancellation=${set.echoCancellation} noiseSuppression=${set.noiseSuppression} autoGainControl=${set.autoGainControl}`);
      changed();
      return true;
    }, (e) => {
      if (starting === p) starting = null;
      step(`microphone: ${e?.cause?.name ?? e?.name ?? "Error"}: ${e?.cause?.message ?? e?.message ?? e}`, undefined, e?.contextState ? `AudioContext ${e.contextState}` : undefined);
      changed();
      if (!(isMicrophoneError(e) && e.code === "stopped")) report(e);
      return false;
    });
    return (starting = p);
  }
  let draining: Promise<void> | null = null;
  function stop(): Promise<void> {
    if (starting) { starting = null; microphone.stop(); return Promise.resolve(); }
    if (!recording) return draining ?? Promise.resolve(); // already stopped: the same wait for the last words
    recording = false;
    microphone.stop();
    changed();
    const d: Promise<void> = stream.stop().catch(() => {}).then(() => { if (draining === d) draining = null; }); // what was said since the last commit still gets its final pass
    return (draining = d);
  }
  async function dispose() {
    if (disposed) return;
    recording = false; starting = null;
    microphone.stop();
    disposed = true;
    loadGen++;
    abort?.abort();
    stream.reset();
    model = "failed"; wake();
    const t = transcriber; transcriber = null;
    await t?.dispose();
  }

  // The first load starts after createDictation has returned, so no callback runs before the caller holds the object.
  ready = Promise.resolve().then(() => (loadGen === 0 && !disposed ? load() : ready));
  ready.catch(() => {});
  return {
    start, stop, load, dispose,
    get ready() { return ready; },
    get state() { return state(); },
    stats: () => stream.stats(),
    idle: () => !recording && !starting && stream.idle(),
    params: stream.params,
    get transcriber() { return transcriber; },
  };
}
