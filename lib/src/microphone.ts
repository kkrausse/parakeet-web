// Layer 3, the microphone: getUserMedia -> AudioWorklet (assets/worklet.js, a polyphase sinc resampler) -> 40 ms
// blocks of 16 kHz mono float PCM. The iOS rules are in here: the AudioContext is created and resumed inside the
// call to start() before anything is awaited (Safari only starts audio from a user gesture), it runs at the device
// rate, the graph ends in a muted gain on the output so every browser keeps pulling it, and the context is resumed
// when the page comes back to the front.
import { directoryUrl } from "./runtime";

export type MicrophoneErrorCode =
  | "insecure-context" // no https (or localhost)
  | "unsupported" // no getUserMedia, AudioContext or AudioWorklet in this browser
  | "permission-denied" | "no-device" | "device-busy"
  | "timeout" // the browser neither granted nor refused in time (startTimeoutMs)
  | "ended" // the device went away while recording
  | "stopped" // stop() was called while start() was still at work
  | "failed";
export interface MicrophoneError extends Error { code: MicrophoneErrorCode; cause?: unknown }
export function isMicrophoneError(e: unknown): e is MicrophoneError { return e instanceof Error && typeof (e as any).code === "string" && (e as any).microphone === true; }
function micError(code: MicrophoneErrorCode, message: string, cause?: unknown): MicrophoneError {
  return Object.assign(new Error(message), { name: "MicrophoneError", code, cause, microphone: true as const });
}
function fromDomError(e: any): MicrophoneError {
  if (isMicrophoneError(e)) return e;
  const n = e?.name ?? "";
  if (n === "NotAllowedError" || n === "SecurityError") return micError("permission-denied", "Microphone access was refused. Allow the microphone for this site in the browser's settings, then start again.", e);
  if (n === "NotFoundError" || n === "OverconstrainedError") return micError("no-device", "No microphone was found on this device.", e);
  if (n === "NotReadableError" || n === "AbortError") return micError("device-busy", "The microphone could not be opened (another app may be using it).", e);
  return micError("failed", `The microphone could not be started: ${n} ${e?.message ?? e}`.trim(), e);
}

/** off; starting (start() at work); live; silent (4 s with only zeros: muted, or blocked by the system); interrupted (the browser paused capture, e.g. the tab is in the background). */
export type MicrophoneStatus = "off" | "starting" | "live" | "silent" | "interrupted";
export interface MicrophoneInfo {
  /** The AudioContext's rate (the device's); the blocks are always 16 kHz. */
  sampleRate: number;
  contextState: string; baseLatency: number | null;
  /** MediaTrackSettings without the device ids. */
  track: Record<string, unknown>;
  trackLabelPresent: boolean; voiceProcessing: boolean;
}
export interface MicrophoneOptions {
  /** The folder holding the package's assets (worklet.js). */
  assetsUrl: string | URL;
  /** A block of 16 kHz mono PCM (640 samples, 40 ms; the array is yours) and its peak, 0 to 1. */
  onAudio(pcm: Float32Array, peak: number): void;
  onStatus?(status: MicrophoneStatus): void;
  /** The device went away while recording; the microphone has stopped itself. */
  onEnded?(e: MicrophoneError): void;
  /** Browser echo cancellation and noise suppression. Default false: the model gets the raw microphone. */
  voiceProcessing?: boolean;
  /** Keep the screen awake while recording, where the browser allows it. Default true. */
  wakeLock?: boolean;
  /**
   * Give up (code "timeout") when the browser has neither granted nor refused the microphone after this long.
   * Default 30000: some browsers never answer where capture is unavailable (Playwright's WebKit does this), which
   * cannot be told from a person looking at the permission prompt. 0 waits for ever.
   */
  startTimeoutMs?: number;
  /** "mic-track" (mute / unmute / ended) and "audio-context" (state changes). */
  onDiagnostic?(kind: string, data: Record<string, unknown>): void;
}
export interface Microphone {
  /**
   * Call it directly from the tap or click handler, with nothing awaited before it. Rejects with a MicrophoneError
   * (its message can be shown to the person as it is).
   */
  start(overrides?: { voiceProcessing?: boolean }): Promise<MicrophoneInfo>;
  /** Releases the microphone (the browser's recording indicator goes off). Safe to call at any time. */
  stop(): void;
  readonly status: MicrophoneStatus;
}

const SILENT_AFTER_MS = 4000;
const START_TIMEOUT_MS = 30000;

export function createMicrophone(options: MicrophoneOptions): Microphone {
  const workletUrl = new URL("worklet.js", directoryUrl(options.assetsUrl)).href;
  let status: MicrophoneStatus = "off";
  let session = 0; // start() and stop() each begin a new one; work of an older one is dropped
  let ctx: AudioContext | null = null, stream: MediaStream | null = null, node: AudioWorkletNode | null = null, wake: any = null;
  let startedAt = 0, sawSignal = false;
  const setStatus = (s: MicrophoneStatus) => { if (s !== status) { status = s; options.onStatus?.(s); } };
  const diag = (kind: string, data: Record<string, unknown>) => options.onDiagnostic?.(kind, data);
  async function acquireWake() { if (options.wakeLock === false) return; try { wake = await (navigator as any).wakeLock?.request("screen"); } catch { wake = null; } }
  function onVisible() {
    if (document.visibilityState !== "visible" || status === "off" || status === "starting") return;
    void ctx?.resume().catch(() => {});
    if (!wake || wake.released) void acquireWake();
  }
  function release() {
    document.removeEventListener("visibilitychange", onVisible);
    try { node?.port.postMessage("stop"); node?.disconnect(); } catch { /* gone */ }
    stream?.getTracks().forEach((t) => t.stop());
    if (ctx) ctx.onstatechange = null;
    void ctx?.close().catch(() => {});
    try { void wake?.release(); } catch { /* gone */ }
    node = null; stream = null; ctx = null; wake = null;
  }
  function stop() {
    session++;
    release();
    setStatus("off");
  }
  async function start(overrides: { voiceProcessing?: boolean } = {}): Promise<MicrophoneInfo> {
    if (status !== "off") throw micError("failed", "The microphone is already started.");
    if (!navigator.mediaDevices?.getUserMedia) throw micError(isSecureContext ? "unsupported" : "insecure-context", isSecureContext ? "This browser has no microphone capture (getUserMedia)." : "Microphone capture needs https (or localhost); this page was opened over plain http.");
    const AC: typeof AudioContext | undefined = (window as any).AudioContext ?? (window as any).webkitAudioContext;
    if (!AC) throw micError("unsupported", "This browser has no Web Audio (AudioContext).");
    const mine = ++session;
    setStatus("starting");
    // Created and resumed inside the tap, before any await: Safari only starts audio from a user gesture.
    // The context runs at the device rate (usually 48 kHz); the worklet resamples to 16 kHz.
    const c = (ctx = new AC({ latencyHint: "interactive" }));
    void c.resume().catch(() => {});
    const current = () => { if (mine !== session) throw micError("stopped", "The microphone was stopped while it was starting."); };
    try {
      if (!c.audioWorklet) throw micError("unsupported", "AudioWorklet is not available in this browser.");
      const proc = overrides.voiceProcessing ?? options.voiceProcessing ?? false;
      const asked = navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: proc, noiseSuppression: proc, autoGainControl: true }, video: false });
      const limit = options.startTimeoutMs ?? START_TIMEOUT_MS;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const s = await (limit > 0 ? Promise.race([asked, new Promise<never>((_, rej) => { timer = setTimeout(() => rej(micError("timeout", `The browser did not answer the microphone request within ${Math.round(limit / 1000)} s (no permission prompt answered, or capture is not available here).`)), limit); })]) : asked)
        .catch((e) => { void asked.then((late) => late.getTracks().forEach((t) => t.stop()), () => {}); throw e; }) // a grant that arrives after the timeout is released at once
        .finally(() => clearTimeout(timer));
      if (mine !== session) { s.getTracks().forEach((t) => t.stop()); current(); }
      stream = s;
      await c.audioWorklet.addModule(workletUrl).catch((e) => { throw micError("failed", `The audio worklet could not be loaded from ${workletUrl} (is assetsUrl the folder holding worklet.js?): ${e?.message ?? e}`, e); });
      current();
      if (c.state !== "running") await c.resume().catch(() => {});
      current();
      const src = c.createMediaStreamSource(s);
      const n = (node = new AudioWorkletNode(c, "pk-resample16k", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers" }));
      n.port.onmessage = (e) => {
        if (node !== n || mine !== session || status === "starting") return;
        if (e.data.rate) return;
        if (e.data.peak > 0) sawSignal = true;
        options.onAudio(e.data.pcm, e.data.peak);
        if (mine !== session) return; // stopped from inside onAudio
        if (!sawSignal && performance.now() - startedAt > SILENT_AFTER_MS) { if (status === "live") setStatus("silent"); }
        else if (sawSignal && status === "silent") setStatus("live");
      };
      // Through a muted gain to the output so every browser keeps pulling the graph; nothing is played.
      const mute = c.createGain(); mute.gain.value = 0;
      src.connect(n); n.connect(mute); mute.connect(c.destination);
      const track = s.getAudioTracks()[0];
      track?.addEventListener("ended", () => {
        if (stream !== s) return;
        stop();
        options.onEnded?.(micError("ended", "The microphone went away; recording stopped."));
      });
      for (const evName of ["mute", "unmute", "ended"]) track?.addEventListener(evName, () => diag("mic-track", { event: evName, readyState: track.readyState, muted: track.muted }));
      c.onstatechange = () => {
        diag("audio-context", { state: c.state, sampleRate: c.sampleRate, recording: status !== "off" && status !== "starting" });
        if (ctx !== c || status === "off" || status === "starting") return;
        if (c.state !== "running") setStatus("interrupted");
        else if (status === "interrupted") setStatus(sawSignal ? "live" : performance.now() - startedAt > SILENT_AFTER_MS ? "silent" : "live");
      };
      document.addEventListener("visibilitychange", onVisible);
      const set: Record<string, unknown> = { ...(track?.getSettings?.() ?? {}), deviceId: undefined, groupId: undefined };
      startedAt = performance.now(); sawSignal = false;
      setStatus("live");
      void acquireWake();
      return { sampleRate: c.sampleRate, contextState: c.state, baseLatency: c.baseLatency ?? null, track: set, trackLabelPresent: !!track?.label, voiceProcessing: proc };
    } catch (e: any) {
      const err = Object.assign(fromDomError(e), { contextState: c.state });
      if (mine === session) { release(); setStatus("off"); }
      else void c.close().catch(() => {}); // stop() already released what this session had, except a context it had replaced
      throw err;
    }
  }
  return { start, stop, get status() { return status; } };
}
