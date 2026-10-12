import { type MicrophoneError, type MicrophoneStatus } from "./microphone";
import { type Step } from "./runtime";
import { type CommitEvent, type PassKind, type StreamEvents, type StreamParams, type StreamStats } from "./stream";
import { type LoadProgress, type Transcriber, type TranscriberAdvanced, type TranscriberError, type TranscriberInfo, type TranscribeResult } from "./transcriber";
export interface DictationState {
    /** loading (also while a new model replaces the old one); ready; failed (onError said why; load() tries again). */
    model: "loading" | "ready" | "failed";
    recording: boolean;
    microphone: MicrophoneStatus;
}
export type DictationError = TranscriberError | MicrophoneError;
/** A pass about to run. `sinceLoad`: passes finished since this model was loaded; `longestBefore`: the longest of them, seconds. */
export interface PassStart {
    n: number;
    sinceLoad: number;
    kind: PassKind;
    audioSeconds: number;
    longestBefore: number;
}
/** A pass that came back. `ms`: wall time as the page saw it, queueing included. `result` holds the text: keep it on the device. */
export interface PassDone {
    n: number;
    sinceLoad: number;
    kind: PassKind;
    audioSeconds: number;
    ms: number;
    endSample: number;
    result: TranscribeResult;
}
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
    onLevel?(level: {
        peak: number;
        speech: boolean;
        capturedSeconds: number;
    }): void;
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
    start(overrides?: {
        voiceProcessing?: boolean;
        transcribe?: boolean;
    }): Promise<boolean>;
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
export declare function createDictation(options: DictationOptions): Dictation;
