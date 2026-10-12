export type MicrophoneErrorCode = "insecure-context" | "unsupported" | "permission-denied" | "no-device" | "device-busy" | "timeout" | "ended" | "stopped" | "failed";
export interface MicrophoneError extends Error {
    code: MicrophoneErrorCode;
    cause?: unknown;
}
export declare function isMicrophoneError(e: unknown): e is MicrophoneError;
/** off; starting (start() at work); live; silent (4 s with only zeros: muted, or blocked by the system); interrupted (the browser paused capture, e.g. the tab is in the background). */
export type MicrophoneStatus = "off" | "starting" | "live" | "silent" | "interrupted";
export interface MicrophoneInfo {
    /** The AudioContext's rate (the device's); the blocks are always 16 kHz. */
    sampleRate: number;
    contextState: string;
    baseLatency: number | null;
    /** MediaTrackSettings without the device ids. */
    track: Record<string, unknown>;
    trackLabelPresent: boolean;
    voiceProcessing: boolean;
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
    start(overrides?: {
        voiceProcessing?: boolean;
    }): Promise<MicrophoneInfo>;
    /** Releases the microphone (the browser's recording indicator goes off). Safe to call at any time. */
    stop(): void;
    readonly status: MicrophoneStatus;
}
export declare function createMicrophone(options: MicrophoneOptions): Microphone;
