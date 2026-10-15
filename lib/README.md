# parakeet-web

Speech to text in the browser, on the device: NVIDIA Parakeet on transcribe.cpp + ggml's WebGPU backend, compiled
to WASM. Dictation with text appearing as you speak, or offline transcription of a clip with word times. No server
takes part; the library sends nothing anywhere and stores only the model file (in the browser's origin-private
file system).

This file is how to use it. Building it, the runtime it is compiled from and the demo pages are in the repository's
README ([`kkrausse/parakeet-web`](https://github.com/kkrausse/parakeet-web)).

## Install

The package is not on a registry. It is installed from the repository by tag (the tag points at the `release`
branch, which holds the built package and nothing else):

```sh
bun add github:kkrausse/parakeet-web#v0.1.0
bunx parakeet-web-assets public/parakeet-web     # copies the runtime files into a folder your app serves
```

(`bun add /path/to/parakeet-web-0.1.0.tgz`, the tarball `lib/build.ts` leaves beside it, installs the same files.)

The second line copies `node_modules/parakeet-web/dist/assets/` (12 MB: `worker.js`, `worklet.js`, two builds
of the WASM module, `pk-web.js` + `pk-web.wasm` with JSPI and `pk-web-asyncify.js` + `pk-web-asyncify.wasm`
without, and `LICENSES.txt`, the licence texts of what the WASM is compiled from). They are loaded by URL at run time; a browser fetches one of the two builds (4.9 or 7.0 MB), never both.
Copy them again after upgrading the package. Nothing depends on how your bundler treats workers or WASM: the
library only ever does `new Worker(assetsUrl + "worker.js")`.

- The folder must be on the **same origin** as the page (a worker cannot be started from another origin).
- No COOP / COEP headers are needed (one WASM thread, no SharedArrayBuffer).
- The microphone needs **https** (or `localhost`).

## The model file

Not in the package. Host one GGUF file as a static file and pass its URL as `modelUrl`.

| file | size | notes |
|---|---|---|
| `parakeet-tdt_ctc-110m-Q4_0.gguf` | 81 MB | **recommended**; the one every measurement below used |
| `parakeet-tdt_ctc-110m-Q8_0.gguf` | 129 MB | same model, 8-bit weights; follows the reference punctuation more closely |
| `parakeet-tdt-0.6b-v2-Q4_0.gguf` | 438 MB | larger model, about 0.5 GB of GPU memory; not run through the streaming engine |

Where they come from: the 110m F32 and Q8_0 files are from the Hugging Face repo
`handy-computer/parakeet-tdt_ctc-110m-gguf`; Q4_0 is made from the F32 one with transcribe.cpp's quantizer
(`transcribe-quantize parakeet-tdt_ctc-110m-F32.gguf parakeet-tdt_ctc-110m-Q4_0.gguf --quant Q4_0`). The models are
NVIDIA's, licensed CC BY 4.0: wherever you host a model file, credit NVIDIA Parakeet and say the file was converted
and quantized.

- The file is downloaded once per browser and kept in OPFS (directory `pk-models`, under the file's name). A later
  visit asks the server for the size only (`HEAD`) and reads the stored copy if it matches, so the server must
  answer `HEAD` with a `Content-Length`. **Give a changed model a new file name**: same name and same size means
  the stored copy is used.
- From another origin the file needs CORS headers (`Access-Control-Allow-Origin`).
- Without OPFS (private windows, some embedded browsers) the file is fetched again on every visit.

## Dictation in ten lines

```ts
import { createDictation } from "parakeet-web";

const textarea = document.querySelector("textarea")!, button = document.querySelector("button")!;
let committed = "";
const dictation = createDictation({
  assetsUrl: "/parakeet-web/",
  modelUrl: "/models/parakeet-tdt_ctc-110m-Q4_0.gguf",
  onCommit(words) { committed += (committed ? " " : "") + words; textarea.value = committed; },
  onTentative(guess) { textarea.value = committed + (guess ? " " + guess : ""); },
  onError(e) { alert(e.message); },
});
button.onclick = () => (dictation.state.recording ? dictation.stop() : dictation.start());
```

`lib/example/` in the repository (not in the package) holds this as a complete small app, bundled with `bun build`.

What to know:

- **`start()` must be called directly in the click / tap handler**, with nothing awaited before it. It creates and
  resumes the AudioContext before it returns, which is what iOS requires. It never rejects: it resolves `true`
  once recording, `false` if the microphone could not be started (then `onError` has a message that can be shown
  as it is).
- The model starts loading when `createDictation` is called. `start()` may be called before it is ready: the audio
  waits and is transcribed when the model is. To delay the 81 MB download until the first tap, call
  `createDictation` in that tap handler and `start()` right after it.
- **Committed text only grows.** `onCommit` gets the next words (join them with a space); `onTentative` gets the
  current guess for what follows, replacing the previous guess. A word is committed about 1.2-1.6 s after it was
  said; the guess appears within about 0.4 s.
- `stop()` resolves when the last words have been committed. `dispose()` frees the model, the GPU device and the
  worker; make a new dictation afterwards if needed.
- `load(otherModelUrl)` switches models (or retries after a failure). It can be called while recording.
- `onState({ model, recording, microphone })`: `model` is `loading | ready | failed`; `microphone` is
  `off | starting | live | silent | interrupted` (`silent`: 4 s of zeros, muted or blocked by the system;
  `interrupted`: the browser paused capture, typically the tab went to the background; capture resumes when it is
  in front again).
- `onError(e)`: a microphone error (`e.code`: `permission-denied`, `no-device`, `device-busy`, `insecure-context`,
  `unsupported`, `timeout`, `ended`) or a model error (`e.kind`: `load`, `run`, or `gpu` with `e.deviceLost`). After
  a model error `state.model` is `failed` and nothing is transcribed until `load()` is called.
- `onLoadProgress({ phase, fraction })` for a progress bar; `onLevel({ peak, speech })` for a level meter.

Options beyond those: `maxUtteranceSeconds` (longest audio one pass sees; default 12 on phones, 24 elsewhere; GPU
memory grows with it), `backend: "cpu"`, `voiceProcessing` (browser echo cancellation and noise suppression, off by
default: the model gets the raw microphone), `wakeLock` (default on), and `advanced` (below).

## The three layers

Each works without the others. `createDictation` is these three wired together (`src/dictation.ts`, 250 lines).

### 1. Transcriber: a clip in, text and word times out

```ts
import { createTranscriber } from "parakeet-web";

const transcriber = await createTranscriber({
  assetsUrl: "/parakeet-web/", modelUrl: "/models/parakeet-tdt_ctc-110m-Q4_0.gguf",
  onProgress: (p) => console.log(p.phase, Math.round(p.fraction * 100) + "%"),
});
const { text, words } = await transcriber.transcribe(pcm);   // Float32Array, 16 kHz mono
// words: [{ w: "going", s: 0.32, e: 0.56, p: 0.98 }, ...]   seconds from the start of pcm; p = lowest token probability
console.log(transcriber.info.backend);                        // "WebGPU: WebGPU" or "CPU"
await transcriber.dispose();
```

It owns the worker, picks the JSPI build where the browser has `WebAssembly.Suspending` and the ASYNCIFY build
otherwise, uses WebGPU where a device can be made and one WASM thread otherwise (`info.webgpu`, `info.cpuReason`),
keeps the model in OPFS, and runs a warm-up pass so shaders are compiled before the first real clip. With
`maxSeconds` it also runs one pass that long at load, so the GPU working buffer is made once. Calls to
`transcribe` are run one at a time, in order. Clips over 60 s are encoded in 30 s windows and decoded once.

Failures: `createTranscriber` rejects with an error carrying `kind` (`load`, or `gpu` with `deviceLost` and the
browser's messages in `gpuErrors`). After load, a failed pass or a GPU error (shader rejected, validation, out of
memory, device lost) rejects that `transcribe` call, calls `onError` and stops the transcriber (`state` is
`failed`): dispose it and make a new one. Pass `signal` to abort a load.

### 2. Stream: the streaming engine

```ts
import { createStream } from "parakeet-web";

const stream = createStream(
  (pcm) => transcriber.transcribe(pcm, { transfer: true }),
  { onCommit: (c) => append(c.text), onTentative: (text) => showGuess(text) },
  { capS: 12 },
);
stream.push(block);      // 16 kHz mono PCM, any block length, as it arrives
await stream.stop();     // no more audio for now: the rest is committed
```

Pure: no DOM, worker, microphone or timer. It keeps recent audio in one buffer, transcribes it again and again,
and commits a word once successive passes agree on it and enough audio follows it; an energy gate only decides
when passes are worth running. Feed it from a file, a WebRTC track, or anything else that yields 16 kHz PCM; give
it any `transcribe` function (the stand-in model in `web/replay/gate.ts` in the repository is one). The rule and
its numbers are in `docs/streaming-engine.md` there. The numbers (`StreamParams`) were tuned against recordings; change them
with a measurement in hand.

### 3. Microphone: 16 kHz blocks

```ts
import { createMicrophone } from "parakeet-web";

const mic = createMicrophone({ assetsUrl: "/parakeet-web/", onAudio: (pcm, peak) => stream.push(pcm) });
button.onclick = () => mic.start().catch((e) => alert(e.message));   // inside the tap
mic.stop();
```

`getUserMedia` (mono, auto gain on, echo cancellation and noise suppression off unless `voiceProcessing`) into an
AudioWorklet that resamples from the device rate to 16 kHz (polyphase windowed sinc) and posts 40 ms blocks. The
iOS rules are inside: the AudioContext is created and resumed synchronously in `start()`, the graph ends in a
muted gain on the output so the browser keeps pulling it, a context the browser suspends is resumed when the page
is visible again, track `ended` stops the microphone and calls `onEnded`, and a screen wake lock is held while
recording where available. `start()` rejects with a `MicrophoneError` (`code`, and a message for the person);
if the browser neither grants nor refuses within `startTimeoutMs` (default 30 s) the code is `timeout`.

### Diagnostics (`advanced`)

Everything the demo page shows and logs comes out of the library through optional callbacks; the library itself
touches no DOM (beyond audio capture and a `visibilitychange` listener while recording), no `localStorage`, and
no network other than `assetsUrl` and `modelUrl`.

- `advanced.onStep({ name, ms?, detail? })`: the step trail, one readable line per thing done (module ready,
  adapter and limits, storage path, download, load, warm-up, GPU errors, recording started).
- `advanced.onDiagnostic(kind, data)`: structured events: `adapter` (the WebGPU probe), `loaded` (`TranscriberInfo`),
  `load-heap`, `warmup-pass`, `recording`, `mic-track`, `audio-context`.
- `advanced.onPassStart` / `onPass` (dictation): every pass, with its wall time and the full `TranscribeResult`
  (`ms`, `melMs`, `encodeMs`, `decodeMs`, WASM heap, `gpu`: the worker's GPU accounting, see `GpuStats`).
- `advanced.streamEvents` (dictation): every engine event: `onGate`, `onPass`, `onUtteranceEnd`, `onSkip`
  (audio dropped because passes are slower than real time), `onCommit` with word times.
- `advanced.stream` (the engine's numbers), `advanced.transcriber` (`variant`, `f16`, `gpuLimits`, `storage`,
  `env`, `verbose`, `gpuWatch`, `gcMb`, `warmup`), `advanced.sizingPass`.
- `dictation.stats()` (lag, buffer, gate, commit reasons), `dictation.transcriber` (the loaded transcriber, for
  offline passes on the same model and `gpuStats()`), `formatGpuStats(stats)` for one line of text.

`web/src/live.ts` in the repository (the demo page) uses all of these: it is the reference for a page with a collector, a soak test and a
crash guard built on the callbacks.

## Browser support, as measured

| where | result |
|---|---|
| Chrome 154, Linux, RTX 2080 Ti (`shader-f16` through a Dawn flag, and stock f32-only before the refactor) | **Measured with this package.** WebGPU, JSPI build. Fake microphone, 110m Q4_0, 12 s cap: a pass every 250 ms takes 80-84 ms median (105-160 at the 95th percentile) on a 5.9 s buffer, text trails speech by 0.1 s median and 0.4-0.7 s at worst, GPU process 194-201 MiB, WASM heap 46 MB. An app built from the tarball with `bun build` gave the same. |
| Chrome, same box, ASYNCIFY build (`advanced.transcriber.variant`) | **Measured with this package** on the same 60 s clip: 79 / 102 ms median / p95 per pass, the same lag and memory. This is the build a browser without JSPI gets. |
| Playwright's Linux WebKit 27.2 | **Measured with this package**: `createTranscriber` + `createStream` fed from a file. No WebGPU there (`navigator.gpu` is undefined in workers), so the CPU backend, one WASM thread, JSPI build: 1.1 s per pass on a 6-7 s buffer (about 6x real time), so live text would trail by seconds. Same text as an offline pass. `getUserMedia` never answers in that build: `createMicrophone().start()` rejects with `timeout`. |
| iPhone | **Ran with this package on 2026-10-12, on WebGPU, as reported by the owner** (the demo page built on the library). No numbers from that run are recorded here. Before the library existed, the page it was lifted out of ran on an iPhone 16 Pro Max in Safari and Brave on WebGPU with the ASYNCIFY build (no JSPI in Safari before 27), passes of 30-80 ms on 110m Q4_0, as reported by its owner and its diagnostics log. |
| iOS 18.3 simulator Safari | Not run with this package. Before: no WebGPU adapter in the simulator, CPU path with the ASYNCIFY build worked. |
| Firefox | Not working when last tried (Playwright Firefox 157 exposes an adapter and refuses every device). The CPU fallback there was not tried. |
| Android | Never run. |

Without WebGPU the library still works but slowly; `info.webgpu` / `info.cpuReason` say so and the dictation's
`stats().lagS` shows by how much. When passes fall 3 s behind, words are committed without waiting for agreement;
past 30 s of backlog audio is skipped (`advanced.streamEvents.onSkip`).

### Checked for this package (11 Oct, Chrome 154 on diesel2 unless said, load average 8-13)

- Demo page, fake microphone, 60 s clip, `phone=1`, 110m Q4_0, before and after the move onto the library:
  same pass counts and times, same kind of text (the committed text of this clip is not byte-stable between runs
  either way): 207-210 passes, 79-83 ms median, 8-15 of 120 committed words differing from an offline pass.
- An app outside this repository, installed from the tarball, assets copied with `parakeet-web-assets`, bundled
  with `bun build`, dictating into a textarea: 120-121 words for the 122-word reference clip, 12-14 differing
  (case and punctuation ignored), no page error.
- Model switch while recording, and `dispose()` followed by a fresh `createDictation` five times in one page: GPU
  process memory (nvidia-smi) 184-185 MiB each time the model is ready and 45 MiB after each `dispose()`; the
  worker's accounting before the worker is stopped shows the weights freed (81.2 -> 0 MiB) and 2 small buffers
  (1 MiB) left, which go with the worker.
- `bun web/replay/gate.ts` exits 0; the replay page in Chrome gives the numbers it gave before.

## Known limits

- English only (the models are). Offline models re-run over a sliding buffer: a word is final 1.2-1.6 s after it
  was said, and punctuation and capitalization at a commit boundary can differ from what one pass over the whole
  recording would give (8-15 of 120 words differ on the 60 s test clip from run to run, case and punctuation
  ignored: "Quatro" / "Quatre", "Wrap" / "RAP", a doubled "three").
- Memory: about 185-200 MiB of GPU memory and 46 MB of WASM heap for 110m Q4_0 at a 12 s cap; 81 MB of browser
  storage for the model. The GPU working buffer is sized once for `maxUtteranceSeconds`.
- One model per transcriber, one WASM thread for mel and the decoder. Two transcribers in one page are two workers
  and two GPU devices (not measured).
- The assets folder must be same-origin. The worker is a module worker (`type: "module"`).
- A browser tab in the background: capture is paused by the browser (`microphone: "interrupted"`); nothing is
  buffered meanwhile.
- `load()` during a recording drops the words the old model had not yet committed if a pass was in flight.
- `Dictation.state.model` reads `failed` after `dispose()`.

## Not tested

- This package on a phone beyond the one iPhone run reported above (nothing was measured there); desktop Safari,
  Firefox, Android; a real microphone under measurement (every measured microphone run used Chrome's fake device
  at 44.1 kHz, so the 48 kHz resampler path, Bluetooth input and real room noise were not measured through the
  library); the wake lock; a tab going to the background while recording; track `ended`
  and `mute` events; `voiceProcessing: true`.
- Vite, webpack or any bundler other than `bun build`; a page served from a sub-path with a relative `assetsUrl`
  other than the demo's `pk/`; a cross-origin `modelUrl`.
- Device loss, and a GPU error in the middle of a recording. (A shader the browser rejects at load was checked:
  `advanced.transcriber.env: { GGML_WEBGPU_BREAK_SHADER: "soft_max" }` gives `kind: "gpu"` with the compiler's
  message, and the demo page shows it.)
- The 0.6b models and Q8_0 through `createDictation` beyond one model switch; recordings over a minute through
  the library (the engine itself was measured to 6.5 min before the move); the audio-skipping path with the real
  model; two dictations at once.
- The private-window fallback (no OPFS).

## Build

In the repository (its README has the whole of it, including how the WASM is built from upstream transcribe.cpp
plus the patches in `patches/`):

```sh
bun scripts/runtime.ts source && bun scripts/runtime.ts build   # the patched runtime and its two WASM builds, into vendor/
cd lib && bun install && bun build.ts     # dist/ (index.js, .d.ts, assets/) and parakeet-web-0.1.0.tgz beside this file
bun scripts/release.ts --push             # the built package onto the `release` branch, tagged v<version>
```

`build.ts` copies the two Emscripten builds from `vendor/build-web*/bin/` (`PK_WASM_BUILDS` overrides the
directory); prebuilt WASM is not in git on source branches. The tarball is `bun pm pack`; ESM only, types included.

Layout: `src/index.ts` (exports), `src/dictation.ts`, `src/transcriber.ts`, `src/stream.ts`, `src/microphone.ts`,
`src/runtime.ts` (worker RPC, shared types), `src/worker.ts` and `src/worklet.ts` (built into `dist/assets/`),
`bin/copy-assets.js` (`parakeet-web-assets`), `example/` (the ten-line app).
