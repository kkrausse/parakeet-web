# parakeet-web

**Speech to text that runs in a browser tab, on the device, in about 200 MiB of GPU memory.**

NVIDIA Parakeet on [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) + ggml's WebGPU backend,
compiled to WASM. Dictation with text appearing as you speak, or offline transcription of a clip with word times.
No server takes part: the library sends nothing anywhere and stores only the model file (in the browser's
origin-private file system).

**Try it: <https://kkrausse.github.io/parakeet-web/>** (live dictation; 81 MB model download on the first visit,
cached after that). It has run on an iPhone in Safari and Brave on WebGPU.

- 81 MB model, about **120 MiB of GPU memory held by the library** while dictating (81 MiB of 4-bit weights +
  41 MiB of working memory), flat for as long as you talk.
- A pass every 250 ms takes about 80 ms on a desktop GPU; text trails speech by 0.1 s median.
- 2.5-5x less peak GPU memory and 4-8x less tab memory than the same models on onnxruntime-web, same clips,
  same machine ([Memory](#memory)).
- Manual: **[`lib/README.md`](lib/README.md)** (install, the model file, every option, browser support as
  measured, limits, what was not tested).

## Memory

Memory is what decides whether speech recognition can live in a tab at all, and on a phone it is the hard wall:
iOS kills a Safari tab's web content process at 2 GiB. Most of the work here went into keeping it low and flat.

### Per model

Chrome 154 on Linux, RTX 2080 Ti, benchmark page over a 7 s, a 13.7 s and a 56 s clip in one page (the 56 s clip
sets the peaks). "Held by the library" is the bytes the backend asks of the GPU device (the worker's own
accounting, the same on any device); the other columns are what the browser process uses.

| Model | Download | Held by the library: weights + working (12 s dictation cap) | GPU memory after load / peak | Tab (renderer) peak |
| --- | ---: | ---: | ---: | ---: |
| **110m Q4_0** (recommended, the demo's) | **81 MB** | **81 + 41 MiB** | **151 / 217 MiB** | 315 MiB |
| 110m Q8_0 | 129 MB | 129 + 42 MiB | 199 / 268 MiB | 309 MiB |
| 0.6b v2 Q4_0 | 438 MB | 438 + 57 MiB | 497 / 590 MiB | 324 MiB |
| 0.6b v2 Q8_0 | 696 MB | 696 + 58 MiB | 760 / 849 MiB | 396 MiB |
| 0.6b v2 F16 | 1.2 GB | | 1243 / 1383 MiB | 459 MiB |
| *onnxruntime-web, 110m WebGPU fp32* | *455 MB* | | *495 / 1091 MiB* | *1200 MiB* |
| *onnxruntime-web, 0.6b WebGPU fp16* | | | *1245 / 1544 MiB* | *2657 MiB* |
| *onnxruntime-web, 0.6b WebGPU fp32* | | | *2423 / 2771 MiB* | *4375 MiB* |

A blank page with the worker and the WASM module loaded is already about 150-175 MiB of renderer memory and
57 MiB of GPU memory, so the tab column is mostly the browser. Working memory does not depend on the weight
type. Live dictation with 110m Q4_0 through the library: Chrome's whole GPU process 194-201 MiB, WASM heap 46 MB,
renderer 252-313 MiB, unchanged from the first minute to the last of a 6.5 min recording (1428 passes) and
over a 400-pass soak test.

### What brought it down

Each line is one change, measured before and after on the same machine.

| Change | Before | After |
| --- | ---: | ---: |
| Quantized weights stay packed in GPU memory; a matmul shader reads the 4-bit / 8-bit blocks directly | 0.6b F16: 1243 MiB after load | 0.6b Q4_0: **497 MiB** |
| The model file is read in place from OPFS through a sync access handle, tensor by tensor, instead of being copied into the WASM heap (0.6b Q4_0, renderer peak) | 1219 MiB | **343 MiB** |
| The subsampling convolutions run in 10 s time tiles, and a freed compute buffer is reused instead of a second one being made (GPU peak over the three clips) | 0.6b Q4_0: 969 MiB; 110m Q8_0: 386 MiB | **590 MiB; 268 MiB** |
| Audio over 60 s is encoded in 30 s windows with 4 s of overlap (110m Q8_0, GPU peak at 2.1 / 4.4 / 8.9 min) | 616 / 1330 MiB / - | **248 / 247 / 246 MiB** |
| Mel spectrogram without whole-clip copies (WASM heap at 2.1 / 4.4 / 8.9 min) | 124 / 184 / 282 MB | **80 / 115 / 178 MB** |
| A size-classed GPU buffer pool, plus one sizing pass at start (200 live dictation passes, GPU buffers created after load) | 1958 MiB in 109 buffers | **44 MiB in 2**, working memory flat at 42 MiB |
| No giant WASM functions (clang and binaryen had inlined every ggml op into two functions of 160k and 260k lines; JavaScriptCore's optimizing compiler ate gigabytes on them). This was why the iPhone killed the tab at 2 GiB. iOS 18.3 simulator Safari, 250 passes, web content process | 3.9-4.7 GiB peak | **0.83-0.98 GiB** (CPU path, weights in the heap) |

The measurements, the attempts that did not work and the result files are in
[`docs/history/work-log.md`](docs/history/work-log.md).

## Install

```sh
bun add github:kkrausse/parakeet-web#v0.1.0
bunx parakeet-web-assets public/parakeet-web     # copies the runtime files into a folder your app serves
```

The package is not on a registry; the tag points at the `release` branch, whose tree is the built package
(index.js, types, the worker, the worklet and both WASM builds, 12 MB). Then host one model file
(`parakeet-tdt_ctc-110m-Q4_0.gguf`, 81 MB: download the demo's copy from
<https://kkrausse.github.io/parakeet-web/models/parakeet-tdt_ctc-110m-Q4_0.gguf>, or make it as the manual says) and:

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

[`lib/example/`](lib/example/) is this as a complete small app. The assets must be on the same origin as the
page; no COOP / COEP headers are needed; the microphone needs https or `localhost`.

## Layers

Each is usable alone; `createDictation` is the three wired together.

| | |
|---|---|
| `createTranscriber` | A clip in (16 kHz mono PCM), text and word times out. Owns the worker, picks the JSPI or the ASYNCIFY build, WebGPU or CPU, keeps the model in OPFS, warms up. |
| `createStream` | The streaming engine: recent audio is transcribed again and again, a word is committed once successive passes agree on it. Pure (no DOM, worker, microphone or timer). Rule and measurements: [`docs/streaming-engine.md`](docs/streaming-engine.md). |
| `createMicrophone` | `getUserMedia` into an AudioWorklet that resamples to 16 kHz and posts 40 ms blocks, with the iOS rules inside. |

## Browser support, as measured

The table with numbers is in the manual. In short:

- **Chrome 154, Linux, RTX 2080 Ti**: measured with this package, WebGPU, both the JSPI and the ASYNCIFY build:
  a pass every 250 ms takes about 80 ms, text trails speech by 0.1 s median.
- **iPhone**: the library build ran on WebGPU on an iPhone on 2026-10-12, as reported by the owner; before the
  library, the same runtime ran there in Safari and Brave with passes of 30-80 ms. No numbers from the library
  run are recorded here.
- **Playwright's Linux WebKit 27.2**: works on the CPU backend (no WebGPU there), about 6x real time per pass;
  no microphone in that build.
- **Firefox**: not working when last tried. **Android**: never run.

Limits (English only, a word is final 1.2-1.6 s after it was said, same-origin assets) and the list of what was
**not tested** are in the manual: [Known limits](lib/README.md#known-limits), [Not tested](lib/README.md#not-tested);
the engine's own list is at the end of [`docs/streaming-engine.md`](docs/streaming-engine.md).

## Layout

```
lib/            the package: src/, bin/ (parakeet-web-assets), build.ts, README.md (the manual), example/
web/            demo and tests: live.html (dictation page), index.html (benchmark page), replay.html + replay/
                (the engine against recordings), libtest.html, drive-*.ts (Playwright drivers), env.ts (paths, port)
scripts/        runtime.ts (patched source + WASM), release.ts, pages.ts, lv.sh and other run helpers
patches/        the runtime changes, 0001-0024, as `git format-patch` files
runtime.json    which upstream commit the patches apply to, and the tree they must produce
docs/           streaming-engine.md
results/        result files the docs cite, and where the drivers write
vendor/ cache/  git-ignored: the patched runtime source and its builds; browser profiles and scratch
```

## The runtime

The WASM is built from **upstream [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT) at
commit `c63b18e2b4292760f71721ad86778e721c92ebeb` + the patches in [`patches/`](patches/)** (24: the WebGPU
backend made to run in browsers and on phones, a browser module `pk-web`, decoder and mel speed-ups, long-audio
chunking, GPU accounting, word times). [`runtime.json`](runtime.json) is the one place that names the upstream
repository, the base commit and the git tree hash base + patches must give.

```sh
bun scripts/runtime.ts source     # clone upstream at the pinned commit into vendor/transcribe.cpp, `git am` the patches, check the tree hash
bun scripts/runtime.ts build      # emcmake + make: vendor/build-web (JSPI) and vendor/build-web-asyncify; `build prof` adds the CPU-profile build
bun scripts/runtime.ts verify     # is vendor/transcribe.cpp exactly base + patches?
```

`build` needs the Emscripten SDK (6.0.12 was used) at `$PK_CACHE/emsdk` and Dawn's `emdawnwebgpu` package at
`$PK_CACHE/emdawn/emdawnwebgpu_pkg` (`PK_CACHE` defaults to `~/devfs/cache/parakeet-ggml-webgpu`); it does not
fetch them. What it runs, per build:

```sh
source $PK_CACHE/emsdk/emsdk_env.sh
emcmake cmake -S vendor/transcribe.cpp -B vendor/build-web -DCMAKE_BUILD_TYPE=Release -DGGML_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON \
  -DEMDAWNWEBGPU_DIR=$PK_CACHE/emdawn/emdawnwebgpu_pkg -DTRANSCRIBE_BUILD_EXAMPLES=ON -DTRANSCRIBE_BUILD_TESTS=OFF \
  -DTRANSCRIBE_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF "-DPK_WEB_MEMORY_FLAGS=-sALLOW_MEMORY_GROWTH=1;-sINITIAL_MEMORY=32MB;-sMAXIMUM_MEMORY=4GB" \
  -DCMAKE_C_FLAGS=-ffile-prefix-map=$PWD/vendor/transcribe.cpp/=transcribe.cpp/ -DCMAKE_CXX_FLAGS=-ffile-prefix-map=$PWD/vendor/transcribe.cpp/=transcribe.cpp/
make -C vendor/build-web -j8 pk-web          # ASYNCIFY: -DGGML_WEBGPU_JSPI=OFF into vendor/build-web-asyncify
```

**Adding a runtime change**: commit it in `vendor/transcribe.cpp` (branch `parakeet-web`, on top of the others),
then `bun scripts/runtime.ts export` rewrites `patches/` from those commits and updates the tree hash in
`runtime.json`; commit both here. To move to a newer upstream: rebase that branch onto the new commit, put the new
hash in `runtime.json` as `base`, export.

## Build, run, test

Models and test audio are not in git; they are read from `$PK_CACHE` (`gguf/`, `audio/`). The local server's port
is `PK_PORT` (default 8796) and browsers open on the X display `PK_DISPLAY` (default `:99`); `web/env.ts` is the
one place that knows these. Browser profiles and scratch go to `cache/` here, not `/tmp`.

```sh
bun scripts/runtime.ts source && bun scripts/runtime.ts build      # once, and after a runtime change
(cd lib && bun install); cd web && bun install && bun build.ts     # everything: lib/dist + tarball, then web/dist (pages, assets in pk/, clips, hard-linked models) and web/dist-live
(cd lib && bun build.ts)                                           # the library alone
(cd web && bun serve.ts --no-coi)                                  # http://127.0.0.1:8796/live.html ; the drivers default to this server

bun web/replay/gate.ts [--clip iv90|ivgap] [--verbose]             # the engine against the old endpointer, stand-in model, no browser; exit 1 = the engine lost words
scripts/lv.sh NAME t56 "phone=1&model=s4&diag=0"                   # one fake-microphone run of the demo page in Chrome -> results/live/NAME.json
python3 scripts/lsum.py NAME [NAME ...]                            # one line per run
URL=https://kkrausse.github.io/parakeet-web/ STOCK=1 scripts/lv.sh NAME t56 "phone=1"   # the same against a published copy
bun web/drive-ready.ts URL [--browser webkit|chrome]               # does a copy of the live page reach "ready", on which backend, fetching the model from where
bun web/drive-replay.ts NAME [--clip iv90] [--conditions clean,phone] [--engines old,new] [--browser webkit]   # real model on recordings -> results/replay/NAME.json
bun web/drive-lib.ts lifecycle NAME | webkit NAME                  # the API: model switch, dispose and recreate; transcriber + stream in WebKit -> results/lib/
bun web/drive-consumer.ts NAME --wav $PK_CACHE/audio/live/t56.wav --expect $PK_CACHE/audio/live/t56.txt --url http://127.0.0.1:8795/   # an app built on the package, from outside -> results/consumer/
bun web/drive-soak.ts NAME "model=s4&phone=1" --passes 300         # soak test -> results/soak/
bun web/drive-diag.ts foreign|kill                                 # error attribution and the crash guard of the demo page
scripts/cr.sh NAME "model=s4&clip=a07&runs=5"                      # benchmark page, one Chrome run -> results/browser/NAME.json (cs.sh: with a summary)
python3 scripts/live-clips.py; python3 scripts/replay-clips.py     # make the test audio in $PK_CACHE/audio/
```

Chrome on the machine this was measured on has `shader-f16` only with a Dawn flag, which the scripts pass unless
`STOCK=1`. Linux WebKit has no WebGPU and no fake microphone; Chrome with the fake microphone is the real check.
`scripts/` also holds the native-build helpers the runtime work used (`nb.sh`, `nt.sh`, `lt.sh`, `prof.sh`, ...);
they expect a native build under `$PK_CACHE` and are documented in the work log, not here.

## Release

```sh
# bump "version" in lib/package.json, commit, then:
bun scripts/release.ts --push
```

builds the package, commits exactly its files as the next commit of the `release` branch (which shares no history
with `main`), tags it `v<version>` and pushes both. Apps then `bun add github:kkrausse/parakeet-web#v<version>`.

## Demo page on GitHub Pages

```sh
bun scripts/pages.ts build        # dist-pages/: the live page alone, one model (110m Q4_0) beside it, no diagnostics collector
bun scripts/pages.ts publish      # dist-pages/ becomes the single commit of gh-pages, pushed
```

The public build posts nothing anywhere (the demo's diagnostics collector is compiled out) and `build` refuses to
finish if a private host name or a machine path is in any shipped file. The model ships inside the branch (81 MB, under GitHub's 100 MB file limit) because no
Q4_0 file is published on Hugging Face (`handy-computer/parakeet-tdt_ctc-110m-gguf` has F32, F16, Q8_0, Q4_K_M,
Q5_K_M and Q6_K; Q4_K has no direct kernel in this runtime), and same-origin needs no CORS. Checked from a local
server under `/parakeet-web/`: Chrome with the fake microphone, and Linux WebKit to "ready" on the CPU backend.
Pages serves the `gh-pages` branch at <https://kkrausse.github.io/parakeet-web/>.

## Credits and licences

- Models: NVIDIA Parakeet ([parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m),
  [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)), **CC BY 4.0**: attribution is
  required wherever a model file is redistributed, with a note of changes. The GGUF conversions are
  [handy-computer's](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf); the Q4_0 file the demo
  ships was quantized from their F32 file with transcribe.cpp's quantizer. The demo page carries this notice.
- Runtime: transcribe.cpp and ggml, MIT. Their licence texts, with Emscripten's and Dawn's, are built into
  `dist/assets/LICENSES.txt`, which travels with the WASM files.
- This repository's own code has no licence file yet.

## History

Twenty-seven commits of this work came along from where it started: the `parakeet-ggml-web/` directory of
`kkrausse/random`, on branches that were deleted there on 12 Oct 2026. The benchmark history, the attempt log, the
phone investigation and the hand-off notes (a 128 KB README), the benchmark result files, and the earlier native /
onnxruntime-web benchmark branch are copied into [`docs/history/`](docs/history/README.md).
