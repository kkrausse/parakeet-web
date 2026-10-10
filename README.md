# Parakeet TDT 0.6b v2 in a browser tab on transcribe.cpp + ggml WebGPU

transcribe.cpp (`c63b18e2`) with ggml's WebGPU backend, compiled to WASM (Emscripten 6.0.12,
emdawnwebgpu), running in real Chrome 154 on diesel2 (RTX 2080 Ti, driver 580, Vulkan). The
quantized GGUF weights (Q8_0, Q4_0) stay packed in GPU memory; a new matmul shader reads the
blocks directly. The encoder (all 1293 nodes) runs on WebGPU, mel and the TDT decoder on WASM
(one thread). The comparison is the onnxruntime-web page in `../parakeet-webgpu-bench/`
("In the browser" in its README), same clips, same driver script, same metrics.

Short answer: it works, on the real adapter, with the native transcript. On the 7 s clip Q4_0 is as
fast as onnxruntime-web's fp16 (144 vs 145 ms total, encoder 56 vs 67 ms) with **250 MiB of
renderer memory instead of 2.4-4.4 GB**, 0.5 GB of GPU memory instead of 1.2-2.4 GB, and a model
that is ready 0.9 s after its bytes are local. On the 56 s clip it is slower than onnxruntime-web
(1.0 s vs 0.75 s). Nothing was run on a phone.

Published (tailnet only): https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-ggml-browser/

## Results

Chrome 154, headed on Xvfb, `--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan --enable-dawn-features=vulkan_enable_f16_on_nvidia`, fresh profile per row, one
discarded warm-up then 10 warm runs, **median / worst** in ms. "enc" is everything up to the encoder
output being on the CPU (wall - mel - decode; it includes the read-back, as the ORT rows do).
Pass of 10 Oct 02:59-03:01, load average about 4, CPU pressure under 1.2% (`results/browser/machine-load.txt`).

| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ggml WebGPU, Q4_0** (JSPI build) | 56 / 71 | 144 / 161 | 85 / 92 | 244 / 272 | 440 / 443 | 1022 / 1050 | 55 |
| **ggml WebGPU, Q8_0** | 64 / 82 | 146 / 196 | 87 / 94 | 247 / 263 | 404 / 417 | 979 / 1011 | 57 |
| **ggml WebGPU, F16** | 78 / 85 | 162 / 197 | 94 / 96 | 260 / 280 | 515 / 517 | 1096 / 1121 | 51 |
| ggml WebGPU, Q4_0, flash attention on (`flash=1`, after the fix below) | 74 / 82 | 154 / 176 | 97 / 100 | 250 / 273 | 620 / 624 | 1172 / 1218 | 48 |
| ggml WebGPU, Q8_0, flash attention on | 64 / 80 | 158 / 175 | 99 / 102 | 265 / 287 | 582 / 586 | 1151 / 1205 | 49 |
| ggml WebGPU, Q4_0, ASYNCIFY build | 63 / 76 | 158 / 191 | 87 / 91 | 247 / 267 | 444 / 447 | 1019 / 1041 | 55 |
| ORT-web WebGPU, fp16 encoder, decoder on WASM (9-10 Oct, other README) | 67 / 113 | 145 / 210 | 95 / 127 | 225 / 248 | 232 / 240 | 753 / 830 | 75 |
| ORT-web WebGPU, fp32 encoder | 90 / 145 | 173 / 263 | 108 / 149 | 262 / 296 | 314 / 339 | 897 / 1037 | 63 |
| ORT-web all WASM, int8, 1 thread (its smallest configuration) | 2083 / 2192 | 2232 / 2327 | 4200 / 4522 | 4435 / 4734 | 20223 / 20937 | 21084 / 21694 | 3 |
| ggml, stock Chrome (no `shader-f16`): silently on one WASM thread | 24693 / 24737 | 24773 / 24825 | | | | | |

| Runtime | Model load (bytes local to ready) | First 7 s transcribe (encoder part) | Page start to first transcript, localhost | GPU memory after load / peak | Renderer RSS after load / peak | Chrome GPU process RSS peak | WASM heap (malloc in use) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ggml Q4_0** | 0.87 s | 343 ms (200) | 5.0 s | **501** / 969 MiB | **250 / 343 MiB** | 341 MiB | 107 (47) MB |
| **ggml Q8_0** | 1.15 s | 357 ms (211) | 6.5 s | 760 / 1229 MiB | 313 / 410 MiB | 413 MiB | 107 (47) MB |
| **ggml F16** | 1.75 s | 347 ms (201) | 11.4 s | 1244 / 1709 MiB | 313 / 446 MiB | 407 MiB | 107 (47) MB |
| ggml Q4_0 ASYNCIFY | 0.84 s | 421 ms (253) | 5.1 s | 506 / 973 MiB | 247 / 580 MiB | 344 MiB | 107 (47) MB |
| ORT-web WebGPU fp16 | 3.9 s | 497 ms (403) | 8.0 s | 1245 / 1544 MiB | renderer peak 2657 MiB | 1045 MiB | |
| ORT-web WebGPU fp32 | 6.7 s | 661 ms (544) | 16.7 s | 2423 / 2771 MiB | renderer peak 4375 MiB | 1734 MiB | |
| ORT-web WASM int8, 1 thread | 3.6 s | 2086 ms (1972) | 7.7-9.5 s | none | renderer peak 2341 MiB | 255 MiB | |

Peaks are over all three clips; the 56 s clip sets them (+470 MiB of GPU memory, WASM heap
62 -> 107 MB). For the 7 s clip alone Q4_0 peaks at 541 MiB GPU and 355 MiB renderer RSS
(`published-q4.json`). A blank page with the worker and module loaded is about 145-175 MiB of
renderer RSS and 57 MiB of GPU memory, so the model costs roughly 100 MiB of renderer memory.

### Where the model file lives (Q4_0, 7 s clip, 3 warm runs)

| `store=` | Model load | Renderer RSS after load / peak |
| --- | ---: | ---: |
| `opfs` (default): streamed to OPFS, read in place through a sync access handle | 0.87 s | 250 / 343 MiB |
| `opfs-blob`: OPFS `File` mounted with Emscripten WORKERFS | 3.5 s | 247 / 876 MiB |
| `blob`: `fetch().blob()` mounted with WORKERFS | 3.2 s | 338 / 847 MiB |
| `memfs`: whole file copied into the WASM heap | 0.79 s | 1114 / 1219 MiB |

The loader already streams tensor by tensor into `ggml_backend_tensor_set`, so no load-from-buffer
path was needed: the file only has to be readable without being in the heap. WORKERFS does that
but each read is a `FileReaderSync` slice that becomes garbage (peak 850 MiB). Replacing the
WORKERFS node's `read` with `FileSystemSyncAccessHandle.read()` straight into `HEAPU8` has no
garbage and is as fast as MEMFS. The file also persists, so a second visit does not download.

### What was really in use

- Adapter: `nvidia / turing / NVIDIA GeForce RTX 2080 Ti`, `isFallbackAdapter` false; Chrome's GPU
  process holds 0.5 GB (Q4_0) to 1.24 GB (F16) in `nvidia-smi`. ggml reports backend `WebGPU: WebGPU`
  and the native scheduler dump shows all 1293 encoder nodes on it, 0 on CPU (no-flash path).
- **Decoder is on the CPU already.** The feasibility note's "5-13 ms per token on WebGPU" was a
  load artifact: `decoder.cpp` always builds its LSTM and joint graphs on a CPU backend. Native:
  58 ms for the 7 s clip on one thread, 30 ms on four. In the browser (one WASM thread, SIMD): 78-83 ms
  for 47 tokens / 88 frames, now more than half of the 7 s total. About 455 M multiply-adds; it is
  arithmetic-bound, so the remaining lever is threads (needs COOP/COEP) -- not done.
- Mel is transcribe.cpp's own (C++ in WASM): 6.5 ms for 7 s, 53 ms for 56 s.
- `shader-f16`: required (`ggml-webgpu.cpp` puts it in `required_features`; 27 of 44 shaders have
  `enable f16;`). Stock Chrome on this box lacks it (NVIDIA driver 580 < 615.71), the WebGPU device
  is then not registered and **transcribe.cpp silently falls back to the CPU backend**: correct
  text, 24.7 s for the 7 s clip, 0.94 GB renderer. The page prints a warning when the backend is not WebGPU.
- Transcripts: F16 and Q8_0 give the native F16 text on all three clips, stable over the runs.
  Q4_0 is identical on the 7 s and 13.7 s clips and says "E3Equity." for "E3 Equity." on the 56 s
  clip; native CPU Q4_0 gives byte-identical text to the browser, so that is the quantization, not the kernel.

## What was changed in transcribe.cpp / ggml

Branch `webgpu-browser` in `~/devfs/cache/parakeet-ggml-webgpu/transcribe.cpp` (local only), exported
as `patches/000*.patch` (apply on `c63b18e2` with `git am`).

1. `examples/web/`: `pk-web` Emscripten module (ES6, MODULARIZE; JSPI or ASYNCIFY per
   `GGML_WEBGPU_JSPI`) exporting `pk_load / pk_run / pk_json / pk_setenv / pk_heap_in_use`.
2. Loader (`transcribe-load-common.cpp`): 1 MiB stream buffer. libc++ refills 4 KiB at a time and
   through WORKERFS every refill is a synchronous Blob slice: the Q8_0 model took **171 s** to load
   in Chrome; 4.9 s with the buffer, 1.2 s with OPFS in-place reads.
3. **`mul_mat_direct.wgsl`**, the kernel that matters. Browsers do not have
   `chromium_experimental_subgroup_matrix`, so ggml's fast native matmul is unavailable and
   everything went through `mul_mat_reg_tile` (shared-memory tiles, scalar f16 staging, two
   `workgroupBarrier`s per k-tile), which was 82% of GPU time. The new shader has no workgroup
   memory: each invocation owns a 4x4 output tile, loads the four weight rows as one `vec4<u32>`
   straight from the packed buffer and the four activation rows as a `mat4x4<f32>`, accumulates
   integers per 32-element block and applies the block scale once. Q8_0 / Q4_0 blocks (34 / 18
   bytes) are walked in pairs, which are whole words (17 / 9), so no per-load alignment branch.
   Also handles F16 and F32 weights. `GGML_WEBGPU_NO_DIRECT=1` restores the old shader.
4. `GGML_WEBGPU_BROWSER=1` (native only): no subgroup-matrix, no Dawn "fast" toggles, so the
   native binary runs the kernels a browser gets and can be profiled with timestamp queries.
5. Lazy `synchronize` in WASM (one GPU-process round trip less per transcribe, about 7 ms).
6. **Flash attention fixed** (`flash_attn.wgsl`, `flash_attn_tile.wgsl`): the hypothesis was right.
   The mask offset used the batch index and query row only, so every head read head 0's mask; the
   CPU backend indexes `(head % mask->ne[2]) * mask->nb[2]`. New trailing param `stride_mask2`.
   The F16 `CONCAT` is still avoided with `TRANSCRIBE_F32_MASK_CONCAT=1` (the page sets it for
   `flash=1`). Native and Chrome: byte-identical text to the matmul+softmax path on all three clips
   for Q4_0 and Q8_0. **It is not faster and not smaller here**, so it stays off: without
   subgroup-matrix the flash kernel is slower (native 7 s 57 vs 51 ms, 56 s 556 vs 380 ms; Chrome
   rows above) and the GPU peak is the same to the MiB (native Q4_0 56 s: 923 vs 923).
7. `ggml_backend_sched_new`: 256-split budget in WASM. It malloc'ed 165 MB for split-input copies;
   native never touches it, the WASM heap grew from 64 to 182 MB on the first run.

## Attempt log (encoder, 7 s clip unless noted)

Native numbers are `transcribe-bench` on Dawn/Vulkan with `GGML_WEBGPU_BROWSER=1`, min of 10; text
checked each time. Chrome numbers are medians from this page.

| Change | Before | After | Kept |
| --- | --- | --- | --- |
| Baseline in Chrome (reg_tile shader) | | Q8_0 enc 154, total 262-283 | |
| Native with the subgroup-matrix kernel (not available in browsers) | | Q8_0 60, Q4_0 61 | reference |
| Native browser-like baseline | | Q8_0 140, Q4_0 137, F16 94 | |
| `mul_mat_direct` v1 (per-word unaligned loads with a branch) | 140 / 137 / 94 | 85 / 57.5 / 78 | superseded |
| Block pairs, rows loaded as `vec4<u32>`, carry word | 85 / 57.5 / 78 | 58 / 48 / 59 | yes |
| Workgroup 8x8 tiles -> 4x32 | 58 / 48 / 59 | 51 / 46.6 / 54 | yes |
| other workgroup shapes (m x n tiles): 4x16 51/55/54, 2x32 61/64/62, 1x64 85/93/86, 16x4 69/57/85, 2x16 55/52/55, 8x16 54/48/56, 8x32 57/53/63, 4x24 51.5/45/54, 8x24 57/54/63, 4x64 51/48/54, 3x32 64/61/69 | | | no |
| Wider column tile with local arrays (4x8, 4x16 outputs per invocation) | 58 / 48 / 59 | 65 / 67 / 97 and 105 / 137 / 123 | no |
| hand-unrolled 4x8 (F16) | 58 | 77 | no |
| 8 rows x 4 cols (F16 hand-written; Q8/Q4 generated) | 58; 51 / 44 | 71; 58 / 56 | no |
| 8x8 outputs (generated) | 51 / 44 | 84 / 156 | no |
| smaller tiles: 4x2 (mat2x4) 58.5 / 58; 4x1 (dot) 62 / 53.5; 2x4 (F16) 58 | | | no |
| 4x4 as two mat2x4 | 51 / 44 | 52 / 44 | no (same) |
| Dawn robustness etc. off (`GGML_WEBGPU_BROWSER=2`) | 51 / 46 | 49 / 46 | n/a in a browser |
| Matmul loop cut to one iteration (floor for everything else) | F16 58 | 14 | measurement |
| Chrome: direct kernel | Q8_0 enc 154 | 90 (Q4_0 83, F16 79) | yes |
| Chrome: lazy synchronize | total 168 | 161 | yes |
| Chrome: sched split budget | heap 182 MB | 62 MB | yes |
| Chrome: OPFS in-place reads | load 4.9 s, peak 856 MiB | 1.2 s, 419 MiB (Q8_0) | yes |

Findings worth keeping:

- After the direct kernel the Q8_0/Q4_0 matmul is about 33-40 ms of a 45-51 ms native encoder
  (1.2 T multiply-adds/s, roughly 18% of the card's FP32 peak); everything else is a 14 ms floor.
  The tile sweeps say 4x4 per invocation is a local optimum on this GPU; both bigger and smaller
  are slower, so it is not simply weight bandwidth. The remaining 2x (what subgroup-matrix gets
  natively: 40 ms of GPU for Q8_0 where reg_tile needed 126) needs a different idea.
- The matmul scales linearly with clip length (56 s: 252 of 389 ms of GPU time is
  `mul_mat_direct_q8_0`, attention `mul_mat_reg_tile_f32_f32` 48 ms, F16 conv 34 ms), which is why
  the 56 s row loses to onnxruntime-web.
- In Chrome the CPU side of an encoder run (encoding ~1300 dispatches through emdawnwebgpu) is
  only 11 ms; the rest is GPU time plus about 7-10 ms per round trip to the GPU process.
  Chrome's GPU time for the same kernels is 10-15 ms above native Dawn.
- Tint/Dawn pitfall: a `var s: array<vec4<f32>, N>;` declared inside a loop body was **not**
  re-zeroed per iteration (wrong transcript); `var s = array<vec4<f32>, N>();` is. Not reported upstream.
- Q4_K_M: Q4_K blocks have no direct kernel, so it runs the old shader (enc 137 / 213 / 765 ms). Use Q4_0.

## Phone page

`web/` is the page: idle on open, three one-tap presets with the smallest first (Q4_0, 438 MB
download), adapter line with `shader-f16` present/absent and JSPI present/absent, every finished
step written to the screen and to `localStorage` ("Earlier runs" after a reload shows the last
step a killed tab reached), model kept in OPFS. It picks the JSPI build when
`WebAssembly.Suspending` exists and the ASYNCIFY build otherwise. No threads are used, so it needs
no COOP/COEP headers and no service worker.

Verified from Chrome on diesel2 against the published link: Q4_0 with both builds loads from the
Pi, runs on WebGPU and gives the native transcript (JSPI total 165 ms, ASYNCIFY 191 ms).

**Nothing was run on a phone, Safari or Firefox.** What bears on it, from documentation only:

- JSPI: shipped in Chrome 137. One (unofficial) source says it is in Safari 27 beta and not in
  earlier Safari; no WebKit release note was found. The ASYNCIFY build exists for that case and is
  measured above (same speed class, 6.0 MB wasm instead of 4.3 MB).
- `shader-f16` on iOS Safari: not confirmed either way. Without it the page falls back to one WASM
  thread with the weights in the heap, which will not be usable on a phone.
- OPFS `createSyncAccessHandle` is worker-only and documented by WebKit; the page uses it in the
  worker. `store=blob` in the query string is the fallback to try if it fails there.
- Desktop numbers that bear on a phone: Q4_0 needs about 0.5 GB of GPU memory and 0.1 GB of
  renderer memory over a blank page for the 7 s clip; long clips add GPU memory (56 s: +470 MiB).

## Build, run, deploy

Everything heavy is in `~/devfs/cache/parakeet-ggml-webgpu/` (`transcribe.cpp/` source,
`build-web/` JSPI, `build-web-asyncify/`, `build-web-prof/` CPU-profile JSPI, `build-webgpu/` native,
`build-webgpu-prof/` native with timestamp queries, `gguf/`, `audio/`, `emsdk/`, `emdawn/`, `dawn/`).
Scripts are copied in `scripts/` here; they assume those paths.

```sh
R=~/devfs/cache/parakeet-ggml-webgpu; source $R/emsdk/emsdk_env.sh
# WASM, JSPI (ASYNCIFY: -DGGML_WEBGPU_JSPI=OFF into build-web-asyncify; profile: -DGGML_WEBGPU_CPU_PROFILE=ON into build-web-prof)
emcmake cmake -S $R/transcribe.cpp -B $R/build-web -DCMAKE_BUILD_TYPE=Release -DGGML_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON \
  -DEMDAWNWEBGPU_DIR=$R/emdawn/emdawnwebgpu_pkg -DTRANSCRIBE_BUILD_EXAMPLES=ON -DTRANSCRIBE_BUILD_TESTS=OFF \
  -DTRANSCRIBE_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF "-DPK_WEB_MEMORY_FLAGS=-sALLOW_MEMORY_GROWTH=1;-sINITIAL_MEMORY=32MB;-sMAXIMUM_MEMORY=4GB"
make -C $R/build-web -j8 pk-web
scripts/wb.sh                      # rebuild every build-web* dir and web/dist
scripts/nb.sh [ENV=1 ...]          # native rebuild + browser-like bench of Q8_0 Q4_0 F16 (QUANTS, CLIP, ITERS)
scripts/prof.sh Q8_0 [ENV=1 ...]   # per-shader GPU ms per run (native, browser-like)

cd web && bun install && bun build.ts            # dist/: page, worker, wasm, clips, hard-linked GGUFs
tmux new -d -s pkg-serve 'bun serve.ts 8791'     # local server (running now in tmux session pkg-serve)
scripts/cr.sh NAME "model=q4&clip=a07&runs=5"    # one Chrome run -> results/browser/NAME.json
web/bench.sh [row ...]                           # the table rows, quiet-window wait, systemd slice
bun scripts/table.ts [row ...]                   # tables from results/browser
~/devfs/repos/kkrausse/random/scripts/deploy-artifact.sh "$PWD/web/dist" parakeet-ggml-browser   # private shelf
```

Query string: `model=q4|q8|f16`, `clip=a07|a14|a56|all`, `runs=N`, `store=opfs|opfs-blob|blob|memfs`,
`flash=0|1`, `variant=jspi|asyncify|prof`, `verbose=1`, `env=NAME=VALUE` (repeatable, passed to
`setenv` before load: `GGML_WEBGPU_NO_DIRECT=1`, `GGML_WEBGPU_LAZY_SYNC=0`, ...), `auto=1`.
Chrome on this box needs `--enable-dawn-features=vulkan_enable_f16_on_nvidia` (the scripts pass it).

## Not measured, not done

- Any phone, Safari, Firefox; a second visit (model already in OPFS, warm shader cache); WER.
- The flash vec/split shaders (`T_q == 1` paths) still ignore the head in the mask offset; not reached here.
- The `shader-f16` dependency was not removed.
- Threads for the decoder; streaming; the smaller model.
- The machine was shared (load average about 4); CPU-bound numbers (decode, mel, load, worst cases) are the soft ones.

## Next

In the order I would take them:

1. **Long-clip GPU memory.** The +430-470 MiB at 56 s is not attention (flash on or off peaks the
   same). Not traced further; the suspects are the subsampling convolutions' activations
   (256 channels x T/2 x 64, about 184 MB per tensor at 56 s). Chunking long audio (the library's
   streaming/longform paths) is the likely answer for a phone, and it was not tried in the browser.
2. **`shader-f16`** (backlog 6): decides whether the page starts at all where the feature is missing.
   Looks tractable for quantized models with f32 activations: make the feature optional, strip
   `enable f16;` and map `f16` to `f32` in the shader preprocessor when it is absent, read F16
   weights as `u32` + `unpack2x16float` (or convert the few F16 tensors, conv weights and biases,
   to F32 at load), and reject F16-typed ops in `supports_op`. `mul_mat_direct` for Q8_0/Q4_0 needs
   f16 only for the `enable` line.
3. **Decoder threads**: 80 of 144 ms on the 7 s clip. A pthreads build needs `crossOriginIsolated`
   (the other page's `sw.js` shows how on the header-less shelf). Native says 58 -> 30 ms with four threads.
4. **Matmul, long clips**: the direct kernel is 252 of 389 ms on the 56 s clip. Ideas not tried:
   f16 activations (halves the dominant activation traffic), `dot4I8Packed` with quantized
   activations (ggml already has a `quantize_q8` path for mat-vec), a Q4_K direct kernel so
   Q4_K_M is usable, quantizing the F16 conv weights (144 MB of the Q4_0 file).
5. Fuse ADD/MUL/SCALE/NORM chains: the non-matmul floor is 14 ms native over ~950 dispatches.
6. Smaller model (backlog 8): not looked at.
