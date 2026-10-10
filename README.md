# Parakeet in a browser tab on transcribe.cpp + ggml WebGPU

transcribe.cpp (`c63b18e2` + the patches here) with ggml's WebGPU backend, compiled to WASM
(Emscripten 6.0.12, emdawnwebgpu), running in real Chrome 154 on diesel2 (RTX 2080 Ti, driver 580,
Vulkan). Quantized GGUF weights (Q8_0, Q4_0) stay packed in GPU memory; a matmul shader reads the
blocks directly. The encoder runs on WebGPU, mel and the TDT decoder on one WASM thread. Two models:
NVIDIA `parakeet-tdt_ctc-110m` (the page default) and `parakeet-tdt-0.6b-v2`. The comparison is the
onnxruntime-web page in `../parakeet-webgpu-bench/` (same clips, same driver, same metrics).

Short answer (10 Oct, third pass; the second pass is the "before" in the tables):

- **Nothing above the WebGPU spec defaults is required any more.** The backend used to pass the
  adapter's whole limits struct to `requestDevice`; it now raises only the two byte-size limits (at
  most 1 GiB, never more than the adapter has) and gives the same text on a device created with no
  limits and no features at all. The page probes the device request from JS first, prints the
  adapter's limits and the browser's own rejection message, and shows shader compilation errors
  (pipeline label, compiler message, `getCompilationInfo` lines), validation errors and device loss
  as `GPU ERROR` steps in the persisted trail instead of aborting. All 42 WGSL sources that reach
  the device validate with naga 30 (Firefox's compiler) and translate to MSL, HLSL and SPIR-V.
  **No second browser ran the model**: Playwright's Firefox 157 exposes an adapter here but refuses
  every device; no WebKit build with WebGPU exists for Linux.
- **110M model, Q8_0, the default**: 7 s clip **62 -> 49 ms**, 56 s clip **319 -> 226 ms** (249x
  real time; stock Chrome without `shader-f16`: 48 / 240 ms), GPU memory 199 MiB after load, peak
  over the three clips **386 -> 268 MiB**, renderer RSS peak 309 MiB, reference text on all clips.
- **0.6b, Q4_0**: 7 s 104 -> **87 ms**, 56 s 784 -> **670 ms** (onnxruntime-web fp16: 145 / 753),
  GPU peak 799 -> **590 MiB**. Its 56 s encoder is still 425 ms against 232.
- Long clips: the subsampling convs run in 10 s time tiles (exact), so their activations (175 MB
  at 56 s for 0.6b) no longer grow with the clip and no tensor exceeds a 128 MiB binding at 56 s.
  What still grows is attention, quadratically (31 MB per relative-position score tensor at 56 s).

Nothing was run on a phone.

Published (tailnet only): https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-ggml-browser/

## Results

Chrome 154, headed on Xvfb, `--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan`; the "f16" rows add `--enable-dawn-features=vulkan_enable_f16_on_nvidia`, the
"stock" rows do not (adapter then has no `shader-f16`). Fresh profile per row, one discarded warm-up
then 10 warm runs, **median / worst** in ms. "enc" is everything up to the encoder output being on
the CPU (wall - mel - decode; it includes the read-back, as the ORT rows do). Third pass: 10 Oct
05:22-05:25, load average 2.5-3.0, CPU pressure avg10 under 0.8% (`results/browser/machine-load.txt`).
Second-pass rows: 03:54-03:58, load 2.7-4.1 (their JSON is in `results/browser-pass2/`); first pass
02:59-03:01, load about 4. **Warm runs repeat one clip, so the decoder's per-token memo (below) is
fully warm in them**; the decode time of the first run of each clip is given under the table.

| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ggml, 110m Q8_0**, f16 (`ggml-s8`) | 30 / 32 | 49 / 52 | 29 / 32 | 65 / 71 | 93 / 99 | 226 / 231 | 249 |
| **ggml, 110m Q8_0, stock Chrome** | 28 / 33 | 48 / 53 | 31 / 34 | 67 / 70 | 107 / 108 | 240 / 249 | 233 |
| ggml, 110m Q4_0, f16 | 24 / 31 | 43 / 50 | 28 / 30 | 66 / 68 | 102 / 103 | 238 / 243 | 236 |
| ggml, 110m Q4_0, stock Chrome | 27 / 34 | 48 / 58 | 28 / 30 | 65 / 68 | 116 / 116 | 253 / 258 | 222 |
| before (second pass): ggml, 110m Q8_0, f16 | 32 / 34 | 62 / 64 | 35 / 38 | 93 / 97 | 102 / 103 | 319 / 322 | 176 |
| before: ggml, 110m Q8_0, stock Chrome | 31 / 33 | 62 / 65 | 35 / 37 | 93 / 96 | 115 / 119 | 333 / 338 | 168 |
| ORT-web, 110m, WebGPU fp32 encoder, decoder on WASM (other README) | 117 / 190 | 174 / 277 | 131 / 153 | 228 / 250 | 134 / 141 | 498 / 517 | 113 |
| ORT-web, 110m, WebGPU fp16 encoder (needs the Chrome flag) | 125 / 147 | 192 / 220 | 123 / 153 | 220 / 259 | 121 / 145 | 503 / 527 | 112 |
| ORT-web, 110m, all WASM, int8, 1 thread | 1056 / 1064 | 1122 / 1143 | 2102 / 2421 | 2249 / 2567 | 10147 / 10550 | 10700 / 11207 | 5 |
| ORT-web, 110m, all WASM, fp32, 4 threads | 225 / 282 | 274 / 335 | 434 / 461 | 512 / 561 | 2296 / 2412 | 2586 / 2733 | 22 |
| **ggml, 0.6b Q4_0**, f16 (JSPI build) | 49 / 55 | 87 / 93 | 78 / 80 | 147 / 153 | 425 / 429 | 670 / 684 | 84 |
| **ggml, 0.6b Q4_0, stock Chrome** | 50 / 68 | 88 / 109 | 81 / 84 | 151 / 161 | 462 / 464 | 718 / 725 | 78 |
| ggml, 0.6b Q8_0, f16 | 54 / 72 | 90 / 117 | 81 / 84 | 152 / 155 | 387 / 390 | 629 / 648 | 89 |
| ggml, 0.6b Q8_0, stock Chrome | 54 / 71 | 92 / 112 | 83 / 86 | 152 / 159 | 424 / 426 | 668 / 679 | 84 |
| ggml, 0.6b F16, f16 | 56 / 90 | 116 / 173 | 89 / 93 | 214 / 226 | 498 / 504 | 908 / 967 | 62 |
| ggml, 0.6b Q4_0, ASYNCIFY build | 50 / 70 | 89 / 111 | 80 / 83 | 147 / 168 | 429 / 429 | 675 / 688 | 83 |
| before (second pass): ggml, 0.6b Q4_0, f16 | 55 / 66 | 104 / 116 | 82 / 83 | 176 / 183 | 435 / 437 | 784 / 793 | 72 |
| before (second pass): ggml, 0.6b Q4_0, stock Chrome | 54 / 59 | 106 / 116 | 84 / 87 | 179 / 183 | 474 / 476 | 813 / 833 | 69 |
| before (second pass): ggml, 0.6b Q8_0, f16 / stock | 56 / 64, 57 / 74 | 110 / 114, 109 / 123 | 85 / 87, 87 / 89 | 179 / 185, 182 / 186 | 399 / 403, 437 / 438 | 738 / 758, 782 / 789 | 76, 72 |
| before (second pass): ggml, 0.6b F16, f16 / stock (stock not rerun) | 73 / 83, 61 / 78 | 155 / 182, 149 / 163 | 93 / 94, 95 / 97 | 250 / 263, 254 / 273 | 510 / 511, 551 / 552 | 1061 / 1094, 1105 / 1138 | 53, 51 |
| first pass: ggml, 0.6b Q4_0 | 56 / 71 | 144 / 161 | 85 / 92 | 244 / 272 | 440 / 443 | 1022 / 1050 | 55 |
| first pass: ggml, 0.6b Q8_0 | 64 / 82 | 146 / 196 | 87 / 94 | 247 / 263 | 404 / 417 | 979 / 1011 | 57 |
| before: ggml, 0.6b Q4_0, stock Chrome: silently on one WASM thread | 24693 / 24737 | 24773 / 24825 | | | | | |
| before: ggml, 0.6b Q4_0 / Q8_0, flash attention on (`flash=1`) | 74 / 82, 64 / 80 | 154 / 176, 158 / 175 | 97 / 100, 99 / 102 | 250 / 273, 265 / 287 | 620 / 624, 582 / 586 | 1172 / 1218, 1151 / 1205 | 48, 49 |
| ORT-web, 0.6b, WebGPU fp16 encoder, decoder on WASM | 67 / 113 | 145 / 210 | 95 / 127 | 225 / 248 | 232 / 240 | 753 / 830 | 75 |
| ORT-web, 0.6b, WebGPU fp32 encoder | 90 / 145 | 173 / 263 | 108 / 149 | 262 / 296 | 314 / 339 | 897 / 1037 | 63 |
| ORT-web, 0.6b, all WASM, int8, 1 thread | 2083 / 2192 | 2232 / 2327 | 4200 / 4522 | 4435 / 4734 | 20223 / 20937 | 21084 / 21694 | 3 |

Where the time goes now (warm median ms; mel / encoder / decoder): 110m Q8_0 7 s 4.4 / 30 / 15,
56 s 33 / 93 / 98; 0.6b Q4_0 7 s 4.5 / 49 / 33, 56 s 37 / 425 / 206. Second pass: 110m 5.7 / 32 /
24 and 45 / 102 / 172; 0.6b 6 / 55 / 45 and 49 / 435 / 300. Decode on the **first** run of the
56 s clip (memo cold for its text): 110m 124 ms, 0.6b Q4_0 244 ms, Q8_0 228, F16 388.
Inside the 110m 56 s decode (library line, warm): enc_proj now on the GPU, pred 51, joint 40.

| Runtime | Download | Model load (bytes local to ready) | First 7 s transcribe (encoder part) | Page start to first transcript, localhost | GPU memory after load / peak | Renderer RSS after load / peak | Chrome GPU process RSS peak | WASM heap (malloc in use) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ggml 110m Q8_0** | 129 MiB | 0.32 s | 223 ms (151) | 3.1 s | **199 / 268 MiB** | **214 / 309 MiB** | 324 MiB | 67 (28) MB |
| ggml 110m Q8_0, stock Chrome | 129 MiB | 0.32 s | 261 ms (179) | 3.2 s | 198 / 268 MiB | 212 / 316 MiB | 326 MiB | 67 (28) MB |
| ggml 110m Q4_0 | 81 MiB | 0.29 s | 211 ms (146) | 2.7 s | 151 / 217 MiB | 210 / 315 MiB | 323 MiB | 68 (28) MB |
| before (second pass): ggml 110m Q8_0 | 129 MiB | 0.33 s | 214 ms (146) | 3.0 s | 199 / 386 MiB | 212 / 303 MiB | 322 MiB | 55 (16) MB |
| ORT-web 110m WebGPU fp32 | 455 MiB | 3.5 s | 520 ms (445) | 6.7 s | 495 / 1091 MiB | 1196 / 1200 MiB | 675 MiB | |
| ORT-web 110m WASM int8, 1 thread | 130 MiB | 2.2 s | 1232 ms (1133) | 5.6 s | none | 801 / 804 MiB | 253 MiB | |
| **ggml 0.6b Q4_0** | 438 MiB | 0.78 s | 279 ms (189) | 4.9 s | **497 / 590 MiB** | **211 / 324 MiB** | 335 MiB | 80 (37) MB |
| ggml 0.6b Q4_0, stock Chrome | 438 MiB | 1.13 s | 326 ms (220) | 5.4 s | 501 / 591 MiB | 249 / 324 MiB | 336 MiB | 96 (37) MB |
| ggml 0.6b Q8_0 | 696 MiB | 1.05 s | 333 ms (221) | 6.5 s | 760 / 849 MiB | 286 / 396 MiB | 404 MiB | 96 (37) MB |
| ggml 0.6b F16 | 1.2 GiB | 1.76 s | 365 ms (209) | 9.4 s | 1243 / 1383 MiB | 318 / 459 MiB | 409 MiB | 107 (59) MB |
| ggml 0.6b Q4_0 ASYNCIFY | 438 MiB | 0.94 s | 383 ms (266) | 7.5 s | 501 / 647 MiB | 221 / 564 MiB | 345 MiB | 96 (37) MB |
| before (second pass): ggml 0.6b Q4_0 | 438 MiB | 0.81 s | 313 ms (204) | 5.0 s | 501 / 799 MiB | 225 / 319 MiB | 340 MiB | 80 (26) MB |
| before (second pass): ggml 0.6b Q8_0 / F16 | 696 MiB / 1.2 GiB | 1.00 / 1.78 s | 298 / 348 ms | 6.3 / 8.8 s | 760 / 1057, 1243 / 1479 MiB | 278 / 388, 316 / 457 MiB | 408 / 407 MiB | 67 (26) / 107 (47) MB |
| first pass: ggml 0.6b Q4_0 | 438 MiB | 0.87 s | 343 ms (200) | 5.0 s | 501 / 969 MiB | 250 / 343 MiB | 341 MiB | 107 (47) MB |
| ORT-web 0.6b WebGPU fp16 | | 3.9 s | 497 ms (403) | 8.0 s | 1245 / 1544 MiB | renderer peak 2657 MiB | 1045 MiB | |
| ORT-web 0.6b WebGPU fp32 | | 6.7 s | 661 ms (544) | 16.7 s | 2423 / 2771 MiB | renderer peak 4375 MiB | 1734 MiB | |
| ORT-web 0.6b WASM int8, 1 thread | | 3.6 s | 2086 ms (1972) | 7.7-9.5 s | none | renderer peak 2341 MiB | 255 MiB | |

Peaks are over all three clips in one page; the 56 s clip sets them. The GPU peak is weights + one
compute buffer; with the time-tiled subsampling that buffer is about 55 MiB at 56 s for 0.6b (230
before) and 40 for 110m. The WASM heap in use grew by 11 MB: the decoder's per-token memo (up to
10.5 MB) and the projection read back from the GPU. A blank page with the worker and module loaded
is about 145-175 MiB of renderer RSS and 57 MiB of GPU memory.

**Second visit** (same Chrome profile kept: model in OPFS, shader cache on disk; default config,
7 s clip, `scripts/visit2.sh`): the model file is ready in **5-6 ms** instead of a download (583 ms
from localhost, 2.3-3.3 s from the Pi), load is unchanged (0.31-0.34 s: the weights are still
uploaded to the GPU), the first transcribe is **153-192 ms** instead of 232 (its encoder part
85-105 instead of 163 ms: pipelines come from Chrome's shader cache), and page start to first
transcript is 2.5 s instead of 3.1 s. Those last figures include 1.4 s of pauses the page inserts so
the driver can sample memory; without them a repeat visit is about 1.1 s to the first transcript.
Stock Chrome (f32-only shaders) second visit: first transcribe 161 ms, encoder part 91.

### Transcripts

- 0.6b: F16 and Q8_0 give the native F16 text on all three clips; Q4_0 is identical on the 7 s and
  13.7 s clips and says "E3Equity." for "E3 Equity." on the 56 s clip (the quantization: native CPU
  Q4_0 gives the same bytes). Every change in this pass kept each weight type's text byte-identical
  and stable over the warm runs, on the f16 and the f32-only path.
- 110m: reference is onnx-asr 0.12 / ONNX Runtime fp32 (`../parakeet-webgpu-bench/results/110m/`).
  The **F32 and Q8_0** GGUFs reproduce it exactly on all three clips (native and Chrome, f16 and
  stock). **Q4_0 does not**: "fortnight," for "fortnight.", a trailing comma on the 13.7 s clip, and
  on the 56 s clip "Inc, where" / "Quatro, a unique" / "Wrap Z, and" / "Unifund" (reference: "Inc.
  where", "Quatro a unique", "Wrap Z and", "unifund"). Words are the same. That is why Q8_0 is the default.

### Where the model file lives (0.6b Q4_0, 7 s clip, 3 warm runs, first pass)

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
- **Decoder is on the CPU**: `decoder.cpp` always builds its LSTM and joint graphs on a CPU backend
  (the feasibility note's "5-13 ms per token on WebGPU" was a load artifact). It was memory bound on
  fp32 mirrors of the weights, and the WASM build was missing ggml's SIMD quant kernels; both fixed
  below (0.6b 7 s: 80 -> 45 ms; 110m: 24 ms). Still one thread.
- Mel is transcribe.cpp's own (C++ in WASM): 6 ms for 7 s, 45-49 ms for 56 s.
- `shader-f16`: **optional now.** Stock Chrome on this box lacks it (NVIDIA driver 580 < 615.71);
  the page says which path is in use ("shader path: f16" / "f32 only"). On the f32-only path the
  adapter line reads "shader-f16 absent (f32-only shaders)" and the backend still reports `WebGPU`.
  Cost of not having it here: nothing on short clips, +9% on the 56 s encoder (0.6b 435 -> 474 ms,
  110m 102 -> 115 ms), no memory difference.

## What was changed in transcribe.cpp / ggml

Branch `webgpu-browser` in `~/devfs/cache/parakeet-ggml-webgpu/transcribe.cpp` (local only), exported
as `patches/00*.patch` (0001-0014; apply on `c63b18e2` with `git am`).

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

8. **`shader-f16` optional** (patch 0007). The feature is requested only when the adapter has it
   (`GGML_WEBGPU_NO_F16=1` forces the f32-only path; `=2` keeps the device feature and only swaps
   the shaders). Without it every shader is preprocessed with `NO_F16` and the macro `f16=f32`, and
   its `enable f16;` line is dropped, so workgroup staging and block scales are f32. Buffers that
   really hold half floats cannot be bound as `array<f16>` then: `supports_op` rejects F16-typed
   ops and operands, except as the weights of `mul_mat_direct`, `conv_2d` and `conv_2d_dw`, which
   read them as packed `u32` + `unpack2x16float`. `TRANSCRIBE_F32_POINTWISE=1` (the page sets it on
   this path) makes the two pre-encode pointwise convs use an F32 im2col: `ggml_conv_2d` hardcodes
   an F16 one, which would bounce to the CPU and back (+10 ms on the 7 s clip, measured).
   Not covered: the IQ quant types (their shaders use `bitcast<vec2<f16>>`), flash attention (F16
   masks; the page turns it off on this path), any graph with F16 activations.
9. **Decoder weights stay packed on the CPU** (patch 0008). The LSTM and joint matrices were
   dequantized to fp32 at load (0.6b: 26 MB walked once per emitted token). They now stay Q8_0
   blocks (a Q4_0 matrix is rewritten losslessly: same scale, nibble - 8 as int8) and the CPU
   backend's integer dot kernels run on them, as for any quantized ggml CPU model.
   `TRANSCRIBE_DECODER_F32=1` restores the old behaviour. F16/F32 models are unchanged.
10. **WASM SIMD quant kernels were never compiled** (same patch): the Emscripten toolchain reports
   `CMAKE_SYSTEM_PROCESSOR=x86`, ggml-cpu's CMake wants "wasm", so `arch/wasm/quants.c` was left out
   and every quantized dot was scalar. One-line CMake fix.
11. **Compute buffer reuse** (patch 0009). transcribe.cpp frees its graph allocator after each run;
   Dawn releases a destroyed buffer only after a later submit completes, so back-to-back runs held
   two compute buffers. The backend keeps the last freed compute buffer and gives it to the next
   allocation that fits (`GGML_WEBGPU_BUFFER_CACHE=0` off; `ggml_backend_webgpu_trim()` /
   `pk_trim` / `trim=1` on the page drop it). `GGML_WEBGPU_LOG_ALLOC=1` prints every buffer.

Third pass (patches 0010-0014):

12. **Device limits and errors** (patch 0010). `requestDevice` gets only `maxStorageBufferBindingSize`
   and `maxBufferSize`, each `min(adapter, 1 GiB)`; every other limit is left undefined, so no key
   a browser might not know is sent. Limits are then read from the device, not the adapter (a
   default device has 256-byte offset alignment whatever the adapter offers). The 1-D workgroup size
   is `min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX, 256)`. Allocations are split
   at 256 MiB, the default `maxBufferSize` (`GGML_WEBGPU_CHUNK_MB`), and the compute-buffer cache
   holds several chunks. `GGML_WEBGPU_LIMITS=default` requests nothing at all. Every pipeline is
   created inside a validation error scope: a failure is recorded with the pipeline label, the
   message and up to four `getCompilationInfo` lines. In WASM an uncaptured error or a lost device
   is recorded instead of `abort()`: `ggml_backend_webgpu_last_error()`, `pk_gpu_errors()`.
   `GGML_WEBGPU_DUMP_SHADERS=<dir>` writes each preprocessed WGSL source,
   `GGML_WEBGPU_BREAK_SHADER=<label>` makes one invalid to exercise the report.
13. **Time-tiled subsampling** (patch 0011, `TRANSCRIBE_PRE_ENCODE_TILE=<encoder frames>`, the page
   sets 128 = 10 s). The three stride-2 stages run on overlapping mel slices (16 mel frames of
   halo) and the slices' outputs are concatenated along time; the first two output frames of an
   inner slice, whose receptive field touched the slice's own zero padding, are dropped. Each kept
   frame is computed from the same inputs by the same ops. Offline, unmasked, single-utterance
   stems only; clips under 1.5 tiles are not tiled.
14. **Mel** (patch 0012): the radix-2 FFT recomputed its twiddles with a loop-carried complex
   multiply; the same recurrence now fills a table once per size (bit-identical STFT). The mel
   source is compiled with `-msimd128` in WASM.
15. **Decoder caches** (patch 0013): layer 0's `Wx @ embedding(token)` depends on the token only and
   is memoized per token id for the life of the model (computed by the same `mul_mat`, so the same
   bits; up to 10.5 MB; `TRANSCRIBE_DECODER_NO_WX_CACHE=1` off). The joint's
   `pred_w @ pred_state + pred_b` is its own graph, recomputed only when `pred_state` changed.
   The prediction-network output itself was already reused across blank steps upstream.
16. **Encoder projection on the GPU** (patch 0014, `TRANSCRIBE_ENC_PROJ_GPU=1`, set by the page):
   `enc_w @ enc_out + enc_b` is the last node of the encoder graph and the only thing read back
   for TDT/RNNT heads (plain runs: no prompt, no length masking, no dumps). Not bit-identical to
   the CPU fp32 GEMM (the direct kernel sums per 32-element block), text identical.

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

Second pass (10 Oct, 03:12-04:00). Native = `nb.sh` / `deccmp.sh` / `gpupeak.sh`; Chrome = this page, medians.

| Change | Before | After | Kept |
| --- | --- | --- | --- |
| f32-only shaders, native browser-like, encoder min ms Q4_0 / Q8_0 / F16 | 46.5 / 51.5 / 54.0 (f16) | 47.3 / 52.1 / 54.6 | yes |
| same, but the pointwise convs left on ggml_conv_2d (F16 im2col falls to the CPU, 5 splits) | 46 / 51 / 53 | 56.5 / 62 / 64 | no: `TRANSCRIBE_F32_POINTWISE=1` |
| `TRANSCRIBE_F32_POINTWISE=1` with f16 available | 46.2 / 51.0 / 53.4 | 46.4 / 51.2 / 53.6 | not set on the f16 path (no gain, larger im2col) |
| Chrome stock (no `shader-f16`), 0.6b Q4_0, 7 s total | 24773 (one WASM thread) | 148, then 106 with the decoder work | yes |
| Chrome f32-only vs f16, 56 s encoder, 0.6b Q4_0 / Q8_0 / F16; 110m Q8_0 | 435 / 399 / 510; 102 | 474 / 437 / 551; 115 | cost of no f16, not traced |
| Decoder: packed weights, native 1 thread, decode ms 7 / 13.7 / 56 s (0.6b Q4_0) | 54 / 104 / 361 | 21 / 44 / 142 | superseded |
| Decoder: packed weights in Chrome **before** the CMake fix (scalar quant kernels) | 74 / 146 / 580 | 195 / 362 / 1273 (Q4_0), 156 / 293 / 1031 (Q8_0) | no |
| WASM SIMD quant kernels compiled in (CMake fix) | 81 / 145 / 547 (fp32 mirrors) | 64 / 107 / 395 (Q4_0), 43 / 90 / 303 (Q8_0) | yes |
| Q4_0 decoder weights rewritten as Q8_0 blocks (lossless) | 64 / 107 / 395 | 48 / 85 / 293 | yes |
| WASM heap in use after the decoder change (0.6b) | 37-47 MB | 16-26 MB | yes |
| 110m natively: F32 / Q8_0 / Q4_0, 7 s encoder ms (browser-like), 1-thread decode | | 32.6 / 20.2 / 21.2, decode 25 / 10 / 10 | |
| GPU peak, 56 s clip, native (weights 437 MiB, compute buffer 230 MiB): 1 run / repeated runs | 693 / 923 MiB | 693 / 693 MiB (buffer cache) | yes |
| same, 110m Q8_0 | 438 | 295 | yes |
| Chrome GPU peak over the three clips, 0.6b Q4_0 / 110m Q8_0 | 969 / 490 MiB | 799 / 347-386 MiB | yes |
| Buffer cache, encoder time | 418 ms (56 s, native) | 419 ms | no speed effect |
| `-msimd128` for every source (transcribe.cpp too, not just ggml-cpu), Chrome: mel / decode / total, 110m 56 s | 45.4 / 172 / 319 | 49.3 / 173 / 325 (0.6b: 48.7 -> 51.8 mel, 784 -> 785 total) | no (reverted; same text) |
| `trim=1`: GPU memory 2 s after dropping the cached buffer, Chrome, 110m / 0.6b at 56 s | 351 / 743 MiB peak | 347 / 738 MiB | knob kept, **no effect seen** |
| same with a trivial queue write + submit after the destroy | 347 | 347 | no (reverted) |

Third pass (10 Oct, 04:20-05:25). Native = `scripts/nt.sh` (browser-like, 1 decoder thread, min of
3-8, text compared with `out/ref/` saved from the second-pass build for 5 models x 3 clips);
Chrome = this page, warm medians. Every kept row: text identical on all 15 model/clip pairs natively
(f16 and f32-only shaders) and in Chrome.

| Change | Before | After | Kept |
| --- | --- | --- | --- |
| Only the two byte limits requested (was: the adapter's whole limits struct) | | same text and speed | yes |
| 1-D workgroup size cap, native encoder ms, 110m 7 s / 56 s, 0.6b Q4_0 7 s / 56 s | 1024: 20.3 / 91.7 / 46.6 / 420.6 | 512: 19.4 / 84.4 / 45.0 / 410.0; **256: 19.1 / 82.3 / 44.5 / 408.9**; 128: 19.3 / 82.9 / 44.6 / 410.6; 64: 19.2 / 82.8 / 44.8 / 411.1 | 256 (the spec default) |
| same in Chrome, 110m 56 s encoder, f16 / stock | 102 / 115 | 93.5 / 105 | yes |
| Spec-default device (`GGML_WEBGPU_LIMITS=default`), native, 56 s encoder, 0.6b Q4_0 / Q8_0 / F16 | 417 / 380 / 491 (adapter limits) | **1098 / 1049 / 1181**: conv0's 184 MB output exceeds the 128 MiB binding, the op runs on the CPU; text same | fixed by the tiling below |
| Spec-default device, everything else (110m all clips, 0.6b 7 s and 13.7 s) | | same text, same speed (weights in 128 MiB chunks) | yes |
| Buffers split at 256 MiB instead of one buffer per allocation, native encoder | 19.1 / 82.2 / 44.6 / 407 | 19.0 / 82.6 / 44.6 / 411 (4096 MiB chunks) | yes (no cost) |
| Chrome, spec-default device, 110m GPU peak: one cached chunk -> chunk pool | 351 (adapter limits) | 495 with a one-buffer cache; with the pool and the tiling 304 (268 at adapter limits), native 294 vs 295 | pool |
| Error scope + `popErrorScope` per pipeline, Chrome first 7 s transcribe (cold) | 214 | 215 / 242 / 256 / 223 over four fresh profiles | yes (no measurable cost) |
| Broken shader on purpose (`env=GGML_WEBGPU_BREAK_SHADER=soft_max`), Chrome | abort | step "GPU ERROR: shader/pipeline 'soft_max_inplace' failed: Error while parsing WGSL: :160:32 error: unresolved value ..." and the run marked failed | yes |
| naga 30.0.1 on the 42 distinct WGSL sources of 110m Q8_0/Q4_0 and 0.6b Q4_0/F16, f16 and f32-only | | 0 validation failures; MSL / HLSL / SPIR-V output for all (memset needs `--override` values) | check only |
| Playwright Firefox 157 on Xvfb, WebGPU prefs on | | adapter with `shader-f16` and JSPI present, but `requestDevice({})` -> "OperationError: Not enough memory left." even with no limits and sandboxes off | could not run |
| Time-tiled stem, native GPU peak at 56 s (nvidia-smi), 110m Q8_0 / 0.6b Q4_0 | 295 / 693 MiB | 212 / 540 MiB (tiles of 32, 64, 128 frames: the same) | yes |
| same, native encoder ms 110m / 0.6b Q4_0 at 56 s | 82.2 / 407.1 | 82.6 / 402.0 | no speed effect |
| same on a spec-default device, 0.6b 56 s encoder Q4_0 / F16 | 1098 / 1181 | 409 / 482 | yes |
| same in Chrome, GPU peak over three clips, 110m / 0.6b Q4_0 | 353 / 744 MiB | 268 / 590 MiB | yes |
| Final build on a spec-default device in Chrome (`env=GGML_WEBGPU_LIMITS=default`): 110m f16, total 7 / 13.7 / 56 s; 0.6b Q4_0 stock (no f16, no limits, no features) | 110m at adapter limits: 49 / 65 / 226 | 110m 48 / 67 / 231, GPU 199 / 304 MiB; 0.6b stock 100 / 150 / 698, 56 s encoder 454 (462 at adapter limits), GPU 503 / 593 MiB; reference text | works |
| Tile 64 instead of 128 frames, Chrome 110m 13.7 s encoder (3 tiles vs none) | 33.8 | 37.1 | no: 128, and no tiling under 1.5 tiles |
| FFT twiddle table, Chrome mel ms at 56 s, 110m / 0.6b | 45.2 / 49 | 33.7 / 36.7 (native 13.6 -> 12.5) | yes |
| `-msimd128` on the mel source, Chrome STFT / mel total at 56 s | 25 / 34.5 | 22.9 / 32.6 | yes |
| hand-written f64x2 butterflies on top | 22.9 | 22.4 | no (removed) |
| Joint prediction half cached across blanks, native decode 110m 56 s | 72 | 66 | yes |
| Per-token Wx memo, native decode 56 s, 110m / 0.6b Q4_0 / F16, warm | 66 / 123 / 373 | 47 / 100 / 288 | yes |
| same, first run of a text (memo cold), native 110m pred ms | 41 | 39 (292 tokens, almost all distinct); warm 23 | gain is on repeated tokens only |
| both in Chrome, 110m total 7 s / 56 s; 0.6b Q4_0 | 62 / 309; 101 / 789 | 47.8 / 244.6; 92.5 / 698 | yes |
| enc_proj in the encoder graph, native decode 56 s 110m / 0.6b Q4_0 | 47 / 100 | 44 / 89 (encoder unchanged) | yes |
| same in Chrome, total 56 s, 110m / 0.6b Q4_0 | 244.6 / 698 | 225.9 / 672.5 | yes |

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
- **The long-clip GPU growth is the compute buffer, twice.** `GGML_WEBGPU_LOG_ALLOC=1`: one 437 MiB
  weight buffer at load, then one compute buffer per run, allocated and freed each time: 29 MiB for
  7 s, 230 MiB for 56 s (0.6b). nvidia-smi: 491 / 575 / 923 MiB for 7 / 13.7 / 56 s with repeated
  runs, 693 for a single 56 s run. So the "+470 MiB" was 2 x 230: the previous run's buffer was
  still awaiting deletion. Inside the 230 MiB the scheduler dump shows the first subsampling conv's
  output at 175 MB (256 channels x T/2 x 64 fp32), then 43 MB after the stride-2 depthwise conv;
  attention is not it.
- The decoder is mostly the LSTM. The library's own debug line (`verbose=1` shows it in the page),
  56 s clip, Chrome, one WASM thread, packed weights: 110m Q8_0 enc_proj 19 ms, pred 100, joint 50,
  confidence 3 (292 tokens, 482 joint steps); 0.6b Q4_0 enc_proj 35, pred 227, joint 41, confidence 4
  (312 tokens). Both work out to 9-10 G multiply-adds/s for the Q8_0 dot kernel
  (`ggml_vec_dot_q8_0_q8_0`, WASM SIMD), which re-extends the same activation block to i16 for every
  one of the 2560 gate rows. Native, 4 threads, 0.6b: enc_proj 7.7, pred 45, joint 8.7.
- Chrome does not hand a destroyed buffer's memory back while the page is idle: 2 s after `pk_trim`
  nvidia-smi still shows the peak, with or without a follow-up submit. So "GPU peak" in these tables
  is also what the tab holds afterwards; whether it shrinks later was not watched.
- A 1-D workgroup of 1024 invocations is slower than 256 for the element-wise shaders on this card
  (110m 56 s encoder -10%); 64 to 512 are within noise of each other.
- With spec-default limits the only thing that did not fit was conv0's output on long clips; a
  tensor above `maxStorageBufferBindingSize` silently runs on the CPU (`supports_op`), it does not fail.
- In WASM the fp64 radix-2 STFT is 22 of the 33 ms of mel at 56 s and 5x slower than native
  (4.6 ms); explicit f64x2 SIMD does not help. A single-precision real FFT would, but not bit-identically.
- The decoder's remaining cost is integer dot products: `Wh @ h` per emitted token and
  `out_w @ act` per joint step, both through `ggml_vec_dot_q8_0_q8_0` at about 10 G multiply-adds/s.
- A zsh trap that cost a wrong reading: `scripts/nb.sh $e` with `e="A=1 B=1"` passes one argument
  (zsh does not word-split), so the second variable was silently not set.

## Phone page

`web/` is the page: idle on open, one-tap presets with the smallest verified configuration first
(110m Q8_0, 129 MiB download; then 110m Q4_0, 0.6b Q4_0, Q8_0, F16), an adapter line with
`shader-f16` present / absent and JSPI present / absent, the shader path in use as a step, every
finished step written to the screen and to `localStorage` ("Earlier runs" after a reload shows the
last step a killed tab reached), model kept in OPFS. It picks the JSPI build when
`WebAssembly.Suspending` exists and the ASYNCIFY build otherwise, and the f32-only shader path
when the adapter has no `shader-f16`. No threads, so no COOP/COEP headers and no service worker.

Verified from Chrome on diesel2 against the published link, **stock Chrome with no Dawn flag**:
the default (110m Q8_0) loads from the Pi, runs on WebGPU with f32-only shaders and gives the
reference text on all three clips (third pass: 7 s total 49 ms, 13.7 s 67 ms, 56 s 239 ms, GPU
199 MiB after load, 266 peak, renderer RSS peak 324 MiB; `results/browser/published-default-stock.json`).
The model download from the Pi was 3.3 s. 0.6b Q4_0 was verified this way in the second pass only
(7 s total 108 ms; `published-q4-stock.json`).

### What the page reports when something is missing

Each of these is a step on screen and in `localStorage` (so it survives a killed tab):

- `navigator.gpu` undefined in the worker, or `requestAdapter()` null: said in those words; the model
  would then run on one WASM thread and the page warns "not on WebGPU".
- `adapter limits: binding N MiB, buffer N MiB, N invocations/workgroup, ...`, followed by any limit
  **below the WebGPU spec default** by name and value.
- The device request is tried from JS first, as a ladder: (1) the two byte limits + `shader-f16` if
  present, (2) spec-default limits, (3) no features either. Every refusal is printed with the
  browser's own error (`device request FAILED: requestDevice({...}) -> OperationError: ...`) and the
  WASM backend is configured for the first rung that worked; if none does the run stops there.
- `GPU ERROR (load|clip): ...` for a shader the browser's WGSL compiler rejects (pipeline label,
  message, `getCompilationInfo` line:column), any validation or out-of-memory error, and
  `device lost (reason N): message`. The run is then marked failed even if text came out.
- A tensor larger than the binding limit does not fail: that op runs on the CPU. The page notes it
  when the adapter's binding limit is under 256 MiB.

Checked against the spec rather than against Dawn: no Chromium-only WGSL extension or Dawn toggle
is used in the WASM build (`chromium_experimental_subgroup_matrix`, the Dawn toggles and
`ImplicitDeviceSynchronization` are native-only, behind `#ifndef __EMSCRIPTEN__`); the shaders in
use contain no `requires` directive, only `enable f16;` on the f16 path; `dot4I8Packed` is used only
if `wgslLanguageFeatures` has it (not by the shaders this model runs); at most 8 bindings per
shader; the one function-scope array that relies on implicit zero-init (`acc` in
`mul_mat_reg_tile`) is declared outside loops, which the spec guarantees, and the in-loop case that
Tint got wrong uses an explicit `array<...>()` initializer.

**Nothing was run on a phone or in Safari, and no browser other than Chrome ran the model.**

- Second WGSL compiler: naga 30.0.1 (the compiler Firefox uses) validates all 42 distinct sources
  and translates them to MSL, HLSL and SPIR-V. WebKit's WGSL compiler was not available: there is no
  Linux WebKit build with WebGPU (Playwright's WebKit and Epiphany are not installed, and WebKitGTK
  does not ship it). The nearest proxy left untried is the iOS 18.3 simulator on the Mac with the
  WebGPU feature flag, loading the published page.
- Firefox: Playwright's Firefox 157 with `dom.webgpu.enabled` exposes an adapter (limits 2048 MiB,
  `shader-f16`, JSPI available) but every `requestDevice`, including `{}`, fails with
  "OperationError: Not enough memory left." on this box (`web/drive-ff.ts`,
  `results/browser/ff-s8.json`). The page reported it as designed; the model did not run.
- JSPI: shipped in Chrome 137 and present in Firefox 157 here. One (unofficial) source says it is
  in Safari 27 beta and not in earlier Safari. The ASYNCIFY build exists for that case and is
  measured above (same speed, renderer RSS peak 564 MiB instead of 324).
- `shader-f16` on iOS Safari: not confirmed either way; it does not decide whether the page starts.
- OPFS `createSyncAccessHandle` is worker-only and documented by WebKit; `store=blob` is the fallback.
- Desktop numbers that bear on a phone: 110m Q8_0 needs about 0.2 GB of GPU memory after load
  (0.27 with a 56 s clip) in buffers of at most 256 MiB, and 0.31 GB of renderer memory at peak,
  about 0.15 GB over a blank page; 0.6b Q4_0 0.5 GB / 0.6 GB of GPU memory.

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
scripts/nt.sh [ENV=1 ...]          # native text + mel/enc/dec ms, MODELS="s8 s4 q4 q8 f16" x CLIPS="a07 a14 a56", against out/ref (SAVE=1 rewrites the references)

cd web && bun install && bun build.ts            # dist/: page, worker, wasm, clips, hard-linked GGUFs
tmux new -d -s pkg-serve 'bun serve.ts 8791'     # local server (running now in tmux session pkg-serve)
scripts/cr.sh NAME "model=q4&clip=a07&runs=5"    # one Chrome run -> results/browser/NAME.json; STOCK=1 drops the Dawn f16 flag
scripts/cs.sh NAME "model=s8&clip=all&runs=5"    # the same with a compact summary (clips, memory, error, steps matching STEPS=regex)
scripts/visit2.sh "model=s8&clip=a07&runs=3" 3   # repeat visits on one kept Chrome profile (model in OPFS, shader cache warm)
scripts/pub.sh                                   # build, publish in place (private) and verify the published link from stock Chrome
PLAYWRIGHT_BROWSERS_PATH=$R/pw-browsers bun web/drive-ff.ts ff-s8 "model=s8&clip=a07&runs=3"   # Playwright Firefox (fails at requestDevice here)
for f in $R/out/wgsl-uniq1/*.wgsl; do $R/naga/bin/naga $f /tmp/x.metal; done   # second WGSL compiler; sources from GGML_WEBGPU_DUMP_SHADERS=<dir>
scripts/deccmp.sh [MODEL_PREFIX] [THREADS]       # decoder A/B (fp32 mirrors vs packed): text + decode ms, 3 clips
$R/scripts/gpupeak.sh GGUF CLIP [ENV=1 ...]      # native GPU peak (nvidia-smi) of repeated runs; ITERS=1 for one
web/bench.sh [row ...]                           # the table rows, quiet-window wait, systemd slice
bun scripts/table.ts [row ...]                   # tables from results/browser
~/devfs/repos/kkrausse/random/scripts/deploy-artifact.sh "$PWD/web/dist" parakeet-ggml-browser   # private shelf
```

Models: 0.6b GGUFs as before; 110m from `handy-computer/parakeet-tdt_ctc-110m-gguf` (F32 and Q8_0
downloaded into `$R/gguf/`), Q4_0 made locally: `build-webgpu/bin/transcribe-quantize
gguf/parakeet-tdt_ctc-110m-F32.gguf gguf/parakeet-tdt_ctc-110m-Q4_0.gguf --quant Q4_0`.

Query string: `model=s8|s4|q4|q8|f16` (s = the 110m model), `f16=0` (f32-only shaders even where
`shader-f16` exists), `trim=1`, `clip=a07|a14|a56|all`, `runs=N`, `store=opfs|opfs-blob|blob|memfs`,
`flash=0|1`, `variant=jspi|asyncify|prof`, `verbose=1`, `env=NAME=VALUE` (repeatable, passed to
`setenv` before load: `GGML_WEBGPU_NO_DIRECT=1`, `GGML_WEBGPU_LAZY_SYNC=0`, ...), `auto=1`.
Chrome on this box has `shader-f16` only with `--enable-dawn-features=vulkan_enable_f16_on_nvidia`
(the scripts pass it unless `STOCK=1`; `bench.sh` has `-stock` rows without it).
Environment knobs added in this pass: `GGML_WEBGPU_NO_F16`, `TRANSCRIBE_F32_POINTWISE`,
`TRANSCRIBE_DECODER_F32`, `GGML_WEBGPU_BUFFER_CACHE=0`, `GGML_WEBGPU_LOG_ALLOC`.
Third pass: `GGML_WEBGPU_LIMITS=default`, `GGML_WEBGPU_WG_SIZE`, `GGML_WEBGPU_CHUNK_MB`,
`GGML_WEBGPU_DUMP_SHADERS`, `GGML_WEBGPU_BREAK_SHADER`, `TRANSCRIBE_PRE_ENCODE_TILE` (page: 128),
`TRANSCRIBE_ENC_PROJ_GPU` (page: 1), `TRANSCRIBE_DECODER_NO_WX_CACHE`. Any of them can be overridden
from the query string, e.g. `env=GGML_WEBGPU_LIMITS=default`, `env=TRANSCRIBE_PRE_ENCODE_TILE=0`.
`verbose=1` now also shows the library's `mel:` timing line beside `decoder:`.

## Not measured, not done

- Any phone, Safari, or any browser other than Chrome running the model; WebKit's WGSL compiler on
  any shader; WER (three clips only: "identical text" means these three).
- Clips longer than 56 s. The stem's memory is now bounded, attention's is not (quadratic in clip
  length), and nothing chunks long audio at the application level.
- 0.6b F16 on stock Chrome in the third pass (the row shown is the second pass).
- Decode time on unseen text beyond the first run of each of the three clips; how fast the
  per-token memo warms up in real dictation.
- 110m F32 / F16 in the browser; 110m with flash attention; whether GPU memory ever drops after
  `trim=1`; why the f32-only path costs 9-15% on the 56 s encoder.
- Backlog items 4 and 5 were not started: per-op profile of the 0.6b encoder at 56 s, op fusion,
  quantizing the F16 conv weights.
- Decoder threads (pthreads, COOP/COEP); relaxed-SIMD dot kernels; streaming.
- The flash vec/split shaders (`T_q == 1` paths) still ignore the head in the mask offset; not reached here.
- The machine was shared (load average 2.5-4); CPU-bound numbers (decode, mel, load, worst cases) are the soft ones.

## Next

In the order I would take them:

1. **A WebKit compile of the shaders.** Cheapest route found: the iOS 18.3 simulator on the Mac
   (`ios-build`), WebGPU feature flag on, opening the published link; the page's step trail and
   `GPU ERROR` steps are the output. Then a real phone.
2. **0.6b encoder matmul at 56 s** (425 vs 232 ms for onnxruntime-web): `scripts/prof.sh` with
   `CLIP=a56`; the direct kernel was tuned on the 7 s shape (last profile: `mul_mat_direct_q8_0`
   252 of 389 ms of GPU time, attention `mul_mat_reg_tile_f32_f32` 48, F16 conv 34). Not tried:
   f16 activations where available, `dot4I8Packed` with quantized activations, a Q4_K direct
   kernel, quantizing the F16 pointwise conv weights (144 MB of the 0.6b Q4_0 file).
3. **Decoder dot kernel**: a Q8_0 mat-vec that extends the activation to i16 once per vector and
   keeps the activation's block scales as fp32 (estimated 15-20% of pred + joint, which are 51 + 40
   of the 226 ms 110m 56 s total); relaxed SIMD `i32x4.relaxed_dot_i8x16_i7x16_add_s` where present.
4. **Mel**: a single-precision real FFT (22 of 33 ms at 56 s is the fp64 complex FFT). Not
   bit-identical, so it needs the text check on more than three clips.
5. **Long audio**: application-level chunking with the model's context for clips over a minute
   (attention memory is quadratic), or the library's ChunkedLimited attention if the checkpoint allows.
6. Fuse ADD/MUL/SCALE/NORM chains: the non-matmul floor is 14 ms native over ~950 dispatches (0.6b);
   for 110m at 7 s the whole encoder is 19 ms native and 30 in Chrome.
7. Fill the per-token memo for the whole vocabulary at load (1026 mat-vecs, about 175 ms in WASM)
   if first-utterance decode time matters more than load time.
8. IQ quant types and flash attention on the f32-only path, if either is ever wanted.
