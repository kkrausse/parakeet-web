# Parakeet in a browser tab on transcribe.cpp + ggml WebGPU

transcribe.cpp (`c63b18e2` + the patches here) with ggml's WebGPU backend, compiled to WASM
(Emscripten 6.0.12, emdawnwebgpu), running in real Chrome 154 on diesel2 (RTX 2080 Ti, driver 580,
Vulkan). Quantized GGUF weights (Q8_0, Q4_0) stay packed in GPU memory; a matmul shader reads the
blocks directly. The encoder runs on WebGPU, mel and the TDT decoder on one WASM thread. Two models:
NVIDIA `parakeet-tdt_ctc-110m` (the page default) and `parakeet-tdt-0.6b-v2`. The comparison is the
onnxruntime-web page in `../parakeet-webgpu-bench/` (same clips, same driver, same metrics).

Short answer (10 Oct, second pass):

- **It starts in stock Chrome now.** `shader-f16` is optional: without it the backend compiles
  f32-only shaders. Stock Chrome on this box (no Dawn flag, no `shader-f16`) went from 24.7 s on one
  WASM thread to 106 ms for the 7 s clip (0.6b Q4_0), same text, same memory as the f16 path.
- **110M model, Q8_0, the default**: 135 MB download, 7 s clip in **62 ms**, 56 s clip in **319 ms**
  (176x real time), 199 MiB of GPU memory after load, 303 MiB renderer RSS peak, and the text of the
  fp32 reference on all three clips. onnxruntime-web with the same model: 174 ms / 498 ms on WebGPU
  fp32 (1.2 GB renderer), 1122 ms / 10.7 s on WASM int8 (0.8 GB).
- **0.6b, Q4_0**: 7 s clip 144 -> **104 ms**, 56 s clip 1022 -> **784 ms** (onnxruntime-web fp16:
  145 / 753 ms), GPU peak 969 -> 799 MiB, renderer RSS peak 319 MiB (onnxruntime-web: 2657).
  The decoder went from 80 to 45 ms on the 7 s clip; the 56 s encoder is still 435 ms against 232.

Nothing was run on a phone.

Published (tailnet only): https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-ggml-browser/

## Results

Chrome 154, headed on Xvfb, `--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
--use-angle=vulkan`; the "f16" rows add `--enable-dawn-features=vulkan_enable_f16_on_nvidia`, the
"stock" rows do not (adapter then has no `shader-f16`). Fresh profile per row, one discarded warm-up
then 10 warm runs, **median / worst** in ms. "enc" is everything up to the encoder output being on
the CPU (wall - mel - decode; it includes the read-back, as the ORT rows do). Pass of 10 Oct
03:54-03:58, load average 2.7-4.1, CPU pressure avg10 under 0.4% (`results/browser/machine-load.txt`).
"Before" rows are the first pass (02:59-03:01, same method, load about 4).

| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ggml, 110m Q8_0**, f16 (`ggml-s8`) | 32 / 34 | 62 / 64 | 35 / 38 | 93 / 97 | 102 / 103 | 319 / 322 | 176 |
| **ggml, 110m Q8_0, stock Chrome** | 31 / 33 | 62 / 65 | 35 / 37 | 93 / 96 | 115 / 119 | 333 / 338 | 168 |
| ggml, 110m Q4_0, f16 | 30 / 34 | 62 / 69 | 37 / 39 | 94 / 97 | 111 / 134 | 331 / 354 | 169 |
| ggml, 110m Q4_0, stock Chrome | 31 / 34 | 63 / 69 | 37 / 39 | 95 / 98 | 124 / 125 | 341 / 350 | 164 |
| ORT-web, 110m, WebGPU fp32 encoder, decoder on WASM (other README) | 117 / 190 | 174 / 277 | 131 / 153 | 228 / 250 | 134 / 141 | 498 / 517 | 113 |
| ORT-web, 110m, WebGPU fp16 encoder (needs the Chrome flag) | 125 / 147 | 192 / 220 | 123 / 153 | 220 / 259 | 121 / 145 | 503 / 527 | 112 |
| ORT-web, 110m, all WASM, int8, 1 thread | 1056 / 1064 | 1122 / 1143 | 2102 / 2421 | 2249 / 2567 | 10147 / 10550 | 10700 / 11207 | 5 |
| ORT-web, 110m, all WASM, fp32, 4 threads | 225 / 282 | 274 / 335 | 434 / 461 | 512 / 561 | 2296 / 2412 | 2586 / 2733 | 22 |
| **ggml, 0.6b Q4_0**, f16 (JSPI build) | 55 / 66 | 104 / 116 | 82 / 83 | 176 / 183 | 435 / 437 | 784 / 793 | 72 |
| **ggml, 0.6b Q4_0, stock Chrome** | 54 / 59 | 106 / 116 | 84 / 87 | 179 / 183 | 474 / 476 | 813 / 833 | 69 |
| ggml, 0.6b Q8_0, f16 | 56 / 64 | 110 / 114 | 85 / 87 | 179 / 185 | 399 / 403 | 738 / 758 | 76 |
| ggml, 0.6b Q8_0, stock Chrome | 57 / 74 | 109 / 123 | 87 / 89 | 182 / 186 | 437 / 438 | 782 / 789 | 72 |
| ggml, 0.6b F16, f16 | 73 / 83 | 155 / 182 | 93 / 94 | 250 / 263 | 510 / 511 | 1061 / 1094 | 53 |
| ggml, 0.6b F16, stock Chrome | 61 / 78 | 149 / 163 | 95 / 97 | 254 / 273 | 551 / 552 | 1105 / 1138 | 51 |
| ggml, 0.6b Q4_0, ASYNCIFY build | 54 / 68 | 111 / 121 | 83 / 85 | 179 / 182 | 440 / 442 | 780 / 788 | 72 |
| before: ggml, 0.6b Q4_0 | 56 / 71 | 144 / 161 | 85 / 92 | 244 / 272 | 440 / 443 | 1022 / 1050 | 55 |
| before: ggml, 0.6b Q8_0 | 64 / 82 | 146 / 196 | 87 / 94 | 247 / 263 | 404 / 417 | 979 / 1011 | 57 |
| before: ggml, 0.6b Q4_0, stock Chrome: silently on one WASM thread | 24693 / 24737 | 24773 / 24825 | | | | | |
| before: ggml, 0.6b Q4_0 / Q8_0, flash attention on (`flash=1`) | 74 / 82, 64 / 80 | 154 / 176, 158 / 175 | 97 / 100, 99 / 102 | 250 / 273, 265 / 287 | 620 / 624, 582 / 586 | 1172 / 1218, 1151 / 1205 | 48, 49 |
| ORT-web, 0.6b, WebGPU fp16 encoder, decoder on WASM | 67 / 113 | 145 / 210 | 95 / 127 | 225 / 248 | 232 / 240 | 753 / 830 | 75 |
| ORT-web, 0.6b, WebGPU fp32 encoder | 90 / 145 | 173 / 263 | 108 / 149 | 262 / 296 | 314 / 339 | 897 / 1037 | 63 |
| ORT-web, 0.6b, all WASM, int8, 1 thread | 2083 / 2192 | 2232 / 2327 | 4200 / 4522 | 4435 / 4734 | 20223 / 20937 | 21084 / 21694 | 3 |

Where the time goes now (median ms; mel / encoder / decoder): 110m Q8_0 7 s 5.7 / 32 / 24, 56 s
45 / 102 / 172; 0.6b Q4_0 7 s 6 / 55 / 45, 56 s 49 / 435 / 300. The decoder is the largest part of
the 110m long clip, the encoder matmul of the 0.6b one.

| Runtime | Download | Model load (bytes local to ready) | First 7 s transcribe (encoder part) | Page start to first transcript, localhost | GPU memory after load / peak | Renderer RSS after load / peak | Chrome GPU process RSS peak | WASM heap (malloc in use) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ggml 110m Q8_0** | 129 MiB | 0.33 s | 214 ms (146) | 3.0 s | **199** / 386 MiB | **212 / 303 MiB** | 322 MiB | 55 (16) MB |
| ggml 110m Q8_0, stock Chrome | 129 MiB | 0.34 s | 217 ms (147) | 2.8 s | 199 / 347 MiB | 213 / 302 MiB | 322 MiB | 55 (16) MB |
| ggml 110m Q4_0 | 81 MiB | 0.30 s | 219 ms (148) | 2.7 s | 151 / 299 MiB | 212 / 306 MiB | 320 MiB | 55 (16) MB |
| ORT-web 110m WebGPU fp32 | 455 MiB | 3.5 s | 520 ms (445) | 6.7 s | 495 / 1091 MiB | 1196 / 1200 MiB | 675 MiB | |
| ORT-web 110m WASM int8, 1 thread | 130 MiB | 2.2 s | 1232 ms (1133) | 5.6 s | none | 801 / 804 MiB | 253 MiB | |
| **ggml 0.6b Q4_0** | 438 MiB | 0.81 s | 313 ms (204) | 5.0 s | **501** / 799 MiB | **225 / 319 MiB** | 340 MiB | 80 (26) MB |
| ggml 0.6b Q4_0, stock Chrome | 438 MiB | 0.80 s | 309 ms (211) | 4.5 s | 501 / 799 MiB | 233 / 317 MiB | 344 MiB | 80 (26) MB |
| ggml 0.6b Q8_0 | 696 MiB | 1.00 s | 298 ms (196) | 6.3 s | 760 / 1057 MiB | 278 / 388 MiB | 408 MiB | 67 (26) MB |
| ggml 0.6b F16 | 1.2 GiB | 1.78 s | 348 ms (202) | 8.8 s | 1243 / 1479 MiB | 316 / 457 MiB | 407 MiB | 107 (47) MB |
| ggml 0.6b Q4_0 ASYNCIFY | 438 MiB | 0.83 s | 336 ms (225) | 6.5 s | 502 / 799 MiB | 212 / 622 MiB | 342 MiB | 80 (26) MB |
| before: ggml 0.6b Q4_0 | 438 MiB | 0.87 s | 343 ms (200) | 5.0 s | 501 / 969 MiB | 250 / 343 MiB | 341 MiB | 107 (47) MB |
| ORT-web 0.6b WebGPU fp16 | | 3.9 s | 497 ms (403) | 8.0 s | 1245 / 1544 MiB | renderer peak 2657 MiB | 1045 MiB | |
| ORT-web 0.6b WebGPU fp32 | | 6.7 s | 661 ms (544) | 16.7 s | 2423 / 2771 MiB | renderer peak 4375 MiB | 1734 MiB | |
| ORT-web 0.6b WASM int8, 1 thread | | 3.6 s | 2086 ms (1972) | 7.7-9.5 s | none | renderer peak 2341 MiB | 255 MiB | |

Peaks are over all three clips in one page; the 56 s clip sets them. The GPU peak is weights + one
compute buffer (0.6b at 56 s: 230 MiB, of which the first subsampling conv's output is 175 MB) plus
whatever is in transit when the clip length changes; before the buffer cache it was weights + two.
A blank page with the worker and module loaded is about 145-175 MiB of renderer RSS and 57 MiB of
GPU memory.

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
reference text on all three clips (7 s total 68 ms, 56 s 335 ms, GPU 199 MiB after load, 349 peak,
renderer RSS peak 309 MiB; `results/browser/published-default-stock.json`); 0.6b Q4_0 the same way
(7 s total 108 ms, GPU peak 541 MiB; `published-q4-stock.json`). The model download from the Pi was 2.3 s.

**Nothing was run on a phone, Safari or Firefox.** What bears on it, from documentation only:

- JSPI: shipped in Chrome 137. One (unofficial) source says it is in Safari 27 beta and not in
  earlier Safari; no WebKit release note was found. The ASYNCIFY build exists for that case and is
  measured above (same speed, 6.0 MB wasm instead of 4.3 MB, renderer RSS peak 622 MiB instead of 319).
- `shader-f16` on iOS Safari: not confirmed either way, and it no longer decides whether the page starts.
  The f32-only shaders rely on `unpack2x16float` and on nothing outside core WGSL; they were compiled
  by Tint only, never by WebKit's WGSL compiler.
- OPFS `createSyncAccessHandle` is worker-only and documented by WebKit; the page uses it in the
  worker. `store=blob` in the query string is the fallback to try if it fails there.
- Desktop numbers that bear on a phone: 110m Q8_0 needs about 0.2 GB of GPU memory after load
  (0.35-0.39 with a 56 s clip) and 0.3 GB of renderer memory at peak, about 0.15 GB over a blank
  page; 0.6b Q4_0 0.5 GB / 0.8 GB of GPU memory.

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
scripts/cr.sh NAME "model=q4&clip=a07&runs=5"    # one Chrome run -> results/browser/NAME.json; STOCK=1 drops the Dawn f16 flag
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

## Not measured, not done

- Any phone, Safari, Firefox; WebKit's WGSL compiler on the f32-only shaders; a second visit (model
  already in OPFS, warm shader cache); WER (three clips only: "identical text" means these three).
- 110m F32 / F16 in the browser (native F32 only); 110m with flash attention; whether GPU memory
  ever drops after `trim=1` beyond the 2 s that were watched.
- Why the f32-only path costs 9% on the 56 s encoder (suspects: f32 workgroup staging in
  `mul_mat_reg_tile` for attention, the F32 im2col of the pointwise convs).
- Decoder threads (pthreads, COOP/COEP); relaxed-SIMD dot kernels; streaming.
- The flash vec/split shaders (`T_q == 1` paths) still ignore the head in the mask offset; not reached here.
- The machine was shared (load average 3-4); CPU-bound numbers (decode, mel, load, worst cases) are the soft ones.

## Next

In the order I would take them:

1. **Decoder, long clips** (172 of 319 ms for 110m at 56 s; 300 of 784 for 0.6b). Not tried:
   `enc_proj` (T x d_enc x joint_h fp32 GEMM on one WASM thread, 460 M multiply-adds for 0.6b at
   56 s) as a last node of the encoder graph on the GPU, so only the projection is read back;
   caching `pred_w @ pred_out` across blank steps (190 of 482 joint steps at 56 s for 110m); a
   mat-vec routine for the Q8_0 LSTM matrices that extends the activation once instead of per row;
   a faster FFT for mel (45 ms at 56 s, 14% of the 110m total; `-msimd128` on its own did nothing);
   pthreads if a COOP/COEP path is acceptable.
2. **The 230 MiB compute buffer** for long clips: it is the first subsampling conv's output
   (175 MB at 56 s for 0.6b, 110 MB for 110m). Running conv0 -> ReLU -> depthwise conv2 in time
   tiles inside the graph would be exact and cap it; chunked encoding would not be exact.
3. **0.6b encoder matmul, long clips** (435 vs 232 ms for onnxruntime-web at 56 s): the direct
   kernel is 252 of 389 ms of GPU time. Not tried: f16 activations where available,
   `dot4I8Packed` with quantized activations, a Q4_K direct kernel, quantizing the F16 pointwise
   conv weights (144 MB of the 0.6b Q4_0 file; they already go through `mul_mat_direct`, so a
   requantized file may simply work).
4. Fuse ADD/MUL/SCALE/NORM chains: the non-matmul floor is 14 ms native over ~950 dispatches (0.6b);
   for 110m at 7 s the whole encoder is 20 ms native and 32 in Chrome, so dispatch count is the next thing there.
5. IQ quant types and flash attention on the f32-only path, if either is ever wanted.
