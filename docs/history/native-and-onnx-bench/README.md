# Parakeet 0.6b on native WebGPU runtimes vs the dictation service

How fast does NVIDIA Parakeet 0.6b run through a native (non-browser) WebGPU runtime on diesel2
(RTX 2080 Ti, driver 580, 12 cores), compared with what `../dictation-server-linux` uses today
(transcribe.cpp 0.3.1, ggml, Vulkan)? Measured 2026-10-09. Benchmark code only. The same model in
an actual browser (Chrome, onnxruntime-web) is in [In the browser](#in-the-browser), added the next day,
and the small 110m model in the browser is in [The small model](#the-small-model-parakeet-tdt_ctc-110m).

Short answer: ONNX Runtime's WebGPU provider (Dawn on Vulkan) works and really runs on the GPU, at
40–100x real time depending on clip length. transcribe.cpp on Vulkan is 2–3x faster on the
encoder, is ready 2.5–3.5x sooner and is what the service already has. Burn on wgpu imports and runs the encoder
correctly but is 10x slower than ONNX Runtime WebGPU.

## Results

Same model everywhere unless noted: `parakeet-tdt-0.6b-v2`, as istupakov's ONNX export (fp32) and
handy-computer's GGUF. Same three clips, raw 16 kHz mono f32: the first 7.0 s of the service's
LibriSpeech fixture, the whole 13.7 s fixture, and transcribe.cpp's `product-names.wav` (56.1 s).
One discarded warm-up, then 10 runs; cells are **median / worst** in ms. "enc" is the encoder
alone, "total" is mel + encoder + decode for the whole clip.

| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT 1.28 WebGPU EP (Dawn/Vulkan), fp32 encoder, decoder on CPU | 71 / 75 | 174 / 189 | 91 / 98 | 221 / 266 | 265 / 270 | 671 / 781 | 84 |
| same, fp16 encoder | 58 / 105 | 144 / 176 | 66 / 69 | 157 / 172 | 208 / 211 | 557 / 713 | 101 |
| same, fp32, decoder on WebGPU too | 46 / 47 | 427 / 445 | 63 / 64 | 749 / 772 | 261 / 265 | 2862 / 2978 | 20 |
| ORT 1.28 CPU EP, fp32, 12 cores | 432 / 497 | 494 / 554 | 1006 / 1311 | 1173 / 1523 | 4890 / 5250 | 5473 / 6021 | 10 |
| Burn 0.22, `wgpu` feature, fp32, encoder only | 543 / 576 | | 824 / 838 | | 2751 / 2758 | | |
| Burn 0.22, `vulkan` feature (SPIR-V), fp32, encoder only | 474 / 481 | | 742 / 744 | | 2535 / 2573 | | |
| transcribe.cpp 0.3.1 Vulkan, TDT v2 F16 GGUF | 23 / 23 | 76 / 124 | 32 / 33 | 117 / 121 | 84 / 103 | 388 / 435 | 145 |
| transcribe.cpp 0.3.1 Vulkan, TDT v2 F32 GGUF | 37 / 40 | 89 / 93 | 34 / 34 | 120 / 126 | 84 / 84 | 381 / 513 | 147 |
| transcribe.cpp 0.3.1 Vulkan, unified F16 GGUF, offline (the service's weights) | 25 / 25 | 80 / 99 | 34 / 36 | 125 / 139 | 86 / 120 | 442 / 564 | 127 |

| Runtime | Process start to model loaded | First 7 s transcribe in the process (encoder part) | GPU memory after load / peak | RSS peak |
| --- | ---: | ---: | ---: | ---: |
| ORT WebGPU, fp32 | 4.4 s (encoder session 4.0 s) | 382 ms (282) | 2376 / 2721 MiB | 375 MiB |
| ORT WebGPU, fp16 | 3.1 s (2.8 s) | 531 ms (474) | 1196 / 1494 MiB | 383 MiB |
| ORT WebGPU, fp32, decoder on WebGPU | 4.0 s | 640 ms (184) | 2417 / 2938 MiB | 338 MiB |
| ORT CPU, fp32 | 3.2 s | 493 ms (436) | none | 2900 MiB |
| Burn `wgpu` | 4.4 s | encoder 1094 ms | 3867 / 4534 MiB | 526 MiB |
| Burn `vulkan` (SPIR-V) | 3.2 s | encoder 951 ms | 2489 / 2868 MiB | 532 MiB |
| transcribe.cpp Vulkan, TDT F16 | 1.2 s | 91 ms (38) | 1188 / 1209 MiB | 273 MiB |
| transcribe.cpp Vulkan, TDT F32 | 6.7 s | 110 ms (52) | 2366 / 2463 MiB | 278 MiB |
| transcribe.cpp Vulkan, unified F16 | 2.6 s | 92 ms (38) | 1190 / 1213 MiB | 272 MiB |

Peak is over all three clips; the 56 s clip sets it. Load times are from a warm page cache and
vary a lot with what else is reading the disk (the two F16 GGUF rows differ only by that).

**Cold start and the on-disk pipeline cache.** Process start to the first transcript of the 7 s
clip, with NVIDIA's shader cache (`__GL_SHADER_DISK_CACHE_PATH`) emptied first, then three more
processes that find it:

| Runtime | Empty cache | Next three processes |
| --- | ---: | ---: |
| ORT WebGPU fp32 | 4.9 s (first run 746 ms) | 4.5, 4.5, 4.8 s (first run 230–270 ms) |
| ORT WebGPU fp16 | 3.7 s (first run 845 ms) | 3.1, 3.5, 4.1 s (first run 360–660 ms) |
| transcribe.cpp Vulkan, TDT F16 | 8.2 s (first run 6.8 s) | 1.4, 1.3, 1.3 s (first run 86–91 ms) |

The driver cache matters much less to ONNX Runtime (0.5 s) than to ggml (7 s), but ONNX Runtime
pays 200–600 ms on the first inference of every process regardless (Dawn rebuilds its pipelines;
ORT has no pipeline cache of its own on disk), and session creation, not shaders, is its long pole.

**The service's unit of work** (unified F16, buffered streaming: 5.6 s left + 560 ms chunk +
560 ms right, a ~6.7 s window per chunk, only the chunk's frames decoded):

- `tcpp_bench.py --stream`, own process, chunks fed back to back, quiet machine: median 46 ms,
  worst 93 ms per full-window chunk on the 13.7 s clip (14 chunks x 10 runs); 45 ms median, 346 ms
  worst on the 56 s clip. This reproduces the 46 ms in the service's README.
- The live service on :9876, from a client (`scripts/live.ts`), machine moderately busy: 49 ms
  median per full-window chunk back to back on the 7 s clip; 86 ms median (worst 553) on the
  13.7 s clip, and 177–199 ms when paced in real time. See the caveat on load below.
- ONNX Runtime WebGPU was not run as a streaming loop. Its cost for the same unit would be mel
  (about 8 ms) + a 7 s encoder pass (46–71 ms fp32, 58 ms fp16) + decoding seven frames on the CPU
  (a few ms): an estimate of 60–85 ms per chunk against 46 ms measured for transcribe.cpp.

### What ran where

- **ORT WebGPU encoder**: 1512 nodes on `WebGpuExecutionProvider` (all MatMul 289, Conv 77,
  LayerNormalization 120, Softmax 24, Add, Mul, Transpose, Reshape, Where, Pad, Slice, ...), 495 on
  `CPUExecutionProvider`. The CPU ones are the shape arithmetic ORT keeps on the CPU on purpose
  (Unsqueeze 176, Gather 150, Concat 101, Slice 26, Squeeze 24 and a handful of others on shape
  tensors); they take about 5 ms per run in the profile. No silent fallback: the process holds
  2.4 GB of GPU memory in `nvidia-smi` and the encoder is 6–18x faster than the CPU provider.
  With the fp16 encoder: 1490 on WebGPU, 495 on CPU, 1.2 GB.
- **Decoder + joint** (26 nodes): all on CPU by default. Placed on WebGPU, all 26 run there and it
  is about 4x slower end to end (7 ms per token step instead of 1–2 ms), because greedy TDT
  decoding is one tiny session run and GPU read-back per token.
- **Mel front end** (`nemo128.onnx`, 35 nodes): CPU in every ORT row.
- **transcribe.cpp**: everything on `Vulkan0`, including the per-token decoder; mel on CPU.
- Encoder time on ORT WebGPU grows slowly with length (71 ms for 7 s, 265 ms for 56 s) and fp16
  barely helps: at short lengths it is bound by dispatching ~1500 kernels, not by GPU arithmetic.
  The fp32 encoder is also faster (46 vs 71 ms at 7 s) when the decoder keeps the GPU busy between
  encoder runs; cause not established (the same idle-GPU effect the service README notes).

### Transcripts

All TDT v2 rows (ORT WebGPU fp32 and fp16, ORT WebGPU decoder, ORT CPU, transcribe.cpp F16 and F32)
give byte-identical text on all three clips, stable across runs:

> going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight. He'll have to put in an appearance at some place of worship on Sunday morning, and he can come to us immediately afterwards.

> Welcome to QuirkQuidQuill Inc., where finance meets innovation. Explore diverse offerings from the P3 Quattro, a unique investment portfolio quadrant, to the O3 Omni, a platform for intricate derivative trading strategies. Delve into unconventional bond markets with our B3 Bond X and experience non-standard equity trading with E3 Equity. Personalize your wealth management with W3 RAPZ and anticipate market trends with the O2 Outlier, our forward-thinking financial forecasting tool. Explore venture capital world with U3Unifund or move your money with the M3 Mover, our sophisticated monetary transfer module. At QuirkQuidQuill Inc., we turn complex finance into creative solutions. Join us in redefining financial services.

The unified model (a different model, the one the service runs) says "draughty" for "drafty",
"Quirk Quid Quill Inc.", and offline drops the punctuation of the 13.7 s clip; streaming, it gives
the service's usual text. Burn's encoder output matches ORT CPU to 2e-4 absolute (values up to
0.8) on all three clips; it was not decoded to text.

### Not like for like, not measured, and failures

- **Machine load.** Other work (Rust and C++ builds, several Chrome instances) had 5–10 of the 12
  cores busy for most of the session. The tables above are from a pass taken while it was quiet
  (`results/machine-load.txt` has the CPU pressure at the start of each row, and the last line for a
  name is the run that is in `results/`; the tcpp rows are from 22:49–22:51, the ORT and Burn rows
  from 22:51–22:56). `bench.sh` runs the benchmarks in a
  high-weight systemd slice, which helps but does not isolate them. CPU-bound numbers (ORT CPU,
  decoder times, load times, worst cases) are the least reliable.
- **`scripts/measure.ts` of the service did not give usable numbers.** Run twice on a spare port
  (it starts its own instance, it does not touch :9876) while the machine was loaded: ready in
  1.5–2.2 s, 2.2–6.2 s wall per 13.7 s recording, then the 192 s real-time recording failed with
  `AssertionError: overload` both times. Output is in `results/service-measure.txt`. The service
  README's own figures (ready 1.0–1.1 s, 106 ms per chunk in real time) were taken on a quiet
  machine and were not reproduced here; the real-time figures from `live.ts` above are inflated by
  the same load.
- **Service vs ORT is two models.** The service runs `parakeet-unified-en-0.6b` with a chunked
  attention mask; the ORT and Burn rows are `parakeet-tdt-0.6b-v2`. Same 0.6b FastConformer encoder
  size; the transcribe.cpp TDT rows are the like-for-like comparison, and the unified offline row
  shows the two models cost the same in transcribe.cpp.
- **Precision.** ORT fp32 against GGUF F32 and ORT fp16 against GGUF F16 are the matched pairs. The
  fp16 ONNX is my conversion (`scripts/onnx-fp16.py`), not a published file.
- **transcribe.cpp stage times** (mel/encode/decode) are the library's own `transcribe_get_timings`;
  totals are wall time around `transcribe_run` including reading the text back. ORT and Burn times
  are wall time around calls that return CPU tensors.
- **ORT CUDA was not run** (needs 2.6 GB of CUDA 13 and cuDNN 9 libraries that are no longer on the
  box). The earlier spike measured it on the unified model: 25 ms per chunk fp32, 17–19 ms fp16,
  3.4 / 2.4 GB (`../dictation-server-linux/README.md`).
- **ORT WebGPU segfaults at process exit** (ORT 1.28 + Dawn teardown, after all results are
  printed; `Segmentation fault (core dumped)`, exit 139). `ort-bench` leaves through `_exit`.
- **Burn**: nothing blocked. `onnx2burn` imported all operators of the fp32 encoder (64 s, 2.5 GB
  `.bpk`), the generated code compiled, and the output is correct. It is simply slow: 474–543 ms
  for the 7 s window, 7–10x ORT WebGPU, with 2.5–3.9 GB of GPU memory. Both builds report the
  device fingerprint `spirv_4318_7687`, so I cannot confirm from the output that the `wgpu`-feature
  build used WGSL shaders; the two builds do differ in speed and memory. Encoder only: no mel
  front end or decoder was ported, and fp16 was not tried.
- Not measured: word error rate, a streaming loop on ORT WebGPU, cold-disk start, the WebGPU EP's
  options (graph capture, buffer cache modes, layout), wgpu directly without Burn.

## In the browser

Added 2026-10-10: the same model in real Chrome 154 on this machine, through `onnxruntime-web`
1.30.0 (npm has no 1.28), from a static page in `web/`. Headed Chrome on its own Xvfb display,
WebGPU on Vulkan. Same three clips, one discarded warm-up, 10 warm runs, median / worst in ms.
Every `session.run()` resolves to CPU tensors, so the times include the read-back.

Short answer: it works and it is on the 2080 Ti. The fp32 encoder is 20–80% slower than native
ONNX Runtime WebGPU measured in the same window (90 vs 50 ms for 7 s, 314 vs 262 ms for 56 s) and
the whole transcribe is 30–60% slower, at 40–63x real time. It costs far more memory than native:
the renderer keeps 2.4 GB (fp16) or 4.4 GB (fp32) of RSS on top of the GPU memory, where the
native process holds 0.4 GB. The fp16 encoder does not load in stock Chrome on this driver. All
on WASM, the int8 export runs at 3x real time on one thread and 7–8x on four.

| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Chrome, ORT-web WebGPU, fp32 encoder, decoder on WASM** | 90 / 145 | 173 / 263 | 108 / 149 | 262 / 296 | 314 / 339 | 897 / 1037 | 63 |
| native ORT 1.28 WebGPU fp32, rerun in the same window | 50 / 59 | 110 / 118 | 66 / 68 | 171 / 197 | 262 / 264 | 679 / 772 | 83 |
| native ORT 1.28 WebGPU fp32, table above (9 Oct) | 71 / 75 | 174 / 189 | 91 / 98 | 221 / 266 | 265 / 270 | 671 / 781 | 84 |
| **Chrome, same, fp16 encoder** (needs a Chrome flag here, see below) | 67 / 113 | 145 / 210 | 95 / 127 | 225 / 248 | 232 / 240 | 753 / 830 | 75 |
| native ORT 1.28 WebGPU fp16, rerun in the same window | 48 / 52 | 97 / 100 | 56 / 58 | 151 / 159 | 208 / 211 | 653 / 938 | 86 |
| native ORT 1.28 WebGPU fp16, table above | 58 / 105 | 144 / 176 | 66 / 69 | 157 / 172 | 208 / 211 | 557 / 713 | 101 |
| **Chrome, fp32, decoder on WebGPU too** | 68 / 76 | 558 / 638 | 81 / 109 | 902 / 939 | 337 / 363 | 3793 / 3994 | 15 |
| native, fp32, decoder on WebGPU, table above | 46 / 47 | 427 / 445 | 63 / 64 | 749 / 772 | 261 / 265 | 2862 / 2978 | 20 |
| Chrome, fp32, ORT-web's older JS WebGPU provider (JSEP build) | 103 / 151 | 222 / 369 | 96 / 103 | 235 / 266 | 280 / 285 | 809 / 949 | 69 |
| **Chrome, all WASM, fp32, 12 threads** | 734 / 806 | 821 / 995 | 1328 / 1554 | 1500 / 1724 | 8578 / 13377 | 9487 / 14395 | 6 |
| native ORT 1.28 CPU fp32, 12 cores, table above | 432 / 497 | 494 / 554 | 1006 / 1311 | 1173 / 1523 | 4890 / 5250 | 5473 / 6021 | 10 |
| Chrome, all WASM, int8, 12 threads | 933 / 1066 | 1074 / 1184 | 2072 / 2495 | 2258 / 2703 | 7684 / 8067 | 8252 / 8535 | 7 |
| Chrome, all WASM, int8, 4 threads (ORT-web's default) | 913 / 1075 | 989 / 1133 | 1652 / 1750 | 1780 / 1878 | 6300 / 7311 | 6619 / 7630 | 8 |
| **Chrome, all WASM, int8, 1 thread** (what a page without COOP/COEP headers gets) | 2083 / 2192 | 2232 / 2327 | 4200 / 4522 | 4435 / 4734 | 20223 / 20937 | 21084 / 21694 | 3 |
| Chrome, int8 encoder asked to run on WebGPU | 3746 / 3955 | 3814 / 4038 | 5065 / 5215 | 5186 / 5339 | 12318 / 14624 | 12683 / 15054 | 4 |

| Runtime | Encoder session creation | Model fetch (localhost) | First 7 s transcribe (encoder part) | GPU memory after load / peak | RSS peak | Chrome GPU process RSS peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chrome WebGPU fp32 | 6.7 s | 7.3 s | 661 ms (544) | 2423 / 2771 MiB | renderer 4375 MiB | 1734 MiB |
| native ORT WebGPU fp32, same window | 17.8 s (4.0 s on 9 Oct) | | 366 ms (320) | 2376 / 2721 MiB | 369 MiB | |
| Chrome WebGPU fp16 | 3.9 s | 1.6 s | 497 ms (403) | 1245 / 1544 MiB | renderer 2657 MiB | 1045 MiB |
| native ORT WebGPU fp16, same window | 8.5 s (2.8 s on 9 Oct) | | 432 ms (386) | 1196 / 1494 MiB | 377 MiB | |
| Chrome WebGPU fp32, decoder on WebGPU | 5.4 s | 4.5 s | 1021 ms (398) | 2464 / 2810 MiB | renderer 4357 MiB | 1784 MiB |
| Chrome JSEP fp32 | 4.9 s | 4.1 s | 1797 ms (1672) | 866 / 2984 MiB | renderer 5365 MiB | 2653 MiB |
| Chrome WASM fp32, 12 threads | 5.1 s | 3.0 s | 1205 ms (1081) | none | renderer 5860 MiB | 255 MiB |
| Chrome WASM int8, 4 threads | 4.7 s | 0.9 s | 2014 ms (1906) | none | renderer 2354 MiB | 255 MiB |
| Chrome WASM int8, 1 thread | 3.6 s | 0.9 s | 2086 ms (1972) | none | renderer 2341 MiB | 255 MiB |
| Chrome int8 on WebGPU | 4.7 s | 1.0 s | 3963 ms (3861) | 104 / 1352 MiB | renderer 2406 MiB | 1127 MiB |

GPU memory is what `nvidia-smi` charges to Chrome's GPU process; RSS is from `/proc`. Each row is a
new Chrome with an empty profile, so "first" has no Chrome shader cache (NVIDIA's own driver cache
in `~/.cache/nvidia` was warm). Page start to first transcript of the 7 s clip, files served from
localhost: 16.7 s fp32, 8.0 s fp16, 7.7–9.5 s int8 on WASM. The native session-creation times of
this window are 3–4x the 9 October ones: weights were being read from disk again, not a regression.

### The small model: parakeet-tdt_ctc-110m

Added later on 2026-10-10: NVIDIA `parakeet-tdt_ctc-110m` (FastConformer, about 110M parameters,
hybrid TDT + CTC; the TDT head is used) in the same page, same Chrome, same clips, same method.
It is the page's default and first tap.

Short answer: all on WASM with one thread, the int8 export does the 7 s clip in 1.1 s against
2.2 s for the 0.6b, with 0.8 GB of renderer memory against 2.3 GB and a 130 MiB download against
631 MiB. The fp32 weights on WASM are faster than int8 (0.8 s on one thread, 0.27 s on four) but
cost 1.7 GB and 455 MiB. On WebGPU it is no faster than the 0.6b for short clips in Chrome: the
encoder takes 120–135 ms whatever the clip length (native: 35–103 ms), so it only wins on the 56 s
clip; GPU memory is 0.5 GB (fp32) or 0.3 GB (fp16) after load. It is clearly less accurate on the
product-name clip (below).

| Runtime | 7.0 s enc | 7.0 s total | 13.7 s enc | 13.7 s total | 56.1 s enc | 56.1 s total | x real time, 56 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **110m, Chrome, all WASM, int8, 1 thread** | 1056 / 1064 | 1122 / 1143 | 2102 / 2421 | 2249 / 2567 | 10147 / 10550 | 10700 / 11207 | 5 |
| 0.6b, same (table above) | 2083 / 2192 | 2232 / 2327 | 4200 / 4522 | 4435 / 4734 | 20223 / 20937 | 21084 / 21694 | 3 |
| 110m, native ORT 1.28 CPU, int8, 1 thread, same window | 347 / 368 | 364 / 385 | 718 / 779 | 746 / 810 | 3401 / 3549 | 3514 / 3649 | 16 |
| **110m, Chrome, all WASM, int8, 4 threads** | 434 / 504 | 486 / 555 | 830 / 882 | 906 / 967 | 3992 / 5465 | 4326 / 5812 | 13 |
| 0.6b, same | 913 / 1075 | 989 / 1133 | 1652 / 1750 | 1780 / 1878 | 6300 / 7311 | 6619 / 7630 | 8 |
| **110m, Chrome, all WASM, fp32, 1 thread** | 730 / 769 | 807 / 846 | 1484 / 1505 | 1625 / 1647 | 7437 / 8843 | 8034 / 9433 | 7 |
| 110m, native ORT 1.28 CPU, fp32, 1 thread, same window | 275 / 305 | 344 / 384 | 515 / 553 | 634 / 688 | 2609 / 2688 | 2979 / 3081 | 19 |
| **110m, Chrome, all WASM, fp32, 4 threads** | 225 / 282 | 274 / 335 | 434 / 461 | 512 / 561 | 2296 / 2412 | 2586 / 2733 | 22 |
| 0.6b, Chrome, all WASM, fp32, 12 threads (no 1- or 4-thread row exists) | 734 / 806 | 821 / 995 | 1328 / 1554 | 1500 / 1724 | 8578 / 13377 | 9487 / 14395 | 6 |
| **110m, Chrome, WebGPU fp32 encoder, decoder on WASM** | 117 / 190 | 174 / 277 | 131 / 153 | 228 / 250 | 134 / 141 | 498 / 517 | 113 |
| 0.6b, same | 90 / 145 | 173 / 263 | 108 / 149 | 262 / 296 | 314 / 339 | 897 / 1037 | 63 |
| 110m, native ORT 1.28 WebGPU fp32, same window | 35 / 49 | 66 / 84 | 53 / 74 | 128 / 157 | 103 / 150 | 381 / 471 | 147 |
| **110m, Chrome, WebGPU fp16 encoder** (needs the same Chrome flag) | 125 / 147 | 192 / 220 | 123 / 153 | 220 / 259 | 121 / 145 | 503 / 527 | 112 |
| 0.6b, same | 67 / 113 | 145 / 210 | 95 / 127 | 225 / 248 | 232 / 240 | 753 / 830 | 75 |
| 110m, native ORT 1.28 WebGPU fp16, same window | 33 / 46 | 67 / 77 | 58 / 69 | 122 / 129 | 97 / 102 | 367 / 394 | 153 |

| Runtime | Download | Model fetch (localhost) | Encoder session creation | Page start to first 7 s transcript | First 7 s transcribe (encoder part) | GPU memory after load / peak | Renderer RSS after load / peak | Chrome GPU process RSS peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **110m WASM int8, 1 thread** | 130 MiB | 0.2 s | 2.2 s | 5.6 s | 1232 ms (1133) | none | 801 / 804 MiB | 253 MiB |
| 0.6b WASM int8, 1 thread | 631 MiB | 0.9 s | 3.6 s | 7.7–9.5 s | 2086 ms (1972) | none | 2341 MiB peak | 255 MiB |
| 110m WASM int8, 4 threads | 130 MiB | 0.2 s | 2.5 s | 5.4 s | 747 ms (683) | none | 813 / 817 MiB | 253 MiB |
| 110m WASM fp32, 1 thread | 455 MiB | 0.6 s | 2.5 s | 6.0 s | 900 ms (796) | none | 1453 / 1672 MiB | 253 MiB |
| 110m WASM fp32, 4 threads | 455 MiB | 0.7 s | 2.4 s | 5.5 s | 381 ms (313) | none | 1462 / 1689 MiB | 253 MiB |
| 110m WebGPU fp32 | 455 MiB | 0.7 s | 3.5 s | 6.7 s | 520 ms (445) | 495 / 1091 MiB | 1196 / 1200 MiB | 675 MiB |
| 0.6b WebGPU fp32 | 2397 MiB | 7.3 s | 6.7 s | 16.7 s | 661 ms (544) | 2423 / 2771 MiB | 4375 MiB peak | 1734 MiB |
| 110m WebGPU fp16 | 238 MiB | 0.4 s | 3.2 s | 6.2 s | 647 ms (563) | 280 / 561 MiB | 1237 / 1245 MiB | 525 MiB |
| 0.6b WebGPU fp16 | 1216 MiB | 1.6 s | 3.9 s | 8.0 s | 497 ms (403) | 1245 / 1544 MiB | 2657 MiB peak | 1045 MiB |
| 110m native ORT CPU int8, 1 thread | | | 1.1 s | | 355 ms (342) | none | process 1091 MiB peak | |
| 110m native ORT CPU fp32, 1 thread | | | 1.4 s | | 424 ms (335) | none | process 1352 MiB peak | |
| 110m native ORT WebGPU fp32 | | | 1.9 s | | 427 ms (392) | 449 / 1041 MiB | process 321 MiB peak | |
| 110m native ORT WebGPU fp16 | | | 1.7 s | | 467 ms (437) | 234 / 511 MiB | process 600 MiB peak | |

Same conventions as the tables above: one discarded warm-up then 10 warm runs, median / worst in
ms; each Chrome row is a new Chrome with an empty profile; peaks are over all three clips. Rows ran
02:01–02:12 with load average 5–10 and CPU pressure (avg10) under 5% at the start of every row
(`results/browser/machine-load.txt`). `performance.measureUserAgentSpecificMemory()` answered 339 MB
for int8 and 1268 MB for fp32 on one thread, and did not answer in the WebGPU and fp32 4-thread rows.

**Where the model came from.** `OpenVoiceOS/nvidia-parakeet-tdt_ctc-110m-onnx` on Hugging Face
(revision `40cd0e2`): an export in the same onnx-asr layout as istupakov's 0.6b one (`encoder-model`
and `decoder_joint-model`, fp32 and int8, `vocab.txt`), so the page and `ort-bench` take it with
parameters instead of a second code path. The fp16 encoder is our own conversion of its fp32 encoder
(`scripts/onnx-fp16.py --inline`, 218 MiB, one file). sherpa-onnx's
`csukuangfj/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000` was looked at and not used: that
repository holds one `model.onnx`, the CTC head, and its `-int8` twin lists no files. Nothing was
exported from NeMo.

**What is different from the 0.6b**, all handled in `web/src/main.ts` (`FAMILIES`) and `web/src/mel.ts`:

| | 0.6b v2 | 110m |
| --- | --- | --- |
| Mel bins | 128 | 80 |
| Encoder output width | 1024 | 512 |
| Subsampling | 8 (88 frames for 7 s) | 8 (88 frames for 7 s) |
| Prediction network state | 2 LSTM layers: `[2, 1, 640]` | 1 layer: `[1, 1, 640]` |
| Vocabulary | 1024 tokens, blank id 1024 | 1024 tokens (a different set), blank id 1024 |
| TDT durations | 0–4 (5 logits after the vocabulary) | 0–4 |
| Max symbols per frame | 10 | 10 (`max_tokens_per_step` in the export's config) |
| Mel front end shipped with the export | `nemo128.onnx`, older onnx-asr: reflect padding, n/160 + 1 frames, all normalised | `nemo80.onnx` of onnx-asr 0.12: zero padding, floor(n/160) valid frames, statistics over those only, the frame after them set to 0, encoder told the valid length |

The front-end difference is between onnx-asr versions, not between the models; each model is fed
what its own reference feeds it. The JS mel matches the dumped `nemo80.onnx` features to 1.3e-4
worst case, 8e-7 on average (`bun web/check-mel.ts`).

**Reference.** onnx-asr 0.12.0 (the library the export is for; native onnxruntime 1.31, CPU) and
`ort-bench` (ONNX Runtime 1.28, CPU) give the same fp32 text on all three clips
(`results/110m/reference-*.jsonl`); that text is what the page compares against. In Chrome the fp32
weights reproduce it exactly on WASM (1 and 4 threads) and on WebGPU, stable over the warm runs.
The other weights sit on near-ties and do not reproduce across runtimes:

- **fp16**: 7 s and 13.7 s identical to fp32. On the 56 s clip one or two commas become full stops
  ("innovation. Explore", "equity. Personalize"): two in native ORT (CPU and WebGPU), one in Chrome.
- **int8**: differs from fp32 on every clip, and from one ONNX Runtime to the next. 7 s: "draughty"
  for "drafty" and no final full stop (same in all three runtimes). 13.7 s: "drafty" where fp32 has
  "draughty"; onnx-asr adds a trailing comma. 56 s: see the table; onnx-asr 1.31 and Chrome have
  "Quatro Unique Investment Portfolio Quadrant", native 1.28 has "Quatro a unique investment
  portfolio quadrant", and "non standard" / "nonstandard" flips too.

**Transcripts beside the 0.6b ones** (Chrome; 0.6b fp32 is identical to its native text):

| Clip | 0.6b fp32 | 110m fp32 | 110m int8 (WASM) |
| --- | --- | --- | --- |
| 7 s | ...in drafty schoolrooms day after day for a fortnight. | identical | ...in **draughty** schoolrooms day after day for a **fortnight** (no full stop) |
| 13.7 s | ...drafty schoolrooms ... immediately afterwards. | ...**draughty** schoolrooms ... immediately **afterwards** (no full stop) | ...drafty schoolrooms ... immediately **afterwards** (no full stop) |

The 56 s product-name clip, phrase by phrase:

| 0.6b fp32 | 0.6b int8 | 110m fp32 | 110m int8 (WASM) |
| --- | --- | --- | --- |
| QuirkQuidQuill Inc., where | Quirk Quid Quill Inc., where | Quirk Quid Quill Inc. where | QuirkQuid Quill Inc. where |
| innovation. Explore | same | innovation, explore | innovation, explore |
| the P3 Quattro, a unique investment portfolio quadrant, to | same | the P three Quatro a unique investment portfolio quadrant to | the P three Quatro Unique Investment Portfolio Quadrant to |
| the O3 Omni | same | the O three Omni | the O three Omni |
| our B3 Bond X and experience | same | our B three Bond X **and our** and experience | our B three Bond X **and our** and experience |
| non-standard equity trading with E3 Equity. Personalize | ...with E3. Personalize | non standard equity trading with E three equity, personalize | same as 110m fp32 |
| W3 RAPZ | W3 RAP Z | W three Wrap Z | W three Wrap Z |
| the O2 Outlier, our forward-thinking | same | the O two outlier, our forward thinking | same as 110m fp32 |
| U3Unifund | U3 Unifund | U three unifund | U three Unifund |
| the M3 Mover | same | the M three mover | the M three mover |
| At QuirkQuidQuill Inc., we | At QuirkQuid Quill Inc., we | At **Cork** Quid Quill Inc. we | At **Cork Quill Inc,** we |

The rest of the 56 s clip is word for word the same in all four. So on these clips the 110m writes
digits as words, does not capitalise the product names or new sentences reliably, drops hyphens,
inserts two words that were not said ("and our"), and mishears "Quirk" as "Cork" once; its int8
also drops "Quid" there. The 0.6b gets all of those right. Three clips are not a word error rate.

Other things seen:

- **The WebGPU encoder time does not scale with clip length in Chrome**: 117, 131 and 134 ms for
  7, 13.7 and 56 s (fp16: 125, 123, 121), where native ONNX Runtime WebGPU takes 35, 53 and 103 ms
  with the same weights. Something costs about 80–100 ms per run regardless of the audio; it was not
  investigated. The practical result is that for dictation-length clips the 110m on WebGPU in the
  browser is not faster than the 0.6b (90 ms for 7 s).
- **fp32 beats int8 on WASM** at both thread counts (730 vs 1056 ms, 225 vs 434 ms for 7 s), and
  natively on one thread too (275 vs 347 ms). int8 buys download size (130 vs 455 MiB) and memory
  (0.8 vs 1.7 GB), not speed.
- **Chrome WASM is about 3x native on one thread** (1056 vs 347 ms int8, 730 vs 275 ms fp32).
- **fp16 fails in stock Chrome the same way as the 0.6b**: `Error: Can't create a session.
  ERROR_CODE: 1, ERROR_MESSAGE: shader_helper.cc:416 GenerateSourceCode Program Transpose requires
  f16 but the device does not support it.` (`chrome-110m-webgpu-fp16-stock.json`). The fp16 rows
  used `--enable-dawn-features=vulkan_enable_f16_on_nvidia`. fp16 halves GPU memory and the download;
  it does not make the encoder faster and the renderer RSS is the same 1.2 GB.
- Decoder + joint on WASM: 482 steps for the 56 s clip, 1.1 ms per step on one thread, 0.4–0.6 ms
  on four. Mel front end in JS: 10–15 ms for 7 s, 73–87 ms for 56 s.
- Not measured for the 110m: any phone, Safari or Firefox; int8 on WebGPU and the decoder on
  WebGPU (both were losses for the 0.6b); the JSEP build; more than 4 threads; a warm Chrome shader
  cache; the CTC head; sherpa-onnx's export; word error rate.

### What was really in use

- **Adapter**: `navigator.gpu.requestAdapter()` reports vendor `nvidia`, architecture `turing`,
  and with `--enable-webgpu-developer-features` device `0x1e07`, "NVIDIA GeForce RTX 2080 Ti",
  `isFallbackAdapter` false. `chrome://gpu` data: `ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA GeForce RTX
  2080 Ti), NVIDIA-580.178.4.0)`, WebGPU and Vulkan enabled. Chrome's GPU process holds 2.4 GB
  (fp32) or 1.2 GB (fp16) in `nvidia-smi`, the same as the native process. Not SwiftShader or llvmpipe.
- **Chrome flags** that were needed: `--enable-unsafe-webgpu --ignore-gpu-blocklist
  --enable-features=Vulkan --use-angle=vulkan`. Linux WebGPU is not on by default in Chrome 154.
- **`shader-f16` is absent in stock Chrome on this driver.** Dawn gates f16 on NVIDIA Vulkan
  drivers older than 615.71 (this box has 580.178), and the fp16 encoder then fails at session
  creation with `ERROR_CODE: 1, ERROR_MESSAGE: shader_helper.cc:416 GenerateSourceCode Program
  Transpose requires f16 but the device does not support it.` The fp16 row was run with
  `--enable-dawn-features=vulkan_enable_f16_on_nvidia`; with it the feature appears and the
  transcripts are right. Native ORT 1.28 ships an older Dawn without that gate.
- **Runtime build**: `ort.webgpu.min.mjs` with `ort-wasm-simd-threaded.asyncify.wasm`, which is
  the C++ WebGPU provider compiled to WASM (the same provider as native). The JSEP row is the
  older provider written in JS (`ort.all.min.mjs`); warm speed is about the same, its first run
  is 3x slower.
- **Decoder + joint** on WASM, 4 threads: 1.1–1.4 ms per token step against 0.8–1.1 native. On
  WebGPU it is about 9 ms per step. On the 56 s clip the fp32 browser row is 218 ms behind native:
  96 ms of that is the decoder, 77 ms the mel front end, 52 ms the encoder.
- **Mel front end** is JS (`web/src/mel.ts`), 10–25 ms for 7 s and 80–150 ms for 56 s, against
  7 and 63 ms for `nemo128.onnx` native. onnxruntime-web cannot run `nemo128.onnx`: `Could not
  find an implementation for Cast(13) node with name 'node_Cast_6'` (a cast to float64). The JS
  version matches the dumped native features to 6e-6 (`bun web/check-mel.ts`).
- **int8 on WebGPU** is not a GPU run: the quantized operators have no WebGPU kernels, the
  encoder is slower than plain WASM and GPU memory stays near zero until the first run.
- **WASM threads** need `crossOriginIsolated`. `web/serve.ts` sends COOP/COEP; the published copy
  cannot (tailscale serve, no custom headers) and runs one thread unless the page's opt-in service
  worker is turned on.
- The 2.4 GB fp32 weights do not fit a JS `ArrayBuffer` in Chrome (`RangeError: Array buffer
  allocation failed`); the page reads files above 2 GB into a `WebAssembly.Memory` buffer.

### Transcripts in the browser

fp32 and fp16, on WebGPU (both builds), WASM and with the decoder on WebGPU: identical to the
native text on all three clips, stable over the warm runs. The int8 export is identical on the
7 s and 13.7 s clips and differs on the 56 s clip in five places (four on WebGPU), all in the
product names: "Quirk Quid Quill Inc.", "E3." for "E3 Equity.", "RAP Z", "U3 Unifund",
"QuirkQuid Quill Inc."

### Phone page

`https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-webgpu-browser/` (tailnet only,
https). Models are served from the same place: the shelf is a plain directory on the Pi behind
`tailscale serve` with 2.6 TB free and no size limit, so nothing is fetched from Hugging Face. It
opens idle with seven one-tap configurations, the small model first and smallest first within each
model: 110m int8 all on WASM (130 MiB, also what the page selects with no parameters), 110m fp16
with the encoder on WebGPU (238 MiB), 110m fp32 with the encoder on WebGPU (455 MiB), 110m fp32
all on WASM, then the 0.6b ones: int8 all on WASM (631 MiB), fp16 on WebGPU (1.25 GB), fp32 on
WebGPU (2.4 GB). Each runs the 7 s clip with 3 warm runs. Every finished step is written to the page and to `localStorage`, and a reload
lists earlier runs with the last step each one reached. Model files go to Cache Storage when the
quota allows, so a second attempt does not download again.

Checked from Chrome on diesel2 against the published link, 110m: the bare link with no model
parameter picks 110m int8 on WASM (one thread, model fetched from the Pi in 4.9 s, 7 s clip in
1.14 s, renderer 0.8 GB RSS, the int8 text above); fp16 on WebGPU with the Dawn flag (encoder
119 ms) and fp32 on WebGPU (encoder 152 ms) load, run and give the native fp32 transcript
(`results/browser/published-110m-*.json`). The page at 390 px wide has no horizontal scroll. Earlier,
0.6b: int8 on WASM (one thread, 7 s clip in 2.1 s, renderer 2.3 GB RSS) and fp16 on WebGPU with
the Dawn flag (encoder 61 ms) both load, run and give the native transcript; the threads button
turns `crossOriginIsolated` on. **Nothing was run on a phone.** The desktop numbers that bear on
it: the smallest configuration is now 130 MiB to download and peaks at 0.8 GB of renderer memory
(0.6b int8: 631 MiB and 2.3 GB), one WASM thread does 7 s of audio in 1.1 s on this desktop CPU,
and fp16 needs `shader-f16`.

### Browser: not measured, and caveats

- **Load.** A first pass ran with 20–35 runnable tasks on the 12 cores and is kept in
  `results/browser/pass1-loaded/` (fp32 encoder 138 ms at 7 s there). The tables are from a second
  pass at 01:28–01:45 with load average 8–9 falling to 4 and CPU pressure under 5%
  (`results/browser/machine-load.txt`), except "int8 on WebGPU", which is from the end of the first
  pass when it was already quiet. The int8 12- and 4-thread rows started quiet but something else
  began during them; 12 threads slower than 4 is that, or oversubscription. Other Chrome instances
  were using the GPU lightly throughout.
- The native rows were rerun in that window so each browser row has a neighbour under the same
  load; they came out faster at 7 s than on 9 October (50 vs 71 ms), so compare with those.
- onnxruntime-web 1.30.0 against native ORT 1.28: two versions apart.
- `performance.measureUserAgentSpecificMemory()` resolved before the end of the run in six of the
  nine rows (1.3–5.4 GB, in each JSON) and not in the fp32, fp16 and JSEP WebGPU rows.
  `performance.memory` JS heap is in each JSON too but excludes WASM memory, so RSS is the number
  to read.
- Not measured: any phone, Safari or Firefox; a Chrome with a warm shader cache (second visit);
  node placement in the browser; a streaming loop; WASM with more than 4 GB (the fp32 model fits,
  renderer 5.9 GB); `parakeet.js`; word error rate of int8 beyond the three clips.
- The 110m rows have their own list at the end of [The small model](#the-small-model-parakeet-tdt_ctc-110m).

## Layout

- `ort-bench/`: Rust, `ort` 2.0.0-rc.13 with the `webgpu` feature. Its prebuilt ONNX Runtime 1.28
  (static, plus `libwebgpu_dawn.so`) is downloaded at build time. Mel via the export's own
  `nemo128.onnx` (or `nemo80.onnx` for the 110m: whichever is in the model directory), greedy TDT
  decode loop in `src/main.rs`; the LSTM state shape is read from the decoder.
- `scripts/reference-110m.py`: the 110m reference transcripts from onnx-asr, and where `nemo80.onnx`
  comes from. `scripts/onnx-fp16.py`, `scripts/mel-filterbank.py`: fp16 encoder and JS-mel filterbank
  for either model.
- `burn-bench/`: Rust, Burn 0.22; includes the code `onnx2burn` generates into `cache/burn-gen/`.
- `scripts/tcpp_bench.py`: transcribe.cpp through ctypes, reusing `../dictation-server-linux/server.py`
  bindings and the libraries in that directory's `.cache` (read-only).
- `scripts/live.ts`: client timings against the running service. `scripts/placement.ts`: node
  placement from a verbose ORT log. `scripts/table.ts`: the tables above from `results/`.
- `web/`: the browser page (bun + TypeScript, no framework). `src/main.ts` and `src/mel.ts` are the
  page, `build.ts` writes `web/dist/` (page, onnxruntime-web files, clips, hard links to the models
  in `cache/`), `serve.ts` serves it with COOP/COEP, `drive.ts` runs one configuration in Chrome on
  Xvfb and writes `results/browser/<name>.json`, `bench.sh` runs the rows. `sw.js` is the opt-in
  service worker that gives a header-less host cross-origin isolation.
  `scripts/browser-table.ts` prints the browser tables (`110m` argument: the small model's rows).
- `cache` is a symlink to `~/devfs/cache/parakeet-webgpu-bench` (models, audio, build output; 15 GB).

## Rerun

```sh
cd parakeet-webgpu-bench
C=~/devfs/cache/parakeet-webgpu-bench; mkdir -p $C/onnx-tdt-v2 $C/gguf $C/audio; ln -sfn $C cache

# models
for f in config.json decoder_joint-model.onnx encoder-model.onnx encoder-model.onnx.data nemo128.onnx vocab.txt; do
  curl -L -o cache/onnx-tdt-v2/$f https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/$f; done
for q in F16 F32; do curl -L -o cache/gguf/parakeet-tdt-0.6b-v2-$q.gguf \
  https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-$q.gguf; done
uv run --no-project --python 3.12 --with onnx --with onnxconverter-common scripts/onnx-fp16.py cache/onnx-tdt-v2 cache/onnx-tdt-v2-fp16

# audio (product-names.wav is in transcribe.cpp v0.3.1's samples/)
F=../dictation-server-linux/fixtures/librispeech-sample.flac
ffmpeg -i $F -t 7 -f f32le -ar 16000 -ac 1 cache/audio/a07.f32
ffmpeg -i $F -f f32le -ar 16000 -ac 1 cache/audio/a14.f32
ffmpeg -i product-names.wav -f f32le -ar 16000 -ac 1 cache/audio/a56.f32

# build
(cd ort-bench && cargo build --release)
cargo install burn-onnx --version 0.22.0 --root cache/burn-tools
cache/burn-tools/bin/onnx2burn cache/onnx-tdt-v2/encoder-model.onnx cache/burn-gen
(cd burn-bench && cargo build --release && cp ../cache/target-burn/release/burn-bench ../cache/target-burn/release/burn-bench-wgsl \
  && cargo build --release --features vulkan && cp ../cache/target-burn/release/burn-bench ../cache/target-burn/release/burn-bench-spirv)

# everything, or named rows (names are the results/*.jsonl files)
./bench.sh
./bench.sh ort-webgpu-fp32 tcpp-vulkan-tdt-f16
bun scripts/table.ts

# single runs
cache/target-ort/release/ort-bench --model cache/onnx-tdt-v2 --enc-ep webgpu --dec-ep cpu --runs 10 cache/audio/a07.f32
cache/target-ort/release/ort-bench --model cache/onnx-tdt-v2 --enc-ep cpu --runs 10 cache/audio/a07.f32
python3 scripts/tcpp_bench.py cache/gguf/parakeet-tdt-0.6b-v2-F16.gguf --runs 10 cache/audio/a07.f32
python3 scripts/tcpp_bench.py ../dictation-server-linux/.cache/models/parakeet-unified-en-0.6b-F16.gguf --stream cache/audio/a14.f32
cache/target-burn/release/burn-bench-wgsl --backend wgpu cache/dump/a07      # after an ort-bench --dump cache/dump run; run-all.sh does it

# node placement
cache/target-ort/release/ort-bench --verbose --profile cache/logs/prof --runs 2 cache/audio/a07.f32 2> cache/logs/verbose.err
bun scripts/placement.ts cache/logs/verbose.err cache/logs/prof_*.json

# the service
bun scripts/live.ts http://127.0.0.1:9876 10 cache/audio/a07.f32 cache/audio/a14.f32
bun ../dictation-server-linux/scripts/measure.ts 19877
```

### Browser rows

```sh
cd parakeet-webgpu-bench
# once: int8 export and the mel filterbank for the JS front end
mkdir -p cache/onnx-tdt-v2-int8
for f in encoder-model.int8.onnx decoder_joint-model.int8.onnx; do
  curl -L -o cache/onnx-tdt-v2-int8/$f https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/$f; done
uv run --no-project --python 3.12 --with onnx --with numpy scripts/mel-filterbank.py cache/onnx-tdt-v2

cd web && bun install && bun build.ts
tmux new -d -s pkb-serve 'bun serve.ts 8787'       # http://127.0.0.1:8787/ with COOP/COEP
./bench.sh                                          # every row; waits up to QUIET_WAIT s for CPU pressure under QUIET %
QUIET=5 QUIET_WAIT=300 ./bench.sh native-ort-webgpu-fp32 chrome-webgpu-fp32
bun ../scripts/browser-table.ts

# one configuration (query string is the page's; add --chrome-arg ARG for extra Chrome flags)
bun drive.ts try "model=fp16&enc=webgpu&dec=wasm&clip=a07&runs=3&cache=0" --chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia
# page parameters: model=110m-int8|110m-fp16|110m-fp32|int8|fp16|fp32 (the last three are the 0.6b)
#                  clip=a07|a14|a56|all  enc=webgpu|wasm  dec=wasm|webgpu
#                  rt=webgpu|jsep  runs=N  threads=N  cache=0|1  auto=1  base=<models url>

# the small model (parakeet-tdt_ctc-110m). Once: export, reference transcripts + nemo80.onnx, filterbank, fp16
S=cache/onnx-tdt-ctc-110m; cd ..; mkdir -p $S results/110m cache/dump-110m
for f in config.json vocab.txt decoder_joint-model.int8.onnx decoder_joint-model.onnx decoder_joint-model.onnx_data \
  encoder-model.int8.onnx encoder-model.onnx encoder-model.onnx_data; do
  curl -L -o $S/$f https://huggingface.co/OpenVoiceOS/nvidia-parakeet-tdt_ctc-110m-onnx/resolve/main/$f; done
uv run --no-project --python 3.12 --with onnx-asr==0.12.0 --with onnxruntime --with numpy scripts/reference-110m.py $S \
  cache/audio/a07.f32 cache/audio/a14.f32 cache/audio/a56.f32 > results/110m/reference-onnx-asr.jsonl
uv run --no-project --python 3.12 --with onnx --with numpy scripts/mel-filterbank.py $S
uv run --no-project --python 3.12 --with onnx --with onnxconverter-common scripts/onnx-fp16.py $S $S-fp16 --inline
(cd ort-bench && cargo build --release)
# native reference and the features check-mel.ts compares against (same again with --model $S-fp16, and for
# int8 with --encoder encoder-model.int8.onnx --decoder decoder_joint-model.int8.onnx)
cache/target-ort/release/ort-bench --model $S --enc-ep cpu --runs 0 --dump cache/dump-110m \
  cache/audio/a07.f32 cache/audio/a14.f32 cache/audio/a56.f32 > results/110m/reference-ort-cpu-fp32.jsonl
cd web && bun check-mel.ts && bun build.ts
QUIET=5 QUIET_WAIT=600 ./bench.sh $(sed -n 's/^\(row\|native\) \([a-z0-9-]*110m[a-z0-9-]*\).*/\2/p' bench.sh)   # the eleven 110m rows
bun ../scripts/browser-table.ts 110m
bun drive.ts try "model=110m-int8&enc=wasm&dec=wasm&threads=1&clip=a07&runs=3&cache=0"

# publish for the phone (private tailnet shelf; 5.0 GB the first time) and check it
~/devfs/repos/kkrausse/random/scripts/deploy-artifact.sh dist parakeet-webgpu-browser
bun drive.ts published-110m-default "clip=a07&runs=3" --url https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-webgpu-browser/
bun drive.ts published-int8-wasm "model=int8&clip=a07&runs=3" --url https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-webgpu-browser/
```

The dictation service was running the whole time (idle, 1.2 GB of GPU memory) and was not
restarted or reconfigured.
