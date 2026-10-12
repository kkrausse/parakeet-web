// NeMo's log-mel front end, as nemo128.onnx / nemo80.onnx compute it (128 bins for the 0.6b models,
// 80 for the 110m; the number of bins is the filterbank's width. Those graphs cannot run in
// onnxruntime-web: they cast to float64 for the STFT and the web build has no such Cast kernel):
// pre-emphasis 0.97, reflect pad 256, 512-point STFT with a 400-sample symmetric Hann window and hop
// 160, power spectrum, mel filterbank (taken from the ONNX file), log(x + 2^-24), then per-feature
// mean / standard deviation normalisation over the clip.
//
// The two exports come with front ends from different onnx-asr versions, and they are not the same:
//   "reflect" (the nemo128.onnx beside istupakov's 0.6b export): reflect padding, n/160 + 1 frames, all
//             of them valid and all in the mean / deviation.
//   "zero"    (nemo80.onnx of onnx-asr 0.12, used for the 110m; follows current NeMo): zero padding,
//             floor(n/160) valid frames; the statistics are over the valid frames only and the one
//             frame after them is set to 0. The encoder is told the valid length.
export type MelStyle = "reflect" | "zero";
const N_FFT = 512, HOP = 160, WIN = 400, BINS = 257;
const window = new Float64Array(N_FFT);
for (let i = 0; i < WIN; i++) window[(N_FFT - WIN) / 2 + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1));
const cos = new Float64Array(N_FFT / 2), sin = new Float64Array(N_FFT / 2);
for (let i = 0; i < N_FFT / 2; i++) cos[i] = Math.cos((2 * Math.PI * i) / N_FFT), sin[i] = -Math.sin((2 * Math.PI * i) / N_FFT);
const rev = new Uint16Array(N_FFT);
for (let i = 0; i < N_FFT; i++) { let r = 0; for (let b = 0; b < 9; b++) r |= ((i >> b) & 1) << (8 - b); rev[i] = r; }

function fft(re: Float64Array, im: Float64Array) {
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1, stride = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const a = start + k, b = a + half, wr = cos[k * stride], wi = sin[k * stride];
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr, im[b] = im[a] - ti, re[a] += tr, im[a] += ti;
      }
    }
  }
}

/** Returns features as [mels, frames] row-major (the encoder's audio_signal without the batch axis). */
export function logMel(audio: Float32Array, filterbank: Float32Array, style: MelStyle = "reflect"): { features: Float32Array; frames: number; validFrames: number; mels: number } {
  const MELS = filterbank.length / BINS;
  const n = audio.length, pad = N_FFT / 2;
  const x = new Float64Array(n + 2 * pad);
  const pre = (i: number) => (i === 0 ? audio[0] : Math.fround(audio[i] - Math.fround(Math.fround(0.97) * audio[i - 1])));
  for (let i = 0; i < n; i++) x[pad + i] = pre(i);
  if (style === "reflect") for (let i = 0; i < pad; i++) x[pad - 1 - i] = x[pad + 1 + i], x[pad + n + i] = x[pad + n - 2 - i];
  const frames = Math.floor(n / HOP) + 1;
  const valid = style === "reflect" ? frames : Math.floor(n / HOP);
  const features = new Float32Array(MELS * frames);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), power = new Float32Array(BINS);
  // Each mel filter is a narrow triangle: only walk its non-zero bins.
  const lo = new Uint16Array(MELS), hi = new Uint16Array(MELS);
  for (let m = 0; m < MELS; m++) {
    let a = BINS, b = 0;
    for (let k = 0; k < BINS; k++) if (filterbank[k * MELS + m] !== 0) a = Math.min(a, k), b = k + 1;
    lo[m] = Math.min(a, b), hi[m] = b;
  }
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < N_FFT; i++) re[rev[i]] = x[f * HOP + i] * window[i];
    im.fill(0);
    fft(re, im);
    for (let b = 0; b < BINS; b++) power[b] = re[b] * re[b] + im[b] * im[b];
    for (let m = 0; m < MELS; m++) {
      let s = 0;
      for (let b = lo[m]; b < hi[m]; b++) s += power[b] * filterbank[b * MELS + m];
      features[m * frames + f] = Math.log(Math.fround(s) + 5.9604645e-8);
    }
  }
  for (let m = 0; m < MELS; m++) {
    const row = features.subarray(m * frames, (m + 1) * frames);
    let mean = 0;
    for (let f = 0; f < valid; f++) mean += row[f];
    mean /= valid;
    let sq = 0;
    for (let f = 0; f < valid; f++) sq += (row[f] - mean) ** 2;
    const std = Math.sqrt(sq / (valid - 1)) + 1e-5;
    for (let f = 0; f < valid; f++) row[f] = (row[f] - mean) / std;
    row.fill(0, valid);
  }
  return { features, frames, validFrames: valid, mels: MELS };
}
