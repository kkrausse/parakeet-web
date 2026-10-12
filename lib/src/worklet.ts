// AudioWorklet for the live page: microphone at the device rate in, 16 kHz mono float PCM out, in
// blocks of BLOCK samples posted to the page with their peak. The resampler is a polyphase
// Kaiser-windowed sinc low-pass (cutoff just under the lower Nyquist), exact for rational ratios
// such as 48000 -> 16000 (1 phase) and 44100 -> 16000 (160 phases); odd rates use 512 quantized phases.
// (The platform requires a class here; everything else in this project is functions.)
export {};
declare const sampleRate: number;
declare function registerProcessor(name: string, ctor: unknown): void;
declare class AudioWorkletProcessor { readonly port: MessagePort; constructor(); }

const OUT_RATE = 16000;
const BLOCK = 640; // 40 ms: two 20 ms endpointing frames per message
const ZERO_CROSSINGS = 12; // per side
const KAISER_BETA = 8.6; // about -85 dB stop band
const MAX_PHASES = 512;

const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
function besselI0(x: number) {
  let sum = 1, term = 1;
  for (let k = 1; k < 40; k++) { term *= (x / (2 * k)) ** 2; sum += term; if (term < 1e-12 * sum) break; }
  return sum;
}

interface Resampler { push(input: Float32Array, emit: (sample: number) => void): void }
/** Streaming resampler from `inRate` to `outRate`. */
function makeResampler(inRate: number, outRate: number): Resampler {
  if (inRate === outRate) return { push(input, emit) { for (let i = 0; i < input.length; i++) emit(input[i]); } };
  const g = gcd(inRate, outRate);
  const L = outRate / g, M = inRate / g; // output n sits at input position n * M / L
  const fc = 0.5 * Math.min(1, outRate / inRate) * 0.94; // cycles per input sample
  const half = Math.ceil(ZERO_CROSSINGS / (2 * fc)); // taps each side of the centre
  const taps = 2 * half;
  const phases = Math.min(L, MAX_PHASES);
  const table = new Float32Array(phases * taps);
  const i0b = besselI0(KAISER_BETA);
  for (let p = 0; p < phases; p++) {
    const frac = p / phases; // output position = base + frac, taps cover base-half+1 .. base+half
    let sum = 0;
    for (let k = 0; k < taps; k++) {
      const x = k - half + 1 - frac;
      const r = x / half;
      const w = Math.abs(r) >= 1 ? 0 : besselI0(KAISER_BETA * Math.sqrt(1 - r * r)) / i0b;
      const a = 2 * Math.PI * fc * x;
      const v = (Math.abs(a) < 1e-9 ? 1 : Math.sin(a) / a) * w;
      table[p * taps + k] = v; sum += v;
    }
    for (let k = 0; k < taps; k++) table[p * taps + k] /= sum; // unity gain at DC for every phase
  }
  // `buf` holds input samples from absolute index `buf0`; the first `half - 1` are zeros (history before the stream).
  let buf = new Float32Array(taps + 4096), n = half - 1, buf0 = -(half - 1);
  let base = 0, num = 0; // next output is at input position base + num / L
  return {
    push(input, emit) {
      if (n + input.length > buf.length) { const b = new Float32Array((n + input.length) * 2); b.set(buf.subarray(0, n)); buf = b; }
      buf.set(input, n); n += input.length;
      for (;;) {
        let phase = L === phases ? num : Math.round((num / L) * phases), b = base;
        if (phase === phases) phase = 0, b++;
        const first = b - half + 1 - buf0;
        if (first + taps > n) break;
        let acc = 0;
        const t = phase * taps;
        for (let k = 0; k < taps; k++) acc += buf[first + k] * table[t + k];
        emit(acc);
        num += M;
        while (num >= L) num -= L, base++;
      }
      const drop = Math.min(n, base - half + 1 - buf0); // nothing before this index is needed again
      if (drop > 0) { buf.copyWithin(0, drop, n); n -= drop; buf0 += drop; }
    },
  };
}

class Resample16k extends AudioWorkletProcessor {
  private rs = makeResampler(sampleRate, OUT_RATE);
  private out = new Float32Array(BLOCK);
  private fill = 0;
  private peak = 0;
  private mono = new Float32Array(128);
  private live = true;
  private emit = (s: number) => {
    this.out[this.fill++] = s;
    const a = s < 0 ? -s : s;
    if (a > this.peak) this.peak = a;
    if (this.fill === BLOCK) {
      const pcm = this.out;
      this.port.postMessage({ pcm, peak: this.peak }, [pcm.buffer]);
      this.out = new Float32Array(BLOCK); this.fill = 0; this.peak = 0;
    }
  };
  constructor() {
    super();
    this.port.onmessage = (e) => { if (e.data === "stop") this.live = false; };
    this.port.postMessage({ rate: sampleRate });
  }
  process(inputs: Float32Array[][]) {
    if (!this.live) return false;
    const ch = inputs[0];
    if (!ch || ch.length === 0 || ch[0].length === 0) return true; // nothing connected yet
    let x = ch[0];
    if (ch.length > 1) { // average the channels
      if (this.mono.length !== x.length) this.mono = new Float32Array(x.length);
      this.mono.fill(0);
      for (const c of ch) for (let i = 0; i < x.length; i++) this.mono[i] += c[i] / ch.length;
      x = this.mono;
    }
    this.rs.push(x, this.emit);
    return true;
  }
}
registerProcessor("pk-resample16k", Resample16k);
