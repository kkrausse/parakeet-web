// lib/src/worklet.ts
var OUT_RATE = 16000;
var BLOCK = 640;
var ZERO_CROSSINGS = 12;
var KAISER_BETA = 8.6;
var MAX_PHASES = 512;
var gcd = (a, b) => b ? gcd(b, a % b) : a;
function besselI0(x) {
  let sum = 1, term = 1;
  for (let k = 1;k < 40; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < 0.000000000001 * sum)
      break;
  }
  return sum;
}
function makeResampler(inRate, outRate) {
  if (inRate === outRate)
    return { push(input, emit) {
      for (let i = 0;i < input.length; i++)
        emit(input[i]);
    } };
  const g = gcd(inRate, outRate);
  const L = outRate / g, M = inRate / g;
  const fc = 0.5 * Math.min(1, outRate / inRate) * 0.94;
  const half = Math.ceil(ZERO_CROSSINGS / (2 * fc));
  const taps = 2 * half;
  const phases = Math.min(L, MAX_PHASES);
  const table = new Float32Array(phases * taps);
  const i0b = besselI0(KAISER_BETA);
  for (let p = 0;p < phases; p++) {
    const frac = p / phases;
    let sum = 0;
    for (let k = 0;k < taps; k++) {
      const x = k - half + 1 - frac;
      const r = x / half;
      const w = Math.abs(r) >= 1 ? 0 : besselI0(KAISER_BETA * Math.sqrt(1 - r * r)) / i0b;
      const a = 2 * Math.PI * fc * x;
      const v = (Math.abs(a) < 0.000000001 ? 1 : Math.sin(a) / a) * w;
      table[p * taps + k] = v;
      sum += v;
    }
    for (let k = 0;k < taps; k++)
      table[p * taps + k] /= sum;
  }
  let buf = new Float32Array(taps + 4096), n = half - 1, buf0 = -(half - 1);
  let base = 0, num = 0;
  return {
    push(input, emit) {
      if (n + input.length > buf.length) {
        const b = new Float32Array((n + input.length) * 2);
        b.set(buf.subarray(0, n));
        buf = b;
      }
      buf.set(input, n);
      n += input.length;
      for (;; ) {
        let phase = L === phases ? num : Math.round(num / L * phases), b = base;
        if (phase === phases)
          phase = 0, b++;
        const first = b - half + 1 - buf0;
        if (first + taps > n)
          break;
        let acc = 0;
        const t = phase * taps;
        for (let k = 0;k < taps; k++)
          acc += buf[first + k] * table[t + k];
        emit(acc);
        num += M;
        while (num >= L)
          num -= L, base++;
      }
      const drop = Math.min(n, base - half + 1 - buf0);
      if (drop > 0) {
        buf.copyWithin(0, drop, n);
        n -= drop;
        buf0 += drop;
      }
    }
  };
}

class Resample16k extends AudioWorkletProcessor {
  rs = makeResampler(sampleRate, OUT_RATE);
  out = new Float32Array(BLOCK);
  fill = 0;
  peak = 0;
  mono = new Float32Array(128);
  live = true;
  emit = (s) => {
    this.out[this.fill++] = s;
    const a = s < 0 ? -s : s;
    if (a > this.peak)
      this.peak = a;
    if (this.fill === BLOCK) {
      const pcm = this.out;
      this.port.postMessage({ pcm, peak: this.peak }, [pcm.buffer]);
      this.out = new Float32Array(BLOCK);
      this.fill = 0;
      this.peak = 0;
    }
  };
  constructor() {
    super();
    this.port.onmessage = (e) => {
      if (e.data === "stop")
        this.live = false;
    };
    this.port.postMessage({ rate: sampleRate });
  }
  process(inputs) {
    if (!this.live)
      return false;
    const ch = inputs[0];
    if (!ch || ch.length === 0 || ch[0].length === 0)
      return true;
    let x = ch[0];
    if (ch.length > 1) {
      if (this.mono.length !== x.length)
        this.mono = new Float32Array(x.length);
      this.mono.fill(0);
      for (const c of ch)
        for (let i = 0;i < x.length; i++)
          this.mono[i] += c[i] / ch.length;
      x = this.mono;
    }
    this.rs.push(x, this.emit);
    return true;
  }
}
registerProcessor("pk-resample16k", Resample16k);
