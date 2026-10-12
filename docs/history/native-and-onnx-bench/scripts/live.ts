// Client-side timings against the running dictation service (it is not restarted or reconfigured).
// Run: bun scripts/live.ts [url=http://127.0.0.1:9876] [runs=10] CLIP.f32 ...
// Each clip is streamed in 80 ms frames: `runs` times as fast as the service acknowledges, then once
// on a microphone's schedule. A chunk time is frame sent -> ack for the frames that ran inference,
// as in ../../dictation-server-linux/scripts/measure.ts. Prints one JSON object per clip.
import { readFileSync } from "node:fs";

const [url = "http://127.0.0.1:9876", runsArg = "10", ...clips] = process.argv.slice(2);
const runs = Number(runsArg);
const pause = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const stats = (values: number[]) => {
  const sorted = values.toSorted((a, b) => a - b);
  const r = (v: number) => Math.round(v * 10) / 10;
  return sorted.length ? { median: r(sorted[sorted.length >> 1]!), worst: r(sorted.at(-1)!), best: r(sorted[0]!) } : null;
};

async function record(audio: Uint8Array, realtime: boolean) {
  const id = crypto.randomUUID();
  const ws = new WebSocket(`${url.replace(/^http/, "ws")}/v1/stream`);
  const sentAt: number[] = [];
  const chunks: { at: number; ms: number }[] = [];
  let slow = false, acked = 0, final = "", failure: string | undefined, ready = false, done = false;
  ws.onmessage = event => {
    const value = JSON.parse(String(event.data));
    if (value.type === "ready") ready = true;
    if (value.type === "ack") {
      const ms = performance.now() - sentAt[acked]!;
      if (ms > 5 && !slow) chunks.push({ at: acked * 0.08, ms });
      slow = ms > 5;
      acked++;
    }
    if (value.type === "final") final = value.text;
    if (value.type === "error") failure = value.code;
    if (value.type === "done") done = true;
  };
  ws.onopen = () => ws.send(JSON.stringify({ type: "start", version: 1, recordingId: id, sampleRate: 16000, channels: 1, format: "f32le" }));
  const until = async (condition: () => boolean) => { while (!condition()) { if (failure) throw new Error(failure); await pause(1); } };
  await until(() => ready);
  const begun = performance.now();
  for (let offset = 0, frame = 0; offset < audio.length; offset += 5120, frame++) {
    // Real time: a microphone's schedule. Otherwise one frame in flight, so each chunk time is clean.
    if (realtime) await pause(begun + (frame + 1) * 80 - performance.now());
    else await until(() => acked === frame);
    sentAt.push(performance.now());
    ws.send(audio.subarray(offset, offset + 5120));
  }
  await until(() => acked === sentAt.length);
  const stop = performance.now();
  ws.send(JSON.stringify({ type: "stop", recordingId: id }));
  await until(() => done);
  // Chunks from 6.2 s on are encoded with the full 5.6 s of left context.
  return { text: final, full: chunks.filter(c => c.at >= 6.16).map(c => c.ms), all: chunks.map(c => c.ms), streamMs: stop - begun, flushMs: performance.now() - stop };
}

const status = await (await fetch(`${url}/v1/status`)).json() as any;
console.log(JSON.stringify({ record: "load", runtime: "live dictation service", url, modelId: status.modelId, state: status.state }));
for (const path of clips) {
  const audio = new Uint8Array(readFileSync(path));
  const fast = [];
  for (let i = 0; i < runs; i++) fast.push(await record(audio, false));
  const paced = await record(audio, true);
  console.log(JSON.stringify({
    record: "audio", file: path, seconds: Math.round(audio.length / 640) / 100, warm_runs: runs,
    back_to_back: { full_window_chunk_ms: stats(fast.flatMap(r => r.full)), all_chunk_ms: stats(fast.flatMap(r => r.all)),
      stream_ms: stats(fast.map(r => r.streamMs)), flush_ms: stats(fast.map(r => r.flushMs)), total_ms: stats(fast.map(r => r.streamMs + r.flushMs)) },
    real_time: { full_window_chunk_ms: stats(paced.full), all_chunk_ms: stats(paced.all), flush_ms: Math.round(paced.flushMs) },
    text: fast[0]!.text, text_stable: fast.every(r => r.text === fast[0]!.text) && paced.text === fast[0]!.text,
  }));
}
process.exit(0);
