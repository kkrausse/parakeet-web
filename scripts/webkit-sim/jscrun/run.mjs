// jsc -m [JSC options] run.mjs -- BUILD PASSES : the page's CPU path in the jsc shell (ASYNCIFY build, no WebGPU).
// Prints pass times; the caller measures peak memory and CPU with /usr/bin/time -l.
const [build = "new", passesArg = "60", capArg = "3"] = globalThis.arguments ?? [];
globalThis.self = globalThis;
globalThis.window = globalThis;
globalThis.console ??= { log: print, warn: print, error: print, info: print, debug: print };
globalThis.document ??= { currentScript: null };
globalThis.setTimeout ??= (f) => Promise.resolve().then(f);
globalThis.location = { href: "file:///jscrun/" };
globalThis.navigator ??= { userAgent: "jsc", hardwareConcurrency: 1 };
globalThis.crypto ??= { getRandomValues(a) { for (let i = 0; i < a.length; i++) a[i] = (Math.random() * 256) | 0; return a; } };
globalThis.performance ??= { now: () => preciseTime() * 1000 };
const now = () => preciseTime() * 1000;
// the module fetches its .wasm: hand it the file
globalThis.fetch = async (u) => { const bytes = read(`pk-${build}.wasm`, "binary"); return { ok: true, status: 200, url: String(u), arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) }; };
const { default: createPk } = await import(`./pk-${build}.mjs`);
const lines = [];
const M = await createPk({ print: (s) => lines.push(s), printErr: (s) => lines.push(s), locateFile: (f) => f });
M.FS.writeFile("/m.gguf", read("m.gguf", "binary"));
const env = { TRANSCRIBE_BACKENDS: "cpu", TRANSCRIBE_NO_FLASH: "1", GGML_WEBGPU_NO_F16: "1", TRANSCRIBE_F32_POINTWISE: "1", TRANSCRIBE_PRE_ENCODE_TILE: "128", TRANSCRIBE_ENC_PROJ_GPU: "1", TRANSCRIBE_MEL_REAL_FFT: "1",
  TRANSCRIBE_PARAKEET_CHUNK_S: "30", TRANSCRIBE_PARAKEET_CHUNK_HALO_S: "4", TRANSCRIBE_PARAKEET_CHUNK_MIN_S: "60" };
for (const [k, v] of Object.entries(env)) M.ccall("pk_setenv", null, ["string", "string"], [k, v]);
let t = now();
const st = await M.ccall("pk_load", "number", ["string", "number"], ["/m.gguf", 1], { async: true });
print(`load ${st} in ${Math.round(now() - t)} ms, backend ${M.UTF8ToString(M.ccall("pk_backend", "number", [], []))}, heap ${Math.round(M.wasmMemory.buffer.byteLength / 2 ** 20)} MB`);
const clip = new Float32Array(read("a30.f32", "binary").buffer);
let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const n = Number(passesArg), cap = Number(capArg), ms = [];
const t0 = now();
for (let i = 1; i <= n; i++) {
  const len = Math.round((0.3 + rnd() * (cap - 0.3)) * 16000), off = Math.floor(rnd() * (clip.length - len));
  const ptr = M._malloc(len * 4);
  M.HEAPF32.set(clip.subarray(off, off + len), ptr >> 2);
  t = now();
  let r;
  try { r = await M.ccall("pk_run", "number", ["number", "number"], [ptr, len], { async: true }); }
  catch (e) { print(`pass ${i} FAILED: ${e}\n${String(e.stack).split("\n").slice(0, 14).join("\n")}`); break; }
  ms.push(now() - t);
  M._free(ptr);
  if (r !== 0) { print(`pass ${i}: status ${r}`); break; }
  if (i === 1 || i % 20 === 0) print(`pass ${i}: ${Math.round(ms.at(-1))} ms for ${(len / 16000).toFixed(1)} s, heap ${Math.round(M.wasmMemory.buffer.byteLength / 2 ** 20)} MB, ${Math.round((now() - t0) / 1000)} s in`);
}
const s = [...ms].sort((a, b) => a - b);
print(`done ${ms.length} of ${n} passes, median ${Math.round(s[s.length >> 1] ?? 0)} ms, total ${Math.round((now() - t0) / 1000)} s`);
