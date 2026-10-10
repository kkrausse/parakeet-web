// Worker: owns the WASM module (transcribe.cpp + ggml WebGPU), the model file and the GPU device.
// The model is never copied into the WASM heap as a whole: it is stored as a file the browser keeps
// on disk (OPFS, or a fetch Blob) and mounted read-only through Emscripten's WORKERFS, so the
// loader's per-tensor reads pull one tensor at a time. store=memfs is the naive full-copy baseline.
export {};
declare const self: any;

let M: any = null;
let variant = "";
const post = (m: any, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const heapMb = () => (M ? Math.round(M.wasmMemory.buffer.byteLength / 2 ** 20) : 0);
const takeLog = () => (M ? (M.UTF8ToString(M.ccall("pk_log", "number", [], [])) as string) : "");

async function adapterInfo() {
  const gpu = (navigator as any).gpu as GPU | undefined;
  if (!gpu) return { available: false, reason: "navigator.gpu is undefined in the worker" };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { available: false, reason: "requestAdapter() returned null" };
    const i = adapter.info;
    return {
      available: true, vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description,
      isFallbackAdapter: (i as any).isFallbackAdapter ?? (adapter as any).isFallbackAdapter ?? null,
      shaderF16: adapter.features.has("shader-f16"), features: [...adapter.features].sort(),
      maxBufferSizeMb: Math.round(adapter.limits.maxBufferSize / 2 ** 20),
      maxStorageBufferBindingSizeMb: Math.round(adapter.limits.maxStorageBufferBindingSize / 2 ** 20),
    };
  } catch (e) { return { available: false, reason: String(e) }; }
}

async function init(msg: { base: string; variant: string }) {
  variant = msg.variant;
  const url = new URL(`${msg.variant === "asyncify" ? "pk-web-asyncify" : "pk-web"}.js`, msg.base).href;
  const t0 = performance.now();
  const { default: createPk } = await import(url);
  const lines: string[] = [];
  M = await createPk({ print: (s: string) => lines.push(s), printErr: (s: string) => { lines.push(s); post({ type: "stderr", line: s }); } });
  return { ms: performance.now() - t0, heapMb: heapMb(), adapter: await adapterInfo(), jspi: "Suspending" in WebAssembly };
}

/** Get the model as a disk-backed Blob (or bytes for memfs) without keeping it in JS memory. */
async function obtain(url: string, name: string, store: string, id: number): Promise<{ blob?: Blob; handle?: any; bytes?: Uint8Array; from: string; mb: number }> {
  const progress = (got: number, total: number) => post({ type: "progress", id, got, total });
  if (store === "opfs" || store === "opfs-blob") {
    const done = async (h: any, from: string) => {
      const f: File = await h.getFile();
      // opfs: read straight from a sync access handle into the WASM heap (no Blob slices, no garbage).
      return store === "opfs" ? { handle: await h.createSyncAccessHandle(), from, mb: f.size / 2 ** 20 } : { blob: f, from, mb: f.size / 2 ** 20 };
    };
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("pk-models", { create: true });
    const head = await fetch(url, { method: "HEAD" });
    const want = Number(head.headers.get("content-length") ?? 0);
    const h = await dir.getFileHandle(name, { create: true });
    let f = await h.getFile();
    if (want > 0 && f.size === want) return done(h, "OPFS (already stored)");
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const acc = await (h as any).createSyncAccessHandle();
    acc.truncate(0);
    const reader = resp.body!.getReader();
    let got = 0, last = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      acc.write(value, { at: got });
      got += value.length;
      if (got - last > 16 << 20) last = got, progress(got, want);
    }
    acc.flush(); acc.close();
    return done(h, "network, streamed to OPFS");
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  if (store === "blob") {
    const blob = await resp.blob();
    return { blob, from: "network, as a Blob", mb: blob.size / 2 ** 20 };
  }
  // memfs: the whole file inside the WASM heap (what a page gets with FS.writeFile). Baseline only.
  const bytes = new Uint8Array(await resp.arrayBuffer());
  return { bytes, from: "network, whole file copied into the WASM heap (MEMFS)", mb: bytes.length / 2 ** 20 };
}

async function load(msg: { id: number; url: string; name: string; store: string; env: Record<string, string>; threads: number; verbose: boolean }) {
  const t0 = performance.now();
  const got = await obtain(msg.url, msg.name, msg.store, msg.id);
  const fetchMs = performance.now() - t0;
  const FS = M.FS;
  try { FS.unmount("/m"); } catch { /* first load */ }
  try { FS.mkdir("/m"); } catch { /* exists */ }
  let path = `/m/${msg.name}`;
  if (got.blob) FS.mount(M.WORKERFS, { blobs: [{ name: msg.name, data: got.blob }] }, "/m");
  else if (got.handle) {
    // WORKERFS node for the directory plumbing, with read() replaced by the OPFS handle.
    const h = got.handle, size = h.getSize();
    FS.mount(M.WORKERFS, { blobs: [{ name: msg.name, data: { size } }] }, "/m");
    const node = FS.lookupPath(path).node;
    node.stream_ops = { ...M.WORKERFS.stream_ops, read(_s: any, buffer: Uint8Array, offset: number, length: number, position: number) {
      if (position >= size) return 0;
      return h.read(buffer.subarray(offset, offset + Math.min(length, size - position)), { at: position });
    } };
  }
  else path = `/${msg.name}`, FS.writeFile(path, got.bytes!), (got.bytes = undefined);
  for (const [k, v] of Object.entries(msg.env)) M.ccall("pk_setenv", null, ["string", "string"], [k, v]);
  M.ccall("pk_set_log_level", null, ["number"], [msg.verbose ? 1 : 0]);
  const t1 = performance.now();
  const st = await M.ccall("pk_load", "number", ["string", "number"], [path, msg.threads], { async: true });
  const loadMs = performance.now() - t1;
  const log = takeLog();
  got.handle?.close(); // nothing reads the file after load
  if (st !== 0) throw new Error(`pk_load failed with status ${st}\n${log}`);
  if (got.bytes === undefined && !got.blob && !got.handle) try { FS.unlink(path); } catch { /* keep */ }
  return { fetchMs, loadMs, from: got.from, mb: got.mb, heapMb: heapMb(), backend: M.UTF8ToString(M.ccall("pk_backend", "number", [], [])), log };
}

async function run(msg: { pcm: Float32Array }) {
  const n = msg.pcm.length;
  const ptr = M._malloc(n * 4);
  M.HEAPF32.set(msg.pcm, ptr >> 2);
  const t0 = performance.now();
  const st = await M.ccall("pk_run", "number", ["number", "number"], [ptr, n], { async: true });
  const wallMs = performance.now() - t0;
  M._free(ptr);
  const json = M.UTF8ToString(M.ccall("pk_json", "number", [], []));
  const log = takeLog();
  if (st !== 0) throw new Error(`pk_run failed with status ${st}: ${json}\n${log}`);
  return { ...JSON.parse(json), wallMs, heapMb: heapMb(), log };
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    const out = msg.type === "init" ? await init(msg) : msg.type === "load" ? await load(msg) : msg.type === "run" ? await run(msg)
      : msg.type === "free" ? await M.ccall("pk_free", null, [], [], { async: true }) : null;
    post({ type: "result", id: msg.id, out });
  } catch (err: any) {
    post({ type: "result", id: msg.id, error: `${err?.message ?? err}${M ? "\n" + takeLog() : ""}`.trim() });
  }
};
