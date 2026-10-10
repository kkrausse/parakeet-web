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
const usedMb = () => (M ? Math.round((M.ccall("pk_heap_in_use", "number", [], []) / 2 ** 20) * 10) / 10 : 0);
/** A load stage for the page's step trail (persisted and sent to the collector). */
const stage = (name: string, detail?: string) => post({ type: "stage", name, detail });
const errText = (e: any) => `${e?.name ?? "Error"}: ${e?.message ?? e}`;
const takeLog = () => { try { return M ? (M.UTF8ToString(M.ccall("pk_log", "number", [], [])) as string) : ""; } catch { return ""; } }; // throws once the module has aborted

// ---------- GPU accounting ----------
// Two independent counts. (1) The backend's own (pk_gpu_stats): every buffer it created and destroyed, live bytes
// by purpose, weight bytes by tensor type, and how many bind groups / encoders / submits / queue writes it made.
// (2) This worker's view of the JS objects (gpuwatch, on unless gpuwatch=0): GPUDevice / GPUBuffer methods are
// wrapped, so objects created, buffers destroy()ed and objects the JS garbage collector has since collected are
// counted without trusting the C++ side. A WebGPU object other than a buffer has no destroy(): what it holds on the
// GPU side is released when the collector finds the JS object, and the collector does not see GPU memory.
const MIB = 2 ** 20;
const mb1 = (bytes: number) => Math.round((bytes / MIB) * 10) / 10;
interface Kind { c: number; f: number } // created, finalized (collected by the JS GC)
const watch = { on: false, registry: false, kinds: {} as Record<string, Kind>, bufDestroyed: 0, bufLiveBytes: 0, bufCreatedBytes: 0 };
function installGpuWatch() {
  const g: any = self;
  if (watch.on || !g.GPUDevice || !g.GPUBuffer) return;
  watch.on = true;
  const FR: any = (g as any).FinalizationRegistry;
  const reg = typeof FR === "function" ? new FR((kind: string) => { watch.kinds[kind].f++; }) : null;
  watch.registry = !!reg;
  const sizes = new WeakMap<object, number>();
  const wrap = (proto: any, method: string, kind: string, made?: (o: any, args: any[]) => void) => {
    const orig = proto?.[method];
    if (typeof orig !== "function") return;
    const k = (watch.kinds[kind] ??= { c: 0, f: 0 });
    proto[method] = function (this: any, ...args: any[]) {
      const o = orig.apply(this, args);
      if (o && typeof o === "object" && typeof o.then !== "function") { k.c++; reg?.register(o, kind); made?.(o, args); }
      return o;
    };
  };
  wrap(g.GPUDevice.prototype, "createBuffer", "buffer", (o, a) => { const n = Number(a[0]?.size ?? 0); sizes.set(o, n); watch.bufLiveBytes += n; watch.bufCreatedBytes += n; });
  wrap(g.GPUDevice.prototype, "createBindGroup", "bindGroup");
  wrap(g.GPUDevice.prototype, "createBindGroupLayout", "bindGroupLayout");
  wrap(g.GPUComputePipeline?.prototype, "getBindGroupLayout", "bindGroupLayout");
  wrap(g.GPUDevice.prototype, "createShaderModule", "shaderModule");
  wrap(g.GPUDevice.prototype, "createComputePipeline", "pipeline");
  wrap(g.GPUDevice.prototype, "createCommandEncoder", "commandEncoder");
  wrap(g.GPUCommandEncoder?.prototype, "finish", "commandBuffer");
  wrap(g.GPUCommandEncoder?.prototype, "beginComputePass", "computePass");
  const destroy = g.GPUBuffer.prototype.destroy;
  g.GPUBuffer.prototype.destroy = function (this: any) {
    const n = sizes.get(this);
    if (n !== undefined) { sizes.delete(this); watch.bufDestroyed++; watch.bufLiveBytes -= n; }
    return destroy.apply(this, arguments as any);
  };
}
/** Compact numbers for a pass line: MiB to one decimal, cumulative counts. null when the backend has no GPU stats. */
function gpuBrief() {
  if (!M) return null;
  let s: any;
  try { s = JSON.parse(M.UTF8ToString(M.ccall("pk_gpu_stats", "number", [], []))); } catch { return null; }
  const byType: Record<string, number> = {};
  for (const [k, v] of Object.entries(s.weights_by_type ?? {})) byType[k] = mb1(v as number);
  const working = s.compute.bytes + s.compute_idle.bytes + s.staging.bytes + s.params.bytes + s.tensor_other.bytes + s.other.bytes;
  const out: any = {
    weightsMb: mb1(s.weights.bytes), workingMb: mb1(working), computeMb: mb1(s.compute.bytes + s.compute_idle.bytes), stagingMb: mb1(s.staging.bytes), liveMb: mb1(s.live_bytes), buffers: s.live_count,
    byType, created: s.buffers_created, destroyed: s.buffers_destroyed, createdMb: mb1(s.bytes_created), poolHits: s.pool_hits, poolMisses: s.pool_misses,
    pipelines: s.pipelines, layouts: s.bind_group_layouts, bindGroups: s.bind_groups, encoders: s.command_encoders, submits: s.submits, writes: s.write_buffers, maps: s.maps,
  };
  if (watch.on) {
    const k = (name: string) => watch.kinds[name] ?? { c: 0, f: 0 };
    let made = 0, collected = 0;
    for (const v of Object.values(watch.kinds)) made += v.c, collected += v.f;
    out.js = { registry: watch.registry, objects: made, collected, buffers: k("buffer").c, buffersDestroyed: watch.bufDestroyed, buffersCollected: k("buffer").f, bufferLiveMb: mb1(watch.bufLiveBytes),
      bindGroups: k("bindGroup").c, bindGroupsCollected: k("bindGroup").f, layouts: k("bindGroupLayout").c, encoders: k("commandEncoder").c, commandBuffers: k("commandBuffer").c, commandBuffersCollected: k("commandBuffer").f,
      pipelines: k("pipeline").c, shaderModules: k("shaderModule").c };
  }
  return out;
}
// gc=<MiB>: after each pass that many MiB of ArrayBuffers are allocated and dropped at once. A browser's collector
// is driven by how much JS memory was allocated; a pass makes about a thousand small WebGPU objects whose JS side is
// a few bytes each, so without this nothing tells it that they (and what they hold in the GPU process) are garbage.
// The buffers are never written, so they cost address space, not memory. 0 turns it off.
let gcMb = 0;
function gcNudge() {
  let n = 0;
  for (let i = 0; i < gcMb; i += 4) n += new ArrayBuffer(4 * MIB).byteLength;
  return n;
}

// What the backend needs from WebGPU. It requests no limit above the spec default except the two byte
// sizes, each at most 1 GiB and never more than the adapter offers; everything works at the defaults,
// only a tensor larger than maxStorageBufferBindingSize then runs on the CPU (slow, same text).
const SPEC_DEFAULT: Record<string, number> = {
  maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456, maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupStorageSize: 16384, maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupsPerDimension: 65535, maxBindGroups: 4, minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256,
};
const GIB = 2 ** 30;

async function adapterInfo(wantF16: boolean, limits: string) {
  const gpu = (navigator as any).gpu as GPU | undefined;
  if (!gpu) return { available: false, reason: "navigator.gpu is undefined in the worker (WebGPU is not enabled in this browser, or not exposed to workers)" };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { available: false, reason: "requestAdapter() returned null (no usable GPU, or the browser blocks this one)" };
    const i = adapter.info ?? ({} as GPUAdapterInfo);
    const lim: Record<string, number> = {};
    const below: string[] = [];
    for (const [k, d] of Object.entries(SPEC_DEFAULT)) {
      const v = Number((adapter.limits as any)[k]);
      lim[k] = v;
      if (Number.isNaN(v)) below.push(`${k} is not reported`);
      else if (k.startsWith("min") ? v > d : v < d) below.push(`${k} = ${v}, spec default ${d}`);
    }
    const allLimits: Record<string, number> = {}; // everything the adapter reports, for the diagnostics
    for (const k in adapter.limits) { const v = (adapter.limits as any)[k]; if (typeof v === "number") allLimits[k] = v; }
    const f16 = adapter.features.has("shader-f16");
    // The same request the WASM backend makes, tried from JS first so a rejection shows the browser's own message.
    const requiredFeatures = (f16 && wantF16 ? ["shader-f16"] : []) as GPUFeatureName[];
    const requiredLimits: Record<string, number> = limits === "default" ? {} : {
      maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, GIB), maxBufferSize: Math.min(lim.maxBufferSize, GIB),
      maxComputeInvocationsPerWorkgroup: Math.min(lim.maxComputeInvocationsPerWorkgroup, 256), maxComputeWorkgroupSizeX: Math.min(lim.maxComputeWorkgroupSizeX, 256),
    };
    // Ladder: what the backend would ask for; then the spec-default limits; then no feature either.
    const rungs: { limits: string; f16: boolean; desc: GPUDeviceDescriptor }[] = [{ limits, f16: requiredFeatures.length > 0, desc: { requiredFeatures, requiredLimits } }];
    if (limits !== "default") rungs.push({ limits: "default", f16: requiredFeatures.length > 0, desc: { requiredFeatures } });
    if (requiredFeatures.length) rungs.push({ limits: "default", f16: false, desc: {} });
    let device: string | null = null, deviceError: string | null = null, plan: { limits: string; f16: boolean } | null = null;
    const probeErrors: string[] = [];
    for (const r of rungs) {
      try {
        const a2 = (await gpu.requestAdapter({ powerPreference: "high-performance" })) ?? adapter;
        const dev = await a2.requestDevice(r.desc);
        device = `ok: binding ${Math.round(dev.limits.maxStorageBufferBindingSize / 2 ** 20)} MiB, buffer ${Math.round(dev.limits.maxBufferSize / 2 ** 20)} MiB, ${dev.limits.maxComputeInvocationsPerWorkgroup} invocations`;
        dev.destroy();
        plan = { limits: r.limits, f16: r.f16 };
        break;
      } catch (e: any) { probeErrors.push(`requestDevice(${JSON.stringify(r.desc)}) -> ${e?.name ?? "Error"}: ${e?.message ?? e}`); }
    }
    if (!plan) deviceError = probeErrors.at(-1) ?? "requestDevice failed";
    return {
      available: true, vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description,
      isFallbackAdapter: (i as any).isFallbackAdapter ?? (adapter as any).isFallbackAdapter ?? null,
      shaderF16: f16, features: [...adapter.features].sort(),
      wgslLanguageFeatures: [...((gpu as any).wgslLanguageFeatures ?? [])].sort(),
      limits: lim, allLimits, belowSpecDefault: below, requested: { requiredFeatures, requiredLimits }, deviceProbe: device, deviceError, probeErrors, plan,
      maxBufferSizeMb: Math.round(lim.maxBufferSize / 2 ** 20),
      maxStorageBufferBindingSizeMb: Math.round(lim.maxStorageBufferBindingSize / 2 ** 20),
    };
  } catch (e) { return { available: false, reason: String(e) }; }
}

/** New device errors since the last call (shader compilation with the compiler's messages, validation, out of memory, device lost). */
let gpuErrorsSeen = 0;
function gpuErrors(): string[] {
  if (!M) return [];
  try {
    const n = M.ccall("pk_gpu_error_count", "number", [], []) as number;
    if (n === gpuErrorsSeen) return [];
    const all = (M.UTF8ToString(M.ccall("pk_gpu_errors", "number", [], [])) as string).split("\n").filter(Boolean);
    const fresh = all.slice(Math.min(gpuErrorsSeen, all.length));
    if (n > all.length && fresh.length) fresh.push(`(${n} device errors in total; only the first ${all.length} lines are kept)`);
    gpuErrorsSeen = n;
    return fresh;
  } catch { return []; } // the module aborted
}

async function init(msg: { base: string; variant: string; wantF16: boolean; limits: string; gpuwatch?: boolean; gcMb?: number }) {
  variant = msg.variant;
  gcMb = Math.max(0, Number(msg.gcMb ?? 0) || 0);
  if (msg.gpuwatch !== false) try { installGpuWatch(); } catch { /* counters only */ }
  const url = new URL(`pk-web${msg.variant === "jspi" ? "" : "-" + msg.variant}.js`, msg.base).href;
  const t0 = performance.now();
  const { default: createPk } = await import(url);
  const lines: string[] = [];
  M = await createPk({ print: (s: string) => { lines.push(s); post({ type: "stdout", line: s }); }, printErr: (s: string) => { lines.push(s); post({ type: "stderr", line: s }); } });
  return { ms: performance.now() - t0, heapMb: heapMb(), adapter: await adapterInfo(msg.wantF16, msg.limits), jspi: "Suspending" in WebAssembly, gpuwatch: watch.on, finalizationRegistry: watch.registry, gcMb };
}

interface Got { blob?: Blob; handle?: any; bytes?: Uint8Array; from: string; mb: number; path: string }
/** The model file kept in OPFS; downloaded once, streamed to disk chunk by chunk (never whole in memory). */
async function obtainOpfs(url: string, name: string, store: string, progress: (got: number, total: number) => void): Promise<Got> {
  const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("pk-models", { create: true });
  const head = await fetch(url, { method: "HEAD" });
  const want = Number(head.headers.get("content-length") ?? 0);
  const h: any = await dir.getFileHandle(name, { create: true });
  const have = (await h.getFile()).size;
  stage(`model file: ${want} bytes on the server, ${have} bytes stored in this browser (OPFS)`);
  const open = async (from: string): Promise<Got> => {
    const f: File = await h.getFile();
    if (store === "opfs") {
      // read straight from a sync access handle into the WASM heap, one tensor at a time (no Blob slices, no garbage)
      try {
        const handle = await h.createSyncAccessHandle();
        stage("storage path: OPFS sync access handle (the file stays on disk; tensors are read one at a time)");
        return { handle, from, mb: f.size / 2 ** 20, path: "opfs-sync" };
      } catch (e) { stage(`OPFS sync access handle unavailable for reading (${errText(e)}): reading the stored file as a File instead`); }
    }
    stage("storage path: OPFS File (the file stays on disk; read in slices)");
    return { blob: f, from, mb: f.size / 2 ** 20, path: "opfs-file" };
  };
  if (want > 0 && have === want) return open("OPFS (already stored)");
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  const reader = resp.body!.getReader();
  let got = 0, last = 0, how = "";
  const tick = () => { if (got - last >= 16 << 20) { last = got; progress(got, want); stage(`download: ${got} of ${want} bytes written to OPFS (${how}), WASM heap ${heapMb()} MB`); } };
  let acc: any = null;
  // store=opfs-writable: as if there were no sync access handle (createWritable to store, a File to read)
  try { if (store === "opfs-writable") throw new Error("store=opfs-writable"); acc = await h.createSyncAccessHandle(); } catch (e) { stage(`OPFS sync access handle unavailable for writing (${errText(e)}): trying createWritable`); }
  if (acc) {
    how = "sync access handle";
    try {
      acc.truncate(0);
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const n = acc.write(value, { at: got });
        if (n !== value.length) throw new Error(`short write to OPFS at byte ${got}: ${n} of ${value.length} (storage full?)`);
        got += value.length;
        tick();
      }
      acc.flush();
    } finally { acc.close(); }
  } else {
    if (typeof h.createWritable !== "function") throw new Error("this browser's OPFS has neither createSyncAccessHandle nor createWritable");
    how = "createWritable";
    const w = await h.createWritable();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        await w.write(value);
        got += value.length;
        tick();
      }
    } catch (e) { try { await w.abort(); } catch { /* gone */ } throw e; }
    await w.close();
  }
  stage(`download complete: ${got} bytes in OPFS (${how})${want > 0 && got !== want ? `; EXPECTED ${want}` : ""}`);
  return open("network, streamed to OPFS");
}

/** Get the model as a disk-backed file (or bytes for memfs) without keeping it in JS memory. Every path taken is a stage. */
async function obtain(url: string, name: string, store: string, id: number): Promise<Got> {
  const progress = (got: number, total: number) => post({ type: "progress", id, got, total });
  if (store === "opfs" || store === "opfs-blob" || store === "opfs-writable") {
    try { return await obtainOpfs(url, name, store, progress); }
    catch (e) { stage(`OPFS path FAILED (${errText(e)}): falling back to a fetch Blob, which the browser holds whole for this visit and fetches again next time`); }
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  if (store !== "memfs") {
    const blob = await resp.blob();
    stage(`storage path: fetch Blob (${blob.size} bytes held by the browser; not kept between visits)`);
    return { blob, from: "network, as a Blob", mb: blob.size / 2 ** 20, path: "blob" };
  }
  // memfs: the whole file inside the WASM heap (what a page gets with FS.writeFile). Baseline only.
  const bytes = new Uint8Array(await resp.arrayBuffer());
  stage(`storage path: MEMFS (${bytes.length} bytes copied into the WASM heap; baseline only)`);
  return { bytes, from: "network, whole file copied into the WASM heap (MEMFS)", mb: bytes.length / 2 ** 20, path: "memfs" };
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
  stage(`loading the model into the backend: WASM heap ${heapMb()} MB before`);
  // While pk_load is suspended on the GPU the worker's timers run: the heap size goes to the collector twice a second.
  const ticker = setInterval(() => post({ type: "diag", kind: "load-heap", data: { heapMb: heapMb(), ms: Math.round(performance.now() - t1) } }), 500);
  let st: number;
  try { st = await M.ccall("pk_load", "number", ["string", "number"], [path, msg.threads], { async: true }); } finally { clearInterval(ticker); }
  const loadMs = performance.now() - t1;
  const log = takeLog();
  got.handle?.close(); // nothing reads the file after load
  const gpuErr = gpuErrors();
  if (st !== 0) throw new Error(`pk_load failed with status ${st}\n${gpuErr.join("\n")}\n${log}`);
  if (got.bytes === undefined && !got.blob && !got.handle) try { FS.unlink(path); } catch { /* keep */ }
  return { fetchMs, loadMs, from: got.from, storagePath: got.path, mb: got.mb, heapMb: heapMb(), heapUsedMb: usedMb(), backend: M.UTF8ToString(M.ccall("pk_backend", "number", [], [])), log, gpuErrors: gpuErr, gpu: gpuBrief() };
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
  const gpuErr = gpuErrors();
  if (st !== 0) { // the message goes to the diagnostics collector: no transcript text in it
    let brief = json;
    try { const j = JSON.parse(json); delete j.text; brief = JSON.stringify(j); } catch { brief = `(${json.length} bytes of unparsed output)`; }
    throw new Error(`pk_run failed with status ${st}: ${brief}\n${gpuErr.join("\n")}\n${log}`);
  }
  if (gcMb > 0) gcNudge();
  return { ...JSON.parse(json), gpuErrors: gpuErr, wallMs, heapMb: heapMb(), heapUsedMb: usedMb(), log, gpu: gpuBrief() };
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    const out = msg.type === "init" ? await init(msg) : msg.type === "load" ? await load(msg) : msg.type === "run" ? await run(msg)
      : msg.type === "trim" ? (M.ccall("pk_trim", null, [], []), gpuBrief())
      : msg.type === "gpu" ? gpuBrief()
      : msg.type === "set" ? ((gcMb = Math.max(0, Number(msg.gcMb ?? gcMb) || 0)), { gcMb })
      : msg.type === "free" ? await M.ccall("pk_free", null, [], [], { async: true }) : null;
    post({ type: "result", id: msg.id, out });
  } catch (err: any) {
    post({ type: "result", id: msg.id, error: `${err?.message ?? err}${M ? "\n" + gpuErrors().join("\n") + "\n" + takeLog() : ""}`.trim(), stack: String(err?.stack ?? "") });
  }
};
