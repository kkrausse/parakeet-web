// lib/src/worker.ts
var M = null;
var variant = "";
var post = (m, transfer = []) => self.postMessage(m, transfer);
var heapMb = () => M ? Math.round(M.wasmMemory.buffer.byteLength / 2 ** 20) : 0;
var usedMb = () => M ? Math.round(M.ccall("pk_heap_in_use", "number", [], []) / 2 ** 20 * 10) / 10 : 0;
var stage = (name, detail) => post({ type: "stage", name, detail });
var errText = (e) => `${e?.name ?? "Error"}: ${e?.message ?? e}`;
var takeLog = () => {
  try {
    return M ? M.UTF8ToString(M.ccall("pk_log", "number", [], [])) : "";
  } catch {
    return "";
  }
};
var MIB = 2 ** 20;
var mb1 = (bytes) => Math.round(bytes / MIB * 10) / 10;
var watch = { on: false, registry: false, kinds: {}, bufDestroyed: 0, bufLiveBytes: 0, bufCreatedBytes: 0 };
function installGpuWatch() {
  const g = self;
  if (watch.on || !g.GPUDevice || !g.GPUBuffer)
    return;
  watch.on = true;
  const FR = g.FinalizationRegistry;
  const reg = typeof FR === "function" ? new FR((kind) => {
    watch.kinds[kind].f++;
  }) : null;
  watch.registry = !!reg;
  const sizes = new WeakMap;
  const wrap = (proto, method, kind, made) => {
    const orig = proto?.[method];
    if (typeof orig !== "function")
      return;
    const k = watch.kinds[kind] ??= { c: 0, f: 0 };
    proto[method] = function(...args) {
      const o = orig.apply(this, args);
      if (o && typeof o === "object" && typeof o.then !== "function") {
        k.c++;
        reg?.register(o, kind);
        made?.(o, args);
      }
      return o;
    };
  };
  wrap(g.GPUDevice.prototype, "createBuffer", "buffer", (o, a) => {
    const n = Number(a[0]?.size ?? 0);
    sizes.set(o, n);
    watch.bufLiveBytes += n;
    watch.bufCreatedBytes += n;
  });
  wrap(g.GPUDevice.prototype, "createBindGroup", "bindGroup");
  wrap(g.GPUDevice.prototype, "createBindGroupLayout", "bindGroupLayout");
  wrap(g.GPUComputePipeline?.prototype, "getBindGroupLayout", "bindGroupLayout");
  wrap(g.GPUDevice.prototype, "createShaderModule", "shaderModule");
  wrap(g.GPUDevice.prototype, "createComputePipeline", "pipeline");
  wrap(g.GPUDevice.prototype, "createCommandEncoder", "commandEncoder");
  wrap(g.GPUCommandEncoder?.prototype, "finish", "commandBuffer");
  wrap(g.GPUCommandEncoder?.prototype, "beginComputePass", "computePass");
  const destroy = g.GPUBuffer.prototype.destroy;
  g.GPUBuffer.prototype.destroy = function() {
    const n = sizes.get(this);
    if (n !== undefined) {
      sizes.delete(this);
      watch.bufDestroyed++;
      watch.bufLiveBytes -= n;
    }
    return destroy.apply(this, arguments);
  };
}
function gpuBrief() {
  if (!M)
    return null;
  let s;
  try {
    s = JSON.parse(M.UTF8ToString(M.ccall("pk_gpu_stats", "number", [], [])));
  } catch {
    return null;
  }
  const byType = {};
  for (const [k, v] of Object.entries(s.weights_by_type ?? {}))
    byType[k] = mb1(v);
  const working = s.compute.bytes + s.compute_idle.bytes + s.staging.bytes + s.params.bytes + s.tensor_other.bytes + s.other.bytes;
  const out = {
    weightsMb: mb1(s.weights.bytes),
    workingMb: mb1(working),
    computeMb: mb1(s.compute.bytes + s.compute_idle.bytes),
    stagingMb: mb1(s.staging.bytes),
    liveMb: mb1(s.live_bytes),
    buffers: s.live_count,
    byType,
    created: s.buffers_created,
    destroyed: s.buffers_destroyed,
    createdMb: mb1(s.bytes_created),
    poolHits: s.pool_hits,
    poolMisses: s.pool_misses,
    pipelines: s.pipelines,
    layouts: s.bind_group_layouts,
    bindGroups: s.bind_groups,
    encoders: s.command_encoders,
    submits: s.submits,
    writes: s.write_buffers,
    maps: s.maps
  };
  if (watch.on) {
    const k = (name) => watch.kinds[name] ?? { c: 0, f: 0 };
    let made = 0, collected = 0;
    for (const v of Object.values(watch.kinds))
      made += v.c, collected += v.f;
    out.js = {
      registry: watch.registry,
      objects: made,
      collected,
      buffers: k("buffer").c,
      buffersDestroyed: watch.bufDestroyed,
      buffersCollected: k("buffer").f,
      bufferLiveMb: mb1(watch.bufLiveBytes),
      bindGroups: k("bindGroup").c,
      bindGroupsCollected: k("bindGroup").f,
      layouts: k("bindGroupLayout").c,
      encoders: k("commandEncoder").c,
      commandBuffers: k("commandBuffer").c,
      commandBuffersCollected: k("commandBuffer").f,
      pipelines: k("pipeline").c,
      shaderModules: k("shaderModule").c
    };
  }
  return out;
}
var gcMb = 0;
function gcNudge() {
  let n = 0;
  for (let i = 0;i < gcMb; i += 4)
    n += new ArrayBuffer(4 * MIB).byteLength;
  return n;
}
var SPEC_DEFAULT = {
  maxStorageBufferBindingSize: 134217728,
  maxBufferSize: 268435456,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupStorageSize: 16384,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupsPerDimension: 65535,
  maxBindGroups: 4,
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256
};
var GIB = 2 ** 30;
async function adapterInfo(wantF16, limits) {
  const gpu = navigator.gpu;
  if (!gpu)
    return { available: false, reason: "navigator.gpu is undefined in the worker (WebGPU is not enabled in this browser, or not exposed to workers)" };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter)
      return { available: false, reason: "requestAdapter() returned null (no usable GPU, or the browser blocks this one)" };
    const i = adapter.info ?? {};
    const lim = {};
    const below = [];
    for (const [k, d] of Object.entries(SPEC_DEFAULT)) {
      const v = Number(adapter.limits[k]);
      lim[k] = v;
      if (Number.isNaN(v))
        below.push(`${k} is not reported`);
      else if (k.startsWith("min") ? v > d : v < d)
        below.push(`${k} = ${v}, spec default ${d}`);
    }
    const allLimits = {};
    for (const k in adapter.limits) {
      const v = adapter.limits[k];
      if (typeof v === "number")
        allLimits[k] = v;
    }
    const f16 = adapter.features.has("shader-f16");
    const requiredFeatures = f16 && wantF16 ? ["shader-f16"] : [];
    const requiredLimits = limits === "default" ? {} : {
      maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, GIB),
      maxBufferSize: Math.min(lim.maxBufferSize, GIB),
      maxComputeInvocationsPerWorkgroup: Math.min(lim.maxComputeInvocationsPerWorkgroup, 256),
      maxComputeWorkgroupSizeX: Math.min(lim.maxComputeWorkgroupSizeX, 256)
    };
    const rungs = [{ limits, f16: requiredFeatures.length > 0, desc: { requiredFeatures, requiredLimits } }];
    if (limits !== "default")
      rungs.push({ limits: "default", f16: requiredFeatures.length > 0, desc: { requiredFeatures } });
    if (requiredFeatures.length)
      rungs.push({ limits: "default", f16: false, desc: {} });
    let device = null, deviceError = null, plan = null;
    const probeErrors = [];
    for (const r of rungs) {
      try {
        const a2 = await gpu.requestAdapter({ powerPreference: "high-performance" }) ?? adapter;
        const dev = await a2.requestDevice(r.desc);
        device = `ok: binding ${Math.round(dev.limits.maxStorageBufferBindingSize / 2 ** 20)} MiB, buffer ${Math.round(dev.limits.maxBufferSize / 2 ** 20)} MiB, ${dev.limits.maxComputeInvocationsPerWorkgroup} invocations`;
        dev.destroy();
        plan = { limits: r.limits, f16: r.f16 };
        break;
      } catch (e) {
        probeErrors.push(`requestDevice(${JSON.stringify(r.desc)}) -> ${e?.name ?? "Error"}: ${e?.message ?? e}`);
      }
    }
    if (!plan)
      deviceError = probeErrors.at(-1) ?? "requestDevice failed";
    return {
      available: true,
      vendor: i.vendor,
      architecture: i.architecture,
      device: i.device,
      description: i.description,
      isFallbackAdapter: i.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
      shaderF16: f16,
      features: [...adapter.features].sort(),
      wgslLanguageFeatures: [...gpu.wgslLanguageFeatures ?? []].sort(),
      limits: lim,
      allLimits,
      belowSpecDefault: below,
      requested: { requiredFeatures, requiredLimits },
      deviceProbe: device,
      deviceError,
      probeErrors,
      plan,
      maxBufferSizeMb: Math.round(lim.maxBufferSize / 2 ** 20),
      maxStorageBufferBindingSizeMb: Math.round(lim.maxStorageBufferBindingSize / 2 ** 20)
    };
  } catch (e) {
    return { available: false, reason: String(e) };
  }
}
var gpuErrorsSeen = 0;
function gpuErrors() {
  if (!M)
    return [];
  try {
    const n = M.ccall("pk_gpu_error_count", "number", [], []);
    if (n === gpuErrorsSeen)
      return [];
    const all = M.UTF8ToString(M.ccall("pk_gpu_errors", "number", [], [])).split(`
`).filter(Boolean);
    const fresh = all.slice(Math.min(gpuErrorsSeen, all.length));
    if (n > all.length && fresh.length)
      fresh.push(`(${n} device errors in total; only the first ${all.length} lines are kept)`);
    gpuErrorsSeen = n;
    return fresh;
  } catch {
    return [];
  }
}
async function init(msg) {
  variant = msg.variant;
  gcMb = Math.max(0, Number(msg.gcMb ?? 0) || 0);
  if (msg.gpuwatch !== false)
    try {
      installGpuWatch();
    } catch {}
  const url = new URL(`pk-web${msg.variant === "jspi" ? "" : "-" + msg.variant}.js`, msg.base).href;
  const t0 = performance.now();
  const { default: createPk } = await import(url);
  const lines = [];
  M = await createPk({ print: (s) => {
    lines.push(s);
    post({ type: "stdout", line: s });
  }, printErr: (s) => {
    lines.push(s);
    post({ type: "stderr", line: s });
  } });
  return { ms: performance.now() - t0, heapMb: heapMb(), adapter: await adapterInfo(msg.wantF16, msg.limits), jspi: "Suspending" in WebAssembly, gpuwatch: watch.on, finalizationRegistry: watch.registry, gcMb };
}
async function obtainOpfs(url, name, store, progress) {
  const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("pk-models", { create: true });
  const head = await fetch(url, { method: "HEAD" });
  const want = Number(head.headers.get("content-length") ?? 0);
  const h = await dir.getFileHandle(name, { create: true });
  const have = (await h.getFile()).size;
  stage(`model file: ${want} bytes on the server, ${have} bytes stored in this browser (OPFS)`);
  const open = async (from) => {
    const f = await h.getFile();
    if (store === "opfs") {
      try {
        const handle = await h.createSyncAccessHandle();
        stage("storage path: OPFS sync access handle (the file stays on disk; tensors are read one at a time)");
        return { handle, from, mb: f.size / 1048576, path: "opfs-sync" };
      } catch (e) {
        stage(`OPFS sync access handle unavailable for reading (${errText(e)}): reading the stored file as a File instead`);
      }
    }
    stage("storage path: OPFS File (the file stays on disk; read in slices)");
    return { blob: f, from, mb: f.size / 1048576, path: "opfs-file" };
  };
  if (want > 0 && have === want)
    return open("OPFS (already stored)");
  const resp = await fetch(url);
  if (!resp.ok)
    throw new Error(`fetch ${url}: ${resp.status}`);
  const reader = resp.body.getReader();
  let got = 0, last = 0, how = "";
  const tick = () => {
    if (got - last >= 16777216) {
      last = got;
      progress(got, want);
      stage(`download: ${got} of ${want} bytes written to OPFS (${how}), WASM heap ${heapMb()} MB`);
    }
  };
  let acc = null;
  try {
    if (store === "opfs-writable")
      throw new Error("store=opfs-writable");
    acc = await h.createSyncAccessHandle();
  } catch (e) {
    stage(`OPFS sync access handle unavailable for writing (${errText(e)}): trying createWritable`);
  }
  if (acc) {
    how = "sync access handle";
    try {
      acc.truncate(0);
      for (;; ) {
        const { done, value } = await reader.read();
        if (done)
          break;
        const n = acc.write(value, { at: got });
        if (n !== value.length)
          throw new Error(`short write to OPFS at byte ${got}: ${n} of ${value.length} (storage full?)`);
        got += value.length;
        tick();
      }
      acc.flush();
    } finally {
      acc.close();
    }
  } else {
    if (typeof h.createWritable !== "function")
      throw new Error("this browser's OPFS has neither createSyncAccessHandle nor createWritable");
    how = "createWritable";
    const w = await h.createWritable();
    try {
      for (;; ) {
        const { done, value } = await reader.read();
        if (done)
          break;
        await w.write(value);
        got += value.length;
        tick();
      }
    } catch (e) {
      try {
        await w.abort();
      } catch {}
      throw e;
    }
    await w.close();
  }
  stage(`download complete: ${got} bytes in OPFS (${how})${want > 0 && got !== want ? `; EXPECTED ${want}` : ""}`);
  return open("network, streamed to OPFS");
}
async function obtain(url, name, store, id) {
  const progress = (got, total) => post({ type: "progress", id, got, total });
  if (store === "opfs" || store === "opfs-blob" || store === "opfs-writable") {
    try {
      return await obtainOpfs(url, name, store, progress);
    } catch (e) {
      stage(`OPFS path FAILED (${errText(e)}): falling back to a fetch Blob, which the browser holds whole for this visit and fetches again next time`);
    }
  }
  const resp = await fetch(url);
  if (!resp.ok)
    throw new Error(`fetch ${url}: ${resp.status}`);
  if (store !== "memfs") {
    const blob = await resp.blob();
    stage(`storage path: fetch Blob (${blob.size} bytes held by the browser; not kept between visits)`);
    return { blob, from: "network, as a Blob", mb: blob.size / 1048576, path: "blob" };
  }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  stage(`storage path: MEMFS (${bytes.length} bytes copied into the WASM heap; baseline only)`);
  return { bytes, from: "network, whole file copied into the WASM heap (MEMFS)", mb: bytes.length / 1048576, path: "memfs" };
}
async function load(msg) {
  const t0 = performance.now();
  const got = await obtain(msg.url, msg.name, msg.store, msg.id);
  const fetchMs = performance.now() - t0;
  const FS = M.FS;
  try {
    FS.unmount("/m");
  } catch {}
  try {
    FS.mkdir("/m");
  } catch {}
  let path = `/m/${msg.name}`;
  if (got.blob)
    FS.mount(M.WORKERFS, { blobs: [{ name: msg.name, data: got.blob }] }, "/m");
  else if (got.handle) {
    const h = got.handle, size = h.getSize();
    FS.mount(M.WORKERFS, { blobs: [{ name: msg.name, data: { size } }] }, "/m");
    const node = FS.lookupPath(path).node;
    node.stream_ops = { ...M.WORKERFS.stream_ops, read(_s, buffer, offset, length, position) {
      if (position >= size)
        return 0;
      return h.read(buffer.subarray(offset, offset + Math.min(length, size - position)), { at: position });
    } };
  } else
    path = `/${msg.name}`, FS.writeFile(path, got.bytes), got.bytes = undefined;
  for (const [k, v] of Object.entries(msg.env))
    M.ccall("pk_setenv", null, ["string", "string"], [k, v]);
  M.ccall("pk_set_log_level", null, ["number"], [msg.verbose ? 1 : 0]);
  const t1 = performance.now();
  stage(`loading the model into the backend: WASM heap ${heapMb()} MB before`);
  const ticker = setInterval(() => post({ type: "diag", kind: "load-heap", data: { heapMb: heapMb(), ms: Math.round(performance.now() - t1) } }), 500);
  let st;
  try {
    st = await M.ccall("pk_load", "number", ["string", "number"], [path, msg.threads], { async: true });
  } finally {
    clearInterval(ticker);
  }
  const loadMs = performance.now() - t1;
  const log = takeLog();
  got.handle?.close();
  const gpuErr = gpuErrors();
  if (st !== 0)
    throw new Error(`pk_load failed with status ${st}
${gpuErr.join(`
`)}
${log}`);
  if (got.bytes === undefined && !got.blob && !got.handle)
    try {
      FS.unlink(path);
    } catch {}
  return { fetchMs, loadMs, from: got.from, storagePath: got.path, mb: got.mb, heapMb: heapMb(), heapUsedMb: usedMb(), backend: M.UTF8ToString(M.ccall("pk_backend", "number", [], [])), log, gpuErrors: gpuErr, gpu: gpuBrief() };
}
async function run(msg) {
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
  if (st !== 0) {
    let brief = json;
    try {
      const j = JSON.parse(json);
      delete j.text;
      brief = JSON.stringify(j);
    } catch {
      brief = `(${json.length} bytes of unparsed output)`;
    }
    throw new Error(`pk_run failed with status ${st}: ${brief}
${gpuErr.join(`
`)}
${log}`);
  }
  if (gcMb > 0)
    gcNudge();
  let words = [];
  try {
    words = JSON.parse(M.UTF8ToString(M.ccall("pk_words", "number", [], [])));
  } catch {}
  return { ...JSON.parse(json), words, gpuErrors: gpuErr, wallMs, heapMb: heapMb(), heapUsedMb: usedMb(), log, gpu: gpuBrief() };
}
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    const out = msg.type === "init" ? await init(msg) : msg.type === "load" ? await load(msg) : msg.type === "run" ? await run(msg) : msg.type === "trim" ? (M.ccall("pk_trim", null, [], []), gpuBrief()) : msg.type === "gpu" ? gpuBrief() : msg.type === "set" ? (gcMb = Math.max(0, Number(msg.gcMb ?? gcMb) || 0), { gcMb }) : msg.type === "free" ? await M.ccall("pk_free", null, [], [], { async: true }) : null;
    post({ type: "result", id: msg.id, out });
  } catch (err) {
    post({ type: "result", id: msg.id, error: `${err?.message ?? err}${M ? `
` + gpuErrors().join(`
`) + `
` + takeLog() : ""}`.trim(), stack: String(err?.stack ?? "") });
  }
};
