// lib/src/runtime.ts
function directoryUrl(url) {
  const u = new URL(String(url), globalThis.document?.baseURI ?? location.href);
  if (!u.pathname.endsWith("/"))
    u.pathname += "/";
  return u.href;
}
var absoluteUrl = (url) => new URL(String(url), globalThis.document?.baseURI ?? location.href).href;
function detectDevice() {
  const ua = navigator.userAgent;
  const ios = /iPhone|iPad|iPod/.test(ua) || /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
  const phone = ios || /Android/.test(ua);
  return { ios, phone, why: ios ? "iOS/iPadOS user agent" : phone ? "Android user agent" : "desktop user agent" };
}
function formatGpuStats(g, prev) {
  if (!g)
    return "GPU: none (CPU backend)";
  const types = Object.entries(g.byType ?? {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(", ");
  let s = `GPU: weights ${g.weightsMb} MiB${types ? ` (${types})` : ""}, working ${g.workingMb} MiB, ${g.buffers} buffers`;
  if (prev)
    s += `, last pass +${g.created - prev.created} buffers +${g.bindGroups - prev.bindGroups} bind groups +${g.encoders - prev.encoders} encoders +${g.writes - prev.writes} writes`;
  s += ` · since load ${g.created} buffers made, ${g.destroyed} destroyed`;
  if (g.js)
    s += ` · JS objects ${g.js.objects} made, ${g.js.registry ? `${g.js.collected} collected` : "collection not observable"}; JS buffers ${g.js.buffers - g.js.buffersDestroyed} live (${g.js.bufferLiveMb} MiB)`;
  return s;
}
function createRpc(url, onEvent) {
  const worker = new Worker(url, { type: "module" });
  let nextId = 1;
  const pending = new Map;
  const failAll = (why) => {
    for (const p of pending.values())
      p.reject(new Error(why));
    pending.clear();
  };
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type !== "result")
      return onEvent(m);
    const p = pending.get(m.id);
    if (!p)
      return;
    pending.delete(m.id);
    if (m.error && m.stack)
      onEvent({ type: "stack", stack: m.stack });
    m.error ? p.reject(new Error(m.error)) : p.resolve(m.out);
  };
  worker.onerror = (e) => failAll(e.message ? `worker error: ${e.message}` : `the worker could not be started from ${url} (is assetsUrl the folder holding worker.js, on this origin?)`);
  return {
    call(type, body = {}, transfer = []) {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ type, id, ...body }, transfer);
      });
    },
    terminate() {
      worker.terminate();
      failAll("worker terminated");
    }
  };
}
function adapterSteps(a, step) {
  step(a.available ? `adapter: ${a.vendor} ${a.architecture} ${a.description || a.device || ""}; shader-f16 ${a.shaderF16 ? "present" : "ABSENT"}; fallback=${a.isFallbackAdapter}` : `no WebGPU adapter: ${a.reason}`);
  if (!a.available)
    return step("WebGPU is unavailable: the model would run on one WASM thread (very slow)");
  const L = a.limits, mib = (x) => Math.round(x / 2 ** 20);
  step(`adapter limits: binding ${mib(L.maxStorageBufferBindingSize)} MiB, buffer ${mib(L.maxBufferSize)} MiB, ${L.maxComputeInvocationsPerWorkgroup} invocations/workgroup, workgroup storage ${L.maxComputeWorkgroupStorageSize} B, ${L.maxStorageBuffersPerShaderStage} storage buffers/stage`, undefined, a.belowSpecDefault.length ? `BELOW THE WEBGPU SPEC DEFAULT: ${a.belowSpecDefault.join("; ")}` : "all at or above the WebGPU spec defaults, which is all this runtime needs");
  for (const e of a.probeErrors ?? [])
    step(`device request FAILED: ${e}`);
  if (!a.plan)
    throw new Error(`WebGPU is present but no device could be created, even with the spec-default limits and no features: ${a.deviceError}`);
  step(`device probe ${a.deviceProbe}`, undefined, a.plan.limits === "default" ? "spec-default limits" : `requested ${JSON.stringify(a.requested.requiredLimits)} ${a.requested.requiredFeatures.join(",") || "no features"}`);
  if (L.maxStorageBufferBindingSize < 2 ** 28)
    step(`note: tensors above ${mib(L.maxStorageBufferBindingSize)} MiB cannot be bound on this adapter; long clips may run partly on the CPU`);
}
function backendEnv(a, o) {
  const useF16 = !!a.shaderF16 && o.f16 && a.plan?.f16 !== false;
  const flash = o.flash && useF16;
  const pathLine = useF16 ? "shader path: f16" : `shader path: f32 only (${!a.shaderF16 ? "adapter has no shader-f16" : o.f16 ? "the device was refused with shader-f16" : "f16 turned off"})`;
  const env = {
    TRANSCRIBE_NO_FLASH: flash ? "" : "1",
    TRANSCRIBE_F32_MASK_CONCAT: flash ? "1" : "",
    GGML_WEBGPU_NO_F16: useF16 ? "" : "1",
    TRANSCRIBE_F32_POINTWISE: useF16 ? "" : "1",
    TRANSCRIBE_PRE_ENCODE_TILE: "128",
    TRANSCRIBE_ENC_PROJ_GPU: "1",
    TRANSCRIBE_MEL_REAL_FFT: "1",
    TRANSCRIBE_PARAKEET_CHUNK_S: "30",
    TRANSCRIBE_PARAKEET_CHUNK_HALO_S: "4",
    TRANSCRIBE_PARAKEET_CHUNK_MIN_S: "60",
    ...a.plan?.limits === "default" ? { GGML_WEBGPU_LIMITS: "default" } : {},
    ...o.extra
  };
  return { useF16, flash, pathLine, env };
}
function noise(samples, seed = 1) {
  const out = new Float32Array(samples);
  for (let i = 0;i < out.length; i++) {
    seed = seed * 1664525 + 1013904223 >>> 0;
    out[i] = (seed / 2 ** 32 - 0.5) * 0.002;
  }
  return { pcm: out, seed };
}

// lib/src/microphone.ts
function isMicrophoneError(e) {
  return e instanceof Error && typeof e.code === "string" && e.microphone === true;
}
function micError(code, message, cause) {
  return Object.assign(new Error(message), { name: "MicrophoneError", code, cause, microphone: true });
}
function fromDomError(e) {
  if (isMicrophoneError(e))
    return e;
  const n = e?.name ?? "";
  if (n === "NotAllowedError" || n === "SecurityError")
    return micError("permission-denied", "Microphone access was refused. Allow the microphone for this site in the browser's settings, then start again.", e);
  if (n === "NotFoundError" || n === "OverconstrainedError")
    return micError("no-device", "No microphone was found on this device.", e);
  if (n === "NotReadableError" || n === "AbortError")
    return micError("device-busy", "The microphone could not be opened (another app may be using it).", e);
  return micError("failed", `The microphone could not be started: ${n} ${e?.message ?? e}`.trim(), e);
}
var SILENT_AFTER_MS = 4000;
var START_TIMEOUT_MS = 30000;
function createMicrophone(options) {
  const workletUrl = new URL("worklet.js", directoryUrl(options.assetsUrl)).href;
  let status = "off";
  let session = 0;
  let ctx = null, stream = null, node = null, wake = null;
  let startedAt = 0, sawSignal = false;
  const setStatus = (s) => {
    if (s !== status) {
      status = s;
      options.onStatus?.(s);
    }
  };
  const diag = (kind, data) => options.onDiagnostic?.(kind, data);
  async function acquireWake() {
    if (options.wakeLock === false)
      return;
    try {
      wake = await navigator.wakeLock?.request("screen");
    } catch {
      wake = null;
    }
  }
  function onVisible() {
    if (document.visibilityState !== "visible" || status === "off" || status === "starting")
      return;
    ctx?.resume().catch(() => {});
    if (!wake || wake.released)
      acquireWake();
  }
  function release() {
    document.removeEventListener("visibilitychange", onVisible);
    try {
      node?.port.postMessage("stop");
      node?.disconnect();
    } catch {}
    stream?.getTracks().forEach((t) => t.stop());
    if (ctx)
      ctx.onstatechange = null;
    ctx?.close().catch(() => {});
    try {
      wake?.release();
    } catch {}
    node = null;
    stream = null;
    ctx = null;
    wake = null;
  }
  function stop() {
    session++;
    release();
    setStatus("off");
  }
  async function start(overrides = {}) {
    if (status !== "off")
      throw micError("failed", "The microphone is already started.");
    if (!navigator.mediaDevices?.getUserMedia)
      throw micError(isSecureContext ? "unsupported" : "insecure-context", isSecureContext ? "This browser has no microphone capture (getUserMedia)." : "Microphone capture needs https (or localhost); this page was opened over plain http.");
    const AC = window.AudioContext ?? window.webkitAudioContext;
    if (!AC)
      throw micError("unsupported", "This browser has no Web Audio (AudioContext).");
    const mine = ++session;
    setStatus("starting");
    const c = ctx = new AC({ latencyHint: "interactive" });
    c.resume().catch(() => {});
    const current = () => {
      if (mine !== session)
        throw micError("stopped", "The microphone was stopped while it was starting.");
    };
    try {
      if (!c.audioWorklet)
        throw micError("unsupported", "AudioWorklet is not available in this browser.");
      const proc = overrides.voiceProcessing ?? options.voiceProcessing ?? false;
      const asked = navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: proc, noiseSuppression: proc, autoGainControl: true }, video: false });
      const limit = options.startTimeoutMs ?? START_TIMEOUT_MS;
      let timer;
      const s = await (limit > 0 ? Promise.race([asked, new Promise((_, rej) => {
        timer = setTimeout(() => rej(micError("timeout", `The browser did not answer the microphone request within ${Math.round(limit / 1000)} s (no permission prompt answered, or capture is not available here).`)), limit);
      })]) : asked).catch((e) => {
        asked.then((late) => late.getTracks().forEach((t) => t.stop()), () => {});
        throw e;
      }).finally(() => clearTimeout(timer));
      if (mine !== session) {
        s.getTracks().forEach((t) => t.stop());
        current();
      }
      stream = s;
      await c.audioWorklet.addModule(workletUrl).catch((e) => {
        throw micError("failed", `The audio worklet could not be loaded from ${workletUrl} (is assetsUrl the folder holding worklet.js?): ${e?.message ?? e}`, e);
      });
      current();
      if (c.state !== "running")
        await c.resume().catch(() => {});
      current();
      const src = c.createMediaStreamSource(s);
      const n = node = new AudioWorkletNode(c, "pk-resample16k", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers" });
      n.port.onmessage = (e) => {
        if (node !== n || mine !== session || status === "starting")
          return;
        if (e.data.rate)
          return;
        if (e.data.peak > 0)
          sawSignal = true;
        options.onAudio(e.data.pcm, e.data.peak);
        if (mine !== session)
          return;
        if (!sawSignal && performance.now() - startedAt > SILENT_AFTER_MS) {
          if (status === "live")
            setStatus("silent");
        } else if (sawSignal && status === "silent")
          setStatus("live");
      };
      const mute = c.createGain();
      mute.gain.value = 0;
      src.connect(n);
      n.connect(mute);
      mute.connect(c.destination);
      const track = s.getAudioTracks()[0];
      track?.addEventListener("ended", () => {
        if (stream !== s)
          return;
        stop();
        options.onEnded?.(micError("ended", "The microphone went away; recording stopped."));
      });
      for (const evName of ["mute", "unmute", "ended"])
        track?.addEventListener(evName, () => diag("mic-track", { event: evName, readyState: track.readyState, muted: track.muted }));
      c.onstatechange = () => {
        diag("audio-context", { state: c.state, sampleRate: c.sampleRate, recording: status !== "off" && status !== "starting" });
        if (ctx !== c || status === "off" || status === "starting")
          return;
        if (c.state !== "running")
          setStatus("interrupted");
        else if (status === "interrupted")
          setStatus(sawSignal ? "live" : performance.now() - startedAt > SILENT_AFTER_MS ? "silent" : "live");
      };
      document.addEventListener("visibilitychange", onVisible);
      const set = { ...track?.getSettings?.() ?? {}, deviceId: undefined, groupId: undefined };
      startedAt = performance.now();
      sawSignal = false;
      setStatus("live");
      acquireWake();
      return { sampleRate: c.sampleRate, contextState: c.state, baseLatency: c.baseLatency ?? null, track: set, trackLabelPresent: !!track?.label, voiceProcessing: proc };
    } catch (e) {
      const err = Object.assign(fromDomError(e), { contextState: c.state });
      if (mine === session) {
        release();
        setStatus("off");
      } else
        c.close().catch(() => {});
      throw err;
    }
  }
  return { start, stop, get status() {
    return status;
  } };
}

// lib/src/stream.ts
var SAMPLE_RATE = 16000;
var SR = SAMPLE_RATE;
var FRAME = 320;
var GRID = 1280;
var DEFAULT_PARAMS = {
  capS: 24,
  gapMs: 250,
  minNewS: 0.1,
  stableS: 0.5,
  minPasses: 2,
  holdS: 1.2,
  keepS: 4,
  emptyHoldS: 1,
  forceHoldS: 1,
  lagS: 3,
  backlogS: 30,
  hangS: 1.5,
  preRollS: 0.4,
  gateRatio: 2.5,
  gateAbsRms: 0.0004,
  floorWindowS: 4,
  onsetFrames: 2,
  probeEveryS: 2,
  probeS: 3,
  silencePeak: 0.0001,
  maxWordS: 1.2
};
var normalizeWord = (w) => w.toLowerCase().replace(/[^\p{L}\p{N}']+/gu, "");
var outWord = (x) => ({ w: x.w, s: x.s / SR, e: x.e / SR, p: x.p });
function createStream(transcribe, events = {}, options = {}) {
  const { now = () => performance.now(), ...over } = options;
  const P = { ...DEFAULT_PARAMS, ...over };
  const capN = Math.round(P.capS * SR), capacity = capN + Math.round(P.backlogS * SR);
  const closedKeepN = Math.round(Math.max(P.probeS, P.preRollS + 0.2) * SR);
  const buf = new Float32Array(capacity + 4 * FRAME);
  let start = 0, len = 0, captured = 0;
  const subN = 25, subs = new Float32Array(Math.max(2, Math.round(P.floorWindowS * 50 / subN)));
  let subFill = 0, subCur = Infinity, subAt = 0, floor = 0, lastRms = 0, speech = false, run = 0, r1 = 0, r2 = 0, frames = 0;
  const frame = new Float32Array(FRAME);
  let frameFill = 0;
  let lastSpeech = -1;
  let open = false, openedBy = "gate", uttStart = 0, uttWords = 0;
  let boundary = 0;
  let tail = [], tent = [];
  let hypEnd = 0;
  let drain = null;
  let skipFrom = -1, holding = false;
  let busy = false, failed = false, gen = 0, epoch = 0, lastPassStart = -Infinity, lastProbeAt = 0, closedSince = 0;
  let stopWaiters = [];
  const zero = () => ({
    capturedS: 0,
    open: false,
    busy: false,
    failed: false,
    skipping: false,
    bufferS: 0,
    maxBufferS: 0,
    lagS: 0,
    maxLagS: 0,
    noiseFloor: 0,
    threshold: 0,
    lastRms: 0,
    speech: false,
    gateOpens: 0,
    gateMisses: 0,
    modelHolds: 0,
    probes: 0,
    probesSilent: 0,
    passes: 0,
    livePasses: 0,
    finalPasses: 0,
    stale: 0,
    reusedFinals: 0,
    utterances: 0,
    commits: 0,
    wordsCommitted: 0,
    byReason: { agree: 0, idle: 0, stop: 0, cap: 0, lag: 0, skip: 0 },
    skippedS: 0,
    trims: 0,
    anchorShifts: 0
  });
  let S = zero();
  const threshold = () => Math.max(P.gateAbsRms, floor * P.gateRatio);
  const end = () => start + len;
  function shiftTo(abs) {
    const n = Math.min(len, Math.max(0, abs - start));
    if (n <= 0)
      return;
    buf.copyWithin(0, n, len);
    len -= n;
    start += n;
  }
  function gateFrame(endAbs) {
    let s = 0;
    for (let i = 0;i < FRAME; i++)
      s += frame[i] * frame[i];
    const r = Math.sqrt(s / FRAME);
    const sm = frames >= 2 ? (r + r1 + r2) / 3 : r;
    r2 = r1;
    r1 = r;
    frames++;
    if (sm < subCur)
      subCur = sm;
    if (++subAt >= subN) {
      subs[subFill++ % subs.length] = subCur;
      subCur = Infinity;
      subAt = 0;
    }
    let f = subCur;
    for (let i = 0;i < Math.min(subFill, subs.length); i++)
      if (subs[i] < f)
        f = subs[i];
    floor = f;
    lastRms = r;
    speech = r > threshold();
    run = speech ? run + 1 : 0;
    if (run >= P.onsetFrames) {
      lastSpeech = endAbs;
      if (!open && !drain)
        openUtterance(Math.max(start, closedSince, endAbs - run * FRAME - Math.round(P.preRollS * SR)), "gate");
    }
  }
  function openUtterance(at, by) {
    at = start + Math.floor(Math.max(0, at - start) / GRID) * GRID;
    shiftTo(at);
    open = true;
    openedBy = by;
    uttStart = start;
    uttWords = 0;
    boundary = start;
    tail = [];
    tent = [];
    hypEnd = start;
    gen++;
    if (by === "gate")
      S.gateOpens++;
    else
      S.gateMisses++;
    events.onGate?.({ open: true, atSample: captured, by: by === "gate" ? "energy" : "probe", noiseFloor: floor, threshold: threshold() });
  }
  function closeUtterance(endAbs, reason) {
    const u = { startSample: uttStart, endSample: endAbs, reason, words: uttWords, opened: openedBy };
    open = false;
    drain = null;
    tent = [];
    tail = [];
    gen++;
    shiftTo(endAbs);
    closedSince = endAbs;
    lastProbeAt = captured;
    boundary = endAbs;
    hypEnd = endAbs;
    S.utterances++;
    events.onTentative?.("", [], captured);
    events.onGate?.({ open: false, atSample: captured, by: reason, noiseFloor: floor, threshold: threshold() });
    events.onUtteranceEnd?.(u);
  }
  function push(pcm) {
    if (failed) {
      captured += pcm.length;
      return;
    }
    const at = captured;
    captured += pcm.length;
    if (skipFrom >= 0)
      return void pump();
    if (len + pcm.length > capacity) {
      skipFrom = at;
      if (open)
        drain = "skip";
      else {
        shiftTo(end());
        endSkip();
      }
      return void pump();
    }
    buf.set(pcm, len);
    len += pcm.length;
    for (let i = 0;i < pcm.length; i++) {
      frame[frameFill++] = pcm[i];
      if (frameFill === FRAME) {
        frameFill = 0;
        gateFrame(at + i + 1);
      }
    }
    if (!open) {
      if (len > closedKeepN + SR / 2)
        shiftTo(end() - closedKeepN);
    } else if (!drain) {
      if (quietNow() && hypEnd > uttStart)
        drain = "idle";
    }
    if (len / SR > S.maxBufferS)
      S.maxBufferS = len / SR;
    const lag = open ? (end() - hypEnd) / SR : 0;
    if (lag > S.maxLagS)
      S.maxLagS = lag;
    pump();
  }
  function endSkip() {
    if (skipFrom < 0)
      return;
    S.skippedS += (captured - skipFrom) / SR;
    events.onSkip?.({ fromSample: skipFrom, toSample: captured });
    skipFrom = -1;
    start = captured;
    len = 0;
    closedSince = captured;
    lastProbeAt = captured;
    frameFill = 0;
    run = 0;
  }
  async function pump() {
    if (busy || failed)
      return;
    let kind, from = 0, n = 0;
    if (open) {
      if (drain) {
        if (drain === "idle" && hypEnd >= Math.max(lastSpeech + 0.3 * SR, end() - 0.5 * SR)) {
          S.reusedFinals++;
          commit(tent.length, "idle");
          finishDrain(hypEnd);
          return pump();
        }
        kind = "final";
      } else {
        if (end() - hypEnd < P.minNewS * SR || now() - lastPassStart < P.gapMs)
          return;
        kind = "live";
      }
      n = Math.min(len, capN);
      if (n < FRAME) {
        if (drain)
          finishDrain(end());
        return;
      }
    } else {
      if (!P.probeEveryS || captured - lastProbeAt < P.probeEveryS * SR)
        return;
      from = Math.max(0, len - Math.round(P.probeS * SR), closedSince - start);
      from = Math.ceil(from / GRID) * GRID;
      n = len - from;
      if (n < SR)
        return;
      lastProbeAt = captured;
      let peak = 0;
      for (let i = from;i < len && peak < P.silencePeak; i += 2) {
        const a = Math.abs(buf[i]);
        if (a > peak)
          peak = a;
      }
      if (peak < P.silencePeak) {
        S.probesSilent++;
        return;
      }
      kind = "probe";
      S.probes++;
    }
    const req = { kind, startSample: start + from, endSample: start + from + n };
    const myGen = gen, myEpoch = epoch, t0 = now();
    busy = true;
    lastPassStart = t0;
    let r;
    try {
      r = await transcribe(buf.slice(from, from + n), req);
    } catch (e) {
      if (myEpoch !== epoch)
        return;
      busy = false;
      failed = true;
      events.onError?.(e);
      settle();
      return;
    }
    if (myEpoch !== epoch)
      return;
    busy = false;
    S.passes++;
    if (kind === "final")
      S.finalPasses++;
    else if (kind === "live")
      S.livePasses++;
    const ev = { ...req, ms: now() - t0, words: r.words?.length ?? 0, committed: 0, tentative: 0, stale: myGen !== gen, atSample: captured };
    if (myGen !== gen)
      S.stale++;
    else {
      const before = S.wordsCommitted;
      if (kind === "probe")
        onProbe(req, r);
      else
        onPass(req, r, kind === "final");
      ev.committed = S.wordsCommitted - before;
      ev.tentative = tent.length;
    }
    events.onPass?.(ev);
    settle();
    return pump();
  }
  function settle() {
    if (!idle())
      return;
    const w = stopWaiters;
    stopWaiters = [];
    for (const f of w)
      f();
  }
  function toAbs(r, startSample) {
    const out = [];
    for (const x of r.words ?? []) {
      const n = normalizeWord(x.w);
      if (!n)
        continue;
      const ws = startSample + Math.round(x.s * SR), we = Math.max(ws, Math.min(startSample + Math.round(x.e * SR), ws + Math.round(P.maxWordS * SR)));
      out.push({ w: x.w, n, s: ws, e: we, p: x.p ?? 1, since: 0, seen: 1 });
    }
    return out;
  }
  function firstFresh(h) {
    let k0 = 0;
    while (k0 < h.length && (h[k0].s + h[k0].e) / 2 < boundary)
      k0++;
    if (!tail.length || tail[tail.length - 1].e < boundary - GRID)
      return k0;
    const lastE = tail[tail.length - 1].e;
    const match = (k) => {
      let m = 0;
      while (m < 4 && m < k && m < tail.length && h[k - 1 - m].n === tail[tail.length - 1 - m].n)
        m++;
      return m;
    };
    let best = k0, bestM = match(k0);
    for (const d of [-1, 1, -2, 2]) {
      const k = k0 + d;
      if (k < 1 || k > h.length || Math.abs(h[k - 1].e - lastE) > 0.4 * SR)
        continue;
      const m = match(k);
      if (m > bestM)
        best = k, bestM = m;
    }
    if (best !== k0)
      S.anchorShifts++;
    return best;
  }
  function commit(n, reason) {
    if (n <= 0)
      return;
    const ws = tent.splice(0, n);
    const lastE = ws[ws.length - 1].e;
    boundary = Math.max(boundary, tent.length ? Math.min(lastE, tent[0].s) : lastE);
    tail.push(...ws);
    if (tail.length > 48)
      tail = tail.slice(-24);
    uttWords += n;
    S.commits++;
    S.wordsCommitted += n;
    S.byReason[reason] += n;
    events.onCommit?.({ text: ws.map((x) => x.w).join(" "), words: ws.map(outWord), reason, atSample: captured });
  }
  function onPass(req, r, final) {
    const h = toAbs(r, req.startSample), passEnd = req.endSample;
    const fresh = h.slice(firstFresh(h));
    let j = 0;
    for (const x of fresh) {
      x.since = passEnd;
      x.seen = 1;
      for (let k = j;k < tent.length && tent[k].s <= x.s + 3 * GRID; k++) {
        if (tent[k].n === x.n && Math.abs(tent[k].s - x.s) <= 3 * GRID) {
          x.since = tent[k].since;
          x.seen = tent[k].seen + 1;
          j = k + 1;
          break;
        }
      }
    }
    tent = fresh;
    hypEnd = passEnd;
    const full = req.endSample - req.startSample >= capN - SR / 2, behind = end() - passEnd;
    if (drain === "idle" && !quietNow())
      drain = null;
    if (final && drain && behind <= 0) {
      commit(tent.length, drain);
      return finishDrain(passEnd);
    }
    const hurry = drain === "stop" || drain === "skip" ? drain : behind >= P.lagS * SR ? "lag" : null;
    let n = 0;
    while (n < tent.length && passEnd - tent[n].e >= P.holdS * SR && (hurry || tent[n].seen >= P.minPasses && passEnd - tent[n].since >= P.stableS * SR))
      n++;
    commit(n, hurry ?? "agree");
    if (full) {
      let m = 0;
      while (m < tent.length && passEnd - tent[m].s >= P.forceHoldS * SR)
        m++;
      commit(m, hurry ?? "cap");
      boundary = Math.max(boundary, passEnd - Math.round(P.keepS * SR));
    }
    boundary = Math.max(boundary, Math.min(passEnd, tent.length ? tent[0].s : passEnd) - Math.round(P.emptyHoldS * SR));
    trim(behind >= P.lagS * SR);
    events.onTentative?.(tent.map((x) => x.w).join(" "), tent.map(outWord), captured);
  }
  function quietNow() {
    const lastWord = tent.length ? tent[tent.length - 1].e : tail.length ? tail[tail.length - 1].e : uttStart;
    const gateQuiet = captured - Math.max(lastSpeech, 0) >= P.hangS * SR, modelQuiet = captured - lastWord >= P.hangS * SR;
    if (gateQuiet && !modelQuiet && lastSpeech < lastWord - 0.3 * SR) {
      if (!holding)
        holding = true, S.modelHolds++;
    } else
      holding = false;
    return gateQuiet && modelQuiet;
  }
  function finishDrain(endAbs) {
    const reason = drain ?? "idle";
    closeUtterance(endAbs, reason);
    if (reason === "skip") {
      shiftTo(end());
      endSkip();
    }
    settle();
  }
  function trim(lagging) {
    const target = boundary - Math.round((lagging ? Math.min(P.keepS, 1) : P.keepS) * SR);
    if (target - start < SR / 2)
      return;
    let cut = target, gap = -1;
    for (let i = tail.length - 1;i >= 0; i--) {
      const w = tail[i];
      if (w.s > target + SR / 2)
        continue;
      if (w.s < target - SR / 2)
        break;
      const prevE = i > 0 ? tail[i - 1].e : w.s, g = w.s - prevE;
      if (g > gap)
        gap = g, cut = Math.round((prevE + w.s) / 2);
    }
    cut = start + Math.floor((Math.min(cut, boundary) - start) / GRID) * GRID;
    if (cut <= start)
      return;
    shiftTo(cut);
    S.trims++;
    let drop = 0;
    while (drop < tail.length && tail[drop].e <= start)
      drop++;
    if (drop)
      tail.splice(0, drop);
  }
  function onProbe(req, r) {
    if (open || drain || skipFrom >= 0)
      return;
    const h = toAbs(r, req.startSample);
    if (!h.length)
      return;
    openUtterance(req.startSample, "probe");
    lastSpeech = Math.max(lastSpeech, h[h.length - 1].e);
    for (const x of h)
      x.since = req.endSample;
    tent = h;
    hypEnd = req.endSample;
    events.onTentative?.(tent.map((x) => x.w).join(" "), tent.map(outWord), captured);
  }
  function idle() {
    return !busy && !open && skipFrom < 0;
  }
  function stop() {
    if (failed)
      return Promise.resolve();
    if (open && drain !== "skip")
      drain = "stop";
    if (!open) {
      gen++;
      if (skipFrom >= 0)
        endSkip();
    }
    if (idle())
      return Promise.resolve();
    const p = new Promise((res) => stopWaiters.push(res));
    pump();
    return p;
  }
  function reset() {
    gen++;
    epoch++;
    busy = false;
    failed = false;
    start = 0;
    len = 0;
    captured = 0;
    open = false;
    drain = null;
    skipFrom = -1;
    holding = false;
    tent = [];
    tail = [];
    boundary = 0;
    hypEnd = 0;
    uttStart = 0;
    uttWords = 0;
    subFill = 0;
    subCur = Infinity;
    subAt = 0;
    floor = 0;
    lastRms = 0;
    speech = false;
    run = 0;
    r1 = r2 = 0;
    frames = 0;
    frameFill = 0;
    lastSpeech = -1;
    lastPassStart = -Infinity;
    lastProbeAt = 0;
    closedSince = 0;
    S = zero();
    settle();
  }
  function stats() {
    return {
      ...S,
      byReason: { ...S.byReason },
      capturedS: captured / SR,
      open,
      busy,
      failed,
      skipping: skipFrom >= 0,
      bufferS: len / SR,
      lagS: open ? Math.max(0, (end() - hypEnd) / SR) : 0,
      noiseFloor: floor,
      threshold: threshold(),
      lastRms,
      speech
    };
  }
  return { push, stop, reset, idle, stats, params: P };
}

// lib/src/transcriber.ts
function isTranscriberError(e) {
  return e instanceof Error && typeof e.kind === "string" && Array.isArray(e.gpuErrors);
}
function fail(e, kind, where, gpuErrors = []) {
  const err = e instanceof Error ? e : new Error(String(e));
  return Object.assign(err, { kind, where, gpuErrors, deviceLost: gpuErrors.some((x) => /device lost/i.test(x)) });
}
var abortError = () => {
  try {
    return new DOMException("the load was aborted", "AbortError");
  } catch {
    return Object.assign(new Error("the load was aborted"), { name: "AbortError" });
  }
};
async function createTranscriber(options) {
  const adv = options.advanced ?? {};
  const assets = directoryUrl(options.assetsUrl), modelUrl = absoluteUrl(options.modelUrl);
  const modelFile = decodeURIComponent(new URL(modelUrl).pathname.split("/").pop() || "model.gguf");
  const variant = adv.variant ?? ("Suspending" in WebAssembly ? "jspi" : "asyncify");
  const cpu = options.backend === "cpu", verbose = !!adv.verbose, wantF16 = adv.f16 !== false;
  const device = detectDevice();
  const limits = adv.env?.GGML_WEBGPU_LIMITS ?? (adv.gpuLimits ? adv.gpuLimits === "default" ? "default" : "" : device.ios ? "default" : "");
  const signal = options.signal;
  let state = "loading";
  let silent = false;
  const quiet = () => silent || !!signal?.aborted;
  const step = (name, ms, detail, sensitive) => {
    if (!quiet())
      adv.onStep?.({ name, ms: ms === undefined ? undefined : Math.round(ms * 10) / 10, detail, ...sensitive ? { sensitive } : {} });
  };
  const diag = (kind, data) => {
    if (!quiet())
      adv.onDiagnostic?.(kind, data);
  };
  const progress = (p) => {
    if (!quiet())
      options.onProgress?.(p);
  };
  if (signal?.aborted)
    throw abortError();
  progress({ phase: "runtime", fraction: 0.02 });
  const rpc = createRpc(new URL("worker.js", assets).href, (m) => {
    if (quiet())
      return;
    if (m.type === "progress")
      return progress({ phase: "model", loadedBytes: m.got, totalBytes: m.total, fraction: 0.05 + 0.8 * (m.got / Math.max(1, m.total)) });
    if (m.type === "stage")
      return step(m.name, undefined, m.detail);
    if (m.type === "diag")
      return diag(m.kind, m.data);
    if (m.type === "stdout") {
      if (verbose)
        step(`wasm: ${String(m.line).slice(0, 400)}`, undefined, undefined, true);
      return;
    }
    if (m.type === "stderr") {
      if (verbose || /error|abort|fail/i.test(m.line))
        step(`wasm: ${String(m.line).slice(0, 400)}`);
      return;
    }
    if (m.type === "stack" && verbose)
      step(`stack: ${String(m.stack).slice(0, 2500)}`);
  });
  const onAbort = () => rpc.terminate();
  signal?.addEventListener("abort", onAbort, { once: true });
  let gpu = null;
  const gpuFailure = (where, errs) => {
    if (!errs?.length)
      return null;
    for (const e of errs.slice(0, 12))
      step(`GPU ERROR (${where}): ${e.slice(0, 700)}`);
    return fail(new Error(`GPU error (${where}): ${errs.slice(0, 4).join(`
`).slice(0, 900)}`), "gpu", where, errs);
  };
  let info;
  try {
    const init = await rpc.call("init", { base: assets, variant, wantF16, limits, gpuwatch: adv.gpuWatch !== false, gcMb: adv.gcMb ?? 0 });
    const a = init.adapter;
    diag("adapter", { adapter: a, jspi: init.jspi, heapMb: init.heapMb, variant });
    step("WASM module ready", init.ms, `heap ${init.heapMb} MB; JSPI ${init.jspi ? "available" : "absent"}`);
    adapterSteps(a, step);
    const { useF16, flash, pathLine, env } = backendEnv(a, { f16: wantF16, flash: !!adv.flashAttention, extra: { ...cpu ? { TRANSCRIBE_BACKENDS: "cpu" } : {}, ...adv.env ?? {} } });
    if (adv.flashAttention && !flash)
      step("flash attention needs shader-f16: using matmul + softmax attention instead");
    if (a.available && !cpu)
      step(pathLine);
    if (a.available && !cpu)
      step(a.plan?.limits === "default" ? `WebGPU limits: spec defaults, nothing raised${limits === "default" ? device.ios && !adv.gpuLimits && !adv.env?.GGML_WEBGPU_LIMITS ? " (the default on iOS)" : " (requested)" : " (the raised request was refused)"}` : `WebGPU limits: the two byte limits raised to ${JSON.stringify(a.requested?.requiredLimits)}`);
    progress({ phase: "model", fraction: 0.05 });
    let store = adv.storage ?? "opfs";
    if (store.startsWith("opfs")) {
      try {
        await navigator.storage.getDirectory();
      } catch (e) {
        store = "blob";
        step(`no origin-private file system (${e?.name ?? e}): the model file is fetched again on every visit`);
      }
    }
    const ld = await rpc.call("load", { url: modelUrl, name: modelFile, store, env, threads: adv.threads ?? 1, verbose });
    step(`model file ready (${ld.from})`, ld.fetchMs, `${ld.mb.toFixed(0)} MB from ${modelUrl}; storage path ${ld.storagePath}`);
    step(`model loaded on ${ld.backend}`, ld.loadMs, `WASM heap ${ld.heapMb} MB, ${ld.heapUsedMb} MB in use`);
    gpu = ld.gpu ?? null;
    if (ld.gpu)
      step(`after load: ${formatGpuStats(ld.gpu)}`, undefined, `JS object counting ${init.gpuwatch ? init.finalizationRegistry ? "on" : "on, no FinalizationRegistry" : "off"}; collector nudge ${init.gcMb ? `${init.gcMb} MiB per pass` : "off"}`);
    if (ld.log && verbose)
      step(`library log: ${ld.log.trim().slice(0, 600)}`);
    const webgpu = /webgpu/i.test(ld.backend);
    const cpuReason = webgpu ? null : cpu ? "the CPU backend was requested" : a.available ? `the backend reported "${ld.backend}"` : String(a.reason);
    info = {
      backend: ld.backend,
      webgpu,
      shaderF16: webgpu && useF16,
      variant,
      jspi: !!init.jspi,
      adapter: a,
      adapterName: a.available ? `${a.vendor} ${a.architecture} ${a.description || a.device || ""}`.trim() : null,
      gpuLimits: a.plan ? a.plan.limits === "default" ? "default" : "raised" : null,
      cpuReason,
      modelUrl,
      modelFile,
      modelBytes: Math.round(ld.mb * 2 ** 20),
      modelFrom: ld.from,
      storagePath: ld.storagePath,
      fetchMs: ld.fetchMs,
      loadMs: ld.loadMs,
      heapMb: ld.heapMb,
      heapUsedMb: ld.heapUsedMb,
      gpuAtLoad: ld.gpu ?? null
    };
    if (!webgpu)
      step(`WARNING: not on WebGPU (backend "${ld.backend}"): ${cpuReason}`);
    diag("loaded", { info, gpuWatch: init.gpuwatch, gcMb: init.gcMb, cpuRequested: cpu });
    let bad = gpuFailure("load", ld.gpuErrors);
    if (bad)
      throw bad;
    if (adv.warmup !== false) {
      progress({ phase: "warmup", fraction: 0.92 });
      step("warm-up pass starting (1.5 s of noise: shaders compile here)");
      const n1 = noise(SAMPLE_RATE * 1.5);
      const w = await rpc.call("run", { pcm: n1.pcm }, [n1.pcm.buffer]);
      step("warm-up pass (1.5 s of noise)", w.wallMs, `WASM heap ${w.heapMb} MB, ${w.heapUsedMb} MB in use`);
      gpu = w.gpu ?? null;
      diag("warmup-pass", { kind: "warm-up", seconds: 1.5, ms: w.wallMs, heapMb: w.heapMb, gpu });
      if (bad = gpuFailure("warm-up", w.gpuErrors))
        throw bad;
      if (webgpu && options.maxSeconds) {
        const longS = options.maxSeconds + 1;
        step(`sizing pass starting (${longS} s of noise: the working buffer is made for the longest utterance)`);
        const n2 = noise(Math.round(SAMPLE_RATE * longS), n1.seed);
        const z = await rpc.call("run", { pcm: n2.pcm }, [n2.pcm.buffer]);
        gpu = z.gpu ?? null;
        step(`sizing pass (${longS} s of noise)`, z.wallMs, z.gpu ? formatGpuStats(z.gpu) : `WASM heap ${z.heapMb} MB`);
        diag("warmup-pass", { kind: "sizing", seconds: longS, ms: z.wallMs, heapMb: z.heapMb, gpu });
        if (bad = gpuFailure("sizing pass", z.gpuErrors))
          throw bad;
      }
    }
  } catch (e) {
    signal?.removeEventListener("abort", onAbort);
    const aborted = !!signal?.aborted;
    state = "disposed";
    silent = true;
    rpc.terminate();
    if (aborted)
      throw abortError();
    throw isTranscriberError(e) ? e : fail(e, "load", "load");
  }
  signal?.removeEventListener("abort", onAbort);
  if (signal?.aborted) {
    state = "disposed";
    silent = true;
    rpc.terminate();
    throw abortError();
  }
  state = "ready";
  progress({ phase: "ready", fraction: 1 });
  let chain = Promise.resolve(), queued = 0;
  const failed = (e) => {
    if (state === "ready") {
      state = "failed";
      options.onError?.(e);
    }
    return e;
  };
  async function pass(pcm, transfer) {
    if (state !== "ready")
      throw fail(new Error(state === "disposed" ? "the transcriber was disposed" : "the transcriber has stopped after an error"), state === "disposed" ? "disposed" : "run", "pass");
    let r;
    try {
      r = await rpc.call("run", { pcm }, transfer ? [pcm.buffer] : []);
    } catch (e) {
      if (state === "disposed")
        throw fail(e, "disposed", "pass");
      throw failed(fail(e, "run", "pass"));
    }
    gpu = r.gpu ?? null;
    const bad = gpuFailure("pass", r.gpuErrors);
    if (bad)
      throw failed(bad);
    return { text: r.text, words: r.words ?? [], ms: r.wallMs, melMs: r.mel_ms, encodeMs: r.encode_ms, decodeMs: r.decode_ms, tokens: r.n_tokens, heapMb: r.heapMb, heapUsedMb: r.heapUsedMb, gpu, log: r.log ?? "" };
  }
  return {
    info,
    get state() {
      return state;
    },
    transcribe(pcm, o) {
      queued++;
      const p = chain.then(() => pass(pcm, !!o?.transfer)).finally(() => {
        queued--;
      });
      chain = p.catch(() => {});
      return p;
    },
    gpuStats: () => gpu,
    async releaseWorkingMemory() {
      if (state !== "ready")
        return null;
      const p = chain.then(() => rpc.call("trim"));
      chain = p.catch(() => {});
      return gpu = await p ?? null;
    },
    async dispose() {
      if (state === "disposed")
        return null;
      const clean = state === "ready" && queued === 0;
      state = "disposed";
      let after = null;
      if (clean)
        try {
          after = await Promise.race([rpc.call("trim").then(() => rpc.call("free")).then(() => rpc.call("gpu")), new Promise((res) => setTimeout(() => res(null), 3000))]);
        } catch {}
      silent = true;
      rpc.terminate();
      return after;
    }
  };
}

// lib/src/dictation.ts
function createDictation(options) {
  const adv = options.advanced ?? {};
  const ev = adv.streamEvents ?? {};
  const capS = options.maxUtteranceSeconds ?? adv.stream?.capS ?? (detectDevice().phone ? 12 : 24);
  let modelUrl = options.modelUrl;
  let model = "loading";
  let transcriber = null;
  let loadGen = 0, abort = null;
  let recording = false, starting = null, transcribing = true, captured = 0, disposed = false;
  let micStatus = "off";
  let passes = 0, sinceLoad = 0, longest = 0;
  const waiters = [];
  const wake = () => {
    for (const w of waiters.splice(0))
      w();
  };
  const state = () => ({ model, recording, microphone: micStatus });
  let told = "";
  const changed = () => {
    const s = state(), key = `${s.model} ${s.recording} ${s.microphone}`;
    if (disposed || key === told)
      return;
    told = key;
    options.onState?.(s);
  };
  const step = (name, ms, detail) => {
    if (!disposed)
      adv.onStep?.({ name, ms, detail });
  };
  const report = (e) => {
    if (!disposed)
      options.onError?.(e);
  };
  const transcribe = async (pcm, req) => {
    while (model === "loading")
      await new Promise((res) => waiters.push(res));
    const t = transcriber;
    if (model !== "ready" || !t)
      throw new Error("the model is not running");
    const gen = loadGen, n = ++passes, audioSeconds = pcm.length / SAMPLE_RATE;
    adv.onPassStart?.({ n, sinceLoad, kind: req.kind, audioSeconds, longestBefore: longest / SAMPLE_RATE });
    if (pcm.length > longest)
      longest = pcm.length;
    const t0 = performance.now();
    let result;
    try {
      result = await t.transcribe(pcm, { transfer: true });
    } catch (e) {
      if (gen !== loadGen && !disposed)
        queueMicrotask(() => {
          if (stream.stats().failed)
            stream.reset();
        });
      throw e;
    }
    if (gen !== loadGen)
      throw new Error("the model was reloaded");
    adv.onPass?.({ n, sinceLoad: ++sinceLoad, kind: req.kind, audioSeconds, ms: performance.now() - t0, endSample: req.endSample, result });
    return result;
  };
  const stream = createStream(transcribe, {
    onCommit(c) {
      ev.onCommit?.(c);
      options.onCommit?.(c.text, c);
    },
    onTentative(text, words, at) {
      ev.onTentative?.(text, words, at);
      options.onTentative?.(text);
    },
    onUtteranceEnd(u) {
      ev.onUtteranceEnd?.(u);
    },
    onPass(p) {
      ev.onPass?.(p);
    },
    onGate(g) {
      ev.onGate?.(g);
    },
    onSkip(s) {
      ev.onSkip?.(s);
    },
    onError(e) {
      ev.onError?.(e);
    }
  }, { ...adv.stream, capS });
  const microphone = createMicrophone({
    assetsUrl: options.assetsUrl,
    voiceProcessing: options.voiceProcessing,
    wakeLock: options.wakeLock,
    onAudio(pcm, peak) {
      if (!recording)
        return;
      captured += pcm.length;
      let speech = false;
      if (transcribing) {
        stream.push(pcm);
        speech = stream.stats().speech;
      }
      options.onLevel?.({ peak, speech, capturedSeconds: captured / SAMPLE_RATE });
    },
    onStatus(s) {
      micStatus = s;
      changed();
    },
    onEnded(e) {
      stop();
      report(e);
    },
    onDiagnostic: (kind, data) => {
      if (!disposed)
        adv.onDiagnostic?.(kind, data);
    }
  });
  let ready;
  function load(url = modelUrl) {
    if (disposed)
      return Promise.reject(new Error("the dictation was disposed"));
    const gen = ++loadGen;
    abort?.abort();
    const old = transcriber;
    transcriber = null;
    modelUrl = url;
    model = "loading";
    sinceLoad = 0;
    longest = 0;
    old?.dispose();
    if (!recording || stream.stats().failed)
      stream.reset();
    changed();
    const mine = abort = new AbortController;
    const p = createTranscriber({
      assetsUrl: options.assetsUrl,
      modelUrl: url,
      backend: options.backend,
      signal: mine.signal,
      maxSeconds: adv.sizingPass === false ? undefined : stream.params.capS,
      onProgress: (pr) => {
        if (gen === loadGen && !disposed)
          options.onLoadProgress?.(pr);
      },
      onError(e) {
        if (gen !== loadGen || disposed)
          return;
        model = "failed";
        changed();
        report(e);
      },
      advanced: {
        ...adv.transcriber,
        onStep: (s) => {
          if (gen === loadGen && !disposed) {
            adv.transcriber?.onStep?.(s);
            adv.onStep?.(s);
          }
        },
        onDiagnostic: (kind, data) => {
          if (gen === loadGen && !disposed) {
            adv.transcriber?.onDiagnostic?.(kind, data);
            adv.onDiagnostic?.(kind, data);
          }
        }
      }
    }).then((t) => {
      if (gen !== loadGen || disposed) {
        t.dispose();
        throw Object.assign(new Error("the load was replaced by a newer one"), { name: "AbortError" });
      }
      transcriber = t;
      model = "ready";
      changed();
      wake();
      return t.info;
    }, (e) => {
      if (gen === loadGen && !disposed) {
        model = "failed";
        changed();
        wake();
        report(e);
      }
      throw e;
    });
    p.catch(() => {});
    return ready = p;
  }
  function start(overrides = {}) {
    if (disposed)
      return Promise.resolve(false);
    if (recording)
      return Promise.resolve(true);
    if (starting)
      return starting;
    const mic = microphone.start({ voiceProcessing: overrides?.voiceProcessing });
    const p = mic.then(async (info) => {
      if (!stream.idle() && model === "ready")
        await stream.stop();
      if (starting !== p || disposed)
        return false;
      stream.reset();
      captured = 0;
      transcribing = overrides?.transcribe !== false;
      recording = true;
      starting = null;
      adv.onDiagnostic?.("recording", { ...info, transcribing });
      const set = info.track;
      step(`recording: context ${info.contextState} at ${info.sampleRate} Hz -> 16000 Hz, track ${set.sampleRate ?? "?"} Hz, ${set.channelCount ?? "?"} ch, echoCancellation=${set.echoCancellation} noiseSuppression=${set.noiseSuppression} autoGainControl=${set.autoGainControl}`);
      changed();
      return true;
    }, (e) => {
      if (starting === p)
        starting = null;
      step(`microphone: ${e?.cause?.name ?? e?.name ?? "Error"}: ${e?.cause?.message ?? e?.message ?? e}`, undefined, e?.contextState ? `AudioContext ${e.contextState}` : undefined);
      changed();
      if (!(isMicrophoneError(e) && e.code === "stopped"))
        report(e);
      return false;
    });
    return starting = p;
  }
  let draining = null;
  function stop() {
    if (starting) {
      starting = null;
      microphone.stop();
      return Promise.resolve();
    }
    if (!recording)
      return draining ?? Promise.resolve();
    recording = false;
    microphone.stop();
    changed();
    const d = stream.stop().catch(() => {}).then(() => {
      if (draining === d)
        draining = null;
    });
    return draining = d;
  }
  async function dispose() {
    if (disposed)
      return;
    recording = false;
    starting = null;
    microphone.stop();
    disposed = true;
    loadGen++;
    abort?.abort();
    stream.reset();
    model = "failed";
    wake();
    const t = transcriber;
    transcriber = null;
    await t?.dispose();
  }
  ready = Promise.resolve().then(() => loadGen === 0 && !disposed ? load() : ready);
  ready.catch(() => {});
  return {
    start,
    stop,
    load,
    dispose,
    get ready() {
      return ready;
    },
    get state() {
      return state();
    },
    stats: () => stream.stats(),
    idle: () => !recording && !starting && stream.idle(),
    params: stream.params,
    get transcriber() {
      return transcriber;
    }
  };
}
export {
  DEFAULT_PARAMS,
  SAMPLE_RATE,
  createDictation,
  createMicrophone,
  createStream,
  createTranscriber,
  detectDevice,
  formatGpuStats,
  isMicrophoneError,
  isTranscriberError,
  normalizeWord
};
