// Diagnostics collector for the Parakeet pages (web/src/diag.ts) and the reader for what it wrote.
// The pages POST {sid, page, wall, events:[{seq, t, kind, ...}]} as text/plain; each event becomes one
// JSON line in $DIR/diag-YYYY-MM-DD.jsonl with the receive time. Loopback only: the tailnet reaches it
// through `tailscale serve` (https), nothing else does. No audio or transcript text is ever sent.
//
//   bun diag-collector.ts serve [port]          listen on 127.0.0.1:<port> (default 4795)
//   bun diag-collector.ts show [N] [--full]     the latest N sessions (default 3), readable; --full prints every event's fields
//   bun diag-collector.ts show <sid> [--full]   one session by id (a prefix is enough)
//   bun diag-collector.ts tail                  follow new events as they arrive
// env: PK_DIAG_DIR (default ~/devfs/cache/parakeet-ggml-webgpu/diag)
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, watch } from "node:fs";
import { join } from "node:path";

const DIR = process.env.PK_DIAG_DIR ?? join(process.env.HOME!, "devfs/cache/parakeet-ggml-webgpu/diag");
const [mode = "show", ...rest] = Bun.argv.slice(2);
const MAX_BODY = 512 * 1024;
// The shelf, this box's own tailnet names, and local test servers. Anything else gets no CORS header (the pages send no-cors anyway).
const allowed = (origin: string) => /^https:\/\/[a-z0-9-]+\.guineafowl-truck\.ts\.net(:\d+)?$/.test(origin) || /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin);

interface Line { rt: string; client?: string; sid: string; page: string; seq: number; t: number; kind: string; [k: string]: unknown }

function serve(port: number) {
  mkdirSync(DIR, { recursive: true });
  const cors = (req: Request): Record<string, string> => {
    const o = req.headers.get("origin") ?? "";
    return allowed(o) ? { "Access-Control-Allow-Origin": o, "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "content-type", "Access-Control-Max-Age": "86400", Vary: "Origin" } : {};
  };
  Bun.serve({
    port, hostname: "127.0.0.1",
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(req) });
      if (req.method === "GET" && path === "/health") return new Response("ok\n", { headers: cors(req) });
      if (req.method !== "POST" || path !== "/log") return new Response("not found\n", { status: 404, headers: cors(req) });
      try {
        const text = await req.text();
        if (text.length > MAX_BODY) return new Response("too large\n", { status: 413, headers: cors(req) });
        const body = JSON.parse(text);
        const sid = String(body.sid ?? "").replace(/[^a-z0-9]/gi, "").slice(0, 16), page = String(body.page ?? "").slice(0, 16);
        if (!sid || !Array.isArray(body.events)) return new Response("bad request\n", { status: 400, headers: cors(req) });
        const rt = new Date().toISOString();
        const client = req.headers.get("tailscale-user-login") ?? req.headers.get("x-forwarded-for") ?? undefined; // set by tailscale serve
        let out = "";
        for (const e of body.events.slice(0, 400)) if (e && typeof e === "object") out += JSON.stringify({ rt, sid, page, client, ...e }) + "\n";
        appendFileSync(join(DIR, `diag-${rt.slice(0, 10)}.jsonl`), out);
        return new Response(null, { status: 204, headers: cors(req) });
      } catch (e) { return new Response(`bad request: ${e}\n`, { status: 400, headers: cors(req) }); }
    },
  });
  console.log(`diag collector on http://127.0.0.1:${port}/log -> ${DIR}`);
}

function readAll(days = 4): Line[] {
  if (!existsSync(DIR)) return [];
  const files = readdirSync(DIR).filter((f) => /^diag-.*\.jsonl$/.test(f)).sort().slice(-days);
  const lines: Line[] = [];
  for (const f of files) for (const l of readFileSync(join(DIR, f), "utf8").split("\n")) if (l) try { lines.push(JSON.parse(l)); } catch { /* a torn line */ }
  return lines;
}
const short = (v: unknown, n = 300) => { const s = typeof v === "string" ? v : JSON.stringify(v); return s === undefined ? "" : s.length > n ? s.slice(0, n) + "..." : s; };
const mib = (x: unknown) => (typeof x === "number" ? `${Math.round(x / 2 ** 20)} MiB` : "?");
/** One event as one readable line. */
function eventLine(e: Line, full: boolean): string {
  const head = `${(e.t / 1000).toFixed(1).padStart(7)}s #${String(e.seq).padEnd(4)}`;
  const { rt, sid, page, seq, t, kind, client, ...data } = e;
  if (full) return `${head} ${kind} ${JSON.stringify(data)}`;
  const d = data as any;
  switch (kind) {
    case "step": return `${head} ${d.name}${d.ms !== undefined ? `: ${d.ms} ms` : ""}${d.detail ? `  (${short(d.detail, 400)})` : ""}`;
    case "env": return `${head} ENV ${d.ua}\n${" ".repeat(15)}deviceMemory=${d.deviceMemory} cores=${d.hardwareConcurrency} touch=${d.maxTouchPoints} navigator.gpu=${d.gpu} JSPI=${d.jspi} OPFS api=${d.opfsApi} screen=${d.screen} phone=${d.device?.phone} ios=${d.device?.ios} cap=${d.capS ?? "-"} limits=${d.limits}\n${" ".repeat(15)}${d.url}`;
    case "adapter": {
      const a = d.adapter ?? {};
      if (!a.available) return `${head} ADAPTER none: ${a.reason}`;
      return `${head} ADAPTER ${a.vendor} ${a.architecture} ${a.description || a.device || ""} fallback=${a.isFallbackAdapter} | features: ${(a.features ?? []).join(" ") || "none"}\n${" ".repeat(15)}limits: binding ${mib(a.limits?.maxStorageBufferBindingSize)}, buffer ${mib(a.limits?.maxBufferSize)}, invocations ${a.limits?.maxComputeInvocationsPerWorkgroup}, storage buffers/stage ${a.limits?.maxStorageBuffersPerShaderStage}${a.belowSpecDefault?.length ? ` | BELOW SPEC: ${a.belowSpecDefault.join("; ")}` : ""}\n${" ".repeat(15)}device ladder: ${a.plan ? `${a.deviceProbe} with limits=${a.plan.limits || "raised"} f16=${a.plan.f16}` : `NO DEVICE: ${a.deviceError}`}${a.probeErrors?.length ? ` | refused: ${a.probeErrors.join(" | ")}` : ""} | WASM heap ${d.heapMb} MB`;
    }
    case "chosen": return `${head} CHOSEN ${d.model} (${d.file}) build=${d.build} backend=${d.backend} shaders=${d.shaderPath} limits=${d.limits === "default" ? "spec default" : "raised"} storage=${d.storagePath} (${d.from}) ${d.fileBytes} bytes | fetch ${d.fetchMs} ms, load ${d.loadMs} ms | WASM heap ${d.heapMb} MB (${d.heapUsedMb} in use)`;
    case "pass": return `${head} pass ${d.clip ? d.clip + " " : ""}${d.n}${d.final ? " final" : ""}: ${d.ms} ms for ${d.audioS} s (mel ${d.melMs}, decode ${d.decodeMs}) heap ${d.heapMb} MB`;
    case "pass-begin": return `${head} pass ${d.n} starting: ${d.audioS} s${d.longest ? " (longest so far)" : ""}${d.final ? " final" : ""}`;
    case "heartbeat": return `${head} heartbeat ${d.phase}${d.recording ? " recording" : ""} heap ${d.heapMb} MB js ${d.jsHeapMb ?? "?"} MB passes ${d.passes} lag ${d.lagS} s buffer ${d.bufferS} s backlog ${d.backlogS} s ctx ${d.audioContext} ${d.visibility}`;
    case "load-heap": return `${head} loading: WASM heap ${d.heapMb} MB at ${d.ms} ms`;
    case "lifecycle": return `${head} LIFECYCLE ${d.name}${d.persisted !== undefined ? ` persisted=${d.persisted}` : ""} (${d.visibility})`;
    case "foreign-error": return `${head} FOREIGN, IGNORED ${d.label}: ${short(d.text)} [${d.why}]`;
    case "previous-session": return `${head} PREVIOUS ${d.line}${d.error ? ` | its error: ${short(d.error, 200)}` : ""}`;
    case "guard": return `${head} GUARD: not loading automatically (previous session ${d.prev?.sid} died in phase ${d.prev?.phase})`;
    default: return `${head} ${kind} ${short(data, 500)}`;
  }
}
function show(args: string[]) {
  const full = args.includes("--full"), all = args.includes("--passes");
  const arg = args.find((a) => !a.startsWith("--"));
  const lines = readAll(arg && !/^\d+$/.test(arg) ? 30 : 4);
  const sessions = new Map<string, Line[]>();
  for (const l of lines) (sessions.get(l.sid) ?? sessions.set(l.sid, []).get(l.sid)!).push(l);
  let ids = [...sessions.keys()]; // in order of first arrival
  if (arg && !/^\d+$/.test(arg)) ids = ids.filter((s) => s.startsWith(arg));
  else ids = ids.slice(-Number(arg ?? 3));
  if (!ids.length) return console.log(`no sessions in ${DIR}`);
  for (const id of ids) {
    const ev = sessions.get(id)!.sort((a, b) => a.seq - b.seq);
    const last = ev.at(-1)!, lastStep = ev.findLast((e) => e.kind === "step") as any;
    const closed = ev.some((e) => e.kind === "lifecycle" && (e as any).name === "pagehide");
    console.log(`\n===== session ${id} · ${ev[0].page} · first ${ev[0].rt} · last ${last.rt} · ${ev.length} events${ev[0].client ? ` · ${ev[0].client}` : ""}`);
    console.log(`      last step: ${lastStep?.name ?? "none"} · ${closed ? "pagehide seen (closed or navigated)" : "no pagehide (still open, or the tab was killed)"}`);
    let prevSeq = -1, passes = 0;
    for (const e of ev) {
      if (e.seq !== prevSeq + 1 && prevSeq >= 0) console.log(`${" ".repeat(8)}(${e.seq - prevSeq - 1} event(s) missing)`);
      prevSeq = e.seq;
      // passes are many: the first five, every 20th and the slow ones, unless --passes
      if (e.kind === "pass" && !all && !full) { passes++; const d = e as any; if (!(d.n <= 5 || d.n % 20 === 0 || d.ms > 1000)) continue; }
      if (e.kind === "load-heap" && !all && !full && (e as any).ms % 2000 >= 500) continue;
      console.log(eventLine(e, full));
    }
    if (passes && !all && !full) console.log(`${" ".repeat(8)}(${passes} pass records; --passes shows all)`);
  }
}
function tail() {
  mkdirSync(DIR, { recursive: true });
  const file = () => join(DIR, `diag-${new Date().toISOString().slice(0, 10)}.jsonl`);
  let pos = existsSync(file()) ? statSync(file()).size : 0, cur = file();
  const pump = () => {
    if (file() !== cur) cur = file(), (pos = 0);
    if (!existsSync(cur)) return;
    const buf = readFileSync(cur);
    if (buf.length <= pos) return;
    const chunk = buf.subarray(pos).toString("utf8"), end = chunk.lastIndexOf("\n");
    if (end < 0) return;
    pos += Buffer.byteLength(chunk.slice(0, end + 1));
    for (const l of chunk.slice(0, end).split("\n")) try { const e = JSON.parse(l) as Line; console.log(`${e.sid} ${eventLine(e, false)}`); } catch { /* torn */ }
  };
  watch(DIR, pump);
  setInterval(pump, 2000);
}

if (mode === "serve") serve(Number(rest[0] ?? 4795));
else if (mode === "tail") tail();
else if (mode === "show") show(rest);
else { console.error("usage: bun diag-collector.ts serve [port] | show [N|sid] [--full] [--passes] | tail"); process.exit(2); }
