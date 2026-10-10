// Remote diagnostics for both pages: every step of the trail is POSTed, as it happens, to a small
// collector on the user's own machine (scripts/diag-collector.ts on diesel2, tailnet only), so a tab the
// browser kills still leaves its last step somewhere. Never audio, never transcript text. Sending fails
// silently. Also here: telling the page's own errors from ones thrown by scripts it did not load
// (wallet / extension / in-app-browser injections), and the per-page "how did the last visit end" record.
declare const DIAG_URL: string; // build.ts (PK_DIAG_URL); "" = no collector

export interface Diag {
  sid: string; // per page load
  url: string | null;
  /** Queue one event. `urgent` (default) sends now; otherwise within a second (pass records). */
  send(kind: string, data?: Record<string, unknown>, urgent?: boolean): void;
  flush(beacon?: boolean): void;
  note(): string;
  foreign: string[];
}

function newSid() {
  const abc = "abcdefghjkmnpqrstuvwxyz23456789"; // no look-alikes: it is read out or typed
  const r = new Uint8Array(6);
  try { crypto.getRandomValues(r); } catch { for (let i = 0; i < r.length; i++) r[i] = Math.random() * 256; }
  return [...r].map((x) => abc[x % abc.length]).join("");
}

export function createDiag(page: string, params: URLSearchParams): Diag {
  const q = params.get("diag");
  const url = q === "0" ? null : q && /^https?:/.test(q) ? q : (typeof DIAG_URL === "string" && DIAG_URL) || null;
  const sid = newSid();
  let seq = 0, queue: Record<string, unknown>[] = [], timer: ReturnType<typeof setTimeout> | null = null;
  let fails = 0, quietUntil = 0, sent = 0;
  const d: Diag = {
    sid, url, foreign: [],
    send(kind, data = {}, urgent = true) {
      if (!url) return;
      queue.push({ seq: seq++, t: Math.round(performance.now()), kind, ...data });
      if (queue.length > 300) queue.splice(0, queue.length - 300); // collector unreachable: keep the newest
      if (urgent) d.flush();
      else if (!timer) timer = setTimeout(() => d.flush(), 1000);
    },
    flush(beacon = false) {
      if (timer) clearTimeout(timer), (timer = null);
      if (!url || !queue.length) return;
      if (fails >= 3 && Date.now() < quietUntil && !beacon) return; // unreachable: one attempt every 30 s
      const body = JSON.stringify({ sid, page, wall: Date.now(), events: queue });
      queue = [];
      try {
        if (beacon && navigator.sendBeacon?.(url, body)) return;
        // text/plain + no-cors: a "simple" request, so no preflight; keepalive lets it outlive the page (64 KiB cap)
        void fetch(url, { method: "POST", body, keepalive: body.length < 60000, mode: "no-cors", credentials: "omit", cache: "no-store" })
          .then(() => { fails = 0; sent++; }, () => { if (++fails >= 3) quietUntil = Date.now() + 30000; });
      } catch { /* never the page's problem */ }
    },
    note() {
      if (!url) return "Diagnostics are not sent anywhere (diag=0, or no collector in this build).";
      let host = url;
      try { host = new URL(url).host; } catch { /* as given */ }
      return `Diagnostics: this page's steps (no audio, no text) go to your own machine (${host}). Session ${sid}.`
        + (d.foreign.length ? ` ${d.foreign.length} error${d.foreign.length > 1 ? "s" : ""} from scripts that are not this page's ignored.` : "");
    },
  };
  return d;
}

// ---------- whose error is it ----------
const OWN_BASE = new URL(".", location.href).href; // the page's directory: every script this page loads lives here
const ownScript = (u: string) => u.startsWith(OWN_BASE) && /\.js(?:[?#:]|$)/.test(u.slice(OWN_BASE.length));
function stackIsOwn(stack: unknown) {
  if (typeof stack !== "string" || !stack) return null; // no stack: cannot tell
  const urls = stack.match(/(?:https?|file):\/\/[^\s)'"]+/g) ?? [];
  return urls.some(ownScript);
}
// Globals and schemes that only injected code talks about (wallets, in-app browsers, extensions).
const INJECTED = /\bethereum\b|\b(?:solana|phantom|tronWeb|tronLink|web3|keplr|coinbaseWallet|trustwallet|__firefox__|__gCrWeb|ReactNativeWebView|webkit\.messageHandlers|zaloJSV2|instantSearchSDKJSBridge)\b|(?:safari-web-extension|safari-extension|chrome-extension|moz-extension|webkit-masked-url|user-script):/i;

export interface Verdict { own: boolean; why: string; text: string }
/** An `error` or `unhandledrejection` event: is it from one of this page's scripts? */
export function classifyError(e: ErrorEvent | PromiseRejectionEvent): Verdict {
  if ("reason" in e) {
    const r: any = e.reason;
    const text = `${r?.name ? r.name + ": " : ""}${r?.message ?? r}`.slice(0, 600);
    const own = stackIsOwn(r?.stack);
    if (own) return { own: true, why: "stack is in this page's scripts", text };
    if (INJECTED.test(text) || INJECTED.test(String(r?.stack ?? ""))) return { own: false, why: "mentions an injected global or an extension", text };
    if (own === false) return { own: false, why: "no frame of the stack is in this page's scripts", text };
    return { own: false, why: "rejection without a stack: origin unknown", text };
  }
  const text = `${e.message}${e.filename ? ` (${e.filename}:${e.lineno}:${e.colno})` : ""}`.slice(0, 600);
  if (stackIsOwn((e.error as any)?.stack)) return { own: true, why: "stack is in this page's scripts", text };
  if (/^script error\.?$/i.test(String(e.message).trim()) && !e.filename) return { own: false, why: "cross-origin script (the browser withholds the details)", text };
  if (!e.filename) return { own: false, why: "no source file: an injected or evaluated script", text };
  if (!ownScript(e.filename)) return { own: false, why: e.filename.split(/[?#]/)[0] === location.href.split(/[?#]/)[0] ? "an inline script; this page has none" : "source file is not one of this page's scripts", text };
  return { own: true, why: "source file is one of this page's scripts", text };
}

/** Installs the two global handlers. Foreign errors go to the collector and a counter, never to `onOwn`. */
export function watchErrors(diag: Diag, onOwn: (label: string, text: string) => void, onForeign: () => void) {
  const handle = (label: string, e: ErrorEvent | PromiseRejectionEvent) => {
    const v = classifyError(e);
    if (v.own) return onOwn(label, v.text);
    if (diag.foreign.length < 50) {
      diag.foreign.push(`${label}: ${v.text} [${v.why}]`);
      diag.send("foreign-error", { label, text: v.text, why: v.why, note: "foreign, ignored" });
      onForeign();
    }
  };
  addEventListener("error", (e) => handle("uncaught error", e as ErrorEvent));
  addEventListener("unhandledrejection", (e) => handle("unhandled rejection", e as PromiseRejectionEvent));
}

// ---------- environment and page lifecycle ----------
export interface Device { ios: boolean; phone: boolean; why: string }
/** iOS / iPadOS (every browser there is WebKit) and phones in general. `phone=1|0` forces both, for testing. */
export function detectDevice(params: URLSearchParams): Device {
  const ua = navigator.userAgent, forced = params.get("phone");
  if (forced === "1" || forced === "0") return { ios: forced === "1", phone: forced === "1", why: `phone=${forced}` };
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const phone = ios || /Android/.test(ua);
  return { ios, phone, why: ios ? "iOS/iPadOS user agent" : phone ? "Android user agent" : "desktop user agent" };
}

export function sendEnvironment(diag: Diag, extra: Record<string, unknown>) {
  const n: any = navigator, pm: any = (performance as any).memory;
  diag.send("env", {
    ua: n.userAgent, platform: n.platform ?? null, deviceMemory: n.deviceMemory ?? null, hardwareConcurrency: n.hardwareConcurrency ?? null,
    maxTouchPoints: n.maxTouchPoints ?? null, gpu: "gpu" in navigator, jspi: "Suspending" in WebAssembly, secureContext: isSecureContext,
    crossOriginIsolated, screen: `${screen.width}x${screen.height}@${devicePixelRatio}`, standalone: n.standalone ?? null,
    opfsApi: !!n.storage?.getDirectory, jsHeapLimitMb: pm ? Math.round(pm.jsHeapSizeLimit / 2 ** 20) : null, online: n.onLine,
    connection: n.connection?.effectiveType ?? null, visibility: document.visibilityState, url: location.href, ...extra,
  });
  try {
    void n.storage?.estimate?.().then((s: any) => diag.send("storage-estimate", { quotaMb: Math.round((s.quota ?? 0) / 2 ** 20), usageMb: Math.round((s.usage ?? 0) / 2 ** 20) }), () => {});
    void n.storage?.persisted?.().then((p: boolean) => diag.send("storage-persisted", { persisted: p }), () => {});
  } catch { /* not there */ }
}

/** How the last visit of this page ended, kept in localStorage: `closed` is only set by pagehide, which a killed tab never fires. */
export interface Life { sid: string; phase: string; hidden: boolean; closed: boolean; wall: number; model?: string; cpu?: boolean; at?: string }
export function lifeStore(key: string, diag: Diag) {
  let prev: Life | null = null;
  try { prev = JSON.parse(localStorage.getItem(key) ?? "null"); } catch { /* none */ }
  const cur: Life = { sid: diag.sid, phase: "open", hidden: document.visibilityState !== "visible", closed: false, wall: Date.now() };
  const save = () => { cur.wall = Date.now(); try { localStorage.setItem(key, JSON.stringify(cur)); } catch { /* private mode */ } };
  save();
  const ev = (name: string, data: Record<string, unknown> = {}) => diag.send("lifecycle", { name, visibility: document.visibilityState, ...data });
  document.addEventListener("visibilitychange", () => { cur.hidden = document.visibilityState !== "visible"; save(); ev("visibilitychange"); if (cur.hidden) diag.flush(true); });
  addEventListener("pagehide", (e) => { cur.closed = true; save(); ev("pagehide", { persisted: (e as PageTransitionEvent).persisted }); diag.flush(true); });
  addEventListener("pageshow", (e) => { cur.closed = false; save(); ev("pageshow", { persisted: (e as PageTransitionEvent).persisted }); });
  document.addEventListener("freeze", () => { ev("freeze"); diag.flush(true); });
  document.addEventListener("resume", () => ev("resume"));
  addEventListener("online", () => ev("online"));
  addEventListener("offline", () => ev("offline"));
  return {
    prev,
    phase(p: string, more: Partial<Life> = {}) { cur.phase = p; Object.assign(cur, more); save(); },
    /** Before the page navigates itself (retry links): the next load must not read this visit as killed. */
    closing() { cur.closed = true; save(); },
  };
}
