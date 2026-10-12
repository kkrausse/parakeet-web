// Opt-in only (the "Turn WASM threads on" button). The static host sends no COOP/COEP headers, and
// without them there is no SharedArrayBuffer, so onnxruntime-web's WASM runs on one thread. This adds
// the two headers to same-origin responses. Model files are left alone: they need no header and are
// too big to pass through a worker.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (u.origin !== location.origin || u.pathname.includes("/models/") || e.request.method !== "GET") return;
  e.respondWith(fetch(e.request).then((r) => {
    if (r.status === 0) return r;
    const h = new Headers(r.headers);
    h.set("Cross-Origin-Opener-Policy", "same-origin");
    h.set("Cross-Origin-Embedder-Policy", "require-corp");
    h.set("Cross-Origin-Resource-Policy", "same-origin");
    return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
  }));
});
