// Local static server for dist/. Sends COOP/COEP so WASM threads work; --no-coi leaves them off,
// which is what the published copy (tailscale serve, no custom headers) looks like.
// usage: bun serve.ts [port] [--no-coi]
import { join, normalize } from "node:path";
const port = Number(Bun.argv.find((a) => /^\d+$/.test(a)) ?? 8787);
const coi = !Bun.argv.includes("--no-coi");
const root = join(import.meta.dir, "dist");
Bun.serve({
  port, hostname: "127.0.0.1", idleTimeout: 0,
  fetch(req) {
    let p = normalize(decodeURIComponent(new URL(req.url).pathname));
    if (p.endsWith("/")) p += "index.html";
    const file = Bun.file(join(root, p));
    if (!join(root, p).startsWith(root)) return new Response("no", { status: 403 });
    const headers: Record<string, string> = { "Cache-Control": "no-store" };
    if (coi) headers["Cross-Origin-Opener-Policy"] = "same-origin", headers["Cross-Origin-Embedder-Policy"] = "require-corp";
    return file.exists().then((ok) => (ok ? new Response(file, { headers }) : new Response("not found", { status: 404 })));
  },
});
console.log(`http://127.0.0.1:${port}/ serving ${root}${coi ? " with COOP/COEP" : ""}`);
