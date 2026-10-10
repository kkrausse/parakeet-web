// Static server for ./site on the Mac (simulator Safari reaches it as http://localhost:8793/), POST /report appends to report.log
import { join, normalize } from "node:path";
import { appendFileSync } from "node:fs";
const root = join(import.meta.dir, "site");
Bun.serve({ port: 8793, hostname: "127.0.0.1", idleTimeout: 0, async fetch(req) {
  const u = new URL(req.url);
  if (req.method === "POST") { appendFileSync(join(import.meta.dir, "report.log"), (await req.text()) + "\n"); return new Response("ok"); }
  let p = normalize(decodeURIComponent(u.pathname)); if (p.endsWith("/")) p += "index.html";
  const f = Bun.file(join(root, p));
  return (await f.exists()) ? new Response(f, { headers: { "Cache-Control": "no-store" } }) : new Response("nf", { status: 404 });
} });
