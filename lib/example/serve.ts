// Any static file server will do; this one is ten lines of Bun.
import { join, normalize } from "node:path";
const root = join(import.meta.dir, "public");
Bun.serve({ port: 8795, hostname: "127.0.0.1", idleTimeout: 0, fetch(req) {
  let p = normalize(decodeURIComponent(new URL(req.url).pathname));
  if (p.endsWith("/")) p += "index.html";
  const file = Bun.file(join(root, p));
  return file.exists().then((ok) => (ok ? new Response(file) : new Response("not found", { status: 404 })));
} });
console.log("http://127.0.0.1:8795/");
