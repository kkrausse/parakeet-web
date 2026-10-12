#!/usr/bin/env node
// Copies the runtime files (worker, worklet, the two WASM builds) into a folder the app serves.
// usage: parakeet-web-assets <destination folder>     e.g. bunx parakeet-web-assets public/parakeet-web
import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const dest = process.argv[2];
if (!dest) { console.error("usage: parakeet-web-assets <destination folder>"); process.exit(1); }
const from = join(dirname(fileURLToPath(import.meta.url)), "..", "dist", "assets");
mkdirSync(dest, { recursive: true });
cpSync(from, dest, { recursive: true });
console.log(`parakeet-web: ${readdirSync(from).join(", ")} -> ${resolve(dest)}`);
