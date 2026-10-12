#!/bin/bash
# Rebuild the WASM module(s) from vendor/transcribe.cpp and then the site.   usage: wb.sh
W=$(cd "$(dirname "$0")/.." && pwd)
cd $W && bun scripts/runtime.ts build && cd web && bun build.ts | tail -1
