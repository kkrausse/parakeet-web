#!/bin/bash
# Rebuild the WASM module(s) and the site.   usage: wb.sh
R=~/devfs/cache/parakeet-ggml-webgpu; cd $R; source emsdk/emsdk_env.sh >/dev/null 2>&1
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web/web
for d in build-web build-web-asyncify; do
  [ -d $d ] || continue
  cmake -S transcribe.cpp -B $d > /dev/null 2>&1
  nice make -C $d -j8 pk-web > out/$d.log 2>&1 || { grep -E "error" -A4 out/$d.log | head -30; exit 1; }
done
cd $W && bun build.ts | tail -1
