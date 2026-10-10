#!/bin/bash
# One run of the live page in Chrome with the fake microphone playing a test wav; summary on stdout,
# everything in results/live/NAME.json.   usage: lv.sh NAME CLIP ["query"] [drive-live.ts args ...]
# CLIP is a name in $R/audio/live (t14 t56 g2 g5: fixtures with silences, made by scripts/live-clips.py).
# STOCK=1 drops the Dawn f16 flag (f32-only shaders); URL=... runs against another copy of the page.
R=~/devfs/cache/parakeet-ggml-webgpu
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web
name=$1; clip=$2; q=${3:-}; shift; shift; shift
F16="--chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia"; [ -n "$STOCK" ] && F16=""
cd $W/web && bun drive-live.ts "$name" "$q" --wav $R/audio/live/$clip.wav --f32 $R/audio/live/$clip.f32 --expect $R/audio/live/$clip.txt \
  ${URL:+--url "$URL"} $F16 "$@"
