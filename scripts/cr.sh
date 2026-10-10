#!/bin/bash
# One Chrome run against the local server (tmux pkg-serve, port 8791).  usage: [STOCK=1] cr.sh NAME "query" [drive args]
# STOCK=1: no Dawn f16 toggle, i.e. what stock Chrome on this box (NVIDIA driver 580) gives: no shader-f16.
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web/web; cd $W
n=$1; q=$2; shift 2
timeout 900 bun drive.ts "$n" "$q" --url http://127.0.0.1:8791/ --timeout-min 12 $([ -z "$STOCK" ] && echo --chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia) "$@" 2>&1 | grep -E "loadMs|fetchMs|backend|RssMib(After|Peak)|GpuMib|first|rror" | tr -s ' ' | tr '\n' ' '; echo
