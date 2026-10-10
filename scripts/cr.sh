#!/bin/bash
# One Chrome run against the local server (tmux pkg-serve, port 8791).  usage: cr.sh NAME "query" [drive args]
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web/web; cd $W
n=$1; q=$2; shift 2
timeout 900 bun drive.ts "$n" "$q" --url http://127.0.0.1:8791/ --timeout-min 12 --chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia "$@" 2>&1 | grep -E "loadMs|fetchMs|RssMib(After|Peak)|GpuMib|first|rror" | tr -s ' ' | tr '\n' ' '; echo
