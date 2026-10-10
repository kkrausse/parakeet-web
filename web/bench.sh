#!/bin/sh
# Browser rows on diesel2: real Chrome on Xvfb, WebGPU on the NVIDIA card, one fresh Chrome and
# profile per row, in the same high-weight systemd slice as ../../parakeet-webgpu-bench/bench.sh.
# Results: ../results/browser/<name>.json, load at the start of each row in machine-load.txt.
# usage: ./bench.sh [name ...]      (server: tmux new -d -s pkg-serve 'bun serve.ts 8791')
cd "$(dirname "$0")"
F16="--chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia"
Q="clip=all&runs=${RUNS:-10}"
quiet() { # wait (up to QUIET_WAIT seconds, default 240) for a moment when other work on the box leaves the CPUs mostly alone
  i=0
  while [ $i -lt $(( ${QUIET_WAIT:-240} / 5 )) ]; do
    p=$(sed -n 's/^some avg10=\([0-9]*\).*/\1/p' /proc/pressure/cpu)
    [ "$p" -lt "${QUIET:-5}" ] && return
    i=$((i + 1)); sleep 5
  done
}
row() {
  name=$1; shift
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then return; fi
  quiet
  echo "== $name" >&2
  echo "$(date +%F\ %T) $name $(cat /proc/loadavg) | $(head -1 /proc/pressure/cpu)" >> ../results/browser/machine-load.txt
  systemd-run --user --wait --pipe --collect --quiet --slice=parakeetbench.slice -p CPUWeight=10000 -p Nice=0 \
    -E PATH="$PATH" -E HOME="$HOME" --working-directory="$PWD" bun drive.ts "$name" "$@" --url http://127.0.0.1:8791/ 2>&1 | grep -E '"a[0-9]+:|error|RssMibPeak|GpuMibPeak'
}
ONLY="$*"
mkdir -p ../results/browser
systemctl --user start parakeetbench.slice 2>/dev/null
systemctl --user set-property --runtime parakeetbench.slice CPUWeight=10000 2>/dev/null
row ggml-q4            "model=q4&$Q" $F16
row ggml-q8            "model=q8&$Q" $F16
row ggml-f16           "model=f16&$Q" $F16
row ggml-q4km          "model=q4km&$Q" $F16   # needs PK_EXTRA_MODELS at build time
row ggml-q4-flash      "model=q4&flash=1&$Q" $F16
row ggml-q8-flash      "model=q8&flash=1&$Q" $F16
row ggml-q4-stock      "model=q4&$Q"                         # no Dawn toggle: stock Chrome, no shader-f16 on driver 580 -> f32-only shaders
row ggml-q8-stock      "model=q8&$Q"
row ggml-f16-stock     "model=f16&$Q"
row ggml-q4-nof16      "model=q4&f16=0&$Q" $F16              # f16 available but not used (same Chrome flags as the f16 rows)
row ggml-q4-blob       "model=q4&store=blob&clip=a07&runs=3" $F16
row ggml-q4-opfs-blob  "model=q4&store=opfs-blob&clip=a07&runs=3" $F16
row ggml-q4-memfs      "model=q4&store=memfs&clip=a07&runs=3" $F16
row ggml-q4-asyncify   "model=q4&variant=asyncify&$Q" $F16
