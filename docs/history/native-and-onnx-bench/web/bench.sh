#!/bin/sh
# Browser rows on diesel2: real Chrome on Xvfb, WebGPU on the NVIDIA card, one fresh Chrome and
# profile per row, in the same high-weight systemd slice as ../bench.sh. Results: ../results/browser/.
# usage: ./bench.sh [name ...]      (bun serve.ts must be running: tmux new -d -s pkb-serve 'bun serve.ts 8787')
cd "$(dirname "$0")"
F16="--chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia"
Q="clip=all&runs=${RUNS:-10}&cache=0"
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
  echo "$(date +%T) $name $(cat /proc/loadavg) | $(head -1 /proc/pressure/cpu)" >> ../results/browser/machine-load.txt
  systemd-run --user --wait --pipe --collect --quiet --slice=parakeetbench.slice -p CPUWeight=10000 -p Nice=0 \
    -E PATH="$PATH" -E HOME="$HOME" --working-directory="$PWD" bun drive.ts "$name" "$@"
}
ONLY="$*"
mkdir -p ../results/browser
systemctl --user start parakeetbench.slice
systemctl --user set-property --runtime parakeetbench.slice CPUWeight=10000
native() { # the native ONNX Runtime WebGPU row again, in the same window as its browser row
  name=$1; shift
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then return; fi
  quiet
  echo "== $name" >&2
  echo "$(date +%T) $name $(cat /proc/loadavg) | $(head -1 /proc/pressure/cpu)" >> ../results/browser/machine-load.txt
  (cd .. && systemd-run --user --wait --pipe --collect --quiet --slice=parakeetbench.slice -p CPUWeight=10000 -p Nice=0 \
    -E PATH="$PATH" -E HOME="$HOME" -E __GL_SHADER_DISK_CACHE_PATH="$PWD/cache/shaders-ort" --working-directory="$PWD" \
    cache/target-ort/release/ort-bench "$@" --runs "${RUNS:-10}" cache/audio/a07.f32 cache/audio/a14.f32 cache/audio/a56.f32) > "../results/browser/$name.jsonl"
}
native native-ort-webgpu-fp32     --model cache/onnx-tdt-v2 --enc-ep webgpu --dec-ep cpu
row chrome-webgpu-fp32            "model=fp32&enc=webgpu&dec=wasm&$Q"
native native-ort-webgpu-fp16     --model cache/onnx-tdt-v2-fp16 --enc-ep webgpu --dec-ep cpu
row chrome-webgpu-fp16            "model=fp16&enc=webgpu&dec=wasm&$Q" $F16
row chrome-webgpu-fp16-stock      "model=fp16&enc=webgpu&dec=wasm&$Q"            # no Dawn toggle: expected to fail on driver 580
row chrome-webgpu-fp32-dec-webgpu "model=fp32&enc=webgpu&dec=webgpu&$Q"
row chrome-jsep-fp32              "model=fp32&enc=webgpu&dec=wasm&rt=jsep&$Q"
row chrome-wasm-fp32-12t          "model=fp32&enc=wasm&dec=wasm&threads=12&$Q"
row chrome-wasm-int8-12t          "model=int8&enc=wasm&dec=wasm&threads=12&$Q"
row chrome-wasm-int8-4t           "model=int8&enc=wasm&dec=wasm&threads=4&$Q"
row chrome-wasm-int8-1t           "model=int8&enc=wasm&dec=wasm&threads=1&$Q"
row chrome-webgpu-int8            "model=int8&enc=webgpu&dec=wasm&$Q"

# parakeet-tdt_ctc-110m, the small model: ./bench.sh $(sed -n 's/^\(row\|native\) \([a-z0-9-]*110m[a-z0-9-]*\).*/\2/p' bench.sh)
S=cache/onnx-tdt-ctc-110m
I8="--encoder encoder-model.int8.onnx --decoder decoder_joint-model.int8.onnx"
native native-110m-ort-webgpu-fp32 --model $S --enc-ep webgpu --dec-ep cpu
row chrome-110m-webgpu-fp32        "model=110m-fp32&enc=webgpu&dec=wasm&$Q"
native native-110m-ort-webgpu-fp16 --model $S-fp16 --enc-ep webgpu --dec-ep cpu
row chrome-110m-webgpu-fp16        "model=110m-fp16&enc=webgpu&dec=wasm&$Q" $F16
row chrome-110m-webgpu-fp16-stock  "model=110m-fp16&enc=webgpu&dec=wasm&$Q"       # no Dawn toggle: expected to fail on driver 580
native native-110m-ort-cpu-int8-1t --model $S $I8 --enc-ep cpu --threads 1 --dec-threads 1
row chrome-110m-wasm-int8-1t       "model=110m-int8&enc=wasm&dec=wasm&threads=1&$Q"
row chrome-110m-wasm-int8-4t       "model=110m-int8&enc=wasm&dec=wasm&threads=4&$Q"
native native-110m-ort-cpu-fp32-1t --model $S --enc-ep cpu --threads 1 --dec-threads 1
row chrome-110m-wasm-fp32-1t       "model=110m-fp32&enc=wasm&dec=wasm&threads=1&$Q"
row chrome-110m-wasm-fp32-4t       "model=110m-fp32&enc=wasm&dec=wasm&threads=4&$Q"
