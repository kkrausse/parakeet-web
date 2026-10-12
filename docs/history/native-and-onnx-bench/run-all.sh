#!/bin/sh
# Runs every benchmark in sequence and writes results/*.jsonl. See README.md for setup.
# usage: ./run-all.sh [name ...]   (no names: everything)
set -eu
cd "$(dirname "$0")"
HERE=$PWD
ORT=cache/target-ort/release/ort-bench
CLIPS="cache/audio/a07.f32 cache/audio/a14.f32 cache/audio/a56.f32"
UNIFIED=${UNIFIED_GGUF:-$HOME/devfs/repos/kkrausse/random/dictation-server-linux/.cache/models/parakeet-unified-en-0.6b-F16.gguf}
RUNS=${RUNS:-10}
mkdir -p results cache/logs cache/shaders-ort cache/shaders-tcpp

want() { [ $# -eq 0 ] && return 0; for n in $WANT; do [ "$n" = "$1" ] && return 0; done; [ -z "$WANT" ]; }
WANT="$*"
run() { name=$1; shift; want "$name" || return 0; echo "== $name" >&2
  # What else the box was doing: share of time some task waited for a CPU over the last 10 s.
  echo "$name $(date +%T) cpu-pressure $(head -1 /proc/pressure/cpu | cut -d' ' -f2) loadavg $(cut -d' ' -f1 /proc/loadavg)" >> results/machine-load.txt
  "$@" > "results/$name.jsonl" 2> "cache/logs/$name.err" || echo "FAILED: $name (cache/logs/$name.err)" >&2; }

# NVIDIA's on-disk Vulkan pipeline cache, pinned so the cold-start runs can empty it.
ort() { __GL_SHADER_DISK_CACHE_PATH=$HERE/cache/shaders-ort "$ORT" "$@"; }
tcpp() { __GL_SHADER_DISK_CACHE_PATH=$HERE/cache/shaders-tcpp python3 scripts/tcpp_bench.py "$@"; }

run ort-webgpu-fp32            ort --model cache/onnx-tdt-v2      --enc-ep webgpu --dec-ep cpu    --runs $RUNS $CLIPS
run ort-webgpu-fp16            ort --model cache/onnx-tdt-v2-fp16 --enc-ep webgpu --dec-ep cpu    --runs $RUNS $CLIPS
run ort-webgpu-fp32-dec-webgpu ort --model cache/onnx-tdt-v2      --enc-ep webgpu --dec-ep webgpu --runs $RUNS $CLIPS
run ort-cpu-fp32               ort --model cache/onnx-tdt-v2      --enc-ep cpu    --dec-ep cpu    --runs $RUNS $CLIPS
# Burn runs the encoder only, on mel features dumped by ort-bench (which also dumps the reference output).
if want burn-wgpu-fp32 || want burn-vulkan-spirv-fp32; then
  mkdir -p cache/dump cache/shaders-burn
  [ -e cache/dump/a56.feat-128x5611.f32 ] || "$ORT" --model cache/onnx-tdt-v2 --enc-ep cpu --runs 0 --dump cache/dump $CLIPS > /dev/null 2>&1
fi
burn() { __GL_SHADER_DISK_CACHE_PATH=$HERE/cache/shaders-burn "$@" --runs $RUNS cache/dump/a07 cache/dump/a14 cache/dump/a56; }
run burn-wgpu-fp32         burn cache/target-burn/release/burn-bench-wgsl  --backend wgpu
run burn-vulkan-spirv-fp32 burn cache/target-burn/release/burn-bench-spirv --backend vulkan
run tcpp-vulkan-tdt-f16        tcpp cache/gguf/parakeet-tdt-0.6b-v2-F16.gguf --runs $RUNS $CLIPS
run tcpp-vulkan-tdt-f32        tcpp cache/gguf/parakeet-tdt-0.6b-v2-F32.gguf --runs $RUNS $CLIPS
run tcpp-vulkan-unified-f16    tcpp "$UNIFIED" --runs $RUNS $CLIPS
run tcpp-vulkan-unified-f16-stream tcpp "$UNIFIED" --stream --runs $RUNS cache/audio/a14.f32 cache/audio/a56.f32

# Process start to first transcript of the 7 s clip: once with an empty pipeline cache, then three
# more processes that find it on disk.
cold() {
  dir=$1; shift
  rm -rf "$dir"; mkdir -p "$dir"
  for i in 1 2 3 4; do __GL_SHADER_DISK_CACHE_PATH=$dir "$@" --runs 0 cache/audio/a07.f32 | tail -1; done
}
run cold-ort-webgpu-fp32 cold "$HERE/cache/shaders-cold" "$ORT" --model cache/onnx-tdt-v2
run cold-ort-webgpu-fp16 cold "$HERE/cache/shaders-cold" "$ORT" --model cache/onnx-tdt-v2-fp16
run cold-tcpp-vulkan-tdt-f16 cold "$HERE/cache/shaders-cold" python3 scripts/tcpp_bench.py cache/gguf/parakeet-tdt-0.6b-v2-F16.gguf
