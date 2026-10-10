#!/bin/bash
# Per-shader GPU ms per run (browser-like native, timestamp queries). usage: prof.sh QUANT [env...]
R=~/devfs/cache/parakeet-ggml-webgpu; cd $R; q=$1; shift
cmake -S transcribe.cpp -B build-webgpu-prof > /dev/null 2>&1
nice make -C build-webgpu-prof -j6 transcribe-bench > out/build-prof.log 2>&1 || { grep error -A4 out/build-prof.log | head; exit 1; }
env GGML_WEBGPU_BROWSER=1 TRANSCRIBE_NO_FLASH=1 "$@" build-webgpu-prof/bin/transcribe-bench --model gguf/parakeet-tdt-0.6b-v2-$q.gguf --sample audio/${CLIP:-a07}.wav --iters 10 --warmup 2 --quiet --threads 4 > out/prof-$q.txt 2>&1
python3 - out/prof-$q.txt <<'PY'
import re,sys
rows=[]; tot=0
for l in open(sys.argv[1]):
    m=re.match(r'ggml_webgpu:  (\S+): ([\d.]+) ms \(([\d.]+)%\)',l)
    if m and not m.group(1) in ('graph_compute','set_tensor','get_tensor','reg_get_device','memset_tensor','clear'): rows.append((float(m.group(2))/12,m.group(1)))
    m=re.match(r'ggml_webgpu: total gpu time.*: ([\d.]+)',l)
    if m: tot=float(m.group(1))/12
print(f"total gpu {tot:.1f} ms/run")
for v,n in sorted(rows,reverse=True)[:14]: print(f"  {v:6.2f}  {n}")
PY
