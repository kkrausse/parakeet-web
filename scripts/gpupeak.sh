#!/bin/bash
# Peak GPU MiB (nvidia-smi, this process) of one native browser-like run. usage: gpupeak.sh GGUF CLIP [env...]
R=~/devfs/cache/parakeet-ggml-webgpu; M=$1; C=$2; shift 2
env GGML_WEBGPU_BROWSER=1 TRANSCRIBE_NO_FLASH=1 "$@" $R/build-webgpu/bin/transcribe-bench --model $R/gguf/$M --sample $R/audio/$C.wav --iters ${ITERS:-6} --warmup 0 --quiet --threads 4 > $R/out/gp.json 2> $R/out/gp.err &
PID=$!; PEAK=0; MIN=999999
while kill -0 $PID 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v p=$PID '$1==p{print $2+0}')
  [ -n "$m" ] && [ "$m" -gt "$PEAK" ] && PEAK=$m; sleep 0.05
done
python3 -c "
import json,sys; d=json.load(open('$R/out/gp.json')); print('$M $C $*', 'gpu_peak_MiB', $PEAK, 'enc_min %.1f'%d['summary']['encode_ms']['min'], d['hyp_text'][:40])"
