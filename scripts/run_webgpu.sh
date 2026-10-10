#!/bin/bash
# usage: run_webgpu.sh GGUF TAG [env assignments...]   -> bench json + peak GPU MiB + op placement
R=~/devfs/cache/parakeet-ggml-webgpu; M=$1; TAG=$2; shift 2
env "$@" $R/build-webgpu/bin/transcribe-bench --model $M --sample $R/audio/a07.wav --iters 8 --warmup 2 --quiet > $R/out/bench-$TAG.json 2> $R/out/bench-$TAG.err &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v p=$PID '$1==p{print $2+0}')
  [ -n "$m" ] && [ "$m" -gt "$PEAK" ] && PEAK=$m; sleep 0.2
done
python3 - $R/out/bench-$TAG.json $TAG $PEAK <<'PY'
import json,sys,statistics as s
try: d=json.load(open(sys.argv[1]))
except Exception as e: print(sys.argv[2],'FAILED',e); sys.exit()
it=d['per_iter']
f=lambda k:(round(min(x[k] for x in it),1), round(s.median(x[k] for x in it),1))
print(sys.argv[2], d['backend'], 'enc best/median', f('encode_ms'), 'dec', f('decode_ms'), 'load', round(d['load_ms']), 'gpu_peak_MiB', sys.argv[3]); print('  ', d['hyp_text'])
PY
