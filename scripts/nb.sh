#!/bin/bash
# Native rebuild + browser-like bench of the three weight types.  usage: nb.sh [extra env...]   (prints encoder min/mean and text check)
R=~/devfs/cache/parakeet-ggml-webgpu; cd $R
cmake -S transcribe.cpp -B build-webgpu > /dev/null 2>&1
nice make -C build-webgpu -j6 transcribe-bench transcribe-cli > out/build-native.log 2>&1 || { grep -E "error" -A4 out/build-native.log | head -30; exit 1; }
EXP="going along slushy country roads and speaking to damp audiences in drafty schoolrooms day after day for a fortnight."
for q in ${QUANTS:-Q8_0 Q4_0 F16}; do
  env GGML_WEBGPU_BROWSER=1 ${FLASH:+TRANSCRIBE_F32_MASK_CONCAT=1} ${FLASH:-TRANSCRIBE_NO_FLASH=1} "$@" build-webgpu/bin/transcribe-bench --model gguf/parakeet-tdt-0.6b-v2-$q.gguf --sample audio/${CLIP:-a07}.wav --iters ${ITERS:-10} --warmup 2 --quiet --threads 4 > out/nb-$q.json 2> out/nb-$q.err < /dev/null || true
  python3 - out/nb-$q.json $q "$EXP" ${CLIP:-a07} <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print(sys.argv[2],'FAILED',e, open(sys.argv[1].replace('.json','.err')).read()[-160:]); sys.exit()
e=d['summary']['encode_ms']; ok = d['hyp_text']==sys.argv[3] if sys.argv[4]=='a07' else d['hyp_text'][:60]
print(f"{sys.argv[2]:5s} enc min {e['min']:.1f} mean {e['mean']:.1f}  dec {d['summary']['decode_ms']['min']:.0f}  text_ok={ok}")
PY
done
