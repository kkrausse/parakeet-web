#!/bin/bash
# Decoder A/B: text and decode ms with the fp32 decoder mirrors vs packed weights. usage: deccmp.sh [MODEL_PREFIX] [THREADS]
R=~/devfs/cache/parakeet-ggml-webgpu; cd $R
P=${1:-parakeet-tdt-0.6b-v2}; T=${2:-1}
for q in ${QUANTS:-Q4_0 Q8_0}; do for c in a07 a14 a56; do
  for m in TRANSCRIBE_DECODER_F32=1 X=1; do
    env GGML_WEBGPU_BROWSER=1 TRANSCRIBE_NO_FLASH=1 $m "${@:3}" build-webgpu/bin/transcribe-bench --model gguf/$P-$q.gguf --sample audio/$c.wav --iters 5 --warmup 1 --quiet --threads $T > out/dc-$m.json 2>out/dc-$m.err < /dev/null || echo FAIL $q $c $m
  done
  python3 - $q $c <<'PY'
import json,sys
a=json.load(open('out/dc-TRANSCRIBE_DECODER_F32=1.json')); b=json.load(open('out/dc-X=1.json'))
print(sys.argv[1],sys.argv[2],'same_text',a['hyp_text']==b['hyp_text'],'dec f32 %.1f packed %.1f'%(a['summary']['decode_ms']['min'],b['summary']['decode_ms']['min']), 'enc %.0f'%b['summary']['encode_ms']['min'])
if a['hyp_text']!=b['hyp_text']: print('  f32   :',a['hyp_text']); print('  packed:',b['hyp_text'])
PY
done; done
