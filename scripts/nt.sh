#!/bin/bash
# Native browser-like text + timing check over models x clips against saved references (out/ref).
# usage: [MODELS="s8 s4 q4 q8 f16"] [CLIPS="a07 a14 a56"] [SAVE=1] [ITERS=5] nt.sh [ENV=1 ...]
R=~/devfs/cache/parakeet-ggml-webgpu; cd $R; mkdir -p out/ref
declare -A G=([s8]=parakeet-tdt_ctc-110m-Q8_0 [s4]=parakeet-tdt_ctc-110m-Q4_0 [s32]=parakeet-tdt_ctc-110m-F32 [q4]=parakeet-tdt-0.6b-v2-Q4_0 [q8]=parakeet-tdt-0.6b-v2-Q8_0 [f16]=parakeet-tdt-0.6b-v2-F16)
for m in ${MODELS:-s8 s4 q4 q8 f16}; do for c in ${CLIPS:-a07 a14 a56}; do
  env GGML_WEBGPU_BROWSER=1 TRANSCRIBE_NO_FLASH=1 "$@" ${BIN:-build-webgpu/bin/transcribe-bench} --model gguf/${G[$m]}.gguf --sample audio/$c.wav --iters ${ITERS:-5} --warmup 1 --quiet --threads ${THREADS:-1} > out/nt.json 2> out/nt.err < /dev/null || { echo "$m $c FAILED: $(tail -c 300 out/nt.err)"; continue; }
  python3 - $m $c ${SAVE:-0} <<'PY'
import json,sys,os
m,c,save=sys.argv[1:4]; d=json.load(open('out/nt.json')); s=d['summary']; ref=f'out/ref/{m}-{c}.txt'
if save=='1': open(ref,'w').write(d['hyp_text'])
ok = os.path.exists(ref) and open(ref).read()==d['hyp_text']
print(f"{m:3s} {c} mel {s['mel_ms']['min']:.1f} enc {s['encode_ms']['min']:.1f} dec {s['decode_ms']['min']:.1f} text={'same' if ok else 'DIFFERENT: '+d['hyp_text'][:200]}")
PY
done; done
