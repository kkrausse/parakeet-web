#!/bin/bash
# Long-audio native run: text saved to out/lt/NAME.txt, timings, GPU peak (nvidia-smi, this process), word diff against REF.
# usage: [ITERS=3] [REF=name] lt.sh NAME MODEL(s8|s4|q4|q8|f16) CLIP [ENV=1 ...]
R=~/devfs/cache/parakeet-ggml-webgpu; cd $R; mkdir -p out/lt; N=$1; M=$2; C=$3; shift 3
declare -A G=([s8]=parakeet-tdt_ctc-110m-Q8_0 [s4]=parakeet-tdt_ctc-110m-Q4_0 [q4]=parakeet-tdt-0.6b-v2-Q4_0 [q8]=parakeet-tdt-0.6b-v2-Q8_0 [f16]=parakeet-tdt-0.6b-v2-F16)
env GGML_WEBGPU_BROWSER=1 TRANSCRIBE_NO_FLASH=1 TRANSCRIBE_PRE_ENCODE_TILE=128 TRANSCRIBE_ENC_PROJ_GPU=1 "$@" build-webgpu/bin/transcribe-bench --model gguf/${G[$M]}.gguf --sample audio/$C.wav --iters ${ITERS:-3} --warmup 0 --quiet --threads ${THREADS:-1} > out/lt/$N.json 2> out/lt/$N.err < /dev/null &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v p=$PID '$1==p{print $2+0}')
  [ -n "$m" ] && [ "$m" -gt "$PEAK" ] && PEAK=$m; sleep 0.05
done
wait $PID || { echo "$N FAILED: $(tail -c 400 out/lt/$N.err)"; exit 1; }
python3 - "$N" "$M" "$C" "$PEAK" "${REF:-}" "$*" <<'PY'
import json,sys,os,re,difflib,wave
n,m,c,peak,ref,env=sys.argv[1:7]; d=json.load(open(f'out/lt/{n}.json')); s=d['summary']; t=d['hyp_text']
open(f'out/lt/{n}.txt','w').write(t)
w=wave.open(f'audio/{c}.wav'); secs=w.getnframes()/16000
tot=s['mel_ms']['min']+s['encode_ms']['min']+s['decode_ms']['min']
line=f"{n:14s} {m} {c} {secs:6.1f}s mel {s['mel_ms']['min']:.0f} enc {s['encode_ms']['min']:.0f} dec {s['decode_ms']['min']:.0f} total {tot:.0f} ms = {secs*1000/tot:.0f}x RT | GPU peak {peak} MiB | {len(t.split())} words"
if ref and os.path.exists(f'out/lt/{ref}.txt'):
    r=open(f'out/lt/{ref}.txt').read()
    if r==t: line+=' | text identical to '+ref
    else:
        norm=lambda x: re.sub(r"[^\w\s']",'',x.lower()).split()
        a,b=r.split(),t.split(); na,nb=norm(r),norm(t)
        ops=[o for o in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if o[0]!='equal']
        wops=[o for o in difflib.SequenceMatcher(None,na,nb,autojunk=False).get_opcodes() if o[0]!='equal']
        line+=f" | vs {ref}: {len(ops)} differing spans, {len(wops)} after dropping punctuation/case"
        for tag,i1,i2,j1,j2 in ops[:int(os.environ.get('SHOW','6'))]: line+=f"\n      [{' '.join(a[max(0,i1-2):i2+1])}] -> [{' '.join(b[max(0,j1-2):j2+1])}]"
if os.path.exists(f'audio/{c}.seq') and all(os.path.exists(f'out/ref/{m}-{x}.txt') for x in open(f'audio/{c}.seq').read().split()):
    # the clip is a concatenation of fixtures: word errors against the fixtures' own single-shot transcripts, joined
    norm=lambda x: re.sub(r"[^\w\s']",'',x.lower()).split()
    r=norm(' '.join(open(f'out/ref/{m}-{x}.txt').read() for x in open(f'audio/{c}.seq').read().split())); h=norm(t)
    prev=list(range(len(h)+1))
    for i,rw in enumerate(r,1):
        cur=[i]+[0]*len(h)
        for j,hw in enumerate(h,1): cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(rw!=hw))
        prev=cur
    line+=f"\n      word errors vs the fixtures' own transcripts joined: {prev[-1]} of {len(r)} ({100*prev[-1]/len(r):.1f}%)"
print(line)
PY
