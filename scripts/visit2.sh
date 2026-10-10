#!/bin/bash
# Second-visit timings: same Chrome profile kept between runs (model in OPFS, shader cache warm).  usage: [STOCK=1] visit2.sh "query" [n]
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web; cd $W/web
for i in $(seq 1 ${2:-3}); do
  timeout 600 bun drive.ts visit-$i "$1" --url http://127.0.0.1:8791/ --keep-profile $([ -z "$STOCK" ] && echo --chrome-arg --enable-dawn-features=vulkan_enable_f16_on_nvidia) > /dev/null 2>&1
  python3 - $W/results/browser/visit-$i.json $i <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); p=d['page']; c=p['clips'][0]; st={s['name'].split(' (')[0].split(':')[0]:s for s in d['steps']}
wasm=[s for s in d['steps'] if s['name'].startswith('WASM module ready')][0]
print(f"visit {sys.argv[2]}: wasm module {wasm['ms']} ms | model file {p['load']['fetchMs']} ms ({p['load']['from']}) | load {p['load']['loadMs']} ms | first {c['clip']} {c['firstRunMs']['total']:.0f} ms (enc {c['firstRunMs']['enc']:.0f}) | page start to first transcript {c['startToFirstTranscriptMs']} ms | f16={p['shaderF16Used']}")
PY
done
