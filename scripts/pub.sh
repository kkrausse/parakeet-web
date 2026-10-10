#!/bin/bash
# Rebuild the site, publish it in place (private shelf) and verify the published link from stock Chrome.  usage: pub.sh [query]
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web; cd $W/web && bun build.ts | tail -1
~/devfs/repos/kkrausse/random/scripts/deploy-artifact.sh "$W/web/dist" parakeet-ggml-browser 2>&1 | tail -2
U=https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-ggml-browser/
timeout 900 bun drive.ts published-default-stock "${1:-clip=all&runs=5}" --url $U --timeout-min 12 > /dev/null 2>&1
STEPS="shader path|ERROR|FAILED|model file" python3 - $W/results/browser/published-default-stock.json <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); p=d.get('page') or {}; m=d.get('memory',{})
print(d.get('url'))
for c in p.get('clips',[]):
    g=lambda k: (c.get(k) or {}).get('median')
    print(f"  {c['clip']} first {c['firstRunMs']['total']:.0f} | mel {g('preMs')} enc {g('encMs')} dec {g('decMs')} total {g('totalMs')} match={c['matchesNative']} stable={c.get('textStable')}")
print(f"  model {p.get('model',{}).get('label')} | f16={p.get('shaderF16Used')} backend={p.get('load',{}).get('backend')} | load {p.get('load',{}).get('loadMs')} fetch {p.get('load',{}).get('fetchMs')} | gpu {m.get('gpuProcessGpuMibAfterLoad')}/{m.get('gpuProcessGpuMibPeak')} renderer {m.get('rendererRssMibAfterLoad')}/{m.get('rendererRssMibPeak')} | error={d.get('error')}")
PY
