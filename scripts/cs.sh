#!/bin/bash
# One Chrome run, compact summary (clips, memory, error, interesting steps).  usage: [STOCK=1] [STEPS=regex] cs.sh NAME "query" [drive args]
W=~/devfs/repos/kkrausse/random/.claude/worktrees/parakeet-ggml-web/parakeet-ggml-web; n=$1
$W/scripts/cr.sh "$@" > /dev/null
python3 - $W/results/browser/$n.json "${STEPS:-ERROR|FAILED|BELOW|WARNING|note:}" <<'PY'
import json,sys,re
d=json.load(open(sys.argv[1])); p=d.get('page') or {}; m=d.get('memory',{})
for c in p.get('clips',[]):
    g=lambda k: (c.get(k) or {}).get('median')
    print(f"  {c['clip']} first {c['firstRunMs']['total']:.0f} (dec {c['firstRunMs']['dec']:.0f}) | mel {g('preMs')} enc {g('encMs')} dec {g('decMs')} total {g('totalMs')} (worst {(c.get('totalMs') or {}).get('worst')}) match={c['matchesNative']} stable={c.get('textStable')}")
print(f"  load {p.get('load',{}).get('loadMs')} ms fetch {p.get('load',{}).get('fetchMs')} | gpu {m.get('gpuProcessGpuMibAfterLoad')}/{m.get('gpuProcessGpuMibPeak')} MiB renderer {m.get('rendererRssMibAfterLoad')}/{m.get('rendererRssMibPeak')} MiB | f16={p.get('shaderF16Used')} backend={p.get('load',{}).get('backend')} | {d['machineAtStart']['loadavg'][:14]}")
if d.get('error'): print("  ERROR:", d['error'][:1500])
for s in d.get('steps') or []:
    if re.search(sys.argv[2], s['name']+' '+(s.get('detail') or '')): print("  step:", (s['name']+'  '+(s.get('detail') or ''))[:700])
for l in (d.get('console') or [])[:4]: print("  console:", l[:300])
PY
