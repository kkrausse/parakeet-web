# Compact summary of results/live/NAME.json files.   usage: python3 scripts/lsum.py NAME [NAME ...]
import json, os, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/live')
for n in sys.argv[1:]:
    d = json.load(open(f'{D}/{n}.json'))
    if d.get('error'): print(n, 'ERROR', d['error']); continue
    L = d['live']; m = d['memoryAfterWarmup']; g = lambda x, k: (x or {}).get(k)
    r = lambda k: f"{m[k]['first']}->{m[k]['last']} ({m[k]['min']}-{m[k]['max']})"
    v = lambda k: f"{d[k]['normalized']['differing']}/{d[k]['normalized']['words']} words ({d[k]['raw']['differing']} raw)" if k in d else '-'
    reasons = {}
    for s in L['segments']: reasons[s['reason']] = reasons.get(s['reason'], 0) + 1
    print(f"{n}: {d['recordSeconds']} s, {L['backend']}{' f16' if L['shaderF16'] else ' f32-only' if L['webgpu'] else ''} {L['variant']}, mic {L['inputRate']} Hz | passes {L['passCount']} (stale {L['staleCount']}), commits {L['finalCount']} {reasons} ({L.get('reusedCount',0)} without a final pass)"
          f" | provisional ms median/p95/max {g(d['provisionalPassMs'],'median')}/{g(d['provisionalPassMs'],'p95')}/{g(d['provisionalPassMs'],'max')} on {g(d['provisionalAudioS'],'median')} s median, final {g(d['finalPassMs'],'median')}/{g(d['finalPassMs'],'max')}"
          f" | lag median/max {g(d['lagS'],'median')}/{d['maxLagS']} s, drain {d['drainMs']} ms, dropped {L['droppedS']} s"
          f" | vs offline {v('vsOffline')}, vs joined references {v('vsExpected')}"
          f" | GPU MiB {r('gpuMib')}, renderer RSS {r('rendererRssMib')}, WASM heap {r('wasmHeapMb')}, JS heap {r('jsHeapMb')} | errors {len(L['errors'])} | load {d['loadavgStart'].split()[0]}")
