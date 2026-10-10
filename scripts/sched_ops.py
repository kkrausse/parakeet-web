import re, sys, collections
graphs=[]; cur=None; be=None
for line in open(sys.argv[1], errors='replace'):
    m=re.search(r'## SPLIT #(\d+): (.*?) # (\d+) inputs', line)
    if m:
        if m.group(1)=='0':
            cur={'splits':0,'ops':collections.Counter(),'nodes':collections.defaultdict(list)}; graphs.append(cur)
        cur['splits']+=1; be='CPU' if m.group(2).startswith('CPU') else 'WebGPU'; continue
    m=re.search(r'node #\s*(\d+) \(\s*(\S+)\): +(\S.*?) \(', line)
    if m and cur is not None:
        cur['ops'][(be,m.group(2))]+=1
        if be=='CPU': cur['nodes'][m.group(2)].append(m.group(3))
sig=collections.Counter()
for i,g in enumerate(graphs):
    key=(g['splits'],tuple(sorted(g['ops'].items())))
    sig[key]+=1
    if sig[key]>1: continue
    n=sum(g['ops'].values())
    print(f"--- graph {i}: {n} nodes, {g['splits']} splits")
    for b in ('WebGPU','CPU'):
        print(' ',b, sum(v for (bb,_),v in g['ops'].items() if bb==b), dict(sorted(((o,v) for (bb,o),v in g['ops'].items() if bb==b), key=lambda x:-x[1])))
    for op,names in g['nodes'].items(): print('   CPU',op,len(names),names[:4])
print('graph kinds and repeat counts:',[(k[0],v) for k,v in sig.items()])
