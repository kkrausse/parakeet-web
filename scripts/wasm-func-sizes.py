import re,sys
# per function: name, number of locals, text lines (a proxy for body size), whether asyncify-instrumented
cur=None; rows=[]
loc=0; lines=0; asy=False
for line in open(sys.argv[1], errors='replace'):
    if line.startswith(' (func '):
        if cur: rows.append((cur,loc,lines,asy))
        m=re.match(r' \(func (\$\S+)', line); cur=m.group(1) if m else '?'
        loc=0; lines=0; asy=False
    elif cur:
        lines+=1
        s=line.lstrip()
        if s.startswith('(local '): loc+=1
        if '__asyncify_state' in s: asy=True
if cur: rows.append((cur,loc,lines,asy))
print('functions',len(rows),'instrumented',sum(1 for r in rows if r[3]))
print('total lines',sum(r[2] for r in rows),'in instrumented',sum(r[2] for r in rows if r[3]))
for r in sorted(rows,key=lambda r:-r[2])[:int(sys.argv[2]) if len(sys.argv)>2 else 40]:
    print(f'{r[1]:7d} locals {r[2]:8d} lines {"ASY" if r[3] else "   "} {r[0][:150]}')
