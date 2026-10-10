U=C86CCB3B-ABF0-47C0-9722-181B89E510C6
D=$(find ~/Library/Developer/CoreSimulator/Devices/$U/data/Containers/Data/Application -type d -path "*com.apple.mobilesafari/WebsiteData*" -prune -print | head -1)
D=$(dirname "$D")/WebsiteData
python3 - "$D" <<'PY'
import sys,os,sqlite3,json,glob
root=sys.argv[1]
fs=[os.path.join(dp,f) for dp,dn,fn in os.walk(root) for f in fn if f.endswith('.sqlite3') and 'ocal' in (dp+f)]
for f in fs:
    try:
        con=sqlite3.connect('file:'+f+'?mode=ro',uri=True)
        for k,v in con.execute('select key,value from ItemTable'):
            if k!='pkggml:runs': continue
            runs=json.loads(bytes(v).decode('utf-16-le'))
            for r in runs[-int(os.environ.get('LAST','1')):]:
                print('RUN',r['started'],r['config']['model'],r['config']['variant'],'done=',r['done'],'error=',r.get('error'))
                print('UA',r['ua'])
                for s in r['steps']:
                    print('%7.1fs  %s%s%s'%(s['t']/1000,s['name'],(': %s ms'%s['ms']) if 'ms' in s else '',('  (%s)'%s['detail']) if s.get('detail') else ''))
    except Exception as e: print('ERR',f,e)
if not fs: print('no localStorage db under',root)
PY
