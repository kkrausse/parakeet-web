# Chrome result vs a native lt.sh transcript: differing spans.  usage: cmpn.py RESULT_NAME NATIVE_LT_NAME
import json,sys,os,difflib
d=json.load(open(os.path.dirname(os.path.abspath(__file__))+f'/../results/browser/{sys.argv[1]}.json'))
t=d['page']['clips'][-1]['text'].split(); n=open(os.path.expanduser(f'~/devfs/cache/parakeet-ggml-webgpu/out/lt/{sys.argv[2]}.txt')).read().split()
ops=[o for o in difflib.SequenceMatcher(None,n,t,autojunk=False).get_opcodes() if o[0]!='equal']
print(sys.argv[1],'vs native',sys.argv[2],':',len(ops),'differing spans of',len(n),'words')
