# Test audio for the live page, written to ~/devfs/cache/parakeet-ggml-webgpu/audio/live/: the fixtures
# with digital silence between and after them (so pauses are there to be detected), as 16 kHz mono wav
# for Chrome's fake microphone, the same samples as .f32 for the offline pass, and the 110m reference
# text of the parts joined (.txt).   usage: python3 scripts/live-clips.py
import array, os, re, wave
A = os.path.expanduser('~/devfs/cache/parakeet-ggml-webgpu/audio/')
src = open(os.path.join(os.path.dirname(__file__), '../web/src/main.ts')).read()
ref = dict(re.findall(r'"(a\d\d)": "([^"]+)"', src[src.index('EXPECTED_110M'):src.index('const CLIPS')]))
def rd(n):
    w = wave.open(A + n + '.wav'); a = array.array('h'); a.frombytes(w.readframes(w.getnframes())); return a
def wr(name, parts):
    x = array.array('h')
    for p in parts: x.extend(array.array('h', [0]) * int(p * 16000) if isinstance(p, (int, float)) else rd(p))
    w = wave.open(A + 'live/' + name + '.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes()); w.close()
    array.array('f', [v / 32768 for v in x]).tofile(open(A + 'live/' + name + '.f32', 'wb'))
    open(A + 'live/' + name + '.txt', 'w').write(' '.join(ref[p] for p in parts if isinstance(p, str)))
    print(name, len(x) / 16000, 's')
os.makedirs(A + 'live', exist_ok=True)
wr('t14', ['a14', 6])
wr('t56', ['a56', 6])
wr('g2', ['a56', 3, 'a14', 3, 'a56', 6])
wr('g5', ['a56', 3, 'a14', 2, 'a56', 3, 'a07', 4, 'a56', 2, 'a14', 3, 'a56', 3, 'a07', 6])
