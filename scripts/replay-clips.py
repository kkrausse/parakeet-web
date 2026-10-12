# Recordings for the replay harness (web/drive-replay.ts, web/replay/gate.ts), written to
# ~/devfs/cache/parakeet-ggml-webgpu/audio/replay/ as 16 kHz mono float32:
#   iv6    6.5 min of an interview (one speaker, conversational, natural pauses); source: the autoanimate
#          repo's scenes/gromen-live/work/dl/src.m4a, from 2:00
#   iv90   its first 90 s
#   ivgap  iv90 cut at three pauses with 8, 5 and 12 s of digital silence put in and 6 s after (a condition's
#          room noise fills them): the gate has to close and open again
#   death  transcribe.cpp/samples/death.wav (3.9 min, a speech to an audience, its own room noise)
#   sil60  60 s of digital silence: with a condition's noise on top, a room where nobody speaks
#   iv6.wav  iv6 + 4 s of silence as a wav for Chrome's fake microphone (web/drive-live.ts)
# usage: python3 scripts/replay-clips.py
import array, os, subprocess, wave
R = os.environ.get('PK_CACHE', os.path.expanduser('~/devfs/cache/parakeet-ggml-webgpu')) + '/'
out = R + 'audio/replay/'
os.makedirs(out, exist_ok=True)
src = os.path.expanduser('~/devfs/repos/kkrausse/autoanimate/scenes/gromen-live/work/dl/src.m4a')
def ff(args, dst): subprocess.run(['ffmpeg', '-v', 'error', '-y', *args, '-ac', '1', '-ar', '16000', '-f', 'f32le', dst], check=True)
ff(['-ss', '120', '-t', '390', '-i', src], out + 'iv6.f32')
ff(['-i', R + 'transcribe.cpp/samples/death.wav'], out + 'death.f32')
x = array.array('f'); x.frombytes(open(out + 'iv6.f32', 'rb').read())
x[:90 * 16000].tofile(open(out + 'iv90.f32', 'wb'))
def quietest(at, span=1.5):  # the quietest 200 ms within span seconds of `at`: a pause to cut in
    best, bi = 1e9, 0
    for i in range(int((at - span) * 16000), int((at + span) * 16000), 320):
        e = sum(v * v for v in x[i:i + 3200])
        if e < best: best, bi = e, i + 1600
    return bi
cuts = [0, quietest(20), quietest(45), quietest(70), 90 * 16000]
g = array.array('f')
for k, sil in enumerate([8, 5, 12, 6]):
    g.extend(x[cuts[k]:cuts[k + 1]]); g.extend(array.array('f', [0.0]) * (sil * 16000))
g.tofile(open(out + 'ivgap.f32', 'wb'))
array.array('f', [0.0] * (60 * 16000)).tofile(open(out + 'sil60.f32', 'wb'))
w = wave.open(out + 'iv6.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(array.array('h', [max(-32768, min(32767, int(v * 32767))) for v in x] + [0] * (4 * 16000)).tobytes()); w.close()
(x + array.array('f', [0.0]) * (4 * 16000)).tofile(open(out + 'iv6pad.f32', 'wb'))
print('cuts at', [c / 16000 for c in cuts], '->', {n: os.path.getsize(out + n) for n in sorted(os.listdir(out))})
