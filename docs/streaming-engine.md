# Streaming engine: the rule and its measurements

Moved here unchanged from the work log this library came out of (`kkrausse/random`, branch `parakeet-live-lib`,
`parakeet-ggml-web/README.md`, section "Streaming engine"). Written 11 Oct 2026, before the engine moved into
`lib/src/stream.ts`; "the page" is the demo page in `web/`. Result files named here are in `results/`.

## Streaming engine (11 Oct): commit by agreement, the gate only saves work

**What was wrong.** On an iPhone (110m Q4_0, raw microphone) the transcript came in chunks and
stopped for seconds at a time: provisional text grew for 1.7 s and was never committed, then 7 s
with no pass at all; 12 commits in 53 s, several of one word. Passes took 30-80 ms, so it was not
speed. The page's energy endpointer decided what was speech, and three things in it lose words.
Reproduced offline (`bun web/replay/gate.ts` and `bun web/replay/sweep.ts`, no model needed; `web/replay/old.ts` is that
endpointer lifted out of the page, same arithmetic) on a 90 s interview excerpt made phone-like:

| what the old endpointer did | measured (90 s interview, stand-in model that can read everything) |
|---|---|
| Noise floor rose 5% per 20 ms frame on every frame it had called non-speech, the quiet parts of speech included; threshold 3x the floor | Speech at full level, room noise 12 dB under it, level drifting +-6 dB and phrases fading 10 dB: floor up to 0.083 (true noise 0.016), **19.7% of speech never given to a pass**, 57 of 238 words gone. With the creep switched off (`floorUp: 0`): 0%. Flat-level speech hides it (1.6%). |
| Absolute threshold 0.004 RMS (-48 dBFS) | A raw microphone 24 dB down with the level drifting and phrases fading: speech frames have a median RMS of 0.003. 19-26% of speech never given to a pass at every noise level tried, 19% with the creep off (1.4% when the level is flat). 34 dB down ("far"): 68%, longest hole 21 s. |
| Utterances with under 120 ms of flagged frames dropped without a pass | 2-4 per 90 s in those conditions, all but one with provisional text already on screen (the "1.7 s that vanished"). |

So the diagnosis from the phone log holds, with one addition: at a low raw level the fixed
threshold alone does most of the damage; the creep is what makes a normal level fail once there is
room noise and the level moves. With the real model in Chrome the old logic committed 192 of 239
words in the "phone" condition and 89 of 238 in "far" (table below).

**The rule now.** The model decides what is speech; the gate decides only when passes are worth
running.

- One buffer of recent audio. While the gate is open it is transcribed again when the previous
  pass is done, not less than 250 ms after that pass started, and only with 100 ms of new audio.
  Each pass returns words with times (`pk_words()`, 80 ms grid); they are mapped to absolute time.
- A word is **committed** when (a) every word before it is, (b) the last two passes or more had
  the same word (case and punctuation ignored) within 240 ms of the same place, for at least
  `stableS` = 0.5 s of audio, and (c) `holdS` = 1.2 s of audio follows its end. Committed text is
  append-only. Words of a new pass that lie before the committed point are left context; where that
  point falls in the new pass is found by time and corrected by matching the last committed words
  (a word can sit a frame earlier or later from pass to pass; this correction moved the split in
  about 30% of passes).
- After a commit the buffer start moves up to `keepS` = 4 s before the committed point, into the
  widest gap between committed words there, on the 80 ms grid. In continuous speech the buffer is
  5.8 s at the median and passes are that long.
- A word's end time is capped at 1.2 s after its start: a closing full stop is often emitted only
  after the pause ("avail." from 1.9 s to 11.5 s), and such a word otherwise never gets its 1.2 s
  of audio after it. Found by the first replay with silences in it.
- **Full buffer**: a pass over the whole cap commits every word that began 1 s before its end and
  moves on. Nothing waits behind a full buffer. (Not reached in any run with a real-time model:
  longest buffer 10 s at a 12 s cap.)
- **Idle**: when the gate has been closed for `hangS` = 1.5 s and the model's newest word is that
  old too, the remaining words are committed (from the newest pass if it saw the end, else after
  one more pass) and the utterance ends. If the model still produces words under a closed gate the
  utterance stays open ("held open by the model" in the stats).
- **Gate**: 20 ms frames; noise floor = the lowest 60 ms average of the last 4 s (minimum
  statistics: speech cannot raise it above its own quietest 60 ms, and it falls at once); speech
  when RMS > max(0.0004, 2.5 x floor); two such frames open it, 0.4 s of pre-roll. Nothing that
  has words on screen is ever discarded because of it.
- **Probe**: with the gate closed, every 2 s one pass over the last 3 s (not before the end of
  the previous utterance; not when the window is digital silence). Words there mean the gate
  missed speech: the utterance opens from the start of that window and the miss is counted. With
  the gate made deaf on purpose (`thr=1`) the engine still committed 236 of 242 and 235 of 236
  words through probes alone, first text 0.84-1.32 s after a word at the 95th percentile, longest
  stretch with no text 0.5 s.
- GPU error, device loss or a failed pass: the engine stops (`onError`), the page shows it and
  offers a reload. A pass result for an utterance that no longer exists is dropped and counted.

`holdS` and `keepS` came from a sweep (3 min of the interview in two conditions, 2 min of a second
speaker; errors against the whole-file pass, s4):

| holdS / stableS / keepS | phone | noisy | 2nd speaker | commit delay median / p95 |
|---|---|---|---|---|
| 0.5 / 0.5 / 2 | 5.7% | 7.9% | 2.7% | 0.84 / 1.3-2.0 s |
| 0.8 / 0.5 / 2 | 5.5% | 7.6% | 4.2% | 1.0 / 1.4-2.2 s |
| 1.2 / 0.5 / 2 | 3.9% | 7.9% | 2.1% | 1.36 / 1.5-2.4 s |
| 1.6 / 0.5 / 2 | 3.1% | 6.2% | 1.8% | 1.76 / 1.9-2.5 s |
| 0.8 / 0.25 / 2 | 6.7% | 9.1% | 2.7% | 0.96 / 1.2-1.4 s |
| 0.8 / 1.0 / 2 | 3.3% | 6.0% | 2.4% | 1.3-1.4 / 1.8-4.6 s |
| 0.8 / 0.5 / 1 | 6.7% | 10.1% | 3.6% | 1.0 / 1.4-2.0 s |
| 0.8 / 0.5 / 4 | 3.9% | 6.4% | 4.2% | 1.0 / 1.4-1.5 s |
| **1.2 / 0.5 / 4** | 3.5% | 7.0% | 1.8% | 1.36 / 1.5-1.9 s |

Tentative text is on screen 0.3 s after a word at the 95th percentile either way; the commit delay
is only how long it stays grey.

**Measured** (`web/drive-replay.ts`: the real model in Chrome 154 on diesel2, WebGPU, f16 shaders,
110m Q4_0, cap 12 s as on a phone; audio fed on a virtual clock in which a pass takes 80 ms, so
the numbers do not depend on how busy the machine was, which was very: load 25 from another job).
Reference = one whole-file pass over the same conditioned audio. "wrong" = committed words the
reference does not have there (substituted or extra), "missed" = reference words not committed,
"no text" = longest run of reference words that never had any text, tentative or committed, over
them. Conditions (`web/replay/lib.ts`): `low` -26 dB; `phone` -24 dB, level drifting +-6 dB, phrases
fading 10 dB, room noise at -62 dBFS; `noisy` the same with noise at -54 dBFS; `far` -34 dB, +-8 dB,
12 dB fades, noise at -68 dBFS. The noise is synthetic (pink, 120 Hz hum, slow swell, a knock
every 6 s or so).

| recording, condition | engine | words committed / reference | errors | wrong | missed | no text | commit delay median / p95 | utterances |
|---|---|---|---|---|---|---|---|---|
| interview 90 s, clean | old | 245 / 238 | 8.82% | 19 | 2 | 0 s | 5.66 / 11 s | 16 fragments, 0 dropped |
| interview 90 s, clean | new | 241 / 238 | 5.88% | 10 | 4 | 0 s | 1.36 / 1.6 s | 1 |
| interview 90 s, low | old | 238 / 238 | 5.46% | 8 | 5 | 0 s | 2.3 / 7.46 s | 22 fragments, 2 dropped |
| interview 90 s, low | new | 242 / 238 | 5.88% | 11 | 3 | 0 s | 1.36 / 1.76 s | 1 |
| interview 90 s, phone | old | 192 / 239 | 23.85% | 9 | 48 | 2.8 s | 1.78 / 6.18 s | 21 fragments, 2 dropped |
| interview 90 s, phone | new | 241 / 239 | 4.6% | 8 | 3 | 0 s | 1.36 / 1.84 s | 1 |
| interview 90 s, noisy | old | 149 / 235 | 40% | 8 | 86 | 3.6 s | 1.3 / 3.5 s | 23 fragments, 4 dropped |
| interview 90 s, noisy | new | 236 / 235 | 5.96% | 11 | 3 | 0 s | 1.36 / 1.96 s | 1 |
| interview 90 s, far | old | 89 / 238 | 64.71% | 4 | 150 | 20.72 s | 1.3 / 3.86 s | 12 fragments, 2 dropped |
| interview 90 s, far | new | 240 / 238 | 5.88% | 10 | 4 | 0 s | 1.36 / 1.64 s | 2 |
| the same with 8, 5, 12, 6 s gaps, clean | old | 245 / 240 | 5.42% | 12 | 1 | 0 s | 4.58 / 10.22 s | 15 fragments, 0 dropped |
| the same with 8, 5, 12, 6 s gaps, clean | new | 242 / 240 | 3.33% | 6 | 2 | 0 s | 1.36 / 1.68 s | 4 |
| the same with 8, 5, 12, 6 s gaps, phone | old | 208 / 242 | 19.01% | 10 | 36 | 2.88 s | 2.22 / 6.42 s | 18 fragments, 4 dropped |
| the same with 8, 5, 12, 6 s gaps, phone | new | 243 / 242 | 4.96% | 9 | 3 | 0 s | 1.36 / 1.68 s | 7 |
| the same with 8, 5, 12, 6 s gaps, noisy | old | 155 / 236 | 38.14% | 9 | 81 | 5.6 s | 1.34 / 4.02 s | 22 fragments, 3 dropped |
| the same with 8, 5, 12, 6 s gaps, noisy | new | 239 / 236 | 4.24% | 9 | 1 | 0 s | 1.36 / 2.16 s | 6 |
| the same with 8, 5, 12, 6 s gaps, far | old | 102 / 237 | 60.34% | 7 | 136 | 18.88 s | 1.18 / 3.46 s | 18 fragments, 2 dropped |
| the same with 8, 5, 12, 6 s gaps, far | new | 240 / 237 | 5.49% | 11 | 2 | 0 s | 1.36 / 1.88 s | 6 |
| interview 6.5 min, phone | old | 953 / 1073 | 18.64% | 58 | 142 | 2.88 s | 1.9 / 7.34 s | 94 fragments, 12 dropped |
| interview 6.5 min, phone | new | 1089 / 1073 | 4.38% | 38 | 9 | 0 s | 1.36 / 2.08 s | 2 |
| interview 6.5 min, clean | old | 1100 / 1071 | 5.98% | 60 | 4 | 0 s | 6.86 / 11.16 s | 57 fragments, 0 dropped |
| interview 6.5 min, clean | new | 1093 / 1071 | 4.76% | 46 | 5 | 0 s | 1.36 / 1.76 s | 1 |
| interview 6.5 min, noisy | old | 757 / 1061 | 36.29% | 66 | 319 | 5.92 s | 1.38 / 5.34 s | 101 fragments, 20 dropped |
| interview 6.5 min, noisy | new | 1076 / 1061 | 6.97% | 62 | 12 | 0.16 s | 1.36 / 2.08 s | 3 |
| 2nd speaker 3.9 min, clean | old | 636 / 638 | 1.25% | 4 | 4 | 1.04 s | 2.86 / 7.94 s | 42 fragments, 1 dropped |
| 2nd speaker 3.9 min, clean | new | 640 / 638 | 0.47% | 3 | 0 | 0 s | 1.36 / 1.52 s | 2 |
| 2nd speaker 3.9 min, low | old | 0 / 638 | 100% | 0 | 638 | 23.84 s | - | 0 fragments, 29 dropped |
| 2nd speaker 3.9 min, low | new | 641 / 638 | 0.94% | 6 | 0 | 0 s | 1.36 / 1.56 s | 3 |
| gaps, phone, gate disabled (probes only) | new | 236 / 242 | 7.02% | 7 | 10 | 0.48 s | 1.36 / 2.04 s | 7 |
| gaps, noisy, gate disabled (probes only) | new | 235 / 236 | 4.66% | 8 | 3 | 0.24 s | 1.4 / 2.04 s | 6 |

Old against new on the reproduction case (`iv90`, `phone`): 192 of 239 words, 48 missed, 2.8 s with
no text, 21 fragments, 2 utterances dropped with text on screen -> 241 of 239, 3 missed, none
without text, one utterance. The old logic's errors on clean audio (5-9%) are fragments
transcribed without context; the new engine's (3-6%) are the same kind of difference a different
context gives this model ("percent" / "per cent", "a hundred" / "one hundred and", a number read
two ways), not words lost at the commit point: it commits 0-2% more words than the reference has.

**Room with nobody speaking** (60 s of the synthetic noise alone, both levels): the gate opened 12
times (the knocks), 7 probes ran, 0 words committed.

**6.5 minutes** (`iv6`, replay): buffer 5.8 s median, 6.6 s at the 95th percentile, 10.0 s at most
(cap 12); 1392 passes; WASM heap 46 MB, GPU working memory 41 MiB in 7 buffers and no buffer made
after warm-up, all unchanged from the first minute to the last; JS heap 52-85 MB, sawtooth (the
harness keeps its log there). The same recording in real time through the **page**, Chrome's fake microphone at 44.1 kHz, `phone=1` (`results/live/e-iv6-phone.json`, machine load 10-14): 1428 passes, median 84 ms (95th percentile 135 ms, longest 314 ms) on a median 5.8 s of audio; text trailed speech by 0.1 s at the median and 0.5 s at most; buffer 7.6 s at most; 1090 words, all committed by agreement, 65 of 1071 differing from the whole-file pass; WASM heap 46 MB throughout, GPU working memory 41 MiB with no buffer made after warm-up, GPU process memory 194-203 MiB, renderer RSS 252-313 MiB (298 at the end), JS heap 5-6 MB.

**Slower than real time** (`replay/gate.ts`-style run with the stand-in model, a pass taking a
multiple of its audio's length): 0.3x and 0.5x: everything committed, text 6-7 s behind, buffer at
most 15 s; 0.8x: everything, 12 s behind; 1.5x: 174 of 238 words, 23 s of audio skipped, buffer
never over cap + 30 s. Being 6 s behind at 0.3x is not good: passes over a whole lagging buffer
are long. Transcribing only the tail when behind would fix it; not done.

**WebKit** (Playwright's Linux build, WebKit 27.2, `--browser webkit`): the page loads and the
model runs. `navigator.gpu` is undefined in the worker there, so it runs on the **CPU backend, one
WASM thread**; JSPI is present in that build, so it took the JSPI module, not the ASYNCIFY one.
Audio was fed straight into the engine on the replay page (`window.__pkr.run`): **getUserMedia,
the AudioWorklet and the resampler were not exercised in WebKit.**
Interview 90 s, 110m Q4_0, cap 12 s, new engine (`results/replay/webkit-iv90*.json`):

| run | words committed / reference | errors | wrong | missed | no text | commit delay median / p95 | first text after a word, median / p95 | longest buffer |
|---|---|---|---|---|---|---|---|---|
| Chrome, WebGPU, pass = 80 ms (virtual), clean | 241 / 238 | 5.88% | 10 | 4 | 0 s | 1.36 / 1.6 s | -0.16 / 0.36 s | 7.1 s |
| Chrome, WebGPU, pass = 80 ms (virtual), phone | 241 / 239 | 4.6% | 8 | 3 | 0 s | 1.36 / 1.84 s | -0.12 / 0.36 s | 7.9 s |
| WebKit, CPU, pass = 80 ms (virtual): the engine and the model, not the speed, clean | 241 / 238 | 5.88% | 10 | 4 | 0 s | 1.36 / 1.6 s | -0.16 / 0.36 s | 7.1 s |
| WebKit, CPU, pass = 80 ms (virtual): the engine and the model, not the speed, phone | 240 / 239 | 4.18% | 7 | 3 | 0 s | 1.36 / 1.6 s | -0.12 / 0.36 s | 7.2 s |
| WebKit, CPU, passes at their real speed (1.6-1.9 s for 7.6 s of audio, 4x real time), clean | 244 / 238 | 2.94% | 7 | 0 | 0 s | 5.36 / 8.04 s | 2.68 / 4.8 s | 14.6 s |
| WebKit, CPU, passes at their real speed (1.6-1.9 s for 7.6 s of audio, 4x real time), phone | 243 / 239 | 2.51% | 6 | 0 | 0 s | 4.48 / 7.68 s | 2.24 / 3.44 s | 14 s |

With the clock taken out, WebKit on the CPU gives the same result as Chrome on the GPU (the same words on the clean recording). At its real speed there the text trails by seconds, because a pass takes 1.6-1.9 s: words then commit as "lag" or "cap" (49 and 19 of them) and the buffer reaches 14.6 s, under the 12 s cap + 30 s bound.

**Not tested**: a phone (nothing here ran on one; the phone numbers that started this are the
user's logs); Safari; a real microphone or real room noise (the conditions are arithmetic on two
clean recordings); automatic gain control moving the level; two people talking over each other;
music or a television in the room (the gate will be open all the time: passes run continuously,
which is the design, and what the model writes for that was not looked at); languages other than
English; `s8` and the 0.6b models through the new engine (only `s4` was run); the 24 s desktop cap;
recordings over 6.5 min; the skip path with the real model (stand-in model only); model reload
while recording; the probe on real missed speech (it never fired with the gate as shipped, only
with the gate disabled).

Files: `lib/src/stream.ts` (engine; moved from `web/src/` on 11 Oct, unchanged), `web/replay/old.ts` (the old endpointer), `web/replay/lib.ts`
(conditions, virtual-clock runner, measurements), `web/replay/gate.ts` (bun, stand-in model:
reproduction and guard, exit 1 when the engine loses words), `web/src/replay-page.ts` +
`web/replay.html` + `web/drive-replay.ts` (real model in Chrome or WebKit),
`scripts/replay-clips.py` (the recordings), `results/replay/*.json`.

