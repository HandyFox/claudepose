"""Synthesize a short song with a known lead sheet, for testing the recording pipeline.

Writes OUT/vocals.wav, OUT/no_vocals.wav, OUT/mix.wav and OUT/truth.json.
The "voice" is a vibrato sawtooth-ish tone; the band is a pad + bass + kick/hat.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
SR = 22050
BPM = 96.0
BEAT = 60.0 / BPM
LEAD_IN = 1.0  # seconds of band before bar 1 (one bar count-in played by drums only)

# key G major, 4/4. Melody tokens in ScoreScribe notation (written = sounding pitch).
truth = {
    "title": "Test Song", "key": 1, "time": "4/4", "tempo": 96,
    "measures": [
        {"chords": [{"beat": 1, "symbol": "G"}], "melody": "B4/4[lyric=Sing] B4/4[lyric=a] D5/4[lyric=sim-] B4/4[lyric=ple]"},
        {"chords": [{"beat": 1, "symbol": "Em"}], "melody": "G4/2[lyric=song] E4/2[lyric=now]"},
        {"chords": [{"beat": 1, "symbol": "C"}], "melody": "E4/4[lyric=Light] G4/4[lyric=the] C5/4[lyric=morn-] E5/4[lyric=ing]"},
        {"chords": [{"beat": 1, "symbol": "D"}], "melody": "D5/2.[lyric=sky] r/4"},
        {"chords": [{"beat": 1, "symbol": "G"}], "melody": "D5/4[lyric=Hold] B4/4[lyric=me] G4/4[lyric=near] B4/4[lyric=and]"},
        {"chords": [{"beat": 1, "symbol": "C"}], "melody": "C5/4[lyric=hum] E5/4[lyric=a] G4/4[lyric=lit-] E4/4[lyric=tle]"},
        {"chords": [{"beat": 1, "symbol": "D"}, {"beat": 3, "symbol": "D7"}], "melody": "F#4/2[lyric=while] A4/2[lyric=we]"},
        {"chords": [{"beat": 1, "symbol": "G"}], "melody": "G4/1[lyric=go]"},
    ],
}
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
CHORD = {"G": [55, 59, 62], "Em": [52, 55, 59], "C": [48, 52, 55], "D": [50, 54, 57], "D7": [50, 54, 57, 60]}
BASS = {"G": 43, "Em": 40, "C": 36, "D": 38, "D7": 38}


def midi_of(tok):
    p = tok.split("/")[0]
    step, rest = p[0], p[1:]
    acc = 1 if rest.startswith("#") else -1 if rest.startswith("b") else 0
    octv = int(rest.lstrip("#b"))
    return 12 * (octv + 1) + NOTE[step] + (1 if step == "F" and acc == 0 else acc) if False else 12 * (octv + 1) + NOTE[step] + acc


DUR = {"1": 4, "2": 2, "4": 1, "8": 0.5}
total_beats = 4 * len(truth["measures"])
n = int(SR * (LEAD_IN * 2 + total_beats * BEAT + 1.5))
voc = np.zeros(n)
band = np.zeros(n)
t_all = np.arange(n) / SR


def tone(f, start, dur, amp, harmonics=(1, .5, .33, .25, .2), vib=0.0, att=0.02, rel=0.08):
    i0, i1 = int(start * SR), int((start + dur) * SR)
    t = np.arange(i1 - i0) / SR
    ph = 2 * np.pi * f * t
    if vib:
        ph += (vib / 5.5) * np.sin(2 * np.pi * 5.5 * t) * 2 * np.pi * f / f  # gentle vibrato
    s = sum(a * np.sin(k * ph) for k, a in enumerate(harmonics, start=1))
    env = np.minimum(1, t / att) * np.minimum(1, (dur - t) / rel).clip(0)
    return i0, s * env * amp


start = LEAD_IN * 2  # two bars of count-in? keep one bar of drums + one bar pickup space
# drums from the very start (count-in) to the end
for b in range(int((start + total_beats * BEAT) / BEAT) + 1):
    t0 = start - 4 * BEAT + b * BEAT
    if t0 < 0:
        continue
    i0 = int(t0 * SR)
    k = np.exp(-np.arange(int(0.12 * SR)) / (0.02 * SR)) * np.sin(2 * np.pi * 60 * np.arange(int(0.12 * SR)) / SR)
    amp = 0.9 if (b % 4) == 0 else 0.5
    band[i0:i0 + len(k)] += amp * k
beat = 0.0
for m in truth["measures"]:
    syms = m["chords"]
    for ci, c in enumerate(syms):
        b0 = (c["beat"] - 1)
        b1 = syms[ci + 1]["beat"] - 1 if ci + 1 < len(syms) else 4
        for bb in range(int(b0), int(b1)):
            t0 = start + (beat + bb) * BEAT
            for p in CHORD[c["symbol"]]:
                i0, s = tone(440 * 2 ** ((p - 69) / 12), t0, BEAT * 0.95, 0.10)
                band[i0:i0 + len(s)] += s
            i0, s = tone(440 * 2 ** ((BASS[c["symbol"]] - 69) / 12), t0, BEAT * 0.9, 0.22, harmonics=(1, .6, .3))
            band[i0:i0 + len(s)] += s
    pos = 0.0
    for tok in m["melody"].split():
        base = tok.split("[")[0]
        d = base.split("/")[1]
        q = DUR[d.rstrip(".")] * (1.5 if d.endswith(".") else 1)
        if not base.startswith("r"):
            f = 440 * 2 ** ((midi_of(base) - 69) / 12)
            i0, s = tone(f, start + (beat + pos) * BEAT, q * BEAT * 0.92, 0.35,
                         harmonics=(1, .7, .45, .3, .2, .12), vib=0.3)
            voc[i0:i0 + len(s)] += s
        pos += q
    beat += 4

rng = np.random.default_rng(1)
voc += rng.normal(0, 0.002, n)
band += rng.normal(0, 0.002, n)
mix = voc + band
for name, sig in (("vocals", voc), ("no_vocals", band), ("mix", mix)):
    sf.write(os.path.join(out, f"{name}.wav"), (sig / (np.abs(mix).max() + 1e-9) * 0.9).astype(np.float32), SR)
# what Whisper would hear: the words with times, plus two deliberate mishearings
words = []
beat = 0.0
for m in truth["measures"]:
    pos = 0.0
    for tok in m["melody"].split():
        base = tok.split("[")[0]
        d = base.split("/")[1]
        q = DUR[d.rstrip(".")] * (1.5 if d.endswith(".") else 1)
        if "lyric=" in tok:
            syl = tok.split("lyric=")[1].rstrip("]")
            t0 = start + (beat + pos) * BEAT
            if syl.endswith("-"):
                words.append({"word": syl[:-1], "start": round(t0, 2), "end": round(t0 + q * BEAT, 2), "cont": True})
            elif words and words[-1].get("cont"):
                words[-1]["word"] += syl
                words[-1]["end"] = round(t0 + q * BEAT, 2)
                words[-1]["cont"] = False
            else:
                words.append({"word": syl, "start": round(t0, 2), "end": round(t0 + q * BEAT, 2)})
        pos += q
    beat += 4
for w in words:
    w.pop("cont", None)
    w["prob"] = 0.9
words[0]["word"] = "Sing,"
words[11]["word"] = "Bold"          # "Hold" misheard
words[13]["word"] = "beer"          # "near" misheard
truth["first_beat_time"] = start
json.dump(truth, open(os.path.join(out, "truth.json"), "w"), indent=1)
json.dump(words, open(os.path.join(out, "whisper_words.json"), "w"), indent=1)
print("wrote", out, f"{n / SR:.1f}s", len(words), "words")
