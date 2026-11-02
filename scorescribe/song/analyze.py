"""Listen to a recording with open-source tools and write down what's measurable.

Claude can't hear audio, so this stage turns the recording into numbers and text
Claude can reason about:

- Demucs (optional) separates the voice from the accompaniment.
- librosa finds the beats, the tempo, a 12-pitch-class "chroma" profile and the
  bass note for every beat, and the sung pitch (pYIN) which is cut into notes.
- Chord candidates per beat come from chord templates matched against the chroma
  and bass, smoothed with a Viterbi pass so chords don't flicker.
- The key comes from Krumhansl-Schmuckler profiles.
- faster-whisper (optional) writes down the words with timestamps.

Everything is saved to analysis.json (full detail) and analysis.md (compact, for
reading), with times also expressed as "beat numbers" so Claude can line notes,
words and chords up on the beat grid.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
QUALITIES = {  # suffix: intervals
    "": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6], "sus4": [0, 5, 7], "sus2": [0, 2, 7], "aug": [0, 4, 8], "6": [0, 4, 7, 9],
}
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".mp4", ".webm"}


def is_audio(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXT


# ------------------------------------------------------------------ helpers

def _decode(src: str, dst: Path, log):
    if not shutil.which("ffmpeg"):
        if Path(src).suffix.lower() == ".wav":
            shutil.copy(src, dst)
            return
        raise RuntimeError("ffmpeg is needed to read this audio file (Linux: sudo apt install ffmpeg)")
    log("Decoding audio ...")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "2", "-ar", "44100", str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg could not read the file: " + r.stderr.strip()[:300])


def _separate(wav: Path, wd: Path, log) -> tuple[Path, Path, bool]:
    """Returns (vocals, accompaniment, separated?)."""
    try:
        import demucs  # noqa: F401
    except Exception:
        log("  Demucs not installed: analysing the full mix (melody tracking is less reliable). "
            "Install it with: pip install demucs")
        return wav, wav, False
    out = wd / "separated"
    log("Separating vocals from accompaniment with Demucs (a few minutes on a CPU) ...")
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", "cpu",
                        "-o", str(out), str(wav)], capture_output=True, text=True)
    voc = out / "htdemucs" / wav.stem / "vocals.wav"
    acc = out / "htdemucs" / wav.stem / "no_vocals.wav"
    if r.returncode != 0 or not voc.exists():
        log("  Demucs failed, analysing the full mix instead: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200])
        return wav, wav, False
    return voc, acc, True


def split_chord_name(n: str) -> tuple[int, str]:
    """'Ebm7' -> (3, 'm7') for the template names used here."""
    r = n[:2] if len(n) > 1 and n[1] in "#b" else n[:1]
    return PC.index(r), n[len(r):]


def _template_scores(chroma: np.ndarray, bass: np.ndarray | None) -> tuple[list[str], np.ndarray]:
    """Score every chord template for every beat. chroma: (12, n). Returns names, (n_chords, n)."""
    names, mats = [], []
    for root in range(12):
        for q, ivs in QUALITIES.items():
            t = np.zeros(12)
            for i, iv in enumerate(ivs):
                t[(root + iv) % 12] = 1.0 if i < 3 else 0.7
            names.append(PC[root] + q)
            mats.append(t / np.linalg.norm(t))
    T = np.array(mats)                       # (k, 12)
    c = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    S = T @ c                                # cosine similarity (k, n)
    # simpler chords are more likely; 4-note templates must earn their extra note
    penalty = {"": 0.0, "m": 0.0, "7": 0.04, "maj7": 0.05, "m7": 0.04, "6": 0.05,
               "sus4": 0.05, "sus2": 0.05, "dim": 0.06, "aug": 0.08}
    parsed = [split_chord_name(n) for n in names]
    S = S - np.array([penalty[q] for _, q in parsed])[:, None]
    if bass is not None:
        for i, (root, q) in enumerate(parsed):
            tones = {(root + iv) % 12 for iv in QUALITIES[q]}
            S[i] += np.where(bass == root, 0.06, np.where(np.isin(bass, list(tones)), 0.02, -0.04))
    return names, S


def _viterbi_chords(names: list[str], S: np.ndarray, stay: float = 0.82) -> list[int]:
    import librosa
    k, n = S.shape
    P = np.exp((S - S.max(axis=0, keepdims=True)) * 25.0)
    P = P / P.sum(axis=0, keepdims=True)
    trans = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(trans, stay)
    return list(librosa.sequence.viterbi(P, trans))


def _key(chroma_total: np.ndarray) -> list[dict]:
    out = []
    for root in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(np.roll(prof, root), chroma_total)[0, 1]
            out.append({"key": f"{PC[root]} {mode}", "score": round(float(r), 3)})
    return sorted(out, key=lambda d: -d["score"])[:3]


def _notes_from_f0(times, midi, voiced, prob, onsets, min_dur=0.09) -> list[dict]:
    """Cut a pitch track into notes: voiced runs, split at pitch jumps and at onsets."""
    notes = []
    n = len(midi)
    i = 0
    onset_set = set(onsets)
    while i < n:
        if not voiced[i] or not np.isfinite(midi[i]):
            i += 1
            continue
        j = i
        seg = [i]
        while j + 1 < n and voiced[j + 1] and np.isfinite(midi[j + 1]):
            cur = np.median(midi[seg[-5:]] if len(seg) >= 5 else midi[seg])
            jump = abs(midi[j + 1] - cur) > 0.7 and (j + 3 < n and all(
                voiced[j + k] and np.isfinite(midi[j + k]) and abs(midi[j + k] - cur) > 0.7 for k in (1, 2, 3)))
            if jump or (j + 1) in onset_set and len(seg) > 4:
                break
            j += 1
            seg.append(j)
        t0, t1 = float(times[i]), float(times[j]) + float(times[1] - times[0])
        if t1 - t0 >= min_dur:
            m = float(np.median(midi[i:j + 1]))
            notes.append({"start": round(t0, 3), "end": round(t1, 3), "midi": round(m, 2),
                          "conf": round(float(np.mean(prob[i:j + 1])), 2)})
        i = j + 1
    # merge tiny gaps between equal pitches (vibrato dropouts)
    merged = []
    for nt in notes:
        if merged and abs(merged[-1]["midi"] - nt["midi"]) < 0.5 and nt["start"] - merged[-1]["end"] < 0.025:
            merged[-1]["end"] = nt["end"]
        else:
            merged.append(nt)
    return merged


def midi_name(m: float) -> str:
    r = int(round(m))
    return f"{PC[r % 12]}{r // 12 - 1}"


def _words(vocals: Path, log, model_size: str, language: str | None) -> tuple[list[dict], str | None]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        log("  faster-whisper not installed: no automatic lyrics (pip install faster-whisper). "
            "You can still type the lyrics in.")
        return [], None
    log(f"Transcribing the words with Whisper ({model_size}) ...")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segs, info = model.transcribe(str(vocals), word_timestamps=True, vad_filter=True, language=language,
                                      condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in s.words or []:
                words.append({"word": w.word.strip(), "start": round(float(w.start), 2),
                              "end": round(float(w.end), 2), "prob": round(float(w.probability), 2)})
        return words, getattr(info, "language", None)
    except Exception as e:
        log(f"  Whisper failed ({str(e)[:160]}); continuing without lyrics")
        return [], None


# ------------------------------------------------------------------ main

def analyze(src: str, workdir: str, separate: bool = True, lyrics: bool = True, whisper_model: str = "small",
            language: str | None = None, lyrics_text: str | None = None, log=print) -> dict:
    try:
        import librosa
    except ImportError:
        raise RuntimeError('Recording support isn\'t installed. On the server run: pip install -e ".[audio]" '
                           "(see 'Songs from recordings' in the README)")

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    wav = wd / "audio.wav"
    _decode(src, wav, log)
    vocals, accomp, separated = (_separate(wav, wd, log) if separate else (wav, wav, False))

    sr = 22050
    log("Finding the beat ...")
    y_mix, _ = librosa.load(str(wav), sr=sr, mono=True)
    duration = len(y_mix) / sr
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y_mix, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    if len(beat_times) < 8:
        raise RuntimeError("couldn't find a steady beat in this recording")
    accents = [round(float(onset_env[b]), 2) for b in beats]
    acc_max = max(accents) or 1.0
    accents = [round(a / acc_max, 2) for a in accents]

    log("Measuring harmony on every beat ...")
    y_acc, _ = librosa.load(str(accomp), sr=sr, mono=True)
    if not separated:
        y_acc = librosa.effects.harmonic(y_acc)
    chroma = librosa.feature.chroma_cqt(y=y_acc, sr=sr, hop_length=hop, bins_per_octave=36)
    C = librosa.cqt(y_acc, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"), n_bins=36, bins_per_octave=12)
    bass_energy = np.abs(C)
    bounds = list(beats) + [chroma.shape[1]]
    bchroma = np.stack([chroma[:, bounds[k]:max(bounds[k] + 1, bounds[k + 1])].mean(axis=1)
                        for k in range(len(beats))], axis=1)
    bbass = np.stack([bass_energy[:, bounds[k]:max(bounds[k] + 1, bounds[k + 1])].mean(axis=1)
                      for k in range(len(beats))], axis=1)
    bass_pc = np.array([int(np.argmax(bbass[:, k])) % 12 for k in range(bbass.shape[1])])
    bass_strength = bbass.max(axis=0) / (bbass.max() + 1e-9)
    names, S = _template_scores(bchroma, bass_pc)
    path = _viterbi_chords(names, S)
    keys = _key(chroma.sum(axis=1))

    log("Tracking the sung melody ...")
    y_voc, _ = librosa.load(str(vocals), sr=sr, mono=True)
    f0, vflag, vprob = librosa.pyin(y_voc, fmin=65.0, fmax=1050.0, sr=sr, frame_length=2048, hop_length=256)
    ftimes = librosa.times_like(f0, sr=sr, hop_length=256)
    midi = librosa.hz_to_midi(f0)
    voc_on = librosa.onset.onset_detect(y=y_voc, sr=sr, hop_length=256, units="frames", backtrack=True)
    notes = _notes_from_f0(ftimes, midi, vflag, vprob, list(voc_on))

    def beat_pos(t: float) -> float:
        bt = beat_times
        if t <= bt[0]:
            return round(float((t - bt[0]) / (60.0 / tempo)), 2)
        if t >= bt[-1]:
            return round(float(len(bt) - 1 + (t - bt[-1]) / (60.0 / tempo)), 2)
        return round(float(np.interp(t, bt, np.arange(len(bt)))), 2)

    for nt in notes:
        nt["name"] = midi_name(nt["midi"])
        nt["beat"] = beat_pos(nt["start"])
        nt["beats_long"] = round(beat_pos(nt["end"]) - nt["beat"], 2)

    words, lang = ([], None)
    if lyrics and not lyrics_text:
        words, lang = _words(vocals, log, whisper_model, language)
    for w in words:
        w["beat"] = beat_pos(w["start"])

    beats_out = []
    for k, t in enumerate(beat_times):
        col = S[:, k]
        top = np.argsort(-col)[:3]
        cvec = bchroma[:, k] / (bchroma[:, k].max() + 1e-9)
        beats_out.append({
            "i": k, "t": round(float(t), 2), "accent": accents[k],
            "bass": PC[bass_pc[k]], "bass_strength": round(float(bass_strength[k]), 2),
            "chroma": [round(float(x), 2) for x in cvec],
            "candidates": [{"chord": names[j], "score": round(float(col[j]), 3)} for j in top],
            "smoothed": names[path[k]],
        })
    result = {
        "source": Path(src).name, "duration": round(duration, 1), "tempo": round(tempo, 1),
        "separated": separated, "keys": keys, "beats": beats_out, "notes": notes, "words": words,
        "language": lang, "lyrics_text": lyrics_text or None,
        "pitch_track": {"hop_s": round(256 / sr, 5), "t0": float(ftimes[0]),
                        "midi": [None if not (v and np.isfinite(m)) else round(float(m), 2) for m, v in zip(midi, vflag)]},
    }
    (wd / "analysis.json").write_text(json.dumps(result))
    (wd / "analysis.md").write_text(analysis_markdown(result), encoding="utf-8")
    log(f"  {len(beat_times)} beats at ~{tempo:.0f} BPM, {len(notes)} sung notes, {len(words)} words; "
        f"likely key {keys[0]['key']}")
    return result


def analysis_markdown(a: dict) -> str:
    lines = [f"# Analysis of {a['source']}", "",
             f"- Duration {a['duration']} s, tempo about {a['tempo']} beats per minute "
             f"(the beat tracker can lock onto half or double the real tempo; judge from the melody and words).",
             f"- Voice separated from accompaniment: {'yes' if a['separated'] else 'NO - full mix, melody less reliable'}.",
             "- Likely keys: " + ", ".join(f"{k['key']} ({k['score']})" for k in a["keys"]), ""]
    if a.get("lyrics_text"):
        lines += ["## Lyrics supplied by the user", "", a["lyrics_text"], ""]
    if a["words"]:
        lines += ["## Words heard (Whisper; beat number in brackets, may contain mishearings)", "",
                  " ".join(f"{w['word']}[{w['beat']}]" for w in a["words"]), ""]
    lines += ["## Beat by beat", "",
              "Columns: beat number | time s | accent (onset strength 0-1; downbeats are often stronger) | bass | "
              "chord candidates (template fit, higher is better) | smoothed chord | strongest pitch classes | "
              "sung notes starting in this beat (pitch, position within the beat, length in beats)", ""]
    by_beat: dict[int, list[dict]] = {}
    for nt in a["notes"]:
        by_beat.setdefault(int(np.floor(nt["beat"])), []).append(nt)
    for b in a["beats"]:
        top = [PC[i] for i in np.argsort(b["chroma"])[::-1][:4] if b["chroma"][i] > 0.35]
        notes = by_beat.get(b["i"], [])
        ns = " ".join(f"{n['name']}@{n['beat'] - b['i']:.2f}~{n['beats_long']:.2f}" + ("?" if n["conf"] < 0.4 else "")
                      for n in notes)
        cands = ", ".join(f"{c['chord']} {c['score']:.2f}" for c in b["candidates"])
        lines.append(f"{b['i']} | {b['t']} | {b['accent']} | {b['bass']} | {cands} | {b['smoothed']} | "
                     f"{' '.join(top)} | {ns}")
    early = [n for n in a["notes"] if n["beat"] < 0]
    if early:
        lines += ["", "Notes before the first tracked beat: " + " ".join(
            f"{n['name']}@{n['beat']}~{n['beats_long']}" for n in early)]
    return "\n".join(lines) + "\n"
