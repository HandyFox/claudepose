"""Lead sheets from recordings: format, checks against the audio analysis, MusicXML.

A lead sheet (leadsheet.json) is what Claude writes after studying analysis.md:

    {"title": "...", "artist": "...", "key": 1, "mode": "major", "time": "4/4", "tempo": 96,
     "clef": "G2", "beat_ql": 1.0, "downbeat_beat": 3,
     "measures": [{"section": "Verse 1", "chords": [{"beat": 1, "symbol": "G"}, {"beat": 3, "symbol": "D/F#"}],
                   "melody": "B4/4[lyric=Sing] B4/4[lyric=a] D5/4[lyric=sim-] B4/4[lyric=ple]"}, ...]}

- melody uses ScoreScribe's token notation (see the scan prompts); lyrics ride on the notes.
- chords: "beat" counts quarter notes from 1 (beat 2.5 = the "and" of 2); symbols like C, Am7, F#m7b5,
  Bb/D, Gsus4, N.C.
- downbeat_beat: the analysis beat number where beat 1 of the first FULL measure falls. A first
  measure marked "incomplete": true is a pickup that ends there.
- beat_ql: quarter notes per tracked beat (1.0 normally; 0.5 if the tracker counted eighths,
  2.0 if it counted half notes; 1.5 for 6/8 tracked in dotted quarters).
"""
from __future__ import annotations

import difflib
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..model import assemble
from .analyze import PC, QUALITIES

FLAT_ROOT = re.compile(r"^([A-G])b")


def m21_figure(sym: str) -> str:
    s = sym.strip()
    s = FLAT_ROOT.sub(r"\1-", s)
    s = re.sub(r"/([A-G])b", r"/\1-", s)
    return s


def parse_chord(sym: str):
    """music21 ChordSymbol / NoChord, or raises ValueError."""
    from music21 import harmony
    s = (sym or "").strip()
    if s.upper() in ("N.C.", "NC", "N.C", "N"):
        return harmony.NoChord()
    try:
        h = harmony.ChordSymbol(m21_figure(s))
        if not h.pitches:
            raise ValueError("no pitches")
        return h
    except Exception as e:
        raise ValueError(f"can't read chord symbol '{sym}' ({str(e)[:80]})")


def chord_pcs(sym: str) -> tuple[set[int], int | None, int | None]:
    """(pitch classes, root pc, bass pc) for a chord symbol; empty for N.C."""
    from music21 import harmony
    h = parse_chord(sym)
    if isinstance(h, harmony.NoChord):
        return set(), None, None
    return {p.pitchClass for p in h.pitches}, h.root().pitchClass, h.bass().pitchClass


def to_systems(ls: dict) -> list[dict]:
    measures = []
    for i, m in enumerate(ls.get("measures") or []):
        md = {"staves": [{"staff": 1, "voices": [m.get("melody") or "R"]}]}
        if i == 0:
            md["time"] = ls.get("time") or "4/4"
            md["key"] = ls.get("key", 0)
            md["staves"][0]["clef"] = ls.get("clef") or "G2"
            if ls.get("tempo"):
                md["tempo"] = {"bpm": ls["tempo"]}
        for k in ("time", "key", "incomplete", "barline", "repeat_start", "volta"):
            if k in m and (i > 0 or k in ("incomplete", "barline", "repeat_start", "volta")):
                md[k] = m[k]
        if m.get("section"):
            md["staves"][0]["texts"] = [m["section"]]
        measures.append(md)
    return [{"measures": measures}]


def _doc(ls: dict):
    return assemble(to_systems(ls), [{"name": ls.get("part_name") or "Voice", "staves": 1}],
                    ls.get("title"), ls.get("artist"), stitch=False)


def build_score(ls: dict):
    from music21 import harmony, metadata
    from ..build import build_score as _bs
    doc = _doc(ls)
    score = _bs(doc)
    part = score.parts[0]
    ms = list(part.getElementsByClass("Measure"))
    for m21m, md in zip(ms, ls.get("measures") or []):
        for c in md.get("chords") or []:
            try:
                h = parse_chord(str(c.get("symbol", "")))
            except ValueError:
                continue
            off = max(0.0, float(c.get("beat", 1)) - 1.0)
            if isinstance(h, harmony.ChordSymbol):
                h.writeAsChord = False
            m21m.insert(off, h)
    if ls.get("artist"):
        score.metadata.composer = ls["artist"]
    if ls.get("title"):
        score.metadata.title = ls["title"]
    elif score.metadata is None:
        score.insert(0, metadata.Metadata())
    return doc, score


def write(ls: dict, out_path: str, shape_notes: str | None = None) -> str:
    from ..build import tidy_musicxml
    _, score = build_score(ls)
    out = str(score.write("musicxml", fp=out_path))
    try:
        tidy_musicxml(out)
    except Exception:
        pass
    if shape_notes:
        from ..shapes import apply_shape_notes
        apply_shape_notes(out, shape_notes)
    return out


# ------------------------------------------------------------------ timing

def _measure_spans(doc, ls: dict) -> list[tuple[float, float]]:
    """(start, length) of every measure in analysis-beat units."""
    bq = float(ls.get("beat_ql") or 1.0)
    db = float(ls.get("downbeat_beat") or 0)
    spans = []
    pos = None
    for i, M in enumerate(doc.measures):
        length = float(M.length) / bq
        if i == 0:
            if M.incomplete:
                ql = _first_voice_len(M)
                pos = db - ql / bq
                length = ql / bq
            else:
                pos = db
        elif M.incomplete:
            length = _first_voice_len(M) / bq
        spans.append((pos, length))
        pos += length
    return spans


def _first_voice_len(M) -> float:
    st = M.staves.get(1)
    if not st or not st.voices:
        return float(M.length)
    return float(sum((M.length if e.kind == "mrest" else e.ql) for e in st.voices[0]))


def _beat_time(analysis: dict, b: float) -> float:
    bt = np.array([x["t"] for x in analysis["beats"]])
    per = 60.0 / analysis["tempo"]
    if b <= 0:
        return float(bt[0] + b * per)
    if b >= len(bt) - 1:
        return float(bt[-1] + (b - len(bt) + 1) * per)
    return float(np.interp(b, np.arange(len(bt)), bt))


def _melody_events(score):
    """(measure index, offset ql, length ql, midi or None, lyric) for the melody."""
    from music21 import harmony
    out = []
    part = score.parts[0]
    for mi, m in enumerate(part.getElementsByClass("Measure")):
        for n in m.recurse().notesAndRests:
            if isinstance(n, harmony.Harmony):  # chord symbols are Chords to music21; not melody
                continue
            if n.duration.isGrace or (n.isRest and n.style.hideObjectOnPrint):
                continue
            midi = None if n.isRest else (n.pitches[-1].midi)
            lyr = None
            if not n.isRest and n.lyrics:
                ly = n.lyrics[0]
                lyr = (ly.text or "") + ("-" if ly.syllabic in ("begin", "middle") else "")
            out.append((mi, float(n.getOffsetInHierarchy(m)), float(n.quarterLength), midi, lyr))
    return out


# ------------------------------------------------------------------ checks

def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def _lyric_words(events) -> list[str]:
    words, cur = [], ""
    for _, _, _, midi, lyr in events:
        if not lyr:
            continue
        t = lyr.strip()
        if t.endswith("-"):
            cur += t[:-1]
        else:
            words.append(cur + t)
            cur = ""
    if cur:
        words.append(cur)
    return words


def _key_pcs(fifths: int, mode: str) -> set[int]:
    tonic = (7 * fifths) % 12  # major tonic
    return {(tonic + i) % 12 for i in (0, 2, 4, 5, 7, 9, 11)}


def check(ls: dict, analysis: dict) -> dict:
    """Compare a lead sheet with the analysis. Returns {"errors", "checks", "notes", "stats", "text"}."""
    errors, checks, notes = [], [], []
    if not ls.get("measures"):
        return {"errors": ["no measures"], "checks": [], "notes": [], "stats": {}, "text": "ERROR: no measures"}
    if ls.get("downbeat_beat") is None:
        errors.append("set `downbeat_beat` (the analysis beat number of beat 1 of the first full measure)")
    doc = _doc(ls)
    for i in doc.issues:
        if i.level == "error":
            errors.append(f"measure {i.measure}: {i.msg}")
        elif i.level == "warn" and "clef" not in i.msg:
            checks.append(f"measure {i.measure}: {i.msg}")
    chord_list = []  # (measure no, beat, symbol, start, end) in analysis beats
    spans = _measure_spans(doc, ls)
    bq = float(ls.get("beat_ql") or 1.0)
    for mi, (md, (start, length)) in enumerate(zip(ls["measures"], spans), start=1):
        cs = sorted(md.get("chords") or [], key=lambda c: float(c.get("beat", 1)))
        mlen_ql = length * bq
        for k, c in enumerate(cs):
            sym = str(c.get("symbol", ""))
            try:
                parse_chord(sym)
            except ValueError as e:
                errors.append(f"measure {mi}: {e}")
                continue
            b = float(c.get("beat", 1))
            if b < 1 or b - 1 >= mlen_ql + 1e-6:
                errors.append(f"measure {mi}: chord {sym} at beat {b} is outside the measure")
                continue
            c_start = start + (b - 1) / bq
            chord_list.append([mi, b, sym, c_start, None])
    for k, c in enumerate(chord_list):
        nxt = chord_list[k + 1][3] if k + 1 < len(chord_list) else spans[-1][0] + spans[-1][1]
        c[4] = nxt

    # ---- harmony vs audio
    beats = analysis["beats"]
    from .analyze import _template_scores
    chord_hits = 0
    fifths = int(ls.get("key", 0) or 0)
    kp = _key_pcs(fifths, ls.get("mode", "major"))
    for mi, b, sym, s, e in chord_list:
        idx = [i for i in range(int(np.floor(s + 0.01)), int(np.ceil(e - 0.01))) if 0 <= i < len(beats)]
        if not idx:
            continue
        tones, root, bass = chord_pcs(sym)
        if not tones:
            continue
        ch = np.mean([beats[i]["chroma"] for i in idx], axis=0)
        bass_votes = {}
        for i in idx:
            if beats[i]["bass_strength"] > 0.15:
                bass_votes[beats[i]["bass"]] = bass_votes.get(beats[i]["bass"], 0) + 1
        heard_bass = max(bass_votes, key=bass_votes.get) if bass_votes else None
        hb = PC.index(heard_bass) if heard_bass else None
        names, S = _template_scores(ch[:, None], np.array([hb]) if hb is not None else None)
        order = np.argsort(-S[:, 0])
        best = names[order[0]]
        # score of the written chord with the same scoring (its closest template)
        t = np.zeros(12)
        for pc in tones:
            t[pc] = 1.0
        mine = float((t / np.linalg.norm(t)) @ (ch / (np.linalg.norm(ch) + 1e-9)))
        best_score = float(S[order[0], 0])
        from .analyze import split_chord_name
        broot, bq_ = split_chord_name(best)
        best_tones = {(broot + iv) % 12 for iv in QUALITIES[bq_]}
        where = f"measure {mi} beat {b:g}"
        if best_tones != tones and best_score - mine > 0.07:
            alt = ", ".join(f"{names[j]} {S[j, 0]:.2f}" for j in order[:3])
            checks.append(f"{where}: {sym} fits the audio less well ({mine:.2f}) than {alt}. Bass heard: {heard_bass or '?'}")
        else:
            chord_hits += 1
        if hb is not None and bass is not None and hb != bass and len(idx) >= 1:
            if hb in tones:
                checks.append(f"{where}: bass heard {heard_bass}, a chord tone of {sym} - an inversion ({sym}/{heard_bass})?")
            else:
                checks.append(f"{where}: bass heard {heard_bass} is not in {sym} - wrong root, or a passing bass note?")
        outside = sorted(PC[p] for p in tones - kp)
        if outside:
            notes.append(f"{where}: {sym} has {', '.join(outside)}, outside the key (borrowed/secondary chord - fine if the audio agrees)")

    # ---- melody vs audio
    try:
        _, score = build_score(ls)
        events = _melody_events(score)
    except Exception as e:
        errors.append(f"could not build the melody: {str(e)[:150]}")
        events = []
    pt = analysis.get("pitch_track") or {}
    track = np.array([np.nan if v is None else v for v in pt.get("midi", [])], dtype=float)
    hop, t0 = pt.get("hop_s", 0.0116), pt.get("t0", 0.0)
    ok = total = 0
    for mi, off, ql, midi, lyr in events:
        s_beat = spans[mi][0] + off / bq
        e_beat = s_beat + ql / bq
        ts, te = _beat_time(analysis, s_beat), _beat_time(analysis, e_beat)
        i0, i1 = int((ts - t0) / hop), int((te - t0) / hop)
        seg = track[max(0, i0 + 2):max(i0 + 3, i1 - 2)]
        voiced = seg[np.isfinite(seg)]
        frac = len(voiced) / max(1, len(seg))
        where = f"measure {mi + 1} beat {off + 1:g}"
        if midi is None:
            if ql / bq >= 0.75 and frac > 0.6:
                sung = int(round(float(np.median(voiced))))
                checks.append(f"{where}: rest, but singing was measured there (around {PC[sung % 12]}{sung // 12 - 1})")
            continue
        total += 1
        if frac < 0.25:
            if ql / bq >= 0.5:
                checks.append(f"{where}: note {_name(midi)} but almost no singing was measured there (rest? timing?)")
            continue
        heard = float(np.median(voiced))
        d = heard - midi
        if abs(d) <= 0.75:
            ok += 1
        elif abs(abs(d) - 12) <= 0.75 or abs(abs(d) - 24) <= 0.75:
            checks.append(f"{where}: {_name(midi)} is an octave off from what was sung ({_name(heard)})")
        else:
            checks.append(f"{where}: wrote {_name(midi)}, measured {_name(heard)} ({d:+.1f} semitones)")
    # melody notes on chord changes / strong beats that aren't chord tones (informational)
    for mi, off, ql, midi, lyr in events:
        if midi is None or ql < 1.0:
            continue
        pos = spans[mi][0] + off / bq
        cur = next((c for c in chord_list if c[3] - 1e-6 <= pos < c[4] - 1e-6), None)
        if cur and abs(off) < 1e-6:
            tones, _, _ = chord_pcs(cur[2])
            if tones and midi % 12 not in tones:
                notes.append(f"measure {mi + 1} beat 1: melody {_name(midi)} is not in {cur[2]} (suspension/appoggiatura, or a wrong chord/note?)")

    # ---- lyrics vs what Whisper heard
    heard_words = [_norm_word(w["word"]) for w in analysis.get("words") or [] if _norm_word(w["word"])]
    written = [_norm_word(w) for w in _lyric_words(events) if _norm_word(w)]
    lyr_stats = None
    if heard_words:
        sm = difflib.SequenceMatcher(a=written, b=heard_words, autojunk=False)
        lyr_stats = round(sm.ratio(), 2)
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op == "equal":
                continue
            wa, wb = " ".join(written[a0:a1]), " ".join(heard_words[b0:b1])
            if op == "replace":
                notes.append(f"lyrics: you wrote \"{wa}\" where Whisper heard \"{wb}\" (Whisper often mishears; keep yours if it makes more sense)")
            elif op == "delete":
                notes.append(f"lyrics: \"{wa}\" not heard by Whisper")
            elif op == "insert":
                checks.append(f"lyrics: Whisper heard \"{wb}\" that isn't in your lyrics")
    stats = {"chords_fitting": f"{chord_hits}/{len(chord_list)}", "melody_matching": f"{ok}/{total}",
             "lyrics_similarity": lyr_stats, "measures": len(ls["measures"])}
    lines = ["Lead sheet check", "",
             f"Chords that fit the audio: {stats['chords_fitting']}  |  melody notes matching the sung pitch: "
             f"{stats['melody_matching']}" + (f"  |  lyrics vs Whisper: {lyr_stats:.0%} similar" if lyr_stats is not None else ""),
             ""]
    for title, items in (("ERRORS (must fix)", errors), ("CHECK (look at these in analysis.md and decide)", checks),
                         ("NOTES (for your judgement)", notes)):
        lines.append(f"{title}: {len(items)}")
        lines += [f"  - {x}" for x in items]
        lines.append("")
    return {"errors": errors, "checks": checks, "notes": notes, "stats": stats, "text": "\n".join(lines)}


def _name(m: float) -> str:
    r = int(round(m))
    return f"{PC[r % 12]}{r // 12 - 1}"


# ------------------------------------------------------------------ chord chart

def chord_chart(ls: dict, per_line: int = 4) -> str:
    try:
        _, score = build_score(ls)
        events = _melody_events(score)
    except Exception:
        events = []
    by_m: dict[int, list] = {}
    for mi, off, ql, midi, lyr in events:
        if lyr:
            by_m.setdefault(mi, []).append((off, lyr.strip()))
    out = []
    head = ls.get("title") or "Untitled"
    if ls.get("artist"):
        head += f" - {ls['artist']}"
    key = ls.get("key", 0)
    out += [head, f"Key: {_key_name(int(key or 0), ls.get('mode', 'major'))}   Time: {ls.get('time', '4/4')}   "
                  f"Tempo: {ls.get('tempo', '?')}", ""]
    ms = ls.get("measures") or []
    i = 0
    while i < len(ms):
        group = []
        j = i
        while j < len(ms) and len(group) < per_line:
            if j > i and ms[j].get("section"):
                break
            group.append(j)
            j += 1
        if ms[i].get("section"):
            out.append(f"[{ms[i]['section']}]")
        chord_line, lyric_line = "", ""
        for mi in group:
            sylls = by_m.get(mi, [])
            chords = sorted(ms[mi].get("chords") or [], key=lambda c: float(c.get("beat", 1)))
            ci = 0
            for off, syl in sylls:
                while ci < len(chords) and float(chords[ci].get("beat", 1)) - 1 <= off + 1e-6:
                    col = len(lyric_line)
                    if len(chord_line) > col:
                        lyric_line += " " * (len(chord_line) - col)
                        col = len(lyric_line)
                    chord_line = chord_line.ljust(col) + str(chords[ci]["symbol"]) + " "
                    ci += 1
                lyric_line += syl[:-1] if syl.endswith("-") else syl + " "
            while ci < len(chords):  # chords with no syllable after them in this measure
                col = max(len(lyric_line), len(chord_line))
                chord_line = chord_line.ljust(col) + str(chords[ci]["symbol"]) + " "
                lyric_line = lyric_line.ljust(len(chord_line))
                ci += 1
            if not sylls and not chords:
                lyric_line += "- "
        out += [chord_line.rstrip(), lyric_line.rstrip(), ""]
        i = j
    return "\n".join(out).rstrip() + "\n"


def _key_name(fifths: int, mode: str) -> str:
    majors = {-7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F", 0: "C", 1: "G", 2: "D", 3: "A",
              4: "E", 5: "B", 6: "F#", 7: "C#"}
    minors = {-7: "Ab", -6: "Eb", -5: "Bb", -4: "F", -3: "C", -2: "G", -1: "D", 0: "A", 1: "E", 2: "B", 3: "F#",
              4: "C#", 5: "G#", 6: "D#", 7: "A#"}
    f = max(-7, min(7, fifths))
    return f"{minors[f]} minor" if str(mode).startswith("min") else f"{majors[f]} major"


def load(workdir: str) -> dict:
    return json.loads((Path(workdir) / "leadsheet.json").read_text(encoding="utf-8"))


SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "artist": {"type": "string"},
        "key": {"type": "integer", "description": "sharps (+) or flats (-) in the key signature"},
        "mode": {"type": "string", "enum": ["major", "minor"]},
        "time": {"type": "string"}, "tempo": {"type": "number"},
        "clef": {"type": "string", "description": "G2 (treble) or G2-8 (tenor/male voice treble 8vb) or F4 (bass)"},
        "beat_ql": {"type": "number", "description": "quarter notes per analysis beat (usually 1)"},
        "downbeat_beat": {"type": "number", "description": "analysis beat number of beat 1 of the first full measure"},
        "measures": {"type": "array", "items": {"type": "object", "properties": {
            "section": {"type": "string"},
            "incomplete": {"type": "boolean"},
            "time": {"type": "string"}, "key": {"type": "integer"},
            "barline": {"type": "string"},
            "chords": {"type": "array", "items": {"type": "object", "properties": {
                "beat": {"type": "number"}, "symbol": {"type": "string"}}, "required": ["beat", "symbol"]}},
            "melody": {"type": "string"}}, "required": ["melody", "chords"]}},
        "uncertain": {"type": "array", "items": {"type": "object", "properties": {
            "measure": {"type": "integer"}, "note": {"type": "string"}}, "required": ["measure", "note"]}},
        "changes": {"type": "string"},
    },
    "required": ["key", "time", "tempo", "downbeat_beat", "measures"],
}


def ql_of_time(time: str) -> Fraction:
    from ..notation import parse_time
    t = parse_time(time)
    return t[1] if t else Fraction(4)
