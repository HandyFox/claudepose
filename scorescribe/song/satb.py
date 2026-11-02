"""Four-part (SATB) hymn arrangement from a lead sheet: format, part-writing checks, MusicXML.

The soprano is the lead sheet's melody (with its lyrics). Claude writes the other
three voices in arrangement.json:

    {"measures": [{"alto": "D4/2 C#4/2", "tenor": "A3/2 A3/2", "bass": "D3/2 A2/2"}, ...],
     "soprano_changes": {"3": "F#4/2 E4/2"}}        # optional: replace the melody of a bar (hymn-style simplification)

one entry per lead-sheet measure, same notation as everywhere else. The checks
follow common-practice four-part writing: ranges, chord tones against the chord
symbols, the bass on the chord's bass note, complete chords, spacing, voice
crossing, parallel fifths and octaves, and a doubled leading tone.
"""
from __future__ import annotations

from fractions import Fraction

from ..model import assemble
from . import leadsheet as L

VOICES = ("soprano", "alto", "tenor", "bass")
RANGES = {"soprano": (60, 81), "alto": (55, 74), "tenor": (48, 69), "bass": (40, 62)}  # C4-A5, G3-D5, C3-A4, E2-D4
COMFORT = {"soprano": (62, 79), "alto": (57, 72), "tenor": (50, 67), "bass": (43, 60)}
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _nm(m: int) -> str:
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def soprano_of(ls: dict, arr: dict, i: int) -> str:
    ch = (arr.get("soprano_changes") or {}).get(str(i + 1))
    return ch if ch else (ls["measures"][i].get("melody") or "R")


def to_systems(ls: dict, arr: dict) -> list[dict]:
    am = arr.get("measures") or []
    measures = []
    for i, m in enumerate(ls.get("measures") or []):
        a = am[i] if i < len(am) else {}
        s1 = {"staff": 1, "voices": [soprano_of(ls, arr, i), a.get("alto") or "R"]}
        s2 = {"staff": 2, "voices": [a.get("tenor") or "R", a.get("bass") or "R"]}
        md = {"staves": [s1, s2]}
        if i == 0:
            md["time"] = ls.get("time") or "4/4"
            md["key"] = ls.get("key", 0)
            s1["clef"], s2["clef"] = "G2", "F4"
            if ls.get("tempo"):
                md["tempo"] = {"bpm": ls["tempo"]}
        for k in ("time", "key", "incomplete", "barline", "repeat_start", "volta"):
            if k in m and (i > 0 or k in ("incomplete", "barline", "repeat_start", "volta")):
                md[k] = m[k]
        if m.get("section"):
            s1["texts"] = [m["section"]]
        measures.append(md)
    return [{"measures": measures}]


def _doc(ls: dict, arr: dict):
    return assemble(to_systems(ls, arr), [{"name": arr.get("part_name") or "Choir", "staves": 2}],
                    ls.get("title"), ls.get("artist"), stitch=False)


def build_score(ls: dict, arr: dict, chord_symbols: bool = True):
    from music21 import harmony
    from ..build import build_score as _bs
    doc = _doc(ls, arr)
    score = _bs(doc)
    if chord_symbols:
        top = score.parts[0]
        for m21m, md in zip(top.getElementsByClass("Measure"), ls.get("measures") or []):
            for c in md.get("chords") or []:
                try:
                    h = L.parse_chord(str(c.get("symbol", "")))
                except ValueError:
                    continue
                if isinstance(h, harmony.ChordSymbol):
                    h.writeAsChord = False
                m21m.insert(max(0.0, float(c.get("beat", 1)) - 1.0), h)
    if ls.get("title"):
        score.metadata.title = ls["title"]
    if ls.get("artist"):
        score.metadata.composer = ls["artist"]
    return doc, score


def write(ls: dict, arr: dict, out_path: str, shape_notes: str | None = None, chord_symbols: bool = True) -> str:
    from ..build import tidy_musicxml
    _, score = build_score(ls, arr, chord_symbols)
    out = str(score.write("musicxml", fp=out_path))
    try:
        tidy_musicxml(out)
    except Exception:
        pass
    if shape_notes:
        from ..shapes import apply_shape_notes
        apply_shape_notes(out, shape_notes)
    return out


# ------------------------------------------------------------------ analysis of the voices

def _voice_events(score) -> dict[str, list[tuple[Fraction, Fraction, int | None, int]]]:
    """voice -> [(start ql, end ql, midi or None, measure index)] in absolute quarter notes."""
    from music21 import harmony, stream
    out = {v: [] for v in VOICES}
    for si, part in enumerate(score.parts[:2]):
        names = ("soprano", "alto") if si == 0 else ("tenor", "bass")
        for mi, m in enumerate(part.getElementsByClass("Measure")):
            base = Fraction(m.getOffsetBySite(part)).limit_denominator(960)
            voices = list(m.voices) or [m]
            for vi, v in enumerate(voices[:2]):
                name = names[vi] if len(voices) > 1 else names[0]
                for n in v.notesAndRests:
                    if isinstance(n, harmony.Harmony) or n.duration.isGrace:
                        continue
                    if n.isRest and n.style.hideObjectOnPrint:
                        continue
                    st = base + Fraction(n.getOffsetInHierarchy(m)).limit_denominator(960)
                    en = st + Fraction(n.quarterLength).limit_denominator(960)
                    midi = None if n.isRest else max(p.midi for p in n.pitches)
                    if isinstance(v, stream.Measure) and vi > 0:
                        continue
                    out[name].append((st, en, midi, mi))
    for k in out:
        out[k].sort()
    return out


def _sounding(evs, t):
    for st, en, midi, mi in evs:
        if st <= t < en:
            return midi, st == t, mi
    return None, False, None


def _chord_timeline(ls: dict, score) -> list[tuple[Fraction, Fraction, str]]:
    part = score.parts[0]
    ms = list(part.getElementsByClass("Measure"))
    pts = []
    for mi, (m, md) in enumerate(zip(ms, ls.get("measures") or [])):
        base = Fraction(m.getOffsetBySite(part)).limit_denominator(960)
        for c in md.get("chords") or []:
            pts.append((base + Fraction(float(c.get("beat", 1)) - 1).limit_denominator(960), str(c.get("symbol", ""))))
    pts.sort()
    end = Fraction(score.parts[0].highestTime).limit_denominator(960)
    return [(t, pts[k + 1][0] if k + 1 < len(pts) else end, sym) for k, (t, sym) in enumerate(pts)]


def check(ls: dict, arr: dict) -> dict:
    errors, checks, notes = [], [], []
    n_ls, n_arr = len(ls.get("measures") or []), len(arr.get("measures") or [])
    if n_arr != n_ls:
        errors.append(f"arrangement.json has {n_arr} measures but the lead sheet has {n_ls}; write one entry per measure")
    doc = _doc(ls, arr)
    vname = {(1, 1): "soprano", (1, 2): "alto", (2, 1): "tenor", (2, 2): "bass"}
    for i in doc.issues:
        if i.level == "error" or "padded" in i.msg:  # in four parts a short voice is a mistake, not a gap
            who = vname.get((i.staff, i.voice), "")
            errors.append(f"measure {i.measure}{' ' + who if who else ''}: {i.msg}")
    if errors:
        return _report(errors, checks, notes, {})
    _, score = build_score(ls, arr, chord_symbols=False)
    ev = _voice_events(score)
    chords = _chord_timeline(ls, score)
    meas_off = [Fraction(m.getOffsetBySite(score.parts[0])).limit_denominator(960)
                for m in score.parts[0].getElementsByClass("Measure")]

    def where(t: Fraction) -> str:
        mi = max(i for i, o in enumerate(meas_off) if o <= t) if meas_off else 0
        return f"measure {mi + 1} beat {float(t - meas_off[mi]) + 1:g}"

    # ranges
    for v in VOICES[1:]:
        lo, hi = RANGES[v]
        clo, chi = COMFORT[v]
        for st, en, midi, mi in ev[v]:
            if midi is None:
                continue
            if midi < lo or midi > hi:
                checks.append(f"{where(st)} {v}: {_nm(midi)} is outside the {v} range ({_nm(lo)}-{_nm(hi)})")
            elif (midi < clo or midi > chi) and en - st >= 2:
                notes.append(f"{where(st)} {v}: {_nm(midi)} held at the edge of the {v} range")

    # vertical slices at every onset
    onsets = sorted({st for v in VOICES for st, _, m, _ in ev[v]})
    kf = int(ls.get("key", 0) or 0)
    tonic = (7 * kf) % 12 if not str(ls.get("mode", "major")).startswith("min") else ((7 * kf) % 12 + 9) % 12
    leading = (tonic + 11) % 12
    prev = None
    stats = {"slices": 0, "chord_tone_ok": 0}
    for t in onsets:
        cur = {}
        fresh = {}
        for v in VOICES:
            midi, on, _ = _sounding(ev[v], t)
            cur[v] = midi
            fresh[v] = on
        if any(cur[v] is None for v in VOICES):
            prev = None
            continue
        stats["slices"] += 1
        ch = next((c for c in chords if c[0] <= t < c[1]), None)
        tones, root, cbass = (set(), None, None)
        if ch:
            try:
                tones, root, cbass = L.chord_pcs(ch[2])
            except ValueError:
                pass
        strong = (t - meas_off[max(i for i, o in enumerate(meas_off) if o <= t)]) % 1 == 0
        if tones:
            off = [v for v in VOICES[1:] if fresh[v] and cur[v] % 12 not in tones]
            if not off:
                stats["chord_tone_ok"] += 1
            for v in off:
                dur = next(en - st for st, en, m, _ in ev[v] if st == t)
                if strong or dur >= 1:
                    checks.append(f"{where(t)} {v}: {_nm(cur[v])} is not in {ch[2]} "
                                  f"({'on a strong beat' if strong else 'held'}; fine only as a suspension)")
            if t == ch[0]:
                if cbass is not None and cur["bass"] % 12 != cbass:
                    checks.append(f"{where(t)} bass: {_nm(cur['bass'])} but {ch[2]} has {NAMES[cbass]} in the bass"
                                  + (" (write the chord as an inversion only if you mean it)" if cur["bass"] % 12 in tones else ""))
                present = {cur[v] % 12 for v in VOICES}
                third = next((p for p in tones if (p - root) % 12 in (3, 4)), None) if root is not None else None
                if third is not None and third not in present:
                    checks.append(f"{where(t)}: {ch[2]} has no third ({NAMES[third]}) in any voice - the chord sounds empty")
        # spacing and crossing
        s, a, te, b = cur["soprano"], cur["alto"], cur["tenor"], cur["bass"]
        if a > s or te > a or b > te:
            pairs = [n for n, ok in (("alto above soprano", a <= s), ("tenor above alto", te <= a),
                                     ("bass above tenor", b <= te)) if not ok]
            checks.append(f"{where(t)}: voices cross ({', '.join(pairs)})")
        if s - a > 12:
            checks.append(f"{where(t)}: soprano and alto more than an octave apart ({_nm(s)}/{_nm(a)})")
        if a - te > 12:
            checks.append(f"{where(t)}: alto and tenor more than an octave apart ({_nm(a)}/{_nm(te)})")
        if te - b > 19:
            notes.append(f"{where(t)}: tenor and bass more than a twelfth apart ({_nm(te)}/{_nm(b)})")
        if sum(1 for v in VOICES if cur[v] % 12 == leading) > 1 and tones and leading in tones:
            checks.append(f"{where(t)}: the leading tone {NAMES[leading]} is doubled")
        # parallel fifths and octaves between consecutive slices
        if prev is not None:
            for i, v1 in enumerate(VOICES):
                for v2 in VOICES[i + 1:]:
                    p1, p2, c1, c2 = prev[v1], prev[v2], cur[v1], cur[v2]
                    if p1 == c1 and p2 == c2:
                        continue
                    i_prev, i_cur = (p1 - p2) % 12, (c1 - c2) % 12
                    moved_same = (c1 - p1) * (c2 - p2) > 0
                    if moved_same and i_prev == i_cur and i_cur in (0, 7) and p1 != c1:
                        kind = "octaves/unisons" if i_cur == 0 else "fifths"
                        checks.append(f"{where(t)}: parallel {kind} between {v1} and {v2} "
                                      f"({_nm(p1)}-{_nm(p2)} to {_nm(c1)}-{_nm(c2)})")
            if (prev["soprano"] != cur["soprano"] and (cur["soprano"] - cur["bass"]) % 12 in (0, 7)
                    and (cur["soprano"] - prev["soprano"]) * (cur["bass"] - prev["bass"]) > 0
                    and abs(cur["soprano"] - prev["soprano"]) > 2 and (prev["soprano"] - prev["bass"]) % 12 not in (0, 7)):
                notes.append(f"{where(t)}: hidden (direct) {'octave' if (cur['soprano'] - cur['bass']) % 12 == 0 else 'fifth'} "
                             "between soprano and bass (leap in the soprano)")
        # big leaps in inner voices
        if prev is not None:
            for v in ("alto", "tenor"):
                if abs(cur[v] - prev[v]) > 7:
                    notes.append(f"{where(t)} {v}: leap of {abs(cur[v] - prev[v])} semitones (inner voices usually move by step)")
        prev = cur
    stats["chord_tones"] = f"{stats['chord_tone_ok']}/{stats['slices']}"
    return _report(errors, checks, notes, stats)


def _report(errors, checks, notes, stats) -> dict:
    lines = ["Four-part arrangement check", ""]
    if stats:
        lines += [f"Slices where alto, tenor and bass are all chord tones: {stats.get('chord_tones', '?')}", ""]
    for title, items in (("ERRORS (must fix)", errors), ("CHECK (part-writing problems: fix, or keep deliberately)", checks),
                         ("NOTES (style; use judgement)", notes)):
        lines.append(f"{title}: {len(items)}")
        lines += [f"  - {x}" for x in items]
        lines.append("")
    return {"errors": errors, "checks": checks, "notes": notes, "stats": stats, "text": "\n".join(lines)}


SCHEMA = {
    "type": "object",
    "properties": {
        "measures": {"type": "array", "description": "one entry per lead-sheet measure",
                     "items": {"type": "object", "properties": {
                         "alto": {"type": "string"}, "tenor": {"type": "string"}, "bass": {"type": "string"}},
                         "required": ["alto", "tenor", "bass"]}},
        "soprano_changes": {"type": "object", "description": "optional: measure number (string) -> replacement melody tokens",
                            "additionalProperties": {"type": "string"}},
        "changes": {"type": "string"},
    },
    "required": ["measures"],
}
