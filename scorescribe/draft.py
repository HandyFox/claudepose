"""Convert an existing MusicXML file (e.g. Audiveris output) into our token format.

Used to give Claude a draft to correct - Audiveris, oemer, homr or any OMR
engine that writes MusicXML does the first read, Claude fixes it against the
image - and for round-trip tests.
"""
from __future__ import annotations

from fractions import Fraction

from music21 import (articulations, chord, clef, converter, dynamics, expressions, key, layout,
                     meter, note, spanner, stream, tempo)

from .notation import key_alters

DUR_CODE = {"breve": "0", "whole": "1", "half": "2", "quarter": "4", "eighth": "8", "16th": "16",
            "32nd": "32", "64th": "64", "128th": "128"}
ACC_CODE = {1: "#", 2: "##", -1: "b", -2: "bb", 0: "n"}
ART_NAMES = [(articulations.Staccatissimo, "staccatissimo"), (articulations.Staccato, "staccato"),
             (articulations.StrongAccent, "marcato"), (articulations.Accent, "accent"),
             (articulations.Tenuto, "tenuto")]


def _clef_code(c) -> str | None:
    if isinstance(c, clef.PercussionClef):
        return "perc"
    if not getattr(c, "sign", None) or c.sign not in "GFC":
        return None
    code = f"{c.sign}{c.line}"
    if c.octaveChange == -1:
        code += "-8"
    elif c.octaveChange == 1:
        code += "+8"
    return code


def _pitch_tok(p, kalt, state, tied) -> str:
    alter = int(p.accidental.alter) if p.accidental else 0
    shown = p.accidental.displayStatus if p.accidental else None
    implied = state.get((p.step, p.octave), kalt[p.step])
    if shown is None:
        shown = alter != implied
    state[(p.step, p.octave)] = alter
    acc = ACC_CODE.get(alter, "") if shown else ""
    return f"{p.step}{acc}{p.octave}" + ("~" if tied else "")


def _dur_tok(d) -> str | None:
    code = DUR_CODE.get(d.type)
    if code is None:
        return None
    t = code + "." * d.dots
    if d.tuplets:
        tp = d.tuplets[0]
        t += f"t{tp.numberNotesActual}" + (f":{tp.numberNotesNormal}" if tp.numberNotesNormal not in (2, 4, 8) or tp.numberNotesActual in (2, 4) else "")
    return t


def _el_tok(el, kalt, state, measure_len, dyn_at) -> str | None:
    mods = []
    if isinstance(el, note.Rest):
        if el.fullMeasure in (True, "always") or (el.duration.quarterLength == measure_len and el.duration.type == "whole"):
            return "R"
        d = _dur_tok(el.duration)
        return None if d is None else (("s" if el.style.hideObjectOnPrint else "r") + "/" + d)
    if isinstance(el, chord.Chord):
        ns = sorted(el.notes, key=lambda n: n.pitch.ps)
        body = "+".join(_pitch_tok(n.pitch, kalt, state, bool(n.tie and n.tie.type in ("start", "continue"))) for n in ns)
    elif isinstance(el, note.Note):
        body = _pitch_tok(el.pitch, kalt, state, bool(el.tie and el.tie.type in ("start", "continue")))
    else:
        return None
    if el.duration.isGrace:
        d = DUR_CODE.get(el.duration.type, "8")
        prefix = "gs:" if getattr(el.duration, "slash", False) else "g:"
        return prefix + body + "/" + d
    d = _dur_tok(el.duration)
    if d is None:
        return None
    for cls, name in ART_NAMES:
        if any(isinstance(a, cls) for a in el.articulations):
            mods.append(name)
    for x in el.expressions:
        if isinstance(x, expressions.Fermata):
            mods.append("fermata")
        elif isinstance(x, expressions.Trill):
            mods.append("trill")
    mods += dyn_at.pop(float(el.offset), [])
    for ly in el.lyrics:
        txt = ly.text or ""
        if ly.syllabic in ("begin", "middle"):
            txt += "-"
        if txt and "," not in txt and "]" not in txt:
            mods.append(f"lyric{'' if ly.number == 1 else ly.number}={txt.replace(' ', '_')}")
    return body + "/" + d + (f"[{','.join(mods)}]" if mods else "")


def _voices(m: stream.Measure) -> list[stream.Stream]:
    vs = list(m.voices)
    if vs:
        return vs
    return [m]


def score_to_systems(path: str, use_breaks: bool = True) -> tuple[dict, list[dict]]:
    """Returns (meta, systems). Systems split at <print new-system> breaks when present."""
    s = converter.parse(path)
    staves = list(s.parts)
    parts = []
    # group PartStaffs back into parts by shared name / StaffGroup
    seen = []
    for sg in s.getElementsByClass(layout.StaffGroup):
        mem = [x for x in sg.getSpannedElements() if x in staves]
        if len(mem) > 1:
            seen.append(mem)
    i = 0
    while i < len(staves):
        g = next((g for g in seen if g[0] is staves[i]), None)
        if g:
            parts.append({"name": staves[i].partName or "Piano", "staves": len(g)})
            i += len(g)
        else:
            parts.append({"name": staves[i].partName or f"Part {len(parts) + 1}", "staves": 1})
            i += 1
    md = s.metadata
    meta = {"title": md.title if md else None, "composer": md.composer if md else None, "parts": parts}

    measure_lists = [list(p.getElementsByClass(stream.Measure)) for p in staves]
    n = min(len(ml) for ml in measure_lists)
    cur_key = 0
    cur_ts = None
    volta_of: dict[int, str] = {}
    for rb in staves[0].getElementsByClass(spanner.RepeatBracket):
        for m in rb.getSpannedElements():
            volta_of[id(m)] = str(rb.number).split(",")[0].strip()
    systems: list[dict] = [{"measures": []}]
    for mi in range(n):
        top = measure_lists[0][mi]
        if use_breaks and mi > 0:
            sl = top.getElementsByClass(layout.SystemLayout)
            pl = top.getElementsByClass(layout.PageLayout)
            if (sl and sl[0].isNew) or (pl and pl[0].isNew):
                systems.append({"measures": []})
        md_ = {"staves": []}
        ts = top.getElementsByClass(meter.TimeSignature)
        if ts:
            cur_ts = ts[0]
            md_["time"] = ts[0].ratioString
        ks = top.getElementsByClass(key.KeySignature)
        if ks:
            cur_key = ks[0].sharps
            md_["key"] = cur_key
        if len(systems[-1]["measures"]) == 0:
            md_["key"] = cur_key
        mlen = Fraction(cur_ts.barDuration.quarterLength).limit_denominator(64) if cur_ts else Fraction(4)
        if mi == 0 and top.paddingLeft:
            md_["incomplete"] = True
        if top.rightBarline is not None:
            t = getattr(top.rightBarline, "type", "")
            if getattr(top.rightBarline, "direction", None) == "end":
                md_["barline"] = "repeat_end"
            elif t in ("final", "light-heavy"):
                md_["barline"] = "final"
            elif t in ("double", "light-light"):
                md_["barline"] = "double"
        if top.leftBarline is not None and getattr(top.leftBarline, "direction", None) == "start":
            md_["repeat_start"] = True
        if id(top) in volta_of:
            md_["volta"] = volta_of[id(top)]
        mms = top.recurse().getElementsByClass(tempo.MetronomeMark)
        if mms:
            mm = mms[0]
            md_["tempo"] = {k: v for k, v in (("text", mm.text), ("bpm", None if mm.numberImplicit else mm.number)) if v}
        kalt = key_alters(cur_key)
        for st_i, ml in enumerate(measure_lists, start=1):
            m = ml[mi]
            entry = {"staff": st_i, "voices": []}
            cl = m.getElementsByClass(clef.Clef)
            if cl:
                code = _clef_code(cl[0])
                if code:
                    entry["clef"] = code
            state: dict = {}
            dyn_at: dict[float, list[str]] = {}
            for d in m.recurse().getElementsByClass(dynamics.Dynamic):
                if d.value:
                    dyn_at.setdefault(float(d.offset), []).append(d.value)
            for v in _voices(m):
                toks = []
                for el in v.notesAndRests:
                    t = _el_tok(el, kalt, state, mlen, dyn_at)
                    if t:
                        toks.append(t)
                if toks:
                    entry["voices"].append(" ".join(toks))
            if not entry["voices"]:
                entry["voices"] = ["R"]
            md_["staves"].append(entry)
        systems[-1]["measures"].append(md_)
    # clefs must be present at each system start: carry them
    last_clef: dict[int, str] = {}
    for sy in systems:
        for k, mdict in enumerate(sy["measures"]):
            for e in mdict["staves"]:
                if "clef" in e:
                    last_clef[e["staff"]] = e["clef"]
                elif k == 0 and e["staff"] in last_clef:
                    e["clef"] = last_clef[e["staff"]]
    return meta, [sy for sy in systems if sy["measures"]]


def system_text(sysd: dict) -> str:
    """Compact human/Claude-readable dump of one system."""
    import json
    return json.dumps(sysd["measures"], ensure_ascii=False, separators=(",", ":"))
