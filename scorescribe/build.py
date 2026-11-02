"""Turn the assembled Doc into a music21 score and MusicXML for MuseScore."""
from __future__ import annotations

from fractions import Fraction

from music21 import (articulations, bar, clef, duration, dynamics, expressions,
                     key, layout, metadata, meter, note, chord, pitch, spanner,
                     stream, tempo)

from .model import Doc, MeasureM
from .notation import ACC_ALTER, DUR_TYPE, Event, key_alters

FLAG_COLOR = "#D0021B"     # failed an automatic check
UNSURE_COLOR = "#E07B00"   # Claude marked it uncertain (only when asked for)

import contextvars  # noqa: E402

MARK_UNCERTAIN: contextvars.ContextVar[bool] = contextvars.ContextVar("mark_uncertain", default=False)
OTTAVA_SHIFT = {"8va": 1, "8vb": -1, "15ma": 2, "15mb": -2}

ARTIC_CLASSES = {
    "staccato": articulations.Staccato, "staccatissimo": articulations.Staccatissimo,
    "accent": articulations.Accent, "tenuto": articulations.Tenuto,
    "marcato": articulations.StrongAccent, "upbow": articulations.UpBow,
    "downbow": articulations.DownBow, "harmonic": articulations.Harmonic,
    "breath": articulations.BreathMark,
}
EXPR_CLASSES = {
    "fermata": expressions.Fermata, "trill": expressions.Trill,
    "mordent": expressions.Mordent, "turn": expressions.Turn,
}


def make_clef(code: str):
    if code == "perc":
        return clef.PercussionClef()
    if code == "tab":
        return clef.TabClef()
    shift = 0
    base = code
    if code.endswith("-8"):
        shift, base = -1, code[:-2]
    elif code.endswith("+8"):
        shift, base = 1, code[:-2]
    return clef.clefFromString(base, octaveShift=shift)


def make_duration(ev: Event) -> duration.Duration:
    d = duration.Duration(type=DUR_TYPE[ev.dur_code], dots=ev.dots)
    if ev.tuplet:
        n, m = ev.tuplet
        t = duration.Tuplet(n, m)
        t.setDurationType(DUR_TYPE[ev.dur_code], ev.dots)
        d.appendTuplet(t)
    return d


class _Ctx:
    """Per-staff running state while building."""

    def __init__(self):
        self.last_in_voice: dict[int, object] = {}   # voice idx -> last m21 note/chord
        self.last_ev_in_voice: dict[int, Event] = {}
        self.open_slur: dict[int, list] = {}
        self.open_hairpin: tuple[str, list] | None = None
        self.open_ottava: tuple[str, list] | None = None
        self.open_pedal: list | None = None
        self.spanners: list = []
        self.prev_accs: dict = {}  # (step, octave) -> alter for altered notes in the previous measure


def _resolve_alters(M: MeasureM, st: int, voices: list[list[Event]], ctx: _Ctx):
    """Apply key signature, in-measure accidental carry and ties to written pitches."""
    kalt = key_alters(M.eff_key)
    state: dict[tuple[str, int], int] = {}
    timeline = []
    for v_i, evs in enumerate(voices):
        t = Fraction(0)
        for order, e in enumerate(evs):
            timeline.append((t, 0 if e.is_grace else 1, v_i, order, e))
            t += e.ql if e.kind != "mrest" else M.length
    timeline.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    prev_in_voice = dict(ctx.last_ev_in_voice)
    shown: dict[tuple[str, int], int] = {}  # what the reader would assume, for display decisions
    for _, _, v_i, _, e in timeline:
        prev = prev_in_voice.get(v_i)
        for p in e.pitches:
            tied_from = None
            if prev is not None:
                for q in prev.pitches:
                    if q.tie and q.step == p.step and q.octave == p.octave:
                        tied_from = q
                        break
            if p.acc is not None:
                p.alter = ACC_ALTER[p.acc]
                state[(p.step, p.octave)] = p.alter
            elif tied_from is not None:
                p.alter = tied_from.alter
            elif (p.step, p.octave) in state:
                p.alter = state[(p.step, p.octave)]
            else:
                p.alter = kalt[p.step]
            # Show an accidental only where engraving rules need one: the pitch differs
            # from what the key signature and earlier accidentals in this measure imply.
            # (Claude may write F#4 for a note the key signature already sharpens; that
            # must not turn into a printed sharp.) Tied-over notes never get one.
            key_ = (p.step, p.octave)
            implied = shown.get(key_, kalt[p.step])
            p.show = tied_from is None and p.alter != implied
            # Courtesy accidentals are not added automatically (editions differ, and an
            # accidental that isn't on the page is clutter); one the reader saw printed is
            # written with "!" (F#!4) and is kept.
            if p.force:
                p.show = True
            shown[key_] = p.alter
        if e.pitches:
            prev_in_voice[v_i] = e
    ctx.prev_accs = {k: v for k, v in shown.items() if v != kalt[k[0]]}


def _m21_pitch(p, shift: int) -> pitch.Pitch:
    pp = pitch.Pitch(step=p.step, octave=p.octave)
    if p.alter or p.show:
        pp.accidental = pitch.Accidental(p.alter)  # 0 -> natural
        pp.accidental.displayStatus = bool(p.show)
        pp.accidental.displayType = "normal" if p.show else "never"
    return pp


def _make_element(ev: Event, M: MeasureM, color: str | None):
    if ev.kind == "mrest":
        r = note.Rest(quarterLength=M.length)
        r.fullMeasure = True
        el = r
    elif ev.kind in ("rest", "space"):
        el = note.Rest()
        el.duration = make_duration(ev)
        if ev.kind == "space":
            el.style.hideObjectOnPrint = True
    else:
        shift = 0
        for s in ev.spanners:
            shift = OTTAVA_SHIFT.get(s, shift)
        pitches = [_m21_pitch(p, shift) for p in ev.pitches]
        if len(pitches) == 1:
            el = note.Note(pitches[0])
        else:
            el = chord.Chord([note.Note(p) for p in pitches])
        el.duration = make_duration(ev)
        if ev.grace:
            el = el.getGrace()
            el.duration.slash = ev.grace == "slash"
        for a in ev.articulations:
            if a in ARTIC_CLASSES:
                el.articulations.append(ARTIC_CLASSES[a]())
            elif a in EXPR_CLASSES:
                x = EXPR_CLASSES[a]()
                if a == "fermata":
                    x.type = "upright"
                el.expressions.append(x)
            elif a == "arpeggio" and isinstance(el, chord.Chord):
                el.expressions.append(expressions.ArpeggioMark("normal"))
        for f in ev.fingerings:
            try:
                el.articulations.append(articulations.Fingering(int(f)))
            except ValueError:
                el.articulations.append(articulations.Fingering(f))
        for num, txt in ev.lyrics:
            el.addLyric(txt, lyricNumber=num)
    if color and ev.kind != "space":
        el.style.color = color
    return el


def _set_ties(el, ev: Event, prev_el, prev_ev: Event | None):
    from music21 import tie
    if ev.kind not in ("note", "chord"):
        return
    els = [el] if isinstance(el, note.Note) else list(el.notes)
    prev_notes = []
    if prev_el is not None and prev_ev is not None:
        prev_notes = [prev_el] if isinstance(prev_el, note.Note) else list(getattr(prev_el, "notes", []))
    for n, p in zip(els, ev.pitches):
        stop = False
        for pn, pp in zip(prev_notes, prev_ev.pitches if prev_ev else []):
            if pp.tie and pn.pitch.step == n.pitch.step and pn.pitch.octave == n.pitch.octave:
                stop = True
                break
        if stop and p.tie:
            n.tie = tie.Tie("continue")
        elif stop:
            n.tie = tie.Tie("stop")
        elif p.tie:
            n.tie = tie.Tie("start")
    if isinstance(el, note.Note) and ev.pitches:
        pass


def _fill_staff_measure(m21m: stream.Measure, M: MeasureM, st: int, ctx: _Ctx, top_staff: bool):
    S = M.staves[st]
    voices = [list(v) for v in S.voices]
    _resolve_alters(M, st, voices, ctx)
    containers = []
    multi = len(voices) > 1
    for v_i, evs in enumerate(voices):
        cont = stream.Voice(id=str(v_i + 1)) if multi else m21m
        offset = Fraction(0)
        for ev in evs:
            color = None
            if None in M.flag_staves or st in M.flag_staves:
                color = FLAG_COLOR
            elif MARK_UNCERTAIN.get() and (None in M.unsure_staves or st in M.unsure_staves):
                color = UNSURE_COLOR
            el = _make_element(ev, M, color)
            if ev.kind in ("note", "chord"):
                _set_ties(el, ev, ctx.last_in_voice.get(v_i), ctx.last_ev_in_voice.get(v_i))
            for d in ev.dynamics:
                cont.insert(float(offset), dynamics.Dynamic(d))
            for t in ev.texts:
                cont.insert(float(offset), expressions.TextExpression(t))
            cont.insert(float(offset), el) if multi else None
            if not multi:
                m21m.insert(float(offset), el)
            if ev.kind in ("note", "chord"):
                _spanners(el, ev, v_i, ctx)
                if not ev.grace:
                    ctx.last_in_voice[v_i] = el
                    ctx.last_ev_in_voice[v_i] = ev
            offset += M.length if ev.kind == "mrest" else ev.ql
        # pad short voices with invisible rests so the measure imports cleanly
        target = M.length if not M.incomplete else None
        if target is not None and offset < target and v_i > 0:
            gap = target - offset
            r = note.Rest(quarterLength=gap)
            r.style.hideObjectOnPrint = True
            cont.insert(float(offset), r)
        if multi:
            containers.append(cont)
    for c in containers:
        m21m.insert(0, c)
    for t in S.texts:
        te = expressions.TextExpression(t)
        te.placement = "above"  # rehearsal marks / section names sit above the staff
        m21m.insert(0, te)


def _spanners(el, ev: Event, v_i: int, ctx: _Ctx):
    sp = set(ev.spanners)
    # slurs (per voice)
    if v_i in ctx.open_slur:
        ctx.open_slur[v_i].append(el)
    if "slur_start" in sp:
        if v_i in ctx.open_slur and len(ctx.open_slur[v_i]) > 1:
            ctx.spanners.append(spanner.Slur(ctx.open_slur[v_i][:-1]))
        ctx.open_slur[v_i] = [el]
    if "slur_end" in sp and v_i in ctx.open_slur:
        els = ctx.open_slur.pop(v_i)
        if len(els) > 1:
            ctx.spanners.append(spanner.Slur(els))
    # hairpins (per staff)
    if ctx.open_hairpin:
        ctx.open_hairpin[1].append(el)
    for kind in ("cresc_start", "dim_start"):
        if kind in sp:
            _close_hairpin(ctx)
            ctx.open_hairpin = (kind, [el])
    if "hairpin_end" in sp:
        _close_hairpin(ctx)
    # ottava lines
    ot = next((s for s in ev.spanners if s in OTTAVA_SHIFT), None)
    if ot:
        if ctx.open_ottava and ctx.open_ottava[0] == ot:
            ctx.open_ottava[1].append(el)
        else:
            _close_ottava(ctx)
            ctx.open_ottava = (ot, [el])
    elif ctx.open_ottava and v_i == 0:
        _close_ottava(ctx)
    # pedal
    if ctx.open_pedal is not None:
        ctx.open_pedal.append(el)
    if "ped_start" in sp:
        _close_pedal(ctx)
        ctx.open_pedal = [el]
    if "ped_end" in sp:
        _close_pedal(ctx)


def _close_hairpin(ctx: _Ctx):
    if ctx.open_hairpin and len(ctx.open_hairpin[1]) >= 1:
        kind, els = ctx.open_hairpin
        cls = dynamics.Crescendo if kind == "cresc_start" else dynamics.Diminuendo
        ctx.spanners.append(cls(els[0], els[-1]) if len(els) > 1 else cls(els[0]))
    ctx.open_hairpin = None


def _close_ottava(ctx: _Ctx):
    if ctx.open_ottava:
        kind, els = ctx.open_ottava
        ctx.spanners.append(spanner.Ottava(els, type=kind, transposing=True))
    ctx.open_ottava = None


def _close_pedal(ctx: _Ctx):
    if ctx.open_pedal:
        try:
            ctx.spanners.append(expressions.PedalMark(ctx.open_pedal))
        except Exception:
            pass
    ctx.open_pedal = None


def build_score(doc: Doc) -> stream.Score:
    score = stream.Score()
    md = metadata.Metadata()
    md.title = doc.title or "Transcribed score"
    if doc.composer:
        md.composer = doc.composer
    score.insert(0, md)

    staff_streams: dict[int, stream.Stream] = {}
    st = 1
    groups = []
    for p_i, part in enumerate(doc.parts):
        members = []
        for k in range(part["staves"]):
            s = stream.PartStaff() if part["staves"] > 1 else stream.Part()
            s.id = f"P{p_i + 1}S{k + 1}"
            s.partName = part["name"]
            s.partAbbreviation = part["name"][:5] + ("." if len(part["name"]) > 5 else "")
            staff_streams[st] = s
            members.append(s)
            st += 1
        if len(members) > 1:
            groups.append(layout.StaffGroup(members, name=part["name"], symbol="brace", barTogether=True))

    ctxs = {s: _Ctx() for s in staff_streams}
    top_of_part = set()
    st = 1
    for part in doc.parts:
        top_of_part.add(st)
        st += part["staves"]

    volta_open: dict[int, tuple[str, list]] = {}
    for M in doc.measures:
        for s, strm in staff_streams.items():
            m = stream.Measure(number=M.number)
            if M.time:
                ratio, _, symbol = M.time[0].partition("|")
                ts = meter.TimeSignature(ratio)
                if symbol:
                    ts.symbol = symbol
                m.insert(0, ts)
            if M.key is not None:
                m.insert(0, key.KeySignature(M.key))
            S = M.staves[s]
            if S.clef:
                m.insert(0, make_clef(S.clef))
            if M.tempo_text or M.tempo_bpm:
                if s == 1:
                    if M.tempo_bpm:
                        mm = tempo.MetronomeMark(text=M.tempo_text, number=M.tempo_bpm,
                                                 referent=duration.Duration(1.0))
                    else:
                        mm = tempo.MetronomeMark(text=M.tempo_text)
                        mm.numberImplicit = True
                    mm.placement = "above"
                    m.insert(0, mm)
            if M.repeat_start:
                m.leftBarline = bar.Repeat(direction="start")
            if M.barline == "repeat_end":
                m.rightBarline = bar.Repeat(direction="end", times=M.repeat_times)
            elif M.barline in ("double", "final", "dashed"):
                m.rightBarline = bar.Barline({"double": "double", "final": "final", "dashed": "dashed"}[M.barline])
            _fill_staff_measure(m, M, s, ctxs[s], s in top_of_part)
            if M.incomplete:
                full = M.length
                actual = Fraction(m.duration.quarterLength).limit_denominator(1024)
                if actual < full:
                    m.paddingLeft = float(full - actual) if M is doc.measures[0] else 0.0
                    if M is not doc.measures[0]:
                        m.paddingRight = float(full - actual)
            strm.append(m)
            # voltas: on every staff (music21 merges piano staves and keeps the lower barline)
            if True:
                cur = volta_open.get(s)
                if M.volta:
                    if cur and cur[0] == M.volta:
                        cur[1].append(m)
                    else:
                        if cur:
                            strm.insert(0, spanner.RepeatBracket(cur[1], number=cur[0]))
                        volta_open[s] = (M.volta, [m])
                elif cur:
                    strm.insert(0, spanner.RepeatBracket(cur[1], number=cur[0]))
                    volta_open.pop(s)
    for s, cur in volta_open.items():
        staff_streams[s].insert(0, spanner.RepeatBracket(cur[1], number=cur[0]))

    for s, c in ctxs.items():
        _close_hairpin(c)
        _close_ottava(c)
        _close_pedal(c)
        for sl in c.open_slur.values():
            if len(sl) > 1:
                c.spanners.append(spanner.Slur(sl))
        for sp in c.spanners:
            staff_streams[s].insert(0, sp)

    for s in sorted(staff_streams):
        score.insert(0, staff_streams[s])
    for g in groups:
        score.insert(0, g)
    return score


def write_musicxml(doc: Doc, path: str, mark_uncertain: bool = False, shape_notes: str | None = None) -> str:
    token = MARK_UNCERTAIN.set(mark_uncertain)
    try:
        score = build_score(doc)
    finally:
        MARK_UNCERTAIN.reset(token)
    out = str(score.write("musicxml", fp=path))
    try:
        tidy_musicxml(out)
    except Exception:
        pass  # the untidied file is still valid
    if shape_notes:
        from .shapes import apply_shape_notes
        apply_shape_notes(out, shape_notes)
    return out


def tidy_musicxml(path: str) -> int:
    """Turn invisible rests in secondary voices into <forward> gaps.

    MuseScore shows invisible rests as grey clutter on screen. Voices 2-4 may
    simply have gaps, which is how a hand-made score looks, so those rests are
    replaced by <forward>. The main voice on each staff keeps its rests, because
    MuseScore requires that voice to be complete. Returns how many were replaced.
    """
    import xml.etree.ElementTree as ET

    with open(path, "rb") as f:
        raw = f.read()
    head_end = raw.find(b"<score-partwise")
    header = raw[:head_end] if head_end > 0 else b'<?xml version="1.0" encoding="utf-8"?>\n'
    root = ET.fromstring(raw)
    replaced = 0
    for measure in root.iter("measure"):
        kids = list(measure)
        main_voice: dict[str, str] = {}
        for el in kids:
            if el.tag == "note" and el.find("voice") is not None:
                st = el.findtext("staff", "1")
                main_voice.setdefault(st, el.findtext("voice"))
        for el in kids:
            if el.tag != "note" or el.get("print-object") != "no" or el.find("rest") is None:
                continue
            if el.find("chord") is not None or el.find("duration") is None:
                continue
            voice, st = el.findtext("voice", "1"), el.findtext("staff", "1")
            if main_voice.get(st) == voice:
                continue
            fwd = ET.Element("forward")
            ET.SubElement(fwd, "duration").text = el.findtext("duration")
            ET.SubElement(fwd, "voice").text = voice
            if el.find("staff") is not None:
                ET.SubElement(fwd, "staff").text = st
            fwd.tail = el.tail
            pos = list(measure).index(el)
            measure.remove(el)
            measure.insert(pos, fwd)
            replaced += 1
    body = ET.tostring(root, encoding="utf-8")
    if body.startswith(b"<?xml"):
        body = body[body.find(b"?>") + 2:].lstrip()
    with open(path, "wb") as f:
        f.write(header + body)
    return replaced
