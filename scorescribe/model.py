"""Assemble per-system transcriptions into one score and check every measure."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .notation import (Event, parse_key, parse_time, parse_voice, norm_clef)

BARLINES = {"regular", "double", "final", "repeat_end", "dashed", "none"}

# diatonic number of the bottom staff line for each clef (C0 = 0)
CLEF_BOTTOM = {"G2": 30, "G2-8": 30, "G2+8": 30, "G1": 32, "F4": 18, "F4-8": 18, "F4+8": 18,
               "F3": 20, "F5": 16, "C1": 28, "C2": 26, "C3": 24, "C4": 22, "C5": 20}
STEP_IDX = {s: i for i, s in enumerate("CDEFGAB")}


@dataclass
class Issue:
    system: int
    measure: int  # index within system, 1-based
    staff: int | None
    voice: int | None
    level: str  # error | warn
    msg: str

    def where(self) -> str:
        w = f"measure {self.measure}"
        if self.staff:
            w += f", staff {self.staff}"
        if self.voice:
            w += f", voice {self.voice}"
        return w


@dataclass
class StaffM:
    staff: int
    clef: str | None
    voices: list[list[Event]]
    texts: list[str]


@dataclass
class MeasureM:
    system: int
    idx: int
    number: int = 0
    time: tuple[str, Fraction] | None = None
    key: int | None = None
    eff_time: tuple[str, Fraction] = ("4/4", Fraction(4))
    eff_key: int = 0
    incomplete: bool = False
    barline: str = "regular"
    repeat_start: bool = False
    repeat_times: int | None = None
    volta: str | None = None
    tempo_text: str | None = None
    tempo_bpm: float | None = None
    staves: dict[int, StaffM] = field(default_factory=dict)
    eff_clefs: dict[int, str] = field(default_factory=dict)
    flagged: bool = False
    uncertain: bool = False
    flag_staves: set = field(default_factory=set)    # staves that failed a check (None = whole measure)
    unsure_staves: set = field(default_factory=set)  # staves Claude was unsure about (None = whole measure)

    @property
    def length(self) -> Fraction:
        return self.eff_time[1]


@dataclass
class Doc:
    title: str | None
    composer: str | None
    parts: list[dict]
    measures: list[MeasureM]
    issues: list[Issue]

    @property
    def n_staves(self) -> int:
        return sum(p["staves"] for p in self.parts)


def normalize_parts(parts) -> list[dict]:
    out = []
    for p in parts or []:
        try:
            n = int(p.get("staves", 1))
        except (TypeError, ValueError):
            n = 1
        out.append({"name": str(p.get("name") or f"Part {len(out) + 1}"), "staves": max(1, min(n, 4))})
    return out or [{"name": "Piano", "staves": 2}]


def voice_len(events: list[Event], bar: Fraction) -> Fraction:
    total = Fraction(0)
    for e in events:
        total += bar if e.kind == "mrest" else e.ql
    return total


def fmt_q(q: Fraction) -> str:
    """Quarter-note length as a friendly string, e.g. 3.5 beats."""
    return f"{float(q):g} quarter-note beats"


# ------------------------------------------------------------------ stitching

_SPACER_UNITS = [(Fraction(4), "1"), (Fraction(2), "2"), (Fraction(1), "4"), (Fraction(1, 2), "8"),
                 (Fraction(1, 4), "16"), (Fraction(1, 8), "32"), (Fraction(1, 16), "64")]


def filler(q: Fraction, rest: str = "s") -> str | None:
    """Rest/spacer tokens adding up to q quarter notes, e.g. 3 -> 's/2 s/4'."""
    out = []
    for size, code in _SPACER_UNITS:
        while q >= size:
            out.append(f"{rest}/{code}")
            q -= size
    return " ".join(out) if q == 0 else None


def _staff_lengths(md: dict, bar: Fraction) -> dict[int, Fraction] | None:
    """Voice-1 length per staff, or None if anything doesn't parse."""
    out = {}
    for st in md.get("staves") or []:
        try:
            num = int(st.get("staff"))
        except (TypeError, ValueError):
            return None
        voices = st.get("voices") or []
        if isinstance(voices, str):
            voices = [voices]
        if not voices:
            continue
        evs, errs = parse_voice(str(voices[0]))
        if errs or any(e.kind == "mrest" for e in evs):
            return None
        out[num] = voice_len(evs, bar)
    return out


def stitch_systems(systems: list[dict]) -> list[dict]:
    """Join a measure cut in two at an image edge (screenshots, phone photos).

    When one system ends with a partial measure and the next begins with one,
    and on every staff the two parts add up to exactly one full bar, they are
    merged into a single measure. Measures keep their original per-image
    numbers (`_idx`) so messages to Claude still match what it saw.
    """
    out = [dict(s, measures=[dict(m, _idx=i) for i, m in enumerate(s.get("measures") or [], start=1)])
           if s else s for s in systems]
    cur_time = None
    for k in range(len(out)):
        A = out[k]
        if not A:
            continue
        for m in A["measures"][:-1]:
            t = parse_time(m.get("time")) if m.get("time") else None
            cur_time = t or cur_time
        if not A["measures"]:
            continue
        last = A["measures"][-1]
        t = parse_time(last.get("time")) if last.get("time") else None
        cur_time = t or cur_time
        nxt = next((j for j in range(k + 1, len(out)) if out[j] and out[j].get("measures")), None)
        if nxt is None or nxt != k + 1 or cur_time is None:
            continue
        B = out[nxt]
        first = B["measures"][0]
        bt = parse_time(first.get("time")) if first.get("time") else None
        if bt and bt != cur_time:
            continue
        bar = cur_time[1]
        la, lb = _staff_lengths(last, bar), _staff_lengths(first, bar)
        if not la or not lb:
            continue
        a_vals, b_vals = set(la.values()), set(lb.values())
        if len(a_vals) != 1 or len(b_vals) != 1:
            continue
        len_a, len_b = a_vals.pop(), b_vals.pop()
        if not (0 < len_a < bar and 0 < len_b < bar and len_a + len_b == bar):
            continue
        fill_a, fill_b = filler(len_a), filler(len_b)
        if fill_a is None or fill_b is None:
            continue
        sa = {int(st["staff"]): st for st in last.get("staves") or []}
        sb = {int(st["staff"]): st for st in first.get("staves") or []}
        staves = []
        for num in sorted(set(sa) | set(sb)):
            a, b = sa.get(num, {}), sb.get(num, {})
            va = [str(v) for v in (a.get("voices") or [])] if not isinstance(a.get("voices"), str) else [a["voices"]]
            vb = [str(v) for v in (b.get("voices") or [])] if not isinstance(b.get("voices"), str) else [b["voices"]]
            n = max(len(va), len(vb), 1)
            voices = []
            for i in range(n):
                rest = "r" if i == 0 else "s"
                x = va[i] if i < len(va) else filler(len_a, rest)
                y = vb[i] if i < len(vb) else filler(len_b, rest)
                voices.append(f"{x} {y}")
            st = {"staff": num, "voices": voices}
            if a.get("clef"):
                st["clef"] = a["clef"]
            elif b.get("clef") and not a:
                st["clef"] = b["clef"]
            texts = (a.get("texts") or []) + (b.get("texts") or [])
            if texts:
                st["texts"] = texts
            staves.append(st)
        merged = {k2: v for k2, v in last.items() if k2 not in ("staves", "incomplete", "barline")}
        merged["staves"] = staves
        if first.get("barline"):
            merged["barline"] = first["barline"]
        if first.get("volta") and not merged.get("volta"):
            merged["volta"] = first["volta"]
        # a key or clef printed at the start of the next image is normally just the
        # restatement at a new line; if it really changes, it applies from the next measure
        rest_b = B["measures"][1:]
        if rest_b:
            nb = dict(rest_b[0])
            if first.get("key") is not None and "key" not in nb:
                nb["key"] = first["key"]
            nb_st = {int(x["staff"]): dict(x) for x in nb.get("staves") or [] if str(x.get("staff", "")).isdigit()}
            for num, b in sb.items():
                if b.get("clef") and num in nb_st and not nb_st[num].get("clef"):
                    nb_st[num]["clef"] = b["clef"]
            if nb_st:
                nb["staves"] = [nb_st[n2] for n2 in sorted(nb_st)]
            rest_b = [nb] + rest_b[1:]
        A["measures"] = A["measures"][:-1] + [merged]
        moved = []
        kept = []
        for u in B.get("uncertain") or []:
            try:
                (moved if int(u.get("measure", 0)) == 1 else kept).append(u)
            except (TypeError, ValueError, AttributeError):
                kept.append(u)
        if moved:
            A["uncertain"] = list(A.get("uncertain") or []) + [dict(u, measure=last["_idx"]) for u in moved]
        out[nxt] = dict(B, measures=rest_b, uncertain=kept, _stitched=True)
    return out


def assemble(systems: list[dict], parts: list[dict], title=None, composer=None, stitch: bool = True) -> Doc:
    parts = normalize_parts(parts)
    if stitch:
        systems = stitch_systems(systems)
    n_staves = sum(p["staves"] for p in parts)
    issues: list[Issue] = []
    measures: list[MeasureM] = []

    cur_time: tuple[str, Fraction] | None = None
    cur_key = 0
    cur_clefs: dict[int, str] = {}

    for s_i, sysd in enumerate(systems, start=1):
        if not sysd:
            issues.append(Issue(s_i, 0, None, None, "error", "system missing (no transcription)"))
            continue
        ms = sysd.get("measures") or []
        for u in sysd.get("uncertain") or []:
            try:
                issues.append(Issue(s_i, int(u.get("measure", 0)), u.get("staff"), None, "unsure",
                                    f"Unsure: {u.get('note', '')}"))
            except (TypeError, ValueError, AttributeError):
                pass
        if not ms and not sysd.get("_stitched"):
            issues.append(Issue(s_i, 0, None, None, "error", "no measures were returned for this system"))
        for m_i, md in enumerate(ms, start=1):
            m_i = md.get("_idx", m_i)
            M = MeasureM(system=s_i, idx=m_i)
            add = lambda lvl, msg, st=None, v=None: issues.append(Issue(s_i, m_i, st, v, lvl, msg))

            t = parse_time(md.get("time")) if md.get("time") else None
            if md.get("time") and not t:
                add("error", f"unrecognised time signature '{md.get('time')}' (use e.g. '3/4', 'C', 'C|')")
            if t and (cur_time is None or t != cur_time):
                M.time = t
                cur_time = t
            if cur_time is None:
                add("error", "no time signature known yet; report the time signature in the 'time' field of the first measure")
                cur_time = ("4/4", Fraction(4))
                M.time = cur_time
            M.eff_time = cur_time

            if md.get("key") not in (None, ""):
                k = parse_key(md.get("key"))
                if k is None:
                    add("error", f"unrecognised key '{md.get('key')}' (use an integer: number of sharps, negative for flats)")
                elif k != cur_key or not measures:
                    M.key = k
                    cur_key = k
            M.eff_key = cur_key

            M.incomplete = bool(md.get("incomplete") or md.get("pickup"))
            bl = str(md.get("barline") or "regular").lower()
            if bl in ("repeat", "end_repeat", "repeat-end"):
                bl = "repeat_end"
            if bl not in BARLINES:
                add("warn", f"unknown barline '{bl}' ignored")
                bl = "regular"
            M.barline = bl
            M.repeat_start = bool(md.get("repeat_start"))
            if md.get("repeat_times"):
                try:
                    M.repeat_times = int(md["repeat_times"])
                except (TypeError, ValueError):
                    pass
            if md.get("volta"):
                M.volta = str(md["volta"]).strip().rstrip(".")
            tempo = md.get("tempo")
            if isinstance(tempo, dict):
                M.tempo_text = tempo.get("text") or None
                try:
                    M.tempo_bpm = float(tempo["bpm"]) if tempo.get("bpm") else None
                except (TypeError, ValueError):
                    M.tempo_bpm = None
            elif isinstance(tempo, str) and tempo.strip():
                M.tempo_text = tempo.strip()

            bar = M.length
            seen = set()
            for sd in md.get("staves") or []:
                try:
                    st = int(sd.get("staff"))
                except (TypeError, ValueError):
                    add("error", f"staff entry without a valid 'staff' number: {sd!r}"[:200])
                    continue
                if st < 1 or st > n_staves:
                    add("error", f"staff {st} does not exist (the score has {n_staves} staves)", st)
                    continue
                if st in seen:
                    add("error", f"staff {st} appears twice in this measure", st)
                    continue
                seen.add(st)
                clef = None
                if sd.get("clef"):
                    clef = norm_clef(sd.get("clef"))
                    if clef is None:
                        add("warn", f"unknown clef '{sd.get('clef')}' ignored", st)
                    elif cur_clefs.get(st) == clef:
                        clef = None  # courtesy clef at system start, not a change
                    else:
                        cur_clefs[st] = clef
                voices_raw = sd.get("voices") or []
                if isinstance(voices_raw, str):
                    voices_raw = [voices_raw]
                voices = []
                unknown: list[str] = []
                for v_i, vt in enumerate(voices_raw, start=1):
                    evs, errs = parse_voice(str(vt), unknown)
                    for e in errs:
                        add("error", e, st, v_i)
                    voices.append(evs)
                for u in sorted(set(unknown)):
                    add("warn", f"unknown modifier '{u}' ignored", st)
                if not voices or all(not v for v in voices):
                    add("warn", "staff had no notes; filled with a whole-measure rest", st)
                    voices = [parse_voice("R")[0]]
                texts = [str(x) for x in (sd.get("texts") or []) if x]
                M.staves[st] = StaffM(st, clef, voices, texts)

            for st in range(1, n_staves + 1):
                if st not in M.staves:
                    add("warn", "staff missing from transcription; filled with a whole-measure rest", st)
                    M.staves[st] = StaffM(st, None, [parse_voice("R")[0]], [])
                if st not in cur_clefs:
                    guess = "G2" if st == 1 or n_staves == 1 else ("F4" if st == n_staves and parts[-1]["staves"] == 2 else "G2")
                    add("warn", f"no clef reported for staff {st}; assuming {guess}", st)
                    cur_clefs[st] = guess
                    M.staves[st].clef = guess
            M.eff_clefs = dict(cur_clefs)

            # ---- rhythm checks
            lengths = {}
            for st, S in M.staves.items():
                for v_i, evs in enumerate(S.voices, start=1):
                    if any(e.kind == "mrest" for e in evs) and len([e for e in evs if not e.is_grace]) > 1:
                        add("error", "'R' (whole-measure rest) must be the only token in its voice", st, v_i)
                        continue
                    L = voice_len(evs, bar)
                    lengths[(st, v_i)] = L
                    if M.incomplete:
                        continue
                    if v_i == 1 and L != bar:
                        add("error", f"voice adds up to {fmt_q(L)} but the {M.eff_time[0].split("|")[0]} measure needs {fmt_q(bar)}"
                            + (" (if this is a pickup/incomplete measure set \"incomplete\": true)" if m_i == 1 and L < bar else ""), st, v_i)
                    elif v_i > 1 and L > bar:
                        add("error", f"voice adds up to {fmt_q(L)}, more than the {fmt_q(bar)} the {M.eff_time[0].split("|")[0]} measure holds", st, v_i)
                    elif v_i > 1 and L < bar:
                        add("warn", f"voice {v_i} is {fmt_q(L)} long; padded with invisible rests (use 's' spacers to be explicit)", st, v_i)
            if M.incomplete and lengths:
                firsts = {k: v for k, v in lengths.items() if k[1] == 1}
                if len(set(firsts.values())) > 1:
                    detail = ", ".join(f"staff {k[0]}: {float(v):g}" for k, v in sorted(firsts.items()))
                    add("error", f"incomplete measure but staves disagree on its length ({detail} quarter beats)")
                if any(v > bar for v in lengths.values()):
                    add("error", f"incomplete measure is longer than a full {M.eff_time[0].split("|")[0]} measure")

            # ---- octave sanity checks (written pitch vs. clef)
            for st, S in M.staves.items():
                bottom = CLEF_BOTTOM.get(M.eff_clefs.get(st, "G2"))
                if bottom is None:
                    continue
                odd = []
                for evs in S.voices:
                    for e in evs:
                        for p in e.pitches:
                            d = p.octave * 7 + STEP_IDX[p.step]
                            if d < bottom - 10 or d > bottom + 8 + 10:
                                odd.append(f"{p.step}{p.octave}")
                if odd:
                    add("warn", f"pitches {', '.join(sorted(set(odd)))} are more than 5 ledger lines from the "
                        f"{M.eff_clefs.get(st)} staff - check the octave (keep them if they really are written that way)", st)
            measures.append(M)

    # measure numbers: a leading pickup is measure 0
    num = 0 if (measures and measures[0].incomplete) else 1
    for M in measures:
        M.number = num
        num += 1
    err_keys = {(i.system, i.measure) for i in issues if i.level == "error"}
    unsure_keys = {(i.system, i.measure) for i in issues if i.level == "unsure"}
    for M in measures:
        M.flagged = (M.system, M.idx) in err_keys
        M.uncertain = (M.system, M.idx) in unsure_keys
    by_key = {(M.system, M.idx): M for M in measures}
    for i in issues:
        M = by_key.get((i.system, i.measure))
        if M is None:
            continue
        st = i.staff if isinstance(i.staff, int) else None
        if i.level == "error":
            M.flag_staves.add(st)
        elif i.level == "unsure":
            M.unsure_staves.add(st)
    return Doc(title, composer, parts, measures, issues)


def issues_by_system(doc: Doc) -> dict[int, list[Issue]]:
    out: dict[int, list[Issue]] = {}
    for i in doc.issues:
        out.setdefault(i.system, []).append(i)
    return out
