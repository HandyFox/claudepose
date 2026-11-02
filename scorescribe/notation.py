"""The compact token notation Claude writes, and the parser/validator for it.

Claude transcribes what it *sees*: letter, octave and any printed accidental.
Key signatures, accidentals carried through a measure and ties are resolved
here in code, which is far more reliable than asking a model to do it.

Token grammar (one token = one note, chord or rest, tokens separated by spaces):

    [g:|gs:] BODY [/DUR[.][tN[:M]]] [~] [[mod,mod,...]]

    BODY   C4 | F#5 | Bb3 | En4 (printed natural) | C4+E4+G4 (chord)
           r (rest) | s (invisible spacer rest) | R (whole-measure rest)
    DUR    1 2 4 8 16 32 64 128 (0 = breve); dots follow; tN = tuplet
    ~      tie to the next note (on a chord: all notes; or per pitch: C4~+E4)
    g: gs: grace note / slashed grace note (acciaccatura)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

DUR = {
    "0": Fraction(8), "1": Fraction(4), "2": Fraction(2), "4": Fraction(1),
    "8": Fraction(1, 2), "16": Fraction(1, 4), "32": Fraction(1, 8),
    "64": Fraction(1, 16), "128": Fraction(1, 32),
}
DUR_TYPE = {
    "0": "breve", "1": "whole", "2": "half", "4": "quarter", "8": "eighth",
    "16": "16th", "32": "32nd", "64": "64th", "128": "128th",
}
ACC_ALTER = {"##": 2, "#": 1, "n": 0, "b": -1, "bb": -2}

_PITCH = r"[A-G](?:##|#|bb|b|n)?!?\d~?"
TOKEN_RE = re.compile(
    r"^(?P<grace>gs?:)?"
    r"(?P<body>R|r|s|" + _PITCH + r"(?:\+" + _PITCH + r")*)"
    r"(?:/(?P<dur>128|64|32|16|8|4|2|1|0)(?P<dots>\.*)(?:t(?P<tn>\d+)(?::(?P<tm>\d+))?)?)?"
    r"(?P<tie>~)?"
    r"(?:\[(?P<mods>[^\]]*)\])?$"
)
PITCH_RE = re.compile(r"^(?P<step>[A-G])(?P<acc>##|#|bb|b|n)?(?P<force>!)?(?P<oct>\d)(?P<tie>~)?$")

ARTICULATIONS = {
    "staccato": "staccato", "stacc": "staccato", ".": "staccato",
    "staccatissimo": "staccatissimo", "accent": "accent", ">": "accent",
    "tenuto": "tenuto", "-": "tenuto", "marcato": "marcato", "^": "marcato",
    "fermata": "fermata", "trill": "trill", "tr": "trill", "mordent": "mordent",
    "turn": "turn", "arpeggio": "arpeggio", "upbow": "upbow", "downbow": "downbow",
    "harmonic": "harmonic", "breath": "breath",
}
DYNAMICS = {
    "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff",
    "sf", "sfz", "sffz", "fp", "rfz", "rf", "sfp", "fz",
}
SPANNERS = {
    "slur_start": "slur_start", "(": "slur_start",
    "slur_end": "slur_end", ")": "slur_end",
    "cresc_start": "cresc_start", "dim_start": "dim_start",
    "decresc_start": "dim_start", "hairpin_end": "hairpin_end",
    "8va": "8va", "8vb": "8vb", "15ma": "15ma", "15mb": "15mb",
    "ped_start": "ped_start", "ped_end": "ped_end",
}
SHARP_ORDER = "FCGDAEB"


@dataclass
class P:
    """One written pitch."""
    step: str
    octave: int
    acc: str | None  # printed accidental, None if none printed
    tie: bool = False
    alter: int = 0  # resolved later
    show: bool = False  # whether engraving needs an accidental here (resolved later)
    force: bool = False  # courtesy accidental printed in the source (written F#!4)


@dataclass
class Event:
    kind: str  # note | chord | rest | space | mrest
    pitches: list[P] = field(default_factory=list)
    dur_code: str | None = None
    dots: int = 0
    tuplet: tuple[int, int] | None = None
    grace: str | None = None  # None | 'grace' | 'slash'
    tie_all: bool = False
    articulations: list[str] = field(default_factory=list)
    dynamics: list[str] = field(default_factory=list)
    spanners: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    lyrics: list[tuple[int, str]] = field(default_factory=list)
    fingerings: list[str] = field(default_factory=list)
    raw: str = ""
    ql: Fraction = Fraction(0)  # sounding length in quarter notes (0 for grace)

    @property
    def is_grace(self) -> bool:
        return self.grace is not None


class TokenError(ValueError):
    pass


def default_tuplet_normal(n: int) -> int:
    if n == 2:
        return 3
    if n == 4:
        return 3
    m = 1
    while m * 2 < n:
        m *= 2
    return m


def parse_token(tok: str, unknown_mods: list[str] | None = None) -> Event:
    m = TOKEN_RE.match(tok)
    if not m:
        raise TokenError(f"cannot parse token '{tok}'")
    body = m.group("body")
    ev = Event(kind="note", raw=tok)
    g = m.group("grace")
    if g:
        ev.grace = "slash" if g == "gs:" else "grace"
    if body == "R":
        ev.kind = "mrest"
    elif body in ("r", "s"):
        ev.kind = "rest" if body == "r" else "space"
    else:
        parts = body.split("+")
        for pt in parts:
            pm = PITCH_RE.match(pt)
            if not pm:
                raise TokenError(f"bad pitch '{pt}' in '{tok}'")
            ev.pitches.append(P(pm.group("step"), int(pm.group("oct")), pm.group("acc"), bool(pm.group("tie")),
                                force=bool(pm.group("force") and pm.group("acc"))))
        ev.kind = "chord" if len(parts) > 1 else "note"

    dur = m.group("dur")
    if ev.kind != "mrest":
        if dur is None:
            raise TokenError(f"missing duration in '{tok}' (e.g. '{body}/4')")
        ev.dur_code = dur
        ev.dots = len(m.group("dots") or "")
        if ev.dots > 3:
            raise TokenError(f"too many dots in '{tok}'")
        if m.group("tn"):
            n = int(m.group("tn"))
            mm = int(m.group("tm")) if m.group("tm") else default_tuplet_normal(n)
            if n < 2 or mm < 1:
                raise TokenError(f"bad tuplet in '{tok}'")
            ev.tuplet = (n, mm)
        q = DUR[dur]
        q = q * (2 - Fraction(1, 2 ** ev.dots))
        if ev.tuplet:
            q = q * Fraction(ev.tuplet[1], ev.tuplet[0])
        ev.ql = Fraction(0) if ev.grace else q
    elif dur is not None:
        ev.dur_code = dur  # 'R/1' tolerated; length comes from the time signature

    if m.group("tie"):
        ev.tie_all = True
        for p in ev.pitches:
            p.tie = True

    mods = m.group("mods")
    if mods:
        for raw in mods.split(","):
            mod = raw.strip()
            if not mod:
                continue
            low = mod.lower()
            if low in ARTICULATIONS:
                ev.articulations.append(ARTICULATIONS[low])
            elif mod in DYNAMICS:
                ev.dynamics.append(mod)
            elif low in SPANNERS:
                ev.spanners.append(SPANNERS[low])
            elif low.startswith("text="):
                ev.texts.append(mod[5:].replace("_", " "))
            elif re.match(r"^lyric\d?=", low):
                num = int(low[5]) if low[5].isdigit() else 1
                ev.lyrics.append((num, mod.split("=", 1)[1].replace("_", " ")))
            elif low.startswith("fing="):
                ev.fingerings.append(mod[5:])
            elif unknown_mods is not None:
                unknown_mods.append(mod)
    return ev


def parse_voice(text: str, unknown_mods: list[str] | None = None) -> tuple[list[Event], list[str]]:
    events, errors = [], []
    for tok in text.split():
        try:
            events.append(parse_token(tok, unknown_mods))
        except TokenError as e:
            errors.append(str(e))
    return events, errors


# ---------------------------------------------------------------- signatures

def parse_time(ts) -> tuple[str, Fraction] | None:
    """Returns (music21 time string, measure length in quarters)."""
    if ts is None:
        return None
    s = str(ts).strip().replace(" ", "")
    if not s:
        return None
    if s.lower() in ("c", "common"):
        return "4/4|common", Fraction(4)   # shown as the C symbol
    if s.lower() in ("c|", "cut", "alla_breve", "¢"):
        return "2/2|cut", Fraction(2)      # shown as the cut-time symbol
    m = re.match(r"^([\d+]+)/(\d+)$", s)
    if not m:
        return None
    num = sum(int(x) for x in m.group(1).split("+") if x)
    den = int(m.group(2))
    if num <= 0 or den not in (1, 2, 4, 8, 16, 32, 64):
        return None
    return s, Fraction(num * 4, den)


KEY_NAMES = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}


def parse_key(k) -> int | None:
    if k is None or k == "":
        return None
    if isinstance(k, bool):
        return None
    if isinstance(k, (int, float)):
        v = int(k)
        return v if -7 <= v <= 7 else None
    s = str(k).strip()
    m = re.match(r"^([+-]?\d)$", s)
    if m:
        return parse_key(int(s))
    m = re.match(r"^(\d)\s*(#|sharps?)$", s)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d)\s*(b|flats?)$", s)
    if m:
        return -int(m.group(1))
    m = re.match(r"^([A-G](?:#|b)?)", s)
    if m:
        name = m.group(1)
        minor = "minor" in s.lower() or s.endswith("m")
        if minor:
            rel = {"A": "C", "E": "G", "B": "D", "F#": "A", "C#": "E", "G#": "B", "D#": "F#",
                   "A#": "C#", "D": "F", "G": "Bb", "C": "Eb", "F": "Ab", "Bb": "Db",
                   "Eb": "Gb", "Ab": "Cb"}
            name = rel.get(name, name)
        return KEY_NAMES.get(name)
    return None


def key_alters(fifths: int) -> dict[str, int]:
    alters = {s: 0 for s in "ABCDEFG"}
    if fifths > 0:
        for s in SHARP_ORDER[:fifths]:
            alters[s] = 1
    elif fifths < 0:
        for s in SHARP_ORDER[::-1][: -fifths]:
            alters[s] = -1
    return alters


CLEFS = {"G2", "G1", "F4", "F3", "F5", "C1", "C2", "C3", "C4", "C5", "G2-8", "G2+8", "F4-8", "F4+8", "perc", "tab"}


def norm_clef(c) -> str | None:
    if not c:
        return None
    s = str(c).strip()
    aliases = {"treble": "G2", "bass": "F4", "alto": "C3", "tenor": "C4", "soprano": "C1",
               "treble8vb": "G2-8", "treble_8vb": "G2-8", "tenor_voice": "G2-8", "percussion": "perc"}
    s = aliases.get(s.lower(), s)
    return s if s in CLEFS else None
