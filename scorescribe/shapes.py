"""Shape-note (Aiken 7-shape or 4-shape fa-so-la) noteheads for any MusicXML file.

MusicXML has the shapes built in (<notehead>do</notehead> ... ti), and MuseScore
imports them as its shape-note heads. Shapes follow movable do: do is the tonic
of the major key the key signature names, so minor keys are la-based, as in
shape-note hymnals. Altered notes keep the shape of their letter name.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

STEPS = "CDEFGAB"
# major tonic letter for each key signature (-7 .. +7 fifths)
TONIC = {f: n[0] for f, n in zip(range(-7, 8),
                                 ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"])}
SCHEMES = {
    "aiken": ["do", "re", "mi", "fa", "so", "la", "ti"],
    # 4-shape (Sacred Harp style): fa so la fa so la mi
    "4shape": ["fa", "so", "la", "fa", "so", "la", "mi"],
}
ALIASES = {"aiken": "aiken", "7": "aiken", "7shape": "aiken", "seven": "aiken",
           "4shape": "4shape", "4": "4shape", "four": "4shape", "sacredharp": "4shape", "fasola": "4shape"}
# <notehead> must come before these children of <note>
_AFTER = ("notehead-text", "staff", "beam", "notations", "lyric", "play", "listen")


def normalize(scheme: str | None) -> str | None:
    if not scheme:
        return None
    s = str(scheme).lower().replace("-", "").replace(" ", "").replace("_", "")
    if s in ("normal", "none", "round", "off"):
        return None
    if s not in ALIASES:
        raise ValueError(f"unknown shape-note scheme '{scheme}' (use aiken or 4shape)")
    return ALIASES[s]


def shape_for(step: str, fifths: int, scheme: str) -> str:
    tonic = TONIC.get(max(-7, min(7, fifths)), "C")
    degree = (STEPS.index(step) - STEPS.index(tonic)) % 7
    return SCHEMES[scheme][degree]


def apply_shape_notes(path: str, scheme: str, out_path: str | None = None) -> int:
    """Rewrite noteheads in a MusicXML file (in place, or to out_path). Returns notes changed."""
    scheme = normalize(scheme)
    if scheme is None:
        return 0
    with open(path, "rb") as f:
        raw = f.read()
    start = raw.find(b"<score-partwise")
    if start < 0:
        raise ValueError("not an uncompressed score-partwise MusicXML file")
    header = raw[:start]
    root = ET.fromstring(raw)
    changed = 0
    for part in root.iter("part"):
        keys: dict[int, int] = {}   # staff number -> fifths (0 = applies to all)
        for measure in part.iter("measure"):
            for el in list(measure):
                if el.tag == "attributes":
                    for k in el.findall("key"):
                        fifths = k.findtext("fifths")
                        if fifths is None:
                            continue
                        num = k.get("number")
                        if num:
                            keys[int(num)] = int(fifths)
                        else:
                            keys = {0: int(fifths)}
                elif el.tag == "note":
                    p = el.find("pitch")
                    if p is None or el.find("rest") is not None:
                        continue
                    step = p.findtext("step")
                    if not step or step not in STEPS:
                        continue
                    staff = int(el.findtext("staff", "1") or 1)
                    fifths = keys.get(staff, keys.get(0, 0))
                    shape = shape_for(step, fifths, scheme)
                    nh = el.find("notehead")
                    if nh is None:
                        nh = ET.Element("notehead")
                        kids = list(el)
                        pos = next((i for i, c in enumerate(kids) if c.tag in _AFTER), len(kids))
                        el.insert(pos, nh)
                    elif nh.text and nh.text not in ("normal", "do", "re", "mi", "fa", "fa up", "so", "la", "ti"):
                        continue  # keep special heads (x, slash, diamond harmonics ...)
                    nh.text = shape
                    changed += 1
    body = ET.tostring(root, encoding="utf-8")
    if body.startswith(b"<?xml"):
        body = body[body.find(b"?>") + 2:].lstrip()
    with open(out_path or path, "wb") as f:
        f.write((header or b'<?xml version="1.0" encoding="utf-8"?>\n') + body)
    return changed


def convert_file(src: str, dst: str, scheme: str) -> int:
    """Convert any MusicXML (.musicxml/.xml or compressed .mxl) to shape notes."""
    import tempfile
    import zipfile
    from pathlib import Path

    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            name = None
            if "META-INF/container.xml" in z.namelist():
                c = ET.fromstring(z.read("META-INF/container.xml"))
                rf = next((e for e in c.iter() if e.tag.endswith("rootfile")), None)
                name = rf.get("full-path") if rf is not None else None
            name = name or next(n for n in z.namelist() if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF"))
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "score.musicxml"
                tmp.write_bytes(z.read(name))
                n = apply_shape_notes(str(tmp), scheme)
                Path(dst).write_bytes(tmp.read_bytes())
                return n
    return apply_shape_notes(src, scheme, dst)
