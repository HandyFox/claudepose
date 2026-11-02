"""Shape-note conversion: movable do from the key signature, mid-piece key changes, .mxl input."""
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scorescribe.build import write_musicxml  # noqa: E402
from scorescribe.model import assemble  # noqa: E402
from scorescribe.shapes import convert_file, shape_for  # noqa: E402

# key-by-key spot checks
assert shape_for("D", 2, "aiken") == "do" and shape_for("C", 2, "aiken") == "ti"       # D major
assert shape_for("E", -3, "aiken") == "do" and shape_for("C", -3, "aiken") == "la"     # Eb major / C minor (la-based)
assert shape_for("F", 6, "aiken") == "do" and shape_for("E", 6, "aiken") == "ti"       # F# major
assert shape_for("C", -7, "aiken") == "do"                                            # Cb major
assert [shape_for(s, 0, "4shape") for s in "CDEFGAB"] == ["fa", "so", "la", "fa", "so", "la", "mi"]

# a score with a key change and a two-staff part (G major -> Bb major)
sy = [{"measures": [
    {"time": "4/4", "key": 1, "staves": [{"staff": 1, "clef": "G2", "voices": ["G4/4 A4/4 B4/4 F#4/4"]},
                                          {"staff": 2, "clef": "F4", "voices": ["G2+D3/1"]}]},
    {"key": -2, "staves": [{"staff": 1, "voices": ["Bb4/4 C5/4 D5/4 r/4"]}, {"staff": 2, "voices": ["Bb2/1"]}]}]}]
with tempfile.TemporaryDirectory() as td:
    out = write_musicxml(assemble(sy, [{"name": "Piano", "staves": 2}]), os.path.join(td, "s.musicxml"), shape_notes="aiken")
    heads = [(n.find("pitch").findtext("step"), n.findtext("notehead")) for n in ET.parse(out).getroot().iter("note")
             if n.find("pitch") is not None]
    assert heads == [("G", "do"), ("A", "re"), ("B", "mi"), ("F", "ti"), ("G", "do"), ("D", "so"),
                     ("B", "do"), ("C", "re"), ("D", "mi"), ("B", "do")], heads
    # compressed .mxl from the music21 corpus (C major Mozart)
    from music21 import corpus
    src = str(corpus.getWork("mozart/k545/movement1_exposition.mxl"))
    n = convert_file(src, os.path.join(td, "k545_aiken.musicxml"), "aiken")
    first = next(x for x in ET.parse(os.path.join(td, "k545_aiken.musicxml")).getroot().iter("note") if x.find("pitch") is not None)
    assert n > 100 and (first.find("pitch").findtext("step"), first.findtext("notehead")) == ("C", "do"), (n, ET.tostring(first)[:200])
print("SHAPES OK")
