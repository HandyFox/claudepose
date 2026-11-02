"""Measures cut in two at an image edge are rejoined; numbering for Claude is preserved."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scorescribe.model import assemble, filler  # noqa: E402

A = {"measures": [
    {"time": "4/4", "key": 2, "staves": [{"staff": 1, "clef": "G2", "voices": ["D4+F4/1"]}, {"staff": 2, "clef": "F4", "voices": ["D3/1"]}]},
    {"incomplete": True, "staves": [{"staff": 1, "voices": ["D4+F4/2.[lyric=us]"]}, {"staff": 2, "voices": ["D3+A3/2."]}]}]}
B = {"uncertain": [{"measure": 1, "note": "x"}, {"measure": 2, "note": "y"}], "measures": [
    {"key": 2, "incomplete": True, "staves": [{"staff": 1, "clef": "G2", "voices": ["D4+F4/4[lyric=That]"]}, {"staff": 2, "clef": "F4", "voices": ["F3+A3/4", "D3/4"]}]},
    {"staves": [{"staff": 1, "voices": ["B4/1"]}, {"staff": 2, "voices": ["G3/1"]}]}]}
doc = assemble([A, B], [{"name": "Piano", "staves": 2}])
errs = [i for i in doc.issues if i.level == "error"]
assert not errs, [(i.where(), i.msg) for i in errs]
assert len(doc.measures) == 3, len(doc.measures)
m = doc.measures[1]
assert (m.system, m.idx, m.incomplete) == (1, 2, False), (m.system, m.idx, m.incomplete)
assert len(m.staves[2].voices) == 2  # second voice from image B padded at the front
assert doc.measures[2].idx == 2      # keeps Claude's own numbering for image B
uns = sorted((i.system, i.measure) for i in doc.issues if i.level == "unsure")
assert uns == [(1, 2), (2, 2)], uns
# lengths that don't add up to a bar are NOT joined
B2 = dict(B, measures=[dict(B["measures"][0], staves=[{"staff": 1, "voices": ["D4/2"]}, {"staff": 2, "voices": ["D3/2"]}])] + B["measures"][1:])
doc2 = assemble([A, B2], [{"name": "Piano", "staves": 2}])
assert len(doc2.measures) == 4
assert filler(3) == "s/2 s/4" and filler(0.5 if False else __import__("fractions").Fraction(3, 4)) == "s/8 s/16"
# pickup at the very start stays a pickup
P = {"measures": [{"time": "3/4", "key": 0, "incomplete": True, "staves": [{"staff": 1, "clef": "G2", "voices": ["C4/4"]}]},
                  {"staves": [{"staff": 1, "voices": ["C4/2."]}]}]}
d3 = assemble([P], [{"name": "V", "staves": 1}])
assert d3.measures[0].number == 0 and d3.measures[0].incomplete
print("STITCH OK")
