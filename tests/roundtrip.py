"""Round-trip check: MusicXML -> tokens -> assemble -> MusicXML; compare sounding pitches and rhythms."""
import sys
import tempfile
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from music21 import converter, stream  # noqa: E402
from scorescribe.draft import score_to_systems  # noqa: E402
from scorescribe.model import assemble  # noqa: E402
from scorescribe.build import write_musicxml  # noqa: E402


def signature(path):
    s = converter.parse(path)
    out = []
    for p in s.parts:
        for m in p.getElementsByClass(stream.Measure):
            ev = []
            for n in m.recurse().notesAndRests:
                if n.duration.isGrace:
                    continue
                if n.isRest:
                    continue
                ps = tuple(sorted(int(x.ps) for x in n.pitches))
                ev.append((round(float(n.getOffsetInHierarchy(m)), 3), ps, round(float(n.quarterLength), 3)))
            out.append(sorted(ev))
    return out


def main(path):
    meta, systems = score_to_systems(path)
    doc = assemble(systems, meta["parts"], meta["title"], meta["composer"])
    errs = [i for i in doc.issues if i.level == "error"]
    for e in errs[:10]:
        print("  issue:", e.system, e.where(), e.msg)
    with tempfile.TemporaryDirectory() as td:
        out = write_musicxml(doc, os.path.join(td, "o.musicxml"))
        a, b = signature(path), signature(out)
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"{os.path.basename(path)}: {len(systems)} systems, {len(doc.measures)} measures, "
          f"{len(errs)} check errors, {len(bad)}/{len(a)} staff-measures differ (len {len(a)} vs {len(b)})")
    for i in bad[:3]:
        print("   truth:", a[i][:6])
        print("   built:", b[i][:6])
    return len(bad) == 0 and len(a) == len(b)


if __name__ == "__main__":
    ok = all([main(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
