"""Runs the full pipeline with a fake Claude that answers from ground truth (with injected mistakes)."""
import copy
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scorescribe.draft import score_to_systems  # noqa: E402
from scorescribe.pipeline import Transcriber  # noqa: E402

pdf, truth, chunks = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
meta, systems = score_to_systems(truth, use_breaks=False)
allm = systems[0]["measures"]
truth_sys, i = [], 0
for c in chunks:
    truth_sys.append({"measures": copy.deepcopy(allm[i:i + c])})
    i += c
truth_sys[0]["parts"] = meta["parts"]
truth_sys[0]["title"] = "Maple Leaf Rag"
for k, sy in enumerate(truth_sys):  # every system restates clefs + key
    for st in sy["measures"][0]["staves"]:
        st.setdefault("clef", "G2" if st["staff"] == 1 else "F4")
    sy["measures"][0].setdefault("key", -4)


class Block(types.SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


calls = []


def fake_create(self, messages):
    first_user = messages[0]["content"]
    text = first_user[-1]["text"]
    idx = int(text.split("This is system ")[1].split(" ")[0])
    last = messages[-1]["content"][-1].get("text", "")
    kind = "first" if len(messages) == 1 else ("repair" if "Automatic checks" in last else "proof")
    calls.append((idx, kind))
    res = copy.deepcopy(truth_sys[idx - 1])
    if kind == "first" and idx == 2:
        v = res["measures"][1]["staves"][0]["voices"][0].split()
        res["measures"][1]["staves"][0]["voices"][0] = " ".join(v[:-1])  # drop a note -> rhythm error
    if kind == "first" and idx == 3:
        res["measures"][0]["staves"][1]["voices"][0] += " C4/4"  # overfull
    if kind == "proof" and idx == 4:
        res["uncertain"] = [{"measure": 2, "staff": 1, "note": "smudged beam"}]
        res["changes"] = "marked m2 uncertain"
    if kind == "repair":
        assert "ERROR" in last, last
        res["changes"] = "fixed rhythm"
    blk = Block(type="tool_use", id=f"tu_{idx}_{len(messages)}", name="submit_transcription", input=res)
    usage = types.SimpleNamespace(input_tokens=1000, output_tokens=300, cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0)
    return types.SimpleNamespace(content=[blk], usage=usage)


Transcriber._create = fake_create
t = Transcriber(api_key="x", workers=3, passes=1)
with tempfile.TemporaryDirectory() as wd:
    out = os.path.join(wd, "out.musicxml")
    r = t.run([pdf], wd, out)
    print(json.dumps(r, indent=1))
    print(open(os.path.join(wd, "report.md")).read())
    assert r["red"] == [], r
    assert r["orange"] == [19], r
    print("calls:", sorted(calls))
    # resume: nothing should be re-read in the first pass
    calls.clear()
    t2 = Transcriber(api_key="x", workers=3, proofread=False, passes=1)
    r2 = t2.run([pdf], wd, out)
    assert not calls, calls
print("MOCK PIPELINE OK")
