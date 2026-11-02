"""API route for recordings: draft -> check -> review -> fix, with a stand-in model and stand-in Demucs/Whisper."""
import copy
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scorescribe.song.analyze as A  # noqa: E402
from scorescribe.pipeline import Transcriber  # noqa: E402
from scorescribe.song import leadsheet as L  # noqa: E402
from scorescribe.song.run import transcribe_song  # noqa: E402

S = sys.argv[1]  # folder made by make_test_song.py
A._separate = lambda wav, wd, log: (Path(S, "vocals.wav"), Path(S, "no_vocals.wav"), True)
A._words = lambda v, log, m, l: (json.load(open(Path(S, "whisper_words.json"))), "en")
truth = dict(json.load(open(Path(S, "truth.json"))), clef="G2", beat_ql=1.0, downbeat_beat=3, mode="major")
truth.pop("first_beat_time", None)
seen = []


class Block(types.SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


def fake_create(self, messages):
    assert self.tool["name"] == "submit_leadsheet" and "LEAD SHEET" in self.system_prompt
    last = messages[-1]["content"][-1]["text"]
    if len(messages) == 1:
        assert "Beat by beat" in last and "Words heard" in last
        seen.append("draft")
        ls = copy.deepcopy(truth)
        ls["measures"][1]["chords"][0]["symbol"] = "C"            # wrong chord: should be Em
        ls["measures"][6]["melody"] = "F#4/2[lyric=while] A4/4[lyric=we]"  # rhythm error
    elif last.startswith("Second pass"):
        seen.append("review")
        assert "C fits the audio less well" in last and "Em" in last, last[:600]
        assert "voice adds up to 3" in last
        ls = copy.deepcopy(truth)
        ls["changes"] = "measure 2 is Em (bass E, E-G-B measured); fixed measure 7 rhythm"
    else:
        seen.append("fix")
        ls = copy.deepcopy(truth)
    blk = Block(type="tool_use", id=f"tu{len(messages)}", name="submit_leadsheet", input=ls)
    return types.SimpleNamespace(content=[blk], usage=types.SimpleNamespace(
        input_tokens=4000, output_tokens=900, cache_read_input_tokens=0, cache_creation_input_tokens=0))


Transcriber._create = fake_create
with tempfile.TemporaryDirectory() as wd:
    out = os.path.join(wd, "song.musicxml")
    r = transcribe_song(str(Path(S, "mix.wav")), wd, out, api_key="test", log=print)
    assert seen == ["draft", "review"], seen
    assert r["stats"]["chords_fitting"] == "9/9" and r["stats"]["melody_matching"] == "22/22", r["stats"]
    x = open(out).read()
    assert x.count("<harmony") == 9 and x.count("<lyric") == 22
    assert "Sing a simple song now" in open(os.path.join(wd, "chords.txt")).read()
    # resume: analysis is reused, not redone
    msgs = []
    seen.clear()
    transcribe_song(str(Path(S, "mix.wav")), wd, out, api_key="test", log=msgs.append, passes=1)
    assert any("Reusing the earlier audio analysis" in m for m in msgs) and seen == ["draft", "fix"], (seen, msgs[:3])
print("SONG API ROUTE OK")
