"""Four-part arrangement: part-writing checks, the API route's arrange/review passes, and the agent route."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scorescribe.song.analyze as A  # noqa: E402
from scorescribe.pipeline import Transcriber  # noqa: E402
from scorescribe.song import satb  # noqa: E402
from scorescribe.song.run import prepare, transcribe_song  # noqa: E402

S = sys.argv[1]
A._separate = lambda wav, wd, log: (Path(S, "vocals.wav"), Path(S, "no_vocals.wav"), True)
A._words = lambda v, log, m, l: (json.load(open(Path(S, "whisper_words.json"))), "en")
truth = dict(json.load(open(Path(S, "truth.json"))), clef="G2", beat_ql=1.0, downbeat_beat=3, mode="major")
truth.pop("first_beat_time", None)
DRAFT = {"measures": [
    {"alto": "G4/4 G4/4 G4/4 G4/4", "tenor": "D4/4 D4/4 B3/4 D4/4", "bass": "G2/4 G2/4 G2/4 G2/4"},
    {"alto": "E4/2 E4/2", "tenor": "B3/2 G3/2", "bass": "E3/2 E3/2"},
    {"alto": "C4/4 E4/4 E4/4 G4/4", "tenor": "G3/4 G3/4 G3/4 C4/4", "bass": "C3/4 C3/4 C3/4 C3/4"},
    {"alto": "F#4/2. r/4", "tenor": "A3/2. r/4", "bass": "D3/2. r/4"},
    {"alto": "G4/4 G4/4 D4/4 G4/4", "tenor": "B3/4 D4/4 B3/4 D4/4", "bass": "G2/4 G2/4 G2/4 G2/4"},
    {"alto": "G4/4 G4/4 E4/4 C4/4", "tenor": "E4/4 C4/4 C4/4 G3/4", "bass": "C3/4 C3/4 C3/4 C3/4"},
    {"alto": "D4/2 C4/2", "tenor": "A3/2 F#3/2", "bass": "D3/2 D3/2"},
    {"alto": "B3/1", "tenor": "G3/1", "bass": "G2/1"}]}
FIXED = copy.deepcopy(DRAFT)
FIXED["measures"][1]["alto"] = "E4/2 B3/2"
FIXED["measures"][5]["alto"] = "G4/4 G4/4 E4/4 E4/4"
FIXED["measures"][5]["tenor"] = "E4/4 C4/4 C4/4 C4/4"

# 1) the checker: draft has 2x parallel octaves + 1x parallel fifths; fixed version is clean
d, f = satb.check(truth, DRAFT), satb.check(truth, FIXED)
assert len([c for c in d["checks"] if "parallel" in c]) == 3, d["checks"]
assert not f["errors"] and not f["checks"], f["text"]

# 2) API route: lead sheet passes, then arrange -> part-writing review -> clean
seen = []


class Block(types.SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return dict(vars(self))


def fake_create(self, messages):
    last = messages[-1]["content"][-1]["text"]
    if self.tool["name"] == "submit_leadsheet":
        seen.append("leadsheet")
        out = copy.deepcopy(truth)
    else:
        assert "four-part choir" in self.system_prompt
        if len(messages) == 1:
            assert '"melody"' in last and "submit_arrangement" in last
            seen.append("arrange")
            out = copy.deepcopy(DRAFT)
        else:
            assert last.startswith("Part-writing review") and "parallel fifths between tenor and bass" in last, last[:400]
            seen.append("review")
            out = dict(copy.deepcopy(FIXED), changes="contrary motion against the bass into bars 3 and 7")
    return types.SimpleNamespace(content=[Block(type="tool_use", id=f"t{len(messages)}", name=self.tool["name"], input=out)],
                                 usage=types.SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                                                             cache_creation_input_tokens=0))


Transcriber._create = fake_create
with tempfile.TemporaryDirectory() as wd:
    out = os.path.join(wd, "song.musicxml")
    r = transcribe_song(str(Path(S, "mix.wav")), wd, out, api_key="x", satb=True, log=lambda *a: None)
    assert seen[-3:] == ["leadsheet", "arrange", "review"] or seen[-2:] == ["arrange", "review"], seen
    assert r["satb"] and r["satb_flags"] == 0 and os.path.exists(os.path.join(wd, "song_leadsheet.musicxml")), r
    x = open(out).read()
    assert x.count("<part ") == 1 and x.count("<staff>2</staff>") > 10 and x.count("<lyric") == 22 and "<harmony" not in x

# 3) subscription route: prepare with SATB, agent writes the files, check + build via the commands in CLAUDE.md
with tempfile.TemporaryDirectory() as wd:
    prepare(str(Path(S, "mix.wav")), wd, stem="hymn", satb=True, log=lambda *a: None)
    md = open(os.path.join(wd, "CLAUDE.md")).read()
    assert "arrangement.json" in md and "Arranging guide" in md and "hymn_leadsheet.musicxml" in md
    json.dump(truth, open(os.path.join(wd, "leadsheet.json"), "w"))
    cmd = [sys.executable, "-m", "scorescribe"]
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    r1 = subprocess.run(cmd + ["song-build", "."], cwd=wd, capture_output=True, text=True, env=env)
    assert r1.returncode == 1 and "hasn't been written yet" in r1.stdout, r1.stdout  # refuses without the arrangement
    json.dump(DRAFT, open(os.path.join(wd, "arrangement.json"), "w"))
    r2 = subprocess.run(cmd + ["song-check", "."], cwd=wd, capture_output=True, text=True, env=env)
    assert "parallel octaves" in r2.stdout and "Four-part arrangement check" in r2.stdout, r2.stdout[-800:]
    json.dump(FIXED, open(os.path.join(wd, "arrangement.json"), "w"))
    r3 = subprocess.run(cmd + ["song-build", "."], cwd=wd, capture_output=True, text=True, env=env)
    assert r3.returncode == 0 and os.path.exists(os.path.join(wd, "hymn.musicxml")) and \
        os.path.exists(os.path.join(wd, "hymn_leadsheet.musicxml")), r3.stdout + r3.stderr
print("SATB OK")
