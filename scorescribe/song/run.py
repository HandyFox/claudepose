"""Recording -> lead sheet: the API route (automatic passes) and the subscription route (agent instructions)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import leadsheet as L
from . import prompts as P
from .analyze import analysis_markdown, analyze


def _python_cmd() -> str:
    exe = sys.executable or "python"
    return f'"{exe}" -m scorescribe' if " " in exe else f"{exe} -m scorescribe"


def get_analysis(audio: str, wd: Path, resume: bool, log, **kw) -> dict:
    p = wd / "analysis.json"
    if resume and p.exists():
        log("Reusing the earlier audio analysis ...")
        a = json.loads(p.read_text())
        if kw.get("lyrics_text") and a.get("lyrics_text") != kw["lyrics_text"]:
            a["lyrics_text"] = kw["lyrics_text"]
            p.write_text(json.dumps(a))
            (wd / "analysis.md").write_text(analysis_markdown(a), encoding="utf-8")
        return a
    return analyze(audio, str(wd), log=log, **kw)


def finish(ls: dict, analysis: dict, wd: Path, out_path: str, shape_notes: str | None, log, usage: str = "") -> dict:
    (wd / "leadsheet.json").write_text(json.dumps(ls, indent=1, ensure_ascii=False), encoding="utf-8")
    out = L.write(ls, out_path, shape_notes)
    chart = L.chord_chart(ls)
    (wd / "chords.txt").write_text(chart, encoding="utf-8")
    r = L.check(ls, analysis)
    unsure = [f"- measure {u.get('measure')}: {u.get('note')}" for u in ls.get("uncertain") or []]
    rep = [f"# {ls.get('title') or 'Song'} - transcription report", "", f"Output: {out}", f"Chord chart: {wd / 'chords.txt'}", ""]
    if usage:
        rep += [f"Usage: {usage}", ""]
    rep += [r["text"], "", f"## Unsure: {len(unsure)}"] + unsure
    (wd / "report.md").write_text("\n".join(rep), encoding="utf-8")
    log("")
    log(f"Done. Lead sheet: {out}")
    log(f"  {r['stats']['measures']} measures; chords fitting the audio {r['stats']['chords_fitting']}, "
        f"melody notes matching {r['stats']['melody_matching']}")
    log(f"  chord chart: {wd / 'chords.txt'}")
    return {"output": out, "chart": str(wd / "chords.txt"), "measures": r["stats"]["measures"], "stats": r["stats"],
            "orange": [u.get("measure") for u in ls.get("uncertain") or []], "red": [], "report": str(wd / "report.md")}


# ------------------------------------------------------------------ API route

def transcribe_song(audio: str, workdir: str, out_path: str, model: str = "claude-opus-5-5", api_key: str | None = None,
                    effort: str | None = "high", passes: int = 2, lyrics_text: str | None = None, separate: bool = True,
                    whisper_model: str = "small", language: str | None = None, shape_notes: str | None = None,
                    title: str | None = None, resume: bool = True, log=print) -> dict:
    from ..pipeline import Session, Transcriber
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a = get_analysis(audio, wd, resume, log, separate=separate, whisper_model=whisper_model, language=language,
                     lyrics_text=lyrics_text)
    t = Transcriber(api_key=api_key, model=model, effort=effort, workers=1, log=log)
    t.system_prompt = P.GUIDE
    t.tool = {"name": P.TOOL_NAME, "description": "Submit the complete lead sheet.", "input_schema": L.SCHEMA}
    sess = Session(SimpleNamespace(index=0, n_staves=1), b"")
    md = (wd / "analysis.md").read_text(encoding="utf-8")
    hint = f"\nThe song's title is \"{title}\".\n" if title else ""
    log(f"Pass 1: {model} writes the lead sheet from the analysis ...")
    ls = t._ask(sess, [{"type": "text", "text": P.FIRST + hint + md}])
    if title and not ls.get("title"):
        ls["title"] = title
    (wd / "leadsheet.json").write_text(json.dumps(ls, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in range(2, passes + 1):
        r = L.check(ls, a)
        log(f"  check: {len(r['errors'])} errors, {len(r['checks'])} things to look at; "
            f"chords fitting {r['stats']['chords_fitting']}, melody matching {r['stats']['melody_matching']}")
        log(f"Pass {k}: harmony and melody review ...")
        ls = t._ask(sess, t._reply_content(sess, P.REVIEW + r["text"]))
        (wd / "leadsheet.json").write_text(json.dumps(ls, indent=1, ensure_ascii=False), encoding="utf-8")
        if ls.get("changes"):
            log("  changes: " + str(ls["changes"])[:300])
    for _ in range(2):
        r = L.check(ls, a)
        if not r["errors"]:
            break
        log(f"Fixing {len(r['errors'])} remaining error(s) ...")
        ls = t._ask(sess, t._reply_content(sess, P.FIX + "\n".join("- " + e for e in r["errors"])))
    res = finish(ls, a, wd, out_path, shape_notes, log, str(t.usage))
    log(f"  took {time.time() - t0:.0f}s")
    return res


# ------------------------------------------------------------------ subscription route (Claude Code / Codex)

def prepare(audio: str, workdir: str, stem: str | None = None, lyrics_text: str | None = None, separate: bool = True,
            whisper_model: str = "small", language: str | None = None, shape_notes: str | None = None,
            title: str | None = None, log=print) -> dict:
    wd = Path(workdir)
    a = get_analysis(audio, wd, True, log, separate=separate, whisper_model=whisper_model, language=language,
                     lyrics_text=lyrics_text)
    stem = stem or Path(audio).stem or "song"
    man = {"kind": "song", "stem": stem, "output": f"{stem}.musicxml", "shape_notes": shape_notes, "title": title}
    (wd / "song-manifest.json").write_text(json.dumps(man, indent=1))
    text = agent_instructions(man)
    (wd / "CLAUDE.md").write_text(text, encoding="utf-8")
    (wd / "AGENTS.md").write_text(text, encoding="utf-8")
    log(f"  ready for your agent in {wd}")
    return {"analysis": a, "manifest": man}


def agent_instructions(man: dict) -> str:
    cmd = _python_cmd()
    shape = f" --shape-notes {man['shape_notes']}" if man.get("shape_notes") else ""
    title = f' The song is called "{man["title"]}".' if man.get("title") else ""
    return f"""# Make a lead sheet of this recording with ScoreScribe

A recording has been analysed by audio software; the results are in `analysis.md` (read ALL of it) and
`analysis.json` (same data, full detail). Make a lead sheet: melody + lyrics + chord symbols.{title}
**You do the musical thinking; ScoreScribe's commands check your work against the audio and build the file.**
Run them from this folder.

## Passes

1. **Draft.** Following the guide below, write the complete lead sheet to `leadsheet.json`.
2. **Check against the audio:**

       {cmd} song-check .

   It reports ERRORS (must fix: rhythms that don't add up, unreadable chords), CHECK items (chords that fit the
   measured harmony or bass worse than an alternative, melody notes that differ from the sung pitch, octave
   slips, rests where singing was measured, words the speech recogniser heard that you left out) and NOTES.
3. **Harmony and melody review (second pass).** For every ERROR and CHECK item, look at those beats in
   `analysis.md` again and fix it, or keep your version if you're confident the measurement misled. Then review
   the whole harmony like a music theorist: does each chord make sense in the key, does the progression move
   logically with cadences at phrase ends, is the harmonic rhythm steady, do repeated sections match, do
   strong-beat melody notes fit the chords? Correct lyric mishearings. Update `leadsheet.json` and run the
   check again. Repeat until there are no ERRORS and every CHECK item is either fixed or deliberately kept.
4. **Build:**

       {cmd} song-build .{shape}

   This writes `{man['output']}` (MusicXML for MuseScore, with chord symbols and lyrics) and `chords.txt`
   (a lyrics-with-chords chart). Tell the user they're ready, with a short summary of the song (key, tempo,
   form) and anything you were unsure about.

If you run out of usage partway through, `leadsheet.json` keeps your progress; continue from it next time.

## Guide

{P.GUIDE}
"""


def check_dir(workdir: str, log=print) -> dict:
    wd = Path(workdir)
    a = json.loads((wd / "analysis.json").read_text())
    p = wd / "leadsheet.json"
    if not p.exists():
        log("leadsheet.json doesn't exist yet: write your draft first.")
        return {"errors": ["no leadsheet.json"]}
    try:
        ls = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log(f"leadsheet.json is not valid JSON: {e}")
        return {"errors": ["invalid json"]}
    r = L.check(ls, a)
    log(r["text"])
    if not r["errors"]:
        man = json.loads((wd / "song-manifest.json").read_text()) if (wd / "song-manifest.json").exists() else {}
        shape = f" --shape-notes {man['shape_notes']}" if man.get("shape_notes") else ""
        log(("No errors. " + ("Review the CHECK items above; when you're done, build with:" if r["checks"]
                              else "Build with:")) + f"\n  {_python_cmd()} song-build .{shape}")
    return r


def build_dir(workdir: str, out: str | None = None, shape_notes: str | None = None, log=print) -> dict:
    wd = Path(workdir)
    a = json.loads((wd / "analysis.json").read_text())
    ls = json.loads((wd / "leadsheet.json").read_text(encoding="utf-8"))
    man = json.loads((wd / "song-manifest.json").read_text()) if (wd / "song-manifest.json").exists() else {}
    r = L.check(ls, a)
    if r["errors"]:
        log("Can't build yet - fix these first (run song-check for the full report):")
        for e in r["errors"]:
            log("  - " + e)
        return {"errors": r["errors"]}
    out = out or str(wd / man.get("output", "song.musicxml"))
    return finish(ls, a, wd, out, shape_notes or man.get("shape_notes"), log)
