"""Recording -> lead sheet: the API route (automatic passes) and the subscription route (agent instructions)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import leadsheet as L
from . import prompts as P
from . import satb as SATB
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


def finish(ls: dict, analysis: dict, wd: Path, out_path: str, shape_notes: str | None, log, usage: str = "",
           arr: dict | None = None, satb_chords: bool = False) -> dict:
    """Write the outputs. With an arrangement, out_path is the SATB hymn score and the lead sheet goes
    next to it as <stem>_leadsheet.musicxml; otherwise out_path is the lead sheet."""
    (wd / "leadsheet.json").write_text(json.dumps(ls, indent=1, ensure_ascii=False), encoding="utf-8")
    outp = Path(out_path)
    lead_path = str(outp.with_name(outp.stem + "_leadsheet" + outp.suffix)) if arr else out_path
    lead = L.write(ls, lead_path, shape_notes)
    satb_out, sr = None, None
    if arr:
        (wd / "arrangement.json").write_text(json.dumps(arr, indent=1, ensure_ascii=False), encoding="utf-8")
        satb_out = SATB.write(ls, arr, out_path, shape_notes, chord_symbols=satb_chords)
        sr = SATB.check(ls, arr)
    chart = L.chord_chart(ls)
    (wd / "chords.txt").write_text(chart, encoding="utf-8")
    r = L.check(ls, analysis)
    unsure = [f"- measure {u.get('measure')}: {u.get('note')}" for u in ls.get("uncertain") or []]
    rep = [f"# {ls.get('title') or 'Song'} - transcription report", ""]
    if satb_out:
        rep += [f"Four-part (SATB) score: {satb_out}"]
    rep += [f"Lead sheet: {lead}", f"Chord chart: {wd / 'chords.txt'}", ""]
    if usage:
        rep += [f"Usage: {usage}", ""]
    rep += [r["text"], ""]
    if sr:
        rep += [sr["text"], ""]
    rep += [f"## Unsure: {len(unsure)}"] + unsure
    (wd / "report.md").write_text("\n".join(rep), encoding="utf-8")
    log("")
    if satb_out:
        log(f"Done. Four-part score: {satb_out}")
        log(f"  part-writing: {len(sr['checks'])} thing(s) still flagged; chord tones {sr['stats'].get('chord_tones', '?')}")
        log(f"  lead sheet: {lead}")
    else:
        log(f"Done. Lead sheet: {lead}")
    log(f"  {r['stats']['measures']} measures; chords fitting the audio {r['stats']['chords_fitting']}, "
        f"melody notes matching {r['stats']['melody_matching']}")
    log(f"  chord chart: {wd / 'chords.txt'}")
    return {"output": satb_out or lead, "leadsheet": lead, "satb": bool(satb_out), "chart": str(wd / "chords.txt"),
            "measures": r["stats"]["measures"], "stats": r["stats"],
            "satb_flags": len(sr["checks"]) if sr else None,
            "orange": [u.get("measure") for u in ls.get("uncertain") or []], "red": [], "report": str(wd / "report.md")}


def arrange(t, ls: dict, wd: Path, passes: int, log) -> dict:
    """Ask the model for alto/tenor/bass, check the part-writing, review, fix."""
    from ..pipeline import Session
    t.system_prompt = P.ARRANGE
    t.tool = {"name": P.ARRANGE_TOOL, "description": "Submit the complete four-part arrangement.",
              "input_schema": SATB.SCHEMA}
    sess = Session(SimpleNamespace(index=0, n_staves=2), b"")
    slim = {k: v for k, v in ls.items() if k not in ("uncertain", "changes")}
    log("Arranging: writing alto, tenor and bass ...")
    arr = t._ask(sess, [{"type": "text", "text": P.ARRANGE_FIRST + json.dumps(slim, ensure_ascii=False)}])
    (wd / "arrangement.json").write_text(json.dumps(arr, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in range(2, max(2, passes) + 1):
        r = SATB.check(ls, arr)
        log(f"  part-writing check: {len(r['errors'])} errors, {len(r['checks'])} problems flagged")
        if not r["errors"] and not r["checks"] and k > 2:
            break
        log("Part-writing review ...")
        arr = t._ask(sess, t._reply_content(sess, P.ARRANGE_REVIEW + r["text"]))
        (wd / "arrangement.json").write_text(json.dumps(arr, indent=1, ensure_ascii=False), encoding="utf-8")
        if arr.get("changes"):
            log("  changes: " + str(arr["changes"])[:300])
    for _ in range(2):
        r = SATB.check(ls, arr)
        if not r["errors"]:
            break
        log(f"Fixing {len(r['errors'])} remaining arrangement error(s) ...")
        arr = t._ask(sess, t._reply_content(sess, P.ARRANGE_REVIEW + r["text"]))
    return arr


# ------------------------------------------------------------------ API route

def transcribe_song(audio: str, workdir: str, out_path: str, model: str = "claude-opus-5-5", api_key: str | None = None,
                    effort: str | None = "high", passes: int = 2, lyrics_text: str | None = None, separate: bool = True,
                    whisper_model: str = "small", language: str | None = None, shape_notes: str | None = None,
                    title: str | None = None, resume: bool = True, satb: bool = False, satb_chords: bool = False,
                    log=print) -> dict:
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
    arr = None
    if satb:
        if L.check(ls, a)["errors"]:
            log("  (skipping the four-part arrangement: the lead sheet still has errors)")
        else:
            arr = arrange(t, ls, wd, passes, log)
    res = finish(ls, a, wd, out_path, shape_notes, log, str(t.usage), arr=arr, satb_chords=satb_chords)
    log(f"  took {time.time() - t0:.0f}s")
    return res


# ------------------------------------------------------------------ subscription route (Claude Code / Codex)

def prepare(audio: str, workdir: str, stem: str | None = None, lyrics_text: str | None = None, separate: bool = True,
            whisper_model: str = "small", language: str | None = None, shape_notes: str | None = None,
            title: str | None = None, satb: bool = False, satb_chords: bool = False, log=print) -> dict:
    wd = Path(workdir)
    a = get_analysis(audio, wd, True, log, separate=separate, whisper_model=whisper_model, language=language,
                     lyrics_text=lyrics_text)
    stem = stem or Path(audio).stem or "song"
    man = {"kind": "song", "stem": stem, "output": f"{stem}.musicxml", "shape_notes": shape_notes, "title": title,
           "satb": bool(satb), "satb_chords": bool(satb_chords)}
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
    if man.get("satb"):
        satb_steps = """3b. **Four-part arrangement (the user asked for SATB).** When the lead sheet checks out, write
   `arrangement.json`: alto, tenor and bass for every measure, following the arranging guide at the end of
   this file. `song-check .` then also checks the part-writing (parallel fifths/octaves, missing thirds,
   notes outside the chord, wrong bass, crossing, spacing, ranges, doubled leading tone). Do a part-writing
   review pass: fix every flagged item unless it's a deliberate exception, sing each voice through in your
   head for smoothness, and check that the cadences resolve. Run the check again until it's clean.
"""
        built = (f"This writes `{man['output']}` (the four-part hymn score: soprano and alto on the treble staff, tenor and "
                 f"bass on the bass staff, lyrics in between), `{Path(man['output']).stem}_leadsheet.musicxml` (melody, "
                 "lyrics and chord symbols) and `chords.txt` (a lyrics-with-chords chart).")
    else:
        satb_steps = ""
        built = (f"This writes `{man['output']}` (MusicXML for MuseScore, with chord symbols and lyrics) and `chords.txt` "
                 "(a lyrics-with-chords chart).")
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
{satb_steps}4. **Build:**

       {cmd} song-build .{shape}

   {built}
   Tell the user they're ready, with a short summary of the song (key, tempo, form) and anything you were
   unsure about.

If you run out of usage partway through, `leadsheet.json` keeps your progress; continue from it next time.

## Guide

{P.GUIDE}
""" + (f"""
## Arranging guide (four-part harmony)

{P.ARRANGE}
""" if man.get("satb") else "")


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
    man = json.loads((wd / "song-manifest.json").read_text()) if (wd / "song-manifest.json").exists() else {}
    errors = list(r["errors"])
    checks = list(r["checks"])
    ap = wd / "arrangement.json"
    if man.get("satb") or ap.exists():
        if not ap.exists():
            log("Four-part arrangement: arrangement.json doesn't exist yet (write it once the lead sheet is clean).")
            if man.get("satb"):
                errors.append("no arrangement.json")
        elif r["errors"]:
            log("Four-part arrangement: fix the lead sheet errors first.")
        else:
            try:
                arr = json.loads(ap.read_text(encoding="utf-8"))
                sr = SATB.check(ls, arr)
                log(sr["text"])
                errors += sr["errors"]
                checks += sr["checks"]
            except json.JSONDecodeError as e:
                log(f"arrangement.json is not valid JSON: {e}")
                errors.append("invalid arrangement.json")
    if not errors:
        shape = f" --shape-notes {man['shape_notes']}" if man.get("shape_notes") else ""
        log(("No errors. " + ("Review the CHECK items above; when you're done, build with:" if checks
                              else "Build with:")) + f"\n  {_python_cmd()} song-build .{shape}")
    return {"errors": errors, "checks": checks}


def build_dir(workdir: str, out: str | None = None, shape_notes: str | None = None, log=print) -> dict:
    wd = Path(workdir)
    a = json.loads((wd / "analysis.json").read_text())
    ls = json.loads((wd / "leadsheet.json").read_text(encoding="utf-8"))
    man = json.loads((wd / "song-manifest.json").read_text()) if (wd / "song-manifest.json").exists() else {}
    errors = list(L.check(ls, a)["errors"])
    arr = None
    ap = wd / "arrangement.json"
    if ap.exists() and not errors:
        arr = json.loads(ap.read_text(encoding="utf-8"))
        errors += SATB.check(ls, arr)["errors"]
    elif man.get("satb") and not ap.exists():
        errors.append("the four-part arrangement (arrangement.json) hasn't been written yet")
    if errors:
        log("Can't build yet - fix these first (run song-check for the full report):")
        for e in errors:
            log("  - " + e)
        return {"errors": errors}
    out = out or str(wd / man.get("output", "song.musicxml"))
    return finish(ls, a, wd, out, shape_notes or man.get("shape_notes"), log, arr=arr,
                  satb_chords=bool(man.get("satb_chords")))
