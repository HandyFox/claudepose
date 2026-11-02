"""Agent mode: let Claude Code (Claude Pro/Max) or Codex (ChatGPT plans) do the reading.

No API key is used. `prepare` splits the score into system images and writes an
instruction file (CLAUDE.md for Claude Code, AGENTS.md for Codex) into the work
folder. You open that folder in the agent and ask it to transcribe the score.
The agent looks at each image, writes systems/sysNNN.json, and runs `check`,
which reports rhythm/measure-count problems and draws proof engravings to compare
against the originals. When everything checks out it runs `rebuild` to write the
MusicXML. All the code-side checking is the same as in API mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import prompts, render
from .imaging import crop_systems, load_pages
from .model import assemble, issues_by_system, normalize_parts

MANIFEST = "scorescribe-manifest.json"


def _python_cmd() -> str:
    exe = sys.executable or "python"
    return f'"{exe}" -m scorescribe' if " " in exe else f"{exe} -m scorescribe"


def _guide() -> str:
    g = prompts.SYSTEM_PROMPT
    g = g.replace("You will be shown ONE system of printed (or neatly handwritten) sheet music at a time, cropped from a page, "
                  "and you transcribe it exactly, measure by measure, by calling the `submit_transcription` tool.",
                  "Each system (one line of music, all its staves) has been cropped from the page into its own image; "
                  "you transcribe each one exactly, measure by measure, into a JSON file.")
    g = g.replace("Before calling the tool,", "Before saving each file,")
    return g


def _schema_text() -> str:
    return json.dumps(prompts.TOOL["input_schema"], indent=1)


def prepare(inputs: list[str], workdir: str, stem: str | None = None, parts: list[dict] | None = None,
            staves_per_system: int | None = None, dpi: int = 400, zoom: bool = True,
            shape_notes: str | None = None, log=print) -> dict:
    wd = Path(workdir)
    (wd / "systems").mkdir(parents=True, exist_ok=True)
    stem = stem or Path(inputs[0]).stem or "score"
    log("Loading pages ...")
    pages = load_pages(inputs, dpi=dpi)
    log("Finding staves and systems ...")
    crops, overviews = crop_systems(pages, staves_per_system=staves_per_system, log=log)
    for i, ov in enumerate(overviews, start=1):
        ov.convert("RGB").save(wd / f"page{i:02d}_systems.jpg", quality=80)
    systems = []
    for c in crops:
        name = f"sys{c.index:03d}"
        (wd / "systems" / f"{name}.png").write_bytes(c.png_bytes())
        zooms = []
        if zoom:
            for k, z in enumerate(c.zoom_pngs(), start=1):
                (wd / "systems" / f"{name}_zoom{k}.png").write_bytes(z)
                zooms.append(f"systems/{name}_zoom{k}.png")
        systems.append({"system": c.index, "page": c.page, "image": f"systems/{name}.png", "zooms": zooms,
                        "json": f"systems/{name}.json", "staves_detected": c.n_staves,
                        "measures_detected": c.measure_count})
    manifest = {"stem": stem, "output": f"{stem}.musicxml", "parts": normalize_parts(parts) if parts else None,
                "shape_notes": shape_notes, "systems": systems}
    (wd / MANIFEST).write_text(json.dumps(manifest, indent=1))
    text = instructions(manifest)
    (wd / "CLAUDE.md").write_text(text, encoding="utf-8")
    (wd / "AGENTS.md").write_text(text, encoding="utf-8")
    log(f"  {len(systems)} systems prepared in {wd}")
    return manifest


def instructions(manifest: dict) -> str:
    cmd = _python_cmd()
    out = manifest["output"]
    shape = f" --shape-notes {manifest['shape_notes']}" if manifest.get("shape_notes") else ""
    rows = []
    for s in manifest["systems"]:
        extra = []
        if s["zooms"]:
            extra.append("magnified halves: " + ", ".join(f"`{z}`" for z in s["zooms"]))
        if s["staves_detected"]:
            extra.append(f"{s['staves_detected']} staves detected")
        if s["measures_detected"]:
            extra.append(f"barline detection counted {s['measures_detected']} measures (usually right)")
        rows.append(f"| {s['system']} | `{s['image']}` | `{s['json']}` | {'; '.join(extra)} |")
    parts = manifest.get("parts")
    layout = ("The layout is known: " + ", ".join(f"{p['name']} ({p['staves']} staves)" if p["staves"] > 1
                                                  else f"{p['name']} (1 staff)" for p in parts) + ". Number staves "
              "globally from 1 at the top.") if parts else (
        "Work out the layout from system 1 (instrument names and staves per instrument, top to bottom) and put it in "
        "`parts` in `systems/sys001.json`.")
    return f"""# Transcribe this score with ScoreScribe

You are transcribing printed sheet music into MusicXML for MuseScore. ScoreScribe has already cut the pages into
one image per system (a line of music with all its staves). **Your job is to read the images; ScoreScribe's
command-line tools check your work and build the file.** Use only the commands below, from this folder.

## Steps

1. For each system in the table below, in order: open the system image AND its magnified halves (view the image
   files), then write the transcription to the JSON file shown, in the format described under *Format*. Write the
   whole system, every measure, as a single JSON object. {layout}
2. After every few systems, and at the end, run the checker from this folder:

       {cmd} agent-check .

   It reports, per system, rhythm errors (voices that don't add up to the time signature), measure counts that
   disagree with barline detection, and suspicious octaves. It also draws an engraving of what you wrote,
   `systems/sysNNN_engraved.png`.
3. Fix every ERROR it reports: look at that measure in the images again and correct the JSON. For a measure-count
   mismatch, recount the barlines; the detector can be fooled by long stems or repeat signs, so keep your count if
   you're sure.
4. Proofread: for each system, view `systems/sysNNN_engraved.png` next to `systems/sysNNN.png` and compare measure
   by measure: pitches and octaves, accidentals, rhythms, rests, ties, number of measures. (The engraving shows
   measures numbered from 1 within the system and accidentals as resolved from the key signature; ignore layout,
   stem direction and beaming style.) Fix differences, then run the checker again.
5. When the checker reports no errors, build the file:

       {cmd} rebuild . -o "{out}"{shape}

   Then tell the user the file `{out}` in this folder is ready to open in MuseScore, and list any measures you
   were unsure about.

If you run out of usage partway through, that's fine: the JSON files you've written are kept. Next time, just
continue with the systems that don't have a JSON file yet, then do the checks.

## Systems

| System | Image | Write to | Notes |
|---|---|---|---|
{chr(10).join(rows)}

The page overviews `page01_systems.jpg` etc. show where each system was cut, in case you need context.

## Format

Each `systems/sysNNN.json` is one JSON object with this schema (`measures` is required):

```json
{_schema_text()}
```

Example (a piano system's first measure):

```json
{{"parts": [{{"name": "Piano", "staves": 2}}],
 "measures": [
  {{"time": "3/4", "key": -2, "staves": [
     {{"staff": 1, "clef": "G2", "voices": ["Bb4/4.[mf,slur_start] C5/8 D5/4[slur_end]"]}},
     {{"staff": 2, "clef": "F4", "voices": ["Bb2+F3+D4/2."]}}]}}],
 "uncertain": [{{"measure": 1, "staff": 1, "note": "the dot on beat 1 is faint"}}]}}
```

## How to read the music (follow exactly)

{_guide()}
"""


def check(workdir: str, render_proofs: bool = True, log=print) -> dict:
    wd = Path(workdir)
    manifest = json.loads((wd / MANIFEST).read_text())
    systems, missing, bad_json = [], [], []
    for s in manifest["systems"]:
        p = wd / s["json"]
        if not p.exists():
            missing.append(s["system"])
            systems.append({})
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("measures"), list):
                raise ValueError("needs a JSON object with a 'measures' list")
            systems.append(data)
        except Exception as e:
            bad_json.append((s["system"], str(e)[:200]))
            systems.append({})
    parts = manifest.get("parts") or next((x.get("parts") for x in systems if x.get("parts")), None)
    report = {"missing": missing, "bad_json": bad_json, "systems": {}, "errors": 0}
    if not parts:
        log("No layout yet: put `parts` (e.g. [{\"name\": \"Piano\", \"staves\": 2}]) in systems/sys001.json.")
        report["errors"] += 1
        return report
    present = [i for i, x in enumerate(systems) if x]
    doc = assemble(systems, parts) if present else None
    by_sys = issues_by_system(doc) if doc else {}
    for s, data in zip(manifest["systems"], systems):
        n = s["system"]
        if not data:
            continue
        iss = [i for i in by_sys.get(n, []) if i.level in ("error", "warn")]
        lines = [f"ERROR {i.where()}: {i.msg}" if i.level == "error" else f"check {i.where()}: {i.msg}" for i in iss]
        got = len(data.get("measures") or [])
        if s.get("measures_detected") and got != s["measures_detected"]:
            lines.append(f"check measure count: you have {got} measures, barline detection counted "
                         f"{s['measures_detected']} - recount (keep yours if you're sure)")
        proof = None
        if render_proofs and render.available():
            try:
                from .pipeline import render_system
                png = render_system(doc, n, data)
                proof = f"systems/sys{n:03d}_engraved.png"
                (wd / proof).write_bytes(png)
            except Exception as e:
                lines.append(f"(could not draw the proof engraving: {str(e)[:120]})")
        errs = sum(1 for i in iss if i.level == "error")
        report["errors"] += errs
        report["systems"][n] = {"lines": lines, "proof": proof, "errors": errs}

    log(f"ScoreScribe check of {wd.resolve()}")
    for s in manifest["systems"]:
        n = s["system"]
        if n in missing:
            log(f"\nSystem {n}: not transcribed yet ({s['json']} missing)")
            continue
        bj = next((e for k, e in bad_json if k == n), None)
        if bj:
            log(f"\nSystem {n}: {s['json']} is not valid: {bj}")
            report["errors"] += 1
            continue
        r = report["systems"][n]
        head = "OK" if not r["lines"] else f"{r['errors']} error(s)" if r["errors"] else "OK, with notes"
        log(f"\nSystem {n}: {head}")
        for line in r["lines"]:
            log("  " + line)
        if r["proof"]:
            log(f"  proof: compare {r['proof']} with {s['image']}")
    log("")
    if missing or bad_json:
        log(f"Not finished: {len(missing)} system(s) still to transcribe, {len(bad_json)} invalid file(s).")
    elif report["errors"]:
        log(f"{report['errors']} error(s) to fix, then run the check again.")
    else:
        shape = f" --shape-notes {manifest['shape_notes']}" if manifest.get("shape_notes") else ""
        log("ALL CHECKS PASSED. After proofreading against the engravings, build the file with:")
        log(f'  {_python_cmd()} rebuild . -o "{manifest["output"]}"{shape}')
    report["complete"] = not missing and not bad_json and not report["errors"]
    return report
