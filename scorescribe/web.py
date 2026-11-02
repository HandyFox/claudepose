"""Small local browser UI: drop in a PDF, watch progress, download MusicXML."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from .gemini import QuotaExhausted, is_gemini

HOME = Path(os.environ.get("SCORESCRIBE_HOME", Path.home() / "ScoreScribe"))
CONFIG = HOME / "config.json"
JOBS: dict[str, dict] = {}
app = Flask(__name__)
SETTINGS = ("parts", "staves_per_system", "model", "effort", "workers", "accuracy", "zoom", "proofread",
            "mark_uncertain", "shape_notes", "lyrics", "title", "separate", "satb", "satb_chords")
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".mp4", ".webm"}


def _config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except Exception:
        return {}


def _save_config(**kv):
    HOME.mkdir(parents=True, exist_ok=True)
    cfg = _config()
    cfg.update(kv)
    CONFIG.write_text(json.dumps(cfg))
    try:
        os.chmod(CONFIG, 0o600)
    except OSError:
        pass


def _job_dir(jid: str) -> Path:
    return HOME / "jobs" / jid


def _key_for(model: str, form) -> str | None:
    """The API key for this model: typed in the form, then environment, then saved."""
    gem = is_gemini(model)
    typed = (form.get("gemini_key") if gem else form.get("api_key")) or ""
    typed = typed.strip()
    if typed and form.get("remember") == "on":
        _save_config(**({"gemini_key": typed} if gem else {"api_key": typed}))
    if gem:
        return typed or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or _config().get("gemini_key")
    return typed or os.environ.get("ANTHROPIC_API_KEY") or _config().get("api_key")


def _no_key_message(model: str) -> str:
    if is_gemini(model):
        return "Add your Gemini API key (free at aistudio.google.com/apikey) or set GEMINI_API_KEY."
    return "Add your Anthropic API key (console.anthropic.com) or set ANTHROPIC_API_KEY."


@app.get("/")
def index():
    cfg = _config()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or cfg.get("api_key"))
    has_gem = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or cfg.get("gemini_key"))
    return (PAGE.replace("__HASKEY__", "true" if has_key else "false")
            .replace("__HASGEM__", "true" if has_gem else "false")
            .replace("__LASTMODEL__", json.dumps(cfg.get("last_model") or "claude-opus-5-5")))


@app.post("/start")
def start():
    files = request.files.getlist("files")
    if not files or not files[0].filename:
        return jsonify(error="Choose a PDF or images first."), 400
    form = request.form
    mode = form.get("mode", "full")
    model = form.get("model") or "claude-opus-5-5"
    song = Path(files[0].filename).suffix.lower() in AUDIO_EXT
    if song and mode == "segment":
        return jsonify(error="That's a recording; the system-split preview is for sheet music."), 400
    if mode == "full" and model.startswith("agent:"):
        mode = "agent"  # no API key: the user's own Claude Code / Codex does the reading
    if song:
        mode = "song-agent" if mode == "agent" else "song"
    key = None if mode not in ("full", "song") else _key_for(model, form)
    if mode in ("full", "song") and not key:
        return jsonify(error=_no_key_message(model)), 400
    if mode != "segment":
        _save_config(last_model=model)

    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = _job_dir(jid)
    (d / "input").mkdir(parents=True)
    inputs = []
    for i, f in enumerate(files):
        name = f"{i:03d}_" + "".join(c for c in Path(f.filename).name if c.isalnum() or c in "._-")
        f.save(d / "input" / name)
        inputs.append(name)
    draft = None
    df = request.files.get("draft")
    if df and df.filename:
        draft = "draft" + Path(df.filename).suffix
        df.save(d / draft)
    meta = {"stem": Path(files[0].filename).stem or "score", "mode": mode, "inputs": inputs, "draft": draft,
            "settings": {k: form.get(k) for k in SETTINGS}}
    if mode != "segment":
        (d / "job.json").write_text(json.dumps(meta, indent=1))
    _launch(jid, meta, key)
    return jsonify(id=jid)


@app.post("/resume/<jid>")
def resume(jid):
    d = _job_dir(jid)
    try:
        meta = json.loads((d / "job.json").read_text())
    except Exception:
        return jsonify(error="That job can't be resumed (it was made by an older version)."), 400
    if JOBS.get(jid, {}).get("status") == "running":
        return jsonify(id=jid)
    model = request.form.get("model") or meta["settings"].get("model") or "claude-opus-5-5"
    meta["settings"]["model"] = model  # you may switch reader, e.g. to Gemini when Claude credit runs out
    key = _key_for(model, request.form)
    if not key:
        return jsonify(error=_no_key_message(model)), 400
    (d / "job.json").write_text(json.dumps(meta, indent=1))
    _launch(jid, meta, key, resumed=True)
    return jsonify(id=jid)


def _launch(jid: str, meta: dict, key: str | None, resumed: bool = False):
    d = _job_dir(jid)
    settings = meta.get("settings") or {}
    mode, stem = meta.get("mode", "full"), meta.get("stem", "score")
    inputs = [str(d / "input" / n) for n in meta["inputs"]]
    draft = str(d / meta["draft"]) if meta.get("draft") else None
    job = {"id": jid, "log": [], "status": "running", "mode": mode, "stem": stem, "result": None}
    JOBS[jid] = job

    def log(msg=""):
        job["log"].append(str(msg))

    def staves():
        v = (settings.get("staves_per_system") or "").strip()
        return int(v) if v.isdigit() and int(v) > 0 else None

    def work():
        try:
            if mode in ("song", "song-agent"):
                from .song import run as song
                opts = dict(lyrics_text=(settings.get("lyrics") or "").strip() or None,
                            separate=settings.get("separate") != "off", shape_notes=settings.get("shape_notes") or None,
                            title=(settings.get("title") or "").strip() or None,
                            satb=settings.get("satb") == "on", satb_chords=settings.get("satb_chords") == "on")
                if mode == "song-agent":
                    song.prepare(inputs[0], str(d), stem=stem, log=log, **opts)
                    job["result"] = {"agent": (settings.get("model") or "agent:claude").split(":", 1)[1],
                                     "folder": str(d.resolve()), "song": True, "pages": 0,
                                     "api_key_env": bool(os.environ.get("ANTHROPIC_API_KEY"))}
                else:
                    model = settings.get("model") or "claude-opus-5-5"
                    if resumed:
                        log(f"Resuming with {model}; the audio analysis is reused ...")
                    acc = settings.get("accuracy") or "high"
                    res = song.transcribe_song(inputs[0], str(d), str(d / f"{stem}.musicxml"), model=model, api_key=key,
                                               effort=settings.get("effort") or "high",
                                               passes=1 if acc == "standard" else 2, log=log, **opts)
                    res["song"] = True
                    job["result"] = res
            elif mode == "agent":
                from .agent import prepare
                from .cli import parse_parts
                m = prepare(inputs, str(d), stem=stem, parts=parse_parts(settings.get("parts")),
                            staves_per_system=staves(), zoom=settings.get("zoom") == "on",
                            shape_notes=settings.get("shape_notes") or None, log=log)
                job["result"] = {"agent": (settings.get("model") or "agent:claude").split(":", 1)[1],
                                 "folder": str(d.resolve()), "systems": len(m["systems"]),
                                 "pages": len(list(d.glob("page*_systems.jpg"))),
                                 "api_key_env": bool(os.environ.get("ANTHROPIC_API_KEY"))}
            elif mode == "segment":
                from .imaging import crop_systems, load_pages
                pages = load_pages(inputs, dpi=400)
                crops, ovs = crop_systems(pages, staves_per_system=staves(), log=log)
                for i, ov in enumerate(ovs, start=1):
                    ov.convert("RGB").save(d / f"page{i:02d}_systems.jpg", quality=80)
                log(f"{len(crops)} systems found. Green lines mark detected barlines. If a box is wrong, set 'staves per system' and preview again.")
                job["result"] = {"pages": len(ovs)}
            else:
                from .cli import parse_parts
                from .pipeline import Transcriber
                model = settings.get("model") or "claude-opus-5-5"
                acc = settings.get("accuracy") or "high"
                effort = settings.get("effort") or "high"
                if acc == "max":
                    effort = "xhigh" if effort in ("high", "medium", "low") else effort
                w = (settings.get("workers") or "").strip()
                workers = int(w) if w.isdigit() and int(w) > 0 else (2 if is_gemini(model) else 4)
                if resumed:
                    log(f"Resuming with {model}; finished work is reused ...")
                t = Transcriber(api_key=key, model=model, effort=effort, workers=workers,
                                proofread=settings.get("proofread") == "on",
                                passes=1 if acc == "standard" else 2, zoom=settings.get("zoom") == "on",
                                mark_uncertain=settings.get("mark_uncertain") == "on",
                                shape_notes=settings.get("shape_notes") or None, log=log)
                out = d / f"{stem}.musicxml"
                res = t.run(inputs, str(d), str(out), parts_hint=parse_parts(settings.get("parts")),
                            staves_per_system=staves(), draft=draft, resume=True)
                res["pages"] = len(list(d.glob("page*_systems.jpg")))
                job["result"] = res
            job["status"] = "done"
        except QuotaExhausted as e:
            log("")
            log("PAUSED: " + str(e))
            job["result"] = {"message": str(e)}
            job["status"] = "paused"
        except Exception as e:  # surface everything in the UI
            import traceback
            cause = e.__cause__ or e.__context__
            log("ERROR: " + str(e) + (f" (cause: {type(cause).__name__}: {str(cause)[:200]})" if cause else ""))
            log(traceback.format_exc())
            job["status"] = "error"

    threading.Thread(target=work, daemon=True).start()


@app.get("/jobs")
def jobs():
    out = []
    root = HOME / "jobs"
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)[:40] if root.exists() else []
    for d in dirs:
        try:
            meta = json.loads((d / "job.json").read_text())
        except Exception:
            continue
        stem = meta.get("stem", "score")
        live = JOBS.get(d.name, {}).get("status")
        model = meta.get("settings", {}).get("model") or ""
        pngs = [p for p in (d / "systems").glob("sys???.png")] if (d / "systems").exists() else []
        done_sys = [p for p in (d / "systems").glob("sys???.json")] if (d / "systems").exists() else []
        if live == "running":
            state = "running"
        elif (d / f"{stem}.musicxml").exists() and live != "paused":
            state = "done"
        elif model.startswith("agent:"):
            state = "agent"
        else:
            state = "unfinished"
        out.append({"id": d.name, "stem": stem, "state": state, "model": model,
                    "systems": len(pngs), "read": len(done_sys), "folder": str(d.resolve()),
                    "song": meta.get("mode", "").startswith("song")})
        if len(out) >= 12:
            break
    return jsonify(jobs=out)


@app.post("/shapes")
def shapes():
    """Convert an uploaded MusicXML file to shape notes and send it back."""
    import tempfile
    from .shapes import convert_file
    f = request.files.get("file")
    scheme = request.form.get("scheme") or "aiken"
    if not f or not f.filename:
        return jsonify(error="Choose a MusicXML file (.musicxml, .xml or .mxl) first."), 400
    stem = Path(f.filename).stem or "score"
    td = Path(tempfile.mkdtemp(prefix="shapes_"))
    src = td / ("in" + (Path(f.filename).suffix or ".musicxml"))
    f.save(src)
    out = td / f"{stem}_{scheme}.musicxml"
    try:
        n = convert_file(str(src), str(out), scheme)
    except Exception as e:
        return jsonify(error=f"Could not convert that file: {e}"), 400
    resp = send_file(out, as_attachment=True, download_name=out.name)
    resp.headers["X-Notes-Changed"] = str(n)
    return resp


@app.get("/status/<jid>")
def status(jid):
    job = JOBS.get(jid) or abort(404)
    since = int(request.args.get("since", 0))
    return jsonify(status=job["status"], log=job["log"][since:], n=len(job["log"]), result=job["result"],
                   stem=job["stem"], mode=job["mode"], id=jid)


@app.get("/file/<jid>/<path:name>")
def file(jid, name):
    if jid not in JOBS and not _job_dir(jid).exists():
        abort(404)
    p = (_job_dir(jid) / name).resolve()
    if not str(p).startswith(str(_job_dir(jid).resolve())) or not p.exists():
        abort(404)
    return send_file(p, as_attachment=request.args.get("dl") == "1")


def serve(port: int = 8765, open_browser: bool = True):
    url = f"http://127.0.0.1:{port}"
    print(f"ScoreScribe is running at {url}  (Ctrl+C to stop). Jobs are saved in {HOME / 'jobs'}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScoreScribe</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap" rel="stylesheet">
<style>
:root{--paper:#f6f2ea;--ink:#1d1b18;--muted:#6f685d;--line:#d9d1c3;--card:#fffdf9;--accent:#8a2d1c;--accent2:#c0612b;--ok:#2f6b3a;--warn:#b5651d}
@media (prefers-color-scheme: dark){:root{--paper:#191714;--ink:#efe9de;--muted:#a59c8e;--line:#3a352e;--card:#221f1b;--accent:#e08a63;--accent2:#e0a463;--ok:#7fbf8a;--warn:#e0a463}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:36px 20px 80px}
header{display:flex;align-items:baseline;gap:16px;border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:28px;
  background-image:repeating-linear-gradient(to bottom,transparent 0 7px,var(--line) 7px 8px);background-size:100% 40px;background-repeat:no-repeat;background-position:0 100%}
h1{font-family:Fraunces,serif;font-size:40px;margin:0;letter-spacing:-.5px}
header p{margin:0;color:var(--muted)}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:22px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:20px}
h2{font-family:Fraunces,serif;font-size:19px;margin:0 0 14px}
label{display:block;font-weight:500;font-size:13px;margin:14px 0 5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
input[type=text],input[type=password],input[type=number],select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:4px;background:var(--paper);color:var(--ink);font:inherit}
.drop{border:2px dashed var(--line);border-radius:6px;padding:26px;text-align:center;cursor:pointer;transition:.15s}
.drop.over,.drop:hover{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 6%,transparent)}
.drop b{font-family:Fraunces,serif;font-size:18px}
.files{font-size:13px;color:var(--muted);margin-top:8px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.check{display:flex;gap:8px;align-items:center;margin-top:12px;font-size:14px}
.btns{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap}
button{font:600 14px "IBM Plex Sans";padding:11px 18px;border-radius:4px;border:1px solid var(--ink);background:var(--ink);color:var(--paper);cursor:pointer}
button.ghost{background:transparent;color:var(--ink)}
button:disabled{opacity:.45;cursor:default}
details summary{cursor:pointer;margin-top:16px;color:var(--muted);font-size:14px}
pre{font:12.5px/1.55 "IBM Plex Mono",monospace;background:var(--paper);border:1px solid var(--line);border-radius:4px;padding:12px;height:340px;overflow:auto;white-space:pre-wrap;margin:0}
.status{font-size:14px;margin-bottom:10px;color:var(--muted)}
.status.done{color:var(--ok)}.status.error{color:var(--accent)}
.result{margin-top:16px;display:none}
.result a.big{display:inline-block;background:var(--accent);color:#fff;text-decoration:none;padding:12px 18px;border-radius:4px;font-weight:600}
.legend{font-size:13px;color:var(--muted);margin-top:10px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}
.status.paused{color:var(--warn)}
.job{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:8px 0;border-top:1px solid var(--line);font-size:14px}
.job small{display:block}
.job button{padding:6px 12px;font-size:13px}
.pages{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:22px}
.pages img{width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}
small{color:var(--muted)}
</style></head><body><div class="wrap">
<header><h1>ScoreScribe</h1><p>Sheet music &rarr; MusicXML for MuseScore</p></header>
<div class="grid">
<form class="card" id="f">
  <h2>1 &middot; Your score</h2>
  <div class="drop" id="drop"><b>Drop sheet music or a song recording here</b><br><small>PDF or page images (read in file-name order) &middot; or one recording: mp3, wav, m4a, flac&hellip;</small>
    <input type="file" id="files" name="files" multiple accept=".pdf,image/*,audio/*,.m4a,.flac,.ogg,.opus" hidden></div>
  <div class="files" id="flist"></div>
  <div id="songopts" style="display:none">
    <small>A recording: you'll get a lead sheet with the melody, lyrics and chord symbols.</small>
    <label>Song title (optional)</label><input type="text" name="title" placeholder="e.g. Amazing Grace">
    <label>Lyrics (optional, makes the words exact)</label><textarea name="lyrics" rows="5" style="width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:4px;background:var(--paper);color:var(--ink);font:inherit" placeholder="Paste the lyrics here if you have them"></textarea>
    <div class="check"><input type="checkbox" id="sepchk" checked onchange="document.getElementById('sepval').value=this.checked?'on':'off'"><label for="sepchk" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Separate the voice from the band first (slower, much better melody)</label><input type="hidden" name="separate" id="sepval" value="on"></div>
    <div class="check"><input type="checkbox" name="satb" id="satbchk"><label for="satbchk" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Four-part harmony (SATB): Claude arranges alto, tenor and bass in hymn style under the melody</label></div>
    <div class="check" style="margin-left:24px"><input type="checkbox" name="satb_chords" id="satbcc"><label for="satbcc" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Also print chord symbols above the four-part score</label></div>
  </div>
  <div class="row">
    <div><label>Instruments (optional)</label><input type="text" name="parts" placeholder="Piano:2  or  S,A,T,B"></div>
    <div><label>Staves per system (optional)</label><input type="number" name="staves_per_system" min="1" placeholder="auto"></div>
  </div>
  <label>Noteheads</label>
  <select name="shape_notes">
    <option value="" selected>Normal (round)</option>
    <option value="aiken">Shape notes: Aiken 7-shape (do re mi …)</option>
    <option value="4shape">Shape notes: 4-shape (fa so la mi)</option>
  </select>
  <div id="accwrap"><label>Accuracy</label>
  <select name="accuracy">
    <option value="high" selected>High — reads each system twice and settles disagreements</option>
    <option value="max">Maximum — twice, with extra thinking (slowest)</option>
    <option value="standard">Standard — one reading (cheapest)</option>
  </select></div>
  <label>Reader</label>
  <select name="model" id="model">
    <optgroup label="Claude (Anthropic credits)">
      <option value="claude-opus-5-5">Claude Opus 5.5 — most accurate</option>
      <option value="claude-sonnet-5-5">Claude Sonnet 5.5 — cheaper</option>
      <option value="claude-haiku-5-5">Claude Haiku 5.5 — cheapest</option>
    </optgroup>
    <optgroup label="Your subscription — no API key">
      <option value="agent:claude">My Claude Pro/Max plan, via Claude Code</option>
      <option value="agent:codex">My ChatGPT plan, via Codex</option>
    </optgroup>
    <optgroup label="Gemini (free tier available)">
      <option value="gemini-3.8-flash">Gemini 3.8 Flash — free tier</option>
      <option value="gemini-3.6-flash">Gemini 3.6 Flash — free tier</option>
      <option value="gemini-3.5-flash-lite">Gemini 3.5 Flash-Lite — free tier, fastest</option>
    </optgroup>
  </select>
  <div id="claudekey"><label>Anthropic API key</label><input type="password" name="api_key" placeholder="sk-ant-..."></div>
  <div id="gemkey" style="display:none"><label>Gemini API key <a href="https://aistudio.google.com/apikey" target="_blank" style="text-transform:none;letter-spacing:0">get a free one</a></label><input type="password" name="gemini_key" placeholder="AIza...">
    <small>Free-tier requests may be used by Google to improve its products. Daily limits apply; if one runs out, the run pauses and you can resume it later.</small></div>
  <div id="agentnote" style="display:none"><small>No API key needed. ScoreScribe prepares a folder; you then open it in Claude Code (Claude desktop app &rarr; Code tab) or Codex and ask it to transcribe the score. It uses your plan's normal usage limits.</small></div>
  <div class="check"><input type="checkbox" name="remember" id="rem"><label for="rem" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Remember keys on this computer</label></div>
  <details><summary>Advanced</summary>
    <div class="row">
      <div><label>Effort (thinking)</label><select name="effort"><option>high</option><option>xhigh</option><option>max</option><option>medium</option><option>low</option></select></div>
      <div><label>Parallel systems</label><input type="number" name="workers" placeholder="auto" min="1" max="16"></div>
    </div>
    <div class="row">
      <div><label>Draft MusicXML (e.g. Audiveris)</label><input type="file" name="draft" accept=".mxl,.musicxml,.xml"></div>
    </div>
    <div class="check"><input type="checkbox" name="zoom" id="zm" checked><label for="zm" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Magnified views: also send each system in two enlarged halves</label></div>
    <div class="check"><input type="checkbox" name="mark_uncertain" id="mu"><label for="mu" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Colour notes the reader was unsure about orange (otherwise they're listed in the report)</label></div>
    <div class="check"><input type="checkbox" name="proofread" id="pr" checked><label for="pr" style="margin:0;text-transform:none;letter-spacing:0;font-size:14px">Proofread: the reader compares an engraving of its transcription with the original</label></div>
  </details>
  <div class="btns"><button type="button" class="ghost" id="seg">Preview system split (free)</button><button type="button" id="go">Transcribe</button></div>
</form>
<div class="card">
  <h2>2 &middot; Progress</h2>
  <div class="status" id="st">Waiting for a score.</div>
  <pre id="log"></pre>
  <div class="result" id="res"></div>
  <h2 style="margin-top:22px">Recent runs</h2>
  <div id="jobs"><small>None yet.</small></div>
</div>
</div>
<div class="pages" id="pages"></div>
<form class="card" id="shp" style="margin-top:22px">
  <h2>Convert any MusicXML to shape notes</h2>
  <small>For a score you already have, for example one exported from MuseScore (File &rarr; Export &rarr; MusicXML). Shapes follow the key signature (movable do).</small>
  <div class="row" style="margin-top:6px">
    <div><label>MusicXML file</label><input type="file" name="file" accept=".musicxml,.xml,.mxl"></div>
    <div><label>Shapes</label><select name="scheme"><option value="aiken">Aiken 7-shape</option><option value="4shape">4-shape (fa so la mi)</option></select></div>
  </div>
  <div class="btns"><button type="button" id="shpgo">Convert &amp; download</button><span id="shpst" class="status" style="margin:10px 0 0"></span></div>
</form>
</div>
<script>
const hasKey = __HASKEY__, hasGem = __HASGEM__;
if (hasKey){document.querySelector('[name=api_key]').placeholder='saved key will be used';}
if (hasGem){document.querySelector('[name=gemini_key]').placeholder='saved key will be used';}
const modelSel=document.getElementById('model');
modelSel.value=__LASTMODEL__; if(!modelSel.value) modelSel.value='claude-opus-5-5';
function syncKeys(){const g=modelSel.value.startsWith('gemini'), a=modelSel.value.startsWith('agent:');
  document.getElementById('gemkey').style.display=g?'':'none'; document.getElementById('claudekey').style.display=(g||a)?'none':'';
  document.getElementById('agentnote').style.display=a?'':'none'; document.getElementById('rem').parentElement.style.display=a?'none':'';
  document.getElementById('go').textContent=a?'Prepare for my subscription':'Transcribe';
  document.getElementById('accwrap').style.display=a?'none':'';}
modelSel.onchange=syncKeys; syncKeys();
const drop=document.getElementById('drop'), fi=document.getElementById('files'), fl=document.getElementById('flist');
let chosen=[];
drop.onclick=()=>fi.click();
fi.onchange=()=>{chosen=[...fi.files];show()};
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{chosen=[...ev.dataTransfer.files];show()});
const AUDIO=/\.(mp3|wav|m4a|aac|flac|ogg|oga|opus|wma|aiff?|mp4|webm)$/i;
function isSong(){return chosen.length>0 && AUDIO.test(chosen[0].name)}
function show(){chosen.sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));fl.textContent=chosen.map(f=>f.name).join(' · ');
  document.getElementById('songopts').style.display=isSong()?'':'none';
  ['seg'].forEach(id=>document.getElementById(id).style.display=isSong()?'none':'')}
const logEl=document.getElementById('log'), st=document.getElementById('st'), res=document.getElementById('res'), pages=document.getElementById('pages');
async function start(mode){
  if(!chosen.length){st.textContent='Choose a PDF or images first.';st.className='status error';return}
  const fd=new FormData(document.getElementById('f')); fd.delete('files'); chosen.forEach(f=>fd.append('files',f)); fd.append('mode',mode);
  document.getElementById('go').disabled=document.getElementById('seg').disabled=true;
  logEl.textContent=''; res.style.display='none'; pages.innerHTML=''; st.className='status'; st.textContent=mode=='segment'?'Finding systems…':(isSong()?'Listening to the recording… (separating the voice can take several minutes)':(modelSel.value.startsWith('agent:')?'Preparing the folder…':'Working… (a page usually takes a few minutes)'));
  const r=await fetch('/start',{method:'POST',body:fd}); const j=await r.json();
  if(!r.ok){st.textContent=j.error;st.className='status error';done();return}
  poll(j.id,0);
}
function done(){document.getElementById('go').disabled=document.getElementById('seg').disabled=false}
async function poll(id,n){
  const j=await (await fetch(`/status/${id}?since=${n}`)).json();
  if(j.log.length){logEl.textContent+=j.log.join('\n')+'\n';logEl.scrollTop=logEl.scrollHeight}
  if(j.status=='running'){setTimeout(()=>poll(id,j.n),1500);return}
  done(); loadJobs();
  if(j.status=='error'){st.textContent='Something went wrong — see the log.';st.className='status error';return}
  if(j.status=='paused'){st.textContent='Paused — '+((j.result||{}).message||'')+' Use Resume under Recent runs (you can also switch the reader first).';st.className='status paused';return}
  const r=j.result||{};
  if(j.mode=='agent'){showAgent(r);return}
  for(let i=1;i<=(r.pages||0);i++){const a=document.createElement('a');a.href=`/file/${id}/page${String(i).padStart(2,'0')}_systems.jpg`;a.target='_blank';a.innerHTML=`<img src="${a.href}" alt="page ${i}">`;pages.appendChild(a)}
  if(j.mode=='segment'){st.textContent='Split ready — check the boxes below.';st.className='status done';return}
  st.textContent=`Done: ${r.measures} measures.`;st.className='status done';
  if(r.song){
    res.innerHTML=`<a class="big" href="/file/${id}/${encodeURIComponent(j.stem)}.musicxml?dl=1">${r.satb?'Download four-part score (MusicXML)':'Download lead sheet (MusicXML)'}</a>
     ${r.satb?`&nbsp; <a href="/file/${id}/${encodeURIComponent(j.stem)}_leadsheet.musicxml?dl=1">lead sheet</a>`:''}
     &nbsp; <a href="/file/${id}/chords.txt" target="_blank">chord chart</a> &nbsp; <a href="/file/${id}/report.md" target="_blank">report</a>
     <p><small>Chords fitting the audio: ${r.stats.chords_fitting} &middot; melody notes matching the singing: ${r.stats.melody_matching}</small></p>
     ${(r.orange||[]).length?`<p>Worth a listen (unsure): measures ${r.orange.join(', ')}</p>`:''}
     <div class="legend">Open it in MuseScore (File → Open): melody with lyrics and chord symbols.</div>`;
    res.style.display='block'; return}
  const red=(r.red||[]).length?`<p>Red measures (failed checks): ${r.red.join(', ')}</p>`:'';
  const or=(r.orange||[]).length?`<p>Worth a listen (the reader was unsure): measures ${r.orange.join(', ')}</p>`:'';
  res.innerHTML=`<a class="big" href="/file/${id}/${encodeURIComponent(j.stem)}.musicxml?dl=1">Download MusicXML</a>
   &nbsp; <a href="/file/${id}/report.md" target="_blank">report</a>${red}${or}
   <div class="legend">Open the file in MuseScore (File → Open). <span style="background:#D0021B"></span>Red notes failed an automatic check.</div>`;
  res.style.display='block';
}
function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
function agentSteps(agent, folder, warnKey, song){
  const ask = song ? 'Make the lead sheet' : 'Transcribe this score';
  const f=esc(folder);
  const copy=`<button type="button" class="ghost" style="padding:6px 12px;font-size:13px" data-copy="${f}" onclick="navigator.clipboard.writeText(this.dataset.copy);this.textContent='Copied'">Copy folder path</button>`;
  const steps = agent=='codex'
    ? `<ol><li>Open a terminal in this folder (or <code>cd</code> to it):<br><code>${f}</code> ${copy}</li><li>Start Codex, signed in with your ChatGPT account: <code>codex</code></li><li>Type: <b>${ask}</b></li></ol>`
    : `<ol><li>Open the <b>Claude desktop app</b> &rarr; <b>Code</b> tab, and choose this folder:<br><code>${f}</code> ${copy}<br><small>(or in a terminal: <code>cd</code> there and run <code>claude</code>)</small></li><li>Type: <b>${ask}</b></li><li>Allow it to run the ScoreScribe check commands when it asks.</li></ol>`;
  const warn = (warnKey && agent!='codex') ? `<p class="status error">Heads-up: this computer has an ANTHROPIC_API_KEY environment variable. Claude Code uses that key (pay-per-use) instead of your subscription while it's set. Remove it under Windows Settings &rarr; "Edit environment variables for your account", then restart the app.</p>` : '';
  return warn+steps+`<p><small>The finished file appears here under Recent runs (Download) once the agent writes it. If you hit your plan's usage limit, just say "continue" later — finished parts are kept.</small></p>`;
}
function showAgent(r){
  st.textContent=r.song?'Ready: the recording has been analysed.':`Ready: ${r.systems} systems prepared.`; st.className='status done';
  res.innerHTML=agentSteps(r.agent, r.folder, r.api_key_env, r.song); res.style.display='block';
}
async function resumeJob(id){
  const fd=new FormData(); const f=document.getElementById('f');
  ['model','api_key','gemini_key','remember'].forEach(k=>{const el=f.querySelector(`[name=${k}]`); if(!el)return;
    if(el.type=='checkbox'){if(el.checked)fd.append(k,'on')} else fd.append(k,el.value)});
  logEl.textContent=''; res.style.display='none'; st.className='status'; st.textContent='Resuming…';
  const r=await fetch(`/resume/${id}`,{method:'POST',body:fd}); const j=await r.json();
  if(!r.ok){st.textContent=j.error;st.className='status error';return}
  document.getElementById('go').disabled=document.getElementById('seg').disabled=true; poll(j.id,0); loadJobs();
}
async function loadJobs(){
  const box=document.getElementById('jobs');
  try{const j=await (await fetch('/jobs')).json();
    if(!j.jobs.length){box.innerHTML='<small>None yet.</small>';return}
    box.innerHTML='';
    j.jobs.forEach(x=>{const row=document.createElement('div');row.className='job';
      const when=x.id.slice(0,4)+'-'+x.id.slice(4,6)+'-'+x.id.slice(6,8)+' '+x.id.slice(9,11)+':'+x.id.slice(11,13);
      const info=document.createElement('div');info.innerHTML=`<b></b><small></small>`;
      info.querySelector('b').textContent=x.stem;
      info.querySelector('small').textContent=`${when} · ${x.model||''} · ${x.state=='done'?'finished':x.state=='running'?'running…':(x.song?'unfinished':`unfinished (${x.read}/${x.systems} systems read)`)}`;
      row.appendChild(info);
      if(x.state=='agent'){info.querySelector('small').textContent=`${when} · waiting for ${x.model=='agent:codex'?'Codex':'Claude Code'} to finish`;
        const b=document.createElement('button');b.className='ghost';b.textContent='Show steps';
        b.onclick=()=>{res.innerHTML=agentSteps(x.model.split(':')[1], x.folder, false, x.song);res.style.display='block';st.textContent='Open this folder in your agent:';st.className='status'};row.appendChild(b)}
      else if(x.state=='done'){const a=document.createElement('a');a.href=`/file/${x.id}/${encodeURIComponent(x.stem)}.musicxml?dl=1`;a.textContent='Download';row.appendChild(a)}
      else if(x.state=='unfinished'){const b=document.createElement('button');b.className='ghost';b.textContent='Resume';b.onclick=()=>resumeJob(x.id);row.appendChild(b)}
      box.appendChild(row)});
  }catch(e){}
}
loadJobs();
document.getElementById('go').onclick=()=>start('full');
document.getElementById('seg').onclick=()=>start('segment');
document.getElementById('shpgo').onclick=async()=>{
  const st=document.getElementById('shpst'); const fd=new FormData(document.getElementById('shp'));
  if(!fd.get('file')||!fd.get('file').name){st.textContent='Choose a file first.';st.className='status error';return}
  st.textContent='Converting…';st.className='status';
  const r=await fetch('/shapes',{method:'POST',body:fd});
  if(!r.ok){const j=await r.json();st.textContent=j.error;st.className='status error';return}
  const b=await r.blob(); const a=document.createElement('a');
  const name=(fd.get('file').name.replace(/\.[^.]+$/,'')||'score')+'_'+fd.get('scheme')+'.musicxml';
  a.href=URL.createObjectURL(b); a.download=name; a.click();
  st.textContent=`Done: ${r.headers.get('X-Notes-Changed')} notes shaped. Open it in MuseScore.`;st.className='status done';
};
</script></body></html>"""
