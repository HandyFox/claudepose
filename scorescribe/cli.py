"""Command line: scorescribe transcribe | segment | rebuild | gui"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_parts(text: str | None) -> list[dict] | None:
    """'Piano:2' or 'Soprano,Alto,Tenor,Bass' or 'Violin:1, Piano:2'"""
    if not text:
        return None
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, n = chunk.partition(":")
        out.append({"name": name.strip(), "staves": int(n) if n.strip().isdigit() else 1})
    return out or None


def _work_dir(args, inputs) -> str:
    if args.workdir:
        return args.workdir
    out = Path(args.output) if getattr(args, "output", None) else Path(inputs[0]).with_suffix(".musicxml")
    return str(out.with_name(out.stem + "_work"))


def cmd_transcribe(args):
    from .gemini import DEFAULT_GEMINI_MODEL, QuotaExhausted, is_gemini
    from .pipeline import Transcriber
    if args.gemini and not is_gemini(args.model):
        args.model = DEFAULT_GEMINI_MODEL
    if args.model is None:
        args.model = "claude-opus-5-5"
    gem = is_gemini(args.model)
    if gem and not (args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("No Gemini API key. Set GEMINI_API_KEY or pass --api-key (free key: https://aistudio.google.com/apikey).")
    if not gem and not (args.api_key or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("No API key. Set ANTHROPIC_API_KEY or pass --api-key (get one at https://console.anthropic.com).\n"
                 "Or use a free Gemini key: --gemini with GEMINI_API_KEY set.")
    if args.workers is None:
        args.workers = 2 if gem else 4
    output = args.output or str(Path(args.inputs[0]).with_suffix(".musicxml"))
    wd = _work_dir(args, args.inputs)
    draft = args.draft
    if args.audiveris and not draft:
        from .audiveris import find_audiveris, run_audiveris
        exe = find_audiveris(args.audiveris)
        if not exe:
            print("Audiveris not found; continuing without a draft (pass its path: --audiveris /path/to/Audiveris)")
        else:
            draft = run_audiveris(exe, args.inputs, os.path.join(wd, "audiveris"))
    effort, passes = args.effort, args.passes
    if args.max_accuracy:
        effort, passes = "xhigh", 2
    if args.fast:
        passes = 1
    t = Transcriber(api_key=args.api_key, model=args.model, effort=None if effort == "default" else effort,
                    thinking=not args.no_thinking, workers=args.workers, repair_rounds=args.repair_rounds,
                    proofread=not args.no_proofread, passes=passes, zoom=not args.no_zoom,
                    mark_uncertain=args.mark_uncertain, shape_notes=args.shape_notes)
    if args.fresh:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)
    try:
        t.run(args.inputs, wd, output, parts_hint=parse_parts(args.parts), staves_per_system=args.staves_per_system,
              dpi=args.dpi, whole_pages=args.whole_pages, resume=True, draft=draft)
    except QuotaExhausted as e:
        print(f"\nPAUSED: {e}\nRun the same command again later to continue; finished systems are not re-read.")
        sys.exit(3)


def cmd_segment(args):
    from .imaging import crop_systems, load_pages
    wd = Path(_work_dir(args, args.inputs))
    (wd / "systems").mkdir(parents=True, exist_ok=True)
    pages = load_pages(args.inputs, dpi=args.dpi)
    crops, overviews = crop_systems(pages, staves_per_system=args.staves_per_system, whole_pages=args.whole_pages)
    for i, ov in enumerate(overviews, start=1):
        ov.convert("RGB").save(wd / f"page{i:02d}_systems.jpg", quality=80)
    for c in crops:
        (wd / "systems" / f"sys{c.index:03d}.png").write_bytes(c.png_bytes())
    print(f"{len(crops)} systems. Check {wd}/page*_systems.jpg; crops are in {wd}/systems/")


def cmd_rebuild(args):
    from .pipeline import rebuild
    out = args.output or str(Path(args.workdir).with_name(Path(args.workdir).name.replace("_work", "") + ".musicxml"))
    rebuild(args.workdir, out, mark_uncertain=args.mark_uncertain, shape_notes=args.shape_notes)


def cmd_shapes(args):
    from .shapes import convert_file
    src = Path(args.input)
    out = args.output or str(src.with_name(f"{src.stem}_{args.scheme}.musicxml"))
    n = convert_file(str(src), out, args.scheme)
    print(f"{n} notes given {args.scheme} shapes -> {out}")


def cmd_agent_prepare(args):
    from .agent import prepare
    wd = _work_dir(args, args.inputs)
    m = prepare(args.inputs, wd, parts=parse_parts(args.parts), staves_per_system=args.staves_per_system,
                dpi=args.dpi, zoom=not args.no_zoom, shape_notes=args.shape_notes)
    print(f"\nReady. Open this folder in Claude Code (Claude desktop app > Code tab) or Codex:\n  {Path(wd).resolve()}\n"
          f"and say: Transcribe this score.   ({len(m['systems'])} systems; instructions are in CLAUDE.md / AGENTS.md)")


def cmd_agent_check(args):
    from .agent import check
    r = check(args.workdir, render_proofs=not args.no_proofs)
    sys.exit(0 if r.get("complete") else 1)


def _lyrics_arg(args):
    if not getattr(args, "lyrics", None):
        return None
    p = Path(args.lyrics)
    return p.read_text(encoding="utf-8") if p.exists() else args.lyrics


def cmd_song(args):
    from .gemini import DEFAULT_GEMINI_MODEL, QuotaExhausted, is_gemini
    from .song.run import transcribe_song
    if args.gemini and not is_gemini(args.model):
        args.model = DEFAULT_GEMINI_MODEL
    args.model = args.model or "claude-opus-5-5"
    gem = is_gemini(args.model)
    if gem and not (args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("No Gemini API key. Set GEMINI_API_KEY or pass --api-key.")
    if not gem and not (args.api_key or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("No API key. Set ANTHROPIC_API_KEY, use --gemini, or use your subscription: scorescribe song-prepare")
    output = args.output or str(Path(args.audio).with_suffix(".musicxml"))
    wd = _work_dir(args, [args.audio])
    if args.fresh:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)
    try:
        transcribe_song(args.audio, wd, output, model=args.model, api_key=args.api_key,
                        effort=None if args.effort == "default" else args.effort, passes=args.passes,
                        lyrics_text=_lyrics_arg(args), separate=not args.no_separate, whisper_model=args.whisper_model,
                        language=args.language, shape_notes=args.shape_notes, title=args.title,
                        satb=args.satb, satb_chords=args.satb_chords)
    except QuotaExhausted as e:
        print(f"\nPAUSED: {e}\nRun the same command again later; the audio analysis is kept.")
        sys.exit(3)


def cmd_song_prepare(args):
    from .song.run import prepare
    wd = _work_dir(args, [args.audio])
    prepare(args.audio, wd, lyrics_text=_lyrics_arg(args), separate=not args.no_separate,
            whisper_model=args.whisper_model, language=args.language, shape_notes=args.shape_notes, title=args.title,
            satb=args.satb, satb_chords=args.satb_chords)
    print(f"\nReady. Open this folder in Claude Code (or Codex):\n  {Path(wd).resolve()}\n"
          "and say: Make the lead sheet.   (instructions are in CLAUDE.md / AGENTS.md)")


def cmd_song_check(args):
    from .song.run import check_dir
    r = check_dir(args.workdir)
    sys.exit(1 if r.get("errors") else 0)


def cmd_song_build(args):
    from .song.run import build_dir
    r = build_dir(args.workdir, args.output, args.shape_notes)
    sys.exit(1 if r.get("errors") else 0)


def cmd_gui(args):
    from .web import serve
    serve(port=args.port, open_browser=not args.no_browser)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scorescribe",
                                 description="Sheet music OCR with Claude -> MusicXML for MuseScore")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--staves-per-system", type=int, help="force grouping (e.g. 2 for piano) if detection is off")
        p.add_argument("--whole-pages", action="store_true", help="send whole pages instead of cropping systems")
        p.add_argument("--dpi", type=int, default=400, help="PDF render resolution (default 400)")
        p.add_argument("-w", "--workdir", help="work folder (default: <output>_work)")

    t = sub.add_parser("transcribe", help="PDF/images -> MusicXML")
    t.add_argument("inputs", nargs="+", help="PDF(s), image(s) or a folder of page images, in order")
    t.add_argument("-o", "--output", help="output .musicxml (default: next to the input)")
    t.add_argument("--parts", help="layout if you know it, e.g. 'Piano:2' or 'Violin,Piano:2' or 'S,A,T,B'")
    t.add_argument("--model", default=None,
                   help="model: claude-opus-5-5 (default), claude-sonnet-5-5, ... or a Gemini model such as gemini-3.8-flash")
    t.add_argument("--gemini", action="store_true",
                   help="use Google Gemini (default gemini-3.8-flash; works with a free key in GEMINI_API_KEY)")
    t.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max", "default"],
                   help="how hard Claude thinks (default high)")
    t.add_argument("--no-thinking", action="store_true", help="disable adaptive thinking (cheaper, less accurate)")
    t.add_argument("--workers", type=int, default=None, help="systems processed in parallel (default 4; 2 for Gemini)")
    t.add_argument("--repair-rounds", type=int, default=2, help="max fix-up rounds for measures that fail checks")
    t.add_argument("--no-proofread", action="store_true", help="skip the engraving-vs-original proofreading pass")
    t.add_argument("--passes", type=int, default=2, choices=[1, 2],
                   help="independent readings per system; 2 = read twice and settle disagreements (default)")
    t.add_argument("--no-zoom", action="store_true", help="don't send magnified views of each system")
    t.add_argument("--max-accuracy", action="store_true", help="two readings + xhigh effort (slowest, most accurate)")
    t.add_argument("--fast", action="store_true", help="one reading per system (cheaper, less accurate)")
    t.add_argument("--shape-notes", choices=["aiken", "4shape"],
                   help="write shape-note heads: aiken (7-shape do-re-mi) or 4shape (fa-so-la)")
    t.add_argument("--mark-uncertain", action="store_true",
                   help="colour notes Claude was unsure about orange (default: list them in the report only)")
    t.add_argument("--draft", help="MusicXML from another OMR program (e.g. Audiveris) for Claude to correct")
    t.add_argument("--audiveris", nargs="?", const="auto", help="run Audiveris first and use its output as the draft")
    t.add_argument("--api-key", help="API key (default: ANTHROPIC_API_KEY, or GEMINI_API_KEY for Gemini models)")
    t.add_argument("--fresh", action="store_true", help="ignore cached results in the work folder")
    common(t)
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("segment", help="only split pages into systems (free; check before spending tokens)")
    s.add_argument("inputs", nargs="+")
    s.add_argument("-o", "--output")
    common(s)
    s.set_defaults(func=cmd_segment)

    r = sub.add_parser("rebuild", help="rebuild MusicXML from a work folder after editing its JSON (no API calls)")
    r.add_argument("workdir")
    r.add_argument("-o", "--output")
    r.add_argument("--mark-uncertain", action="store_true", help="colour notes Claude was unsure about orange")
    r.add_argument("--shape-notes", choices=["aiken", "4shape"], help="write shape-note heads")
    r.set_defaults(func=cmd_rebuild)

    sh = sub.add_parser("shapes", help="convert any MusicXML file (e.g. exported from MuseScore) to shape notes")
    sh.add_argument("input", help=".musicxml, .xml or .mxl")
    sh.add_argument("-o", "--output", help="output file (default: <input>_aiken.musicxml)")
    sh.add_argument("--scheme", default="aiken", choices=["aiken", "4shape"], help="aiken (7-shape) or 4shape")
    sh.set_defaults(func=cmd_shapes)

    ap_ = sub.add_parser("agent-prepare",
                         help="no API key: prepare a folder for Claude Code / Codex to transcribe with your subscription")
    ap_.add_argument("inputs", nargs="+")
    ap_.add_argument("-o", "--output", help="(sets the work folder name)")
    ap_.add_argument("--parts", help="layout if you know it, e.g. 'Piano:2'")
    ap_.add_argument("--no-zoom", action="store_true", help="don't make magnified halves")
    ap_.add_argument("--shape-notes", choices=["aiken", "4shape"], help="shape-note heads in the final file")
    common(ap_)
    ap_.set_defaults(func=cmd_agent_prepare)

    ac = sub.add_parser("agent-check", help="check the JSON an agent wrote and draw proof engravings")
    ac.add_argument("workdir")
    ac.add_argument("--no-proofs", action="store_true", help="skip drawing the proof engravings")
    ac.set_defaults(func=cmd_agent_check)

    def song_opts(p):
        p.add_argument("audio", help="the recording: mp3, wav, m4a, flac, ogg ...")
        p.add_argument("--lyrics", help="the lyrics (a text file, or the text itself) - makes the words exact")
        p.add_argument("--title", help="song title")
        p.add_argument("--no-separate", action="store_true", help="skip Demucs vocal separation (faster, less accurate)")
        p.add_argument("--whisper-model", default="small", help="Whisper size: tiny, base, small (default), medium, large-v3")
        p.add_argument("--language", help="lyrics language code, e.g. en (default: detect)")
        p.add_argument("--shape-notes", choices=["aiken", "4shape"], help="shape-note heads in the output")
        p.add_argument("--satb", action="store_true",
                       help="also arrange the song for four-part choir (Claude writes alto, tenor and bass)")
        p.add_argument("--satb-chords", action="store_true", help="print chord symbols above the four-part score too")
        p.add_argument("-w", "--workdir", help="work folder (default: <output>_work)")

    sg = sub.add_parser("song", help="recording -> lead sheet (melody, lyrics, chords) using an API key")
    song_opts(sg)
    sg.add_argument("-o", "--output", help="output .musicxml (default: next to the recording)")
    sg.add_argument("--model", default=None, help="claude-opus-5-5 (default), claude-sonnet-5-5, or a Gemini model")
    sg.add_argument("--gemini", action="store_true", help="use Gemini (GEMINI_API_KEY)")
    sg.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max", "default"])
    sg.add_argument("--passes", type=int, default=2, help="draft + review passes (default 2)")
    sg.add_argument("--api-key")
    sg.add_argument("--fresh", action="store_true", help="redo the audio analysis")
    sg.set_defaults(func=cmd_song)

    sp = sub.add_parser("song-prepare", help="no API key: analyse a recording for Claude Code / Codex to transcribe")
    song_opts(sp)
    sp.add_argument("-o", "--output", help="(sets the work folder name)")
    sp.set_defaults(func=cmd_song_prepare)

    sc = sub.add_parser("song-check", help="check an agent-written leadsheet.json against the audio analysis")
    sc.add_argument("workdir")
    sc.set_defaults(func=cmd_song_check)

    sb = sub.add_parser("song-build", help="build the MusicXML + chord chart from leadsheet.json")
    sb.add_argument("workdir")
    sb.add_argument("-o", "--output")
    sb.add_argument("--shape-notes", choices=["aiken", "4shape"])
    sb.set_defaults(func=cmd_song_build)

    g = sub.add_parser("gui", help="open the browser interface")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    g.set_defaults(func=cmd_gui)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
