"""Optionally run Audiveris first and hand its MusicXML to Claude as a draft."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_audiveris(hint: str | None = None) -> str | None:
    if hint and hint not in ("auto", "1", "true"):
        return hint if Path(hint).exists() or shutil.which(hint) else None
    for name in ("audiveris", "Audiveris", "audiveris.bat", "Audiveris.bat"):
        p = shutil.which(name)
        if p:
            return p
    for p in ("/Applications/Audiveris.app/Contents/MacOS/Audiveris",
              r"C:\Program Files\Audiveris\Audiveris.exe", r"C:\Program Files\Audiveris\bin\Audiveris.bat",
              "/opt/audiveris/bin/Audiveris"):
        if Path(p).exists():
            return p
    return None


def run_audiveris(exe: str, inputs: list[str], outdir: str, log=print) -> str | None:
    """Runs Audiveris in batch mode and returns the path of the exported MusicXML (or None)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-batch", "-export", "-output", str(out), "--", *[str(Path(i).resolve()) for i in inputs]]
    log("Running Audiveris for a first draft: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=False, timeout=60 * 30, capture_output=True, text=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"  Audiveris failed: {e}")
        return None
    found = sorted(out.rglob("*.mxl")) + sorted(out.rglob("*.musicxml")) + sorted(out.rglob("*.xml"))
    if not found:
        log("  Audiveris produced no MusicXML; continuing without a draft")
        return None
    if len(found) > 1:
        log(f"  Audiveris wrote {len(found)} files (movements); using {found[0].name}")
    return str(found[0])
