"""Engraving of a transcription back to an image (Verovio + resvg, or CairoSVG).

Used by the proofreading pass: Claude compares the original crop with an
engraving of what it transcribed, which catches wrong pitches and rhythms far
better than re-reading its own text.

resvg (pip package ``resvg-py``) is preferred because it ships ready-made
wheels for Windows, macOS and Linux. CairoSVG needs the Cairo C library,
which Windows doesn't have, so it is only a fallback.
"""
from __future__ import annotations

import io
import os
import threading

_LOCK = threading.Lock()  # Verovio's resource loading is not thread-safe


def _rasterizer():
    try:
        import resvg_py

        def raster(svg: str, width: int) -> bytes:
            return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, background="#ffffff"))
        return raster
    except Exception:
        pass
    try:
        import cairosvg  # raises OSError on import when the Cairo DLL is missing

        def raster(svg: str, width: int) -> bytes:
            return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, background_color="white")
        return raster
    except Exception:
        return None


def available() -> bool:
    try:
        import verovio  # noqa: F401
    except Exception:
        return False
    return _rasterizer() is not None


def missing_hint() -> str:
    try:
        import verovio  # noqa: F401
        has_v = True
    except Exception:
        has_v = False
    need = (["verovio"] if not has_v else []) + (["resvg-py"] if _rasterizer() is None else [])
    return "pip install " + " ".join(need) if need else ""


def musicxml_to_pngs(xml: str, width_px: int = 2400, page_width: int = 2400, scale: int = 45,
                     one_system_per_page: bool = False) -> list[bytes]:
    with _LOCK:
        return _render(xml, width_px, page_width, scale, one_system_per_page)


def _render(xml, width_px, page_width, scale, one_system_per_page):
    import verovio

    raster = _rasterizer()
    if raster is None:
        raise RuntimeError("no SVG rasterizer: " + missing_hint())
    tk = verovio.toolkit(False)
    # set explicitly: in worker threads Verovio otherwise looks in its build directory
    tk.setResourcePath(os.path.join(os.path.dirname(verovio.__file__), "data"))
    tk.setOptions({
        "pageWidth": page_width,
        "pageHeight": 60000 if one_system_per_page else 2970,
        "adjustPageHeight": True,
        "scale": scale,
        "header": "none",
        "footer": "none",
        "breaks": "auto",
        "svgViewBox": True,
        "mnumInterval": 1,
        "pageMarginLeft": 50, "pageMarginRight": 50, "pageMarginTop": 50, "pageMarginBottom": 50,
    })
    if not tk.loadData(xml):
        raise RuntimeError("Verovio could not load the MusicXML")
    return [raster(tk.renderToSVG(p), width_px) for p in range(1, tk.getPageCount() + 1)]


def stack_pngs(pngs: list[bytes]) -> bytes:
    from PIL import Image
    ims = [Image.open(io.BytesIO(b)).convert("L") for b in pngs]
    w = max(i.width for i in ims)
    h = sum(i.height for i in ims)
    canvas = Image.new("L", (w, h), 255)
    y = 0
    for i in ims:
        canvas.paste(i, (0, y))
        y += i.height
    buf = io.BytesIO()
    canvas.save(buf, "PNG", optimize=True)
    return buf.getvalue()
