"""Page loading, deskewing, staff detection and system cropping.

Each system is cropped and sent to Claude on its own, so the model sees the
notes at (close to) full scan resolution instead of a downscaled full page.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
MAX_EDGE = 2576          # high-resolution vision tier (Opus 4.7 and later)
MAX_PIXELS = 3_500_000   # ~4784 visual tokens


@dataclass
class Staff:
    top: float
    bottom: float
    space: float
    x0: int
    x1: int
    lines: list[float] = field(default_factory=list)


@dataclass
class SystemCrop:
    page: int            # 1-based
    index: int           # 1-based, across the whole score
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 on the page image
    n_staves: int
    image: Image.Image   # grayscale crop
    staves: list[Staff] = field(default_factory=list)   # coordinates relative to the crop
    barlines: list[float] = field(default_factory=list)  # x of each measure-ending barline (crop coords)

    @property
    def measure_count(self) -> int | None:
        """Measures counted from detected barlines, or None when detection isn't usable."""
        return len(self.barlines) if self.barlines else None

    def png_bytes(self, max_edge: int = MAX_EDGE, max_pixels: int = MAX_PIXELS) -> bytes:
        return _png(fit_for_claude(self.image, max_edge, max_pixels))

    def zoom_pngs(self, max_edge: int = MAX_EDGE, max_pixels: int = MAX_PIXELS) -> list[bytes]:
        """Two overlapping, magnified halves of a wide system (empty list if not worth it).

        Cuts are placed on a detected barline near the middle when possible, so
        no measure is split, and the halves overlap a little either way.
        """
        w, h = self.image.size
        if w < 1.5 * h or w < 900:
            return []
        mid = w / 2
        cut = mid
        inner = [x for x in self.barlines[:-1] if 0.3 * w < x < 0.7 * w]
        if inner:
            cut = min(inner, key=lambda x: abs(x - mid))
        ov = int(0.06 * w)
        pieces = [(0, min(w, int(cut) + ov)), (max(0, int(cut) - ov), w)]
        out = []
        for x0, x1 in pieces:
            piece = self.image.crop((x0, 0, x1, h))
            pw, ph = piece.size
            sc = min(max_edge / max(pw, ph), math.sqrt(max_pixels / (pw * ph)))
            if sc > 1.05:
                piece = piece.resize((int(pw * sc), int(ph * sc)), Image.LANCZOS)
            out.append(_png(fit_for_claude(piece, max_edge, max_pixels)))
        return out


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def fit_for_claude(im: Image.Image, max_edge: int = MAX_EDGE, max_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = im.size
    s = min(1.0, max_edge / max(w, h), math.sqrt(max_pixels / (w * h)))
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


# ------------------------------------------------------------------ loading

def load_pages(paths: list[str | Path], dpi: int = 300) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT | {".pdf"})
            pages.extend(load_pages(files, dpi))
            continue
        if p.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(p))
            for i in range(len(pdf)):
                pil = pdf[i].render(scale=dpi / 72.0, grayscale=True).to_pil().convert("L")
                pages.append(np.array(pil))
            pdf.close()
        elif p.suffix.lower() in IMAGE_EXT:
            im = Image.open(p)
            n = getattr(im, "n_frames", 1)
            for f in range(n):
                im.seek(f)
                frame = im.convert("RGBA")
                bg = Image.new("RGBA", frame.size, (255, 255, 255, 255))
                pages.append(np.array(Image.alpha_composite(bg, frame).convert("L")))
        else:
            raise ValueError(f"unsupported file type: {p}")
    return pages


# ---------------------------------------------------------------- cleanup

def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def _rotate(img: np.ndarray, angle: float, border=255) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


def estimate_skew(gray: np.ndarray) -> float:
    scale = 1200 / max(gray.shape[1], 1)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    b = binarize(small)

    def score(a):
        r = _rotate(b, a, border=0)
        prof = r.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    best = max(np.arange(-3.0, 3.01, 0.25), key=score)
    best = max(np.arange(best - 0.25, best + 0.251, 0.05), key=score)
    return float(best)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    a = estimate_skew(gray)
    if abs(a) < 0.04:
        return gray, 0.0
    return _rotate(gray, a), a


# ------------------------------------------------------------ staff finding

def _row_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    runs = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r - prev > 1:
            runs.append((int(start), int(prev)))
            start = r
        prev = r
    runs.append((int(start), int(prev)))
    return runs


def find_staves(bin_img: np.ndarray) -> list[Staff]:
    h, w = bin_img.shape
    kw = max(20, w // 30)
    lines_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    prof = (lines_img > 0).sum(axis=1)
    if prof.max() == 0:
        return []
    # pass 1: confident (long) lines give the staff space and line thickness
    strong = _row_runs(np.where(prof >= max(0.25 * prof.max(), w * 0.08))[0])
    if len(strong) < 4:
        return []
    thick = float(np.median([b - a + 1 for a, b in strong]))
    c = np.array([(a + b) / 2 for a, b in strong])
    gaps = np.round(np.diff(c)).astype(int)
    gaps = gaps[gaps > thick * 2]
    if len(gaps) == 0:
        return []
    vals, counts = np.unique(gaps, return_counts=True)
    sp = float(vals[np.argmax(counts)])
    near = gaps[np.abs(gaps - sp) <= 1]
    sp = float(np.mean(near)) if len(near) else sp
    if sp < 4:
        return []
    # pass 2: comb filter - a staff is 5 rows spaced `sp` apart that are all
    # (or 4 of 5, when a beam or slur hides one) covered by long horizontal ink
    prof_s = np.maximum.reduce([np.roll(prof, k) for k in (-1, 0, 1)])
    thr_lo = max(w * 0.04, 3 * sp)
    offs = np.round(np.arange(5) * sp).astype(int)
    cands = []
    for y in range(0, h - offs[-1]):
        vals = prof_s[y + offs]
        if (vals >= thr_lo).sum() >= 4:
            cands.append((float(vals.sum()), y))
    cands.sort(reverse=True)
    taken: list[int] = []
    for _, y in cands:
        if all(abs(y - t) > 4.6 * sp for t in taken):
            taken.append(y)
    staves: list[Staff] = []
    half = max(1, int(round(thick)))
    for y in sorted(taken):
        ys = []
        count = np.zeros(w, dtype=np.int32)
        for k in range(5):
            c = y + offs[k]
            lo, hi = max(0, int(c - 0.25 * sp)), min(h, int(c + 0.25 * sp) + 1)
            r = lo + int(np.argmax(prof[lo:hi]))
            ys.append(float(r))
            # a band of rows, so a line drifting by a pixel (slight skew / warp) still counts
            band = lines_img[max(0, r - half):min(h, r + half + 1)].max(axis=0) > 0
            count += band
        cols = np.where(count >= 4)[0]
        if len(cols) == 0:
            continue
        # staff extent = longest run of columns where (almost) all five lines are present,
        # bridging small gaps (e.g. where a thick barline or notehead interrupts the opening)
        runs, a, p = [], int(cols[0]), int(cols[0])
        for v in cols[1:]:
            v = int(v)
            if v - p > max(3, int(0.5 * sp)):
                runs.append((a, p))
                a = v
            p = v
        runs.append((a, p))
        x0, x1 = max(runs, key=lambda r: r[1] - r[0])
        if x1 - x0 < 6 * sp:
            continue
        staves.append(Staff(ys[0], ys[-1], sp, x0, x1, ys))
    return staves


def _connected(bin_img: np.ndarray, a: Staff, b: Staff) -> bool:
    """True if a vertical line (system barline / bracket) joins staves a and b at the left."""
    y0, y1 = int(a.bottom + 2), int(b.top - 2)
    if y1 <= y0:
        return True
    sp = a.space
    x = min(a.x0, b.x0)
    xa, xb = max(0, int(x - 2.5 * sp)), int(x + 1.0 * sp)
    region = bin_img[y0:y1, xa:xb] > 0
    if region.size == 0:
        return False
    col_fill = region.mean(axis=0)
    return bool(col_fill.max() >= 0.92)


def group_systems(bin_img: np.ndarray, staves: list[Staff], staves_per_system: int | None = None) -> list[list[Staff]]:
    if not staves:
        return []
    if staves_per_system:
        k = staves_per_system
        return [staves[i:i + k] for i in range(0, len(staves), k)]
    systems = [[staves[0]]]
    for a, b in zip(staves, staves[1:]):
        if _connected(bin_img, a, b):
            systems[-1].append(b)
        else:
            systems.append([b])
    return systems


# ---------------------------------------------------------------- barlines

def find_barlines(b: np.ndarray, staves: list[Staff]) -> list[float]:
    """x positions of barlines that END a measure, in the coordinates of `b`.

    A barline is a thin vertical stroke covering a staff from its top line to its
    bottom line. In a multi-staff system it must be found on every staff at the
    same x (that rejects stems); on a single staff, strokes with a notehead at
    either end or that run past the staff are rejected. Double and repeat
    barlines are merged, and a start-repeat right after the clef/key/time
    signature is not counted as a measure. Returns [] when nothing is found.
    """
    if not staves:
        return []
    h, w = b.shape
    sp = float(np.median([s.space for s in staves]))
    lines_img = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(3 * sp)), 1)))
    nolines = cv2.subtract(b, lines_img)
    # close small gaps left where staff lines were removed, then keep tall strokes
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(3.6 * sp)))))
    per_staff: list[list[tuple[float, float]]] = []  # (x, width)
    for st in staves:
        t, bt = int(round(st.top)), int(round(st.bottom))
        x_lo, x_hi = max(0, st.x0 - 2), min(w, st.x1 + 3)
        band = vert[t:bt + 1, x_lo:x_hi] > 0
        if band.size == 0:
            per_staff.append([])
            continue
        cover = band.mean(axis=0)
        cols = np.where(cover >= 0.9)[0]
        cands = []
        for a, z in _runs(cols):
            x = x_lo + (a + z) / 2
            width = z - a + 1
            if width > 0.8 * sp:          # a wide block (beam end, cluster), not a line
                continue
            if len(staves) == 1 and not _clean_ends(nolines, vert, st, int(x_lo + a), int(x_lo + z), sp):
                continue
            cands.append((x, float(width)))
        per_staff.append(cands)

    if len(staves) == 1:
        xs = per_staff[0]
    else:
        # must appear on every staff at the same x
        cons = []
        for x, wd in per_staff[0]:
            if all(any(abs(x - y) <= 0.6 * sp for y, _ in other) for other in per_staff[1:]):
                cons.append((x, wd))
        # Where barlines run through the gap between two staves (piano, organ, grouped
        # parts), require that: stems never bridge the gap. Gaps that barlines don't
        # cross (choir staves with lyrics in between) fall back to the stem filter.
        def crosses(x, a, c):
            y0, y1 = int(a.bottom + 0.5 * sp), int(c.top - 0.5 * sp)
            if y1 <= y0:
                return True
            xi = int(round(x))
            col = b[y0:y1, max(0, xi - 2):min(w, xi + 3)].max(axis=1) > 0
            return bool(col.mean() >= 0.9)
        gaps = list(zip(staves, staves[1:]))
        cross = [[crosses(x, a, c) for a, c in gaps] for x, _ in cons]
        joined = [sum(cr[g] for cr in cross) >= max(2, 0.3 * len(cons)) for g in range(len(gaps))]
        xs = []
        for (x, wd), cr in zip(cons, cross):
            if any(j and not c for j, c in zip(joined, cr)):
                continue
            if not any(joined):
                xi_a, xi_z = int(x - wd / 2), int(x + wd / 2)
                if not all(_clean_ends(nolines, vert, st, xi_a, xi_z, sp) for st in staves):
                    continue
            xs.append((x, wd))
    if not xs:
        return []
    # merge double / repeat barlines
    groups: list[list[tuple[float, float]]] = [[xs[0]]]
    for x, wd in xs[1:]:
        if x - groups[-1][-1][0] <= 1.3 * sp:
            groups[-1].append((x, wd))
        else:
            groups.append([(x, wd)])
    start = min(st.x0 for st in staves)
    end = max(st.x1 for st in staves)
    thin = float(np.median([wd for g in groups for _, wd in g]))
    out = []
    for i, g in enumerate(groups):
        x = g[-1][0]
        if x - start < 1.0 * sp:      # the system's opening line
            continue
        has_thick = any(wd >= max(2.2 * thin, 0.3 * sp) for _, wd in g)
        if not out and has_thick and x - start < 16 * sp and end - x > 8 * sp:
            continue                  # start-repeat after clef/key/time signature
        out.append(float(x))
    # the system normally ends with a barline at the right edge of the staff
    if out and end - out[-1] > 2.5 * sp:
        out.append(float(end))
    return out


def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    if len(idx) == 0:
        return []
    out = []
    a = p = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v != p + 1:
            out.append((a, p))
            a = v
        p = v
    out.append((a, p))
    return out


def _clean_ends(nolines: np.ndarray, vert: np.ndarray, st: Staff, xa: int, xz: int, sp: float) -> bool:
    """For single staves: reject stems (notehead blob at an end, or a stroke running past the staff)."""
    h, w = nolines.shape
    t, bt = int(round(st.top)), int(round(st.bottom))
    col = vert[:, xa:xz + 1].max(axis=1) > 0
    above = col[max(0, int(t - 0.8 * sp)):max(0, int(t - 0.35 * sp))]
    below = col[min(h, int(bt + 0.35 * sp)):min(h, int(bt + 0.8 * sp))]
    if (above.size and above.mean() > 0.5) or (below.size and below.mean() > 0.5):
        return False
    r = int(1.3 * sp)
    for y0 in (t - int(0.6 * sp), bt - int(0.6 * sp)):
        y0 = max(0, y0)
        win = nolines[y0:y0 + int(1.2 * sp), max(0, xa - r):min(w, xz + r + 1)] > 0
        stroke = nolines[y0:y0 + int(1.2 * sp), xa:xz + 1] > 0
        extra = win.sum() - stroke.sum()
        if extra > 0.45 * sp * sp:    # a notehead next to the stroke
            return False
    return True


# --------------------------------------------------------------- cropping

def crop_systems(pages: list[np.ndarray], staves_per_system: int | None = None,
                 do_deskew: bool = True, whole_pages: bool = False,
                 log=print) -> tuple[list[SystemCrop], list[Image.Image]]:
    crops: list[SystemCrop] = []
    overviews: list[Image.Image] = []
    idx = 1
    for p_i, gray in enumerate(pages, start=1):
        if do_deskew:
            gray, ang = deskew(gray)
            if ang:
                log(f"  page {p_i}: corrected skew of {ang:+.2f} deg")
        b = binarize(gray)
        h, w = gray.shape
        staves = [] if whole_pages else find_staves(b)
        systems = group_systems(b, staves, staves_per_system)
        overview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if not systems:
            if not whole_pages:
                log(f"  page {p_i}: no staves detected, sending the whole page")
            box = _ink_box(b, 0, h, w, pad=20)
            crops.append(SystemCrop(p_i, idx, box, 0, Image.fromarray(gray[box[1]:box[3], box[0]:box[2]])))
            crops[-1].barlines = []
            idx += 1
            overviews.append(Image.fromarray(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB)))
            continue
        bounds = []
        for s_i, sy in enumerate(systems):
            top, bot = sy[0].top, sy[-1].bottom
            sp = sy[0].space
            y0 = (systems[s_i - 1][-1].bottom + top) / 2 if s_i > 0 else max(0, top - 9 * sp)
            y1 = (bot + systems[s_i + 1][0].top) / 2 if s_i + 1 < len(systems) else min(h, bot + 9 * sp)
            if s_i == 0 and p_i == 1:
                y0 = 0  # keep the title / tempo marking with the first system
            bounds.append((int(y0), int(y1)))
        for s_i, (sy, (y0, y1)) in enumerate(zip(systems, bounds)):
            box = _ink_box(b, y0, y1, w, pad=int(sy[0].space * 1.5))
            box = (box[0], y0, box[2], y1)
            rel = [Staff(st.top - y0, st.bottom - y0, st.space, st.x0 - box[0], st.x1 - box[0],
                         [y - y0 for y in st.lines]) for st in sy]
            crop_b = b[y0:y1, box[0]:box[2]]
            try:
                bars = find_barlines(crop_b, rel)
            except Exception:
                bars = []
            crops.append(SystemCrop(p_i, idx, box, len(sy), Image.fromarray(gray[y0:y1, box[0]:box[2]]), rel, bars))
            for x in bars:
                cv2.line(overview, (int(box[0] + x), int(sy[0].top - 2 * sy[0].space)),
                         (int(box[0] + x), int(sy[-1].bottom + 2 * sy[0].space)), (30, 160, 60), 3)
            cv2.rectangle(overview, (box[0], y0 + 3), (box[2], y1 - 3), (40, 40, 220), 4)
            label = f"S{idx} ({len(sy)} staves" + (f", {len(bars)} measure{'s' if len(bars) != 1 else ''})" if bars else ")")
            cv2.putText(overview, label, (box[0] + 10, y0 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 220), 4)
            idx += 1
        log(f"  page {p_i}: {len(staves)} staves in {len(systems)} systems")
        overviews.append(Image.fromarray(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB)))
    return crops, overviews


def _ink_box(b: np.ndarray, y0: int, y1: int, w: int, pad: int = 20) -> tuple[int, int, int, int]:
    band = b[y0:y1]
    cols = np.where(band.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return (0, y0, w, y1)
    x0 = max(0, int(cols.min()) - pad)
    x1 = min(w, int(cols.max()) + pad)
    return (x0, y0, x1, y1)
