"""Render corpus scores to fake 'scanned' PDFs for testing (ground truth known)."""
import io, sys, random
import numpy as np
import cairosvg, verovio
from music21 import corpus
from PIL import Image, ImageFilter

out = sys.argv[1]
works = {"mozart_k545": "mozart/k545/movement1_exposition.mxl",
         "joplin": "joplin/maple_leaf_rag.mxl",
         "bach": "bach/bwv66.6"}
random.seed(3)
for name, path in works.items():
    s = corpus.parse(path)
    if name == "joplin":
        s = s.measures(1, 24)
    xml_path = f"{out}/{name}_truth.musicxml"
    s.write("musicxml", fp=xml_path)
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2159, "pageHeight": 2794, "scale": 40, "adjustPageHeight": False,
                   "footer": "none", "breaks": "auto"})
    tk.loadData(open(xml_path).read())
    pages = []
    for p in range(1, min(tk.getPageCount(), 3) + 1):
        png = cairosvg.svg2png(bytestring=tk.renderToSVG(p).encode(), output_width=2550, background_color="white")
        im = Image.open(io.BytesIO(png)).convert("L")
        # simulate a scan: slight rotation, blur, noise, grey paper
        im = im.rotate(random.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255)
        im = im.filter(ImageFilter.GaussianBlur(0.8))
        a = np.array(im).astype(np.float32) * 0.93 + 10
        a += np.random.normal(0, 6, a.shape)
        pages.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
    pages[0].save(f"{out}/{name}.pdf", save_all=True, append_images=pages[1:], resolution=300)
    print(name, len(pages), "pages")
