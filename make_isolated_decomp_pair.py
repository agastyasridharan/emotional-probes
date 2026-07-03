#!/usr/bin/env python3
"""
Concatenate the two isolated s2 score-decomposition figures (qwen3-32b and
qwen3-235b) horizontally into a single side-by-side image, in the same folder.
Left = qwen3-32b, right = qwen3-235b (ascending model size). Pixel-exact: the
existing PNGs are pasted unchanged (flattened onto white), with a small white
gap between them.

Run:  python3 make_isolated_decomp_pair.py
"""

from pathlib import Path
from PIL import Image

FOLDER = Path(__file__).resolve().parent / "figures" / "1_effect" / "s2_score_decomposition_isolated"
PANELS = ["qwen3-32b.png", "qwen3-235b.png"]
GAP = 24          # white pixels between panels
OUT = FOLDER / "pair_qwen3-32b_qwen3-235b.png"


def flatten(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    return bg


def main():
    imgs = [flatten(FOLDER / p) for p in PANELS]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + GAP * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, (h - im.height) // 2))   # vertically centred
        x += im.width + GAP
    canvas.save(OUT)
    print(f"wrote {OUT}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
