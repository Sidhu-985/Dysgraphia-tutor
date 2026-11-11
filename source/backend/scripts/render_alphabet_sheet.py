"""Render all 52 letter guides (four-line paper, stroke colours, numbered
start dots, direction arrows) to a PNG contact sheet for visual review.
Usage (from backend/): python scripts/render_alphabet_sheet.py OUT.png"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.letter_templates import ASCENDER_TOP, BASELINE, DESCENDER, GLYPHS, LOWERCASE, UPPERCASE, X_HEIGHT  # noqa: E402

COLORS = [(30, 111, 104), (110, 91, 168), (180, 83, 42), (47, 108, 168)]
CELL, PAD, S = 190, 10, 1.7          # cell px, padding, px per guide unit
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    LABEL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except OSError:
    FONT = LABEL = ImageFont.load_default()


def cell(ch):
    im = Image.new("RGB", (CELL, CELL + 22), (255, 253, 248))
    d = ImageDraw.Draw(im)
    oy = PAD

    def P(x, y):
        return (PAD + 5 + x * S, oy + y * S)
    for y, w, col in ((ASCENDER_TOP, 2, (190, 180, 220)), (X_HEIGHT, 1, (200, 190, 230)),
                      (BASELINE, 3, (160, 145, 210)), (DESCENDER, 1, (215, 210, 235))):
        if y == X_HEIGHT:
            for x0 in range(0, 100, 6):
                d.line([P(x0, y), P(x0 + 3, y)], fill=col, width=w)
        else:
            d.line([P(0, y), P(100, y)], fill=col, width=w)
    for i, s in enumerate(GLYPHS[ch]):
        col = COLORS[i % 4]
        d.line([P(*p) for p in s], fill=col, width=6, joint="curve")
        # arrow at the middle of the stroke
        L = [math.dist(s[k], s[k + 1]) for k in range(len(s) - 1)]
        tot, acc = sum(L), 0.0
        for k, l in enumerate(L):
            if acc + l >= tot * 0.55 and l > 0:
                f = (tot * 0.55 - acc) / l
                x = s[k][0] + f * (s[k + 1][0] - s[k][0]); y = s[k][1] + f * (s[k + 1][1] - s[k][1])
                h = math.atan2(s[k + 1][1] - s[k][1], s[k + 1][0] - s[k][0])
                for a in (h - 0.6, h + 0.6):
                    d.line([P(x - 5 * math.cos(a), y - 5 * math.sin(a)), P(x, y)], fill="white", width=3)
                break
            acc += l
    placed = []
    for i, s in enumerate(GLYPHS[ch]):
        cx, cy = P(*s[0])
        if any(math.dist((cx, cy), q) < 20 for q in placed):   # same start as an earlier stroke: move along it
            tot = sum(math.dist(s[k], s[k + 1]) for k in range(len(s) - 1))
            want, acc = min(0.45 * tot, 24 / S), 0.0
            for k in range(len(s) - 1):
                l = math.dist(s[k], s[k + 1])
                if acc + l >= want and l > 0:
                    f = (want - acc) / l
                    cx, cy = P(s[k][0] + f * (s[k + 1][0] - s[k][0]), s[k][1] + f * (s[k + 1][1] - s[k][1]))
                    break
                acc += l
        placed.append((cx, cy))
        d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(46, 139, 58), outline="white", width=2)
        d.text((cx, cy), str(i + 1), fill="white", font=FONT, anchor="mm")
    d.text((CELL / 2, CELL + 8), ("capital " if ch.isupper() else "small ") + ch, fill=(31, 42, 46), font=LABEL,
           anchor="mm")
    return im


def sheet(chars, cols=9):
    rows = math.ceil(len(chars) / cols)
    out = Image.new("RGB", (cols * CELL, rows * (CELL + 22)), "white")
    for i, ch in enumerate(chars):
        out.paste(cell(ch), ((i % cols) * CELL, (i // cols) * (CELL + 22)))
    return out


if __name__ == "__main__":
    a, b = sheet(UPPERCASE), sheet(LOWERCASE)
    full = Image.new("RGB", (a.width, a.height + b.height + 10), "white")
    full.paste(a, (0, 0)); full.paste(b, (0, a.height + 10))
    full.save(sys.argv[1])
    print("saved", sys.argv[1], full.size)
