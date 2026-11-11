"""Automated visual check of the hand-authored letter guides.

Each guide is rasterised and compared with every letter A-Z / a-z rendered
from a real font with a school-style single-storey 'a' (Poppins). The font is
placed on the same three guide lines (baseline, x-height, ascender), so 'c'
and 'C' are told apart by size, as on lined paper. The guide's own letter
must be the closest match. The one accepted tie: in a sans-serif font,
lowercase 'l' and capital 'I' are both a plain vertical bar.

Skipped when the font or Pillow/SciPy are missing (they are not runtime
dependencies of the backend).
"""

from __future__ import annotations

import os
import string

import numpy as np
import pytest

from app.letter_templates import (ASCENDER_TOP, BASELINE, DESCENDER, GLYPHS, LOWER_DESCENDERS, LOWER_DOTTED,
                                  LOWERCASE, UPPERCASE, X_HEIGHT)

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
ndimage = pytest.importorskip("scipy.ndimage")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    os.path.expanduser("~/.fonts/Poppins-Regular.ttf"),
]
FONT = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
pytestmark = pytest.mark.skipif(FONT is None, reason="Poppins-Regular.ttf not installed")

PX = 2.0          # raster pixels per guide unit
G = 240           # raster covers -10..110 guide units (descenders reach 100)
TOL = 6.0 * (BASELINE - ASCENDER_TOP) / 80.0   # guide units: the original 6 units for an 80-unit letter box, kept relative


def _font_ink():
    f = ImageFont.truetype(FONT, 400)
    cap, xh, pp, ll = f.getbbox("H"), f.getbbox("x"), f.getbbox("p"), f.getbbox("l")
    base, xtop, ctop, dbot, ltop = cap[3], xh[1], cap[1], pp[3], ll[1]
    ink = {}
    for ch in string.ascii_letters:
        l, t, r, b = f.getbbox(ch)
        im = Image.new("L", (r - l + 20, b - t + 20), 0)
        ImageDraw.Draw(im).text((10 - l, 10 - t), ch, font=f, fill=255)
        ys, xs = np.nonzero(np.array(im) > 127)
        fx, fy = (xs + l - 10).astype(float), (ys + t - 10).astype(float)
        if ch.isupper():
            uy = BASELINE - (base - fy) / (base - ctop) * (BASELINE - ASCENDER_TOP)
        else:
            uy = np.where(fy >= xtop, BASELINE - (base - fy) / (base - xtop) * (BASELINE - X_HEIGHT),
                          X_HEIGHT - (xtop - fy) / (xtop - ltop) * (X_HEIGHT - ASCENDER_TOP))
            # (lowercase ascenders: the font's 'l' top -> the top line, as on school paper)
            # below the baseline: font descender bottom -> DESCENDER line
            uy = np.where(fy > base, BASELINE + (fy - base) / (dbot - base) * (DESCENDER - BASELINE), uy)
        ink[ch] = (fx, uy, (BASELINE - ASCENDER_TOP) / (base - ctop))
    return ink


def _raster(x, y):
    m = np.zeros((G, G), bool)
    m[np.clip(((y + 10) * PX).astype(int), 0, G - 1), np.clip(((x + 10) * PX).astype(int), 0, G - 1)] = True
    return m


def _points(strokes):
    out = []
    for s in strokes:
        s = np.array(s, float)
        for a, b in zip(s[:-1], s[1:]):
            out += list(np.linspace(a, b, max(2, int(np.hypot(*(b - a)) * PX))))
    return np.array(out)


def _score(strokes, ink) -> float:
    """Symmetric match: share of guide points near font ink and share of
    font ink near the guide (harmonic mean). Widths are matched; vertical
    positions are not (they carry the size information)."""
    p = _points(strokes)
    fx, fy = ink[0], ink[1]
    w_t, w_f = np.ptp(p[:, 0]), np.ptp(fx)
    # fx is in font pixels. Match widths when both letters are wide enough;
    # a narrow letter (l, i, I) keeps the font's own proportions instead
    # (font pixels -> guide units with the same scale as the vertical axis).
    s_font = ink[2]
    k = w_t / w_f if (w_t > 10 and w_f * s_font > 10) else s_font
    fxm = (fx - (fx.min() + fx.max()) / 2) * k + (p[:, 0].min() + p[:, 0].max()) / 2
    fm, tm = _raster(fxm, fy), _raster(p[:, 0], p[:, 1])
    d_f = ndimage.distance_transform_edt(~fm) / PX
    d_t = ndimage.distance_transform_edt(~tm) / PX
    a, b = np.mean(d_f[tm] <= TOL), np.mean(d_t[fm] <= TOL)
    return float(2 * a * b / (a + b + 1e-9))


@pytest.fixture(scope="module")
def ink():
    return _font_ink()


def test_all_52_letters_have_a_glyph():
    assert sorted(GLYPHS) == sorted(UPPERCASE + LOWERCASE)
    assert len(UPPERCASE) == len(LOWERCASE) == 26


def ranking_for(char, ink):
    scores = {c: _score(GLYPHS[char], ink[c]) for c in string.ascii_letters}
    return sorted(scores, key=scores.get, reverse=True)


@pytest.mark.parametrize("char", sorted(GLYPHS))
def test_guide_matches_its_own_letter_best(char, ink):
    ranking = ranking_for(char, ink)
    if char == "l":
        assert set(ranking[:2]) == {"l", "I"}, ranking[:4]
    elif char == "I":
        assert "I" in ranking[:3] and ranking[0] != "l", ranking[:4]
    else:
        assert ranking[0] == char, f"{char!r} guide looks most like {ranking[:3]}"


@pytest.mark.parametrize("char", list(LOWERCASE))
def test_upper_and_lowercase_guides_are_distinct(char):
    """The small letter is not the capital: the mean distance between the two
    drawings (symmetric nearest-point distance, guide units) is clearly
    above zero, for every pair including look-alikes such as c/C, o/O, s/S."""
    up, lo = _points(GLYPHS[char.upper()]), _points(GLYPHS[char])
    d_ul = np.sqrt(((up[:, None, :] - lo[None, :, :]) ** 2).sum(-1))
    mean_nn = (d_ul.min(1).mean() + d_ul.min(0).mean()) / 2
    assert mean_nn > 3.0, (char, round(float(mean_nn), 2))


def _expected_top(ch):
    if ch.isupper() or ch in "bdfhkl":
        return ASCENDER_TOP
    if ch == "t":
        return None          # t rises part-way above the x-height
    if ch in LOWER_DOTTED:
        return None          # the dot sits above the x-height
    return X_HEIGHT


def test_guides_respect_orientation_and_guide_lines():
    for ch, strokes in GLYPHS.items():
        pts = np.array([p for s in strokes for p in s])
        bottom = DESCENDER if ch in LOWER_DESCENDERS else BASELINE
        if ch == "Q":       # the tail of Q crosses the baseline a little
            assert BASELINE < pts[:, 1].max() <= BASELINE + 12, ch
        else:
            assert pts[:, 1].max() == pytest.approx(bottom, abs=1.0), ch      # sits on its line
        top = _expected_top(ch)
        if top is not None:
            assert pts[:, 1].min() == pytest.approx(top, abs=1.0), ch          # reaches its top line
        else:
            assert ASCENDER_TOP < pts[:, 1].min() < X_HEIGHT, ch
        if ch in LOWER_DOTTED:   # the body of i/j starts at the x-height
            assert min(p[1] for p in strokes[0]) == pytest.approx(X_HEIGHT, abs=1.0), ch
        assert 0 <= pts[:, 0].min() and pts[:, 0].max() <= 100, ch
        # first stroke starts in the top half of the letter (y grows downward)
        ref_top = top if top is not None else X_HEIGHT
        assert strokes[0][0][1] <= (ref_top + BASELINE) / 2 + 1, ch
