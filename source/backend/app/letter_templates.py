"""
Letter and word guide templates (hand-authored, NOT clinically validated)
==========================================================================
All 52 English letters (A-Z, a-z) in a primary-school "ball-and-stick" print
style. Each glyph is a list of strokes. A stroke is a polyline [[x, y], ...]
in a 0-100 box with y pointing DOWN (screen coordinates). Strokes are drawn
in the listed order and direction; each pen lift starts a new stroke. Order
and direction follow common school print conventions: start at the top,
pull down, then slide across; round letters start near the top and go
anticlockwise (c, o, a, d, g, q) or clockwise after a stem (b, p). They are
a design choice, not a clinical standard.

Guide lines (four-line primary-school paper, three equal zones):
  ASCENDER_TOP = 10   capitals and tall letters (A, b, d, f, h, k, l)
  X_HEIGHT     = 40   top of short lowercase letters (a, c, m, o)
  BASELINE     = 70   every letter sits on it
  DESCENDER    = 100  tails of g, j, p, q, y
Before 2026-11-08 the box used ASCENDER_TOP 10, X_HEIGHT 45 and BASELINE 90,
which left only 10 units for descenders. The lines were moved so that
descenders fit inside the same 0-100 box. The 14 earlier glyphs keep their
shapes and stroke orders (rescaled to the new lines); N now starts at the
top like every other capital.

Correctness is checked automatically (tests/test_letter_templates.py): every
glyph is rasterised and compared with all 52 letters of a real font placed on
the same four lines; the intended letter must be the best match.
"""

from __future__ import annotations

import math

ASCENDER_TOP = 10.0
X_HEIGHT = 40.0
BASELINE = 70.0
DESCENDER = 100.0

Stroke = list[list[float]]

UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
LOWER_ASCENDERS = frozenset("bdfhklt")      # reach above the x-height (t only part-way)
LOWER_DESCENDERS = frozenset("gjpqy")       # go below the baseline
LOWER_DOTTED = frozenset("ij")


def _arc(cx: float, cy: float, rx: float, ry: float, deg_from: float, deg_to: float, n: int = 24) -> Stroke:
    """Elliptical arc; angles in degrees, mathematical orientation (0 = right,
    90 = up on screen). deg_to > deg_from runs anticlockwise on screen,
    deg_to < deg_from clockwise."""
    pts = []
    for i in range(n + 1):
        a = math.radians(deg_from + (deg_to - deg_from) * i / n)
        pts.append([round(cx + rx * math.cos(a), 2), round(cy - ry * math.sin(a), 2)])
    return pts


def _line(*pts: tuple[float, float]) -> Stroke:
    return [[float(x), float(y)] for x, y in pts]


def _join(*parts: Stroke) -> Stroke:
    """One continuous stroke made of several parts (no pen lift)."""
    out = [list(p) for p in parts[0]]
    for part in parts[1:]:
        out += [list(p) for p in (part[1:] if part and out and part[0] == out[-1] else part)]
    return out


def _dot(x: float, y: float, r: float = 3.0) -> Stroke:
    """A dot (i, j) drawn as a tiny closed circle so it is visible as ink."""
    return _arc(x, y, r, r, 90, 450, n=12)


A, X, B, D = ASCENDER_TOP, X_HEIGHT, BASELINE, DESCENDER
_MID = (A + B) / 2          # 40: middle of a capital
_XMID = (X + B) / 2         # 55: middle of a short lowercase letter
_XR = (B - X) / 2           # 15: radius of a lowercase bowl
_DOT_Y = 14.0               # dots of i and j, just below the top line

GLYPHS: dict[str, list[Stroke]] = {
    # ============================== UPPERCASE ==============================
    # A: slant down left, slant down right (both from the top), bar across
    "A": [_line((50, A), (18, B)), _line((50, A), (82, B)), _line((29.3, 52), (70.7, 52))],
    # B: stem down; one stroke: top bump, middle, bigger bottom bump
    "B": [_line((25, A), (25, B)),
          _join(_line((25, A), (52, A)), _arc(52, 25, 15, 15, 90, -90), _line((52, 40), (25, 40), (56, 40)),
                _arc(56, 55, 16, 15, 90, -90), _line((56, B), (25, B)))],
    # C: start near "1 o'clock" and curve anticlockwise to "5 o'clock"
    "C": [_arc(52, _MID, 30, 30, 45, 315)],
    # D: stem down; then across the top, round the right side, back along the bottom
    "D": [_line((25, A), (25, B)), _join(_line((25, A), (45, A)), _arc(45, _MID, 32, 30, 90, -90), _line((45, B), (25, B)))],
    # E: stem down, then top, middle and bottom bars left to right
    "E": [_line((25, A), (25, B)), _line((25, A), (75, A)), _line((25, _MID), (65, _MID)), _line((25, B), (75, B))],
    # F: stem down, top bar, middle bar
    "F": [_line((25, A), (25, B)), _line((25, A), (75, A)), _line((25, _MID), (65, _MID))],
    # G: like C, then up and in to make the shelf
    "G": [_join(_arc(52, _MID, 30, 30, 45, 340), _line((80.19, 50.26), (80, 42), (60, 42)))],
    # H: left stem, right stem, bar across the middle
    "H": [_line((20, A), (20, B)), _line((80, A), (80, B)), _line((20, _MID), (80, _MID))],
    # I: stem down, then the top and bottom bars (school print capital I)
    "I": [_line((50, A), (50, B)), _line((30, A), (70, A)), _line((30, B), (70, B))],
    # J: down, then hook to the left
    "J": [_join(_line((64, A), (64, 52)), _arc(46, 52, 18, 18, 0, -120))],
    # K: stem down; slant in to the middle and slant out to the baseline (one stroke)
    "K": [_line((25, A), (25, B)), _line((75, A), (25, 42), (77, B))],
    # L: one stroke, down then right along the baseline
    "L": [_line((25, A), (25, B), (75, B))],
    # M: left stem down; from the top: down to the middle, up, and the right stem down
    "M": [_line((15, A), (15, B)), _line((15, A), (50, 52), (85, A), (85, B))],
    # N: left stem down; from the top: slant down, then up the right side
    "N": [_line((20, A), (20, B)), _line((20, A), (80, B), (80, A))],
    # O: start at the top and go anticlockwise all the way round
    "O": [_arc(50, _MID, 28, 30, 90, 450, n=36)],
    # P: stem down; top bump
    "P": [_line((25, A), (25, B)), _join(_line((25, A), (52, A)), _arc(52, 26, 16, 16, 90, -90), _line((52, 42), (25, 42)))],
    # Q: O, then the tail, which crosses the baseline a little
    "Q": [_arc(50, _MID, 28, 30, 90, 450, n=36), _line((60, 56), (84, 80))],
    # R: like P, then the leg
    "R": [_line((25, A), (25, B)), _join(_line((25, A), (52, A)), _arc(52, 26, 16, 16, 90, -90), _line((52, 42), (25, 42))),
          _line((46, 42), (76, B))],
    # S: one stroke: upper curve anticlockwise from the top right, lower curve clockwise
    "S": [_arc(50, 25, 26, 15, 25, 270) + _arc(50, 55, 26, 15, 90, -155)[1:]],
    # T: down from the middle of the top, then the top bar left to right
    "T": [_line((50, A), (50, B)), _line((15, A), (85, A))],
    # U: down, round the bottom, up
    "U": [_join(_line((20, A), (20, 46)), _arc(50, 46, 30, 24, 180, 360), _line((80, 46), (80, A)))],
    # V: slant down, slant up (one stroke)
    "V": [_line((18, A), (50, B), (82, A))],
    # W: down, up, down, up (one stroke)
    "W": [_line((8, A), (29, B), (50, 26), (71, B), (92, A))],
    # X: slant down to the right, then slant down to the left
    "X": [_line((20, A), (80, B)), _line((80, A), (20, B))],
    # Y: slant in to the middle; from the right: slant in, then straight down
    "Y": [_line((20, A), (50, _MID)), _line((80, A), (50, _MID), (50, B))],
    # Z: across, slant down, across (one stroke)
    "Z": [_line((20, A), (80, A), (20, B), (80, B))],

    # ============================== lowercase ==============================
    # a (single-storey): round anticlockwise from the top, then the stem down on the right
    "a": [_arc(45, _XMID, 16, _XR, 90, 450, n=36), _line((61, X), (61, B))],
    # b: tall stem down; then round clockwise from the stem
    "b": [_line((30, A), (30, B)), _arc(46, _XMID, 16, _XR, 180, -180, n=36)],
    # c: like C, between the x-height and the baseline
    "c": [_arc(52, _XMID, 16, _XR, 45, 315)],
    # d: round anticlockwise, then the tall stem down on the right
    "d": [_arc(44, _XMID, 16, _XR, 90, 450, n=36), _line((60, A), (60, B))],
    # e: across the middle, then round anticlockwise over the top and under
    "e": [_join(_line((32, _XMID), (64, _XMID)), _arc(48, _XMID, 16, _XR, 0, 315))],
    # f: hook at the top, down to the baseline; then the cross bar
    "f": [_join(_arc(58, 22, 12, 12, 20, 180), _line((46, 22), (46, B))), _line((32, X), (62, X))],
    # g (single-storey): round anticlockwise, then stem down and hook left below the line
    "g": [_arc(45, _XMID, 16, _XR, 90, 450, n=36), _join(_line((61, X), (61, 80)), _arc(46, 80, 15, 20, 0, -185))],
    # h: tall stem down, back up, hump over and down
    "h": [_join(_line((28, A), (28, B), (28, _XMID)), _arc(44, _XMID, 16, _XR, 180, 0), _line((60, _XMID), (60, B)))],
    # i: stem down; then the dot
    "i": [_line((50, X), (50, B)), _dot(50, _DOT_Y)],
    # j: stem down below the line, hook left; then the dot
    "j": [_join(_line((58, X), (58, 82)), _arc(44, 82, 14, 18, 0, -110)), _dot(58, _DOT_Y)],
    # k: tall stem down; slant in and slant out (one stroke)
    "k": [_line((30, A), (30, B)), _line((62, X), (30, 57), (64, B))],
    # l: one straight stroke down from the top line
    "l": [_line((50, A), (50, B))],
    # m: one stroke: stem down, back up, first hump, back up, second hump
    "m": [_join(_line((20, X), (20, B), (20, 52)), _arc(33.5, 52, 13.5, 12, 180, 0, n=12), _line((47, 52), (47, B), (47, 52)),
                _arc(60.5, 52, 13.5, 12, 180, 0, n=12), _line((74, 52), (74, B)))],
    # n: stem down, back up, hump over and down
    "n": [_join(_line((28, X), (28, B), (28, _XMID)), _arc(44, _XMID, 16, _XR, 180, 0), _line((60, _XMID), (60, B)))],
    # o: like O, between the x-height and the baseline
    "o": [_arc(50, _XMID, 14, _XR, 90, 450, n=36)],
    # p: stem down below the line; then round clockwise from the stem
    "p": [_line((30, X), (30, D)), _arc(46, _XMID, 16, _XR, 180, -180, n=36)],
    # q: round anticlockwise, then the stem down below the line
    "q": [_arc(44, _XMID, 16, _XR, 90, 450, n=36), _line((60, X), (60, D))],
    # r: stem down, back up, small arch to the right
    "r": [_join(_line((32, X), (32, B), (32, _XMID)), _arc(48, _XMID, 16, _XR, 180, 60))],
    # s: small S between the x-height and the baseline
    "s": [_arc(50, 47.5, 14, 7.5, 25, 270) + _arc(50, 62.5, 14, 7.5, 90, -155)[1:]],
    # t: stem down from part-way up, curving right at the bottom; then the cross bar
    "t": [_join(_line((44, 18), (44, 62)), _arc(52, 62, 8, 8, 180, 300)), _line((32, X), (62, X))],
    # u: down, round the bottom, up, then the stem back down
    "u": [_join(_line((28, X), (28, _XMID)), _arc(44, _XMID, 16, _XR, 180, 360), _line((60, _XMID), (60, X), (60, B)))],
    # v: slant down, slant up
    "v": [_line((30, X), (50, B), (70, X))],
    # w: down, up, down, up
    "w": [_line((18, X), (34, B), (50, 47), (66, B), (82, X))],
    # x: slant down to the right, then slant down to the left
    "x": [_line((32, X), (68, B)), _line((68, X), (32, B))],
    # y: short slant in to the baseline; from the right: long slant down below the line
    "y": [_line((30, X), (50, B)), _line((70, X), (30, D))],
    # z: across, slant down, across
    "z": [_line((30, X), (70, X), (30, B), (70, B))],
}
assert set(GLYPHS) == set(UPPERCASE) | set(LOWERCASE), "every letter needs exactly one glyph"

# Horizontal advance used when glyphs are placed side by side in a word
# (in glyph-box units; each glyph is drawn in a 0-100 box).
WORD_ADVANCE = 90.0


def letter_zone(char: str) -> str:
    """Where a letter sits on the four lines: 'capital', 'ascender',
    'x_height' or 'descender' (used by the app to explain the letter)."""
    if char.isupper():
        return "capital"
    if char in LOWER_DESCENDERS:
        return "descender"
    if char in LOWER_ASCENDERS:
        return "ascender"
    return "x_height"


def glyph_strokes(char: str) -> list[Stroke]:
    if char not in GLYPHS:
        raise KeyError(f"no guide glyph for {char!r}")
    return [[list(p) for p in s] for s in GLYPHS[char]]


def word_strokes(word: str) -> list[Stroke]:
    """Glyph strokes placed left to right, letter by letter (no joins)."""
    out: list[Stroke] = []
    for i, ch in enumerate(word):
        dx = i * WORD_ADVANCE
        out += [[[round(x + dx, 2), y] for x, y in s] for s in glyph_strokes(ch)]
    return out


def flatten(strokes: list[Stroke]) -> list[list[float]]:
    """Concatenated points of all strokes, in drawing order. This is the
    backward-compatible `reference_template` (old clients draw it as one
    polyline; the feature extractor compares the child's concatenated
    strokes with it)."""
    return [list(p) for s in strokes for p in s]
