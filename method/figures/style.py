r"""The drawing style of the figures on torux's method page: ink on paper, as code.

Every figure of method/README.md is drawn with this module and nothing else, so that all
of them look inked by one draughtsman for one book, a physics monograph of
about 1962: black and white, a few weights of line, dashes, hatching and stipple, lettering
in the book's own face, nothing decorative. The module holds the whole vocabulary and writes
each figure twice, <name>-light.svg for GitHub's light page and <name>-dark.svg for its dark
one. This docstring is the whole manual.

Running
-------
Put the script in method/figures/ beside this file and run it with a python that has numpy
and matplotlib, such as the torux environment (built with numpy 2.5 and matplotlib 3.11).
Nothing else is needed: the faces come with matplotlib. Previews need rsvg-convert (Homebrew
librsvg).

    import numpy as np
    import style

    fig = style.Figure(height=420)                       # always 838 wide
    v = fig.view(box=(40, 20, 380, 380), x=(-1.2, 1.2), y=(-1.2, 1.2), equal=True)
    t = np.linspace(0, 2 * np.pi, 400)
    v.fill(np.cos(t), np.sin(t), tone="stipple")
    v.line(np.cos(t), np.sin(t), weight="heavy", closed=True)
    v.line([-1.15, 1.15], [0, 0], weight="fine", dash="dashdot")
    v.mark(0, 0)
    v.leader(0.95, -1.1, "last closed surface", 0.62, -0.79)
    fig.save("example", previews="/path/to/scratch")     # example-light.svg, example-dark.svg

save() writes both files beside this module (or at a path you give), refuses a file that is
not ink and paper alone, and prints their sizes. With previews= it also renders each at 838
and 324 px wide. Look at all four before calling a figure done: 324 px is a phone, and the
dark page is drawn with its own, finer, values, so it is a picture of its own to judge.

The page
--------
A figure is 838 units wide and as high as you make it. GitHub shows it 838 px wide on a
laptop, so a unit is a pixel there, and 324 px wide on a phone. The Figure's own calls take
page units, x from the left and y DOWN from the top. A View maps your coordinates, y UP,
onto a box of the page. Compute the geometry however you like (numpy, contourpy, torux) and
hand the module arrays.

    (0,0) +---------------------------------------------- 838 ---> x  (page units)
          |
          |   box=(left, top, width, height)
          |   +----------------------------------+
          |   | ^ y (yours)                      |   v = fig.view(box, x=(x0, x1),
          |   | |                                |                y=(y0, y1))
          |   | +-----> x (yours)                |   v.line(x_yours, y_yours)
          |   +----------------------------------+
          v
          y  (page units)

The vocabulary
--------------
Every choice is a word from these lists. There is no colour, no grey, no opacity, no other
face and no other size. If a figure needs a word that is missing, it is added here, for all.

    weight  fine        hatching, leaders, construction, a family of many curves
            medium      the ordinary line: outlines, frames, curves (the default)
            heavy       the one thing the figure is about: once, twice at most

    dash    solid       ----------   what is there and seen (the default)
            dashed      --  --  --   hidden or behind, such as the back of a torus, or a level
            dotted      . . . . . .  a reference: a rate, an asymptote, an extrapolation
            dashdot     ---- . ----  an axis or line of symmetry (the chain line)
            dashdotdot  ---- .. ---  a second family, an alternative position

    tone    stipple     light        a region to notice, not to weigh (no surfaces here)
            hatch       middle       a band or region that matters (a wall, an island band)
            crosshatch  dark         the one region that has to stand out
            ink         solid        small shapes only
            paper       blank        covers what is under it, to hide lines (painter's order)
            flip=True hatches the other way, for a region that meets another hatched one.

    size    normal      every label, symbol and axis name (22 units)
            small       the numbers on axes, and an aside (19 units)

    mark    dot         a point that matters: a magnetic axis, an O point
            circle      data
            cross       an X point drawn on its own, or a point that is not there

Lettering. Words are upright and whatever is between dollars is mathematics, with italic
symbols, as in the text of a book: "magnetic axis", "$\psi$", "surface $\psi=0.5$",
"$\mathbf{B}\cdot\nabla\psi=0$". Put numbers in dollars too, so that a minus is a minus:
"$-0.5$". In Python write labels as raw strings, r"$\iota$". The face is Computer Modern,
through matplotlib's mathtext (most of LaTeX's mathematics, none of its text commands). A
label that would need a glyph from another face is refused, and so are the characters that
the roman face sets as something else (" < > \ _ { | } ~ ^ outside dollars).

Lines join and end round, as a technical pen draws them. The module lays out dashes itself:
an open line starts and ends on a dash, a closed one holds a whole number of them, and a line
too short for its pattern is drawn solid. Hatching lies at 45 degrees on one grid for the
whole page and stipple on one field of dots, so regions with the same tone join with no seam.
Arrowheads are narrow, filled and sized by the weight of their line.

Drawing calls, on a Figure, a View or an Axes, in its own coordinates
---------------------------------------------------------------------
line(x, y, weight="medium", dash="solid", closed=False, arrow=None, halo=False)
    A NaN in x or y lifts the pen. arrow: "end", "start" or "both", or numbers between 0 and
    1 (arrow=[0.5]) for heads part way along each piece, to show a direction of travel.
    halo=True first cuts a channel of paper either side of the line through everything drawn
    before it, so that the line passes in front: draw what is behind first.
arrow(x0, y0, x1, y1, weight="medium", dash="solid", both=False)
    A straight arrow, its head at (x1, y1).
fill(x, y, tone="hatch", flip=False)
    Lays a tone inside a closed curve, or inside several, fill([xa, xb], [ya, yb]), by the
    even-odd rule, so a curve inside another makes a hole (a band, an annulus). It draws no
    edge: add line(..., closed=True) for one.
mark(x, y, kind="dot")
    A mark at each point.
text(x, y, s, anchor="c", size="normal", gap=None, baseline=False, knockout=True)
    anchor names the point of the label's ink box that sits at (x, y), and so where the
    label lies. gap moves it that far off the point (5 units, or 0 for "c").

        nw     n     ne        "sw": the label sits above and to the right of the point.
          +----+----+          "e": the label sits to the left, level with the point.
        w +    c    + e        baseline=True puts the baseline on y instead, for rows of words.
          +----+----+          knockout=True clears the paper just round the label, so that
        sw     s     se        lines and tones stop short of it.

    Returns the ink box (left, top, right, bottom) in page units.
leader(x, y, s, tx, ty, end=None, size="normal")
    A fine straight leader from (x, y) to the target (tx, ty), with the label s at (x, y) on
    the side away from the target. end: None (it touches a curve), "dot" (it points at a
    place inside an area) or "arrow".
page(x, y)
    The page coordinates of points of this view.

The figure
----------
Figure(height)                                      838 wide, page units
fig.view(box, x, y, equal=False, xlog=False, ylog=False, clip=False)
    equal=True keeps one scale for both axes, the ranges growing to fill the box.
    clip=True hides what falls outside the box.
fig.axes(box, x, y, xlog=False, ylog=False)         a plot, below
fig.inset(view, x, y, box)
    A magnified inset: outlines the region x=(x0, x1), y=(y0, y1) of view with a fine line,
    lays paper over box with a medium frame, joins the two by fine lines, and returns a view
    of the region, with equal scales and clipped, to draw the detail into again.
fig.save(name, previews=None)                       both SVG files, and PNGs if asked
fig.svg(page)                                       the SVG text for "light" or "dark"
style.measure(s, size="normal")                     (width, height) of a label, undrawn

Plots
-----
ax = fig.axes(box=(left, top, width, height), x=(x0, x1), y=(y0, y1), ylog=True)
ax.line(...), ax.mark(...) and the rest draw inside it, clipped to the box.
ax.frame("box")       "box": four sides, ticks inward on all of them (the default)
                      "open": left and bottom sides. "arrows": the same, ending in arrowheads
ax.xticks(major=None, minor=None, labels=None)   numbered ticks at major, short ones at minor
ax.yticks(major=None, minor=None, labels=None)   chosen for you when left out (decades on log)
ax.xlabel(s, at=None)   "centre" under the numbers (box) or "end" past the axis (otherwise)
ax.ylabel(s, at="top")  set level above the axis, as the old books do, or "centre" at the left
Ticks point inward, numbers are small, names normal. The calls may come in any order: frame,
ticks and names go on when the figure is saved, over the plot.

The order of drawing is the order of the calls, then frames and ticks, then all the
lettering over everything.

House rules, so that every figure is in one style
-------------------------------------------------
- No title, legend, caption, grid or units box inside a figure: the caption is the page's.
  Name a curve or a region with a label beside it or with a leader.
- One heavy thing. At most two tones besides paper. Fine lines for families of curves.
- Labels are a symbol or a few lower-case words. Put them on clear paper, and run leaders
  across lines nearly square and never across each other. A knockout is for the label that
  has to sit in a tone, not an excuse for one that sits on a line.
- Leave at least 10 units of paper inside the edge of the figure.
- Nothing finer than the page can show: lines closer than about 5 units merge at 324 px.
- Look at the four previews, and judge the dark one as a drawing in its own right.

What the files are
------------------
Exactly two colours each: ink #000000 on paper #ffffff, and ink #f0f6fc on paper #0d1117
(GitHub's dark page). The paper is a rectangle in the file, so one opened alone looks right.
Lettering is outlines, each glyph defined once. Only paths, circles, rectangles, <use> and
a rectangular clip, so Chrome, Safari, Firefox and rsvg-convert draw them alike (checked).
A figure with a few thousand points of curve comes to about 100 KB. The values behind the
words, different for each page, are in PAGES and the constants below it.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties, get_font
from matplotlib.path import Path as _MPath
from matplotlib.textpath import text_to_path

__all__ = ["Figure", "View", "Axes", "measure", "WIDTH", "WEIGHTS", "DASHES", "TONES", "MARKS",
           "SIZES", "ANCHORS", "PAGES"]

WIDTH = 838
HERE = Path(__file__).resolve().parent

WEIGHTS = ("fine", "medium", "heavy")
DASHES = ("solid", "dashed", "dotted", "dashdot", "dashdotdot")
TONES = ("stipple", "hatch", "crosshatch", "ink", "paper")
MARKS = ("dot", "circle", "cross")
SIZES = {"normal": 22.0, "small": 19.0}
ANCHORS = ("c", "n", "s", "e", "w", "ne", "nw", "se", "sw")

# What differs between the two pages. Light ink on a dark page spreads to the eye, so the dark
# page has its own, finer, values: thinner lines, thinner hatching, smaller dots, no emboldening.
PAGES = {
    "light": {
        "ink": "#000000", "paper": "#ffffff",
        "weight": {"fine": 1.0, "medium": 1.7, "heavy": 3.0},
        "hatch": {"pitch": 13 / math.sqrt(2), "width": 1.0},
        "crosshatch": {"pitch": 13 / math.sqrt(2), "width": 0.9},
        "stipple": {"spacing": 6.0, "dot": 1.9},
        "dot": 1.8,          # smallest dot of a dotted line
        "embolden": 0.3,     # ink added round every letter
    },
    "dark": {
        "ink": "#f0f6fc", "paper": "#0d1117",
        "weight": {"fine": 0.95, "medium": 1.5, "heavy": 2.6},
        "hatch": {"pitch": 13 / math.sqrt(2), "width": 0.8},
        "crosshatch": {"pitch": 13 / math.sqrt(2), "width": 0.75},
        "stipple": {"spacing": 6.5, "dot": 1.6},
        "dot": 1.6,
        "embolden": 0.0,
    },
}

# Dash patterns as seen on the page, in units, for a medium line: dash, gap, dash, gap and so on,
# a dash of 0 being a round dot. Scaled by weight, then stretched a little so that every line
# starts and ends on a dash and a closed curve has a whole number of them.
_DASH = {
    "dashed": (15.0, 7.0),
    "dotted": (0.0, 5.5),
    "dashdot": (24.0, 6.0, 0.0, 6.0),
    "dashdotdot": (24.0, 5.5, 0.0, 5.5, 0.0, 5.5),
}
_DASH_SCALE = {"fine": 0.85, "medium": 1.0, "heavy": 1.3}
_ARROW = {"fine": (10.0, 2.7), "medium": (12.5, 3.4), "heavy": (15.5, 4.3)}  # length, half width
_TICK = {"major": 8.0, "minor": 4.5}
_GAP = 5.0        # between a point and the label that names it
_KNOCK = 3.0      # paper kept clear round a label
_HALO = 3.5       # paper cut on either side of a haloed line
_MARK = {"dot": 3.3, "circle": 4.3, "cross": 5.4, "end": 2.0}   # radii
_NUMERAL = 0.6446  # height of a Computer Modern numeral, in ems

try:
    from matplotlib import _mathtext, _mathtext_data
    # matplotlib's Computer Modern table lacks a few symbols that the fonts themselves carry,
    # and would borrow them from STIX, another face: point it at the CM glyphs.
    for _name, _glyph in {"\\partial": ("cmmi10", 0x40), "\\star": ("cmmi10", 0x3F),
                          "\\perp": ("cmsy10", 0x3F), "\\parallel": ("cmsy10", 0x6B),
                          "\\dagger": ("cmsy10", 0x79)}.items():
        _mathtext_data.latex_to_bakoma.setdefault(_name, _glyph)
    # Sub- and superscripts at 0.8 of the size instead of TeX's 0.7, as TeX itself does for
    # small type: at 324 px wide an exponent at 0.7 does not read.
    _mathtext.SHRINK_FACTOR = 0.8
except (ImportError, AttributeError):
    pass

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
_ROMAN = _FONT_DIR / "cmr10.ttf"
_PROP = FontProperties(fname=str(_ROMAN), math_fontfamily="cm")
_EM = text_to_path.FONT_SCALE      # glyph outlines come at this many units to the em
_LIGATURES = (("ffi", "\xb1"), ("ffl", "\xb2"), ("ff", "\xae"), ("fi", "\xaf"), ("fl", "\xb0"))
_PLAIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?()[]-+=/*'`&%@#")


def _choose(value, choices, what):
    if value not in choices:
        raise ValueError(f"{what} must be one of {tuple(choices)}, not {value!r}")
    return value


# ------------------------------------------------------------------------------- writing numbers


def _num(v, nd):
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def _join(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i and not t.startswith("-"):
            out.append(" ")
        out.append(t)
    return "".join(out)


def _poly_d(p, closed=False, nd=2):
    q = np.round(np.asarray(p, float), nd)
    d = "M" + _join([_num(q[0, 0], nd), _num(q[0, 1], nd)])
    if len(q) > 1:
        d += "l" + _join([_num(v, nd) for v in np.diff(q, axis=0).ravel()])
    return d + ("z" if closed else "")


def _outline_d(verts, codes, nd=1):
    """SVG path data of a matplotlib outline (vertices, codes): relative, rounded."""
    q = np.round(np.asarray(verts, float), nd)
    out, cur, i, n = [], np.zeros(2), 0, len(codes)
    steps = {_MPath.MOVETO: ("m", 1), _MPath.LINETO: ("l", 1), _MPath.CURVE3: ("q", 2), _MPath.CURVE4: ("c", 3)}
    while i < n:
        c = codes[i]
        if c in steps:
            letter, k = steps[c]
            out.append(letter + _join([_num(v, nd) for v in (q[i:i + k] - cur).ravel()]))
            cur = q[i + k - 1]
            i += k
        else:
            if c == _MPath.CLOSEPOLY:
                out.append("z")
            i += 1
    d = "".join(out)
    return "M" + d[1:] if d.startswith("m") else d


# ------------------------------------------------------------------------------- geometry


def _pieces(X, Y):
    """Split a polyline at non-finite points into pieces of two or more points."""
    p = np.column_stack([np.ravel(X), np.ravel(Y)]).astype(float)
    good = np.isfinite(p).all(axis=1)
    out, start = [], None
    for i, g in enumerate(np.append(good, False)):
        if g and start is None:
            start = i
        elif not g and start is not None:
            out.append(p[start:i])
            start = None
    return [q for q in out if len(q) >= 2]


def _dedupe(p):
    keep = np.ones(len(p), bool)
    keep[1:] = np.hypot(*np.diff(p, axis=0).T) > 1e-9
    return p[keep]


def _simplify(p, tol=0.02):
    """Ramer, Douglas and Peucker: drop the points the eye cannot see, keep the ends."""
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, s = p[i], p[i + 1:j]
        ab = p[j] - a
        L2 = ab @ ab
        t = np.clip(((s - a) @ ab) / L2, 0, 1) if L2 > 0 else np.zeros(len(s))
        d = np.hypot(*(s - a - t[:, None] * ab).T)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return p[keep]


def _length(p, closed=False):
    q = np.vstack([p, p[:1]]) if closed else p
    return float(np.hypot(*np.diff(q, axis=0).T).sum())


def _at(p, s):
    """Point and unit tangent at arc length s along the polyline p."""
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = min(max(s, 0.0), cum[-1])
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    t = 0.0 if seg[k] == 0 else (s - cum[k]) / seg[k]
    d = p[k + 1] - p[k]
    nd = np.hypot(*d)
    return p[k] + t * d, (d / nd if nd else np.array([1.0, 0.0]))


def _trim(p, s0, s1):
    """The part of the polyline p between arc lengths s0 and s1."""
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    a, _ = _at(p, s0)
    b, _ = _at(p, s1)
    return np.vstack([a, p[(cum > s0) & (cum < s1)], b])


def _inside(pts, rings):
    """Even-odd rule: inside an odd number of the rings."""
    inside = np.zeros(len(pts), bool)
    for r in rings:
        inside ^= _MPath(np.vstack([r, r[:1]]), closed=True).contains_points(pts)
    return inside


def _hatch(rings, pitch, angle):
    """Segments, inside the rings by the even-odd rule, of the lines at `angle` degrees (as seen,
    counter-clockwise from the x axis), pitch apart. The lines lie on one grid for the whole page,
    so neighbouring regions with the same tone join without a seam."""
    a = math.radians(angle)
    dvec = np.array([math.cos(a), -math.sin(a)])     # along the lines (page y points down)
    nvec = np.array([-dvec[1], dvec[0]])             # across them
    ks, us = [], []
    for r in rings:
        p0, p1 = r, np.roll(r, -1, axis=0)
        s0, s1 = p0 @ nvec, p1 @ nvec
        u0, u1 = p0 @ dvec, p1 @ dvec
        k0 = np.ceil(np.minimum(s0, s1) / pitch).astype(np.int64)
        k1 = np.ceil(np.maximum(s0, s1) / pitch).astype(np.int64)
        cnt = np.maximum(k1 - k0, 0)
        if cnt.sum() == 0:
            continue
        edge = np.repeat(np.arange(len(r)), cnt)
        k = np.repeat(k0, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        t = (k * pitch - s0[edge]) / (s1[edge] - s0[edge])
        ks.append(k)
        us.append(u0[edge] + t * (u1[edge] - u0[edge]))
    if not ks:
        return np.zeros((0, 2, 2))
    k, u = np.concatenate(ks), np.concatenate(us)
    order = np.lexsort((u, k))
    k, u = k[order], u[order]
    if len(k) % 2 or not np.array_equal(k[0::2], k[1::2]):
        raise RuntimeError("hatching: a region's outline does not close")
    c, a0, a1 = k[0::2] * pitch, u[0::2], u[1::2]
    keep = a1 - a0 > 0.4
    c, a0, a1 = c[keep], a0[keep], a1[keep]
    return np.stack([c[:, None] * nvec + a0[:, None] * dvec, c[:, None] * nvec + a1[:, None] * dvec], axis=1)


_NOISE = {}


def _blue_noise(width, height, spacing, seed=7):
    """Points no closer than `spacing` that fill the page evenly and with no lattice for the eye
    to find (Poisson-disc sampling, thrown in phases on a grid so that it vectorises). The same
    field serves every region, so stippled regions that touch join without a seam."""
    key = (math.ceil(width), math.ceil(height), spacing, seed)
    if key not in _NOISE:
        r = spacing
        h = r / math.sqrt(2)
        nx, ny, P = int(math.ceil(key[0] / h)) + 1, int(math.ceil(key[1] / h)) + 1, 2
        gx = np.full((ny + 2 * P, nx + 2 * P), np.nan)
        gy = np.full_like(gx, np.nan)
        rng = np.random.default_rng(seed)
        near = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)]
        for _ in range(16):
            for pj in range(3):
                for pi in range(3):
                    jj, ii = np.meshgrid(np.arange(pj, ny, 3), np.arange(pi, nx, 3), indexing="ij")
                    jj, ii = jj.ravel(), ii.ravel()
                    empty = np.isnan(gx[jj + P, ii + P])
                    jj, ii = jj[empty], ii[empty]
                    x = (ii + rng.random(len(ii))) * h
                    y = (jj + rng.random(len(jj))) * h
                    ok = np.ones(len(ii), bool)
                    for di, dj in near:
                        qx, qy = gx[jj + P + dj, ii + P + di], gy[jj + P + dj, ii + P + di]
                        with np.errstate(invalid="ignore"):
                            ok &= ~((qx - x) ** 2 + (qy - y) ** 2 < r * r)
                    gx[jj[ok] + P, ii[ok] + P] = x[ok]
                    gy[jj[ok] + P, ii[ok] + P] = y[ok]
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        _NOISE[key] = pts[np.isfinite(pts[:, 0])]
    return _NOISE[key]


def _crosses(a, b, corners):
    """Whether the segment ab passes through the inside of the rectangle with these corners."""
    lo, hi = corners.min(axis=0) + 0.5, corners.max(axis=0) - 0.5
    t = np.linspace(0.02, 0.98, 97)[:, None]
    p = a + t * (b - a)
    return bool(((lo < p) & (p < hi)).all(axis=1).any())


# ------------------------------------------------------------------------------- lettering

_LAYOUTS = {}


def _split_math(s):
    """The plain and the mathematical parts of s, as (is_math, text) pairs."""
    parts, buf, in_math, i = [], "", False, 0
    while i < len(s):
        if s.startswith("\\$", i):
            buf += "\\$"
            i += 2
            continue
        if s[i] == "$":
            parts.append((in_math, buf))
            buf, in_math = "", not in_math
        else:
            buf += s[i]
        i += 1
    if in_math:
        raise ValueError(f"label {s!r}: a $ is not closed")
    parts.append((False, buf))
    return [p for p in parts if p[1]]


def _layout(s):
    """The glyphs of a label: ([(glyph, x, y, scale)], {glyph: (verts, codes)}, [rule outlines],
    ink box (x0, y0, x1, y1)), in units of 1/100 em with y up and the baseline at y = 0."""
    if s in _LAYOUTS:
        return _LAYOUTS[s]
    parts = _split_math(s)
    text = ""
    for is_math, t in parts:
        if is_math:
            text += "$" + t + "$"
            continue
        bad = sorted(set(t) - _PLAIN)
        if bad:
            raise ValueError(f"label {s!r}: {bad} cannot be set as words in this face, put it in $...$")
        for a, b in _LIGATURES:
            t = t.replace(a, b)
        text += t
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if any(m for m, _ in parts):
            info, gmap, rects = text_to_path.get_glyphs_mathtext(_PROP, text)
        else:
            font = get_font(str(_ROMAN))
            font.set_size(_EM, text_to_path.DPI)
            info, gmap, rects = text_to_path.get_glyphs_with_font(font, text)
    if caught:
        raise ValueError(f"label {s!r}: " + " ".join(str(w.message) for w in caught))
    for key, *_ in info:
        if not key.lower().startswith("cm"):
            raise ValueError(f"label {s!r} needs {key.split('-')[0]}, which is not Computer Modern: "
                             "rephrase it with symbols the CM fonts have")
    verts = [np.asarray(gmap[key][0]) * sc + [x, y] for key, x, y, sc in info]
    codes = [np.asarray(gmap[key][1]) for key, *_ in info]
    verts += [np.asarray(v, float) for v, _ in rects]
    codes += [np.asarray(c) for _, c in rects]
    if verts:
        e = _MPath(np.concatenate(verts), np.concatenate(codes)).get_extents()
        box = (e.x0, e.y0, e.x1, e.y1)
    else:
        box = (0.0, 0.0, 0.0, 0.0)
    _LAYOUTS[s] = (info, {key: gmap[key] for key, *_ in info}, rects, box)
    return _LAYOUTS[s]


def measure(s, size="normal"):
    """Width and height, in page units, of the ink of the label s."""
    _choose(size, SIZES, "size")
    x0, y0, x1, y1 = _layout(s)[3]
    k = SIZES[size] / _EM
    return (x1 - x0) * k, (y1 - y0) * k


# ------------------------------------------------------------------------------- drawing


class _Drawing:
    """The drawing calls that a figure and its views share, in the coordinates of each."""

    _fig: "Figure"
    _clip = None

    def _map(self, x, y):
        raise NotImplementedError

    def _xy(self, x, y):
        X, Y = self._map(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(X, float), np.asarray(Y, float)

    def page(self, x, y):
        """Page coordinates of the points (x, y) of this view."""
        return self._xy(x, y)

    def line(self, x, y, weight="medium", dash="solid", closed=False, arrow=None, halo=False):
        """A line through the points (x, y). A NaN in x or y lifts the pen.

        weight: "fine", "medium" or "heavy". dash: "solid", "dashed", "dotted", "dashdot" or
        "dashdotdot". closed: join the last point to the first. arrow: None, "end", "start" or
        "both" for heads at the ends, or a number or list of numbers between 0 and 1 for heads
        part way along that show the direction of travel. halo: cut a channel of paper either
        side of the line through whatever is already drawn, so that it reads as in front.
        """
        _choose(weight, WEIGHTS, "weight")
        _choose(dash, DASHES, "dash")
        if isinstance(arrow, str):
            _choose(arrow, ("end", "start", "both"), "arrow")
        hl = _ARROW[weight][0]
        pieces, heads = [], []
        for p in _pieces(*self._xy(x, y)):
            p = _simplify(_dedupe(p))
            if len(p) < 2:
                continue
            L = _length(p)
            if isinstance(arrow, str) and not closed and L > hl:
                ends = ("start", "end") if arrow == "both" else (arrow,)
                for end in ends:
                    tip, _ = _at(p, L if end == "end" else 0.0)
                    back, _ = _at(p, L - 0.8 * hl if end == "end" else 0.8 * hl)
                    heads.append((tip, tip - back))
                p = _trim(p, 0.5 * hl if "start" in ends else 0.0, L - 0.5 * hl if "end" in ends else L)
            elif arrow is not None and not isinstance(arrow, str):
                for f in np.atleast_1d(arrow):
                    c, t = _at(p, float(f) * L)
                    heads.append((c + 0.5 * hl * t, t))
            pieces.append(p)
        if pieces:
            self._fig._ops.append({"kind": "line", "pieces": pieces, "closed": bool(closed), "weight": weight,
                                   "dash": dash, "halo": bool(halo), "clip": self._clip})
        for tip, direction in heads:
            self._fig._ops.append({"kind": "head", "tip": tip, "dir": direction, "weight": weight, "clip": self._clip})

    def arrow(self, x0, y0, x1, y1, weight="medium", dash="solid", both=False):
        """A straight arrow from (x0, y0) to (x1, y1), with its head at the second point."""
        self.line([x0, x1], [y0, y1], weight=weight, dash=dash, arrow="both" if both else "end")

    def fill(self, x, y, tone="hatch", flip=False):
        """Lay a tone over the region inside the closed curve (x, y), or inside several,
        fill([x0, x1], [y0, y1]), by the even-odd rule, so that a curve inside another makes a
        hole. Nothing is outlined: draw the edge with line(..., closed=True) if it is to be seen.

        tone: "stipple" (light), "hatch" (middle), "crosshatch" (dark), "ink" (solid) or
        "paper" (blank, and it hides what is under it). flip: hatch the other way, for a region
        that meets another hatched one.
        """
        _choose(tone, TONES, "tone")
        if (isinstance(x, (list, tuple)) and len(x) and np.ndim(x[0]) >= 1) or np.ndim(x) == 2:
            pairs = list(zip(x, y))
        else:
            pairs = [(x, y)]
        rings = []
        for a, b in pairs:
            X, Y = self._xy(a, b)
            r = np.column_stack([X.ravel(), Y.ravel()])
            r = _dedupe(r[np.isfinite(r).all(axis=1)])
            if len(r) > 1 and np.allclose(r[0], r[-1]):
                r = r[:-1]
            if len(r) >= 3:
                rings.append(r)
        if rings:
            self._fig._ops.append({"kind": "fill", "rings": rings, "tone": tone, "flip": bool(flip), "clip": self._clip})

    def mark(self, x, y, kind="dot"):
        """A mark at each of the points (x, y): "dot" (solid: a point that matters, such as an
        axis or an O point), "circle" (open: data) or "cross" (a point that is not there, or an
        X point drawn on its own)."""
        _choose(kind, MARKS, "mark")
        X, Y = self._xy(np.atleast_1d(x), np.atleast_1d(y))
        pts = np.column_stack([X.ravel(), Y.ravel()])
        self._fig._ops.append({"kind": "mark", "pts": pts[np.isfinite(pts).all(axis=1)], "mark": kind, "clip": self._clip})

    def text(self, x, y, s, anchor="c", size="normal", gap=None, baseline=False, knockout=True):
        """Letter s at (x, y). Words are set upright and whatever is between dollars is
        mathematics, with italic symbols: "magnetic axis", "$\\iota$", "surface $\\psi=0.5$".

        anchor: which point of the label's ink sits at (x, y): "c", "n", "s", "e", "w", "ne",
        "nw", "se" or "sw" ("sw" puts the label above and to the right of the point). gap: how
        far the label stands off from the point, in page units (5 unless the anchor is "c").
        baseline: put the baseline at y instead, for words in a row. knockout: clear the paper
        just round the label, so that lines and tones stop short of it. Returns the ink box
        (left, top, right, bottom) in page units.
        """
        X, Y = self._xy(x, y)
        return self._fig._letter(float(X), float(Y), s, anchor, size, gap, baseline, knockout)

    def leader(self, x, y, s, tx, ty, end=None, size="normal"):
        """The label s with a fine straight leader from (x, y) to the target (tx, ty). The label
        stands at (x, y) on the side away from the target.

        end: None (the leader touches the target: a curve), "dot" (a small dot on the target: a
        place inside an area) or "arrow".
        """
        _choose(end, (None, "dot", "arrow"), "end")
        X, Y = (float(v) for v in self._xy(x, y))
        TX, TY = (float(v) for v in self._xy(tx, ty))
        ang = math.degrees(math.atan2(-(TY - Y), TX - X)) % 360
        anchor = ["e", "ne", "n", "nw", "w", "sw", "s", "se"][int(((ang + 22.5) % 360) // 45)]
        box = self._fig._letter(X, Y, s, anchor, size, _GAP, False, True)
        start, target = np.array([X, Y]), np.array([TX, TY])
        v = target - start
        L = float(np.hypot(*v))
        if L < 1:
            return box
        u = v / L
        stop = target - 0.5 * _ARROW["fine"][0] * u if end == "arrow" else target
        self._fig._ops.append({"kind": "line", "pieces": [np.array([start, stop])], "closed": False,
                               "weight": "fine", "dash": "solid", "halo": False, "clip": None})
        if end == "arrow":
            self._fig._ops.append({"kind": "head", "tip": target, "dir": u, "weight": "fine", "clip": None})
        elif end == "dot":
            self._fig._ops.append({"kind": "mark", "pts": target[None, :], "mark": "end", "clip": None})
        return box


class View(_Drawing):
    """Data coordinates mapped onto a box of the page, x to the right and y up."""

    def __init__(self, fig, box, x, y, equal=False, xlog=False, ylog=False, clip=False):
        self._fig = fig
        self.box = tuple(float(v) for v in box)
        left, top, width, height = self.box
        self.xlog, self.ylog = bool(xlog), bool(ylog)
        x0, x1 = (math.log10(v) for v in x) if xlog else (float(v) for v in x)
        y0, y1 = (math.log10(v) for v in y) if ylog else (float(v) for v in y)
        if equal:
            s = min(width / (x1 - x0), height / (y1 - y0))
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            x0, x1 = cx - 0.5 * width / s, cx + 0.5 * width / s
            y0, y1 = cy - 0.5 * height / s, cy + 0.5 * height / s
        self.lim = (x0, x1, y0, y1)
        self._clip = fig._clip_id(self.box) if clip else None

    @property
    def scale(self):
        """Page units per data unit along x and along y (equal views have the two the same)."""
        x0, x1, y0, y1 = self.lim
        return self.box[2] / (x1 - x0), self.box[3] / (y1 - y0)

    def _map(self, x, y):
        x0, x1, y0, y1 = self.lim
        left, top, width, height = self.box
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.xlog:
                x = np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan)
            if self.ylog:
                y = np.where(y > 0, np.log10(np.where(y > 0, y, 1.0)), np.nan)
        return left + (x - x0) / (x1 - x0) * width, top + (y1 - y) / (y1 - y0) * height


class Axes(View):
    """A View for a plot: frame(), xticks(), yticks(), xlabel() and ylabel(), in any order. What
    is drawn into it is clipped to its box. The frame, ticks and names are put on when the figure
    is saved, over the plot."""

    def __init__(self, fig, box, x, y, xlog=False, ylog=False):
        super().__init__(fig, box, x, y, xlog=xlog, ylog=ylog, clip=True)
        self._x, self._y = tuple(float(v) for v in x), tuple(float(v) for v in y)
        self._frame, self._ticks, self._names, self._done = "box", {}, {}, False
        fig._axes.append(self)

    def frame(self, kind="box"):
        """The frame: "box" (four sides, ticks inward on all of them, the default), "open" (the
        left and bottom sides) or "arrows" (left and bottom ending in arrowheads, for a sketch)."""
        self._frame = _choose(kind, ("box", "open", "arrows"), "frame")

    def xticks(self, major=None, minor=None, labels=None):
        """Ticks along x: longer ones at the values `major`, numbered, shorter ones at `minor`.
        Left out, they are chosen: round numbers, or decades on a log axis. labels: strings to
        print instead of the numbers, such as "$\\pi$", or False for none."""
        self._ticks["x"] = (major, minor, labels)

    def yticks(self, major=None, minor=None, labels=None):
        """Ticks along y, as xticks."""
        self._ticks["y"] = (major, minor, labels)

    def xlabel(self, s, at=None):
        """Name the x axis: "centre" (under the numbers, the default for a box) or "end" (past
        the end of the axis, the default for an open frame or arrows)."""
        self._names["x"] = (s, at)

    def ylabel(self, s, at="top"):
        """Name the y axis, set level as the old books do: "top" (above the axis, the default)
        or "centre" (left of the numbers, half way up)."""
        self._names["y"] = (s, at)

    def _line(self, pieces, closed=False, butt=False):
        self._fig._late.append({"kind": "line", "pieces": [np.asarray(p, float) for p in pieces], "closed": closed,
                                "weight": "medium", "dash": "solid", "halo": False, "clip": None, "butt": butt,
                                "mitre": True})

    def _finish(self):
        L, T, w, h = self.box
        R, B = L + w, T + h
        over = 16.0 if self._frame == "arrows" else 0.0
        if self._frame == "box":
            self._line([[[L, T], [R, T], [R, B], [L, B]]], closed=True)
        elif self._frame == "open":
            self._line([[[L, T], [L, B], [R, B]]])
        else:
            hl = _ARROW["medium"][0]
            self._line([[[L, T - over + 0.5 * hl], [L, B], [R + over - 0.5 * hl, B]]])
            for tip, u in (([L, T - over], [0.0, -1.0]), ([R + over, B], [1.0, 0.0])):
                self._fig._late.append({"kind": "head", "tip": np.array(tip), "dir": np.array(u),
                                        "weight": "medium", "clip": None})
        numeral = _NUMERAL * SIZES["small"]
        extent = {"x": R, "y": L}          # how far the numbers reach: right of x, left of y
        for axis, (major, minor, labels) in self._ticks.items():
            log = self.xlog if axis == "x" else self.ylog
            lim = self._x if axis == "x" else self._y
            if major is None:
                major, auto = _auto_ticks(lim, log)
                minor = auto if minor is None else minor
            major, minor = list(major), list(minor if minor is not None else [])
            other = (self._y if axis == "x" else self._x)[0]

            def place(v):
                X, Y = self._xy(v, other) if axis == "x" else self._xy(other, v)
                q = float(X if axis == "x" else Y)
                lo, hi = (L, R) if axis == "x" else (T, B)
                return q if lo - 0.01 <= q <= hi + 0.01 else None

            segs = []
            for values, length in ((major, _TICK["major"]), (minor, _TICK["minor"])):
                for v in values:
                    q = place(v)
                    if q is None:
                        continue
                    if axis == "x":
                        segs.append([[q, B], [q, B - length]])
                        if self._frame == "box":
                            segs.append([[q, T], [q, T + length]])
                    else:
                        segs.append([[L, q], [L + length, q]])
                        if self._frame == "box":
                            segs.append([[R, q], [R - length, q]])
            if segs:
                self._line(segs, butt=True)
            if labels is False:
                continue
            if labels is None or labels is True:
                labels = [_tick_label(v, log) for v in major]
            for v, text in zip(major, labels):
                q = place(v)
                if q is None or not text:
                    continue
                if axis == "x":
                    box = self._fig._letter(q, B + 7.0 + numeral, text, "c", "small", 0.0, True, False)
                    extent["x"] = max(extent["x"], box[2])
                else:
                    box = self._fig._letter(L - 7.0, q + 0.5 * numeral, text, "e", "small", 0.0, True, False)
                    extent["y"] = min(extent["y"], box[0])
        if "x" in self._names:
            text, at = self._names["x"]
            at = at or ("centre" if self._frame == "box" else "end")
            if _choose(at, ("centre", "end"), "at") == "centre":
                self._fig._letter(L + 0.5 * w, B + 7.0 + numeral + 9.0, text, "n", "normal", 0.0, False, False)
            else:
                self._fig._letter(max(R + over + 7.0, extent["x"] + 6.0), B, text, "w", "normal", 0.0, False, False)
        if "y" in self._names:
            text, at = self._names["y"]
            if _choose(at, ("top", "centre"), "at") == "top":
                self._fig._letter(L, T - over - 9.0, text, "s", "normal", 0.0, False, False)
            else:
                self._fig._letter(extent["y"] - 12.0, T + 0.5 * h, text, "e", "normal", 0.0, False, False)


def _auto_ticks(lim, log):
    lo, hi = min(lim), max(lim)
    if log:
        a, b = math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9)
        step = max(1, math.ceil((b - a + 1) / 6))
        major = [10.0 ** k for k in range(b, a - 1, -step)][::-1]
        if step == 1:
            minor = [m * 10.0 ** k for k in range(a - 1, b + 1) for m in range(2, 10) if lo <= m * 10.0 ** k <= hi]
        else:
            minor = [10.0 ** k for k in range(a, b + 1) if 10.0 ** k not in major]
        return major, minor
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    return [first + i * step for i in range(int(math.floor((hi - first) / step + 1e-9)) + 1)], []


def _tick_label(v, log):
    if log:
        k = int(round(math.log10(v)))
        return "$1$" if k == 0 else "$10$" if k == 1 else f"$10^{{{k}}}$"
    if abs(v - round(v)) < 1e-9:
        return f"${int(round(v))}$"
    return f"${v:g}$"


# ------------------------------------------------------------------------------- the figure


class Figure(_Drawing):
    """A figure 838 units wide and `height` units high, drawn once and written for both pages.
    Its own drawing calls take page units: x from the left edge, y down from the top edge."""

    def __init__(self, height):
        self.width, self.height = float(WIDTH), float(height)
        self._fig = self
        self._ops = []      # the drawing, in the order it was made
        self._late = []     # frames and ticks, over the drawing
        self._letters = []  # the lettering, over everything
        self._clips = {}
        self._axes = []     # plots, whose frames and ticks go on at the end

    def _map(self, x, y):
        return x, y

    def view(self, box, x, y, equal=False, xlog=False, ylog=False, clip=False):
        """A View: data coordinates x = (x0, x1), y = (y0, y1) mapped onto box = (left, top,
        width, height), in page units. equal: one scale for both axes (the ranges grow to fill
        the box). clip: hide what falls outside the box."""
        return View(self, box, x, y, equal=equal, xlog=xlog, ylog=ylog, clip=clip)

    def axes(self, box, x, y, xlog=False, ylog=False):
        """An Axes for a plot: a View with a frame, ticks and axis labels."""
        return Axes(self, box, x, y, xlog=xlog, ylog=ylog)

    def inset(self, view, x, y, box, connect=True):
        """A magnified inset. The region x = (x0, x1), y = (y0, y1) of `view` is outlined there
        with a fine line and shown again, enlarged, in box = (left, top, width, height): on
        paper, framed, and joined to its source by two fine lines. Returns the inset's View
        (equal scales, clipped to the box): draw into it as into any other view."""
        X, Y = view.page([x[0], x[1], x[1], x[0]], [y[1], y[1], y[0], y[0]])
        src = np.column_stack([X, Y])
        left, top, width, height = (float(v) for v in box)
        dst = np.array([[left, top], [left + width, top], [left + width, top + height], [left, top + height]])
        self._ops.append({"kind": "line", "pieces": [src], "closed": True, "weight": "fine", "dash": "solid",
                          "halo": False, "clip": None})
        if connect:
            free = [i for i in range(4) if not (_crosses(src[i], dst[i], src) or _crosses(src[i], dst[i], dst))]
            if len(free) > 2:
                free = list(max(((a, b) for a in free for b in free if a < b),
                                key=lambda ab: np.hypot(*(src[ab[0]] + dst[ab[0]] - src[ab[1]] - dst[ab[1]]))))
            for i in free:
                self._ops.append({"kind": "line", "pieces": [np.array([src[i], dst[i]])], "closed": False,
                                  "weight": "fine", "dash": "solid", "halo": False, "clip": None})
        self._ops.append({"kind": "fill", "rings": [dst], "tone": "paper", "flip": False, "clip": None})
        self._late.append({"kind": "line", "pieces": [dst], "closed": True, "weight": "medium", "dash": "solid",
                           "halo": False, "clip": None, "mitre": True})
        return View(self, box, x, y, equal=True, clip=True)

    def _clip_id(self, box):
        key = tuple(round(v, 3) for v in box)
        return self._clips.setdefault(key, f"c{len(self._clips)}")

    def _letter(self, X, Y, s, anchor, size, gap, baseline, knockout):
        _choose(anchor, ANCHORS, "anchor")
        _choose(size, SIZES, "size")
        info, gmap, rects, (bx0, by0, bx1, by1) = _layout(s)
        k = SIZES[size] / _EM
        if gap is None:
            gap = 0.0 if anchor == "c" else _GAP
        l, r, t, b = bx0 * k, bx1 * k, -by1 * k, -by0 * k      # ink box about the origin, y down
        hx = l if "w" in anchor else r if "e" in anchor else 0.5 * (l + r)
        vy = t if anchor[0] == "n" else b if anchor[0] == "s" else 0.5 * (t + b)
        dx = gap if "w" in anchor else -gap if "e" in anchor else 0.0
        dy = gap if anchor[0] == "n" else -gap if anchor[0] == "s" else 0.0
        ox = X + dx - hx
        oy = Y if baseline else Y + dy - vy
        box = (ox + l, oy + t, ox + r, oy + b)
        self._letters.append({"info": info, "gmap": gmap, "rects": rects, "o": (ox, oy), "k": k,
                              "box": box, "knockout": bool(knockout)})
        return box

    # --------------------------------------------------------------- writing
    def save(self, name, previews=None):
        """Write <name>-light.svg and <name>-dark.svg, beside this module when name is a bare
        name, and with previews=<directory> also PNG renders of each at 838 and 324 px wide
        there (by rsvg-convert). Refuses to write a file that is not ink and paper alone."""
        stem = Path(name)
        if stem.suffix == ".svg":
            stem = stem.with_suffix("")
        if len(stem.parts) == 1:
            stem = HERE / stem
        paths = []
        for page in ("light", "dark"):
            svg = self.svg(page)
            _verify(svg, PAGES[page])
            path = stem.with_name(f"{stem.name}-{page}.svg")
            path.write_text(svg)
            paths.append(path)
        report = ", ".join(f"{p.name} {p.stat().st_size / 1024:.0f} KB" for p in paths)
        if previews is not None:
            tool = shutil.which("rsvg-convert")
            if tool is None:
                raise RuntimeError("previews need rsvg-convert (brew install librsvg)")
            out = Path(previews)
            out.mkdir(parents=True, exist_ok=True)
            for p in paths:
                for w in (838, 324):
                    subprocess.run([tool, "-w", str(w), "-o", str(out / f"{p.stem}-{w}.png"), str(p)], check=True)
            report += f", previews in {out}"
        print(report)
        return paths

    def svg(self, page):
        """The figure as SVG text, for the page "light" or "dark"."""
        P = PAGES[_choose(page, PAGES, "page")]
        ink, paper = P["ink"], P["paper"]
        for ax in self._axes:
            if not ax._done:
                ax._finish()
                ax._done = True
        defs, body = [], []
        for (l, t, w, h), cid in self._clips.items():
            defs.append(f'<clipPath id="{cid}"><rect x="{_num(l, 2)}" y="{_num(t, 2)}" '
                        f'width="{_num(w, 2)}" height="{_num(h, 2)}"/></clipPath>')
        gid = {}
        for L in self._letters:
            for key, *_ in L["info"]:
                if key not in gid:
                    gid[key] = f"g{len(gid)}"
                    defs.append(f'<path id="{gid[key]}" d="{_outline_d(*L["gmap"][key])}"/>')
        for op in self._ops + self._late:
            items = self._render(op, P)
            if items and op.get("clip"):
                items = [f'<g clip-path="url(#{op["clip"]})">' + "".join(items) + "</g>"]
            body.extend(items)
        # lettering last: every knockout first, so that none cuts into another label
        knock, letters = [], []
        emb = P["embolden"]
        for L in self._letters:
            if L["knockout"]:
                x0, y0, x1, y1 = L["box"]
                m = _KNOCK
                knock.append(f'<rect x="{_num(x0 - m, 2)}" y="{_num(y0 - m, 2)}" width="{_num(x1 - x0 + 2 * m, 2)}" '
                             f'height="{_num(y1 - y0 + 2 * m, 2)}" fill="{paper}" stroke="none"/>')
            k, (ox, oy) = L["k"], L["o"]
            uses = []
            for key, x, y, sc in L["info"]:
                if sc == 1:
                    uses.append(f'<use href="#{gid[key]}" x="{_num(x, 1)}" y="{_num(y, 1)}"/>')
                else:
                    uses.append(f'<use href="#{gid[key]}" transform="matrix({_num(sc, 4)} 0 0 {_num(sc, 4)} '
                                f'{_num(x, 1)} {_num(y, 1)})"/>')
            uses += [f'<path d="{_outline_d(v, c)}"/>' for v, c in L["rects"]]
            stroke = f' stroke-width="{_num(emb / k, 2)}"' if emb > 0 else ' stroke="none"'
            letters.append(f'<g transform="matrix({_num(k, 5)} 0 0 {_num(-k, 5)} {_num(ox, 2)} {_num(oy, 2)})" '
                           f'fill="{ink}"{stroke}>' + "".join(uses) + "</g>")
        body += knock + letters
        H = _num(self.height, 2)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{H}" viewBox="0 0 {WIDTH} {H}">\n'
                + (f"<defs>{''.join(defs)}</defs>\n" if defs else "")
                + f'<rect width="{WIDTH}" height="{H}" fill="{paper}"/>\n'
                + f'<g fill="none" stroke="{ink}" stroke-linecap="round" stroke-linejoin="round">\n'
                + "\n".join(body) + "\n</g>\n</svg>\n")

    def _render(self, op, P):
        ink, paper = P["ink"], P["paper"]
        kind, out = op["kind"], []
        if kind == "line":
            w = P["weight"][op["weight"]]
            closed = op["closed"]
            if op["halo"]:
                d = "".join(_poly_d(p, closed) for p in op["pieces"])
                out.append(f'<path d="{d}" stroke="{paper}" stroke-width="{_num(w + 2 * _HALO, 2)}"/>')
            if op.get("butt"):
                d = "".join(_poly_d(p) for p in op["pieces"])
                out.append(f'<path d="{d}" stroke-width="{_num(w, 2)}" stroke-linecap="butt"/>')
            elif op["dash"] == "solid":
                d = "".join(_poly_d(p, closed) for p in op["pieces"])
                join = ' stroke-linejoin="miter" stroke-linecap="square"' if op.get("mitre") else ""
                out.append(f'<path d="{d}" stroke-width="{_num(w, 2)}"{join}/>')
            else:
                strokes, dots, dot = [], [], max(w, P["dot"])
                for p in op["pieces"]:
                    dashes, centres = _dash_geometry(p, closed, op["dash"], op["weight"], w, dot)
                    strokes += dashes
                    dots += centres
                if strokes:
                    d = "".join(_poly_d(q) for q in strokes)
                    out.append(f'<path d="{d}" stroke-width="{_num(w, 2)}"/>')
                if dots:
                    out.append(f'<path d="{_dots_d(np.array(dots))}" stroke-width="{_num(dot, 2)}"/>')
        elif kind == "head":
            hl, hw = _ARROW[op["weight"]]
            u = np.asarray(op["dir"], float)
            u = u / np.hypot(*u)
            n = np.array([-u[1], u[0]])
            tip = np.asarray(op["tip"], float)
            pts = np.array([tip, tip - hl * u + hw * n, tip - 0.78 * hl * u, tip - hl * u - hw * n])
            out.append(f'<path d="{_poly_d(pts, True)}" fill="{ink}" stroke="none"/>')
        elif kind == "mark":
            r, mk, w = _MARK[op["mark"]], op["mark"], P["weight"]["medium"]
            for x, y in op["pts"]:
                cx, cy = _num(x, 2), _num(y, 2)
                if mk in ("dot", "end"):
                    out.append(f'<circle cx="{cx}" cy="{cy}" r="{_num(r, 2)}" fill="{ink}" stroke="none"/>')
                elif mk == "circle":
                    out.append(f'<circle cx="{cx}" cy="{cy}" r="{_num(r, 2)}" fill="{paper}" stroke-width="{_num(w, 2)}"/>')
                else:
                    a = r / math.sqrt(2)
                    d = (f"M{_num(x - a, 2)} {_num(y - a, 2)}l{_num(2 * a, 2)} {_num(2 * a, 2)}"
                         f"M{_num(x - a, 2)} {_num(y + a, 2)}l{_num(2 * a, 2)} {_num(-2 * a, 2)}")
                    out.append(f'<path d="{d}" stroke-width="{_num(w, 2)}"/>')
        elif kind == "fill":
            tone, rings = op["tone"], op["rings"]
            if tone in ("ink", "paper"):
                d = "".join(_poly_d(r, True) for r in rings)
                out.append(f'<path d="{d}" fill="{ink if tone == "ink" else paper}" fill-rule="evenodd" stroke="none"/>')
            elif tone in ("hatch", "crosshatch"):
                spec = P[tone]
                angles = [-45.0 if op["flip"] else 45.0] if tone == "hatch" else [45.0, -45.0]
                segs = [s for a in angles for s in _hatch(rings, spec["pitch"], a)]
                if segs:
                    q = np.round(np.array(segs), 1)
                    d = "".join("M" + _join([_num(s[0, 0], 1), _num(s[0, 1], 1)]) + "l"
                                + _join([_num(s[1, 0] - s[0, 0], 1), _num(s[1, 1] - s[0, 1], 1)]) for s in q)
                    out.append(f'<path d="{d}" stroke-width="{_num(spec["width"], 2)}" stroke-linecap="butt"/>')
            else:
                spec = P["stipple"]
                lo = np.min([r.min(axis=0) for r in rings], axis=0)
                hi = np.max([r.max(axis=0) for r in rings], axis=0)
                pts = _blue_noise(max(self.width, hi[0]) + 10, max(self.height, hi[1]) + 10, spec["spacing"])
                pts = pts[((pts >= lo) & (pts <= hi)).all(axis=1)]
                pts = pts[_inside(pts, rings)]
                if len(pts):
                    out.append(f'<path d="{_dots_d(pts[np.lexsort((pts[:, 0], pts[:, 1]))])}" '
                               f'stroke-width="{_num(spec["dot"], 2)}"/>')
        return out


def _dots_d(pts):
    """Round dots at the points: each a stroke 0.1 long with round caps. Every renderer draws
    these alike, where a stroke of length 0 comes out heavier in WebKit."""
    q = np.round(np.asarray(pts, float), 1)
    d = ["M" + _join([_num(q[0, 0], 1), _num(q[0, 1], 1)]) + "h.1"]
    step = np.diff(q, axis=0) - [0.1, 0.0]
    d += ["m" + _join([_num(a, 1), _num(b, 1)]) + "h.1" for a, b in step]
    return "".join(d)


def _dash_geometry(p, closed, dash, weight, w, dot):
    """The dashes (as polylines) and dots (as centres) of a patterned line. The pattern is laid
    out along the line and stretched a little so that an open line begins and ends on its first
    element and a closed one holds a whole number of periods. A line too short for its pattern
    is drawn whole."""
    if closed:
        p = np.vstack([p, p[:1]])
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    L = cum[-1]
    # elements along the centre line: a dash of visible length d runs d - w between its caps,
    # a dot is a point, and a gap g between visible ends is g plus the two half caps beside it
    pat = np.array(_DASH[dash]) * _DASH_SCALE[weight]
    elems = [("dot", 0.0) if v == 0 else ("dash", max(v - w, 0.5)) for v in pat[0::2]]
    caps = [0.5 * (dot if kind == "dot" else w) for kind, _ in elems]
    gaps = [pat[2 * i + 1] + caps[i] + caps[(i + 1) % len(elems)] for i in range(len(elems))]
    T = sum(length for _, length in elems) + sum(gaps)
    first = elems[0][1]
    if closed:
        n = max(2, int(round(L / T)))
        scale, tail = L / (n * T), False
    elif L >= T + first or (elems[0][0] == "dot" and L >= T):
        n = max(1, int(round((L - first) / T)))
        scale, tail = L / (n * T + first), True
    else:
        return [p], []

    def piece(a, b):
        inner = p[(cum > a) & (cum < b)]
        return np.vstack([_point(p, cum, a), inner, _point(p, cum, b)])

    dashes, dots, s = [], [], 0.0
    for k in range(n + (1 if tail else 0)):
        for i, (kind, length) in enumerate(elems):
            if tail and k == n and i > 0:
                break
            if kind == "dot":
                dots.append(_point(p, cum, s))
            else:
                dashes.append(piece(s, min(s + length * scale, L)))
            s += (length + gaps[i]) * scale
    return dashes, dots


def _point(p, cum, s):
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(p) - 2))
    seg = cum[k + 1] - cum[k]
    t = 0.0 if seg == 0 else (s - cum[k]) / seg
    return p[k] + t * (p[k + 1] - p[k])


def _verify(svg, page):
    """Refuse a file that is anything but ink and paper."""
    colours = {c.lower() for c in re.findall(r"#[0-9a-fA-F]{3,8}\b", svg)}
    extra = colours - {page["ink"], page["paper"]}
    if extra:
        raise RuntimeError(f"colours other than ink and paper: {sorted(extra)}")
    for word in ("opacity", "gradient", "pattern", "filter", "mask", "rgb(", "<image", "<text", "font"):
        if word in svg:
            raise RuntimeError(f"not allowed in a figure of two colours: {word}")
