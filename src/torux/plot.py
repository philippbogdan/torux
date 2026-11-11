"""The one picture of an answer: the surface inside its coils, coloured by the strength of the field on it, beside
three of its cross-sections.

    s = torux.solve("examples/quasr_0000954.json")
    s.plot()                     # the figure: a window, or the cell's picture in a notebook
    s.plot("surface.png")        # a file: .png, .pdf or .svg
    s.plot("surface.png", dark=True)

The surface is drawn in three dimensions with its coils round it, as someone standing beside the machine sees it.
matplotlib sorts whole artists by depth, not their parts, so the depth is resolved here: the surface's faces are
painted far to near, and every short piece of coil is set behind the surface or in front of it by the depth of the
painted surface at every pixel its stroke can reach, so that no coil behind the surface is drawn over it.  Everything
drawn comes from the answer: the surface's own Fourier series, the field of its source, the magnetic axis its
evidence holds.  Needs matplotlib (pip install ".[plot]" in the torux folder).
"""
from __future__ import annotations

import math
import os

import numpy as np

COPPER = "#c8763c"
PAGES = {
    False: dict(page="#ffffff", ink="#1f2328", muted="#59636e", rule="#d1d9e0", line="#0b7d66"),
    True: dict(page="#0d1117", ink="#f0f6fc", muted="#9198a1", rule="#3d444d", line="#3fe6c0"),
}
HUE = 173.5                 # the repository's green as an OKLCH hue angle, degrees
WIDTH = 8.38                # inches: 838 pixels at 100 dpi, a README's width. Files are written at twice that
FILE_DPI = 200
DEPTH_DPI = 200             # the resolution at which the painted surface's depth is read
ELEVATION = 30.0            # the default view: degrees above the midplane
AZIMUTH = -62.0             # degrees from the x axis to where the camera stands
DISTANCE = 3.6              # the camera's distance in units of the scene's radius
COIL_PT = 1.25              # a coil's stroke at middle depth, points
MARK_PT2 = 2.2              # a magnet's mark, square points
FONTS = ("Helvetica Neue", "Helvetica", "Segoe UI", "Noto Sans", "Arial", "Liberation Sans", "DejaVu Sans")
CAPTION_PT = 9.4            # the line of words (13 px at a README's width). Smaller only when a long name or
CAPTION_MIN_PT = 7.5        # note needs it, and never below this
LABEL_PT = 8.6              # tick labels and the panels' words (12 px)
TITLE = 0.25                # inches above a panel for its words
R_LABEL = 0.46              # inches from the sections' foot to the baseline of their R label
BAR = 0.075                 # the colour scale's thickness, inches


def plot(answer, path=None, *, dark=False):
    """The surface inside its coils, coloured by |B| in tesla, beside three cross-sections in metres: the
    matplotlib figure (in a window, if there is one, that does not block), or with ``path`` (.png, .pdf, .svg) the
    file, whose path is returned.  ``dark=True`` draws it for a dark page."""
    if getattr(answer, "surface", None) is None:
        raise ValueError(f"no surface to draw: the status is {getattr(answer, 'status', None)!r}")
    ext = check_path(path)
    p = _picture(answer)
    if path is not None:
        fig = _figure(p, dark)
        fig.savefig(os.fspath(path), dpi=FILE_DPI, facecolor=fig.get_facecolor(), metadata=_metadata(ext))
        return path
    shown = _shown_by()
    fig = _figure(p, dark, pyplot=shown is not None)
    if shown == "window":
        fig.show()                                  # does not block. pyplot.show() keeps it open at a script's end
    return fig


def check_path(path):
    """The suffix of a file to draw to (None for no file), once matplotlib is there to draw it: ValueError for a
    name that is not .png, .pdf or .svg, ImportError without matplotlib."""
    ext = None
    if path is not None:
        ext = os.path.splitext(os.fspath(path))[1].lower()
        if ext not in (".png", ".pdf", ".svg"):
            raise ValueError(f"cannot draw to {os.fspath(path)!r}: name a .png, .pdf or .svg file")
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError('drawing needs matplotlib: pip install ".[plot]" in the torux folder') from e
    return ext


# ------------------------------------------------------------------------------------------------ what is drawn
class _Picture:
    """What the figure shows, in SI units and screen coordinates, computed from an answer by ``_picture``."""


def _picture(answer):
    """Everything the figure draws, from the answer and its source: the surface and |B| on it, the parts of the
    source that can be drawn and their depth against the surface, the sections, the axis, the words, the layout."""
    src = _source(answer)
    T = answer.surface
    p = _Picture()
    p.name = str(getattr(src, "id", "") or "source")
    p.nfp = int(round(float(T.nfp))) if float(T.nfp) >= 1 and float(T.nfp).is_integer() else 1
    p.flux = abs(float(T.flux))
    p.iota = float(T.iota)                        # (an Answer's iota is only for a verified surface)
    p.status = str(answer.status)
    ev = answer.evidence or {}
    ind, sol = ev.get("independent") or {}, ev.get("solver") or {}
    p.answer = p.status in ("surface", "range")
    p.check = ind.get("bn_sup", ind.get("bn_max")) if ind.get("ok") else None
    p.own = sol.get("bn_off")                     # the solver's own check: all there is when verify=False
    p.why = ev.get("unverified") or ev.get("status_at_deadline") or p.status
    # the surface on a grid of the whole torus, and the centres of its faces (which way each looks). |B| follows,
    # at the corners of the faces that are painted
    p.theta = np.linspace(0.0, 2 * np.pi, 97)
    p.phi = np.linspace(0.0, 2 * np.pi, 481)
    p.vertices = _grid(answer, p.theta, p.phi)
    thc, phc = 0.5 * (p.theta[1:] + p.theta[:-1]), 0.5 * (p.phi[1:] + p.phi[:-1])
    p.centres = _grid(answer, thc, phc)
    # the sections at the start, a quarter and a half of a field period, and the magnetic axis in them
    phi0 = float(getattr(T, "phi0", 0.0))
    p.section_phi = phi0 + np.array([0.0, 0.25, 0.5]) * 2 * np.pi / p.nfp
    p.section_theta = np.linspace(0.0, 2 * np.pi, 513)
    p.sections = []
    for ph in p.section_phi:
        x = _surface_points(answer, p.section_theta, np.full_like(p.section_theta, ph))
        p.sections.append(np.column_stack([np.hypot(x[:, 0], x[:, 1]), x[:, 2]]))
    p.axis = _axis(answer, src, p.section_phi, p.sections)
    p.lines, p.marks, p.left_out = _parts(src, float(np.ptp(p.vertices.reshape(-1, 3), axis=0).max()) / 600)
    _scene(p, answer.field)
    return p


def _grid(answer, theta, phi):
    """Cartesian points (len(theta), len(phi), 3) of the surface on a tensor grid of its angles."""
    T = answer.surface
    if hasattr(T, "geometry_grid"):
        R, Z = T.geometry_grid(theta, phi)
        R, Z = np.asarray(R), np.asarray(Z)
        P = np.broadcast_to(np.asarray(phi)[None, :], R.shape)
        return np.stack([R * np.cos(P), R * np.sin(P), Z], -1)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return _surface_points(answer, TH.ravel(), PH.ravel()).reshape(len(theta), len(phi), 3)


def _surface_points(answer, theta, phi):
    """Points of the torus the answer holds, verified or not: a torus that is not verified is drawn, and its caption
    says what it is, where answer.points refuses to give its points as an answer's."""
    from .answer import _points
    return np.asarray(_points(answer.surface, theta, phi)[0])


def _source(answer):
    """The torux.Source behind an answer: its own, or the one its field was built from."""
    from .source import Source
    for s in (getattr(answer, "source", None), getattr(getattr(answer, "_engine", None), "source", None)):
        if isinstance(s, Source):
            return s
    raise ValueError("this answer does not hold its source, so neither its coils nor the field on its surface can be "
                     "drawn: plot the answer torux.solve returns")


def _axis(answer, src, phis, sections):
    """The magnetic axis (R, Z) in each section, from the closed field lines the solver's evidence holds: a line
    that lies inside all three sections and runs along the answer's own field there, an elliptic one first. None
    when the evidence holds no such line (nothing is marked then)."""
    from matplotlib.path import Path
    T = answer.surface
    periods = list(dict.fromkeys(float(v) for v in (T.nfp, getattr(src, "nfp", T.nfp)) if float(v) > 0))
    origins = list(dict.fromkeys((float(getattr(T, "phi0", 0.0)), 0.0)))
    best = None
    for a in ((answer.evidence or {}).get("solver") or {}).get("axes") or []:
        try:
            c = {k: np.asarray(a.get(k, ()), float).ravel() for k in ("rc", "rs", "zc", "zs")}
        except (TypeError, ValueError):
            continue
        n = len(c["rc"])
        if not n or len(c["zs"]) != n:
            continue
        for k in ("rs", "zc"):
            if len(c[k]) != n:
                c[k] = np.zeros(n)
        k = np.arange(n)
        for nfp in periods:
            for phi0 in origins:
                ang = np.outer(nfp * (phis - phi0), k)
                C, S = np.cos(ang), np.sin(ang)
                R, Z = C @ c["rc"] + S @ c["rs"], C @ c["zc"] + S @ c["zs"]
                dR = nfp * (C @ (k * c["rs"]) - S @ (k * c["rc"]))
                dZ = nfp * (C @ (k * c["zs"]) - S @ (k * c["zc"]))
                if not all(Path(s).contains_point((r, z)) for s, r, z in zip(sections, R, Z)):
                    continue
                cp, sp = np.cos(phis), np.sin(phis)
                x = np.column_stack([R * cp, R * sp, Z])
                t = np.column_stack([dR * cp - R * sp, dR * sp + R * cp, dZ])
                B = np.asarray(answer.field(x), float)
                along = np.linalg.norm(np.cross(t, B), axis=-1) / (np.linalg.norm(t, axis=-1) *
                                                                    np.linalg.norm(B, axis=-1))
                if not np.all(along < 1e-6):
                    continue
                res = a.get("residue")
                elliptic = res is not None and 0 < float(res) < 1
                if best is None or (elliptic and not best[0]):
                    best = (elliptic, np.column_stack([R, Z]))
    return None if best is None else best[1]


def _parts(src, ds):
    """The source as polylines (each (k, 3), m, with straight edges cut into pieces of at most ``ds`` metres) and
    point marks ((k, 3), m), and what is not drawn, in words."""
    lines, marks, left = [], [], []
    off = idle = 0
    for part in src.parts:
        if part.kind == "fourier":
            on = part.current != 0
            if np.any(on):
                x, _ = part.points(np.linspace(0.0, 1.0, 1025))
                lines.extend(x[on])
            off += int(np.sum(~on))
        elif part.kind == "polyline":
            for i in range(part.n):
                if part.current[i] != 0:
                    q = part.loop(i)
                    lines.append(_dense(np.vstack([q, q[:1]]), ds))
            off += int(np.sum(part.current == 0))
        elif part.kind == "segments":
            on = part.current != 0
            for a, b in zip(part.a[on], part.b[on]):
                lines.append(_dense(np.vstack([a, b]), ds))
            idle += int(np.sum(~on))
        elif part.kind == "rings":
            t = np.linspace(0.0, 2 * np.pi, 1025)
            for R, Z, current in zip(part.R, part.Z, part.current):
                if current != 0:
                    lines.append(np.column_stack([R * np.cos(t), R * np.sin(t), np.full_like(t, Z)]))
            off += int(np.sum(part.current == 0))
        elif part.kind == "dipoles":
            on = np.linalg.norm(part.moment, axis=1) > 0
            marks.append(part.pos[on])
        elif part.kind == "axipoly":
            left.append(f"{part.n} axisymmetric conductor{'s' if part.n != 1 else ''}")
        elif part.kind == "toroidal":
            left.append("the 1/R toroidal field")
        else:
            left.append(f"parts of kind {part.kind!r}")
    if idle:
        left.insert(0, f"{idle} segment{'s' if idle != 1 else ''} without current")
    if off:
        left.insert(0, f"{off} coil{'s' if off != 1 else ''} without current")
    marks = np.concatenate(marks) if marks else np.zeros((0, 3))
    return lines, marks, left


def _dense(q, ds):
    """A polyline with every edge cut into pieces no longer than ds, to be set in front or behind one by one."""
    a, b = q[:-1], q[1:]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / ds)).astype(int)
    edge = np.repeat(np.arange(len(a)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return np.vstack([a[edge] + (b - a)[edge] * t[:, None], q[-1:]])


# ------------------------------------------------------------------------------------------------ the scene
class _Camera:
    """A perspective camera at ``eye`` looking at ``target``, z up."""

    def __init__(self, target, radius, elevation=ELEVATION, azimuth=AZIMUTH, distance=DISTANCE):
        e, a = math.radians(elevation), math.radians(azimuth)
        back = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        self.target = np.asarray(target, float)
        self.eye = self.target + distance * radius * back
        self.forward = -back
        right = np.cross(self.forward, [0.0, 0.0, 1.0])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.forward)

    def project(self, x):
        """Screen coordinates (..., 2) and depth along the line of sight (...) of points x (..., 3)."""
        v = np.asarray(x, float) - self.eye
        d = v @ self.forward
        return np.stack([(v @ self.right) / d, (v @ self.up) / d], -1), d


def _scene(p, field):
    """The camera, the surface's faces in painting order and |B| at their corners, the layout, and every piece of
    the source set in front of the surface or behind it."""
    V = p.vertices
    every = np.concatenate([V.reshape(-1, 3)] + list(p.lines) + [p.marks])
    zmid = 0.5 * (every[:, 2].max() + every[:, 2].min())
    radius = float(np.max(np.hypot(np.hypot(every[:, 0], every[:, 1]), every[:, 2] - zmid)))
    cam = p.camera = _Camera([0.0, 0.0, zmid], radius)
    # the faces that look towards the camera (a closed surface hides all the others), painted far to near
    q = np.stack([V[:-1, :-1], V[1:, :-1], V[1:, 1:], V[:-1, 1:]], axis=2)
    nrm = np.cross(q[:, :, 2] - q[:, :, 0], q[:, :, 3] - q[:, :, 1])
    nrm *= 1.0 if np.sum(nrm * p.centres) > 0 else -1.0          # outward: the enclosed volume is positive
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    to_eye = cam.eye - p.centres
    cosv = np.sum(nrm * to_eye, axis=-1) / np.linalg.norm(to_eye, axis=-1)
    q2, qd = cam.project(q)
    depth = qd.mean(axis=-1)
    front = cosv > -0.05
    order = np.argsort(-depth[front], kind="stable")
    p.faces = q2[front][order]
    p.face_depth = depth[front][order]
    # |B| at every distinct corner of the painted faces, and how squarely each corner faces the camera
    nth, nph = V.shape[0] - 1, V.shape[1] - 1
    I, J = np.nonzero(front)
    I, J = I[order], J[order]
    corner = np.stack([(I % nth) * nph + J % nph, ((I + 1) % nth) * nph + J % nph,
                       ((I + 1) % nth) * nph + (J + 1) % nph, (I % nth) * nph + (J + 1) % nph], -1)
    ids, p.face_corner = np.unique(corner, return_inverse=True)
    p.face_corner = p.face_corner.reshape(corner.shape)
    Vp = V[:-1, :-1].reshape(-1, 3)
    p.B_points = Vp[ids]
    p.B_values = np.linalg.norm(np.asarray(field(p.B_points), float), axis=-1)
    p.B_range = (float(p.B_values.min()), float(p.B_values.max()))
    G = V[:-1, :-1]
    vn = np.cross(np.roll(G, -1, 0) - np.roll(G, 1, 0), np.roll(G, -1, 1) - np.roll(G, 1, 1))
    vn *= 1.0 if np.sum(vn * G) > 0 else -1.0
    vn = vn.reshape(-1, 3)[ids]
    vn /= np.linalg.norm(vn, axis=-1, keepdims=True)
    eye = cam.eye - p.B_points
    p.B_cos = np.sum(vn * eye, axis=-1) / np.linalg.norm(eye, axis=-1)
    lines2 = [cam.project(ln) for ln in p.lines]
    marks2 = cam.project(p.marks) if len(p.marks) else (np.zeros((0, 2)), np.zeros(0))
    xy = np.concatenate([p.faces.reshape(-1, 2)] + [s for s, _ in lines2] + [marks2[0]])
    _layout(p, xy)
    alld = np.concatenate([d for _, d in lines2] + [marks2[1], p.face_depth])
    p.depth_range = near, far = float(alld.min()), float(alld.max())
    # the painted surface's depth, nearest within a stroke's reach of every pixel: half the widest stroke (the
    # nearest, strokes thin with distance) or mark, and a point for the smoothing of its edge
    zbuf = _depth_buffer(p.faces, p.face_depth, p.xlim, p.ylim, p.machine[2], p.machine[3])
    reach = 0.5 * max(COIL_PT * 0.5 * (near + far) / near, math.sqrt(MARK_PT2)) + 1.0
    zmin = _nearest_within(zbuf, int(math.ceil(reach / 72 * DEPTH_DPI)))
    p.seg, p.seg_depth, p.seg_front = [], [], []
    for s2, d in lines2:
        ahead = _ahead(s2, d, zmin, p)
        p.seg.append(np.stack([s2[:-1], s2[1:]], 1))
        p.seg_depth.append(np.maximum(d[:-1], d[1:]))
        p.seg_front.append(ahead[:-1] & ahead[1:])
    if p.seg:
        p.seg, p.seg_depth, p.seg_front = (np.concatenate(p.seg), np.concatenate(p.seg_depth),
                                           np.concatenate(p.seg_front))
    else:
        p.seg, p.seg_depth, p.seg_front = np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0, bool)
    p.mark_xy, p.mark_depth = marks2
    p.mark_front = _ahead(p.mark_xy, p.mark_depth, zmin, p) if len(p.mark_depth) else np.zeros(0, bool)


def _layout(p, xy):
    """Where everything goes, in inches: the machine on the left, at the aspect of what it shows. The three
    sections in a row on one scale to its right, the colour scale under them. One line of words below."""
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    mw = 5.0
    mh = float(np.clip(mw * (y1 - y0) / (x1 - x0), 2.5, 4.0))
    # the sections: each its own R window, one Z window, one scale
    pad = 0.06
    Rr = [(s[:, 0].min(), s[:, 0].max()) for s in p.sections]
    Zall = np.concatenate([s[:, 1] for s in p.sections])
    zl, zh = float(Zall.min()), float(Zall.max())
    span = max(max(b - a for a, b in Rr), zh - zl)
    m = pad * span
    Rr = [(a - m, b + m) for a, b in Rr]
    zl, zh = zl - m, zh + m
    left, gap, width = 5.62, 0.16, 8.26 - 5.62
    scale = min((width - 2 * gap) / sum(b - a for a, b in Rr), 2.0 / (zh - zl))
    sh = scale * (zh - zl)
    # the column, top down: the angles, the panels, their R ticks and the R label, a gap, the colour scale's
    # title, the scale, its ticks
    col_h = TITLE + sh + R_LABEL + 0.22 + TITLE + BAR + 0.22
    body = max(mh, col_h)
    cap = 0.46
    H = body + cap
    p.size = (WIDTH, H)
    p.machine = (0.0, cap + 0.5 * (body - mh), mw, mh)
    top = cap + 0.5 * (body + col_h)
    x = left
    p.panels = []
    for a, b in Rr:
        w = scale * (b - a)
        p.panels.append(((x, top - TITLE - sh, w, sh), (a, b), (zl, zh)))
        x += w + gap
    p.bar = (left, top - TITLE - sh - R_LABEL - 0.22 - TITLE - BAR, 1.55, BAR)
    p.caption_at = (0.25, 0.16)
    # the machine's limits: what it shows, centred, one scale on both screen axes
    W, Hm = mw, mh
    s = max((x1 - x0) / (W * 0.95), (y1 - y0) / (Hm * 0.95))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    p.xlim, p.ylim = (cx - 0.5 * s * W, cx + 0.5 * s * W), (cy - 0.5 * s * Hm, cy + 0.5 * s * Hm)


def _depth_buffer(faces, depth, xlim, ylim, W, H):
    """The depth of the painted face at every pixel of a W x H inch panel (inf where none is), by painting the
    faces far to near, unsmoothed, in colours that encode their depth."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    f = Figure(figsize=(W, H), dpi=DEPTH_DPI)
    FigureCanvasAgg(f)
    f.patch.set_facecolor((0.0, 0.0, 0.0))
    a = f.add_axes((0, 0, 1, 1))
    a.set_axis_off()
    a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    lo, hi = float(depth.min()), float(depth.max())
    k = np.round((depth - lo) / max(hi - lo, 1e-300) * (2 ** 24 - 2)).astype(np.int64) + 1
    rgb = np.stack([(k >> 16) & 255, (k >> 8) & 255, k & 255], -1) / 255.0
    a.add_collection(PolyCollection(faces, facecolors=rgb, edgecolors="none", linewidths=0, antialiased=False),
                     autolim=False)
    f.canvas.draw()
    buf = np.asarray(f.canvas.buffer_rgba())[..., :3].astype(np.int64)
    k = (buf[..., 0] << 16) | (buf[..., 1] << 8) | buf[..., 2]
    return np.where(k == 0, np.inf, lo + (k - 1) / (2 ** 24 - 2) * (hi - lo))


def _nearest_within(z, r):
    """The least depth within r pixels of every pixel, along both screen axes (a square, so a little more than a
    stroke can reach)."""
    from scipy.ndimage import minimum_filter
    return minimum_filter(z, size=2 * r + 1, mode="nearest")


def _pixels(s2, p, shape):
    """Column and row of the depth buffer's pixel under screen positions s2, as Agg paints them: the panel
    spans W x H inches at DEPTH_DPI, the buffer is cut to whole pixels, rows count down from its top."""
    W, H = p.machine[2] * DEPTH_DPI, p.machine[3] * DEPTH_DPI
    x = (s2[:, 0] - p.xlim[0]) / (p.xlim[1] - p.xlim[0]) * W
    y = (s2[:, 1] - p.ylim[0]) / (p.ylim[1] - p.ylim[0]) * H
    return np.floor(x).astype(int), np.floor(shape[0] - y).astype(int)


def _ahead(s2, d, zmin, p):
    """Whether points at screen positions s2 and depths d are nearer than the painted surface everywhere their
    stroke reaches."""
    col, row = _pixels(s2, p, zmin.shape)
    inside = (col >= 0) & (col < zmin.shape[1]) & (row >= 0) & (row < zmin.shape[0])
    z = np.full(len(d), np.inf)
    z[inside] = zmin[row[inside], col[inside]]
    return d < z


# ------------------------------------------------------------------------------------------------ colour
def _oklab_to_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    lin = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)


def _ramp(dark=False, n=256):
    """The colour scale of |B|: one hue, the repository's green, in equal perceptual steps, lighter towards weak
    field on a white page and brighter towards strong field on a dark one.  (n, 3) sRGB in [0, 1]."""
    h = math.radians(HUE)
    L = np.linspace(0.30, 0.94, 400) if dark else np.linspace(0.95, 0.36, 400)
    cap = 0.15 if dark else 0.12
    lo, hi = np.zeros_like(L), np.full_like(L, 0.4)
    for _ in range(40):                      # the most chroma sRGB holds at each lightness
        mid = 0.5 * (lo + hi)
        rgb = _oklab_to_srgb(L, mid * math.cos(h), mid * math.sin(h))
        inside = np.all((rgb >= 0) & (rgb <= 1), axis=-1)
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    C = np.minimum(cap, 0.92 * lo)           # a little inside it, and no more than the repository's own greens
    C = np.convolve(np.pad(C, 20, mode="edge"), np.ones(41) / 41, mode="valid")
    lab = np.column_stack([L, C * math.cos(h), C * math.sin(h)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lab, axis=0), axis=1))])
    u = np.linspace(0.0, s[-1], n)                                  # equal steps along the curve in OKLab
    lab = np.column_stack([np.interp(u, s, lab[:, k]) for k in range(3)])
    return np.clip(_oklab_to_srgb(lab[:, 0], lab[:, 1], lab[:, 2]), 0.0, 1.0)


def _rgb(c):
    from matplotlib.colors import to_rgb
    return np.array(to_rgb(c))


# ------------------------------------------------------------------------------------------------ drawing
def _figure(p, dark, pyplot=False):
    from matplotlib.figure import Figure
    c = PAGES[bool(dark)]
    if pyplot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=p.size, dpi=100, facecolor=c["page"])
    else:
        fig = Figure(figsize=p.size, dpi=100, facecolor=c["page"])
    font = _font()
    W, H = p.size

    def at(r):
        return (r[0] / W, r[1] / H, r[2] / W, r[3] / H)

    _machine(fig.add_axes(at(p.machine)), p, c, dark)
    for k, (rect, rl, zl) in enumerate(p.panels):
        _section(fig.add_axes(at(rect)), p, k, rect, rl, zl, c, font)
    first, last = p.panels[0][0], p.panels[-1][0]
    fig.text((0.5 * (first[0] + last[0] + last[2])) / W, (first[1] - R_LABEL) / H, "R (m)", color=c["muted"],
             fontsize=LABEL_PT, fontfamily=font, ha="center", va="baseline")
    _colour_scale(fig.add_axes(at(p.bar)), p, c, font, dark)
    text, size = _caption(p, font, W - 2 * p.caption_at[0])
    fig.text(p.caption_at[0] / W, p.caption_at[1] / H, text, color=c["muted"], fontsize=size, fontfamily=font,
             ha="left", va="baseline")
    return fig


def _colours(values, lo, hi, dark):
    """The ramp's colour of every value, interpolated between its steps (no banding)."""
    col = _ramp(dark, 1024)
    u = np.clip((np.asarray(values, float) - lo) / max(hi - lo, 1e-300), 0.0, 1.0) * (len(col) - 1)
    i = np.minimum(u.astype(int), len(col) - 2)
    f = (u - i)[:, None]
    return col[i] * (1 - f) + col[i + 1] * f


def _machine(ax, p, c, dark):
    from matplotlib.collections import LineCollection
    ax.set_axis_off()
    ax.set_xlim(*p.xlim)
    ax.set_ylim(*p.ylim)
    lo, hi = p.B_range
    rgb = _colours(p.B_values, lo, hi, dark)
    # where the surface is seen edge on it darkens a little, so that the torus reads as a solid
    rim = (np.clip(1.0 - np.abs(p.B_cos), 0.0, 1.0) ** 3)[:, None]
    toward = _rgb("#000000") if not dark else _rgb(c["page"])
    rgb = rgb * (1 - 0.30 * rim) + toward * 0.30 * rim
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], -1)[p.face_corner]          # (faces, 4 corners, 4)
    tri = np.stack([p.faces[:, [0, 1, 2]], p.faces[:, [0, 2, 3]]], 1).reshape(-1, 3, 2)
    col = np.stack([rgba[:, [0, 1, 2]], rgba[:, [0, 2, 3]]], 1).reshape(-1, 3, 4)
    near, far = p.depth_range
    page = _rgb(c["page"])

    def copper(depth):
        t = ((depth - near) / max(far - near, 1e-300))[:, None]
        return _rgb(COPPER) * (1 - 0.42 * t) + page * 0.42 * t

    def coils(sel, z):
        o = np.argsort(-p.seg_depth[sel], kind="stable")
        d = p.seg_depth[sel][o]
        lc = LineCollection(p.seg[sel][o], colors=copper(d), linewidths=COIL_PT * 0.5 * (near + far) / d,
                            capstyle="round", joinstyle="round", zorder=z)
        ax.add_collection(lc, autolim=False)

    coils(~p.seg_front, 1)
    ax.add_artist(_Shaded(tri, col, ax.transData, zorder=2))
    coils(p.seg_front, 3)
    for sel, z in ((~p.mark_front, 1.5), (p.mark_front, 3.5)):
        if np.any(sel):
            o = np.argsort(-p.mark_depth[sel], kind="stable")
            ax.scatter(p.mark_xy[sel][o, 0], p.mark_xy[sel][o, 1], s=MARK_PT2, c=copper(p.mark_depth[sel][o]),
                       linewidths=0, zorder=z)
    for a in ax.collections + ax.artists:
        a.set_rasterized(True)                      # in a .pdf or .svg the machine is an image, the rest is drawn


_SHADED = []


def _Shaded(triangles, colours, transform, zorder):
    """An artist that paints triangles in the order given, each with its corners' colours blended across it (on a
    renderer that cannot blend, each in the mean of its corners' colours, in the same order)."""
    if not _SHADED:
        from matplotlib.artist import Artist, allow_rasterization
        from matplotlib.collections import PolyCollection

        class Shaded(Artist):
            def __init__(self, triangles, colours, transform, zorder):
                super().__init__()
                self._tri, self._col = triangles, colours
                self.set_transform(transform)
                self.set_zorder(zorder)

            @allow_rasterization
            def draw(self, renderer):
                if not self.get_visible() or not len(self._tri):
                    return
                gc = renderer.new_gc()
                try:
                    renderer.draw_gouraud_triangles(gc, self._tri, self._col, self.get_transform().frozen())
                except NotImplementedError:
                    flat = self._col.mean(axis=1)
                    pc = PolyCollection(self._tri, facecolors=flat, edgecolors=flat, linewidths=0.3)
                    pc.set_transform(self.get_transform())
                    pc.set_figure(self.get_figure())
                    pc.draw(renderer)
                finally:
                    gc.restore()

        _SHADED.append(Shaded)
    return _SHADED[0](triangles, colours, transform, zorder)


def _section(ax, p, k, rect, rlim, zlim, c, font):
    s = p.sections[k]
    ax.fill(s[:, 0], s[:, 1], color=c["line"], alpha=0.10, lw=0)
    ax.plot(s[:, 0], s[:, 1], color=c["line"], lw=1.3, solid_joinstyle="round")
    if p.axis is not None:
        ax.plot([p.axis[k, 0]], [p.axis[k, 1]], ls="none", marker="+", ms=5.5, mew=0.9, color=c["ink"])
    ax.set_xlim(*rlim)
    ax.set_ylim(*zlim)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(k == 0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["rule"])
        ax.spines[side].set_linewidth(0.8)
    rt, rl = _ticks(rlim[0], rlim[1], rect[2], font, along=True, most=3)
    ax.set_xticks(rt, rl)
    if k == 0:
        zt, zl = _ticks(zlim[0], zlim[1], rect[3], font, along=False)
        ax.set_yticks(zt, zl)
        ax.set_ylabel("Z (m)", color=c["muted"], fontsize=LABEL_PT, fontfamily=font, labelpad=2)
    else:
        ax.set_yticks([])
    ax.tick_params(colors=c["rule"], labelcolor=c["muted"], labelsize=LABEL_PT, width=0.8, length=3, pad=2,
                   labelfontfamily=font[0])
    ax.set_title(f"φ = {_plain(math.degrees(p.section_phi[k]))}°", color=c["muted"], fontsize=LABEL_PT,
                 fontfamily=font, loc="left", pad=4)


def _colour_scale(ax, p, c, font, dark):
    from matplotlib.colors import ListedColormap
    lo, hi = p.B_range
    ax.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=ListedColormap(_ramp(dark, 512)),
              extent=(lo, hi, 0, 1), interpolation="nearest")
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    t, labels = _ticks(lo, hi, p.bar[2], font, along=True)
    ax.set_xticks(t, labels)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", colors=c["rule"], labelcolor=c["muted"], labelsize=LABEL_PT, width=0.8, length=2.5,
                   pad=2, labelfontfamily=font[0])
    ax.set_title("|B| on the surface (T)", color=c["muted"], fontsize=LABEL_PT, fontfamily=font, loc="left", pad=4)


# ------------------------------------------------------------------------------------------------ words and numbers
def _width(text, font, size):
    """The width of a line of text in inches, as matplotlib sets it (with its mathematics, if it has any)."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties
    w, _, _ = RendererAgg(8, 8, 72).get_text_width_height_descent(text, FontProperties(family=font, size=size),
                                                                  ismath=_has_math(text))
    return w / 72.0


def _has_math(text):
    return text.replace("\\$", "").count("$") >= 2


def _literal(text):
    """Words to be set as they are, even where they hold a dollar sign."""
    return str(text).replace("$", "\\$")


def _ticks(lo, hi, length, font, along, most=5):
    """At most ``most`` ticks in [lo, hi] on an axis ``length`` inches long, on round numbers written with the same
    decimals: labels side by side (``along``) keep a gap of one and a half lines between them, labels one above
    the other keep 1.6 lines from baseline to baseline."""
    e0 = int(math.floor(math.log10(hi - lo)))
    for e in range(e0 - 2, e0 + 2):                    # the finest step whose labels have room, steps 1, 2, 5
        for m in (1, 2, 5):
            step = m * 10.0 ** e
            k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
            if not 2 <= k1 - k0 + 1 <= most:
                continue
            dec = max(0, -e)
            t = np.round(np.arange(k0, k1 + 1) * step, dec + 3)
            labels = [f"{v:.{dec}f}".replace("-", "−") for v in t]
            t = np.clip(t, lo, hi)                     # a round number a hair past an end is that end
            labels = ["0" if lab.strip("0.−") == "" else lab for lab in labels]
            gap = (length / (hi - lo)) * step
            if along:
                need = max(0.5 * (_width(a, font, LABEL_PT) + _width(b, font, LABEL_PT))
                           for a, b in zip(labels[:-1], labels[1:])) + 1.5 * LABEL_PT / 72
            else:
                need = 1.6 * LABEL_PT / 72
            if gap >= need:
                return t, labels
    # one tick: the roundest number inside, nearest the middle
    mid = 0.5 * (lo + hi)
    for e in range(e0 + 1, e0 - 4, -1):
        for m in (1, 5, 2):
            step = m * 10.0 ** e
            v = round(mid / step) * step
            if lo <= v <= hi:
                return np.array([v]), [_plain(round(v, 12))]
    return np.array([]), []


def _plain(v):
    """A number with as few digits as it needs and a true minus sign."""
    s = f"{v:.6g}"
    if "." in s and "e" not in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "0") else s.replace("-", "−")


def _num(v, sig):
    """A number to ``sig`` significant digits, in the form a reader writes it (10 to a power past 1e-4)."""
    if v == 0:
        return "0"
    if not math.isfinite(v):
        return "not known"
    e = int(math.floor(math.log10(abs(v))))
    if -4 <= e < 6:
        s = f"{v:.{max(sig - 1 - e, 0)}f}"
    else:
        m = f"{v / 10 ** e:.{sig - 1}f}"
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        s = f"{m} × 10{str(e).translate(sup)}"
    return s.replace("-", "−")


def _caption(p, font, room):
    """The one line of words and its size: the source, the flux, the rotational transform, the field across the
    surface by the independent check (or that it is not verified), and what is not drawn.  It always fits in
    ``room`` inches: at a smaller size if it must. Then with the name shortened in its middle (to no fewer than
    16 characters), then the list of what is not drawn cut at its end, then the reason a surface is not verified.
    Each cut marked with an ellipsis.  The numbers and the claim itself are never cut."""
    numbers = [f"{_num(p.flux, 4)} Wb" + (", the outermost surface verified" if p.status == "range" else ""),
               f"$\\iota$ = {_num(p.iota, 4)}"]
    if p.answer and p.check is not None:
        claim, reason = f"field across it {_num(float(p.check), 2)} of |B| by the independent check", ""
    elif p.answer and p.own is not None:
        claim, reason = (f"field across it {_num(float(p.own), 2)} of |B| by the solver's own check, "
                         "not checked independently"), ""
    else:
        claim, reason = "not verified: ", str(p.why).replace("_", " ")
    note = "not drawn: " + ", ".join(p.left_out) if p.left_out else ""
    sep = "    "

    def line(name, reason, note):
        words = [_literal(name)] + numbers + [claim + _literal(reason)] + ([_literal(note)] if note else [])
        return sep.join(words)

    def fits(text):
        return _width(text, font, size) <= room

    size = CAPTION_PT
    w = _width(line(p.name, reason, note), font, size)
    if w > room:
        size = max(CAPTION_MIN_PT, CAPTION_PT * room / w * 0.995)
    name = p.name
    if not fits(line(name, reason, note)):
        name = _shorten(name, lambda n: fits(line(n, reason, note)), middle=True, keep=16)
    if note and not fits(line(name, reason, note)):
        note = _shorten(note, lambda t: fits(line(name, reason, t)), middle=False, keep=len("not drawn: ") + 1)
    if reason and not fits(line(name, reason, note)):
        reason = _shorten(reason, lambda t: fits(line(name, t, note)), middle=False, keep=1)
    return line(name, reason, note), size


def _shorten(text, fits, middle, keep):
    """The longest cut of ``text`` that fits and keeps at least ``keep`` of its characters, an ellipsis where it
    was cut (its middle, or its end). The text itself when it is no longer than ``keep``."""
    def cut(k):
        if k >= len(text):
            return text
        return text[:(k + 1) // 2] + "\u2026" + text[len(text) - k // 2:] if middle else text[:k] + "\u2026"
    lo, hi = min(keep, len(text)), len(text)
    while lo < hi:
        k = (lo + hi + 1) // 2
        lo, hi = (k, hi) if fits(cut(k)) else (lo, k - 1)
    return cut(lo)


def _font():
    """The first of the repository's typefaces this machine has, with DejaVu Sans for any glyph it lacks."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    first = next((f for f in FONTS if f in have), "DejaVu Sans")
    return [first, "DejaVu Sans"] if first != "DejaVu Sans" else ["DejaVu Sans"]


def _shown_by():
    """How a figure is seen here: "window" for a GUI backend, "pyplot" for a notebook backend that shows the
    figures pyplot holds, None where it is shown as a notebook cell's value (inline) or not at all."""
    import matplotlib
    try:
        from matplotlib.backends import BackendFilter, backend_registry
        gui = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    except ImportError:
        from matplotlib.rcsetup import interactive_bk as gui
    b = str(matplotlib.get_backend()).lower()
    if b in ("nbagg", "notebook", "webagg") or "ipympl" in b or "widget" in b:
        return "pyplot"
    return "window" if b in {x.lower() for x in gui} else None


def _metadata(ext):
    return {".png": {"Software": None}, ".pdf": {"Creator": None, "Producer": None, "CreationDate": None},
            ".svg": {"Creator": None, "Date": None}}[ext]
