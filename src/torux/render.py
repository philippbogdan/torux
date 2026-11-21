"""answer.render: one picture of an answer in the look of torux's README hero, traced from the answer itself.

    s = torux.solve("examples/quasr_0000954.json")
    s.render("mine.png")                    # for a light page
    s.render("mine-dark.png", dark=True)    # for a dark page
    s.render("closed.png", cut=False)       # the machine closed: no section opened

Slow where answer.plot is fast: minutes of path tracing on a laptop's cores, and the first call compiles the tracer.
Ctrl-C stops it within seconds, as a KeyboardInterrupt (torux.interrupt).  One picture and nothing else: no axes, no
scale, no words.  Everything drawn comes from the answer and its source:

    the source                         -> its coils as round copper tubes of one drawing width, magnets as small steel
                                          pills along their moments, axisymmetric conductors as the solids they sweep
    a verified surface ('surface', or  -> the surface as the field lines that weave it (lines of its own straight-
    a range's outermost torus)            field-line angle), the nested surfaces inside it as fainter lines traced in
                                          the source's field from between the magnetic axis and the surface, and a
                                          faint glow filling it
    no surface: an island band         -> the island chain the certificate names, where it is.  As wide as ISLAND_PX
                                          at README size or wider, at its true width, from lines traced from the
                                          magnetic axis to the chain's elliptic point and told by their rotation: those
                                          locked to the chain weave its islands round the elliptic lines at their
                                          centres, those on surfaces round the axis weave the nested surfaces inside
                                          it, with a glow.  Thinner: its elliptic closed field lines (the islands'
                                          centres), traced from the certificate's own points, as tubes of light
                                          winding round the magnetic axis, the only lines drawn, and a glow filling
                                          the chain
    no surface: beyond the edge        -> field lines traced from the magnetic axis outwards: those on surfaces round
                                          it as light up to the last of them, and a haze of lines started where the
                                          certificate rules out any surface, each followed until it leaves the machine
                                          or nears a coil
    anything else                      -> the source alone: torux claims nothing about the plasma, so none is drawn
    an answer that holds no source     -> ValueError before any work, saying why: one read from its JSON (solve
                                          again), one that ended in an error before the field was built, and the
                                          proof torux.nonstell gives that a source which is not a stellarator has no
                                          surface, in the proof's words

Where there is a plasma, a wedge of it is cut away in front of one cross-section (cut=True, the default), and in that
section the punctures of field lines through it show as points of light: the nested surfaces as rings, the island
chain as its elliptic points (a wide one's islands as closed curves round them), the chaotic edge as a scatter.  Each
point is a traced line's crossing of the plane or of a plane the source's field periods make the same, turned into
it.  With cut=False the machine is drawn closed, from the same view and at the same size: every line drawn all the
way round, the glow filling the whole plasma, no section opened and no points of light.

Plasma is never drawn solid: only light and fibres.  The widths of coils, fibres, tubes and points are a drawing's,
not data.  The machine is turned about its own axis to the view that frames it widest while opening a wide section.
Decoration, encoding nothing: the worn edges, faint tarnish and hair-thin scratches on the metal, its sheen, the slow
swell in brightness along each fibre and its faded ends, and on the dark page the pool of light on the floor and the
plasma's glow on the metal round it.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import pathtrace as PT

README_WIDTH = 1676        # pixels: a README's 838 at twice the density of a Retina screen
SPP = 192                  # samples per pixel where anything stands (fewer on the bare floor)
ELEVATION = 40.0           # degrees: the camera looks down on the machine at this angle
FIT = (4.6, 3.5)           # the machine is scaled to this width (or height) in the drawing's units
MARGIN = (0.05, 0.06)      # the picture's margin round the machine, of its width and height
CUT = 40.0                 # degrees of the plasma cut away in front of the cross-section a cut picture opens
ISLAND_PX = 6.0            # an island chain this many pixels wide at README size or more is drawn at its true width
WIDE = 2.0 / 3.0           # the section it opens is at least this share of the plasma's widest
SECTION = dict(rings=(2, 6), px_per_ring=14, periods=160)   # the lines whose punctures show in it: two to six,
                                                             # one for each 14 pixels of the section at README size
BAND = 64                  # rows traced at once
CHUNK = 16                 # rows of a band in one call of the compiled tracer, between which Ctrl-C is looked for
FIELD_SLAB = 4096          # points given to the field engine at once

# The look: the hero's, sizes in the drawing's units, where the machine is FIT wide
STYLE = dict(
    unit=1.0,
    floor_z=0.0,
    seed=20261111,
    min_spp=24,
    max_depth=4,
    edge_fade=32.0,                  # pixels (at README_WIDTH) over which the floor's shadows fade at the border
    pool_power=2.0,
    pool_radius=0.56 * 0.72 * 4.6,
    coil_radius=0.015,
    n_fourier=1600,
    n_revolve=512,
    magnet_radius_of_spacing=0.45,
    light=dict(dir=[-0.45, 0.35, 0.82], cone_deg=9.0, key=[2.2, 2.1, 1.95], sky=[0.3, 0.32, 0.36],
               floor_albedo=[0.85, 0.85, 0.85]),
    materials=dict(coil=dict(albedo_ref="#c8763c", sheen=[0.045, 0.32]),
                   magnet=dict(albedo_ref="#9aa0a8", sheen=[0.08, 0.22])),
    wear=dict(edge_width=0.014, tarnish_freq=4.0, tarnish_range=[0.62, 0.84], tarnish_cavity=0.25,
              cavity_reach=0.06, scratch_cell=0.32, scratch_prob=0.14, scratch_width=0.0022,
              scratch_length=[0.08, 0.26], tube_wear_freq=3.5, tube_wear_range=[0.62, 0.86]),
    plasma=dict(seed=20261109, glancing_cap=3.0, glow_samples=16, indirect_samples=16, taper_transits=0.35,
                outer_brightness=[0.25, 1.0], inner_brightness=0.5, swell_turns=[0.31, 0.53, 0.97],
                swell_depth=0.7, light_reach=0.12, floor_map_step=0.04, light_samples=(16, 64, 3)),
    lines=dict(seed=20261109, points_per_transit=1536, outer_batches=8, outer_transits=1.25,
               inner_fractions=[0.2, 0.4, 0.6, 0.8], strand_spacing=0.036, outer_lines_min=8,
               outer_lines_max=48),
)
# the fibres of light, by channel: a drawing radius, and the soft halo round them (weight, width)
CHANNELS = {
    "plasma": dict(index=0, radius=0.0025, halo=0.25, sigma=0.004),
    "threads": dict(index=1, radius=0.0018, halo=0.0, sigma=0.0),
    "tubes": dict(index=2, radius=0.0050, halo=0.30, sigma=0.004),
    "points": dict(index=3, radius=0.0030, halo=0.25, sigma=0.004),
    "greys": dict(index=4, radius=0.0025, halo=0.0, sigma=0.0),
    "beads": dict(index=5, radius=0.0050, halo=0.35, sigma=0.004),
    "islands": dict(index=6, radius=0.0025, halo=0.25, sigma=0.004),
}
PALETTES = {
    False: dict(
        bg="#ffffff",
        coil=dict(base="#e2b08c", worn="#fbe3cf", tarnish="#ad8263", scratch="#fbe6d4", worn_amount=1.0,
                  tarnish_amount=0.14, scratch_amount=0.9),
        magnet=dict(base="#c4c9d0", worn="#e6e9ed", tarnish="#8f969f", scratch="#f4f5f7", worn_amount=0.6,
                    tarnish_amount=0.25, scratch_amount=0.8),
        sheen_tint="#ffeedf", sheen=1.0, sheen_worn=3.0,
        plasma=dict(mode="veil", body="#3fd1b0", body_gain=0.42, body_strength=5.0, indirect=0.0),
        channels=dict(
            plasma=dict(index=0, colour="#0e8a72", gain=0.92, strength=3.0, mode="veil"),
            threads=dict(index=1, colour="#59636e", gain=0.5, strength=2.0, mode="veil"),
            tubes=dict(index=2, colour="#0b7d66", gain=0.95, strength=2.2, mode="veil"),
            points=dict(index=3, colour="#0e8a72", gain=0.95, strength=3.0, mode="veil"),
            greys=dict(index=4, colour="#3e4651", gain=0.6, strength=2.5, mode="veil"),
            beads=dict(index=5, colour="#086553", gain=1.0, strength=2.5, mode="veil"),
            islands=dict(index=6, colour="#0b7d66", gain=0.95, strength=3.0, mode="veil")),
        channel_order=("threads", "greys", "glow", "plasma", "points", "islands", "tubes", "beads"),
    ),
    True: dict(
        bg="#0d1117",
        floor_pool=dict(colour="#1a212c", shadow_depth=0.85),
        coil=dict(base="#b97b4a", worn="#f6c193", tarnish="#6a4931", scratch="#f7cfa6", worn_amount=1.0,
                  tarnish_amount=0.18, scratch_amount=0.9),
        magnet=dict(base="#8c939c", worn="#c3c8ce", tarnish="#5d636b", scratch="#dfe3e7", worn_amount=0.6,
                    tarnish_amount=0.3, scratch_amount=0.8),
        sheen_tint="#ffd8b5", sheen=1.3, sheen_worn=4.0,
        plasma=dict(mode="emit", body="#10b892", body_gain=0.32, body_strength=5.0, indirect=0.1,
                    indirect_colour="#1fe0b4", indirect_floor=0.06, indirect_floor_from=0.35),
        channels=dict(
            plasma=dict(index=0, colour="#1fe0b4", gain=1.3, strength=0.75, mode="emit", hot="#9ff7dc",
                        hot_gain=0.22, hot_from=1.4),
            threads=dict(index=1, colour="#b5bfcc", gain=0.5, strength=1.5, mode="emit"),
            tubes=dict(index=2, colour="#1fe0b4", gain=1.4, strength=0.9, mode="emit", hot="#9ff7dc",
                       hot_gain=0.3, hot_from=1.2),
            points=dict(index=3, colour="#1fe0b4", gain=1.3, strength=0.9, mode="emit", hot="#9ff7dc",
                        hot_gain=0.22, hot_from=1.4),
            greys=dict(index=4, colour="#b5bfcc", gain=0.6, strength=1.5, mode="emit"),
            beads=dict(index=5, colour="#1fe0b4", gain=1.6, strength=1.0, mode="emit", hot="#c8fbeb",
                       hot_gain=0.45, hot_from=1.0),
            islands=dict(index=6, colour="#1fe0b4", gain=1.45, strength=0.85, mode="emit", hot="#c8fbeb",
                         hot_gain=0.3, hot_from=1.2)),
        channel_order=("threads", "greys", "glow", "plasma", "points", "islands", "tubes", "beads"),
    ),
}


# ============================================================================ the call
def render(answer, path, *, dark=False, cut=True):
    """The picture of ``answer`` (module docstring) as a PNG file at ``path``, for a dark page with ``dark=True``,
    with a wedge of the plasma cut away in front of one cross-section, or with ``cut=False`` the machine closed.
    Returns the path.  An answer is traced once for each of cut and closed: the other page's picture then takes no
    time, and the second of cut and closed takes the tracing alone."""
    from .plot import check_folder
    name = os.fspath(path)
    if os.path.splitext(name)[1].lower() != ".png":
        raise ValueError(f"cannot render to {name!r}: name a .png file")
    check_folder(name)
    _source(answer)                             # an answer with no coils to draw says so before any work
    images = picture(answer, cut=cut)
    PT.write_png(name, images[bool(dark)])
    return path


def picture(answer, width=README_WIDTH, spp=SPP, log=None, cut=True):
    """Both pages' pictures of an answer, {False: light page, True: dark page}, each uint8 (H, W, 3), cut or closed,
    kept on the answer for the next call."""
    cache = answer.__dict__.setdefault("_render", {})
    key = (int(width), int(spp), bool(cut))
    if key not in cache:
        drawn = answer.__dict__.get("_render_drawn")
        if drawn is None:
            drawn = answer.__dict__["_render_drawn"] = content(answer, log=log)
        scene, pool = build(drawn, width, cut=cut)
        cache[key] = trace_pages(scene, pool, spp, log=log)
    return cache[key]


def trace_pages(scene, pool, spp, log=None, keep=None):
    """Path trace a scene band by band and composite both pages (``keep``, a list: each band's buffers appended)."""
    from .interrupt import held
    W, H = scene.W, scene.H
    out = {k: np.zeros((H, W, 3), np.uint8) for k in (False, True)}
    with held() as asked:
        for r0 in range(0, H, BAND):
            r1 = min(H, r0 + BAND)
            # traced CHUNK rows at a time, with Ctrl-C looked for between them: every pixel's samples are its own
            # (their random numbers start from the pixel's index), so a band is the same to the bit however it is cut
            parts = []
            for c0 in range(r0, r1, CHUNK):
                parts.append(scene.render((c0, min(r1, c0 + CHUNK)), (0, W), spp))
                if asked():
                    return out              # leaving the block raises KeyboardInterrupt
            buf = {k: np.concatenate([b[k] for b in parts], axis=1 if k == "A" else 0) for k in parts[0]}
            if keep is not None:
                keep.append(((r0, r1), {k: np.asarray(v, np.float16) for k, v in buf.items()}))
            for k in (False, True):
                out[k][r0:r1] = PT.to8(PT.composite(buf, scene, PALETTES[k], (r0, r1), (0, W), pool=pool))
            if asked():
                return out
            if log and (r1 == H or (r1 // BAND) % 8 == 0):
                log(f"  traced rows {r1}/{H}")
    return out


# ============================================================================ what is drawn
def _source(answer):
    from .source import Source
    for s in (getattr(answer, "source", None), getattr(getattr(answer, "_engine", None), "source", None)):
        if isinstance(s, Source):
            if s.has_surface_current:
                from .answer import RENDER_SHEET
                raise ValueError(RENDER_SHEET)
            return s
    from .answer import without_source
    raise ValueError(f"nothing to draw: {without_source(answer)}")


def content(answer, log=None):
    """What the picture draws, in metres: the source's solids, the fibres of light by channel, the closed region
    the plasma's glow fills, and in words what each is and where it comes from (``about``)."""
    src = _source(answer)
    out = dict(solids=solids(src), fibres=[], region=None, about=[], dots=[])
    out["turn"] = _turn(_solid_points(out["solids"]))
    out["cut"] = None                            # set by _orient once the plasma's sections are known
    out["about"].append("the source: " + (", ".join(out["solids"]["about"]) or "nothing that can be drawn"))
    st = answer.status
    cert = (answer.evidence or {}).get("certificate") or {}
    kind = str(cert.get("kind", ""))
    if st in ("surface", "range") and answer.surface is not None:
        _surface(answer, src, out, log)
    elif st == "no_surface" and kind.startswith("island_band") and (cert.get("O_line") or {}).get("points"):
        _islands(answer, src, cert, out, log)
    elif st == "no_surface" and kind.startswith("edge_barrier") and cert.get("axis_point") is not None:
        _edge(answer, src, cert, out, log)
    else:
        out["about"].append(f"no plasma: the answer is {st!r}, which holds no surface or certificate to draw")
    if out["dots"]:
        n = sum(len(d[0]) for d in out["dots"])
        out["about"].append(f"the plasma cut away over {CUT:g} degrees in front of its cross-section at phi = "
                            f"{math.degrees(out['cut'][0]) % 360:.1f} degrees, where {n} punctures of field lines "
                            f"through "
                            f"that plane and the planes the source's {int(src.nfp or 1)} field periods make the same "
                            f"show as points of light")
    return out


def solids(src):
    """The source's parts as solids, in metres: polylines to sweep a round tube along (closed or open), magnets as
    positions and moments, axisymmetric conductors as (R, Z) polygons.  Parts without current are left out, as
    answer.plot leaves them out."""
    paths, magnets, polys, about = [], [], [], []
    n = STYLE["n_fourier"]
    for part in src.parts:
        if part.kind == "fourier":
            on = part.current != 0
            if np.any(on):
                x, _ = part.points(np.arange(n) / n)
                paths += [(q, True) for q in x[on]]
                about.append(f"{int(on.sum())} coils")
        elif part.kind == "polyline":
            k = 0
            for i in range(part.n):
                if part.current[i] != 0:
                    paths.append((part.loop(i), True))
                    k += 1
            if k:
                about.append(f"{k} polygonal coils")
        elif part.kind == "segments":
            on = part.current != 0
            for a, b in zip(part.a[on], part.b[on]):
                paths.append((np.vstack([a, b]), False))
            if np.any(on):
                about.append(f"{int(on.sum())} straight segments")
        elif part.kind == "rings":
            t = 2 * np.pi * np.arange(n) / n
            k = 0
            for R, Z, current in zip(part.R, part.Z, part.current):
                if current != 0:
                    paths.append((np.column_stack([R * np.cos(t), R * np.sin(t), np.full(n, Z)]), True))
                    k += 1
            if k:
                about.append(f"{k} rings")
        elif part.kind == "dipoles":
            on = np.linalg.norm(part.moment, axis=1) > 0
            if np.any(on):
                magnets.append((part.pos[on], part.moment[on]))
                about.append(f"{int(on.sum())} magnets")
        elif part.kind == "axipoly":
            k = 0
            for i in range(part.n):
                if part.current[i] != 0:
                    polys.append(np.asarray(part.vertices[part.start[i]:part.start[i + 1]], float))
                    k += 1
            if k:
                about.append(f"{k} axisymmetric conductors")
    return dict(paths=paths, magnets=magnets, polys=polys, about=about)


def _solid_points(sol):
    pts = [q for q, _ in sol["paths"]]
    pts += [p for p, _ in sol["magnets"]]
    ph = 2 * np.pi * np.arange(96) / 96
    for P in sol["polys"]:
        pts.append(np.stack([np.outer(P[:, 0], np.cos(ph)).ravel(), np.outer(P[:, 0], np.sin(ph)).ravel(),
                             np.repeat(P[:, 1], len(ph))], -1))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def view():
    """The camera's axes: it looks along ``forward`` from the side of -y, ELEVATION degrees down."""
    el = math.radians(ELEVATION)
    f = np.array([0.0, math.cos(el), -math.sin(el)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)


def _rot(turn):
    """The rotation about the machine's own vertical axis by ``turn`` degrees."""
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _turn(points):
    """The turn of the machine about its own axis (degrees, in steps of 15) at which the camera sees it widest for
    its height: the picture's shape. Turning keeps every shape and proportion (the hero turns machines too)."""
    if len(points) < 2:
        return 0.0
    f, r, u = view()
    best = None
    for turn in range(0, 180, 15):
        q = points @ _rot(turn).T
        aspect = np.ptp(q @ u) / max(np.ptp(q @ r), 1e-12)
        if best is None or aspect < best[0] - 1e-9:
            best = (aspect, float(turn))
    return best[1]


def _orient(out):
    """The machine's turn and the cross-section the cut opens, once the plasma's sections are known: of the turns
    whose cut can open a section at least WIDE of the widest the plasma has (as the camera sees it, across), the one
    that gives the widest picture for its height.  The cut is the plane at the picture's right or left whose section
    is the wider: (its angle in the source's frame, -1 if the wedge in front of it lies at smaller angles, +1 if at
    larger)."""
    R, Z, ph = out["region"]
    w = np.ptp(R, axis=1)
    pts = _solid_points(out["solids"])
    if not len(pts):
        pts = np.stack([R * np.cos(ph)[:, None], R * np.sin(ph)[:, None], Z], -1).reshape(-1, 3)

    def width(phi):
        return float(w[int(np.argmin(np.abs(np.angle(np.exp(1j * (ph - phi))))))])
    f, r, u = view()
    best = None
    for turn in range(0, 360, 15):
        q = pts @ _rot(turn).T
        aspect = np.ptp(q @ u) / max(np.ptp(q @ r), 1e-12)
        right, left = math.radians(-turn), math.radians(180 - turn)
        cut = (right, -1.0) if width(right) >= width(left) else (left, 1.0)
        if width(cut[0]) < WIDE * w.max():
            continue
        if best is None or aspect < best[0] - 1e-9:
            best = (aspect, float(turn), cut)
    out["turn"], out["cut"] = best[1], best[2]


def _scale(points, turn=0.0):
    """Drawing units per metre: what is drawn, turned, fills FIT."""
    f, r, u = view()
    q = np.asarray(points, float) @ _rot(turn).T
    w = np.ptp(q @ r)
    h = np.ptp(q @ u)
    return float(min(FIT[0] / max(w, 1e-12), FIT[1] / max(h, 1e-12)))


# ---------------------------------------------------------------------------- field lines
class Tracer:
    """Field lines of the answer's source, traced in the cylindrical angle by torux's own tracer
    (torux.fasttrace.LineTracer, Gragg-Bulirsch-Stoer of order 12) in macro steps of a few degrees, and drawn
    between its steps by cubic Hermite interpolation on the field's own direction at each step."""

    def __init__(self, answer, src, sign):
        from .fasttrace import LineTracer
        self.eng = answer._engine
        self.nfp = max(1, int(src.nfp or 1))
        self.sign = float(sign)
        self.tracer = LineTracer(self.eng, self.nfp, k=6, sign=self.sign)
        self.steps = max(8, int(math.ceil(96 / self.nfp)))         # macro steps per field period
        self.per = 2 * np.pi / self.nfp

    def B(self, x):
        x = np.asarray(x, float).reshape(-1, 3)
        out = [np.asarray(self.eng.evaluate(x[i:i + FIELD_SLAB], grad=False, A=False)["B"], float)
               for i in range(0, len(x), FIELD_SLAB)]
        return np.concatenate(out) if out else np.zeros((0, 3))

    def trace(self, RZ, phi0, periods, stop=None):
        """Lines from (R, Z) in the plane phi0 over ``periods`` field periods (negative: backwards), one period
        at a time.  Per line: its states (k, 2) at every macro step, their angles (k,), and whether it ran its whole
        course (a line is stopped where the tracer cannot follow it in the angle, or where ``stop(states, angles)``
        says so)."""
        RZ = np.asarray(RZ, float).reshape(-1, 2)
        L = len(RZ)
        d = 1.0 if periods >= 0 else -1.0
        total = abs(int(round(periods * self.steps)))
        h = d * self.per / self.steps
        states = [[RZ[i].copy()] for i in range(L)]
        y = RZ.copy()
        alive = np.ones(L, bool)
        done = 0
        while done < total and np.any(alive):
            n = min(self.steps, total - done)
            idx = np.flatnonzero(alive)
            ph = phi0 + done * h
            o = self.tracer.trace(y[idx], ph, n * h, nsteps=n, record=True)
            rec = np.stack(o["rec"])                         # (n, l, 2)
            for j, i in enumerate(idx):
                k = int(o["steps"][j])
                pts = rec[:k, j]
                if stop is not None and k:
                    bad = np.asarray(stop(pts, ph + h * np.arange(1, k + 1)), bool)
                    if np.any(bad):
                        k = int(np.argmax(bad))
                        pts = pts[:k]
                states[i].extend(pts)
                if k < n:
                    alive[i] = False
                else:
                    y[i] = rec[n - 1, j]
            done += n
        out = []
        for i in range(L):
            s = np.array(states[i])
            out.append((s, phi0 + h * np.arange(len(s)), bool(alive[i])))
        return out

    def dense(self, states, phis, per_turn=None):
        """Cartesian points of a traced line between its macro steps: cubic Hermite in the angle on the field's
        direction dR/dphi, dZ/dphi at each step, about per_turn points a turn."""
        s, ph = np.asarray(states, float), np.asarray(phis, float)
        x = np.stack([s[:, 0] * np.cos(ph), s[:, 0] * np.sin(ph), s[:, 1]], -1)
        if len(s) < 2:
            return x
        sub = max(1, int(round((per_turn or STYLE["lines"]["points_per_transit"]) / (self.nfp * self.steps))))
        B = self.B(x)
        c, sn = np.cos(ph), np.sin(ph)
        BR = B[:, 0] * c + B[:, 1] * sn
        Bp = -B[:, 0] * sn + B[:, 1] * c
        Bp = np.where(np.abs(Bp) > 1e-300, Bp, 1e-300)
        dy = np.stack([s[:, 0] * BR / Bp, s[:, 0] * B[:, 2] / Bp], -1)
        hh = np.diff(ph)
        t = (np.arange(sub) / sub)[None, :, None]
        y0, y1 = s[:-1, None, :], s[1:, None, :]
        d0, d1 = (dy[:-1] * hh[:, None])[:, None, :], (dy[1:] * hh[:, None])[:, None, :]
        Y = ((2 * t ** 3 - 3 * t ** 2 + 1) * y0 + (t ** 3 - 2 * t ** 2 + t) * d0 + (-2 * t ** 3 + 3 * t ** 2) * y1
             + (t ** 3 - t ** 2) * d1).reshape(-1, 2)
        P = (ph[:-1, None] + hh[:, None] * t[..., 0]).reshape(-1)
        Y = np.vstack([Y, s[-1:]])
        P = np.append(P, ph[-1])
        return np.stack([Y[:, 0] * np.cos(P), Y[:, 0] * np.sin(P), Y[:, 1]], -1)


def _crossings(tr, states, phis, phi_c):
    """(R, Z) where a traced line crosses the planes phi_c + k 2 pi / nfp (every one in its course: the same plane
    of the source, turned by its own symmetry), each by cubic Hermite interpolation in the angle between the macro
    steps either side, on the field's direction there."""
    s, ph = np.asarray(states, float), np.asarray(phis, float)
    if len(s) < 2:
        return np.zeros((0, 2))
    lo, hi = min(ph[0], ph[-1]), max(ph[0], ph[-1])
    k = np.arange(math.ceil((lo - phi_c) / tr.per), math.floor((hi - phi_c) / tr.per) + 1)
    targets = phi_c + k * tr.per
    if not len(targets):
        return np.zeros((0, 2))
    asc = ph[-1] >= ph[0]
    pp = ph if asc else ph[::-1]
    ss = s if asc else s[::-1]
    i = np.clip(np.searchsorted(pp, targets) - 1, 0, len(pp) - 2)
    idx = np.unique(np.concatenate([i, i + 1]))
    x = np.stack([ss[idx, 0] * np.cos(pp[idx]), ss[idx, 0] * np.sin(pp[idx]), ss[idx, 1]], -1)
    B = tr.B(x)
    c, sn = np.cos(pp[idx]), np.sin(pp[idx])
    BR = B[:, 0] * c + B[:, 1] * sn
    Bp = -B[:, 0] * sn + B[:, 1] * c
    Bp = np.where(np.abs(Bp) > 1e-300, Bp, 1e-300)
    d = np.zeros((len(pp), 2))
    d[idx] = np.stack([ss[idx, 0] * BR / Bp, ss[idx, 0] * B[:, 2] / Bp], -1)
    h = pp[i + 1] - pp[i]
    t = ((targets - pp[i]) / np.where(h != 0, h, 1.0))[:, None]
    y = ((2 * t ** 3 - 3 * t ** 2 + 1) * ss[i] + (t ** 3 - 2 * t ** 2 + t) * d[i] * h[:, None]
         + (-2 * t ** 3 + 3 * t ** 2) * ss[i + 1] + (t ** 3 - t ** 2) * d[i + 1] * h[:, None])
    return y


def _ring_count(out):
    """How many nested surfaces the cut shows: one for each SECTION['px_per_ring'] pixels across the plasma's
    section at README size, within SECTION['rings']."""
    R, Z, _ = out["region"]
    size = float(np.median(np.maximum(np.ptp(R, axis=1), np.ptp(Z, axis=1))))
    px = size * out["scale"] * README_WIDTH / (FIT[0] * (1 + 2 * MARGIN[0]))
    return int(np.clip(round(px / SECTION["px_per_ring"]), *SECTION["rings"]))


def _section_lines(tr, axis, edge, phi0, phi_c, out, stop=None):
    """The nested surfaces in the cut: lines traced from points evenly between the magnetic axis and ``edge`` in the
    plane phi0 for SECTION['periods'] field periods, their punctures of the cut's planes as points of light.  As many
    as the section has room for at README size."""
    n = _ring_count(out)
    fr = (np.arange(n) + 1.0) / (n + 1.0)
    y0 = np.column_stack([axis[0] + fr * (edge[0] - axis[0]), axis[1] + fr * (edge[1] - axis[1])])
    n = 0
    for st, ph, _ in tr.trace(y0, phi0, SECTION["periods"], stop=stop):
        p = _crossings(tr, st, ph, phi_c)
        if len(p):
            out["dots"].append((p, "points", 1.0))
            n += 1
    return n


def _bphi_sign(answer, x):
    x = np.atleast_2d(np.asarray(x, float))
    B = np.asarray(answer._engine.evaluate(x, grad=False, A=False)["B"], float)
    ph = np.arctan2(x[:, 1], x[:, 0])
    bp = -B[:, 0] * np.sin(ph) + B[:, 1] * np.cos(ph)
    return 1.0 if np.median(bp) >= 0 else -1.0


def _fibre(pts, per_turn, rng, bright, channel="plasma"):
    """One line as a chain of glowing segments: faded ends and a slow random swell in brightness (decoration)."""
    P = STYLE["plasma"]
    n = len(pts) - 1
    if n < 2:
        return None
    i = np.arange(n) + 0.5
    ends = max(1, min(int(P["taper_transits"] * per_turn), n // 3))
    taper = PT.smoothstep(i / ends) * PT.smoothstep((n - i) / ends)
    turns = i / per_turn
    swell = np.zeros(n)
    for wl in P["swell_turns"]:
        swell += np.sin(2 * np.pi * (turns / wl + rng.uniform()))
    swell = 0.5 + 0.5 * swell / len(P["swell_turns"])
    mod = 1.0 - P["swell_depth"] + P["swell_depth"] * swell
    return pts[:-1], pts[1:], bright * taper * mod, CHANNELS[channel]["index"]


def _loop(pts, bright, channel):
    """A closed line as glowing segments of one brightness: it has no ends to fade."""
    return pts, np.roll(pts, -1, axis=0), np.full(len(pts), float(bright)), CHANNELS[channel]["index"]


def _inner_lines(tr, axis, edge, phi0, n_out, per_turn, rng, out, fractions=None):
    """The nested surfaces inside: lines traced from points between the magnetic axis and ``edge`` in the plane
    phi0, each for as many turns as its share of the outer lines (the hero's rule)."""
    P = STYLE["plasma"]
    fr = fractions or STYLE["lines"]["inner_fractions"]
    turns = [max(2, int(round(n_out * f))) for f in fr]
    y0 = np.array([[axis[0] + f * (edge[0] - axis[0]), axis[1] + f * (edge[1] - axis[1])] for f in fr])
    lines = tr.trace(y0, phi0, max(turns) * tr.nfp)
    kept = 0
    for (s, ph, whole), T in zip(lines, turns):
        k = min(len(s), T * tr.nfp * tr.steps + 1)
        if k < 2 * tr.steps:
            continue
        f = _fibre(tr.dense(s[:k], ph[:k], per_turn), per_turn, rng, P["inner_brightness"])
        if f is not None:
            out["fibres"].append(f)
            kept += 1
    return kept


def _outer_count(scale, perimeter):
    Ls = STYLE["lines"]
    n = int(np.clip(round(scale * perimeter / Ls["strand_spacing"]), Ls["outer_lines_min"], Ls["outer_lines_max"]))
    nb = Ls["outer_batches"]
    return max(1, int(round(n / nb))) * nb


# ---------------------------------------------------------------------------- a verified surface
def _torus_points(T, theta, phi):
    from .answer import _points
    return np.asarray(_points(T, theta, phi)[0])


def _torus_sections(T, nth, nph):
    """R, Z (nph, nth) of a torus on a grid of its angle theta and the cylindrical angle round the machine."""
    from .answer import _arrays, _synth
    rc, rs, zc, zs = _arrays(T)
    th = 2 * np.pi * np.arange(nth) / nth
    ph = 2 * np.pi * np.arange(nph) / nph
    ps = float(T.nfp) * (ph - float(getattr(T, "phi0", 0.0)))
    return _synth(rc, rs, th, ps).T, _synth(zc, zs, th, ps).T, ph


def _inside(poly, pts):
    """Whether points (n, 2) lie inside the closed polygon (m, 2) (the even-odd rule)."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    dy = np.where(y1 != y0, y1 - y0, 1e-300)
    cross = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / dy)
    return np.sum(cross, axis=1) % 2 == 1


def _perimeter(R, Z):
    return float(np.sum(np.hypot(np.diff(np.append(R, R[0])), np.diff(np.append(Z, Z[0])))))


def _axis_point(answer, src, phi, section):
    """The magnetic axis (R, Z) at the angle phi: of the closed lines the solver's evidence holds, the one inside
    the section given that runs along the field there (an elliptic one first).  None if there is none."""
    T = answer.surface
    axes = list((((answer.evidence or {}).get("solver") or {}).get("axes")) or []) + \
        list((answer.record or {}).get("axes") or [])
    periods = list(dict.fromkeys(float(v) for v in (getattr(T, "nfp", 0) or 0, src.nfp) if v and float(v) > 0))
    origins = list(dict.fromkeys((float(getattr(T, "phi0", 0.0)) if T is not None else 0.0, 0.0)))
    best = None
    for a in axes:
        if not isinstance(a, dict):
            continue
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
                ang = nfp * (phi - phi0) * k
                C, S = np.cos(ang), np.sin(ang)
                R, Z = float(C @ c["rc"] + S @ c["rs"]), float(C @ c["zc"] + S @ c["zs"])
                if not _inside(section, np.array([[R, Z]]))[0]:
                    continue
                dR = float(nfp * (C @ (k * c["rs"]) - S @ (k * c["rc"])))
                dZ = float(nfp * (C @ (k * c["zs"]) - S @ (k * c["zc"])))
                x = np.array([[R * math.cos(phi), R * math.sin(phi), Z]])
                tng = np.array([dR * math.cos(phi) - R * math.sin(phi), dR * math.sin(phi) + R * math.cos(phi), dZ])
                B = np.asarray(answer._engine.evaluate(x, grad=False, A=False)["B"], float)[0]
                if np.linalg.norm(np.cross(tng, B)) / (np.linalg.norm(tng) * np.linalg.norm(B)) > 1e-6:
                    continue
                res = a.get("residue")
                elliptic = res is not None and 0 < float(res) < 1
                if best is None or (elliptic and not best[0]):
                    best = (elliptic, (R, Z))
    return None if best is None else best[1]


def _surface(answer, src, out, log):
    """A verified surface: its field lines from its own coefficients, the nested surfaces inside it traced from
    between the axis and it, the glow filling it."""
    T = answer.surface
    Ls = STYLE["lines"]
    P = STYLE["plasma"]
    nfp = int(round(float(T.nfp))) if float(T.nfp) >= 1 else 1
    iota = float(T.iota)
    phi0 = float(getattr(T, "phi0", 0.0))
    R, Z, ph = _torus_sections(T, 96, max(160, int(160 * max(1, min(nfp, 5)) / 2) * 2))
    out["region"] = (R, Z, ph)
    _orient(out)
    torus = np.stack([R * np.cos(ph)[:, None], R * np.sin(ph)[:, None], Z], -1).reshape(-1, 3)
    scale = _scale(np.concatenate([_solid_points(out["solids"]), torus]), out["turn"])
    out["scale"] = scale
    th = 2 * math.pi * np.arange(512) / 512
    per = []
    for p in phi0 + 2 * math.pi * np.arange(16) / (16 * nfp):
        x = _torus_points(T, th, np.full(len(th), p))
        per.append(_perimeter(np.hypot(x[:, 0], x[:, 1]), x[:, 2]))
    n_out = _outer_count(scale, float(np.mean(per)))
    # the outer lines: theta = theta0 + iota (phi - phi_b) in the torus's own straight-field-line angle, from
    # random theta0 in eight batches of starting angles phi_b round the machine
    S = int(math.ceil(Ls["points_per_transit"] / nfp))
    dphi = 2 * math.pi / (nfp * S)
    nsec = int(round(Ls["outer_transits"] * nfp * S))
    rng = np.random.default_rng(Ls["seed"])
    nb = Ls["outer_batches"]
    per_turn = nfp * S
    for b in range(nb):
        p0 = phi0 + 2 * math.pi * b / nb
        for th0 in rng.uniform(0, 2 * math.pi, n_out // nb):
            p = p0 + dphi * np.arange(nsec + 1)
            f = _fibre(_torus_points(T, th0 + iota * (p - p0), p), per_turn, rng, rng.uniform(*P["outer_brightness"]))
            out["fibres"].append(f)
    # the nested surfaces inside: from between the axis and the section's outboard point at phi0
    thc = 2 * math.pi * np.arange(4096) / 4096
    xs = _torus_points(T, thc, np.full(len(thc), phi0))
    sec = np.column_stack([np.hypot(xs[:, 0], xs[:, 1]), xs[:, 2]])
    ax = _axis_point(answer, src, phi0, sec)
    where = "the solver's magnetic axis"
    if ax is None:
        ax = (float(sec[:, 0].mean()), float(sec[:, 1].mean()))
        where = "the section's centroid (the evidence holds no axis inside it)"
    edge = sec[int(np.argmax(sec[:, 0]))]
    tr = Tracer(answer, src, _bphi_sign(answer, xs[:64]))
    kept = _inner_lines(tr, ax, edge, phi0, n_out, per_turn, rng, out)
    # the cut: the surface's own punctures (a field line's angle advances by iota 2 pi / nfp a period, and the torus
    # is the same in every period) and the nested surfaces' from lines of their own
    m = np.arange(SECTION["periods"])
    xc = _torus_points(T, iota * 2 * math.pi / nfp * m, np.full(len(m), out["cut"][0]))
    out["dots"].append((np.column_stack([np.hypot(xc[:, 0], xc[:, 1]), xc[:, 2]]), "points", 1.0))
    rings = _section_lines(tr, ax, edge, phi0, out["cut"][0], out)
    out["about"] += [f"the verified surface at {abs(float(T.flux)):.6g} Wb (iota {iota:.6g}) as {n_out} of its field "
                     f"lines, {Ls['outer_transits']} turns each, from its own coefficients",
                     f"{kept} lines traced in the source's field from between {where} and the surface's outboard "
                     f"point: the nested surfaces",
                     f"in the cut, the surface's own punctures from its coefficients and {rings} more lines' "
                     f"punctures",
                     "a glow filling the verified surface"]


# ---------------------------------------------------------------------------- an island chain
def _strands(tr, pts, phi_s, turns=1):
    """Closed lines through the points pts (n, 2) of one orbit in the plane phi_s: each traced ``turns`` turns,
    then joined end to the start nearest it into closed loops.  Returns the loops (Cartesian, dense) and the
    traced pieces (states, angles) in order."""
    pieces = tr.trace(pts, phi_s, turns * tr.nfp)
    ends = np.array([s[-1] for s, _, _ in pieces])
    nxt = [int(np.argmin(np.hypot(*(pts - e).T))) for e in ends]
    gap = max(float(np.hypot(*(pts[n] - e))) for n, e in zip(nxt, ends))
    seen, loops = set(), []
    for k in range(len(pts)):
        if k in seen:
            continue
        chain, j = [], k
        while j not in seen:
            seen.add(j)
            chain.append(j)
            j = nxt[j]
        loops.append(chain)
    return pieces, loops, gap


def _islands(answer, src, cert, out, log):
    """The island chain the certificate names: its elliptic closed lines traced from the certificate's points in
    its plane, as tubes, the nested surfaces inside it traced from between the magnetic axis and the chain, and a
    glow filling the chain."""
    phi_s = float(cert.get("phi_s", 0.0))
    O = np.asarray(cert["O_line"]["points"], float)
    X = np.asarray((cert.get("X_line") or {}).get("points") or np.zeros((0, 2)), float)
    a = np.asarray(cert.get("axis_point") or O.mean(axis=0), float)
    p, q = int(cert["chain"]["p"]), int(cert["chain"]["q"])
    sign = _bphi_sign(answer, np.array([[a[0] * math.cos(phi_s), a[0] * math.sin(phi_s), a[1]]]))
    tr = Tracer(answer, src, sign)
    per_turn = STYLE["lines"]["points_per_transit"]
    # the chain's elliptic lines (the islands' centres) and hyperbolic ones, each point traced one turn round
    o_pieces, o_loops, o_gap = _strands(tr, O, phi_s)
    x_pieces = _strands(tr, X, phi_s)[0] if len(X) else []
    ax_piece = tr.trace(a[None], phi_s, tr.nfp)[0]
    # the glow: the region inside the chain. Its section at each angle is the closed curve through the chain's lines
    # there: a trigonometric interpolant through them in their order round the axis, resampled in that angle
    allp = list(o_pieces) + list(x_pieces)
    n = min(len(s) for s, _, _ in allp + [ax_piece])
    S = np.stack([s[:n] for s, _, _ in allp])               # (lines, n, 2)
    phis = allp[0][1][:n]
    A = ax_piece[0][:n]
    nth = 96
    m = len(S)
    keep = np.arange(0, n - 1, max(1, (n - 1) // 240))
    Rg, Zg = [], []
    for j in keep:
        c = S[:, j]
        o = np.argsort(np.arctan2(c[:, 1] - A[j, 1], c[:, 0] - A[j, 0]))
        curve = np.column_stack([_trig_resample(c[o, 0], 8 * nth), _trig_resample(c[o, 1], 8 * nth)])
        r_, z_ = _by_angle(curve, A[j], nth)
        Rg.append(r_)
        Zg.append(z_)
    Rg, Zg = np.array(Rg), np.array(Zg)
    out["region"] = (Rg, Zg, phis[keep])
    _orient(out)
    region_pts = np.stack([Rg * np.cos(phis[keep])[:, None], Rg * np.sin(phis[keep])[:, None], Zg], -1).reshape(-1, 3)
    scale = _scale(np.concatenate([_solid_points(out["solids"]), region_pts]), out["turn"])
    out["scale"] = scale
    perimeter = float(np.mean([_perimeter(Rg[j], Zg[j]) for j in range(0, len(Rg), max(1, len(Rg) // 16))]))
    n_out = _outer_count(scale, perimeter)
    width = _band_width(cert, O, a) * scale * README_WIDTH / (FIT[0] * (1 + 2 * MARGIN[0]))
    if width >= ISLAND_PX:
        _wide_islands(cert, out, log, tr, O, a, phi_s, o_pieces, o_loops, o_gap, n_out, width)
        return
    # the tubes: each loop of the chain's elliptic lines, closed
    for chain in o_loops:
        seg = [tr.dense(*o_pieces[k][:2], per_turn) for k in chain]
        loop = np.concatenate([s[:-1] for s in seg])
        out["fibres"].append(_loop(loop, 1.0, "tubes"))
    # the cut: the chain's elliptic points there (each line's first crossing of its planes), and inside the chain the
    # nested surfaces' punctures
    beads = [c[0] for c in (_crossings(tr, st, ph, out["cut"][0]) for st, ph, _ in o_pieces) if len(c)]
    out["dots"].append((np.array(beads).reshape(-1, 2), "beads", 1.0))
    edge = O[int(np.argmax(O[:, 0]))] if not len(X) or np.max(O[:, 0]) >= np.max(X[:, 0]) else X[int(np.argmax(X[:, 0]))]
    rings = _section_lines(tr, a, edge, phi_s, out["cut"][0], out)
    nfp = int(src.nfp or 1)
    out["about"] += [f"in the cut, the chain's {len(beads)} elliptic points there as beads, and inside them the "
                     f"punctures of {rings} lines traced from between the axis and the chain"]
    out["about"] += [f"the island chain at iota = {_frac(p * nfp, q)} ({q} islands in each cross-section): its "
                     f"{len(O)} elliptic points in the certificate's plane phi = {phi_s:.6g}, traced one turn each "
                     f"and closed into {len(o_loops)} lines (ends within {o_gap:.1e} m of a start), drawn as tubes, "
                     f"the only lines drawn round the machine, so that nothing is taken for them",
                     f"a glow filling the chain: its section at each angle through its {m} closed lines there"]
    del n_out


def _band_width(cert, O, a):
    """The island chain's radial width (m): the certificate's band of fluxes (between its separatrices) as radii
    about the axis, on the chain's own distance from it."""
    lo, hi = (cert.get("interval") or [None, None])[:2]
    if lo is None or not hi or hi <= lo:
        return 0.0
    r = float(np.mean(np.hypot(*(O - a).T)))
    mid = 0.5 * (lo + hi)
    return r * (math.sqrt(hi / mid) - math.sqrt(lo / mid))


ISLANDS = dict(
    ray=24,                # lines from the axis to the chain's outboard elliptic point, evenly spaced, the last on it
    periods=1024,          # each followed this many field periods (at least 64 turns)
    halves=1e-4,           # its rotation numbers over the two halves of them agree this well: not chaos
    rational=2e-5,         # and within this of a fraction of order 40 or less: locked to an island chain
    sections=256,          # periods of the last surface's line that give the glow's sections and the rings
    core=0.4,              # the nested surfaces inside the chain at this share of a surface's brightness: the
                           # islands round them are the picture's subject
)


def _island_kind(st, steps, centre, whole, q):
    """For the wide chain's lines, followed ISLANDS['periods'] field periods: 'surface' for a line on a torus round
    the axis (its rotation the same over both halves, not a rational of order at most 40), 'island' for one locked to
    the chain's rotation (a multiple of 1 / q), 'chaos' for one whose rotation differs between the halves, 'other'
    for a regular line locked to another rational, 'lost' for one that left. And its rotation number."""
    if not whole:
        return "lost", np.nan
    rho, r1, r2 = _rotation(st[::steps], centre)
    if not abs(r1 - r2) < ISLANDS["halves"]:
        return "chaos", rho
    if abs(rho * q - round(rho * q)) / q < ISLANDS["rational"]:
        return "island", rho
    qs = np.arange(1, 41)
    if np.min(np.abs(rho * qs - np.round(rho * qs)) / qs) < ISLANDS["rational"]:
        return "other", rho
    return "surface", rho


def _wide_islands(cert, out, log, tr, O, a, phi_s, o_pieces, o_loops, o_gap, n_out, width):
    """An island chain wide enough to see, at its true width.  Lines traced from the magnetic axis to the chain's
    outboard elliptic point, each told by its rotation numbers: those on surfaces round the axis give the nested
    surfaces inside the chain (woven light, a glow filling them, rings in the cut), those locked to the chain's
    rotation weave its islands (strands along each island's tube, closed curves round its centre in the cut), and
    the elliptic lines run at the islands' centres."""
    from scipy.spatial import cKDTree
    from math import gcd
    p, q = int(cert["chain"]["p"]), int(cert["chain"]["q"])
    nfp = tr.nfp
    per_turn = STYLE["lines"]["points_per_transit"]
    Ls = STYLE["lines"]
    P = STYLE["plasma"]
    sol = out["solids"]
    cp = np.concatenate([qq for qq, _ in sol["paths"]]) if sol["paths"] else _solid_points(sol)
    tree = cKDTree(cp)
    Rc = np.hypot(cp[:, 0], cp[:, 1])
    box = (Rc.min(), Rc.max(), cp[:, 2].min(), cp[:, 2].max())
    near = 2.0 * STYLE["coil_radius"] / out["scale"]

    def stop(states, phis):
        R, Z = states[:, 0], states[:, 1]
        x = np.stack([R * np.cos(phis), R * np.sin(phis), Z], -1)
        d, _ = tree.query(x)
        return (d < near) | (R < box[0]) | (R > box[1]) | (Z < box[2]) | (Z > box[3])

    K = max(ISLANDS["periods"], 64 * nfp)
    Ks = ISLANDS["sections"] * tr.steps + 1                 # the first periods of a line, for its sections and rings
    # the core: lines from the axis towards the chain's outboard elliptic point, the last on a surface inside it
    k0 = int(np.argmax(O[:, 0]))
    o0 = O[k0]
    t = (np.arange(ISLANDS["ray"]) + 1) / ISLANDS["ray"]
    seeds = a + np.outer(t, o0 - a)
    lines = tr.trace(seeds, phi_s, K, stop=stop)
    kinds = [_island_kind(st, tr.steps, a, whole, q) for st, _, whole in lines]
    if log:
        log("  islands: the ray's lines: " + " ".join(k[0][0] for k in kinds))
    surf = [k for k, (kind, _) in enumerate(kinds) if kind == "surface"]
    rng = np.random.default_rng(Ls["seed"])
    rings_drawn, kept_inner, n_pieces = 0, 0, 0
    out["region"] = None
    if surf:
        last = surf[-1]
        st_last, ph_last, _ = lines[last]
        rho = kinds[last][1]
        Rg, Zg, phg = _sections_of_line(tr, st_last[:Ks], rho, phi_s)
        out["region"] = (Rg, Zg, phg)                       # the glow fills the core, up to its last surface
        need = int(math.ceil(n_out * Ls["outer_transits"] * nfp)) * tr.steps + 1
        dense = tr.dense(st_last[:need], ph_last[:need], per_turn)
        piece = int(round(Ls["outer_transits"] * per_turn))
        n_pieces = min(n_out, (len(dense) - 1) // piece)
        first = len(out["fibres"])
        for k in range(n_pieces):
            out["fibres"].append(_fibre(dense[k * piece:(k + 1) * piece + 1], per_turn, rng,
                                        rng.uniform(*P["outer_brightness"])))
        kept_inner = _inner_lines(tr, a, seeds[last], phi_s, n_out, per_turn, rng, out)
        out["fibres"][first:] = [None if f is None else (f[0], f[1], f[2] * ISLANDS["core"], f[3])
                                 for f in out["fibres"][first:]]
        pick = sorted(set(np.round(np.linspace(0, len(surf) - 1, min(_ring_count(out), len(surf)))).astype(int)))
        for j in pick:
            c = _crossings(tr, lines[surf[j]][0][:Ks], lines[surf[j]][1][:Ks], out["cut"][0])
            if len(c):
                out["dots"].append((c, "points", 1.0))
                rings_drawn += 1
    # the islands: the ray's lines locked to the chain's rotation, from its separatrix in to the elliptic point (whose
    # own line, the ray's last, is the elliptic line drawn below)
    T = q * nfp // gcd(q, nfp)              # field periods for a line to come back to its island in the same plane
    g = gcd(q, nfp)                         # distinct island tubes: the source's turns give the rest
    span = T * tr.steps
    keep = [lines[k][:2] for k, (kind, _) in enumerate(kinds) if kind == "island" and k < len(kinds) - 1]
    # each as strands along its island's tube: pieces of one closure each, started at the returns to the island whose
    # angles round its centre are spread evenly (the outermost line the most, as the surface's own lines)
    strands, gap, cut_pts, n_isl = 0, 0.0, [], 0
    if keep:
        ret = keep[0][0][:-1:span]
        o = np.argsort(np.arctan2(ret[:, 1] - o0[1], ret[:, 0] - o0[0]))
        n_isl = _outer_count(out["scale"], _perimeter(ret[o, 0], ret[o, 1]))
    for j, (st, ph) in enumerate(keep):
        outer = j == 0
        n_ret = (len(st) - 1) // span
        ang = np.arctan2(st[:n_ret * span:span, 1] - o0[1], st[:n_ret * span:span, 0] - o0[0])
        srt = np.sort(ang)
        gap = max(gap, float(np.max(np.diff(np.append(srt, srt[0] + 2 * np.pi)))))
        n = n_isl if outer else max(4, n_isl // 2)
        aims = -np.pi + 2 * np.pi * (np.arange(n) + 0.5) / n
        picks = sorted({int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - t)))))) for t in aims})
        for k in picks:
            d = tr.dense(st[k * span:(k + 1) * span + 1], ph[k * span:(k + 1) * span + 1], per_turn)
            b = rng.uniform(*P["outer_brightness"]) if outer else P["inner_brightness"]
            for r in range(g):
                out["fibres"].append(_fibre(d @ _rot(360.0 * r / nfp).T, per_turn, rng, b, channel="islands"))
            strands += 1
        c = _crossings(tr, st, ph, out["cut"][0])
        if len(c):
            cut_pts.append(c)
    # the elliptic lines at the islands' centres, closed
    for chain in o_loops:
        seg = [tr.dense(*o_pieces[k][:2], per_turn) for k in chain]
        loop = np.concatenate([s_[:-1] for s_ in seg])
        out["fibres"].append(_loop(loop, 1.0, "islands"))
    beads = [c[0] for c in (_crossings(tr, st, ph, out["cut"][0]) for st, ph, _ in o_pieces) if len(c)]
    out["dots"].append((np.array(beads).reshape(-1, 2), "beads", 1.0))
    if cut_pts:
        out["dots"].append((np.concatenate(cut_pts), "islands", 1.0))
    count = {k: sum(1 for kind, _ in kinds if kind == k) for k in ("surface", "island", "chaos", "other", "lost")}
    out["about"] += [f"the island chain at iota = {_frac(p * nfp, q)} ({q} islands in each cross-section, "
                     f"{width:.0f} pixels wide at README size) at its true width, from {ISLANDS['ray']} lines traced "
                     f"{K} field periods from the certificate's axis point to its elliptic point at R = {o0[0]:.4f} m, "
                     f"told by their rotation numbers: {count['surface']} on surfaces round the axis, "
                     f"{count['island']} locked to the chain, {count['other']} to other chains, {count['chaos']} "
                     f"chaotic, {count['lost']} that left",
                     f"inside the chain, the last of them on a surface drawn as {n_pieces} pieces of "
                     f"{Ls['outer_transits']} turns and {kept_inner} more lines inside it, at {ISLANDS['core']:g} of a "
                     f"surface's brightness, and a glow filling it",
                     f"the islands: the {len(keep)} lines locked to the chain before the elliptic point as {strands} "
                     f"strands of one closure ({T} field periods) each, started where their returns to the island lie "
                     f"evenly round its centre (no gap between returns wider than {math.degrees(gap):.0f} degrees), "
                     f"turned by the source's own symmetry onto its {g} island tube{'s' if g != 1 else ''}; and the "
                     f"{len(O)} elliptic points traced one turn each and closed into {len(o_loops)} lines (ends within "
                     f"{o_gap:.1e} m of a start) at their centres",
                     f"in the cut, {len(beads)} elliptic points as beads, the punctures of the lines locked to the "
                     f"chain as closed curves round them, and {rings_drawn} rings of lines on surfaces inside"]


def _frac(a, b):
    from fractions import Fraction
    return str(Fraction(int(a), int(b)))


def _by_angle(curve, c, nth):
    """A closed curve (k, 2) round the point c, resampled at nth equally spaced polar angles about c (from -pi):
    sections of one torus so sampled join up whatever their own parameter."""
    ang = np.arctan2(curve[:, 1] - c[1], curve[:, 0] - c[0])
    o = np.argsort(ang)
    ang, pts = ang[o], curve[o]
    ang = np.concatenate([ang[-1:] - 2 * np.pi, ang, ang[:1] + 2 * np.pi])
    pts = np.vstack([pts[-1:], pts, pts[:1]])
    al = -np.pi + 2 * np.pi * np.arange(nth) / nth
    return np.interp(al, ang, pts[:, 0]), np.interp(al, ang, pts[:, 1])


def _trig_resample(v, n):
    """Values v (m,) at equally spaced points of a closed curve, resampled at n by trigonometric interpolation
    (the curve the points lie on, when they are its samples in order)."""
    m = len(v)
    F = np.fft.rfft(v) / m
    k = np.arange(len(F))
    if m % 2 == 0:
        F[-1] *= 0.5                                   # the Nyquist term, shared by its two frequencies
    t = 2 * np.pi * np.arange(n) / n
    w = np.where(k == 0, 1.0, 2.0)
    return np.real(np.exp(1j * np.outer(t, k)) @ (w * F))


# ---------------------------------------------------------------------------- the edge
EDGE = dict(
    ray=32,                # seeds on the ray from the axis out to the certificate's curve
    periods=256,           # each followed this many field periods (at least 64 turns)
    halves=1e-4,           # its rotation numbers over the two halves of them agree this well: not chaos
    gap=6.0,               # its punctures, in the order of that rotation, never jump more than this many times the
                           # median step: one smooth curve (an island's or a chaotic line's zigzag between pieces)
    haze=140,              # lines of the haze, from where the certificate rules out any surface: between its curve
    haze_reach=1.4,        # and this many times as far from the axis
    haze_turns=6,          # each followed this many turns either way at most
    haze_bright=0.45,      # a haze: each line faint, many together
)


def _rotation(punctures, centre):
    """The rotation number (turns per field period) of a line's punctures in one plane round the point centre, by
    the weighted Birkhoff average (it converges faster than any power of the number of periods on an invariant
    circle, and not at all in chaos), over all of them and over each half."""
    ang = np.arctan2(punctures[:, 1] - centre[1], punctures[:, 0] - centre[0])
    d = np.angle(np.exp(1j * np.diff(ang)))

    def avg(x):
        t = (np.arange(len(x)) + 0.5) / len(x)
        w = np.exp(-1.0 / (t * (1.0 - t)))
        return float(np.sum(w * x) / np.sum(w) / (2 * np.pi))
    if len(d) < 16:
        return np.nan, np.nan, np.nan
    return avg(d), avg(d[:len(d) // 2]), avg(d[len(d) // 2:])


def _kind_of_line(st, steps, centre, whole):
    """'surface' for a line on a torus round the axis: its rotation the same over both halves of its course, and its
    punctures, in the order of that rotation, one smooth curve once round the axis. 'island' for a line that winds
    with the rotation p / q of a chain (q <= 40) but is not one curve: it goes round an island's elliptic line.
    'chaos' for one whose rotation differs between the halves, 'other' for the rest, 'lost' for one that left. And its
    rotation number."""
    if not whole:
        return "lost", np.nan
    pts = st[::steps]
    rho, r1, r2 = _rotation(pts, centre)
    if not abs(r1 - r2) < EDGE["halves"]:
        return "chaos", rho
    k = np.arange(len(pts))
    o = np.argsort(np.mod(k * rho, 1.0))
    q = pts[o]
    step = np.hypot(*(np.roll(q, -1, axis=0) - q).T)
    ang = np.unwrap(np.arctan2(q[:, 1] - centre[1], q[:, 0] - centre[0]))
    winds = abs(round((ang[-1] - ang[0] + np.angle(np.exp(1j * (ang[0] - ang[-1])))) / (2 * np.pi))) == 1
    if winds and np.max(step) < EDGE["gap"] * np.median(step):
        return "surface", rho
    qs = np.arange(1, 41)
    if np.min(np.abs(rho * qs - np.round(rho * qs)) / qs) < 1e-4:
        return "island", rho
    return "other", rho


def _edge(answer, src, cert, out, log):
    """Beyond the edge: lines traced from the magnetic axis outwards in the certificate's plane.  Those on surfaces
    are light up to the last of them, and beyond it a haze of the lines that leave, each followed until it leaves
    the machine or nears a coil."""
    from scipy.spatial import cKDTree
    phi_s = float(cert.get("plane_phi", 0.0) or 0.0)
    a = np.asarray(cert["axis_point"], float)
    poly = np.asarray(cert.get("polygon") or [], float).reshape(-1, 2)
    if not len(poly):
        orb = np.asarray((cert.get("orbit") or {}).get("points") or [], float).reshape(-1, 2)
        poly = orb[np.argsort(np.arctan2(orb[:, 1] - a[1], orb[:, 0] - a[0]))] if len(orb) >= 3 else poly
    sign = _bphi_sign(answer, np.array([[a[0] * math.cos(phi_s), a[0] * math.sin(phi_s), a[1]]]))
    tr = Tracer(answer, src, sign)
    per_turn = STYLE["lines"]["points_per_transit"]
    Ls = STYLE["lines"]
    P = STYLE["plasma"]
    sol = out["solids"]
    cp = np.concatenate([q for q, _ in sol["paths"]]) if sol["paths"] else _solid_points(sol)
    tree = cKDTree(cp)
    Rc = np.hypot(cp[:, 0], cp[:, 1])
    box = (Rc.min(), Rc.max(), cp[:, 2].min(), cp[:, 2].max())
    near = 2.0 * STYLE["coil_radius"] / _scale(_solid_points(sol), out["turn"])   # inside a coil's drawn tube

    def stop(states, phis):
        R, Z = states[:, 0], states[:, 1]
        x = np.stack([R * np.cos(phis), R * np.sin(phis), Z], -1)
        d, _ = tree.query(x)
        return (d < near) | (R < box[0]) | (R > box[1]) | (Z < box[2]) | (Z > box[3])

    # lines from the ray along R from the axis out to the certificate's curve, which no surface crosses
    rb = _ray_crossing(poly, a, 0.0) if len(poly) >= 3 else None
    if rb is None:
        rb = 0.5 * (box[1] - a[0])
    fr = (np.arange(EDGE["ray"]) + 1) / EDGE["ray"]
    seeds = np.column_stack([a[0] + fr * rb, np.full(len(fr), a[1])])
    K = max(EDGE["periods"], 64 * tr.nfp)
    lines = tr.trace(seeds, phi_s, K, stop=stop)
    kinds = [_kind_of_line(s, tr.steps, a, whole) for s, _, whole in lines]
    surf = [k for k, (kind, _) in enumerate(kinds) if kind == "surface"]
    if log:
        log("  edge: the ray's lines: " + " ".join(k[0][0] for k in kinds))
    if not surf:
        out["about"].append("no field line traced from the axis lies on a surface round it: nothing drawn as "
                            "confined")
        return
    last = surf[-1]
    s_last = seeds[last]
    st_last, ph_last, _ = lines[last]
    rho = kinds[last][1]
    Rg, Zg, phg = _sections_of_line(tr, st_last, rho, phi_s)
    out["region"] = (Rg, Zg, phg)
    _orient(out)
    region_pts = np.stack([Rg * np.cos(phg)[:, None], Rg * np.sin(phg)[:, None], Zg], -1).reshape(-1, 3)
    scale = _scale(np.concatenate([_solid_points(sol), region_pts]), out["turn"])
    out["scale"] = scale
    perimeter = float(np.mean([_perimeter(Rg[j], Zg[j]) for j in range(0, len(Rg), max(1, len(Rg) // 16))]))
    n_out = _outer_count(scale, perimeter)
    rng = np.random.default_rng(Ls["seed"])
    # the last surface: its one line, cut into as many pieces as a surface has outer lines
    dense = tr.dense(st_last, ph_last, per_turn)
    piece = int(round(Ls["outer_transits"] * per_turn))
    n_pieces = min(n_out, (len(dense) - 1) // piece)
    for k in range(n_pieces):
        out["fibres"].append(_fibre(dense[k * piece:(k + 1) * piece + 1], per_turn, rng,
                                    rng.uniform(*P["outer_brightness"])))
    kept = _inner_lines(tr, a, s_last, phi_s, n_out, per_turn, rng, out)
    # the haze: lines from where the certificate rules out a surface, past its curve, both ways until they leave
    hz = []
    sec0 = np.column_stack([Rg[0], Zg[0]])
    for al in rng.uniform(-np.pi, np.pi, EDGE["haze"]):
        r_in = _ray_crossing(sec0, a, al)
        if r_in is None:
            continue
        r_out = _ray_crossing(poly, a, al) if len(poly) >= 3 else None
        r_out = 1.15 * r_in if r_out is None else max(r_out, r_in)
        r = r_out * (1.0 + rng.uniform(0.0, 1.0) * (EDGE["haze_reach"] - 1.0))
        hz.append([a[0] + r * math.cos(al), a[1] + r * math.sin(al)])
    hz = np.array(hz).reshape(-1, 2)
    fw = tr.trace(hz, phi_s, EDGE["haze_turns"] * tr.nfp, stop=stop)
    bw = tr.trace(hz, phi_s, -EDGE["haze_turns"] * tr.nfp, stop=stop)
    per_haze = per_turn // 4
    nh, left = 0, 0
    for (sf, pf, wf), (sb, pb, wb) in zip(fw, bw):
        s = np.vstack([sb[::-1], sf[1:]])
        p = np.concatenate([pb[::-1], pf[1:]])
        if len(s) < 4:
            continue
        f = _fibre(tr.dense(s, p, per_haze), per_haze, rng, EDGE["haze_bright"], channel="threads")
        if f is not None:
            out["fibres"].append(f)
            nh += 1
            left += int(not (wf and wb))
    # the cut: the punctures of the ray's lines on surfaces (six of them at most, the last among them) and of the
    # haze, where it is
    pick = sorted(set(np.round(np.linspace(0, len(surf) - 1, min(_ring_count(out), len(surf)))).astype(int)))
    for j in pick:
        st_, ph_, _ = lines[surf[j]]
        c = _crossings(tr, st_, ph_, out["cut"][0])
        if len(c):
            out["dots"].append((c, "points", 1.0))
    hd = [_crossings(tr, np.vstack([sb[::-1], sf[1:]]), np.concatenate([pb[::-1], pf[1:]]), out["cut"][0])
          for (sf, pf, _), (sb, pb, _) in zip(fw, bw) if len(sf) + len(sb) > 3]
    hd = [c for c in hd if len(c)]
    if hd:
        out["dots"].append((np.concatenate(hd), "greys", 1.0))
    count = {k: sum(1 for kind, _ in kinds if kind == k) for k in ("surface", "island", "chaos", "other", "lost")}
    out["about"] += [f"{EDGE['ray']} field lines traced {K} field periods from the ray along R from the "
                     f"certificate's magnetic axis point to its curve, in its plane: {count['surface']} on surfaces "
                     f"round the axis, {count['island']} round islands, {count['chaos']} chaotic, {count['other']} "
                     f"neither, {count['lost']} that left (by their rotation numbers, weighted Birkhoff averages, and "
                     f"whether their punctures in that order make one curve)",
                     f"the last of those on a surface, at R = {s_last[0]:.6f} m, drawn as {n_pieces} pieces of "
                     f"{Ls['outer_transits']} turns, and {kept} traced from inside it as the nested surfaces",
                     f"a haze of {nh} lines from between the certificate's curve and {EDGE['haze_reach']} times as "
                     f"far from the axis, where no surface can be, followed until they leave the machine or near a "
                     f"coil (at most {EDGE['haze_turns']} turns either way): {left} of them left",
                     "a glow filling the last surface: its sections from its line's punctures in the order of its "
                     "rotation"]


def _ray_crossing(poly, a, alpha):
    """The distance from a along the direction alpha to the farthest crossing of the closed polygon poly (m, 2), or
    None if the ray does not cross it."""
    d = np.array([math.cos(alpha), math.sin(alpha)])
    p0, p1 = poly, np.roll(poly, -1, axis=0)
    e = p1 - p0
    w = p0 - a
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(den) > 1e-300
    t = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1.0), -1.0)
    u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1.0), -1.0)
    hit = ok & (t > 0) & (u >= 0) & (u <= 1)
    return float(t[hit].max()) if np.any(hit) else None


def _sections_of_line(tr, states, rho, phi_s, nth=96):
    """The sections of the torus a traced line lies on, at every macro step of a field period: the line's states
    there in every period (one curve, by the source's own symmetry of nfp periods), ordered along it by the line's
    rotation (puncture k of phase j lies at the torus angle k rho + j rho / steps, in turns: the angle the field
    line advances by evenly), resampled at nth even angles.  The same angle labels every section, so the sections
    join round the machine.  R, Z (nfp * steps, nth) and the angles phi."""
    st = np.asarray(states, float)
    K = (len(st) - 1) // tr.steps
    k = np.arange(K)
    ug = np.arange(nth) / nth
    Rs, Zs = [], []
    for j in range(tr.steps):
        pts = st[j:K * tr.steps:tr.steps][:K]
        u = np.mod(k[:len(pts)] * rho + j * rho / tr.steps, 1.0)
        o = np.argsort(u)
        u, pts = u[o], pts[o]
        u = np.concatenate([u[-1:] - 1.0, u, u[:1] + 1.0])
        pts = np.vstack([pts[-1:], pts, pts[:1]])
        Rs.append(np.interp(ug, u, pts[:, 0]))
        Zs.append(np.interp(ug, u, pts[:, 1]))
    Rs, Zs = np.array(Rs), np.array(Zs)
    R = np.concatenate([Rs] * tr.nfp)
    Z = np.concatenate([Zs] * tr.nfp)
    ph = phi_s + tr.per / tr.steps * np.arange(tr.nfp * tr.steps)
    return R, Z, ph


# ============================================================================ the scene
def build(drawn, width, cut=True):
    """The tracer's scene of what is drawn: turned about the machine's axis, scaled to the drawing's units,
    standing on the floor, framed so that the machine fills the picture, the plasma cut open or with ``cut=False``
    closed.  Returns the scene and the dark page's pool of light."""
    sol = drawn["solids"]
    pts = _solid_points(sol)
    fib = [f for f in drawn["fibres"] if f is not None]
    fpts = np.concatenate([f[0] for f in fib] + [f[1][-1:] for f in fib]) if fib else np.zeros((0, 3))
    allp = np.concatenate([pts, fpts]) if len(fpts) else pts
    if not len(allp):
        raise ValueError("nothing to draw: the source has no part with current and the answer no plasma")
    turn = float(drawn.get("turn", 0.0))
    Rt = _rot(turn)
    s = drawn.get("scale") or _scale(allp, turn)
    r0 = STYLE["coil_radius"]
    z0 = -s * float(allp[:, 2].min()) + r0                 # the machine stands on the floor
    off = np.array([0.0, 0.0, z0])

    def place(x):
        return (np.asarray(x, float) @ Rt.T) * s + off
    caps_a, caps_b, caps_r, caps_m = [], [], [], []
    for q, closed in sol["paths"]:
        Q = place(q)
        if closed:
            caps_a.append(Q)
            caps_b.append(np.roll(Q, -1, axis=0))
        else:
            caps_a.append(Q[:-1])
            caps_b.append(Q[1:])
        caps_r.append(np.full(len(caps_a[-1]), r0))
        caps_m.append(np.zeros(len(caps_a[-1]), np.int32))
    for pos, mom in sol["magnets"]:
        rad, half = _glyph(pos, mom)
        u = (mom / np.linalg.norm(mom, axis=1, keepdims=True)) @ Rt.T
        c = place(pos)
        caps_a.append(c - u * half * s)
        caps_b.append(c + u * half * s)
        caps_r.append(np.full(len(c), rad * s))
        caps_m.append(np.ones(len(c), np.int32))
    if caps_a:
        A, B = np.concatenate(caps_a), np.concatenate(caps_b)
        CR, CM = np.concatenate(caps_r), np.concatenate(caps_m)
    else:
        A = B = np.zeros((0, 3))
        CR = np.zeros(0)
        CM = np.zeros(0, np.int32)
    tv, tn, tdp, tdr, tcv = [], [], [], [], []
    for P in sol["polys"]:
        V, Nn, dp, dr, cv = PT.revolve(P, STYLE["n_revolve"])
        tv.append(place(V.reshape(-1, 3)).reshape(-1, 3, 3))
        tn.append(Nn @ Rt.T)
        tdp.append(dp * s)
        tdr.append(dr * s)
        tcv.append(cv)
    if tv:
        tris = (np.concatenate(tv), np.concatenate(tn), np.concatenate(tdp), np.concatenate(tdr), np.concatenate(tcv))
    else:
        tris = (np.zeros((0, 3, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)))
    # the cut: no light in the wedge in front of the opened cross-section, and in that section its points
    cut = drawn.get("cut") if cut and drawn.get("dots") else None
    if fib:
        FA = np.concatenate([f[0] for f in fib])
        FB = np.concatenate([f[1] for f in fib])
        FW = np.concatenate([f[2] for f in fib])
        FC = np.concatenate([np.full(len(f[0]), f[3], np.int32) for f in fib])
        if cut is not None:
            keep = ~_in_wedge(0.5 * (FA + FB), cut)
            FA, FB, FW, FC = FA[keep], FB[keep], FW[keep], FC[keep]
    else:
        FA = FB = np.zeros((0, 3))
        FW = np.zeros(0)
        FC = np.zeros(0, np.int32)
    if cut is not None:
        e_phi = np.array([-math.sin(cut[0]), math.cos(cut[0]), 0.0])
        for RZ, ch, bright in drawn["dots"]:
            if not len(RZ):
                continue
            P = np.stack([RZ[:, 0] * math.cos(cut[0]), RZ[:, 0] * math.sin(cut[0]), RZ[:, 1]], -1)
            half = CHANNELS[ch]["radius"] / s * e_phi         # each a short rod across the plane: a point of light
            FA = np.concatenate([FA, P - half])
            FB = np.concatenate([FB, P + half])
            FW = np.concatenate([FW, np.full(len(P), float(bright))])
            FC = np.concatenate([FC, np.full(len(P), CHANNELS[ch]["index"], np.int32)])
    fibres = (place(FA), place(FB), FW, FC) if len(FA) else \
        (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int32))
    volume = np.zeros((0, 3, 3))
    glow = np.zeros((0, 4))
    if drawn.get("region") is not None:
        R, Z, ph = drawn["region"]
        if cut is None:
            X = np.stack([R * np.cos(ph)[:, None], R * np.sin(ph)[:, None], Z], -1)
            volume = place(PT.section_mesh(X).reshape(-1, 3)).reshape(-1, 3, 3)
        else:
            X = _open_sections(R, Z, ph, cut)
            volume = place(PT.tube_mesh(X).reshape(-1, 3)).reshape(-1, 3, 3)
        nth, nph, shells = STYLE["plasma"]["light_samples"]
        jt = np.linspace(0, R.shape[1], nth, endpoint=False).astype(int)
        jp = np.linspace(0, R.shape[0], nph, endpoint=False).astype(int)
        E = PT.section_samples(R[np.ix_(jp, jt)], Z[np.ix_(jp, jt)], ph[jp], shells)
        glow = np.concatenate([place(E[:, :3]), E[:, 3:] * s ** 3], 1)
    # the frame: the machine with a margin, its shadow falling where it falls and fading out at the border
    f, r, u = view()
    every = place(allp)
    x, y = every @ r, every @ u
    w, h = np.ptp(x) * (1 + 2 * MARGIN[0]), np.ptp(y) * (1 + 2 * MARGIN[1])
    W = int(width)
    H = int(round(W * h / w / 2)) * 2
    centre = 0.5 * (x.max() + x.min()) * r + 0.5 * (y.max() + y.min()) * u
    cam = PT.Camera(f, centre, max(0.5 * h, 0.5 * w * H / W))
    channels = [None] * len(CHANNELS)
    for v in CHANNELS.values():
        channels[v["index"]] = v
    scene = PT.Scene((A, B, CR, CM), tris, fibres, channels, volume, glow, cam, W, H, STYLE)
    scene.place = (s, off, Rt)                             # units per metre, the lift onto the floor, the turn
    return scene, ((0.0, 0.0), STYLE["pool_radius"])


def _wedge(cut):
    """The cut-away wedge as the angles it runs between, rising: CUT degrees on the camera's side of the opened
    plane."""
    phi, side = cut
    w = math.radians(CUT)
    return (phi - w, phi) if side < 0 else (phi, phi + w)


def _in_wedge(x, cut):
    """Whether points (n, 3), in the source's own frame, lie inside the cut-away wedge (its two planes, where the
    cross-section's points sit, are not inside it)."""
    a, b = _wedge(cut)
    d = np.mod(np.arctan2(x[:, 1], x[:, 0]) - a, 2 * np.pi)
    return (d > 1e-9) & (d < b - a - 1e-9)


def _open_sections(R, Z, ph, cut):
    """The region's sections from the wedge's far edge round the machine to its near edge, each end an exact
    section (linear between its neighbours): (nphi, ntheta, 3) points, open at the two ends."""
    a, b = _wedge(cut)
    span = 2 * np.pi - (b - a)
    rel = np.mod(ph - b, 2 * np.pi)                       # 0 at the wedge's far edge, rising round the machine
    o = np.argsort(rel)
    rel, R, Z = rel[o], R[o], Z[o]

    def at(x):
        r = np.concatenate([rel - 2 * np.pi, rel, rel + 2 * np.pi])
        RR, ZZ = np.concatenate([R, R, R]), np.concatenate([Z, Z, Z])
        j = np.clip(np.searchsorted(r, x) - 1, 0, len(r) - 2)
        f = (x - r[j]) / (r[j + 1] - r[j])
        return RR[j] + f * (RR[j + 1] - RR[j]), ZZ[j] + f * (ZZ[j + 1] - ZZ[j])
    inner = np.flatnonzero((rel > 1e-9) & (rel < span - 1e-9))
    secs = [at(0.0)] + [(R[i], Z[i]) for i in inner] + [at(span)]
    Rs = np.array([x for x, _ in secs])
    Zs = np.array([y for _, y in secs])
    phis = b + np.concatenate([[0.0], rel[inner], [span]])
    return np.stack([Rs * np.cos(phis)[:, None], Rs * np.sin(phis)[:, None], Zs], -1)


def _glyph(pos, mom):
    """Each point dipole drawn as a small pill along its moment, sized from the array's own spacing so that the
    pills tile like the grid they sit on (a drawing's glyph, not the magnet's shape): radius and half length, m."""
    from scipy.spatial import cKDTree
    if len(pos) < 3:
        return 0.01, 0.01
    mh = mom / np.linalg.norm(mom, axis=1, keepdims=True)
    k = min(24, len(pos))
    dist, idx = cKDTree(pos).query(pos, k=k)
    along, across = [], []
    for a in range(0, len(pos), 7):
        dv = pos[idx[a, 1:]] - pos[a]
        cth = np.abs(dv @ mh[a]) / np.maximum(dist[a, 1:], 1e-300)
        if np.any(cth > 0.9):
            along.append(dist[a, 1:][cth > 0.9].min())
        if np.any(cth < 0.3):
            across.append(dist[a, 1:][cth < 0.3].min())
    near = float(np.median(dist[:, 1]))
    sp_along = float(np.median(along)) if along else near
    sp_across = float(np.median(across)) if across else near
    return STYLE["magnet_radius_of_spacing"] * sp_across, 0.5 * sp_along
