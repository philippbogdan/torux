"""Poincaré sections: field lines of a source's own field, traced and recorded where they cross planes of constant
toroidal angle.

    p = torux.poincare("examples/w7x.json")          # torux chooses where lines start, how many and how far
    p.plot("w7x.png")                                 # .png, .pdf or .svg, dark=True for a dark page
    p.R, p.Z                                          # (lines, planes, crossings) in metres, NaN once a line has left
    p.iota                                            # each line's rotational transform, NaN where it has none
    p = torux.poincare(src, planes=[0.0, 0.3], lines=24, turns=400, starts=[(1.1, 0.0)])

    s = torux.solve("examples/quasr_0000954.json")
    s.poincare().plot("q954.png")                     # the same, with the surface torux found drawn over it

Lines start on the outboard side of the magnetic axis in the first plane, evenly from the axis out to the last of
PROBES trial lines that stayed in the machine, and EDGE_EXTRA beyond it.  Between neighbouring lines that stayed, up
to CHAINS island chains (the fewest islands first) get an elliptic point by Newton's method on the map of as many
field periods as the chain has islands, and two lines started round it.  Every plane stands for every plane a whole
number of field periods from it, so a line crosses each plane once a field period.  A line whose rotational transform
has not settled, or that lies on a surface and whose crossings still leave a gap along its curve wider than CLOSE_PX
pixels of the picture plot draws (838 across), is followed on, doubling its length up to FOLLOW_CAP times, until the
transform settles and the curve closes (_follow): a curve means a surface, separate loops mean islands.  A line on a
surface whose curve is still open then (or round a chain of more than ISLANDS islands) is drawn in grey
(Poincare.open).  The tracer (torux.poincare_trace) follows lines with a field within 1e-14 of an engine whose nodes
give it to rounding down to DMIN_SHARE of the magnetic axis' distance from the nearest conductor: a line has left the
machine where it comes nearer a conductor than that, turns back in phi next to a coil or leaves the box round the
source, LEFT_PAD of the conductors' extent beyond them.  Where lines still go after a sixteenth of their length and will take more than SAY_AFTER seconds, the call
says then about how long (_say_how_long), and before lines are followed on, how far.  Ctrl-C stops it within seconds, as
a KeyboardInterrupt.

The rotational transform of a line is its mean turn about the magnetic axis per turn round the machine, signed as
torux signs iota (VMEC's sign): the weighted Birkhoff average of its turn in each field period.  It converges fast on
a surface, and to the chain's rational inside an island chain.  Where the average over the first half differs from
that over the whole by more than IOTA_TOL the line has none: it wanders, or lies so near a resonance that the length
followed does not settle it.  The transform is given to the decimals that difference leaves it, those whose unit is
no smaller than the difference (_decimals): the average over the whole is nearer the limit than that over its first
half, so the digits kept have settled.

The poincare of an answer (Answer.poincare) traces the answer's source, starts one more line on its surface, and
holds what its certificate holds, followed into every plane: an island chain's elliptic points, or the X line an edge
certificate starts from and the curve its manifolds make, which no surface crosses.
"""
from __future__ import annotations

import math
import time

import numpy as np

LINES = 16                  # lines from the axis to the edge, evenly
PERIODS = 600               # field periods each line is followed, at least (turns = PERIODS / nfp)
PROBES = 32                 # lines that find the edge
PROBE_PERIODS = 80          # field periods a probe is followed
PROBE_STEPS = 12            # steps per field period of a probe (where it goes, not to rounding)
DMIN_SHARE = 0.3            # a line nearer a conductor than this share of the axis' distance from it has left
EDGE_EXTRA = 2              # lines started beyond the last line that stayed, to show where lines leave
LEFT_PAD = 0.5              # a line has left once it is this share of the conductors' extent outside it (_traced_box):
                            # surfaces and islands reach past the conductors' extent (CNT at 90 degrees, I_IL / I_PF =
                            # 4.70: the iota -6/19 island to |Z| = 0.501 m, the coils to 0.405 m)
IOTA_TOL = 1e-4             # the Birkhoff averages of the first half of a line with a rotational transform and of
                            # all of it agree to this, rad a field period. Near a resonance a surface's average settles
                            # slowly: QUASR 1100073's lines next to the axis, 0.1 % from 1/9 a period, differ by
                            # 2e-5 to 6e-5 at 600 periods and by 3e-10 to 1e-7 at 4,800. Lines that wander: 2e-4 to
                            # 2e-3, and no less at 4,800
CHAINS = 3                  # island chains looked for, the fewest islands first
QMAX = 12                   # islands in a chain at most (per plane)
THIN = 1e-4                 # an elliptic point of smaller residue has islands too thin to draw: no lines round it
KEYS = ("R", "Z", "wind", "lost", "periods", "stop", "y")
SAY_AFTER = 30.0            # seconds of tracing past which the call says how long it will take
SAY_PERIODS = 16            # the call tells how long after this share of the lines' length (8 to 64 periods)
ISLANDS = 120               # islands a period in a chain a line is told to go round, at most: W7-X's lines at the edge
                            # go round chains of 71 and 77 (rotational transforms 75/71 and 85/77)
CLOSE_PX = 2.0              # a curve is closed to the eye when no two of its crossings next along it are further apart
                            # than this, in pixels of the picture plot draws at 838 pixels across. At 4 the gaps of a
                            # line near a rational read as a ring of dashes, which a reader takes for islands
FOLLOW_CAP = 8              # a line whose transform has not settled or whose curve has not closed is followed for up
                            # to this many times the length the others are: the lines round QUASR 797878's six islands
                            # settle within 4 times, and the lines on surfaces of QUASR 219457 close within 4 times but
                            # for one, 954's within 8 but for one
STEP_TOL = 1e-15            # the one-period map's truncation error the steps are chosen for, a share of the axis' R:
                            # about where the rounding of a period's thousands of field evaluations leaves it
STEP_CHOICES = (8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256)
ORDERS = (5, 6)             # sequences of the extrapolation tried (orders 10, 12), the cheapest for its steps kept.
                            # Order 14 saves a tenth more, and on QUASR 954's surface its lines strayed 3e-11 m in
                            # 1,200 periods, ten times as far


class Poincare:
    """Field lines of a source traced through planes of constant toroidal angle.

    planes   (P,) the planes' cylindrical angles, rad.  Each stands for every plane a whole number of field periods
             from it, so a line crosses it once a field period
    R, Z     (L, P, K): where line i crosses plane j for the k-th time, m.  NaN once the line has left, and past its
             own length: K is the longest line's
    start    (L, 2): R, Z where each line starts, m, in the first plane
    left     (L,) bool: the line left the machine before it was followed the whole way
    left_at  (L, 3): R, phi, Z where it left, phi counted on along the line from the first plane, so that
             phi / (2 pi) is the turns it made first.  NaN for a line that stayed
    stop_reason  SurfaceCurrent traces carry a tuple of reasons for stopping early, with None on completion.
                 For these traces, left_at is the last accepted exterior state.
    iota     (L,): the line's rotational transform about the magnetic axis, torux's sign, rounded to the decimals it
             has settled to. NaN where it has none
    axis     (P, 2): R, Z of the magnetic axis in each plane, or None where torux found none
    turns    toroidal turns each line is followed (field periods / nfp), the most for a line that leaves
    followed (L,): the field periods each line was followed: turns x nfp, up to FOLLOW_CAP times that for a line whose
             transform had not settled or whose curve had not closed, fewer for a line that left
    open     (L,) bool: the line has a rotational transform that tells no chain of ISLANDS islands or fewer, and after
             all its length its crossings still leave a gap wider than CLOSE_PX along its curve: a surface very near a
             rational, or a finer chain. Drawn in grey
    nfp      field periods of the source
    steps, order  the tracer's steps per field period and its order
    seconds  how long it took
    surface_line  the line started on the answer's surface (Answer.poincare), else None
    held     what the answer's certificate holds, in each plane (Answer.poincare), else None: a dict with
             "elliptic" (an island chain's elliptic points) or "hyperbolic" (the X line an edge certificate starts
             from), a list of (k, 2) arrays one a plane, and for an edge certificate "barrier", the curve no surface
             crosses, (n, 2) a plane with NaN where it breaks
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def profile(self):
        """(n, 2): each line that has a rotational transform, as (distance of its start from the axis in m, iota), in
        order of the distance.  Without an axis, the distance from the first line's start."""
        ok = np.isfinite(self.iota)
        ref = self.axis[0] if self.axis is not None else self.start[0]
        d = np.hypot(self.start[ok, 0] - ref[0], self.start[ok, 1] - ref[1])
        o = np.argsort(d)
        return np.column_stack([d[o], self.iota[ok][o]])

    def plot(self, path=None, *, dark=False):
        """The crossings in every plane, in ink, on one scale in metres: the matplotlib figure, or with ``path``
        (.png, .pdf, .svg) the file, whose path is returned. ``dark=True`` for a dark page. The poincare of an answer
        draws the answer's surface over them: green only where the independent check verified it."""
        from .plot import poincare_plot
        return poincare_plot(self, path, dark=dark)

    def __str__(self):
        L = len(self.start)
        ok = np.isfinite(self.iota)
        exterior = hasattr(self, "stop_reason")
        rows = [("Poincaré section", f"{L} field line{'s' if L != 1 else ''} of {self.source_id} through "
                                     f"{len(self.planes)} plane{'s' if len(self.planes) != 1 else ''} at phi = "
                                     + ", ".join(_deg(p) for p in self.planes)),
                ("followed", f"{_num(self.turns)} turn{'s' if self.turns != 1 else ''} ({self.periods} field "
                             f"period{'s' if self.periods != 1 else ''}) each, or until it left"
                             + (f", {more} on to up to {int(self.followed.max())} periods to settle their rotational "
                                f"transforms and close their curves"
                                if (more := int(np.sum(self.followed > self.periods))) else "")),
                ("with an iota", f"{int(ok.sum())}, from {np.nanmin(self.iota):.6g} to {np.nanmax(self.iota):.6g}"
                                 if ok.any() else "none"),
                ("stopped early" if exterior else "left the machine", f"{int(self.left.sum())}")]
        n_open = int(np.sum(getattr(self, "open", ())))
        if n_open:
            rows.append(("curves not closed", f"{n_open}, drawn in grey: their crossings still leave gaps along their "
                                              f"curve, and their rotational transform tells no chain of {ISLANDS} "
                                              f"islands or fewer (a surface very near a rational, or a finer chain)"))
        rows.append(("time", f"{self.seconds:.1f} s"))
        from .answer import _table
        return _table(rows)

    __repr__ = __str__

    def lines_table(self):
        """One row a line: where it starts, its rotational transform, or when it left."""
        reasons = getattr(self, "stop_reason", None)
        last = "trace" if reasons is not None else "rotational transform"
        head = f"{'line':>4}  {'R (m)':>12}  {'Z (m)':>12}  {last}"
        rows = [head]
        period = 2 * np.pi / self.nfp
        is_open = getattr(self, "open", np.zeros(len(self.start), bool))
        for i, (st, io) in enumerate(zip(self.start, self.iota)):
            if reasons is not None:
                if reasons[i] is None:
                    what = f"completed {self.followed[i]} field periods"
                else:
                    advance = self.left_at[i, 1] / (2 * np.pi)
                    reason = {
                        "boundary": "exterior boundary cutoff",
                        "quadrature_limit": "sheet quadrature limit",
                        "toroidal_direction": "toroidal direction lost",
                        "coordinates": "invalid cylindrical coordinates",
                        "undefined_field": "undefined field",
                        "integration_limit": "integration step limit",
                    }[reasons[i]]
                    what = f"stopped after {advance:.4g} turns ({reason})"
            elif self.left[i]:
                what = f"left after {self.left_at[i, 1] / (2 * np.pi):.4g} turns"
            elif np.isfinite(io):
                what = f"{io:.{_decimals(self._spread[i], period)}f}"
                if is_open[i]:
                    what += f"   (its curve not closed in {self.followed[i]} field periods: grey)"
            else:
                what = f"none settled in {self.followed[i]} field periods"
            if i == getattr(self, "surface_line", None):
                what += "   (started on the answer's surface)"
            rows.append(f"{i + 1:>4}  {st[0]:>12.6f}  {st[1]:>12.6f}  {what}")
        return "\n".join(rows)


def poincare(source, *, planes=None, lines=None, turns=None, starts=None):
    """Field lines of ``source``'s own field, traced through planes of constant toroidal angle (a torux.Poincare).

    source   anything torux.load reads, or a torux.Source
    planes   cylindrical angles of the planes, rad (default: the start, a quarter and a half of a field period)
    lines    how many lines start from the magnetic axis out to the edge, evenly (default 16). torux adds lines in
             the island chains it finds and two beyond the edge
    turns    toroidal turns each line is followed (default: 600 field periods)
    starts   (n, 2): R, Z in m where lines start in the first plane, instead of torux's choice

    A SurfaceCurrent source requires explicit starts. Its lines stop at the
    limit of the exterior field, and no axis-based transform is estimated.
    """
    from .api import Result
    from .source import Source
    from .sources import load
    if isinstance(source, Result):
        raise TypeError("torux.poincare takes a source: the section of an answer, with its surface drawn over the "
                        "crossings, is answer.poincare()")
    src = source if isinstance(source, Source) else load(source)
    return _poincare(src, None, planes, lines, turns, starts)


def of_answer(answer, *, planes=None, lines=None, turns=None, starts=None):
    """Answer.poincare: the answer's source, with a line started on the answer's surface."""
    from .source import Source
    src = None
    for s in (getattr(answer, "source", None), getattr(getattr(answer, "_engine", None), "source", None)):
        if isinstance(s, Source):
            src = s
            break
    if src is None:
        from .answer import without_source
        raise ValueError(f"no field lines to trace: {without_source(answer)}")
    return _poincare(src, answer, planes, lines, turns, starts)


# ------------------------------------------------------------------------------------------------ the whole call
def _poincare(src, answer, planes, lines, turns, starts):
    from .poincare_trace import Schedule, Tracer
    t_start = time.time()
    nfp = int(src.nfp) if float(src.nfp) >= 1 and float(src.nfp).is_integer() else 1
    period = 2 * np.pi / nfp
    phi_ref = float(getattr(answer.surface, "phi0", 0.0)) if answer is not None and answer.surface is not None else 0.0
    planes = _planes(planes, period, phi_ref)
    nper = _periods(turns, nfp)
    if lines is not None and (int(lines) != lines or lines < 1):
        raise ValueError(f"lines={lines!r}: give how many lines start from the axis to the edge, at least 1")
    if starts is not None:
        if lines is not None:
            raise ValueError("give lines= or starts=, not both: starts= says where every line starts")
        starts = np.asarray(starts, float).reshape(-1, 2) if np.size(starts) else np.zeros((0, 2))
        if not len(starts) or not np.all(np.isfinite(starts)) or np.any(starts[:, 0] <= 0):
            raise ValueError("starts= takes (R, Z) pairs in metres with R > 0, at least one")
    if src.has_surface_current:
        if starts is None:
            raise ValueError("poincare with SurfaceCurrent requires exterior starts=(R, Z) pairs")
        if any(p.nfp % nfp for p in src.parts if p.kind == "surface_current"):
            raise ValueError("the source field period must be a period of every SurfaceCurrent")
        from .surface_trace import exterior_section
        return exterior_section(_engine(src, None), planes, starts, nper, nfp, t_start)
    phi0 = planes[0]
    rel = np.mod(planes - phi0, period)
    order = np.argsort(rel)
    box, size = _box(src)
    tbox = _traced_box(box, size)
    if answer is None:
        proved = _proved(src)
        if proved:
            raise ValueError(f"{src.id}: no section to trace: {proved}")
    if starts is not None:
        out = (starts[:, 0] < tbox[0]) | (starts[:, 0] > tbox[1]) | (starts[:, 1] < tbox[2]) | (starts[:, 1] > tbox[3])
        if out.any():
            R0, Z0 = starts[np.argmax(out)]
            raise ValueError(f"a line cannot start at R, Z = {R0:g}, {Z0:g} m: that is outside the machine, the box "
                             f"round its conductors, R {tbox[0]:.3g} to {tbox[1]:.3g} m and Z {tbox[2]:.3g} to "
                             f"{tbox[3]:.3g} m")
    tm = {}
    t = time.time()
    axis = _axis(_engine(src, answer), src, answer, nfp)
    eng = _engine(src, None, _dmin(src, axis, phi0))
    tm["axis"], t = time.time() - t, time.time()
    sign, bref = _sign(eng, src, axis, phi0, box)
    # round a magnetic axis a line has left only past the grown box (surfaces reach beyond the conductors); without
    # one there are no surfaces to cut off, and lines leave the box round the conductors as in 0.2.2
    tr = Tracer(eng, tbox if axis is not None else box, eng.dmin, sign, bref=bref,
                region=_region(src, axis, eng.dmin, box))
    ax_at = None if axis is None else axis
    if starts is None:
        seeds, steps = _choose_starts(tr, src, axis, phi0, period, box, size, LINES if lines is None else int(lines),
                                      nper, nfp)
    else:
        near = ~_clear(src, starts, phi0, tr.dstop, tr.nodes)
        if near.any():
            R0, Z0 = starts[np.argmax(near)]
            raise ValueError(f"a line cannot start at R, Z = {R0:g}, {Z0:g} m: that is within {tr.dstop:.3g} m of a "
                             f"conductor, where a line has left the machine")
        seeds, steps = starts, _steps(tr, starts[:4], phi0, period, ax_at, starts[0, 0])
    tm["starts"], t = time.time() - t, time.time()
    on_surface = None
    if answer is not None and getattr(answer, "surface", None) is not None and starts is None:
        seeds = np.concatenate([seeds, [_on_surface(answer, axis, phi0)]])
        on_surface = len(seeds) - 1
    sc_main = Schedule(rel[order], period, steps, ax_at, phi0)
    o = tr.run(seeds, sc_main, nper, phi0, progress=_say_how_long(seeds, nper, src))
    tm["lines"], t = time.time() - t, time.time()
    if seeds is not starts and axis is not None:
        extra = _chain_starts(tr, o, seeds, src, axis, phi0, period, nfp, size, steps)
        tm["chains"], t = time.time() - t, time.time()
        if len(extra):
            o2 = tr.run(extra, sc_main, nper, phi0)
            tm["chain lines"], t = time.time() - t, time.time()
            seeds = np.concatenate([seeds, extra])
            o = {k: np.concatenate([o[k], o2[k]]) for k in KEYS}
        # in order of the distance from the axis along the start line, as a profile reads
        a0 = np.asarray(axis(np.array([phi0])), float).ravel()
        s = np.argsort(np.hypot(seeds[:, 0] - a0[0], seeds[:, 1] - a0[1]), kind="stable")
        seeds = seeds[s]
        if on_surface is not None:
            on_surface = int(np.flatnonzero(s == on_surface)[0])
        o = {k: o[k][s] for k in KEYS}
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    held = _held(answer, tr, planes, period, steps) if answer is not None else None
    tm["certificate"], t = time.time() - t, time.time()
    ax_planes = None
    if axis is not None:
        aR, aZ = axis(planes)
        ax_planes = np.column_stack([aR, aZ])
    is_open = np.zeros(len(seeds), bool)
    if axis is not None:
        px = _pixels(o["R"][:, inv], o["Z"][:, inv], o["lost"], seeds, ax_planes, answer, held, planes)
        o, is_open = _follow(tr, o, sc_main, nper, phi0, period, px, src, tm.get("lines", 0.0) / max(1, nper))
        tm["following on"], t = time.time() - t, time.time()
    R, Z = o["R"][:, inv], o["Z"][:, inv]
    iota, spread = _iota(o["wind"], period, spread=True) if axis is not None else (np.full(len(seeds), np.nan),) * 2
    left = o["lost"]
    iota[left] = np.nan
    iota = np.array([round(v, _decimals(s, period)) if np.isfinite(v) else v for v, s in zip(iota, spread)])
    return Poincare(planes=planes, R=R, Z=Z, start=np.array(seeds, float), left=left,
                    left_at=np.where(left[:, None], o["stop"], np.nan), iota=iota, axis=ax_planes,
                    turns=nper / nfp, periods=nper, followed=np.asarray(o["periods"], int), nfp=nfp,
                    open=is_open & ~left, source_id=str(src.id), seconds=time.time() - t_start,
                    steps=steps, order=2 * tr.k, surface_line=on_surface, held=held, _answer=answer, _source=src,
                    _spread=spread, _timings=tm, _nodes=eng.info.n_nodes)


def _through(tr, pts, phi_s, planes, period, steps, nper):
    """Points (n, 2) in the plane phi_s followed through every plane for nper field periods: (n, P, nper) R and Z,
    the planes in the order given."""
    from .poincare_trace import Schedule
    rel = np.mod(planes - phi_s, period)
    rel = np.where(rel < 1e-12, 0.0, rel)
    # the schedule starts at phi_s itself: a plane there is reached at the end of each period
    cuts = [0.0] + sorted(r for r in rel if r > 0)
    sc = Schedule(np.array(cuts), period, steps, None, phi_s)
    o = tr.run(np.asarray(pts, float).reshape(-1, 2), sc, nper, phi_s)
    col = {c: k for k, c in enumerate(cuts)}          # Schedule numbers the plane at cuts[k] k, the period's end 0
    R = np.stack([o["R"][:, col[r]] for r in rel], 1)
    Z = np.stack([o["Z"][:, col[r]] for r in rel], 1)
    return R, Z


def _held(answer, tr, planes, period, steps):
    """What an island-chain or edge certificate holds, followed into every plane (Poincare.held), or None."""
    c = (answer.evidence or {}).get("certificate") or {}
    kind = str(c.get("kind", ""))
    try:
        if kind.startswith("island_band"):
            line = c.get("O_line") or {}
            pts = np.asarray(line.get("points") or [], float).reshape(-1, 2)
            q = int((c.get("chain") or {}).get("q") or len(pts))
            if not len(pts) or q < 1:
                return None
            R, Z = _through(tr, pts[:1], float(c.get("phi_s", 0.0)), planes, period, steps, q)
            return dict(elliptic=[np.column_stack([R[0, j], Z[0, j]]) for j in range(len(planes))])
        if kind.startswith("edge_barrier"):
            orbit = c.get("orbit") or {}
            pts = np.asarray(orbit.get("points") or [], float).reshape(-1, 2)
            q = int(orbit.get("q") or len(pts))
            phi_s = float(c.get("plane_phi", 0.0))
            out = {}
            if len(pts) and q >= 1:
                R, Z = _through(tr, pts[:1], phi_s, planes, period, steps, q)
                out["hyperbolic"] = [np.column_stack([R[0, j], Z[0, j]]) for j in range(len(planes))]
            poly = np.asarray(c.get("polygon") or [], float).reshape(-1, 2)
            if len(poly):
                poly = poly[::max(1, len(poly) // 2000)]
                R, Z = _through(tr, poly, phi_s, planes, period, steps, 1)
                out["barrier"] = [np.column_stack([R[:, j, 0], Z[:, j, 0]]) for j in range(len(planes))]
            return out or None
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _on_surface(answer, axis, phi0):
    """(R, Z) of the answer's surface in the first plane where the start line crosses it: outboard of the axis at
    the axis' height, found by bisection on the surface's poloidal angle.  Where it has no such crossing, theta = 0."""
    from .plot import _surface_points

    def at(th):
        x = _surface_points(answer, np.atleast_1d(th), np.full(np.size(th), phi0))
        return np.hypot(x[:, 0], x[:, 1]), x[:, 2]
    th = np.linspace(0.0, 2 * np.pi, 1025)
    R, Z = at(th)
    if axis is not None:
        aR, aZ = (float(np.ravel(v)[0]) for v in axis(np.array([phi0])))
        f = Z - aZ
        k = np.flatnonzero((np.sign(f[:-1]) != np.sign(f[1:])) & (0.5 * (R[:-1] + R[1:]) > aR))
        if len(k):
            a, b = th[k[0]], th[k[0] + 1]
            fa = f[k[0]]
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = at(m)[1][0] - aZ
                if np.sign(fm) == np.sign(fa):
                    a, fa = m, fm
                else:
                    b = m
            Rm, Zm = at(0.5 * (a + b))
            return [float(Rm[0]), float(Zm[0])]
    return [float(R[0]), float(Z[0])]


def _say_how_long(seeds, nper, src):
    """A progress for Tracer.run: once the lines have been followed SAY_PERIODS periods (a sixteenth of their length,
    8 to 64), where lines are still going and they will take more than SAY_AFTER seconds in all, how long: the time so
    far and the rest of the periods at the pace of the last piece, which only the lines still going cost.  To the
    logger "torux" at INFO and through torux.ask.say (the command line writes it to stderr), as a search of
    torux.solve says what it asks and how long it may take.  Lines that leave within the first periods (a field whose
    lines do not stay) are gone by then, and nothing is said, however busy the machine."""
    check = min(nper, max(8, min(64, nper // SAY_PERIODS)))
    t0, last, said = time.time(), [time.time(), 0], []

    def progress(done, lost):
        now = time.time()
        pace = (now - last[0]) / max(done - last[1], 1)
        last[:] = [now, done]
        if said or done < check or done >= nper:
            return
        said.append(True)
        if np.all(lost):
            return
        est = (now - t0) + pace * (nper - done)
        if est > SAY_AFTER:
            from .ask import _say
            n = len(seeds)
            _say(f"{getattr(src, 'id', None) or 'the source'}: tracing {n} field line{'s' if n != 1 else ''} through "
                 f"{nper} field periods, about {_duration(est)}, then more for any lines it follows on")
    return progress


def _duration(s):
    if s < 90:
        return f"{s:.0f} s"
    return f"{s / 60:.0f} minutes"


def _planes(planes, period, phi_ref):
    if planes is None:
        return phi_ref + np.array([0.0, 0.25, 0.5]) * period
    p = np.atleast_1d(np.asarray(planes, float)).ravel()
    if not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("planes= takes the cylindrical angles of one or more planes, in radians")
    if np.any(np.abs(p) > 2 * np.pi + 1e-9):
        v = float(p[np.argmax(np.abs(p) > 2 * np.pi + 1e-9)])
        raise ValueError(f"planes= takes angles in radians, within one turn (-2π to 2π): {v:g} rad is "
                         f"{math.degrees(v):g}°. Where {v:g} is meant in degrees, give math.radians({v:g})")
    rel = np.mod(p - p[0], period)
    rel = np.where(np.isclose(rel, period, rtol=0, atol=1e-12), 0.0, rel)
    if len(np.unique(np.round(rel, 12))) != len(p):
        raise ValueError("planes= holds two planes a whole number of field periods apart: they are the same plane")
    return p


def _periods(turns, nfp):
    if turns is None:
        return int(math.ceil(PERIODS / nfp)) * nfp
    t = float(turns)
    if not (t > 0 and math.isfinite(t)):
        raise ValueError(f"turns={turns!r}: give the toroidal turns each line is followed, a positive number")
    if round(t * nfp) < 1:
        raise ValueError(f"turns={turns!r}: a line crosses each plane once a field period, so it is followed whole "
                         f"periods, at least one: {_num(1 / nfp)} turns")
    return int(round(t * nfp))


def _engine(src, answer, dmin=None):
    if answer is not None and getattr(answer, "_engine", None) is not None:
        return answer._engine
    from .field import FieldEngine
    # nodes in multiples of 16: the tracer may sum a coil far from a line on every 2nd .. 16th of them
    return FieldEngine(src, dmin=dmin or src.meta.get("dmin", 0.05), node_multiple=16)


def _dmin(src, axis, phi0):
    """The distance from the conductors beyond which the field is computed to rounding, and nearer which a line
    has left the machine: DMIN_SHARE of the magnetic axis' distance from the nearest conductor, at least the
    source's own (a QUASR design's: a fifth of its coils' distance from its outermost surface)."""
    own = float(src.meta.get("dmin", 0.05))
    x = _conductor_points(src)
    if axis is None or not len(x):
        return own
    from scipy.spatial import cKDTree
    ph = phi0 + np.linspace(0.0, 2 * np.pi, 257)[:-1]
    R, Z = axis(ph)
    d = float(cKDTree(x).query(np.column_stack([R * np.cos(ph), R * np.sin(ph), Z]))[0].min())
    return max(own, DMIN_SHARE * d)


def _deg(phi):
    from .plot import _plain
    return _plain(round(math.degrees(phi), 6)) + "°"


def _num(x):
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.3g}"


# ------------------------------------------------------------------------------------------------ the source
def _region(src, axis, dstop, box, n=1024, rng=None):
    """n points where lines can go: round the magnetic axis, out to one and a half times its distance from the
    nearest conductor, no nearer a conductor than dstop (nearer, a line has left).  Without an axis, in the box."""
    from scipy.spatial import cKDTree
    rng = rng or np.random.default_rng(1)
    x = _conductor_points(src)
    ph = rng.uniform(0.0, 2 * np.pi, 4 * n)
    if axis is not None:
        aR, aZ = axis(ph)
        ring = np.column_stack([aR * np.cos(ph), aR * np.sin(ph), aZ])
        reach = 1.5 * float(cKDTree(x).query(ring)[0].min()) if len(x) else 1.0
        rr, th = reach * np.sqrt(rng.uniform(0.0, 1.0, 4 * n)), rng.uniform(0.0, 2 * np.pi, 4 * n)
        R, Z = aR + rr * np.cos(th), aZ + rr * np.sin(th)
    else:
        R, Z = rng.uniform(box[0], box[1], 4 * n), rng.uniform(box[2], box[3], 4 * n)
    pts = np.column_stack([R * np.cos(ph), R * np.sin(ph), Z])[R > 0]
    if len(x):
        pts = pts[cKDTree(x).query(pts)[0] >= dstop]
    return pts[:n]


def _conductor_points(src):
    """Points on every conductor of the source (m), for its extent."""
    pts = []
    for part in src.parts:
        if part.kind == "fourier":
            x, _ = part.points(np.linspace(0.0, 1.0, 257))
            pts.append(x.reshape(-1, 3))
        elif part.kind == "polyline":
            pts.append(np.asarray(part.points, float).reshape(-1, 3))
        elif part.kind == "segments":
            pts += [np.asarray(part.a, float), np.asarray(part.b, float)]
        elif part.kind == "dipoles":
            pts.append(np.asarray(part.pos, float))
        elif part.kind == "rings":
            t = np.linspace(0.0, 2 * np.pi, 65)
            for R, Z in zip(np.atleast_1d(part.R), np.atleast_1d(part.Z)):
                pts.append(np.column_stack([R * np.cos(t), R * np.sin(t), np.full_like(t, Z)]))
        elif part.kind == "axipoly":
            for V in getattr(part, "vertices", []) or []:
                V = np.asarray(V, float).reshape(-1, 2)
                pts.append(np.column_stack([V[:, 0], np.zeros(len(V)), V[:, 1]]))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _box(src):
    """(Rmin, Rmax, Zmin, Zmax) a little larger than the conductors' extent, and the extent's size (m).  Where the
    conductors have almost no extent in R or in Z (one ring, rings at one radius, coils in one plane), that side is
    as wide as the other, or as half the largest R."""
    x = _conductor_points(src)
    if not len(x):
        return np.array([1e-6, 1e6, -1e6, 1e6]), 1.0
    R = np.hypot(x[:, 0], x[:, 1])
    lo, hi = np.array([R.min(), x[:, 2].min()]), np.array([R.max(), x[:, 2].max()])
    wide = max(float(max(hi - lo)), 0.5 * float(R.max()))
    for d in (0, 1):
        if hi[d] - lo[d] < 0.25 * wide:
            mid = 0.5 * (lo[d] + hi[d])
            lo[d], hi[d] = mid - 0.5 * wide, mid + 0.5 * wide
    size = float(max(hi - lo))
    pad = 0.05 * size
    return np.array([max(1e-6, lo[0] - pad), hi[0] + pad, lo[1] - pad, hi[1] + pad]), size


def _traced_box(box, size):
    """The box outside which a line round a magnetic axis has left: the box round the conductors (_box) grown on every
    side by LEFT_PAD of their extent.  The box round the conductors alone, 5 % of their extent past them, cut off lines
    on surfaces that reach beyond it.  Where lines start (the probes that find the edge) and the lines of a source
    with no axis keep the box round the conductors."""
    pad = LEFT_PAD * size
    return np.array([max(1e-6, box[0] - pad), box[1] + pad, box[2] - pad, box[3] + pad])


def _clear(src, y, phi0, d, nodes=()):
    """Which points (R, Z) of the plane phi0 are at least d from every conductor (and from the points nodes)."""
    from scipy.spatial import cKDTree
    x = np.concatenate([_conductor_points(src), np.asarray(nodes, float).reshape(-1, 3)])
    y = np.asarray(y, float).reshape(-1, 2)
    if not len(x) or d <= 0:
        return np.ones(len(y), bool)
    p = np.column_stack([y[:, 0] * math.cos(phi0), y[:, 0] * math.sin(phi0), y[:, 1]])
    return cKDTree(x).query(p)[0] >= d


def _proved(src):
    """torux.solve's proof that a source which is not a stellarator has no surface (torux.nonstell), in the words its
    answer says it in, or None: a stellarator, or a source the proof does not settle."""
    from . import nonstell as NS
    from .answer import proof_words
    try:
        if not NS.scope(src)[0]:
            return None
        return proof_words(NS.classify(src))
    except Exception:  # noqa: BLE001
        return None


def _sign(eng, src, axis, phi0, box):
    """The sign of B_phi where the lines go, and |B| there: at the axis, else in the middle of the box."""
    if axis is not None:
        R, Z = (float(np.ravel(v)[0]) for v in axis(np.array([phi0])))
    else:
        R, Z = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    c, s = math.cos(phi0), math.sin(phi0)
    B = np.asarray(eng.evaluate(np.array([[R * c, R * s, Z]]), grad=False, A=False)["B"], float)[0]
    bp = -B[0] * s + B[1] * c
    return (1.0 if bp >= 0 else -1.0), float(np.linalg.norm(B))


def _axis(eng, src, answer, nfp):
    """The magnetic axis as a callable phi -> (R, Z), or None: from the answer's evidence when it holds the axis,
    else found as torux.solve finds it (the elliptic period-one line nearest the conductors' median radius)."""
    if answer is not None:
        f = _axis_of_answer(answer, eng, nfp)
        if f is not None:
            return f
    from .analysis import find_axes
    from .axis_general import find_axes_general
    try:
        with np.errstate(all="ignore"):                 # numpy's notes on a search's own arithmetic
            axes = find_axes(eng, src) if src.stellsym else find_axes_general(eng, src)
    except Exception:  # noqa: BLE001
        return None
    ell = [a for a in axes if 0 < a.residue < 1]
    if not ell:
        return None
    pts = src.sample_points()
    Rm = float(np.median(np.hypot(pts[:, 0], pts[:, 1]))) if len(pts) else 1.0
    ax = min(ell, key=lambda a: abs(float(np.ravel(a.at(np.array([0.0]))[0])[0]) - Rm))
    return lambda phi: tuple(np.asarray(v, float) for v in ax.at(np.atleast_1d(np.asarray(phi, float))))


def _axis_of_answer(answer, eng, nfp):
    """The elliptic axis the solver's evidence holds, checked to run along the field."""
    T = answer.surface
    axes = ((answer.evidence or {}).get("solver") or {}).get("axes") or []
    periods = list(dict.fromkeys(v for v in (int(round(float(getattr(T, "nfp", nfp)))), nfp) if v >= 1)) \
        if T is not None else [nfp]
    origins = list(dict.fromkeys((float(getattr(T, "phi0", 0.0)) if T is not None else 0.0, 0.0)))
    test = np.linspace(0.0, 2 * np.pi, 7)[:-1]
    best = None
    for a in axes:
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
        res = a.get("residue")
        if res is None or not 0 < float(res) < 1:
            continue
        for q in periods:
            for p0 in origins:
                def at(phi, c=c, q=q, p0=p0, n=n):
                    ang = np.outer(q * (np.atleast_1d(np.asarray(phi, float)) - p0), np.arange(n))
                    C, S = np.cos(ang), np.sin(ang)
                    return C @ c["rc"] + S @ c["rs"], C @ c["zc"] + S @ c["zs"]
                if _along_field(eng, at, test) and best is None:
                    best = at
    return best


def _along_field(eng, at, phis, tol=1e-6):
    h = 1e-6
    R, Z = at(phis)
    R1, Z1 = at(phis + h)
    R0, Z0 = at(phis - h)
    dR, dZ = (R1 - R0) / (2 * h), (Z1 - Z0) / (2 * h)
    c, s = np.cos(phis), np.sin(phis)
    t = np.column_stack([dR * c - R * s, dR * s + R * c, dZ])
    B = np.asarray(eng.evaluate(np.column_stack([R * c, R * s, Z]), grad=False, A=False)["B"], float)
    x = np.linalg.norm(np.cross(t, B), axis=-1) / (np.linalg.norm(t, axis=-1) * np.linalg.norm(B, axis=-1))
    return bool(np.all(np.isfinite(x)) and np.all(x < tol))


# ------------------------------------------------------------------------------------------------ where lines start
def _steps(tr, y, phi0, period, axis, Rscale):
    """Steps per field period, and the extrapolation's order, for an error of STEP_TOL of Rscale in a field period at
    the least cost: for k in ORDERS, the one-period map of the lines y in n and in 2n steps differ by the error of n
    steps (2^(2k) times that of 2n), measured at the first n of STEP_CHOICES where it is clear of rounding, and the
    error falls as n^-(2k) from there.  The tracer is left at the order chosen."""
    from .poincare_trace import Schedule
    y = np.asarray(y, float).reshape(-1, 2)

    def end(n):
        o = tr.run(y, Schedule([0.0], period, n, None, phi0), 1, phi0)
        return np.column_stack([o["R"][:, 0, 0], o["Z"][:, 0, 0]])

    def need():
        for n in STEP_CHOICES:
            a, b = end(n), end(2 * n)
            ok = np.all(np.isfinite(a), -1) & np.all(np.isfinite(b), -1)
            if not ok.any():
                return 24                   # no line to measure on stays a period: the lines go nowhere either way
            diff = float(np.max(np.abs(a[ok] - b[ok]))) / Rscale
            if diff < 1e-10:
                m = n * max(1.0, diff / STEP_TOL) ** (1.0 / (2 * tr.k))
                return int(next((c for c in STEP_CHOICES if c >= m), STEP_CHOICES[-1]))
        return STEP_CHOICES[-1]
    best = None
    for k in ORDERS:
        tr.set_order(k)
        n = need()
        if best is None or (1 + k * k) * n < (1 + best[0] ** 2) * best[1]:
            best = (k, n)
    tr.set_order(best[0])
    return best[1]


def _ray(tr, axis, phi0, box):
    """The start line: from the axis outward along R in the first plane, as far as the box or the conductors let
    a line start (m)."""
    aR, aZ = (float(np.ravel(v)[0]) for v in axis(np.array([phi0])))
    r = np.linspace(0.0, box[1] - aR, 2001)[1:]
    R = aR + r
    if len(tr.nodes) and tr.dstop > 0:
        from scipy.spatial import cKDTree
        x = np.column_stack([R * math.cos(phi0), R * math.sin(phi0), np.full_like(R, aZ)])
        near = cKDTree(tr.nodes).query(x)[0] < 2 * tr.dstop
        if near.any():
            r = r[:max(1, int(np.argmax(near)))]
    return aR, aZ, float(r[-1])


def _choose_starts(tr, src, axis, phi0, period, box, size, nlines, nper, nfp):
    """Where lines start: evenly on the start line from the axis out to the last of PROBES lines that stayed for
    PROBE_PERIODS, and EDGE_EXTRA beyond it.  Without an axis, evenly across the middle of the box."""
    from .poincare_trace import Schedule
    if axis is None:
        # across the middle of the box, clear of the conductors, where a line goes once round the machine
        R = np.linspace(box[0], box[1], 4 * nlines + 2)[1:-1]
        y = np.column_stack([R, np.full(len(R), 0.5 * (box[2] + box[3]))])
        y = y[_clear(src, y, phi0, max(tr.dstop, 0.02 * size), tr.nodes)]
        o = tr.run(y, Schedule([0.0], period, PROBE_STEPS, None, phi0), 1, phi0)
        y = y[~o["lost"]]
        if not len(y):
            raise ValueError(f"{getattr(src, 'id', None) or 'the source'}: no section to trace: torux found no "
                             f"magnetic axis, and no field line started across the box round the conductors goes "
                             f"round the machine for one field period. If these are a tokamak's coils or a dipole, say "
                             f"so with kind='tokamak' or kind='dipole' (--kind): torux.solve settles those by proof")
        y = y[np.unique(np.round(np.linspace(0, len(y) - 1, min(nlines, len(y)))).astype(int))]
        return y, _steps(tr, y[::max(1, len(y) // 4)], phi0, period, None, float(np.median(y[:, 0])))
    aR, aZ, rmax = _ray(tr, axis, phi0, box)
    r = rmax * np.arange(1, PROBES + 1) / PROBES
    probe = np.column_stack([aR + r, np.full(PROBES, aZ)])
    # the probes find the edge in the box round the conductors, as in 0.2.2, so that the lines start where they did
    traced, tr.box = tr.box, np.asarray(box, float)
    try:
        o = tr.run(probe, Schedule([0.0], period, PROBE_STEPS, None, phi0), PROBE_PERIODS, phi0)
    finally:
        tr.box = traced
    stayed = ~o["lost"]
    last = r[np.flatnonzero(stayed)[-1]] if stayed.any() else r[0]
    # the steps are chosen on the inner half, where surfaces are: further out, nearer the coils, a period's error
    # may be a few times larger, which over a line's length stays far below what a picture or a profile can show
    steps = _steps(tr, np.column_stack([aR + last * np.array([0.15, 0.3, 0.45]), np.full(3, aZ)]), phi0, period,
                   axis, aR)
    rs = last * np.arange(1, nlines + 1) / nlines
    beyond = r[(r > last) & ~stayed][:EDGE_EXTRA]
    rs = np.concatenate([rs, beyond])
    return np.column_stack([aR + rs, np.full(len(rs), aZ)]), steps


# ------------------------------------------------------------------------------------------------ following on
def _gaps(o, i, rho, px):
    """The largest gap between crossings of line i next along its curve, over the planes, in pixels of the picture
    (px a metre): the crossings ordered by k rho mod 1, the order a turn of rho a period gives points of a curve (exact
    on an invariant circle, whatever its shape)."""
    out = 0.0
    for j in range(o["R"].shape[1]):
        R, Z = o["R"][i, j], o["Z"][i, j]
        k = np.flatnonzero(np.isfinite(R))
        if len(k) < 8:
            return np.inf
        s = np.argsort(np.mod((k + 1) * rho, 1.0))
        x, y = R[k][s], Z[k][s]
        d = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
        out = max(out, float(d.max()) * px)
    return out


def _island(rho, spread=0.0, qmax=ISLANDS):
    """Whether a mean turn a period of rho (turns) is a rational of at most qmax islands, to three times its own
    spread (turns), within 1e-7 to 1e-6: a line round an island chain, whose crossings are loops apart and close no
    curve round the axis.  A surface's mean turn lies that near such a rational once in a hundred by chance, and those
    of the 61 to 120 islands most of them are near would read as a dotted curve."""
    from fractions import Fraction
    f = Fraction(float(rho)).limit_denominator(qmax)
    return abs(float(f) - rho) < max(1e-7, min(3.0 * spread, 1e-6))


def _pixels(R, Z, lost, seeds, axis, answer, held, planes):
    """Pixels a metre is in the picture plot draws of these crossings (torux.plot.poincare_windows), at 838 pixels
    across, with the answer's sections and its certificate's marks."""
    from .plot import _surface_points, poincare_pixels, poincare_windows
    sections = None
    if answer is not None and getattr(answer, "surface", None) is not None:
        th = np.linspace(0.0, 2 * np.pi, 1025)
        sections = []
        for ph in planes:
            x = _surface_points(answer, th, np.full_like(th, ph))
            sections.append(np.column_stack([np.hypot(x[:, 0], x[:, 1]), x[:, 2]]))
    held = held or {}
    marks = next(([np.asarray(m, float)[np.all(np.isfinite(m), -1)] for m in held[k]] for k in ("elliptic",
                  "hyperbolic") if held.get(k) is not None), None)
    Rr, Zr = poincare_windows(R, Z, lost, seeds, axis, sections, marks, held.get("barrier"))
    return poincare_pixels(Rr, Zr)


def _kind(o, i, period, px):
    """What line i is so far: 'left', 'none' (its transform has not settled), 'island', 'closed' (a surface whose
    crossings close its curve to CLOSE_PX) or 'open' (a surface whose crossings do not yet)."""
    if o["lost"][i]:
        return "left"
    iota, spread = _iota(o["wind"][i:i + 1], period, spread=True)
    if not np.isfinite(iota[0]):
        return "none"
    rho = iota[0] * period / (2 * np.pi)
    if _island(rho, spread[0] / (2 * np.pi)):
        return "island"
    return "closed" if _gaps(o, i, rho, px) <= CLOSE_PX else "open"


def _follow(tr, o, sc, nper, phi0, period, px, src, per_period):
    """Lines whose rotational transform has not settled, or whose crossings leave a gap wider than CLOSE_PX along
    the curve of their surface, are followed on, doubling their length up to FOLLOW_CAP times nper, until it settles
    and the curve closes.  A line followed on that leaves is one that left.  Returns the lines' data, now as long as
    the longest, NaN past each line's own length, and which lines lie on a surface whose curve is still open.
    per_period: seconds a period of every line took, for whether to say first how far they may go."""
    L, P = o["R"].shape[:2]
    todo = [i for i in range(L) if _kind(o, i, period, px) in ("none", "open")]
    n = nper
    if todo:
        import numba
        threads = max(1, numba.get_num_threads())
        est = per_period * math.ceil(len(todo) / threads) / math.ceil(L / threads) * nper
        if est > SAY_AFTER:             # the first doubling alone would be long: how far it may go, not how long
            from .ask import _say
            _say(f"{getattr(src, 'id', None) or 'the source'}: following {len(todo)} of the lines on, up to "
                 f"{FOLLOW_CAP * nper} field periods, until their rotational transforms settle and their curves "
                 f"close")
    while todo and n < FOLLOW_CAP * nper:
        more = min(n, FOLLOW_CAP * nper - n)
        idx = np.array(todo)
        theta = o["wind"][idx, n]
        r = tr.run(o["y"][idx], sc, more, phi0, theta0=theta)
        grow = {}
        for k in ("R", "Z"):
            a = np.full((L, P, n + more), np.nan)
            a[:, :, :o[k].shape[2]] = o[k]
            a[idx, :, n:] = r[k]
            grow[k] = a
        w = np.full((L, n + more + 1), np.nan)
        w[:, :o["wind"].shape[1]] = o["wind"]
        w[idx, n + 1:] = r["wind"][:, 1:]
        grow["wind"] = w
        o = dict(o, **grow)
        o["periods"] = o["periods"].copy()
        o["periods"][idx] = n + r["periods"]
        o["lost"] = o["lost"].copy()
        o["lost"][idx] = r["lost"]
        o["stop"] = o["stop"].copy()
        o["stop"][idx] = np.where(r["lost"][:, None], r["stop"] + [0.0, n * period, 0.0], np.nan)
        o["y"] = o["y"].copy()
        o["y"][idx] = r["y"]
        n += more
        todo = [i for i in todo if _kind(o, i, period, px) in ("none", "open")]
    is_open = np.array([_kind(o, i, period, px) == "open" for i in range(L)], bool)
    return o, is_open


# ------------------------------------------------------------------------------------------------ island chains
def _chain_starts(tr, o, seeds, src, axis, phi0, period, nfp, size, steps):
    """Starts inside the island chains among the lines: for the CHAINS rationals p/q (q <= QMAX islands) between the
    mean turns per period of two neighbouring lines that stayed, the fewest islands first, and no line already goes
    round, an elliptic fixed point of the q-period map by Newton's method from points between the two lines, then
    lines started on the segment from it outward, at the largest distance whose line still turns p/q times a period
    and inside it."""
    from fractions import Fraction
    stayed = np.flatnonzero(~o["lost"] & (o["periods"] > 0))
    n = np.maximum(o["periods"], 1)
    mean = np.array([(o["wind"][i, n[i]] - o["wind"][i, 0]) / n[i] for i in range(len(seeds))]) / (2 * np.pi)
    aR, aZ = (float(np.ravel(v)[0]) for v in axis(np.array([phi0])))
    d = np.hypot(seeds[:, 0] - aR, seeds[:, 1] - aZ)
    ok = stayed[np.argsort(d[stayed])]
    wanted = []
    for i, j in zip(ok[:-1], ok[1:]):
        slack = 2.0 / min(n[i], n[j])
        lo, hi = min(mean[i], mean[j]) - slack, max(mean[i], mean[j]) + slack
        for q in range(1, QMAX + 1):
            for p in range(math.ceil(lo * q), math.floor(hi * q) + 1):
                f = Fraction(p, q)
                if f.denominator == q and all(f != w[0] for w in wanted):
                    wanted.append((f, i, j))
    wanted.sort(key=lambda w: (w[0].denominator, abs(float(w[0]))))
    # a chain a line already goes round (its mean turn the chain's to within what its length can tell) is drawn
    wanted = [w for w in wanted if not np.any(np.abs(mean[stayed] - float(w[0])) < 1.0 / n[stayed])]
    out = []
    for f, i, j in wanted[:CHAINS]:
        found = _elliptic_point(tr, o, i, j, f, mean, axis, phi0, period, steps)
        if found is None or abs(found[1]) < THIN:
            continue                    # no elliptic point, or islands far thinner than a pixel
        out += _island_lines(tr, found[0], o, j, f, axis, phi0, period, steps, size)
    return np.array(out).reshape(-1, 2)


def _curve(o, i, a0):
    """Line i's crossings of the first plane as (angle about a0, distance from a0), sorted by angle."""
    R, Z = o["R"][i, 0], o["Z"][i, 0]
    k = np.isfinite(R)
    th = np.arctan2(Z[k] - a0[1], R[k] - a0[0])
    rr = np.hypot(Z[k] - a0[1], R[k] - a0[0])
    s = np.argsort(th)
    return th[s], rr[s]


def _radius(curve, th):
    t, r = curve
    if len(t) < 8:
        return np.full(np.shape(th), np.nan)
    return np.interp(np.mod(th + np.pi, 2 * np.pi) - np.pi, t, r, period=2 * np.pi)


def _map(tr, y, q, phi0, period, steps):
    from .poincare_trace import Schedule
    o = tr.run(y, Schedule([0.0], period, steps, None, phi0), q, phi0)
    return np.column_stack([o["R"][:, 0, q - 1], o["Z"][:, 0, q - 1]])


def _elliptic_point(tr, o, i, j, f, mean, axis, phi0, period, steps, guesses=4, its=6):
    """A fixed point of the q-period map between lines i and j with |trace| < 2, by Newton's method from points
    between the two lines (their crossings interpolated at the share where p/q lies), and its residue (2 - trace) / 4,
    or None."""
    q = f.denominator
    aR, aZ = (float(np.ravel(v)[0]) for v in axis(np.array([phi0])))
    ci, cj = _curve(o, i, (aR, aZ)), _curve(o, j, (aR, aZ))
    th = 2 * np.pi * (np.arange(guesses) + 0.5) / guesses
    ri, rj = _radius(ci, th), _radius(cj, th)
    rho = mean[[i, j]]
    w = float(np.clip((float(f) - rho[0]) / (rho[1] - rho[0]), 0.1, 0.9)) if rho[1] != rho[0] else 0.5
    rr = ri + w * (rj - ri)
    x = np.column_stack([aR + rr * np.cos(th), aZ + rr * np.sin(th)])
    x = x[np.all(np.isfinite(x), -1)]
    scale = float(np.nanmax(rj)) if np.isfinite(rj).any() else 0.1
    h = 1e-7 * scale
    for _ in range(its):
        if not len(x):
            return None
        y = np.concatenate([x, x + [h, 0.0], x + [0.0, h]])
        m = _map(tr, y, q, phi0, period, steps).reshape(3, len(x), 2)
        F = m[0] - x
        J = np.stack([(m[1] - m[0]) / h, (m[2] - m[0]) / h], -1)          # J[:, :, c] = d map / d x_c
        good = np.all(np.isfinite(F), -1) & np.all(np.isfinite(J.reshape(len(x), -1)), -1)
        x, F, J = x[good], F[good], J[good]
        if not len(x):
            return None
        A = J - np.eye(2)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        dx = np.column_stack([A[:, 1, 1] * F[:, 0] - A[:, 0, 1] * F[:, 1],
                              -A[:, 1, 0] * F[:, 0] + A[:, 0, 0] * F[:, 1]]) / np.where(det == 0, np.nan, det)[:, None]
        n = np.linalg.norm(dx, axis=-1)
        dx = np.where((n > 0.2 * scale)[:, None], dx * (0.2 * scale / np.maximum(n, 1e-300))[:, None], dx)
        x = x - dx
        tr_ = J[:, 0, 0] + J[:, 1, 1]
        done = np.isfinite(n) & (n < 1e-11 * scale) & (np.abs(tr_) < 2)
        if done.any():
            k = np.flatnonzero(done)[0]
            return x[k], float((2.0 - tr_[k]) / 4.0)
        x = x[np.all(np.isfinite(x), -1)]
    return None


def _island_lines(tr, O, o, j, f, axis, phi0, period, steps, size, tries=6, look=80):
    """Starts round the elliptic point O of the chain p/q: on the segment from O away from the axis, at half and nine
    tenths of the largest distance (of 2^-k of the way to line j, k = 1 .. tries) whose line turns p/q times a period
    over ``look`` periods.  Nothing when no such line is wider than a thousandth of the source's size."""
    aR, aZ = (float(np.ravel(v)[0]) for v in axis(np.array([phi0])))
    th = math.atan2(O[1] - aZ, O[0] - aR)
    rO = math.hypot(O[1] - aZ, O[0] - aR)
    rj = float(_radius(_curve(o, j, (aR, aZ)), np.array([th]))[0])
    if not np.isfinite(rj) or rj <= rO:
        return []
    u = np.array([math.cos(th), math.sin(th)])
    t = (rj - rO) * 0.5 ** np.arange(1, tries + 1)
    y = O[None, :] + t[:, None] * u
    from .poincare_trace import Schedule
    sc = Schedule([0.0], period, steps, axis, phi0)
    r = tr.run(y, sc, look, phi0)
    rho = np.array([_birkhoff(np.diff(w)) if np.all(np.isfinite(w)) else np.nan for w in r["wind"]]) / (2 * np.pi)
    inside = np.isfinite(rho) & (np.abs(rho - float(f)) < 1e-6) & ~r["lost"]
    if not inside.any():
        return []
    w = float(t[np.flatnonzero(inside)[0]])
    if w < 1e-3 * size:
        return []
    return [O + 0.5 * w * u, O + 0.9 * w * u]


# ------------------------------------------------------------------------------------------------ rotational transform
def _iota(wind, period, spread=False):
    """The rotational transform of each line from its angle about the axis at the end of each field period: the
    weighted Birkhoff average of the turn in each period over the whole line.  NaN where it has not settled: where
    the average over the first half differs from it by more than IOTA_TOL (a line on a surface or an island converges,
    one that wanders does not), or where the line has fewer than 64 periods.  spread: also that difference."""
    wind = np.asarray(wind, float)
    out = np.full(len(wind), np.nan)
    gap = np.full(len(wind), np.nan)
    for i, w in enumerate(wind):
        k = np.isfinite(w)
        n = int(np.argmin(k)) if not k.all() else len(w)
        d = np.diff(w[:n])
        if len(d) < 64:
            continue
        whole = _birkhoff(d)
        gap[i] = abs(_birkhoff(d[:len(d) // 2]) - whole)
        if gap[i] < IOTA_TOL:
            out[i] = whole / period
    return (out, gap) if spread else out


def _decimals(spread, period):
    """The decimals a rotational transform has settled to: those whose unit is no smaller than the difference
    between the Birkhoff averages over the first half of the line and over all of it (spread, rad a period, here in
    the transform's own units), at most 13 (rounding's)."""
    s = float(spread) / period
    if not np.isfinite(s):
        return 13
    return int(min(13, max(0, math.floor(-math.log10(max(s, 1e-13))))))


def _birkhoff(d):
    t = (np.arange(len(d)) + 0.5) / len(d)
    w = np.exp(-1.0 / (t * (1.0 - t)))
    return float(np.sum(w * d) / np.sum(w))
