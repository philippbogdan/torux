"""The no-axis stage of the axis path: what a source has when torux.analysis.find_axes finds no elliptic period-one line.

On the axis path (torux.analysis.analyse) a source without an elliptic stellarator-symmetric period-one line would
end as 'no_axis', a non-answer.  This stage looks for what can still hold flux surfaces, in this order, each step
only while the budget lasts:

1. Closed lines the axis search misses.  Symmetric closed lines of the q-period map for every q dividing nfp
   (q = 1: the magnetic axis; q > 1: closed lines that go once around the machine -- fixed points of the full-turn
   map P^nfp -- with nfp / q field periods of their own), through the symmetry line Z = 0 of the plane phi = 0: the
   scan of that line (the design boundary's section +- one minor radius densely, and for q = 1 the coils' 0.05-1.5
   Rm on another grid; the tracing engine), every sign change of Z after q / 2 periods solved by the spectral Newton
   of torux.axis with nfp / q periods (from a circle, and from the traced half-orbit where the circle fails or lands on
   another line), its monodromy spectral.  Newton also from the design boundary's m = 0 row.  An elliptic line counts
   only if lines next to it survive (confined_near: an elliptic line the field puts at a B_phi sign change next to a
   coil has none).  An elliptic q = 1 line goes back to analyse() (the ordinary family).
2. Full-turn axes (q > 1, q | nfp, elliptic): their own family of tori (torux.family.Continuation with nfp / q
   periods, phi0 = 0) -- genuine toroidal flux surfaces (one closed curve in every plane phi = const, going once
   around), with nfp / q-fold symmetry instead of nfp-fold -- run to the request or, without one, outward (a range
   lower bound), refined and accepted as every surface (torux.seeds.accept: off-grid B.n at the arithmetic limit, flux,
   resonance guard, quadrature, simple sections around the line, the independent reference check).
3. The request from the design boundary (CoilStellaration sources carry theirs; the request is the coil field's
   flux through it at phi = 0): the boundary's field-line-angle torus (torux.design.sfl_guess_from_surface) and the
   torus Newton at the request from it, then torux.seeds.seed_and_walk (the invariant circle at the request by Newton
   from the base's section; regular orbits located around the request on the outboard symmetry line, their circles
   and tori walked to the request) -- only where the lines seed_and_walk calibrates on survive a period (its step
   ladder does not end on lost lines) -- with the index +1 line inside the boundary as the reference line (an
   inverse-hyperbolic period-one line: rotation ~ 1/2 per period, tori outside its period-two chain), else the
   boundary's geometric axis; a torus is returned only after the acceptance of torux.seeds.accept again with the
   period-one line inside it (Newton from the torus's own m = 0 curve) as the axis of the topology check.
4. A survey: lines on the symmetry line across the section and next to the inverse-hyperbolic lines inside it,
   traced ~100 periods (torux.seeds.SectionTracer; each line with its own B_phi sign; lines lost within one period
   dropped first) and classified (regular / island / chaotic / lost).
5. Tori from the survey's regular orbits (their invariant circles, the torus at fixed transform, Newton): walked to
   the request (torux.seeds.seed_and_walk with that torus as its base) or, without a request, refined and accepted at
   their own flux (a verified lower bound on the range).
No 'no surface' claim is made here: that needs a certificate, not the absence of regular orbits.

What the symmetric-line scan says (the lemma behind the record's 'intervals').  Let Gamma be a stellarator-symmetric
invariant torus (covered, as in torux.barrier: no coil inside, B_phi of one sign on it, one closed curve gamma in the
plane phi = 0) on which the period map P has a non-integer rotation number, D the disk gamma bounds.  The reversors
I0 = S0 ((R, Z) -> (R, -Z)) and I1 = P S0 are orientation-reversing involutions of the closed disk (P = I1 I0,
S0 P S0 = P^-1), so each fixes an arc across it with two ends on gamma (Kerekjarto: such an involution of the disk is
conjugate to a reflection); on gamma the two pairs of ends interleave, since if they did not, I0 and I1 would agree
at a point of gamma and P = I1 I0 would have a fixed point there (rotation number an integer).  Interleaved ends
make the arcs cross inside D: a point fixed by I0 and I1, i.e. a point x of the symmetry line Z = 0 in D with
P(x) = x -- a symmetric period-one line, a root of the scan's Z(pi / nfp).  And D's points are not lost (Lemma L of
torux.edgebound).  So such a torus meets the symmetry line in an interval of surviving points holding a root: a scan
interval of surviving lines without a sign change -- were it certified (validated enclosures of the half-period
map and of lost segments, as torux.edgebound's segment certificates) -- carries no stellarator-symmetric torus of
non-integer rotation.  The record gives the evidence (sampled, not certified).

Entry point: noaxis_stage(eng, source, flux, axes, deadline=..., quick=...) -> dict with
    'kind': 'axis'      (an elliptic period-one line: rec['axis'] is the torux.axis.Axis for analyse())
            'surface'   (rec['surface'] / rec['guard'] in torux.analysis's record form; rec['surface']['nfp'] may be
                         nfp / q for a full-turn axis's torus)
            'range'     (rec['range']: a verified lower bound on the flux the source holds)
            'none'      (nothing found; rec['evidence'], rec['lines'], rec['scans'])
    and 'search_complete' (False when the deadline cut the closed-line search: the caller reports 'undetermined').
"""
from __future__ import annotations

import json as _json
import math
import os as _os
import time
import traceback

import numpy as np

from . import axis as AX
from . import budget as _budget
from .envfail import environment_failure

# The stage's switches (TORUX_NOAXIS='{"on": true}' JSON overrides entries; analyse(noaxis={...}) too)
NOAXIS = dict(
    on=False,             # off by default (it costs a median 18.5 s and seldom answers); TORUX_NOAXIS='{"on": true}'
                          # runs it
    qs="divisors",        # the periods of the closed lines searched: every divisor of nfp (full path)
    qs_quick=(1, 2),      # quick mode: q = 1, and q = 2 where it divides nfp (the q = 3, 5 scans of nfp = 3, 5 sources
                          # cost ~50 % of the line search and seldom answer)
    nscan=96,             # lines of the symmetric-line scan over the section (+ 56 over the coils' 0.05-1.5 Rm for q = 1)
    seeds=True,           # step 3 (the request from the design boundary)
    seeds_share=0.6,      # its share of what is left of the budget
    family_share=0.5,     # a full-turn axis's family: its share of what is left
    survey=True,          # step 4 (evidence)
    survey_lines=8,       # lines across the section (+ 8 next to each index +1 line inside it)
    survey_periods=128,
    survey_share=0.6,     # the survey's share of what is left (the rest: the caller's records and verification)
    orbit_seeds=True,     # step 5 (tori from the survey's regular orbits)
    survey_quick_all=False,  # quick mode: the survey for every source (else only without a request or with an
                             # inverse-hyperbolic line inside the section)
    min_left=20.0,        # seconds a step needs to start
)
if _os.environ.get("TORUX_NOAXIS"):
    NOAXIS.update(_json.loads(_os.environ["TORUX_NOAXIS"]))


# ============================================================================ design boundary and geometry
class _Skip(Exception):
    pass


def design_boundary(source):
    """The design boundary given with a source (Source.design_boundary, a surface of torux.io.surfaces), or None."""
    return getattr(source, "design_boundary", None)


def section_crossings(bnd, phi, nq=2048):
    """(R_out, R_in) of the boundary's section at cylindrical angle phi (rad): its crossings of Z = 0."""
    th = np.arange(nq) / nq
    r, _, _ = bnd.gamma_and_derivs(np.full(nq, phi / (2 * np.pi)), th)
    R = np.hypot(r[:, 0], r[:, 1])
    Z = r[:, 2]
    s = np.sign(Z)
    Rc = []
    for i in np.flatnonzero(s != np.roll(s, -1)):
        j = (i + 1) % nq
        if Z[i] == Z[j]:
            continue
        a = Z[i] / (Z[i] - Z[j])
        Rc.append(R[i] + a * (R[j] - R[i]))
    return (max(Rc) if Rc else float(R.max())), (min(Rc) if Rc else float(R.min()))


def geometry(source, bnd):
    """The region the stage looks in, at phi = 0: dict(R_in, R_out, a, R_c, axis=(rc, zs) of a reference curve).
    From the design boundary when there is one (its section and its m = 0 row), else from the coils (the largest
    circle about the median coil radius free of coil points in (R, Z))."""
    nfp = int(source.nfp)
    pts = source.sample_points()
    Rm = float(np.median(np.hypot(pts[:, 0], pts[:, 1]))) if len(pts) else 1.0
    if bnd is not None:
        Ro, Ri = section_crossings(bnd, 0.0)
        rc, zs = np.array([0.5 * (Ro + Ri)]), np.array([0.0])
        g = dict(R_in=Ri, R_out=Ro, source="design boundary")
    else:
        Rp, Zp = np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2]
        a_c = float(np.min(np.hypot(Rp - Rm, Zp))) if len(pts) else 0.3 * Rm
        g = dict(R_in=Rm - 0.9 * a_c, R_out=Rm + 0.9 * a_c, source="coils' hole")
        rc, zs = np.array([Rm]), np.array([0.0])
    g["a"] = 0.5 * (g["R_out"] - g["R_in"])
    g["R_c"] = 0.5 * (g["R_out"] + g["R_in"])
    g["Rm"] = Rm
    g["axis"] = (np.asarray(rc, float), np.asarray(zs, float))
    g["nfp"] = nfp
    return g


def alive_intervals(R, ok, Zend):
    """The runs of consecutive scan lines that survive the half-orbit: [R_first, R_last, lines, sign changes of Z in
    the run].  A stellarator-symmetric invariant circle of P with non-integer rotation per period meets the symmetry
    line in an interval of surviving points that holds a symmetric period-one point (the fixed point sets of the
    reversors S0 and P S0 are arcs across its disk whose ends interleave on the circle, so they cross there): a run
    without a sign change -- on a scan fine enough -- has none (evidence, not a certificate: the scan samples)."""
    out = []
    i = 0
    n = len(R)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        z = np.sign(Zend[i:j + 1])
        out.append([float(R[i]), float(R[j]), int(j - i + 1), int(np.sum(z[1:] != z[:-1]))])
        i = j + 1
    return out


def divisors(n):
    return [q for q in range(1, int(n) + 1) if int(n) % q == 0]


def lost_within(eng, nfp, R0s, sign, nper=1, steps=48):
    """Lines from (R_i, phi = 0, Z = 0) traced nper periods in the direction of B_phi = sign (torux.fasttrace.LineTracer,
    order 12, `steps` macro steps per period): the boolean mask of the lines stopped (B_phi of the other sign, or
    |B_phi| < 1e-3 |B|).  torux.seeds.SectionTracer.calibrate climbs its step ladder without end on lost lines (every
    pair differs by inf): its calibration lines are checked here first."""
    from .fasttrace import LineTracer
    R0s = np.asarray(R0s, float)
    tr = LineTracer(eng, nfp, k=6, steps_per_period=steps, sign=sign)
    o = tr.trace(np.stack([R0s, 0 * R0s], -1), 0.0, nper * 2 * np.pi / nfp, nsteps=steps * nper)
    return np.asarray(o["lost"]) | ~np.all(np.isfinite(o["y"]), -1)


# ============================================================================ symmetric closed lines of P^q
def line_record(ax, q, how, pts=None):
    R0 = float(ax.at(np.array([0.0]))[0][0])
    rec = dict(q=int(q), R=R0, residue=float(ax.residue), iota=float(ax.iota), N=int(ax.N), res=float(ax.res),
               how=how, kind=kind_of(ax.residue))
    if pts is not None and len(pts):
        rec["d_coil"] = line_coil_distance(ax, pts)
    return rec


def line_coil_distance(ax, pts, n=256):
    """Smallest distance from the closed line (n points over its own period) to the coil sample points."""
    ph = np.linspace(0, 2 * np.pi / ax.nfp, n, endpoint=False)
    R, Z = ax.at(ph)
    X = np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(pts, float)).query(X)
    return float(np.min(d))


def confined_near(eng, nfp, ax, a, nper=20, offsets=(0.01, 0.03)):
    """Do lines next to a closed line stay?  Lines from its point at phi = 0 offset by +-offsets x a along Z = 0,
    traced nper field periods (torux.fasttrace.LineTracer in its B_phi direction; eng: the tracing engine): the fraction
    that survive.  An
    elliptic line the field model puts within a hair of a B_phi sign change (next to a coil) has none (CoilStellaration
    DizUwnvg: residue 0.29, lines 4e-4 m from it lost in 2 periods), and no torus to start a family from."""
    from .seeds import bphi_sign
    R0, Z0 = ax.at(np.array([0.0]))
    dR = np.concatenate([-np.asarray(offsets), np.asarray(offsets)]) * a
    lost = lost_within(eng, nfp, R0[0] + dR, bphi_sign(eng, ax), nper=nper, steps=24)
    return float(np.mean(~lost))


def kind_of(residue):
    if not np.isfinite(residue):
        return "unknown"
    if residue < 0:
        return "hyperbolic"
    if residue < 1:
        return "elliptic"
    return "inverse_hyperbolic"


def resolve(eng, ax, nfp_q, Ns=(32, 48, 64, 96, 128)):
    """The closed line resolved to its own spectrum (as torux.analysis.find_axes does), monodromy spectral."""
    for N2 in Ns:
        tail = max(np.max(np.abs(ax.rc[-3:])), np.max(np.abs(ax.zs[-3:]))) / max(abs(ax.rc[0]), 1e-300)
        if tail < 1e-13:
            break
        if ax.N >= N2:
            continue
        ax2 = AX.solve_axis(eng, nfp_q, float(ax.rc[0]), N=N2, guess=(ax.rc, ax.zs))
        if not ax2.converged:
            break
        ax = ax2
    AX.spectral_axis_monodromy(eng, ax)
    return ax


def traced_root(eng, nfp, q, R_lo, R_hi, sign, steps=48, tol=1e-11, maxit=40):
    """The symmetric closed line of the q-period map through the scan's bracket [R_lo, R_hi] on Z = 0 at phi = 0, by
    the Illinois (regula falsi) iteration on Z after q / 2 periods (torux.fasttrace.LineTracer, order 12, `steps` macro
    steps a period), and Greene's residue of P^q from a centred finite-difference Jacobian of the traced q-period
    map.  For brackets whose spectral Newton fails or lands on another line (CoilStellaration DKKBkAr9: the line
    through R = 0.8145, residue 0.136, only this way).  Returns dict(R, residue, closure) or None."""
    from .fasttrace import LineTracer
    T = 2 * np.pi / nfp
    tr = LineTracer(eng, nfp, k=6, steps_per_period=steps, sign=sign)

    def zend(R):
        o = tr.trace(np.array([[R, 0.0]]), 0.0, q * T / 2, nsteps=max(1, steps * q // 2))
        return np.nan if o["lost"][0] else float(o["y"][0, 1])
    a, b = float(R_lo), float(R_hi)
    fa, fb = zend(a), zend(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or np.sign(fa) == np.sign(fb):
        return None
    side = 0
    c = a
    for _ in range(maxit):
        c = (a * fb - b * fa) / (fb - fa)
        fc = zend(c)
        if not np.isfinite(fc):
            return None
        if abs(fc) < tol * max(1.0, abs(c)) or abs(b - a) < 1e-13 * abs(c):
            break
        if np.sign(fc) == np.sign(fb):
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
    h = 1e-6 * max(1.0, abs(c))
    pts = np.array([[c, 0.0], [c + h, 0.0], [c - h, 0.0], [c, h], [c, -h]])
    o = tr.trace(pts, 0.0, q * T, nsteps=steps * q)
    if np.any(o["lost"]):
        return None
    y = o["y"]
    J = np.stack([(y[1] - y[2]) / (2 * h), (y[3] - y[4]) / (2 * h)], 1)
    return dict(R=float(c), residue=float((2 - np.trace(J)) / 4), closure=float(np.max(np.abs(y[0] - pts[0]))))


def same_line(a, b, nfp_q, tol=1e-8):
    phs = np.linspace(0, 2 * np.pi / nfp_q, 33)[:-1]
    Ra, Za = a.at(phs)
    Rb, Zb = b.at(phs)
    return np.max(np.abs(Ra - Rb)) < tol and np.max(np.abs(Za - Zb)) < tol


def closed_lines(eng, source, geo, qs, nscan=160, deadline=None, known=(), verbose=False):
    """Symmetric closed lines of the q-period map through Z = 0 at phi = 0, q in qs, found by a scan of that line
    over the section (+- a) and the coils' 0.05-1.5 Rm (sign changes of Z after q / 2 periods), each solved by the
    spectral Newton with nfp / q periods from a circle and from the traced half-orbit.  q = 1 also from the
    reference curve (the boundary's m = 0 row).  Returns ([(record, Axis)] (Axis.nfp = nfp / q), scans, complete):
    complete is False when the deadline cut the search (a q not scanned, a root not solved)."""
    nfp = int(source.nfp)
    pts = source.sample_points()
    out = []
    for a in known:
        out.append((line_record(a, 1, "find_axes", pts=pts), a))
    Rm = geo["Rm"]
    lo = max(0.05 * Rm, geo["R_in"] - geo["a"])
    hi = geo["R_out"] + geo["a"]
    R_sec = np.linspace(lo, hi, nscan)
    # the section densely, and the coils' range (find_axes's 0.05-1.5 Rm) on another grid: a root find_axes's grid
    # brackets but whose circle Newton fails or lands elsewhere is bracketed again at another point (CoilStellaration
    # DcckqeUg: an elliptic line through R = 1.865 from the root at 1.223 of a 480-line scan, none from 96 / 192)
    R_all = np.unique(np.concatenate([R_sec, np.linspace(0.05 * Rm, 1.5 * Rm, 56)]))
    # the scans locate sign changes only: on the tracing engine (quadrature for 1e-12 instead of 1e-17, fewer nodes);
    # every root is solved on the solver's engine
    from .seeds import trace_engine
    teng = trace_engine(eng, source)

    def known_point(ax, q):
        """A line already found: the same q and curve, or (q > 1) a line of a period dividing q through the same point
        (a fixed point of P is one of P^q)."""
        R0, Z0 = ax.at(np.array([0.0]))
        for r, b in out:
            if r["q"] == q and same_line(ax, b, nfp / q, tol=1e-7 * Rm):
                return True
            if q % r["q"] == 0 and q != r["q"]:
                # (1e-5 Rm: the q-period Newton's copy of a period-one line sat 4e-6 m from it on CoilStellaration
                # D6764kbr, its residue that of M^3; distinct lines of different periods lie much further apart)
                Rb, Zb = b.at(np.array([0.0]))
                if abs(Rb[0] - R0[0]) < 1e-5 * Rm and abs(Zb[0] - Z0[0]) < 1e-5 * Rm:
                    return True
        return False

    def add(ax, q, how):
        if not ax.converged:
            return None
        nq = nfp / q
        if known_point(ax, q):
            return None
        try:
            ax = resolve(eng, ax, nq)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            return None
        if not np.isfinite(ax.residue) or known_point(ax, q):
            return None
        rec = line_record(ax, q, how, pts=pts)
        if 0 < ax.residue < 1:
            rec["confined_near"] = confined_near(teng, nfp, ax, geo["a"])
        out.append((rec, ax))
        if verbose:
            print(f"   [noaxis] line q={q} R(0) {rec['R']:.8f} residue {rec['residue']:.6g} ({how})", flush=True)
        return ax

    # the reference curve (design boundary's m = 0 row) as a Newton start for the period-one line
    rc, zs = geo["axis"]
    try:
        if len(rc) > 1:
            add(AX.solve_axis(eng, nfp, float(rc[0]), N=max(16, len(rc) - 1), guess=(rc, zs)), 1, "boundary m=0 row")
        R0g = float(np.sum(rc))
        g = AX.traced_axis_guess(eng, nfp, R0g)
        if g is not None:
            add(AX.solve_axis(eng, nfp, R0g, N=16, guess=g), 1, "traced from the boundary's axis point")
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
    scans = {}
    complete = True
    traced_only = []
    for q in qs:
        if deadline is not None and time.time() > deadline:
            complete = False
            break
        nq = nfp / q
        Rs = R_all if q == 1 else R_sec
        # Z at phi = pi / nq (q / 2 periods) of lines from (R, 0, 0), each with its own B_phi sign (torux.axis's host
        # tracer; the jitted scan on engines without CPU kernels)
        if AX._host(teng):
            end, _ = AX._trace_half_period(teng, nq, Rs, 12 * q, 6)
            Zend, Rs_q = end[:, 1], Rs
        else:
            Rs_q, Zend, _ = AX.symmetric_line_scan(teng, nq, float(Rs[0]), float(Rs[-1]), nscan=len(Rs), nsteps=12 * q)
        ok = np.isfinite(Zend)
        # near misses: a local minimum of |Z| well below its neighbours without a sign change is where an elliptic line
        # and its hyperbolic partner can share one grid cell (CoilStellaration DLh5syBM: 1.6 mm apart, the section
        # grid 6.6 mm): the two cells around it scanned again 8 times finer
        extra = []
        for i in range(1, len(Rs_q) - 1):
            if not (ok[i - 1] and ok[i] and ok[i + 1]):
                continue
            z0, zl, zr = abs(Zend[i]), abs(Zend[i - 1]), abs(Zend[i + 1])
            if (np.sign(Zend[i - 1]) == np.sign(Zend[i]) == np.sign(Zend[i + 1]) and z0 < zl and z0 < zr
                    and z0 < 0.25 * min(zl, zr)):
                extra.append(np.linspace(Rs_q[i - 1], Rs_q[i + 1], 17)[1:-1])
        if extra:
            Rx = np.concatenate(extra)
            if AX._host(teng):
                ex, _ = AX._trace_half_period(teng, nq, Rx, 12 * q, 6)
                Zx = ex[:, 1]
            else:
                Zx = np.array([AX.symmetric_line_scan(teng, nq, float(r), float(r), nscan=1, nsteps=12 * q)[1][0]
                               for r in Rx])
            Rs_q = np.concatenate([Rs_q, Rx])
            Zend = np.concatenate([Zend, Zx])
            order = np.argsort(Rs_q, kind="stable")
            Rs_q, Zend = Rs_q[order], Zend[order]
            ok = np.isfinite(Zend)
        roots = []
        for i in range(len(Rs_q) - 1):
            if ok[i] and ok[i + 1] and np.sign(Zend[i]) != np.sign(Zend[i + 1]):
                roots.append(float(Rs_q[i] - Zend[i] * (Rs_q[i + 1] - Rs_q[i]) / (Zend[i + 1] - Zend[i])))
        scans[q] = dict(alive=int(ok.sum()), n=len(Rs_q), roots=roots, refined=int(sum(len(e) for e in extra)),
                        alive_R=[float(Rs_q[ok].min()), float(Rs_q[ok].max())] if ok.any() else None,
                        R_range=[float(Rs_q[0]), float(Rs_q[-1])], intervals=alive_intervals(Rs_q, ok, Zend))
        for R0 in roots:
            if deadline is not None and time.time() > deadline:
                complete = False
                break
            if any(r["q"] == q and abs(r["R"] - R0) < 1e-6 * Rm for r, _ in out):
                continue
            Nq = 16 * q
            # the root's own line: within 1.5 grid spacings of the interpolated root (another line 0.02 Rm away is not
            # it: CoilStellaration DmGrZLqQ's elliptic line at 1.3992 sits 0.017 from a hyperbolic one at 1.3820)
            i_r = int(np.clip(np.searchsorted(Rs_q, R0), 1, len(Rs_q) - 1))
            h_r = float(Rs_q[i_r] - Rs_q[i_r - 1])
            near_root = lambda: any(q % r["q"] == 0 and abs(r["R"] - R0) < max(1.5 * h_r, 1e-6 * Rm) for r, _ in out)
            try:
                ax = AX.solve_axis(eng, nq, R0, N=Nq, guess=(np.array([R0]), np.array([0.0])))
                how = "circle"
                if ax.converged:
                    add(ax, q, how)
                if not near_root():
                    # (a circle Newton that lands on another line: the line through the root from its traced half-orbit)
                    g = AX.traced_axis_guess(eng, nq, R0, N=Nq, nsteps=24 * q)
                    if g is not None:
                        add(AX.solve_axis(eng, nq, R0, N=Nq, guess=g), q, "traced")
                if not near_root():
                    # the spectral Newton does not hold this line: the root and its residue on the traced map (a record
                    # of the line; no Axis for a family)
                    lo_, hi_ = float(Rs_q[i_r - 1]), float(Rs_q[i_r])
                    Bq = np.asarray(eng.evaluate(np.array([[R0, 0.0, 0.0]]), grad=False, A=False)["B"])[0]
                    tr_ = traced_root(eng, nfp, q, lo_, hi_, 1.0 if Bq[1] > 0 else -1.0)
                    if tr_ is not None:
                        rec_t = dict(q=int(q), R=tr_["R"], residue=tr_["residue"], iota=float("nan"), N=0,
                                     res=tr_["closure"], how="traced map (no spectral line)", kind=kind_of(tr_["residue"]),
                                     spectral=False)
                        if 0 < tr_["residue"] < 1:
                            rec_t["confined_near"] = confined_near(teng, nfp, AX.Axis(np.array([tr_["R"]]),
                                                                                      np.array([0.0]), nq), geo["a"])
                        traced_only.append(rec_t)
            except Exception as e:  # noqa: BLE001
                if environment_failure(e):
                    raise
    for r in traced_only:
        scans.setdefault(r["q"], {}).setdefault("traced_only", []).append(r)
    return out, scans, complete


# ============================================================================ tori around a line
def line_inside(eng, torus, nfp_q):
    """The symmetric closed line inside a torus (Newton from the torus's m = 0 curve, with nfp_q periods), as the
    axis of the topology check: (Axis or None)."""
    N = torus.N
    rc0 = np.asarray(torus.rc[0, N:])
    zs0 = -np.asarray(torus.zs[0, N:])
    Nax = max(16, min(64, len(rc0) - 1))
    g = (rc0[:Nax + 1], np.concatenate([[0.0], zs0[1:Nax + 1]]))
    try:
        ax = AX.solve_axis(eng, nfp_q, float(rc0[0]), N=Nax, guess=g)
        if not ax.converged:
            R0 = float(np.sum(rc0))
            gg = AX.traced_axis_guess(eng, nfp_q, R0, N=Nax)
            if gg is not None:
                ax = AX.solve_axis(eng, nfp_q, R0, N=Nax, guess=gg)
        if not ax.converged:
            return None
        return resolve(eng, ax, nfp_q)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        return None


def family_on_line(eng, source, line, q, flux, deadline, quick=True, tol=3e-15, dmin=None, verbose=False):
    """The family of tori around an elliptic closed line of the q-period map (q | nfp: nfp / q periods), to the
    request (|flux|) or, without one, outward to 4 x the flux of a disc reaching the nearest coil.  Refinement and the
    acceptance of torux.seeds.accept (the reference check included).  Returns (kind, rec): kind 'surface' / 'range'
    with the surface record (nfp, q in it), or 'undetermined' with the family's profile."""
    from .family import Continuation, refine_to_limit
    from .analysis import torus_record, lean_continuation
    from .seeds import accept, linear_shear, shear_estimate
    nfp_q = source.nfp / q
    rec = dict(q=int(q), nfp_q=nfp_q, line_R=float(line.at(np.array([0.0]))[0][0]), line_residue=float(line.residue),
               line_iota=float(line.iota))
    pts = source.sample_points()
    cont = Continuation(eng, nfp_q, M0=6, N0=max(16, min(line.N, 64)), verbose=verbose)
    if flux is not None:
        target = abs(float(flux))
    else:
        R_ax, Z_ax = line.at(np.array([0.0]))
        pa = np.array([R_ax[0], 0.0, Z_ax[0]])
        Bax = float(np.linalg.norm(np.asarray(eng.evaluate(pa[None, :], grad=False, A=False)["B"])[0]))
        d_coil = float(np.min(np.linalg.norm(pts - pa[None, :], axis=1))) if len(pts) else 0.3 * R_ax[0]
        target = 4.0 * Bax * math.pi * d_coil ** 2
    rec["target"] = target
    fam_deadline = deadline if flux is not None or deadline is None else time.time() + 0.6 * (deadline - time.time())
    fam = cont.run(line, target, deadline=fam_deadline, **({"ds_min": 0.03} if quick else {}))
    tori = list(fam.tori)
    rec["family_stop"] = fam.stop_reason
    rec["profile"] = [torus_record(t) for t in tori]
    if not tori:
        return "undetermined", rec

    def shear_at(t):
        sh = shear_estimate(tori, abs(t.flux)) if len(tori) >= 2 else np.nan
        if not np.isfinite(sh):
            try:
                sh = linear_shear(cont, t) * math.copysign(1.0, t.flux)
            except Exception:  # noqa: BLE001
                sh = np.nan
        return sh

    if flux is not None:
        reached = [t for t in tori if abs(abs(t.flux) - target) <= 1e-12 * target]
        if not reached:
            return "undetermined", rec
        cands = [reached[-1]]
    else:
        cands = sorted([t for t in tori if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-6],
                       key=lambda t: -abs(t.flux))[:3]
    for t in cands:
        if deadline is not None and time.time() > deadline:
            break
        rc_ = lean_continuation(cont, source, t, dmin if dmin is not None else source.meta.get("dmin", 0.05))
        best = refine_to_limit(rc_, t, t.flux, target=tol, deadline=deadline, verbose=verbose,
                               Mcap=320 if quick else 512, Ncap=320 if quick else 512)
        ok, acc = accept(eng, source, line, best, abs(best.flux), shear_at(best), refine_engine=rc_.engine,
                         verbose=verbose)
        rec.setdefault("tried", []).append(dict(flux=float(best.flux), M=int(best.M), N=int(best.N),
                                                bn_off=float(best.bn_off), ok=bool(ok)))
        if ok:
            s = torus_record(best, full=True)
            s.update(nfp=nfp_q, q=int(q), quad_err_half=acc["quad_err_half"], topology=acc["topology"],
                     reference=acc.get("verify"), phi0=0.0)
            rec["surface"] = s
            rec["guard"] = acc["guard"]
            return ("surface" if flux is not None else "range"), rec
    return "undetermined", rec


# ============================================================================ the request from the design boundary
def request_from_boundary(eng, source, bnd, request, ref_axis, deadline, tol=3e-15, dmin=None, verbose=False,
                          quick=True):
    """torux.seeds.seed_and_walk at |request| with the design boundary's field-line-angle torus as its base and
    ref_axis (a torux.axis.Axis: the index +1 line inside the boundary, else its geometric axis) as the reference line.
    The returned torus (accepted by torux.seeds.accept against ref_axis) is accepted again against the period-one line
    found inside it.  Returns (TorusState or None, log)."""
    from .design import sfl_guess_from_surface
    from .family import Continuation
    from .seeds import seed_and_walk, accept
    log = {}
    t0 = time.time()
    base = None
    for M, N in ((24, 24), (16, 16)):
        try:
            base = sfl_guess_from_surface(bnd, eng, M, N)
            cont = Continuation(eng, source.nfp, M0=6, N0=N)
            base.flux = float(cont.problem(M, N).flux_of(base.rc, base.zs))
            break
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            log.setdefault("base_errors", []).append(repr(e))
            base = None
    if base is None:
        log["outcome"] = "no base"
        return None, log
    log["base"] = dict(M=int(base.M), N=int(base.N), flux=float(base.flux), iota=float(base.iota),
                       **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in base.info.items()})
    aerr = base.info.get("angle_newton_err", np.nan)
    usable = bool(np.isfinite(aerr) and aerr < 1e-6 and np.isfinite(base.flux) and np.all(np.isfinite(base.rc)))
    log["base"]["usable"] = usable
    # (not usable: no field-line angle on the boundary -- B_phi changes sign on it, the fit's f = B^theta / B^phi is
    # singular and the angle map does not converge; seed_and_walk then places the request by the axis field alone)
    # the torus Newton at the request from the base itself first (the CoilStellaration design path), at 16 x 16 and
    # raised only once it converges (a 24 x 24 Newton on a 1e5-node engine is ~10 s an iteration on 2 cores)
    try:
        if not usable:
            raise _Skip()
        cont = Continuation(eng, source.nfp, M0=6, N0=16)
        f_s = math.copysign(abs(request), base.flux)
        b16 = base.resized(16, 16)
        b16.flux = f_s
        sol = cont.solve(b16, f_s, 16, 16, maxit=12)
        log["design_newton"] = dict(conv=bool(sol.converged), res=float(sol.res), bn_off=float(sol.bn_off),
                                    iota=float(sol.iota), its=int(sol.its))
        if sol.converged and np.isfinite(sol.bn_off):
            sol = cont.raise_until(sol, f_s, cont.bn_cont, max_raise=4 if quick else 8, min_gain=3.0 if quick else 1.0,
                                   low_gain_patience=2)
            log["design_newton"].update(M=int(sol.M), N=int(sol.N), bn_raised=float(sol.bn_off))
            if sol.bn_off < 1e-5:
                base = sol          # a converged torus at the request: the seeds' first stage refines it
                log["design_torus"] = True
    except _Skip:
        log["design_newton"] = "skipped (no usable base)"
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        log["design_newton_error"] = repr(e)
    if quick and not log.get("design_torus"):
        # quick mode: seed_and_walk only from a converged design torus at the request (its refinement first).  Its
        # tracing stages seldom find a torus on a source without an axis, and can run 100-650 s past their
        # share (its tracer calibration and orbit traces have no deadline inside): the full path's
        log["outcome"] = "no design torus at the request (quick mode: no tracing)"
        log["wall"] = time.time() - t0
        return None, log
    # seed_and_walk calibrates its tracer on lines at 0.5, 0.8, 1.0, 1.03 of the way from the reference line to the
    # base's outboard point: if one of them is lost within a period the calibration does not end -- and no regular
    # orbit lies there to seed a torus with
    from .seeds import outboard_point, bphi_sign
    Rax = float(ref_axis.at(np.array([0.0]))[0][0])
    Rb = outboard_point(base)[0] if usable else np.nan
    if not (np.isfinite(Rb) and Rb > Rax):
        # (as seed_and_walk places an unusable base: the minor radius from the request and the reference line's field)
        Bax = np.asarray(eng.evaluate(np.array([[Rax, 0.0, float(ref_axis.at(np.array([0.0]))[1][0])]]), grad=False,
                                      A=False)["B"])[0]
        Rb = Rax + math.sqrt(abs(request) / (math.pi * max(abs(Bax[1]), 1e-300)))
    Rcal = Rax + (Rb - Rax) * np.array([0.5, 0.8, 1.0, 1.03])
    # (8 periods: seed_and_walk's tracer calibration and orbit traces have no deadline inside -- on lines that leave
    # within a few periods they can run 100-650 s past the stage's share)
    nper_cal = 8
    lost = lost_within(eng, source.nfp, Rcal, bphi_sign(eng, ref_axis), nper=nper_cal)
    log["calibration_lines"] = dict(R=Rcal.tolist(), lost=lost.tolist(), periods=nper_cal)
    if np.any(lost):
        log["outcome"] = f"lines at the request lost within {nper_cal} period(s)"
        log["wall"] = time.time() - t0
        return None, log
    T, slog = seed_and_walk(eng, source, abs(request), base, ref_axis, deadline=deadline, verbose=verbose, tol=tol,
                            dmin=dmin, trace=not quick)
    log["seeds"] = slog
    log["wall"] = time.time() - t0
    if T is None:
        log["outcome"] = slog.get("outcome", "none")
        return None, log
    # the reference line is not this torus's axis: its own period-one line inside it
    ax_in = line_inside(eng, T, source.nfp)
    log["line_inside"] = None if ax_in is None else line_record(ax_in, 1, "Newton from the torus's m = 0 curve")
    if ax_in is not None:
        from .seeds import shear_estimate
        sh = slog.get("shear_flux", np.nan)
        ok, acc = accept(eng, source, ax_in, T, abs(request), sh if sh is not None else np.nan, verbose=verbose)
        log["accept_line_inside"] = dict(ok=bool(ok), topology=acc["topology"])
        if not ok:
            log["outcome"] = "rejected against the line inside"
            return None, log
    log["outcome"] = "surface"
    return T, log


# ============================================================================ evidence
MAX_STEPS = 64        # the survey's limit on the tracer's macro steps per period (CoilStellaration lines away from coils: 32)


def capped_calibrate(tr, y0, k=6, sp_max=MAX_STEPS, deadline=None):
    """torux.seeds.SectionTracer.calibrate's step-doubling test for one order k on a ladder that ends at sp_max: sets
    tr.sp[(k, level)] for the levels met and returns True when the long level is met, False otherwise (the lines need
    more steps a period than the survey spends -- they pass next to coils).  SectionTracer.calibrate climbs to 2048
    steps a period; on a few lines each step is a small engine call (~ms), so its ladder alone can take 200+ s."""
    y0 = np.asarray(y0, float).reshape(-1, 2)
    Rs = float(np.max(np.abs(y0[:, 0])))
    prev = None
    sp = 8
    while sp <= 2 * sp_max:
        if deadline is not None and time.time() > deadline:
            break
        o = tr._map(y0, k, sp)
        if prev is not None:
            ok = ~(o["lost"] | prev[1])
            d = float(np.max(np.linalg.norm(o["y"][ok] - prev[0][ok], axis=1))) / Rs if np.any(ok) else np.inf
            for lev, tl in tr.tol.items():
                if (k, lev) not in tr.sp and d <= tl:
                    tr.sp[(k, lev)] = prev[2]
            if all((k, lev) in tr.sp for lev in tr.tol):
                break
        prev = (o["y"], o["lost"], sp)
        sp *= 2
    return (k, "long") in tr.sp


def orbits_chunked(tr, y0, nper, deadline, chunk=16):
    """torux.seeds.SectionTracer.orbits in pieces of `chunk` periods, the deadline checked between pieces (orbits checks
    it only between its own 50-period chunks): dict(sec (L, n + 1, 2), dang (L, n), lost (L,), n)."""
    y0 = np.asarray(y0, float).reshape(-1, 2)
    L = len(y0)
    secs, dangs = [y0[:, None, :].copy()], []
    lost = np.zeros(L, bool)
    y = y0.copy()
    n = 0
    while n < nper:
        if deadline is not None and time.time() > deadline:
            break
        m = int(min(chunk, nper - n))
        o = tr.orbits(y, m, level="long")
        sec = np.array(o["sec"][:, 1:, :], copy=True)
        dang = np.array(o["dang"][:, :m], copy=True)
        sec[lost] = np.nan
        dang[lost] = np.nan
        lost |= np.asarray(o["lost"], bool)
        secs.append(sec)
        dangs.append(dang)
        y = np.array(o["sec"][:, -1, :], copy=True)
        y[lost] = y0[lost]                # (a lost line restarts from its own finite start, marked lost)
        n += m
    return dict(sec=np.concatenate(secs, 1), dang=np.concatenate(dangs, 1) if dangs else np.zeros((L, 0)), lost=lost,
                n=n)


def survey(eng, source, geo, ref_axis, nlines=24, nper=100, deadline=None, traces=None, near=()):
    """Lines on Z = 0, phi = 0 across the section (+- 0.3 a) and next to the index +1 lines `near` (R values: 8 lines
    within +- 0.1 a of each), nper periods (torux.seeds.SectionTracer, each line with its own B_phi sign), classified
    by torux.seeds._orbit_record about ref_axis.  Returns the counts and the orbits; the section points of regular
    orbits go to traces[R0] = (sec, sign) when a dict is given.  (CoilStellaration DQnVai: the only regular orbits
    lie within 0.5 a of its inverse-hyperbolic line, between chaotic ones -- invariant circles beyond its period-two
    chain, rotation 0.35-0.41 per period; a uniform row of 24 lines across the section saw none.)"""
    from . import seeds as SD
    R0s = list(np.linspace(geo["R_in"] - 0.3 * geo["a"], geo["R_out"] + 0.3 * geo["a"], nlines))
    for Rl in near:
        R0s += list(float(Rl) + 0.1 * geo["a"] * np.array([-1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0]))
    R0s = np.unique(np.round(np.array(R0s), 12))
    R0s = R0s[R0s > 0.02 * geo["Rm"]]
    Bs = np.asarray(eng.evaluate(np.stack([R0s, 0 * R0s, 0 * R0s], -1), grad=False, A=False)["B"])
    sgn = np.where(Bs[:, 1] > 0, 1.0, -1.0)
    teng = SD.trace_engine(eng, source)
    orbs = [None] * len(R0s)
    for sg in (1.0, -1.0):
        idx = np.flatnonzero(sgn == sg)
        if len(idx) == 0:
            continue
        lost1 = lost_within(eng, int(source.nfp), R0s[idx], sg)
        for i in idx[lost1]:
            orbs[i] = dict(R0=float(R0s[i]), kind="lost", n=0, periods=0, sign=sg, rot=None, digits=None, flux=None)
        idx = idx[~lost1]
        if len(idx) == 0:
            continue
        # (one level: the survey's long traces; the short level's 1e-11 asked 2048 steps a period next to coils)
        tr = SD.SectionTracer(teng, int(source.nfp), ref_axis, sign=sg, tol_short=1e-9)
        y0 = np.stack([R0s[idx], 0 * R0s[idx]], -1)
        if not capped_calibrate(tr, y0[:: max(1, len(y0) // 4)], deadline=deadline):
            # lines that need more than MAX_STEPS steps a period pass next to coils: not traced (a survey of 16 lines
            # over 128 periods at 1024 steps ran past the stage's deadline by minutes)
            for i in idx:
                orbs[i] = dict(R0=float(R0s[i]), kind="steps", n=0, periods=None, sign=sg, rot=None, digits=None,
                               flux=None, steps=f"> {MAX_STEPS}")
            continue
        o = orbits_chunked(tr, y0, nper, deadline, chunk=8)
        for j, i in enumerate(idx):
            sec = o["sec"][j]
            per = int(np.sum(np.all(np.isfinite(sec), 1))) - 1
            try:
                rot_, dig_ = SD.rotation(o["dang"][j]) if not o["lost"][j] else (np.nan, np.nan)
                if not o["lost"][j] and np.isfinite(dig_) and dig_ < 3.0:
                    # (Birkhoff averages that disagree in the third digit: chaotic, without _orbit_record's circle-misfit
                    # scans -- up to 1600 least-squares fits an orbit, the survey's largest cost after the traces)
                    d = dict(R0=float(R0s[i]), n=int(o["dang"].shape[1]), kind="chaotic", rot=float(rot_), digits=float(dig_),
                             flux=None)
                else:
                    d = SD._orbit_record(R0s[i], sec, o["dang"][j], bool(o["lost"][j]), teng)
            except Exception as e:  # noqa: BLE001
                d = dict(R0=float(R0s[i]), kind="error", err=repr(e))
            d = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}
            d.update(periods=per, sign=sg)
            orbs[i] = d
            if traces is not None and d.get("kind") == "regular":
                traces[float(R0s[i])] = (np.array(sec), sg)
    kinds = {}
    for d in orbs:
        if d is not None:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    lost_at = [d["periods"] for d in orbs if d is not None and d["kind"] == "lost"]
    return dict(kinds=kinds, nper=nper, n=len(R0s), lost_periods_median=float(np.median(lost_at)) if lost_at else None,
                lost_within_one=int(sum(1 for d in orbs if d is not None and d["kind"] == "lost" and d["periods"] == 0)),
                orbits=orbs)


def orbit_torus(eng, source, ref_axis, orbit, sec, sign, request, deadline, tol=3e-15, dmin=None, verbose=False,
                quick=True):
    """A torus from a regular orbit of the survey: its invariant circle (torux.seeds.fit_circle in its rotation phase),
    the torus through it at its own transform (torux.seeds.torus_from_circle, Newton at fixed iota), then
    - with a request: torux.seeds.seed_and_walk at |request| with that torus as its base (circle Newton at the request
      from its section, regular orbits located around the request, walks);
    - without one: refinement at its own flux and the acceptance (a verified lower bound on the range).
    Every torus returned is accepted again against the period-one line inside it.  Returns (TorusState or None,
    log)."""
    from . import seeds as SD
    nfp = int(source.nfp)
    log = dict(orbit={k: orbit.get(k) for k in ("R0", "rot", "flux", "digits", "fit_err")}, attempts=[])
    rot = float(orbit["rot"])
    C = SD.fit_circle(sec, rot)
    if C is None or C.nm < 4:
        log["outcome"] = "no circle"
        return None, log
    C.rot = rot
    ctx = SD._Ctx(eng, source, ref_axis, abs(float(orbit["flux"])), deadline, verbose, tol, (384, 768),
                  dmin if dmin is not None else source.meta.get("dmin", 0.05))
    ctx.sgn = sign
    teng = SD.trace_engine(eng, source)
    tr = SD.SectionTracer(teng, nfp, ref_axis, sign=sign)
    n = len(sec)
    tr.calibrate(np.asarray(sec)[[0, n // 3, (2 * n) // 3]])
    ctx.tracer = tr
    sol = SD._torus_from_circle_solve(ctx, C, rot, max(C.nm, 24), 24, "survey orbit", log, fixed_iota=rot * nfp)
    if sol is None:
        log["outcome"] = "no torus from the circle"
        return None, log
    ctx.start = sol

    def shear(best):
        try:
            return SD.linear_shear(ctx.cont, best) * math.copysign(1.0, best.flux)
        except Exception:  # noqa: BLE001
            return np.nan
    ctx.shear = shear
    if request is not None:
        T, slog = SD.seed_and_walk(eng, source, abs(request), sol, ref_axis, deadline=deadline, verbose=verbose,
                                   tol=tol, dmin=ctx.dmin, trace=not quick)
        log["seeds"] = slog
        how = "survey orbit -> seeds"
    else:
        ctx.request = abs(float(sol.flux))        # the torus at its own flux (fixed transform): the range's bound
        T = SD._finish(ctx, sol, "survey orbit", log)
        how = "survey orbit"
    if T is None:
        log["outcome"] = "not accepted"
        return None, log
    ax_in = line_inside(eng, T, nfp)
    log["line_inside"] = None if ax_in is None else line_record(ax_in, 1, "Newton from the torus's m = 0 curve")
    if ax_in is not None:
        ok, acc = SD.accept(eng, source, ax_in, T, abs(T.flux), shear(T), verbose=verbose)
        log["accept_line_inside"] = dict(ok=bool(ok), topology=acc["topology"])
        if not ok:
            log["outcome"] = "rejected against the line inside"
            return None, log
        log.setdefault("attempts", []).append(dict(how=how + ": line inside", accept=acc))
    log["outcome"] = "surface"
    log["how"] = how
    return T, log


# ============================================================================ the stage
def noaxis_stage(eng, source, flux, axes, deadline=None, quick=True, tol=3e-15, dmin=None, verbose=False, opts=None):
    """See the module docstring.  axes: find_axes's lines (none elliptic).  Returns the stage's record (kind, ...)."""
    o = dict(NOAXIS)
    o.update(opts or {})
    t0 = time.time()
    deadline = _budget.earliest(deadline)
    rec = dict(kind="none", stages=[], timings={})

    def left():
        return np.inf if deadline is None else deadline - time.time()

    def stage(name, t_s, **kw):
        rec["stages"].append(dict(stage=name, t=round(t_s - t0, 2), dt=round(time.time() - t_s, 2), **kw))

    nfp = int(source.nfp)
    bnd = design_boundary(source)
    geo = geometry(source, bnd)
    rec["geometry"] = {k: v for k, v in geo.items() if k != "axis"}
    qsel = o["qs_quick"] if quick and "qs" not in (opts or {}) else o["qs"]
    qs = divisors(nfp) if qsel == "divisors" else [int(q) for q in qsel if nfp % int(q) == 0]
    # ------------------------------------------------------------ 1. closed lines
    t_s = time.time()
    try:
        lines, scans, complete = closed_lines(eng, source, geo, qs, nscan=int(o["nscan"]), deadline=deadline,
                                              known=axes, verbose=verbose)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        lines, scans, complete = [(line_record(a, 1, "find_axes"), a) for a in axes], {}, True
        rec["lines_error"] = repr(e)
        rec["lines_traceback"] = traceback.format_exc()[-3000:]
    rec["lines"] = [r for r, _ in lines]
    rec["scans"] = {str(q): s for q, s in scans.items()}
    rec["search_complete"] = bool(complete)
    rec["lines_traced"] = [x for sc in scans.values() for x in sc.get("traced_only", [])]
    stage("lines", t_s, n=len(lines))
    inside = lambda r: abs(r["R"] - geo["R_c"]) <= geo["a"] * 1.0
    usable = lambda r: r.get("confined_near", 1.0) > 0
    ell1 = [(r, a) for r, a in lines if r["q"] == 1 and r["kind"] == "elliptic" and usable(r)]
    if ell1:
        r, a = min(ell1, key=lambda ra: abs(ra[0]["R"] - geo["R_c"]))
        rec["kind"] = "axis"
        rec["axis_line"] = r
        rec["axis"] = a
        rec["timings"]["total"] = time.time() - t0
        return rec
    # ------------------------------------------------------------ 2. full-turn axes (q > 1, q | nfp)
    full = sorted([(r, a) for r, a in lines if r["q"] > 1 and r["kind"] == "elliptic" and usable(r)],
                  key=lambda ra: (not inside(ra[0]), abs(ra[0]["R"] - geo["R_c"])))
    for r, a in full:
        if left() < o["min_left"]:
            break
        t_s = time.time()
        dl = None if deadline is None else time.time() + o["family_share"] * left()
        try:
            with _budget.scope(dl):
                kind, frec = family_on_line(eng, source, a, r["q"], flux, dl, quick=quick, tol=tol, dmin=dmin,
                                            verbose=verbose)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            kind, frec = "error", dict(error=repr(e), traceback=traceback.format_exc()[-3000:])
        rec.setdefault("full_turn", []).append(dict(line=r, kind=kind,
                                                    **{k: v for k, v in frec.items() if k not in ("surface",)}))
        stage("full_turn_family", t_s, q=r["q"], outcome=kind)
        if kind in ("surface", "range"):
            rec["kind"] = kind
            rec["surface"] = frec["surface"]
            rec["guard"] = frec.get("guard")
            if kind == "range":
                rec["range"] = dict(verified_lower_bound=abs(frec["surface"]["flux"]), verified=True,
                                    note=f"tori around a closed line of the {r['q']}-period map (nfp / q = "
                                         f"{nfp // r['q']} periods): a lower bound")
            rec["timings"]["total"] = time.time() - t0
            return rec
    # ------------------------------------------------------------ 3. the request from the design boundary
    if o["seeds"] and flux is not None and bnd is not None and left() > o["min_left"]:
        t_s = time.time()
        plus = [(r, a) for r, a in lines if r["q"] == 1 and r["kind"] == "inverse_hyperbolic" and inside(r)]
        if plus:
            r, ref = min(plus, key=lambda ra: abs(ra[0]["R"] - geo["R_c"]))
            rec["reference_line"] = r
        else:
            ref = AX.Axis(geo["axis"][0], geo["axis"][1], nfp)
            rec["reference_line"] = dict(kind="geometric axis", R=float(np.sum(geo["axis"][0])))
        dl = None if deadline is None else time.time() + o["seeds_share"] * left()
        try:
            with _budget.scope(dl):
                T, slog = request_from_boundary(eng, source, bnd, flux, ref, dl, tol=tol, dmin=dmin, verbose=verbose,
                                                quick=quick)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            T, slog = None, dict(error=repr(e), traceback=traceback.format_exc()[-3000:])
        rec["request"] = slog
        stage("request_from_boundary", t_s, outcome=slog.get("outcome"))
        if T is not None:
            from .seeds import surface_record
            s, g = surface_record(T, slog["seeds"])
            s["nfp"] = nfp
            rec["kind"] = "surface"
            rec["surface"] = s
            rec["guard"] = g
            rec["timings"]["total"] = time.time() - t0
            return rec
    # ------------------------------------------------------------ 4. evidence
    plus = sorted([(r, a) for r, a in lines if r["q"] == 1 and r["kind"] == "inverse_hyperbolic" and inside(r)],
                  key=lambda ra: abs(ra[0]["R"] - geo["R_c"]))
    # quick mode: the survey where it can answer -- no request (a range from regular orbits), or an inverse-hyperbolic
    # line inside the section (regular orbits beyond its period-two chain); elsewhere the scan's surviving intervals
    # are the evidence (the survey's 128-period traces find no regular orbit away from such lines)
    run_survey = o["survey"] and (not quick or flux is None or bool(plus) or o.get("survey_quick_all"))
    if run_survey and left() > o["min_left"]:
        t_s = time.time()
        ref = plus[0][1] if plus else AX.Axis(geo["axis"][0], geo["axis"][1], nfp)
        dl = None if deadline is None else time.time() + o["survey_share"] * left()
        traces = {}
        try:
            ev = survey(eng, source, geo, ref, nlines=int(o["survey_lines"]), nper=int(o["survey_periods"]),
                        deadline=dl, traces=traces, near=[r["R"] for r, _ in plus[:2]])
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            ev = dict(error=repr(e))
        rec["evidence"] = ev
        stage("survey", t_s, kinds=ev.get("kinds"))
        # ------------------------------------------------------------ 5. tori from the survey's regular orbits
        reg = [d for d in (ev.get("orbits") or []) if d and d.get("kind") == "regular" and d.get("flux")
               and float(d["R0"]) in traces]
        if reg and o.get("orbit_seeds", True) and left() > o["min_left"]:
            # with a request: the orbit nearest it in flux; without: the outermost (the largest flux)
            if flux is not None:
                reg = sorted(reg, key=lambda d: abs(math.log(float(d["flux"]) / abs(flux))))
            else:
                reg = sorted(reg, key=lambda d: -float(d["flux"]))
            for d in reg[:2]:
                if left() < o["min_left"]:
                    break
                t_s = time.time()
                sec, sg = traces[float(d["R0"])]
                dl = None if deadline is None else time.time() + 0.8 * left()
                try:
                    with _budget.scope(dl):
                        T, olog = orbit_torus(eng, source, ref, d, sec, sg, flux, dl, tol=tol, dmin=dmin,
                                              verbose=verbose, quick=quick)
                except Exception as e:  # noqa: BLE001
                    if environment_failure(e):
                        raise
                    T, olog = None, dict(error=repr(e), traceback=traceback.format_exc()[-3000:])
                rec.setdefault("orbit_seeds", []).append({k: v for k, v in olog.items() if k != "attempts"})
                stage("orbit_seed", t_s, outcome=olog.get("outcome"))
                if T is not None:
                    from .seeds import surface_record
                    s, g = surface_record(T, olog.get("seeds") or olog)
                    s["nfp"] = nfp
                    rec["surface"] = s
                    rec["guard"] = g
                    if flux is not None:
                        rec["kind"] = "surface"
                    else:
                        rec["kind"] = "range"
                        rec["range"] = dict(verified_lower_bound=abs(float(T.flux)), verified=True,
                                            note="a torus through a regular orbit of the survey: a lower bound")
                    break
    rec["timings"]["total"] = time.time() - t0
    return rec
