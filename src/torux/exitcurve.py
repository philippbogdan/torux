"""Upper bounds on the flux of every flux surface around the magnetic axis from EXIT curves: closed curves in the
symmetry half plane every point of which is certified to lie outside every covered torus -- lost (torux.edgebound's
lost curve) or with an iterate of the field-period map outside an already certified curve.  The certified curves march
inward from the lost region towards the last torus, each certified with a short window (no long traces, no
exponential stretching), so the bound follows lines that need hundreds of periods to be lost.

    from torux.exitcurve import exit_curve_bound
    rec = exit_curve_bound(source, axis_point=(R_a, 0.0), T_last=torus, request=None, budget_s=600)
    rec["ok"], rec["F_up"]     # every covered torus encloses at most F_up

The analysis hook is edge_bound_exit (the lost curve first, returned when it holds the request, then the march from
it; never above the lost curve's bound).

The theorem and what is strict and what validated are in THEOREM_DOC; the converse-KAM analysis (why the twist /
cone criterion cannot be made sound for this problem, and the restricted class for which it is) in CONVERSE_KAM_DOC.
The numerics reuse torux.edgebound's field rules (calibrated per-coil nested trapezoid rules, checked against the
reference evaluator in every run), its compiled arc-length GBS tracer and its 1D lost-piece test.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from .edgebound import (ExactTracer, _gbs, _cross, _fval, _step_ctl, H_COIL, Q_FAC, Q_DECAY, COIL_GUARD_P,
                        _seg_base_test, supported)

THEOREM_DOC = """THEOREM (exit curves).  Setting, covered tori Gamma (T1-T3), V, D, flux, s, g = s R B_phi, Lemma A and lost points
as in torux/barrier.py and torux/edgebound.py.  The source is nfp-periodic and stellarator symmetric (rotation Rot by
2 pi / nfp about the z axis, reflection S (R, phi, Z) -> (R, -phi, -Z)); then Rot and S map covered tori to covered
tori (they map sources to sources, field lines to field lines, s B_phi to s B_phi on the image, the axis to the axis).

OUT.  O is the set of points z of Sigma that lie in no closed disc D-bar of a covered torus.

(O1) Lost points are in O (Lemma L).  Points with g <= 0 and points on a coil are in O (g > 0 on V-bar, T1).
(O2) Let z in Sigma and x a point of the field line through z (either direction, any arc length) that lies in a plane
     phi = 2 pi k / nfp (k any integer, phi unwrapped or not).  If Rot^-k x is in O, then z is in O.
     Proof: if z is in D-bar of a covered Gamma, the line through z stays in V-bar (Lemma A), so x is in V-bar and in
     the plane; V-bar meets that plane in the closed disc bounded by the Jordan curve Gamma n {phi = 2 pi k / nfp}
     (T2: Gamma is transverse to the planes), and Rot^-k of that disc is D-bar of the covered torus Rot^-k Gamma.  So
     Rot^-k x is in a covered D-bar: not in O.
(O3) S(O) = O (S maps covered discs to covered discs).
(O4) If C is a closed curve in O with winding number 1 about a, every covered D lies in the component K_a(C) of
     Sigma \\ C that contains a (D is connected, contains a, misses C), hence Sigma \\ K_a(C) is in O and
         flux(Gamma) = int_D s B_phi <= int_{K_a(C)} (s B_phi)^+     (Lemma A: s B_phi > 0 on D).
So O is closed under the period map and its inverse, under S, and under 'everything outside a certified curve'.
EXIT CURVES.  C_0 a lost curve (every point lost: torux.edgebound.lost_curve_bound, run first); C_{j+1} a closed polygon each
of whose pieces lies outside K_a(C_i) (k = 0), or has, for some k in [1, W], all its points' k-th crossings (forward
lines; backward lines via S: the forward lines of the mirror piece) outside K_a(C_i), for some certified i <= j, or is
lost.  By (O1)-(O4) every C_j lies in O and every covered torus has flux <= int_{K_a(C_j)} (s B_phi)^+ for every j:
the bound is the smallest over the certified curves.  Each curve needs only its own pieces' certificates.
SELECTION (no part of any certificate).  A detection map (loose tracer: tolerance 1e-5, rules eps 1e-6) follows every
node of a grid of spacing r_eff / 16 (lost_curve_bound's grid) forward and backward through W = 8 crossings; a node is
'exit-looking' when a crossing lies outside the newest curve by h / 4 or it is lost within 4 periods.  Each march step
routes (Dijkstra, upper half, symmetry row to symmetry row) through exit-looking nodes inside-or-on the newest curve
and at most `band` = 4 node spacings inside it, pulled towards the axis component of the rest, edges along the newest
curve (certified) at 4 x cost; the route's uncertified edges are certified together (certify_edges: every piece of
every edge in one parallel batch); a failed edge is blocked and the route redone; a failed step is retried with half
the band; when the march stalls with time left the window is doubled (8 -> 16 -> 32) and the detection redone inside
the newest curve.

THE PIECE CERTIFICATE (exit).  A piece [p - eta e, p + eta e] of an edge of the polygon (Z >= 0 half, its mirror
image by (O3)); the lines from u = -1, -1/2, 0, 1/2, 1 followed (edgebound's compiled GBS in arc length on the
calibrated field rules) through their plane crossings.  At a crossing k reached by all five with s B_phi / |B| >=
F_SMOOTH on every accepted step up to it (no turning point: the crossing map is smooth on the piece) and every
line at least COIL_GUARD_P (0.5, torux.edgebound's period certificate) x (the largest extent of the 3D image at the
crossings up to k + rho) from every coil, the images Y(u) (rotated back
to phi = 0, in (R, Z)) give the quadratic interpolant I(u) through u = -1, 0, 1; its error e2 at u = +-1/2 must be at
most Q_DECAY x a quarter of |I''| (the cubic content small against the quadratic) or 1e-3 of the chord; then the image
of the piece lies in I([-1, 1]) + ball(rho), rho = Q_FAC e2 + tau, tau = 2 x the summed GBS step estimates / the
smallest f (a position error moves the crossing by at most its size over the line's angle to the plane) + 1e-12 +
2 eps_f x arc length x max(1, stretch) (the field model's error).  I is covered by m sub-arcs (centre I(u_c),
chord I'(u_c) dl, remainder |I''| dl^2 / 2: exact for the polynomial); each sub-arc's segment dilated by rho +
remainder must keep a positive distance from the target polygon and its centre must lie outside it (winding
number 0 about the centre: the dilated segment is connected and misses the polygon, so it lies outside).
THE PIECE CERTIFICATE (lost): torux.edgebound._seg_base_test (the lost curve's).

What is strict and what is validated
------------------------------------
STRICT (given the computed numbers): (O1)-(O4) and the bound; the symmetry closure; the polygon tests (exact
segment-segment distances and winding numbers in floating point far above rounding); the polygon's simplicity and
winding number (checked on the vertices); Stokes on the polygon; the second-derivative remainder of the polynomial.
VALIDATED (error models checked against independent computations in every run, as in torux/edgebound.py):
  * the enclosure model of a piece's crossing images (the quadratic interpolant's error from u = +-1/2, factor
    Q_FAC = 3; the cubic terms peak within 1.03 x their value there) -- checked on sampled certified pieces by 9
    extra lines whose crossings must all lie outside the target polygon;
  * the trace error (2 x the summed GBS estimates): sampled certified crossings re-traced by scipy DOP853 in arc
    length on the reference field (torux.reference.fast64) with an event at the plane -- the distance must be below
    the error bound -- and the re-traced crossing must lie outside the target polygon;
  * the smoothness of the crossing map on a piece (s B_phi / |B| >= F_SMOOTH along the five lines) and the coil guard;
  * the field (calibrated nested trapezoid rules, eps 1e-10; against the reference evaluator at 48 points per run);
  * the exact nfp and stellarator symmetry of the source (QUASR / CoilStellaration coil sets are generated by the
    symmetry group; edgebound's lost curve relies on S already) -- measured in every run (symmetry_defect: the
    field rules' B at 24 random points against B at the rotated / reflected points, ~1e-12; refused above 1e-9);
  * floating-point rounding is not interval-bounded.
NEGATIVE TEST in every run with a verified torus: its section must lie inside every certified curve (2048 section
points inside, and the exact smallest distance between curve and section polygons > 0), its flux below every F_up,
and no chord (length ~ h / 4) of the section may be certified as an exit or lost piece against the certified curves
(8 chords); a violation refuses the run as a false certificate.
"""

CONVERSE_KAM_DOC = """Converse KAM for this map: what holds, and why it is not used for certificates.

THE CONE LEMMA (MacKay-Percival, in iterate form).  Coordinates (theta, rho) near an orbit z_j = P^j z_0 of the
period map, V = d/d rho the 'vertical'.  Let Gamma be a P-invariant Jordan curve through the orbit with inside D, and
  (G) near z_i, z_m and z_n (i <= m <= n) Gamma is the graph rho = h(theta) of a Lipschitz function, D below it.
If u = DP^(m-i)(z_i) V has a positive theta component and v = DP^-(n-m)(z_n) V a negative one, then
        slope(v) <= every Dini derivative of h at theta(z_m) <= slope(u),
so slope(u) < slope(v) (the 'cones cross') excludes every such Gamma through the orbit.
Proof: a short vertical segment sigma at z_i has its lower half in D and its upper half outside (G at z_i); P^(m-i)
preserves D and its outside, so P^(m-i)(sigma) passes through z_m from D to the outside; its tangent u points to
increasing theta, so near z_m it is a graph over theta, below h on the left of theta(z_m) and above it on the right
(G at z_m): its slope is at least every Dini derivative of h there.  Backwards the same with v pointing to decreasing
theta.  QED.  No twist is needed at the intermediate points; the classical twist condition (DP V has a positive theta
component everywhere) only makes the theta components of u, v have the right signs.

THE GAP: (G).  The theorem would need (G) for EVERY covered torus through the curve C, and Birkhoff's graph theorem is
the only source of it: an invariant essential circle of an area-preserving map that is a twist map on an invariant
annulus containing it is a Lipschitz graph.  Its proof uses the twist on the vertical segments below every point of
the circle -- the whole 'vertical hull' of the circle, pockets outside it included -- and an invariant bottom.  Here
  * the circles to exclude are the sections of arbitrary covered tori through C; all that is known of their position
    is that they contain the axis and lie inside the certified lost curve.  Their vertical hulls reach the lost
    region, where lines escape within a period or two and P is not defined (B_phi changes sign) or far from twist;
  * the bottom must be an exactly invariant circle -- the verified outer torus is invariant only to its residual
    (B.n <= 1e-14: validated), and a circle crossing it is not excluded without Denjoy-minimality of the true torus,
    which is not proven -- or the axis blown up, the twist then needed down to the axis;
  * the twist itself fails on much of the region (measured on 26 devices with rays from the axis as the vertical
    foliation, one field period): inside the
    verified / design section a median 40 % of the samples (3-49 %) twist against the majority sign, on 1 <= rho <=
    1.3 only a median 57 % twist with it, and beyond rho = 1.3 the period map is undefined (the line is lost first)
    on a median 35 %; with reversed shear invariant 'meanders' that are not graphs exist (non-twist maps): (G) is
    then genuinely false.
A local Birkhoff theorem (twist only near the circle) can be shown for C^1 circles on which P acts minimally (with the
isotopy lift of DP rotating the vertical by less than pi, the index floor((alpha - pi/2) / pi) of the lifted tangent
angle alpha cannot increase along orbits and drops at a vertical tangency, which a dense orbit forbids), not for
rational or merely continuous circles, which covered tori may have.  Hence converse KAM is SOUND here only for the
restricted class of covered tori whose sections satisfy (G) near the orbits used (e.g. star-shaped sections about a,
with rays as the vertical) -- a weaker answer than 'no surface' -- and no certificate of this module uses it.
"""

F_SMOOTH = 0.2          # smallest s B_phi / |B| along the five lines of a piece up to the crossing used (no turning point)
M_SUB = 8               # sub-arcs covering the quadratic image of a piece


# ============================================================================ multi-crossing tracer
@njit(parallel=True, cache=True)
def multicross_kernel(X0, s, ncross, S_max, beta, d_fail, tol, H0, H_max, k, P, dphi):
    """Lines from X0 (L, 3) along s B / |B| through their first ncross crossings of the planes phi = phi_0 + m dphi
    (m = 1..ncross, phi unwrapped from the start's angle phi_0), until lost (f = s B_phi / |B| <= -beta, then to the
    minimum of f), the arc length S_max, a coil (distance lower bound < d_fail) or a failed step.
    Returns XC (L, ncross, 3) the crossings (NaN beyond the last), SC (L, ncross) their arc lengths, EC the summed GBS
    estimates there, FC the smallest f on accepted steps up to each crossing, DC the smallest coil-distance lower bound
    up to it; nc (L,) crossings recorded; lost (L,) bool; sf, se (L,) the arc lengths of the first f <= -beta and of
    the lost point (the minimum of f after it); status (L,) 0 all crossings, 1 lost, 2 arc length, 4 coil, 5 fail."""
    L = X0.shape[0]
    XC = np.full((L, ncross, 3), np.nan)
    SC = np.full((L, ncross), np.nan); EC = np.full((L, ncross), np.nan)
    FC = np.full((L, ncross), np.nan); DC = np.full((L, ncross), np.nan)
    nc = np.zeros(L, np.int64)
    lostv = np.zeros(L, np.bool_)
    sfv = np.full(L, np.nan); sev = np.full(L, np.nan)
    XE = np.full((L, 3), np.nan)
    status = np.full(L, 5, np.int64)
    dfail2 = d_fail * d_fail
    for i in prange(L):
        x = X0[i].copy()
        xn = np.empty(3); xt = np.empty(3)
        sacc = 0.0
        eacc = 0.0
        H = H0
        phi0 = math.atan2(x[1], x[0])
        phi = phi0
        m = 1
        lost_on = False
        best_f = np.inf
        best_s = 0.0
        stat = 2
        f, q = _fval(x, s, P)
        qmin = q
        fm = f
        it = 0
        while it < 2000000:
            it += 1
            Hs = min(H, S_max - sacc) if not lost_on else H
            if Hs <= 0:
                stat = 2
                break
            ok, err, q = _gbs(x, Hs, k, s, P, xn)
            if q < qmin:
                qmin = q
            if q < dfail2:
                stat = 4
                break
            sc = 1.0 + math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
            if (not ok) or err > tol * sc:
                H = Hs * (0.25 if not ok else max(0.2, _step_ctl(err, Hs, tol, sc, k)))
                if H < 1e-11:
                    stat = 5
                    break
                continue
            phin = phi + math.atan2(x[0] * xn[1] - x[1] * xn[0], x[0] * xn[0] + x[1] * xn[1])
            while (not lost_on) and m <= ncross and phin >= phi0 + m * dphi:
                hc, ec = _cross(x, xt, phi, phi0 + m * dphi, Hs, phin, k, s, P)
                for c in range(3):
                    XC[i, m - 1, c] = xt[c]
                SC[i, m - 1] = sacc + hc
                EC[i, m - 1] = eacc + ec
                FC[i, m - 1] = fm
                DC[i, m - 1] = math.sqrt(qmin)
                nc[i] = m
                m += 1
            for c in range(3):
                x[c] = xn[c]
            sacc += Hs
            eacc += err
            phi = phin
            if m > ncross and not lost_on:
                stat = 0
                break
            f, q = _fval(x, s, P)
            if f < fm:
                fm = f
            if f <= -beta and not lost_on:
                lost_on = True
                sfv[i] = sacc
            if lost_on:
                if f < best_f:
                    best_f = f
                    best_s = sacc
                    for c in range(3):
                        XE[i, c] = x[c]
                elif f > best_f + 0.02 or sacc > best_s + 0.3:
                    stat = 1
                    break
            H = min(Hs * min(3.0, max(0.2, _step_ctl(err, Hs, tol, sc, k))), H_max)
            H = min(H, max(H_COIL * math.sqrt(q), 1e-6))
        if lost_on:
            lostv[i] = True
            sev[i] = best_s
            stat = 1
        status[i] = stat
    return XC, SC, EC, FC, DC, nc, lostv, sfv, sev, status, XE


def multicross(tr: ExactTracer, X0, ncross, S_max):
    """multicross_kernel with the tracer's field rules, sign, tolerance and threshold (dict of arrays)."""
    X0 = np.ascontiguousarray(np.asarray(X0, float).reshape(-1, 3))
    keys = ("XC", "SC", "EC", "FC", "DC", "nc", "lost", "sf", "se", "status", "XE")
    if len(X0) == 0:
        return {k_: np.zeros((0, max(int(ncross), 1), 3)) if k_ == "XC" else np.zeros((0, max(int(ncross), 1)))
                if k_ in ("SC", "EC", "FC", "DC") else (np.zeros((0, 3)) if k_ == "XE" else np.zeros(0))
                for k_ in keys}
    t0 = time.time()
    kern = getattr(tr, "mc_kernel", None) or multicross_kernel      # torux.exitcurve_nearcoil: near-coil steps
    out = kern(X0, tr.s, int(ncross), float(S_max), tr.beta, tr.d_fail, tr.tol, 0.02, 0.1, tr.k, tr.P,
               2 * math.pi / tr.nfp)
    tr.t += time.time() - t0
    tr.n_lines += len(X0)
    return dict(zip(keys, out))


def crossing_RZ(XC, ncross, nfp):
    """(R, Z) of the crossings XC (..., ncross, 3) rotated back to phi = 0 (the m-th crossing lies in phi_0 + m dphi;
    phi_0 = 0 for starts in Sigma): rotation by -m 2 pi / nfp of (x, y) leaves R = hypot(x, y) and Z."""
    return np.stack([np.hypot(XC[..., 0], XC[..., 1]), XC[..., 2]], -1)


# ============================================================================ polygon geometry (numba)
@njit(cache=True)
def _orient(px, py, qx, qy, rx, ry):
    return (qx - px) * (ry - py) - (qy - py) * (rx - px)


@njit(cache=True)
def _pt_seg(px, py, ux, uy, vx, vy):
    wx = vx - ux; wy = vy - uy
    L2 = wx * wx + wy * wy
    t = 0.0
    if L2 > 0:
        t = ((px - ux) * wx + (py - uy) * wy) / L2
        t = min(1.0, max(0.0, t))
    ex = ux + t * wx - px; ey = uy + t * wy - py
    return math.sqrt(ex * ex + ey * ey)


@njit(cache=True)
def _seg_seg_dist(ax, ay, bx, by, cx, cy, dx, dy):
    """Distance between the segments [a, b] and [c, d] in the plane (0 if they intersect)."""
    o1 = _orient(ax, ay, bx, by, cx, cy); o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay); o4 = _orient(cx, cy, dx, dy, bx, by)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        return 0.0
    d1 = _pt_seg(ax, ay, cx, cy, dx, dy); d2 = _pt_seg(bx, by, cx, cy, dx, dy)
    d3 = _pt_seg(cx, cy, ax, ay, bx, by); d4 = _pt_seg(dx, dy, ax, ay, bx, by)
    return min(min(d1, d2), min(d3, d4))


@njit(cache=True)
def _winding(Cx, Cy, px, py):
    n = Cx.shape[0]
    tot = 0.0
    for i in range(n):
        j = (i + 1) % n
        a0 = math.atan2(Cy[i] - py, Cx[i] - px)
        a1 = math.atan2(Cy[j] - py, Cx[j] - px)
        d = a1 - a0
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        tot += d
    return int(round(tot / (2 * math.pi)))


@njit(cache=True)
def _segs_outside(Cx, Cy, A, Bv, R):
    """Every dilated segment {A[q] + t (Bv[q] - A[q]), t in [0, 1]} + ball(R[q]) outside the closed polygon (Cx, Cy)?
    (distance to every edge > R[q], and the midpoint's winding number 0).  Returns the smallest margin
    (distance - R) over the segments (negative: some segment fails), or -1e300 when a centre is inside."""
    n = Cx.shape[0]
    worst = 1e300
    for q in range(A.shape[0]):
        mx = 0.5 * (A[q, 0] + Bv[q, 0]); my = 0.5 * (A[q, 1] + Bv[q, 1])
        dmin = 1e300
        for i in range(n):
            j = (i + 1) % n
            d = _seg_seg_dist(A[q, 0], A[q, 1], Bv[q, 0], Bv[q, 1], Cx[i], Cy[i], Cx[j], Cy[j])
            if d < dmin:
                dmin = d
        mg = dmin - R[q]
        if mg < worst:
            worst = mg
        if mg <= 0:
            return mg
        if _winding(Cx, Cy, mx, my) != 0:
            return -1e300
    return worst


@njit(parallel=True, cache=True)
def points_inside(Cx, Cy, X, Y):
    """Winding number != 0 of the closed polygon about each point (X, Y) (n,)."""
    out = np.zeros(X.shape[0], np.bool_)
    for q in prange(X.shape[0]):
        out[q] = _winding(Cx, Cy, X[q], Y[q]) != 0
    return out


@njit(parallel=True, cache=True)
def points_outside_margin(Cx, Cy, X, Y):
    """Signed distance of each point to the closed polygon: + outside (winding 0), - inside."""
    out = np.empty(X.shape[0])
    n = Cx.shape[0]
    for q in prange(X.shape[0]):
        dmin = 1e300
        for i in range(n):
            j = (i + 1) % n
            d = _seg_seg_dist(X[q], Y[q], X[q], Y[q], Cx[i], Cy[i], Cx[j], Cy[j])
            if d < dmin:
                dmin = d
        out[q] = dmin if _winding(Cx, Cy, X[q], Y[q]) == 0 else -dmin
    return out


@njit(cache=True)
def _inside_eo(Cx, Cy, px, py):
    """Even-odd point-in-polygon (the polygons here are simple: checked)."""
    n = Cx.shape[0]
    c = False
    for i in range(n):
        j = (i + 1) % n
        yi = Cy[i]; yj = Cy[j]
        if (yi > py) != (yj > py):
            xin = Cx[i] + (py - yi) * (Cx[j] - Cx[i]) / (yj - yi)
            if px < xin:
                c = not c
    return c


@njit(parallel=True, cache=True)
def signed_margin(Cx, Cy, X, Y):
    """Signed distance of the points (X, Y) to the closed simple polygon: + outside, - inside (NaN for NaN points)."""
    out = np.empty(X.shape[0])
    n = Cx.shape[0]
    for q in prange(X.shape[0]):
        if not (X[q] == X[q]):
            out[q] = np.nan
            continue
        dmin = 1e300
        for i in range(n):
            j = (i + 1) % n
            d = _pt_seg(X[q], Y[q], Cx[i], Cy[i], Cx[j], Cy[j])
            if d < dmin:
                dmin = d
        out[q] = -dmin if _inside_eo(Cx, Cy, X[q], Y[q]) else dmin
    return out


# ============================================================================ flux inside a closed polygon
def curve_flux_bound(tr: ExactTracer, source, curve, sign, h):
    """F_up of the closed counter-clockwise polygon `curve` (torux.edgebound.lost_curve_bound's formula): -s x the loop
    integral of A (Gauss-Legendre 6 per edge, the 3-point difference added twice), + the (s B_phi)^- bound on a raster
    of spacing h / 2 inside (per cell max(0, 2 |grad B_phi| x half diagonal - s B_phi(centre)) x area), the cells
    within 2 sqrt 2 h / 2 of a coil crossing replaced by 1.5 mu0 |I| rho + 2 max|B| pi rho^2, + 1e-14 |F|."""
    from .barrier import inside_polygon, coil_crossings
    E = np.concatenate([curve, np.roll(curve, -1, axis=0)], axis=1)

    def loop(n):
        xg, wg = np.polynomial.legendre.leggauss(n)
        tq = 0.5 * (xg + 1.0); wt = 0.5 * wg
        Pq = E[:, None, :2] + tq[None, :, None] * (E[:, None, 2:] - E[:, None, :2])
        Xq = np.stack([Pq[..., 0], 0 * Pq[..., 0], Pq[..., 1]], -1).reshape(-1, 3)
        A = tr.BGA(Xq)[2].reshape(len(E), n, 3)
        dd = E[:, 2:] - E[:, :2]
        return float(np.sum(wt[None, :] * (A[..., 0] * dd[:, None, 0] + A[..., 2] * dd[:, None, 1])))
    I6, I3 = loop(6), loop(3)
    F = -sign * I6
    hh = 0.5 * h
    lo_ = curve.min(0); hi_ = curve.max(0)
    gr = np.stack(np.meshgrid(np.arange(lo_[0] + hh / 2, hi_[0], hh), np.arange(lo_[1] + hh / 2, hi_[1], hh),
                              indexing="ij"), -1).reshape(-1, 2)
    gr = gr[inside_polygon(curve, gr)]
    B, D, _, _ = tr.BGA(np.stack([gr[:, 0], 0 * gr[:, 0], gr[:, 1]], -1))
    bp = sign * B[:, 1]
    reach = 2.0 * np.linalg.norm(D[:, 1, :], axis=1) * hh / math.sqrt(2.0)
    cc = coil_crossings(source)
    cc = cc[inside_polygon(curve, cc)] if len(cc) else cc
    rho_c = 2.0 * hh * math.sqrt(2.0)
    near = np.zeros(len(gr), bool)
    disk_corr = 0.0
    Imax = max(float(np.max(np.abs(p.current))) for p in source.parts if p.kind == "fourier")
    for c_ in cc:
        near |= np.hypot(gr[:, 0] - c_[0], gr[:, 1] - c_[1]) < rho_c + hh
        th_ = 2 * np.pi * np.arange(16) / 16
        ring = np.stack([c_[0] + rho_c * np.cos(th_), 0 * th_, c_[1] + rho_c * np.sin(th_)], -1)
        Bb = float(np.max(np.linalg.norm(tr.B(ring)[:, :3], axis=1)))
        disk_corr += 1.5 * 4e-7 * math.pi * Imax * (rho_c + hh) + 2.0 * Bb * math.pi * (rho_c + hh) ** 2
    neg = (bp - reach < 0) & ~near
    corr = float(np.sum(np.where(near, 0.0, np.maximum(reach - bp, 0.0))) * hh * hh) + disk_corr
    F_up = F + corr + 2 * abs(I6 - I3) + 1e-14 * abs(F)
    return float(F_up), dict(F=F, quad_err=abs(I6 - I3), neg_corr=corr, n_neg=int(neg.sum()), coils_inside=int(len(cc)),
                             coil_disk_corr=disk_corr)


# ============================================================================ the exit test of a piece
def exit_test(Y5, EC5, FC5, DC5, ext5, S5, eta, targets, ef, m_sub=M_SUB):
    """Y5 (5, 2): the crossing images (u = -1, -1/2, 0, 1/2, 1) of a piece at one crossing; EC5, FC5 their summed
    GBS estimates and smallest f up to it; DC5 their smallest coil-distance bounds up to it; ext5 the largest 3D
    extent of the five images at the crossings up to it; S5 their arc lengths; eta the piece's half length; targets:
    list of (Cx, Cy) closed polygons (each certified: its outside is in O).  Returns (target index, margin) when the
    curved enclosure lies outside one of them, else (None, reason)."""
    if not (np.all(np.isfinite(Y5)) and float(np.min(FC5)) >= F_SMOOTH):
        return None, "fsmooth"
    Ym, Yh1, Y0, Yh2, Yp = Y5
    I = lambda u: Ym * 0.5 * u * (u - 1) + Y0 * (1 - u * u) + Yp * 0.5 * u * (u + 1)
    e2 = max(float(np.linalg.norm(I(-0.5) - Yh1)), float(np.linalg.norm(I(0.5) - Yh2)))
    d2 = float(np.linalg.norm(Ym - 2 * Y0 + Yp))
    lin = 0.5 * float(np.linalg.norm(Yp - Ym))
    if not (e2 <= Q_DECAY * 0.25 * d2 + 1e-15 or e2 <= 1e-3 * lin):
        return None, "cubic"
    tau = 2.0 * float(np.max(EC5)) / max(float(np.min(FC5)), F_SMOOTH) + 1e-12 \
        + 2.0 * ef * float(np.max(S5)) * max(1.0, lin / max(eta, 1e-300))
    rho = Q_FAC * e2 + tau
    if not float(np.min(DC5)) >= COIL_GUARD_P * (float(ext5) + rho):
        return None, "guard"
    dl = 1.0 / m_sub
    cen = -1.0 + dl * (2 * np.arange(m_sub) + 1)
    Pc = np.stack([I(u) for u in cen])
    Pu = np.stack([(Yp - Ym) * 0.5 + (Ym - 2 * Y0 + Yp) * u for u in cen]) * dl
    r_sub = 0.5 * dl * dl * d2
    A = np.ascontiguousarray(Pc - Pu); Bv = np.ascontiguousarray(Pc + Pu)
    R = np.full(m_sub, rho + r_sub)
    for ti, (Cx, Cy) in enumerate(targets):
        mg = _segs_outside(Cx, Cy, A, Bv, R)
        if mg > 0:
            return ti, float(mg)
    return None, "inside"


# ============================================================================ segment certificates (exit or lost)
_FR = (0.0, 0.25, 0.5, 0.75, 1.0)
_MIR = np.array([1.0, 1.0, -1.0])


def _piece_rows(cache, ts, d_):
    return [cache[(t, d_)] for t in ts]


def _certify_exit_segment(tr, sgn, p0, p1, dsgn, targets, W, parc, max_split, deadline, stats, n_cand=4,
                          max_pieces=64):
    """Every point of the segment p0 -> p1 (Sigma, Z >= 0) in O?  k = 0: the segment outside a target polygon.  Else
    pieces halved (up to max_split times, at most max_pieces tried) until each has an exit certificate (exit_test
    against one of the target polygons at some crossing k <= W of forward lines, or -- the mirror piece's forward
    lines -- backward ones) or a lost certificate (torux.edgebound._seg_base_test).  The midpoint line of a piece is
    traced first; the other four only when it shows a crossing outside a target.  Returns the list of certified
    pieces (kind, t0, t1, direction, k or arc length, target, margin) or None."""
    first = 1 if dsgn >= 0 else -1
    L01 = float(np.linalg.norm(p1 - p0))
    # k = 0: the segment itself outside a certified curve (exact geometry, no trace)
    for ti, (Cx, Cy) in enumerate(targets):
        mg0 = _segs_outside(Cx, Cy, np.array([[p0[0], p0[2]]]), np.array([[p1[0], p1[2]]]), np.zeros(1))
        if mg0 > 0:
            stats["outside_edges"] = stats.get("outside_edges", 0) + 1
            return [("outside", 0.0, 1.0, 0, 0, ti, float(mg0))]
    todo = [(0.0, 1.0, 0)]
    done = []
    cache = {}                  # (t, d) -> (row, crossings traced)
    ef = max(tr.check_err or 0.0, tr.eps)
    tf = lambda S, lin, e_: 2.0 * ef * S * max(1.0, lin / max(e_, 1e-300))
    n_pieces = 0

    def trace(ts, d_, ncr):
        new = [t for t in ts if (t, d_) not in cache or cache[(t, d_)][1] < ncr]
        new = sorted(set(new))
        if new:
            X = np.array([p0 + t * (p1 - p0) for t in new]) * (_MIR if d_ == -1 else 1.0)
            o = multicross(tr, X, ncr, ncr * parc * 3.0)
            stats["lines"] = stats.get("lines", 0) + len(new)
            for m_, t in enumerate(new):
                cache[(t, d_)] = ({k_: v_[m_] for k_, v_ in o.items()}, ncr)

    def mid_margins(row):
        n = int(row["nc"])
        if n < 1 or not targets:
            return np.zeros(0)
        Y = np.stack([np.hypot(row["XC"][:n, 0], row["XC"][:n, 1]), row["XC"][:n, 2]], -1)
        mg = np.full(n, -np.inf)
        for (Cx, Cy) in targets:
            mg = np.maximum(mg, signed_margin(Cx, Cy, np.ascontiguousarray(Y[:, 0]), np.ascontiguousarray(Y[:, 1])))
        return np.where(row["FC"][:n] >= F_SMOOTH, mg, -np.inf)

    while todo:
        if time.time() > deadline:
            stats["seg_timeout"] = stats.get("seg_timeout", 0) + 1
            return None
        batch, todo = todo[:48], todo[48:]
        n_pieces += len(batch)
        res = [None] * len(batch)
        progress = np.zeros(len(batch), bool)
        for d_ in (first, -first):
            left = [q for q in range(len(batch)) if res[q] is None]
            if not left:
                break
            mids = [0.5 * (batch[q][0] + batch[q][1]) for q in left]
            trace(mids, d_, W)
            # the other four lines of the pieces whose midpoint has a crossing outside a target (to its last one)
            need = {}
            mgs = {}
            for q, tm in zip(left, mids):
                mg = mid_margins(cache[(tm, d_)][0])
                mgs[q] = mg
                if np.any(mg > 0):
                    progress[q] = True
                    need[q] = int(np.max(np.flatnonzero(mg > 0))) + 1
            if need:
                kmax = max(need.values())
                trace([batch[q][0] + f * (batch[q][1] - batch[q][0]) for q in need for f in _FR], d_, kmax)
            for q in left:
                t0, t1, dep = batch[q]
                eta = 0.5 * L01 * (t1 - t0)
                if q in need:
                    rows = [cache[(t0 + f * (t1 - t0), d_)][0] for f in _FR]
                    ncom = int(min(r_["nc"] for r_ in rows))
                    mg = mgs[q][:ncom]
                    if ncom >= 1:
                        XC = np.stack([r_["XC"][:ncom] for r_ in rows])          # (5, ncom, 3)
                        Y = np.stack([np.hypot(XC[..., 0], XC[..., 1]), XC[..., 2]], -1)
                        ext = 2.0 * np.max(np.linalg.norm(XC - XC[2:3], axis=-1), axis=0)
                        extc = np.maximum.accumulate(ext)
                        for kk in np.argsort(-mg)[:n_cand]:
                            if not mg[kk] > 0:
                                break
                            g = lambda key: np.array([r_[key][kk] for r_ in rows])
                            ti, why = exit_test(Y[:, kk], g("EC"), g("FC"), g("DC"), extc[kk], g("SC"), eta, targets,
                                                ef)
                            stats["exit_tests"] = stats.get("exit_tests", 0) + 1
                            if ti is not None:
                                res[q] = ("exit", t0, t1, d_, int(kk) + 1, int(ti), float(why))
                                break
                            stats["exit_fail_" + why] = stats.get("exit_fail_" + why, 0) + 1
                mid = cache[(0.5 * (t0 + t1), d_)][0]
                if res[q] is None and bool(mid["lost"]):
                    progress[q] = True
                    s1, s2 = float(mid["sf"]), float(mid["se"])
                    if not np.isfinite(s1):
                        s1 = s2
                    ss = np.linspace(s1, s2, 4) if s2 > s1 else np.full(4, s2)
                    pa_ = p0 + t0 * (p1 - p0); pb_ = p0 + t1 * (p1 - p0); pm = 0.5 * (pa_ + pb_)
                    P3 = np.stack([pa_, pm, pb_]); P2 = np.stack([0.75 * pa_ + 0.25 * pb_, 0.25 * pa_ + 0.75 * pb_])
                    if d_ == -1:
                        P3 = P3 * _MIR; P2 = P2 * _MIR
                    sp = _seg_base_test(tr, sgn, P3, P2, ss, eta, tf)
                    stats["base_tests"] = stats.get("base_tests", 0) + 1
                    if sp is not None:
                        res[q] = ("lost", t0, t1, d_, float(sp), -1, 0.0)
        for q, (t0, t1, dep) in enumerate(batch):
            if res[q] is not None:
                done.append(res[q])
            elif not progress[q]:
                stats["seg_noprogress"] = stats.get("seg_noprogress", 0) + 1
                return None
            elif dep < max_split and n_pieces + len(todo) + 2 <= max_pieces:
                tm = 0.5 * (t0 + t1)
                todo += [(t0, tm, dep + 1), (tm, t1, dep + 1)]
            else:
                stats["seg_maxsplit"] = stats.get("seg_maxsplit", 0) + 1
                return None
    return done


# ============================================================================ the driver
def _comp4(M, ia, ja):
    """4-connected component of M containing (ia, ja) and whether it touches the border (None if M[ia, ja] false)."""
    from scipy import ndimage
    if not M[ia, ja]:
        return None, True
    lab, _ = ndimage.label(M)
    C = lab == lab[ia, ja]
    return C, bool(C[0].any() or C[-1].any() or C[:, 0].any() or C[:, -1].any())


def exit_curve_bound(source, axis_point, T_last=None, core=None, scale_poly=None, request=None, budget_s=600.0,
                     h_rel=1 / 16, W_exit=8, W_lost=4.0, d_fail_rel=0.03, eps=1e-10, tol=1e-10, reach_d=1.8,
                     max_split=9, max_routes=24, max_steps=60, verbose=False, sign=None, with_evidence=True,
                     det_share=0.35, beta_det=0.01, beta=0.02, hug=1.5, verified_sections=None, n_targets=4,
                     min_gain=0.01, evidence_s=90.0, hug0=0.2, edge_cap_frac=1 / 12, adversarial=8, lost_share=0.4,
                     start_curve=None, band=4.0, on_penalty=4.0, W_max=32, esc_frac=0.2, coil_steps=None):
    """The exit-curve bound (THEOREM_DOC) around axis_point (R, Z = 0 in phi = 0); returns a JSON-able record (ok,
    F_up, curve, steps, ...).  T_last: the outermost verified torus (core and scale; negative test), or core /
    scale_poly polygons (efficiency only: nodes inside the core are never tried, which never makes a bound wrong).
    The first curve C_0 is torux.edgebound.lost_curve_bound's (lost_share of the budget; or start_curve: a certified
    record of it); the exit curves march inward from it.
    coil_steps (TORUX_EXITCURVE_COILSTEPS=1; off by default): the march's tracers shorten a step whose ball reaches
    within d_fail of a coil instead of ending the line (torux.exitcurve_nearcoil: lines starting within ~ 2 cm of a
    coil are traced at all; no certificate changes)."""
    import heapq
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from .verify import torus_points
    from .barrier import signed_area, inside_polygon, closed_polygon_simple, winding_number
    t0 = time.time()
    deadline = t0 + budget_s
    a = np.asarray(axis_point, float)
    out = dict(kind="exit_curve", ok=False, axis_point=a.tolist(), request=request, rigorous=True)
    why = supported(source)
    if why is not None:
        out["reason"] = why
        return out
    if not bool(getattr(source, "stellsym", True)):
        out["reason"] = "the exit curve needs stellarator symmetry (the mirror half)"
        return out
    cp = np.concatenate([p.points(np.arange(1024) / 1024)[0].reshape(-1, 3) for p in source.parts
                         if p.kind == "fourier"])
    tree = cKDTree(cp)
    sec_last = None
    if T_last is not None:
        th = 2 * np.pi * np.arange(2048) / 2048
        Rl, Zl = T_last.geometry_on(th, np.zeros_like(th))
        sec_last = np.stack([Rl, Zl], -1)
        tt, pp = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                             np.linspace(0, 2 * np.pi / source.nfp, 32), indexing="ij")
        xt, _, _ = torus_points(T_last, tt.ravel(), pp.ravel())
        d_last = float(np.min(tree.query(xt)[0]))
        core = sec_last if core is None else core
        out["F_last"] = abs(float(T_last.flux))
    sp_ = np.asarray(core if scale_poly is None else scale_poly, float)
    r_max = float(np.max(np.hypot(sp_[:, 0] - a[0], sp_[:, 1] - a[1])))
    r_eff = math.sqrt(abs(signed_area(sp_)) / math.pi)
    if T_last is None:
        d_last = float(np.min(tree.query(np.stack([sp_[:, 0], 0 * sp_[:, 0], sp_[:, 1]], -1))[0]))
    d_fail = d_fail_rel * d_last
    if coil_steps is None:
        import os
        coil_steps = os.environ.get("TORUX_EXITCURVE_COILSTEPS", "0") == "1"
    TR = ExactTracer
    if coil_steps:
        from .exitcurve_nearcoil import NearCoilTracer as TR
    tr = TR(source, source.nfp, 1.0, d_fail=d_fail, tol=tol, eps=eps, beta=beta)
    if sign is None:
        sign = float(np.sign(tr.B(np.array([[a[0], 0.0, a[1]]]))[0, 1]))
    tr.s = sign
    det = TR(source, source.nfp, sign, d_fail=d_fail, tol=1e-5, eps=1e-6, beta=beta_det)
    out["coil_steps"] = bool(coil_steps)
    tr.check()
    parc = 2 * math.pi * a[0] / source.nfp
    out.update(d_last=d_last, d_fail=d_fail, check_err_multilevel=tr.check_err, sign=sign, W_exit=W_exit,
               W_lost=W_lost, h_rel=h_rel)
    # (O2) and (O3) use the source's nfp and stellarator symmetry: measured in every run (relative defect of B at 24
    # random points of the box against B at their rotated / reflected images); a run on an asymmetric field is refused
    sym_def = symmetry_defect(tr, source.nfp, a, r_max + reach_d * d_last)
    out["symmetry_defect"] = sym_def
    if not (sym_def["nfp"] < 1e-9 and sym_def["stellsym"] < 1e-9):
        out["reason"] = f"the field is not nfp / stellarator symmetric to 1e-9 ({sym_def})"
        return out
    # node grid: the upper half (row 0 on the symmetry line), mirrored for the topology
    h = h_rel * r_eff
    ext = r_max + reach_d * d_last
    i0 = int(math.ceil(ext / h))
    Rn = a[0] + h * np.arange(-i0, i0 + 1)
    Zn = h * np.arange(0, i0 + 1)
    RR, ZZ = np.meshgrid(Rn, Zn, indexing="ij")
    nR_, nZ_ = RR.shape
    incore = _poly_in_core(core, RR, ZZ)
    cand = ~incore & (RR > 0.02)
    idx = np.flatnonzero(cand.ravel())
    X = np.stack([RR.ravel()[idx], 0 * idx, ZZ.ravel()[idx]], -1)
    # C_0: the lost curve (torux.edgebound.lost_curve_bound, its own detection, route and tightening)
    lc = start_curve
    if lc is None and lost_share > 0:
        from .edgebound import lost_curve_bound
        lc = lost_curve_bound(source, a, T_last=T_last, core=core, scale_poly=scale_poly, request=request,
                              budget_s=lost_share * budget_s, h_rel=h_rel, verbose=False, sign=sign,
                              with_evidence=with_evidence)
    if lc is not None:
        out["lost_curve"] = {k_: v_ for k_, v_ in lc.items() if k_ not in ("curve",)}
    # DETECTION (selection only): forward lines of the nodes and of their mirror images (= backward lines), loose rules,
    # W_exit crossings or lost; outermost first, in chunks between deadline checks
    t1 = time.time()
    nI = len(idx)
    dlost = np.full((2, nI), np.inf)
    dimg = np.full((2, nI, W_exit, 2), np.nan)
    dxe = np.full((2, nI, 3), np.nan)
    order = np.argsort(-np.hypot(X[:, 0] - a[0], X[:, 2] - a[1]))
    n_det = 0
    t_det_end = time.time() + det_share * max(deadline - time.time(), 0.0)
    held0 = bool(lc is not None and lc.get("ok") and request is not None and lc["F_up"] < abs(request))
    for lo in range(0, nI, 2048):
        if time.time() > t_det_end or held0:
            break
        sel = order[lo:lo + 2048]
        for d_, XX in enumerate((X[sel], X[sel] * _MIR)):
            o = multicross(det, XX, W_exit, max(W_exit, W_lost) * parc * 3.0)
            dlost[d_, sel] = np.where(o["lost"], o["se"] / parc, np.inf)
            dimg[d_, sel] = np.stack([np.hypot(o["XC"][..., 0], o["XC"][..., 1]), o["XC"][..., 2]], -1)
            dxe[d_, sel] = o["XE"]
        n_det = lo + len(sel)
    out["detection"] = dict(nodes=int(nI), traced=int(n_det), t=time.time() - t1, h=h,
                            lost=int(np.sum(np.min(dlost, 0) <= W_lost)))
    if verbose:
        print(f"  detection {n_det}/{nI} nodes in {time.time() - t1:.0f} s, lost {out['detection']['lost']}", flush=True)

    def full(M_half):
        """(nR_, nZ_) upper-half map -> (nR_, 2 nZ_ - 1) full map (mirror rows)."""
        return np.concatenate([M_half[:, :0:-1], M_half], axis=1)
    ja_full = nZ_ - 1
    good_edges = {}          # permanent: (node, node) -> certified pieces (every point in O)
    curves = []              # certified curves: dict(curve, F_up, flux, step, ...)
    targets = []             # (Cx, Cy) of the certified curves, newest first
    target_ids = []          # their indices in curves
    stats = dict(routes=0)
    best = None
    lost_node = np.zeros(RR.size, bool); lost_node[idx] = np.min(dlost, 0) <= W_lost
    lost_dir = np.zeros(RR.size, int)
    lost_dir[idx] = np.where(dlost[0] <= dlost[1], 1, -1)
    dl_min = np.full(RR.size, np.inf); dl_min[idx] = np.min(dlost, 0)
    pos2 = np.full(RR.size, -1, np.int64); pos2[idx] = np.arange(nI); pos2 = pos2.reshape(RR.shape)
    onO = np.zeros(RR.shape, bool)          # nodes on a certified curve (in O): always usable by a route
    hist = []
    bfull = full(_bphi_grid(tr, RR, ZZ, sign))
    if lc is not None and lc.get("ok") and lc.get("curve"):
        C0 = np.asarray(lc["curve"], float)
        if signed_area(C0) < 0:
            C0 = C0[::-1]
        curves.append(dict(step=-1, curve=C0, F_up=float(lc["F_up"]), flux=lc.get("flux"), path=[], n_exit=0,
                           n_lost=int(lc.get("pieces") or 0), n_outside=0, t=time.time() - t0, routes=lc.get("routes"),
                           kind="lost_curve"))
        best = curves[-1]
        targets = [(np.ascontiguousarray(C0[:, 0]), np.ascontiguousarray(C0[:, 1]))]
        target_ids = [0]
        # C_0's vertices are nodes of this grid (the same h, box and origin as lost_curve_bound's): its certified
        # edges are reusable (every point in O)
        ii = np.rint((C0[:, 0] - Rn[0]) / h).astype(int); jj = np.rint(C0[:, 1] / h).astype(int)
        on = (np.abs(C0[:, 0] - Rn[0] - ii * h) < 1e-9 * h) & (np.abs(C0[:, 1] - jj * h) < 1e-9 * h)
        if np.all(on):
            nodes0 = list(zip(ii.tolist(), jj.tolist()))
            for q in range(len(nodes0)):
                n0, n1 = nodes0[q], nodes0[(q + 1) % len(nodes0)]
                if n0[1] >= 0 and n1[1] >= 0 and max(abs(n0[0] - n1[0]), abs(n0[1] - n1[1])) == 1:
                    pc = [("lost_curve", 0.0, 1.0, 0, 0, 0, 0.0)]
                    good_edges[(n0, n1)] = pc; good_edges[(n1, n0)] = pc
            for n_ in nodes0:
                if n_[1] >= 0:
                    onO[n_] = True
        out["C0_edges_reused"] = len(good_edges) // 2
        hist.append(dict(step=-1, kind="lost_curve", F_up=float(lc["F_up"]), t=time.time() - t0))
    band_cur = float(band)
    W_cur = int(W_exit)

    def escalate():
        """The march stalled with time left: the window doubled (up to W_max) and the detection redone for the nodes
        inside the newest curve (the others are outside it: k = 0).  Returns True when it was done."""
        nonlocal W_cur, dimg, band_cur
        if not targets or W_cur >= W_max or time.time() > deadline - esc_frac * budget_s:
            return False
        Cx_, Cy_ = targets[0]
        ins = signed_margin(Cx_, Cy_, np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 2])) < 0
        sel_ = np.flatnonzero(ins)
        Wn = 2 * W_cur
        new = np.full((2, nI, Wn, 2), np.nan)
        new[:, :, :W_cur] = dimg
        t_e = time.time()
        for d_, XX in enumerate((X[sel_], X[sel_] * _MIR)):
            o = multicross(det, XX, Wn, Wn * parc * 3.0)
            dlost[d_, sel_] = np.where(o["lost"], o["se"] / parc, np.inf)
            new[d_, sel_] = np.stack([np.hypot(o["XC"][..., 0], o["XC"][..., 1]), o["XC"][..., 2]], -1)
            dxe[d_, sel_] = o["XE"]
        dimg = new
        W_cur = Wn
        band_cur = float(band)
        out.setdefault("escalations", []).append(dict(W=Wn, nodes=int(len(sel_)), t=time.time() - t_e))
        if verbose:
            print(f"  window -> {Wn} crossings: {len(sel_)} nodes re-traced in {time.time() - t_e:.0f} s", flush=True)
        return True

    def _escalate_safe():
        try:
            return escalate()
        except Exception as e:  # noqa: BLE001  (selection only: a failure ends the march, never a certificate)
            out["escalation_error"] = repr(e)
            return False

    for step in range(max_steps):
        if time.time() > deadline - 0.03 * budget_s:
            break
        if best is not None and request is not None and best["F_up"] < abs(request):
            hist.append(dict(step=step, stop="request held"))
            break
        # exit-looking nodes for the current targets: a crossing (either direction) outside a target polygon by > h / 4
        mrg = np.full(RR.size, -np.inf)
        mdir = np.zeros(RR.size, int)
        kbest = np.zeros(RR.size, int)
        if targets:
            Y = dimg.reshape(-1, 2)
            m_all = np.full(len(Y), -np.inf)
            for (Cx, Cy) in targets:
                m_all = np.maximum(m_all, np.nan_to_num(signed_margin(Cx, Cy, np.ascontiguousarray(Y[:, 0]),
                                                                      np.ascontiguousarray(Y[:, 1])), nan=-np.inf))
            m_all = m_all.reshape(2, nI, dimg.shape[2])
            mbest = np.max(m_all, axis=2)                    # (2, nI)
            mrg[idx] = np.max(mbest, axis=0)
            mdir[idx] = np.where(mbest[0] >= mbest[1], 1, -1)
            kb = np.argmax(m_all, axis=2)                    # (2, nI) the best crossing per direction
            kbest[idx] = np.where(mbest[0] >= mbest[1], kb[0], kb[1])
            # k = 0: the node itself outside a certified curve
            m0 = np.full(RR.size, -np.inf)
            for (Cx, Cy) in targets:
                m0 = np.maximum(m0, signed_margin(Cx, Cy, np.ascontiguousarray(RR.ravel()),
                                                  np.ascontiguousarray(ZZ.ravel())))
            mrg = np.maximum(mrg, m0)
        lostm = np.zeros(RR.size, bool); lostm[idx] = np.min(dlost, 0) <= W_lost
        ex_ok = mrg > 0.25 * h
        exit_ = ex_ok | lostm
        # a node's badness for the route: exit nodes by their margin, lost nodes by their escape time (periods)
        bad = np.minimum(np.where(ex_ok, 1.0 / (0.25 + np.maximum(mrg, 0.0) / h), np.inf),
                         np.where(lostm, 0.25 + 0.5 * dl_min, np.inf))
        dirn = np.where(ex_ok & ~(lostm & (0.25 + 0.5 * dl_min < 1.0 / (0.25 + np.maximum(mrg, 0.0) / h))), mdir,
                        lost_dir)
        mode_exit = (ex_ok & ~(lostm & (0.25 + 0.5 * dl_min < 1.0 / (0.25 + np.maximum(mrg, 0.0) / h)))).reshape(RR.shape)
        exit_ = exit_.reshape(RR.shape); bad2 = bad.reshape(RR.shape); dirn = dirn.reshape(RR.shape)
        kb2 = kbest.reshape(RR.shape)
        onO_use = onO
        if targets:
            # a step's ambition: nodes inside the newest curve no deeper than `band` node spacings (its exits are then
            # short, cheap to certify), never outside it: every new curve lies inside-or-on the newest (nodes on a
            # certified curve inside-or-on it are always usable: their edges are certified)
            Cx_, Cy_ = targets[0]
            sm = signed_margin(Cx_, Cy_, np.ascontiguousarray(RR.ravel()), np.ascontiguousarray(ZZ.ravel()))
            sm = sm.reshape(RR.shape)
            exit_ &= (sm >= -band_cur * h) & (sm <= 1e-9 * h)
            onO_use = onO & (sm <= 1e-9 * h)
        exit_ |= onO_use
        onN = (onO_use & (sm >= -1e-9 * h)) if targets else np.zeros(RR.shape, bool)     # nodes on the newest curve
        bad2 = np.where(onO, np.minimum(bad2, 0.25), bad2)

        def spread(n0, n1, L):
            """How differently the two nodes' lines behave, over their distance (detection data): the distance of
            their images at the better node's exit crossing, or of their lost points; 10 when they differ in kind or
            direction."""
            if (n0, n1) in good_edges:
                return 0.0
            q0 = pos2[n0]; q1 = pos2[n1]
            if q0 < 0 or q1 < 0:
                return 10.0
            d0 = 0 if dirn[n0] == 1 else 1
            if dirn[n0] != dirn[n1] or mode_exit[n0] != mode_exit[n1]:
                return 10.0
            if mode_exit[n0]:
                k_ = kb2[n0] if bad2[n0] <= bad2[n1] else kb2[n1]
                dd = dimg[d0, q0, k_] - dimg[d0, q1, k_]
            else:
                dd = dxe[d0, q0] - dxe[d0, q1]
            v = float(np.sqrt(np.sum(dd * dd))) / L
            return min(v, 50.0) if np.isfinite(v) else 10.0
        Kc, border = _comp4(full(~exit_), i0, ja_full)
        if Kc is None or border:
            hist.append(dict(step=step, stop="no closed exit set around the axis"))
            break
        Kc = ndimage.binary_fill_holes(Kc)
        F_est = float(np.sum(np.maximum(bfull[Kc], 0.0)) * h * h)
        hist.append(dict(step=step, F_est=F_est, exit_nodes=int(exit_.sum()), K_nodes=int(Kc.sum()),
                         t=time.time() - t0))
        if best is not None and band_cur == band and F_est > (1.0 - min_gain) * best["F_up"]:
            hist[-1]["stop"] = "no gain expected"
            break
        if len(hist) >= 2 and hist[-2].get("K_nodes") == hist[-1]["K_nodes"]:
            if _escalate_safe():
                hist[-1]["escalate"] = W_cur
                continue
            hist[-1]["stop"] = "converged (the survivors did not change)"
            break
        dK = ndimage.distance_transform_edt(~Kc)[:, ja_full:]          # upper half, node units
        hw = hug if targets else hug0          # the first (lost) curve robust, the exit curves hugging the survivors
        blocked = set(); blocked_nodes = set()

        def route():
            INF = np.inf
            dist = np.full(RR.shape, INF)
            prev = {}
            pq = []
            for i in range(0, i0):
                if exit_[i, 0] and dK[i, 0] < 1e9:
                    dist[i, 0] = 0.0
                    heapq.heappush(pq, (0.0, i, 0))
            while pq:
                dcur, i, j = heapq.heappop(pq)
                if dcur > dist[i, j]:
                    continue
                if j == 0 and i > i0:
                    path = [(i, j)]
                    while path[-1] in prev:
                        path.append(prev[path[-1]])
                    return path[::-1]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        a_, b_ = i + di, j + dj
                        if not (0 <= a_ < nR_ and 0 <= b_ < nZ_) or not exit_[a_, b_] or (a_, b_) in blocked_nodes:
                            continue
                        if b_ == 0 and not a_ > i0:
                            continue
                        if j == 0 and b_ == 0:
                            continue
                        e = ((i, j), (a_, b_))
                        if e in blocked:
                            continue
                        L = h * math.hypot(di, dj)
                        c = L * (1.0 + hw * max(0.0, 0.5 * (dK[i, j] + dK[a_, b_]) - 1.0)) \
                            * (1.0 + 0.5 * (bad2[i, j] + bad2[a_, b_])) * (1.0 + spread((i, j), (a_, b_), L))
                        if onN[i, j] and onN[a_, b_]:
                            c *= on_penalty         # along the newest curve: only where no inner edge certifies
                        elif e in good_edges:
                            c *= 0.5
                        nd = dcur + c
                        if nd < dist[a_, b_]:
                            dist[a_, b_] = nd
                            prev[(a_, b_)] = (i, j)
                            heapq.heappush(pq, (nd, a_, b_))
            return None

        path = None
        n_r = 0
        while n_r < max_routes and time.time() < deadline - 0.03 * budget_s:
            n_r += 1
            stats["routes"] += 1
            pth = route()
            if pth is None:
                break
            failed = None
            todo_e = [(e0, e1) for (e0, e1) in zip(pth[:-1], pth[1:]) if (e0, e1) not in good_edges]
            if todo_e:
                if time.time() > deadline - 0.03 * budget_s:
                    failed = "time"
                else:
                    segs = [(np.array([RR[e0], 0.0, ZZ[e0]]), np.array([RR[e1], 0.0, ZZ[e1]])) for (e0, e1) in todo_e]
                    dsg = [int((dirn[e0] if dirn[e0] != 0 else dirn[e1]) or 1) for (e0, e1) in todo_e]
                    t_cap = deadline - 0.03 * budget_s
                    rlist = certify_edges(tr, sign, segs, dsg, targets, W_cur, parc, max_split, t_cap, stats)
                    for (e0, e1), pcs in zip(todo_e, rlist):
                        if pcs is None:
                            blocked.add((e0, e1)); blocked.add((e1, e0))
                            worse = e0 if bad2[e0] > bad2[e1] else e1
                            if worse[1] > 0:
                                blocked_nodes.add(worse)
                            failed = (e0, e1)
                        elif isinstance(pcs, list):
                            pcs = [pc[:5] + ((target_ids[pc[5]] if pc[5] >= 0 else -1),) + pc[6:] for pc in pcs]
                            good_edges[(e0, e1)] = pcs; good_edges[(e1, e0)] = pcs
                        elif failed is None:
                            failed = "pending"
                    if failed == "pending" and time.time() > deadline - 0.03 * budget_s:
                        failed = "time"
            if failed is None:
                path = pth
                break
            if failed == "time":
                break
        if path is None:
            if band_cur > 1.0 and targets and time.time() < deadline - 0.05 * budget_s:
                band_cur = max(1.0, 0.5 * band_cur)          # a less ambitious step
                hist[-1]["retry_band"] = band_cur
                continue
            if _escalate_safe():
                hist[-1]["escalate"] = W_cur
                continue
            hist[-1]["stop"] = "no certified route"
            break
        band_cur = float(band)
        upper = np.array([[RR[q], ZZ[q]] for q in path])
        curve = np.concatenate([upper, upper[-2:0:-1] * np.array([1.0, -1.0])])
        if signed_area(curve) < 0:
            curve = curve[::-1]
        if not closed_polygon_simple(curve, 1e-12 * h) or abs(winding_number(curve, a)) != 1:
            hist[-1]["stop"] = "route not a simple closed curve around the axis"
            break
        F_up, fl = curve_flux_bound(tr, source, curve, sign, h)
        kinds = [pc[0] for e in zip(path[:-1], path[1:]) for pc in good_edges[e]]
        rec = dict(step=step, curve=curve, F_up=F_up, flux=fl, path=path, n_exit=kinds.count("exit"),
                   n_lost=kinds.count("lost"), n_outside=kinds.count("outside"), t=time.time() - t0, routes=n_r)
        curves.append(rec)
        for n_ in path:
            onO[n_] = True
        hist[-1].update(F_up=F_up, n_exit=rec["n_exit"], n_lost=rec["n_lost"], n_outside=rec["n_outside"], routes=n_r)
        if verbose:
            print(f"  step {step}: F_up {F_up:.5g} (est {F_est:.5g}), {rec['n_exit']} exit + {rec['n_lost']} lost "
                  f"pieces, {n_r} routes, t {time.time() - t0:.0f} s", flush=True)
        if best is None or F_up < best["F_up"]:
            best = rec
        targets = ([(np.ascontiguousarray(curve[:, 0]), np.ascontiguousarray(curve[:, 1]))] + targets)[:n_targets]
        target_ids = ([len(curves) - 1] + target_ids)[:n_targets]
        if request is not None and F_up < abs(request):
            hist[-1]["stop"] = "request held"
            break
    out["steps"] = hist
    out["routes"] = stats["routes"]
    out["seg_stats"] = {k: v for k, v in stats.items() if k != "routes"}
    out["lines_traced"] = tr.n_lines
    out["timings"] = dict(total=time.time() - t0, tracer=tr.t, detection=det.t)
    if best is None:
        out.setdefault("reason", "no certified curve within the budget")
        return out
    curve = best["curve"]
    out.update(ok=True, F_up=float(best["F_up"]), flux=best["flux"], curve=curve.tolist(), n_vertices=len(curve),
               n_curves=len(curves), F_up_steps=[float(c["F_up"]) for c in curves],
               curves_all=[np.asarray(c["curve"]).tolist() for c in curves],
               F_up_first=float(curves[0]["F_up"]), pieces_exit=best["n_exit"], pieces_lost=best["n_lost"])
    secs = [np.asarray(s_, float) for s_ in (verified_sections or [])]
    if sec_last is not None:
        secs.append(sec_last)
    if secs:
        # NEGATIVE TEST: every verified torus section inside every certified curve, every F_up above its flux
        outside = 0
        for c in curves:
            for sec in secs:
                outside += int(np.sum(~inside_polygon(c["curve"], sec)))
        out["verified_section_outside"] = outside
        # exact: the smallest distance between a certified curve and a verified section (must be > 0)
        dmin = np.inf
        for c in curves:
            C = c["curve"]
            for sec in secs:
                dmin = min(dmin, _poly_poly_dist(np.ascontiguousarray(C), np.ascontiguousarray(sec)))
        out["verified_section_distance"] = float(dmin)
        if not dmin > 0:
            outside += 1
        # ADVERSARIAL: chords of the verified sections (their points are on invariant tori, in no O) tried as exit /
        # lost pieces against the certified curves; none may be certified
        if adversarial and curves:
            rng = np.random.default_rng(1)
            tg = [(np.ascontiguousarray(c["curve"][:, 0]), np.ascontiguousarray(c["curve"][:, 1]))
                  for c in curves[::-1][:n_targets]]
            n_adv = 0
            st_adv = {}
            dl_adv = time.time() + min(120.0, 0.1 * budget_s)
            for sec in secs:
                up = sec[sec[:, 1] > 0]
                # chords of length ~ h / 4 (within ~ h^2 kappa / 128 of the torus: deep in no exit's reach)
                st_ = max(1, int(round(0.25 * h / max(float(np.median(np.linalg.norm(np.diff(up, axis=0), axis=1))),
                                                       1e-300))))
                for k_ in rng.permutation(max(len(up) - st_, 1))[:adversarial]:
                    if time.time() > dl_adv or k_ + st_ >= len(up):
                        break
                    q0 = np.array([up[k_, 0], 0.0, up[k_, 1]])
                    q1 = np.array([up[k_ + st_, 0], 0.0, up[k_ + st_, 1]])
                    pcs = _certify_exit_segment(tr, sign, q0, q1, 1, tg, W_exit, parc, 3, dl_adv, st_adv,
                                                max_pieces=8)
                    n_adv += 1
                    if pcs is not None:
                        outside += 1
                        out.setdefault("adversarial_certified", []).append([q0.tolist(), q1.tolist()])
            out["adversarial_tried"] = n_adv
        if T_last is not None:
            out["ratio"] = out["F_up"] / abs(float(T_last.flux))
        if outside or (T_last is not None and min(c["F_up"] for c in curves) < abs(float(T_last.flux))):
            out.update(ok=False, false_certificate=True,
                       reason="FALSE CERTIFICATE: a verified torus is not inside a certified curve, or F_up below it")
    if request is not None and out.get("ok"):
        out["holds_request"] = bool(out["F_up"] < abs(request))
    if with_evidence and out.get("ok"):
        try:
            out["evidence"] = exit_evidence(source, tr, curves, good_edges, RR, ZZ, sign, W_exit, parc,
                                            deadline=time.time() + evidence_s)
        except Exception as e:  # noqa: BLE001
            out["evidence"] = dict(error=repr(e))
    out["timings"]["total"] = time.time() - t0
    return out


def symmetry_defect(tr, nfp, a, ext, n=24, seed=3):
    """Largest relative difference |B(Rot x) - Rot B(x)| / |B| (Rot: rotation by 2 pi / nfp about z) and
    |B(S x) - S* B(x)| / |B| (S (x, y, z) -> (x, -y, -z), B -> (-B_x, B_y, B_z)) at n random points of the box
    (radius ext about the axis point, random toroidal angle) farther than 2 d_fail from the coils."""
    rng = np.random.default_rng(seed)
    R = a[0] + ext * rng.uniform(-1, 1, n); Z = ext * rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n)
    R = np.maximum(R, 0.05 * a[0])
    X = np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)
    c, s_ = math.cos(2 * math.pi / nfp), math.sin(2 * math.pi / nfp)
    Rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    o = tr.B(X); B = o[:, :3]
    ok = o[:, 3] > 2 * tr.d_fail
    oR = tr.B(X @ Rot.T); BR = oR[:, :3]
    Sx = X * np.array([1.0, -1.0, -1.0])
    oS = tr.B(Sx); BS = oS[:, :3]
    ok &= (oR[:, 3] > 2 * tr.d_fail) & (oS[:, 3] > 2 * tr.d_fail)
    nb = np.linalg.norm(B, axis=1)
    dR = np.linalg.norm(BR - B @ Rot.T, axis=1) / nb
    dS = np.linalg.norm(BS - B * np.array([-1.0, 1.0, 1.0]), axis=1) / nb
    return dict(nfp=float(np.max(dR[ok])) if ok.any() else np.inf, stellsym=float(np.max(dS[ok])) if ok.any() else np.inf,
                n=int(ok.sum()))


@njit(cache=True)
def _poly_poly_dist(C, S):
    """Smallest distance between the edges of the closed polygons C (n, 2) and S (m, 2)."""
    n = C.shape[0]; m = S.shape[0]
    best = 1e300
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(m):
            j2 = (j + 1) % m
            d = _seg_seg_dist(C[i, 0], C[i, 1], C[i2, 0], C[i2, 1], S[j, 0], S[j, 1], S[j2, 0], S[j2, 1])
            if d < best:
                best = d
    return best


def _poly_in_core(core, RR, ZZ):
    from .barrier import inside_polygon
    if core is None:
        return np.zeros(RR.shape, bool)
    return inside_polygon(np.asarray(core, float), np.stack([RR.ravel(), ZZ.ravel()], -1)).reshape(RR.shape)


def _bphi_grid(tr, RR, ZZ, sign):
    B = tr.B(np.stack([RR.ravel(), 0 * RR.ravel(), ZZ.ravel()], -1))[:, :3]
    return (sign * B[:, 1]).reshape(RR.shape)


# ============================================================================ evidence (independent checks)
def exit_evidence(source, tr, curves, good_edges, RR, ZZ, sign, W, parc, n_exit=6, n_lost=3, seed=0, deadline=None):
    """Independent checks of the certified curves (recorded; any failure is a refusal of the record by the caller's
    soundness tests, never part of the bound):
      exit pieces  (sampled) the midpoint line re-traced by scipy DOP853 in arc length on the reference field
                   (torux.reference.fast64) to its k-th plane crossing (event on the unwrapped angle): its distance to
                   our crossing against the error bound 2 x summed GBS estimates / f, and its signed distance outside
                   the target polygon (must be > 0); 9 extra lines of the piece (exact tracer) to the k-th crossing:
                   every image outside the target polygon (the enclosure model checked);
      lost pieces  (sampled) torux.edgebound.curve_evidence's checks (reference re-trace: g < 0; 9 extra lines);
      flux         -s x the loop integral of the reference field's vector potential along the final curve (Gauss-
                   Legendre 8 per edge) against the Stokes value of the bound."""
    from .reference.fast64 import Field
    from .edgebound import _ref_cross, curve_evidence
    rng = np.random.default_rng(seed)
    F = Field(source)
    out = {}
    items = []
    seen = set()
    final = min(curves, key=lambda c: c["F_up"])          # the curve of the bound
    path_edges = list(zip(final["path"][:-1], final["path"][1:]))
    for e in path_edges:
        for pc in good_edges[e]:
            key = (e, pc[1], pc[2])
            if key not in seen:
                seen.add(key)
                items.append((e, pc))
    ex = [it for it in items if it[1][0] == "exit"]
    lo = [it for it in items if it[1][0] == "lost"]
    # target polygons by index: the certified curves before the one whose pieces they are
    rows = []
    for t in (rng.permutation(len(ex))[:n_exit] if ex else []):
        if deadline is not None and time.time() > deadline:
            break
        (e0, e1), (kind, t0, t1, d_, k, ti, mg) = ex[t]
        p0 = np.array([RR[e0], 0.0, ZZ[e0]]); p1 = np.array([RR[e1], 0.0, ZZ[e1]])
        tg = curves[ti] if 0 <= ti < len(curves) else None        # the target curve of the piece's certificate
        S_ev = max(3.0 * k, 96.0) * parc         # the lines stop at their k-th crossing; arc length only a cap
        pm = (p0 + 0.5 * (t0 + t1) * (p1 - p0)) * (_MIR if d_ == -1 else 1.0)
        o = multicross(tr, pm[None], k, S_ev)
        Xk = o["XC"][0, k - 1]
        est = 2.0 * float(o["EC"][0, k - 1]) / max(float(o["FC"][0, k - 1]), F_SMOOTH) + 1e-12
        xr = _ref_cross(F.B, pm, sign, 2 * math.pi * k / source.nfp, S_ev)
        row = dict(edge=[list(e0), list(e1)], t=[float(t0), float(t1)], dir=int(d_), k=int(k), margin=float(mg),
                   est=est, target=int(ti), reached=bool(o["nc"][0] >= k))
        if xr is not None:
            row["dist_ref"] = float(np.linalg.norm(xr - Xk))
            row["ratio"] = row["dist_ref"] / (est + 1e-10)
            if tg is not None:
                C = tg["curve"]
                row["ref_margin"] = float(signed_margin(np.ascontiguousarray(C[:, 0]), np.ascontiguousarray(C[:, 1]),
                                                        np.array([math.hypot(xr[0], xr[1])]), np.array([xr[2]]))[0])
        tt = np.linspace(t0, t1, 9)
        P = (p0[None] + tt[:, None] * (p1 - p0)[None]) * (_MIR if d_ == -1 else 1.0)
        o9 = multicross(tr, P, k, S_ev)
        if tg is not None and np.all(o9["nc"] >= k):
            C = tg["curve"]
            Y = o9["XC"][:, k - 1]
            m9 = signed_margin(np.ascontiguousarray(C[:, 0]), np.ascontiguousarray(C[:, 1]),
                               np.ascontiguousarray(np.hypot(Y[:, 0], Y[:, 1])), np.ascontiguousarray(Y[:, 2]))
            row["extra_min_margin"] = float(np.min(m9))
            row["extra_all_outside"] = bool(np.all(m9 > 0))
        rows.append(row)
    out["exit"] = rows
    out["exit_all_ok"] = bool(all(r_.get("extra_all_outside", False) and r_.get("ref_margin", 1.0) > 0 for r_ in rows))
    out["exit_ratio_max"] = max((r_["ratio"] for r_ in rows if "ratio" in r_), default=None)
    if lo:
        cert = {}
        for t in rng.permutation(len(lo))[:n_lost]:
            (e0, e1), pc = lo[t]
            cert.setdefault((e0, e1), []).append((pc[1], pc[2], pc[3], pc[4]))
        nxy = {q: (float(RR[q]), float(ZZ[q])) for e in cert for q in e}
        out["lost"] = curve_evidence(source, tr, cert, sign, nxy, n=n_lost, deadline=deadline)
    # the flux by the reference potential
    C = final["curve"]
    E = np.concatenate([C, np.roll(C, -1, axis=0)], axis=1)
    xg, wg = np.polynomial.legendre.leggauss(8)
    tq = 0.5 * (xg + 1); wq = 0.5 * wg
    Pq = E[:, None, :2] + tq[None, :, None] * (E[:, None, 2:] - E[:, None, :2])
    Xq = np.stack([Pq[..., 0], 0 * Pq[..., 0], Pq[..., 1]], -1).reshape(-1, 3)
    A = F.evaluate(Xq, grad=False)["A"].reshape(len(E), 8, 3)
    dd = E[:, 2:] - E[:, :2]
    Fref = -float(sign) * float(np.sum(wq[None] * (A[..., 0] * dd[:, None, 0] + A[..., 2] * dd[:, None, 1])))
    out["flux_reference"] = Fref
    out["flux_rel_diff"] = abs(Fref - final["flux"]["F"]) / max(abs(final["flux"]["F"]), 1e-300)
    return out



# ============================================================================ the analysis hook
EC_CAP = 600.0          # seconds an analysis gives the exit-curve march at most (TORUX_EXITCURVE_CAP), after the lost curve


def edge_bound_exit(eng, source, T_last, request=None, deadline=None, axis=None, note="", verbose=False, cap_s=None,
                    lost_cap_s=None, far=None, far_plane=None, **kw):
    """The exit-curve bound as a stage of torux.analysis.edge_bound, with torux.edgebound.edge_bound_survivor's interface
    and record (kind 'edge_survivor', ok, F_up, holds_request, interval, rigorous=True).  Two stages: the lost curve
    (torux.edgebound.lost_curve_bound, its own cap TORUX_EDGEBOUND_CAP, 600 s) -- returned at once when it holds the
    request -- then the march from it (exit_curve_bound with start_curve, cap TORUX_EXITCURVE_CAP, 600 s), whose bound
    is never above the lost curve's.  The axis point from `axis` or, without one, torux.barrier.find_axis_point inside
    T_last; each stage's budget min(cap, what is left - 30 s).
    far (TORUX_EXITCURVE_FAR=1): when the lost curve finds no closed curve in its default box, the far start
    (torux.exitcurve_far.far_start: C_0 in a box beyond the coils, then the march) in the plane far_plane
    (TORUX_EXITCURVE_FAR_PLANE: 'phi0', 'phi_half' or 'both' -- phi0 first, then phi = pi / nfp with what is left),
    cap TORUX_EXITCURVE_FAR_CAP (1800 s)."""
    import os
    from .edgebound import axis_point_of, lost_curve_bound, EB_CAP
    cap_l = float(os.environ.get("TORUX_EDGEBOUND_CAP", EB_CAP)) if lost_cap_s is None else float(lost_cap_s)
    cap = float(os.environ.get("TORUX_EXITCURVE_CAP", EC_CAP)) if cap_s is None else float(cap_s)

    def left(c):
        return c if deadline is None else min(c, deadline - time.time() - 30.0)
    if left(cap_l) < 30.0:
        return dict(ok=False, kind="edge_survivor", method="exit_curve", reason="no time left", note=note,
                    rigorous=True)
    a = axis_point_of(axis)
    if a is None:
        from .barrier import _setup_section, find_axis_point
        sec, R2 = _setup_section(eng, source, T_last)
        orb = find_axis_point(sec, R2[1] + 0.1 * (R2[0] - R2[1]), R2[0] - 0.1 * (R2[0] - R2[1]))
        if orb is None:
            return dict(ok=False, kind="edge_survivor", method="exit_curve", reason="axis point not found inside T_last",
                        note=note, rigorous=True)
        a = np.asarray(orb.pts[0], float)
    lc = lost_curve_bound(source, a, T_last=T_last, request=request, budget_s=left(cap_l), verbose=verbose)
    far = (os.environ.get("TORUX_EXITCURVE_FAR", "0") == "1") if far is None else bool(far)
    if lc.get("ok") and request is not None and lc.get("holds_request"):
        rec = lc
    elif far and not lc.get("ok") and left(60.0) >= 60.0:
        from .exitcurve_far import far_start, FAR_CAP
        cap_f = float(os.environ.get("TORUX_EXITCURVE_FAR_CAP", FAR_CAP))
        plane = far_plane or os.environ.get("TORUX_EXITCURVE_FAR_PLANE", "both")
        planes = ("phi0", "phi_half") if plane == "both" else (plane,)
        rec, t_f = None, time.time()
        for k_, pl in enumerate(planes):
            share = left(cap_f - (time.time() - t_f)) / (len(planes) - k_)
            if share < 60.0:
                break
            r_ = far_start(source, a, T_last=T_last, request=request, plane=pl, budget_s=share, verbose=verbose,
                           **kw)
            if rec is None or (r_.get("ok") and (not rec.get("ok") or r_["F_up"] < rec["F_up"])):
                rec = r_
            if rec.get("ok") and request is not None and rec.get("holds_request"):
                break
        rec = rec if rec is not None else lc
        rec["lost_curve_default_box"] = {k_: v_ for k_, v_ in lc.items() if k_ not in ("curve",)}
    elif left(cap) < 30.0:
        rec = lc
    else:
        rec = exit_curve_bound(source, a, T_last=T_last, request=request, budget_s=left(cap), verbose=verbose,
                               start_curve=lc, lost_share=0.0, **kw)
    rec["note"] = note
    rec["rigorous"] = True
    rec["method"] = rec.get("kind")
    rec["kind"] = "edge_survivor"
    if request is not None:
        rec["interval"] = (rec.get("F_up"), np.inf) if rec.get("ok") else None
    return rec


# ============================================================================ batched certification of many edges
class _PreArc:
    """Stands in for the tracer in torux.edgebound._seg_base_test: its arc() returns lines traced beforehand (in one
    parallel batch for many pieces), BGA() is the tracer's."""

    def __init__(self, tr, res):
        self.tr, self.res = tr, res

    def arc(self, X0, S):
        return self.res

    def BGA(self, X):
        return self.tr.BGA(X)


def certify_edges(tr, sgn, edges, dsgns, targets, W, parc, max_split, deadline, stats, n_cand=4, max_pieces=64,
                  max_fail=2, batch=128):
    """_certify_exit_segment for many edges at once (the pieces of all edges traced together, so the compiled tracer
    runs as many lines in parallel as there are pieces).  edges: list of (p0, p1) in Sigma (Z >= 0); dsgns: the
    first direction to try per edge.  Returns per edge the certified pieces, None (failed) or 'pending' (not finished:
    the deadline, or max_fail other edges failed first)."""
    ne = len(edges)
    res = [None] * ne
    state = ["active"] * ne
    todo = {e: [] for e in range(ne)}
    done = {e: [] for e in range(ne)}
    npc = [0] * ne
    L01 = [float(np.linalg.norm(p1 - p0)) for p0, p1 in edges]
    for e, (p0, p1) in enumerate(edges):
        for ti, (Cx, Cy) in enumerate(targets):
            mg0 = _segs_outside(Cx, Cy, np.array([[p0[0], p0[2]]]), np.array([[p1[0], p1[2]]]), np.zeros(1))
            if mg0 > 0:
                stats["outside_edges"] = stats.get("outside_edges", 0) + 1
                res[e] = [("outside", 0.0, 1.0, 0, 0, ti, float(mg0))]
                state[e] = "done"
                break
        if state[e] == "active":
            todo[e] = [(0.0, 1.0, 0)]
    cache = {}
    ef = max(tr.check_err or 0.0, tr.eps)
    tf = lambda S, lin, e_: 2.0 * ef * S * max(1.0, lin / max(e_, 1e-300))
    n_fail = 0

    def pt(e, t, d_):
        p0, p1 = edges[e]
        x = p0 + t * (p1 - p0)
        return x * _MIR if d_ == -1 else x

    def trace(keys, ncr):
        new = sorted({k_ for k_ in keys if k_ not in cache or cache[k_][1] < ncr})
        if not new:
            return
        X = np.array([pt(*k_) for k_ in new])
        o = multicross(tr, X, ncr, ncr * parc * 3.0)
        stats["lines"] = stats.get("lines", 0) + len(new)
        for m_, k_ in enumerate(new):
            cache[k_] = ({kk: vv[m_] for kk, vv in o.items()}, ncr)

    def mid_margins(row):
        n = int(row["nc"])
        if n < 1 or not targets:
            return np.zeros(0)
        Y = np.stack([np.hypot(row["XC"][:n, 0], row["XC"][:n, 1]), row["XC"][:n, 2]], -1)
        mg = np.full(n, -np.inf)
        for (Cx, Cy) in targets:
            mg = np.maximum(mg, signed_margin(Cx, Cy, np.ascontiguousarray(Y[:, 0]), np.ascontiguousarray(Y[:, 1])))
        return np.where(row["FC"][:n] >= F_SMOOTH, mg, -np.inf)

    while True:
        act = [e for e in range(ne) if state[e] == "active"]
        if not act:
            break
        if time.time() > deadline:
            for e in act:
                state[e] = "pending"
            stats["seg_timeout"] = stats.get("seg_timeout", 0) + 1
            break
        items = []                     # (edge, t0, t1, dep)
        for e in act:
            while todo[e] and len(items) < batch:
                t0_, t1_, dep = todo[e].pop(0)
                items.append((e, t0_, t1_, dep))
                npc[e] += 1
            if len(items) >= batch:
                break
        rs = [None] * len(items)
        prog = np.zeros(len(items), bool)
        for pass_ in (0, 1):
            left = [q for q in range(len(items)) if rs[q] is None]
            if not left:
                break
            dq = {q: (dsgns[items[q][0]] if pass_ == 0 else -dsgns[items[q][0]]) for q in left}
            trace([(items[q][0], 0.5 * (items[q][1] + items[q][2]), dq[q]) for q in left], W)
            need = {}
            mgs = {}
            for q in left:
                e, t0_, t1_, _ = items[q]
                mg = mid_margins(cache[(e, 0.5 * (t0_ + t1_), dq[q])][0])
                mgs[q] = mg
                if np.any(mg > 0):
                    prog[q] = True
                    need[q] = int(np.max(np.flatnonzero(mg > 0))) + 1
            if need:
                kmax = max(need.values())
                trace([(items[q][0], items[q][1] + f * (items[q][2] - items[q][1]), dq[q]) for q in need for f in _FR],
                      kmax)
            for q in need:
                e, t0_, t1_, _ = items[q]
                eta = 0.5 * L01[e] * (t1_ - t0_)
                rows = [cache[(e, t0_ + f * (t1_ - t0_), dq[q])][0] for f in _FR]
                ncom = int(min(r_["nc"] for r_ in rows))
                mg = mgs[q][:ncom]
                if ncom < 1:
                    continue
                XC = np.stack([r_["XC"][:ncom] for r_ in rows])
                Y = np.stack([np.hypot(XC[..., 0], XC[..., 1]), XC[..., 2]], -1)
                ext = 2.0 * np.max(np.linalg.norm(XC - XC[2:3], axis=-1), axis=0)
                extc = np.maximum.accumulate(ext)
                for kk in np.argsort(-mg)[:n_cand]:
                    if not mg[kk] > 0:
                        break
                    g = lambda key: np.array([r_[key][kk] for r_ in rows])
                    ti, why = exit_test(Y[:, kk], g("EC"), g("FC"), g("DC"), extc[kk], g("SC"), eta, targets, ef)
                    stats["exit_tests"] = stats.get("exit_tests", 0) + 1
                    if ti is not None:
                        rs[q] = ("exit", t0_, t1_, dq[q], int(kk) + 1, int(ti), float(why))
                        break
                    stats["exit_fail_" + why] = stats.get("exit_fail_" + why, 0) + 1
            # lost pieces: the five lines of every candidate traced to its four arc lengths in one batch
            lq = []
            for q in left:
                if rs[q] is not None:
                    continue
                e, t0_, t1_, _ = items[q]
                mid = cache[(e, 0.5 * (t0_ + t1_), dq[q])][0]
                if bool(mid["lost"]):
                    prog[q] = True
                    s1, s2 = float(mid["sf"]), float(mid["se"])
                    if not np.isfinite(s1):
                        s1 = s2
                    lq.append((q, np.linspace(s1, s2, 4) if s2 > s1 else np.full(4, s2)))
            if lq:
                X0 = []; S = []
                for q, ss in lq:
                    e, t0_, t1_, _ = items[q]
                    for f in (0.0, 0.5, 1.0, 0.25, 0.75):         # _seg_base_test's order: P3 (u = -1, 0, 1), P2
                        X0.append(pt(e, t0_ + f * (t1_ - t0_), dq[q]))
                        S.append(ss)
                Xc, ec, okl, dm = tr.arc(np.array(X0), np.array(S))
                stats["base_lines"] = stats.get("base_lines", 0) + len(X0)
                for m_, (q, ss) in enumerate(lq):
                    e, t0_, t1_, _ = items[q]
                    sl = slice(5 * m_, 5 * m_ + 5)
                    pre = _PreArc(tr, (Xc[sl], ec[sl], okl[sl], dm[sl]))
                    P3 = np.array([pt(e, t0_ + f * (t1_ - t0_), dq[q]) for f in (0.0, 0.5, 1.0)])
                    P2 = np.array([pt(e, t0_ + f * (t1_ - t0_), dq[q]) for f in (0.25, 0.75)])
                    sp = _seg_base_test(pre, sgn, P3, P2, ss, 0.5 * L01[e] * (t1_ - t0_), tf)
                    stats["base_tests"] = stats.get("base_tests", 0) + 1
                    if sp is not None:
                        rs[q] = ("lost", t0_, t1_, dq[q], float(sp), -1, 0.0)
        for q, (e, t0_, t1_, dep) in enumerate(items):
            if state[e] != "active":
                continue
            if rs[q] is not None:
                done[e].append(rs[q])
            elif not prog[q]:
                stats["seg_noprogress"] = stats.get("seg_noprogress", 0) + 1
                state[e] = "failed"
            elif dep < max_split and npc[e] + len(todo[e]) + 2 <= max_pieces:
                tm = 0.5 * (t0_ + t1_)
                todo[e] += [(t0_, tm, dep + 1), (tm, t1_, dep + 1)]
            else:
                stats["seg_maxsplit"] = stats.get("seg_maxsplit", 0) + 1
                state[e] = "failed"
        for e in range(ne):
            if state[e] == "active" and not todo[e]:
                state[e] = "done"
                res[e] = done[e]
            elif state[e] == "failed" and res[e] is None:
                n_fail += 1
                res[e] = None
                state[e] = "failed_counted"
        if n_fail >= max_fail:
            for e in range(ne):
                if state[e] == "active":
                    state[e] = "pending"
            break
    return [("pending" if state[e] == "pending" else res[e]) for e in range(ne)]
