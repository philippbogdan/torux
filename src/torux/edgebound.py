"""Upper bounds on the flux of every flux surface around the magnetic axis from certified LOST points: a closed curve
of lost points around the axis (lost_curve_bound, the default) or certified lost cells (edge_upper_bound, the
"survivor flux": every certified cell inside the wall lowers the bound).

    from torux.edgebound import lost_curve_bound, edge_upper_bound
    rec = lost_curve_bound(source, axis_point=(R_a, 0.0), T_last=torus, request=None, budget_s=300)
    rec["ok"], rec["F_up"]     # every covered torus encloses at most F_up

The analysis hook is edge_bound_survivor; the lost-curve theorem and its numerics are in LOST_CURVE_DOC below,
the
survivor-flux theorem here.

Setting (as in torux/barrier.py)
--------------------------------------------------
Sigma is the half plane phi = 0 (a stellarator-symmetry plane), s the sign of B_phi on the magnetic axis, a the
axis point in Sigma, g = s (x B_y - y B_x) = s R B_phi.  A COVERED torus Gamma bounds a solid torus V with (T1) no
source in V, (T2) s B_phi > 0 on Gamma, (T3) the axis in V and Gamma meeting Sigma in one Jordan curve gamma
around a; D = int(gamma), flux(Gamma) = int_D s B_phi dR dZ.  Every surface the solver returns is covered.

LEMMA A (torux.barrier).  g is harmonic in the vacuum, so g > 0 on V-bar.
LOST.  z in Sigma is lost if its field line (either direction) reaches a point with g < 0.
LEMMA L.  No point of D-bar is lost (a line through V stays in V-bar, where g > 0).

THEOREM (survivor flux).  Let L be a closed subset of Sigma all of whose points are lost, and K the connected
component of Sigma \\ L that contains a.  Then every covered torus has
        flux(Gamma) <= int_K (s B_phi)^+ dR dZ.
Proof: D-bar misses L (Lemma L); D is connected and contains a, so D lies in K; on D s B_phi > 0 (Lemma A), so
flux(Gamma) = int_D s B_phi <= int_K (s B_phi)^+.  QED.
No closed curve of lost points is needed beyond one that makes K bounded: every certified lost cell inside the
wall lowers the bound by its flux, and every cell left uncertified only makes the bound larger (never wrong).
With K bounded, int_K (s B_phi)^+ = int_K s B_phi + int_K (s B_phi)^- and int_K s B_phi = -s x the loop integral of
A along the boundary of K (counter-clockwise in (R, Z); Stokes), so
        F_up = -s oint_{dK} (A_R dR + A_Z dZ) + sum over cells of K where s B_phi may be negative of a bound of
               int (s B_phi)^-  + the edge quadrature's error.
SYMMETRY.  For a stellarator-symmetric source S(R, Z) = (R, -Z) maps field lines to field lines (reversed) and g to
g, so the mirror image of a lost cell is lost: L is taken symmetric, a forward certificate of S c certifies c.

CELLS AND THEIR CERTIFICATES.  A uniform grid of square level-0 cells (side h0 = r_eff / 8, r_eff the effective minor
radius of T_last's section or of a scale polygon) on a box around a, symmetric in Z; a quadtree of pieces below it
(depth D, finest side hf = h0 / 2^D, the certified set kept on the finest raster).  A piece Q (centre c, half side eta)
has the 9-point stencil c + (u, v) eta, u, v in {-1, 0, 1}, and 4 extra points (+-1/2, +-1/2) (its children's
centres).  Every line is followed on the source's own filaments (nested trapezoid rules per coil, the level chosen
from the coil's distance and CALIBRATED against level 0, accurate to 1e-10 at distances >= d_fail; field checked
against the independent reference evaluator in every run) in arc length (dx/ds = s B / |B|, compiled GBS of order 8
with step control, one line per thread) to its first crossing of the plane phi = 2 pi / nfp (the period map P, onto
Sigma by the nfp symmetry) and on, up to a window of arc length, until it reaches f = s B_phi / |B| <= -beta = -0.05
and the minimum of f after it (the lost point).  Lines that come within d_fail of a coil fail.
  * BASE certificate (every line of Q reaches g < 0): the 13 lines are followed to 4 arc lengths between the centre's
    first f <= -beta and its lost point.  At each, the image of Q lies in
      (affine)  X_00 + [-1, 1] A_u + [-1, 1] A_v + ball(rho), rho = 2 x 1.5625 x the largest nodal residual rn of the
                affine model + tau (1.5625: the Lebesgue constant of 3 x 3 tensor interpolation -- a biquadratic
                remainder is at most 1.5625 x its largest nodal value; factor 2 for the higher terms), used when
                rn <= 0.25 x (|A_u| + |A_v|);
      (curved)  I([-1, 1]^2) + ball(rho_q), I the biquadratic interpolant of the 9 images, rho_q = 3 x its largest
                error at the 4 extra points + tau (the u^3 / v^3 errors peak within 1.03 x their value there, u^3 v
                within 2.05 x), used when that error is <= 0.5 rn and rn <= 0.5 (|A_u| + |A_v|); I is covered by
                m x m sub-squares, each an affine piece with the exact second-derivative remainder of the polynomial;
    tau = 2 x the summed step estimates + 2 eps_f x arc length x the piece's stretch (the field model's error).
    g < 0 on it when g(X) + |grad g . A_u| + |grad g . A_v| + |grad g| rho + G2 r^2 / 2 < 0 at every (sub-)centre X,
    G2 = 2 x the largest difference quotient of grad g between the image points, r the (sub-)enclosure's radius;
    every stencil line keeps >= 2 x the image's extent from every coil (no coil between the lines of the piece).
  * PERIOD certificate (Q's image under P lies in certified cells): the same enclosures of P(Q) from the plane
    crossings (2D); every finest cell within reach must be certified (never Q's own: a piece is not certified by
    itself); the lines keep f >= 0.2 up to the plane (P smooth on Q: no turning point of phi) and >= 2 x the image's
    extent from the coils.
  The level-0 cells are processed in rings (ray ratio to the scale polygon 1.3, 1.6, 2, 2.5, 3.2, 4, rest) from the
outside in, down to the first ring with less than 10 % of its area certified (the confined region); the period certificates are iterated to a fixed point (loss
propagates inward one period per round); then, level by level, the pending pieces on the boundary of K are split
and the rounds repeated (the work goes where it lowers the bound).

What is strict and what is validated
------------------------------------
STRICT (given the computed numbers): Lemma A, Lemma L, the theorem; the symmetry; the component and its boundary
(integer grid topology: 8-connected flood fill, a superset of the true component); Stokes on the cell boundary; the
second-derivative bound of the biquadratic polynomial; the coverage tests (separating axes, conservative).
VALIDATED (error models with stated assumptions, each checked against an independent computation in every run):
  * the enclosure models (affine: biquadratic remainder; curved: errors up to fourth order, the Taylor terms
    decaying); checked on sampled certified pieces by 5 x 5 stencils (every extra point inside: ratio < 1);
  * the trace error: 2 x the summed GBS step estimates; checked against the independent reference tracer
    (torux.reference.trace, Gauss-Legendre IRK on the fast64 field) on sampled plane crossings and against scipy DOP853
    on the reference field on sampled lost points (which must have g < 0 by the reference field);
  * the Hessian bound G2 of g (difference quotients of the gradient at the image points, factor 2);
  * the field: the nested trapezoid rules with the calibrated level distances (eps 1e-10 at d >= d_fail);
    the multilevel B against the reference evaluator at 48 points in every run;
  * the smoothness of P on a piece from f >= 0.2 on its stencil lines and the coil-distance guard;
  * floating-point rounding is not interval-bounded (all quantities are far above 1e-12 relative).
NEGATIVE TEST in every run with a verified torus (T_last, verified_sections): no certified cell may meet its section
(its points are never lost), and F_up >= its flux; a violation refuses the run as a false certificate.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

# status codes of the tracer
ST_PLANE, ST_LOST, ST_ARC, ST_SMAX, ST_COIL, ST_FAIL = 0, 1, 2, 3, 4, 5
RING_STOP = 0.1         # the descent through the rings stops at a ring with less than this fraction certified
NK_MIN = 16             # nodes of the coarsest nested rule of a coil
H_COIL = 0.5            # arc-length steps at most this x the coil-distance lower bound (the levels' margin is the step)
LEB33 = 1.5625          # Lebesgue constant of 3 x 3 tensor interpolation on equispaced nodes (1.25^2)
NONLIN = 0.25           # largest nodal residual of the affine model, relative to |A_u| + |A_v|, for a certificate
COIL_GUARD = 2.0        # base certificates: every stencil line keeps at least this x the image's extent (at the arc
                        # length tested) from every coil (no coil between the lines of a piece: the map is smooth on it)
COIL_GUARD_P = 0.5      # period certificates: the lines keep this x the final image's extent from the coils up to the
                        # plane (the image is smaller before; a coil between the lines would also show as nonlinearity)
F_SMOOTH = 0.2          # smallest s B_phi / |B| along a stencil line up to the plane for a period certificate


# ============================================================================ exact field, compiled tracer
@njit(fastmath=True, cache=True)
def _levels(x0, x1, x2, P, margin, lev):
    """The level per coil for every point within `margin` of x (P = (r, w, off, Nc, Kc, vmax, Cq, dk)): coil c's
    level-0 rule has Nc[c] nodes (a power of 2 times a base count) in r[off[c]:off[c+1]], w the level-0 weights; the
    rule of level k takes every 2^k-th node with weight 2^k w.  The level is the coarsest whose strip-model node count
    at the coil's distance lower bound suffices (need = vmax[c] Cq / d_lb, Cq = ln(2 / eps) / (2 pi strip)) AND
    d_lb >= dk[c, k], the calibrated distance beyond which level k agrees with level 0 to eps (multilevel_nodes);
    d_lb = the distance to a level's nodes less half their largest arc-length spacing vmax 2^k / N_c (the coarsest
    level, refined two levels at a time while that half spacing exceeds a quarter of the distance), less the margin.  Returns the smallest d_lb over the coils (0 if negative)."""
    r, w, off, Nc, Kc, vmax, Cq, dk = P
    dmin = 1e300
    for c in range(off.shape[0] - 1):
        o0 = off[c]
        N = Nc[c]
        kd = Kc[c]
        while True:
            sK = 1 << kd
            q2 = 1e300
            for j in range(o0, o0 + N, sK):
                d0 = x0 - r[j, 0]; d1 = x1 - r[j, 1]; d2 = x2 - r[j, 2]
                q = d0 * d0 + d1 * d1 + d2 * d2
                if q < q2:
                    q2 = q
            mk = 0.5 * vmax[c] * sK / N
            dlb = math.sqrt(q2) - mk - margin
            # a finer node set where the spacing margin is not small against the distance (the nearby coils)
            if kd == 0 or mk < 0.25 * (dlb + mk):
                break
            kd = max(kd - 2, 0)
        if dlb < dmin:
            dmin = dlb
        k = 0
        if dlb > 0:
            need = vmax[c] * Cq / dlb
            while k < Kc[c] and N / (1 << (k + 1)) >= need and dlb >= dk[c, k + 1]:
                k += 1
        lev[c] = k
    return max(dmin, 0.0)


@njit(fastmath=True, cache=True)
def _bs_lev(x0, x1, x2, P, lev):
    r, w, off, Nc, Kc, vmax, Cq, dk = P
    a0 = 0.0; a1 = 0.0; a2 = 0.0
    for c in range(off.shape[0] - 1):
        o0 = off[c]
        st = 1 << lev[c]
        b0 = 0.0; b1 = 0.0; b2 = 0.0
        for j in range(o0, o0 + Nc[c], st):
            d0 = x0 - r[j, 0]; d1 = x1 - r[j, 1]; d2 = x2 - r[j, 2]
            q = d0 * d0 + d1 * d1 + d2 * d2
            ir = 1.0 / math.sqrt(q)
            ir3 = ir * ir * ir
            b0 += (w[j, 1] * d2 - w[j, 2] * d1) * ir3
            b1 += (w[j, 2] * d0 - w[j, 0] * d2) * ir3
            b2 += (w[j, 0] * d1 - w[j, 1] * d0) * ir3
        a0 += st * b0; a1 += st * b1; a2 += st * b2
    return a0, a1, a2


@njit(cache=True)
def _bs(x0, x1, x2, P):
    """B at x with the levels for x itself; returns (B_x, B_y, B_z, d_lb^2)."""
    lev = np.empty(P[2].shape[0] - 1, np.int64)
    d = _levels(x0, x1, x2, P, 0.0, lev)
    a0, a1, a2 = _bs_lev(x0, x1, x2, P, lev)
    return a0, a1, a2, d * d


@njit(cache=True)
def _vel(x, sig, P, lev, out):
    b0, b1, b2 = _bs_lev(x[0], x[1], x[2], P, lev)
    bm = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
    if not (bm > 0.0) or not math.isfinite(bm):
        return False
    out[0] = sig * b0 / bm; out[1] = sig * b1 / bm; out[2] = sig * b2 / bm
    return True


@njit(cache=True)
def _gbs(x, H, k, sig, P, xout):
    """One GBS macro step (modified midpoint, n_j = 2 j, Aitken-Neville in h^2) of dx/ds = sig B/|B| from x over H,
    the coils' levels fixed for the step (chosen for every point within |H| of x: the stages stay in that ball).
    Returns (ok, err, d_lb^2 of the ball)."""
    T = np.empty((k, 3))
    f0 = np.empty(3); f = np.empty(3)
    z0 = np.empty(3); z1 = np.empty(3); z2 = np.empty(3)
    lev = np.empty(P[2].shape[0] - 1, np.int64)
    dl = _levels(x[0], x[1], x[2], P, abs(H), lev)
    qmin = dl * dl
    ok = _vel(x, sig, P, lev, f0)
    if not ok:
        return False, 0.0, qmin
    for jj in range(k):
        n = 2 * (jj + 1)
        hh = H / n
        for c in range(3):
            z0[c] = x[c]
            z1[c] = x[c] + hh * f0[c]
        for m in range(1, n):
            ok = _vel(z1, sig, P, lev, f)
            if not ok:
                return False, 0.0, qmin
            for c in range(3):
                z2[c] = z0[c] + 2.0 * hh * f[c]
                z0[c] = z1[c]
                z1[c] = z2[c]
        for c in range(3):
            T[jj, c] = z1[c]
    # Neville
    tab = np.empty((k, k, 3))
    for jj in range(k):
        for c in range(3):
            tab[jj, 0, c] = T[jj, c]
        for kk in range(1, jj + 1):
            ratio = ((jj + 1.0) / (jj + 1.0 - kk)) ** 2 - 1.0
            for c in range(3):
                tab[jj, kk, c] = tab[jj, kk - 1, c] + (tab[jj, kk - 1, c] - tab[jj - 1, kk - 1, c]) / ratio
    err = 0.0
    for c in range(3):
        xout[c] = tab[k - 1, k - 1, c]
        e = abs(tab[k - 1, k - 1, c] - tab[k - 1, k - 2, c]) if k > 1 else 0.0
        if e > err:
            err = e
        if not math.isfinite(xout[c]):
            return False, err, qmin
    return True, err, qmin


@njit(cache=True)
def _fval(x, s, P):
    """f = s B_phi / |B| and the node distance^2 at x."""
    b0, b1, b2, q = _bs(x[0], x[1], x[2], P)
    R = math.sqrt(x[0] * x[0] + x[1] * x[1])
    bp = (x[0] * b1 - x[1] * b0) / R
    bm = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
    return s * bp / bm, q


@njit(cache=True)
def _step_ctl(err, Hs, tol, sc, k):
    return 0.9 * (tol * sc / max(err, 1e-300)) ** (1.0 / (2 * k - 1))


@njit(cache=True)
def _cross(x, xt, phi, phi_t, Hs, phin, k, s, P):
    """The crossing of phi = phi_t inside the step from x (angle phi) of length Hs (end angle phin >= phi_t): Illinois
    regula falsi on the step length; xt receives the crossing point.  Returns (step length, its estimate)."""
    a_ = 0.0; ga = phi - phi_t
    b_ = Hs; gb = phin - phi_t
    side = 0
    hc = Hs
    ec = 0.0
    for _ in range(60):
        hc = b_ - gb * (b_ - a_) / (gb - ga)
        okc, ec, qc = _gbs(x, hc, k, s, P, xt)
        if not okc:
            break
        gc = phi + math.atan2(x[0] * xt[1] - x[1] * xt[0], x[0] * xt[0] + x[1] * xt[1]) - phi_t
        if abs(gc) < 1e-15:
            break
        if gc * gb > 0:
            b_ = hc; gb = gc
            if side == -1:
                ga *= 0.5
            side = -1
        else:
            a_ = hc; ga = gc
            if side == 1:
                gb *= 0.5
            side = 1
        if abs(b_ - a_) < 1e-15 * (1.0 + Hs):
            break
    return hc, ec


MAXS = 384              # accepted step states kept per traced line (for positions at later arc lengths)


@njit(parallel=True, cache=True)
def trace_kernel(X0, s, S_max, ext, phi_t, beta, d_fail, tol, H0, H_max, k, P):
    """Lines from X0 (L, 3) along s B / |B| (s = +-1: the forward direction where s B_phi > 0) until the plane
    phi = phi_t (unwrapped from the start's angle) is crossed -- recorded -- and on until lost (f = s B_phi / |B| <=
    -beta, then to the minimum of f), the arc length S_max (ext x S_max for a line whose f fell below 0.3: near its
    escape), a coil (distance lower bound < d_fail) or a failed step.
    Returns Xp (L, 3) the crossing (NaN if none), sp (L,) its arc length, ep (L,) the summed step estimates up to it,
    fp (L,) the smallest f at accepted step ends before it; Xe (L, 3), se, ee: the lost point (the minimum of f after
    f <= -beta) or the last point; sf (L,) the arc length where f first fell to -beta (NaN: never); status (L,),
    fmin (L,), dmin (L,) the smallest coil-distance lower bound on the path, nst (L,) accepted steps, dp (L,) the
    smallest coil-distance lower bound up to the crossing."""
    L = X0.shape[0]
    Xp = np.full((L, 3), np.nan); sp = np.full(L, np.nan); ep = np.full(L, np.nan); fp = np.full(L, np.nan)
    dp = np.full(L, np.nan)
    Xe = np.full((L, 3), np.nan); se = np.zeros(L); ee = np.zeros(L); sf = np.full(L, np.nan)
    status = np.full(L, ST_FAIL); fmin = np.full(L, np.inf); dmin = np.full(L, np.inf)
    nst = np.zeros(L, np.int64)
    traj = np.full((L, MAXS, 6), np.nan)
    dfail2 = d_fail * d_fail
    for i in prange(L):
        x = X0[i].copy()
        traj[i, 0, 0] = 0.0; traj[i, 0, 1] = x[0]; traj[i, 0, 2] = x[1]; traj[i, 0, 3] = x[2]; traj[i, 0, 4] = 0.0
        traj[i, 0, 5] = 1e300
        xn = np.empty(3); xt = np.empty(3)
        sacc = 0.0
        eacc = 0.0
        H = H0
        phi = math.atan2(x[1], x[0])
        crossed = False
        lost_on = False
        best_f = np.inf
        best_x = x.copy(); best_s = 0.0; best_e = 0.0
        stat = ST_SMAX
        f, q = _fval(x, s, P)
        qmin = q
        fm = f
        smax = S_max
        it = 0
        while it < 400000:
            it += 1
            Hs = min(H, smax - sacc) if not lost_on else H
            if Hs <= 0:
                break
            ok, err, q = _gbs(x, Hs, k, s, P, xn)
            if q < qmin:
                qmin = q
            if q < dfail2:
                stat = ST_COIL
                break
            sc = 1.0 + math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
            if (not ok) or err > tol * sc:
                H = Hs * (0.25 if not ok else max(0.2, _step_ctl(err, Hs, tol, sc, k)))
                if H < 1e-11:
                    stat = ST_FAIL
                    break
                continue
            phin = phi + math.atan2(x[0] * xn[1] - x[1] * xn[0], x[0] * xn[0] + x[1] * xn[1])
            if (not crossed) and phin >= phi_t:
                hc, ec = _cross(x, xt, phi, phi_t, Hs, phin, k, s, P)
                for c in range(3):
                    Xp[i, c] = xt[c]
                sp[i] = sacc + hc
                ep[i] = eacc + ec
                fp[i] = fm
                dp[i] = math.sqrt(qmin)
                crossed = True
            for c in range(3):
                x[c] = xn[c]
            sacc += Hs
            eacc += err
            phi = phin
            nst[i] += 1
            if nst[i] < MAXS:
                traj[i, nst[i], 0] = sacc; traj[i, nst[i], 1] = x[0]; traj[i, nst[i], 2] = x[1]
                traj[i, nst[i], 3] = x[2]; traj[i, nst[i], 4] = eacc; traj[i, nst[i], 5] = math.sqrt(qmin)
            f, q = _fval(x, s, P)
            if f < fm:
                fm = f
                if fm < 0.3 and smax == S_max:
                    smax = ext * S_max
            if f <= -beta and not lost_on:
                lost_on = True
                sf[i] = sacc
            if lost_on:
                if f < best_f:
                    best_f = f
                    for c in range(3):
                        best_x[c] = x[c]
                    best_s = sacc; best_e = eacc
                elif f > best_f + 0.02 or sacc > best_s + 0.3:
                    stat = ST_LOST
                    break
            elif sacc >= smax:
                stat = ST_SMAX
                break
            H = min(Hs * min(3.0, max(0.2, _step_ctl(err, Hs, tol, sc, k))), H_max)
            H = min(H, max(H_COIL * math.sqrt(q), 1e-6))
        if lost_on:
            stat = ST_LOST if stat != ST_COIL and stat != ST_FAIL else stat
            if stat == ST_COIL or stat == ST_FAIL:
                stat = ST_LOST          # f <= -beta was reached before: the line is lost; its lost point is kept
            for c in range(3):
                Xe[i, c] = best_x[c]
            se[i] = best_s; ee[i] = best_e
        else:
            for c in range(3):
                Xe[i, c] = x[c]
            se[i] = sacc; ee[i] = eacc
        status[i] = stat
        fmin[i] = fm
        dmin[i] = math.sqrt(qmin)
    return Xp, sp, ep, fp, Xe, se, ee, sf, status, fmin, dmin, nst, dp, traj


@njit(parallel=True, cache=True)
def positions_kernel(traj, S_tgt, s, d_fail, k, P):
    """Positions of recorded lines at arc lengths S_tgt (L, m): one GBS step from the last recorded state before each
    target (traj (L, MAXS, 6) = (s, x, y, z, summed estimate, smallest coil distance so far) at accepted steps).
    Returns X (L, m, 3), e (L, m) the summed estimate there, ok (L,) (a target beyond the record, or a failed step:
    False), dm (L, m) the smallest coil-distance bound up to the target."""
    L = traj.shape[0]
    m = S_tgt.shape[1]
    X = np.full((L, m, 3), np.nan); E = np.full((L, m), np.nan); okl = np.ones(L, np.bool_)
    DM = np.full((L, m), np.nan)
    for i in prange(L):
        x0 = np.empty(3); xo = np.empty(3)
        for j in range(m):
            st = S_tgt[i, j]
            kk = -1
            last = -1
            for q in range(traj.shape[1]):
                if not (traj[i, q, 0] == traj[i, q, 0]):
                    break
                last = q
                if traj[i, q, 0] <= st:
                    kk = q
                else:
                    break
            if kk < 0 or (kk == last and st > traj[i, kk, 0]):
                okl[i] = False          # the target lies beyond the record
                continue
            for c in range(3):
                x0[c] = traj[i, kk, 1 + c]
            H = st - traj[i, kk, 0]
            if H <= 0:
                for c in range(3):
                    X[i, j, c] = x0[c]
                E[i, j] = traj[i, kk, 4]
                DM[i, j] = traj[i, kk, 5]
                continue
            ok, err, q2 = _gbs(x0, H, k, s, P, xo)
            if not ok or q2 < d_fail * d_fail:
                okl[i] = False
                continue
            for c in range(3):
                X[i, j, c] = xo[c]
            E[i, j] = traj[i, kk, 4] + err
            DM[i, j] = min(traj[i, kk, 5], math.sqrt(q2))
    return X, E, okl, DM


@njit(parallel=True, cache=True)
def arc_kernel(X0, s, S_tgt, d_fail, tol, H0, H_max, k, P):
    """Lines from X0 (L, 3) along s B / |B| with their positions at the arc lengths S_tgt (L, m) (ascending).
    Returns Xc (L, m, 3), ec (L, m) the summed step estimates there, ok (L,) all reached, dmin (L,)."""
    L = X0.shape[0]
    m = S_tgt.shape[1]
    Xc = np.full((L, m, 3), np.nan); ec = np.full((L, m), np.nan)
    okl = np.zeros(L, np.bool_); dmin = np.full(L, np.inf)
    dfail2 = d_fail * d_fail
    for i in prange(L):
        x = X0[i].copy()
        xn = np.empty(3)
        sacc = 0.0
        eacc = 0.0
        H = H0
        qmin = 1e300
        jt = 0
        good = True
        it = 0
        while jt < m and it < 400000:
            it += 1
            if sacc >= S_tgt[i, jt] - 1e-15:
                for c in range(3):
                    Xc[i, jt, c] = x[c]
                ec[i, jt] = eacc
                jt += 1
                continue
            Hs = min(H, S_tgt[i, jt] - sacc)
            ok, err, q = _gbs(x, Hs, k, s, P, xn)
            if q < qmin:
                qmin = q
            if q < dfail2:
                good = False
                break
            sc = 1.0 + math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
            if (not ok) or err > tol * sc:
                H = Hs * (0.25 if not ok else max(0.2, _step_ctl(err, Hs, tol, sc, k)))
                if H < 1e-11:
                    good = False
                    break
                continue
            for c in range(3):
                x[c] = xn[c]
            sacc += Hs
            eacc += err
            H = min(Hs * min(3.0, max(0.2, _step_ctl(err, Hs, tol, sc, k))), H_max)
            H = min(H, max(H_COIL * math.sqrt(q), 1e-6))
        okl[i] = good and jt == m
        dmin[i] = math.sqrt(qmin)
    return Xc, ec, okl, dmin


@njit(parallel=True, cache=True)
def _coil_B(X, r, w, st):
    out = np.zeros((X.shape[0], 3))
    for i in prange(X.shape[0]):
        a0 = 0.0; a1 = 0.0; a2 = 0.0
        for j in range(0, r.shape[0], st):
            d0 = X[i, 0] - r[j, 0]; d1 = X[i, 1] - r[j, 1]; d2 = X[i, 2] - r[j, 2]
            q = d0 * d0 + d1 * d1 + d2 * d2
            ir = 1.0 / math.sqrt(q)
            ir3 = ir * ir * ir
            a0 += (w[j, 1] * d2 - w[j, 2] * d1) * ir3
            a1 += (w[j, 2] * d0 - w[j, 0] * d2) * ir3
            a2 += (w[j, 0] * d1 - w[j, 1] * d0) * ir3
        out[i, 0] = st * a0; out[i, 1] = st * a1; out[i, 2] = st * a2
    return out


@njit(parallel=True, cache=True)
def _coil_dlb(X, r, N, K, vmax):
    """The distance lower bound of _levels (the same refinement) for one coil's nodes r (N, 3) at points X."""
    out = np.empty(X.shape[0])
    for i in prange(X.shape[0]):
        kd = K
        while True:
            sK = 1 << kd
            q2 = 1e300
            for j in range(0, N, sK):
                d0 = X[i, 0] - r[j, 0]; d1 = X[i, 1] - r[j, 1]; d2 = X[i, 2] - r[j, 2]
                q = d0 * d0 + d1 * d1 + d2 * d2
                if q < q2:
                    q2 = q
            mk = 0.5 * vmax * sK / N
            dlb = math.sqrt(q2) - mk
            if kd == 0 or mk < 0.25 * (dlb + mk):
                break
            kd = max(kd - 2, 0)
        out[i] = dlb
    return out


def multilevel_nodes(source, d_fail, eps=1e-10, strip=0.8, n_cal=48, seed=0, safety=1.25):
    """Per Fourier coil (nonzero current): nested trapezoid rules for torux.edgebound._bs.  Level 0 has N_c = 2^K N_K
    nodes with N_c >= the strip-model count for relative error eps at distance d_fail (torux.field.engine.nodes_needed)
    and >= 4 order + 8, N_K = NK_MIN the coarsest level.  The strip model ignores the growth of a high-order coil's own
    modes off the real line, so every level k >= 1 is also CALIBRATED: at n_cal random points of the coil x 28
    distances (d_fail .. max(64 d_fail, 12 x the coil's radius), geometric; random directions) the level-k field is
    compared with level 0; dk[c, k] = safety x
    the largest distance lower bound (the kernel's d_lb) of a sample where they differ by more than eps / 4 relative
    to the coil's own field there (level k is used only beyond it).  Returns P = (r, w, off, Nc, Kc, vmax, Cq, dk)."""
    from .field.engine import nodes_needed, fourier_nodes_np
    rng = np.random.default_rng(seed)
    rs, ws, Ns, Ks, vm, dks = [], [], [], [], [], []
    for p in source.parts:
        if p.kind != "fourier":
            if p.kind == "toroidal" and float(getattr(p, "RB", 0.0)) == 0.0:
                continue
            raise ValueError(f"edgebound: only Fourier filaments are supported (part {p.kind})")
        xs, dxs = p.points(np.arange(4096) / 4096)
        vmax = np.linalg.norm(dxs, axis=-1).max(axis=1) * (1 + 1e-3)
        for i in range(p.n):
            if p.current[i] == 0.0:
                continue
            need = max(nodes_needed(vmax[i], d_fail, eps, strip), 4 * p.order + 8)
            NK = NK_MIN          # the coarsest rules are calibrated (dk): far from the coil they may be coarser than
                                 # the coil's own spectrum needs near it
            K = max(0, int(math.ceil(math.log2(max(need, NK) / NK))))
            N = NK * 2 ** K
            r, w, _ = fourier_nodes_np(p.coef[i], p.current[i], N)
            # calibration samples
            c0 = r[rng.integers(0, N, n_cal)]
            size = float(np.max(np.linalg.norm(r - r.mean(0), axis=1)))
            lad = np.geomspace(d_fail, max(64 * d_fail, 12 * size), 28)
            dirn = rng.normal(size=(n_cal, len(lad), 3)); dirn /= np.linalg.norm(dirn, axis=-1, keepdims=True)
            X = (c0[:, None, :] + dirn * lad[None, :, None]).reshape(-1, 3)
            dlb = _coil_dlb(np.ascontiguousarray(X), r, N, K, float(vmax[i]))     # as the kernel computes it
            ok = dlb > 0
            X, dlb = X[ok], dlb[ok]

            def Bl(st):
                return _coil_B(np.ascontiguousarray(X), r, w, st)
            B0 = Bl(1)
            nb = np.linalg.norm(B0, axis=1)
            dk = np.full(K + 1, 0.0)
            for k in range(1, K + 1):
                e = np.linalg.norm(Bl(2 ** k) - B0, axis=1) / nb
                bad = e > 0.25 * eps
                dk[k] = safety * float(np.max(dlb[bad])) if np.any(bad) else 0.0
                # a rule coarser than the coil's own spectrum errs at every distance (its low multipoles are wrong):
                # usable only when its errors end well inside the calibrated range
                if dk[k] > lad[-1] / 4:
                    dk[k] = np.inf
                dk[k] = max(dk[k], dk[k - 1])
            rs.append(r); ws.append(w); Ns.append(N); Ks.append(K); vm.append(vmax[i]); dks.append(dk)
    off = np.concatenate([[0], np.cumsum(Ns)]).astype(np.int64)
    Cq = math.log(2.0 / eps) / (2 * math.pi * strip)
    Km = max(Ks) if Ks else 0
    dk = np.full((len(Ns), Km + 2), np.inf)
    for c, d_ in enumerate(dks):
        dk[c, :len(d_)] = d_
    return (np.ascontiguousarray(np.concatenate(rs)), np.ascontiguousarray(np.concatenate(ws)), off,
            np.array(Ns, np.int64), np.array(Ks, np.int64), np.array(vm, float), float(Cq), np.ascontiguousarray(dk))


@njit(parallel=True, cache=True)
def _bga_many(X, P):
    """B (3), grad B (9, row-major dB_a/dx_b), A (3) and the coil-distance lower bound at X (n, 3), with the levels of
    _levels for each point."""
    r, w, off, Nc, Kc, vmax, Cq, dk = P
    n = X.shape[0]
    out = np.zeros((n, 16))
    for i in prange(n):
        lev = np.empty(off.shape[0] - 1, np.int64)
        dl = _levels(X[i, 0], X[i, 1], X[i, 2], P, 0.0, lev)
        acc = np.zeros(15)
        for c in range(off.shape[0] - 1):
            st = 1 << lev[c]
            loc = np.zeros(15)
            for j in range(off[c], off[c] + Nc[c], st):
                d0 = X[i, 0] - r[j, 0]; d1 = X[i, 1] - r[j, 1]; d2 = X[i, 2] - r[j, 2]
                q = d0 * d0 + d1 * d1 + d2 * d2
                ir = 1.0 / math.sqrt(q)
                ir3 = ir * ir * ir
                w0 = w[j, 0]; w1 = w[j, 1]; w2 = w[j, 2]
                c0 = w1 * d2 - w2 * d1; c1 = w2 * d0 - w0 * d2; c2 = w0 * d1 - w1 * d0
                loc[0] += c0 * ir3; loc[1] += c1 * ir3; loc[2] += c2 * ir3
                loc[12] += w0 * ir; loc[13] += w1 * ir; loc[14] += w2 * ir
                f5 = 3.0 * ir3 * ir * ir
                e0 = c0 * f5; e1 = c1 * f5; e2 = c2 * f5
                u0 = w0 * ir3; u1 = w1 * ir3; u2 = w2 * ir3
                loc[3] -= e0 * d0; loc[4] += -u2 - e0 * d1; loc[5] += u1 - e0 * d2
                loc[6] += u2 - e1 * d0; loc[7] -= e1 * d1; loc[8] += -u0 - e1 * d2
                loc[9] += -u1 - e2 * d0; loc[10] += u0 - e2 * d1; loc[11] -= e2 * d2
            for t in range(15):
                acc[t] += st * loc[t]
        for t in range(15):
            out[i, t] = acc[t]
        out[i, 15] = dl
    return out


@njit(parallel=True, cache=True)
def _bs_many(X, P):
    out = np.empty((X.shape[0], 4))
    for i in prange(X.shape[0]):
        b0, b1, b2, q = _bs(X[i, 0], X[i, 1], X[i, 2], P)
        out[i, 0] = b0; out[i, 1] = b1; out[i, 2] = b2; out[i, 3] = math.sqrt(q)
    return out


class ExactTracer:
    """The source's Fourier filaments by nested trapezoid rules (multilevel_nodes, _bs: the level per coil from the
    target's distance), summed inline in the compiled arc-length tracer above."""

    def __init__(self, source, nfp, sign, d_fail, tol=1e-10, k=4, beta=0.05, eps=1e-10, engine=None):
        self.P = multilevel_nodes(source, d_fail, eps)
        self.nfp, self.s = int(nfp), float(sign)
        self.d_fail, self.tol, self.k, self.beta, self.eps = float(d_fail), float(tol), int(k), float(beta), eps
        self.n_lines = 0
        self.n_steps = 0
        self.t = 0.0
        self.t_arc = 0.0
        self.n_arc = 0
        self.check_err = None
        self.source = source
        self.engine = engine

    def check(self, n=48, seed=0, ref=None):
        """The multilevel field against the independent reference evaluator (torux.reference.fast64) at n points between
        1.5 and 8 d_fail from random coil nodes (farther than d_fail from every coil): the largest relative
        difference of B (recorded in every run)."""
        from .reference.fast64 import Field
        F = ref if ref is not None else Field(self.source)
        rng = np.random.default_rng(seed)
        r = self.P[0]
        c = r[rng.integers(0, len(r), n)]
        dirn = rng.normal(size=(n, 3)); dirn /= np.linalg.norm(dirn, axis=1)[:, None]
        x = c + dirn * self.d_fail * rng.uniform(1.5, 8.0, (n, 1))
        o = self.B(x)
        ok = o[:, 3] > self.d_fail
        if not np.any(ok):
            return None
        Br = F.B(x[ok])
        self.check_err = float(np.max(np.linalg.norm(o[ok, :3] - Br, axis=1) / np.linalg.norm(Br, axis=1)))
        return self.check_err

    def B(self, X):
        """(n, 4): B and the coil-distance lower bound."""
        return _bs_many(np.ascontiguousarray(np.asarray(X, float).reshape(-1, 3)), self.P)

    def BGA(self, X):
        """B (n, 3), grad B (n, 3, 3) [a, b] = dB_a / dx_b, A (n, 3), coil-distance lower bound (n,)."""
        o = _bga_many(np.ascontiguousarray(np.asarray(X, float).reshape(-1, 3)), self.P)
        return o[:, 0:3], o[:, 3:12].reshape(-1, 3, 3), o[:, 12:15], o[:, 15]

    def run(self, X0, S_max, ext=2.5):
        """trace_kernel on X0 (L, 3): dict of its outputs."""
        X0 = np.ascontiguousarray(np.asarray(X0, float).reshape(-1, 3))
        keys = ("Xp", "sp", "ep", "fp", "Xe", "se", "ee", "sf", "status", "fmin", "dmin", "nst", "dp", "traj")
        if len(X0) == 0:
            return {k_: np.zeros((0, 3)) if k_ in ("Xp", "Xe") else (np.zeros((0, MAXS, 6)) if k_ == "traj" else np.zeros(0))
                    for k_ in keys}
        t0 = time.time()
        out = trace_kernel(X0, self.s, float(S_max), float(ext), 2 * math.pi / self.nfp, self.beta, self.d_fail,
                           self.tol, 0.02, 0.1, self.k, self.P)
        self.t += time.time() - t0
        self.n_lines += len(X0)
        o = dict(zip(keys, out))
        self.n_steps += int(np.sum(o["nst"]))
        return o

    def positions(self, traj, S_tgt):
        """positions_kernel: (X (L, m, 3), e (L, m), ok (L,), dm (L, m))."""
        t0 = time.time()
        out = positions_kernel(np.ascontiguousarray(traj), np.ascontiguousarray(np.asarray(S_tgt, float)), self.s,
                               self.d_fail, self.k, self.P)
        self.t += time.time() - t0
        return out

    def arc(self, X0, S_tgt):
        """Positions at the arc lengths S_tgt (L, m): (Xc (L, m, 3), ec (L, m), ok (L,), dmin (L,))."""
        X0 = np.ascontiguousarray(np.asarray(X0, float).reshape(-1, 3))
        S = np.ascontiguousarray(np.asarray(S_tgt, float).reshape(len(X0), -1))
        t0 = time.time()
        out = arc_kernel(X0, self.s, S, self.d_fail, self.tol, 0.02, 0.1, self.k, self.P)
        self.t += time.time() - t0
        self.t_arc += time.time() - t0
        self.n_arc += len(X0)
        self.n_lines += len(X0)
        return out


# ============================================================================ geometry helpers
def _poly_in(C, P):
    from .barrier import inside_polygon
    return inside_polygon(C, P)


def ray_crossing_max(poly, centre, theta):
    from .escape import ray_crossings
    return ray_crossings(poly, centre, theta)


@njit(cache=True)
def _cells_near_parallelogram(c0, c1, a0, a1, b0, b1, rho, R0, Z0, h, nR, nZ, out):
    """Grid cells (i, j) within distance rho of the parallelogram {c + u A + v B: |u|, |v| <= 1} (A = (a0, a1),
    B = (b0, b1)); conservative: a cell is listed when the separating-axis test on the cell (square) against the
    parallelogram dilated by rho (its two normals, the two axes) does not separate them.  Returns the count written
    to out (k, 2); -1 if out overflows or a cell lies outside the grid."""
    # bounding box
    ex = abs(a0) + abs(b0) + rho
    ey = abs(a1) + abs(b1) + rho
    i0 = int(math.floor((c0 - ex - R0) / h)); i1 = int(math.floor((c0 + ex - R0) / h))
    j0 = int(math.floor((c1 - ey - Z0) / h)); j1 = int(math.floor((c1 + ey - Z0) / h))
    if i0 < 0 or j0 < 0 or i1 >= nR or j1 >= nZ:
        return -1
    # normals of the parallelogram's sides
    nA0 = -a1; nA1 = a0
    nB0 = -b1; nB1 = b0
    lA = math.sqrt(nA0 * nA0 + nA1 * nA1); lB = math.sqrt(nB0 * nB0 + nB1 * nB1)
    cnt = 0
    hh = 0.5 * h
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            x = R0 + (i + 0.5) * h; y = Z0 + (j + 0.5) * h
            dx = x - c0; dy = y - c1
            sep = False
            # axis normals n_B (perpendicular to B: the A-extent)
            for t in range(2):
                if t == 0:
                    n0 = nB0; n1 = nB1; ln = lB
                    half = abs(a0 * nB0 + a1 * nB1)
                else:
                    n0 = nA0; n1 = nA1; ln = lA
                    half = abs(b0 * nA0 + b1 * nA1)
                if ln <= 0:
                    continue
                proj = abs(dx * n0 + dy * n1)
                sq = hh * (abs(n0) + abs(n1))
                if proj > half + sq + rho * ln:
                    sep = True
                    break
            if not sep:
                if cnt >= out.shape[0]:
                    return -1
                out[cnt, 0] = i; out[cnt, 1] = j
                cnt += 1
    return cnt


Q_FAC = 3.0             # rho_q = Q_FAC x the largest sampled error of the biquadratic interpolant (at (+-1/2, +-1/2):
                        # the u^3 / v^3 error peaks within 1.03 x its value there, the u^3 v terms within 2.05 x)
QNONLIN = 0.5           # the curved enclosure is tried up to this nonlinearity
Q_DECAY = 0.5           # ... and only when the sampled interpolation error is at most this fraction of the quadratic
                        # content (the Taylor terms decay: the factor Q_FAC is then a bound with margin)


def _biquad(P9):
    """Coefficient form of the biquadratic interpolant of P9 (9, d) on u, v in {-1, 0, 1} (stencil order): returns
    a function (u, v arrays) -> (n, d) and the bounds Muu, Muv, Mvv of |I_uu|, |I_uv|, |I_vv| on [-1, 1]^2."""
    G = P9.reshape(3, 3, -1)            # [v index, u index, d]

    def lag(t):
        return np.stack([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], -1)

    def I(u, v):
        return np.einsum("nb,na,bad->nd", lag(np.asarray(v, float)), lag(np.asarray(u, float)), G)
    d2u = G[:, 0] - 2 * G[:, 1] + G[:, 2]            # (3 [v], d): I_uu at v nodes; quadratic in v
    d2v = G[0] - 2 * G[1] + G[2]                     # (3 [u], d)
    Muu = 1.25 * float(np.max(np.linalg.norm(d2u, axis=-1)))
    Mvv = 1.25 * float(np.max(np.linalg.norm(d2v, axis=-1)))
    # I_uv bilinear: its largest value at the corners
    lp = lambda t: np.array([t - 0.5, -2.0 * t, t + 0.5])
    Muv = max(float(np.linalg.norm(np.einsum("b,a,bad->d", lp(vv), lp(uu), G))) for uu in (-1, 1) for vv in (-1, 1))
    return I, Muu, Muv, Mvv


def _biquad_cover(P9, P4, tau, cert, R0, Z0, h, nR, nZ, buf, m_max=32):
    """Every fine cell within reach of the curved enclosure I([-1, 1]^2) + ball(rho_q) of a 2D image certified?
    Returns (ok, covered fraction)."""
    I, Muu, Muv, Mvv = _biquad(P9)
    uv4 = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    e4 = float(np.max(np.linalg.norm(I(uv4[:, 0], uv4[:, 1]) - P4, axis=1)))
    u9 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float); v9 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
    rn = _affine_enclosure(P9, u9, v9)[3]
    if not e4 <= Q_DECAY * rn + 1e-15:
        return False, 0.0           # the interpolation error is not small against the quadratic content
    rho_q = Q_FAC * e4 + tau
    M2 = Muu + 2 * Muv + Mvv
    # sub-squares of half width delta: remainder 0.5 delta^2 M2 <= h / 4
    m = int(min(m_max, max(2, math.ceil(1.0 / math.sqrt(max(h / (2.0 * max(M2, 1e-300)), 1e-300))))))
    dl = 1.0 / m
    r_sub = 0.5 * dl * dl * M2
    cen = -1.0 + dl * (2 * np.arange(m) + 1)
    uu, vv = np.meshgrid(cen, cen, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    eps = 1e-4
    Pc = I(uu, vv)
    Pu = (I(uu + eps, vv) - I(uu - eps, vv)) / (2 * eps)       # exact for a quadratic in u (central difference)
    Pv = (I(uu, vv + eps) - I(uu, vv - eps)) / (2 * eps)
    seen = np.zeros(cert.shape, np.bool_)
    tot = 0
    good = 0
    for q in range(len(uu)):
        a = Pu[q] * dl; b = Pv[q] * dl
        n = _cells_near_parallelogram(Pc[q, 0], Pc[q, 1], a[0], a[1], b[0], b[1], rho_q + r_sub, R0, Z0, h, nR, nZ,
                                      buf)
        if n < 0:
            return False, 0.0
        ii, jj = buf[:n, 0], buf[:n, 1]
        new = ~seen[ii, jj]
        seen[ii[new], jj[new]] = True
        tot += int(new.sum()); good += int(np.sum(cert[ii[new], jj[new]]))
        if good < tot:
            return False, good / max(tot, 1)
    return True, 1.0


# ============================================================================ the certificate
class CellGrid:
    """Square cells of side h over [R0, R0 + nR h] x [Z0, -Z0] (Z0 = -nZ h / 2: symmetric in Z, nZ even)."""

    def __init__(self, R0, h, nR, nZ):
        self.R0, self.h, self.nR, self.nZ = float(R0), float(h), int(nR), int(nZ)
        self.Z0 = -0.5 * self.nZ * self.h

    def centre(self, i, j):
        return self.R0 + (np.asarray(i) + 0.5) * self.h, self.Z0 + (np.asarray(j) + 0.5) * self.h

    def mirror(self, j):
        return self.nZ - 1 - np.asarray(j)


def _stencil(cR, cZ, eta):
    u = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float)
    v = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
    return cR + u * eta, cZ + v * eta, u, v


def _affine_enclosure(P, u, v):
    """P (9, d) images at stencil (u, v); returns centre, A_u, A_v, rho_nodes (the largest nodal residual of the
    affine model)."""
    c = P[4]
    Au = 0.5 * (P[5] - P[3])
    Av = 0.5 * (P[7] - P[1])
    res = P - c[None] - u[:, None] * Au[None] - v[:, None] * Av[None]
    return c, Au, Av, float(np.max(np.linalg.norm(res, axis=1)))


class Certifier:
    """State of one edge-bound computation (see edge_upper_bound).  Pieces are squares of the quadtree over the level-0
    grid (side h0 / 2^l, l = 0..D); the certified set is kept on the finest raster (side hf = h0 / 2^D)."""

    def __init__(self, engine, source, axis_point, sign, grid: CellGrid, tracer: ExactTracer, period_arc,
                 depth=3, stellsym=True, win_periods=1.25, split_frac=0.3, verbose=False, det_tracer=None):
        self.eng, self.src, self.a = engine, source, np.asarray(axis_point, float)
        self.s = float(sign)
        self.G0, self.tr = grid, tracer
        self.D = int(depth)
        self.f = 2 ** self.D
        self.Gf = CellGrid(grid.R0, grid.h / self.f, grid.nR * self.f, grid.nZ * self.f)
        self.period_arc = float(period_arc)
        self.sym = bool(stellsym)
        self.win = float(win_periods)
        self.split_frac = float(split_frac)
        self.verbose = verbose
        self.cert = np.zeros((self.Gf.nR, self.Gf.nZ), bool)
        self.pending = set()                         # pieces (l, i, j) not yet certified
        self.cache = {}                              # finest-node-grid key -> node index
        self.nodes = {k: [] for k in ("Xp", "sp", "ep", "fp", "Xe", "se", "ee", "sf", "status", "fmin", "dmin", "dp")}
        self.base_done = {}
        self.split_done = set()
        self.stats = dict(base=0, period=0, split=0, base_fail=0, rounds=0)
        self.enc_log = []
        self.period_pieces = []
        self._deadline = None
        self.cover = {}                 # piece -> covered fraction of its last period enclosure
        self.anylost = {}               # piece -> some stencil line lost
        # detection (selection only, no part of any certificate): the same lines traced with loose rules and
        # tolerance decide which pieces get exact traces
        self.det_tr = det_tracer
        self.det_cache = {}
        self.traj = {}                  # exact node index -> recorded states of a lost line
        self.det_nodes = {k: [] for k in ("Xp", "sp", "ep", "fp", "Xe", "se", "ee", "sf", "status", "fmin", "dmin", "dp")}
        self.n_skipped = 0

    # ---------------------------------------------------------------- pieces and their stencils
    def piece_geom(self, l, i, j):
        hl = self.G0.h / 2 ** l
        return self.G0.R0 + (i + 0.5) * hl, self.G0.Z0 + (j + 0.5) * hl, 0.5 * hl

    def stencil_keys(self, l, i, j):
        """Keys (integers on the grid of spacing hf / 4) of the piece's 9 stencil points."""
        m = 2 ** (self.D - l + 1)
        u = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]); v = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        return (2 * i + 1 + u) * m, (2 * j + 1 + v) * m

    def extra_keys(self, l, i, j):
        """Keys of the piece's 4 points (u, v) = (+-1/2, +-1/2) (its children's centres)."""
        m = 2 ** (self.D - l)
        du = np.array([-1, 1, -1, 1]); dv = np.array([-1, -1, 1, 1])
        return (2 * i + 1) * 2 * m + du * m, (2 * j + 1) * 2 * m + dv * m

    def node_xy(self, ki, kj):
        hn = self.Gf.h / 4.0
        return self.G0.R0 + ki * hn, self.G0.Z0 + kj * hn

    def fine_range(self, l, i, j):
        m = 2 ** (self.D - l)
        return slice(i * m, (i + 1) * m), slice(j * m, (j + 1) * m)

    def trace_keys(self, KI, KJ, deadline=None, chunk=1024, det=False):
        """Trace (cached) the points of the keys: exact (det=False) or detection; chunks of `chunk` lines between
        deadline checks.  Returns the node index per key (-1: not traced, the deadline passed)."""
        cache, nodes, tr = (self.det_cache, self.det_nodes, self.det_tr) if det else (self.cache, self.nodes, self.tr)
        new = [(a, b) for a, b in set(zip(KI.tolist(), KJ.tolist())) if (a, b) not in cache]
        for lo in range(0, len(new), chunk):
            if deadline is not None and time.time() > deadline:
                break
            part = new[lo:lo + chunk]
            ki = np.array([a for a, _ in part]); kj = np.array([b for _, b in part])
            R, Z = self.node_xy(ki, kj)
            o = tr.run(np.stack([R, 0 * R, Z], -1), self.win * self.period_arc)
            n0 = len(nodes["sp"])
            for k_ in nodes:
                nodes[k_].extend(list(o[k_]))
            if not det:
                for m_ in np.flatnonzero(o["status"] == ST_LOST):
                    self.traj[n0 + int(m_)] = o["traj"][m_]
            for m_, key in enumerate(part):
                cache[key] = n0 + m_
        return np.array([cache.get((a, b), -1) for a, b in zip(KI.tolist(), KJ.tolist())])

    def det_select(self, pcs, deadline, buf):
        """Detection traces of the pieces' stencils; returns the pieces worth exact traces: every detected line lost
        (a base candidate), or every line crossing the plane with its (detected) image enclosure at least half in
        certified cells (a period candidate).  Also records anylost / cover for the refinement."""
        KI, KJ = [], []
        for pc in pcs:
            a_, b_ = self.stencil_keys(*pc)
            KI.append(a_); KJ.append(b_)
        idx = self.trace_keys(np.concatenate(KI), np.concatenate(KJ), deadline=deadline, det=True).reshape(-1, 9)
        D = {k: np.asarray(v) for k, v in self.det_nodes.items()}
        Gf = self.Gf
        u = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float)
        v = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
        sel = []
        for pc, ix in zip(pcs, idx):
            if np.any(ix < 0):
                continue
            st = D["status"][ix]
            lost = st == ST_LOST
            self.anylost[pc] = bool(np.any(lost))
            if np.all(lost):
                sel.append(pc)
                continue
            if not np.all(np.isfinite(D["sp"][ix])):
                continue
            Xp = D["Xp"][ix]
            P = np.stack([np.hypot(Xp[:, 0], Xp[:, 1]), Xp[:, 2]], -1)
            c, Au, Av, rn = _affine_enclosure(P, u, v)
            n = _cells_near_parallelogram(c[0], c[1], Au[0], Au[1], Av[0], Av[1], 0.0, Gf.R0, Gf.Z0, Gf.h, Gf.nR,
                                          Gf.nZ, buf)
            frac = float(np.mean(self.cert[buf[:n, 0], buf[:n, 1]])) if n > 0 else 0.0
            self.cover[pc] = frac
            if frac >= 0.5:
                sel.append(pc)
        return sel

    def node_arrays(self):
        return {k: np.asarray(v) for k, v in self.nodes.items()}

    # ---------------------------------------------------------------- tests
    def tau_field(self, S, lin, eta):
        """Position error from the field model's error (relative eps_f = max(the checked difference against the
        reference, the rules' eps)): 2 eps_f x arc length x the piece's stretch."""
        ef = max(self.tr.check_err or 0.0, self.tr.eps)
        return 2.0 * ef * S * max(1.0, lin / max(eta, 1e-300))

    def base_enclosure_test(self, Xn, en, eta=None, S=0.0, dpath=np.inf):
        """g < 0 on the enclosure of a piece's image from its 9 stencil images Xn (9, 3) at a common arc length (step
        estimates en (9,)).  Returns (ok, details)."""
        u = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float)
        v = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
        c, Au, Av, rn = _affine_enclosure(Xn, u, v)
        lin = float(np.linalg.norm(Au) + np.linalg.norm(Av))
        rho = 2.0 * LEB33 * rn + 2.0 * float(np.max(en)) + 1e-12 + (self.tau_field(S, lin, eta) if eta else 0.0)
        if not (rn <= NONLIN * lin + 1e-15):
            return False, dict(why="nonlinear", rn=rn, lin=lin)
        Bn, Dn, _, dl = self.tr.BGA(Xn)
        g = self.s * (Xn[:, 0] * Bn[:, 1] - Xn[:, 1] * Bn[:, 0])
        dg = self.s * (np.stack([Bn[:, 1], -Bn[:, 0], 0 * Bn[:, 0]], -1)
                       + Xn[:, 0:1] * Dn[:, 1, :] - Xn[:, 1:2] * Dn[:, 0, :])
        rr = lin + rho
        dd = np.linalg.norm(Xn - c[None], axis=1)
        dq = np.linalg.norm(dg - dg[4][None], axis=1)
        G2 = 2.0 * float(np.max(np.where(dd > 0, dq / np.where(dd > 0, dd, 1.0), 0.0)))
        bound = float(g[4] + abs(dg[4] @ Au) + abs(dg[4] @ Av) + np.linalg.norm(dg[4]) * rho + 0.5 * G2 * rr * rr)
        ok = bool(bound < -1e-9 * (abs(g[4]) + 1.0) and np.all(g < 0) and rr < 0.5 * float(dl[4])
                  and dpath >= COIL_GUARD * rr)
        return ok, dict(bound=bound, g=float(g[4]), rr=rr, dc=float(dl[4]))

    def base_curved_test(self, X9, X4, tau, m_max=8, eta=None, S=0.0, dpath=np.inf):
        """g < 0 on the curved enclosure I([-1, 1]^2) + ball(rho_q) of a piece's 3D image (biquadratic interpolant of
        the 9 stencil images, its error sampled at the 4 points (+-1/2, +-1/2), rho_q = Q_FAC x that + tau): m x m
        sub-squares, each an affine piece of I with the exact remainder of the polynomial, g linearised at its centre
        with the Hessian bound G2 = 2 x the largest difference quotient of grad g between neighbouring centres."""
        I, Muu, Muv, Mvv = _biquad(X9)
        uv4 = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        e4 = float(np.max(np.linalg.norm(I(uv4[:, 0], uv4[:, 1]) - X4, axis=1)))
        u9 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float); v9 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
        _, Au_, Av_, rn = _affine_enclosure(X9, u9, v9)
        lin_ = np.linalg.norm(Au_) + np.linalg.norm(Av_)
        if not (e4 <= Q_DECAY * rn + 1e-15 and rn <= QNONLIN * lin_):
            return False
        rho_q = Q_FAC * e4 + tau + (self.tau_field(S, lin_, eta) if eta else 0.0)
        M2 = Muu + 2 * Muv + Mvv
        m = m_max
        dl = 1.0 / m
        r_sub = 0.5 * dl * dl * M2
        cen = -1.0 + dl * (2 * np.arange(m) + 1)
        uu, vv = np.meshgrid(cen, cen, indexing="ij")
        uu, vv = uu.ravel(), vv.ravel()
        eps = 1e-4
        Pc = I(uu, vv)
        Pu = (I(uu + eps, vv) - I(uu - eps, vv)) / (2 * eps) * dl
        Pv = (I(uu, vv + eps) - I(uu, vv - eps)) / (2 * eps) * dl
        B, D, _, dlb = self.tr.BGA(Pc)
        g = self.s * (Pc[:, 0] * B[:, 1] - Pc[:, 1] * B[:, 0])
        dg = self.s * (np.stack([B[:, 1], -B[:, 0], 0 * B[:, 0]], -1) + Pc[:, 0:1] * D[:, 1, :] - Pc[:, 1:2] * D[:, 0, :])
        # Hessian bound from neighbouring centres (grid neighbours in u and v)
        gi = dg.reshape(m, m, 3); Xi = Pc.reshape(m, m, 3)
        q1 = np.linalg.norm(gi[1:] - gi[:-1], axis=-1) / np.maximum(np.linalg.norm(Xi[1:] - Xi[:-1], axis=-1), 1e-300)
        q2 = np.linalg.norm(gi[:, 1:] - gi[:, :-1], axis=-1) / np.maximum(np.linalg.norm(Xi[:, 1:] - Xi[:, :-1], axis=-1), 1e-300)
        G2 = 2.0 * float(max(np.max(q1), np.max(q2)))
        rq = np.linalg.norm(Pu, axis=1) + np.linalg.norm(Pv, axis=1) + rho_q + r_sub
        bound = (g + np.abs(np.sum(dg * Pu, 1)) + np.abs(np.sum(dg * Pv, 1)) + np.linalg.norm(dg, axis=1) * (rho_q + r_sub)
                 + 0.5 * G2 * rq * rq)
        ext = float(np.max(np.linalg.norm(X9 - X9[4][None], axis=1))) + rho_q
        return bool(np.all(bound < -1e-9 * (np.abs(g) + 1.0)) and np.all(rq < 0.5 * dlb) and dpath >= COIL_GUARD * ext)

    def base_test(self, items, N, n_check=4):
        """items: (piece, node idx (9,)) with every stencil line lost.  The 9 lines (from their recorded states: one
        GBS step from the last state before each arc length; full traces when a target lies beyond a record) and the
        4 at (+-1/2, +-1/2) (traced) are taken to n_check arc lengths between the centre's first f <= -beta and its
        lost point; g < 0 is tested on the affine enclosure of the piece's image at each, then on the curved one: the
        piece is certified when one passes (returns the arc length that passed, or False)."""
        if not items:
            return []
        n = len(items)
        SS = np.zeros((n, n_check))
        X4 = []
        for t_, (pc, idx) in enumerate(items):
            cR, cZ, eta = self.piece_geom(*pc)
            s1, s2 = float(N["sf"][idx[4]]), float(N["se"][idx[4]])
            if not np.isfinite(s1):
                s1 = s2
            SS[t_] = np.linspace(s1, s2, n_check) if s2 > s1 else np.full(n_check, s2)
            X4.append(np.stack([cR + 0.5 * eta * np.array([-1, 1, -1, 1.0]), np.zeros(4),
                                cZ + 0.5 * eta * np.array([-1, -1, 1, 1.0])], -1))
        # the 9 stencil lines from their records
        T9 = np.full((n * 9, MAXS, 6), np.nan)
        have = np.ones(n * 9, bool)
        for t_, (pc, idx) in enumerate(items):
            for m_, ix in enumerate(idx):
                tj = self.traj.get(int(ix))
                if tj is None:
                    have[9 * t_ + m_] = False
                else:
                    T9[9 * t_ + m_] = tj
        X9, E9, ok9, D9 = self.tr.positions(T9, np.repeat(SS, 9, axis=0))
        ok9 &= have
        X9 = X9.reshape(n, 9, n_check, 3); E9 = E9.reshape(n, 9, n_check); ok9 = ok9.reshape(n, 9)
        D9 = D9.reshape(n, 9, n_check)
        redo = np.flatnonzero(~np.all(ok9, axis=1))
        if len(redo):
            X0 = []
            for t_ in redo:
                cR, cZ, eta = self.piece_geom(*items[t_][0])
                R, Z, _, _ = _stencil(cR, cZ, eta)
                X0.append(np.stack([R, 0 * R, Z], -1))
            Xr, er, okr, dmr = self.tr.arc(np.concatenate(X0), np.repeat(SS[redo], 9, axis=0))
            D9[redo] = np.repeat(dmr.reshape(len(redo), 9, 1), n_check, axis=2)
            X9[redo] = Xr.reshape(len(redo), 9, n_check, 3)
            E9[redo] = er.reshape(len(redo), 9, n_check)
            ok9[redo] = okr.reshape(len(redo), 9)
        out = [False] * n
        # the affine enclosure first (no extra lines); the 4 extra lines only for the pieces that need the curved one
        need = []
        u9 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float); v9 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
        for t_ in range(n):
            if not np.all(ok9[t_]):
                continue
            pc, idx = items[t_]
            eta = self.piece_geom(*pc)[2]
            for m in range(n_check - 1, -1, -1):
                S = float(SS[t_, m])
                ok, _ = self.base_enclosure_test(X9[t_, :, m, :], E9[t_, :, m], eta=eta, S=S,
                                                 dpath=float(np.min(D9[t_, :, m])))
                if ok:
                    out[t_] = S
                    break
            if out[t_] is False:
                rl = [_affine_enclosure(X9[t_, :, m, :], u9, v9) for m in range(n_check)]
                if any(r_[3] <= QNONLIN * (np.linalg.norm(r_[1]) + np.linalg.norm(r_[2])) for r_ in rl):
                    need.append(t_)
        if not need:
            return out
        Xe4, e4, ok4, dm4 = self.tr.arc(np.concatenate([X4[t_] for t_ in need]), np.repeat(SS[need], 4, axis=0))
        Xe4 = Xe4.reshape(len(need), 4, n_check, 3); e4 = e4.reshape(len(need), 4, n_check)
        ok4 = ok4.reshape(len(need), 4); dm4 = dm4.reshape(len(need), 4)
        for q_, t_ in enumerate(need):
            if not np.all(ok4[q_]):
                continue
            pc, idx = items[t_]
            eta = self.piece_geom(*pc)[2]
            for m in range(n_check - 1, -1, -1):
                S = float(SS[t_, m])
                dp_ = float(min(np.min(D9[t_, :, m]), np.min(dm4[q_])))
                ok = self.base_curved_test(X9[t_, :, m, :], Xe4[q_, :, m, :],
                                           2.0 * float(max(np.max(E9[t_, :, m]), np.max(e4[q_, :, m]))) + 1e-12,
                                           eta=eta, S=S, dpath=dp_)
                if ok:
                    self.stats["base_quad"] = self.stats.get("base_quad", 0) + 1
                    out[t_] = S
                    break
        return out

    def period_enclosure(self, idx, N):
        u = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float)
        v = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
        Xp = N["Xp"][idx]
        P = np.stack([np.hypot(Xp[:, 0], Xp[:, 1]), Xp[:, 2]], -1)
        c, Au, Av, rn = _affine_enclosure(P, u, v)
        rho = 2.0 * LEB33 * rn + 2.0 * float(np.max(N["ep"][idx])) + 1e-12
        return c, Au, Av, rn, rho

    def period_test(self, items, N, buf):
        """(piece, idx) whose 9 stencil lines crossed the plane: (ok, covered fraction).  First the affine enclosure
        (parallelogram + rho); where it fails, the curved one: the biquadratic interpolant I of the 9 images, its error
        sampled at the 4 points (+-1/2, +-1/2) (the children's centres, traced and cached: they serve the children
        too), rho_q = Q_FAC x the largest sampled error + the trace error, and I([-1, 1]^2) covered by m x m
        sub-parallelograms with the exact second-derivative bound of the polynomial I; every fine cell within reach
        must be certified (the piece's own cells are not, so a piece mapped onto itself is never certified)."""
        Gf = self.Gf
        out = [None] * len(items)
        quad = []
        for t_, (pc, idx) in enumerate(items):
            c, Au, Av, rn, rho = self.period_enclosure(idx, N)
            lin = np.linalg.norm(Au) + np.linalg.norm(Av)
            rho += self.tau_field(float(np.max(N["sp"][idx])), lin, self.piece_geom(*pc)[2])
            # the period map is smooth on the piece when every line keeps f >= F_SMOOTH up to the plane (no turning
            # point of phi on the way)
            self.stats["pt_tested"] = self.stats.get("pt_tested", 0) + 1
            if not np.min(N["fp"][idx]) >= F_SMOOTH:
                self.stats["pt_fsmooth"] = self.stats.get("pt_fsmooth", 0) + 1
                out[t_] = (False, 0.0)
                continue
            if not np.min(N["dp"][idx]) >= COIL_GUARD_P * (lin + rho):
                self.stats["pt_guard"] = self.stats.get("pt_guard", 0) + 1
                out[t_] = (False, 0.0)
                continue
            frac = 0.0
            if rn <= NONLIN * lin:
                n = _cells_near_parallelogram(c[0], c[1], Au[0], Au[1], Av[0], Av[1], rho, Gf.R0, Gf.Z0, Gf.h, Gf.nR,
                                              Gf.nZ, buf)
                if n > 0:
                    cov = self.cert[buf[:n, 0], buf[:n, 1]]
                    if np.all(cov):
                        out[t_] = (True, 1.0)
                        continue
                    frac = float(np.mean(cov))
            if rn > 0.02 * lin and rn <= QNONLIN * lin:
                quad.append(t_)
            elif rn > QNONLIN * lin:
                self.stats["pt_nonlin"] = self.stats.get("pt_nonlin", 0) + 1
            else:
                self.stats["pt_cover"] = self.stats.get("pt_cover", 0) + 1
            out[t_] = (False, frac)
        if quad:
            KI, KJ = [], []
            for t_ in quad:
                a_, b_ = self.extra_keys(*items[t_][0])
                KI.append(a_); KJ.append(b_)
            ex = self.trace_keys(np.concatenate(KI), np.concatenate(KJ), deadline=self._deadline).reshape(-1, 4)
            N = self.node_arrays()
            for t_, ix4 in zip(quad, ex):
                pc, idx = items[t_]
                if np.any(ix4 < 0):
                    continue
                if not np.all(np.isfinite(N["sp"][ix4])) or np.min(N["fp"][ix4]) < F_SMOOTH:
                    continue
                ext_ = float(np.max(np.linalg.norm(N["Xp"][idx] - N["Xp"][idx[4]][None], axis=1)))
                if min(np.min(N["dp"][idx]), np.min(N["dp"][ix4])) < COIL_GUARD_P * ext_:
                    continue
                Xp = N["Xp"][idx]; X4 = N["Xp"][ix4]
                P9 = np.stack([np.hypot(Xp[:, 0], Xp[:, 1]), Xp[:, 2]], -1)
                P4 = np.stack([np.hypot(X4[:, 0], X4[:, 1]), X4[:, 2]], -1)
                tau = 2.0 * float(max(np.max(N["ep"][idx]), np.max(N["ep"][ix4]))) + 1e-12
                lin_ = float(np.linalg.norm(P9[5] - P9[3]) + np.linalg.norm(P9[7] - P9[1])) / 2.0
                tau += self.tau_field(float(np.max(N["sp"][idx])), lin_, self.piece_geom(*pc)[2])
                ok, frac = _biquad_cover(P9, P4, tau, self.cert, Gf.R0, Gf.Z0, Gf.h, Gf.nR, Gf.nZ, buf)
                out[t_] = (bool(ok), float(frac))
                if ok:
                    self.stats["period_quad"] = self.stats.get("period_quad", 0) + 1
        return out

    def certify_piece(self, pc):
        si, sj = self.fine_range(*pc)
        self.cert[si, sj] = True
        if self.sym:
            nZ = self.Gf.nZ
            j0, j1 = sj.start, sj.stop
            self.cert[si, nZ - j1:nZ - j0] = True

    def split(self, pc):
        l, i, j = pc
        self.pending.discard(pc)
        kids = [(l + 1, 2 * i + di, 2 * j + dj) for di in (0, 1) for dj in (0, 1)]
        for k_ in kids:
            si, sj = self.fine_range(*k_)
            if not np.all(self.cert[si, sj]):
                self.pending.add(k_)
        self.stats["split"] += 1

    def rounds(self, deadline=None, max_rounds=400, buf=None):
        """Trace the stencils of the pending pieces (cached) and apply the base and period certificates until nothing
        changes (the period certificates propagate loss one period per round)."""
        self._deadline = deadline
        if buf is None:
            buf = np.zeros((1 << 18, 2), np.int64)
        for rnd in range(max_rounds):
            if deadline is not None and time.time() > deadline:
                return False
            self.stats["rounds"] += 1
            for pc in list(self.pending):
                si, sj = self.fine_range(*pc)
                if np.all(self.cert[si, sj]):
                    self.pending.discard(pc)
            pcs = sorted(self.pending)
            if not pcs:
                return True
            if self.det_tr is not None:
                n_all = len(pcs)
                pcs = self.det_select(pcs, deadline, buf)
                self.n_skipped = n_all - len(pcs)
                if not pcs:
                    return True
            KI, KJ = [], []
            for pc in pcs:
                a_, b_ = self.stencil_keys(*pc)
                KI.append(a_); KJ.append(b_)
            idx = self.trace_keys(np.concatenate(KI), np.concatenate(KJ), deadline=deadline).reshape(-1, 9)
            full = np.all(idx >= 0, axis=1)
            pcs = [pc for pc, f_ in zip(pcs, full) if f_]
            idx = idx[full]
            if not len(pcs):
                return False
            N = self.node_arrays()
            st = N["status"][idx]
            crossed = np.isfinite(N["sp"][idx])
            lost = st == ST_LOST
            newly = []
            bi = [(pc, ix) for pc, ix, lo in zip(pcs, idx, lost) if np.all(lo) and pc not in self.base_done]
            for lo in range(0, len(bi), 128):
                if deadline is not None and time.time() > deadline:
                    break
                part = bi[lo:lo + 128]
                for (pc, _), okb in zip(part, self.base_test(part, N)):
                    self.base_done[pc] = okb
                    self.stats["base" if okb is not False else "base_fail"] += 1
            newly += [pc for pc in pcs if self.base_done.get(pc, False) is not False]
            pi = [(pc, ix) for pc, ix, cr in zip(pcs, idx, crossed)
                  if np.all(cr) and self.base_done.get(pc, False) is False]
            for (pc, _), (okp, frac) in zip(pi, self.period_test(pi, N, buf)):
                if self.det_tr is None:
                    self.cover[pc] = frac
                if okp:
                    newly.append(pc)
                    self.period_pieces.append(pc)
                    self.stats["period"] += 1
            if self.det_tr is None:
                for pc, lo_ in zip(pcs, lost):
                    self.anylost[pc] = bool(np.any(lo_))
            for pc in newly:
                self.certify_piece(pc)
                self.pending.discard(pc)
            if self.verbose:
                dt_ = self.det_tr.t if self.det_tr is not None else 0.0
                print(f"   round {self.stats['rounds']}: certified fine cells {int(self.cert.sum())}, pending "
                      f"{len(self.pending)} (skipped {self.n_skipped}), nodes {len(self.cache)} (detected "
                      f"{len(self.det_cache)}), tracer {self.tr.t:.1f}s (detection {dt_:.1f}s), {self.stats}", flush=True)
            if not newly:
                return True
        return True

    def has_wall(self):
        Gf = self.Gf
        ia = int(math.floor((self.a[0] - Gf.R0) / Gf.h)); ja = int(math.floor((self.a[1] - Gf.Z0) / Gf.h))
        comp = component_bound(self.cert, (ia, ja))
        return comp is not None and not comp[1], comp

    def run(self, deadline=None, rings=None, grow=None):
        """Level-0 pieces ring by ring from the outside in (see below); then, level by level, the pending pieces on the boundary of K (inside K, next to a certified cell) are split
        and the rounds repeated: the work goes where it lowers the bound."""
        buf = np.zeros((1 << 18, 2), np.int64)
        if grow is not None:
            # the band first; then, while the exact component K reaches level-0 cells not yet processed (outside the
            # band, not in the core), those cells join and the rounds repeat
            done, incore = grow["done"], grow["incore"]
            for _ in range(50):
                # refine the processed pieces until nothing promising is left (a piece with a lost detection line, or
                # a period image already partly certified, is split down to the finest level)
                for _lvl in range(self.D + 1):
                    if not self.rounds(deadline, buf=buf):
                        return
                    todo = [pc for pc in self.pending if pc[0] < self.D and pc not in self.split_done
                            and (self.anylost.get(pc, False) or self.cover.get(pc, 0.0) >= self.split_frac)]
                    if not todo:
                        break
                    for pc in todo:
                        self.split_done.add(pc)
                        self.split(pc)
                    if self.verbose:
                        print(f"   refine band: split {len(todo)} pieces (levels {sorted(set(p[0] for p in todo))})",
                              flush=True)
                ok, comp = self.has_wall()
                if comp is None:
                    return
                K, _ = comp
                Kc = K.reshape(self.G0.nR, self.f, self.G0.nZ, self.f).any(axis=(1, 3))
                new = Kc & ~done & ~incore
                if not new.any():
                    break
                done |= new
                self.pending |= set((0, int(i_), int(j_)) for i_, j_ in zip(*np.nonzero(new)))
                if self.verbose:
                    print(f"   grow: {int(new.sum())} level-0 cells added outward", flush=True)
            rings = []
        rings = rings if rings is not None else [set(self.pending)]
        self.rings_used = 0
        # from the outside in: the outer rings (lines lost quickly) close the wall, the inner ones are added while
        # they still yield certified cells (a ring with less than RING_STOP of its cells certified ends the descent:
        # the confined region); the period certificates then propagate loss inward
        for rg in rings[::-1]:
            before = int(self.cert.sum())
            self.pending |= rg
            self.rings_used += 1
            if not self.rounds(deadline, buf=buf):
                return
            n_fine = len(rg) * self.f * self.f
            gained = int(self.cert.sum()) - before
            if self.verbose:
                print(f"   ring {self.rings_used}: {len(rg)} cells, certified {gained / max(n_fine, 1):.2f} of its area",
                      flush=True)
            if gained < RING_STOP * n_fine and self.rings_used > 1:
                break
        for _ in range(4 * self.D):
            ok, comp = self.has_wall()
            if comp is None:
                return
            K, _ = comp
            # fine cells of K next to a certified cell
            Cp = np.pad(self.cert, 1)
            nb = np.zeros_like(self.cert)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb |= Cp[1 + di:1 + di + self.cert.shape[0], 1 + dj:1 + dj + self.cert.shape[1]]
            edge = K & nb
            todo = []
            for pc in self.pending:
                if pc[0] >= self.D:
                    continue
                # promising only: a stencil line lost (mixed, or a failed base test) or a period image already partly
                # in certified cells; a piece whose lines all stay and map into uncertified cells is left whole
                if not (self.anylost.get(pc, True) or self.cover.get(pc, 1.0) >= self.split_frac):
                    continue
                si, sj = self.fine_range(*pc)
                if np.any(edge[si, sj]):
                    todo.append(pc)
            if not todo:
                return
            for pc in todo:
                self.split(pc)
            if self.verbose:
                print(f"   refine: split {len(todo)} boundary pieces (levels {sorted(set(p[0] for p in todo))})", flush=True)
            if not self.rounds(deadline, buf=buf):
                return


def component_bound(cert: np.ndarray, start):
    """8-connected component of the uncertified cells containing start (i, j); None if it reaches the grid border."""
    from collections import deque
    nR, nZ = cert.shape
    seen = np.zeros_like(cert)
    if cert[start]:
        return None
    dq = deque([start])
    seen[start] = True
    border = False
    while dq:
        i, j = dq.popleft()
        if i == 0 or j == 0 or i == nR - 1 or j == nZ - 1:
            border = True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if 0 <= a < nR and 0 <= b < nZ and not seen[a, b] and not cert[a, b]:
                    seen[a, b] = True
                    dq.append((a, b))
    return seen, border


def region_flux(tr, grid: CellGrid, K, sign, ng=6):
    """-s x the loop integral of A over the boundary of the cell set K (counter-clockwise; Stokes) by Gauss-Legendre
    with ng points per edge; the error estimate from ng // 2 points; plus the (s B_phi)^- bound over K's cells."""
    nR, nZ = K.shape
    h = grid.h
    edges = []          # (x0, y0, x1, y1) counter-clockwise around K
    Kp = np.pad(K, 1)
    ii, jj = np.nonzero(K)
    for i, j in zip(ii, jj):
        x0 = grid.R0 + i * h; y0 = grid.Z0 + j * h
        x1 = x0 + h; y1 = y0 + h
        if not Kp[i + 1, j]:      # below (j - 1) empty: bottom edge left -> right
            edges.append((x0, y0, x1, y0))
        if not Kp[i + 2, j + 1]:  # right empty: right edge bottom -> top
            edges.append((x1, y0, x1, y1))
        if not Kp[i + 1, j + 2]:  # top empty: top edge right -> left
            edges.append((x1, y1, x0, y1))
        if not Kp[i, j + 1]:      # left empty: left edge top -> bottom
            edges.append((x0, y1, x0, y0))
    E = np.array(edges, float)

    def loop(n):
        xg, wg = np.polynomial.legendre.leggauss(n)
        t = 0.5 * (xg + 1.0); wt = 0.5 * wg
        P = E[:, None, :2] + t[None, :, None] * (E[:, None, 2:] - E[:, None, :2])
        X = np.stack([P[..., 0], 0 * P[..., 0], P[..., 1]], -1).reshape(-1, 3)
        A = tr.BGA(X)[2].reshape(len(E), n, 3)
        d = E[:, 2:] - E[:, :2]
        return float(np.sum(wt[None, :] * (A[..., 0] * d[:, None, 0] + A[..., 2] * d[:, None, 1])))
    I6 = loop(ng)
    I3 = loop(max(2, ng // 2))
    F = -sign * I6
    # (s B_phi)^- over K: the cells whose centre value minus the gradient's reach can be negative
    cR, cZ = grid.centre(ii, jj)
    X = np.stack([cR, 0 * cR, cZ], -1)
    B, DB, _, _ = tr.BGA(X)
    bp = sign * B[:, 1]
    gbp = np.linalg.norm(DB[:, 1, :], axis=1)
    reach = 2.0 * gbp * h / math.sqrt(2.0)
    # cells next to a coil crossing: the analytic disk bound (as in lost_curve_bound)
    from .barrier import coil_crossings
    cc = coil_crossings(tr.source)
    rho_c = 2.0 * h * math.sqrt(2.0)
    near = np.zeros(len(cR), bool)
    disk_corr = 0.0
    Imax = max(float(np.max(np.abs(p.current))) for p in tr.source.parts if p.kind == "fourier")
    for c_ in cc:
        nb_ = np.hypot(cR - c_[0], cZ - c_[1]) < rho_c + h
        if not nb_.any():
            continue
        near |= nb_
        th_ = 2 * np.pi * np.arange(16) / 16
        ring = np.stack([c_[0] + rho_c * np.cos(th_), 0 * th_, c_[1] + rho_c * np.sin(th_)], -1)
        Bb = float(np.max(np.linalg.norm(tr.B(ring)[:, :3], axis=1)))
        disk_corr += 1.5 * 4e-7 * math.pi * Imax * (rho_c + h) + 2.0 * Bb * math.pi * (rho_c + h) ** 2
    neg = (bp - reach < 0) & ~near
    corr = float(np.sum(np.where(near, 0.0, np.maximum(reach - bp, 0.0))) * h * h) + disk_corr
    return dict(F=F, quad_err=abs(I6 - I3), neg_corr=corr, n_neg_cells=int(neg.sum()), n_edges=len(E),
                coil_disk_corr=disk_corr)


def edge_upper_bound(source, axis_point, T_last=None, core=None, scale_poly=None, request=None, engine=None, budget_s=300.0,
                     h0_rel=1 / 8, depth=3, reach_d=1.3, win_periods=1.25, d_fail_rel=0.03, eps=1e-10, verbose=False,
                     sign=None, tol=1e-10, with_evidence=True, evidence_s=60.0, verified_sections=None,
                     ring_edges=(1.3, 1.6, 2.0, 2.5, 3.2, 4.0), detect=True, det_win=2.5, band=3):
    """The survivor-flux upper bound (module docstring) around axis_point (R, Z) in phi = 0.
    T_last: the outermost verified torus, or core: a polygon (n, 2) in phi = 0 believed confined -- level-0 cells
    inside it are not processed (efficiency only: uncertified cells count in full, so any core is sound);
    scale_poly: a polygon setting the scales when it is not the core (default: the core); its mean
    radius about a sets the level-0 cell size h0 = h0_rel x r_mean (finest hf = h0 / 2^depth); the box reaches
    its largest radius + reach_d x the distance d_last of T_last (or the core) to the coils."""
    from .verify import torus_points
    from scipy.spatial import cKDTree
    t0 = time.time()
    deadline = t0 + budget_s
    a = np.asarray(axis_point, float)
    out = dict(kind="edge_survivor", ok=False, axis_point=a.tolist(), request=request)
    why = supported(source)
    if why is not None:
        out["reason"] = why
        return out
    cp = []
    for p in source.parts:
        if p.kind == "fourier":
            x, _ = p.points(np.arange(1024) / 1024)
            cp.append(x.reshape(-1, 3))
    cp = np.concatenate(cp) if cp else source.sample_points()
    tree = cKDTree(cp)
    if T_last is not None:
        th = 2 * np.pi * np.arange(1024) / 1024
        Rl, Zl = T_last.geometry_on(th, np.zeros_like(th))
        core = np.stack([Rl, Zl], -1)
        tt, pp = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                             np.linspace(0, 2 * np.pi / source.nfp, 32), indexing="ij")
        xt, _, _ = torus_points(T_last, tt.ravel(), pp.ravel())
        d_last = float(np.min(tree.query(xt)[0]))
        out["F_last"] = abs(float(T_last.flux))
    sp_ = core if scale_poly is None else scale_poly
    if sp_ is not None:
        sp_ = np.asarray(sp_, float)
        rad = np.hypot(sp_[:, 0] - a[0], sp_[:, 1] - a[1])
        r_max = float(np.max(rad))
        from .barrier import signed_area
        r_mean = math.sqrt(abs(signed_area(sp_)) / math.pi)        # the section's effective minor radius
        if T_last is None:
            d_last = float(np.min(tree.query(np.stack([sp_[:, 0], 0 * sp_[:, 0], sp_[:, 1]], -1))[0]))
        if core is not None:
            core = np.asarray(core, float)
    else:
        d_last = float(np.min(tree.query(np.array([[a[0], 0.0, a[1]]]))[0]))
        r_mean = r_max = 0.4 * d_last
    d_fail = d_fail_rel * d_last
    eng = engine
    tr = ExactTracer(source, source.nfp, 1.0, d_fail=d_fail, tol=tol, eps=eps, engine=eng)
    if sign is None:
        sign = float(np.sign(tr.B(np.array([[a[0], 0.0, a[1]]]))[0, 1]))
    tr.s = float(sign)
    det = None
    if detect:
        det = ExactTracer(source, source.nfp, sign, d_fail=d_fail, tol=1e-5, eps=1e-6)
    t_chk = time.time()
    tr.check()
    out["t_field_check"] = time.time() - t_chk
    h0 = h0_rel * r_mean
    ext = r_max + reach_d * d_last
    nR = int(math.ceil(2 * ext / h0)) + 2
    nZ = 2 * (int(math.ceil(ext / h0)) + 1)
    G = CellGrid(a[0] - ext - h0, h0, nR, nZ)
    period_arc = 2 * math.pi * a[0] / source.nfp
    C = Certifier(eng, source, a, sign, G, tr, period_arc, depth=depth, stellsym=getattr(source, "stellsym", True),
                  win_periods=win_periods, verbose=verbose, det_tracer=det)
    ii, jj = np.meshgrid(np.arange(nR), np.arange(nZ), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    cR, cZ = G.centre(ii, jj)
    if core is not None:
        incore = np.ones(len(cR), bool)
        for du in (-0.5, 0.5):
            for dv in (-0.5, 0.5):
                incore &= _poly_in(core, np.stack([cR + du * h0, cZ + dv * h0], -1))
    else:
        incore = np.zeros(len(cR), bool)
    # DETECTION MAP (selection only): the centre line of every level-0 cell outside the core, traced with the loose
    # tracer, forward and -- the mirror cell's forward line -- backward; a cell is "lost-looking" when either line
    # reaches f <= -beta within the window.  K_det = the axis' component of the rest.  The exact work goes to a band
    # of `band` cells around K_det (and grows outward wherever the exact component reaches past it).
    t1 = time.time()
    out.update(h0=h0, hf=h0 / 2 ** depth, grid=[nR, nZ], depth=depth, d_last=d_last, d_fail=d_fail,
               n_cells0=int(np.sum(~incore)), n_core0=int(incore.sum()), sign=sign, nodes=int(len(tr.P[0])),
               nodes_coarse=int(np.sum(tr.P[3] >> tr.P[4])), calib_dk=np.round(tr.P[7][:, 1:4], 4).tolist()[:4],
               check_err_multilevel=tr.check_err)
    dtr = det if det is not None else tr
    sel = ~incore
    Xc = np.stack([cR[sel], 0 * cR[sel], cZ[sel]], -1)
    od = dtr.run(Xc, det_win * period_arc, ext=1.0)
    lost_c = np.zeros(len(cR), bool)
    lost_c[np.flatnonzero(sel)] = od["status"] == ST_LOST
    L0 = lost_c.reshape(nR, nZ)
    if C.sym:
        L0 = L0 | L0[:, ::-1]
    comp_d = component_bound(L0, (int(math.floor((a[0] - G.R0) / h0)), int(math.floor((a[1] - G.Z0) / h0))))
    out["detection_map"] = dict(lost_cells=int(L0.sum()), cells=int(sel.sum()), window_periods=det_win,
                                t=time.time() - t1)
    if comp_d is None or comp_d[1]:
        out["timings"] = dict(setup=t1 - t0, detection_map=time.time() - t1)
        out["reason"] = "no wall: the detected lost cells do not enclose the axis inside the box"
        out["stats"] = dict(C.stats)
        return out
    Kd = comp_d[0]

    def dilate(M, r_):
        P_ = np.pad(M, r_)
        o_ = np.zeros_like(M)
        for di in range(-r_, r_ + 1):
            for dj in range(-r_, r_ + 1):
                o_ |= P_[r_ + di:r_ + di + M.shape[0], r_ + dj:r_ + dj + M.shape[1]]
        return o_
    inc2 = incore.reshape(nR, nZ)
    bandm = dilate(Kd, band) & ~dilate(Kd & ~dilate(~Kd, 2), 0) & ~inc2      # within `band` of K_det, not deep inside
    C.pending = set((0, int(i_), int(j_)) for i_, j_ in zip(*np.nonzero(bandm)))
    out["band_cells"] = int(bandm.sum())
    C.run(deadline=deadline, rings=None, grow=dict(done=bandm.copy(), incore=inc2))
    C.rings_used = 0
    out["timings"] = dict(setup=t1 - t0, certify=time.time() - t1, tracer=tr.t, tracer_arc=tr.t_arc, n_arc=tr.n_arc,
                          n_lines=tr.n_lines, detection=det.t if det is not None else 0.0,
                          n_detected=det.n_lines if det is not None else 0)
    Gf = C.Gf
    ia = int(math.floor((a[0] - Gf.R0) / Gf.h)); ja = int(math.floor((a[1] - Gf.Z0) / Gf.h))
    if T_last is not None or verified_sections:
        # NEGATIVE TEST (every run with a verified torus): no certified cell may meet a verified torus's section --
        # its points are on an invariant torus, never lost.  Any hit is a false certificate: the run is refused.
        secs = list(verified_sections or [])
        if T_last is not None:
            th = 2 * np.pi * np.arange(8192) / 8192
            Rl, Zl = T_last.geometry_on(th, np.zeros_like(th))
            secs.append(np.stack([Rl, Zl], -1))
        hits = 0
        for sec in secs:
            ii_ = np.floor((sec[:, 0] - Gf.R0) / Gf.h).astype(int); jj_ = np.floor((sec[:, 1] - Gf.Z0) / Gf.h).astype(int)
            inb = (ii_ >= 0) & (ii_ < Gf.nR) & (jj_ >= 0) & (jj_ < Gf.nZ)
            hits += int(np.sum(C.cert[ii_[inb], jj_[inb]]))
        out["verified_section_hits"] = hits
        if hits:
            out["reason"] = f"FALSE CERTIFICATE: {hits} certified cells meet a verified torus section"
            out["false_certificate"] = True
            out["stats"] = dict(C.stats)
            return out
    comp = component_bound(C.cert, (ia, ja))
    out["stats"] = dict(C.stats, certified_fine=int(C.cert.sum()), pending=len(C.pending), traced=len(C.cache),
                        steps=tr.n_steps)
    out["_cert"] = C.cert
    out["_grid"] = (Gf.R0, Gf.Z0, Gf.h, Gf.nR, Gf.nZ)
    out["_C"] = C
    if comp is None:
        out["reason"] = "the axis cell is certified lost"
        return out
    K, border = comp
    out["_K"] = K
    out["K_cells"] = int(K.sum())
    if border:
        out["reason"] = "no closed wall of certified lost cells inside the box"
        return out
    fl = region_flux(tr, Gf, K, sign)
    F_up = fl["F"] + fl["neg_corr"] + 2 * fl["quad_err"] + 1e-14 * abs(fl["F"])
    out.update(ok=True, F_up=float(F_up), flux=fl)
    if T_last is not None:
        out["ratio"] = F_up / abs(float(T_last.flux))
        if F_up < abs(float(T_last.flux)):
            out.update(ok=False, false_certificate=True, reason="FALSE CERTIFICATE: F_up below the verified torus")
    if request is not None:
        out["holds_request"] = bool(F_up < abs(request))
    t_ev = time.time()
    if with_evidence:
        try:
            out["evidence"] = evidence(C, source, deadline=time.time() + evidence_s)
        except Exception as e:  # noqa: BLE001
            out["evidence"] = dict(error=repr(e))
    out["timings"]["evidence"] = time.time() - t_ev
    out["timings"]["total"] = time.time() - t0
    return out


# ============================================================================ evidence (independent checks)
def _ref_arclength(Bref, x0, sig, S, rtol=1e-12, atol=1e-13):
    """scipy DOP853 in arc length on the reference field from x0 to arc length S: the end point (3,), or None."""
    from scipy.integrate import solve_ivp

    def rhs(_, y):
        b = Bref(y[None])[0]
        return sig * b / np.linalg.norm(b)
    try:
        sol = solve_ivp(rhs, (0.0, float(S)), np.asarray(x0, float), method="DOP853", rtol=rtol, atol=atol)
    except Exception:  # noqa: BLE001
        return None
    return sol.y[:, -1] if sol.success else None


def _ref_cross(Bref, x0, sig, phi_t, S_max, rtol=1e-12, atol=1e-13):
    """scipy DOP853 in arc length on the reference field from x0 (in phi = 0) to the first crossing of phi = phi_t
    (event on the unwrapped angle, dense-output root): the crossing point (3,), or None."""
    from scipy.integrate import solve_ivp

    def rhs(_, y):
        b = Bref(y[None, :3])[0]
        bm = np.linalg.norm(b)
        v = sig * b / bm
        R2 = y[0] * y[0] + y[1] * y[1]
        return np.array([v[0], v[1], v[2], (y[0] * v[1] - y[1] * v[0]) / R2])

    ev = lambda _, y: y[3] - phi_t
    ev.terminal = True
    ev.direction = 1.0
    y0 = np.array([x0[0], x0[1], x0[2], math.atan2(x0[1], x0[0])])
    try:
        sol = solve_ivp(rhs, (0.0, float(S_max)), y0, method="DOP853", rtol=rtol, atol=atol, events=ev)
    except Exception:  # noqa: BLE001
        return None
    if sol.status != 1 or not len(sol.y_events[0]):
        return None
    return sol.y_events[0][0][:3]


def evidence(C: Certifier, source, n_period=6, n_lost=4, n_enc=8, seed=0, deadline=None, ref=None):
    """Independent checks of a finished certification (recorded, not part of the bound):
      period     sampled plane crossings vs the reference tracer (torux.reference.trace.poincare: Gauss-Legendre IRK in
                 phi on the fast64 field): distance against 2 x the step-estimate sum (ratio, must be < 1);
      lost       sampled lost points vs scipy DOP853 in arc length on the reference field to the same arc length:
                 distance, and g = s R B_phi there by the reference field (must be < 0);
      enclosure  5 x 5 stencils of sampled certified pieces (period and base): the largest distance of an extra
                 point from the affine model, over the enclosure's rho (must be < 1)."""
    from .reference.fast64 import Field
    from .reference.trace import poincare
    rng = np.random.default_rng(seed)
    F = ref if ref is not None else Field(source)
    N = C.node_arrays()
    out = {}
    s = C.s
    inv = {v_: k_ for k_, v_ in C.cache.items() if isinstance(k_, tuple) and len(k_) == 2}
    # period crossings
    cr = np.flatnonzero(np.isfinite(N["sp"]))
    rows = []
    for t in (rng.choice(cr, min(n_period, len(cr)), replace=False) if len(cr) else []):
        if deadline is not None and time.time() > deadline:
            break
        R0, Z0 = C.node_xy(*inv[int(t)])
        # two independent references: the Gauss-Legendre IRK tracer in phi (torux.reference.trace; its own
        # step-doubling error ref_err -- it is singular where B_phi nearly vanishes on the way) and scipy DOP853 in arc
        # length on the reference field with an event at the plane (no such singularity)
        o = poincare(F, np.array([R0]), np.array([Z0]), np.array([2 * np.pi / source.nfp]), 1, h=2 * np.pi / 1024)
        Rr, Zr = float(o["R"][0, 0, 0]), float(o["Z"][0, 0, 0])
        Xp = N["Xp"][t]
        d = math.hypot(math.hypot(Xp[0], Xp[1]) - Rr, Xp[2] - Zr)
        xr = _ref_cross(F.B, np.array([R0, 0.0, Z0]), s, 2 * np.pi / source.nfp, 3.0 * C.period_arc)
        d2 = float(np.linalg.norm(xr - Xp)) if xr is not None else None
        rows.append(dict(R0=R0, Z0=Z0, dist_irk=d, ref_err_irk=float(o["err"][0, 0, 0]), dist=d2,
                         est=2 * float(N["ep"][t]) + 1e-12))
    out["period"] = rows
    # the ratio of the DOP853 reference distance to our estimate (+ 1e-10: the reference's own tolerance)
    out["period_ratio_max"] = max((r_["dist"] / (r_["est"] + 1e-10) for r_ in rows if r_["dist"] is not None),
                                  default=None)
    # lost points
    lo = np.flatnonzero(N["status"] == ST_LOST)
    rows = []
    for t in (rng.choice(lo, min(n_lost, len(lo)), replace=False) if len(lo) else []):
        if deadline is not None and time.time() > deadline:
            break
        R0, Z0 = C.node_xy(*inv[int(t)])
        xe = _ref_arclength(F.B, [R0, 0.0, Z0], s, float(N["se"][t]))
        if xe is None:
            rows.append(dict(R0=R0, Z0=Z0, ok=False))
            continue
        b = F.B(xe[None])[0]
        gref = s * (xe[0] * b[1] - xe[1] * b[0])
        rows.append(dict(R0=R0, Z0=Z0, dist=float(np.linalg.norm(xe - N["Xe"][t])), est=2 * float(N["ee"][t]),
                         g_ref=float(gref), ok=bool(gref < 0)))
    out["lost"] = rows
    out["lost_all_negative"] = bool(all(r_.get("ok") for r_ in rows))
    # the enclosure model on 5 x 5 stencils
    rows = []
    base = [(pc, sb) for pc, sb in C.base_done.items() if sb is not False]
    per = list(C.period_pieces)
    picks = [("base", base[i]) for i in rng.permutation(len(base))[:n_enc // 2]] + \
            [("period", (per[i], None)) for i in rng.permutation(len(per))[:n_enc - n_enc // 2]]
    uu, vv = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    u9 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float)
    v9 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
    for kind, (pc, sb) in picks:
        if deadline is not None and time.time() > deadline:
            break
        cR, cZ, eta = C.piece_geom(*pc)
        X0 = np.stack([cR + uu * eta, 0 * uu, cZ + vv * eta], -1)
        if kind == "period":
            o = C.tr.run(X0, C.win * C.period_arc)
            if not np.all(np.isfinite(o["sp"])):
                rows.append(dict(kind=kind, piece=list(pc), ratio=None, note="an extra line did not cross"))
                continue
            P = np.stack([np.hypot(o["Xp"][:, 0], o["Xp"][:, 1]), o["Xp"][:, 2]], -1)
            idx = np.array([C.cache[k_] for k_ in zip(*C.stencil_keys(*pc))])
            c, Au, Av, rn, rho = C.period_enclosure(idx, N)
        else:
            Xc, ec, okl, _ = C.tr.arc(X0, np.full((len(X0), 1), sb))
            if not np.all(okl):
                rows.append(dict(kind=kind, piece=list(pc), ratio=None, note="an extra line failed"))
                continue
            P = Xc[:, 0, :]
            c, Au, Av, rn = _affine_enclosure(P[[0, 10, 20, 2, 12, 22, 4, 14, 24]], u9, v9)   # the 9-point subset
            rho = 2.0 * LEB33 * rn + 2.0 * float(np.max(ec)) + 1e-12
        res = P - c[None] - uu[:, None] * Au[None] - vv[:, None] * Av[None]
        worst = float(np.max(np.linalg.norm(res, axis=1)))
        rows.append(dict(kind=kind, piece=list(pc), rho=float(rho), worst=worst, ratio=worst / max(rho, 1e-300)))
    out["enclosure"] = rows
    out["enclosure_ratio_max"] = max((r_["ratio"] for r_ in rows if r_.get("ratio") is not None), default=None)
    return out


# ============================================================================ the analysis hook
EB_CAP = 600.0          # seconds an analysis gives the survivor bound at most (TORUX_EDGEBOUND_CAP)


def axis_point_of(axis):
    """(R, Z) at phi = 0 of an axis (torux.axis.Axis with .at, or a pair)."""
    if axis is None:
        return None
    if hasattr(axis, "at"):
        R_, Z_ = axis.at(np.array([0.0]))
        return np.array([float(R_[0]), float(Z_[0])])
    return np.asarray(axis, float)[:2]


def edge_bound_survivor(eng, source, T_last, request=None, deadline=None, axis=None, note="", verbose=False,
                        cap_s=None, **kw):
    """The survivor-flux bound as torux.analysis.edge_bound's first stage: the axis point from `axis` (the analysis'
    elliptic axis) or, without one, torux.barrier.find_axis_point inside T_last; budget min(cap, what is left - 30 s).
    Returns a JSON-able record (kind 'edge_survivor', ok, F_up, holds_request, rigorous=True)."""
    import os
    cap = float(os.environ.get("TORUX_EDGEBOUND_CAP", EB_CAP)) if cap_s is None else float(cap_s)
    left = cap if deadline is None else min(cap, deadline - time.time() - 30.0)
    if left < 30.0:
        return dict(ok=False, kind="edge_survivor", reason="no time left", note=note, rigorous=True)
    a = axis_point_of(axis)
    if a is None:
        from .barrier import _setup_section, find_axis_point
        sec, R2 = _setup_section(eng, source, T_last)
        orb = find_axis_point(sec, R2[1] + 0.1 * (R2[0] - R2[1]), R2[0] - 0.1 * (R2[0] - R2[1]))
        if orb is None:
            return dict(ok=False, kind="edge_survivor", reason="axis point not found inside T_last", note=note,
                        rigorous=True)
        a = np.asarray(orb.pts[0], float)
    import os as _o
    method = _o.environ.get("TORUX_EDGEBOUND_METHOD", "curve")
    if method == "cells":
        rec = edge_upper_bound(source, a, T_last=T_last, request=request, budget_s=left, verbose=verbose, **kw)
    else:
        rec = lost_curve_bound(source, a, T_last=T_last, request=request, budget_s=left, verbose=verbose, **kw)
    for k_ in ("_C", "_K", "_cert", "_grid"):
        rec.pop(k_, None)
    rec["note"] = note
    rec["rigorous"] = True
    rec["method"] = rec.get("kind")
    rec["kind"] = "edge_survivor"
    if request is not None:
        rec["interval"] = (rec.get("F_up"), np.inf) if rec.get("ok") else None
    return rec


# ============================================================================ the lost CURVE (1D) certificate
LOST_CURVE_DOC = """THEOREM (lost curve).  C a simple closed polygon in Sigma with
winding number 1 around a, every point of which is lost.  Then every covered torus has flux <= int_{int C} (s B_phi)^+
(D-bar has no lost point, D is connected and contains a, so D lies in int C; s B_phi > 0 on D).  For a
stellarator-symmetric source C = a path in Z >= 0 from the symmetry line inboard of a to the symmetry line outboard of
it, with its mirror image (the mirror of a lost point is lost).  F_up = -s x the loop integral of A along C (Stokes;
Gauss-Legendre 6 points per edge, the 3-point difference added) + a bound of int_{int C} (s B_phi)^- (a raster at
h / 2: per cell max(0, 2 |grad B_phi| x half diagonal - s B_phi(centre)) x area; within 2 sqrt 2 h / 2 of a coil
crossing inside C instead 1.5 mu0 |I| rho + 2 max|B| pi rho^2, the filament's own field bounded analytically).
SEGMENT CERTIFICATES.  An edge of the path is split into 1D pieces [p - eta e, p + eta e] (adaptively, halving up to 9
times); a piece is lost when, at one of 4 arc lengths between its midpoint line's first f <= -0.02 and its lost point
(the minimum of f), the curve of images lies where g < 0: the images of the piece's ends and midpoint (u = -1, 0, 1)
give the quadratic interpolant I(u), its error sampled at u = +-1/2 (2 extra lines) gives rho = 3 x that + tau (+ the
field-model term) -- used only when that error is at most half the quadratic content --; I is covered by 8 sub-arcs
with the exact second-derivative remainder of the polynomial; g linearised at every sub-centre with the Hessian bound
G2 (2 x the largest difference quotient of grad g between sub-centres); the lines keep >= 2 x the image's extent from
the coils and >= d_fail from every coil node.  Forward lines, or -- the mirror piece's forward lines -- backward ones
(the detected direction first).  The midpoint lines are followed up to 4 field periods.
ROUTE.  A detection map (the loose tracer: tolerance 1e-5, rules eps 1e-6, threshold f <= -0.01, 4 field periods;
centre lines of a node grid of spacing r_eff / 16 over the box r_max + 1.8 d_last about a, forward and backward)
marks lost-looking nodes; Dijkstra over them (8-neighbours, interior nodes off the symmetry row) with edge cost =
length x (1 + the spread of the two lines' lost points over their start distance) x (1 + escape arc length / period
arc) x (1 + w x the distance beyond the scale polygon); an edge that fails is blocked and the path re-routed (certified
edges are kept and cost a tenth); after the first certified curve, w is raised (x4 per step) and the cheapest
certified tighter curve kept.  The detection only chooses where to certify; no part of it enters the bound."""


def _seg_base_test(tr, sgn, P3, P2, s_list, eta, tau_field_fn, m_sub=8):
    """1D piece: P3 (3, 3) start points (u = -1, 0, 1), P2 (2, 3) extras (u = -1/2, 1/2); every line traced (arc mode)
    to the arc lengths s_list; returns the arc length at which g < 0 holds on the curved enclosure, or None."""
    X0 = np.concatenate([P3, P2])
    S = np.tile(np.asarray(s_list, float), (5, 1))
    Xc, ec, okl, dm = tr.arc(X0, S)
    if not np.all(okl):
        return None
    for m in range(len(s_list) - 1, -1, -1):
        Y = Xc[:, m, :]
        Ym, Y0, Yp, Yh1, Yh2 = Y[0], Y[1], Y[2], Y[3], Y[4]
        # quadratic interpolant through u = -1, 0, 1
        I = lambda u: Ym * 0.5 * u * (u - 1) + Y0 * (1 - u * u) + Yp * 0.5 * u * (u + 1)
        e2 = max(np.linalg.norm(I(-0.5) - Yh1), np.linalg.norm(I(0.5) - Yh2))
        d2 = np.linalg.norm(Ym - 2 * Y0 + Yp)              # |I''| (constant)
        lin = 0.5 * np.linalg.norm(Yp - Ym)
        if not (e2 <= Q_DECAY * 0.25 * d2 + 1e-15 or e2 <= 1e-3 * lin):
            continue                                        # the cubic content is not small against the quadratic
        tau = 2.0 * float(np.max(ec[:, m])) + 1e-12 + tau_field_fn(float(s_list[m]), lin, eta)
        rho = Q_FAC * e2 + tau
        dl = 1.0 / m_sub
        cen = -1.0 + dl * (2 * np.arange(m_sub) + 1)
        Pc = np.stack([I(u) for u in cen])
        Pu = np.stack([(Yp - Ym) * 0.5 + (Ym - 2 * Y0 + Yp) * u for u in cen]) * dl     # I'(u) dl
        r_sub = 0.5 * dl * dl * d2
        B, D, _, dlb = tr.BGA(Pc)
        g = sgn * (Pc[:, 0] * B[:, 1] - Pc[:, 1] * B[:, 0])
        dg = sgn * (np.stack([B[:, 1], -B[:, 0], 0 * B[:, 0]], -1) + Pc[:, 0:1] * D[:, 1, :] - Pc[:, 1:2] * D[:, 0, :])
        q1 = np.linalg.norm(dg[1:] - dg[:-1], axis=1) / np.maximum(np.linalg.norm(Pc[1:] - Pc[:-1], axis=1), 1e-300)
        G2 = 2.0 * float(np.max(q1)) if len(q1) else 0.0
        rq = np.linalg.norm(Pu, axis=1) + rho + r_sub
        bound = g + np.abs(np.sum(dg * Pu, 1)) + np.linalg.norm(dg, axis=1) * (rho + r_sub) + 0.5 * G2 * rq * rq
        ext = float(np.max(np.linalg.norm(Y - Y0[None], axis=1))) + rho
        if np.all(bound < -1e-9 * (np.abs(g) + 1.0)) and np.all(rq < 0.5 * dlb) and float(np.min(dm)) >= COIL_GUARD * ext:
            return float(s_list[m])
    return None


def supported(source):
    """None when every current-carrying part is a Fourier filament (what the tracer sums), else the reason."""
    bad = sorted({p.kind for p in source.parts if p.kind != "fourier"
                  and not (p.kind == "toroidal" and float(getattr(p, "RB", 0.0)) == 0.0)})
    if bad or not any(p.kind == "fourier" for p in source.parts):
        return f"unsupported source parts {bad or ['none']} (Fourier filaments only)"
    return None


def lost_curve_bound(source, axis_point, T_last=None, core=None, scale_poly=None, request=None, budget_s=300.0,
                     h_rel=1 / 16, det_win=4.0, d_fail_rel=0.03, eps=1e-10, tol=1e-10, reach_d=1.8, max_split=8,
                     max_routes=40, w_out=2.0, verbose=False, sign=None, with_evidence=True, det_share=0.4,
                     beta_det=0.01, beta=0.02, seg_win=4.0):
    """The lost-curve bound (LOST_CURVE_DOC) around axis_point; returns a JSON-able record (ok, F_up, curve, ...)."""
    import heapq
    from .verify import torus_points
    from .barrier import signed_area, inside_polygon
    from scipy.spatial import cKDTree
    t0 = time.time()
    deadline = t0 + budget_s
    a = np.asarray(axis_point, float)
    out = dict(kind="lost_curve_v2", ok=False, axis_point=a.tolist(), request=request, rigorous=True)
    why = supported(source)
    if why is not None:
        out["reason"] = why
        return out
    cp = []
    for p in source.parts:
        if p.kind == "fourier":
            x, _ = p.points(np.arange(1024) / 1024)
            cp.append(x.reshape(-1, 3))
    cp = np.concatenate(cp)
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
    sp_ = core if scale_poly is None else scale_poly
    sp_ = np.asarray(sp_, float)
    r_max = float(np.max(np.hypot(sp_[:, 0] - a[0], sp_[:, 1] - a[1])))
    r_eff = math.sqrt(abs(signed_area(sp_)) / math.pi)
    if T_last is None:
        d_last = float(np.min(tree.query(np.stack([sp_[:, 0], 0 * sp_[:, 0], sp_[:, 1]], -1))[0]))
    d_fail = d_fail_rel * d_last
    tr = ExactTracer(source, source.nfp, 1.0, d_fail=d_fail, tol=tol, eps=eps, beta=beta)
    if sign is None:
        sign = float(np.sign(tr.B(np.array([[a[0], 0.0, a[1]]]))[0, 1]))
    tr.s = sign
    det = ExactTracer(source, source.nfp, sign, d_fail=d_fail, tol=1e-5, eps=1e-6, beta=beta_det)
    tr.check()
    period_arc = 2 * math.pi * a[0] / source.nfp
    sym = bool(getattr(source, "stellsym", True))
    out.update(d_last=d_last, d_fail=d_fail, check_err_multilevel=tr.check_err, sign=sign)
    if not sym:
        out["reason"] = "the lost-curve route needs stellarator symmetry (the mirror half)"
        return out
    # node grid (upper half, row 0 on the symmetry line)
    h = h_rel * r_eff
    ext = r_max + reach_d * d_last
    i0 = int(math.ceil(ext / h))
    Rn = a[0] + h * np.arange(-i0, i0 + 1)
    Zn = h * np.arange(0, i0 + 1)
    RR, ZZ = np.meshgrid(Rn, Zn, indexing="ij")
    nR_, nZ_ = RR.shape
    cand = ~(_poly_in(core, np.stack([RR.ravel(), ZZ.ravel()], -1)).reshape(RR.shape)) if core is not None else \
        np.ones(RR.shape, bool)
    cand &= RR > 0.02
    idx = np.flatnonzero(cand.ravel())
    X = np.stack([RR.ravel()[idx], 0 * idx, ZZ.ravel()[idx]], -1)
    t1 = time.time()
    # forward lines of the upper-half nodes and of their mirror images (= the backward lines), outermost first, in
    # chunks between deadline checks (nodes not reached count as not lost)
    order = np.argsort(-np.hypot(X[:, 0] - a[0], X[:, 2] - a[1]))
    of = {k_: np.full(len(X), np.nan) for k_ in ("se", "sf")}; of["status"] = np.full(len(X), ST_SMAX)
    of["Xe"] = np.full((len(X), 3), np.nan)
    ob = {k_: v.copy() for k_, v in of.items()}
    for lo in range(0, len(order), 2048):
        if time.time() > t0 + det_share * budget_s:
            break
        sel = order[lo:lo + 2048]
        for o_, XX in ((of, X[sel]), (ob, X[sel] * np.array([1.0, 1.0, -1.0]))):
            r_ = det.run(XX, det_win * period_arc, ext=1.0)
            for k_ in ("se", "sf", "status", "Xe"):
                o_[k_][sel] = r_[k_]
    lf = of["status"] == ST_LOST; lb = ob["status"] == ST_LOST
    lost = np.zeros(RR.size, bool); lost[idx] = lf | lb
    dirn = np.zeros(RR.size, int); dirn[idx] = np.where(lf & (~lb | (of["se"] <= ob["se"])), 1, np.where(lb, -1, 0))
    sesc = np.full(RR.size, np.inf); sesc[idx] = np.where(dirn[idx] == 1, of["se"], np.where(dirn[idx] == -1, ob["se"], np.inf))
    XE = np.full((RR.size, 3), np.nan)
    XE[idx] = np.where((dirn[idx] == 1)[:, None], of["Xe"], ob["Xe"])
    lost = lost.reshape(RR.shape); dirn = dirn.reshape(RR.shape); sesc = sesc.reshape(RR.shape)
    XE = XE.reshape(nR_, nZ_, 3)
    out["detection"] = dict(nodes=int(len(idx)), lost=int(lost.sum()), t=time.time() - t1, h=h)
    # distance beyond the scale polygon (ray ratio)
    thr = 2 * np.pi * np.arange(720) / 720
    rT = ray_crossing_max(sp_, a, thr)
    rT = np.where(np.isfinite(rT), rT, np.nanmax(rT))
    thc = np.mod(np.arctan2(ZZ - a[1], RR - a[0]), 2 * np.pi)
    rho = np.hypot(RR - a[0], ZZ - a[1]) / np.interp(thc, np.concatenate([thr, [2 * np.pi]]), np.concatenate([rT, rT[:1]]))
    blocked = set()
    blocked_nodes = set()
    ia0 = i0

    def route(w_o, disc=0.1):
        """Dijkstra from the symmetry-line nodes inboard of a to those outboard (interior nodes Z > 0); certified edges
        cost disc x their cost."""
        INF = np.inf
        dist = np.full(RR.shape, INF)
        prev = {}
        pq = []
        for i in range(0, ia0):
            if lost[i, 0]:
                dist[i, 0] = 0.0
                heapq.heappush(pq, (0.0, i, 0))
        while pq:
            dcur, i, j = heapq.heappop(pq)
            if dcur > dist[i, j]:
                continue
            if j == 0 and i > ia0:
                path = [(i, j)]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return path[::-1]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a_, b_ = i + di, j + dj
                    if not (0 <= a_ < nR_ and 0 <= b_ < nZ_) or not lost[a_, b_] or (a_, b_) in blocked_nodes:
                        continue
                    if b_ == 0 and not (a_ > ia0 or (i, j) == (a_, b_)):
                        continue                    # the symmetry row only at the two ends
                    if j == 0 and b_ == 0:
                        continue
                    if ((i, j), (a_, b_)) in blocked:
                        continue
                    L = h * math.hypot(di, dj)
                    if dirn[i, j] == dirn[a_, b_] and dirn[i, j] != 0:
                        spread = float(np.linalg.norm(XE[i, j] - XE[a_, b_])) / L
                    else:
                        spread = 50.0
                    c = L * (1.0 + min(spread, 50.0)) * (1.0 + 0.5 * (sesc[i, j] + sesc[a_, b_]) / period_arc) \
                        * (1.0 + w_o * max(0.0, 0.5 * (rho[i, j] + rho[a_, b_]) - 1.0))
                    if ((i, j), (a_, b_)) in good_edges:
                        c *= disc
                    nd = dcur + c
                    if nd < dist[a_, b_]:
                        dist[a_, b_] = nd
                        prev[(a_, b_)] = (i, j)
                        heapq.heappush(pq, (nd, a_, b_))
        return None

    certified = {}           # edge -> list of certified pieces (for the record)
    n_lines0 = tr.n_lines
    good_edges = set()
    stats = dict(routes=0)

    def certify_route(w_o, t_end, disc=0.1):
        """Routes (re-routed around failed edges) until one is fully certified; returns its path or None."""
        while stats["routes"] < max_routes and time.time() < t_end:
            stats["routes"] += 1
            path = route(w_o, disc)
            if path is None:
                return None
            failed = None
            for (e0, e1) in zip(path[:-1], path[1:]):
                if (e0, e1) in good_edges:
                    continue
                if time.time() > t_end:
                    return None
                p0 = np.array([RR[e0], 0.0, ZZ[e0]]); p1 = np.array([RR[e1], 0.0, ZZ[e1]])
                dsgn = dirn[e0] if dirn[e0] != 0 else dirn[e1]
                ok_edge = _certify_segment(tr, sign, p0, p1, int(dsgn), det, period_arc, max_split, t_end, seg_win)
                if ok_edge is None:
                    blocked.add((e0, e1)); blocked.add((e1, e0))
                    # and the slower-escaping end node (its other edges would most likely fail too)
                    bad_node = e0 if sesc[e0] > sesc[e1] else e1
                    if bad_node[1] > 0:
                        blocked_nodes.add(bad_node)
                    failed = (e0, e1)
                    break
                good_edges.add((e0, e1)); good_edges.add((e1, e0))
                certified[(e0, e1)] = ok_edge
            if failed is None:
                return path
            if verbose:
                print(f"   route {stats['routes']} (w {w_o:g}): edge {failed} failed, re-routing "
                      f"({len(good_edges) // 2} edges certified)", flush=True)
        return None

    def curve_flux(path):
        upper = np.array([[RR[q], ZZ[q]] for q in path])
        C_ = np.concatenate([upper, upper[-2:0:-1] * np.array([1.0, -1.0])])
        if signed_area(C_) < 0:
            C_ = C_[::-1]
        E_ = np.concatenate([C_, np.roll(C_, -1, axis=0)], axis=1)
        xg, wg = np.polynomial.legendre.leggauss(6)
        tq = 0.5 * (xg + 1.0); wt = 0.5 * wg
        Pq = E_[:, None, :2] + tq[None, :, None] * (E_[:, None, 2:] - E_[:, None, :2])
        Xq = np.stack([Pq[..., 0], 0 * Pq[..., 0], Pq[..., 1]], -1).reshape(-1, 3)
        A = tr.BGA(Xq)[2].reshape(len(E_), 6, 3)
        dd = E_[:, 2:] - E_[:, :2]
        return -sign * float(np.sum(wt[None, :] * (A[..., 0] * dd[:, None, 0] + A[..., 2] * dd[:, None, 1])))

    # the first route: robust (a third of what is left after the detection for each later tightening step)
    path = certify_route(w_out, deadline)
    best = None
    tight_log = []
    if path is not None:
        best = (curve_flux(path), path, w_out)
        tight_log.append(dict(w=w_out, F=best[0], t=time.time() - t0))
        # tightening: a stronger preference for nodes near the scale polygon, certified edges reused
        w_t = w_out
        while time.time() < deadline - 0.05 * budget_s and w_t < 1e4:
            w_t *= 4.0
            p2 = certify_route(w_t, deadline, disc=0.7)      # certified edges barely discounted: inner routes tried
            if p2 is None:
                break
            F2 = curve_flux(p2)
            tight_log.append(dict(w=w_t, F=F2, t=time.time() - t0))
            if F2 < best[0]:
                best = (F2, p2, w_t)
        path = best[1]
    out["routes"] = stats["routes"]
    out["tightening"] = tight_log
    out["lines_traced"] = tr.n_lines - n_lines0
    out["timings"] = dict(total=time.time() - t0, tracer=tr.t, detection=det.t)
    if path is None:
        out.setdefault("reason", "no certified route within the budget")
        return out
    upper = np.array([[RR[q], ZZ[q]] for q in path])
    curve = np.concatenate([upper, upper[-2:0:-1] * np.array([1.0, -1.0])])     # closed: path + mirror (no repeats)
    # orientation counter-clockwise in (R, Z)
    if signed_area(curve) < 0:
        curve = curve[::-1]
    # the theorem's polygon: simple, winding once around a (checked exactly on the vertices; the flux formula needs it)
    from .barrier import closed_polygon_simple, winding_number
    if not closed_polygon_simple(curve, 1e-12 * h) or abs(winding_number(curve, a)) != 1:
        out["reason"] = "the certified route is not a simple closed curve around the axis"
        return out
    # flux inside: -s x the loop integral of A (Gauss-Legendre 6 per edge; 3-point estimate of the error)
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
    # (s B_phi)^- inside: a raster at spacing h / 2, the cells whose value minus the gradient's reach is negative
    hh = 0.5 * h
    lo_ = curve.min(0); hi_ = curve.max(0)
    gr = np.stack(np.meshgrid(np.arange(lo_[0] + hh / 2, hi_[0], hh), np.arange(lo_[1] + hh / 2, hi_[1], hh),
                              indexing="ij"), -1).reshape(-1, 2)
    gr = gr[inside_polygon(curve, gr)]
    B, D, _, _ = tr.BGA(np.stack([gr[:, 0], 0 * gr[:, 0], gr[:, 1]], -1))
    bp = sign * B[:, 1]
    reach = 2.0 * np.linalg.norm(D[:, 1, :], axis=1) * hh / math.sqrt(2.0)
    # coil crossings inside the curve: the cells within rho_c of one are bounded analytically instead (the filament's
    # own field: |B| <= 1.5 mu0 |I| / (2 pi d) next to it, so int_{d < rho_c} |B| <= 1.5 mu0 |I| rho_c, plus the
    # rest of the field: 2 x the largest |B| on the circle d = rho_c times pi rho_c^2)
    from .barrier import coil_crossings
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
    out.update(ok=True, F_up=float(F_up), flux=dict(F=F, quad_err=abs(I6 - I3), neg_corr=corr, n_neg=int(neg.sum()),
                                                    coils_inside=int(len(cc)), coil_disk_corr=disk_corr),
               curve=curve.tolist(), n_vertices=len(curve),
               pieces=int(sum(len(v) for v in certified.values())))
    if sec_last is not None:
        # NEGATIVE TEST: the verified torus's section must lie inside the curve of lost points
        insd = inside_polygon(curve, sec_last)
        out["verified_section_outside"] = int(np.sum(~insd))
        out["ratio"] = F_up / abs(float(T_last.flux))
        if np.any(~insd) or F_up < abs(float(T_last.flux)):
            out.update(ok=False, false_certificate=True,
                       reason="FALSE CERTIFICATE: the verified torus is not inside the curve, or F_up below its flux")
    if request is not None and out.get("ok"):
        out["holds_request"] = bool(F_up < abs(request))
    if with_evidence and out.get("ok"):
        try:
            nxy = {q: (float(RR[q]), float(ZZ[q])) for e in certified for q in e}
            out["evidence"] = curve_evidence(source, tr, certified, sign, nxy, deadline=time.time() + 60)
        except Exception as e:  # noqa: BLE001
            out["evidence"] = dict(error=repr(e))
    out["timings"]["total"] = time.time() - t0
    return out


def _certify_segment(tr, sgn, p0, p1, dsgn, det, period_arc, max_split, deadline, seg_win=4.0):
    """Every point of the segment p0 -> p1 (in Sigma, Z >= 0) lost?  Pieces halved until each passes the 1D base test
    (forward lines for direction +1, the mirror piece's forward lines -- the backward ones -- for -1; the detected
    direction first, the other when it fails).  Returns the list of certified pieces (t0, t1, direction, arc length)
    or None."""
    todo = [(0.0, 1.0, 0)]
    done = []
    mir = np.array([1.0, 1.0, -1.0])
    ef = max(tr.check_err or 0.0, tr.eps)
    tf = lambda S, lin, e_: 2.0 * ef * S * max(1.0, lin / max(e_, 1e-300))
    first = 1 if dsgn >= 0 else -1
    while todo:
        if time.time() > deadline:
            return None
        batch, todo = todo[:64], todo[64:]
        mids = np.array([p0 + 0.5 * (t0 + t1) * (p1 - p0) for (t0, t1, _) in batch])
        res = [None] * len(batch)
        lost_any = np.zeros(len(batch), bool)
        for d_ in (first, -first):
            left = [q for q in range(len(batch)) if res[q] is None]
            if not left:
                break
            o = tr.run(mids[left] * (mir if d_ == -1 else 1.0), seg_win * period_arc, ext=1.0)
            for m_, q in enumerate(left):
                if o["status"][m_] != ST_LOST:
                    continue
                lost_any[q] = True
                t0, t1, dep = batch[q]
                pa_ = p0 + t0 * (p1 - p0); pb_ = p0 + t1 * (p1 - p0); pm = mids[q]
                eta = 0.5 * float(np.linalg.norm(pb_ - pa_))
                s1, s2 = float(o["sf"][m_]), float(o["se"][m_])
                if not np.isfinite(s1):
                    s1 = s2
                ss = np.linspace(s1, s2, 4) if s2 > s1 else np.full(4, s2)
                P3 = np.stack([pa_, pm, pb_]); P2 = np.stack([0.75 * pa_ + 0.25 * pb_, 0.25 * pa_ + 0.75 * pb_])
                if d_ == -1:
                    P3 = P3 * mir; P2 = P2 * mir
                sp = _seg_base_test(tr, sgn, P3, P2, ss, eta, tf)
                if sp is not None:
                    res[q] = (t0, t1, d_, sp)
        for q, (t0, t1, dep) in enumerate(batch):
            if res[q] is not None:
                done.append(res[q])
            elif not lost_any[q]:
                return None                 # a point of the segment whose line is not lost in the window: give up
            elif dep < max_split:
                tm = 0.5 * (t0 + t1)
                todo += [(t0, tm, dep + 1), (tm, t1, dep + 1)]
            else:
                return None
    return done


def curve_evidence(source, tr, certified, sign, nodes_xy, n=6, seed=0, deadline=None):
    """Independent checks of a lost curve on sampled certified pieces: the midpoint line re-traced with scipy DOP853 on
    the reference field (torux.reference.fast64) to the certified arc length -- g there by the reference field must be
    < 0, its distance to our point is recorded -- and 9 extra points of the piece traced to that arc length (all must
    have g < 0: the enclosure model checked)."""
    from .reference.fast64 import Field
    rng = np.random.default_rng(seed)
    F = Field(source)
    rows = []
    items = [(e, pc) for e, pcs in certified.items() for pc in pcs]
    mir = np.array([1.0, 1.0, -1.0])
    for k in rng.permutation(len(items))[:n]:
        if deadline is not None and time.time() > deadline:
            break
        (e0, e1), (t0, t1, d_, sp) = items[k]
        p0 = np.array([nodes_xy[e0][0], 0.0, nodes_xy[e0][1]]); p1 = np.array([nodes_xy[e1][0], 0.0, nodes_xy[e1][1]])
        tt = np.linspace(t0, t1, 9)
        P = p0[None] + tt[:, None] * (p1 - p0)[None]
        if d_ == -1:
            P = P * mir
        Xc, ec, okl, _ = tr.arc(P, np.full((9, 1), sp))
        B = tr.BGA(Xc[:, 0, :])[0]
        g = sign * (Xc[:, 0, 0] * B[:, 1] - Xc[:, 0, 1] * B[:, 0])
        xe = _ref_arclength(F.B, P[4], sign, sp)
        row = dict(edge=[list(e0), list(e1)], t=[float(t0), float(t1)], dir=int(d_), s=float(sp),
                   extra_all_negative=bool(np.all(okl) and np.all(g < 0)), g_extra_max=float(np.max(g)))
        if xe is not None:
            b = F.B(xe[None])[0]
            row.update(g_ref=float(sign * (xe[0] * b[1] - xe[1] * b[0])), dist_ref=float(np.linalg.norm(xe - Xc[4, 0])))
        rows.append(row)
    return dict(pieces=rows, all_negative=bool(all(r_["extra_all_negative"] and r_.get("g_ref", -1) < 0 for r_ in rows)))
