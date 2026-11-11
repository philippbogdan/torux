"""Near-coil step control for the exit march's tracers (torux.exitcurve, option coil_steps): torux.edgebound's compiled
arc-length GBS with one change -- a step whose ball reaches within d_fail of a coil is SHORTENED (to half the line's
own clearance beyond d_fail) instead of ending the line.

Why.  edgebound's kernels take the coils' rule levels for the ball of radius |H| about the step's start and end a line
as soon as that ball's distance lower bound is below d_fail ('a coil').  The first step is H0 = 0.02 (m) and later
steps are capped at H_COIL x the line's distance bound, so a line that STARTS within about H0 + d_fail of a coil (2 to
3 cm: CoilStellaration's coils come within a few cm of the plasma, and some cross the symmetry plane inside the march's
curves) ends at its first step without being traced at all: no exit certificate, no lost certificate, and in the march
no progress for its edge (and the detection map marks it neither exit-looking nor lost).  With this change such lines
are traced; a line ends when ITS OWN distance lower bound (margin 0) is below NC_END x d_fail (1.5: the shortened
steps stay >= d_fail / 4, so no line creeps along a coil just outside d_fail -- version 1 ended at d_fail and one
range's run went past its time limit that way), and after NC_MAX_IT step attempts (a cost guard).
No new theory and no certificate changes:
  STRICT / VALIDATED exactly as in torux.exitcurve.THEOREM_DOC and torux.edgebound: the rule levels are chosen for the
  ball of the step actually taken (the calibrated rules, accurate to eps at distances >= d_fail), a step is accepted
  only when its GBS estimate is below the tolerance, the estimates are summed as before (the trace-error model, checked
  against scipy DOP853 on the reference field), and the coil-distance records (DC, dmin) are the accepted steps' ball
  bounds (a valid lower bound of the line's distance on its path; the shortened-away balls were not used).
  Lines the original kernels trace are traced bit for bit identically; a line
  ended early only gives fewer certificates.

    from torux.exitcurve_nearcoil import NearCoilTracer
    tr = NearCoilTracer(source, nfp, sign, d_fail=..., tol=..., eps=..., beta=...)   # an ExactTracer
    torux.exitcurve.multicross(tr, ...) uses tr.mc_kernel; tr.arc(...) the arc kernel below.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from .edgebound import ExactTracer, _gbs, _cross, _fval, _step_ctl, H_COIL


@njit(parallel=True, cache=True)
def multicross_kernel_nc(X0, s, ncross, S_max, beta, d_fail, tol, H0, H_max, k, P, dphi, d_end, max_it):
    """torux.exitcurve.multicross_kernel with near-coil step shortening (same outputs): a step whose ball bound is below
    d_fail is shortened to half the line's own clearance beyond d_fail while that own bound is >= d_end (>= d_fail),
    else the line ends at the coil (status 4); at most max_it step attempts per line (then status 5)."""
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
        dend2 = d_end * d_end
        if q < dend2:
            stat = 4
            it = max_it
        while it < max_it:
            it += 1
            Hs = min(H, S_max - sacc) if not lost_on else H
            if Hs <= 0:
                stat = 2
                break
            ok, err, q = _gbs(x, Hs, k, s, P, xn)
            if q < dfail2:
                # the step's ball reaches within d_fail: shorten it unless the line itself is within d_end
                f_, q0 = _fval(x, s, P)
                if q0 < dend2:
                    stat = 4
                    break
                H = min(0.5 * Hs, 0.5 * (math.sqrt(q0) - d_fail))
                if H < 1e-11:
                    stat = 5
                    break
                continue
            if q < qmin:
                qmin = q
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


@njit(parallel=True, cache=True)
def arc_kernel_nc(X0, s, S_tgt, d_fail, tol, H0, H_max, k, P, d_end, max_it):
    """torux.edgebound.arc_kernel with near-coil step shortening (same outputs; d_end, max_it as in
    multicross_kernel_nc)."""
    L = X0.shape[0]
    m = S_tgt.shape[1]
    Xc = np.full((L, m, 3), np.nan); ec = np.full((L, m), np.nan)
    okl = np.zeros(L, np.bool_); dmin = np.full(L, np.inf)
    dfail2 = d_fail * d_fail
    dend2 = d_end * d_end
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
        f_, q0 = _fval(x, 1.0, P)
        if q0 < dend2:
            good = False
            it = max_it
            qmin = q0
        while jt < m and it < max_it:
            it += 1
            if sacc >= S_tgt[i, jt] - 1e-15:
                for c in range(3):
                    Xc[i, jt, c] = x[c]
                ec[i, jt] = eacc
                jt += 1
                continue
            Hs = min(H, S_tgt[i, jt] - sacc)
            ok, err, q = _gbs(x, Hs, k, s, P, xn)
            if q < dfail2:
                f_, q0 = _fval(x, s, P)
                if q0 < dend2:
                    if q0 < qmin:
                        qmin = q0
                    good = False
                    break
                H = min(0.5 * Hs, 0.5 * (math.sqrt(q0) - d_fail))
                if H < 1e-11:
                    good = False
                    break
                continue
            if q < qmin:
                qmin = q
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


NC_END = 1.5            # a line ends where its own coil-distance bound is below NC_END x d_fail (the shortened steps
                        # stay >= (NC_END - 1) / 2 x d_fail: no creeping along a coil just outside d_fail)
NC_MAX_IT = 200000      # step attempts per line at most (a cost guard; a line past it fails: no certificate from it)


class NearCoilTracer(ExactTracer):
    """ExactTracer whose arc() and (through torux.exitcurve.multicross) multi-crossing traces shorten steps near coils."""

    def mc_kernel(self, X0, s, ncross, S_max, beta, d_fail, tol, H0, H_max, k, P, dphi):
        return multicross_kernel_nc(X0, s, ncross, S_max, beta, d_fail, tol, H0, H_max, k, P, dphi,
                                    NC_END * d_fail, NC_MAX_IT)

    def arc(self, X0, S_tgt):
        X0 = np.ascontiguousarray(np.asarray(X0, float).reshape(-1, 3))
        S = np.ascontiguousarray(np.asarray(S_tgt, float).reshape(len(X0), -1))
        t0 = time.time()
        out = arc_kernel_nc(X0, self.s, S, self.d_fail, self.tol, 0.02, 0.1, self.k, self.P, NC_END * self.d_fail,
                            NC_MAX_IT)
        self.t += time.time() - t0
        self.t_arc += time.time() - t0
        self.n_arc += len(X0)
        self.n_lines += len(X0)
        return out
