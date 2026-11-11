"""Independent field-line tracer on the fast64 field.

Field lines with the toroidal angle as the independent variable:
    dR/dphi = R B_R / B_phi,   dZ/dphi = R B_Z / B_phi.
Integrator: s-stage Gauss-Legendre implicit Runge-Kutta (collocation at the Gauss points,
order 2s, symmetric and symplectic for canonical systems; default s = 6, order 12).  The
Butcher coefficients are computed from the Legendre nodes in mpmath (no tabulated constants).
The stage equations are solved by fixed-point iteration started from the extrapolated
collocation polynomial of the previous step, until the update is below tol (default 1e-15,
relative) or stagnates at the rounding floor.

Poincare sections: every requested section angle (and its 2 pi m repetitions) is an exact step
end -- the interval between consecutive sections is divided into equal steps of size <= h.
Error estimate: each line is integrated twice, with h and h/2; the h/2 result is returned with
    err = |y_h - y_{h/2}|   (step doubling: a bound on the h/2 error when truncation dominates,
                            and of the size of the accumulated rounding when that dominates)
    err_richardson = err / (2^{2s} - 1)   (valid only when truncation dominates, i.e. when err
                            falls by 2^{2s} as h halves)
The solution update is Kahan-compensated (the rounding of y_n + delta is carried to the next
step), which removes the round-off random walk of long traces.

    out = poincare(source_or_field, R0, Z0, phis, nturns, phi0=0.0, h=..., stages=6)
    out["R"], out["Z"]: (nlines, nturns, nsec) section points (NaN after a line is lost),
    out["err"], out["err_richardson"]: error estimates, out["lost"]: (nlines,) first lost crossing or -1,
    out["evals"]: field evaluations per line.
"""
from __future__ import annotations

import functools
import math

import mpmath
import numpy as np
from numba import njit, prange

from .fast64 import Field, eval_point_B
from .quadrules import gauss_legendre_mp


@functools.lru_cache(maxsize=None)
def gauss_irk(s):
    """(A (s,s), b (s,), c (s,), E (s, s+1)) of the s-stage Gauss-Legendre collocation method.
    E extrapolates the collocation polynomial through (0, y_n), (c_i, Y_i) to 1 + c_j."""
    xs, ws = gauss_legendre_mp(s)
    with mpmath.workdps(40):
        c = [(1 + x) / 2 for x in xs]
        b = [w / 2 for w in ws]
        A = []
        for i in range(s):
            row = []
            for j in range(s):
                # integral_0^{c_i} l_j(tau) d tau, l_j Lagrange basis on the c nodes
                coeffs = [mpmath.mpf(1)]
                den = mpmath.mpf(1)
                for m in range(s):
                    if m == j:
                        continue
                    coeffs = [0] + coeffs
                    for q in range(len(coeffs) - 1):
                        coeffs[q] -= c[m] * coeffs[q + 1]
                    den *= c[j] - c[m]
                val = sum(coeffs[q] * c[i] ** (q + 1) / (q + 1) for q in range(len(coeffs)))
                row.append(val / den)
            A.append(row)
        nodes = [mpmath.mpf(0)] + c
        E = []
        for j in range(s):
            tgt = 1 + c[j]
            row = []
            for i in range(s + 1):
                v = mpmath.mpf(1)
                for m in range(s + 1):
                    if m != i:
                        v *= (tgt - nodes[m]) / (nodes[i] - nodes[m])
                row.append(v)
            E.append(row)
    f = lambda M: np.array([[float(v) for v in r] for r in M])
    return f(A), np.array([float(v) for v in b]), np.array([float(v) for v in c]), f(E)


@njit(cache=True, error_model="numpy")
def _rhs(P, phi, R, Z):
    c = math.cos(phi)
    s = math.sin(phi)
    Bx, By, Bz = eval_point_B(P, R * c, R * s, Z)
    BR = Bx * c + By * s
    Bp = -Bx * s + By * c
    return R * BR / Bp, R * Bz / Bp


@njit(cache=True, error_model="numpy")
def _integrate(P, R, Z, phi0, targets, h, A, b, c, E, tol, maxit, bounds, outR, outZ, stat):
    """Integrate one line from phi0 through the sorted target angles; outR/outZ[k] at targets[k].
    stat = [evaluations, max iterations, lost index (-1), nonconvergent steps]."""
    s = b.shape[0]
    Y = np.empty((s, 2))
    F = np.empty((s, 2))
    Yn = np.empty((s, 2))
    have_prev = False
    Rprev = R
    Zprev = Z
    eR = 0.0
    eZ = 0.0
    phi = phi0
    hs = h
    nt = targets.shape[0]
    stat[2] = -1.0
    for k in range(nt):
        dphi = targets[k] - phi
        n = int(math.ceil(dphi / h - 1e-12))
        if n < 1:
            n = 1
        hs_new = dphi / n
        if k > 0 and abs(hs_new - hs) > 1e-12 * abs(hs):
            have_prev = False            # step size changed: the extrapolation weights assume equal steps
        hs = hs_new
        for step in range(n):
            # initial guess
            if have_prev:
                for j in range(s):
                    gR = E[j, 0] * Rprev          # collocation polynomial: node 0 is the previous y_n
                    gZ = E[j, 0] * Zprev
                    for i in range(s):
                        gR += E[j, i + 1] * Y[i, 0]
                        gZ += E[j, i + 1] * Y[i, 1]
                    Yn[j, 0] = gR
                    Yn[j, 1] = gZ
                for j in range(s):
                    Y[j, 0] = Yn[j, 0]
                    Y[j, 1] = Yn[j, 1]
            else:
                for j in range(s):
                    Y[j, 0] = R
                    Y[j, 1] = Z
            conv = False
            prev = np.inf
            for it in range(maxit):
                for i in range(s):
                    F[i, 0], F[i, 1] = _rhs(P, phi + c[i] * hs, Y[i, 0], Y[i, 1])
                stat[0] += s
                diff = 0.0
                for i in range(s):
                    nR = R
                    nZ = Z
                    for j in range(s):
                        nR += hs * A[i, j] * F[j, 0]
                        nZ += hs * A[i, j] * F[j, 1]
                    d = max(abs(nR - Y[i, 0]), abs(nZ - Y[i, 1]))
                    if d > diff:
                        diff = d
                    Y[i, 0] = nR
                    Y[i, 1] = nZ
                if it + 1 > stat[1]:
                    stat[1] = it + 1
                if not (diff == diff):
                    break
                # converged: below tol, or at the rounding floor (the update stopped shrinking
                # while already below 1e3 tol)
                sc = 1.0 + abs(R) + abs(Z)
                if diff <= tol * sc or (diff >= 0.5 * prev and diff <= 1e3 * tol * sc):
                    conv = True
                    break
                prev = diff
            if not conv:
                stat[3] += 1
            # update with the derivatives of the last iterate (equals b^T A^-1 (Y - y_n) for the
            # converged stages; differs from F(Y*) by O(h L tol))
            dR = 0.0
            dZ = 0.0
            for i in range(s):
                dR += b[i] * F[i, 0]
                dZ += b[i] * F[i, 1]
            Rprev = R
            Zprev = Z
            # compensated (Kahan) update: the rounding of y + delta is carried to the next step
            dRt = hs * dR + eR
            dZt = hs * dZ + eZ
            Rn = R + dRt
            Zn = Z + dZt
            eR = (R - Rn) + dRt
            eZ = (Z - Zn) + dZt
            R = Rn
            Z = Zn
            phi = (targets[k] - dphi) + (step + 1) * hs
            have_prev = True
            if not (R > bounds[0] and R < bounds[1] and abs(Z) < bounds[2]) or not (R == R and Z == Z):
                stat[2] = k
                for q in range(k, nt):
                    outR[q] = np.nan
                    outZ[q] = np.nan
                return
        phi = targets[k]
        outR[k] = R + eR
        outZ[k] = Z + eZ


@njit(parallel=True, cache=True, error_model="numpy")
def _trace_many(P, R0, Z0, phi0, targets, h, A, b, c, E, tol, maxit, bounds, doubling):
    n = R0.shape[0]
    nt = targets.shape[0]
    R1 = np.full((n, nt), np.nan)
    Z1 = np.full((n, nt), np.nan)
    R2 = np.full((n, nt), np.nan)
    Z2 = np.full((n, nt), np.nan)
    st = np.zeros((n, 2, 4))
    for i in prange(n):
        _integrate(P, R0[i], Z0[i], phi0, targets, h, A, b, c, E, tol, maxit, bounds, R1[i], Z1[i], st[i, 0])
        if doubling:
            _integrate(P, R0[i], Z0[i], phi0, targets, 0.5 * h, A, b, c, E, tol, maxit, bounds, R2[i], Z2[i],
                       st[i, 1])
    return R1, Z1, R2, Z2, st


@njit(parallel=True, cache=True, error_model="numpy")
def _trace_many_pairs(P, R0, Z0, phi0, targets, h, A, b, c, E, tol, maxit, bounds, doubling):
    """_trace_many with the two step sizes of a line as separate parallel tasks: the same
    _integrate calls on the same inputs, so the same results; a few lines (the heteroclinic orbit's one) use twice the
    threads."""
    n = R0.shape[0]
    nt = targets.shape[0]
    m = 2 if doubling else 1
    R1 = np.full((n, nt), np.nan)
    Z1 = np.full((n, nt), np.nan)
    R2 = np.full((n, nt), np.nan)
    Z2 = np.full((n, nt), np.nan)
    st = np.zeros((n, 2, 4))
    for k in prange(n * m):
        i = k // m
        if k % m == 0:
            _integrate(P, R0[i], Z0[i], phi0, targets, h, A, b, c, E, tol, maxit, bounds, R1[i], Z1[i], st[i, 0])
        else:
            _integrate(P, R0[i], Z0[i], phi0, targets, 0.5 * h, A, b, c, E, tol, maxit, bounds, R2[i], Z2[i],
                       st[i, 1])
    return R1, Z1, R2, Z2, st


def poincare(source, R0, Z0, phis, nturns, phi0=0.0, h=None, stages=6, tol=1e-15, maxit=60,
             bounds=(1e-6, 1e6, 1e6), doubling=True):
    """Poincare sections of field lines started at (R0, Z0, phi0).

    phis: section angles (rad); sections are taken at phis + 2 pi m (m = 0..nturns-1) that lie
    after phi0 (a section equal to phi0 is taken after one full turn).  h: maximum step in phi
    (default 2 pi / 128).  Returns dict (see module docstring)."""
    F = source if isinstance(source, Field) else Field(source)
    R0 = np.atleast_1d(np.asarray(R0, dtype=float))
    Z0 = np.atleast_1d(np.asarray(Z0, dtype=float))
    phis = np.atleast_1d(np.asarray(phis, dtype=float))
    rel = np.sort(np.mod(phis - phi0, 2 * np.pi))
    rel = np.where(rel <= 1e-14, 2 * np.pi, rel)
    rel = np.sort(rel)
    nsec = len(rel)
    targets = (phi0 + rel[None, :] + 2 * np.pi * np.arange(nturns)[:, None]).ravel()
    if h is None:
        h = 2 * np.pi / 128
    A, b, c, E = gauss_irk(stages)
    tm = _trace_many_pairs if doubling else _trace_many
    R1, Z1, R2, Z2, st = tm(F.packed, R0, Z0, float(phi0), targets, float(h), A, b, c, E, float(tol),
                                     int(maxit), np.asarray(bounds, dtype=float), bool(doubling))
    shape = (len(R0), nturns, nsec)
    out = {"phi": (targets.reshape(nturns, nsec)), "h": h, "stages": stages}
    if doubling:
        diff = np.hypot(R1 - R2, Z1 - Z2)
        out.update(R=R2.reshape(shape), Z=Z2.reshape(shape), err=diff.reshape(shape),
                   err_richardson=(diff / (2.0 ** (2 * stages) - 1)).reshape(shape),
                   R_h=R1.reshape(shape), Z_h=Z1.reshape(shape))
    else:
        out.update(R=R1.reshape(shape), Z=Z1.reshape(shape))
    lost = st[:, 0, 2].astype(int)
    if doubling:
        l2 = st[:, 1, 2].astype(int)
        lost = np.where((l2 >= 0) & ((lost < 0) | (l2 < lost)), l2, lost)
    out["lost"] = lost
    out["evals"] = st[:, :, 0].sum(1)
    out["max_iter"] = st[:, :, 1].max(1)
    out["nonconverged_steps"] = st[:, :, 3].sum(1)
    return out
