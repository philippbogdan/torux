"""Fast CPU field-line tracing of many lines at once (numba field kernels behind the engine).

torux.trace integrates inside jax.jit, which on a CPU runs the slow XLA kernels.  Here every
stage of the integrator is ONE concrete ``engine.evaluate(x, grad=False, A=False)`` call on all
current points, which the engine dispatches to its numba kernels (torux.field.cpu).

Integrator: Gragg-Bulirsch-Stoer extrapolation over macro steps H, per line (H may differ in
size and sign between lines): modified midpoint rule with n_j = 2 j substeps (j = 1..k), the k
sequences advanced together (the m-th substep of every sequence with n_j > m is one evaluation
call: 2k calls and k^2 L points per macro step), Aitken-Neville extrapolation in h^2 (order 2k).
The local error estimate is |T_kk - T_k,k-1| (the difference of the last two diagonal entries:
the error of the order 2k-2 value, an overestimate of the order 2k value's error).

Two parameterisations:
* in the toroidal angle phi (``LineTracer``): dR/dphi = R B_R / B_phi, dZ/dphi = R B_Z / B_phi; a
  line is *stopped* (flagged ``lost``) at the first stage where |B_phi| < bphi_frac |B|, B_phi has
  the wrong sign (opposite to the orientation ``sign``), R <= 0, or the field is not finite --
  the line cannot be followed in phi there.  The last state before the failing macro step is kept,
  so the line can be continued in arc length.
* in arc length (``trace_arclength``): dx/ds = sigma B / |B| in Cartesian coordinates, adaptive
  step per line (accept when the estimate is below tol), until a per-line stop condition holds;
  it follows lost lines through the turning point of phi (torux.barrier lost-point proofs).

Accuracy is checked against torux.trace (JAX GBS) and the independent reference tracer
torux.reference.trace (Gauss-Legendre IRK on the fast64 field).
"""
from __future__ import annotations

import math
import time

import numpy as np


# ----------------------------------------------------------------------------- GBS core
def _neville(T, ns):
    """Aitken-Neville extrapolation in h^2 of T[j] (k, L, d) from step counts ns.  Returns the
    extrapolated value and |T_kk - T_k,k-1| (L,)."""
    k = len(ns)
    tab = [T[0]]
    for j in range(1, k):
        row = [T[j]]
        for kk in range(1, j + 1):
            r = (ns[j] / ns[j - kk]) ** 2 - 1.0
            row.append(row[kk - 1] + (row[kk - 1] - tab[kk - 1]) / r)
        tab = row
    y = tab[-1]
    err = np.max(np.abs(tab[-1] - tab[-2]), axis=-1) if k > 1 else np.zeros(y.shape[0])
    return y, err


def gbs_step(f, t, y, H, k=6):
    """One extrapolated macro step for every line.

    f(t (n,), y (n, d)) -> (dy (n, d), ok (n,)); t (L,), y (L, d), H (L,) (any signs).
    Returns y_new (L, d), err (L,), ok (L,) (False where any stage failed)."""
    L, d = y.shape
    ns = [2 * j for j in range(1, k + 1)]
    f0, ok = f(t, y)
    ok = ok.copy()
    f0 = np.where(ok[:, None], f0, 0.0)
    hs = np.stack([H / n for n in ns])                      # (k, L)
    zprev = np.broadcast_to(y, (k, L, d)).copy()
    zcur = zprev + hs[:, :, None] * f0[None]
    T = [None] * k
    if ns[0] == 1:
        T[0] = zcur[0].copy()
    for m in range(1, ns[-1]):
        J = [j for j in range(k) if ns[j] > m]
        tt = np.concatenate([t + m * hs[j] for j in J])
        zz = np.concatenate([zcur[j] for j in J])
        fz, okz = f(tt, zz)
        okz = okz.reshape(len(J), L)
        ok &= np.all(okz, axis=0)
        fz = np.where(okz.reshape(-1)[:, None], fz, 0.0).reshape(len(J), L, d)
        for a, j in enumerate(J):
            znew = zprev[j] + 2.0 * hs[j][:, None] * fz[a]
            zprev[j] = zcur[j]
            zcur[j] = znew
            if m + 1 == ns[j]:
                T[j] = znew.copy()
    ynew, err = _neville(T, ns)
    return ynew, err, ok


# ----------------------------------------------------------------------------- phi tracing
class LineTracer:
    """Field lines in the toroidal angle for many lines at once.

    engine: torux.field.engine.FieldEngine (or any object with evaluate(x, grad, A) -> dict(B=...)).
    nfp: field periods (sets the default macro step: ``steps_per_period`` per 2 pi / nfp).
    sign: orientation of B_phi (+1 / -1); lines where B_phi takes the other sign are stopped.
    bphi_frac: lines are stopped where |B_phi| < bphi_frac |B| (torux.trace's convention, 1e-3)."""

    def __init__(self, engine, nfp, k=6, steps_per_period=6, bphi_frac=1e-3, sign=1.0, field=None):
        self.engine = engine
        self.nfp = int(nfp)
        self.k = int(k)
        self.sp = int(steps_per_period)
        self.frac = float(bphi_frac)
        self.sign = float(sign)
        self.period = 2 * np.pi / self.nfp
        self._field = field                    # optional B(x (n,3)) -> (n,3) (e.g. the reference field)
        self.n_eval = 0
        self.n_calls = 0

    def B(self, x):
        self.n_calls += 1
        self.n_eval += len(x)
        if self._field is not None:
            return np.asarray(self._field(x), dtype=float)
        from . import fast_trace as _so4
        if _so4.on("bonly") or isinstance(self.engine, _so4.LeanTraceEngine):
            # (fast_trace 'bonly') the same kernels and order without evaluate's other arrays and JAX conversions
            return _so4.field_B(self.engine, x)
        return np.asarray(self.engine.evaluate(x, grad=False, A=False)["B"], dtype=float)

    def rhs(self, phi, y):
        R, Z = y[:, 0], y[:, 1]
        c, s = np.cos(phi), np.sin(phi)
        x = np.stack([R * c, R * s, Z], -1)
        B = self.B(x)
        BR = B[:, 0] * c + B[:, 1] * s
        Bp = -B[:, 0] * s + B[:, 1] * c
        Bm = np.sqrt(np.sum(B * B, -1))
        ok = np.isfinite(Bm) & (R > 0) & (self.sign * Bp >= self.frac * Bm) & np.all(np.isfinite(y), -1)
        Bps = np.where(ok, Bp, 1.0)
        return np.stack([R * BR / Bps, R * B[:, 2] / Bps], -1), ok

    def trace(self, y0, phi0, dphi, nsteps=None, record=None):
        """Integrate lines y0 (L, 2) from phi0 (scalar or (L,)) over dphi (scalar or (L,); any sign)
        in nsteps equal macro steps (default: steps_per_period per field period of |dphi|).

        Returns dict(y (L,2) end points (last good state for stopped lines), err (L,) summed
        estimates, lost (L,) bool, phi (L,) angle of y, steps (L,) completed macro steps,
        rec (list of (L,2) states after each macro step) if record)."""
        from . import fast_kernels as _so5
        if _so5.on("ftrace"):
            # (fast_kernels 'ftrace') the same integration, all stages of all lines in one compiled call: the same bits
            o = _so5.fused_trace(self, y0, phi0, dphi, nsteps, record)
            if o is not None:
                return o
        y = np.array(y0, dtype=float).reshape(-1, 2)
        L = y.shape[0]
        phi = np.broadcast_to(np.asarray(phi0, dtype=float), (L,)).copy()
        dphi = np.broadcast_to(np.asarray(dphi, dtype=float), (L,)).copy()
        if nsteps is None:
            nsteps = max(1, int(math.ceil(np.max(np.abs(dphi)) / self.period * self.sp - 1e-9)))
        H = dphi / nsteps
        err = np.zeros(L)
        lost = ~np.all(np.isfinite(y), -1)
        steps = np.zeros(L, int)
        rec = [] if record else None
        for i in range(nsteps):
            act = np.flatnonzero(~lost)
            if len(act) == 0:
                if rec is not None:
                    rec.append(y.copy())
                continue
            yn, e, ok = gbs_step(self.rhs, phi[act], y[act], H[act], self.k)
            ok &= np.all(np.isfinite(yn), -1)
            good = act[ok]
            y[good] = yn[ok]
            phi[good] = phi[good] + H[good]
            err[good] += e[ok]
            steps[good] += 1
            lost[act[~ok]] = True
            if rec is not None:
                rec.append(y.copy())
        out = dict(y=y, err=err, lost=lost, phi=phi, steps=steps)
        if rec is not None:
            out["rec"] = rec
        return out

    def period_map(self, y, nper=1, direction=+1, phi0=0.0, nsteps=None):
        """nper field periods forward (direction=+1) or backward (-1) from the plane phi0; per-line
        direction allowed (array).  Returns (y, lost, err)."""
        y = np.asarray(y, dtype=float).reshape(-1, 2)
        d = np.broadcast_to(np.asarray(direction, dtype=float), (len(y),))
        o = self.trace(y, phi0, d * nper * self.period,
                       nsteps=nsteps if nsteps is not None else self.sp * nper)
        return o["y"], o["lost"], o["err"]


# ----------------------------------------------------------------------------- arc-length tracing
def trace_arclength(Bfun, x0, sigma, stop, S_max, H0=1e-3, tol=1e-12, k=6, H_max=0.02, max_steps=20000,
                    record=False):
    """Field lines in arc length, dx/ds = sigma B/|B|, adaptive step per line.

    Bfun(x (n,3)) -> B (n,3).  x0 (L,3), sigma (L,) = +-1.  stop(x (n,3)) -> bool (n,): a line ends at
    the first accepted step where stop holds.  Lines end also at S_max.  Returns dict(x, s, stopped,
    err (summed local estimates), n_steps, path (list per line) if record)."""
    x = np.array(x0, dtype=float).reshape(-1, 3)
    L = len(x)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (L,)).copy()

    def f(t, xx, sg):
        B = np.asarray(Bfun(xx), dtype=float)
        Bm = np.sqrt(np.sum(B * B, -1))
        ok = np.isfinite(Bm) & (Bm > 0)
        return sg[:, None] * B / np.where(ok, Bm, 1.0)[:, None], ok

    s = np.zeros(L)
    H = np.full(L, float(H0))
    err = np.zeros(L)
    done = np.zeros(L, bool)
    stopped = np.asarray(stop(x), bool).copy()
    done |= stopped
    nst = np.zeros(L, int)
    paths = [[x[i].copy()] for i in range(L)] if record else None
    it = 0
    while not np.all(done) and it < max_steps:
        it += 1
        act = np.flatnonzero(~done)
        Ha = np.minimum(H[act], S_max - s[act])

        def fa(t, xx, _sg=sig[act]):
            n = len(xx) // len(act)
            return f(t, xx, np.tile(_sg, n))
        xn, e, ok = gbs_step(fa, s[act], x[act], Ha, k)
        acc = ok & (e <= tol) & np.all(np.isfinite(xn), -1)
        ia = act[acc]
        x[ia] = xn[acc]
        s[ia] += Ha[acc]
        err[ia] += e[acc]
        nst[ia] += 1
        if record:
            for i in ia:
                paths[i].append(x[i].copy())
        fac = np.where(ok & (e > 0), 0.9 * (tol / np.maximum(e, 1e-300)) ** (1.0 / (2 * k - 1)), 3.0)
        fac = np.clip(fac, 0.2, 3.0)
        fac = np.where(ok, fac, 0.25)
        H[act] = np.minimum(Ha * fac, H_max)
        if len(ia):
            st = np.asarray(stop(x[ia]), bool)
            stopped[ia[st]] = True
            done[ia[st]] = True
        done |= s >= S_max * (1 - 1e-12)
        done |= H < 1e-14
    out = dict(x=x, s=s, stopped=stopped, err=err, n_steps=nst)
    if record:
        out["path"] = [np.array(p) for p in paths]
    return out
