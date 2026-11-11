"""Tori from outside where the design-surface Newton and the continuation fail: circles, seeds, walks.

Requests next to low-order rationals with little shear, or needing very high resolution, can have a flux surface
that neither Newton from the design surface nor the continuation from the axis reaches.  This module finds it on
the solver's own machinery (torux.torus, torux.family).  Entry point:

    seed_and_walk(engine, source, request, base, axis, deadline=None, verbose=False) -> (TorusState or None, log)

Pieces:
* design_guess: torux.design.sfl_guess_from_boozer with a step-limited angle Newton.  The undamped one diverges on
  strongly shaped design surfaces (QUASR 0136030, 2039780, 2093639: angle errors 1.3-25 rad), so their design tori
  were not the surface; with the step limit their design Newton converges at once (0136030: 4 iterations at
  M, N = 40, 80).  This, not resonance, is why the design path missed 0136030 and 2039780.
* SectionTracer: field lines from the plane phi = 0, many at once, by a compiled Gragg-Bulirsch-Stoer integrator
  (torux.fasttrace's scheme) with the Biot-Savart sum of the engine's filament nodes inline, parallel over lines
  (3-10x torux.fasttrace.LineTracer, which makes one engine call per stage; agreement 1e-13, against the independent
  reference tracer 1.6e-14 per period); macro steps per period calibrated per order
  to 1e-9 (long traces) and 1e-11 (one-period maps) by step doubling -- set by the coils' distance, not nfp (QUASR
  0136030, nfp 1: 48 and 64 steps of order 12; LineTracer's default 6 gives 1e-2).
* rotation: weighted Birkhoff averages; digits from the whole trace and both halves (the halves alone can agree to
  rounding while both are off next to a rational).
* locate: lines on the outboard symmetry line around the request's predicted position (from the base torus), traced
  ~150 poloidal turns and 3 sweeps of the nearest low-order rational's chain; regular orbits' fluxes by loop integrals
  of A along their fitted circles; the window narrowed to the bracket of regular orbits around the request.
* Circle, fit_circle, refine_rotation: the symmetric invariant circle of an orbit in its rotation phase (modes
  limited by the largest phase gap, misfit on held-out points), the rotation refined to the least misfit about the
  trace's own Birkhoff average.  torus_from_circle: the torus at any (M, N) from the lines launched on a circle and
  followed over one field period, put on the solver's grid by a phase shift per section.
* circle_newton: the invariant circle of the period map by Gauss-Newton on its symmetric Fourier coefficients, the
  flux through it fixed (rotation free) or the rotation fixed; 2K + 2 one-period lines per residual, the tangent
  map by forward differences, K grown where the floor is the truncation.  The 1-D problem integrates the toroidal
  variation exactly (its size does not grow with the torus's N) and needs no trace sweeping the circle.  It is the
  base polish (the circle at the request's flux from the failed torus's own section: no tracing at all) and the
  circle polish (from a refinement's plateaued torus).
* lsq_seed: the least-squares torus through an orbit's samples in two angles (aliasing bound on the samples per
  period, trace long enough to separate the closest frequencies, held-out misfit), for orbits whose circle is gapped.
* walk: from a seeded torus in sqrt(flux) to the request, resolution raised where a step converges unresolved.
* accept: off-grid B.n <= torux.analysis.BN_SURFACE (1e-14) at the request (flux to 1e-12), the resonance guard
  with the shear from the family or from one linear solve with the torus Jacobian (linear_shear), the quadrature
  estimate, simple sections around the axis, and the independent reference check (torux.verify.verify_torus,
  torux.verify.acceptance).  Nothing is returned without it.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import budget as _budget
from .torus import Basis, TorusState, coefficients_from_grid


# ============================================================================ design guess (robust angle map)
def design_guess(surf, iota, M, N, newton=60, max_step=0.25):
    """torux.design.sfl_guess_from_boozer with a step-limited angle Newton.

    The solver's angles at a design-surface point solve theta_B - iota (phi_B - phi_c) = theta, phi_c = phi (2D
    Newton per grid point).  Undamped, the iteration jumps branches where d phi_c / d phi_B is small (strongly
    shaped sections, large |iota|): on QUASR 0136030, 2039780 and 2093639 it ends with angle errors of 1.3 to 25 rad
    (12 iterations) and the guess is not the surface -- its spectrum never decays, start_resolution asks for
    M, N = 224, 256 and Newton diverges from a 0.12 m residual.  Steps limited to max_step rad converge to 1e-15 in
    7 to 10 iterations on all three; where the undamped one converges the result is identical."""
    from .torus import TorusState as _TS
    nfp = surf.nfp
    th = np.arange(256) / 256
    r0, _, rt0 = surf.gamma_and_derivs(np.zeros(256), th)
    R0s = np.hypot(r0[:, 0], r0[:, 1])
    area = np.mean(R0s * rt0[:, 2] - r0[:, 2] * (r0[:, 0] * rt0[:, 0] + r0[:, 1] * rt0[:, 1]) / R0s)
    flip = area < 0
    if flip:
        base = surf

        class _Flipped:
            nfp = base.nfp

            def gamma_and_derivs(self, p, t):
                r, rp, rt = base.gamma_and_derivs(p, -np.asarray(t))
                return r, rp, -rt
        surf = _Flipped()
    b = Basis(nfp, M, N)
    TH, PH = np.meshgrid(b.theta, b.phi, indexing="ij")
    th_t, ph_t = TH.ravel(), PH.ravel()
    tB = th_t.copy()
    pB = ph_t.copy()
    err = np.inf
    for it in range(newton):
        r, rp, rt = surf.gamma_and_derivs(pB / (2 * np.pi), tB / (2 * np.pi))
        R2 = r[:, 0] ** 2 + r[:, 1] ** 2
        pc = np.arctan2(r[:, 1], r[:, 0])
        pc = pB + ((pc - pB + np.pi) % (2 * np.pi) - np.pi)
        dpc_dp = (r[:, 0] * rp[:, 1] - r[:, 1] * rp[:, 0]) / R2 / (2 * np.pi)
        dpc_dt = (r[:, 0] * rt[:, 1] - r[:, 1] * rt[:, 0]) / R2 / (2 * np.pi)
        f1 = tB - iota * (pB - pc) - th_t
        f2 = (pc - ph_t + np.pi) % (2 * np.pi) - np.pi
        err = float(np.max(np.abs(np.concatenate([f1, f2]))))
        if err < 1e-14:
            break
        a11 = 1 + iota * dpc_dt
        a12 = -iota * (1 - dpc_dp)
        a21, a22 = dpc_dt, dpc_dp
        det = a11 * a22 - a12 * a21
        dt = (f1 * a22 - a12 * f2) / det
        dp = (a11 * f2 - a21 * f1) / det
        s = np.minimum(1.0, max_step / np.maximum(np.maximum(np.abs(dt), np.abs(dp)), 1e-300))
        tB -= s * dt
        pB -= s * dp
    r, _, _ = surf.gamma_and_derivs(pB / (2 * np.pi), tB / (2 * np.pi))
    R = np.hypot(r[:, 0], r[:, 1]).reshape(TH.shape)
    Z = r[:, 2].reshape(TH.shape)
    rc, zs = coefficients_from_grid(b, R, Z)
    st = _TS(np.asarray(rc), np.asarray(zs), float(iota), 0.0, M, N, nfp)
    st.info["angle_newton_err"] = err
    st.info["angle_newton_its"] = it + 1
    st.info["flipped"] = bool(flip)
    return st


# ============================================================================ small helpers
def _period(nfp):
    return 2 * np.pi / nfp


def bphi_sign(engine, axis):
    """Sign of B_phi on the axis at phi = 0 (field lines are traced in that direction of phi)."""
    R, Z = axis.at(np.array([0.0]))
    B = np.asarray(engine.evaluate(np.array([[float(R[0]), 0.0, float(Z[0])]]), grad=False, A=False)["B"])
    return 1.0 if B[0, 1] >= 0 else -1.0


def outboard_point(t: TorusState):
    """The outboard symmetric point of a torus at phi = phi0 (theta = 0 or pi, whichever has the larger R; QUASR
    design tori in the solver's angles have theta = 0 inboard).  Returns (R, theta)."""
    R0, _ = t.geometry_on(np.array([0.0]), np.array([t.phi0]))
    R1, _ = t.geometry_on(np.array([np.pi]), np.array([t.phi0]))
    return (float(R0[0]), 0.0) if R0[0] >= R1[0] else (float(R1[0]), float(np.pi))


def near_rational(x, qmax=30):
    """(p, q, |x - p/q|) of the nearest rational with q <= qmax, weighting nothing (smallest distance)."""
    best = None
    for q in range(1, qmax + 1):
        p = round(x * q)
        d = abs(x - p / q)
        if best is None or d < best[2] - 1e-15:
            best = (int(p), q, d)
    return best


def near_resonance_need(rot, qmax=30):
    """Periods a line of rotation rot (turns per period) needs to sweep its circle about one and a half times:
    next to p/q its section phases advance by q |rot - p/q| per period."""
    need = 0
    for q in range(1, qmax + 1):
        d = abs(rot - round(rot * q) / q)
        if d > 0:
            need = max(need, int(np.ceil(1.6 / (q * d))))
    return need


# ============================================================================ tracing
GBS_EVALS = {k: 1 + k * k for k in range(1, 10)}          # field evaluations per macro step of gbs_step(k)

_BS_KERNEL = []


def _bs_kernel():
    """B of Fourier-filament nodes at a few targets, parallel over source chunks (the engine's kernel is parallel
    over blocks of 16 targets: a tracer step with 8 - 50 points keeps one to three threads busy and pads the
    block), the source loop vectorised (fastmath: reassociated sums, ~1e-15 relative -- tracing needs 1e-11)."""
    if not _BS_KERNEL:
        from numba import njit, prange

        @njit(parallel=True, fastmath=True, cache=True)
        def kern(x, r, w, nchunk):
            T = x.shape[0]
            S = r.shape[0]
            part = np.zeros((nchunk, T, 3))
            for c in prange(nchunk):
                j0 = c * S // nchunk
                j1 = (c + 1) * S // nchunk
                for t in range(T):
                    p0 = x[t, 0]; p1 = x[t, 1]; p2 = x[t, 2]
                    a0 = 0.0; a1 = 0.0; a2 = 0.0
                    for j in range(j0, j1):
                        d0 = p0 - r[j, 0]; d1 = p1 - r[j, 1]; d2 = p2 - r[j, 2]
                        q = d0 * d0 + d1 * d1 + d2 * d2
                        ir = 1.0 / np.sqrt(q)
                        ir3 = ir * ir * ir
                        a0 += (w[j, 1] * d2 - w[j, 2] * d1) * ir3
                        a1 += (w[j, 2] * d0 - w[j, 0] * d2) * ir3
                        a2 += (w[j, 0] * d1 - w[j, 1] * d0) * ir3
                    part[c, t, 0] = a0; part[c, t, 1] = a1; part[c, t, 2] = a2
            out = np.zeros((T, 3))
            for c in range(nchunk):
                out += part[c]
            return out
        _BS_KERNEL.append(kern)
    return _BS_KERNEL[0]


_GBS_LINES = []


def _gbs_lines_kernel():
    """Compiled GBS integration of many lines in phi (the scheme of torux.fasttrace.gbs_step: modified midpoint with
    n_j = 2 j substeps, j = 1..k, Aitken-Neville in h^2), the Biot-Savart sum of the filament nodes inline, parallel
    over lines.  The Python tracer makes one engine call per stage with a few points each (QUASR 1063007, 24 lines:
    ~1e4 evaluations/s on this machine where the kernel does 1e5+); here a call integrates whole periods."""
    if not _GBS_LINES:
        from numba import njit, prange

        @njit(fastmath=True, cache=True)
        def rhs(phi, R, Z, r, w, sign, frac):
            c = np.cos(phi); s_ = np.sin(phi)
            x = R * c; y = R * s_
            a0 = 0.0; a1 = 0.0; a2 = 0.0
            for j in range(r.shape[0]):
                d0 = x - r[j, 0]; d1 = y - r[j, 1]; d2 = Z - r[j, 2]
                q = d0 * d0 + d1 * d1 + d2 * d2
                ir = 1.0 / np.sqrt(q)
                ir3 = ir * ir * ir
                a0 += (w[j, 1] * d2 - w[j, 2] * d1) * ir3
                a1 += (w[j, 2] * d0 - w[j, 0] * d2) * ir3
                a2 += (w[j, 0] * d1 - w[j, 1] * d0) * ir3
            BR = a0 * c + a1 * s_
            Bp = -a0 * s_ + a1 * c
            Bm = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            ok = (sign * Bp >= frac * Bm) and (R > 0) and np.isfinite(Bm)
            if not ok:
                return 0.0, 0.0, False
            return R * BR / Bp, R * a2 / Bp, True

        @njit(parallel=True, fastmath=False, cache=True)
        def run(y0, phi0, H, nsteps, k, r, w, sign, frac, rec_every, axR, axZ, per):
            """y0 (L, 2); nsteps macro steps of size H from phi0; records every rec_every steps (rec (L, nrec, 2));
            per > 0: angle increments about the axis (axR, axZ at the per + 1 step phases of a period, unwrapped
            along the steps) summed over each `per` steps (dang (L, nsteps // per)).  lost (L,) the step a line was
            lost at (-1: never)."""
            L = y0.shape[0]
            nrec = nsteps // rec_every + 1
            rec = np.full((L, nrec, 2), np.nan)
            nd = nsteps // per if per > 0 else 0
            dang = np.full((L, max(nd, 1)), np.nan)
            lost = np.full(L, -1, np.int64)
            for i in prange(L):
                R = y0[i, 0]; Z = y0[i, 1]
                rec[i, 0, 0] = R; rec[i, 0, 1] = Z
                T = np.empty((k, 2))
                phi = phi0
                acc = 0.0
                ang = 0.0
                if per > 0:
                    ang = np.arctan2(Z - axZ[0], R - axR[0])
                for st in range(nsteps):
                    f0R, f0Z, ok = rhs(phi, R, Z, r, w, sign, frac)
                    if not ok:
                        lost[i] = st
                        break
                    good = True
                    for jj in range(k):
                        n = 2 * (jj + 1)
                        h = H / n
                        z0R = R; z0Z = Z
                        z1R = R + h * f0R; z1Z = Z + h * f0Z
                        for m in range(1, n):
                            fR, fZ, ok2 = rhs(phi + m * h, z1R, z1Z, r, w, sign, frac)
                            if not ok2:
                                good = False
                                break
                            z2R = z0R + 2.0 * h * fR; z2Z = z0Z + 2.0 * h * fZ
                            z0R = z1R; z0Z = z1Z
                            z1R = z2R; z1Z = z2Z
                        if not good:
                            break
                        T[jj, 0] = z1R; T[jj, 1] = z1Z
                    if not good:
                        lost[i] = st
                        break
                    # Aitken-Neville in h^2 (as torux.fasttrace._neville): tab[j][kk] from tab[j][kk-1], tab[j-1][kk-1]
                    tab0 = np.empty((k, k)); tab1 = np.empty((k, k))
                    for jj in range(k):
                        tab0[jj, 0] = T[jj, 0]; tab1[jj, 0] = T[jj, 1]
                        for kk in range(1, jj + 1):
                            ratio = ((jj + 1.0) / (jj + 1.0 - kk)) ** 2 - 1.0
                            tab0[jj, kk] = tab0[jj, kk - 1] + (tab0[jj, kk - 1] - tab0[jj - 1, kk - 1]) / ratio
                            tab1[jj, kk] = tab1[jj, kk - 1] + (tab1[jj, kk - 1] - tab1[jj - 1, kk - 1]) / ratio
                    R = tab0[k - 1, k - 1]; Z = tab1[k - 1, k - 1]
                    if not (np.isfinite(R) and np.isfinite(Z)):
                        lost[i] = st
                        break
                    phi = phi + H
                    if per > 0:
                        q = (st + 1) % per
                        a = np.arctan2(Z - axZ[q], R - axR[q])
                        d = a - ang
                        d -= 2.0 * np.pi * np.round(d / (2.0 * np.pi))
                        acc += d
                        ang = a
                        if q == 0:
                            dang[i, (st + 1) // per - 1] = acc
                            acc = 0.0
                    if (st + 1) % rec_every == 0:
                        rec[i, (st + 1) // rec_every, 0] = R
                        rec[i, (st + 1) // rec_every, 1] = Z
            return rec, dang, lost
        _GBS_LINES.append(run)
    return _GBS_LINES[0]


class FastB:
    """B(x) for tracing: the engine's Fourier-filament nodes summed by _bs_kernel (engines with filaments only;
    anything else falls back to engine.evaluate).  Checked against engine.evaluate at construction."""

    def __init__(self, engine, check=True):
        import numba
        self.engine = engine
        P = getattr(engine, "_np", None)
        only_fil = (P is not None and getattr(engine, "cpu_kernels", False) and len(P["r"]) > 0
                    and len(P["sa"]) == 0 and len(P["dp"]) == 0 and not any(getattr(engine, "_rest_kinds", ()) or ())
                    and not getattr(engine, "analytic", None))
        self.ok = bool(only_fil)
        if self.ok:
            self.r = np.ascontiguousarray(np.asarray(P["r"], float))
            self.w = np.ascontiguousarray(np.asarray(P["w"], float))
            self.nchunk = int(max(1, 2 * numba.get_num_threads()))
            if check:
                rng = np.random.default_rng(0)
                c = np.mean(self.r, axis=0)
                rad = float(np.median(np.linalg.norm(self.r[:, :2], axis=1)))
                ph = rng.uniform(0, 2 * np.pi, 16)
                x = np.stack([rad * np.cos(ph), rad * np.sin(ph), c[2] + 0 * ph], -1)
                Be = np.asarray(engine.evaluate(x, grad=False, A=False)["B"])
                Bf = self(x)
                self.check_err = float(np.max(np.linalg.norm(Bf - Be, axis=1) / np.linalg.norm(Be, axis=1)))
                if not self.check_err < 1e-12:
                    self.ok = False

    SMALL = 96          # below: parallel over sources; above: the engine's target-parallel kernel

    def __call__(self, x):
        x = np.ascontiguousarray(np.asarray(x, float).reshape(-1, 3))
        if not self.ok or len(x) >= self.SMALL:
            from . import fast_trace as _so4
            if _so4.on("bonly"):
                return _so4.field_B(self.engine, x)        # (fast_trace 'bonly') same kernels, same order
            return np.asarray(self.engine.evaluate(x, grad=False, A=False)["B"])
        return _bs_kernel()(x, self.r, self.w, self.nchunk)


def trace_engine(engine, source, eps=1e-12):
    """The engine for tracing: the same per-coil distances (region) with the quadrature rule for 1e-12 instead of
    1e-17 (QUASR 1063007: 4128 nodes instead of 5760, B within 1.3e-15 of the solver's engine at sample points --
    the rule's estimate is conservative).  Falls back to the engine itself."""
    try:
        from .field import FieldEngine
        e = FieldEngine(source, dmin=engine.dmin, region_pts=getattr(engine, "region_pts", None), eps=eps)
        return e if e.info.n_nodes < engine.info.n_nodes else engine
    except Exception:  # noqa: BLE001
        return engine


class SectionTracer:
    """Field lines from the plane phi = 0 for many lines at once on the exact engine (torux.fasttrace.LineTracer:
    one engine call per integrator stage for all lines), with the macro steps per period calibrated per
    extrapolation order k to a local error target per period (step doubling on sample lines).

    The steps a field period needs are set by the coils' distance, not by nfp: QUASR 0136030 (nfp 1, coils 0.27 m
    from the surface) needs 48 macro steps of order 12 per period for 3e-11 (6 steps, LineTracer's default, give
    1e-2); torux.reference.trace checks the setting."""

    def __init__(self, engine, nfp, axis, sign=None, tol_long=1e-9, tol_short=1e-11, fast=True):
        from .fasttrace import LineTracer
        self.engine, self.nfp, self.axis = engine, int(nfp), axis
        self.fastB = FastB(engine) if fast else None
        if self.fastB is not None and not self.fastB.ok:
            self.fastB = None
        self.sign = bphi_sign(engine, axis) if sign is None else float(sign)
        self.T = _period(self.nfp)
        self.tol = dict(long=float(tol_long), short=float(tol_short))
        self.sp = {}                  # (k, level) -> macro steps per period
        self._LT = LineTracer
        self.n_eval = 0
        self.calib = {}

    def _tracer(self, k):
        return self._LT(self.engine, self.nfp, k=k, steps_per_period=8, sign=self.sign, field=self.fastB)

    def _kernel(self, y0, k, nsteps, rec_every, per=0, sp=None):
        """The compiled tracer over nsteps macro steps of T / sp (sp = per where given): (rec (L, n, 2), dang,
        lost (L,))."""
        y0 = np.ascontiguousarray(np.asarray(y0, float).reshape(-1, 2))
        if per > 0:
            Ra, Za = self.axis.at(self.T * np.arange(per + 1) / per)
        else:
            Ra = Za = np.zeros(2)
        H = self.T / (per if per > 0 else sp)
        rec, dang, lost = _gbs_lines_kernel()(y0, 0.0, H, int(nsteps), int(k), self.fastB.r, self.fastB.w, self.sign,
                                              1e-3, int(rec_every), np.ascontiguousarray(Ra), np.ascontiguousarray(Za),
                                              int(per))
        self.n_eval += len(y0) * nsteps * GBS_EVALS[k]
        return rec, dang, lost >= 0

    def _map(self, y0, k, nsteps, nout=None):
        """One period in nsteps macro steps (nout outputs if given): dict(y, lost[, rec list])."""
        if self.fastB is not None:
            every = nsteps // nout if nout else nsteps
            rec, _, lost = self._kernel(y0, k, nsteps, every, sp=nsteps)
            y = rec[:, -1, :].copy()
            for i in np.flatnonzero(~np.all(np.isfinite(y), -1)):
                # (a lost line: its last finite state, as LineTracer keeps it)
                fin = np.flatnonzero(np.all(np.isfinite(rec[i]), -1))
                y[i] = rec[i, fin[-1]] if len(fin) else np.nan
            o = dict(y=y, lost=lost)
            if nout:
                o["rec"] = [rec[:, j, :] for j in range(1, rec.shape[1])]
            return o
        tr = self._tracer(k)
        o = tr.trace(np.asarray(y0, float), 0.0, self.T, nsteps=nsteps, record=nout is not None)
        self.n_eval += tr.n_eval
        return o

    def calibrate(self, y0, ks=(6, 4, 3)):
        """Steps per period for each order k and level: the smallest on the ladder whose step-doubling difference
        over one period is below the level's tolerance (relative to R) at the sample points y0 (L, 2)."""
        y0 = np.asarray(y0, float).reshape(-1, 2)
        Rs = float(np.max(np.abs(y0[:, 0])))
        for k in ks:
            prev = None
            res = []
            # a doubling ladder (the pair's difference ~ the coarser count's error); lower orders need more steps
            # than order 12, so their ladders start at the k = 6 long-level count (order 2k error ~ h^2k: the k = 4
            # and 3 counts were 1.5-2x and 2.7-6x k = 6's on QUASR 0136030)
            start = 8 if k == ks[0] or (ks[0], "long") not in self.sp else self.sp[(ks[0], "long")] * (1 if k >= 4 else 2)
            ladder = [start * 2 ** j for j in range(9)]
            for sp in ladder:
                o = self._map(y0, k, sp)
                y = o["y"]
                if prev is not None:
                    ok = ~(o["lost"] | prev[1])
                    d = float(np.max(np.linalg.norm(y[ok] - prev[0][ok], axis=1))) / Rs if np.any(ok) else np.inf
                    res.append((sp // 2 if sp > 4 else sp, d))
                    for lev, tl in self.tol.items():
                        if (k, lev) not in self.sp and d <= tl:
                            # the coarser of the pair meets the tolerance (its error ~ the difference)
                            self.sp[(k, lev)] = prev[2]
                    if all((k, lev) in self.sp for lev in self.tol):
                        break
                prev = (y, o["lost"], sp)
            for lev in self.tol:
                self.sp.setdefault((k, lev), ladder[-1])
            self.calib[k] = res
        return self.sp

    def best_k(self, level, nout=None):
        """(k, steps per period) with the fewest evaluations per period; with nout outputs per period the steps are
        a multiple of nout."""
        best = None
        for (k, lev), sp in self.sp.items():
            if lev != level:
                continue
            steps = sp if nout is None else nout * int(math.ceil(sp / nout))
            cost = GBS_EVALS[k] * steps
            if best is None or cost < best[2]:
                best = (k, steps, cost)
        return best[0], best[1]

    def period_map(self, y0, level="short"):
        """One period from phi = 0 for every line: (y (L, 2), lost (L,))."""
        k, sp = self.best_k(level)
        o = self._map(y0, k, sp)
        return o["y"], o["lost"]

    def one_period(self, y0, nout, level="short"):
        """States at phi_j = j T / nout, j = 0..nout (L, nout + 1, 2), and lost (L,)."""
        k, steps = self.best_k(level, nout)
        if self.fastB is not None:
            rec, _, lost = self._kernel(y0, k, steps, steps // nout, sp=steps)
            return rec, lost
        o = self._map(y0, k, steps, nout=nout)
        every = steps // nout
        rec = [np.asarray(y0, float)] + [o["rec"][i] for i in range(every - 1, steps, every)]
        return np.stack(rec, 1), o["lost"]

    def orbits(self, y0, nper, deadline=None, fine=0, level="long"):
        """Trace lines nper periods from phi = 0.  Returns dict(sec (L, n + 1, 2) section points at phi = 0, dang
        (L, n) per-period angle increments about the axis (unwrapped along the macro steps), lost (L,), n (periods
        done), fine (L, n * fine + 1, 2) samples at fine points per period if fine > 0)."""
        y = np.array(y0, float).reshape(-1, 2)
        L = len(y)
        k, sp = self.best_k(level, fine if fine else None)
        if self.fastB is not None:
            # whole chunks of periods per compiled call (the deadline checked between chunks)
            dl = _budget.earliest(deadline)
            every = sp // fine if fine else sp
            secs, fines, dangs = [y[:, None, :].copy()], [y[:, None, :].copy()], []
            lost = np.zeros(L, bool)
            n = 0
            chunk = int(max(1, min(nper, 50)))
            y_start = y.copy()
            while n < nper:
                if dl is not None and time.time() > dl:
                    break
                m = min(chunk, nper - n)
                rec, dang, lo = self._kernel(y, k, m * sp, every, per=sp)
                was_lost = lost.copy()
                lost |= lo
                if was_lost.any():
                    # lines lost in an earlier chunk ran this one from a dummy start: their records are NaN
                    rec = np.array(rec, copy=True); rec[was_lost] = np.nan
                    dang = np.array(dang, copy=True); dang[was_lost] = np.nan
                if fine:
                    fines.append(rec[:, 1:, :])
                    secs.append(rec[:, fine::fine, :])
                else:
                    secs.append(rec[:, 1:, :])
                dangs.append(dang[:, :m])
                y = rec[:, -1, :].copy()
                # a lost line restarts from its own (finite) start and stays marked lost: restarted from NaN, the
                # compiled right-hand side divided by a NaN B_phi (ZeroDivisionError in the fastmath kernel when
                # a probe was lost within the first chunk, QUASR 23786)
                y[lost] = y_start[lost]
                n += m
            out = dict(sec=np.concatenate(secs, 1), dang=np.concatenate(dangs, 1) if dangs else np.zeros((L, 0)),
                       lost=lost, n=n, k=k, sp=sp)
            if fine:
                out["fine"] = np.concatenate(fines, 1)
            return out
        tr = self._tracer(k)
        phis = self.T * np.arange(sp + 1) / sp
        Ra, Za = self.axis.at(phis)
        sec = [y.copy()]
        dang = []
        fines = [y.copy()] if fine else None
        every = sp // fine if fine else 0
        lost = np.zeros(L, bool)
        ang = np.arctan2(y[:, 1] - Za[0], y[:, 0] - Ra[0])
        n = 0
        deadline = _budget.earliest(deadline)
        for it in range(int(nper)):
            if deadline is not None and time.time() > deadline:
                break
            o = tr.trace(y, 0.0, self.T, nsteps=sp, record=True)
            acc = np.zeros(L)
            prev = ang
            for j, yj in enumerate(o["rec"]):
                a = np.arctan2(yj[:, 1] - Za[j + 1], yj[:, 0] - Ra[j + 1])
                d = a - prev
                d -= 2 * np.pi * np.round(d / (2 * np.pi))
                acc += d
                prev = a
                if fine and (j + 1) % every == 0:
                    fines.append(np.where(o["lost"][:, None], np.nan, yj))
            ang = prev
            lost |= o["lost"]
            y = o["y"]
            sec.append(np.where(lost[:, None], np.nan, y))
            dang.append(np.where(lost, np.nan, acc))
            n += 1
        self.n_eval += tr.n_eval
        out = dict(sec=np.stack(sec, 1), dang=np.stack(dang, 1) if dang else np.zeros((L, 0)), lost=lost, n=n,
                   k=k, sp=sp)
        if fine:
            out["fine"] = np.stack(fines, 1)
        return out


# ============================================================================ Birkhoff averages
def _bump(n):
    t = (np.arange(n) + 0.5) / n
    w = np.exp(-1.0 / (t * (1 - t)))
    return w / w.sum()


def birkhoff(f):
    """Weighted Birkhoff average (bump weight exp(-1/(t(1-t))), Das, Sander, Yorke, Meiss)."""
    f = np.asarray(f, float)
    return float(np.sum(_bump(len(f)) * f))


def rotation(dang):
    """Rotation number (turns per period) and digits from per-period increments: -log10 of the largest difference
    among the weighted Birkhoff averages of the whole trace and of its two halves.  The halves alone can agree to
    rounding while both are far off: next to a rational p/q the averages' error is the bump weight's transform at
    the near-resonant frequency, which can take the same value on both halves; the whole trace's average, at twice
    the length, does not share it."""
    dang = np.asarray(dang, float)
    n = len(dang)
    if n < 20 or not np.all(np.isfinite(dang)):
        return np.nan, np.nan
    r = birkhoff(dang) / (2 * np.pi)
    r1 = birkhoff(dang[: n // 2]) / (2 * np.pi)
    r2 = birkhoff(dang[n // 2:]) / (2 * np.pi)
    d = max(abs(r1 - r2), abs(r - r1), abs(r - r2))
    return r, float(-np.log10(max(d, 1e-17)))


# ============================================================================ invariant circles
class Circle:
    """A symmetric closed curve in the plane phi = 0: R(t) = sum_k a_k cos 2 pi k t, Z(t) = sum_k b_k sin 2 pi k t
    (t the rotation phase, t = 0 the outboard symmetric point)."""

    def __init__(self, a, b, rot=np.nan, fit_err=np.nan):
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)
        self.b[0] = 0.0
        self.nm = len(self.a) - 1
        self.rot = float(rot)
        self.fit_err = float(fit_err)

    def __call__(self, t, deriv=0):
        t = np.asarray(t, float).ravel()
        k = np.arange(self.nm + 1)
        ang = 2 * np.pi * np.outer(t, k)
        w = (2 * np.pi * k) ** deriv
        c, s = np.cos(ang), np.sin(ang)
        if deriv % 4 == 0:
            return np.stack([c @ (w * self.a), s @ (w * self.b)], -1)
        if deriv % 4 == 1:
            return np.stack([-s @ (w * self.a), c @ (w * self.b)], -1)
        if deriv % 4 == 2:
            return np.stack([-c @ (w * self.a), -s @ (w * self.b)], -1)
        return np.stack([s @ (w * self.a), -c @ (w * self.b)], -1)

    def resized(self, nm):
        a = np.zeros(nm + 1); b = np.zeros(nm + 1)
        m = min(nm, self.nm)
        a[:m + 1] = self.a[:m + 1]; b[:m + 1] = self.b[:m + 1]
        return Circle(a, b, self.rot, self.fit_err)

    def tail(self, width=2):
        """Largest amplitude of the last `width` modes relative to the largest k >= 1 amplitude."""
        amp = np.hypot(self.a, self.b)
        sc = max(float(np.max(amp[1:])), 1e-300)
        return float(np.max(amp[-width:]) / sc)

    def area(self, nq=1024):
        t = np.arange(nq) / nq
        P, D = self(t), self(t, 1)
        return float(np.mean(P[:, 0] * D[:, 1] - P[:, 1] * D[:, 0]) / 2)


def phase_gap(rot, n):
    """Largest gap between the rotation phases of n consecutive section points."""
    ph = np.sort(np.mod(np.arange(n) * rot, 1.0))
    return float(max(np.max(np.diff(ph)), 1.0 - ph[-1] + ph[0])) if n > 1 else 1.0


def _circle_lsq(t, P, nm):
    k = np.arange(nm + 1)
    ang = 2 * np.pi * np.outer(t, k)
    Ac, As = np.cos(ang), np.sin(ang[:, 1:])
    a, *_ = np.linalg.lstsq(Ac, P[:, 0], rcond=None)
    b, *_ = np.linalg.lstsq(As, P[:, 1], rcond=None)
    return a, np.concatenate([[0.0], b])


def fit_circle(sec, rot, nm=None, max_rows=None, nm_cap=320, per_mode=4):
    """The symmetric invariant circle through section points sec (n, 2) (sec[0] the launch point on Z = 0) in
    their rotation phase t_i = i rot mod 1: least squares on the even-indexed points, misfit the larger of that
    and the misfit at the odd-indexed points it did not use (aliasing or a wrong rotation shows there).  Modes:
    at most nm_cap, n / (2 per_mode), and 1 / (2 gap) (gap the largest phase interval: next to a rational the
    points fall in short arcs and more modes are arbitrary between them).  max_rows: an even subsample over the
    whole trace for the fit (a wrong rotation's phase error grows along the trace).  Returns Circle or None."""
    sec = np.asarray(sec, float)
    good = np.all(np.isfinite(sec), axis=1)
    if not np.all(good):
        return None
    n = len(sec)
    gap = phase_gap(rot, n)
    nm_max = int(min(nm_cap, n // (2 * per_mode), 0.5 / max(gap, 1e-12)))
    nm = nm_max if nm is None else int(min(nm, nm_max))
    if nm < 2:
        return None
    t = np.mod(np.arange(n) * rot, 1.0)
    ev, od = np.arange(0, n, 2), np.arange(1, n, 2)
    if max_rows is not None and len(ev) > max_rows:
        ev = ev[np.linspace(0, len(ev) - 1, max_rows).astype(int)]
        od = od[np.linspace(0, len(od) - 1, max_rows).astype(int)]
    a, b = _circle_lsq(t[ev], sec[ev], nm)
    C = Circle(a, b, rot)
    e1 = np.max(np.abs(C(t[ev]) - sec[ev]))
    e2 = np.max(np.abs(C(t[od]) - sec[od])) if len(od) else 0.0
    C.fit_err = float(max(e1, e2))
    C.gap = gap
    return C


def refine_rotation(sec, rot0, width, nm=64, max_rows=1500, scan=41):
    """The rotation at which the circle fitted to sec has its least misfit: a
    scan of `scan` values over rot0 +- width, then golden section between the best one's neighbours.  The trace's
    own Birkhoff average is the centre (a torus's transform need not be this line's: QUASR 142814's plateaued
    torus crosses the symmetry line 2e-5 m from the circle at its flux, 1e-6 off in rotation)."""
    def mis(r):
        C = fit_circle(sec, r, nm=nm, max_rows=max_rows)
        return C.fit_err if C is not None else np.inf
    grid_r = rot0 + width * np.linspace(-1, 1, scan)
    vals = [mis(r) for r in grid_r]
    k = int(np.argmin(vals))
    lo, hi = grid_r[max(k - 1, 0)], grid_r[min(k + 1, len(grid_r) - 1)]
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = mis(x1), mis(x2)
    for _ in range(48):
        if hi - lo < 1e-15:
            break
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = mis(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = mis(x2)
    return float(0.5 * (lo + hi)), float(min(f1, f2))


def circle_field(engine, C, nq=512, A=True):
    """B (and A) at nq points of the circle in the plane phi = 0 (Cartesian x = (R, 0, Z))."""
    t = np.arange(nq) / nq
    P = C(t)
    x = np.stack([P[:, 0], 0 * P[:, 0], P[:, 1]], -1)
    o = engine.evaluate(x, grad=False, A=A)
    return t, P, np.asarray(o["B"]), (np.asarray(o["A"]) if A else None)


def circle_flux(engine, C, nq=512):
    """Toroidal flux through the circle: -int A . dC over the counter-clockwise loop (the solver's convention:
    positive where B_phi > 0), by the trapezoid rule (spectral for the periodic integrand)."""
    t, P, B, Av = circle_field(engine, C, nq)
    D = C(t, 1)
    loop = float(np.mean(Av[:, 0] * D[:, 0] + Av[:, 2] * D[:, 1]))
    orient = 1.0 if C.area() > 0 else -1.0
    return -orient * loop


def section_circle(t: TorusState, nm=None, nq=None):
    """The torus's section at phi = phi0 as a Circle in its own angle, phase origin at the outboard point (the
    torus's theta is its straight-field-line angle, i.e. the rotation phase of its section)."""
    _, th0 = outboard_point(t)
    nm = nm or t.M
    nq = nq or max(64, 4 * nm + 8)
    th = th0 + 2 * np.pi * np.arange(nq) / nq
    R, Z = t.geometry_on(th, np.full(nq, t.phi0))
    s = np.arange(nq) / nq
    a, b = _circle_lsq(s, np.stack([R, Z], -1), nm)
    C = Circle(a, b, t.iota / t.nfp)
    if C.area() < 0:        # theta clockwise: reverse the phase
        C = Circle(a, -b, -t.iota / t.nfp)
    return C


# ============================================================================ tori from circles and orbits
def torus_from_circle(tracer: SectionTracer, C: Circle, rot, M, N, level="short"):
    """The straight-field-line torus through the invariant circle C (rotation rot per period): the lines from
    C(t_i), t_i = i / ntheta, followed over one field period with nphi outputs give X(2 pi t_i + iota phi_j, phi_j),
    iota = rot nfp; a phase shift per section puts them on the uniform grid of torux.torus.Basis(nfp, M, N), whose
    projection is the torus at (M, N).  Returns TorusState (flux not set) or None if a line is lost."""
    nfp = tracer.nfp
    b = Basis(nfp, M, N)
    nth, nph = b.ntheta, b.nphi
    t = np.arange(nth) / nth
    P0 = C(t)
    traj, lost = tracer.one_period(P0, nph, level=level)
    if np.any(lost) or not np.all(np.isfinite(traj)):
        return None
    P = traj[:, :nph, :]                                   # phi_j = j T / nph, j < nph
    iota = rot * nfp
    m = np.fft.fftfreq(nth, 1.0 / nth)[:, None]
    phj = (tracer.T / nph) * np.arange(nph)[None, :]
    shift = np.exp(-1j * m * iota * phj)
    if nth % 2 == 0:
        shift[nth // 2] = 0.0
    Rg = np.real(np.fft.ifft(np.fft.fft(P[:, :, 0], axis=0) * shift, axis=0))
    Zg = np.real(np.fft.ifft(np.fft.fft(P[:, :, 1], axis=0) * shift, axis=0))
    rc, zs = coefficients_from_grid(b, Rg, Zg)
    st = TorusState(np.asarray(rc), np.asarray(zs), float(iota), 0.0, M, N, nfp)
    # the projection's residual on the grid (the stellarator-symmetric parity: what the symmetric basis cannot hold)
    R2, Z2 = st.geometry_grid(b.theta, b.phi)
    st.info["sym_misfit"] = float(max(np.max(np.abs(R2 - Rg)), np.max(np.abs(Z2 - Zg))))
    return st


def lsq_seed(tracer: SectionTracer, R0, rot, M, N, nper=None, deadline=None, Z0=0.0):
    """The least-squares torus through a traced orbit: samples nsub times
    per period on the line theta = iota phi (iota = rot nfp from a long trace: the seed's own short trace can be
    1e-4 off next to a resonance), R and Z fitted as cosine and sine series at (M, N); nsub above the aliasing
    bound 2 M |iota| / nfp + 2 N, the trace long enough to separate the two closest frequencies of the modes, an
    irregular subset of 12 rows per unknown, misfit also at the samples between the fitted ones.  Returns
    (TorusState, fit_err) or (None, inf)."""
    nfp = tracer.nfp
    iota = rot * nfp
    nsub = max(32, int(np.ceil(2 * M * abs(iota) / nfp)) + 2 * N + 8)
    dm = np.arange(1, 2 * M + 1)[:, None]
    dn = np.arange(-2 * N, 2 * N + 1)[None, :]
    dw = np.abs(dm * abs(iota) - dn * nfp)
    dw_min = max(float(dw[dw > 1e-12].min()) if np.any(dw > 1e-12) else 1.0, 2e-3)
    nper = nper or min(2000, max(200, int(np.ceil(4 * (2 * M + 1) * (2 * N + 1) / nsub)) + 50, int(np.ceil(nfp / dw_min))))
    o = tracer.orbits(np.array([[R0, Z0]]), nper, deadline=deadline, fine=2 * nsub, level="long")
    if o["n"] < nper or o["lost"][0]:
        return None, np.inf
    pc2 = o["fine"][0]
    pc, pc_odd = pc2[0::2], pc2[1::2]
    dphi = tracer.T / nsub
    phis = dphi * np.arange(len(pc))
    b = Basis(nfp, M, N)
    Mc, Ms = b.modes_c, b.modes_s
    rng = np.random.default_rng(0)
    nrow = 12 * max(len(Mc), len(Ms))
    sel = np.sort(rng.choice(len(pc), nrow, replace=False)) if len(pc) > nrow else np.arange(len(pc))
    ps = phis[sel]
    th = iota * ps
    Ac = np.cos(np.outer(th, Mc[:, 0]) - np.outer(nfp * ps, Mc[:, 1]))
    As = np.sin(np.outer(th, Ms[:, 0]) - np.outer(nfp * ps, Ms[:, 1]))
    rc_, *_ = np.linalg.lstsq(Ac, pc[sel, 0], rcond=None)
    zs_, *_ = np.linalg.lstsq(As, pc[sel, 1], rcond=None)
    fit = max(np.max(np.abs(Ac @ rc_ - pc[sel, 0])), np.max(np.abs(As @ zs_ - pc[sel, 1])))
    so = np.sort(rng.choice(len(pc_odd), min(len(pc_odd), nrow), replace=False))
    po = phis[so] + 0.5 * dphi
    to = iota * po
    Rf = np.cos(np.outer(to, Mc[:, 0]) - np.outer(nfp * po, Mc[:, 1])) @ rc_
    Zf = np.sin(np.outer(to, Ms[:, 0]) - np.outer(nfp * po, Ms[:, 1])) @ zs_
    fit = float(max(fit, np.max(np.abs(Rf - pc_odd[so, 0])), np.max(np.abs(Zf - pc_odd[so, 1]))))
    rc = np.zeros((M + 1, 2 * N + 1)); zs = np.zeros((M + 1, 2 * N + 1))
    rc[Mc[:, 0], Mc[:, 1] + N] = rc_
    zs[Ms[:, 0], Ms[:, 1] + N] = zs_
    st = TorusState(rc, zs, float(iota), 0.0, M, N, nfp)
    # theta clockwise in (R, Z) for this orbit's sense of rotation: flip to the solver's counter-clockwise theta
    R_, Z_ = st.geometry_grid(np.linspace(0, 2 * np.pi, 256, endpoint=False), np.array([0.0]))
    area = 0.5 * np.sum(R_[:, 0] * np.roll(Z_[:, 0], -1) - np.roll(R_[:, 0], -1) * Z_[:, 0])
    if area < 0:
        st = flip_theta(st)
    st.info["lsq_fit"] = fit
    return st, fit


def flip_theta(t: TorusState):
    """The same torus with theta -> -theta: cos(m th - n ps) -> cos(m th + n ps), so (m, n) -> (m, -n) for m >= 1
    (the m = 0 row is unchanged), sin -> -sin, iota -> -iota."""
    rc = t.rc.copy(); zs = t.zs.copy()
    rc[1:] = t.rc[1:, ::-1]
    zs[1:] = -t.zs[1:, ::-1]
    out = TorusState(rc, zs, -float(t.iota), t.flux, t.M, t.N, t.nfp, phi0=t.phi0)
    out.info = dict(t.info)
    return out


# ============================================================================ invariant circle by Newton
def circle_newton(tracer: SectionTracer, engine, C0: Circle, flux_target=None, rot_fixed=None, K=None, maxit=12,
                  tol=1e-12, deadline=None, verbose=False, delta=1e-7, log=None, max_grow=3, K_cap=320,
                  seed_stop=1e-7):
    """The invariant circle of the period map P (plane phi = 0) by Gauss-Newton on its symmetric Fourier
    coefficients: P(C(t_j)) = C(t_j + rho) at 2K + 2 collocation phases, with either the flux through C fixed
    (rho free: the circle at the requested flux, no walk) or rho fixed.  Each residual is one period of 2K + 2
    lines (all in one tracer batch); the Jacobian takes the period map's tangent by forward differences (two more
    lines per point) and is reused while the residual falls tenfold per iteration.  A stall whose spectral tail is
    above the residual is the circle's truncation: K grows 1.5x from the current circle (QUASR 0136030: 4.7e-9 at
    K = 32, the torus there needs M = 92).  The 1-D problem integrates the
    toroidal variation exactly, so its size does not grow with the torus's N (QUASR 2039780's torus needs
    N = 623), and no trace has to sweep the circle (next to 3/10, QUASR 142814's line takes ~33,000 periods to cover
    its circle once).  Returns (Circle or None, info)."""
    K = int(K or C0.nm)
    info = dict(K=K, hist=[], free_rotation=free_rot(rot_fixed), flux_target=flux_target, Ks=[])
    C = C0.resized(K)
    rho = float(C0.rot if rot_fixed is None else rot_fixed)
    deadline = _budget.earliest(deadline)
    best = None                     # (honest residual, circle) over the levels
    floors = []
    for grow in range(max_grow + 1):
        info["Ks"].append(K)
        C, rho, lbest, stop = _circle_newton_K(tracer, engine, C, rho, rot_fixed, flux_target, K, maxit, tol,
                                               deadline, verbose, delta, info, None)
        if lbest is None:
            break
        # the residual between the collocation points (held out): a truncated circle satisfies the map at its own
        # 2K + 2 phases far better than between them (QUASR 1335043 at K = 48: 1.2e-7 there, 6.1e-6 between)
        Cl = lbest[1]
        th = (np.arange(2 * K + 2) + 0.5) / (2 * K + 2)
        yh, lh = tracer.period_map(Cl(th), level="short")
        Cl.res_off = float(np.max(np.abs(yh - Cl(np.mod(th + Cl.rot, 1.0))))) if not np.any(lh) else np.inf
        Cl.res_map = max(getattr(Cl, "res_map", np.inf), Cl.res_off)
        info.setdefault("res_off", []).append((K, Cl.res_off))
        if verbose:
            print(f"   circle newton K {K}: held-out residual {Cl.res_off:.2e}", flush=True)
        honest = max(lbest[0], Cl.res_off)
        floors.append(honest)
        if best is None or honest < best[0]:
            best = (honest, Cl)
        # a stall with the spectrum's tail above the residual is the truncation: more modes, from this circle.  A
        # growth that does not lower the floor 3x shows the floor is the tracer's (the integrator is not exactly
        # reversible, so no symmetric circle satisfies its map beyond its error: 4.6e-10 at K = 48 and 72 on QUASR
        # 0136030 with the 1e-11 tracer)
        tail_abs = float(np.max(np.hypot(C.a[-2:], C.b[-2:])))
        # (a seed needs no more: the torus Newton and the refinement take the resolution further -- growing K from
        # 2e-9 / 4.7e-9 / 1.2e-7 cost 60-100 s on QUASR 0164894, 0136030, 1335043 and changed nothing downstream)
        seed_done = (best[1].res_map < seed_stop * max(1.0, abs(best[1].a[0]))
                     and (getattr(best[1], "flux_err", None) is None or best[1].flux_err < 1e-3))
        held_bad = Cl.res_off > 10 * max(lbest[0], 1e-300)
        if stop in ("deadline", "line lost") or grow == max_grow or K >= K_cap or seed_done:
            break
        if (stop == "converged" or tail_abs < 0.1 * lbest[0] or (len(floors) >= 2 and floors[-1] > floors[-2] / 3)) \
                and not held_bad:
            break
        K = int(min(K_cap, max(K + 8, round(1.5 * K))))
        C = C.resized(K)
        if verbose:
            print(f"   circle newton: tail {tail_abs:.1e}, held-out {Cl.res_off:.1e} above the residual {lbest[0]:.1e}:"
                  f" K -> {K}", flush=True)
    if best is None:
        return None, info
    nF, Cb = best
    info.update(K=Cb.nm, res=nF, converged=bool(nF < 1e3 * tol * max(1.0, abs(Cb.a[0]))), rho=Cb.rot,
                tail=Cb.tail(), res_map=float(getattr(Cb, "res_map", nF)) / max(1.0, abs(Cb.a[0])),
                flux_err=getattr(Cb, "flux_err", None))
    Cb.fit_err = nF
    return Cb, info


def free_rot(rot_fixed):
    return rot_fixed is None


def _circle_newton_K(tracer, engine, C, rho, rot_fixed, flux_target, K, maxit, tol, deadline, verbose, delta, info,
                     best):
    free = rot_fixed is None
    nc = 2 * K + 2
    tj = np.arange(nc) / nc
    k = np.arange(K + 1)
    ang = 2 * np.pi * np.outer(tj, k)
    cj, sj = np.cos(ang), np.sin(ang)
    Rsc = float(abs(C.a[0]))
    J = None
    last = None
    stop = "maxit"
    hist0 = len(info["hist"])
    for it in range(maxit):
        if deadline is not None and time.time() > deadline:
            info["stopped"] = stop = "deadline"
            break
        Pj = C(tj)
        need_J = J is None or (last is not None and info["hist"][-1]["nF"] > 0.1 * last)
        if need_J:
            y0 = np.concatenate([Pj, Pj + [delta * Rsc, 0.0], Pj + [0.0, delta * Rsc]])
        else:
            y0 = Pj
        y, lost = tracer.period_map(y0, level="short")
        if np.any(lost[:nc]) or not np.all(np.isfinite(y[:nc])):
            info["stopped"] = stop = "line lost"
            break
        Q = C(np.mod(tj + rho, 1.0))
        F1 = (y[:nc] - Q).ravel()                         # (R_0, Z_0, R_1, Z_1, ...)
        res = float(np.max(np.abs(F1)))
        rows = [F1]
        flux = None
        if free:
            flux = circle_flux(engine, C)
            rows.append([(flux - flux_target) / abs(flux_target) * Rsc])
        F = np.concatenate(rows)
        nF = float(np.max(np.abs(F)))
        info["hist"].append(dict(res=res, nF=nF, rho=rho, flux=flux, jac=bool(need_J), K=K,
                                 flux_err=None if flux is None else abs(flux / flux_target - 1)))
        if verbose:
            print(f"   circle newton K {K} it {it}: res {res:.2e} |F| {nF:.2e} rho {rho:.12f}"
                  + (f" flux {flux:.10e}" if flux is not None else ""), flush=True)
        if best is None or nF < best[0]:
            best = (nF, Circle(C.a.copy(), C.b.copy(), rho))
            best[1].res_map = res
            best[1].flux_err = None if flux is None else abs(flux / flux_target - 1)
        if nF < 1e3 * tol * max(1.0, Rsc):
            # (1e-9 of the radius: the torus Newton from such a circle converges at once; tighter is its job)
            stop = "converged"
            break
        mine = info["hist"][hist0:]
        if len(mine) >= 3 and mine[-2]["jac"] and mine[-3]["jac"] and nF > 0.5 * mine[-2]["nF"] \
                and mine[-2]["nF"] > 0.5 * mine[-3]["nF"]:
            stop = "stalled"                                # two fresh Jacobians without a 2x gain each
            break
        if need_J:
            DPr = (y[nc:2 * nc] - y[:nc]) / (delta * Rsc)    # columns of the tangent map
            DPz = (y[2 * nc:] - y[:nc]) / (delta * Rsc)
            cq = np.cos(2 * np.pi * np.outer(np.mod(tj + rho, 1.0), k))
            sq = np.sin(2 * np.pi * np.outer(np.mod(tj + rho, 1.0), k))
            nun = (K + 1) + K + (1 if free else 0)
            Jm = np.zeros((2 * nc + (1 if free else 0), nun))
            # d/da_k: DP [cos, 0] - [cos(t + rho), 0]
            Jm[0:2 * nc:2, :K + 1] = DPr[:, 0:1] * cj - cq
            Jm[1:2 * nc:2, :K + 1] = DPr[:, 1:2] * cj
            # d/db_k (k >= 1): DP [0, sin] - [0, sin(t + rho)]
            Jm[0:2 * nc:2, K + 1:2 * K + 1] = DPz[:, 0:1] * sj[:, 1:]
            Jm[1:2 * nc:2, K + 1:2 * K + 1] = DPz[:, 1:2] * sj[:, 1:] - sq[:, 1:]
            if free:
                D = C(np.mod(tj + rho, 1.0), 1)
                Jm[0:2 * nc:2, -1] = -D[:, 0]
                Jm[1:2 * nc:2, -1] = -D[:, 1]
                # flux row: d Phi = int B_phi (dR Z' - dZ R') dt (counter-clockwise), scaled as its equation
                tq, P, B, _ = circle_field(engine, C, nq=max(512, 4 * K + 8), A=False)
                Dq = C(tq, 1)
                orient = 1.0 if C.area() > 0 else -1.0
                Bp = B[:, 1]
                cq2 = np.cos(2 * np.pi * np.outer(tq, k))
                sq2 = np.sin(2 * np.pi * np.outer(tq, k))
                dfa = orient * np.mean(Bp[:, None] * cq2 * Dq[:, 1:2], axis=0)
                dfb = -orient * np.mean(Bp[:, None] * sq2[:, 1:] * Dq[:, 0:1], axis=0)
                Jm[-1, :K + 1] = dfa / abs(flux_target) * Rsc
                Jm[-1, K + 1:2 * K + 1] = dfb / abs(flux_target) * Rsc
            J = Jm
        last = nF
        du, *_ = np.linalg.lstsq(J, -F, rcond=None)
        step = float(np.max(np.abs(du[:2 * K + 1])))
        shape = float(max(np.max(np.abs(C.a[1:])), np.max(np.abs(C.b[1:]))))
        lam = 1.0 if step <= 0.3 * shape else 0.3 * shape / step
        a = C.a + lam * du[:K + 1]
        b = C.b.copy()
        b[1:] += lam * du[K + 1:2 * K + 1]
        if free:
            rho += lam * float(du[-1])
        C = Circle(a, b, rho)
    if best is not None and best[1].nm == K:
        C = Circle(best[1].a.copy(), best[1].b.copy(), best[1].rot)
        rho = best[1].rot
    return C, rho, best, stop


# ============================================================================ orbits around the request
def circle_rotation(sec, rot0, digits, nm=64, R=1.0):
    """The rotation of a section orbit by the least misfit of its circle in the rotation phase, searched about a
    rough estimate rot0 (a Birkhoff average with `digits` digits): a scan fine enough for the misfit's V-shaped
    minimum (half-width ~ 1 / (2 pi n) in rotation for n points), then golden section (refine_rotation).  Returns
    (rotation, relative misfit at it, nm).  A regular orbit's points lie on a smooth curve in their phase, so the
    misfit falls to the tracer's level at the true rotation -- whatever the Birkhoff averages' convergence: next to
    1/2 on QUASR 0136030's strongly shaped section the averages had 2 digits at 300 periods, where the misfit fixes
    the rotation; chaotic and island orbits have no small misfit."""
    n = len(sec)
    nm = int(min(nm, n // 12))          # (a strongly shaped section needs its modes, or truncation hides the minimum)
    if nm < 4 or not np.isfinite(rot0):
        return np.nan, np.inf, nm
    width = float(np.clip(3 * 10 ** (-max(digits, 0.0)), 5e-5, 0.02))
    step = 0.25 / (2 * np.pi * n)
    scan = int(np.clip(2 * width / step, 41, 1601))
    r, mis = refine_rotation(sec, rot0, width, nm=nm, max_rows=min(n, 1200), scan=scan)
    return r, mis / max(R, 1e-300), nm


def _orbit_record(R0, sec, dang, lost, engine, nm_flux=64):
    o = dict(R0=float(R0), n=int(len(dang)))
    if lost or len(dang) < 20:
        o.update(kind="lost" if lost else "short", rot=np.nan, digits=np.nan, flux=np.nan)
        return o
    rot, dig = rotation(dang)
    o.update(rot=rot, digits=dig)
    if dig < 10 and np.isfinite(dig) and dig >= 1.0:
        # the circle's misfit fixes the rotation where the averages converge slowly
        Rm = abs(float(np.mean(sec[:, 0])))
        r2, mis, nmc = circle_rotation(sec, rot, dig, R=Rm)
        o.update(rot_fit=r2, misfit=mis)
        # (threshold 1e-5 of R: the tracer's drift over a few hundred periods leaves ~1e-6 on regular orbits --
        # QUASR 0136030, 307 periods at 1e-9 per period: 1.2e-6 with order 6, 1.7e-7 with order 12; chaotic and
        # island orbits misfit by 1e-3 and more; a seed only starts Newton, acceptance is independent.)  The scan's
        # 24 modes can leave a truncation misfit on strongly shaped sections (QUASR 1335043, whose torus needs
        # M ~ 260: every probe read 'chaotic'); the circle with as many modes as the phases allow, its rotation
        # refined again within the scan's minimum, decides.
        if np.isfinite(r2):
            C = fit_circle(sec, r2)
            if C is not None and C.nm > nmc and C.fit_err > 1e-5 * Rm:
                n = len(sec)
                r3, _ = refine_rotation(sec, r2, 2.0 / (2 * np.pi * n), nm=min(C.nm, 160), max_rows=min(n, 2400))
                C3 = fit_circle(sec, r3)
                if C3 is not None and C3.fit_err < C.fit_err:
                    C, r2 = C3, r3
            if C is not None and C.nm >= 8 and C.fit_err < 1e-5 * Rm:
                o.update(rot=r2, digits=float(max(dig, 10.0)), digits_from="circle misfit", fit_err=C.fit_err,
                         flux=abs(circle_flux(engine, C)), kind="regular", circle_modes=C.nm)
                return o
            o["fit_err_full"] = None if C is None else float(C.fit_err)
    if dig < 6:
        o["kind"] = "chaotic"
        o["flux"] = np.nan
        return o
    C = fit_circle(sec, rot, nm=nm_flux)
    if C is None or C.nm < 4:
        o.update(kind="gapped", flux=np.nan)
        return o
    o["fit_err"] = C.fit_err
    o["flux"] = abs(circle_flux(engine, C))
    p, q, d = near_rational(rot, 64)
    if dig >= 10:
        o["kind"] = "island" if d < max(10 * 10 ** (-dig), 1e-11) else "regular"
    else:
        o["kind"] = "undecided"
    return o


def locate(tracer: SectionTracer, engine, axis, base: TorusState, request, deadline=None, nprobe=24, rounds=4,
           nper=None, verbose=False, R_base=None, F_base=None, nper_min=0):
    """Regular orbits around the request on the outboard symmetry line.  The window is centred on the predicted
    position R_ax + (R_b - R_ax) sqrt(request / F_b) (R_b, F_b the base torus's outboard point and flux; the
    flux-radius relation flux ~ (R - R_ax)^2) and narrowed round by round to the bracket of regular orbits around
    the request; orbits short of 10 digits that matter are traced 4 times longer once.  Returns dict(probes,
    below, above, rounds) with probes sorted by R0 (each with kind, rot, digits, flux, and its section points)."""
    Ra, Za = axis.at(np.array([0.0]))
    Rax = float(Ra[0])
    Rb = outboard_point(base)[0] if R_base is None else float(R_base)
    if F_base is not None:
        Fb = float(F_base)
    else:
        Fb = abs(float(base.flux)) if np.isfinite(base.flux) and base.flux != 0 else np.nan
    out = dict(R_axis=Rax, R_base=Rb, F_base=Fb, rounds=[])
    if Rb > Rax and np.isfinite(Fb) and Fb > 0:
        R_est = Rax + (Rb - Rax) * math.sqrt(request / Fb)
        # (a base at the request -- a design torus -- is within ~1e-3 of the radius of the surface: a narrow window
        # first; widened below when it holds no regular orbit on one side)
        w = (0.015 if abs(Fb / request - 1) < 0.05 else 0.06) * (R_est - Rax)
        lo, hi = R_est - w, R_est + w
    else:
        # no usable base: the whole ray from near the axis, coarse
        lo, hi = Rax + 0.05 * abs(Rb - Rax), Rax + 1.3 * abs(Rb - Rax)
        R_est = 0.5 * (lo + hi)
    out["R_est"] = R_est
    rot_g = abs(base.iota / tracer.nfp) if np.isfinite(base.iota) and base.iota != 0 else abs(axis.iota / tracer.nfp)
    if nper is None:
        # ~150 poloidal turns, and 3 sweeps of the circle where a rational p/q (q <= 20) is near: the phases advance
        # q |rot - p/q| per period there and the Birkhoff averages converge on that scale (QUASR 1063007, 5.5e-4 from
        # 1/4 per period: every line read 3.3 digits at 120 periods)
        sweep = max([1.0 / (q * abs(rot_g - round(rot_g * q) / q)) for q in range(1, 21)
                     if abs(rot_g - round(rot_g * q) / q) > 0] + [0.0])
        # (and 16 section points per circle mode the section needs, nper_min: QUASR 1335043's section needs ~48
        # modes for a 1e-5 misfit -- at 330 periods every probe read 'chaotic')
        nper = int(np.clip(max(150.0 / max(rot_g, 1e-3), 3.0 * sweep, nper_min), 300, 4000))
    probes = []
    deadline = _budget.earliest(deadline)
    traces = {}
    a = b = None
    for rnd in range(rounds):
        if deadline is not None and time.time() > deadline:
            break
        R0s = np.linspace(lo, hi, nprobe)
        t0 = time.time()
        o = tracer.orbits(np.stack([R0s, 0 * R0s], -1), nper, deadline=deadline)
        new = []
        for i, R0 in enumerate(R0s):
            rec = _orbit_record(R0, o["sec"][i], o["dang"][i], o["lost"][i], engine)
            traces[rec["R0"]] = (o["sec"][i], o["dang"][i])
            new.append(rec)
        probes = sorted(probes + new, key=lambda r: r["R0"])
        # undecided orbits nearest the request: 4 times longer, once per round
        und = [r for r in new if r["kind"] == "undecided"]
        if und and (deadline is None or time.time() < deadline):
            und = sorted(und, key=lambda r: abs(math.log(max(r["flux"], 1e-300) / request)))[:max(4, nprobe // 4)]
            o2 = tracer.orbits(np.array([[r["R0"], 0.0] for r in und]), min(4 * nper, 8000), deadline=deadline)
            for j, r in enumerate(und):
                rec = _orbit_record(r["R0"], o2["sec"][j], o2["dang"][j], o2["lost"][j], engine)
                if rec["n"] > r["n"] and (rec["kind"] in ("regular", "island") or rec["digits"] > r["digits"]):
                    r.update(rec)
                    traces[r["R0"]] = (o2["sec"][j], o2["dang"][j])
        reg = [r for r in probes if r["kind"] == "regular" and np.isfinite(r["flux"])]
        below = [r for r in reg if r["flux"] < request]
        above = [r for r in reg if r["flux"] >= request]
        a = max(below, key=lambda r: r["R0"]) if below else None
        b = min(above, key=lambda r: r["R0"]) if above else None
        kinds = {}
        for r in new:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        out["rounds"].append(dict(lo=lo, hi=hi, nper=nper, kinds=kinds, wall=time.time() - t0,
                                  below=None if a is None else (a["R0"], a["flux"]),
                                  above=None if b is None else (b["R0"], b["flux"])))
        if verbose:
            print(f"   locate round {rnd}: [{lo:.6f}, {hi:.6f}] {kinds} below {None if a is None else (round(a['R0'], 7), a['flux'])}"
                  f" above {None if b is None else (round(b['R0'], 7), b['flux'])} ({time.time() - t0:.0f}s)", flush=True)
        if a is not None and b is not None and b["R0"] > a["R0"]:
            if (b["flux"] - a["flux"]) <= 2e-3 * request:
                break
            if (lo, hi) == (a["R0"], b["R0"]):
                # the bracket did not narrow: what lies between is not regular (QUASR 0216433's 4/15 chain: chaotic and
                # undecided probes only; a third round repeated the second's window)
                break
            lo, hi = a["R0"], b["R0"]
        elif a is not None:
            w = hi - lo
            lo, hi = a["R0"], a["R0"] + 1.5 * w
        elif b is not None:
            w = hi - lo
            lo, hi = max(Rax + 1e-3 * abs(Rb - Rax), b["R0"] - 1.5 * w), b["R0"]
        else:
            w = hi - lo
            lo, hi = max(Rax + 1e-3 * abs(Rb - Rax), lo - w), hi + w
    out["probes"] = probes
    out["below"] = a
    out["above"] = b
    out["traces"] = traces
    return out


# ============================================================================ walks, refinement, acceptance
def walk(cont, sol: TorusState, flux_signed, deadline=None, bn_tol=1e-5, grow=4, verbose=False, caps=(256, 512)):
    """From a converged torus in s = sqrt(|flux|) to the requested flux (either direction): first step a third
    of the distance, steps growing 1.5x after a success and halved after a failure down to 1e-3 of the distance;
    a step that converges but is not resolved (off-grid B.n above bn_tol: next to a resonance the spectra widen
    as the walk approaches it) is retried at 1.4 M + 2, 1.4 N + 2, up to `grow` times.  Predictor: the last two
    tori extrapolated linearly in s.  Returns (tori, reached, info)."""
    sgn = math.copysign(1.0, flux_signed)
    s0, st = math.sqrt(abs(sol.flux)), math.sqrt(abs(flux_signed))
    tori = [sol]
    s_cur, ds = s0, (st - s0) / 3.0
    M, N = sol.M, sol.N
    info = dict(start=abs(sol.flux), target=abs(flux_signed), steps=0, failures=0, raises=0)
    deadline = _budget.earliest(deadline)
    if abs(st - s0) <= 1e-15 * st:
        return tori, True, info
    while abs(st - s_cur) > 1e-15 * st:
        if deadline is not None and time.time() > deadline:
            info["stopped"] = "deadline"
            break
        s_new = s_cur + ds
        if (s_new - st) * math.copysign(1.0, ds) > 0:
            s_new = st
        last = tori[-1].resized(M, N) if (tori[-1].M, tori[-1].N) != (M, N) else tori[-1]
        if len(tori) >= 2:
            prev = tori[-2].resized(M, N)
            sl, sp = math.sqrt(abs(tori[-1].flux)), math.sqrt(abs(tori[-2].flux))
            a = (s_new - sl) / (sl - sp) if sl != sp else 0.0
            guess = TorusState(last.rc + a * (last.rc - prev.rc), last.zs + a * (last.zs - prev.zs),
                               last.iota + a * (last.iota - prev.iota), sgn * s_new ** 2, M, N, last.nfp)
        else:
            guess = TorusState(last.rc.copy(), last.zs.copy(), last.iota, sgn * s_new ** 2, M, N, last.nfp)
        new = cont.solve(guess, sgn * s_new ** 2, M, N, maxit=12)
        ok = new.converged and np.isfinite(new.bn_off)
        if verbose:
            print(f"   walk s {s_new / st:.8f} M,N {M},{N}: conv {new.converged} bn {new.bn_off:.2e} iota {new.iota:.10f}",
                  flush=True)
        if ok and new.bn_off < bn_tol:
            tori.append(new)
            s_cur = s_new
            info["steps"] += 1
            ds *= 1.5
        elif ok and grow > 0 and (M < caps[0] or N < caps[1]):
            M, N = min(caps[0], int(1.4 * M) + 2), min(caps[1], int(1.4 * N) + 2)
            grow -= 1
            info["raises"] += 1
        else:
            info["failures"] += 1
            ds *= 0.5
            if abs(ds) < 1e-3 * abs(st - s0):
                break
    info["reached"] = abs(tori[-1].flux)
    info["resolution"] = [M, N]
    return tori, abs(st - math.sqrt(abs(tori[-1].flux))) <= 1e-15 * st, info


def linear_shear(cont, sol: TorusState, max_unknowns=6000):
    """d iota / d flux of the torus family at a converged torus from one linear solve with the torus problem's exact
    Jacobian (the flux equation's row is (flux - target) * flux_scale: J du = flux_scale d target e_last), instead of
    two more Newton solves (QUASR 0142814 next to 3/10: 89 s for the acceptance with the two solves).  NaN where the
    system is too large for the dense solve."""
    from .torus import dense_solve
    prob = cont.problem(sol.M, sol.N)
    b = prob.v.basis if prob.np_path else prob.basis
    n = b.nc + b.ns + 1
    if n > max_unknowns:
        return np.nan
    u = np.asarray(b.pack(sol.rc, sol.zs, sol.iota))
    R00 = float(sol.rc[0, sol.N])
    fs = R00 / max(abs(float(sol.flux)), 1e-300)
    F, J, flux, res = prob.system(u, float(sol.flux), fs)
    e = np.zeros(n)
    e[-1] = 1.0
    du = np.asarray(dense_solve(J, e))
    return float(fs * du[-1])


def shear_estimate(tori, request):
    """d iota / d flux at the request from tori (TorusState) near it: the two nearest on either side, else the
    two nearest."""
    pts = sorted({(abs(float(t.flux)), float(t.iota)) for t in tori if t.converged and np.isfinite(t.iota)})
    if len(pts) < 2:
        return np.nan
    f = np.array([p[0] for p in pts]); io = np.array([p[1] for p in pts])
    j = np.argsort(np.abs(f - request))[:2]
    if f[j[0]] == f[j[1]]:
        return np.nan
    return float((io[j[1]] - io[j[0]]) / (f[j[1]] - f[j[0]]))


def accept(engine, source, axis, best: TorusState, request, shear_flux, refine_engine=None, verify=True,
           verbose=False):
    """Every check a returned torus must pass: off-grid B.n <= torux.analysis.BN_SURFACE (1e-14) at the request (flux
    to 1e-12 relative), the
    resonance guard (torux.guard.resonance_guard, shear d iota / d Phi' = 2 pi d iota / d flux), the quadrature
    estimate <= 1e-8, sections simple and enclosing the axis (torux.analysis.torus_topology), and the independent
    reference (torux.verify.verify_torus + torux.verify.acceptance: reference B.n at the arithmetic limit, flux on 6
    sections to 1e-12, the double-double tier within 1e-14).  Returns (ok, record)."""
    from .guard import resonance_guard
    from .torus import TorusProblem
    from .analysis import torus_topology
    eng = refine_engine or engine
    t0 = time.time()
    rec = dict(M=int(best.M), N=int(best.N), bn_off=float(best.bn_off), iota=float(best.iota), flux=float(best.flux),
               timings={})
    try:
        from .analysis import BN_SURFACE
    except ImportError:              # (older torux.analysis: the same bar)
        BN_SURFACE = 1e-14
    rec["flux_ok"] = bool(abs(abs(best.flux) - request) <= 1e-12 * request)
    # the solver's surface bar (torux.analysis.BN_SURFACE, 1e-14): tori that plateau above it next to rationals are
    # near-invariant tori in thin chains, not surfaces
    rec["bn_bar"] = float(BN_SURFACE)
    rec["bn_ok"] = bool(np.isfinite(best.bn_off) and best.bn_off <= BN_SURFACE)
    g = resonance_guard(eng, best, shear=shear_flux * 2 * np.pi if np.isfinite(shear_flux) else np.nan)
    rec["guard"] = dict(ok=bool(g["ok"]), h=float(g["h"]), x_over_w=float(g["x_over_w"]), worst=g["worst"],
                        n_checked=len(g["checked"]), shear_flux=float(shear_flux))
    rec["timings"]["guard"] = time.time() - t0
    rec["quad_err_half"] = float(TorusProblem(eng, best.nfp, best.M, best.N).quadrature_check(best.rc, best.zs))
    topo = torus_topology(best, axis)
    rec["timings"]["quad_topology"] = time.time() - t0
    rec["topology"] = topo
    ok = rec["flux_ok"] and rec["bn_ok"] and g["ok"] and rec["quad_err_half"] <= 1e-8 and topo["ok"]
    if ok and verify:
        from .verify import verify_torus, acceptance
        v = verify_torus(source, best, nlines=0)
        v["ok"] = acceptance(v)
        rec["verify"] = v
        ok = ok and v["ok"]
        rec["timings"]["verify"] = time.time() - t0
    rec["ok"] = bool(ok)
    if verbose:
        vv = rec.get("verify") or {}
        print(f"   accept: M,N {best.M},{best.N} bn {best.bn_off:.2e} flux_ok {rec['flux_ok']} guard {g['ok']} "
              f"(h {g['h']:.2e}) quad {rec['quad_err_half']:.1e} topo {topo['ok']} ref bn {vv.get('bn_max', np.nan):.2e} "
              f"ref flux {vv.get('flux_rel_err', np.nan):.1e} -> {ok}", flush=True)
    return ok, rec


class _Ctx:
    def __init__(self, engine, source, axis, request, deadline, verbose, tol, caps, dmin):
        from .family import Continuation
        self.engine, self.source, self.axis = engine, source, axis
        self.request, self.deadline, self.verbose, self.tol = float(request), deadline, verbose, tol
        self.caps = caps
        self.dmin = dmin
        self.nfp = int(source.nfp)
        self.cont = Continuation(engine, self.nfp, Mmax=caps[0], Nmax=caps[1])
        self.tracer = None
        self.sgn = None
        self.family = []          # converged tori (for the shear)
        self.t0 = time.time()

    def left(self):
        return np.inf if self.deadline is None else self.deadline - time.time()

    def say(self, msg):
        if self.verbose:
            print(f"  [seeds {time.time() - self.t0:6.1f}s] {msg}", flush=True)


def _torus_resolution(ctx, C, rot, M, N, tail_tol=1e-7, tries=3):
    """torus_from_circle at (M, N), N raised while the toroidal tail exceeds tail_tol."""
    from .family import ladder_up
    st = None
    for _ in range(tries):
        st = torus_from_circle(ctx.tracer, C, rot, M, N)
        if st is None:
            return None
        tm, tn = st.tails()
        if tn <= tail_tol or N >= ctx.caps[1]:
            break
        N = ladder_up(N, 1.5, cap=ctx.caps[1])
    return st


def _finish(ctx, sol, how, log, polish=True):
    """Refinement to the rounding floor on a lean engine, shear, acceptance; the circle polish where the refinement
    plateaus.  Returns the accepted torus or None."""
    from .family import refine_to_limit
    from .analysis import lean_continuation
    t0 = time.time()
    rc_ = lean_continuation(ctx.cont, ctx.source, sol, ctx.dmin)
    best = refine_to_limit(rc_, sol, sol.flux, target=ctx.tol, deadline=ctx.deadline, verbose=ctx.verbose,
                           Mcap=ctx.caps[0], Ncap=ctx.caps[1])
    ent = dict(how=how, refine_hist=[(int(m), int(n), float(b)) for m, n, b in best.info.get("refine_hist", [])],
               refine_stop=best.info.get("refine_stop"), refine_wall=time.time() - t0, M=best.M, N=best.N,
               bn_off=float(best.bn_off), iota=float(best.iota), refine_nodes=rc_.engine.info.n_nodes)
    log["attempts"].append(ent)
    ctx.say(f"{how}: refined to M,N {best.M},{best.N} bn {best.bn_off:.2e} iota {best.iota:.13f} "
            f"({ent['refine_stop']}, {time.time() - t0:.0f}s)")
    t1 = time.time()
    shear = ctx.shear(best)
    ent["shear_wall"] = time.time() - t1
    ok, rec = accept(ctx.engine, ctx.source, ctx.axis, best, ctx.request, shear, refine_engine=rc_.engine,
                     verbose=ctx.verbose)
    ent["accept"] = rec
    if ok:
        return best
    if polish and not rec["bn_ok"] and np.isfinite(best.bn_off) and best.bn_off < 1e-5 and ctx.left() > 60:
        # the circle polish: the invariant circle through the plateaued torus's section, at the request's flux
        pol = circle_polish(ctx, best, log)
        if pol is not None:
            return pol
    return None


def circle_polish(ctx, torus, log, K=None):
    """A refinement that plateaued next to a low-order rational (the resonant direction held where the start put
    it: QUASR 142814 at B.n 3.6e-8 from M = 28 to 384) made again from the invariant circle at the
    request's flux: Newton on the circle (section of the plateaued torus as the start), the torus carried from it
    over a field period, Newton at the request, the refinement and acceptance."""
    t0 = time.time()
    K = K or int(min(256, max(48, 1.5 * torus.M)))
    C0 = section_circle(torus, nm=K)
    ent = dict(how="circle polish", K=K)
    log["attempts"].append(ent)
    C, info = circle_newton(ctx.tracer, ctx.engine, C0, flux_target=ctx.sgn * ctx.request, K=K, deadline=ctx.deadline,
                            verbose=ctx.verbose)
    ent["circle"] = {k: v for k, v in info.items() if k != "hist"}
    ent["circle_hist"] = [h["nF"] for h in info["hist"]]
    ctx.say(f"circle polish: K {K} conv {info.get('converged')} res {info.get('res', np.nan):.2e} rho {info.get('rho', np.nan):.13f}"
            f" ({time.time() - t0:.0f}s)")
    if C is None or not _seed_ok(info):
        C = traced_circle(ctx, torus, ent)
        if C is None:
            return None
    sol = _torus_from_circle_solve(ctx, C, C.rot, max(torus.M, C.nm, 24), max(torus.N, 24), "circle polish", log)
    if sol is None:
        return None
    return _finish(ctx, sol, "circle polish", log, polish=False)


def traced_circle(ctx, torus, ent, max_periods=120000):
    """The circle polish: the line through the torus's outboard symmetric point traced
    long enough to sweep its invariant circle about 1.6 times (next to p/q its phases advance q |rot - p/q| per
    period; at most max_periods and what half the remaining budget allows at the measured rate), its rotation refined
    to the circle's least misfit about the trace's own Birkhoff average, the circle fitted.  Returns Circle or None.
    (The fallback of the circle Newton: one line cannot be batched, so this is the slow path.)"""
    Rb, _ = outboard_point(torus)
    rot0 = torus.iota / ctx.nfp
    need = max(6000, min(max_periods, near_resonance_need(abs(rot0))))
    t0 = time.time()
    probe = ctx.tracer.orbits(np.array([[Rb, 0.0]]), 50, deadline=ctx.deadline)
    rate = (time.time() - t0) / max(probe["n"], 1)
    afford = int(0.5 * ctx.left() / max(rate, 1e-9)) if np.isfinite(ctx.left()) else need
    nper = int(min(need, afford))
    ent["traced"] = dict(R0=Rb, need=need, periods=nper, rate=rate)
    if nper < 2000:
        return None
    o = ctx.tracer.orbits(np.array([[Rb, 0.0]]), nper, deadline=ctx.deadline)
    if o["lost"][0] or o["n"] < 1000:
        return None
    rot, dig = rotation(o["dang"][0])
    if not np.isfinite(rot) or abs(rot - rot0) > 1e-3:
        return None
    rot, mis = refine_rotation(o["sec"][0], rot, max(1e-6, 10 * 10 ** (-dig)), nm=64)
    C = fit_circle(o["sec"][0], rot)
    ent["traced"].update(rot=rot, digits=dig, misfit=mis, nm=None if C is None else C.nm,
                         fit=None if C is None else C.fit_err, wall=time.time() - t0)
    if C is None or C.nm < 8:
        return None
    C.rot = rot
    return C


def _torus_from_circle_solve(ctx, C, rot, M, N, how, log, flux_signed=None, fixed_iota=None):
    """Torus from the circle at (M, N) (N raised to its toroidal tail), Newton at the given flux (default the
    request) or at a fixed transform.  Returns the converged torus or None."""
    t0 = time.time()
    st = _torus_resolution(ctx, C, rot, M, N)
    ent = dict(how=how + ": torus from circle", M=M, N=None if st is None else st.N)
    log["attempts"].append(ent)
    if st is None:
        ent["fail"] = "line lost"
        return None
    flux_signed = ctx.sgn * ctx.request if flux_signed is None else flux_signed
    st.flux = flux_signed if fixed_iota is None else ctx.cont.problem(st.M, st.N).flux_of(st.rc, st.zs)
    ent["seed_flux"] = float(ctx.cont.problem(st.M, st.N).flux_of(st.rc, st.zs))
    ent["seed_bn"] = float(ctx.cont.problem(st.M, st.N).check(st.rc, st.zs, st.iota)[1])
    sol = ctx.cont.solve(st, st.flux, st.M, st.N, maxit=15, fixed_iota=fixed_iota)
    ent.update(conv=bool(sol.converged), bn=float(sol.bn_off), iota=float(sol.iota), its=int(sol.its),
               wall=time.time() - t0, N=st.N)
    ctx.say(f"{how}: torus {st.M},{st.N} (seed flux {ent['seed_flux']:.8e}, seed bn {ent['seed_bn']:.1e}): conv "
            f"{sol.converged} bn {sol.bn_off:.2e} iota {sol.iota:.12f} its {sol.its} ({time.time() - t0:.0f}s)")
    if not sol.converged or not np.isfinite(sol.bn_off):
        return None
    if sol.bn_off > 1e-5:
        sol = ctx.cont.raise_until(sol, sol.flux, ctx.cont.bn_cont, max_raise=3, fixed_iota=fixed_iota)
        ent["raised"] = dict(M=sol.M, N=sol.N, bn=float(sol.bn_off))
    ctx.family.append(sol)
    return sol


SEED_RES = 1e-6      # a circle whose period-map residual is below this (relative to R) seeds the torus Newton
                     # (Newton converges from a circle fitted to 2.4e-6 m: QUASR 142814)


def _seed_ok(info):
    """A circle seeds the torus when its period-map residual is below SEED_RES (relative) and its flux within 1e-3
    of the target: the torus Newton at the request does the rest.  (Next to a chain the circle's flux couples to its
    resonant harmonic and the flux row converges slowly: QUASR 142814's circle at 3e-6 from 3/10 reached a map
    residual of 1.3e-8 with the flux still 7e-7 off.)"""
    if info.get("converged"):
        return True
    fe = info.get("flux_err")
    return bool(info.get("res_map", np.inf) < SEED_RES and (fe is None or fe < 1e-3))


def _plausible(base: TorusState, axis):
    """A base torus whose sections are simple closed curves around the axis, with a finite flux and transform."""
    from .analysis import torus_topology
    try:
        if not (np.all(np.isfinite(base.rc)) and np.all(np.isfinite(base.zs)) and np.isfinite(base.iota)):
            return False
        return bool(torus_topology(base, axis)["ok"])
    except Exception:  # noqa: BLE001
        return False


def _base_K(base: TorusState, tol=1e-9, cap=192):
    """Modes the base torus's section carries above tol (relative), on the ladder."""
    from .family import LADDER
    C = section_circle(base, nm=min(base.M + 8, 512), nq=4 * base.M + 64)
    amp = np.hypot(C.a, C.b)
    sc = max(float(np.max(amp[1:])), 1e-300)
    k = int(np.max(np.flatnonzero(amp / sc > tol), initial=8)) + 4
    return int(min(cap, next((L for L in LADDER if L >= max(24, k)), LADDER[-1])))


def seed_and_walk(engine, source, request, base: TorusState, axis, deadline=None, verbose=False, tol=3e-15,
                  caps=(384, 768), dmin=None, base_polish=True, trace=True, max_orbits=3, nprobe=24):
    """The torus at |flux| = request where Newton from the design surface and the continuation fail.

    engine: torux.field.FieldEngine; source: its Source (stellarator-symmetric); request: |flux| (Wb); base: a
    TorusState near the request (the failed design torus, or a continuation's last torus); axis: torux.axis.Axis.
    Order:
    A. the base polish: the invariant circle at the requested flux by Newton from the base's section (no
       tracing), the torus from it, Newton at the request, refinement, acceptance (skipped for a base whose
       sections are not simple curves around the axis);
    B. tracing: regular orbits located around the request; for the nearest in flux (at most max_orbits) their
       circles (rotation refined) -> (i) Newton on the circle at the requested flux -> torus -> Newton ->
       refinement; (ii) the torus from the circle at the orbit's own flux (fixed transform) walked to the request
       with resolution growth -> refinement; (iii) the least-squares seed where the circle has too few modes.
    Every refinement that plateaus gets the circle polish.  Returns (TorusState accepted at the request or None,
    log); a torus is returned only if it passed accept() (off-grid B.n < 1e-12, guard, quadrature, topology,
    flux, and the independent reference check)."""
    t_start = time.time()
    deadline = _budget.earliest(deadline)
    log = dict(request=float(request), attempts=[], timings={}, outcome="none")
    if not getattr(source, "stellsym", True):
        log["outcome"] = "unsupported (no stellarator symmetry)"
        return None, log
    dmin = dmin if dmin is not None else getattr(engine, "dmin", None) or source.meta.get("dmin", 0.05)
    ctx = _Ctx(engine, source, axis, request, deadline, verbose, tol, caps, dmin)
    ctx.sgn = bphi_sign(engine, axis)
    Ra, _ = axis.at(np.array([0.0]))
    Rax = float(Ra[0])
    plausible = _plausible(base, axis)
    log["base"] = dict(plausible=plausible, M=int(base.M), N=int(base.N), iota=float(base.iota),
                       flux=float(base.flux), bn_off=float(base.bn_off))
    Rb, _ = outboard_point(base)
    if not plausible:
        # no usable shape.  Its outboard symmetric point can still be right (a design surface whose angle map did
        # not converge is exact at its symmetric points: QUASR 0136030's broken guess has R = 0.90377 there, the
        # surface 0.90378); used if it lies within a few minor radii estimated from the request and the axis field
        # (flux ~ pi a^2 |B_phi|), else that estimate
        Bax = np.asarray(engine.evaluate(np.array([[Rax, 0.0, 0.0]]), grad=False, A=False)["B"])[0]
        a_est = math.sqrt(request / (math.pi * max(abs(Bax[1]), 1e-300)))
        Fb = abs(float(base.flux)) if np.isfinite(base.flux) else np.nan
        if not (np.isfinite(Rb) and Rax < Rb < Rax + 3 * a_est and np.isfinite(Fb) and Fb > 0):
            Rb, Fb = Rax + a_est, np.nan
        log["base"]["outboard_used"] = dict(R=Rb, F=Fb, a_est=a_est)
        base = TorusState(base.rc, base.zs, base.iota if np.isfinite(base.iota) else axis.iota, Fb, base.M,
                          base.N, base.nfp)
    t0 = time.time()
    teng = trace_engine(engine, source)
    tr = SectionTracer(teng, ctx.nfp, axis, sign=ctx.sgn)
    # (the lines that matter start within a few % of the base's outboard point: 1.1x reached towards the coils of
    # QUASR 2039780, 0.15 m from a surface of minor radius 0.125 m, and asked for 128 steps per period)
    tr.calibrate(np.stack([Rax + (Rb - Rax) * np.array([0.5, 0.8, 1.0, 1.03]), np.zeros(4)], -1))
    ctx.tracer = tr
    log["tracer"] = dict(nodes=int(teng.info.n_nodes), engine_nodes=int(engine.info.n_nodes),
                         fastB=None if tr.fastB is None else tr.fastB.check_err,
                         sp={f"{k}/{lev}": v for (k, lev), v in tr.sp.items()},
                         calib={str(k): v for k, v in tr.calib.items()}, wall=time.time() - t0)
    ctx.say(f"tracer: {log['tracer']['sp']} ({time.time() - t0:.0f}s)")
    ctx.orbit_pts = []

    def shear(best):
        pts = list(ctx.family) + [TorusState(np.zeros((1, 1)), np.zeros((1, 1)), p[1], p[0], 0, 0, ctx.nfp,
                                             converged=True) for p in ctx.orbit_pts]
        near = [t for t in pts if abs(abs(t.flux) / request - 1) < 0.05]
        sh = shear_estimate(near, request) if len(near) >= 2 else np.nan
        how = "family"
        s0 = ctx.start if getattr(ctx, "start", None) is not None else best
        if not np.isfinite(sh):
            # one linear solve with the start torus's exact Jacobian
            try:
                sh = linear_shear(ctx.cont, s0) * math.copysign(1.0, s0.flux)
                how = "linear"
            except Exception:  # noqa: BLE001
                sh = np.nan
        if not np.isfinite(sh):
            # (too large for the dense solve) one more torus at the start resolution 1e-4 in flux from it (the other
            # side only if that one fails), Newton from the converged start torus
            fam = [s0]
            for f in (1e-4, -1e-4):
                g = ctx.cont.solve(s0, s0.flux * (1 + f), s0.M, s0.N, maxit=8)
                if g.converged:
                    fam.append(g)
                    break
            sh = shear_estimate(fam, request)
            how = "one solve"
        log["shear_flux"] = sh
        log["shear_how"] = how
        return sh
    ctx.shear = shear

    def done(best, how):
        log["outcome"] = "surface"
        log["how"] = how
        log["timings"]["total"] = time.time() - t_start
        log["surface"] = dict(M=int(best.M), N=int(best.N), bn_off=float(best.bn_off), iota=float(best.iota),
                              flux=float(best.flux))
        ctx.say(f"accepted ({how}) in {time.time() - t_start:.0f}s")
        return best, log

    # ---------------------------------------------------------------- A0: the base refined further
    # (a base that is already a converged torus at the request, B.n < 1e-5, whose refinement stopped -- e.g. under
    # torux.analysis's caps M, N <= 320, where QUASR 2039780 needs N ~ 623 and 2093639 N = 768: the refinement with this
    # module's caps first, without the circle polish; the circle comes next if it plateaus)
    if (base_polish and plausible and np.isfinite(base.bn_off) and base.bn_off < 1e-5 and np.isfinite(base.flux)
            and abs(abs(base.flux) / request - 1) <= 1e-12 and ctx.left() > 120):
        t0 = time.time()
        b0 = TorusState(np.asarray(base.rc), np.asarray(base.zs), float(base.iota), ctx.sgn * request, base.M, base.N,
                        base.nfp, bn_off=float(base.bn_off), converged=True)
        ctx.start = b0
        best = _finish(ctx, b0, "base refine", log, polish=False)
        log["timings"]["base_refine"] = time.time() - t0
        if best is not None:
            return done(best, "base refine")
    # ---------------------------------------------------------------- A: base polish
    if base_polish and plausible and ctx.left() > 60:
        t0 = time.time()
        K = max(_base_K(base), 32)
        C0 = section_circle(base, nm=K)
        C, info = circle_newton(tr, engine, C0, flux_target=ctx.sgn * request, K=K, deadline=deadline,
                                verbose=verbose)
        log["attempts"].append(dict(how="base polish: circle", K0=K, K=info.get("K"), conv=info.get("converged"),
                                    res=info.get("res"), rho=info.get("rho"), hist=[h["nF"] for h in info["hist"]],
                                    wall=time.time() - t0))
        ctx.say(f"base polish: circle K {info.get('K')} conv {info.get('converged')} res {info.get('res', np.nan):.2e} "
                f"rho {info.get('rho', np.nan):.13f} ({time.time() - t0:.0f}s)")
        if C is not None and _seed_ok(info):
            sol = _torus_from_circle_solve(ctx, C, C.rot, max(C.nm, 24), max(base.N, 24), "base polish", log)
            if sol is not None:
                ctx.start = sol
                best = _finish(ctx, sol, "base polish", log)
                if best is not None:
                    return done(best, "base polish")
        log["timings"]["base_polish"] = time.time() - t0
    if not trace or ctx.left() < 60:
        log["timings"]["total"] = time.time() - t_start
        return None, log

    # ---------------------------------------------------------------- B: tracing
    t0 = time.time()
    try:
        k_need = _base_K(base, tol=1e-6, cap=128) if plausible else 48
    except Exception:  # noqa: BLE001
        k_need = 48
    log["locate_k_need"] = k_need
    loc = locate(tr, engine, axis, base, request, deadline=deadline, nprobe=nprobe, verbose=verbose, R_base=Rb,
                 F_base=abs(float(base.flux)) if np.isfinite(base.flux) else np.nan, nper_min=16 * k_need)
    log["locate"] = {k: v for k, v in loc.items() if k not in ("traces",)}
    log["timings"]["locate"] = time.time() - t0
    reg = [r for r in loc["probes"] if r["kind"] == "regular" and np.isfinite(r["flux"])]
    ctx.orbit_pts = [(r["flux"], r["rot"] * ctx.nfp) for r in reg]
    # nearest in flux first, alternating sides
    below = sorted([r for r in reg if r["flux"] < request], key=lambda r: request - r["flux"])
    above = sorted([r for r in reg if r["flux"] >= request], key=lambda r: r["flux"] - request)
    cands = []
    for i in range(max(len(below), len(above))):
        for side in (below, above):
            if i < len(side):
                cands.append(side[i])
    cands = sorted(cands[:2 * max_orbits], key=lambda r: abs(math.log(r["flux"] / request)))[:max_orbits]
    if not cands:
        # no orbit certified regular within the tracing (next to a low-order rational the averages converge slowly):
        # the best-converged undecided ones nearest the request (acceptance decides, not this)
        und = [r for r in loc["probes"] if r["kind"] == "undecided" and np.isfinite(r["flux"]) and r["digits"] >= 6]
        cands = sorted(und, key=lambda r: abs(math.log(r["flux"] / request)))[:max_orbits]
        log["undecided_candidates"] = len(cands)
    for o in cands:
        if ctx.left() < 60:
            break
        sec, dang = loc["traces"][o["R0"]]
        t1 = time.time()
        rot = o["rot"]
        C = fit_circle(sec, rot)
        ent = dict(how="orbit", R0=o["R0"], rot=rot, digits=o["digits"], flux=o["flux"],
                   nm=None if C is None else C.nm, fit=None if C is None else C.fit_err)
        log["attempts"].append(ent)
        if C is not None and len(sec) > 1500 and o["digits"] < 13:
            rot, mis = refine_rotation(sec, rot, max(10 * 10 ** (-o["digits"]), 1e-12), nm=min(C.nm, 64))
            C = fit_circle(sec, rot)
            ent.update(rot_refined=rot, fit_refined=None if C is None else C.fit_err)
        ctx.say(f"orbit R0 {o['R0']:.7f} flux {o['flux']:.8e} rot {rot:.12f} digits {o['digits']:.1f}: circle "
                f"{None if C is None else C.nm} modes, fit {None if C is None else C.fit_err:.1e}")
        if C is not None and C.nm >= 8:
            C.rot = rot
            # (i) Newton on the circle at the requested flux
            Ci, info = circle_newton(tr, engine, C, flux_target=ctx.sgn * request, K=C.nm, deadline=deadline,
                                     verbose=verbose)
            ent["circle_newton"] = dict(conv=info.get("converged"), res=info.get("res"), rho=info.get("rho"),
                                        hist=[h["nF"] for h in info["hist"]])
            if Ci is not None and _seed_ok(info):
                sol = _torus_from_circle_solve(ctx, Ci, Ci.rot, max(Ci.nm, 24), max(base.N, 24), "orbit circle", log)
                if sol is not None:
                    ctx.start = sol
                    best = _finish(ctx, sol, "orbit circle -> request", log)
                    if best is not None:
                        return done(best, "orbit circle -> request")
            # (ii) the torus at the orbit's own flux (its transform fixed), walked to the request
            if ctx.left() > 60:
                sol = _torus_from_circle_solve(ctx, C, rot, max(C.nm, 24), max(base.N, 24), "orbit torus", log,
                                               fixed_iota=rot * ctx.nfp)
                if sol is not None:
                    best = _walk_and_finish(ctx, sol, "orbit walk", log)
                    if best is not None:
                        return done(best, "orbit walk")
        elif ctx.left() > 60:
            # (iii) least squares in two angles (a few modes)
            st, fit = lsq_seed(tr, o["R0"], rot, 16, 16, deadline=deadline)
            ent["lsq_fit"] = fit
            if st is not None:
                st.flux = ctx.cont.problem(16, 16).flux_of(st.rc, st.zs)
                sol = ctx.cont.solve(st, st.flux, 16, 16, maxit=15)
                ent["lsq_newton"] = dict(conv=bool(sol.converged), bn=float(sol.bn_off))
                if sol.converged:
                    best = _walk_and_finish(ctx, sol, "lsq walk", log)
                    if best is not None:
                        return done(best, "lsq walk")
        ent["wall"] = time.time() - t1
    log["timings"]["total"] = time.time() - t_start
    return None, log


def _walk_and_finish(ctx, sol, how, log):
    t0 = time.time()
    tori, reached, info = walk(ctx.cont, sol, ctx.sgn * ctx.request, deadline=ctx.deadline, verbose=ctx.verbose,
                               caps=(min(ctx.caps[0], 256), min(ctx.caps[1], 512)))
    info["wall"] = time.time() - t0
    log["attempts"].append(dict(how=how + ": walk", **info))
    ctx.family.extend(tori)
    ctx.say(f"{how}: walk {info['steps']} steps, {info['raises']} raises, reached {info['reached']:.10e} "
            f"({'reached' if reached else 'short'}, {time.time() - t0:.0f}s)")
    if not reached:
        return None
    ctx.start = tori[-1]
    return _finish(ctx, tori[-1], how, log)


def surface_record(T: TorusState, log):
    """(surface record, guard record) for an accepted seed_and_walk torus, in the form torux.analysis records a
    verified surface (torus_record(full=True) + quad_err_half + topology; the reference check under 'reference')."""
    from .analysis import torus_record
    acc = next((a["accept"] for a in reversed(log.get("attempts", [])) if (a.get("accept") or {}).get("ok")), None)
    rec = torus_record(T, full=True)
    if acc is not None:
        rec.update(quad_err_half=acc["quad_err_half"], topology=acc["topology"], reference=acc.get("verify"))
    rec["refine_hist"] = T.info.get("refine_hist")
    return rec, (None if acc is None else acc["guard"])
