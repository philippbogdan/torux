"""Precise tier of the independent reference field evaluator.

    ref = Reference(source)              # source: torux.source.Source, a list of parts, or one part
    r = ref.evaluate(points, grad=True)  # points (N, 3) Cartesian, m
    # or: r = evaluate(source, points)

``r`` is a dict of numpy arrays:
    B (N,3) [T], A (N,3) [T m], dB (N,3,3) [T/m] with dB[n, i, j] = dB_i/dx_j  (float64, rounded)
    B_lo, A_lo, dB_lo   residuals: B + B_lo equals the double-double result (~1e-30 relative),
                        so errors of another engine can be measured without the reference's
                        own final rounding
    err_B, err_A, err_dB (N,)  absolute error estimate of the double-double result (max over
                        components; quadrature a-posteriori estimate + rounding allowance).
                        The float64 outputs add at most 0.5 ulp per component.
    scale_B, scale_A, scale_dB (N,)  sum of the absolute values of everything summed (integrand
                        L1 norms for curves and polygons, per-element norms for closed forms):
                        the cancellation-aware magnitude.  err/|B| and err/scale_B are the two
                        relative accuracies.
    dmin (N,)           distance to the nearest current element / node seen (diagnostic)
    flag (N,)           1 if a refinement limit was reached (point on or extremely close to a
                        filament): value not certified
    work (N,)           number of curve panels evaluated (cost diagnostic)

Vector potential: A of filaments is mu0 I/(4 pi) int dl/|x - r| (Coulomb gauge for closed
curves; for open Segments the plain sum of segment potentials), dipoles mu0/(4 pi) m x r/r^3,
rings and axisymmetric polygons A_phi (elliptic-integral potential), ToroidalField
A = -RB ln(R / 1 m) z_hat.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import dd
from .precise_closed import dipoles_dd, rings_dd, segments_dd, toroidal_dd
from .precise_curves import FourierPrepared


def _parts_of(source):
    if hasattr(source, "parts"):
        return list(source.parts)
    if isinstance(source, (list, tuple)):
        return list(source)
    return [source]


@njit(cache=True, error_model="numpy")
def _dd_add_arrays(x, y):
    out = np.empty_like(x)
    xf = x.ravel()
    yf = y.ravel()
    of = out.ravel()
    for i in range(xf.shape[0]):
        of[i] = dd.add(xf[i], yf[i])
    return out


@njit(cache=True, error_model="numpy")
def _split_round(x):
    """dd array -> (rounded float64, residual) with rounded + residual = x (to ~1e-32 rel)."""
    xf = x.ravel()
    hi = np.empty(xf.shape[0])
    lo = np.empty(xf.shape[0])
    for i in range(xf.shape[0]):
        s, e = dd.two_sum(xf[i].real, xf[i].imag)
        hi[i] = s
        lo[i] = e
    return hi, lo


class Reference:
    """Precise reference evaluator bound to one source (caches curve node values)."""

    def __init__(self, source, n_gl=32, Lc=4):
        self.parts = _parts_of(source)
        self._prep = []
        for p in self.parts:
            k = p.kind
            if k == "fourier":
                self._prep.append(FourierPrepared(p, n_gl=n_gl, Lc=Lc))
            elif k == "polyline":
                s = p.as_segments()
                self._prep.append(s)
            elif k == "axipoly":
                from .precise_axipoly import AxipolyPrepared
                self._prep.append(AxipolyPrepared(p))
            else:
                self._prep.append(p)

    def evaluate(self, points, grad=True, rtol=1e-20, eta=1.0, maxlev=48):
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        N = pts.shape[0]
        tot = np.zeros((N, 15), dtype=np.complex128)
        err = np.zeros((N, 3))
        scale = np.zeros((N, 3))
        dmin = np.full(N, np.inf)
        flag = np.zeros(N, dtype=np.int64)
        work = np.zeros(N)
        for part, prep in zip(self.parts, self._prep):
            k = part.kind
            fl = None
            wk = None
            dm = None
            if k == "fourier":
                out, e, s, dm, fl, wk = prep.evaluate(pts, rtol=rtol, eta=eta, maxlev=maxlev, grad=grad)
            elif k in ("polyline", "segments"):
                seg = prep
                out, e, s, dm = segments_dd(pts, seg.a, seg.b, seg.current, grad)
            elif k == "dipoles":
                out, e, s, dm = dipoles_dd(pts, part.pos, part.moment, grad)
            elif k == "rings":
                out, e, s, dm = rings_dd(pts, part.R, part.Z, part.current, grad)
            elif k == "axipoly":
                out, e, s, dm, fl = prep.evaluate(pts, grad=grad, rtol=rtol)
            elif k == "toroidal":
                out, e, s = toroidal_dd(pts, float(part.RB), grad)
            else:
                raise ValueError(f"unknown part kind {k!r}")
            tot = _dd_add_arrays(tot, out)
            err += e
            scale += s
            if dm is not None:
                dmin = np.minimum(dmin, dm)
            if fl is not None:
                flag |= fl
            if wk is not None:
                work += wk
        hi, lo = _split_round(tot)
        hi = hi.reshape(N, 15)
        lo = lo.reshape(N, 15)
        r = {
            "B": hi[:, 0:3].copy(), "A": hi[:, 3:6].copy(), "dB": hi[:, 6:15].reshape(N, 3, 3).copy(),
            "B_lo": lo[:, 0:3].copy(), "A_lo": lo[:, 3:6].copy(), "dB_lo": lo[:, 6:15].reshape(N, 3, 3).copy(),
            "err_B": err[:, 0], "err_A": err[:, 1], "err_dB": err[:, 2],
            "scale_B": scale[:, 0], "scale_A": scale[:, 1], "scale_dB": scale[:, 2],
            "dmin": dmin, "flag": flag, "work": work,
        }
        if not grad:
            for key in ("dB", "dB_lo", "err_dB", "scale_dB"):
                r.pop(key)
        return r


def evaluate(source, points, grad=True, rtol=1e-20, **kw):
    """One-shot precise evaluation (builds the per-source caches each call)."""
    return Reference(source, **kw).evaluate(points, grad=grad, rtol=rtol)


def B(source, points):
    return evaluate(source, points, grad=False)["B"]
