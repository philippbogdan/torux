"""Quadrature rules computed in mpmath (50 digits), returned as dd (complex128) or float64.

* ``gauss_legendre(n)``: nodes/weights on [-1, 1] by Newton iteration on P_n at 50 digits.
* ``tanh_sinh01(h, tmax)``: double-exponential rule on [0, 1] (nodes u_k, weights w_k),
  nodes accurate near u = 0 (u_k = 1/(1 + exp(-pi sinh(kh)))).

Rules are cached in memory; computing them takes milliseconds.
"""
from __future__ import annotations

import functools

import mpmath
import numpy as np

from .dd import mp_to_dd


def _legendre_and_deriv(n, x):
    p0, p1 = mpmath.mpf(1), x
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    dp = n * (x * p1 - p0) / (x * x - 1)
    return p1, dp


@functools.lru_cache(maxsize=None)
def gauss_legendre_mp(n, dps=50):
    """Nodes and weights (lists of mpf) on [-1, 1], ascending."""
    x0, _ = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    with mpmath.workdps(dps + 10):
        for xi in x0:
            x = mpmath.mpf(float(xi))
            for _ in range(100):
                p, dp = _legendre_and_deriv(n, x)
                dx = p / dp
                x -= dx
                if abs(dx) < mpmath.mpf(10) ** (-(dps + 5)):
                    break
            p, dp = _legendre_and_deriv(n, x)
            xs.append(+x)
            ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


@functools.lru_cache(maxsize=None)
def gauss_legendre_dd(n):
    """(nodes, weights) as complex128 dd arrays on [-1, 1]."""
    xs, ws = gauss_legendre_mp(n)
    X = np.array([complex(*mp_to_dd(x)) for x in xs])
    W = np.array([complex(*mp_to_dd(w)) for w in ws])
    return X, W


@functools.lru_cache(maxsize=None)
def gauss_legendre_f64(n):
    xs, ws = gauss_legendre_mp(n)
    return np.array([float(x) for x in xs]), np.array([float(w) for w in ws])


@functools.lru_cache(maxsize=None)
def tanh_sinh01_mp(h, tmax, dps=50):
    us, ws = [], []
    with mpmath.workdps(dps + 10):
        hh = mpmath.mpf(h)
        k = int(round(tmax / h))
        for j in range(-k, k + 1):
            t = j * hh
            s = mpmath.pi / 2 * mpmath.sinh(t)
            u = 1 / (1 + mpmath.exp(-2 * s))          # (1 + tanh s)/2
            w = hh * (mpmath.pi / 4) * mpmath.cosh(t) / mpmath.cosh(s) ** 2
            us.append(u)
            ws.append(w)
    return us, ws


@functools.lru_cache(maxsize=None)
def tanh_sinh01_dd(h, tmax):
    us, ws = tanh_sinh01_mp(h, tmax)
    U = np.array([complex(*mp_to_dd(u)) for u in us])
    W = np.array([complex(*mp_to_dd(w)) for w in ws])
    return U, W


@functools.lru_cache(maxsize=None)
def tanh_sinh01_f64(h, tmax):
    us, ws = tanh_sinh01_mp(h, tmax)
    return np.array([float(u) for u in us]), np.array([float(w) for w in ws])
