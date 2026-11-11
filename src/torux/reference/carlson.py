"""Carlson's symmetric integral R_D(x, y, z) with forward-mode derivatives d/dy, d/dz.

Duplication (DLMF 19.26.20 / 19.36):  R_D(x,y,z) = 3 sum_m 4^-m / (sqrt(z_m) (z_m + lam_m))
                                                  + 4^-n R_D(x_n, y_n, z_n),
lam = sqrt(x y) + sqrt(x z) + sqrt(y z),  x_{m+1} = (x_m + lam_m)/4 (same for y, z).
The tail R_D(x_n, y_n, z_n) is summed with Carlson's general expansion of the
R-hypergeometric function R_{-a}(b; z) = A^-a sum_k (a)_k/(c)_k T_k(b; Z),
A = sum b_i z_i / c, Z_i = 1 - z_i/A, T_k from the generating function
prod_i (1 - t Z_i)^-b_i; for R_D, a = 3/2, b = (1/2, 1/2, 3/2), c = 5/2 and
(a)_k/(c)_k = 3/(3 + 2k).  No tabulated series coefficients are needed.

Derivatives are carried through every step as dual numbers (value, d/dy, d/dz)
(x is the constant 0 in all uses here, but its iterates depend on y, z).

``rd_dual_dd`` (double-double, truncation ~1e-33) and ``rd_dual_f64`` (float64).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import dd

# coefficients (b)_m / m! for b = 1/2 and b = 3/2, m = 0..8
_C_HALF = np.array([1.0, 0.5, 0.375, 0.3125, 0.2734375, 0.24609375, 0.2255859375, 0.20947265625,
                    0.196380615234375])
_C_3HALF = np.array([1.0, 1.5, 1.875, 2.1875, 2.4609375, 2.70703125, 2.9326171875, 3.14208984375,
                     3.338470458984375])


# ------------------------------------------------------------------ dd dual helpers
@njit(cache=True, error_model="numpy")
def _tmul(a0, a1, a2, b0, b1, b2):
    return (dd.mul(a0, b0), dd.add(dd.mul(a1, b0), dd.mul(a0, b1)), dd.add(dd.mul(a2, b0), dd.mul(a0, b2)))


@njit(cache=True, error_model="numpy")
def _tsqrt(a0, a1, a2):
    if a0.real == 0.0:
        z = complex(0.0, 0.0)
        return z, z, z
    s = dd.sqrt(a0)
    t = dd.mul_pow2(s, 2.0)
    return s, dd.div(a1, t), dd.div(a2, t)


@njit(cache=True, error_model="numpy")
def _tdiv(a0, a1, a2, b0, b1, b2):
    q = dd.div(a0, b0)
    return q, dd.div(dd.sub(a1, dd.mul(q, b1)), b0), dd.div(dd.sub(a2, dd.mul(q, b2)), b0)


@njit(cache=True, error_model="numpy")
def rd_dual_dd(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    """R_D and its derivatives along the dual directions (all dd).

    Inputs are dual numbers (value, d1, d2) for x, y, z.  Returns (R, dR_1, dR_2).
    """
    s0 = complex(0.0, 0.0)
    s1 = complex(0.0, 0.0)
    s2 = complex(0.0, 0.0)
    fac = 1.0
    for it in range(200):
        # A = (x + y + 3 z) / 5
        a0 = dd.div_d(dd.add(dd.add(x0, y0), dd.mul_d(z0, 3.0)), 5.0)
        a1 = dd.div_d(dd.add(dd.add(x1, y1), dd.mul_d(z1, 3.0)), 5.0)
        a2 = dd.div_d(dd.add(dd.add(x2, y2), dd.mul_d(z2, 3.0)), 5.0)
        ah = a0.real
        r = max(abs(ah - x0.real), abs(ah - y0.real), abs(ah - z0.real)) / ah
        if r < 1e-5:
            break
        sx0, sx1, sx2 = _tsqrt(x0, x1, x2)
        sy0, sy1, sy2 = _tsqrt(y0, y1, y2)
        sz0, sz1, sz2 = _tsqrt(z0, z1, z2)
        p0, p1, p2 = _tmul(sx0, sx1, sx2, sy0, sy1, sy2)
        q0, q1, q2 = _tmul(sx0, sx1, sx2, sz0, sz1, sz2)
        u0, u1, u2 = _tmul(sy0, sy1, sy2, sz0, sz1, sz2)
        l0 = dd.add(dd.add(p0, q0), u0)
        l1 = dd.add(dd.add(p1, q1), u1)
        l2 = dd.add(dd.add(p2, q2), u2)
        # term = fac / (sqrt(z) (z + lam))
        w0, w1, w2 = _tmul(sz0, sz1, sz2, dd.add(z0, l0), dd.add(z1, l1), dd.add(z2, l2))
        t0, t1, t2 = _tdiv(complex(fac, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), w0, w1, w2)
        s0 = dd.add(s0, t0)
        s1 = dd.add(s1, t1)
        s2 = dd.add(s2, t2)
        fac *= 0.25
        x0 = dd.mul_pow2(dd.add(x0, l0), 0.25)
        x1 = dd.mul_pow2(dd.add(x1, l1), 0.25)
        x2 = dd.mul_pow2(dd.add(x2, l2), 0.25)
        y0 = dd.mul_pow2(dd.add(y0, l0), 0.25)
        y1 = dd.mul_pow2(dd.add(y1, l1), 0.25)
        y2 = dd.mul_pow2(dd.add(y2, l2), 0.25)
        z0 = dd.mul_pow2(dd.add(z0, l0), 0.25)
        z1 = dd.mul_pow2(dd.add(z1, l1), 0.25)
        z2 = dd.mul_pow2(dd.add(z2, l2), 0.25)
    # tail: A^-3/2 sum_k 3/(3+2k) T_k, Z_i = 1 - z_i / A  (as duals)
    X0, X1, X2 = _tdiv(dd.sub(a0, x0), dd.sub(a1, x1), dd.sub(a2, x2), a0, a1, a2)
    Y0, Y1, Y2 = _tdiv(dd.sub(a0, y0), dd.sub(a1, y1), dd.sub(a2, y2), a0, a1, a2)
    Z0, Z1, Z2 = _tdiv(dd.sub(a0, z0), dd.sub(a1, z1), dd.sub(a2, z2), a0, a1, a2)
    N = 6
    # powers
    XP = np.empty((N + 1, 3), dtype=np.complex128)
    YP = np.empty((N + 1, 3), dtype=np.complex128)
    ZP = np.empty((N + 1, 3), dtype=np.complex128)
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    XP[0, 0] = one; XP[0, 1] = zero; XP[0, 2] = zero
    YP[0, 0] = one; YP[0, 1] = zero; YP[0, 2] = zero
    ZP[0, 0] = one; ZP[0, 1] = zero; ZP[0, 2] = zero
    for m in range(1, N + 1):
        XP[m, 0], XP[m, 1], XP[m, 2] = _tmul(XP[m - 1, 0], XP[m - 1, 1], XP[m - 1, 2], X0, X1, X2)
        YP[m, 0], YP[m, 1], YP[m, 2] = _tmul(YP[m - 1, 0], YP[m - 1, 1], YP[m - 1, 2], Y0, Y1, Y2)
        ZP[m, 0], ZP[m, 1], ZP[m, 2] = _tmul(ZP[m - 1, 0], ZP[m - 1, 1], ZP[m - 1, 2], Z0, Z1, Z2)
    # S_m = sum_{i+j=m} c_i c_j X^i Y^j   (b = 1/2, 1/2)
    S = np.empty((N + 1, 3), dtype=np.complex128)
    for m in range(N + 1):
        c0 = zero; c1 = zero; c2 = zero
        for i in range(m + 1):
            j = m - i
            e0, e1, e2 = _tmul(XP[i, 0], XP[i, 1], XP[i, 2], YP[j, 0], YP[j, 1], YP[j, 2])
            cc = _C_HALF[i] * _C_HALF[j]
            c0 = dd.add(c0, dd.mul_d(e0, cc))
            c1 = dd.add(c1, dd.mul_d(e1, cc))
            c2 = dd.add(c2, dd.mul_d(e2, cc))
        S[m, 0] = c0; S[m, 1] = c1; S[m, 2] = c2
    g0 = one; g1 = zero; g2 = zero
    for n in range(2, N + 1):
        c0 = zero; c1 = zero; c2 = zero
        for k in range(n + 1):
            m = n - k
            e0, e1, e2 = _tmul(S[m, 0], S[m, 1], S[m, 2], ZP[k, 0], ZP[k, 1], ZP[k, 2])
            cc = _C_3HALF[k]
            c0 = dd.add(c0, dd.mul_d(e0, cc))
            c1 = dd.add(c1, dd.mul_d(e1, cc))
            c2 = dd.add(c2, dd.mul_d(e2, cc))
        f = 3.0 / (3.0 + 2.0 * n)
        g0 = dd.add(g0, dd.mul_d(c0, f))
        g1 = dd.add(g1, dd.mul_d(c1, f))
        g2 = dd.add(g2, dd.mul_d(c2, f))
    # A^-3/2 as dual
    sa0, sa1, sa2 = _tsqrt(a0, a1, a2)
    d0, d1, d2 = _tmul(a0, a1, a2, sa0, sa1, sa2)
    h0, h1, h2 = _tdiv(g0, g1, g2, d0, d1, d2)
    h0 = dd.mul_d(h0, fac)
    h1 = dd.mul_d(h1, fac)
    h2 = dd.mul_d(h2, fac)
    return (dd.add(dd.mul_d(s0, 3.0), h0), dd.add(dd.mul_d(s1, 3.0), h1), dd.add(dd.mul_d(s2, 3.0), h2))


@njit(cache=True, error_model="numpy")
def rd_dual0_dd(y, z):
    """R_D(0, y, z) and (dR/dy, dR/dz) for dd y, z > 0."""
    zero = complex(0.0, 0.0)
    one = complex(1.0, 0.0)
    return rd_dual_dd(zero, zero, zero, y, one, zero, z, zero, one)


# ------------------------------------------------------------------ float64 version
@njit(cache=True, error_model="numpy")
def rd_dual_f64(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    fac = 1.0
    a0 = a1 = a2 = 0.0
    for it in range(200):
        a0 = (x0 + y0 + 3.0 * z0) / 5.0
        a1 = (x1 + y1 + 3.0 * z1) / 5.0
        a2 = (x2 + y2 + 3.0 * z2) / 5.0
        r = max(abs(a0 - x0), abs(a0 - y0), abs(a0 - z0)) / a0
        if r < 1e-3:
            break
        sx0 = math.sqrt(x0)
        if sx0 > 0.0:
            sx1 = x1 / (2.0 * sx0)
            sx2 = x2 / (2.0 * sx0)
        else:
            sx1 = 0.0
            sx2 = 0.0
        sy0 = math.sqrt(y0)
        sy1 = y1 / (2.0 * sy0)
        sy2 = y2 / (2.0 * sy0)
        sz0 = math.sqrt(z0)
        sz1 = z1 / (2.0 * sz0)
        sz2 = z2 / (2.0 * sz0)
        l0 = sx0 * sy0 + sx0 * sz0 + sy0 * sz0
        l1 = sx1 * sy0 + sx0 * sy1 + sx1 * sz0 + sx0 * sz1 + sy1 * sz0 + sy0 * sz1
        l2 = sx2 * sy0 + sx0 * sy2 + sx2 * sz0 + sx0 * sz2 + sy2 * sz0 + sy0 * sz2
        w0 = sz0 * (z0 + l0)
        w1 = sz1 * (z0 + l0) + sz0 * (z1 + l1)
        w2 = sz2 * (z0 + l0) + sz0 * (z2 + l2)
        t0 = fac / w0
        s0 += t0
        s1 -= t0 * w1 / w0
        s2 -= t0 * w2 / w0
        fac *= 0.25
        x0 = (x0 + l0) * 0.25
        x1 = (x1 + l1) * 0.25
        x2 = (x2 + l2) * 0.25
        y0 = (y0 + l0) * 0.25
        y1 = (y1 + l1) * 0.25
        y2 = (y2 + l2) * 0.25
        z0 = (z0 + l0) * 0.25
        z1 = (z1 + l1) * 0.25
        z2 = (z2 + l2) * 0.25
    # Z_i = 1 - z_i/A = (A - z_i)/A as duals
    X0 = (a0 - x0) / a0
    X1 = ((a1 - x1) - X0 * a1) / a0
    X2 = ((a2 - x2) - X0 * a2) / a0
    Y0 = (a0 - y0) / a0
    Y1 = ((a1 - y1) - Y0 * a1) / a0
    Y2 = ((a2 - y2) - Y0 * a2) / a0
    Z0 = (a0 - z0) / a0
    Z1 = ((a1 - z1) - Z0 * a1) / a0
    Z2 = ((a2 - z2) - Z0 * a2) / a0
    # explicit elementary-symmetric form of the series through degree 5 (from the generic expansion)
    N = 5
    XP0 = np.empty(N + 1); XP1 = np.empty(N + 1); XP2 = np.empty(N + 1)
    YP0 = np.empty(N + 1); YP1 = np.empty(N + 1); YP2 = np.empty(N + 1)
    ZP0 = np.empty(N + 1); ZP1 = np.empty(N + 1); ZP2 = np.empty(N + 1)
    XP0[0] = 1.0; XP1[0] = 0.0; XP2[0] = 0.0
    YP0[0] = 1.0; YP1[0] = 0.0; YP2[0] = 0.0
    ZP0[0] = 1.0; ZP1[0] = 0.0; ZP2[0] = 0.0
    for m in range(1, N + 1):
        XP0[m] = XP0[m - 1] * X0
        XP1[m] = XP1[m - 1] * X0 + XP0[m - 1] * X1
        XP2[m] = XP2[m - 1] * X0 + XP0[m - 1] * X2
        YP0[m] = YP0[m - 1] * Y0
        YP1[m] = YP1[m - 1] * Y0 + YP0[m - 1] * Y1
        YP2[m] = YP2[m - 1] * Y0 + YP0[m - 1] * Y2
        ZP0[m] = ZP0[m - 1] * Z0
        ZP1[m] = ZP1[m - 1] * Z0 + ZP0[m - 1] * Z1
        ZP2[m] = ZP2[m - 1] * Z0 + ZP0[m - 1] * Z2
    S0 = np.zeros(N + 1); S1 = np.zeros(N + 1); S2 = np.zeros(N + 1)
    for m in range(N + 1):
        for i in range(m + 1):
            j = m - i
            cc = _C_HALF[i] * _C_HALF[j]
            S0[m] += cc * XP0[i] * YP0[j]
            S1[m] += cc * (XP1[i] * YP0[j] + XP0[i] * YP1[j])
            S2[m] += cc * (XP2[i] * YP0[j] + XP0[i] * YP2[j])
    g0 = 1.0
    g1 = 0.0
    g2 = 0.0
    for n in range(2, N + 1):
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        for k in range(n + 1):
            m = n - k
            cc = _C_3HALF[k]
            c0 += cc * S0[m] * ZP0[k]
            c1 += cc * (S1[m] * ZP0[k] + S0[m] * ZP1[k])
            c2 += cc * (S2[m] * ZP0[k] + S0[m] * ZP2[k])
        f = 3.0 / (3.0 + 2.0 * n)
        g0 += f * c0
        g1 += f * c1
        g2 += f * c2
    sa = math.sqrt(a0)
    d0 = a0 * sa                    # A^{3/2}
    d1 = 1.5 * sa * a1
    d2 = 1.5 * sa * a2
    h0 = g0 / d0
    h1 = (g1 - h0 * d1) / d0
    h2 = (g2 - h0 * d2) / d0
    return 3.0 * s0 + fac * h0, 3.0 * s1 + fac * h1, 3.0 * s2 + fac * h2


@njit(cache=True, error_model="numpy")
def rd_dual0_f64(y, z):
    return rd_dual_f64(0.0, 0.0, 0.0, y, 1.0, 0.0, z, 0.0, 1.0)
