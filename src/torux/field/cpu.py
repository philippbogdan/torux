"""CPU kernels of the solver's field engine (numba): the same quadrature as torux.field.kernels
(trapezoid nodes of Fourier filaments with the half-rule sum, dipoles), as loops parallel over
blocks of targets.  The innermost loop runs over the targets of a block (independent
accumulators: SIMD without reassociation); sources are summed in blocks of SB in order, and the
block sums are added with Neumaier compensation, so a sum of S terms carries the rounding of an
SB-term sum of block-sized partials (~1 ulp of the result) instead of growing with S.
Written independently of torux.reference (which uses Gauss-Legendre panels).  Used where the
engine runs on CPUs: XLA's CPU code for the pairwise reduction with the gradient materialises
(T, S, 3, 3) products and is 10-50x slower than these loops.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

TB = 16          # targets per block (SIMD lanes x unroll)
SB = 32          # sources per in-order block


def _make_filament(need_grad, need_A):
    NG = bool(need_grad)
    NA = bool(need_A)
    K = 18

    @njit(parallel=True, cache=True, fastmath=False)
    def kern(xs, r, w, wh, out):
        T = xs.shape[1]
        S = r.shape[0]
        nblk = (T + TB - 1) // TB
        for b in prange(nblk):
            i0 = b * TB
            p0 = np.empty(TB); p1 = np.empty(TB); p2 = np.empty(TB)
            for t in range(TB):
                ii = min(i0 + t, T - 1)
                p0[t] = xs[0, ii]; p1[t] = xs[1, ii]; p2[t] = xs[2, ii]
            tot = np.zeros((K, TB))
            cmp = np.zeros((K, TB))
            acc = np.empty((K, TB))
            for j0 in range(0, S, SB):
                acc[:, :] = 0.0
                for j in range(j0, min(S, j0 + SB)):
                    r0 = r[j, 0]; r1 = r[j, 1]; r2_ = r[j, 2]
                    w0 = w[j, 0]; w1 = w[j, 1]; w2 = w[j, 2]
                    h = wh[j]
                    for t in range(TB):
                        d0 = p0[t] - r0; d1 = p1[t] - r1; d2 = p2[t] - r2_
                        q = d0 * d0 + d1 * d1 + d2 * d2
                        ir = 1.0 / math.sqrt(q)
                        ir3 = ir * ir * ir
                        c0 = w1 * d2 - w2 * d1
                        c1 = w2 * d0 - w0 * d2
                        c2 = w0 * d1 - w1 * d0
                        b0 = c0 * ir3; b1 = c1 * ir3; b2 = c2 * ir3
                        acc[0, t] += b0; acc[1, t] += b1; acc[2, t] += b2
                        acc[15, t] += h * b0; acc[16, t] += h * b1; acc[17, t] += h * b2
                        if NA:
                            acc[3, t] += w0 * ir; acc[4, t] += w1 * ir; acc[5, t] += w2 * ir
                        if NG:
                            # dB_a/dx_b = (w x e_b)_a / r^3 - 3 (w x d)_a d_b / r^5
                            f = 3.0 * ir3 * ir * ir
                            e0 = c0 * f; e1 = c1 * f; e2 = c2 * f
                            u0 = w0 * ir3; u1 = w1 * ir3; u2 = w2 * ir3
                            acc[6, t] -= e0 * d0
                            acc[7, t] += -u2 - e0 * d1
                            acc[8, t] += u1 - e0 * d2
                            acc[9, t] += u2 - e1 * d0
                            acc[10, t] -= e1 * d1
                            acc[11, t] += -u0 - e1 * d2
                            acc[12, t] += -u1 - e2 * d0
                            acc[13, t] += u0 - e2 * d1
                            acc[14, t] -= e2 * d2
                for k in range(K):
                    for t in range(TB):
                        s = tot[k, t]; v = acc[k, t]
                        u = s + v
                        if abs(s) >= abs(v):
                            cmp[k, t] += (s - u) + v
                        else:
                            cmp[k, t] += (v - u) + s
                        tot[k, t] = u
            for t in range(TB):
                ii = i0 + t
                if ii < T:
                    for k in range(K):
                        out[ii, k] = tot[k, t] + cmp[k, t]
    return kern


def _make_dipole(need_grad, need_A):
    NG = bool(need_grad)
    NA = bool(need_A)
    K = 15

    @njit(parallel=True, cache=True, fastmath=False)
    def kern(xs, p, m, out):
        T = xs.shape[1]
        D = p.shape[0]
        nblk = (T + TB - 1) // TB
        for b in prange(nblk):
            i0 = b * TB
            p0 = np.empty(TB); p1 = np.empty(TB); p2 = np.empty(TB)
            for t in range(TB):
                ii = min(i0 + t, T - 1)
                p0[t] = xs[0, ii]; p1[t] = xs[1, ii]; p2[t] = xs[2, ii]
            tot = np.zeros((K, TB))
            cmp = np.zeros((K, TB))
            acc = np.empty((K, TB))
            for j0 in range(0, D, SB):
                acc[:, :] = 0.0
                for j in range(j0, min(D, j0 + SB)):
                    q0 = p[j, 0]; q1 = p[j, 1]; q2 = p[j, 2]
                    m0 = m[j, 0]; m1 = m[j, 1]; m2 = m[j, 2]
                    for t in range(TB):
                        d0 = p0[t] - q0; d1 = p1[t] - q1; d2 = p2[t] - q2
                        r2 = d0 * d0 + d1 * d1 + d2 * d2
                        ir = 1.0 / math.sqrt(r2)
                        ir2 = ir * ir
                        ir3 = ir2 * ir
                        md = m0 * d0 + m1 * d1 + m2 * d2
                        tt = 3.0 * md * ir2
                        acc[0, t] += (tt * d0 - m0) * ir3
                        acc[1, t] += (tt * d1 - m1) * ir3
                        acc[2, t] += (tt * d2 - m2) * ir3
                        if NA:
                            acc[3, t] += (m1 * d2 - m2 * d1) * ir3
                            acc[4, t] += (m2 * d0 - m0 * d2) * ir3
                            acc[5, t] += (m0 * d1 - m1 * d0) * ir3
                        if NG:
                            # dB_a/dx_b = 3 (m_a d_b + m_b d_a + md delta_ab) / r^5 - 15 md d_a d_b / r^7
                            ir5 = 3.0 * ir3 * ir2
                            g = 5.0 * md * ir2
                            a00 = ir5 * (2.0 * m0 * d0 + md - g * d0 * d0)
                            a11 = ir5 * (2.0 * m1 * d1 + md - g * d1 * d1)
                            a22 = ir5 * (2.0 * m2 * d2 + md - g * d2 * d2)
                            a01 = ir5 * (m0 * d1 + m1 * d0 - g * d0 * d1)
                            a02 = ir5 * (m0 * d2 + m2 * d0 - g * d0 * d2)
                            a12 = ir5 * (m1 * d2 + m2 * d1 - g * d1 * d2)
                            acc[6, t] += a00; acc[7, t] += a01; acc[8, t] += a02
                            acc[9, t] += a01; acc[10, t] += a11; acc[11, t] += a12
                            acc[12, t] += a02; acc[13, t] += a12; acc[14, t] += a22
                for k in range(K):
                    for t in range(TB):
                        s = tot[k, t]; v = acc[k, t]
                        u = s + v
                        if abs(s) >= abs(v):
                            cmp[k, t] += (s - u) + v
                        else:
                            cmp[k, t] += (v - u) + s
                        tot[k, t] = u
            for t in range(TB):
                ii = i0 + t
                if ii < T:
                    for k in range(K):
                        out[ii, k] = tot[k, t] + cmp[k, t]
    return kern


_FIL = {}
_DIP = {}


def filament(x, r, w, wh, need_grad=True, need_A=True):
    """Fourier-filament nodes: (T, 18) = B, A, G (row-major dB_a/dx_b), Bh (half rule)."""
    key = (bool(need_grad), bool(need_A))
    if key not in _FIL:
        _FIL[key] = _make_filament(*key)
    xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64).T)
    out = np.empty((xs.shape[1], 18))
    _FIL[key](xs, np.ascontiguousarray(r, dtype=np.float64), np.ascontiguousarray(w, dtype=np.float64),
              np.ascontiguousarray(wh, dtype=np.float64), out)
    return out


def dipoles(x, p, m, need_grad=True, need_A=True):
    """Point dipoles: (T, 15) = B, A, G."""
    key = (bool(need_grad), bool(need_A))
    if key not in _DIP:
        _DIP[key] = _make_dipole(*key)
    xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64).T)
    out = np.empty((xs.shape[1], 15))
    _DIP[key](xs, np.ascontiguousarray(p, dtype=np.float64), np.ascontiguousarray(m, dtype=np.float64), out)
    return out


def _make_segment(need_grad, need_A):
    NG = bool(need_grad)
    NA = bool(need_A)
    K = 15

    @njit(parallel=True, cache=True, fastmath=False)
    def kern(xs, a, bb, cw, out):
        """Straight segments a -> b (exact): with S = r_i + r_f, q = r_i r_f |u_i + u_f|^2
        (= S^2 - L^2 without cancellation), f = 2 S / (r_i r_f q):
            B = cw f (dl x R_i),   A = cw dl / L ln((S + L)^2 / q) = cw dl / L 2 atanh(L / S),
            dB_a/dx_b = cw f [(dl x e_b)_a - (dl x R_i)_a g_b],
            g = (S^2 + L^2) / (S q) (u_i + u_f) + u_i / r_i + u_f / r_f."""
        T = xs.shape[1]
        D = a.shape[0]
        nblk = (T + TB - 1) // TB
        for b in prange(nblk):
            i0 = b * TB
            p0 = np.empty(TB); p1 = np.empty(TB); p2 = np.empty(TB)
            for t in range(TB):
                ii = min(i0 + t, T - 1)
                p0[t] = xs[0, ii]; p1[t] = xs[1, ii]; p2[t] = xs[2, ii]
            tot = np.zeros((K, TB))
            cmp = np.zeros((K, TB))
            acc = np.empty((K, TB))
            for j0 in range(0, D, SB):
                acc[:, :] = 0.0
                for j in range(j0, min(D, j0 + SB)):
                    a0 = a[j, 0]; a1 = a[j, 1]; a2 = a[j, 2]
                    l0 = bb[j, 0] - a0; l1 = bb[j, 1] - a1; l2 = bb[j, 2] - a2
                    L2 = l0 * l0 + l1 * l1 + l2 * l2
                    L = math.sqrt(L2)
                    c = cw[j]
                    for t in range(TB):
                        i0_ = p0[t] - a0; i1_ = p1[t] - a1; i2_ = p2[t] - a2
                        f0 = i0_ - l0; f1 = i1_ - l1; f2 = i2_ - l2
                        ri = math.sqrt(i0_ * i0_ + i1_ * i1_ + i2_ * i2_)
                        rf = math.sqrt(f0 * f0 + f1 * f1 + f2 * f2)
                        iri = 1.0 / ri; irf = 1.0 / rf
                        s0 = i0_ * iri + f0 * irf; s1 = i1_ * iri + f1 * irf; s2 = i2_ * iri + f2 * irf
                        q = ri * rf * (s0 * s0 + s1 * s1 + s2 * s2)
                        S = ri + rf
                        fac = c * 2.0 * S / (ri * rf * q)
                        c0 = l1 * i2_ - l2 * i1_
                        c1 = l2 * i0_ - l0 * i2_
                        c2 = l0 * i1_ - l1 * i0_
                        acc[0, t] += fac * c0; acc[1, t] += fac * c1; acc[2, t] += fac * c2
                        if NA:
                            # ln((S + L)^2 / q) = 2 atanh(L / S): the atanh form for L / S < 1/2 (no log
                            # of a number near 1), the log form near the segment
                            y = L / S
                            lg = (2.0 * math.atanh(y) if y < 0.5 else math.log((S + L) * (S + L) / q)) * c / L
                            acc[3, t] += l0 * lg; acc[4, t] += l1 * lg; acc[5, t] += l2 * lg
                        if NG:
                            h = (S * S + L2) / (S * q)
                            g0 = h * s0 + i0_ * iri * iri + f0 * irf * irf
                            g1 = h * s1 + i1_ * iri * iri + f1 * irf * irf
                            g2 = h * s2 + i2_ * iri * iri + f2 * irf * irf
                            e0 = fac * c0; e1 = fac * c1; e2 = fac * c2
                            acc[6, t] -= e0 * g0
                            acc[7, t] += -fac * l2 - e0 * g1
                            acc[8, t] += fac * l1 - e0 * g2
                            acc[9, t] += fac * l2 - e1 * g0
                            acc[10, t] -= e1 * g1
                            acc[11, t] += -fac * l0 - e1 * g2
                            acc[12, t] += -fac * l1 - e2 * g0
                            acc[13, t] += fac * l0 - e2 * g1
                            acc[14, t] -= e2 * g2
                for k in range(K):
                    for t in range(TB):
                        s = tot[k, t]; v = acc[k, t]
                        u = s + v
                        if abs(s) >= abs(v):
                            cmp[k, t] += (s - u) + v
                        else:
                            cmp[k, t] += (v - u) + s
                        tot[k, t] = u
            for t in range(TB):
                ii = i0 + t
                if ii < T:
                    for k in range(K):
                        out[ii, k] = tot[k, t] + cmp[k, t]
    return kern


_SEG = {}


def segments(x, a, b, cw, need_grad=True, need_A=True):
    """Straight segments a -> b with weights cw = mu0 I / 4 pi: (T, 15) = B, A, G."""
    key = (bool(need_grad), bool(need_A))
    if key not in _SEG:
        _SEG[key] = _make_segment(*key)
    xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64).T)
    out = np.empty((xs.shape[1], 15))
    _SEG[key](xs, np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64),
              np.ascontiguousarray(cw, dtype=np.float64), out)
    return out


def set_threads():
    """Numba threads = TORUX_THREADS, else the cores this process may use (the affinity mask)."""
    import os
    import numba
    n = int(os.environ.get("TORUX_THREADS", "0") or 0)
    if n <= 0:
        try:
            n = len(os.sched_getaffinity(0))
        except Exception:
            n = os.cpu_count() or 1
    n = max(1, min(n, numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    return n
