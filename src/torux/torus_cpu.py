"""Host (numba) kernels of the torus Newton system: the Toeplitz-plus-Hankel Jacobian blocks gathered
from the grid FFT of a coefficient function, fused into one parallel loop (numpy would materialise every
n^2 index and gather array).  Same formulas as torux.torus._TorusConsts.blocks."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _blocks(ahat, mi, ni, mj, nj, ntheta, nphi, kind, zero_c, sign, out):
    """out[i, j] = sign * value for row modes (mi, ni) and column modes (mj, nj):
    kind 0 (s, c): -(Im Hp + Im Hm); 1 (s, s): Re Hm - Re Hp; 2 (c, c): Re Hm + Re Hp; 3 (c, s): Im Hm - Im Hp,
    Hp = ahat[(mi + mj), -(ni + nj)], Hm = ahat[(mi - mj), -(ni - nj)] (indices mod the grid; the grid has
    ntheta > 2 M and nphi > 2 N points, so one conditional wrap replaces the modulo); rows of the cosine
    tests (kinds 2, 3) halved where zero_c (the (0, 0) mode).  out may be a strided view."""
    nr = mi.shape[0]
    nc = mj.shape[0]
    for i in prange(nr):
        half = sign * (0.5 if (kind >= 2 and zero_c[i]) else 1.0)
        a = mi[i]
        bn = ni[i]
        for j in range(nc):
            p0 = a + mj[j]
            if p0 >= ntheta:
                p0 -= ntheta
            q0 = a - mj[j]
            if q0 < 0:
                q0 += ntheta
            p1 = -(bn + nj[j])
            if p1 < 0:
                p1 += nphi
            q1 = nj[j] - bn
            if q1 < 0:
                q1 += nphi
            hp = ahat[p0, p1]
            hm = ahat[q0, q1]
            if kind == 0:
                v = -(hp.imag + hm.imag)
            elif kind == 1:
                v = hm.real - hp.real
            elif kind == 2:
                v = hm.real + hp.real
            else:
                v = hm.imag - hp.imag
            out[i, j] = half * v


_KIND = {("s", "c"): 0, ("s", "s"): 1, ("c", "c"): 2, ("c", "s"): 3}


def blocks(ahat, modes_r, modes_c, ntheta, nphi, rk, ck, zero_c, out=None, sign=1.0):
    mi, ni = (np.ascontiguousarray(a, dtype=np.int64) for a in modes_r)
    mj, nj = (np.ascontiguousarray(a, dtype=np.int64) for a in modes_c)
    assert ntheta > 2 * max(int(mi.max()), int(mj.max())) and nphi > 2 * max(int(np.abs(ni).max()), int(np.abs(nj).max()))
    if out is None:
        out = np.empty((mi.shape[0], mj.shape[0]))
    zc = np.ascontiguousarray(zero_c if rk == "c" else np.zeros(mi.shape[0], bool))
    _blocks(np.ascontiguousarray(ahat), mi, ni, mj, nj, int(ntheta), int(nphi), _KIND[(rk, ck)], zc, float(sign), out)
    return out


@njit(cache=True)
def _lu_solve(A, B):
    """Gaussian elimination with partial pivoting (LAPACK dgesv's algorithm, unblocked), single-threaded."""
    n = A.shape[0]
    k_rhs = B.shape[1]
    for k in range(n):
        p = k
        amax = abs(A[k, k])
        for i in range(k + 1, n):
            if abs(A[i, k]) > amax:
                amax = abs(A[i, k])
                p = i
        if p != k:
            for j in range(n):
                t = A[k, j]; A[k, j] = A[p, j]; A[p, j] = t
            for j in range(k_rhs):
                t = B[k, j]; B[k, j] = B[p, j]; B[p, j] = t
        piv = A[k, k]
        for i in range(k + 1, n):
            f = A[i, k] / piv
            if f != 0.0:
                for j in range(k + 1, n):
                    A[i, j] -= f * A[k, j]
                for j in range(k_rhs):
                    B[i, j] -= f * B[k, j]
    for i in range(n - 1, -1, -1):
        for j in range(k_rhs):
            s = B[i, j]
            for c in range(i + 1, n):
                s -= A[i, c] * B[c, j]
            B[i, j] = s / A[i, i]
    return B


SMALL = 400


def small_solve(A, b):
    """A x = b for small dense systems (n <= SMALL) without the BLAS thread pool: multithreaded BLAS on a
    65 x 65 system took 0.4 s per call on a loaded machine (np.linalg.solve), this takes microseconds.
    Larger systems go to LAPACK."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b)
    if A.shape[0] > SMALL or np.iscomplexobj(A) or np.iscomplexobj(b):
        return np.linalg.solve(A, b)
    vec = b.ndim == 1
    B = np.array(b.reshape(-1, 1) if vec else b, dtype=np.float64, copy=True, order="C")
    x = _lu_solve(np.array(A, copy=True, order="C"), B)
    return x[:, 0] if vec else x
