"""The exterior field of a toroidal current sheet (torux.SurfaceCurrent): B, A and grad B at points outside it.

    B(x) = mu0/(4 pi) int K x (x - y) / |x - y|^3 dS,   A(x) = mu0/(4 pi) int K / |x - y| dS,
    K dS = (c_phi r_theta - c_theta r_phi) dtheta dphi,

with (c_theta, c_phi) the derivatives of the current potential and its secular parts I / 2 pi, G / 2 pi, or the
covariant components the part supplies.  The surface's parameters are theta and phi, phi the cylindrical angle.

Quadrature.  One field period of the parameter domain is cut into ntheta x nphi panels, each with a p x p tensor
Gauss-Legendre rule.  A panel is used for a target when the target lies at least ETA times the panel's radius (the
largest distance from its centre to its corners and edge midpoints, with a margin) from its centre: the rule's error
is then about 1e-17 of the panel's share.  A panel nearer than that is cut into four, down to MAX_DEPTH levels.  The
other field periods are the same panels turned.  The panel counts resolve every mode of the series to RESOLVE of its
quantity's scale, from the amplitudes the series carries.

Node data are exact: positions, tangents and current at every node are the Fourier series summed there (separable
sums over m and n), never interpolated.

Separations.  Next to the surface the separation x - y of a target from a node is far smaller than either, and the
rounding of positions to binary64 (half an ulp of the major radius) would enter the field divided by the distance.  So
no separation is formed from positions in binary64.  Each kept panel holds its centre in double-double and its nodes as
offsets from the centre, the offsets by an exact difference formula of the series (exp(i m a) - 1 = -2 sin^2(m a / 2)
+ i sin(m a)), accurate relative to their own size; the target is turned into each field period in double-double; the
separation is (x - c) - (y - c), accurate relative to itself.  Panels built for one target (below the kept levels)
take the target's closest point on the surface, in double-double, as their origin instead.  Each panel's sum is added
to the target's with Neumaier compensation.

signed_distance(points) is the distance to the surface, negative inside, from the closest point found by Newton's
method on the series.  evaluate() refuses points inside or nearer than the quadrature resolves, or with nan_outside
gives NaN there (the solver's mode: a Newton step that leaves the exterior is a failed step).  How close to the surface
the field holds rounding level is measured (torux.sheet, docs/surface-current.md), not assumed.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..source import MU0

TAU = 2 * np.pi
MU0_4PI = MU0 / (4 * np.pi)
ORDER = 12             # Gauss-Legendre points per panel side
ETA = 2.0              # a panel is used for a target at least ETA radii from its centre
RADIUS_MARGIN = 1.1    # on the radius measured from corners and edge midpoints (the panel's edges bulge between them)
MAX_DEPTH = 26         # levels of subdivision below the root panels
CACHE = 2              # kept levels below the root (as far as CACHE_BYTES allows)
CACHE_BYTES = 96 * 1024 ** 2
RESOLVE = 1e-17        # each mode resolved by the root panels' rule to this fraction of its quantity's scale


class SurfaceCurrentDomainError(ValueError):
    """A target is outside the supported exterior domain."""

    def __init__(self, message, reason="boundary"):
        super().__init__(message)
        self.reason = reason


# ----------------------------------------------------------------------------- double-double constants
def _dd_of(x):
    """(hi, lo) of an mpmath number."""
    hi = float(x)
    import mpmath
    lo = float(x - mpmath.mpf(hi))
    return hi, lo


def _constants():
    import mpmath
    mpmath.mp.dps = 50
    p = mpmath.pi / 2
    p1 = float(p)
    p2 = float(p - mpmath.mpf(p1))
    p3 = float(p - mpmath.mpf(p1) - mpmath.mpf(p2))
    sin_c = np.array([_dd_of(mpmath.mpf((-1) ** j) / mpmath.factorial(2 * j + 1)) for j in range(15)])
    cos_c = np.array([_dd_of(mpmath.mpf((-1) ** j) / mpmath.factorial(2 * j)) for j in range(16)])
    return np.array([p1, p2, p3]), sin_c, cos_c


_PIO2, _SIN_C, _COS_C = _constants()
_TAU_DD = np.array([4 * _PIO2[0], 4 * _PIO2[1]])        # 2 pi in double-double (scaling by 4 is exact)


def _rotations_dd(nfp):
    """The rotations by 2 pi j / nfp about z, j = 0 .. nfp - 1, in double-double: (nfp, 2, 3, 3) hi and lo."""
    import mpmath
    mpmath.mp.dps = 50
    out = np.zeros((nfp, 2, 3, 3))
    for j in range(nfp):
        a = 2 * mpmath.pi * j / nfp
        ch, cl = _dd_of(mpmath.cos(a))
        sh, sl = _dd_of(mpmath.sin(a))
        out[j, 0] = [[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]]
        out[j, 1] = [[cl, -sl, 0.0], [sl, cl, 0.0], [0.0, 0.0, 0.0]]
    return out


# ----------------------------------------------------------------------------- double-double arithmetic
@njit(cache=True)
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@njit(cache=True)
def _quick(a, b):
    s = a + b
    return s, b - (s - a)


@njit(cache=True)
def _split(a):
    t = 134217729.0 * a
    hi = t - (t - a)
    return hi, a - hi


@njit(cache=True)
def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@njit(cache=True)
def _dd_add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    t, f = _two_sum(al, bl)
    e += t
    s, e = _quick(s, e)
    e += f
    return _quick(s, e)


@njit(cache=True)
def _dd_mul(ah, al, bh, bl):
    p, e = _two_prod(ah, bh)
    e += ah * bl + al * bh
    return _quick(p, e)


@njit(cache=True)
def _dd_mul_d(ah, al, b):
    p, e = _two_prod(ah, b)
    e += al * b
    return _quick(p, e)


@njit(cache=True)
def _dd_sincos(x, pio2, sin_c, cos_c):
    """cos x, sin x in double-double for a binary64 x (|x| up to a few turns): (ch, cl, sh, sl)."""
    k = math.floor(x / pio2[0] + 0.5)
    ph, pe = _two_prod(k, pio2[0])
    rh, rl = _two_sum(x, -ph)
    rl -= pe
    rh, rl = _quick(rh, rl)
    qh, qe = _two_prod(k, pio2[1])
    rh, rl = _dd_add(rh, rl, -qh, -qe)
    rh, rl = _dd_add(rh, rl, -k * pio2[2], 0.0)
    r2h, r2l = _dd_mul(rh, rl, rh, rl)
    sh, sl = sin_c[14, 0], sin_c[14, 1]
    for j in range(13, -1, -1):
        sh, sl = _dd_mul(sh, sl, r2h, r2l)
        sh, sl = _dd_add(sh, sl, sin_c[j, 0], sin_c[j, 1])
    sh, sl = _dd_mul(sh, sl, rh, rl)
    ch, cl = cos_c[15, 0], cos_c[15, 1]
    for j in range(14, -1, -1):
        ch, cl = _dd_mul(ch, cl, r2h, r2l)
        ch, cl = _dd_add(ch, cl, cos_c[j, 0], cos_c[j, 1])
    q = int(k) % 4
    if q < 0:
        q += 4
    if q == 0:
        return ch, cl, sh, sl
    if q == 1:
        return -sh, -sl, ch, cl
    if q == 2:
        return -ch, -cl, -sh, -sl
    return sh, sl, -ch, -cl


@njit(cache=True)
def _dd_cmul(a, b, out):
    """out = a b, complex double-double numbers as (re_hi, re_lo, im_hi, im_lo)."""
    p1h, p1l = _dd_mul(a[0], a[1], b[0], b[1])
    p2h, p2l = _dd_mul(a[2], a[3], b[2], b[3])
    p3h, p3l = _dd_mul(a[0], a[1], b[2], b[3])
    p4h, p4l = _dd_mul(a[2], a[3], b[0], b[1])
    rh, rl = _dd_add(p1h, p1l, -p2h, -p2l)
    ih, il = _dd_add(p3h, p3l, p4h, p4l)
    out[0] = rh
    out[1] = rl
    out[2] = ih
    out[3] = il


@njit(cache=True)
def _dd_div_i(ah, al, n):
    """(ah + al) / n in double-double, n a positive integer."""
    qh = ah / n
    p, e = _two_prod(qh, float(n))
    ql = (((ah - p) - e) + al) / n
    return _quick(qh, ql)


@njit(cache=True)
def _grid_dd(i, n, wh, wl):
    """i w / n in double-double: the i-th of n equal parts of w (the boundaries of every panel level, so that a
    panel's halves tile it to double-double rounding)."""
    ph, pl = _dd_mul_d(wh, wl, float(i))
    return _dd_div_i(ph, pl, n)


@njit(cache=True)
def _dd_phase(xh, xl, pio2, sin_c, cos_c, out):
    """exp(i (xh + xl)) in double-double, xl small against xh (first order in it): out (4,)."""
    ch, cl, sh, sl = _dd_sincos(xh, pio2, sin_c, cos_c)
    rh, rl = _dd_add(ch, cl, -sh * xl, 0.0)
    ih, il = _dd_add(sh, sl, ch * xl, 0.0)
    out[0] = rh
    out[1] = rl
    out[2] = ih
    out[3] = il


@njit(cache=True)
def _phase_tables(thh, thl, phh, phl, nfp, pio2, sin_c, cos_c, Et, Ep, cp):
    """Et[m] = exp(i m theta), Ep[n + N] = exp(-i n nfp phi) and cp = exp(i phi), all double-double (rows of 4), at a
    parameter point given in double-double."""
    M1 = Et.shape[0]
    N2 = Ep.shape[0]
    N = (N2 - 1) // 2
    tmp = np.empty(4)
    et = np.empty(4)
    _dd_phase(thh, thl, pio2, sin_c, cos_c, et)
    _dd_phase(phh, phl, pio2, sin_c, cos_c, cp)
    Et[0, 0], Et[0, 1], Et[0, 2], Et[0, 3] = 1.0, 0.0, 0.0, 0.0
    for m in range(1, M1):
        _dd_cmul(Et[m - 1], et, tmp)
        Et[m] = tmp
    eN = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(nfp):
        _dd_cmul(eN, cp, tmp)
        eN[:] = tmp
    eN[2] = -eN[2]
    eN[3] = -eN[3]
    Ep[N, 0], Ep[N, 1], Ep[N, 2], Ep[N, 3] = 1.0, 0.0, 0.0, 0.0
    for n in range(1, N + 1):
        _dd_cmul(Ep[N + n - 1], eN, tmp)
        Ep[N + n] = tmp
        Ep[N - n, 0] = tmp[0]
        Ep[N - n, 1] = tmp[1]
        Ep[N - n, 2] = -tmp[2]
        Ep[N - n, 3] = -tmp[3]


@njit(cache=True)
def _dd_point_tab(C2, Et, Ep, cp, out):
    """The surface point from the phase tables (_phase_tables), double-double: out (6,) = x, y, z as hi, lo."""
    M1 = C2.shape[1]
    N2 = C2.shape[2]
    val = np.zeros(4)                       # R hi, lo, Z hi, lo
    for m in range(M1):
        for q in range(2):
            sr, srl, si, sil = 0.0, 0.0, 0.0, 0.0
            for n in range(N2):
                cr = C2[q, m, n].real
                ci = C2[q, m, n].imag
                if cr == 0.0 and ci == 0.0:
                    continue
                ah, al = _dd_mul_d(Ep[n, 0], Ep[n, 1], cr)
                bh, bl = _dd_mul_d(Ep[n, 2], Ep[n, 3], ci)
                sr, srl = _dd_add(sr, srl, ah, al)
                sr, srl = _dd_add(sr, srl, -bh, -bl)
                ah, al = _dd_mul_d(Ep[n, 2], Ep[n, 3], cr)
                bh, bl = _dd_mul_d(Ep[n, 0], Ep[n, 1], ci)
                si, sil = _dd_add(si, sil, ah, al)
                si, sil = _dd_add(si, sil, bh, bl)
            ah, al = _dd_mul(Et[m, 0], Et[m, 1], sr, srl)
            bh, bl = _dd_mul(Et[m, 2], Et[m, 3], si, sil)
            vh, vl = _dd_add(val[2 * q], val[2 * q + 1], ah, al)
            vh, vl = _dd_add(vh, vl, -bh, -bl)
            val[2 * q] = vh
            val[2 * q + 1] = vl
    xh, xl = _dd_mul(val[0], val[1], cp[0], cp[1])
    yh, yl = _dd_mul(val[0], val[1], cp[2], cp[3])
    out[0] = xh
    out[1] = xl
    out[2] = yh
    out[3] = yl
    out[4] = val[2]
    out[5] = val[3]


@njit(cache=True)
def _dd_point(C2, nfp, thh, thl, phh, phl, pio2, sin_c, cos_c, out):
    """The surface point r(theta, phi) in double-double, the parameters in double-double: out (6,)."""
    Et = np.empty((C2.shape[1], 4))
    Ep = np.empty((C2.shape[2], 4))
    cp = np.empty(4)
    _phase_tables(thh, thl, phh, phl, nfp, pio2, sin_c, cos_c, Et, Ep, cp)
    _dd_point_tab(C2, Et, Ep, cp, out)


@njit(cache=True)
def _dd_turn(Rh, Rl, x0, x1, x2, out):
    """out (6,) = R^T x in double-double, R (hi Rh, lo Rl) a rotation about z, x binary64."""
    for a in range(3):
        sh, sl = 0.0, 0.0
        for k in range(3):
            xv = x0 if k == 0 else (x1 if k == 1 else x2)
            ph, pl = _two_prod(Rh[k, a], xv)
            pl += Rl[k, a] * xv
            sh, sl = _dd_add(sh, sl, ph, pl)
        out[2 * a] = sh
        out[2 * a + 1] = sl


# ----------------------------------------------------------------------------- the series
def sheet_series(part):
    """Complex coefficients C[q, m, n + N] of q = R, Z, c_theta, c_phi with f = Re sum C exp(i (m theta - n nfp phi)),
    m >= 0 (a mode with m < 0 is folded onto (-m, -n) by conjugation).  Returns (C, M, N)."""
    m = np.asarray(part.modes[:, 0], np.int64)
    n = np.asarray(part.modes[:, 1], np.int64)
    flip = m < 0
    mm, nn = np.where(flip, -m, m), np.where(flip, -n, n)
    M, N = int(mm.max()), int(np.abs(nn).max())

    def coef(c, s):
        z = (np.zeros(len(m)) if c is None else np.asarray(c, float)) - 1j * (0.0 if s is None else np.asarray(s, float))
        return np.where(flip, np.conj(z), z)

    C = np.zeros((4, M + 1, 2 * N + 1), complex)
    np.add.at(C[0], (mm, nn + N), coef(part.rmnc, part.rmns))
    np.add.at(C[1], (mm, nn + N), coef(part.zmnc, part.zmns))
    if part.covariant is None:
        k = np.zeros((M + 1, 2 * N + 1), complex)
        np.add.at(k, (mm, nn + N), coef(part.kappac, part.kappas))
        mg = np.arange(M + 1)[:, None]
        ng = np.arange(-N, N + 1)[None, :]
        C[2] = 1j * mg * k
        C[3] = -1j * ng * part.nfp * k
        C[2, 0, N] += part.I / TAU
        C[3, 0, N] += part.G / TAU
    else:
        cov = np.asarray(part.covariant, float)
        np.add.at(C[2], (mm, nn + N), coef(cov[0, 0], cov[0, 1]))
        np.add.at(C[3], (mm, nn + N), coef(cov[1, 0], cov[1, 1]))
    # the imaginary part of an m = n = 0 coefficient multiplies sin(0): it is no part of the series
    C[:, 0, N] = C[:, 0, N].real
    return C, M, N


def _sum_arrays(C, nfp):
    """The six series summed over n first (R, R_phi, Z, Z_phi, c_theta, c_phi), m and the phase factor of phi."""
    M1, N2 = C.shape[1], C.shape[2]
    N = (N2 - 1) // 2
    nv = -np.arange(-N, N + 1) * float(nfp)          # phase = m theta + nv phi
    CN = np.empty((6, M1, N2), complex)
    CN[0] = C[0]
    CN[1] = C[0] * (1j * nv)[None, :]
    CN[2] = C[1]
    CN[3] = C[1] * (1j * nv)[None, :]
    CN[4] = C[2]
    CN[5] = C[3]
    return np.ascontiguousarray(CN), np.arange(M1, dtype=float), nv


def grid_values(CN, mv, nv, theta, phi):
    """R, R_theta, R_phi, Z, Z_theta, Z_phi, c_theta, c_phi (8, A, B) on the tensor grid theta (A,) x phi (B,)."""
    Ep = np.exp(1j * np.outer(phi, nv))               # (B, N2)
    T = np.einsum("qmn,bn->qmb", CN, Ep)              # (6, M1, B)
    Et = np.exp(1j * np.outer(theta, mv))             # (A, M1)
    Etm = Et * (1j * mv)[None, :]
    out = np.empty((8, len(theta), len(phi)))
    out[0] = (Et @ T[0]).real
    out[1] = (Etm @ T[0]).real
    out[2] = (Et @ T[1]).real
    out[3] = (Et @ T[2]).real
    out[4] = (Etm @ T[2]).real
    out[5] = (Et @ T[3]).real
    out[6] = (Et @ T[4]).real
    out[7] = (Et @ T[5]).real
    return out


# ----------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def _values(CN, mv, nv, th, ph, out):
    """grid_values for small grids, inside numba: out (8, len(th), len(ph))."""
    M1 = CN.shape[1]
    N2 = CN.shape[2]
    A = th.shape[0]
    B = ph.shape[0]
    T = np.zeros((6, M1, B), np.complex128)
    for b in range(B):
        for n in range(N2):
            a = nv[n] * ph[b]
            e = complex(math.cos(a), math.sin(a))
            for q in range(6):
                for m in range(M1):
                    T[q, m, b] += CN[q, m, n] * e
    out[:, :, :] = 0.0
    for k in range(A):
        for m in range(M1):
            a = mv[m] * th[k]
            e = complex(math.cos(a), math.sin(a))
            em = e * complex(0.0, mv[m])
            for b in range(B):
                out[0, k, b] += (e * T[0, m, b]).real
                out[1, k, b] += (em * T[0, m, b]).real
                out[2, k, b] += (e * T[1, m, b]).real
                out[3, k, b] += (e * T[2, m, b]).real
                out[4, k, b] += (em * T[2, m, b]).real
                out[5, k, b] += (e * T[3, m, b]).real
                out[6, k, b] += (e * T[4, m, b]).real
                out[7, k, b] += (e * T[5, m, b]).real


@njit(cache=True)
def _positions(CN, mv, nv, th, ph, out):
    """Cartesian positions (len(th), len(ph), 3) on a tensor grid (R and Z only), binary64."""
    M1 = CN.shape[1]
    N2 = CN.shape[2]
    A = th.shape[0]
    B = ph.shape[0]
    T = np.zeros((2, M1, B), np.complex128)
    for b in range(B):
        for n in range(N2):
            a = nv[n] * ph[b]
            e = complex(math.cos(a), math.sin(a))
            for m in range(M1):
                T[0, m, b] += CN[0, m, n] * e
                T[1, m, b] += CN[2, m, n] * e
    for k in range(A):
        for b in range(B):
            R = 0.0
            Z = 0.0
            for m in range(M1):
                a = mv[m] * th[k]
                e = complex(math.cos(a), math.sin(a))
                R += (e * T[0, m, b]).real
                Z += (e * T[1, m, b]).real
            out[k, b, 0] = R * math.cos(ph[b])
            out[k, b, 1] = R * math.sin(ph[b])
            out[k, b, 2] = Z


@njit(cache=True)
def _em1(a):
    """exp(i a) - 1, accurate relative to its size for small a."""
    s = math.sin(0.5 * a)
    return complex(-2.0 * s * s, math.sin(a))


@njit(cache=True)
def _origin(CN, Et, Ep, D, S):
    """For an origin with phase tables Et, Ep (double-double rows, _phase_tables): D[q] = C_q exp(i (m theta0 - n nfp
    phi0)) (q = R, Z), S[q, n] = sum_m D[q, m, n], and R there (binary64)."""
    M1 = CN.shape[1]
    N2 = CN.shape[2]
    R0 = 0.0
    for m in range(M1):
        et = complex(Et[m, 0] + Et[m, 1], Et[m, 2] + Et[m, 3])
        for n in range(N2):
            e = et * complex(Ep[n, 0] + Ep[n, 1], Ep[n, 2] + Ep[n, 3])
            D[0, m, n] = CN[0, m, n] * e
            D[1, m, n] = CN[2, m, n] * e
            R0 += D[0, m, n].real
    for q in range(2):
        for n in range(N2):
            sm = 0j
            for m in range(M1):
                sm += D[q, m, n]
            S[q, n] = sm
    return R0


@njit(cache=True)
def _offsets(D, S, mv, nv, R0, ph0, a, b, nodes):
    """Positions of the nodes (th0 + a_k, ph0 + b_l) less the origin's, exact relative to their size, into
    nodes[k * p + l, 0:3]: r - r0 = Re sum D (exp(i (m a + nv b)) - 1), the bracket written as (exp(i m a) - 1)
    exp(i nv b) + (exp(i nv b) - 1)."""
    M1 = D.shape[1]
    N2 = D.shape[2]
    p = a.shape[0]
    T = np.zeros((2, M1, p), np.complex128)
    Eb = np.empty((N2, p), np.complex128)
    Fb = np.empty((N2, p), np.complex128)
    for l in range(p):
        for n in range(N2):
            x = nv[n] * b[l]
            Eb[n, l] = complex(math.cos(x), math.sin(x))
            Fb[n, l] = _em1(x)
    for q in range(2):
        for m in range(M1):
            for n in range(N2):
                d = D[q, m, n]
                if d == 0:
                    continue
                for l in range(p):
                    T[q, m, l] += d * Eb[n, l]
    for l in range(p):
        s0 = 0.0
        s1 = 0.0
        for n in range(N2):
            s0 += (Fb[n, l] * S[0, n]).real
            s1 += (Fb[n, l] * S[1, n]).real
        phl = ph0 + b[l]
        hb = math.sin(0.5 * b[l])
        dc = -2.0 * math.sin(ph0 + 0.5 * b[l]) * hb        # cos(ph0 + b) - cos(ph0)
        ds = 2.0 * math.cos(ph0 + 0.5 * b[l]) * hb         # sin(ph0 + b) - sin(ph0)
        c = math.cos(phl)
        s = math.sin(phl)
        for k in range(p):
            dR = s0
            dZ = s1
            for m in range(1, M1):
                f = _em1(mv[m] * a[k])
                dR += (f * T[0, m, l]).real
                dZ += (f * T[1, m, l]).real
            j = k * p + l
            nodes[j, 0] = dR * c + R0 * dc
            nodes[j, 1] = dR * s + R0 * ds
            nodes[j, 2] = dZ


@njit(cache=True)
def _weights(vals, ph, wt, wp, nodes):
    """The current weights mu0/(4 pi) K dS of the nodes into nodes[:, 3:6] from the series values (8, p, p)."""
    p = ph.shape[0]
    for l in range(p):
        c = math.cos(ph[l])
        s = math.sin(ph[l])
        for k in range(p):
            R = vals[0, k, l]
            Rt = vals[1, k, l]
            Rp = vals[2, k, l]
            Zt = vals[4, k, l]
            Zp = vals[5, k, l]
            ct = vals[6, k, l]
            cp = vals[7, k, l]
            W = MU0_4PI * wt[k] * wp[l]
            wR = (cp * Rt - ct * Rp) * W
            wP = (-ct * R) * W
            wZ = (cp * Zt - ct * Zp) * W
            j = k * p + l
            nodes[j, 3] = wR * c - wP * s
            nodes[j, 4] = wR * s + wP * c
            nodes[j, 5] = wZ


@njit(cache=True)
def _accumulate(acc, v0, v1, v2, nodes, grad, potential):
    """The panel's sum at the target whose separation from the panel's origin is v; nodes hold offsets from it."""
    for j in range(nodes.shape[0]):
        d0 = v0 - nodes[j, 0]
        d1 = v1 - nodes[j, 1]
        d2 = v2 - nodes[j, 2]
        r2 = d0 * d0 + d1 * d1 + d2 * d2
        ir = 1.0 / math.sqrt(r2)
        ir3 = ir * ir * ir
        w0 = nodes[j, 3]
        w1 = nodes[j, 4]
        w2 = nodes[j, 5]
        c0 = w1 * d2 - w2 * d1
        c1 = w2 * d0 - w0 * d2
        c2 = w0 * d1 - w1 * d0
        acc[0] += c0 * ir3
        acc[1] += c1 * ir3
        acc[2] += c2 * ir3
        if potential:
            acc[3] += w0 * ir
            acc[4] += w1 * ir
            acc[5] += w2 * ir
        if grad:
            f = 3.0 * ir3 * ir * ir
            e0 = c0 * f
            e1 = c1 * f
            e2 = c2 * f
            u0 = w0 * ir3
            u1 = w1 * ir3
            u2 = w2 * ir3
            acc[6] -= e0 * d0
            acc[7] += -u2 - e0 * d1
            acc[8] += u1 - e0 * d2
            acc[9] += u2 - e1 * d0
            acc[10] -= e1 * d1
            acc[11] += -u0 - e1 * d2
            acc[12] += -u1 - e2 * d0
            acc[13] += u0 - e2 * d1
            acc[14] -= e2 * d2


@njit(cache=True)
def _neumaier(tot, cmp, v):
    for k in range(tot.shape[0]):
        s = tot[k]
        u = s + v[k]
        if abs(s) >= abs(v[k]):
            cmp[k] += (s - u) + v[k]
        else:
            cmp[k] += (v[k] - u) + s
        tot[k] = u


@njit(cache=True)
def _children(CN, mv, nv, th0, dth, ph0, dph, grid, cen, rad):
    """Centres and radii of the four halves of a panel, from the panel's 5 x 5 grid of points (binary64)."""
    tg = np.empty(5)
    pg = np.empty(5)
    for k in range(5):
        tg[k] = th0 + 0.25 * k * dth
        pg[k] = ph0 + 0.25 * k * dph
    _positions(CN, mv, nv, tg, pg, grid)
    for a in range(2):
        for b in range(2):
            c = 2 * a + b
            ic, jc = 2 * a + 1, 2 * b + 1
            r = 0.0
            for i in range(2 * a, 2 * a + 3):
                for j in range(2 * b, 2 * b + 3):
                    d = 0.0
                    for e in range(3):
                        d += (grid[i, j, e] - grid[ic, jc, e]) ** 2
                    r = max(r, math.sqrt(d))
            for e in range(3):
                cen[c, e] = grid[ic, jc, e]
            rad[c] = RADIUS_MARGIN * r


@njit(cache=True)
def _origin_panel(CN, C2, nfp, thh, thl, phh, phl, pio2, sin_c, cos_c, Et, Ep, cp, D, S, y):
    """The origin data of a panel or target at (theta, phi) given in double-double: D, S (_origin), its position y in
    double-double (6,); returns R there (binary64)."""
    _phase_tables(thh, thl, phh, phl, nfp, pio2, sin_c, cos_c, Et, Ep, cp)
    _dd_point_tab(C2, Et, Ep, cp, y)
    return _origin(CN, Et, Ep, D, S)


@njit(parallel=True, cache=True)
def _build_keys(CN, mv, nv, C2, nfp, g, gw, ntheta, nphi, keys, tau, pio2, sin_c, cos_c, cdd, nodes):
    """Panels by key (level, i, J), J counting phi from 0 over the whole turn: their lower corners in double-double
    (cdd (P, 6)) and their nodes as offsets from the corner with their weights (nodes (P, p * p, 6)).  Boundaries:
    i tau / (ntheta 2^level) and J tau / (nfp nphi 2^level) in double-double, so that a panel's halves tile it and
    the field periods tile the turn, to double-double rounding."""
    p = g.shape[0]
    M1 = CN.shape[1]
    N2 = CN.shape[2]
    for idx in prange(keys.shape[0]):
        lv = keys[idx, 0]
        i = keys[idx, 1]
        J = keys[idx, 2]
        na = ntheta << lv
        nb = (nphi << lv) * nfp
        tLh, tLl = _grid_dd(i, na, tau[0], tau[1])
        tRh, tRl = _grid_dd(i + 1, na, tau[0], tau[1])
        pLh, pLl = _grid_dd(J, nb, tau[0], tau[1])
        pRh, pRl = _grid_dd(J + 1, nb, tau[0], tau[1])
        wth, wtl = _dd_add(tRh, tRl, -tLh, -tLl)
        wph, wpl = _dd_add(pRh, pRl, -pLh, -pLl)
        Et = np.empty((M1, 4))
        Ep = np.empty((N2, 4))
        cp = np.empty(4)
        D = np.empty((2, M1, N2), np.complex128)
        S = np.empty((2, N2), np.complex128)
        y = np.empty(6)
        R0 = _origin_panel(CN, C2, nfp, tLh, tLl, pLh, pLl, pio2, sin_c, cos_c, Et, Ep, cp, D, S, y)
        a = np.empty(p)
        b = np.empty(p)
        th = np.empty(p)
        ph = np.empty(p)
        wt = np.empty(p)
        wp = np.empty(p)
        for k in range(p):
            a[k] = g[k] * wth + g[k] * wtl
            b[k] = g[k] * wph + g[k] * wpl
            th[k] = tLh + a[k]
            ph[k] = pLh + b[k]
            wt[k] = gw[k] * wth
            wp[k] = gw[k] * wph
        _offsets(D, S, mv, nv, R0, pLh, a, b, nodes[idx])
        vals = np.empty((8, p, p))
        _values(CN, mv, nv, th, ph, vals)
        _weights(vals, ph, wt, wp, nodes[idx])
        cdd[idx] = y


@njit(parallel=True, cache=True)
def _plan(points, rdd, CN, mv, nv, ntheta, nphi, nfp, eta, cdepth, offsets, cen, rad, max_depth, keys, nkeys,
          status):
    """The panels below the kept levels each target uses: keys[t, k] = (level, i, J), k < nkeys[t]; status[t] 1 where
    a panel at max_depth is still too near the target, 2 where it needs more panels than keys holds."""
    T = points.shape[0]
    MAXB = keys.shape[1]
    nroot = ntheta * nphi
    S_ = 4 * max_depth + 8
    for t in prange(T):
        st_l = np.empty(S_, np.int64)
        st_i = np.empty(S_, np.int64)
        st_j = np.empty(S_, np.int64)
        st_c = np.empty((S_, 3))
        st_r = np.empty(S_)
        grid = np.empty((5, 5, 3))
        ccen = np.empty((4, 3))
        crad = np.empty(4)
        x0g = points[t, 0]
        x1g = points[t, 1]
        x2g = points[t, 2]
        nk = 0
        for r in range(nfp):
            R = rdd[r, 0]
            x0 = R[0, 0] * x0g + R[1, 0] * x1g + R[2, 0] * x2g
            x1 = R[0, 1] * x0g + R[1, 1] * x1g + R[2, 1] * x2g
            x2 = R[0, 2] * x0g + R[1, 2] * x1g + R[2, 2] * x2g
            for root in range(nroot):
                size = 1
                st_l[0] = 0
                st_i[0] = root // nphi
                st_j[0] = root % nphi
                while size > 0:
                    size -= 1
                    lv = st_l[size]
                    i = st_i[size]
                    j = st_j[size]
                    kept = lv <= cdepth
                    if kept:
                        scale = 1 << lv
                        idx = offsets[lv] + i * nphi * scale + j
                        rr = rad[idx]
                        dd = (x0 - cen[idx, 0]) ** 2 + (x1 - cen[idx, 1]) ** 2 + (x2 - cen[idx, 2]) ** 2
                    else:
                        rr = st_r[size]
                        dd = (x0g - st_c[size, 0]) ** 2 + (x1g - st_c[size, 1]) ** 2 + (x2g - st_c[size, 2]) ** 2
                    if dd >= (eta * rr) ** 2:
                        if not kept:
                            if nk < MAXB:
                                keys[t, nk, 0] = lv
                                keys[t, nk, 1] = i
                                keys[t, nk, 2] = j
                            nk += 1
                    elif lv >= max_depth:
                        status[t] = 1
                    elif kept and lv + 1 <= cdepth:
                        for aa in range(2):
                            for bb in range(2):
                                st_l[size] = lv + 1
                                st_i[size] = 2 * i + aa
                                st_j[size] = 2 * j + bb
                                size += 1
                    else:
                        scale = 1 << lv
                        jj = j + r * nphi * scale if kept else j       # built panels count phi over the whole turn
                        dth = TAU / (ntheta * scale)
                        dph = TAU / (nfp * nphi * scale)
                        _children(CN, mv, nv, i * dth, dth, jj * dph, dph, grid, ccen, crad)
                        for aa in range(2):
                            for bb in range(2):
                                st_l[size] = lv + 1
                                st_i[size] = 2 * i + aa
                                st_j[size] = 2 * jj + bb
                                for e in range(3):
                                    st_c[size, e] = ccen[2 * aa + bb, e]
                                st_r[size] = crad[2 * aa + bb]
                                size += 1
        nkeys[t] = min(nk, MAXB)
        if nk > MAXB:
            status[t] = 2


@njit(parallel=True, cache=True)
def _sum(points, rdd, ntheta, nphi, nfp, eta, cdepth, offsets, cen, rad, cdd, nodes, bidx, nkeys, bcdd, bnodes,
         grad, potential, out):
    """B, A, grad B (out, (T, 15)) at the points: the kept panels, period by period in each period's frame with the
    target turned there in double-double, and the built panels bidx[t, :nkeys[t]] (rows of bcdd, bnodes) in the
    global frame.  Separations (x - c) - (y - c), c a panel's corner in double-double."""
    T = points.shape[0]
    nroot = ntheta * nphi
    S_ = 4 * cdepth + 8
    for t in prange(T):
        tot = np.zeros(15)
        cmp = np.zeros(15)
        ptot = np.zeros(15)
        pcmp = np.zeros(15)
        acc = np.zeros(15)
        glob = np.zeros(15)
        st_l = np.empty(S_, np.int64)
        st_i = np.empty(S_, np.int64)
        st_j = np.empty(S_, np.int64)
        xr = np.empty(6)
        x0g = points[t, 0]
        x1g = points[t, 1]
        x2g = points[t, 2]
        for r in range(nfp):
            _dd_turn(rdd[r, 0], rdd[r, 1], x0g, x1g, x2g, xr)
            x0, x1, x2 = xr[0], xr[2], xr[4]
            ptot[:] = 0.0
            pcmp[:] = 0.0
            for root in range(nroot):
                size = 1
                st_l[0] = 0
                st_i[0] = root // nphi
                st_j[0] = root % nphi
                while size > 0:
                    size -= 1
                    lv = st_l[size]
                    i = st_i[size]
                    j = st_j[size]
                    scale = 1 << lv
                    idx = offsets[lv] + i * nphi * scale + j
                    dd = (x0 - cen[idx, 0]) ** 2 + (x1 - cen[idx, 1]) ** 2 + (x2 - cen[idx, 2]) ** 2
                    if dd >= (eta * rad[idx]) ** 2:
                        acc[:] = 0.0
                        w0 = (xr[0] - cdd[idx, 0]) + (xr[1] - cdd[idx, 1])
                        w1 = (xr[2] - cdd[idx, 2]) + (xr[3] - cdd[idx, 3])
                        w2 = (xr[4] - cdd[idx, 4]) + (xr[5] - cdd[idx, 5])
                        _accumulate(acc, w0, w1, w2, nodes[idx], grad, potential)
                        _neumaier(ptot, pcmp, acc)
                    elif lv < cdepth:
                        for aa in range(2):
                            for bb in range(2):
                                st_l[size] = lv + 1
                                st_i[size] = 2 * i + aa
                                st_j[size] = 2 * j + bb
                                size += 1
            for k in range(15):
                ptot[k] += pcmp[k]
            R = rdd[r, 0]
            for aa in range(3):
                glob[aa] = R[aa, 0] * ptot[0] + R[aa, 1] * ptot[1] + R[aa, 2] * ptot[2]
                glob[3 + aa] = R[aa, 0] * ptot[3] + R[aa, 1] * ptot[4] + R[aa, 2] * ptot[5]
            if grad:
                for aa in range(3):
                    for bb in range(3):
                        sg = 0.0
                        for c in range(3):
                            for d in range(3):
                                sg += R[aa, c] * ptot[6 + 3 * c + d] * R[bb, d]
                        glob[6 + 3 * aa + bb] = sg
            else:
                for k in range(6, 15):
                    glob[k] = 0.0
            _neumaier(tot, cmp, glob)
        for k in range(nkeys[t]):
            u = bidx[t, k]
            acc[:] = 0.0
            w0 = (x0g - bcdd[u, 0]) - bcdd[u, 1]
            w1 = (x1g - bcdd[u, 2]) - bcdd[u, 3]
            w2 = (x2g - bcdd[u, 4]) - bcdd[u, 5]
            _accumulate(acc, w0, w1, w2, bnodes[u], grad, potential)
            _neumaier(tot, cmp, acc)
        for k in range(15):
            out[t, k] = tot[k] + cmp[k]


@njit(cache=True)
def _rz2(C2, mv, nv, th, ph):
    """R, Z and their derivatives to second order at (th, ph): (R, Rt, Rp, Rtt, Rtp, Rpp) and the same of Z."""
    M1 = C2.shape[1]
    N2 = C2.shape[2]
    out = np.zeros(12)
    for q in range(2):
        for m in range(M1):
            a = mv[m] * th
            et = complex(math.cos(a), math.sin(a))
            s0 = 0j
            s1 = 0j
            s2 = 0j
            for n in range(N2):
                b = nv[n] * ph
                e = C2[q, m, n] * complex(math.cos(b), math.sin(b))
                s0 += e
                s1 += e * complex(0.0, nv[n])
                s2 -= e * nv[n] * nv[n]
            im = complex(0.0, mv[m])
            out[6 * q + 0] += (et * s0).real
            out[6 * q + 1] += (et * im * s0).real
            out[6 * q + 2] += (et * s1).real
            out[6 * q + 3] += (et * im * im * s0).real
            out[6 * q + 4] += (et * im * s1).real
            out[6 * q + 5] += (et * s2).real
    return out


@njit(cache=True)
def _cart(v, q, ph):
    """Cartesian position, r_theta, r_phi, r_tt, r_tp, r_pp from the cylindrical series values v (12,)."""
    c, s = math.cos(ph), math.sin(ph)
    R, Rt, Rp, Rtt, Rtp, Rpp = v[0], v[1], v[2], v[3], v[4], v[5]
    Z, Zt, Zp, Ztt, Ztp, Zpp = v[6], v[7], v[8], v[9], v[10], v[11]
    q[0, 0], q[0, 1], q[0, 2] = R * c, R * s, Z
    q[1, 0], q[1, 1], q[1, 2] = Rt * c, Rt * s, Zt
    q[2, 0], q[2, 1], q[2, 2] = Rp * c - R * s, Rp * s + R * c, Zp
    q[3, 0], q[3, 1], q[3, 2] = Rtt * c, Rtt * s, Ztt
    q[4, 0], q[4, 1], q[4, 2] = Rtp * c - Rt * s, Rtp * s + Rt * c, Ztp
    q[5, 0], q[5, 1], q[5, 2] = (Rpp - R) * c - 2 * Rp * s, (Rpp - R) * s + 2 * Rp * c, Zpp


@njit(parallel=True, cache=True)
def _foot(points, th0, ph0, C2, mv, nv, iters, res):
    """Closest points of the surface by Newton on |x - r(theta, phi)|^2 from (th0, ph0): res (T, 4) = theta, phi,
    signed distance (by the outward normal there), and the gradient's size relative to the distance at the end."""
    T = points.shape[0]
    for t in prange(T):
        th, ph = th0[t], ph0[t]
        q = np.empty((6, 3))
        x = points[t]
        gnorm = np.inf
        for it in range(iters):
            v = _rz2(C2, mv, nv, th, ph)
            _cart(v, q, ph)
            e0, e1, e2 = x[0] - q[0, 0], x[1] - q[0, 1], x[2] - q[0, 2]
            g1 = -(e0 * q[1, 0] + e1 * q[1, 1] + e2 * q[1, 2])
            g2 = -(e0 * q[2, 0] + e1 * q[2, 1] + e2 * q[2, 2])
            h11 = q[1, 0] ** 2 + q[1, 1] ** 2 + q[1, 2] ** 2 - (e0 * q[3, 0] + e1 * q[3, 1] + e2 * q[3, 2])
            h12 = (q[1, 0] * q[2, 0] + q[1, 1] * q[2, 1] + q[1, 2] * q[2, 2]
                   - (e0 * q[4, 0] + e1 * q[4, 1] + e2 * q[4, 2]))
            h22 = q[2, 0] ** 2 + q[2, 1] ** 2 + q[2, 2] ** 2 - (e0 * q[5, 0] + e1 * q[5, 1] + e2 * q[5, 2])
            det = h11 * h22 - h12 * h12
            if det > 0 and h11 > 0:
                d1 = (h22 * g1 - h12 * g2) / det
                d2 = (h11 * g2 - h12 * g1) / det
            else:
                # gradient step scaled by the metric (the Hessian is not positive here)
                d1 = g1 / max(q[1, 0] ** 2 + q[1, 1] ** 2 + q[1, 2] ** 2, 1e-300)
                d2 = g2 / max(q[2, 0] ** 2 + q[2, 1] ** 2 + q[2, 2] ** 2, 1e-300)
            # at most a tenth of a radian in theta and the same in phi per step
            lim = max(abs(d1) / 0.1, abs(d2) / 0.1, 1.0)
            th -= d1 / lim
            ph -= d2 / lim
            gnorm = math.sqrt(g1 * g1 + g2 * g2)
            if abs(d1) < 1e-15 and abs(d2) < 1e-15:
                break
        v = _rz2(C2, mv, nv, th, ph)
        _cart(v, q, ph)
        e0, e1, e2 = x[0] - q[0, 0], x[1] - q[0, 1], x[2] - q[0, 2]
        n0 = q[2, 1] * q[1, 2] - q[2, 2] * q[1, 1]
        n1 = q[2, 2] * q[1, 0] - q[2, 0] * q[1, 2]
        n2 = q[2, 0] * q[1, 1] - q[2, 1] * q[1, 0]
        nn = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        dist = math.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
        side = e0 * n0 + e1 * n1 + e2 * n2
        res[t, 0] = th
        res[t, 1] = ph
        res[t, 2] = dist if side >= 0 else -dist
        tn = math.sqrt(q[1, 0] ** 2 + q[1, 1] ** 2 + q[1, 2] ** 2) + math.sqrt(q[2, 0] ** 2 + q[2, 1] ** 2 + q[2, 2] ** 2)
        res[t, 3] = gnorm / max(dist * tn, 1e-300) if nn > 0 else np.inf


# ----------------------------------------------------------------------------- the sheet
def _panel_counts(C, M, N, nfp, order, lengths):
    """Root panels (ntheta, nphi) per field period: each mode's share of the series resolved by the order-p Gauss rule
    to RESOLVE of its scale (the rule's error on exp(i w x) over [-1, 1] is ~ w^2p 2^(2p+1) (p!)^4 / ((2p+1) (2p)!^3)),
    and panels no more than twice as long one way as the other."""
    scale = np.array([max(np.abs(C[q]).sum(), 1e-300) for q in range(4)])
    rel = (np.abs(C) / scale[:, None, None]).max(axis=0)       # each mode's share of its quantity, the largest of four
    p = order
    lg = math.lgamma
    logc = (2 * p + 1) * math.log(2) + 4 * lg(p + 1) - math.log(2 * p + 1) - 3 * lg(2 * p + 1)

    def worst(freq, count):
        w = np.pi * freq / count
        with np.errstate(divide="ignore"):
            lw = np.where(w > 0, 2 * p * np.log(np.maximum(w, 1e-300)), -np.inf)
        return np.exp(lw + logc)

    mfreq = np.arange(M + 1)[:, None] * np.ones((1, 2 * N + 1))
    nfreq = np.abs(np.arange(-N, N + 1))[None, :] * np.ones((M + 1, 1))
    ntheta = 6
    while ntheta < 4096 and np.max(rel * worst(mfreq, ntheta)) > RESOLVE:
        ntheta += 1
    nphi = 6 if N > 0 else 4
    while nphi < 4096 and np.max(rel * worst(nfreq, nphi)) > RESOLVE:
        nphi += 1
    Lt, Lp = lengths
    while Lt / ntheta > 2 * Lp / nphi:
        ntheta += 1
    while Lp / nphi > 2 * Lt / ntheta:
        nphi += 1
    return int(ntheta), int(nphi)


class SheetField:
    """The exterior field of one SurfaceCurrent part.  evaluate(points, grad, potential) -> (T, 15): B, A, grad B
    (row-major dB_i/dx_j), Cartesian, SI."""

    def __init__(self, part, order=ORDER, eta=ETA, cache_bytes=None):
        if not isinstance(order, (int, np.integer)) or order < 4:
            raise ValueError("sheet_order must be an integer, at least 4")
        if not np.isfinite(eta) or eta < 1.2:
            raise ValueError("sheet_eta must be at least 1.2")
        self.part = part
        self.nfp = int(part.nfp)
        self.order = int(order)
        self.eta = float(eta)
        C, M, N = sheet_series(part)
        self.M, self.N = M, N
        self.CN, self.mv, self.nv = _sum_arrays(C, self.nfp)
        self.C2 = np.ascontiguousarray(self.CN[[0, 2]])
        x, w = np.polynomial.legendre.leggauss(self.order)
        self.g, self.gw = (x + 1) / 2, w / 2
        self.rdd = np.ascontiguousarray(_rotations_dd(self.nfp))
        self.rotations = np.ascontiguousarray(self.rdd[:, 0])
        # sections, lengths and orientation; the panel counts
        nt = max(8 * (M + 1), 256)
        npp = max(8 * (N + 1), 64)
        th = TAU * np.arange(nt) / nt
        ph = TAU * np.arange(npp) / (npp * self.nfp)
        V = grid_values(self.CN, self.mv, self.nv, th, ph)
        R, Rt, Rp, Z, Zt, Zp = V[:6]
        if R.min() <= 0:
            raise ValueError("SurfaceCurrent requires a boundary at positive R")
        area = np.sum(R * Zt - Z * Rt, axis=0) * 0.5 * TAU / nt          # the section's signed area at each phi
        if np.any(area <= 0):
            raise ValueError("SurfaceCurrent requires each constant-phi section to run counterclockwise in (R, Z) as "
                             "theta increases")
        Lt = float(np.mean(np.sum(np.hypot(Rt, Zt), axis=0) * TAU / nt))
        Lp = float(np.mean(np.sum(np.sqrt(Rp ** 2 + R ** 2 + Zp ** 2), axis=1) * TAU / (npp * self.nfp)))
        self.lengths = (Lt, Lp)
        vol = float(np.mean(np.sum(R * (R * Zt - Z * Rt), axis=0) * 0.5 * TAU / nt) * TAU)   # Pappus over phi
        R0 = float(np.mean(R))
        self.size = dict(R0=R0, area=float(np.mean(area)), volume=vol,
                         minor_radius=float(np.sqrt(abs(vol) / (2 * np.pi ** 2 * R0))),
                         Rmax=float(np.max(np.hypot(R, Z))), Lt=Lt, Lp=Lp)
        self.ntheta, self.nphi = _panel_counts(C, M, N, self.nfp, self.order, self.lengths)
        cache_bytes = CACHE_BYTES if cache_bytes is None else int(cache_bytes)
        per_panel = 8 * (6 * self.order ** 2 + 10)
        depth = 0
        for lv in range(1, CACHE + 1):
            panels = self.ntheta * self.nphi * (4 ** (lv + 1) - 1) // 3
            if panels * per_panel > cache_bytes:
                break
            depth = lv
        self.cache_depth = depth
        self._cache = None
        self._tree = None
        self.max_depth = MAX_DEPTH
        self.built = 0                 # panels built for targets so far (a measure of the near-field work)

    # ------------------------------------------------------------------ the kept panels
    def _build_cache(self):
        p = self.order
        cens, rads, cdds, nodes, offsets = [], [], [], [], [0]
        for lv in range(self.cache_depth + 1):
            na, nb = self.ntheta << lv, self.nphi << lv
            dth, dph = TAU / na, TAU / (self.nfp * nb)
            cdd = np.empty((na * nb, 6))
            nd = np.empty((na * nb, p * p, 6))
            ii, jj = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
            keys = np.ascontiguousarray(np.stack([np.full(na * nb, lv), ii.ravel(), jj.ravel()], -1).astype(np.int64))
            _build_keys(self.CN, self.mv, self.nv, self.C2, self.nfp, self.g, self.gw, self.ntheta, self.nphi, keys,
                        _TAU_DD, _PIO2, _SIN_C, _COS_C, cdd, nd)
            # centres and radii from the grid of corners, edge midpoints and centres (binary64: the acceptance test)
            tg = 0.5 * dth * np.arange(2 * na + 1)
            pg = 0.5 * dph * np.arange(2 * nb + 1)
            P = np.empty((len(tg), len(pg), 3))
            _positions(self.CN, self.mv, self.nv, tg, pg, P)
            cen = P[1::2, 1::2].reshape(-1, 3)
            rad = np.zeros((na, nb))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    q = P[1 + di:2 * na + 1 + di:2, 1 + dj:2 * nb + 1 + dj:2]
                    rad = np.maximum(rad, np.linalg.norm(q - P[1::2, 1::2], axis=-1))
            rad = RADIUS_MARGIN * rad.ravel()
            # nodes lie inside, but a strongly curved panel's nodes can stand beyond the corners' circle
            corner = cdd[:, 0::2]
            rad = np.maximum(rad, RADIUS_MARGIN * np.max(np.linalg.norm(nd[:, :, :3] + (corner - cen)[:, None, :],
                                                                      axis=-1), axis=1))
            cens.append(cen)
            rads.append(rad)
            cdds.append(cdd)
            nodes.append(nd)
            offsets.append(offsets[-1] + na * nb)
        self._cache = (np.ascontiguousarray(np.concatenate(cens)), np.ascontiguousarray(np.concatenate(rads)),
                       np.ascontiguousarray(np.concatenate(cdds)), np.ascontiguousarray(np.concatenate(nodes)),
                       np.asarray(offsets, np.int64))

    @property
    def panel_bytes(self):
        return 0 if self._cache is None else sum(a.nbytes for a in self._cache[:4])

    # ------------------------------------------------------------------ distance to the surface
    def _samples(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            nt = max(4 * self.ntheta, 4 * (self.M + 1), 64)
            npp = max(4 * self.nphi, 4 * (self.N + 1), 16)
            # one period and a margin of the neighbouring ones, so that the nearest sample is found across the cut
            th = TAU * np.arange(nt) / nt
            ph = TAU * (np.arange(-npp // 2, npp + npp // 2) / npp) / self.nfp
            V = grid_values(self.CN, self.mv, self.nv, th, ph)
            R, Rt, Rp, Z, Zt, Zp = V[:6]
            c, s = np.cos(ph)[None, :], np.sin(ph)[None, :]
            P = np.stack([R * c, R * s, Z], -1)
            rt = np.stack([Rt * c, Rt * s, Zt], -1)
            rp = np.stack([Rp * c - R * s, Rp * s + R * c, Zp], -1)
            nrm = np.cross(rp, rt)
            nrm /= np.linalg.norm(nrm, axis=-1)[..., None]
            TH, PH = np.meshgrid(th, ph, indexing="ij")
            step = max(float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=-1))),
                       float(np.max(np.linalg.norm(np.diff(P, axis=1), axis=-1))))
            self._tree = (cKDTree(P.reshape(-1, 3)), TH.ravel(), PH.ravel(), step, nrm.reshape(-1, 3),
                          P.reshape(-1, 3))
        return self._tree

    def signed_distance(self, points, with_foot=False):
        """Distance (m) of each point from the surface, negative inside.  Within FAR sample spacings of the surface
        it is the distance to the closest point found by Newton's method on the series (with_foot: also (T, 4) with
        that point's theta and phi, phi the point's own cylindrical angle, the distance, and the distance's gradient
        relative to its size there, small at a converged closest point); farther, the distance to the nearest
        sample, which overstates it by less than a spacing, and theta, phi NaN."""
        pts = np.ascontiguousarray(np.asarray(points, float).reshape(-1, 3))
        if not len(pts):
            return (np.zeros(0), np.zeros((0, 4))) if with_foot else np.zeros(0)
        tree, TH, PH, step, nrm, P = self._samples()
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        k = np.floor(np.mod(phi, TAU) / (TAU / self.nfp)).astype(np.int64) % self.nfp
        loc = np.einsum("tba,tb->ta", self.rotations[k], pts)
        ds, idx = tree.query(loc, k=1)
        res = np.full((len(pts), 4), np.nan)
        side = np.einsum("ta,ta->t", loc - P[idx], nrm[idx])
        res[:, 2] = np.where(side >= 0, ds, -ds)
        near = ds < self.FAR * step
        if near.any():
            sub = np.empty((int(near.sum()), 4))
            _foot(np.ascontiguousarray(loc[near]), TH[idx[near]], PH[idx[near]], self.C2, self.mv, self.nv, 40, sub)
            res[near] = sub
        if with_foot:
            res[:, 1] += k * TAU / self.nfp
            return res[:, 2].copy(), res
        return res[:, 2].copy()

    FAR = 4.0

    def clearance(self, r, z, phi):
        """Signed distance from the surface of the point at cylindrical (r, phi, z), m (positive outside)."""
        return float(self.signed_distance([[r * math.cos(phi), r * math.sin(phi), z]])[0])

    # ------------------------------------------------------------------ the field
    PLAN = 768             # built panels a target may use before it is planned alone with more room
    CHUNK = 512            # targets planned together (their shared panels built once)
    BUILT_BYTES = 192 * 1024 ** 2

    def evaluate(self, points, grad=True, potential=True, nan_outside=False):
        """(T, 15): B, A, grad B at Cartesian points.  Points inside the surface or too near it for the quadrature
        raise SurfaceCurrentDomainError, or with nan_outside give NaN rows."""
        pts = np.ascontiguousarray(np.asarray(points, float).reshape(-1, 3))
        out = np.zeros((len(pts), 15))
        if not len(pts):
            return out
        d = self.signed_distance(pts)
        bad = ~(d > self.hard_limit)
        if bad.any() and not nan_outside:
            i = int(np.argmax(bad))
            raise SurfaceCurrentDomainError(f"target {i} is inside or too close to the surface current boundary")
        if self._cache is None:
            self._build_cache()
        good = np.nonzero(~bad)[0]
        res = np.zeros((len(good), 15))
        failed = np.zeros(len(good), bool)
        # near targets first in their own chunks (their panels are many and shared), the rest without built panels
        order = good[np.argsort(d[good], kind="stable")]
        pos = {int(k): n for n, k in enumerate(good)}
        k0 = 0
        while k0 < len(order):
            k1 = min(len(order), k0 + self.CHUNK)
            idx = order[k0:k1]
            o, f = self._chunk(pts[idx], bool(grad), bool(potential), self.PLAN)
            rows = np.array([pos[int(k)] for k in idx])
            res[rows] = o
            failed[rows] = f
            k0 = k1
        if failed.any():
            if not nan_outside:
                raise SurfaceCurrentDomainError("target is too close for the surface current quadrature to resolve",
                                                "quadrature_limit")
            res[failed] = np.nan
        out[good] = res
        out[bad] = np.nan
        return out

    def _chunk(self, pts, grad, potential, room):
        cen, rad, cdd, nodes, offsets = self._cache
        pts = np.ascontiguousarray(pts)
        T = len(pts)
        keys = np.zeros((T, room, 3), np.int64)
        nkeys = np.zeros(T, np.int64)
        status = np.zeros(T, np.int64)
        _plan(pts, self.rdd, self.CN, self.mv, self.nv, self.ntheta, self.nphi, self.nfp, self.eta, self.cache_depth,
              offsets, cen, rad, self.max_depth, keys, nkeys, status)
        out = np.zeros((T, 15))
        failed = status == 1
        more = status == 2
        if more.any():
            o, f = self._chunk(pts[more], grad, potential, 4 * room)
            out[more], failed[more] = o, f
        ok = ~more
        mask = np.arange(room)[None, :] < nkeys[:, None]
        mask[~ok] = False
        flat = keys[mask]
        per = 8 * (6 * self.order ** 2 + 6)
        if len(flat):
            uniq, inv = np.unique(flat, axis=0, return_inverse=True)
            if len(uniq) * per > self.BUILT_BYTES and T > 1:
                # too many distinct panels to hold at once: halve the targets
                h = T // 2
                o1, f1 = self._chunk(pts[:h], grad, potential, room)
                o2, f2 = self._chunk(pts[h:], grad, potential, room)
                return np.concatenate([o1, o2]), np.concatenate([f1, f2]) | failed
            bcdd = np.empty((len(uniq), 6))
            bnodes = np.empty((len(uniq), self.order ** 2, 6))
            _build_keys(self.CN, self.mv, self.nv, self.C2, self.nfp, self.g, self.gw, self.ntheta, self.nphi,
                        np.ascontiguousarray(uniq), _TAU_DD, _PIO2, _SIN_C, _COS_C, bcdd, bnodes)
            bidx = np.zeros((T, room), np.int64)
            bidx[mask] = np.asarray(inv).ravel()
            self.built += len(uniq)
        else:
            bcdd = np.zeros((1, 6))
            bnodes = np.zeros((1, self.order ** 2, 6))
            bidx = np.zeros((T, room), np.int64)
        sub = np.nonzero(ok)[0]
        if len(sub):
            o = np.zeros((len(sub), 15))
            _sum(np.ascontiguousarray(pts[sub]), self.rdd, self.ntheta, self.nphi, self.nfp, self.eta, self.cache_depth,
                 offsets, cen, rad, cdd, nodes, np.ascontiguousarray(bidx[sub]), np.ascontiguousarray(nkeys[sub]),
                 bcdd, bnodes, grad, potential, o)
            out[sub] = o
        return out, failed

    @property
    def hard_limit(self):
        """The smallest distance (m) the quadrature resolves: a few panel radii at the deepest level."""
        if not hasattr(self, "_hard"):
            Lt, Lp = self.lengths
            r0 = 0.5 * math.hypot(Lt / self.ntheta, Lp / self.nphi)
            self._hard = 4 * self.eta * r0 / 2 ** self.max_depth
        return self._hard


def sheet_field(part, order=None, eta=None):
    """The SheetField of a part at (order, eta), built once per part and kept with it: every engine of a source
    shares its panels."""
    order = ORDER if order is None else int(order)
    eta = ETA if eta is None else float(eta)
    store = part.__dict__.setdefault("_sheet_fields", {})
    key = (order, eta)
    if key not in store:
        store[key] = SheetField(part, order=order, eta=eta)
    return store[key]


# ----------------------------------------------------------------------------- the flux inside the boundary
ANNULUS = (0.1, 0.05)         # the boundary's own flux through annuli this wide (minor radii; the second checks the first)


def interior_curve(S, nphi=64):
    """A closed curve inside every section of the boundary: the m = 0 part of the series, (R_c, Z_c)(phi) as complex
    coefficients (cR, cZ) of exp(-i n nfp phi), or None where it leaves a section.  The flux tube of the source's
    vector potential (FieldEngine) lies on it."""
    N = S.N
    cR = S.CN[0, 0].copy()
    cZ = S.CN[2, 0].copy()
    ph = TAU * np.arange(nphi) / (nphi * S.nfp)
    E = np.exp(1j * np.outer(ph, S.nv))
    Rc, Zc = (E @ cR).real, (E @ cZ).real
    nt = max(8 * (S.M + 1), 256)
    th = TAU * np.arange(nt) / nt
    V = grid_values(S.CN, S.mv, S.nv, th, ph)
    R, Z = V[0], V[3]
    # winding number of each section about the curve's point
    a = np.arctan2(Z - Zc[None, :], R - Rc[None, :])
    ang = np.unwrap(np.concatenate([a, a[:1]]), axis=0)
    turns = (ang[-1] - ang[0]) / TAU
    if np.any(np.abs(turns - 1) > 1e-6):
        return None
    return cR, cZ


def flux_tube_A(points, cR, cZ, nv, c):
    """c grad(theta) / (2 pi) at Cartesian points, theta = atan2(Z - Z_c(phi), R - R_c(phi)): a gradient off the
    curve, so it adds nothing to B, and c to -oint A . dl around every loop that goes once round the curve
    (counterclockwise in (R, Z)) ... with the opposite sign: -oint (c / 2 pi) grad(theta) . dl = -c."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    R = np.hypot(x, y)
    ph = np.arctan2(y, x)
    E = np.exp(1j * np.outer(ph, nv))
    Rc, Zc = (E @ cR).real, (E @ cZ).real
    dRc, dZc = (E @ (1j * nv * cR)).real, (E @ (1j * nv * cZ)).real
    rR, rZ = R - Rc, z - Zc
    q = rR * rR + rZ * rZ
    gR = -rZ / q
    gZ = rR / q
    gP = (rZ * dRc - rR * dZc) / (q * R)
    cp, sp = x / R, y / R
    k = c / TAU
    return np.stack([k * (gR * cp - gP * sp), k * (gR * sp + gP * cp), k * gZ], -1)


def boundary_flux(engine_eval, S, nsec=4, ntheta=None, widths=ANNULUS, order=16):
    """The flux of the source's own field (coils and sheet, no flux tube) through the boundary's sections at nsec
    toroidal angles of one field period: Phi = -oint A . dl round a curve outside, the boundary moved out by w along
    its sections' normals, less int B_phi dR dZ over the annulus between them (Gauss-Legendre of `order` points across
    it, the trapezoid rule round it: the field outside a sheet is analytic up to it, so both converge geometrically,
    and no point is nearer the boundary than 0.005 w).  engine_eval(points, want_A) -> dict with B or A.  Returns
    dict(flux: the mean, spread: max - min over the sections, check: the largest change with the second width,
    sections, angles, widths)."""
    a = S.size["minor_radius"]
    ntheta = ntheta or max(2 * (S.M + 1) + 32, 96)
    th = TAU * np.arange(ntheta) / ntheta
    phis = (np.arange(nsec) + 0.5) * TAU / (S.nfp * nsec)
    x, wg = np.polynomial.legendre.leggauss(order)
    sg, wg = 0.5 * (x + 1), 0.5 * wg
    # the narrowest convex radius of curvature of any section: the annulus may not fold
    nt2 = max(8 * (S.M + 1), 256)
    th2 = TAU * np.arange(nt2) / nt2
    ph2 = TAU * np.arange(max(8 * (S.N + 1), 32)) / (max(8 * (S.N + 1), 32) * S.nfp)
    V2 = grid_values(S.CN, S.mv, S.nv, th2, ph2)
    Rt, Zt = V2[1], V2[4]
    # second theta derivatives from the series
    Ep = np.exp(1j * np.outer(ph2, S.nv))
    Et = np.exp(1j * np.outer(th2, S.mv)) * (1j * S.mv) ** 2
    Rtt = (Et @ np.einsum("mn,bn->mb", S.CN[0], Ep)).real
    Ztt = (Et @ np.einsum("mn,bn->mb", S.CN[2], Ep)).real
    kappa = (Rt * Ztt - Zt * Rtt) / np.hypot(Rt, Zt) ** 3        # > 0 where the section is convex
    kmax = float(max(np.max(kappa), 1e-300))
    out = {}
    w0 = min(widths[0] * a, 0.4 / kmax)
    for w_ in widths:
        w = w0 * w_ / widths[0]
        vals = []
        for ph in phis:
            V = grid_values(S.CN, S.mv, S.nv, th, np.array([ph]))[:, :, 0]
            R, Rt_, Z, Zt_ = V[0], V[1], V[3], V[4]
            Ep1 = np.exp(1j * ph * S.nv)
            Et2 = np.exp(1j * np.outer(th, S.mv)) * (1j * S.mv) ** 2
            Rtt_ = (Et2 @ (S.CN[0] @ Ep1)).real
            Ztt_ = (Et2 @ (S.CN[2] @ Ep1)).real
            L = np.hypot(Rt_, Zt_)
            nR, nZ = Zt_ / L, -Rt_ / L
            Lt = (Rt_ * Rtt_ + Zt_ * Ztt_) / L
            nRt, nZt = (Ztt_ * L - Zt_ * Lt) / L ** 2, -(Rtt_ * L - Rt_ * Lt) / L ** 2
            c, s = math.cos(ph), math.sin(ph)
            # the annulus (theta, s) -> (R, Z) + s w n, and its Jacobian det[d/ds, d/dtheta]
            Rq = R[:, None] + w * sg[None, :] * nR[:, None]
            Zq = Z[:, None] + w * sg[None, :] * nZ[:, None]
            dRt = Rt_[:, None] + w * sg[None, :] * nRt[:, None]
            dZt = Zt_[:, None] + w * sg[None, :] * nZt[:, None]
            J = w * (nR[:, None] * dZt - nZ[:, None] * dRt)
            if not np.all(J > 0):
                raise ValueError("the annulus outside the boundary folds: torux cannot measure the boundary's flux")
            P = np.stack([Rq * c, Rq * s, Zq], -1).reshape(-1, 3)
            B = np.asarray(engine_eval(P, False)["B"]).reshape(ntheta, order, 3)
            Bphi = -B[..., 0] * s + B[..., 1] * c
            annulus = TAU * np.mean(np.sum(Bphi * J * wg[None, :], axis=1))
            # the outer curve's loop integral
            Ro, Zo = R + w * nR, Z + w * nZ
            Po = np.stack([Ro * c, Ro * s, Zo], -1)
            A = np.asarray(engine_eval(Po, True)["A"])
            AR = A[:, 0] * c + A[:, 1] * s
            dRo, dZo = Rt_ + w * nRt, Zt_ + w * nZt
            outer = -TAU * np.mean(AR * dRo + A[:, 2] * dZo)
            vals.append(float(outer - annulus))
        out[w_] = (np.array(vals), w)
    f1, w1 = out[widths[0]]
    f2, w2 = out[widths[1]]
    return dict(flux=float(np.mean(f1)), spread=float(np.max(f1) - np.min(f1)), sections=f1.tolist(),
                check=float(np.max(np.abs(f1 - f2))), angles=phis.tolist(), widths=[w1, w2], ntheta=int(ntheta),
                order=int(order))


def source_fingerprint(source):
    """A hash of every part's data: a kept result of the source is reused only for the same parts."""
    import dataclasses
    import hashlib
    h = hashlib.sha256()
    for p in source.parts:
        h.update(p.kind.encode())
        for f in dataclasses.fields(p):
            v = getattr(p, f.name)
            if isinstance(v, np.ndarray):
                h.update(np.ascontiguousarray(v).tobytes())
            else:
                h.update(repr(v).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- the solver's mode
_SOLVER_MODE = [False]


class solver_mode:
    """Inside, every FieldEngine built without sheet_nan gives NaN at points inside a sheet or too near it, instead of
    an error: a Newton step or a traced line that leaves the exterior is a failed step, never an answer."""

    def __enter__(self):
        self._was = _SOLVER_MODE[0]
        _SOLVER_MODE[0] = True
        return self

    def __exit__(self, *exc):
        _SOLVER_MODE[0] = self._was
        return False
