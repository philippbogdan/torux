"""Precise tier, FourierCurves: Biot-Savart line integrals in double-double.

B(x)    = mu0 I/(4 pi) int_0^1 r'(t) x (x - r(t)) / |x - r(t)|^3 dt
A(x)    = mu0 I/(4 pi) int_0^1 r'(t) / |x - r(t)| dt
dB_i/dx_j: derivative of the B integrand under the integral sign.

Quadrature: Gauss-Legendre (n = 32 by default, nodes/weights in dd) on dyadic
panels of [0, 1).  Level-0 panels: M0 = 2^m.  For each panel P (already
evaluated) the two halves are evaluated; P is accepted, with value Q(left) +
Q(right), when
  (a priori)      the smallest distance from x to P's nodes >= eta * half arclength(P)
                  (the nearest complex singularity of the integrand is far enough
                  for exponential convergence), and
  (a posteriori)  |Q(P) - Q(left) - Q(right)| <= rtol * L1(left + right) for each of
                  B, A, dB (L1 = quadrature of |integrand|, the cancellation-aware scale).
Otherwise the halves are refined in turn.  The reported error is the sum of the
|Q(P) - Q(halves)| of accepted panels (a conservative bound for the accepted, finer
value) plus a dd rounding term.

The curve r(t), r'(t) is evaluated in dd (sin/cos of 2 pi t in dd, angle-addition
recurrence for k t), so the only inputs taken as exact are the float64 Fourier
coefficients, the currents and the evaluation points.  Node values of the
dyadic panels of levels 0..Lc are cached per coil (independent of x).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import dd
from .dd import TWO_PI_HI, TWO_PI_LO
from .quadrules import gauss_legendre_dd
from .constants import MU0_4PI_HI, MU0_4PI_LO


@njit(cache=True, error_model="numpy")
def curve_eval_dd(coef, t):
    """r(t), r'(t) (dd) for one curve; coef (3, 2K+1) float64, t dd."""
    K = (coef.shape[1] - 1) // 2
    c1, s1 = dd.cossin2pi(t)
    ck = c1
    sk = s1
    px = complex(coef[0, 0], 0.0)
    py = complex(coef[1, 0], 0.0)
    pz = complex(coef[2, 0], 0.0)
    zero = complex(0.0, 0.0)
    dx = zero
    dy = zero
    dz = zero
    for k in range(1, K + 1):
        sx, cx = coef[0, 2 * k - 1], coef[0, 2 * k]
        sy, cy = coef[1, 2 * k - 1], coef[1, 2 * k]
        sz, cz = coef[2, 2 * k - 1], coef[2, 2 * k]
        px = dd.add(px, dd.add(dd.mul_d(sk, sx), dd.mul_d(ck, cx)))
        py = dd.add(py, dd.add(dd.mul_d(sk, sy), dd.mul_d(ck, cy)))
        pz = dd.add(pz, dd.add(dd.mul_d(sk, sz), dd.mul_d(ck, cz)))
        fk = float(k)
        dx = dd.add(dx, dd.mul_d(dd.sub(dd.mul_d(ck, sx), dd.mul_d(sk, cx)), fk))
        dy = dd.add(dy, dd.mul_d(dd.sub(dd.mul_d(ck, sy), dd.mul_d(sk, cy)), fk))
        dz = dd.add(dz, dd.mul_d(dd.sub(dd.mul_d(ck, sz), dd.mul_d(sk, cz)), fk))
        if k < K:
            cn = dd.sub(dd.mul(ck, c1), dd.mul(sk, s1))
            sk = dd.add(dd.mul(sk, c1), dd.mul(ck, s1))
            ck = cn
    tp = complex(TWO_PI_HI, TWO_PI_LO)
    return px, py, pz, dd.mul(dx, tp), dd.mul(dy, tp), dd.mul(dz, tp)


@njit(cache=True, error_model="numpy")
def _node_t(M0, lev, j, xi):
    """t of the GL node xi (dd, in [-1,1]) in dyadic panel (lev, j)."""
    w = 1.0 / (M0 * (2.0 ** lev))
    half = 0.5 * w
    centre = (j + 0.5) * w
    return dd.add_d(dd.mul_pow2(xi, half), centre), half


@njit(parallel=True, cache=True, error_model="numpy")
def build_cache_dd(coef, M0, Lc, glx):
    """Node positions/derivatives for all panels of levels 0..Lc, every coil."""
    nc = coef.shape[0]
    n = glx.shape[0]
    npan = M0 * (2 ** (Lc + 1) - 1)
    pos = np.zeros((nc, npan, n, 3), dtype=np.complex128)
    der = np.zeros((nc, npan, n, 3), dtype=np.complex128)
    for task in prange(nc * npan):
        ci = task // npan
        p = task % npan
        # level of panel p
        lev = 0
        off = 0
        while p >= off + M0 * (2 ** lev):
            off += M0 * (2 ** lev)
            lev += 1
        j = p - off
        for i in range(n):
            t, _ = _node_t(M0, lev, j, glx[i])
            px, py, pz, dx, dy, dz = curve_eval_dd(coef[ci], t)
            pos[ci, p, i, 0] = px
            pos[ci, p, i, 1] = py
            pos[ci, p, i, 2] = pz
            der[ci, p, i, 0] = dx
            der[ci, p, i, 1] = dy
            der[ci, p, i, 2] = dz
    return pos, der


@njit(cache=True, error_model="numpy")
def _panel_dd(coef, pos, der, M0, Lc, lev, j, x0, x1, x2, glx, glw, grad, Q, info):
    """Gauss-Legendre sum over panel (lev, j).  Q[15] (dd) = B(3), A(3), dB(9) integrals
    (without mu0 I / 4 pi); info = [L1_B, L1_A, L1_dB, min node distance, half arclength]."""
    n = glx.shape[0]
    zero = complex(0.0, 0.0)
    for k in range(15):
        Q[k] = zero
    l1b = 0.0
    l1a = 0.0
    l1g = 0.0
    dmin = np.inf
    harc = 0.0
    cached = lev <= Lc
    p = M0 * (2 ** lev - 1) + j
    half = 0.5 / (M0 * (2.0 ** lev))
    for i in range(n):
        if cached:
            rx = pos[p, i, 0]
            ry = pos[p, i, 1]
            rz = pos[p, i, 2]
            tx = der[p, i, 0]
            ty = der[p, i, 1]
            tz = der[p, i, 2]
        else:
            t, _ = _node_t(M0, lev, j, glx[i])
            rx, ry, rz, tx, ty, tz = curve_eval_dd(coef, t)
        w = dd.mul_pow2(glw[i], half)
        Rx = dd.sub(x0, rx)
        Ry = dd.sub(x1, ry)
        Rz = dd.sub(x2, rz)
        R2 = dd.add(dd.add(dd.sqr(Rx), dd.sqr(Ry)), dd.sqr(Rz))
        Rn = dd.sqrt(R2)
        inv1 = dd.div(w, Rn)            # w / |R|
        wB = dd.div(inv1, R2)           # w / |R|^3
        cx = dd.sub(dd.mul(ty, Rz), dd.mul(tz, Ry))
        cy = dd.sub(dd.mul(tz, Rx), dd.mul(tx, Rz))
        cz = dd.sub(dd.mul(tx, Ry), dd.mul(ty, Rx))
        Q[0] = dd.add(Q[0], dd.mul(cx, wB))
        Q[1] = dd.add(Q[1], dd.mul(cy, wB))
        Q[2] = dd.add(Q[2], dd.mul(cz, wB))
        Q[3] = dd.add(Q[3], dd.mul(tx, inv1))
        Q[4] = dd.add(Q[4], dd.mul(ty, inv1))
        Q[5] = dd.add(Q[5], dd.mul(tz, inv1))
        rn = Rn.real
        speed = np.sqrt(tx.real * tx.real + ty.real * ty.real + tz.real * tz.real)
        cn = np.sqrt(cx.real * cx.real + cy.real * cy.real + cz.real * cz.real)
        wf = w.real
        l1b += cn * wB.real
        l1a += speed * inv1.real
        if grad:
            q = dd.mul_d(dd.div(wB, R2), 3.0)      # 3 w / |R|^5
            a0 = dd.mul(cx, q)
            a1 = dd.mul(cy, q)
            a2 = dd.mul(cz, q)
            ux = dd.mul(tx, wB)
            uy = dd.mul(ty, wB)
            uz = dd.mul(tz, wB)
            Q[6] = dd.sub(Q[6], dd.mul(a0, Rx))
            Q[7] = dd.sub(Q[7], dd.add(uz, dd.mul(a0, Ry)))
            Q[8] = dd.add(Q[8], dd.sub(uy, dd.mul(a0, Rz)))
            Q[9] = dd.add(Q[9], dd.sub(uz, dd.mul(a1, Rx)))
            Q[10] = dd.sub(Q[10], dd.mul(a1, Ry))
            Q[11] = dd.sub(Q[11], dd.add(ux, dd.mul(a1, Rz)))
            Q[12] = dd.sub(Q[12], dd.add(uy, dd.mul(a2, Rx)))
            Q[13] = dd.add(Q[13], dd.sub(ux, dd.mul(a2, Ry)))
            Q[14] = dd.sub(Q[14], dd.mul(a2, Rz))
            l1g += cn * rn * q.real + 1.5 * speed * wB.real
        if rn < dmin:
            dmin = rn
        harc += wf * speed
    info[0] = l1b
    info[1] = l1a
    info[2] = l1g
    info[3] = dmin
    info[4] = 0.5 * harc


@njit(cache=True, error_model="numpy")
def _maxdiff(Qp, QL, QR, k0, k1):
    e = 0.0
    for k in range(k0, k1):
        d = dd.sub(dd.sub(Qp[k], QL[k]), QR[k])
        a = abs(d.real)
        if a > e:
            e = a
    return e


@njit(cache=True, error_model="numpy")
def adapt_curve_dd(coef, pos, der, M0, Lc, x0, x1, x2, glx, glw, rtol, eta, maxlev, grad, acc, res):
    """Adaptive integral over one curve.  acc[15] += integrals (dd);
    res = [errB, errA, errG, L1B, L1A, L1G, dmin, flag, npanels]."""
    cap = M0 + 2 * maxlev + 8
    stackQ = np.zeros((cap, 15), dtype=np.complex128)
    stackI = np.zeros((cap, 5))
    stackL = np.zeros(cap, dtype=np.int64)
    stackJ = np.zeros(cap, dtype=np.int64)
    QL = np.zeros(15, dtype=np.complex128)
    QR = np.zeros(15, dtype=np.complex128)
    IL = np.zeros(5)
    IR = np.zeros(5)
    sp = 0
    for j in range(M0 - 1, -1, -1):
        _panel_dd(coef, pos, der, M0, Lc, 0, j, x0, x1, x2, glx, glw, grad, stackQ[sp], stackI[sp])
        stackL[sp] = 0
        stackJ[sp] = j
        if stackI[sp, 3] < res[6]:
            res[6] = stackI[sp, 3]
        sp += 1
    while sp > 0:
        sp -= 1
        lev = stackL[sp]
        j = stackJ[sp]
        _panel_dd(coef, pos, der, M0, Lc, lev + 1, 2 * j, x0, x1, x2, glx, glw, grad, QL, IL)
        _panel_dd(coef, pos, der, M0, Lc, lev + 1, 2 * j + 1, x0, x1, x2, glx, glw, grad, QR, IR)
        res[8] += 2
        if IL[3] < res[6]:
            res[6] = IL[3]
        if IR[3] < res[6]:
            res[6] = IR[3]
        accept = False
        eB = 0.0
        eA = 0.0
        eG = 0.0
        if lev + 1 >= maxlev:
            accept = True
            res[7] = 1.0
            eB = np.inf
            eA = np.inf
            eG = np.inf
        elif stackI[sp, 3] >= eta * stackI[sp, 4]:
            eB = _maxdiff(stackQ[sp], QL, QR, 0, 3)
            eA = _maxdiff(stackQ[sp], QL, QR, 3, 6)
            if grad:
                eG = _maxdiff(stackQ[sp], QL, QR, 6, 15)
            if (eB <= rtol * (IL[0] + IR[0]) and eA <= rtol * (IL[1] + IR[1])
                    and eG <= rtol * (IL[2] + IR[2])):
                accept = True
        if accept:
            for k in range(15):
                acc[k] = dd.add(acc[k], dd.add(QL[k], QR[k]))
            res[0] += eB
            res[1] += eA
            res[2] += eG
            res[3] += IL[0] + IR[0]
            res[4] += IL[1] + IR[1]
            res[5] += IL[2] + IR[2]
        else:
            for k in range(15):
                stackQ[sp, k] = QR[k]
                stackQ[sp + 1, k] = QL[k]
            for k in range(5):
                stackI[sp, k] = IR[k]
                stackI[sp + 1, k] = IL[k]
            stackL[sp] = lev + 1
            stackJ[sp] = 2 * j + 1
            stackL[sp + 1] = lev + 1
            stackJ[sp + 1] = 2 * j
            sp += 2


@njit(parallel=True, cache=True, error_model="numpy")
def fourier_dd(points, coef, cur, pos, der, M0, Lc, glx, glw, rtol, eta, maxlev, grad):
    """All curves of one FourierCurves part at all points.

    Returns out (N, 15) dd [B, A, dB row-major dB[i, j] = dB_i/dx_j],
    err (N, 3) [B, A, dB] absolute, scale (N, 3) L1 scales, dmin (N,), flag (N,), work (N,)."""
    N = points.shape[0]
    nc = coef.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    dmin = np.full(N, np.inf)
    flag = np.zeros(N, dtype=np.int64)
    work = np.zeros(N)
    c0 = complex(MU0_4PI_HI, MU0_4PI_LO)
    for p in prange(N):
        x0 = complex(points[p, 0], 0.0)
        x1 = complex(points[p, 1], 0.0)
        x2 = complex(points[p, 2], 0.0)
        acc = np.zeros(15, dtype=np.complex128)
        res = np.zeros(9)
        for ci in range(nc):
            for k in range(15):
                acc[k] = complex(0.0, 0.0)
            for k in range(9):
                res[k] = 0.0
            res[6] = np.inf
            adapt_curve_dd(coef[ci], pos[ci], der[ci], M0, Lc, x0, x1, x2, glx, glw, rtol, eta, maxlev,
                           grad, acc, res)
            C = dd.mul_d(c0, cur[ci])
            ac = abs(C.real)
            for k in range(15):
                out[p, k] = dd.add(out[p, k], dd.mul(C, acc[k]))
            for g in range(3):
                err[p, g] += ac * (res[g] + 1e-30 * res[3 + g])
                scale[p, g] += ac * res[3 + g]
            if res[6] < dmin[p]:
                dmin[p] = res[6]
            if res[7] > 0:
                flag[p] = 1
            work[p] += res[8]
    return out, err, scale, dmin, flag, work


# ------------------------------------------------------------------ python-side preparation
class FourierPrepared:
    """Per-part precomputation (node cache) for the precise tier."""

    def __init__(self, part, n_gl=32, Lc=4, M0=None):
        coef = np.ascontiguousarray(part.coef, dtype=np.float64)
        K = (coef.shape[2] - 1) // 2
        if M0 is None:
            M0 = 4
            while M0 < K // 2:
                M0 *= 2
        self.M0 = int(M0)
        self.Lc = int(Lc)
        self.coef = coef
        self.cur = np.ascontiguousarray(part.current, dtype=np.float64)
        self.glx, self.glw = gauss_legendre_dd(n_gl)
        self.pos, self.der = build_cache_dd(coef, self.M0, self.Lc, self.glx)

    def evaluate(self, points, rtol=1e-20, eta=1.0, maxlev=48, grad=True):
        return fourier_dd(np.ascontiguousarray(points, dtype=np.float64), self.coef, self.cur, self.pos,
                          self.der, self.M0, self.Lc, self.glx, self.glw, float(rtol), float(eta), int(maxlev),
                          bool(grad))
