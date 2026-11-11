"""Precise tier, AxisymPolygons: uniform toroidal current density J = I/|area| over a polygonal
(R, Z) cross-section; field = mu0 J/(4 pi) times the area integral of the unit-C ring kernel
(precise_closed.ring_kernel_dd) over the cross-section.  All arithmetic in double-double.

B and A (area integrals):
  * point outside the polygon (winding 0, distance > 0): the ear-clipped triangulation is
    integrated with Duffy-mapped n x n Gauss-Legendre (n = 16) on triangles, refined by 4-way
    midpoint subdivision until (a priori) dist(point, triangle) >= eta2 * diam(triangle) and
    (a posteriori) |Q(T) - sum Q(children)| <= rtol * L1(children).
  * point inside or on the boundary: polar decomposition about the point p (signed fan
    triangles (p, v_k, v_k+1); Duffy apex at p, so the 1/r kernel singularity is cancelled):
        int_P f dA = sum_k c_k int_0^1 ds int_0^1 u f(p + u (q_k(s) - p)) du,
        c_k = (v_k - p) x (v_k+1 - v_k)
    with tanh-sinh in u (h = 1/16 against h = 1/8 for the error estimate; handles the u log u
    endpoint behaviour of the curvature terms) and adaptive Gauss-Legendre in s.
Gradient (valid inside and outside; Green's theorem, polygon counter-clockwise):
    B_rho     = -J oint A_ring dR'            (diagnostic duplicate of the area integral)
    dB_rho/dz =  J oint B_rho,ring dR'
    dB_rho/drho = -J oint (B_z,ring - A_ring/rho) dR'
    dB_z/dz   = -(dB_rho/drho + B_rho/rho),     dB_z/drho = dB_rho/dz - mu0 J chi(inside)
(1D adaptive Gauss-Legendre along the edges with dR' != 0).  On the boundary the gradient is
undefined (NaN, flag 2).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import dd
from .constants import MU0_4PI_HI, MU0_4PI_LO
from .geometry import ear_clip, signed_area
from .precise_closed import ring_kernel_dd, ring_kernel_off_dd
from .quadrules import gauss_legendre_dd, tanh_sinh01_dd

ETA2 = 1.5        # a priori: distance >= ETA2 * triangle diameter
ETA1 = 1.0        # a priori (1D): distance >= ETA1 * half length of the edge piece


# ------------------------------------------------------------------ geometry in float64
@njit(cache=True, error_model="numpy")
def _seg_dist(px, pz, ax, az, bx, bz):
    lx = bx - ax
    lz = bz - az
    ll = lx * lx + lz * lz
    t = ((px - ax) * lx + (pz - az) * lz) / ll if ll > 0 else 0.0
    t = min(1.0, max(0.0, t))
    dx = px - ax - t * lx
    dz = pz - az - t * lz
    return np.sqrt(dx * dx + dz * dz)


@njit(cache=True, error_model="numpy")
def _winding(px, pz, V):
    """winding number of polygon V (n, 2) around (px, pz) (0 outside)."""
    n = V.shape[0]
    w = 0
    for i in range(n):
        ax, az = V[i, 0], V[i, 1]
        bx, bz = V[(i + 1) % n, 0], V[(i + 1) % n, 1]
        if az <= pz:
            if bz > pz and (bx - ax) * (pz - az) - (px - ax) * (bz - az) > 0:
                w += 1
        else:
            if bz <= pz and (bx - ax) * (pz - az) - (px - ax) * (bz - az) < 0:
                w -= 1
    return w


@njit(cache=True, error_model="numpy")
def _tri_dist(px, pz, t):
    """distance from p to triangle t (3 x 2 float), 0 if inside."""
    d = min(_seg_dist(px, pz, t[0, 0], t[0, 1], t[1, 0], t[1, 1]),
            _seg_dist(px, pz, t[1, 0], t[1, 1], t[2, 0], t[2, 1]),
            _seg_dist(px, pz, t[2, 0], t[2, 1], t[0, 0], t[0, 1]))
    c0 = (t[1, 0] - t[0, 0]) * (pz - t[0, 1]) - (t[1, 1] - t[0, 1]) * (px - t[0, 0])
    c1 = (t[2, 0] - t[1, 0]) * (pz - t[1, 1]) - (t[2, 1] - t[1, 1]) * (px - t[1, 0])
    c2 = (t[0, 0] - t[2, 0]) * (pz - t[2, 1]) - (t[0, 1] - t[2, 1]) * (px - t[2, 0])
    if (c0 >= 0 and c1 >= 0 and c2 >= 0) or (c0 <= 0 and c1 <= 0 and c2 <= 0):
        return 0.0
    return d


# ------------------------------------------------------------------ 2D triangle rule
@njit(cache=True, error_model="numpy")
def _tri_rule(T, rho, z, glu, glwu, Q, info):
    """Duffy GL over triangle T (3 x 2 dd, ccw); Q[5] = int (A, A/rho, B_rho, B_rho/rho, B_z) dA,
    info = [L1_B, L1_A]."""
    zero = complex(0.0, 0.0)
    for k in range(5):
        Q[k] = zero
    e1r = dd.sub(T[1, 0], T[0, 0])
    e1z = dd.sub(T[1, 1], T[0, 1])
    e2r = dd.sub(T[2, 0], T[0, 0])
    e2z = dd.sub(T[2, 1], T[0, 1])
    jac = dd.absd(dd.sub(dd.mul(e1r, e2z), dd.mul(e1z, e2r)))
    fr = dd.sub(e2r, e1r)
    fz = dd.sub(e2z, e1z)
    kr = np.zeros(7, dtype=np.complex128)
    l1b = 0.0
    l1a = 0.0
    n = glu.shape[0]
    for i in range(n):
        u = glu[i]
        wu = dd.mul(dd.mul(glwu[i], u), jac)
        for j in range(n):
            v = glu[j]
            w = dd.mul(wu, glwu[j])
            Rp = dd.add(T[0, 0], dd.mul(u, dd.add(e1r, dd.mul(v, fr))))
            Zp = dd.add(T[0, 1], dd.mul(u, dd.add(e1z, dd.mul(v, fz))))
            ring_kernel_dd(Rp, Zp, rho, z, False, kr)
            for k in range(5):
                Q[k] = dd.add(Q[k], dd.mul(w, kr[k]))
            l1b += abs(w.real) * (abs(kr[2].real) + abs(kr[4].real))
            l1a += abs(w.real * kr[0].real)
    info[0] = l1b
    info[1] = l1a


@njit(cache=True, error_model="numpy")
def _tri_children(T, C):
    """4 midpoint children of T (3 x 2 dd) into C (4, 3, 2)."""
    m01r = dd.mul_pow2(dd.add(T[0, 0], T[1, 0]), 0.5)
    m01z = dd.mul_pow2(dd.add(T[0, 1], T[1, 1]), 0.5)
    m12r = dd.mul_pow2(dd.add(T[1, 0], T[2, 0]), 0.5)
    m12z = dd.mul_pow2(dd.add(T[1, 1], T[2, 1]), 0.5)
    m20r = dd.mul_pow2(dd.add(T[2, 0], T[0, 0]), 0.5)
    m20z = dd.mul_pow2(dd.add(T[2, 1], T[0, 1]), 0.5)
    C[0, 0, 0] = T[0, 0]; C[0, 0, 1] = T[0, 1]
    C[0, 1, 0] = m01r; C[0, 1, 1] = m01z
    C[0, 2, 0] = m20r; C[0, 2, 1] = m20z
    C[1, 0, 0] = m01r; C[1, 0, 1] = m01z
    C[1, 1, 0] = T[1, 0]; C[1, 1, 1] = T[1, 1]
    C[1, 2, 0] = m12r; C[1, 2, 1] = m12z
    C[2, 0, 0] = m20r; C[2, 0, 1] = m20z
    C[2, 1, 0] = m12r; C[2, 1, 1] = m12z
    C[2, 2, 0] = T[2, 0]; C[2, 2, 1] = T[2, 1]
    C[3, 0, 0] = m12r; C[3, 0, 1] = m12z
    C[3, 1, 0] = m20r; C[3, 1, 1] = m20z
    C[3, 2, 0] = m01r; C[3, 2, 1] = m01z


@njit(cache=True, error_model="numpy")
def _tri_float(T, out):
    for a in range(3):
        out[a, 0] = T[a, 0].real
        out[a, 1] = T[a, 1].real


@njit(cache=True, error_model="numpy")
def _diam(t):
    d0 = np.hypot(t[1, 0] - t[0, 0], t[1, 1] - t[0, 1])
    d1 = np.hypot(t[2, 0] - t[1, 0], t[2, 1] - t[1, 1])
    d2 = np.hypot(t[0, 0] - t[2, 0], t[0, 1] - t[2, 1])
    return max(d0, d1, d2)


@njit(cache=True, error_model="numpy")
def area_direct(tris, rho, z, glu, glwu, rtol, maxlev, acc, res):
    """Adaptive area integral over the triangles tris (nt, 3, 2) dd (p outside).
    acc[5] += integrals; res = [errB, errA, L1B, L1A, flag, ntri]."""
    nt = tris.shape[0]
    cap = nt + 4 * maxlev + 16
    stT = np.zeros((cap, 3, 2), dtype=np.complex128)
    stQ = np.zeros((cap, 5), dtype=np.complex128)
    stI = np.zeros((cap, 2))
    stL = np.zeros(cap, dtype=np.int64)
    ch = np.zeros((4, 3, 2), dtype=np.complex128)
    chQ = np.zeros((4, 5), dtype=np.complex128)
    chI = np.zeros((4, 2))
    tf = np.zeros((3, 2))
    px = rho.real
    pz = z.real
    sp = 0
    for t in range(nt):
        for a in range(3):
            stT[sp, a, 0] = tris[t, a, 0]
            stT[sp, a, 1] = tris[t, a, 1]
        _tri_rule(stT[sp], rho, z, glu, glwu, stQ[sp], stI[sp])
        stL[sp] = 0
        sp += 1
    while sp > 0:
        sp -= 1
        lev = stL[sp]
        _tri_children(stT[sp], ch)
        for c in range(4):
            _tri_rule(ch[c], rho, z, glu, glwu, chQ[c], chI[c])
        res[5] += 4
        _tri_float(stT[sp], tf)
        dist = _tri_dist(px, pz, tf)
        diam = _diam(tf)
        accept = False
        eB = 0.0
        eA = 0.0
        l1b = chI[0, 0] + chI[1, 0] + chI[2, 0] + chI[3, 0]
        l1a = chI[0, 1] + chI[1, 1] + chI[2, 1] + chI[3, 1]
        if lev + 1 >= maxlev:
            accept = True
            res[4] = 1.0
            eB = np.inf
            eA = np.inf
        elif dist >= ETA2 * diam:
            for k in range(5):
                s = dd.add(dd.add(chQ[0, k], chQ[1, k]), dd.add(chQ[2, k], chQ[3, k]))
                e = abs(dd.sub(stQ[sp, k], s).real)
                if k <= 1:
                    if e > eA:
                        eA = e
                elif k == 2 or k == 4:
                    if e > eB:
                        eB = e
            if eB <= rtol * l1b and eA <= rtol * l1a:
                accept = True
        if accept:
            for k in range(5):
                acc[k] = dd.add(acc[k], dd.add(dd.add(chQ[0, k], chQ[1, k]), dd.add(chQ[2, k], chQ[3, k])))
            res[0] += eB
            res[1] += eA
            res[2] += l1b
            res[3] += l1a
        else:
            for c in range(4):
                for a in range(3):
                    stT[sp + c, a, 0] = ch[c, a, 0]
                    stT[sp + c, a, 1] = ch[c, a, 1]
                for k in range(5):
                    stQ[sp + c, k] = chQ[c, k]
                stI[sp + c, 0] = chI[c, 0]
                stI[sp + c, 1] = chI[c, 1]
                stL[sp + c] = lev + 1
            sp += 4


# ------------------------------------------------------------------ 1D edge integrals (polar and boundary)
@njit(cache=True, error_model="numpy")
def _edge_panel(mode, rho, z, ar, az, br, bz, s0, half, glx, glw, deu, dew, deeven, Q, info):
    """GL sum over the s-panel [s0 - half, s0 + half] of edge a -> b (dd endpoints).
    mode 0 (polar): Q[5] = int ds int_0^1 u f(p + u (q(s) - p)) du  (f = ring kernel), unscaled by c_k;
                    info[2] accumulates the tanh-sinh error estimate.
    mode 1 (boundary): Q[3] = int ds (A, B_rho, B_z - A/rho)(q(s)) (unscaled by dR').
    info = [L1_B, L1_A, err_inner, min distance, half length]."""
    zero = complex(0.0, 0.0)
    for k in range(5):
        Q[k] = zero
    er = dd.sub(br, ar)
    ez = dd.sub(bz, az)
    elen = np.hypot(er.real, ez.real)
    kr = np.zeros(7, dtype=np.complex128)
    acc_h = np.zeros(5, dtype=np.complex128)
    acc_2h = np.zeros(5, dtype=np.complex128)
    l1b = 0.0
    l1a = 0.0
    einner = 0.0
    dmin = np.inf
    n = glx.shape[0]
    for i in range(n):
        s = dd.add(dd.mul_d(glx[i], half), s0)
        ws = dd.mul_d(glw[i], half)
        qr = dd.add(ar, dd.mul(s, er))
        qz = dd.add(az, dd.mul(s, ez))
        if mode == 1:
            ring_kernel_dd(qr, qz, rho, z, False, kr)
            Q[0] = dd.add(Q[0], dd.mul(ws, kr[0]))
            Q[1] = dd.add(Q[1], dd.mul(ws, kr[2]))
            g = dd.sub(kr[4], kr[1])
            Q[2] = dd.add(Q[2], dd.mul(ws, g))
            l1b += abs(ws.real) * (abs(kr[2].real) + abs(g.real))
            l1a += abs(ws.real * kr[0].real)
            d = np.hypot(qr.real - rho.real, qz.real - z.real)
            if d < dmin:
                dmin = d
        else:
            wr = dd.sub(qr, rho)
            wz = dd.sub(qz, z)
            for k in range(5):
                acc_h[k] = zero
                acc_2h[k] = zero
            m = deu.shape[0]
            for j in range(m):
                u = deu[j]
                uwr = dd.mul(u, wr)
                ring_kernel_off_dd(dd.add(rho, uwr), rho, uwr, dd.neg(dd.mul(u, wz)), False, kr)
                wu = dd.mul(dew[j], u)
                for k in range(5):
                    t = dd.mul(wu, kr[k])
                    acc_h[k] = dd.add(acc_h[k], t)
                    if deeven[j]:
                        acc_2h[k] = dd.add(acc_2h[k], dd.mul_pow2(t, 2.0))
            for k in range(5):
                Q[k] = dd.add(Q[k], dd.mul(ws, acc_h[k]))
                e = abs(dd.sub(acc_h[k], acc_2h[k]).real)
                if e > einner:
                    einner = e
            einner_w = abs(ws.real)
            info[2] += einner_w * einner
            l1b += abs(ws.real) * (abs(acc_h[2].real) + abs(acc_h[4].real))
            l1a += abs(ws.real * acc_h[0].real)
            d = np.hypot(wr.real, wz.real)
            if d < dmin:
                dmin = d
    info[0] = l1b
    info[1] = l1a
    info[3] = dmin
    info[4] = half * elen


@njit(cache=True, error_model="numpy")
def edge_adapt(mode, rho, z, ar, az, br, bz, glx, glw, deu, dew, deeven, rtol, maxlev, acc, res):
    """Adaptive integral over s in [0, 1] of edge a->b.  acc[5] += ; res = [errB, errA, L1B, L1A, flag]."""
    cap = 2 * maxlev + 8
    stQ = np.zeros((cap, 5), dtype=np.complex128)
    stI = np.zeros((cap, 5))
    stS = np.zeros(cap)       # panel centres (dyadic, exact)
    stH = np.zeros(cap)
    stL = np.zeros(cap, dtype=np.int64)
    QL = np.zeros(5, dtype=np.complex128)
    QR = np.zeros(5, dtype=np.complex128)
    IL = np.zeros(5)
    IR = np.zeros(5)
    nq = 5 if mode == 0 else 3
    sp = 0
    _edge_panel(mode, rho, z, ar, az, br, bz, complex(0.5, 0.0), 0.5, glx, glw, deu, dew, deeven, stQ[0], stI[0])
    stS[0] = 0.5
    stH[0] = 0.5
    stL[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        lev = stL[sp]
        c = stS[sp]
        h = stH[sp]
        h2 = 0.5 * h
        IL[2] = 0.0
        IR[2] = 0.0
        _edge_panel(mode, rho, z, ar, az, br, bz, complex(c - h2, 0.0), h2, glx, glw, deu, dew, deeven, QL, IL)
        _edge_panel(mode, rho, z, ar, az, br, bz, complex(c + h2, 0.0), h2, glx, glw, deu, dew, deeven, QR, IR)
        accept = False
        eB = 0.0
        eA = 0.0
        if lev + 1 >= maxlev:
            accept = True
            res[4] = 1.0
            eB = np.inf
            eA = np.inf
        elif stI[sp, 3] >= ETA1 * stI[sp, 4]:
            for k in range(nq):
                e = abs(dd.sub(dd.sub(stQ[sp, k], QL[k]), QR[k]).real)
                if mode == 0:
                    isA = k <= 1
                else:
                    isA = k == 0
                if isA:
                    if e > eA:
                        eA = e
                else:
                    if e > eB:
                        eB = e
            if eB <= rtol * (IL[0] + IR[0]) and eA <= rtol * (IL[1] + IR[1]):
                accept = True
        if accept:
            for k in range(nq):
                acc[k] = dd.add(acc[k], dd.add(QL[k], QR[k]))
            res[0] += eB + IL[2] + IR[2]
            res[1] += eA + IL[2] + IR[2]
            res[2] += IL[0] + IR[0]
            res[3] += IL[1] + IR[1]
        else:
            for k in range(5):
                stQ[sp, k] = QR[k]
                stQ[sp + 1, k] = QL[k]
                stI[sp, k] = IR[k]
                stI[sp + 1, k] = IL[k]
            stS[sp] = c + h2
            stS[sp + 1] = c - h2
            stH[sp] = h2
            stH[sp + 1] = h2
            stL[sp] = lev + 1
            stL[sp + 1] = lev + 1
            sp += 2


# ------------------------------------------------------------------ driver
@njit(cache=True, error_model="numpy")
def _cart(x0, x1, rho, A, Br, Bror, Bz, drBr, dzBr, drBz, dzBz, grad, out):
    if rho.real > 0.0:
        c = dd.div(x0, rho)
        s = dd.div(x1, rho)
    else:
        c = complex(1.0, 0.0)
        s = complex(0.0, 0.0)
    out[0] = dd.mul(Br, c)
    out[1] = dd.mul(Br, s)
    out[2] = Bz
    out[3] = dd.neg(dd.mul(A, s))
    out[4] = dd.mul(A, c)
    out[5] = complex(0.0, 0.0)
    if grad:
        c2 = dd.sqr(c)
        s2 = dd.sqr(s)
        cs = dd.mul(c, s)
        m = dd.mul(dd.sub(drBr, Bror), cs)
        out[6] = dd.add(dd.mul(drBr, c2), dd.mul(Bror, s2))
        out[7] = m
        out[8] = dd.mul(dzBr, c)
        out[9] = m
        out[10] = dd.add(dd.mul(drBr, s2), dd.mul(Bror, c2))
        out[11] = dd.mul(dzBr, s)
        out[12] = dd.mul(drBz, c)
        out[13] = dd.mul(drBz, s)
        out[14] = dzBz
    else:
        for k in range(6, 15):
            out[k] = complex(0.0, 0.0)


@njit(parallel=True, cache=True, error_model="numpy")
def axipoly_dd(points, verts, vstart, tris, tstart, cur, area, glu, glwu, glx, glw, deu, dew, deeven,
               rtol, maxlev, grad):
    N = points.shape[0]
    npoly = cur.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    dmin = np.full(N, np.inf)
    flag = np.zeros(N, dtype=np.int64)
    bdiag = np.zeros((N, npoly), dtype=np.complex128)     # B_rho from the boundary formula (check)
    c0 = complex(MU0_4PI_HI, MU0_4PI_LO)
    for p in prange(N):
        x0 = complex(points[p, 0], 0.0)
        x1 = complex(points[p, 1], 0.0)
        z = complex(points[p, 2], 0.0)
        rho = dd.sqrt(dd.add(dd.sqr(x0), dd.sqr(x1)))
        tmp = np.zeros(15, dtype=np.complex128)
        acc = np.zeros(5, dtype=np.complex128)
        accb = np.zeros(5, dtype=np.complex128)
        res = np.zeros(6)
        for ip in range(npoly):
            if cur[ip] == 0.0:
                continue
            v0 = vstart[ip]
            v1 = vstart[ip + 1]
            nv = v1 - v0
            V = np.empty((nv, 2))
            for i in range(nv):
                V[i, 0] = verts[v0 + i, 0]
                V[i, 1] = verts[v0 + i, 1]
            orient = 1.0 if area[ip] > 0 else -1.0
            J = dd.mul_d(c0, cur[ip] / abs(area[ip]))      # mu0 J / (4 pi); quotient rounded once
            # exact current-density scaling: use cur/|area| in dd
            J = dd.mul(c0, dd.div_d(complex(cur[ip], 0.0), abs(area[ip])))
            d = np.inf
            for i in range(nv):
                di = _seg_dist(rho.real, z.real, V[i, 0], V[i, 1], V[(i + 1) % nv, 0], V[(i + 1) % nv, 1])
                if di < d:
                    d = di
            if d < dmin[p]:
                dmin[p] = d
            wind = _winding(rho.real, z.real, V)
            inside = wind != 0
            for k in range(5):
                acc[k] = complex(0.0, 0.0)
            for k in range(6):
                res[k] = 0.0
            if (not inside) and d > 0.0:
                t0 = tstart[ip]
                t1 = tstart[ip + 1]
                area_direct(tris[t0:t1], rho, z, glu, glwu, rtol, maxlev, acc, res)
            else:
                # polar fan about p
                for i in range(nv):
                    ar = complex(V[i, 0], 0.0)
                    az = complex(V[i, 1], 0.0)
                    br = complex(V[(i + 1) % nv, 0], 0.0)
                    bz = complex(V[(i + 1) % nv, 1], 0.0)
                    ck = dd.sub(dd.mul(dd.sub(ar, rho), dd.sub(bz, az)), dd.mul(dd.sub(az, z), dd.sub(br, ar)))
                    if ck.real == 0.0:
                        continue
                    acce = np.zeros(5, dtype=np.complex128)
                    rese = np.zeros(5)
                    edge_adapt(0, rho, z, ar, az, br, bz, glx, glw, deu, dew, deeven, rtol, maxlev, acce, rese)
                    sgn = orient
                    for k in range(5):
                        acc[k] = dd.add(acc[k], dd.mul_d(dd.mul(ck, acce[k]), sgn))
                    ack = abs(ck.real)
                    res[0] += ack * rese[0]
                    res[1] += ack * rese[1]
                    res[2] += ack * rese[2]
                    res[3] += ack * rese[3]
                    if rese[4] > 0:
                        res[4] = 1.0
            A = dd.mul(J, acc[0])
            Br = dd.mul(J, acc[2])
            Bror = dd.mul(J, acc[3])
            Bz = dd.mul(J, acc[4])
            aj = abs(J.real)
            err[p, 0] += aj * (res[0] + 1e-29 * res[2])
            err[p, 1] += aj * (res[1] + 1e-29 * res[3])
            scale[p, 0] += aj * res[2]
            scale[p, 1] += aj * res[3]
            if res[4] > 0:
                flag[p] |= 1
            zero = complex(0.0, 0.0)
            drBr = zero
            dzBr = zero
            drBz = zero
            dzBz = zero
            if grad:
                if d == 0.0:
                    flag[p] |= 2
                    nan = complex(np.nan, 0.0)
                    drBr = nan
                    dzBr = nan
                    drBz = nan
                    dzBz = nan
                else:
                    for k in range(5):
                        accb[k] = complex(0.0, 0.0)
                    eg = 0.0
                    lg = 0.0
                    for i in range(nv):
                        if V[(i + 1) % nv, 0] == V[i, 0]:
                            continue
                        ar = complex(V[i, 0], 0.0)
                        az = complex(V[i, 1], 0.0)
                        br = complex(V[(i + 1) % nv, 0], 0.0)
                        bz = complex(V[(i + 1) % nv, 1], 0.0)
                        acce = np.zeros(5, dtype=np.complex128)
                        rese = np.zeros(5)
                        edge_adapt(1, rho, z, ar, az, br, bz, glx, glw, deu, dew, deeven, rtol, maxlev, acce, rese)
                        dR = dd.mul_d(dd.sub(br, ar), orient)
                        for k in range(3):
                            accb[k] = dd.add(accb[k], dd.mul(dR, acce[k]))
                        adr = abs(dR.real)
                        eg += adr * (rese[0] + rese[1])
                        lg += adr * (rese[2] + rese[3])
                        if rese[4] > 0:
                            flag[p] |= 1
                    bdiag[p, ip] = dd.neg(dd.mul(J, accb[0]))
                    dzBr = dd.mul(J, accb[1])
                    drBr = dd.neg(dd.mul(J, accb[2]))
                    dzBz = dd.neg(dd.add(drBr, Bror))
                    drBz = dzBr
                    if inside:
                        # mu0 J = 4 pi * (mu0 J / 4 pi)
                        drBz = dd.sub(dzBr, dd.mul(J, complex(4.0 * dd.PI_HI, 4.0 * dd.PI_LO)))
                    err[p, 2] += aj * (eg + 1e-29 * lg) * 2.0
                    scale[p, 2] += aj * lg
            _cart(x0, x1, rho, A, Br, Bror, Bz, drBr, dzBr, drBz, dzBz, grad, tmp)
            for k in range(15):
                out[p, k] = dd.add(out[p, k], tmp[k])
    return out, err, scale, dmin, flag, bdiag


class AxipolyPrepared:
    def __init__(self, part, n2=16, n1=32, de_h=1.0 / 16, de_tmax=4.0):
        V = np.ascontiguousarray(part.vertices, dtype=np.float64)
        st = np.ascontiguousarray(part.start, dtype=np.int64)
        self.verts = V
        self.vstart = st
        self.cur = np.ascontiguousarray(part.current, dtype=np.float64)
        tris = []
        tst = [0]
        area = []
        for i in range(len(self.cur)):
            P = V[st[i]:st[i + 1]]
            a = signed_area(P)
            area.append(a)
            for (i0, i1, i2) in ear_clip(P):
                tris.append([[P[i0, 0], P[i0, 1]], [P[i1, 0], P[i1, 1]], [P[i2, 0], P[i2, 1]]])
            tst.append(len(tris))
        self.tris = np.array(tris, dtype=np.float64).reshape(-1, 3, 2).astype(np.complex128)
        self.tstart = np.array(tst, dtype=np.int64)
        self.area = np.array(area)
        x, w = gauss_legendre_dd(n2)
        # map [-1, 1] -> [0, 1]: u = (1 + x)/2, weight w/2
        self.glu = np.array([dd.add_d(dd.mul_pow2(xi, 0.5), 0.5) for xi in x])
        self.glwu = np.array([dd.mul_pow2(wi, 0.5) for wi in w])
        self.glx, self.glw = gauss_legendre_dd(n1)
        u, wu = tanh_sinh01_dd(de_h, de_tmax)
        self.deu, self.dew = u, wu
        m = len(u) // 2
        self.deeven = np.array([((i - m) % 2 == 0) for i in range(len(u))])

    def evaluate(self, points, grad=True, rtol=1e-20, maxlev=60, return_diag=False):
        out, err, scale, dmin, flag, bdiag = axipoly_dd(
            np.ascontiguousarray(points, dtype=np.float64), self.verts, self.vstart, self.tris, self.tstart,
            self.cur, self.area, self.glu, self.glwu, self.glx, self.glw, self.deu, self.dew, self.deeven,
            float(rtol), int(maxlev), bool(grad))
        if return_diag:
            return out, err, scale, dmin, flag, bdiag
        return out, err, scale, dmin, flag
