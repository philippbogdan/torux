"""fast64 tier: plain float64 (numba, compensated summation), independent of the solver's engine.

    F = Field(source)                        # packs the parts, builds curve node caches
    r = F.evaluate(points, grad=True)        # dict B (N,3), A (N,3), dB (N,3,3) [dB_i/dx_j]
    Bonly = F.B(points)

Algorithms (same mathematics as ``precise``, float64 arithmetic, a-priori error control):
  FourierCurves  Gauss-Legendre n = 16 on dyadic panels of t, M0 = 2^ceil(log2 K) level-0 panels
                 (ETA_CURVE raised from 2 to 4 on 5 Nov: at 2, QUASR 10614's coil 2 at 0.6 m was off by
                 2e-13 relative against the double-double tier; at 4 every point agrees to ~1e-15)
                 (resolves the curve's own Fourier content), a panel is used when the smallest
                 node distance >= ETA_CURVE * its half arclength, else bisected.  Curve node
                 values of levels 0..Lc are precomputed in double-double and rounded.
  Segments       cancellation-free closed form (Hanson-Hirshman-type denominator choice).
  Dipoles        closed form.
  Rings          Landen-transformed Carlson R_D forms (carlson.rd_dual_f64).
  AxisymPolygons Duffy Gauss-Legendre on triangles (point outside), polar tanh-sinh (inside),
                 Green's-theorem boundary integrals for the gradient (as precise, a priori only).
  ToroidalField  closed form.
Every sum over quadrature nodes and source elements is a TwoSum-compensated sum.

``eval_point_B(P, x, y, z)`` / ``eval_point(P, x, y, z, grad, out)`` are njit-callable on the
packed tuple ``F.packed`` (used by ``torux.reference.trace``).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .carlson import rd_dual0_f64
from .geometry import ear_clip, signed_area
from . import dd
from .precise_curves import build_cache_dd, curve_eval_dd
from .quadrules import gauss_legendre_dd, gauss_legendre_f64, tanh_sinh01_f64

C_MU0_4PI = 1e-7          # mu0 / (4 pi)
ETA_CURVE = 4.0
N_GL = 16


# ------------------------------------------------------------------ compensated accumulation
@njit(cache=True, inline="always", error_model="numpy")
def _acc(s, c, k, v):
    a = s[k]
    t = a + v
    bb = t - a
    c[k] += (a - (t - bb)) + (v - bb)
    s[k] = t


# ------------------------------------------------------------------ Fourier curves
@njit(cache=True, error_model="numpy")
def curve_eval_f64(coef, t):
    K = (coef.shape[1] - 1) // 2
    th = 2.0 * math.pi * t
    c1 = math.cos(th)
    s1 = math.sin(th)
    ck = c1
    sk = s1
    px = coef[0, 0]
    py = coef[1, 0]
    pz = coef[2, 0]
    dx = 0.0
    dy = 0.0
    dz = 0.0
    for k in range(1, K + 1):
        if k > 1:
            # direct evaluation every 8 steps limits recurrence drift
            if k % 8 == 0:
                ck = math.cos(th * k)
                sk = math.sin(th * k)
            else:
                cn = ck * c1 - sk * s1
                sk = sk * c1 + ck * s1
                ck = cn
        px += coef[0, 2 * k - 1] * sk + coef[0, 2 * k] * ck
        py += coef[1, 2 * k - 1] * sk + coef[1, 2 * k] * ck
        pz += coef[2, 2 * k - 1] * sk + coef[2, 2 * k] * ck
        dx += k * (coef[0, 2 * k - 1] * ck - coef[0, 2 * k] * sk)
        dy += k * (coef[1, 2 * k - 1] * ck - coef[1, 2 * k] * sk)
        dz += k * (coef[2, 2 * k - 1] * ck - coef[2, 2 * k] * sk)
    tp = 2.0 * math.pi
    return px, py, pz, dx * tp, dy * tp, dz * tp


@njit(cache=True, error_model="numpy")
def _curve_coil(coef, pos, der, harc, M0, Lc, glx, glw, glx_dd, x0, x1, x2, grad, maxlev, S, Cc, cur):
    """Accumulate one coil (current cur) into the compensated sums S, Cc (15).
    Node positions are hi + lo pairs (cache: rounded double-double; deeper levels: evaluated in
    double-double on the fly), and x - r = (x - r_hi) - r_lo, so the rounding error of x - r is
    relative to |x - r| rather than to |r| (matters within ~cm of a coil)."""
    n = glx.shape[0]
    cap = M0 + 2 * maxlev + 8
    stL = np.empty(cap, dtype=np.int64)
    stJ = np.empty(cap, dtype=np.int64)
    rx = np.empty(n)
    ry = np.empty(n)
    rz = np.empty(n)
    lx = np.empty(n)
    ly = np.empty(n)
    lz = np.empty(n)
    tx = np.empty(n)
    ty = np.empty(n)
    tz = np.empty(n)
    sp = 0
    for j in range(M0 - 1, -1, -1):
        stL[sp] = 0
        stJ[sp] = j
        sp += 1
    cf = C_MU0_4PI * cur
    dmin_all = np.inf
    while sp > 0:
        sp -= 1
        lev = stL[sp]
        j = stJ[sp]
        half = 0.5 / (M0 * (2.0 ** lev))
        centre = (j + 0.5) * 2.0 * half
        p = M0 * (2 ** lev - 1) + j
        h = 0.0
        if lev <= Lc:
            for i in range(n):
                rx[i] = pos[p, i, 0]
                ry[i] = pos[p, i, 1]
                rz[i] = pos[p, i, 2]
                lx[i] = pos[p, i, 3]
                ly[i] = pos[p, i, 4]
                lz[i] = pos[p, i, 5]
                tx[i] = der[p, i, 0]
                ty[i] = der[p, i, 1]
                tz[i] = der[p, i, 2]
            h = harc[p]
        else:
            for i in range(n):
                t = dd.add_d(dd.mul_pow2(glx_dd[i], half), centre)
                a0, a1, a2, b0, b1, b2 = curve_eval_dd(coef, t)
                rx[i] = a0.real
                ry[i] = a1.real
                rz[i] = a2.real
                lx[i] = a0.imag
                ly[i] = a1.imag
                lz[i] = a2.imag
                tx[i] = b0.real
                ty[i] = b1.real
                tz[i] = b2.real
                h += glw[i] * half * math.sqrt(b0.real ** 2 + b1.real ** 2 + b2.real ** 2)
            h *= 0.5
        d2min = np.inf
        for i in range(n):
            d2 = (x0 - rx[i]) ** 2 + (x1 - ry[i]) ** 2 + (x2 - rz[i]) ** 2
            if d2 < d2min:
                d2min = d2
        if d2min < (ETA_CURVE * h) ** 2 and lev < maxlev:
            stL[sp] = lev + 1
            stJ[sp] = 2 * j + 1
            stL[sp + 1] = lev + 1
            stJ[sp + 1] = 2 * j
            sp += 2
            continue
        if d2min < dmin_all:
            dmin_all = d2min
        for i in range(n):
            w = glw[i] * half * cf
            Rx = (x0 - rx[i]) - lx[i]
            Ry = (x1 - ry[i]) - ly[i]
            Rz = (x2 - rz[i]) - lz[i]
            R2 = Rx * Rx + Ry * Ry + Rz * Rz
            Rn = math.sqrt(R2)
            i1 = w / Rn
            i3 = i1 / R2
            cx = ty[i] * Rz - tz[i] * Ry
            cy = tz[i] * Rx - tx[i] * Rz
            cz = tx[i] * Ry - ty[i] * Rx
            _acc(S, Cc, 0, cx * i3)
            _acc(S, Cc, 1, cy * i3)
            _acc(S, Cc, 2, cz * i3)
            _acc(S, Cc, 3, tx[i] * i1)
            _acc(S, Cc, 4, ty[i] * i1)
            _acc(S, Cc, 5, tz[i] * i1)
            if grad:
                q = 3.0 * i3 / R2
                a0 = cx * q
                a1 = cy * q
                a2 = cz * q
                ux = tx[i] * i3
                uy = ty[i] * i3
                uz = tz[i] * i3
                _acc(S, Cc, 6, -a0 * Rx)
                _acc(S, Cc, 7, -uz - a0 * Ry)
                _acc(S, Cc, 8, uy - a0 * Rz)
                _acc(S, Cc, 9, uz - a1 * Rx)
                _acc(S, Cc, 10, -a1 * Ry)
                _acc(S, Cc, 11, -ux - a1 * Rz)
                _acc(S, Cc, 12, -uy - a2 * Rx)
                _acc(S, Cc, 13, ux - a2 * Ry)
                _acc(S, Cc, 14, -a2 * Rz)
    return math.sqrt(dmin_all)


# ------------------------------------------------------------------ segments
@njit(cache=True, error_model="numpy")
def _segment(a, b, cur, x0, x1, x2, grad, S, Cc):
    Lx = b[0] - a[0]
    Ly = b[1] - a[1]
    Lz = b[2] - a[2]
    L2 = Lx * Lx + Ly * Ly + Lz * Lz
    if L2 == 0.0 or cur == 0.0:
        return
    L = math.sqrt(L2)
    Rix = x0 - a[0]
    Riy = x1 - a[1]
    Riz = x2 - a[2]
    Rfx = x0 - b[0]
    Rfy = x1 - b[1]
    Rfz = x2 - b[2]
    ri2 = Rix * Rix + Riy * Riy + Riz * Riz
    rf2 = Rfx * Rfx + Rfy * Rfy + Rfz * Rfz
    ri = math.sqrt(ri2)
    rf = math.sqrt(rf2)
    dot = Rix * Rfx + Riy * Rfy + Riz * Rfz
    P = ri * rf
    # Ri x Rf = L x Ri = L x Rf: use the form with the smallest rounding bound (product of the
    # factors' lengths): Ri x Rf beside the segment, L x (shorter of Ri, Rf) farther away
    if max(ri, rf) <= L:
        cx = Riy * Rfz - Riz * Rfy
        cy = Riz * Rfx - Rix * Rfz
        cz = Rix * Rfy - Riy * Rfx
    elif ri <= rf:
        cx = Ly * Riz - Lz * Riy
        cy = Lz * Rix - Lx * Riz
        cz = Lx * Riy - Ly * Rix
    else:
        cx = Ly * Rfz - Lz * Rfy
        cy = Lz * Rfx - Lx * Rfz
        cz = Lx * Rfy - Ly * Rfx
    if dot >= 0.0:
        den = P + dot
    else:
        den = (cx * cx + cy * cy + cz * cz) / (P - dot)
    Ssum = ri + rf
    if den <= 0.0:
        for k in range(15):
            S[k] = np.nan
        return
    C = C_MU0_4PI * cur
    f = C * Ssum / (P * den)
    _acc(S, Cc, 0, cx * f)
    _acc(S, Cc, 1, cy * f)
    _acc(S, Cc, 2, cz * f)
    eps = L / Ssum
    if eps <= 0.5:
        lr = 2.0 * math.atanh(eps)
    else:
        lr = math.log((Ssum + L) ** 2 / (2.0 * den))
    q = C * lr / L
    _acc(S, Cc, 3, Lx * q)
    _acc(S, Cc, 4, Ly * q)
    _acc(S, Cc, 5, Lz * q)
    if grad:
        gx = Rix / ri + Rfx / rf
        gy = Riy / ri + Rfy / rf
        gz = Riz / ri + Rfz / rf
        c1 = (Ssum * Ssum + L2) / (2.0 * Ssum * den)
        d0 = -f * (gx * c1 + Rix / ri2 + Rfx / rf2)
        d1 = -f * (gy * c1 + Riy / ri2 + Rfy / rf2)
        d2 = -f * (gz * c1 + Riz / ri2 + Rfz / rf2)
        _acc(S, Cc, 6, cx * d0)
        _acc(S, Cc, 7, cx * d1 - Lz * f)
        _acc(S, Cc, 8, cx * d2 + Ly * f)
        _acc(S, Cc, 9, cy * d0 + Lz * f)
        _acc(S, Cc, 10, cy * d1)
        _acc(S, Cc, 11, cy * d2 - Lx * f)
        _acc(S, Cc, 12, cz * d0 - Ly * f)
        _acc(S, Cc, 13, cz * d1 + Lx * f)
        _acc(S, Cc, 14, cz * d2)


# ------------------------------------------------------------------ dipoles
@njit(cache=True, error_model="numpy")
def _dipole(pos, m, x0, x1, x2, grad, S, Cc):
    rx = x0 - pos[0]
    ry = x1 - pos[1]
    rz = x2 - pos[2]
    r2 = rx * rx + ry * ry + rz * rz
    rn = math.sqrt(r2)
    i3 = C_MU0_4PI / (r2 * rn)
    i5 = i3 / r2
    mr = m[0] * rx + m[1] * ry + m[2] * rz
    t = 3.0 * mr * i5
    _acc(S, Cc, 0, rx * t - m[0] * i3)
    _acc(S, Cc, 1, ry * t - m[1] * i3)
    _acc(S, Cc, 2, rz * t - m[2] * i3)
    _acc(S, Cc, 3, (m[1] * rz - m[2] * ry) * i3)
    _acc(S, Cc, 4, (m[2] * rx - m[0] * rz) * i3)
    _acc(S, Cc, 5, (m[0] * ry - m[1] * rx) * i3)
    if grad:
        q = 3.0 * i5
        w = 5.0 * t / r2
        rr = (rx, ry, rz)
        for i in range(3):
            for j in range(3):
                v = rr[i] * m[j] + m[i] * rr[j]
                if i == j:
                    v += mr
                _acc(S, Cc, 6 + 3 * i + j, q * v - rr[i] * rr[j] * w)


# ------------------------------------------------------------------ rings
@njit(cache=True, error_model="numpy")
def ring_kernel_f64(a, zs, rho, z, grad, res):
    """Unit-C ring: res = A, A/rho, B_rho, B_rho/rho, B_z, dB_z/dzeta, dB_z/drho."""
    ring_kernel_off_f64(a, rho, a - rho, z - zs, grad, res)


@njit(cache=True, error_model="numpy")
def ring_kernel_off_f64(a, rho, amr, zeta, grad, res):
    """Offsets amr = a - rho, zeta = z - zs given directly."""
    apr = a + rho
    z2 = zeta * zeta
    al2 = amr * amr + z2
    be2 = apr * apr + z2
    al = math.sqrt(al2)
    be = math.sqrt(be2)
    ab = al * be
    u = 4.0 * ab
    v = (al + be) ** 2
    R, Ru, Rv = rd_dual0_f64(u, v)
    K0 = 32.0 * a * a / 3.0
    res[1] = K0 * R
    res[0] = res[1] * rho
    res[3] = -K0 * zeta * (4.0 * (al2 + be2) * Ru + 2.0 * v * Rv) / ab
    res[2] = res[3] * rho
    du = 8.0 * rho * (z2 - amr * apr) / ab
    if amr >= 0.0:
        dn = apr * al + amr * be
        Delta = 4.0 * a * rho * z2 / dn if dn > 0.0 else 0.0
    else:
        Delta = apr * al - amr * be
    dv = 2.0 * (al + be) * Delta / ab
    res[4] = K0 * (2.0 * R + rho * (Ru * du + Rv * dv))
    if grad:
        P, Py, Pz = rd_dual0_f64(be2, al2)
        Q, Qy, Qz = rd_dual0_f64(al2, be2)
        K1 = 4.0 * a / 3.0
        dPdz = 2.0 * zeta * (Py + Pz)
        dQdz = 2.0 * zeta * (Qy + Qz)
        dPdr = 2.0 * apr * Py - 2.0 * amr * Pz
        dQdr = 2.0 * apr * Qz - 2.0 * amr * Qy
        res[5] = K1 * (amr * dPdz + apr * dQdz)
        res[6] = K1 * (Q - P + amr * dPdr + apr * dQdr)


@njit(cache=True, error_model="numpy")
def _cyl_to_cart(x0, x1, rho, A, Br, Bror, Bz, drBr, dzBr, drBz, dzBz, grad, S, Cc):
    if rho > 0.0:
        c = x0 / rho
        s = x1 / rho
    else:
        c = 1.0
        s = 0.0
    _acc(S, Cc, 0, Br * c)
    _acc(S, Cc, 1, Br * s)
    _acc(S, Cc, 2, Bz)
    _acc(S, Cc, 3, -A * s)
    _acc(S, Cc, 4, A * c)
    if grad:
        m = (drBr - Bror) * c * s
        _acc(S, Cc, 6, drBr * c * c + Bror * s * s)
        _acc(S, Cc, 7, m)
        _acc(S, Cc, 8, dzBr * c)
        _acc(S, Cc, 9, m)
        _acc(S, Cc, 10, drBr * s * s + Bror * c * c)
        _acc(S, Cc, 11, dzBr * s)
        _acc(S, Cc, 12, drBz * c)
        _acc(S, Cc, 13, drBz * s)
        _acc(S, Cc, 14, dzBz)


@njit(cache=True, error_model="numpy")
def _ring(a, zs, cur, x0, x1, x2, rho, grad, S, Cc, kr):
    if cur == 0.0 or a == 0.0:
        return
    ring_kernel_f64(a, zs, rho, x2, grad, kr)
    C = C_MU0_4PI * cur
    Bror = C * kr[3]
    dzBz = C * kr[5]
    drBz = C * kr[6]
    _cyl_to_cart(x0, x1, rho, C * kr[0], C * kr[2], Bror, C * kr[4], -(Bror + dzBz), drBz, drBz, dzBz,
                 grad, S, Cc)


# ------------------------------------------------------------------ axisymmetric polygons (float64)
@njit(cache=True, error_model="numpy")
def _seg_dist(px, pz, ax, az, bx, bz):
    lx = bx - ax
    lz = bz - az
    ll = lx * lx + lz * lz
    t = ((px - ax) * lx + (pz - az) * lz) / ll if ll > 0 else 0.0
    t = min(1.0, max(0.0, t))
    return math.hypot(px - ax - t * lx, pz - az - t * lz)


@njit(cache=True, error_model="numpy")
def _tri_dist(px, pz, t):
    d = min(_seg_dist(px, pz, t[0, 0], t[0, 1], t[1, 0], t[1, 1]),
            _seg_dist(px, pz, t[1, 0], t[1, 1], t[2, 0], t[2, 1]),
            _seg_dist(px, pz, t[2, 0], t[2, 1], t[0, 0], t[0, 1]))
    c0 = (t[1, 0] - t[0, 0]) * (pz - t[0, 1]) - (t[1, 1] - t[0, 1]) * (px - t[0, 0])
    c1 = (t[2, 0] - t[1, 0]) * (pz - t[1, 1]) - (t[2, 1] - t[1, 1]) * (px - t[1, 0])
    c2 = (t[0, 0] - t[2, 0]) * (pz - t[2, 1]) - (t[0, 1] - t[2, 1]) * (px - t[2, 0])
    if (c0 >= 0 and c1 >= 0 and c2 >= 0) or (c0 <= 0 and c1 <= 0 and c2 <= 0):
        return 0.0
    return d


@njit(cache=True, error_model="numpy")
def _winding(px, pz, V, v0, v1):
    w = 0
    n = v1 - v0
    for i in range(n):
        ax, az = V[v0 + i, 0], V[v0 + i, 1]
        k = v0 + (i + 1) % n
        bx, bz = V[k, 0], V[k, 1]
        if az <= pz:
            if bz > pz and (bx - ax) * (pz - az) - (px - ax) * (bz - az) > 0:
                w += 1
        else:
            if bz <= pz and (bx - ax) * (pz - az) - (px - ax) * (bz - az) < 0:
                w -= 1
    return w


@njit(cache=True, error_model="numpy")
def _poly_area_direct(tris, rho, z, glu, glwu, acc, maxlev):
    """acc[5] += int (A, A/rho, B_rho, B_rho/rho, B_z) over triangles (p outside), a priori subdivision."""
    nt = tris.shape[0]
    cap = nt + 4 * maxlev + 16
    st = np.empty((cap, 3, 2))
    stL = np.empty(cap, dtype=np.int64)
    kr = np.zeros(7)
    sp = 0
    for t in range(nt):
        st[sp] = tris[t]
        stL[sp] = 0
        sp += 1
    n = glu.shape[0]
    while sp > 0:
        sp -= 1
        T = st[sp].copy()
        lev = stL[sp]
        d = _tri_dist(rho, z, T)
        diam = max(math.hypot(T[1, 0] - T[0, 0], T[1, 1] - T[0, 1]),
                   math.hypot(T[2, 0] - T[1, 0], T[2, 1] - T[1, 1]),
                   math.hypot(T[0, 0] - T[2, 0], T[0, 1] - T[2, 1]))
        if d < 2.0 * diam and lev < maxlev:
            m01 = 0.5 * (T[0] + T[1])
            m12 = 0.5 * (T[1] + T[2])
            m20 = 0.5 * (T[2] + T[0])
            st[sp, 0] = T[0]; st[sp, 1] = m01; st[sp, 2] = m20
            st[sp + 1, 0] = m01; st[sp + 1, 1] = T[1]; st[sp + 1, 2] = m12
            st[sp + 2, 0] = m20; st[sp + 2, 1] = m12; st[sp + 2, 2] = T[2]
            st[sp + 3, 0] = m12; st[sp + 3, 1] = m20; st[sp + 3, 2] = m01
            for c in range(4):
                stL[sp + c] = lev + 1
            sp += 4
            continue
        e1r = T[1, 0] - T[0, 0]
        e1z = T[1, 1] - T[0, 1]
        e2r = T[2, 0] - T[0, 0]
        e2z = T[2, 1] - T[0, 1]
        jac = abs(e1r * e2z - e1z * e2r)
        for i in range(n):
            u = glu[i]
            wu = glwu[i] * u * jac
            for j in range(n):
                v = glu[j]
                Rp = T[0, 0] + u * (e1r + v * (e2r - e1r))
                Zp = T[0, 1] + u * (e1z + v * (e2z - e1z))
                ring_kernel_f64(Rp, Zp, rho, z, False, kr)
                w = wu * glwu[j]
                for k in range(5):
                    acc[k] += w * kr[k]


@njit(cache=True, error_model="numpy")
def _poly_edge(mode, rho, z, ar, az, br, bz, glx, glw, deu, dew, acc, maxlev):
    """mode 0: polar integral (acc[5], unscaled by c_k); mode 1: boundary integrals (acc[3])."""
    cap = 2 * maxlev + 8
    stS = np.empty(cap)
    stH = np.empty(cap)
    stL = np.empty(cap, dtype=np.int64)
    kr = np.zeros(7)
    er = br - ar
    ez = bz - az
    elen = math.hypot(er, ez)
    sp = 1
    stS[0] = 0.5
    stH[0] = 0.5
    stL[0] = 0
    n = glx.shape[0]
    while sp > 0:
        sp -= 1
        c = stS[sp]
        h = stH[sp]
        lev = stL[sp]
        d = _seg_dist(rho, z, ar + (c - h) * er, az + (c - h) * ez, ar + (c + h) * er, az + (c + h) * ez)
        if d < 1.5 * h * elen and lev < maxlev:
            stS[sp] = c + 0.5 * h
            stS[sp + 1] = c - 0.5 * h
            stH[sp] = 0.5 * h
            stH[sp + 1] = 0.5 * h
            stL[sp] = lev + 1
            stL[sp + 1] = lev + 1
            sp += 2
            continue
        for i in range(n):
            s = c + h * glx[i]
            ws = h * glw[i]
            qr = ar + s * er
            qz = az + s * ez
            if mode == 1:
                ring_kernel_f64(qr, qz, rho, z, False, kr)
                acc[0] += ws * kr[0]
                acc[1] += ws * kr[2]
                acc[2] += ws * (kr[4] - kr[1])
            else:
                wr = qr - rho
                wz = qz - z
                for j in range(deu.shape[0]):
                    u = deu[j]
                    ring_kernel_off_f64(rho + u * wr, rho, u * wr, -u * wz, False, kr)
                    wu = ws * dew[j] * u
                    for k in range(5):
                        acc[k] += wu * kr[k]


@njit(cache=True, error_model="numpy")
def _polygon(V, v0, v1, tris, cur, area, x0, x1, x2, rho, grad, glu, glwu, glx, glw, deu, dew, S, Cc):
    if cur == 0.0:
        return
    nv = v1 - v0
    J = C_MU0_4PI * cur / abs(area)
    orient = 1.0 if area > 0 else -1.0
    z = x2
    d = np.inf
    for i in range(nv):
        k = v0 + (i + 1) % nv
        di = _seg_dist(rho, z, V[v0 + i, 0], V[v0 + i, 1], V[k, 0], V[k, 1])
        if di < d:
            d = di
    inside = _winding(rho, z, V, v0, v1) != 0
    acc = np.zeros(5)
    maxlev = 50
    if (not inside) and d > 0.0:
        _poly_area_direct(tris, rho, z, glu, glwu, acc, maxlev)
    else:
        for i in range(nv):
            k = v0 + (i + 1) % nv
            ar, az, br, bz = V[v0 + i, 0], V[v0 + i, 1], V[k, 0], V[k, 1]
            ck = (ar - rho) * (bz - az) - (az - z) * (br - ar)
            if ck == 0.0:
                continue
            e = np.zeros(5)
            _poly_edge(0, rho, z, ar, az, br, bz, glx, glw, deu, dew, e, maxlev)
            for q in range(5):
                acc[q] += orient * ck * e[q]
    A = J * acc[0]
    Br = J * acc[2]
    Bror = J * acc[3]
    Bz = J * acc[4]
    drBr = 0.0
    dzBr = 0.0
    drBz = 0.0
    dzBz = 0.0
    if grad:
        if d == 0.0:
            drBr = dzBr = drBz = dzBz = np.nan
        else:
            b = np.zeros(5)
            for i in range(nv):
                k = v0 + (i + 1) % nv
                ar, az, br, bz = V[v0 + i, 0], V[v0 + i, 1], V[k, 0], V[k, 1]
                if br == ar:
                    continue
                e = np.zeros(5)
                _poly_edge(1, rho, z, ar, az, br, bz, glx, glw, deu, dew, e, maxlev)
                dR = orient * (br - ar)
                for q in range(3):
                    b[q] += dR * e[q]
            dzBr = J * b[1]
            drBr = -J * b[2]
            dzBz = -(drBr + Bror)
            drBz = dzBr - (4.0 * math.pi * J if inside else 0.0)
    _cyl_to_cart(x0, x1, rho, A, Br, Bror, Bz, drBr, dzBr, drBz, dzBz, grad, S, Cc)


# ------------------------------------------------------------------ one point, all parts
@njit(cache=True, error_model="numpy")
def eval_point(P, x0, x1, x2, grad, out):
    """out[15] = B, A, dB (row-major dB_i/dx_j) at (x0, x1, x2) for the packed source P."""
    (fc_coef, fc_cur, fc_pos, fc_der, fc_harc, fc_M0, fc_Lc, glx, glw, glx_dd,
     sg_a, sg_b, sg_cur, dp_pos, dp_m, rg_R, rg_Z, rg_cur,
     pg_V, pg_vst, pg_tri, pg_tst, pg_cur, pg_area, glu, glwu, glx1, glw1, deu, dew, RB) = P
    S = np.zeros(15)
    Cc = np.zeros(15)
    dmin = np.inf
    M0 = fc_M0[0]
    Lc = fc_Lc[0]
    for ci in range(fc_cur.shape[0]):
        if fc_cur[ci] == 0.0:
            continue
        d = _curve_coil(fc_coef[ci], fc_pos[ci], fc_der[ci], fc_harc[ci], M0, Lc, glx, glw, glx_dd, x0, x1, x2, grad,
                        48, S, Cc, fc_cur[ci])
        if d < dmin:
            dmin = d
    for s in range(sg_cur.shape[0]):
        _segment(sg_a[s], sg_b[s], sg_cur[s], x0, x1, x2, grad, S, Cc)
    for s in range(dp_pos.shape[0]):
        _dipole(dp_pos[s], dp_m[s], x0, x1, x2, grad, S, Cc)
    rho = math.sqrt(x0 * x0 + x1 * x1)
    if rg_cur.shape[0] > 0:
        kr = np.zeros(7)
        for s in range(rg_cur.shape[0]):
            _ring(rg_R[s], rg_Z[s], rg_cur[s], x0, x1, x2, rho, grad, S, Cc, kr)
    for ip in range(pg_cur.shape[0]):
        _polygon(pg_V, pg_vst[ip], pg_vst[ip + 1], pg_tri[pg_tst[ip]:pg_tst[ip + 1]], pg_cur[ip], pg_area[ip],
                 x0, x1, x2, rho, grad, glu, glwu, glx1, glw1, deu, dew, S, Cc)
    if RB != 0.0:
        r2 = x0 * x0 + x1 * x1
        f = RB / r2
        _acc(S, Cc, 0, -f * x1)
        _acc(S, Cc, 1, f * x0)
        _acc(S, Cc, 5, -0.5 * RB * math.log(r2))
        if grad:
            g = f / r2
            _acc(S, Cc, 6, 2.0 * g * x0 * x1)
            _acc(S, Cc, 7, g * (x1 * x1 - x0 * x0))
            _acc(S, Cc, 9, g * (x1 * x1 - x0 * x0))
            _acc(S, Cc, 10, -2.0 * g * x0 * x1)
    for k in range(15):
        out[k] = S[k] + Cc[k]
    return dmin


@njit(cache=True, error_model="numpy")
def eval_point_B(P, x0, x1, x2):
    out = np.empty(15)
    eval_point(P, x0, x1, x2, False, out)
    return out[0], out[1], out[2]


@njit(parallel=True, cache=True, error_model="numpy")
def eval_points(P, pts, grad):
    N = pts.shape[0]
    out = np.zeros((N, 15))
    dmin = np.zeros(N)
    for p in prange(N):
        o = np.empty(15)
        dmin[p] = eval_point(P, pts[p, 0], pts[p, 1], pts[p, 2], grad, o)
        for k in range(15):
            out[p, k] = o[k]
    return out, dmin


# ------------------------------------------------------------------ packing
def _parts_of(source):
    if hasattr(source, "parts"):
        return list(source.parts)
    if isinstance(source, (list, tuple)):
        return list(source)
    return [source]


class Field:
    """float64 reference field of a source (see module docstring)."""

    def __init__(self, source, Lc=5):
        parts = _parts_of(source)
        fc = [p for p in parts if p.kind == "fourier"]
        seg = [p for p in parts if p.kind == "segments"] + [p.as_segments() for p in parts if p.kind == "polyline"]
        dip = [p for p in parts if p.kind == "dipoles"]
        rng = [p for p in parts if p.kind == "rings"]
        pg = [p for p in parts if p.kind == "axipoly"]
        RB = float(sum(p.RB for p in parts if p.kind == "toroidal"))
        # Fourier: pad to a common order
        if fc:
            K = max(p.order for p in fc)
            coefs = []
            curs = []
            for p in fc:
                c = np.zeros((p.n, 3, 2 * K + 1))
                c[:, :, :p.coef.shape[2]] = p.coef
                coefs.append(c)
                curs.append(p.current)
            coef = np.concatenate(coefs)
            cur = np.concatenate(curs)
        else:
            K = 1
            coef = np.zeros((0, 3, 3))
            cur = np.zeros(0)
        M0 = 4
        while M0 < K:
            M0 *= 2
        glx_dd, glw_dd = gauss_legendre_dd(N_GL)
        if len(cur):
            pos, der = build_cache_dd(coef, M0, Lc, glx_dd)
            hi_, lo_ = pos.real.copy(), pos.imag.copy()     # dd values are normalised: hi = round(hi + lo)
            posf = np.concatenate([hi_, lo_], axis=-1)       # (nc, npan, n, 6)
            derf = der.real + der.imag
            speed = np.sqrt((derf ** 2).sum(-1))                 # (nc, npan, n)
            levels = np.concatenate([np.full(M0 * 2 ** l, l) for l in range(Lc + 1)])
            half = 0.5 / (M0 * 2.0 ** levels)
            glw = glw_dd.real + glw_dd.imag
            harc = 0.5 * (speed * glw[None, None, :]).sum(-1) * half[None, :]
        else:
            posf = np.zeros((0, 1, N_GL, 6))
            derf = np.zeros((0, 1, N_GL, 3))
            harc = np.zeros((0, 1))
        glx, glw = gauss_legendre_f64(N_GL)
        if seg:
            sa = np.concatenate([s.a for s in seg])
            sb = np.concatenate([s.b for s in seg])
            sc = np.concatenate([s.current for s in seg])
        else:
            sa = np.zeros((0, 3))
            sb = np.zeros((0, 3))
            sc = np.zeros(0)
        dpos = np.concatenate([d.pos for d in dip]) if dip else np.zeros((0, 3))
        dm = np.concatenate([d.moment for d in dip]) if dip else np.zeros((0, 3))
        rR = np.concatenate([r.R for r in rng]) if rng else np.zeros(0)
        rZ = np.concatenate([r.Z for r in rng]) if rng else np.zeros(0)
        rI = np.concatenate([r.current for r in rng]) if rng else np.zeros(0)
        # polygons
        Vs, vst, tris, tst, pc, pa = [], [0], [], [0], [], []
        for p in pg:
            for i in range(p.n):
                P_ = p.vertices[p.start[i]:p.start[i + 1]]
                Vs.append(P_)
                vst.append(vst[-1] + len(P_))
                for (i0, i1, i2) in ear_clip(P_):
                    tris.append([P_[i0], P_[i1], P_[i2]])
                tst.append(len(tris))
                pc.append(p.current[i])
                pa.append(signed_area(P_))
        pV = np.concatenate(Vs) if Vs else np.zeros((0, 2))
        ptri = np.array(tris, dtype=float).reshape(-1, 3, 2)
        gx2, gw2 = gauss_legendre_f64(12)
        glu = 0.5 * (1 + gx2)
        glwu = 0.5 * gw2
        gx1, gw1 = gauss_legendre_f64(20)
        glx1 = gx1
        glw1 = gw1
        deu, dew = tanh_sinh01_f64(1.0 / 8, 3.5)
        self.packed = (np.ascontiguousarray(coef), np.ascontiguousarray(cur), np.ascontiguousarray(posf),
                       np.ascontiguousarray(derf), np.ascontiguousarray(harc), np.array([M0], dtype=np.int64),
                       np.array([Lc], dtype=np.int64), glx, glw, glx_dd,
                       np.ascontiguousarray(sa), np.ascontiguousarray(sb), np.ascontiguousarray(sc),
                       np.ascontiguousarray(dpos), np.ascontiguousarray(dm),
                       np.ascontiguousarray(rR), np.ascontiguousarray(rZ), np.ascontiguousarray(rI),
                       np.ascontiguousarray(pV), np.array(vst, dtype=np.int64), np.ascontiguousarray(ptri),
                       np.array(tst, dtype=np.int64), np.array(pc, dtype=float), np.array(pa, dtype=float),
                       glu, glwu, glx1, glw1, deu, dew, RB)

    def evaluate(self, points, grad=True):
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        out, dmin = eval_points(self.packed, pts, bool(grad))
        N = pts.shape[0]
        r = {"B": out[:, 0:3].copy(), "A": out[:, 3:6].copy(), "dmin_curve": dmin}
        if grad:
            r["dB"] = out[:, 6:15].reshape(N, 3, 3).copy()
        return r

    def B(self, points):
        return self.evaluate(points, grad=False)["B"]


def evaluate(source, points, grad=True):
    return Field(source).evaluate(points, grad=grad)
