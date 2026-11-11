"""Precise tier, closed-form parts in double-double: Segments (and PolylineCurves),
Dipoles, Rings, ToroidalField.

Straight segment a -> b (current I), Ri = x - a, Rf = x - b, ri = |Ri|, rf = |Rf|,
L = |b - a|, S = ri + rf:
    B = mu0 I/(4 pi) (Ri x Rf) S / (ri rf den),   den = ri rf + Ri.Rf = ((S^2 - L^2)/2)
(Ri x Rf = L x Ri = L x Rf: the form with the smallest rounding bound is used, i.e. L x (shorter
of Ri, Rf) away from the segment, free of the R/L cancellation of Ri x Rf there.)
    A = mu0 I/(4 pi) (b - a)/L  ln((S + L)/(S - L))
den is evaluated without cancellation: ri rf + Ri.Rf when Ri.Rf >= 0 (x off the
segment's "shadow", the Hanson-Hirshman regime), |Ri x Rf|^2/(ri rf - Ri.Rf) otherwise
(x beside the segment).  ln((S+L)/(S-L)) = 2 atanh(L/S) (series for L/S <= 0.05) or
ln((S+L)^2/(2 den)).  Gradient: d(Ri x Rf)/dx_j = (b - a) x e_j and
d ln f/dx_j = -g_j (S^2+L^2)/(2 S den) - Ri_j/ri^2 - Rf_j/rf^2, g = Ri/ri + Rf/rf, f = S/(ri rf den).

Ring of radius a at height Z0 (unit current along +phi), field point (rho, z),
zeta = z - Z0, alpha^2 = (a-rho)^2 + zeta^2, beta^2 = (a+rho)^2 + zeta^2,
u = 4 alpha beta, v = (alpha+beta)^2 (the Landen-transformed moduli):
    A_phi = (32/3) C a^2 rho R_D(0, u, v)                      (C = mu0 I / 4 pi)
    B_rho = -(32/3) C a^2 rho zeta/(alpha beta) [4(alpha^2+beta^2) R_D,u + 2 v R_D,v]
    B_z   = (32/3) C a^2 [2 R_D + rho (R_D,u du/drho + R_D,v dv/drho)]
(all terms of A_phi and B_rho of one sign: no cancellation near the axis or far away).
Gradient from P = R_D(0, beta^2, alpha^2), Q = R_D(0, alpha^2, beta^2) and their
derivatives:  B_z = (4Ca/3)[(a-rho) P + (a+rho) Q],  B_rho = (4 C a zeta/3)(P - Q).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import dd
from .carlson import rd_dual0_dd
from .constants import MU0_4PI_HI, MU0_4PI_LO

RTOL_ROUND = 1e-29     # dd rounding allowance relative to the L1 scale of closed forms


# ------------------------------------------------------------------ segments
@njit(cache=True, error_model="numpy")
def seg_kernel_dd(x0, x1, x2, a, b, grad, out):
    """Unit-current (C = 1) segment a->b at dd point x.  out[15] (dd) is SET.
    Returns (|B|, |A|, |dB|, cond) as floats; cond >= 1 amplification of rounding in dB."""
    z = complex(0.0, 0.0)
    Rix = dd.sub(x0, complex(a[0], 0.0))
    Riy = dd.sub(x1, complex(a[1], 0.0))
    Riz = dd.sub(x2, complex(a[2], 0.0))
    Rfx = dd.sub(x0, complex(b[0], 0.0))
    Rfy = dd.sub(x1, complex(b[1], 0.0))
    Rfz = dd.sub(x2, complex(b[2], 0.0))
    s, e = dd.two_sum(b[0], -a[0])
    Lx = complex(s, e)
    s, e = dd.two_sum(b[1], -a[1])
    Ly = complex(s, e)
    s, e = dd.two_sum(b[2], -a[2])
    Lz = complex(s, e)
    L2 = dd.add(dd.add(dd.sqr(Lx), dd.sqr(Ly)), dd.sqr(Lz))
    for k in range(15):
        out[k] = z
    if L2.real == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    L = dd.sqrt(L2)
    ri2 = dd.add(dd.add(dd.sqr(Rix), dd.sqr(Riy)), dd.sqr(Riz))
    rf2 = dd.add(dd.add(dd.sqr(Rfx), dd.sqr(Rfy)), dd.sqr(Rfz))
    ri = dd.sqrt(ri2)
    rf = dd.sqrt(rf2)
    dot = dd.dot3(Rix, Riy, Riz, Rfx, Rfy, Rfz)
    P = dd.mul(ri, rf)
    # Ri x Rf = L x Ri = L x Rf: the form with the smallest rounding bound (|Ri||Rf| beside the
    # segment, |L| min(|Ri|, |Rf|) farther away, where Ri x Rf would cancel by R/L)
    if max(ri.real, rf.real) <= L.real:
        cx, cy, cz = dd.cross3(Rix, Riy, Riz, Rfx, Rfy, Rfz)
    elif ri.real <= rf.real:
        cx, cy, cz = dd.cross3(Lx, Ly, Lz, Rix, Riy, Riz)
    else:
        cx, cy, cz = dd.cross3(Lx, Ly, Lz, Rfx, Rfy, Rfz)
    if not dd.is_neg(dot):
        den = dd.add(P, dot)
    else:
        c2 = dd.add(dd.add(dd.sqr(cx), dd.sqr(cy)), dd.sqr(cz))
        den = dd.div(c2, dd.sub(P, dot))
    S = dd.add(ri, rf)
    if den.real <= 0.0:
        for k in range(15):
            out[k] = complex(np.nan, 0.0)
        return np.inf, np.inf, np.inf, np.inf
    f = dd.div(S, dd.mul(P, den))
    out[0] = dd.mul(cx, f)
    out[1] = dd.mul(cy, f)
    out[2] = dd.mul(cz, f)
    eps = dd.div(L, S)
    if eps.real <= 0.05:
        lr = dd.mul_pow2(dd.atanh_small(eps), 2.0)
    else:
        lr = dd.log(dd.div(dd.sqr(dd.add(S, L)), dd.mul_pow2(den, 2.0)))
    q = dd.div(lr, L)
    out[3] = dd.mul(Lx, q)
    out[4] = dd.mul(Ly, q)
    out[5] = dd.mul(Lz, q)
    nb = abs(f.real) * np.sqrt(cx.real ** 2 + cy.real ** 2 + cz.real ** 2)
    na = abs(lr.real)
    ng = 0.0
    cond = 1.0
    if grad:
        gx = dd.add(dd.div(Rix, ri), dd.div(Rfx, rf))
        gy = dd.add(dd.div(Riy, ri), dd.div(Rfy, rf))
        gz = dd.add(dd.div(Riz, ri), dd.div(Rfz, rf))
        c1 = dd.div(dd.add(dd.sqr(S), L2), dd.mul(dd.mul_pow2(S, 2.0), den))
        mf = dd.neg(f)
        d0 = dd.mul(mf, dd.add(dd.mul(gx, c1), dd.add(dd.div(Rix, ri2), dd.div(Rfx, rf2))))
        d1 = dd.mul(mf, dd.add(dd.mul(gy, c1), dd.add(dd.div(Riy, ri2), dd.div(Rfy, rf2))))
        d2 = dd.mul(mf, dd.add(dd.mul(gz, c1), dd.add(dd.div(Riz, ri2), dd.div(Rfz, rf2))))
        lxf = dd.mul(Lx, f)
        lyf = dd.mul(Ly, f)
        lzf = dd.mul(Lz, f)
        out[6] = dd.mul(cx, d0)
        out[7] = dd.sub(dd.mul(cx, d1), lzf)
        out[8] = dd.add(dd.mul(cx, d2), lyf)
        out[9] = dd.add(dd.mul(cy, d0), lzf)
        out[10] = dd.mul(cy, d1)
        out[11] = dd.sub(dd.mul(cy, d2), lxf)
        out[12] = dd.sub(dd.mul(cz, d0), lyf)
        out[13] = dd.add(dd.mul(cz, d1), lxf)
        out[14] = dd.mul(cz, d2)
        ng = 0.0
        for k in range(6, 15):
            ng += out[k].real ** 2
        ng = np.sqrt(ng)
        cond = 1.0 + S.real / np.sqrt(den.real)
    return nb, na, ng, cond


@njit(parallel=True, cache=True, error_model="numpy")
def segments_dd(points, a, b, cur, grad):
    N = points.shape[0]
    S = a.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    dmin = np.full(N, np.inf)
    c0 = complex(MU0_4PI_HI, MU0_4PI_LO)
    for p in prange(N):
        x0 = complex(points[p, 0], 0.0)
        x1 = complex(points[p, 1], 0.0)
        x2 = complex(points[p, 2], 0.0)
        tmp = np.zeros(15, dtype=np.complex128)
        for s in range(S):
            if cur[s] == 0.0:
                continue
            nb, na, ng, cond = seg_kernel_dd(x0, x1, x2, a[s], b[s], grad, tmp)
            C = dd.mul_d(c0, cur[s])
            ac = abs(C.real)
            for k in range(15):
                out[p, k] = dd.add(out[p, k], dd.mul(C, tmp[k]))
            scale[p, 0] += ac * nb
            scale[p, 1] += ac * na
            scale[p, 2] += ac * ng
            err[p, 2] += ac * ng * cond * RTOL_ROUND
            # distance to the segment (diagnostic, float64)
            ax, ay, az = points[p, 0] - a[s, 0], points[p, 1] - a[s, 1], points[p, 2] - a[s, 2]
            lx, ly, lz = b[s, 0] - a[s, 0], b[s, 1] - a[s, 1], b[s, 2] - a[s, 2]
            ll = lx * lx + ly * ly + lz * lz
            tt = (ax * lx + ay * ly + az * lz) / ll if ll > 0 else 0.0
            tt = min(1.0, max(0.0, tt))
            d = np.sqrt((ax - tt * lx) ** 2 + (ay - tt * ly) ** 2 + (az - tt * lz) ** 2)
            if d < dmin[p]:
                dmin[p] = d
        err[p, 0] += RTOL_ROUND * scale[p, 0]
        err[p, 1] += RTOL_ROUND * scale[p, 1]
    return out, err, scale, dmin


# ------------------------------------------------------------------ dipoles
@njit(parallel=True, cache=True, error_model="numpy")
def dipoles_dd(points, pos, mom, grad):
    N = points.shape[0]
    D = pos.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    dmin = np.full(N, np.inf)
    c0 = complex(MU0_4PI_HI, MU0_4PI_LO)
    for p in prange(N):
        x0 = complex(points[p, 0], 0.0)
        x1 = complex(points[p, 1], 0.0)
        x2 = complex(points[p, 2], 0.0)
        acc = np.zeros(15, dtype=np.complex128)
        for s in range(D):
            rx = dd.sub(x0, complex(pos[s, 0], 0.0))
            ry = dd.sub(x1, complex(pos[s, 1], 0.0))
            rz = dd.sub(x2, complex(pos[s, 2], 0.0))
            mx = complex(mom[s, 0], 0.0)
            my = complex(mom[s, 1], 0.0)
            mz = complex(mom[s, 2], 0.0)
            r2 = dd.add(dd.add(dd.sqr(rx), dd.sqr(ry)), dd.sqr(rz))
            rn = dd.sqrt(r2)
            if rn.real < dmin[p]:
                dmin[p] = rn.real
            inv3 = dd.recip(dd.mul(r2, rn))
            inv5 = dd.div(inv3, r2)
            mr = dd.dot3(mx, my, mz, rx, ry, rz)
            t = dd.mul_d(dd.mul(mr, inv5), 3.0)       # 3 (m.r)/r^5
            bx = dd.sub(dd.mul(rx, t), dd.mul(mx, inv3))
            by = dd.sub(dd.mul(ry, t), dd.mul(my, inv3))
            bz = dd.sub(dd.mul(rz, t), dd.mul(mz, inv3))
            acc[0] = dd.add(acc[0], bx)
            acc[1] = dd.add(acc[1], by)
            acc[2] = dd.add(acc[2], bz)
            cx, cy, cz = dd.cross3(mx, my, mz, rx, ry, rz)
            acc[3] = dd.add(acc[3], dd.mul(cx, inv3))
            acc[4] = dd.add(acc[4], dd.mul(cy, inv3))
            acc[5] = dd.add(acc[5], dd.mul(cz, inv3))
            mn = np.sqrt(mom[s, 0] ** 2 + mom[s, 1] ** 2 + mom[s, 2] ** 2)
            scale[p, 0] += 2.0 * mn * inv3.real
            scale[p, 1] += mn * inv3.real * rn.real
            if grad:
                # dB_i/dx_j = 3/r^5 [delta_ij (m.r) + r_i m_j + m_i r_j - 5 r_i r_j (m.r)/r^2]
                q = dd.mul_d(inv5, 3.0)
                w = dd.mul_d(dd.div(t, r2), 5.0)     # 15 (m.r)/r^7
                rr = (rx, ry, rz)
                mm = (mx, my, mz)
                for i in range(3):
                    for j in range(3):
                        v = dd.add(dd.mul(rr[i], mm[j]), dd.mul(mm[i], rr[j]))
                        if i == j:
                            v = dd.add(v, mr)
                        g = dd.sub(dd.mul(q, v), dd.mul(dd.mul(rr[i], rr[j]), w))
                        acc[6 + 3 * i + j] = dd.add(acc[6 + 3 * i + j], g)
                scale[p, 2] += 9.0 * mn * inv3.real / rn.real
        for k in range(15):
            out[p, k] = dd.mul(c0, acc[k])
        for g in range(3):
            scale[p, g] *= MU0_4PI_HI
            err[p, g] = RTOL_ROUND * scale[p, g]
    return out, err, scale, dmin


# ------------------------------------------------------------------ rings
@njit(cache=True, error_model="numpy")
def ring_kernel_dd(a, zs, rho, z, grad, res):
    """Unit-C ring (radius a, height zs; dd) at (rho, z) (dd).
    res[0..6] (dd) = A_phi, A_phi/rho, B_rho, B_rho/rho, B_z, dB_z/dzeta, dB_z/drho
    (the last two only if grad).  Returns an L1-type magnitude for rounding estimates."""
    return ring_kernel_off_dd(a, rho, dd.sub(a, rho), dd.sub(z, zs), grad, res)


@njit(cache=True, error_model="numpy")
def ring_kernel_off_dd(a, rho, amr, zeta, grad, res):
    """As ring_kernel_dd with the offsets amr = a - rho and zeta = z - zs given directly
    (exact for source points generated as p + u w, where a - rho = u w_R underflows rho + u w_R)."""
    apr = dd.add(a, rho)
    z2 = dd.sqr(zeta)
    al2 = dd.add(dd.sqr(amr), z2)
    be2 = dd.add(dd.sqr(apr), z2)
    al = dd.sqrt(al2)
    be = dd.sqrt(be2)
    ab = dd.mul(al, be)
    u = dd.mul_d(ab, 4.0)
    v = dd.sqr(dd.add(al, be))
    R, Ru, Rv = rd_dual0_dd(u, v)
    K0 = dd.div_d(dd.mul_d(dd.sqr(a), 32.0), 3.0)
    Aor = dd.mul(K0, R)
    res[1] = Aor
    res[0] = dd.mul(Aor, rho)
    t = dd.add(dd.mul(dd.mul_d(dd.add(al2, be2), 4.0), Ru), dd.mul(dd.mul_d(v, 2.0), Rv))
    Bror = dd.neg(dd.div(dd.mul(dd.mul(K0, zeta), t), ab))
    res[3] = Bror
    res[2] = dd.mul(Bror, rho)
    du = dd.div(dd.mul(dd.mul_d(rho, 8.0), dd.sub(z2, dd.mul(amr, apr))), ab)
    if not dd.is_neg(amr):
        dn = dd.add(dd.mul(apr, al), dd.mul(amr, be))
        if dn.real > 0.0:
            Delta = dd.div(dd.mul(dd.mul_d(dd.mul(a, rho), 4.0), z2), dn)
        else:
            Delta = complex(0.0, 0.0)
    else:
        Delta = dd.sub(dd.mul(apr, al), dd.mul(amr, be))
    dv = dd.div(dd.mul(dd.mul_d(dd.add(al, be), 2.0), Delta), ab)
    tz1 = dd.mul_pow2(R, 2.0)
    tz2 = dd.mul(rho, dd.add(dd.mul(Ru, du), dd.mul(Rv, dv)))
    res[4] = dd.mul(K0, dd.add(tz1, tz2))
    mag = abs(K0.real) * (abs(tz1.real) + abs(tz2.real) + abs(R.real) * (1.0 + rho.real)) + abs(res[2].real)
    if grad:
        P, Py, Pz = rd_dual0_dd(be2, al2)    # Py = dP/d(beta^2), Pz = dP/d(alpha^2)
        Q, Qy, Qz = rd_dual0_dd(al2, be2)    # Qy = dQ/d(alpha^2), Qz = dQ/d(beta^2)
        K1 = dd.div_d(dd.mul_d(a, 4.0), 3.0)
        z2x = dd.mul_pow2(zeta, 2.0)
        dPdz = dd.mul(z2x, dd.add(Py, Pz))
        dQdz = dd.mul(z2x, dd.add(Qy, Qz))
        amr2 = dd.mul_pow2(amr, 2.0)
        apr2 = dd.mul_pow2(apr, 2.0)
        dPdr = dd.sub(dd.mul(apr2, Py), dd.mul(amr2, Pz))
        dQdr = dd.sub(dd.mul(apr2, Qz), dd.mul(amr2, Qy))
        res[5] = dd.mul(K1, dd.add(dd.mul(amr, dPdz), dd.mul(apr, dQdz)))
        res[6] = dd.mul(K1, dd.add(dd.sub(Q, P), dd.add(dd.mul(amr, dPdr), dd.mul(apr, dQdr))))
        mag += abs(K1.real) * (abs(P.real) + abs(Q.real) + abs(amr.real * dPdr.real) + abs(apr.real * dQdr.real)
                               + abs(amr.real * dPdz.real) + abs(apr.real * dQdz.real))
    return mag


@njit(cache=True, error_model="numpy")
def ring_cart_dd(x0, x1, rho, kr, grad, out):
    """Cartesian B, A, dB (dd, out[15] SET) from ring kernel results kr (scaled by C already)."""
    if rho.real > 0.0:
        c = dd.div(x0, rho)
        s = dd.div(x1, rho)
    else:
        c = complex(1.0, 0.0)
        s = complex(0.0, 0.0)
    A = kr[0]
    Br = kr[2]
    Bror = kr[3]
    Bz = kr[4]
    out[0] = dd.mul(Br, c)
    out[1] = dd.mul(Br, s)
    out[2] = Bz
    out[3] = dd.neg(dd.mul(A, s))
    out[4] = dd.mul(A, c)
    out[5] = complex(0.0, 0.0)
    if grad:
        dzBz = kr[5]
        drBz = kr[6]
        drBr = dd.neg(dd.add(Bror, dzBz))
        dzBr = drBz
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
def rings_dd(points, R, Z, cur, grad):
    N = points.shape[0]
    nr = R.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    dmin = np.full(N, np.inf)
    c0 = complex(MU0_4PI_HI, MU0_4PI_LO)
    for p in prange(N):
        x0 = complex(points[p, 0], 0.0)
        x1 = complex(points[p, 1], 0.0)
        z = complex(points[p, 2], 0.0)
        rho = dd.sqrt(dd.add(dd.sqr(x0), dd.sqr(x1)))
        kr = np.zeros(7, dtype=np.complex128)
        tmp = np.zeros(15, dtype=np.complex128)
        for s in range(nr):
            if cur[s] == 0.0 or R[s] == 0.0:
                continue
            mag = ring_kernel_dd(complex(R[s], 0.0), complex(Z[s], 0.0), rho, z, grad, kr)
            C = dd.mul_d(c0, cur[s])
            for k in range(7):
                kr[k] = dd.mul(C, kr[k])
            ring_cart_dd(x0, x1, rho, kr, grad, tmp)
            for k in range(15):
                out[p, k] = dd.add(out[p, k], tmp[k])
            ac = abs(C.real)
            nb = np.sqrt(tmp[0].real ** 2 + tmp[1].real ** 2 + tmp[2].real ** 2)
            scale[p, 0] += nb
            scale[p, 1] += abs(kr[0].real)
            ng = 0.0
            for k in range(6, 15):
                ng += tmp[k].real ** 2
            scale[p, 2] += np.sqrt(ng)
            for g in range(3):
                err[p, g] += RTOL_ROUND * 10.0 * ac * mag
            d = np.sqrt((rho.real - R[s]) ** 2 + (z.real - Z[s]) ** 2)
            if d < dmin[p]:
                dmin[p] = d
    return out, err, scale, dmin


# ------------------------------------------------------------------ toroidal field
@njit(parallel=True, cache=True, error_model="numpy")
def toroidal_dd(points, RB, grad):
    N = points.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    rb = complex(RB, 0.0)
    for p in prange(N):
        x = complex(points[p, 0], 0.0)
        y = complex(points[p, 1], 0.0)
        r2 = dd.add(dd.sqr(x), dd.sqr(y))
        if r2.real == 0.0:
            for k in range(15):
                out[p, k] = complex(np.nan, 0.0)
            continue
        f = dd.div(rb, r2)
        out[p, 0] = dd.neg(dd.mul(f, y))
        out[p, 1] = dd.mul(f, x)
        out[p, 5] = dd.neg(dd.mul_pow2(dd.mul(rb, dd.log(r2)), 0.5))      # A_z = -RB ln R
        if grad:
            g = dd.div(f, r2)                                             # RB / R^4
            xy2 = dd.mul_pow2(dd.mul(x, y), 2.0)
            d = dd.sub(dd.sqr(y), dd.sqr(x))
            out[p, 6] = dd.mul(g, xy2)
            out[p, 7] = dd.mul(g, d)
            out[p, 9] = dd.mul(g, d)
            out[p, 10] = dd.neg(dd.mul(g, xy2))
        scale[p, 0] = abs(f.real) * np.sqrt(r2.real)
        scale[p, 1] = abs(out[p, 5].real)
        scale[p, 2] = 2.0 * abs(f.real)
        for g_ in range(3):
            err[p, g_] = 1e-30 * scale[p, g_] + (1e-30 * abs(RB) if g_ == 1 else 0.0)
    return out, err, scale
