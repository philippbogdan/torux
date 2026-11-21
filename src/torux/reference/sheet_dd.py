"""Double-double tier of the SurfaceCurrent field (the precise tier's part; conventions and quadrature in sheet.py).

    P = SheetPrecise(part)
    out, err, scale, dist, flag, work = P.evaluate(points, grad=True, rtol=1e-20)

out (N, 15) dd (complex128 carriers, see dd.py): B, A, dB as in precise.  The rule of sheet.py with Gauss-Legendre
orders Q1 and Q2 on the same panels, everything after the choice of panels (nodes, weights, the boundary's
Fourier sums, the kernel, the sums) in double-double: err is |I(Q2) - I(Q1)| (max over components of B, A, dB
separately) plus a rounding allowance, the result I(Q2).  Where the error of B or A exceeds rtol times the
integrand's L1 norm (scale) the panels are halved, up to twice; flag 1 if it still does.  The error of dB is reported
as estimated: next to the boundary it is dd rounding amplified by the cancellation of the 1 / r^3 integrand, about
1e-32 (a / d)^2 relative (a the minor radius, d the distance).  The panels and the centres of grading are
chosen in float64 (any choice is a valid rule; only the evaluation of the rule needs the precision).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import dd
from .quadrules import gauss_legendre_dd
from .sheet import (BCAP, NC_MAX, NODE_MAX, NQ, TWO_PI, WU, SheetField, _centres, _row_centres, _row_frame,
                    _row_point, _row_setup, _wrap)

Q1, Q2 = 16, 22
ROUND_REL = 1e-30          # rounding allowance relative to the L1 norm of the integrand


def prepare_dd(part):
    """Exact coefficient arrays: m >= 0, n, and for R, Z, X1, X2 the complex (cos, -sin) pairs as given; X1 = kappa
    (kind 1) or the covariant theta component (kind 0), X2 the covariant phi component (kind 0)."""
    modes = np.asarray(part.modes, dtype=np.int64)
    K = len(modes)
    zero = np.zeros(K)

    def g(name):
        v = getattr(part, name)
        return zero.copy() if v is None else np.asarray(v, dtype=float)

    C = np.zeros((4, K), dtype=np.complex128)
    C[0] = g("rmnc") - 1j * g("rmns")
    C[1] = g("zmnc") - 1j * g("zmns")
    if part.covariant is not None:
        cov = np.asarray(part.covariant, dtype=float)
        C[2] = cov[0, 0] - 1j * cov[0, 1]
        C[3] = cov[1, 0] - 1j * cov[1, 1]
        kind = 0
    else:
        C[2] = g("kappac") - 1j * g("kappas")
        kind = 1
    m = modes[:, 0].copy()
    n = modes[:, 1].copy()
    flip = m < 0
    m[flip] = -m[flip]
    n[flip] = -n[flip]
    C[:, flip] = np.conj(C[:, flip])
    keep = np.any(C != 0, axis=0)
    if not keep.any():
        keep[0] = True
    return (np.ascontiguousarray(m[keep]), np.ascontiguousarray(n[keep]), np.ascontiguousarray(C[:, keep]), kind,
            float(part.I), float(part.G))


# ------------------------------------------------------------------ dd helpers
@njit(cache=True, error_model="numpy")
def _sinh_cosh(u):
    e = dd.exp(u)
    ei = dd.recip(e)
    return dd.mul_pow2(dd.sub(e, ei), 0.5), dd.mul_pow2(dd.add(e, ei), 0.5)


@njit(cache=True, error_model="numpy")
def _build_rule_dd(cent, bsc, nc, Lmax, wu, glx, glw, nodes, wts):
    """sheet._build_rule with the same panels, nodes and weights in double-double."""
    q = glx.shape[0]
    cnt = 0
    cap = nodes.shape[0]
    twopi = complex(dd.TWO_PI_HI, dd.TWO_PI_LO)
    if nc == 0:
        npan = max(1, int(math.ceil(TWO_PI / Lmax)))
        if npan * q > cap:
            return -1
        h = dd.div_d(twopi, float(npan))
        mpi = dd.neg(complex(dd.PI_HI, dd.PI_LO))
        for p in range(npan):
            a = dd.add(mpi, dd.mul_d(h, float(p)))
            hh = dd.mul_pow2(h, 0.5)
            for i in range(q):
                nodes[cnt] = dd.add(a, dd.mul(hh, dd.add_d(glx[i], 1.0)))
                wts[cnt] = dd.mul(hh, glw[i])
                cnt += 1
        return cnt
    for i in range(nc):
        c = cent[i]
        if nc == 1:
            cl = dd.sub(dd.D(c), twopi)
            cr = dd.add(dd.D(c), twopi)
        else:
            cl = dd.D(cent[i - 1]) if i > 0 else dd.sub(dd.D(cent[nc - 1]), twopi)
            cr = dd.D(cent[i + 1]) if i < nc - 1 else dd.add(dd.D(cent[0]), twopi)
        for sd in range(2):
            Ld = dd.mul_pow2(dd.sub(dd.D(c), cl), 0.5) if sd == 0 else dd.mul_pow2(dd.sub(cr, dd.D(c)), 0.5)
            L = Ld.real
            sgn = -1.0 if sd == 0 else 1.0
            if L <= 0:
                continue
            b = bsc[i]
            U = math.asinh(L / b)
            shU, _ = _sinh_cosh(dd.D(U))
            be = dd.div(Ld, shU)
            beh = be.real
            u = 0.0
            while u < U:
                un = min(u + wu, U)
                lim = math.asinh(math.sinh(u) + Lmax / beh)
                if lim < un:
                    un = lim
                half = dd.mul_pow2(dd.sub(dd.D(un), dd.D(u)), 0.5)
                mid = dd.mul_pow2(dd.add(dd.D(un), dd.D(u)), 0.5)
                if cnt + q > cap:
                    return -1
                for k in range(q):
                    uu = dd.add(mid, dd.mul(half, glx[k]))
                    sh, ch = _sinh_cosh(uu)
                    nodes[cnt] = dd.add_d(dd.mul_d(dd.mul(be, sh), sgn), c)
                    wts[cnt] = dd.mul(dd.mul(half, glw[k]), dd.mul(be, ch))
                    cnt += 1
                u = un
    return cnt


@njit(cache=True, error_model="numpy")
def _row_dd(m, n, nfp, Nmax, Cd, ph, P0r, P0i, Ppr, Ppi, E_r, E_i):
    """Row coefficients at the absolute angle ph (dd): P0[g, mm] = sum C[g, k] exp(-i n nfp ph) over the modes of
    m = mm, Pp the same with the factor n nfp."""
    twopi = complex(dd.TWO_PI_HI, dd.TWO_PI_LO)
    c1, s1 = dd.cossin2pi(dd.div(dd.mul_d(ph, float(nfp)), twopi))
    # exp(-i nfp ph) = c1 - i s1
    E_r[Nmax] = complex(1.0, 0.0)
    E_i[Nmax] = complex(0.0, 0.0)
    er = complex(1.0, 0.0)
    ei = complex(0.0, 0.0)
    ms1 = dd.neg(s1)
    for k in range(1, Nmax + 1):
        nr = dd.sub(dd.mul(er, c1), dd.mul(ei, ms1))
        ni = dd.add(dd.mul(er, ms1), dd.mul(ei, c1))
        er, ei = nr, ni
        E_r[Nmax + k] = er
        E_i[Nmax + k] = ei
        E_r[Nmax - k] = er
        E_i[Nmax - k] = dd.neg(ei)
    for g in range(4):
        for mm in range(P0r.shape[1]):
            P0r[g, mm] = 0.0
            P0i[g, mm] = 0.0
            Ppr[g, mm] = 0.0
            Ppi[g, mm] = 0.0
    for k in range(m.shape[0]):
        er = E_r[Nmax + n[k]]
        ei = E_i[Nmax + n[k]]
        w = float(n[k] * nfp)
        mm = m[k]
        for g in range(4):
            cr = Cd[g, k].real
            ci = Cd[g, k].imag
            if cr == 0.0 and ci == 0.0:
                continue
            pr = dd.sub(dd.mul_d(er, cr), dd.mul_d(ei, ci))
            pi = dd.add(dd.mul_d(ei, cr), dd.mul_d(er, ci))
            P0r[g, mm] = dd.add(P0r[g, mm], pr)
            P0i[g, mm] = dd.add(P0i[g, mm], pi)
            if w != 0.0:
                Ppr[g, mm] = dd.add(Ppr[g, mm], dd.mul_d(pr, w))
                Ppi[g, mm] = dd.add(Ppi[g, mm], dd.mul_d(pi, w))


@njit(cache=True, error_model="numpy")
def _integrate(m, n, nfp, Mmax, Nmax, C, C2, const, md, nd, Md, Nd, Cd, kind, Itp, Gtp, grid, Lth, Lph, wu, glx, glw, x, grad,
               th0, ph0, oc, ob, no, out, scale):
    """One rule (GL order of glx) at x: out (15 dd) before the factor mu0 / 4 pi, scale (3) the L1 norms.
    Returns (nodes, flag)."""
    K = m.shape[0]
    # float64 pivot data for the centres of the inner rules
    pv = np.empty((NQ, K), dtype=np.complex128)
    for k in range(K):
        a = m[k] * th0 - n[k] * nfp * ph0
        e = complex(math.cos(a), math.sin(a))
        for q in range(NQ):
            pv[q, k] = C[q, k] * e
    Rs = 0.0
    Zs = 0.0
    for k in range(K):
        Rs += pv[0, k].real
        Zs += pv[1, k].real
    c0, s0 = math.cos(ph0), math.sin(ph0)
    X0 = x[0] - Rs * c0
    X1 = x[1] - Rs * s0
    X2 = x[2] - Zs
    onodes = np.empty(NODE_MAX, dtype=np.complex128)
    owts = np.empty(NODE_MAX, dtype=np.complex128)
    nout = _build_rule_dd(oc, ob, no, Lph, wu, glx, glw, onodes, owts)
    if nout < 0:
        return 0, 1
    P = np.empty((NQ, Mmax + 1), dtype=np.complex128)
    Bn = np.empty(2 * Nmax + 1, dtype=np.complex128)
    base = np.empty(NQ)
    v = np.empty(NQ + 2)
    sc = np.empty(NQ + 2)
    inodes = np.empty(NODE_MAX, dtype=np.complex128)
    iwts = np.empty(NODE_MAX, dtype=np.complex128)
    ic = np.empty(NC_MAX)
    ib = np.empty(NC_MAX)
    samp = np.empty(max(32, 4 * (Mmax + 1)))
    Mp = Md + 1
    P0r = np.empty((4, Mp), dtype=np.complex128)
    P0i = np.empty((4, Mp), dtype=np.complex128)
    Ppr = np.empty((4, Mp), dtype=np.complex128)
    Ppi = np.empty((4, Mp), dtype=np.complex128)
    E_r = np.empty(2 * Nd + 1, dtype=np.complex128)
    E_i = np.empty(2 * Nd + 1, dtype=np.complex128)
    S = np.zeros(15, dtype=np.complex128)
    rs = np.zeros(15, dtype=np.complex128)
    for q in range(3):
        scale[q] = 0.0
    twopi = complex(dd.TWO_PI_HI, dd.TWO_PI_LO)
    x0 = dd.D(x[0])
    x1 = dd.D(x[1])
    x2 = dd.D(x[2])
    itp = dd.div(dd.D(Itp), twopi)
    gtp = dd.div(dd.D(Gtp), twopi)
    flag = 0
    total = 0
    for jo in range(nout):
        Phi = onodes[jo]
        W = owts[jo]
        # centres of the inner rule, in float64
        Phf = Phi.real + Phi.imag
        dR, dZ = _row_setup(m, n, nfp, Mmax, Nmax, pv, Phf, P, Bn)
        for q in range(NQ):
            t = 0.0
            for mm in range(Mmax + 1):
                t += P[q, mm].real
            base[q] = t
        dc, ds, cp_, sp_ = _row_frame(c0, s0, Phf)
        ni = _row_centres(P, Mmax, base, dR, dZ, const, X0, X1, X2, Rs, dc, ds, cp_, sp_, v, sc, samp, ic, ib)
        nin = _build_rule_dd(ic, ib, ni, Lth, wu, glx, glw, inodes, iwts)
        if nin < 0:
            flag = 1
            continue
        total += nin
        # the cut in double-double
        ph = dd.add_d(Phi, ph0)
        _row_dd(md, nd, nfp, Nd, Cd, ph, P0r, P0i, Ppr, Ppi, E_r, E_i)
        cph, sph = dd.cossin2pi(dd.div(ph, twopi))
        for q in range(15):
            rs[q] = 0.0
        for ji in range(nin):
            th = dd.add_d(inodes[ji], th0)
            w = iwts[ji]
            e1r, e1i = dd.cossin2pi(dd.div(th, twopi))
            # sums over mm of P exp(i mm th): S0 (value), St (factor mm), Sp (Pp)
            R = P0r[0, 0]
            Z = P0r[1, 0]
            Rt = complex(0.0, 0.0)
            Zt = complex(0.0, 0.0)
            Rp = Ppi[0, 0]
            Zp = Ppi[1, 0]
            X1t = complex(0.0, 0.0)
            X1p = Ppi[2, 0]
            X1v = P0r[2, 0]
            X2v = P0r[3, 0]
            er = complex(1.0, 0.0)
            ei = complex(0.0, 0.0)
            for mm in range(1, Mp):
                nr = dd.sub(dd.mul(er, e1r), dd.mul(ei, e1i))
                ei = dd.add(dd.mul(er, e1i), dd.mul(ei, e1r))
                er = nr
                fm = float(mm)
                # Re(P e) = Pr er - Pi ei ; Im(P e) = Pr ei + Pi er
                R = dd.add(R, dd.sub(dd.mul(P0r[0, mm], er), dd.mul(P0i[0, mm], ei)))
                Z = dd.add(Z, dd.sub(dd.mul(P0r[1, mm], er), dd.mul(P0i[1, mm], ei)))
                Rt = dd.add(Rt, dd.mul_d(dd.add(dd.mul(P0r[0, mm], ei), dd.mul(P0i[0, mm], er)), fm))
                Zt = dd.add(Zt, dd.mul_d(dd.add(dd.mul(P0r[1, mm], ei), dd.mul(P0i[1, mm], er)), fm))
                Rp = dd.add(Rp, dd.add(dd.mul(Ppr[0, mm], ei), dd.mul(Ppi[0, mm], er)))
                Zp = dd.add(Zp, dd.add(dd.mul(Ppr[1, mm], ei), dd.mul(Ppi[1, mm], er)))
                if kind == 1:
                    X1t = dd.add(X1t, dd.mul_d(dd.add(dd.mul(P0r[2, mm], ei), dd.mul(P0i[2, mm], er)), fm))
                    X1p = dd.add(X1p, dd.add(dd.mul(Ppr[2, mm], ei), dd.mul(Ppi[2, mm], er)))
                else:
                    X1v = dd.add(X1v, dd.sub(dd.mul(P0r[2, mm], er), dd.mul(P0i[2, mm], ei)))
                    X2v = dd.add(X2v, dd.sub(dd.mul(P0r[3, mm], er), dd.mul(P0i[3, mm], ei)))
            # R_t = -Im St, R_p = Im Sp
            Rt = dd.neg(Rt)
            Zt = dd.neg(Zt)
            if kind == 1:
                ct = dd.sub(itp, X1t)
                cpv = dd.add(gtp, X1p)
            else:
                ct = X1v
                cpv = X2v
            y0 = dd.mul(R, cph)
            y1 = dd.mul(R, sph)
            D0 = dd.sub(x0, y0)
            D1 = dd.sub(x1, y1)
            D2 = dd.sub(x2, Z)
            rt0 = dd.mul(Rt, cph)
            rt1 = dd.mul(Rt, sph)
            rp0 = dd.sub(dd.mul(Rp, cph), y1)
            rp1 = dd.add(dd.mul(Rp, sph), y0)
            J0 = dd.sub(dd.mul(cpv, rt0), dd.mul(ct, rp0))
            J1 = dd.sub(dd.mul(cpv, rt1), dd.mul(ct, rp1))
            J2 = dd.sub(dd.mul(cpv, Zt), dd.mul(ct, Zp))
            r2 = dd.add(dd.add(dd.sqr(D0), dd.sqr(D1)), dd.sqr(D2))
            r = dd.sqrt(r2)
            ir = dd.recip(r)
            ir1 = dd.mul(ir, w)
            ir3 = dd.mul(dd.mul(ir1, ir), ir)
            Xc0 = dd.sub(dd.mul(J1, D2), dd.mul(J2, D1))
            Xc1 = dd.sub(dd.mul(J2, D0), dd.mul(J0, D2))
            Xc2 = dd.sub(dd.mul(J0, D1), dd.mul(J1, D0))
            rs[0] = dd.add(rs[0], dd.mul(Xc0, ir3))
            rs[1] = dd.add(rs[1], dd.mul(Xc1, ir3))
            rs[2] = dd.add(rs[2], dd.mul(Xc2, ir3))
            rs[3] = dd.add(rs[3], dd.mul(J0, ir1))
            rs[4] = dd.add(rs[4], dd.mul(J1, ir1))
            rs[5] = dd.add(rs[5], dd.mul(J2, ir1))
            Jn = math.sqrt(J0.real ** 2 + J1.real ** 2 + J2.real ** 2)
            wa = abs(w.real * W.real)
            irf = ir.real
            scale[0] += wa * Jn * irf * irf
            scale[1] += wa * Jn * irf
            if grad:
                ir5 = dd.mul_d(dd.mul(dd.mul(ir3, ir), ir), 3.0)
                a0 = dd.mul(Xc0, ir5)
                a1 = dd.mul(Xc1, ir5)
                a2 = dd.mul(Xc2, ir5)
                rs[6] = dd.sub(rs[6], dd.mul(a0, D0))
                rs[7] = dd.sub(rs[7], dd.add(dd.mul(J2, ir3), dd.mul(a0, D1)))
                rs[8] = dd.sub(dd.add(rs[8], dd.mul(J1, ir3)), dd.mul(a0, D2))
                rs[9] = dd.sub(dd.add(rs[9], dd.mul(J2, ir3)), dd.mul(a1, D0))
                rs[10] = dd.sub(rs[10], dd.mul(a1, D1))
                rs[11] = dd.sub(rs[11], dd.add(dd.mul(J0, ir3), dd.mul(a1, D2)))
                rs[12] = dd.sub(rs[12], dd.add(dd.mul(J1, ir3), dd.mul(a2, D0)))
                rs[13] = dd.sub(dd.add(rs[13], dd.mul(J0, ir3)), dd.mul(a2, D1))
                rs[14] = dd.sub(rs[14], dd.mul(a2, D2))
                scale[2] += 4.0 * wa * Jn * irf * irf * irf
        for q in range(15):
            S[q] = dd.add(S[q], dd.mul(W, rs[q]))
    for q in range(15):
        out[q] = dd.mul_d(S[q], 1e-7)
    for q in range(3):
        scale[q] *= 1e-7
    return total, flag


@njit(cache=True, error_model="numpy")
def _eval_target_dd(m, n, nfp, Mmax, Nmax, C, C2, const, md, nd, Md, Nd, Cd, kind, Itp, Gtp, grid, Lth, Lph, wu, gx1, gw1, gx2,
                    gw2, x, grad, rtol, out, err, scale):
    cth = np.empty(NC_MAX)
    cph = np.empty(NC_MAX)
    cd = np.empty(NC_MAX)
    csig = np.empty(NC_MAX)
    ncen, side = _centres(m, n, nfp, C, C2, const, grid, x, cth, cph, cd, csig)
    th0 = cth[0]
    ph0 = cph[0]
    dist = cd[0] if side > 0 else -cd[0]
    oc = np.empty(NC_MAX)
    ob = np.empty(NC_MAX)
    no = 0
    for k in range(ncen):
        b = cd[k] / csig[k]
        if b < BCAP:
            oc[no] = _wrap(cph[k] - ph0)
            ob[no] = b
            no += 1
    for a in range(no):
        for b2 in range(a + 1, no):
            if oc[b2] < oc[a]:
                oc[a], oc[b2] = oc[b2], oc[a]
                ob[a], ob[b2] = ob[b2], ob[a]
    o1 = np.empty(15, dtype=np.complex128)
    s1 = np.empty(3)
    s2 = np.empty(3)
    flag = 1
    work = 0.0
    fac = 1.0
    for level in range(3):
        n1, f1 = _integrate(m, n, nfp, Mmax, Nmax, C, C2, const, md, nd, Md, Nd, Cd, kind, Itp, Gtp, grid, Lth * fac, Lph * fac,
                            wu * fac, gx1, gw1, x, grad, th0, ph0, oc, ob, no, o1, s1)
        n2, f2 = _integrate(m, n, nfp, Mmax, Nmax, C, C2, const, md, nd, Md, Nd, Cd, kind, Itp, Gtp, grid, Lth * fac, Lph * fac,
                            wu * fac, gx2, gw2, x, grad, th0, ph0, oc, ob, no, out, s2)
        work += n1 + n2
        for q in range(3):
            scale[q] = s2[q]
            err[q] = 0.0
        for q in range(15):
            e = abs(dd.sub(out[q], o1[q]).real)
            g = 0 if q < 3 else (1 if q < 6 else 2)
            if e > err[g]:
                err[g] = e
        for q in range(3):
            err[q] += ROUND_REL * scale[q]
        # B and A decide (dB's estimate is its rounding next to the boundary, err[2], reported as it is)
        ok = f1 == 0 and f2 == 0 and err[0] <= rtol * scale[0] and err[1] <= rtol * scale[1]
        if ok:
            flag = 0
            break
        fac *= 0.5
    return dist, work, flag


@njit(parallel=True, cache=True, error_model="numpy")
def _eval_points_dd(m, n, nfp, Mmax, Nmax, C, C2, const, md, nd, Md, Nd, Cd, kind, Itp, Gtp, grid, Lth, Lph, wu, gx1, gw1, gx2, gw2,
                    pts, grad, rtol):
    N = pts.shape[0]
    out = np.zeros((N, 15), dtype=np.complex128)
    err = np.zeros((N, 3))
    scale = np.zeros((N, 3))
    dist = np.zeros(N)
    work = np.zeros(N)
    flag = np.zeros(N, dtype=np.int64)
    for p in prange(N):
        o = np.empty(15, dtype=np.complex128)
        e = np.empty(3)
        s = np.empty(3)
        x = np.empty(3)
        x[0], x[1], x[2] = pts[p, 0], pts[p, 1], pts[p, 2]
        d, w, f = _eval_target_dd(m, n, nfp, Mmax, Nmax, C, C2, const, md, nd, Md, Nd, Cd, kind, Itp, Gtp, grid, Lth, Lph, wu, gx1,
                                  gw1, gx2, gw2, x, grad, rtol, o, e, s)
        dist[p] = d
        work[p] = w
        flag[p] = f
        for k in range(15):
            out[p, k] = o[k]
        for k in range(3):
            err[p, k] = e[k]
            scale[p, k] = s[k]
    return out, err, scale, dist, work, flag


class SheetPrecise:
    """Double-double field of one SurfaceCurrent part outside its boundary."""

    def __init__(self, part, min_dist=None, wu=WU):
        kw = {} if min_dist is None else {"min_dist": min_dist}
        self.f64 = SheetField(part, n_gl=Q1, wu=wu, **kw)
        self.dd = prepare_dd(part)
        self.gx1, self.gw1 = gauss_legendre_dd(Q1)
        self.gx2, self.gw2 = gauss_legendre_dd(Q2)
        self.minor_radius = self.f64.minor_radius

    def distance(self, points):
        return self.f64.distance(points)

    def evaluate(self, points, grad=True, rtol=1e-20, check=True):
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        if check and len(pts):
            self.f64.refuse(self.f64.distance(pts))
        F = self.f64
        m, n, C, C2, const = F.geo
        md, nd, Cd, kind, I, G = self.dd
        Md, Nd = int(md.max()), int(np.abs(nd).max())
        out, err, scale, dist, work, flag = _eval_points_dd(m, n, F.nfp, F.Mmax, F.Nmax, C, C2, const, md, nd, Md, Nd,
                                                            Cd, kind, I,
                                                            G, F.grid, F.Lth, F.Lph, F.wu, self.gx1, self.gw1,
                                                            self.gx2, self.gw2, pts, bool(grad), float(rtol))
        return out, err, scale, dist, flag, work
