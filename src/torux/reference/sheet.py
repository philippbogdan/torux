"""Field of a SurfaceCurrent part outside its boundary, for the independent reference (float64 tier).

    S = SheetField(part)                 # part: torux.source.SurfaceCurrent
    r = S.evaluate(points, grad=True)    # dict B (N,3) [T], A (N,3) [T m], dB (N,3,3) [T/m], dist (N,) [m]
    S.distance(points)                   # signed distance to the boundary (outside positive), m
    S.minor_radius                       # sqrt(mean cross-section area / pi), m

The sheet.  Boundary r(theta, phi) = (R cos phi, R sin phi, Z), R and Z Fourier series in the phase
m theta - n nfp phi (phi the cylindrical angle).  Current per unit dtheta dphi
    J = K |r_phi x r_theta| = c_phi r_theta - c_theta r_phi,
where (c_theta, c_phi) are the angular derivatives of the current potential kappa (with the secular parts I / 2 pi
and G / 2 pi) or, in the covariant form, the given (B . r_theta, B . r_phi) / mu0.  Both follow from
K = n x grad_s kappa (resp. n x B / mu0) with n along r_phi x r_theta: for any tangential vector v with covariant
components (v_theta, v_phi), (r_phi x r_theta) x v = v_phi r_theta - v_theta r_phi.
    B = mu0/(4 pi) int J x (x - y) / |x - y|^3 dtheta dphi,  A = mu0/(4 pi) int J / |x - y| dtheta dphi,
    dB[i, j] = dB_i / dx_j.

The quadrature (written for this module; the solver's engine uses another discretisation).  An iterated integral,
outer in phi and inner in theta, each a one-dimensional rule built per target point:
  * the points of the boundary nearest to the target (the local minima of the distance; found on a coarse grid and
    refined by Newton's method in both angles) are the centres of the outer rule; the nearest is the pivot;
  * about each centre c the variable is graded by phi = c +- b sinh(u), b = d / sigma (d the distance at the centre,
    sigma the length of r_phi orthogonal to r_theta): the complex singularities of the inner integral at
    c +- i b move to u = +- i pi / 2 for every d, and Gauss-Legendre panels of fixed width in u, also limited in
    their extent in phi by the Fourier content of the boundary, integrate it;
  * for each outer node the inner rule is built the same way about the nearest points of that toroidal cut, with
    b = rho / |r_theta|, rho the distance from the target to the cut;
  * where no point of the boundary is within BCAP (in angle units) of the target the rule is plain Gauss-Legendre
    panels of limited extent.
Positions near the target are never formed absolutely: every quadrature point is written as the pivot plus an
offset computed from exp(i m Theta) - 1 and exp(-i n nfp Phi) - 1 (Theta, Phi the angular offsets from the pivot),
so x - y keeps its relative accuracy however close the target is to the boundary.  Every sum is compensated.

Accuracy (float64, against the double-double tier): B and A within 1e-15 of the L1 norm of the integrand (scale_B,
scale_A; 1e-15 to 3e-15 of |B| where the sheet's field does not cancel), from far away down to 1e-12 of the minor
radius from the boundary.  dB's integrand cancels next to the boundary (terms of size 1 / d summing to a result of
size 1 / a): its error grows to about 1e-16 a / d of |dB|.  Per point, err_B, err_A, err_dB = ROUND * scale_B,
ROUND * scale_A, DB_ROUND * scale_dB bound the rounding (largest component; the measured errors are below a
quarter of these bounds).  Targets inside the boundary, or closer to it than MIN_DIST
times the minor radius, are refused (ValueError).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .quadrules import gauss_legendre_f64

MU0_4PI = 1e-7
N_GL = 12            # Gauss-Legendre nodes per panel (float64 tier)
WU = 1.0             # largest panel width in the graded variable u
C_EXT = 2.0          # largest panel extent in an angle: C_EXT / (largest wavenumber of the boundary in that angle)
L_CAP = 0.5          # ... and never more than L_CAP rad
BCAP = 1.0           # a nearest point is a centre of grading when its b is below BCAP (rad)
MIN_DIST = 1e-12     # targets closer to the boundary than MIN_DIST * minor radius are refused
ROUND = 3e-15        # bound of the rounding error of B and A: ROUND times the L1 norm of the integrand
DB_ROUND = 2e-14     # the same for dB
NODE_MAX = 16384     # nodes of one one-dimensional rule
NC_MAX = 8           # centres of one one-dimensional rule
NQ = 8               # R, Z, R_theta, Z_theta, R_phi, Z_phi, c_theta, c_phi
TWO_PI = 2.0 * math.pi


# ------------------------------------------------------------------ preparation (numpy)
def prepare(part):
    """Arrays of a SurfaceCurrent: modes with m >= 0, complex coefficients C[q] (value = Re sum C[q, k]
    exp(i (m theta - n nfp phi)) + const[q]) for q in R, Z, R_t, Z_t, R_p, Z_p, c_t, c_p, and second derivatives
    C2 = R_tt, R_tp, R_pp, Z_tt, Z_tp, Z_pp."""
    nfp = int(part.nfp)
    modes = np.asarray(part.modes, dtype=np.int64)
    K = len(modes)
    zero = np.zeros(K)

    def g(name):
        v = getattr(part, name)
        return zero.copy() if v is None else np.asarray(v, dtype=float)

    cR = g("rmnc") - 1j * g("rmns")
    cZ = g("zmnc") - 1j * g("zmns")
    const = np.zeros(NQ)
    if part.covariant is not None:
        cov = np.asarray(part.covariant, dtype=float)
        ct = cov[0, 0] - 1j * cov[0, 1]
        cp = cov[1, 0] - 1j * cov[1, 1]
        kap = None
    else:
        kap = g("kappac") - 1j * g("kappas")
        const[6] = float(part.I) / TWO_PI
        const[7] = float(part.G) / TWO_PI
    m = modes[:, 0].copy()
    n = modes[:, 1].copy()
    flip = m < 0
    # cos(-m t + n w) = cos(m t - n w), sin(-psi) = -sin(psi): the conjugate coefficient
    m[flip] = -m[flip]
    n[flip] = -n[flip]
    cR = np.where(flip, np.conj(cR), cR)
    cZ = np.where(flip, np.conj(cZ), cZ)
    if kap is not None:
        kap = np.where(flip, np.conj(kap), kap)
        ct = 1j * m * kap
        cp = -1j * n * nfp * kap
    else:
        ct = np.where(flip, np.conj(ct), ct)
        cp = np.where(flip, np.conj(cp), cp)
    im, iw = 1j * m, -1j * n * nfp
    C = np.ascontiguousarray(np.stack([cR, cZ, im * cR, im * cZ, iw * cR, iw * cZ, ct, cp]).astype(np.complex128))
    C2 = np.ascontiguousarray(np.stack([im * im * cR, im * iw * cR, iw * iw * cR,
                                        im * im * cZ, im * iw * cZ, iw * iw * cZ]).astype(np.complex128))
    keep = np.any(C != 0, axis=0) | np.any(C2 != 0, axis=0)
    if not keep.any():
        keep[0] = True
    m, n, C, C2 = m[keep], n[keep], np.ascontiguousarray(C[:, keep]), np.ascontiguousarray(C2[:, keep])
    geo = (np.ascontiguousarray(m, dtype=np.int64), np.ascontiguousarray(n, dtype=np.int64), C, C2, const)
    Mmax = int(m.max()) if len(m) else 0
    Nmax = int(np.abs(n).max()) if len(n) else 0
    return geo, nfp, Mmax, Nmax


def effective_wavenumbers(geo, nfp, q):
    """The wavenumbers in theta and in phi that set the largest panel extent: max |k| eta_k^(1/(2q)) over the
    modes, eta_k the mode's amplitude relative to the largest of its kind (geometry: R and Z; current: c_theta and
    c_phi), modes with |k| eta_k below 1e-17 left out (they cannot move the field by more than that); at least 1."""
    m, n, C, _, _ = geo
    kt, kp = 1.0, 1.0
    for rows in ((0, 1), (6, 7)):
        amp = np.max(np.abs(C[list(rows)]), axis=0)
        ref = amp[(m != 0) | (n != 0)].max() if np.any(((m != 0) | (n != 0)) & (amp > 0)) else 0.0
        if ref == 0:
            continue
        eta = amp / ref
        for k, (mm, nn) in enumerate(zip(m, n)):
            if eta[k] == 0:
                continue
            w = eta[k] ** (1.0 / (2 * q))
            if mm * eta[k] > 1e-17:
                kt = max(kt, mm * w)
            if abs(nn) * nfp * eta[k] > 1e-17:
                kp = max(kp, abs(nn) * nfp * w)
    return kt, kp


def _grid(geo, nfp, nth, nph):
    """Cartesian points of the boundary on a uniform (phi, theta) grid, and R, Z, R_theta, Z_theta there."""
    m, n, C, _, _ = geo
    th = TWO_PI * np.arange(nth) / nth
    ph = TWO_PI * np.arange(nph) / nph
    Et = np.exp(1j * np.outer(th, m))                     # (nth, K)
    out = np.empty((4, nph, nth))
    for j in range(nph):
        E = Et * np.exp(-1j * n * nfp * ph[j])[None, :]
        for q in range(4):
            out[q, j] = np.real(E @ C[q])
    R, Z, Rt, Zt = out
    pts = np.stack([R * np.cos(ph)[:, None], R * np.sin(ph)[:, None], Z], -1)
    return np.ascontiguousarray(pts), R, Z, Rt, Zt


# ------------------------------------------------------------------ small numba helpers
@njit(cache=True, inline="always", error_model="numpy")
def _acc(s, c, k, v):
    a = s[k]
    t = a + v
    bb = t - a
    c[k] += (a - (t - bb)) + (v - bb)
    s[k] = t


@njit(cache=True, error_model="numpy")
def _wrap(a):
    """a reduced to [-pi, pi)."""
    return a - TWO_PI * math.floor((a + math.pi) / TWO_PI)


@njit(cache=True, error_model="numpy")
def _point_abs(m, n, nfp, C, C2, const, th, ph, out):
    """At absolute angles: out = R, Z, R_t, Z_t, R_p, Z_p, c_t, c_p, R_tt, R_tp, R_pp, Z_tt, Z_tp, Z_pp."""
    for q in range(14):
        out[q] = 0.0
    for k in range(m.shape[0]):
        a = m[k] * th - n[k] * nfp * ph
        ca = math.cos(a)
        sa = math.sin(a)
        for q in range(NQ):
            out[q] += C[q, k].real * ca - C[q, k].imag * sa
        for q in range(6):
            out[8 + q] += C2[q, k].real * ca - C2[q, k].imag * sa
    for q in range(NQ):
        out[q] += const[q]


@njit(cache=True, error_model="numpy")
def _cart(R, Z, Rt, Zt, Rp, Zp, cph, sph, r, rt, rp):
    r[0] = R * cph
    r[1] = R * sph
    r[2] = Z
    rt[0] = Rt * cph
    rt[1] = Rt * sph
    rt[2] = Zt
    rp[0] = Rp * cph - R * sph
    rp[1] = Rp * sph + R * cph
    rp[2] = Zp


@njit(cache=True, error_model="numpy")
def _newton2d(m, n, nfp, C, C2, const, x, th, ph):
    """Nearest point of the boundary to x from (th, ph): (th, ph, distance, sigma, signed side)."""
    v = np.empty(14)
    r = np.empty(3)
    rt = np.empty(3)
    rp = np.empty(3)
    D = np.empty(3)
    for it in range(60):
        _point_abs(m, n, nfp, C, C2, const, th, ph, v)
        cph, sph = math.cos(ph), math.sin(ph)
        _cart(v[0], v[1], v[2], v[3], v[4], v[5], cph, sph, r, rt, rp)
        for i in range(3):
            D[i] = x[i] - r[i]
        f0 = D[0] * D[0] + D[1] * D[1] + D[2] * D[2]
        gt = -(D[0] * rt[0] + D[1] * rt[1] + D[2] * rt[2])
        gp = -(D[0] * rp[0] + D[1] * rp[1] + D[2] * rp[2])
        R, Rt, Rp, Rtt, Rtp, Rpp = v[0], v[2], v[4], v[8], v[9], v[10]
        rtt0, rtt1, rtt2 = v[8] * cph, v[8] * sph, v[11]
        rtp0, rtp1, rtp2 = Rtp * cph - Rt * sph, Rtp * sph + Rt * cph, v[12]
        rpp0, rpp1, rpp2 = Rpp * cph - 2 * Rp * sph - R * cph, Rpp * sph + 2 * Rp * cph - R * sph, v[13]
        htt = rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2] - (D[0] * rtt0 + D[1] * rtt1 + D[2] * rtt2)
        htp = rt[0] * rp[0] + rt[1] * rp[1] + rt[2] * rp[2] - (D[0] * rtp0 + D[1] * rtp1 + D[2] * rtp2)
        hpp = rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2] - (D[0] * rpp0 + D[1] * rpp1 + D[2] * rpp2)
        det = htt * hpp - htp * htp
        if htt > 0 and det > 0:
            dt = -(hpp * gt - htp * gp) / det
            dp = -(htt * gp - htp * gt) / det
        else:
            gtt = rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2]
            gpp = rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2]
            dt = -gt / gtt
            dp = -gp / gpp
        # limit the step, then halve it until the distance does not grow
        lim = 0.3
        sc = max(abs(dt), abs(dp))
        if sc > lim:
            dt *= lim / sc
            dp *= lim / sc
        lam = 1.0
        for ls in range(30):
            th1 = th + lam * dt
            ph1 = ph + lam * dp
            _point_abs(m, n, nfp, C, C2, const, th1, ph1, v)
            c1, s1 = math.cos(ph1), math.sin(ph1)
            e0 = x[0] - v[0] * c1
            e1 = x[1] - v[0] * s1
            e2 = x[2] - v[1]
            if e0 * e0 + e1 * e1 + e2 * e2 <= f0 * (1 + 1e-15) or lam * sc < 1e-15:
                break
            lam *= 0.5
        th, ph = th + lam * dt, ph + lam * dp
        if lam * max(abs(dt), abs(dp)) < 1e-15:
            break
    _point_abs(m, n, nfp, C, C2, const, th, ph, v)
    cph, sph = math.cos(ph), math.sin(ph)
    _cart(v[0], v[1], v[2], v[3], v[4], v[5], cph, sph, r, rt, rp)
    for i in range(3):
        D[i] = x[i] - r[i]
    d = math.sqrt(D[0] * D[0] + D[1] * D[1] + D[2] * D[2])
    gtt = rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2]
    gtp = rt[0] * rp[0] + rt[1] * rp[1] + rt[2] * rp[2]
    gpp = rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2]
    sig = math.sqrt(max(gpp - gtp * gtp / gtt, 1e-300))
    # outward normal r_phi x r_theta
    N0 = rp[1] * rt[2] - rp[2] * rt[1]
    N1 = rp[2] * rt[0] - rp[0] * rt[2]
    N2 = rp[0] * rt[1] - rp[1] * rt[0]
    side = D[0] * N0 + D[1] * N1 + D[2] * N2
    return _wrap(th), _wrap(ph), d, sig, side


@njit(cache=True, error_model="numpy")
def _centres(m, n, nfp, C, C2, const, grid, x, cth, cph, cd, csig):
    """The nearest points of the boundary to x: local minima in phi of the distance to each coarse cut, refined.
    Sorted by distance (cth[0], cph[0] the nearest).  Returns (count, signed side of the nearest)."""
    nph, nth = grid.shape[0], grid.shape[1]
    rowmin = np.empty(nph)
    rowarg = np.empty(nph, dtype=np.int64)
    for j in range(nph):
        best = 1e300
        bi = 0
        for i in range(nth):
            d0 = x[0] - grid[j, i, 0]
            d1 = x[1] - grid[j, i, 1]
            d2 = x[2] - grid[j, i, 2]
            e = d0 * d0 + d1 * d1 + d2 * d2
            if e < best:
                best = e
                bi = i
        rowmin[j] = best
        rowarg[j] = bi
    # candidate rows: local minima (periodic), the global one first
    cand = np.empty(nph, dtype=np.int64)
    nc = 0
    for j in range(nph):
        a = rowmin[(j - 1) % nph]
        b = rowmin[(j + 1) % nph]
        if rowmin[j] <= a and rowmin[j] < b:
            cand[nc] = j
            nc += 1
    if nc == 0:
        cand[0] = int(np.argmin(rowmin))
        nc = 1
    vals = np.empty(nc)
    for k in range(nc):
        vals[k] = rowmin[cand[k]]
    order = np.argsort(vals)
    cnt = 0
    side0 = 0.0
    for kk in range(min(nc, 3 * NC_MAX)):
        j = cand[order[kk]]
        th, ph, d, sig, side = _newton2d(m, n, nfp, C, C2, const, x, TWO_PI * rowarg[j] / nth, TWO_PI * j / nph)
        dup = False
        for q in range(cnt):
            if abs(_wrap(th - cth[q])) < 1e-7 and abs(_wrap(ph - cph[q])) < 1e-7:
                dup = True
        if dup:
            continue
        if cnt < NC_MAX:
            cth[cnt] = th
            cph[cnt] = ph
            cd[cnt] = d
            csig[cnt] = sig
            if cnt == 0:
                side0 = side
            cnt += 1
    # sort by distance
    for a in range(cnt):
        for b in range(a + 1, cnt):
            if cd[b] < cd[a]:
                cth[a], cth[b] = cth[b], cth[a]
                cph[a], cph[b] = cph[b], cph[a]
                cd[a], cd[b] = cd[b], cd[a]
                csig[a], csig[b] = csig[b], csig[a]
    if cnt > 0:
        # side of the nearest, recomputed after sorting
        th, ph, d, sig, side = _newton2d(m, n, nfp, C, C2, const, x, cth[0], cph[0])
        side0 = side
    return cnt, side0


@njit(cache=True, error_model="numpy")
def _build_rule(cent, bsc, nc, Lmax, wu, glx, glw, nodes, wts):
    """A rule for a 2 pi periodic integral in an offset variable: graded about the centres cent[:nc] (ascending,
    scale bsc), plain panels of extent <= Lmax when nc == 0.  Returns the node count (-1: too many nodes)."""
    q = glx.shape[0]
    cnt = 0
    cap = nodes.shape[0]
    if nc == 0:
        npan = max(1, int(math.ceil(TWO_PI / Lmax)))
        h = TWO_PI / npan
        if npan * q > cap:
            return -1
        for p in range(npan):
            a = -math.pi + p * h
            for i in range(q):
                nodes[cnt] = a + 0.5 * h * (1.0 + glx[i])
                wts[cnt] = 0.5 * h * glw[i]
                cnt += 1
        return cnt
    for i in range(nc):
        c = cent[i]
        if nc == 1:
            cl = c - TWO_PI
            cr = c + TWO_PI
        else:
            cl = cent[i - 1] if i > 0 else cent[nc - 1] - TWO_PI
            cr = cent[i + 1] if i < nc - 1 else cent[0] + TWO_PI
        for sd in range(2):
            L = 0.5 * (c - cl) if sd == 0 else 0.5 * (cr - c)
            sgn = -1.0 if sd == 0 else 1.0
            if L <= 0:
                continue
            b = bsc[i]
            U = math.asinh(L / b)
            be = L / math.sinh(U)
            u = 0.0
            while u < U:
                un = min(u + wu, U)
                lim = math.asinh(math.sinh(u) + Lmax / be)
                if lim < un:
                    un = lim
                half = 0.5 * (un - u)
                mid = 0.5 * (un + u)
                if cnt + q > cap:
                    return -1
                for k in range(q):
                    uu = mid + half * glx[k]
                    nodes[cnt] = c + sgn * be * math.sinh(uu)
                    wts[cnt] = half * glw[k] * be * math.cosh(uu)
                    cnt += 1
                u = un
    return cnt


@njit(cache=True, error_model="numpy")
def _row_setup(m, n, nfp, Mmax, Nmax, pv, Phi, P, Bn):
    """Coefficients of the cut at phi = pivot + Phi, about the pivot's theta: P[q, mm] (complex, value of q at
    Theta = Re sum_mm P[q, mm] exp(i mm Theta)), and the offsets (R, Z)(theta*, phi* + Phi) - (R*, Z*)."""
    # Bn[N + k] = exp(-i k nfp Phi) - 1
    s = math.sin(0.5 * nfp * Phi)
    a1 = complex(-2.0 * s * s, -math.sin(nfp * Phi))
    Bn[Nmax] = 0.0
    ak = complex(0.0, 0.0)
    for k in range(1, Nmax + 1):
        ak = ak + a1 + ak * a1
        Bn[Nmax + k] = ak
        Bn[Nmax - k] = ak.conjugate()
    for q in range(NQ):
        for mm in range(Mmax + 1):
            P[q, mm] = 0.0
    dR = 0.0
    dZ = 0.0
    for k in range(m.shape[0]):
        b = Bn[Nmax + n[k]]
        e = 1.0 + b
        for q in range(NQ):
            P[q, m[k]] += pv[q, k] * e
        dR += (pv[0, k] * b).real
        dZ += (pv[1, k] * b).real
    return dR, dZ


@njit(cache=True, error_model="numpy")
def _row_point(P, Mmax, base, dR, dZ, Theta, const, out, second, acc):
    """At theta = theta* + Theta on the current cut: out = dR, dZ (offsets from the pivot), R_t, Z_t, R_p, Z_p,
    c_t, c_p [, R_tt, Z_tt].  base[q] = Re sum_mm P[q, mm] (the values at Theta = 0)."""
    s = math.sin(0.5 * Theta)
    a1r = -2.0 * s * s
    a1i = math.sin(Theta)
    ar = 0.0
    ai = 0.0
    for q in range(NQ + 2):
        acc[q] = 0.0
    for mm in range(1, Mmax + 1):
        nr = ar + a1r + (ar * a1r - ai * a1i)
        ni = ai + a1i + (ar * a1i + ai * a1r)
        ar, ai = nr, ni
        for q in range(NQ):
            acc[q] += P[q, mm].real * ar - P[q, mm].imag * ai
        if second:
            w = -float(mm * mm)
            # second theta derivative of Re P e^{i mm Theta}: -mm^2 Re P (1 + A)
            acc[NQ] += w * (P[0, mm].real * (1.0 + ar) - P[0, mm].imag * ai)
            acc[NQ + 1] += w * (P[1, mm].real * (1.0 + ar) - P[1, mm].imag * ai)
    out[0] = dR + acc[0]
    out[1] = dZ + acc[1]
    for q in range(2, NQ):
        out[q] = base[q] + acc[q] + const[q]
    if second:
        out[NQ] = acc[NQ]
        out[NQ + 1] = acc[NQ + 1]


@njit(cache=True, error_model="numpy")
def _row_frame(c0, s0, Phi):
    """cos and sin of phi* + Phi as offsets from those of phi* (dc, ds) and in full (cp, sp)."""
    sh = math.sin(0.5 * Phi)
    cm1 = -2.0 * sh * sh
    sP = math.sin(Phi)
    dc = c0 * cm1 - s0 * sP
    ds = s0 * cm1 + c0 * sP
    return dc, ds, c0 + dc, s0 + ds


@njit(cache=True, error_model="numpy")
def _row_centres(P, Mmax, base, dR, dZ, const, X0, X1, X2, Rs, dc, ds, cp_, sp_, v, sc, samp, ic, ib):
    """Nearest points of the current cut to the target (offsets Theta from the pivot, ascending, in ic) with their
    grading scales b = rho / |r_theta| (ib), those with b < BCAP only.  Returns their count."""
    nsamp = samp.shape[0]
    # nearest points of this cut: coarse samples, local minima, Newton in Theta
    for i in range(nsamp):
        Th = -math.pi + TWO_PI * i / nsamp
        _row_point(P, Mmax, base, dR, dZ, Th, const, v, False, sc)
        Rn = Rs + v[0]
        e0 = X0 - (Rs * dc + v[0] * cp_)
        e1 = X1 - (Rs * ds + v[0] * sp_)
        e2 = X2 - v[1]
        samp[i] = e0 * e0 + e1 * e1 + e2 * e2
    ni = 0
    # local minima, nearest first
    for i in range(nsamp):
        if samp[i] <= samp[(i - 1) % nsamp] and samp[i] < samp[(i + 1) % nsamp] and ni < NC_MAX:
            Th = -math.pi + TWO_PI * i / nsamp
            for it in range(40):
                _row_point(P, Mmax, base, dR, dZ, Th, const, v, True, sc)
                e0 = X0 - (Rs * dc + v[0] * cp_)
                e1 = X1 - (Rs * ds + v[0] * sp_)
                e2 = X2 - v[1]
                t0, t1, t2 = v[2] * cp_, v[2] * sp_, v[3]
                tt0, tt1, tt2 = v[NQ] * cp_, v[NQ] * sp_, v[NQ + 1]
                g1 = -(e0 * t0 + e1 * t1 + e2 * t2)
                g2 = t0 * t0 + t1 * t1 + t2 * t2 - (e0 * tt0 + e1 * tt1 + e2 * tt2)
                if g2 <= 0:
                    g2 = t0 * t0 + t1 * t1 + t2 * t2
                st = -g1 / g2
                if abs(st) > 0.3:
                    st = 0.3 if st > 0 else -0.3
                Th += st
                if abs(st) < 1e-15:
                    break
            _row_point(P, Mmax, base, dR, dZ, Th, const, v, False, sc)
            e0 = X0 - (Rs * dc + v[0] * cp_)
            e1 = X1 - (Rs * ds + v[0] * sp_)
            e2 = X2 - v[1]
            rho = math.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
            rt = math.sqrt(v[2] * v[2] + v[3] * v[3])
            b = rho / rt
            if b < BCAP:
                Th = _wrap(Th)
                dup = False
                for q in range(ni):
                    if abs(_wrap(Th - ic[q])) < 1e-9:
                        dup = True
                if not dup:
                    ic[ni] = Th
                    ib[ni] = b
                    ni += 1
    for a in range(ni):
        for b2 in range(a + 1, ni):
            if ic[b2] < ic[a]:
                ic[a], ic[b2] = ic[b2], ic[a]
                ib[a], ib[b2] = ib[b2], ib[a]
    return ni


@njit(cache=True, error_model="numpy")
def _eval_target(m, n, nfp, Mmax, Nmax, C, C2, const, grid, Lth, Lph, wu, glx, glw, x, grad, out):
    """B, A, dB (out[0:3], [3:6], [6:15], before the factor mu0 / 4 pi) at x.  Returns (signed distance, nodes,
    flag) with flag 1 when a rule ran out of nodes."""
    cth = np.empty(NC_MAX)
    cph = np.empty(NC_MAX)
    cd = np.empty(NC_MAX)
    csig = np.empty(NC_MAX)
    ncen, side = _centres(m, n, nfp, C, C2, const, grid, x, cth, cph, cd, csig)
    th0 = cth[0]
    ph0 = cph[0]
    dist = cd[0] if side > 0 else -cd[0]
    K = m.shape[0]
    # pivot coefficients
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
    # outer rule
    oc = np.empty(NC_MAX)
    ob = np.empty(NC_MAX)
    no = 0
    for k in range(ncen):
        b = cd[k] / csig[k]
        if b < BCAP:
            oc[no] = _wrap(cph[k] - ph0)
            ob[no] = b
            no += 1
    # ascending
    for a in range(no):
        for b2 in range(a + 1, no):
            if oc[b2] < oc[a]:
                oc[a], oc[b2] = oc[b2], oc[a]
                ob[a], ob[b2] = ob[b2], ob[a]
    onodes = np.empty(NODE_MAX)
    owts = np.empty(NODE_MAX)
    nout = _build_rule(oc, ob, no, Lph, wu, glx, glw, onodes, owts)
    flag = 0
    if nout < 0:
        for q in range(18):
            out[q] = np.nan
        return dist, 0, 1
    P = np.empty((NQ, Mmax + 1), dtype=np.complex128)
    Bn = np.empty(2 * Nmax + 1, dtype=np.complex128)
    base = np.empty(NQ)
    v = np.empty(NQ + 2)
    sc = np.empty(NQ + 2)
    inodes = np.empty(NODE_MAX)
    iwts = np.empty(NODE_MAX)
    ic = np.empty(NC_MAX)
    ib = np.empty(NC_MAX)
    S = np.zeros(18)
    Sc = np.zeros(18)
    rs = np.zeros(18)
    rc = np.zeros(18)
    nsamp = max(32, 4 * (Mmax + 1))
    samp = np.empty(nsamp)
    total_nodes = 0
    for jo in range(nout):
        Phi = onodes[jo]
        W = owts[jo]
        dR, dZ = _row_setup(m, n, nfp, Mmax, Nmax, pv, Phi, P, Bn)
        for q in range(NQ):
            t = 0.0
            for mm in range(Mmax + 1):
                t += P[q, mm].real
            base[q] = t
        dc, ds, cp_, sp_ = _row_frame(c0, s0, Phi)
        ni = _row_centres(P, Mmax, base, dR, dZ, const, X0, X1, X2, Rs, dc, ds, cp_, sp_, v, sc, samp, ic, ib)
        nin = _build_rule(ic, ib, ni, Lth, wu, glx, glw, inodes, iwts)
        if nin < 0:
            flag = 1
            continue
        total_nodes += nin
        for q in range(18):
            rs[q] = 0.0
            rc[q] = 0.0
        for ji in range(nin):
            _row_point(P, Mmax, base, dR, dZ, inodes[ji], const, v, False, sc)
            w = iwts[ji]
            Rn = Rs + v[0]
            D0 = X0 - (Rs * dc + v[0] * cp_)
            D1 = X1 - (Rs * ds + v[0] * sp_)
            D2 = X2 - v[1]
            rt0, rt1, rt2 = v[2] * cp_, v[2] * sp_, v[3]
            rp0, rp1, rp2 = v[4] * cp_ - Rn * sp_, v[4] * sp_ + Rn * cp_, v[5]
            ct, cpp = v[6], v[7]
            J0 = cpp * rt0 - ct * rp0
            J1 = cpp * rt1 - ct * rp1
            J2 = cpp * rt2 - ct * rp2
            r2 = D0 * D0 + D1 * D1 + D2 * D2
            ir = 1.0 / math.sqrt(r2)
            ir3 = ir * ir * ir * w
            ir1 = ir * w
            X_0 = J1 * D2 - J2 * D1
            X_1 = J2 * D0 - J0 * D2
            X_2 = J0 * D1 - J1 * D0
            _acc(rs, rc, 0, X_0 * ir3)
            _acc(rs, rc, 1, X_1 * ir3)
            _acc(rs, rc, 2, X_2 * ir3)
            _acc(rs, rc, 3, J0 * ir1)
            _acc(rs, rc, 4, J1 * ir1)
            _acc(rs, rc, 5, J2 * ir1)
            # L1 norms of the integrands (B, A, dB), for the rounding estimate
            Jn = math.sqrt(J0 * J0 + J1 * J1 + J2 * J2) * abs(ir1)
            rs[15] += Jn * ir
            rs[16] += Jn
            rs[17] += 4.0 * Jn * ir * ir
            if grad:
                ir5 = 3.0 * ir3 * ir * ir
                # d/dx_j (J x D)_i / r^3 = (J x e_j)_i / r^3 - 3 (J x D)_i D_j / r^5
                _acc(rs, rc, 6, -X_0 * D0 * ir5)
                _acc(rs, rc, 7, -J2 * ir3 - X_0 * D1 * ir5)
                _acc(rs, rc, 8, J1 * ir3 - X_0 * D2 * ir5)
                _acc(rs, rc, 9, J2 * ir3 - X_1 * D0 * ir5)
                _acc(rs, rc, 10, -X_1 * D1 * ir5)
                _acc(rs, rc, 11, -J0 * ir3 - X_1 * D2 * ir5)
                _acc(rs, rc, 12, -J1 * ir3 - X_2 * D0 * ir5)
                _acc(rs, rc, 13, J0 * ir3 - X_2 * D1 * ir5)
                _acc(rs, rc, 14, -X_2 * D2 * ir5)
        for q in range(18):
            _acc(S, Sc, q, W * (rs[q] + rc[q]))
    for q in range(15):
        out[q] = MU0_4PI * (S[q] + Sc[q])
    for q in range(15, 18):
        out[q] = MU0_4PI * abs(S[q])
    return dist, total_nodes, flag


@njit(parallel=True, cache=True, error_model="numpy")
def _eval_points(m, n, nfp, Mmax, Nmax, C, C2, const, grid, Lth, Lph, wu, glx, glw, pts, grad):
    N = pts.shape[0]
    out = np.zeros((N, 18))
    dist = np.zeros(N)
    work = np.zeros(N)
    flag = np.zeros(N, dtype=np.int64)
    for p in prange(N):
        o = np.empty(18)
        x = np.empty(3)
        x[0], x[1], x[2] = pts[p, 0], pts[p, 1], pts[p, 2]
        d, w, f = _eval_target(m, n, nfp, Mmax, Nmax, C, C2, const, grid, Lth, Lph, wu, glx, glw, x, grad, o)
        dist[p] = d
        work[p] = w
        flag[p] = f
        for k in range(18):
            out[p, k] = o[k]
    return out, dist, work, flag


@njit(parallel=True, cache=True, error_model="numpy")
def _distances(m, n, nfp, C, C2, const, grid, pts):
    N = pts.shape[0]
    dist = np.zeros(N)
    for p in prange(N):
        cth = np.empty(NC_MAX)
        cph = np.empty(NC_MAX)
        cd = np.empty(NC_MAX)
        csig = np.empty(NC_MAX)
        x = np.empty(3)
        x[0], x[1], x[2] = pts[p, 0], pts[p, 1], pts[p, 2]
        cnt, side = _centres(m, n, nfp, C, C2, const, grid, x, cth, cph, cd, csig)
        dist[p] = cd[0] if side > 0 else -cd[0]
    return dist


# ------------------------------------------------------------------ the part
class SheetField:
    """float64 field of one SurfaceCurrent part outside its boundary (see the module docstring)."""

    def __init__(self, part, n_gl=N_GL, min_dist=MIN_DIST, wu=WU, c_ext=C_EXT, l_cap=L_CAP):
        self.part = part
        self.geo, self.nfp, self.Mmax, self.Nmax = prepare(part)
        kth, kph = effective_wavenumbers(self.geo, self.nfp, int(n_gl))
        self.Lth = min(l_cap, c_ext / kth)
        self.Lph = min(l_cap, c_ext / kph)
        self.wu = float(wu)
        nth = max(64, 8 * (self.Mmax + 1))
        nph = max(128, 8 * (self.Nmax * self.nfp + 1))
        self.grid, R, Z, Rt, Zt = _grid(self.geo, self.nfp, nth, nph)
        # cross-section areas (exact for the trigonometric polynomial at this sampling), orientation
        area = 0.5 * np.mean(R * Zt - Z * Rt, axis=1) * TWO_PI
        if np.any(area <= 0):
            raise ValueError("SurfaceCurrent: a toroidal cut does not run counterclockwise in (R, Z)")
        if np.any(R <= 0):
            raise ValueError("SurfaceCurrent: R must be positive on the boundary")
        self.minor_radius = float(np.sqrt(np.mean(area) / np.pi))
        self.glx, self.glw = gauss_legendre_f64(int(n_gl))
        self.min_dist = float(min_dist)

    def _args(self):
        m, n, C, C2, const = self.geo
        return m, n, self.nfp, C, C2, const

    def distance(self, points):
        """Signed distance of each point to the boundary, m (outside positive)."""
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        m, n, nfp, C, C2, const = self._args()
        return _distances(m, n, nfp, C, C2, const, self.grid, pts)

    def refuse(self, dist):
        a = self.minor_radius
        inside = dist <= 0
        if inside.any():
            k = int(np.argmin(dist))
            raise ValueError(f"SurfaceCurrent: {int(inside.sum())} of {dist.size} points are inside the boundary "
                             f"(the reference evaluates the sheet only outside it; point {k} is "
                             f"{-dist[k]:.3e} m inside)")
        near = dist < self.min_dist * a
        if near.any():
            k = int(np.argmin(dist))
            raise ValueError(f"SurfaceCurrent: {int(near.sum())} of {dist.size} points are closer to the boundary "
                             f"than {self.min_dist:g} of its minor radius {a:.6g} m, where the reference does not "
                             f"hold its accuracy (point {k} is {dist[k] / a:.3e} of the minor radius outside)")

    def evaluate(self, points, grad=True, check=True):
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        N = pts.shape[0]
        if check and N:
            self.refuse(self.distance(pts))
        m, n, nfp, C, C2, const = self._args()
        out, dist, work, flag = _eval_points(m, n, nfp, self.Mmax, self.Nmax, C, C2, const, self.grid,
                                             self.Lth, self.Lph, self.wu, self.glx, self.glw, pts, bool(grad))
        if flag.any():
            raise ValueError(f"SurfaceCurrent: the quadrature ran out of nodes at {int(flag.sum())} points")
        r = {"B": out[:, 0:3].copy(), "A": out[:, 3:6].copy(), "dist": dist, "work": work,
             "scale_B": out[:, 15].copy(), "scale_A": out[:, 16].copy()}
        r["err_B"] = ROUND * r["scale_B"]
        r["err_A"] = ROUND * r["scale_A"]
        if grad:
            r["dB"] = out[:, 6:15].reshape(N, 3, 3).copy()
            r["scale_dB"] = out[:, 17].copy()
            r["err_dB"] = DB_ROUND * r["scale_dB"]
        return r
