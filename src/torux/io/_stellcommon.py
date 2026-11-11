"""Shared helpers of the stellarator loaders (torux.io.*): paths, part builders,
a small check-only Biot-Savart evaluator and the numerical symmetry check.

The Biot-Savart routines here exist only to *verify* loader claims (symmetry,
duplicate detection); they are not the field engine (torux.field) and not the
reference evaluator (torux.reference).

Symmetry check (``verify_symmetry``): a claim (nfp, stellsym) is accepted only if
the field of the expanded parts satisfies, at test points away from all current
elements,

    B(R x) = R B(x)        R = rotation by 2 pi / nfp about z
    B(F x) = -F B(x)       F = diag(1, -1, -1)   (stellarator symmetry)

to a relative tolerance (default 1e-9 of max |B|).  The deviations measured are
returned so that they can be recorded in ``Source.meta['symmetry_check']``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from torux.source import (MU0, FLIP, FourierCurves, PolylineCurves, Segments, Dipoles, Rings, Source,
                        rotation_z)

def data_path(p):
    """The file's path (relative paths are relative to the working directory)."""
    return Path(p)


def rel_path(p):
    """The file's path as written in a source's provenance."""
    return str(p)


def seg_set_maps(a, b, I, M, sign, tol):
    """Does {(M a, M b, sign I)} equal {(a, b, I)} (orientation-free)?  Returns max deviation."""
    from scipy.spatial import cKDTree
    scale = max(float(np.abs(np.concatenate([a, b])).max()), 1e-300)
    Is = max(float(np.abs(I).max()), 1e-300)
    key = np.concatenate([np.concatenate([a, b], 1), np.concatenate([b, a], 1)])
    cur = np.concatenate([I, -I])
    tree = cKDTree(key)
    k2 = np.concatenate([a @ M.T, b @ M.T], 1)
    d, idx = tree.query(k2)
    dev = max(float(d.max()) / scale, float(np.abs(cur[idx] - sign * I).max()) / Is)
    return dev


# --------------------------------------------------------------------------- part builders
def fourier_part(coefs, currents):
    """FourierCurves from a list of (3, 2K_i+1) SIMSOPT-order coefficient arrays (padded to max K)."""
    coefs = [np.asarray(c, dtype=np.float64) for c in coefs]
    if not coefs:
        raise ValueError("no curves")
    nmax = max(c.shape[1] for c in coefs)
    out = np.zeros((len(coefs), 3, nmax))
    for i, c in enumerate(coefs):
        assert c.shape[0] == 3 and c.shape[1] % 2 == 1, c.shape
        out[i, :, :c.shape[1]] = c
    return FourierCurves(out, np.asarray(currents, dtype=np.float64))


def trim_fourier(part, tol=0.0):
    """Drop trailing harmonics that are exactly zero (tol=0) for every curve."""
    K = part.order
    while K > 0:
        blk = part.coef[:, :, 2 * K - 1:2 * K + 1]
        if np.all(np.abs(blk) <= tol):
            K -= 1
        else:
            break
    if K == part.order:
        return part
    return FourierCurves(part.coef[:, :, :2 * K + 1].copy(), part.current.copy())


def transform_fourier(coef, M):
    """Apply a 3x3 linear map to SIMSOPT-order coefficients coef (3, 2K+1) -> (3, 2K+1)."""
    return np.asarray(M) @ np.asarray(coef)


def polyline_part(loops, currents):
    loops = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in loops]
    start = np.zeros(len(loops) + 1, dtype=np.int64)
    start[1:] = np.cumsum([len(p) for p in loops])
    return PolylineCurves(np.concatenate(loops) if loops else np.zeros((0, 3)), start,
                          np.asarray(currents, dtype=np.float64))


# --------------------------------------------------------------------------- check-only Biot-Savart
try:
    import numba as _nb

    @_nb.njit(parallel=True, cache=True, fastmath=False)
    def _bs_q_kernel(r, dr, w, x):
        nx = x.shape[0]
        nq = r.shape[0]
        B = np.zeros((nx, 3))
        for i in _nb.prange(nx):
            bx = 0.0
            by = 0.0
            bz = 0.0
            for q in range(nq):
                dx = x[i, 0] - r[q, 0]
                dy = x[i, 1] - r[q, 1]
                dz = x[i, 2] - r[q, 2]
                d2 = dx * dx + dy * dy + dz * dz
                inv = w[q] / (d2 * np.sqrt(d2))
                lx = dr[q, 0]
                ly = dr[q, 1]
                lz = dr[q, 2]
                bx += (ly * dz - lz * dy) * inv
                by += (lz * dx - lx * dz) * inv
                bz += (lx * dy - ly * dx) * inv
            B[i, 0] = bx
            B[i, 1] = by
            B[i, 2] = bz
        return B

    @_nb.njit(parallel=True, cache=True, fastmath=False)
    def _bs_seg_kernel(a, b, I, x):
        nx = x.shape[0]
        ns = a.shape[0]
        B = np.zeros((nx, 3))
        for i in _nb.prange(nx):
            bx = 0.0
            by = 0.0
            bz = 0.0
            for s in range(ns):
                ax = x[i, 0] - a[s, 0]
                ay = x[i, 1] - a[s, 1]
                az = x[i, 2] - a[s, 2]
                cx = x[i, 0] - b[s, 0]
                cy = x[i, 1] - b[s, 1]
                cz = x[i, 2] - b[s, 2]
                na = np.sqrt(ax * ax + ay * ay + az * az)
                nb = np.sqrt(cx * cx + cy * cy + cz * cz)
                dot = ax * cx + ay * cy + az * cz
                f = I[s] * (na + nb) / (na * nb * (na * nb + dot))
                bx += (ay * cz - az * cy) * f
                by += (az * cx - ax * cz) * f
                bz += (ax * cy - ay * cx) * f
            B[i, 0] = bx
            B[i, 1] = by
            B[i, 2] = bz
        return B

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def _bs_quadrature_points(r, dr, w_current, x, chunk=4096):
    """sum_q w_q I_q dr_q x (x - r_q) / |x - r_q|^3 * mu0/4pi, vectorised over targets."""
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))
    if HAVE_NUMBA:
        return _bs_q_kernel(np.ascontiguousarray(r), np.ascontiguousarray(dr),
                            np.ascontiguousarray(w_current, dtype=np.float64), x) * (MU0 / (4 * np.pi))
    B = np.zeros_like(x)
    for s in range(0, len(r), chunk):
        rr = r[s:s + chunk]
        dd = dr[s:s + chunk] * w_current[s:s + chunk, None]
        d = x[:, None, :] - rr[None, :, :]                        # (nx, nq, 3)
        inv = 1.0 / np.linalg.norm(d, axis=2) ** 3
        B += np.einsum("xq,xqa->xa", inv, np.cross(dd[None, :, :], d))
    return B * (MU0 / (4 * np.pi))


def bs_fourier(part, x, nq=None):
    """Trapezoid-rule Biot-Savart of FourierCurves (exponentially convergent for smooth curves)."""
    if nq is None:
        nq = max(256, 4 * part.order + 16)
    t = np.arange(nq) / nq
    r, dr = part.points(t)
    w = np.repeat(part.current, nq) / nq
    return _bs_quadrature_points(r.reshape(-1, 3), dr.reshape(-1, 3), w, x)


def bs_segments(a, b, current, x, chunk=8192):
    """Exact field of straight segments a->b with current I (standard closed form)."""
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x, dtype=np.float64)))
    if HAVE_NUMBA:
        return _bs_seg_kernel(np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64),
                              np.ascontiguousarray(current, dtype=np.float64), x) * (MU0 / (4 * np.pi))
    B = np.zeros_like(x)
    for s in range(0, len(a), chunk):
        aa, bb, II = a[s:s + chunk], b[s:s + chunk], current[s:s + chunk]
        ra = x[:, None, :] - aa[None]
        rb = x[:, None, :] - bb[None]
        na = np.linalg.norm(ra, axis=2)
        nb = np.linalg.norm(rb, axis=2)
        c = np.cross(ra, rb)
        den = na * nb * (na * nb + np.einsum("xsa,xsa->xs", ra, rb))
        fac = (na + nb) / den * II[None, :]
        B += np.einsum("xs,xsa->xa", fac, c)
    return B * (MU0 / (4 * np.pi))


def bs_polyline(part, x):
    seg = part.as_segments()
    return bs_segments(seg.a, seg.b, seg.current, x)


def bs_dipoles(part, x):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    B = np.zeros_like(x)
    for s in range(0, part.n, 8192):
        p, m = part.pos[s:s + 8192], part.moment[s:s + 8192]
        d = x[:, None, :] - p[None]
        r = np.linalg.norm(d, axis=2)
        md = np.einsum("xsa,sa->xs", d, m)
        B += np.einsum("xs,xsa->xa", 3 * md / r ** 5, d) - np.einsum("xs,sa->xa", 1 / r ** 3, m)
    return B * (MU0 / (4 * np.pi))


def bs_rings(part, x):
    """Exact field of coaxial circular loops (complete elliptic integrals); I > 0 along +phi."""
    from scipy.special import ellipk, ellipe
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    R = np.hypot(x[:, 0], x[:, 1])[:, None]
    z = x[:, 2][:, None]
    a = part.R[None, :]
    dz = z - part.Z[None, :]
    I = part.current[None, :]
    q = (a + R) ** 2 + dz ** 2
    m = 4 * a * R / q
    K, E = ellipk(m), ellipe(m)
    d2 = (a - R) ** 2 + dz ** 2
    c = MU0 * I / (2 * np.pi * np.sqrt(q))
    Bz = c * (K + (a ** 2 - R ** 2 - dz ** 2) / d2 * E)
    with np.errstate(invalid="ignore", divide="ignore"):
        BR = np.where(R > 0, c * dz / R * (-K + (a ** 2 + R ** 2 + dz ** 2) / d2 * E), 0.0)
    BR, Bz = BR.sum(axis=1), Bz.sum(axis=1)
    Rs = R[:, 0]
    cph = np.where(Rs > 0, x[:, 0] / np.where(Rs > 0, Rs, 1), 1.0)
    sph = np.where(Rs > 0, x[:, 1] / np.where(Rs > 0, Rs, 1), 0.0)
    return np.stack([BR * cph, BR * sph, Bz], axis=1)


def bs_parts(parts, x, nq=None):
    B = 0.0
    for p in parts:
        if p.kind == "fourier":
            B = B + bs_fourier(p, x, nq)
        elif p.kind == "polyline":
            B = B + bs_polyline(p, x)
        elif p.kind == "segments":
            B = B + bs_segments(p.a, p.b, p.current, x)
        elif p.kind == "dipoles":
            B = B + bs_dipoles(p, x)
        elif p.kind == "rings":
            B = B + bs_rings(p, x)
        else:
            raise NotImplementedError(p.kind)
    return B


# --------------------------------------------------------------------------- test points / symmetry
def sample_points(parts):
    pts = []
    for p in parts:
        if p.kind == "fourier":
            x, _ = p.points(np.linspace(0, 1, 64, endpoint=False))
            pts.append(x.reshape(-1, 3))
        elif p.kind == "polyline":
            pts.append(p.points)
        elif p.kind == "segments":
            pts.append(np.concatenate([p.a, p.b]))
        elif p.kind == "dipoles":
            pts.append(p.pos)
        elif p.kind == "rings":
            ph = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            pts.append(np.stack([np.outer(p.R, np.cos(ph)).ravel(), np.outer(p.R, np.sin(ph)).ravel(),
                                 np.repeat(p.Z, len(ph))], axis=1))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _min_dist(x, P, chunk=2048):
    out = np.full(len(x), np.inf)
    for s in range(0, len(P), chunk):
        d = np.linalg.norm(x[:, None, :] - P[None, s:s + chunk], axis=2).min(axis=1)
        out = np.minimum(out, d)
    return out


def probe_points(parts, n=24, seed=12345, min_rel_dist=0.15):
    """Points in the toroidal region enclosed by the coils, away from every current element.

    The region is centred on the circle R0 = median coil radius, with a minor radius
    estimated as the median distance of the coil points from that circle.  Points closer
    to the current elements than ``min_rel_dist`` times that minor radius are rejected.
    """
    P = sample_points(parts)
    R = np.hypot(P[:, 0], P[:, 1])
    R0 = float(np.median(R))
    Z0 = float(np.median(P[:, 2]))
    rm = float(np.median(np.hypot(R - R0, P[:, 2] - Z0)))
    if not np.isfinite(rm) or rm <= 0:
        rm = 0.3 * R0
    rng = np.random.default_rng(seed)
    pts = []
    for scale in (0.6, 0.4, 0.25, 0.9, 1.3):
        m = 8 * n
        rho = scale * rm * np.sqrt(rng.random(m))
        th = 2 * np.pi * rng.random(m)
        ph = 2 * np.pi * rng.random(m)
        RR = R0 + rho * np.cos(th)
        ZZ = Z0 + rho * np.sin(th)
        x = np.stack([RR * np.cos(ph), RR * np.sin(ph), ZZ], axis=1)
        ok = _min_dist(x, P) > min_rel_dist * rm
        pts.extend(list(x[ok]))
        if len(pts) >= n:
            break
    if len(pts) < 4:
        raise RuntimeError("could not place symmetry test points away from the coils")
    return np.asarray(pts[:n])


def symmetry_deviation(parts, nfp, stellsym, x=None, nq=None):
    """Relative deviations (rot, flip) of the field from the claimed symmetry at test points."""
    if x is None:
        x = probe_points(parts)
    B = bs_parts(parts, x, nq)
    scale = float(np.max(np.linalg.norm(B, axis=1)))
    rot = flip = None
    if nfp > 1:
        Rm = rotation_z(2 * np.pi / nfp)
        BR = bs_parts(parts, x @ Rm.T, nq)
        rot = float(np.max(np.linalg.norm(BR - B @ Rm.T, axis=1)) / scale)
    if stellsym:
        BF = bs_parts(parts, x @ FLIP.T, nq)
        flip = float(np.max(np.linalg.norm(BF + B @ FLIP.T, axis=1)) / scale)
    return rot, flip, scale


def verify_symmetry(parts, nfp_candidates=(1,), stellsym_claim=True, tol=1e-9, nq=None, x=None):
    """Largest candidate nfp (and its divisors) and stellsym that the field satisfies numerically.

    Returns (nfp, stellsym, record) where record holds the measured deviations of every
    claim tested.
    """
    if x is None:
        x = probe_points(parts)
    cands = sorted({int(c) for c in nfp_candidates if c and int(c) >= 1}, reverse=True)
    tried = {}
    nfp_ok = 1
    for c in cands:
        if c == 1:
            break
        divs = [d for d in range(c, 1, -1) if c % d == 0]
        found = False
        for d in divs:
            if d in tried:
                dev = tried[d]
            else:
                dev, _, _ = symmetry_deviation(parts, d, False, x=x, nq=nq)
                tried[d] = dev
            if dev <= tol:
                nfp_ok = d
                found = True
                break
        if found:
            break
    ss = False
    fdev = None
    if stellsym_claim:
        _, fdev, _ = symmetry_deviation(parts, 1, True, x=x, nq=nq)
        ss = bool(fdev <= tol)
    rec = {"rot_dev": {str(k): v for k, v in tried.items()}, "flip_dev": fdev, "tol": tol,
           "n_points": int(len(x)), "method": "Biot-Savart at test points: B(Rx)=RB(x), B(Fx)=-FB(x)"}
    return nfp_ok, ss, rec


def coefficient_symmetry(part, nfp, stellsym, tol=1e-12):
    """Fast exact check for FourierCurves whose copies are parametrisation-preserving images.

    True if for every curve i the rotated (and, for stellsym, the flipped) coefficient set
    with the same current equals some curve j with the same current (flip: current reversed),
    to tol relative.  A False result does not disprove the symmetry (the copies may be
    reparametrised); use verify_symmetry then.
    """
    C = part.coef
    I = part.current
    scale = max(np.abs(C).max(), 1e-300)
    Iscale = max(np.abs(I).max(), 1e-300)
    flat = C.reshape(len(C), -1)

    def matches(M, sgn):
        D = np.einsum("ab,nbk->nak", M, C).reshape(len(C), -1)
        used = np.zeros(len(C), bool)
        for i in range(len(C)):
            d = np.abs(flat - D[i][None, :]).max(axis=1) / scale
            di = np.abs(I - sgn * I[i]) / Iscale
            ok = np.where((d <= tol) & (di <= tol))[0]
            if len(ok) == 0:
                return False
            used[ok[0]] = True
        return bool(used.all())

    res = True
    if nfp > 1:
        res = res and matches(rotation_z(2 * np.pi / nfp), 1.0)
    if stellsym:
        res = res and matches(FLIP, -1.0)
    return res


def apply_verified_symmetry(src, nfp_candidates, stellsym_claim=True, tol=1e-9, fast=True):
    """Set src.nfp / src.stellsym to the verified values and record the evidence in meta."""
    cands = [int(c) for c in nfp_candidates if c]
    if fast and len(src.parts) == 1 and src.parts[0].kind == "fourier" and cands:
        c = max(cands)
        if coefficient_symmetry(src.parts[0], c, stellsym_claim):
            src.nfp, src.stellsym = c, bool(stellsym_claim)
            src.meta["symmetry_check"] = {"method": "exact coefficient matching of transformed curves "
                                          "(rotation 2pi/nfp; flip F with current reversal)",
                                          "nfp": c, "stellsym": bool(stellsym_claim), "tol": 1e-12}
            return src
    nfp, ss, rec = verify_symmetry(src.parts, cands or [1], stellsym_claim, tol=tol)
    src.nfp, src.stellsym = int(nfp), bool(ss)
    rec["nfp"] = nfp
    rec["stellsym"] = ss
    src.meta["symmetry_check"] = rec
    return src


# --------------------------------------------------------------------------- summaries
def source_stats(src):
    """n_filaments, filament_kind, max_order_or_points, total_|I| of a Source."""
    nf = 0
    kinds = []
    mx = 0
    totI = 0.0
    for p in src.parts:
        if p.kind == "fourier":
            nf += p.n
            kinds.append("fourier")
            mx = max(mx, p.order)
            totI += float(np.abs(p.current).sum())
        elif p.kind == "polyline":
            nf += p.n
            kinds.append("polyline")
            mx = max(mx, int(np.diff(p.start).max()) if p.n else 0)
            totI += float(np.abs(p.current).sum())
        elif p.kind == "segments":
            nf += p.n
            kinds.append("segments")
            totI += float(np.abs(p.current).sum())
        elif p.kind == "rings":
            nf += p.n
            kinds.append("rings")
            totI += float(np.abs(p.current).sum())
        else:
            kinds.append(p.kind)
    return nf, "+".join(sorted(set(kinds))), mx, totI


# --------------------------------------------------------------------------- vector potential / flux
def a_fourier(part, x, nq=None):
    """Vector potential of FourierCurves: mu0 I / 4 pi \\oint dr / |x - r| (trapezoid rule)."""
    if nq is None:
        nq = max(256, 4 * part.order + 16)
    t = np.arange(nq) / nq
    r, dr = part.points(t)
    r = r.reshape(-1, 3)
    dr = dr.reshape(-1, 3) * (np.repeat(part.current, nq) / nq)[:, None]
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    A = np.zeros_like(x)
    for s in range(0, len(r), 4096):
        d = np.linalg.norm(x[:, None, :] - r[None, s:s + 4096], axis=2)
        A += (1.0 / d) @ dr[s:s + 4096]
    return A * (MU0 / (4 * np.pi))


def a_segments(a, b, current, x):
    """Exact vector potential of straight segments: mu0 I/4pi u ln((ra+rb+L)/(ra+rb-L))."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    A = np.zeros_like(x)
    L = np.linalg.norm(b - a, axis=1)
    u = (b - a) / L[:, None] * current[:, None]
    for s in range(0, len(a), 4096):
        ra = np.linalg.norm(x[:, None, :] - a[None, s:s + 4096], axis=2)
        rb = np.linalg.norm(x[:, None, :] - b[None, s:s + 4096], axis=2)
        LL = L[None, s:s + 4096]
        A += np.log((ra + rb + LL) / (ra + rb - LL)) @ u[s:s + 4096]
    return A * (MU0 / (4 * np.pi))


def a_parts(parts, x):
    A = 0.0
    for p in parts:
        if p.kind == "fourier":
            A = A + a_fourier(p, x)
        elif p.kind == "polyline":
            seg = p.as_segments()
            A = A + a_segments(seg.a, seg.b, seg.current, x)
        elif p.kind == "segments":
            A = A + a_segments(p.a, p.b, p.current, x)
        elif p.kind == "rings":
            # ring: A_phi from elliptic integrals
            from scipy.special import ellipk, ellipe
            x = np.atleast_2d(np.asarray(x, dtype=np.float64))
            R = np.hypot(x[:, 0], x[:, 1])[:, None]
            dz = x[:, 2][:, None] - p.Z[None]
            a_ = p.R[None]
            m = 4 * a_ * R / ((a_ + R) ** 2 + dz ** 2)
            k = np.sqrt(m)
            Aphi = MU0 * p.current[None] / (np.pi * k) * np.sqrt(a_ / R) * ((1 - m / 2) * ellipk(m) - ellipe(m))
            Aphi = Aphi.sum(axis=1)
            A = A + np.stack([-Aphi * x[:, 1] / R[:, 0], Aphi * x[:, 0] / R[:, 0], 0 * Aphi], axis=1)
        else:
            raise NotImplementedError(p.kind)
    return A


def toroidal_flux_through(parts, surf, phi=0.0, ntheta=256):
    """\\oint A . dl around the theta-loop of a surface at quadpoint phi, oriented so that the
    result is the flux of B through the loop along +e_phi (positive when B_phi > 0)."""
    th = np.arange(ntheta) / ntheta
    r, _, rt = surf.gamma_and_derivs(np.full_like(th, phi), th)
    A = a_parts(parts, r)
    flux = float(np.sum(np.einsum("ia,ia->i", A, rt)) / ntheta)
    # orientation: the loop's normal (by the right-hand rule) must point along +e_phi
    c = r.mean(axis=0)
    nvec = 0.5 * np.sum(np.cross(r - c, rt), axis=0) / ntheta
    ephi = np.array([-np.sin(np.arctan2(c[1], c[0])), np.cos(np.arctan2(c[1], c[0])), 0.0])
    return flux if float(nvec @ ephi) > 0 else -flux
