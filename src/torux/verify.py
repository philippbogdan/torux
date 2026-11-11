"""Independent verification of a computed flux surface with the reference field (torux.reference),
which shares no code with the solver's engine.

verify_torus(source, torus) returns:
  bn_max, bn_median    max / median |B.n|/(|B||n|) on a random tensor sample of the torus (64 x 64 random theta,
                       phi a field period; reference B; the torus's points and normals in long double), not the
                       solver's grid;
  bn_sup               the largest value found by local maximisation (Nelder-Mead in theta, phi) from the 16
                       largest samples: the estimate of the maximum over the whole surface;
  flux, flux_rel_err   the enclosed flux by loop integrals of the reference A on random
                       cross-sections (trapezoid in theta, spectrally accurate), against the torus's;
  trace                reference-traced lines started on the torus: largest distance of their
                       section points from the torus cross-section, relative to the minor radius;
  precise              the extended-precision tier at a few points: |B_fast64 - B_precise| / |B|; where fp64 does
                       not decide, B.n at every sample point and the local maximisation again in double-double.
"""
from __future__ import annotations

import time

import numpy as np

from .torus import TorusState


def torus_points(t: TorusState, theta, phi, deriv=False):
    """Cartesian points and (optionally) X_theta x X_phi at arrays theta, phi.  Separable evaluation:
    cos(m th - n ps) = cos m th cos n ps + sin m th sin n ps, the n-sums one BLAS product per point
    set (no (points, M, N) arrays)."""
    theta = np.asarray(theta, float).ravel()
    phi = np.asarray(phi, float).ravel()
    m = np.arange(t.M + 1)
    n = np.arange(-t.N, t.N + 1)
    am = np.outer(theta, m)
    an = np.outer(t.nfp * (phi - t.phi0), n)
    Cm, Sm, Cn, Sn = np.cos(am), np.sin(am), np.cos(an), np.sin(an)
    rcC, rcS = Cn @ t.rc.T, Sn @ t.rc.T            # (P, M+1): sum_n rc_mn cos n ps, sin n ps
    zsC, zsS = Cn @ t.zs.T, Sn @ t.zs.T
    # R = sum rc cos(m th - n ps) = sum_m [Cm rcC + Sm rcS]; Z = sum zs sin(...) = sum_m [Sm zsC - Cm zsS]
    R = np.sum(Cm * rcC + Sm * rcS, axis=1)
    Z = np.sum(Sm * zsC - Cm * zsS, axis=1)
    x = np.stack([R * np.cos(phi), R * np.sin(phi), Z], -1)
    if not deriv:
        return x, R, Z
    # d/dtheta: m factor; d/dphi: -(-n nfp) via the n-weighted sums
    Rt = np.sum(m * (-Sm * rcC + Cm * rcS), axis=1)
    Zt = np.sum(m * (Cm * zsC + Sm * zsS), axis=1)
    nw = n * t.nfp
    rcCn, rcSn = Cn @ (t.rc * nw).T, Sn @ (t.rc * nw).T
    zsCn, zsSn = Cn @ (t.zs * nw).T, Sn @ (t.zs * nw).T
    # d/dphi cos(m th - n ps) = n nfp sin(m th - n ps) = n nfp [Sm Cn - Cm Sn]
    Rp = np.sum(Sm * rcCn - Cm * rcSn, axis=1)
    # d/dphi sin(m th - n ps) = -n nfp cos(m th - n ps) = -n nfp [Cm Cn + Sm Sn]
    Zp = -np.sum(Cm * zsCn + Sm * zsSn, axis=1)
    cp, sp = np.cos(phi), np.sin(phi)
    eR = np.stack([cp, sp, 0 * cp], -1); eP = np.stack([-sp, cp, 0 * cp], -1); eZ = np.stack([0 * cp, 0 * cp, 1 + 0 * cp], -1)
    Xt = Rt[:, None] * eR + Zt[:, None] * eZ
    Xp = Rp[:, None] * eR + R[:, None] * eP + Zp[:, None] * eZ
    return x, R, Z, np.cross(Xt, Xp), (Rt, Zt)


def section_distance(t: TorusState, phi, Rq, Zq, nsample=2048, newton=6):
    """Distance of points (Rq, Zq) in the plane phi from the torus cross-section curve there:
    nearest sample, then Newton on d^2(theta) with the series' derivatives."""
    m = np.arange(t.M + 1)[:, None]
    n = np.arange(-t.N, t.N + 1)[None, :]
    psi = t.nfp * (phi - t.phi0)

    def curve(th, k=0):
        ang = m[None] * th[:, None, None] - n[None] * psi
        c, s = np.cos(ang), np.sin(ang)
        mk = m[None] ** k
        if k % 4 == 0:
            Rv, Zv = np.sum(mk * t.rc * c, (1, 2)), np.sum(mk * t.zs * s, (1, 2))
        elif k % 4 == 1:
            Rv, Zv = np.sum(-mk * t.rc * s, (1, 2)), np.sum(mk * t.zs * c, (1, 2))
        elif k % 4 == 2:
            Rv, Zv = np.sum(-mk * t.rc * c, (1, 2)), np.sum(-mk * t.zs * s, (1, 2))
        else:
            Rv, Zv = np.sum(mk * t.rc * s, (1, 2)), np.sum(-mk * t.zs * c, (1, 2))
        return Rv, Zv
    ths = 2 * np.pi * np.arange(nsample) / nsample
    Rc, Zc = curve(ths)
    Rq = np.asarray(Rq); Zq = np.asarray(Zq)
    j = np.argmin((Rq[:, None] - Rc[None, :]) ** 2 + (Zq[:, None] - Zc[None, :]) ** 2, axis=1)
    th = ths[j]
    for _ in range(newton):
        R0, Z0 = curve(th); R1, Z1 = curve(th, 1); R2, Z2 = curve(th, 2)
        g = (R0 - Rq) * R1 + (Z0 - Zq) * Z1
        h = R1 ** 2 + Z1 ** 2 + (R0 - Rq) * R2 + (Z0 - Zq) * Z2
        th = th - g / np.where(h > 0, h, R1 ** 2 + Z1 ** 2)
    R0, Z0 = curve(th)
    return np.hypot(R0 - Rq, Z0 - Zq)


LD = np.longdouble
REF_VERSION = 2           # 2: the torus's points and normals in long double, and the estimated maximum over the surface


def _series_ld(Cm, Sm, m, Cn, Sn, nw, a, b):
    """F, dF/dtheta, dF/dphi on a tensor grid for F = sum a_mn cos(m th - n ps) + b_mn sin(m th - n ps), ps = nfp
    (phi - phi0): cos(m th - n ps) = Cm Cn + Sm Sn, sin(m th - n ps) = Sm Cn - Cm Sn; the n-sums first (per phi),
    then the m-sums; d/dphi brings n nfp (nw).  a or b may be None."""
    F = Ft = Fp = 0
    if a is not None:
        aC, aS = Cn @ a.T, Sn @ a.T
        awC, awS = Cn @ (a * nw).T, Sn @ (a * nw).T
        F = F + Cm @ aC.T + Sm @ aS.T
        Ft = Ft + (Cm * m) @ aS.T - (Sm * m) @ aC.T
        Fp = Fp + Sm @ awC.T - Cm @ awS.T
    if b is not None:
        bC, bS = Cn @ b.T, Sn @ b.T
        bwC, bwS = Cn @ (b * nw).T, Sn @ (b * nw).T
        F = F + Sm @ bC.T - Cm @ bS.T
        Ft = Ft + (Cm * m) @ bC.T + (Sm * m) @ bS.T
        Fp = Fp - (Cm @ bwC.T + Sm @ bwS.T)
    return F, Ft, Fp


def surface_ld(coefs, nfp, phi0, theta, phi):
    """Points and normals X_theta x X_phi of the Fourier torus R = sum rc cos + rs sin, Z = sum zc cos + zs sin (of
    m theta - n nfp (phi - phi0); coefs = (rc, rs, zc, zs), arrays (M+1, 2N+1) with column j <-> n = j - N, or None)
    on the tensor grid theta x phi (rows theta), in long double (eps 1.1e-19): (x, nvec), each (nt np, 3).

    The fp64 evaluation (BLAS sums, torus_points) puts the points a few ulps off the surface (~1e-15 m); next to a
    coil |grad B| / |B| x 1e-15 m is ~1e-14 of B.n, the size of the bar (gradb QH land_small_lengths 0.649574 /
    1.2519: fp64 B.n 2.5e-14 -- double-double B the same -- at a point where the long-double surface gives 1.1e-15)."""
    a = next(c for c in coefs if c is not None)
    M, N = a.shape[0] - 1, (a.shape[1] - 1) // 2
    th = np.asarray(theta, float).ravel().astype(LD)
    ph = np.asarray(phi, float).ravel().astype(LD)
    m = np.arange(M + 1).astype(LD)
    n = np.arange(-N, N + 1).astype(LD)
    ps = LD(nfp) * (ph - LD(phi0))
    am, an = np.outer(th, m), np.outer(ps, n)
    Cm, Sm, Cn, Sn = np.cos(am), np.sin(am), np.cos(an), np.sin(an)
    nw = n * LD(nfp)
    ld = lambda c: None if c is None or not np.any(c) else np.asarray(c, float).astype(LD)  # noqa: E731
    rc, rs, zc, zs = (ld(c) for c in coefs)
    R, Rt, Rp = _series_ld(Cm, Sm, m, Cn, Sn, nw, rc, rs)
    Z, Zt, Zp = _series_ld(Cm, Sm, m, Cn, Sn, nw, zc, zs)
    cp, sp = np.cos(ph)[None, :], np.sin(ph)[None, :]
    nR, nP, nZ = -R * Zt, Zt * Rp - Rt * Zp, R * Rt
    x = np.stack(np.broadcast_arrays(R * cp, R * sp, Z), -1).reshape(-1, 3)
    nv = np.stack(np.broadcast_arrays(nR * cp - nP * sp, nR * sp + nP * cp, nZ), -1).reshape(-1, 3)
    return x, nv


def _bn_ld(x, nv, B64, dB):
    """|B.n| / (|B| |n|) at long-double points x (normals nv) from B and grad B (fp64) at the points rounded to fp64:
    B moved to x by grad B . (x - fl(x)) (the rounding of the points would otherwise be ~|grad B| / |B| x 1e-16 m)."""
    dx = x - x.astype(float).astype(LD)
    B = np.asarray(B64, float).astype(LD) + np.einsum("pij,pj->pi", np.asarray(dB, float).astype(LD), dx)
    return (np.abs(np.sum(B * nv, -1)) / (np.sqrt(np.sum(B * B, -1)) * np.sqrt(np.sum(nv * nv, -1)))).astype(float)


def _distinct_tops(values, TH, PH, k, hth, hph):
    """Indices of the k largest values, skipping those within 2 steps of one already taken (theta periodic; the
    phi samples lie in one field period)."""
    out = []
    for i in np.argsort(values)[::-1]:
        if all(abs(np.angle(np.exp(1j * (TH[i] - TH[j])))) > 2 * hth or abs(PH[i] - PH[j]) > 2 * hph for j in out):
            out.append(int(i))
            if len(out) == k:
                break
    return out


def zoom_sup(coefs, nfp, phi0, starts, h, bn_of, levels=5, side=9, shrink=4.0):
    """Local maximisation of |B.n| / (|B||n|) from each start (theta, phi) by zooming tensor grids: side x side points
    at spacing h = (h_theta, h_phi) around the start, then around each grid's best point at spacing h / shrink, ...,
    levels times (spacing h / shrink^(levels - 1) at the end).  One field call per level for all starts (bn_of(x, nv)
    on long-double points and normals).  Returns (max, [theta, phi]) over every point evaluated."""
    best, where = -np.inf, None
    cur = [tuple(map(float, s_)) for s_ in starts]
    hth, hph = float(h[0]), float(h[1])
    off = np.arange(side) - side // 2
    for _ in range(levels):
        xs, nvs, grids = [], [], []
        for (a, b) in cur:
            th, ph = a + off * hth, b + off * hph
            x, nv = surface_ld(coefs, nfp, phi0, th, ph)
            xs.append(x); nvs.append(nv); grids.append((th, ph))
        bn = np.asarray(bn_of(np.concatenate(xs), np.concatenate(nvs))).reshape(len(cur), side * side)
        nxt = []
        for k, (th, ph) in enumerate(grids):
            i = int(np.argmax(bn[k]))
            a, b = float(th[i // side]), float(ph[i % side])
            nxt.append((a, b))
            if bn[k, i] > best:
                best, where = float(bn[k, i]), [a, b]
        cur = nxt
        hth, hph = hth / shrink, hph / shrink
    return best, where


def normal_component(source, coefs, nfp, phi0, F, rng, nside=64, precise_pts=8, nstart=16):
    """The reference's |B.n| / (|B| |n|) on a Fourier torus (coefs as surface_ld): on a random tensor sample (nside
    random theta x nside random phi over a field period; points and normals in long double, B from the fp64 reference
    F moved to them by its gradient), then local maximisation (zoom_sup) from the nstart largest distinct samples,
    first steps a quarter of the shortest wavelength of the torus's modes: bn_sup estimates the maximum over the
    whole surface, which the sample maximum misses by up to 2-3x next to coils.  The double-double tier
    (torux.reference.precise) at the 8 largest and 8 random samples measures fp64's own rounding; where fp64 does not
    decide (fp64_decides), every sample point and the local maximisation (8 starts) again in double-double."""
    from .reference import precise
    period = 2 * np.pi / nfp
    th = rng.uniform(0, 2 * np.pi, nside)
    ph = rng.uniform(0, period, nside) + phi0
    x, nv = surface_ld(coefs, nfp, phi0, th, ph)
    x64 = x.astype(float)
    o = F.evaluate(x64, grad=True)
    bn = _bn_ld(x, nv, o["B"], o["dB"])
    TH, PH = np.repeat(th, nside), np.tile(ph, nside)
    out = dict(ref=REF_VERSION, bn_max=float(np.max(bn)), bn_median=float(np.median(bn)), npts=int(bn.size))
    a = next(c for c in coefs if c is not None)
    M, N = a.shape[0] - 1, (a.shape[1] - 1) // 2
    h = (2 * np.pi / (4 * M + 4), period / (4 * max(N, 1) + 4))

    def bn64(xx, nn):
        oo = F.evaluate(xx.astype(float), grad=True)
        return _bn_ld(xx, nn, oo["B"], oo["dB"])

    tops = _distinct_tops(bn, TH, PH, nstart, *h)
    s_, w_ = zoom_sup(coefs, nfp, phi0, [(TH[i], PH[i]) for i in tops], h, bn64)
    out["bn_sup"], out["bn_sup_at"] = max(s_, out["bn_max"]), w_
    if precise_pts:
        P = precise.Reference(source)
        idx = np.unique(np.concatenate([np.argsort(bn)[-precise_pts:],
                                        rng.choice(bn.size, min(precise_pts, bn.size), replace=False)]))
        Bp = np.asarray(P.evaluate(x64[idx], grad=False)["B"], float)
        out["precise_vs_fast64"] = float(np.max(np.linalg.norm(Bp - o["B"][idx], axis=-1) / np.linalg.norm(Bp, axis=-1)))
        out["bn_precise_max"] = float(np.max(_bn_ld(x[idx], nv[idx], Bp, o["dB"][idx])))
        if not fp64_decides(out):
            Bd = np.asarray(P.evaluate(x64, grad=False)["B"], float)
            bnd = _bn_ld(x, nv, Bd, o["dB"])
            out["bn_max_dd"] = float(np.max(bnd))
            out["bn_median_dd"] = float(np.median(bnd))
            out["dd_vs_fast64_max"] = float(np.max(np.linalg.norm(Bd - o["B"], axis=-1) / np.linalg.norm(Bd, axis=-1)))

            def bndd(xx, nn):
                x6 = xx.astype(float)
                oo = F.evaluate(x6, grad=True)
                return _bn_ld(xx, nn, np.asarray(P.evaluate(x6, grad=False)["B"], float), oo["dB"])

            tops = _distinct_tops(bnd, TH, PH, max(1, nstart // 2), *h)
            s_, w_ = zoom_sup(coefs, nfp, phi0, [(TH[i], PH[i]) for i in tops], h, bndd, side=7)
            out["bn_sup_dd"], out["bn_sup_dd_at"] = max(s_, out["bn_max_dd"]), w_
    return out


def fp64_decides(v):
    """(reference 2) The fp64 tier's values decide: its sup below REF_BN / 2, its rounding (double-double against fp64
    at the 8 largest and 8 random samples) below 1e-14, and the sup plus twice that rounding below REF_BN."""
    e = v.get("precise_vs_fast64", np.inf)
    return bool(v["bn_sup"] < 0.5 * REF_BN and e < 1e-14 and v["bn_sup"] + 2 * e < REF_BN)


def verify_torus(source, t: TorusState, npts=4000, nsec=6, nlines=8, nperiods=200, seed=0, precise_pts=8,
                 ref_field=None, deadline=None):
    from .reference import fast64
    out = {}
    t0 = time.time()
    F = ref_field or fast64.Field(source)
    rng = np.random.default_rng(seed)
    period = 2 * np.pi / t.nfp
    out.update(normal_component(source, (t.rc, None, None, t.zs), t.nfp, t.phi0, F, rng,
                                nside=int(np.ceil(np.sqrt(npts))), precise_pts=precise_pts))
    # flux on random cross-sections
    nth = max(256, 8 * t.M)
    thg = 2 * np.pi * np.arange(nth) / nth
    fl = []
    for k in range(nsec):
        p0 = rng.uniform(0, period) + t.phi0
        xs, Rs, Zs, _, (Rt, Zt) = torus_points(t, thg, np.full(nth, p0), deriv=True)
        A = F.evaluate(xs, grad=False)["A"]
        cp, sp = np.cos(p0), np.sin(p0)
        AR = A[:, 0] * cp + A[:, 1] * sp
        fl.append(-2 * np.pi * np.mean(AR * Rt + A[:, 2] * Zt))
    fl = np.array(fl)
    out["flux_sections"] = fl.tolist()
    out["flux"] = float(np.mean(fl))
    out["flux_rel_err"] = float(np.max(np.abs(fl - t.flux)) / abs(t.flux)) if t.flux else np.nan
    # minor radius (mean distance from the m = 0 curve, rough) for the trace distances
    # minor radius: sqrt(cross-section area / pi) in the plane phi0
    thc = 2 * np.pi * np.arange(1024) / 1024
    _, Rc, Zc = torus_points(t, thc, np.full(1024, t.phi0))
    area = 0.5 * abs(np.sum(Rc * np.roll(Zc, -1) - np.roll(Rc, -1) * Zc))
    a_minor = float(np.sqrt(area / np.pi))
    out["minor_radius"] = a_minor
    # (the double-double tier is part of normal_component: where fp64 does not decide -- its own rounding the size of
    # the margin, large coil fields cancelling to a small net field: CoilStellaration D4hhuVCf's fp64 B was 2-2.8e-14
    # off double-double at its 8 worst points, where the double-double B.n is 4e-17 - 2.3e-15 -- every sample point
    # and the local maximisation again in double-double, ~5 ms a point)
    # traced lines
    if nlines and (deadline is None or time.time() < deadline):
        from .reference import trace as RT
        th0 = rng.uniform(0, 2 * np.pi, nlines)
        x0, R0, Z0 = torus_points(t, th0, np.full(nlines, t.phi0))
        # sections every 2 pi (full turns); compare with the torus section at phi0 (all field periods alike).
        # With a deadline the lines are traced in chunks of 10 turns while time remains (at least one chunk)
        chunk = nperiods if deadline is None else min(10, nperiods)
        Rs, Zs, errs, lost = [], [], [], np.full(nlines, -1)
        Rc, Zc = R0, Z0
        done = 0
        while done < nperiods:
            n = min(chunk, nperiods - done)
            res = RT.poincare(F, Rc, Zc, [t.phi0], n, phi0=t.phi0)
            Rs.append(res["R"][:, :, 0]); Zs.append(res["Z"][:, :, 0])
            errs.append(np.nanmax(res.get("err", np.array([np.nan]))))
            lost = np.where((lost < 0) & (res["lost"] >= 0), done + res["lost"], lost)
            Rc, Zc = res["R"][:, -1, 0], res["Z"][:, -1, 0]
            done += n
            if deadline is not None and time.time() > deadline or np.any(~np.isfinite(Rc)):
                break
        Rsec, Zsec = np.concatenate(Rs, 1), np.concatenate(Zs, 1)
        dmax = 0.0
        for i in range(nlines):
            ok = np.isfinite(Rsec[i])
            if lost[i] >= 0 or not np.all(ok):
                dmax = np.inf
                continue
            d = section_distance(t, t.phi0, Rsec[i], Zsec[i])
            dmax = max(dmax, float(np.max(d)))
        out["trace_dist_max_rel"] = dmax / a_minor if np.isfinite(dmax) else np.inf
        out["trace_turns"] = int(Rsec.shape[1])
        out["trace_err_est"] = float(np.nanmax(errs))
    out["wall"] = time.time() - t0
    return out


def verify_record(source, surf_rec, nfp, nlines=0, nperiods=200):
    """Independent verification of a surface record (dict with rc, zs, iota, flux, M, N[, phi0]):
    verify_torus plus the acceptance flag ok = acceptance(v): reference B.n at the arithmetic limit, flux to
    1e-12, double-double tier within 1e-14 of fast64, traced lines (if any) within 1e-9 of the minor radius."""
    t0 = time.time()
    t = TorusState(np.array(surf_rec["rc"]), np.array(surf_rec["zs"]), float(surf_rec["iota"]),
                   float(surf_rec["flux"]), int(surf_rec["M"]), int(surf_rec["N"]), int(nfp),
                   phi0=float(surf_rec.get("phi0", 0.0)))
    v = verify_torus(source, t, nlines=nlines, nperiods=nperiods)
    v["ok"] = acceptance(v)
    v["wall"] = time.time() - t0
    return v


# The reference's B.n on a torus at the arithmetic limit: median 2.4e-15, 99 % below 7.3e-15 over 5342 surfaces
# whose solver B.n is at most 1e-14 (torux.analysis.BN_SURFACE).  Where the fp64 tier reads 2e-14 or more it is its
# own rounding near coils (fp64 against double-double up to 2e-13 at single points: QUASR 1784571 reads 1.8e-13 in
# fp64 and 2.5e-15 in double-double), so the bar is 2e-14 in fp64 or, where fp64 does not decide, in double-double
# at every sample point.  (A bar scaled by the fp64 noise measured at a few points would not do: the worst point is
# where the noise is largest, so a noise measured elsewhere does not bound it.)
# The torus's points and normals are in long double (fp64 points are a few ulps off the surface, and next to coils
# that alone moves B.n by up to ~2e-14 -- in either direction -- with the double-double B unchanged), and the bar
# applies to the estimated maximum over the surface (local maximisation from the largest samples), not to the
# samples alone.
REF_BN = 2e-14
REF_BN_NOISE = 1.0


def ref_bn_bound(v):
    """The fp64 tier's bar: REF_BN (the noise-scaled bar max(REF_BN, 10 x noise) of older records is not used)."""
    return REF_BN


def acceptance(v):
    """The acceptance of an independent verification: reference B.n at the arithmetic limit, flux to
    1e-12, traced lines (if any) within 1e-9 of the minor radius.  B.n, reference 2 (v['ref'] = 2): the estimated
    maximum over the surface (long-double surface) below REF_BN / 2 in fp64 with fp64's rounding small against the
    margin (fp64_decides), or else the double-double tier's sample maximum and estimated maximum both below REF_BN.
    Reference 1 (older records): the fp64 sample maximum below 2e-14 with its noise (double-double against fp64 at
    8 points) below 1e-14, or else the double-double tier at every sample point below 2e-14."""
    if v.get("ref", 1) >= 2:
        if fp64_decides(v):
            bn_ok = True
        else:
            bn_ok = (v.get("bn_sup_dd") is not None and v.get("bn_max_dd") is not None
                     and max(v["bn_sup_dd"], v["bn_max_dd"]) < REF_BN)
    else:
        bn_ok = v["bn_max"] < REF_BN and v.get("precise_vs_fast64", 0.0) < 1e-14
        if not bn_ok and v.get("bn_max_dd") is not None:
            bn_ok = v["bn_max_dd"] < REF_BN
    return bool(bn_ok and v["flux_rel_err"] < 1e-12 and v.get("trace_dist_max_rel", 0.0) < 1e-9)
