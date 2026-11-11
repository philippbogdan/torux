"""Independent verification of a general (non-stellarator-symmetric) torus with the reference field
(torux.reference), which shares no code with the solver's engine: the analogue of torux.verify for
torux.torus_general.GeneralTorus.

verify_general(source, torus) returns
  bn_max, bn_median,   |B.n|/(|B||n|) as torux.verify (torux.verify.normal_component, reference 2): long-double points
  bn_sup               and normals on a random tensor sample, the maximum by local maximisation, the double-double
                       tier where fp64 does not decide;
  flux, flux_rel_err   the enclosed flux by loop integrals of the reference A on random cross-sections
                       (trapezoid in theta), against the torus's;
  precise_vs_fast64,   the double-double tier at a few points (B difference and its B.n);
  bn_precise_max       (bn_max_dd, bn_sup_dd where fp64 does not decide)
  trace_dist_max_rel   reference-traced lines started on the torus (Gauss-Legendre IRK): largest
                       distance of their section points from the torus cross-section / minor radius.
"""
from __future__ import annotations

import time

import numpy as np

from .torus_general import GeneralTorus, torus_points


def _section_series(t: GeneralTorus, phi):
    """(A_m, B_m) for R and Z at the angle phi: R(theta) = sum_m A_m cos m theta + B_m sin m theta."""
    n = np.arange(-t.N, t.N + 1)
    psi = t.nfp * (phi - t.phi0)
    cn, sn = np.cos(n * psi), np.sin(n * psi)
    out = []
    for a, b in ((t.rc, t.rs), (t.zc, t.zs)):
        # a cos(m th - n ps) + b sin(m th - n ps) = (a cn - b sn) cos m th + (a sn + b cn) sin m th
        out.append((a @ cn - b @ sn, a @ sn + b @ cn))
    return out


def section_curve(t: GeneralTorus, phi, theta, k=0):
    """k-th theta derivative of (R, Z) of the cross-section at phi."""
    (RA, RB), (ZA, ZB) = _section_series(t, phi)
    m = np.arange(t.M + 1)
    ang = np.outer(np.asarray(theta, float), m)
    c, s = np.cos(ang), np.sin(ang)
    mk = m.astype(float) ** k

    def ev(A, B):
        # d^k/dth^k (A cos + B sin): cycle cos -> -sin -> -cos -> sin
        r = k % 4
        if r == 0:
            return c @ (mk * A) + s @ (mk * B)
        if r == 1:
            return -s @ (mk * A) + c @ (mk * B)
        if r == 2:
            return -c @ (mk * A) - s @ (mk * B)
        return s @ (mk * A) - c @ (mk * B)
    return ev(RA, RB), ev(ZA, ZB)


def section_distance(t: GeneralTorus, phi, Rq, Zq, nsample=2048, newton=6):
    ths = 2 * np.pi * np.arange(nsample) / nsample
    Rc, Zc = section_curve(t, phi, ths)
    Rq = np.asarray(Rq); Zq = np.asarray(Zq)
    j = np.argmin((Rq[:, None] - Rc[None, :]) ** 2 + (Zq[:, None] - Zc[None, :]) ** 2, axis=1)
    th = ths[j]
    for _ in range(newton):
        R0, Z0 = section_curve(t, phi, th, 0)
        R1, Z1 = section_curve(t, phi, th, 1)
        R2, Z2 = section_curve(t, phi, th, 2)
        g = (R0 - Rq) * R1 + (Z0 - Zq) * Z1
        h = R1 ** 2 + Z1 ** 2 + (R0 - Rq) * R2 + (Z0 - Zq) * Z2
        th = th - g / np.where(h > 0, h, R1 ** 2 + Z1 ** 2)
    R0, Z0 = section_curve(t, phi, th, 0)
    return np.hypot(R0 - Rq, Z0 - Zq)


def verify_general(source, t: GeneralTorus, npts=4000, nsec=6, nlines=8, nperiods=200, seed=0, precise_pts=8,
                   ref_field=None, deadline=None):
    from .reference import fast64
    out = {}
    t0 = time.time()
    F = ref_field or fast64.Field(source)
    rng = np.random.default_rng(seed)
    period = 2 * np.pi / t.nfp
    # B.n as torux.verify (reference 2): long-double surface on a random tensor sample, local maximisation, the
    # double-double tier where fp64 does not decide
    from .verify import normal_component
    out.update(normal_component(source, (t.rc, t.rs, t.zc, t.zs), t.nfp, t.phi0, F, rng,
                                nside=int(np.ceil(np.sqrt(npts))), precise_pts=precise_pts))
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
    thc = 2 * np.pi * np.arange(1024) / 1024
    Rc, Zc = section_curve(t, t.phi0, thc)
    area = 0.5 * abs(np.sum(Rc * np.roll(Zc, -1) - np.roll(Rc, -1) * Zc))
    a_minor = float(np.sqrt(area / np.pi))
    out["minor_radius"] = a_minor
    if nlines and (deadline is None or time.time() < deadline):
        from .reference import trace as RT
        th0 = rng.uniform(0, 2 * np.pi, nlines)
        _, R0, Z0 = torus_points(t, th0, np.full(nlines, t.phi0))
        res = RT.poincare(F, R0, Z0, [t.phi0], nperiods, phi0=t.phi0)
        Rsec, Zsec = res["R"][:, :, 0], res["Z"][:, :, 0]
        dmax = 0.0
        for i in range(nlines):
            ok = np.isfinite(Rsec[i])
            if res["lost"][i] >= 0 or not np.all(ok):
                dmax = np.inf
                continue
            d = section_distance(t, t.phi0, Rsec[i], Zsec[i])
            dmax = max(dmax, float(np.max(d)))
        out["trace_dist_max_rel"] = dmax / a_minor if np.isfinite(dmax) else np.inf
        out["trace_turns"] = int(Rsec.shape[1])
        out["trace_err_est"] = float(np.nanmax(res.get("err", np.array([np.nan]))))
    out["wall"] = time.time() - t0
    return out


def verify_record_general(source, rec, nfp, nlines=0, nperiods=200):
    """verify_general on a record of torux.family_general.torus_record(full=True), with the acceptance
    flag of torux.verify.verify_record (torux.verify.acceptance: reference B.n at the arithmetic limit, flux to
    1e-12, double-double within 1e-14, traced lines within 1e-9 of the minor radius)."""
    from .family_general import torus_from_record
    t0 = time.time()
    t = torus_from_record(rec, nfp)
    v = verify_general(source, t, nlines=nlines, nperiods=nperiods)
    from .verify import acceptance
    v["ok"] = acceptance(v)
    v["wall"] = time.time() - t0
    return v
