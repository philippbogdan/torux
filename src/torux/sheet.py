"""The solve for a source with a current sheet (torux.SurfaceCurrent): from the plasma's boundary outward.

A sheet's boundary is a flux surface of the field outside it, and inside it the source's field is not the plasma's
(for the virtual-casing current K = n x B / mu0 it is zero): there is no magnetic axis to start from, and nothing
inside the boundary is evidence of anything.  So the family of tori starts next to the boundary, from the boundary
itself moved out along its sections' normals and carried into the field's straight-field-line angles
(torux.design.sfl_guess_from_surface), and is continued outward in the full toroidal flux as torux.analysis.analyse
continues from the axis, with the same jumps over the chains it stalls at.  The torus at the request is refined to the
arithmetic limit and accepted by the same rules (off-grid B.n at most torux.analysis.BN_SURFACE, the resonance guard,
the coils' quadrature check, simple sections), and by three more that only a sheet needs:

  clearance    every sampled point of the torus outside the boundary and at least CLEARANCE minor radii of the
               boundary from it (the distance down to which the sheet's field is measured to hold rounding level);
  encloses     every section of the torus goes once round the boundary's section in its plane;
  sheet_check  the sheet's field on the torus moves by at most SHEET_CHECK of |B| when its quadrature is refined
               (Gauss order 16 and panels used from 3 radii, against 12 and 2).

A full toroidal flux is the part's flux inside the boundary plus the flux between the boundary and the torus (the
engine's vector potential carries the first, torux.field.engine.FieldEngine.flux_tube).  There are no certificates
that a surface cannot exist: their theorems are about tori with no current inside them, and a plasma's boundary has
current inside.  Where nothing reaches the request the answer is undetermined, and says why.
"""
from __future__ import annotations

import logging
import math
import time
import traceback

import numpy as np

log = logging.getLogger("torux")

CLEARANCE = 1e-5          # minor radii: the nearest a verified surface may come to the boundary
SHEET_CHECK = 2e-15       # of |B|: the sheet's field under a refined quadrature
START = 0.02              # minor radii: the start torus's distance from the boundary (less for a request nearer it)
FAR = 4.0                 # without a request the family aims at FAR x the flux of a disc reaching the nearest coil


def sheet_of(src):
    parts = [p for p in src.parts if p.kind == "surface_current"]
    return parts[0] if len(parts) == 1 else None


def refusal(src, flux=None):
    """One sentence why torux.solve does not take this source with a sheet (and this flux), or None."""
    parts = [p for p in src.parts if p.kind == "surface_current"]
    if len(parts) > 1:
        return "torux.solve takes a source with one SurfaceCurrent, not several"
    part = parts[0]
    if part.flux is None:
        return ("a source with a SurfaceCurrent needs the toroidal flux inside its boundary to label a surface: give "
                "SurfaceCurrent(..., flux=) in Wb")
    if part.nfp % max(int(src.nfp), 1):
        return (f"the source's {src.nfp} field periods are not a period of its SurfaceCurrent's {part.nfp}: give the "
                f"source the field periods its coils and sheet share")
    if flux is not None and abs(float(flux)) <= abs(part.flux):
        return (f"the flux asked for, {abs(float(flux)):.12g} Wb, is not more than the {abs(part.flux):.12g} Wb inside "
                f"the SurfaceCurrent's boundary: a surface of this source lies outside it")
    return None


def symmetric(src):
    """Stellarator symmetry of a source with a sheet: the source's claim, and the sheet's coefficients."""
    part = sheet_of(src)
    return bool(getattr(src, "stellsym", False)) and part is not None and part.symmetric()


class OffsetBoundary:
    """The boundary moved by delta (m) along the normals of its constant-phi sections: a surface whose toroidal angle
    is the cylindrical one, in the interface of torux.io.surfaces (gamma_and_derivs of fractions of a turn)."""

    def __init__(self, S, delta):
        self.S = S
        self.nfp = S.nfp
        self.delta = float(delta)

    def gamma_and_derivs(self, phi, theta):
        from .field.surface_current import grid_values
        phi, theta = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
        shp = phi.shape
        P = 2 * np.pi * phi.ravel()
        T = 2 * np.pi * theta.ravel()
        S = self.S
        # pointwise series (the grid routine on one point at a time, vectorised over a diagonal)
        Ep = np.exp(1j * np.outer(P, S.nv))
        Et = np.exp(1j * np.outer(T, S.mv))
        tab = {}
        for name, q in (("R", 0), ("Rp", 1), ("Z", 2), ("Zp", 3)):
            tab[name] = np.einsum("tm,mn,tn->t", Et, S.CN[q], Ep)
        im = 1j * S.mv
        R = tab["R"].real
        Rt = np.einsum("tm,mn,tn->t", Et * im, S.CN[0], Ep).real
        Rtt = np.einsum("tm,mn,tn->t", Et * im * im, S.CN[0], Ep).real
        Rp = tab["Rp"].real
        Rtp = np.einsum("tm,mn,tn->t", Et * im, S.CN[1], Ep).real
        Z = tab["Z"].real
        Zt = np.einsum("tm,mn,tn->t", Et * im, S.CN[2], Ep).real
        Ztt = np.einsum("tm,mn,tn->t", Et * im * im, S.CN[2], Ep).real
        Zp = tab["Zp"].real
        Ztp = np.einsum("tm,mn,tn->t", Et * im, S.CN[3], Ep).real
        L = np.hypot(Rt, Zt)
        nR, nZ = Zt / L, -Rt / L                       # outward for a counterclockwise section
        # derivatives of the unit normal along theta and phi
        Lt = (Rt * Rtt + Zt * Ztt) / L
        Lp = (Rt * Rtp + Zt * Ztp) / L
        nRt, nZt = (Ztt * L - Zt * Lt) / L ** 2, -(Rtt * L - Rt * Lt) / L ** 2
        nRp, nZp = (Ztp * L - Zt * Lp) / L ** 2, -(Rtp * L - Rt * Lp) / L ** 2
        d = self.delta
        R2, Z2 = R + d * nR, Z + d * nZ
        R2t, Z2t = Rt + d * nRt, Zt + d * nZt
        R2p, Z2p = Rp + d * nRp, Zp + d * nZp
        c, s = np.cos(P), np.sin(P)
        r = np.stack([R2 * c, R2 * s, Z2], -1)
        rt = 2 * np.pi * np.stack([R2t * c, R2t * s, Z2t], -1)
        rp = 2 * np.pi * np.stack([R2p * c - R2 * s, R2p * s + R2 * c, Z2p], -1)
        return r.reshape(shp + (3,)), rp.reshape(shp + (3,)), rt.reshape(shp + (3,))


def _budget_gone():
    from . import budget as _budget
    return _budget.expired()


def boundary_leak(eng, S, part, nth=48, nph=16, d=1e-6):
    """|B.n| / |B| on the boundary from outside, on a grid of one field period: how far the boundary is from a flux
    surface of the field outside it (zero for an exact equilibrium boundary whose sheet and coils match it).  The
    field is taken d, 2d and 4d minor radii out along the boundary's normals and B.n extrapolated to the boundary
    (cubic error: the neighbouring surfaces are not parallel to it, which alone reads about d of |B| one way).
    Returns (max, median)."""
    th = 2 * np.pi * np.arange(nth) / nth
    ph = 2 * np.pi * (np.arange(nph) + 0.5) / (nph * part.nfp)
    TH, PH = (v.ravel() for v in np.meshgrid(th, ph, indexing="ij"))
    m, n = part.modes[:, 0].astype(float), part.modes[:, 1].astype(float) * part.nfp
    ang = np.outer(TH, m) - np.outer(PH, n)
    c, s = np.cos(ang), np.sin(ang)
    z0 = np.zeros(len(m))
    rs = z0 if part.rmns is None else part.rmns
    zc = z0 if part.zmnc is None else part.zmnc
    R, Z = c @ part.rmnc + s @ rs, s @ part.zmns + c @ zc
    Rt, Zt = (-s * m) @ part.rmnc + (c * m) @ rs, (c * m) @ part.zmns + (-s * m) @ zc
    Rp, Zp = (s * n) @ part.rmnc - (c * n) @ rs, (-c * n) @ part.zmns + (s * n) @ zc
    cp, sp = np.cos(PH), np.sin(PH)
    rt = np.stack([Rt * cp, Rt * sp, Zt], -1)
    rp = np.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp], -1)
    nn = np.cross(rp, rt)
    nn /= np.linalg.norm(nn, axis=1)[:, None]
    X = np.stack([R * cp, R * sp, Z], -1)
    h = d * S.size["minor_radius"]
    vals = []
    for k in (1, 2, 4):
        B = np.asarray(eng.evaluate(X + k * h * nn, grad=False, A=False)["B"])
        vals.append((np.sum(B * nn, -1), np.linalg.norm(B, axis=-1)))
    # Richardson: (8 f(h) - 6 f(2h) + f(4h)) / 3 is f(0) to O(h^3)
    bn0 = (8 * vals[0][0] - 6 * vals[1][0] + vals[2][0]) / 3
    b0 = (8 * vals[0][1] - 6 * vals[1][1] + vals[2][1]) / 3
    bn = np.abs(bn0) / b0
    bn = bn[np.isfinite(bn)]
    return (float(np.max(bn)), float(np.median(bn))) if len(bn) else (np.nan, np.nan)


def geometry_resolution(part):
    """The largest m and |n| of the boundary's own series (coefficients above 1e-14 of its scale)."""
    amp = np.abs(part.rmnc) + np.abs(part.zmns)
    if part.rmns is not None:
        amp = amp + np.abs(part.rmns)
    if part.zmnc is not None:
        amp = amp + np.abs(part.zmnc)
    keep = amp > 1e-14 * amp.max()
    return int(np.abs(part.modes[keep, 0]).max()), int(np.abs(part.modes[keep, 1]).max())


def start_torus(cont, eng, S, part, delta, M, N, verbose=False):
    """A torus of the field next to the boundary: the boundary moved out by delta, in the field's straight-field-line
    angles, solved by Newton at its own flux and raised until its off-grid B.n is below the family's bar.  Where
    Newton levels off above its tolerance (the series in straight-field-line angles needs more modes than the
    boundary's own: VMEC's angle is not one), the guess is made again a ladder step finer, up to three times.
    Returns (torus or None, record)."""
    from .design import sfl_guess_from_surface
    from .family import ladder_up
    rec = dict(delta_m=float(delta), M=int(M), N=int(N), tries=[])
    surf = OffsetBoundary(S, delta)
    R00 = float(S.size["R0"])
    sol = None
    for step in range(4):
        g = sfl_guess_from_surface(surf, eng, M, N, Ml=min(M, max(16, M - 4)), Nl=min(N, max(16, N - 4)))
        f0 = cont.problem(M, N).flux_of(g.rc, g.zs)
        if not np.isfinite(f0):
            rec["failed"] = "the moved boundary's flux is not finite (a point inside the sheet)"
            return None, rec
        g.flux = f0
        sol = cont.solve(g, f0, M, N, maxit=25)
        rec["tries"].append(dict(M=int(M), N=int(N), converged=bool(sol.converged), res=float(sol.res),
                                 its=int(sol.its), lambda_fit_res=g.info.get("lambda_fit_res"),
                                 iota=float(sol.iota)))
        if sol.converged or not (np.isfinite(sol.res) and sol.res < 1e-4 * R00) or _budget_gone():
            break
        M, N = ladder_up(M, cap=cont.Mmax), ladder_up(N, cap=cont.Nmax)
    rec.update(flux_guess=float(f0), converged=bool(sol.converged), res=float(sol.res), its=int(sol.its), M=int(M),
               N=int(N))
    rec["guess"] = {k: g.info.get(k) for k in ("lambda_fit_res", "lambda_fit_iota", "boundary_bn_max", "flipped")}
    if not sol.converged:
        rec["failed"] = "Newton from the moved boundary did not converge"
        return None, rec
    sol = cont.raise_until(sol, f0, cont.bn_cont)
    rec.update(bn_off=float(sol.bn_off), iota=float(sol.iota), flux=float(sol.flux), M_raised=int(sol.M),
               N_raised=int(sol.N))
    if not (np.isfinite(sol.bn_off) and sol.bn_off < cont.bn_accept):
        rec["failed"] = "the start torus did not reach the family's bar"
        return None, rec
    return sol, rec


def acceptance(eng, S, T, a, phi_planes=4):
    """The checks a sheet adds to a torus's acceptance (module docstring): dict(clearance_m, clearance_ok, encloses,
    sheet_check, sheet_check_ok, ok)."""
    from .field.surface_current import interior_curve, sheet_field
    out = {}
    nfp = int(T.nfp)
    nt, npp = 4 * T.M + 32, 4 * T.N + 32
    th = 2 * np.pi * np.arange(nt) / nt
    ph = T.phi0 + 2 * np.pi * np.arange(npp) / (npp * nfp)
    R, Z = T.geometry_grid(th, ph)
    PH = np.broadcast_to(ph[None, :], R.shape)
    X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1).reshape(-1, 3)
    d = S.signed_distance(X)
    dmin = float(np.min(d))
    out["clearance_m"] = dmin
    out["clearance_minor_radii"] = dmin / a
    out["clearance_needed_m"] = CLEARANCE * a
    out["clearance_ok"] = bool(np.isfinite(dmin) and dmin >= CLEARANCE * a)
    # each section once round the boundary's interior curve
    curve = interior_curve(S)
    enc = True
    if curve is None:
        enc = False
    else:
        cR, cZ = curve
        for k in range(phi_planes):
            p0 = T.phi0 + 2 * np.pi * k / (phi_planes * nfp)
            e = np.exp(1j * p0 * S.nv)
            Rc, Zc = float((e @ cR).real), float((e @ cZ).real)
            th2 = 2 * np.pi * np.arange(4 * T.M + 64) / (4 * T.M + 64)
            Rs, Zs = T.geometry_grid(th2, np.array([p0]))
            ang = np.arctan2(Zs[:, 0] - Zc, Rs[:, 0] - Rc)
            ang = np.unwrap(np.concatenate([ang, ang[:1]]))
            if abs((ang[-1] - ang[0]) / (2 * np.pi) - 1) > 1e-6:
                enc = False
    out["encloses"] = bool(enc)
    # the sheet's field at a sample of the torus (and its points nearest the boundary) under a refined quadrature
    rng = np.random.default_rng(7)
    pick = rng.choice(len(X), min(256, len(X)), replace=False)
    near = np.argsort(d)[:64]
    P = X[np.unique(np.concatenate([pick, near]))]
    part = S.part
    b1 = S.evaluate(P, grad=False, potential=False, nan_outside=True)[:, :3]
    b2 = sheet_field(part, 16, 3.0).evaluate(P, grad=False, potential=False, nan_outside=True)[:, :3]
    Btot = np.linalg.norm(np.asarray(eng.evaluate(P, grad=False, A=False)["B"]), axis=1)
    chk = float(np.max(np.linalg.norm(b1 - b2, axis=1) / Btot))
    out["sheet_check"] = chk
    out["sheet_check_points"] = int(len(P))
    out["sheet_check_ok"] = bool(np.isfinite(chk) and chk <= SHEET_CHECK)
    out["ok"] = bool(out["clearance_ok"] and out["encloses"] and out["sheet_check_ok"])
    return out


def analyse_sheet(source, flux=None, dmin=None, tol=3e-15, verbose=False, max_jumps=6, Mmax=128, Nmax=128,
                  budget_s=None):
    """torux.analysis.analyse for a source with one SurfaceCurrent (module docstring).  The same record."""
    from .envfail import environment_failure
    from .field.surface_current import solver_mode
    t0 = time.time()
    out = dict(id=source.id, nfp=source.nfp, request_flux=flux, status="error", timings={}, chains=[], jumps=[],
               path="sheet")
    try:
        with solver_mode():
            _analyse(source, flux, dmin, tol, verbose, max_jumps, Mmax, Nmax, budget_s, t0, out)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        out["status"] = "error"
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    out["timings"]["total"] = time.time() - t0
    return out


def _analyse(source, flux, dmin, tol, verbose, max_jumps, Mmax, Nmax, budget_s, t0, out):
    from . import budget as _budget
    from .analysis import (BN_SURFACE, CERT_RESERVE, FULL_CAPS, lean_continuation, near_coil_retry, torus_record,
                           torus_topology, chain_record)
    from .chains import analyse_chain, first_rational_ahead, jump_over, nearest_rational_ahead
    from .family import Continuation, refine_to_limit
    from .field import FieldEngine
    from .guard import resonance_guard
    deadline = t0 + budget_s if budget_s else None
    part = sheet_of(source)
    if dmin is None:
        dmin = source.meta.get("dmin", 0.05)
    eng = FieldEngine(source, dmin=dmin)
    S = eng.sheets[0]
    log.info("sheet: %d x %d panels a period, modes m <= %d, |n| <= %d; measuring the boundary's own flux", S.ntheta,
             S.nphi, S.M, S.N)
    tube = eng.flux_tube()
    log.info("sheet: the boundary's own flux %.6g Wb (sections differ by %.2g, widths by %.2g) in %.0f s",
             tube["boundary"]["flux"], tube["boundary"]["spread"], tube["boundary"]["check"], time.time() - t0)
    a = S.size["minor_radius"]
    psi_in = float(tube["flux_inside"])
    bf = tube["boundary"]
    out["engine"] = dict(nodes=eng.info.n_nodes, dmin=dmin, per_coil=False)
    out["sheet"] = dict(flux_inside_Wb=psi_in, own_flux_Wb=bf["flux"], own_flux_spread_Wb=bf["spread"],
                        own_flux_check_Wb=bf["check"], minor_radius_m=a, clearance_m=CLEARANCE * a,
                        panels=[int(S.ntheta), int(S.nphi)], modes=[int(S.M), int(S.N)])
    leak = boundary_leak(eng, S, part)
    out["sheet"]["boundary_bn_max"], out["sheet"]["boundary_bn_median"] = leak
    out["timings"]["boundary"] = time.time() - t0
    nfp = int(source.nfp)
    Mg, Ng = geometry_resolution(part)
    # the toroidal resolution counts the source's field periods: a sheet of nfp k * source nfp has modes k n there
    k = part.nfp // max(nfp, 1)
    M0 = max(8, min(Mmax, Mg + 2))
    N0 = max(8, min(Nmax, k * Ng + 2))
    cont = Continuation(eng, nfp, M0=M0, N0=N0, verbose=verbose, Mmax=max(Mmax, M0), Nmax=max(Nmax, N0))
    cont.fmeas_off = True
    if flux is not None:
        target = abs(float(flux))
    else:
        pts = source.sample_points()
        x0 = np.array([float(np.real(S.CN[0, 0].sum())), 0.0, float(np.real(S.CN[2, 0].sum()))])
        d_coil = float(np.min(np.linalg.norm(pts - x0[None, :], axis=1))) if len(pts) else 0.5 * x0[0]
        Bx = float(np.linalg.norm(np.asarray(eng.evaluate(
            np.array([[x0[0] + 1.02 * a, 0.0, x0[2]]]), grad=False, A=False)["B"])[0]))
        target = abs(psi_in) + FAR * Bx * np.pi * d_coil ** 2
        out["range_target"] = target
    # the start: nearer the boundary than the request, at most START minor radii from it
    Bs = np.asarray(eng.evaluate(np.array([[float(np.real(S.CN[0, 0].sum())) + 1.05 * a, 0.0,
                                            float(np.real(S.CN[2, 0].sum()))]]), grad=False, A=False)["B"])[0]
    est = (target - abs(psi_in)) / max(abs(float(Bs[1])) * S.size["Lt"], 1e-300)    # rough distance of the request
    tries = []
    T0 = None
    if flux is not None:
        # nearer the boundary than the request: 0.4 and 0.2 of its distance (at most START and START / 2 minor
        # radii), then, where those found no torus (the boundary leaks, or islands stand next to it), 0.6, 0.8, 0.9
        deltas = [min(START * a, 0.4 * est), min(0.5 * START * a, 0.2 * est), 0.6 * est, 0.8 * est, 0.9 * est]
        deltas = deltas[:2] + [d for d in deltas[2:] if d > deltas[0]]
    else:
        deltas = [f * a for f in (START, 0.5 * START, 2.5 * START, 5 * START, 10 * START)]
    for delta in deltas:
        if flux is not None and delta < CLEARANCE * a:
            out["status"] = "undetermined"
            out["why"] = (f"the surface asked for lies about {est:.3g} m from the SurfaceCurrent's boundary, nearer than "
                          f"the {CLEARANCE:g} minor radii ({CLEARANCE * a:.3g} m) down to which the sheet's field is held "
                          f"to rounding level")
            return
        if _budget.expired():
            break
        T0, rec = start_torus(cont, eng, S, part, delta, M0, N0, verbose=verbose)
        tries.append(rec)
        log.info("sheet: start torus %.3g m out: %s (%.0f s)", delta, "converged at %.9g Wb" % T0.flux
                 if T0 is not None else rec.get("failed"), time.time() - t0)
        if T0 is not None and (flux is None or abs(T0.flux) < target):
            break
        T0 = None
    out["start"] = tries
    out["timings"]["start"] = time.time() - t0
    if T0 is None:
        out["status"] = "undetermined"
        reach = [r["delta_m"] / a for r in tries]
        out["why"] = (f"no torus next to the SurfaceCurrent's boundary converged (started {min(reach):.3g} to "
                      f"{max(reach):.3g} of its minor radius out), so no family could start from it" if reach else
                      "no torus next to the SurfaceCurrent's boundary was tried within the budget")
        if np.isfinite(leak[0]) and leak[0] > 1e-6:
            out["why"] += (f": the field crosses the boundary by up to {leak[0]:.2g} of |B| just outside it (median "
                           f"{leak[1]:.2g}), so the boundary is not a flux surface of the coils and the sheet at that "
                           f"level")
        return
    fam_deadline = deadline
    if flux is not None and deadline is not None:
        fam_deadline = deadline - min(CERT_RESERVE, 0.2 * (deadline - t0))
    elif deadline is not None:
        fam_deadline = t0 + 0.6 * (deadline - t0)
    fam = cont.run(None, target, deadline=fam_deadline, start=T0)
    tori = list(fam.tori)
    njump = 0
    while fam.stop_reason == "stalled" and njump < max_jumps and (fam_deadline is None or time.time() < fam_deadline):
        last = tori[-1]
        diota = tori[-1].iota - tori[-2].iota if len(tori) >= 2 else 0.0
        pr = nearest_rational_ahead(last.iota, diota, nfp, qmax=40)
        if pr is None:
            break
        p, q = pr
        from .crossed import QMAX as _QX
        first = first_rational_ahead(last.iota, pr, nfp, qmax=_QX)
        if first is not None:
            d1 = None if fam_deadline is None else time.time() + 0.25 * max(0.0, fam_deadline - time.time())
            ch1 = keep_outside(S, analyse_chain(eng, tori, first[0], first[1], verbose=verbose, deadline=d1,
                                                axis=None), a)
            out["chains"].append(dict(chain_record(ch1), stalled_at=True))
        ch = keep_outside(S, analyse_chain(eng, tori, p, q, verbose=verbose, deadline=fam_deadline, axis=None), a)
        out["chains"].append(dict(chain_record(ch), stalled_at=first is None))
        direction = math.copysign(1.0, diota) if diota else 1.0
        with _budget.scope(fam_deadline):
            new = jump_over(cont, tori, ch, direction=direction, verbose=verbose, axis=None)
        if new is not None and not _outside(S, new, a):
            new = None
        out["jumps"].append(dict(p=p, q=q, ok=new is not None, flux_after=None if new is None else new.flux,
                                 **({} if first is None else dict(stalled_at=[int(first[0]), int(first[1])]))))
        if new is None:
            break
        njump += 1
        if abs(new.flux) >= target:
            out["jumped_past"] = True
            tori.append(new)
            break
        fam = cont.run(None, target, start=new, deadline=fam_deadline)
        tori.extend(fam.tori)
    out["profile"] = [torus_record(t) for t in tori]
    out["family_stop"] = fam.stop_reason
    out["timings"]["family"] = time.time() - t0
    log.info("sheet: the family %s at %.9g Wb after %d tori (%.0f s)", fam.stop_reason, abs(tori[-1].flux), len(tori),
             time.time() - t0)
    fl = np.array([t.flux for t in tori])
    io = np.array([t.iota for t in tori])

    def shear_at(f):
        j = int(np.argmin(np.abs(fl - f)))
        lo, hi = max(0, j - 1), min(len(tori) - 1, j + 1)
        return (io[hi] - io[lo]) / (fl[hi] - fl[lo]) if hi > lo else np.nan

    def accept(sol, dl):
        """Refine sol and apply every check: (best, record, ok, why-not)."""
        rc_ = lean_continuation(cont, source, sol, dmin)
        best = refine_to_limit(rc_, sol, sol.flux, target=tol, deadline=dl, verbose=verbose, Mcap=FULL_CAPS[0],
                               Ncap=FULL_CAPS[1])
        q_ = rc_.problem(best.M, best.N).quadrature_check(best.rc, best.zs)
        nc_info = None
        if not (q_ <= 1e-8):
            nc = near_coil_retry(cont, source, best, dmin, tol, deadline=dl, caps=FULL_CAPS)
            nc_info = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(q_))
            if nc is not None:
                rc_, best, q_ = nc[0], nc[1], nc[2]
        g = resonance_guard(rc_.engine, best, shear=shear_at(best.flux) * 2 * np.pi)
        rec = torus_record(best, full=True)
        rec["quad_err_half"] = float(q_)
        if nc_info is not None:
            rec["near_coil"] = nc_info
        rec["topology"] = torus_topology(best, None)
        log.info("sheet: refined to M, N = %d, %d, off-grid B.n %.2g (%.0f s); checking it against the sheet",
                 best.M, best.N, best.bn_off, time.time() - t0)
        rec["sheet"] = acceptance(rc_.engine, S, best, a)
        guard = dict(ok=g["ok"], h=g["h"], x_over_w=g["x_over_w"], worst=g["worst"], n_checked=len(g["checked"]),
                     m_max=g.get("m_max"), d_max=g.get("d_max"))
        why = []
        if not best.bn_off <= BN_SURFACE:
            why.append(f"the field across it is {best.bn_off:.2g} of |B| after refinement, above {BN_SURFACE:g}")
        if not g["ok"]:
            why.append("the resonance guard refused it")
        if not q_ <= 1e-8:
            why.append("the coils' quadrature check refused it")
        if not rec["topology"].get("ok", rec["topology"].get("simple", False)):
            why.append("a section is not a simple closed curve")
        sh = rec["sheet"]
        if not sh["clearance_ok"]:
            why.append(f"it comes within {sh['clearance_m']:.3g} m of the SurfaceCurrent's boundary, nearer than the "
                       f"{CLEARANCE:g} minor radii down to which the sheet's field is held to rounding level"
                       if sh["clearance_m"] > 0 else "it reaches the SurfaceCurrent's boundary")
        if not sh["encloses"]:
            why.append("a section does not go once round the SurfaceCurrent's boundary")
        if not sh["sheet_check_ok"]:
            why.append(f"the sheet's field on it moves by {sh['sheet_check']:.2g} of |B| under a refined quadrature")
        return best, rec, guard, rc_, not why, why

    if flux is not None:
        reached = [t for t in tori if abs(abs(t.flux) - target) <= 1e-12 * target]
        if reached:
            best, rec, guard, rc_, ok, why = accept(reached[-1], deadline)
            out["surface"] = rec
            out["guard"] = guard
            out["refine_hist"] = best.info.get("refine_hist")
            out["refine_stop"] = best.info.get("refine_stop")
            out["engine"]["refine_nodes"] = rc_.engine.info.n_nodes
            sh = rec["sheet"]
            if ok:
                out["status"] = "surface"
            elif not sh["clearance_ok"] and sh["clearance_m"] > 0 and sh["encloses"]:
                out["status"] = "undetermined"
                out["why"] = "; ".join(why)
            else:
                out["status"] = "surface_unverified"
                out["why"] = "; ".join(why)
        else:
            out["status"] = "undetermined"
            stop = {"stalled": "stalled", "deadline": "ran out of time", "reached": "reached it"}.get(
                fam.stop_reason, fam.stop_reason)
            out["why"] = (f"the family from the SurfaceCurrent's boundary {stop} at {abs(tori[-1].flux):.9g} Wb (iota "
                          f"{tori[-1].iota:.6g}), short of the request; a source with a sheet has no certificates")
        out["timings"]["refine"] = time.time() - t0
        return
    # the range: the outermost tori the family reached, refined and checked, outermost first (up to three)
    out["status"] = "family"
    cands = sorted([t for t in tori if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-6],
                   key=lambda t: -abs(t.flux))
    tried = []
    best_ok = None
    for outer in cands[:3]:
        if tried and deadline is not None and time.time() > deadline:
            break
        with _budget.scope(deadline):
            best, rec, guard, rc_, ok, why = accept(outer, deadline)
        tried.append(dict(flux=abs(float(outer.flux)), bn_off=float(best.bn_off), M=int(best.M), N=int(best.N),
                          guard=bool(guard["ok"]), ok=bool(ok), why=why))
        if ok:
            best_ok = (best, rec, guard)
            break
        if best_ok is None:
            best_ok = (best, rec, guard)
    if cands and best_ok is not None:
        best, rec, guard = best_ok
        ok = tried[-1]["ok"] if tried else False
        ok = any(t["ok"] for t in tried)
        out["range"] = dict(verified_lower_bound=abs(best.flux) if ok else None, outer_torus=rec,
                            guard=dict(ok=guard["ok"], h=guard["h"], worst=guard["worst"]), verified=bool(ok),
                            tried=tried)
        out["surface"] = rec


def keep_outside(S, ch, a):
    """A chain's closed lines with every sampled point outside the boundary by the clearance; the others are dropped
    (counted in ch.info['lines_inside_boundary']): nothing that passes through the boundary is evidence."""
    kept, dropped = [], 0
    for ln in ch.lines:
        ph = ln.phi0 + 2 * np.pi * ln.q / ln.nfp * np.arange(64 * ln.q) / (64 * ln.q)
        R, Z = ln.at(ph)
        X = np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)
        d = S.signed_distance(X)
        if np.all(np.isfinite(d)) and np.min(d) >= CLEARANCE * a:
            kept.append(ln)
        else:
            dropped += 1
    if dropped:
        ch.info["lines_inside_boundary"] = dropped
        ch.lines = kept
        ids = {id(ln) for ln in kept}
        ch.O = [ln for ln in ch.O if id(ln) in ids]
        ch.X = [ln for ln in ch.X if id(ln) in ids]
    return ch


def _outside(S, T, a):
    """A torus beyond a chain's jump: every sampled point outside the boundary by the clearance."""
    nt, npp = 2 * T.M + 16, 2 * T.N + 16
    th = 2 * np.pi * np.arange(nt) / nt
    ph = T.phi0 + 2 * np.pi * np.arange(npp) / (npp * T.nfp)
    R, Z = T.geometry_grid(th, ph)
    PH = np.broadcast_to(ph[None, :], R.shape)
    X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1).reshape(-1, 3)
    d = S.signed_distance(X)
    return bool(np.all(np.isfinite(d)) and np.min(d) >= CLEARANCE * a)
