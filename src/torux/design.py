"""Design surfaces published with a source (QUASR optimisation surfaces, VMEC/FOCUS boundaries):
planar cross-sections and the toroidal flux they enclose in the coil field.

A design surface is any object with ``gamma_and_derivs(phi_frac, theta_frac)`` (torux.io.surfaces):
its own angles need not have the cylindrical angle as toroidal angle (SurfaceXYZTensorFourier
rotates (xhat, yhat) by phi), so the planar section at cylindrical angle phi_c is found per theta
by Newton on atan2(y, x) = phi_c.  The flux through the section is the loop integral of A
(counter-clockwise in (R, Z): Phi = int B_phi dR dZ, the solver's convention).
"""
from __future__ import annotations

import numpy as np


def planar_section(surf, phi_c, ntheta=512, newton=12):
    """(R, Z, dR/dtheta, dZ/dtheta) of the section at cylindrical angle phi_c (rad), theta the surface's
    own poloidal angle sampled uniformly (ntheta points), and the surface-phi fractions used."""
    th = np.arange(ntheta) / ntheta
    ps = np.full(ntheta, phi_c / (2 * np.pi))
    for _ in range(newton):
        r, rp, rt = surf.gamma_and_derivs(ps, th)
        ang = np.arctan2(r[:, 1], r[:, 0])
        d = (ang - phi_c + np.pi) % (2 * np.pi) - np.pi
        # d(angle)/d(phi_frac) = (x dy - y dx) / R^2
        R2 = r[:, 0] ** 2 + r[:, 1] ** 2
        dang = (r[:, 0] * rp[:, 1] - r[:, 1] * rp[:, 0]) / R2
        ps = ps - d / dang
    r, rp, rt = surf.gamma_and_derivs(ps, th)
    ang = np.arctan2(r[:, 1], r[:, 0])
    err = float(np.max(np.abs((ang - phi_c + np.pi) % (2 * np.pi) - np.pi)))
    R = np.hypot(r[:, 0], r[:, 1]); Z = r[:, 2]
    # tangent along the planar section: d/dtheta at fixed cylindrical angle
    R2 = r[:, 0] ** 2 + r[:, 1] ** 2
    dang_p = (r[:, 0] * rp[:, 1] - r[:, 1] * rp[:, 0]) / R2
    dang_t = (r[:, 0] * rt[:, 1] - r[:, 1] * rt[:, 0]) / R2
    dps_dt = -dang_t / dang_p
    dr = rt + rp * dps_dt[:, None]                    # derivative w.r.t. theta fraction
    c, s = np.cos(phi_c), np.sin(phi_c)
    dR = dr[:, 0] * c + dr[:, 1] * s
    dZ = dr[:, 2]
    return R, Z, dR, dZ, err


def section_flux(A_eval, phi_c, R, Z, dR, dZ):
    """Loop integral of A along the section, oriented counter-clockwise in (R, Z):
    returns Phi = int B_phi dR dZ (trapezoid rule over the uniform theta samples)."""
    c, s = np.cos(phi_c), np.sin(phi_c)
    x = np.stack([R * c, R * s, Z], -1)
    A = np.asarray(A_eval(x))
    AR = A[:, 0] * c + A[:, 1] * s
    AZ = A[:, 2]
    loop = np.mean(AR * dR + AZ * dZ)                  # per unit theta fraction
    area = np.mean(R * dZ - Z * dR) * 0.5               # signed area in (R, Z) (counter-clockwise > 0)
    sign = 1.0 if area > 0 else -1.0
    # Stokes with the e_phi normal: int B_phi dR dZ = - (counter-clockwise loop integral of A)
    return float(-sign * loop), float(abs(area))


def design_flux(surf, A_eval, nsec=4, ntheta=512):
    """Flux enclosed by a design surface: mean and spread over nsec planar sections in one period."""
    nfp = getattr(surf, "nfp", 1)
    vals, errs, areas = [], [], []
    for k in range(nsec):
        phi_c = 2 * np.pi * k / (nsec * nfp)
        R, Z, dR, dZ, err = planar_section(surf, phi_c, ntheta)
        f, a = section_flux(A_eval, phi_c, R, Z, dR, dZ)
        vals.append(f); errs.append(err); areas.append(a)
    vals = np.array(vals)
    return dict(flux=float(np.mean(vals)), spread=float(np.max(vals) - np.min(vals)), sections=vals.tolist(),
                angle_err=float(max(errs)), areas=areas)


def _trig_powers(a, K):
    """cos(k a), sin(k a) for k = 0..K, (P, K + 1) each, by powers of exp(i a) (angle addition: K complex
    products instead of K cos / sin evaluations; equal to rounding, about k ulps)."""
    e = np.exp(1j * a)
    z = np.empty((a.shape[0], K + 1), complex)
    z[:, 0] = 1.0
    for k in range(1, K + 1):
        z[:, k] = z[:, k - 1] * e
    return z.real, z.imag


def _fast_gamma_and_derivs(sf, phi, theta):
    """SurfaceXYZTensorFourier.gamma_and_derivs with the basis by _trig_powers (no clamped dimensions): for the
    iterations of the angle map (fast_design 'angle'), whose converged angles are then evaluated exactly."""
    TWO = 2 * np.pi
    P = TWO * np.asarray(phi, float).ravel()
    T = TWO * np.asarray(theta, float).ravel()
    mp, nt, nfp = sf.mpol, sf.ntor, sf.nfp
    cp, sp = _trig_powers(nfp * P, nt)
    ct, st = _trig_powers(T, mp)
    kn = np.arange(nt + 1) * nfp
    km = np.arange(mp + 1)
    # columns as SurfaceXYZTensorFourier._basis: n = 0..nt cos(n nfp phi), then sin(1..nt); likewise in theta
    v = np.concatenate([cp, sp[:, 1:]], 1)
    dv = np.concatenate([-kn * sp, (kn * cp)[:, 1:]], 1)
    w = np.concatenate([ct, st[:, 1:]], 1)
    dw = np.concatenate([-km * st, (km * ct)[:, 1:]], 1)
    xh, xhp, xht = sf._field(sf.x, 0, v, dv, w, dw, P, T)
    yh, yhp, yht = sf._field(sf.y, 1, v, dv, w, dw, P, T)
    z, zp, zt = sf._field(sf.z, 2, v, dv, w, dw, P, T)
    c, s = np.cos(P), np.sin(P)
    X = xh * c - yh * s
    Y = xh * s + yh * c
    Xp = xhp * c - yhp * s - xh * s - yh * c
    Yp = xhp * s + yhp * c + xh * c - yh * s
    Xt = xht * c - yht * s
    Yt = xht * s + yht * c
    return (np.stack([X, Y, z], -1), TWO * np.stack([Xp, Yp, zp], -1), TWO * np.stack([Xt, Yt, zt], -1))


def sfl_guess_from_boozer(surf, iota, M, N, newton=60):
    """A torus guess in the solver's angles from a design surface parametrised in Boozer-like
    straight-field-line angles (QUASR's surfaces): the solver's theta at a surface point is
    theta_B - iota (phi_B - phi_c), phi_c its cylindrical angle (constant along field lines up to
    the theta origin, which stellarator symmetry fixes at the symmetric point).  For each grid
    point (theta, phi_c) the surface angles are found by 2D Newton; returns (R, Z) grid values and
    the TorusState of their projection at resolution (M, N)."""
    from .torus import Basis, TorusState, coefficients_from_grid
    nfp = surf.nfp
    # orientation: the solver's theta runs counter-clockwise in (R, Z); flip the surface's if not
    th = np.arange(256) / 256
    r0, _, rt0 = surf.gamma_and_derivs(np.zeros(256), th)
    R0s = np.hypot(r0[:, 0], r0[:, 1])
    area = np.mean(R0s * rt0[:, 2] - r0[:, 2] * (r0[:, 0] * rt0[:, 0] + r0[:, 1] * rt0[:, 1]) / R0s)
    flip = area < 0
    if flip:
        base = surf

        class _Flipped:
            nfp = base.nfp

            def gamma_and_derivs(self, p, t):
                r, rp, rt = base.gamma_and_derivs(p, -np.asarray(t))
                return r, rp, -rt
        surf = _Flipped()
    from . import fast_design as _so
    quad_stop = _so.on("angle")
    s_prev = None
    geval = surf.gamma_and_derivs
    if quad_stop:
        from .io.surfaces import SurfaceXYZTensorFourier
        base_s = base if flip else surf
        if isinstance(base_s, SurfaceXYZTensorFourier) and not any(base_s.clamped_dims):
            if flip:
                def geval(p, t, _b=base_s):
                    r, rp, rt = _fast_gamma_and_derivs(_b, p, -np.asarray(t))
                    return r, rp, -rt
            else:
                def geval(p, t, _b=base_s):
                    return _fast_gamma_and_derivs(_b, p, t)
    b = Basis(nfp, M, N)
    TH, PH = np.meshgrid(b.theta, b.phi, indexing="ij")
    th_t, ph_t = TH.ravel(), PH.ravel()
    tB = th_t.copy(); pB = ph_t.copy()
    for _ in range(newton):
        r, rp, rt = geval(pB / (2 * np.pi), tB / (2 * np.pi))
        R2 = r[:, 0] ** 2 + r[:, 1] ** 2
        pc = np.arctan2(r[:, 1], r[:, 0])
        # unwrap pc near pB
        pc = pB + ((pc - pB + np.pi) % (2 * np.pi) - np.pi)
        dpc_dp = (r[:, 0] * rp[:, 1] - r[:, 1] * rp[:, 0]) / R2 / (2 * np.pi)
        dpc_dt = (r[:, 0] * rt[:, 1] - r[:, 1] * rt[:, 0]) / R2 / (2 * np.pi)
        f1 = tB - iota * (pB - pc) - th_t
        f2 = pc - ph_t
        f2 = (f2 + np.pi) % (2 * np.pi) - np.pi
        # Jacobian wrt (tB, pB)
        a11 = 1 + iota * dpc_dt; a12 = -iota * (1 - dpc_dp)
        a21 = dpc_dt; a22 = dpc_dp
        det = a11 * a22 - a12 * a21
        dt = (f1 * a22 - a12 * f2) / det
        dp = (a11 * f2 - a21 * f1) / det
        # steps capped at 0.25 rad: on strongly shaped surfaces the full Newton step jumps to another branch
        # of the angle map and ends there (QUASR 0136030, 2039780, 2093639: angle errors 3.1, 25, 1.3 rad, a
        # design torus that is not the surface, and a design Newton that diverged at every resolution); capped,
        # the map converges to 4e-15 in 7-10 iterations and is unchanged where it converged before
        sc = np.minimum(1.0, 0.25 / np.maximum(np.maximum(np.abs(dt), np.abs(dp)), 1e-300))
        tB -= sc * dt; pB -= sc * dp
        s = max(float(np.max(np.abs(dt))), float(np.max(np.abs(dp))))
        if s < 1e-15:
            break            # quadratic convergence: the next step would be below rounding
        if quad_stop and float(np.min(sc)) == 1.0:
            # (fast_design 'angle') the 1e-15 test lies under the rounding floor of angles near 2 pi (steps
            # level off at 1e-15 - 8e-15, so all `newton` iterations ran).  After a full step s the error is
            # K s^2 with K the observed contraction s / s_prev^2 (0.5 - 3 on QUASR surfaces): stop when that
            # is below 1e-17 rad, or when the steps stop shrinking at the floor
            if s_prev is not None and s < 1e-8 and (s / s_prev ** 2) * s ** 2 < 1e-17:
                break
            if s_prev is not None and s < 1e-13 and s > 0.5 * s_prev:
                break
        s_prev = s
    r, _, _ = surf.gamma_and_derivs(pB / (2 * np.pi), tB / (2 * np.pi))
    R = np.hypot(r[:, 0], r[:, 1]).reshape(TH.shape)
    Z = r[:, 2].reshape(TH.shape)
    rc, zs = coefficients_from_grid(b, R, Z)
    st = TorusState(np.asarray(rc), np.asarray(zs), float(iota), 0.0, M, N, nfp)
    if quad_stop:
        # the residual at the returned angles (from the final surface evaluation: no extra cost)
        pc = np.arctan2(r[:, 1], r[:, 0])
        pc = pB + ((pc - pB + np.pi) % (2 * np.pi) - np.pi)
        f1 = tB - iota * (pB - pc) - th_t
        f2 = (pc - ph_t + np.pi) % (2 * np.pi) - np.pi
    err = float(np.max(np.abs(np.concatenate([f1, f2]))))
    st.info["angle_newton_err"] = err
    st.info["flipped"] = bool(flip)
    return st


def quasr_design_flux(surf, A_eval, nq=None):
    """QUASR's definition of the flux enclosed by an optimisation surface: the loop
    integral of A along the surface's own theta curve at phi_s = 0, by the trapezoid rule on its
    quadrature points theta_j = j / nq (nq = 2 mpol + 1 for the QUASR surfaces), with the sign of
    the solver's convention (int B_phi dR dZ for a counter-clockwise section)."""
    nq = nq or (2 * surf.mpol + 1)
    th = np.arange(nq) / nq
    r, rp, rt = surf.gamma_and_derivs(np.zeros(nq), th)
    A = np.asarray(A_eval(r))
    loop = float(np.mean(np.sum(A * rt, -1)))
    # orientation of the section in (R, Z)
    thf = np.arange(256) / 256
    r2, _, rt2 = surf.gamma_and_derivs(np.zeros(256), thf)
    R2 = np.hypot(r2[:, 0], r2[:, 1])
    dR = (r2[:, 0] * rt2[:, 0] + r2[:, 1] * rt2[:, 1]) / R2
    area = np.mean(R2 * rt2[:, 2] - r2[:, 2] * dR)
    return -loop if area > 0 else loop


def sfl_guess_from_surface(surf, engine, M, N, Ml=None, Nl=None, iota0=None):
    """A torus guess in the solver's angles from a stellarator-symmetric design boundary R, Z (theta,
    phi) whose poloidal angle is not a field-line angle (VMEC/ConStellaration boundaries): the field-line
    angle theta + lambda(theta, phi) of the coil field projected on the boundary, fitted by linear least
    squares (lambda a sine series, iota one unknown):

        f (1 + lambda_theta) + lambda_phi = iota,   f = B^theta / B^phi (contravariant, tangential part),

    then each solver grid point (theta_s, phi) mapped back by a 1D Newton on theta + lambda = theta_s and
    projected at resolution (M, N).  The boundary need not be a flux surface; the torus Newton takes it
    from there.  Returns the TorusState (info: fit residual, iota, orientation)."""
    from .torus import Basis, TorusState, coefficients_from_grid
    nfp = int(surf.nfp)
    Ml = Ml or min(M, 16)
    Nl = Nl or min(N, 16)
    # orientation: the solver's theta runs counter-clockwise in (R, Z); flip the boundary's if not
    th = np.arange(256) / 256
    r0, _, rt0 = surf.gamma_and_derivs(np.zeros(256), th)
    R0s = np.hypot(r0[:, 0], r0[:, 1])
    area = np.mean(R0s * rt0[:, 2] - r0[:, 2] * (r0[:, 0] * rt0[:, 0] + r0[:, 1] * rt0[:, 1]) / R0s)
    sgn_t = -1.0 if area < 0 else 1.0
    # collocation grid over one field period (theta, phi in radians; the surface takes fractions)
    nt, npp = max(4 * Ml + 8, 48), max(4 * Nl + 8, 32)
    T = 2 * np.pi * np.arange(nt) / nt
    P = 2 * np.pi * np.arange(npp) / (npp * nfp)
    TT, PP = np.meshgrid(T, P, indexing="ij")
    t, p = TT.ravel(), PP.ravel()
    r, rp, rt = surf.gamma_and_derivs(p / (2 * np.pi), sgn_t * t / (2 * np.pi))
    rp = rp / (2 * np.pi)
    rt = sgn_t * rt / (2 * np.pi)
    B = np.asarray(engine.evaluate(r, grad=False, A=False)["B"])
    # tangential contravariant components: B ~ a r_theta + b r_phi (normal part dropped)
    g11 = np.sum(rt * rt, -1); g12 = np.sum(rt * rp, -1); g22 = np.sum(rp * rp, -1)
    h1 = np.sum(B * rt, -1); h2 = np.sum(B * rp, -1)
    det = g11 * g22 - g12 ** 2
    a = (g22 * h1 - g12 * h2) / det
    b = (g11 * h2 - g12 * h1) / det
    n_hat = np.cross(rt, rp)
    bn = np.abs(np.sum(B * n_hat, -1)) / (np.linalg.norm(n_hat, axis=-1) * np.linalg.norm(B, axis=-1))
    f = a / b
    # lambda = sum l_mn sin(m theta - n nfp phi): m = 0, n >= 1; m >= 1, |n| <= Nl
    modes = [(0, n) for n in range(1, Nl + 1)] + [(m, n) for m in range(1, Ml + 1) for n in range(-Nl, Nl + 1)]
    mm = np.array([m for m, _ in modes], float)
    nn = np.array([n for _, n in modes], float) * nfp
    ang = np.outer(t, mm) - np.outer(p, nn)
    C = np.cos(ang)
    A_ = np.concatenate([C * (f[:, None] * mm[None, :] - nn[None, :]), -np.ones((len(t), 1))], 1)
    # rows weighted by b (the equation times B^phi: f is singular nowhere B^phi > 0, but rows scale alike)
    w = b / np.mean(np.abs(b))
    sol, *_ = np.linalg.lstsq(A_ * w[:, None], -f * w, rcond=None)
    lam, iota = sol[:-1], float(sol[-1])
    fit_res = float(np.max(np.abs(A_ @ sol + f)))
    # solver grid: theta_s = theta + lambda(theta, phi) -> theta by Newton (phi the same)
    bs = Basis(nfp, M, N)
    TS, PS = np.meshgrid(bs.theta, bs.phi, indexing="ij")
    ts, ps = TS.ravel(), PS.ravel()
    tv = ts.copy()
    for _ in range(30):
        angv = np.outer(tv, mm) - np.outer(ps, nn)
        g = tv + np.sin(angv) @ lam - ts
        dg = 1 + np.cos(angv) @ (lam * mm)
        step = g / dg
        tv = tv - step
        if float(np.max(np.abs(step))) < 1e-14:
            break
    rr, _, _ = surf.gamma_and_derivs(ps / (2 * np.pi), sgn_t * tv / (2 * np.pi))
    R = np.hypot(rr[:, 0], rr[:, 1]).reshape(TS.shape)
    Z = rr[:, 2].reshape(TS.shape)
    rc, zs = coefficients_from_grid(bs, R, Z)
    st = TorusState(np.asarray(rc), np.asarray(zs), iota if iota0 is None else float(iota0), 0.0, M, N, nfp)
    st.info.update(lambda_fit_res=fit_res, lambda_fit_iota=iota, flipped=bool(sgn_t < 0),
                   boundary_bn_max=float(np.max(bn)), boundary_bn_rms=float(np.sqrt(np.mean(bn ** 2))),
                   angle_newton_err=float(np.max(np.abs(g))), lambda_max=float(np.max(np.abs(lam))))
    return st
