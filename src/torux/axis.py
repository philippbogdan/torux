"""Magnetic axes: closed field lines that close after one field period.

Spectral formulation: the axis is a curve (R, Z)(phi) with period 2 pi / nfp in phi,
R = sum_n rc_n cos(n psi), Z = sum_n zs_n sin(n psi) (psi = nfp phi) for
stellarator-symmetric fields (general series otherwise), solving

    dR/dphi = R B_R / B_phi,   dZ/dphi = R B_Z / B_phi

by Galerkin projection and Newton with the exact Jacobian (one field + gradient
evaluation on the phi grid per iteration).  Initial guesses come from field lines launched
on the symmetry line Z = 0, phi = 0 that return to Z = 0 at phi = pi / nfp (every
stellarator-symmetric closed line of period one crosses both symmetry planes there).

The monodromy (linearised one-period map) gives Greene's residue r = (2 - tr M)/4
(elliptic for 0 < r < 1) and the rotational transform on axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .torus import fft_size, _xp
from .torus_cpu import small_solve
from . import trace as T

jax.config.update("jax_enable_x64", True)


@dataclass
class Axis:
    rc: np.ndarray        # (N+1,)
    zs: np.ndarray        # (N+1,) (zs[0] unused = 0)
    nfp: int
    res: float = np.nan
    converged: bool = False
    monodromy: np.ndarray = None
    residue: float = np.nan
    iota: float = np.nan
    info: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.rc) - 1

    def at(self, phi):
        n = np.arange(self.N + 1)
        a = np.outer(np.asarray(phi) * self.nfp, n)
        return np.cos(a) @ self.rc, np.sin(a) @ self.zs

    def dphi(self, phi):
        n = np.arange(self.N + 1)
        a = np.outer(np.asarray(phi) * self.nfp, n)
        return -np.sin(a) @ (self.nfp * n * self.rc), np.cos(a) @ (self.nfp * n * self.zs)


def _field_cyl(engine, R, Z, phi, grad):
    xp = _xp(R, Z)
    c, s = xp.cos(phi), xp.sin(phi)
    x = xp.stack([R * c, R * s, Z], -1)
    o = engine.evaluate(x, grad=grad, A=False)
    B = xp.asarray(o["B"])
    eR = xp.stack([c, s, 0 * c], -1)
    eP = xp.stack([-s, c, 0 * c], -1)
    BR, Bp, BZ = xp.sum(B * eR, -1), xp.sum(B * eP, -1), B[:, 2]
    if not grad:
        return BR, Bp, BZ, None
    G = xp.asarray(o["G"])
    GR = xp.einsum("tab,tb->ta", G, eR)
    GZ = G[:, :, 2]
    d = dict(BR_R=xp.sum(GR * eR, -1), Bp_R=xp.sum(GR * eP, -1), BZ_R=GR[:, 2],
             BR_Z=xp.sum(GZ * eR, -1), Bp_Z=xp.sum(GZ * eP, -1), BZ_Z=GZ[:, 2])
    return BR, Bp, BZ, d


def _axis_body(engine, u, nfp, N, nphi):
    """F, J of the axis Galerkin system: jnp inside jit (GPU, analytic sources), numpy on the host (CPU kernels:
    no tracing, no compilation per resolution)."""
    jnp = _xp(u)
    psi = 2 * np.pi * np.arange(nphi) / nphi
    n = np.arange(N + 1)
    Cj, Sj = jnp.asarray(np.cos(np.outer(psi, n))), jnp.asarray(np.sin(np.outer(psi, n)))
    phij = jnp.asarray(psi) / nfp
    nn = jnp.asarray(n.astype(float))
    w = np.full(N + 1, 2.0 / nphi); w[0] = 1.0 / nphi
    wj = jnp.asarray(w)
    rc_, zs_ = u[:N + 1], jnp.concatenate([jnp.zeros(1), u[N + 1:]])
    R = Cj @ rc_
    Z = Sj @ zs_
    Rp = -Sj @ (nfp * nn * rc_)
    Zp = Cj @ (nfp * nn * zs_)
    BR, Bp, BZ, d = _field_cyl(engine, R, Z, phij, True)
    ib = 1 / Bp
    VR, VZ = R * BR * ib, R * BZ * ib
    FR = Rp - VR
    FZ = Zp - VZ
    aRR = BR * ib + R * (d["BR_R"] * Bp - BR * d["Bp_R"]) * ib ** 2
    aRZ = R * (d["BR_Z"] * Bp - BR * d["Bp_Z"]) * ib ** 2
    aZR = BZ * ib + R * (d["BZ_R"] * Bp - BZ * d["Bp_R"]) * ib ** 2
    aZZ = R * (d["BZ_Z"] * Bp - BZ * d["Bp_Z"]) * ib ** 2
    PR = (Sj.T @ FR)[1:] * (2.0 / nphi)
    PZ = (Cj.T @ FZ) * wj
    F = jnp.concatenate([PR, PZ])
    dR_rc = Cj; dRp_rc = -Sj * (nfp * nn)[None, :]
    dZ_zs = Sj[:, 1:]; dZp_zs = (Cj * (nfp * nn)[None, :])[:, 1:]
    JFR_rc = dRp_rc - aRR[:, None] * dR_rc
    JFR_zs = -aRZ[:, None] * dZ_zs
    JFZ_rc = -aZR[:, None] * dR_rc
    JFZ_zs = dZp_zs - aZZ[:, None] * dZ_zs
    J = jnp.concatenate([
        jnp.concatenate([(Sj.T @ JFR_rc)[1:] * (2.0 / nphi), (Sj.T @ JFR_zs)[1:] * (2.0 / nphi)], 1),
        jnp.concatenate([(Cj.T @ JFZ_rc) * wj[:, None], (Cj.T @ JFZ_zs) * wj[:, None]], 1)], 0)
    return F, J


@partial(jax.jit, static_argnames=("kinds", "chunk", "N", "nphi"))
def _axis_shared(params, u, nfp, kinds, chunk, N, nphi):
    from .field.engine import EngineStub
    return _axis_body(EngineStub(params, kinds, chunk), u, nfp, N, nphi)


def solve_axis(engine, nfp, R0, Z0=0.0, N=None, tol=1e-14, maxit=30, verbose=False, guess=None):
    """Newton for a stellarator-symmetric closed field line of period one field period.

    guess: optional (rc, zs) arrays; else a circle at R0 (and Z0 = 0 by symmetry)."""
    N = N or 24
    nphi = fft_size(max(3 * N + 4, 32))
    rc = np.zeros(N + 1); zs = np.zeros(N + 1)
    if guess is not None:
        g0, g1 = guess
        k = min(len(g0), N + 1)
        rc[:k] = g0[:k]; zs[:k] = g1[:k]
    else:
        rc[0] = R0
    host = bool(getattr(engine, "cpu_kernels", False)) and not getattr(engine, "analytic", None)
    if host:
        sysj = lambda u: _axis_body(engine, u, float(nfp), N, nphi)
    elif not getattr(engine, "analytic", None):
        sysj = lambda u: _axis_shared(engine.params, u, float(nfp), kinds=engine.kinds, chunk=engine.target_chunk,
                                      N=N, nphi=nphi)
    else:
        sysj = jax.jit(lambda u: _axis_body(engine, u, float(nfp), N, nphi))
    u = np.concatenate([rc, zs[1:]]) if host else jnp.asarray(np.concatenate([rc, zs[1:]]))
    hist = []
    conv = False
    for it in range(maxit):
        F, J = sysj(u)
        nF = float(np.max(np.abs(np.asarray(F))))
        hist.append(nF)
        if verbose:
            print(f"  axis it {it} |F| {nF:.3e} R0 {float(u[0]):.12f}")
        if not np.isfinite(nF):
            break
        if nF < tol * max(1.0, float(u[0])):
            conv = True
            break
        du = small_solve(J, -F) if host else jnp.linalg.solve(J, -F)
        lam = 1.0
        step = float(np.max(np.abs(np.asarray(du))))
        if step > 0.2 * float(u[0]):
            lam = 0.2 * float(u[0]) / step
        u = u + lam * du
    rc_ = np.asarray(u[:N + 1]); zs_ = np.concatenate([[0.0], np.asarray(u[N + 1:])])
    ax = Axis(rc_, zs_, nfp, res=hist[-1] if hist else np.nan, converged=conv)
    ax.info["hist"] = hist
    return ax


def spectral_axis_monodromy(engine, axis: Axis, nsteps=None):
    """Monodromy, residue and transform on axis from one field + gradient evaluation along the
    axis (torux.orbits.spectral_monodromy): no field-line tracing."""
    from .orbits import spectral_monodromy
    nfp = axis.nfp
    M, ang = spectral_monodromy(engine, axis.rc, axis.zs, float(nfp), 0.0, 2 * np.pi / nfp,
                                nsteps=nsteps)
    tr = float(np.trace(M))
    axis.monodromy = M
    axis.residue = float((2 - tr) / 4)
    axis.info["tangent_rotation_per_period"] = ang
    axis.info["det"] = float(np.linalg.det(M))
    if 0 < axis.residue < 1:
        th = float(np.arccos(np.clip(tr / 2, -1, 1)))
        cands = np.array([kk * 2 * np.pi + sg * th for kk in range(-50, 51) for sg in (1, -1)])
        rot = cands[np.argmin(np.abs(cands - ang))]
        axis.iota = float(rot * nfp / (2 * np.pi))
    return axis


def monodromy(engine, axis: Axis, nsteps=64, k=8, nper=4, npieces=16):
    """One-field-period map Jacobian at the axis (forward-mode differentiation of the tracer),
    Greene's residue r = (2 - tr M) / 4 and the transform on axis per toroidal turn.

    The eigenvalues give the rotation per period only modulo 2 pi and up to sign; the branch
    is fixed by the angle a tangent vector sweeps under the linearised flow over ``nper``
    periods (counter-clockwise in the (R, Z) plane = positive iota, as for the tori), which
    differs from nper times the rotation by less than pi."""
    nfp = axis.nfp
    R0, Z0 = axis.at(np.array([0.0]))
    y0 = jnp.asarray([R0[0], Z0[0]])
    f = T.make_rhs(engine)
    period = 2 * np.pi / nfp
    sub = max(2, nsteps // npieces)

    @jax.jit
    def piece_jvp(y, v, ph0):
        def piece(yy):
            out, _ = T.trace(engine, yy[None, :], ph0, period / npieces, sub, k, f)
            return out[0]
        return jax.jvp(piece, (y,), (v,))

    y = y0
    cols = []
    for e in (jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0])):
        yy, v = y0, e
        for i in range(npieces):
            yy, v = piece_jvp(yy, v, i * period / npieces)
        cols.append(np.asarray(v))
    M = np.stack(cols, 1)
    tr = float(np.trace(M))
    residue = (2 - tr) / 4
    y, v = y0, jnp.asarray([1.0, 0.0])
    ang = 0.0
    for i in range(nper * npieces):
        y2, v2 = piece_jvp(y, v, (i % npieces) * period / npieces)
        a1 = np.arctan2(float(v[1]), float(v[0])); a2 = np.arctan2(float(v2[1]), float(v2[0]))
        ang += (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        y, v = y2, v2 / jnp.linalg.norm(v2)
    axis.monodromy = M
    axis.residue = float(residue)
    axis.info["tangent_rotation_per_period"] = ang / nper
    axis.info["closure_error"] = float(np.max(np.abs(np.asarray(y) - np.asarray(y0))))
    if 0 < residue < 1:
        th = float(np.arccos(np.clip(tr / 2, -1, 1)))
        cands = np.array([kk * 2 * np.pi + sg * th for kk in range(-50, 51) for sg in (1, -1)])
        rot = cands[np.argmin(np.abs(cands - ang / nper))]
        axis.iota = float(rot * nfp / (2 * np.pi))
    return axis


def _host(engine):
    """The engine's numba CPU kernels serve the field (the host tracer torux.fasttrace instead of the jitted torux.trace,
    whose XLA CPU kernels are 10-100x slower)."""
    return bool(getattr(engine, "cpu_kernels", False)) and not getattr(engine, "analytic", None)


def _trace_half_period(engine, nfp, R, nsteps, k, record=False):
    """Lines from (R_i, phi = 0, Z = 0) over [0, pi / nfp] by torux.fasttrace.LineTracer (each line with the sign of
    its own B_phi at the start): (end points (L, 2), states after each macro step [(L, 2)] if record), NaN for lines
    the tracer stops (|B_phi| < 1e-3 |B|, B_phi changing sign, a non-finite field)."""
    from .fasttrace import LineTracer
    R = np.asarray(R, float)
    B = np.asarray(engine.evaluate(np.stack([R, 0 * R, 0 * R], -1), grad=False, A=False)["B"], float)
    end = np.full((len(R), 2), np.nan)
    rec = [np.full((len(R), 2), np.nan) for _ in range(nsteps)] if record else None
    for sg in (1.0, -1.0):
        idx = np.flatnonzero(np.isfinite(B[:, 1]) & (sg * B[:, 1] > 0))       # B_phi = B_y at phi = 0
        if len(idx) == 0:
            continue
        tr = LineTracer(engine, nfp, k=k, steps_per_period=2 * nsteps, sign=sg)
        o = tr.trace(np.stack([R[idx], 0 * R[idx]], -1), 0.0, np.pi / nfp, nsteps=nsteps, record=record)
        ok = ~o["lost"] & np.all(np.isfinite(o["y"]), -1)
        end[idx[ok]] = o["y"][ok]
        if record:
            for j, yj in enumerate(o["rec"]):
                rec[j][idx[ok]] = yj[ok]
    return end, rec


def symmetric_line_scan(engine, nfp, Rmin, Rmax, nscan=64, nsteps=16, k=6):
    """Z at phi = pi / nfp of lines from (R, phi = 0, Z = 0); sign changes bracket
    stellarator-symmetric closed lines (axes and period-1 X lines).  Returns (R, Z_end, R_end); NaN where the
    host tracer stops a line."""
    R = np.linspace(Rmin, Rmax, nscan)
    if _host(engine):
        end, _ = _trace_half_period(engine, nfp, R, nsteps, k)
        return R, end[:, 1], end[:, 0]
    y0 = jnp.asarray(np.stack([R, 0 * R], -1))
    y, e = T.trace(engine, y0, 0.0, np.pi / nfp, nsteps, k)
    return R, np.asarray(y[:, 1]), np.asarray(y[:, 0])


def traced_axis_guess(engine, nfp, R0, N=16, nsteps=24, k=6):
    """A Fourier guess (rc, zs), arrays of N + 1, for solve_axis from the traced symmetric line through (R0, phi = 0,
    Z = 0): traced over [0, pi / nfp], mirrored by stellarator symmetry (R(-phi) = R(phi), Z(-phi) = -Z(phi)) into one
    period and fitted with N + 1 harmonics.  When R0 is a bracketed root of Z(pi / nfp) the line closes to the
    tracer's accuracy, so the guess lies inside Newton's basin even where a circle at R0 does not (the no-axis
    diagnosis: 16 elliptic period-one lines on CoilStellaration and cws sets reached only this way).  None if the
    line is lost within half a period.  The host tracer where the engine has CPU kernels (2 s where the jitted
    tracer took 160 s, the same guess: CoilStellaration DG9Wpp)."""
    if _host(engine):
        end, rec = _trace_half_period(engine, nfp, np.array([float(R0)]), nsteps, k, record=True)
        if not np.all(np.isfinite(end)):
            return None
        ys = [np.array([[R0, 0.0]])] + [r[:1] for r in rec]
    else:
        ys = [np.array([[R0, 0.0]])]
        y = ys[0]
        h = np.pi / nfp / nsteps
        for i in range(nsteps):
            y, _ = T.trace(engine, y, i * h, h, 1, k)
            y = np.asarray(y)
            if not np.all(np.isfinite(y)):
                return None
            ys.append(y)
    half = np.concatenate(ys, 0)                                    # (nsteps + 1, 2), phi = 0 .. pi / nfp
    Rf = np.concatenate([half[:-1, 0], half[::-1, 0][:-1]])         # psi = nfp phi over [0, 2 pi): 2 nsteps samples
    Zf = np.concatenate([half[:-1, 1], -half[::-1, 1][:-1]])
    psi = 2 * np.pi * np.arange(len(Rf)) / len(Rf)
    n = np.arange(N + 1)
    rc = np.array([np.mean(Rf * np.cos(m * psi)) * (1 if m == 0 else 2) for m in n])
    zs = np.array([np.mean(Zf * np.sin(m * psi)) * 2 for m in n])
    return rc, zs
