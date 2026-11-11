"""Closed field lines of period q field periods (island chains' O and X lines; the axis is q = 1).

Spectral formulation: the line is a curve (R, Z)(phi) of period L = 2 pi q / nfp, a Fourier
series in nu (phi - phi0) with nu = nfp / q.  Stellarator-symmetric lines through the
symmetry plane at phi0 have R a cosine and Z a sine series; general lines carry both.
The field-line equations dR/dphi = R B_R / B_phi, dZ/dphi = R B_Z / B_phi are imposed by
Galerkin projection on an oversampled phi grid and solved by Newton with the exact Jacobian
(one field + gradient evaluation per iteration).  The q-period map's monodromy comes from
the variational equation integrated along the converged line (tracer with jvp), giving
Greene's residue r = (2 - tr M)/4: r < 0 hyperbolic (X), 0 < r < 1 elliptic (O), r > 1
reflection-hyperbolic.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .torus import fft_size
from . import trace as T
from .torus_cpu import small_solve

jax.config.update("jax_enable_x64", True)


@dataclass
class ClosedLine:
    q: int
    nfp: int
    phi0: float
    a: np.ndarray            # R coefficients: cos (K+1); for general lines also sin in b
    b: np.ndarray            # Z coefficients: sin (K+1) (b[0] unused) for symmetric lines
    symmetric: bool = True
    aR_s: np.ndarray = None  # general lines: R sine and Z cosine parts
    bZ_c: np.ndarray = None
    res: float = np.nan
    converged: bool = False
    residue: float = np.nan
    monodromy: np.ndarray = None
    info: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.a) - 1

    def at(self, phi):
        nu = self.nfp / self.q
        k = np.arange(self.K + 1)
        ang = np.outer(nu * (np.asarray(phi) - self.phi0), k)
        R = np.cos(ang) @ self.a
        Z = np.sin(ang) @ self.b
        if not self.symmetric:
            R = R + np.sin(ang) @ self.aR_s
            Z = Z + np.cos(ang) @ self.bZ_c
        return R, Z

    def section_points(self, phi_s=0.0):
        """The q points of the line in the section phi = phi_s (mod 2 pi / nfp)."""
        P = 2 * np.pi / self.nfp
        phis = phi_s + P * np.arange(self.q)
        return np.stack(self.at(phis), -1)


def _cyl(engine, R, Z, phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    x = jnp.stack([R * c, R * s, Z], -1)
    o = engine.evaluate(x, grad=True, A=False)
    B, G = o["B"], o["G"]
    eR = jnp.stack([c, s, 0 * c], -1)
    eP = jnp.stack([-s, c, 0 * c], -1)
    BR, Bp, BZ = jnp.sum(B * eR, -1), jnp.sum(B * eP, -1), B[:, 2]
    GR = jnp.einsum("tab,tb->ta", G, eR)
    GZ = G[:, :, 2]
    ib = 1 / Bp
    aRR = BR * ib + R * (jnp.sum(GR * eR, -1) * Bp - BR * jnp.sum(GR * eP, -1)) * ib ** 2
    aRZ = R * (jnp.sum(GZ * eR, -1) * Bp - BR * jnp.sum(GZ * eP, -1)) * ib ** 2
    aZR = BZ * ib + R * (GR[:, 2] * Bp - BZ * jnp.sum(GR * eP, -1)) * ib ** 2
    aZZ = R * (GZ[:, 2] * Bp - BZ * jnp.sum(GZ * eP, -1)) * ib ** 2
    return R * BR * ib, R * BZ * ib, aRR, aRZ, aZR, aZZ


def _line_body(engine, u, nu, phi0, L, K, ngrid, symmetric):
    tt = np.arange(ngrid) / ngrid
    k = np.arange(K + 1)
    ang = 2 * np.pi * np.outer(tt, k)
    C, S = jnp.asarray(np.cos(ang)), jnp.asarray(np.sin(ang))
    kk = jnp.asarray(k.astype(float)) * nu
    w = np.full(K + 1, 2.0 / ngrid); w[0] = 1.0 / ngrid
    wj = jnp.asarray(w)
    phij = phi0 + L * jnp.asarray(tt)
    if symmetric:
        aC, aS = u[:K + 1], jnp.zeros(K + 1)
        bC, bS = jnp.zeros(K + 1), jnp.concatenate([jnp.zeros(1), u[K + 1:]])
    else:
        aC = u[:K + 1]
        aS = jnp.concatenate([jnp.zeros(1), u[K + 1:2 * K + 1]])
        bC = u[2 * K + 1:3 * K + 2]
        bS = jnp.concatenate([jnp.zeros(1), u[3 * K + 2:]])
    R = C @ aC + S @ aS
    Z = S @ bS + C @ bC
    Rp = -S @ (kk * aC) + C @ (kk * aS)
    Zp = C @ (kk * bS) - S @ (kk * bC)
    VR, VZ, aRR, aRZ, aZR, aZZ = _cyl(engine, R, Z, phij)
    FR, FZ = Rp - VR, Zp - VZ
    if symmetric:
        cR, dcR = C, -S * kk[None, :]
        sZ, dsZ = S[:, 1:], (C * kk[None, :])[:, 1:]
        JR_a = (S.T @ (dcR - aRR[:, None] * cR))[1:] * (2.0 / ngrid)
        JR_b = (S.T @ (-aRZ[:, None] * sZ))[1:] * (2.0 / ngrid)
        JZ_a = (C.T @ (-aZR[:, None] * cR)) * wj[:, None]
        JZ_b = (C.T @ (dsZ - aZZ[:, None] * sZ)) * wj[:, None]
        F = jnp.concatenate([(S.T @ FR)[1:] * (2.0 / ngrid), (C.T @ FZ) * wj])
        J = jnp.concatenate([jnp.concatenate([JR_a, JR_b], 1), jnp.concatenate([JZ_a, JZ_b], 1)], 0)
        return F, J
    bases = [(C, -S * kk[None, :]), (S[:, 1:], (C * kk[None, :])[:, 1:])]
    tests = [(C, wj), (S[:, 1:], jnp.full(K, 2.0 / ngrid))]
    Fs = []
    for Fg in (FR, FZ):
        for (Tm, tw) in tests:
            Fs.append((Tm.T @ Fg) * tw)
    F = jnp.concatenate(Fs)
    blocks = []
    for comp, (aX, aY) in enumerate(((aRR, aRZ), (aZR, aZZ))):
        for (Tm, tw) in tests:
            row = []
            for (Bm, dBm) in bases:
                own = dBm if comp == 0 else 0.0 * dBm
                row.append((Tm.T @ (own - aX[:, None] * Bm)) * tw[:, None])
            for (Bm, dBm) in bases:
                own = dBm if comp == 1 else 0.0 * dBm
                row.append((Tm.T @ (own - aY[:, None] * Bm)) * tw[:, None])
            blocks.append(jnp.concatenate(row, 1))
    return F, jnp.concatenate(blocks, 0)


@partial(jax.jit, static_argnames=("K", "ngrid", "symmetric", "kinds", "chunk"))
def _line_shared(params, u, nu, phi0, L, K, ngrid, symmetric, kinds, chunk):
    from .field.engine import EngineStub
    return _line_body(EngineStub(params, kinds, chunk), u, nu, phi0, L, K, ngrid, symmetric)


def solve_closed_line(engine, nfp, q, guess_R, guess_Z, phi0=0.0, K=None, symmetric=True, tol=1e-13,
                      maxit=40, verbose=False, max_step=None, deadline=None):
    """Newton for a closed field line of period q field periods.

    guess_R, guess_Z: callables phi -> R, Z (vectorised) giving the starting curve over
    [phi0, phi0 + 2 pi q / nfp).  deadline (absolute time, None: none): no iteration after it (the line
    unconverged, info["cut"] = True)."""
    K = K or max(16, 12 * q)
    K = int(((K + 7) // 8) * 8)                      # fewer distinct compiled shapes
    ngrid = fft_size(max(3 * K + 4, 32))
    L = 2 * np.pi * q / nfp
    nu = nfp / q
    tt = np.arange(ngrid) / ngrid
    phi = phi0 + L * tt
    k = np.arange(K + 1)
    ang = 2 * np.pi * np.outer(tt, k)
    C, S = np.cos(ang), np.sin(ang)
    w = np.full(K + 1, 2.0 / ngrid); w[0] = 1.0 / ngrid
    R0 = np.asarray(guess_R(phi)); Z0 = np.asarray(guess_Z(phi))
    a = (C.T @ R0) * w
    b = (S.T @ Z0) * (2.0 / ngrid)
    if symmetric:
        u = np.concatenate([a, b[1:]])
    else:
        aS = (S.T @ R0) * (2.0 / ngrid)
        bC = (C.T @ Z0) * w
        u = np.concatenate([a, aS[1:], bC, b[1:]])
    if not getattr(engine, "analytic", None):
        sysj = lambda uu: _line_shared(engine.params, uu, float(nu), float(phi0), float(L), K=K, ngrid=ngrid,
                                       symmetric=symmetric, kinds=engine.kinds, chunk=engine.target_chunk)
    else:
        sysj = jax.jit(lambda uu: _line_body(engine, uu, float(nu), float(phi0), float(L), K, ngrid, symmetric))
    u = jnp.asarray(u)
    hist = []
    conv = False
    scale = float(np.max(np.abs(R0)))
    max_step = max_step or 0.05 * scale
    cut = False
    for it in range(maxit):
        if deadline is not None and time.time() > deadline:
            cut = True
            break
        F, J = sysj(u)
        nF = float(jnp.max(jnp.abs(F)))
        hist.append(nF)
        if verbose:
            print(f"   line q={q} it {it} |F| {nF:.3e}")
        if not np.isfinite(nF):
            break
        if nF < tol * scale:
            conv = True
            break
        du = jnp.linalg.solve(J, -F)
        st = float(jnp.max(jnp.abs(du)))
        lam = 1.0 if st <= max_step else max_step / st
        u = u + lam * du
    un = np.asarray(u)
    if symmetric:
        aC, aS, bC, bS = un[:K + 1], None, None, np.concatenate([[0.0], un[K + 1:]])
    else:
        aC = un[:K + 1]; aS = np.concatenate([[0.0], un[K + 1:2 * K + 1]])
        bC = un[2 * K + 1:3 * K + 2]; bS = np.concatenate([[0.0], un[3 * K + 2:]])
    line = ClosedLine(q, nfp, phi0, aC, bS, symmetric, aS, bC, res=hist[-1] if hist else np.nan, converged=conv)
    line.info["hist"] = hist
    line.info["K"] = K
    if cut:
        line.info["cut"] = True
    return line


def line_monodromy(engine, line: ClosedLine, nsteps=None):
    """q-period map Jacobian at the line's point in the plane phi0, residue (spectral: one field +
    gradient evaluation along the line, no tracing)."""
    L = 2 * np.pi * line.q / line.nfp
    M, ang = spectral_monodromy(engine, line.a, line.b, line.nfp / line.q, line.phi0, L, nsteps=nsteps,
                                aR_s=None if line.symmetric else line.aR_s, bZ_c=None if line.symmetric else line.bZ_c)
    line.monodromy = M
    line.residue = float((2 - np.trace(M)) / 4)
    line.info["det"] = float(np.linalg.det(M))
    line.info["angle"] = ang
    return line


def line_monodromy_traced(engine, line: ClosedLine, steps_per_period=32, k=6):
    """q-period map Jacobian at the line's point in the section phi = phi0, by the tracer's jvp."""
    nfp, q = line.nfp, line.q
    period = 2 * np.pi / nfp
    R0, Z0 = line.at(np.array([line.phi0]))
    y0 = jnp.asarray([R0[0], Z0[0]])
    f = T.make_rhs(engine)

    @jax.jit
    def per_jvp(y, v, ph0):
        def piece(yy):
            out, _ = T.trace(engine, yy[None, :], ph0, period, steps_per_period, k, f)
            return out[0]
        return jax.jvp(piece, (y,), (v,))

    cols = []
    yend = None
    for e in (jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0])):
        y, v = y0, e
        for i in range(q):
            y, v = per_jvp(y, v, line.phi0 + i * period)
        cols.append(np.asarray(v))
        yend = np.asarray(y)
    M = np.stack(cols, 1)
    line.monodromy = M
    line.residue = float((2 - np.trace(M)) / 4)
    line.info["closure_error"] = float(np.max(np.abs(yend - np.asarray(y0))))
    return line


def spectral_monodromy(engine, coeff_R, coeff_Z, nu, phi0, L, n_grid=None, nsteps=None, symmetric=True,
                       aR_s=None, bZ_c=None, return_angle=True):
    """Monodromy of the variational equation dY/dphi = DV(X(phi), phi) Y along a closed line
    X(phi) = sum a_k cos(k nu (phi - phi0)) (+ sin), Z = sum b_k sin(...) (+ cos), over its
    period L = 2 pi / nu, from ONE field + gradient evaluation on a phi grid: DV is periodic and
    analytic, so its trigonometric interpolant on the grid is spectrally accurate, and the 2x2
    linear system is integrated with the interpolant (no further field evaluations).

    Returns (M, angle): the monodromy and the angle swept by the image of e_R (for the
    rotation-number branch).  Integrator: Gauss-Legendre collocation (s = 6, order 12) with
    nsteps equal steps; exact solves of the linear stage system per step."""
    K = len(coeff_R) - 1
    n_grid = n_grid or max(64, 4 * K + 8)
    n_grid += n_grid % 2
    tt = np.arange(n_grid) / n_grid
    phi = phi0 + L * tt
    k = np.arange(K + 1)
    ang = 2 * np.pi * np.outer(tt, k)
    C, S = np.cos(ang), np.sin(ang)
    R = C @ coeff_R + (S @ aR_s if aR_s is not None else 0.0)
    Z = S @ coeff_Z + (C @ bZ_c if bZ_c is not None else 0.0)
    _, _, aRR, aRZ, aZR, aZZ = _cyl(engine, jnp.asarray(R), jnp.asarray(Z), jnp.asarray(phi))
    Avals = np.stack([np.asarray(aRR), np.asarray(aRZ), np.asarray(aZR), np.asarray(aZZ)], -1)   # (n, 4)
    F = np.fft.rfft(Avals, axis=0) / n_grid                                                       # (n/2+1, 4)
    F[1:-1] *= 2.0
    F[-1] *= 1.0
    kk = np.arange(F.shape[0])

    def A_at(p):
        t = (np.atleast_1d(p) - phi0) / L
        e = np.exp(2j * np.pi * np.outer(t, kk))           # (m, nk)
        v = np.real(e @ F)                                   # (m, 4)
        return v.reshape(-1, 2, 2)

    # Gauss-Legendre collocation for Y' = A(phi) Y
    s = 6
    xg, wg = np.polynomial.legendre.leggauss(s)
    cg = 0.5 * (xg + 1)
    bg = 0.5 * wg
    # Butcher A matrix of the collocation method: a_ij = int_0^{c_i} l_j(t) dt
    V = np.vander(cg, s, increasing=True)
    Ab = np.zeros((s, s))
    for i in range(s):
        pw = cg[i] ** (np.arange(1, s + 1)) / np.arange(1, s + 1)
        Ab[i] = small_solve(V.T, pw)
    nsteps = nsteps or max(64, 2 * K)
    h = L / nsteps
    Y = np.eye(2)
    angle = 0.0
    v_prev = np.array([1.0, 0.0])
    I2s = np.eye(2 * s)
    for i in range(nsteps):
        p0 = phi0 + i * h
        Ast = A_at(p0 + cg * h)                              # (s, 2, 2)
        # stage equations: K_i = A_i (Y + h sum_j a_ij K_j)  -> (I - h blockdiag(A_i) (a (x) I)) K = blockdiag(A_i) Y
        Mbig = np.zeros((2 * s, 2 * s))
        for a in range(s):
            for b in range(s):
                Mbig[2 * a:2 * a + 2, 2 * b:2 * b + 2] = h * Ab[a, b] * Ast[a]
        rhs = np.concatenate([Ast[a] @ Y for a in range(s)], axis=0)       # (2s, 2)
        Kst = small_solve(I2s - Mbig, rhs)
        Y = Y + h * sum(bg[a] * Kst[2 * a:2 * a + 2] for a in range(s))
        v = Y[:, 0]
        a1 = math.atan2(v_prev[1], v_prev[0]); a2 = math.atan2(v[1], v[0])
        angle += (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        v_prev = v
    return Y, angle
