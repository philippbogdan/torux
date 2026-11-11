"""Magnetic axes without stellarator symmetry: period-one closed field lines as general Fourier
series in phi, their monodromy, residue and transform on axis, and the search for them.

The axis is R(phi) = sum_n rc_n cos(n psi) + rs_n sin(n psi), Z likewise (psi = nfp phi), solving
dR/dphi = R B_R / B_phi, dZ/dphi = R B_Z / B_phi by Galerkin projection on cosines and sines and
Newton with the exact Jacobian (solve_line_general: the same system as torux.orbits.solve_closed_line
(..., symmetric=False), but the field evaluated on concrete points -- the engine's numba kernels on
CPUs -- with a numpy assembly; the JAX path of torux.orbits costs ~4.5 s per iteration on HSX's 18k
segments).  No phase freedom: phi is the time of a non-autonomous periodic flow, a closed line is
isolated.  The monodromy is spectral (torux.orbits.spectral_monodromy with the sine and cosine
parts): one field + gradient evaluation along the line, Gauss-Legendre collocation of the
variational equation.

Starting guesses without a symmetry line: (1) given guesses, (2) circles about the coils' median
radius and height, (3) a scan of the one-period map P on a grid of the section phi = 0 (numpy RK4
on the solver's engine), local minima of |P(x) - x| refined together by Newton on P
(finite-difference Jacobian), the traced line then handed to the spectral Newton.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import orbits as OR
from .torus_cpu import small_solve


@dataclass
class GeneralAxis:
    rc: np.ndarray          # (K+1,)
    rs: np.ndarray          # (K+1,), rs[0] = 0
    zc: np.ndarray
    zs: np.ndarray          # zs[0] = 0
    nfp: int
    phi0: float = 0.0
    res: float = np.nan
    converged: bool = False
    monodromy: np.ndarray = None
    residue: float = np.nan
    iota: float = np.nan
    info: dict = field(default_factory=dict)
    general: bool = True

    @property
    def N(self):
        return len(self.rc) - 1

    def at(self, phi):
        n = np.arange(self.N + 1)
        a = np.outer(self.nfp * (np.asarray(phi, float) - self.phi0), n)
        c, s = np.cos(a), np.sin(a)
        return c @ self.rc + s @ self.rs, c @ self.zc + s @ self.zs

    def tail(self):
        k = max(1, min(3, self.N))
        t = max(np.max(np.abs(self.rc[-k:])), np.max(np.abs(self.rs[-k:])), np.max(np.abs(self.zc[-k:])),
                np.max(np.abs(self.zs[-k:])))
        return float(t / max(abs(self.rc[0]), 1e-300))

    def as_line(self):
        return OR.ClosedLine(1, self.nfp, self.phi0, self.rc, self.zs, False, self.rs, self.zc, res=self.res,
                             converged=self.converged)

    def record(self):
        return dict(rc=self.rc.tolist(), rs=self.rs.tolist(), zc=self.zc.tolist(), zs=self.zs.tolist(),
                    residue=self.residue, iota=self.iota, res=self.res, K=self.N,
                    asym=float(max(np.max(np.abs(self.rs)), np.max(np.abs(self.zc)))))

    @classmethod
    def from_symmetric(cls, ax):
        """A torux.axis.Axis (R cosine, Z sine) as a general axis."""
        z = np.zeros_like(np.asarray(ax.rc, float))
        return cls(np.array(ax.rc, float), z.copy(), z.copy(), np.array(ax.zs, float), ax.nfp, res=ax.res,
                   converged=ax.converged, monodromy=ax.monodromy, residue=ax.residue, iota=ax.iota)


def _guess_callables(guess, nfp):
    if isinstance(guess, GeneralAxis):
        return (lambda p: guess.at(p)[0]), (lambda p: guess.at(p)[1])
    if hasattr(guess, "rc") and hasattr(guess, "zs") and hasattr(guess, "at"):        # torux.axis.Axis
        return (lambda p: guess.at(p)[0]), (lambda p: guess.at(p)[1])
    if callable(guess[0]):
        return guess
    if np.ndim(guess[0]) == 1:
        # (rc, zs) of a stellarator-symmetric axis (torux.analysis.analyse's axis_guesses), psi = nfp phi
        rc_, zs_ = np.asarray(guess[0], float), np.asarray(guess[1], float)

        def sym(p, a, f):
            n = np.arange(len(a))
            return f(np.outer(nfp * np.asarray(p, float), n)) @ a
        return (lambda p: sym(p, rc_, np.cos)), (lambda p: sym(p, zs_, np.sin))
    R0, Z0 = float(guess[0]), float(guess[1])
    return (lambda p: np.full(np.shape(p), R0)), (lambda p: np.full(np.shape(p), Z0))


def _cyl_field(engine, R, Z, phi):
    """V = (R B_R, R B_Z) / B_phi and its (R, Z) Jacobian entries at concrete points (the engine's
    numba kernels on CPUs)."""
    c, s = np.cos(phi), np.sin(phi)
    x = np.stack([R * c, R * s, Z], -1)
    o = engine.evaluate(x, grad=True, A=False)
    B, G = np.asarray(o["B"]), np.asarray(o["G"])
    eR = np.stack([c, s, 0 * c], -1)
    eP = np.stack([-s, c, 0 * c], -1)
    BR, Bp, BZ = np.sum(B * eR, -1), np.sum(B * eP, -1), B[:, 2]
    GR = np.einsum("tab,tb->ta", G, eR)
    GZ = G[:, :, 2]
    ib = 1 / Bp
    aRR = BR * ib + R * (np.sum(GR * eR, -1) * Bp - BR * np.sum(GR * eP, -1)) * ib ** 2
    aRZ = R * (np.sum(GZ * eR, -1) * Bp - BR * np.sum(GZ * eP, -1)) * ib ** 2
    aZR = BZ * ib + R * (GR[:, 2] * Bp - BZ * np.sum(GR * eP, -1)) * ib ** 2
    aZZ = R * (GZ[:, 2] * Bp - BZ * np.sum(GZ * eP, -1)) * ib ** 2
    return R * BR * ib, R * BZ * ib, aRR, aRZ, aZR, aZZ


class _LineSystem:
    """Galerkin system of a general closed line of period L = 2 pi q / nfp (unknowns
    [aC (K+1), aS (K), bC (K+1), bS (K)]: R = sum aC cos + aS sin, Z = sum bC cos + bS sin in
    k nu (phi - phi0), nu = nfp / q), the field evaluated on concrete points between numpy halves."""

    def __init__(self, nfp, q, phi0, K, ngrid):
        self.K, self.ngrid = K, ngrid
        self.L = 2 * np.pi * q / nfp
        nu = nfp / q
        tt = np.arange(ngrid) / ngrid
        self.phi = phi0 + self.L * tt
        k = np.arange(K + 1)
        ang = 2 * np.pi * np.outer(tt, k)
        C, S = np.cos(ang), np.sin(ang)
        kk = k * nu
        self.Bm = np.concatenate([C, S[:, 1:]], 1)                       # values
        self.dBm = np.concatenate([-S * kk, (C * kk)[:, 1:]], 1)        # d/dphi
        w = np.full(K + 1, 2.0 / ngrid); w[0] = 1.0 / ngrid
        self.T = np.concatenate([C * w, S[:, 1:] * (2.0 / ngrid)], 1)    # test (projection) matrix
        self.n1 = 2 * K + 1

    def coeffs_of(self, R, Z):
        Pm = np.linalg.pinv(self.Bm)
        return np.concatenate([Pm @ R, Pm @ Z])

    def __call__(self, engine, u):
        n1 = self.n1
        uR, uZ = u[:n1], u[n1:]
        R, Z = self.Bm @ uR, self.Bm @ uZ
        Rp, Zp = self.dBm @ uR, self.dBm @ uZ
        VR, VZ, aRR, aRZ, aZR, aZZ = _cyl_field(engine, R, Z, self.phi)
        F = np.concatenate([self.T.T @ (Rp - VR), self.T.T @ (Zp - VZ)])
        B, dB, T = self.Bm, self.dBm, self.T
        J = np.block([[T.T @ (dB - aRR[:, None] * B), T.T @ (-aRZ[:, None] * B)],
                      [T.T @ (-aZR[:, None] * B), T.T @ (dB - aZZ[:, None] * B)]])
        return F, J


def solve_line_general(engine, nfp, q, guess, K=16, phi0=0.0, tol=1e-13, maxit=40, verbose=False, max_step=None):
    """Newton for a general closed line of period q field periods (numba field on concrete points).
    guess: as solve_axis_general.  Returns a GeneralAxis-like record (q = 1: the axis)."""
    from .torus import fft_size
    gR, gZ = _guess_callables(guess, nfp)
    K = int(((K + 7) // 8) * 8)
    ngrid = fft_size(max(3 * K + 4, 32))
    sysm = _LineSystem(nfp, q, phi0, K, ngrid)
    R0, Z0 = np.asarray(gR(sysm.phi), float), np.asarray(gZ(sysm.phi), float)
    u = sysm.coeffs_of(R0, Z0)
    scale = float(np.max(np.abs(R0)))
    max_step = max_step or 0.05 * scale
    hist = []
    conv = False
    best = (np.inf, u)
    for it in range(maxit):
        F, J = sysm(engine, u)
        nF = float(np.max(np.abs(F)))
        hist.append(nF)
        if verbose:
            print(f"   line q={q} K={K} it {it} |F| {nF:.3e}", flush=True)
        if not np.isfinite(nF):
            break
        if nF < best[0]:
            best = (nF, u)
        if nF < tol * scale:
            conv = True
            break
        # no progress for 10 iterations (a start outside the basin): stop early
        if it >= 10 and nF > 0.5 * min(hist[:-10]):
            break
        try:
            du = small_solve(J, -F)
        except np.linalg.LinAlgError:
            break
        st = float(np.max(np.abs(du)))
        lam = 1.0 if st <= max_step else max_step / st
        u = u + lam * du
    nF, u = best
    n1 = sysm.n1
    rc, rs = u[:K + 1], np.concatenate([[0.0], u[K + 1:n1]])
    zc, zs = u[n1:n1 + K + 1], np.concatenate([[0.0], u[n1 + K + 1:]])
    nfp_eff = nfp / q if q != 1 else nfp
    ax = GeneralAxis(rc, rs, zc, zs, nfp_eff, phi0=phi0, res=float(nF) if np.isfinite(nF) else np.nan, converged=conv)
    ax.info.update(hist=hist, K=K, q=q)
    return ax


def solve_axis_general(engine, nfp, guess, K=None, tol=1e-13, maxit=40, verbose=False, Kmax=192, tail_tol=1e-13):
    """Newton for a period-one closed field line on the general basis, resolution raised until the
    last harmonics fall below tail_tol (relative to R_00).

    guess: GeneralAxis, torux.axis.Axis, (R(phi), Z(phi)) callables, or (R0, Z0) (a circle)."""
    K = K or (max(16, guess.N) if hasattr(guess, "N") else 16)
    ax = solve_line_general(engine, nfp, 1, guess, K=K, tol=tol, maxit=maxit, verbose=verbose)
    if not ax.converged:
        return ax
    for K2 in (24, 32, 48, 64, 96, 128, 192):
        if K2 > Kmax or ax.tail() < tail_tol:
            break
        if K2 <= ax.N:
            continue
        ax2 = solve_line_general(engine, nfp, 1, ax, K=K2, tol=tol, maxit=maxit, verbose=verbose)
        if not ax2.converged:
            break
        ax = ax2
    return ax


def axis_monodromy(engine, ax: GeneralAxis, nsteps=None):
    """Spectral monodromy of the one-period map at the axis, Greene's residue and the transform on
    axis (per toroidal turn; branch from the angle a tangent vector sweeps, counter-clockwise in the
    (R, Z) plane = positive iota, as torux.axis)."""
    nfp = ax.nfp
    Mo, ang = OR.spectral_monodromy(engine, ax.rc, ax.zs, float(nfp), ax.phi0, 2 * np.pi / nfp, nsteps=nsteps,
                                    aR_s=ax.rs, bZ_c=ax.zc)
    tr = float(np.trace(Mo))
    ax.monodromy = Mo
    ax.residue = float((2 - tr) / 4)
    ax.info["tangent_rotation_per_period"] = ang
    ax.info["det"] = float(np.linalg.det(Mo))
    ax.iota = np.nan
    if 0 < ax.residue < 1:
        th = float(np.arccos(np.clip(tr / 2, -1, 1)))
        cands = np.array([kk * 2 * np.pi + sg * th for kk in range(-50, 51) for sg in (1, -1)])
        rot = cands[np.argmin(np.abs(cands - ang))]
        ax.iota = float(rot * nfp / (2 * np.pi))
    return ax


# ============================================================================ period-map scan
def _rhs(engine, phi, R, Z, bphi_frac=1e-3):
    c, s = np.cos(phi), np.sin(phi)
    x = np.stack([R * c, R * s, Z], -1)
    ok = np.isfinite(x).all(-1)
    xs = np.where(ok[:, None], x, np.array([1.0, 0.0, 0.0]))
    B = np.asarray(engine.evaluate(xs, grad=False, A=False)["B"])
    BR = B[:, 0] * c + B[:, 1] * s
    Bp = -B[:, 0] * s + B[:, 1] * c
    Bm = np.linalg.norm(B, axis=-1)
    good = ok & (np.abs(Bp) >= bphi_frac * Bm) & (R > 0)
    Bp = np.where(good, Bp, np.nan)
    return R * BR / Bp, R * B[:, 2] / Bp


def trace_rk4(engine, R, Z, phi0, dphi, nsteps):
    """Classical RK4 in phi for many lines at once (guesses only; lost lines become NaN)."""
    R = np.array(R, float)
    Z = np.array(Z, float)
    h = dphi / nsteps
    phi = float(phi0)
    for _ in range(nsteps):
        k1 = _rhs(engine, phi, R, Z)
        k2 = _rhs(engine, phi + h / 2, R + h / 2 * k1[0], Z + h / 2 * k1[1])
        k3 = _rhs(engine, phi + h / 2, R + h / 2 * k2[0], Z + h / 2 * k2[1])
        k4 = _rhs(engine, phi + h, R + h * k3[0], Z + h * k3[1])
        R = R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Z = Z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        phi += h
    return R, Z


def traced_curve(engine, R0, Z0, phis, steps_per_period, nfp):
    """R, Z of the line from (R0, 0, Z0) at the increasing angles phis (>= 0)."""
    out_R, out_Z = [], []
    R, Z, ph = np.array([R0], float), np.array([Z0], float), 0.0
    hmax = 2 * np.pi / nfp / steps_per_period
    for p in np.asarray(phis, float):
        if p > ph:
            n = max(1, int(math.ceil((p - ph) / hmax)))
            R, Z = trace_rk4(engine, R, Z, ph, p - ph, n)
            ph = p
        out_R.append(R[0]); out_Z.append(Z[0])
    return np.array(out_R), np.array(out_Z)


def period_map_scan(engine, source, nfp, n=11, steps=None, width=0.6, newton_its=8, max_seeds=6, verbose=False,
                    broad=False):
    """Fixed points of the one-period map: grid scan of |P(x) - x| on the section phi = 0 (RK4,
    ``steps`` per period) about the coils' median (R, Z) (half-width ``width`` x the median coil
    distance from it), or with broad=True over the bounding box of the coils' (R, Z) projection;
    local minima refined together by Newton on P (finite-difference Jacobian, 3 lines per seed and
    iteration).  Returns [(R, Z, |P(x) - x|)] sorted by |P(x) - x|."""
    pts = source.sample_points()
    if not len(pts):
        return []
    steps = steps or max(24, 96 // max(1, int(nfp)))
    Rc = np.hypot(pts[:, 0], pts[:, 1])
    Zc = pts[:, 2]
    Rm, Zm = float(np.median(Rc)), float(np.median(Zc))
    rco = float(np.median(np.hypot(Rc - Rm, Zc - Zm)))
    if broad:
        Rg = np.linspace(np.min(Rc), np.max(Rc), n + 2)[1:-1]
        Zg = np.linspace(np.min(Zc), np.max(Zc), n + 2)[1:-1]
        rco = 0.5 * max(np.ptp(Rc), np.ptp(Zc))
    else:
        Rg = Rm + width * rco * np.linspace(-1, 1, n)
        Zg = Zm + width * rco * np.linspace(-1, 1, n)
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    P = 2 * np.pi / nfp
    R1, Z1 = trace_rk4(engine, RR.ravel(), ZZ.ravel(), 0.0, P, steps)
    d = np.hypot(R1 - RR.ravel(), Z1 - ZZ.ravel()).reshape(n, n)
    d = np.where(np.isfinite(d), d, np.inf)
    seeds = []
    for i in range(n):
        for j in range(n):
            if not np.isfinite(d[i, j]):
                continue
            nb = d[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if d[i, j] <= np.min(nb):
                seeds.append((d[i, j], RR[i, j], ZZ[i, j]))
    seeds.sort()
    X = np.array([[r, z] for _, r, z in seeds[:max_seeds]], float).reshape(-1, 2)
    if not len(X):
        return []
    hstep = 1e-6 * Rm
    active = np.ones(len(X), bool)
    gnorm = np.full(len(X), np.inf)
    done = np.zeros(len(X), bool)
    for _ in range(newton_its):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        x = X[idx]
        Rs = np.concatenate([x[:, 0], x[:, 0] + hstep, x[:, 0]])
        Zs = np.concatenate([x[:, 1], x[:, 1], x[:, 1] + hstep])
        Rp, Zp = trace_rk4(engine, Rs, Zs, 0.0, P, steps)
        k = len(idx)
        for a, i in enumerate(idx):
            r = [Rp[a], Rp[a + k], Rp[a + 2 * k]]
            z = [Zp[a], Zp[a + k], Zp[a + 2 * k]]
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(z))):
                active[i] = False
                continue
            g = np.array([r[0] - X[i, 0], z[0] - X[i, 1]])
            gnorm[i] = np.linalg.norm(g)
            if gnorm[i] < 1e-8 * Rm:
                active[i] = False
                done[i] = True
                continue
            Jm = np.array([[(r[1] - r[0]) / hstep - 1, (r[2] - r[0]) / hstep],
                           [(z[1] - z[0]) / hstep, (z[2] - z[0]) / hstep - 1]])
            try:
                dx = small_solve(Jm, -g)
            except np.linalg.LinAlgError:
                active[i] = False
                continue
            if np.linalg.norm(dx) > 0.3 * rco:
                dx *= 0.3 * rco / np.linalg.norm(dx)
            X[i] = X[i] + dx
            if gnorm[i] < 1e-6 * Rm and np.linalg.norm(dx) < 1e-8 * Rm:
                active[i] = False
                done[i] = True
    out = []
    for i in np.argsort(gnorm):
        # a guess for the spectral Newton needs only a small one-period displacement
        if not (done[i] or gnorm[i] < 1e-5 * Rm):
            continue
        if any(math.hypot(X[i, 0] - a, X[i, 1] - b) < 1e-6 * Rm for a, b, _ in out):
            continue
        out.append((float(X[i, 0]), float(X[i, 1]), float(gnorm[i])))
        if verbose:
            print(f"   map fixed point R {X[i, 0]:.9f} Z {X[i, 1]:.9f} |P-x| {gnorm[i]:.1e}", flush=True)
    return out


def find_axes_general(engine, source, guesses=(), scan=False, verbose=False, K0=16, steps=None):
    """Period-one closed lines (magnetic axes and period-one X lines) without symmetry assumptions.
    Returns GeneralAxis objects with monodromy, residue and (elliptic ones) iota."""
    nfp = source.nfp
    pts = source.sample_points()
    Rc = np.hypot(pts[:, 0], pts[:, 1]) if len(pts) else np.array([1.0])
    Rm = float(np.median(Rc))
    Zm = float(np.median(pts[:, 2])) if len(pts) else 0.0
    out = []

    def add(ax):
        if not ax.converged:
            return
        phs = np.linspace(0, 2 * np.pi / nfp, 33)[:-1]
        Ra, Za = ax.at(phs)
        if not (np.all(np.isfinite(Ra)) and np.all(Ra > 0)):
            return
        for a in out:
            Rb, Zb = a.at(phs)
            if np.max(np.abs(Ra - Rb)) < 1e-8 * Rm and np.max(np.abs(Za - Zb)) < 1e-8 * Rm:
                return
        axis_monodromy(engine, ax)
        out.append(ax)
        if verbose:
            print(f"   axis R(0) {Ra[0]:.10f} Z(0) {Za[0]:.10f} residue {ax.residue:.6f} iota {ax.iota:.12f} "
                  f"K {ax.N} res {ax.res:.1e}", flush=True)

    def have():
        return any(0 < a.residue < 1 for a in out)
    for g in guesses:
        try:
            add(solve_axis_general(engine, nfp, g, K=K0))
        except Exception:  # noqa: BLE001
            pass
        if have():
            break
    if not have():
        for f in (1.0, 0.95, 1.05, 0.9, 1.1):
            try:
                add(solve_axis_general(engine, nfp, (Rm * f, Zm), K=K0))
            except Exception:  # noqa: BLE001
                pass
            if have():
                break
    steps = steps or max(24, 96 // max(1, int(nfp)))
    for broad in (False, True):
        if not (scan or not have()):
            break
        for (R0, Z0, _) in period_map_scan(engine, source, nfp, n=15 if broad else 11, steps=steps, verbose=verbose,
                                           broad=broad):
            memo = {}

            def curve(p, R0=R0, Z0=Z0, memo=memo):
                key = (len(p), float(p[0]), float(p[-1]))
                if key not in memo:
                    memo[key] = traced_curve(engine, R0, Z0, p, steps, nfp)
                return memo[key]
            try:
                add(solve_axis_general(engine, nfp, ((lambda p, c=curve: c(p)[0]), (lambda p, c=curve: c(p)[1])),
                                       K=K0))
            except Exception:  # noqa: BLE001
                pass
    return out
