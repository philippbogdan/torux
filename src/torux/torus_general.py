"""Invariant tori (flux surfaces) without stellarator symmetry: the parameterisation method on
the general double Fourier basis.

A flux surface with rotational transform iota is a torus X(theta, phi) = (R, Z)(theta, phi) on
which field lines are straight lines of slope iota,

    E(X, iota) := (d_phi + iota d_theta) X - V(X) = 0,    V = (R B_R / B_phi, R B_Z / B_phi).

Basis.  R = sum rc cos(m theta - n psi) + rs sin(m theta - n psi), Z likewise with zc, zs,
psi = nfp (phi - phi0), m = 0..M, n = -N..N; the cosine coefficients use m = 0 with n >= 0 and
m >= 1 with all n, the sine coefficients m = 0 with n >= 1 and m >= 1 with all n (the redundant
halves of the m = 0 row dropped).  Both residuals are Galerkin-projected on cosines and sines
(the same index sets) on the full oversampled one-field-period grid (2.5x; no symmetry: every
grid point is evaluated).

Two degeneracies the symmetric problem does not have, and what removes them:

* theta-translation.  E is invariant under X(theta, phi) -> X(theta + c, phi): at a solution
  X_theta (with d iota = 0) is a null vector of the Jacobian (the flux is translation invariant
  too).  Phase condition (the integral phase condition of periodic-orbit continuation):

      <X - X_ref, X_ref,theta> = 0      (mean over the grid; <X_ref, X_ref,theta> = 0 identically)

  with X_ref the starting torus of the solve (the predictor in a continuation, the previous
  resolution's torus in a refinement).  It is linear in the coefficients,
      sum_mn m/2 (rc rs_ref - rs rc_ref + zc zs_ref - zs zc_ref) = 0,
  and its derivative along the null direction, <X_theta, X_ref,theta>, is ~ <|X_theta|^2> > 0
  for X near X_ref: the one-parameter family X(theta + c) meets it transversally, at an
  isolated c.  It is satisfied by every stellarator-symmetric torus when X_ref is symmetric
  (it involves only the products of symmetric with antisymmetric coefficients), so a
  symmetric field gives back the symmetric torus.

* a redundant equation.  For ANY closed torus (not only invariant ones), with
  Omega = B_phi dR ^ dZ the invariant 2-form of the field-line flow (div B = 0),
      Omega(X_theta, E) = -B . (X_theta x X_phi),
  whose integral over the torus is the net flux through a closed surface: zero.  So the
  2 x (number of modes) projected equations plus the flux equation, with the coefficients and
  iota as unknowns, are square but one equation is (up to truncation) a combination of the
  others.  Unfolding parameter lambda (one more unknown):

      E(X, iota) - lambda nu = 0,    nu = (-Z_theta, R_theta) / (B_phi |X_theta|^2),

  Omega(X_theta, nu) = 1, so integrating Omega(X_theta, .) gives lambda <1> = <Omega(X_theta, E)>:
  lambda is zero in the continuum and of the size of the product of two spectral tails in the
  Galerkin system (it is reported; the off-grid check measures E itself, without lambda).
  nu is taken at the current iterate (frozen within a Newton step; its derivative enters only
  multiplied by lambda ~ 0).  The approximate-reducibility preconditioner of the Newton-Krylov
  solver inverts exactly this unfolded operator (lambda absorbs the mean of the normal
  component, the phase condition fixes the mean of the tangential one).

Unknowns u = [rc, rs, zc, zs (masked, flattened), iota, lambda]; equations [P_c E_R, P_s E_R,
P_c E_Z, P_s E_Z, flux (or iota) closure, phase].  Flux closure exactly as torux.torus:
Phi = -2 pi mean(A_R R_theta + A_Z Z_theta).  The dense Jacobian is assembled exactly from the
grid FFTs of the four coefficient functions (Toeplitz + Hankel, all four parity blocks).

Field: grid points in one jitted function, ``engine.evaluate(x, grad=True, A=True)`` on
concrete points (numba kernels on CPUs), F and J in a second jitted function (shapes static:
the programs are shared by every source).
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

jax.config.update("jax_enable_x64", True)


# ============================================================================ basis
class GeneralBasis:
    """General double Fourier basis on a one-field-period grid (theta x psi).

    Coefficient arrays have shape (M+1, 2N+1), column j <-> n = j - N.  Cosine coefficients
    (rc, zc) use the mask ``mask_c`` (m = 0: n >= 0; m >= 1: all n), sine coefficients (rs, zs)
    ``mask_s`` (mask_c without (0, 0))."""

    def __init__(self, nfp, M, N, ntheta=None, nphi=None, oversample=2.5, shift=False, phi0=0.0):
        self.nfp = int(nfp) if float(nfp).is_integer() else float(nfp)
        self.M, self.N = int(M), int(N)
        self.phi0 = float(phi0)
        self.ntheta = ntheta or fft_size(max(int(oversample * M) + 4, 2 * M + 2, 16))
        self.nphi = nphi or (fft_size(max(int(oversample * N) + 4, 2 * N + 2, 16)) if N > 0 else 1)
        self.shift = shift
        m = np.arange(M + 1)
        n = np.arange(-N, N + 1)
        self.m, self.n = m, n
        th0 = np.pi / self.ntheta if shift else 0.0
        ps0 = np.pi / self.nphi if shift else 0.0
        self.theta = th0 + 2 * np.pi * np.arange(self.ntheta) / self.ntheta
        self.psi = ps0 + 2 * np.pi * np.arange(self.nphi) / self.nphi
        self.phi = self.phi0 + self.psi / self.nfp
        mt = np.outer(self.theta, m)
        np_ = np.outer(self.psi, n)
        self.Ct, self.St = jnp.asarray(np.cos(mt)), jnp.asarray(np.sin(mt))
        self.Cp, self.Sp = jnp.asarray(np.cos(np_)), jnp.asarray(np.sin(np_))
        mc = np.ones((M + 1, 2 * N + 1), bool)
        mc[0, :N] = False
        ms = mc.copy()
        ms[0, N] = False
        self.mask_c, self.mask_s = mc, ms
        self.ic = np.flatnonzero(mc.ravel())
        self.is_ = np.flatnonzero(ms.ravel())
        self.nc, self.ns = len(self.ic), len(self.is_)
        self.nco = self.nc + self.ns                 # coefficients per component
        self.size = 2 * self.nco + 2                 # unknowns (= equations)
        self.mm = jnp.asarray(np.repeat(m[:, None], 2 * N + 1, 1).astype(float))
        self.nn = jnp.asarray(np.repeat(n[None, :], M + 1, 0).astype(float))
        self.modes_c = np.stack(np.unravel_index(self.ic, mc.shape), -1) - np.array([0, N])
        self.modes_s = np.stack(np.unravel_index(self.is_, ms.shape), -1) - np.array([0, N])
        self.zero_c = jnp.asarray((self.modes_c[:, 0] == 0) & (self.modes_c[:, 1] == 0))

    # ------------------------------------------------------------------ packing
    def pack(self, rc, rs, zc, zs, iota, lam=0.0):
        f = lambda a, idx: jnp.asarray(a, dtype=jnp.float64).ravel()[idx]
        return jnp.concatenate([f(rc, self.ic), f(rs, self.is_), f(zc, self.ic), f(zs, self.is_),
                                jnp.atleast_1d(jnp.asarray(iota, dtype=jnp.float64)),
                                jnp.atleast_1d(jnp.asarray(lam, dtype=jnp.float64))])

    def unpack(self, u):
        shape = (self.M + 1, 2 * self.N + 1)
        size = shape[0] * shape[1]
        nc, ns = self.nc, self.ns
        put = lambda v, idx: jnp.zeros(size).at[idx].set(v, unique_indices=True).reshape(shape)
        rc = put(u[0:nc], self.ic)
        rs = put(u[nc:nc + ns], self.is_)
        zc = put(u[nc + ns:2 * nc + ns], self.ic)
        zs = put(u[2 * nc + ns:2 * nc + 2 * ns], self.is_)
        return rc, rs, zc, zs, u[-2], u[-1]

    # ------------------------------------------------------------------ synthesis / projection
    def synth(self, ac, as_):
        """sum ac cos(m theta - n psi) + as sin(m theta - n psi) on the grid."""
        return self.Ct @ (ac @ self.Cp.T - as_ @ self.Sp.T) + self.St @ (ac @ self.Sp.T + as_ @ self.Cp.T)

    def proj(self, f):
        """Galerkin projections (pc, ps): pc = 2 <f, cos> (<f, 1> at (0, 0)), ps = 2 <f, sin>."""
        g = self.ntheta * self.nphi
        fC, fS = f @ self.Cp, f @ self.Sp
        pc = (self.Ct.T @ fC + self.St.T @ fS) * (2.0 / g)
        pc = pc.at[0, self.N].multiply(0.5)
        ps = (self.St.T @ fC - self.Ct.T @ fS) * (2.0 / g)
        return pc, ps

    def projv(self, f):
        pc, ps = self.proj(f)
        return jnp.concatenate([pc.ravel()[self.ic], ps.ravel()[self.is_]])

    def means(self, f):
        """[mean(f cos phi_j) for cosine modes, mean(f sin phi_j) for sine modes] (no factor 2)."""
        pc, ps = self.proj(f)
        c = pc.ravel()[self.ic]
        return jnp.concatenate([jnp.where(self.zero_c, c, 0.5 * c), 0.5 * ps.ravel()[self.is_]])

    def geometry(self, rc, rs, zc, zs):
        nfp, mm, nn = self.nfp, self.mm, self.nn
        R = self.synth(rc, rs)
        Z = self.synth(zc, zs)
        Rt = self.synth(mm * rs, -mm * rc)
        Zt = self.synth(mm * zs, -mm * zc)
        Rp = nfp * self.synth(-nn * rs, nn * rc)
        Zp = nfp * self.synth(-nn * zs, nn * zc)
        return R, Z, Rt, Zt, Rp, Zp

    def points(self, R, Z):
        """Cartesian points (T, 3) of the full grid and their toroidal angles."""
        P = jnp.broadcast_to(jnp.asarray(self.phi)[None, :], (self.ntheta, self.nphi)).ravel()
        Rf, Zf = R.ravel(), Z.ravel()
        return jnp.stack([Rf * jnp.cos(P), Rf * jnp.sin(P), Zf], -1), P


def cylindrical_from(B, A, G, phi, need_grad=True, need_A=True):
    """Cylindrical components of Cartesian B, A and dB_c/dR, dB_c/dZ at points of toroidal angle phi."""
    c, s = jnp.cos(phi), jnp.sin(phi)
    eR = jnp.stack([c, s, jnp.zeros_like(c)], -1)
    eP = jnp.stack([-s, c, jnp.zeros_like(c)], -1)
    out = dict(BR=jnp.sum(B * eR, -1), Bp=jnp.sum(B * eP, -1), BZ=B[:, 2])
    if need_A:
        out["AR"] = jnp.sum(A * eR, -1)
        out["Ap"] = jnp.sum(A * eP, -1)
        out["AZ"] = A[:, 2]
    if need_grad:
        GR = jnp.einsum("tab,tb->ta", G, eR)
        GZ = G[:, :, 2]
        out["dBR_R"], out["dBp_R"], out["dBZ_R"] = jnp.sum(GR * eR, -1), jnp.sum(GR * eP, -1), GR[:, 2]
        out["dBR_Z"], out["dBp_Z"], out["dBZ_Z"] = jnp.sum(GZ * eR, -1), jnp.sum(GZ * eP, -1), GZ[:, 2]
    return out


def field_on_grid(engine, basis: GeneralBasis, R, Z, need_grad=True, need_A=True):
    x, P = basis.points(R, Z)
    o = engine.evaluate(x, grad=need_grad, A=need_A)
    cyl = cylindrical_from(o["B"], o.get("A"), o.get("G"), P, need_grad, need_A)
    return {k: v.reshape(basis.ntheta, basis.nphi) for k, v in cyl.items()}


# ============================================================================ state
@dataclass
class GeneralTorus:
    rc: np.ndarray
    rs: np.ndarray
    zc: np.ndarray
    zs: np.ndarray
    iota: float
    flux: float
    M: int
    N: int
    nfp: int
    res: float = np.nan              # max |projected residual| at the solution (m)
    bn_off: float = np.nan           # max |B.n|/|B| on the half-shifted grid
    res_off: float = np.nan
    its: int = 0
    converged: bool = False
    tail: float = np.nan
    info: dict = field(default_factory=dict)
    phi0: float = 0.0
    lam: float = 0.0                 # unfolding parameter at the solution (truncation level)
    general: bool = True

    def arrays(self):
        return self.rc, self.rs, self.zc, self.zs

    def resized(self, M, N):
        out = []
        m = min(M, self.M)
        n = min(N, self.N)
        for a in self.arrays():
            b = np.zeros((M + 1, 2 * N + 1))
            b[:m + 1, N - n:N + n + 1] = np.asarray(a)[:m + 1, self.N - n:self.N + n + 1]
            out.append(b)
        return GeneralTorus(*out, self.iota, self.flux, M, N, self.nfp, phi0=self.phi0)

    def copy(self):
        t = GeneralTorus(*(np.array(a, copy=True) for a in self.arrays()), self.iota, self.flux, self.M, self.N,
                         self.nfp, res=self.res, bn_off=self.bn_off, res_off=self.res_off, its=self.its,
                         converged=self.converged, tail=self.tail, info=dict(self.info), phi0=self.phi0, lam=self.lam)
        return t

    def amplitude(self):
        return np.sqrt(self.rc ** 2 + self.rs ** 2 + self.zc ** 2 + self.zs ** 2)

    def tails(self):
        """Relative amplitude of the last two poloidal and toroidal mode rows (truncation measure)."""
        a = self.amplitude()
        scale = max(np.max(a[1:, :]), 1e-300)
        tm = np.max(a[max(self.M - 1, 1):, :]) / scale if self.M >= 2 else 0.0
        tn = (max(np.max(a[:, :2]), np.max(a[:, -2:])) / scale) if self.N >= 2 else 0.0
        return float(tm), float(tn)

    def asymmetry(self):
        """max(|rs|, |zc|) relative to the largest m >= 1 coefficient: 0 for a stellarator-symmetric
        torus in symmetric phase."""
        a = self.amplitude()
        scale = max(np.max(a[1:, :]), 1e-300)
        return float(max(np.max(np.abs(self.rs)), np.max(np.abs(self.zc))) / scale)

    def geometry_grid(self, theta, phi):
        """R, Z on the tensor grid theta (nt,) x phi (np,)."""
        m = np.arange(self.M + 1)
        n = np.arange(-self.N, self.N + 1)
        at = np.outer(np.asarray(theta, float), m)
        ap = np.outer(self.nfp * (np.asarray(phi, float) - self.phi0), n)
        Ct, St, Cp, Sp = np.cos(at), np.sin(at), np.cos(ap), np.sin(ap)

        def syn(ac, as_):
            return Ct @ (ac @ Cp.T - as_ @ Sp.T) + St @ (ac @ Sp.T + as_ @ Cp.T)
        return syn(self.rc, self.rs), syn(self.zc, self.zs)

    def geometry_on(self, theta, phi):
        """R, Z at arbitrary (theta, phi) arrays (same shape)."""
        x, R, Z = torus_points(self, theta, phi)
        sh = np.shape(theta)
        return R.reshape(sh), Z.reshape(sh)

    def shifted(self, c):
        """The same torus with theta -> theta + c (coefficients rotated)."""
        m = np.arange(self.M + 1)[:, None]
        cc, ss = np.cos(m * c), np.sin(m * c)
        # a cos(phi + m c) + b sin(phi + m c) = (a cc + b ss) cos phi + (b cc - a ss) sin phi
        out = []
        for a, b in ((self.rc, self.rs), (self.zc, self.zs)):
            out += [a * cc + b * ss, b * cc - a * ss]
        t = self.copy()
        t.rc, t.rs, t.zc, t.zs = out
        mc = np.ones_like(t.rc, bool); mc[0, :self.N] = False
        ms = mc.copy(); ms[0, self.N] = False
        t.rc, t.zc = t.rc * mc, t.zc * mc
        t.rs, t.zs = t.rs * ms, t.zs * ms
        return t

    @classmethod
    def from_symmetric(cls, t):
        """A torux.torus.TorusState (R cosine, Z sine series) as a general torus."""
        z = np.zeros_like(np.asarray(t.rc))
        return cls(np.array(t.rc, float), z.copy(), z.copy(), np.array(t.zs, float), float(t.iota), float(t.flux),
                   int(t.M), int(t.N), t.nfp, res=getattr(t, "res", np.nan), bn_off=getattr(t, "bn_off", np.nan),
                   phi0=float(getattr(t, "phi0", 0.0)))


def torus_points(t: GeneralTorus, theta, phi, deriv=False):
    """Cartesian points of the general torus at arrays theta, phi (separable evaluation) and,
    with deriv, N = X_theta x X_phi and (R_theta, Z_theta)."""
    theta = np.asarray(theta, float).ravel()
    phi = np.asarray(phi, float).ravel()
    m = np.arange(t.M + 1)
    n = np.arange(-t.N, t.N + 1)
    am = np.outer(theta, m)
    an = np.outer(t.nfp * (phi - t.phi0), n)
    Cm, Sm, Cn, Sn = np.cos(am), np.sin(am), np.cos(an), np.sin(an)
    mf = m[:, None].astype(float)
    nw = (n * t.nfp)[None, :].astype(float)

    def val(ac, as_):
        aC, aS = Cn @ ac.T, Sn @ ac.T
        bC, bS = Cn @ as_.T, Sn @ as_.T
        return np.sum(Cm * (aC - bS) + Sm * (aS + bC), axis=1)
    R = val(t.rc, t.rs)
    Z = val(t.zc, t.zs)
    x = np.stack([R * np.cos(phi), R * np.sin(phi), Z], -1)
    if not deriv:
        return x, R, Z
    # d/dtheta: (a, b) -> (m b, -m a); d/dphi: (a, b) -> (-n nfp b, n nfp a)
    Rt = val(mf * t.rs, -mf * t.rc)
    Zt = val(mf * t.zs, -mf * t.zc)
    Rp = val(-nw * t.rs, nw * t.rc)
    Zp = val(-nw * t.zs, nw * t.zc)
    cp, sp = np.cos(phi), np.sin(phi)
    eR = np.stack([cp, sp, 0 * cp], -1)
    eP = np.stack([-sp, cp, 0 * cp], -1)
    eZ = np.stack([0 * cp, 0 * cp, 1 + 0 * cp], -1)
    Xt = Rt[:, None] * eR + Zt[:, None] * eZ
    Xp = Rp[:, None] * eR + R[:, None] * eP + Zp[:, None] * eZ
    return x, R, Z, np.cross(Xt, Xp), (Rt, Zt)


def phase_row(basis: GeneralBasis, ref: GeneralTorus):
    """Row of the phase condition <X, X_ref,theta> / |X_ref,theta|_rms = 0 in the packed coefficients:
    d/drc = m rs_ref / 2, d/drs = -m rc_ref / 2, d/dzc = m zs_ref / 2, d/dzs = -m zc_ref / 2."""
    r = ref.resized(basis.M, basis.N)
    m = np.arange(basis.M + 1)[:, None].astype(float)
    nrm = math.sqrt(float(np.sum(0.5 * m ** 2 * (r.rc ** 2 + r.rs ** 2 + r.zc ** 2 + r.zs ** 2))))
    nrm = nrm if nrm > 0 else 1.0
    Prc, Prs = 0.5 * m * r.rs / nrm, -0.5 * m * r.rc / nrm
    Pzc, Pzs = 0.5 * m * r.zs / nrm, -0.5 * m * r.zc / nrm
    return jnp.asarray(np.concatenate([Prc.ravel()[basis.ic], Prs.ravel()[basis.is_],
                                       Pzc.ravel()[basis.ic], Pzs.ravel()[basis.is_]]))


# ============================================================================ the Newton system
_CONSTS = {}


class _GConsts:
    """Shape-only constants of the general torus problem at one resolution (shared by all sources)."""

    def __init__(self, nfp, M, N, oversample, phi0):
        self.nfp = nfp
        self.basis = GeneralBasis(nfp, M, N, oversample=oversample, phi0=phi0)
        b = self.basis
        self.check_basis = GeneralBasis(nfp, M, N, ntheta=b.ntheta, nphi=b.nphi, shift=True, phi0=phi0)
        self.modes = {"s": (jnp.asarray(b.modes_s[:, 0], jnp.int32), jnp.asarray(b.modes_s[:, 1], jnp.int32)),
                      "c": (jnp.asarray(b.modes_c[:, 0], jnp.int32), jnp.asarray(b.modes_c[:, 1], jnp.int32))}
        pos_c = {tuple(k): j for j, k in enumerate(map(tuple, b.modes_c))}
        self.cols_c = np.array([pos_c[tuple(k)] for k in map(tuple, b.modes_s)])
        self.ms_ = jnp.asarray(b.modes_s[:, 0].astype(float))
        self.ns_ = jnp.asarray(b.modes_s[:, 1].astype(float))

    def blocks(self, ahat, rk, ck):
        """Galerkin matrix of multiplication by a: test functions of kind rk, basis functions of kind
        ck (c = cos, s = sin of m theta - n psi), from the grid Fourier coefficients ahat."""
        b = self.basis
        mi, ni = self.modes[rk]
        mj, nj = self.modes[ck]
        mi, ni, mj, nj = mi[:, None], ni[:, None], mj[None, :], nj[None, :]
        ip = ((mi + mj) % b.ntheta) * b.nphi + (-(ni + nj)) % b.nphi
        im = ((mi - mj) % b.ntheta) * b.nphi + (-(ni - nj)) % b.nphi
        af = ahat.ravel()
        Hp, Hm = af[ip], af[im]
        if rk == "s" and ck == "c":
            Mx = -(Hp.imag + Hm.imag)
        elif rk == "s" and ck == "s":
            Mx = Hm.real - Hp.real
        elif rk == "c" and ck == "c":
            Mx = Hm.real + Hp.real
        else:
            Mx = Hm.imag - Hp.imag
        if rk == "c":
            Mx = jnp.where(self.basis.zero_c[:, None], 0.5 * Mx, Mx)
        return Mx

    def gblock(self, ahat):
        return jnp.concatenate([jnp.concatenate([self.blocks(ahat, "c", "c"), self.blocks(ahat, "c", "s")], 1),
                                jnp.concatenate([self.blocks(ahat, "s", "c"), self.blocks(ahat, "s", "s")], 1)], 0)


def _grid_quantities(b: GeneralBasis, u, Bx, Ax, Gx):
    rc, rs, zc, zs, iota, lam = b.unpack(u)
    R, Z, Rt, Zt, Rp, Zp = b.geometry(rc, rs, zc, zs)
    _, P = b.points(R, Z)
    g = {k: v.reshape(b.ntheta, b.nphi) for k, v in cylindrical_from(Bx, Ax, Gx, P).items()}
    BR, Bp, BZ = g["BR"], g["Bp"], g["BZ"]
    ib = 1.0 / Bp
    t2 = Rt ** 2 + Zt ** 2
    G = dict(R=R, Z=Z, Rt=Rt, Zt=Zt, Rp=Rp, Zp=Zp, Bp=Bp, iota=iota, lam=lam,
             rc=rc, rs=rs, zc=zc, zs=zs)
    G["wR"], G["wZ"] = -Zt * ib / t2, Rt * ib / t2
    G["FRg"] = Rp + iota * Rt - R * BR * ib - lam * G["wR"]
    G["FZg"] = Zp + iota * Zt - R * BZ * ib - lam * G["wZ"]
    G["flux"] = -2 * jnp.pi * jnp.mean(g["AR"] * Rt + g["AZ"] * Zt)
    G["aRR"] = BR * ib + R * (g["dBR_R"] * Bp - BR * g["dBp_R"]) * ib ** 2
    G["aRZ"] = R * (g["dBR_Z"] * Bp - BR * g["dBp_Z"]) * ib ** 2
    G["aZR"] = BZ * ib + R * (g["dBZ_R"] * Bp - BZ * g["dBp_R"]) * ib ** 2
    G["aZZ"] = R * (g["dBZ_Z"] * Bp - BZ * g["dBp_Z"]) * ib ** 2
    return G


def _residual(b: GeneralBasis, G, u, flux_target, flux_scale, ph_row, iota_t, fixed):
    FR = b.projv(G["FRg"])
    FZ = b.projv(G["FZg"])
    last = (G["iota"] - iota_t) if fixed else (G["flux"] - flux_target) * flux_scale
    phase = ph_row @ u[:2 * b.nco]
    return jnp.concatenate([FR, FZ, jnp.atleast_1d(last), jnp.atleast_1d(phase)]), FR, FZ


def _system_post(c: _GConsts, u, Bx, Ax, Gx, flux_target, flux_scale, ph_row, iota_t, fixed):
    """F, J, flux, res, lam from the field (Cartesian B, A, G) at the grid points of u."""
    b = c.basis
    G = _grid_quantities(b, u, Bx, Ax, Gx)
    F, FR, FZ = _residual(b, G, u, flux_target, flux_scale, ph_row, iota_t, fixed)
    ng = b.ntheta * b.nphi
    hat = lambda a: jnp.fft.fft2(a) / ng
    JRR = -c.gblock(hat(G["aRR"]))
    JRZ = -c.gblock(hat(G["aRZ"]))
    JZR = -c.gblock(hat(G["aZR"]))
    JZZ = -c.gblock(hat(G["aZZ"]))
    # (d_phi + iota d_theta): cos(m th - n ps) -> (n nfp - iota m) sin, sin -> (iota m - n nfp) cos
    d = c.ns_ * c.nfp - G["iota"] * c.ms_
    rows_s = b.nc + np.arange(b.ns)
    JRR = JRR.at[rows_s, c.cols_c].add(d).at[c.cols_c, rows_s].add(-d)
    JZZ = JZZ.at[rows_s, c.cols_c].add(d).at[c.cols_c, rows_s].add(-d)
    Ji = jnp.concatenate([b.projv(G["Rt"]), b.projv(G["Zt"])])
    Jl = -jnp.concatenate([b.projv(G["wR"]), b.projv(G["wZ"])])
    n2 = 2 * b.nco
    if fixed:
        Jlast = jnp.zeros(n2 + 2).at[n2].set(1.0)
    else:
        mz = b.means(G["Bp"] * G["Zt"])
        mr = b.means(G["Bp"] * G["Rt"])
        Jlast = jnp.concatenate([2 * jnp.pi * mz, -2 * jnp.pi * mr, jnp.zeros(2)]) * flux_scale
    Jph = jnp.concatenate([ph_row, jnp.zeros(2)])
    top = jnp.concatenate([JRR, JRZ, Ji[:b.nco, None], Jl[:b.nco, None]], axis=1)
    mid = jnp.concatenate([JZR, JZZ, Ji[b.nco:, None], Jl[b.nco:, None]], axis=1)
    J = jnp.concatenate([top, mid, Jlast[None, :], Jph[None, :]], axis=0)
    res = jnp.maximum(jnp.max(jnp.abs(FR)), jnp.max(jnp.abs(FZ)))
    return F, J, G["flux"], res


@partial(jax.jit, static_argnames=("key",))
def _grid_x_shared(u, key):
    b = _CONSTS[key].basis
    rc, rs, zc, zs, _, _ = b.unpack(u)
    R, Z, *_ = b.geometry(rc, rs, zc, zs)
    return b.points(R, Z)[0]


@partial(jax.jit, static_argnames=("key", "fixed"))
def _system_post_shared(u, Bx, Ax, Gx, flux_target, flux_scale, ph_row, iota_t, key, fixed):
    return _system_post(_CONSTS[key], u, Bx, Ax, Gx, flux_target, flux_scale, ph_row, iota_t, fixed)


def _consts(nfp, M, N, oversample=2.5, phi0=0.0):
    key = ("general", float(nfp), int(M), int(N), float(oversample), float(phi0))
    c = _CONSTS.get(key)
    if c is None:
        c = _CONSTS[key] = _GConsts(nfp, M, N, oversample, phi0)
    return key, c


class GeneralTorusProblem:
    """The general torus problem at resolution (M, N) for one engine (dense Newton, checks)."""

    def __init__(self, engine, nfp, M, N, oversample=2.5, phi0=0.0):
        self.engine = engine
        self.nfp = nfp
        self.phi0 = phi0
        self.key, self.c = _consts(nfp, M, N, oversample, phi0)
        self.basis = self.c.basis
        self.check_basis = self.c.check_basis

    # ------------------------------------------------------------------ pieces
    def pack(self, t: GeneralTorus, iota=None, lam=0.0):
        t = t if (t.M, t.N) == (self.basis.M, self.basis.N) else t.resized(self.basis.M, self.basis.N)
        return self.basis.pack(t.rc, t.rs, t.zc, t.zs, t.iota if iota is None else iota, lam)

    def field_at(self, u, need_grad=True, need_A=True):
        x = _grid_x_shared(jnp.asarray(u), key=self.key)
        return self.engine.evaluate(x, grad=need_grad, A=need_A)

    def system(self, u, flux_target, flux_scale, ph_row, iota_t=0.0, fixed=False):
        o = self.field_at(u)
        return _system_post_shared(u, o["B"], o["A"], o["G"], flux_target, flux_scale, ph_row, iota_t,
                                   key=self.key, fixed=bool(fixed))

    # ------------------------------------------------------------------ checks
    def check(self, t: GeneralTorus):
        """Off-grid residual (m) and max |B.n|/|B| on the half-shifted grid (no lambda: E itself)."""
        cb = self.check_basis
        t = t if (t.M, t.N) == (cb.M, cb.N) else t.resized(cb.M, cb.N)
        R, Z, Rt, Zt, Rp, Zp = cb.geometry(*(jnp.asarray(a) for a in t.arrays()))
        g = field_on_grid(self.engine, cb, R, Z, need_grad=False, need_A=False)
        BR, Bp, BZ = g["BR"], g["Bp"], g["BZ"]
        FR = Rp + t.iota * Rt - R * BR / Bp
        FZ = Zp + t.iota * Zt - R * BZ / Bp
        nR, nP, nZ = -R * Zt, Zt * Rp - Rt * Zp, R * Rt
        Bn = BR * nR + Bp * nP + BZ * nZ
        bn = jnp.abs(Bn) / (jnp.sqrt(BR ** 2 + Bp ** 2 + BZ ** 2) * jnp.sqrt(nR ** 2 + nP ** 2 + nZ ** 2))
        return float(jnp.maximum(jnp.max(jnp.abs(FR)), jnp.max(jnp.abs(FZ)))), float(jnp.max(bn))

    def quadrature_check(self, t: GeneralTorus):
        """Largest a posteriori error estimate (N/2 rule) of the engine's quadrature on the torus."""
        cb = self.check_basis
        t = t if (t.M, t.N) == (cb.M, cb.N) else t.resized(cb.M, cb.N)
        R, Z, *_ = cb.geometry(*(jnp.asarray(a) for a in t.arrays()))
        x, _ = cb.points(R, Z)
        o = self.engine.evaluate(x, grad=False, A=False, err=True)
        return float(jnp.max(o["err_half"]))

    def flux_of(self, t: GeneralTorus):
        b = self.basis
        t = t if (t.M, t.N) == (b.M, b.N) else t.resized(b.M, b.N)
        R, Z, Rt, Zt, Rp, Zp = b.geometry(*(jnp.asarray(a) for a in t.arrays()))
        g = field_on_grid(self.engine, b, R, Z, need_grad=False, need_A=True)
        return float(-2 * jnp.pi * jnp.mean(g["AR"] * Rt + g["AZ"] * Zt))

    # ------------------------------------------------------------------ Newton
    def newton(self, state: GeneralTorus, flux_target, tol=1e-13, maxit=20, verbose=False, deadline=None,
               max_step_frac=0.5, fixed_iota=None, filt=0.0, ref: GeneralTorus = None):
        """Damped Newton at the requested flux (or at fixed_iota: the flux equation replaced by
        iota = fixed_iota).  ref: the phase reference (default: the start).  filt > 0: Tikhonov-
        filtered SVD steps (lam = filt * s_max), as torux.torus.TorusProblem.newton."""
        b = self.basis
        st = state.resized(b.M, b.N)
        fixed = fixed_iota is not None
        u = b.pack(st.rc, st.rs, st.zc, st.zs, st.iota if not fixed else fixed_iota, 0.0)
        ph = phase_row(b, ref if ref is not None else st)
        R00 = float(st.rc[0, b.N])
        ft = flux_target if not fixed else (st.flux if st.flux else 1.0)
        flux_scale = R00 / max(abs(ft), 1e-300)
        iota_t = float(fixed_iota) if fixed else 0.0
        hist = []
        best = None
        n2 = 2 * b.nco
        for it in range(maxit):
            if deadline is not None and time.time() > deadline:
                break
            F, J, flux, res = self.system(u, ft, flux_scale, ph, iota_t, fixed)
            nF = float(jnp.max(jnp.abs(F)))
            hist.append((nF, float(res), float(u[-2]), float(flux), float(u[-1])))
            if verbose:
                print(f"   gen it {it:2d} |F| {nF:.3e} res {float(res):.3e} iota {float(u[-2]):.15f} "
                      f"flux {float(flux):.12e} lam {float(u[-1]):.2e}", flush=True)
            if not np.isfinite(nF):
                break
            if best is None or nF < best[0]:
                best = (nF, u, float(flux))
            if nF < tol * max(1.0, R00):
                break
            if filt > 0:
                Uj, sj, Vtj = jnp.linalg.svd(J, full_matrices=False)
                lam_t = filt * sj[0]
                du = Vtj.T @ ((sj / (sj ** 2 + lam_t ** 2)) * (Uj.T @ -F))
            else:
                du = jnp.linalg.solve(J, -F)
            if not bool(jnp.all(jnp.isfinite(du))):
                break
            step = float(jnp.max(jnp.abs(du[:n2])))
            shape = float(jnp.max(jnp.abs(u[1:n2])))
            lam = 1.0 if step <= max_step_frac * shape else max_step_frac * shape / step
            u = u + lam * du
        nF, u, flux_b = best if best is not None else (np.inf, u, np.nan)
        rc, rs, zc, zs, iota, lamb = b.unpack(u)
        fl = float(flux_target) if not fixed else flux_b
        out = GeneralTorus(np.asarray(rc), np.asarray(rs), np.asarray(zc), np.asarray(zs), float(iota), fl, b.M, b.N,
                           self.nfp, res=float(nF), its=len(hist), converged=bool(nF < 1e3 * tol * max(1.0, R00)),
                           phi0=self.phi0, lam=float(lamb))
        out.info["hist"] = hist
        out.tail = max(out.tails())
        return out


# ============================================================================ Newton-Krylov
def _fft_freqs(nt, nph):
    m = np.fft.fftfreq(nt, 1.0 / nt)
    k = np.fft.fftfreq(nph, 1.0 / nph)
    if nt % 2 == 0:
        m[nt // 2] = 0.0
    if nph % 2 == 0:
        k[nph // 2] = 0.0
    return m[:, None], k[None, :]


class _NK:
    """Matrix-free pieces of the general problem (inside the shared jitted step).

    Preconditioner (approximate reducibility, as torux.torus.KrylovTorus): in the frame
    [t, nu], t = X_theta, nu = (-Z_theta, R_theta)/(B_phi |t|^2), the unfolded linearised operator
    is, at an invariant torus,
        (d + iota d_th) eta1 + T eta2 + d iota = s1,    (d + iota d_th) eta2 - d lambda = s2,
    with s = F^-1 r: d lambda = -<s2>; <eta2> = -d flux / 2 pi; d iota = <s1 - T eta2>; <eta1>
    from the phase condition."""

    def __init__(self, c: _GConsts, reg):
        self.c = c
        self.b = c.basis
        self.reg = reg
        m, k = _fft_freqs(self.b.ntheta, self.b.nphi)
        self.m, self.k = jnp.asarray(m), jnp.asarray(k)

    def _L(self, f, iota):
        lam = 1j * (iota * self.m + self.k * self.b.nfp)
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(f) * lam))

    def _Linv(self, f, iota):
        lam = 1j * (iota * self.m + self.k * self.b.nfp)
        a2 = jnp.abs(lam) ** 2
        inv = jnp.where(a2 > 0, jnp.conj(lam) / (a2 + self.reg ** 2), 0.0)
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(f) * inv))

    def _split(self, v):
        b = self.b
        nc, ns, nco = b.nc, b.ns, b.nco
        shape = (b.M + 1, 2 * b.N + 1)
        size = shape[0] * shape[1]
        put = lambda x, idx: jnp.zeros(size).at[idx].set(x, unique_indices=True).reshape(shape)
        Rc, Rs = put(v[:nc], b.ic), put(v[nc:nco], b.is_)
        Zc, Zs = put(v[nco:nco + nc], b.ic), put(v[nco + nc:2 * nco], b.is_)
        return Rc, Rs, Zc, Zs, v[-2], v[-1]

    def Jv(self, G, v, flux_scale, ph_row, fixed):
        b = self.b
        nfp = b.nfp
        drc, drs, dzc, dzs, diota, dlam = self._split(v)
        dR = b.synth(drc, drs)
        dZ = b.synth(dzc, dzs)
        iota = G["iota"]
        dd = b.nn * nfp - iota * b.mm                      # cos -> dd sin, sin -> -dd cos
        LR = b.synth(-dd * drs, dd * drc)
        LZ = b.synth(-dd * dzs, dd * dzc)
        jR = LR + diota * G["Rt"] - G["aRR"] * dR - G["aRZ"] * dZ - dlam * G["wR"]
        jZ = LZ + diota * G["Zt"] - G["aZR"] * dR - G["aZZ"] * dZ - dlam * G["wZ"]
        if fixed:
            last = diota
        else:
            last = 2 * jnp.pi * jnp.mean(G["Bp"] * (G["Zt"] * dR - G["Rt"] * dZ)) * flux_scale
        return jnp.concatenate([b.projv(jR), b.projv(jZ), jnp.atleast_1d(last),
                                jnp.atleast_1d(ph_row @ v[:2 * b.nco])])

    def precond(self, G, r, flux_scale, ph_row, fixed):
        b = self.b
        iota = G["iota"]
        rRc, rRs, rZc, rZs, rlast, rph = self._split(r)
        rR = b.synth(rRc, rRs)
        rZ = b.synth(rZc, rZs)
        tR, tZ, Bp = G["Rt"], G["Zt"], G["Bp"]
        t2 = tR ** 2 + tZ ** 2
        nR, nZ = -tZ / (Bp * t2), tR / (Bp * t2)
        s1 = Bp * (nZ * rR - nR * rZ)
        s2 = Bp * (-tZ * rR + tR * rZ)
        wR = self._L(nR, iota) - G["aRR"] * nR - G["aRZ"] * nZ
        wZ = self._L(nZ, iota) - G["aZR"] * nR - G["aZZ"] * nZ
        T = Bp * (nZ * wR - nR * wZ)
        dlam = -jnp.mean(s2)
        e20 = self._Linv(s2, iota)
        if fixed:
            diota = rlast
            Tm = jnp.mean(T)
            c2 = (jnp.mean(s1) - diota - jnp.mean(T * e20)) / jnp.where(jnp.abs(Tm) > 0, Tm, 1.0)
            eta2 = e20 + c2
            eta1 = self._Linv(s1 - T * eta2 - diota, iota)
        else:
            eta2 = e20 - (rlast / flux_scale) / (2 * jnp.pi)
            q1 = s1 - T * eta2
            diota = jnp.mean(q1)
            eta1 = self._Linv(q1, iota)
        xR = eta1 * tR + eta2 * nR
        xZ = eta1 * tZ + eta2 * nZ
        v0 = jnp.concatenate([b.projv(xR), b.projv(xZ)])
        # the tangent direction t = X_theta in coefficients: (a, b) -> (m b, -m a)
        mm = b.mm
        tvec = jnp.concatenate([(mm * G["rs"]).ravel()[b.ic], (-mm * G["rc"]).ravel()[b.is_],
                                (mm * G["zs"]).ravel()[b.ic], (-mm * G["zc"]).ravel()[b.is_]])
        pt = ph_row @ tvec
        c1 = (rph - ph_row @ v0) / jnp.where(jnp.abs(pt) > 0, pt, 1.0)
        v = v0 + c1 * tvec
        return jnp.concatenate([v, jnp.atleast_1d(diota), jnp.atleast_1d(dlam)])

    def step(self, u, Bx, Ax, Gx, flux_target, flux_scale, ph_row, iota_t, gtol, restart, maxiter, fixed):
        G = _grid_quantities(self.b, u, Bx, Ax, Gx)
        F, _, _ = _residual(self.b, G, u, flux_target, flux_scale, ph_row, iota_t, fixed)
        A = lambda v: self.Jv(G, v, flux_scale, ph_row, fixed)
        Mi = lambda r: self.precond(G, r, flux_scale, ph_row, fixed)
        du, info = jax.scipy.sparse.linalg.gmres(A, -F, M=Mi, tol=gtol, atol=0.0, restart=restart,
                                                 maxiter=maxiter, solve_method="batched")
        res_lin = jnp.max(jnp.abs(A(du) + F))
        return du, F, G["flux"], res_lin


@partial(jax.jit, static_argnames=("key", "restart", "maxiter", "fixed", "reg"))
def _nk_step_shared(u, Bx, Ax, Gx, flux_target, flux_scale, ph_row, iota_t, gtol, key, restart, maxiter, fixed, reg):
    return _NK(_CONSTS[key], reg).step(u, Bx, Ax, Gx, flux_target, flux_scale, ph_row, iota_t, gtol, restart,
                                       maxiter, fixed)


class GeneralKrylov:
    """Matrix-free Newton-Krylov (GMRES, approximate-reducibility right preconditioner) for a
    GeneralTorusProblem: J v costs two syntheses and two projections on the grid."""

    def __init__(self, prob: GeneralTorusProblem, reg=1e-12):
        self.prob = prob
        self.reg = reg

    def newton(self, state: GeneralTorus, flux_target, tol=1e-13, maxit=20, verbose=False, gtol=1e-8, restart=60,
               maxiter=3, max_step_frac=0.5, deadline=None, fixed_iota=None, ref: GeneralTorus = None):
        prob = self.prob
        b = prob.basis
        st = state.resized(b.M, b.N)
        fixed = fixed_iota is not None
        u = b.pack(st.rc, st.rs, st.zc, st.zs, st.iota if not fixed else fixed_iota, 0.0)
        ph = phase_row(b, ref if ref is not None else st)
        R00 = float(st.rc[0, b.N])
        ft = flux_target if not fixed else (st.flux if st.flux else 1.0)
        flux_scale = R00 / max(abs(ft), 1e-300)
        iota_t = float(fixed_iota) if fixed else 0.0
        best = None
        hist = []
        n2 = 2 * b.nco
        for it in range(maxit):
            if deadline is not None and time.time() > deadline:
                break
            o = prob.field_at(u)
            du, F, flux, rl = _nk_step_shared(u, o["B"], o["A"], o["G"], ft, flux_scale, ph, iota_t, gtol,
                                              key=prob.key, restart=restart, maxiter=maxiter, fixed=fixed,
                                              reg=self.reg)
            nF = float(jnp.max(jnp.abs(F)))
            hist.append((nF, float(rl), float(u[-2]), float(flux), float(u[-1])))
            if verbose:
                print(f"   gen nk it {it:2d} |F| {nF:.3e} lin {float(rl):.2e} iota {float(u[-2]):.15f}", flush=True)
            if not np.isfinite(nF):
                break
            if best is None or nF < best[0]:
                best = (nF, u, float(flux))
            if nF < tol * max(1.0, R00):
                break
            step = float(jnp.max(jnp.abs(du[:n2])))
            shape = float(jnp.max(jnp.abs(u[1:n2])))
            lam = 1.0 if step <= max_step_frac * shape else max_step_frac * shape / step
            u = u + lam * du
        nF, u, flux_b = best if best is not None else (np.inf, u, np.nan)
        rc, rs, zc, zs, iota, lamb = b.unpack(u)
        fl = float(flux_target) if not fixed else flux_b
        out = GeneralTorus(np.asarray(rc), np.asarray(rs), np.asarray(zc), np.asarray(zs), float(iota), fl, b.M, b.N,
                           prob.nfp, res=float(nF), its=len(hist), converged=bool(nF < 1e3 * tol * max(1.0, R00)),
                           phi0=prob.phi0, lam=float(lamb))
        out.info["hist"] = hist
        out.info["solver"] = "newton-krylov"
        out.tail = max(out.tails())
        return out


# ============================================================================ resonance guard
def bdotN_harmonics(engine, torus: GeneralTorus, harmonics, extra=8):
    """|[B.N]_(m,n)| at the harmonics (phase m theta - n psi; cosine and sine parts) and the scale
    mean |B||N| (N = X_theta x X_phi), on a general grid of the torus padded by ``extra`` modes."""
    Mq = max([m for m, _ in harmonics] + [torus.M]) + extra
    Nq = max([abs(n) for _, n in harmonics] + [torus.N]) + extra
    b = GeneralBasis(torus.nfp, Mq, Nq, phi0=torus.phi0)
    st = torus.resized(Mq, Nq)
    R, Z, Rt, Zt, Rp, Zp = b.geometry(*(jnp.asarray(a) for a in st.arrays()))
    g = field_on_grid(engine, b, R, Z, need_grad=False, need_A=False)
    nR, nP, nZ = -Zt * R, Zt * Rp - Rt * Zp, Rt * R
    BN = g["BR"] * nR + g["Bp"] * nP + g["BZ"] * nZ
    scale = float(jnp.mean(jnp.sqrt(g["BR"] ** 2 + g["Bp"] ** 2 + g["BZ"] ** 2) * jnp.sqrt(nR ** 2 + nP ** 2 + nZ ** 2)))
    c, s = (np.asarray(a) for a in b.proj(BN))
    out = []
    for m, n in harmonics:
        j = n + Nq
        out.append(float(np.hypot(c[m, j], s[m, j])))
    return out, scale


def resonance_guard(engine, torus: GeneralTorus, shear, mmax=None, bn=None):
    """The resonance guard of torux.guard on the general basis: at every near-resonant harmonic (m, n),
    m <= 2M + 10, h = m |iota'| eps / d^2 <= 1/16 (eps = |[B.N]_mn|, iota' = d iota / d Phi')."""
    from .guard import near_resonances, H_MAX, PREFILTER
    mmax = mmax or 2 * torus.M + 10
    bn = torus.bn_off if bn is None else bn
    out = dict(ok=True, h=0.0, x_over_w=np.inf, worst=None, checked=[], shear=shear)
    if not np.isfinite(bn):
        return dict(out, ok=False, h=np.inf, reason="bn not finite")
    thr = np.sqrt(2.0 * max(bn, 1e-300) / PREFILTER)
    harm = near_resonances(torus.iota, torus.nfp, mmax, thr)
    if not harm:
        return out
    eps, scale = bdotN_harmonics(engine, torus, [(m, n) for m, n, _ in harm])
    out["scale"] = scale
    for (m, n, d), e in zip(harm, eps):
        h = m * abs(shear) * e / d ** 2 if np.isfinite(shear) else np.inf
        out["checked"].append(dict(m=m, n=n, d=d, eps=e, eps_rel=e / scale, h=h))
        if h > out["h"]:
            out.update(h=float(h), worst=(m, n, d, e), x_over_w=float(0.5 / np.sqrt(h)) if h > 0 else np.inf)
    out["ok"] = bool(out["h"] <= H_MAX)
    return out
