"""Invariant tori (flux surfaces) of a vacuum field: the parameterisation method.

A flux surface with rotational transform iota is a torus X(theta, phi) = (R, Z)(theta, phi)
in cylindrical coordinates on which field lines are straight lines of slope iota:

    (d_phi + iota d_theta) R = V_R(R, Z, phi) = R B_R / B_phi
    (d_phi + iota d_theta) Z = V_Z(R, Z, phi) = R B_Z / B_phi

R, Z are double Fourier series in (theta, psi = nfp phi).  For stellarator-symmetric fields
and tori R is a cosine and Z a sine series in m theta - n psi; the R-residual is then odd
and is tested with sines, the Z-residual even and tested with cosines (Galerkin on an
oversampled grid, exact for the grid representation).  The unknowns are the coefficients
and iota, the extra equation fixes the enclosed toroidal flux

    Phi = - int_0^{2 pi} A . X_theta d theta      (any phi; averaged over the phi grid)

(positive for B_phi > 0 with theta counter-clockwise in the (R, Z) plane).  The symmetry
removes the theta-translation gauge.

The Newton system is assembled exactly: products a(theta, phi) * basis function project
onto the test functions through the grid Fourier coefficients of a at sum and difference
modes (Toeplitz-plus-Hankel), O(n^2) gathers after one FFT per coefficient function.
The field (B, grad B, A) is evaluated once per iteration on half the grid (stellarator
images: B_R odd, B_phi, B_Z even; A likewise; gradients by the chain rule).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from . import budget as _budget


def _so2():
    from . import fast_family
    return fast_family

jax.config.update("jax_enable_x64", True)


def _fft_workers():
    """Threads for the host FFTs (scipy.fft): TORUX_FFT_WORKERS, else TORUX_THREADS, else OMP_NUM_THREADS."""
    for v in ("TORUX_FFT_WORKERS", "TORUX_THREADS", "OMP_NUM_THREADS"):
        try:
            n = int(os.environ.get(v) or 0)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return 1


_FFT_W = _fft_workers()


def _sfft():
    import scipy.fft as sf
    return sf


def _xp(*arrs):
    """numpy where every array argument is a host numpy array (the CPU path, no JIT: no tracing per
    process), jax.numpy otherwise (traced inside jit, device arrays on GPUs)."""
    for a in arrs:
        if a is not None and hasattr(a, "shape") and not isinstance(a, (np.ndarray, np.generic)):
            return jnp
    return np


def _scatter(xp, size, idx, vals):
    """zeros(size) with vals at the (unique) indices idx."""
    if xp is np:
        out = np.zeros(size, dtype=np.result_type(vals, np.float64))
        out[idx] = vals
        return out
    return jnp.zeros(size).at[idx].set(vals, unique_indices=True)


def _add_at(xp, M, rows, cols, vals):
    """M with vals added at the (unique) positions (rows, cols)."""
    if xp is np:
        M = np.array(M, copy=True)
        M[rows, cols] += vals
        return M
    return M.at[rows, cols].add(vals)


def fft_size(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = None
    for a in range(0, 16):
        for b in range(0, 10):
            for c in range(0, 7):
                v = 2 ** a * 3 ** b * 5 ** c
                if v >= n and (best is None or v < best):
                    best = v
    return best


# ============================================================================ basis
class Basis:
    """Stellarator-symmetric double Fourier basis on a one-field-period grid.

    Coefficient arrays rc, zs have shape (M+1, 2N+1), column j <-> n = j - N.
    Used entries: rc: m = 0 with n >= 0, and m >= 1 all n; zs: m = 0 with n >= 1, m >= 1.
    """

    def __init__(self, nfp, M, N, ntheta=None, nphi=None, oversample=2.5, shift=False, phi0=0.0):
        # nfp may be fractional: an island torus around a closed line of period q field periods
        # has period 2 pi q / nfp in phi, i.e. nfp_eff = nfp / q; phi0 shifts the phi origin to
        # the symmetry plane the torus is symmetric about.
        self.nfp, self.M, self.N = (int(nfp) if float(nfp).is_integer() else float(nfp)), int(M), int(N)
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
        self.psi = ps0 + 2 * np.pi * np.arange(self.nphi) / self.nphi       # psi = nfp phi
        self.phi = self.phi0 + self.psi / self.nfp
        mt = np.outer(self.theta, m)
        np_ = np.outer(self.psi, n)
        self.Ct, self.St = jnp.asarray(np.cos(mt)), jnp.asarray(np.sin(mt))       # (nt, M+1)
        self.Cp, self.Sp = jnp.asarray(np.cos(np_)), jnp.asarray(np.sin(np_))     # (np, 2N+1)
        mc = np.ones((M + 1, 2 * N + 1), bool)
        mc[0, :N] = False                       # rc: m = 0, n < 0 unused
        ms = mc.copy()
        ms[0, N] = False                        # zs: (0, 0) unused
        self.mask_c, self.mask_s = mc, ms
        self.ic = np.flatnonzero(mc.ravel())
        self.is_ = np.flatnonzero(ms.ravel())
        self.nc, self.ns = len(self.ic), len(self.is_)
        self.mm = jnp.asarray(np.repeat(m[:, None], 2 * N + 1, 1).astype(float))
        self.nn = jnp.asarray(np.repeat(n[None, :], M + 1, 0).astype(float))
        # modes as integer lists for the gathers
        self.modes_c = np.stack(np.unravel_index(self.ic, mc.shape), -1) - np.array([0, N])
        self.modes_s = np.stack(np.unravel_index(self.is_, ms.shape), -1) - np.array([0, N])
        # canonical half of the grid under (j, k) -> (-j, -k)
        J, Kk = np.meshgrid(np.arange(self.ntheta), np.arange(self.nphi), indexing="ij")
        if not shift:
            Jp, Kp = (-J) % self.ntheta, (-Kk) % self.nphi
        else:
            # shifted grid: theta_j = (j + 1/2) h -> -theta_j = theta_{-j-1}
            Jp, Kp = (-J - 1) % self.ntheta, (-Kk - 1) % self.nphi
        lin, linp = J * self.nphi + Kk, Jp * self.nphi + Kp
        canon = lin <= linp
        self.canon = jnp.asarray(np.flatnonzero(canon.ravel()))
        # for every grid point: index into the canonical list and the parity flag
        pos = -np.ones(lin.size, int)
        pos[np.flatnonzero(canon.ravel())] = np.arange(canon.sum())
        partner = linp.ravel()
        src = np.where(canon.ravel(), pos[lin.ravel()], pos[partner])
        self.fill_idx = jnp.asarray(src)
        self.fill_sign = jnp.asarray(np.where(canon.ravel(), 1.0, -1.0))   # -1 where the image is used

    # ------------------------------------------------------------------ packing
    def pack(self, rc, zs, iota):
        xp = _xp(self.Ct)
        return xp.concatenate([xp.asarray(rc).ravel()[self.ic], xp.asarray(zs).ravel()[self.is_],
                               xp.atleast_1d(xp.asarray(iota, dtype=float))])

    def unpack(self, u):
        xp = _xp(u)
        shape = (self.M + 1, 2 * self.N + 1)
        rc = _scatter(xp, shape[0] * shape[1], self.ic, u[:self.nc]).reshape(shape)
        zs = _scatter(xp, shape[0] * shape[1], self.is_, u[self.nc:self.nc + self.ns]).reshape(shape)
        return rc, zs, u[-1]

    # ------------------------------------------------------------------ host FFTs
    def _host(self, a):
        """True on the host path (numpy basis and argument): synthesis and projection by real FFTs,
        O(g log g) per call where the dense products cost O(g (M + N)) and multithreaded BLAS calls on
        small matrices stall on a loaded machine.  Same sums to rounding."""
        return isinstance(self.Ct, np.ndarray) and isinstance(a, np.ndarray)

    def _fw(self):
        """Mode (m, n) at row m, column (-n) mod nphi of the half spectrum (rfft along theta); the
        shifted grid's phases e^{i (m th0 - n ps0)}; the synthesis weights undo irfft's doubling of the
        rows m >= 1."""
        w = self.__dict__.get("_fw_cache")
        if w is None:
            nt, nps = self.ntheta, self.nphi
            m = np.arange(self.M + 1)[:, None]
            n = np.arange(-self.N, self.N + 1)[None, :]
            th0 = np.pi / nt if self.shift else 0.0
            ps0 = np.pi / nps if self.shift else 0.0
            ph = np.exp(1j * (m * th0 - n * ps0))
            q = (-np.arange(-self.N, self.N + 1)) % nps
            syn = ph * np.where(m == 0, 1.0, 0.5) * (nt * nps)
            w = self._fw_cache = (q, syn, -1j * syn, np.conj(ph) * (2.0 / (nt * nps)))
        return w

    def _synth_np(self, a, sine):
        q, syn, syn_s, _ = self._fw()
        C = np.zeros(a.shape[:-2] + (self.ntheta // 2 + 1, self.nphi), complex)
        C[..., :self.M + 1, q] = a * (syn_s if sine else syn)
        return _sfft().irfftn(C, s=(self.nphi, self.ntheta), axes=(-1, -2), workers=_FFT_W)

    def _proj_np(self, f, sine):
        q, _, _, pro = self._fw()
        S = _sfft().rfftn(f, axes=(-1, -2), workers=_FFT_W)[..., :self.M + 1, :][..., q] * pro
        if sine:
            return -S.imag
        P = np.array(S.real)
        P[..., 0, self.N] *= 0.5
        return P

    # ------------------------------------------------------------------ synthesis
    def synth_c(self, a):
        """sum a_mn cos(m theta - n psi) on the grid."""
        if self._host(a):
            return self._synth_np(a, False)
        return self.Ct @ a @ self.Cp.T + self.St @ a @ self.Sp.T

    def synth_s(self, a):
        """sum a_mn sin(m theta - n psi)."""
        if self._host(a):
            return self._synth_np(a, True)
        return self.St @ a @ self.Cp.T - self.Ct @ a @ self.Sp.T

    def geometry(self, rc, zs):
        nfp = self.nfp
        R = self.synth_c(rc)
        Z = self.synth_s(zs)
        Rt = -self.synth_s(self.mm * rc)
        Zt = self.synth_c(self.mm * zs)
        Rp = nfp * self.synth_s(self.nn * rc)
        Zp = -nfp * self.synth_c(self.nn * zs)
        return R, Z, Rt, Zt, Rp, Zp

    # ------------------------------------------------------------------ projection
    def proj_c(self, f):
        """Galerkin projection on cos(m theta - n psi): 2 <f, cos> (<f, 1> for (0,0))."""
        if self._host(f):
            return self._proj_np(f, False)
        g = self.ntheta * self.nphi
        P = (self.Ct.T @ f @ self.Cp + self.St.T @ f @ self.Sp) * (2.0 / g)
        if _xp(P) is np:
            P = np.array(P, copy=True)
            P[0, self.N] *= 0.5
            return P
        return P.at[0, self.N].multiply(0.5)

    def proj_s(self, f):
        if self._host(f):
            return self._proj_np(f, True)
        g = self.ntheta * self.nphi
        return (self.St.T @ f @ self.Cp - self.Ct.T @ f @ self.Sp) * (2.0 / g)

    def mean(self, f):
        return _xp(f).mean(f)


def coefficients_from_grid(basis: Basis, R, Z):
    """Fourier coefficients (rc, zs) of grid values R, Z (stellarator-symmetric part)."""
    rc = basis.proj_c(jnp.asarray(R))
    zs = basis.proj_s(jnp.asarray(Z))
    return rc * basis.mask_c, zs * basis.mask_s


# ============================================================================ field on grid
def cylindrical_from(B, A, G, phi, need_grad=True, need_A=True):
    """Cylindrical components of Cartesian B, A and dB_c/dR, dB_c/dZ at points of toroidal angle phi."""
    xp = _xp(B, phi)
    c, s = xp.cos(phi), xp.sin(phi)
    eR = xp.stack([c, s, xp.zeros_like(c)], -1)
    eP = xp.stack([-s, c, xp.zeros_like(c)], -1)
    out = dict(BR=xp.sum(B * eR, -1), Bp=xp.sum(B * eP, -1), BZ=B[:, 2])
    if need_A:
        out["AR"] = xp.sum(A * eR, -1)
        out["Ap"] = xp.sum(A * eP, -1)
        out["AZ"] = A[:, 2]
    if need_grad:
        GR = xp.einsum("tab,tb->ta", G, eR)        # dB/dR (Cartesian components)
        GZ = G[:, :, 2]
        out["dBR_R"], out["dBp_R"], out["dBZ_R"] = xp.sum(GR * eR, -1), xp.sum(GR * eP, -1), GR[:, 2]
        out["dBR_Z"], out["dBp_Z"], out["dBZ_Z"] = xp.sum(GZ * eR, -1), xp.sum(GZ * eP, -1), GZ[:, 2]
    return out


def cylindrical_field(engine, R, Z, phi, need_grad=True, need_A=True):
    """B, A in cylindrical components and dB_c/dR, dB_c/dZ at points (R, phi, Z) (flat arrays)."""
    c, s = jnp.cos(phi), jnp.sin(phi)
    x = jnp.stack([R * c, R * s, Z], -1)
    o = engine.evaluate(x, grad=need_grad, A=need_A)
    return cylindrical_from(o["B"], o.get("A"), o.get("G"), phi, need_grad, need_A)


# parity of each cylindrical quantity under the stellarator image (theta, phi) -> (-theta, -phi)
PARITY = dict(BR=-1.0, Bp=1.0, BZ=1.0, AR=-1.0, Ap=1.0, AZ=1.0,
              dBR_R=-1.0, dBp_R=1.0, dBZ_R=1.0, dBR_Z=1.0, dBp_Z=-1.0, dBZ_Z=-1.0)


def grid_points(basis: Basis, R, Z, symmetric=True):
    """Cartesian points (T, 3) of the basis grid (only the canonical half under stellarator
    symmetry when ``symmetric``) and their toroidal angles."""
    xp = _xp(R)
    nt, nph = basis.ntheta, basis.nphi
    P = xp.broadcast_to(xp.asarray(basis.phi)[None, :], (nt, nph)).ravel()
    Rf, Zf = R.ravel(), Z.ravel()
    if symmetric:
        idx = basis.canon
        Rf, Zf, P = Rf[idx], Zf[idx], P[idx]
    return xp.stack([Rf * xp.cos(P), Rf * xp.sin(P), Zf], -1), P


def fill_grid(basis: Basis, cyl, symmetric=True):
    """Grid arrays (ntheta, nphi) from cylindrical quantities at the points of grid_points."""
    nt, nph = basis.ntheta, basis.nphi
    if symmetric:
        out = {}
        for k, v in cyl.items():
            full = v[basis.fill_idx]
            sign = _xp(v).where(basis.fill_sign > 0, 1.0, PARITY[k])
            out[k] = (full * sign).reshape(nt, nph)
        return out
    return {k: v.reshape(nt, nph) for k, v in cyl.items()}


def field_from_points(basis: Basis, B, A, G, symmetric=True, need_grad=True, need_A=True):
    """Field quantities on the grid from Cartesian B, A, G at the points of grid_points."""
    xp = _xp(B)
    nt, nph = basis.ntheta, basis.nphi
    P = xp.broadcast_to(xp.asarray(basis.phi)[None, :], (nt, nph)).ravel()
    if symmetric:
        P = P[basis.canon]
    return fill_grid(basis, cylindrical_from(B, A, G, P, need_grad, need_A), symmetric)


def field_on_grid(engine, basis: Basis, R, Z, need_grad=True, need_A=True, symmetric=True):
    """Field quantities on the basis grid (arrays (ntheta, nphi)), evaluating only the
    canonical half under stellarator symmetry when ``symmetric``."""
    x, P = grid_points(basis, R, Z, symmetric)
    o = engine.evaluate(x, grad=need_grad, A=need_A)
    return fill_grid(basis, cylindrical_from(o["B"], o.get("A"), o.get("G"), P, need_grad, need_A), symmetric)


# ============================================================================ the torus problem
@dataclass
class TorusState:
    rc: np.ndarray
    zs: np.ndarray
    iota: float
    flux: float
    M: int
    N: int
    nfp: int
    res: float = np.nan              # max |projected residual| (m)
    bn_off: float = np.nan           # max |B.n|/|B| on the half-shifted grid
    res_off: float = np.nan
    its: int = 0
    converged: bool = False
    tail: float = np.nan
    info: dict = field(default_factory=dict)
    phi0: float = 0.0                # phi origin (symmetry plane) of the angle psi = nfp (phi - phi0)

    def resized(self, M, N):
        rc = np.zeros((M + 1, 2 * N + 1)); zs = np.zeros((M + 1, 2 * N + 1))
        m = min(M, self.M); n = min(N, self.N)
        rc[:m + 1, N - n:N + n + 1] = self.rc[:m + 1, self.N - n:self.N + n + 1]
        zs[:m + 1, N - n:N + n + 1] = self.zs[:m + 1, self.N - n:self.N + n + 1]
        return TorusState(rc, zs, self.iota, self.flux, M, N, self.nfp, phi0=self.phi0)

    def tails(self):
        """Relative amplitude of the last two poloidal and toroidal mode rows (truncation measure)."""
        a = np.sqrt(self.rc ** 2 + self.zs ** 2)
        scale = max(np.max(a[1:, :]), 1e-300)
        tm = np.max(a[max(self.M - 1, 1):, :]) / scale if self.M >= 2 else 0.0
        tn = (max(np.max(a[:, :2]), np.max(a[:, -2:])) / scale) if self.N >= 2 else 0.0
        return float(tm), float(tn)

    def geometry_grid(self, theta, phi):
        """R, Z on the tensor grid theta (nt,) x phi (np,) (separable synthesis: cheap at any M, N)."""
        m = np.arange(self.M + 1)
        n = np.arange(-self.N, self.N + 1)
        at = np.outer(np.asarray(theta, float), m)
        ap = np.outer(self.nfp * (np.asarray(phi, float) - self.phi0), n)
        Ct, St, Cp, Sp = np.cos(at), np.sin(at), np.cos(ap), np.sin(ap)
        # cos(m th - n ps) = cos cos + sin sin;  sin(m th - n ps) = sin cos - cos sin
        R = Ct @ self.rc @ Cp.T + St @ self.rc @ Sp.T
        Z = St @ self.zs @ Cp.T - Ct @ self.zs @ Sp.T
        return R, Z

    def geometry_on(self, theta, phi):
        """R, Z at arbitrary (theta, phi) arrays."""
        m = np.arange(self.M + 1)[:, None]
        n = np.arange(-self.N, self.N + 1)[None, :]
        ang = m * np.asarray(theta)[..., None, None] - n * self.nfp * (np.asarray(phi)[..., None, None] - self.phi0)
        R = np.sum(self.rc * np.cos(ang), axis=(-2, -1))
        Z = np.sum(self.zs * np.sin(ang), axis=(-2, -1))
        return R, Z


_CONSTS = {}
_ARRS = {}
_BASIS_ARRAYS = ("Ct", "St", "Cp", "Sp", "mm", "nn", "canon", "fill_idx", "fill_sign")


def _arrays(key):
    """The large arrays of the constants at `key`, passed to the shared jitted functions as arguments:
    as closure constants XLA constant-folds expressions of them at compile time (GMRES applies the
    operator to its zero start vector: Ct @ 0 @ Cp^T folded, seconds per instruction)."""
    a = _ARRS.get(key)
    if a is None:
        c = _CONSTS[key]
        a = dict(basis={k: jnp.asarray(getattr(c.basis, k)) for k in _BASIS_ARRAYS},
                 check={k: jnp.asarray(getattr(c.check_basis, k)) for k in _BASIS_ARRAYS},
                 modes_s=tuple(c.modes["s"]), modes_c=tuple(c.modes["c"]), zero_c=c.zero_c)
        _ARRS[key] = a
    return a


def _view(key, arrs):
    """_CONSTS[key] with its large arrays replaced by the (traced) `arrs`."""
    import copy
    c = _CONSTS[key]
    v = copy.copy(c)
    v.basis = copy.copy(c.basis)
    v.check_basis = copy.copy(c.check_basis)
    for k in _BASIS_ARRAYS:
        setattr(v.basis, k, arrs["basis"][k])
        setattr(v.check_basis, k, arrs["check"][k])
    v.modes = {"s": arrs["modes_s"], "c": arrs["modes_c"]}
    v.zero_c = arrs["zero_c"]
    return v


class _TorusConsts:
    """Shape-only constants of the torus problem at one resolution (shared by all sources)."""

    def __init__(self, nfp, M, N, oversample, phi0):
        self.nfp = nfp
        self.basis = Basis(nfp, M, N, oversample=oversample, phi0=phi0)
        b = self.basis
        self.check_basis = Basis(nfp, M, N, ntheta=b.ntheta, nphi=b.nphi, shift=True, phi0=phi0)
        # mode lists for the Jacobian blocks; the grid FFT indices of the sum and difference modes are
        # formed inside the jitted assembly (fused into the gathers) instead of stored (n^2 int32 each)
        self.modes = {"s": (jnp.asarray(b.modes_s[:, 0], jnp.int32), jnp.asarray(b.modes_s[:, 1], jnp.int32)),
                      "c": (jnp.asarray(b.modes_c[:, 0], jnp.int32), jnp.asarray(b.modes_c[:, 1], jnp.int32))}
        self.zero_c = jnp.asarray((b.modes_c[:, 0] == 0) & (b.modes_c[:, 1] == 0))
        pos_c = {tuple(k): j for j, k in enumerate(map(tuple, b.modes_c))}
        self.cols_c = np.array([pos_c[tuple(k)] for k in map(tuple, b.modes_s)])

    def blocks(self, ahat, rk, ck):
        b = self.basis
        if _xp(ahat) is np:
            from .torus_cpu import blocks as _blocks_cpu
            return _blocks_cpu(ahat.reshape(b.ntheta, b.nphi), self.modes[rk], self.modes[ck], b.ntheta, b.nphi,
                               rk, ck, np.asarray(self.zero_c))
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
            Mx = _xp(Mx).where(self.zero_c[:, None], 0.5 * Mx, Mx)
        return Mx


def _system_body(c: _TorusConsts, engine, u, flux_target, flux_scale, symmetric):
    """The Newton system with the field from ``engine`` (traced: used for derivatives with
    respect to the source; the solves evaluate the field between two jitted halves)."""
    b = c.basis
    rc, zs, iota = b.unpack(u)
    R, Z, *_ = b.geometry(rc, zs)
    x, _ = grid_points(b, R, Z, symmetric)
    o = engine.evaluate(x, grad=True, A=True)
    return _system_post(c, u, o["B"], o["A"], o["G"], flux_target, flux_scale, symmetric)


def _system_post(c: _TorusConsts, u, Bx, Ax, Gx, flux_target, flux_scale, symmetric):
    """F, J, flux, res from the field (Cartesian B, A, G) at the points of grid_points."""
    xp = _xp(u, Bx)
    b = c.basis
    rc, zs, iota = b.unpack(u)
    R, Z, Rt, Zt, Rp, Zp = b.geometry(rc, zs)
    g = field_from_points(b, Bx, Ax, Gx, symmetric)
    BR, Bp, BZ = g["BR"], g["Bp"], g["BZ"]
    ib = 1.0 / Bp
    VR, VZ = R * BR * ib, R * BZ * ib
    FRg = Rp + iota * Rt - VR
    FZg = Zp + iota * Zt - VZ
    flux = -2 * np.pi * xp.mean(g["AR"] * Rt + g["AZ"] * Zt)
    FR = b.proj_s(FRg).ravel()[b.is_]
    FZ = b.proj_c(FZg).ravel()[b.ic]
    F = xp.concatenate([FR, FZ, xp.atleast_1d((flux - flux_target) * flux_scale)])
    aRR = BR * ib + R * (g["dBR_R"] * Bp - BR * g["dBp_R"]) * ib ** 2
    aRZ = R * (g["dBR_Z"] * Bp - BR * g["dBp_Z"]) * ib ** 2
    aZR = BZ * ib + R * (g["dBZ_R"] * Bp - BZ * g["dBp_R"]) * ib ** 2
    aZZ = R * (g["dBZ_Z"] * Bp - BZ * g["dBp_Z"]) * ib ** 2
    ng = b.ntheta * b.nphi
    hat = (lambda a: _sfft().fft2(a, workers=_FFT_W) / ng) if xp is np else (lambda a: xp.fft.fft2(a) / ng)
    if xp is np:
        J = _jacobian_np(c, b, hat, aRR, aRZ, aZR, aZZ, iota, Rt, Zt, Bp, flux_scale)
        res = np.maximum(np.max(np.abs(FR)), np.max(np.abs(FZ)))
        return F, J, flux, res
    JRR = -c.blocks(hat(aRR), "s", "c")
    JRZ = -c.blocks(hat(aRZ), "s", "s")
    JZR = -c.blocks(hat(aZR), "c", "c")
    JZZ = -c.blocks(hat(aZZ), "c", "s")
    nfp = c.nfp
    ms_, ns_ = b.modes_s[:, 0], b.modes_s[:, 1]
    rows_s = np.arange(b.ns)
    # diagonal (d_phi + iota d_theta): R row (sin test, mode (m,n)) <- rc_(m,n): (n nfp - iota m);
    # Z row (cos test (m,n)) <- zs_(m,n): (iota m - n nfp); the (0,0) Z row has none
    JRR = _add_at(xp, JRR, rows_s, c.cols_c, ns_ * nfp - iota * ms_)
    JZZ = _add_at(xp, JZZ, c.cols_c, rows_s, iota * ms_ - ns_ * nfp)
    Jiota_R = b.proj_s(Rt).ravel()[b.is_]
    Jiota_Z = b.proj_c(Zt).ravel()[b.ic]
    cfl = b.proj_c(Bp * Zt).ravel()[b.ic]
    cfl = xp.where(c.zero_c, cfl, 0.5 * cfl)
    sfl = 0.5 * b.proj_s(Bp * Rt).ravel()[b.is_]
    Jflux = xp.concatenate([2 * np.pi * cfl, -2 * np.pi * sfl, xp.zeros(1)]) * flux_scale
    top = xp.concatenate([JRR, JRZ, Jiota_R[:, None]], axis=1)
    mid = xp.concatenate([JZR, JZZ, Jiota_Z[:, None]], axis=1)
    J = xp.concatenate([top, mid, Jflux[None, :]], axis=0)
    res = xp.maximum(xp.max(xp.abs(FR)), xp.max(xp.abs(FZ)))
    return F, J, flux, res


def _jacobian_np(c, b, hat, aRR, aRZ, aZR, aZZ, iota, Rt, Zt, Bp, flux_scale):
    """The Newton Jacobian on the host, written block by block into one preallocated matrix (numba
    gathers; no n^2 temporaries or concatenations)."""
    from .torus_cpu import blocks
    ns, nc = b.ns, b.nc
    n = ns + nc + 1
    J = np.empty((n, n))
    zc = np.asarray(c.zero_c)
    blocks(hat(aRR), c.modes["s"], c.modes["c"], b.ntheta, b.nphi, "s", "c", zc, out=J[:ns, :nc], sign=-1.0)
    blocks(hat(aRZ), c.modes["s"], c.modes["s"], b.ntheta, b.nphi, "s", "s", zc, out=J[:ns, nc:nc + ns], sign=-1.0)
    blocks(hat(aZR), c.modes["c"], c.modes["c"], b.ntheta, b.nphi, "c", "c", zc, out=J[ns:ns + nc, :nc], sign=-1.0)
    blocks(hat(aZZ), c.modes["c"], c.modes["s"], b.ntheta, b.nphi, "c", "s", zc, out=J[ns:ns + nc, nc:nc + ns], sign=-1.0)
    nfp = c.nfp
    ms_, ns_ = b.modes_s[:, 0], b.modes_s[:, 1]
    rows_s = np.arange(ns)
    J[rows_s, c.cols_c] += ns_ * nfp - iota * ms_                 # (d_phi + iota d_theta) on R rows
    J[ns + c.cols_c, nc + rows_s] += iota * ms_ - ns_ * nfp       # and on Z rows
    J[:ns, nc + ns] = b.proj_s(Rt).ravel()[b.is_]
    J[ns:ns + nc, nc + ns] = b.proj_c(Zt).ravel()[b.ic]
    cfl = b.proj_c(Bp * Zt).ravel()[b.ic]
    cfl = np.where(zc, cfl, 0.5 * cfl)
    sfl = 0.5 * b.proj_s(Bp * Rt).ravel()[b.is_]
    J[n - 1, :nc] = 2 * np.pi * cfl * flux_scale
    J[n - 1, nc:nc + ns] = -2 * np.pi * sfl * flux_scale
    J[n - 1, n - 1] = 0.0
    return J


@partial(jax.jit, static_argnames=("key", "symmetric"))
def _grid_x_shared(u, arrs, key, symmetric):
    """Field evaluation points of the torus u (shapes static: shared by every source)."""
    b = _view(key, arrs).basis
    rc, zs, iota = b.unpack(u)
    R, Z, *_ = b.geometry(rc, zs)
    return grid_points(b, R, Z, symmetric)[0]


@partial(jax.jit, static_argnames=("key", "symmetric"))
def _system_post_shared(u, Bx, Ax, Gx, flux_target, flux_scale, arrs, key, symmetric):
    return _system_post(_view(key, arrs), u, Bx, Ax, Gx, flux_target, flux_scale, symmetric)


@partial(jax.jit, static_argnames=("key", "symmetric", "check"))
def _points_shared(rc, zs, arrs, key, symmetric, check):
    """Field points of the torus (rc, zs) on the basis grid, or on the half-shifted check grid."""
    c = _view(key, arrs)
    b = c.check_basis if check else c.basis
    R, Z, *_ = b.geometry(rc, zs)
    return grid_points(b, R, Z, symmetric)[0]


def _check_post(cb, rc, zs, iota, Bx, symmetric):
    """Max off-grid invariance residual (m) and max |B.n| / (|B| |n|) from B at the check-grid points."""
    xp = _xp(rc, Bx)
    R, Z, Rt, Zt, Rp, Zp = cb.geometry(rc, zs)
    g = field_from_points(cb, Bx, None, None, symmetric, need_grad=False, need_A=False)
    BR, Bp, BZ = g["BR"], g["Bp"], g["BZ"]
    FR = Rp + iota * Rt - R * BR / Bp
    FZ = Zp + iota * Zt - R * BZ / Bp
    nR, nP, nZ = -R * Zt, Zt * Rp - Rt * Zp, R * Rt
    Bn = BR * nR + Bp * nP + BZ * nZ
    bn = xp.abs(Bn) / (xp.sqrt(BR ** 2 + Bp ** 2 + BZ ** 2) * xp.sqrt(nR ** 2 + nP ** 2 + nZ ** 2))
    return xp.maximum(xp.max(xp.abs(FR)), xp.max(xp.abs(FZ))), xp.max(bn)


@partial(jax.jit, static_argnames=("key", "symmetric"))
def _check_post_shared(rc, zs, iota, Bx, arrs, key, symmetric):
    return _check_post(_view(key, arrs).check_basis, rc, zs, iota, Bx, symmetric)


def _flux_post(b, rc, zs, Ax, symmetric):
    xp = _xp(rc, Ax)
    R, Z, Rt, Zt, Rp, Zp = b.geometry(rc, zs)
    g = field_from_points(b, xp.zeros_like(Ax), Ax, None, symmetric, need_grad=False, need_A=True)
    return -2 * np.pi * xp.mean(g["AR"] * Rt + g["AZ"] * Zt)


@partial(jax.jit, static_argnames=("key", "symmetric"))
def _flux_post_shared(rc, zs, Ax, arrs, key, symmetric):
    return _flux_post(_view(key, arrs).basis, rc, zs, Ax, symmetric)


_VIEWS_NP = {}


def _view_np(key):
    """_CONSTS[key] with every array on the host (numpy): the CPU path without JIT."""
    v = _VIEWS_NP.get(key)
    if v is None:
        a = _arrays(key)
        host = lambda t: jax.tree_util.tree_map(np.asarray, t)
        v = _VIEWS_NP[key] = _view(key, host(a))
        v.zero_c = np.asarray(a["zero_c"])
    return v


class TorusProblem:
    """The torus problem at resolution (M, N) for one engine.  A Newton system is two module-level
    jitted functions with only shapes static (shared by every source): the grid points of u, and
    F, J from the field at those points; the engine evaluates the field in between (numba
    kernels on CPUs, the JAX kernels on GPUs)."""

    def __init__(self, engine, nfp, M, N, oversample=2.5, symmetric=True, phi0=0.0):
        self.engine = engine
        self.nfp = nfp
        self.phi0 = phi0
        self.symmetric = symmetric
        self.key = (float(nfp), int(M), int(N), float(oversample), float(phi0))
        c = _CONSTS.get(self.key)
        if c is None:
            c = _CONSTS[self.key] = _TorusConsts(nfp, M, N, oversample, phi0)
        self.c = c
        self.basis = c.basis
        self.check_basis = c.check_basis
        self._modes, self._zero_c = c.modes, c.zero_c
        self.shared = True
        # CPU kernels active: the whole Newton system on the host in numpy/scipy (no JIT tracing per
        # process or shape); otherwise the jitted functions (GPU, or JAX on CPU when asked)
        self.np_path = bool(getattr(engine, "cpu_kernels", False)) and os.environ.get("TORUX_HOST_TORUS", "1") != "0"
        self.v = _view_np(self.key) if self.np_path else None

    # ------------------------------------------------------------------ pieces
    def geometry(self, u):
        rc, zs, iota = self.basis.unpack(u)
        return rc, zs, iota, self.basis.geometry(rc, zs)

    def field_at(self, u, need_grad=True, need_A=True):
        """The engine's field at the grid points of u (Cartesian B, A, G)."""
        if self.np_path:
            b = self.v.basis
            rc, zs, _ = b.unpack(np.asarray(u))
            R, Z, *_ = b.geometry(rc, zs)
            x, _ = grid_points(b, R, Z, self.symmetric)
            o = self.engine.evaluate(x, grad=need_grad, A=need_A)
            return {k: np.asarray(v) for k, v in o.items()}
        x = _grid_x_shared(jnp.asarray(u), _arrays(self.key), key=self.key, symmetric=self.symmetric)
        return self.engine.evaluate(x, grad=need_grad, A=need_A)

    def system(self, u, flux_target, flux_scale):
        o = self.field_at(u)
        if self.np_path:
            return _system_post(self.v, np.asarray(u, float), o["B"], o["A"], o["G"], float(flux_target),
                                float(flux_scale), self.symmetric)
        return _system_post_shared(u, o["B"], o["A"], o["G"], flux_target, flux_scale, _arrays(self.key), key=self.key,
                                   symmetric=self.symmetric)

    # ------------------------------------------------------------------ checks
    # (each a jitted pair around the engine call, shared by every source at this resolution: no
    # op-by-op dispatch of the array expressions)
    def check(self, rc, zs, iota):
        """Off-grid residual (m) and max |B.n|/|B| on the half-shifted grid."""
        from . import fast_design as _so
        keep = _so.on("quad")
        if self.np_path:
            cb = self.v.check_basis
            rc, zs = np.asarray(rc, float), np.asarray(zs, float)
            R, Z, *_ = cb.geometry(rc, zs)
            x, _ = grid_points(cb, R, Z, self.symmetric)
            o = self.engine.evaluate(x, grad=False, A=False, err=keep)
            if keep:
                self._keep_err(rc, zs, o)
            res, bn = _check_post(cb, rc, zs, float(iota), np.asarray(o["B"]), self.symmetric)
            return float(res), float(bn)
        if keep:
            rc_h, zs_h = np.asarray(rc, float), np.asarray(zs, float)
        rc, zs = jnp.asarray(rc), jnp.asarray(zs)
        x = _points_shared(rc, zs, _arrays(self.key), key=self.key, symmetric=self.symmetric, check=True)
        o = self.engine.evaluate(x, grad=False, A=False, err=keep)
        if keep:
            self._keep_err(rc_h, zs_h, o)
        res, bn = _check_post_shared(rc, zs, jnp.asarray(float(iota)), o["B"], _arrays(self.key), key=self.key,
                                     symmetric=self.symmetric)
        return float(res), float(bn)

    def _keep_err(self, rc, zs, o):
        """(fast_design 'quad') the half rule's estimate from the check's own evaluation: the same points and
        engine quadrature_check would evaluate (the half-rule sum comes with every evaluation)."""
        self._check_err = (np.asarray(rc, float).tobytes() + np.asarray(zs, float).tobytes(), id(self.engine),
                           float(np.max(np.asarray(o["err_half"]))))

    def quadrature_check(self, rc, zs):
        """Largest a posteriori error estimate of the field engine's quadrature on the torus (the
        N/2 rule's relative error; the N-point rule's is about its square): the torus's field is at
        rounding level when this is <= 1e-8."""
        from . import fast_design as _so
        kept = getattr(self, "_check_err", None)
        if _so.on("quad") and kept is not None and kept[1] == id(self.engine) and \
                kept[0] == np.asarray(rc, float).tobytes() + np.asarray(zs, float).tobytes():
            return kept[2]
        if self.np_path:
            cb = self.v.check_basis
            R, Z, *_ = cb.geometry(np.asarray(rc, float), np.asarray(zs, float))
            x, _ = grid_points(cb, R, Z, self.symmetric)
        else:
            x = _points_shared(jnp.asarray(rc), jnp.asarray(zs), _arrays(self.key), key=self.key,
                               symmetric=self.symmetric, check=True)
        o = self.engine.evaluate(x, grad=False, A=False, err=True)
        return float(np.max(np.asarray(o["err_half"])))

    def flux_of(self, rc, zs):
        if self.np_path:
            b = self.v.basis
            rc, zs = np.asarray(rc, float), np.asarray(zs, float)
            R, Z, *_ = b.geometry(rc, zs)
            x, _ = grid_points(b, R, Z, self.symmetric)
            o = self.engine.evaluate(x, grad=False, A=True)
            return float(_flux_post(b, rc, zs, np.asarray(o["A"]), self.symmetric))
        rc, zs = jnp.asarray(rc), jnp.asarray(zs)
        x = _points_shared(rc, zs, _arrays(self.key), key=self.key, symmetric=self.symmetric, check=False)
        o = self.engine.evaluate(x, grad=False, A=True)
        return float(_flux_post_shared(rc, zs, o["A"], _arrays(self.key), key=self.key, symmetric=self.symmetric))

    # ------------------------------------------------------------------ Newton
    def newton(self, state: TorusState, flux_target, tol=1e-13, maxit=20, verbose=False, deadline=None,
               max_step_frac=0.5, fixed_iota=None, filt=0.0, patience=None, floor_stop=None, predict=None,
               diverge=False):
        """Damped Newton at the requested flux (or, with fixed_iota, at that transform: the flux
        equation is then replaced by iota = fixed_iota).

        filt > 0: Tikhonov-filtered least-squares steps on the Jacobian's singular directions,
        sum_i s_i / (s_i^2 + lam^2) (u_i . -F) v_i with lam = filt * s_max.  Next to a resonance the
        Newton system has a nearly free direction (s_min / s_max ~ the small divisor, a normal
        displacement at the resonant harmonic) along which the plain step amplifies the projected
        residual's error by 1 / s_min; the filtered step leaves it where the start put it, and the
        resonance guard judges what the torus then leaves there.

        patience: stop after that many consecutive iterations that do not halve the best residual (a Newton from
        a continuation predictor converges quadratically; one that does not is a step into a resonance or past
        the resolution's truncation floor, and the remaining iterations are wasted).  None: maxit decides.

        floor_stop (None: off): stop at the arithmetic floor (see newton_floor_reached).
        predict (None: off; a dict carried between solves): stop after a full step whose Newton-Kantorovich
        prediction puts the next residual below the tolerance (see newton_predicts), without evaluating it."""
        b = self.v.basis if self.np_path else self.basis
        st = state.resized(b.M, b.N)
        u = b.pack(st.rc, st.zs, st.iota if fixed_iota is None else fixed_iota)
        if predict is not None and predict.get("grid"):
            predict["_fresh"] = True     # (fast_family 'cpgrid') C from this solve's own steps once it has measured one
        R00 = float(st.rc[0, b.N])
        ft = flux_target if fixed_iota is None else (st.flux if st.flux else 1.0)
        flux_scale = R00 / max(abs(ft), 1e-300)
        hist = []
        best = None
        stall = 0
        slow = 0
        deadline = _budget.earliest(deadline)
        timed_out = False
        predicted = None
        last_step = None
        for it in range(maxit):
            if deadline is not None and time.time() > deadline:
                timed_out = True
                break
            F, J, flux, res = self.system(u, ft, flux_scale)
            if fixed_iota is not None:
                if self.np_path:
                    F = np.array(F, copy=True); J = np.array(J, copy=True)
                    F[-1] = u[-1] - fixed_iota
                    J[-1] = 0.0
                    J[-1, -1] = 1.0
                else:
                    F = F.at[-1].set(u[-1] - fixed_iota)
                    J = J.at[-1].set(0.0).at[-1, -1].set(1.0)
            nF = float(np.max(np.abs(np.asarray(F))))
            hist.append((nF, float(res), float(u[-1]), float(flux)))
            if verbose:
                print(f"   it {it:2d} |F| {nF:.3e} res {float(res):.3e} iota {float(u[-1]):.15f} flux {float(flux):.12e}")
            if not np.isfinite(nF):
                break
            slow = slow + 1 if (best is not None and nF > 0.5 * best[0]) else 0
            if best is None or nF < best[0]:
                best = (nF, u)
                stall = 0
            else:
                stall += 1
            if nF < tol * max(1.0, R00):
                best = (nF, u)
                break
            if floor_stop is not None and newton_floor_reached(F, tol, R00, flux_scale, ft, floor_stop):
                best = (nF, u)
                break
            if predict is not None:
                newton_learn(predict, last_step, nF, tol, R00)
            if stall >= 2 and best[0] < 1e3 * tol * max(1.0, R00):
                break            # no new best in two steps at a tolerance-level residual: the rounding floor
            if patience is not None and slow >= patience:
                break
            if diverge and _so2().diverged([h_[0] for h_ in hist], R00):
                break            # (fast_family 'cdiverge') a continuation step's Newton that has diverged or stalled
            if filt > 0:
                # on the host (LAPACK): n <= Continuation.FILTER_MAX, and device SVDs of this size are a
                # needless risk on GPUs
                Jn, Fn = np.asarray(J), np.asarray(F)
                Uj, sj, Vtj = np.linalg.svd(Jn, full_matrices=False)
                lam_t = filt * sj[0]
                du = Vtj.T @ ((sj / (sj ** 2 + lam_t ** 2)) * (Uj.T @ -Fn))
                du = du if self.np_path else jnp.asarray(du)
            else:
                du = dense_solve(J, -F)
            if not bool(np.all(np.isfinite(np.asarray(du)))):
                break
            step = float(np.max(np.abs(np.asarray(du[:-1]))))
            shape = float(np.max(np.abs(np.asarray(u[1:b.nc]))))
            lam = 1.0 if step <= max_step_frac * shape else max_step_frac * shape / step
            if predict is not None:
                dun = np.asarray(du)
                if filt == 0:
                    xq = _xp(J, du)          # on the device that holds J (no n^2 transfer on GPUs)
                    r_lin = float(xq.max(xq.abs(J @ du + F)))
                else:
                    r_lin = np.inf
                last_step = (_step_norm(b, dun, predict), r_lin) + ((nF,) if predict.get("grid") else ()) if lam == 1.0 else None
                pr = newton_predicts(predict, lam, last_step, tol, R00) if last_step is not None else None
                if pr is not None:
                    u = u + du
                    best = (pr, u)
                    predicted = pr
                    break
            u = u + lam * du
        nF, u = best if best is not None else (np.inf, u)
        rc, zs, iota = b.unpack(u)
        fl = float(flux_target) if fixed_iota is None else self.flux_of(np.asarray(rc), np.asarray(zs))
        out = TorusState(np.asarray(rc), np.asarray(zs), float(iota), fl, b.M, b.N, self.nfp,
                         res=float(nF), its=len(hist), converged=bool(nF < 1e3 * tol * max(1.0, R00)),
                         phi0=self.phi0)
        out.info["hist"] = hist
        if predicted is not None:
            out.info["predicted"] = predicted
        if timed_out:
            out.info["stopped"] = "deadline"
        out.tail = max(out.tails())
        return out


FLUX_FLOOR_ULPS = 8.0


def newton_floor_reached(F, tol, R00, flux_scale, flux_target, c):
    """The arithmetic floor of the torus equations (fast_design 'floor'): the Galerkin residual below tol
    (relative to the major radius) and the flux equation's residual within c x tol, or within FLUX_FLOOR_ULPS
    of the flux's own resolution.  The flux part is (flux - target) x R00 / |target|: at convergence it
    bounces between 1 and 7 ulps of the target (the flux is a grid mean of A . X_theta whose rounding is a few
    ulps), and no Newton step can take it lower: waiting for two iterations without a new best spends 2-4
    field evaluations per refinement level."""
    Fn = np.asarray(F)
    R0 = max(1.0, abs(float(R00)))
    gal = float(np.max(np.abs(Fn[:-1]))) if Fn.shape[0] > 1 else 0.0
    if not (gal < tol * R0):
        return False
    fl = abs(float(Fn[-1]))
    unit = abs(float(flux_scale)) * float(np.spacing(abs(float(flux_target)))) if flux_target else 0.0
    return fl < max(c * tol * R0, FLUX_FLOOR_ULPS * unit)


def _step_norm(b, du, state):
    """The step's size in the Newton-Kantorovich prediction: |du|_1 (it bounds the largest grid displacement), or with
    state['grid'] (fast_family 'cpgrid') the largest grid displacement itself, max |dR|, |dZ| over the grid, plus
    |d iota| -- the quantity the quadratic remainder is made of.  The constant C is learned and used in the same norm,
    and (with 'grid') only from steps that reduced the residual: the near-axis start torus's first step (residual
    4e-3 -> 0.15) had set C = 2.8e4 for a whole family, and no later step could be predicted."""
    du = np.asarray(du)
    if not (state is not None and state.get("grid")):
        return float(np.sum(np.abs(du)))
    drc, dzs, dio = b.unpack(du)
    dR, dZ = np.asarray(b.synth_c(drc)), np.asarray(b.synth_s(dzs))
    return float(max(np.max(np.abs(dR)), np.max(np.abs(dZ))) + abs(float(dio)))


PREDICT_SAFETY = 100.0
PREDICT_VALID = 1e-14          # a step's measured remainder counts only if the next residual is above this
                               # (relative): below it the residual is at the rounding floor, not the remainder


def newton_learn(state, last_step, nF, tol, R00):
    """Record the quadratic constant C of the Newton remainder from the last full step: the residual after a
    step du from u is F(u + du) = r_lin + R with |R| <= C |du|_1^2 (|du|_1 bounds the step's largest grid
    displacement).  Measured where the new residual is well above the rounding floor; the largest C seen is
    kept (QUASR 954, 25455, 1635364, 10614: C = 0.1 - 14)."""
    if last_step is None:
        return
    l1, r_lin = last_step[0], last_step[1]
    if not (np.isfinite(l1) and l1 > 0 and np.isfinite(r_lin)):
        return
    if len(last_step) > 2 and not (nF < last_step[2]):
        return           # (fast_family 'cpgrid') a step that did not reduce the residual is not in Newton's quadratic regime
    R0 = max(1.0, abs(float(R00)))
    if nF > PREDICT_VALID * R0:
        C = (nF + r_lin) / l1 ** 2
        if state.pop("_fresh", False):
            state["C"] = C       # (fast_family 'cpgrid') a new solve's first measurement replaces the family's last torus's
        else:
            state["C"] = max(C, state.get("C") or 0.0)
        state["n"] = state.get("n", 0) + 1


def newton_predicts(state, lam, last_step, tol, R00):
    """The bound r_lin + PREDICT_SAFETY C |du|_1^2 on the residual after the full step du, when it is below the
    tolerance (fast_design 'predict'), else None.  Needs a measured C (newton_learn)."""
    C = state.get("C")
    if C is None or lam != 1.0:
        return None
    l1, r_lin = last_step[0], last_step[1]
    R0 = max(1.0, abs(float(R00)))
    bound = r_lin + PREDICT_SAFETY * C * l1 ** 2
    if np.isfinite(bound) and r_lin < tol * R0 and bound < tol * R0:
        return float(bound)
    return None


def dense_solve(J, b):
    """J x = b: LAPACK through scipy on CPUs (multithreaded BLAS; TORUX_DENSE=xla for XLA's LU), XLA's
    solver on GPUs."""
    import os
    if _xp(J) is np or (jax.default_backend() == "cpu" and os.environ.get("TORUX_DENSE", "scipy") == "scipy"):
        import scipy.linalg as sla
        Jn, bn = np.asarray(J), np.asarray(b)
        so3 = _so3()
        if (so3.on("gpulu") or so3.on("mixlu")) and Jn.ndim == 2 and Jn.shape[0] >= so3.DENSE_FAST_MIN:
            x = _dense_fast(Jn, bn, so3)
            if x is not None:
                return x if _xp(J) is np else jnp.asarray(x)
        x = sla.lu_solve(sla.lu_factor(Jn, check_finite=False), bn, check_finite=False)
        return x if _xp(J) is np else jnp.asarray(x)
    return jnp.linalg.solve(J, b)


def _so3():
    from . import fast_newton
    return fast_newton


DENSE_STATS = {}        # (fast_newton 'mixlu' / 'gpulu') counts: solves, refinement steps, float64 fallbacks


def _dense_fast(J, b, so3):
    """(fast_newton) J x = b for the large dense Newton systems (the family's dense backups) without the host's
    float64 LU, or None (then the float64 LU decides):

    'gpulu': on a card (a task with torux.fast_family 'cudafil'), the float64 LU of the card's solver (cuSOLVER through
    JAX) -- a backward-stable float64 solve like LAPACK's, at the card's fp64 rate;
    'mixlu': the float32 LU of J with float64 iterative refinement (r = b - J x in float64, x += LU32^-1 r) until the
    residual is at the float64 LU's backward-error level, max|r| <= MIXLU_C eps (max_i sum_j |J_ij| max|x| + max|b|),
    within MIXLU_STEPS steps; a backward-stable float64 solution then, as the float64 LU's (forward errors of both
    ~ cond(J) eps).  Refinement converges when cond(J) is well below 1 / eps32 ~ 1e7; otherwise the float64 LU."""
    if so3.on("gpulu"):
        x = _gpu_solve(J, b)
        if x is not None:
            DENSE_STATS["gpu"] = DENSE_STATS.get("gpu", 0) + 1
            return x
    if not so3.on("mixlu"):
        return None
    import scipy.linalg as sla
    if not (np.all(np.isfinite(J)) and np.all(np.isfinite(b))):
        return None
    lu = sla.lu_factor(J.astype(np.float32), check_finite=False)
    if not np.all(np.isfinite(lu[0])):
        return None
    eps = np.finfo(np.float64).eps
    anorm = float(np.max(np.sum(np.abs(J), axis=1)))
    bn = float(np.max(np.abs(b)))
    x = sla.lu_solve(lu, b.astype(np.float32), check_finite=False).astype(np.float64)
    last = np.inf
    for k in range(so3.MIXLU_STEPS):
        r = b - J @ x
        rn = float(np.max(np.abs(r)))
        if not np.isfinite(rn):
            break
        if rn <= so3.MIXLU_C * eps * (anorm * float(np.max(np.abs(x))) + bn):
            DENSE_STATS["mix"] = DENSE_STATS.get("mix", 0) + 1
            DENSE_STATS["mix_steps"] = DENSE_STATS.get("mix_steps", 0) + k
            return x
        if rn > 0.5 * last:
            break                    # refinement not contracting: cond(J) too large for the float32 factors
        last = rn
        x = x + sla.lu_solve(lu, r.astype(np.float32), check_finite=False).astype(np.float64)
    DENSE_STATS["mix_fallback"] = DENSE_STATS.get("mix_fallback", 0) + 1
    return None


_GPU_SOLVE = []


def _gpu_solve(J, b):
    """(fast_newton 'gpulu') J x = b in float64 on the process's CUDA device (jnp.linalg.solve: cuSOLVER getrf/getrs),
    or None without one."""
    try:
        if not _GPU_SOLVE:
            devs = [d for d in jax.devices() if d.platform == "gpu"]
            _GPU_SOLVE.append((devs[0], jax.jit(jnp.linalg.solve)) if devs else None)
        if _GPU_SOLVE[0] is None:
            return None
        dev, f = _GPU_SOLVE[0]
        x = np.asarray(f(jax.device_put(J, dev), jax.device_put(b, dev)))
        return x if np.all(np.isfinite(x)) else None
    except Exception:  # noqa: BLE001
        return None


# ============================================================================ Newton-Krylov
def _fft_freqs(nt, nph):
    m = np.fft.fftfreq(nt, 1.0 / nt)
    k = np.fft.fftfreq(nph, 1.0 / nph)
    if nt % 2 == 0:
        m[nt // 2] = 0.0
    if nph % 2 == 0:
        k[nph // 2] = 0.0
    return m[:, None], k[None, :]


class KrylovTorus:
    """Matrix-free Newton-Krylov for the torus problem of a TorusProblem (any resolution).

    J v costs two syntheses and two projections on the grid (the field and its gradient are
    evaluated once per Newton step).  Preconditioner: approximate reducibility.  In the frame
    F = [t, nu], t = X_theta, nu = (-Z_theta, R_theta) / (B_phi |t|^2) (so that the invariant
    form Omega = B_phi dR ^ dZ gives Omega(t, nu) = 1), the linearised invariance operator
    L = (d_phi + iota d_theta) - DV is, at an invariant torus, upper triangular with the
    constant-coefficient diagonal (d_phi + iota d_theta):
        L(F eta) = F [(d + iota d_th) eta1 + T eta2 ;  (d + iota d_th) eta2]
    with T = (F^-1 L nu)_1, and the flux change is -2 pi <eta2>.  Two cohomological solves by
    FFT invert it (small divisors regularised); used as the right preconditioner of GMRES on
    the exact Galerkin Jacobian.
    """

    def __init__(self, prob: "TorusProblem", reg=1e-12):
        self.prob = prob
        b = prob.basis
        self.reg = reg
        m, k = _fft_freqs(b.ntheta, b.nphi)
        on_host = bool(getattr(prob, "np_path", False))
        self.m = m if on_host else jnp.asarray(m)
        self.k = k if on_host else jnp.asarray(k)
        self._host = None
        if on_host:
            # the operators on the host basis (numpy arrays): the CPU path's Newton-Krylov step
            h = KrylovTorus.__new__(KrylovTorus)
            h.prob = _ProbView(prob.v, prob.engine, prob.symmetric)
            h.reg, h.m, h.k, h._host = reg, m, k, None
            self._host = h

    def _L(self, f, iota):
        """(d_phi + iota d_theta) f on the grid, pseudo-spectrally."""
        b = self.prob.basis
        xp = _xp(f)
        if xp is np:
            # host: real FFTs over the half spectrum (the multiplier is odd and imaginary, Nyquist included)
            kh = self.k[:, :f.shape[-1] // 2 + 1]
            sf = _sfft()
            return sf.irfft2(sf.rfft2(f, workers=_FFT_W) * (1j * (iota * self.m + kh * b.nfp)), s=f.shape[-2:],
                             workers=_FFT_W)
        lam = 1j * (iota * self.m + self.k * b.nfp)
        return xp.real(xp.fft.ifft2(xp.fft.fft2(f) * lam))

    def _Linv(self, f, iota):
        """Zero-mean solution g of (d_phi + iota d_theta) g = f - <f>, small divisors regularised."""
        b = self.prob.basis
        xp = _xp(f)
        if xp is np:
            ck = (float(iota), f.shape)
            c = self.__dict__.get("_linv_cache")
            if c is not None and c[0] == ck:
                inv = c[1]              # (fast_newton 'pcache') the same divisors as the last call at this iota
            else:
                kh = self.k[:, :f.shape[-1] // 2 + 1]
                lam = 1j * (iota * self.m + kh * b.nfp)
                a2 = np.abs(lam) ** 2
                inv = np.where(a2 > 0, np.conj(lam) / (a2 + self.reg ** 2), 0.0)
                if _so3().on("pcache"):
                    self._linv_cache = (ck, inv)
            sf = _sfft()
            return sf.irfft2(sf.rfft2(f, workers=_FFT_W) * inv, s=f.shape[-2:], workers=_FFT_W)
        lam = 1j * (iota * self.m + self.k * b.nfp)
        a2 = xp.abs(lam) ** 2
        inv = xp.where(a2 > 0, xp.conj(lam) / (a2 + self.reg ** 2), 0.0)
        return xp.real(xp.fft.ifft2(xp.fft.fft2(f) * inv))

    def _grid(self, u):
        prob = self.prob
        b = prob.basis
        rc, zs, iota, (R, Z, *_) = prob.geometry(u)
        x, _ = grid_points(b, R, Z, prob.symmetric)
        o = prob.engine.evaluate(x, grad=True, A=True)
        return self._grid_from(u, o["B"], o["A"], o["G"])

    def _grid_from(self, u, Bx, Ax, Gx):
        prob = self.prob
        b = prob.basis
        rc, zs, iota, (R, Z, Rt, Zt, Rp, Zp) = prob.geometry(u)
        g = field_from_points(b, Bx, Ax, Gx, prob.symmetric)
        BR, Bp, BZ = g["BR"], g["Bp"], g["BZ"]
        ib = 1.0 / Bp
        G = dict(R=R, Z=Z, Rt=Rt, Zt=Zt, Rp=Rp, Zp=Zp, Bp=Bp, iota=iota, rc=rc, zs=zs)
        G["aRR"] = BR * ib + R * (g["dBR_R"] * Bp - BR * g["dBp_R"]) * ib ** 2
        G["aRZ"] = R * (g["dBR_Z"] * Bp - BR * g["dBp_Z"]) * ib ** 2
        G["aZR"] = BZ * ib + R * (g["dBZ_R"] * Bp - BZ * g["dBp_R"]) * ib ** 2
        G["aZZ"] = R * (g["dBZ_Z"] * Bp - BZ * g["dBp_Z"]) * ib ** 2
        G["FRg"] = Rp + iota * Rt - R * BR * ib
        G["FZg"] = Zp + iota * Zt - R * BZ * ib
        G["flux"] = -2 * np.pi * _xp(Rt).mean(g["AR"] * Rt + g["AZ"] * Zt)
        return G

    def _F(self, G, flux_target, flux_scale, iota_t=None):
        b = self.prob.basis
        FR = b.proj_s(G["FRg"]).ravel()[b.is_]
        FZ = b.proj_c(G["FZg"]).ravel()[b.ic]
        last = (G["flux"] - flux_target) * flux_scale if iota_t is None else (G["iota"] - iota_t)
        xp = _xp(FR)
        return xp.concatenate([FR, FZ, xp.atleast_1d(last)])

    def _unpack_eq(self, r):
        """Equation-space vector -> (sin coefficients of F_R, cos coefficients of F_Z) arrays and flux part."""
        b = self.prob.basis
        shape = (b.M + 1, 2 * b.N + 1)
        xp = _xp(r)
        rs = _scatter(xp, shape[0] * shape[1], b.is_, r[:b.ns]).reshape(shape)
        rcc = _scatter(xp, shape[0] * shape[1], b.ic, r[b.ns:b.ns + b.nc]).reshape(shape)
        return rs, rcc, r[-1]

    def _Jv(self, G, v, flux_scale, fixed=False):
        b = self.prob.basis
        nfp = b.nfp
        drc, dzs, diota = b.unpack(v)
        dR = b.synth_c(drc)
        dZ = b.synth_s(dzs)
        iota = G["iota"]
        # (d_phi + iota d_theta) acts diagonally on the coefficients; the rest pointwise
        LR = b.synth_s((b.nn * nfp - iota * b.mm) * drc)          # (n nfp - iota m) sin
        LZ = b.synth_c((iota * b.mm - b.nn * nfp) * dzs)
        jR = LR + diota * G["Rt"] - G["aRR"] * dR - G["aRZ"] * dZ
        jZ = LZ + diota * G["Zt"] - G["aZR"] * dR - G["aZZ"] * dZ
        JR = b.proj_s(jR).ravel()[b.is_]
        JZ = b.proj_c(jZ).ravel()[b.ic]
        xp = _xp(JR)
        if fixed:
            return xp.concatenate([JR, JZ, xp.atleast_1d(diota)])
        dflux = 2 * np.pi * xp.mean(G["Bp"] * (G["Zt"] * dR - G["Rt"] * dZ)) * flux_scale
        return xp.concatenate([JR, JZ, xp.atleast_1d(dflux)])

    def _precond(self, G, r, flux_scale, fixed=False):
        b = self.prob.basis
        iota = G["iota"]
        rs, rcc, rflux = self._unpack_eq(r)
        rR = b.synth_s(rs)
        rZ = b.synth_c(rcc)
        tR, tZ, Bp = G["Rt"], G["Zt"], G["Bp"]
        pc = G.get("_pcache") if isinstance(G, dict) else None
        if pc is None:
            t2 = tR ** 2 + tZ ** 2
            nR, nZ = -tZ / (Bp * t2), tR / (Bp * t2)
            # T = first component of F^-1 L nu, L nu = (d + iota d_th) nu - DV nu
            wR = self._L(nR, iota) - G["aRR"] * nR - G["aRZ"] * nZ
            wZ = self._L(nZ, iota) - G["aZR"] * nR - G["aZZ"] * nZ
            T = Bp * (nZ * wR - nR * wZ)
            if _xp(T) is np and _so3().on("pcache"):
                # (fast_newton 'pcache') nR, nZ, T depend on the Newton iterate only, not on the GMRES vector r:
                # the same arrays for every preconditioner application of this Newton step (bit for bit)
                G["_pcache"] = (nR, nZ, T)
        else:
            nR, nZ, T = pc
        s1 = Bp * (nZ * rR - nR * rZ)
        s2 = Bp * (-tZ * rR + tR * rZ)
        if fixed:
            # the last equation fixes d iota; the mean of eta2 (the flux change) is set by the
            # solvability of the first equation: <s1 - T eta2> = d iota (needs the twist <T>)
            diota = rflux
            e20 = self._Linv(s2, iota)
            xp = _xp(T)
            Tm = xp.mean(T)
            c = (xp.mean(s1) - diota - xp.mean(T * e20)) / xp.where(xp.abs(Tm) > 0, Tm, 1.0)
            eta2 = e20 + c
            q1 = s1 - T * eta2 - diota
            eta1 = self._Linv(q1, iota)
        else:
            dflux = rflux / flux_scale
            eta2 = self._Linv(s2, iota) - dflux / (2 * np.pi)
            q1 = s1 - T * eta2
            diota = _xp(q1).mean(q1)
            eta1 = self._Linv(q1, iota)
        xR = eta1 * tR + eta2 * nR
        xZ = eta1 * tZ + eta2 * nZ
        drc = b.proj_c(xR) * b.mask_c
        dzs = b.proj_s(xZ) * b.mask_s
        return b.pack(drc, dzs, diota)

    def _newton_step(self, u, Bx, Ax, Gx, flux_target, flux_scale, gtol, iota_t, restart=40, maxiter=4, fixed=False):
        """One Newton-Krylov step from the field (Cartesian B, A, G) at the grid points of u."""
        G = self._grid_from(u, Bx, Ax, Gx)
        F = self._F(G, flux_target, flux_scale, iota_t if fixed else None)
        A = lambda v: self._Jv(G, v, flux_scale, fixed)
        Mi = lambda r: self._precond(G, r, flux_scale, fixed)
        du, info = jax.scipy.sparse.linalg.gmres(A, -F, M=Mi, tol=gtol, atol=0.0, restart=restart,
                                                 maxiter=maxiter, solve_method="batched")
        res_lin = jnp.max(jnp.abs(A(du) + F))
        return du, F, G["flux"], res_lin

    def _newton_step_np(self, u, Bx, Ax, Gx, flux_target, flux_scale, gtol, iota_t, restart=40, maxiter=4, fixed=False):
        """The Newton-Krylov step on the host (numpy, scipy's GMRES with the same operator and
        right preconditioner): no JIT tracing per process or shape."""
        from scipy.sparse.linalg import LinearOperator, gmres
        G = self._grid_from(u, Bx, Ax, Gx)
        F = self._F(G, flux_target, flux_scale, iota_t if fixed else None)
        n = F.shape[0]
        A = LinearOperator((n, n), matvec=lambda v: self._Jv(G, v, flux_scale, fixed), dtype=float)
        Mi = LinearOperator((n, n), matvec=lambda r_: self._precond(G, r_, flux_scale, fixed), dtype=float)
        try:
            du, info = gmres(A, -F, rtol=gtol, atol=0.0, restart=restart, maxiter=maxiter, M=Mi)
        except TypeError:          # scipy < 1.12: tol instead of rtol
            du, info = gmres(A, -F, tol=gtol, atol=0.0, restart=restart, maxiter=maxiter, M=Mi)
        res_lin = float(np.max(np.abs(A.matvec(du) + F)))
        return du, F, G["flux"], res_lin

    def newton(self, state: TorusState, flux_target, tol=1e-13, maxit=20, verbose=False, gtol=1e-6,
               restart=40, maxiter=4, max_step_frac=0.5, deadline=None, fixed_iota=None, patience=None,
               floor_stop=None, predict=None, diverge=False):
        prob = self.prob
        b = self._host.prob.basis if self._host is not None else prob.basis
        st = state.resized(b.M, b.N)
        fixed = fixed_iota is not None
        if predict is not None and predict.get("grid"):
            predict["_fresh"] = True     # (fast_family 'cpgrid', see TorusProblem.newton)
        u = b.pack(st.rc, st.zs, st.iota if not fixed else fixed_iota)
        R00 = float(st.rc[0, b.N])
        ft = flux_target if not fixed else (st.flux if st.flux else 1.0)
        flux_scale = R00 / max(abs(ft), 1e-300)
        iota_t = float(fixed_iota) if fixed else 0.0
        best = None
        hist = []
        stall = 0
        slow = 0
        deadline = _budget.earliest(deadline)
        timed_out = False
        predicted = None
        last_step = None
        for it in range(maxit):
            if deadline is not None and time.time() > deadline:
                timed_out = True
                break
            o = prob.field_at(u)
            if self._host is not None:
                du, F, flux, rl = self._host._newton_step_np(u, o["B"], o["A"], o["G"], ft, flux_scale, gtol, iota_t,
                                                             restart=restart, maxiter=maxiter, fixed=fixed)
            else:
                du, F, flux, rl = _nk_step_shared(u, o["B"], o["A"], o["G"], ft, flux_scale, gtol, iota_t,
                                                  _arrays(prob.key), key=prob.key, symmetric=prob.symmetric,
                                                  restart=restart, maxiter=maxiter, fixed=fixed, reg=self.reg)
            nF = float(np.max(np.abs(np.asarray(F))))
            hist.append((nF, float(rl), float(u[-1]), float(flux)))
            if verbose:
                print(f"   nk it {it:2d} |F| {nF:.3e} lin {float(rl):.2e} iota {float(u[-1]):.15f}", flush=True)
            if not np.isfinite(nF):
                break
            slow = slow + 1 if (best is not None and nF > 0.5 * best[0]) else 0
            if best is None or nF < best[0]:
                best = (nF, u)
                stall = 0
            else:
                stall += 1
            if nF < tol * max(1.0, R00):
                break
            if floor_stop is not None and newton_floor_reached(F, tol, R00, flux_scale, ft, floor_stop):
                best = (nF, u)
                break
            if predict is not None:
                newton_learn(predict, last_step, nF, tol, R00)
            if stall >= 2 and best[0] < 1e3 * tol * max(1.0, R00):
                break            # no new best in two steps at a tolerance-level residual: the rounding floor
            if patience is not None and slow >= patience:
                break            # (see TorusProblem.newton)
            if diverge and _so2().diverged([h_[0] for h_ in hist], R00):
                break            # (fast_family 'cdiverge', see TorusProblem.newton)
            step = float(np.max(np.abs(np.asarray(du[:-1]))))
            shape = float(np.max(np.abs(np.asarray(u[1:b.nc]))))
            lam = 1.0 if step <= max_step_frac * shape else max_step_frac * shape / step
            if predict is not None and bool(np.all(np.isfinite(np.asarray(du)))):
                last_step = (_step_norm(b, np.asarray(du), predict), float(rl)) + ((nF,) if predict.get("grid") else ()) if lam == 1.0 else None
                pr = newton_predicts(predict, lam, last_step, tol, R00) if last_step is not None else None
                if pr is not None:
                    u = u + du
                    best = (pr, u)
                    predicted = pr
                    break
            u = u + lam * du
        nF, u = best if best is not None else (np.inf, u)
        rc, zs, iota = b.unpack(u)
        fl = float(flux_target) if not fixed else prob.flux_of(np.asarray(rc), np.asarray(zs))
        out = TorusState(np.asarray(rc), np.asarray(zs), float(iota), fl, b.M, b.N, prob.nfp,
                         res=float(nF), its=len(hist), converged=bool(nF < 1e3 * tol * max(1.0, R00)),
                         phi0=prob.phi0)
        out.info["hist"] = hist
        out.info["solver"] = "newton-krylov"
        if predicted is not None:
            out.info["predicted"] = predicted
        if timed_out:
            out.info["stopped"] = "deadline"
        return out


class _ProbView:
    """What KrylovTorus needs of a TorusProblem, inside the shared jitted step."""

    def __init__(self, c, engine, symmetric):
        self.c, self.basis, self.engine, self.symmetric, self.nfp = c, c.basis, engine, symmetric, c.nfp

    def geometry(self, u):
        rc, zs, iota = self.basis.unpack(u)
        return rc, zs, iota, self.basis.geometry(rc, zs)


@partial(jax.jit, static_argnames=("key", "symmetric", "restart", "maxiter", "fixed", "reg"))
def _nk_step_shared(u, Bx, Ax, Gx, flux_target, flux_scale, gtol, iota_t, arrs, key, symmetric, restart, maxiter, fixed,
                    reg):
    c = _view(key, arrs)
    kt = KrylovTorus.__new__(KrylovTorus)
    kt.prob = _ProbView(c, None, symmetric)
    kt.reg = reg
    m, k = _fft_freqs(c.basis.ntheta, c.basis.nphi)
    kt.m, kt.k = jnp.asarray(m), jnp.asarray(k)
    return kt._newton_step(u, Bx, Ax, Gx, flux_target, flux_scale, gtol, iota_t, restart=restart, maxiter=maxiter,
                           fixed=fixed)
