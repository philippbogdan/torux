"""The field strength on a verified flux surface in Boozer angles, and how quasi-symmetric it is.

    b = boozer(result)          # result: a torux.Result whose status is 'surface' or 'range'
    print(b)
    b.modes                     # every harmonic, largest first: fields m, n, B (T)
    b.accuracy                  # T: how far a harmonic may be from exact
    b.quasisymmetry             # SIMSOPT's NonQuasiSymmetricRatio for the helicity that makes it smallest
    b.helicity                  # that helicity: (m, n) of the symmetric harmonics, printed by name
    b(theta_b, zeta_b)          # |B| from the harmonics

Convention.  |B| = sum of B_mn cos(m theta_B - n nfp zeta_B) over m >= 0 (and n >= 0 where m = 0): booz_xform's
xm_b = m, xn_b = n nfp, bmnc_b = B_mn, in VMEC's angle directions.  theta_B runs round the cross section from its
outboard point upward (counter-clockwise with R to the right and Z up), zeta_B with the cylindrical angle phi. Both
are zero at the outboard point of the symmetry plane phi = phi0 (phi0 = 0 for every QUASR design).

Method.  In a vacuum B = grad Phi.  On the surface X(theta, phi), theta its straight-field-line angle and phi the
cylindrical angle, Phi = G phi + I theta + nu(theta, phi) with nu periodic: dPhi/dtheta = B . X_theta and
dPhi/dphi = B . X_phi give G and I (their means, where I, the toroidal current inside the surface, is zero in a vacuum)
and nu, spectrally.  The Boozer angles are the straight-field-line angles in which Phi = G zeta_B + I theta_B:

    zeta_B = phi + lambda,   theta_B = theta + iota lambda,   lambda = nu / (G + iota I).

A harmonic is an integral over the Boozer angles, done on a uniform grid in (theta, phi) with the Jacobian
d(theta_B, zeta_B)/d(theta, phi) = 1 + lambda_phi + iota lambda_theta. The integrand is smooth and periodic, so the
trapezoid rule converges spectrally.  The harmonics are computed on grids 1.4 times finer per direction in turn
until two agree to 1e-13 of the mean field (at most four grids).  The stated accuracy is the largest of: the last
two grids' largest difference over every harmonic, the largest difference between the series and |B| at the
previous grid's points (which the last grid's integrals never used), and the largest sine part (zero under
stellarator symmetry).

Quasi-symmetry.  For a helicity (M, N), B_QS is the area-weighted mean of |B| along the lines on which
M theta_B - N nfp zeta_B is constant, and the number is int (|B| - B_QS)^2 dS / int B_QS^2 dS.  For (1, 0) this is
SIMSOPT's NonQuasiSymmetricRatio on a surface parametrised by Boozer angles, for (0, 1) its quasi-poloidal form,
and for (1, N) the same in the helical frame, as QUASR computes it for quasi-helical designs.  The helicity
reported is the one of (1, 0), (1, +-1), (1, +-2) and (0, 1) with the smallest number.  QUASR's qs_error is the
mean of this number over a design's stored surfaces (the square root of that mean is the 'quality of
quasisymmetry' of QUASR's papers).
"""
from __future__ import annotations

import math
import time
from typing import NamedTuple

import numpy as np

BLOCK = 16384               # points summed at once

__all__ = ["boozer", "Boozer", "Helicity"]


class Helicity(NamedTuple):
    """The symmetric harmonics (k m, k n) of |B|: (1, 0) quasi-axisymmetric, (1, n) quasi-helical, (0, 1)
    quasi-poloidal, in the harmonics' own convention cos(m theta_B - n nfp zeta_B)."""
    m: int
    n: int

    def __str__(self):
        if self.m == 0:
            return "quasi-poloidal"
        if self.n == 0:
            return "quasi-axisymmetric"
        return f"quasi-helical, n = {self.n:+d}"


_CANDIDATES = (Helicity(1, 0), Helicity(1, 1), Helicity(1, -1), Helicity(1, 2), Helicity(1, -2), Helicity(0, 1))

MODES = np.dtype([("m", np.int64), ("n", np.int64), ("B", np.float64)])


class Boozer:
    """|B| on a flux surface in Boozer angles: its harmonics and their accuracy (T), and its quasi-symmetry."""

    def __init__(self, modes, accuracy, quasisymmetry, helicity, G, iota, nfp, wall_s, ratios, phi0):
        self.modes = modes                   # every harmonic, largest |B_mn| first: fields m, n, B (T)
        self.accuracy = accuracy             # T: how far a harmonic may be from exact (see the module's docstring)
        self.quasisymmetry = quasisymmetry   # SIMSOPT's NonQuasiSymmetricRatio for self.helicity (QUASR's f_QS)
        self.helicity = helicity             # the helicity that makes it smallest
        self.G = G                           # T m: B = G grad zeta_B + (a multiple of grad psi) in a vacuum
        self.iota = iota                     # the rotational transform in these angles' directions
        self.nfp = nfp
        self.wall_s = wall_s
        self._ratios = ratios                # every helicity tried: {Helicity: ratio}
        self._phi0 = phi0

    def __call__(self, theta_b, zeta_b):
        """|B| (T) from the harmonics at Boozer angles theta_b, zeta_b (rad)."""
        th = np.asarray(theta_b, float)
        ze = np.asarray(zeta_b, float)
        m, n, b = self.modes["m"], self.modes["n"], self.modes["B"]
        return np.cos(th[..., None] * m - ze[..., None] * (n * self.nfp)) @ b

    def __repr__(self):
        b0 = float(self.modes["B"][(self.modes["m"] == 0) & (self.modes["n"] == 0)][0])
        rows = [("mean field", f"{b0:.10g} T"),
                ("quasi-symmetry", f"{self.quasisymmetry:.4e} for {self.helicity}: SIMSOPT's NonQuasiSymmetricRatio"
                                   + (" in the helical frame" if self.helicity.m and self.helicity.n else "")
                                   + " (QUASR's f_QS)")]
        others = sorted((v, str(h)) for h, v in self._ratios.items() if h != self.helicity)
        if others:
            rows.append(("next closest", f"{others[0][0]:.4e} for {others[0][1]}"))
        top = [r for r in self.modes if not (r["m"] == 0 and r["n"] == 0)][:6]
        cell = [f"({r['m']}, {r['n']:+d}) {r['B'] / b0:+.4e}" for r in top]
        rows.append(("largest B_mn / B_00", "  ".join(cell[:3])))
        if cell[3:]:
            rows.append(("", "  ".join(cell[3:])))
        mmax, nmax = int(self.modes["m"].max()), int(np.abs(self.modes["n"]).max())
        rows.append(("harmonics", f"{len(self.modes)} (m <= {mmax}, |n| <= {nmax}), each within {self.accuracy:.0e} T"))
        origin = "" if self._phi0 == 0 else f", zeta_B from phi = {self._phi0:.6g}"
        rows.append(("convention", f"cos(m theta_B - n nfp zeta_B), nfp = {self.nfp}: booz_xform's xm_b = m, "
                                   f"xn_b = n nfp{origin}"))
        rows.append(("time", f"{self.wall_s:.1f} s"))
        from .answer import _table
        return _table(rows)

    __str__ = __repr__


# ------------------------------------------------------------------------------------------------ the computation

def _odd(n):
    n = int(np.ceil(n))
    return n + 1 - n % 2


def _surface_grid(t, nth, nph):
    """Points, tangents X_theta, X_phi (Cartesian) of a stellarator-symmetric torus on the uniform grid
    theta_j = 2 pi j / nth, phi_k = phi0 + 2 pi k / (nfp nph). Arrays (nth, nph, 3)."""
    th = 2 * np.pi * np.arange(nth) / nth
    ph = t.phi0 + 2 * np.pi * np.arange(nph) / (t.nfp * nph)
    m = np.arange(t.M + 1)
    n = np.arange(-t.N, t.N + 1)
    at = np.outer(th, m)
    ap = np.outer(t.nfp * (ph - t.phi0), n)
    Ct, St, Cp, Sp = np.cos(at), np.sin(at), np.cos(ap), np.sin(ap)
    rc, zs = np.asarray(t.rc, float), np.asarray(t.zs, float)
    # R = sum rc cos(m th - n ps), Z = sum zs sin(m th - n ps), with ps = nfp (phi - phi0)
    R = Ct @ rc @ Cp.T + St @ rc @ Sp.T
    Z = St @ zs @ Cp.T - Ct @ zs @ Sp.T
    rcm, zsm = rc * m[:, None], zs * m[:, None]
    Rt = -St @ rcm @ Cp.T + Ct @ rcm @ Sp.T
    Zt = Ct @ zsm @ Cp.T + St @ zsm @ Sp.T
    rcn, zsn = rc * (n * t.nfp)[None, :], zs * (n * t.nfp)[None, :]
    # d/dphi cos(m th - n ps) = n nfp sin(...), d/dphi sin(...) = -n nfp cos(...)
    Rp = St @ rcn @ Cp.T - Ct @ rcn @ Sp.T
    Zp = -(Ct @ zsn @ Cp.T + St @ zsn @ Sp.T)
    c, s = np.cos(ph)[None, :], np.sin(ph)[None, :]
    x = np.stack([R * c, R * s, Z], -1)
    xt = np.stack([Rt * c, Rt * s, Zt], -1)
    xp = np.stack([Rp * c - R * s, Rp * s + R * c, Zp], -1)
    return th, ph, x, xt, xp


def _orientation(t):
    """(theta shift, theta sign): the outboard symmetry point of the plane phi0 and the counter-clockwise
    direction, as offsets of the torus's own theta."""
    th = np.array([0.0, np.pi, 0.0])
    ph = np.full(3, t.phi0)
    R, Z = t.geometry_on(th, ph)
    shift = 0.0 if R[0] >= R[1] else np.pi
    # dZ/dtheta at the outboard point: positive for counter-clockwise
    h = 1e-4
    _, Zh = t.geometry_on(np.array([shift + h, shift - h]), np.full(2, t.phi0))
    sign = 1.0 if Zh[0] > Zh[1] else -1.0
    return shift, sign


def _half(nth, nph):
    """Flat indices of one point of each stellarator-symmetric pair (theta, phi - phi0) ~ (-theta, phi0 - phi) of the
    grid (odd sizes), and of every point's partner."""
    j, k = np.meshgrid(np.arange(nth), np.arange(nph), indexing="ij")
    partner = ((-j) % nth) * nph + (-k) % nph
    keep = (k <= nph // 2) & ((k > 0) | (j <= nth // 2))
    return np.flatnonzero(keep.ravel()), partner.ravel()


def _transform(t, field, nth, nph, shift, sign, half=True):
    """Boozer angles, |B|, the area element and the Jacobian on one grid, with G, I and iota in the stated
    directions.  half: the field only at one point of each stellarator-symmetric pair (|B|, B . X_theta and
    B . X_phi are even under the symmetry). Otherwise at every point."""
    th, ph, x, xt, xp = _surface_grid(t, nth, nph)
    if half:
        idx, partner = _half(nth, nph)
        P = x.reshape(-1, 3)
        Bh = np.asarray(field(P[idx]), float)
        vals = np.full((nth * nph, 3), np.nan)
        vals[idx] = np.stack([np.linalg.norm(Bh, axis=-1), np.sum(Bh * xt.reshape(-1, 3)[idx], -1),
                              np.sum(Bh * xp.reshape(-1, 3)[idx], -1)], -1)
        miss = np.isnan(vals[:, 0])
        vals[miss] = vals[partner[miss]]
        modB, Bt, Bp = (vals[:, i].reshape(nth, nph) for i in range(3))
        B = None
    else:
        B = np.asarray(field(x.reshape(-1, 3)), float).reshape(x.shape)
        modB = np.linalg.norm(B, axis=-1)
        Bt = np.sum(B * xt, -1)                   # dPhi/dtheta
        Bp = np.sum(B * xp, -1)                   # dPhi/dphi
    dS = np.linalg.norm(np.cross(xt, xp), axis=-1)
    Ft, Fp = np.fft.fft2(Bt), np.fft.fft2(Bp)
    I = float(Ft[0, 0].real) / Bt.size
    G = float(Fp[0, 0].real) / Bp.size
    km = np.fft.fftfreq(nth, 1.0 / nth)[:, None]          # e^{i m theta}
    kp = np.fft.fftfreq(nph, 1.0 / nph)[None, :] * t.nfp  # e^{i k nfp (phi - phi0)}
    nu_hat = np.zeros_like(Ft)
    mm = np.broadcast_to(km, Ft.shape)
    pp = np.broadcast_to(kp, Ft.shape)
    a = mm != 0
    nu_hat[a] = Ft[a] / (1j * mm[a])
    b = (mm == 0) & (pp != 0)
    nu_hat[b] = Fp[b] / (1j * pp[b])
    # the two derivatives' consistency: curl B = 0 on the surface, and the grid's resolution of it
    curl = np.max(np.abs(1j * pp * Ft - 1j * mm * Fp)) / Bt.size
    iota = float(t.iota)
    den = G + iota * I
    lam = np.fft.ifft2(nu_hat).real / den
    lam_t = np.fft.ifft2(1j * mm * nu_hat).real / den
    lam_p = np.fft.ifft2(1j * pp * nu_hat).real / den
    TH = th[:, None] + iota * lam
    ZE = (ph[None, :] - t.phi0) + lam
    J = 1.0 + lam_p + iota * lam_t
    # to the stated directions: theta_B from the outboard point, counter-clockwise
    TH = sign * (TH - shift)
    return dict(theta=TH.ravel(), zeta=ZE.ravel(), modB=modB.ravel(), J=J.ravel(), dS=dS.ravel(), G=G,
                I=sign * I, iota=sign * iota, curl=float(curl), Jmin=float(J.min()),
                native=dict(x=x, xt=xt, xp=xp, B=B, I=I, iota=iota, lam_t=lam_t, lam_p=lam_p))


def _harmonics(theta, zeta, w, nfp, Mb, Nb):
    """For each column of w (P, k): c_mn sum_p w_p cos(m theta_p - n nfp zeta_p) / P, m = 0..Mb, n = -Nb..Nb
    (c = 2, and 1 at (0, 0)). Arrays (k, Mb + 1, 2 Nb + 1), and each column's largest sine sum (zero under
    stellarator symmetry)."""
    P, k = w.shape
    S = np.zeros((k * (Mb + 1), 2 * Nb + 1), complex)
    for a in range(0, P, BLOCK):                  # in blocks: a few tens of megabytes at a time
        sl = slice(a, a + BLOCK)
        Em = np.exp(1j * np.outer(theta[sl], np.arange(Mb + 1)))
        En = np.exp(-1j * nfp * np.outer(zeta[sl], np.arange(-Nb, Nb + 1)))
        S += (Em[:, None, :] * w[sl, :, None]).reshape(Em.shape[0], -1).T @ En
    S = (S / P).reshape(k, Mb + 1, 2 * Nb + 1)
    c = np.full(S.shape[1:], 2.0)
    c[0, Nb] = 1.0
    return c * S.real, np.max(np.abs(S.imag), axis=(1, 2))


def _pack(Bmn, Mb, Nb):
    out = []
    for m in range(Mb + 1):
        for j, n in enumerate(range(-Nb, Nb + 1)):
            if m == 0 and n < 0:
                continue
            out.append((m, n, Bmn[m, j]))
    a = np.array(out, dtype=MODES)
    return a[np.argsort(-np.abs(a["B"]), kind="stable")]


def _series(theta, zeta, Bmn, nfp, Mb, Nb):
    """sum over the half set of Bmn cos(m theta - n nfp zeta) at points."""
    W = Bmn.copy()
    W[0, :Nb] = 0.0                       # m = 0, n < 0 duplicates n > 0
    out = np.empty(len(theta))
    for a in range(0, len(theta), BLOCK):         # in blocks, as _harmonics: megabytes at a time at any resolution
        sl = slice(a, a + BLOCK)
        Em = np.exp(1j * np.outer(theta[sl], np.arange(Mb + 1)))
        En = np.exp(-1j * nfp * np.outer(zeta[sl], np.arange(-Nb, Nb + 1)))
        out[sl] = np.real(np.sum((Em @ W) * En, axis=1))
    return out


def _qs_ratios(g, h, theta, zeta, modB, dS, nfp, Mb, Nb):
    """NonQuasiSymmetricRatio for every candidate helicity: g, h the Boozer-angle harmonics of |B| dS_B and dS_B
    (dS_B the area element per d theta_B d zeta_B)."""
    out = {}
    for hel in _CANDIDATES:
        ks = [k for k in range(0, Mb + Nb + 1) if k * hel.m <= Mb and abs(k * hel.n) <= Nb]
        gk = np.array([g[k * hel.m, k * hel.n + Nb] for k in ks])
        hk = np.array([h[k * hel.m, k * hel.n + Nb] for k in ks])
        chi = hel.m * theta - hel.n * nfp * zeta
        num, den = [], []
        for a in range(0, len(chi), BLOCK):
            sl = slice(a, a + BLOCK)
            C = np.cos(np.outer(chi[sl], ks))
            BQS = (C @ gk) / (C @ hk)
            num.append(float(np.sum((modB[sl] - BQS) ** 2 * dS[sl])))
            den.append(float(np.sum(BQS ** 2 * dS[sl])))
        out[hel] = math.fsum(num) / math.fsum(den)
    return out


def boozer(result) -> Boozer:
    """The Boozer spectrum of |B| on the result's verified surface, and its quasi-symmetry."""
    t0 = time.time()
    status = getattr(result, "status", None)
    t = getattr(result, "surface", None)
    if status not in ("surface", "range") or t is None:
        raise ValueError(f"there is no verified surface to read the field strength on: the status is {status!r}")
    if hasattr(t, "rs"):
        raise ValueError("this surface is not stellarator-symmetric, and the Boozer spectrum is computed only for "
                         "stellarator-symmetric surfaces")
    nfp, M, N = int(t.nfp), int(t.M), int(t.N)
    Mb, Nb = M, N
    shift, sign = _orientation(t)
    nth, nph = _odd(2 * (Mb + M) + 1), _odd(2 * (Nb + N) + 1)
    prev = None
    for _ in range(4):
        q = _transform(t, result.field, nth, nph, shift, sign)
        if q["Jmin"] <= 0:
            raise ValueError("the Boozer angles are not single-valued on this surface (the transformation's Jacobian "
                             "changes sign), so it has no Boozer spectrum")
        w = np.stack([q["modB"] * q["J"], q["modB"] * q["dS"], q["dS"]], -1)
        (Bmn, g, h), (sym, _, _) = _harmonics(q["theta"], q["zeta"], w, nfp, Mb, Nb)
        ratios = _qs_ratios(g, h, q["theta"], q["zeta"], q["modB"], q["dS"], nfp, Mb, Nb)
        if prev is not None:
            pq, pB = prev
            # the finer grid's harmonics against the coarser's, and its series against |B| at the coarser grid's
            # points, which its own integrals never used
            recon = np.max(np.abs(_series(pq["theta"], pq["zeta"], Bmn, nfp, Mb, Nb) - pq["modB"]))
            accuracy = float(max(np.max(np.abs(Bmn - pB)), recon, 2 * sym))
            if accuracy <= 1e-13 * Bmn[0, Nb]:
                break
        prev = (q, Bmn)
        nth, nph = _odd(1.4 * nth), _odd(1.4 * nph)
    if q["curl"] > 1e-8 * abs(q["G"]):
        raise ValueError("a current crosses this surface, so the field is not the gradient of a potential on it and "
                         "its Boozer angles cannot be read from the field")
    hel = min(ratios, key=ratios.get)
    return Boozer(modes=_pack(Bmn, Mb, Nb), accuracy=accuracy, quasisymmetry=ratios[hel], helicity=hel, G=q["G"],
                  iota=q["iota"], nfp=nfp, wall_s=time.time() - t0, ratios=ratios, phi0=float(t.phi0))
