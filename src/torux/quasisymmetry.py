"""The two-term and three-term quasisymmetry residuals of the field on a verified flux surface.

    q = quasisymmetry(result)                     # result: a torux.Result whose status is 'surface' or 'range'
    q = quasisymmetry(result, helicity=(1, -1))   # for a helicity of one's choosing
    print(q)
    q.two_term          # <f_C^2>
    q.three_term        # <f_T^2>^(1/2): the flux-surface rms of f_T
    q.helicity          # (M, n): the helicity f_C is measured against
    q.f_C, q.f_T        # both at every point of the grid q.theta x q.phi: answer.points(q.theta[j], q.phi[k])
    q.weight            # |sqrt(g)| there: two_term = sum(f_C^2 weight) / sum(weight)
    q.accuracy          # how far each of the two may be from exact: (two-term, three-term)
    q.checks            # what the computation rests on, measured on its grid (Method)

Definitions.  psi = Psi / 2 pi, Psi the toroidal flux inside the surface.  G and I are the covariant components of B
in Boozer angles, B = G grad zeta_B + I grad theta_B + K grad psi, iota the rotational transform, R the major radius,
and B, where it is a number, |B|:

    f_C = [(N - iota M) B x grad B . grad psi - (M G + N I) B . grad B] / B^3,     N = n nfp
    f_T = R^2 grad psi x grad B . grad(B . grad B) / B^4

two_term is <f_C^2> and three_term is <f_T^2>^(1/2), <.> the flux-surface average.  Both are dimensionless.  On a
surface where |B| depends on the Boozer angles only through M theta_B - n nfp zeta_B, f_C of that helicity vanishes,
and f_T vanishes for any helicity.  In a vacuum I = 0.  The formula keeps I, as computed, and q.checks['|I|/|G|']
says how close to zero it is (rounding, for every field torux reads).

Conventions.  The helicity (M, n) is written as torux.boozer writes the harmonics, cos(m theta_B - n nfp zeta_B),
theta_B running in VMEC's poloidal direction (counter-clockwise, with R to the right and Z up) and zeta_B with the
cylindrical angle: (1, 0) quasi-axisymmetric, (1, n) quasi-helical, (0, 1) quasi-poloidal.
  SIMSOPT (1.11.1, from its source): helicity_m = M and helicity_n = n, with N = n nfp and VMEC's poloidal angle.
    f_C is the expression of its QuasisymmetryRatioResidual, whose total() is the sum over surfaces of weight times
    <f_C^2>, and so two_term for this one surface at weight 1.
  DESC (0.17.4): its helicity's N includes nfp and its poloidal angle runs the other way, so the same field is
    (M, -n nfp) there, with iota of the other sign: the quasi-helical (1, -1) of a four-period field is DESC's (1, 4).
    f_C is its f_C_normalized for that helicity, and f_T its f_T_normalized, R and B local, of which three_term is
    the flux-surface rms.
Unless one is given, the helicity is read from the field: the largest harmonic B_mn of |B| in Boozer angles other than
B_00, divided by the greatest common divisor of m and n (helicity_from_spectrum).  Values of f_C for different
helicities carry different factors, so the helicity is never chosen by comparing them.

Method.  Every quantity in f_C and f_T is a derivative of |B| along the surface, so both are computed on the surface
alone, in its own straight-field-line angles (theta, phi), phi the cylindrical angle, where
B = grad psi x grad theta + iota grad phi x grad psi.  There B . grad phi = 1 / sqrt(g), grad psi is
(B . grad phi) X_theta x X_phi, and with B_theta = B . X_theta and B_phi = B . X_phi,

    B x grad B . grad psi           = (B . grad phi) (B_theta dB/dphi - B_phi dB/dtheta),
    B . grad B                      = B^theta dB/dtheta + B^phi dB/dphi,
    grad psi x grad B . grad Y      = (B . grad phi) (dB/dtheta dY/dphi - dB/dphi dY/dtheta),   Y = B . grad B,

B^theta and B^phi from B = B^theta X_theta + B^phi X_phi.  G and I are the means of B_phi and B_theta over the
angles, since along the surface the field is the gradient of G phi + I theta plus a periodic part where no current
crosses it, and the flux-surface average weights each point by |sqrt(g)|.  So the computation rests on three things,
each measured on every grid and refused beyond its bound (ValueError): the field along the surface (B . n / |B| at
most 1e-10), theta straight for it (|B^theta / B^phi - iota| at most 1e-9), and no current across it (the surface
curl of B_theta, B_phi at most 1e-8 of |G|).  Each derivative is spectral, and the averages are sums on a uniform grid,
which converge spectrally for a smooth periodic integrand.  The grid starts at twice the surface's harmonics per
direction and is made 1.4 times finer per direction until both numbers agree with the coarser grid's to 1e-12 of
themselves (at most four grids).  The stated accuracy is the last two grids' difference.  The helicity is turned into
the surface's own angle directions (its theta may run clockwise) before f_C is formed, so that it means what it means
in Boozer angles.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .boozer import Helicity, _half, _odd, _orientation, _surface_grid

__all__ = ["quasisymmetry", "Quasisymmetry", "helicity_from_spectrum"]


class Quasisymmetry:
    """The two-term and three-term quasisymmetry residuals on a flux surface, for one helicity, and f_C and f_T on the
    grid they were computed on."""

    def __init__(self, two_term, three_term, helicity, given, accuracy, nfp, wall_s, theta, phi, f_C, f_T, weight,
                 checks):
        self.two_term = two_term       # <f_C^2> for self.helicity
        self.three_term = three_term   # <f_T^2>^(1/2), the flux-surface rms of f_T
        self.helicity = helicity       # (M, n), as torux.boozer writes the harmonics
        self.given = given             # True where the helicity was given, False where it was read from the field
        self.accuracy = accuracy       # (two-term, three-term): the last two grids' difference
        self.nfp = nfp
        self.wall_s = wall_s
        self.theta = theta             # the surface's own straight-field-line angle, as answer.points takes it (rad)
        self.phi = phi                 # the cylindrical angle, over one field period (rad)
        self.f_C = f_C                 # (len(theta), len(phi))
        self.f_T = f_T
        self.weight = weight           # |sqrt(g)|, the flux-surface average's weight (m^3 per Wb rad^2)
        self.checks = checks           # {'B.n/|B|', 'straight', 'curl/|G|', '|I|/|G|'}: largest values on the grid

    def __repr__(self):
        h = self.helicity
        name = f"{h}, " if h.m == 1 or h == (0, 1) else ""
        whence = "as given" if self.given else "the helicity of the largest harmonic of |B| in Boozer angles"
        c = self.checks
        rows = [("two-term residual", f"<f_C^2> = {self.two_term:.4e} for {name}(M, n) = ({h.m}, {h.n:+d}): {whence}"),
                ("three-term residual", f"<f_T^2>^(1/2) = {self.three_term:.4e}, the flux-surface rms of f_T"),
                ("definitions", f"f_C = [(N - iota M) B x grad B . grad psi - (M G + N I) B . grad B] / B^3, "
                                f"N = n nfp, nfp = {self.nfp}"),
                ("", "f_T = R^2 grad psi x grad B . grad(B . grad B) / B^4, psi = Psi / 2 pi, <.> the flux-surface "
                     "average"),
                ("convention", f"SIMSOPT's QuasisymmetryRatioResidual for helicity_m = {h.m}, helicity_n = {h.n}. "
                               f"DESC's f_C_normalized and f_T_normalized for helicity ({h.m}, {-h.n * self.nfp}), its "
                               f"poloidal angle the other way"),
                ("rests on", f"B . n / |B| {c['B.n/|B|']:.0e}, theta straight to {c['straight']:.0e}, |I| / |G| "
                             f"{c['|I|/|G|']:.0e}"),
                ("accuracy", f"{self.accuracy[0]:.0e} and {self.accuracy[1]:.0e}"),
                ("time", f"{self.wall_s:.1f} s")]
        from .answer import _table
        return _table(rows)

    __str__ = __repr__


def helicity_from_spectrum(b) -> Helicity:
    """The helicity of a torux.boozer.Boozer's largest harmonic other than B_00, (m, n) divided by their greatest
    common divisor: (1, 0) for a quasi-axisymmetric field, (1, n) quasi-helical, (0, 1) quasi-poloidal."""
    for r in b.modes:                     # largest |B_mn| first
        m, n = int(r["m"]), int(r["n"])
        if m or n:
            g = math.gcd(m, n)
            return Helicity(0, 1) if m == 0 else Helicity(m // g, n // g)
    raise ValueError("|B| has no harmonic other than its mean, so it has no helicity to read")


def _helicity(h) -> Helicity:
    try:
        m, n = (int(v) for v in h)
        if (m, n) != tuple(h):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"helicity {h!r}: give two integers (M, n), as torux.boozer writes the harmonics "
                         f"cos(m theta_B - n nfp zeta_B)") from None
    if m < 0 or (m == 0 and n == 0):
        raise ValueError(f"helicity ({m}, {n}): M >= 0 and (M, n) not (0, 0), as torux.boozer writes the harmonics")
    return Helicity(m, n)


def _derivatives(f, nfp):
    """d/dtheta and d/dphi of f on a uniform periodic grid of odd sizes: theta over 2 pi on axis 0, phi over one
    field period on axis 1."""
    F = np.fft.fft2(f)
    km = np.fft.fftfreq(f.shape[0], 1.0 / f.shape[0])[:, None]
    kp = np.fft.fftfreq(f.shape[1], 1.0 / f.shape[1])[None, :] * nfp
    return np.fft.ifft2(1j * km * F).real, np.fft.ifft2(1j * kp * F).real


def _on_grid(t, field, nth, nph, M, N):
    """f_C for (M, N), N = n nfp in the surface's own angle directions, f_T, their flux-surface averages and what they
    rest on, on the uniform grid of the surface's angles: theta_j = 2 pi j / nth, phi_k = phi0 + 2 pi k / (nfp nph)."""
    th, ph, x, xt, xp = _surface_grid(t, nth, nph)
    # |B|, B . X_theta and B . X_phi are even under stellarator symmetry: the field at one point of each pair
    idx, partner = _half(nth, nph)
    Bh = np.asarray(field(x.reshape(-1, 3)[idx]), float)
    nrm = np.cross(xt, xp).reshape(-1, 3)[idx]
    vals = np.full((nth * nph, 3), np.nan)
    vals[idx] = np.stack([np.linalg.norm(Bh, axis=-1), np.sum(Bh * xt.reshape(-1, 3)[idx], -1),
                          np.sum(Bh * xp.reshape(-1, 3)[idx], -1)], -1)
    bn = float(np.max(np.abs(np.sum(Bh * nrm, -1)) / (np.linalg.norm(nrm, axis=-1) * vals[idx, 0])))
    miss = np.isnan(vals[:, 0])
    vals[miss] = vals[partner[miss]]
    modB, Bt, Bp = (vals[:, i].reshape(nth, nph) for i in range(3))
    gtt, gtp, gpp = np.sum(xt * xt, -1), np.sum(xt * xp, -1), np.sum(xp * xp, -1)
    det = gtt * gpp - gtp ** 2
    Bsup_t = (gpp * Bt - gtp * Bp) / det           # B^theta
    Bsup_p = (gtt * Bp - gtp * Bt) / det           # B^phi = B . grad phi = 1 / sqrt(g)
    iota = float(t.iota)
    G, I = float(np.mean(Bp)), float(np.mean(Bt))
    # the surface curl of (B_theta, B_phi): zero where no current crosses the surface
    Ft, Fp = np.fft.fft2(Bt), np.fft.fft2(Bp)
    km = np.fft.fftfreq(nth, 1.0 / nth)[:, None]
    kp = np.fft.fftfreq(nph, 1.0 / nph)[None, :] * t.nfp
    curl = float(np.max(np.abs(kp * Ft - km * Fp)) / Bt.size)
    checks = {"B.n/|B|": bn, "straight": float(np.max(np.abs(Bsup_t / Bsup_p - iota))), "curl/|G|": curl / abs(G),
              "|I|/|G|": abs(I) / abs(G)}
    dB_t, dB_p = _derivatives(modB, t.nfp)
    BxgB_gpsi = Bsup_p * (Bt * dB_p - Bp * dB_t)
    BgB = Bsup_t * dB_t + Bsup_p * dB_p
    dY_t, dY_p = _derivatives(BgB, t.nfp)
    R = np.hypot(x[..., 0], x[..., 1])
    f_C = ((N - iota * M) * BxgB_gpsi - (M * G + N * I) * BgB) / modB ** 3
    f_T = R ** 2 * Bsup_p * (dB_t * dY_p - dB_p * dY_t) / modB ** 4
    w = 1.0 / np.abs(Bsup_p)
    W = math.fsum(w.ravel())
    return dict(theta=th, phi=ph, f_C=f_C, f_T=f_T, w=w, G=G, I=I, checks=checks,
                two_term=math.fsum((f_C ** 2 * w).ravel()) / W,
                three_term=math.sqrt(math.fsum((f_T ** 2 * w).ravel()) / W))


# what the computation rests on, and how far each may be from exact before it is refused
BOUNDS = {"B.n/|B|": (1e-10, "the field crosses the surface by {:.1e} of |B|, so it is not a flux surface of this "
                             "field"),
          "straight": (1e-9, "theta is not a straight-field-line angle of this field on the surface: B^theta / B^phi "
                             "differs from iota by {:.1e}"),
          "curl/|G|": (1e-8, "a current crosses this surface ({:.1e} of |G|), so G and I cannot be read from the field "
                             "along it")}


def _rests_on(checks):
    for k, (bound, said) in BOUNDS.items():
        if not checks[k] <= bound:
            raise ValueError(said.format(checks[k]))


def quasisymmetry(result, helicity=None) -> Quasisymmetry:
    """The two-term residual <f_C^2> and the three-term residual <f_T^2>^(1/2) on the result's verified surface.
    helicity: (M, n), as torux.boozer writes the harmonics. By default the helicity of the largest harmonic of |B|
    in Boozer angles."""
    t0 = time.time()
    status = getattr(result, "status", None)
    t = getattr(result, "surface", None)
    if status not in ("surface", "range") or t is None:
        raise ValueError(f"there is no verified surface to measure quasisymmetry on: the status is {status!r}")
    if hasattr(t, "rs"):
        raise ValueError("this surface is not stellarator-symmetric, and the quasisymmetry residuals are computed only "
                         "for stellarator-symmetric surfaces")
    given = helicity is not None
    if given:
        hel = _helicity(helicity)
    else:
        from .boozer import boozer
        hel = helicity_from_spectrum(boozer(result))
    nfp, M, N = int(t.nfp), int(t.M), int(t.N)
    _, sign = _orientation(t)
    # M theta_B - n nfp zeta_B in the stated directions is M theta - sign n nfp phi in the surface's own (up to sign)
    Mc, Nc = hel.m, sign * hel.n * nfp
    nth, nph = _odd(4 * M + 1), _odd(4 * N + 1)
    prev = None
    for _ in range(4):
        q = _on_grid(t, result.field, nth, nph, Mc, Nc)
        _rests_on(q["checks"])
        two, three = q["two_term"], q["three_term"]
        if prev is not None:
            accuracy = (abs(two - prev[0]), abs(three - prev[1]))
            if accuracy[0] <= 1e-12 * two and accuracy[1] <= 1e-12 * three:
                break
        prev = (two, three)
        nth, nph = _odd(1.4 * nth), _odd(1.4 * nph)
    return Quasisymmetry(two_term=two, three_term=three, helicity=hel, given=given, accuracy=accuracy, nfp=nfp,
                         wall_s=time.time() - t0, theta=q["theta"], phi=q["phi"], f_C=q["f_C"], f_T=q["f_T"],
                         weight=q["w"], checks=q["checks"])
