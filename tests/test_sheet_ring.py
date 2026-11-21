"""An exact case with plasma current, in closed form: a circular current ring inside the toroidal field RB / R.  The
ring's poloidal flux psi(R, Z) (elliptic integrals) has closed contours round it, and with the toroidal field they
are the flux surfaces.  Gamma is the contour through R = 1.1, Z = 0, written as a Fourier series to rounding, and
on it the sheet K = n x B / mu0.  The source is the toroidal field and the sheet: outside Gamma its field is the
ring's and the toroidal field's exactly (virtual casing), so torux.solve at a full flux F must return the contour
whose toroidal flux is F.  The truth here is scipy's elliptic integrals, nothing of torux's."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

import torux
from torux.field import FieldEngine

MU0 = 4e-7 * math.pi
RC, IC, RB, RG, M = 1.0, 2.2e4, 1.0, 0.1, 24


def psi(R, Z):
    R, Z = np.asarray(R, float), np.asarray(Z, float)
    m = 4 * RC * R / ((RC + R) ** 2 + Z ** 2)
    return R * MU0 * IC / (np.pi * np.sqrt(m)) * np.sqrt(RC / R) * ((1 - m / 2) * ellipk(m) - ellipe(m))


def B_ring(R, Z):
    R, Z = np.asarray(R, float), np.asarray(Z, float)
    a2, b2 = (R + RC) ** 2 + Z ** 2, (RC - R) ** 2 + Z ** 2
    m = 4 * RC * R / a2
    K, E = ellipk(m), ellipe(m)
    c = MU0 * IC / (2 * np.pi * np.sqrt(a2))
    return c * Z / R * (-K + (RC ** 2 + R ** 2 + Z ** 2) / b2 * E), c * (K + (RC ** 2 - R ** 2 - Z ** 2) / b2 * E)


def contour(level, nth):
    th = 2 * np.pi * np.arange(nth) / nth
    r = np.array([brentq(lambda rr: psi(RC + rr * math.cos(t), rr * math.sin(t)) - level, 1e-4, 0.9, xtol=1e-17,
                         rtol=8.9e-16, maxiter=200) for t in th])
    return RC + r * np.cos(th), r * np.sin(th)


def tor_flux(R, Z):
    """RB times the integral of ln R dZ round a section sampled uniformly in a periodic parameter."""
    k = np.fft.fftfreq(len(R), 1.0 / len(R))
    dZ = np.real(np.fft.ifft(1j * k * np.fft.fft(Z)))
    return float(RB * np.sum(np.log(R) * dZ) * 2 * np.pi / len(R))


def truth_level(F):
    return brentq(lambda L: tor_flux(*contour(L, 256)) - F, psi(RC + 0.4, 0.0), psi(RC + 0.02, 0.0), xtol=1e-18,
                  rtol=8.9e-16, maxiter=200)


def iota_of(level, nth=1024):
    R, Z = contour(level, nth)
    k = np.fft.fftfreq(nth, 1.0 / nth)
    dR, dZ = (np.real(np.fft.ifft(1j * k * np.fft.fft(v))) for v in (R, Z))
    BR, BZ = B_ring(R, Z)
    return 1.0 / (np.sum(RB * np.hypot(dR, dZ) / (R * R * np.hypot(BR, BZ))) / nth)


@pytest.fixture(scope="module")
def ring():
    LG = float(psi(RC + RG, 0.0))
    R, Z = contour(LG, 1024)
    fR, fZ = np.fft.rfft(R) / len(R), np.fft.rfft(Z) / len(Z)
    rmnc = np.concatenate([[fR[0].real], 2 * fR[1:M + 1].real])
    zmns = np.concatenate([[0.0], -2 * fZ[1:M + 1].imag])
    Mk = M + 40
    th = 2 * np.pi * np.arange(1024) / 1024
    m_ = np.arange(M + 1)[:, None]
    R3, Z3 = rmnc @ np.cos(m_ * th), zmns @ np.sin(m_ * th)
    R3t, Z3t = rmnc @ (-m_ * np.sin(m_ * th)), zmns @ (m_ * np.cos(m_ * th))
    BR, BZ = B_ring(R3, Z3)
    fb = np.fft.rfft((BR * R3t + BZ * Z3t) / MU0) / len(th)      # B.r_theta / mu0: the ring's (the toroidal field has none)
    cov = np.zeros((2, 2, Mk + 1))
    cov[0, 0] = np.concatenate([[fb[0].real], 2 * fb[1:Mk + 1].real])
    cov[0, 1] = np.concatenate([[0.0], -2 * fb[1:Mk + 1].imag])
    cov[1, 0, 0] = RB / MU0                                     # B.r_phi = R B_phi
    rm, zm = np.zeros(Mk + 1), np.zeros(Mk + 1)
    rm[:M + 1], zm[:M + 1] = rmnc, zmns
    part = torux.SurfaceCurrent(1, [(m, 0) for m in range(Mk + 1)], rm, zm, covariant=cov, flux=tor_flux(R, Z))
    return torux.Source("ring-sheet", [torux.ToroidalField(RB), part], nfp=1, stellsym=True), part


def test_the_sheet_makes_the_ring_field_outside(ring):
    src, part = ring
    E = FieldEngine(torux.Source("sheet", [part], nfp=1))
    rng = np.random.default_rng(1)
    t, ph = rng.uniform(0, 2 * np.pi, 24), rng.uniform(0, 2 * np.pi, 24)
    r = RG * (1 + rng.uniform(0.15, 0.5, 24))
    R, Z = RC + r * np.cos(t), r * np.sin(t)
    B = np.asarray(E.B(np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)))
    BR, BZ = B_ring(R, Z)
    ref = np.stack([BR * np.cos(ph), BR * np.sin(ph), BZ], -1)
    assert np.max(np.linalg.norm(B - ref, axis=1) / np.linalg.norm(ref, axis=1)) < 2e-14


@pytest.mark.slow
def test_solve_returns_the_ring_contour(ring):
    src, part = ring
    F = 1.3 * part.flux
    s = torux.solve(src, flux=F, budget_s=1800)
    assert s.status == "surface", str(s)
    L = truth_level(F)
    th = 2 * np.pi * np.arange(512) / 512
    rc, zs = np.asarray(s.surface.rc), np.asarray(s.surface.zs)
    Mm, Nn = rc.shape[0] - 1, (rc.shape[1] - 1) // 2
    for ph in (0.0, 1.0, 2.5):
        ang = np.arange(Mm + 1)[:, None, None] * th - np.arange(-Nn, Nn + 1)[None, :, None] * ph
        R = np.sum(rc[:, :, None] * np.cos(ang), axis=(0, 1))
        Z = np.sum(zs[:, :, None] * np.sin(ang), axis=(0, 1))
        BR, BZ = B_ring(R, Z)
        assert np.max(np.abs(psi(R, Z) - L) / (R * np.hypot(BR, BZ))) < RING_POSITION_M
        assert abs(tor_flux(R, Z) - F) < 1e-13 * F
    assert abs(abs(s.iota) - abs(iota_of(L))) < RING_IOTA


RING_POSITION_M = 1e-13     # measured 1.0e-14 m (the truth's own elliptic integrals read 5.6e-15 m), 21 Nov
RING_IOTA = 1e-13           # measured 1.2e-15


@pytest.mark.parametrize("orientation", [1, pytest.param(-1, marks=pytest.mark.slow)])
def test_check_measures_the_full_flux_of_the_ring_contour(ring, tmp_path, orientation):
    """torux.check's measurement on the truth contour at 1.3 times the flux inside the boundary, its theta either way
    round: the full flux (the part's flux inside the boundary and the flux between), and the field across it."""
    import json
    src, part = ring
    F = 1.3 * part.flux
    L = truth_level(F)
    R, Z = contour(L, 512)
    fR, fZ = np.fft.rfft(R) / len(R), np.fft.rfft(Z) / len(Z)
    Mc = 40
    rc = np.concatenate([[fR[0].real], 2 * fR[1:Mc + 1].real])[:, None]
    zs = orientation * np.concatenate([[0.0], -2 * fZ[1:Mc + 1].imag])[:, None]
    p = tmp_path / "contour.json"
    p.write_text(json.dumps(dict(surface=dict(nfp=1, M=Mc, N=0, rc=rc.tolist(), zs=zs.tolist()))))
    # check's own measurement at 64 points and two sections (torux.check takes 4,096 and six: minutes here)
    from torux.check import measure_at
    rng = np.random.default_rng(0)
    c = measure_at(src, str(p), 2 * np.pi * rng.random(64), 2 * np.pi * rng.random(64), [0.3, 2.1], flux=F)
    assert abs(c.flux / F - 1) < 1e-13
    assert c.leak < 1e-13
