"""A source with a current sheet: the sheet's field at rounding level next to its boundary, the full toroidal flux
its vector potential carries, and what torux.solve and torux.check refuse, in one sentence."""
import numpy as np
import pytest

import torux
from torux.field import FieldEngine
from torux.field.surface_current import SheetField
from torux.source import MU0


def d_shape(M=24, R0=1.0, a=0.3, k=1.5, dl=0.3):
    """An axisymmetric D-shaped boundary as a cosine / sine series in theta."""
    t = 2 * np.pi * np.arange(256) / 256
    R = R0 + a * np.cos(t + dl * np.sin(t))
    Z = k * a * np.sin(t)
    rc = np.array([np.mean(R * np.cos(m * t)) * (1 if m == 0 else 2) for m in range(M + 1)])
    zs = np.array([np.mean(Z * np.sin(m * t)) * 2 for m in range(M + 1)])
    return [(m, 0) for m in range(M + 1)], rc, zs


def boundary_points(modes, rc, zs, theta, phi, d):
    m = np.array([mm for mm, _ in modes])
    R = np.cos(np.outer(theta, m)) @ rc
    Z = np.sin(np.outer(theta, m)) @ zs
    Rt = -(np.sin(np.outer(theta, m)) * m) @ rc
    Zt = (np.cos(np.outer(theta, m)) * m) @ zs
    nR, nZ = Zt / np.hypot(Rt, Zt), -Rt / np.hypot(Rt, Zt)
    return np.stack([(R + d * nR) * np.cos(phi), (R + d * nR) * np.sin(phi), Z + d * nZ], -1)


def test_toroidal_solenoid_has_no_field_outside_down_to_1e_5_of_the_minor_radius():
    """An ideal toroidal solenoid (only G) on a shaped boundary: zero field outside, exactly.  The sheet's field
    there, against the field inside (mu0 G / (2 pi R0)), stays at rounding level however near the boundary."""
    modes, rc, zs = d_shape()
    G = 1e6
    sheet = torux.SurfaceCurrent(1, modes, rc, zs, G=G, kappas=np.zeros(len(modes)))
    S = SheetField(sheet)
    a = S.size["minor_radius"]
    th = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.1
    worst = 0.0
    for d in (0.1, 1e-2, 1e-3, 1e-4, 1e-5):
        for phi in (0.0, 0.3):
            B = S.evaluate(boundary_points(modes, rc, zs, th, phi, d * a), grad=False, potential=False)[:, :3]
            worst = max(worst, float(np.max(np.linalg.norm(B, axis=1))))
    assert worst / (MU0 * G / (2 * np.pi)) < 1e-15


def test_the_full_flux_is_the_boundarys_plus_the_flux_outside_it():
    """A solenoid sheet with a 1/R field: outside the boundary the field is RB / R alone, so the full flux round any
    loop outside is the part's flux plus RB times the integral of dR dZ / R between the boundary and the loop."""
    modes, rc, zs = d_shape(M=16)
    RB, psi = 0.7, 0.123
    sheet = torux.SurfaceCurrent(1, modes, rc, zs, G=3e5, kappas=np.zeros(len(modes)), flux=psi)
    src = torux.Source("solenoid", [sheet, torux.ToroidalField(RB)])
    E = FieldEngine(src)
    th = 2 * np.pi * np.arange(512) / 512
    m = np.array([mm for mm, _ in modes])
    Rg, Zg = np.cos(np.outer(th, m)) @ rc, np.sin(np.outer(th, m)) @ zs
    Zg_t = (np.cos(np.outer(th, m)) * m) @ zs
    for phi, (R0c, Z0c, r) in ((0.0, (1.0, 0.0, 0.6)), (1.1, (1.02, 0.05, 0.62))):
        Rc, Zc = R0c + r * np.cos(th), Z0c + r * np.sin(th)
        x = np.stack([Rc * np.cos(phi), Rc * np.sin(phi), Zc], -1)
        A = np.asarray(E.evaluate(x, grad=False, A=True)["A"])
        AR = A[:, 0] * np.cos(phi) + A[:, 1] * np.sin(phi)
        loop = -2 * np.pi * np.mean(AR * (-r * np.sin(th)) + A[:, 2] * (r * np.cos(th)))
        # int dR dZ / R over a region = oint ln(R) dZ round it (Green), counterclockwise
        inner = 2 * np.pi * np.mean(np.log(Rg) * Zg_t)
        outer = 2 * np.pi * np.mean(np.log(Rc) * r * np.cos(th))
        assert abs(loop - (psi + RB * (outer - inner))) < 1e-14 * psi


def test_a_source_with_a_sheet_is_refused_in_one_sentence_where_it_cannot_be_solved(tmp_path):
    modes, rc, zs = d_shape(M=8)
    bare = torux.SurfaceCurrent(1, modes, rc, zs, G=3e5, kappas=np.zeros(len(modes)))
    src = torux.Source("no flux", [bare, torux.ToroidalField(1.0)], stellsym=True)
    for ask in (dict(flux=0.3), dict(largest=True), dict(minor_radius=0.5)):
        with pytest.raises(ValueError, match="flux inside its boundary"):
            torux.solve(src, **ask)
    held = torux.SurfaceCurrent(1, modes, rc, zs, G=3e5, kappas=np.zeros(len(modes)), flux=0.2)
    src = torux.Source("with flux", [held, torux.ToroidalField(1.0)], stellsym=True)
    with pytest.raises(ValueError, match="not more than"):
        torux.solve(src, flux=0.1)
    with pytest.raises(ValueError, match="gradients"):
        torux.solve(src, flux=0.3, gradients=True)
    with pytest.raises(ValueError, match="trace_lines"):
        torux.solve(src, flux=0.3, trace_lines=2)
    two = torux.Source("two", [held, held, torux.ToroidalField(1.0)], stellsym=True)
    with pytest.raises(ValueError, match="one SurfaceCurrent"):
        torux.solve(two, flux=0.3)
    with pytest.raises(ValueError, match="already has minor radius"):
        torux.solve(src, minor_radius=0.1)
