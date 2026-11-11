"""Checks of the evaluator against exact answers.

    python results/evaluator/test_evaluator.py          # float64 tier anywhere; the precise tier on x86-64

1. B and A of a circular loop against the exact field (complete elliptic integrals, mpmath at 40 digits).
2. An exact invariant torus: a level set of the loop's poloidal flux R A_phi (a torus round the wire; field lines
   lie on it), computed to 40 digits and written as a double-precision rz_fourier series.  The evaluator must
   find |B.n| / |B| at the floor that the double-precision coefficients leave.
3. A SIMSOPT surface through the xyz_fourier conversion: positions and derivatives against SIMSOPT's own.
"""
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import Coils, judge_bn, points, precise_dtype  # noqa: E402

mp.mp.dps = 40
A_LOOP, I_LOOP = 1.0, 1.0e6
LOOP = [dict(xc=[0.0, A_LOOP], xs=[0.0, 0.0], yc=[0.0, 0.0], ys=[0.0, A_LOOP], zc=[0.0, 0.0], zs=[0.0, 0.0],
             current_A=I_LOOP)]


def exact_loop(R, Z):
    """(B_R, B_Z, A_phi) of a loop of radius a in z = 0 carrying I (counter-clockwise seen from +z)."""
    a, I, mu = mp.mpf(A_LOOP), mp.mpf(I_LOOP), 4 * mp.pi * mp.mpf("1e-7")
    R, Z = mp.mpf(R), mp.mpf(Z)  # (exact conversions of what is passed)
    m = 4 * a * R / ((a + R) ** 2 + Z ** 2)
    K, E = mp.ellipk(m), mp.ellipe(m)
    k = mp.sqrt(m)
    Aphi = mu * I / (mp.pi * k) * mp.sqrt(a / R) * ((1 - m / 2) * K - E)
    pre = mu * I / (2 * mp.pi * mp.sqrt((a + R) ** 2 + Z ** 2))
    den = (a - R) ** 2 + Z ** 2
    BZ = pre * (K + (a ** 2 - R ** 2 - Z ** 2) / den * E)
    BR = pre * Z / R * (-K + (a ** 2 + R ** 2 + Z ** 2) / den * E)
    return BR, BZ, Aphi


def test_loop_field(dtype, tol):
    rng = np.random.default_rng(1)
    pts = []
    for _ in range(12):
        R, Z, ph = 0.3 + 1.4 * rng.random(), -0.6 + 1.2 * rng.random(), 2 * np.pi * rng.random()
        if (R - 1) ** 2 + Z ** 2 < 0.04:
            continue
        pts.append((R, Z, ph))
    x = np.array([[R * np.cos(p), R * np.sin(p), Z] for R, Z, p in pts])
    C = Coils(LOOP, dtype)
    B = C.field(x.astype(dtype), "B")
    A = C.field(x.astype(dtype), "A")
    worst_B = worst_A = 0.0
    for xi, b, av in zip(x, B, A):
        # the exact field at the double-precision point itself
        X, Y, Z = (mp.mpf(float(v)) for v in xi)
        R = mp.sqrt(X ** 2 + Y ** 2)
        BR, BZ, Aphi = exact_loop(R, Z)
        cp, sp = X / R, Y / R
        ex = [BR * cp, BR * sp, BZ]
        nb = mp.sqrt(sum(e ** 2 for e in ex))
        worst_B = max(worst_B, float(mp.sqrt(sum((mp.mpf(str(bi)) - e) ** 2 for bi, e in zip(b, ex))) / nb))
        exA = [-Aphi * sp, Aphi * cp, 0]
        worst_A = max(worst_A, float(mp.sqrt(sum((mp.mpf(str(ai)) - e) ** 2 for ai, e in zip(av, exA))) / abs(Aphi)))
    print(f"  loop field vs exact ({np.dtype(dtype).name}): B {worst_B:.1e}, A {worst_A:.1e} (relative)")
    assert worst_B < tol and worst_A < tol, (worst_B, worst_A)


def loop_flux_torus(rho0=0.25, M=48):
    """A level set psi = R A_phi = psi0 round the wire, R = 1 + rho cos t, Z = rho sin t: rz_fourier, 40 digits."""
    def psi(R, Z):
        return R * exact_loop(R, Z)[2]
    psi0 = psi(1 + rho0, 0)
    n = 4 * M
    Rs, Zs = [], []
    for j in range(n):
        t = 2 * mp.pi * j / n
        rho = mp.findroot(lambda r: psi(1 + r * mp.cos(t), r * mp.sin(t)) - psi0, rho0)
        Rs.append(1 + rho * mp.cos(t))
        Zs.append(rho * mp.sin(t))
    rc, zs = [], []
    for m in range(M + 1):
        c = sum(Rs[j] * mp.cos(2 * mp.pi * m * j / n) for j in range(n)) * (2 if m else 1) / n
        s = sum(Zs[j] * mp.sin(2 * mp.pi * m * j / n) for j in range(n)) * 2 / n
        rc.append(float(c))
        zs.append(float(s))
    print(f"  torus: |rc_M| {abs(rc[-1]):.1e}, |zs_M| {abs(zs[-1]):.1e} (spectral tail)")
    return dict(form="rz_fourier", m=list(range(M + 1)), n=[0] * (M + 1), rc=rc, rs=[0.0] * (M + 1),
                zc=[0.0] * (M + 1), zs=zs)


def test_invariant_torus(dtype, bar):
    surf = loop_flux_torus()
    th, ph = points(600)
    bn, nq, _ = judge_bn(LOOP, surf, dtype, th, ph)
    print(f"  exact torus ({np.dtype(dtype).name}): max |B.n|/|B| {bn.max():.2e}, median {np.median(bn):.1e}, "
          f"nodes {nq}")
    assert bn.max() < bar, bn.max()


def test_simsopt_conversion():
    try:
        from simsopt._core import load
    except ImportError:
        print("  (SIMSOPT not installed here: conversion check skipped)")
        return
    from forms import from_simsopt_xyz_tensor, simsopt_check_points, check_points
    found = sorted((HERE.parents[1] / "examples").glob("quasr_*.json"))
    if not found:
        print("  (no QUASR download in examples/: conversion check skipped)")
        return
    f = found[0]
    s = load(str(f))[0][-1]
    d = from_simsopt_xyz_tensor(s)
    err = check_points(d, simsopt_check_points(s, 64))
    print(f"  SIMSOPT SurfaceXYZTensorFourier -> xyz_fourier: largest position difference {err:.1e} m")
    assert err < 1e-13


if __name__ == "__main__":
    print("float64 tier")
    test_loop_field(np.float64, 1e-14)
    test_invariant_torus(np.float64, 1e-14)
    test_simsopt_conversion()
    dt = precise_dtype()
    if dt is None:
        print("precise tier: longdouble is not extended precision here; skipped")
    else:
        print("precise tier")
        test_loop_field(dt, 1e-17)
        test_invariant_torus(dt, 1e-15)
    print("all passed")
