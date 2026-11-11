"""The Boozer spectrum of |B| and its quasi-symmetry, against code that shares none with it.

Quick tests rest on torux.reference (the independent evaluator, which shares no code with the solver or with
torux.boozer) and on QUASR's own table. The slow ones on SIMSOPT 1.11.1's BoozerExact surfaces and its
NonQuasiSymmetricRatio, and skip where SIMSOPT is not installed.
"""
import gc
import json
import os
import sys

import numpy as np
import pytest

from conftest import QUASR_954

import torux
from torux import boozer as BZ
from torux.boozer import Helicity, boozer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARE = os.path.join(REPO, "results", "compare300")


def inputs(design):
    return os.path.join(COMPARE, "inputs", f"quasr_{design}.json")


def metadata(design):
    with open(inputs(design)) as fh:
        return json.load(fh)["database_metadata"]


@pytest.fixture(scope="module")
def b954(solved_954):
    return boozer(solved_954)


def _mean(b):
    return float(b.modes["B"][(b.modes["m"] == 0) & (b.modes["n"] == 0)][0])


def _reference_grid(res, nth, nph):
    """The Boozer transformation on a grid the spectrum never used, every field value from torux.reference."""
    from torux.reference import fast64
    F = fast64.Field(res._engine.source)
    t = res.surface
    shift, sign = BZ._orientation(t)
    return BZ._transform(t, lambda x: F.evaluate(np.asarray(x, float))["B"], nth, nph, shift, sign, half=False)


# ------------------------------------------------------------------------------------------------ quick

def test_series_is_the_independent_field_strength(solved_954, b954):
    """The harmonics, summed at the Boozer angles of points on a grid neither the spectrum nor the solver used,
    give the reference evaluator's |B| there."""
    q = _reference_grid(solved_954, 151, 97)
    err = np.max(np.abs(b954(q["theta"], q["zeta"]) - q["modB"]))
    # the series is good to its accuracy against the solver's field. The two evaluators differ by their rounding
    engine = np.linalg.norm(solved_954.field(q["native"]["x"].reshape(-1, 3)), axis=-1)
    rounding = np.max(np.abs(engine - q["modB"]))
    assert rounding < 1e-14 * _mean(b954)
    assert err <= b954.accuracy + rounding, (err, b954.accuracy, rounding)


def test_angles_are_boozer_angles_of_the_independent_field(solved_954, b954):
    """In Boozer angles a vacuum field's covariant components are constant: B . X_zeta_B = G and B . X_theta_B = I
    (zero: no current inside the surface), with B from the reference evaluator."""
    q = _reference_grid(solved_954, 151, 97)
    nat = q["native"]
    io, lt, lp = nat["iota"], nat["lam_t"][..., None], nat["lam_p"][..., None]
    # d(theta_B, zeta_B)/d(theta, phi) = [[1 + iota lt, iota lp], [lt, 1 + lp]]. X_theta, X_phi from X_thB, X_zeB
    a, b_, c, d = 1 + io * lt, io * lp, lt, 1 + lp
    det = a * d - b_ * c
    x_thb = (d * nat["xt"] - c * nat["xp"]) / det
    x_zeb = (a * nat["xp"] - b_ * nat["xt"]) / det
    B = nat["B"]
    G = b954.G
    assert np.max(np.abs(np.sum(B * x_zeb, -1) - G)) <= 1e-13 * abs(G)
    assert np.max(np.abs(np.sum(B * x_thb, -1))) <= 1e-13 * abs(G)
    assert abs(q["G"] / G - 1) < 1e-13


def test_harmonics_converge_to_the_stated_accuracy(solved_954, b954):
    """A grid finer than any the spectrum used gives every harmonic within the stated accuracy, which is at the
    rounding level of |B|."""
    t = solved_954.surface
    shift, sign = BZ._orientation(t)
    q = BZ._transform(t, solved_954.field, 2 * (4 * t.M + 1) + 1, 2 * (4 * t.N + 1) + 1, shift, sign, half=False)
    (Bmn,), (sym,) = BZ._harmonics(q["theta"], q["zeta"], (q["modB"] * q["J"])[:, None], t.nfp, t.M, t.N)
    fine = {(m, n): Bmn[m, n + t.N] for m in range(t.M + 1) for n in range(-t.N, t.N + 1) if m or n >= 0}
    err = max(abs(r["B"] - fine[(int(r["m"]), int(r["n"]))]) for r in b954.modes)
    assert err <= b954.accuracy, (err, b954.accuracy)
    assert b954.accuracy < 1e-13 * _mean(b954)
    assert sym < 1e-14 * _mean(b954)


def test_mode_convention(b954):
    m, n, B = b954.modes["m"], b954.modes["n"], b954.modes["B"]
    assert len(set(zip(m.tolist(), n.tolist()))) == len(m) == (b954.modes["m"].max() + 1) * (2 * np.abs(n).max() + 1) \
        - np.abs(n).max()
    assert np.all(n[m == 0] >= 0)
    assert np.all(np.diff(np.abs(B)) <= 0)                     # largest first
    top = b954.modes[1]
    assert (top["m"], top["n"]) == (1, 0) and top["B"] < 0       # weaker outboard, where theta_B = 0
    assert b954.nfp == 2 and b954.helicity == Helicity(1, 0)
    assert abs(b954.iota - QUASR_954["iota"]) < 1e-11            # the solver's theta there runs counter-clockwise


def test_print_reads_in_ten_seconds(b954):
    s = str(b954)
    lines = s.splitlines()
    assert len(lines) <= 9
    for part in ("mean field", "quasi-axisymmetric", "NonQuasiSymmetricRatio", "(1, +0)", "xm_b = m", "within"):
        assert part in s, part


def test_seconds(b954):
    assert b954.wall_s < 15


@pytest.mark.parametrize("design", ["2405145", "1856224"])
def test_quasr_qs_error_on_designs_with_one_stored_surface(design):
    """QUASR's qs_error is the mean of NonQuasiSymmetricRatio over a design's stored surfaces. With one stored
    surface it is this number on that surface, computed by QUASR on its own surface (whose field crosses it by
    about 1e-4 of |B|): agreement to 2 %, and the same helicity."""
    meta = metadata(design)
    assert meta["Nsurfaces"] == 1
    res = torux.solve(inputs(design))
    assert res.status == "surface"
    b = boozer(res)
    assert abs(b.quasisymmetry / meta["qs_error"] - 1) < 0.02, (b.quasisymmetry, meta["qs_error"])
    assert (b.helicity.m, abs(b.helicity.n)) == (1, int(meta["helicity"]))


def test_refusals(solved_954):
    from torux.api import Result
    with pytest.raises(ValueError, match="no verified surface"):
        boozer(Result("no_surface", 1.0, 0.0))
    with pytest.raises(ValueError, match="no verified surface"):
        boozer(Result("surface_unverified", 1.0, 0.0, surface=solved_954.surface))

    class General:
        rs = None
    with pytest.raises(ValueError, match="not stellarator-symmetric"):
        boozer(Result("surface", 1.0, 0.0, surface=General()))


# ------------------------------------------------------------------------------------------------ against SIMSOPT

# two quasi-axisymmetric and two quasi-helical QUASR designs that are surfaces for both codes in
# results/compare300.csv, SIMSOPT's at 1e-14 there
DESIGNS = ["0605060", "0190703", "2405145", "1856224"]


def _simsopt():
    pytest.importorskip("simsopt")
    if COMPARE not in sys.path:
        sys.path.insert(0, COMPARE)
    import simsopt_common
    return simsopt_common


def simsopt_boozer(design, levels=(10, 14, 18, 22)):
    """SIMSOPT's BoozerExact surface for the design's coils and requested flux, as results/compare300 runs it,
    climbing mpol = ntor = levels. The level with the smallest off-grid Boozer residual."""
    C = _simsopt()
    from simsopt.field import BiotSavart
    from simsopt.geo import BoozerSurface, ToroidalFlux
    rec = C.load_design(design)
    nfp = int(rec["nfp"])
    probe = C.start_xyz(rec, np.linspace(0, 1 / nfp, 24, endpoint=False), np.linspace(0, 1, 24, endpoint=False))
    nq, _ = C.coil_quadrature(rec, probe.gamma().reshape(-1, 3))
    coils = C.simsopt_coils(rec, nq)
    bs, bs_tf, bs_d = BiotSavart(coils), BiotSavart(coils), BiotSavart(coils)
    G, iota = C.boozer_estimates(probe, bs)
    Bmean = float(np.mean(np.linalg.norm(bs.B(), axis=1)))
    loop0 = C.start_xyz(rec, [0.0], np.linspace(0, 1, 512, endpoint=False))
    target = float(np.sign(ToroidalFlux(loop0, bs_tf).J())) * float(rec["requested_toroidal_flux_Wb"])
    cur, best = C.start_xyz(rec, [0.0], [0.0]), None
    for M in levels:
        s = C.lift_xyz(cur, M, M, np.linspace(0, 1 / (2 * nfp), M + 1, endpoint=False),
                       np.linspace(0, 1, 2 * M + 1, endpoint=False))
        lab = C.xyz_surface(nfp, True, M, M, [0.0], np.linspace(0, 1, max(256, 8 * M), endpoint=False), dofs=s.dofs)
        booz = BoozerSurface(bs, s, ToroidalFlux(lab, bs_tf), target)
        tol = 1e-14 * abs(G) * Bmean * np.sqrt(6 * M * M + 6 * M + 3)
        r = booz.solve_residual_equation_exactly_newton(tol=tol, maxiter=30 if best is None else 10, iota=iota, G=G)
        if not r["success"]:
            continue
        iota, G = float(r["iota"]), float(r["G"])
        h = C.boozer_offgrid(s, iota, G, bs_d, max(64, 4 * M + 2))
        if best is None or h["diag_boozer_rel_max"] < best["residual"]:
            best = dict(surface=s, iota=iota, G=G, residual=h["diag_boozer_rel_max"], M=M, label=booz.label,
                        target=target)
        cur = s
    assert best is not None, "SIMSOPT's BoozerExact converged at no level"
    best["bs"] = BiotSavart(coils)
    return best


def _xyz(s, qp, qt, mpol=None, ntor=None, dofs=None):
    from simsopt.geo import SurfaceXYZTensorFourier
    return SurfaceXYZTensorFourier(mpol=mpol or s.mpol, ntor=ntor or s.ntor, stellsym=True, nfp=s.nfp,
                                   quadpoints_phi=np.asarray(qp, float), quadpoints_theta=np.asarray(qt, float),
                                   dofs=s.dofs if dofs is None else dofs)


def simsopt_orientation(s):
    """(shift, sign) of SIMSOPT's theta against the stated convention (theta_B = sign (2 pi theta - shift)), from
    its own geometry: the outboard point of the plane phi = 0 and the counter-clockwise direction there."""
    g = _xyz(s, [0.0], [0.0, 0.5, 1e-4, 0.5 + 1e-4]).gamma()[0]
    assert np.max(np.abs(g[:2, 1:])) < 1e-12                       # both on the symmetry line y = z = 0
    assert _xyz(s, [1e-4], [0.0]).gamma()[0, 0, 1] > 0             # phi runs with the cylindrical angle
    R = np.hypot(g[:, 0], g[:, 1])
    out = 0 if R[0] > R[1] else 1
    return out * np.pi, (1 if g[2 + out, 2] > 0 else -1)


def simsopt_harmonics(sim, Mb, Nb, nth, nph):
    """|B| on SIMSOPT's surface (its own BiotSavart), Fourier analysed in SIMSOPT's angles (its surface's
    parameters are the Boozer angles) and written in the stated convention: B[m, n + Nb]."""
    s, bs = sim["surface"], sim["bs"]
    e = _xyz(s, np.arange(nph) / (nph * s.nfp), np.arange(nth) / nth)
    bs.set_points(np.ascontiguousarray(e.gamma().reshape(-1, 3)))
    F = np.fft.fft2(np.linalg.norm(bs.B(), axis=1).reshape(nph, nth).T) / (nth * nph)
    shift, sign = simsopt_orientation(s)
    out = np.zeros((Mb + 1, 2 * Nb + 1))
    for m in range(Mb + 1):
        for j, n in enumerate(range(-Nb, Nb + 1)):
            if m == 0 and n < 0:
                continue
            v = F[m, (-sign * n) % nph]                            # e^{i m theta} e^{-i (sign n) psi}
            out[m, j] = (1.0 if m == 0 and n == 0 else 2.0) * v.real * ((-1) ** m if shift else 1.0)
    return out


def helical_frame(s, N):
    """x'(phi, theta') = x(phi, theta' + N nfp phi) as a SurfaceXYZTensorFourier, fitted exactly (linear least
    squares on SIMSOPT's own d gamma / d coefficients): QUASR's rotated frame, in which a quasi-helical |B| does not
    depend on phi.  Returns the surface and its largest distance from the target points."""
    ntor = s.ntor + s.mpol * abs(N)
    nph, nth = 2 * ntor + 2, 2 * s.mpol + 2
    qp, qt = np.arange(nph) / (nph * s.nfp), np.arange(nth) / nth
    from simsopt.geo import SurfaceXYZTensorFourier
    new = SurfaceXYZTensorFourier(mpol=s.mpol, ntor=ntor, stellsym=True, nfp=s.nfp, quadpoints_phi=qp,
                                  quadpoints_theta=qt)
    tgt = np.stack([_xyz(s, [p], (qt + N * s.nfp * p) % 1.0).gamma()[0] for p in qp])
    x, *_ = np.linalg.lstsq(new.dgamma_by_dcoeff().reshape(-1, new.x.size), tgt.ravel(), rcond=None)
    new.x = x
    return new, float(np.max(np.abs(new.gamma() - tgt)))


def simsopt_ratio(sim, s, sDIM):
    """SIMSOPT's own NonQuasiSymmetricRatio.J() on surface s, for a BoozerSurface already solved (its iota and G
    given). Its derivative, which needs the solve's factorisation and gigabytes, is not computed."""
    from simsopt.geo import BoozerSurface, NonQuasiSymmetricRatio
    booz = BoozerSurface(sim["bs"], s, sim["label"], sim["target"])
    booz.need_to_run_code = False
    booz.res = dict(iota=sim["iota"], G=sim["G"])
    q = NonQuasiSymmetricRatio(booz, sim["bs"], sDIM=sDIM)
    try:
        q.J()
    except KeyError:              # compute() sets J, then reaches for the factorisation the derivative needs
        pass
    return float(q._J)


@pytest.mark.slow
@pytest.mark.parametrize("design", DESIGNS)
def test_agrees_with_simsopt_boozer_surface(design):
    """Mode by mode, the harmonics of |B| on SIMSOPT's own Boozer surface for the same coils and flux, and
    SIMSOPT's NonQuasiSymmetricRatio there (in QUASR's rotated frame for a quasi-helical design), agree with
    torux's to the accuracy of SIMSOPT's surface, which its own off-grid Boozer residual states."""
    res = torux.solve(inputs(design))
    assert res.status == "surface"
    b = boozer(res)
    del res
    gc.collect()
    sim = simsopt_boozer(design)
    eps = sim["residual"]
    Mb, Nb = int(b.modes["m"].max()), int(np.abs(b.modes["n"]).max())
    H = simsopt_harmonics(sim, Mb, Nb, 4 * Mb + 3, 4 * Nb + 3)
    H2 = simsopt_harmonics(sim, Mb, Nb, 6 * Mb + 5, 6 * Nb + 5)
    assert np.max(np.abs(H - H2)) < 1e-14 * H[0, Nb]                 # SIMSOPT's side converged in its grid
    mine = np.zeros_like(H)
    for r in b.modes:
        mine[r["m"], r["n"] + Nb] = r["B"]
    B00 = _mean(b)
    worst = float(np.max(np.abs(H - mine)))
    hel = b.helicity
    if hel.n == 0:
        s, fit = sim["surface"], 0.0
    else:
        s, fit = helical_frame(sim["surface"], simsopt_orientation(sim["surface"])[1] * hel.n)
    sdim = 2 * max(s.mpol, s.ntor)
    J, J2 = simsopt_ratio(sim, s, sdim), simsopt_ratio(sim, s, 3 * sdim // 2)
    meta = metadata(design)
    print(f"\n{design}  QUASR helicity {int(meta['helicity'])}, torux {hel}. SIMSOPT BoozerExact mpol = ntor = "
          f"{sim['M']}, off-grid Boozer residual {eps:.2e}. Largest |difference| over all {len(b.modes)} harmonics "
          f"{worst:.2e} T = {worst / B00:.2e} B_00")
    for r in b.modes[:10]:
        v = H[r["m"], r["n"] + Nb]
        print(f"  ({r['m']:2d}, {r['n']:+3d})  torux {r['B']:+.12e}  SIMSOPT {v:+.12e}  difference {r['B'] - v:+.1e}")
    print(f"  NonQuasiSymmetricRatio  torux {b.quasisymmetry:.10e}  SIMSOPT {J:.10e} ({J2:.10e} on a finer grid)"
          f"  relative difference {b.quasisymmetry / J - 1:+.1e}, helical frame fitted to {fit:.0e} m,"
          f" QUASR qs_error {meta['qs_error']:.4e}")
    assert worst <= eps * B00, (worst, eps)
    assert abs(J2 / J - 1) < 1e-9 and fit < 1e-12
    # the ratio's sensitivity to |B|: |d J| <= 2 sqrt(J) max |d B| / B_QS
    assert abs(b.quasisymmetry - J) <= 2 * np.sqrt(J) * eps + 1e-14, (b.quasisymmetry, J)
    assert b.helicity.m == 1 and abs(b.helicity.n) == int(meta["helicity"])
