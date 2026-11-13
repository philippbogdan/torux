"""answer.system and answer.vjp: torux's equations at any surface, and their derivatives in the coils and the flux."""
import copy

import numpy as np
import pytest

import torux
from torux.field import FieldEngine
from torux.gradients import _vjp_source, fourier_part_data
from torux.torus import TorusProblem


def cut(t, M, N):
    """The solved surface's coefficients cut or padded to M, N."""
    rc, zs = np.zeros((M + 1, 2 * N + 1)), np.zeros((M + 1, 2 * N + 1))
    m, n = min(M, t.M), min(N, t.N)
    rc[:m + 1, N - n:N + n + 1] = t.rc[:m + 1, t.N - n:t.N + n + 1]
    zs[:m + 1, N - n:N + n + 1] = t.zs[:m + 1, t.N - n:t.N + n + 1]
    return rc, zs


def unknowns(rc, zs, iota):
    """u in SIMSOPT's order of a stellarator symmetric SurfaceRZFourier's dofs, then iota."""
    N = (rc.shape[1] - 1) // 2
    return np.concatenate([rc[0, N:], rc[1:].ravel(), zs[0, N + 1:], zs[1:].ravel(), [iota]])


def surface_of(u, M, N):
    rc, zs = np.zeros((M + 1, 2 * N + 1)), np.zeros((M + 1, 2 * N + 1))
    k = N + 1 + M * (2 * N + 1)
    rc[0, N:], rc[1:] = u[:N + 1], u[N + 1:k].reshape(M, 2 * N + 1)
    zs[0, N + 1:], zs[1:] = u[k:k + N], u[k + N:-1].reshape(M, 2 * N + 1)
    return rc, zs, u[-1]


def with_source(answer, src):
    """The answer with the field of another source (the same surface)."""
    a = copy.copy(answer)
    a._engine = FieldEngine(src, dmin=answer._engine.dmin)
    return a


def fourier_source(src, coef, current):
    return torux.Source(src.id, [torux.FourierCurves(coef, current)], nfp=src.nfp, stellsym=src.stellsym)


def test_the_equations_vanish_on_the_solved_surface_and_are_torux_own(solved_954):
    a, t = solved_954, solved_954.surface
    F, J = a.system(t.rc, t.zs, t.iota)
    n = 2 * t.M * (2 * t.N + 1) + 2 * t.N + 2
    assert F.shape == (n,) and J.shape == (n, n) and F.dtype == J.dtype == np.float64
    assert np.max(np.abs(F)) < 1e-14 * t.rc[0, t.N]
    u = unknowns(t.rc, t.zs, t.iota)
    prob = TorusProblem(a._engine, t.nfp, t.M, t.N, phi0=t.phi0)
    F0, J0, _, _ = prob.system(u, t.flux, u[0] / abs(t.flux))
    assert np.array_equal(F, F0) and np.array_equal(J, J0)


def central(a, u, M, N, j, h=1e-6):
    up, down = u.copy(), u.copy()
    up[j] += h
    down[j] -= h
    return (a.system(*surface_of(up, M, N))[0] - a.system(*surface_of(down, M, N))[0]) / (2 * h)


def test_the_jacobian_is_the_derivative_of_the_residual(solved_954):
    a, t = solved_954, solved_954.surface
    M, N = 5, 4                                           # every column, on torux's coarsest grid
    u = unknowns(*cut(t, M, N), t.iota)
    u[1] += 1e-3
    F, J = a.system(*surface_of(u, M, N))
    assert F.shape == (2 * M * (2 * N + 1) + 2 * N + 2,) and np.max(np.abs(F)) > 1e-4
    fd = np.stack([central(a, u, M, N, j) for j in range(len(u))], axis=1)
    fd[-1, 0] -= F[-1] / u[0]                            # the flux row's factor rc[0, N] / |flux|, held fixed in J
    assert np.max(np.abs(fd[:-1] - J[:-1])) <= 1e-9 * np.max(np.abs(J))
    assert np.max(np.abs(fd[-1] - J[-1])) <= 1e-5 * np.max(np.abs(J[-1]))

    M, N = 12, 10                                         # a finer grid: the flux row to finite-difference precision
    u = unknowns(*cut(t, M, N), t.iota)
    u[1] += 1e-3
    F, J = a.system(*surface_of(u, M, N))
    for j in (0, 1, 7, 300, len(u) - 2, len(u) - 1):
        fd = central(a, u, M, N, j)
        if j == 0:
            assert abs(fd[-1] - J[-1, 0]) > 1e-6 * np.max(np.abs(J[-1]))
            fd[-1] -= F[-1] / u[0]
        assert np.max(np.abs(fd - J[:, j])) <= 1e-9 * np.max(np.abs(J))


def test_any_resolution_and_the_unused_entries(solved_954):
    a, t = solved_954, solved_954.surface
    for M, N in [(t.M + 2, t.N + 1), (3, 0), (1, 2)]:
        rc, zs = cut(t, M, N)
        F, J = a.system(rc, zs, t.iota)
        n = 2 * M * (2 * N + 1) + 2 * N + 2
        assert F.shape == (n,) and J.shape == (n, n) and np.all(np.isfinite(J))
        if M > t.M:
            assert np.max(np.abs(F)) < 1e-14 * t.rc[0, t.N]       # the solved surface, padded with zeros
        rc2, zs2 = rc.copy(), zs.copy()
        rc2[0, :N], zs2[0, :N + 1] = 1.0, 1.0                   # not unknowns: ignored
        assert np.array_equal(a.system(rc2, zs2, t.iota)[0], F)


def test_the_product_in_the_coils_is_torux_own_at_the_solved_resolution(solved_954):
    a, t = solved_954, solved_954.surface
    u = unknowns(t.rc, t.zs, t.iota)
    lam = np.random.default_rng(1).standard_normal(len(u))
    v = a.vjp(t.rc, t.zs, t.iota, lam)
    prob = TorusProblem(a._engine, t.nfp, t.M, t.N, phi0=t.phi0)
    o = _vjp_source(a._engine, prob, u, t.flux, u[0] / abs(t.flux), lam)
    coef, current = fourier_part_data(a._engine.source)
    assert np.array_equal(v["coef"], coef) and np.array_equal(v["current"], current)
    assert np.array_equal(v["d_coef"], -o["d_coef"]) and np.array_equal(v["d_current"], -o["d_current"])
    assert v["d_coef"].shape == v["coef"].shape and v["d_current"].shape == v["current"].shape


@pytest.fixture(scope="module")
def with_a_coil_off(quasr_954):
    """QUASR 954 with a fifth coil of zero current in third place: the same field, the same surface."""
    p = quasr_954.parts[0]
    extra = p.coef[0].copy()
    extra[:, 0] += [0.05, -0.03, 0.02]
    coef = np.concatenate([p.coef[:2], extra[None], p.coef[2:]])
    current = np.concatenate([p.current[:2], [0.0], p.current[2:]])
    src = fourier_source(quasr_954, coef, current)
    a = torux.solve(src, flux=quasr_954.request["flux_Wb"], verify=False)
    assert a.status == "surface"
    return src, a


def test_the_product_matches_central_differences_with_a_coil_of_zero_current(with_a_coil_off):
    src, a = with_a_coil_off
    t = a.surface
    M, N = 12, 10
    rc, zs = cut(t, M, N)
    lam = np.random.default_rng(2).standard_normal(2 * M * (2 * N + 1) + 2 * N + 2)
    v = a.vjp(rc, zs, t.iota, lam)
    p = src.parts[0]
    assert np.array_equal(v["coef"], p.coef) and np.array_equal(v["current"], p.current)

    def value(coef=p.coef, current=p.current, answer=None):
        b = answer if answer is not None else with_source(a, fourier_source(src, coef, current))
        return float(lam @ b.system(rc, zs, t.iota)[0])

    scale = np.max(np.abs(p.current))
    for k in (0, 2, 4):                                    # the coil of zero current is the third
        h = 1e-6 * scale
        up, down = p.current.copy(), p.current.copy()
        up[k] += h
        down[k] -= h
        fd = (value(current=up) - value(current=down)) / (2 * h)
        assert abs(fd - v["d_current"][k]) <= 1e-6 * abs(fd)
    assert v["d_current"][2] != 0.0
    for idx in [(0, 0, 2), (3, 2, 5), (2, 1, 1)]:
        h = 1e-6
        up, down = p.coef.copy(), p.coef.copy()
        up[idx] += h
        down[idx] -= h
        fd = (value(coef=up) - value(coef=down)) / (2 * h)
        assert abs(fd - v["d_coef"][idx]) <= 1e-6 * abs(fd) + 1e-9 * np.max(np.abs(v["d_coef"]))
    assert np.all(v["d_coef"][2] == 0.0) and np.any(v["d_coef"][0] != 0.0)
    h = 1e-6 * abs(t.flux)
    shifted = []
    for f in (t.flux + h, t.flux - h):
        b = copy.copy(a)
        b.surface = copy.copy(t)
        b.surface.flux = f
        shifted.append(value(answer=b))
    fd = (shifted[0] - shifted[1]) / (2 * h)
    assert abs(fd - v["d_flux"]) <= 1e-6 * abs(fd)


def test_without_the_source_a_surface_or_fourier_coils_the_calls_say_why(solved_954, tmp_path):
    a, t = solved_954, solved_954.surface
    lam = np.zeros(2 * t.M * (2 * t.N + 1) + 2 * t.N + 2)
    a.save(tmp_path / "a.json")
    read = torux.Answer.read(tmp_path / "a.json")
    for call in (lambda b: b.system(t.rc, t.zs, t.iota), lambda b: b.vjp(t.rc, t.zs, t.iota, lam)):
        with pytest.raises(ValueError, match="does not hold"):
            call(read)
        none = copy.copy(a)
        none.surface = None
        with pytest.raises(ValueError, match="holds none"):
            call(none)
    with pytest.raises(ValueError, match="one shape"):
        a.system(t.rc[:, 1:], t.zs[:, 1:], t.iota)
    with pytest.raises(ValueError, match="one shape"):
        a.vjp(t.rc[:1], t.zs[:1], t.iota, lam)
    with pytest.raises(ValueError, match="one entry per equation"):
        a.vjp(t.rc, t.zs, t.iota, lam[:-1])
    p = a._engine.source.parts[0]
    src = a._engine.source
    rings = torux.Source(src.id, [p, torux.Rings(np.array([1.0]), np.array([0.0]), np.array([0.0]))],
                         nfp=src.nfp, stellsym=src.stellsym)
    with pytest.raises(ValueError, match="rings parts"):
        with_source(a, rings).vjp(t.rc, t.zs, t.iota, lam)


# ---------------------------------------------------------------------------------------------- without stellarator symmetry
def general_cut(t, M, N):
    """The solved surface's four coefficient arrays, in the series of phi itself, cut or padded to M, N."""
    from torux.torus_general import general_arrays
    out = []
    for a in general_arrays(t):
        b = np.zeros((M + 1, 2 * N + 1))
        m, n = min(M, t.M), min(N, t.N)
        b[:m + 1, N - n:N + n + 1] = a[:m + 1, t.N - n:t.N + n + 1]
        out.append(b)
    return out


def general_unknowns(rc, rs, zc, zs, iota):
    """u in SIMSOPT's order of a SurfaceRZFourier's dofs without stellarator symmetry, then iota."""
    N = (rc.shape[1] - 1) // 2
    return np.concatenate([rc[0, N:], rc[1:].ravel(), rs[0, N + 1:], rs[1:].ravel(),
                           zc[0, N:], zc[1:].ravel(), zs[0, N + 1:], zs[1:].ravel(), [iota]])


def general_surface_of(u, M, N):
    out, k = [], 0
    for first in (N, N + 1, N, N + 1):                  # rc and zc keep (0, 0), rs and zs start at n = 1
        a = np.zeros((M + 1, 2 * N + 1))
        a[0, first:] = u[k:k + 2 * N + 1 - first]
        k += 2 * N + 1 - first
        a[1:] = u[k:k + M * (2 * N + 1)].reshape(M, 2 * N + 1)
        k += M * (2 * N + 1)
        out.append(a)
    return (*out, u[-1])


def general_size(M, N):
    return 2 * (2 * M + 1) * (2 * N + 1) + 1


def test_the_unknowns_without_symmetry_are_a_simsopt_surfaces_dofs():
    """SIMSOPT 1.11.1's own order of a SurfaceRZFourier's dofs without stellarator symmetry."""
    geo = pytest.importorskip("simsopt.geo")
    M, N = 3, 2
    s = geo.SurfaceRZFourier(nfp=2, stellsym=False, mpol=M, ntor=N)
    s.x = np.random.default_rng(3).standard_normal(s.x.size)
    assert s.x.size == general_size(M, N) - 1
    assert np.array_equal(general_unknowns(s.rc, s.rs, s.zc, s.zs, 0.3)[:-1], s.get_dofs())
    assert np.array_equal(general_surface_of(general_unknowns(s.rc, s.rs, s.zc, s.zs, 0.3), M, N)[0], s.rc)


def test_without_symmetry_the_equations_vanish_on_the_solved_surface(solved_954_mirrored):
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    assert a.status == "surface" and t.asymmetry() > 1e-3            # rs and zc of a torus without the symmetry
    from torux.torus_general import general_arrays
    F, J = a.system_general(*general_arrays(t), t.iota)
    n = general_size(t.M, t.N)
    assert F.shape == (n,) and J.shape == (n, n) and F.dtype == J.dtype == np.float64
    assert np.max(np.abs(F)) < 1e-14 * t.rc[0, t.N]


def test_without_symmetry_the_equations_are_the_solvers_with_its_multiplier_solved_for(solved_954_mirrored):
    """The projections in F and their Jacobian are those of torux's own Newton system without symmetry
    (torux.torus_general, at its unfolding multiplier zero), less their share of the pivot's: the multiplier solved
    for. The phase condition and the flux row are the solver's. The same to rounding: FFTs here, products there."""
    from torux.torus_general import GeneralEquations, GeneralTorus, GeneralTorusProblem, general_arrays, phase_row
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    M, N = 12, 10
    arrays = general_cut(t, M, N)
    u = general_unknowns(*arrays, t.iota)
    u[1] += 1e-3                                          # off the solution, where every term counts
    F, J = a.system_general(*general_surface_of(u, M, N))
    ref = GeneralTorus(*arrays, t.iota, t.flux, M, N, t.nfp)
    prob = GeneralTorusProblem(a._engine, t.nfp, M, N)
    off = GeneralTorus(*general_surface_of(u, M, N)[:4], t.iota, t.flux, M, N, t.nfp)
    Fs, Js, _, _ = prob.system(prob.pack(off), t.flux, u[0] / abs(t.flux), phase_row(prob.basis, ref))
    Fs, Js = np.asarray(Fs), np.asarray(Js)
    eq = GeneralEquations(a._engine, t.nfp, M, N, general_arrays(t))
    n2, p = len(u) - 1, eq.pivot
    rows = np.arange(n2) != p
    assert np.max(np.abs(F[:n2][rows] - (Fs[:n2] - eq.share * Fs[p])[rows])) < 1e-15 * u[0]
    assert np.max(np.abs(J[:n2][rows] - (Js[:n2, :n2 + 1] - np.outer(eq.share, Js[p, :n2 + 1]))[rows])) \
        <= 1e-12 * np.max(np.abs(J))
    assert abs(F[p] - Fs[n2 + 1]) < 1e-15 * u[0] and np.max(np.abs(J[p, :n2] - Js[n2 + 1, :n2])) < 1e-15
    assert abs(F[-1] - Fs[n2]) < 1e-14 * u[0] and np.max(np.abs(J[-1] - Js[n2, :n2 + 1])) <= 1e-12 * np.max(np.abs(J[-1]))
    assert np.max(np.abs(F[:n2])) > 1e-4


def test_on_a_symmetric_surface_the_general_equations_are_the_symmetric_ones(solved_954):
    """A stellarator-symmetric answer through system_general: its projections of R's residual on sines and of Z's on
    cosines, and their Jacobian in rc, zs and iota, are answer.system's, the others zero to rounding."""
    a, t = solved_954, solved_954.surface
    M, N = 12, 10
    rc, zs = cut(t, M, N)
    z = np.zeros_like(rc)
    Fg, Jg = a.system_general(rc, z, z, zs, t.iota)
    Fs, Js = a.system(rc, zs, t.iota)
    nc, ns = N + 1 + M * (2 * N + 1), N + M * (2 * N + 1)
    rows = np.r_[nc:nc + ns, nc + ns:2 * nc + ns, Fg.size - 1]        # P_s E_R (like rs), P_c E_Z (like zc), flux
    cols = np.r_[0:nc, 2 * nc + ns:2 * nc + 2 * ns, Fg.size - 1]      # rc, zs, iota
    assert np.max(np.abs(Fg[rows] - Fs)) < 1e-14 * t.rc[0, t.N]
    assert np.max(np.abs(Jg[np.ix_(rows, cols)] - Js)) <= 1e-12 * np.max(np.abs(Js))
    rest = np.setdiff1d(np.arange(Fg.size), rows)                     # P_c E_R, P_s E_Z and the phase condition
    assert np.max(np.abs(Fg[rest])) < 1e-14 * t.rc[0, t.N] and np.max(np.abs(Fs[:-1])) > 1e-9


def test_without_symmetry_the_jacobian_is_the_derivative_of_the_residual(solved_954_mirrored):
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    M, N = 5, 4                                           # every column, on torux's coarsest grid
    u = general_unknowns(*general_cut(t, M, N), t.iota)
    u[1] += 1e-3
    F, J = a.system_general(*general_surface_of(u, M, N))
    assert F.shape == (general_size(M, N),) and np.max(np.abs(F)) > 1e-4

    def central(j, h=1e-6):
        up, down = u.copy(), u.copy()
        up[j] += h
        down[j] -= h
        return (a.system_general(*general_surface_of(up, M, N))[0]
                - a.system_general(*general_surface_of(down, M, N))[0]) / (2 * h)
    fd = np.stack([central(j) for j in range(len(u))], axis=1)
    fd[-1, 0] -= F[-1] / u[0]                            # the flux row's factor rc[0, N] / |flux|, held fixed in J
    assert np.max(np.abs(fd[:-1] - J[:-1])) <= 1e-9 * np.max(np.abs(J))
    assert np.max(np.abs(fd[-1] - J[-1])) <= 1e-5 * np.max(np.abs(J[-1]))

    M, N = 12, 10                                         # a finer grid: the flux row to finite-difference precision
    u = general_unknowns(*general_cut(t, M, N), t.iota)
    u[1] += 1e-3
    F, J = a.system_general(*general_surface_of(u, M, N))
    for j in (0, 1, 7, 300, len(u) - 2, len(u) - 1):
        fd = central(j)
        if j == 0:
            assert abs(fd[-1] - J[-1, 0]) > 1e-6 * np.max(np.abs(J[-1]))
            fd[-1] -= F[-1] / u[0]
        assert np.max(np.abs(fd - J[:, j])) <= 1e-9 * np.max(np.abs(J))


def test_without_symmetry_any_resolution_and_the_unused_entries(solved_954_mirrored):
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    for M, N in [(t.M + 2, t.N + 1), (3, 0), (1, 2)]:
        arrays = general_cut(t, M, N)
        F, J = a.system_general(*arrays, t.iota)
        n = general_size(M, N)
        assert F.shape == (n,) and J.shape == (n, n) and np.all(np.isfinite(J))
        if M > t.M:
            assert np.max(np.abs(F)) < 1e-14 * t.rc[0, t.N]       # the solved surface, padded with zeros
        rc, rs, zc, zs = (x.copy() for x in arrays)
        rc[0, :N], rs[0, :N + 1], zc[0, :N], zs[0, :N + 1] = 1.0, 1.0, 1.0, 1.0   # not unknowns: ignored
        assert np.array_equal(a.system_general(rc, rs, zc, zs, t.iota)[0], F)


def test_without_symmetry_the_product_in_the_coils_is_torux_own_at_the_solved_resolution(solved_954_mirrored):
    from torux.gradients import _vjp_source_general
    from torux.torus_general import GeneralEquations, general_arrays
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    arrays = general_arrays(t)
    u = general_unknowns(*arrays, t.iota)
    lam = np.random.default_rng(1).standard_normal(len(u))
    v = a.vjp_general(*arrays, t.iota, lam)
    eq = GeneralEquations(a._engine, t.nfp, t.M, t.N, arrays)
    o = _vjp_source_general(a._engine, eq, u, t.flux, u[0] / abs(t.flux), lam)
    coef, current = fourier_part_data(a._engine.source)
    assert np.array_equal(v["coef"], coef) and np.array_equal(v["current"], current)
    assert np.array_equal(v["d_coef"], -o["d_coef"]) and np.array_equal(v["d_current"], -o["d_current"])
    assert v["d_coef"].shape == v["coef"].shape and v["d_current"].shape == v["current"].shape


@pytest.fixture(scope="module")
def mirrored_with_a_coil_off(quasr_954_mirrored):
    """The mirrored QUASR 954 with a fifth coil of zero current in third place: the same field, the same surface."""
    p = quasr_954_mirrored.parts[0]
    extra = p.coef[0].copy()
    extra[:, 0] += [0.05, -0.03, 0.02]
    coef = np.concatenate([p.coef[:2], extra[None], p.coef[2:]])
    current = np.concatenate([p.current[:2], [0.0], p.current[2:]])
    src = fourier_source(quasr_954_mirrored, coef, current)
    a = torux.solve(src, flux=0.05, verify=False)
    assert a.status == "surface"
    return src, a


def test_without_symmetry_the_product_matches_central_differences_with_a_coil_of_zero_current(mirrored_with_a_coil_off):
    src, a = mirrored_with_a_coil_off
    t = a.surface
    M, N = 12, 10
    arrays = general_cut(t, M, N)
    lam = np.random.default_rng(2).standard_normal(general_size(M, N))
    v = a.vjp_general(*arrays, t.iota, lam)
    p = src.parts[0]
    assert np.array_equal(v["coef"], p.coef) and np.array_equal(v["current"], p.current)

    def value(coef=p.coef, current=p.current, answer=None):
        b = answer if answer is not None else with_source(a, fourier_source(src, coef, current))
        return float(lam @ b.system_general(*arrays, t.iota)[0])

    scale = np.max(np.abs(p.current))
    for k in (0, 2, 4):                                    # the coil of zero current is the third
        h = 1e-6 * scale
        up, down = p.current.copy(), p.current.copy()
        up[k] += h
        down[k] -= h
        fd = (value(current=up) - value(current=down)) / (2 * h)
        assert abs(fd - v["d_current"][k]) <= 1e-6 * abs(fd)
    assert v["d_current"][2] != 0.0
    for idx in [(0, 0, 2), (3, 2, 5), (2, 1, 1)]:
        h = 1e-6
        up, down = p.coef.copy(), p.coef.copy()
        up[idx] += h
        down[idx] -= h
        fd = (value(coef=up) - value(coef=down)) / (2 * h)
        assert abs(fd - v["d_coef"][idx]) <= 1e-6 * abs(fd) + 1e-9 * np.max(np.abs(v["d_coef"]))
    assert np.all(v["d_coef"][2] == 0.0) and np.any(v["d_coef"][0] != 0.0)
    h = 1e-6 * abs(t.flux)
    shifted = []
    for f in (t.flux + h, t.flux - h):
        b = copy.copy(a)
        b.surface = copy.copy(t)
        b.surface.flux = f
        shifted.append(value(answer=b))
    fd = (shifted[0] - shifted[1]) / (2 * h)
    assert abs(fd - v["d_flux"]) <= 1e-6 * abs(fd)


def test_without_symmetry_without_the_source_a_surface_or_fourier_coils_the_calls_say_why(solved_954_mirrored,
                                                                                            tmp_path):
    from torux.torus_general import general_arrays
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    arrays = general_arrays(t)
    lam = np.zeros(general_size(t.M, t.N))
    a.save(tmp_path / "a.json")
    read = torux.Answer.read(tmp_path / "a.json")
    for call in (lambda b: b.system_general(*arrays, t.iota), lambda b: b.vjp_general(*arrays, t.iota, lam)):
        with pytest.raises(ValueError, match="does not hold"):
            call(read)
        none = copy.copy(a)
        none.surface = None
        with pytest.raises(ValueError, match="holds none"):
            call(none)
    with pytest.raises(ValueError, match="one shape"):
        a.system_general(arrays[0], arrays[1][:, 1:], arrays[2], arrays[3], t.iota)
    with pytest.raises(ValueError, match="one shape"):
        a.vjp_general(*(x[:1] for x in arrays), t.iota, lam)
    with pytest.raises(ValueError, match="one entry per equation"):
        a.vjp_general(*arrays, t.iota, lam[:-1])
    with pytest.raises(ValueError, match="stellarator symmetric"):
        a.system(t.rc, t.zs, t.iota)                       # the symmetric pair refuses a surface without the symmetry
    p = a._engine.source.parts[0]
    src = a._engine.source
    rings = torux.Source(src.id, [p, torux.Rings(np.array([1.0]), np.array([0.0]), np.array([0.0]))],
                         nfp=src.nfp, stellsym=src.stellsym)
    with pytest.raises(ValueError, match="rings parts"):
        with_source(a, rings).vjp_general(*arrays, t.iota, lam)


def test_without_symmetry_the_product_through_the_engine_is_the_kernels(solved_954_mirrored):
    """The product in the source through the differentiable engine, a slab of points at a time (what parts other than
    Fourier coils take), and by the filament kernels: the same to rounding."""
    from torux.gradients import _vjp_source_general
    from torux.torus_general import GeneralEquations, general_arrays
    a, t = solved_954_mirrored, solved_954_mirrored.surface
    M, N = 12, 10
    arrays = general_cut(t, M, N)
    u = general_unknowns(*arrays, t.iota)
    lam = np.random.default_rng(3).standard_normal(len(u))
    eq = GeneralEquations(a._engine, t.nfp, M, N, general_arrays(t))
    fast = _vjp_source_general(a._engine, eq, u, t.flux, u[0] / abs(t.flux), lam)
    slow = _vjp_source_general(a._engine, eq, u, t.flux, u[0] / abs(t.flux), lam, fast=False, slab=500)
    for k in ("d_coef", "d_current"):
        assert np.max(np.abs(slow[k] - fast[k])) <= 1e-11 * np.max(np.abs(fast[k])), k
    assert abs(slow["d_flux"] - fast["d_flux"]) <= 1e-12 * abs(fast["d_flux"])


def test_the_series_in_phi_itself_is_the_surface():
    """general_arrays moves phi0 into the coefficients: the points of the series in phi itself are the torus's own,
    for a stellarator-symmetric torus about phi0 = pi / nfp and for a torus without symmetry at phi0 = 0.3."""
    from torux.torus import TorusState
    from torux.torus_general import GeneralTorus, general_arrays, torus_points
    from torux.verify import torus_points as symmetric_points
    rng = np.random.default_rng(4)
    M, N, nfp = 3, 5, 3
    th, ph = rng.uniform(0, 2 * np.pi, 400), rng.uniform(0, 2 * np.pi, 400)
    decay = np.exp(-np.add.outer(np.arange(M + 1), np.abs(np.arange(-N, N + 1))))
    rc, rs, zc, zs = (2e-2 * rng.standard_normal((M + 1, 2 * N + 1)) * decay for _ in range(4))
    for a in (rc, zc):
        a[0, :N] = 0.0
    for a in (rs, zs):
        a[0, :N + 1] = 0.0
    rc[0, N], rc[1, N], zs[1, N] = 1.0, 0.2, 0.25
    for t, points in ((TorusState(rc, zs, 0.3, 0.05, M, N, nfp, phi0=np.pi / nfp), symmetric_points),
                      (GeneralTorus(rc, rs, zc, zs, 0.3, 0.05, M, N, nfp, phi0=0.3), torus_points)):
        moved = GeneralTorus(*general_arrays(t), 0.3, 0.05, M, N, nfp, phi0=0.0)
        assert np.max(np.abs(np.asarray(points(t, th, ph)[0]) - torus_points(moved, th, ph)[0])) < 1e-14
