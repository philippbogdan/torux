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
