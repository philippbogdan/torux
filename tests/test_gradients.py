"""Adjoint gradients of the rotational transform against central differences, and in one coil of a symmetric set
against solves that assume no symmetry."""
import numpy as np

import pytest

import torux
from conftest import QUASR_954
from torux.family import Continuation
from torux.field import FieldEngine
from torux.gradients import adjoint_gradient, adjoint_gradient_general, iota_gradients, symmetric_average


def test_adjoint_against_central_differences(quasr_954, solved_954):
    """The adjoint of the solve on one field period against central differences of that solve, which keeps the
    symmetry: the coil's own entry and its copies' alike."""
    src, res = quasr_954, solved_954
    assert res.status == "surface"
    g, t, part = adjoint_gradient(res._engine, res.surface), res.surface, src.parts[0]
    dmin = res.record["engine"]["dmin"]

    def iota(coef=None, current=None, flux=None):
        """The torus solved again at its own resolution for a changed coil set or flux."""
        s = torux.Source(src.id, [torux.FourierCurves(part.coef if coef is None else coef,
                                                      part.current if current is None else current)],
                         nfp=src.nfp, stellsym=src.stellsym)
        cont = Continuation(FieldEngine(s, dmin=dmin), src.nfp, tol=1e-15)
        sol = cont.problem(t.M, t.N).newton(t, t.flux if flux is None else flux, tol=1e-15, maxit=10)
        assert sol.converged
        return sol.iota

    def central(key, index, h):
        up, down = getattr(part, key).copy(), getattr(part, key).copy()
        up[index] += h
        down[index] -= h
        return (iota(**{key: up}) - iota(**{key: down})) / (2 * h)

    fd = central("current", 0, 1e-6 * abs(part.current[0]))            # the current of the first coil
    assert abs(fd - g["d_current"][0]) <= 1e-5 * abs(fd) + 1e-14
    fd = central("coef", (1, 0, 2), 1e-6)                             # one Fourier coefficient of the second
    assert abs(fd - g["d_coef"][1, 0, 2]) <= 1e-5 * abs(fd) + 1e-12
    h = 1e-6 * abs(t.flux)                                            # the requested flux
    fd = (iota(flux=t.flux + h) - iota(flux=t.flux - h)) / (2 * h)
    assert abs(fd - g["d_flux"]) <= 1e-5 * abs(fd) + 1e-12


def test_adjoint_without_stellarator_symmetry_against_central_differences(quasr_954_mirrored, solved_954_mirrored):
    """The same for QUASR 954 with the mirrored copy of its coil at 3% more current, a source without stellarator
    symmetry, against the transform of the torus solved again at its own resolution by Newton's method on
    answer.system_general, on one field period."""
    import copy

    import scipy.linalg as sla
    from torux.torus_general import GeneralEquations, general_arrays
    src, res = quasr_954_mirrored, solved_954_mirrored
    assert res.status == "surface"
    g, t, part = adjoint_gradient_general(res._engine, res.surface), res.surface, src.parts[0]
    dmin = res.record["engine"]["dmin"]
    eq = GeneralEquations(res._engine, t.nfp, t.M, t.N, general_arrays(t))
    u0 = eq.pack(*general_arrays(t), t.iota)

    def iota(coef=None, current=None, flux=None):
        """The torus solved again at its own resolution for a changed coil set or flux."""
        s = torux.Source(src.id, [torux.FourierCurves(part.coef if coef is None else coef,
                                                      part.current if current is None else current)],
                         nfp=src.nfp, stellsym=src.stellsym)
        b = copy.copy(res)
        b._engine = FieldEngine(s, dmin=dmin)
        if flux is not None:
            b.surface = copy.copy(t)
            b.surface.flux = flux
        u = u0.copy()
        for _ in range(6):
            F, J = b.system_general(*eq.unpack(u))
            if np.max(np.abs(F)) < 1e-15 * u[0]:
                break
            u = u - sla.lu_solve(sla.lu_factor(J.T, overwrite_a=True, check_finite=False), F, trans=1,
                                 check_finite=False)
        assert np.max(np.abs(F)) < 1e-15 * u[0]
        return u[-1]

    def central(key, index, h):
        up, down = getattr(part, key).copy(), getattr(part, key).copy()
        up[index] += h
        down[index] -= h
        return (iota(**{key: up}) - iota(**{key: down})) / (2 * h)

    fd = central("current", 0, 1e-6 * abs(part.current[0]))            # the current of the first coil
    assert abs(fd - g["d_current"][0]) <= 1e-5 * abs(fd) + 1e-14
    fd = central("coef", (1, 0, 2), 1e-6)                             # one Fourier coefficient of the second
    assert abs(fd - g["d_coef"][1, 0, 2]) <= 1e-5 * abs(fd) + 1e-12
    h = 1e-6 * abs(t.flux)                                            # the requested flux
    fd = (iota(flux=t.flux + h) - iota(flux=t.flux - h)) / (2 * h)
    assert abs(fd - g["d_flux"]) <= 1e-5 * abs(fd) + 1e-12


# what changing one coil alone does, every coil's copies left as they are: the source solved again with no symmetry
# assumed (one field period, no stellarator symmetry), from QUASR 954 (two field periods, stellarator symmetric) and
# from the same with the mirrored copy of its coil at 3% more current (two field periods)
CASES = [("quasr_954", None, False), ("quasr_954_mirrored", 0.05, True)]


_SOLVED = {}


def _with_gradients(request, name, flux):
    """The source and its answer with gradients, solved once for the tests that follow."""
    if name not in _SOLVED:
        src = request.getfixturevalue(name)
        _SOLVED[name] = src, torux.solve(src, flux=flux, gradients=True)
    src, res = _SOLVED[name]
    assert res.status == "surface" and "per_coil" not in res.gradients
    return src, res


@pytest.mark.parametrize("name, flux, general", CASES)
def test_one_coil_is_the_symmetric_average(request, name, flux, general):
    """answer.gradients is the group average of the one-period solve's adjoint: unchanged by the symmetry (a coil's
    entries are its copies' mapped), the same in every change that keeps the symmetry (sum_k I_k d iota / d I_k, the
    response to scaling every current, is -flux d iota / d flux), and the copies of one coil carry one number."""
    src, res = _with_gradients(request, name, flux)
    g, t, part = res.gradients, res.surface, src.parts[0]
    one = (adjoint_gradient_general if general else adjoint_gradient)(res._engine, t)
    avg = symmetric_average(src, t, general, one["parts"])
    # (the adjoint computed again agrees with the answer's to rounding)
    assert np.abs(avg[0]["d_coef"] - g["d_iota_d_coef"]).max() <= 1e-11 * np.abs(g["d_iota_d_coef"]).max()
    assert np.abs(avg[0]["d_current"] - g["d_iota_d_current"]).max() <= 1e-11 * np.abs(g["d_iota_d_current"]).max()
    again = symmetric_average(src, t, general, g["parts"])
    assert np.abs(again[0]["d_coef"] - g["d_iota_d_coef"]).max() <= 1e-13 * np.abs(g["d_iota_d_coef"]).max()
    assert abs(g["d_iota_d_flux"] - one["d_flux"]) <= 1e-12 * abs(one["d_flux"])
    scale = -t.flux * g["d_iota_d_flux"]
    assert abs(g["d_iota_d_current"] @ part.current - scale) <= 1e-12 * abs(scale)
    assert abs(one["d_current"] @ part.current - scale) <= 1e-12 * abs(scale)
    d = g["d_iota_d_current"]
    sign = 1.0 if general else -1.0                                   # coil 1 is coil 0 flipped, its current reversed
    assert abs(d[2] - d[0]) <= 1e-13 * abs(d[0]) and abs(d[3] - d[1]) <= 1e-13 * abs(d[1])
    if not general:
        assert abs(d[1] - sign * d[0]) <= 1e-13 * abs(d[0])
    assert np.abs(one["d_current"] - d).max() > 5 * np.abs(d).max()    # what the one-period solve gives a coil


def _alone(src, flux, key, index, rel):
    """d iota / d (one entry of one coil) by central differences of the source solved again with that entry
    changed and no symmetry assumed."""
    part = src.parts[0]
    base = getattr(part, key)
    h = rel * abs(base[index])
    out = []
    for sgn in (1, -1):
        arr = base.copy()
        arr[index] += sgn * h
        s = torux.Source(f"{src.id} alone", [torux.FourierCurves(arr if key == "coef" else part.coef,
                                                                 arr if key == "current" else part.current)])
        res = torux.solve(s, flux=flux, verify=False, budget_s=600)
        assert s.nfp == 1 and not s.stellsym and res.status == "surface", res.status
        out.append(res.iota)
    return (out[0] - out[1]) / (2 * h)


@pytest.mark.parametrize("name, flux, general", CASES)
def test_one_coil_against_solves_without_symmetry(request, name, flux, general):
    """d iota / d (the current of coil 0, and one Fourier coefficient of coil 1) against central differences of
    solves of the whole source with only that entry changed, no symmetry assumed."""
    src, res = _with_gradients(request, name, flux)
    g, f = res.gradients, res.surface.flux
    fd = _alone(src, f, "current", 0, 1e-4)
    assert abs(fd - g["d_iota_d_current"][0]) <= 1e-5 * abs(fd)
    fd = _alone(src, f, "coef", (1, 0, 2), 1e-4)
    assert abs(fd - g["d_iota_d_coef"][1, 0, 2]) <= 1e-5 * abs(fd)


def _fake(src):
    """An adjoint's result in the shape of src's parameters (any numbers)."""
    rng = np.random.default_rng(1)
    p = src.parts[0]
    parts = [dict(kind="fourier", d_coef=rng.standard_normal(p.coef.shape), d_current=rng.standard_normal(p.n))]
    return dict(parts=parts, d_flux=0.5, d_coef=parts[0]["d_coef"], d_current=parts[0]["d_current"])


def test_without_symmetry_unchanged(quasr_954):
    """A source with one field period and no stellarator symmetry: the adjoint's own numbers, the very arrays."""
    from types import SimpleNamespace
    g = _fake(quasr_954)
    out = iota_gradients(quasr_954, SimpleNamespace(nfp=1, phi0=0.0), True, g)
    assert out == dict(objective="iota", d_iota_d_flux=g["d_flux"], parts=g["parts"], d_iota_d_coef=g["d_coef"],
                       d_iota_d_current=g["d_current"])
    assert out["d_iota_d_coef"] is g["d_coef"] and out["parts"] is g["parts"]


def test_copies_that_cannot_be_mapped(quasr_954):
    """A coil whose copy runs along it at another pace (no shift or reversal of its parameter) has no map to it:
    the answer gives no per-coil numbers under the usual keys, says why, and keeps the one-period solve's apart."""
    from types import SimpleNamespace
    p = quasr_954.parts[0]
    t = np.arange(256) / 256
    x, _ = torux.FourierCurves(p.coef[1:2], p.current[1:2]).points(t + 0.05 * np.sin(2 * np.pi * t))
    c = np.fft.rfft(x[0], axis=0) / 256                                    # the copy, resampled
    coef = p.coef.copy()
    K = p.order
    coef[1, :, 0] = c[0].real
    coef[1, :, 2::2] = 2 * c[1:K + 1].real.T
    coef[1, :, 1::2] = -2 * c[1:K + 1].imag.T
    src = torux.Source("954 resampled", [torux.FourierCurves(coef, p.current)], nfp=2, stellsym=True)
    g = _fake(src)
    out = iota_gradients(src, SimpleNamespace(nfp=2, phi0=0.0), False, g)
    assert out["d_iota_d_flux"] == g["d_flux"]
    assert not {"parts", "d_iota_d_coef", "d_iota_d_current"} & set(out)
    assert out["per_coil"].startswith("not given: ")
    assert out["one_period"]["d_iota_d_current"] is g["d_current"] and "one field period" in out["one_period"]["meaning"]


def _symmetric_set(nfp, base_pos, base_vec, flip_vec_sign):
    """The images of base points and vectors under rotation by 2 pi / nfp and the flip, the flip's vectors times
    flip_vec_sign (-1 for a dipole's moment, whose current the flip reverses)."""
    from torux.source import FLIP, rotation_z
    pos, vec = [], []
    for k in range(nfp):
        R = rotation_z(2 * np.pi * k / nfp)
        for M, sv in ((R, 1.0), (R @ FLIP, flip_vec_sign)):
            pos.append(base_pos @ M.T)
            vec.append(sv * base_vec @ M.T)
    return np.concatenate(pos), np.concatenate(vec)


def test_the_maps_are_the_symmetry(quasr_954):
    """The parameters' maps found for every kind of part take the source to itself (a coil whose copy runs the
    other way and from another start included), and the average over the symmetry is unchanged by it."""
    from types import SimpleNamespace
    from torux.gradients import _action, _group
    p = quasr_954.parts[0]
    coef, cur = p.coef.copy(), p.current.copy()
    w = coef[3, :, 2::2] + 1j * coef[3, :, 1::2]                       # coil 3 backwards from t = 0.3
    w = w * np.exp(-2j * np.pi * 0.3 * np.arange(1, p.order + 1))[None, :]
    coef[3, :, 2::2], coef[3, :, 1::2] = w.real, -w.imag
    cur[3] = -cur[3]
    t = np.arange(40) / 40
    x, _ = p.points(t)
    loops = [x[k][::-1] if k == 1 else x[k] for k in range(p.n)]     # polygon loop 1 backwards, its current reversed
    poly = torux.PolylineCurves(np.concatenate(loops), np.arange(p.n + 1) * len(t),
                                np.where(np.arange(p.n) == 1, -1.0, 1.0) * p.current)
    seg = torux.PolylineCurves(np.concatenate([x[k] for k in range(p.n)]), np.arange(p.n + 1) * len(t),
                               p.current).as_segments()
    pos, mom = _symmetric_set(2, np.array([[1.3, 0.2, 0.1], [1.1, 0.5, -0.2]]), np.array([[0.1, 0.2, 0.3],
                                                                                         [0.0, -0.4, 0.2]]), -1.0)
    src = torux.Source("954 every kind", [torux.FourierCurves(coef, cur), poly,
                                         torux.Segments(seg.a, seg.b, seg.current), torux.Dipoles(pos, mom),
                                         torux.Rings(np.array([1.5, 1.5, 0.7]), np.array([0.4, -0.4, 0.0]),
                                                     np.array([2.0, 2.0, -1.0])),
                                         torux.ToroidalField(0.3)], nfp=2, stellsym=True)
    params = [dict(kind="fourier", d_coef=coef, d_current=cur),
              dict(kind="polyline", d_points=poly.points, d_current=poly.current),
              dict(kind="segments", d_a=seg.a, d_b=seg.b, d_current=seg.current),
              dict(kind="dipoles", d_pos=pos, d_moment=mom),
              dict(kind="rings", d_R=np.array([1.5, 1.5, 0.7]), d_Z=np.array([0.4, -0.4, 0.0]),
                   d_current=np.array([2.0, 2.0, -1.0])),
              dict(kind="toroidal", d_RB=np.asarray(0.3))]
    alpha, n, F = _group(SimpleNamespace(nfp=2, phi0=0.0), False)
    from torux.source import rotation_z
    for M, s in ((F, -1.0), (rotation_z(alpha), 1.0)):
        image = _action(src, M, s)(params)
        for a, b in zip(image, params):
            for k in a:
                if k != "kind":
                    assert np.abs(np.asarray(a[k]) - np.asarray(b[k])).max() <= 1e-13 * np.abs(b[k]).max(), (b["kind"], k)
    rng = np.random.default_rng(2)
    g = [{k: (rng.standard_normal(np.shape(v)) if k != "kind" else v) for k, v in q.items()} for q in params]
    avg = symmetric_average(src, SimpleNamespace(nfp=2, phi0=0.0), False, g)
    again = symmetric_average(src, SimpleNamespace(nfp=2, phi0=0.0), False, avg)
    for a, b in zip(again, avg):
        for k in a:
            if k != "kind":
                assert np.abs(np.asarray(a[k]) - np.asarray(b[k])).max() <= 1e-13


# --------------------------------------------------------------------------- field periods without stellarator symmetry
def _three_periods():
    """QUASR 219457 with its coils of negative current at 3% more: three field periods, no stellarator symmetry."""
    from conftest import example
    from torux.io._stellcommon import apply_verified_symmetry
    q = torux.load(example("quasr_0219457.json"))
    p = q.parts[0]
    src = torux.Source("219457 at +3% in its negative currents",
                       [torux.FourierCurves(p.coef, np.where(p.current < 0, 1.03, 1.0) * p.current)])
    apply_verified_symmetry(src, [q.nfp])
    assert src.nfp == 3 and not src.stellsym
    return src, float(q.request["flux_Wb"])


def test_three_periods_without_stellarator_symmetry_have_per_coil_numbers():
    """With field periods and no flip, the group is the rotations alone: torux 0.1.4 tested the flip's closure with
    the identity for a flip, which is the rotation twice, and gave no per-coil numbers for three periods or more."""
    from types import SimpleNamespace
    from torux.gradients import _action
    from torux.source import rotation_z
    src, _ = _three_periods()
    g = _fake(src)
    out = iota_gradients(src, SimpleNamespace(nfp=3, phi0=0.0), True, g)
    assert "per_coil" not in out and "one_period" not in out
    rot = _action(src, rotation_z(2 * np.pi / 3), 1.0)
    acc = cur = g["parts"]
    for _ in range(2):
        cur = rot(cur)
        acc = [{k: (a[k] + c[k] if k != "kind" else a[k]) for k in a} for a, c in zip(acc, cur)]
    want = acc[0]["d_current"] / 3
    assert np.abs(out["d_iota_d_current"] - want).max() <= 1e-14 * np.abs(want).max()
    d = out["d_iota_d_current"]                                         # coil k + 4 is coil k turned by a period
    assert np.abs(d[4:8] - d[0:4]).max() <= 1e-14 * np.abs(d).max() and np.abs(d[8:] - d[:4]).max() <= 1e-14 * np.abs(d).max()


# the central differences' own truncation at a step of 1e-4 of the entry is 3e-5 of them here (1e-3 gives 3e-3)
TOL_219457 = 1e-4


@pytest.mark.slow
def test_three_periods_without_stellarator_symmetry_against_solves_without_symmetry():
    """The per-coil numbers of that source against central differences of solves with only one entry changed and no
    symmetry assumed. The one-period solve's dι/dI of coil 0 is +5.8e-07 per ampere, of the other sign."""
    src, flux = _three_periods()
    res = torux.solve(src, flux=flux, gradients=True, budget_s=1500)
    assert res.status == "surface" and "per_coil" not in res.gradients
    g, f = res.gradients, res.surface.flux
    fd = _alone(src, f, "current", 0, 1e-4)
    assert abs(fd - g["d_iota_d_current"][0]) <= TOL_219457 * abs(fd)
    fd = _alone(src, f, "coef", (1, 0, 2), 1e-4)
    assert abs(fd - g["d_iota_d_coef"][1, 0, 2]) <= TOL_219457 * abs(fd)


# --------------------------------------------------------------------------- coils of zero current, parts of no coil
def test_a_coil_of_zero_current_has_its_numbers(request, quasr_954):
    """QUASR 954 and a copy of each of its coils at zero current. torux 0.1.4 left the copies out of the arrays (4
    entries for 8 coils) and gave zero for their current in parts. A zero-current copy of coil k makes the field of
    coil k as its current grows, so its dι/dI is coil k's, which the solves of the test above hold to the truth, and
    the solve again with only its current changed and no symmetry assumed says the same. Its coefficients make no
    field: zero."""
    p = quasr_954.parts[0]
    src = torux.Source("954 and its coils at zero current", [torux.FourierCurves(np.concatenate([p.coef, p.coef]),
                                                                                 np.concatenate([p.current,
                                                                                                 np.zeros(p.n)]))],
                       nfp=2, stellsym=True)
    res = torux.solve(src, flux=QUASR_954["flux_Wb"], gradients=True)
    g = res.gradients
    assert res.status == "surface" and "per_coil" not in g
    assert g["d_iota_d_current"].shape == (8,) and g["d_iota_d_coef"].shape == (8,) + p.coef.shape[1:]
    _, plain = _with_gradients(request, "quasr_954", None)
    d0 = plain.gradients["d_iota_d_current"]
    d = g["d_iota_d_current"]
    assert np.abs(d[:4] - d0).max() <= 1e-9 * np.abs(d0).max()        # the same coils, solved from their axis
    assert np.abs(d[4:] - d[:4]).max() <= 1e-12 * np.abs(d0).max()
    assert not np.any(g["d_iota_d_coef"][4:]) and not np.any(g["parts"][0]["d_coef"][4:])
    assert np.array_equal(g["parts"][0]["d_current"], g["d_iota_d_current"])
    h = 1e-4 * abs(p.current[0])                                       # the truth: coil 4's current alone
    io = []
    for s in (1, -1):
        cur = src.parts[0].current.copy()
        cur[4] = s * h
        alone = torux.Source("954 alone", [torux.FourierCurves(src.parts[0].coef, cur)])
        r = torux.solve(alone, flux=res.surface.flux, verify=False, budget_s=600)
        assert alone.nfp == 1 and not alone.stellsym and r.status == "surface", r.status
        io.append(r.iota)
    fd = (io[0] - io[1]) / (2 * h)
    assert abs(fd - g["d_iota_d_current"][4]) <= 1e-5 * abs(fd)


def test_a_fourier_part_of_no_coil(quasr_954):
    """A source of QUASR 954's coils and a Fourier part holding none: torux 0.1.4 failed in the symmetric average
    (a reduction of no element), and the gradients were an error, dι/dΨ lost with the rest."""
    p = quasr_954.parts[0]
    none = torux.FourierCurves(np.zeros((0,) + p.coef.shape[1:]), np.zeros(0))
    src = torux.Source("954 and a part of no coil", [p, none], nfp=2, stellsym=True)
    g = torux.solve(src, flux=QUASR_954["flux_Wb"], gradients=True).gradients
    assert "error" not in g and "per_coil" not in g
    assert np.isfinite(g["d_iota_d_flux"]) and g["d_iota_d_current"].shape == (4,)
    assert g["parts"][1]["d_current"].shape == (0,)
