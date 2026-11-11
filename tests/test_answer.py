"""The answer torux.solve returns: the solver's outcome unchanged, its numbers and files checked against code that
shares none with torux's (SIMSOPT 1.11.1, meshio, torux.reference), its words no more than its status."""
import json
import math
import os
import re

import numpy as np
import pytest

import torux
from torux.answer import Answer
from torux.api import Result
from torux.api import solve as api_solve

from conftest import example


def _lines(a):
    return str(a).splitlines()


def _bytes(x):
    return np.ascontiguousarray(np.asarray(x, float)).tobytes()


# ---------------------------------------------------------------------------------------------- the solver unchanged
@pytest.mark.parametrize("name, flux", [("quasr_0000954.json", None), ("quasr_0000954.json", 0.03),
                                        ("d3d_icoils_upper.dat", None)])
def test_the_answer_is_the_solvers_outcome_bit_for_bit(name, flux):
    """torux.solve and torux.api.solve, called the same way in one process: the same status, flux, transform and
    Fourier coefficients, bit for bit."""
    a = torux.solve(example(name), flux)
    r = api_solve(example(name), flux)
    assert isinstance(a, Answer) and isinstance(a, Result)
    assert a.status == r.status
    if r.surface is None:
        assert a.surface is None
        assert a.evidence["proof"]["verdict"] == r.evidence["proof"]["verdict"]
        return
    assert a.iota == r.iota and a.surface.flux == r.surface.flux
    assert (a.surface.M, a.surface.N) == (r.surface.M, r.surface.N)
    for k in ("rc", "zs"):
        assert _bytes(getattr(a.surface, k)) == _bytes(getattr(r.surface, k))
    assert a.flux == abs(r.surface.flux)


# ---------------------------------------------------------------------------------------------- numbers and files
def _simsopt():
    return pytest.importorskip("simsopt.geo")


def _sim_surface(path, phis, thetas):
    """SIMSOPT's own surface from a VMEC input file, at angles in radians (SIMSOPT counts them in turns)."""
    geo = _simsopt()
    return geo.SurfaceRZFourier.from_vmec_input(str(path), quadpoints_phi=np.asarray(phis) / (2 * np.pi),
                                               quadpoints_theta=np.asarray(thetas) / (2 * np.pi))


def _synthetic(general, nfp, phi0, seed):
    """A smooth torus with 3 x 5 random harmonics on top of a rotating ellipse, given as an answer (no solve)."""
    from torux.torus import TorusState
    from torux.torus_general import GeneralTorus
    rng = np.random.default_rng(seed)
    M, N = 3, 5
    shape = (M + 1, 2 * N + 1)
    decay = np.exp(-np.add.outer(np.arange(M + 1), np.abs(np.arange(-N, N + 1))))
    rc, zs = 2e-2 * rng.standard_normal(shape) * decay, 2e-2 * rng.standard_normal(shape) * decay
    rc[0, :N] = 0.0
    zs[0, :N + 1] = 0.0
    rc[0, N], rc[1, N], zs[1, N] = 1.0, 0.2, 0.25
    if general:
        rs, zc = 1e-2 * rng.standard_normal(shape) * decay, 1e-2 * rng.standard_normal(shape) * decay
        rs[0, N] = 0.0
        rc[0, N - 2], zc[0, N - 1] = 3e-3, -2e-3            # m = 0 entries at n < 0: folded onto n > 0 in the file
        t = GeneralTorus(rc, rs, zc, zs, 0.31, 0.05, M, N, nfp, phi0=phi0)
    else:
        t = TorusState(rc, zs, 0.31, 0.05, M, N, nfp, phi0=phi0)
    a = Answer._of(Result("surface", 1.0, 0.0, surface=t, iota=0.31), 0.05)
    return a


def _answers(solved_954):
    return [("quasr 954", solved_954), ("stellarator symmetric, phi0 = pi / nfp", _synthetic(False, 3, np.pi / 3, 1)),
            ("no symmetry, phi0 = 0.3", _synthetic(True, 2, 0.3, 2))]


def test_numbers_and_points_agree_with_simsopt_reading_the_vmec_file(solved_954, tmp_path):
    """SIMSOPT 1.11.1 reads the VMEC input file (SurfaceRZFourier.from_vmec_input) and computes volume, area, the
    radii and the aspect ratio with its own code: the same to 1e-10 relative. Its points at random angles are the
    answer's to 1e-13 m, and so is the cross-section at a random cylindrical angle."""
    _simsopt()
    rng = np.random.default_rng(7)
    for label, a in _answers(solved_954):
        path = tmp_path / "input.check"
        assert a.save(path) == 0.0
        assert path.read_text().splitlines()[0] == "! every Fourier coefficient of the surface: no point moved"
        period = 2 * np.pi / a.nfp
        n = 4 * (a.surface.M + a.surface.N) + 64
        dense = _sim_surface(path, period * np.arange(n) / n, 2 * np.pi * np.arange(n) / n)
        for k in ("volume", "area", "major_radius", "minor_radius", "aspect_ratio"):
            mine, theirs = getattr(a, k), abs(getattr(dense, k)())
            assert abs(mine - theirs) <= 1e-10 * theirs, (label, k, mine, theirs)
        th, ph = rng.uniform(0, 2 * np.pi, 16), rng.uniform(0, 2 * np.pi, 16)
        g = _sim_surface(path, ph, th).gamma()                       # (phi, theta, 3)
        PH, TH = np.meshgrid(ph, th, indexing="ij")
        x = a.points(TH.ravel(), PH.ravel()).reshape(16, 16, 3)
        assert np.max(np.abs(g - x)) < 1e-13, (label, np.max(np.abs(g - x)))
        phi = float(rng.uniform(0, 2 * np.pi))
        R, Z = a.section(phi, n=64)
        gs = _sim_surface(path, [phi], 2 * np.pi * np.arange(64) / 64).gamma()[0]
        assert np.max(np.abs(np.hypot(gs[:, 0], gs[:, 1]) - R)) < 1e-13 and np.max(np.abs(gs[:, 2] - Z)) < 1e-13


def test_a_cut_file_states_how_far_it_moved_the_surface(solved_954, tmp_path):
    """input.<name> cut to mpol, ntor: its first comment line and save's value state the largest displacement, which
    a dense grid of SIMSOPT's points of the full and the cut surfaces confirms (never above it, within 0.1 % of it).
    The uncut file moves nothing (its points are the answer's to 1e-13 m everywhere on that grid)."""
    _simsopt()
    for label, a in _answers(solved_954):
        full, cut = tmp_path / "input.full", tmp_path / "input.cut"
        a.save(full)
        moved = a.save(cut, mpol=3, ntor=2)
        head = cut.read_text().splitlines()[0]
        assert head.startswith("! cut to MPOL = 3, NTOR = 2: ")
        stated = float(re.search(r"more than ([0-9.e+-]+) m", head).group(1))
        assert abs(stated - moved) <= 5e-4 * moved
        n = 512
        th = 2 * np.pi * np.arange(n) / n
        ph = 2 * np.pi / a.nfp * np.arange(n) / n
        gf, gc = _sim_surface(full, ph, th).gamma(), _sim_surface(cut, ph, th).gamma()
        dense = float(np.max(np.linalg.norm(gf - gc, axis=-1)))
        assert dense <= moved * (1 + 1e-12) and moved <= dense * (1 + 1e-3), (label, dense, moved)
        PH, TH = np.meshgrid(ph, th, indexing="ij")
        x = a.points(TH.ravel(), PH.ravel()).reshape(n, n, 3)
        assert np.max(np.abs(gf - x)) < 1e-13, label


def test_the_vtk_file_reads_back_in_meshio_with_the_surface_and_its_field(solved_954, quasr_954, tmp_path):
    """meshio reads the VTK file: its points are the surface's points, its cells close the torus, and |B| on it agrees
    with torux.reference (the evaluator that shares no code with the solver) to 1e-12."""
    meshio = pytest.importorskip("meshio")
    a = solved_954
    path = tmp_path / "surface.vtk"
    assert a.save(path) is None
    m = meshio.read(str(path))
    pts = m.points
    phi = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
    nph = len(np.unique(np.round(phi, 12)))
    nth = len(pts) // nph
    th = np.tile(2 * np.pi * np.arange(nth) / nth, nph)
    ph = np.repeat(2 * np.pi * np.arange(nph) / nph, nth)
    assert np.max(np.abs(pts - a.points(th, ph))) < 1e-14       # the grid is summed as a grid: rounding apart
    quads = [c for c in m.cells if c.type == "quad"]
    assert len(quads) == 1 and len(quads[0].data) == len(pts)
    assert np.all(np.bincount(quads[0].data.ravel(), minlength=len(pts)) == 4)
    from torux.reference import fast64
    pick = np.random.default_rng(3).choice(len(pts), 300, replace=False)
    ref = np.linalg.norm(fast64.Field(quasr_954).evaluate(pts[pick], grad=False)["B"], axis=-1)
    mod_b = np.asarray(m.point_data["modB"]).ravel()[pick]
    assert np.max(np.abs(mod_b - ref) / ref) < 1e-12


def test_a_saved_answer_reads_back_with_the_same_numbers_words_and_files(solved_954, tmp_path):
    a = solved_954
    a.save(tmp_path / "a.json")
    b = Answer.read(tmp_path / "a.json")
    for k in ("status", "flux", "requested_flux", "iota", "nfp", "volume", "area", "major_radius", "minor_radius",
              "aspect_ratio"):
        assert getattr(b, k) == getattr(a, k), k
    assert str(b) == str(a)
    assert np.array_equal(b.profile, a.profile)
    assert all(np.array_equal(u, v) for u, v in zip(b.section(0.4), a.section(0.4)))
    th, ph = np.linspace(0, 6, 50), np.linspace(0, 6, 50)
    assert np.array_equal(b.points(th, ph), a.points(th, ph))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for name, kw in (("input.x", {}), ("input.cut", dict(mpol=5, ntor=4)), ("s.vtk", {})):
        assert a.save(tmp_path / "a" / name, **kw) == b.save(tmp_path / "b" / name, **kw)
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    b.save(tmp_path / "b.json")
    assert json.loads((tmp_path / "b.json").read_text()) == json.loads((tmp_path / "a.json").read_text())
    with pytest.raises(ValueError, match="needs the source"):
        b.field(np.zeros((1, 3)))


# ---------------------------------------------------------------------------------------------- words and refusals
def test_a_surface_says_what_it_is_and_how_it_was_checked(solved_954):
    a = solved_954
    text = str(a)
    lines = _lines(a)
    from torux.cli import report
    assert len(lines) <= 12 and repr(a) == text and report(a) == text
    assert lines[0].split()[:2] == ["surface", "verified:"]
    for v in (f"{a.flux:.12g} Wb", f"{a.iota:.12g}", f"major {a.major_radius:.9g} m",
              f"minor {a.minor_radius:.9g} m", f"aspect ratio {a.aspect_ratio:.6g}", f"{a.volume:.9g} m^3",
              "M, N = 48, 32", "(the solver's own check)", "independent check"):
        assert v in text, v
    assert a.requested_flux == pytest.approx(0.06910690581397087, rel=1e-15)
    p = a.profile
    assert p.shape[1] == 2 and np.all(np.diff(p[:, 0]) > 0) and tuple(p[-1]) == (a.flux, a.iota)


def test_a_proof_is_said_in_words_and_gives_no_numbers(tmp_path):
    a = torux.solve(example("d3d_icoils_upper.dat"))
    text = str(a)
    assert a.status == "no_surface" and len(_lines(a)) <= 12
    assert text.startswith("no surface") and "proved: these coils make no surface" in text
    assert "antisymmetric under reflection" in text and "(numerical)" in text
    assert "ANTISYM" not in text and "verified" not in text
    for k in ("flux", "iota", "nfp", "volume", "minor_radius", "major_radius", "aspect_ratio", "area"):
        with pytest.raises(ValueError, match="no_surface"):
            getattr(a, k)
    with pytest.raises(ValueError, match="no_surface"):
        a.save(tmp_path / "input.x")
    a.save(tmp_path / "a.json")
    assert str(Answer.read(tmp_path / "a.json")) == text


def test_an_undetermined_answer_says_what_was_found_and_what_to_do():
    a = torux.solve(example("quasr_1100073.json"), budget_s=5)
    text = str(a)
    assert a.status in ("undetermined", "timeout") and len(_lines(a)) <= 12
    assert "budget_s=3300" in text and "found" in text
    assert not re.search(r"\b(verified|certified|proved)\b", text)
    with pytest.raises(ValueError, match=a.status):
        a.volume


def test_an_unverified_torus_is_not_presented_as_an_answer(solved_954, tmp_path):
    ev = dict(solved_954.evidence, unverified="the independent verification refused the torus")
    ev["independent"] = dict(ev["independent"], ok=False, bn_sup=4e-14)
    a = Answer._of(Result("surface_unverified", 300.0, 3.0, solved_954.surface, solved_954.iota, ev,
                          record=solved_954.record), solved_954.requested_flux)
    text = str(a)
    assert len(_lines(a)) <= 12
    assert text.startswith("surface unverified") and "not an answer" in text and "4.0e-14" in text
    for k in ("iota", "volume", "flux"):
        with pytest.raises(ValueError, match="not verified"):
            getattr(a, k)
    with pytest.raises(ValueError, match="not verified"):          # nor its points, nor a section
        a.points(0.0, 0.0)
    with pytest.raises(ValueError, match="not verified"):
        a.section()
    assert a.surface is solved_954.surface and len(a.profile) >= 1  # the torus and the family are there to look at
    with pytest.raises(ValueError):
        a.save(tmp_path / "x.vtk")


def test_points_keep_the_shape_of_their_angles(solved_954):
    th, ph = np.linspace(0.1, 2.0, 3), np.linspace(0.3, 1.4, 4)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    flat = solved_954.points(TH.ravel(), PH.ravel())
    assert flat.shape == (12, 3)
    assert np.array_equal(solved_954.points(TH, PH), flat.reshape(3, 4, 3))
    assert np.array_equal(solved_954.points(th[:, None], ph[None, :]), flat.reshape(3, 4, 3))
    assert solved_954.points(0.3, 0.2).shape == (3,)
    assert np.array_equal(solved_954.points(0.3, ph), solved_954.points(np.full(4, 0.3), ph))


def test_save_refuses_a_name_without_a_format(solved_954, tmp_path):
    for name in ("surface.txt", "input.", "boundary"):
        with pytest.raises(ValueError, match="json"):
            solved_954.save(tmp_path / name)
    with pytest.raises(ValueError, match="mpol"):
        solved_954.save(tmp_path / "a.json", mpol=4)


def test_the_command_line_prints_the_answer_and_saves_it(solved_954, tmp_path, capsys):
    from torux.cli import main
    assert main(["solve", example("quasr_0000954.json"), "--save", str(tmp_path / "x.txt")]) == 1
    capsys.readouterr()
    out = [str(tmp_path / "a.json"), str(tmp_path / "input.a"), str(tmp_path / "a.vtk")]
    code = main(["solve", example("quasr_0000954.json"), "--save", out[0], "--save", out[1], "--save", out[2],
                 "--out", str(tmp_path / "legacy")])
    printed = capsys.readouterr().out.splitlines()
    assert code == 0
    assert printed[:-1] == _lines(solved_954)[:-1] and printed[-1].startswith("time")
    assert all(os.path.getsize(p) > 0 for p in out)
    assert Answer.read(out[0]).flux == solved_954.flux == Answer.read(tmp_path / "legacy").flux
    # a cut from the shell: the file VMEC's MPOL and NTOR, and how far the cut moved the surface said on stderr
    cut = str(tmp_path / "input.cut")
    assert main(["solve", example("quasr_0000954.json"), "--save", cut, "--mpol", "9", "--ntor", "8"]) == 0
    said = capsys.readouterr().err
    moved = solved_954.save(tmp_path / "input.same", mpol=9, ntor=8)
    assert open(cut).read() == open(tmp_path / "input.same").read()
    assert "MPOL = 9, NTOR = 8" in said and f"{moved:.3e} m" in said
    assert main(["solve", example("quasr_0000954.json"), "--save", out[0], "--mpol", "9"]) == 1
    assert "--save input.<name>" in capsys.readouterr().err
    # a budget that is no time at all is refused before anything is announced
    for b in ("0", "-5", "nan", "inf"):
        assert main(["solve", example("quasr_0000954.json"), "--budget", b]) == 1
        said = capsys.readouterr().err
        assert "--budget" in said and "positive number" in said and "solving" not in said
    # a command line torux cannot read is an error, 1: 2 is the code of an undetermined answer and of nothing else
    for argv in (["solve"], ["solve", example("quasr_0000954.json"), "--flux"], ["solve", "a.json", "--no-such"], []):
        with pytest.raises(SystemExit) as stop:
            main(argv)
        assert stop.value.code == 1 and "usage: torux" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- the slow statuses
@pytest.mark.slow
def test_a_range_says_how_far_it_was_followed_and_why_not_further():
    a = torux.solve(torux.Request(example("hsx.json")))     # no flux, as a Request: the range, not the largest
    text = str(a)
    print(text)
    assert len(_lines(a)) <= 12
    if a.status == "range":
        assert text.startswith("range") and "at least" in text and "followed" in text and "beyond" in text
        assert a.profile[-1, 0] >= a.flux


@pytest.mark.slow
@pytest.mark.parametrize("name, kind", [("quasr_1100073.json", "edge"), ("quasr_0219457.json", "island")])
def test_a_certificate_is_said_in_words(name, kind):
    a = torux.solve(example(name), budget_s=3300)
    text = str(a)
    print(text)
    assert len(_lines(a)) <= 12
    if a.status == "no_surface":
        assert text.startswith("no surface   ") or text.startswith("no surface")
        assert ("upper bound" in text and "weakest kind" in text) if kind == "edge" else \
            ("none between" in text and "island chain" in text)
        with pytest.raises(ValueError, match="no_surface"):
            a.iota
    else:
        assert "budget_s" in text or "next" in text


# ---------------------------------------------------------------------------------------------- strict JSON
def _strict(path):
    """The file read by a reader that refuses NaN and Infinity, as JSON's own standard does."""
    def refuse(token):
        raise ValueError(f"{path}: {token} is not JSON")
    with open(path) as fh:
        return json.loads(fh.read(), parse_constant=refuse)


def test_every_saved_answer_is_strict_json(solved_954, tmp_path):
    """For every kind of status, with every number the solver left undefined (NaN budget, NaN iota, infinite
    bounds), what Answer.save, the command line's --out and torux.many's save folder write parses strictly."""
    from torux.batch import _error_answer
    nan = float("nan")
    ev = dict(solved_954.evidence, unverified="the independent verification refused the torus",
              certificate=dict(kind="edge_barrier", F_up=float("inf"), ok=False))
    answers = [solved_954, torux.solve(example("d3d_icoils_upper.dat")), _error_answer("it broke", 1.0, None),
               Answer._of(Result("surface_unverified", nan, nan, solved_954.surface, nan, ev,
                                 record=dict(solved_954.record, x=nan)), None),
               Answer._of(Result("undetermined", 300.0, 2.0, None, nan, dict(e=float("-inf"))), 0.05)]
    for k, a in enumerate(answers):
        p = tmp_path / f"a{k}.json"
        a.save(p)
        d = _strict(p)
        assert d["status"] == a.status
        assert Answer.read(p).status == a.status


# ---------------------------------------------------------------------------------------------- memory
def test_a_surface_of_high_resolution_costs_megabytes(solved_954):
    """Volume, area, points, the VTK grid and the field at many points take the memory of a slab whatever the
    surface's resolution: at M, N = 448, 128, which the solver reaches on hard requests, every point held against
    every mode at once would be tens of gigabytes."""
    import tracemalloc
    import types

    from torux import answer as A
    from torux.verify import torus_points
    M, N = 448, 128
    t = types.SimpleNamespace(M=M, N=N, nfp=4, phi0=0.0, rc=np.zeros((M + 1, 2 * N + 1)),
                              zs=np.zeros((M + 1, 2 * N + 1)))
    t.rc[0, N], t.rc[1, N], t.zs[1, N] = 1.2, 0.12, 0.15
    t.rc[1, N + 1], t.zs[1, N + 1], t.rc[300, N + 100] = 0.02, -0.02, 1e-9
    rng = np.random.default_rng(0)
    th, ph = rng.uniform(0, 2 * np.pi, 30000), rng.uniform(0, 2 * np.pi, 30000)
    tracemalloc.start()
    try:
        area, size = A._area(t), A._size(t)
        x = A._vtk_points(t)
        p = A._points(t, th, ph)[0]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 400 * 2 ** 20, f"{peak / 2 ** 20:.0f} MiB"
    assert len(x) == A.VTK_GRID[0] * A.VTK_GRID[1] and p.shape == (30000, 3)
    # the same numbers as the solver's own pointwise evaluation, and a torus of nearly circular section
    assert np.max(np.abs(p[:500] - np.asarray(torus_points(t, th[:500], ph[:500])[0]))) < 1e-14
    k = np.arange(0, len(x), 997)
    nth = A.VTK_GRID[0]
    want = np.asarray(torus_points(t, 2 * np.pi * (k % nth) / nth, 2 * np.pi * (k // nth) / A.VTK_GRID[1])[0])
    assert np.max(np.abs(x[k] - want)) < 1e-14
    assert abs(size["volume"] / (2 * np.pi ** 2 * 1.2 * 0.12 * 0.15) - 1) < 0.05 and area > 0
    # the field at many points is the field at each: slabs change nothing
    q = np.asarray(solved_954.points(th[:9000], ph[:9000]))
    assert np.array_equal(np.asarray(solved_954.field(q)), np.asarray(Result.field(solved_954, q)))
    assert np.asarray(solved_954.field(q.reshape(90, 100, 3))).shape == (90, 100, 3)
