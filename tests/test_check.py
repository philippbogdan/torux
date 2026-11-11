"""torux.check: the field across a surface from another code, against truths that share no code with it.

The truths: the comparison's evaluator (results/evaluator, which shares no code with torux) and its values in
results/compare300.csv. SIMSOPT's own evaluation of its surfaces. Fourier coefficients written by hand and summed
directly here. DESC's and VMEC++'s own evaluation where they install.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pytest

from conftest import EXAMPLES, example

import torux
from torux.check import add_command, check, measure_at, read_surface, run_command

REPO = os.path.dirname(EXAMPLES)
EVALUATOR = os.path.join(REPO, "results", "evaluator")
sys.path.insert(0, EVALUATOR)
import evaluate as EV  # noqa: E402
import forms as FORMS  # noqa: E402
from judge import coils_record, surface_form  # noqa: E402


def _inputs_dir():
    d = os.path.join(REPO, "results", "compare300", "inputs")
    return d if os.path.isdir(d) else None


INPUTS = _inputs_dir()
needs_inputs = pytest.mark.skipif(INPUTS is None, reason="the comparison's input files are not in "
                                                          "results/compare300/inputs")


def _stored_rows():
    import csv
    with open(os.path.join(REPO, "results", "compare300.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["solver"] == "quasr-stored"]
    rows.sort(key=lambda r: float(r["bn_max"]))
    # 24 rows spread over the table, from the least leaky stored surface to the most
    pick = np.unique(np.rint(np.linspace(0, len(rows) - 1, 24)).astype(int))
    return [rows[i] for i in pick]


STORED = _stored_rows()


def _evaluator_angles():
    th, ph = EV.points()                                                    # 2 pi rng.random(4000) twice
    secs = 2 * np.pi * np.random.default_rng(EV.SEED + 1).random(4)         # judge_flux's four sections
    return th, ph, secs


# --------------------------------------------------------------------------- 1. QUASR's stored surfaces
@needs_inputs
@pytest.mark.parametrize("row", STORED, ids=[r["design"] for r in STORED])
def test_stored_surface_reproduces_the_evaluator(row):
    """At the evaluator's own 4,000 points and four sections: its leak to float64's digits, its flux to 1e-9."""
    f = os.path.join(INPUTS, f"quasr_{row['design']}.json")
    src = torux.load(f)
    th, ph, secs = _evaluator_angles()
    c = measure_at(src, read_surface(f), th, ph, secs)
    ext, f64 = float(row["bn_max"]), float(row["f64_bn_max"])
    # float64's digits: the evaluator's own two tiers differ by up to 3.6e-15 on these surfaces, and next to a coil
    # the field's rounding at the worst point (c.rounding, double-double against float64) adds to it
    allow = 2e-15 + abs(f64 - ext) + 2 * c.rounding
    assert abs(c.leak - ext) <= allow, (c.leak, ext, allow)
    assert abs(c.leak - f64) <= allow, (c.leak, f64, allow)
    assert abs(c.flux / float(row["flux_Wb"]) - 1) < 1e-9
    assert not c.notes
    # at check's own 4,096 points the largest value differs a little from the evaluator's
    own = check(src, f)
    assert 0.5 < own.leak / ext < 2.0


# --------------------------------------------------------------------------- 2. torux's own surface
def test_torux_surface_agrees_with_the_evaluator(quasr_954, solved_954):
    """torux's surface for QUASR 954, as judge.py hands it to the evaluator (the rz_fourier form)."""
    res = solved_954
    assert res.status == "surface" and getattr(res.surface, "phi0", 0.0) == 0.0
    form = surface_form(res.surface)
    rec = coils_record(quasr_954, res.surface.flux)
    ans = dict(design="0000954", solver="torux", status="surface", surface=form)
    out = EV.evaluate(rec, ans, precise=EV.precise_dtype() is not None)
    th, ph, secs = _evaluator_angles()
    via_result = measure_at(quasr_954, read_surface(res), th, ph, secs)
    via_form = measure_at(quasr_954, read_surface(form), th, ph, secs)
    for c in (via_result, via_form):
        # two float64 evaluators that share no code, both reading the rounding floor: they agree that it is rounding
        # (a few 1e-15 at worst, its digits depending on the machine's arithmetic), not on its digits
        assert c.leak < 1e-14 and out["f64_bn_max"] < 1e-14
        assert abs(c.leak_median - out["f64_bn_median"]) < 2e-16
        assert abs(c.flux / out["flux_Wb"] - 1) < 1e-13
    assert abs(via_result.flux_error) < 1e-12 and via_result.flux_stated_by == "the answer's"
    # the answer's saved JSON reads back to the same measurement
    c0 = check(quasr_954, res)
    assert c0.leak < 1e-14 and abs(c0.flux_error) < 1e-12


def test_saved_answer_reads_back(quasr_954, solved_954, tmp_path):
    p = tmp_path / "answer.json"
    solved_954.save(p)
    a, b = check(quasr_954, solved_954), check(quasr_954, str(p))
    assert (a.leak, a.flux, a.flux_stated) == (b.leak, b.flux, b.flux_stated)


# --------------------------------------------------------------------------- 3. each kind, two routes
def _direct_rz(m, n, rc, zs, rs, zc, th, ph):
    """R, Z by the plain sum, and the Cartesian points: the truth for hand-written coefficients."""
    a = np.outer(th, m) - np.outer(ph, n)
    R = np.cos(a) @ rc + np.sin(a) @ rs
    Z = np.sin(a) @ zs + np.cos(a) @ zc
    return np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)


# Known coefficients of a non-symmetric, three-period boundary (n holds no factor of the periods here)
HAND = dict(nfp=3, modes=[(0, 0, 1.00, 0.00, 0.000, 0.00), (0, 1, 0.08, 0.06, 0.010, 0.002),
                          (1, 0, 0.20, 0.25, 0.010, 0.010), (1, 1, 0.03, -0.02, 0.004, -0.003),
                          (1, -1, -0.015, 0.012, 0.002, 0.001), (2, 1, 0.004, 0.005, -0.001, 0.0005),
                          (3, -2, 0.0012, -0.0009, 0.0003, 0.0002)])          # (m, n, rc, zs, rs, zc)


def _hand_arrays():
    m, n, rc, zs, rs, zc = (np.array(v) for v in zip(*HAND["modes"]))
    return m.astype(int), n.astype(int), rc, zs, rs, zc


def _hand_points(th, ph):
    m, n, rc, zs, rs, zc = _hand_arrays()
    return _direct_rz(m, n * HAND["nfp"], rc, zs, rs, zc, th, ph)


def _rand_angles(k=300, seed=3):
    rng = np.random.default_rng(seed)
    return 2 * np.pi * rng.random(k), 2 * np.pi * rng.random(k)


def _write_vmec_input(path, phiedge=0.0123):
    lines = ["&INDATA", "  LASYM = T", f"  NFP = {HAND['nfp']}", f"  PHIEDGE = {phiedge!r}", "  MPOL = 4  NTOR = 2"]
    for (m, n, rc, zs, rs, zc) in HAND["modes"]:
        lines.append(f"  RBC({n},{m}) = {rc!r}  ZBS({n},{m}) = {zs!r}  RBS({n},{m}) = {rs!r}  ZBC({n},{m}) = {zc!r}")
    lines.append("/")
    path.write_text("\n".join(lines).replace("e-", "D-") + "\n")


def test_vmec_input_by_hand(tmp_path):
    p = tmp_path / "input.hand"
    _write_vmec_input(p)
    S = read_surface(str(p))
    th, ph = _rand_angles()
    assert np.max(np.abs(S.eval(th, ph)[0] - _hand_points(th, ph))) < 1e-14
    assert S.nfp == 3 and S.flux == 0.0123


def _write_wout(path, fmt, lib, ns=4):
    """A wout from the hand coefficients (the last radial surface), the inner ones scaled about the m = 0 terms."""
    m, n, rc, zs, rs, zc = _hand_arrays()
    nfp = HAND["nfp"]
    s = np.linspace(0, 1, ns)[:, None]
    shrink = np.where(m[None, :] == 0, 1.0, np.sqrt(s))
    arr = dict(rmnc=rc * shrink, zmns=zs * shrink, rmns=rs * shrink, zmnc=zc * shrink)
    phi = 0.0123 * s.ravel()
    if lib == "scipy":
        from scipy.io import netcdf_file
        f = netcdf_file(str(path), "w")
        f.createDimension("radius", ns)
        f.createDimension("mn_mode", m.size)
        f.createVariable("nfp", "i", ())[...] = nfp
        f.createVariable("ns", "i", ())[...] = ns
        f.createVariable("lasym__logical__", "i", ())[...] = 1
        for k, v in (("xm", m.astype(float)), ("xn", (n * nfp).astype(float))):
            f.createVariable(k, "d", ("mn_mode",))[:] = v
        for k, v in arr.items():
            f.createVariable(k, "d", ("radius", "mn_mode"))[:] = v
        f.createVariable("phi", "d", ("radius",))[:] = phi
        f.close()
        return arr
    import netCDF4
    with netCDF4.Dataset(str(path), "w", format=fmt) as f:
        f.createDimension("radius", ns)
        f.createDimension("mn_mode", m.size)
        f.createVariable("nfp", "i4")[...] = nfp
        f.createVariable("ns", "i4")[...] = ns
        f.createVariable("lasym__logical__", "i4")[...] = 1
        f.createVariable("xm", "f8", ("mn_mode",))[:] = m
        f.createVariable("xn", "f8", ("mn_mode",))[:] = n * nfp
        for k, v in arr.items():
            f.createVariable(k, "f8", ("radius", "mn_mode"))[:] = v
        f.createVariable("phi", "f8", ("radius",))[:] = phi
    return arr


def _check_wout(path, arr):
    th, ph = _rand_angles()
    S = read_surface(str(path))
    assert np.max(np.abs(S.eval(th, ph)[0] - _hand_points(th, ph))) < 1e-14
    assert S.nfp == 3 and abs(S.flux - 0.0123) < 1e-16 and "outermost of 4" in S.describe()
    m, n, *_ = _hand_arrays()
    inner = read_surface(str(path), which=1)
    x = _direct_rz(m, n * 3, arr["rmnc"][1], arr["zmns"][1], arr["rmns"][1], arr["zmnc"][1], th, ph)
    assert np.max(np.abs(inner.eval(th, ph)[0] - x)) < 1e-14
    with pytest.raises(ValueError, match="radial surfaces"):
        read_surface(str(path), which=9)


def test_wout_netcdf3_by_scipy(tmp_path):
    p = tmp_path / "wout_hand.nc"
    _check_wout(p, _write_wout(p, None, "scipy"))


@pytest.mark.parametrize("fmt", ["NETCDF3_CLASSIC", "NETCDF4"])
def test_wout_by_netcdf4(tmp_path, fmt):
    pytest.importorskip("netCDF4")
    p = tmp_path / f"wout_{fmt}.nc"
    _check_wout(p, _write_wout(p, fmt, "netCDF4"))


def test_wout_netcdf4_by_h5py(tmp_path, monkeypatch):
    """The NetCDF4 (HDF5) kind read where netCDF4 is not installed."""
    pytest.importorskip("netCDF4")
    pytest.importorskip("h5py")
    p = tmp_path / "wout_h5.nc"
    arr = _write_wout(p, "NETCDF4", "netCDF4")
    monkeypatch.setitem(sys.modules, "netCDF4", None)
    _check_wout(p, arr)


def test_netcdf4_without_a_reader_says_what_to_install(tmp_path, monkeypatch):
    pytest.importorskip("netCDF4")
    p = tmp_path / "wout_h5.nc"
    _write_wout(p, "NETCDF4", "netCDF4")
    monkeypatch.setitem(sys.modules, "netCDF4", None)
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r"pip install \"\.\[hdf5\]\""):
        read_surface(str(p))


def test_form_dictionaries_against_the_evaluators_own_sums():
    """Both forms of results/evaluator/forms.py, evaluated by forms.Surface."""
    rng = np.random.default_rng(5)
    J = 40
    m, n = rng.integers(0, 6, J), rng.integers(-8, 9, J)
    m[0], n[0] = 0, 0
    rz = dict(form="rz_fourier", m=m.tolist(), n=n.tolist(), rc=(1e-2 * rng.normal(size=J)).tolist(),
              zs=(1e-2 * rng.normal(size=J)).tolist(), rs=(1e-3 * rng.normal(size=J)).tolist(),
              zc=(1e-3 * rng.normal(size=J)).tolist())
    rz["rc"][0] = 1.0
    xyz = dict(form="xyz_fourier", m=m.tolist(), n=n.tolist(),
               **{k: (1e-2 * rng.normal(size=J)).tolist() for k in FORMS.XYZ_KEYS})
    th, ph = _rand_angles()
    for d in (rz, xyz):
        mine = read_surface(d).eval(th, ph)
        theirs = FORMS.Surface(d).eval(th, ph)
        for a, b in zip(mine, theirs):
            assert np.max(np.abs(a - b)) < 1e-14


# ---- SIMSOPT (in the environment that holds it)
def _stored_xyz(design="0000954"):
    with open(example("quasr_0000954.json")) as fh:
        d = json.load(fh)
    from torux.io.simsopt_json import Simson
    sim = Simson(d)
    nm = [r["value"] for r in sim.graph[0]][-1]
    return sim, nm


def _simsopt_points(s, th, ph):
    """SIMSOPT's gamma and its derivatives at (theta, phi) = 2 pi (u, v), each point its own copy of s."""
    from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
    out = [[], [], []]
    for a, b in zip(th / (2 * np.pi), ph / (2 * np.pi)):
        kw = dict(nfp=s.nfp, stellsym=s.stellsym, mpol=s.mpol, ntor=s.ntor, quadpoints_phi=[b], quadpoints_theta=[a])
        if isinstance(s, SurfaceXYZTensorFourier):
            e = SurfaceXYZTensorFourier(clamped_dims=s.clamped_dims, **kw)
        else:
            e = SurfaceRZFourier(**kw)
        e.x = s.x
        out[0].append(e.gamma()[0, 0])
        out[1].append(e.gammadash2()[0, 0] / (2 * np.pi))      # d/dtheta
        out[2].append(e.gammadash1()[0, 0] / (2 * np.pi))      # d/dphi
    return [np.array(o) for o in out]


def _simsopt_cases():
    from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
    rng = np.random.default_rng(11)
    sim, nm = _stored_xyz()
    o = sim.objs[nm]
    q = SurfaceXYZTensorFourier(nfp=o["nfp"], stellsym=o["stellsym"], mpol=o["mpol"], ntor=o["ntor"])
    q.x = sim.dofs_x(o)                                                    # QUASR 954's outermost stored surface
    g = SurfaceXYZTensorFourier(nfp=3, stellsym=False, mpol=4, ntor=3, clamped_dims=[True, False, True])
    x = 1e-3 * rng.normal(size=g.x.size)
    g.x = x
    g.set("x(0,0)", 1.0); g.set("x(1,0)", 0.2); g.set("z(5,0)", 0.2)      # a torus of radius 1, minor radius 0.2
    r = SurfaceRZFourier(nfp=4, stellsym=False, mpol=5, ntor=4)
    r.x = 1e-3 * rng.normal(size=r.x.size)
    r.set_rc(0, 0, 1.0); r.set_rc(1, 0, 0.15); r.set_zs(1, 0, 0.15)
    return [("quasr", q), ("tensor_clamped", g), ("rz", r)]


@pytest.mark.parametrize("case", ["quasr", "tensor_clamped", "rz"])
def test_simsopt_object_its_json_and_its_gamma(case, tmp_path, quasr_954):
    pytest.importorskip("simsopt")
    from simsopt import save
    s = dict(_simsopt_cases())[case]
    th, ph = _rand_angles(60, seed=7)
    live = read_surface(s)
    want = _simsopt_points(s, th, ph)
    for a, b in zip(live.eval(th, ph), want):
        assert np.max(np.abs(a - b)) < 1e-13
    p = tmp_path / f"{case}.json"
    save([s], str(p))
    saved = read_surface(str(p))
    for a, b in zip(saved.eval(th, ph), live.eval(th, ph)):
        assert np.max(np.abs(a - b)) < 1e-15
    assert live.nfp == s.nfp == saved.nfp
    a, b = check(quasr_954, s), check(quasr_954, str(p))
    assert abs(a.leak - b.leak) <= 1e-15 * max(1.0, a.leak) and abs(a.flux / b.flux - 1) < 1e-14


def test_quasr_download_outermost_and_which(quasr_954):
    """A QUASR download as the surface: the outermost by default, which= any other. Its outermost is the surface
    whose flux is the file's request."""
    f = example("quasr_0000954.json")
    sim, nm = _stored_xyz()
    names = [r["value"] for r in sim.graph[0]]
    outer = read_surface(f)
    assert nm in outer.describe() and f"outermost of {len(names)}" in outer.describe()
    inner = read_surface(f, which=0)
    assert names[0] in inner.describe() and inner.volume() < outer.volume()
    assert names[1] in read_surface(f, which=names[1]).describe()
    c = check(quasr_954, f)
    # QUASR's flux is the loop at phi = 0 alone. check's sections differ from it by the surface's own leak
    assert abs(c.flux / quasr_954.request["flux_Wb"] - 1) < c.flux_spread
    with pytest.raises(ValueError, match="none of the file's"):
        read_surface(f, which="nothing")


# ---- DESC and VMEC++ (where they install)
def test_desc_surface_and_equilibrium():
    pytest.importorskip("desc")
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.grid import Grid
    m, n, rc, zs, rs, zc = _hand_arrays()
    # a non-symmetric three-period boundary in DESC's double Fourier series, built from the hand coefficients
    from desc.vmec_utils import ptolemy_identity_fwd
    keep = ~((m == 0) & (n < 0))
    mm, nn, R = ptolemy_identity_fwd(m[keep], n[keep], s=rs[keep], c=rc[keep])
    _, _, Z = ptolemy_identity_fwd(m[keep], n[keep], s=zs[keep], c=zc[keep])
    modes = np.column_stack([mm, nn]).astype(int)
    surf = FourierRZToroidalSurface(R_lmn=R.ravel(), Z_lmn=Z.ravel(), modes_R=modes, modes_Z=modes, NFP=3, sym=False,
                                    check_orientation=False)
    th, ph = _rand_angles(200)
    grid = Grid(np.column_stack([np.ones_like(th), th, ph]), sort=False)
    d = surf.compute(["R", "Z", "phi"], grid=grid)
    want = np.stack([d["R"] * np.cos(d["phi"]), d["R"] * np.sin(d["phi"]), d["Z"]], -1)
    got = read_surface(surf).eval(th, ph)[0]
    # the truth is DESC's own evaluation of the surface it holds (whatever the construction above made of the hand
    # boundary): the reader's points are DESC's
    assert np.max(np.abs(got - want)) < 1e-13
    eq = Equilibrium(surface=surf, Psi=0.0123)
    S = read_surface(eq)
    b = eq.surface
    d = b.compute(["R", "Z", "phi"], grid=grid)
    want = np.stack([d["R"] * np.cos(d["phi"]), d["R"] * np.sin(d["phi"]), d["Z"]], -1)
    assert np.max(np.abs(S.eval(th, ph)[0] - want)) < 1e-13 and abs(S.flux - 0.0123) < 1e-15


def test_vmecpp_output_and_its_wout_file(tmp_path):
    vmecpp = pytest.importorskip("vmecpp")
    p = tmp_path / "input.hand"
    # VMEC++ solves a stellarator-symmetric version of the hand boundary at zero pressure
    lines = ["&INDATA", "  LASYM = F", "  NFP = 3", "  PHIEDGE = 0.0123", "  MPOL = 5", "  NTOR = 3",
             "  NS_ARRAY = 9 25", "  NITER_ARRAY = 2000 4000", "  FTOL_ARRAY = 1.0E-10 1.0E-12", "  NCURR = 1",
             "  AC = 0.0", "  AM = 0.0", "  PRES_SCALE = 0.0", "  CURTOR = 0.0", "  LFREEB = F",
             "  RAXIS_CC = 1.0", "  ZAXIS_CS = 0.0"]
    for (mm, nn, a, b, *_r) in HAND["modes"]:
        lines.append(f"  RBC({nn},{mm}) = {a!r}  ZBS({nn},{mm}) = {b!r}")
    p.write_text("\n".join(lines + ["/"]) + "\n")
    out = vmecpp.run(vmecpp.VmecInput.from_file(str(p)))
    w = tmp_path / "wout_hand.nc"
    out.wout.save(str(w))
    th, ph = _rand_angles()
    a, b = read_surface(out), read_surface(str(w))
    assert np.max(np.abs(a.eval(th, ph)[0] - b.eval(th, ph)[0])) < 1e-15
    assert a.nfp == 3 and abs(a.flux - 0.0123) < 1e-12 and abs(b.flux - 0.0123) < 1e-12
    # the boundary VMEC++ was given encloses the volume VMEC++ reports
    assert abs(a.volume(256) / float(out.wout.volume_p) - 1) < 1e-3


# --------------------------------------------------------------------------- 4. surfaces that are not flux surfaces
def _torus_dict(R0, a, nfp_ripple=0, ripple=0.0, figure_eight=False):
    m, n, rc, zs = [0, 1], [0, 0], [R0, a], [0.0, a]
    if figure_eight:
        m, n, rc, zs = [0, 1, 2], [0, 0, 0], [R0, a, 0.0], [0.0, 0.0, a]
    if nfp_ripple:
        m, n, rc, zs = m + [1], n + [nfp_ripple], rc + [ripple], zs + [ripple]
    return dict(form="rz_fourier", m=m, n=n, rc=rc, zs=zs)


def test_a_torus_through_the_coils_leaks_and_returns(quasr_954):
    p = quasr_954.parts[0]
    x, _ = p.points(np.array([0.0]))
    x = np.asarray(x).reshape(-1, 3)[0]
    R, Z = math.hypot(x[0], x[1]), x[2]
    a = math.hypot(R - 1.0, Z)                                            # through the first coil's start point
    t0 = time.time()
    c = check(quasr_954, _torus_dict(1.0, a))
    assert time.time() - t0 < 60
    assert c.leak > 0.1 and np.isfinite(c.leak)


def test_a_self_intersecting_surface_leaks(quasr_954):
    c = check(quasr_954, _torus_dict(1.0, 0.08, figure_eight=True))
    assert c.leak > 0.1 and np.isfinite(c.flux)


def test_contradicting_field_periods_are_measured_and_said(quasr_954):
    assert quasr_954.nfp == 2
    c = check(quasr_954, _torus_dict(1.0, 0.1, nfp_ripple=3, ripple=0.01))
    assert c.nfp == 3 and c.source_nfp == 2 and np.isfinite(c.leak)
    assert "field periods          3 (the coils: 2)" in str(c)


# --------------------------------------------------------------------------- what is said, and what is refused
def test_the_print(quasr_954):
    c = check(quasr_954, example("quasr_0000954.json"), flux=0.07)
    s = str(c)
    lines = s.splitlines()
    assert len(lines) <= 7
    # five rows, each a label and its text (a long text runs on under its own, indented)
    assert [ln.split("  ")[0] for ln in lines if not ln.startswith(" ")] == ["surface", "field periods",
                                                                            "field across it", "toroidal flux",
                                                                            "stated flux"]
    assert "cross-sections" in s and "rounding" in s and ("larger" in s or "smaller" in s)
    assert "M, N = " in lines[0] and lines[1].split()[-1] == "2"
    for word in ("verified", " ok", "pass", "fail", "good"):
        assert word not in s.lower()
    assert c.flux_stated == 0.07 and c.flux_stated_by == "given"
    assert abs(c.flux_error - (c.flux / 0.07 - 1)) < 1e-15


def test_refusals_say_what_is_read(quasr_954, tmp_path):
    with pytest.raises(ValueError, match="VMEC input file"):
        read_surface(object())
    with pytest.raises(ValueError, match="no surface"):
        read_surface(dict(status="no_surface", surface=None))
    p = tmp_path / "x.txt"
    p.write_text("nothing here\n")
    with pytest.raises(ValueError, match="It reads"):
        read_surface(str(p))
    with pytest.raises(ValueError, match="which="):
        read_surface(_torus_dict(1.0, 0.1), which=2)


def test_the_command_line(capsys, tmp_path):
    ap = argparse.ArgumentParser(prog="torux")
    add_command(ap.add_subparsers(dest="cmd"))
    f = example("quasr_0000954.json")
    assert run_command(ap.parse_args(["check", f, f, "--which", "0"])) == 0
    out = capsys.readouterr().out
    assert out.startswith("surface") and "field across it" in out
    assert run_command(ap.parse_args(["check", f, str(tmp_path / "missing.nc")])) == 1
    assert "error" in capsys.readouterr().err
    with pytest.raises(SystemExit):             # the surface is missing: argparse says so, as for any usage error
        ap.parse_args(["check", f])
    assert "SURFACE" in capsys.readouterr().err


# --------------------------------------------------------------------------- 5. fast, and the same every time
def test_seconds_for_a_quasr_design(quasr_954):
    f = example("quasr_0000954.json")
    check(quasr_954, f)                                                     # compiled and warm
    t0 = time.time()
    a = check(quasr_954, f)
    assert time.time() - t0 < 5
    b = check(quasr_954, f)
    assert (a.leak, a.leak_median, a.at, a.flux, a.flux_spread) == (b.leak, b.leak_median, b.at, b.flux, b.flux_spread)
