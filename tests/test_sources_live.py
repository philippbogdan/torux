"""Sources of every kind: QUASR designs by number, live SIMSOPT and DESC objects.

The truth each test rests on: QUASR's own server and the bundled download (byte for byte). SIMSOPT's BiotSavart.B()
and DESC's compute_magnetic_field, whose field codes share none with torux's.  The SIMSOPT tests run where SIMSOPT
imports, the DESC test where DESC imports. Elsewhere they skip.
"""
import os
import shutil
import socket

import numpy as np
import pytest

from conftest import example

from torux import sources
from torux.api import solve
from torux.field import FieldEngine

HOST = "quasr.flatironinstitute.org"


def _online():
    try:
        socket.create_connection((HOST, 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty torux cache of the test's own."""
    monkeypatch.setenv("TORUX_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def offline(monkeypatch):
    """No network: every socket connection fails."""
    def refuse(*a, **k):
        raise OSError("the network is switched off by the test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _same_source(a, b):
    assert a.nfp == b.nfp and a.stellsym == b.stellsym and a.id == b.id
    assert [p.kind for p in a.parts] == [p.kind for p in b.parts]
    for p, q in zip(a.parts, b.parts):
        assert np.array_equal(p.coef, q.coef) and np.array_equal(p.current, q.current)
    assert a.request["flux_Wb"] == b.request["flux_Wb"]
    assert a.meta["dmin"] == b.meta["dmin"]
    assert len(a._design_surfaces) == len(b._design_surfaces)


# --------------------------------------------------------------------------- QUASR by number
def test_quasr_numbers_written_any_way():
    for s in ("quasr:954", "QUASR 954", "quasr_0000954", "quasr:0954", "Quasr/954", "quasr:serial0000954.json",
              "quasr #954"):
        assert sources.quasr_number(s) == 954, s
    assert sources.quasr_number("quasr:1,100,073") == 1100073
    assert sources.quasr_number("quasr:1_100_073") == 1100073
    for s in ("coils.json", "examples/quasr_0000954.json", "quasr", "w7x"):
        assert sources.quasr_number(s) is None, s
    for s in ("quasr:0", "quasr:12,34", "quasr:10000000"):
        with pytest.raises(ValueError):
            sources.quasr_number(s)


@pytest.mark.skipif(not _online(), reason=f"{HOST} cannot be reached from here")
def test_quasr_download_is_the_file_and_the_same_solve(cache, monkeypatch, quasr_954, solved_954):
    """quasr:954 downloads exactly the bundled file. The second call reads the cache and touches no network. The
    source and its solve are the bundled file's, bit for bit."""
    first = sources.load("quasr:954")
    path = cache / "quasr" / "serial0000954.json"
    with open(path, "rb") as a, open(example("quasr_0000954.json"), "rb") as b:
        assert a.read() == b.read()
    _same_source(first, quasr_954)
    assert first.provenance["quasr"]["id"] == 954

    def refuse(*a, **k):
        raise OSError("the network is switched off by the test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    again = sources.load("quasr 0000954")
    _same_source(again, quasr_954)
    res = solve(again)
    assert res.status == solved_954.status == "surface"
    assert res.iota == solved_954.iota
    assert np.array_equal(res.surface.rc, solved_954.surface.rc)
    assert np.array_equal(res.surface.zs, solved_954.surface.zs)


@pytest.mark.skipif(not _online(), reason=f"{HOST} cannot be reached from here")
def test_quasr_number_with_no_design(cache):
    with pytest.raises(ValueError, match="QUASR has no design 1 "):
        sources.load("quasr:1")
    assert not (cache / "quasr" / "serial0000001.json").exists()


def test_quasr_without_network_says_where_to_download(cache, offline):
    with pytest.raises(ConnectionError) as e:
        sources.load("quasr:954")
    msg = str(e.value)
    assert "https://quasr.flatironinstitute.org/simsopt_serials/0000/serial0000954.json" in msg
    assert str(cache / "quasr" / "serial0000954.json") in msg


def test_quasr_from_the_cache_needs_no_network(cache, offline, quasr_954):
    (cache / "quasr").mkdir(parents=True)
    shutil.copy(example("quasr_0000954.json"), cache / "quasr" / "serial0000954.json")
    _same_source(sources.load("quasr:954"), quasr_954)


def test_files_pass_through_unchanged(quasr_954):
    _same_source(sources.load(example("quasr_0000954.json")), quasr_954)
    src = sources.load(example("d3d_icoils_upper.dat"))
    assert src.meta["kind"] == "tokamak" and src.parts[0].n == 6


def test_unknown_object_is_refused():
    with pytest.raises(TypeError, match="torux cannot read a source of type dict"):
        sources.load({"coils": []})


# --------------------------------------------------------------------------- live SIMSOPT objects
def _points_inside(surfaces, n, rng):
    """Random points on the stored surfaces of a QUASR design, inside the coils."""
    out = []
    for s in surfaces:
        g = np.zeros((n, 3))
        s.gamma_lin(g, rng.uniform(0, 1, n), rng.uniform(0, 1, n))
        out.append(g)
    return np.concatenate(out)


def _rel(B, ref):
    return float(np.max(np.linalg.norm(B - ref, axis=1) / np.linalg.norm(ref, axis=1)))


def _B(src, x):
    return np.asarray(FieldEngine(src, dmin=src.meta.get("dmin", 0.05)).evaluate(x, grad=False, A=False)["B"])


def test_simsopt_biotsavart_as_quasr_has_them(quasr_954):
    """QUASR 954's coils as SIMSOPT holds them (one CurveXYZFourier, its RotatedCurves, ScaledCurrents): torux's
    source is the file's, and its field is SIMSOPT's BiotSavart.B() to rounding."""
    pytest.importorskip("simsopt")
    from simsopt import load as simsopt_load
    from simsopt.field import BiotSavart, coils_via_symmetries
    from simsopt.geo import CurveXYZFourier
    surfaces, coils = simsopt_load(example("quasr_0000954.json"))
    bs = BiotSavart(coils)
    src = sources.load(bs)
    assert src.nfp == 2 and src.stellsym
    assert np.array_equal(src.parts[0].coef, quasr_954.parts[0].coef)
    assert np.array_equal(src.parts[0].current, quasr_954.parts[0].current)
    assert src.provenance["files"] == ["a SIMSOPT BiotSavart object"]
    x = _points_inside(surfaces, 200, np.random.default_rng(1))
    B = _B(src, x)
    # SIMSOPT's field with the design's own 160 quadrature points per coil: its trapezoid rule's error
    bs.set_points(x)
    assert _rel(B, bs.B()) < 1e-11
    # the same coils with 4,096 points per coil, where SIMSOPT's quadrature is exact to rounding
    base = coils[0].curve
    dense = CurveXYZFourier(4096, base.order)
    dense.x = base.x
    bsd = BiotSavart(coils_via_symmetries([dense], [coils[0].current], 2, True))
    bsd.set_points(x)
    err = _rel(B, bsd.B())
    print(f"QUASR 954 coils: largest |B_torux - B_simsopt| / |B| = {err:.2e} at {len(x)} points")
    assert err < 1e-14
    assert np.array_equal(sources.load(bsd).parts[0].coef, quasr_954.parts[0].coef)


def test_simsopt_planar_coils():
    """Tilted, shaped planar coils (CurvePlanarFourier) in three field periods with stellarator symmetry: torux's
    field is SIMSOPT's to rounding at random points inside them."""
    pytest.importorskip("simsopt")
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import CurvePlanarFourier
    rng = np.random.default_rng(7)
    order, nfp = 3, 3
    curves = []
    for i in range(2):
        phi = (i + 0.5) * np.pi / nfp / 2
        q = _qmul([np.cos(phi / 2), 0, 0, np.sin(phi / 2)], [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        c = CurvePlanarFourier(4096, order)
        c.x = np.r_[0.3, 0.02 * rng.standard_normal(order), 0.02 * rng.standard_normal(order),
                    np.asarray(q) + 0.05 * rng.standard_normal(4),
                    np.array([np.cos(phi), np.sin(phi), 0.0]) + 0.02 * rng.standard_normal(3)]
        curves.append(c)
    bs = BiotSavart(coils_via_symmetries(curves, [Current(1e5), Current(1.1e5)], nfp, True))
    src = sources.load(bs, nfp=nfp)
    assert src.nfp == nfp and src.stellsym
    assert [p.kind for p in src.parts] == ["fourier"] and src.parts[0].n == 12
    n = 400
    R, Z, P = 1 + 0.15 * rng.uniform(-1, 1, n), 0.15 * rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)
    x = np.stack([R * np.cos(P), R * np.sin(P), Z], 1)
    bs.set_points(x)
    err = _rel(_B(src, x), bs.B())
    print(f"planar coils: largest |B_torux - B_simsopt| / |B| = {err:.2e} at {n} points")
    assert err < 1e-14


def test_simsopt_coil_and_list_of_coils(quasr_954):
    pytest.importorskip("simsopt")
    from simsopt import load as simsopt_load
    _, coils = simsopt_load(example("quasr_0000954.json"))
    every = sources.load(list(coils))
    assert np.array_equal(every.parts[0].coef, quasr_954.parts[0].coef)
    one = sources.load(coils[2])
    assert one.parts[0].n == 1 and np.array_equal(one.parts[0].coef[0], quasr_954.parts[0].coef[2])
    with pytest.raises(ValueError, match="BiotSavart, a Coil or a list of Coils"):
        sources.load(coils[0].curve)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


# --------------------------------------------------------------------------- live DESC objects
def test_desc_coil_sets():
    """DESC coil sets of every class torux reads exactly, saved by DESC's own save and read by torux's reader of
    DESC coil sets: their field is DESC's compute_magnetic_field to rounding."""
    pytest.importorskip("desc")
    import scipy.constants
    # DESC takes mu_0 from scipy.constants (CODATA 2022: 1.25663706127e-6 H/m), torux 4 pi 1e-7 H/m as SIMSOPT and
    # VMEC do: the two fields differ by that ratio, 1.3e-10, which is taken out before they are compared
    mu_ratio = 4e-7 * np.pi / scipy.constants.mu_0
    from desc.coils import CoilSet, FourierPlanarCoil, FourierRZCoil, FourierXYZCoil, MixedCoilSet
    planar = FourierPlanarCoil(current=1e5, center=[1.0, 0.0, 0.0], normal=[0.0, 1.0, 0.2], r_n=0.3)
    ring = CoilSet.linspaced_angular(planar, n=6)
    rz = FourierRZCoil(current=-3e4, R_n=1.6, Z_n=0.0)
    xyz = FourierXYZCoil(current=2e4)
    rng = np.random.default_rng(3)
    n = 300
    R, Z, P = 1 + 0.12 * rng.uniform(-1, 1, n), 0.12 * rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)
    x = np.stack([R * np.cos(P), R * np.sin(P), Z], 1)
    for obj in (planar, ring, MixedCoilSet(ring, rz), xyz):
        src = sources.load(obj)
        ref = np.asarray(obj.compute_magnetic_field(x, basis="xyz", source_grid=4096)) * mu_ratio
        err = _rel(_B(src, x), ref)
        print(f"DESC {type(obj).__name__}: largest |B_torux - B_desc| / |B| = {err:.2e}")
        assert err < 1e-13, type(obj).__name__
    assert not any(os.path.exists(f) for f in src.provenance["files"])
