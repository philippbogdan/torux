"""Surface current sign, derivatives and exterior-domain behaviour, plus the public VMEX case."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import torux
from torux.field import FieldEngine
from torux.field import surface_current
from torux.reference import fast64
from torux.source import MU0


CASES = Path(__file__).parents[1] / "examples/vmex_surface_current"


def circular(**kwargs):
    return torux.SurfaceCurrent(1, [(0, 0), (1, 0)], [1., .2], [0., .2], **kwargs)


def test_surface_potential_sign_gradient_and_coil_superposition():
    current, kappa, secular = 8e3, 130., 2e4
    sheet = circular(I=current, G=secular, kappas=[0., kappa])
    z = .19
    x = np.array([[0., 0., z]])

    def ring_bz(theta):
        r, z0 = 1. + .2*np.cos(theta), .2*np.sin(theta)
        amps = -(current/(2*np.pi) + kappa*np.cos(theta))
        return MU0*amps*r*r/(2*(r*r + (z-z0)**2)**1.5)

    engine = FieldEngine(torux.Source("sheet", [sheet]))
    expected = quad(ring_bz, 0., 2*np.pi, epsabs=1e-14)[0]
    np.testing.assert_allclose(engine.B(x), [[0., 0., expected]], atol=3e-13, rtol=0.)

    x = np.array([[1.4, -.3, .27]])
    raw = circular(covariant=[[[current/(2*np.pi), kappa], [0., 0.]], [[secular/(2*np.pi), 0.], [0., 0.]]])
    raw_engine = FieldEngine(torux.Source("raw", [raw]))
    np.testing.assert_allclose(raw_engine.B(x), engine.B(x), atol=1e-15, rtol=0.)

    v = engine.evaluate(x)
    h = 1e-5
    dB, dA = [], []
    for e in h*np.eye(3):
        a = engine.evaluate(x + e, grad=False)
        b = engine.evaluate(x - e, grad=False)
        dB.append((np.asarray(a["B"] - b["B"])[0])/(2*h))
        dA.append((np.asarray(a["A"] - b["A"])[0])/(2*h))
    np.testing.assert_allclose(v["G"][0], np.asarray(dB).T, atol=2e-11, rtol=1e-8)
    jac = np.asarray(dA).T
    curl = [jac[2, 1]-jac[1, 2], jac[0, 2]-jac[2, 0], jac[1, 0]-jac[0, 1]]
    np.testing.assert_allclose(v["B"][0], curl, atol=2e-11, rtol=1e-8)
    mixed = FieldEngine(torux.Source("sheet + toroidal field", [sheet, torux.ToroidalField(1.)]))
    X, Y, _ = x[0]
    np.testing.assert_allclose(mixed.B(x) - v["B"], np.array([[-Y, X, 0.]])/(X*X + Y*Y), atol=1e-14, rtol=0.)


def test_source_round_trip_and_exterior_only(tmp_path):
    path = tmp_path / "surface.json"
    src = torux.Source("surface", [circular(), torux.ToroidalField(1.)])
    src.save(path)
    restored = torux.load(path)
    assert isinstance(restored.parts[0], torux.SurfaceCurrent)
    np.testing.assert_array_equal(restored.parts[0].modes, src.parts[0].modes)
    eng = FieldEngine(restored)
    for x in ([[1., 0., 0.]], [[1.2, 0., 0.]]):
        with pytest.raises(ValueError, match="inside or too close"):
            eng.B(x)
    with pytest.raises(ValueError, match="starts"):
        torux.poincare(restored, turns=1)
    with pytest.raises(ValueError, match="inside or too close"):
        torux.poincare(restored, starts=[(1.2, 0.)], turns=1)
    with pytest.raises(ValueError, match="flux inside its boundary"):
        torux.solve(restored, flux=.1)
    with pytest.raises(ValueError, match="flux inside its boundary"):
        torux.solve(restored, largest=True)
    inside = tmp_path / "input.inside"
    inside.write_text("&INDATA\n NFP = 1\n PHIEDGE = 0.01\n RBC(0,0) = 1.0\n RBC(0,1) = 0.1\n ZBS(0,1) = 0.1\n/\n")
    with pytest.raises(ValueError, match="inside the SurfaceCurrent's boundary"):
        torux.check(restored, str(inside))
    with pytest.raises(ValueError, match="inside the boundary"):
        fast64.Field(restored).evaluate([[1., 0., 0.]])
    section = torux.poincare(restored, planes=[0., -.5, .5], starts=[(1.21, 0.)], turns=1)
    assert not section.left[0]
    assert section.stop_reason == (None,)
    assert section.R.shape == (1, 3, 1)
    np.testing.assert_allclose(section.R, 1.21, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(section.Z, 0., atol=1e-12, rtol=0.)


def test_exterior_trace_stops_before_recording_interior_points():
    rings = torux.Rings([1.], [0.], [1e5])
    src = torux.Source("boundary-test", [circular(), torux.ToroidalField(1.), rings])
    section = torux.poincare(src, planes=[0.], starts=[(1., -.201)], turns=1)
    assert section.left[0]
    assert section.stop_reason == ("boundary",)
    assert "exterior boundary cutoff" in section.lines_table()
    r, phi, z = section.left_at[0]
    assert np.hypot(r - 1., z) >= .2
    assert 0 < phi < 2*np.pi
    assert np.isnan(section.R).all()


def test_sheet_with_no_toroidal_field_stops_with_a_reason():
    src = torux.Source("poloidal-field", [circular(I=8e3)])
    section = torux.poincare(src, planes=[0.], starts=[(1.4, .25)], turns=1)
    assert section.stop_reason == ("toroidal_direction",)
    assert "toroidal direction lost" in section.lines_table()


def test_surface_targets_require_cartesian_rows():
    engine = FieldEngine(torux.Source("sheet", [circular(I=8e3)]))
    with pytest.raises(ValueError, match="Cartesian points"):
        engine.B(np.zeros((3, 4)))
    assert engine.B(np.empty((0, 3))).shape == (0, 3)
    b = engine.B([1.4, 0., .2])
    np.testing.assert_array_equal(b, engine.B([[1.4, 0., .2]]))


def test_four_period_potential_round_trip_with_separate_coils(tmp_path):
    modes = [(0, 0), (1, 0), (1, 1)]
    r, z, kappa = [1., .2, 0.], [0., .2, 0.], 1300.
    sheet = torux.SurfaceCurrent(4, modes, r, z, kappas=[0., 0., kappa])
    path = tmp_path / "sheet.json"
    torux.Source("sheet", [sheet], nfp=4).save(path)
    coils = tmp_path / "toroidal.json"
    torux.Source("toroidal", [torux.ToroidalField(1.)]).save(coils)
    src = torux.load(coils, path, nfp=4)
    assert src.nfp == 4

    cov = [[[0., 0., kappa], [0., 0., 0.]], [[0., 0., -4*kappa], [0., 0., 0.]]]
    raw = torux.Source("raw", [torux.SurfaceCurrent(4, modes, r, z, covariant=cov), *src.parts[:1]], nfp=4)
    x = np.array([[1.4, .3, .15]])
    rot = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    engine = FieldEngine(src)
    field, rotated = engine.evaluate(x), engine.evaluate(x @ rot)
    np.testing.assert_allclose(field["B"], FieldEngine(raw).B(x), atol=2e-15, rtol=0.)
    np.testing.assert_allclose(rotated["B"], field["B"] @ rot, atol=2e-14, rtol=0.)
    np.testing.assert_allclose(rotated["A"], field["A"] @ rot, atol=2e-14, rtol=0.)
    np.testing.assert_allclose(rotated["G"], rot.T @ field["G"] @ rot, atol=2e-13, rtol=0.)


def test_panel_storage_cap_preserves_quadrature(monkeypatch):
    """Panels kept or built where a target needs them are the same panels: the field agrees to rounding."""
    source = torux.Source("sheet", [circular(I=8e3, G=4e3)])
    x = np.array([[1.202, 0., 0.], [1.5, .1, .2], [1.2002, .01, .003]])
    cached = FieldEngine(source)
    expected = cached.evaluate(x)
    assert cached.sheets[0].cache_depth >= 1 and cached.sheets[0].panel_bytes > 0

    monkeypatch.setattr(surface_current, "CACHE_BYTES", 128)
    fresh = torux.Source("sheet", [circular(I=8e3, G=4e3)])
    uncached = FieldEngine(fresh)
    actual = uncached.evaluate(x)
    assert uncached.sheets[0].cache_depth == 0
    for key in ("B", "A"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-15 * float(np.max(np.abs(expected[key]))), rtol=0.)
    np.testing.assert_allclose(actual["G"], expected["G"], atol=1e-12 * float(np.max(np.abs(expected["G"]))), rtol=0.)


@pytest.mark.parametrize("case", ["current", "beta1"])
def test_public_vmex_direct_field(case):
    src = torux.load(CASES / (case + ".json"))
    ref = json.loads((CASES / "reference.json").read_text())[case]
    engine = FieldEngine(torux.Source(src.id, [src.parts[1]], nfp=src.nfp))
    errors = np.linalg.norm(np.asarray(engine.B(ref["points"])) - ref["B"], axis=1)
    assert errors.max() < 2e-11


def test_public_vmex_island_orbit_through_native_poincare():
    src = torux.load(CASES / "current.json")
    edge = np.sum(src.parts[1].rmnc)
    section = torux.poincare(src, planes=[0.], starts=[(edge + .00497555, 0.)], turns=2)
    assert not section.left[0]
    assert section.followed[0] == 4
    assert abs(section.R[0, 0, 1] - section.start[0, 0]) > .1
    assert np.linalg.norm([section.R[0, 0, -1] - section.start[0, 0], section.Z[0, 0, -1]]) < 1e-5
