"""Sources of several kinds, from files and built in code."""
import numpy as np
import pytest

import torux

from conftest import example


def test_quasr_download_brings_its_surfaces_and_request(quasr_954):
    assert quasr_954.nfp == 2 and quasr_954.stellsym
    assert quasr_954.request["flux_Wb"] > 0
    assert len(quasr_954._design_surfaces) >= 1
    assert quasr_954.meta["kind"] == "stellarator"


def test_fourier_filament_json():
    src = torux.load(example("hsx.json"))
    assert src.nfp == 4 and src.stellsym
    assert [p.kind for p in src.parts] == ["fourier"] and src.parts[0].n == 48


def test_gpec_coil_geometry_is_a_tokamak_source():
    src = torux.load(example("d3d_icoils_upper.dat"))
    assert src.meta["kind"] == "tokamak"
    assert src.parts[0].kind == "polyline" and src.parts[0].n == 6


def test_source_built_in_code_round_trips(tmp_path):
    coef = np.zeros((1, 3, 3))
    coef[0, 0, 2] = 1.0
    coef[0, 1, 1] = 1.0
    src = torux.Source(id="ring", parts=[torux.FourierCurves(coef, np.array([1e5]))])
    p = tmp_path / "ring.json"
    src.save(p)
    back = torux.load(p)
    assert np.array_equal(back.parts[0].coef, coef) and back.parts[0].current[0] == 1e5


def test_unknown_format_is_refused(tmp_path):
    p = tmp_path / "nothing.txt"
    p.write_text("not a coil file\n")
    with pytest.raises(ValueError):
        torux.load(p)


def test_perturbation_coils_hold_no_surface():
    """DIII-D's upper I-coils: no vacuum flux surface, by proof (antisymmetry, Ampere, the coils' shadows)."""
    res = torux.solve(example("d3d_icoils_upper.dat"))
    assert res.status == "no_surface"
    assert res.evidence["proof"]["verdict"] == "NO_TOROIDAL_FLUX_SURFACE"
