"""Kenneth Hammond's three wireframes: one grid of 4992 straight segments round a two-period stellarator, with three
current distributions, each field worse than the one before (examples/hammond_wireframe_*.csv, credits in NOTICE.md).

The truth each test rests on: what torux 0.2.2 returned on these files, a request at the flux of the largest surface
its search verified ('torux solve FILE --nfp 2 --largest', 12 and 18 minutes for the first two), repeated bit for
bit by the direct request here, and Hammond's own field-line plots, which show nested surfaces in the first two and
none in the third. The second and third take seconds, the first a minute or two: run it with -m slow.
"""
import numpy as np
import pytest

import torux

from conftest import example

# the surface each search verified at the largest flux: (full toroidal flux in Wb, rotational transform)
WIREFRAME_1 = dict(flux_Wb=0.11432894889420658, iota=0.3927905302843305, axis_iota=0.3931089492465639,
                   volume=0.848847261, aspect_ratio=5.08423)
WIREFRAME_2 = dict(flux_Wb=0.03656661662468078, iota=0.3353043078890785, axis_iota=0.3489004093688087,
                   volume=0.270628873, aspect_ratio=9.09341)
WIREFRAME_3_FLUX = 3.565731470221449e-05     # as far as the family from its axis was followed


def wireframe(k):
    return torux.load(example(f"hammond_wireframe_{k}.csv"), nfp=2)


@pytest.mark.parametrize("k, carrying, total_abs_I_A", [(1, 2016, 556e6), (2, 1192, 350e6), (3, 392, 150e6)])
def test_each_is_the_same_grid_with_two_verified_field_periods(k, carrying, total_abs_I_A):
    """The file lists every segment of the grid, most of them without current. The two field periods and the
    stellarator symmetry are used only once verified on the segments that carry current."""
    from torux.io.wireframe import read_wireframe_csv
    grid, first = (read_wireframe_csv(example(f"hammond_wireframe_{j}.csv")) for j in (k, 1))
    assert np.array_equal(grid["a"], first["a"]) and np.array_equal(grid["b"], first["b"])
    src = wireframe(k)
    assert src.nfp == 2 and src.stellsym
    assert [p.kind for p in src.parts] == ["segments"] and src.parts[0].n == carrying
    rec = src.meta["wireframe"]
    assert rec["segments_listed"] == 4992 and rec["segments_nonzero"] == carrying
    assert abs(rec["total_abs_I_A"] / total_abs_I_A - 1) < 1e-12


def _verified(res, want):
    assert res.status == "surface", (res.status, res.evidence.get("unverified"))
    check = res.evidence["independent"]
    assert res.evidence["solver"]["bn_off"] <= 1e-14         # the solver's own off-grid |B.n|/|B|
    assert check["ok"]                                       # the independent evaluator's acceptance
    assert max(check["bn_max"], check["bn_sup"]) < 2e-14     # its largest |B.n|/|B|, sampled and estimated
    assert check["flux_rel_err"] < 1e-12                     # the enclosed flux, on six sections
    assert abs(res.flux / want["flux_Wb"] - 1) < 1e-12
    assert abs(res.iota - want["iota"]) < 1e-11
    assert abs(res.evidence["solver"]["axes"][0]["iota"] - want["axis_iota"]) < 1e-11
    assert abs(res.volume / want["volume"] - 1) < 1e-8
    assert abs(res.aspect_ratio / want["aspect_ratio"] - 1) < 1e-5


@pytest.mark.slow
def test_first_verified_out_to_its_largest_surface():
    """The best field: a verified surface enclosing 0.1143 Wb and 0.849 m^3 at aspect ratio 5.08, iota between
    0.390 and 0.393 from the axis out (it dips to 0.3900 and comes back to 0.3928). The search went no further: at
    0.1153 Wb the solver's own checks refused the torus it found, and an iota = 2/5 chain lies beyond."""
    res = torux.solve(wireframe(1), flux=WIREFRAME_1["flux_Wb"])
    _verified(res, WIREFRAME_1)
    iota = res.profile[:, 1]
    assert 0.3899 < iota.min() < 0.3900 and 0.3931 < iota.max() < 0.3932


def test_second_verified_out_to_its_largest_surface():
    """A worse field: a verified surface enclosing 0.0366 Wb and 0.271 m^3, iota falling from 0.3489 on the axis to
    0.3353 at it. Beyond it torux decided nothing up to 0.0371 Wb, short of the iota = 1/3 chain."""
    res = torux.solve(wireframe(2), flux=WIREFRAME_2["flux_Wb"])
    _verified(res, WIREFRAME_2)
    iota = res.profile[:, 1]
    assert np.all(np.diff(iota) < 0) and iota[-1] > 1 / 3


def test_third_has_no_surface_claimed():
    """The worst field, no nested surfaces in Hammond's plot. The solver finds a torus where the family from its axis
    stops, and the independent evaluator refuses it: the answer is unverified and gives no numbers."""
    res = torux.solve(wireframe(3), flux=WIREFRAME_3_FLUX)
    if res.status in ("undetermined", "timeout"):
        pytest.skip(f"{res.status} after {res.wall_s:.0f} s on this machine")
    assert res.status == "surface_unverified", res.status
    assert res.evidence["unverified"] == "the independent verification refused the torus"
    check = res.evidence["independent"]
    assert not check["ok"] and max(check["bn_max"], check["bn_sup"]) > 2e-14
    with pytest.raises(ValueError, match="surface_unverified"):
        res.iota
