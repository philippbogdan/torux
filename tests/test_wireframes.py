"""Kenneth Hammond's three wireframes: one grid of 4992 straight segments round a two-period stellarator, with three
current distributions, each field worse than the one before (examples/hammond_wireframe_*.csv, credits in NOTICE.md).

The truth each test rests on: what torux 0.2.2 returned on these files, a request at the flux of the largest surface
its search verified ('torux solve FILE --nfp 2 --largest', 12 and 18 minutes for the first two), repeated bit for
bit by the direct request here. For the third, what a request returns at the flux where the family from its axis
stopped, on Intel, AMD and Apple processors. The second and third take seconds, the first a minute or two: run it
with -m slow.
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
# the torus at the flux where the family from the third's axis stopped, verified or not
WIREFRAME_3 = dict(flux_Wb=3.565731470221449e-05, iota=0.1624923114296464, axis_iota=0.1844633737416216,
                   volume=1.81623142436e-4, aspect_ratio=147.677)


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
    res = torux.solve(wireframe(1), flux=WIREFRAME_1["flux_Wb"], budget_s=3300)   # an hour, as the certificates
    if res.status in ("undetermined", "timeout"):
        pytest.skip(f"{res.status} after {res.wall_s:.0f} s on this machine")
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


def test_third_holds_a_thin_torus_inboard_of_the_others():
    """The worst field. Where the family from its magnetic axis stops, the solver finds a torus of major radius
    0.59 m, against 1.04 m for the surfaces of the other two, and 4 mm minor radius: aspect ratio 148. The
    independent evaluator puts the field across it under 1e-13 of |B|, either side of its 2e-14 bar with the
    processor: verified on Intel and AMD (under 8e-15), refused on an Apple M4 (4e-14 to 5e-14). Either answer, the
    same torus."""
    from torux.answer import _size
    res = torux.solve(wireframe(3), flux=WIREFRAME_3["flux_Wb"])
    if res.status in ("undetermined", "timeout"):
        pytest.skip(f"{res.status} after {res.wall_s:.0f} s on this machine")
    check = res.evidence["independent"]
    assert res.status == ("surface" if check["ok"] else "surface_unverified"), (res.status, check["ok"])
    assert max(check["bn_max"], check["bn_sup"]) < 1e-13
    T = res.surface
    assert abs(abs(float(T.flux)) / WIREFRAME_3["flux_Wb"] - 1) < 1e-12
    assert abs(res.__dict__["_iota"] - WIREFRAME_3["iota"]) < 1e-11       # the torus's, verified or not
    axes = [a["iota"] for a in res.evidence["solver"]["axes"] if np.isfinite(a["iota"])]
    assert any(abs(a - WIREFRAME_3["axis_iota"]) < 1e-11 for a in axes), axes
    g = _size(T)
    assert abs(g["volume"] / WIREFRAME_3["volume"] - 1) < 1e-8
    aspect = (g["volume"] / (2 * np.pi * g["mean_section"])) / np.sqrt(g["mean_section"] / np.pi)
    assert abs(aspect / WIREFRAME_3["aspect_ratio"] - 1) < 1e-5
