"""A verified surface at the limit of double-precision arithmetic."""
import numpy as np

from conftest import QUASR_954


def test_request_is_the_flux_of_the_stored_surface(quasr_954):
    assert abs(quasr_954.request["flux_Wb"] / QUASR_954["flux_Wb"] - 1) < 1e-12


def test_status_and_both_checks(solved_954):
    res = solved_954
    assert res.status == "surface"
    solver, check = res.evidence["solver"], res.evidence["independent"]
    assert solver["bn_off"] <= 1e-14                      # the solver's own off-grid |B.n|/|B|
    assert check["ok"]                                    # the independent evaluator's acceptance
    assert max(check["bn_max"], check["bn_sup"]) < 2e-14  # its largest |B.n|/|B|, sampled and estimated
    assert check["flux_rel_err"] < 1e-12                  # the enclosed flux, on six sections
    assert abs(abs(res.surface.flux) / QUASR_954["flux_Wb"] - 1) < 1e-12


def test_rotational_transform_matches_the_table(solved_954):
    assert abs(abs(solved_954.iota) - abs(QUASR_954["iota"])) < 1e-11


def test_field_is_tangent_at_points_never_used_by_the_solve(solved_954):
    """|B.n|/|B| at random points of the surface, with the normal from finite differences of the surface itself."""
    res = solved_954
    rng = np.random.default_rng(1)
    th, ph = 2 * np.pi * rng.random(200), 2 * np.pi * rng.random(200)
    h = 1e-6
    x = res.points(th, ph)
    xt = (res.points(th + h, ph) - res.points(th - h, ph)) / (2 * h)
    xp = (res.points(th, ph + h) - res.points(th, ph - h)) / (2 * h)
    n = np.cross(xt, xp)
    n /= np.linalg.norm(n, axis=1)[:, None]
    B = res.field(x)
    bn = np.abs(np.sum(B * n, axis=1)) / np.linalg.norm(B, axis=1)
    assert bn.max() < 1e-8                                # limited by the finite-difference normal, not the surface


def test_result_round_trips_through_json(solved_954, tmp_path):
    import json
    p = tmp_path / "result.json"
    solved_954.save(p)
    d = json.loads(p.read_text())
    assert d["status"] == "surface"
    s = d["surface"]
    assert np.array_equal(np.array(s["rc"]), np.asarray(solved_954.surface.rc))
    assert s["M"] == solved_954.surface.M and s["N"] == solved_954.surface.N
