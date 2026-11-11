"""Certified no-surface answers.  Minutes each on a recent desktop: run with -m slow.

A certificate needs time, and how much depends on the machine: the tests give the solver an hour, as the results in
results/compare300.csv do, and it returns as soon as it has its answer (two to three minutes on a 2023 desktop
processor).  Where that is not enough, the answer is 'undetermined' and the test is skipped. What never happens is a
surface, because none exists at these fluxes, and an error is a failure.
"""
import pytest

import torux

from conftest import example

pytestmark = pytest.mark.slow


def certificate(name):
    src = torux.load(example(name))
    res = torux.solve(src, budget_s=3300)
    path = (res.record or {}).get("path")
    assert res.status not in ("surface", "range", "error"), (res.status, path, res.evidence.get("error"))
    if res.status != "no_surface":
        pytest.skip(f"{res.status} after {res.wall_s:.0f} s on this machine")
    c = res.evidence["certificate"]
    assert c["ok"], (c.get("kind"), path)
    return abs(src.request["flux_Wb"]), c


def test_edge_barrier():
    """QUASR 1100073: the requested flux lies beyond the last surface. A barrier of manifolds bounds every surface."""
    flux, c = certificate("quasr_1100073.json")
    assert c["kind"] == "edge_barrier", c["kind"]
    assert c["F_up"] < flux                              # no surface encloses more than F_up


def test_island_band():
    """QUASR 0219457: the requested flux falls inside an island chain."""
    flux, c = certificate("quasr_0219457.json")
    assert c["kind"].startswith("island_band"), c["kind"]
    lo, hi = c["interval"]
    assert lo < flux < hi                                # no surface encloses a flux in (lo, hi)
