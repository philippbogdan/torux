"""Never a guess: a request that cannot be settled in its budget is not answered."""
import torux

from conftest import example


def test_a_budget_too_small_claims_nothing():
    res = torux.solve(example("quasr_0000954.json"), budget_s=0.05)
    assert res.status not in ("surface", "no_surface", "range")


def test_without_the_independent_check_agreeing_there_is_no_surface_claim(solved_954):
    """The acceptance rule itself: a check that fails turns 'surface' into 'surface_unverified'."""
    from torux.api import Request, Result, _truthful
    out = Result("surface", 1.0, 0.0, evidence={"independent": {"ok": False}})
    _truthful(out, Request(None, verify=True))
    assert out.status == "surface_unverified"
    v = dict(solved_954.evidence["independent"])
    from torux.verify import acceptance
    assert acceptance(v)
    v_bad = dict(v, bn_sup=5e-14, bn_max=5e-14, bn_sup_dd=5e-14, bn_max_dd=5e-14)
    assert not acceptance(v_bad)                          # a torus off by 5e-14 is refused
    assert not acceptance(dict(v, flux_rel_err=1e-11))    # so is one enclosing another flux
