"""A surface asked for by its minor radius or volume, and the largest surface the coils hold (torux.ask).

Truth that shares no code with torux.ask: SIMSOPT 1.11.1's own SurfaceRZFourier measures of the surfaces returned
(in the environment that holds SIMSOPT. Elsewhere those tests skip), closed forms for tori whose size is known
exactly, and the shipped solver's own answer to a direct request at the flux each search returned."""
import math
import os
import time

import numpy as np
import pytest

import torux
from torux import api, ask

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
Q954 = os.path.join(EX, "quasr_0000954.json")
Q1100073 = os.path.join(EX, "quasr_1100073.json")
HSX = os.path.join(EX, "hsx.json")
D3D = os.path.join(EX, "d3d_icoils_upper.dat")


def _simsopt_size(T):
    """Volume and minor radius of a torux surface by SIMSOPT's own SurfaceRZFourier, its coefficients copied in
    (R = sum rc[m, n + N] cos(m theta - n nfp phi) is SIMSOPT's convention), on a full-torus grid fine enough for
    its quadrature to be exact."""
    simsopt_geo = pytest.importorskip("simsopt.geo")
    assert float(getattr(T, "phi0", 0.0)) == 0.0
    M, N, nfp = int(T.M), int(T.N), int(T.nfp)
    nth, nph = 3 * M + 4, nfp * (3 * N + 4) + 4
    s = simsopt_geo.SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=M, ntor=N,
                                     quadpoints_phi=np.arange(nph) / nph, quadpoints_theta=np.arange(nth) / nth)
    rc, zs = np.asarray(T.rc), np.asarray(T.zs)
    for m in range(M + 1):
        for n in range(-N, N + 1):
            if m == 0 and n < 0:
                continue
            s.set_rc(m, n, float(rc[m, n + N]))
            if not (m == 0 and n == 0):
                s.set_zs(m, n, float(zs[m, n + N]))
    # the m = 0, n < 0 entries of the torus repeat n > 0 (cos and sin of -n psi): fold them into n > 0
    if N:
        for n in range(1, N + 1):
            s.set_rc(0, n, float(rc[0, N + n] + rc[0, N - n]))
            s.set_zs(0, n, float(zs[0, N + n] - zs[0, N - n]))
    return dict(volume=abs(float(s.volume())), minor_radius=float(s.minor_radius())), s


def _same_surface(a, b):
    return (a.surface is not None and b.surface is not None and a.surface.M == b.surface.M
            and a.surface.N == b.surface.N and np.array_equal(np.asarray(a.surface.rc), np.asarray(b.surface.rc))
            and np.array_equal(np.asarray(a.surface.zs), np.asarray(b.surface.zs)))


def _requests(res):
    s = res.evidence["search"]
    return [(e["flux"], e["status"], e["wall_s"]) for e in s["requests"]]


# --------------------------------------------------------------------------- quick
def test_one_way_to_ask():
    for kw in (dict(flux=0.05, minor_radius=0.1), dict(volume=0.3, largest=True), dict(flux=0.05, largest=True)):
        with pytest.raises(ValueError, match="give one of"):
            ask.solve(Q954, **kw)
    for kw in (dict(minor_radius=-0.1), dict(volume=0.0), dict(minor_radius=float("nan")), dict(volume="big")):
        with pytest.raises(ValueError, match="positive number"):
            ask.solve(Q954, **kw)


def _torus(rc_entries, zs_entries, M, N, nfp):
    from torux.torus import TorusState
    rc = np.zeros((M + 1, 2 * N + 1))
    zs = np.zeros((M + 1, 2 * N + 1))
    for (m, n), v in rc_entries.items():
        rc[m, n + N] = v
    for (m, n), v in zs_entries.items():
        zs[m, n + N] = v
    return TorusState(rc, zs, 0.0, 0.0, M, N, nfp)


def test_size_closed_forms():
    # a circular torus: V = 2 pi^2 R0 a^2, minor radius a
    R0, a = 1.3, 0.21
    s = ask.size(_torus({(0, 0): R0, (1, 0): a}, {(1, 0): a}, 1, 0, 1))
    assert abs(s["volume"] - 2 * math.pi ** 2 * R0 * a * a) <= 1e-15 * s["volume"]
    assert abs(s["minor_radius"] - a) <= 1e-15 * a
    # an ellipse turning with the field periods, centred at R0 in every plane:
    # R = R0 + a cos th + b cos(th - nfp phi), Z = a sin th - b sin(th - nfp phi):
    # every section has the area pi (a^2 - b^2) and its centroid at R0, so V = 2 pi R0 pi (a^2 - b^2)
    a, b, nfp = 0.2, 0.07, 5
    T = _torus({(0, 0): R0, (1, 0): a, (1, 1): b}, {(1, 0): a, (1, 1): -b}, 4, 3, nfp)
    s = ask.size(T)
    assert abs(s["minor_radius"] - math.sqrt(a * a - b * b)) <= 2e-15
    assert abs(s["volume"] - 2 * math.pi * R0 * math.pi * (a * a - b * b)) <= 1e-14 * s["volume"]


def test_proof_answers_every_question():
    """DIII-D's upper I-coils have no toroidal flux surface, by proof: with no flux, by size and for the largest
    the answer is the same proof the direct call gives."""
    ref = api.solve(D3D)
    assert ref.status == "no_surface" and ref.evidence.get("proof")
    for kw in (dict(), dict(largest=True), dict(minor_radius=0.1)):
        res = ask.solve(D3D, **kw)
        assert res.status == ref.status and res.surface is None
        assert res.evidence["proof"]["verdict"] == ref.evidence["proof"]["verdict"]
        assert res.evidence["search"]["words"][0].startswith("asked: ")


def test_flux_request_unchanged():
    """A request by flux goes to the shipped solver unchanged."""
    a = ask.solve(Q954, flux=0.03)
    b = api.solve(Q954, flux=0.03)
    assert a.status == b.status == "surface" and _same_surface(a, b)
    assert "search" not in a.evidence


@pytest.mark.parametrize("which,target", [("minor_radius", 0.12), ("volume", 0.3)])
def test_size_quasr954(which, target):
    """The surface of QUASR 954 with minor radius 0.12 m and with volume 0.3 m^3: SIMSOPT's own measure of it equals
    the request, and it is the shipped solver's answer to a direct request at its flux, bit for bit."""
    pytest.importorskip("simsopt.geo")
    t = time.time()
    res = ask.solve(Q954, **{which: target})
    wall = time.time() - t
    s = res.evidence["search"]
    print(f"\n{which}={target}: {res.status} at {s['flux']!r} Wb, reached {s['reached']!r}, "
          f"{len(s['requests'])} requests in {wall:.1f} s: {_requests(res)}")
    assert res.status == "surface"
    assert abs(s["reached"] - target) <= ask.RTOL * target
    assert s["repeatable"]
    direct = api.solve(Q954, flux=s["flux"])
    assert direct.status == "surface" and _same_surface(res, direct)
    sim, s_sim = _simsopt_size(res.surface)
    print(f"SIMSOPT: volume {sim['volume']!r}, minor radius {sim['minor_radius']!r}")
    assert abs(sim[which] - target) <= 1e-9 * target
    # the coefficients SIMSOPT measured are the surface's: its points at random angles are torux's
    rng = np.random.default_rng(1)
    th, ph = 2 * np.pi * rng.random(200), 2 * np.pi * rng.random(200)
    g = np.zeros((200, 3))
    s_sim.gamma_lin(g, ph / (2 * np.pi), th / (2 * np.pi))
    assert np.max(np.abs(g - res.points(th, ph))) <= 1e-13


# --------------------------------------------------------------------------- the search's own words, no solve
def _search(monkeypatch, question="largest", asked=None, budget=100.0):
    """A search with nothing asked yet, and the lines it says: the solver's budget estimate replaced by a constant."""
    import types
    said = []
    monkeypatch.setattr(ask, "say", said.append)
    monkeypatch.setattr(api, "estimate_budget", lambda src, flux: 60.0)
    return ask._Search(types.SimpleNamespace(id="x"), question, asked, budget, {}), said


def test_the_request_without_a_flux_says_how_far_it_got(monkeypatch):
    """The family from the axis: the line that ends the request says the flux verified and the flux followed to."""
    S, said = _search(monkeypatch)
    family = [dict(flux=0.02, converged=True, bn_off=1e-9), dict(flux=-0.0427, converged=True, bn_off=1e-9),
              dict(flux=0.05, converged=False, bn_off=1.0), dict(flux=0.06, converged=True, bn_off=float("nan"))]
    monkeypatch.setattr(ask, "_request", lambda src, flux, b, o: api.Result(
        "range", b, 1.0, record=dict(range=dict(verified_lower_bound=-0.0108), profile=family)))
    S.ask(None, "the family from the axis")
    assert said[-1].startswith("request 1, no flux (the family from the axis): range (verified to 0.0108 Wb, "
                               "followed to 0.0427 Wb) in ")
    assert S.log[0]["family_reached_Wb"] == 0.0427 and S.log[0]["verified_lower_bound_Wb"] == -0.0108
    # nothing verified in the family: only how far it was followed. A request at a flux says neither
    monkeypatch.setattr(ask, "_request", lambda src, flux, b, o: api.Result(
        "undetermined", b, 1.0, record=dict(profile=family)))
    S.ask(None, "the family from the axis")
    assert ": undetermined (followed to 0.0427 Wb) in " in said[-1]
    S.ask(0.03, "a direct request")
    assert said[-1].startswith("request 3, 0.03 Wb: undetermined in ")


def test_the_largest_says_where_it_could_not_decide_below_its_answer(monkeypatch):
    """A flux below the answer at which torux could not decide is said, and kept in the evidence: surfaces inside a
    verified surface are expected, so an open request there is something the reader should know."""
    S, _ = _search(monkeypatch)
    lo = 0.05
    entry = dict(flux=lo, budget_s=60.0, wall_s=3.0, status="surface")
    r = api.Result("surface", 60.0, 3.0, surface=object(), iota=0.3, record={})
    r._ask_entry = entry
    S.log.append(entry)
    S.best[lo], S.size_of[lo] = r, dict(volume=1.0, minor_radius=0.1)
    why = "neither a surface nor a certificate within the budget"
    S.failed = {0.07: dict(budget_s=120.0, why=why, default_budget=True)}
    res = ask._largest_answer(S, lo)
    s = res.evidence["search"]
    assert s["undecided_below"] == [] and not any(w.startswith("below it") for w in s["words"])
    assert "beyond it: at 0.07 Wb torux could not decide within 120 s (" + why + ")" in s["words"]
    below = "below it: at 0.03 Wb torux could not decide within 45 s"
    for failed, cert, words in (
            ({0.03: 45.0}, [], below),
            ({0.03: 45.0, 0.02: 60.0}, [], below + ", nor at 1 smaller flux"),
            ({0.03: 45.0, 0.02: 60.0, 0.01: 60.0}, [], below + ", nor at 2 smaller fluxes"),
            # a request an island band's certificate settles is not open
            ({0.03: 45.0, 0.02: 60.0}, [(0.02, dict(kind="island_band", no_surface_between_Wb=[0.019, 0.021]), None)],
             below)):
        S.failed = {f: dict(budget_s=b, why=why) for f, b in failed.items()}
        S.failed[0.07] = dict(budget_s=120.0, why=why, default_budget=True)
        S.certs = cert
        s = ask._largest_answer(S, lo).evidence["search"]
        assert words in s["words"], s["words"]
        open_below = sorted(f for f in failed if not cert or f != 0.02)
        assert [u["flux_Wb"] for u in s["undecided_below"]] == open_below
        assert ("below it", words[len("below it: "):]) in ask.search_rows(res)


def test_an_open_search_says_what_to_do_next(monkeypatch):
    """Undetermined after a search: the answer names no requested flux of its own (the search's requests are in the
    evidence) and its next step is the question's, a larger budget, or for a size a smaller one."""
    from torux.answer import Answer
    for question, asked, nxt in (("largest", None, "give a larger budget_s than 100"),
                                 ("minor_radius", 0.3, "ask for a smaller size, or give a larger budget_s than 100")):
        S, _ = _search(monkeypatch, question, asked)
        res = S.finish(api.Result("undetermined", 100.0, 99.0), ["found: no verified surface"])
        text = str(Answer._of(res, 0.4 if asked else None))
        rows = dict(line.split(None, 1) for line in text.splitlines() if not line.startswith(" "))
        assert rows["next"] == nxt and "requested flux" not in text and text.startswith("undetermined")
        assert ("asked", f"the surface of minor radius {asked:.9g} m" if asked else
                "the largest surface torux can verify") in ask.search_rows(res)


def test_a_size_is_never_answered_with_the_status_of_a_surface_it_does_not_return(monkeypatch):
    """The request at the flux a size would need can come back 'surface' with its refinement cut short by the
    budget: that torus is not verified, so the answer has no surface, and it must not keep the word either."""
    from torux.answer import Answer
    S, _ = _search(monkeypatch, "minor_radius", 0.3)
    S.size_of = {0.05: dict(minor_radius=0.1, volume=1.0)}
    cut = api.Result("surface", 60.0, 59.0, surface=object(), iota=0.3, record=dict(refine_stop="deadline"))
    assert not ask.verified(cut)
    assert ask._why(cut) == "a surface was found and the request's budget ended before its refinement did"
    S.failed = {0.2: dict(budget_s=60.0, why=ask._why(cut))}
    res = ask._size_not_held(S, "minor_radius", 0.3, 0.2, 0.2, None, cut)
    assert res.status == "undetermined" and res.surface is None and cut.status == "surface"
    s = res.evidence["search"]
    assert s["flux"] == 0.2 and s["reached"] is None
    assert ("largest verified here: minor radius 0.1 m, at 0.05 Wb, by this search's requests (largest=True looks "
            "for the largest the coils hold)") in s["words"]
    assert "at that flux: torux could not decide within 60 s (" + ask._why(cut) + ")" in s["words"]
    a = Answer._of(res, None)
    assert str(a).startswith("undetermined") and a.requested_flux is None
    with pytest.raises(ValueError, match="undetermined"):
        a.iota


def test_after_a_search_no_flux_was_the_one_asked(monkeypatch):
    """Asked by size: the answer has no requested flux, whatever fluxes the search asked on the way (the flux of an
    earlier request whose certificate rules the size out is in the search's record, and is not the flux asked)."""
    S, _ = _search(monkeypatch, "minor_radius", 0.3)
    S.size_of = {0.05: dict(minor_radius=0.1, volume=1.0)}
    cert = dict(kind="edge_barrier", ok=True, F_up=0.061)
    res = ask._size_not_held(S, "minor_radius", 0.3, 0.2855, 0.0695, cert, None)
    assert res.status == "no_surface" and res.evidence["search"]["flux"] == 0.2855
    assert res.evidence["search"]["flux_asked"] == 0.0695
    monkeypatch.setattr(ask, "solve", lambda *a, **k: res)
    a = torux.solve(Q954, minor_radius=0.3)
    assert a.status == "no_surface" and a.requested_flux is None and "requested flux" not in str(a)
    from torux import batch
    row = batch._row(a, None, None)
    assert row["flux_Wb"] == 0.2855 and row["note"].endswith("flux_Wb is the flux that size would need")


def test_an_answer_says_when_the_budget_left_no_room_for_what_was_asked_with_it(monkeypatch):
    """Gradients or traced lines are asked of the answer in one more request: when the budget has no room for it,
    the answer comes without them and says so."""
    S, _ = _search(monkeypatch)
    S.opts = dict(gradients=True, trace_lines=0)
    r = api.Result("surface", 60.0, 3.0, surface=object(), iota=0.3, record={})
    S.best[0.05] = r
    monkeypatch.setattr(S, "ask", lambda *a, **k: None)
    assert S.final(0.05) is r
    words = S.finish(r, ["reached: x"]).evidence["search"]["words"]
    assert ("note: the budget ended before the surface could be solved again with the gradients asked for: this "
            "answer is without them") in words
    S2, _ = _search(monkeypatch)
    S2.best[0.05] = r
    assert S2.final(0.05) is r and S2.missed is None


# --------------------------------------------------------------------------- slow: minutes each
@pytest.mark.slow
@pytest.mark.parametrize("which,target", [("minor_radius", 0.11), ("volume", 0.3)])
def test_size_hsx(which, target):
    """HSX has no design surface to start from: the same bar from its coils alone."""
    pytest.importorskip("simsopt.geo")
    t = time.time()
    res = ask.solve(HSX, **{which: target})
    wall = time.time() - t
    s = res.evidence["search"]
    print(f"\nHSX {which}={target}: {res.status} at {s['flux']!r} Wb, reached {s['reached']!r}, "
          f"{len(s['requests'])} requests in {wall:.1f} s: {_requests(res)}")
    assert res.status == "surface"
    sim, _ = _simsopt_size(res.surface)
    print(f"SIMSOPT: volume {sim['volume']!r}, minor radius {sim['minor_radius']!r}")
    assert abs(sim[which] - target) <= 1e-9 * target
    if s["repeatable"]:
        direct = api.solve(HSX, flux=s["flux"])
        assert _same_surface(res, direct)


@pytest.mark.slow
def test_size_too_large_quasr1100073():
    """QUASR 1100073: a certificate says no surface encloses more than 0.0611 Wb.  A minor radius beyond the last
    surface comes back as no surface or undetermined, with the words, inside its budget."""
    t = time.time()
    res = ask.solve(Q1100073, minor_radius=0.15)
    wall = time.time() - t
    s = res.evidence["search"]
    print(f"\n{res.status} in {wall:.0f} s of {s['budget_s']:.0f}:\n  " + "\n  ".join(s["words"])
          + f"\n{_requests(res)}")
    assert res.status in ("no_surface", "undetermined", "timeout")
    assert res.surface is None and s["reached"] is None
    assert wall <= s["budget_s"] + 120
    assert any("would need" in w for w in s["words"])


@pytest.mark.slow
def test_largest_hsx():
    """HSX: at least 0.0359 Wb verified, more than the no-flux request returns (0.0108 Wb), inside the stated
    budget, with what limits it, and the same surface as a direct request at its flux."""
    t = time.time()
    res = ask.solve(HSX, largest=True)
    wall = time.time() - t
    s = res.evidence["search"]
    print(f"\n{res.status} {s['flux']!r} Wb in {wall:.0f} s of {s['budget_s']:.0f}:\n  " + "\n  ".join(s["words"])
          + "\n" + "\n".join(map(str, s["requests"])))
    assert res.status == "surface" and s["flux"] >= 0.0359
    assert wall <= s["budget_s"] + 120
    assert s["beyond"] is not None
    rng = [e for e in s["requests"] if e["flux"] is None][0]
    assert s["flux"] > (rng["verified_lower_bound_Wb"] or 0.0)
    # a direct request repeats the surface whenever the solver's path to it does not depend on the clock
    if s["repeatable"]:
        direct = api.solve(HSX, flux=s["flux"])
        assert _same_surface(res, direct)
    else:
        assert any(w.startswith("note: a request for that flux") for w in s["words"])


@pytest.mark.slow
def test_largest_quasr1100073():
    """QUASR 1100073: a verified surface below the certified bound (0.0611 Wb), the bound reported if a certificate
    was reached within the budget."""
    res = ask.solve(Q1100073, largest=True)
    s = res.evidence["search"]
    print("\n  " + "\n  ".join(s["words"]) + "\n" + "\n".join(map(str, s["requests"])))
    assert res.status == "surface" and s["flux"] < 0.0611
    if s["beyond"]["kind"] == "certified bound":
        assert s["beyond"]["no_surface_above_Wb"] >= s["flux"]
    # a direct request repeats the surface whenever the solver's path to it does not depend on the clock
    if s["repeatable"]:
        direct = api.solve(Q1100073, flux=s["flux"])
        assert _same_surface(res, direct)
    else:
        assert any(w.startswith("note: a request for that flux") for w in s["words"])


@pytest.mark.slow
def test_largest_quasr954():
    """QUASR 954 carries its own request and design surfaces: the largest is at least its design flux."""
    src = torux.load(Q954)
    res = ask.solve(Q954, largest=True)
    s = res.evidence["search"]
    print("\n  " + "\n  ".join(s["words"]) + "\n" + "\n".join(map(str, s["requests"])))
    assert res.status == "surface" and s["flux"] >= src.request["flux_Wb"]
    # a direct request repeats the surface whenever the solver's path to it does not depend on the clock
    if s["repeatable"]:
        direct = api.solve(Q954, flux=s["flux"])
        assert _same_surface(res, direct)
    else:
        assert any(w.startswith("note: a request for that flux") for w in s["words"])
