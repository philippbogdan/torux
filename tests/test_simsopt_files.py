"""SIMSOPT JSON files as their owners write them: what torux reads, and what it refuses rather than guess.

Each file here is written by the test from the bundled QUASR 954 download or from curves given by hand, so that
what it should read is known without torux: the coils of a BoozerSurface file (the coils its surface was computed
for), a Current object stating two values, several coil sets in one file, and the helical curve classes. The truth
for the helical curves is their definition in SIMSOPT, written out again below. Where SIMSOPT imports, its own
CurveXYZFourierSymmetries is compared too.
"""
import argparse
import copy
import json

import numpy as np
import pytest

from conftest import example

import torux
from torux.check import add_command, check, run_command


def _ref(name):
    return {"$type": "ref", "value": name}


def _quasr_954():
    with open(example("quasr_0000954.json")) as fh:
        return json.load(fh)


def _boozer_file(path, change=None):
    """QUASR 954 written as SIMSOPT writes a BoozerSurface: its coils as BiotSavart1, its outermost stored surface
    as the surface of BoozerSurface1, and the graph the BoozerSurface. ``change(d)`` edits the file first."""
    d = _quasr_954()
    surfaces, coils = d["graph"]
    outer = surfaces[-1]["value"]
    o = d["simsopt_objs"]
    o["BiotSavart1"] = {"@module": "simsopt.field.biotsavart", "@class": "BiotSavart", "@name": "BiotSavart1",
                        "coils": coils}
    o["Volume1"] = {"@module": "simsopt.geo.surfaceobjectives", "@class": "Volume", "@name": "Volume1",
                    "surface": _ref(outer)}
    o["BoozerSurface1"] = {"@module": "simsopt.geo.boozersurface", "@class": "BoozerSurface",
                           "@name": "BoozerSurface1", "biotsavart": _ref("BiotSavart1"), "surface": _ref(outer),
                           "label": _ref("Volume1"), "targetlabel": 0.1, "constraint_weight": None}
    d["graph"] = _ref("BoozerSurface1")
    if change is not None:
        change(d)
    with open(path, "w") as fh:
        json.dump(d, fh)
    return str(path)


def _coil_current(d, coil):
    """The current a coil of the file carries, as SIMSOPT resolves it (A)."""
    o = d["simsopt_objs"]

    def value(name):
        c = o[name]
        if c["@class"] == "ScaledCurrent":
            return c["scale"] * value(c["current_to_scale"]["value"])
        x = o[c["dofs"]["value"]]["x"]
        return (x["data"] if isinstance(x, dict) else x)[0]
    return value(o[coil]["current"]["value"])


def _two_valued(saved_over_dofs):
    """An edit: the first coil carries a Current of its own, whose DOFs hold the coil's current and whose
    'current' key holds that times saved_over_dofs, as in a file whose current was set past its DOFs."""
    def change(d):
        o = d["simsopt_objs"]
        first = d["simsopt_objs"]["BiotSavart1"]["coils"][0]["value"]
        i0 = _coil_current(d, first)
        o["dofs_of_Current9"] = {"@module": "simsopt._core.optimizable", "@class": "DOFs", "@name": "dofs_of_Current9",
                                 "x": [i0], "names": ["x0"]}
        o["Current9"] = {"@module": "simsopt.field.coil", "@class": "Current", "@name": "Current9",
                         "current": i0 * saved_over_dofs, "dofs": _ref("dofs_of_Current9")}
        o[first]["current"] = _ref("Current9")
    return change


# --------------------------------------------------------------------------- the coils a surface was computed for
def test_a_boozer_surface_with_its_own_coils_reads_as_quasr_does(tmp_path, quasr_954):
    f = _boozer_file(tmp_path / "boozer.json")
    a = check(f, f)
    b = check(quasr_954, example("quasr_0000954.json"))       # the same coils and surface, from QUASR's own file
    assert a.leak == pytest.approx(b.leak, rel=1e-9) and a.flux == pytest.approx(b.flux, rel=1e-12)
    assert a.leak < 1e-5 and not a.notes
    c = check(quasr_954, f)                                  # the same coils from another file: the same field
    assert c.leak == pytest.approx(b.leak, rel=1e-9)


def test_a_boozer_surface_against_other_coils_is_refused(tmp_path, quasr_954, quasr_954_mirrored):
    """A surface checked against coils it was not computed for: torux 0.1.3 measured the mismatch of the files and
    printed it as the surface's leak."""
    f = _boozer_file(tmp_path / "boozer.json")
    with pytest.raises(ValueError, match="was not computed for these coils"):
        check(quasr_954_mirrored, f)                         # one coil of a pair at 3% more current
    scaled = copy.deepcopy(quasr_954)
    scaled.parts[0].current = scaled.parts[0].current * (1 + 1e-6)
    with pytest.raises(ValueError, match="BoozerSurface1"):
        check(scaled, f)
    ap = argparse.ArgumentParser(prog="torux")
    add_command(ap.add_subparsers(dest="cmd"))
    g = _boozer_file(tmp_path / "boozer_mirror.json", change=_two_valued(1.0))   # the same coils, written again
    assert run_command(ap.parse_args(["check", g, f])) == 0


# --------------------------------------------------------------------------- a Current with two values
def test_a_current_with_two_values_is_refused_until_chosen(tmp_path, quasr_954):
    """A file whose Current states 'current' at 3.95 times its DOFs. torux 0.1.3 took the DOFs without a word, and
    the field across the file's own surface came out of order one. The file does not say which."""
    f = _boozer_file(tmp_path / "two.json", change=_two_valued(3.95))
    with pytest.raises(ValueError, match="Current9 states two values") as e:
        torux.load(f)
    assert "currents='saved'" in str(e.value) and "--currents" in str(e.value)
    with pytest.raises(ValueError, match="two values"):
        check(f, f)
    dofs = torux.load(f, currents="dofs")
    saved = torux.load(f, currents="saved")
    assert np.array_equal(dofs.parts[0].current, quasr_954.parts[0].current)
    assert saved.parts[0].current[0] == pytest.approx(3.95 * quasr_954.parts[0].current[0], rel=1e-15)
    assert np.array_equal(saved.parts[0].current[1:], quasr_954.parts[0].current[1:])
    assert saved.meta["currents"]["taken"] == "saved" and "Current9" in saved.meta["currents"]["differing"]
    a, b = check(dofs, f), check(saved, f)                   # each against its own file read the same way
    assert a.leak < 1e-5 and b.leak > 1e-2
    assert any("'saved' ones were taken" in n for n in b.notes)
    with pytest.raises(ValueError, match="currents="):
        torux.load(f, currents="key")


def test_a_current_whose_two_values_agree_to_rounding_is_read(tmp_path, quasr_954):
    f = _boozer_file(tmp_path / "ulp.json", change=_two_valued(1 + 2 ** -52))
    src = torux.load(f)
    assert np.array_equal(src.parts[0].current, quasr_954.parts[0].current) and "currents" not in src.meta


# --------------------------------------------------------------------------- several coil sets
def _two_sets(d):
    o = d["simsopt_objs"]
    o["BiotSavart2"] = dict(o["BiotSavart1"], **{"@name": "BiotSavart2"})


def test_several_coil_sets_name_the_choice_and_the_command_line_takes_it(tmp_path, capsys):
    f = _boozer_file(tmp_path / "sets.json", change=_two_sets)
    with pytest.raises(ValueError, match="--coils") as e:
        torux.load(f)
    assert "BoozerSurface1 was computed with BiotSavart1" in str(e.value)
    assert torux.load(f, which="BiotSavart2").meta["field_object"] == "BiotSavart2"
    assert torux.load(f, which=0).meta["field_object"] == "BiotSavart1"
    with pytest.raises(ValueError, match="matches 0"):
        torux.load(f, which="BiotSavart3")
    ap = argparse.ArgumentParser(prog="torux")
    add_command(ap.add_subparsers(dest="cmd"))
    assert run_command(ap.parse_args(["check", f, f])) == 1
    assert "--coils" in capsys.readouterr().err
    assert run_command(ap.parse_args(["check", f, f, "--coils", "BiotSavart2"])) == 0
    assert run_command(ap.parse_args(["check", f, f, "--coils", "1", "--which", "0"])) == 0
    assert capsys.readouterr().out.startswith("surface")
    g = _boozer_file(tmp_path / "two.json", change=_two_valued(3.95))
    assert run_command(ap.parse_args(["check", g, g, "--currents", "dofs"])) == 0
    from torux.cli import main
    assert main(["solve", f, "--coils", "BiotSavart3"]) == 1    # refused before any solve
    assert "matches 0" in capsys.readouterr().err


# --------------------------------------------------------------------------- helical curves
def _helical_by_hand(x, order, nfp, stellsym, ntor, t):
    """CurveXYZFourierSymmetries as SIMSOPT's docstring defines it, term by term."""
    xh, yh, z = (np.zeros_like(t) for _ in range(3))
    w = 2 * np.pi * t
    if stellsym:
        xc, ys, zs = x[:order + 1], x[order + 1:2 * order + 1], x[2 * order + 1:]
        for m in range(order + 1):
            xh += xc[m] * np.cos(nfp * m * w)
        for m in range(1, order + 1):
            yh += ys[m - 1] * np.sin(nfp * m * w)
            z += zs[m - 1] * np.sin(nfp * m * w)
    else:
        b = 2 * order + 1
        for j, v in enumerate((xh, yh, z)):
            c, s = x[j * b:j * b + order + 1], x[j * b + order + 1:(j + 1) * b]
            for m in range(order + 1):
                v += c[m] * np.cos(nfp * m * w)
            for m in range(1, order + 1):
                v += s[m - 1] * np.sin(nfp * m * w)
    a = ntor * w
    return np.stack([xh * np.cos(a) - yh * np.sin(a), xh * np.sin(a) + yh * np.cos(a), z], axis=1)


def _names(order, stellsym):
    """The DOF names SIMSOPT gives these curves (CurveXYZFourierSymmetries._make_names)."""
    if stellsym:
        return ([f"xc({i})" for i in range(order + 1)] + [f"ys({i})" for i in range(1, order + 1)]
                + [f"zs({i})" for i in range(1, order + 1)])
    return [f"{a}{cs}({i})" for a in "xyz" for cs, lo in (("c", 0), ("s", 1)) for i in range(lo, order + 1)]


def _helical_file(path, cls, x, order, nfp, stellsym, ntor=None, names=None):
    curve = {"@module": "simsopt.geo", "@class": cls, "@name": f"{cls}1", "quadpoints": list(np.arange(64) / 64),
             "order": order, "nfp": nfp, "stellsym": stellsym, "dofs": _ref("d1")}
    if ntor is not None:
        curve["ntor"] = ntor
    o = {"d1": {"@class": "DOFs", "@name": "d1", "x": list(x), "names": names or _names(order, stellsym)},
         f"{cls}1": curve,
         "d2": {"@class": "DOFs", "@name": "d2", "x": [1.0], "names": ["x0"]},
         "Current1": {"@class": "Current", "@name": "Current1", "current": 1.0, "dofs": _ref("d2")},
         "ScaledCurrent1": {"@class": "ScaledCurrent", "@name": "ScaledCurrent1", "scale": 1e5,
                            "current_to_scale": _ref("Current1")},
         "Coil1": {"@class": "Coil", "@name": "Coil1", "curve": _ref(f"{cls}1"), "current": _ref("ScaledCurrent1")},
         "BiotSavart1": {"@class": "BiotSavart", "@name": "BiotSavart1", "coils": [_ref("Coil1")]}}
    with open(path, "w") as fh:
        json.dump({"@module": "simsopt._core.json", "@class": "SIMSON", "@version": "1.4.1",
                   "graph": _ref("BiotSavart1"), "simsopt_objs": o}, fh)
    return str(path)


HELICAL = [("CurveXYZHelical", 4, True, None), ("CurveXYZFourierSymmetries", 4, True, 1),
           ("CurveXYZFourierSymmetries", 3, False, 2), ("CurveXYZFourierSymmetries", 5, False, -2)]


@pytest.mark.parametrize("cls,nfp,stellsym,ntor", HELICAL, ids=[f"{c}-{n}-{s}-{t}" for c, n, s, t in HELICAL])
def test_helical_curves_read_exactly(tmp_path, cls, nfp, stellsym, ntor):
    """The helical coil sets of Jorge's simple-coil paper (CurveXYZHelical, SIMSOPT 1.4.1.dev38) did not load in
    torux 0.1.3: 'curve class CurveXYZHelical'."""
    order = 5
    rng = np.random.default_rng(nfp)
    x = 0.02 * rng.normal(size=3 * order + 1 if stellsym else 3 * (2 * order + 1))
    x[0] = 1.0
    f = _helical_file(tmp_path / "helical.json", cls, x, order, nfp, stellsym, ntor)
    src = torux.load(f)
    t = rng.random(400)
    got, _ = src.parts[0].points(t)
    want = _helical_by_hand(x, order, nfp, stellsym, 1 if ntor is None else ntor, t)
    assert np.max(np.abs(got[0] - want)) < 1e-14
    assert src.parts[0].current[0] == 1e5
    if stellsym:
        assert src.nfp == nfp and src.stellsym


def test_helical_curves_against_simsopt(tmp_path):
    pytest.importorskip("simsopt")
    from simsopt.geo import CurveXYZFourierSymmetries
    rng = np.random.default_rng(7)
    t = rng.random(300)
    for nfp, stellsym, ntor in ((4, True, 1), (3, False, 2)):
        c = CurveXYZFourierSymmetries(t, 6, nfp, stellsym, ntor=ntor)
        x = 0.02 * rng.normal(size=c.x.size)
        x[0] = 1.0
        c.x = x
        got, _ = torux.load(_helical_file(tmp_path / "h.json", "CurveXYZFourierSymmetries", x, 6, nfp, stellsym,
                                          ntor)).parts[0].points(t)
        assert np.max(np.abs(got[0] - c.gamma())) < 1e-13


def test_a_helical_curve_in_another_layout_is_refused(tmp_path):
    names = _names(3, True)
    names[1], names[4] = names[4], names[1]
    f = _helical_file(tmp_path / "h.json", "CurveXYZHelical", np.r_[1.0, np.zeros(9)], 3, 2, True, names=names)
    with pytest.raises(NotImplementedError, match="not the layout"):
        torux.load(f)
