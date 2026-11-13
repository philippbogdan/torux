"""SIMSOPT JSON files as their owners write them: the helical curve classes.

Each file here is written by the test from curves given by hand, so that what it should read is known without torux.
The truth is the curves' definition in SIMSOPT, written out again below. Where SIMSOPT imports, its own
CurveXYZFourierSymmetries is compared too.
"""
import json

import numpy as np
import pytest

import torux


def _ref(name):
    return {"$type": "ref", "value": name}


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
