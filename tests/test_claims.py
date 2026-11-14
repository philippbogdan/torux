"""Every number the README states, held to the tables in results/."""
import importlib.util
import math
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _numbers():
    spec = importlib.util.spec_from_file_location("claims", os.path.join(ROOT, "results", "claims.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.quasr(), m.compare()


def sci(x):
    """1.2 × 10⁻¹⁵, as the README writes it."""
    e = math.floor(math.log10(x))
    if round(x / 10 ** e, 1) >= 10:
        e += 1
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{x / 10 ** e:.1f} × 10{sup}"


def test_readme_numbers_are_the_tables():
    q, c = _numbers()
    t, stored = c["torux"], c["quasr-stored"]
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    text = re.sub(r"\s+", " ", text)
    floor = 3.89e-16                                    # results/evaluator/test_output.txt: an exact torus in double precision
    claims = {
        f"{q['surfaces']:,} of the {q['designs']:,} coil sets": "QUASR surfaces",
        f"For {q['island_bands']:,} more it returned a certificate": "QUASR's island-band certificates, the headline",
        f"for another {q['edge_barriers']:,} a weaker one": "QUASR's edge-barrier certificates, beside the headline",
        f"the other {q['undetermined']:,} it calls undetermined": "QUASR undetermined",
        f"median max |B·n|/|B| of {sci(t['bn_median'])}": "torux's median in the comparison",
        f"worst of its {t['surfaces']} is at {sci(t['bn_max'])}": "torux's worst in the comparison",
        f"certify: {t['island_bands']} of these {t['designs']}, and {t['edge_barriers']} more by its weaker kind":
            "certificates in the comparison, by kind",
        f"The other {t['undetermined']} stayed undetermined.": "undetermined in the comparison",
        f"within {t['bn_median'] / floor:.1f} times what an exact surface written in double precision reads":
            "torux's median against the evaluator's reading of an exact torus",
    }
    missing = [f"{what}: {frag!r}" for frag, what in claims.items() if frag not in text]
    assert not missing, missing
    # the rounded statements
    assert 0.5e-3 < stored["bn_median"] < 2e-3                                             # "about a thousandth"
    assert round(math.log10(stored["bn_median"] / t["bn_median"])) == 12                   # "twelve orders of magnitude"
    assert 100 < stored["terms_median"] and stored["terms_max"] < 1000 < t["terms_min"]    # "a few hundred ... thousands"
    assert q["bn_solver_max"] <= 1e-14 and q["bn_check_max"] < 2e-14 and q["flux_err_max"] < 1e-12
    assert 0.74 < q["undetermined_under_1_min"] < 0.80                                     # "three in four ... a minute"
    assert q["island_bands"] + q["edge_barriers"] == q["certified_none"]                   # every certificate is one or the other
    assert q["surfaces"] + q["island_bands"] + q["edge_barriers"] + q["undetermined"] == q["designs"]   # the opening's four counts
    assert t["island_bands"] + t["edge_barriers"] == t["certified_none"]
    assert t["surfaces"] + t["island_bands"] + t["edge_barriers"] + t["undetermined"] == t["designs"]


def test_document_numbers_are_the_tables():
    q, c = _numbers()
    t = c["torux"]
    text = re.sub(r"\s+", " ", open(os.path.join(ROOT, "docs", "verification.md"), encoding="utf-8").read())
    claims = [
        f"Across the {q['surfaces']:,} QUASR surfaces: the solver's check reads {sci(q['bn_solver_median'])} in the median "
        f"and at most {sci(q['bn_solver_max'])}",
        f"the evaluator's estimated maximum {sci(q['bn_check_median'])} in the median and at most {sci(q['bn_check_max'])}",
        f"the flux is off by {sci(q['flux_err_median'])} in the median",
        f"On torux's {t['surfaces']} surfaces there it reads {sci(t['bn_median'])} in the median and {sci(t['bn_max'])} at worst",
        f"**Island bands** ({q['island_bands']:,} of QUASR's certificates)",
        f"**Edge barriers** ({q['edge_barriers']:,})",
    ]
    missing = [frag for frag in claims if frag not in text]
    assert not missing, missing
    text = re.sub(r"\s+", " ", open(os.path.join(ROOT, "results", "README.md"), encoding="utf-8").read())
    assert f"each of the {q['designs']:,} coil sets" in text
    assert f"each of torux's {t['surfaces']} surfaces in the comparison" in text
    spec = importlib.util.spec_from_file_location("claims", os.path.join(ROOT, "results", "claims.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    b = m.both()
    assert not any(("table surface" in k and "hour no_surface" in k) or ("table no_surface" in k and "hour surface" in k)
                   for k in b)                                                             # the two tables never contradict
    # the usage page's one number from the comparison: what VMEC++ leaks, beside what a cut of torux's file costs
    usage = re.sub(r"\s+", " ", open(os.path.join(ROOT, "docs", "usage.md"), encoding="utf-8").read())
    v = c["vmecpp"]["bn_median"]
    cut = float(re.search(r"input\.q954_cut: .*? field across it +([0-9.e+-]+) of \|B\| at worst", usage).group(1))
    assert f"leak a median {sci(v)} of |B| (the README's table), sixty times more than this cut" in usage
    assert 55 < v / cut < 65
    agree = b.get("table surface, one hour surface", 0)
    table_surfaces = sum(v for k, v in b.items() if k.startswith("table surface"))
    assert f"the table has a surface for {table_surfaces} of them, and {agree} of those are surfaces again" in text


def test_picture_numbers_are_the_records():
    """What the README and the atlas say about the pictures, held to visuals/sources.json and visuals/one-line.json."""
    import json
    V = os.path.join(ROOT, "visuals")
    with open(os.path.join(V, "sources.json"), encoding="utf-8") as fh:
        src = json.load(fh)["sources"]
    surf = [s for s in src if s["answer"] == "surface"]
    none = [s for s in src if s["answer"] == "no_surface"]
    readme = re.sub(r"\s+", " ", open(os.path.join(ROOT, "README.md"), encoding="utf-8").read())
    assert len(src) == 12 and len(surf) == 9 and len(none) == 3
    assert "welve kinds of magnet" in readme and "Nine carry the surface" in readme and "The other three" in readme
    atlas = re.sub(r"\s+", " ", open(os.path.join(V, "README.md"), encoding="utf-8").read())
    for s in surf:
        # every drawn surface is at the bar of results/README.md, and the atlas quotes its record
        assert s["bn_solver"] <= 1e-14 and s["bn_check"] < 2e-14 and s["flux_err"] < 1e-12, s["name"]
        assert (f"{sci(s['bn_solver'])} of its strength by the solver's check and {sci(s['bn_check'])} by the "
                f"independent evaluator's") in atlas, s["name"]
    for s in none:
        assert s["verdict"] in ("NO_TOROIDAL_FLUX_SURFACE", "DIPOLE_CLOSED_FIELD_LINES"), s["name"]
    with open(os.path.join(V, "one-line.json"), encoding="utf-8") as fh:
        one = json.load(fh)["values"]
    e = math.floor(math.log10(one["dist_max_m"]))
    bound = math.ceil(one["dist_max_m"] / 10 ** (e - 1)) / 10            # "stays within": rounded up
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    assert f"followed for {one['laps']} laps ({one['length_m'] / 1000:.1f} km)" in atlas
    assert f"stays within {bound:.1f} × 10{sup} m of the surface" in atlas and bound < 10
    assert one["solver_bn_off"] <= 1e-14 and one["ref_bn_sup"] < 2e-14


def test_no_design_has_both_answers_and_every_certificate_holds_its_request():
    import pandas as pd
    t = pd.read_parquet(os.path.join(ROOT, "results", "quasr.parquet"))
    assert t.id.is_unique and len(t) == 371701
    band = t[t.certificate == "island_band"]
    edge = t[t.certificate == "edge_barrier"]
    assert ((band.excluded_lo_Wb < band.flux_Wb.abs()) & (band.flux_Wb.abs() < band.excluded_hi_Wb)).all()
    assert (edge.bound_Wb < edge.flux_Wb.abs()).all()
    s = t[t.outcome == "surface"]
    assert s.certificate.eq("").all() and s.bn_solver.notna().all() and s.bn_check.notna().all()


LABELS = {"simsopt-boozer-exact": "SIMSOPT BoozerExact", "simsopt-boozer-ls": "SIMSOPT BoozerLS",
          "simsopt-qfm": "SIMSOPT QFM", "desc": "DESC", "vmecpp": "VMEC++"}


def secs(x):
    """4.2 s or 1,770 s, as the README writes a time."""
    return f"{x:,.0f} s" if x >= 100 else f"{x:.2g} s"


def test_every_time_is_the_time_to_the_returned_surface():
    """'Seconds' and the table's last column show when a tool had the surface it returned (answer_s), never the length
    of a run that was set to fill its hour."""
    import csv
    _, c = _numbers()
    with open(os.path.join(ROOT, "results", "compare300.csv")) as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["solver"], {})[r["design"]] = r
    for name, mine in by.items():
        for r in mine.values():
            if name == "quasr-stored" or r["status"] != "surface":
                assert r["answer_s"] == "", (name, r["design"])
                continue
            assert 0 < float(r["answer_s"]) <= float(r["wall_s"]) + 1e-9, (name, r["design"])
            if name == "torux":
                assert r["answer_s"] == r["wall_s"], (name, r["design"])
    text = re.sub(r"\s+", " ", open(os.path.join(ROOT, "README.md"), encoding="utf-8").read())
    t = c["torux"]
    assert f"torux: {secs(t['answer_s_median'])} in the median, and then it stops." in text
    filled = [LABELS[k] for k in sorted(LABELS, key=lambda k: c[k]["answer_s_median"])
              if c[k]["ran_to_the_hour"] > 0.5 * c[k]["surfaces"]]
    assert filled and (" and ".join(filled) + " nearly always until the hour ends") in text
    solved = {d for d, r in by["torux"].items() if r["status"] == "surface"}
    for k, label in LABELS.items():
        both = [d for d in solved if by[k][d]["status"] == "surface"]
        faster = sum(float(by[k][d]["answer_s"]) < float(by["torux"][d]["answer_s"]) for d in both)
        better = sum(float(by[k][d]["bn_max"]) < float(by["torux"][d]["bn_max"]) for d in both)
        line = (f"{label}: {secs(c[k]['answer_s_median'])}. On the {len(both)} coil sets both solved it was faster on "
                f"{faster} and more accurate on {better}.")
        assert line in text, line
        assert f"| {label} | {c[k]['surfaces']} of {c[k]['designs']} |" in text
        assert f"| {secs(c[k]['answer_s_median'])[:-2]} |" in text


def test_the_readme_does_not_put_the_memory_limit_on_the_tool():
    """What stopped BoozerExact's runs at 12 GiB was the comparison's own program (results/compare300/README.md says
    so): the README makes no sentence of it."""
    text = re.sub(r"\s+", " ", open(os.path.join(ROOT, "README.md"), encoding="utf-8").read())
    assert "which is what stops it" not in text and "given 16" not in text
    page = re.sub(r"\s+", " ", open(os.path.join(ROOT, "results", "compare300", "README.md"), encoding="utf-8").read())
    assert "part of that memory is this program's and not SIMSOPT's" in page


def test_two_requests_near_the_edge():
    """What the usage and results pages say of outcomes that depend on the clock, held to results/edge_requests.csv."""
    import csv
    import re
    with open(os.path.join(ROOT, "results", "edge_requests.csv")) as fh:
        a, b, c, d = list(csv.DictReader(fh))
    assert a["source"] == b["source"] and a["flux_Wb"] == b["flux_Wb"]
    assert (a["outcome"], b["outcome"]) == ("surface", "undetermined")
    assert c["source"] == d["source"] and float(d["flux_Wb"]) > float(c["flux_Wb"])
    assert (c["outcome"], d["outcome"]) == ("no_surface", "undetermined")
    assert round(float(a["flux_Wb"]) / 0.06910690581397087) == 6          # six times QUASR 954's design flux
    for page in ("docs/usage.md", "results/README.md"):
        with open(os.path.join(ROOT, page), encoding="utf-8") as fh:
            t = re.sub(r"\s+", " ", fh.read())
        for frag in (f"verified it in {float(a['seconds']):.0f} s", f"after {float(b['seconds']):.0f} s in the next process",
                     f"gave up after {float(d['seconds']):,.0f} s", f"gets its certificate in {float(c['seconds']):.0f} s"):
            assert frag in t, (page, frag)
