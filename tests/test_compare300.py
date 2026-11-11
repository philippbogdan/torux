"""What results/compare300/README.md says of the comparison, held to results/compare300.csv and the files beside it."""
import csv
import hashlib
import json
import os
import re
import statistics

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C300 = os.path.join(ROOT, "results", "compare300")


def rows():
    with open(os.path.join(ROOT, "results", "compare300.csv")) as fh:
        out = {}
        for r in csv.DictReader(fh):
            out.setdefault(r["solver"], []).append(r)
        return out


def text():
    with open(os.path.join(C300, "README.md"), encoding="utf-8") as fh:
        return re.sub(r"\s+", " ", fh.read())


def has(fragments):
    t = text()
    missing = [f for f in fragments if f not in t]
    assert not missing, missing


def level(r):
    return int(re.search(r"mpol=ntor=(\d+)", r["solver_status"]).group(1))


def test_the_sample_is_the_stated_draw():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    ids = sorted(int(i) for i in pd.read_parquet(os.path.join(ROOT, "results", "quasr.parquet"), columns=["id"]).id)
    assert len(ids) == 371701
    pick = np.random.default_rng(20261107).choice(len(ids), size=300, replace=False)
    drawn = sorted(f"{ids[i]:07d}" for i in pick)
    by = rows()
    for name, mine in by.items():
        assert sorted(r["design"] for r in mine) == drawn, name
    has(["numpy.random.default_rng(20261107).choice(371701, 300, replace=False)"])


def test_every_tool_was_given_the_file_that_is_here():
    by = rows()
    files = sorted(f for f in os.listdir(os.path.join(C300, "inputs")) if f.endswith(".json"))
    assert files == sorted(f"quasr_{r['design']}.json" for r in by["torux"])
    for r in by["torux"][::25]:
        with open(os.path.join(C300, "inputs", f"quasr_{r['design']}.json"), "rb") as fh:
            rec = json.loads(fh.read())
        assert rec["id"] == r["design"] and rec["coils"] and rec["start"]["surface"]["kind"] == "SurfaceXYZTensorFourier"
        for name, mine in by.items():
            row = next(x for x in mine if x["design"] == r["design"])
            assert float(row["requested_flux_Wb"]) == rec["requested_toroidal_flux_Wb"], (name, r["design"])
    assert len([f for f in os.listdir(os.path.join(C300, "inputs", "pilot")) if f.endswith(".json")]) == 6
    with open(os.path.join(C300, "inputs.sha256")) as fh:
        listed = dict(line.split()[::-1] for line in fh if line.strip())
    for name in ("quasr_0003051.json", "quasr_2663906.json", "pilot/quasr_0925611.json"):
        with open(os.path.join(C300, "inputs", name), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == listed[name], name


def test_budget_columns():
    by = rows()
    for name, mine in by.items():
        if name == "quasr-stored":
            continue
        assert {r["cpu_model"] for r in mine} == {"AMD EPYC 7742 64-Core Processor"}, name
        assert {r["threads"] for r in mine} == {"4"}, name
        assert max(float(r["wall_s"]) for r in mine) < 3700, name
    has(["3,600 seconds, 12 GiB of memory, four compute threads on eight cores of an AMD EPYC 7742"])


def test_boozer_exact():
    x = rows()["simsopt-boozer-exact"]
    s = [r for r in x if r["status"] == "surface"]
    none = [r for r in x if r["status"] != "surface"]
    at40 = sum(level(r) == 40 for r in s)
    stopped = sum("12 GiB memory limit" in r["solver_status"] for r in s)
    slowest = max(float(r["wall_s"]) for r in x)
    assert max(level(r) for r in s) == 40
    assert all("12 GiB memory limit" in r["reason"] and "mpol=ntor=10" in r["reason"] for r in none)
    has([f"its slowest run took {slowest:.0f} s", f"{at40} of its {len(s)} surfaces are at M = 40, the top of the ladder",
         f"{stopped} runs passed 12 GiB inside a level", f"The {len(none)} that returned nothing",
         "part of that memory is this program's and not SIMSOPT's"])
    with open(os.path.join(C300, "boozer_exact_m44.csv")) as fh:
        more = list(csv.DictReader(fh))
    pairs = [(float(r["bn_M40"]), float(r["bn_M44"])) for r in more]
    assert sum(r["answer_M"] == "44" for r in more) == 6
    has(["thirty of these designs were also run with 16 GiB and a ladder to M = 48, and on twelve of them M = 40 and "
         "M = 44 both converged",
         "On six of the twelve the answer itself, the level with the smallest off-grid residual, moved to M = 44"])
    ratio = sorted(a / b for a, b in pairs)
    assert len(pairs) == 12 and round(statistics.median(ratio), 1) == 1.7 and 19.5 < ratio[-1] < 20.5
    more = [1 / r for r in ratio if r < 1]
    assert len(more) == 3 and round(max(more), 2) == 1.13
    has(["M = 44 leaks a median 1.7 times less than M = 40: 20 times less at most, and on three of the twelve up to "
         "1.13 times more"])


def test_boozer_ls():
    x = rows()["simsopt-boozer-ls"]
    s = [r for r in x if r["status"] == "surface"]
    at32 = sum(level(r) == 32 for r in s)
    hour = sum("3600 s budget" in r["solver_status"] for r in s)
    assert max(level(r) for r in s) == 32 and sum("M=36 needs about 14.0 GB" in r["solver_status"] for r in s) > 200
    has([f"{at32} of its {len(s)} surfaces are at M = 32", f"{hour} runs were stopped by the hour"])


def test_qfm():
    x = rows()["simsopt-qfm"]
    s = [r for r in x if r["status"] == "surface"]
    hour = sum("3600 s budget" in r["solver_status"] for r in s)
    at32 = sum(level(r) == 32 for r in s)
    has([f"{hour} of its {len(s)} runs were stopped by the hour", f"{at32} surfaces are at M = 32"])
    memory = [r["design"] for r in s if "12 GiB memory limit" in r["solver_status"]]
    assert memory == ["0003051"] and "after mpol=ntor=32" in next(r for r in s if r["design"] == "0003051")["solver_status"]
    has(["One, design 0003051, was stopped at 12 GiB inside those passes"])


def test_desc():
    x = rows()["desc"]
    s = [r for r in x if r["status"] == "surface"]
    none = [r for r in x if r["status"] != "surface"]
    assert all(r["solver_status"].startswith("level 0 (L,M,N)=(9,6,6)") for r in s)
    assert {r["n_modes"] for r in s} == {"85"}
    memory = sum("GiB of the task" in r["solver_status"] for r in s)
    hour = sum("s left" in r["solver_status"] for r in s)
    assert memory + hour == len(s)
    bad = sum("bad approximation" in r["reason"] for r in none)
    stored = statistics.median(int(r["n_modes"]) for r in rows()["quasr-stored"])
    has([f"Every one of its {len(s)} surfaces is at the first level, M = N = 6", "a boundary of 85 Fourier terms",
         f"a median {stored:.0f}", f"On {memory} of the {len(s)} the program's estimate of the memory",
         f"on {hour} its estimate of the time", f"Of the {len(none)} without a surface, {bad} ended"])
    with open(os.path.join(C300, "desc_ladder.csv")) as fh:
        ladder = {r["design"]: r for r in csv.DictReader(fh)}
    assert len(ladder) == len(x)
    for r in x:
        why = "memory" if "GiB of the task" in r["solver_status"] else "time" if "s left" in r["solver_status"] else ""
        assert ladder[r["design"]]["second_level_not_begun_for"] == why, r["design"]
    held = [float(v["process_peak_GiB"]) for v in ladder.values() if v["second_level_not_begun_for"] == "memory"]
    assert max(held) < 9.6 and min(float(v["second_level_memory_estimate_GiB"]) for v in ladder.values()
                                    if v["second_level_not_begun_for"] == "memory") > 12
    has(["the process never held more than 9.6 GiB"])


def test_vmecpp():
    x = rows()["vmecpp"]
    s = [r for r in x if r["status"] == "surface"]
    top = sum("answer from level 10 of 10" in r["solver_status"] for r in s)
    hour = sum(float(r["wall_s"]) >= 3500 for r in s)
    has([f"as it did for {hour} of its {len(s)} surfaces", f"{top} went through the last level"])


def test_torux_and_its_three_exceptions():
    x = {r["design"]: r for r in rows()["torux"]}
    with open(os.path.join(C300, "timing_exceptions.csv")) as fh:
        exc = list(csv.DictReader(fh))
    assert [e["design"] for e in exc] == ["0042793", "0946659", "0954058"]
    for e in exc:
        assert float(x[e["design"]]["wall_s"]) == float(e["wall_s_median"])
        assert float(e["wait_fraction_min"]) > 0.05 and float(e["other_work_fraction_max"]) < 0.05
    lo = min(float(e["wait_fraction_min"]) for e in exc)
    hi = max(float(e["wait_fraction_max"]) for e in exc)
    other = max(float(e["other_work_fraction_max"]) for e in exc)
    has(["Three designs (0042793, 0946659, 0954058)",
         f"their {exc[0]['timed_runs']}, {exc[1]['timed_runs']} and {exc[2]['timed_runs']} timed runs",
         f"{100 * lo:.1f} to {100 * hi:.1f}%", f"at most {100 * other:.1f}%"])


def sci(x, digits):
    """x as the page writes it: 5.199098215 × 10⁻⁷."""
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m} × 10{str(int(e)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def test_a_qfm_run_does_not_repeat():
    with open(os.path.join(C300, "qfm_twice.csv")) as fh:
        a, b = list(csv.DictReader(fh))
    assert a["design"] == b["design"] == "0003051" and a["levels"] == b["levels"] == "4,6,8,10"
    assert a["offgrid_bn_max"] != b["offgrid_bn_max"]
    has([f"one leaks {sci(float(a['offgrid_bn_max']), 1)} of |B| and the other {sci(float(b['offgrid_bn_max']), 1)}"])


def test_rows_run_again():
    with open(os.path.join(C300, "reruns.csv")) as fh:
        again = {r["solver"]: r for r in csv.DictReader(fh)}
    table = {s: {r["design"]: r for r in rs} for s, rs in rows().items()}
    names = {"torux": "torux", "simsopt-boozer-exact": "SIMSOPT BoozerExact", "simsopt-boozer-ls": "SIMSOPT BoozerLS",
             "vmecpp": "VMEC++", "simsopt-qfm": "SIMSOPT QFM"}
    digits = {"torux": 1, "simsopt-boozer-exact": 9, "simsopt-boozer-ls": 9, "vmecpp": 7, "simsopt-qfm": 1}
    for key, r in again.items():
        t = table[key][r["design"]]
        # the table's side of every rerun is the published row
        assert float(r["leak_table"]) == float(t["bn_max"]) and int(r["terms_table"]) == int(t["n_modes"])
        assert abs(float(r["seconds_table"]) - float(t["wall_s"])) < 0.006
        assert r["terms_rerun"] == r["terms_table"]
        secs = (f"{float(r['seconds_rerun']):.1f}, {float(r['seconds_table']):.1f}" if key == "torux" else
                f"{float(r['seconds_rerun']):,.0f}, {float(r['seconds_table']):,.0f}")
        has([f"| {names[key]} | {sci(float(r['leak_rerun']), digits[key])} | {sci(float(r['leak_table']), digits[key])} "
             f"| {secs} |"])
    ratio = {k: float(r["leak_rerun"]) / float(r["leak_table"]) for k, r in again.items()}
    assert all(abs(ratio[k] - 1) < 2e-7 for k in ("simsopt-boozer-exact", "simsopt-boozer-ls", "vmecpp"))
    assert max(float(again["torux"]["leak_rerun"]), float(again["torux"]["leak_table"])) < 1e-15
    has([f"design 0003051 came back at {sci(float(again['simsopt-qfm']['leak_rerun']), 1)} where the table has "
         f"{sci(float(again['simsopt-qfm']['leak_table']), 1)}"])
    # off x86-64 the judge reads in double precision: on torux's surfaces that is its own rounding
    f64 = statistics.median(float(r["f64_bn_max"]) for r in rows()["torux"] if r["status"] == "surface")
    has([f"that reading is the judge's own rounding, a median {sci(f64, 1)}"])


def test_the_readme_points_here_and_the_versions_are_the_pinned_ones():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        assert "(results/compare300/README.md)" in fh.read()
    pins = {}
    for name in ("simsopt", "desc", "vmecpp", "torux"):
        with open(os.path.join(C300, "requirements", name + ".txt")) as fh:
            pins[name] = fh.read()
    assert "simsopt==1.11.1" in pins["simsopt"] and "desc-opt==0.17.3" in pins["desc"] and "vmecpp==0.9.3" in pins["vmecpp"]
    has(["SIMSOPT 1.11.1, DESC 0.17.3, VMEC++ 0.9.3, torux 0.1.0"])
    # every package of each environment as it was installed: what the pins asked for is in it, as pinned
    for name, text in pins.items():
        def held(lines):
            return {k.strip().lower().replace("_", "-"): v.strip() for k, v in
                    (ln.split("==") for ln in lines.splitlines() if "==" in ln and not ln.startswith("#"))}
        with open(os.path.join(C300, "requirements", name + ".freeze.txt")) as fh:
            freeze = held(fh.read())
        asked = held(text)
        assert asked and all(freeze.get(k) == v for k, v in asked.items()), name
        assert len(freeze) >= len(asked) and not any("/" in v or "@" in v for v in freeze.values())
    has(["every package it then held"])
