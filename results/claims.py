"""Every number the README states, recomputed from the tables beside this file.

    python results/claims.py            # prints them. tests/test_claims.py holds the README to them

quasr.parquet     one row per QUASR design (371,701): torux's answer at the flux the design's outermost stored
                  surface encloses.  Columns are described in results/README.md.
compare300.csv    300 random QUASR designs (seed 20261107) through each tool, every returned surface measured by one
                  evaluator that shares no code with any tool, in extended precision at 4,000 random points.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def quasr():
    import pandas as pd
    t = pd.read_parquet(os.path.join(HERE, "quasr.parquet"))
    s = t[t.outcome == "surface"]
    c = t[t.outcome == "no_surface"]
    return dict(
        designs=len(t),
        surfaces=len(s),
        certified_none=len(c),
        island_bands=int((c.certificate == "island_band").sum()),
        edge_barriers=int((c.certificate == "edge_barrier").sum()),
        undetermined=int((t.outcome == "undetermined").sum()),
        answered_share=(len(s) + len(c)) / len(t),
        surface_share=len(s) / len(t),
        bn_solver_median=float(s.bn_solver.median()),
        bn_solver_max=float(s.bn_solver.max()),
        bn_check_median=float(s.bn_check.median()),
        bn_check_max=float(s.bn_check.max()),
        flux_err_median=float(s.flux_err.median()),
        flux_err_max=float(s.flux_err.max()),
        solve_s_median=float(s.solve_s.median()),
        solve_s_p90=float(s.solve_s.quantile(0.9)),
        check_s_median=float(s.check_s.median()),
        certificate_s_median=float(c.solve_s.median()),
        undetermined_s_median=float(t[t.outcome == "undetermined"].solve_s.median()),
        undetermined_under_1_min=float((t[t.outcome == "undetermined"].solve_s < 60).mean()),
    )


def both():
    """The 300 designs of the comparison in both tables: torux's one-hour answer against the row of quasr.parquet."""
    import pandas as pd
    t = pd.read_parquet(os.path.join(HERE, "quasr.parquet"), columns=["id", "outcome"]).set_index("id").outcome
    with open(os.path.join(HERE, "compare300.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["solver"] == "torux"]
    out = {}
    for r in rows:
        hour = r["solver_status"] if r["solver_status"] in ("surface", "no_surface") else "undetermined"
        key = f"table {t.loc[int(r['design'])]}, one hour {hour}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def compare():
    with open(os.path.join(HERE, "compare300.csv")) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for name in sorted({r["solver"] for r in rows}):
        mine = [r for r in rows if r["solver"] == name]
        surf = [r for r in mine if r["status"] == "surface"]
        bn = [float(r["bn_max"]) for r in surf]
        d = dict(designs=len(mine), surfaces=len(surf), bn_median=median(bn), bn_max=max(bn), bn_min=min(bn),
                 at_1e12=sum(b <= 1e-12 for b in bn))
        terms = [int(float(r["n_modes"])) for r in surf if r.get("n_modes")]
        if terms:
            d.update(terms_min=min(terms), terms_median=median(terms), terms_max=max(terms))
        if name != "quasr-stored":
            d.update(answer_s_median=median([float(r["answer_s"]) for r in surf]),          # when it had the surface
                     ran_to_the_hour=sum(float(r["wall_s"]) >= 3500 for r in surf),          # runs that filled the budget
                     surface_s_median=median([float(r["wall_s"]) for r in surf]),            # the whole run
                     surface_s_p90=sorted(float(r["wall_s"]) for r in surf)[int(0.9 * len(surf))],
                     cpu=sorted({r["cpu_model"] for r in mine}), threads=sorted({r["threads"] for r in mine}))
        if name == "torux":
            d.update(certified_none=sum(r["solver_status"] == "no_surface" for r in mine),
                     island_bands=sum(r.get("certificate") == "island_band" for r in mine),
                     edge_barriers=sum(r.get("certificate") == "edge_barrier" for r in mine),
                     undetermined=sum(r["solver_status"] not in ("surface", "no_surface") for r in mine),
                     certificate_s_median=median([float(r["wall_s"]) for r in mine if r["solver_status"] == "no_surface"]),
                     flux_err_max=max(abs(float(r["flux_rel_err"])) for r in surf))
        out[name] = d
    if "quasr-stored" in out and "torux" in out:
        out["stored_over_torux_median"] = out["quasr-stored"]["bn_median"] / out["torux"]["bn_median"]
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(dict(quasr=quasr(), compare300=compare(), both_tables=both()), indent=1))
