"""One rerun beside the published row: the evaluator on a runner's answer, in the columns of compare300.csv.

    python row.py simsopt-boozer-exact 0003051                  # reads compare300-out/simsopt-boozer-exact/
    python row.py desc 0003051 --out DIR                        # an answer in DIR/answers/quasr_0003051.json

Runs the comparison's evaluator (../evaluator/evaluate.py, or the directory COMPARE300_EVALUATOR names) on
DIR/answers/quasr_<id>.json with the design's coils from inputs/ beside this file, and prints the rerun above the
table's own row for the same design and solver: what the tool said, the leak the evaluator measures, the flux and
the time.  With --csv it prints both rows whole, in the table's columns (answer_s, which the table reads from each
run's log, is left empty for the rerun).  The precise tier needs a long double with extended precision (x86-64).
Elsewhere only the float64 tier runs and the leak compared is f64_bn_max.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR = Path(os.environ.get("COMPARE300_EVALUATOR") or HERE.parent / "evaluator").resolve()
KEYS = ("torux", "simsopt-boozer-exact", "simsopt-boozer-ls", "simsopt-qfm", "desc", "vmecpp")
COLS = ["design", "solver", "status", "solver_status", "reason", "wall_s", "requested_flux_Wb", "bn_max", "bn_median",
        "bn_p90", "f64_bn_max", "flux_Wb", "flux_rel_err", "flux_spread_rel", "check_points_max_m", "n_modes",
        "eval_error", "cpu_model", "threads", "certificate", "answer_s", "busy_cores"]
EVAL = COLS[7:17]


def load_evaluator():
    path = EVALUATOR / "evaluate.py"
    if not path.exists():
        sys.exit(f"no evaluate.py in {EVALUATOR}: set COMPARE300_EVALUATOR to the evaluator's directory")
    spec = importlib.util.spec_from_file_location("evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def certificate(ans):
    """island_band or edge_barrier for a certified no-surface answer of torux, as the table names them."""
    if ans.get("solver") != "torux" or ans.get("solver_status") != "no_surface":
        return ""
    kind = str(((ans.get("solver_reported") or {}).get("certificate") or {}).get("kind", ""))
    return next((name for name in ("island_band", "edge_barrier") if kind.startswith(name)), kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solver", choices=KEYS)
    ap.add_argument("design", help="the 7-digit QUASR id")
    ap.add_argument("--out", help="the runner's output directory (default: compare300-out/<solver> under the "
                                  "current directory)")
    ap.add_argument("--table", default=str(HERE.parent / "compare300.csv"), help="the published table")
    ap.add_argument("--csv", action="store_true", help="print both rows whole, in the table's columns")
    a = ap.parse_args()
    design = a.design.removeprefix("quasr_").zfill(7)
    out = Path(a.out) if a.out else Path.cwd() / "compare300-out" / a.solver
    ans_path = out / "answers" / f"quasr_{design}.json"
    if not ans_path.exists():
        sys.exit(f"no answer at {ans_path}: run the solver first, or pass --out")
    coils = next((p for p in (HERE / "inputs" / f"quasr_{design}.json", HERE / "inputs" / "pilot" /
                              f"quasr_{design}.json") if p.exists()), None)
    if coils is None:
        sys.exit(f"no inputs/quasr_{design}.json beside {Path(__file__).name}")
    ev = load_evaluator()
    import numpy as np
    precise = ev.precise_dtype() is not None
    ans = json.loads(ans_path.read_text())
    rec = json.loads(coils.read_text())
    host = ans.get("host") or {}
    row = dict(design=design, solver=a.solver, status=ans.get("status"), solver_status=ans.get("solver_status", ""),
               reason=(ans.get("reason") or "").replace("\n", " ")[:200], wall_s=ans.get("wall_s", ""),
               requested_flux_Wb=ans.get("requested_flux_Wb", ""),
               cpu_model=host.get("cpu_model") or host.get("cpu") or host.get("processor") or "",
               threads=next((host[k] for k in ("cores_used", "threads_used", "threads")
                             if isinstance(host.get(k), int) and not isinstance(host.get(k), bool)), ""),
               certificate=certificate(ans),
               busy_cores=next((f"{float(c['busy_cores']):.2f}" for c in [host.get("cores") or {}]
                                if c.get("busy_cores") is not None), ""))
    if ans.get("status") == "surface":
        try:
            res = ev.evaluate(rec, ans, precise=precise)
        except Exception as e:  # noqa: BLE001  (as evaluate.py records a failed evaluation)
            res = dict(eval_error=repr(e))
        for k in EVAL:
            row[k] = res.get(k, "")
    table = None
    tpath = Path(a.table)
    if tpath.exists():
        with open(tpath, newline="") as f:
            table = next((r for r in csv.DictReader(f) if r["design"] == design and r["solver"] == a.solver), None)
    key = "bn_max" if precise else "f64_bn_max"
    if a.csv:
        w = csv.writer(sys.stdout)
        w.writerow(COLS)
        w.writerow(["" if row.get(k) is None else row.get(k, "") for k in COLS])
        if table is not None:
            w.writerow([table.get(k, "") for k in COLS])
        else:
            print(f"only the rerun's row is printed: " + (f"no table at {tpath} (pass --table)" if not tpath.exists()
                  else f"the table has no row for {a.solver} {design}"), file=sys.stderr)
        return
    if not precise:
        print(f"long double here is not extended precision (eps {float(np.finfo(np.longdouble).eps):.1e}): the leak "
              f"below is the evaluator's float64 reading, f64_bn_max")

    def line(name, r):
        def num(k, fmt):
            v = r.get(k)
            return format(float(v), fmt) if v not in (None, "") else "-"
        return (f"{name:<11}{str(r.get('status') or '-'):<9}{num(key, '.9e'):>17}{num('flux_rel_err', '+.2e'):>12}"
                f"{str(r.get('n_modes') or '-'):>9}{num('wall_s', ',.1f'):>10}")

    print(f"{'':<11}{'status':<9}{'leak':>17}{'flux error':>12}{'terms':>9}{'seconds':>10}")
    print(line("this rerun", row))
    if table is not None:
        print(line("the table", table))
    elif not tpath.exists():
        print(f"no table at {tpath} (pass --table)")
    else:
        print(f"the table has no row for {a.solver} {design} (the six pilot designs are not in it)")
    print(f"this rerun: {row.get('solver_status') or row.get('reason') or ''}")
    if table is not None:
        print(f"the table:  {table.get('solver_status') or table.get('reason') or ''}")

if __name__ == "__main__":
    main()
