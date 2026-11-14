"""torux.many: many designs at once, each in its own process.

The truth each test rests on: single solves by torux.api.solve in this process (the solver, which shares no code
with the batch), the columns results/README.md documents and the values results/quasr.parquet holds, and the
operating system's own account of processes, signals and CPU time.
"""
import csv
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from conftest import example

import torux
from torux import batch
from torux.api import solve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECOND_FLUX = 0.04          # QUASR 954 again, inside its outermost stored surface (0.0691 Wb)


def _documented_columns():
    """The columns of results/quasr.parquet as results/README.md lists them."""
    with open(os.path.join(REPO, "results", "README.md")) as fh:
        text = fh.read()
    block = text.split("| column | meaning |")[1].split("\n\n")[0]
    cols = []
    for line in block.splitlines():
        m = re.match(r"\|\s*(`[^|]+`)\s*\|", line)
        if m:
            cols += re.findall(r"`([^`]+)`", m.group(1))
    return cols


def _rows(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def _raw_iota(x):
    """The solver's iota as the Result holds it (an Answer's iota property raises outside a verified surface)."""
    return x.__dict__["_iota"] if "_iota" in x.__dict__ else x.iota


def _same(a, b):
    """Two answers are the same solve: status, iota and every Fourier coefficient, bit for bit."""
    assert a.status == b.status
    ia, ib = _raw_iota(a), _raw_iota(b)
    assert (ia == ib) or (np.isnan(ia) and np.isnan(ib))
    assert (a.surface is None) == (b.surface is None)
    if a.surface is not None:
        for k in ("rc", "zs"):
            assert np.array_equal(np.asarray(getattr(a.surface, k)), np.asarray(getattr(b.surface, k)))
        assert a.surface.flux == b.surface.flux and (a.surface.M, a.surface.N) == (b.surface.M, b.surface.N)
        assert a.evidence["independent"]["bn_sup"] == b.evidence["independent"]["bn_sup"]
    if "proof" in b.evidence:
        assert a.evidence["proof"]["verdict"] == b.evidence["proof"]["verdict"]


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("many")
    second = d / "second.json"
    shutil.copy(example("quasr_0000954.json"), second)
    srcs = [example("quasr_0000954.json"), str(second), example("d3d_icoils_upper.dat")]
    table, save = d / "answers.csv", d / "answers"
    answers = batch.many(srcs, flux=[None, SECOND_FLUX, None], jobs=2, table=str(table), save=str(save))
    return dict(dir=d, sources=srcs, answers=answers, table=table, save=save)


def test_many_is_the_single_solves_bit_for_bit(batch_run, solved_954):
    a954, a2, ad3d = batch_run["answers"]
    _same(a954, solved_954)
    _same(a2, solve(batch_run["sources"][1], SECOND_FLUX))
    _same(ad3d, solve(example("d3d_icoils_upper.dat")))
    assert a954.status == a2.status == "surface" and ad3d.status == "no_surface"
    assert abs(abs(a2.surface.flux) - SECOND_FLUX) < 1e-12
    # the answers carry their field: rebuilt from the source, the solver's own engine to the last bit
    x = np.array([[0.9, 0.1, 0.05], [1.1, -0.2, 0.0]])
    assert np.array_equal(a954.field(x), solved_954.field(x))
    assert np.array_equal(a954.points(np.array([0.3]), np.array([0.2])),
                          solved_954.points(np.array([0.3]), np.array([0.2])))


def test_the_table_has_the_documented_columns_and_values(batch_run, solved_954):
    head, rows = _rows(batch_run["table"])
    # the source, the columns of results/quasr.parquet, the size of each verified surface, and the question asked
    assert head == ["source"] + _documented_columns() + ["minor_radius_m", "volume_m3", "asked"]
    with open(os.path.join(REPO, "results", "README.md"), encoding="utf-8") as fh:
        page = fh.read()
    assert "then `minor_radius_m` and `volume_m3` of the surface, then `asked`, the question the table answers" in page
    assert {r["asked"] for r in rows} == {"flux"}
    by = {r["source"]: r for r in rows}
    assert len(rows) == 3 and set(by) == {os.path.normpath(s) for s in batch_run["sources"]}
    r = by[os.path.normpath(example("quasr_0000954.json"))]
    s, ind = solved_954, solved_954.evidence["independent"]
    assert r["outcome"] == "surface" and r["note"] == "" and r["nfp"] == "2"
    assert float(r["iota"]) == s.iota and (int(r["M"]), int(r["N"])) == (s.surface.M, s.surface.N)
    assert float(r["bn_solver"]) == s.evidence["solver"]["bn_off"] and float(r["bn_check"]) == ind["bn_sup"]
    assert float(r["flux_err"]) == ind["flux_rel_err"]
    if 2 * batch._threads() > batch._cores():       # two solves at once share cores here: their seconds are left empty
        assert r["check_s"] == r["solve_s"] == ""
    else:
        assert float(r["check_s"]) > 0 and float(r["solve_s"]) > 0
    assert r["certificate"] == r["excluded_lo_Wb"] == r["bound_Wb"] == ""
    assert float(r["minor_radius_m"]) == s.minor_radius and float(r["volume_m3"]) == s.volume
    # against the published row of all of QUASR (another machine: the numbers agree to the last few digits)
    pd = pytest.importorskip("pandas")
    pub = pd.read_parquet(os.path.join(REPO, "results", "quasr.parquet")).set_index("id").loc[954]
    assert r["outcome"] == pub.outcome and int(r["nfp"]) == pub.nfp
    assert float(r["flux_Wb"]) == pytest.approx(pub.flux_Wb, rel=1e-13)
    assert float(r["iota"]) == pytest.approx(pub.iota, rel=1e-12)
    assert float(r["bn_check"]) < 1e-14 and float(r["flux_err"]) < 1e-12
    r2 = by[os.path.normpath(batch_run["sources"][1])]
    assert r2["outcome"] == "surface" and float(r2["flux_Wb"]) == SECOND_FLUX
    rd = by[os.path.normpath(example("d3d_icoils_upper.dat"))]
    assert rd["outcome"] == "no_surface" and rd["certificate"] == "no_toroidal_flux_surface"
    assert rd["iota"] == rd["M"] == rd["bn_check"] == ""
    # each answer also saved as JSON, the coefficients as solved
    names = sorted(os.listdir(batch_run["save"]))
    assert names == ["d3d_icoils_upper.json", "quasr_0000954.json", "second.json"]
    with open(batch_run["save"] / "quasr_0000954.json") as fh:
        saved = json.load(fh)
    assert np.array_equal(np.array(saved["surface"]["rc"]), solved_954.surface.rc)


def test_a_second_run_solves_nothing_again(batch_run, monkeypatch):
    before = batch_run["table"].read_bytes()

    def no(*a, **k):
        raise AssertionError("a design already in the table was solved again")
    monkeypatch.setattr(batch, "_start", no)
    again = batch.many(batch_run["sources"], flux=[None, SECOND_FLUX, None], jobs=2, table=str(batch_run["table"]))
    assert again == [None, None, None]
    assert batch_run["table"].read_bytes() == before
    # with the save folder that holds them, each answer comes back as saved
    first = batch_run["answers"]
    saved = batch.many(batch_run["sources"], flux=[None, SECOND_FLUX, None], jobs=2, table=str(batch_run["table"]),
                       save=str(batch_run["save"]))
    assert batch_run["table"].read_bytes() == before
    for a, b in zip(saved, first):
        assert a.status == b.status and str(a).splitlines()[0] == str(b).splitlines()[0]
        if b.surface is not None:
            assert np.array_equal(a.surface.rc, b.surface.rc) and np.array_equal(a.surface.zs, b.surface.zs)


def _children(pid):
    """{pid: command line} of the processes whose parent is ``pid`` (from ps)."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    kids = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[1]) == pid:
            kids[int(parts[0])] = parts[2]
    return kids


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def test_a_missing_source_and_a_killed_solve_are_error_rows(tmp_path):
    """W7-X from its coils alone takes minutes: its process is killed with SIGKILL two seconds in.  A file that
    does not exist fails in its reader.  QUASR 954 finishes as if neither had happened."""
    srcs = [str(tmp_path / "no_such_coils.json"), example("w7x.json"), example("quasr_0000954.json")]
    killed = []

    def killer():
        t_end = time.time() + 120
        while time.time() < t_end and not killed:
            for pid, cmd in _children(os.getpid()).items():
                if "torux.batch" in cmd:
                    job = next(w for w in cmd.split() if w.endswith(".job"))
                    try:
                        import pickle
                        with open(job, "rb") as fh:
                            src = pickle.load(fh)["source"]
                    except OSError:
                        continue
                    if src == example("w7x.json"):
                        time.sleep(2.0)
                        os.kill(pid, signal.SIGKILL)
                        killed.append(pid)
            time.sleep(0.1)
    th = threading.Thread(target=killer, daemon=True)
    th.start()
    table = tmp_path / "t.csv"
    ans = batch.many(srcs, flux=[None, 2.2, None], jobs=2, table=str(table))
    th.join(1)
    assert killed
    assert ans[0].status == "error" and "FileNotFoundError" in ans[0].evidence["error"]
    assert ans[1].status == "error" and "SIGKILL" in ans[1].evidence["error"]
    assert ans[2].status == "surface"
    _, rows = _rows(table)
    by = {r["source"]: r for r in rows}
    assert by[os.path.normpath(srcs[0])]["outcome"] == "error"
    assert "SIGKILL" in by[os.path.normpath(srcs[1])]["note"] and by[os.path.normpath(srcs[1])]["outcome"] == "error"
    assert by[os.path.normpath(srcs[2])]["outcome"] == "surface"
    assert not _alive(killed[0])


SCRIPT = """
import sys, torux.batch as b
b.many([sys.argv[1], sys.argv[1]], flux=[2.2, 2.0], jobs=2)
"""


def _script():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen([sys.executable, "-c", SCRIPT, example("w7x.json")], env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _wait_children(pid, n, timeout=60):
    t = time.time() + timeout
    while time.time() < t:
        kids = [k for k, c in _children(pid).items() if "torux.batch" in c]
        if len(kids) >= n:
            return kids
        time.sleep(0.1)
    raise AssertionError(f"the batch started fewer than {n} workers in {timeout} s")


def test_ctrl_c_stops_everything():
    """SIGINT to the batch's process group, as a terminal's Ctrl-C sends it: the batch stops at once, and no worker
    is left running."""
    p = _script()
    kids = _wait_children(p.pid, 2)
    time.sleep(1.0)
    os.killpg(p.pid, signal.SIGINT)
    try:
        rc = p.wait(timeout=15)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    err = p.stderr.read().decode()
    assert rc != 0 and "KeyboardInterrupt" in err
    t = time.time() + 5
    while time.time() < t and any(_alive(k) for k in kids):
        time.sleep(0.1)
    assert not any(_alive(k) for k in kids)


def test_workers_stop_when_their_batch_is_killed():
    """SIGKILL to the batch alone (it can clean nothing up): its workers see their lifeline close and stop."""
    p = _script()
    kids = _wait_children(p.pid, 2)
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)
    t = time.time() + 10
    while time.time() < t and any(_alive(k) for k in kids):
        time.sleep(0.1)
    assert not any(_alive(k) for k in kids)


def test_thread_arithmetic(monkeypatch, tmp_path):
    """By default jobs x TORUX_THREADS fits the cores. Asked for more, the table leaves its seconds empty."""
    threads = int(os.environ["TORUX_THREADS"])
    assert threads == torux._threads()
    monkeypatch.setattr(batch, "_cores", lambda: 3 * threads + 1)
    seen = []
    real = batch._pool
    monkeypatch.setattr(batch, "_pool", lambda todo, jobs, args, finish: seen.append(jobs))
    batch.many([example("d3d_icoils_upper.dat")])
    assert seen == [3]
    monkeypatch.setattr(batch, "_pool", real)
    srcs = []
    for i in range(3):
        srcs.append(str(tmp_path / f"d{i}.dat"))
        shutil.copy(example("d3d_icoils_upper.dat"), srcs[-1])
    monkeypatch.setattr(batch, "_cores", lambda: threads)
    with pytest.warns(UserWarning, match="share cores"):
        batch.many(srcs, jobs=3, table=str(tmp_path / "shared.csv"))
    _, rows = _rows(tmp_path / "shared.csv")
    assert [r["outcome"] for r in rows] == ["no_surface"] * 3
    assert all(r["solve_s"] == "" and r["check_s"] == "" for r in rows)


def test_a_solve_uses_no_more_cores_than_its_threads(monkeypatch):
    """One worker with TORUX_THREADS=1, measured by the operating system: its CPU time is at most its wall time."""
    for v in ("TORUX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(v, "1")
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    a = batch.many([example("quasr_0000954.json")], jobs=1)[0]
    wall = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    print(f"one thread: {cpu:.2f} s of CPU in {wall:.2f} s")
    assert a.status == "surface"
    assert cpu <= 1.1 * wall + 0.2


def test_a_table_cut_off_mid_row_is_continued(tmp_path):
    table = tmp_path / "t.csv"
    d = str(tmp_path / "d.dat")
    shutil.copy(example("d3d_icoils_upper.dat"), d)
    table.write_text(",".join(batch.COLUMNS) + "\n" + os.path.normpath(d) + ",,1,,no_su")
    ans = batch.many([d], table=str(table))
    assert ans[0].status == "no_surface"
    _, rows = _rows(table)
    assert len(rows) == 1 and rows[0]["certificate"] == "no_toroidal_flux_surface"


def test_requests_that_cannot_be_run_are_refused(tmp_path):
    d = example("d3d_icoils_upper.dat")
    with pytest.raises(ValueError, match=".csv"):
        batch.many([d], table=str(tmp_path / "t.parquet"))
    with pytest.raises(ValueError, match="twice"):
        batch.many([d, d], table=str(tmp_path / "t.csv"))
    with pytest.raises(ValueError, match="one per source"):
        batch.many([d, d], flux=[0.1])
    with pytest.raises(ValueError, match="whole number"):
        batch.many([d], jobs=0)
    (tmp_path / "other.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not a table torux.many wrote"):
        batch.many([d], table=str(tmp_path / "other.csv"))


def test_every_design_is_asked_the_same_question_as_one(tmp_path, solved_954):
    """A scan by minor radius: each answer is torux.solve's with the same arguments, bit for bit, and its row holds
    the answer's own minor radius and volume."""
    q = example("quasr_0000954.json")
    one = torux.solve(q, minor_radius=0.12)
    table = tmp_path / "t.csv"
    got = batch.many([q], minor_radius=0.12, table=str(table), jobs=1)[0]
    _same(got, one)
    assert got.evidence["search"]["question"] == "minor_radius" and got.minor_radius == one.minor_radius
    _, rows = _rows(table)
    assert float(rows[0]["minor_radius_m"]) == got.minor_radius and float(rows[0]["volume_m3"]) == got.volume
    assert float(rows[0]["flux_Wb"]) == got.flux and rows[0]["outcome"] == "surface"
    with pytest.raises(ValueError, match="not flux and minor_radius"):
        batch.many([q], flux=0.05, minor_radius=0.12)
    with pytest.raises(ValueError, match="positive number of metres"):
        batch.many([q], minor_radius=-0.1)


def test_an_error_row_is_tried_again_and_replaced(tmp_path, capsys):
    """The same command again: a design whose row says error is solved again, and its row replaced, never doubled.
    a table written in the columns of results/quasr.parquet alone is continued in them."""
    d = str(tmp_path / "late.dat")
    table = tmp_path / "t.csv"
    first = batch.many([d], table=str(table))
    assert first[0].status == "error" and _rows(table)[1][0]["outcome"] == "error"
    shutil.copy(example("d3d_icoils_upper.dat"), d)
    rep = batch._Report(sys.stdout)
    batch._run([d], None, None, str(table), None, None, 3e-15, True, False, rep)
    out = capsys.readouterr().out
    assert "0 of 1 designs are already in" in out and "1 whose row says error is tried again" in out
    head, rows = _rows(table)
    assert len(rows) == 1 and rows[0]["outcome"] == "no_surface"
    old = tmp_path / "old.csv"
    assert batch.COLUMNS == batch.COLUMNS_SIZES + ("asked",) and batch.COLUMNS_SIZES[:18] == batch.COLUMNS_QUASR
    old.write_text(",".join(batch.COLUMNS_QUASR) + "\n")
    batch.many([d], table=str(old))
    head, rows = _rows(old)
    assert tuple(head) == batch.COLUMNS_QUASR and len(rows) == 1 and rows[0]["outcome"] == "no_surface"


def test_command_line(tmp_path, capsys):
    import argparse
    ap = argparse.ArgumentParser(prog="torux")
    batch.add_parser(ap.add_subparsers(dest="cmd"))
    table = tmp_path / "a.csv"
    a = ap.parse_args(["many", example("d3d_icoils_upper.dat"), str(tmp_path / "missing.json"), "--table", str(table),
                       "--jobs", "2"])
    assert batch.main(a) == 0
    out = capsys.readouterr().out
    assert "no_surface" in out and "error" in out and "1 error  1 no surface" in out
    a = ap.parse_args(["many", example("d3d_icoils_up*.dat"), "--table", str(table)])
    assert batch.main(a) == 0
    assert "1 of 1 designs are already in" in capsys.readouterr().out


def test_the_first_line_says_what_is_coming(tmp_path, capsys):
    """Before the first design ends there is nothing to print: the count and the pace are said at once, on stderr,
    so that stdout stays the lines of the designs."""
    import argparse
    ap = argparse.ArgumentParser(prog="torux")
    batch.add_parser(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args(["many", example("d3d_icoils_upper.dat"), "--table", str(tmp_path / "a.csv")])
    assert batch.main(a) == 0
    said = capsys.readouterr()
    assert said.err.splitlines()[0] == "torux: 1 design to solve, 1 at a time: each is a line below when it ends"
    assert "to solve" not in said.out
    batch._Report(sys.stdout).starting(3, 2)
    assert capsys.readouterr().err == "torux: 3 designs to solve, 2 at a time: each is a line below when it ends\n"
    assert batch.main(a) == 0                      # nothing left to solve: nothing is announced
    assert "to solve" not in capsys.readouterr().err


def test_a_size_that_was_not_settled_is_explained_in_its_row():
    """Asked by size and not settled: flux_Wb is the flux that size would need, and the row's note says so. A
    request by flux, and a size that was found, need no note."""
    from torux.answer import Answer
    from torux.api import Result
    need = "flux_Wb is the flux that size would need"
    for status, question, asked, note in (
            ("undetermined", "minor_radius", 0.3, "asked for minor radius 0.3 m: " + need),
            ("timeout", "volume", 0.5, "timeout, asked for volume 0.5 m^3: " + need),
            ("undetermined", None, None, "")):
        # flux_asked is the request whose certificate ruled the size out: it is not the flux the size would need
        ev = dict(search=dict(question=question, asked=asked, flux=-0.4, flux_asked=0.07)) if question else {}
        row = batch._row(Answer._of(Result(status, 100.0, 99.0, evidence=ev), None if question else -0.4), None,
                         None if question else -0.4)
        assert (row["outcome"], row["note"], row["flux_Wb"]) == ("undetermined", note, 0.4)
    # a search cut short before it came to a flux: no flux is written, and the note claims none
    ev = dict(search=dict(question="minor_radius", asked=0.3, flux=None))
    row = batch._row(Answer._of(Result("timeout", 100.0, 99.0, evidence=ev), None), None, None)
    assert (row["note"], row["flux_Wb"]) == ("timeout, asked for minor radius 0.3 m", None)


def _table_of(tmp_path, name, lines):
    """A table as torux.many writes it, with the rows given as (source, outcome, asked)."""
    path = tmp_path / name
    k, q = batch.COLUMNS.index("outcome"), batch.COLUMNS.index("asked")
    out = [",".join(batch.COLUMNS)]
    for source, outcome, asked in lines:
        cells = [""] * len(batch.COLUMNS)
        cells[0], cells[k], cells[q] = source, outcome, asked
        out.append(",".join(cells))
    path.write_text("\n".join(out) + "\n")
    return path


def test_rows_of_other_designs_stay_when_a_table_is_continued(tmp_path):
    """Continuing a table solves again the designs of this run whose rows say error, and touches no other row:
    an error row of a design that is not in this run stays."""
    d = str(tmp_path / "d.dat")
    shutil.copy(example("d3d_icoils_upper.dat"), d)
    table = _table_of(tmp_path, "t.csv", [("elsewhere.json", "error", "flux"), (os.path.normpath(d), "error", "flux"),
                                           ("kept.json", "surface", "flux")])
    ans = batch.many([d], table=str(table))
    assert ans[0].status == "no_surface"
    _, rows = _rows(table)
    assert [(r["source"], r["outcome"]) for r in rows] == [
        ("elsewhere.json", "error"), ("kept.json", "surface"), (os.path.normpath(d), "no_surface")]
    assert rows[-1]["asked"] == "flux"


def test_a_table_answers_one_question(tmp_path, monkeypatch):
    """A design with a row is not solved again, so a table continued with another question would answer the old
    one in silence: it is refused, and says which question the table holds."""
    d = str(tmp_path / "d.dat")
    shutil.copy(example("d3d_icoils_upper.dat"), d)
    table = _table_of(tmp_path, "mr.csv", [(os.path.normpath(d), "surface", "minor radius 0.12 m")])
    for kw, word in ((dict(minor_radius=0.10), "minor radius 0.1 m"), (dict(largest=True), "largest"),
                     (dict(), "a flux"), (dict(volume=0.3), "volume 0.3 m^3")):
        with pytest.raises(ValueError, match="another question") as e:
            batch.many([d], table=str(table), **kw)
        assert "minor radius 0.12 m" in str(e.value) and word in str(e.value)
    assert batch.many([d], table=str(table), minor_radius=0.12) == [None]      # the same question: nothing to solve
    # and a flux: the row of a design answers one flux
    table = _table_of(tmp_path, "f.csv", [(os.path.normpath(d), "surface", "flux")])
    text = table.read_text().splitlines()
    cells = text[1].split(",")
    cells[batch.COLUMNS.index("flux_Wb")] = "0.05"
    table.write_text(text[0] + "\n" + ",".join(cells) + "\n")
    with pytest.raises(ValueError, match="at 0.05 Wb, where this run asks 0.06 Wb"):
        batch.many([d], table=str(table), flux=0.06)
    assert batch.many([d], table=str(table), flux=-0.05) == [None] and batch.many([d], table=str(table)) == [None]
    assert batch._asked(None, dict(minor_radius=0.12)) == "minor radius 0.12 m"
    assert batch._asked(0.05, {}) == batch._asked(None, {}) == "flux"
    assert batch._asked(None, dict(largest=True)) == "largest"


def test_a_design_is_known_by_its_file_however_its_path_is_spelled(tmp_path, monkeypatch):
    d = tmp_path / "designs" / "d.dat"
    d.parent.mkdir()
    shutil.copy(example("d3d_icoils_upper.dat"), d)
    monkeypatch.chdir(tmp_path)
    table = _table_of(tmp_path, "t.csv", [(os.path.join("designs", "d.dat"), "no_surface", "flux")])
    for spelled in (str(d), os.path.join("designs", "..", "designs", "d.dat"), os.path.join(".", "designs", "d.dat")):
        assert batch.many([spelled], table=str(table)) == [None]
    assert len(_rows(table)[1]) == 1
    with pytest.raises(ValueError, match="twice"):
        batch.many([str(d), os.path.join("designs", "d.dat")], table=str(tmp_path / "u.csv"))
    assert batch._key("quasr:954") == "quasr:954" and batch._key("no-such-file.json") == "no-such-file.json"


def test_answer_file_names(tmp_path):
    items = batch._items([str(tmp_path / "a" / "coils.json"), str(tmp_path / "b" / "coils.json"), "quasr 954"])
    names = batch._save_names(items)
    assert sorted(names.values()) == ["a_coils", "b_coils", "quasr_0000954"]
    assert [it.name for it in items][2] == "quasr:954"
    items = batch._items([example("hsx.json"), example("w7x.json")])
    assert sorted(batch._save_names(items).values()) == ["hsx", "w7x"]


def test_a_solve_far_past_its_budget_is_stopped(monkeypatch):
    """The process of a solve that runs past its budget by more than the margin is stopped, and says so."""
    monkeypatch.setattr(batch, "_OVERRUN", (0.0, 3.0))
    a = batch.many([example("w7x.json")], flux=2.2, jobs=1)[0]
    assert a.status == "error" and "was stopped" in a.evidence["error"]


def test_blas_loaded_before_torux_gets_torux_threads():
    """numpy and scipy imported before torux: every BLAS pool threadpoolctl finds has TORUX_THREADS threads or
    fewer, as when torux comes first (a pool's thread count changes a solve's last bits).  Under Accelerate there
    is no pool to see."""
    code = ("import json, numpy, scipy.linalg, torux, os; from threadpoolctl import threadpool_info; "
            "print(json.dumps([os.environ['TORUX_THREADS'], [(i['internal_api'], i['num_threads']) "
            "for i in threadpool_info() if i['user_api'] == 'blas']]))")
    env = dict(os.environ, TORUX_THREADS="2")
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.pop(k, None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    n, pools = json.loads(out.stdout.strip().splitlines()[-1])
    assert n == "2" and all(t <= 2 for _, t in pools), pools


def test_the_advice_of_a_refusal_works_in_many(tmp_path, capsys):
    """A SIMSOPT file of several coil sets, and one whose Current states two values: their refusals say --coils and
    --currents, which torux 0.1.4's many did not take. Now it does, for every design, and torux.many takes which=
    and currents= as torux.load does."""
    from test_simsopt_files import _boozer_file, _two_sets, _two_valued
    from torux.cli import main
    sets = _boozer_file(tmp_path / "sets.json", change=_two_sets)
    two = _boozer_file(tmp_path / "two.json", change=_two_valued(3.95))
    table = str(tmp_path / "t.csv")
    assert main(["many", sets, two, "--table", table, "--flux", "0.05", "--jobs", "2"]) == 0
    notes = " ".join(r["note"] for r in _rows(table)[1])
    assert "--coils NAME" in notes and "--currents saved" in notes
    capsys.readouterr()
    assert main(["many", sets, two, "--table", table, "--flux", "0.05", "--jobs", "2", "--coils", "BiotSavart1",
                 "--currents", "dofs"]) == 0
    head, rows = _rows(table)
    assert sorted(r["outcome"] for r in rows) == ["surface", "surface"]
    got = batch.many([sets, two], 0.05, which="BiotSavart3", currents="dofs", jobs=2)
    assert all(a.status == "error" and "matches 0" in a.evidence["error"] for a in got)


def _renamed_sets(d):
    """Two coil sets named as another SIMSOPT session would name them: BiotSavart3 and BiotSavart4."""
    o = d["simsopt_objs"]
    o["BiotSavart3"] = dict(o.pop("BiotSavart1"), **{"@name": "BiotSavart3"})
    o["BiotSavart4"] = dict(o["BiotSavart3"], **{"@name": "BiotSavart4"})
    o["BoozerSurface1"]["biotsavart"] = {"$type": "ref", "value": "BiotSavart3"}


def test_coils_and_currents_once_per_source_are_each_its_own_design(tmp_path, capsys):
    """torux many's help said to give --coils once per source file, and doing so gave every design the whole list,
    which torux.load refused for a file of one: every design an error, naming which=. Now --coils and --currents
    given once are every design's, given once per SOURCE each is its own design's, - for one that needs none, and
    any other count is one error, naming the option, before any design is solved. torux.many takes which= and
    currents= the same way."""
    import argparse
    from test_simsopt_files import _boozer_file, _two_sets, _two_valued
    from torux.cli import main
    a = _boozer_file(tmp_path / "a.json", change=_two_sets)                 # BiotSavart1, BiotSavart2
    b = _boozer_file(tmp_path / "b.json", change=_renamed_sets)             # BiotSavart3, BiotSavart4
    two = _boozer_file(tmp_path / "two.json", change=_two_valued(3.95))
    q = example("quasr_0000954.json")
    table = tmp_path / "t.csv"
    assert main(["many", a, b, q, "--table", str(table), "--flux", "0.05", "--coils", "BiotSavart1", "--coils",
                 "BiotSavart3"]) == 1
    err = capsys.readouterr().err
    assert "--coils" in err and "which=" not in err and "3 SOURCE" in err and not table.exists()
    assert main(["many", a, b, "--table", str(table), "--flux", "0.05", "--currents", "dofs", "--currents", "dofs",
                 "--currents", "saved"]) == 1
    err = capsys.readouterr().err
    assert "--currents" in err and "currents=" not in err and not table.exists()
    assert main(["many", a, b, two, "--table", str(table), "--flux", "0.05", "--jobs", "3", "--coils", "BiotSavart2",
                 "--coils", "BiotSavart4", "--coils", "-", "--currents", "-", "--currents", "-", "--currents",
                 "dofs"]) == 0
    head, rows = _rows(table)
    assert sorted(r["outcome"] for r in rows) == ["surface"] * 3, [r["note"] for r in rows]
    with pytest.raises(ValueError, match="which= has 2 entries for 3 sources"):
        batch.many([a, b, q], 0.05, which=["BiotSavart1", "BiotSavart3"])
    with pytest.raises(ValueError, match="currents= has 1 entry for 2 sources"):
        batch.many([a, b], 0.05, which=0, currents=["dofs"])
    got = batch.many([a, b, q], 0.05, which=["BiotSavart2", "BiotSavart3", None], jobs=3)
    assert [g.status for g in got] == ["surface"] * 3
    ap = argparse.ArgumentParser(prog="torux")
    sub = ap.add_subparsers(dest="cmd")
    batch.add_parser(sub)
    from torux.check import add_command
    add_command(sub)
    said = {k: " ".join(sub.choices[k].format_help().split()) for k in ("many", "check")}
    assert "Given once, it applies to every design" in said["many"] and "once per SOURCE" in said["many"]
    assert "once per file in order" in said["check"]
