"""Many designs at once: each solved in its own process, a fixed number at a time, every answer kept.

    answers = torux.many(sorted(glob.glob("designs/*.json")), jobs=8, table="answers.csv", save="answers/")
    answers = torux.many(files, minor_radius=0.1, table="answers.csv")     # the same question of every design

    torux many designs/*.json --table answers.csv [--flux WB | --minor-radius M | --volume M3 | --largest]
               [--budget S] [--jobs N] [--save DIR]

Each design is solved by a fresh Python process that imports this module and runs one job, never a fork of
the caller: a fork of a process whose JAX or numba has started its threads can hang, and a spawned
multiprocessing child imports the caller's main script again.  A fresh interpreter has neither problem on any
system, and the result of a solve does not depend on what was solved before it in the same process.

The processes run ``jobs`` at a time, each with the TORUX_THREADS threads every torux solve uses, so that by
default they ask for no more cores than the machine (or the job's CPU allocation) has.  A design that fails in
any way, a reader's exception, a crash or a kill, is that design's answer, with status 'error' and the reason. The
same command again tries it again.  Ctrl-C stops every process the batch started. A process whose batch has gone
stops itself.
"""
from __future__ import annotations

import csv
import math
import numbers
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import warnings

# The columns of results/quasr.parquet (results/README.md), after the source's name.
COLUMNS_QUASR = ("source", "id", "nfp", "flux_Wb", "outcome", "note", "iota", "M", "N", "bn_solver", "bn_check",
                 "flux_err", "solve_s", "check_s", "certificate", "excluded_lo_Wb", "excluded_hi_Wb", "bound_Wb")
# and the size of each verified surface, which a scan by size or for the largest is read for
COLUMNS_SIZES = COLUMNS_QUASR + ("minor_radius_m", "volume_m3")
# and the question the row answers: a table holds the answers to one question
COLUMNS = COLUMNS_SIZES + ("asked",)

# a stated budget is a promise up to one last step: past twice it, and five minutes more, the process is stopped
_OVERRUN = (2.0, 300.0)
# the process's own watchdog goes first. One that has itself hung is stopped by the batch this many seconds later
_GRACE = 15.0
# what a fresh interpreter runs for one design
_WORKER = "import sys; from torux.batch import _worker_main; _worker_main(sys.argv[1], sys.argv[2])"


def many(sources, flux=None, *, minor_radius=None, volume=None, largest=False, jobs=None, table=None, save=None,
         budget_s=None, tol=3e-15, verify=True, gradients=False):
    """Solve many sources at once, each in its own process, ``jobs`` at a time, and return the answers in the
    order given.

    sources   a list of anything torux.load takes (a tuple of files is one source), or a glob pattern
    flux, minor_radius, volume, largest
              the question asked of every design, as torux.solve asks it of one: a full toroidal flux (Wb, and for
              flux also one per source), a minor radius (m), a volume (m^3) or the largest surface. None of them:
              each source's own request, or its largest surface when it carries none
    jobs      processes at once. Default: the machine's cores divided by the threads each solve uses
    table     a .csv file with one row a design, written as each finishes.  A table that exists is continued:
              the designs already in it are not solved again
    save      a folder where each answer is also written as JSON
    The rest are torux.solve's.  A design that fails is an answer with status 'error' and the reason.  A design
    already in the table comes back as the answer saved for it in ``save`` when that folder holds it, else None."""
    return _run(sources, flux, jobs, table, save, budget_s, tol, verify, gradients, None,
                dict(minor_radius=minor_radius, volume=volume, largest=largest))


# --------------------------------------------------------------------------- the batch
class _Item:
    __slots__ = ("payload", "name", "quasr", "flux", "answer", "skip", "key")

    def __init__(self, payload, name, quasr=None):
        self.payload, self.name, self.quasr = payload, name, quasr
        self.flux, self.answer, self.skip = None, None, False
        self.key = _key(name)


def _key(name):
    """What a design is known by in a table: its file, however the path to it is spelled from this folder. A name
    that is no file here (a QUASR number, a source built in code, a file named from another folder) is itself."""
    name = str(name)
    parts = name.split("+")
    if all(os.path.exists(q) for q in parts):
        return "+".join(os.path.realpath(q) for q in parts)
    return name


def _asked(flux, size):
    """The question of a run, as its rows record it."""
    if size.get("minor_radius") is not None:
        return f"minor radius {float(size['minor_radius']):.12g} m"
    if size.get("volume") is not None:
        return f"volume {float(size['volume']):.12g} m^3"
    if size.get("largest"):
        return "largest"
    return "flux"


def _expand(s):
    """A pattern such as designs/*.json as the files it matches, in order (where no shell has expanded it)."""
    if os.path.exists(s) or not any(c in s for c in "*?["):
        return [s]
    import glob
    found = sorted(glob.glob(s))
    if not found:
        raise ValueError(f"no file matches {s!r}")
    return found


def _items(sources):
    from .source import Source
    from .sources import load, quasr_number
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    flat = []
    for x in list(sources):
        flat += _expand(os.fspath(x)) if isinstance(x, (str, os.PathLike)) else [x]
    out = []
    for x in flat:
        if isinstance(x, str):
            n = quasr_number(x) if not os.path.exists(x) else None
            out.append(_Item(x, f"quasr:{n}" if n is not None else os.path.normpath(x), n))
        elif isinstance(x, (tuple, list)) and x and all(isinstance(p, (str, os.PathLike)) for p in x):
            ps = tuple(os.fspath(p) for p in x)
            out.append(_Item(ps, "+".join(os.path.normpath(p) for p in ps)))
        elif isinstance(x, Source):
            out.append(_Item(x, x.id))
        else:
            # a live SIMSOPT or DESC object: written and read here, and the Source goes to the worker
            try:
                src = load(x)
                out.append(_Item(src, src.id))
            except Exception as e:  # noqa: BLE001
                it = _Item(None, getattr(x, "name", None) or type(x).__name__)
                it.answer = _error_answer(f"{type(e).__name__}: {e}", 0.0, None)
                out.append(it)
    return out


def _cores():
    """The cores this process may use: its affinity mask, and on Linux its cgroup's CPU quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    q = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            a, b = fh.read().split()[:2]
            if a != "max":
                q = float(a) / float(b)
    except (OSError, ValueError):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
                a = float(fh.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
                b = float(fh.read())
            if a > 0 and b > 0:
                q = a / b
        except (OSError, ValueError):
            pass
    if q:
        n = min(n, max(1, int(q)))
    return max(1, n)


def _threads():
    """The threads each solve uses (TORUX_THREADS, as torux set it on import)."""
    from . import _threads as threads
    return threads()


def _run(sources, flux, jobs, table, save, budget_s, tol, verify, gradients, report, size=None):
    from .ask import one_question
    size = {**dict(minor_radius=None, volume=None, largest=False), **(size or {})}
    one_question(flux if flux is None or isinstance(flux, numbers.Real) else None, **size)
    if flux is not None and not isinstance(flux, numbers.Real):
        for f in flux:
            one_question(f)
        if any(f is not None for f in flux) and (size["minor_radius"] is not None or size["volume"] is not None
                                                 or size["largest"]):
            one_question(1.0, **size)            # says: one of flux, minor_radius, volume or largest
    if budget_s is not None and (isinstance(budget_s, bool) or not isinstance(budget_s, numbers.Real)
                                 or not math.isfinite(budget_s) or budget_s <= 0):
        raise ValueError(f"budget_s={budget_s!r}: give the seconds for each design's whole question as a positive "
                         f"number")
    items = _items(sources)
    if flux is None or isinstance(flux, numbers.Real):
        for it in items:
            it.flux = None if flux is None else float(flux)
    else:
        fl = list(flux)
        if len(fl) != len(items):
            raise ValueError(f"{len(fl)} flux{'es' if len(fl) != 1 else ''} for {len(items)} sources: give one flux, "
                             f"or one per source")
        for it, f in zip(items, fl):
            it.flux = None if f is None else float(f)
    threads, cores = _threads(), _cores()
    if jobs is None:
        jobs = max(1, cores // threads)
    if int(jobs) != jobs or jobs < 1:
        raise ValueError(f"jobs={jobs!r}: give a whole number of processes, at least 1")
    jobs = int(jobs)
    if table is not None or save is not None:
        seen, shown = {}, {}
        for it in items:
            seen[it.key] = seen.get(it.key, 0) + 1
            shown.setdefault(it.key, it.name)
        dup = sorted(shown[k] for k, v in seen.items() if v > 1)
        if dup:
            raise ValueError(("a source appears" if len(dup) == 1 else f"{len(dup)} sources appear")
                             + f" twice ({', '.join(map(repr, dup[:5]))}): a table or a save folder needs each "
                               f"source once")
    names = _save_names(items) if save is not None else None
    if save is not None:
        os.makedirs(save, exist_ok=True)
    asked = _asked(flux, size)
    tab = _Table(table, items, asked) if table is not None else None
    try:
        if tab is not None:
            for it in items:
                if it.key in tab.done:
                    it.skip = True
            if report is not None:
                report.skipped(sum(it.skip for it in items), len(items), table,
                               sum(it.key in tab.retried for it in items))
        args = dict(budget_s=budget_s, tol=tol, verify=verify, gradients=gradients, **size)
        todo = [it for it in items if it.answer is None and not it.skip]
        at_once = min(jobs, len(todo))
        if report is not None and todo:
            report.starting(len(todo), at_once)
        shared = at_once * threads > cores
        if shared:
            warnings.warn(f"{at_once} processes of {threads} threads ask for {at_once * threads} cores of {cores}: "
                          f"the solves share cores, so the table leaves their seconds empty", stacklevel=3)

        def finish(it, row):
            row["asked"] = asked
            if shared:
                row["solve_s"] = row["check_s"] = None
            if tab is not None:
                tab.write(row)
            if save is not None:
                it.answer.save(os.path.join(save, names[it.name] + ".json"))
            if report is not None:
                report.done(it, row)

        for it in items:                         # failed before any process was needed
            if it.answer is not None and not it.skip:
                finish(it, _row(it.answer, None, it.flux, it.name, it.quasr))
        _pool(todo, jobs, args, finish)
    finally:
        if tab is not None:
            tab.close()
    for it in items:
        if it.skip:          # already in the table: the answer saved for it, when the save folder holds it
            it.answer = None
            f = None if save is None else os.path.join(save, names[it.name] + ".json")
            if f is not None and os.path.exists(f):
                from .answer import Answer
                try:
                    it.answer = Answer.read(f)
                except (OSError, ValueError):
                    pass
    return [it.answer for it in items]


# --------------------------------------------------------------------------- the processes
class _Job:
    def __init__(self, it, proc, out, err, errfh, t0, overrun):
        self.it, self.proc, self.out, self.err, self.errfh, self.t0 = it, proc, out, err, errfh, t0
        self.overrun = overrun
        self.budget = None           # the budget the process stated, once it has
        self.stopped = False         # stopped by the batch, past the limit of its own watchdog


def _child_env():
    """This process's environment, with the folder of the torux it runs first on the path where a fresh
    interpreter would not find it by itself (a source tree, a folder added to sys.path by hand)."""
    import site
    env = dict(os.environ)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    usual = {os.path.abspath(p) for p in site.getsitepackages() + [site.getusersitepackages()]}
    if here not in usual:
        env["PYTHONPATH"] = here + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _pool(todo, jobs, args, finish):
    if not todo:
        return
    env = _child_env()
    with tempfile.TemporaryDirectory(prefix="torux-many-") as tmp:
        running = []
        queue = list(enumerate(todo))
        try:
            while queue or running:
                while queue and len(running) < jobs:
                    k, it = queue.pop(0)
                    running.append(_start(it, k, tmp, env, args))
                still = []
                for j in running:
                    if j.proc.poll() is None:
                        if not _overdue(j):
                            still.append(j)
                            continue
                        j.stopped = True
                        _stop([j.proc])
                    it = j.it
                    it.answer, row = _collect(j)
                    finish(it, row)
                running = still
                if running:
                    time.sleep(0.05)
        except BaseException:
            _stop([j.proc for j in running])
            raise
        finally:
            for j in running:
                j.errfh.close()


def _start(it, k, tmp, env, args):
    base = os.path.join(tmp, str(k))
    job = dict(source=it.payload, flux=it.flux, overrun=_OVERRUN, budget_file=base + ".budget", **args)
    with open(base + ".job", "wb") as fh:
        pickle.dump(job, fh, protocol=pickle.HIGHEST_PROTOCOL)
    errfh = open(base + ".err", "wb")
    # stdin stays open as a lifeline: when this process goes, the worker sees it close and stops
    proc = subprocess.Popen([sys.executable, "-c", _WORKER, base + ".job", base + ".out"], stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=errfh, env=env, close_fds=True)
    return _Job(it, proc, base + ".out", base + ".err", errfh, time.time(), _OVERRUN)


def _overdue(j):
    """Whether a process is past the limit of its own watchdog and still there: the watchdog writes every thread's
    stack before it ends the process, and on a busy machine that writing has been seen to hang."""
    if j.budget is None:
        try:
            with open(os.path.splitext(j.out)[0] + ".budget") as fh:
                j.budget = float(fh.read())
        except (OSError, ValueError):
            return False
    return time.time() - j.t0 > j.overrun[0] * j.budget + j.overrun[1] + _GRACE


def _stop(procs):
    """Stop every process: SIGTERM, and SIGKILL for any still there two seconds later. Then reap them all."""
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass
    t = time.time() + 2.0
    for p in procs:
        try:
            p.wait(timeout=max(0.0, t - time.time()))
        except subprocess.TimeoutExpired:
            pass
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        if p.stdin:
            p.stdin.close()


def _collect(j):
    """The answer and the row of a finished worker. A worker that left none gives an error answer saying why."""
    it = j.it
    wall = time.time() - j.t0
    rc = j.proc.returncode
    budget = None
    if j.proc.stdin:
        j.proc.stdin.close()
    j.errfh.close()
    if rc == 0 and os.path.exists(j.out):
        try:
            with open(j.out, "rb") as fh:
                out = pickle.load(fh)
            ans = out["answer"]
            if out.get("engine") is not None:
                ans._engine = _Engine(it.payload, out["engine"])
            row = out["row"]
            row.update(source=it.name, id=it.quasr)
            return ans, row
        except Exception as e:  # noqa: BLE001
            why = f"its answer could not be read back ({type(e).__name__}: {e})"
    else:
        try:
            with open(j.err, "rb") as fh:
                text = fh.read()
        except OSError:
            text = b""
        tail = text[-4000:].decode("utf-8", errors="replace")
        try:
            with open(os.path.splitext(j.out)[0] + ".budget") as fh:
                budget = float(fh.read())
        except (OSError, ValueError):
            budget = None
        if j.stopped or _watchdog(rc, text, wall, budget, j):
            why = (f"its process ran {wall:.0f} s, past {j.overrun[0]:g} times its budget" +
                   (f" of {budget:.0f} s" if budget else "") + f" and {j.overrun[1]:.0f} s more, and was stopped")
        elif rc is not None and rc < 0:
            import signal
            try:
                sig = signal.Signals(-rc).name
            except ValueError:
                sig = f"signal {-rc}"
            why = f"its process was stopped by {sig} after {wall:.0f} s"
        else:
            last = [ln for ln in tail.strip().splitlines() if ln.strip()]
            why = f"its process ended with exit code {rc} after {wall:.0f} s"
            if last:
                why += f": {last[-1].strip()}"
    ans = _error_answer("the solve failed: " + why, wall, budget)
    return ans, _row(ans, None, it.flux, it.name, it.quasr)


def _watchdog(rc, err, wall, budget, j):
    """Whether a worker was stopped by its own watchdog (faulthandler: exit code 1, its dump headed 'Timeout (',
    which on a process of many threads may lie anywhere in what it wrote), told from a crash on any platform: the
    exit code, the dump's header at the start of a line, and a wall time past the watchdog's limit."""
    if rc != 1:
        return False
    limit = None if not budget else j.overrun[0] * budget + j.overrun[1]
    headed = re.search(rb"(^|\n)Timeout \(\d+:\d\d:\d\d(\.\d+)?\)!", err) is not None
    return headed and (limit is None or wall >= 0.9 * limit)


class _Engine:
    """The field engine of an answer solved in another process, rebuilt from its source when first used."""

    def __init__(self, source, dmin):
        self._source, self._dmin, self._e = source, dmin, None

    def __getattr__(self, k):
        if k.startswith("__") or k in ("_source", "_dmin", "_e"):
            raise AttributeError(k)
        if self._e is None:
            from .field import FieldEngine
            self._e = FieldEngine(_load_payload(self._source), dmin=self._dmin)
        return getattr(self._e, k)

    def __getstate__(self):
        return dict(_source=self._source, _dmin=self._dmin, _e=None)

    def __setstate__(self, d):
        self.__dict__.update(d)


def _load_payload(s):
    from .source import Source
    from .sources import load
    if isinstance(s, Source):
        return s
    if isinstance(s, tuple):
        return load(*s)
    return load(s)


# --------------------------------------------------------------------------- one design, in its own process
def _solve(source, flux, **kw):
    from .answer import solve                    # the one call of the solve: torux.solve
    return solve(source, flux, **kw)


def _error_answer(reason, wall_s, budget_s):
    from .answer import Answer
    return Answer("error", float(budget_s) if budget_s else float("nan"), float(wall_s),
                  evidence=dict(error=" ".join(str(reason).split())))


def _work(job):
    """{'answer', 'row', 'engine'} of one job: its source loaded, solved, the answer stripped of its engine."""
    import faulthandler
    from .api import estimate_budget
    t0 = time.time()
    src, request, budget = None, job["flux"], None
    try:
        src = _load_payload(job["source"])
        sized = job.get("minor_radius") is not None or job.get("volume") is not None or job.get("largest")
        if request is None and not sized:
            request = (src.request or {}).get("flux_Wb")
        if job["budget_s"]:
            budget = float(job["budget_s"])
        elif request is None:              # a size, or the largest: torux.solve asks it by a search
            from .ask import stated_budget
            budget = stated_budget(src, "minor_radius" if job.get("minor_radius") is not None else
                                   "volume" if job.get("volume") is not None else "largest")
        else:
            budget = estimate_budget(src, request)
        if job.get("budget_file"):
            with open(job["budget_file"], "w") as fh:
                fh.write(repr(budget))
        over = job.get("overrun", _OVERRUN)
        faulthandler.dump_traceback_later(over[0] * budget + over[1], exit=True)
        res = _solve(src, job["flux"], minor_radius=job.get("minor_radius"), volume=job.get("volume"),
                     largest=bool(job.get("largest")), budget_s=job["budget_s"], tol=job["tol"],
                     verify=job["verify"], gradients=job["gradients"])
        faulthandler.cancel_dump_traceback_later()
    except Exception as e:  # noqa: BLE001
        res = _error_answer(f"{type(e).__name__}: {e}", time.time() - t0, budget or job["budget_s"])
    eng = res._engine
    dmin = None
    if eng is not None:
        dmin = float(((res.record or {}).get("engine") or {}).get("dmin", 0.05))
    res._engine = None
    return dict(answer=res, row=_row(res, src, request), engine=dmin)


def _row(res, src, request, name=None, quasr=None):
    """The table's row of one answer (results/README.md says what each column is)."""
    ev = res.evidence or {}
    st = res.status
    word = {"surface": ("surface", ""), "no_surface": ("no_surface", ""), "range": ("range", ""),
            "undetermined": ("undetermined", ""), "surface_unverified": ("undetermined", "torus_not_verified"),
            "range_unverified": ("undetermined", "torus_not_verified"), "no_axis": ("undetermined", "no_axis"),
            "timeout": ("undetermined", "timeout")}
    outcome, note = word.get(st, (st, ""))
    if st == "error":
        note = " ".join(str(ev.get("error", "")).split())
    row = dict.fromkeys(COLUMNS)
    row.update(source=name, id=quasr, nfp=None if src is None else int(src.nfp), outcome=outcome, note=note,
               flux_Wb=None if request is None else abs(float(request)))
    s = res.surface
    if st in ("surface", "range") and s is not None:
        ind = ev.get("independent") or {}
        sol = ev.get("solver") or {}
        row.update(iota=float(res.iota), M=int(s.M), N=int(s.N), bn_solver=sol.get("bn_off"),
                   bn_check=ind.get("bn_sup"), flux_err=ind.get("flux_rel_err"))
        if st == "range" or row["flux_Wb"] is None:      # a range, or a surface asked by size or the largest
            row["flux_Wb"] = abs(float(s.flux))
        if hasattr(res, "minor_radius"):
            row.update(minor_radius_m=res.minor_radius, volume_m3=res.volume)
    search = ev.get("search") or {}
    if row["flux_Wb"] is None and search.get("question") in ("minor_radius", "volume") and st != "error":
        # a size the coils were not shown to hold: the flux it would need, where the search came to one
        what = (f"minor radius {search['asked']:.9g} m" if search["question"] == "minor_radius"
                else f"volume {search['asked']:.9g} m^3")
        said = f"asked for {what}"
        if search.get("flux") is not None:
            row["flux_Wb"] = abs(float(search["flux"]))
            said += ": flux_Wb is the flux that size would need"
        row["note"] = (f"{note}, " if note else "") + said
    bound = [o.get("no_surface_above_Wb") for o in ((ev.get("search") or {}).get("outward") or [])
             if o.get("kind") == "certified bound"]
    if bound and bound[0] is not None:                   # the largest: no surface encloses more than this
        row["bound_Wb"] = float(bound[0])
    if st != "error":
        ind = ev.get("independent") or {}
        check = ind.get("wall") if isinstance(ind, dict) else None
        row["check_s"] = check
        row["solve_s"] = res.wall_s - (check or 0.0)
    if st == "no_surface":
        c = ev.get("certificate") or {}
        p = ev.get("proof") or {}
        if c:
            kind = str(c.get("kind", "")).replace("_manifold", "").replace("_pairs", "")
            row["certificate"] = kind or None
            iv = c.get("interval")
            if kind == "island_band" and iv:
                row["excluded_lo_Wb"], row["excluded_hi_Wb"] = float(iv[0]), float(iv[1])
            if kind == "edge_barrier" and c.get("F_up") is not None:
                row["bound_Wb"] = float(c["F_up"])
        elif p.get("verdict"):
            row["certificate"] = str(p["verdict"]).lower()
    return row


# --------------------------------------------------------------------------- the table and the saved answers
class _Table:
    """A CSV table written a row at a time, continued when it exists: a design whose row says 'error' is tried
    again, its old row taken out first, so that no design ever has two rows."""

    def __init__(self, path, items, asked=None):
        path = os.fspath(path)
        if not path.lower().endswith(".csv"):
            raise ValueError(f"{path}: the table is CSV, written a row at a time: give a name ending in .csv")
        self.path, self.done, self.retried, self.cols = path, set(), set(), COLUMNS
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        if not fresh:
            with open(path, "rb+") as fh:
                data = fh.read()
                if not data.endswith(b"\n"):
                    # the last row was cut off by an interruption: it goes, and its design is solved again
                    fh.truncate(data.rfind(b"\n") + 1)
            with open(path, newline="", encoding="utf-8") as fh:
                rd = csv.reader(fh)
                head = next(rd, None)
                rows = [r for r in rd if r]
            if head is None:
                fresh = True
            elif tuple(head) not in (COLUMNS, COLUMNS_SIZES, COLUMNS_QUASR):
                raise ValueError(f"{path} is not a table torux.many wrote (its columns differ): give another "
                                 f"name, or move it away")
            else:
                # a table written in fewer columns is continued in its own
                self.cols = tuple(head)
                k = self.cols.index("outcome")
                if "asked" in self.cols and asked is not None:
                    q = self.cols.index("asked")
                    other = sorted({r[q] for r in rows if len(r) > q and r[q] and r[k] != "error"} - {asked})
                    if other:
                        raise ValueError(f"{path} holds the answers to another question ({other[0]}, where this "
                                         f"run asks {'for a flux' if asked == 'flux' else 'for ' + asked}): a table "
                                         f"answers one question, so give this one another name")
                # the same for a flux: a design whose row answers another flux than this run asks of it
                f = self.cols.index("flux_Wb")
                want = {it.key: abs(it.flux) for it in items if it.flux is not None}
                for r in rows:
                    w = want.get(_key(r[0])) if len(r) > max(k, f) and r[k] != "error" and r[f] else None
                    if w is not None and abs(abs(float(r[f])) - w) > 1e-12 * w:
                        raise ValueError(f"{path} holds the answer for {r[0]} at {abs(float(r[f])):.12g} Wb, where "
                                         f"this run asks {w:.12g} Wb: a table answers one question, so give this "
                                         f"one another name")
                # a design of this run whose row says error is solved again, its old row taken out first: the rows
                # of every other design stay as they are
                mine = {it.key for it in items}
                again = [r for r in rows if len(r) > k and r[k] == "error" and _key(r[0]) in mine]
                keep = [r for r in rows if r not in again]
                self.done = {_key(r[0]) for r in keep if not (len(r) > k and r[k] == "error")}
                self.retried = {_key(r[0]) for r in again} - self.done
                if len(keep) < len(rows):
                    tmp = path + ".part"
                    with open(tmp, "w", newline="", encoding="utf-8") as fh:
                        w = csv.writer(fh)
                        w.writerow(self.cols)
                        w.writerows(keep)
                    os.replace(tmp, path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "w" if fresh else "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if fresh:
            self.w.writerow(self.cols)
            self.fh.flush()

    def write(self, row):
        self.w.writerow([_cell(row.get(c)) for c in self.cols])
        self.fh.flush()
        try:
            os.fsync(self.fh.fileno())
        except OSError:  # pragma: no cover
            pass

    def close(self):
        self.fh.close()


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ""
    return str(v)


def _save_names(items):
    """A file name for each answer: the source file's name without its suffix, or, where two files share a name,
    its path from the files' common folder. In letters a file system takes everywhere."""
    paths = [it.payload for it in items if isinstance(it.payload, str) and it.quasr is None]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    common = None
    if len(set(stems)) < len(stems):
        try:
            common = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in paths])
        except ValueError:          # different drives
            common = None
    out = {}
    for it in items:
        if it.quasr is not None:
            base = f"quasr_{it.quasr:07d}"
        elif isinstance(it.payload, str):
            p = os.path.abspath(it.payload)
            base = os.path.splitext(os.path.relpath(p, common) if common else os.path.basename(p))[0]
        elif isinstance(it.payload, tuple):
            base = "+".join(os.path.splitext(os.path.basename(p))[0] for p in it.payload)
        else:
            base = str(it.name)
        out[it.name] = re.sub(r"[^A-Za-z0-9.+=-]+", "_", base).strip("_.") or "source"
    seen = {}
    for k, v in out.items():
        seen.setdefault(v, []).append(k)
    clash = [ks for ks in seen.values() if len(ks) > 1]
    if clash:
        raise ValueError(f"{clash[0][0]!r} and {clash[0][1]!r} would be saved under one name: rename one")
    return out


# --------------------------------------------------------------------------- the command line
def add_parser(sub):
    """The 'many' command, on the torux command line's subcommands."""
    m = sub.add_parser("many", help="many designs at once, each in its own process, a row of a table each",
                       description="Each design solved as torux solve solves it, in its own process, a few at a "
                                   "time, a row of the table each as it finishes.  The same command again continues "
                                   "the table and solves only the designs not yet in it.")
    m.add_argument("files", nargs="+", metavar="SOURCE", help="the designs: coil or magnet files, or quasr:NUMBER")
    m.add_argument("--table", required=True, metavar="FILE", help="a .csv file, one row a design as it finishes. A "
                                                                  "table that exists is continued")
    ask = m.add_mutually_exclusive_group()
    ask.add_argument("--flux", type=float, metavar="WB", help="in every design, the surface enclosing this full "
                                                              "toroidal flux, in Wb (default: each design's own "
                                                              "request, or its largest surface)")
    ask.add_argument("--minor-radius", type=float, metavar="M", help="in every design, the surface of this minor "
                                                                     "radius, in m")
    ask.add_argument("--volume", type=float, metavar="M3", help="in every design, the surface enclosing this volume, "
                                                                "in m^3")
    ask.add_argument("--largest", action="store_true", help="in every design, the largest surface torux can verify")
    m.add_argument("--budget", type=float, metavar="S", help="seconds for each design's whole question (default: "
                                                             "stated from its size)")
    m.add_argument("--jobs", type=int, metavar="N", help="designs at once (default: the cores divided by "
                                                         "TORUX_THREADS)")
    m.add_argument("--save", metavar="DIR", help="a folder where each answer is also written as JSON")
    return m


class _Report:
    def __init__(self, out):
        self.out, self.counts, self.width = out, {}, 12

    def starting(self, n, at_once):
        """On stderr, so that the table's lines stay what stdout carries: a design's line comes when it ends."""
        print(f"torux: {n} design{'s' if n != 1 else ''} to solve, {at_once} at a time: each is a line below when "
              f"it ends", file=sys.stderr, flush=True)

    def skipped(self, k, n, table, again=0):
        if k or again:
            more = f", and {again} whose row says error {'is' if again == 1 else 'are'} tried again" if again else ""
            print(f"{k} of {n} designs are already in {table}{more}", file=self.out, flush=True)

    def done(self, it, row):
        o = row["outcome"]
        self.counts[o] = self.counts.get(o, 0) + 1
        t = it.answer.wall_s
        extra = f"   {row['note']}" if row.get("note") else ""
        print(f"{o:<{self.width}}  {it.name}   {t:.1f} s{extra}", file=self.out, flush=True)


def main(a, out=None):
    """Run 'torux many' from its parsed arguments. 0 once every design has its row."""
    out = out or sys.stdout
    rep = _Report(out)
    try:
        _run(a.files, a.flux, a.jobs, a.table, a.save, a.budget, 3e-15, True, False, rep,
             dict(minor_radius=a.minor_radius, volume=a.volume, largest=a.largest))
    except KeyboardInterrupt:
        print("stopped: every design not in the table yet is solved when the same command runs again",
              file=sys.stderr)
        return 130
    except (ValueError, OSError) as e:
        print(f"error   {e}", file=sys.stderr)
        return 1
    if rep.counts:
        print("  ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(rep.counts.items())), file=out)
    return 0


def _lifeline(fd):
    try:
        while os.read(fd, 65536):
            pass
    except OSError:
        pass
    os._exit(3)


def _worker_main(job_path, out_path):
    import signal
    import threading
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl-C is the batch's to handle: it stops this process
    threading.Thread(target=_lifeline, args=(sys.stdin.fileno(),), daemon=True).start()
    with open(job_path, "rb") as fh:
        job = pickle.load(fh)
    out = _work(job)
    try:
        data = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:  # noqa: BLE001
        out["answer"].record = None
        out["answer"].evidence["record"] = f"the solver's record could not be sent back ({type(e).__name__})"
        data = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + ".part", "wb") as fh:
        fh.write(data)
    os.replace(out_path + ".part", out_path)
    sys.stderr.flush()
    os._exit(0)                  # nothing is left to finalise, and the lifeline thread is still reading


if __name__ == "__main__":
    _worker_main(*sys.argv[1:3])
