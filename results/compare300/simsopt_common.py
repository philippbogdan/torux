r"""What the three SIMSOPT harnesses share (run_boozer_exact.py, run_boozer_ls.py, run_qfm.py).

Each reads inputs/quasr_<id>.json beside this file (inputs/pilot/ for the six tuning designs) and writes
DIR/answers/quasr_<id>.json and DIR/logs/quasr_<id>.log.  Here: the design's coils as SIMSOPT coils, the coil
quadrature rule, exact changes of resolution, the first values of G and iota, SIMSOPT's own off-grid residual, the
resolution ladder, and the process that keeps the budget and writes the answer.

The budget is 3,600 s of wall time, 12 GiB of resident memory and 4 threads per design.  The parent process runs
the solver in a child process, which stops itself at 3,600 s; the parent stops it at 12 GiB resident (or past the
budget's grace) and writes the answer from the child's last checkpoint, so a run cut short by time or memory still
gets its answer file.  The child writes a checkpoint after every resolution level: the best candidate the solver
itself declared converged so far, chosen by the solver's own diagnostics.  Nothing from the evaluator is used to
choose an answer; ../evaluator/forms.py only converts SIMSOPT's coefficients to the evaluator's form, exactly.

Everything here uses SIMSOPT 1.11.1 as released.
"""
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"                  # the 300 sample designs; inputs/pilot/ holds the six tuning designs
# the evaluator's directory (forms.py): ../evaluator/ beside this folder, or the directory this variable names
EVALUATOR = Path(os.environ.get("COMPARE300_EVALUATOR") or HERE.parent / "evaluator").resolve()
BUDGET_S = 3600.0                         # wall time per design
CORES = 4                                 # threads per design
MEM_GB = 12.0                             # resident memory of the solver process, GiB (12 x 2^30 bytes)

# exactly 4 threads per design: SIMSOPT's C++ (OpenMP), numpy/scipy's OpenBLAS, and anything else that might
# start threads (MKL, numexpr, numba, XLA)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = str(CORES)
os.environ["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=true "
                           f"intra_op_parallelism_threads={CORES}")
os.environ["JAX_PLATFORMS"] = "cpu"              # SIMSOPT imports jax; nothing here runs on a card
# How idle threads wait (each solver keeps its library default unless the other measures better under the meter).
# The parent sets SIMSOPT_HARNESS_OMP_WAIT from the run's `omp_wait` setting before starting a solver process:
# "default" leaves OpenMP and OpenBLAS to their own (spinning) defaults, "passive" makes idle threads sleep.
WAIT_VARS = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OPENBLAS_THREAD_TIMEOUT")
if os.environ.get("SIMSOPT_HARNESS_OMP_WAIT", "default") == "passive":
    os.environ.update(OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0", OPENBLAS_THREAD_TIMEOUT="4")
else:
    for _k in WAIT_VARS:
        os.environ.pop(_k, None)

try:
    import numpy as np  # noqa: E402
except ImportError:                       # main() says so in one line; --help works without it
    np = None

sys.path.insert(0, str(EVALUATOR))
try:
    import forms  # noqa: E402  (the evaluator's exact converters)
except ImportError:
    forms = None
import fair_cores  # noqa: E402  (the timing rule's meter)

GRACE_S = 120.0                           # the solver stops itself at the budget; the parent kills only past this

# ------------------------------------------------------------------------------------------------ inputs, coils

def input_path(design, pilot=False):
    return (INPUTS / "pilot" if pilot else INPUTS) / f"quasr_{design}.json"


def load_design(design, pilot=False):
    return json.loads(input_path(design, pilot).read_text())


def simsopt_coils(rec, nq):
    """The design's coil coefficients, as given, as SIMSOPT CurveXYZFourier coils with nq quadrature points
    each."""
    from simsopt.field import Coil, Current
    from simsopt.geo import CurveXYZFourier
    out = []
    for c in rec["coils"]:
        K = len(c["xc"]) - 1
        cu = CurveXYZFourier(nq, K)
        x = []
        for a in "xyz":                    # SIMSOPT's order per axis: c0, s1, c1, s2, c2, ...
            x.append(c[a + "c"][0])
            for k in range(1, K + 1):
                x += [c[a + "s"][k], c[a + "c"][k]]
        cu.x = np.array(x)
        out.append(Coil(cu, Current(c["current_A"])))
    return out


def coil_quadrature(rec, pts, ladder=(96, 128, 160, 192, 256, 320, 400, 512, 640, 800, 1024, 1280, 1600, 2048),
                    accept=1e-12, factor=2):
    """The coil-field quadrature, chosen per design from SIMSOPT's own BiotSavart at the points it will solve on.

    The trapezoid rule on a smooth closed coil converges geometrically.  N0 is the first rule whose field agrees
    with the doubled rule to `accept` (max |B(N) - B(2N)| / |B| at pts); the rule used is factor * N0, whose error
    is then about the square of that, far under double-precision rounding (B's own rounding floor is ~5e-15)."""
    from simsopt.field import BiotSavart
    table = []
    cache = {}

    def B(n):
        if n not in cache:
            bs = BiotSavart(simsopt_coils(rec, n))
            bs.set_points(np.ascontiguousarray(pts))
            cache[n] = bs.B().copy()
        return cache[n]

    n0 = ladder[-1]
    for n in ladder:
        d = float(np.max(np.linalg.norm(B(n) - B(2 * n), axis=1) / np.linalg.norm(B(2 * n), axis=1)))
        table.append([n, d])
        cache.pop(n, None)
        if d <= accept:
            n0 = n
            break
    return int(factor * n0), table


# ------------------------------------------------------------------------------------------------ surfaces

def xyz_surface(nfp, stellsym, mpol, ntor, qp, qt, dofs=None):
    from simsopt.geo import SurfaceXYZTensorFourier
    kw = dict(nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor, quadpoints_phi=np.asarray(qp, float),
              quadpoints_theta=np.asarray(qt, float))
    if dofs is not None:
        kw["dofs"] = dofs
    return SurfaceXYZTensorFourier(**kw)


def start_xyz(rec, qp, qt):
    st = rec["start"]["surface"]
    s = xyz_surface(st["nfp"], st["stellsym"], st["mpol"], st["ntor"], qp, qt)
    s.x = np.array(st["x"])
    return s


def lift_xyz(s, mpol, ntor, qp, qt):
    """The same surface at resolution (mpol, ntor) >= s's, exactly: SIMSOPT's tensor coefficients zero-padded
    (index i <= mpol is cos(i theta), above it sin((i - mpol) theta); likewise j for phi)."""
    assert mpol >= s.mpol and ntor >= s.ntor
    new = xyz_surface(s.nfp, s.stellsym, mpol, ntor, qp, qt)
    for name in ("xcs", "ycs", "zcs"):
        A = np.asarray(getattr(s, name)).copy()
        B = getattr(new, name)
        B[:, :] = 0.0
        for i in range(2 * s.mpol + 1):
            ii = i if i <= s.mpol else i - s.mpol + mpol
            for j in range(2 * s.ntor + 1):
                jj = j if j <= s.ntor else j - s.ntor + ntor
                B[ii, jj] = A[i, j]
    new.x = np.asarray(new.get_dofs()).copy()
    return new


def same_surface_err(a, b, n=17, seed=3):
    """max |gamma_a - gamma_b| at random points (both SurfaceXYZTensorFourier): checks a lift is exact."""
    rng = np.random.default_rng(seed)
    qp, qt = np.sort(rng.random(n)), np.sort(rng.random(n))
    ea = xyz_surface(a.nfp, a.stellsym, a.mpol, a.ntor, qp, qt)
    ea.x = a.x
    eb = xyz_surface(b.nfp, b.stellsym, b.mpol, b.ntor, qp, qt)
    eb.x = b.x
    return float(np.max(np.abs(ea.gamma() - eb.gamma())))


def boozer_estimates(s, bs):
    """G and iota (with their signs) that make s closest to a Boozer surface of the field, in least squares:
    G B / |B|^2 = x_phi + iota x_theta (SIMSOPT's angles in [0, 1), so G is 2 pi times the radian G)."""
    pts = s.gamma().reshape(-1, 3)
    bs.set_points(np.ascontiguousarray(pts))
    B = bs.B()
    xp = s.gammadash1().reshape(-1, 3)
    xt = s.gammadash2().reshape(-1, 3)
    b2 = np.sum(B * B, 1)[:, None]
    A = np.stack([(B / b2).ravel(), -xt.ravel()], 1)
    (G, iota), *_ = np.linalg.lstsq(A, xp.ravel(), rcond=None)
    return float(G), float(iota)


# ------------------------------------------------------------------------------------------------ provenance

def versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {"python": platform.python_version()}
    for p in ("simsopt", "numpy", "scipy", "jax"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            pass
    try:
        cfg = np.show_config(mode="dicts")
        out["numpy_blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:  # noqa: BLE001
        pass
    return out


def host():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        if sys.platform == "darwin":
            try:
                cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            except (OSError, subprocess.SubprocessError):
                pass
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return dict(cpu=cpu, cpus=affinity, threads_used=CORES,
                threads={k: os.environ.get(k) for k in THREAD_VARS + ("XLA_FLAGS",) + WAIT_VARS},
                cpus_in_machine=os.cpu_count(), gpu=None,
                mem_limit_GB=float(os.environ.get("SIMSOPT_HARNESS_MEM_GB", MEM_GB)))


def maxrss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def rss_gb(pid):
    """Resident memory of a running process, GiB (the unit of maxrss_gb): /proc on Linux, ps elsewhere."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 ** 2
        return None
    except FileNotFoundError:
        if sys.platform.startswith("linux"):
            return None
    except OSError:
        return None
    try:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()) / 1024 ** 2
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj) + "\n")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------ answers

def xyz_answer_parts(s, extra_native=None):
    """surface (exact xyz_fourier), check points (SIMSOPT's own gamma) and native coefficients of a
    SurfaceXYZTensorFourier."""
    surf = forms.from_simsopt_xyz_tensor(s)
    cp = forms.simsopt_check_points(s, k=32)
    native = dict(kind="SurfaceXYZTensorFourier", nfp=int(s.nfp), stellsym=bool(s.stellsym), mpol=int(s.mpol),
                  ntor=int(s.ntor), x=np.asarray(s.x).tolist(), dof_names=list(s.local_dof_names),
                  angles="theta = 2 pi quadpoints_theta, phi = 2 pi quadpoints_phi (SIMSOPT's [0, 1) parameters)")
    native.update(extra_native or {})
    return surf, cp, native


def rz_answer_parts(s, extra_native=None, k=32, seed=7):
    """SurfaceRZFourier -> rz_fourier exactly (rc, zs copied, n times nfp), with check points from SIMSOPT."""
    from simsopt.geo import SurfaceRZFourier
    m, n, rc, rs, zc, zs = [], [], [], [], [], []
    for mm in range(s.mpol + 1):
        for nn in range(-s.ntor, s.ntor + 1):
            if mm == 0 and nn < 0:
                continue
            m.append(mm)
            n.append(nn * s.nfp)
            rc.append(float(s.rc[mm, nn + s.ntor]))
            zs.append(float(s.zs[mm, nn + s.ntor]))
            if not s.stellsym:
                rs.append(float(s.rs[mm, nn + s.ntor]))
                zc.append(float(s.zc[mm, nn + s.ntor]))
    surf = dict(form="rz_fourier", m=m, n=n, rc=rc, zs=zs)
    if not s.stellsym:
        surf.update(rs=rs, zc=zc)
    else:
        surf.update(rs=[0.0] * len(m), zc=[0.0] * len(m))
    rng = np.random.default_rng(seed)
    u, v = rng.random(k), rng.random(k)
    cp = dict(theta=[], phi=[], xyz=[])
    for a, b in zip(u, v):
        e = SurfaceRZFourier(nfp=s.nfp, stellsym=s.stellsym, mpol=s.mpol, ntor=s.ntor,
                             quadpoints_phi=[b], quadpoints_theta=[a])
        e.x = s.x
        cp["theta"].append(2 * np.pi * a)
        cp["phi"].append(2 * np.pi * b)
        cp["xyz"].append(e.gamma()[0, 0].tolist())
    native = dict(kind="SurfaceRZFourier", nfp=int(s.nfp), stellsym=bool(s.stellsym), mpol=int(s.mpol),
                  ntor=int(s.ntor), rc=np.asarray(s.rc).tolist(), zs=np.asarray(s.zs).tolist(),
                  x=np.asarray(s.x).tolist(), dof_names=list(s.local_dof_names),
                  layout="rc[m, n + ntor], zs[m, n + ntor]: R = sum rc cos(m theta - n nfp phi), "
                         "Z = sum zs sin(m theta - n nfp phi); theta = 2 pi quadpoints_theta, phi = 2 pi "
                         "quadpoints_phi is the cylindrical angle")
    if not s.stellsym:
        native.update(rs=np.asarray(s.rs).tolist(), zc=np.asarray(s.zc).tolist())
    native.update(extra_native or {})
    return surf, cp, native


# ------------------------------------------------------------------------------------------------ diagnostics

def offgrid_xyz(s, nd):
    """A SurfaceXYZTensorFourier sharing s's dofs on an nd x nd grid over one field period, offset from every
    collocation and quadrature grid the solvers use (irrational shifts)."""
    return xyz_surface(s.nfp, s.stellsym, s.mpol, s.ntor, (np.arange(nd) + 0.318309886) / nd / s.nfp,
                       (np.arange(nd) + 0.577215665) / nd, dofs=s.dofs)


def boozer_offgrid(s, iota, G, bs_d, nd):
    """SIMSOPT's own Boozer residual G B - |B|^2 (x_phi + iota x_theta) relative to |G| |B|, at points the solve
    never saw; its normal part is |B.n|/|B|, also reported."""
    from simsopt.geo.surfaceobjectives import boozer_surface_residual
    sd = offgrid_xyz(s, nd)
    rr = boozer_surface_residual(sd, iota, G, bs_d, derivatives=0)[0].reshape(nd, nd, 3)
    B = bs_d.B().reshape(nd, nd, 3)
    nB = np.linalg.norm(B, axis=2)
    rel = np.linalg.norm(rr, axis=2) / (abs(G) * nB)
    nrm = sd.normal()
    bn = np.abs(np.sum(B * nrm, 2)) / (nB * np.linalg.norm(nrm, axis=2))
    return dict(diag_boozer_rel_max=float(rel.max()), diag_boozer_rel_rms=float(np.sqrt(np.mean(rel ** 2))),
                diag_bn_max=float(bn.max()), diag_bn_rms=float(np.sqrt(np.mean(bn ** 2))), diag_points=nd * nd)


# ------------------------------------------------------------------------------------------------ the ladder

class Ladder:
    """A resolution ladder under the budget's clock and memory.

    Before a level: skip it (and stop) if its predicted peak memory passes the cap or its predicted time (the last
    level's time scaled by the cube of the unknowns, the dense solves' cost, once a level of 1,000 unknowns or more
    takes 60 s or more) would not fit the budget.
    After a level: a converged level becomes the answer if its diagnostic (the solver's own measure of its answer
    at points it did not solve on) is the smallest so far; the ladder stops after `stop_after_no_gain` converged
    levels in a row that do not halve the best diagnostic, or `stop_after_failures` unconverged levels in a row."""

    def __init__(self, S, t0, budget, state, ckpt, save):
        self.S, self.t0, self.budget, self.state, self.ckpt, self.save = S, t0, budget, state, ckpt, save
        self.best = np.inf
        self.no_gain = self.failures = 0
        self.last = None

    def fits(self, M, mem_gb, nunk):
        el = time.time() - self.t0
        if mem_gb > self.S["mem_cap_gb"]:
            self.state["stop"] = f"level M={M} needs about {mem_gb:.1f} GB (cap {self.S['mem_cap_gb']} GB)"
            return False
        if self.last is not None and self.last[1] >= self.S.get("predict_min_s", 60.0) and \
                self.last[0] >= self.S.get("predict_min_unknowns", 1000):
            # only a level that took a while is dominated by the dense work that scales; cheaper levels are mostly
            # fixed costs (imports, field set-up, BFGS) and would predict nonsense; the parent's kill is the backstop
            n0, t0, it0 = self.last
            pred = t0 * (nunk / n0) ** 3 * (max(it0, 3) + 2) / (it0 + 2)
            if el + pred > self.budget - self.S["margin_s"]:
                self.state["stop"] = f"level M={M} predicted {pred:.0f} s with {self.budget - el:.0f} s left"
                return False
        return True

    def ran(self, nunk, seconds, iterations):
        self.last = (nunk, seconds, max(1, iterations))

    def converged(self, value, make_best):
        """Returns True when this level became the answer."""
        self.failures = 0
        if value < self.best:
            self.no_gain = 0 if value < 0.5 * self.best else self.no_gain + 1
            self.best = value
            self.state["best"] = make_best()
            return True
        self.no_gain += 1
        return False

    def failed(self):
        self.failures += 1

    def more(self):
        if self.failures >= self.S["stop_after_failures"]:
            self.state["stop"] = f"no convergence at {self.failures} levels in a row"
            return False
        if self.no_gain >= self.S["stop_after_no_gain"]:
            self.state["stop"] = f"{self.no_gain} converged levels in a row without halving the diagnostic"
            return False
        return True

    def finish(self, fail_status):
        st = self.state
        st.setdefault("stop", "ladder complete")
        if st["best"] is None:
            fails = [h for h in st["history"] if not h.get("success")]
            errs = [h for h in fails if "error" in h]
            st["solver_status"] = fail_status if fails else "no level ran"
            if not fails:
                st["reason"] = "out of time"
            elif errs and len(errs) == len(fails):
                st["reason"] = "error: " + errs[-1]["error"]
            else:
                st["reason"] = "not converged"
            st["reason"] += f" ({st['stop']})"
        else:
            st["best"]["solver_status"] += f"; ladder: {st['stop']}"
        log("stop:", st["stop"])
        self.save()


def short(h):
    return {k: (round(v, 3) if isinstance(v, float) and abs(v) > 1e-3 else v) for k, v in h.items()}



def timing_definition(budget, mem_gb):
    return (f"wall_s is the time from the start of the solver's work on the design (before its input file is read) "
            f"to its answer (the solver's last checkpoint written), measured inside one fresh process per design "
            f"that had already imported SIMSOPT and finished one untimed warm-up solve on a pilot design; exactly "
            f"{CORES} threads (OpenMP, OpenBLAS, MKL, numba, XLA); the warm-up and the timed run each under "
            f"fair_cores.Meter where Linux provides its scheduler accounting (host.cores_warmup, host.cores); the "
            f"solver stops itself at the {budget:,.0f} s budget on this clock, and the harness stops it at "
            f"{mem_gb:g} GiB of resident memory")


RULE = ("the timed run counts only if its threads waited for a CPU at most 5% of the time they ran and other work "
        "used its CPUs for at most 5% of that time (fair_cores.fair)")


class TimeUp(BaseException):
    """Raised by the budget alarm inside the solver process (BaseException: the ladders' `except Exception` do not
    catch it), so the solver stops itself at the budget and its meter can still be read."""


def _time_up(signum, frame):
    raise TimeUp()


def _ended(p):
    """True once the solver process has ended: on Linux without reaping it (waitid with WNOWAIT), so that the
    backstop meter can still be read; elsewhere (no waitid, and nothing to meter) Popen.poll."""
    if hasattr(os, "waitid"):
        try:
            return os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return True
    return p.poll() is not None


def meter_available():
    return fair_cores is not None and os.path.exists(f"/proc/{os.getpid()}/task/{os.getpid()}/schedstat")


def check_environment():
    """One line and exit if what the solve needs is missing (the help text needs none of it)."""
    req = HERE / "requirements" / "simsopt.txt"
    if np is None:
        sys.exit(f"numpy is not installed in this Python: pip install -r {req}")
    if forms is None:
        sys.exit(f"no evaluator/forms.py in {EVALUATOR}: set COMPARE300_EVALUATOR to the evaluator's directory")
    import importlib.util
    if importlib.util.find_spec("simsopt") is None:
        sys.exit(f"SIMSOPT is not installed in this Python: pip install -r {req} (the runs used simsopt==1.11.1)")
    from importlib.metadata import version
    if version("simsopt") != "1.11.1":
        print(f"note: SIMSOPT {version('simsopt')} here; the comparison ran SIMSOPT 1.11.1", flush=True)


def main(key, label, solve, default_settings, describe, warmup=None):
    """Command line shared by the three harnesses.

        python run_<solver>.py DESIGN [--pilot] [--out DIR] [--budget S] [--mem-gb G] [--set name=value ...]
                               [--no-warmup] [--force]

    Writes DIR/answers/quasr_<id>.json and DIR/logs/quasr_<id>.log (DIR defaults to compare300-out/<key> under
    the current directory).  An existing answer is kept unless --force.
    warmup: dict(design=<pilot id>, overrides={...}) for the untimed warm-up solve."""
    import argparse
    ap = argparse.ArgumentParser(description=describe, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", nargs="?", help="the 7-digit QUASR id: inputs/quasr_<id>.json")
    ap.add_argument("--pilot", action="store_true", help="read inputs/pilot/ (the six tuning designs)")
    ap.add_argument("--set", action="append", default=[], help="override a setting: name=value (JSON value)")
    ap.add_argument("--out", default=str(Path.cwd() / "compare300-out" / key),
                    help=f"output directory (default: compare300-out/{key} under the current directory)")
    ap.add_argument("--budget", type=float, default=BUDGET_S, help="wall-time budget of the timed solve, s")
    ap.add_argument("--mem-gb", type=float, default=MEM_GB, help="the solver process's resident-memory limit, GiB")
    ap.add_argument("--force", action="store_true", help="replace an existing answer")
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up solve (quick checks only)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    settings = dict(default_settings)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in settings:
            raise SystemExit(f"unknown setting {k}; known: {sorted(settings)}")
        settings[k] = json.loads(v)
    if a.child:                                   # the solver itself, in its own process
        child(json.loads(Path(a.child).read_text()), solve)
        return
    if not a.design:
        ap.error("DESIGN is required")
    check_environment()
    os.environ["SIMSOPT_HARNESS_MEM_GB"] = str(a.mem_gb)
    run_one(key, label, a.design, a.pilot, settings, Path(a.out), a.budget, a.force,
            warmup=None if a.no_warmup else warmup, mem_gb=a.mem_gb)


def child(job, solve):
    """One design in one process: the untimed warm-up solve on a pilot design, then the timed design, each under
    fair_cores.Meter where Linux provides it.  The timed run stops itself at the budget."""
    metered = meter_available()
    warm = cores_w = None
    if job.get("warmup"):
        w = job["warmup"]
        ws = dict(job["settings"])
        ws.update(w.get("overrides", {}))
        ws["save_levels"] = False
        tw = time.time()
        log(f"warm-up: pilot {w['design']} with {w.get('overrides')} (untimed)")
        m = fair_cores.Meter().start() if metered else None
        try:
            solve(load_design(w["design"], True), ws, Path(job["ckpt"]).with_suffix(".warmup.json"), tw,
                  w.get("budget", 900.0))
            warm = dict(design=w["design"], overrides=w.get("overrides", {}), seconds=time.time() - tw)
        except Exception as e:  # noqa: BLE001
            warm = dict(design=w["design"], error=repr(e)[:300], seconds=time.time() - tw)
        cores_w = m.stop() if m else None
        log("warm-up done", warm, "cores", cores_w)
        print("=" * 100, flush=True)
    m = fair_cores.Meter().start() if metered else None
    t_start = time.time()
    atomic_write(job["start"], dict(t_start=t_start, warmup=warm, cores_warmup=cores_w))
    signal.signal(signal.SIGALRM, _time_up)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, job["budget"]))
    out_of_time = False
    try:
        try:
            rec = load_design(job["design"], job["pilot"])      # the timed part starts before the inputs are read
            solve(rec, job["settings"], Path(job["ckpt"]), t_start, job["budget"])
        except TimeUp:
            out_of_time = True
            log("the budget's alarm stopped the solver")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    t_end = time.time()
    cores = m.stop() if m else None
    atomic_write(job["done"], dict(t_end=t_end, cores=cores, out_of_time=out_of_time))


def run_one(key, label, design, pilot, settings, out, budget, force=False, warmup=None, warmup_timeout=600.0,
            mem_gb=MEM_GB):
    """Runs the design once and writes its answer; returns "ok" or "skipped"."""
    ans_path = out / "answers" / f"quasr_{design}.json"
    log_path = out / "logs" / f"quasr_{design}.log"
    if ans_path.exists() and not force:
        print(f"{ans_path} exists; skipped (--force replaces it)", flush=True)
        return "skipped"
    rec = load_design(design, pilot)
    work = out / "work"
    ckpt = work / f"quasr_{design}.ckpt.json"
    start_f, done_f = work / f"quasr_{design}.start.json", work / f"quasr_{design}.done.json"
    for p in (ckpt, ckpt.with_suffix(".json.tmp"), start_f, done_f, ckpt.with_suffix(".warmup.json")):
        if p.exists():
            p.unlink()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    job = dict(design=design, pilot=pilot, settings=settings, ckpt=str(ckpt), budget=budget, warmup=warmup,
               start=str(start_f), done=str(done_f))
    jobf = work / f"quasr_{design}.job.json"
    jobf.write_text(json.dumps(job))
    cmd = [sys.executable, "-u", str(Path(sys.argv[0]).resolve()), "--child", str(jobf)]
    killed = None
    t_start = t_end = parent_meter = None
    rss_peak = 0.0
    how = None
    with open(log_path, "w") as lf:
        lf.write(f"# {key} design {design}  host {json.dumps(host())}\n"
                 f"# settings {json.dumps(settings)}\n")
        lf.flush()
        env = dict(os.environ, SIMSOPT_HARNESS_OMP_WAIT=str(settings.get("omp_wait", "default")))
        t_launch = time.time()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        st = None
        while p.poll() is None and time.time() - t_launch < warmup_timeout:
            if start_f.exists():
                try:
                    st = json.loads(start_f.read_text())
                    break
                except ValueError:
                    pass
            time.sleep(0.5)
        if st is None and start_f.exists():
            st = json.loads(start_f.read_text())
        if st is None:                 # the warm-up did not finish: nothing was timed
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            rc = p.wait()
            how = (f"the untimed warm-up did not finish (solver process exit code {rc} after "
                   f"{time.time() - t_launch:.0f} s): nothing timed")
            with open(log_path, "a") as lf2:
                lf2.write(f"# {how}\n")
        else:
            t_start = st["t_start"]
            # a second meter, in this process over the solver process (the Meter covers child processes and keeps a
            # thread's last reading after it ends): the record if the solver process is killed or dies before its own
            pm = fair_cores.Meter().start() if meter_available() else None
            # the solver stops itself at the budget; this process stops it past the budget's grace (the backstop)
            # or when its resident memory passes the limit, reading it five times a second.  The backstop meter
            # is read while the solver process is still alive or an unreaped zombie: once reaped, its whole life
            # (warm-up included) would enter this process's getrusage and hide foreign work on the cpus
            deadline = t_start + budget + GRACE_S
            while True:
                if _ended(p):
                    break
                r = rss_gb(p.pid)
                if r is not None:
                    rss_peak = max(rss_peak, r)
                    if r > mem_gb:
                        killed = "memory"
                        break
                if time.time() > deadline:
                    killed = "killed past the budget's grace"
                    break
                time.sleep(0.2)
            parent_meter = pm.stop() if pm else None
            if killed:
                os.killpg(p.pid, signal.SIGKILL)
            rc = p.wait()
        t_stop = time.time()
    done = json.loads(done_f.read_text()) if done_f.exists() else {}
    t_end = done.get("t_end")
    meter = done.get("cores")
    meter_source = "the solver process's own Meter over its timed run" if meter else None
    if meter is None and t_start is not None and parent_meter is not None:
        meter = parent_meter
        meter_source = "the harness's Meter over the solver process (the solver process did not stop its own)"
    wall = None if t_start is None else (t_end if t_end is not None else t_stop) - t_start
    warm = st.get("warmup") if st else None
    cores_w = st.get("cores_warmup") if st else None
    ck = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    best = ck.get("best")
    hist = ck.get("history", [])
    if how is not None:
        pass
    elif done.get("out_of_time") or killed == "killed past the budget's grace":
        how = f"stopped at the {budget:.0f} s budget"
    elif killed == "memory":
        lvl = ""
        if hist:
            lvl = f" during the level after mpol=ntor={hist[-1].get('M')}"
        how = f"stopped by the harness at the {mem_gb:g} GiB memory limit{lvl}"
    elif rc != 0:
        how = f"solver process exited with code {rc}" + (" (killed: memory?)" if rc in (-9, 137) else "")
    else:
        how = None
    hostd = dict(ck.get("host") or host())
    hostd.update(cores_warmup=cores_w, cores=meter, cores_source=meter_source,
                 rss_peak_GB_seen_by_harness=round(rss_peak, 3) if rss_peak else None)
    is_fair = None if meter is None else bool(fair_cores.fair(meter))
    if meter is not None:
        fc = dict(sha256=fair_cores.SHA256, metered=True, fair=is_fair, rule=RULE,
                  wait_fraction=meter["wait_fraction"], foreign_fraction=meter.get("foreign_fraction"),
                  warmup_fair=None if cores_w is None else bool(fair_cores.fair(cores_w)))
    else:
        fc = dict(sha256=fair_cores.SHA256, metered=False, fair=None, rule=RULE,
                  why="nothing was metered: the meter reads Linux's per-thread scheduler accounting "
                      "(/proc/<pid>/task/<tid>/schedstat), which this machine does not have"
                  if not meter_available() else "nothing was metered: the timed run did not start")
    ans = dict(design=design, solver=key, solver_label=label, versions=ck.get("versions") or versions(),
               settings=settings, host=hostd, requested_flux_Wb=rec["requested_toroidal_flux_Wb"],
               wall_s=wall, timing=dict(definition=timing_definition(budget, mem_gb), warmup=warm,
                                        t_start_unix=t_start),
               fair_cores=fc, speed=ck.get("speed"), history=hist)
    if best is not None:
        ans.update(status="surface", solver_status=best["solver_status"] + (f"; then {how}" if how else ""),
                   reason=None, surface=best["surface"], check_points=best["check_points"],
                   solver_reported=best["solver_reported"], native=best["native"])
    else:
        if how and how.startswith("stopped at"):
            reason = f"out of time: {how}"
        elif killed == "memory":
            reason = f"out of memory: {how}"
            if ck.get("reason"):
                reason += f"; {ck['reason']}"
        else:
            reason = ck.get("reason") or f"error: {how or 'no checkpoint'}"
            if how and ck.get("reason"):
                reason += f"; {how}"
        ans.update(status="none", solver_status=ck.get("solver_status") or how or "no result", reason=reason,
                   surface=None, check_points=None, solver_reported=ck.get("solver_reported"), native=None)
    atomic_write(ans_path, ans)
    wf, ff = fc.get("wait_fraction"), fc.get("foreign_fraction")
    with open(log_path, "a") as lf:
        lf.write(f"# answer: status {ans['status']}, wall {wall if wall is None else round(wall, 1)} s, "
                 f"wait_fraction {wf}, foreign_fraction {ff}, counts under the timing rule {is_fair}, "
                 f"solver_status {ans['solver_status']!r}, reason {ans['reason']!r}\n")
    for q in (ckpt, jobf, start_f, done_f, ckpt.with_suffix(".warmup.json")):
        if q.exists():
            q.unlink()
    try:
        work.rmdir()
    except OSError:
        pass
    print(f"quasr_{design}: {ans['status']} ({wall if wall is None else round(wall)} s, "
          f"wait {wf if wf is None else round(wf, 3)}, foreign {ff if ff is None else round(ff, 3)}, "
          f"counts under the timing rule: {'not metered' if is_fair is None else is_fair}) "
          f"{ans['solver_status'][:120]} -> {ans_path}", flush=True)
    return "ok"
