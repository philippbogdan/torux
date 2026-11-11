"""VMEC++ 0.9.3 (Proxima Fusion's reimplementation of VMEC, `vmecpp==0.9.3`, released and unmodified) on one design
of the comparison: the free-boundary equilibrium of the design's coils in vacuum (no pressure, no current) at the
requested toroidal flux, whose boundary is the magnetic surface enclosing that flux, written as the design's answer.

    python run_vmecpp.py 0003051                       # inputs/quasr_0003051.json -> compare300-out/vmecpp
    python run_vmecpp.py 1101401 --pilot --settings my_settings.json

Flow: this process imports VMEC++, runs an untimed warm-up solve on pilot 1101401 (twice: below), then solves the
design under the wall-time budget and writes the answer.

What it does:

1. The coils' field.  VMEC++ 0.9.3 takes the external field of a free-boundary run as a function of position
   (`vmecpp.MagneticFieldFunction`), evaluated at the plasma boundary points in every vacuum update with no grid and
   no interpolation.  It is the Biot-Savart integral of the input's Fourier coils (vmecpp_field.py), its trapezoid
   rule doubled until the field on and around the start boundary no longer changes (1e-14 of |B|), evaluated by a
   single-threaded nogil kernel inside VMEC++'s own 4 vacuum threads (no other thread computes).
2. The start.  QUASR's VMEC namelist of its stored boundary.  Where VMEC++ does not converge from it, the starts of
   the settings' "start_order" in turn (below).
3. phiedge.  The requested flux, signed so that VMEC++'s toroidal field has the direction of the coils' field
   (VMEC++ checks this itself: rbtor and bsubvvac must have the same sign).
4. A ladder of resolutions (settings "levels"), each a full VMEC++ run hot-restarted from the previous level's
   solution interpolated to the new resolution (`vmecpp.interpolate_solution`, VMEC++'s own continuation).  Each
   level converges or not by VMEC++'s own criterion (ier_flag 0 with fsqr, fsqz, fsql under ftol).  The answer is
   the highest level that converged; a level that does not converge ends the ladder.
5. The answer: the last flux surface's rmnc, zmns (cos/sin(m theta - n zeta), n already times nfp, zeta the
   cylindrical angle) written as the `rz_fourier` form, plus 32 check points that VMEC++'s own geometry
   (`vmecpp.geometry.evaluate` on the converged state) puts at s = 1.

Settings and why (vmecpp_settings.json).  Chosen on the six pilot designs (0224040, 0925611, 1101401, 1981226,
2126458, 2559727; none of them among the 300), scored by |B.n|/|B| at the evaluator's 4,000 points (max/median in
float64).  VMEC++'s own documentation (README, docs/tutorials/free_boundary.md, docs/tutorials/debugging_failures.md,
examples/ of the v0.9.3 tag) gives the starting points; the numbers below are pilot measurements.

- coils' field: `vmecpp.MagneticFieldFunction`, the Biot-Savart sum of the input's Fourier coils, N nodes per coil
  doubled until the field on the start boundary and on copies scaled 0.85 and 1.15 about it changes by under 1e-14
  of |B| (128 to 512 nodes on the pilots).  It is evaluated by a single-threaded `nogil` numba kernel that VMEC++'s
  4 vacuum threads run side by side (it agrees with SIMSOPT's kernel to 3e-16).  SIMSOPT's OpenMP kernel called
  from VMEC++'s vacuum threads would start a thread team per calling thread (5.2 busy cores measured on a pilot,
  more than the 4 allowed).  VMEC++ 0.9.3 evaluates the function at the boundary points in every vacuum update: no
  grid, no interpolation, no coil discretisation error; the field agrees with the evaluator's to 1.3e-15.  The mgrid
  route was measured too (an exact-field response table, spacing min(a/50, 2.5 mm), VMEC++'s 4-point
  interpolation): identical boundaries on pilot 1101401 (8.95e-6 against 8.89e-6, same iteration counts), so a grid
  only buys speed; the function costs 5 to 30% of the run time and was kept.
- free-boundary method: NESTOR (VMEC++'s default); `only_coils` (VMEC++'s coils-only vacuum pressure, valid in
  vacuum) as a fallback: in the start list, and for any level that fails, which is retried once from the same
  converged state with the other method (the design keeps the method that converged).  On 0925611 NESTOR gave
  3.3e-5 against only_coils 4.5e-5 at the same resolution; on pilot 2126458 NESTOR converged only level 0 (1.0e-2)
  where only_coils reached ns 101 (5.9e-4).  `BIEST` is listed in 0.9.3's enum but not implemented.
- phiedge: the requested flux, signed as the coils' mean B_phi on the start boundary: VMEC++'s toroidal field then
  points along the coils' (it stops with "rbtor and bsubvvac must have the same sign" otherwise).
- profiles: am = 0, pres_scale = 0, ncurr = 1, ac = 0, curtor = 0: vacuum, no pressure, no toroidal current (as
  VMEC++'s W7-X vacuum example).
- start: QUASR's namelist boundary; if level 0 does not converge from it, a fixed-boundary VMEC++ equilibrium inside
  it and free boundary from that; then the same inside the namelist boundary cut to m <= 3 and shrunk 0.8 about its
  m = 0 curve, then m <= 2 shrunk 0.6; then the namelist and the m <= 3 start again with `only_coils`.  QUASR's
  namelist is a fit with its own poloidal angle and carries slowly decaying high-m content on many designs (m >= 5
  amplitude above 1% of the minor radius on 107 of the 300); VMEC++ then cannot build a valid initial Jacobian
  (pilot 0224040 fails from it at every resolution, even from VMEC++'s own three-surface retry, and SIMSOPT's
  `condense_spectrum`, the remedy the docs name, did not help).  A smoothed, shrunk boundary clears coils that hug
  the plasma, and a fixed-boundary equilibrium inside it gives the free-boundary run a valid interior: 0224040 then
  converges.  `only_coils` converged pilot 2126458 where NESTOR did not.
- ladder: L0 ns 16, mpol 11, ntor 10, ftol 1e-10; L1 ns 51 (11, 10), 1e-11; L2 ns 101 (13, 12), 1e-11; L3 ns 201
  (15, 14), 1e-11; L4 to L10 ns 201 (15, 14) with ftol 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18.  Each level
  hot-restarts from the previous one.  mpol 11, ntor 10 hold every mode of the namelist (m <= 10, |n| <= 10).  The
  boundary error falls with ns up to 201 and not after (0925611 at mpol 13, ftol 1e-14: 3.3e-5 at ns 101, 2.5e-5 at
  151, 2.3e-5 at 201, 2.4e-5 at 301) and with ftol down to at least 1e-16 (3.3e-5, 2.7e-5, 2.6e-5 for 1e-14, 1e-15,
  1e-16 at ns 101; along the ladder at ns 201, 1101401 went 6.0e-5, 3.6e-5, 1.4e-5 for 1e-14, 1e-15, 1e-16 and
  then 4.7e-6 (median 8.1e-7) at 1e-17, so 1e-17 and 1e-18 are in the ladder: VMEC++'s documentation recommends
  final tolerances below 1e-16, "values as low as 1e-19 can be beneficial"; whether the last levels fit in 3,600 s
  depends on the design).  More Fourier modes than 15, 14 changed nothing (17, 16: 3.08e-5 against 3.08e-5).  Some
  designs stall at a residual of 1e-12 to 1e-11 (2559727 at ns 51 never reached 1e-12), so ns is raised at the
  loosest tolerance first and the tolerance tightened after: a stall then costs only the levels above it.
- nzeta: 36 for every level: at least 2 ntor + 4 = 32 at the top level; 48 and 72 changed nothing.
- ntheta, vacuum_mpol, vacuum_ntor: VMEC++'s defaults (2 mpol + 6; the plasma's mpol, ntor): doubling ntheta changed
  the error by 4%, raising the vacuum potential's cutoff to 20, 16 by 4%.
- nvacskip, delt, iteration style: VMEC++'s defaults (1, which it lengthens itself as the run converges; 1.0; VMEC
  8.52).  nstep 500 (VMEC++'s output interval) and niter per level as in the settings file.
- threads: 4 (`max_threads`); the field kernel runs in those 4 threads.
- answer: the highest level that converged (VMEC++'s ier_flag 0 with fsqr, fsqz, fsql all under ftol); a level that
  does not converge ends the ladder; past the budget (less 30 s to write) the iteration callback stops VMEC++.

Timing.  Each design runs in its own process, which first imports VMEC++ and solves pilot 1101401 with a two-level
miniature ladder (untimed warm-up: the first calls into VMEC++, the field kernel and JAX).  The warm-up is that
miniature solve twice: a first pass makes the process's first calls (modules' first use, numba's and XLA's
compilation: about 20 s, mostly on one thread), then the warm-up proper; both are metered.  `wall_s` then runs from
the start of the design's work, before its input file is read, to its answer (the chosen level's surface and its
check points): the coil-field quadrature, every VMEC++ start attempt and every level are inside it.  The 3,600 s
budget is counted on the same clock; a SIGALRM 120 s past it is the last resort.  Most designs use the whole budget:
the ladder keeps tightening until it.

Threads.  VMEC++ `max_threads = 4`, and the field inside VMEC++ is computed by those 4 threads themselves;
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS = 4 before any library loads (SIMSOPT's kernel for the
quadrature test runs before VMEC++ starts); XLA's Eigen pool off (`--xla_cpu_multi_thread_eigen=false`).  OpenMP
waiting is libgomp's default (no OMP_WAIT_POLICY): passive waiting doubled VMEC++'s median time on shared cores.
The comparison ran each process on 8 cpus of its own for these 4 threads.

Memory.  VMEC++'s peak at the top level (ns 201, mpol 15, ntor 14, 56 coils) measured 0.63 GB with Python and JAX;
nothing limits it here, and every answer records `host.peak_rss_mb`.

The timing rule.  Each timed run of the comparison counted only if its threads waited for a CPU at most 5% of the
time they ran and other work used its CPUs for at most 5% of that time.  On Linux the warm-up passes and the design
each run under `fair_cores.Meter` (fair_cores.py beside this file; readings in `host.cores_warmup_first_pass`,
`host.cores_warmup` and `host.cores`, its sha256 in `fair_cores_sha256`), and `fair_cores` and `timing_rule_met`
say whether this timed run would have counted.  The meter only measures.  Elsewhere nothing is metered and the
answer says so.
"""
import os

# Exactly 4 threads whatever cores the process is given.  VMEC++ gets max_threads = 4; every other thread pool in
# the process is capped before its library loads: the coil-field kernel (simsoptpp, OpenMP, for the quadrature
# test), BLAS, and XLA (VMEC++'s geometry, used for the check points, is JAX).
_THREADS = os.environ.get("VMECPP_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()

import sys  # noqa: E402

# the cpus this process was given (children inherit them); recorded in host.cpus
HOST_CPUS = dict(cpus=sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None

import argparse
import contextlib
import json
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EVALUATOR = Path(os.environ.get("COMPARE300_EVALUATOR") or HERE.parent / "evaluator")
TIMING_RULE = ("a timed run counts only if its threads waited for a CPU at most 5% of the time they ran and other "
               "work used its CPUs for at most 5% of that time (or 0.2 s in all)")

_MISSING = None
try:
    import numpy as np
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import simsopt  # noqa: E402
    import vmecpp  # noqa: E402
    from vmecpp import geometry as vgeom  # noqa: E402
    from vmecpp.cpp import _vmecpp  # noqa: E402
    from vmecpp_field import CoilField, converge_nodes  # noqa: E402
except ImportError as _e:
    _MISSING = _e

import fair_cores  # noqa: E402  the timing rule's meter (fair_cores.py beside this file)

SOLVER, LABEL = "vmecpp", "VMEC++"
N_CHECK, CHECK_SEED = 32, 20261107

# The settings every design runs with (chosen on the pilot designs; the docstring gives each reason).
DEFAULT_SETTINGS = json.loads((HERE / "vmecpp_settings.json").read_text())


# ------------------------------------------------------------------------------------------------------------
# inputs

def parse_namelist(nml):
    """QUASR's VMEC namelist -> {(m, n): [rbc, zbs]} with RBC(n, m), ZBS(n, m) as VMEC writes them."""
    modes = {}
    for key, idx in (("RBC", 0), ("ZBS", 1)):
        for n, m, v in re.findall(key + r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*([-+0-9.eEdD]+)", nml):
            modes.setdefault((int(m), int(n)), [0.0, 0.0])[idx] = float(v.replace("d", "e").replace("D", "e"))
    return modes


def modes_to_simsopt(modes, nfp):
    from simsopt.geo import SurfaceRZFourier
    M = max(m for m, _ in modes)
    N = max(abs(n) for _, n in modes)
    s = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=M, ntor=N)
    for (m, n), (r, z) in modes.items():
        if m == 0 and n < 0:
            continue
        s.set_rc(m, n, r)
        if m > 0 or n > 0:
            s.set_zs(m, n, z)
    return s


def simsopt_to_arrays(s, mpol, ntor):
    """SurfaceRZFourier -> VMEC++ rbc, zbs of shape (mpol, 2 ntor + 1), truncated or zero-padded."""
    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    for m in range(min(mpol - 1, s.mpol) + 1):
        for n in range(-min(ntor, s.ntor), min(ntor, s.ntor) + 1):
            if m == 0 and n < 0:
                continue
            rbc[m, n + ntor] = s.get_rc(m, n)
            zbs[m, n + ntor] = s.get_zs(m, n)
    return rbc, zbs


def surface_points(s, ntheta=64, nphi=64, scale=1.0):
    """Cartesian points of a SurfaceRZFourier over one field period, optionally scaled about each cross-section's
    centre of (R, Z) (scale > 1 moves outward)."""
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / s.nfp, nphi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    R = np.zeros_like(TH)
    Z = np.zeros_like(TH)
    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            if m == 0 and n < 0:
                continue
            a = m * TH - n * s.nfp * PH
            R += s.get_rc(m, n) * np.cos(a)
            Z += s.get_zs(m, n) * np.sin(a)
    Rc = R.mean(axis=0, keepdims=True)
    Zc = Z.mean(axis=0, keepdims=True)
    R = Rc + scale * (R - Rc)
    Z = Zc + scale * (Z - Zc)
    return np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1).reshape(-1, 3)


# ------------------------------------------------------------------------------------------------------------
# one VMEC++ run, as vmecpp.run does it for a magnetic-field function, keeping the C++ output for the geometry

class Stop(Exception):
    pass


def run_step(inp, field, restart_from, threads, deadline, log, every=500):
    """One VMEC++ run.  Returns (VmecOutput or None, C++ output or None, info dict)."""
    cpp_indata = inp._to_cpp_vmecindata()
    initial_state = None
    if restart_from is not None:
        initial_state = _vmecpp.HotRestartState(wout=restart_from.wout._to_cpp_wout(),
                                                indata=restart_from.input._to_cpp_vmecindata())
    info = dict(stopped_for_time=False, last=None, method=str(getattr(inp.free_boundary_method, "value",
                                                                       inp.free_boundary_method)))

    def callback(state):
        if state.iteration % every == 0:
            info["last"] = dict(iteration=int(state.iteration), ns=int(state.ns), fsqr=float(state.fsqr),
                                fsqz=float(state.fsqz), fsql=float(state.fsql))
            print(f"  iter {state.iteration:6d} ns {state.ns:4d} fsqr {state.fsqr:.3e} fsqz {state.fsqz:.3e} "
                  f"fsql {state.fsql:.3e} vac {bool(state.vacuum_pressure_active)} "
                  f"t {time.time() - T0:7.1f}s", file=log, flush=True)
        if time.time() > deadline:
            info["stopped_for_time"] = True
            return False
        return True

    t = time.time()
    try:
        if field is None:
            cpp_out = _vmecpp.run(cpp_indata, initial_state=initial_state, max_threads=threads,
                                  verbose=vmecpp.OutputMode.LEGACY.value, iteration_callback=callback)
        elif isinstance(field, vmecpp.MagneticFieldResponseTable):
            cpp_indata.mgrid_file = "NONE"
            cpp_out = _vmecpp.run(cpp_indata, magnetic_response_table=field._to_cpp_magnetic_field_response_table(),
                                  initial_state=initial_state, max_threads=threads,
                                  verbose=vmecpp.OutputMode.LEGACY.value, iteration_callback=callback)
        else:
            cpp_indata.mgrid_file = "NONE"
            cpp_out = _vmecpp.run(cpp_indata, magnetic_field=field, initial_state=initial_state,
                                  max_threads=threads, verbose=vmecpp.OutputMode.LEGACY.value,
                                  iteration_callback=callback)
    except Stop:
        raise
    except Exception as e:  # noqa: BLE001  VMEC++ reports its failures as exceptions
        info.update(error=str(e).strip(), wall_s=time.time() - t)
        return None, None, info
    info["wall_s"] = time.time() - t
    wout = vmecpp.VmecWOut._from_cpp_wout(cpp_out.wout)
    out = vmecpp.VmecOutput(input=inp, wout=wout, **vmecpp._output_tables_from_cpp(cpp_out))
    return out, cpp_out, info


def converged(wout):
    ok_flag = int(wout.ier_flag) in (0, 11)
    tol = float(wout.ftolv)
    ok_res = max(float(wout.fsqr), float(wout.fsqz), float(wout.fsql)) <= tol
    return ok_flag and ok_res


def level_input(base, lv):
    step = base.model_copy(deep=True)
    for k, v in lv.items():
        if k in ("name",):
            continue
        if k in ("ns", "ftol", "niter"):
            continue
        setattr(step, k, v)
    step.ns_array = np.asarray([lv["ns"]], dtype=np.int64)
    step.ftol_array = np.asarray([lv["ftol"]], dtype=float)
    step.niter_array = np.asarray([lv["niter"]], dtype=np.int64)
    return step


# ------------------------------------------------------------------------------------------------------------
# the answer

def check_points_from_library(cpp_out, nfp, k=N_CHECK, seed=CHECK_SEED):
    """k points VMEC++'s own geometry evaluates at s = 1, (theta, zeta) uniform on the torus."""
    g = vgeom.make(cpp_out)
    rng = np.random.default_rng(seed)
    th, ze = 2 * np.pi * rng.random(k), 2 * np.pi * rng.random(k)
    out = dict(theta=[], phi=[], xyz=[])
    with vmecpp.enable_x64(True):
        for a, b in zip(th, ze):
            v = np.asarray(vgeom.evaluate(g, jnp.asarray([1.0, a, b], dtype=jnp.float64)))
            R, Z = float(v[0, 0]), float(v[1, 0])
            out["theta"].append(float(a))
            out["phi"].append(float(b))
            out["xyz"].append([R * np.cos(b), R * np.sin(b), Z])
    return out


def own_eval(surface, pts):
    """The answer's rz_fourier series summed directly (a check of the conversion, independent of forms.py)."""
    m = np.asarray(surface["m"], float)
    n = np.asarray(surface["n"], float)
    worst = 0.0
    for a, b, x in zip(pts["theta"], pts["phi"], pts["xyz"]):
        ang = m * a - n * b
        R = np.dot(surface["rc"], np.cos(ang)) + np.dot(surface["rs"], np.sin(ang))
        Z = np.dot(surface["zc"], np.cos(ang)) + np.dot(surface["zs"], np.sin(ang))
        worst = max(worst, float(np.max(np.abs(np.array([R * np.cos(b), R * np.sin(b), Z]) - np.asarray(x)))))
    return worst


def host_info():
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True).stdout.strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:  # noqa: BLE001
        pass
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        load = [round(v, 2) for v in os.getloadavg()]
    except OSError:
        load = None
    return dict(cpu=cpu, cores_requested=4, threads_used=THREADS, cores_in_affinity_mask=cores,
                logical_cpus=os.cpu_count(), load_average_at_start=load, gpu=None)


def versions():
    import scipy
    return dict(vmecpp=_pkg_version("vmecpp"), simsopt=simsopt.__version__, numpy=np.__version__,
                scipy=scipy.__version__, jax=jax.__version__, python=platform.python_version())


def _pkg_version(name):
    from importlib.metadata import version
    return version(name)


# ------------------------------------------------------------------------------------------------------------

def solve(design, src, settings, budget, log):
    t0 = time.time()      # the design's clock starts here, before its input is read
    deadline = t0 + budget - settings.get("write_margin_s", 30)
    rec = json.loads((Path(src) / f"quasr_{design}.json").read_text())
    nfp = int(rec["nfp"])
    flux = float(rec["requested_toroidal_flux_Wb"])
    assert rec["stellarator_symmetric"]
    res = dict(design=design, solver=SOLVER, solver_label=LABEL, versions=versions(), settings=settings,
               host=host_info(), requested_flux_Wb=flux)
    print(f"design {design} nfp {nfp} coils {len(rec['coils'])} flux {flux!r}", file=log, flush=True)

    # 1. start boundary
    modes = parse_namelist(rec["start"]["vmec_namelist"])
    start = modes_to_simsopt(modes, nfp)

    # 2. the coils' field, its quadrature converged around the start boundary
    test = np.concatenate([surface_points(start, 48, 48, sc) for sc in settings["field_test_scales"]])
    nq, hist = converge_nodes(rec["coils"], test, tol=settings["field_tol"])
    field = CoilField(rec["coils"], nq)
    print(f"field nodes per coil {nq}; doubling history {hist}", file=log, flush=True)
    bphi = np.mean(field.cyl(*_cyl(surface_points(start, 16, 16)))[1])
    vac_field = field.cyl
    fs = settings.get("field", {"kind": "function"})
    if fs["kind"] == "grid":
        vac_field = response_table(field, start, nfp, fs, log)
    sign = float(np.sign(bphi)) * settings["phiedge_sign_vs_bphi"]
    phiedge = sign * flux
    print(f"mean B_phi on start {bphi:+.4e} T -> phiedge {phiedge!r}", file=log, flush=True)

    # 3. levels
    lv0 = settings["levels"][0]
    base = vmecpp.VmecInput(
        lasym=False, nfp=nfp, mpol=lv0["mpol"], ntor=lv0["ntor"],
        ns_array=np.array([lv0["ns"]]), ftol_array=np.array([lv0["ftol"]]), niter_array=np.array([lv0["niter"]]),
        phiedge=phiedge, ncurr=1, pmass_type="power_series", am=np.array([0.0]), pres_scale=0.0, gamma=0.0,
        pcurr_type="power_series", ac=np.array([0.0]), curtor=0.0, lfreeb=True, mgrid_file="NONE",
        extcur=np.array([1.0] if settings.get("field", {}).get("kind") == "grid" else []), raxis_c=np.zeros(lv0["ntor"] + 1), zaxis_s=np.zeros(lv0["ntor"] + 1),
        rbc=np.zeros((lv0["mpol"], 2 * lv0["ntor"] + 1)), zbs=np.zeros((lv0["mpol"], 2 * lv0["ntor"] + 1)),
        return_outputs_even_if_not_converged=True, **settings.get("common", {}))

    levels, best, start_attempts, level_surfaces = [], None, [], []
    prev = None
    for li, lv in enumerate(settings["levels"]):
        step = level_input(base, lv)
        if prev is None:
            # first level: from the start boundary as it is, then (if VMEC++ does not converge from it) from
            # smoother truncations of it
            for name in settings["start_order"]:
                # "namelist": QUASR's boundary as it is; "_m<K>": its modes m <= K only, a smoother start for a
                # boundary VMEC++ cannot begin from; "_s<f>": every m >= 1 mode times f, the cross-sections
                # shrunk about their m = 0 centre, a start clear of coils that hug the plasma (the free
                # boundary then finds its own shape and size at the requested flux)
                # "fixed+<start>": first a fixed-boundary VMEC++ equilibrium inside that boundary, from which the
                # free-boundary run then hot-restarts (an interior already in force balance)
                # "...@only_coils": the same start with VMEC++'s other free-boundary method for vacuum, kept for every
                # later level of the design if the start converges with it
                spec, _, method = name.partition("@")
                fixed_first = spec.startswith("fixed+")
                bname = spec.split("+", 1)[1] if fixed_first else spec
                step.free_boundary_method = method or base.free_boundary_method
                parts = dict((t[0], t[1:]) for t in bname.split("_")[1:])
                step.rbc, step.zbs = simsopt_to_arrays(start, lv["mpol"], lv["ntor"])
                if "m" in parts:
                    kmax = int(parts["m"])
                    if kmax >= lv["mpol"] - 1:
                        continue
                    step.rbc[kmax + 1:] = 0.0
                    step.zbs[kmax + 1:] = 0.0
                if "s" in parts:
                    step.rbc[1:] *= float(parts["s"])
                    step.zbs[1:] *= float(parts["s"])
                step.raxis_c = np.zeros(lv["ntor"] + 1)
                step.zaxis_s = np.zeros(lv["ntor"] + 1)
                print(f"level {li} {lv} from the {name} start", file=log, flush=True)
                guess = None
                if fixed_first:
                    fx = step.model_copy(deep=True)
                    fx.lfreeb = False
                    fx.ftol_array = np.asarray([settings.get("fixed_ftol", 1e-12)])
                    fout, _, finfo = run_step(fx, None, None, THREADS, deadline, log)
                    fok = fout is not None and converged(fout.wout)
                    print(f"  fixed-boundary start: {'converged' if fok else 'not converged'} "
                          f"({finfo.get('error', '')[:200]})", file=log, flush=True)
                    if not fok:
                        start_attempts.append(dict(level=li, start=name, converged=False,
                                                   solver_status="fixed-boundary start did not converge"))
                        continue
                    guess = fout
                out, cpp, info = run_step(step, vac_field, guess, THREADS, deadline, log)
                info["start"] = name
                rec_try = _level_record(li, lv, out, info, field)
                start_attempts.append(rec_try)
                if (out is not None and converged(out.wout)) or info.get("stopped_for_time"):
                    break
            base_boundary = (step.rbc, step.zbs)
            method_used = step.free_boundary_method
            res["start_attempts"] = start_attempts
        else:
            step.rbc = vmecpp.VmecInput.resize_2d_coeff(base_boundary[0], mpol_new=lv["mpol"], ntor_new=lv["ntor"])
            step.zbs = vmecpp.VmecInput.resize_2d_coeff(base_boundary[1], mpol_new=lv["mpol"], ntor_new=lv["ntor"])
            step.raxis_c = np.zeros(lv["ntor"] + 1)
            step.zaxis_s = np.zeros(lv["ntor"] + 1)
            step.free_boundary_method = method_used
            guess = vmecpp.interpolate_solution(prev, step)
            print(f"level {li} {lv} from level {li - 1}", file=log, flush=True)
            out, cpp, info = run_step(step, vac_field, guess, THREADS, deadline, log)
            other = settings.get("method_fallback", {}).get(str(getattr(method_used, "value", method_used)))
            if other and not (out is not None and converged(out.wout)) and not info.get("stopped_for_time"):
                # the level failed with this free-boundary method: the same level, from the same converged state,
                # with VMEC++'s other vacuum method; if that converges the design keeps it from here on
                levels.append(_level_record(li, lv, out, info, field))
                print(f"level {li} -> {levels[-1]}", file=log, flush=True)
                print(f"level {li} {lv} from level {li - 1} with {other}", file=log, flush=True)
                step.free_boundary_method = other
                guess = vmecpp.interpolate_solution(prev, step)
                out, cpp, info = run_step(step, vac_field, guess, THREADS, deadline, log)
                if out is not None and converged(out.wout):
                    method_used = step.free_boundary_method
        ok = out is not None and converged(out.wout)
        levels.append(_level_record(li, lv, out, info, field))
        print(f"level {li} -> {levels[-1]}", file=log, flush=True)
        if not ok:
            break
        prev = out
        best = (li, lv, out, cpp)
        best_rec = levels[-1]
        w = out.wout
        level_surfaces.append(dict(level=li, m=[int(v) for v in w.xm], n=[int(v) for v in w.xn],
                                   rmnc=[float(v) for v in w.rmnc[:, -1]], zmns=[float(v) for v in w.zmns[:, -1]]))
        if time.time() > deadline:
            break

    res["levels"] = levels
    res["level_surfaces"] = level_surfaces      # every converged level's boundary (the answer is the last)
    res["field"] = dict(kind="vmecpp.MagneticFieldFunction (Biot-Savart of the Fourier coils, trapezoid rule)",
                        nodes_per_coil=nq, doubling_history=hist, calls=field.calls, points=field.points, seconds=field.seconds)
    res["wall_s"] = time.time() - t0
    if best is None:
        last = levels[-1] if levels else {}
        res.update(status="none", surface=None, check_points=None, native=None,
                   solver_status=last.get("solver_status", "no level ran"),
                   reason=_reason(last), solver_reported=last.get("reported"))
        return res
    li, lv, out, cpp = best
    w = out.wout
    surface = dict(form="rz_fourier", m=[int(v) for v in w.xm], n=[int(v) for v in w.xn],
                   rc=[float(v) for v in w.rmnc[:, -1]], rs=[0.0] * len(w.xm), zc=[0.0] * len(w.xm),
                   zs=[float(v) for v in w.zmns[:, -1]])
    pts = check_points_from_library(cpp, nfp)
    res.update(status="surface", reason=None,
               solver_status=best_rec["solver_status"] + f"; answer from level {li} of {len(settings['levels']) - 1}"
               + f" ({best_rec['method']})",
               surface=surface, check_points=pts, solver_reported=dict(best_rec["reported"], method=best_rec["method"]),
               native=dict(kind="VMEC++ wout, last flux surface (s = 1): R = sum rmnc cos(xm theta - xn zeta), "
                                "Z = sum zmns sin(xm theta - xn zeta), zeta the cylindrical angle",
                           nfp=nfp, ns=int(w.ns), mpol=int(w.mpol), ntor=int(w.ntor), signgs=int(w.signgs),
                           xm=[int(v) for v in w.xm], xn=[int(v) for v in w.xn],
                           rmnc=[float(v) for v in w.rmnc[:, -1]], zmns=[float(v) for v in w.zmns[:, -1]],
                           lmns=[float(v) for v in w.lmns[:, -1]],
                           raxis_cc=[float(v) for v in w.raxis_cc], zaxis_cs=[float(v) for v in w.zaxis_cs],
                           phiedge=float(out.input.phiedge)))
    res["wall_s"] = time.time() - t0     # to the answer: the chosen level's surface and its check points
    res["check_points_own_eval_max_m"] = own_eval(surface, pts)
    print(f"answer level {li}; check points (own sum of the written series) {res['check_points_own_eval_max_m']:.2e} m",
          file=log, flush=True)
    return res


def response_table(field, start, nfp, fs, log):
    """The coils' exact field on a cylindrical grid (one circuit of unit current, extcur = [1]) for VMEC++'s
    bicubic interpolation: spacing fs["h"] (m), the start boundary's box widened by fs["margin"] (m),
    fs["nphi"] planes per field period (a multiple of every level's nzeta)."""
    t = time.time()
    x = surface_points(start, 128, 64)
    R, Z = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
    a = 0.25 * ((R.max() - R.min()) + (Z.max() - Z.min()))        # a rough minor radius
    h = min(fs["h_over_a"] * a, fs["h_max"])
    margin = fs["margin_over_a"] * a
    fs = dict(fs, h=h, margin=margin)
    r0, r1 = R.min() - margin, R.max() + margin
    z0, z1 = Z.min() - margin, Z.max() + margin
    nr = int(np.ceil((r1 - r0) / h)) + 1
    nz = int(np.ceil((z1 - z0) / h)) + 1
    nphi = int(fs["nphi"])
    rg = np.linspace(r0, r1, nr)
    zg = np.linspace(z0, z1, nz)
    pg = 2 * np.pi * np.arange(nphi) / (nfp * nphi)
    P, Zm, Rm = np.meshgrid(pg, zg, rg, indexing="ij")          # phi slowest, then z, then r
    br, bp, bz = field.cyl(Rm.ravel(), P.ravel(), Zm.ravel())
    params = vmecpp.MakegridParameters(normalize_by_currents=False, assume_stellarator_symmetry=False,
                                       number_of_field_periods=nfp, r_grid_minimum=r0, r_grid_maximum=r1,
                                       number_of_r_grid_points=nr, z_grid_minimum=z0, z_grid_maximum=z1,
                                       number_of_z_grid_points=nz, number_of_phi_grid_points=nphi)
    tab = vmecpp.MagneticFieldResponseTable(parameters=params, b_r=br[None, :], b_p=bp[None, :], b_z=bz[None, :])
    print(f"field grid {nr} x {nz} x {nphi} (h {h:.2e} m, margin {margin:.2e} m) in {time.time() - t:.1f}s",
          file=log, flush=True)
    return tab


def _cyl(x):
    return np.hypot(x[:, 0], x[:, 1]), np.arctan2(x[:, 1], x[:, 0]), x[:, 2]


def _level_record(li, lv, out, info, field):
    r = dict(level=li, settings=lv, method=info.get("method"), wall_s=info.get("wall_s"), start=info.get("start"),
             stopped_for_time=info.get("stopped_for_time"), field_calls=field.calls)
    if out is None:
        r.update(converged=False, solver_status="error: " + info.get("error", "")[:2000], reported=info.get("last"))
        return r
    w = out.wout
    conv = converged(w)
    r["converged"] = conv
    r["solver_status"] = (f"ier_flag {int(w.ier_flag)} ({w.reason}); fsqr {float(w.fsqr):.3e}, fsqz {float(w.fsqz):.3e}, "
                          f"fsql {float(w.fsql):.3e} against ftolv {float(w.ftolv):.1e} -> "
                          + ("converged" if conv else "not converged")
                          + f" in {int(w.itfsq)} iterations (ns {int(w.ns)}, mpol {int(w.mpol)}, ntor {int(w.ntor)})")
    r["reported"] = dict(ier_flag=int(w.ier_flag), fsqr=float(w.fsqr), fsqz=float(w.fsqz), fsql=float(w.fsql),
                         ftolv=float(w.ftolv), itfsq=int(w.itfsq), ns=int(w.ns), mpol=int(w.mpol), ntor=int(w.ntor),
                         iota_axis=float(w.iotaf[0]), iota_edge=float(w.iotaf[-1]),
                         toroidal_flux_edge_Wb=float(w.phi[-1]), phiedge=float(out.input.phiedge),
                         volume_m3=float(w.volume_p), aspect=float(w.aspect), Rmajor_m=float(w.Rmajor_p),
                         Aminor_m=float(w.Aminor_p), ctor_A=float(w.ctor), rbtor=float(w.rbtor),
                         betatotal=float(w.betatotal))
    return r


def _reason(last):
    if not last:
        return "error: no VMEC++ run"
    if last.get("stopped_for_time"):
        return "out of time"
    st = last.get("solver_status", "")
    if st.startswith("error"):
        return "error: " + st[7:][:300]
    return "not converged: " + st


THREADS = int(_THREADS)
T0 = time.time()
WARMUP_DESIGN, WARMUP_SRC = "1101401", HERE / "inputs" / "pilot"
TIMING = ("wall_s: wall-clock seconds from the start of this design's work (before its input file is read) to its "
          "answer (the chosen level's surface and its check points), everything the design needs included: the "
          "coil-field quadrature, every VMEC++ start and level.  Measured in a process that had already imported "
          "VMEC++ and finished an untimed warm-up solve on pilot {warmup}.  {threads} threads: VMEC++ max_threads "
          "{threads}, the field kernel's OpenMP and BLAS capped at {threads}, XLA's Eigen at 1; the process on the "
          "cpus in host.cpus.")


def warmup_settings(settings):
    """The design settings with a two-level miniature ladder and the fixed-boundary start: every code path of a
    real solve (field quadrature, fixed- and free-boundary VMEC++, interpolate_solution, the check points)."""
    w = json.loads(json.dumps(settings))
    w["start_order"] = ["fixed+namelist"]
    w["levels"] = [dict(ns=16, mpol=6, ntor=5, ftol=1e-9, niter=4000), dict(ns=16, mpol=7, ntor=6, ftol=1e-9,
                                                                          niter=4000)]
    return w


def set_setting(settings, kv):
    """--set NAME=VALUE: VALUE as JSON where it parses, else the string; NAME may be dotted (common.nstep)."""
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except ValueError:
        pass
    *path, last = k.split(".")
    d = settings
    for p in path:
        d = d[p]
    if last not in d:
        raise SystemExit(f"--set {k}: no such setting")
    d[last] = v


def forms_check(res, log):
    """The evaluator's forms.check_points on the answer's own check points (a check of the written surface only)."""
    if not res.get("surface"):
        return
    try:
        sys.path.insert(0, str(EVALUATOR))
        import forms
        print(f"forms.check_points reproduces the check points to "
              f"{forms.check_points(res['surface'], res['check_points']):.2e} m", file=log, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"check points not compared: no forms.py in {EVALUATOR} ({type(e).__name__}: {e})", file=log,
              flush=True)


def main():
    global THREADS, T0
    ap = argparse.ArgumentParser(
        description="VMEC++ 0.9.3 on one design: the vacuum free-boundary equilibrium of its coils at the "
                    "requested flux, written as DIR/answers/quasr_<id>.json with its log in DIR/logs/.")
    ap.add_argument("design", help="7-digit design id (inputs/quasr_<id>.json beside this file)")
    ap.add_argument("--pilot", action="store_true", help="read the design from inputs/pilot/")
    ap.add_argument("--out", default=str(Path.cwd() / "compare300-out" / SOLVER),
                    help="answers/ and logs/ go here (default ./compare300-out/vmecpp)")
    ap.add_argument("--settings", default=None, help="a JSON file of settings (default: vmecpp_settings.json)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override one setting (VALUE as JSON; NAME may be dotted, e.g. common.nstep)")
    ap.add_argument("--budget", type=float, default=3600.0, help="wall-time budget per design, s")
    ap.add_argument("--mem-gb", type=float, default=12.0,
                    help="memory per design, GiB: recorded only (VMEC++ peaks near 0.63 GB)")
    ap.add_argument("--threads", type=int, default=int(_THREADS), help="VMEC++ max_threads")
    ap.add_argument("--warmup", default=WARMUP_DESIGN, help="pilot design for the untimed warm-up ('' for none)")
    ap.add_argument("--warmup-budget", type=float, default=1200.0,
                    help="seconds the warm-up's first pass may take (its first imports can be slow on a shared "
                         "file system)")
    ap.add_argument("--no-warmup", action="store_true", help="no untimed warm-up: wall_s then includes first calls")
    ap.add_argument("--force", action="store_true", help="run even if the answer exists")
    a = ap.parse_args()
    if _MISSING is not None:
        print(f"VMEC++ or its dependencies are not installed in this Python ({_MISSING}; pip install -r "
              f"requirements/vmecpp.txt)", flush=True)
        sys.exit(1)
    if a.no_warmup:
        a.warmup = ""
    THREADS = a.threads
    T0 = time.time()
    settings = json.loads(Path(a.settings).read_text()) if a.settings else DEFAULT_SETTINGS
    for kv in a.set:
        set_setting(settings, kv)
    src = HERE / "inputs" / ("pilot" if a.pilot else "")
    if not (src / f"quasr_{a.design}.json").exists():
        print(f"no input file {src / f'quasr_{a.design}.json'}", flush=True)
        sys.exit(1)
    out = Path(a.out)
    for sub in ("answers", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dst = out / "answers" / f"quasr_{a.design}.json"
    if dst.exists() and not a.force:
        print(f"{a.design}: {dst} exists (--force to run again)", flush=True)
        return
    logp = out / "logs" / f"quasr_{a.design}.log"
    # everything (Python and VMEC++'s own C++ output) goes to the log
    fd = os.open(logp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    task_out = os.fdopen(os.dup(1), "w")      # this command's own output: one line at the end
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    log = sys.stdout

    def alarm(*_):
        raise Stop("out of time (harness alarm)")

    signal.signal(signal.SIGALRM, alarm)
    not_metered = []

    def meter():
        """fair_cores' Meter, started; None where it cannot measure (it reads Linux's /proc)."""
        try:
            return fair_cores.Meter().start()
        except (RuntimeError, OSError) as e:
            not_metered.append(str(e))
            return None

    # An untimed warm-up solve on a pilot design in this process (imports, first calls into VMEC++, the field
    # kernel and JAX), so that wall_s holds only what the design itself needs; it runs under the meter.
    # The warm-up is the miniature solve twice: a first pass that makes the first calls (modules' first use, numba's
    # and XLA's compilation: about 20 s, mostly one thread, its wait for a CPU bursty and unlike the solve's), then
    # the warm-up proper, whose compute is VMEC++ and the field as in the timed run.  On shared cores the two passes
    # on the same cores waited 0.02 to 0.22 and 0.001 to 0.016 of their run time.
    cores_warmup = cores_first = None
    warm_status = None
    if a.warmup:
        t = time.time()
        try:
            m = meter()
            first = solve(a.warmup, WARMUP_SRC, warmup_settings(settings), a.warmup_budget, log)
            cores_first = m.stop() if m is not None else None
            print(f"warm-up first pass on pilot {a.warmup} (untimed): {first['status']} in "
                  f"{time.time() - t:.1f}s; cores {json.dumps(cores_first)}", file=log, flush=True)
            t = time.time()
            m = meter()
            warm = solve(a.warmup, WARMUP_SRC, warmup_settings(settings), 300, log)
            cores_warmup = m.stop() if m is not None else None
            warm_status = warm["status"]
            print(f"warm-up on pilot {a.warmup} (untimed): {warm['status']} in {time.time() - t:.1f}s; cores "
                  f"{json.dumps(cores_warmup)}", file=log, flush=True)
        except Exception as e:  # noqa: BLE001
            warm_status = f"failed: {e!r}"
            print(f"warm-up on pilot {a.warmup} failed after {time.time() - t:.1f}s: {e!r}", file=log, flush=True)
        if warm_status != "surface":
            print(f"warm-up did not converge ({warm_status}); the design runs all the same", file=log, flush=True)
        print("=" * 100, file=log, flush=True)
    T0 = time.time()
    t_design = T0
    m = meter()
    signal.alarm(int(a.budget) + 120)   # last resort; the iteration callback stops VMEC++ at the budget
    try:
        res = solve(a.design, src, settings, a.budget, log)
    except Exception as e:  # noqa: BLE001  a failure is an answer
        traceback.print_exc()
        res = dict(design=a.design, solver=SOLVER, solver_label=LABEL, versions=versions(), settings=settings,
                   host=host_info(), requested_flux_Wb=None, status="none",
                   solver_status=f"harness error: {e!r}"[:2000],
                   reason=("out of time" if isinstance(e, Stop) else f"error: {e!r}"[:500]),
                   wall_s=time.time() - t_design, surface=None, check_points=None, solver_reported=None,
                   native=None)
        with contextlib.suppress(Exception):
            res["requested_flux_Wb"] = json.loads((src / f"quasr_{a.design}.json").read_text())[
                "requested_toroidal_flux_Wb"]
    signal.alarm(0)
    cores = m.stop() if m is not None else None
    res["timing"] = TIMING.format(warmup=a.warmup or "none", threads=THREADS)
    import resource
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2.0 ** 20 if sys.platform == "darwin" else 1024.0)
    res["host"] = dict(res.get("host") or {}, cores_warmup=cores_warmup, cores=cores, cores_warmup_first_pass=cores_first,
                       cpus=HOST_CPUS, peak_rss_mb=peak_mb, warmup_status=warm_status, mem_gib=a.mem_gb,
                       mem_enforced=False)
    res["fair_cores_sha256"] = fair_cores.SHA256
    res["fair_cores"] = None if cores is None else bool(fair_cores.fair(cores))
    res["timing_rule"] = TIMING_RULE
    res["timing_rule_met"] = res["fair_cores"]
    if cores is None:
        res["not_metered"] = not_metered[-1] if not_metered else "no reading of the timed run"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(res) + "\n")
    tmp.replace(dst)
    rule = ("not metered" if cores is None else
            f"wait_fraction {cores['wait_fraction']:.4f}, timing rule met {res['timing_rule_met']}")
    print(f"wrote {dst}: status {res['status']}, wall {res['wall_s']:.1f}s, {rule}; peak resident memory "
          f"{peak_mb / 1024:.2f} GiB of the {a.mem_gb} GiB allowed", flush=True)
    forms_check(res, log)
    task_out.write(f"{a.design} {res['status']} {res['wall_s']:.0f}s {res.get('reason') or ''} ({rule})\n")
    task_out.flush()


if __name__ == "__main__":
    main()
