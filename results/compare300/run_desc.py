"""DESC 0.17.3 (`desc-opt`, released and unmodified) on one design of the comparison: the vacuum free-boundary
equilibrium of the design's coils at the requested toroidal flux, written as the design's answer file.

    python run_desc.py 0003051                                  # inputs/quasr_0003051.json -> compare300-out/desc
    python run_desc.py 0925611 --pilot --set 'levels=[[6,6]]'   # a pilot design, one setting overridden

Two processes: this one (the supervisor) starts the worker, watches its memory and the wall-time budget and writes
the answer; the worker makes its thread pools four wide, imports DESC, runs an untimed warm-up solve on pilot
0925611, then solves the design and leaves one candidate file per finished level of its resolution ladder.

The worker, per design:
  1. The coils exactly as given: one FourierXYZCoil per coil of the input (every copy, no symmetry rebuilt),
     X_n with cos coefficients at modes 0..K and sin at -1..-K, in one CoilSet; Biot-Savart by DESC's own
     trapezoid rule on `coil_points` nodes per coil, doubled until B settles.
  2. The start: QUASR's stored outermost surface (the surface whose flux defines the request) cut at cylindrical
     planes and fitted by DESC's FourierRZToroidalSurface.from_values (or, with start="namelist", QUASR's VMEC
     namelist of it converted by DESC's own `ptolemy_identity_fwd`).  Psi carries the sign of the coils' toroidal
     field on that surface.
  3. A vacuum Equilibrium (zero pressure, zero current) on that boundary, solved fixed-boundary (DESC's automatic
     continuation as the fallback), then DESC's free-boundary route: `VacuumBoundaryError(field_fixed=True)`
     minimised by `proximal-lsq-exact` with ForceBalance, FixPressure, FixCurrent, FixPsi as constraints.  The
     ladder raises (L, M, N) level by level, each continuing from the last; a level's surface is a candidate when
     DESC's optimiser reports success and DESC's own is_nested() check passes.
  4. The answer: the highest converged level.  Its boundary written as `rz_fourier` by exact product-to-sum
     arithmetic (halves and sums only), and 32 check points computed by DESC on its own surface.

Settings and why.  Chosen on the six pilot designs (0224040, 0925611, 1101401, 1981226, 2126458, 2559727; none of
them among the 300), by DESC's own diagnostics and the comparison's evaluator run on the pilot answers.

- route: `VacuumBoundaryError(field_fixed=True)` minimised by `proximal-lsq-exact` with ForceBalance, FixPressure,
  FixCurrent and FixPsi as constraints: DESC's documented free-boundary route (tutorial `free_boundary_equilibrium`,
  vacuum stellarator section), which recommends VacuumBoundaryError over BoundaryError for vacuum.
- profiles: pressure 0 and current 0 (PowerSeriesProfile); Psi = the requested flux with the sign of the coils'
  B_phi on the start.  DESC's coordinates are right handed in (rho, theta, zeta) with zeta the cylindrical angle,
  so Psi carries the sign of B_phi.
- coils: one FourierXYZCoil per input coil in one CoilSet: exactly the given coefficients and currents.
- coil quadrature: 512 nodes per coil (`LinearGrid(zeta=512)` as `field_grid`), doubled per design until B on the
  start surface and 15% beyond it changes by under 1e-14 of |B| when the nodes double.  DESC's default for these
  order-16 coils is 75 nodes: B on QUASR's stored surface is then wrong by up to 4.1e-4 of |B| (0224040), 7e-7
  (0925611), 2.5e-8 (1101401), larger than DESC's own error on easy designs.  192 nodes reach rounding (5e-16) on
  all six pilots; 512 is the margin; no pilot needed doubling.
- start: QUASR's stored surface (`start.surface`, evaluated from its SIMSOPT tensor-Fourier coefficients) cut at 64
  cylindrical planes per period x 128 points, fitted by DESC's FourierRZToroidalSurface.from_values with QUASR's
  poloidal parameter as theta, at the first level's (M, N).  The two RZ starts the inputs offer are this fit and
  QUASR's VMEC namelist; on the pilots the fit is the better flux surface (coils' max |B.n|/|B|: 3.1e-5 against
  3.8e-4 for 0925611, 1.4e-5 against 6.0e-5 for 1101401, 2.4e-4 against 1.7e-3 for 2126458, 1.9e-2 against 2.6e-2
  for 2559727); a polar angle about the coils' magnetic axis was tried and is far worse (2.9e-2 for 0925611 at
  M = 8).
- interior: `Equilibrium.solve` (lsq-exact, ftol = xtol = 1e-10, gtol = 1e-12, maxiter 200); if DESC finds the
  surfaces unnested or the normalised force above 1e-2, `desc.continuation.solve_continuation_automatic` is tried
  and kept if better.  DESC's default initial guess (boundary scaled toward its m = 0 axis) folds for strongly bent
  cross-sections (0224040: sqrt(g) < 0 in 6% of the volume); automatic continuation is DESC's documented robust
  fixed-boundary route.
- resolution: the ladder M = N = 6, 8, 10 with L = ceil(1.5 M), DESC's default collocation grids (2L, 2M, 2N),
  passed again at each `change_resolution`.  L = M leaves the edge harmonics without radial freedom: on 0925611
  at M = 6, L = 9 gives max |B.n|/|B| 4.5e-6 against 1.0e-5 at L = 6.  Each level starts from the last.  A level
  is begun only if three times the last level's time is still left (from M = 6 to 8 the level time grew 1.2 to 8
  times on the pilots, typically 6: the guard leans towards trying, since a level stopped at the budget costs time
  but not the answer) and its memory (1.9 x the last level's peak, measured 5.3 -> 10.9 GB and 6.1 -> 11.4 GB from
  M = 6 to 8) fits in the 12 GiB (memory is counted in GiB, 2^30 bytes): in practice M = 10 does not fit, and a
  level stopped at the budget leaves the last converged one as the answer.
- boundary-error grid: `LinearGrid(rho=1, M=2M, N=2N, NFP)` per period, full theta (DESC's default): twice the
  boundary's own resolution.
- free modes: all boundary modes free at every level.  The start is already close to the answer; the tutorial's
  k-ladder (free |m|, |n| <= 2, 4, ...) is meant for a circular start and costs one optimisation per step (four per
  level at M = 8: 1,300 to 2,700 s on the pilots).
- outer stopping: ftol 1e-3, xtol 1e-8, gtol 1e-8 (DESC's default), maxiter 100.  DESC's default ftol for
  proximal-lsq-exact is 1e-2; at 1e-8 the optimiser creeps at its floor (cost falling by 1e-4 of itself per step)
  for the whole hour at M = 6 and never reaches the next level; 1e-3 stops within a factor of about 1.2 of that
  floor in |B.n|.  gtol is DESC's own first-order optimality test: it does not need the reduction ratio that a
  collapsing trust region denies at the floor.
- inner (equilibrium) solves: ftol 1e-6, xtol 1e-10, gtol 1e-12, maxiter 50; perturbation order 2 (DESC's
  default).  Looser inner solves (DESC's defaults, or ftol 1e-4 with 20 iterations) make the outer objective noisy
  enough that the trust region collapses ("A bad approximation caused failure to predict improvement", DESC issue
  #802) on 2 of 4 pilots at M = 6; these settings converged on all four.  Perturbation order 1 failed more often.
- restarts: a free-boundary step that ends without success is restarted from where it stopped, up to twice: a fresh
  Jacobian and trust region at the floor often lets DESC's own criterion be met.
- converged: DESC's optimiser reports success and `Equilibrium.is_nested()` is true.  DESC's tutorial-default
  route reports success on 0224040 and 1981226 with force error 1.0 and 0.42 of the magnetic pressure gradient and
  |B.n|/|B| 0.11 and 0.32.
- device and threads: CPU; XLA's pools, BLAS and OpenMP at 4 threads; `desc.config["avail_mem"]` 4.5 GB, DESC's
  Jacobian batching budget: three eighths of the 12 GiB per design.  DESC runs on a card too, but every
  tool in the comparison ran on the same 4 CPU threads.
- compilation cache: JAX's persistent compilation cache, fresh in each process: DESC's documented performance tip
  (it opens the free-boundary tutorial); with the warm-up it keeps design-independent compilation out of `wall_s`.

Pilot evidence (max |B.n|/|B| of the coils on the answer, DESC's own value; the evaluator's float64 value agreed
within 4% on every pilot answer it was run on; times from assorted Intel Core, Xeon, Ryzen and EPYC machines, so
indicative only; QUASR's stored surface is the start).  First the stored surface itself, then DESC's tutorial route
(L = M = N = 8, k-ladder, DESC's defaults, namelist start, 75 coil nodes), then these settings (trial A: the final
inner and outer tolerances, without gtol 1e-8 and the time guard):
- 0224040 (nfp 3, aspect ratio 3.3): 1.4e-2; tutorial "success" at 0.11 with force error 1.0, 1236 s (a false
  success); these settings: none, the fixed-boundary interior is not nested from either start (namelist, stored
  surface fit) and DESC's automatic continuation stops with "something is probably very wrong with your desired
  equilibrium".
- 0925611: 3.1e-5; tutorial none ("bad approximation" after 2743 s at 1.5e-6); these settings: M = 6 4.5e-6
  (1173 s), M = 8 cut at the budget; with the final settings: M = 6 4.5e-6 (DESC's gtol met, 1965 s; M = 8 not
  started by the time guard), wall 2221 s.
- 1101401: 1.4e-5; tutorial 9.4e-8, 1474 s; these settings: M = 6 1.3e-7 (1217 s); M = 8 6.6e-8 when started
  there directly (2657 s).
- 1981226 (nfp 7, iota 2.9): 5.8e-3; tutorial "success" at 0.32 with the flux 18% off (a false success); with the
  final settings: none, out of time: the fixed-boundary interior at M = 6 is not nested (force error 0.30) and
  DESC's automatic continuation was still running at the budget (fits of the stored surface stay at 0.1 to 0.2 even
  at M = 16).
- 2126458: 2.4e-4; tutorial 4.2e-3, 2325 s; these settings: M = 6 9.1e-4 (299 s).
- 2559727: 1.9e-2; tutorial not run; these settings: M = 8 4.2e-3 (2631 s for the level).
Other trials on 0925611, 1101401, 2126458 and 2559727, same start:
- tight tolerances (outer ftol = xtol 1e-8, inner ftol = xtol 1e-10, 100 inner iterations): M = 6 not finished in
  3600 s on 0925611 and 1101401 (50 to 180 s per outer iteration, creeping at the floor).
- DESC's default inner tolerances: the same accuracy at M = 6; M = 8 reached on 0925611 (7.9e-7); 2559727 and
  2126458 (M = 8) end in "bad approximation" after restarts.
- inner ftol 1e-4 with 20 iterations: M = 6 fails on 1101401 and 2559727.
- perturbation order 1: M = 6 fails on 2126458 and 2559727.
- L = M: 2 to 15 times faster at M = 6 but 2 to 11 times worse (0925611 1.0e-5, 1101401 1.4e-6); M = 8 fails on
  0925611 and 2126458.
- start at M = 8 (no M = 6 level): 1101401 6.6e-8 at 2657 s; 0925611 and 2126458 out of time with no answer.
- Biot-Savart chunks of 64 points: peak memory at M = 6 down from about 6.2 to 5 GB, slower; M = 8 still needs
  about 11 GB.

Timing.  `wall_s` runs from the worker reading the design's input to the answer (the supervisor seeing the worker
end), in a process that has already imported DESC and finished the warm-up: imports, and compilation that the
warm-up already did and DESC's compilation cache returns unchanged, are excluded; every compilation the design's own
shapes cause (another number of coils, NFP, resolution) is included.  The 3600 s budget is counted from the same
start: at the budget less 45 s the supervisor sends SIGTERM, the worker stops between Python steps, and the meter
still reads the whole timed run (SIGKILL only if it has not stopped 35 s later).

Threads.  XLA sizes every thread pool it makes (intra-op, dispatch, compilation) by the cpus in the process's
affinity mask at that moment, so pools made under a mask of 8 cpus would hold up to 8 busy threads (with 17 cpus a
DESC process held about 100 threads, more than 4 of them busy at once, and its warm-ups waited for a CPU 37 to 63% of
their running time).  So the worker makes JAX's CPU client, and compiles one function, while it holds `threads` of
the cpus it was given (the first 4: `host.compute_cpus`): every XLA pool is then made with 4 threads.  Every thread
then returns to all the cpus the process was given (the comparison gave it 8: 4 compute threads on 8 cores), where
pools busy at once, Python and the meter use the spare cpus.  `host.threads_by_cpu_mask` records how many threads
hold which mask.  DESC's stack (XLA's Eigen pool, OpenBLAS) keeps its default thread waiting; no OpenMP wait policy
is set.

Memory.  The supervisor watches the worker's resident memory throughout, logs the largest it saw and its cgroup's
memory record, and stops a worker past `mem_guard_gib` (12 GiB) itself, as at the budget, rather than meet the
kernel's OOM killer; the levels finished so far stand.

The timing rule.  Each timed run of the comparison counted only if its threads waited for a CPU at most 5% of the
time they ran and other work used its CPUs for at most 5% of that time.  On Linux the worker runs the warm-up and the
design each under `fair_cores.Meter` (fair_cores.py beside this file; readings in `host.cores_warmup` and
`host.cores`, its sha256 in `fair_cores_sha256`), and `timing_rule_met` says whether this timed run would have
counted.  The meter only measures.  Elsewhere nothing is metered and the answer says so.

The answer's `surface` is DESC's boundary `FourierRZToroidalSurface` (sums of G_m(theta) G_n(NFP zeta), G_k = cos
for k >= 0 and sin |k| below, zeta the cylindrical angle) written as `rz_fourier` (`to_rz_fourier`).
`check_points` are 32 random (theta, phi) at which DESC's own `surface.compute("x", basis="xyz")` evaluated the
answer; the evaluator's forms.py reproduces them to about 5e-16 m.  `native` holds DESC's R_lmn, Z_lmn and modes
(rebuild with `FourierRZToroidalSurface(R_lmn, Z_lmn, modes_R, modes_Z, NFP, sym)`) and the equilibrium's L, M, N,
Psi.  `solver_reported` is DESC's own view of its answer: the boundary objective, max and rms |B.n|/|B| of the coils
on a (4M+9) x (4N+9) boundary grid per period (DESC's Biot-Savart at the quadrature used), the largest mismatch of
the interior field and the coils' field on the boundary, the volume-averaged force error normalised by the magnetic
pressure gradient, iota on axis and edge, Psi and the optimiser's message.  `levels` lists every level the worker
finished, with DESC's messages, times and peak memory.
"""
import argparse
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER, LABEL = "desc", "DESC"
EVALUATOR = Path(os.environ.get("COMPARE300_EVALUATOR") or HERE.parent / "evaluator")
TIMING_RULE = ("a timed run counts only if its threads waited for a CPU at most 5% of the time they ran and other "
               "work used its CPUs for at most 5% of that time (or 0.2 s in all)")

# The settings every design runs with (chosen on the pilots; the docstring gives each one's reason).
SETTINGS = dict(
    levels=[[6, 6], [8, 8], [10, 10]],   # (M, N) of the ladder; boundary resolution = (M, N)
    L_factor=1.5,              # L = ceil(L_factor * M): radial freedom for the edge harmonics
    grid_factor=2,             # L_grid, M_grid, N_grid = grid_factor * (L, M, N) (DESC's default is 2)
    eval_grid_factor=2,        # VacuumBoundaryError nodes: LinearGrid(M=f*M, N=f*N) per field period
    coil_points=512,           # Biot-Savart nodes per coil (DESC's default for these coils: 75)
    coil_tol=1e-14,            # ... doubled until B changes by less than this on doubling (the start surface)
    start="stored_fit",        # QUASR's stored surface fitted by DESC ("namelist": QUASR's VMEC namelist)
    fit_planes=64,             # cross-sections per field period for that fit
    fit_points=128,            # points per cross-section
    continuation_fallback=True,  # solve_continuation_automatic when the first fixed-boundary solve is unhealthy
    free_modes="all",          # all boundary modes free at every level ("ladder": the tutorial's k-ladder)
    fix_solve=dict(ftol=1e-10, xtol=1e-10, gtol=1e-12, maxiter=200),   # fixed-boundary solve of the start
    opt=dict(ftol=1e-3, xtol=1e-8, gtol=1e-8, maxiter=100),            # free-boundary (outer) stopping rules
    solve_options=dict(ftol=1e-6, xtol=1e-10, gtol=1e-12, maxiter=50),  # inner equilibrium solves
    perturb_options=dict(),    # inner perturbation (DESC's default: order 2)
    restarts=2,                # a free-boundary step without success is restarted where it stopped, this often
    ladder_guard=True,         # start a level only if its estimated time and memory fit
    time_growth=3.0,           # start the next level only if 3x the last level's time is left (pilots, M 6 -> 8:
                               # x1.2 to x8, typically x6; a level stopped at the budget leaves the last
                               # converged one as the answer, so the guard errs towards trying)
    rss_growth=1.9,            # next level's peak memory = this x the last level's (pilots: 5.3 -> 10.9 GB, 6.1 -> 11.4 GB,
                               # in 1e9 bytes: 4.9 -> 10.2, 5.7 -> 10.6 GiB)
    mem_gib=12,                # the memory per design, GiB (12 GiB for every tool); a level starts only if its
                               # estimated peak is at most this
    budget_s=3600,             # the wall-time limit per design
    margin_s=45,               # kept for writing the answer after the worker is stopped
    threads=4,                 # threads per design (XLA's pools, BLAS, OpenMP)
    pools_four_wide=True,      # make XLA's pools while the process holds `threads` of its cpus (docstring, Threads)
    jax_cache=True,            # DESC's documented JAX compilation cache, fresh per process (see warmup)
    warmup="0925611",          # untimed warm-up solve on this pilot design in the same process first
    warmup_levels=[[6, 6], [8, 8]],  # ... at the levels a design reaches in 12 GiB, one iteration per solve
    mem_guard_gib=12.0,        # GiB: the supervisor stops a worker past this rather than meet the OOM killer, and
                               # logs the memory it saw
    bs_chunk_size=None,        # VacuumBoundaryError's Biot-Savart chunk of evaluation points (None: all at once)
    avail_mem_gb=4.5,          # DESC's Jacobian batching budget (desc.config["avail_mem"]): three eighths of the
                               # memory per design (DESC reads it in GB)
    n_check=32,                # check points (at least 16)
)


# ---------------------------------------------------------------------------------------------------------
# helpers shared by both processes

def input_path(design, pilot=False):
    """inputs/quasr_<id>.json beside this file (inputs/pilot/ for a pilot design)."""
    return HERE / "inputs" / ("pilot" if pilot else "") / f"quasr_{design}.json"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            return next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
    except (OSError, StopIteration):
        if sys.platform == "darwin":
            try:
                return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            except Exception:  # noqa: BLE001
                pass
    return platform.processor() or None


def host_info(threads=4):
    seen = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    info = dict(machine=platform.machine(), cpu=cpu_model(), threads=threads, cores_visible=seen, gpu=None,
                device="gpu" if os.environ.get("DESC_DEVICE", "cpu") in ("gpu", "card", "cuda") else "cpu")
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      text=True, timeout=20, stderr=subprocess.DEVNULL).strip()
        info["gpu"] = out or None
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------------------------------------------------
# the worker: DESC itself

def worker(args):
    S = json.loads(args.settings_json)
    t0 = float(args.t0)
    cand_dir = Path(args.cand)
    cand_dir.mkdir(parents=True, exist_ok=True)
    # Four-wide pools. XLA sizes its thread pools (intra-op, dispatch, compilation) by the cpus in the affinity
    # mask when they are made, so a pool made under an 8-cpu mask would hold 8 busy threads (with 17 cpus a DESC
    # process held about 100 threads, more than 4 of them busy at once, and its warm-ups waited 37 to 63% of their
    # running time). So JAX and its CPU client are made, and one function compiled, while the process holds
    # `threads` of the cpus it was given (the first 4): every XLA pool is then made with 4 threads. All threads
    # then return to every cpu the process was given (8 in the comparison), where pools busy at once and helper
    # threads (Python, the meter) take the spare cpus. XLA's intra-op pool, BLAS and OpenMP are set to 4 as well;
    # OpenMP waiting is left at its default.
    ncores = int(S["threads"])
    sys.path.insert(0, str(HERE))
    host_cpus, compute_cpus = None, None
    if S["pools_four_wide"] and hasattr(os, "sched_setaffinity"):
        given = sorted(os.sched_getaffinity(0))
        host_cpus = dict(cpu_model=cpu_model(), cpus=given)
        if len(given) > ncores:
            compute_cpus = given[:ncores]
            os.sched_setaffinity(0, compute_cpus)
    os.environ["XLA_FLAGS"] = (f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={ncores} "
                               + os.environ.get("XLA_FLAGS", "")).strip()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(ncores)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # CPU unless a card is asked for (DESC_DEVICE=gpu): the sample runs on 4 CPU cores (set before jax loads)
    device = "gpu" if os.environ.get("DESC_DEVICE", "cpu") in ("gpu", "card", "cuda") else "cpu"
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import numpy as np
    import desc

    desc.set_device(device)
    desc.config["avail_mem"] = min(desc.config.get("avail_mem") or np.inf, float(S["avail_mem_gb"]))
    import jax
    import jax.numpy as jnp
    jax.jit(lambda a: jnp.sin(a) @ a)(jnp.ones((64, 64))).block_until_ready()   # client and pools made now
    if host_cpus:
        # every thread made so far (each pool of 4) may now run on every cpu the process was given, so two pools
        # busy at once use the spare cpus instead of waiting for each other
        for t in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(t), host_cpus["cpus"])
            except OSError:
                pass
    pool_cpus, names = {}, {}
    for t in os.listdir("/proc/self/task") if os.path.isdir("/proc/self/task") else []:
        try:
            m = tuple(sorted(os.sched_getaffinity(int(t))))
            pool_cpus[m] = pool_cpus.get(m, 0) + 1
            nm = open(f"/proc/self/task/{t}/comm").read().strip().rstrip("0123456789")
            names[nm] = names.get(nm, 0) + 1
        except OSError:
            pass
    (cand_dir / "cpus.json").write_text(json.dumps(dict(host_cpus=host_cpus, compute_cpus=compute_cpus,
                                                        threads_by_mask={",".join(map(str, k)): v for k, v in
                                                                         pool_cpus.items()},
                                                        threads_by_name=names)))
    if S["jax_cache"]:
        # DESC's documented compilation cache, fresh for this process: what the warm-up compiled and the design
        # compiles again identically is read back; what the design's shapes need anew is compiled in its time
        import shutil
        cache = cand_dir / "jax_cache"
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    from desc.coils import CoilSet, FourierXYZCoil
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.grid import Grid, LinearGrid
    from desc.objectives import (FixBoundaryR, FixBoundaryZ, FixCurrent, FixPressure, FixPsi, ForceBalance,
                                 ObjectiveFunction, VacuumBoundaryError)
    from desc.profiles import PowerSeriesProfile
    from desc.vmec_utils import ptolemy_identity_fwd

    def say(*a):
        print(f"[{time.time() - t0:8.1f}s]", *a, flush=True)

    say(f"desc {desc.__version__} jax {jax.__version__} device {jax.devices()[0]} x64 {jax.config.jax_enable_x64}"
        f" cores {ncores} avail_mem {desc.config['avail_mem']}")
    def run(d, S, cand_dir, deadline):
        """One design, from its input record to candidate files in cand_dir (one per finished level)."""
        t_run = time.time()
        nfp = int(d["nfp"])
        sym = bool(d["stellarator_symmetric"])
        flux = float(d["requested_toroidal_flux_Wb"])

        # 1. coils, exactly as given
        def coil(c):
            K = len(c["xc"]) - 1

            def co(a):
                out = np.zeros(2 * K + 1)
                out[K:] = c[a + "c"]                       # modes 0..K: cos
                out[:K] = np.asarray(c[a + "s"][1:])[::-1]  # modes -K..-1: sin K..1
                return out
            return FourierXYZCoil(current=float(c["current_A"]), X_n=co("x"), Y_n=co("y"), Z_n=co("z"),
                                  modes=np.arange(-K, K + 1))

        field = CoilSet(*[coil(c) for c in d["coils"]])

        # 2. the start: QUASR's stored outermost surface (the surface whose flux defines the request), fitted by
        # DESC's own least squares (FourierRZToroidalSurface.from_values) on cylindrical cross-sections, with QUASR's
        # poloidal parameter as theta; or (start="namelist") QUASR's VMEC namelist of it, converted by DESC's
        # ptolemy_identity_fwd.  On the pilots the fit is a better flux surface than the namelist at M >= 8
        # (|B.n|/|B| 3e-5 against 4e-4, 2e-4 against 2e-3, ...): the namelist is a coarser fit of the same surface.
        if S["start"] == "namelist":
            import re
            nl = d["start"]["vmec_namelist"]
            rbc, zbs = {}, {}
            for key, store in (("RBC", rbc), ("ZBS", zbs)):
                for n, m, v in re.findall(key + r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*([-+0-9.eEdD]+)", nl):
                    store[(int(m), int(n))] = float(v.replace("D", "e").replace("d", "e"))
            modes = sorted(set(rbc) | set(zbs))
            m0 = np.array([k[0] for k in modes])
            n0 = np.array([k[1] for k in modes])
            mR, nR, xR = ptolemy_identity_fwd(m0, n0, np.zeros(len(modes)),
                                              np.array([rbc.get(k, 0.0) for k in modes]))
            mZ, nZ, xZ = ptolemy_identity_fwd(m0, n0, np.array([zbs.get(k, 0.0) for k in modes]),
                                              np.zeros(len(modes)))
            surf0 = FourierRZToroidalSurface(R_lmn=xR[0], Z_lmn=xZ[0], modes_R=np.vstack([mR, nR]).T.astype(int),
                                             modes_Z=np.vstack([mZ, nZ]).T.astype(int), NFP=nfp, sym=sym,
                                             check_orientation=True)
            say(f"start: namelist M={int(np.max(np.abs(m0)))} N={int(np.max(np.abs(n0)))}")
        else:
            coords, theta = stored_cross_sections(d["start"]["surface"], nfp, S["fit_planes"], S["fit_points"], np)
            M0, N0 = S["levels"][0]
            surf0 = FourierRZToroidalSurface.from_values(coords, theta, M=M0, N=N0, NFP=nfp, sym=sym)
            say(f"start: QUASR's stored surface fitted at M={M0} N={N0} on {S['fit_planes']} planes x "
                f"{S['fit_points']} points")
        say(f"{len(field)} coils, nfp {nfp}, flux {flux}")

        # the coil quadrature: `coil_points` nodes per coil (a Grid: an int fails under jit), doubled while B on
        # the start surface and 15% beyond it changes by more than `coil_tol` of |B| when the nodes double
        g = LinearGrid(M=10, N=10, NFP=nfp)
        xs = np.asarray(surf0.compute("x", grid=g, basis="xyz")["x"])
        Rc = np.hypot(xs[:, 0], xs[:, 1]).mean()
        ph = np.arctan2(xs[:, 1], xs[:, 0])
        ctr = np.stack([Rc * np.cos(ph), Rc * np.sin(ph), 0 * ph], -1)
        probe = np.concatenate([xs, ctr + 1.15 * (xs - ctr)])
        npts = int(S["coil_points"])
        while True:
            B1 = np.asarray(field.compute_magnetic_field(probe, basis="xyz", source_grid=LinearGrid(zeta=npts)))
            B2 = np.asarray(field.compute_magnetic_field(probe, basis="xyz", source_grid=LinearGrid(zeta=2 * npts)))
            chg = float(np.max(np.linalg.norm(B2 - B1, axis=1) / np.linalg.norm(B2, axis=1)))
            say(f"coil quadrature: {npts} nodes per coil, change on doubling {chg:.2e}")
            if chg <= S["coil_tol"] or npts >= 8192:
                break
            npts *= 2
        field_grid = LinearGrid(zeta=npts)
        S["coil_points_used"] = npts

        # Psi's sign: that of the coils' toroidal field on the start surface (DESC's zeta is the cylindrical angle,
        # and its coordinates are right handed, so B^zeta, and with it Psi, has the sign of B_phi)
        xs = np.asarray(surf0.compute("x", grid=g)["x"])
        Bs = np.asarray(field.compute_magnetic_field(xs, source_grid=field_grid))
        Psi = float(np.sign(np.mean(Bs[:, 1])) * flux)
        say(f"Psi = {Psi} (mean B_phi {np.mean(Bs[:, 1]):.4g} T)")

        def res(M, N):
            L = int(math.ceil(S["L_factor"] * M))
            f = S["grid_factor"]
            return dict(L=L, M=M, N=N, L_grid=int(f * L), M_grid=int(f * M), N_grid=int(f * N))

        levels = [tuple(x) for x in S["levels"]]
        M, N = levels[0]
        r = res(M, N)
        surf = surf0.copy()
        surf.change_resolution(M=M, N=N)
        eq = Equilibrium(Psi=Psi, NFP=nfp, sym=sym, surface=surf, pressure=PowerSeriesProfile([0.0]),
                         current=PowerSeriesProfile([0.0]), **r)
        say(f"fixed-boundary solve at {r}")
        eq.solve(verbose=2, **S["fix_solve"])
        fb = interior_health(eq, np)
        say(f"fixed-boundary solve done: {fb}")
        if S["continuation_fallback"] and not fb["ok"]:
            # DESC's documented robust route for a hard fixed-boundary solve: automatic continuation from the
            # axisymmetric part of the boundary, shaping added in adaptive steps
            say("interior not nested or not in force balance: solve_continuation_automatic")
            eq0 = Equilibrium(Psi=Psi, NFP=nfp, sym=sym, surface=surf, pressure=PowerSeriesProfile([0.0]),
                              current=PowerSeriesProfile([0.0]), **r)
            from desc.continuation import solve_continuation_automatic
            try:
                fam = solve_continuation_automatic(eq0, verbose=2, **S["fix_solve"])
                eq2 = fam[-1]
                fb2 = interior_health(eq2, np)
                say(f"continuation done: {fb2}")
                if fb2["force"] < fb["force"] or (fb2["ok"] and not fb["ok"]):
                    eq = eq2
            except RuntimeError as e:   # DESC: "Automatic continuation failed with bdry_step=..."
                say(f"continuation failed ({' '.join(str(e).split())[:200]}); the free boundary starts from the "
                    f"first solve")

        times = []
        for li, (M, N) in enumerate(levels):
            r = res(M, N)
            if li > 0:
                # (the collocation grids are passed with the resolution: change_resolution keeps old grids otherwise)
                eq.change_resolution(**r)
            tl = time.time()
            reset_peak_rss()
            f = S["eval_grid_factor"]
            eval_grid = LinearGrid(rho=np.array([1.0]), M=int(f * M), N=int(f * N), NFP=nfp, sym=False)
            objective = ObjectiveFunction(VacuumBoundaryError(eq=eq, field=field, field_fixed=True,
                                                              eval_grid=eval_grid, field_grid=field_grid,
                                                              bs_chunk_size=S["bs_chunk_size"]))
            constraints = (ForceBalance(eq=eq), FixCurrent(eq=eq), FixPressure(eq=eq), FixPsi(eq=eq))
            ks = [None] if S["free_modes"] == "all" else list(range(2, max(M, N) + 1, 2))
            if ks[-1] is not None and ks[-1] < max(M, N):
                ks.append(max(M, N))
            steps = []
            ok = False
            msg = ""
            # a step that ends without DESC's success (its trust region collapsing at the error floor: "a bad
            # approximation caused failure to predict improvement", or maxiter) is restarted from where it stopped,
            # up to `restarts` times: a fresh Jacobian and trust region at the same point
            for k in ks:
                cons = constraints
                if k is not None:
                    Rm = eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :]
                    Zm = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
                    cons = cons + ((FixBoundaryR(eq=eq, modes=Rm),) if len(Rm) else ()) + \
                        ((FixBoundaryZ(eq=eq, modes=Zm),) if len(Zm) else ())
                say(f"level {li} {r} free modes max(|m|,|n|) <= {k if k is not None else 'all'}")
                for attempt in range(1 + int(S["restarts"])):
                    eq, out = eq.optimize(objective, cons, optimizer="proximal-lsq-exact", verbose=3,
                                          options=dict(solve_options=dict(S["solve_options"]),
                                                       perturb_options=dict(S["perturb_options"])), copy=False,
                                          **S["opt"])
                    ok = bool(out["success"])
                    msg = " ".join(str(out["message"]).split())
                    steps.append(dict(k=k, attempt=attempt, success=ok, message=msg, nfev=int(out.get("nfev", -1)),
                                      nit=int(out.get("nit", -1)), cost=float(out["cost"]), t=time.time() - t_run))
                    say(f"level {li} step k={k} attempt {attempt}: success={ok} '{msg}' nfev={out.get('nfev')} "
                        f"cost={float(out['cost']):.3e}")
                    if ok:
                        break
            nested = bool(eq.is_nested())
            # converged: DESC's optimiser reports success and DESC's own check finds nested flux surfaces
            lvl = dict(level=li, **r, steps=steps, success=ok, nested=nested, converged=bool(ok and nested),
                       message=msg, wall_level_s=time.time() - tl, coil_points=npts)
            times.append(time.time() - tl)
            diag = diagnostics(eq, field, field_grid, objective, np, LinearGrid)
            lvl["diagnostics"] = diag
            lvl["peak_rss_gib"] = peak_rss_gib()
            say(f"level {li} diagnostics {json.dumps(diag)}")
            cand = dict(lvl, surface=to_rz_fourier(eq.surface, np), native=native(eq, np),
                        check_points=check_points(eq.surface, S["n_check"], np, Grid), t=time.time() - t_run)
            write_json(cand_dir / f"level{li:02d}.json", cand)
            # the next level only if it can finish and fit in memory: its time and peak memory are estimated from
            # this level's by the growth measured on the pilots (time x2 or more, memory x1.9 per step of 2 in M)
            if li + 1 < len(levels) and S["ladder_guard"]:
                est = times[-1] * S["time_growth"]
                left = deadline - time.time()
                rss = lvl["peak_rss_gib"] * S["rss_growth"]
                say(f"next level estimate {est:.0f}s ({left:.0f}s left), {rss:.1f} GiB ({S['mem_gib']} GiB allowed)")
                why = None
                if est > left:
                    why = dict(reason="budget", estimate_s=est, left_s=left, next_level=li + 1)
                elif rss > S["mem_gib"]:
                    why = dict(reason="memory", estimate_gib=rss, mem_gib=S["mem_gib"], next_level=li + 1)
                if why:
                    say(f"stopping the ladder: {why}")
                    write_json(cand_dir / "stopped.json", why)
                    break

    # An untimed warm-up solve on a pilot design in this process (imports, and compilation that does not
    # depend on the design, are then excluded from the design's time), with the same levels as far as a design
    # can reach in memory and only a few iterations per solve.
    # The warm-up and the timed design each run under fair_cores' Meter, which only measures (Linux; elsewhere
    # nothing is metered and the answer says so).
    sys.path.insert(0, str(HERE))
    import fair_cores

    def meter_start():
        try:
            return fair_cores.Meter().start()
        except (RuntimeError, OSError) as e:
            write_json(cand_dir / "not_metered.json", dict(why=str(e)))
            return None

    def meter_say(what, c):
        if c is None:
            say(f"{what} cores: not metered")
            return
        say(f"{what} cores: wait_fraction {c['wait_fraction']:.4f}, foreign_fraction "
            f"{c.get('foreign_fraction', 0):.4f}, busy_cores {c['busy_cores']:.2f}, "
            f"timing rule met {fair_cores.fair(c)}")

    meter_w = meter_start()
    if S.get("warmup"):
        tw = time.time()
        W = json.loads(json.dumps(S))
        W.update(levels=S["warmup_levels"], restarts=0, ladder_guard=False, continuation_fallback=False,
                 fix_solve=dict(S["fix_solve"], maxiter=1), opt=dict(S["opt"], maxiter=1),
                 solve_options=dict(S["solve_options"], maxiter=1))
        wdir = cand_dir / "warmup"
        wdir.mkdir(parents=True, exist_ok=True)
        try:
            say(f"warm-up on pilot {S['warmup']} (untimed)")
            run(json.loads(input_path(S["warmup"], pilot=True).read_text()), W, wdir, time.time() + 1e9)
        except Exception as e:  # noqa: BLE001 - the warm-up only prepares the process
            say(f"warm-up ended early: {type(e).__name__}: {' '.join(str(e).split())[:300]}")
        say(f"warm-up done in {time.time() - tw:.1f}s")
    cores_w = meter_w.stop() if meter_w is not None else None
    if cores_w is not None:
        write_json(cand_dir / "cores_warmup.json", cores_w)
    meter_say("warm-up", cores_w)

    # the design: its time starts here, with reading its input. The supervisor's SIGTERM at the budget ends the
    # solve between Python steps, so the meter still reads the whole timed run
    class BudgetStop(BaseException):
        pass

    def on_term(signum, frame):
        raise BudgetStop()

    import signal
    signal.signal(signal.SIGTERM, on_term)
    meter = meter_start()
    t_design = time.time()
    (cand_dir / "design_start").write_text(repr(t_design))
    say("design start")
    stopped = False
    try:
        d = json.loads(Path(args.input).read_text())
        run(d, S, cand_dir, t_design + S["budget_s"] - S["margin_s"])
    except BudgetStop:
        stopped = True
        say("stopped at the budget")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        cores = meter.stop() if meter is not None else None
        if cores is not None:
            cores["stopped_at_budget"] = stopped
            write_json(cand_dir / "cores.json", cores)
        meter_say("design", cores)
    say("worker done")


def proc_rss_gib(pid):
    """Resident memory of a process, GiB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) * 1024 / 2**30
    except OSError:
        pass
    return 0.0


def cgroup_memory():
    """This process's cgroup memory limit, peak and OOM-kill count (cgroup v2, else v1), for the log."""
    out = {}
    try:
        rel = open("/proc/self/cgroup").read().strip().splitlines()
        v2 = [ln.split("::", 1)[1] for ln in rel if ln.startswith("0::")]
        if v2:
            base = "/sys/fs/cgroup" + v2[0]
            for k in ("memory.max", "memory.peak", "memory.current"):
                try:
                    out[k] = open(os.path.join(base, k)).read().strip()
                except OSError:
                    pass
            try:
                out["memory.events"] = dict(ln.split() for ln in open(os.path.join(base, "memory.events")))
            except OSError:
                pass
        mem = [ln.split(":", 2)[2] for ln in rel if ":memory:" in ln]
        if mem:
            base = "/sys/fs/cgroup/memory" + mem[0]
            for k in ("memory.limit_in_bytes", "memory.max_usage_in_bytes", "memory.failcnt"):
                try:
                    out[k] = open(os.path.join(base, k)).read().strip()
                except OSError:
                    pass
    except OSError:
        pass
    return out


def reset_peak_rss():
    """Reset the kernel's peak-RSS mark of this process (Linux: /proc/self/clear_refs, 5)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def peak_rss_gib():
    """Peak resident memory since the last reset (VmHWM), else since the process began (ru_maxrss)."""
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmHWM:"):
                    return int(ln.split()[1]) * 1024 / 2**30
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)


def stored_cross_sections(s, nfp, nplanes, nu, np):
    """QUASR's stored SurfaceXYZTensorFourier (its degrees of freedom, SIMSOPT's definition) on the planes
    phi = 2 pi k / (nfp nplanes): for each u = j / nu, Newton on v for the point's cylindrical angle.  Returns
    (R, phi, Z) coordinates and theta = 2 pi u."""
    mpol, ntor, ss = int(s["mpol"]), int(s["ntor"]), bool(s["stellsym"])
    dofs = np.asarray(s["x"], float)
    mats, k = {}, 0
    for c in "xyz":
        A = np.zeros((2 * mpol + 1, 2 * ntor + 1))
        for i in range(2 * mpol + 1):
            for j in range(2 * ntor + 1):
                if ss and (((j <= ntor and i > mpol) or (j > ntor and i <= mpol)) if c == "x" else
                           ((j <= ntor and i <= mpol) or (j > ntor and i > mpol))):
                    continue
                A[i, j] = dofs[k]
                k += 1
        mats[c] = A
    assert k == len(dofs)
    sn = int(s["nfp"])

    def xyz(u, v):
        th, ph = 2 * np.pi * u, 2 * np.pi * v
        W = np.stack([np.cos(i * th) if i <= mpol else np.sin((i - mpol) * th) for i in range(2 * mpol + 1)], -1)
        V = np.stack([np.cos(sn * j * ph) if j <= ntor else np.sin(sn * (j - ntor) * ph)
                      for j in range(2 * ntor + 1)], -1)
        h = {c: np.einsum("pi,ij,pj->p", W, mats[c], V) for c in "xyz"}
        return np.stack([h["x"] * np.cos(ph) - h["y"] * np.sin(ph), h["x"] * np.sin(ph) + h["y"] * np.cos(ph),
                         h["z"]], -1)

    u = np.arange(nu) / nu
    R, P, Z, T = [], [], [], []
    for kp in range(nplanes):
        phi0 = 2 * np.pi * kp / (nfp * nplanes)
        v = np.full(nu, phi0 / (2 * np.pi))
        for _ in range(60):
            x = xyz(u, v)
            f = np.angle(np.exp(1j * (np.arctan2(x[:, 1], x[:, 0]) - phi0)))
            xp = xyz(u, v + 1e-7)
            fp = (np.angle(np.exp(1j * (np.arctan2(xp[:, 1], xp[:, 0]) - phi0))) - f) / 1e-7
            dv = f / fp
            v = v - dv
            if np.max(np.abs(dv)) < 1e-15:
                break
        x = xyz(u, v)
        R.append(np.hypot(x[:, 0], x[:, 1])); Z.append(x[:, 2]); P.append(np.full(nu, phi0)); T.append(2 * np.pi * u)
    return np.stack([np.concatenate(R), np.concatenate(P), np.concatenate(Z)], -1), np.concatenate(T)


def interior_health(eq, np):
    """DESC's own check of a fixed-boundary solve: the sign of the Jacobian sqrt(g) over the volume (nested
    surfaces) and the volume-averaged force error normalised by the magnetic pressure gradient."""
    from desc.grid import QuadratureGrid
    g = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    sg = np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"])
    v = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
    force = float(v["<|F|>_vol"] / v["<|grad(|B|^2)|/2mu0>_vol"])
    nested = bool(np.all(np.sign(sg) == np.sign(np.median(sg)))) and bool(eq.is_nested())
    return dict(nested=nested, force=force, ok=bool(nested and force < 1e-2))


def diagnostics(eq, field, field_grid, objective, np, LinearGrid):
    """DESC's own measures of its answer: the boundary objective, |B.n|/|B| of the coils on a fine boundary
    grid, the interior-exterior mismatch, the force residual, iota and Psi."""
    out = dict(Psi=float(eq.Psi), L=int(eq.L), M=int(eq.M), N=int(eq.N))
    try:
        f = objective.compute_scaled_error(objective.x(*objective.things))
        out["boundary_error_rms"] = float(np.sqrt(np.mean(np.asarray(f) ** 2)))
        out["boundary_error_max"] = float(np.max(np.abs(np.asarray(f))))
    except Exception as e:  # noqa: BLE001
        out["boundary_error_note"] = repr(e)
    g = LinearGrid(rho=np.array([1.0]), M=4 * eq.M + 8, N=4 * eq.N + 8, NFP=eq.NFP, sym=False)
    data = eq.compute(["x", "n_rho", "B", "|B|"], grid=g, basis="xyz")
    xyz = np.asarray(data["x"])
    Bc = np.asarray(field.compute_magnetic_field(xyz, basis="xyz", source_grid=field_grid))
    n_xyz = np.asarray(data["n_rho"])          # (basis="xyz" returns every vector in xyz components)
    bn = np.abs(np.sum(Bc * n_xyz, 1)) / np.linalg.norm(Bc, axis=1)
    out["coil_Bn_over_B_max"] = float(bn.max())
    out["coil_Bn_over_B_rms"] = float(np.sqrt(np.mean(bn ** 2)))
    Bin = np.asarray(data["B"])
    out["B_in_minus_B_coils_over_B_max"] = float(np.max(np.linalg.norm(Bin - Bc, axis=1) /
                                                        np.linalg.norm(Bc, axis=1)))
    try:
        v = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
        out["force_vol_avg_normalized"] = float(v["<|F|>_vol"] / v["<|grad(|B|^2)|/2mu0>_vol"])
    except Exception as e:  # noqa: BLE001
        out["force_note"] = repr(e)
    gi = LinearGrid(rho=np.array([0.0, 1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    iota = np.asarray(eq.compute("iota", grid=gi)["iota"])
    iota = gi.compress(iota)
    out["iota_axis"], out["iota_edge"] = float(iota[0]), float(iota[-1])
    return out


def to_rz_fourier(surf, np):
    """DESC's FourierRZToroidalSurface (sums of G_m(theta) G_n(NFP zeta), G_k = cos k for k >= 0, sin |k| below)
    -> rz_fourier, exactly: cos a cos b = (cos(a-b) + cos(a+b))/2, sin a sin b = (cos(a-b) - cos(a+b))/2,
    sin a cos b = (sin(a-b) + sin(a+b))/2, cos a sin b = (sin(a+b) - sin(a-b))/2 with a = |m| theta,
    b = |n| NFP zeta; then (0, -n) folded onto (0, n) (cos even, sin odd)."""
    acc = {}

    def add(m, n, key, v):
        if m == 0 and n < 0:
            n = -n
            if key.endswith("s"):
                v = -v
        e = acc.setdefault((m, n), {})
        e[key] = e.get(key, 0.0) + v

    for ax, coef, modes in (("r", surf.R_lmn, surf.R_basis.modes), ("z", surf.Z_lmn, surf.Z_basis.modes)):
        for c, (_, m, n) in zip(np.asarray(coef, dtype=float), np.asarray(modes)):
            c = float(c)
            if c == 0.0:
                continue
            a, b = abs(int(m)), abs(int(n)) * int(surf.NFP)
            h = 0.5 * c
            if b == 0:                                     # G_n = 1
                if m >= 0:
                    add(a, 0, ax + "c", c)                 # cos(a theta)
                else:
                    add(a, 0, ax + "s", c)                 # sin(a theta)
            elif m >= 0 and n >= 0:                        # cos a cos b
                add(a, b, ax + "c", h); add(a, -b, ax + "c", h)
            elif m < 0 and n < 0:                          # sin a sin b
                add(a, b, ax + "c", h); add(a, -b, ax + "c", -h)
            elif m < 0 and n >= 0:                         # sin a cos b
                add(a, b, ax + "s", h); add(a, -b, ax + "s", h)
            else:                                          # cos a sin b
                add(a, -b, ax + "s", h); add(a, b, ax + "s", -h)
    keys = sorted(acc)
    out = dict(form="rz_fourier", m=[k[0] for k in keys], n=[k[1] for k in keys])
    for k in ("rc", "rs", "zc", "zs"):
        out[k] = [acc[key].get(k, 0.0) for key in keys]
    return out


def native(eq, np):
    s = eq.surface
    return dict(library="desc", object="FourierRZToroidalSurface (Equilibrium.surface)", NFP=int(s.NFP),
                sym=bool(s.sym), R_lmn=np.asarray(s.R_lmn, float).tolist(),
                modes_R=np.asarray(s.R_basis.modes[:, 1:], int).tolist(),
                Z_lmn=np.asarray(s.Z_lmn, float).tolist(), modes_Z=np.asarray(s.Z_basis.modes[:, 1:], int).tolist(),
                equilibrium=dict(L=int(eq.L), M=int(eq.M), N=int(eq.N), Psi=float(eq.Psi),
                                 spectral_indexing=str(eq.spectral_indexing)),
                rebuild="FourierRZToroidalSurface(R_lmn, Z_lmn, modes_R, modes_Z, NFP=NFP, sym=sym)")


def check_points(surf, k, np, Grid, seed=7):
    """k points (theta, phi uniform) evaluated by DESC on its own surface; phi is DESC's zeta (cylindrical)."""
    rng = np.random.default_rng(seed)
    th, ph = 2 * np.pi * rng.random(k), 2 * np.pi * rng.random(k)
    nodes = np.stack([np.ones(k), th, ph], -1)
    grid = Grid(nodes, jitable=True)
    x = np.asarray(surf.compute("x", grid=grid, basis="xyz")["x"])
    return dict(theta=th.tolist(), phi=ph.tolist(), xyz=x.tolist())


# ---------------------------------------------------------------------------------------------------------
# the supervisor

def versions():
    try:
        out = subprocess.check_output([sys.executable, "-c",
                                       "import desc, jax, jaxlib, numpy, scipy, sys; print(desc.__version__, "
                                       "jax.__version__, jaxlib.__version__, numpy.__version__, scipy.__version__,"
                                       " sys.version.split()[0])"], text=True, timeout=300,
                                      stderr=subprocess.DEVNULL).split()
        return dict(zip(["desc", "jax", "jaxlib", "numpy", "scipy", "python"], out))
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e))


def parse_value(v):
    """A --set value: JSON where it parses (numbers, lists, objects, true, false, null), else the string itself."""
    try:
        return json.loads(v)
    except ValueError:
        return v


def forms_check(ans, log_path):
    """The evaluator's forms.check_points on the answer's own check points (a check of the written surface only;
    nothing here chooses or rejects an answer)."""
    if not ans.get("surface"):
        return
    try:
        sys.path.insert(0, str(EVALUATOR))
        import forms
        msg = f"forms.check_points reproduces the check points to {forms.check_points(ans['surface'], ans['check_points']):.2e} m"
    except Exception as e:  # noqa: BLE001
        msg = f"check points not compared: no forms.py in {EVALUATOR} ({type(e).__name__}: {e})"
    with open(log_path, "a") as log:
        log.write(f"# {msg}\n")


def main():
    ap = argparse.ArgumentParser(
        description="DESC 0.17.3 on one design: the vacuum free-boundary equilibrium of its coils at the requested "
                    "flux, written as DIR/answers/quasr_<id>.json with its log in DIR/logs/.")
    ap.add_argument("design", help="7-digit design id (inputs/quasr_<id>.json beside this file)")
    ap.add_argument("--pilot", action="store_true", help="read the design from inputs/pilot/")
    ap.add_argument("--out", default=str(Path.cwd() / "compare300-out" / SOLVER),
                    help="answers/ and logs/ go here (default ./compare300-out/desc)")
    ap.add_argument("--budget", type=float, default=None, help="wall-time budget per design, s (default 3600)")
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="memory per design, GiB: the ladder's limit and the supervisor's guard (default 12)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override one setting (VALUE as JSON, e.g. --set 'levels=[[6,6]]'); see SETTINGS")
    ap.add_argument("--no-warmup", action="store_true", help="no untimed warm-up: wall_s then includes compilation")
    ap.add_argument("--force", action="store_true", help="run even if the answer exists")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--input", help=argparse.SUPPRESS)
    ap.add_argument("--cand", help=argparse.SUPPRESS)
    ap.add_argument("--t0", help=argparse.SUPPRESS)
    ap.add_argument("--settings-json", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    if importlib.util.find_spec("desc") is None:
        print("DESC is not installed in this Python (pip install -r requirements/desc.txt)", flush=True)
        sys.exit(1)

    t0 = time.time()
    S = json.loads(json.dumps(SETTINGS))
    if args.budget is not None:
        S["budget_s"] = args.budget
    if args.mem_gb is not None:
        S["mem_gib"] = S["mem_guard_gib"] = args.mem_gb
    if args.no_warmup:
        S["warmup"] = None
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in S:
            ap.error(f"--set {k}: no such setting (one of {', '.join(S)})")
        S[k] = parse_value(v)
    design = args.design
    inp = input_path(design, args.pilot)
    if not inp.exists():
        print(f"no input file {inp}", flush=True)
        sys.exit(1)
    d = json.loads(inp.read_text())
    out = Path(args.out)
    if (out / "answers" / f"quasr_{design}.json").exists() and not args.force:
        print(f"{design}: {out / 'answers' / f'quasr_{design}.json'} exists (--force to run again)", flush=True)
        return
    log_path = out / "logs" / f"quasr_{design}.log"
    cand = out / "work" / f"quasr_{design}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if cand.exists():
        import shutil
        shutil.rmtree(cand, ignore_errors=True)
    cand.mkdir(parents=True, exist_ok=True)
    marker = cand / "design_start"
    cmd = [sys.executable, str(Path(__file__).resolve()), design, "--worker", "--input", str(inp),
           "--cand", str(cand), "--t0", repr(t0), "--settings-json", json.dumps(S)]
    killed, rc, err, t_design = False, None, None, None
    import signal

    def stop(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return proc.returncode

    with open(log_path, "w") as log:
        log.write(f"# DESC free boundary, design {design}, input {inp}\n# settings {json.dumps(S)}\n")
        log.write(f"# host {json.dumps(host_info(S['threads']))}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # imports and the untimed warm-up first: the design's clock starts when the worker reads its input.
        # One loop watches the worker throughout: its memory (largest RSS seen; a guard at the memory per
        # design), the design's start and its budget.
        max_rss, guard, deadline, term_at = 0.0, None, None, None
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            now = time.time()
            r = proc_rss_gib(proc.pid)
            max_rss = max(max_rss, r)
            if t_design is None and marker.exists():
                t_design = float(marker.read_text())
                deadline = t_design + S["budget_s"] - S["margin_s"]
            if r > S["mem_guard_gib"] and guard is None:
                guard = dict(rss_gib=r, phase="design" if t_design else "warm-up", t=now - t0)
                if t_design is None:
                    stop(proc)
                    rc = proc.returncode
                    break
                term_at = now                  # the design stops as at the budget; its levels so far stand
                killed = True
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if deadline and now > deadline and term_at is None:
                killed = True
                term_at = now
                # SIGTERM first: the worker stops between Python steps and its meter reads the whole run
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if term_at is not None and now - term_at > S["margin_s"] - 10:
                rc = stop(proc)
                break
            time.sleep(0.5)
        if t_design is None and marker.exists():    # a worker that ended between two polls
            t_design = float(marker.read_text())
        mem = cgroup_memory()
        log.write(f"\n# supervisor: worker exit {rc}; largest RSS seen {max_rss:.2f} GiB; memory guard "
                  f"{guard}; cgroup {json.dumps(mem)}\n")
        if killed and guard is None:
            log.write(f"# supervisor: worker stopped at the budget ({time.time() - t_design:.1f}s of the design)\n")
        if t_design is None and guard:
            log.write(f"\n# supervisor: warm-up past the {S['mem_guard_gib']} GiB memory guard "
                      f"({guard['rss_gib']:.1f} GiB); nothing timed\n")
            log.flush()
            print(f"{design}: the warm-up went past the {S['mem_guard_gib']} GiB memory guard "
                  f"({guard['rss_gib']:.1f} GiB); nothing timed", flush=True)
            sys.exit(1)
    wall = time.time() - t_design if t_design else None

    levels = []
    for p in sorted(cand.glob("level*.json")):
        levels.append(json.loads(p.read_text()))
    stopped = json.loads((cand / "stopped.json").read_text()) if (cand / "stopped.json").exists() else None
    tail = ""
    try:
        tail = "".join(log_path.read_text().splitlines(True)[-40:])
    except OSError:
        pass
    if rc not in (0, None) and not killed:
        if rc < 0:
            sig = signal.Signals(-rc).name if -rc in signal.Signals._value2member_map_ else str(-rc)
            oom = (mem.get("memory.events") or {}).get("oom_kill")
            err = (f"worker killed by {sig}" + (f" (the supervisor's memory guard, {guard['rss_gib']:.1f} GiB)"
                                                if guard else "")
                   + f"; largest RSS seen {max_rss:.1f} GiB" + (f"; cgroup oom_kill count {oom}" if oom else ""))
        else:
            err = next((ln for ln in reversed(tail.splitlines())
                        if ("Error" in ln or "Exception" in ln) and not ln.startswith(("Building", "Timer"))),
                       f"worker exit {rc}")
        if t_design is None:
            # the worker died in its imports or the untimed warm-up, before this design began: no answer
            with open(log_path, "a") as log:
                log.write(f"\n# supervisor: worker died before the design began ({err.strip()}); no answer\n")
            print(f"{design}: worker died before the design began: {err.strip()} (log {log_path})", flush=True)
            sys.exit(1)
    converged = [c for c in levels if c.get("converged", c["success"])]
    best = converged[-1] if converged else None   # the highest converged level (the ladder only rises)
    n_levels = len(S["levels"])
    lvl_txt = "; ".join(f"level {c['level']} (L,M,N)=({c['L']},{c['M']},{c['N']}): "
                        f"{'success' if c['success'] else 'no success'}"
                        f"{'' if c.get('nested', True) else ' but surfaces not nested (is_nested False)'}: "
                        f"{c['message']}" for c in levels)
    unfinished = None
    if killed and guard:
        unfinished = f"level {len(levels)} stopped by the {S['mem_guard_gib']} GiB memory guard ({guard['rss_gib']:.1f} GiB)"
    elif killed:
        unfinished = f"level {len(levels)} stopped at the {S['budget_s']} s budget"
    elif err:
        unfinished = f"error: {err.strip()}"
    elif stopped and stopped["reason"] == "budget":
        unfinished = f"levels from {stopped['next_level']} not started: estimated {stopped['estimate_s']:.0f} s, " \
                     f"{stopped['left_s']:.0f} s left"
    elif stopped:
        unfinished = f"levels from {stopped['next_level']} not started: estimated {stopped['estimate_gib']:.1f} GiB " \
                     f"of the {stopped['mem_gib']} GiB allowed"
    solver_status = lvl_txt + (("; " + unfinished) if unfinished else "") if (lvl_txt or unfinished) else "no output"
    ans = dict(design=design, solver=SOLVER, solver_label=LABEL, versions=versions(), settings=S,
               host=host_info(S["threads"]), requested_flux_Wb=float(d["requested_toroidal_flux_Wb"]),
               status="surface" if best else "none", solver_status=solver_status, reason=None,
               wall_s=wall,
               wall_s_definition=(f"from the worker reading this design's input to the answer (the supervisor "
                                  f"seeing the worker end), in a process that had imported DESC and finished an "
                                  f"untimed warm-up solve on pilot {S['warmup']} (levels {S['warmup_levels']}, one "
                                  f"iteration per solve); every compilation this design causes is included; "
                                  f"{S['threads']} threads (XLA intra-op, BLAS, OpenMP), XLA's pools made while the "
                                  f"process held {S['threads']} of its cpus (host.compute_cpus), every thread then "
                                  f"on all the cpus it was given (host.cpus); the budget {S['budget_s']} s is "
                                  f"counted from the same start; the warm-up and this timed run each under "
                                  f"fair_cores' Meter where it can measure (host.cores_warmup, host.cores)")
               if S.get("warmup") else
               "from the worker reading this design's input to the answer, with no warm-up: imports and every "
               "compilation included",
               surface=None, check_points=None, solver_reported=None, native=None,
               levels=[{k: v for k, v in c.items() if k not in ("surface", "native", "check_points")}
                       for c in levels])
    if levels:
        S["coil_points_used"] = levels[-1]["coil_points"]
    if best:
        ans.update(surface=best["surface"], check_points=best["check_points"], native=best["native"],
                   solver_reported=dict(level=best["level"], **best["diagnostics"],
                                        optimizer_message=best["message"]))
    else:
        if killed and guard:
            ans["reason"] = f"error: memory past {S['mem_guard_gib']} GiB at level {len(levels)}"
        elif killed:
            ans["reason"] = "out of time"
        elif err:
            ans["reason"] = f"error: {err.strip()}"
        elif levels and levels[-1]["success"] and not levels[-1].get("nested", True):
            ans["reason"] = "not converged: optimiser success but surfaces not nested"
        elif levels:
            ans["reason"] = "not converged: " + levels[-1]["message"]
        else:
            ans["reason"] = "error: no level finished"
        if levels:
            ans["solver_reported"] = dict(level=levels[-1]["level"], **levels[-1]["diagnostics"],
                                          optimizer_message=levels[-1]["message"])
    # the meter's readings of the warm-up and of this timed run, and whether the timed run met the timing rule
    sys.path.insert(0, str(HERE))
    import fair_cores
    cw = json.loads((cand / "cores_warmup.json").read_text()) if (cand / "cores_warmup.json").exists() else None
    cr = json.loads((cand / "cores.json").read_text()) if (cand / "cores.json").exists() else None
    cj = json.loads((cand / "cpus.json").read_text()) if (cand / "cpus.json").exists() else {}
    nm = json.loads((cand / "not_metered.json").read_text()) if (cand / "not_metered.json").exists() else None
    ans["host"].update(cores_warmup=cw, cores=cr, cpus=cj.get("host_cpus"), compute_cpus=cj.get("compute_cpus"),
                       threads_by_cpu_mask=cj.get("threads_by_mask"), threads_by_name=cj.get("threads_by_name"),
                       largest_rss_gib=max_rss)
    if cj.get("host_cpus"):
        ans["host"]["cpu"] = cj["host_cpus"]["cpu_model"]
    ans["fair_cores_sha256"] = fair_cores.SHA256
    ans["timing_rule"] = TIMING_RULE
    ans["timing_rule_met"] = bool(fair_cores.fair(cr)) if cr is not None else None
    if cr is None:
        ans["not_metered"] = (nm or {}).get("why") or "no reading of the timed run"
    write_json(out / "answers" / f"quasr_{design}.json", ans)
    rule = ("not metered" if cr is None else f"wait_fraction {cr['wait_fraction']:.4f}, foreign_fraction "
            f"{cr.get('foreign_fraction', 0):.4f}, timing rule met {ans['timing_rule_met']}")
    with open(log_path, "a") as log:
        log.write(f"\n# answer: status {ans['status']}, wall {wall:.1f}s, {rule}, {solver_status}\n")
    forms_check(ans, log_path)
    print(f"{design} {ans['status']} {wall:.0f}s {ans['reason'] or ''} ({rule})")


if __name__ == "__main__":
    main()
