"""torux 0.1.0 (the package `torux` as pip installed it in this Python, unmodified) on one design of the comparison:
the magnetic surface of the design's coils that encloses the requested toroidal flux, written as the design's answer
file with torux's full record beside its log.

    python run_torux.py 0003051                       # inputs/quasr_0003051.json -> compare300-out/torux
    python run_torux.py 0925611 --pilot --no-warmup

Flow: this process (the supervisor) starts a worker process and stops it if the design runs past the wall-time
limit (3,600 s from the design's start); the worker imports torux, runs two untimed warm-up solves on pilot designs
(0925611 with the run's settings, 0224040 with a 240 s budget), then solves the design and writes its answer.

What runs: torux.api.solve(torux.Request(source, flux, budget_s, tol, verify=True, trace_lines=0, gradients=False)),
the solver's own entry point (torux.solve answers through it), on a torux.Source built from the design's input
exactly as torux.load builds it from such a file: the coils by torux.io.fourier_json.load_fourier_json (the reader
of exactly that format), the
request from the file's requested_toroidal_flux_Wb, the design surface from the file's `start` block (QUASR's
outermost stored surface as torux.io.surfaces.SurfaceXYZTensorFourier, in Source._design_surfaces, where
torux.api.solve looks for design surfaces), the coil quadrature floor 0.2 x the least coil-to-start-surface distance.
Two differences from torux.load: the database row (`database_metadata`) is dropped, since only the coils, the flux
and the start surface may reach a solver, and the distance is computed once here so that the answer can record it.
torux.api.solve then runs the design-surface path (torux.analysis.analyse_design, its full path: resonant stages and
late stages on) and the independent verification.  Nothing of the package is replaced or patched.

The answer is `surface` only when torux's own criterion accepts the torus: status 'surface' (refined to the
arithmetic limit, resonance guard, quadrature estimate, topology; and its independent verification,
torux.verify.acceptance, ok: the package claims 'surface' only when both agree).  Everything else is `none`, torux's
outcome kept in `solver_status`.

The torus (torux.torus.TorusState: R = sum rc[m, n + N] cos(m theta - n nfp phi), Z = sum zs[m, n + N] sin(...),
theta its straight-field-line angle, phi the cylindrical angle, phi0 = 0) is written as rz_fourier term by term:
one entry per nonzero native coefficient, m_j = m, n_j = n nfp, no arithmetic on any coefficient.  Check points are
torux.verify.torus_points (the package's own evaluation of the torus, what torux.Result.points calls) at 32 random
(theta, phi).

Settings and why:
- entry point: torux.api.solve(torux.Request(...)), the tool as torux documents it: it states a budget, runs the full
  path and checks every surface independently.
- design surface: the `start` block only (QUASR's outermost stored surface).  For all 306 designs (the 300 and the
  six pilots) the start surface equals, dof for dof, the outermost surface torux's own QUASR reader takes from the
  database's serial file.  torux's own QUASR runs also pass QUASR's inner surfaces (axis guess, iota profile); that
  is database information beyond the coils, so it is not used here (on the six pilots every QUASR surface in place
  of the start block gave the same outcome).
- iota hints: none; torux takes the rotational transform from the axis it finds.
- budget_s 3300 s: the 3,600 s limit less a margin for torux's documented overrun (it stops "to within about one
  Newton iteration") and process start; within it torux's own stopping rules decide.  Its own default (300 s +
  0.05 s per coil element, 320 to 340 s here) is smaller; a larger budget is what resolves some hard designs.
- tol 3e-15 (torux's default): the refinement's off-grid |B.n|/|B| target, "the rounding floor".  1e-15 on the
  pilots: three of four tori bit-identical (the refinement plateaued and torux kept its best), the fourth slightly
  worse by the evaluator's precise tier (max 2.04e-15 against 1.88e-15, median 2.21e-16 against 2.30e-16) at 1.3 to
  2.6 times the wall time: it does not pay.
- dmin (coil quadrature floor): 0.2 x the least coil-to-design-surface distance, computed from the coils and the
  start surface (torux's rule for QUASR, and torux.load's for a file with a start block).  Every engine on the solve
  path sizes each coil's trapezoid rule from that coil's own distance to the region where the field is needed
  (spectral convergence, error about 1e-17); dmin is only a floor on that distance.  0.2 x the least distance never
  coarsens a rule; torux's library default (0.05 m) would coarsen it on the 7 of the 300 whose nearest coil is under
  0.1 m from the surface.
- verification: verify=True, trace_lines=0: its own acceptance (torux.verify.acceptance) is part of what torux calls
  a surface; traced field lines are an optional extra (minutes).
- stage policy and speed switches: the package's defaults (torux.analysis.POLICY, recorded in `settings.policy`;
  the recommended sets of torux.fast_design, fast_family, fast_newton, fast_kernels and fast_trace, recorded in
  `settings.fast_switches`): "an API call gets the whole solver"; the speed switches change wall time only (the same
  torus to rounding over 36 requests).  No TORUX_* variable but threads and caches may be set (checked below).
- device: CPU, exactly 4 threads in every pool (below).  torux's CPU field kernels use Neumaier-compensated sums
  (B against the exact sum 1.9e-16); its JAX kernels, the GPU path, use tree sums (4.0e-16).
- memory: torux needs far less than the 12 GiB per design: its largest worker peak was 4.1 GiB, except 1399833 at
  7.55 GiB with the warm-up.  Nothing in torux adapts to available memory, and nothing here limits it; the
  supervisor logs the worker's peak and the answer records it (`host.worker_peak_rss_gib`).

Pilot designs (six outside the 300; the evaluator's precise tier on the answers):
- 0224040: none, undetermined (design Newton failed; seeds, continuation probe, proxy band, bracket, continuation),
  2757 s.
- 0925611: surface, M, N = 48, 40, torux's off-grid B.n 7.5e-16, the evaluator's max / median 6.8e-16 / 1.2e-16,
  54 s; tol 1e-15 the identical torus.
- 1101401: surface, 160, 24, 1.1e-15, 8.3e-16 / 1.1e-16, 70 s; tol 1e-15 the identical torus.
- 1981226: surface, 80, 512, 2.8e-15, 1.88e-15 / 2.30e-16, 157 s; tol 1e-15: 112, 512, 2.04e-15 / 2.21e-16, 211 s.
- 2126458: none, undetermined (edge barrier hit its deadline in setup; measured with 3 threads), 2920 s.
- 2559727: surface, 56, 64, 2.2e-15, 2.35e-15 / 2.0e-16, 68 s; tol 1e-15 the identical torus.

Timing.  `wall_s` runs from reading the design's input file to its finished answer (the Source and design surface
built, dmin, torux.api.solve with its independent verification, the rz_fourier copy and the check points; the files
written after are not counted), in a worker that has already imported torux and run the two untimed warm-up solves
(pilot 0925611 with the run's settings: the design path, refinement and verification; pilot 0224040 with a 240 s
budget: the resonant path's stages), so the numba kernels and JAX programs that do not depend on the design are
compiled before the timed design.  What numba and JAX keep on disk of their compiled programs goes to a folder made
new for each design (work/quasr_<id>.cache), so the timed design reuses what its own worker's warm-up compiled and
nothing an earlier design left.  `wall_detail` holds the definition, torux's own clock and the solve alone.  The
supervisor counts the 3,600 s limit from the design's start; a worker stopped there, or one that dies, gives a
`none` answer saying so, metered by the supervisor (its meter covers the worker).

Threads.  Every pool is set to 4 before numpy and JAX load: numba (TORUX_THREADS, NUMBA_NUM_THREADS), BLAS and
OpenMP, torux's FFT workers (TORUX_FFT_WORKERS), and XLA's intra-op, compile and execute pools (NPROC, PJRT_NPROC;
measured in each worker: 4 threads each).  XLA's `foreach` pool (literal operations during compilation) follows the
cpus the process may use (the comparison gave each process 8 cpus for its 4 threads).  OpenMP and BLAS waiting are
at the libraries' defaults.  `host.process_threads_after_warmup` counts the worker's threads by pool.

The timing rule.  Each timed run of the comparison counted only if its threads waited for a CPU at most 5% of the
time they ran and other work used its CPUs for at most 5% of that time.  On Linux the warm-up and the design each
run under `fair_cores.Meter` (fair_cores.py beside this file; readings in `host.cores_warmup` and `host.cores`, its
sha256 in `settings.fair_cores_sha256`), and `timing_rule_met` says whether this timed run would have counted.  The
meter only measures.  Elsewhere nothing is metered and the answer says so.
"""
import os
import sys
import time

T_PROC = time.time()
# Environment variables of the package that change what the solver does (its stage policy, its work-reduction
# switches, its experimental stages): none may be set, so that every run is the package's defaults.  Thread counts and
# cache locations are set below by this runner.
_TORUX_ENV_ALLOWED = ("TORUX_THREADS", "TORUX_FFT_WORKERS", "TORUX_CACHE_DIR", "TORUX_JAX_CACHE_DIR", "TORUX_JAX_CACHE")
_foreign = sorted(k for k in os.environ if k.startswith("TORUX_") and k not in _TORUX_ENV_ALLOWED)
if _foreign:
    print(f"environment sets {', '.join(f'{k}={os.environ[k]!r}' for k in _foreign)}: these change torux's behaviour "
          f"away from the package's defaults; unset them (exit 1)", flush=True)
    sys.exit(1)
# 4 cores: every thread pool torux sizes (numba via TORUX_THREADS and NUMBA_NUM_THREADS, BLAS / OpenMP, its FFT
# workers via TORUX_FFT_WORKERS) is set to 4, before numpy loads, whatever cores the process is given
CORES = 4
# the cpus this process was given (the worker inherits them); recorded in host.cpus
CPUS = dict(cpus=sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# TORUX_THREADS: the package's one number for its pools (numba's set_threads, and its default for BLAS / OpenMP and
# the FFT workers); NPROC / PJRT_NPROC: the size of XLA's CPU thread pools (jaxlib reads them; else it takes every
# schedulable CPU)
for _v in ("TORUX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS", "TORUX_FFT_WORKERS", "NPROC", "PJRT_NPROC"):
    os.environ[_v] = str(CORES)
THREADS = dict(numba=CORES, blas_openmp=CORES, fft_workers=CORES, xla=CORES)
EXTRA_HOST = {}
# compilation caches: the supervisor gives each design's worker a folder made new for that design (below); a
# process that imports this file any other way keeps them out of the package's folder
_tmp = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_tmp, "torux-compare300-numba-cache"))
os.environ.setdefault("TORUX_JAX_CACHE_DIR", os.path.join(_tmp, "torux-compare300-jax-cache"))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
SOLVER, LABEL = "torux", "torux"
TORUX_VERSION = "0.1.0"            # the package the comparison ran
EVALUATOR = Path(os.environ.get("COMPARE300_EVALUATOR") or HERE.parent / "evaluator")
TIMING_RULE = ("a timed run counts only if its threads waited for a CPU at most 5% of the time they ran and other "
               "work used its CPUs for at most 5% of that time (or 0.2 s in all)")
WALL_LIMIT_S = 3600.0
WARMUPS = (("0925611", None), ("0224040", 240.0))   # (pilot design, budget_s: None for the run's own)
N_CHECK = 32
# the package's environment variables in effect for the solve (recorded in settings.torux_env)
TORUX_ENV = {k: os.environ[k] for k in sorted(os.environ) if k.startswith("TORUX_")}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def host_info():
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        import subprocess
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    out = dict(cpu_model=cpu, cores_used=CORES, threads=dict(THREADS), cpus=CPUS,
               cores_visible=affinity, machine_cores=os.cpu_count(), platform=platform.platform())
    try:
        import jax
        out["jax_devices"] = [str(d) for d in jax.devices()]
        out["gpu"] = [str(d) for d in jax.devices() if d.platform != "cpu"] or None
    except Exception as e:  # noqa: BLE001
        out["jax_devices"] = repr(e)
        out["gpu"] = None
    try:
        out["mem_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    out.update(EXTRA_HOST)
    return out


def process_threads():
    """Threads of this process by name (Linux): the evidence that every pool is 4 wide."""
    import collections
    import re
    names = collections.Counter()
    try:
        for t in os.listdir("/proc/self/task"):
            with open(f"/proc/self/task/{t}/comm") as fh:
                names[re.sub(r"[-_ ]?\d+$", "", fh.read().strip())] += 1
    except OSError:
        return None
    return dict(names)


def installed_version():
    """The installed package's version from its metadata (torux not imported), or None."""
    import importlib.metadata as md
    try:
        return md.version("torux")
    except md.PackageNotFoundError:
        return None


def versions():
    import importlib
    out = dict(torux_version=G["torux"].__version__ if G else installed_version(), python=platform.python_version())
    for m in ("numpy", "scipy", "jax", "jaxlib", "numba", "llvmlite", "mpmath", "gmpy2"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def coil_surface_distance(src, surf):
    """Least distance between the coils (2048 points each, from their coefficients) and the design surface (a
    256 x 256 grid of the full torus): the quantity torux's rule for QUASR takes its dmin from (0.2 x QUASR's
    min_coil2surface_dist), computed here from the coils and the start surface (the same computation as
    torux.load's for a file with a start block)."""
    import numpy as np
    from scipy.spatial import cKDTree
    from torux.io.surfaces import surface_grid
    r, _, _, _ = surface_grid(surf, nphi=256, ntheta=256, full_torus=True)
    tree = cKDTree(r.reshape(-1, 3))
    best = np.inf
    for p in src.parts:
        x, _ = p.points(np.arange(2048) / 2048)
        d, _ = tree.query(np.asarray(x).reshape(-1, 3), k=1)
        best = min(best, float(d.min()))
    return best


def rz_terms(T):
    """TorusState / GeneralTorus -> rz_fourier, one term per nonzero native coefficient (no arithmetic)."""
    import numpy as np
    general = hasattr(T, "rs")
    keys = ("rc", "rs", "zc", "zs") if general else ("rc", "zs")
    arrs = {k: np.asarray(getattr(T, k), dtype=np.float64) for k in keys}
    M, N, nfp = int(T.M), int(T.N), int(T.nfp)
    assert all(a.shape == (M + 1, 2 * N + 1) for a in arrs.values()), {k: a.shape for k, a in arrs.items()}
    out = dict(form="rz_fourier", m=[], n=[], **{k: [] for k in keys})
    for m in range(M + 1):
        for j in range(2 * N + 1):
            vals = {k: a[m, j] for k, a in arrs.items()}
            if all(v == 0.0 for v in vals.values()):
                continue
            out["m"].append(m)
            out["n"].append((j - N) * nfp)
            for k, v in vals.items():
                out[k].append(float(v))
    return out


def eval_rz(form, theta, phi):
    """Direct evaluation of the written rz_fourier form (a self-check of the copy; not the evaluator's code)."""
    import numpy as np
    m = np.asarray(form["m"], float)
    n = np.asarray(form["n"], float)
    a = np.outer(theta, m) - np.outer(phi, n)
    C, S = np.cos(a), np.sin(a)
    z = np.zeros(len(m))
    R = C @ np.asarray(form.get("rc", z)) + S @ np.asarray(form.get("rs", z))
    Z = C @ np.asarray(form.get("zc", z)) + S @ np.asarray(form.get("zs", z))
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], -1)


def check_points(T, design):
    import numpy as np
    rng = np.random.default_rng(int(design) + 20261107)
    th = 2 * np.pi * rng.random(N_CHECK)
    ph = 2 * np.pi * rng.random(N_CHECK)
    if hasattr(T, "rs"):
        from torux.torus_general import torus_points
    else:
        from torux.verify import torus_points
    x = np.asarray(torus_points(T, th, ph)[0], dtype=np.float64)
    return dict(theta=th.tolist(), phi=ph.tolist(), xyz=x.tolist(),
                evaluated_by=("torux.torus_general.torus_points" if hasattr(T, "rs") else "torux.verify.torus_points"))


def jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return repr(o)


def pick(d, keys):
    return {k: d.get(k) for k in keys if isinstance(d, dict) and k in d}


G = {}
FAST_MODULES = ("fast_design", "fast_family", "fast_newton", "fast_kernels", "fast_trace")


def load_torux():
    """Import torux and what a solve needs (once per process).  Another version than the comparison's: exit 1."""
    if G:
        return G
    t_imp = time.time()
    import importlib
    import torux
    if getattr(torux, "__version__", None) != TORUX_VERSION:
        print(f"torux {getattr(torux, '__version__', None)} is installed; the comparison ran torux {TORUX_VERSION} "
              f"(exit 1)", flush=True)
        sys.exit(1)
    from torux import analysis as AN
    from torux.io.fourier_json import load_fourier_json
    from torux.io.surfaces import SurfaceXYZTensorFourier, simson_surfaces
    fast = {m: importlib.import_module(f"torux.{m}") for m in FAST_MODULES}
    G.update(torux=torux, AN=AN, fast=fast, load_fourier_json=load_fourier_json,
             SurfaceXYZTensorFourier=SurfaceXYZTensorFourier, simson_surfaces=simson_surfaces,
             import_s=time.time() - t_imp)
    return G


WALL_DEFINITION = ("from reading the design's input file to its finished answer (the Source and design surface built, "
                   "dmin, torux.api.solve with its independent verification, the rz_fourier copy and the check "
                   "points), excluding the files written after")
INDEPENDENT_KEYS = ("ok", "ref", "bn_max", "bn_median", "npts", "bn_sup", "bn_sup_at", "flux", "flux_rel_err",
                    "precise_vs_fast64", "bn_precise_max", "bn_max_dd", "bn_median_dd", "bn_sup_dd", "bn_sup_dd_at",
                    "dd_vs_fast64_max", "skipped", "wall")


def run_design(a, coils, answer_path, record_path, wall_note=None, meter=None):
    """torux on one design: writes the answer and torux's full record; returns the answer.  meter: a started
    fair_cores.Meter, stopped when the answer is complete; its result goes into host.cores."""
    t0 = time.time()
    g = load_torux()
    np = __import__("numpy")
    torux, AN = g["torux"], g["AN"]
    load_fourier_json, SurfaceXYZTensorFourier, simson_surfaces = (g["load_fourier_json"], g["SurfaceXYZTensorFourier"],
                                                                   g["simson_surfaces"])
    import_s = g["import_s"]
    coils_path = Path(coils).resolve()
    d = json.loads(coils_path.read_text())
    design = d["id"]
    flux = float(d["requested_toroidal_flux_Wb"])
    print(f"design {design}: torux {torux.__version__} (the installed package), flux {flux!r} Wb, budget "
          f"{a.budget} s, tol {a.tol}, dmin rule {a.dmin}, surfaces {a.surfaces}", flush=True)

    # the Source as torux.load builds it from this file (torux.load._load_json), with the id "quasr/<id>"
    src = load_fourier_json(str(coils_path), source_id=f"quasr/{design}", request_from_file=True,
                            request_provenance="the file's requested_toroidal_flux_Wb")
    src.meta.pop("database_metadata", None)          # nothing of the database row reaches the solver
    src.meta["kind"] = "stellarator"                 # torux.load's default kind
    assert src.request["flux_Wb"] == flux
    st = d["start"]["surface"]
    start_surf = SurfaceXYZTensorFourier(st["nfp"], st["stellsym"], st["mpol"], st["ntor"], np.asarray(st["x"], float),
                                         quadpoints_phi=st["quadpoints_phi"], quadpoints_theta=st["quadpoints_theta"],
                                         name="start")
    surfs = [start_surf]                             # the start block's surface only (a.surfaces == "start")
    src._design_surfaces = surfs                     # where torux.api.solve takes a source's design surfaces from
    dist = coil_surface_distance(src, start_surf)
    if a.dmin == "campaign":
        src.meta["dmin"] = 0.2 * dist
    dmin = src.meta.get("dmin", 0.05)

    req = torux.Request(source=src, flux=flux, budget_s=a.budget, tol=a.tol, verify=True, trace_lines=0,
                        gradients=False)
    t_s = time.time()
    res = torux.api.solve(req)
    solve_s = time.time() - t_s
    rec = res.record or {}
    ev = res.evidence or {}
    indep = ev.get("independent") or {}
    print(f"design {design}: torux status {res.status}, path {rec.get('path')}, solve {solve_s:.1f} s, "
          f"independent ok {indep.get('ok')}", flush=True)

    T = res.surface
    verified = res.status == "surface" and T is not None and indep.get("ok") is True
    # the package names a surface its independent verification did not confirm 'surface_unverified' with
    # evidence['unverified'] (torux.api._truthful); such an outcome is written as below.  A 'surface_unverified'
    # without evidence['unverified'] is the analysis's own (a torus that failed torux's own surface criteria) and
    # is kept as it is.
    api_unverified = res.status == "surface_unverified" and "unverified" in ev
    if (res.status == "surface" and not verified) or api_unverified:
        solver_status = ("surface, rejected by its independent verification" if indep.get("ok") is False
                         else f"surface, independent verification not done ({indep.get('skipped', 'no record')})")
    else:
        solver_status = res.status
    reason = None
    if not verified:
        cert = ev.get("certificate") or rec.get("certificate") or {}
        reasons = {
            "no_surface": ("no surface: torux certifies that no flux surface encloses the requested flux "
                           f"({rec.get('path')}; certificate {cert.get('kind', cert.get('type', ''))})"),
            "undetermined": ("not converged: no verified surface and no certificate within the budget "
                             f"({rec.get('resonant_path') or rec.get('path')})"),
            "timeout": "out of time: torux's budget ran out before an outcome",
            "no_axis": "not converged: torux found no magnetic axis to start from",
            "error": f"error: {(rec.get('error') or ev.get('error') or '')[:300]}",
            "range": "not converged: torux returned a range, not the surface at the request",
        }
        reason = reasons.get(res.status, f"not converged: torux outcome {solver_status}")
        if res.status == "surface" or api_unverified:
            reason = "not converged: " + solver_status

    fast = {m: sorted(mod.active()) for m, mod in g["fast"].items()}
    answer = dict(design=design, solver="torux", solver_label="torux", versions=versions(),
                  settings=dict(
                      entry="torux.api.solve(torux.Request(source, flux, budget_s, tol, verify=True, trace_lines=0, "
                            "gradients=False)); source = torux.io.fourier_json.load_fourier_json(inputs/coils file) "
                            "as torux.load builds it, database_metadata dropped; design surfaces "
                            "(Source._design_surfaces) = [the start block's surface]",
                      budget_s=a.budget, tol=a.tol, verify=True, trace_lines=0, gradients=False,
                      dmin_rule=("0.2 x least coil-to-design-surface distance (torux's rule for QUASR; "
                                 "torux.load's rule for a file with a start block)"
                                 if a.dmin == "campaign" else "torux FieldEngine default 0.05 m"),
                      dmin_m=dmin, coil_to_surface_m=dist, design_surfaces=a.surfaces, iota_hints=None,
                      policy=jsonable(AN.POLICY), fast_switches=fast, torux_env=dict(TORUX_ENV),
                      wall_limit_s=a.limit_s, cores=CORES, mem_gb=a.mem_gb,
                      coils_file=coils_path.relative_to(HERE).as_posix() if coils_path.is_relative_to(HERE)
                      else coils_path.name, coils_sha256=sha256(coils_path)),
                  host=host_info(), requested_flux_Wb=flux,
                  status="surface" if verified else "none", solver_status=solver_status, reason=reason,
                  wall_s=None, wall_detail=dict(definition=WALL_DEFINITION + (("; " + wall_note) if wall_note else ""),
                                                process_s=None, import_s=import_s, solve_s=solve_s,
                                                torux_wall_s=res.wall_s, torux_budget_s=res.budget_s),
                  surface=None, check_points=None, solver_reported=None, native=None)

    s = rec.get("surface") if isinstance(rec.get("surface"), dict) else {}
    sr = dict(status=res.status, path=rec.get("path"), iota=float(res.iota) if np.isfinite(res.iota) else None,
              budget_s=res.budget_s, wall_s=res.wall_s,
              budget_exceeded_s=ev.get("budget_exceeded_s"), status_at_deadline=ev.get("status_at_deadline"))
    if ev.get("unverified"):
        sr["unverified"] = ev.get("unverified")
    if T is not None:
        sr.update(flux_Wb=float(T.flux), M=int(T.M), N=int(T.N),
                  bn_off_grid=s.get("bn_off"), residual_m=s.get("res"), quad_err_half=s.get("quad_err_half"),
                  guard=jsonable(rec.get("guard")), topology=jsonable(s.get("topology")),
                  refine_hist=jsonable(rec.get("refine_hist")), refine_stop=rec.get("refine_stop"),
                  independent_verification=jsonable(pick(indep, INDEPENDENT_KEYS)))
    if ev.get("certificate") or rec.get("certificate"):
        cert = ev.get("certificate") or rec.get("certificate")
        sr["certificate"] = jsonable({k: v for k, v in cert.items()
                                      if not isinstance(v, (list, dict)) or k in ("holds_request",)})
    if rec.get("resonant_path"):
        sr["resonant_path"] = rec.get("resonant_path")
    if rec.get("policy"):
        sr["policy_stages"] = jsonable((rec.get("policy") or {}).get("stages"))
    if rec.get("near_coil"):
        sr["near_coil"] = jsonable({k: v for k, v in rec["near_coil"].items() if not isinstance(v, (list, dict))})
    answer["solver_reported"] = jsonable(sr)

    if verified:
        phi0 = float(getattr(T, "phi0", 0.0))
        assert phi0 == 0.0, f"phi0 = {phi0}: the copy needs phi0 = 0"
        form = rz_terms(T)
        cp = check_points(T, design)
        self_check = float(np.max(np.abs(eval_rz(form, np.asarray(cp["theta"]), np.asarray(cp["phi"]))
                                          - np.asarray(cp["xyz"]))))
        print(f"design {design}: {len(form['m'])} terms, M {T.M} N {T.N}, check points self-check {self_check:.2e} m",
              flush=True)
        if not self_check < 1e-12:
            raise RuntimeError(f"conversion self-check {self_check:.3e} m")
        answer["surface"] = form
        answer["check_points"] = cp
        answer["native"] = dict(
            library=f"torux {torux.__version__} (the installed package)",
            cls=type(T).__module__ + "." + type(T).__name__,
            convention=("R(theta, phi) = sum_{m=0..M, n=-N..N} rc[m][n + N] cos(m theta - n nfp (phi - phi0)) "
                        "(+ rs sin for a general torus); Z = sum zs[m][n + N] sin(...) (+ zc cos); theta the "
                        "straight-field-line poloidal angle, phi the cylindrical angle; rebuild with "
                        "torux.torus.TorusState(rc, zs, iota, flux, M, N, nfp)"),
            M=int(T.M), N=int(T.N), nfp=int(T.nfp), phi0=phi0, iota=float(T.iota), flux=float(T.flux),
            **{k: np.asarray(getattr(T, k)).tolist() for k in (("rc", "rs", "zc", "zs") if hasattr(T, "rs") else ("rc", "zs"))})
        answer["solver_reported"]["check_points_self_check_m"] = self_check

    wall_s = time.time() - t0
    answer["wall_s"] = wall_s
    import fair_cores
    answer["host"]["cores"] = meter.stop() if meter is not None else None
    answer["settings"]["fair_cores_sha256"] = fair_cores.SHA256
    answer["wall_detail"]["process_s"] = time.time() - T_PROC
    Path(record_path).write_text(json.dumps(jsonable(dict(design=design, status=res.status, record=rec,
                                                       evidence=ev, budget_s=res.budget_s, wall_s=res.wall_s))) + "\n")
    tmp = Path(answer_path).with_suffix(".tmp")
    tmp.write_text(json.dumps(answer, allow_nan=False) + "\n")
    tmp.replace(answer_path)
    print(f"design {design}: answer {answer['status']} ({solver_status}), wall {wall_s:.1f} s", flush=True)
    return answer


# ---------------------------------------------------------------------------------------------------------
# the worker: the warm-up, then the timed design, in one process

def meter_start(why):
    """fair_cores' Meter, started; None where it cannot measure (it reads Linux's /proc), the reason in `why`."""
    import fair_cores
    try:
        return fair_cores.Meter().start()
    except (RuntimeError, OSError) as e:
        why.append(str(e))
        return None


def worker(a):
    # the worker dies with its supervisor: nothing of it may stay behind on the cpus (an orphan worker is other
    # work to the next run's meter)
    try:
        import ctypes
        import signal as _signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(_signal.SIGKILL))     # PR_SET_PDEATHSIG
        if os.getppid() == 1:
            sys.exit(1)
    except OSError:
        pass
    import fair_cores
    out, design = Path(a.out), a.design
    work = out / "work"
    (work / "warmup").mkdir(parents=True, exist_ok=True)
    not_metered = []

    t_w = time.time()
    load_torux()
    meter = meter_start(not_metered)
    warm = []
    for pid, budget in (() if a.no_warmup else WARMUPS):
        b = argparse.Namespace(**vars(a))
        if budget is not None:
            b.budget = budget
        t = time.time()
        ans = run_design(b, HERE / "inputs" / "pilot" / f"quasr_{pid}.json",
                         work / "warmup" / f"quasr_{pid}.json", work / "warmup" / f"quasr_{pid}.record.json")
        warm.append(dict(design=pid, budget_s=b.budget, wall_s=time.time() - t, solver_status=ans["solver_status"]))
    cores_warmup = meter.stop() if meter is not None else None
    threads = process_threads()
    ok = fair_cores.fair(cores_warmup) if cores_warmup is not None else None
    (work / f"quasr_{design}.warmup_meter.json").write_text(json.dumps(dict(cores_warmup=cores_warmup, fair=ok,
                                                                            warmups=warm, threads=threads)) + "\n")
    rule = ("not metered" if cores_warmup is None else
            f"wait_fraction {cores_warmup['wait_fraction']:.4f}, busy_cores {cores_warmup['busy_cores']:.2f}, "
            f"timing rule met {ok}")
    print(f"warm-up done in {time.time() - t_w:.1f} s, {rule}: {warm}; process threads by name: {threads}",
          flush=True)
    if warm:
        note = (f"in a process that had imported torux and run untimed warm-up solves on pilot designs "
                f"{', '.join(w['design'] for w in warm)} ({time.time() - t_w:.0f} s, fair_cores wait_fraction "
                + (f"{cores_warmup['wait_fraction']:.4f})" if cores_warmup is not None else "not metered)"))
    else:
        note = "in a process that had run no warm-up: imports and compilation included"
    EXTRA_HOST.update(process_threads_after_warmup=threads, warmup=warm, cores_warmup=cores_warmup)

    ans_path = out / "answers" / f"quasr_{design}.json"
    rec = out / "logs" / f"quasr_{design}.record.json"
    staging = work / f"quasr_{design}.staging.json"
    start = work / f"quasr_{design}.start"
    start.write_text(json.dumps(dict(t=time.time(), pid=os.getpid())) + "\n")
    print(f"# design {design}; threads {THREADS}; worker pid {os.getpid()}; {note}", flush=True)
    m = meter_start(not_metered)
    try:
        run_design(a, Path(a.input), staging, rec, wall_note=note, meter=m)
    except Exception as e:  # noqa: BLE001  (the design's answer says so)
        traceback.print_exc()
        d = json.loads(Path(a.input).read_text())
        t0 = json.loads(start.read_text())["t"]
        host = host_info()
        host["cores"] = m.stop() if m is not None else None
        out_ = dict(design=design, solver="torux", solver_label="torux", versions=versions(),
                    settings=dict(vars(a), fair_cores_sha256=fair_cores.SHA256), host=host,
                    requested_flux_Wb=float(d["requested_toroidal_flux_Wb"]), status="none",
                    solver_status="error (harness)", reason=f"error: {e!r}"[:400], wall_s=time.time() - t0,
                    surface=None, check_points=None, solver_reported=None, native=None)
        staging.write_text(json.dumps(out_) + "\n")
    ans = json.loads(staging.read_text())
    cores = ans["host"].get("cores")
    ans["timing_rule"] = TIMING_RULE
    ans["timing_rule_met"] = bool(fair_cores.fair(cores)) if cores is not None else None
    if cores is None:
        ans["not_metered"] = not_metered[-1] if not_metered else "no reading of the timed run"
    staging.write_text(json.dumps(ans, allow_nan=False) + "\n")
    staging.replace(ans_path)
    rule = ("not metered" if cores is None else
            f"wait_fraction {cores['wait_fraction']:.4f}, timing rule met {ans['timing_rule_met']}")
    print(f"{design}: {ans['status']} wall {ans['wall_s']:.1f} s, {rule}", flush=True)
    forms_check(ans)


def forms_check(ans):
    """The evaluator's forms.check_points on the answer's own check points (a check of the written surface only)."""
    if not ans.get("surface"):
        return
    try:
        sys.path.insert(0, str(EVALUATOR))
        import forms
        print(f"# forms.check_points reproduces the check points to "
              f"{forms.check_points(ans['surface'], ans['check_points']):.2e} m", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# check points not compared: no forms.py in {EVALUATOR} ({type(e).__name__}: {e})", flush=True)


# ---------------------------------------------------------------------------------------------------------
# the supervisor

def none_answer(a, wall, solver_status, reason):
    d = json.loads(Path(a.input).read_text())
    cpu = None
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    options = ["--budget", str(a.budget), "--tol", str(a.tol), "--dmin", a.dmin, "--mem-gb", str(a.mem_gb)]
    return dict(design=a.design, solver="torux", solver_label="torux",
                versions=dict(torux_version=installed_version(), python=sys.version.split()[0]),
                settings=dict(torux_answer_options=options, wall_limit_s=a.limit_s),
                host=dict(cpu_model=cpu, cores=4, threads=4, gpu=None),
                requested_flux_Wb=float(d["requested_toroidal_flux_Wb"]), status="none", solver_status=solver_status,
                reason=reason, wall_s=wall, surface=None, check_points=None, solver_reported=None, native=None)


def main():
    ap = argparse.ArgumentParser(
        description="torux 0.1.0 on one design: the magnetic surface of its coils that encloses the requested "
                    "flux, written as DIR/answers/quasr_<id>.json with its log and torux's record in DIR/logs/.")
    ap.add_argument("design", help="7-digit design id (inputs/quasr_<id>.json beside this file)")
    ap.add_argument("--pilot", action="store_true", help="read the design from inputs/pilot/")
    ap.add_argument("--out", default=str(Path.cwd() / "compare300-out" / SOLVER),
                    help="answers/ and logs/ go here (default ./compare300-out/torux)")
    ap.add_argument("--budget", type=float, default=3300.0, help="torux's stated budget, s (Request.budget_s)")
    ap.add_argument("--limit-s", type=float, default=WALL_LIMIT_S,
                    help="wall-time limit per design, s: the supervisor stops torux there")
    ap.add_argument("--mem-gb", type=float, default=12.0, help="memory per design, GiB: recorded (torux needs less)")
    ap.add_argument("--tol", type=float, default=3e-15, help="Request.tol: the off-grid |B.n|/|B| target")
    ap.add_argument("--dmin", default="campaign", choices=("campaign", "api"),
                    help="campaign: 0.2 x the least coil-to-design-surface distance (also what torux.load sets for "
                         "such a file); api: the field engine's library default, 0.05 m")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override one setting: budget, limit_s, mem_gb, tol or dmin")
    ap.add_argument("--no-warmup", action="store_true", help="no untimed warm-up: wall_s then includes compilation")
    ap.add_argument("--force", action="store_true", help="run even if the answer exists")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--input", help=argparse.SUPPRESS)
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in ("budget", "limit_s", "mem_gb", "tol", "dmin"):
            ap.error(f"--set {k}: not one of budget, limit_s, mem_gb, tol, dmin")
        setattr(a, k, v if k == "dmin" else float(v))
    a.surfaces = "start"
    a.input = a.input or str(HERE / "inputs" / ("pilot" if a.pilot else "") / f"quasr_{a.design}.json")
    if a.worker:
        worker(a)
        return
    if installed_version() is None:
        print("torux is not installed in this Python (pip install . in the repository)", flush=True)
        sys.exit(1)
    if installed_version() != TORUX_VERSION:
        print(f"torux {installed_version()} is installed; the comparison ran torux {TORUX_VERSION}", flush=True)
        sys.exit(1)
    if not Path(a.input).exists():
        print(f"no input file {a.input}", flush=True)
        sys.exit(1)
    import resource
    import subprocess
    import fair_cores
    out = Path(a.out)
    for sub in ("answers", "logs", "work"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ans = out / "answers" / f"quasr_{a.design}.json"
    if ans.exists() and not a.force:
        print(f"{a.design}: {ans} exists (--force to run again)", flush=True)
        return
    ans.unlink(missing_ok=True)
    start = out / "work" / f"quasr_{a.design}.start"
    start.unlink(missing_ok=True)
    log = out / "logs" / f"quasr_{a.design}.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), a.design, "--worker", "--input", a.input, "--out", str(out),
           "--budget", repr(a.budget), "--limit-s", repr(a.limit_s), "--mem-gb", repr(a.mem_gb), "--tol", repr(a.tol),
           "--dmin", a.dmin] + (["--no-warmup"] if a.no_warmup else [])
    # the programs numba and JAX compile are kept on disk in a folder made new for this design: the timed design
    # reuses what its own worker's warm-up compiled and nothing an earlier design left behind
    import shutil
    cache = out / "work" / f"quasr_{a.design}.cache"
    shutil.rmtree(cache, ignore_errors=True)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache / "numba"), TORUX_JAX_CACHE_DIR=str(cache / "jax"))
    not_metered = []
    with open(log, "w") as fh:
        fh.write(f"# torux {TORUX_VERSION} on design {a.design}: {' '.join(cmd)}\n")
        fh.flush()
        # the same process group: a stop reaches it
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        killed = False
        metering, meter = False, None     # whether this supervisor's meter follows the design, and the meter
        while True:
            rc = p.poll()
            if rc is not None:
                break
            cur = start.exists() and not ans.exists()
            if cur != metering:
                if meter is not None:
                    meter.stop()
                metering, meter = cur, (meter_start(not_metered) if cur else None)
            if cur:
                t0 = json.loads(start.read_text())["t"]
                if time.time() - t0 > a.limit_s:
                    p.kill()                                   # torux starts threads, no processes
                    p.wait()
                    rc, killed = None, True
                    break
            time.sleep(2)
        parent_cores = meter.stop() if meter is not None else None
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
        fh.write(f"# worker exit {rc}{' (killed: past the limit)' if killed else ''}; peak resident memory of the "
                 f"worker {peak:.2f} GiB of the {a.mem_gb} GiB allowed\n")
    if rc not in (0, None) and not start.exists():
        # the worker died before it started the design (an import or set-up failure, not a design's outcome)
        print(f"{a.design}: the worker ended (exit {rc}) before the design started; see {log}", flush=True)
        sys.exit(1)
    if start.exists() and not ans.exists():
        t0 = json.loads(start.read_text())["t"]
        if killed:
            res = none_answer(a, time.time() - t0, "no outcome: stopped at the wall-time limit",
                              f"out of time: the harness stopped torux at {a.limit_s:.0f} s")
        else:
            res = none_answer(a, time.time() - t0, f"error: worker exit {rc}",
                              "error: the process ended without an answer")
        # the run's meter (this supervisor's, from the design's start, covering the worker) and the warm-up's
        res["host"].update(cores=parent_cores, cores_used=4, cpus=CPUS,
                           threads=dict(numba=4, blas_openmp=4, fft_workers=4, xla=4))
        wm = out / "work" / f"quasr_{a.design}.warmup_meter.json"
        if wm.exists():
            w = json.loads(wm.read_text())
            res["host"].update(cores_warmup=w["cores_warmup"], process_threads_after_warmup=w.get("threads"),
                               warmup=w.get("warmups"))
        res["settings"]["fair_cores_sha256"] = fair_cores.SHA256
        res["wall_detail"] = dict(definition="from the design's start (its input read) to the stop of its worker, "
                                             "in a worker that had imported torux and run untimed warm-up solves "
                                             "on pilot designs 0925611, 0224040; metered by the supervisor process "
                                             "(its fair_cores.Meter covers the worker)")
        res["timing_rule"] = TIMING_RULE
        res["timing_rule_met"] = bool(fair_cores.fair(parent_cores)) if parent_cores is not None else None
        if parent_cores is None:
            res["not_metered"] = not_metered[-1] if not_metered else "no reading of the timed run"
        with open(log, "a") as lf:
            lf.write(f"# {res['reason']} after {res['wall_s']:.1f} s\n")
        ans.write_text(json.dumps(res) + "\n")
    start.unlink(missing_ok=True)
    res = json.loads(ans.read_text())
    res["host"]["worker_peak_rss_gib"] = peak
    ans.write_text(json.dumps(res, allow_nan=False) + "\n")
    print(f"{a.design} {res['status']} {res['wall_s']:.1f}s {res.get('reason') or ''}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
