"""torux: the vacuum magnetic field and flux surfaces of any external source for toroidal confinement.

    import torux

    s = torux.solve("quasr:954")                       # a QUASR design: its own request
    s = torux.solve("coils.json", flux=0.0691)         # the surface enclosing this full toroidal flux, Wb
    s = torux.solve("coils.json", minor_radius=0.12)   # or of this minor radius, m, or volume=, m^3
    s = torux.solve("coils.json", largest=True)        # or the largest torux can verify
    print(s)                                           # what it found, how it was checked, or why there is none
    s.iota, s.volume, s.minor_radius, s.profile, s.section(phi)
    s.save("input.mine")                               # VMEC input. Also *.json, *.vtk
    s.plot("surface.png"), s.boozer()
    torux.check("coils.json", "wout_mine.nc")          # how close another code's surface is to a flux surface
    torux.many(files, table="answers.csv")             # many designs at once, each in its own process

    s.status        'surface' | 'no_surface' | 'range' | 'undetermined' | 'no_axis' | 'timeout' | 'error', or
                    'surface_unverified' / 'range_unverified'

A request is answered 'surface' only when the solver's acceptance and an independent evaluator whose field code
shares none with it both agree, and 'no_surface' only with a certificate or a proof.  Everything else is
'undetermined': never a guess.

Environment:
    TORUX_THREADS     threads of every pool, BLAS included even when numpy or scipy was imported first (default:
                      four, or the cores available if fewer)
    TORUX_CACHE_DIR   where JAX's compiled programs are kept between processes (default ~/.cache/torux).
                      TORUX_JAX_CACHE=0 turns that cache off. QUASR downloads are kept there too.  numba keeps
                      its own cache beside the package.
"""
import os as _os

__version__ = "0.1.4"


def _threads():
    try:
        n = int(_os.environ.get("TORUX_THREADS") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        try:
            avail = len(_os.sched_getaffinity(0))
        except AttributeError:
            avail = _os.cpu_count() or 1
        n = max(1, min(4, avail))
    return n


# One number for every pool, set before numpy loads its BLAS: an uncapped BLAS on a large or shared machine makes
# the dense solves many times slower.
_os.environ["TORUX_THREADS"] = str(_threads())
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_v, _os.environ["TORUX_THREADS"])


def _cap_loaded_blas():
    """A BLAS loaded before torux (numpy or scipy imported first) started with as many threads as the machine has,
    and the last bits of a solve depend on that count: each such pool is capped to TORUX_THREADS."""
    import sys
    if not any(m in sys.modules for m in ("numpy", "scipy")):
        return                            # nothing loaded yet: the variables above hold every pool
    from threadpoolctl import ThreadpoolController
    n = int(_os.environ["TORUX_THREADS"])
    for lib in ThreadpoolController().lib_controllers:
        if lib.user_api == "blas" and (lib.num_threads or 0) > n:
            lib.set_num_threads(n)


_cap_loaded_blas()


def _cache_dir():
    d = _os.environ.get("TORUX_CACHE_DIR")
    if not d:
        base = _os.environ.get("XDG_CACHE_HOME") or _os.path.join(_os.path.expanduser("~"), ".cache")
        d = _os.path.join(base, "torux")
    return d


try:
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    if _os.environ.get("TORUX_JAX_CACHE", "1") != "0":
        _d = _os.environ.get("TORUX_JAX_CACHE_DIR") or _os.environ.get("JAX_COMPILATION_CACHE_DIR") \
            or _os.path.join(_cache_dir(), "jax")
        try:
            _os.makedirs(_d, exist_ok=True)
        except OSError:
            _d = None                    # no writable cache directory: compile without the persistent cache
        if _d:
            _jax.config.update("jax_compilation_cache_dir", _d)
        try:
            _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        except Exception:  # noqa: BLE001
            pass
except Exception:  # noqa: BLE001
    pass

from .source import (AxisymPolygons, Dipoles, FourierCurves, PolylineCurves, Rings, Segments,  # noqa: E402
                     Source, ToroidalField)
from .sources import load  # noqa: E402
from .api import Request, Result  # noqa: E402
from .answer import Answer, solve  # noqa: E402
from .check import Check, check  # noqa: E402
from .batch import many  # noqa: E402

__all__ = ["solve", "load", "check", "many", "Answer", "Check", "Request", "Result", "Source", "FourierCurves",
           "PolylineCurves", "Segments", "Dipoles", "Rings", "AxisymPolygons", "ToroidalField", "__version__"]
