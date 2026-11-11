"""Work reductions of the design path: switches.

Each switch replaces a piece of work by a cheaper one that gives the same torus.  The recommended set is on by
default.  TORUX_FAST_DESIGN overrides it per process: a comma-separated list of names, "all" for the recommended set,
"every" with the experimental ones too, "none" for none.  In code: ``set_opts(...)`` / the ``opts(...)`` context
manager.

  angle      torux.design.sfl_guess_from_boozer: the angle-map Newton stops once its quadratic convergence has
             taken the error below rounding (its fixed stop test, 1e-15, lies under the rounding floor of angles near
             2 pi, 8.9e-16 - 8e-15).  Same guess to rounding; no field evaluations.
  probe      torux.analysis.analyse_design: the design guess at the start resolution is the start-resolution
             probe's own torus truncated (the probe's grid is finer: same guess up to the probe's smaller
             aliasing) instead of a second angle-map solve.
  floor      Newton (dense and Krylov) in the refinement stops at the arithmetic floor: the Galerkin residual
             below the tolerance and the flux equation within FLUX_FLOOR of its own rounding (relative flux
             error <= 10 x the tolerance).  Waiting for "no new best in two iterations" instead spends 2-4 field
             evaluations per level bouncing between 1 and 7 ulps of the flux.
  quad       the a posteriori quadrature estimate (TorusProblem.quadrature_check) taken from the final off-grid
             check's own evaluation (same points, same engine, same B; the half rule's sum comes with every
             evaluation) instead of evaluating the check grid again.  Bit-identical result.
  fluxsign   the design solve's flux sign (all that is used of the design guess's flux when the request is
             given) from one planar section (64 points) instead of the whole grid.
  predict    the final Newton iteration of a refinement level is confirmed by its Newton-Kantorovich
             prediction instead of another field evaluation: after a step |du| whose quadratic remainder
             bound is below the arithmetic floor (and a linear residual below it) the iterate is returned
             unevaluated; the independent off-grid check then decides whether it is accepted.
  krylov     the design solve and its raises by Newton-Krylov at every size (the dense Newton remains the
             fallback of a failed Krylov solve): below 2500 unknowns the dense LU (n^3 / 3 flops a step) is
             the largest share of the design phase's time.  Same torus to rounding.
  npnodes    every FieldEngine builds its Fourier coils' trapezoid nodes with numpy (field.engine.fourier_nodes_np)
             instead of JAX's op-by-op fourier_nodes, which compiles once per new node count.  Bit-identical to the
             JAX nodes wherever checked.
  nodes      (experimental) the refinement's lean engine takes each coil's node count from the coil's measured
             analyticity strip on the torus (its own N and N/2 rules at the torus's closest points) instead of the
             strip model 0.8 d / max|r'| with half the distance (torux.field.engine.measured_nodes).  The per-coil
             measurement costs more time than the field work it saves, and its quadrature estimate can come within
             3x of the acceptance bar.
  jump       (experimental) the design solve's resolution raises (to the continuation level bn_cont) by the spectral
             prediction (torux.family.predict_resolution to bn_cont / 10, at most 2x a raise) instead of 1.25x
             ladder steps.  The design torus may end at another resolution; the refinement then predicts the
             final one from it, so the final (M, N) can differ.
"""
from __future__ import annotations

import contextlib
import os

NAMES = ("angle", "probe", "floor", "quad", "fluxsign", "predict", "nodes", "krylov", "jump", "npnodes")
# "all": the recommended set (same torus to rounding at the same resolution); "every": with the experimental ones.
EXPERIMENTAL = ("jump", "nodes")
RECOMMENDED = tuple(n for n in NAMES if n not in EXPERIMENTAL)

_on = set()


def _from_env():
    """The switches on: unset TORUX_FAST_DESIGN means the recommended set; TORUX_FAST_DESIGN=none (or off, base) turns
    every switch off."""
    v = os.environ.get("TORUX_FAST_DESIGN")
    if v is None or not v.strip():
        return set(RECOMMENDED)
    v = v.strip()
    if v in ("none", "off", "base"):
        return set()
    if v == "all":
        return set(RECOMMENDED)
    if v == "every":
        return set(NAMES)
    out = {s.strip() for s in v.split(",") if s.strip()}
    bad = out - set(NAMES)
    if bad:
        raise ValueError(f"TORUX_FAST_DESIGN: unknown switches {sorted(bad)} (known: {NAMES})")
    return out


_on = _from_env()


def on(name):
    return name in _on


def set_opts(names):
    """Replace the active switches (an iterable of names, or "all")."""
    global _on
    names = set(RECOMMENDED) if names == "all" else set(NAMES) if names == "every" else set(names)
    bad = names - set(NAMES)
    if bad:
        raise ValueError(f"unknown switches {sorted(bad)}")
    _on = names


def active():
    return sorted(_on)


@contextlib.contextmanager
def opts(names):
    """Run the block with exactly these switches on."""
    global _on
    old = set(_on)
    set_opts(names)
    try:
        yield
    finally:
        _on = old
