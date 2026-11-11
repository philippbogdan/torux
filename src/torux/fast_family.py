"""Work reductions of the path from the axis: switches for the family of tori, the engine's distance search and
the topology test.

The recommended set is on by default (the same status and bit-identical tori with and without it).  TORUX_FAST_FAMILY
overrides it per process: a comma-separated list of names, "all" for the recommended set, "every" with the
experimental ones, "none" for none.  In code: ``set_opts(...)`` / the ``opts(...)`` context manager.  Independent of
torux.fast_design.

  cpredict   the family from the axis (torux.analysis.analyse: the start torus, the continuation steps and their
             raises) carries the Newton-Kantorovich state of torux.fast_design 'predict': a solve stops after a full
             step whose bound r_lin + 100 C |du|^2 (C measured on earlier steps) is below the tolerance, without
             evaluating the field at the new iterate.  The returned iterate is the one the evaluated Newton would
             have returned whenever its next residual is below the tolerance; the off-grid check still decides the
             step.  Not the refinement: refinement levels end a little above the 3e-15 target so often that a
             predicted level is solved again evaluated, more work than it saves.
  kdfast     FieldEngine's per-coil distance to the region by an exact coarse-to-fine search (every 16th of the
             4096 coil samples first, then only the samples that can still hold the minimum): the same minimum,
             hence the same node counts, about 10x fewer tree queries.
  topofast   analysis.torus_topology tests only the chord pairs whose bounding boxes overlap (a crossing needs
             that): the same crossings, ~1/30 of the orientation tests.
  cdiverge   (experimental) a continuation step's Newton (Continuation.run: the step and its plateau retry) stops
             once it has stayed diverged for DIVERGE_RUN consecutive iterates -- each residual above DIVERGE_ABS x
             R00, or each above DIVERGE_FACTOR x its first residual and DIVERGE_FLOOR x R00.  The step then fails as
             after maxit iterations (ds halved).  A heuristic (see diverged): damped Newton can come back from
             residuals of order R00.
  cquad      (experimental) the continuation's predictor quadratic in s through the last three tori (Lagrange)
             instead of linear through two: its coefficient error is 0.02-0.3 of the linear one's away from
             resonances and about equal next to a stall; the corrector is the same Newton, so the accepted tori are
             the same to the Newton tolerance.
  cpgrid     (experimental, with cpredict) the Newton-Kantorovich prediction measures a step by its largest grid
             displacement (max |dR|, |dZ| over the grid, + |d iota|) instead of |du|_1, which overstates the steps
             that fill many new modes (a raise's) by up to ~sqrt(n): such steps are then predicted too.
  rstop      (experimental) refine_to_limit ends once the best off-grid B.n is at the acceptance bar (1e-14,
             torux.analysis.BN_SURFACE) and a raise gained less than 2x, instead of waiting for two such raises: the
             3e-15 target lies at the B.n floor of many tori, and the levels climbed after reaching the bar move
             B.n by a few per cent at most.  The accepted torus may then be one ladder level lower (same status; it
             still passes the bar and the independent check).
  cudafil    (experimental; machines with a CUDA card) the Fourier-filament sums by the hand-written float64 kernel
             torux.field.cuda_fil (NVRTC + driver API), with the torus Newton on the host; everything else
             unchanged.  On an A30: 2.3-2.8e10 target-node pairs/s for B + grad B + A (17-21x the engine's JAX
             kernels on the same card), B and grad B within 2e-15 of the numba kernel per output column, A within
             2e-14; the solves' tori equal to the CPU path's to <= 3e-12.  Off by default:
             TORUX_FAST_FAMILY=all,cudafil turns it on.
"""
from __future__ import annotations

import contextlib
import os

NAMES = ("cpredict", "cdiverge", "kdfast", "topofast", "cudafil", "cquad", "cpgrid", "rstop")
EXPERIMENTAL = ("cdiverge", "cudafil", "cquad", "cpgrid", "rstop")
RECOMMENDED = tuple(n for n in NAMES if n not in EXPERIMENTAL)

DIVERGE_FACTOR = 100.0      # residual above this many times the step's first residual: diverged
DIVERGE_ABS = 0.05          # ... or above this fraction of the major radius (a torus that far from invariant)
DIVERGE_RUN = 5             # ... at each of this many consecutive evaluated iterates
DIVERGE_FLOOR = 1e-4        # (the factor rule only above this fraction of the major radius)


def _from_env():
    v = os.environ.get("TORUX_FAST_FAMILY")
    if v is None or not v.strip():
        return set(RECOMMENDED)
    v = v.strip()
    if v in ("none", "off", "base"):
        return set()
    if v == "all":
        return set(RECOMMENDED)
    if v == "every":
        return set(NAMES)
    out = set()
    for s in v.split(","):
        s = s.strip()
        if not s:
            continue
        out |= set(RECOMMENDED) if s == "all" else {s}
    bad = out - set(NAMES)
    if bad:
        raise ValueError(f"TORUX_FAST_FAMILY: unknown switches {sorted(bad)} (known: {NAMES})")
    return out


_on = _from_env()


def on(name):
    return name in _on


def active():
    return sorted(_on)


def set_opts(names):
    global _on
    names = set(RECOMMENDED) if names == "all" else set(NAMES) if names == "every" else set(names)
    bad = names - set(NAMES)
    if bad:
        raise ValueError(f"unknown switches {sorted(bad)}")
    _on = names


@contextlib.contextmanager
def opts(names):
    global _on
    old = set(_on)
    set_opts(names)
    try:
        yield
    finally:
        _on = old


def diverged(hist, R00):
    """The 'cdiverge' rule on a Newton history (|F| per evaluated iterate, first entry the start): the last
    DIVERGE_RUN residuals all above DIVERGE_ABS x R00, or all above DIVERGE_FACTOR x the start and DIVERGE_FLOOR x R00
    (or NaN).  A heuristic: on 893 recorded continuation-step Newton runs (560 converged) it stops 211 of the 333
    that did not converge and none that converged; with 4 instead of 5 it stops one that converges (residuals 2.1,
    3.1, 0.91, 0.12 R00, converged at the tenth iterate), with 3 four.  Damped Newton does come back from
    residuals of order R00, so the rule is experimental: it can only turn a converging step into a failed one (ds
    halved), never accept anything.
    """
    if len(hist) < 1:
        return False
    last = hist[-1]
    if not (last == last):          # NaN
        return True
    if len(hist) < DIVERGE_RUN + 1:
        return False
    F0 = hist[0]
    R0 = max(1.0, abs(float(R00)))
    tail = hist[-DIVERGE_RUN:]
    return all(f > DIVERGE_ABS * R0 for f in tail) or all(f > DIVERGE_FACTOR * F0 and f > DIVERGE_FLOOR * R0 for f in tail)
