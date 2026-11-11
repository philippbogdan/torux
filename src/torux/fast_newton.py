"""Work reductions of the family's engine and of the Newton linear algebra: switches.

The recommended set (pcache, fmeas) is on by default.  TORUX_FAST_NEWTON overrides it per process: a comma-separated
list of names, "all" for the recommended set, "every" with the experimental ones, "none" for none.  In code:
``set_opts(...)`` / the ``opts(...)`` context manager.  Independent of torux.fast_design and torux.fast_family.

kfall, flean and fmeas act inside the family only (torux.family.Continuation.run: the near-axis start, the continuation
steps, their plateau retries and resolution raises); the refinement of the request's / the range's torus, the chains,
jumps and certificates run as without them.  mixlu, gpulu and pcache act wherever their solver runs.

  fmeas   the family's lean engine (Continuation._lean_update: the region ahead of the family, the current torus
          scaled LEAN_SCALE about its m = 0 curve) with each coil's node count measured
          (torux.field.engine.family_nodes): at the region points where the coil's trapezoid error is largest (the 64
          narrowest strips d / |r'| and the 32 nearest), the fewest nodes (candidates from 4 order + 8 up by 1.25x; the
          upper of two consecutive that both meet it) at which the coil's field is within FMEAS_EPS of its field with
          the strip model's count, pointwise relative to the whole source's |B| there.  The engine built from these
          counts is then certified (Continuation._fmeas_certify): every Cartesian component of a trapezoid Biot-Savart
          field is a sum of derivatives of 1 / |x - r|, so the difference of two engines' fields is harmonic off the
          coils and its largest magnitude inside a coil-free solid torus lies on its surface; on the torus scaled
          FMEAS_SURFACE = 1.25 (the family solves tori up to ~1.09x before the next rebuild) the difference from the
          standard engine must be <= FMEAS_CERT min |B| at all samples, no coil within one sample spacing of it and
          none inside it -- else this rebuild takes the standard engine.  When the family ends, the standard engine
          for the last region is restored, so everything after the family (the refinement's lean engine is compared
          with it; jumps, certificates) sees the standard engine.  A caller sets Continuation.fmeas_off for families
          whose outcome the budget sets: torux.analysis.analyse keeps fmeas for quick mode's requests only.  Not on a
          card (fast_family 'cudafil'): there the filament sums are a few per cent of the time and the measurement and
          certificate are host work.  The standard counts are 3.7-12x (median 5.1x) the computed per-coil trapezoid
          floors on the family's tori.
  pcache  the Newton-Krylov preconditioner (torux.torus.KrylovTorus._precond, host path) keeps, per Newton step, the
          arrays that depend on the iterate only (the normal field nR, nZ and the twist T = Bp (nZ wR - nR wZ): two
          of its four FFT pairs) and _Linv's divisors per iota, instead of recomputing them at every GMRES iteration:
          the same arithmetic on the same inputs, so the same bits (1.35-2x per preconditioner application).
  flean   (experimental) the family's lean engine by the strip model with FLEAN_EPS and FLEAN_MARGIN instead of
          1e-17 and 0.5.  Not at rounding everywhere: eps 1e-15 rules reach B errors of 6e-15 on the family's tori
          and O(1) on tori 1.2x near coils (the dmin cap); eps 1e-17 with margin 1.0 reaches 4e-15.
  kfall   (experimental) the dense backup of a failed family Newton-Krylov replaced by Newton-Krylov with tight GMRES
          (KFALL_GTOL, restart KFALL_RESTART, KFALL_MAXITER cycles) from the same guess.  Not equivalent to the dense
          Newton: the preconditioned GMRES does not reach its tolerance at near-resonant iterates, so it changes
          family paths.
  mixlu   (experimental) the dense Newton's systems from DENSE_FAST_MIN unknowns by the float32 LU and float64
          iterative refinement to the float64 LU's backward-error level, else the float64 LU
          (torux.torus._dense_fast).  1.3-1.6x at cond <= 1e5 but 0.25-0.8x at cond >= 1e6 (refinement fails, then
          the float64 LU too); the family's dense backups have cond ~3e5-5e6.
  gpulu   (experimental; machines with a CUDA card and fast_family 'cudafil') the same dense systems by the card's
          float64 LU (cuSOLVER through JAX); the host's float64 LU without a card.
"""
from __future__ import annotations

import contextlib
import os

NAMES = ("kfall", "flean", "fmeas", "mixlu", "gpulu", "pcache")
# recommended: pcache (bit-identical) and fmeas (scoped to quick mode's requests by torux.analysis.analyse)
EXPERIMENTAL = ("kfall", "flean", "mixlu", "gpulu")
RECOMMENDED = tuple(n for n in NAMES if n not in EXPERIMENTAL)

KFALL_GTOL = 1e-10          # 'kfall': GMRES relative tolerance of the backup Newton-Krylov
KFALL_RESTART = 60
KFALL_MAXITER = 8           # restart cycles per Newton iteration (at most 480 GMRES iterations)

FLEAN_EPS = 1e-15           # 'flean': the family's lean engine's trapezoid relative error at the region
FLEAN_MARGIN = 1.0          # ... and the share of each coil's distance to the region a target may approach
DENSE_FAST_MIN = 2000       # 'mixlu' / 'gpulu': dense Newton systems from this many unknowns
MIXLU_C = 4.0               # 'mixlu': refinement ends at max|r| <= MIXLU_C eps (|J|_inf |x|_inf + |b|_inf)
MIXLU_STEPS = 8             # ... within this many steps, else the float64 LU
FMEAS_CERT = 1e-15          # 'fmeas': the certified bound on |B_fmeas - B_release| / min |B| inside the region
FMEAS_EPS = 5e-16           # 'fmeas': each coil's field at its closest region points within this of the model's
                            # (pointwise, relative to the whole source's |B| there)


def _params_from_env():
    """TORUX_FAST_NEWTON_PARAMS='FLEAN_MARGIN=0.75,KFALL_MAXITER=4' overrides the constants above (comparisons)."""
    v = os.environ.get("TORUX_FAST_NEWTON_PARAMS")
    if not v:
        return
    g = globals()
    for item in v.split(","):
        k, _, x = item.partition("=")
        k = k.strip()
        if k not in ("KFALL_GTOL", "KFALL_RESTART", "KFALL_MAXITER", "FLEAN_EPS", "FLEAN_MARGIN", "FMEAS_EPS",
                     "DENSE_FAST_MIN", "MIXLU_C", "MIXLU_STEPS", "FMEAS_CERT"):
            raise ValueError(f"TORUX_FAST_NEWTON_PARAMS: unknown parameter {k}")
        g[k] = type(g[k])(float(x)) if isinstance(g[k], float) else int(x)


_params_from_env()


def _from_env():
    v = os.environ.get("TORUX_FAST_NEWTON")
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
        raise ValueError(f"TORUX_FAST_NEWTON: unknown switches {sorted(bad)} (known: {NAMES})")
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
