"""Independent reference magnetic-field evaluator (B, A, dB_i/dx_j) for every part kind of
torux.source, written without reference to the solver's field engine.

Tiers
  precise  double-double arithmetic (~32 digits), adaptive quadrature with a-posteriori error
           estimates, cancellation-free closed forms; value + error estimate per point.
               from torux.reference import precise
               ref = precise.Reference(source); r = ref.evaluate(points)        # dict, see precise
  fast64   float64 with compensated summation, a-priori quadrature control; ~1e-15 relative
           away from the sources.
               from torux.reference import fast64
               F = fast64.Field(source); r = F.evaluate(points)                 # B, A, dB
  mp       mpmath truth at 50 digits with different formulas/quadrature (validation only).
  trace    field-line tracer (Gauss-Legendre IRK in phi) on fast64, Poincare sections with
           step-doubling error estimates.
               from torux.reference import trace
               out = trace.poincare(source, R0, Z0, phis, nturns)

Conventions: SI, mu0 = 4 pi 1e-7 exactly; dB[n, i, j] = dB_i/dx_j; A in the Coulomb gauge
(filaments: mu0 I/(4 pi) oint dl/|x - r|; dipoles mu0/(4 pi) m x r/r^3; rings/polygons the
elliptic A_phi; toroidal field A = -RB ln(R/1 m) z_hat).
"""
import glob as _glob
import hashlib as _hashlib
import os as _os


def _invalidate_stale_numba_cache():
    """numba's on-disk cache (cache=True) does not notice edits to functions called from another
    module (e.g. trace -> fast64 -> dd), so cached machine code could silently embed old code.
    The cache of this package is therefore dropped whenever any of its sources changes."""
    here = _os.path.dirname(_os.path.abspath(__file__))
    h = _hashlib.sha256()
    for fn in sorted(_glob.glob(_os.path.join(here, "*.py"))):
        with open(fn, "rb") as f:
            h.update(f.read())
    tag = h.hexdigest()
    cache_dir = _os.path.join(here, "__pycache__")
    stamp = _os.path.join(cache_dir, "numba_source_hash.txt")
    try:
        with open(stamp) as f:
            old = f.read().strip()
    except OSError:
        old = None
    if old != tag:
        for fn in _glob.glob(_os.path.join(cache_dir, "*.nbi")) + _glob.glob(_os.path.join(cache_dir, "*.nbc")):
            try:
                _os.remove(fn)
            except OSError:
                pass
        try:
            _os.makedirs(cache_dir, exist_ok=True)
            with open(stamp, "w") as f:
                f.write(tag)
        except OSError:
            pass


_invalidate_stale_numba_cache()
