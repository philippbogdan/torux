"""A torux surface, judged by the evaluator beside this file.

    python results/evaluator/judge.py examples/quasr_0000954.json [--flux WB]

Solves the request with torux, writes the surface in the evaluator's form (each coefficient copied, none computed)
and the coils as Fourier filaments, and prints what the evaluator measures: the largest |B.n|/|B| at its 4,000 random
points and the enclosed flux against the request.  On x86-64 the measurement is in extended precision; elsewhere in
float64, whose own rounding is about 1e-15.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate, precise_dtype  # noqa: E402


def coils_record(src, flux):
    """The source's Fourier filaments as the evaluator reads them."""
    coils = []
    for p in src.parts:
        if p.kind != "fourier":
            raise SystemExit("the evaluator judges surfaces of Fourier-filament coils only")
        K = p.order
        for i in range(p.n):
            c = np.asarray(p.coef[i])                       # (3, 2K + 1): [c0, s1, c1, ..., sK, cK]
            d = {"current_A": float(p.current[i])}
            for a, ax in enumerate("xyz"):
                d[ax + "c"] = [float(c[a, 0])] + [float(c[a, 2 * k]) for k in range(1, K + 1)]
                d[ax + "s"] = [0.0] + [float(c[a, 2 * k - 1]) for k in range(1, K + 1)]
            coils.append(d)
    return dict(coils=coils, requested_toroidal_flux_Wb=abs(float(flux)))


def surface_form(t):
    """The torus as rz_fourier terms: one per nonzero coefficient, m_j = m, n_j = n nfp."""
    keys = [k for k in ("rc", "rs", "zc", "zs") if hasattr(t, k)]
    arr = {k: np.asarray(getattr(t, k), dtype=np.float64) for k in keys}
    out = dict(form="rz_fourier", m=[], n=[], **{k: [] for k in keys})
    for m in range(t.M + 1):
        for j in range(2 * t.N + 1):
            vals = {k: a[m, j] for k, a in arr.items()}
            if all(v == 0.0 for v in vals.values()):
                continue
            out["m"].append(m)
            out["n"].append((j - t.N) * int(t.nfp))
            for k, v in vals.items():
                out[k].append(float(v))
    return out


def main():
    import torux
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--flux", type=float)
    a = ap.parse_args()
    src = torux.load(a.file)
    res = torux.solve(src, a.flux)
    print("torux:", res.status, f"in {res.wall_s:.1f} s")
    if res.surface is None:
        return 2
    if getattr(res.surface, "phi0", 0.0) != 0.0:
        raise SystemExit("this torus is written about another symmetry plane: not handled here")
    answer = dict(design=src.id, solver="torux", status="surface", surface=surface_form(res.surface))
    out = evaluate(coils_record(src, res.surface.flux), answer, precise=precise_dtype() is not None)
    tier = "extended precision" if "bn_max" in out else "float64"
    print(f"evaluator ({tier}): largest |B.n|/|B| {out.get('bn_max', out['f64_bn_max']):.3g} at 4,000 points, "
          f"median {out.get('bn_median', out['f64_bn_median']):.3g}; flux off the request by {abs(out['flux_rel_err']):.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
