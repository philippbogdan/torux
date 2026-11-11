"""A verified flux surface, and what stands behind the word.

    python examples/surface.py
"""
import os

import numpy as np

import torux
from torux.cli import report

here = os.path.dirname(os.path.abspath(__file__))
res = torux.solve(os.path.join(here, "quasr_0000954.json"))        # a QUASR download, as it comes
print(report(res))

# the surface itself: Fourier coefficients of R and Z in straight-field-line angles, and points on it
s = res.surface
print(f"\ncoefficients         rc, zs of shape {s.rc.shape}, {s.nfp} field periods")
x = res.points(theta=np.linspace(0, 2 * np.pi, 5), phi=np.zeros(5))
print(f"five points at phi = 0, R from {np.hypot(x[:, 0], x[:, 1]).min():.4f} to {np.hypot(x[:, 0], x[:, 1]).max():.4f} m")

# what stands behind "verified": the solver's own check and the independent evaluator's
print(f"solver, off its grid  {res.evidence['solver']['bn_off']:.2e}")
check = res.evidence["independent"]
print(f"independent, sampled  {check['bn_max']:.2e} at {check['npts']} points; estimated maximum {check['bn_sup']:.2e}")
