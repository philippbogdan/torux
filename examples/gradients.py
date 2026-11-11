"""The derivative of the rotational transform in every coil parameter, by the adjoint method.

    python examples/gradients.py
"""
import os

import numpy as np

import torux

here = os.path.dirname(os.path.abspath(__file__))
res = torux.solve(os.path.join(here, "quasr_0000954.json"), gradients=True)
g = res.gradients
print(f"{res.status}, iota {res.iota:.12g}")
print(f"d iota / d flux            {g['d_iota_d_flux']:.6g} per Wb")
print(f"d iota / d current         {np.asarray(g['d_iota_d_current']).shape[0]} coils, largest {np.abs(g['d_iota_d_current']).max():.3g} per A")
print(f"d iota / d coefficient     {np.asarray(g['d_iota_d_coef']).size} numbers, largest {np.abs(g['d_iota_d_coef']).max():.3g} per m")
