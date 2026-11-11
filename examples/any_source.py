"""Sources that are not a QUASR design.

    python examples/any_source.py
"""
import os

import torux
from torux.cli import report

here = os.path.dirname(os.path.abspath(__file__))

# HSX, a built stellarator: 48 coils and no starting surface, so torux starts from the magnetic axis
print("HSX")
print(report(torux.solve(os.path.join(here, "hsx.json"), flux=0.0359221439332229, budget_s=900)))

# DIII-D's upper I-coils, a tokamak's perturbation coils: their vacuum field has no toroidal flux surface, by proof
print("\nDIII-D upper I-coils")
print(report(torux.solve(os.path.join(here, "d3d_icoils_upper.dat"))))
