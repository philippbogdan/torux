"""CNT, the Columbia Non-neutral Torus: four circular coils, two of them interlocked, aspect ratio about 2.

    python examples/cnt.py

The coils are cnt.json (written by make_cnt.py). The surface asked for lies inside the region of closed flux
surfaces, between the iota = 1/5 and 2/9 island chains.
"""
import os

import torux

here = os.path.dirname(os.path.abspath(__file__))
coils = os.path.join(here, "cnt.json")

s = torux.solve(coils, flux=0.011624274070939846)
print(s)

if s.status == "surface":
    # the independent solution (README.md): iota = 0.2079427045422565, the noble number between 1/5 and 2/9
    print(f"\nrotational transform  {s.iota:.16g}, expected 0.2079427045422565")
    print(f"minor radius          {s.minor_radius:.12g} m, expected 0.136080864196 m")
    print(f"aspect ratio          {s.aspect_ratio:.9g}, expected 2.25417683")
