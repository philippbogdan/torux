"""Two requests with no surface, and the certificates that say so.

    python examples/no_surface.py

Two to three minutes each on a recent desktop.  A certificate needs room, so each call gets an hour and returns as
soon as it has its answer. On a slow or busy machine the answer may be 'undetermined' instead.
"""
import os

import torux
from torux.cli import report

here = os.path.dirname(os.path.abspath(__file__))

for name, what in (("quasr_1100073.json", "the requested flux lies beyond the last surface the coils make"),
                   ("quasr_0219457.json", "the requested flux falls inside an island chain")):
    src = torux.load(os.path.join(here, name))
    res = torux.solve(src, budget_s=3300)
    print(f"{name}: {what}")
    print(report(res, src.request["flux_Wb"]), end="\n\n")
