"""Constants of the reference evaluator.

mu0 is taken as exactly 4 pi 1e-7 H/m (the pre-2019 SI value used by every data
source, torux.source.MU0 is its float64 rounding), so mu0/(4 pi) = 1e-7 exactly; the
reference carries 1e-7 as a double-double.
"""
import mpmath

from .dd import mp_to_dd

with mpmath.workdps(60):
    MU0_4PI_HI, MU0_4PI_LO = mp_to_dd(mpmath.mpf(1) / 10 ** 7)
