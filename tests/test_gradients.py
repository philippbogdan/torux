"""Adjoint gradients of the rotational transform against central differences."""
import numpy as np

import torux
from torux.family import Continuation
from torux.field import FieldEngine


def test_adjoint_against_central_differences(quasr_954):
    src = quasr_954
    res = torux.solve(src, gradients=True)
    assert res.status == "surface"
    g, t, part = res.gradients, res.surface, src.parts[0]
    dmin = res.record["engine"]["dmin"]

    def iota(coef=None, current=None, flux=None):
        """The torus solved again at its own resolution for a changed coil set or flux."""
        s = torux.Source(src.id, [torux.FourierCurves(part.coef if coef is None else coef,
                                                      part.current if current is None else current)],
                         nfp=src.nfp, stellsym=src.stellsym)
        cont = Continuation(FieldEngine(s, dmin=dmin), src.nfp, tol=1e-15)
        sol = cont.problem(t.M, t.N).newton(t, t.flux if flux is None else flux, tol=1e-15, maxit=10)
        assert sol.converged
        return sol.iota

    def central(key, index, h):
        up, down = getattr(part, key).copy(), getattr(part, key).copy()
        up[index] += h
        down[index] -= h
        return (iota(**{key: up}) - iota(**{key: down})) / (2 * h)

    fd = central("current", 0, 1e-6 * abs(part.current[0]))            # the current of the first coil
    assert abs(fd - g["d_iota_d_current"][0]) <= 1e-5 * abs(fd) + 1e-14
    fd = central("coef", (1, 0, 2), 1e-6)                             # one Fourier coefficient of the second
    assert abs(fd - g["d_iota_d_coef"][1, 0, 2]) <= 1e-5 * abs(fd) + 1e-12
    h = 1e-6 * abs(t.flux)                                            # the requested flux
    fd = (iota(flux=t.flux + h) - iota(flux=t.flux - h)) / (2 * h)
    assert abs(fd - g["d_iota_d_flux"]) <= 1e-5 * abs(fd) + 1e-12
