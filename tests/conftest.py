"""Shared fixtures: the bundled inputs and one solved request."""
import os

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

# QUASR design 954 at the flux its outermost stored surface encloses: the row of results/quasr.parquet
QUASR_954 = dict(flux_Wb=0.06910690581397087, iota=-0.09752297411441443)


def example(name):
    return os.path.join(EXAMPLES, name)


@pytest.fixture(scope="session")
def quasr_954():
    import torux
    return torux.load(example("quasr_0000954.json"))


@pytest.fixture(scope="session")
def solved_954(quasr_954):
    import torux
    return torux.solve(quasr_954)


@pytest.fixture(scope="session")
def quasr_954_mirrored(quasr_954):
    """QUASR 954 with the mirrored copy of its coil at 3% more current: two field periods, no stellarator symmetry."""
    import numpy as np
    import torux
    from torux.io._stellcommon import apply_verified_symmetry
    p = quasr_954.parts[0]
    nb = p.n // (2 * quasr_954.nfp)                       # the coils of one half period, then their mirror images
    current = np.array([c * (1.03 if (k // nb) % 2 else 1.0) for k, c in enumerate(p.current)])
    src = torux.Source("quasr/954 mirrored at +3%", [torux.FourierCurves(p.coef, current)])
    apply_verified_symmetry(src, [quasr_954.nfp])
    assert src.nfp == 2 and not src.stellsym
    return src


@pytest.fixture(scope="session")
def solved_954_mirrored(quasr_954_mirrored):
    import torux
    return torux.solve(quasr_954_mirrored, flux=0.05)
