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
