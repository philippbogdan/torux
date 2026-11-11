"""The field: against a closed form, and the solver's engine against the independent reference evaluator."""
import numpy as np
from scipy.special import ellipe, ellipk

import torux
from torux.field import FieldEngine
from torux.reference import fast64

MU0 = 4e-7 * np.pi


def ring_source(R=1.3, current=2.5e5):
    """One circular coil of radius R in the plane z = 0, as a Fourier filament."""
    coef = np.zeros((1, 3, 3))
    coef[0, 0, 2] = R          # x = R cos 2 pi t
    coef[0, 1, 1] = R          # y = R sin 2 pi t
    return torux.Source(id="ring", parts=[torux.FourierCurves(coef, np.array([current]))])


def ring_field(R, current, r, z):
    """B_r, B_z of a circular loop from complete elliptic integrals."""
    k2 = 4 * R * r / ((R + r) ** 2 + z ** 2)
    K, E = ellipk(k2), ellipe(k2)
    c = MU0 * current / (2 * np.pi * np.sqrt((R + r) ** 2 + z ** 2))
    d = (R - r) ** 2 + z ** 2
    return c * z / r * (-K + (R ** 2 + r ** 2 + z ** 2) / d * E), c * (K + (R ** 2 - r ** 2 - z ** 2) / d * E)


def test_engine_against_the_closed_form_of_a_ring():
    R, I = 1.3, 2.5e5
    src = ring_source(R, I)
    rng = np.random.default_rng(0)
    r, z, phi = 0.2 + 0.7 * rng.random(64), 0.5 * (rng.random(64) - 0.5), 2 * np.pi * rng.random(64)
    x = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    B = np.asarray(FieldEngine(src, dmin=0.2).evaluate(x, grad=False, A=False)["B"])
    Br, Bz = ring_field(R, I, r, z)
    exact = np.stack([Br * np.cos(phi), Br * np.sin(phi), Bz], -1)
    assert np.max(np.linalg.norm(B - exact, axis=1) / np.linalg.norm(exact, axis=1)) < 1e-13


def test_reference_against_the_closed_form_of_a_ring():
    R, I = 1.3, 2.5e5
    src = ring_source(R, I)
    rng = np.random.default_rng(0)
    r, z, phi = 0.2 + 0.7 * rng.random(64), 0.5 * (rng.random(64) - 0.5), 2 * np.pi * rng.random(64)
    x = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    B = np.asarray(fast64.Field(src).B(x))
    Br, Bz = ring_field(R, I, r, z)
    exact = np.stack([Br * np.cos(phi), Br * np.sin(phi), Bz], -1)
    assert np.max(np.linalg.norm(B - exact, axis=1) / np.linalg.norm(exact, axis=1)) < 1e-13


def test_engine_against_the_reference_on_real_coils(solved_954, quasr_954):
    """Two evaluators that share no code agree to rounding on the surface of a real coil set."""
    rng = np.random.default_rng(2)
    x = solved_954.points(2 * np.pi * rng.random(256), 2 * np.pi * rng.random(256))
    B = solved_954.field(x)
    Bref = np.asarray(fast64.Field(quasr_954).B(x))
    assert np.max(np.linalg.norm(B - Bref, axis=1) / np.linalg.norm(Bref, axis=1)) < 2e-14


def test_field_is_divergence_free_and_curl_free(solved_954, quasr_954):
    rng = np.random.default_rng(3)
    x = solved_954.points(2 * np.pi * rng.random(64), 2 * np.pi * rng.random(64))
    eng = FieldEngine(quasr_954, dmin=quasr_954.meta.get("dmin", 0.05))
    G = np.asarray(eng.evaluate(x, grad=True, A=False)["G"])        # G[n, i, j] = dB_i / dx_j
    scale = np.linalg.norm(G, axis=(1, 2))
    div = np.abs(np.trace(G, axis1=1, axis2=2))
    curl = np.linalg.norm(G - np.transpose(G, (0, 2, 1)), axis=(1, 2))
    assert np.max(div / scale) < 1e-12 and np.max(curl / scale) < 1e-12
