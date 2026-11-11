"""Field-line tracing (JAX, vectorised over many lines).

The toroidal angle phi is the independent variable:
    dR/dphi = R B_R / B_phi,   dZ/dphi = R B_Z / B_phi.
Integrator: Gragg-Bulirsch-Stoer extrapolation of the modified midpoint rule over fixed
macro steps H (step sequence 2, 4, ..., 2k; order 2k; Aitken-Neville extrapolation in h^2),
which vectorises without per-line step control; the difference between the last two
diagonal entries of the extrapolation tableau is returned as a local error estimate.
Lines whose B_phi changes sign or that leave a bounding box are marked lost (NaN).
Variational equations come for free from jax.jacfwd through the integrator.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_rhs(engine, bphi_frac=1e-3):
    """f(phi, y) with y (L, 2) = (R, Z) -> dy/dphi (L, 2).  phi scalar or (L,).

    Where |B_phi| < bphi_frac |B| (a line about to turn back in phi, or next to a coil) the
    right-hand side is NaN: the line is lost (it cannot be followed in phi) and stays NaN."""
    def f(phi, y):
        R, Z = y[:, 0], y[:, 1]
        phi = jnp.broadcast_to(phi, R.shape)
        c, s = jnp.cos(phi), jnp.sin(phi)
        x = jnp.stack([R * c, R * s, Z], -1)
        B = engine.evaluate(x, grad=False, A=False)["B"]
        BR = B[:, 0] * c + B[:, 1] * s
        Bp = -B[:, 0] * s + B[:, 1] * c
        Bm = jnp.sqrt(jnp.sum(B * B, -1))
        ok = (jnp.abs(Bp) >= bphi_frac * Bm) & (R > 0)
        Bp = jnp.where(ok, Bp, jnp.nan)
        return jnp.stack([R * BR / Bp, R * B[:, 2] / Bp], -1)
    return f


def gbs_step(f, phi, y, H, k=8):
    """One extrapolated macro step of size H from (phi, y).  Returns y_new, err (L,).

    Modified midpoint sequences n_j = 2 j (j = 1..k), each a fori_loop (small graphs),
    then Aitken-Neville extrapolation in h^2: order 2k."""
    f0 = f(phi, y)
    T = []
    for j in range(1, k + 1):
        n = 2 * j
        h = H / n
        z1 = y + h * f0

        def body(i, zz, h=h):
            z0, z1 = zz
            return (z1, z0 + 2 * h * f(phi + i * h, z1))
        _, zn = jax.lax.fori_loop(1, n, body, (y, z1))
        T.append(zn)
    ns = [2 * j for j in range(1, k + 1)]
    tab = [T[0]]
    for j in range(1, k):
        row = [T[j]]
        for kk in range(1, j + 1):
            r = (ns[j] / ns[j - kk]) ** 2 - 1.0
            row.append(row[kk - 1] + (row[kk - 1] - tab[kk - 1]) / r)
        tab = row
    y_new = tab[-1]
    err = jnp.max(jnp.abs(tab[-1] - tab[-2]), axis=-1)
    return y_new, err


def trace(engine, y0, phi0, dphi, nsteps, k=8, f=None):
    """Integrate lines y0 (L, 2) from phi0 over dphi in nsteps macro steps.
    Returns y (L, 2) and the summed local error estimates (L,)."""
    f = f or make_rhs(engine)
    H = dphi / nsteps

    def body(i, carry):
        y, e = carry
        y, err = gbs_step(f, phi0 + i * H, y, H, k)
        return y, e + err
    y, e = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(y0), jnp.zeros(y0.shape[0])))
    return y, e


def poincare(engine, y0, phi0, nfp, nperiods, steps_per_period=8, k=8, f=None):
    """Section points after each field period: (nperiods, L, 2) and error estimates."""
    f = f or make_rhs(engine)
    period = 2 * np.pi / nfp

    def one(carry, _):
        y, ph = carry
        y2, e = trace(engine, y, ph, period, steps_per_period, k, f)
        return (y2, ph + period), (y2, e)
    (_, _), (ys, es) = jax.lax.scan(one, (jnp.asarray(y0), jnp.asarray(phi0, dtype=jnp.float64)), None, length=nperiods)
    return ys, es
