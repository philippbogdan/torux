"""Analytic vacuum fields in the JAX engine: Dommaschk potentials (torux.io.analytic spec).

B = B0R0 grad Phi_h with Phi_h = sum_{m,l} (a D_{m,l} + b N_{m,l-1}) cos(m phi)
                                          + (c D_{m,l} + d N_{m,l-1}) sin(m phi),
the D, N functions finite sums of c R^p (ln R)^q Z^j built exactly (Fractions) by
torux.io.analytic.C_functions; B and grad B by automatic differentiation of the potential
(the gradient tensor is the potential's Hessian: symmetric and traceless to rounding).

Vector potential: only toroidal fluxes through planar sections are ever taken from A, i.e.
only (curl A)_phi is used, so the part supplies A = -G e_Z with
G(R, phi, Z) = int_{R_c}^{R} B_phi(R', phi, Z) dR' (Gauss-Legendre, 32 nodes; R_c a fixed
radius inside the domain): (curl A)_phi = dG/dR = B_phi, and the loop integral of A around
a planar section is minus the toroidal flux through it, as for the coil parts.
"""
from __future__ import annotations

from functools import partial
from math import factorial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _dn_tables(m, n, kind):
    from torux.io.analytic import C_functions
    kmax = n // 2 if kind == "D" else (n - 1) // 2
    if n < 0 or kmax < 0:
        return []
    Cs = C_functions(m, kmax, kind)
    out = []
    for k in range(kmax + 1):
        e = n - 2 * k
        terms = [(int(p), int(q), float(c)) for (p, q), c in Cs[k].items()]
        out.append((e, 1.0 / factorial(e), terms))
    return out


def make_dommaschk(spec):
    """Returns (phi_h(x) scalar for one Cartesian point x (3,), B0R0)."""
    B0R0 = float(spec.get("B0R0", 1.0))
    if spec.get("include_toroidal", False):
        raise ValueError("analytic Dommaschk part must not carry the toroidal term (use a ToroidalField part)")
    tabs = []
    for (m, l, a, b, c, d) in spec["terms"]:
        tabs.append((int(m), float(a), float(b), float(c), float(d), _dn_tables(int(m), int(l), "D"),
                     _dn_tables(int(m), int(l) - 1, "N")))

    def poly(tab, R, lnR, Z):
        s = 0.0
        for e, inv, terms in tab:
            cr = 0.0
            for p, q, cc in terms:
                cr = cr + cc * R ** p * (lnR ** q if q else 1.0)
            s = s + (Z ** e if e else 1.0) * inv * cr
        return s

    def phi_h(x):
        R = jnp.sqrt(x[0] * x[0] + x[1] * x[1])
        ph = jnp.arctan2(x[1], x[0])
        Z = x[2]
        lnR = jnp.log(R)
        tot = 0.0
        for (m, a, b, c, d, tD, tN) in tabs:
            D = poly(tD, R, lnR, Z)
            N = poly(tN, R, lnR, Z)
            tot = tot + (a * D + b * N) * jnp.cos(m * ph) + (c * D + d * N) * jnp.sin(m * ph)
        return tot
    return phi_h, B0R0


def analytic_part(spec, R_c=1.0, nquad=32):
    """f(x (T,3)) -> (B, A, G) for an analytic part spec (dict with 'class')."""
    cls = spec.get("class", "")
    if not cls.endswith("DommaschkField"):
        raise ValueError(f"unknown analytic part {cls}")
    phi_h, B0R0 = make_dommaschk(spec)
    gradf = jax.grad(phi_h)
    hessf = jax.hessian(phi_h)
    xg, wg = np.polynomial.legendre.leggauss(nquad)
    xg, wg = jnp.asarray(xg), jnp.asarray(wg)

    def Bphi_at(R, ph, Z):
        x = jnp.stack([R * jnp.cos(ph), R * jnp.sin(ph), Z])
        g = gradf(x)
        return B0R0 * (-jnp.sin(ph) * g[0] + jnp.cos(ph) * g[1])

    def one(x, need_grad=True, need_A=True):
        B = B0R0 * gradf(x)
        Gm = B0R0 * hessf(x) if need_grad else jnp.zeros((3, 3))
        if need_A:
            R = jnp.sqrt(x[0] * x[0] + x[1] * x[1])
            ph = jnp.arctan2(x[1], x[0])
            # G = int_{R_c}^{R} B_phi dR'
            Rq = 0.5 * (R + R_c) + 0.5 * (R - R_c) * xg
            vals = jax.vmap(lambda r: Bphi_at(r, ph, x[2]))(Rq)
            Gint = 0.5 * (R - R_c) * jnp.sum(wg * vals)
            A = jnp.stack([0.0 * R, 0.0 * R, -Gint])
        else:
            A = jnp.zeros(3)
        return B, A, Gm

    @partial(jax.jit, static_argnames=("need_grad", "need_A"))
    def f(x, need_grad=True, need_A=True):
        return jax.vmap(lambda y: one(y, need_grad, need_A))(x)
    return f
