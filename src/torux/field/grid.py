"""Tabulated field for bulk field-line tracing (escape bounds, manifolds, Poincare sections).

(B_R, B_phi, B_Z) are tabulated on a uniform (R, phi, Z) grid over one field period and
interpolated by tensor-product Lagrange polynomials of odd degree p (p + 1 nodes per direction,
the stencil centred on the cell), periodic in phi.  Queries outside the box give NaN (the
tracer then counts the line as lost).  ``check`` measures the interpolation error against the
exact engine at random points of the box (away from the coils), so every use of the table can
state its accuracy; an exact evaluation remains the reference for anything near a decision.
The interface matches FieldEngine.evaluate for B only (grad=False, A=False).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _lagrange_weights(t, p):
    """Weights of the p + 1 Lagrange basis polynomials on nodes 0..p at positions t (k,)."""
    nodes = np.arange(p + 1, dtype=float)
    w = []
    for j in range(p + 1):
        num = 1.0
        den = 1.0
        for m in range(p + 1):
            if m != j:
                num = num * (t - nodes[m])
                den = den * (nodes[j] - nodes[m])
        w.append(num / den)
    return jnp.stack(w, -1)


@partial(jax.jit, static_argnames=("p", "shape"))
def _interp(table, lo, h, x, p, shape):
    nR, nP, nZ = shape
    R = jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    ph = jnp.arctan2(x[:, 1], x[:, 0])
    Z = x[:, 2]
    period = h[1] * nP
    php = jnp.mod(ph, period)
    uR = (R - lo[0]) / h[0]
    uP = php / h[1]
    uZ = (Z - lo[2]) / h[2]
    half = (p - 1) // 2
    iR = jnp.floor(uR).astype(jnp.int32) - half
    iP = jnp.floor(uP).astype(jnp.int32) - half
    iZ = jnp.floor(uZ).astype(jnp.int32) - half
    inside = (iR >= 0) & (iR + p < nR) & (iZ >= 0) & (iZ + p < nZ)
    iRc = jnp.clip(iR, 0, nR - p - 1)
    iZc = jnp.clip(iZ, 0, nZ - p - 1)
    wR = _lagrange_weights(uR - iRc, p)
    wP = _lagrange_weights(uP - iP, p)
    wZ = _lagrange_weights(uZ - iZc, p)
    offs = jnp.arange(p + 1)
    IR = iRc[:, None] + offs[None, :]
    IP = jnp.mod(iP[:, None] + offs[None, :], nP)
    IZ = iZc[:, None] + offs[None, :]
    # gather (k, p+1, p+1, p+1, 3)
    vals = table[IR[:, :, None, None], IP[:, None, :, None], IZ[:, None, None, :]]
    out = jnp.einsum("ka,kb,kc,kabcd->kd", wR, wP, wZ, vals)
    c, s = jnp.cos(ph), jnp.sin(ph)
    Bx = out[:, 0] * c - out[:, 1] * s
    By = out[:, 0] * s + out[:, 1] * c
    B = jnp.stack([Bx, By, out[:, 2]], -1)
    return jnp.where(inside[:, None], B, jnp.nan)


class FieldGrid:
    """Lagrange-interpolated table of an engine's field over one field period."""

    def __init__(self, engine, nfp, Rmin, Rmax, Zmin, Zmax, nR=96, nphi=64, nZ=96, p=7, batch=16384,
                 coil_pts=None, d_c=None):
        """coil_pts, d_c: table nodes within d_c of any coil point (plus one stencil width) are NaN, so
        the interpolant is NaN there and a line entering the coils' neighbourhood is lost; the table
        then needs its accuracy only outside it (see check)."""
        self.engine, self.nfp, self.p = engine, nfp, p
        self.lo = np.array([Rmin, 0.0, Zmin])
        self.h = np.array([(Rmax - Rmin) / (nR - 1), 2 * np.pi / nfp / nphi, (Zmax - Zmin) / (nZ - 1)])
        self.shape = (nR, nphi, nZ)
        R = Rmin + self.h[0] * np.arange(nR)
        P = self.h[1] * np.arange(nphi)
        Z = Zmin + self.h[2] * np.arange(nZ)
        RR, PP, ZZ = np.meshgrid(R, P, Z, indexing="ij")
        x = np.stack([RR * np.cos(PP), RR * np.sin(PP), ZZ], -1).reshape(-1, 3)
        Bs = []
        for lo in range(0, len(x), batch):
            Bs.append(np.asarray(engine.evaluate(x[lo:lo + batch], grad=False, A=False)["B"]))
        B = np.concatenate(Bs)
        c, s = np.cos(PP.ravel()), np.sin(PP.ravel())
        cyl = np.stack([B[:, 0] * c + B[:, 1] * s, -B[:, 0] * s + B[:, 1] * c, B[:, 2]], -1)
        cyl = cyl.reshape(nR, nphi, nZ, 3)
        if coil_pts is not None and d_c is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray(coil_pts))
            reach = d_c + p * float(np.max(self.h[[0, 2]]))
            d, _ = tree.query(x, k=1, distance_upper_bound=reach)
            near = np.isfinite(d).reshape(nR, nphi, nZ)
            cyl[near] = np.nan
            self.excluded_fraction = float(near.mean())
        self.table = jnp.asarray(cyl)
        self.coil_pts, self.d_c = coil_pts, d_c
        self.analytic = []
        self.kinds = None

    def evaluate(self, x, grad=False, A=False, err=False):
        if grad or A:
            raise ValueError("the tabulated field gives B only")
        x = jnp.asarray(x, dtype=jnp.float64)
        return dict(B=_interp(self.table, jnp.asarray(self.lo), jnp.asarray(self.h), x, self.p, self.shape))

    def check(self, npts=2000, seed=0, min_dist=None, coil_pts=None):
        """Max and median |B_grid - B_exact| / |B_exact| at random points of the box (optionally only
        points farther than min_dist from the coil sample points)."""
        rng = np.random.default_rng(seed)
        nR, nP, nZ = self.shape
        R = self.lo[0] + rng.uniform(self.p, nR - 1 - self.p, npts) * self.h[0]
        P = rng.uniform(0, 2 * np.pi / self.nfp, npts)
        Z = self.lo[2] + rng.uniform(self.p, nZ - 1 - self.p, npts) * self.h[2]
        x = np.stack([R * np.cos(P), R * np.sin(P), Z], -1)
        if min_dist is not None and coil_pts is not None:
            d = np.min(np.linalg.norm(x[:, None, :] - coil_pts[None, ::max(1, len(coil_pts) // 2000), :], axis=-1), axis=1)
            x = x[d > min_dist]
        Bg = np.asarray(self.evaluate(x)["B"])
        Be = np.asarray(self.engine.evaluate(x, grad=False, A=False)["B"])
        e = np.linalg.norm(Bg - Be, axis=1) / np.linalg.norm(Be, axis=1)
        ok = np.isfinite(e)
        return dict(max=float(np.max(e[ok])), median=float(np.median(e[ok])), n=int(ok.sum()))
