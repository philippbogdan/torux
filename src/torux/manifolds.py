"""Unstable manifolds of hyperbolic closed field lines (X lines) in a Poincare section.

The q-period map P (section phi = phi0, the X line's symmetry plane) has the X point x* with
monodromy M, unstable eigenvalue lambda (|lambda| > 1; lambda < 0 for reflection-hyperbolic
lines, then P^2 is used) and eigenvector v.  A branch is grown from a fundamental domain
x* + s v, s in [s0, lambda s0) (side = +-1), by mapping the latest piece of the curve with P
and inserting points (by interpolation along the previous piece; transverse errors contract
under forward iteration, so this is stable) wherever consecutive images are farther apart
than ``dmax`` or the curve turns by more than ``amax`` between them (Krauskopf-Osinga style
growth).  The curve is a polyline of section points with, per point, the iterate count.

Uses: island-chain separatrices (closest approach of a branch to the neighbouring X points),
lost-separatrix edge certificates (a branch point whose forward orbit leaves the domain), and
the fluxes of the regions they bound (loop integrals of A along closed polylines).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from . import trace as T

jax.config.update("jax_enable_x64", True)


@dataclass
class Branch:
    pts: np.ndarray                  # (n, 2) section points (R, Z) along the branch, from x* outward
    it: np.ndarray                   # (n,) iterate count of each point
    lam: float
    x_star: np.ndarray
    v: np.ndarray
    side: int
    q: int
    phi0: float
    lost_at: int = -1                # index of the first point found lost (orbit left the domain), or -1
    info: dict = field(default_factory=dict)


class SectionMap:
    """P^q from phi0 to phi0 + q 2 pi / nfp for many points at once (jitted), with a lost flag
    for lines that leave the box or whose B_phi changes sign (non-finite results)."""

    def __init__(self, engine, nfp, q, phi0=0.0, steps_per_period=12, k=6, box=None):
        self.engine, self.nfp, self.q, self.phi0 = engine, nfp, q, phi0
        self.f = T.make_rhs(engine)
        period = 2 * np.pi / nfp
        self.box = box   # (Rmin, Rmax, Zmax)

        @jax.jit
        def one_period(y, ph):
            yy, e = T.trace(engine, y, ph, period, steps_per_period, k, self.f)
            return yy, e
        self._one = one_period
        self.period = period

    def __call__(self, y, nper=None):
        y = jnp.asarray(y)
        nper = nper or self.q
        err = jnp.zeros(y.shape[0])
        for i in range(nper):
            y, e = self._one(y, self.phi0 + i * self.period)
            err = err + e
        y = np.asarray(y)
        lost = ~np.all(np.isfinite(y), axis=1)
        if self.box is not None:
            Rmin, Rmax, Zmax = self.box
            lost |= (y[:, 0] < Rmin) | (y[:, 0] > Rmax) | (np.abs(y[:, 1]) > Zmax)
        return y, lost, np.asarray(err)


def eig_unstable(M):
    w, V = np.linalg.eig(M)
    i = int(np.argmax(np.abs(w)))
    lam = float(np.real(w[i]))
    v = np.real(V[:, i])
    return lam, v / np.linalg.norm(v)


def grow_branch(smap: SectionMap, x_star, M, side=+1, s0=None, scale=1.0, n0=24, dmax=None, amax=0.3,
                max_iter=2000, max_pts=200000, stop=None, deadline=None, verbose=False):
    """Grow one branch of W^u(x*).  stop(pts_new) -> bool ends the growth (e.g. closest approach
    to a target reached).  Returns a Branch."""
    import time
    lam, v = eig_unstable(np.asarray(M))
    twice = lam < 0
    if twice:
        lam = lam * lam
    if lam <= 1.0:
        raise ValueError("not hyperbolic")
    mu = lam - 1.0
    # start at a distance where the linear approximation is accurate: weaker hyperbolicity
    # needs a larger start (the growth per map is (lambda - 1) s)
    if s0 is None:
        s0 = scale * (1e-6 if mu > 0.1 else 1e-5 if mu > 0.01 else 1e-3 if mu > 1e-4 else 3e-3)
    dmax = dmax or 0.01 * scale
    nper = smap.q * (2 if twice else 1)
    s = s0 * lam ** (np.arange(n0) / n0)
    piece = np.asarray(x_star)[None, :] + side * s[:, None] * v[None, :]
    pts = [piece]
    its = [np.zeros(len(piece), int)]
    lost_at = -1
    total = len(piece)
    for k in range(1, max_iter + 1):
        if deadline is not None and time.time() > deadline:
            break
        img, lost, _ = smap(piece, nper)
        # refinement: where consecutive images are far apart or turn sharply, insert midpoints of the pre-images
        for _ in range(12):
            d = np.linalg.norm(np.diff(img, axis=0), axis=1)
            bad = d > dmax
            if len(img) >= 3:
                t1 = img[1:-1] - img[:-2]; t2 = img[2:] - img[1:-1]
                cosang = np.sum(t1 * t2, 1) / (np.linalg.norm(t1, axis=1) * np.linalg.norm(t2, axis=1) + 1e-300)
                sharp = np.arccos(np.clip(cosang, -1, 1)) > amax
                bad[:-1] |= sharp
                bad[1:] |= sharp
            bad &= ~(lost[:-1] | lost[1:])
            if not np.any(bad):
                break
            idx = np.flatnonzero(bad)
            mids = 0.5 * (piece[idx] + piece[idx + 1])
            mimg, mlost, _ = smap(mids, nper)
            piece = np.insert(piece, idx + 1, mids, axis=0)
            img = np.insert(img, idx + 1, mimg, axis=0)
            lost = np.insert(lost, idx + 1, mlost)
            if len(piece) > max_pts:
                break
        # the new piece starts where the previous one ended (continuity): drop the first point (image of
        # the previous piece's first point equals the previous piece's last point up to the fundamental-domain wrap)
        pts.append(img)
        its.append(np.full(len(img), k))
        total += len(img)
        if np.any(lost) and lost_at < 0:
            lost_at = total - len(img) + int(np.argmax(lost))
        if verbose and k % 50 == 0:
            print(f"   branch it {k}: {total} pts, extent {np.max(np.linalg.norm(img - x_star, axis=1)):.3e}")
        if stop is not None and stop(img):
            break
        if np.all(lost):
            break
        piece = img[~lost] if np.any(lost) else img
        if total > max_pts:
            break
    P = np.concatenate(pts)
    I = np.concatenate(its)
    return Branch(P, I, lam, np.asarray(x_star), v, side, smap.q, smap.phi0, lost_at,
                  info=dict(s0=s0, twice=twice, iters=k))


def polygon_flux(engine, poly, phi0):
    """Toroidal flux through the region bounded by a closed polyline (R, Z) in the plane phi0:
    loop integral of A (Gauss 3-point per edge), counter-clockwise orientation -> int B_phi dR dZ."""
    P = np.asarray(poly)
    Q = np.roll(P, -1, axis=0)
    g = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    w = np.array([5 / 9, 8 / 9, 5 / 9])
    tot = 0.0
    pts = []
    for gi in g:
        pts.append(0.5 * (P + Q) + 0.5 * gi * (Q - P))
    pts = np.concatenate(pts)
    c, s = np.cos(phi0), np.sin(phi0)
    x = np.stack([pts[:, 0] * c, pts[:, 0] * s, pts[:, 1]], -1)
    A = np.asarray(engine.evaluate(x, grad=False, A=True)["A"])
    AR = A[:, 0] * c + A[:, 1] * s
    AZ = A[:, 2]
    n = len(P)
    dR = (Q - P)[:, 0]; dZ = (Q - P)[:, 1]
    for j in range(3):
        sl = slice(j * n, (j + 1) * n)
        tot += 0.5 * w[j] * np.sum(AR[sl] * dR + AZ[sl] * dZ)
    area = 0.5 * np.sum(P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1])
    return float(-tot if area > 0 else tot), float(abs(area))
