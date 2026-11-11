"""Upper bounds on the flux any flux surface can enclose, from field lines that escape.

Let U be the region where B_phi keeps one sign with |B_phi| >= 1e-3 |B| (torux.trace marks lines
leaving it as lost: they cannot be followed in phi), restricted to a box around the device.  An
invariant torus Gamma whose interior lies in U confines every line started inside it: such a
line stays inside Gamma forever and is never lost.  Hence a lost line starts outside every such
torus, and so does every point of its orbit.

Exterior bound.  In the plane phi0, rays from the axis point carry start points r_j(theta_i).
r*(theta_i) is the smallest sampled radius beyond which every sampled point of the ray escaped
within the traced length.  A torus encloses the axis and no escaped point, so on each ray it
crosses the ray inside r*(theta_i) (a crossing beyond would put a point of the torus -- which
never escapes -- where every sample escaped: excluded up to the sampling, see below).  The
polygon C through the points r*(theta_i) then contains every torus's section, and
Phi(Gamma) <= Phi(C) (loop integral of A).  The sampling assumption (no torus reaches between
two neighbouring rays, or between two radial samples, beyond the escaped samples there) is
tested by refinement: the bound is recomputed with twice the rays and radial samples and must
agree; the bound is reported with that check and with the traced lengths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from . import trace as T

jax.config.update("jax_enable_x64", True)


def staged_escape(engine, nfp, pts, phi0=0.0, stages=(4, 40, 400), steps_per_period=10, k=4, box=None,
                  batch=8192, deadline=None):
    """Periods after which each start point (n, 2) in the plane phi0 was lost (NaN or out of the
    box), or -1 if it survived all stages; lines are traced in stages, survivors continued."""
    import time
    f = T.make_rhs(engine)
    period = 2 * np.pi / nfp

    @jax.jit
    def one(y, ph):
        yy, _ = T.trace(engine, y, ph, period, steps_per_period, k, f)
        return yy
    n = len(pts)
    lost_at = np.full(n, -1, int)
    y_all = np.array(pts, float)
    alive = np.ones(n, bool)
    done_periods = 0
    for st in stages:
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        for lo in range(0, len(idx), batch):
            sel = idx[lo:lo + batch]
            y = jnp.asarray(y_all[sel])
            a = np.ones(len(sel), bool)
            for i in range(done_periods, st):
                if deadline is not None and time.time() > deadline:
                    break
                y = one(y, phi0 + i * period)
                yn = np.array(y)          # a writable copy (np.asarray of a JAX array is read-only)
                bad = ~np.all(np.isfinite(yn), axis=1)
                if box is not None:
                    Rmin, Rmax, Zmax = box
                    bad |= (yn[:, 0] < Rmin) | (yn[:, 0] > Rmax) | (np.abs(yn[:, 1]) > Zmax)
                newly = bad & a
                lost_at[sel[newly]] = i + 1
                a &= ~bad
                if not a.any():
                    break
                if bad.any():
                    yn[bad] = yn[a][0] if a.any() else yn[bad]
                    y = jnp.asarray(yn)
            y_all[sel] = np.asarray(y)
            alive[sel] = a
        done_periods = st
    return lost_at


def exterior_bound(engine, nfp, centre, r_in, r_out, nray=128, nrad=48, phi0=0.0, stages=(4, 40, 400),
                   box=None, deadline=None):
    """r*(theta) on nray rays from centre between r_in (inside: e.g. the last verified torus's
    distance) and r_out, and the polygon of the bound.  Returns dict."""
    th = 2 * np.pi * (np.arange(nray) + 0.5) / nray
    r_in = np.broadcast_to(np.asarray(r_in, float), (nray,))
    r_out = np.broadcast_to(np.asarray(r_out, float), (nray,))
    tt = np.linspace(0.0, 1.0, nrad)
    rr2 = r_in[:, None] + (r_out - r_in)[:, None] * tt[None, :]          # (nray, nrad)
    R = centre[0] + rr2 * np.cos(th)[:, None]
    Z = centre[1] + rr2 * np.sin(th)[:, None]
    pts = np.stack([R.ravel(), Z.ravel()], -1)
    lost = staged_escape(engine, nfp, pts, phi0, stages, box=box, deadline=deadline).reshape(nray, nrad)
    esc = lost >= 0
    rstar = np.full(nray, np.inf)
    for i in range(nray):
        keep = np.flatnonzero(~esc[i])
        j = 0 if len(keep) == 0 else keep[-1] + 1
        if j < nrad:
            rstar[i] = rr2[i, j]
    poly = np.stack([centre[0] + rstar * np.cos(th), centre[1] + rstar * np.sin(th)], -1)
    return dict(theta=th, radii=rr2, lost=lost, rstar=rstar, polygon=poly, complete=bool(np.all(np.isfinite(rstar))))


def ray_crossings(poly, centre, theta):
    """Largest distance from centre at which each ray theta meets the closed polygon (n, 2)."""
    P = np.asarray(poly) - np.asarray(centre)[None, :]
    Q = np.roll(P, -1, axis=0)
    out = np.zeros(len(theta))
    for i, t in enumerate(theta):
        d = np.array([np.cos(t), np.sin(t)])
        # solve P + s (Q - P) = r d for s in [0, 1], r > 0
        e = Q - P
        den = e[:, 0] * d[1] - e[:, 1] * d[0]
        ok = np.abs(den) > 1e-300
        s_ = np.where(ok, (P[:, 1] * d[0] - P[:, 0] * d[1]) / np.where(ok, den, 1.0), -1.0)
        r = np.where(ok, (P[:, 0] + s_ * e[:, 0]) * d[0] + (P[:, 1] + s_ * e[:, 1]) * d[1], -1.0)
        hit = ok & (s_ >= 0) & (s_ <= 1) & (r > 0)
        out[i] = np.max(r[hit]) if hit.any() else np.nan
    return out


def polygon_flux_A(engine, poly, phi0):
    from .manifolds import polygon_flux
    return polygon_flux(engine, poly, phi0)
