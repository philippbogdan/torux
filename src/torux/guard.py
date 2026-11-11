"""The resonance guard: is a computed torus a true flux surface outside every near-resonant chain?

The off-grid B.n measures a torus's error only through the harmonics it carries; a resonant
harmonic k = (m, n) with small divisor d = |m iota - n nfp| that it lacks enters B.n multiplied
by d.  The forcing the torus leaves at k is eps = |[B.N]_k| (N = X_theta x X_phi in its own
straight-field-line angles; B.N is the flux crossing the torus per radian^2, Wb).  With the
shear iota' = d iota / d Phi' (Phi' = flux / 2 pi) the pendulum model of the chain has half
width w = 2 sqrt(eps / (m |iota'|)) in Phi' and puts the torus x = d / (m |iota'|) from the
resonant surface; h = m |iota'| eps / d^2 = (w / x)^2 / 4 is the Newton-Kantorovich quantity
of the mode (h = 1/4 at the separatrix).  Accept for h <= 1/16 at every near-resonant k
(m <= 2 M + 10), i.e. the torus at least two half-widths outside every chain.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import jax.numpy as jnp

from .torus import Basis, field_on_grid

H_MAX = 1.0 / 16.0
PREFILTER = 1e-4


def bdotN_harmonics(engine, torus, harmonics, extra=8, symmetric=True):
    """|[B.N]_(m,n)| at the harmonics (phase m theta - n psi) and the scale mean |B||N|."""
    Mq = max([m for m, _ in harmonics] + [torus.M]) + extra
    Nq = max([abs(n) for _, n in harmonics] + [torus.N]) + extra
    b = Basis(torus.nfp, Mq, Nq)
    st = torus.resized(Mq, Nq)
    R, Z, Rt, Zt, Rp, Zp = b.geometry(jnp.asarray(st.rc), jnp.asarray(st.zs))
    g = field_on_grid(engine, b, R, Z, need_grad=False, need_A=False, symmetric=symmetric)
    nR, nP, nZ = -Zt * R, Zt * Rp - Rt * Zp, Rt * R
    BN = g["BR"] * nR + g["Bp"] * nP + g["BZ"] * nZ
    scale = float(jnp.mean(jnp.sqrt(g["BR"] ** 2 + g["Bp"] ** 2 + g["BZ"] ** 2) * jnp.sqrt(nR ** 2 + nP ** 2 + nZ ** 2)))
    # B.N is odd under the stellarator image (sin series) for symmetric tori; take both parts
    c = np.asarray(b.proj_c(BN)); s = np.asarray(b.proj_s(BN))
    out = []
    for m, n in harmonics:
        j = n + Nq
        out.append(float(np.hypot(c[m, j], s[m, j])))
    return out, scale


def near_resonances(iota, nfp, mmax, thr):
    """Harmonics (m, n), m <= mmax, gcd(m, n) = 1, with 0 < |m iota - n nfp| < thr."""
    io = float(iota)
    out = []
    for m in range(1, mmax + 1):
        n0 = int(np.round(io * m / nfp))
        for n in (n0 - 1, n0, n0 + 1):
            d = abs(m * io - n * nfp)
            if 0 < d < thr and gcd(m, abs(n)) == 1:
                out.append((m, n, d))
    return out


def resonance_guard(engine, torus, shear, mmax=None, bn=None, symmetric=True):
    """dict(ok, h, x_over_w, worst=(m, n, d, eps), checked=[...]).  shear = d iota / d Phi' (1/Wb)."""
    mmax = mmax or 2 * torus.M + 10
    bn = torus.bn_off if bn is None else bn
    out = dict(ok=True, h=0.0, x_over_w=np.inf, worst=None, checked=[], shear=shear)
    if not np.isfinite(bn):
        return dict(out, ok=False, h=np.inf, reason="bn not finite")
    # a harmonic matters only if eps / scale / d^2 can exceed PREFILTER; eps <= ~2 bn scale
    thr = np.sqrt(2.0 * max(bn, 1e-300) / PREFILTER)
    harm = near_resonances(torus.iota, torus.nfp, mmax, thr)
    if not harm:
        return out
    eps, scale = bdotN_harmonics(engine, torus, [(m, n) for m, n, _ in harm], symmetric=symmetric)
    out["scale"] = scale
    for (m, n, d), e in zip(harm, eps):
        h = m * abs(shear) * e / d ** 2 if np.isfinite(shear) else np.inf
        out["checked"].append(dict(m=m, n=n, d=d, eps=e, eps_rel=e / scale, h=h))
        if h > out["h"]:
            out.update(h=float(h), worst=(m, n, d, e), x_over_w=float(0.5 / np.sqrt(h)) if h > 0 else np.inf)
    out["ok"] = bool(out["h"] <= H_MAX)
    return out


def family_shear(tori):
    """d iota / d Phi' along a family (central differences in Phi' = flux / 2 pi), per torus."""
    if len(tori) < 2:
        return [np.nan] * len(tori)
    P = np.array([t.flux for t in tori]) / (2 * np.pi)
    I = np.array([t.iota for t in tori])
    g = np.gradient(I, P)
    return list(g)
