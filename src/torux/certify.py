"""Certificates that no flux surface encloses a requested flux, built from invariant tori only.

Island-band certificate.  Let T1 and T2 be verified rotational tori (around the magnetic axis)
enclosing fluxes F1 < F2, an island chain p/q between them, and I a verified island torus around
the chain's O line (period q field periods), whose cross-sections in one plane are q closed
curves C_k (the islands' circles), each enclosing flux f.  Suppose a rotational invariant torus
Gamma enclosed F with F1 < F < F2.  Then (i) Gamma meets neither T1, T2 nor any C_k: invariant
circles of different rotation number cannot share a point (the orbit of a common point would
have two rotation numbers; Gamma's rotation number cannot be p/q, else it would carry p/q
periodic orbits and, not passing through the O points, separate them from their island circles
it does not meet -- the degenerate unsplit separatrix encloses exactly the separatrix flux,
outside the open interval below); so Gamma lies between T1 and T2.  (ii) The map permutes the
islands and preserves the inside of Gamma, so all islands lie on one side of Gamma.  (iii) If
all are inside, Gamma's interior contains T1's and the q disjoint island disks: F >= F1 + q f;
if all are outside, Gamma's interior lies in T2's minus the island disks: F <= F2 - q f.
Hence no rotational torus encloses a flux in the open interval (F2 - q f, F1 + q f), which is
nonempty when 2 q f > F2 - F1.  Fluxes are |Phi| with the field's orientation.

The ingredients are tori computed and checked like any other (off-grid B.n, guard, independent
check); the island torus additionally must lie outside T1 and inside T2 (checked on the grid).
"""
from __future__ import annotations

import math

import numpy as np

from .family import Continuation, line_torus_start, refine_to_limit
from . import orbits as OR


def island_tori(engine, line: OR.ClosedLine, M0=6, N0=None, target_frac=0.9, verbose=False, deadline=None,
                width_est=None, Mmax=96, Nmax=256):
    """Tori around a closed (O) line of period q, continued outward from the line in their own
    flux; returns the list of accepted tori (TorusState with nfp = nfp/q, phi0 = line.phi0).

    width_est: optional estimate of one island's flux (Wb); the continuation aims at
    target_frac of it and stops where it stalls (the separatrix or secondary chains)."""
    from .family import LADDER
    nfp_eff = line.nfp / line.q
    # the torus must carry the O line's whole spectrum (K harmonics over its q periods)
    N = N0 or next((L for L in LADDER if L >= max(16, line.K)), LADDER[-1])
    Nmax = max(Nmax, 2 * N)
    cont = Continuation(engine, nfp_eff, M0=M0, N0=N, verbose=verbose, Mmax=Mmax, Nmax=Nmax, phi0=line.phi0)
    # scale of the start: a small fraction of the line's distance to its neighbours
    pts = line.section_points(line.phi0)
    if len(pts) > 1:
        d = np.min([np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i)])
    else:
        d = 0.01 * line.a[0]
    eps = 1e-3 * d
    # the transform around the O line from its monodromy (rotation per q-period map)
    io0 = 0.0
    if line.monodromy is not None:
        tr = float(np.trace(line.monodromy))
        if abs(tr) < 2:
            io0 = math.acos(tr / 2) / (2 * math.pi) * nfp_eff
    st = line_torus_start(line, M0, N, eps, iota=io0)
    f0 = cont.problem(M0, N).flux_of(st.rc, st.zs)
    sol = cont.solve(st, f0, M0, N, maxit=40)
    if not sol.converged:
        return [], cont
    target = width_est * target_frac if width_est else abs(f0) * 1e6
    fam = cont.run(None, abs(target), start=sol, deadline=deadline)
    return fam.tori, cont


def island_band(F1, F2, q, f_island):
    """The excluded open interval (F2 - q f, F1 + q f) (|flux|), or None if empty."""
    lo, hi = F2 - q * f_island, F1 + q * f_island
    return (lo, hi) if lo < hi else None


def _section_polygon(t, phi, n=2048):
    """Cross-section curve of a torus (TorusState, any nfp/phi0) in the plane phi: (n, 2)."""
    th = 2 * np.pi * np.arange(n) / n
    R, Z = t.geometry_on(th, np.full(n, phi))
    return np.stack([R, Z], -1)


def _inside(poly, pts):
    """Even-odd point-in-polygon test (pts (k, 2))."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y0 > y) != (y1 > y)
    xin = x0 + (y - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1e-300)
    return (np.sum(cond & (x < xin), axis=1) % 2) == 1


def island_band_certificate(T1, T2, island_torus, line, request_flux=None):
    """The island-band certificate (module docstring) from verified tori T1 (inside the chain),
    T2 (outside) and an island torus around the chain's O line.  Checks the geometric conditions
    (the island circles lie outside T1 and inside T2 in the O line's plane, and do not cross
    them) and returns dict(ok, interval, holds_request, details)."""
    q = line.q
    F1, F2, f = abs(T1.flux), abs(T2.flux), abs(island_torus.flux)
    phi0 = line.phi0
    out = dict(F1=F1, F2=F2, q=q, f_island=f, phi0=phi0)
    P1, P2 = _section_polygon(T1, phi0), _section_polygon(T2, phi0)
    # the island torus' q cross-sections in the plane phi0 (+ k field periods)
    P = 2 * np.pi / line.nfp
    circles = [_section_polygon(island_torus, phi0 + k * P) for k in range(q)]
    pts = np.concatenate(circles)
    in1 = _inside(P1, pts)
    in2 = _inside(P2, pts)
    out["islands_outside_T1"] = bool(not np.any(in1))
    out["islands_inside_T2"] = bool(np.all(in2))
    # the sampled curves must also keep apart by more than their sampling error (a curve of n samples
    # deviates from its polygon by at most the largest chord sagitta, ~ h^2 kappa / 8 <= h / 4 at the
    # samples' spacing h for curvature radius >= h/2): min distance > 2 (h_1 + h_2) / 4 between each pair
    from scipy.spatial import cKDTree

    def spacing(C):
        return float(np.max(np.linalg.norm(np.roll(C, -1, 0) - C, axis=1)))
    hi = max(spacing(c) for c in circles)
    d1 = float(np.min(cKDTree(P1).query(pts)[0]))
    d2 = float(np.min(cKDTree(P2).query(pts)[0]))
    out["min_dist_T1_islands"], out["min_dist_T2_islands"] = d1, d2
    out["sampling_margin"] = dict(T1=0.5 * (spacing(P1) + hi), T2=0.5 * (spacing(P2) + hi))
    # the q island sections must be pairwise disjoint (F >= F1 + q f counts q disjoint discs)
    dij = np.inf
    for i in range(len(circles)):
        tree_i = cKDTree(circles[i])
        for j in range(i + 1, len(circles)):
            dij = min(dij, float(np.min(tree_i.query(circles[j])[0])))
        if len(circles) > 1:
            # each section must also be disjoint from the others' interiors (nested sections would share area)
            others = np.concatenate([c for k, c in enumerate(circles) if k != i])
            if np.any(_inside(circles[i], others)):
                dij = -1.0
    out["min_dist_between_islands"] = float(dij)
    sep_ok = d1 > out["sampling_margin"]["T1"] and d2 > out["sampling_margin"]["T2"] and dij > hi
    out["separated"] = bool(sep_ok)
    band = island_band(F1, F2, q, f)
    out["interval"] = band
    out["ok"] = bool(band is not None and out["islands_outside_T1"] and out["islands_inside_T2"] and sep_ok
                     and F1 < F2)
    if request_flux is not None and band is not None:
        out["holds_request"] = bool(band[0] < abs(request_flux) < band[1])
        out["margin_rel"] = float(min(abs(request_flux) - band[0], band[1] - abs(request_flux)) / abs(request_flux))
    return out
