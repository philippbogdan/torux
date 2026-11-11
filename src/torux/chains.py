"""Island chains next to a torus family: their closed lines, residues and estimated bands; jumps.

Conventions: iota per toroidal turn, rotation per field period rho = iota / nfp; the chain
p/q (reduced, p signed like iota) has its lines close after q field periods with p poloidal
turns, resonant harmonic (m, n) = (q, p) of the torus basis cos(m theta - n nfp phi).

Pendulum estimate of the chain's half width in enclosed flux Phi (Wb), from the O line's
residue R_O and the shear diota/dPhi at the resonance (standard-map normal form of the
q-period map, twist T = 2 pi q drho/dJ, J = Phi / 2 pi):
    dPhi_half = 2 nfp sqrt(R_O) / (pi q |d iota / d Phi|)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import budget as _budget
from . import orbits as OR
from .torus import TorusState


@dataclass
class Chain:
    p: int
    q: int
    iota_r: float
    lines: list = field(default_factory=list)       # ClosedLine (converged, with residues)
    O: list = field(default_factory=list)
    X: list = field(default_factory=list)
    flux_r: float = np.nan                           # flux of the resonant surface (interpolated)
    shear: float = np.nan                            # d iota / d Phi at the resonance
    half_width_est: float = np.nan                   # pendulum estimate (Wb)
    info: dict = field(default_factory=dict)


def rationals_between(rho_a, rho_b, qmax):
    """Reduced p/q (q <= qmax) strictly between rho_a and rho_b (either order)."""
    lo, hi = min(rho_a, rho_b), max(rho_a, rho_b)
    out = set()
    for q in range(1, qmax + 1):
        for p in range(math.ceil(lo * q), math.floor(hi * q) + 1):
            if lo < p / q < hi and math.gcd(p, q) == 1:
                out.add((p, q))
    return sorted(out, key=lambda t: t[1])


def nearest_rational_ahead(iota, diota, nfp, qmax=40, window=0.05):
    """The lowest-order rational rho = p/q ahead of iota in the direction diota, within window*|iota|."""
    rho = iota / nfp
    step = math.copysign(window * max(abs(iota), 1e-3) / nfp, diota if diota != 0 else 1.0)
    cands = rationals_between(rho, rho + step, qmax)
    if not cands:
        return None
    # lowest q first; among equal q the nearest
    cands.sort(key=lambda t: (t[1], abs(t[0] / t[1] - rho)))
    return cands[0]


def chain_lines(engine, torus: TorusState, p, q, K=None, verbose=False, deadline=None, info=None):
    """Closed lines of the chain p/q started on the torus's curves theta = theta0 + iota_r phi for the
    symmetric starting angles; returns converged, distinct lines with residues.  deadline (absolute, None: none):
    no start after it and each Newton cut at it (info["cut"] = True then; the lines found so far are returned)."""
    nfp = torus.nfp
    iota_r = p / q * nfp
    starts = []
    for th in (0.0, math.pi, -math.pi * p / q, math.pi - math.pi * p / q):
        starts.append(th)
    K = K or max(24, 10 * q + 2 * torus.N)
    lines = []
    for th0 in starts:
        for phi0 in (0.0, math.pi / nfp):
            # the line through the symmetry plane phi0 at theta0 + iota_r phi0 in {0, pi}
            gR = lambda ph, t0=th0: torus.geometry_on(t0 + iota_r * ph, ph)[0]
            gZ = lambda ph, t0=th0: torus.geometry_on(t0 + iota_r * ph, ph)[1]
            thc = th0 + iota_r * phi0
            if not (abs(math.sin(thc)) < 1e-9):
                continue
            if deadline is not None and time.time() > deadline:
                if info is not None:
                    info["cut"] = True
                return lines
            ln = OR.solve_closed_line(engine, nfp, q, gR, gZ, phi0=phi0, K=K, symmetric=True, deadline=deadline)
            if ln.info.get("cut") and info is not None:
                info["cut"] = True
            if not ln.converged:
                if verbose:
                    print(f"    chain {p}/{q} start theta0 {th0:.3f} phi0 {phi0:.3f}: not converged ({ln.res:.2e})")
                continue
            ln = OR.line_monodromy(engine, ln)
            pts = ln.section_points(0.0)
            dup = False
            for other in lines:
                op = other.section_points(0.0)
                dmin = np.min(np.linalg.norm(pts[:, None, :] - op[None, :, :], axis=-1))
                if dmin < 1e-8:
                    dup = True
                    break
            if verbose:
                print(f"    chain {p}/{q} start theta0 {th0:.3f} phi0 {phi0:.3f}: residue {ln.residue:.4e} dup {dup}")
            if not dup:
                lines.append(ln)
    return lines


def analyse_chain(engine, tori, p, q, verbose=False, deadline=None):
    """Chain p/q next to the family `tori` (TorusState list): lines, residues, flux and shear at
    the resonance (interpolated in the family), pendulum half width.  deadline: chain_lines' (ch.info["cut"])."""
    nfp = tori[-1].nfp
    iota_r = p / q * nfp
    ch = Chain(p, q, iota_r)
    # the family torus nearest in iota as the starting surface
    k = int(np.argmin([abs(t.iota - iota_r) for t in tori]))
    ch.lines = chain_lines(engine, tori[k], p, q, verbose=verbose, deadline=deadline, info=ch.info)
    ch.O = [l for l in ch.lines if 0 < l.residue < 1]
    ch.X = [l for l in ch.lines if not (0 < l.residue < 1)]
    fl = np.array([t.flux for t in tori]); io = np.array([t.iota for t in tori])
    if len(tori) >= 2:
        dI = np.gradient(io, fl)
        # linear extrapolation from the last two tori in the direction of the resonance
        j = k if 0 < k < len(tori) else len(tori) - 1
        ch.shear = float(dI[j])
        if ch.shear != 0:
            ch.flux_r = float(fl[j] + (iota_r - io[j]) / ch.shear)
    if ch.O and np.isfinite(ch.shear) and ch.shear != 0:
        RO = max(l.residue for l in ch.O)
        ch.half_width_est = float(2 * nfp * math.sqrt(RO) / (math.pi * q * abs(ch.shear)))
    return ch


def noble_offset(iota_r, nfp, q, dist):
    """An iota beyond the resonance by about dist, with a noble (golden) correction so that the
    rotation number is far from every low-order rational."""
    g = (math.sqrt(5) - 1) / 2
    return iota_r + dist * (1 + g * 1e-3)


def jump_over(cont, tori, chain: Chain, direction, max_tries=6, verbose=False, axis=None):
    """Fixed-iota Newton beyond the chain: iota = iota_r + direction * dist (1 + noble), dist from
    the chain's estimated width (or 1e-4 |iota|) doubling per try; predictor extrapolated from the
    family.  Returns the torus beyond (TorusState) or None."""
    last = tori[-1]
    nfp = last.nfp
    # candidate distances beyond the resonance (in iota): a geometric ladder from 1e-5 |iota| to
    # 3e-2 |iota|, with the pendulum estimate's 1.25 half widths (and 1.5, 2x) inserted; sorted, so
    # the nearest torus beyond the chain is tried first (the pendulum width is unreliable for wide,
    # strongly sheared chains)
    io = max(abs(last.iota), 1e-2)
    cands = list(io * np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]))
    if np.isfinite(chain.half_width_est) and np.isfinite(chain.shear) and chain.shear != 0:
        w = chain.half_width_est * abs(chain.shear)
        cands += [1.25 * w, 1.5 * w, 2.0 * w]
    cands = sorted(set(float(c) for c in cands))
    # distances well inside the chain are futile: the family itself stopped |iota_r - iota_last| short of the
    # resonance on its side, and the chain's half width on the far side is of the same order, so candidates
    # below a quarter of that gap are skipped (CoilStellaration D8r9jBW9...: the first five of eleven tries sat
    # 3e-6..3e-4 from the 2/7 resonance after the family had stopped 3.6e-3 short of it; each try two
    # high-resolution solves)
    gap = abs(chain.iota_r - last.iota)
    cands = [c for c in cands if c > 0][: max(max_tries, 8)] if gap == 0 else \
        ([c for c in cands if c >= 0.25 * gap] or cands[-3:])
    for t, dist in enumerate(cands):
        if _budget.expired():
            return None             # (each try would stop at once, after setting up its high-resolution solves)
        target_iota = noble_offset(chain.iota_r, nfp, chain.q, direction * dist)
        # predictor: extrapolate the family in iota to the target
        if len(tori) >= 2:
            a, b = tori[-2], tori[-1]
            if b.iota != a.iota:
                lam = (target_iota - b.iota) / (b.iota - a.iota)
            else:
                lam = 1.0
            lam = float(np.clip(lam, -2, 20))
            ap = a.resized(b.M, b.N)
            guess = TorusState(b.rc + lam * (b.rc - ap.rc), b.zs + lam * (b.zs - ap.zs), target_iota,
                               b.flux + lam * (b.flux - a.flux), b.M, b.N, nfp)
        else:
            guess = TorusState(last.rc.copy(), last.zs.copy(), target_iota, last.flux, last.M, last.N, nfp)
        for (M, N) in dict.fromkeys(((last.M, last.N), cont.grow(last))):
            # patience 3: a fixed-iota Newton inside the chain does not converge, and each of its iterations is a
            # high-resolution solve (CoilStellaration D8r9jBW9...: 22 such tries at M = 80-112, N = 128, 20
            # iterations each, were most of its 30 minutes)
            sol = cont.solve(guess, guess.flux, M, N, fixed_iota=target_iota, maxit=20, patience=3)
            if verbose:
                print(f"   jump try {t} iota {target_iota:.10f} M,N {M},{N} conv {sol.converged} bn {sol.bn_off:.2e} "
                      f"flux {sol.flux:.6e}")
            if (sol.converged and sol.bn_off < cont.bn_accept and abs(sol.flux) > abs(last.flux)
                    and sol.flux * last.flux > 0):
                # only a torus of the axis's family: its sections simple closed curves around the axis
                # (CoilStellaration DDMaikF39a8...: a 1/3 jump converged onto a curve through the chain's X line
                # that wound -2 times around the axis, with the flux sign flipped)
                from .analysis import torus_topology
                topo = torus_topology(sol, axis)
                if not topo["ok"]:
                    if verbose:
                        print(f"   jump try {t}: rejected, sections not simple around the axis ({topo})", flush=True)
                    continue
                sol = cont.raise_until(sol, sol.flux, cont.bn_cont, fixed_iota=target_iota)
                return sol
    return None
