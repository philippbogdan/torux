"""The axis path's start for closed lines next to coils and for resonant axes.

On 297 CoilStellaration sources the flux-surface family never produced a torus although torux.analysis.find_axes
returned an elliptic period-one line (122 'near-axis torus did not converge', 176 'stalled' with no torus at
bn_off < 1e-6, one in both).  Three causes, all in how the family starts:

1. The line is not a field line (82: 81 of the first group, 1 of the second).  The Galerkin axis Newton
   (torux.axis.solve_axis) converges to a solution of the truncated system that no field line follows: unresolved
   (tail 1e-5..2e-2; N = 16..48 for 68 of them: find_axes's raise to the next N diverged and the unresolved line was
   kept), its residual between the collocation points 0.03..200 R0 (B_phi nearly vanishes on it), the line traced from
   its phi = 0 point leaves it by 7 cm (median; 25 of 82 lost within the period), also when solved at N = 192, 256.  find_axes then ends its search; a
   near-axis torus around a non-line does not exist, so the start's Newton stalls at a residual ~0.2 in every variant.
   They lie 0.7-2.6 cm from coil filaments (10-90 %), 0.1-0.6 m from the design axis.
2. The line is a field line next to a coil whose spectrum needs N > 128, find_axes's last rung (199: 36 and 163;
   194 at N = 128, 5 at N = 16, 48 whose solve at N = 192 lands on the field line next to them): tail 1e-11..5e-3, the
   traced line within 1e-4 m of it at N = 128..256 (median 1e-7 m), 1.3-3.9 cm from coils.  The near-axis torus starts
   at N0 = axis N = Nmax = 128, cannot be raised, and either does not converge or converges at bn_off 1e-6..1e-1 --
   above bn_accept and the range's 1e-6 -- so the family stalls and no torus is refined.  At N = 192-256 the same start
   reaches bn_off < 1e-6 on 110 of the 199; their tori then mostly need N > 320 (quick mode's refinement cap) to
   reach the acceptance bar.
3. A resolved axis next to a low-order resonance of its rotation per period (15: 1/3 for 8, 1/4 for 4, 1/5, ~0 for 2):
   from the circle at eps = 0.004 R0 Newton's first step moves iota by ~1e-2, across the small divisor (3 iota - nfp
   changes sign), and diverges -- or converges at the resonance to a torus that is not invariant (bn_off 1e-4) and the
   family stalls; the step scales as eps^2, and eps = 0.001 or 0.00025 R0 converges in 5-6 iterations to bn_off
   1e-10..1e-14.

The remedies (switch NEARAXIS['on'], on by default: TORUX_NEARAXIS=0 turns it off, or TORUX_NEARAXIS='{"on": ...}' overriding
entries):
- check_line (torux.analysis.find_axes, analyse's call only): a line still unresolved after find_axes's ladder
  (tail >= tail_ok) is kept only if its off-grid residual (a 4x finer grid, shifted half a spacing), as a position
  (divided by nfp N), is at most line_tol x R0 -- at its N, or at N = 192, 256 (solved from it); otherwise it is set
  aside (recorded, not returned) and the search goes on (the next guess, circles, the scan), as if find_axes had not
  found it.  Resolved lines are not evaluated.
- start_ladder (Continuation.start): when the start at eps = 0.004 R0 does not converge, eps = 0.001, 0.00025 R0;
  the ladder's solves, and the first attempt at N > 128 (only an axis check_line resolved), stop once diverged
  (torux.fast_family.diverged): a diverging first attempt at N = 192 spent the family's budget (300 s) before the ladder
  ran.  Starts at N <= 128 that converge are untouched (the rule would cut 4 of 176 slow converging ones).
- resolve_start (Continuation.start): when the start torus, raised, stays above start_bn at the axis's own N with
  the axis unresolved, the axis is resolved further (N = 192, 256), the family's toroidal cap raised to it, and the
  start solved there.
- a start that converged above start_bn with its rotation per period within res_tol of 1/3 or 1/4 (a torus in the
  resonance's island chain, not invariant: CoilStellaration D67czaES, axis rotation 0.3367, the start at 0.004 R0 at
  iota 1.66789, 0.0012 from 5/3, bn_off 2.7e-4) is solved on the ladder's smaller circles before it is raised
  (1.4e-10 at 0.001 R0); any start still above start_bn after raising and resolve_start is tried there too; the start
  with the smaller bn_off is kept.
Only sources whose start fails or whose start torus could not be a range candidate (bn_off >= 1e-6) or whose line is
not a field line take a different path.  Nothing here accepts anything: a torus still counts only through the
existing refinement, guard and verification.
"""
from __future__ import annotations

import json as _json
import os as _os

import numpy as np

NEARAXIS = dict(
    on=True,                # on; TORUX_NEARAXIS=0 turns it off
    tail_ok=1e-13,          # find_axes's own criterion of a resolved line (relative size of the last 3 harmonics)
    line_tol=1e-4,          # offgrid / (nfp N) (a position, relative to R0) above which an unresolved line is no field
                            # line (classification: field lines <= 1e-5 after the raise; of 82 non-lines 79 above it,
                            # 3 below -- kept, as before; no field line set aside)
    over=4,                 # the off-grid grid: over x the Galerkin grid's 3 N + 4 points per period, shifted
    eps=(0.001, 0.00025),   # the start radii (x R0) tried after Continuation.start's 0.004
    start_bn=1e-6,          # a start torus above this (torux.analysis's range-candidate bar) at the axis's N: resolve
    Ns=(192, 256),          # the axis's further resolutions (check_line at the last rung, resolve_start)
    res_q=(3, 4),           # the low-order resonances of the rotation per period a start torus is checked against
    res_tol=0.01,           # |iota / nfp - p / q| below which a start above start_bn is taken for one in an island chain
)


def opts():
    o = dict(NEARAXIS)
    env = (_os.environ.get("TORUX_NEARAXIS") or "").strip()
    if env.lower() in ("1", "on", "true"):
        o["on"] = True
    elif env.lower() in ("0", "off", "false"):
        o["on"] = False
    elif env:
        try:
            o.update(_json.loads(env))
        except ValueError:
            pass
    return o


def on():
    return bool(opts().get("on"))


def tail(ax):
    """Relative size of the last three harmonics of a closed line (torux.analysis.find_axes's measure)."""
    return float(max(np.max(np.abs(ax.rc[-3:])), np.max(np.abs(ax.zs[-3:]))) / max(abs(ax.rc[0]), 1e-300))


def offgrid_residual(engine, ax, over=4):
    """(max |R' - R B_R / B_phi|, |Z' - R B_Z / B_phi| / R0, max err_half) of the line ax (torux.axis.Axis) on
    over x (3 N + 4) points per period shifted half a spacing (between the Galerkin collocation points), with the
    engine's quadrature estimate there.  A field line resolved at N gives its truncation error; a solution of the
    truncated system that no field line follows gives 1e-3..1e-1."""
    nfp, N = ax.nfp, ax.N
    n = int(over) * (3 * N + 4)
    ph = (np.arange(n) + 0.5) * 2 * np.pi / nfp / n
    R, Z = ax.at(ph)
    Rp, Zp = ax.dphi(ph)
    x = np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)
    o = engine.evaluate(x, grad=False, A=False, err=True)
    B = np.asarray(o["B"], float)
    c, s = np.cos(ph), np.sin(ph)
    BR, Bp, BZ = B[:, 0] * c + B[:, 1] * s, -B[:, 0] * s + B[:, 1] * c, B[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = max(np.max(np.abs(Rp - R * BR / Bp)), np.max(np.abs(Zp - R * BZ / Bp))) / abs(float(ax.rc[0]))
    err = np.asarray(o.get("err_half", np.zeros(1)), float)
    return (float(r) if np.isfinite(r) else np.inf), (float(np.max(err)) if err.size else np.nan)


def check_line(engine, ax, solve_axis=None, o=None):
    """(line or None, record) for a converged closed line after find_axes's resolution ladder.  A resolved line
    (tail < tail_ok) is returned with record None, unevaluated.  An unresolved one is judged by its off-grid residual's
    position equivalent, offgrid / (nfp N) (a truncated line's derivative residual is its position error times the
    harmonic numbers ~ nfp N): returned as it is at most line_tol; above it, solved at N = 192, 256 first (with
    solve_axis: a field line next to a coil truncated at find_axes's N, or a nearby one) and returned there if that
    passes; otherwise None (not a field line)."""
    o = o or opts()
    t0 = tail(ax)
    if t0 < o["tail_ok"]:
        return ax, None
    r, err = offgrid_residual(engine, ax, o.get("over", 4))
    pos = r / (float(ax.nfp) * max(ax.N, 1))
    rec = dict(N=int(ax.N), R0=float(ax.rc[0]), tail=t0, offgrid=r, pos=pos, err_half=err)
    best = ax
    if pos > o["line_tol"] and solve_axis is not None:
        rec["raised"] = []
        for N2 in o["Ns"]:
            if best.N >= N2:
                continue
            a2 = solve_axis(engine, ax.nfp, float(best.rc[0]), N=N2, guess=(best.rc, best.zs))
            rr = dict(N=int(N2), converged=bool(a2.converged), tail=tail(a2))
            rec["raised"].append(rr)
            if not a2.converged:
                break
            best = a2
            rr["offgrid"], rr["err_half"] = offgrid_residual(engine, best, o.get("over", 4))
            pos = rr["pos"] = rr["offgrid"] / (float(best.nfp) * best.N)
            if pos <= o["line_tol"] or tail(best) < o["tail_ok"]:
                break
    rec.update(N_final=int(best.N), pos_final=pos, kept=bool(pos <= o["line_tol"]))
    return (best if rec["kept"] else None), rec


def at_resonance(iota, nfp, o=None):
    """Is the rotation per period iota / nfp within res_tol of p / q for q in res_q?"""
    o = o or opts()
    if not np.isfinite(iota):
        return False
    r = abs(float(iota) / float(nfp)) % 1.0
    for q in o["res_q"]:
        for p in range(0, q + 1):
            if abs(r - p / q) < o["res_tol"]:
                return True
    return False


def start_ladder(cont, axis, M, N, eps_list=None):
    """Continuation.start's guess (axis + circle, Newton at the circle's own flux) at smaller radii: the first that
    converges, else the last attempt.  Returns (sol, f0, eps, log); sol None if the budget allowed no attempt."""
    from .family import axis_torus_start
    from . import budget as _budget
    log = []
    sol, f0, fe = None, None, None
    iota = axis.iota if np.isfinite(axis.iota) else 0.0
    for fe in (eps_list if eps_list is not None else opts()["eps"]):
        if _budget.expired():
            break
        st = axis_torus_start(axis, M, N, float(fe) * axis.rc[0], iota)
        f0 = cont.problem(M, N).flux_of(st.rc, st.zs)
        sol = cont.solve(st, f0, M, N, maxit=25, diverge=True)
        log.append(dict(eps=float(fe), converged=bool(sol.converged), res=float(sol.res), its=int(sol.its)))
        if sol.converged:
            break
    return sol, f0, fe, log


def resolve_start(cont, axis, sol, M, eps_frac, o=None):
    """A converged start torus (raised) above start_bn at N = the axis's N with the axis unresolved: the axis resolved
    further along o['Ns'] (torux.axis.solve_axis from it), cont.Nmax raised to that N, the start solved there (same
    radius) and raised.  Returns (sol, f0, record) -- the better start, or (None, None, record) if nothing improved."""
    from .family import axis_torus_start
    from . import axis as AX
    from . import budget as _budget
    o = o or opts()
    rec = dict(bn=float(sol.bn_off), N=int(sol.N), axis_N=int(axis.N), axis_tail=tail(axis), raised=[])
    best_ax = axis
    for N2 in o["Ns"]:
        if best_ax.N >= N2 or tail(best_ax) < o["tail_ok"] or _budget.expired():
            continue
        a2 = AX.solve_axis(cont.engine, cont.nfp, float(best_ax.rc[0]), N=N2, guess=(best_ax.rc, best_ax.zs))
        rec["raised"].append(dict(N=int(N2), converged=bool(a2.converged), tail=tail(a2)))
        if not a2.converged:
            break
        best_ax = a2
    if best_ax is axis or _budget.expired():
        return None, None, rec
    best_ax.iota = axis.iota
    N2 = best_ax.N
    cont.Nmax = max(cont.Nmax, N2)
    st = axis_torus_start(best_ax, M, N2, eps_frac * best_ax.rc[0], axis.iota if np.isfinite(axis.iota) else 0.0)
    f0 = cont.problem(M, N2).flux_of(st.rc, st.zs)
    s2 = cont.solve(st, f0, M, N2, maxit=25, diverge=True)
    rec.update(converged=bool(s2.converged), bn_N2=float(s2.bn_off))
    if not s2.converged:
        return None, None, rec
    s2 = cont.raise_until(s2, f0, cont.bn_cont)
    rec.update(bn_raised=float(s2.bn_off), MN=[int(s2.M), int(s2.N)])
    if not (s2.bn_off < sol.bn_off):
        return None, None, rec
    return s2, f0, rec
