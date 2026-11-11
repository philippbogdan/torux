"""The design path's magnetic axis: the period-one line inside the innermost design surface.

torux.analysis.analyse_design takes as its line D (the line every answer's torus must enclose, torux.analysis.
torus_topology, and the island-band guard's, torux.islandband 'Axis identity') the first elliptic line it finds: the
Newton solve from the innermost design surface's m = 0 row (torux.axis.solve_axis), else torux.analysis.find_axes'
lines (from circles at the coils' median radius).  On 246 of 371,694 QUASR designs that line lies far outside the
innermost design surface, and the design can never be answered (surfaces fail the topology check around D, bands
the guard).  Three causes:
  - the design's own line is inverse hyperbolic (residue > 1, rotation 1/2 per period; QUASR's on-axis iota then
    nfp / 2 exactly: 243 of the 246), the solve converged to it inside the innermost surface, but find_axes also found
    an elliptic line elsewhere (near the coils or the device's centre) and that one became D -- the inverse branch
    applies only when no elliptic line exists at all;
  - the solve stalled on the true axis at the engine's rounding floor (|F| 2-5e-14 after quadratic convergence, above
    solve_axis' 1e-14 R0; QUASR 0830578, per-coil engine): reported unconverged, find_axes took a far line;
  - the solve converged to another elliptic line outside (0252504: R(0) 1.305 m, the axis at 0.975 m).

repair (switch AXSEL['on']; TORUX_AXSEL=0 or TORUX_AXSEL='{"on": false}' turns it off): only when D lies FAR
outside the innermost design surface -- its point on Z = 0 at phi = 0 or pi / nfp farther outside the surface's
crossings of Z = 0 than the surface's width there (AXSEL['far'] widths; nearer misses are left alone: on high-iota
designs QUASR's innermost surface misses the coils' true axis by up to 1.5 widths, the line's iota QUASR's to 1e-7) --
or when there is no elliptic line: the candidates are the design solve (a stalled one accepted when its last residual is
at most stall_tol R0 and the line is resolved, tail <= tail_tol, and passes torux.nearaxis.offgrid_residual <= offgrid_tol
between the collocation points), find_axes' lines, and the solve from the line traced
through the innermost surface's centre (torux.axis.traced_axis_guess); the candidate nearest inside the surface (measure
<= far) becomes D, an elliptic one before an inverse-hyperbolic one.  Otherwise nothing changes.  Every candidate
is a solved period-one line of the field:
nothing here accepts an answer.

continuation_axis_ok: analyse()'s continuation from the design path (the continuation probe, the fallback) chooses its
own axis -- the elliptic line nearest D's R0, which for an inverse-hyperbolic D is another line; with AXSEL on its
answer counts only if that axis is D (within same_rel R0, or same_width x the innermost surface's width, on 32 planes),
else the record is 'undetermined' (continuation_status 'other axis').
"""
from __future__ import annotations

import json as _json
import math
import os as _os

import numpy as np

AXSEL = dict(
    on=True,            # TORUX_AXSEL=0 turns it off
    far=1.0,            # D far outside: farther outside the innermost surface's Z = 0 crossings than its width there
    stall_tol=1e-12,    # a stalled design solve's last |F| (x R0) at most this ...
    tail_tol=1e-10,     # ... resolved (relative size of its last three harmonics) ...
    offgrid_tol=1e-10,  # ... and its off-grid residual (torux.nearaxis.offgrid_residual, relative) at most this
    traced=True,        # also the solve from the line traced through the innermost surface's centre
    same_rel=1e-4,      # two lines are the same within this x R0 on 32 planes (torux.analysis's inverse branch's) ...
    same_width=0.1,     # ... or (continuation_axis_ok) this x the innermost design surface's narrowest width on Z = 0
    check_continuation=True,   # analyse()'s continuation answers only around D
)


def opts():
    o = dict(AXSEL)
    env = (_os.environ.get("TORUX_AXSEL") or "").strip()
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


def surface_intervals(surf0):
    """The innermost design surface's crossings of Z = 0 in the planes phi = 0 and pi / nfp: [(R_lo, R_hi)] x 2 (a
    stellarator-symmetric surface's points at theta = 0, 1/2 of those planes)."""
    nfp = int(surf0.nfp)
    out = []
    for ph in (0.0, 0.5 / nfp):
        g = np.asarray(surf0.gamma(np.array([ph, ph]), np.array([0.0, 0.5])), float)
        R = np.hypot(g[:, 0], g[:, 1])
        out.append((float(np.min(R)), float(np.max(R))))
    return out


def outside_measure(line, intervals, nfp):
    """max over the two planes of (distance of the line's point on Z = 0 outside the interval) / (interval width); 0
    inside both.  line: torux.axis.Axis or a record dict (rc, zs)."""
    rc = np.asarray(line.rc if hasattr(line, "rc") else line["rc"], float)
    n = np.arange(len(rc))
    m = 0.0
    for k, (lo, hi) in enumerate(intervals):
        R = float(rc @ np.cos(n * k * math.pi))
        d = 0.0 if lo < R < hi else min(abs(R - lo), abs(R - hi))
        m = max(m, d / max(hi - lo, 1e-300))
    return m


def accept_stalled(engine, ax, o=None):
    """(ax, record): a solve_axis result that did not meet its tolerance, accepted (converged = True) when it stalled at
    the rounding floor: last residual <= stall_tol R0, resolved (tail <= tail_tol), off-grid residual <= offgrid_tol."""
    from .nearaxis import offgrid_residual, tail
    o = o or opts()
    hist = list(ax.info.get("hist") or [])
    R0 = max(abs(float(ax.rc[0])), 1e-300)
    rec = dict(last=float(hist[-1]) if hist else None, min=float(min(hist)) if hist else None, its=len(hist))
    if not hist or not np.isfinite(hist[-1]) or hist[-1] > o["stall_tol"] * max(1.0, R0):
        rec["accepted"] = False
        return ax, rec
    rec["tail"] = tail(ax)
    if rec["tail"] > o["tail_tol"]:
        rec["accepted"] = False
        return ax, rec
    r, err = offgrid_residual(engine, ax)
    rec["offgrid"], rec["err_half"] = r, err
    rec["accepted"] = bool(np.isfinite(r) and r <= o["offgrid_tol"])
    if rec["accepted"]:
        ax.converged = True
        ax.info["stall_accepted"] = dict(rec)
    return ax, rec


def same_line(a, b, rel=1e-4, atol=0.0):
    """Two period-one lines (Axis or record dicts with rc, zs, nfp) are one: max |dR|, |dZ| <= max(rel R0, atol) on 32
    planes."""
    def at(x, nfp):
        rc = np.asarray(x.rc if hasattr(x, "rc") else x["rc"], float)
        zs = np.asarray(x.zs if hasattr(x, "zs") else x["zs"], float)
        ph = np.linspace(0, 2 * np.pi / nfp, 33)[:-1]
        n = np.arange(len(rc))
        A = np.outer(ph * nfp, n)
        return np.cos(A) @ rc, np.sin(A) @ zs
    nfp = int(a.nfp if hasattr(a, "nfp") else b.nfp)
    Ra, Za = at(a, nfp)
    Rb, Zb = at(b, nfp)
    R0 = abs(float(np.asarray(a.rc if hasattr(a, "rc") else a["rc"])[0]))
    return bool(max(np.max(np.abs(Ra - Rb)), np.max(np.abs(Za - Zb))) <= max(rel * R0, atol))


def _rec(a, m, how):
    return dict(R0=float(np.sum(a.rc)), residue=float(a.residue), iota=float(a.iota), outside=float(m), how=how,
                N=int(a.N))


def repair(engine, source, surf0, ax_design, ell, axes, out, find_axes, called_find_axes, o=None):
    """(ell, axes) with D = ell[0] the line inside the innermost design surface (module docstring), or unchanged.
    out (the analysis record) gets 'axis_select' when the repair ran; out['axis_kind'] follows the line taken."""
    from . import axis as AX
    from .envfail import environment_failure
    o = o or opts()
    nfp = int(source.nfp)
    try:
        iv = surface_intervals(surf0)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        return ell, axes
    D = ell[0] if ell else None
    mD = outside_measure(D, iv, nfp) if D is not None else math.inf
    if D is not None and mD <= o["far"]:
        return ell, axes                       # the standard line: nothing recorded, nothing changed
    rec = dict(D=None if D is None else _rec(D, mD, "standard"), intervals=iv, candidates=[])
    cands = []
    a0 = ax_design
    if a0 is not None and not a0.converged:
        a0, st = accept_stalled(engine, a0, o)
        rec["stall"] = st
    if a0 is not None and a0.converged:
        if not np.isfinite(a0.residue):
            try:
                AX.spectral_axis_monodromy(engine, a0)
            except Exception as e:  # noqa: BLE001
                if environment_failure(e):
                    raise
        cands.append((a0, "design solve" + (" (stall accepted)" if "stall_accepted" in a0.info else "")))
    for a in axes:
        cands.append((a, "standard"))
    if not called_find_axes:
        try:
            for a in find_axes(engine, source):
                cands.append((a, "find_axes"))
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            rec["find_axes_error"] = repr(e)
    if o.get("traced"):
        lo, hi = iv[0]
        try:
            g_ = AX.traced_axis_guess(engine, nfp, 0.5 * (lo + hi))
            if g_ is not None:
                a_t = AX.solve_axis(engine, nfp, 0.5 * (lo + hi), N=16, guess=g_)
                if not a_t.converged:
                    a_t, _ = accept_stalled(engine, a_t, o)
                if a_t.converged:
                    AX.spectral_axis_monodromy(engine, a_t)
                    cands.append((a_t, "traced from the innermost surface's centre"))
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            rec["traced_error"] = repr(e)
    uniq = []
    for a, how in cands:
        if not (a.converged and np.isfinite(a.residue)):
            continue
        if any(same_line(a, b, o["same_rel"]) for b, _ in uniq):
            continue
        uniq.append((a, how))
    scored = [(outside_measure(a, iv, nfp), a, how) for a, how in uniq]
    rec["candidates"] = [_rec(a, m, how) for m, a, how in scored]
    inside = [(m, a, how) for m, a, how in scored if m <= o["far"]]
    ell_in = sorted([t for t in inside if 0 < t[1].residue < 1], key=lambda t: t[0])
    inv_in = sorted([t for t in inside if t[1].residue >= 1], key=lambda t: t[0])
    pick = ell_in[0] if ell_in else (inv_in[0] if inv_in else None)
    if pick is None:
        rec["taken"] = None
        out["axis_select"] = rec
        return ell, axes
    m, a, how = pick
    rec["taken"] = _rec(a, m, how)
    others = [b for _, b, _ in scored if b is not a]
    if 0 < a.residue < 1:
        ell = [a] + [b for b in others if 0 < b.residue < 1]
        out.pop("axis_kind", None)
    else:
        ell = [a]
        out["axis_kind"] = "inverse_hyperbolic"
    axes = [a] + others
    out["axis_select"] = rec
    return ell, axes


def continuation_axis_ok(r, D, surf0=None, o=None):
    """analyse()'s record r (from the design path, axis_guesses = [D]): is the axis its family started from D?  Its
    choice is reproduced (the elliptic line of r['axes'] nearest D's R0).  The same line within same_rel R0 or, given
    the innermost design surface, within same_width x its narrowest width on Z = 0: D is the design path's 16-mode
    line, analyse()'s is resolved further (up to N = 128) -- on a strongly shaped nfp = 8 axis (QUASR 1991526, last
    modes 2e-4) the two differ by 4.6e-4 m = 6e-4 R0, 0.005 of the surface's width; another period-one line lies widths
    away."""
    o = o or opts()
    ell = [a for a in (r.get("axes") or []) if a.get("residue") is not None and 0 < a["residue"] < 1]
    if not ell:
        return True                       # no family (no_axis): nothing to answer with
    A = min(ell, key=lambda a: abs(float(a["rc"][0]) - float(D.rc[0])))
    atol = 0.0
    if surf0 is not None:
        try:
            atol = o["same_width"] * min(hi - lo for lo, hi in surface_intervals(surf0))
        except Exception:  # noqa: BLE001
            atol = 0.0
    return same_line(D, A, o["same_rel"], atol)
