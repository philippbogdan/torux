"""Far-start exit curves: the exit march (torux.exitcurve) started from a closed curve in O far beyond the coils, where
the lost curve of torux.edgebound's default box does not close, and optionally in the second stellarator-symmetry plane
phi = pi / nfp (by the rotated source).  The theory is torux.exitcurve.THEOREM_DOC's, see FAR_DOC."""
from __future__ import annotations

import math

import numpy as np

FAR_DOC = """FAR START.  (O4) needs only some closed curve C in O with winding number 1 about a; (O1) puts the points with
g = s R B_phi <= 0 and the lost points in O.  So C_0 may be any certified lost polygon -- its pieces by the lost
certificate (torux.edgebound._seg_base_test; a piece where g <= 0 already is lost at arc length ~ 0) -- in a box
reaching far beyond the coils, built in the upper half (symmetry row to symmetry row, Z >= 0) and closed by its
mirror image ((O3): S maps O onto O).  Its winding number about a and its simplicity are checked exactly on the
vertices; the box keeps R > 0 (the curve lies in the half plane Sigma).  The march from it is THEOREM_DOC's.
SECOND PLANE.  For phi = pi / nfp: the source rotated by -pi / nfp about the z axis (the coil coefficients rotated
exactly) is nfp-periodic and stellarator symmetric about ITS phi = 0 (S' = Rot S of the original; measured every
run).  Its covered tori are the rotations of the original's (an isometry maps sources, field lines, s B_phi and the
axis), and a covered torus's toroidal flux is the same through every plane phi = const (div B = 0 in V, B.n = 0 on
Gamma), so a bound proved for the rotated source in its phi = 0 plane bounds the original's covered tori.  The axis
point there is the original axis point carried half a period along its field line."""


def rotated_source(source, psi):
    """The source rotated by -psi about the z axis (the plane phi = psi of the original becomes phi = 0): Fourier
    filament coefficients rotated exactly (x' = x cos psi + y sin psi, y' = -x sin psi + y cos psi); parts without
    currents kept.  psi = 0 returns the source itself."""
    if psi == 0.0:
        return source
    import copy
    from .source import FourierCurves
    c, s = math.cos(psi), math.sin(psi)
    parts = []
    for p in source.parts:
        if p.kind == "fourier":
            co = np.array(p.coef, float)
            x, y = co[:, 0, :].copy(), co[:, 1, :].copy()
            co[:, 0, :] = c * x + s * y
            co[:, 1, :] = -s * x + c * y
            parts.append(FourierCurves(co, np.array(p.current, float)))
        elif p.kind == "toroidal" and float(getattr(p, "RB", 0.0)) == 0.0:
            parts.append(p)
        else:
            raise ValueError(f"exitcurve_far: cannot rotate part {p.kind}")
    out = copy.copy(source)
    out.parts = parts
    out.id = f"{source.id}@rot{psi:.12g}"
    return out


def axis_point_in_plane(source, axis_point, psi, det_s=None, d_fail=1e-3):
    """The axis point (R, Z) in the plane phi = psi: the line through (R_a, 0, Z_a) followed to its first crossing of
    phi = psi (psi = pi / nfp: half a period), with the loose tracer.  None if it does not get there."""
    from .edgebound import ExactTracer
    from .exitcurve import multicross
    n2 = 2.0 * math.pi / psi
    if abs(n2 - round(n2)) > 1e-9:
        raise ValueError("psi must be 2 pi / integer")
    tr = ExactTracer(source, int(round(n2)), 1.0, d_fail=d_fail, tol=1e-9, eps=1e-9, beta=0.01)
    X = np.array([[axis_point[0], 0.0, axis_point[1]]])
    s = float(np.sign(tr.B(X)[0, 1])) if det_s is None else float(det_s)
    tr.s = s
    o = multicross(tr, X, 1, 20.0 * psi * axis_point[0])
    if o["nc"][0] < 1:
        return None
    x = o["XC"][0, 0]
    return np.array([math.hypot(x[0], x[1]), x[2]])


def coil_cage_radius(source, axis_point):
    """The largest distance from the axis point, in the (R, Z) projection, of a point of any coil (every toroidal
    angle: the cage's outer legs need not cross phi = 0)."""
    pts = np.concatenate([p.points(np.arange(512) / 512)[0].reshape(-1, 3) for p in source.parts
                          if p.kind == "fourier"])
    return float(np.max(np.hypot(np.hypot(pts[:, 0], pts[:, 1]) - axis_point[0], pts[:, 2] - axis_point[1])))


def far_exit_bound(source, axis_point, T_last=None, core=None, scale_poly=None, request=None, budget_s=1800.0,
                   far_reach=None, lost_frac=1.0 / 3.0, band=12.0, h_rel=1 / 16, verbose=False, with_evidence=True,
                   verified_sections=None, F_verified=None, max_half_nodes=150, lost_window=16.0, W_exit=16, **kw):
    """The far start (FAR_DOC) in the phi = 0 plane of `source` (pass rotated_source(...) and the carried axis point for
    the second plane): C_0 = torux.edgebound.lost_curve_bound on a box reaching far_reach x d_last beyond the scale
    polygon (default: 2.5 d_last beyond the coil cage, at least the default 1.8), lost_frac of the budget; then the
    exit march (torux.exitcurve.exit_curve_bound, start_curve = C_0) on the same grid, band `band` node spacings per
    step.  h is enlarged when the box would exceed max_half_nodes nodes from the axis point to its edge (at most
    ~ 2 max_half_nodes^2 nodes; efficiency only).  Lines lost within lost_window periods make C_0 (the lost curve's
    detection and segment windows) and the march's lost nodes; the march's detection and exit window is W_exit
    crossings.  T_last, or verified_sections + F_verified: the
    negative test.  Returns exit_curve_bound's record with 'far' details."""
    import time
    from scipy.spatial import cKDTree
    from .barrier import signed_area
    from .edgebound import lost_curve_bound
    from .exitcurve import exit_curve_bound
    t0 = time.time()
    a = np.asarray(axis_point, float)
    if T_last is not None:
        th = 2 * np.pi * np.arange(1024) / 1024
        Rl, Zl = T_last.geometry_on(th, np.zeros_like(th))
        sec = np.stack([Rl, Zl], -1)
        core = sec if core is None else core
        scale_poly = sec if scale_poly is None else scale_poly
    sp_ = np.asarray(scale_poly if scale_poly is not None else core, float)
    r_max = float(np.max(np.hypot(sp_[:, 0] - a[0], sp_[:, 1] - a[1])))
    r_eff = math.sqrt(abs(signed_area(sp_)) / math.pi)
    cp = np.concatenate([p.points(np.arange(1024) / 1024)[0].reshape(-1, 3) for p in source.parts
                         if p.kind == "fourier"])
    d_last = float(np.min(cKDTree(cp).query(np.stack([sp_[:, 0], 0 * sp_[:, 0], sp_[:, 1]], -1))[0]))
    r_cage = coil_cage_radius(source, a)
    if far_reach is None:
        far_reach = max(1.8, (max(r_cage, r_max) + 2.5 * d_last - r_max) / d_last)
    ext = r_max + far_reach * d_last
    h_rel_eff = max(float(h_rel), ext / (max_half_nodes * r_eff))
    far = dict(far_reach=far_reach, r_cage=r_cage, r_max=r_max, d_last=d_last, h_rel=h_rel_eff, ext=ext,
               lost_window=lost_window, W_exit=int(W_exit))
    lc = lost_curve_bound(source, a, T_last=T_last, core=core, scale_poly=scale_poly, request=request,
                          budget_s=lost_frac * budget_s, h_rel=h_rel_eff, reach_d=far_reach, verbose=verbose,
                          with_evidence=with_evidence, det_win=lost_window, seg_win=lost_window)
    far["t_lost"] = time.time() - t0
    far["lost_curve_ok"] = bool(lc.get("ok"))
    if lc.get("ok") and request is not None and lc.get("holds_request"):
        lc["far"] = far
        return lc
    rec = exit_curve_bound(source, a, T_last=T_last, core=core, scale_poly=scale_poly, request=request,
                           budget_s=max(budget_s - (time.time() - t0), 60.0), start_curve=lc, lost_share=0.0,
                           reach_d=far_reach, h_rel=h_rel_eff, band=band, verbose=verbose, W_lost=lost_window,
                           W_exit=int(W_exit),
                           with_evidence=with_evidence, verified_sections=verified_sections, **kw)
    rec["far"] = far
    if F_verified is not None and rec.get("ok"):
        rec["F_verified"] = float(F_verified)
        if min(rec.get("F_up_steps") or [rec["F_up"]]) < float(F_verified):
            rec.update(ok=False, false_certificate=True, reason="FALSE CERTIFICATE: F_up below the verified flux")
    return rec


def carry_polygon(source, poly, psi, sign):
    """The polygon (n, 2) of the plane phi = 0 carried to the plane phi = psi along the field lines (loose tracer;
    points whose lines do not get there dropped).  Efficiency only (cores and scales)."""
    from .edgebound import ExactTracer
    from .exitcurve import multicross
    n2 = int(round(2.0 * math.pi / psi))
    tr = ExactTracer(source, n2, sign, d_fail=1e-3, tol=1e-8, eps=1e-8, beta=0.01)
    X = np.stack([poly[:, 0], 0 * poly[:, 0], poly[:, 1]], -1)
    o = multicross(tr, X, 1, 20.0 * psi * float(np.max(poly[:, 0])))
    ok = o["nc"] >= 1
    return np.stack([np.hypot(o["XC"][ok, 0, 0], o["XC"][ok, 0, 1]), o["XC"][ok, 0, 2]], -1)


def far_start(source, axis_point, T_last=None, core=None, scale_poly=None, request=None, plane="phi0",
              budget_s=1800.0, **kw):
    """far_exit_bound in the plane phi = 0 ('phi0') or phi = pi / nfp ('phi_half': the rotated source, the axis point
    carried half a period, T_last's own section there for the negative test and the scales, or the phi = 0 core /
    scale polygons carried along the lines)."""
    a = np.asarray(axis_point, float)
    if plane == "phi0":
        return far_exit_bound(source, a, T_last=T_last, core=core, scale_poly=scale_poly, request=request,
                              budget_s=budget_s, **kw)
    psi = math.pi / source.nfp
    src_p = rotated_source(source, psi)
    a_p = axis_point_in_plane(source, a, psi)
    if a_p is None:
        return dict(ok=False, kind="exit_curve", reason="the axis point did not reach the plane phi = pi / nfp")
    extra = {}
    if T_last is not None:
        th = 2 * np.pi * np.arange(1024) / 1024
        R, Z = T_last.geometry_on(th, np.full_like(th, psi))
        sec = np.stack([R, Z], -1)
        core_p, scale_p = sec, sec
        extra = dict(verified_sections=[sec], F_verified=abs(float(T_last.flux)))
    else:
        from .edgebound import ExactTracer
        sgn = float(np.sign(ExactTracer(source, source.nfp, 1.0, d_fail=1e-3).B(np.array([[a[0], 0, a[1]]]))[0, 1]))
        scale_p = carry_polygon(source, np.asarray(scale_poly if scale_poly is not None else core, float), psi, sgn)
        core_p = carry_polygon(source, np.asarray(core, float), psi, sgn) if core is not None else None
        if core_p is None or len(core_p) < 16:
            core_p = a_p[None, :] + 0.3 * (scale_p - a_p[None, :])
    rec = far_exit_bound(src_p, a_p, core=core_p, scale_poly=scale_p, request=request, budget_s=budget_s,
                         **extra, **kw)
    rec["plane"] = plane
    rec["axis_point_plane"] = a_p.tolist()
    return rec


FAR_CAP = 1800.0        # seconds an analysis gives the far start at most (TORUX_EXITCURVE_FAR_CAP)
