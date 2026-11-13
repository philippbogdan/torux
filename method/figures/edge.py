"""Figure 5 of the method page: the edge barrier.

    python method/figures/edge.py              # draw from data/edge.npz and edge.json
    python method/figures/edge.py --compute    # solve the request again, keep what the certificate computed, then draw

The request is the one examples/no_surface.py makes for QUASR 1100073:

    torux.solve("examples/quasr_1100073.json", budget_s=3300)

The answer is no_surface with an edge-barrier certificate. torux.analysis.edge_bound has two routes: the lost-curve
bound of torux.edgebound, which is off unless TORUX_EDGEBOUND=1, and the edge barrier of torux.barrier. This one is
the second, the manifold certificate with box crossings (curve class B). Its record keeps the closed curve, the X
points, the crossings and the lost point, but not the manifold branches, so the solve is watched:
torux.barrier.grow_step, torux.barrier.barrier_from_orbit and torux.barrier.edge_barrier are wrapped, and the
generating curves the certificate grew, its section, its lost-point candidates and the torus it started from are
kept, with every torus the analysis recorded (torux.analysis.torus_record).

Everything drawn lies in the half plane phi = 0 at true scale, R and Z in metres:

  * The bounding curve: the certificate's own polygon (its record), split into its arcs of unstable branches and of
    stable branches at the X points and the certified crossings, which are vertices of the polygon.
  * The X points: the orbit's points in its record.
  * Flux surfaces: three tori the solve found, and the outermost surface torux verifies. For that one the solve's
    tori are asked for directly (torux.solve at their flux), from the one the certificate started from (its F_last)
    inward, until one comes back verified. All as their series at phi = 0.
  * The branches that leave, in a small view of the whole plane: the generating curve u- of the certificate (its
    fundamental domain, the same object) grown again by torux.barrier.grow_step without the certificate's cut, to
    more field periods, and its mirror image, which is the stable branch (stellarator symmetry, as the certificate
    itself uses). The main view leaves them out, so that the curve and the surface asked for read clearly.
  * The lost point of the certificate's record, and the line that reaches it, traced again from its start.
  * The design's outermost stored surface, the one the request is the flux of, cut at phi = 0 by
    torux.design.planar_section.

The checks are in edge.json: the loop integral of A round the drawn polygon with the certificate's tube and
quadrature terms reproduces its bound, the reference field gives the same flux, the regrown branches agree with the
certificate's own curves, the lost point is reproduced, every drawn surface lies inside the curve and the design
surface outside it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data", "edge.npz")
NUMBERS = os.path.join(HERE, "edge.json")
SOURCE = os.path.join(REPO, "examples", "quasr_1100073.json")
BUDGET_S = 3300

# the regrowth of the branches that leave (drawing only: the certificate's own growth stopped once it had its curve)
L_FAR = 145             # field periods of growth shown in the inset (the pieces then hold 25,000 points)
GROW_DMAX = 4e-3        # spacing of the regrown curves, times the section's scale
GROW_AMAX = 0.15        # turning angle between neighbouring segments, rad
GROW_MAX_PTS = 30000
TOL_FAR = 1e-4          # m: Douglas-Peucker tolerance of the stored branches (the inset)
SURFACES = (0.0083709, 0.0154866, 0.0237475)   # Wb: tori of the solve drawn inside the outermost (picked by flux, to 1e-6)
VERIFY_BUDGET_S = 600   # each direct request for the outermost surface
VERIFY_TRIES = 3


# ============================================================================================ the run

def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run_request():
    """torux.solve on the request with the edge barrier watched. Returns the answer and what was kept."""
    import torux
    import torux.analysis as TA
    import torux.barrier as TB

    seen = dict(grow={}, bfo=[], edge=[], tori=[])
    orig = dict(grow_step=TB.grow_step, barrier_from_orbit=TB.barrier_from_orbit, edge_barrier=TB.edge_barrier,
                torus_record=TA.torus_record)

    def grow_step(sec, curves, dmax, amax, **kw):
        g = seen["grow"].setdefault(id(curves), dict(sec=sec, curves=curves, dmax=dmax, amax=amax, kw=dict(kw), calls=0))
        g["calls"] += 1
        return orig["grow_step"](sec, curves, dmax, amax, **kw)

    def barrier_from_orbit(sec, orb, a, *args, **kw):
        r = orig["barrier_from_orbit"](sec, orb, a, *args, **kw)
        scr = kw.get("screened") or {}
        seen["bfo"].append(dict(sec=sec, x0=np.array(orb.pts[0]), q=int(orb.q), ok=bool(r.get("ok")),
                                lost_c={k: dict(v) for k, v in (scr.get("lost_c") or {}).items()},
                                loc={k: {kk: vv for kk, vv in v.items()} for k, v in (scr.get("loc") or {}).items()}))
        return r

    def edge_barrier(engine, source, T_last, **kw):
        snap = _snapshot(T_last)
        r = orig["edge_barrier"](engine, source, T_last, **kw)
        seen["edge"].append(dict(engine=engine, source=source, T_last=snap, result=r,
                                 kw={k: v for k, v in kw.items() if k != "deadline"}))
        return r

    def torus_record(t, full=False):
        if float(getattr(t, "phi0", 0.0)) == 0.0 and getattr(t, "rc", None) is not None:
            seen["tori"].append(_snapshot(t))
        return orig["torus_record"](t, full)

    TB.grow_step, TB.barrier_from_orbit, TB.edge_barrier, TA.torus_record = \
        grow_step, barrier_from_orbit, edge_barrier, torus_record
    t0 = time.time()
    try:
        res = torux.solve(SOURCE, budget_s=BUDGET_S)
    finally:
        TB.grow_step, TB.barrier_from_orbit, TB.edge_barrier, TA.torus_record = \
            orig["grow_step"], orig["barrier_from_orbit"], orig["edge_barrier"], orig["torus_record"]
    return res, seen, time.time() - t0


def _snapshot(t):
    """What the figure needs of a torus, copied when it is seen (phi0 = 0: its symmetry plane is the section)."""
    from types import SimpleNamespace
    if float(getattr(t, "phi0", 0.0)) != 0.0:
        raise ValueError("a torus whose symmetry plane is not phi = 0")
    return SimpleNamespace(flux=float(t.flux), iota=float(t.iota), M=int(t.M), N=int(t.N),
                           bn_off=float(getattr(t, "bn_off", np.nan)), converged=bool(getattr(t, "converged", True)),
                           rc=np.array(t.rc, float), zs=np.array(t.zs, float))


def _series(t):
    """The section of torus t at phi = 0 as two series in theta: R_m (cosines) and Z_m (sines)."""
    return np.asarray(t.rc, float).sum(axis=1), np.asarray(t.zs, float).sum(axis=1)


def section_xy(Rm, Zm, theta):
    m = np.arange(len(Rm))
    a = np.outer(np.asarray(theta, float), m)
    return np.cos(a) @ np.asarray(Rm), np.sin(a) @ np.asarray(Zm)


def section_dtheta(Rm, Zm, theta):
    m = np.arange(len(Rm))
    a = np.outer(np.asarray(theta, float), m)
    return -np.sin(a) @ (m * np.asarray(Rm)), np.cos(a) @ (m * np.asarray(Zm))


def _dense(P, h=2e-5):
    """The closed polygon P with points every h along its edges (for distances to it)."""
    C = np.concatenate([P, P[:1]])
    seg = np.diff(C, axis=0)
    L = np.hypot(*seg.T)
    return np.concatenate([C[i] + np.linspace(0, 1, max(2, int(math.ceil(L[i] / h))), endpoint=False)[:, None] * seg[i]
                           for i in range(len(seg))])


def _inside(P, pts):
    from matplotlib.path import Path
    return Path(np.concatenate([P, P[:1]])).contains_points(np.asarray(pts))


def _polyline_distance(P, Q):
    """Distance from every point of P to the polyline Q."""
    a, b = Q[:-1], Q[1:]
    ab = b - a
    L2 = np.sum(ab * ab, 1)
    out = np.empty(len(P))
    for k in range(0, len(P), 128):
        p = P[k:k + 128]
        t = np.clip(np.sum((p[:, None, :] - a[None]) * ab[None], -1) / np.where(L2 > 0, L2, 1.0)[None], 0.0, 1.0)
        out[k:k + 128] = np.min(np.linalg.norm(a[None] + t[..., None] * ab[None] - p[:, None, :], axis=-1), axis=1)
    return out


def _simplify(p, tol):
    """Ramer, Douglas and Peucker on a polyline in metres: the ends kept, nothing moved more than tol."""
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b, s = p[i], p[j], p[i + 1:j]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 == 0:
            d = np.hypot(*(s - a).T)
        else:
            t = np.clip(((s - a) @ ab) / L2, 0.0, 1.0)
            d = np.hypot(*(s - a - t[:, None] * ab).T)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return p[keep]


def branch_polyline(X, pieces, k, q, Lmax, base=0, d=1):
    """The branch at X_k of a generating curve as torux.barrier.branch_polyline joins it: X_k, then the pieces of
    levels first, first + q, ... (first = d (k - base) mod q), the later pieces without their first vertex (the
    same curve point as the previous piece's last)."""
    first = (d * (k - base)) % q
    pts = [np.asarray(X[k], float)[None]]
    for L in range(first, Lmax + 1, q):
        P = pieces.get(L)
        if P is None:
            break
        pts.append(P if len(pts) == 1 else P[1:])
    return np.concatenate(pts)


def compute():
    import torux
    import torux.analysis as TA
    import torux.barrier as TB
    from torux.design import planar_section, section_flux
    from torux.reference import fast64

    t_start = time.time()
    res, seen, wall = run_request()
    ev = res.evidence or {}
    cert = ev.get("certificate") or {}
    rec = res.record or {}
    if res.status != "no_surface" or cert.get("kind") != "edge_barrier" or not cert.get("ok"):
        raise RuntimeError(f"expected no_surface with an edge-barrier certificate, got {res.status} "
                           f"({cert.get('kind')}, ok {cert.get('ok')})")
    q = int(cert["orbit"]["q"])
    X = np.array(cert["orbit"]["points"], float)
    poly = np.array(cert["polygon"], float)
    H = np.array([c["h"] for c in cert["crossings"]], float)
    a = np.array(cert["axis_point"], float)

    # ---- the certificate's section and generating curves (the grow_step calls of the certified orbit)
    match = [g for g in seen["grow"].values() if int(g["curves"][0].q) == q and np.array_equal(g["curves"][0].x0, X[0])]
    if len(match) != 1:
        raise RuntimeError(f"{len(match)} grown curve sets match the certified orbit")
    g = match[0]
    sec = g["sec"]
    sec.hard_deadline = None
    TB._HARD[0] = None
    if float(sec.phi0) != float(cert.get("plane_phi", 0.0)) or float(sec.phi0) != 0.0:
        raise RuntimeError("the certificate is not in the plane phi = 0")
    curves = {c.name: c for c in g["curves"]}
    bfo = [b for b in seen["bfo"] if b["ok"] and b["q"] == q and np.array_equal(b["x0"], X[0])]
    eb = [e for e in seen["edge"] if e["result"].get("ok") and abs(e["result"].get("F_up", 0) - cert["F_up"]) == 0.0]
    if len(bfo) != 1 or len(eb) != 1:
        raise RuntimeError("the certified orbit's barrier or edge_barrier call was not seen once")
    bfo, eb = bfo[0], eb[0]
    T_last = eb["T_last"]
    src = eb["source"]

    # ---- the bounding curve: its arcs, its flux with the certificate's terms, the reference field's flux
    iX = [int(np.flatnonzero((poly[:, 0] == p[0]) & (poly[:, 1] == p[1]))[0]) for p in X]
    iH = [int(np.flatnonzero((poly[:, 0] == p[0]) & (poly[:, 1] == p[1]))[0]) for p in H]
    F, qerr = sec.flux(poly)
    Fref, _ = sec.flux(poly, field=sec.ref_field())
    eb_ = cert["error_budget"]
    bound = F + qerr + eb_["tube"]
    perim = float(np.sum(np.hypot(*np.diff(np.concatenate([poly, poly[:1]]), axis=0).T)))
    dense = _dense(poly)
    from scipy.spatial import cKDTree
    tree = cKDTree(dense)

    # ---- flux surfaces: the tori the analysis recorded, the drawn ones checked against the reference field
    Fld = fast64.Field(src)
    th = 2 * np.pi * np.arange(2048) / 2048

    def torus_entry(t, role):
        Rm, Zm = _series(t)
        R, Z = section_xy(Rm, Zm, th)
        Rt, Zt = section_dtheta(Rm, Zm, th)
        A = Fld.evaluate(np.stack([R, np.zeros_like(R), Z], -1), grad=False)["A"]
        flux_ref = float(-2 * np.pi * np.mean(A[:, 0] * Rt + A[:, 2] * Zt))
        dd, _ = tree.query(np.stack([R, Z], -1))
        return dict(role=role, flux=float(t.flux), iota=float(t.iota), M=int(t.M), N=int(t.N),
                    bn_off=float(t.bn_off), converged=bool(t.converged), flux_reference=flux_ref,
                    flux_rel_diff=abs(abs(flux_ref) - abs(float(t.flux))) / abs(float(t.flux)),
                    inside_curve=bool(np.all(_inside(poly, np.stack([R, Z], -1)))),
                    min_distance_to_curve_m=float(dd.min()), R_m=Rm.tolist(), Z_m=Zm.tolist())

    surfaces = []
    for f in SURFACES:
        cand = [t for t in seen["tori"] if t.converged and abs(abs(float(t.flux)) - f) < 1e-6]
        if not cand:
            raise RuntimeError(f"no recorded torus at {f} Wb")
        surfaces.append(torus_entry(cand[0], "a torus of the solve"))
    last = torus_entry(T_last, "the torus the certificate started from (F_last)")
    recorded = sorted({round(abs(float(t.flux)), 9): (float(t.iota), float(t.bn_off), bool(t.converged))
                       for t in seen["tori"]}.items())

    # ---- the outermost surface torux holds: the solve's tori from F_last inward, each asked for directly, until one is
    # verified (the solver's check and the independent evaluator agree)
    from types import SimpleNamespace
    by_flux = {}
    for t in seen["tori"]:
        if t.converged:
            by_flux.setdefault(round(abs(float(t.flux)), 12), t)
    order = sorted((t for t in by_flux.values() if abs(float(t.flux)) <= abs(T_last.flux) * (1 + 1e-12)),
                   key=lambda t: -abs(float(t.flux)))
    attempts, verified = [], None
    for t in order[:VERIFY_TRIES]:
        f_ = abs(float(t.flux))
        t_v = time.time()
        r_ = torux.solve(SOURCE, f_, budget_s=VERIFY_BUDGET_S)
        attempts.append(dict(flux_Wb=f_, status=r_.status, wall_s=time.time() - t_v, text=str(r_),
                             call=f"torux.solve('examples/quasr_1100073.json', {f_!r}, budget_s={VERIFY_BUDGET_S})"))
        if r_.status == "surface":
            verified = r_
            break
    if verified is None:
        raise RuntimeError(f"none of the solve's outermost {VERIFY_TRIES} tori was verified when asked for")
    sv = verified.surface
    if float(getattr(sv, "phi0", 0.0)) != 0.0:
        raise RuntimeError("the verified surface's symmetry plane is not phi = 0")
    ind = verified.evidence.get("independent") or {}
    outer = torus_entry(SimpleNamespace(flux=float(sv.flux), iota=float(verified.iota), M=int(sv.M), N=int(sv.N),
                                        bn_off=float((verified.evidence.get("solver") or {}).get("bn_off")),
                                        converged=True, rc=np.array(sv.rc), zs=np.array(sv.zs)),
                        "the outermost surface torux verifies: " + attempts[-1]["call"])
    outer.update(independent_bn_sup=ind.get("bn_sup", ind.get("bn_max")), independent_flux_rel_err=ind.get("flux_rel_err"))

    # ---- the design's outermost stored surface, cut at phi = 0
    surf = src._design_surfaces[-1]
    Rd, Zd, dRd, dZd, ang_err = planar_section(surf, 0.0, ntheta=1024)
    eng = eb["engine"]
    fd, area_d = section_flux(lambda x: np.asarray(eng.evaluate(np.asarray(x), grad=False, A=True)["A"]), 0.0,
                              Rd, Zd, dRd, dZd)
    Ad = Fld.evaluate(np.stack([Rd, np.zeros_like(Rd), Zd], -1), grad=False)["A"]
    fd_ref = float(-np.sign(np.mean(Rd * dZd - Zd * dRd)) * np.mean(Ad[:, 0] * dRd + Ad[:, 2] * dZd))
    dd_design, _ = tree.query(np.stack([Rd, Zd], -1))
    design = dict(section_flux_Wb=float(fd), section_flux_reference_Wb=fd_ref, request_Wb=float(src.request["flux_Wb"]),
                  angle_error_rad=float(ang_err), inside_curve_fraction=float(np.mean(_inside(poly, np.stack([Rd, Zd], -1)))),
                  distance_to_curve_m=[float(dd_design.min()), float(dd_design.max())])

    # ---- the lost point: the certified one, its line traced again from its start with the certificate's map
    lp = cert["lost_point"]
    cand = bfo["lost_c"].get(lp["curve"])
    z = np.array(cand["start"][0], float)
    line = [z.copy()]
    for n in range(int(cand["steps"])):
        z, lost, _ = sec.step(z[None], np.array([float(cand["d"])]))
        z = z[0]
        line.append(z.copy())
        if lost[0]:
            raise RuntimeError(f"the lost line stopped after {n + 1} periods")
    nxt, lost_next, _ = sec.step(z[None], np.array([float(cand["d"])]))
    line = np.array(line)
    lost_rep = dict(steps=int(cand["steps"]), curve=lp["curve"], difference_m=float(np.linalg.norm(z - np.array(lp["point"]))),
                    next_period_stopped=bool(lost_next[0]), start_m=line[0].tolist(),
                    what="the certificate's lost line, from its start on the fundamental domain of its branch, one period "
                         "map at a time (FieldSection.step, the certificate's map): its last point is the lost point, and "
                         "the next period stops (the line turns, B_phi changes sign)")

    # ---- the branches that leave: generating curve u- regrown from the certificate's own fundamental domain
    gen = curves["u-"]
    cu = copy.deepcopy(gen)
    cu.pieces = cu.pieces[:1]
    cu.active, cu.lost, cu.stop_reason = True, None, None
    t_g = time.time()
    while cu.level() < L_FAR and cu.active:
        TB.grow_step(sec, [cu], GROW_DMAX * sec.scale, GROW_AMAX, max_pts=GROW_MAX_PTS, cut=None)
    grow_s = time.time() - t_g
    pieces = {p.level: np.array(p.pts) for p in cu.pieces}
    # agreement with the certificate's own curve over the levels it grew
    agree = 0.0
    for p in gen.pieces[1:]:
        if p.level in pieces:
            agree = max(agree, float(np.max(_polyline_distance(np.asarray(p.pts), pieces[p.level]))))
    bound_err = float(max(np.max(p.err) for p in gen.pieces))
    turning = []
    for L, P in pieces.items():
        if len(P) > 2:
            u, v = P[1:-1] - P[:-2], P[2:] - P[1:-1]
            c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-300)
            turning.append((L, float(np.max(np.arccos(np.clip(c, -1, 1))))))
    resolved_to = max([L for L, t in turning if t <= GROW_AMAX + 1e-9 and all(t2 <= GROW_AMAX + 1e-9 for L2, t2 in turning if L2 <= L)], default=0)
    far = [_simplify(branch_polyline(X, pieces, k, q, L_FAR, gen.base, gen.d), TOL_FAR) for k in range(q)]

    # ---- the record
    def nan_join(lines):
        return np.concatenate([np.concatenate([l, [[np.nan, np.nan]]]) for l in lines]).astype(np.float32)

    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    arrays = dict(polygon=poly, iX=np.array(iX), iH=np.array(iH), X=X, H=H, axis=a, lost=np.array(lp["point"]),
                  lost_line=line,
                  design=np.stack([Rd, Zd], -1), far=nan_join(far),
                  outer_R=np.array(outer["R_m"]), outer_Z=np.array(outer["Z_m"]))
    for i, s in enumerate(surfaces):
        arrays[f"surface{i}_R"] = np.array(s["R_m"])
        arrays[f"surface{i}_Z"] = np.array(s["Z_m"])
    np.savez_compressed(DATA, **arrays)

    lpc = {k: lp.get(k) for k in ("point", "curve", "steps_from_X", "direction", "radius", "err", "arc_length", "beta",
                                  "end_ratio", "upper_bound_f", "lipschitz", "nonlinearity", "integration_err")}
    lpc["reference"] = lp.get("reference")
    numbers = dict(
        figure="Fig. 5 of method/README.md: the edge barrier of QUASR 1100073 at its own request, in the half plane phi = 0",
        request=dict(source="examples/quasr_1100073.json", call="torux.solve('examples/quasr_1100073.json', budget_s=3300)",
                     flux_Wb=float(src.request["flux_Wb"]), provenance=src.request.get("provenance"),
                     what="the requested flux, the full toroidal flux the file's outermost stored surface encloses"),
        torux_version=getattr(torux, "__version__", None),
        answer=dict(status=res.status, wall_s=wall, text=str(res),
                    note="wall time of this run, the printed time of the answer included"),
        route=dict(
            edge_bound_first_stage="torux.edgebound (the lost-curve bound): not run, TORUX_EDGEBOUND is "
                                   f"{os.environ.get('TORUX_EDGEBOUND', 'unset')} and torux.analysis.EDGEBOUND is {TA.EDGEBOUND}",
            edge_bound_second_stage="torux.barrier.edge_barrier: the certificate",
            edge_barrier_options=_plain(eb["kw"]),
            certificate=cert.get("certificate"), curve_class=cert.get("curve_class"),
            what="certificate 'manifold': hyperbolic closed line, manifold arcs closed by certified crossing boxes "
                 "(class B), Lemma B by a lost point and the boxes. The lost ring was not tried (try_ring=False), and "
                 "the reversible closing gave no closed family (closed_families empty)"),
        certificate=dict(
            F_up_Wb=cert["F_up"], margin_rel=cert.get("margin_rel"), holds_request=cert.get("holds_request"),
            F_last_Wb=cert["F_last"], iota_last=cert.get("iota_last"), plane_phi=cert.get("plane_phi"),
            note=cert.get("note"), axis_point_m=cert["axis_point"], axis_residue=cert.get("axis_residue"),
            orbit=dict(q=q, p=cert["orbit"].get("p"), residue=cert["orbit"]["residue"], lam=cert["orbit"]["lam"],
                       e_X_m=cert["orbit"]["e_X"], points_m=cert["orbit"]["points"],
                       closure_reference=(cert.get("evidence") or {}).get("closure_reference")),
            crossings=[dict(gap=c["gap"], h_m=c["h"], angle_rad=c["angle"], box_diam_m=c["box_diam"],
                            u_curve=c["u_curve"], s_curve=c["s_curve"]) for c in cert["crossings"]],
            error_budget=eb_, excluded=cert.get("excluded"), lemma_B=cert.get("lemma_B"),
            lost_point=lpc, flux_reference=(cert.get("evidence") or {}).get("flux_reference"),
            growth=cert.get("growth"), start_error=cert.get("start_error"), timings_s=cert.get("timings")),
        drawn=dict(
            bounding_curve=dict(
                what="the certificate's polygon C_c, as its record keeps it: 11 arcs of unstable branches (generating "
                     "curve u+, X_k to the crossing h_k, solid) and 11 of stable ones (its mirror Su+, h_k to X_k+1, "
                     "dashed), split at the X points and crossings, which are vertices of the polygon",
                vertices=int(len(poly)), perimeter_m=perim,
                flux_Wb=F, quadrature_error_Wb=qerr, tube_Wb=eb_["tube"], bound_Wb=bound,
                bound_minus_certificate_Wb=bound - cert["F_up"],
                flux_reference_field_Wb=Fref, flux_rel_diff_reference=abs(Fref - F) / abs(F),
                how="FieldSection.flux of the certificate's own section (loop integral of A, Gauss-Legendre 5 points a "
                    "side, the 3-point rule's difference its error), plus the tube term of the record, is the bound "
                    "F_up. flux_reference_field_Wb is the same integral on torux.reference.fast64"),
            x_points=dict(n=q, on_curve="every X point and every crossing h_k is a vertex of the polygon (exact match)",
                          polygon_index=iX, crossing_index=iH),
            surfaces=dict(drawn=[{k: v for k, v in s.items() if k not in ("R_m", "Z_m")} for s in surfaces + [outer]],
                          certificate_start={k: v for k, v in last.items() if k not in ("R_m", "Z_m")},
                          outermost_requests=attempts,
                          outermost_note="the torus the certificate started from (F_last) is the solver's own check "
                                         "only unless its request comes back verified: the outermost ring drawn is "
                                         "the first of these requests that did",
                          recorded=[dict(flux_Wb=f, iota=i, bn_off=b, converged=c) for f, (i, b, c) in recorded],
                          how="tori torux.analysis recorded during the solve (torus_record), their sections at phi = 0 "
                              "from their series, flux_reference minus the loop integral of A of torux.reference.fast64 "
                              "round the drawn section (2048-point trapezoid)",
                          note="for the three inner tori bn_off is the largest |B.n|/|B| of the solve's own "
                               "acceptance: tori of its continuation, not answers refined to 3e-15 and checked "
                               "independently. The outermost is such an answer"),
            design_surface=dict(design, what="the file's outermost stored surface (the request is its flux, by QUASR's "
                                             "definition: the loop integral of A round its own theta curve at phi_s = 0), "
                                             "cut at cylindrical phi = 0 by torux.design.planar_section, drawn "
                                             "dashed, on a channel of paper and over the curve's own, so that it "
                                             "reads as one closed curve all round"),
            branches=dict(
                what="generating curve u- of the certificate, regrown from its own fundamental domain by "
                     "torux.barrier.grow_step on the certificate's section without its cut, and its mirror image Su-, "
                     "the stable branch (stellarator symmetry): unstable solid, stable dashed, drawn in the small view "
                     "of the whole plane only",
                certificate_growth_levels=int(gen.level()), certificate_cut_m=float(g["kw"]["cut"][1]),
                certificate_dmax_m=float(g["dmax"]), certificate_amax_rad=float(g["amax"]),
                regrown_dmax_m=GROW_DMAX * float(sec.scale), regrown_amax_rad=GROW_AMAX, regrown_levels=int(cu.level()),
                regrown_points=int(sum(len(p) for p in pieces.values())), regrown_s=grow_s,
                resolved_to_level=int(resolved_to),
                max_distance_to_certificate_curve_m=agree, certificate_curve_error_bound_max_m=bound_err,
                levels_inset=L_FAR, tolerance_inset_m=TOL_FAR,
                note="the regrown pieces above resolved_to_level keep the regrown spacing but not the turning limit at "
                     "the narrowest lobe tips (the refinement's parameter floor), well below what the inset can show"),
            lost_point_reproduced=lost_rep,
            scale=dict(main_units_per_m=SCALE, inset_units_per_m=layout()["inset_scale"],
                       note="page units: the figure is 838 wide, shown at 838 px on a laptop and 324 px on a phone")),
        compute=dict(wall_s=time.time() - t_start, note="the solve, the regrowth and the checks"),
    )
    with open(NUMBERS, "w") as fh:
        json.dump(_plain(numbers), fh, indent=1)
        fh.write("\n")
    return numbers


# ============================================================================================ the drawing

SCALE = 1420.0          # page units per metre in the main view
H_FIG = 792
WIDTH = 838             # style.WIDTH
TOP, LEFT, RIGHT, BOTTOM = 42, 64, 18, 60   # page units round the main view (10 of paper beyond its lettering)
R_LEFT = 0.897          # m: the main view's left edge
INSET_R = (1.19, 1.415)    # m: the inset's box, in the main view's coordinates (clear of the frame's ticks)
INSET_Z = (-0.225, 0.015)
INSET_VIEW = ((0.50, 1.64), (-0.62, 0.62))   # m: what the inset shows (the ranges grow to fill its box)


def layout():
    """The main view's box and ranges, and the inset's box and scale, in page units."""
    hm = H_FIG - TOP - BOTTOM
    wm = WIDTH - LEFT - RIGHT
    mz = (-0.5 * hm / SCALE, 0.5 * hm / SCALE)
    mr = (R_LEFT, R_LEFT + wm / SCALE)
    x0, x1 = (LEFT + (r - mr[0]) * SCALE for r in INSET_R)
    y0, y1 = (TOP + (mz[1] - z) * SCALE for z in (INSET_Z[1], INSET_Z[0]))
    (a0, a1), (b0, b1) = INSET_VIEW
    s_in = min((x1 - x0) / (a1 - a0), (y1 - y0) / (b1 - b0))
    return dict(main=(LEFT, TOP, wm, hm), mr=mr, mz=mz, inset=(x0, y0, x1 - x0, y1 - y0), inset_scale=s_in)


def _load():
    with np.load(DATA) as z:
        d = {k: z[k] for k in z.files}
    with open(NUMBERS) as fh:
        nums = json.load(fh)
    return d, nums


def _split(a):
    """The lines of a NaN-separated array."""
    out, cur = [], []
    for p in a.astype(float):
        if np.isnan(p[0]):
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def _arcs(d):
    """The bounding curve's arcs: ('u', points) from h_k to X_k and ('s', points) from X_k+1 to h_k, in polygon
    order (the record's polygon runs X_10, stable, h_9, unstable, X_9 and so on)."""
    P, iX, iH = d["polygon"], list(d["iX"]), list(d["iH"])
    out = []
    for k in range(len(iX)):
        ix, ih = iX[k], iH[k]
        out.append(("u", P[ih:ix + 1]))
        lower = [i for i in iX if i < ih]
        prev = max(lower) if lower else max(iX)
        out.append(("s", P[prev:ih + 1] if prev < ih else np.concatenate([P[prev:], P[:ih + 1]])))
    return out


def draw(previews=None):
    sys.path.insert(0, HERE)
    import style as S

    d, nums = _load()
    q = int(nums["certificate"]["orbit"]["q"])
    X, H = d["X"], d["H"]
    poly = d["polygon"]
    if S.WIDTH != WIDTH:
        raise RuntimeError(f"style.WIDTH is {S.WIDTH}, this figure is laid out for {WIDTH}")
    lay = layout()
    fig = S.Figure(H_FIG)
    mr, mz = lay["mr"], lay["mz"]
    ax = fig.axes(lay["main"], mr, mz)

    def tone(view):
        x0, x1, y0, y1 = view.lim
        pad = 0.02 * (x1 - x0)
        view.fill([np.array([x0 - pad, x1 + pad, x1 + pad, x0 - pad]), poly[:, 0]],
                  [np.array([y0 - pad, y0 - pad, y1 + pad, y1 + pad]), poly[:, 1]], tone="stipple")

    def wall(view, weight="heavy", kinds=True, halo=True):
        arcs = _arcs(d)
        passes = (True, False) if halo else (False,)
        for h in passes:
            for kind, a in arcs:
                view.line(a[:, 0], a[:, 1], weight=weight, dash="solid" if (kind == "u" or not kinds) else "dashed", halo=h)

    # the main view: tone beyond the curve, the design's surface on its own channel of paper, the curve, the design's
    # surface again over the curve's channel (one closed curve all round, though it runs within a millimetre of the
    # curve on the inboard side), the surfaces. The branches that leave are the small view's
    tone(ax)
    Rd, Zd = d["design"][:, 0], d["design"][:, 1]
    ax.line(Rd, Zd, weight="medium", dash="dashed", closed=True, halo=True)
    wall(ax)
    ax.line(Rd, Zd, weight="medium", dash="dashed", closed=True)
    th = 2 * np.pi * np.arange(1440) / 1440
    for i in range(len(SURFACES)):
        ax.line(*section_xy(d[f"surface{i}_R"], d[f"surface{i}_Z"], th), weight="fine", closed=True)
    ax.line(*section_xy(d["outer_R"], d["outer_Z"], th), weight="fine", closed=True)
    ax.mark(X[:, 0], X[:, 1], kind="cross")
    ax.mark(*d["axis"], kind="dot")

    # the bound rounded up (it is an upper bound), the request to the same digits
    F_up = math.ceil(nums["certificate"]["F_up_Wb"] * 1e5) / 1e5
    req = round(nums["request"]["flux_Wb"], 5)
    lx = 1.225
    ax.leader(lx, 0.215, f"X line, period ${q}$", X[3, 0], X[3, 1])
    ax.leader(lx, 0.160, rf"$\Psi\leq {F_up:.5f}$ Wb", H[4, 0], H[4, 1])
    # the design's surface at the outboard midplane, where it runs clear of the curve (between X points)
    out_ = np.flatnonzero(d["design"][:, 0] > d["axis"][0])
    j = int(out_[np.argmin(np.abs(d["design"][out_, 1]))])
    ax.leader(lx, 0.105, rf"asked for, $\Psi_0={req:.5f}$ Wb", d["design"][j, 0], d["design"][j, 1])
    ax.text(lx + 0.005, 0.050, "no flux surface", anchor="w")
    ax.frame("box")
    ax.xticks(major=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4], minor=[0.95, 1.05, 1.15, 1.25, 1.35],
              labels=["$0.9$", "$1.0$", "$1.1$", "$1.2$", "$1.3$", "$1.4$"])
    ax.yticks(major=[-0.2, -0.1, 0.0, 0.1, 0.2], labels=["$-0.2$", "$-0.1$", "$0$", "$0.1$", "$0.2$"])
    ax.xlabel("$R$ (m)")
    ax.ylabel("$Z$ (m)")

    # the inset: the whole plane at a fifth of the scale, the branches grown further, and the lost point
    box = lay["inset"]
    L, T, W, Hb = box
    fig.fill([L, L + W, L + W, L], [T, T, T + Hb, T + Hb], tone="paper")
    ov = fig.view(box, *INSET_VIEW, equal=True, clip=True)
    for b in _split(d["far"]):
        ov.line(b[:, 0], b[:, 1], weight="fine", dash="solid")
        ov.line(b[:, 0], -b[:, 1], weight="fine", dash="dashed")
    wall(ov, weight="medium", kinds=False, halo=False)
    ov.line([mr[0], mr[1], mr[1], mr[0]], [mz[0], mz[0], mz[1], mz[1]], weight="fine", closed=True)
    ov.mark(*d["lost"], kind="dot")
    ov.leader(d["lost"][0] + 0.10, d["lost"][1] - 0.17, "lost point", d["lost"][0], d["lost"][1], size="small")
    fig.line([L, L + W, L + W, L], [T, T, T + Hb, T + Hb], weight="medium", closed=True)
    paths = fig.save(os.path.join(HERE, "edge"), previews=previews)
    sc = nums.get("drawn", {}).get("scale")
    if sc is not None and (sc.get("main_units_per_m") != SCALE or sc.get("inset_units_per_m") != ov.scale[0]):
        sc.update(main_units_per_m=SCALE, inset_units_per_m=ov.scale[0])
        with open(NUMBERS, "w") as fh:
            json.dump(nums, fh, indent=1)
            fh.write("\n")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compute", action="store_true", help="solve the request again and keep what the certificate computed")
    ap.add_argument("--no-draw", action="store_true", help="with --compute: only compute")
    ap.add_argument("--previews", metavar="DIR", help="also write PNG renders at 838 and 324 px into DIR")
    a = ap.parse_args(argv)
    if a.compute or not os.path.exists(DATA):
        compute()
    if not a.no_draw:
        draw(a.previews)


if __name__ == "__main__":
    sys.exit(main())
