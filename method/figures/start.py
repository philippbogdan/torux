"""Figure 2 of the method page: no starting guess, from the magnetic axis outward.

    python method/figures/start.py              # draw from data/start.json
    python method/figures/start.py --compute    # run the request again, keep what the solver did, then draw

The request is the one examples/any_source.py makes for HSX:

    torux.solve("examples/hsx.json", flux=0.0359221439332229, budget_s=900)

The file carries no surface, so torux starts from the magnetic axis. The answer keeps only the flux and iota
of each torus of the continuation (its profile), so the run is watched: torux.family.Continuation.start and
Continuation._run are wrapped, and every torus the continuation accepted is kept, as its section at phi = 0.
At phi = 0 the series in m theta - n N_fp phi collapses to a series in theta alone,

    R(theta, 0) = sum_m (sum_n rc[m, n]) cos(m theta),    Z(theta, 0) = sum_m (sum_n zs[m, n]) sin(m theta),

so the section is kept exactly, as those two rows of numbers per torus. The start circle is the one
torux.family.axis_torus_start was called with. The drawn curves are then checked with torux.reference, which
shares no code with the solver: the flux each section encloses (minus the loop integral of A along it) and
field lines started on each section, traced for one turn, measured against the section they started on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data", "start.json")
NUMBERS = os.path.join(HERE, "start.json")
SOURCE = os.path.join(REPO, "examples", "hsx.json")
FLUX = 0.0359221439332229          # Wb, the request of examples/any_source.py
BUDGET_S = 900


# ---------------------------------------------------------------------------------------------- the run

def _section(t):
    """The section of torus t at phi = 0 as its two series in theta: R_m (cosines) and Z_m (sines)."""
    if float(getattr(t, "phi0", 0.0)) != 0.0:
        raise ValueError("a torus whose symmetry plane is not phi = 0")
    return np.asarray(t.rc, float).sum(axis=1), np.asarray(t.zs, float).sum(axis=1)


def _torus_entry(t):
    Rm, Zm = _section(t)
    return dict(flux=float(t.flux), s=math.sqrt(abs(float(t.flux))), iota=float(t.iota), M=int(t.M), N=int(t.N),
                bn_off=float(t.bn_off), its=int(t.its), R_m=Rm.tolist(), Z_m=Zm.tolist())


def run_request():
    """torux.solve on the request, with the continuation watched. Returns the record kept in data/start.json."""
    import torux
    from torux import family as F

    seen = dict(circles=[], starts=[], runs=[], start_solves=[])
    orig_ats, orig_start, orig_run, orig_solve = F.axis_torus_start, F.Continuation.start, F.Continuation._run, \
        F.Continuation.solve
    inside_start = [False]

    def axis_torus_start(axis, M, N, eps, iota):
        seen["circles"].append(dict(eps=float(eps), R0=float(axis.rc[0]), M=int(M), N=int(N), iota=float(iota)))
        return orig_ats(axis, M, N, eps, iota)

    def solve(self, guess, flux, M, N, *a, **k):
        sol = orig_solve(self, guess, flux, M, N, *a, **k)
        if inside_start[0]:
            seen["start_solves"].append(dict(M=int(M), N=int(N), flux=float(flux), its=int(sol.its),
                                             converged=bool(sol.converged), bn_off=float(sol.bn_off)))
        return sol

    def start(self, axis):
        inside_start[0] = True
        try:
            sol, f0 = orig_start(self, axis)
        finally:
            inside_start[0] = False
        seen["starts"].append(dict(axis=axis, sol=sol, f0=float(f0), ladder=sol.info.get("start_ladder")))
        return sol, f0

    def _run(self, axis, flux_target, ds0, ds_max, ds_min, deadline, start):
        out = orig_run(self, axis, flux_target, ds0, ds_max, ds_min, deadline, start)
        seen["runs"].append(dict(tori=list(out.tori), log=list(out.log), stop=out.stop_reason,
                                 from_jump=start is not None, ds0=ds0, ds_max=ds_max, ds_min=ds_min))
        return out

    F.axis_torus_start, F.Continuation.start, F.Continuation._run, F.Continuation.solve = \
        axis_torus_start, start, _run, solve
    t0 = time.time()
    try:
        res = torux.solve(SOURCE, flux=FLUX, budget_s=BUDGET_S)
    finally:
        F.axis_torus_start, F.Continuation.start, F.Continuation._run, F.Continuation.solve = \
            orig_ats, orig_start, orig_run, orig_solve
    wall = time.time() - t0
    rec = res.record or {}
    if len(seen["starts"]) != 1 or not seen["runs"]:
        raise RuntimeError(f"expected one start and a continuation, saw {len(seen['starts'])} and {len(seen['runs'])}")
    st = seen["starts"][0]
    ax = st["axis"]
    Rax, Zax = ax.at(np.array([0.0]))
    # the family as the analysis assembled it: the first run from the start, each later run from a jump
    family, kind = [], []
    for j, r in enumerate(seen["runs"]):
        for i, t in enumerate(r["tori"]):
            family.append(t)
            kind.append("start" if (j == 0 and i == 0) else ("jump" if (r["from_jump"] and i == 0) else "step"))
    surface = res.surface
    ind = (res.evidence or {}).get("independent") or {}
    sol = (res.evidence or {}).get("solver") or {}
    # what is kept is what the solver had: each section against the torus's own geometry at phi = 0, the family
    # against the profile the answer reports, and the family's torus at the request against the refined one
    th = 2 * np.pi * np.arange(1024) / 1024
    geo = 0.0
    for t in family + [surface]:
        Rg, Zg = t.geometry_grid(th, np.array([0.0]))
        Rs, Zs = section_xy(*_section(t), th)
        geo = max(geo, float(np.max(np.hypot(Rg[:, 0] - Rs, Zg[:, 0] - Zs))))
    prof = rec.get("profile") or []
    prof_diff = max((max(abs(p["flux"] - float(t.flux)), abs(p["iota"] - float(t.iota))) for p, t in zip(prof, family)),
                    default=float("nan")) if len(prof) == len(family) else float("nan")
    Rl, Zl = section_xy(*_section(family[-1]), th)
    Rr, Zr = section_xy(*_section(surface), th)
    consistency = dict(section_vs_geometry_m=geo, profile_count=len(prof), family_count=len(family),
                       profile_max_diff=prof_diff,
                       last_family_vs_refined_m=float(np.max(np.hypot(Rl - Rr, Zl - Zr))))
    out = dict(
        request=dict(source="examples/hsx.json", flux_Wb=FLUX, budget_s=BUDGET_S,
                     call="torux.solve('examples/hsx.json', flux=0.0359221439332229, budget_s=900)"),
        torux_version=getattr(torux, "__version__", None),
        status=res.status, wall_s=wall,
        axis=dict(nfp=int(ax.nfp), R0=float(ax.rc[0]), rc=np.asarray(ax.rc).tolist(), zs=np.asarray(ax.zs).tolist(),
                  iota=float(ax.iota), residue=float(ax.residue), R_phi0=float(Rax[0]), Z_phi0=float(Zax[0])),
        start=dict(circles=seen["circles"], eps=seen["circles"][0]["eps"] if seen["circles"] else None,
                   flux=st["f0"], solves=seen["start_solves"], ladder=st["ladder"]),
        runs=[dict(n_tori=len(r["tori"]), stop=r["stop"], from_jump=r["from_jump"], ds0=r["ds0"], ds_max=r["ds_max"],
                   ds_min=r["ds_min"],
                   log=[{k: (float(v) if isinstance(v, (float, np.floating)) else
                             (bool(v) if isinstance(v, (bool, np.bool_)) else int(v))) for k, v in e.items()}
                        for e in r["log"]]) for r in seen["runs"]],
        family=[dict(_torus_entry(t), kind=k) for t, k in zip(family, kind)],
        chains=rec.get("chains") or [], jumps=rec.get("jumps") or [],
        profile=rec.get("profile"), family_stop=rec.get("family_stop"),
        surface=dict(_torus_entry(surface), bn_off=(rec.get("surface") or {}).get("bn_off"),
                     quad_err_half=sol.get("quad_err_half"),
                     refine_hist=[list(map(float, h)) for h in (rec.get("refine_hist") or [])],
                     refine_stop=rec.get("refine_stop"), guard_ok=(rec.get("guard") or {}).get("ok")),
        independent={k: v for k, v in ind.items() if isinstance(v, (int, float, str, bool)) or v is None},
        timings=rec.get("timings"),
        consistency=consistency,
    )
    return out


# ------------------------------------------------------------------------------------- the sections

def section_xy(Rm, Zm, theta):
    """R, Z of a section series at the angles theta."""
    m = np.arange(len(Rm))
    a = np.outer(np.asarray(theta, float), m)
    return np.cos(a) @ np.asarray(Rm), np.sin(a) @ np.asarray(Zm)


def section_dtheta(Rm, Zm, theta):
    m = np.arange(len(Rm))
    a = np.outer(np.asarray(theta, float), m)
    return -np.sin(a) @ (m * np.asarray(Rm)), np.cos(a) @ (m * np.asarray(Zm))


def distance_to_section(Rm, Zm, P, n=4096):
    """The distance of each point P (k, 2) from the section curve: the nearest of n samples, then Newton on theta."""
    th = 2 * np.pi * np.arange(n) / n
    R, Z = section_xy(Rm, Zm, th)
    out = []
    m = np.arange(len(Rm))
    for p in np.atleast_2d(P):
        t = th[int(np.argmin((R - p[0]) ** 2 + (Z - p[1]) ** 2))]
        for _ in range(30):
            c, s = np.cos(m * t), np.sin(m * t)
            x, z = c @ Rm - p[0], s @ Zm - p[1]
            dx, dz = -(s * m) @ Rm, (c * m) @ Zm
            ddx, ddz = -(c * m * m) @ Rm, -(s * m * m) @ Zm
            g = x * dx + z * dz
            h = dx * dx + dz * dz + x * ddx + z * ddz
            step = g / h if h > 0 else g / (dx * dx + dz * dz)
            t -= step
            if abs(step) < 1e-15:
                break
        c, s = np.cos(m * t), np.sin(m * t)
        out.append(math.hypot(c @ Rm - p[0], s @ Zm - p[1]))
    return np.array(out)


def check_with_reference(data, n_lines=2, n_turns=1):
    """The drawn curves measured by torux.reference (no code shared with the solver): the flux each section
    encloses, and field lines traced from each section for n_turns turns, every field period a return to phi = 0."""
    import torux
    from torux.reference import fast64, trace
    src = torux.load(SOURCE)
    Fld = fast64.Field(src)
    nfp = data["axis"]["nfp"]
    curves = [("family", i, e) for i, e in enumerate(data["family"])] + [("surface", 0, data["surface"])]
    th = 2 * np.pi * np.arange(512) / 512
    rows = []
    for what, i, e in curves:
        Rm, Zm = np.array(e["R_m"]), np.array(e["Z_m"])
        R, Z = section_xy(Rm, Zm, th)
        Rt, Zt = section_dtheta(Rm, Zm, th)
        A = Fld.evaluate(np.stack([R, np.zeros_like(R), Z], -1), grad=False)["A"]
        flux_ref = float(-2 * np.pi * np.mean(A[:, 0] * Rt + A[:, 2] * Zt))      # phi = 0: e_R = x, e_Z = z
        t0 = 2 * np.pi * (np.arange(n_lines) + 0.25) / n_lines
        R0, Z0 = section_xy(Rm, Zm, t0)
        tr = trace.poincare(Fld, R0, Z0, 2 * np.pi * np.arange(1, nfp + 1) / nfp, n_turns, phi0=0.0)
        P = np.stack([tr["R"].ravel(), tr["Z"].ravel()], -1)
        d = distance_to_section(Rm, Zm, P)
        rows.append(dict(curve=what, index=i, flux=e["flux"], flux_reference=flux_ref,
                         flux_rel_diff=abs(abs(flux_ref) - abs(e["flux"])) / abs(e["flux"]),
                         returns=int(len(P)), max_return_distance_m=float(np.max(d)),
                         trace_err_m=float(np.max(tr["err"])), lost=int(np.max(tr["lost"]))))
    # the axis: one field period from its point at phi = 0 returns to it
    ax = data["axis"]
    tr = trace.poincare(Fld, np.array([ax["R_phi0"]]), np.array([ax["Z_phi0"]]),
                        2 * np.pi * np.arange(1, nfp + 1) / nfp, 1, phi0=0.0)
    axis_ret = float(np.max(np.hypot(tr["R"] - ax["R_phi0"], tr["Z"] - ax["Z_phi0"])))
    return dict(curves=rows, axis_return_distance_m=axis_ret,
                note="torux.reference.fast64 field and torux.reference.trace (Gauss-Legendre IRK, step doubling). "
                     "The flux is minus the loop integral of A along the drawn section, by the 512-point trapezoid "
                     "rule. The returns are at phi = 2 pi k / nfp, which the symmetry maps to phi = 0")


def compute():
    data = run_request()
    data["check"] = check_with_reference(data)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w") as fh:
        json.dump(data, fh, indent=1)
    return data


def _resonances(iotas, nfp, mmax=30):
    """The resonances iota = n nfp / m, m <= mmax, that lie between consecutive iotas: (m, n, iota, step index)."""
    out = []
    for i in range(len(iotas) - 1):
        lo, hi = sorted((iotas[i], iotas[i + 1]))
        for m in range(1, mmax + 1):
            for n in range(math.ceil(lo * m / nfp), math.floor(hi * m / nfp) + 1):
                if lo < n * nfp / m < hi and math.gcd(m, n) == 1:
                    out.append(dict(m=m, n=n, iota=n * nfp / m, between=[i, i + 1]))
    return sorted(out, key=lambda r: r["m"])


def numbers(d, K, region, psi_label, circle_label):
    """The numbers the figure letters and a caption would want, each with what it is and where it comes from."""
    s_req = math.sqrt(abs(d["request"]["flux_Wb"]))
    fam = d["family"]
    first = fam[0]
    th = 2 * np.pi * np.arange(4096) / 4096
    R1, Z1 = section_xy(first["R_m"], first["Z_m"], th)
    ax = d["axis"]
    solves = d["start"]["solves"]
    log = [e for r in d["runs"] for e in r["log"]]
    surf = d["surface"]
    ind = d["independent"]
    chk = d.get("check") or {}
    cons = d.get("consistency") or {}
    rows = chk.get("curves") or []
    return {
        "figure": "start: figure 2 of method/README.md, the half-plane phi = 0 of HSX at true scale",
        "example": {"value": "examples/hsx.json", "what": "HSX, the Helically Symmetric eXperiment: 48 modular coils, "
                    "4 field periods (simsopt 1.11's coil set). The file carries no surface, so torux starts from "
                    "the magnetic axis"},
        "request": {"flux_Wb": d["request"]["flux_Wb"], "call": d["request"]["call"],
                    "what": "the request of examples/any_source.py, run with torux " + str(d["torux_version"])},
        "lettered": {"Psi0": psi_label + " Wb", "start_circle": circle_label + " R0", "magnification": K,
                     "what": "the numbers lettered in the figure: the flux of the heavy curve, the radius of the dashed "
                             "circle and the factor of the inset"},
        "Psi0_Wb": {"value": d["request"]["flux_Wb"], "s_Wb_half": s_req,
                    "what": "the flux asked for, and s = sqrt(Psi) there, to which the s of every ring is referred"},
        "answer": {"status": d["status"], "iota": surf["iota"], "M": surf["M"], "N": surf["N"],
                   "refinement": [[int(h[0]), int(h[1]), h[2]] for h in surf["refine_hist"]],
                   "bn_off_solver": surf["bn_off"], "bn_max_independent": ind.get("bn_max"),
                   "flux_rel_err_independent": ind.get("flux_rel_err"), "minor_radius_m": ind.get("minor_radius"),
                   "what": "the surface asked for, refined and verified: the heavy curve. refinement is (M, N, off-grid "
                           "|B.n|/|B|) from the continuation's torus to the refined one (torux.family.refine_to_limit). "
                           "bn_off_solver is the largest |B.n|/|B| on the grid shifted half a cell, bn_max_independent "
                           "the same at 4,096 random points by torux.reference (torux.verify.verify_torus)",
                   "from": "torux.solve: Result.surface, Result.evidence"},
        "axis": {"R_m": ax["R_phi0"], "Z_m": ax["Z_phi0"], "R0_m": ax["R0"], "iota": ax["iota"],
                 "residue": ax["residue"],
                 "what": "the magnetic axis where it crosses phi = 0 (the dot). R0 is the axis's mean major radius, "
                         "the constant term of its R series, the R0 of the start circle's 0.004 R0. iota and Greene's "
                         "residue from its monodromy",
                 "from": "torux.analysis.find_axes, the axis passed to torux.family.Continuation.start"},
        "start_circle": {"radius_m": d["start"]["eps"], "radius_over_R0": d["start"]["eps"] / ax["R0"],
                         "flux_Wb": d["start"]["flux"],
                         "what": "the circle the solver begins from (dashed, in the inset): torux.family.axis_torus_start "
                                 "with eps = 0.004 R0, and the flux it encloses, at which the first surface is solved",
                         "smaller_circles_tried": d["start"]["ladder"] is not None},
        "first_surface": {"newton_iterations": solves[0]["its"] if solves else None,
                          "M": first["M"], "N": first["N"], "flux_Wb": first["flux"], "s_over_s_request": first["s"] / s_req,
                          "iota": first["iota"], "bn_off": first["bn_off"],
                          "height_m": float(Z1.max() - Z1.min()), "width_m": float(R1.max() - R1.min()),
                          "elongation": float((Z1.max() - Z1.min()) / (R1.max() - R1.min())),
                          "what": "the torus Newton's method made of the circle at its flux (the ellipse in the inset): "
                                  "height and width of its section at phi = 0 and their ratio",
                          "from": "torux.family.Continuation.start"},
        "family": {"count": len(fam), "s_over_s_request": [e["s"] / s_req for e in fam],
                   "flux_Wb": [e["flux"] for e in fam], "iota": [e["iota"] for e in fam],
                   "M": [e["M"] for e in fam], "N": [e["N"] for e in fam], "newton_iterations": [e["its"] for e in fam],
                   "bn_off": [e["bn_off"] for e in fam],
                   "steps_tried": len(log), "steps_failed": sum(1 for e in log if not e["ok"]),
                   "steps_s_over_s_request": [(b["s"] - a["s"]) / s_req for a, b in zip(fam, fam[1:])],
                   "chains_met": d["chains"], "jumps": d["jumps"], "stop": d["family_stop"],
                   "resonances_crossed": _resonances([e["iota"] for e in fam], ax["nfp"]),
                   "what": "every torus the continuation accepted, the start first, at s = sqrt(Psi): the thin rings, "
                           "the last of them, at the request, drawn as the refined surface. The step in s starts at 0.05 "
                           "of the request's s, grows 1.5 times per accepted step up to 0.2, halves on a failure, and the "
                           "last is cut to end at the request. resonances_crossed: the rational iota = n N_fp / m, "
                           "m <= 30, that lie between two consecutive tori (between: their indices), each crossed by an "
                           "accepted step with no failure, so no chain stopped the continuation and none was jumped",
                   "from": "torux.family.Continuation._run (FamilyResult.tori, .log), driven by torux.analysis.analyse"},
        "inset": {"magnification": K, "region_m": [region[0], region[1]],
                  "what": "the inset is the region of the dot outlined in the main drawing (width and height in R and "
                          "Z), magnified this many times, line weights unchanged"},
        "check": {"what": "the drawn sections measured by torux.reference, which shares no code with the solver: the "
                          "flux each encloses (minus the loop integral of A along it) against the torus's own, and "
                          "field lines started on each, traced one turn, against the section, at every field period",
                  "flux_rel_diff_max": max((r["flux_rel_diff"] for r in rows), default=None),
                  "return_distance_max_m": max((r["max_return_distance_m"] for r in rows), default=None),
                  "returns_per_curve": rows[0]["returns"] if rows else None,
                  "axis_return_distance_m": chk.get("axis_return_distance_m"),
                  "section_vs_torus_geometry_m": cons.get("section_vs_geometry_m"),
                  "family_vs_answer_profile_max_diff": cons.get("profile_max_diff"),
                  "family_at_request_vs_refined_m": cons.get("last_family_vs_refined_m")},
    }


def _closed(Rm, Zm, n=1440):
    th = 2 * np.pi * np.arange(n) / n
    return section_xy(Rm, Zm, th)


def _clip_box(x, y, box):
    """The points of the closed curve (x, y) inside box = (x0, x1, y0, y1), the pen lifted outside it (NaN)."""
    x0, x1, y0, y1 = box
    xx, yy = np.append(x, x[0]), np.append(y, y[0])
    inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    if inside.all():
        return x, y, True
    return np.where(inside, xx, np.nan), np.where(inside, yy, np.nan), False


# Where things go, in page units (838 wide) and metres: S page units to the metre in the main drawing, its range in
# R and Z and its corner, the inset's region (half widths rx, ry about the axis) and factor K, and where the
# lettering stands.
LAYOUT = dict(S=1050.0, R=(1.37, 1.53), Z=(-0.24, 0.24), left=158.0, top=56.0, H=604.0,
              rx=0.008, ry=0.0105, K=10, inset_left=438.0,
              psi_at=(1.490, 0.14), arrow=(0.020, 0.232), circle_at=25.0)


def draw(stem=None, previews=None):
    """The figure from data/start.json, written as start-light.svg and start-dark.svg beside this file (or at
    stem) with its numbers in start.json. The half-plane phi = 0 at true scale: every torus the continuation
    accepted in the fine line, the surface asked for in the heavy one, the axis a dot, the start circle dashed,
    and the start again in an inset."""
    sys.path.insert(0, HERE)
    import style

    with open(DATA) as fh:
        d = json.load(fh)
    L = LAYOUT
    S, rx, ry, K = L["S"], L["rx"], L["ry"], L["K"]
    Ra, Za = d["axis"]["R_phi0"], d["axis"]["Z_phi0"]
    eps = d["start"]["eps"]
    fam = d["family"]
    surf = d["surface"]
    psi_label = f"{d['request']['flux_Wb']:.3g}"
    circle_label = f"{eps / d['axis']['R0']:.3g}"
    fig = style.Figure(height=L["H"])
    (R0, R1), (Z0, Z1) = L["R"], L["Z"]
    ax = fig.axes((L["left"], L["top"], (R1 - R0) * S, (Z1 - Z0) * S), x=(R0, R1), y=(Z0, Z1))
    # the family: every torus the continuation accepted, the last one (at the request) as the refined surface
    for e in fam[:-1]:
        ax.line(*_closed(e["R_m"], e["Z_m"]), weight="fine", closed=True)
    ax.line(*_closed(surf["R_m"], surf["Z_m"]), weight="heavy", closed=True)
    # the start: the circle round the axis, a speck here and shown again in the inset
    th = 2 * np.pi * np.arange(720) / 720
    circle = (Ra + eps * np.cos(th), Za + eps * np.sin(th))
    ax.line(*circle, weight="fine", dash="dashed", closed=True)
    ax.mark(Ra, Za, kind="dot")
    # the march, outward in s
    ax.arrow(Ra, Za + L["arrow"][0], Ra, L["arrow"][1], weight="medium")
    ax.text(Ra, L["arrow"][1], r"$s=\sqrt{\Psi}$", anchor="w", gap=8)
    ax.text(*L["psi_at"], r"$\Psi_0 = %s$ Wb" % psi_label, anchor="w", gap=0)
    ax.frame("open")
    ax.xticks(major=[1.40, 1.45, 1.50], labels=["$1.40$", "$1.45$", "$1.50$"])
    ax.yticks(major=[-0.2, -0.1, 0.0, 0.1, 0.2], labels=["$-0.2$", "$-0.1$", "$0$", "$0.1$", "$0.2$"])
    ax.xlabel("$R$ (m)")
    ax.ylabel("$Z$ (m)")
    # the start, magnified K times, the inset level with the axis so that its two joining lines open evenly
    bw, bh = 2 * rx * S * K, 2 * ry * S * K
    top = float(ax.page(Ra, Za)[1]) - bh / 2
    iv = fig.inset(ax, x=(Ra - rx, Ra + rx), y=(Za - ry, Za + ry), box=(L["inset_left"], top, bw, bh))
    region = (Ra - 1.5 * rx, Ra + 1.5 * rx, Za - 1.5 * ry, Za + 1.5 * ry)
    for e in fam:
        x, y = _closed(e["R_m"], e["Z_m"])
        x, y, whole = _clip_box(x, y, region)
        if np.isfinite(x).any():
            iv.line(x, y, weight="fine", closed=whole)
    iv.line(*circle, weight="fine", dash="dashed", closed=True)
    iv.mark(Ra, Za, kind="dot")
    cx, cy = L["inset_left"] + bw / 2, top + bh / 2
    fig.leader(cx, top + bh + 36, "magnetic axis", cx, cy)
    a = math.radians(L["circle_at"])
    fig.leader(L["inset_left"] + bw + 36, top + 0.18 * bh, r"$%s\,R_0$" % circle_label,
               cx + eps * S * K * math.cos(a), cy - eps * S * K * math.sin(a))
    fig.text(L["inset_left"] + bw, top, r"$\times\,%d$" % K, anchor="sw", gap=6)
    stem = stem or os.path.join(HERE, "start")
    with open(NUMBERS if stem == os.path.join(HERE, "start") else stem + ".json", "w") as fh:
        json.dump(numbers(d, K, (2 * rx, 2 * ry), psi_label, circle_label), fh, indent=1)
        fh.write("\n")
    return fig.save(stem, previews=previews)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compute", action="store_true", help="run the request again and keep what the solver did")
    ap.add_argument("--no-draw", action="store_true", help="with --compute: only compute")
    ap.add_argument("--previews", metavar="DIR", help="also render PNG previews at 838 and 324 px into DIR")
    a = ap.parse_args(argv)
    if a.compute or not os.path.exists(DATA):
        compute()
    if not a.no_draw:
        draw(previews=a.previews)


if __name__ == "__main__":
    sys.exit(main())
