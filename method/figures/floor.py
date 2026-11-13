"""Figure 3 of the method page: the series falls geometrically, and the refinement goes to the floor.

    python method/figures/floor.py              # draw from data/floor.json
    python method/figures/floor.py --compute    # solve the request again (seconds), keep what the refinement
                                                # read and did, then draw

The request is QUASR 954 at 4.5 times the flux its file asks for:

    src = torux.load("examples/quasr_0000954.json")
    torux.solve(src, flux=4.5 * src.request["flux_Wb"])

HSX at 0.0359221439332229 Wb, the surface the other figures show, refines in two levels only (M, N = 112, 48 and
then 192, 96), as does QUASR 954 at its own request (24, 24 and then 48, 32). This request is the first bundled
one found whose refinement has three levels spread down the spectrum rather than piled at the floor.

What is drawn, and from what:

- The heavy curve: the spectrum of the verified surface, for each m the largest sqrt(R_mn^2 + Z_mn^2) over n,
  relative to the largest with m >= 1 (which is at m = 1). This is the quantity torux.family.predict_resolution
  reads.
- An open circle for each level of the refinement, at that level's M and at the largest |B.n|/|B| the solver
  measured on the grid shifted half a cell from the one it solved on (refine_hist in the answer's evidence). The
  style's own open circle does not read 324 px wide, so these are drawn larger, from its paper tone and its medium
  line.
- The dashed line: the prediction made at the first level. torux.family.predict_resolution is wrapped while the
  request is solved, so the spectrum it read and the M, N it returned are kept. The straight line it fits to the
  last half of that spectrum is recomputed here with torux's own constants and drawn from the start of the fit to
  where it meets the floor the solver aims at (1e-17 of the shape). Every recomputed M, N must be the one the
  solver went to, or nothing is drawn.
- The floor of double precision, hatched and ruled at the largest |B.n|/|B| that the independent evaluator reads
  on an exact invariant torus written in double precision (results/evaluator/test_output.txt, read here).

Lettering that comes within 3 units of a line stops the drawing. The numbers in the figure, and those a caption
needs, are written to floor.json beside this file.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data", "floor.json")
NUMBERS = os.path.join(HERE, "floor.json")
SOURCE = os.path.join(REPO, "examples", "quasr_0000954.json")
SOURCE_NAME = "examples/quasr_0000954.json"
FACTOR = 4.5                    # times the flux the file asks for
EVALUATOR_OUTPUT = os.path.join(REPO, "results", "evaluator", "test_output.txt")


# ---------------------------------------------------------------------------------------------- the run

def spectra(rc, zs, N):
    """What torux.family.predict_resolution reads off a torus: for each m the largest coefficient amplitude
    sqrt(rc^2 + zs^2) over n, for each |n| the largest over m and the sign of n, both relative to the largest
    amplitude with m >= 1 (the scale, returned in metres)."""
    a = np.sqrt(np.asarray(rc, float) ** 2 + np.asarray(zs, float) ** 2)
    scale = max(float(np.max(a[1:])), 1e-300)
    rm = np.max(a, axis=1) / scale
    cn = np.max(a, axis=0) / scale
    nn = np.array([max(cn[N + n], cn[N - n]) for n in range(N + 1)])
    return rm, nn, scale


def run_request():
    """torux.solve on the request, with the refinement watched. Returns the record kept in data/floor.json."""
    import torux
    from torux import family as F

    seen = dict(predict=[], refines=[])
    depth = [0]
    orig_predict, orig_refine = F.predict_resolution, F._refine

    def predict_resolution(t, target, cont, *a, **k):
        out = orig_predict(t, target, cont, *a, **k)
        if depth[0] > 0:
            rm, nn, scale = spectra(t.rc, t.zs, int(t.N))
            seen["predict"].append(dict(refine=len(seen["refines"]), M=int(t.M), N=int(t.N), bn=float(t.bn_off),
                                        scale_m=scale, spectrum_m=rm.tolist(), spectrum_n=nn.tolist(),
                                        args=[float(v) for v in a], kwargs={kk: float(v) for kk, v in k.items()},
                                        Mmax=int(cont.Mmax), Nmax=int(cont.Nmax), went_to=[int(v) for v in out]))
        return out

    def _refine(cont, sol, flux, target, max_levels, deadline, verbose, fixed_iota, hist, best):
        depth[0] += 1
        try:
            out = orig_refine(cont, sol, flux, target, max_levels, deadline, verbose, fixed_iota, hist, best)
        finally:
            depth[0] -= 1
        seen["refines"].append(dict(hist=[[int(m), int(n), float(b)] for m, n, b in out.info.get("refine_hist", [])],
                                    stop=out.info.get("refine_stop"), target=float(target)))
        return out

    src = torux.load(SOURCE)
    own = float(src.request["flux_Wb"])
    flux = FACTOR * own
    F.predict_resolution, F._refine = predict_resolution, _refine
    t0 = time.time()
    try:
        res = torux.solve(src, flux=flux)
    finally:
        F.predict_resolution, F._refine = orig_predict, orig_refine
    wall = time.time() - t0
    if res.status != "surface":
        raise RuntimeError(f"the request came back {res.status!r}, not a verified surface")
    ev = res.evidence
    hist = [[int(m), int(n), float(b)] for m, n, b in ev["solver"]["refine_hist"]]
    # the refinement whose levels are the answer's (a request can refine more than one torus on its way)
    k = [i for i, r in enumerate(seen["refines"]) if r["hist"] == hist]
    if len(k) != 1:
        raise RuntimeError(f"{len(k)} refinements match the answer's refine_hist")
    preds = [p for p in seen["predict"] if p["refine"] == k[0]]
    s = res.surface
    rm, nn, scale = spectra(s.rc, s.zs, int(s.N))
    ind = ev.get("independent") or {}
    return dict(
        request=dict(source=SOURCE_NAME, factor=FACTOR, own_flux_Wb=own, flux_Wb=flux,
                     call=[f"src = torux.load('{SOURCE_NAME}')",
                           f"torux.solve(src, flux={FACTOR} * src.request['flux_Wb'])"]),
        torux_version=getattr(torux, "__version__", None), status=res.status, wall_s=wall, budget_s=res.budget_s,
        surface=dict(M=int(s.M), N=int(s.N), nfp=int(s.nfp), iota=float(s.iota), flux_Wb=float(s.flux),
                     R00_m=float(np.asarray(s.rc)[0, int(s.N)]), scale_m=scale, spectrum_m=rm.tolist(),
                     spectrum_n=nn.tolist()),
        refine_hist=hist, refine_stop=seen["refines"][k[0]]["stop"], refine_target=seen["refines"][k[0]]["target"],
        predictions=preds,
        solver=dict(bn_off=ev["solver"].get("bn_off"), quad_err_half=ev["solver"].get("quad_err_half"),
                    guard_ok=(ev["solver"].get("guard") or {}).get("ok")),
        independent={kk: v for kk, v in ind.items() if isinstance(v, (int, float, str, bool)) or v is None},
        truncated=truncation_check(src, s, hist),
    )


def truncation_check(src, s, hist):
    """The verified surface cut to the M, N of each earlier level and measured by the solver's own off-grid check,
    on a field engine built as the analysis builds its first one: the leak of the coefficients that level lacks."""
    from torux.field import FieldEngine
    from torux.torus import TorusProblem, TorusState
    eng = FieldEngine(src, dmin=src.meta.get("dmin", 0.05))
    T = TorusState(np.asarray(s.rc), np.asarray(s.zs), float(s.iota), float(s.flux), int(s.M), int(s.N), int(s.nfp),
                   phi0=float(getattr(s, "phi0", 0.0)))
    out = []
    for M, N, bn in hist[:-1]:
        t = T.resized(M, N)
        _, b = TorusProblem(eng, int(s.nfp), M, N, phi0=T.phi0).check(t.rc, t.zs, t.iota)
        out.append(dict(M=M, N=N, bn_level=bn, bn_truncated=float(b)))
    return out


def compute():
    data = run_request()
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w") as fh:
        json.dump(data, fh, indent=1)
    return data


# ------------------------------------------------------------------------------------- the prediction

def solver_constants():
    """The floor, the largest factor and the ladder that torux.family.predict_resolution uses, from torux itself."""
    from torux import family as F
    sig = inspect.signature(F.predict_resolution).parameters
    return float(sig["floor"].default), float(sig["maxfac"].default), list(F.LADDER)


def prediction(p, floor, maxfac, ladder):
    """predict_resolution's arithmetic on the spectra it read: the straight line through log(spectrum) over the
    last half of the m (and of the n), where it meets the floor, and the M, N that follow from that."""
    def need(v, cur):
        v = np.asarray(v, float)
        k = np.arange(len(v))
        lo = max(1, len(v) // 2)
        vv = np.maximum(v[lo:], 1e-300)
        good = vv > 1e-300
        if good.sum() < 3:
            return dict(need=cur)
        slope, icpt = np.polyfit(k[lo:][good], np.log(vv[good]), 1)
        if slope >= -1e-3:
            return dict(lo=lo, hi=len(v) - 1, slope=float(slope), icpt=float(icpt), need=int(maxfac * cur))
        kneed = (math.log(floor) - icpt) / slope
        return dict(lo=lo, hi=len(v) - 1, slope=float(slope), icpt=float(icpt), meets_floor_at=float(kneed),
                    plus_two=int(np.ceil(kneed) + 2), need=int(np.clip(np.ceil(kneed) + 2, cur, maxfac * cur)))
    fm = need(p["spectrum_m"], p["M"])
    fn = need(p["spectrum_n"], p["N"]) if p["N"] > 2 else dict(need=p["N"])
    M2 = next((L for L in ladder if L >= fm["need"]), ladder[-1])
    N2 = next((L for L in ladder if L >= fn["need"]), ladder[-1])
    return dict(m=fm, n=fn, M=min(M2, p["Mmax"]), N=min(N2, p["Nmax"]))


def evaluator_floor():
    """The largest |B.n|/|B| the independent evaluator finds, in extended precision, on an exact invariant torus
    written as a double-precision series: what a double-precision surface cannot read below."""
    with open(EVALUATOR_OUTPUT) as fh:
        text = fh.read()
    m = re.search(r"exact torus \(float128\): max \|B\.n\|/\|B\| ([0-9.]+e[-+][0-9]+)", text)
    if m is None:
        raise RuntimeError(f"no extended-precision exact-torus line in {EVALUATOR_OUTPUT}")
    return float(m.group(1))


# ------------------------------------------------------------------------------------- the figure

XR = (0.0, 240.0)               # m across
YR = (1e-18, 10 ** 0.5)          # the spectrum and |B.n|/|B| up


def checked(data):
    """The figure's quantities, each checked against what the solver did."""
    floor, maxfac, ladder = solver_constants()
    hist = data["refine_hist"]
    preds = [prediction(p, floor, maxfac, ladder) for p in data["predictions"]]
    for i, (p, q) in enumerate(zip(data["predictions"], preds)):
        if [q["M"], q["N"]] != p["went_to"] or p["went_to"] != hist[i + 1][:2] or [p["M"], p["N"]] != hist[i][:2]:
            raise RuntimeError(f"prediction {i}: recomputed {q['M']}, {q['N']}, the solver went to {p['went_to']}, "
                               f"its levels are {hist}")
    rm = np.asarray(data["surface"]["spectrum_m"])
    if data["surface"]["M"] != hist[-1][0] or len(rm) != hist[-1][0] + 1:
        raise RuntimeError("the surface is not the last level of the refinement")
    inside = (YR[0] < np.min(rm[1:]) and np.max(rm[1:]) < YR[1] and preds[0]["m"]["meets_floor_at"] < XR[1]
              and all(YR[0] < bn < YR[1] for _, _, bn in hist))
    if not inside:
        raise RuntimeError("the data run outside the plot")
    return dict(floor=floor, maxfac=maxfac, ladder=ladder, preds=preds, edge=evaluator_floor(), rm=rm, hist=hist)


RING = 7.5                       # radius of a level's circle, page units


def _near(box, lines, margin):
    """Whether any of the polylines (page units) comes within margin of the box (left, top, right, bottom)."""
    l, t, r, b = box
    for p in lines:
        p = np.asarray(p, float)
        for a, q in zip(p[:-1], p[1:]):
            n = max(2, int(np.hypot(*(q - a))) + 2)
            s = a + np.linspace(0, 1, n)[:, None] * (q - a)
            if np.any((s[:, 0] > l - margin) & (s[:, 0] < r + margin) & (s[:, 1] > t - margin) & (s[:, 1] < b + margin)):
                return True
    return False


def draw(data, previews=None):
    sys.path.insert(0, HERE)
    import style as S
    c = checked(data)
    rm, hist, edge = c["rm"], c["hist"], c["edge"]
    m = np.arange(len(rm))
    fit = c["preds"][0]["m"]

    fig = S.Figure(455)
    L, T, W, H = 66.0, 52.0, 752.0, 350.0
    ax = fig.axes((L, T, W, H), XR, YR, ylog=True)
    drawn = []                                   # what lettering must keep clear of, in page units

    # the floor of double precision, hatched up to the evaluator's reading of an exact torus, its top ruled
    ax.fill([XR[0], XR[1], XR[1], XR[0]], [YR[0], YR[0], edge, edge], tone="hatch")
    ax.line(XR, [edge, edge], weight="fine")
    # the prediction from the first level: the line through the last half of its spectrum, to the solver's floor
    xs = np.array([fit["lo"], fit["meets_floor_at"]])
    ys = np.exp(fit["icpt"] + fit["slope"] * xs)
    ax.line(xs, ys, weight="medium", dash="dashed", halo=True)
    drawn.append(np.column_stack(ax.page(xs, ys)))
    # the spectrum of the verified surface, from m = 1: where it nears the floor it is cut clear of the hatching and
    # passes in front of the dashed line, and above that it leaves the dashed line whole along the fit
    near = m >= int(np.argmax(rm[1:] < 10 * edge)) + 1
    ax.line(m[near], rm[near], weight="heavy", halo=True)
    ax.line(m[1:], rm[1:], weight="heavy")
    drawn.append(np.column_stack(ax.page(m[1:], rm[1:])))
    # the levels of the refinement: an open circle each, larger than the style's own so that it reads at 324 px
    th = np.linspace(0, 2 * np.pi, 49)
    for M, N, bn in hist:
        X, Y = (float(v) for v in ax.page(M, bn))
        fig.fill(X + RING * np.cos(th), Y + RING * np.sin(th), tone="paper")
        fig.line(X + RING * np.cos(th), Y + RING * np.sin(th), weight="medium", closed=True)
        drawn.append(np.column_stack([X + RING * np.cos(th), Y + RING * np.sin(th)]))

    ax.frame("box")
    ax.xticks(major=[0, 40, 80, 120, 160, 200, 240], minor=list(range(20, 240, 40)))
    ax.yticks(major=[1.0, 1e-4, 1e-8, 1e-12, 1e-16],
              minor=[10.0 ** k for k in range(-18, 1) if k % 4 != 0])
    ax.xlabel("$m$")
    fig.text(L - 52, T - 10, r"$\max_n\,(R_{mn}^2+Z_{mn}^2)^{1/2}$", anchor="sw", gap=0)

    boxes = []
    (M0, N0, b0), (M1, N1, b1), (M2, N2, b2) = hist[:3]
    boxes.append(ax.text(M0, b0, rf"$|\mathbf{{B}}\cdot\mathbf{{n}}|/|B|$ at $M, N = {M0}, {N0}$", anchor="sw",
                         size="small", gap=RING + 3))
    boxes.append(ax.text(M1, b1, f"{M1}, {N1}", anchor="ne", size="small", gap=RING + 2))
    boxes.append(ax.text(M2, b2, f"{M2}, {N2}", anchor="s", size="small", gap=RING + 4))
    xl = 128.0
    boxes.append(ax.text(xl, math.exp(fit["icpt"] + fit["slope"] * xl), f"prediction from $M = {M0}$", anchor="sw",
                         size="small", gap=7))
    boxes.append(ax.text(14, math.sqrt(edge * YR[0]), "floor of double precision", anchor="w"))
    for bx in boxes:
        if _near(bx, drawn, 3.0):
            raise RuntimeError(f"the label at {[round(v) for v in bx]} touches a line")

    return fig.save(os.path.join(HERE, "floor"), previews=previews)


def numbers(data):
    """Write floor.json: every number in the figure and every number its caption needs, each with what it is."""
    c = checked(data)
    rm, hist, edge = c["rm"], c["hist"], c["edge"]
    s, req, ind = data["surface"], data["request"], data["independent"]
    nn = np.asarray(s["spectrum_n"])
    ln10 = math.log(10.0)

    def decades_per(v, a, b):
        k = np.arange(a, b + 1)
        return float(np.polyfit(k, np.log10(np.asarray(v)[a:b + 1]), 1)[0])

    below_edge = int(np.argmax(rm[1:] < edge)) + 1
    below_floor = int(np.argmax(rm[1:] < c["floor"])) + 1
    iota, nfp = s["iota"], s["nfp"]
    spikes = []
    for k in range(1, 10):
        m0 = k * nfp / abs(iota)
        if m0 > len(rm) - 4:
            break
        lo, hi = int(m0) - 3, int(m0) + 4
        spikes.append(dict(resonance_m=round(m0, 1), n=k, largest_nearby_m=lo + int(np.argmax(rm[lo:hi]))))
    preds = []
    for p, q in zip(data["predictions"], c["preds"]):
        f = q["m"]
        preds.append(dict(
            read_at=dict(M=p["M"], N=p["N"], bn=p["bn"]),
            fit_m=[f["lo"], f["hi"]], decades_per_m=f["slope"] / ln10, meets_floor_at_m=f.get("meets_floor_at"),
            M_plus_two=f.get("plus_two"), M_within_maxfac=f["need"],
            fit_n=[q["n"].get("lo"), q["n"].get("hi")], decades_per_n=q["n"].get("slope", float("nan")) / ln10,
            meets_floor_at_n=q["n"].get("meets_floor_at"), N_plus_two=q["n"].get("plus_two"),
            N_within_maxfac=q["n"]["need"], recomputed=[q["M"], q["N"]], solver_went_to=p["went_to"],
            agrees=[q["M"], q["N"]] == p["went_to"]))
    leak = []
    for i, (M, N, bn) in enumerate(hist[:-1]):
        t = data["truncated"][i]
        leak.append(dict(M=M, N=N, bn=bn, first_missing_m=float(rm[M + 1]), bn_over_first_missing_m=bn / float(rm[M + 1]),
                         bn_of_the_surface_cut_to_M_N=t["bn_truncated"], bn_over_that=bn / t["bn_truncated"]))
    out = {
        "figure": "Figure 3 of the method page, method/figures/floor-light.svg and floor-dark.svg, drawn by "
                  "method/figures/floor.py from method/figures/data/floor.json",
        "example": dict(
            source=req["source"], flux_Wb=req["flux_Wb"], own_flux_Wb=req["own_flux_Wb"], factor=req["factor"],
            call=req["call"],
            what="QUASR 954 at 4.5 times the toroidal flux its file asks for: a surface outside the design's own "
                 "boundary, verified",
            why="HSX at 0.0359221439332229 Wb, the surface of the other figures, refines in two levels (112, 48 at "
                "1.3e-9, then 192, 96 at 3.0e-15), and so does QUASR 954 at its own request (24, 24 at 1.9e-11, then "
                "48, 32 at 1.1e-15), each solved separately by the same torux.solve. At 4.5 times its request QUASR "
                "954 refines in three, spread down the spectrum."),
        "surface": dict(status=data["status"], M=s["M"], N=s["N"], nfp=nfp, iota=iota, flux_Wb=s["flux_Wb"],
                        R00_m=s["R00_m"], minor_radius_m=ind.get("minor_radius"),
                        a1_m=s["scale_m"],
                        a1_is="the largest sqrt(R_mn^2 + Z_mn^2) with m >= 1, reached at m = 1: the spectrum is drawn "
                              "relative to it",
                        call="answer.surface (rc, zs), answer.iota"),
        "spectrum": dict(
            what="for each m the largest sqrt(R_mn^2 + Z_mn^2) over n, relative to a1: the heavy curve, m = 1 to M",
            decades_per_m_from_m_10_to_where_it_reaches_the_floor=decades_per(rm, 10, below_edge),
            first_m_below_the_floor_edge=below_edge, first_m_below_1e_17=below_floor,
            at_M=float(rm[-1]), smallest=float(np.min(rm[1:])),
            spikes=spikes,
            spikes_are="the harmonics nearest resonance, where m iota is closest to a multiple n N_fp of N_fp: every "
                       f"N_fp / |iota| = {nfp / abs(iota):.1f} in m",
            n_side=dict(what="the same over m, for each |n|: not drawn", decades_per_n_from_n_4_to_32=decades_per(nn, 4, 32),
                        at_N=float(nn[-1]))),
        "refinement": dict(
            levels=[dict(M=M, N=N, bn=bn) for M, N, bn in hist],
            bn_is="the largest |B.n|/|B| the solver measured at that level on a grid shifted half a cell from the one "
                  "it solved on: the circles",
            target=data["refine_target"], stop=data["refine_stop"],
            call="answer.evidence['solver']['refine_hist']"),
        "prediction": preds,
        "prediction_rule": dict(floor=c["floor"], maxfac=c["maxfac"],
                                rule="torux.family.predict_resolution fits a straight line to the log of the "
                                     "spectrum over its last half. The next M is where that line meets floor, plus "
                                     "2, at most maxfac times the current M, raised to the next value of "
                                     "torux.family.LADDER. N likewise. The dashed line is the first of these fits, "
                                     "drawn from m = fit_m[0] to meets_floor_at_m"),
        "floor": dict(upper_edge=edge,
                      source="results/evaluator/test_output.txt: the largest |B.n|/|B| that the independent "
                             "evaluator, in extended precision, reads on an exact invariant torus written as a "
                             "double-precision series. results/README.md rounds it to 3.9e-16"),
        "leak_against_spectrum": leak,
        "verification": dict(solver_bn_off=data["solver"]["bn_off"], quad_err_half=data["solver"]["quad_err_half"],
                             guard_ok=data["solver"]["guard_ok"], independent_ok=ind.get("ok"),
                             independent_bn_max=ind.get("bn_max"), independent_bn_sup=ind.get("bn_sup"),
                             independent_npts=ind.get("npts"), flux_rel_err=ind.get("flux_rel_err"),
                             call="answer.evidence['solver'] and answer.evidence['independent']"),
        "run": dict(torux_version=data["torux_version"], wall_s=data["wall_s"], budget_s=data["budget_s"]),
    }
    with open(NUMBERS, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compute", action="store_true", help="solve the request again and keep what the refinement did")
    ap.add_argument("--previews", default=None, help="also render PNGs at 838 and 324 px wide into this folder")
    a = ap.parse_args(argv)
    if a.compute or not os.path.exists(DATA):
        data = compute()
    else:
        with open(DATA) as fh:
            data = json.load(fh)
    numbers(data)
    draw(data, previews=a.previews)


if __name__ == "__main__":
    sys.exit(main())
