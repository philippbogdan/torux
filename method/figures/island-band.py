"""Figure 4 of the method page: where an island chain holds the flux asked for, the certificate that no flux
surface encloses it.

    python method/figures/island-band.py              # draw from data/island-band.json
    python method/figures/island-band.py --compute    # solve again (minutes), keep the certificate's curves, draw

The request is the second one examples/no_surface.py makes, at the flux the file carries:

    torux.solve("examples/quasr_0219457.json", budget_s=3300)

It returns no_surface with an island-band certificate (torux.islandband, whose docstring holds the theorem). The
answer keeps the certificate's interval, its chain and the chain's X and O points, but not its two closed curves,
so the run is watched: torux.islandband.island_band_certificate_manifold is called with keep_polygons=True, and
torux.islandband._assemble and _certify_curve are wrapped. That keeps, from the very run that answers, the two
certified polygons, the heteroclinic orbit each was closed with (its point on Z = 0 and that point's images in the
other gaps of the chain) and, for every gap, which X point owns its unstable arc and which its stable arc. Nothing
in the picture is recomputed and nothing comes from a model.

The certified polygons pass the X points and the heteroclinic points between their vertices (the certificate
thins its polygons within their error tube), so each polygon is cut at the foot of each point on its nearest
segment. That adds vertices on the segments and leaves the polygon, and its flux, as they were: --compute checks
the flux of the polygons as drawn with the certificate's own quadrature and with the independent reference
evaluator, and the interval again from them, digit for digit as torux prints it.

What is drawn, in the certificate's symmetry plane phi = 0, with Z = 0 one dash-dot line through both parts:

- right, at true scale: the chain (at this scale its two curves are one fine line, its X points crosses and its O
  points dots), the magnetic axis, and in heavy line the stretch of chain shown on the left.
- left, that stretch straightened. s is arc length along a reference curve, downwards from Z = 0, and n the
  distance from it along its outward normal, drawn STRETCH times larger than s (the scale bars: 10 mm along, 10
  micrometres across). Outward is to the left, as in the section. The two certified curves are heavy, solid where
  they are arcs of unstable branches and dashed where they are the mirrored stable arcs, which meet at the
  heteroclinic points (open circles where that is on Z = 0). Between the curves, cross-hatched, the band no flux
  surface around the axis enters. The X points are where the curves cross: there the unstable arcs of both curves
  make one smooth solid line (the unstable manifold) and the stable arcs one dashed line.
- the reference curve is the planar section at phi = 0 of the outermost surface stored in the file. torux.load
  takes the request as the flux that surface encloses (QUASR's definition), and its planar section at phi = 0
  encloses the same flux to about 5e-11 of it (torux.design.planar_section and section_flux, checked in
  --compute). So the line n = 0, marked by the arrows at both ends of the window, is a curve that encloses the
  requested flux. It is not a flux surface: the certificate proves there is none at that flux. Every X and O point
  of the chain lies within 0.2 micrometres of it.

The fluxes are lettered to 4 significant figures. At 4 figures the fluxes of the two drawn curves are also the ends
of the certified interval, which is narrower than the curves' fluxes by the error terms (7e-8 Wb on each side).
Every number lettered in the figure, and the ones a caption needs, go to island-band.json beside this file.
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
DATA = os.path.join(HERE, "data", "island-band.json")
NUMBERS = os.path.join(HERE, "island-band.json")
SOURCE = os.path.join(REPO, "examples", "quasr_0219457.json")
BUDGET_S = 3300
N_FOURIER = 40          # cosines of R and sines of Z kept for the design section (they reach rounding by m = 25)


# ---------------------------------------------------------------------------------------------- the run

def _fourier_section(R, Z):
    """R = sum a_m cos(m t), Z = sum b_m sin(m t) of a section sampled at t = 2 pi j / n (stellarator symmetric
    at phi = 0: R even and Z odd in t)."""
    n = len(R)
    cR = np.fft.rfft(R) / n
    cZ = np.fft.rfft(Z) / n
    a = np.real(cR[:N_FOURIER]).copy()
    a[1:] *= 2.0
    b = -2.0 * np.imag(cZ[:N_FOURIER])
    b[0] = 0.0
    return a, b


def _eval_fourier(a, b, t):
    m = np.arange(len(a))
    return np.cos(np.outer(t, m)) @ a, np.sin(np.outer(t, m)) @ b


def run_request():
    """torux.solve on the request with the island-band certificate watched. Returns the record kept in DATA."""
    import torux
    import torux.islandband as IB
    from torux.barrier import reference_flux
    from torux.design import planar_section, section_flux

    log = []
    orig = (IB._assemble, IB._certify_curve, IB.island_band_certificate_manifold)

    def assemble(sec, orb, a, cls, cu, cs, rec, hj, ej, *args, **kw):
        out = orig[0](sec, orb, a, cls, cu, cs, rec, hj, ej, *args, **kw)
        log.append(dict(what="assemble", cls=cls, ok=bool(out.get("ok")), hj=np.array(hj, float),
                        gaps=[[int(v) for v in jn["gap"]] for jn in out.get("junctions", [])]))
        return out

    def certify(sec, C, E, a, a_err, corners=None):
        cert, C_fin = orig[1](sec, C, E, a, a_err, corners=corners)
        log.append(dict(what="certify", sec=sec, flux=cert.get("flux"), C=None if C_fin is None else np.array(C_fin)))
        return cert, C_fin

    def manifold(*args, **kw):
        kw["keep_polygons"] = True
        out = orig[2](*args, **kw)
        log.append(dict(what="manifold", out=out))
        return out

    IB._assemble, IB._certify_curve, IB.island_band_certificate_manifold = assemble, certify, manifold
    try:
        src = torux.load(SOURCE)
        res = torux.solve(src, budget_s=BUDGET_S)
    finally:
        IB._assemble, IB._certify_curve, IB.island_band_certificate_manifold = orig
    if res.status != "no_surface":
        raise SystemExit(f"the request came back {res.status!r} after {res.wall_s:.0f} s, not no_surface: run again "
                         f"on a quieter machine")
    c = res.evidence["certificate"]
    if c.get("kind") != "island_band_manifold" or not c.get("ok") or not c.get("holds_request"):
        raise SystemExit(f"not an island-band certificate holding the request: {c.get('kind')}")
    kept = [e["out"] for e in log if e["what"] == "manifold" and e["out"].get("ok")
            and list(e["out"]["interval"]) == list(c["interval"])]
    if not kept:
        raise SystemExit("the certificate's own call was not seen")
    polys = {k: np.asarray(v, float) for k, v in kept[-1]["polygons"].items()}
    X = np.asarray(c["X_line"]["points"], float)
    curves, checks = {}, {}
    for cls in ("A", "B"):
        rec = c["curves"][cls]
        i = [n for n, e in enumerate(log) if e["what"] == "certify" and e["flux"] == rec["flux"]][-1]
        cert, asm = log[i], [e for e in log[:i] if e["what"] == "assemble" and e["cls"] == cls][-1]
        C = polys[cls]
        if not np.array_equal(cert["C"], C):
            raise SystemExit(f"curve {cls}: the kept polygon is not the certified one")
        H = asm["hj"]
        if not np.array_equal(H[0], np.asarray(rec["crossing"]["h"], float)):
            raise SystemExit(f"curve {cls}: the heteroclinic orbit does not start at the recorded crossing")
        sec = cert["sec"]
        P, arcs, info = split_polygon(C, X, H, asm["gaps"])
        F, qe = sec.flux(C)
        Fp, qp = sec.flux(P)
        ref = reference_flux(sec, P)
        checks[cls] = dict(
            flux_record=rec["flux"], flux_again=F, flux_drawn=Fp, quad_err_drawn=qp,
            flux_drawn_reference=ref.get("flux_reference"), reference_rel_diff=ref.get("rel_diff"),
            F_up_drawn=Fp + rec["quad_err"] + rec["tube"], F_low_drawn=Fp - rec["quad_err"] - rec["tube"],
            X_dist_max=info["dX_max"], h_dist_max=info["dH_max"], arcs=len(arcs), alternating=info["alternating"])
        curves[cls] = dict(polygon=C.tolist(), heteroclinic=H.tolist(), gaps=asm["gaps"])
    lo = min(checks[k]["F_up_drawn"] for k in ("A", "B"))
    hi = max(checks[k]["F_low_drawn"] for k in ("A", "B"))
    checks["interval_from_drawn"] = dict(F_lo=lo, F_hi=hi, printed=[f"{lo:.9g}", f"{hi:.9g}"],
                                         record_printed=[f"{v:.9g}" for v in c["interval"]],
                                         same_digits=[f"{lo:.9g}", f"{hi:.9g}"] == [f"{v:.9g}" for v in c["interval"]])
    # the reference curve of the straightened window: the file's outermost surface cut at phi = 0
    surf = src._design_surfaces[-1]
    eng = res._engine
    A_eval = lambda x: np.asarray(eng.evaluate(np.asarray(x, float), grad=False, A=True)["A"])  # noqa: E731
    R, Z, dR, dZ, ang_err = planar_section(surf, float(c["phi_s"]), ntheta=4096)
    F_design, _ = section_flux(A_eval, float(c["phi_s"]), R, Z, dR, dZ)
    a_m, b_m = _fourier_section(R, Z)
    t = 2 * np.pi * np.arange(len(R)) / len(R)
    Rf, Zf = _eval_fourier(a_m, b_m, t)
    checks["design_section"] = dict(flux_planar=F_design, request=float(src.request["flux_Wb"]),
                                    rel_diff=abs(F_design - src.request["flux_Wb"]) / src.request["flux_Wb"],
                                    fourier_max_err=float(max(np.max(np.abs(Rf - R)), np.max(np.abs(Zf - Z)))),
                                    angle_err=float(ang_err))
    keep = ("kind", "ok", "holds_request", "rigorous", "path", "interval", "request", "margin_rel", "chain", "phi_s",
            "axis_point", "axis_e", "section", "chain_order", "bracket_check", "evidence_ok", "error_budget")
    cert_out = {k: c.get(k) for k in keep}
    cert_out["X_line"] = {k: c["X_line"][k] for k in ("points", "residue", "lam", "e_X")}
    cert_out["O_line"] = {k: c["O_line"][k] for k in ("points", "residue")}
    cert_out["curves"] = {cls: {k: c["curves"][cls].get(k) for k in ("flux", "quad_err", "tube", "F_up", "F_low",
                                                                      "n_vertices", "eps_max", "eps_median",
                                                                      "crossing", "simple_as_assembled")}
                          for cls in ("A", "B")}
    cert_out["flux_reference"] = c.get("evidence", {}).get("flux_reference")
    cert_out["timings"] = c.get("timings")
    return dict(
        what="QUASR 0219457 at the flux its file carries: torux's island-band certificate, with its curves",
        call="torux.solve('examples/quasr_0219457.json', budget_s=3300)",
        torux=getattr(torux, "__version__", None), status=res.status, wall_s=res.wall_s,
        printed=str(res).splitlines(), request_flux_Wb=float(src.request["flux_Wb"]),
        nfp=int(src.nfp), certificate=cert_out, curves=curves,
        design_section=dict(phi=float(c["phi_s"]), R_cos=a_m.tolist(), Z_sin=b_m.tolist(), flux_Wb=F_design,
                            how="torux.design.planar_section (4096 points) and section_flux, the file's outermost "
                                "surface"),
        checks=checks)


# ---------------------------------------------------------------------------------------------- geometry

def _nearest_segment(C, p):
    """Index i, parameter t and distance of the point of the closed polygon C nearest p (segment C[i] -> C[i+1])."""
    D = np.roll(C, -1, axis=0)
    e = D - C
    L2 = np.sum(e * e, 1)
    t = np.clip(np.sum((p - C) * e, 1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    d = np.linalg.norm(C + t[:, None] * e - p, axis=1)
    i = int(np.argmin(d))
    return i, float(t[i]), float(d[i])


def split_polygon(C, X, H, gaps):
    """The closed certified polygon C (counter-clockwise) cut where it passes the X points X (q, 2) and the
    heteroclinic points H (q, 2), H[j] the point of gap j = (uo, so): X point uo owns the gap's unstable arc and so
    its stable arc. The polygon passes each point within its error (it was decimated, so the points are not its
    vertices): the foot of each point on the nearest segment is added as a vertex, which leaves the polygon, and so
    its flux, as it was. Returns P (the polygon with the feet added), the arcs dict(kind "u" or "s", gap j, x the X
    point at its end, pts) in the order of P, and the largest distances of the points from C."""
    C = np.asarray(C, float)
    q = len(X)
    ins, dX, dH = [], [], []
    for k in range(q):
        i, t, d = _nearest_segment(C, X[k])
        dX.append(d)
        ins.append((i, t, "x", k))
    for j in range(q):
        i, t, d = _nearest_segment(C, H[j])
        dH.append(d)
        ins.append((i, t, "h", j))
    ins.sort()
    rows, tags, m = [], [], 0
    for i in range(len(C)):
        rows.append(C[i])
        tags.append(("v", i))
        while m < len(ins) and ins[m][0] == i:
            _, t, kind, j = ins[m]
            rows.append(C[i] + t * (C[(i + 1) % len(C)] - C[i]))
            tags.append((kind, j))
            m += 1
    P = np.array(rows)
    marks = [(n, tag[0], tag[1]) for n, tag in enumerate(tags) if tag[0] in ("x", "h")]
    kinds = [mk[1] for mk in marks]
    alternating = all(kinds[i] != kinds[(i + 1) % len(kinds)] for i in range(len(kinds)))
    if not alternating:
        raise ValueError("the X points and heteroclinic points do not alternate along the polygon")
    arcs = []
    for i in range(len(marks)):
        n0, k0, v0 = marks[i]
        n1, k1, v1 = marks[(i + 1) % len(marks)]
        idx = np.arange(n0, n1 + 1) if n1 > n0 else np.concatenate([np.arange(n0, len(P)), np.arange(0, n1 + 1)])
        j, xk = (v1, v0) if k0 == "x" else (v0, v1)
        uo, so = gaps[j]
        if xk not in (uo, so):
            raise ValueError(f"arc of gap {j} ends at X point {xk}, not at {uo} or {so}")
        arcs.append(dict(kind="u" if xk == uo else "s", gap=j, x=int(xk), pts=P[idx]))
    return P, arcs, dict(dX_max=max(dX), dH_max=max(dH), alternating=alternating)


class Straight:
    """Normal coordinates about the reference curve (the design section, R = sum a_m cos m t, Z = sum b_m sin m t):
    s the arc length along it from its inboard crossing of Z = 0, counter-clockwise in (R, Z) (downwards there),
    n the distance along its outward normal."""

    def __init__(self, a_m, b_m, n=1 << 17):
        t = 2 * np.pi * np.arange(n) / n
        R, Z = _eval_fourier(np.asarray(a_m, float), np.asarray(b_m, float), t)
        G = np.stack([R, Z], -1)
        if 0.5 * np.sum(G[:, 0] * np.roll(G[:, 1], -1) - np.roll(G[:, 0], -1) * G[:, 1]) < 0:
            G = G[::-1]
        i0 = int(np.argmin(np.where(np.abs(G[:, 1]) < 1e-3, G[:, 0], np.inf)))
        G = np.roll(G, -i0, axis=0)
        # start exactly on Z = 0 (the section is symmetric, so its inboard crossing is a sample when n is even)
        seg = np.linalg.norm(np.diff(np.vstack([G, G[:1]]), axis=0), axis=1)
        self.G = G
        self.S = np.concatenate([[0.0], np.cumsum(seg)])
        self.perimeter = float(self.S[-1])
        from scipy.spatial import cKDTree
        self._tree = cKDTree(G)

    def coords(self, x):
        """(s, n) of points x (m, 2), s in (-perimeter / 2, perimeter / 2]."""
        x = np.atleast_2d(np.asarray(x, float))
        _, k = self._tree.query(x)
        G, nG = self.G, len(self.G)
        best_d = np.full(len(x), np.inf)
        s = np.zeros(len(x))
        nn = np.zeros(len(x))
        for i0 in (k - 1, k):
            i0 = i0 % nG
            i1 = (i0 + 1) % nG
            a, e = G[i0], G[i1] - G[i0]
            L = np.linalg.norm(e, axis=1)
            u = e / L[:, None]
            t = np.clip(np.sum((x - a) * u, 1), 0.0, L)
            p = a + t[:, None] * u
            d = np.linalg.norm(x - p, axis=1)
            better = d < best_d
            best_d[better] = d[better]
            s[better] = (self.S[i0] + t)[better]
            normal = np.stack([u[:, 1], -u[:, 0]], -1)
            nn[better] = np.sum((x - p) * normal, 1)[better]
        s = (s + 0.5 * self.perimeter) % self.perimeter - 0.5 * self.perimeter
        return np.stack([s, nn], -1)


# ---------------------------------------------------------------------------------------------- the drawing

STRETCH = 500          # across the chain, the window is drawn this many times larger than along it
S_HALF = 30.0          # mm of chain each side of Z = 0 in the window
HEIGHT = 600.0         # page units (the page is 838 wide)
Y0 = 300.0             # the row of Z = 0
FRAME = (14.0, 50.0, 470.0, 500.0)
K_S = FRAME[3] / (2 * S_HALF)    # page units per mm along the chain in the window
SECTION_X = (532.0, 754.0)       # page x of the section's inboard and outboard edges at true scale


def _disc(x, y, r, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return x + r * np.cos(t), y + r * np.sin(t)


def _wb(v, digits=4):
    """A flux in Wb to `digits` significant figures, as lettered."""
    return f"{v:.{digits - 1 - int(math.floor(math.log10(abs(v))))}f}"


def draw(data, previews=None, out=None):
    sys.path.insert(0, HERE)
    import style

    c = data["certificate"]
    X = np.asarray(c["X_line"]["points"], float)
    O = np.asarray(c["O_line"]["points"], float)
    a = np.asarray(c["axis_point"], float)
    ds = data["design_section"]
    st = Straight(ds["R_cos"], ds["Z_sin"])
    outer = max(("A", "B"), key=lambda k: c["curves"][k]["flux"])
    inner = "B" if outer == "A" else "A"
    cut = {}
    for cls in ("A", "B"):
        cv = data["curves"][cls]
        cut[cls] = split_polygon(np.asarray(cv["polygon"]), X, np.asarray(cv["heteroclinic"]), cv["gaps"])[1]

    fig = style.Figure(height=HEIGHT)
    k_n = K_S * STRETCH / 1000.0                  # page units per micrometre across the chain
    left, top, width, height = FRAME
    xc = left + 0.5 * width                       # n = 0
    # the window's view takes page coordinates (y negated, as a view's y runs up): it is there for its clip
    win = fig.view(FRAME, x=(left, left + width), y=(-(top + height), -top), clip=True)

    def page(pts):
        """Page coordinates in the window of section points: x = -n (outward to the left, as in the section), y = s
        downwards (s runs counter-clockwise, which is down the inboard side)."""
        q = st.coords(pts)
        return xc - k_n * q[:, 1] * 1e6, Y0 + K_S * q[:, 0] * 1e3

    def wline(px, py, **kw):
        win.line(px, -np.asarray(py), **kw)

    def near(pts, margin=12.0):
        return np.min(np.abs(st.coords(pts)[:, 0])) * 1e3 < S_HALF + margin

    def on_curve(cls, s_mm):
        """Page point of curve cls at arc length s (mm) of the reference curve."""
        best = None
        for arc in cut[cls]:
            q = st.coords(arc["pts"])
            k = int(np.argmin(np.abs(q[:, 0] * 1e3 - s_mm)))
            d = abs(q[k, 0] * 1e3 - s_mm)
            if best is None or d < best[0]:
                best = (d, xc - k_n * q[k, 1] * 1e6, Y0 + K_S * q[k, 0] * 1e3)
        return best[1], best[2]

    # ---- Z = 0 across the window and the section, beneath everything
    fig.line([10.0, 770.0], [Y0, Y0], weight="fine", dash="dashdot")
    fig.text(776.0, Y0, "$Z=0$", anchor="w", gap=0)

    # ---- the islands: between the outer and the inner curve from one X point to the next, hatched
    by_gap = {cls: {} for cls in ("A", "B")}
    for cls in ("A", "B"):
        for arc in cut[cls]:
            by_gap[cls].setdefault(frozenset(data["curves"][cls]["gaps"][arc["gap"]]), []).append(arc)
    for key, arcs_o in by_gap[outer].items():
        po = np.vstack([ar["pts"] for ar in arcs_o])
        if not near(po):
            continue
        pi = np.vstack([ar["pts"] for ar in by_gap[inner][key]])
        px, py = page(np.vstack([po, pi[::-1]]))
        win.fill(px, -py, tone="crosshatch")
    # ---- the two curves: unstable arcs solid, stable arcs dashed. First with a channel of paper either side, so
    # that the tone stops short of them and the dashes stand clear, then again plain, so that where the curves
    # cross at the X points neither is cut.
    for halo in (True, False):
        for cls in (outer, inner):
            for arc in cut[cls]:
                if near(arc["pts"]):
                    px, py = page(arc["pts"])
                    wline(px, py, weight="heavy", dash="solid" if arc["kind"] == "u" else "dashed", halo=halo)
    # ---- O points (dots, on a little paper cleared in the tone), and the heteroclinic points on Z = 0 (circles).
    # The X points are drawn with their branches, where the curves cross, so they take no mark here.
    qo = st.coords(O)
    xO, yO = page(O[np.abs(qo[:, 0]) * 1e3 < S_HALF])
    for x, y in zip(xO, yO):
        fig.fill(*_disc(x, y, 7.0), tone="paper")
    fig.mark(xO, yO, kind="dot")
    for cls in (outer, inner):
        hx, hy = page(np.asarray(data["curves"][cls]["heteroclinic"][0])[None])
        fig.mark(hx, hy, kind="circle")
    fig.line([left, left + width, left + width, left], [top, top, top + height, top + height], weight="medium",
             closed=True)

    # ---- the section at true scale, on the same row of Z = 0
    # its inboard and outboard edges at SECTION_X in page units, which leaves the end of Z = 0 and its lettering
    # on clear paper beyond the outboard edge
    Co = np.asarray(data["curves"][outer]["polygon"])
    R_in, R_out = float(Co[:, 0].min()), float(Co[:, 0].max())
    k_sec = (SECTION_X[1] - SECTION_X[0]) / (R_out - R_in)
    zh = float(np.max(np.abs(Co[:, 1]))) + 0.016
    sec = fig.view((SECTION_X[0], Y0 - k_sec * zh, SECTION_X[1] - SECTION_X[0], 2 * k_sec * zh), x=(R_in, R_out),
                   y=(-zh, zh), equal=True)
    sec.line(Co[:, 0], Co[:, 1], weight="fine", closed=True)
    sec.mark(X[:, 0], X[:, 1], kind="cross")
    sec.mark(O[:, 0], O[:, 1], kind="dot")
    sec.mark([a[0]], [a[1]], kind="dot")
    sw = (st.S[:-1] + 0.5 * st.perimeter) % st.perimeter - 0.5 * st.perimeter
    sel = np.abs(sw) * 1e3 <= S_HALF
    o = np.argsort(sw[sel])
    G = st.G[sel][o]
    sec.line(G[:, 0], G[:, 1], weight="heavy")
    ex, ey = sec.page(G[[0, -1], 0], G[[0, -1], 1])
    fig.line([ex[0], left + width], [ey[0], top], weight="fine")
    fig.line([ex[1], left + width], [ey[1], top + height], weight="fine")
    axp = sec.page([a[0]], [a[1]])
    fig.leader(float(axp[0][0]) + 30, Y0 + 150, "magnetic axis", float(axp[0][0]), float(axp[1][0]) + 4)

    # ---- lettering in the window
    lo, hi = sorted([c["curves"][k]["flux"] for k in ("A", "B")])
    req = data["request_flux_Wb"]
    tx, ty = on_curve(outer, 15.5)
    fig.leader(tx - 19, ty, rf"$\Psi={_wb(hi)}$ Wb", tx, ty)
    tx, ty = on_curve(inner, -15.5)
    fig.leader(tx + 17, ty, rf"$\Psi={_wb(lo)}$ Wb", tx, ty)
    tx, ty = on_curve(outer, -5.5)
    fig.leader(tx - 52, ty - 16, "unstable", tx, ty)
    tx, ty = on_curve(outer, 5.5)
    fig.leader(tx - 52, ty + 16, "stable", tx, ty)
    qx = st.coords(X)
    k7 = int(np.argmin(np.where(qx[:, 0] > 0, np.abs(qx[:, 0]), np.inf)))
    xX, yX = page(X[k7][None])
    fig.leader(xX[0] + 96, yX[0], "X point", xX[0] + 4, yX[0])
    # the O point of the island below, inside the window
    below = np.where((qo[:, 0] > qx[k7, 0]) & (qo[:, 0] * 1e3 < S_HALF), qo[:, 0], np.inf)
    xo, yo = page(O[int(np.argmin(below))][None])
    fig.leader(xo[0] + 96, yo[0] + 14, "O point", xo[0] + 5, yo[0] + 1)
    # the requested flux: the line n = 0, marked at both ends of the window as a cut is
    for y_a, y_b in ((top - 30, top - 2), (top + height + 30, top + height + 2)):
        fig.line([xc, xc], [y_a, y_b], weight="medium", arrow="end")
    fig.text(xc + 10, top - 19, rf"requested $\Psi={_wb(req)}$ Wb", anchor="w")
    # scale: 10 mm along the chain, 10 micrometres across
    bx, by = left + 76, top + 30 + 10 * K_S
    fig.line([bx, bx, bx + 10 * k_n], [by - 10 * K_S, by, by], weight="medium")
    fig.text(bx - 6, by - 5 * K_S, "10 mm", anchor="e", size="small")
    fig.text(bx + 5 * k_n, by + 4, r"10 $\mu$m", anchor="n", size="small")
    # the conclusion
    fig.text(680.0, 92.0, "no flux surface encloses a flux", anchor="c")
    fig.text(680.0, 120.0, "between the two curves", anchor="c")
    out = out or os.path.join(HERE, "island-band")
    return fig.save(out, previews=previews)


# ---------------------------------------------------------------------------------------------- numbers

def numbers(data):
    """Every number lettered in the figure, and the ones a caption needs, with what each is and where it came from."""
    c = data["certificate"]
    cv = c["curves"]
    outer = max(("A", "B"), key=lambda k: cv[k]["flux"])
    inner = "B" if outer == "A" else "A"
    lo, hi = c["interval"]
    req = data["request_flux_Wb"]
    p, q = c["chain"]["p"], c["chain"]["q"]
    nfp = data["nfp"]
    st = Straight(data["design_section"]["R_cos"], data["design_section"]["Z_sin"])
    X = np.asarray(c["X_line"]["points"], float)
    O = np.asarray(c["O_line"]["points"], float)
    qx, qo = st.coords(X), st.coords(O)
    h = {k: st.coords(np.asarray(data["curves"][k]["heteroclinic"][0])[None])[0] for k in ("A", "B")}
    near_x = np.sort(qx[np.abs(qx[:, 0]) * 1e3 < S_HALF + 10, 0])
    ck = data.get("checks", {})
    sol = "torux.solve: Answer.evidence['certificate']"
    return {
        "figure": "island-band: figure 4 of method/README.md, the island band of QUASR 0219457 in the symmetry plane "
                  "phi = 0, the chain straightened on the left and the whole section at true scale on the right",
        "example": {"value": "examples/quasr_0219457.json",
                    "what": f"QUASR design 0219457, {nfp} field periods, stellarator symmetric. torux.load takes its "
                            "request as the flux its outermost stored surface encloses (QUASR's definition)"},
        "request": {"flux_Wb": req, "call": data["call"], "status": data["status"], "wall_s": data["wall_s"],
                    "torux": data["torux"], "printed": data["printed"],
                    "what": "the run whose certificate is drawn, and what torux printed for it (the time is this "
                            "machine's, docs/usage.md shows 101 s on another)"},
        "lettered": {"outer_curve": f"{_wb(hi)} Wb", "inner_curve": f"{_wb(lo)} Wb", "requested": f"{_wb(req)} Wb",
                     "scale_along": "10 mm", "scale_across": "10 micrometres",
                     "what": "the numbers lettered in the figure, to 4 significant figures: the fluxes of the outer "
                             "and the inner curve (at 4 figures these are also the ends of the certified interval: "
                             "the curves' own fluxes differ from those ends only by the error terms, 7e-8 Wb), the "
                             "requested flux, and the two scale bars of the straightened window"},
        "interval": {"F_lo_Wb": lo, "F_hi_Wb": hi, "width_Wb": hi - lo, "width_over_request": (hi - lo) / req,
                     "request_position": (req - lo) / (hi - lo), "margin_rel": c.get("margin_rel"),
                     "what": "the certificate: no flux surface around the axis encloses a flux in (F_lo, F_hi). "
                             "request_position is where the request falls in it, 0 at F_lo and 1 at F_hi",
                     "from": sol + "['interval']"},
        "curves": {
            "outer": dict(flux_Wb=cv[outer]["flux"], tube_Wb=cv[outer]["tube"], quad_err_Wb=cv[outer]["quad_err"],
                          F_low_Wb=cv[outer]["F_low"], vertices=cv[outer]["n_vertices"],
                          error_max_m=cv[outer]["eps_max"], error_median_m=cv[outer]["eps_median"]),
            "inner": dict(flux_Wb=cv[inner]["flux"], tube_Wb=cv[inner]["tube"], quad_err_Wb=cv[inner]["quad_err"],
                          F_up_Wb=cv[inner]["F_up"], vertices=cv[inner]["n_vertices"],
                          error_max_m=cv[inner]["eps_max"], error_median_m=cv[inner]["eps_median"]),
            "what": "the certificate's two closed curves as drawn: the certified polygons, their loop integrals of A "
                    "(flux), the flux of the error tube and the quadrature error that the interval subtracts "
                    "(F_hi = outer flux - tube - quad_err, F_lo = inner flux + tube + quad_err), their vertex counts "
                    "and the distance within which the true manifold curves lie (largest and median over the "
                    "segments). The outer curve is the certificate's class " + outer + ", the inner its class " + inner,
            "from": sol + "['curves'], the polygons from island_band_certificate_manifold(keep_polygons=True) in the "
                    "same run"},
        "chain": {"iota": nfp * p / q, "p_q_per_field_period": [p, q], "islands": q, "nfp": nfp,
                  "X_residue": c["X_line"]["residue"], "O_residue": c["O_line"]["residue"],
                  "X_multiplier": c["X_line"]["lam"], "X_point_error_m": c["X_line"]["e_X"],
                  "what": "the island chain: rotational transform, rotation per field period, islands in a section, "
                          "Greene's residues of its X and O lines, the X orbit's multiplier per q periods and the "
                          "error bound of its points",
                  "from": sol + "['chain'], ['X_line'], ['O_line']"},
        "plane": {"phi": c["phi_s"], "axis_R_m": c["axis_point"][0], "axis_Z_m": c["axis_point"][1],
                  "what": "the symmetry plane the certificate works in, and the magnetic axis's point in it (the dot)",
                  "from": sol + "['phi_s'], ['axis_point']"},
        "heteroclinic_points": {"outer_R_m": cv[outer]["crossing"]["h"][0], "inner_R_m": cv[inner]["crossing"]["h"][0],
                                "Z_m": 0.0,
                                "what": "the two heteroclinic points on Z = 0 (open circles): where an unstable branch "
                                        "crosses the symmetry line, which puts it on the mirrored stable branch",
                                "from": sol + "['curves'][...]['crossing']['h']"},
        "window": {"stretch": STRETCH, "half_length_mm": S_HALF,
                   "page_units_per_mm_along": K_S, "page_units_per_micrometre_across": K_S * STRETCH / 1000.0,
                   "island_width_at_Z0_micrometres": float((h[outer][1] - h[inner][1]) * 1e6),
                   "X_points_s_mm": [float(v) for v in near_x * 1e3],
                   "X_O_offset_max_micrometres": float(max(np.max(np.abs(qx[:, 1])), np.max(np.abs(qo[:, 1]))) * 1e6),
                   "what": "the straightened window: s, arc length along the reference curve from Z = 0 (downwards), "
                           "is drawn page_units_per_mm_along, and n, the distance from the reference curve, "
                           "stretch times larger. The island on Z = 0 is island_width_at_Z0 wide (between its "
                           "heteroclinic points) "
                           "and runs between the X points at the s given. Every X and O point of the chain lies within "
                           "X_O_offset_max of the reference curve"},
        "reference_curve": {"flux_Wb": data["design_section"]["flux_Wb"],
                            "rel_diff_from_request": abs(data["design_section"]["flux_Wb"] - req) / req,
                            "fourier_terms": len(data["design_section"]["R_cos"]),
                            "what": "the curve n = 0 of the window, marked by the two arrows: the planar section at "
                                    "phi = 0 of the file's outermost surface, the surface that defines the "
                                    "request. It encloses the requested flux to rel_diff. It is a curve with the "
                                    "requested flux, not a flux surface",
                            "from": "torux.design.planar_section and section_flux on the loaded source's design "
                                    "surface"},
        "checks": {"values": ck,
                   "what": "for each curve: the certificate's own quadrature on the kept polygon again (flux_again), "
                           "on the polygon as drawn, cut at the feet of the X and heteroclinic points (flux_drawn), "
                           "and the independent reference evaluator on it (flux_drawn_reference, torux.barrier."
                           "reference_flux), the largest distance of an X point and of a heteroclinic point from the "
                           "polygon, and the interval computed again from the drawn polygons with the record's error "
                           "terms, printed as torux prints it"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compute", action="store_true", help="solve again and keep the certificate's curves (minutes)")
    ap.add_argument("--previews", default=None, help="directory for PNG renders at 838 and 324 px")
    args = ap.parse_args()
    if args.compute or not os.path.exists(DATA):
        t0 = time.time()
        data = run_request()
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        with open(DATA, "w") as fh:
            json.dump(data, fh)
        print(f"kept {DATA} ({os.path.getsize(DATA) / 1024:.0f} KB) after {time.time() - t0:.0f} s")
    with open(DATA) as fh:
        data = json.load(fh)
    draw(data, previews=args.previews)
    with open(NUMBERS, "w") as fh:
        json.dump(numbers(data), fh, indent=1)


if __name__ == "__main__":
    main()
