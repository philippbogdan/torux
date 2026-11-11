"""One source, one call: axis, flux-surface family, chains crossed, the surface at the request.

analyse(source, flux=None) follows the family of tori from the magnetic axis outward,
crossing low-order resonances by fixed-iota jumps where the continuation stalls, refines
the torus at the requested flux to the arithmetic limit and applies the resonance guard,
then (optionally) continues past the request to measure how far the nested surfaces go.
It returns a plain dict (JSON-serialisable) with the evidence of every step.
"""
from __future__ import annotations

import dataclasses
import math
import os as _os
import time
import traceback

import numpy as np

from .field import FieldEngine
from . import axis as AX
from . import budget as _budget
from .envfail import environment_failure
from .family import Continuation, refine_to_limit, FULL_CAPS
from .chains import nearest_rational_ahead, analyse_chain, jump_over
from .guard import resonance_guard
from .torus import TorusState


# A verified surface: refined to the arithmetic limit.  The off-grid |B.n|/|B| of a refined invariant torus
# is 1-3e-15 on almost every torus (4,282 of 4,605 QUASR designs at or below 3e-15); a torus that stops
# above 1e-14 whatever its resolution is not invariant at that level: of 59 such refinement plateaus on QUASR
# designs, 55 had their transform within 1e-6 of a rational (rational-target designs: a thin chain at the
# request), the rest stopped at the resolution cap.  Those are near-invariant tori, not surfaces: they go
# to the resonant path (island band or the tori beyond the chain) or to more resolution.
BN_SURFACE = 1e-14

# seconds the certificates (island band, edge barrier) keep of a request's budget: the stages before them stop
# that much earlier (at most a fifth / quarter of what is left)
CERT_RESERVE = 300.0
# the resonant path's reserve for the edge barrier and the fallback continuation with its certificate
RESONANT_RESERVE = 900.0

# The stage policy of the resonant path: the order of its stages and the refusal of stages that cannot finish
# or do not apply.  None of these changes what a certificate proves or an acceptance threshold; they decide
# which stages run, in which order, and when a request is left undetermined.
POLICY = dict(
    seeds_first=True,     # the seeds' base polish (no tracing) before the proxy band; the tracing after it
    seeds_trace=False,    # the classes (request_class) the seeds' tracing stage (after the proxy band) runs for
    lazy_inner=True,      # inner design tori solved only when the bracket / continuation need them
    late="all",           # the classes the late stages (bracket, band, edge, continuation) run for: every class here
                          # (an API call gets the whole solver); () leaves those classes undetermined
    late_bn=1e-4,         # design B.n separating the 'near' and 'far' classes
    continuation=True,    # the fallback continuation (analyse) after the bracket's certificates
    cont_probe_s=60.0,    # seconds of the early continuation from the inner design tori (0: none)
    cont_probe_share=0.25,  # ... or this share of the request's budget when larger (600 s: 150 s)
    cont_probe_first=False,  # the early continuation before the seeds' base polish (else after it)
    cont_probe=("newton_failed", "near", "refined"),   # the classes it runs for
    proxy=("newton_failed", "near", "far"),            # the classes the proxy band runs for
    tail_s=None,          # seconds kept before the deadline when no late stage runs (None: RESONANT_RESERVE's rule)
    # -- further switches, off by default; none changes what a certificate proves or an acceptance threshold.
    proxy_first=False,    # the proxy band before the early continuation (default: after it)
    rho_chains=False,     # the chains nearest the rotation of the seeds' base-polish circle first (its circle at the
                          # requested flux stalls on the chain the request sits in: QUASR 0017820, rho -0.388907, 7/18)
    traced_lines=False,   # a chain's symmetric closed lines from traced sign changes (torux.chainfind) when Newton from
                          # the design torus's curves finds none (q <= 24, at most 15 % of the time left / 120 s)
    n_chains=3,           # chains tried by the proxy band
    annulus="standard",    # the proxy annulus: "standard" (torux.islandband.proxy_annulus, margins 1.5, 4.0) or "sym" (the
                          # symmetric fit, proxy_annulus_sym, margins 1.5, 4.0, then the standard annulus when a branch
                          # leaves both: the standard fit left the inner surface on the axis for low-order chains of
                          # failed design tori: 'axis point not found')
    inverse_axis=False,   # designs whose period-one line is inverse hyperbolic (axis_kind): that line is the bands' axis
                          # point (torux.islandband.axis_point inverse=True; by default every band there is refused)
    edge_triage=False,    # lines from inside the design surface lost within a few hundred periods: an edge-type request
                          # (beyond the confined region); the proxy band is skipped and the early continuation (whose
                          # edge barrier settles such requests) gets the time to the late stages.  Stage choice only.
    edge_triage_s=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9),   # its start points: fractions of the design torus's outboard radius
    edge_triage_periods=400,
    surrogate_steps=False,  # surrogate node traces start at 24, 48 or 96 steps per period as a 64-node sample needs, and a
                            # build the sample's rate projects far beyond the deadline is refused at once
                            # (torux.islandband._first_steps)
    card_traces=False,    # surrogate node traces by torux.cardtrace on a card when one is present (validation points stay on
                          # the CPU tracer; a sample of every batch is re-traced there).
    force_resonant=False,  # soundness tests only: the resonant path on every request (known surfaces: no certificate
                           # may hold the request)
    skip_surface_stages=False,  # soundness tests only: no seeds, no early continuation (the certificates are tried)
)
# TORUX_POLICY='{"late": ["newton_failed", "near"]}' (JSON) overrides entries
if _os.environ.get("TORUX_POLICY"):
    import json as _json
    POLICY.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in _json.loads(_os.environ["TORUX_POLICY"]).items()})


def torus_record(t: TorusState, full=False):
    d = dict(flux=float(t.flux), iota=float(t.iota), M=int(t.M), N=int(t.N), bn_off=float(t.bn_off),
             res_off=float(t.res_off), res=float(t.res), its=int(t.its), converged=bool(t.converged))
    if full:
        d["rc"] = np.asarray(t.rc).tolist()
        d["zs"] = np.asarray(t.zs).tolist()
    return d


def _bbox_pairs(P, Q, margin=1e-9):
    """Index pairs (i, j), j >= i + 2, of the chords P[i] -> Q[i] whose bounding boxes overlap (each box widened by
    margin x the curve's size): a proper crossing of two chords needs overlapping boxes, so testing only these pairs
    finds every crossing the all-pairs test finds (fast_family 'topofast').  Sweep over the boxes sorted by x."""
    lo = np.minimum(P, Q)
    hi = np.maximum(P, Q)
    eps = margin * max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 1e-300)
    lo = lo - eps
    hi = hi + eps
    n = len(P)
    order = np.argsort(lo[:, 0], kind="stable")
    los, his = lo[order], hi[order]
    end = np.searchsorted(los[:, 0], his[:, 0], side="right")
    cnt = np.maximum(end - np.arange(n) - 1, 0)
    a = np.repeat(np.arange(n), cnt)
    b = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + a + 1
    ok = (los[b, 1] <= his[a, 1]) & (los[a, 1] <= his[b, 1])
    ia, ib = order[a[ok]], order[b[ok]]
    i, j = np.minimum(ia, ib), np.maximum(ia, ib)
    keep = j >= i + 2
    i, j = i[keep], j[keep]
    srt = np.lexsort((j, i))
    return i[srt], j[srt]


def torus_topology(torus, axis=None, n=512, nphi=4):
    """Embedded-torus checks on cross-sections at nphi planes over a field period: each section a simple
    closed curve (no two non-adjacent chords cross) and, given the magnetic axis, the axis point inside it
    (a torus of the axis's family, not an island torus or another axis's)."""
    out = dict(simple=True, encloses_axis=None)
    th = 2 * np.pi * np.arange(n) / n
    inside_all = True
    for k in range(nphi):
        ph = torus.phi0 + k * (2 * np.pi / max(torus.nfp, 1e-300)) / nphi
        R, Z = (v[:, 0] for v in torus.geometry_grid(th, np.array([ph])))
        P = np.stack([R, Z], -1)
        Q = np.roll(P, -1, axis=0)
        d = Q - P
        # segment i vs j crossing: orientation tests (vectorised over all pairs)
        def orient(a, b, c):
            return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
        from . import fast_family as _so2
        if _so2.on("topofast"):
            # (fast_family 'topofast') the same orientation tests on the chord pairs whose bounding boxes overlap
            i, j = _bbox_pairs(P, Q)
            A, B, C, D = P[i], Q[i], P[j], Q[j]
            o1, o2 = orient(A, B, C), orient(A, B, D)
            o3, o4 = orient(C, D, A), orient(C, D, B)
            cross = (o1 * o2 < 0) & (o3 * o4 < 0)
            i, j = i[cross], j[cross]
        else:
            A, B = P[:, None, :], Q[:, None, :]
            C, D = P[None, :, :], Q[None, :, :]
            o1, o2 = orient(A, B, C), orient(A, B, D)
            o3, o4 = orient(C, D, A), orient(C, D, B)
            cross = (o1 * o2 < 0) & (o3 * o4 < 0)
            i, j = np.nonzero(np.triu(cross, 2))
        bad = [(a, b) for a, b in zip(i, j) if not (a == 0 and b == n - 1)]
        if bad:
            out["simple"] = False
        if axis is not None:
            Ra, Za = axis.at(np.array([ph]))
            x, y = float(Ra[0]), float(Za[0])
            c = (P[:, 1] > y) != (Q[:, 1] > y)
            xin = P[:, 0] + (y - P[:, 1]) * (Q[:, 0] - P[:, 0]) / np.where(Q[:, 1] != P[:, 1], Q[:, 1] - P[:, 1], 1e-300)
            inside = (np.sum(c & (x < xin)) % 2) == 1
            inside_all = inside_all and bool(inside)
    if axis is not None:
        out["encloses_axis"] = bool(inside_all)
    out["ok"] = bool(out["simple"] and (axis is None or out["encloses_axis"]))
    return out


# after a failed quadrature check, a torus nearer a coil than NEAR_COIL x dmin gets an engine with dmin half its
# distance (torux.analysis.near_coil_retry): where the check's 1e-8 is reached depends on the coils' shapes (CoilStellaration
# D66xdCGD...: 2.7e-6 at 0.49 dmin; QUASR 954: 4.6e-9 at 0.25 dmin), so every torus within dmin is tried
NEAR_COIL = 1.0


def near_coil_distance(source, torus, th, ph, X, h, within):
    """The distance from the torus to the source's Fourier coils with current where it may be below `within`, else
    inf.  X (len(th), len(ph), 3): the torus on the grid th x ph, spacing h.  A coil whose distance to the grid, less
    h / 2, is below `within` is measured again by zooming in on its four nearest grid points (17 x 17 patches, each a
    quarter the size of the last, four times).  An estimate, not a bound: what it sets is checked by the quadrature
    check."""
    from scipy.spatial import cKDTree
    from .field.engine import min_sample_distance
    tree = None
    close = []
    for p in source.parts:
        if p.kind != "fourier":
            continue
        xs, _ = p.points(np.arange(4096) / 4096)
        for i in range(p.n):
            if p.current[i] == 0.0:
                continue
            if tree is None:
                tree = cKDTree(X.reshape(-1, 3))
            vmax = float(np.max(np.linalg.norm(np.roll(xs[i], -1, 0) - xs[i], axis=-1))) * len(xs[i])
            if min_sample_distance(tree, xs[i], vmax) - 0.5 * h >= within:
                continue
            dd, jj = tree.query(xs[i], k=1)
            js = []
            for j in np.asarray(jj)[np.argsort(dd)]:
                if int(j) not in js:
                    js.append(int(j))
                if len(js) == 4:
                    break
            close.append((xs[i], js))
    if not close:
        return np.inf
    dth, dph = th[1] - th[0], ph[1] - ph[0]
    off = np.linspace(-1.0, 1.0, 17)
    d_close = np.inf
    for xs_i, js in close:
        ctree = cKDTree(xs_i)
        for j in js:
            a, b = np.unravel_index(j, X.shape[:2])
            tc, pc, wt, wp = th[a], ph[b], 2 * dth, 2 * dph
            for _ in range(4):
                tt, pp = tc + wt * off, pc + wp * off
                Rp, Zp = torus.geometry_grid(tt, pp)
                PHp = np.broadcast_to(pp[None, :], Rp.shape)
                P = np.stack([Rp * np.cos(PHp), Rp * np.sin(PHp), Zp], -1).reshape(-1, 3)
                dp = ctree.query(P, k=1)[0]
                k = int(np.argmin(dp))
                ka, kb = np.unravel_index(k, Rp.shape)
                tc, pc, wt, wp = tt[ka], pp[kb], 0.25 * wt, 0.25 * wp
            d_close = min(d_close, float(dp[k]))
    return d_close if d_close < within else np.inf


def lean_continuation(cont, source, torus, dmin, ntheta=96, nphi_fp=48, near_coil=False):
    """A Continuation for refining ``torus``, on an engine whose per-coil node counts come from each
    coil's distance to that torus (half of it may be approached; strip 0.8) instead of to the broad
    region the family explores: B, A, grad B on and near the torus stay at rounding level against the
    double-double reference with 2-10x fewer nodes.  Returns cont
    itself when the lean engine would not be smaller.

    near_coil=True (after a failed quadrature check, torux.analysis.near_coil_retry): a torus closer than NEAR_COIL x dmin
    to a coil gets an engine with dmin half that distance, used even when larger than the family's (else None).  An
    engine never targets less than its dmin, and below about half of it the check (quad_err_half <= 1e-8) refuses the
    torus: CoilStellaration D66xdCGD..., the outer torus of its range 2.5 cm from a coil with dmin 5 cm,
    quad_err_half 2.7e-6; 4e-16 at dmin 1.1 cm."""
    nfp = int(torus.nfp)
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, nphi_fp * max(1, nfp), endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    R, Z = torus.geometry_grid(th, ph)
    X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1)
    hth = np.max(np.linalg.norm(np.roll(X, -1, 0) - X, axis=-1))
    hph = np.max(np.linalg.norm(np.roll(X, -1, 1) - X, axis=-1))
    close = False
    if near_coil:
        d_tc = near_coil_distance(source, torus, th, ph, X, math.hypot(hth, hph), NEAR_COIL * dmin)
        close = bool(np.isfinite(d_tc) and 0 < d_tc < NEAR_COIL * dmin)
        if not close:
            return None
        dmin = 0.5 * d_tc
    old = cont.engine
    nm = 8 if getattr(old, "cpu_kernels", False) else 64
    from . import fast_design as _so
    lean = None
    if _so.on("nodes"):
        # (fast_design 'nodes') the strip model's counts without building its engine (only lean2 below is built)
        from .field.engine import model_nodes
        N_model = model_nodes(source, dmin, X.reshape(-1, 3), region_h=math.hypot(hth, hph), node_multiple=nm)
    if lean is None and not (_so.on("nodes") and len(N_model)):
        lean = FieldEngine(source, dmin=dmin, region_pts=X.reshape(-1, 3), region_h=math.hypot(hth, hph),
                           node_multiple=nm)
    if _so.on("nodes") and len(N_model):
        # (fast_design 'nodes') each coil's node count from its own measured convergence on this torus
        from .field.engine import measured_nodes
        # each coil's closest approach on a 4x denser patch of the torus around the 4 sample points nearest to it
        # (geometry only): the rate is scaled to the distance less half the sample spacing near the coil
        from scipy.spatial import cKDTree
        Xf = X.reshape(-1, 3)
        tree_c = cKDTree(Xf)
        dth, dph = th[1] - th[0], ph[1] - ph[0]
        off = np.arange(-8, 9) / 4.0
        patches = []
        for p in source.parts:
            if p.kind != "fourier":
                continue
            xs_p, _ = p.points(np.arange(4096) / 4096)
            for i in range(p.n):
                if p.current[i] == 0.0:
                    continue
                dd, jj = tree_c.query(xs_p[i][::16], k=1)
                best = []
                for j in np.asarray(jj)[np.argsort(dd)]:
                    if j not in best:
                        best.append(int(j))
                    if len(best) == 4:
                        break
                for j in best:
                    a, b_ = np.unravel_index(j, TH.shape)
                    Rp_, Zp_ = torus.geometry_grid(th[a] + dth * off, ph[b_] + dph * off)
                    PHp = np.broadcast_to((ph[b_] + dph * off)[None, :], Rp_.shape)
                    patches.append(np.stack([Rp_ * np.cos(PHp), Rp_ * np.sin(PHp), Zp_], -1).reshape(-1, 3))
        Xd = np.concatenate([Xf] + patches)
        Nm, ninfo = measured_nodes(source, N_model, Xd, region_h=0.25 * math.hypot(hth, hph))
        lean = FieldEngine(source, dmin=dmin, region_pts=X.reshape(-1, 3), region_h=math.hypot(hth, hph),
                           node_multiple=nm, nodes_per_coil=Nm, nodes_np=True)
        lean.node_info = dict(model_nodes=int(sum(N_model)), per_coil=ninfo)
    if lean.info.n_nodes >= old.info.n_nodes and not close:
        return cont
    if close:
        lean.near_coil = dict(distance=float(d_tc), dmin=float(dmin))
    c2 = Continuation(lean, cont.nfp, M0=cont.M0, N0=cont.N0, bn_cont=cont.bn_cont, bn_accept=cont.bn_accept,
                      tol=cont.tol, Mmax=cont.Mmax, Nmax=cont.Nmax, verbose=cont.verbose, phi0=cont.phi0)
    c2.predict_state = getattr(cont, "predict_state", None)
    return c2


def near_coil_retry(cont, source, best, dmin, tol, deadline=None, caps=(320, 320)):
    """After a refined torus failed the quadrature check (quad_err_half > 1e-8): when it lies nearer a coil than
    NEAR_COIL x dmin, the refinement again on an engine with dmin half that distance (lean_continuation near_coil).
    Returns (rc2, best2, quad2, info) when the new torus passes the check, else None (the caller keeps its torus and
    refusal).  Acceptance is the caller's, on the new engine (B.n, guard, topology).  Tori that pass the check never
    come here."""
    if deadline is not None and time.time() > deadline - 10:
        return None
    t0 = time.time()
    try:
        rc2 = lean_continuation(cont, source, best, dmin, near_coil=True)
        if rc2 is None:
            return None
        from . import fast_family as _so2
        if _so2.on("cpredict"):
            rc2.predict_state = None     # as the callers' refinements
        # the torus's B.n measured again on the near-coil engine: refine_to_limit keeps a finite bn_off, and the one
        # the torus carries is the other engine's (a torus accepted on that stale value can be refused by the
        # independent reference at 1e-13 .. 4e-13)
        start = dataclasses.replace(best, rc=np.array(best.rc), zs=np.array(best.zs), bn_off=np.nan, res_off=np.nan,
                                    info=dict(best.info))
        best2 = refine_to_limit(rc2, start, best.flux, target=tol, deadline=deadline, Mcap=caps[0], Ncap=caps[1])
        if not np.isfinite(best2.bn_off):
            return None
        q2 = rc2.problem(best2.M, best2.N).quadrature_check(best2.rc, best2.zs)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        return None
    info = dict(rc2.engine.near_coil, nodes=int(rc2.engine.info.n_nodes), quad=float(q2), bn_off=float(best2.bn_off),
                wall=time.time() - t0)
    return (rc2, best2, q2, info) if q2 <= 1e-8 else None


# The range's inward fallback.
# When none of the standard candidates -- the family's three outermost converged tori, then the near-coil retries --
# is accepted, inner tori are tried with the same
# acceptance (off-grid B.n at the arithmetic limit, resonance guard, quadrature check; the independent reference checks
# every answer afterwards as before).  Order (range_inner_candidates): with `secure`, the outermost family torus inward
# of the loop's at or below max(fracs) of the outermost candidate's flux (refined as it is: seconds); then at each
# fraction of `fracs`, outermost first, the family's torus there (flux within [near f, f] x that flux) or else a torus
# continued to that flux from the family's two tori around it (a family seldom has one near half the outer flux: its
# steps grow 1.5x; a new torus costs about a minute).  reserve_s / reserve_share: seconds of the candidate loop (at most
# that share of what is left when it starts) kept for the inner candidates when there is one -- the standard loop and
# its near-coil retries end that much earlier -- and, with `resume`, given back afterwards to the candidates the
# loop's deadline cut (range_resume): after a secure inner torus, before the fractions' tori; after a loop candidate
# accepted inward of a cut one, to the cut one.  A resumed or tighter torus only ever replaces an accepted one outward.
RANGE_INNER = dict(
    on=True,
    fracs=(0.5, 0.25),
    near=0.8,
    new_torus=True,
    reserve_s=60.0,      # the reserve: min(60 s, 0.4 x the time left), kept when an inner candidate exists
    reserve_share=0.4,
    resume=True,         # with a reserve: the time the inner candidates leave goes back to the standard candidates
    secure=True,         # first the outermost family torus inward of the loop's (refined as it is: seconds), then the
                         # reserve resumes the standard cut candidates, then the fractions' tori outward of it
    min_s=10.0,          # an inner candidate is not started with less time than this left
)
# TORUX_RANGE_INNER='{"on": true, "reserve_s": 60}' (JSON) overrides entries (comparisons)
if _os.environ.get("TORUX_RANGE_INNER"):
    import json as _json
    RANGE_INNER.update({k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in _json.loads(_os.environ["TORUX_RANGE_INNER"]).items()})


def range_inner_candidates(cands, tried, opts=None, floor=0.0):
    """The inward fallback's candidates (RANGE_INNER): with secure, first the outermost family torus at or below
    max(fracs) x the flux of the outermost candidate;
    then for each fraction f, outermost first, with F = f x that flux, the outermost family torus at or below F if its
    flux is at least near x F (kind 'family'); else, with new_torus, a torus at F continued from the family tori just
    below and above F (kind 'new': lo, hi, flux); else that family torus below F.  Tori in `tried` (entries with 'flux',
    as the standard loop writes them), repeats and candidates at or inward of `floor` (a flux) are left out."""
    o = dict(RANGE_INNER, **(opts or {}))
    if not cands:
        return []
    sgn = math.copysign(1.0, float(cands[0].flux))
    f_out = abs(float(cands[0].flux))
    used = {abs(float(t["flux"])) for t in tried if t.get("flux") is not None}
    fam = sorted(cands, key=lambda t: abs(float(t.flux)))
    out = []
    if o.get("secure") and o["fracs"]:
        below = [t for t in fam if abs(float(t.flux)) <= max(o["fracs"]) * f_out and abs(float(t.flux)) not in used
                 and abs(float(t.flux)) > floor]
        if below:
            out.append(dict(f=None, kind="family", torus=below[-1], secure=True))
            used.add(abs(float(below[-1].flux)))
    for f in o["fracs"]:
        if float(f) * f_out <= floor:
            continue
        F = float(f) * f_out
        below = [t for t in fam if abs(float(t.flux)) <= F]
        above = [t for t in fam if abs(float(t.flux)) > F]
        lo = below[-1] if below else None
        if lo is None:
            continue
        if abs(float(lo.flux)) >= float(o["near"]) * F or not (o["new_torus"] and above):
            if abs(float(lo.flux)) not in used and abs(float(lo.flux)) > floor:
                out.append(dict(f=float(f), kind="family", torus=lo))
                used.add(abs(float(lo.flux)))
        else:
            out.append(dict(f=float(f), kind="new", lo=lo, hi=above[0], flux=sgn * F))
    return out


def range_candidate(cont, source, cand, tori, dmin, tol, deadline=None, axis=None, label="inner"):
    """One inner candidate (range_inner_candidates) taken as the standard range loop takes its candidates: a lean
    engine for it, refine_to_limit, the resonance guard with the family's shear there, the quadrature check, and after a
    failed check the near-coil retry (near_coil_retry).  A 'new' torus is first solved at its flux from the linear
    interpolation in sqrt(flux) of the family tori around it, at the outer one's resolution, and kept only as a member
    of the family: converged with the family's B.n (bn_accept), its transform within the family's step rule of the
    torus below, and after refinement sections simple and around the axis (torus_topology).  Returns dict(ok, entry,
    and when refined: best, g, rec, rc); the entry and record carry dict(f, kind, wall) under `label`."""
    from . import fast_family as _so2
    t0 = time.time()
    fl = np.array([t.flux for t in tori])
    io = np.array([t.iota for t in tori])
    entry = {label: dict(f=cand["f"], kind=cand["kind"], **({"secure": True} if cand.get("secure") else {}))}
    if cand["kind"] == "family":
        T = cand["torus"]
        flux = T.flux
        rc_ = lean_continuation(cont, source, T, dmin)
        if _so2.on("cpredict"):
            rc_.predict_state = None
        j = int(np.argmin(np.abs(fl - T.flux)))
        lo, hi = max(0, j - 1), min(len(tori) - 1, j + 1)
        shear_phi = (io[hi] - io[lo]) / (fl[hi] - fl[lo]) if hi > lo else np.nan
        start = T
    else:
        lo_t, hi_t, flux = cand["lo"], cand["hi"], float(cand["flux"])
        s_lo, s_hi, s_new = (math.sqrt(abs(float(v))) for v in (lo_t.flux, hi_t.flux, flux))
        guess = cont.predict([lo_t, hi_t], s_new, math.copysign(1.0, flux))
        rc_ = lean_continuation(cont, source, guess, dmin)
        if _so2.on("cpredict"):
            rc_.predict_state = None
        shear_phi = (hi_t.iota - lo_t.iota) / (hi_t.flux - lo_t.flux)
        with _budget.scope(deadline):
            sol = rc_.solve(guess, flux, guess.M, guess.N, maxit=10)
            if not sol.converged and np.isfinite(sol.res) and not _budget.expired():
                # once more one ladder step up (as the family's plateau retry)
                M2, N2 = rc_.grow(sol)
                if (M2, N2) != (sol.M, sol.N):
                    sol2 = rc_.solve(sol, flux, M2, N2, maxit=10)
                    if sol2.converged:
                        sol = sol2
        slope = (hi_t.iota - lo_t.iota) / max(s_hi - s_lo, 1e-300)
        member = (sol.converged and np.isfinite(sol.bn_off) and sol.bn_off < cont.bn_accept
                  and abs(sol.iota - lo_t.iota) <= 10 * abs(slope) * (s_new - s_lo) + 1e-3)
        entry[label].update(newton=dict(converged=bool(sol.converged), bn_off=float(sol.bn_off), iota=float(sol.iota),
                                        M=int(sol.M), N=int(sol.N), wall=time.time() - t0))
        if not member:
            entry.update(flux=abs(flux), ok=False)
            entry[label]["wall"] = time.time() - t0
            return dict(ok=False, entry=entry)
        start = sol
    with _budget.scope(deadline):
        best = refine_to_limit(rc_, start, flux, target=tol, deadline=deadline)
    g = resonance_guard(rc_.engine, best, shear=shear_phi * 2 * np.pi)
    rec = torus_record(best, full=True)
    rec["quad_err_half"] = rc_.problem(best.M, best.N).quadrature_check(best.rc, best.zs)
    ok = bool(best.bn_off <= BN_SURFACE and g["ok"] and rec["quad_err_half"] <= 1e-8)
    if rec["quad_err_half"] > 1e-8 and (deadline is None or time.time() < deadline - 10):
        nc = near_coil_retry(cont, source, best, dmin, tol, deadline=deadline)
        nc_info = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(rec["quad_err_half"]))
        entry["near_coil"] = nc_info
        if nc is not None:
            rc_, best, q2 = nc[0], nc[1], nc[2]
            g = resonance_guard(rc_.engine, best, shear=shear_phi * 2 * np.pi)
            rec = torus_record(best, full=True)
            rec["quad_err_half"] = q2
            rec["near_coil"] = nc_info
            ok = bool(best.bn_off <= BN_SURFACE and g["ok"] and q2 <= 1e-8)
    if ok and cand["kind"] == "new" and axis is not None:
        topo = torus_topology(best, axis)
        rec["topology"] = topo
        ok = bool(topo["ok"])
    entry.update(flux=abs(float(best.flux)), bn_off=float(best.bn_off), M=int(best.M), N=int(best.N),
                 guard=bool(g["ok"]), quad=float(rec["quad_err_half"]), ok=ok, refine_stop=best.info.get("refine_stop"))
    entry[label]["wall"] = time.time() - t0
    rec[label] = dict(entry[label])
    return dict(ok=ok, entry=entry, best=best, g=g, rec=rec, rc=rc_)


def range_inner(cont, source, tori, cands, tried, dmin, tol, deadline=None, opts=None, axis=None, verbose=False,
                floor=0.0):
    """The inward fallback (RANGE_INNER): its candidates (range_inner_candidates) in turn while at least min_s seconds
    are left; the first accepted is returned (range_candidate's dict), else None.  Every attempt is appended to
    `tried`."""
    o = dict(RANGE_INNER, **(opts or {}))
    for cand in range_inner_candidates(cands, tried, o, floor=floor):
        if deadline is not None and time.time() > deadline - float(o["min_s"]):
            break
        try:
            r = range_candidate(cont, source, cand, tori, dmin, tol, deadline=deadline, axis=axis)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            tried.append(dict(inner=dict(f=cand["f"], kind=cand["kind"]), ok=False, error=repr(e)[:300],
                              **({"flux": abs(float(cand["torus"].flux))} if cand["kind"] == "family" else {})))
            continue
        tried.append(r["entry"])
        if verbose:
            print(f"  range inner f {cand['f']} {cand['kind']}: {r['entry']}", flush=True)
        if r["ok"]:
            return r
    return None


def range_resume(cont, source, cands, tori, tried, loop_states, quad_refused, dmin, tol, deadline=None, floor=0.0,
                 min_s=10.0):
    """(RANGE_INNER with a reserve, resume) What the reserve took from the standard loop, with the time left: the
    near-coil retries its end left out, the candidates its deadline cut (refined on from their best torus, on their
    engine: a level skipped as too long for the loop's deadline, or a Newton its budget scope stopped), the candidates
    it never reached -- outermost first, only those outward of `floor` (the flux of an accepted torus: the loop's own,
    when it accepted one inward of a cut candidate, or an inner one).  The same acceptance as the standard loop.
    Returns the first accepted as dict(best, g, rec), else None; every attempt goes to `tried` (resumed=...).
    loop_states: (torus, its continuation, refined torus, shear) of each candidate the loop took, in order."""
    n_loop = len(loop_states)
    retried = {e["flux"] for e in tried if "near_coil" in e and "resumed" not in e and "inner" not in e}
    todo = []
    for item in quad_refused:
        if abs(float(item[0].flux)) not in retried:
            todo.append(("near_coil", abs(float(item[0].flux)), item))
    for st in loop_states:
        b = st[2]
        if b.info.get("refine_stop") in ("deadline", "not_converged") and not (b.bn_off <= BN_SURFACE):
            todo.append(("continue", abs(float(st[0].flux)), st))
    for t in cands[n_loop:3]:
        todo.append(("candidate", abs(float(t.flux)), t))
    todo = sorted([x for x in todo if x[1] > floor], key=lambda x: -x[1])
    for kind, f_abs, data in todo:
        if deadline is not None and time.time() > deadline - min_s:
            break
        t0 = time.time()
        try:
            if kind == "candidate":
                r = range_candidate(cont, source, dict(f=None, kind="family", torus=data), tori, dmin, tol,
                                    deadline=deadline, label="resumed")
                tried.append(r["entry"])
                if r["ok"]:
                    return r
                continue
            if kind == "near_coil":
                outer_r, best_r, shear_r, q_before = data
                with _budget.scope(deadline):
                    nc = near_coil_retry(cont, source, best_r, dmin, tol, deadline=deadline)
                nc_info = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(q_before))
                e = dict(flux=f_abs, near_coil=nc_info, resumed=dict(kind=kind))
                tried.append(e)
                if nc is None:
                    continue
                rc2, b2, q2 = nc[0], nc[1], nc[2]
            else:
                outer_r, rc2, b_cut, shear_r = data
                with _budget.scope(deadline):
                    b2 = refine_to_limit(rc2, b_cut, outer_r.flux, target=tol, deadline=deadline)
                q2 = rc2.problem(b2.M, b2.N).quadrature_check(b2.rc, b2.zs)
                e = dict(flux=f_abs, resumed=dict(kind=kind))
                tried.append(e)
                nc_info = None
                if q2 > 1e-8 and (deadline is None or time.time() < deadline - 10):
                    with _budget.scope(deadline):
                        nc = near_coil_retry(cont, source, b2, dmin, tol, deadline=deadline)
                    nc_info = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(q2))
                    e["near_coil"] = nc_info
                    if nc is not None:
                        rc2, b2, q2 = nc[0], nc[1], nc[2]
            g2 = resonance_guard(rc2.engine, b2, shear=shear_r * 2 * np.pi)
            ok2 = bool(b2.bn_off <= BN_SURFACE and g2["ok"] and q2 <= 1e-8)
            e.update(bn_off=float(b2.bn_off), M=int(b2.M), N=int(b2.N), guard=bool(g2["ok"]), quad=float(q2), ok=ok2,
                     refine_stop=b2.info.get("refine_stop"))
            e["resumed"]["wall"] = time.time() - t0
            if ok2:
                rec2 = torus_record(b2, full=True)
                rec2["quad_err_half"] = q2
                if nc_info is not None:
                    rec2["near_coil"] = nc_info
                rec2["resumed"] = dict(e["resumed"])
                return dict(ok=True, best=b2, g=g2, rec=rec2, entry=e)
        except Exception as ex:  # noqa: BLE001
            if environment_failure(ex):
                raise
            tried.append(dict(flux=f_abs, resumed=dict(kind=kind), ok=False, error=repr(ex)[:300]))
    return None


def no_axis_status(out):
    """The status when find_axes returned no elliptic line: 'no_axis' if its search ran to the end, 'undetermined'
    if the budget cut it short (find_axes stops its guesses and its scan at the deadline: CoilStellaration
    DPQR7nCP..., axis found in 23 s on one run, 'no_axis' after 379 s of a 300 s budget on a slower one)."""
    if _budget.expired():
        out["deferred_reason"] = "budget: the axis search was cut short"
        return "undetermined"
    return "no_axis"


def find_axes(engine, source, Rguess=None, scan=False, nscan=48, verbose=False, guesses=(), check_lines=None):
    """Stellarator-symmetric period-one closed lines (magnetic axes and period-one X lines).

    Newton first, from circles at a few radii about a guess (the median radius of the source's
    current elements, or Rguess); the tracing scan of the outboard symmetry half-line only if no
    elliptic line is found or scan=True.  Monodromy, residue and transform are spectral.

    check_lines (None: off): a list -- a converged line still unresolved after the ladder goes through
    torux.nearaxis.check_line (kept only if a field line off-grid), its record appended to the list; a line set aside
    does not end the search (a solution of the truncated Galerkin system next to a coil, followed by no field line)."""
    nfp = source.nfp
    pts = source.sample_points()
    Rm = float(np.median(np.hypot(pts[:, 0], pts[:, 1]))) if len(pts) else 1.0
    R0s = [Rguess] if Rguess else []
    R0s += [Rm * f for f in (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15)]
    out = []

    def add(ax):
        if not ax.converged:
            return
        # resolve the axis to its own spectrum (nfp = 1 devices need many harmonics)
        for N2 in (32, 48, 64, 96, 128):
            tail = max(np.max(np.abs(ax.rc[-3:])), np.max(np.abs(ax.zs[-3:]))) / max(abs(ax.rc[0]), 1e-300)
            if tail < 1e-13 or ax.N >= N2:
                if tail < 1e-13:
                    break
                continue
            ax2 = AX.solve_axis(engine, nfp, float(ax.rc[0]), N=N2, guess=(ax.rc, ax.zs))
            if not ax2.converged:
                break
            ax = ax2
        if check_lines is not None:
            from .nearaxis import check_line
            ax, rec = check_line(engine, ax, AX.solve_axis)
            if rec is not None:
                check_lines.append(rec)
            if ax is None:
                return              # not a field line: set aside, the search goes on
        phs = np.linspace(0, 2 * np.pi / nfp, 33)[:-1]
        Ra, Za = ax.at(phs)
        for a in out:
            Rb, Zb = a.at(phs)
            if np.max(np.abs(Ra - Rb)) < 1e-8 and np.max(np.abs(Za - Zb)) < 1e-8:
                return
        AX.spectral_axis_monodromy(engine, ax)
        out.append(ax)
        if verbose:
            print(f"   axis R(0) {ax.at(np.array([0.0]))[0][0]:.10f} residue {ax.residue:.6f} iota {ax.iota:.12f}")
    from . import axisselect as _axsel
    for g in guesses:
        try:
            rc_g, zs_g = np.asarray(g[0], float), np.asarray(g[1], float)
            a_g = AX.solve_axis(engine, nfp, float(rc_g[0]), N=max(16, len(rc_g) - 1), guess=(rc_g, zs_g))
            if not a_g.converged and _axsel.on():
                # (AXSEL) a guessed line (the design path's D) whose Newton stalled at the rounding floor
                a_g, _ = _axsel.accept_stalled(engine, a_g)
            add(a_g)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
        if any(0 < a.residue < 1 for a in out):
            break
    for R0 in ([] if any(0 < a.residue < 1 for a in out) else R0s):
        if _budget.expired():
            break
        try:
            add(AX.solve_axis(engine, nfp, R0, N=16, guess=(np.array([R0]), np.array([0.0]))))
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
        if any(0 < a.residue < 1 for a in out):
            break
    if (scan or not any(0 < a.residue < 1 for a in out)) and not _budget.expired():
        # from 0.05 Rm: the axes of interlinked-coil devices (CNT, CSX) lie at R = 0.2-0.35 Rm; only reached
        # when the circle guesses found no elliptic line
        tried = []
        # twice as dense once more if no elliptic line was found: an elliptic line next to its hyperbolic partner
        # (within one spacing) leaves no sign change (cws 0.34800: R 0.5426 next to 0.556; the host scan is cheap)
        for ns in (max(nscan, 96), 2 * max(nscan, 96)):
            if ns > max(nscan, 96) and (any(0 < a.residue < 1 for a in out) or _budget.expired()):
                break
            Rs, Zend, _ = AX.symmetric_line_scan(engine, nfp, 0.05 * Rm, 1.5 * Rm, nscan=ns, nsteps=12, k=6)
            ok = np.isfinite(Zend)
            for i in range(len(Rs) - 1):
                if not (ok[i] and ok[i + 1]) or np.sign(Zend[i]) == np.sign(Zend[i + 1]):
                    continue
                if _budget.expired():
                    break
                R0 = Rs[i] - Zend[i] * (Rs[i + 1] - Rs[i]) / (Zend[i + 1] - Zend[i])
                if any(abs(R0 - r) < 1e-4 * Rm for r in tried):
                    continue
                tried.append(R0)
                ax_ = AX.solve_axis(engine, nfp, R0, N=16, guess=(np.array([R0]), np.array([0.0])))
                if not ax_.converged:
                    # a circle at the root is not always inside Newton's basin (rippled or strongly shaped lines):
                    # the traced half-period line, mirrored, is (the no-axis diagnosis: 20 of 28 CoilStellaration
                    # and cws sources without an elliptic line got one this way)
                    g_ = AX.traced_axis_guess(engine, nfp, R0)
                    if g_ is not None:
                        ax_ = AX.solve_axis(engine, nfp, R0, N=16, guess=g_)
                add(ax_)
    return out


def _noaxis_on(opts):
    """The no-axis stage runs unless switched off (torux.noaxis.NOAXIS['on'], TORUX_NOAXIS, analyse(noaxis={'on': False}))."""
    from .noaxis import NOAXIS
    o = dict(NOAXIS)
    o.update(opts or {})
    return bool(o.get("on"))


def _noaxis_stage(out, eng, source, flux, axes, deadline, quick, tol, dmin, verbose, opts):
    """torux.noaxis.noaxis_stage after a complete find_axes found no elliptic line.  Records the stage under
    out['noaxis']; returns 'done' when it answered (out's status, surface, guard / range set), an Axis (an elliptic
    period-one line the search missed: the family starts from it), 'cut' when its own line search was cut by the
    budget, or None (its complete search found nothing usable: the caller reports 'no_axis')."""
    from .noaxis import noaxis_stage
    t_s = time.time()
    try:
        na = noaxis_stage(eng, source, flux, axes, deadline=deadline, quick=quick, tol=tol, dmin=dmin, verbose=verbose,
                          opts=opts)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        out["noaxis"] = dict(kind="error", error=repr(e), traceback=traceback.format_exc()[-3000:])
        out["timings"]["noaxis"] = time.time() - t_s
        return None
    out["noaxis"] = {k: v for k, v in na.items() if k not in ("axis", "surface", "guard")}
    out["timings"]["noaxis"] = time.time() - t_s
    if na["kind"] == "axis":
        return na["axis"]
    if na["kind"] in ("surface", "range"):
        out["surface"] = na["surface"]
        out["guard"] = na.get("guard")
        out["refine_hist"] = na["surface"].pop("refine_hist", None)
        out["path"] = "axis->noaxis"
        if na["kind"] == "range":
            out["range"] = na["range"]
            out["status"] = "family"
        else:
            out["status"] = "surface"
        return "done"
    if not na.get("search_complete", True):
        return "cut"
    return None


@_budget.budgeted
def analyse(source, flux=None, dmin=None, tol=3e-15, beyond=0.0, verbose=False, max_jumps=6,
            Mmax=128, Nmax=128, budget_s=None, axis_guesses=(), region_pts=None, start_tori=(),
            upper_bound=False, quick=False, noaxis=None):
    if not getattr(source, "stellsym", True):
        # no stellarator symmetry: the general basis (R, Z with both parities, a phase condition), axis and
        # family of torux.family_general (no chain jumps or certificates there yet)
        from .family_general import analyse_general
        return analyse_general(source, flux=flux, dmin=dmin, tol=tol, verbose=verbose, Mmax=Mmax, Nmax=Nmax,
                               budget_s=budget_s, axis_guesses=axis_guesses, region_pts=region_pts)
    t0 = time.time()
    deadline = t0 + budget_s if budget_s else None
    out = dict(id=source.id, nfp=source.nfp, request_flux=flux, status="error", timings={}, chains=[], jumps=[])
    try:
        if dmin is None:
            dmin = source.meta.get("dmin", 0.05)
        eng = FieldEngine(source, dmin=dmin, region_pts=region_pts)
        out["engine"] = dict(nodes=eng.info.n_nodes, dmin=dmin, per_coil=region_pts is not None)
        from . import nearaxis as _na
        lines_checked = [] if _na.on() else None
        axes = find_axes(eng, source, guesses=axis_guesses, check_lines=lines_checked)
        out["axes"] = [dict(rc=a.rc.tolist(), zs=a.zs.tolist(), residue=a.residue, iota=a.iota, res=a.res) for a in axes]
        if lines_checked:
            out["axes_checked"] = lines_checked      # (torux.nearaxis) unresolved lines: kept or set aside
        ell = [a for a in axes if 0 < a.residue < 1]
        out["timings"]["axis"] = time.time() - t0
        noaxis_searched = False
        if not ell and not _budget.expired() and _noaxis_on(noaxis):
            # no elliptic period-one line after a complete search: the no-axis stage (torux/noaxis.py) -- lines the
            # search above misses, axes of the full-turn map, the request from the design boundary, evidence
            na = _noaxis_stage(out, eng, source, flux, axes, deadline, quick, tol, dmin, verbose, noaxis)
            if na == "done":
                out["timings"]["total"] = time.time() - t0
                return out
            if na == "cut":
                # the stage's own line search was cut by the budget: not a no-axis finding
                out["status"] = "undetermined"
                out["deferred_reason"] = "budget: the no-axis stage's line search was cut short"
                out["timings"]["total"] = time.time() - t0
                return out
            if na is not None:
                ell = [na]
                axes = list(axes) + [na]
                out["axes"].append(dict(rc=na.rc.tolist(), zs=na.zs.tolist(), residue=na.residue, iota=na.iota,
                                        res=na.res, found_by="noaxis"))
                axis_guesses = ()          # the family's axis is the stage's line, not the one nearest the design guess
            else:
                # the stage's closed-line search ran to its end and found no usable elliptic line: 'no_axis' (its later
                # steps -- seeds, survey -- may have spent the budget; that does not make the axis search a cut one)
                noaxis_searched = True
        if not ell:
            out["status"] = "no_axis" if noaxis_searched else no_axis_status(out)
            return out
        # the elliptic axis nearest the design guess, else the median coil radius
        pts = source.sample_points()
        Rm = float(axis_guesses[0][0][0]) if len(axis_guesses) else np.median(np.hypot(pts[:, 0], pts[:, 1]))
        ax = min(ell, key=lambda a: abs(a.rc[0] - Rm))
        # the toroidal cap at least the axis's N (the start is at N0 = axis N; only torux.nearaxis.check_line resolves
        # an axis past find_axes's 128 = Nmax, so this is a no-op without it)
        cont = Continuation(eng, source.nfp, M0=6, N0=max(16, ax.N), verbose=verbose, Mmax=Mmax,
                            Nmax=max(Nmax, ax.N), lean_source=source, lean_dmin=dmin)
        if not (quick and flux is not None):
            # (fast_newton 'fmeas') only quick mode's requests: a range's wall and a full-path family's are set by
            # their budgets, and a family that reaches farther
            # moved verified lower bounds and budget-marginal certificates both ways
            cont.fmeas_off = True
        from . import fast_family as _so2
        if _so2.on("cpredict"):
            cont.predict_state = {}      # (fast_family 'cpredict') Newton-Kantorovich stops in the family (not its refinement)
            if _so2.on("cpgrid"):
                cont.predict_state["grid"] = True    # (fast_family 'cpgrid') the step measured on the grid
        target = flux if flux is not None else np.inf
        if flux is None:
            # without a request the family is followed outward until it stalls: aim at 4 x the flux of a
            # disc reaching the nearest coil (beyond what any torus can enclose)
            R_ax, Z_ax = ax.at(np.array([0.0]))
            pa = np.array([R_ax[0], 0.0, Z_ax[0]])
            Bax = float(np.linalg.norm(np.asarray(eng.evaluate(pa[None, :], grad=False, A=False)["B"])[0]))
            d_coil = float(np.min(np.linalg.norm(pts - pa[None, :], axis=1))) if len(pts) else 0.3 * R_ax[0]
            target = 4.0 * Bax * np.pi * d_coil ** 2
            out["range_target"] = target
        sgn_target = None
        tori = []
        # the range (no request): the family may use 60 % of the budget, the rest refines and verifies its
        # outermost tori (a family run to the deadline left no time to verify any of them)
        fam_deadline = deadline if (flux is not None or deadline is None) else t0 + 0.6 * (deadline - t0)
        if flux is not None and deadline is not None:
            # with a request the family leaves the certificates (island band, edge barrier) a reserve: they ran on
            # its leftovers
            fam_deadline = deadline - min(CERT_RESERVE, 0.2 * (deadline - t0))
        good_start = [t for t in start_tori if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-6
                      and (not np.isfinite(target) or abs(t.flux) < abs(target))]
        if good_start:
            # continue from the outermost torus already known (design surfaces) instead of the axis
            st0 = max(good_start, key=lambda t: abs(t.flux))
            fam = cont.run(ax, abs(target) if np.isfinite(target) else 1.0, deadline=fam_deadline, start=st0,
                           **({"ds_min": 0.03} if quick else {}))
            out["continuation_start"] = dict(flux=st0.flux, iota=st0.iota)
        else:
            # quick (quick mode): a stall is declared once the step is below 3 % of sqrt(target) (each failed step
            # is a high-resolution Newton; halving to 0.2 % cost minutes on sources that stall anyway)
            fam = cont.run(ax, abs(target) if np.isfinite(target) else 1.0, deadline=fam_deadline,
                           **({"ds_min": 0.03} if quick else {}))
        tori = list(fam.tori)
        njump = 0
        brackets = []          # (torus before the chain, chain, torus beyond it or None)
        if quick and flux is not None and fam.stop_reason != "reached":
            # quick mode: the family stopped before the request (a chain, a chaotic layer, the edge); chains,
            # jumps and certificates are the full path's
            out["profile"] = [torus_record(t) for t in tori]
            out["family_stop"] = fam.stop_reason
            out["status"] = "undetermined"
            out["deferred"] = True
            out["timings"]["family"] = out["timings"]["total"] = time.time() - t0
            return out
        while fam.stop_reason == "stalled" and njump < max_jumps and (fam_deadline is None or time.time() < fam_deadline):
            last = tori[-1]
            diota = tori[-1].iota - tori[-2].iota if len(tori) >= 2 else 0.0
            pr = nearest_rational_ahead(last.iota, diota, source.nfp, qmax=40)
            if pr is None:
                break
            p, q = pr
            # the chain's lines and the jump hold the family's deadline too: their closed-line and fixed-iota
            # Newton solves were never cut
            ch = analyse_chain(eng, tori, p, q, verbose=verbose, deadline=fam_deadline)
            out["chains"].append(dict(p=p, q=q, iota_r=ch.iota_r, flux_r=ch.flux_r, shear=ch.shear,
                                      half_width_est=ch.half_width_est,
                                      lines=[dict(residue=l.residue, R0=l.section_points(0.0)[0].tolist(),
                                                  res=l.res) for l in ch.lines],
                                      **({"cut": True} if ch.info.get("cut") else {})))
            with _budget.scope(fam_deadline):
                new = jump_over(cont, tori, ch, direction=math.copysign(1.0, diota) if diota else 1.0,
                                verbose=verbose, axis=ax)
            out["jumps"].append(dict(p=p, q=q, ok=new is not None, flux_after=None if new is None else new.flux))
            brackets.append((tori[-1], ch, new))
            if new is None:
                break
            njump += 1
            if np.isfinite(target) and abs(new.flux) >= abs(target):
                # the jump passed the request: the request lies in the band it jumped
                out["jumped_past"] = True
                tori.append(new)
                break
            fam = cont.run(ax, abs(target), start=new, deadline=fam_deadline)
            tori.extend(fam.tori)
        out["profile"] = [torus_record(t) for t in tori]
        out["family_stop"] = fam.stop_reason
        out["timings"]["family"] = time.time() - t0
        reached = [t for t in tori if np.isfinite(target) and abs(abs(t.flux) - abs(target)) <= 1e-12 * abs(target)]
        if flux is not None and reached:
            sol = reached[-1]
            rc_ = lean_continuation(cont, source, sol, dmin)
            if _so2.on("cpredict"):
                rc_.predict_state = None     # (fast_family 'cpredict') the family's only: the refinement as without it
            out["engine"]["refine_nodes"] = rc_.engine.info.n_nodes
            caps = (320, 320) if quick else FULL_CAPS
            best = refine_to_limit(rc_, sol, sol.flux, target=tol, deadline=deadline, verbose=verbose, Mcap=caps[0],
                                   Ncap=caps[1])
            q_ = rc_.problem(best.M, best.N).quadrature_check(best.rc, best.zs)
            if q_ > 1e-8:
                # the torus nearer a coil than the engine's dmin: refined again on a near-coil engine (near_coil_retry)
                nc = near_coil_retry(cont, source, best, dmin, tol, deadline=deadline, caps=caps)
                out["near_coil"] = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(q_))
                if nc is not None:
                    rc_, best, q_ = nc[0], nc[1], nc[2]
            out["timings"]["refine"] = time.time() - t0
            # shear from the family near the request
            fl = np.array([t.flux for t in tori]); io = np.array([t.iota for t in tori])
            j = int(np.argmin(np.abs(fl - best.flux)))
            lo, hi = max(0, j - 1), min(len(tori) - 1, j + 1)
            shear_phi = (io[hi] - io[lo]) / (fl[hi] - fl[lo]) if hi > lo else np.nan
            g = resonance_guard(rc_.engine, best, shear=shear_phi * 2 * np.pi)
            out["guard"] = dict(ok=g["ok"], h=g["h"], x_over_w=g["x_over_w"], worst=g["worst"],
                                n_checked=len(g["checked"]))
            out["surface"] = torus_record(best, full=True)
            out["surface"]["quad_err_half"] = q_
            out["refine_hist"] = best.info.get("refine_hist")
            out["refine_stop"] = best.info.get("refine_stop")
            quad_ok = out["surface"]["quad_err_half"] <= 1e-8
            topo = torus_topology(best, ax)
            out["surface"]["topology"] = topo
            out["status"] = ("surface" if (best.bn_off <= BN_SURFACE and g["ok"] and quad_ok and topo["ok"])
                             else "surface_unverified")
        elif flux is not None:
            out["status"] = "undetermined"
            cert = certify_request(eng, cont, brackets, abs(target), tol=tol, deadline=deadline, verbose=verbose,
                                   axis=ax)
            if cert is not None:
                out["certificate"] = cert
                if cert.get("ok") and cert.get("holds_request") and cert.get("rigorous", True):
                    out["status"] = "no_surface"
            out["timings"]["certify"] = time.time() - t0
        else:
            # the range: the outermost torus the family reached, refined and guarded = a verified lower
            # bound on the largest flux a surface encloses; chains met on the way are in out["chains"]
            out["status"] = "family"
            cands = [t for t in tori if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-6]
            # the outermost torus is refined and checked; if it does not reach acceptance (next to the edge's
            # chains or chaos) the next ones inward are tried (up to three): each verified torus is a lower bound
            cands = sorted(cands, key=lambda t: -abs(t.flux))
            fl = np.array([t.flux for t in tori]); io = np.array([t.iota for t in tori])
            tried = []
            quad_refused = []
            # (RANGE_INNER, off by default) the inward fallback's reserve: with an inner candidate to try, the loop
            # below and its near-coil retries end reserve seconds before the deadline (loop_deadline; else deadline)
            inner_on = bool(RANGE_INNER.get("on")) and bool(cands)
            loop_deadline, loop_states = deadline, []
            if inner_on and deadline is not None and range_inner_candidates(
                    cands, [dict(flux=abs(float(t.flux))) for t in cands[:3]]):
                reserve = min(float(RANGE_INNER["reserve_s"]),
                              float(RANGE_INNER["reserve_share"]) * max(0.0, deadline - time.time()))
                if reserve > 0:
                    loop_deadline = deadline - reserve
                    out["range_reserve"] = reserve
            for outer in cands[:3]:
                if tried and loop_deadline is not None and time.time() > loop_deadline:
                    break
                rc_ = lean_continuation(cont, source, outer, dmin)
                if _so2.on("cpredict"):
                    rc_.predict_state = None     # (fast_family 'cpredict') the family's only
                out["engine"]["refine_nodes"] = rc_.engine.info.n_nodes
                with _budget.scope(loop_deadline):
                    best = refine_to_limit(rc_, outer, outer.flux, target=tol, deadline=loop_deadline)
                j = int(np.argmin(np.abs(fl - outer.flux)))
                lo, hi = max(0, j - 1), min(len(tori) - 1, j + 1)
                shear_phi = (io[hi] - io[lo]) / (fl[hi] - fl[lo]) if hi > lo else np.nan
                g = resonance_guard(rc_.engine, best, shear=shear_phi * 2 * np.pi)
                rec = torus_record(best, full=True)
                rec["quad_err_half"] = rc_.problem(best.M, best.N).quadrature_check(best.rc, best.zs)
                ok = best.bn_off <= BN_SURFACE and g["ok"] and rec["quad_err_half"] <= 1e-8
                tried.append(dict(flux=abs(float(outer.flux)), bn_off=float(best.bn_off), M=int(best.M), N=int(best.N),
                                  guard=bool(g["ok"]), ok=bool(ok)))
                if inner_on:
                    loop_states.append((outer, rc_, best, shear_phi))    # (RANGE_INNER resume)
                if ok:
                    break
                if rec["quad_err_half"] > 1e-8:
                    quad_refused.append((outer, best, shear_phi, rec["quad_err_half"]))
            # (near-coil) the candidates outward of the accepted one (all, when none was) that the quadrature check
            # refused -- tori nearer a coil than the engine's dmin -- refined again on a near-coil engine with the time
            # left, outermost first (near_coil_retry): a larger lower bound or a first one, never fewer.  Retrying at
            # once inside the loop above would leave inner candidates untried
            for outer_r, best_r, shear_r, q_before in quad_refused:
                if loop_deadline is not None and time.time() > loop_deadline - 10:
                    break
                with _budget.scope(loop_deadline):
                    nc = near_coil_retry(cont, source, best_r, dmin, tol, deadline=loop_deadline)
                nc_info = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(q_before))
                tried.append(dict(flux=abs(float(outer_r.flux)), near_coil=nc_info))
                if nc is None:
                    continue
                rc2, best2, q2 = nc[0], nc[1], nc[2]
                g2 = resonance_guard(rc2.engine, best2, shear=shear_r * 2 * np.pi)
                ok2 = best2.bn_off <= BN_SURFACE and g2["ok"] and q2 <= 1e-8
                tried[-1].update(bn_off=float(best2.bn_off), M=int(best2.M), N=int(best2.N), guard=bool(g2["ok"]),
                                 ok=bool(ok2))
                if ok2:
                    rec = torus_record(best2, full=True)
                    rec["quad_err_half"] = q2
                    rec["near_coil"] = nc_info
                    best, g, ok = best2, g2, True
                    break
            eb_torus = best if cands else None      # the edge bound (upper_bound) starts where the standard would
            if inner_on:
                out["timings"]["range_loop"] = time.time() - t0
            resume = inner_on and loop_deadline != deadline and RANGE_INNER.get("resume", True)
            if resume and ok:
                # (RANGE_INNER resume) the loop accepted a candidate inward of one its reserve cut: the reserve resumes
                # that one (a larger bound, or nothing)
                t_re = time.time()
                r_re = range_resume(cont, source, cands, tori, tried, loop_states, quad_refused, dmin, tol,
                                    deadline=deadline, floor=abs(float(best.flux)))
                out["timings"]["range_resume"] = time.time() - t_re
                if r_re is not None:
                    best, g, rec = r_re["best"], r_re["g"], r_re["rec"]
            if inner_on and not ok:
                # (RANGE_INNER) none of the standard candidates accepted: inner tori with the time left (range_inner)
                t_in = time.time()
                r_in = range_inner(cont, source, tori, cands, tried, dmin, tol, deadline=deadline, axis=ax,
                                   verbose=verbose)
                out["timings"]["range_inner"] = time.time() - t_in
                if r_in is not None:
                    best, g, rec, ok = r_in["best"], r_in["g"], r_in["rec"], True
                r_re = None
                if resume:
                    # the reserve given back: what it took from the standard loop, outward of an accepted inner torus
                    t_re = time.time()
                    r_re = range_resume(cont, source, cands, tori, tried, loop_states, quad_refused, dmin, tol,
                                        deadline=deadline, floor=abs(float(best.flux)) if ok else 0.0)
                    out["timings"]["range_resume"] = time.time() - t_re
                    if r_re is not None:
                        best, g, rec, ok = r_re["best"], r_re["g"], r_re["rec"], True
                if r_in is not None and r_re is None and r_in["entry"]["inner"].get("secure"):
                    # (secure) the fractions' tori outward of the family torus that answered, with the time left
                    t_ti = time.time()
                    r_ti = range_inner(cont, source, tori, cands, tried, dmin, tol, deadline=deadline, axis=ax,
                                       opts=dict(secure=False), verbose=verbose, floor=abs(float(best.flux)))
                    out["timings"]["range_tighten"] = time.time() - t_ti
                    if r_ti is not None:
                        best, g, rec = r_ti["best"], r_ti["g"], r_ti["rec"]
            if cands:
                out["range"] = dict(verified_lower_bound=abs(best.flux) if ok else None, outer_torus=rec,
                                    guard=dict(ok=g["ok"], h=g["h"], worst=g["worst"]), verified=bool(ok),
                                    tried=tried)
                out["surface"] = rec
                if upper_bound and (deadline is None or time.time() < deadline):
                    # an upper bound on every covered torus's flux: the edge barrier beyond the outermost torus
                    chq = (out.get("chains") or [{}])[-1].get("q")
                    eb = edge_bound(eng, source, eb_torus, request=None, deadline=deadline,
                                    note="range: beyond the outermost torus the family reached", chain_q=chq, axis=ax)
                    out["range"]["upper_bound"] = eb.get("F_up") if eb.get("ok") else None
                    out["range"]["upper_bound_evidence"] = {k: v for k, v in eb.items() if k != "interval"}
        out["timings"]["total"] = time.time() - t0
        return out
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        out["status"] = "error"
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
        out["timings"]["total"] = time.time() - t0
        return out


def certify_request(eng, cont, brackets, request, tol=3e-15, deadline=None, verbose=False, axis=None):
    """Island-band certificate for a request that lies between a torus before a chain (T1) and the
    torus a jump found beyond it (T2): both refined to the rounding floor, island tori around the
    chain's O line, then torux.certify.island_band_certificate.  Returns the certificate dict (with the
    verification numbers of its three tori) or None if no bracket holds the request."""
    from .certify import island_tori, island_band_certificate
    for T1, ch, T2 in brackets:
        if T2 is None:
            if abs(T1.flux) < request:
                return edge_bound(eng, eng.source, T1, request, deadline=deadline, verbose=verbose,
                                  note=f"no torus found beyond the {ch.p}/{ch.q} chain", chain_q=ch.q, axis=axis)
            continue
        F1, F2 = abs(T1.flux), abs(T2.flux)
        if not (F1 < request < F2):
            continue
        if not ch.O:
            return dict(ok=False, reason="no elliptic line found for the chain", chain=(ch.p, ch.q))
        T1r = refine_to_limit(cont, T1, T1.flux, target=tol, deadline=deadline)
        T2r = refine_to_limit(cont, T2, T2.flux, target=tol, deadline=deadline, fixed_iota=None)
        best = None
        for O in ch.O:
            w = (2 * ch.half_width_est / ch.q) if np.isfinite(ch.half_width_est) else None
            it, _ = island_tori(eng, O, width_est=w, deadline=deadline)
            good = [t for t in it if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-9]
            if good:
                cand = max(good, key=lambda t: abs(t.flux))
                if best is None or abs(cand.flux) > abs(best[1].flux):
                    best = (O, cand)
        if best is None:
            return dict(ok=False, reason="no island torus around the chain's O line", chain=(ch.p, ch.q),
                        F1=F1, F2=F2)
        O, I = best
        c = island_band_certificate(T1r, T2r, I, O, request_flux=request)
        c.update(chain=(ch.p, ch.q), O_residue=O.residue, T1=torus_record(T1r), T2=torus_record(T2r),
                 island=dict(flux=float(I.flux), iota=float(I.iota), M=int(I.M), N=int(I.N), bn_off=float(I.bn_off)),
                 T1_bn=float(T1r.bn_off), T2_bn=float(T2r.bn_off))
        # the certificate's tori must be accurate flux surfaces themselves
        c["tori_ok"] = bool(T1r.bn_off < 1e-12 and T2r.bn_off < 1e-12 and I.bn_off < 1e-9)
        shear = (T2r.iota - T1r.iota) / ((T2r.flux - T1r.flux) / (2 * np.pi)) if T2r.flux != T1r.flux else np.nan
        g1, g2 = resonance_guard(eng, T1r, shear=shear), resonance_guard(eng, T2r, shear=shear)
        c["guard"] = dict(T1=dict(ok=g1["ok"], h=g1["h"], worst=g1["worst"]),
                          T2=dict(ok=g2["ok"], h=g2["h"], worst=g2["worst"]))
        c["tori_ok"] = bool(c["tori_ok"] and g1["ok"] and g2["ok"])
        # embedded tori of the right families: T1, T2 around the magnetic axis, the island torus around its O line
        topo = dict(T1=torus_topology(T1r, axis), T2=torus_topology(T2r, axis), island=torus_topology(I, O))
        c["topology"] = topo
        c["tori_ok"] = bool(c["tori_ok"] and all(v["ok"] for v in topo.values()))
        c["ok"] = bool(c["ok"] and c["tori_ok"])
        return c
    if brackets == [] and request is not None:
        return None
    return None


def local_bracket(cont, guess, request, iota_inner, deadline=None, verbose=False, max_offsets=9, screen=None,
                  bisect=3):
    """Fixed-iota tori just inside and just beyond a request whose torus Newton failed (typically: the
    request lies in an island band), solved directly from the failed solution's shape -- no continuation
    through the resonances on the way.  The transform offsets from the failed solution's iota run
    through a geometric ladder (1e-5 .. 1e-1 |iota|, a noble perturbation keeps them off low-order
    rationals), on the inner side (towards iota_inner) and the outer side; the first accepted torus on
    each side (bn_off below the continuation tolerance) is kept.  Returns (T1, T2, log): T1 with
    |flux| < request, T2 with |flux| > request (either may be None)."""
    io_d = float(guess.iota)
    if screen is None:
        # every converged candidate is raised: at the starting resolution a torus just outside a thin
        # chain shows the band's own B.n level (its large resonant response leaves the harmonics of the
        # resonance unresolved), so the starting resolution cannot screen
        screen = cont.bn_accept
    inward = math.copysign(1.0, iota_inner - io_d) if iota_inner != io_d else -1.0
    g = (math.sqrt(5) - 1) / 2
    # the resonance the request most likely sits in (nearest p/q weighted by q^2, q <= 60): tori next to
    # a period-q chain carry its harmonics m = q, 2q, ...; start at M >= 2q + 4 (raising from below
    # stalls: no gain until the resolution passes the resonance's harmonics)
    from .chains import rationals_between
    from .family import LADDER
    rho = io_d / guess.nfp
    near = rationals_between(rho - 0.05 * abs(rho) - 1e-3, rho + 0.05 * abs(rho) + 1e-3, 60)
    q_res = min(near, key=lambda pq: abs(rho - pq[0] / pq[1]) * pq[1] ** 2)[1] if near else 0
    M0 = max(guess.M, next((L for L in LADDER if L >= 2 * q_res + 4), LADDER[-1]))
    M0 = min(M0, max(cont.Mmax, guess.M))
    if M0 != guess.M:
        guess = guess.resized(M0, guess.N)
        guess.bn_off = np.nan
    ladder = abs(io_d) * np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])[:max_offsets]
    T1 = T2 = None
    log = []
    fail = {inward: 0.0, -inward: 0.0}          # largest offset that failed, per side
    okd = {inward: None, -inward: None}         # offset of the accepted torus, per side

    def attempt(d, side, shape):
        target = io_d + side * d * (1 + 1e-3 * g)
        st = TorusState(shape.rc.copy(), shape.zs.copy(), target, shape.flux, shape.M, shape.N, shape.nfp,
                        phi0=shape.phi0)
        sol = cont.solve(st, shape.flux, shape.M, shape.N, fixed_iota=target, maxit=20, patience=3)
        ok = bool(sol.converged and np.isfinite(sol.bn_off) and sol.bn_off < screen)
        if ok:
            sol = cont.raise_until(sol, sol.flux, cont.bn_cont, fixed_iota=target)
            ok = bool(np.isfinite(sol.bn_off) and sol.bn_off < 1e3 * cont.bn_cont)
        log.append(dict(iota=target, side="inner" if side == inward else "outer", ok=ok,
                        flux=float(sol.flux) if np.isfinite(sol.flux) else None, bn=float(sol.bn_off),
                        M=int(sol.M), N=int(sol.N)))
        if verbose:
            print(f"   bracket iota {target:.10f} ({'inner' if side == inward else 'outer'}) ok {ok} "
                  f"flux {sol.flux:.8e} bn {sol.bn_off:.2e} M,N {sol.M},{sol.N}", flush=True)
        return sol if ok else None

    def keep(sol):
        nonlocal T1, T2
        F = abs(sol.flux)
        # a torus collapsed onto the axis satisfies the fixed-iota equations trivially (flux ~ 0): not a bracket
        if not np.isfinite(F) or F < 1e-3 * request:
            return False
        if F < request and (T1 is None or F > abs(T1.flux)):
            T1 = sol
            return True
        if F > request and (T2 is None or F < abs(T2.flux)):
            T2 = sol
            return True
        return False
    for d in ladder:
        for side in (inward, -inward):
            if okd[side] is not None:
                continue
            if deadline is not None and time.time() > deadline:
                return T1, T2, log
            sol = attempt(d, side, guess)
            if sol is None:
                fail[side] = d
            elif keep(sol):
                okd[side] = d
        if okd[inward] is not None and okd[-inward] is not None:
            break
    # bisect each side's offset (geometric mean of the last failure and the success) from the accepted
    # torus: the band certificate needs T1 and T2 within about a band width of the chain
    for side in (inward, -inward):
        for _ in range(bisect):
            if okd[side] is None or fail[side] <= 0 or okd[side] / fail[side] < 1.3:
                break
            if deadline is not None and time.time() > deadline:
                break
            d = math.sqrt(okd[side] * fail[side])
            ref_t = T1 if side == inward else T2
            sol = attempt(d, side, ref_t) if ref_t is not None else None
            if sol is not None and keep(sol):
                okd[side] = d
            else:
                fail[side] = d
    return T1, T2, log


def _manifold_band(eng, T1, T2, p, q, request, lines, deadline=None, verbose=False, bracket_check=True, budget=900.0,
                   axis_inverse=False, axis_line=None):
    """torux.islandband.island_band_certificate_manifold (module docstring there: the theorem, what is strict and
    what is validated) for the chain p/q between T1 and T2 (verified tori, or a proxy annulus with
    bracket_check=False), capped at `budget` seconds; failures are returned as records, not raised.
    axis_line: the analysis's period-one line D (the line its answers' tori enclose); the band's axis point must be
    D's (torux.islandband, 'Axis identity': on the proxy annulus nothing else puts it inside the tori around D) -- without
    it no band is attempted."""
    if axis_line is None:
        return dict(ok=False, kind="island_band_manifold", chain=dict(p=int(p), q=int(q)), rigorous=True,
                    reason="no analysis axis given: the band's axis point cannot be identified with the analysis's line")
    try:
        from .islandband import island_band_certificate_manifold
        d = time.time() + budget
        d = d if deadline is None else min(deadline, d)
        # pairs_fallback: when one orbit's heteroclinic points do not close the band, pairs of X orbits and the
        # other symmetry plane (kind island_band_manifold_pairs: 'excluded' open intervals, holds_request when the
        # request lies in one)
        c = island_band_certificate_manifold(eng, eng.source, T1, T2, p, q, request, deadline=d, verbose=verbose,
                                             lines=lines, bracket_check=bracket_check, pairs_fallback=True,
                                             axis_inverse=axis_inverse, axis_line=axis_line)
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        c = dict(ok=False, kind="island_band_manifold", chain=dict(p=p, q=q), reason=repr(e),
                 traceback=traceback.format_exc()[-2000:])
    c.setdefault("rigorous", True)
    return {k: v for k, v in c.items() if k != "polygons"}


def _traced_windows(base, axis, frac=0.15):
    """Windows on the symmetric half-lines of both planes around the base torus's own crossings of Z = 0:
    {(plane, side): (r_lo, r_hi)}, r the distance from the axis point, +-frac of the base's."""
    nfp = base.nfp
    win = {}
    for plane in (0, 1):
        phi_s = plane * np.pi / nfp
        Ra = float(axis.at(np.array([phi_s]))[0][0])
        R2, _ = base.geometry_on(np.array([0.0, np.pi]), np.array([phi_s, phi_s]))
        for Rb in R2:
            side = 1 if Rb > Ra else -1
            r = abs(float(Rb) - Ra)
            if np.isfinite(r) and r > 0:
                win[(plane, side)] = ((1 - frac) * r, (1 + frac) * r)
    return win


def chain_band_by_proxy(eng, base, request, deadline=None, verbose=False, qmax=60, n_chains=3, rho_hint=None,
                        axis=None, traced=False, axis_inverse=False, annulus="standard"):
    """The manifold island band for the chains the request most likely sits in, without a bracket: the chains
    p/q nearest the base torus's transform (weighted by q^2, q >= 2), their closed lines from the base, a proxy
    annulus of the base's normal offsets around them (torux.islandband.proxy_annulus; the theorem does not use
    the annulus' surfaces, bracket_check=False), widened once when a branch leaves it.  Returns the first
    certificate that holds the request, else the attempts (list) for the record.

    Third-pass options (torux.analysis.POLICY; the defaults are the full path's behaviour): rho_hint, a rotation (per
    period) measured at the request -- the seeds' base-polish circle's -- whose nearest chains (weighted by q^2) come
    first; traced (with axis): a chain whose closed lines Newton does not find from the base's curves gets them from
    traced sign changes on the symmetric half-lines around the base's crossings (torux.chainfind.symmetric_chain_lines);
    annulus="sym": the annulus fitted with the symmetry (torux.islandband.proxy_annulus_sym: the standard fit left the
    inner surface on the axis for low-order chains of failed design tori), the standard annulus after it when a
    branch leaves it at both margins; axis_inverse: the design's inverse-hyperbolic period-one line as the axis point.
    axis: the analysis's period-one line D -- every band's axis point must be D's (torux.islandband 'Axis identity'; no
    band is attempted without it)."""
    from .chains import rationals_between, chain_lines
    from .islandband import proxy_annulus as proxy_annulus_standard, proxy_annulus_sym, chain_flux_estimate
    nfp = base.nfp
    rho = float(base.iota) / nfp

    def nearest(r0, k):
        nr = rationals_between(r0 - 0.05 * abs(r0) - 1e-3, r0 + 0.05 * abs(r0) + 1e-3, qmax)
        return sorted([pq for pq in nr if pq[1] >= 2], key=lambda pq: abs(r0 - pq[0] / pq[1]) * pq[1] ** 2)[:k]
    near = nearest(rho, n_chains)
    if rho_hint is not None and np.isfinite(rho_hint):
        # the circle's rotation and the base's share their sign convention (both per period, signed like iota)
        hint = nearest(float(rho_hint), 2)
        near = hint + [pq for pq in near if pq not in hint]
        near = near[:max(n_chains, len(hint))]
    attempts = []
    # the closed lines of every candidate first, then the chains in the order of their estimated flux's distance
    # from the request
    cand = []
    for (p, q) in near:
        if deadline is not None and time.time() > deadline - 30:
            break
        try:
            lines = chain_lines(eng, base, p, q, verbose=False,
                                deadline=None if deadline is None else deadline - 30)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            attempts.append(dict(chain=dict(p=p, q=q), ok=False, reason=f"closed lines: {e!r}"))
            continue
        how = "newton from the base"
        if not lines and traced and axis is not None and q <= 24 and (deadline is None or time.time() < deadline - 60):
            try:
                from .chainfind import symmetric_chain_lines
                t_l = time.time()
                d_l = None if deadline is None else time.time() + min(120.0, 0.15 * max(deadline - time.time(), 0.0))
                lines, linfo = symmetric_chain_lines(eng, eng.source, axis, p, q, _traced_windows(base, axis),
                                                     deadline=d_l)
                how = f"traced ({time.time() - t_l:.0f} s)"
            except Exception as e:  # noqa: BLE001
                if environment_failure(e):
                    raise
                lines, how = [], f"traced: {e!r}"
        if not lines:
            attempts.append(dict(chain=dict(p=p, q=q), ok=False, reason="no closed lines of the chain",
                                 **({"lines": how} if traced else {})))
            continue
        try:
            est = abs(float(chain_flux_estimate(eng, eng.source, lines, p, q)["flux"]))
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            est = np.inf
        cand.append((abs(abs(request) - est), p, q, lines, how))
    for _, p, q, lines, how in sorted(cand, key=lambda t: t[0]):
        if deadline is not None and time.time() > deadline - 30:
            break
        # annulus="standard": margins 1.5, 4.0 of torux.islandband.proxy_annulus (widened once when a branch leaves it);
        # annulus="sym": the symmetric fit's annulus (narrow, centred on the chain) at 1.5, 4.0, then the standard
        # annulus when a branch leaves both (its width from the residuals of a coarser fit: wide where the base is far
        # from the chain -- QUASR 0182436's 1/4 band (0.0225, 0.0759) was certified on it)
        seq = ((("standard", 1.5), ("standard", 4.0)) if annulus == "standard" else
               (("sym", 1.5), ("sym", 4.0), ("standard", 1.5), ("standard", 4.0)))
        for kind, margin in seq:
            if annulus != "standard" and deadline is not None and time.time() > deadline - 30:
                break
            try:
                Tin, Tout, pinfo = (proxy_annulus_sym(base, lines, margin=margin) if kind == "sym"
                                    else proxy_annulus_standard(base, lines, margin=margin))
            except Exception as e:  # noqa: BLE001
                if environment_failure(e):
                    raise
                if annulus == "standard":
                    attempts.append(dict(chain=dict(p=p, q=q), ok=False, reason=f"proxy annulus: {e!r}"))
                    break
                attempts.append(dict(chain=dict(p=p, q=q), ok=False, reason=f"proxy annulus ({kind}): {e!r}"))
                continue
            c = _manifold_band(eng, Tin, Tout, p, q, request, lines, deadline=deadline, verbose=verbose,
                               bracket_check=False, axis_inverse=axis_inverse, axis_line=axis)
            c["path"] = "proxy annulus"
            c["proxy"] = pinfo if annulus == "standard" else dict(pinfo, fit=kind)
            if traced:
                c["lines_how"] = how
            attempts.append({k: v for k, v in c.items() if k not in ("evidence",)})
            if c.get("ok") and c.get("holds_request"):
                return c, attempts
            lost = any((g or {}).get("stop") in ("lost", "max_pts") for g in (c.get("growth") or {}).values()) \
                if isinstance(c.get("growth"), dict) else False
            if c.get("ok") or not lost:
                break
    return None, attempts


def band_between(eng, cont, T1, T2, request, tol=3e-15, qmax=60, deadline=None, verbose=False, axis=None):
    """Island-band certificate for a request between verified tori T1 (inside) and T2 (beyond): the
    rationals between their transforms (lowest order first), the chain's closed lines from the nearer
    torus, island tori around its O lines, T1 and T2 refined to the rounding floor, then
    torux.certify.island_band_certificate.  Returns the first certificate holding the request, else the
    last attempt (dict) or None when no rational lies between.  axis: the analysis's period-one line D, which the
    manifold band's axis point must be (torux.islandband 'Axis identity')."""
    from .chains import rationals_between, chain_lines
    from .certify import island_tori, island_band_certificate
    nfp = T1.nfp
    rats = rationals_between(T1.iota / nfp, T2.iota / nfp, qmax)
    if not rats:
        return None
    last = None
    T1r = T2r = None
    for (p, q) in rats[:4]:
        if deadline is not None and time.time() > deadline:
            break
        iota_r = p / q * nfp
        base = T1 if abs(T1.iota - iota_r) <= abs(T2.iota - iota_r) else T2
        lines = chain_lines(eng, base, p, q, verbose=verbose)
        O = [l for l in lines if 0 < l.residue < 1]
        if verbose:
            print(f"   band {p}/{q}: {len(lines)} lines, {len(O)} elliptic", flush=True)
        # the manifold band (torux.islandband: exact symmetric heteroclinic points, chains of any width) first;
        # island tori around the O line (which fail for thin chains) only when it does not certify
        c_man = None
        if lines:
            c_man = _manifold_band(eng, T1, T2, p, q, request, lines, deadline=deadline, verbose=verbose, axis_line=axis)
            last = c_man
            if c_man.get("ok") and c_man.get("holds_request"):
                return c_man
        if not O:
            last = dict(ok=False, reason="no elliptic line found for the chain", chain=(p, q), manifold=c_man)
            continue
        if T1r is None:
            T1r = refine_to_limit(cont, T1, T1.flux, target=tol, deadline=deadline, fixed_iota=T1.iota)
            T2r = refine_to_limit(cont, T2, T2.flux, target=tol, deadline=deadline, fixed_iota=T2.iota)
        best = None
        for ln in O:
            it, _ = island_tori(eng, ln, width_est=(abs(T2.flux) - abs(T1.flux)) / q, deadline=deadline)
            good = [t for t in it if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-9]
            if good:
                cand = max(good, key=lambda t: abs(t.flux))
                if best is None or abs(cand.flux) > abs(best[1].flux):
                    best = (ln, cand)
        if best is None:
            # the manifold attempt's own failure is the informative one (it was overwritten by this)
            last = dict(ok=False, reason="no island torus around the chain's O line", chain=(p, q), manifold=c_man)
            continue
        ln, I = best
        c = island_band_certificate(T1r, T2r, I, ln, request_flux=request)
        c.update(kind="island_band", chain=(p, q), O_residue=float(ln.residue), T1=torus_record(T1r),
                 T2=torus_record(T2r), island=dict(flux=float(I.flux), iota=float(I.iota), M=int(I.M), N=int(I.N),
                                                    bn_off=float(I.bn_off)))
        c["tori_ok"] = bool(T1r.bn_off < 1e-12 and T2r.bn_off < 1e-12 and I.bn_off < 1e-9)
        # T1 and T2 lie next to the chain by construction: the resonance guard on both (shear from the pair)
        shear = (T2r.iota - T1r.iota) / ((T2r.flux - T1r.flux) / (2 * np.pi)) if T2r.flux != T1r.flux else np.nan
        g1, g2 = resonance_guard(eng, T1r, shear=shear), resonance_guard(eng, T2r, shear=shear)
        c["guard"] = dict(T1=dict(ok=g1["ok"], h=g1["h"], worst=g1["worst"]),
                          T2=dict(ok=g2["ok"], h=g2["h"], worst=g2["worst"]))
        c["tori_ok"] = bool(c["tori_ok"] and g1["ok"] and g2["ok"])
        # embedded tori of the right families: T1, T2 around the magnetic axis, the island torus around its O line
        topo = dict(T1=torus_topology(T1r, axis), T2=torus_topology(T2r, axis), island=torus_topology(I, ln))
        c["topology"] = topo
        c["tori_ok"] = bool(c["tori_ok"] and all(v["ok"] for v in topo.values()))
        # the three tori checked with the independent reference field (B.n, flux)
        if c["ok"] and c["tori_ok"]:
            from .verify import verify_torus
            vs = {}
            for name, T in (("T1", T1r), ("T2", T2r), ("island", I)):
                try:
                    v = verify_torus(eng.source, T, nlines=0, precise_pts=4)
                    vs[name] = {k: v.get(k) for k in ("bn_max", "bn_median", "flux_rel_err", "precise_vs_fast64")}
                except Exception as e:  # noqa: BLE001
                    if environment_failure(e):
                        raise
                    vs[name] = dict(error=repr(e))
            lim = dict(T1=1e-12, T2=1e-12, island=1e-9)
            c["independent"] = vs
            c["independent_ok"] = bool(all("bn_max" in v and v["bn_max"] < lim[k] and v["flux_rel_err"] < 10 * lim[k]
                                           for k, v in vs.items()))
            c["ok"] = bool(c["ok"] and c["independent_ok"])
        c["ok"] = bool(c["ok"] and c["tori_ok"])
        last = c
        if c["ok"] and c.get("holds_request"):
            return c
    return last


EDGEBOUND = _os.environ.get("TORUX_EDGEBOUND", "0") == "1"     # the lost-curve bound (torux.edgebound) first: OFF by default
# with TORUX_EDGEBOUND=1: TORUX_EDGEBOUND_METHOD=exit marches exit curves from the lost curve
EDGEBOUND_METHOD = _os.environ.get("TORUX_EDGEBOUND_METHOD", "curve")


def edge_bound(eng, source, T_last, request=None, deadline=None, verbose=False, note="", chain_q=None, axis=None):
    """An upper bound on the flux of every covered torus beyond T_last -- the proof of 'no surface beyond F_up'.
    First torux.edgebound (a certified closed curve of lost points around the axis; F_up = the flux inside it; axis: the
    analysis' elliptic axis); a request it holds is returned at once, a range's bound too when less than 120 s are
    left; otherwise the rigorous edge barrier (torux.barrier.edge_barrier: hyperbolic closed lines beyond T_last whose
    manifolds reach the lost region) runs as before and, for a range, the smaller of the two bounds is kept.  Falls
    back to nothing (never to the sampled escape bound) when the modules or a barrier are missing.
    chain_q: the period of the chain the family could not pass, scanned first.  The lost-disk ring is not tried:
    its
    escape map cost 1-5 minutes per device and certified no request on QUASR or CoilStellaration."""
    sv = None
    if EDGEBOUND:
        try:
            if EDGEBOUND_METHOD == "exit":
                from .exitcurve import edge_bound_exit as _survivor
            else:
                from .edgebound import edge_bound_survivor as _survivor
            sv = _survivor(eng, source, T_last, request=request, deadline=deadline, axis=axis, note=note,
                           verbose=verbose)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            sv = dict(ok=False, kind="edge_survivor", error=repr(e), traceback=traceback.format_exc()[-3000:])
        if sv.get("ok") and request is not None and sv.get("holds_request"):
            return sv
        if sv.get("ok") and request is None and deadline is not None and time.time() > deadline - 120.0:
            return sv
    r = _edge_barrier_bound(eng, source, T_last, request=request, deadline=deadline, verbose=verbose, note=note,
                            chain_q=chain_q)
    if sv is None:
        return r
    if request is None and sv.get("ok"):
        # two valid upper bounds: the smaller one (the other in the record)
        if r.get("ok") and r.get("F_up") is not None and r["F_up"] < sv["F_up"]:
            r["survivor"] = {k: v for k, v in sv.items() if k not in ("traceback", "evidence", "curve")}
            return r
        sv["edge_barrier"] = {k: v for k, v in r.items() if k not in ("traceback", "interval")}
        return sv
    r["survivor"] = {k: v for k, v in sv.items() if k not in ("traceback", "evidence", "curve")}
    return r


def _edge_barrier_bound(eng, source, T_last, request=None, deadline=None, verbose=False, note="", chain_q=None):
    """torux.barrier.edge_barrier with at most 5 minutes (edge_bound's second stage)."""
    try:
        from .barrier import edge_barrier
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        return dict(ok=False, kind="edge_barrier", reason=f"torux.barrier unavailable: {e!r}")
    # a barrier that does not certify runs to its deadline (the orbit search), so it gets at most 5 minutes
    d5 = time.time() + 300.0
    deadline = d5 if deadline is None else min(deadline, d5)
    extra = [int(chain_q)] if chain_q else []
    try:
        r = edge_barrier(eng, source, T_last, request=request, deadline=deadline, verbose=verbose, try_ring=False,
                         extra_qs=extra, q_max=max([24] + [q + 1 for q in extra]))
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        return dict(ok=False, kind="edge_barrier", error=repr(e), traceback=traceback.format_exc()[-3000:])
    r["note"] = note
    r.setdefault("rigorous", True)
    return r


def edge_certificate(eng, cont, T_last, request, deadline=None, verbose=False, note="", nray=128, nrad=48,
                     stages=(4, 40, 400)):
    """Exterior escape bound (torux.escape) beyond the last torus: rays from the axis point in the plane
    phi = 0, start points from the last torus's distance out to the coils, traced in stages; the
    polygon outside which every sample escaped bounds every torus's flux.  Checked with twice the
    rays and radial samples."""
    from .escape import exterior_bound, polygon_flux_A
    src = eng.source
    pts = src.sample_points()
    Rc = np.hypot(pts[:, 0], pts[:, 1])
    box = (0.2 * float(np.min(Rc)) if len(Rc) else 0.1, 1.5 * float(np.max(Rc)) if len(Rc) else 10.0,
           1.5 * float(np.max(np.abs(pts[:, 2]))) if len(pts) else 10.0)
    from .escape import ray_crossings
    # rays from the centre of the last torus's section at phi = 0 (the m = 0 row there), starting at
    # the torus's own crossing of each ray (every larger torus encloses it) out to the coils
    th_s = 2 * np.pi * np.arange(2048) / 2048
    Rl, Zl = T_last.geometry_on(th_s, np.zeros(2048))
    c = np.array([float(np.sum(T_last.rc[0])), 0.0])
    poly_last = np.stack([Rl, Zl], -1)
    d_coil = float(np.min(np.hypot(Rc - c[0], pts[:, 2]))) if len(pts) else 3.0

    def radii(nr):
        thr = 2 * np.pi * (np.arange(nr) + 0.5) / nr
        rin = ray_crossings(poly_last, c, thr)
        rin = np.where(np.isfinite(rin), rin, np.nanmax(rin))
        return rin, np.maximum(d_coil, 1.2 * rin)
    rin1, rout1 = radii(nray)
    out = dict(kind="edge", note=note, F_last=abs(T_last.flux), centre=c.tolist(), r_in_max=float(np.max(rin1)),
               r_out=float(np.max(rout1)))
    # bulk tracing on a tabulated field over the region; the coils' neighbourhood (d_c from them) is
    # excluded (lines entering it are lost); its accuracy outside is measured and recorded
    from .field.grid import FieldGrid
    d_last = float(np.min(np.linalg.norm(pts[:, None, [0, 2]] * 0 + np.stack([np.hypot(pts[:, 0], pts[:, 1]),
                                                                              pts[:, 2]], -1)[:, None, :]
                                         - poly_last[None, ::16, :], axis=-1)))
    d_c = 0.25 * d_last
    Rlo, Rhi = max(1e-3, c[0] - 1.05 * float(np.max(rout1))), c[0] + 1.05 * float(np.max(rout1))
    Zhi = 1.05 * float(np.max(rout1))
    G = FieldGrid(eng, src.nfp, Rlo, Rhi, -Zhi, Zhi, nR=96, nphi=64, nZ=96, p=7, coil_pts=pts, d_c=d_c)
    out["grid"] = dict(box=[Rlo, Rhi, -Zhi, Zhi], d_c=d_c, check=G.check(npts=1500, min_dist=d_c + 0.02, coil_pts=pts),
                       excluded=G.excluded_fraction)
    res1 = exterior_bound(G, src.nfp, c, rin1, rout1, nray=nray, nrad=nrad, stages=stages, box=box, deadline=deadline)
    if not res1["complete"]:
        out.update(ok=False, reason="some ray has no escaping exterior within the traced length")
        return out
    F1, _ = polygon_flux_A(eng, res1["polygon"], 0.0)
    rin2, rout2 = radii(2 * nray)
    res2 = exterior_bound(G, src.nfp, c, rin2, rout2, nray=2 * nray, nrad=2 * nrad, stages=stages, box=box,
                          deadline=deadline)
    # the decisive escapes confirmed in the exact field: lines from just beyond r* on every ray
    from .escape import staged_escape
    th2 = res2["theta"]
    rs = res2["rstar"]
    probe = np.stack([c[0] + 1.0001 * rs * np.cos(th2), c[1] + 1.0001 * rs * np.sin(th2)], -1)
    lost_exact = staged_escape(eng, src.nfp, probe, 0.0, stages=(4, 40, max(stages)), box=box, deadline=deadline)
    out["exact_confirmed_fraction"] = float(np.mean(lost_exact >= 0))
    F2, _ = (polygon_flux_A(eng, res2["polygon"], 0.0) if res2["complete"] else (np.inf, 0))
    Fup = max(abs(F1), abs(F2))
    out.update(F_up=Fup, F_up_coarse=abs(F1), F_up_fine=abs(F2), stages=list(stages), nray=nray, nrad=nrad,
               refinement_rel_change=float(abs(abs(F2) - abs(F1)) / max(abs(F1), 1e-300)))
    out["ok"] = bool(res2["complete"] and out["refinement_rel_change"] < 0.05 and out["exact_confirmed_fraction"] == 1.0)
    out["holds_request"] = bool(Fup < request)
    # sampled escapes only: a torus could pass between sampled points; evidence, not a proof (the manifold
    # barrier of torux.barrier is the proof) -- never a 'no_surface' claim on its own
    out["rigorous"] = False
    out["interval"] = (Fup, np.inf)
    out["margin_rel"] = float((request - Fup) / request)
    return out


def start_resolution_probe(surf, iota, Mmin, Nmin, tol, Mb, Nb, cap=160, return_probe=False):
    from .design import sfl_guess_from_boozer
    from .family import LADDER
    g = sfl_guess_from_boozer(surf, iota, Mb, Nb)
    a = np.sqrt(g.rc ** 2 + g.zs ** 2)
    scale = max(np.max(a[1:]), 1e-300)
    rowmax = np.max(a, axis=1) / scale
    colmax = np.max(a, axis=0) / scale
    nmax = np.array([max(colmax[Nb + n], colmax[Nb - n]) for n in range(Nb + 1)])
    mneed = int(np.max(np.flatnonzero(rowmax > tol), initial=0)) + 2
    nneed = int(np.max(np.flatnonzero(nmax > tol), initial=0)) + 2
    if cap > 160 and (mneed >= Mb - 2 or nneed >= Nb - 2) and Nb < 256:
        # the full pass: the spectrum is still above tol at the probe's edge -- probe once more, wider
        return start_resolution_probe(surf, iota, Mmin, Nmin, tol, min(2 * Mb, cap), min(2 * Nb, 256), cap=cap,
                                      return_probe=return_probe)
    M = next((L for L in LADDER if L >= max(Mmin, mneed)), LADDER[-1])
    N = next((L for L in LADDER if L >= max(Nmin, nneed)), LADDER[-1])
    if return_probe:
        return min(M, cap), min(N, cap), g
    return min(M, cap), min(N, cap)


def start_resolution(surf, iota, Mmin=24, Nmin=24, tol=1e-9, cap=160, return_probe=False):
    """Smallest ladder (M, N) >= (Mmin, Nmin) at which the design surface's own spectrum in the
    solver's angles has decayed below tol (relative): the starting resolution of its Newton solve.
    return_probe: also return the probe torus (the design surface in the solver's angles at the probe's
    resolution, >= the returned one unless capped)."""
    from .design import sfl_guess_from_boozer
    from .family import LADDER
    # the probe resolution: twice the surface's own Fourier order (its spectrum in the solver's angles
    # extends a little beyond it), enlarged once if the tail has not decayed
    mp, nt = int(getattr(surf, "mpol", 16)), int(getattr(surf, "ntor", 16))
    Mb = next((L for L in LADDER if L >= max(32, 2 * mp + 8)), LADDER[-1])
    Nb = next((L for L in LADDER if L >= max(32, 2 * nt + 8)), LADDER[-1])
    g = sfl_guess_from_boozer(surf, iota, Mb, Nb)
    a = np.sqrt(g.rc ** 2 + g.zs ** 2)
    scale = max(np.max(a[1:]), 1e-300)
    rowmax = np.max(a, axis=1) / scale                      # per m
    colmax = np.max(a, axis=0) / scale                      # per n (index n + Nb)
    nmax = np.array([max(colmax[Nb + n], colmax[Nb - n]) for n in range(Nb + 1)])
    mneed = int(np.max(np.flatnonzero(rowmax > tol), initial=0)) + 2
    nneed = int(np.max(np.flatnonzero(nmax > tol), initial=0)) + 2
    if (mneed >= Mb - 2 or nneed >= Nb - 2) and (Mb, Nb) != (96, 128):
        from . import fast_design as _so

        def _reach(v):
            # where the probe's own decay, fitted over its last half, crosses tol (inf when it does not decay)
            k = np.arange(len(v)); lo = max(1, len(v) // 2)
            vv = np.maximum(v[lo:], 1e-300)
            sl, ic = np.polyfit(k[lo:], np.log(vv), 1)
            return (np.log(tol) - ic) / sl if sl < -1e-3 else np.inf
        if _so.on("probe") and Mb < 64 and Nb < 64 and max(_reach(rowmax), _reach(nmax)) < 56:
            # (fast_design 'probe') a (64, 64) probe before (96, 128) (79k points, 2.4x as many): where its own
            # spectrum has decayed inside it, its decision is the (96, 128) probe's (the same spectrum up to the
            # smaller probe's aliasing, far below tol)
            g = sfl_guess_from_boozer(surf, iota, 64, 64)
            a = np.sqrt(g.rc ** 2 + g.zs ** 2)
            scale = max(np.max(a[1:]), 1e-300)
            rowmax = np.max(a, axis=1) / scale
            colmax = np.max(a, axis=0) / scale
            nmax = np.array([max(colmax[64 + n], colmax[64 - n]) for n in range(65)])
            mneed = int(np.max(np.flatnonzero(rowmax > tol), initial=0)) + 2
            nneed = int(np.max(np.flatnonzero(nmax > tol), initial=0)) + 2
            if mneed < 62 and nneed < 62:
                M = next((L for L in LADDER if L >= max(Mmin, mneed)), LADDER[-1])
                N = next((L for L in LADDER if L >= max(Nmin, nneed)), LADDER[-1])
                if return_probe:
                    return min(M, cap), min(N, cap), g
                return min(M, cap), min(N, cap)
        return start_resolution_probe(surf, iota, Mmin, Nmin, tol, 96, 128, cap=cap, return_probe=return_probe)
    M = next((L for L in LADDER if L >= max(Mmin, mneed)), LADDER[-1])
    N = next((L for L in LADDER if L >= max(Nmin, nneed)), LADDER[-1])
    if return_probe:
        return min(M, cap), min(N, cap), g
    return min(M, cap), min(N, cap)


def section_flux(engine, t, ntheta=64, phi=None):
    """Toroidal flux of torus t through its section at phi (default phi0): minus the loop integral of A
    along the section (ntheta-point trapezoid rule, spectral for the smooth periodic integrand), the
    convention of TorusProblem.flux_of (which averages the same integral over the grid's sections).  For an
    invariant torus every section carries the same flux; for a guess it fixes the sign (fast_design
    'fluxsign')."""
    th = 2 * np.pi * np.arange(ntheta) / ntheta
    ph = np.array([float(t.phi0 if phi is None else phi)])
    R, Z = (v[:, 0] for v in t.geometry_grid(th, ph))
    m = np.arange(t.M + 1)[:, None]
    # R_theta is the sine series of -m rc, Z_theta the cosine series of m zs: geometry_grid of the
    # torus (rc', zs') = (m zs, -m rc) returns (Z_theta, R_theta)
    d = TorusState(m * np.asarray(t.zs), -m * np.asarray(t.rc), t.iota, t.flux, t.M, t.N, t.nfp, phi0=t.phi0)
    Zt, Rt = (v[:, 0] for v in d.geometry_grid(th, ph))
    c, s = math.cos(ph[0]), math.sin(ph[0])
    x = np.stack([R * c, R * s, Z], -1)
    A = np.asarray(engine.evaluate(x, grad=False, A=True)["A"])
    AR = A[:, 0] * c + A[:, 1] * s
    return float(-2 * np.pi * np.mean(AR * Rt + A[:, 2] * Zt))


class _Skip(Exception):
    """A stage skipped by the policy (POLICY)."""


def request_class(why, bn_design, late_bn=1e-4):
    """The class of a request on the resonant path, known after the design Newton: "newton_failed" (no converged
    design torus), "near" (converged, off-grid B.n below late_bn), "far" (converged, B.n at or above it),
    "refined" (the design torus refined to its plateau: the guard failed or B.n stayed above the surface bar)."""
    if why != "design Newton failed":
        return "refined"
    if bn_design is None or not np.isfinite(bn_design):
        return "newton_failed"
    return "near" if bn_design < late_bn else "far"


def stage_applies(policy, key, cls):
    """POLICY[key] is "all" (the stage runs for every class), a tuple of classes, or False / True."""
    mode = policy.get(key, "all")
    if mode is True or mode == "all":
        return True
    if not mode:
        return False
    return cls in mode


def late_stages_apply(policy, why, bn_design):
    """True when the resonant path's late stages (bracket, band between its tori, edge barrier, the fallback
    continuation) run for this request's class (request_class; POLICY['late']), else the reason they do not (the
    request is then left undetermined)."""
    cls = request_class(why, bn_design, policy.get("late_bn", 1e-4))
    if stage_applies(policy, "late", cls):
        return True
    return (f"late stages (bracket, band, edge barrier, continuation) not run for class '{cls}' "
            f"(design B.n {bn_design:.1e}; {why})")


QUICK_MAX_UNKNOWNS = 30000     # quick mode: design tori above this size are deferred (about M, N = 80, 96)
QUICK_RAISES = 4               # quick mode: at most this many resolution raises of the design torus
DESIGN_PATIENCE = 4            # the full path's design retry stops after this many Newton iterations without halving


@_budget.budgeted(extra="range_budget_s")
def analyse_design(source, surfaces, flux=None, iota_hints=None, dmin=None, tol=3e-15, verbose=False,
                   Mstart=24, Nstart=24, budget_s=None, fallback=True, region_pts="design", profile=True,
                   range_budget_s=0.0, resonant_path=True):
    """Fast path for sources published with design surfaces in straight-field-line (Boozer-like)
    angles (QUASR): the axis from the innermost surface's m = 0 part, then a torus at each design
    surface's flux by Newton from the surface itself (torux.design.sfl_guess_from_boozer), the
    outermost refined to the arithmetic limit and guarded.  Falls back to analyse() (continuation
    from the axis, chains, jumps) where the Newton from the surface fails.

    surfaces: design surfaces, innermost first.  flux: the request (default: the outermost
    surface's planar-section flux).  iota_hints: optional per-surface |iota| (e.g. the database's);
    the sign is taken from the axis."""
    from .design import sfl_guess_from_boozer
    from .torus import TorusProblem
    from . import fast_design as _so
    t0 = time.time()
    deadline = t0 + budget_s if budget_s else None
    out = dict(id=source.id, nfp=source.nfp, request_flux=flux, status="error", timings={}, path="design")
    if _so.active():
        out["fast_design"] = _so.active()
    try:
        if dmin is None:
            dmin = source.meta.get("dmin", 0.05)
        if isinstance(region_pts, str) and region_pts == "design":
            from torux.io.surfaces import surface_grid
            r_, _, _, _ = surface_grid(surfaces[-1], nphi=64, ntheta=64, full_torus=True)
            region_pts = r_.reshape(-1, 3)
        eng = FieldEngine(source, dmin=dmin, region_pts=region_pts)
        out["engine"] = dict(nodes=eng.info.n_nodes, dmin=dmin, per_coil=region_pts is not None)
        nfp = source.nfp
        # axis guess from the innermost surface (iota irrelevant for the m = 0 row)
        g0 = sfl_guess_from_boozer(surfaces[0], 0.0, 4, 16)
        ax_guess = (g0.rc[0, 16:], -g0.zs[0, 16:])
        ax = AX.solve_axis(eng, nfp, float(g0.rc[0, 16]), N=16, guess=ax_guess)
        if ax.converged:
            AX.spectral_axis_monodromy(eng, ax)
        ax0 = ax if ax.converged and np.isfinite(ax.residue) else None
        axes = [ax] if ax.converged else []
        if not (ax.converged and 0 < ax.residue < 1):
            axes = find_axes(eng, source)
        ell = [a for a in axes if 0 < a.residue < 1]
        inv = [a for a in axes if a is not ax0 and np.isfinite(a.residue) and a.residue >= 1]
        if not ell and (inv or (ax0 is not None and ax0.residue >= 1)):
            # the design's period-one line is inverse hyperbolic (rotation 1/2 per period: QUASR designs with the
            # axis transform at a half-integer per period).  A flux surface may enclose it (its fixed-point index is
            # +1, as an elliptic line's); the design path proceeds with it as the line every surface here encloses
            # (topology checks), the transform's sign taken from the design torus itself below.  The design's own
            # line (the scan's copy of it, resolved to its spectrum, when the scan has it), else the scan's line
            # nearest the design's axis (the no-axis diagnosis: 11 QUASR designs whose only symmetric period-one
            # line has residue 1.000-1.030, reached by the scan's traced guesses)
            phs = np.linspace(0, 2 * np.pi / nfp, 33)[:-1]

            def dist(a, b):
                Ra, Za = a.at(phs)
                Rb, Zb = b.at(phs)
                return float(max(np.max(np.abs(Ra - Rb)), np.max(np.abs(Za - Zb))))
            R_d = float(ax0.rc[0]) if ax0 is not None else float(g0.rc[0, 16])
            same = 1e-4 * abs(R_d)                      # distinct period-one lines lie much further apart
            near = min(inv, key=lambda a: abs(float(a.rc[0]) - R_d)) if inv else None
            if ax0 is not None and ax0.residue >= 1 and (near is None or dist(near, ax0) > same):
                pick = ax0
            else:
                pick = near
            ell = [pick]
            axes = [pick] + [a for a in axes if a is not pick and dist(a, pick) > same]
            out["axis_kind"] = "inverse_hyperbolic"
        from . import axisselect as _axsel
        if _axsel.on():
            # (AXSEL) D far outside the innermost design surface, or no elliptic line: the period-one line inside it
            # (the design solve -- a stalled one validated --, find_axes' lines, the line traced through the surface's
            # centre; torux/axisselect.py), elliptic before inverse hyperbolic; otherwise nothing changes
            ell, axes = _axsel.repair(eng, source, surfaces[0], ax, ell, axes, out, find_axes,
                                      called_find_axes=not (ax.converged and 0 < ax.residue < 1))
        out["axes"] = [dict(rc=a.rc.tolist(), zs=a.zs.tolist(), residue=a.residue, iota=a.iota, res=a.res) for a in axes]
        out["timings"]["axis"] = time.time() - t0
        if not ell:
            out["status"] = no_axis_status(out)
            return out
        ax = ell[0]
        sgn = math.copysign(1.0, ax.iota) if np.isfinite(ax.iota) and ax.iota != 0 else -1.0
        cont = Continuation(eng, nfp, verbose=verbose)
        if _so.on("predict"):
            cont.predict_state = {}
        if _so.on("krylov"):
            cont.DENSE_MAX = 0          # (fast_design 'krylov') Newton-Krylov at every size, dense as its fallback
        if out.get("axis_kind") == "inverse_hyperbolic" and iota_hints is not None:
            # the sign of the transform: the design torus at a low resolution for both signs, the smaller residual
            io_h = abs(float(iota_hints[-1]))
            res_s = {}
            for s_try in (-1.0, 1.0):
                st_try = sfl_guess_from_boozer(surfaces[-1], s_try * io_h, 16, 16)
                st_try.flux = cont.problem(16, 16).flux_of(st_try.rc, st_try.zs)
                res_s[s_try] = float(cont.solve(st_try, st_try.flux, 16, 16, maxit=4).res)
            sgn = min(res_s, key=lambda k: res_s[k] if np.isfinite(res_s[k]) else np.inf)
            out["iota_sign_trial"] = {str(k): v for k, v in res_s.items()}
        if not resonant_path:
            # quick mode: no dense Newton behind a failed Newton-Krylov above 6000 unknowns (a torus that far
            # from invariant at that resolution is deferred anyway; the fallback cost 17 LUs of n = 7900)
            cont.DENSE_FALLBACK = 6000
        tori = []
        # profile=False: only the outermost design surface (the request); the inner ones sample
        # iota(flux) for the range pass
        for k, surf in (enumerate(surfaces) if profile else [(len(surfaces) - 1, surfaces[-1])]):
            io = sgn * abs(iota_hints[k]) if iota_hints is not None else (tori[-1].iota if tori else ax.iota)
            outer = k == len(surfaces) - 1
            # the full pass may start as high as the design surface's spectrum needs (QUASR 0136030: N = 256)
            probe_t = None
            if _so.on("probe"):
                Mg, Ng, probe_t = start_resolution(surf, io, Mstart, Nstart, cap=256 if (resonant_path and outer) else 160,
                                                   return_probe=True)
            else:
                Mg, Ng = start_resolution(surf, io, Mstart, Nstart, cap=256 if (resonant_path and outer) else 160)
            if outer and not resonant_path and 2 * (Mg + 1) * (2 * Ng + 1) > QUICK_MAX_UNKNOWNS:
                # quick mode: a design surface whose own spectrum needs this many unknowns is an expensive
                # torus whatever happens (QUASR 2093639: M, N = 112, 160, 446 s of Newton-Krylov); deferred
                out["status"] = "undetermined"
                out["deferred"] = True
                out["deferred_reason"] = f"design surface needs M, N = {Mg}, {Ng}"
                out["timings"]["total"] = time.time() - t0
                return out
            if probe_t is not None and probe_t.M >= Mg and probe_t.N >= Ng:
                # (fast_design 'probe') the probe's torus truncated: the same angle map on a finer grid
                st = probe_t.resized(Mg, Ng)
            else:
                st = sfl_guess_from_boozer(surf, io, Mg, Ng)
            prob = cont.problem(Mg, Ng)
            if outer and flux is not None and _so.on("fluxsign"):
                # (fast_design 'fluxsign') only the sign of the guess's flux is used here: one section suffices
                f_s = math.copysign(abs(flux), section_flux(eng, st))
            else:
                f_s = prob.flux_of(st.rc, st.zs)
                if k == len(surfaces) - 1 and flux is not None:
                    f_s = math.copysign(abs(flux), f_s)
            st.flux = f_s
            # no Newton patience here: a patience of 4 stops one design solve in ten that would converge; slow
            # starts (several damped steps without halving) are common from a design surface
            sol = cont.solve(st, f_s, Mg, Ng, maxit=20 if outer else 10)
            if outer and not sol.converged and resonant_path and not _budget.expired():
                # under-resolution before resonance: the torus at the request may need more modes than the design
                # surface's own spectrum shows (QUASR 2439098: the Newton diverges at 24 x 24 and converges
                # quadratically at 48 x 48 from the same guess, 7 iterations); once more at twice the resolution.
                # Full path only: the retry seldom converges, seconds quick mode need not spend
                from .family import LADDER
                M2 = next((L for L in LADDER if L >= 2 * Mg), LADDER[-1])
                N2 = next((L for L in LADDER if L >= 2 * Ng), LADDER[-1])
                if (M2, N2) != (Mg, Ng):
                    t_r = time.time()
                    st2 = sfl_guess_from_boozer(surf, io, M2, N2)
                    st2.flux = f_s
                    sol2 = cont.solve(st2, f_s, M2, N2, maxit=12, patience=DESIGN_PATIENCE)
                    out["design_retry"] = dict(M=M2, N=N2, converged=bool(sol2.converged), its=len(sol2.info.get("hist", [])),
                                               wall=time.time() - t_r, first=dict(M=Mg, N=Ng, its=len(sol.info.get("hist", []))))
                    if sol2.converged:
                        sol, st, Mg, Ng = sol2, st2, M2, N2
            if sol.converged and outer:
                # quick mode (no resonant path): a raise that gains less than 3x means a torus through a chain
                # or a chaotic layer, and an invariant torus at the request is below the continuation level
                # within four raises from the design resolution; the rest is deferred to the certificate pass
                jkw = {"jump": True} if _so.on("jump") else {}
                sol = (cont.raise_until(sol, f_s, cont.bn_cont, **jkw) if resonant_path
                       else cont.raise_until(sol, f_s, cont.bn_cont, max_raise=QUICK_RAISES, min_gain=3.0, low_gain_patience=2,
                                             **jkw))
            # inner design surfaces only sample iota(flux) (no refinement: the profile, not the request)
            tori.append(sol)
            if verbose:
                print(f"  surface {k}: flux {f_s:.10e} conv {sol.converged} iota {sol.iota:.12f} bn {sol.bn_off:.2e}")
        out["profile"] = [torus_record(t) for t in tori]
        out["timings"]["surfaces"] = time.time() - t0
        last = tori[-1]
        def resonant(shape_t, bn_fail, why):
            """The request's torus could not be verified from its design surface (Newton failed, or the
            refined torus sits in a near-resonant chain): tori at fixed transforms just inside and beyond
            it from the same shape (no continuation through the resonances on the way) and the island band
            of the chain between them; else the continuation from the axis and the known tori."""
            req = abs(last.flux) if flux is None else abs(flux)
            out["resonant_path"] = why
            policy = dict(POLICY)
            out["policy"] = dict(config=policy, stages=[], cls=request_class(why, bn_fail, policy.get("late_bn", 1e-4)))

            def stage(name, t_s, **kw):
                out["policy"]["stages"].append(dict(stage=name, t=round(t_s - t0, 2), dt=round(time.time() - t_s, 2),
                                                   **kw))

            inner_done = []

            def ensure_inner():
                """The inner design surfaces (skipped without the profile) seed the continuation and the bracket.
                (solved only when those run -- the seeds' base polish and the proxy band do not use them)"""
                if inner_done:
                    return
                inner_done.append(True)
                if not profile and len(surfaces) > 1 and len(tori) == 1:
                    t_s = time.time()
                    inner = []
                    for k, surf in enumerate(surfaces[:-1]):
                        if deadline is not None and time.time() > deadline - 60:
                            break
                        io = sgn * abs(iota_hints[k]) if iota_hints is not None else (inner[-1].iota if inner else ax.iota)
                        Mg, Ng = start_resolution(surf, io, Mstart, Nstart)
                        st_k = sfl_guess_from_boozer(surf, io, Mg, Ng)
                        st_k.flux = cont.problem(Mg, Ng).flux_of(st_k.rc, st_k.zs)
                        inner.append(cont.solve(st_k, st_k.flux, Mg, Ng, maxit=10))
                    tori[:0] = inner
                    out["profile"] = [torus_record(t) for t in tori]
                    stage("inner_tori", t_s, n=len(inner))
            if not policy.get("lazy_inner"):
                ensure_inner()
            shape = TorusState(np.asarray(shape_t.rc), np.asarray(shape_t.zs),
                               float(shape_t.iota if np.isfinite(shape_t.iota) else st.iota),
                               float(st.flux), int(shape_t.M), int(shape_t.N), nfp, bn_off=bn_fail)
            T1 = T2 = None
            # the stages before the certificates (proxy band, seeds, bracket) stop early enough to leave the edge
            # barrier and the fallback continuation (whose own certificate is the one that settles edge requests:
            # QUASR 0261318's -1/13 barrier needed 219 s) a reserve: min(900 s, 35 % of what is left)
            late = late_stages_apply(policy, why, bn_fail)
            if late is True or policy.get("tail_s") is None:
                d_pre = None if deadline is None else deadline - min(RESONANT_RESERVE, 0.35 * max(deadline - time.time(), 0.0))
            else:
                # (POLICY['tail_s'] set) no late stage runs for this class: the reserve kept for them goes to
                # the stages before
                d_pre = None if deadline is None else deadline - min(float(policy.get("tail_s", 30.0)),
                                                                     0.1 * max(deadline - time.time(), 0.0))

            def seeds_stage(name, **kw):
                """torux.seeds.seed_and_walk with 40 % of the time to d_pre; True when it returned the surface."""
                try:
                    from .seeds import seed_and_walk, surface_record
                    t_s = time.time()
                    sd_deadline = None if d_pre is None else time.time() + 0.4 * (d_pre - time.time())
                    T_s, slog = seed_and_walk(eng, source, req, shape, ax, deadline=sd_deadline, verbose=verbose,
                                              tol=tol, dmin=dmin, **kw)
                    out[name] = slog
                    stage(name, t_s, outcome=slog.get("outcome"))
                    if T_s is not None:
                        out["surface"], out["guard"] = surface_record(T_s, slog)
                        out["refine_hist"] = out["surface"].pop("refine_hist", None)
                        out["status"] = "surface"
                        out["path"] = "design->seeds"
                        out["timings"]["total"] = time.time() - t0
                        return True
                except Exception as e:  # noqa: BLE001
                    if environment_failure(e):
                        raise
                    out[name + "_error"] = repr(e)
                    out[name + "_traceback"] = traceback.format_exc()
                return False
            # the seeds' base polish first: the invariant circle at the requested flux by Newton from the
            # design torus's section and the torus from it (no tracing; median 17 s when it succeeds, 9 s when not)
            # gives most of the surfaces the seeds find, in far less time than the proxy band takes
            seeds_split = policy.get("seeds_first") and (deadline is None or time.time() < deadline - 120) \
                and not policy.get("skip_surface_stages")
            cls = out["policy"]["cls"]

            edge_type = []

            def probe_seconds():
                probe_s = float(policy.get("cont_probe_s") or 0.0)
                if policy.get("cont_probe_share"):
                    probe_s = max(probe_s, float(policy["cont_probe_share"]) * (budget_s or 0.0))
                if policy.get("skip_surface_stages"):
                    return 0.0
                if edge_type and fallback:
                    # an edge-type request: the continuation and its edge barrier get the time to d_pre
                    return max(probe_s, (d_pre - time.time()) if d_pre is not None else probe_s)
                return probe_s if (fallback and stage_applies(policy, "cont_probe", cls)) else 0.0

            def edge_triage():
                """lines from the outboard symmetry line at fractions s of the design torus's outboard
                distance from the axis (plane phi = 0), traced a few hundred periods with the compiled tracer
                (torux.seeds.SectionTracer): a line from inside the design surface that is lost (B_phi turns or the line
                leaves through the coils) marks the request as edge type -- the torus at the request, were it there and
                near the design surface, would hold that line.  It only chooses stages; nothing is claimed from it."""
                if not policy.get("edge_triage") or (deadline is not None and time.time() > deadline - 120):
                    return False
                t_s = time.time()
                try:
                    from .seeds import SectionTracer, trace_engine, bphi_sign
                    Rax = float(ax.at(np.array([0.0]))[0][0])
                    Rs, _ = shape.geometry_on(np.array([0.0, np.pi]), np.array([0.0, 0.0]))
                    Rb = float(np.max(Rs))
                    if not (np.isfinite(Rb) and Rb > Rax):
                        return False
                    s_v = np.asarray(policy.get("edge_triage_s", (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)), float)
                    tr = SectionTracer(trace_engine(eng, source), nfp, ax, sign=bphi_sign(eng, ax))
                    tr.calibrate(np.stack([Rax + (Rb - Rax) * np.array([0.5, 0.8, 0.9]), np.zeros(3)], -1), ks=(6,))
                    y0 = np.stack([Rax + (Rb - Rax) * s_v, np.zeros(len(s_v))], -1)
                    nper = int(policy.get("edge_triage_periods", 400))
                    o = tr.orbits(y0, nper, deadline=time.time() + 60.0)
                    lost = np.asarray(o["lost"], bool)
                    out["edge_triage"] = dict(s=s_v.tolist(), lost=lost.tolist(), periods=int(o["n"]),
                                              wall=round(time.time() - t_s, 2), edge=bool(lost.any()))
                    stage("edge_triage", t_s, edge=bool(lost.any()), lost=int(lost.sum()))
                    return bool(lost.any())
                except Exception as e:  # noqa: BLE001
                    if environment_failure(e):
                        raise
                    out["edge_triage_error"] = repr(e)
                    return False

            def cont_probe(after_proxy=False):
                """the continuation from the inner design tori with a short budget before the certificates:
                where the design Newton fails at the request, the family from the inner design surfaces often
                reaches it in seconds.  Its
                acceptance is analyse()'s, the fallback's.  Returns analyse()'s record when it answers, else None."""
                probe_s = probe_seconds()
                if after_proxy and deadline is not None:
                    # after the proxy band: its share, or what is left before d_pre (at least 30 s)
                    probe_s = min(probe_s, d_pre - time.time())
                    if probe_s < 30:
                        return None
                if not (fallback and probe_s > 0 and (stage_applies(policy, "cont_probe", cls) or edge_type) and
                        (deadline is None or time.time() < d_pre - probe_s + (5.0 if after_proxy else 0.0))):
                    return None
                ensure_inner()
                t_s = time.time()
                try:
                    rp = analyse(source, flux=req, dmin=dmin, tol=tol, verbose=verbose, budget_s=probe_s,
                                 region_pts=region_pts, start_tori=[t for t in tori[:-1]], axis_guesses=[(ax.rc, ax.zs)])
                except Exception as e:  # noqa: BLE001
                    if environment_failure(e):
                        raise
                    rp = dict(status="error", error=repr(e), traceback=traceback.format_exc())
                stage("cont_probe", t_s, status=rp.get("status"), family_stop=rp.get("family_stop"), budget=probe_s)
                out["continuation_probe"] = {k: rp.get(k) for k in ("status", "family_stop", "timings", "error",
                                                                     "continuation_start", "jumps", "refine_stop")}
                if rp.get("status") == "surface" or (rp.get("status") == "no_surface" and (rp.get("certificate") or {})
                                                     .get("ok") and rp["certificate"].get("holds_request")):
                    from . import axisselect as _axsel
                    if _axsel.on() and _axsel.opts().get("check_continuation") and \
                            not _axsel.continuation_axis_ok(rp, ax, surfaces[0]):
                        # (AXSEL) the probe's family started from another line than D: not an answer about D's tori
                        out["continuation_probe"]["other_axis"] = True
                        return None
                    rp["path"] = "design->continuation probe"
                    rp["design_profile"] = out["profile"]
                    for k in ("resonant_path", "design_surface", "axis_kind", "iota_sign_trial", "seeds", "seeds_error",
                              "policy", "axis_select"):
                        if k in out:
                            rp[k] = out[k]
                    return rp
                return None
            if policy.get("cont_probe_first"):
                rp = cont_probe()
                if rp is not None:
                    return rp
            # the seeds' base polish first: the invariant circle at the requested flux by Newton from the
            # design torus's section and the torus from it (no tracing; median 17 s when it succeeds, 9 s when not)
            # gives most of the surfaces the seeds find, in far less time than the proxy band takes
            if seeds_split:
                if seeds_stage("seeds", trace=False):
                    return out
            # (POLICY 'edge_triage') an edge-type request skips the proxy band; its early continuation gets the time to
            # d_pre (in either stage order)
            if policy.get("edge_triage") and stage_applies(policy, "proxy", cls) and edge_triage():
                edge_type.append(True)
            if not policy.get("cont_probe_first") and not policy.get("proxy_first"):
                rp = cont_probe(after_proxy=bool(edge_type))
                if rp is not None:
                    return rp

            def rho_hint():
                """the rotation (per period) of the seeds' base-polish circle at the requested flux, when its
                Newton ran (converged or stalled): the chain it stalls on is the chain at the request"""
                if not policy.get("rho_chains"):
                    return None
                for a_ in ((out.get("seeds") or {}).get("attempts") or []):
                    if str(a_.get("how", "")).startswith("base polish") and a_.get("rho") is not None \
                            and np.isfinite(a_["rho"]):
                        return float(a_["rho"])
                return None
            # the chain the request sits in (rational-target designs, thin chains): the manifold island band on a
            # proxy annulus around the design torus's closed lines -- no bracket needed (minutes, where the
            # bracket's high-resolution tori took tens of minutes)
            try:
                t_s = time.time()
                if not stage_applies(policy, "proxy", cls) or edge_type:
                    # no proxy band for this class (POLICY 'proxy'): a 'refined' request (a near-invariant torus at
                    # the request: no chain holds it) almost never gets a band; nor for edge-type requests
                    # (edge_triage), which an edge barrier settles far more often than an island band
                    raise _Skip()
                # proxy first: its deadline keeps the early continuation's share before d_pre
                d_proxy = d_pre
                if policy.get("proxy_first") and d_pre is not None:
                    d_proxy = d_pre - probe_seconds()
                pc, patt = chain_band_by_proxy(eng, shape, req, deadline=d_proxy, verbose=verbose,
                                               n_chains=int(policy.get("n_chains", 3)), rho_hint=rho_hint(), axis=ax,
                                               traced=bool(policy.get("traced_lines")),
                                               axis_inverse=bool(policy.get("inverse_axis")
                                                                 and out.get("axis_kind") == "inverse_hyperbolic"),
                                               annulus=str(policy.get("annulus", "standard")))
                out["proxy_band_attempts"] = patt
                stage("proxy_band", t_s, ok=pc is not None, attempts=len(patt),
                      **({"rho_hint": rho_hint()} if policy.get("rho_chains") else {}))
                if pc is not None:
                    out["certificate"] = pc
                    out["status"] = "no_surface"
                    out["path"] = "design->proxy band"
                    out.pop("surface", None)
                    out["timings"]["total"] = time.time() - t0
                    return out
            except _Skip:
                stage("proxy_band", t_s, skipped=True)
            except Exception as e:  # noqa: BLE001
                if environment_failure(e):
                    raise
                out["proxy_band_error"] = repr(e)
            if policy.get("proxy_first") and not policy.get("cont_probe_first"):
                rp = cont_probe(after_proxy=True)
                if rp is not None:
                    return rp
            # the torus at the request by other starts (torux.seeds): the invariant circle of the period map at the
            # requested flux by Newton from the design torus's section (no tracing), else regular orbits traced
            # around the request (circle seeds, walks); accepted only at the arithmetic limit with the reference
            # check
            if (deadline is None or time.time() < deadline - 120) and \
                    stage_applies(policy, "seeds_trace", request_class(why, bn_fail, policy.get("late_bn", 1e-4))):
                if seeds_stage("seeds_trace" if seeds_split else "seeds",
                               **({"base_polish": False} if seeds_split else {})):
                    return out
            if late is not True:
                # the bracket, its band / edge certificates and the fallback continuation do not apply
                # to this class at this budget (POLICY 'late'): the request is left undetermined
                out["policy_stop"] = late
                out["deferred"] = True
                out["deferred_reason"] = f"policy: {late}"
                stage("stop", time.time(), reason=late)
                return None
            ensure_inner()
            inner_ok = [t for t in tori[:-1] if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-6]
            io_in = inner_ok[-1].iota if inner_ok else ax.iota
            try:
                T1, T2, blog = local_bracket(cont, shape, req, io_in, deadline=d_pre, verbose=verbose)
                out["bracket"] = dict(log=blog, T1=None if T1 is None else torus_record(T1),
                                      T2=None if T2 is None else torus_record(T2))
                T_in = T1 if T1 is not None else (inner_ok[-1] if inner_ok else None)
                if T2 is None and T_in is not None and (deadline is None or time.time() < deadline - 60):
                    # no torus beyond the request: the edge barrier from the outermost torus inside
                    eb = edge_bound(eng, source, T_in, request=req, deadline=deadline, verbose=verbose,
                                    note="design request: no torus found beyond it", axis=ax)
                    out["edge_barrier"] = {k: v for k, v in eb.items() if k != "traceback"}
                    if eb.get("ok") and eb.get("holds_request") and eb.get("rigorous", True):
                        out["certificate"] = eb
                        out["status"] = "no_surface"
                        out["path"] = "design->bracket->edge"
                        out.pop("surface", None)
                        out["timings"]["total"] = time.time() - t0
                        return out
                if T1 is not None and T2 is not None:
                    cert = band_between(eng, cont, T1, T2, req, tol=tol, deadline=deadline, verbose=verbose, axis=ax)
                    if cert is not None:
                        out["certificate"] = cert
                        if cert.get("ok") and cert.get("holds_request") and cert.get("rigorous", True):
                            out["status"] = "no_surface"
                            out["path"] = "design->bracket"
                            out.pop("surface", None)      # an unverified torus is not the answer
                            out["timings"]["total"] = time.time() - t0
                            return out
            except Exception as e:  # noqa: BLE001  (recorded; the continuation below still runs)
                if environment_failure(e):
                    raise
                out["bracket_error"] = repr(e)
                out["bracket_traceback"] = traceback.format_exc()
            if fallback and policy.get("continuation", True) and (deadline is None or time.time() < deadline - 30):
                starts = [t for t in tori[:-1]] + ([T1] if T1 is not None else [])
                # the axis this path found seeds the continuation's own axis search
                r = analyse(source, flux=req, dmin=dmin, tol=tol,
                            verbose=verbose, budget_s=None if deadline is None else max(60.0, deadline - time.time()),
                            region_pts=region_pts, start_tori=starts, axis_guesses=[(ax.rc, ax.zs)])
                r["path"] = "design->bracket->continuation"
                r["design_profile"] = out["profile"]
                from . import axisselect as _axsel
                if _axsel.on() and _axsel.opts().get("check_continuation") and \
                        r.get("status") in ("surface", "no_surface") and \
                        not _axsel.continuation_axis_ok(r, ax, surfaces[0]):
                    # (AXSEL) the continuation's family started from another line than D (an inverse-hyperbolic D has
                    # no elliptic copy: analyse() takes the elliptic line nearest it): not an answer about D's tori
                    r["continuation_status"] = f"other axis ({r['status']})"
                    r["status"] = "undetermined"
                if r.get("status") == "no_axis":
                    # the continuation needs an elliptic axis to start from; the design path had a period-one line
                    # (an inverse-hyperbolic one) that surfaces may enclose: not an answer, undetermined
                    r["continuation_status"] = "no_axis"
                    r["status"] = "undetermined"
                for k in ("bracket", "bracket_error", "bracket_traceback", "resonant_path", "design_surface",
                          "proxy_band_attempts", "proxy_band_error", "axis_kind", "iota_sign_trial", "seeds",
                          "seeds_error", "seeds_trace", "seeds_trace_error", "policy", "axis_select"):
                    if k in out:
                        r[k] = out[k]
                if "certificate" in out:
                    # the band attempt between the bracket's tori (kept whatever the continuation then found)
                    r["bracket_certificate"] = {k: v for k, v in out["certificate"].items() if k != "traceback"}
                if "edge_barrier" in out:
                    r["bracket_edge_barrier"] = out["edge_barrier"]
                return r
            return None

        if not (last.converged and last.bn_off < 1e-6) and not resonant_path:
            # quick mode: the request's torus does not come from its design surface; deferred to the
            # certificate pass (bracket, island band, edge barrier, continuation)
            out["status"] = "undetermined"
            out["deferred"] = True
            out["timings"]["total"] = time.time() - t0
            return out
        if POLICY.get("force_resonant") and resonant_path:
            # (soundness tests only) the resonant path whatever the design torus: on requests with a known
            # surface no certificate may hold the request (POLICY 'force_resonant'; with 'skip_surface_stages' the
            # seeds and the early continuation are skipped so that every certificate the path has is tried)
            out["forced_resonant"] = dict(design_converged=bool(last.converged), design_bn=float(last.bn_off))
            shape_t = last if (last.converged and np.isfinite(last.iota)) else st
            bn_fail = float(last.bn_off) if (last.converged and np.isfinite(last.bn_off)) else np.nan
            r = resonant(shape_t, bn_fail, "design Newton failed")
            if r is not None:
                return r
            out["status"] = "undetermined"
            return out
        if not (last.converged and last.bn_off < 1e-6):
            shape_t = last if (last.converged and np.isfinite(last.iota)) else st
            bn_fail = float(last.bn_off) if (last.converged and np.isfinite(last.bn_off)) else np.nan
            r = resonant(shape_t, bn_fail, "design Newton failed")
            if r is not None:
                return r
            out["status"] = "undetermined"
            return out
        rc_ = lean_continuation(cont, source, last, dmin)
        out["engine"]["refine_nodes"] = rc_.engine.info.n_nodes
        if getattr(rc_.engine, "node_info", None):
            out["engine"]["refine_nodes_model"] = rc_.engine.node_info["model_nodes"]
        caps = FULL_CAPS if resonant_path else (320, 320)
        best = refine_to_limit(rc_, last, last.flux, target=tol, deadline=deadline, verbose=verbose, Mcap=caps[0],
                               Ncap=caps[1])
        q_ = rc_.problem(best.M, best.N).quadrature_check(best.rc, best.zs)
        if q_ > 1e-8:
            # the torus nearer a coil than the engine's dmin: refined again on a near-coil engine (near_coil_retry)
            nc = near_coil_retry(cont, source, best, dmin, tol, deadline=deadline, caps=caps)
            out["near_coil"] = dict(nc[3] if nc is not None else {}, ok=nc is not None, quad_before=float(q_))
            if nc is not None:
                rc_, best, q_ = nc[0], nc[1], nc[2]
        out["timings"]["refine"] = time.time() - t0
        conv_tori = [t for t in tori[:-1] if t.converged] + [best]
        if len(conv_tori) >= 2:
            a_, b_ = conv_tori[-2], conv_tori[-1]
            shear_phi = (b_.iota - a_.iota) / (b_.flux - a_.flux)
        else:
            shear_phi = (best.iota - ax.iota) / best.flux
        g = resonance_guard(rc_.engine, best, shear=shear_phi * 2 * np.pi)
        out["guard"] = dict(ok=g["ok"], h=g["h"], x_over_w=g["x_over_w"], worst=g["worst"], n_checked=len(g["checked"]))
        out["surface"] = torus_record(best, full=True)
        out["surface"]["quad_err_half"] = q_
        out["refine_hist"] = best.info.get("refine_hist")
        out["refine_stop"] = best.info.get("refine_stop")
        quad_ok = out["surface"]["quad_err_half"] <= 1e-8
        topo = torus_topology(best, ax)
        out["surface"]["topology"] = topo
        out["status"] = ("surface" if (best.bn_off <= BN_SURFACE and g["ok"] and quad_ok and topo["ok"])
                         else "surface_unverified")
        if out["status"] == "surface_unverified" and not resonant_path:
            out["deferred"] = True
        if out["status"] == "surface_unverified" and resonant_path and (not g["ok"] or best.bn_off > BN_SURFACE) \
                and (deadline is None or time.time() < deadline - 60):
            # the torus at the request sits in or next to a resonant chain (guard) or would not refine:
            # look for the island band instead
            out["design_surface"] = dict(out["surface"], guard=out["guard"], refine_hist=out.get("refine_hist"),
                                         refine_stop=out.get("refine_stop"))
            out["design_surface"].pop("rc", None); out["design_surface"].pop("zs", None)
            r = resonant(best, float(best.bn_off),
                         "guard failed" if not g["ok"] else f"refinement stopped at bn {best.bn_off:.1e}")
            if r is not None and r.get("status") in ("surface", "no_surface"):
                return r
            if r is not None:
                out["resonant_result"] = {k: r.get(k) for k in ("status", "path", "certificate", "bracket_certificate",
                                                                 "family_stop", "chains", "jumps")}
        if range_budget_s > 0 and out["status"] == "surface":
            # the range of fluxes the source holds beyond its design surface: the family continued outward
            # from the verified tori (same axis), chains jumped, the outermost torus refined, guarded and
            # verified = a verified lower bound on the largest flux a surface encloses
            # with an upper bound: the edge barrier beyond the outermost verified torus (the range's other end)
            r = analyse(source, flux=None, dmin=dmin, tol=tol, verbose=verbose, budget_s=range_budget_s,
                        start_tori=[t for t in tori[:-1]] + [best], axis_guesses=[(ax.rc, ax.zs)], upper_bound=True)
            out["range"] = r.get("range")
            out["range_family"] = dict(profile=r.get("profile"), chains=r.get("chains"), jumps=r.get("jumps"),
                                       stop=r.get("family_stop"), status=r.get("status"), timings=r.get("timings"))
            if r.get("surface"):
                out["range_outer_surface"] = r["surface"]
        out["timings"]["total"] = time.time() - t0
        return out
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        out["status"] = "error"
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
        out["timings"]["total"] = time.time() - t0
        return out
