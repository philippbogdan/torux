"""The island chain at a request found from the field itself, not from a torus (torux.analysis POLICY
'traced_lines').

By default the chain the request sits in is sought from the failed design torus: the rationals nearest the
torus's transform (weighted by q^2), their closed lines by Newton from curves on that torus (torux.chains.chain_lines).
Where the design Newton failed, the torus is the design surface itself, whose transform is the design's, not the
field's, and whose curves can be far from the chain: 'no closed lines of the chain', or the wrong chain first.

Here, without any torus:
* rotation_scan: lines from the outboard half of the symmetry line Z = 0 in the plane phi = 0, around the request's
  predicted radius (the design surface's outboard point: the design surface carries the requested flux), traced
  with the compiled tracer (torux.seeds.SectionTracer) and classified by torux.seeds._orbit_record (regular: a smooth
  invariant circle, its flux by the loop integral of A; island; chaotic; lost).  The regular orbits' fluxes bracket
  the request; the rotation numbers between them name the chains that can hold it (rationals between the bracket's
  rotations, the locked rotation of island orbits first).
* symmetric_chain_lines: the stellarator-symmetric closed lines of period q of the chain p/q, from sign changes of Z
  after q half periods along the symmetric half-lines (both symmetry planes, both sides of the axis) in the
  request's annulus (torux.barrier.symmetric_roots), the winding about the axis checked to be p, each line Newton-solved
  as a Fourier curve (torux.orbits.solve_closed_line) from its own traced trajectory, with its spectral monodromy
  (torux.orbits.line_monodromy).  These are the lines torux.islandband's certificates start from.

Nothing here is a claim: the lines and the bracket only choose what the certificates (torux.islandband) and the
acceptance (torux.seeds.accept) then prove or refuse.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

import numpy as np

from . import barrier as BR
from . import orbits as OR
from . import seeds as SD


def rationals_between(lo, hi, qmax):
    """Reduced p/q with lo <= p/q <= hi, q <= qmax, ordered by q."""
    lo, hi = min(lo, hi), max(lo, hi)
    out = set()
    for q in range(1, qmax + 1):
        for p in range(int(math.floor(lo * q)), int(math.ceil(hi * q)) + 1):
            if lo <= p / q <= hi:
                f = Fraction(p, q)
                out.add((f.numerator, f.denominator))
    return sorted(out, key=lambda pq: (pq[1], abs(pq[0])))


def rotation_scan(tracer, engine, axis, R_lo, R_hi, n, nper, deadline=None):
    """Lines from (R, 0) in phi = 0, R in [R_lo, R_hi] (n of them), traced nper periods; their records
    (torux.seeds._orbit_record: R0, kind, rot, digits, flux), sorted by R0, and the traces {R0: (sec, dang)}."""
    R0s = np.linspace(R_lo, R_hi, n)
    o = tracer.orbits(np.stack([R0s, 0 * R0s], -1), nper, deadline=deadline)
    recs, traces = [], {}
    for i, R0 in enumerate(R0s):
        r = SD._orbit_record(R0, o["sec"][i], o["dang"][i], o["lost"][i], engine)
        r["truncated"] = bool(o["n"] < nper)        # cut by the deadline: its kind is not decided by nper periods
        recs.append(r)
        traces[float(R0)] = (o["sec"][i], o["dang"][i])
    return sorted(recs, key=lambda r: r["R0"]), traces


def bracket(probes, request):
    """The regular orbits nearest the request below and above in flux (by R0 along the half-line)."""
    reg = [r for r in probes if r["kind"] == "regular" and np.isfinite(r.get("flux", np.nan))]
    below = [r for r in reg if r["flux"] < request]
    above = [r for r in reg if r["flux"] >= request]
    a = max(below, key=lambda r: r["R0"]) if below else None
    b = min(above, key=lambda r: r["R0"]) if above else None
    return a, b


def locate_request(engine, source, axis, R_design, request, deadline=None, n=32, nper=None, rounds=3,
                   tracer=None, verbose=False, qmax=40, rot_design=None, nper_cap=6000):
    """Where the request lies on the outboard symmetry half-line of phi = 0: a rotation scan in a window around
    R_design (the design surface's outboard point), narrowed to the bracket of regular orbits round by round.
    Returns dict(probes, below, above, kinds_between, rot_window, candidates [(p, q)], tracer, ...).  The
    candidate chains: the rotations of island / locked orbits between the bracket first (their nearest p/q,
    q <= 64), then the rationals between the bracket's rotations (lowest q first, q <= qmax)."""
    t0 = time.time()
    nfp = int(source.nfp)
    Rax = float(axis.at(np.array([0.0]))[0][0])
    if tracer is None:
        teng = SD.trace_engine(engine, source)
        tracer = SD.SectionTracer(teng, nfp, axis, sign=SD.bphi_sign(engine, axis))
        tracer.calibrate(np.stack([Rax + (R_design - Rax) * np.array([0.5, 0.8, 1.0, 1.1]), np.zeros(4)], -1))
    rot_g = abs(axis.iota / nfp)
    if nper is None:
        # ~300 poloidal turns, and 3 sweeps of the circle next to the design's nearest rational (q <= 20: the phases
        # advance q |rot - p/q| per period there; a near-resonant regular orbit traced for less reads 'chaotic')
        rot_d = abs(rot_design) if rot_design is not None else rot_g
        sweep = max([1.0 / (q * abs(rot_d - round(rot_d * q) / q)) for q in range(1, 21)
                     if abs(rot_d - round(rot_d * q) / q) > 1e-12] + [0.0])
        nper = int(np.clip(max(300.0 / max(rot_d, 1e-3), 3.0 * sweep), 400, nper_cap))
    out = dict(R_axis=Rax, R_design=float(R_design), rounds=[], nper=nper)
    lo, hi = Rax + 0.9 * (R_design - Rax), Rax + 1.1 * (R_design - Rax)
    probes, traces = [], {}
    a = b = None
    for rnd in range(rounds):
        if deadline is not None and time.time() > deadline:
            break
        t1 = time.time()
        new, tr_ = rotation_scan(tracer, engine, axis, lo, hi, n, nper if rnd == 0 else min(2 * nper, nper_cap), deadline=deadline)
        probes = sorted(probes + new, key=lambda r: r["R0"])
        traces.update(tr_)
        a, b = bracket(probes, request)
        kinds = {}
        for r in new:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        out["rounds"].append(dict(lo=lo, hi=hi, kinds=kinds, wall=time.time() - t1,
                                  below=None if a is None else (a["R0"], a["flux"]),
                                  above=None if b is None else (b["R0"], b["flux"])))
        if verbose:
            print(f"   scan {rnd}: [{lo:.6f}, {hi:.6f}] {kinds} below {None if a is None else (round(a['R0'], 7), a['flux'])} "
                  f"above {None if b is None else (round(b['R0'], 7), b['flux'])} ({time.time() - t1:.0f}s)", flush=True)
        if a is not None and b is not None:
            if (b["flux"] - a["flux"]) <= 1e-4 * request or (lo, hi) == (a["R0"], b["R0"]):
                break
            lo, hi = a["R0"], b["R0"]
        elif a is not None:
            lo, hi = a["R0"], a["R0"] + 1.5 * (hi - lo)
        elif b is not None:
            lo, hi = max(Rax + 0.05 * (R_design - Rax), b["R0"] - 1.5 * (hi - lo)), b["R0"]
        else:
            w = hi - lo
            lo, hi = max(Rax + 0.05 * (R_design - Rax), lo - w), hi + w
    between = [r for r in probes if (a is None or r["R0"] > a["R0"]) and (b is None or r["R0"] < b["R0"])]
    kinds = {}
    for r in between:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    out.update(probes=probes, below=a, above=b, kinds_between=kinds, traces=traces, tracer=tracer)
    # candidate chains
    cands = []
    isl = [r for r in between if r["kind"] in ("island", "undecided") and np.isfinite(r.get("rot", np.nan))]
    for r in isl:
        p, q, d = SD.near_rational(r["rot"], 64)
        if q >= 2 and d < 1e-6 and (p, q) not in cands:
            cands.append((p, q))
    ra = a["rot"] if a is not None else axis.iota / nfp
    if b is not None:
        rb = b["rot"]
    else:
        rots = [r["rot"] for r in between if np.isfinite(r.get("rot", np.nan))]
        rb = rots[-1] if rots else ra
    out["rot_window"] = (float(ra), float(rb))
    for pq in rationals_between(ra, rb, qmax):
        if pq[1] >= 2 and pq not in cands:
            cands.append(pq)
    out["candidates"] = cands
    if a is not None and b is not None:
        out["gap_rel"] = float((b["flux"] - a["flux"]) / request)
    out["wall"] = time.time() - t0
    return out


def _fourier_guess(rec_R, rec_Z, L, phi0):
    """Callables phi -> R, Z: the trigonometric interpolant of samples at phi0 + L j / n (j < n) over the
    period L (a closed line's trajectory is L-periodic)."""
    n = len(rec_R)
    cR, cZ = np.fft.rfft(rec_R) / n, np.fft.rfft(rec_Z) / n
    k = np.arange(len(cR))
    w = np.where((k == 0) | ((n % 2 == 0) & (k == n // 2)), 1.0, 2.0)

    def ev(c, phi):
        t = (np.asarray(phi, float) - phi0) / L
        return np.real(np.exp(2j * np.pi * np.multiply.outer(t, k)) @ (w * c))
    return (lambda ph: ev(cR, ph)), (lambda ph: ev(cZ, ph))


def symmetric_chain_lines(engine, source, axis, p, q, windows, deadline=None, nscan=48, sign=None,
                          steps_per_period=24, verbose=False, max_lines=4):
    """The stellarator-symmetric closed lines of period q (field periods) and winding p of the chain p/q crossing
    the symmetric half-lines inside `windows`: {(plane, side): (r_lo, r_hi)} with plane 0 / 1 (phi = 0 / pi/nfp),
    side +1 outboard / -1 inboard of the axis, r the distance from the axis point along Z = 0.  Returns a list of
    converged torux.orbits.ClosedLine (with residue and monodromy), distinct."""
    nfp = int(source.nfp)
    T = 2 * np.pi / nfp
    lines = []
    sgn = SD.bphi_sign(engine, axis) if sign is None else float(sign)
    info = []
    for (plane, side), (r_lo, r_hi) in windows.items():
        if deadline is not None and time.time() > deadline:
            break
        phi_s = plane * np.pi / nfp
        Ra = float(axis.at(np.array([phi_s]))[0][0])
        sec = BR.FieldSection(engine, source, sign=sgn, k=6, steps_per_period=steps_per_period, reference=False,
                              scale=abs(r_hi), phi0=phi_s)
        Rv = Ra + side * np.linspace(r_lo, r_hi, nscan)
        Rv = np.sort(Rv)
        # Z after q half periods (a symmetric line through Z = 0 in one symmetry plane crosses Z = 0 again in the
        # next one: half a period later, the planes alternating; after q half periods the line closes by symmetry
        # iff it is a symmetric period-q line)
        o = sec.tracer.trace(np.stack([Rv, 0 * Rv], -1), phi_s, q * np.pi / nfp)
        Z = np.where(o["lost"], np.nan, o["y"][:, 1])
        idx = np.flatnonzero(np.isfinite(Z[:-1]) & np.isfinite(Z[1:]) & (Z[:-1] * Z[1:] < 0))

        def fun(R):
            oo = sec.tracer.trace(np.stack([R, 0 * R], -1), phi_s, q * np.pi / nfp)
            return np.where(oo["lost"], np.nan, oo["y"][:, 1])
        if len(idx) == 0:
            info.append(dict(plane=plane, side=side, roots=0))
            continue
        roots, fr = BR._refine_roots(fun, Rv[idx], Rv[idx + 1], Z[idx], Z[idx + 1], 1e-12 * abs(Ra), maxit=60)
        roots = roots[np.isfinite(roots)]
        nr = 0
        for R0 in roots:
            if deadline is not None and time.time() > deadline:
                break
            # the line over q periods, sampled for its Fourier guess (winding about the axis checked: p turns)
            nout = max(32, 4 * steps_per_period)
            tr = sec.line_tracer(k=6, steps_per_period=nout)
            oo = tr.trace(np.array([[R0, 0.0]]), phi_s, q * T, nsteps=q * nout, record=True)
            if oo["lost"][0]:
                continue
            Y = np.concatenate([[[R0, 0.0]], np.array([r_[0] for r_ in oo["rec"]])])
            phis = phi_s + q * T * np.arange(len(Y)) / (len(Y) - 1)
            Rax_, Zax_ = axis.at(phis)
            ang = np.unwrap(np.arctan2(Y[:, 1] - Zax_, Y[:, 0] - Rax_))
            wind = (ang[-1] - ang[0]) / (2 * np.pi)
            if abs(abs(wind) - abs(p)) > 0.25:
                info.append(dict(plane=plane, side=side, R=float(R0), winding=float(wind), rejected="winding"))
                continue
            gR, gZ = _fourier_guess(Y[:-1, 0], Y[:-1, 1], q * T, phi_s)
            K = max(24, 10 * q + 16)
            ln = OR.solve_closed_line(engine, nfp, q, gR, gZ, phi0=phi_s, K=K, symmetric=True, deadline=deadline)
            if not ln.converged:
                info.append(dict(plane=plane, side=side, R=float(R0), winding=float(wind), rejected="newton",
                                 res=float(ln.res)))
                continue
            ln = OR.line_monodromy(engine, ln)
            pts = ln.section_points(0.0)
            dup = any(np.min(np.linalg.norm(pts[:, None, :] - o_.section_points(0.0)[None], axis=-1)) < 1e-8
                      for o_ in lines)
            info.append(dict(plane=plane, side=side, R=float(R0), winding=float(wind), residue=float(ln.residue),
                             dup=bool(dup)))
            if verbose:
                print(f"    chain {p}/{q} plane {plane} side {side:+d}: R {R0:.8f} winding {wind:.3f} residue "
                      f"{ln.residue:.4e} dup {dup}", flush=True)
            if not dup:
                lines.append(ln)
                nr += 1
            if len(lines) >= max_lines:
                break
        info.append(dict(plane=plane, side=side, roots=int(len(roots)), new_lines=nr))
    return lines, info
