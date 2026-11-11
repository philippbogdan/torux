"""SIMSOPT 1.11.1 QfmSurface on one QUASR design: the quadratic-flux-minimising surface at the requested flux.

    python run_qfm.py 0003051                 # a sample design -> compare300-out/simsopt-qfm/
    python run_qfm.py 0224040 --pilot         # a tuning design (inputs/pilot/)
    python run_qfm.py 0003051 --set 'levels=[4, 6]' --set polish_rels=[] --no-warmup     # a quick check

QfmSurface minimises f = int (B.n)^2 dA / int B^2 dA over a SurfaceRZFourier (phi the cylindrical angle) with the
toroidal-flux label held by a quadratic penalty (minimize_qfm_penalty_constraints_LBFGS, restarted until it reports
convergence).  f is zero on a flux surface.  (SIMSOPT's QFM example follows L-BFGS with SLSQP on the exact label;
SIMSOPT writes that constraint as half the squared error, whose gradient vanishes at the solution, and SLSQP stops
with a singular or rank-deficient subproblem on every pilot.)  The harness climbs a resolution ladder (each level
starts from the previous answer padded exactly with zeros) and keeps the converged level with the smallest f on a
finer grid offset from the quadrature grid (SIMSOPT's own QfmResidual).

The flow: the coil quadrature as run_boozer_exact.py (QFM evaluates B and grad B at every iteration, so the rule's
size is also its speed); the start; levels 4, 6, 8, ..., 24, 28, 32, then three polish passes at the finest level;
the answer is the converged level or pass with the smallest off-grid f, none if none converged, converted exactly
to the evaluator's rz_fourier form (rc and zs copied, n times nfp) with 32 check points from SIMSOPT's own gamma().

Choices that are not settings below, and why:

- Method: QfmSurface(bs, s, ToroidalFlux, target).minimize_qfm_penalty_constraints_LBFGS(tol, maxiter=15000,
  constraint_weight=w), called again from where it stopped (up to 20 calls) until SIMSOPT reports success.  The QFM
  example uses tol 1e-12 and 1,000 iterations, then SLSQP on the exact label.  SIMSOPT passes `tol` to scipy as both
  ftol and gtol; ftol is a one-iteration test, absolute when f < 1 (f is the square of |B.n|/|B|), so tol 1e-12
  reports success at f ~ 1e-11 (|B.n|/|B| ~ 1e-5), 3 to 4 orders worse than QFM can do.  One call stops at scipy's
  15,000-evaluation cap whatever maxiter says, so the call is repeated.
- Surface: SurfaceRZFourier, stellarator symmetric, mpol = ntor = M, phi the cylindrical angle: QfmSurface's
  surface in SIMSOPT's QFM example.
- Quadrature (grid() below): theta 2 (2M+1) points; phi (2M+1) points on a half period, shifted half a spacing
  (SIMSOPT's "half period" range).  f's integrand ((B.n)^2 |N|) has twice the surface's bandwidth; the shifted
  half-period rule integrates a stellarator-symmetric integrand like a full uniform rule (SIMSOPT's docstring: "to
  provide spectral convergence of integrals"); the example uses 25 x 25 points per field period for M = 5 (2.3x).
- Start: the smaller QfmResidual f of two, SurfaceXYZTensorFourier.to_RZFourier() of QUASR's stored surface
  (cross-sections at fixed cylindrical angle, least-squares fit) at mpol = ntor = 10, and QUASR's own VMEC boundary
  (the input's namelist, RBC/ZBS copied); then truncated to M = 4 for the first level; the label's sign from the
  start used.  to_RZFourier gave the smaller f on all six pilots (by factors of 1.06 to 107); on 1981226 (iota 2.9,
  nfp 7) both are poor (f ~ 3e-3, 2% too much flux).
- Label: ToroidalFlux on the phi = 0 plane with max(256, 8M) points sharing the surface's degrees of freedom, as for
  the Boozer solvers; QfmSurface uses only its first derivative.
- Field objects: a fresh BiotSavart for each QfmResidual, its points set before the first evaluation: QfmResidual
  objects sharing one BiotSavart read each other's points (shape errors, or silently stale fields when the sizes
  match).

How the settings were chosen: on six designs outside the sample (inputs/pilot/), each answer measured by the
comparison's evaluator (max |B.n|/|B| at 4,000 points), on the comparison's processor.  f falls fast and then slowly
(0925611 at M = 10: 1.4e-10, then 1.5e-14 after 290 evaluations, 7e-16 after 15,000): QFM is limited by its
optimiser, not by resolution.  A fixed absolute tol (1e-20) is never met where a level creeps at its own floor
(f falling by about 1e-6 of itself per iteration): on four of six pilots M = 10 never converged in the hour,
while still improving.  A tolerance relative to f at each level's start
tightens level by level as f falls:

    design   abs 1e-20      rel 1e-9       rel 1e-7       rel 1e-7 M=4   rel 1e-5 M=4   + weight 1/target^2
    0925611  1.1e-8 365s    1.2e-8 3600s   5.6e-9 3600s   4.8e-8 3600s   2.0e-10 1746s  1.2e-9 454s
    1101401  3.9e-8 16s     4.3e-8 3601s   3.8e-8 56s     3.8e-8 24s     3.8e-8 10s     3.8e-8 8s
    2126458  none           5.8e-6 3600s   2.7e-6 3600s   2.0e-6 2832s   2.1e-6 131s    2.2e-6 73s
    2559727  none           6.1e-5 3600s   1.3e-7 3600s   1.4e-7 3600s   1.0e-9 948s    2.2e-7 3600s (*)
    0224040  none           1.0e-3 3600s   4.1e-4 3600s   4.8e-4 3601s   4.1e-4 358s    4.2e-4 570s
    1981226  none           none           none           none           none           4.7e-2 1361s
    (*) at 0.197 s per iteration (the rel 1e-5 run of the same design: 0.069 s); it still returned the converged
        M = 22 level when the clock ended

The tolerance relative to f at each level's start, from M = 4, matches or beats every other variant on every design
and leaves most of the hour unused on good ones; the scaled label weight is what gives 1981226 an answer at all
(with 100 its flux slipped 50%).  The polish passes then spend the unused time at the finest level.

The comparison's answers: these settings, `--mem-gb 12`, the default budget, SIMSOPT 1.11.1, numpy 2.5.4,
scipy 1.18.1, jax 0.12.0.1, Python 3.12.14, 4 compute threads in a process with 8 cores of an AMD EPYC 7742 to
itself (requirements/simsopt.txt).  Timing, the timing rule's meter and the budget are as run_boozer_exact.py says;
the warm-up is pilot 0925611 at M = 4, 6, 8 (300 iterations, one call, no polish).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simsopt_common as C  # noqa: E402  (sets the thread counts before numpy loads)

np = C.np

KEY, LABEL = "simsopt-qfm", "SIMSOPT QFM"

DEFAULTS = dict(
    levels=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32],
    oversample=2.0,              # quadrature: ntheta = oversample (2M + 1), nphi = half that on a half period
    constraint_weight=1.0,       # flux-label penalty weight = this / target^2 (100 at 0.1 Wb): a fixed 100 let pilot
                                 # 1981226's 0.002 Wb flux slip 50% at M = 4 and 7% at M = 6, so no level kept the
                                 # label (the example's 1 left even 0925611's flux 5e-9 off); as BoozerLS
    lbfgs_tol=1e-30,             # SIMSOPT's tol (scipy's ftol = gtol: success when one iteration lowers f by less than
                                 # tol, an absolute test since f < 1): the floor of the level tolerance ...
    lbfgs_tol_rel=1e-5,          # ... which is max(lbfgs_tol, this x f at the level's start), so it tightens level by
                                 # level as f falls (1e-9 and 1e-7 climbed too slowly: table above)
    start_M=4,                   # first level: the start truncated to M = 4, so a converged surface comes within
                                 # minutes, and each level after starts from a converged one
    polish_rels=[1e-6, 1e-7, 1e-8],  # once the finest level has converged, further passes at it with these relative
                                 # tolerances use the rest of the hour; each converged pass is a candidate, and the
                                 # clock ending a pass loses nothing already converged (0925611 stopped at 1.2e-9
                                 # after 454 s without them)
    lbfgs_maxiter=15000,         # per call (scipy's own cap is 15,000 evaluations per call) ...
    lbfgs_calls=20,              # ... restarted from where it stopped until it reports success (or the budget ends)
    label_accept_rel=1e-4,       # a level converges only if the penalty also kept the flux within this of its target
    start="best",                # "best": QUASR's VMEC boundary or to_RZFourier of the stored surface, the smaller f
    label_ntheta_min=256,        # flux label: trapezoid rule round the phi = 0 loop, max(this, 8 M) points
    quad_accept=1e-12,           # coil quadrature: first rule agreeing with its double to this ...
    quad_factor=2,               # ... then this many times that rule
    stop_after_no_gain=99,       # no stop for lack of gain: once M = 10 has converged, levels take seconds, and on
                                 # pilot 0925611 the off-grid f still fell 20-40% a level when a halving rule stopped
    stop_after_failures=3,       # three unconverged levels in a row end the ladder (the low levels are cheap)
    diag_min=64,                 # off-grid diagnostic grid: max(diag_min, 4 M + 2) squared points per field period
    mem_cap_gb=7.0,              # skip levels predicted past this; the levels fit (QFM peaked at 9.3 GiB, at M = 32
                                 # after its polish passes), and the harness stops it at 12 GiB
    margin_s=60.0,
    time_exponent=2.0,           # level time grows about as unknowns^this (first-order method, not dense solves)
    trace=False,                 # pilot studies: log f along the iterations
    omp_wait="passive",          # how idle threads wait: passive, because it measured better under fair_cores (pilot
                                 # 0925611, M = 10, 1,500 L-BFGS iterations, on the comparison's processor: runs that
                                 # met the timing rule took 16.4 s passive against 24.9 s with the spinning defaults,
                                 # 0.033 against 0.114 s per iteration over all runs; QFM's many small OpenMP regions
                                 # are what spinning pools slow down on shared cores)
    save_levels=False,
)


# the untimed warm-up solve before the timed design (imports and first-call set-up stay out of wall_s), metered
# like the timed run
WARMUP = dict(design="0925611", overrides=dict(levels=[4, 6, 8], lbfgs_maxiter=300, lbfgs_calls=1, polish_rels=[]))


def vmec_boundary(rec, nfp):
    """QUASR's VMEC namelist boundary (RBC(n, m), ZBS(n, m): R = sum rbc cos(m theta - n nfp phi)) as a
    SurfaceRZFourier: SIMSOPT's own convention, coefficients copied."""
    import re
    txt = rec["start"].get("vmec_namelist") or ""
    rb, zb = {}, {}
    for key, n, m, v in re.findall(r"(RBC|ZBS)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*([-+0-9.eEdD]+)", txt):
        (rb if key == "RBC" else zb)[(int(n), int(m))] = float(v.replace("D", "e").replace("d", "e"))
    if not rb:
        return None
    mpol = max(m for _, m in rb)
    ntor = max(abs(n) for n, _ in rb)
    M = max(mpol, ntor)
    s = rz_surface(nfp, True, M, [0.0], [0.0])
    s.rc[:, :] = 0.0
    s.zs[:, :] = 0.0
    for (n, m), v in rb.items():
        s.rc[m, n + M] = v
    for (n, m), v in zb.items():
        if not (m == 0 and n <= 0):
            s.zs[m, n + M] = v
    s.local_full_x = s.get_dofs()
    return s


def ndofs(M):
    """Unknowns of a stellarator-symmetric SurfaceRZFourier with mpol = ntor = M (rc and zs, less zs[0, n <= 0])."""
    return (M + 1) * (2 * M + 1) * 2 - M - 1


def rz_surface(nfp, ss, M, qp, qt, dofs=None):
    from simsopt.geo import SurfaceRZFourier
    kw = dict(nfp=nfp, stellsym=ss, mpol=M, ntor=M, quadpoints_phi=np.asarray(qp, float),
              quadpoints_theta=np.asarray(qt, float))
    if dofs is not None:
        kw["dofs"] = dofs
    return SurfaceRZFourier(**kw)


def grid(M, nfp, q):
    from simsopt.geo import Surface
    nt = int(np.ceil(q * (2 * M + 1)))
    return Surface.get_phi_quadpoints(nphi=int(np.ceil(nt / 2)), range="half period", nfp=nfp), \
        np.linspace(0, 1, nt, endpoint=False)


def lift_rz(s, M, qp, qt):
    """The same surface at mpol = ntor = M >= s's (rc, zs zero-padded), exactly."""
    new = rz_surface(s.nfp, s.stellsym, M, qp, qt)
    new.rc[:, :] = 0.0
    new.zs[:, :] = 0.0
    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            new.rc[m, n + M] = s.rc[m, n + s.ntor]
            new.zs[m, n + M] = s.zs[m, n + s.ntor]
    new.local_full_x = new.get_dofs()
    return new


def truncate_rz(s, M):
    """s's coefficients with m, |n| <= M (a start for a coarser first level; not the same surface)."""
    new = rz_surface(s.nfp, s.stellsym, M, [0.0], [0.0])
    new.rc[:, :] = 0.0
    new.zs[:, :] = 0.0
    for m in range(min(M, s.mpol) + 1):
        for n in range(-min(M, s.ntor), min(M, s.ntor) + 1):
            new.rc[m, n + M] = s.rc[m, n + s.ntor]
            new.zs[m, n + M] = s.zs[m, n + s.ntor]
    new.local_full_x = new.get_dofs()
    return new


def rz_err(a, b, n=17, seed=3):
    rng = np.random.default_rng(seed)
    qp, qt = np.sort(rng.random(n)), np.sort(rng.random(n))
    ea, eb = rz_surface(a.nfp, a.stellsym, a.mpol, qp, qt), rz_surface(b.nfp, b.stellsym, b.mpol, qp, qt)
    ea.x, eb.x = a.x, b.x
    return float(np.max(np.abs(ea.gamma() - eb.gamma())))


def qfm_f(s, coils):
    """SIMSOPT's QfmResidual of s with its own BiotSavart, its points set explicitly (QfmResidual objects sharing
    one BiotSavart can read each other's points)."""
    from simsopt.field import BiotSavart
    from simsopt.geo import QfmResidual
    b = BiotSavart(coils)
    b.set_points(np.ascontiguousarray(s.gamma().reshape(-1, 3)))
    return float(QfmResidual(s, b).J())


def qfm_offgrid(s, coils, nd):
    """SIMSOPT's QfmResidual f and |B.n|/|B| on a grid offset from the quadrature grid (one field period)."""
    from simsopt.field import BiotSavart
    sd = rz_surface(s.nfp, s.stellsym, s.mpol, (np.arange(nd) + 0.318309886) / nd / s.nfp,
                    (np.arange(nd) + 0.577215665) / nd, dofs=s.dofs)
    f = qfm_f(sd, coils)
    bs_d = BiotSavart(coils)
    pts = sd.gamma().reshape(-1, 3)
    bs_d.set_points(np.ascontiguousarray(pts))
    B = bs_d.B()
    nrm = sd.normal().reshape(-1, 3)
    bn = np.abs(np.sum(B * nrm, 1)) / (np.linalg.norm(B, axis=1) * np.linalg.norm(nrm, axis=1))
    return dict(diag_qfm_f=f, diag_qfm_sqrt_f=float(np.sqrt(max(f, 0.0))), diag_bn_max=float(bn.max()),
                diag_bn_rms=float(np.sqrt(np.mean(bn ** 2))), diag_points=nd * nd)


def solve(rec, S, ckpt, t0, budget):
    from simsopt.field import BiotSavart
    from simsopt.geo import QfmSurface, ToroidalFlux

    nfp, ss = int(rec["nfp"]), bool(rec["stellarator_symmetric"])
    req = float(rec["requested_toroidal_flux_Wb"])
    st = rec["start"]["surface"]
    state = dict(best=None, history=[], versions=C.versions(), host=C.host(), reason=None, solver_status=None)

    def save():
        C.atomic_write(ckpt, state)

    C.log("versions", state["versions"])
    C.log("host", state["host"])
    save()

    probe = C.start_xyz(rec, np.linspace(0, 1 / nfp, 24, endpoint=False), np.linspace(0, 1, 24, endpoint=False))
    nq, qtab = C.coil_quadrature(rec, probe.gamma().reshape(-1, 3), accept=S["quad_accept"], factor=S["quad_factor"])
    C.log(f"coil quadrature: {nq} points per coil; |B(N) - B(2N)|/|B| ladder {qtab}")
    coils = C.simsopt_coils(rec, nq)
    bs_tf = BiotSavart(coils)

    # the start: SIMSOPT's to_RZFourier of the stored surface (cross-sections at fixed cylindrical angle, fitted),
    # or QUASR's own VMEC boundary of it; the one with the smaller QFM objective (SIMSOPT's QfmResidual)
    M0 = max(int(st["mpol"]), int(st["ntor"]), min(S["levels"]))
    qp0, qt0 = grid(M0, nfp, S["oversample"])
    sx = C.lift_xyz(C.start_xyz(rec, [0.0], [0.0]), M0, M0, np.linspace(0, 1 / nfp, 2 * (2 * M0 + 1), endpoint=False),
                    np.linspace(0, 1, 2 * (2 * M0 + 1), endpoint=False))
    cands = {"to_RZFourier": sx.to_RZFourier()}
    vb = vmec_boundary(rec, nfp)
    if vb is not None and S["start"] in ("best", "vmec"):
        cands["QUASR VMEC boundary"] = vb
    if S["start"] == "vmec" and vb is not None:
        cands.pop("to_RZFourier")
    fstart = {}
    for k, c in list(cands.items()):
        cands[k] = lift_rz(c, max(M0, c.mpol, c.ntor), qp0, qt0) if max(c.mpol, c.ntor) <= M0 else c
        try:
            fstart[k] = qfm_f(rz_surface(nfp, ss, cands[k].mpol, qp0, qt0, dofs=cands[k].dofs), coils)
        except Exception as e:  # noqa: BLE001
            fstart[k] = float("inf")
            C.log(f"start {k}: {e!r}")
    pick = min(fstart, key=fstart.get)
    cur = cands[pick]
    M0 = max(M0, cur.mpol)
    if S["start_M"] is not None and S["start_M"] < M0:          # a cheaper first level: the start truncated
        M0 = int(S["start_M"])
        cur = truncate_rz(cur, M0)
    # the label's sign from the start actually used (its theta may run either way)
    loop0 = rz_surface(nfp, ss, cur.mpol, [0.0], np.linspace(0, 1, 512, endpoint=False), dofs=cur.dofs)
    f0 = float(ToroidalFlux(loop0, bs_tf).J())
    target = float(np.sign(f0)) * req
    C.log(f"start: {pick} (QFM f of the candidates {fstart}); its flux {f0:.12g}, target label {target:.12g}")
    state["start"] = dict(start=pick, start_f=fstart, start_flux=f0, target_label=target, coil_quadrature=nq,
                          coil_quadrature_ladder=qtab, M0=M0)

    levels = [M0] + [m for m in S["levels"] if m > M0]
    passes = [(m, S["lbfgs_tol_rel"], None) for m in levels] + [(levels[-1], r, r) for r in S["polish_rels"]]
    state["calls"] = []

    def speed():
        """seconds per L-BFGS iteration of this design, overall and per level (machine load shows here)."""
        calls = [c for c in state["calls"] if c["iterations"] > 0]
        if not calls:
            return None
        it = sum(c["iterations"] for c in calls)
        sec = sum(c["seconds"] for c in calls)
        per = {}
        for c in calls:
            e = per.setdefault(str(c["M"]), [0, 0.0])
            e[0] += c["iterations"]
            e[1] += c["seconds"]
        return dict(s_per_iteration=sec / it, iterations=it, seconds_in_lbfgs=sec,
                    per_level={k: dict(iterations=v[0], s_per_iteration=v[1] / v[0]) for k, v in per.items()},
                    calls=len(calls))
    lad = C.Ladder(S, t0, budget, state, ckpt, save)
    lad_exp = S["time_exponent"]
    w = S["constraint_weight"] / target ** 2

    for M, rel, polish in passes:
        D = ndofs(M)
        nt = int(np.ceil(S["oversample"] * (2 * M + 1)))
        mem = (3 * nt * nt / 2 * 3 * D) * 8 / 1e9 + 0.6
        # the ladder's time model is the cube of the unknowns; QFM's first-order cost grows more slowly
        if not lad.fits(M, mem, D ** (lad_exp / 3.0)):
            break
        qp, qt = grid(M, nfp, S["oversample"])
        s = lift_rz(cur, M, qp, qt)
        nth = max(S["label_ntheta_min"], 8 * M)
        lab_s = rz_surface(nfp, ss, M, [0.0], np.linspace(0, 1, nth, endpoint=False), dofs=s.dofs)
        label = ToroidalFlux(lab_s, bs_tf)
        h = dict(M=M, polish=polish, unknowns=D, quadpoints=len(qp) * len(qt), lift_err_m=rz_err(cur, s))
        tl = time.time()
        try:
            bs = BiotSavart(coils)                    # this level's own field object (see qfm_f)
            bs.set_points(np.ascontiguousarray(s.gamma().reshape(-1, 3)))
            q = QfmSurface(bs, s, label, target)
            trace = []
            if S["trace"]:
                orig = q.qfm_penalty_constraints

                def traced(x, derivatives=0, constraint_weight=1, _o=orig):
                    out = _o(x, derivatives=derivatives, constraint_weight=constraint_weight)
                    trace.append(float(out[0] if derivatives else out))
                    return out
                q.qfm_penalty_constraints = traced
            f_in = qfm_f(s, coils)
            stages = []
            ok = False
            f_level = float(q.qfm_penalty_constraints(s.x, derivatives=0, constraint_weight=w))
            tol_level = max(S["lbfgs_tol"], rel * f_level)
            for call in range(S["lbfgs_calls"]):
                tc = time.time()
                f_call = float(q.qfm_penalty_constraints(s.x, derivatives=0, constraint_weight=w))
                tol_call = tol_level
                r1 = q.minimize_qfm_penalty_constraints_LBFGS(tol=tol_call, maxiter=S["lbfgs_maxiter"],
                                                              constraint_weight=w)
                stages.append(dict(stage="L-BFGS-B penalty", tol=tol_call, success=bool(r1["success"]),
                                   iterations=int(r1["iter"]),
                                   message=str(r1["info"].message), fun=float(r1["fun"]),
                                   grad_max=float(np.max(np.abs(r1["gradient"]))), seconds=time.time() - tc))
                stages[-1].update(f_start=f_call)
                state["calls"].append(dict(M=M, iterations=int(r1["iter"]), seconds=stages[-1]["seconds"]))
                state["speed"] = speed()
                save()
                C.log("  call", stages[-1])
                ok = bool(r1["success"])
                if ok or "ABNORMAL" in stages[-1]["message"] and len(stages) > 1 and \
                        stages[-2]["message"] == stages[-1]["message"] and stages[-1]["iterations"] < 10:
                    break
            f_out = qfm_f(s, coils)
            lab_err = float(label.J() - target)
            kept = abs(lab_err) <= S["label_accept_rel"] * abs(target)
            ok = ok and kept
            h.update(success=ok, stages=stages, f_start=f_in, f=f_out, label_err_Wb=lab_err, label_kept=kept,
                     iterations=sum(x["iterations"] for x in stages))
            if trace:
                k = np.unique(np.geomspace(1, len(trace), 40).astype(int)) - 1
                h["trace"] = [[int(i), trace[i]] for i in k]
            del q
        except Exception as e:  # noqa: BLE001
            ok = False
            h.update(success=False, error=repr(e)[:300])
        h["seconds"] = time.time() - tl
        h["maxrss_GB"] = C.maxrss_gb()
        lad.ran(D ** (lad_exp / 3.0), h["seconds"], 3)
        if True:   # every level's surface is diagnosed (the pilot study compares them); only converged ones count
            h.update(qfm_offgrid(s, coils, max(S["diag_min"], 4 * M + 2)))
        if ok:
            def make_best():
                surf, cp, native = C.rz_answer_parts(s)
                return dict(
                    solver_status=(f"L-BFGS-B converged (success=True: {h['stages'][-1]['message']}, tol "
                                   f"{h['stages'][-1]['tol']:.2e} = max({S['lbfgs_tol']:g}, {rel:g} f at "
                                   f"the level's start), after {len(h['stages'])} call(s); flux label within "
                                   f"{abs(h['label_err_Wb']) / abs(target):.1e} of its target) at mpol=ntor={M}"
                                   + (f" (polish pass, tolerance {polish:g} f)" if polish else "") +
                                   ", the converged pass with the smallest QFM residual on the off-grid check"),
                    surface=surf, check_points=cp, native=native,
                    solver_reported=dict(level_M=M, polish=polish, qfm_f=h["f"], flux_label_Wb=lab_err + target,
                                         flux_label_err_Wb=lab_err, offgrid_qfm_f=h["diag_qfm_f"],
                                         offgrid_bn_max=h["diag_bn_max"], offgrid_bn_rms=h["diag_bn_rms"],
                                         stages=h["stages"], coil_quadrature=nq))

            h["chosen"] = lad.converged(h["diag_qfm_f"], make_best)
        else:
            lad.failed()
        cur = s                    # QFM always continues from its latest iterate
        if S["save_levels"]:
            surf, cp, _ = C.rz_answer_parts(s)
            tag = f"M{M:02d}" + (f"p{-int(round(np.log10(polish)))}" if polish else "")
            C.atomic_write(Path(ckpt).parent.parent / "levels" / f"quasr_{rec['id']}_{tag}.json",
                           dict(design=rec["id"], solver=f"{KEY}@{tag}", status="surface", surface=surf,
                                check_points=cp, level={k: v for k, v in h.items() if k != "trace"}))
        state["history"].append(h)
        C.log("level", C.short({k: v for k, v in h.items() if k not in ("trace", "stages")}))
        C.log("  stages", h.get("stages"))
        save()
        if not lad.more():
            break
    lad.finish("the QFM minimisation did not report convergence at any level")


if __name__ == "__main__":
    C.main(KEY, LABEL, solve, DEFAULTS, __doc__, warmup=WARMUP)
