"""SIMSOPT 1.11.1 BoozerLS on one QUASR design: the surface at the requested toroidal flux.

    python run_boozer_ls.py 0003051                 # a sample design -> compare300-out/simsopt-boozer-ls/
    python run_boozer_ls.py 0224040 --pilot         # a tuning design (inputs/pilot/)
    python run_boozer_ls.py 0003051 --set 'levels=[10, 12]' --no-warmup     # a quick check

BoozerLS minimises the mean square of the Boozer residual (G B - |B|^2 (x_phi + iota x_theta)) / |B| over a grid
of quadrature points plus a quadratic penalty on the toroidal-flux label.  As SIMSOPT's
examples/2_Intermediate/boozer.py does: dense BFGS (minimize_boozer_penalty_constraints_LBFGS), then the damped
Gauss-Newton / Levenberg-Marquardt of minimize_boozer_penalty_constraints_ls(method='manual') "to drive it to
machine zero".  Each level above the first starts from the previous answer padded exactly with zeros.  The harness
keeps the converged level whose Boozer residual at points off the quadrature grid (SIMSOPT's own residual) is
smallest.

The flow: coil quadrature, G and iota, the flux label's sign, the exact lift between levels and the choice among
levels are those of run_boozer_exact.py (its docstring gives their reasons).  At the first level the example's
path runs (BFGS, then LM), then LM straight from QUASR's surface, then LM with QUASR's target iota, until one
converges; at later levels LM from the padded previous level, then BFGS and LM.  The answer is the converged level
with the smallest off-grid residual, none if no level converged, converted exactly to the evaluator's xyz_fourier
form with 32 check points from SIMSOPT's own gamma().

Choices that are not settings below, and why:

- Why not run_code's Newton: it needs the flux label's Hessian, which SIMSOPT 1.11.1 builds as four
  (ntheta, 3, D, D) arrays (ToroidalFlux.d2J_by_dsurfacecoefficientsdsurfacecoefficients): 7.8 GiB at M = 10 with
  a fine label rule and 25 GiB at M = 20 even with the surface's own points, so Newton cannot reach the resolutions
  BoozerLS needs.  LM needs first derivatives only and converges to the same minimiser.
- Surface: SurfaceXYZTensorFourier, stellarator symmetric, mpol = ntor = M, as BoozerExact.
- Grid (grid() below): oversample x (2M+1) points per direction over one field period, from phi = theta = 0.  The
  penalty's z(0, 0) term reads the first point, so the grid must start at the origin.  (2M+1) per direction is
  QUASR's own grid, but over a full period half the points are stellarator-symmetric twins, so it barely
  over-determines the unknowns.
- Memory model of an LM level (predicted_mem_gb below): about ten arrays of npts x 3 x D plus J^T J, calibrated on
  the pilots (18.4 GiB measured at M = 40 on 81 x 81 points).  Under the 11.5 GB cap the ladder ends at M = 32
  (9.0 GB predicted, 11.6 GiB peak measured); M = 36 (14 GB) does not fit 12 GiB, and would need about 2,000 s more
  than the budget leaves after M = 32.

How the settings were chosen: on six designs outside the sample (inputs/pilot/), every level kept, each measured by
the comparison's evaluator.

    what was tried                 what it showed
    run_code (BFGS then Newton)    7.8 GiB at M = 10: the label's Hessian (above)  -> the LM path
    LM on a 1x grid (QUASR's)      behaves as collocation (1101401: 7e-5 at M = 10, as BoozerExact); LM tol 1e-12
                                   froze 0925611 at 1.7e-11 from M = 24; BFGS found a collapsed surface on 0224040
                                   (iota 6e9, flux off 100%)  -> 1.5x grid, label check, restarts
    LM tol 1e-14                   every level of four designs failed at |J^T r| ~ (1-5)e-15 G^2
                                   -> LM to its floor, convergence at 3e-14 G^2 (M/10)^2
    label weight 1/target^2        1981226 converges (3.8e-4 at M = 16); with a fixed 100 its flux slipped

With these settings to M = 28, SIMSOPT's own off-grid |B.n|/|B| on the pilots:

    design   BoozerLS (level)       BoozerExact (level)    note
    0925611  1.9e-14 (M = 28)       1.8e-14 (M = 36)       LS reaches the floor 8 levels earlier
    1101401  2.3e-8  (M = 28)       5.7e-5  (M = 10)       m = 8 near-resonant; QFM finds 3.8e-8 too
    2126458  3.3e-6  (M = 24)       3.5e-4  (M = 10)
    2559727  1.7e-9  (M = 28)       2.6e-11 (M = 40)       still falling at 28
    1981226  none (flux slipped)    2.4e-7  (M = 40)       with weight 1/target^2: converges, 3.8e-4 at M = 16
    0224040  (2e-3 at M = 10 -> 6e-4 at M = 18)            3/5 resonance

Level times grow steeply (0925611: 41 s at M = 22, 131 s at 24, 287 s at 26): a ladder to M = 28 took 10 to 18
minutes.

The comparison's answers: these settings, `--mem-gb 12`, the default budget, SIMSOPT 1.11.1, numpy 2.5.4,
scipy 1.18.1, jax 0.12.0.1, Python 3.12.14, 4 compute threads in a process with 8 cores of an AMD EPYC 7742 to
itself (requirements/simsopt.txt).  Timing, the timing rule's meter and the budget are as run_boozer_exact.py says;
the warm-up is pilot 0925611 at M = 4, 10, 12.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simsopt_common as C  # noqa: E402  (sets the thread counts before numpy loads)

np = C.np

KEY, LABEL = "simsopt-boozer-ls", "SIMSOPT BoozerLS"

DEFAULTS = dict(
    levels=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40, 44, 48],
    oversample=1.5,              # quadrature: oversample * (2 M + 1) points in theta and over a full period in phi
                                 # (stored as the stellarator-symmetric half period, both symmetry lines included)
    constraint_weight=1.0,       # flux-label penalty weight = this / target^2 (100 at 0.1 Wb, the weight of
                                 # examples/2_Intermediate/boozer.py on an O(0.1-1) label): the penalty is then on the
                                 # relative error; a fixed 100 let pilot 1981226's 0.002 Wb flux slip 18%, 1.2% and
                                 # 0.3% at M = 10, 12, 14 (the residual pulls the surface towards a better-closing
                                 # one), and weight 1 missed by 1.5e-5 even on 0925611 at M = 4
    ls_method="manual",          # minimize_boozer_penalty_constraints_ls: SIMSOPT's damped Gauss-Newton (LM)
    ls_maxiter=40,               # LM iterations per call, run to the rounding floor: SIMSOPT's 'manual' LM stops at
                                 # the first iterate under its tol, and a loose tol freezes a level at its padded
                                 # start (pilot: 0925611 stuck at 1.7e-11 from M = 24 with tol 1e-12, while
                                 # BoozerExact reached 2e-14)
    ls_tol_floor=1e-15,          # tol passed to SIMSOPT, times G^2: an early exit below the rounding floor
    ls_converged=3e-14,          # a level has converged when the final |J^T r| <= this G^2 (M/10)^2: 10-30 times
                                 # the floor seen on the pilots ((1-5)e-15 G^2, growing with M), scaled like it, as
                                 # BoozerExact's Newton tolerance is; a fixed 1e-14 sits under that floor, and every
                                 # level of four pilots then failed
    label_accept_rel=1e-4,       # a level converges only if the penalty also kept the flux label within this of its
                                 # target (the constrained problem BoozerLS solves; a collapsed surface also has
                                 # zero gradient)
    bfgs_tol=1e-10,              # SIMSOPT's defaults for the BFGS stage of run_code
    bfgs_maxiter=1500,
    limited_memory=False,        # SIMSOPT's default: dense BFGS ("generally closer to optimality")
    weight_inv_modB=True,        # SIMSOPT's default for BoozerLS
    label_ntheta_min=256,        # flux label: trapezoid rule round the phi = 0 loop, max(this, 8 M) points
    quad_accept=1e-12,           # coil quadrature: first rule agreeing with its double to this ...
    quad_factor=2,               # ... then this many times that rule
    stop_after_no_gain=99,       # never stop for lack of gain: levels are cheap and the residual falls unevenly
    stop_after_failures=3,       # stop after this many levels in a row that did not converge
    diag_min=64,                 # off-grid diagnostic grid: max(diag_min, 4 M + 2) squared points per field period
    mem_cap_gb=11.5,             # skip levels whose predicted peak memory passes this (10^9 bytes) under the
                                 # 12 GiB limit: M <= 32 (11.6 GiB peak measured), and M = 36 (14 GB) would not fit
                                 # the clock either
    margin_s=60.0,               # leave this much of the budget when predicting whether a level fits
    omp_wait="default",          # how idle threads wait: the libraries' spinning defaults, kept because passive
                                 # waiting did not measure better under fair_cores (pilot 0925611 on the comparison's
                                 # processor: runs that met the timing rule took 44.8 s spinning against 46.5 s
                                 # passive, and on busy cores both broke the rule about as often, 44 and 32 runs for
                                 # 4 that met it)
    save_levels=False,           # pilot studies: write every converged level as a pseudo-answer
)


# the untimed warm-up solve before the timed design (imports and first-call set-up stay out of wall_s), metered
# like the timed run
WARMUP = dict(design="0925611", overrides=dict(levels=[4, 10, 12]))


def ndofs(M):
    return 6 * M * M + 6 * M + 1


def grid(M, nfp, q):
    """theta: n = ceil(q (2M+1)) points; phi: the half period [0, 1/(2 nfp)] with both symmetry lines, n/2 intervals.
    With its stellarator-symmetric images this is the uniform n x n grid of a full period (the full-period grid
    holds every interior point twice); it starts at phi = theta = 0, where SIMSOPT's z(0, 0) penalty looks."""
    n = int(np.ceil(q * (2 * M + 1)))
    nh = (n + 1) // 2
    return np.linspace(0, 1 / (2 * nfp), nh + 1), np.linspace(0, 1, n, endpoint=False)


def predicted_mem_gb(M, q):
    """LM level: about ten arrays of npts x 3 x D (the gamma derivatives, dB/dc, the residual Jacobian, its
    weighted, scaled and stacked copies) and J^T J; calibrated on the pilots (18.4 GB at M = 40, 81 x 81 points)."""
    D = ndofs(M) + 2
    qp, qt = grid(M, 1, q)
    return (10 * len(qp) * len(qt) * 3 * D + 3 * D * D) * 8 / 1e9 + 0.6


def solve(rec, S, ckpt, t0, budget):
    from simsopt.field import BiotSavart
    from simsopt.geo import BoozerSurface, ToroidalFlux

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
    bs, bs_tf, bs_d = BiotSavart(coils), BiotSavart(coils), BiotSavart(coils)

    G, iota = C.boozer_estimates(probe, bs)
    loop0 = C.start_xyz(rec, [0.0], np.linspace(0, 1, 512, endpoint=False))
    f0 = float(ToroidalFlux(loop0, bs_tf).J())
    target = float(np.sign(f0)) * req
    C.log(f"start: G {G:.12g}  iota {iota:.12g} (QUASR mean_iota {rec['database_metadata']['mean_iota']}), "
          f"flux of stored surface {f0:.12g}, target label {target:.12g}")
    state["start"] = dict(G=G, iota=iota, stored_flux=f0, target_label=target, label_weight=S["constraint_weight"] / target ** 2,
                          coil_quadrature=nq,
                          coil_quadrature_ladder=qtab)

    M0 = max(int(st["mpol"]), int(st["ntor"]))
    levels = [M0] + [m for m in S["levels"] if m > M0]
    cur = C.start_xyz(rec, [0.0], [0.0])
    lad = C.Ladder(S, t0, budget, state, ckpt, save)
    w = S["constraint_weight"] / target ** 2

    for M in levels:
        if not lad.fits(M, predicted_mem_gb(M, S["oversample"]), ndofs(M) + 2):
            break
        qp, qt = grid(M, nfp, S["oversample"])
        s = C.lift_xyz(cur, M, M, qp, qt)
        nth = max(S["label_ntheta_min"], 8 * M)
        lab_s = C.xyz_surface(nfp, ss, M, M, [0.0], np.linspace(0, 1, nth, endpoint=False), dofs=s.dofs)
        label = ToroidalFlux(lab_s, bs_tf)
        h = dict(M=M, unknowns=ndofs(M) + 2, quadpoints=len(qp) * len(qt), lift_err_m=C.same_surface_err(cur, s),
                 iota0=iota, G0=G)
        tl = time.time()
        x_in = s.x.copy()

        def bfgs(booz, i0, g0):
            r = booz.minimize_boozer_penalty_constraints_LBFGS(
                tol=S["bfgs_tol"], maxiter=S["bfgs_maxiter"], constraint_weight=w, iota=i0, G=g0,
                limited_memory=S["limited_memory"], weight_inv_modB=S["weight_inv_modB"], verbose=True)
            booz.need_to_run_code = True
            return r

        conv_tol = S["ls_converged"] * G * G * (M / 10.0) ** 2

        def lm(booz, i0, g0):
            r = booz.minimize_boozer_penalty_constraints_ls(
                tol=S["ls_tol_floor"] * G * G, maxiter=S["ls_maxiter"], constraint_weight=w, iota=i0, G=g0,
                method=S["ls_method"], weight_inv_modB=S["weight_inv_modB"])
            booz.need_to_run_code = True
            return r

        if state["best"] is None:
            # the example's path first, then LM straight from QUASR's surface, then with QUASR's target iota
            paths = [("BFGS+LM", iota), ("LM", iota),
                     ("LM", float(np.sign(iota)) * rec["database_metadata"]["mean_iota"])]
        else:
            paths = [("LM", iota), ("BFGS+LM", iota)]
        tries = []
        ok = False
        for how, i0 in paths:
            s.x = x_in
            tt = time.time()
            try:
                booz = BoozerSurface(bs, s, label, target, constraint_weight=w, options=dict(verbose=True))
                g0 = G
                note = how
                if how == "BFGS+LM":
                    rb = bfgs(booz, i0, g0)
                    note = f"BFGS ({'success' if rb['success'] else 'not converged'}, {rb['iter']} it) + LM"
                    i0, g0 = rb["iota"], rb["G"]
                res = lm(booz, i0, g0)
                lab_err = float(label.J() - target)
                grad = np.asarray(res["gradient"])
                r = np.asarray(res["residual"])
                kept = abs(lab_err) <= S["label_accept_rel"] * abs(target)
                gnorm = float(np.linalg.norm(grad))
                ok = (bool(res["success"]) or gnorm <= conv_tol) and kept
                it_i, it_G = float(res["iota"]), float(res["G"])
                tries.append(dict(how=note, iota0=i0, lm_success=bool(res["success"]), converged=gnorm <= conv_tol,
                                  conv_tol=conv_tol, label_kept=kept, iota=it_i,
                                  label_err_Wb=lab_err, gradient_norm=float(np.linalg.norm(grad)),
                                  seconds=time.time() - tt))
                h.update(success=ok, how=note, iota=it_i, G=it_G, gradient_norm=gnorm, conv_tol=conv_tol,
                         gradient_max=float(np.max(np.abs(grad))), residual_rms=float(np.sqrt(np.mean(r[:-2] ** 2))),
                         label_err_Wb=lab_err)
                del res, booz
            except Exception as e:  # noqa: BLE001
                tries.append(dict(how=how, error=repr(e)[:300], seconds=time.time() - tt))
                h.update(success=False, error=repr(e)[:300])
            if ok:
                break
        if len(tries) > 1:
            h["tries"] = tries
        h["seconds"] = time.time() - tl
        h["maxrss_GB"] = C.maxrss_gb()
        lad.ran(ndofs(M) + 2, tries[-1]["seconds"] if tries else h["seconds"], 5)   # the path that ran last
        if ok:
            h.update(C.boozer_offgrid(s, it_i, it_G, bs_d, max(S["diag_min"], 4 * M + 2)))

            def make_best():
                surf, cp, native = C.xyz_answer_parts(s, dict(iota=it_i, G=it_G))
                return dict(
                    solver_status=(f"Levenberg-Marquardt on the penalty least squares converged ("
                                   f"|J^T r| {h['gradient_norm']:.2e} <= tolerance {h['conv_tol']:.2e} after up to {S['ls_maxiter']} "
                                   f"iterations; flux label within "
                                   f"{abs(h['label_err_Wb']) / abs(target):.1e} of its target) at mpol=ntor={M}, the "
                                   f"converged level with the smallest off-grid Boozer residual"),
                    surface=surf, check_points=cp, native=native,
                    solver_reported=dict(level_M=M, iota=it_i, G=it_G, penalty_gradient_norm=h["gradient_norm"],
                                         residual_rms=h["residual_rms"], path=h["how"], flux_label_Wb=lab_err + target,
                                         flux_label_err_Wb=lab_err,
                                         offgrid_boozer_rel_max=h["diag_boozer_rel_max"],
                                         offgrid_boozer_rel_rms=h["diag_boozer_rel_rms"],
                                         offgrid_bn_max=h["diag_bn_max"], coil_quadrature=nq))

            h["chosen"] = lad.converged(h["diag_boozer_rel_max"], make_best)
            cur, iota, G = s, it_i, it_G
            if S["save_levels"]:
                surf, cp, _ = C.xyz_answer_parts(s)
                C.atomic_write(Path(ckpt).parent.parent / "levels" / f"quasr_{rec['id']}_M{M:02d}.json",
                               dict(design=rec["id"], solver=f"{KEY}@M{M}", status="surface", surface=surf,
                                    check_points=cp, level=h))
        else:
            lad.failed()
        state["history"].append(h)
        C.log("level", C.short({k: v for k, v in h.items() if k != "tries"}))
        if len(tries) > 1:
            C.log("  tries", tries)
        save()
        if not lad.more():
            break
    lad.finish("the penalty minimisation did not converge at any level")


if __name__ == "__main__":
    C.main(KEY, LABEL, solve, DEFAULTS, __doc__, warmup=WARMUP)
