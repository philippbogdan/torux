"""SIMSOPT 1.11.1 BoozerExact on one QUASR design: the surface at the requested toroidal flux.

    python run_boozer_exact.py 0003051              # a sample design -> compare300-out/simsopt-boozer-exact/
    python run_boozer_exact.py 0224040 --pilot      # a tuning design (inputs/pilot/)
    python run_boozer_exact.py 0003051 --set 'levels=[10, 12, 14]' --no-warmup     # a quick check

BoozerExact solves the Boozer residual G B - |B|^2 (x_phi + iota x_theta) = 0 exactly at as many collocation points
as the surface has unknowns, plus the toroidal-flux label, by Newton's method (SIMSOPT's own routine, unmodified).
A solve at one resolution is only as good as its truncation, so the harness climbs a resolution ladder, each level
starting from the previous level's answer padded exactly with zeros, and keeps the converged level whose Boozer
residual between the collocation points (SIMSOPT's own residual, at points the solve never saw) is smallest.

The flow: a coil quadrature from SIMSOPT's own field at QUASR's stored surface; G, iota and the flux label's sign
from the field on that surface; then levels M0 (QUASR's stored resolution, 4 or 10), 12, 14, ..., 28, 32, 36, 40,
each BoozerSurface(bs, s, ToroidalFlux, target, constraint_weight=None).solve_residual_equation_exactly_newton(tol,
maxiter, iota, G); the answer is the converged level (success=True) with the smallest off-grid residual, none if
no level converged, converted exactly to the evaluator's xyz_fourier form with 32 check points from SIMSOPT's own
gamma().

Choices that are not settings below, and why:

- Method: BoozerSurface.solve_residual_equation_exactly_newton (constraint_weight=None), SIMSOPT's BoozerExact as
  released: collocation Newton with iterative refinement of each step.
- Surface: SurfaceXYZTensorFourier, stellarator symmetric, mpol = ntor = M: the only surface BoozerExact supports;
  mpol = ntor as in SIMSOPT's examples.
- Collocation grid: phi linspace(0, 1/(2 nfp), M+1), theta linspace(0, 1, 2M+1), both without the end point: one
  of the three grids SIMSOPT's get_stellsym_mask accepts, with half the points (and memory) of the full-period grid
  for the same equations.
- Flux label: ToroidalFlux on the phi = 0 loop, sharing the surface's degrees of freedom (SIMSOPT's own pattern for
  ToroidalFlux(..., ntheta=)); its target is the requested flux with the sign of SIMSOPT's flux of the stored
  surface, so the sign follows the coils' field.
- G and iota to start: least squares of G B/|B|^2 = x_phi + iota x_theta on the stored surface, which gives both
  signs and values (G agrees with mu0 sum |I| to 1e-6, iota with QUASR's target to about 1e-3).
- The answer among levels: the collocation residual is zero by construction, so the residual between the points,
  |r|/(|G||B|) at (4M+2)^2 points offset from the grid, is the solver's own measure of truncation.  On the six
  pilots it tracks the evaluator at every level (within a factor 2.5) and picks the evaluator's best level on all
  six.

How the settings were chosen: on six designs outside the sample (inputs/pilot/), every level of the ladder kept,
each measured by the comparison's evaluator.  The pilot runs were on several machines, so their times are only
indicative.

    design   iota     answer   SIMSOPT off-grid |B.n|/|B|   evaluator max (median)   wall
    0224040  0.6002   M = 10   4.8e-3                       4.9e-3  (5.5e-4)          47 s   3/5 resonance: M >= 12 diverge
    0925611  0.3014   M = 36   1.8e-14                      1.5e-14 (6.4e-16)        103 s   at the floor from M = 32
    1101401  0.49997  M = 10   5.7e-5                       5.7e-5  (7.0e-6)          38 s   m = 8 resonant (nfp 4)
    1981226  2.90     M = 40   2.4e-7                       2.5e-7  (1.3e-9)         144 s   still falling at M = 40
    2126458  1.200    M = 10   3.5e-4                       3.4e-4  (9.2e-6)          36 s   M >= 16 diverge
    2559727  2.865    M = 40   2.6e-11                      2.2e-11 (1.2e-14)        149 s   about 10x per 4 modes

Time does not limit BoozerExact here (a full ladder to M = 40 takes 2 to 3 minutes on 4 cores); memory does.
M = 40 peaked at 9.8 to 11.8 GiB when Newton took at most 2 steps there, and up to 15.4 GiB when it took 5 or failed;
past 12 GiB the harness stops the solver and the answer is the best level it had converged.  M = 44 (15 GiB or more)
does not fit 12 GiB.  Twelve of the sample designs were also run with 16 GiB, where M = 40 and M = 44 both converged:
by SIMSOPT's own off-grid |B.n|/|B|, M = 44 leaks a median 1.7 times less than M = 40: 20 times less at most, and
on three of the twelve up to 1.13 times more (boozer_exact_m44.csv).

The comparison's answers: these settings, `--mem-gb 12`, the default budget, SIMSOPT 1.11.1, numpy 2.5.4,
scipy 1.18.1, jax 0.12.0.1, Python 3.12.14, 4 compute threads in a process with 8 cores of an AMD EPYC 7742 to
itself (requirements/simsopt.txt).

Timing: wall_s runs from the start of the solver's work on the design (before its input file is read) to its answer
(the solver's last checkpoint written), inside one fresh process per design that has already imported SIMSOPT and
finished one untimed warm-up solve on pilot design 0925611 (a few levels of this solver).  Imports and first-call
set-up are excluded; everything the solver redoes per design (coil quadrature, field, signs, every level, the
off-grid checks, the conversion to the evaluator's form) is included.  The warm-up and the timed run are each
measured by fair_cores.Meter on Linux; the comparison counted a timed run only if its threads waited for a CPU at
most 5% of the time they ran and other work used its CPUs for at most 5% of that time, and the answer records
whether this run would have counted.  The process that writes the answer is not the one that solves: the solver
runs as a child process, which stops itself at 3,600 s; the parent stops it if its resident memory passes 12 GiB,
and writes the answer from the child's last checkpoint (the best converged level so far), so a design that runs out
of time or memory still gets its answer.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simsopt_common as C  # noqa: E402  (sets the thread counts before numpy loads)

np = C.np

KEY, LABEL = "simsopt-boozer-exact", "SIMSOPT BoozerExact"

DEFAULTS = dict(
    levels=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40],   # a collocation solve is only as good as its
                                 # truncation: on the pilots the off-grid residual falls 3 to 8 times per step of 2
                                 # until the floor or a resonance.  12 GiB ends the ladder at M = 40 (M = 44 needs
                                 # 15 GiB or more)
    newton_tol_rel=1e-14,        # Newton tolerance per equation, relative to |G| |B|: SIMSOPT's tol is absolute,
                                 # tol = newton_tol_rel |G| <|B|> sqrt(equations).  SIMSOPT's default (absolute
                                 # 1e-13) is the rounding floor itself from M ~ 26 or for larger G B: with it the
                                 # pilots stalled at 1.1e-13 to 2.9e-13 (4e-16 per equation) and four were declared
                                 # unconverged from M = 10 to 28 while still improving; the relative tolerance
                                 # removes those false failures and leaves the off-grid residuals unchanged
    newton_maxiter=10,           # per level after the first (Newton from the padded previous level needs 2 to 5)
    first_maxiter=30,            # the first level starts from QUASR's surface, further away
    presolve=True,               # restart a first level that fails with a BoozerLS pre-solve (SIMSOPT run_code).
                                 # BoozerExact has no line search: on 0224040 (iota 0.6004, by the 3/5 resonance)
                                 # the field's iota (0.60035) diverges at M = 10 and QUASR's target iota (0.6)
                                 # converges; a BoozerLS pre-solve reaches the same solution (three pilots, M = 10)
    presolve_oversample=1.5,     # its grid: 1.5 (2M + 1) points per direction over one field period
    presolve_weight=100.0,       # its label weight (SIMSOPT's examples/2_Intermediate/boozer.py)
    label_ntheta_min=256,        # flux label: trapezoid rule round the phi = 0 loop, max(this, 8 M) points.  The
                                 # surface's own 2M+1 theta points integrate the flux only to about 1e-4 at M = 10
                                 # (0224040: 21 points against 512 differ by 9e-5); the label is one equation, so a
                                 # converged rule costs nothing
    quad_accept=1e-12,           # coil quadrature: first rule N0 in 96, 128, 160, ... agreeing with its double to
                                 # this (max |B(N0) - B(2 N0)|/|B| on the stored surface) ...
    quad_factor=2,               # ... then this many times that rule (192 to 640 points per coil on the pilots).
                                 # QUASR's 160 points are far too few (2e-8 on 0224040); the trapezoid rule converges
                                 # geometrically, so 2 N0 is about 1e-24 off, under B's own rounding (~5e-15)
    stop_after_no_gain=99,       # never stop for lack of gain: levels are cheap and the residual falls unevenly (on
                                 # pilot 1981226 it goes 6.5e-3, 7.0e-3, 4.2e-3 at M = 14, 16, 18, so a rule of two
                                 # levels without halving would stop there at 1.4e-3, against 2.5e-7 at M = 40)
    stop_after_failures=3,       # stop after this many levels in a row whose Newton did not converge
    diag_min=64,                 # off-grid diagnostic grid: max(diag_min, 4 M + 2) squared points per field period
    mem_cap_gb=11.5,             # skip levels whose predicted peak memory passes this, in 10^9 bytes (10.7 GiB;
                                 # 1.05 x SIMSOPT's dense arrays: 10.4 GB at M = 40); the harness stops the solver
                                 # at 12 GiB resident, keeping the levels it has converged
    margin_s=60.0,               # leave this much of the budget when predicting whether a level fits
    omp_wait="default",          # how idle threads wait: the libraries' spinning defaults, kept because passive
                                 # waiting did not measure better under fair_cores (pilot 0925611 on the comparison's
                                 # processor: runs that met the timing rule took 23.9 s spinning against 25.9 s
                                 # passive, and on busy cores both broke the rule about as often, 48 and 44 runs for
                                 # 4 that met it)
    save_levels=False,           # pilot studies: write every converged level as a pseudo-answer
)


# the untimed warm-up solve before the timed design (imports and first-call set-up stay out of wall_s), metered
# like the timed run
WARMUP = dict(design="0925611", overrides=dict(levels=[4, 10, 12, 14, 16, 18, 20]))


def ndofs(M):
    """Surface unknowns of a stellarator-symmetric SurfaceXYZTensorFourier with mpol = ntor = M."""
    return 6 * M * M + 6 * M + 1


def predicted_mem_gb(M):
    """Peak of one level: SIMSOPT holds the d gamma / d coefficient arrays (npts, 3, D) for gamma and both
    tangents, dB/dc and the residual Jacobian of the same size, the masked Jacobian, LAPACK's copy and the final
    P, L, U (D, D) each."""
    D = ndofs(M) + 2
    npts = (M + 1) * (2 * M + 1)
    return 1.05 * (6 * npts * 3 * D + 6 * D * D) * 8 / 1e9 + 0.6    # pilots: 9.7 GB measured at M = 40


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

    # the coil quadrature, from SIMSOPT's own field at the stored surface (one field period)
    probe = C.start_xyz(rec, np.linspace(0, 1 / nfp, 24, endpoint=False), np.linspace(0, 1, 24, endpoint=False))
    nq, qtab = C.coil_quadrature(rec, probe.gamma().reshape(-1, 3), accept=S["quad_accept"], factor=S["quad_factor"])
    C.log(f"coil quadrature: {nq} points per coil; |B(N) - B(2N)|/|B| ladder {qtab}")
    coils = C.simsopt_coils(rec, nq)
    bs, bs_tf, bs_d = BiotSavart(coils), BiotSavart(coils), BiotSavart(coils)

    # signs and first values of G and iota, from the stored surface and the field
    G, iota = C.boozer_estimates(probe, bs)
    Bmean = float(np.mean(np.linalg.norm(bs.B(), axis=1)))
    loop0 = C.start_xyz(rec, [0.0], np.linspace(0, 1, 512, endpoint=False))
    f0 = float(ToroidalFlux(loop0, bs_tf).J())
    target = float(np.sign(f0)) * req
    C.log(f"start: G {G:.12g}  iota {iota:.12g} (QUASR mean_iota {rec['database_metadata']['mean_iota']}), "
          f"flux of stored surface {f0:.12g}, target label {target:.12g}")
    state["start"] = dict(G=G, iota=iota, Bmean=Bmean, stored_flux=f0, target_label=target, coil_quadrature=nq,
                          coil_quadrature_ladder=qtab)

    M0 = max(int(st["mpol"]), int(st["ntor"]))
    levels = [M0] + [m for m in S["levels"] if m > M0]
    cur = C.start_xyz(rec, [0.0], [0.0])
    lad = C.Ladder(S, t0, budget, state, ckpt, save)

    def newton_at(M, start, iota0, G0, maxiter):
        """SIMSOPT's BoozerExact Newton at mpol = ntor = M from `start` padded exactly; returns (ok, s, h)."""
        qp = np.linspace(0, 1 / (2 * nfp), M + 1, endpoint=False)     # SIMSOPT's stellsym BoozerExact grid
        qt = np.linspace(0, 1, 2 * M + 1, endpoint=False)
        s = C.lift_xyz(start, M, M, qp, qt)
        nth = max(S["label_ntheta_min"], 8 * M)
        lab_s = C.xyz_surface(nfp, ss, M, M, [0.0], np.linspace(0, 1, nth, endpoint=False), dofs=s.dofs)
        label = ToroidalFlux(lab_s, bs_tf)
        tol = S["newton_tol_rel"] * abs(G0) * Bmean * np.sqrt(ndofs(M) + 2)
        h = dict(M=M, unknowns=ndofs(M) + 2, lift_err_m=C.same_surface_err(start, s), iota0=iota0, G0=G0,
                 maxiter=maxiter, tol=tol)
        tl = time.time()
        try:
            booz = BoozerSurface(bs, s, label, target, options=dict(verbose=True))
            res = booz.solve_residual_equation_exactly_newton(tol=tol, maxiter=maxiter, iota=iota0, G=G0,
                                                              verbose=True)
            r = np.asarray(res["residual"])[res["mask"]]
            lab_err = float(label.J() - target)
            h.update(success=bool(res["success"]), iterations=int(res["iter"]), iota=float(res["iota"]),
                     G=float(res["G"]), residual_norm=float(np.linalg.norm(np.concatenate([r, [lab_err]]))),
                     residual_max=float(np.max(np.abs(r))), label_err_Wb=lab_err)
            del res, booz
        except Exception as e:  # noqa: BLE001
            h.update(success=False, error=repr(e)[:300])
        h["seconds"] = time.time() - tl
        h["maxrss_GB"] = C.maxrss_gb()
        return h["success"], s, h

    def ls_presolve(M, start, iota0, G0):
        """SIMSOPT's BoozerLS (run_code: BFGS then Newton on the penalty) at mpol = ntor = M: a start for Newton."""
        n = int(np.ceil(S["presolve_oversample"] * (2 * M + 1)))
        s = C.lift_xyz(start, M, M, np.linspace(0, 1 / nfp, n, endpoint=False), np.linspace(0, 1, n, endpoint=False))
        lab_s = C.xyz_surface(nfp, ss, M, M, [0.0], np.linspace(0, 1, max(S["label_ntheta_min"], 8 * M),
                                                                   endpoint=False), dofs=s.dofs)
        booz = BoozerSurface(bs, s, ToroidalFlux(lab_s, bs_tf), target, constraint_weight=S["presolve_weight"],
                             options=dict(verbose=True))
        res = booz.run_code(iota0, G0)
        return bool(res["success"]), s, float(res["iota"]), float(res["G"])

    for M in levels:
        if not lad.fits(M, predicted_mem_gb(M), ndofs(M) + 2):
            break
        if state["best"] is None:
            # restarts until a first level converges: the field's G and iota, QUASR's target iota, then a BoozerLS
            # pre-solve of the same resolution as Newton's start
            starts = [("field estimate", cur, iota, G),
                      ("QUASR mean_iota", cur, float(np.sign(iota)) * rec["database_metadata"]["mean_iota"], G)]
            if S["presolve"]:
                starts.append(("BoozerLS pre-solve", None, iota, G))
            maxiter = S["first_maxiter"]
        else:
            starts, maxiter = [("previous level", cur, iota, G)], S["newton_maxiter"]
        tries = []
        for how, start, i0, g0 in starts:
            if start is None:
                tl = time.time()
                try:
                    pre_ok, start, i0, g0 = ls_presolve(M, cur, iota, G)
                    tries.append(dict(how=how, success=pre_ok, iota=i0, G=g0, seconds=time.time() - tl))
                except Exception as e:  # noqa: BLE001
                    tries.append(dict(how=how, error=repr(e)[:300], seconds=time.time() - tl))
                    continue
            ok, s, h = newton_at(M, start, i0, g0, maxiter)
            h["start"] = how
            if tries:
                h["tries"] = list(tries)
            tries.append({k: h.get(k) for k in ("start", "success", "iterations", "residual_norm", "iota",
                                                "seconds", "error")})
            if ok:
                break
        lad.ran(ndofs(M) + 2, h["seconds"], h.get("iterations", maxiter))
        if ok:
            it_i, it_G, lab_err, tol = h["iota"], h["G"], h["label_err_Wb"], h["tol"]
            h.update(C.boozer_offgrid(s, it_i, it_G, bs_d, max(S["diag_min"], 4 * M + 2)))

            def make_best():
                surf, cp, native = C.xyz_answer_parts(s, dict(iota=it_i, G=it_G))
                return dict(
                    solver_status=(f"Newton converged (success=True: |residual| {h['residual_norm']:.2e} <= tol "
                                   f"{tol:.2e}) at mpol=ntor={M}, the converged level with the smallest off-grid "
                                   f"Boozer residual"),
                    surface=surf, check_points=cp, native=native,
                    solver_reported=dict(level_M=M, iota=it_i, G=it_G, newton_residual_norm=h["residual_norm"],
                                         newton_tol=tol, newton_iterations=h["iterations"], start=h["start"],
                                         flux_label_Wb=lab_err + target, flux_label_err_Wb=lab_err,
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
            if len(tries) > 1:
                h["tries"] = tries
            lad.failed()
        state["history"].append(h)
        C.log("level", C.short({k: v for k, v in h.items() if k != "tries"}))
        if len(tries) > 1:
            C.log("  tries", tries)
        save()
        if not lad.more():
            break
    lad.finish("Newton did not converge at any level")


if __name__ == "__main__":
    C.main(KEY, LABEL, solve, DEFAULTS, __doc__, warmup=WARMUP)
