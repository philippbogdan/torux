"""The family of flux surfaces around a magnetic axis: continuation in s = sqrt(|flux|).

From the near-axis torus (the axis plus a small circle, which Newton turns into the Floquet
ellipse family in one or two iterations because the problem is nearly linear there) the
tori are followed outward in s: predictor = linear extrapolation of the coefficients in s
from the last two tori, corrector = Newton at the next flux.  Resolutions come from a fixed
ladder (so compiled programs are reused); after each converged step the resolution is raised
while the off-grid |B.n|/|B| exceeds ``bn_cont`` and the frequency-weighted tails say which
direction truncates.  A step is accepted when Newton converges, the off-grid B.n is below
``bn_accept`` and iota moves continuously; a failed step halves ds; below ``ds_min`` the
continuation stops and reports where (a resonance it cannot cross, or the edge).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .torus import TorusProblem, TorusState
from . import budget as _budget
from . import fast_design as _so
from . import fast_family as _so2
from . import fast_newton as _so3

LADDER = [6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
          448, 512, 640, 768, 896, 1024]


def ladder_up(v, factor=1.25, cap=None):
    """Next ladder value at least factor * v (and > v), at most cap -- but never below v: a torus already above the
    cap (a design torus refined past the continuation's Mmax) keeps its resolution."""
    if cap and v >= cap:
        return v
    for L in LADDER:
        if L > v and L >= factor * v:
            return min(L, cap) if cap else L
    return cap or LADDER[-1]


@dataclass
class FamilyResult:
    tori: list = field(default_factory=list)       # accepted TorusState, increasing |flux|
    stop_reason: str = ""
    stop_flux: float = np.nan
    log: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def axis_torus_start(axis, M, N, eps, iota):
    """Axis plus a circle of radius eps: the starting guess near the axis."""
    st = TorusState(np.zeros((M + 1, 2 * N + 1)), np.zeros((M + 1, 2 * N + 1)), iota, 0.0, M, N, axis.nfp)
    na = min(N, axis.N)
    st.rc[0, N:N + na + 1] = axis.rc[:na + 1]
    # Z = sum_n zs[0, n] sin(-n psi) on the m = 0 row; the axis has Z = sum zs_n sin(n psi)
    st.zs[0, N + 1:N + na + 1] = -axis.zs[1:na + 1]
    st.rc[1, N] = eps
    st.zs[1, N] = eps
    return st


class Continuation:
    def __init__(self, engine, nfp, M0=6, N0=None, bn_cont=1e-9, bn_accept=1e-5, tol=1e-12, Mmax=128, Nmax=128,
                 verbose=False, phi0=0.0, lean_source=None, lean_dmin=None):
        self.engine = engine
        # lean_source: the family run on an engine sized for the region ahead of it (rebuilt as it grows)
        self.lean_source, self.lean_dmin = lean_source, lean_dmin
        self._lean_flux = None
        self._broad = engine
        self.nfp = nfp
        self.phi0 = phi0
        self.M0, self.N0 = M0, N0
        self.bn_cont, self.bn_accept, self.tol = bn_cont, bn_accept, tol
        self.Mmax, self.Nmax = Mmax, Nmax
        self.verbose = verbose
        self._probs = {}
        # fast_design 'predict': the Newton-Kantorovich state (measured remainder constant) carried between the
        # solves of one torus (design solve, raises, refinement); None = off (set by the design path)
        self.predict_state = None

    DENSE_MAX = 2500          # unknowns up to which the dense Newton is used (above: Newton-Krylov, which was 2.6x
                              # faster at 5600 unknowns, QUASR 0636385, same torus to rounding)
    DENSE_FALLBACK = 8000     # unknowns up to which the dense Newton backs up a failed Newton-Krylov (the
                              # Jacobian assembly's gathers need ~0.1 n^2 kB: 12000 unknowns need 16 GB)

    def problem(self, M, N):
        key = (M, N)
        if key not in self._probs:
            self._probs[key] = TorusProblem(self.engine, self.nfp, M, N, phi0=self.phi0)
        return self._probs[key]

    def krylov(self, M, N):
        from .torus import KrylovTorus
        key = ("nk", M, N)
        if key not in self._probs:
            self._probs[key] = KrylovTorus(self.problem(M, N))
        return self._probs[key]

    FILTER_D = 1e-4           # a harmonic m <= 2M + 10 this close to resonance -> filtered dense step
    FILTER = 1e-5             # filter level relative to the largest singular value
    FILTER_MAX = 5000         # unknowns up to which the filtered (SVD) step is affordable

    def near_resonant(self, iota, M):
        io = abs(float(iota))
        if not np.isfinite(io):
            return False
        for m in range(1, 2 * M + 11):
            n0 = round(io * m / self.nfp)
            if abs(n0 * self.nfp - io * m) < self.FILTER_D:
                return True
        return False

    def solve(self, guess: TorusState, flux, M, N, maxit=15, fixed_iota=None, filtered_ok=False, patience=None,
              floor_stop=None, predict=True, diverge=False):
        """Newton at (M, N).  floor_stop: stop at the arithmetic floor (torux.torus.newton_floor_reached; None =
        stop when two iterations bring no new best).  predict=False: no Newton-Kantorovich stop even where
        predict_state is set."""
        prob = self.problem(M, N)
        n = prob.basis.nc + prob.basis.ns + 1
        pst = self.predict_state if predict else None
        if filtered_ok and n <= self.FILTER_MAX and self.near_resonant(guess.iota if fixed_iota is None else fixed_iota, M):
            sol = prob.newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota, filt=self.FILTER,
                              floor_stop=floor_stop)
            sol.info["filtered"] = True
            if sol.converged:
                sol.res_off, sol.bn_off = prob.check(sol.rc, sol.zs, sol.iota)
            return sol
        dkw = {"diverge": True} if diverge else {}      # (fast_family 'cdiverge'; no keyword when off)
        if n <= self.DENSE_MAX:
            sol = prob.newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota, patience=patience,
                              floor_stop=floor_stop, predict=pst, **dkw)
        else:
            # inexact Newton: GMRES to 1e-6 (restart 30, 2 cycles) gives the same Newton steps as 1e-8
            # (restart 60, 3 cycles) at half the cost (QUASR 954 at M, N = 48, 32)
            sol = self.krylov(M, N).newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota,
                                           gtol=1e-6, restart=30, maxiter=2, patience=patience,
                                           floor_stop=floor_stop, predict=pst, **dkw)
            if not sol.converged and n <= self.DENSE_FALLBACK and not _budget.expired():
                if getattr(self, "_in_family", False) and _so3.on("kfall"):
                    # (fast_newton 'kfall') the family's backup: the same Newton with tight GMRES, not the dense LU
                    sol = self.krylov(M, N).newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota,
                                                   gtol=_so3.KFALL_GTOL, restart=_so3.KFALL_RESTART,
                                                   maxiter=_so3.KFALL_MAXITER, patience=patience,
                                                   floor_stop=floor_stop, predict=pst, **dkw)
                    sol.info["kfall"] = True
                else:
                    sol = prob.newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota, patience=patience,
                                      floor_stop=floor_stop, predict=pst, **dkw)
        if sol.converged:
            sol.res_off, sol.bn_off = prob.check(sol.rc, sol.zs, sol.iota)
        return sol

    def grow(self, sol: TorusState):
        """Next (M, N) by the frequency-weighted tails."""
        tm, tn = sol.tails()
        wm = tm * max(abs(sol.iota), 1e-3) * sol.M
        wn = tn * self.nfp * sol.N
        M2 = ladder_up(sol.M, cap=self.Mmax) if wm >= 0.1 * wn else sol.M
        N2 = ladder_up(sol.N, cap=self.Nmax) if wn >= 0.1 * wm else sol.N
        return M2, N2

    def raise_until(self, sol: TorusState, flux, bn_target, max_raise=4, fixed_iota=None, min_gain=1.0, jump=False,
                    low_gain_patience=1):
        """Raise the resolution while the off-grid B.n exceeds bn_target and improves (by more than min_gain
        per raise: an invariant torus resolved further gains orders of magnitude per ladder step, a torus
        through a chain or a chaotic layer plateaus).  low_gain_patience: stop after that many consecutive
        raises gaining less than min_gain (2: a torus whose spectrum decays slowly at first -- QUASR 1101649
        gained 1.65x from M = 24 to 32, then 3.1x, 133x, 167x -- is not taken for a plateau after one); the best
        torus of the climb is returned.  jump (fast_design 'jump'): each raise to the resolution at which the
        spectrum, extrapolated from its last half, falls below bn_target / 10 (predict_resolution, at most 2x per
        raise) instead of one 1.25x ladder step."""
        best = sol
        low = 0
        for _ in range(max_raise):
            if sol.bn_off <= bn_target or _budget.expired():
                break
            M2, N2 = self.grow(sol)
            if jump:
                Mp, Np = predict_resolution(sol, bn_target, self, floor=0.1 * bn_target)
                if (max(Mp, sol.M), max(Np, sol.N)) != (sol.M, sol.N):
                    M2, N2 = max(Mp, sol.M), max(Np, sol.N)
            if (M2, N2) == (sol.M, sol.N):
                break
            s2 = self.solve(sol, flux, M2, N2, fixed_iota=fixed_iota)
            if not s2.converged:
                break
            gained = s2.bn_off * min_gain < sol.bn_off
            sol = s2                                   # the climb goes on from the higher resolution
            if s2.bn_off < best.bn_off:
                best = s2
            low = 0 if gained else low + 1
            if low >= low_gain_patience:
                break
        sol = best
        return sol

    def start(self, axis):
        N = self.N0 or max(8, axis.N)
        M = self.M0
        eps = 0.004 * axis.rc[0]
        st = axis_torus_start(axis, M, N, eps, axis.iota if np.isfinite(axis.iota) else 0.0)
        f0 = self.problem(M, N).flux_of(st.rc, st.zs)
        from . import nearaxis as _na
        na = _na.opts()
        # (torux.nearaxis) at N > 128 (an axis check_line resolved past find_axes's ladder) the start's Newton stops once it
        # has diverged (torux.fast_family.diverged): 25 diverging iterations at N = 192 took 300 s, the family's whole
        # budget, before the ladder ran.  At N <= 128 the start is as before
        dv = {"diverge": True} if (na["on"] and N > 128) else {}
        sol = self.solve(st, f0, M, N, maxit=25, **dv)
        efrac, lg, rec = 0.004, None, None
        if not sol.converged and na["on"]:
            # (torux.nearaxis) smaller start circles: next to a low-order resonance on the axis (rotation per period ~1/3,
            # 1/4) Newton's first step from the circle at 0.004 R0 crosses the small divisor; from 0.001 R0 it does not
            s2, f2, e2, lg = _na.start_ladder(self, axis, M, N)
            if s2 is not None and s2.converged:
                sol, f0, efrac = s2, f2, e2
        if sol.converged and na["on"] and sol.bn_off > na["start_bn"] and _na.at_resonance(sol.iota, self.nfp, na):
            # (torux.nearaxis) the start converged at a low-order resonance (rotation per period within res_tol of 1/3,
            # 1/4): a torus in an island chain, not invariant -- the smaller circles before raising it (raising such a
            # torus cost 250 s of a 180 s family budget, CoilStellaration D67czaES)
            small = [e for e in na["eps"] if e < efrac]
            s4, f4, e4, lg4 = _na.start_ladder(self, axis, M, N, small)
            if s4 is not None and s4.converged and s4.bn_off < sol.bn_off:
                sol, f0, efrac = s4, f4, e4
            lg = (lg or []) + [dict(at_resonance=True, **d) for d in lg4]
        if sol.converged:
            sol = self.raise_until(sol, f0, self.bn_cont)
            if na["on"] and sol.bn_off > na["start_bn"] and sol.N >= self.Nmax and _na.tail(axis) >= na["tail_ok"]:
                # (torux.nearaxis) a start torus the cap keeps above the range's bar, around an axis resolved only to
                # that cap (a line next to a coil): the axis and the start at N = 192, 256
                s3, f3, rec = _na.resolve_start(self, axis, sol, M, efrac, na)
                if s3 is not None:
                    sol, f0 = s3, f3
            if na["on"] and sol.bn_off > na["start_bn"]:
                # (torux.nearaxis) a start that converged across the resonance (iota of the circle's Newton wandered past
                # p / q, a torus that is not invariant: bn_off 1e-4): the smaller circles, the best start kept
                small = [e for e in na["eps"] if e < efrac]
                s4, f4, e4, lg4 = _na.start_ladder(self, axis, M, sol.N, small) if small else (None, None, None, [])
                if s4 is not None and s4.converged:
                    s4 = self.raise_until(s4, f4, self.bn_cont)
                    lg4[-1]["bn_raised"] = float(s4.bn_off)
                    if s4.bn_off < sol.bn_off:
                        sol, f0 = s4, f4
                lg = (lg or []) + [dict(after_bn=True, **d) for d in lg4]
        if lg is not None:
            sol.info["start_ladder"] = lg
        if rec is not None:
            sol.info["resolve_start"] = rec
        return sol, f0

    def run(self, axis, flux_target, ds0=0.05, ds_max=0.2, ds_min=2e-3, deadline=None, start=None):
        # the family's own deadline also holds inside its Newton solves (a budget scope): checked only between
        # steps, a continuation step near the edge -- several high-resolution Newton solves -- can overrun it
        if not (_so3.on("kfall") or _so3.on("flean") or _so3.on("fmeas")):
            with _budget.scope(deadline):
                return self._run(axis, flux_target, ds0, ds_max, ds_min, _budget.earliest(deadline), start)
        # (fast_newton) the family's own solves are marked (kfall: their backup Newton; flean: their engine); when
        # the family ends, the standard engine is restored for everything after it
        self._in_family = True
        try:
            with _budget.scope(deadline):
                return self._run(axis, flux_target, ds0, ds_max, ds_min, _budget.earliest(deadline), start)
        finally:
            self._in_family = False
            self._flean_restore()

    def _run(self, axis, flux_target, ds0, ds_max, ds_min, deadline, start):
        t0 = time.time()
        out = FamilyResult()
        if start is None:
            sol, f0 = self.start(axis)
            if not sol.converged:
                out.stop_reason = "near-axis torus did not converge"
                return out
        else:
            sol = start
            f0 = sol.flux
        out.tori.append(sol)
        if not (np.isfinite(f0) and f0 != 0.0):
            # a start without flux has no square-root-flux parameter to step in
            out.stop_reason = "start torus without flux"
            return out
        sgn = math.copysign(1.0, f0)
        s_target = math.sqrt(abs(flux_target))
        s = math.sqrt(abs(f0))
        ds = ds0 * s_target
        while s < s_target * (1 - 1e-12):
            if deadline is not None and time.time() > deadline:
                out.stop_reason = "deadline"
                break
            self._lean_update(out.tori[-1])
            s_new = min(s + ds, s_target)
            guess = self.predict(out.tori, s_new, sgn)
            last = out.tori[-1]
            dv = {"diverge": True} if _so2.on("cdiverge") else {}
            sol = self.solve(guess, sgn * s_new ** 2, last.M, last.N, maxit=10, patience=self.STEP_PATIENCE, **dv)
            if not sol.converged and np.isfinite(sol.res) and sol.res < self.PLATEAU_RETRY * max(1.0, abs(last.rc[0, last.N])):
                # a Newton that levels off just short of the tolerance: the torus at this resolution's truncation
                # floor (the shaping grows outward), not a failed step -- once more, one ladder step up
                M2, N2 = self.grow(sol)
                if (M2, N2) != (sol.M, sol.N) and not _budget.expired():
                    sol2 = self.solve(sol, sgn * s_new ** 2, M2, N2, maxit=10, patience=self.STEP_PATIENCE, **dv)
                    if sol2.converged or (np.isfinite(sol2.res) and sol2.res < sol.res):
                        sol = sol2
            ok = sol.converged and np.isfinite(sol.bn_off)
            if ok:
                sol = self.raise_until(sol, sgn * s_new ** 2, self.bn_cont, max_raise=3)
                ok = sol.bn_off < self.bn_accept
            if ok and len(out.tori) >= 2:
                prev = out.tori[-2]
                sl, sp = math.sqrt(abs(last.flux)), math.sqrt(abs(prev.flux))
                slope = (last.iota - prev.iota) / max(sl - sp, 1e-300)
                ok = abs(sol.iota - last.iota) <= 10 * abs(slope) * (s_new - sl) + 1e-3
            out.log.append(dict(s=s_new, ds=ds, ok=bool(ok), iota=sol.iota, bn=sol.bn_off, M=sol.M, N=sol.N,
                                its=sol.its, res=sol.res, t=time.time() - t0))
            if self.verbose:
                print(f"  s {s_new / s_target:.5f} ds {ds / s_target:.2e} ok {ok} iota {sol.iota:.10f} "
                      f"bn {sol.bn_off:.2e} M,N {sol.M},{sol.N} its {sol.its} t {time.time() - t0:.1f}s", flush=True)
            if ok:
                out.tori.append(sol)
                s = s_new
                ds = min(ds * 1.5, ds_max * s_target)
            else:
                if _budget.expired() or (deadline is not None and time.time() > deadline):
                    # the step was cut by the deadline: not a stall (the label decides what the next pass does)
                    out.stop_reason = "deadline"
                    out.stop_flux = sgn * s ** 2
                    break
                ds *= 0.5
                if ds < ds_min * s_target:
                    out.stop_reason = "stalled"
                    out.stop_flux = sgn * s ** 2
                    break
        else:
            out.stop_reason = "reached"
        out.timings["total"] = time.time() - t0
        return out

    # no Newton patience on a continuation step (None): a patience of 2 stalls families that otherwise reach their
    # surfaces, and slows the rest (each failed step halves ds: more steps).  A step that levels off below
    # PLATEAU_RETRY (relative) is retried one ladder step up before it counts as failed
    STEP_PATIENCE = None
    PLATEAU_RETRY = 1e-6

    LEAN_SCALE = 1.4          # the lean region: the current torus scaled about its m = 0 curve
    LEAN_REBUILD = 0.6        # rebuilt when the family's flux passes this fraction of the region's flux

    def _lean_update(self, torus):
        """With lean_source: (re)build the engine for the region ahead of the family -- the current torus scaled
        by LEAN_SCALE about its m = 0 curve, coils' node counts from their distance to it (as the refinement's
        lean engine) -- once at the start and whenever the family passes LEAN_REBUILD of that region's flux.
        The intermediate tori are stepping stones; the request's torus is refined on its own lean engine and
        verified with the reference field."""
        if self.lean_source is None or not torus.converged:
            return
        F = abs(float(torus.flux))
        if self._lean_flux is not None and F < self.LEAN_REBUILD * self._lean_flux:
            return
        from .field import FieldEngine
        big = torus.resized(torus.M, torus.N)
        big.rc = big.rc.copy(); big.zs = big.zs.copy()
        big.rc[1:] *= self.LEAN_SCALE
        big.zs[1:] *= self.LEAN_SCALE
        nfp = self.nfp
        th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ph = np.linspace(0, 2 * np.pi / nfp, 48, endpoint=False)
        PH = np.broadcast_to(ph[None, :], (96, 48))
        R, Z = big.geometry_grid(th, ph)
        X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1).reshape(-1, 3)
        Xs = [X]
        for k in range(1, int(nfp)):          # every field period (the engine's region is in real space)
            c, sn = math.cos(2 * math.pi * k / nfp), math.sin(2 * math.pi * k / nfp)
            Xs.append(np.stack([c * X[:, 0] - sn * X[:, 1], sn * X[:, 0] + c * X[:, 1], X[:, 2]], -1))
        h = float(np.max(np.linalg.norm(np.diff(X.reshape(96, 48, 3), axis=0), axis=-1)))
        h = math.hypot(h, float(np.max(np.linalg.norm(np.diff(X.reshape(96, 48, 3), axis=1), axis=-1))))
        broad = self._broad
        fkw = {}
        nm = 8 if getattr(broad, "cpu_kernels", False) else 64
        lean = None
        if _so3.on("fmeas") and getattr(self, "_in_family", False) and not getattr(self, "fmeas_off", False) \
                and not _so2.on("cudafil"):
            # (fast_newton 'fmeas') each coil's node count measured at the region points where its trapezoid error is
            # largest, then the engine certified on the whole region against the standard (_fmeas_certify); the
            # release's engine where the certificate does not hold, and restored when the family ends
            from .field.engine import family_nodes
            Xr = np.concatenate(Xs)
            try:
                rel = FieldEngine(self.lean_source, dmin=self.lean_dmin or 0.03, region_pts=Xr, region_h=h,
                                  node_multiple=nm)
            except Exception:  # noqa: BLE001
                return
            lean = rel
            try:
                Nm = [int(n) for n in rel.info.N_per_coil]
                bn = lambda x: np.linalg.norm(np.asarray(rel.evaluate(x, grad=False, A=False)["B"]), axis=1)
                Nf, _ = family_nodes(self.lean_source, Nm, Xr, eps=_so3.FMEAS_EPS, node_multiple=nm, bnorm=bn,
                                     samples=self._coil_samples())
                if sum(Nf) < sum(Nm):
                    cand = FieldEngine(self.lean_source, dmin=self.lean_dmin or 0.03, region_pts=Xr, region_h=h,
                                       node_multiple=nm, nodes_per_coil=Nf, nodes_np=True)
                    if self._fmeas_certify(rel, cand, torus):
                        lean = cand
            except Exception:  # noqa: BLE001
                lean = rel
            self._flean_region = (Xr, h)
            self._flean_rel = rel
        elif _so3.on("flean") and getattr(self, "_in_family", False):
            # (fast_newton 'flean') the stepping stones' node rule; the standard engine for this region is built
            # when the family ends (_flean_restore)
            fkw = dict(eps=_so3.FLEAN_EPS, region_margin=_so3.FLEAN_MARGIN)
        if lean is None:
            try:
                lean = FieldEngine(self.lean_source, dmin=self.lean_dmin or 0.03, region_pts=np.concatenate(Xs),
                                   region_h=h, node_multiple=nm, **fkw)
            except Exception:  # noqa: BLE001
                return
            if fkw:
                self._flean_region = (np.concatenate(Xs), h)     # the region of the engine now in use
                self._flean_rel = None
        self._lean_flux = F * self.LEAN_SCALE ** 2
        new = lean if lean.info.n_nodes < broad.info.n_nodes else broad
        if new is not self.engine:
            self.engine = new
            self._probs = {}
        if self.verbose:
            print(f"  lean engine: {lean.info.n_nodes} nodes for the region to flux {self._lean_flux:.4e}", flush=True)

    # (fast_newton 'fmeas') set by the caller for families whose outcome is set by the budget rather than their speed
    # (torux.analysis.analyse: all but quick mode's requests)
    fmeas_off = False

    def _flean_restore(self):
        """(fast_newton 'flean', 'fmeas') after the family: the standard engine -- its lean engine for the last
        region (the standard node rule), or the broad engine where that is not smaller."""
        reg = getattr(self, "_flean_region", None)
        if reg is None:
            return
        self._flean_region = None
        from .field import FieldEngine
        broad = self._broad
        lean = getattr(self, "_flean_rel", None)
        self._flean_rel = None
        try:
            if lean is None:
                lean = FieldEngine(self.lean_source, dmin=self.lean_dmin or 0.03, region_pts=reg[0], region_h=reg[1],
                                   node_multiple=8 if getattr(broad, "cpu_kernels", False) else 64)
        except Exception:  # noqa: BLE001
            lean = None
        new = lean if (lean is not None and lean.info.n_nodes < broad.info.n_nodes) else broad
        if new is not self.engine:
            self.engine = new
            self._probs = {}

    def _coil_samples(self):
        """(fast_newton 'fmeas') the lean source's coil samples, computed once per family."""
        c = getattr(self, "_coil_samples_cache", None)
        if c is None or c[0] is not self.lean_source:
            from .field.engine import coil_samples
            c = self._coil_samples_cache = (self.lean_source, coil_samples(self.lean_source))
        return c[1]

    FMEAS_SURFACE = 1.25      # 'fmeas': the certificate's surface, the torus scaled about its m = 0 curve (the family
                              # solves tori up to ~1.09x before the next rebuild, at 1.18x the flux)

    def _fmeas_certify(self, rel, cand, torus):
        """(fast_newton 'fmeas') True when the candidate engine's field differs from the standard engine's by at most
        FMEAS_CERT min |B| at every point inside S, the torus scaled FMEAS_SURFACE about its m = 0 curve.  Every
        Cartesian component of a trapezoid (or exact) Biot-Savart field is a sum of first derivatives of 1 / |x - r|,
        harmonic off the coil; so is the difference of the two engines' fields, and its largest magnitude over a
        coil-free solid torus lies on its surface (maximum principle; |dB| is subharmonic).  Checked: no coil within
        h of S's samples (h their spacing: none crosses S between samples) and none inside S (one point of each coil
        against S's cross-section at its toroidal angle; S sampled 96 x 48 per field period, every period); then the
        difference on every other sample and on 4x finer patches around each coil's nearest sample and around the 16
        largest coarse differences, where it peaks.  A check on samples, not a bound between them: the difference varies
        on the scale of the coils' distance from S (more than h), and the family's own grids are measured against a
        reference."""
        from scipy.spatial import cKDTree
        from .field.engine import min_sample_distance
        S = torus.resized(torus.M, torus.N)
        S.rc = S.rc.copy(); S.zs = S.zs.copy()
        S.rc[1:] *= self.FMEAS_SURFACE
        S.zs[1:] *= self.FMEAS_SURFACE
        nfp = int(self.nfp)
        th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ph = np.linspace(0, 2 * np.pi, 48 * nfp, endpoint=False)
        R, Z = S.geometry_grid(th, ph)
        PH = np.broadcast_to(ph[None, :], R.shape)
        X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1)
        h = math.hypot(float(np.max(np.linalg.norm(np.roll(X, -1, 0) - X, axis=-1))),
                       float(np.max(np.linalg.norm(np.roll(X, -1, 1) - X, axis=-1))))
        X = X.reshape(-1, 3)
        tree = cKDTree(X)
        samples = self._coil_samples()
        near = []
        for p, i, xs_i, dxs_i in samples:
            if min_sample_distance(tree, xs_i, float(np.max(np.linalg.norm(dxs_i, axis=1)))) <= h:
                self._fmeas_last = dict(refused="a coil within h of the certificate surface")
                return False
            dd, jj = tree.query(xs_i[::16], k=1)
            near.append(int(np.asarray(jj)[int(np.argmin(dd))]))
        # the difference on a 2x coarser grid of S (every other sample), on 4x finer patches (9 x 9, spanning the
        # neighbouring coarse samples) around each coil's nearest sample, and then around the 16 coarse samples with the
        # largest difference: it peaks next to the coils' closest approaches and varies on the scale of their distance
        off = np.linspace(-2.0, 2.0, 9)
        dth, dph = th[1] - th[0], ph[1] - ph[0]
        nth, nph = len(th), len(ph)

        def patch(j):
            a, b_ = np.unravel_index(j, (nth, nph))
            tp, pp = th[a] + dth * off, ph[b_] + dph * off
            Rp_, Zp_ = S.geometry_grid(tp, pp)
            PP = np.broadcast_to(pp[None, :], Rp_.shape)
            return np.stack([Rp_ * np.cos(PP), Rp_ * np.sin(PP), Zp_], -1).reshape(-1, 3)

        def diff(P):
            Br = np.asarray(rel.evaluate(P, grad=False, A=False)["B"])
            dB = np.asarray(cand.evaluate(P, grad=False, A=False)["B"]) - Br
            return np.linalg.norm(dB, axis=1), np.linalg.norm(Br, axis=1)
        coarse = np.arange(nth * nph).reshape(nth, nph)[::2, ::2].ravel()
        near = sorted(set(near))
        d1, b1 = diff(np.concatenate([X[coarse]] + [patch(j) for j in near]))
        dc = d1[:len(coarse)]
        top = [int(coarse[k]) for k in np.argsort(dc)[::-1][:16] if int(coarse[k]) not in near]
        d2 = diff(np.concatenate([patch(j) for j in top]))[0] if top else np.zeros(0)
        d = float(max(np.max(d1), np.max(d2) if len(d2) else 0.0))
        bmin = float(np.min(b1))
        self._fmeas_last = dict(diff=d, bmin=bmin, nodes=(int(cand.info.n_nodes), int(rel.info.n_nodes)))
        return bool(np.isfinite(d) and d <= _so3.FMEAS_CERT * bmin)

    def predict(self, tori, s_new, sgn):
        last = tori[-1]
        if len(tori) >= 3 and _so2.on("cquad"):
            # (fast_family 'cquad') quadratic extrapolation in s through the last three tori (Lagrange): on recorded
            # families its coefficient error is 0.02-0.3 of the linear predictor's away from resonances, ~1 next to
            # the stall
            t0, t1 = tori[-3].resized(last.M, last.N), tori[-2].resized(last.M, last.N)
            s0, s1, s2 = (math.sqrt(abs(t.flux)) for t in (tori[-3], tori[-2], last))
            if s0 < s1 < s2:
                L0 = (s_new - s1) * (s_new - s2) / ((s0 - s1) * (s0 - s2))
                L1 = (s_new - s0) * (s_new - s2) / ((s1 - s0) * (s1 - s2))
                L2 = (s_new - s0) * (s_new - s1) / ((s2 - s0) * (s2 - s1))
                return TorusState(L0 * t0.rc + L1 * t1.rc + L2 * last.rc, L0 * t0.zs + L1 * t1.zs + L2 * last.zs,
                                  L0 * tori[-3].iota + L1 * tori[-2].iota + L2 * last.iota, sgn * s_new ** 2, last.M,
                                  last.N, self.nfp, phi0=self.phi0)
        if len(tori) >= 2:
            prev = tori[-2]
            sp, sl = math.sqrt(abs(prev.flux)), math.sqrt(abs(last.flux))
            a = (s_new - sl) / (sl - sp)
            p0 = prev.resized(last.M, last.N)
            return TorusState(last.rc + a * (last.rc - p0.rc), last.zs + a * (last.zs - p0.zs),
                              last.iota + a * (last.iota - prev.iota), sgn * s_new ** 2, last.M, last.N, self.nfp,
                              phi0=self.phi0)
        s = math.sqrt(abs(last.flux))
        fac = s_new / s
        g = TorusState(last.rc.copy(), last.zs.copy(), last.iota, sgn * s_new ** 2, last.M, last.N, self.nfp,
                       phi0=self.phi0)
        g.rc[1:] *= fac
        g.zs[1:] *= fac
        return g


FLOOR_C = 10.0       # fast_design 'floor': the flux equation at the floor when its relative error <= FLOOR_C x tol
RSTOP_BN = 1e-14     # fast_family 'rstop': the off-grid B.n at which one non-improving raise ends the refinement
                     # (torux.analysis.BN_SURFACE, the acceptance bar)


FULL_CAPS = (640, 768)


def refine_to_limit(cont: Continuation, sol: TorusState, flux, target=3e-15, max_levels=20, deadline=None,
                    verbose=False, fixed_iota=None, Mcap=320, Ncap=320):
    """Grow the resolution until the off-grid |B.n|/|B| reaches ``target`` or stops improving
    (two levels without a 2x gain), returning the best torus; ``info['refine_hist']`` keeps
    (M, N, bn) per level; a level the deadline cannot afford is skipped (the time per grid point of the last
    level predicts it).  The full (certificate) path raises the caps to FULL_CAPS: QUASR 0019493, 1072451 and
    2295266 stop at 5e-12..2e-9 at M = 320, and 2039780 needs N = 768; quick mode keeps 320."""
    hist = []
    best = sol
    caps_save = (cont.Mmax, cont.Nmax)
    cont.Mmax, cont.Nmax = max(cont.Mmax, Mcap), max(cont.Nmax, Ncap)
    try:
        return _refine(cont, sol, flux, target, max_levels, deadline, verbose, fixed_iota, hist, best)
    finally:
        cont.Mmax, cont.Nmax = caps_save


def predict_resolution(t: TorusState, target, cont, floor=1e-17, maxfac=2.0):
    """(M, N) at which the torus's spectrum, extrapolated geometrically from its last half, falls
    below ``floor`` relative to the shape: one jump to the needed resolution instead of a ladder of
    1.25x steps (each step a Newton solve).  At most maxfac times the current values."""
    a = np.sqrt(t.rc ** 2 + t.zs ** 2)
    scale = max(np.max(a[1:]), 1e-300)
    rm = np.max(a, axis=1) / scale
    cn = np.max(a, axis=0) / scale
    nn = np.array([max(cn[t.N + n], cn[t.N - n]) for n in range(t.N + 1)])

    def need(v, cur):
        k = np.arange(len(v))
        lo = max(1, len(v) // 2)
        vv = np.maximum(v[lo:], 1e-300)
        good = vv > 1e-300
        if good.sum() < 3:
            return cur
        slope, icpt = np.polyfit(k[lo:][good], np.log(vv[good]), 1)
        if slope >= -1e-3:
            return int(maxfac * cur)
        kneed = (np.log(floor) - icpt) / slope
        return int(np.clip(np.ceil(kneed) + 2, cur, maxfac * cur))
    Mn = need(rm, t.M)
    Nn = need(nn, t.N) if t.N > 2 else t.N
    M2 = next((L for L in LADDER if L >= Mn), LADDER[-1])
    N2 = next((L for L in LADDER if L >= Nn), LADDER[-1])
    return min(M2, cont.Mmax), min(N2, cont.Nmax)


def _refine(cont, sol, flux, target, max_levels, deadline, verbose, fixed_iota, hist, best):
    deadline = _budget.earliest(deadline)
    if not np.isfinite(sol.bn_off):
        prob = cont.problem(sol.M, sol.N)
        sol.res_off, sol.bn_off = prob.check(sol.rc, sol.zs, sol.iota)
    hist.append((sol.M, sol.N, sol.bn_off))
    hist_t = []
    stale = 0
    cur = sol
    stop = "max_levels"
    for lev in range(max_levels):
        if best.bn_off <= target:
            stop = "target"
            break
        if deadline is not None and time.time() > deadline:
            stop = "deadline"
            break
        M2, N2 = predict_resolution(cur, target, cont) if lev < 2 else cont.grow(cur)
        if (M2, N2) == (cur.M, cur.N):
            M2, N2 = cont.grow(cur)
        if (M2, N2) == (cur.M, cur.N):
            stop = "cap"
            break
        # cost control: the next level's time predicted from the last level's time per grid point
        # (the work of a Newton iteration is ~ grid points x source nodes); skip what cannot finish
        if deadline is not None and hist_t:
            (Mp, Np, tp) = hist_t[-1]
            gp = (2.5 * Mp + 4) * (2.5 * Np + 4)
            gn = (2.5 * M2 + 4) * (2.5 * N2 + 4)
            pred = tp * gn / max(gp, 1.0)
            if time.time() + pred > deadline:
                best.info["refine_stopped"] = f"next level {M2},{N2} predicted {pred:.0f}s beyond the deadline"
                stop = "deadline"
                break
        # Newton to well below the target (a raised resolution adds modes at zero; a loose tolerance
        # would accept the padded torus without solving for them)
        tol_save = cont.tol
        cont.tol = min(cont.tol, target / 30)
        t_lev = time.time()
        # (with no switch on, the call is exactly the old one: other Continuation-like classes pass through here)
        kw = {"floor_stop": FLOOR_C} if _so.on("floor") else {}
        try:
            new = cont.solve(cur, flux, M2, N2, maxit=12, fixed_iota=fixed_iota, filtered_ok=True, **kw)
            if new.converged and new.info.get("predicted") is not None and not (new.bn_off <= target):
                # (fast_design 'predict') the unevaluated final iterate is short of the target off-grid: the level
                # once more with evaluated iterations before it counts (a prediction never decides acceptance)
                again = cont.solve(new, flux, M2, N2, maxit=12, fixed_iota=fixed_iota, filtered_ok=True,
                                   predict=False, **kw)
                again.info["after_prediction"] = True
                if again.converged and again.bn_off < new.bn_off:
                    new = again
        finally:
            cont.tol = tol_save
        hist_t.append((M2, N2, time.time() - t_lev))
        hist.append((M2, N2, new.bn_off if new.converged else np.nan))
        if verbose:
            print(f"   refine M,N {M2},{N2} conv {new.converged} bn {new.bn_off:.3e}", flush=True)
        if not new.converged:
            stop = "not_converged"
            break
        stale = 0 if new.bn_off < 0.5 * best.bn_off else stale + 1
        if new.bn_off < best.bn_off:
            best = new
        cur = new
        # a plateau far above the rounding floor is a resonant spectral bump the resolution has not
        # passed yet (QUASR 1041330: B.n 1.6e-6 from M = 27 to 58, 6e-12 at 170): keep growing;
        # a plateau at the floor ends the refinement
        if stale >= 2 and best.bn_off <= 1e-13:
            stop = "plateau"            # two raises without a 2x gain at the rounding floor
            break
        if stale >= 1 and best.bn_off <= RSTOP_BN and _so2.on("rstop"):
            # (fast_family 'rstop') at the acceptance bar a raise that gains less than 2x ends the refinement: the
            # levels after it moved B.n by a few per cent at the floor (CoilStellaration: 3.6e-15 -> 3.5e-15 ->
            # 3.4e-15 at M, N = 80..128, 96..128, 128..160) and were 60 % of the axis path's work on surfaces
            stop = "plateau1"
            break
        if stale >= 6:
            stop = "stale"
            break
    else:
        stop = "max_levels" if best.bn_off > target else "target"
    if stop == "max_levels" and best.bn_off <= target:
        stop = "target"
    best.info["refine_hist"] = hist
    best.info["refine_stop"] = stop
    return best


def _inside_curve(R, Z, R0, Z0):
    """(R0, Z0) inside the closed polygon (R, Z) (winding number, nonzero rule)."""
    a = np.arctan2(Z - Z0, R - R0)
    da = np.diff(np.concatenate([a, a[:1]]))
    da = (da + np.pi) % (2 * np.pi) - np.pi
    return abs(float(np.sum(da))) > np.pi


def line_torus_start(line, M, N, eps, iota=0.0):
    """A closed line (torux.orbits.ClosedLine, symmetric about phi0) plus a circle of radius eps: the
    starting guess for the tori around it (island tori around an O line; the axis is q = 1)."""
    nfp_eff = line.nfp / line.q
    st = TorusState(np.zeros((M + 1, 2 * N + 1)), np.zeros((M + 1, 2 * N + 1)), iota, 0.0, M, N, nfp_eff,
                    phi0=line.phi0)
    na = min(N, line.K)
    st.rc[0, N:N + na + 1] = line.a[:na + 1]
    st.zs[0, N + 1:N + na + 1] = -line.b[1:na + 1]
    st.rc[1, N] = eps
    st.zs[1, N] = eps
    return st
