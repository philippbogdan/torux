"""Flux-surface families, refinement to the rounding floor and the one-call analysis for sources
without stellarator symmetry (general basis: torux.torus_general, torux.axis_general).

Mirrors torux.family (Continuation.run / raise_until / refine_to_limit) and torux.analysis.analyse at
a basic level: the axis (general closed-line Newton), the near-axis torus (axis + small circle),
continuation in s = sqrt(|flux|) with linear predictors (the predictor is also the phase
reference of each solve, so consecutive tori share their theta origin), resolution raised by the
frequency-weighted spectral tails, then at the request a lean engine (per-coil nodes from the
torus's own distance) and refinement by predicted resolution jumps until the off-grid
|B.n|/|B| reaches the target or plateaus.  Acceptance as torux.analysis.analyse: 'surface' only for
bn_off <= torux.analysis.BN_SURFACE (the arithmetic limit), quadrature estimate <= 1e-8 and the resonance
guard (general basis).

Not (yet) on the general path: chains / jumps over resonances and the no-surface certificates
(the family stops at a resonance it cannot cross -> 'undetermined').
"""
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .family import LADDER, ladder_up
from .torus_general import GeneralTorus, GeneralTorusProblem, GeneralKrylov, resonance_guard
from .axis_general import GeneralAxis, find_axes_general
from . import budget as _budget


@dataclass
class FamilyResult:
    tori: list = field(default_factory=list)
    stop_reason: str = ""
    stop_flux: float = np.nan
    log: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def axis_torus_start(axis: GeneralAxis, M, N, eps, iota):
    """Axis plus a circle of radius eps (counter-clockwise in (R, Z)): the start near the axis.
    The axis R = sum rc cos(n psi) + rs sin(n psi) is the m = 0 row: cos(0 - n psi) = cos(n psi),
    sin(0 - n psi) = -sin(n psi)."""
    z = lambda: np.zeros((M + 1, 2 * N + 1))
    st = GeneralTorus(z(), z(), z(), z(), iota, 0.0, M, N, axis.nfp, phi0=axis.phi0)
    na = min(N, axis.N)
    st.rc[0, N:N + na + 1] = axis.rc[:na + 1]
    st.rs[0, N + 1:N + na + 1] = -axis.rs[1:na + 1]
    st.zc[0, N:N + na + 1] = axis.zc[:na + 1]
    st.zs[0, N + 1:N + na + 1] = -axis.zs[1:na + 1]
    st.rc[1, N] += eps
    st.zs[1, N] += eps
    return st


class GeneralContinuation:
    DENSE_MAX = 6000          # unknowns up to which the dense Newton is used
    DENSE_FALLBACK = 9000     # unknowns up to which the dense Newton backs up a failed Newton-Krylov
    FILTER_D = 1e-4
    FILTER = 1e-5
    FILTER_MAX = 3000

    def __init__(self, engine, nfp, M0=6, N0=None, bn_cont=1e-9, bn_accept=1e-5, tol=1e-12, Mmax=128, Nmax=128,
                 verbose=False, phi0=0.0):
        self.engine = engine
        self.nfp = nfp
        self.phi0 = phi0
        self.M0, self.N0 = M0, N0
        self.bn_cont, self.bn_accept, self.tol = bn_cont, bn_accept, tol
        self.Mmax, self.Nmax = Mmax, Nmax
        self.verbose = verbose
        self._probs = {}

    def problem(self, M, N):
        key = (M, N)
        if key not in self._probs:
            self._probs[key] = GeneralTorusProblem(self.engine, self.nfp, M, N, phi0=self.phi0)
        return self._probs[key]

    def krylov(self, M, N):
        key = ("nk", M, N)
        if key not in self._probs:
            self._probs[key] = GeneralKrylov(self.problem(M, N))
        return self._probs[key]

    def near_resonant(self, iota, M):
        io = abs(float(iota))
        if not np.isfinite(io):
            return False
        for m in range(1, 2 * M + 11):
            n0 = round(io * m / self.nfp)
            if abs(n0 * self.nfp - io * m) < self.FILTER_D:
                return True
        return False

    def solve(self, guess: GeneralTorus, flux, M, N, maxit=15, fixed_iota=None, filtered_ok=False, ref=None):
        prob = self.problem(M, N)
        n = prob.basis.size
        if filtered_ok and n <= self.FILTER_MAX and self.near_resonant(guess.iota if fixed_iota is None else fixed_iota, M):
            sol = prob.newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota, filt=self.FILTER, ref=ref)
            sol.info["filtered"] = True
        elif n <= self.DENSE_MAX:
            sol = prob.newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota, ref=ref)
        else:
            sol = self.krylov(M, N).newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota,
                                           gtol=1e-8, restart=60, maxiter=3, ref=ref)
            if not sol.converged and n <= self.DENSE_FALLBACK:
                sol = prob.newton(guess, flux, tol=self.tol, maxit=maxit, fixed_iota=fixed_iota, ref=ref)
        if sol.converged:
            sol.res_off, sol.bn_off = prob.check(sol)
        return sol

    def grow(self, sol: GeneralTorus):
        tm, tn = sol.tails()
        wm = tm * max(abs(sol.iota), 1e-3) * sol.M
        wn = tn * self.nfp * sol.N
        M2 = ladder_up(sol.M, cap=self.Mmax) if wm >= 0.1 * wn else sol.M
        N2 = ladder_up(sol.N, cap=self.Nmax) if wn >= 0.1 * wm else sol.N
        return M2, N2

    def raise_until(self, sol: GeneralTorus, flux, bn_target, max_raise=4, fixed_iota=None):
        for _ in range(max_raise):
            if sol.bn_off <= bn_target:
                break
            M2, N2 = self.grow(sol)
            if (M2, N2) == (sol.M, sol.N):
                break
            s2 = self.solve(sol, flux, M2, N2, fixed_iota=fixed_iota)
            if not s2.converged or not (s2.bn_off < sol.bn_off):
                break
            sol = s2
        return sol

    def start(self, axis: GeneralAxis):
        N = self.N0 or max(8, min(axis.N, 64))
        M = self.M0
        eps = 0.004 * axis.rc[0]
        st = axis_torus_start(axis, M, N, eps, axis.iota if np.isfinite(axis.iota) else 0.0)
        f0 = self.problem(M, N).flux_of(st)
        st.flux = f0
        sol = self.solve(st, f0, M, N, maxit=25)
        if sol.converged:
            sol = self.raise_until(sol, f0, self.bn_cont)
        return sol, f0

    def run(self, axis, flux_target, ds0=0.05, ds_max=0.2, ds_min=2e-3, deadline=None, start=None):
        # the family's deadline also holds inside its Newton solves (a budget scope; see torux.family.Continuation.run)
        with _budget.scope(deadline):
            return self._run(axis, flux_target, ds0, ds_max, ds_min, _budget.earliest(deadline), start)

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
        sgn = math.copysign(1.0, f0)
        s_target = math.sqrt(abs(flux_target))
        s = math.sqrt(abs(f0))
        ds = ds0 * s_target
        while s < s_target * (1 - 1e-12):
            if deadline is not None and time.time() > deadline:
                out.stop_reason = "deadline"
                break
            s_new = min(s + ds, s_target)
            guess = self.predict(out.tori, s_new, sgn)
            last = out.tori[-1]
            sol = self.solve(guess, sgn * s_new ** 2, last.M, last.N, maxit=10)
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
                                its=sol.its, res=sol.res, lam=sol.lam, t=time.time() - t0))
            if self.verbose:
                print(f"  gen s {s_new / s_target:.5f} ds {ds / s_target:.2e} ok {ok} iota {sol.iota:.10f} "
                      f"bn {sol.bn_off:.2e} M,N {sol.M},{sol.N} its {sol.its} asym {sol.asymmetry():.1e} "
                      f"t {time.time() - t0:.1f}s", flush=True)
            if ok:
                out.tori.append(sol)
                s = s_new
                ds = min(ds * 1.5, ds_max * s_target)
            else:
                ds *= 0.5
                if ds < ds_min * s_target:
                    out.stop_reason = "stalled"
                    out.stop_flux = sgn * s ** 2
                    break
        else:
            out.stop_reason = "reached"
        out.timings["total"] = time.time() - t0
        return out

    def predict(self, tori, s_new, sgn):
        last = tori[-1]
        if len(tori) >= 2:
            prev = tori[-2]
            sp, sl = math.sqrt(abs(prev.flux)), math.sqrt(abs(last.flux))
            a = (s_new - sl) / (sl - sp)
            p0 = prev.resized(last.M, last.N)
            arr = [x + a * (x - y) for x, y in zip(last.arrays(), p0.arrays())]
            return GeneralTorus(*arr, last.iota + a * (last.iota - prev.iota), sgn * s_new ** 2, last.M, last.N,
                                self.nfp, phi0=self.phi0)
        s = math.sqrt(abs(last.flux))
        fac = s_new / s
        g = last.copy()
        g.flux = sgn * s_new ** 2
        for a in g.arrays():
            a[1:] *= fac
        return g


# ============================================================================ refinement
def predict_resolution(t: GeneralTorus, cont, floor=1e-17, maxfac=2.0):
    """(M, N) at which the spectrum, extrapolated geometrically from its last half, falls below
    ``floor`` relative to the shape (as torux.family.predict_resolution, all four coefficient sets)."""
    a = t.amplitude()
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


def refine_to_limit(cont: GeneralContinuation, sol: GeneralTorus, flux, target=3e-15, max_levels=20, deadline=None,
                    verbose=False, fixed_iota=None, Mcap=320, Ncap=320):
    """Grow the resolution until the off-grid |B.n|/|B| reaches ``target`` or plateaus; the best torus
    with info['refine_hist'] = [(M, N, bn)], info['refine_stop']."""
    caps = (cont.Mmax, cont.Nmax)
    cont.Mmax, cont.Nmax = max(cont.Mmax, Mcap), max(cont.Nmax, Ncap)
    try:
        if not np.isfinite(sol.bn_off):
            sol.res_off, sol.bn_off = cont.problem(sol.M, sol.N).check(sol)
        hist = [(sol.M, sol.N, sol.bn_off)]
        hist_t = []
        best, cur, stale, stop = sol, sol, 0, "max_levels"
        for lev in range(max_levels):
            if best.bn_off <= target:
                stop = "target"
                break
            if deadline is not None and time.time() > deadline:
                stop = "deadline"
                break
            M2, N2 = predict_resolution(cur, cont) if lev < 2 else cont.grow(cur)
            if (M2, N2) == (cur.M, cur.N):
                M2, N2 = cont.grow(cur)
            if (M2, N2) == (cur.M, cur.N):
                stop = "cap"
                break
            if deadline is not None and hist_t:
                (Mp, Np, tp) = hist_t[-1]
                gp = (2.5 * Mp + 4) * (2.5 * Np + 4)
                gn = (2.5 * M2 + 4) * (2.5 * N2 + 4)
                pred = tp * gn / max(gp, 1.0)
                if time.time() + pred > deadline:
                    best.info["refine_stopped"] = f"next level {M2},{N2} predicted {pred:.0f}s beyond the deadline"
                    stop = "deadline"
                    break
            tol_save = cont.tol
            cont.tol = min(cont.tol, target / 30)
            t_lev = time.time()
            try:
                new = cont.solve(cur, flux, M2, N2, maxit=12, fixed_iota=fixed_iota, filtered_ok=True)
            finally:
                cont.tol = tol_save
            hist_t.append((M2, N2, time.time() - t_lev))
            hist.append((M2, N2, new.bn_off if new.converged else np.nan))
            if verbose:
                print(f"   gen refine M,N {M2},{N2} conv {new.converged} bn {new.bn_off:.3e} lam {new.lam:.1e} "
                      f"t {time.time() - t_lev:.1f}s", flush=True)
            if not new.converged:
                stop = "not_converged"
                break
            stale = 0 if new.bn_off < 0.5 * best.bn_off else stale + 1
            prev_tail = max(cur.tails())
            if new.bn_off < best.bn_off:
                best = new
            cur = new
            if stale >= 1 and best.bn_off <= 1e-13 and prev_tail <= 2e-15:
                # the raised torus's spectrum had already reached the coefficients' rounding noise
                # (HSX: m-spectrum flat at ~1e-15 from m ~ 28) and the raise brought no 2x gain:
                # what B.n is left is the field's rounding, not truncation
                stop = "floor"
                break
            if stale >= 2 and best.bn_off <= 1e-13:
                stop = "plateau"
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
    finally:
        cont.Mmax, cont.Nmax = caps


# ============================================================================ one call
def torus_record(t: GeneralTorus, full=False):
    d = dict(basis="general", flux=float(t.flux), iota=float(t.iota), M=int(t.M), N=int(t.N), bn_off=float(t.bn_off),
             res_off=float(t.res_off), res=float(t.res), its=int(t.its), converged=bool(t.converged),
             lam=float(t.lam), asym=t.asymmetry(), phi0=float(t.phi0))
    if full:
        for k in ("rc", "rs", "zc", "zs"):
            d[k] = np.asarray(getattr(t, k)).tolist()
    return d


def torus_from_record(rec, nfp):
    return GeneralTorus(*(np.array(rec[k], float) for k in ("rc", "rs", "zc", "zs")), float(rec["iota"]),
                        float(rec["flux"]), int(rec["M"]), int(rec["N"]), int(nfp), bn_off=float(rec.get("bn_off", np.nan)),
                        phi0=float(rec.get("phi0", 0.0)))


def lean_continuation(cont: GeneralContinuation, source, torus: GeneralTorus, dmin, ntheta=96, nphi_fp=48):
    """A continuation on an engine whose per-coil node counts come from each coil's distance to
    ``torus`` (as torux.analysis.lean_continuation); cont itself when that engine is not smaller."""
    from .field import FieldEngine
    nfp = int(round(torus.nfp)) if float(torus.nfp).is_integer() else 1
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, nphi_fp * max(1, nfp), endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    R, Z = torus.geometry_grid(th, ph)
    X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1)
    hth = np.max(np.linalg.norm(np.roll(X, -1, 0) - X, axis=-1))
    hph = np.max(np.linalg.norm(np.roll(X, -1, 1) - X, axis=-1))
    old = cont.engine
    lean = FieldEngine(source, dmin=dmin, region_pts=X.reshape(-1, 3), region_h=math.hypot(hth, hph),
                       node_multiple=8 if getattr(old, "cpu_kernels", False) else 64)
    if lean.info.n_nodes >= old.info.n_nodes:
        return cont
    return GeneralContinuation(lean, cont.nfp, M0=cont.M0, N0=cont.N0, bn_cont=cont.bn_cont, bn_accept=cont.bn_accept,
                               tol=cont.tol, Mmax=cont.Mmax, Nmax=cont.Nmax, verbose=cont.verbose, phi0=cont.phi0)


def _shear(tori, flux):
    fl = np.array([t.flux for t in tori])
    io = np.array([t.iota for t in tori])
    if len(tori) < 2:
        return np.nan
    j = int(np.argmin(np.abs(fl - flux)))
    lo, hi = max(0, j - 1), min(len(tori) - 1, j + 1)
    return (io[hi] - io[lo]) / (fl[hi] - fl[lo]) if hi > lo else np.nan


def _accept(cont, source, sol, tori, dmin, tol, deadline, verbose, lean, out, t0):
    rc_ = lean_continuation(cont, source, sol, dmin) if lean else cont
    out["engine"]["refine_nodes"] = rc_.engine.info.n_nodes
    best = refine_to_limit(rc_, sol, sol.flux, target=tol, deadline=deadline, verbose=verbose)
    out["timings"]["refine"] = time.time() - t0
    shear_phi = _shear(tori, best.flux)
    g = resonance_guard(rc_.engine, best, shear=shear_phi * 2 * np.pi)
    rec = torus_record(best, full=True)
    rec["quad_err_half"] = rc_.problem(best.M, best.N).quadrature_check(best)
    guard = dict(ok=g["ok"], h=g["h"], x_over_w=g["x_over_w"], worst=g["worst"], n_checked=len(g["checked"]))
    from .analysis import BN_SURFACE
    ok = bool(best.bn_off <= BN_SURFACE and g["ok"] and rec["quad_err_half"] <= 1e-8)
    return best, rec, guard, ok


@_budget.budgeted
def analyse_general(source, flux=None, dmin=None, tol=3e-15, verbose=False, Mmax=128, Nmax=128, budget_s=None,
                    axis_guesses=(), region_pts=None, lean=True, scan=False, M0=6, N0=None):
    """One source without stellarator symmetry, one call: axis, family, the surface at the request
    refined to the arithmetic limit (or, without a request, the outermost torus the family reaches).

    Returns a JSON-serialisable dict with the conventions of torux.analysis.analyse (status: 'surface',
    'surface_unverified', 'undetermined', 'family', 'no_axis', 'error'); tori carry rc, rs, zc, zs."""
    from .field import FieldEngine
    t0 = time.time()
    deadline = t0 + budget_s if budget_s else None
    out = dict(id=source.id, nfp=source.nfp, request_flux=flux, status="error", timings={}, path="general",
               chains=[], jumps=[])
    try:
        if dmin is None:
            dmin = source.meta.get("dmin", 0.05)
        eng = FieldEngine(source, dmin=dmin, region_pts=region_pts)
        out["engine"] = dict(nodes=eng.info.n_nodes, dmin=dmin, per_coil=region_pts is not None)
        axes = find_axes_general(eng, source, guesses=axis_guesses, scan=scan, verbose=verbose)
        out["axes"] = [a.record() for a in axes]
        ell = [a for a in axes if 0 < a.residue < 1]
        out["timings"]["axis"] = time.time() - t0
        if not ell:
            out["status"] = "no_axis"
            return out
        pts = source.sample_points()
        Rm = float(np.median(np.hypot(pts[:, 0], pts[:, 1]))) if len(pts) else 1.0
        if len(axis_guesses):
            g = axis_guesses[0]
            if hasattr(g, "rc"):
                Rm = float(g.rc[0])
            elif np.ndim(g[0]) == 1:
                Rm = float(g[0][0])
            elif not callable(g[0]):
                Rm = float(g[0])
        ax = min(ell, key=lambda a: abs(a.rc[0] - Rm))
        cont = GeneralContinuation(eng, source.nfp, M0=M0, N0=N0 or max(16, min(ax.N, 64)), verbose=verbose,
                                   Mmax=Mmax, Nmax=Nmax)
        target = flux if flux is not None else np.inf
        if flux is None:
            R_ax, Z_ax = ax.at(np.array([0.0]))
            pa = np.array([R_ax[0], 0.0, Z_ax[0]])
            Bax = float(np.linalg.norm(np.asarray(eng.evaluate(pa[None, :], grad=False, A=False)["B"])[0]))
            d_coil = float(np.min(np.linalg.norm(pts - pa[None, :], axis=1))) if len(pts) else 0.3 * R_ax[0]
            target = 4.0 * Bax * np.pi * d_coil ** 2
            out["range_target"] = target
        fam = cont.run(ax, abs(target), deadline=deadline)
        tori = list(fam.tori)
        out["profile"] = [torus_record(t) for t in tori]
        out["family_stop"] = fam.stop_reason
        out["family_stop_flux"] = float(fam.stop_flux) if np.isfinite(fam.stop_flux) else None
        out["timings"]["family"] = time.time() - t0
        reached = [t for t in tori if np.isfinite(target) and abs(abs(t.flux) - abs(target)) <= 1e-12 * abs(target)]
        if flux is not None and reached:
            best, rec, guard, ok = _accept(cont, source, reached[-1], tori, dmin, tol, deadline, verbose, lean, out, t0)
            out["surface"] = rec
            out["guard"] = guard
            out["refine_hist"] = best.info.get("refine_hist")
            out["refine_stop"] = best.info.get("refine_stop")
            out["status"] = "surface" if ok else "surface_unverified"
        elif flux is not None:
            # the family stopped before the request (a resonance it cannot cross, the edge or the
            # deadline); the general path has no chain jumps or no-surface certificates yet
            out["status"] = "undetermined"
            out["reason"] = f"family stopped ({fam.stop_reason}) at flux {fam.stop_flux}"
        else:
            out["status"] = "family"
            cands = [t for t in tori if t.converged and np.isfinite(t.bn_off) and t.bn_off < 1e-6]
            if cands:
                outer = max(cands, key=lambda t: abs(t.flux))
                best, rec, guard, ok = _accept(cont, source, outer, tori, dmin, tol, deadline, verbose, lean, out, t0)
                out["range"] = dict(verified_lower_bound=abs(best.flux) if ok else None, outer_torus=rec, guard=guard,
                                    verified=ok)
                out["surface"] = rec
        out["timings"]["total"] = time.time() - t0
        return out
    except Exception as e:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
        out["timings"]["total"] = time.time() - t0
        return out
