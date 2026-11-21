"""The solver as a tool: one call per request, a stated time budget, a truthful outcome.

    import torux
    res = torux.solve("quasr_0000954.json")                       # the request the file carries
    res = torux.solve("coils.json", flux=0.0691, gradients=True)  # the full toroidal flux in Wb
    res = torux.solve(source)                                     # no flux: the range the source holds

    res.status        'surface' | 'no_surface' | 'range' | 'undetermined' | 'no_axis' | 'timeout' | 'error', or
                      'surface_unverified' / 'range_unverified': a torus the solver found but the requested
                      independent verification did not confirm (or the range's outer torus the solver's own
                      acceptance refused).  'surface' and 'range' are claims only when both agree.
    res.budget_s      the budget stated before solving (from the source's size when not given)
    res.surface       the torus at the request (R cosine / Z sine series in straight-field-line angles)
    res.iota          its rotational transform
    res.evidence      solver checks (off-grid B.n, refinement history, resonance guard, quadrature error),
                      independent verification (reference B.n, flux, double-double tier; reference-traced lines
                      with trace_lines=8), certificates
    res.gradients     d iota / d (every source parameter, each alone: coil coefficients / points / currents, dipole
                      positions / moments, ...; and the flux) by the adjoint method
    res.field(x)      B at Cartesian points (the solver's engine)
    res.points(theta, phi)   Cartesian points of the surface
    res.save(path)    everything above as JSON

The source is a file (torux.load reads it) or a torux.Source.  A source that comes with design surfaces (a QUASR
download) starts from them; any other starts from its magnetic axis.  Without a flux, and with none in the
file, the result is the range the source holds: the family of surfaces from the axis to where it stops, the
outermost torus verified.  The call returns within its stated budget up to the last single step (a Newton
solve or a tracing stage).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Request:
    source: Any                          # a file, or a torux.Source
    flux: Optional[float] = None         # requested full toroidal flux (Wb), signed or not
    budget_s: Optional[float] = None     # None: the solver states one (estimate_budget)
    tol: float = 3e-15                   # off-grid |B.n|/|B| target (the rounding floor)
    verify: bool = True                  # independent check: reference B.n, flux on sections, double-double tier
    trace_lines: int = 0                 # also trace this many lines with the reference tracer (100 periods; minutes)
    gradients: bool = False


@dataclass
class Result:
    status: str
    budget_s: float
    wall_s: float
    surface: Any = None
    iota: float = np.nan
    evidence: dict = field(default_factory=dict)
    gradients: dict = None
    record: dict = None
    _engine: Any = None
    _family: Any = None            # the family's tori and axis behind a verified surface (torux.crossed), not saved

    def field(self, x):
        """B (T) at Cartesian points x (..., 3) in metres."""
        return np.asarray(self._engine.evaluate(np.asarray(x, float), grad=False, A=False)["B"])

    def points(self, theta, phi):
        """Cartesian points of the surface at its poloidal angle theta and the cylindrical angle phi (rad)."""
        if self.surface is None:
            raise ValueError(f"no surface: the status is {self.status!r}")
        if hasattr(self.surface, "rs"):
            from .torus_general import torus_points
        else:
            from .verify import torus_points
        return np.asarray(torus_points(self.surface, theta, phi)[0])

    def to_dict(self):
        """The result as plain Python (JSON-serialisable): status, budget, time, the torus, the evidence."""
        out = dict(status=self.status, budget_s=self.budget_s, wall_s=self.wall_s,
                   iota=None if not np.isfinite(self.iota) else float(self.iota), surface=None,
                   evidence=_plain(self.evidence), gradients=_plain(self.gradients))
        s = self.surface
        if s is not None:
            out["surface"] = dict(nfp=int(s.nfp), M=int(s.M), N=int(s.N), iota=float(s.iota), flux_Wb=float(s.flux),
                                  phi0=float(getattr(s, "phi0", 0.0)),
                                  convention="R = sum rc[m, n + N] cos(m theta - n nfp (phi - phi0)) (+ rs sin), "
                                             "Z = sum zs[m, n + N] sin(m theta - n nfp (phi - phi0)) (+ zc cos); "
                                             "theta the straight-field-line angle, phi the cylindrical angle",
                                  **{k: np.asarray(getattr(s, k)).tolist() for k in ("rc", "rs", "zc", "zs")
                                     if hasattr(s, k)})
        return out

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return repr(o)


def _resolve_source(src):
    """The Source of a request and its design surfaces (or None): a file is read by torux.load."""
    if isinstance(src, (str, os.PathLike)):
        from .load import load
        src = load(src)
    return src, getattr(src, "_design_surfaces", None)


def estimate_budget(source, flux=None):
    """A stated budget (s) from the source's size: the field-evaluation work of the refinement at a typical
    final resolution times a safety factor; a minimum of 300 s and, without a request (the range), twice that."""
    n_el = 0
    for p in source.parts:
        n_el += getattr(p, "n", 1) * (2 * getattr(p, "order", 0) + 1 if p.kind == "fourier" else 1)
    base = 300.0 + 0.05 * n_el
    return float(base if flux is not None else 2 * base)


def solve(source, flux=None, *, budget_s=None, tol=3e-15, verify=True, trace_lines=0, gradients=False) -> Result:
    """One request: the flux surface of ``source`` that encloses ``flux``, verified, or the certificate that
    none exists, or a truthful 'undetermined'.  ``source`` may also be a Request."""
    from . import budget as B
    req = source if isinstance(source, Request) else Request(source, flux, budget_s, tol, verify, trace_lines,
                                                             gradients)
    t0 = time.time()
    src, surfs = _resolve_source(req.source)
    flux = req.flux
    if flux is None and src.request:
        flux = src.request.get("flux_Wb")
    if getattr(src, "has_surface_current", False):
        from .sheet import refusal
        said = refusal(src, flux)
        if said is None and req.gradients:
            said = "gradients=True is not offered for a source with a SurfaceCurrent: its field is not differentiated"
        if said is None and req.trace_lines:
            said = "trace_lines is not offered for a source with a SurfaceCurrent"
        if said is not None:
            raise ValueError(said)
        surfs = None
    budget = float(req.budget_s) if req.budget_s else estimate_budget(src, flux)
    deadline = t0 + budget
    with B.scope(deadline):
        out = _solve(req, src, surfs, budget, t0, deadline)
    _truthful(out, req)
    if out._engine is not None and out.record:
        from .closure import measure_record
        # how well each closed line the record keeps closes, and the edge barrier's screened lines brought to
        # convergence where they can be (torux.closure), before the chains are read from it.  A source with a sheet:
        # on an engine that gives no field inside the sheet's boundary (a line there is not converged, not an error)
        eng = out._engine
        if getattr(src, "has_surface_current", False):
            from .field import FieldEngine
            eng = FieldEngine(src, dmin=out._engine.dmin, sheet_nan=True)
        measure_record(eng, src, out.record)
    try:
        from .chainlist import of_record
        # the island chains the request identified, whatever its outcome (torux.chainlist)
        out.evidence["chains"] = of_record(out.record)
    except Exception:  # noqa: BLE001  (a record of an unexpected shape costs the list, never the answer)
        out.evidence["chains"] = []
    out.wall_s = time.time() - t0
    if out.wall_s > budget:
        out.evidence["budget_exceeded_s"] = out.wall_s - budget
        if out.status not in ("surface", "no_surface"):
            out.evidence["status_at_deadline"] = out.status
            out.status = "timeout"
    return out


def _truthful(out, req):
    """A surface or a range's lower bound is claimed only when the solver's acceptance and the requested independent
    verification agree; otherwise the status is '..._unverified' and evidence['unverified'] says why (the reference
    is the arbiter: e.g. a torus accepted on the solver's own off-grid samples but not at the reference's largest
    B.n next to a coil)."""
    rec = out.record or {}
    if out.status == "range" and not (rec.get("range") or {}).get("verified"):
        out.evidence["unverified"] = "the solver's acceptance refused the range's outer torus"
        out.status = "range_unverified"
    if out.status in ("surface", "range") and req.verify:
        ind = out.evidence.get("independent") or {}
        if not ind.get("ok"):
            out.evidence["unverified"] = ("independent verification skipped: " + str(ind["skipped"])
                                          if ind.get("skipped") else "the independent verification refused the torus")
            out.status += "_unverified"


def _solve(req, src, surfs, budget, t0, deadline):
    from .analysis import analyse, analyse_design
    flux = req.flux
    if flux is None and src.request:
        flux = src.request.get("flux_Wb")
    # sources that are not stellarator-type confinement fields (a tokamak's coils, perturbation coils, dipoles):
    # the structure and its proofs (torux.nonstell); an elliptic period-one line there goes on to the torus pipeline
    try:
        from . import nonstell as NS
        in_scope = bool(NS.scope(src)[0]) and not getattr(src, "has_surface_current", False)
    except Exception:  # noqa: BLE001
        in_scope = False
    ns = None
    if in_scope:
        try:
            ns = NS.classify(src)
        except Exception as e:  # noqa: BLE001
            ns = dict(verdict="ERROR", error=repr(e))
        v = ns.get("verdict", "")
        proved = {"NO_TOROIDAL_FLUX_SURFACE": "no_surface", "NO_VACUUM_INTERIOR_SURFACE": "no_surface",
                  "DIPOLE_CLOSED_FIELD_LINES": "no_surface"}
        if v in proved or v not in ("ELLIPTIC_PERIOD1_LINE", "NEAR_IDENTITY_ELLIPTIC"):
            status = proved.get(v, "undetermined")
            out = Result(status, budget, time.time() - t0, record=ns)
            out.evidence["proof"] = ns
            return out
    # the analysis gets what is left after reserves for the independent verification and the gradients
    reserve = (0.12 * budget if req.verify else 0.0) + (0.15 * budget if req.gradients else 0.0)
    a_budget = max(0.25 * budget, deadline - time.time() - reserve)
    try:
        if getattr(src, "has_surface_current", False):
            # a source with a current sheet: from its boundary outward (torux.sheet)
            from .sheet import analyse_sheet, symmetric
            if not symmetric(src):
                return Result("error", budget, time.time() - t0, evidence=dict(error=(
                    "a source with a SurfaceCurrent is solved with stellarator symmetry, and this one has none (its "
                    "claim, or its sheet's coefficients)")))
            rec = analyse_sheet(src, flux=flux, budget_s=a_budget, tol=req.tol)
        elif surfs is not None:
            no_req = flux is None
            rec = analyse_design(src, surfs, flux=flux, budget_s=(0.7 if no_req else 1.0) * a_budget, tol=req.tol,
                                 range_budget_s=0.3 * a_budget if no_req else 0.0)
        else:
            # (traced) the family crosses a chain its jump could not by a torus from traced lines (torux.beyond):
            # set on the source by the search for the largest (torux.ask)
            rec = analyse(src, flux=flux, budget_s=a_budget, tol=req.tol, traced=getattr(src, "_traced", None))
    except Exception as e:  # noqa: BLE001
        return Result("error", budget, time.time() - t0, evidence=dict(error=repr(e)))
    if ns is not None:
        rec["proof"] = ns
    family = rec.pop("_family", None)
    status = rec.get("status", "error")
    if status == "family":
        status = "range"
    out = Result(status, budget, 0.0, record=rec)
    out._family = family
    s = rec.get("surface")
    from .field import FieldEngine
    out._engine = FieldEngine(src, dmin=rec.get("engine", {}).get("dmin", 0.05))
    general = isinstance(s, dict) and s.get("basis") == "general"
    if isinstance(s, dict) and "rc" in s:
        if general:
            from .family_general import torus_from_record
            T = torus_from_record(s, src.nfp)
        else:
            from .torus import TorusState
            # (a torus around a closed line of the full-turn map carries its own number of periods, torux.noaxis)
            T = TorusState(np.array(s["rc"]), np.array(s["zs"]), s["iota"], s["flux"], s["M"], s["N"],
                           s.get("nfp", src.nfp))
        out.surface, out.iota = T, float(s["iota"])
        out.evidence["solver"] = {k: rec.get(k) for k in ("guard", "refine_hist", "axes", "profile", "path")}
        out.evidence["solver"]["bn_off"] = s.get("bn_off")
        out.evidence["solver"]["quad_err_half"] = s.get("quad_err_half")
        if s.get("sheet") is not None:
            out.evidence["solver"]["sheet"] = s["sheet"]
        g_res = 0.15 * budget if req.gradients else 0.0
        if req.verify:
            v_deadline = deadline - g_res
            if time.time() < v_deadline:
                if general:
                    from .verify_general import verify_general
                    from .verify import acceptance
                    v = verify_general(src, T, nlines=int(req.trace_lines), nperiods=100, deadline=v_deadline)
                    v["ok"] = acceptance(v)
                    out.evidence["independent"] = v
                else:
                    from .verify import verify_torus, acceptance
                    try:
                        v = verify_torus(src, T, nlines=int(req.trace_lines), nperiods=100, deadline=v_deadline)
                        v["ok"] = acceptance(v)
                    except Exception as e:  # noqa: BLE001
                        if not getattr(src, "has_surface_current", False):
                            raise
                        v = dict(skipped=f"the independent evaluator could not measure it ({e})", ok=False)
                    out.evidence["independent"] = v
            else:
                out.evidence["independent"] = dict(skipped="no time left in the budget")
        if req.gradients and general and status in ("surface", "range"):
            if time.time() > deadline - 0.05 * budget:
                out.gradients = dict(skipped="no time left in the budget")
            else:
                from .gradients import adjoint_gradient_general, iota_gradients
                try:
                    g = adjoint_gradient_general(out._engine, T, "iota")
                    out.gradients = iota_gradients(out._engine.source, T, True, g)
                except Exception as e:  # noqa: BLE001
                    out.gradients = dict(error=repr(e))
        elif req.gradients and status in ("surface", "range"):
            if time.time() > deadline - 0.05 * budget:
                out.gradients = dict(skipped="no time left in the budget")
            else:
                from .gradients import adjoint_gradient, adjoint_gradient_mf, iota_gradients
                try:
                    # dense adjoint up to the dense Newton's size, matrix-free above
                    n_unk = 2 * T.M * (2 * T.N + 1) + 2 * T.N + 2
                    g = (adjoint_gradient if n_unk <= 6000 else adjoint_gradient_mf)(out._engine, T, "iota")
                    out.gradients = iota_gradients(out._engine.source, T, False, g)
                except Exception as e:  # noqa: BLE001
                    out.gradients = dict(error=repr(e))
    if rec.get("certificate"):
        out.evidence["certificate"] = rec["certificate"]
    if rec.get("sheet") is not None:
        # a source with a current sheet: its boundary's flux, and why an answer is undetermined where it is
        out.evidence["sheet"] = rec["sheet"]
        if rec.get("why"):
            out.evidence["why"] = rec["why"]
    return out
