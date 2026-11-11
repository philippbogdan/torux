"""A surface asked for by its size, or the largest one the coils hold.

    torux.solve("coils.json", minor_radius=0.15)   # m: the radius of a circle of the mean cross-section's area
    torux.solve("coils.json", volume=0.44)         # m^3: the volume the surface encloses
    torux.solve("coils.json", largest=True)        # the largest surface torux can verify for these coils

Every one is answered by the shipped solver's direct requests at fluxes (torux.api.solve), never by scaling or
interpolating a surface: the surface returned is the solver's answer to a request for its own flux.
res.evidence["search"] says what was asked, what was reached, what stands beyond, the budget and every request
made, with its outcome and time. search_rows(res) gives the same as a few lines to print.

The sizes are VMEC's and SIMSOPT's: the volume enclosed, and sqrt(A / pi) with A the cross-sectional area at
fixed cylindrical angle averaged over the angle.
"""
from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np

# a size is reached when it is this close to the one asked (relative). A search cut short by its budget still
# returns a surface within ACCEPT of it, and none further away
RTOL = 1e-12
ACCEPT = 1e-9
MAX_SIZE_REQUESTS = 24
# the largest: probes between the largest verified flux and the smallest one beyond it that failed stop when the two
# are this close (relative)
GAP = 0.01
MIN_REQUEST_S = 5.0
UNITS = {"minor_radius": "m", "volume": "m^3"}
WORDS = {"minor_radius": "minor radius", "volume": "volume"}


# --------------------------------------------------------------------------- the two places that touch the solver
def _request(src, flux, budget_s, opts):
    """The one call of the solve: the shipped solver's answer to one request."""
    from .api import solve
    return solve(src, flux, budget_s=budget_s, **opts)


def size(surface):
    """The volume (m^3) and the minor radius (m) of a surface, by VMEC's and SIMSOPT's definitions: the volume
    enclosed, and sqrt(A / pi) with A the mean cross-sectional area at fixed cylindrical angle (the measure
    torux.Answer reports)."""
    from .answer import _size
    g = _size(surface)
    return dict(volume=g["volume"], minor_radius=math.sqrt(g["mean_section"] / math.pi),
                mean_cross_section=g["mean_section"])


# --------------------------------------------------------------------------- the one entry point
def solve(source, flux=None, *, minor_radius=None, volume=None, largest=False, budget_s=None, tol=3e-15,
          verify=True, trace_lines=0, gradients=False):
    """The flux surface of ``source`` asked for by one of ``flux`` (Wb), ``minor_radius`` (m), ``volume`` (m^3)
    or ``largest``. With none of them the source's own request, else the largest surface torux can verify."""
    from .api import Request
    one_question(flux, minor_radius, volume, largest)
    opts = dict(tol=tol, verify=verify, trace_lines=trace_lines, gradients=gradients)
    given = flux is not None or minor_radius is not None or volume is not None or largest
    if isinstance(source, Request):
        if given:
            raise ValueError("a torux.Request carries its own flux: give the size or largest with the source instead")
        return _request(source, None, None, {})
    if budget_s is not None and (not _number(budget_s) or not budget_s > 0):
        raise ValueError(f"budget_s={budget_s!r}: give the seconds for the whole question as a positive number")
    if flux is not None:
        return _request(source, flux, budget_s, opts)
    for k, v in (("minor_radius", minor_radius), ("volume", volume)):
        if v is not None:
            return _size_search(_source(source), k, float(v), budget_s, opts)
    src = _source(source)
    own = (src.request or {}).get("flux_Wb")
    if largest or not own:
        return _largest(src, budget_s, opts)
    return _request(src, None, budget_s, opts)


def _number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(v))


def one_question(flux=None, minor_radius=None, volume=None, largest=False):
    """ValueError unless at most one of flux, minor_radius, volume and largest is given, each a number that can be
    asked for."""
    given = [k for k, v in (("flux", flux), ("minor_radius", minor_radius), ("volume", volume)) if v is not None]
    if largest:
        given.append("largest")
    if len(given) > 1:
        raise ValueError(f"give one of flux, minor_radius, volume or largest, not {' and '.join(given)}")
    if largest not in (True, False):
        raise ValueError(f"largest={largest!r}: give largest=True, or leave it out")
    if flux is not None and (not _number(flux) or flux == 0):
        raise ValueError(f"flux={flux!r}: give the full toroidal flux in Wb as a number other than zero")
    for k, v in (("minor_radius", minor_radius), ("volume", volume)):
        if v is not None and (not _number(v) or v <= 0):
            raise ValueError(f"{k}={v!r}: give it as a positive number of "
                             f"{'metres' if k == 'minor_radius' else 'cubic metres'}")


def _source(source):
    if isinstance(source, str) or hasattr(source, "__fspath__"):
        from .sources import load
        return load(source)
    return source


# --------------------------------------------------------------------------- what a search says as it goes
# A search of many minutes says what it asks, how long it may take, and each request as it ends.  In Python it
# prints nothing: the lines go to the logger "torux" at INFO, for whoever listens
# (logging.basicConfig(level=logging.INFO)).  The command line sets `say` to a function that writes them to stderr.
say = None


def _say(text):
    logging.getLogger("torux").info(text)
    if say is not None:
        say(text)


def stated_budget(src, question):
    """The budget (s) a search states when none is given: three times the solver's own for one request, a flux for
    a size ('minor_radius', 'volume'), none for the largest ('largest')."""
    from .api import estimate_budget
    return 3.0 * estimate_budget(src, None if question == "largest" else 1.0)


# --------------------------------------------------------------------------- bookkeeping shared by both searches
class _Search:
    """The requests of one question inside its budget: each one logged with its flux, budget, outcome and time."""

    def __init__(self, src, question, asked, budget_s, opts):
        from .api import estimate_budget
        self.src, self.question, self.asked, self.opts = src, question, asked, opts
        self.stated = budget_s is None
        self.budget = float(stated_budget(src, question) if budget_s is None else budget_s)
        self.t0 = time.time()
        self.deadline = self.t0 + self.budget
        self.what = what = ("the largest surface torux can verify" if question == "largest" else
                            f"the surface of {WORDS[question]} {asked:.9g} {UNITS[question]}")
        _say(f"{getattr(src, 'id', None) or 'the source'}: asking for {what}, by requests at fluxes: up to "
             f"{self.budget:.0f} s ({_minutes(self.budget)}), " +
             ("a budget stated from the source's size since none was given" if self.stated else "the budget given"))
        self.log = []
        self.best = {}            # flux -> Result of the verified requests worth keeping
        self.size_of = {}         # flux -> size of its verified surface
        self.failed = {}          # flux -> log entry of a request that returned no surface
        self.certs = []           # (flux asked, certificate in short, the solver's own) of every valid certificate
        self.missed = None        # what was asked with the answer and the budget left no room for, in words

    def left(self):
        return self.deadline - time.time()

    def default_budget(self, flux):
        from .api import estimate_budget
        return estimate_budget(self.src, flux)

    def ask(self, flux, purpose, budget=None, src=None, opts=None, keep=True):
        """One request at ``flux`` (None: no flux, the family from the axis). Its budget the solver's own default
        for it, or ``budget``, and never more than what is left of the question's."""
        b = self.default_budget(flux) if budget is None else budget
        b = min(b, self.left())
        if b < MIN_REQUEST_S:
            return None
        o = dict(self.opts, gradients=False, trace_lines=0) if opts is None else opts
        t = time.time()
        r = _request(self.src if src is None else src, flux, b, o)
        entry = dict(flux=None if flux is None else float(flux), purpose=purpose, budget_s=round(b, 1),
                     wall_s=round(time.time() - t, 1), status=r.status)
        ok = verified(r)
        if ok and flux is not None:
            sz = size(r.surface)
            entry.update(iota=float(r.iota), **sz)
            if not repeatable(r):
                entry["repeatable"] = False
                entry["path"] = (r.record or {}).get("path")
            self.size_of[float(flux)] = sz
            if keep:
                self.best[float(flux)] = r
        elif flux is not None:
            entry["why"] = _why(r)
            self.failed[float(flux)] = entry
        c = _certificate(r)
        if c is not None:
            entry["certificate"] = c
            self.certs.append((flux, c, (r.evidence or {}).get("certificate") or (r.record or {}).get("certificate")))
        if flux is None:
            rec = r.record or {}
            rg = rec.get("range") or (rec.get("range_family") or {}).get("range") or {}
            fam = _family(rec)
            entry["family_reached_Wb"] = max(fam) if fam else None
            entry["verified_lower_bound_Wb"] = rg.get("verified_lower_bound")
        entry["default_budget"] = bool(b >= self.default_budget(flux))
        self.log.append(entry)
        r._ask_entry = entry
        at = "no flux (the family from the axis)" if flux is None else _wb(float(flux))
        far = ""
        if flux is None and entry.get("family_reached_Wb"):
            lb = entry.get("verified_lower_bound_Wb")
            far = (f" (verified to {_wb(abs(float(lb)))}, " if lb else " (") + f"followed to " \
                  f"{_wb(entry['family_reached_Wb'])})"
        _say(f"request {len(self.log)}, {at}: {r.status.replace('_', ' ')}{far} in {entry['wall_s']:.0f} s, "
             f"{max(0.0, self.left()):.0f} s of the budget left")
        return r

    def finish(self, r, words, brief=False, **fields):
        """The answer: ``r`` (a Result) with the search in its evidence, its time and budget the question's."""
        from .api import Result
        if r is None:
            r = Result("timeout" if self.left() <= MIN_REQUEST_S else "undetermined", self.budget, 0.0)
        r.evidence = dict(r.evidence or {})
        budget_words = (f"{self.budget:.0f} s, stated from the source's size since none was given" if self.stated
                        else f"{self.budget:.0f} s, as given")
        n = len(self.log)
        req_words = (f"requests: {n}, {sum(e['wall_s'] for e in self.log):.0f} s in all "
                     "(answer.evidence['search']['requests'])")
        if self.missed and not brief:
            words = list(words) + [self.missed]
        r.evidence["search"] = dict(question=self.question, asked=self.asked, budget_s=self.budget,
                                    budget_stated=self.stated, wall_s=time.time() - self.t0, requests=self.log,
                                    words=[f"asked: {self.what}"] + ([] if brief else list(words) + [
                                        req_words, f"budget: {budget_words}"]),
                                    **fields)
        r.budget_s = self.budget
        r.wall_s = time.time() - self.t0
        return r

    def final(self, flux):
        """The Result to return at a verified flux: the one in hand, the solver's answer to the request that returned
        it. Asked again only when the gradients or traced lines were asked, which the search's requests leave out."""
        r = self.best[flux]
        if self.opts.get("gradients") or self.opts.get("trace_lines"):
            r2 = self.ask(flux, "the answer again, with the gradients and traced lines asked", opts=self.opts)
            if r2 is not None and verified(r2):
                return r2
            what = " and ".join(w for w, on in (("gradients", self.opts.get("gradients")),
                                                ("traced lines", self.opts.get("trace_lines"))) if on)
            self.missed = (f"note: the budget ended before the surface could be solved again with the {what} asked "
                           f"for: this answer is without them")
        return r

    def by(self, r):
        """Which request returned ``r``: its number, flux (Wb) and budget (s), and the words for it."""
        e = getattr(r, "_ask_entry", {})
        k = next((i + 1 for i, x in enumerate(self.log) if x is e), None)
        return (dict(request=k, flux=e.get("flux"), budget_s=e.get("budget_s")),
                f"solved at that flux (request {k}, with a budget of {e.get('budget_s', 0):.0f} s), not scaled or "
                f"interpolated")


def verified(r):
    """A surface the solver verified, not cut short by a deadline (so the same request gives the same surface)."""
    if r is None or r.status != "surface" or r.surface is None:
        return False
    rec = r.record or {}
    return rec.get("refine_stop") != "deadline"


def repeatable(r):
    """Whether a verified surface's request reached it without a step whose outcome depends on the clock: the
    design path's late stages (after the Newton from the design surface failed) share out the request's budget by
    the time they take, so the same request on a busier or slower machine may end otherwise."""
    rec = r.record or {}
    return not rec.get("resonant_path") and rec.get("refine_stop") != "deadline" and not rec.get("budget_exhausted")


def _terminal(r):
    """An outcome that holds at every flux: a proof (a tokamak's coils, a dipole), no magnetic axis, an error."""
    if r is None:
        return False
    if r.status == "no_surface" and (r.evidence or {}).get("proof") and not (r.evidence or {}).get("certificate"):
        return True
    return r.status in ("no_axis", "error")


def _certificate(r):
    """The certificate of a request in short, when it is valid: kind, and the bound or the excluded interval (Wb)."""
    return _short((r.evidence or {}).get("certificate") or (r.record or {}).get("certificate"))


def _short(c):
    if not c or not c.get("ok") or not c.get("rigorous", True):
        return None
    kind = str(c.get("kind", ""))
    edge = kind.startswith("edge") or (c.get("F_up") is not None and not kind.startswith("island"))
    out = dict(kind="edge barrier" if edge else "island band" if kind.startswith("island")
               else kind.replace("_", " "))
    try:
        if edge:
            if c.get("F_up") is not None and np.isfinite(c["F_up"]):
                out["no_surface_above_Wb"] = float(c["F_up"])
        else:
            iv = c.get("interval")
            if iv is not None and len(iv) == 2 and all(np.isfinite(np.asarray(iv, float))):
                out["no_surface_between_Wb"] = sorted([abs(float(iv[0])), abs(float(iv[1]))])
    except (TypeError, ValueError):
        return None
    if len(out) == 1:
        return None
    return out


def _why(r):
    ev = r.evidence or {}
    if r.status == "no_surface":
        return "certified: no surface" if ev.get("certificate") else "proved: no surface"
    if r.status in ("surface", "range") and (r.record or {}).get("refine_stop") == "deadline":
        return "a surface was found and the request's budget ended before its refinement did"
    if r.status == "surface_unverified" and not ev.get("unverified"):
        bn = (ev.get("solver") or {}).get("bn_off")
        return "a torus was found and the solver's own checks refused it" + (
            f" (field across it {bn:.1e} of |B|)" if bn is not None and np.isfinite(bn) else "")
    return str(ev.get("unverified") or ev.get("error") or (r.record or {}).get("deferred_reason")
               or ev.get("status_at_deadline") or "neither a surface nor a certificate within the budget")


def _family(rec):
    """Fluxes (Wb) of the converged tori the family of a request followed: each a flux a direct request may verify."""
    out = []
    for prof in (rec.get("profile"), (rec.get("range_family") or {}).get("profile")):
        for t in prof or []:
            bn = t.get("bn_off")
            if t.get("converged") and bn is not None and np.isfinite(bn):
                out.append(abs(float(t["flux"])))
    return sorted(set(out))


def _minutes(t):
    return f"{t / 60:.0f} minutes" if t >= 90 else f"{t:.0f} s"


def _wb(x):
    return f"{x:.9g} Wb"


def _sz(k, v):
    return f"{v:.12g} {UNITS[k]}"


# --------------------------------------------------------------------------- a size
def _size_search(src, which, target, budget_s, opts):
    """The verified surface whose minor radius or volume is ``target``: requests at fluxes, each flux from a model
    u(F) = a F + b F^2 (u the volume or the minor radius squared, both ~ F) through the verified surfaces nearest
    the target, until the size is within RTOL of it.  When the model puts the size past a flux at which no surface
    was verified, the solver's answer at the flux the size would need is the answer."""
    S = _Search(src, which, target, budget_s, opts)
    u_of = (lambda s: s["minor_radius"] ** 2) if which == "minor_radius" else (lambda s: s["volume"])
    u_t = target ** 2 if which == "minor_radius" else target
    what = f"{WORDS[which]} {target:.9g} {UNITS[which]}"
    F, concluding = _first_flux(src, which, target), False
    asked = []
    while len(S.log) < MAX_SIZE_REQUESTS:
        cover = _covering(S, F) if S.size_of else None
        if cover is not None:
            return _size_not_held(S, which, target, F, cover[0], cover[2], None)
        r = S.ask(F, f"the flux {what} would need, beyond the last verified surface" if concluding else
                  f"the flux of {what} as the verified surfaces predict" if S.size_of else
                  f"the flux of {what}, a first estimate" if not S.failed else
                  "half the least flux that failed or that a certificate rules out, for a first verified surface")
        if r is None:
            break
        asked.append(F)
        if _terminal(r):
            return S.finish(r, [], brief=True)
        if concluding and not verified(r):
            c = _certificate(r)
            return _size_not_held(S, which, target, F, F, r.evidence.get("certificate") if _covers(c, F) else None, r)
        best = min(S.size_of.items(), key=lambda kv: abs(kv[1][which] - target), default=None)
        if best is not None and abs(best[1][which] - target) <= RTOL * target:
            return _size_found(S, which, target, best[0])
        pts = sorted((f, u_of(s)) for f, s in S.size_of.items())
        if not pts and len(S.failed) >= 4:
            break            # four fluxes, each half the last, and no surface verified at any
        ceiling = min(list(S.failed) + [c.get("no_surface_above_Wb", np.inf) for _, c, _ in S.certs], default=np.inf)
        F_next, concluding = _next_flux(pts, u_t, S.failed, ceiling)
        if F_next is None or any(abs(F_next - a) <= 4e-16 * a for a in asked if a in S.size_of):
            break            # nothing left to ask, or a verified flux again: as close as the arithmetic allows
        same = [a for a in S.failed if abs(F_next - a) <= 1e-12 * a]
        if same and pts:
            # the flux the size would need was already asked and gave no surface
            c = S.failed[same[0]].get("certificate")
            orig = next((o for f, cc, o in S.certs if f == same[0]), None)
            return _size_not_held(S, which, target, F_next, same[0], orig if _covers(c, F_next) else None, None)
        F = F_next
    # cut short (the budget, the number of requests): the closest verified surface if it is within ACCEPT, else none
    best = min(S.size_of.items(), key=lambda kv: abs(kv[1][which] - target), default=None)
    if best is not None and abs(best[1][which] - target) <= ACCEPT * target:
        return _size_found(S, which, target, best[0], cut=True)
    words = ["found: no verified surface of that size " + ("within the budget" if S.left() <= MIN_REQUEST_S
                                                           else f"in {len(S.log)} requests")]
    if best is not None:
        words.append(f"closest verified: {WORDS[which]} {_sz(which, best[1][which])}, at {_wb(best[0])}")
    from .api import Result
    out = Result("timeout" if S.left() <= MIN_REQUEST_S else "undetermined", S.budget, 0.0)
    return S.finish(out, words, reached=None, flux=None,
                    closest=None if best is None else dict(flux=best[0], **best[1]))


def _first_flux(src, which, target):
    """A first flux for a size: from the source's design surface and its flux when it has them (never beyond the
    design flux), else from the field at the coils' median radius (0.7 of the flux a uniform field would need)."""
    own = abs((src.request or {}).get("flux_Wb") or 0.0)
    surfs = getattr(src, "_design_surfaces", None)
    if own and surfs:
        try:
            from .io.surfaces import surface_properties
            p = surface_properties(surfs[-1])
            u_d = p["minor_radius"] ** 2 if which == "minor_radius" else p["volume"]
            u_t = target ** 2 if which == "minor_radius" else target
            if np.isfinite(u_d) and u_d > 0:
                return float(min(own, own * u_t / u_d))
        except Exception:  # noqa: BLE001
            pass
        return float(own)
    if own:
        return float(own)
    from .field import FieldEngine
    pts = src.sample_points()
    R = float(np.median(np.hypot(pts[:, 0], pts[:, 1])))
    ph = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    x = np.stack([R * np.cos(ph), R * np.sin(ph), 0 * ph], -1)
    eng = FieldEngine(src, dmin=src.meta.get("dmin", 0.05))
    B = float(np.mean(np.linalg.norm(np.asarray(eng.evaluate(x, grad=False, A=False)["B"]), axis=-1)))
    area = math.pi * target ** 2 if which == "minor_radius" else target / (2 * math.pi * R)
    return float(0.7 * B * area)


def _next_flux(pts, u_t, failed, ceiling=np.inf):
    """The next flux to ask, and whether it lies beyond a flux at which no surface was verified.

    pts: (flux, u) of the verified surfaces.  The model u = a F + b F^2 through the origin and the two verified
    surfaces nearest the target (u = c F through one), kept inside the bracket the verified surfaces give.  With no
    verified surface yet: half the least flux that failed or that a certificate rules out (``ceiling``)."""
    if not pts:
        return (0.5 * ceiling, False) if np.isfinite(ceiling) else (None, False)
    near = sorted(pts, key=lambda p: abs(p[1] - u_t))[:2]
    F = None
    if len(near) == 2 and near[0][0] != near[1][0]:
        (f1, u1), (f2, u2) = near
        A = np.array([[f1, f1 * f1], [f2, f2 * f2]])
        try:
            a, b = np.linalg.solve(A, np.array([u1, u2]))
            if abs(b) * max(f1, f2) < 1e-12 * abs(a):
                F = u_t / a
            else:
                disc = a * a + 4 * b * u_t
                if disc >= 0:
                    roots = [(-a + s * math.sqrt(disc)) / (2 * b) for s in (1, -1)]
                    roots = [x for x in roots if x > 0]
                    if roots:
                        F = min(roots, key=lambda x: abs(x - f1))
        except np.linalg.LinAlgError:
            F = None
        if F is None or not np.isfinite(F):
            F = f1 + (u_t - u1) * (f2 - f1) / (u2 - u1) if u2 != u1 else None
    if F is None:
        f1, u1 = near[0]
        F = f1 * u_t / u1
    below = [p for p in pts if p[1] < u_t]
    above = [p for p in pts if p[1] > u_t]
    lo = max(below)[0] if below else 0.0
    hi = min(above)[0] if above else np.inf
    if not (lo < F < hi):
        F = 0.5 * (lo + hi) if np.isfinite(hi) else 2 * lo
    walls = [f for f in failed if lo < f < hi]
    if walls and F >= min(walls):
        return float(F), True
    return float(F), False


def _covers(c, F):
    """Whether a certificate (in short) rules out a surface enclosing the flux F."""
    if c is None:
        return False
    up = c.get("no_surface_above_Wb")
    iv = c.get("no_surface_between_Wb")
    return bool((up is not None and F > up) or (iv is not None and iv[0] <= F <= iv[1]))


def _covering(S, F):
    """(flux asked, certificate in short, the solver's own) of an earlier certificate that rules out a surface at
    flux F, or None."""
    for t in S.certs:
        if _covers(t[1], F):
            return t
    return None


def _size_found(S, which, target, flux, cut=False):
    r = S.final(flux)
    sz = S.size_of[flux]
    reached = sz[which]
    answered_by, how = S.by(r)
    words = [f"reached: {WORDS[which]} {_sz(which, reached)}, within {abs(reached - target) / target:.1e} of it"
             + (", the closest the budget allowed" if cut else "") + f", {how}"]
    if not repeatable(r):
        words.append(_clock_words(getattr(r, "_ask_entry", {})))
    return S.finish(r, words, reached=reached, flux=flux, size=sz, repeatable=repeatable(r), answered_by=answered_by)


def _size_not_held(S, which, target, F_need, f_asked, cert, r):
    """The size lies past what the verified surfaces reach: the solver's answer at the flux it would need (``r``,
    the request at ``f_asked``), or the certificate of an earlier request that already rules that flux out."""
    from .api import Result
    words = []
    vs = sorted(S.size_of.items())
    if vs:
        f_last, s_last = max(vs, key=lambda kv: kv[1][which])
        words.append(f"largest verified here: {WORDS[which]} {_sz(which, s_last[which])}, at {_wb(f_last)}, by "
                     f"this search's requests (largest=True looks for the largest the coils hold)")
        words.append(f"would need: about {_wb(F_need)}, extrapolated from the verified surfaces")
    else:
        words.append(f"would need: about {_wb(F_need)}, and no surface was verified on the way")
    if cert is not None:
        words.append(f"at that flux: {_cert_words(_short(cert))}, the certificate of the request at {_wb(f_asked)}")
        if r is None or r.status != "no_surface":
            r = Result("no_surface", S.budget, 0.0)
            r.evidence["certificate"] = cert
    else:
        e = S.failed.get(f_asked, {})
        words.append(f"at that flux: torux could not decide within {e.get('budget_s', 0):.0f} s "
                     f"({e.get('why', 'no surface verified')})")
        if r is None:
            r = Result("undetermined", S.budget, 0.0)
    if r.surface is not None or r.status in ("surface", "range"):
        # a torus that is not verified, or is at another flux, is never the answer to this question: nor is its status
        r = copy.copy(r)
        r.surface, r.iota = None, np.nan
        if r.status in ("surface", "range"):
            r.status = "timeout" if S.left() <= MIN_REQUEST_S else "undetermined"
    return S.finish(r, words, reached=None, flux=F_need, flux_asked=f_asked)


# --------------------------------------------------------------------------- the largest
def _largest(src, budget_s, opts):
    """The largest surface torux can verify for ``src`` by any request it can make within the budget: its own
    request, the family from the magnetic axis (the no-flux request), direct requests at the fluxes that family
    reached and between them, and a request beyond them all for a certificate of what lies past the last surface."""
    S = _Search(src, "largest", None, budget_s, opts)
    state = dict(lo=None)

    def take(f):
        lo = state["lo"]
        if lo is None or f > lo:
            for k in [k for k in S.best if k != f]:
                del S.best[k]          # only the answer's Result is kept (each holds a field engine)
            state["lo"] = f
        else:
            S.best.pop(f, None)

    def probe(f, purpose):
        """A direct request at f with the solver's own default budget (what is left, if less). True when it
        verified."""
        r = S.ask(f, purpose)
        if r is None:
            raise _OutOfTime
        if _terminal(r):
            raise _Terminal(r)
        if verified(r):
            take(f)
            return True
        return False

    # the same coils with no request and no design surfaces of their own: the family from the axis outward
    free = src
    own = (src.request or {}).get("flux_Wb")
    if own or getattr(src, "_design_surfaces", None) is not None:
        free = copy.copy(src)
        free.request = None
        free._design_surfaces = None
    try:
        if own:
            probe(abs(float(own)), "the source's own request")
        rng = S.ask(None, "no flux: the family of surfaces from the magnetic axis outward",
                    budget=min(S.default_budget(None), 0.4 * S.left()), src=free)
        if rng is None:
            raise _OutOfTime
        if _terminal(rng):
            raise _Terminal(rng)
        cands = _family(rng.record or {})
        rg = (rng.record or {}).get("range") or {}
        del rng
        f_r = rg.get("verified_lower_bound")
        if f_r and (state["lo"] is None or f_r > state["lo"]) and f_r not in S.size_of:
            # the no-flux request's own lower bound, asked directly: a likely surface, and the time one takes
            probe(abs(float(f_r)), "the no-flux request's verified lower bound, asked directly")
        def climb(cands):
            """The family's fluxes beyond the largest verified one, outward, up to the first that fails (a surface
            takes seconds to a minute. A failure can take the request's whole budget)."""
            lo = state["lo"]
            c = [f for f in cands if (lo is None or f > lo * (1 + 1e-12)) and f not in S.failed and f not in S.size_of
                 and _covering(S, f) is None]
            if lo is None:
                c = c[len(c) // 2:]    # nothing verified yet: the outer half of the family
            for f in c:
                if not probe(f, "a flux the family from the axis reached"):
                    break

        climb(cands)
        if not any(c.get("no_surface_above_Wb") is not None for _, c, _ in S.certs):
            # beyond everything seen, from the axis, with half the time left (at least the solver's default): what
            # stands past the last surface, a certificate if one can be had. The other half closes the gap below
            seen = list(S.size_of) + list(S.failed) + cands
            if seen and S.left() > 4 * MIN_REQUEST_S:
                f_b = 2.0 * max(seen)
                r = S.ask(f_b, "beyond every flux seen, for what stands past the last surface", src=free,
                          budget=max(S.default_budget(f_b), 0.5 * S.left()))
                if r is None:
                    raise _OutOfTime
                if _terminal(r):
                    raise _Terminal(r)
                more = _family(r.record or {})
                if f_b in S.size_of:   # (asked of the coils alone: a surface there is asked again of the source)
                    S.best.pop(f_b, None)
                    S.size_of.pop(f_b)
                    more.append(f_b)
                del r
                climb(sorted(set(more)))
        # between the largest verified flux and the smallest beyond it that failed or that a certificate rules out,
        # until they are within GAP of each other
        for _ in range(6):
            lo = state["lo"]
            if lo is None:
                break
            up = [f for f in S.failed if f > lo] + [
                c["no_surface_above_Wb"] for _, c, _ in S.certs if c.get("no_surface_above_Wb", 0) > lo]
            if not up or (min(up) - lo) <= GAP * lo:
                break
            f = 0.5 * (lo + min(up))
            if _covering(S, f) is not None or S.left() < 0.25 * S.default_budget(f):
                break                  # (a request with a sliver of its budget decides nothing)
            probe(f, "between the largest verified flux and the least one beyond it that failed")
    except _OutOfTime:
        pass
    except _Terminal as t:
        return S.finish(t.r, [], brief=True)
    return _largest_answer(S, state["lo"])


class _OutOfTime(Exception):
    pass


class _Terminal(Exception):
    def __init__(self, r):
        super().__init__(r.status)
        self.r = r


def _largest_answer(S, lo):
    from .api import Result
    words = []
    fields = dict(flux=lo, beyond=None, outward=[])
    if lo is None:
        words.append("found: no verified surface, by any request within the budget")
        r = Result("timeout" if S.left() <= MIN_REQUEST_S else "undetermined", S.budget, 0.0)
    else:
        r = S.final(lo)
        sz = S.size_of[lo]
        fields["size"] = sz
        fields["repeatable"] = repeatable(r)
        fields["answered_by"], how = S.by(r)
        words.append(f"reached: {_wb(lo)}, {how}")
        if not repeatable(r):
            e = getattr(r, "_ask_entry", {})
            words.append(_clock_words(e))
    # what stands beyond it, outward in flux: each certificate where it stands, and the nearest flux beyond the
    # answer (and beyond each island band) at which torux could not decide
    floor = lo or 0.0
    items = []
    for f, c, _ in S.certs:
        if c.get("no_surface_above_Wb") is not None and c["no_surface_above_Wb"] >= floor:
            items.append((c["no_surface_above_Wb"], dict(c, cert=c["kind"], kind="certified bound", request_Wb=f)))
        elif c.get("no_surface_between_Wb") is not None and c["no_surface_between_Wb"][1] > floor:
            items.append((c["no_surface_between_Wb"][0], dict(c, cert=c["kind"], kind="island band", request_Wb=f)))
    undecided = sorted((f, e) for f, e in S.failed.items() if f > floor and _covering(S, f) is None)
    bands = sorted(it["no_surface_between_Wb"] for _, it in items if it["kind"] == "island band")
    for a in [floor] + [b for _, b in bands]:
        # the nearest undecided flux past a, if no band starts between a and it
        nxt = [(f, e) for f, e in undecided if f > a]
        if nxt and not any(a < b0 < nxt[0][0] for b0, _ in bands):
            # the nearest, and the nearest that had the solver's whole default budget when the nearest had less
            for f, e in [nxt[0]] + ([] if nxt[0][1].get("default_budget") else
                                    [x for x in nxt if x[1].get("default_budget")][:1]):
                items.append((f, dict(kind="undecided", flux_Wb=f, budget_s=e["budget_s"], why=e.get("why"))))
    seen_u, uniq = set(), []
    for x, it in sorted(items, key=lambda t: t[0]):
        key = (it["kind"], x)
        if key not in seen_u:
            seen_u.add(key)
            uniq.append((x, it))
    items = uniq
    # nothing past the least certified bound matters
    cut = next((i for i, (_, it) in enumerate(items) if it["kind"] == "certified bound"), None)
    if cut is not None:
        items = items[:cut + 1]
    fields["outward"] = [it for _, it in items]
    fields["beyond"] = items[0][1] if items else dict(kind="budget")
    for i, (_, it) in enumerate(items):
        label = "beyond it" if i == 0 else "then"
        if it["kind"] == "certified bound":
            words.append(f"{label}: above {_wb(it['no_surface_above_Wb'])} no surface ({it['cert']} certificate of the "
                         f"request at {_wb(it['request_Wb'])})")
        elif it["kind"] == "island band":
            iv = it["no_surface_between_Wb"]
            words.append(f"{label}: from {_wb(iv[0])} to {_wb(iv[1])} no surface ({it['cert']} certificate of the "
                         f"request at {_wb(it['request_Wb'])})")
        else:
            words.append(f"{label}: at {_wb(it['flux_Wb'])} torux could not decide within {it['budget_s']:.0f} s "
                         f"({it['why']})")
    if not items:
        words.append("beyond it: nothing decided, the budget ran out first")
    # a flux below the answer at which torux could not decide: said, since surfaces inside a surface are expected
    under = sorted((f, e) for f, e in S.failed.items() if lo is not None and f < lo and _covering(S, f) is None)
    fields["undecided_below"] = [dict(flux_Wb=f, budget_s=e["budget_s"], why=e.get("why")) for f, e in under]
    if under:
        f, e = under[-1]
        words.append(f"below it: at {_wb(f)} torux could not decide within {e['budget_s']:.0f} s"
                     + (f", nor at {len(under) - 1} smaller flux{'es' if len(under) > 2 else ''}" if len(under) > 1
                        else ""))
    ups = [x for x, it in items if it["kind"] == "certified bound"]
    if lo is not None and ups:
        words.append(f"so: the coils hold at least {_wb(lo)} and at most {_wb(ups[0])}")
    elif lo is not None:
        words.append("so: no certified bound on what the coils hold within the budget")
    if lo is not None:
        for f, c, _ in S.certs:
            if _covers(c, lo):
                words.append(f"disagreement: the certificate of the request at {_wb(f)} rules out the flux of this "
                             "verified surface: one of the two is wrong")
    return S.finish(r, words, **fields)


def _clock_words(e):
    return (f"note: a request for that flux with a budget of {e.get('budget_s', 0):.0f} s returned it in "
            f"{e.get('wall_s', 0):.0f} s, by the solver's design-path stages that share out the budget by the clock: "
            "the same request on a slower or busier machine may come back undetermined")


def _cert_words(c):
    if c.get("no_surface_above_Wb") is not None:
        return f"no surface encloses more than {_wb(c['no_surface_above_Wb'])} ({c['kind']} certificate)"
    iv = c["no_surface_between_Wb"]
    return f"no surface encloses a flux between {_wb(iv[0])} and {_wb(iv[1])} ({c['kind']} certificate)"


# --------------------------------------------------------------------------- printing
def search_rows(res):
    """The search behind an answer asked by size or for the largest, as (label, text) rows in the style of
    torux.cli.report. [] for an answer asked by flux."""
    s = (res.evidence or {}).get("search")
    if not s:
        return []
    rows = []
    for w in s["words"]:
        k, _, v = w.partition(": ")
        rows.append((k, v) if v else ("", w))
    return rows
