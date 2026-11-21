"""A surface asked for by its size, or the largest one the coils hold.

    torux.solve("coils.json", minor_radius=0.15)   # m: the radius of a circle of the mean cross-section's area
    torux.solve("coils.json", volume=0.44)         # m^3: the volume the surface encloses
    torux.solve("coils.json", largest=True)        # the largest surface torux can verify for these coils

Every one is answered by the shipped solver's direct requests at fluxes (torux.api.solve), never by scaling or
interpolating a surface: the surface returned is the solver's answer to a request for its own flux.
res.evidence["search"] says what was asked, what was reached, what stands beyond, the budget and every request
made, with its outcome and time. search_rows(res) gives the same as a few lines to print. The largest also lists the
island chains its answer's family stepped across (torux.crossed), in the time its requests left of the budget, or at
most CROSSED_S past it (CROSSED_SHARE of it, if less) when they left less, and as long again to refine the residues
of their lines.

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
# the largest: requests beyond every flux seen, each twice the last, while each comes back verified
MAX_BEYOND = 8
# the largest: the chains the answer's family stepped across (torux.crossed) take what the search left of the budget,
# and when that is less than the least of these seconds and this share of the budget, that much from the search's end,
# and then as long again to refine their lines' residues (torux.crossed.crossed_chains, refine_s): the search itself is
# never given less (so its answer is the one it would be without them)
CROSSED_S = 90.0
CROSSED_SHARE = 0.075
# the largest: where nothing certified bounds what the coils hold and a request's family stalled at a chain its jump
# could not cross, a stage traces lines beyond it (torux.beyond.cross) with the least of these seconds and this share of
# what is left (CROSS_MIN_S at least), at most MAX_CROSS times (a chain each), when at least CROSS_MIN_S are left
CROSS_STAGE_S = 300.0
CROSS_STAGE_SHARE = 0.3
CROSS_MIN_S = 150.0
MAX_CROSS = 3
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
          verify=True, trace_lines=0, gradients=False, widths_s=None, widths_jobs=None):
    """The flux surface of ``source`` asked for by one of ``flux`` (Wb), ``minor_radius`` (m), ``volume`` (m^3)
    or ``largest``. With none of them the source's own request, else the largest surface torux can verify.
    widths_s (largest only): seconds of its own for closing in on every chain of the answer with verified surfaces
    (torux.widths), widths_jobs requests at once (default 1)."""
    from .api import Request
    one_question(flux, minor_radius, volume, largest)
    widths = widths_question(widths_s, widths_jobs, flux, minor_radius, volume, largest)
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
            return _size_search(_vacuum_source(source), k, float(v), budget_s, opts)
    src = _vacuum_source(source)
    own = (src.request or {}).get("flux_Wb")
    if largest or not own:
        return _largest(src, budget_s, opts, widths)
    if widths is not None:
        raise ValueError("widths_s closes in on the chains of the largest surface's answer: give largest=True with it")
    return _request(src, None, budget_s, opts)


def widths_question(widths_s=None, widths_jobs=None, flux=None, minor_radius=None, volume=None, largest=False):
    """(widths_s, widths_jobs) when the chains are to be closed in on (torux.widths), else None; ValueError for a
    value that cannot be asked, or with a question other than the largest."""
    if widths_s is None:
        if widths_jobs is not None:
            raise ValueError("widths_jobs is the requests at once of widths_s: give widths_s with it")
        return None
    if not _number(widths_s) or not widths_s > 0:
        raise ValueError(f"widths_s={widths_s!r}: give the seconds for closing in on the chains as a positive number")
    if widths_jobs is not None and (isinstance(widths_jobs, bool) or int(widths_jobs) != widths_jobs
                                    or widths_jobs < 1):
        raise ValueError(f"widths_jobs={widths_jobs!r}: give a whole number of requests at once, at least 1")
    if flux is not None or minor_radius is not None or volume is not None:
        raise ValueError("widths_s closes in on the chains of the largest surface's answer: give largest=True with it")
    return float(widths_s), None if widths_jobs is None else int(widths_jobs)


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


def _vacuum_source(source):
    src = _source(source)
    if getattr(src, "has_surface_current", False):
        raise ValueError("SurfaceCurrent supports exterior field evaluation and poincare(starts=...), not solve")
    return src


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
        self.crossed = None       # (verified tori, engine, axis) of the crossed stage, for placing every chain
        self.after = 0.0          # seconds the crossed stage ran past the deadline
        self.traced = []          # the largest: the stages that traced lines beyond a chain (torux.beyond.cross)
        self.widths = None        # (brackets by chain, summary) of the closing in on the chains (torux.widths)
        self.tori = []            # every surface a request verified, kept to place every chain against (torux.bands)
        self.widths_tori = []     # the same, of the closing in
        self.widths_off = []      # and those of the closing in that are not around the magnetic axis

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
        if _verified_torus(r) is not None:
            # (a verified lower bound, the outermost torus of a request's family, is not the surface of a direct
            # request: the words say whose it is, torux.bands and _placed)
            who = "the no-flux request" if flux is None else f"the request at {float(flux):.10g} Wb"
            self.tori.append((r.surface, "search" if r.status == "surface" else LOWER_BOUND + who))
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
        # the chains its family crossed by a torus from traced lines (torux.beyond), or tried to
        tj = [dict(p=j.get("p"), q=j.get("q"), outcome=(j.get("traced") or {}).get("outcome"),
                   flux_after=j.get("flux_after")) for j in ((r.record or {}).get("jumps") or []) if j.get("traced")]
        if tj:
            entry["traced"] = tj
        # the chains its family stalled at with no torus beyond them (p, q): the rational it met first, where the
        # jump tried the lowest-order one past it (the record's jumps keep that one)
        st = [list(j.get("stalled_at") or [j.get("p"), j.get("q")]) for j in ((r.record or {}).get("jumps") or [])
              if not j.get("ok")]
        if st:
            entry["stalled"] = st
        entry["chains"] = [dict(c, request=len(self.log) + 1) for c in _chains(r)]
        if (r.record or {}).get("closed_lines"):
            entry["closed_lines"] = r.record["closed_lines"]     # what measuring its closed lines did and took
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
        if self.after > 0.5:
            budget_words += (f", and {self.after:.0f} s after it for the chains the answer's family stepped across "
                             f"(at most about {2 * min(CROSSED_S, CROSSED_SHARE * self.budget):.0f} s: half to find "
                             "their lines, half to refine their residues)")
        n = len(self.log)
        req_words = (f"requests: {n}, {sum(e['wall_s'] for e in self.log):.0f} s in all "
                     "(answer.evidence['search']['requests'])")
        if self.missed and not brief:
            words = list(words) + [self.missed]
        from .chainlist import merge
        chains = merge([(i + 1, e.get("chains")) for i, e in enumerate(self.log)])
        if self.crossed is not None:
            from .crossed import annotate
            try:
                annotate(chains, *self.crossed)
            except Exception:  # noqa: BLE001  (placing the chains costs only their placement, never the answer)
                pass
        if self.widths is not None:
            from .widths import attach
            attach(chains, self.widths[0])
        bands = self.fit_bands(chains)
        if chains and not brief:
            words = list(words) + _chain_words(chains)
        if self.widths is not None and not brief:
            from .widths import words as widths_words
            words = list(words) + [widths_words(self.widths[1])]
        if bands and not brief and (bands.get("cut") or bands.get("withdrawn")):
            words = list(words) + [_bands_words(bands)]
        r.evidence["search"] = dict(question=self.question, asked=self.asked, budget_s=self.budget,
                                    budget_stated=self.stated, wall_s=time.time() - self.t0, requests=self.log,
                                    words=[f"asked: {self.what}"] + ([] if brief else list(words) + [
                                        req_words, f"budget: {budget_words}"]),
                                    **fields)
        if self.widths is not None:
            r.evidence["search"]["widths"] = self.widths[1]
        if bands is not None:
            r.evidence["search"]["bands"] = bands
        # every chain any request of the search identified, once each (per request in requests[i]['chains'])
        r.evidence["search"]["chains"] = chains
        r.evidence["chains"] = chains
        r.budget_s = self.budget
        r.wall_s = time.time() - self.t0
        return r

    def surfaces(self):
        """Every surface the search verified, as torux.bands Surfaces: its requests', the stepped-across stage's and
        the closing in's, those not around the magnetic axis (where it is known) never placed."""
        from .bands import pool
        axis = self.crossed[2] if self.crossed is not None else None
        mine, off = self.tori, []
        if axis is not None:
            from .analysis import torus_topology
            ok = []
            for t, w in mine:
                try:
                    good = bool(torus_topology(t, axis)["ok"])
                except Exception:  # noqa: BLE001  (a surface whose topology cannot be checked is not placed)
                    good = False
                (ok if good else off).append((t, w))
            mine = ok
        # the direct requests' surfaces first, so that a lower bound a direct request verified again is named as its
        groups = [([t for t, w in mine if w == "search"], "search"),
                  (self.crossed[0] if self.crossed is not None else [], "stepped across"),
                  (self.widths_tori, "widths")]
        groups += [([t for t, w2 in mine if w2 == w], w) for w in dict.fromkeys(w for _, w in mine if w != "search")]
        groups += [([t for t, w in off], "search", False), (self.widths_off, "widths", False)]
        return pool(*groups)

    def fit_bands(self, chains):
        """Every chain's band held against every surface the search verified (torux.bands.fit): its summary, or
        None where there are no chains."""
        if not chains:
            return None
        from .bands import fit
        t = time.time()
        try:
            out = fit(chains, self.surfaces())
        except Exception as e:  # noqa: BLE001  (the fit costs its own summary, never the answer)
            return dict(error=repr(e), wall_s=round(time.time() - t, 1))
        out["wall_s"] = round(time.time() - t, 1)
        return out

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


def _chains(r):
    """The island chains request ``r`` identified (torux.chainlist), [] when its record holds none."""
    from .chainlist import of_record
    try:
        return of_record(r.record)
    except Exception:  # noqa: BLE001  (a record of an unexpected shape costs the list, never the search)
        return []


def _chain_name(c):
    from fractions import Fraction
    io = c.get("iota")
    return f"iota = {Fraction(int(round(io * c['q'])), c['q'])}" if io is not None else f"period {c['q']}"


LOWER_BOUND = "verified lower bound of "     # (torux.bands Surface.where of a request's verified lower bound)


def _surface_words(x, where):
    """A verified surface at the flux x, in words that say whose it is when it is not a direct request's."""
    if where and where.startswith(LOWER_BOUND):
        who = where[len(LOWER_BOUND):]
        own = "the no-flux request's verified torus" if who == "the no-flux request" else \
            f"the verified torus that the family of {who} reached"
        return f"{own} at {x:.4g} Wb, not verified again by a direct request"
    if where == "widths":
        return f"the verified surface at {x:.4g} Wb (a request of the closing in)"
    return f"the verified surface at {x:.4g} Wb"


def _placed(c):
    """Where a chain lies against the verified surfaces, in words ('' when nothing places it). A surface that is not
    a direct request's of the search is named as what it is (torux.bands placement inner_from, outer_from)."""
    iv = c.get("between_verified_Wb")
    if not iv:
        return ""
    pl = c.get("placement") or {}
    frm = [pl.get(f"{k}_from") if pl.get(f"{k}_Wb") == v else None for k, v in zip(("inner", "outer"), iv)]
    if iv[0] is not None and iv[1] is not None:
        if not any(f and f != "search" and f != "stepped across" for f in frm):
            return f"between verified surfaces at {iv[0]:.4g} and {iv[1]:.4g} Wb"
        return f"between {_surface_words(iv[0], frm[0])} and {_surface_words(iv[1], frm[1])}"
    if iv[0] is not None:
        return f"outside {_surface_words(iv[0], frm[0])}"
    return f"inside {_surface_words(iv[1], frm[1])}"


def _unresolved_words(c):
    """A chain's lines whose residues lie within their errors (torux.chainlist's unresolved), in words ('' if none)."""
    u = c.get("unresolved") or []
    if not u:
        return ""
    what = "no type and no half width" if not (c.get("O") or c.get("X")) else "neither O nor X"
    late = int(c.get("lines_unrefined") or 0)
    late = f" ({late} of them not refined in the stage's time)" if late else ""
    if len(u) == 1:
        return f"1 line whose residue lies within its error{late}, so {what}"
    return f"{len(u)} lines whose residues lie within their errors{late}, so {what}"


def _bands_words(b):
    """What holding every chain's band against every verified surface did (torux.bands), in a line."""
    return (f"bands: of {b['chains']} chains, {b['cut']} estimate{'s' if b['cut'] != 1 else ''} cut by a verified "
            f"surface the chain's crossings place on that side and {b['withdrawn']} withdrawn, against "
            f"{b['surfaces']} verified surfaces (answer.evidence['search']['bands'])")


def _chain_words(chains):
    """The chains a search met, in a line, and the ones the answer's family stepped across in another."""
    crossed = [c for c in chains if "stepped across" in c["found_in"]]
    met = [c for c in chains if "stepped across" not in c["found_in"]]
    out = []
    if met:
        parts = []
        for c in met[:6]:
            name = _chain_name(c)
            if c.get("flux_Wb") is not None:
                name += f" near {c['flux_Wb']:.4g} Wb"
                if c.get("flux_from") == "extrapolated":
                    fam = c.get("family_Wb") or []
                    name += (f" by extrapolation from the family torus at {fam[0]:.4g} Wb" if fam
                             else " by extrapolation")
            if c.get("half_width_Wb") is not None:
                name += f", half width about {c['half_width_Wb']:.1e} Wb"
            elif c.get("band_withdrawn"):
                name += f", no band: {c['band_withdrawn']}"
            elif _unresolved_words(c):
                name += f", {_unresolved_words(c)}"
            where = _placed(c)
            if where:
                name += f", its lines {where}"
            k = c["found_by"]
            req = f"request{'s' if len(k) > 1 else ''} {', '.join(str(x) for x in k)}"
            parts.append(name + f" ({req}: {' and '.join(c['found_in'])})")
        more = len(met) - len(parts)
        out.append("chains: " + " | ".join(parts) + (f" | and {more} more" if more > 0 else "")
                   + " (answer.evidence['chains'])")
    if crossed:
        parts = []
        for c in crossed[:12]:
            name = _chain_name(c)
            if c.get("flux_Wb") is not None:
                name += f" at {c['flux_Wb']:.4g} Wb"
            more_ = [w for w in (_placed(c),
                                 f"half width about {c['half_width_Wb']:.1e} Wb" if c.get("half_width_Wb") is not None
                                 else f"no band: {c['band_withdrawn']}" if c.get("band_withdrawn")
                                 else _unresolved_words(c)) if w]
            if c.get("cut"):
                more_.append("cut short by the budget")
            parts.append(name + (f" ({', '.join(more_)})" if more_ else ""))
        more = len(crossed) - len(parts)
        out.append("stepped across: " + " | ".join(parts) + (f" | and {more} more" if more > 0 else "")
                   + f" (the answer's family, chains of q <= {_qmax()}: answer.evidence['chains'])")
    return out


def _qmax():
    from .crossed import QMAX
    return QMAX


def _verified_torus(r):
    """The surface a request verified, whatever its question: the surface at its flux (status 'surface', cut short
    by its deadline or not), or the outermost torus of its family that the solver and the independent evaluator
    accepted (status 'range', the verified lower bound); None otherwise."""
    if r is None or r.surface is None:
        return None
    if r.status == "surface":
        return r.surface
    rg = (r.record or {}).get("range") or {}
    if r.status == "range" and rg.get("verified") and rg.get("verified_lower_bound") is not None:
        return r.surface
    return None


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
    # what it rests on: its construction and closed lines (q, residues, crossings), for a tracer of one's own
    try:
        from .chainlist import provenance
        pv = provenance(c)
    except Exception:  # noqa: BLE001
        pv = None
    if pv is not None:
        out["construction"] = pv["construction"]
        out["lines"] = pv["lines"]
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
    from_axis = False
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
        if not pts and len(S.failed) >= 4 and not from_axis:
            # four fluxes, each half the last, and no surface verified at any: the family from the magnetic axis,
            # which finds the surfaces however small they are, and its verified lower bound asked directly
            from_axis = True
            r = _size_from_axis(S, src)
            if _terminal(r):
                return S.finish(r, [], brief=True)
            pts = sorted((f, u_of(s)) for f, s in S.size_of.items())
            best = min(S.size_of.items(), key=lambda kv: abs(kv[1][which] - target), default=None)
            if best is not None and abs(best[1][which] - target) <= RTOL * target:
                return _size_found(S, which, target, best[0])
        if not pts and len(S.failed) >= 4:
            break            # four fluxes, each half the last, and the family from the axis: no surface verified
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
    stop = "budget" if S.left() <= MIN_REQUEST_S else (
        f"no verified surface at any of {len(S.failed)} fluxes, nor from the axis" if not S.size_of else
        f"{len(S.log)} requests, the most a size asks" if len(S.log) >= MAX_SIZE_REQUESTS else
        "no flux left to ask that the verified surfaces predict")
    return S.finish(out, words, reached=None, flux=None,
                    closest=None if best is None else dict(flux=best[0], **best[1]), stop=stop)


def _without_request(src):
    """The same coils with no request and no design surfaces of their own (the family from the axis outward)."""
    if (src.request or {}).get("flux_Wb") or getattr(src, "_design_surfaces", None) is not None:
        free = copy.copy(src)
        free.request = None
        free._design_surfaces = None
        return free
    return src


def _size_from_axis(S, src):
    """For a size with no surface verified at four fluxes: the no-flux request (the family from the magnetic axis)
    and its verified lower bound asked directly. The last Result (a terminal one ends the search), or None."""
    rng = S.ask(None, "no flux: the family of surfaces from the magnetic axis outward, after four fluxes with no "
                      "surface verified", budget=min(S.default_budget(None), 0.4 * S.left()), src=_without_request(src))
    if rng is None or _terminal(rng):
        return rng
    f_r = ((rng.record or {}).get("range") or {}).get("verified_lower_bound")
    if not f_r or abs(float(f_r)) in S.size_of or abs(float(f_r)) in S.failed:
        return rng
    del rng
    return S.ask(abs(float(f_r)), "the no-flux request's verified lower bound, asked directly")


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
    stop = {}
    if cert is None and S.left() > MIN_REQUEST_S:
        stop = dict(stop="the request at the flux that size would need was undecided within its own budget")
    return S.finish(r, words, reached=None, flux=F_need, flux_asked=f_asked, **stop)


# --------------------------------------------------------------------------- the largest
def _largest(src, budget_s, opts, widths=None):
    """The largest surface torux can verify for ``src`` by any request it can make within the budget: its own
    request, the family from the magnetic axis (the no-flux request), direct requests at the fluxes that family
    reached and between them, and a request beyond them all for a certificate of what lies past the last surface.
    widths: (seconds, requests at once) for closing in on the answer's chains after it (torux.widths), or None."""
    # (traced) the tori the question finds beyond chains from traced lines (torux.beyond), kept for every request on
    # this copy of the source, and where the requests' families stalled
    src = copy.copy(src)
    src._traced = {}
    S = _Search(src, "largest", None, budget_s, opts)
    S.widths_asked = widths
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
        for _ in range(MAX_BEYOND):
            if any(c.get("no_surface_above_Wb") is not None for _, c, _ in S.certs):
                break
            # beyond everything seen, from the axis, with half the time left (at least the solver's default): what
            # stands past the last surface, a certificate if one can be had. The other half closes the gap below.
            # A surface verified there decides nothing about what lies past it: then twice as far again
            seen = list(S.size_of) + list(S.failed) + cands
            if not seen:
                break
            if S.left() <= 4 * MIN_REQUEST_S:
                raise _OutOfTime
            f_b = 2.0 * max(seen)
            r = S.ask(f_b, "beyond every flux seen, for what stands past the last surface", src=free,
                      budget=max(S.default_budget(f_b), 0.5 * S.left()))
            if r is None:
                raise _OutOfTime
            if _terminal(r):
                raise _Terminal(r)
            more = _family(r.record or {})
            if f_b in S.size_of:
                if free is src:
                    take(f_b)          # the source's own answer at f_b: asking it again would give the same
                else:                  # (asked of the coils alone: a surface there is asked again of the source)
                    S.best.pop(f_b, None)
                    S.size_of.pop(f_b)
                    more.append(f_b)
            del r
            climb(sorted(set(more)))
            if state["lo"] is None or state["lo"] < f_b * (1 - 1e-12):
                break                  # no surface at f_b: what lies between is the gap's to close
        else:
            state["stop"] = (f"every request beyond the family verified, out to {MAX_BEYOND} doublings of the flux, "
                             f"and the search asks no further")
        # (traced) where no certificate bounds what the coils hold and a request's family stalled at a chain its jump
        # could not cross: the question's own stage traces lines beyond that chain for a torus (torux.beyond.cross),
        # kept for every later request, whose families then go on from it (or walk back from it to their flux)
        from . import beyond as _bey
        for _ in range(MAX_CROSS):
            if state["lo"] is None or any(c.get("no_surface_above_Wb") is not None for _, c, _ in S.certs):
                break
            ks = _bey.pending(src._traced, state["lo"])
            if not ks or S.left() < CROSS_MIN_S:
                break
            t_c = time.time()
            _say(f"tracing beyond the chain p/q = {ks[0][0]}/{ks[0][1]} where the families stalled, for a torus beyond "
                 f"it")
            log = _bey.cross(src, src._traced, floor=state["lo"], deadline=min(
                S.deadline - MIN_REQUEST_S, t_c + min(CROSS_STAGE_S, max(CROSS_MIN_S, CROSS_STAGE_SHARE * S.left()))))
            if log is None:
                break
            S.traced.append({k: v for k, v in log.items() if k not in ("probes",)})
            _say(f"traced beyond p/q = {log['p']}/{log['q']}: {log['outcome']} in {time.time() - t_c:.0f} s, "
                 f"{max(0.0, S.left()):.0f} s of the budget left")
        crossed = [t for t in S.traced if t.get("outcome") == "torus"]
        passed = False
        for t in crossed:
            # the traced torus's own flux: the requests' families reach that torus at once (reuse)
            f_t = abs(float(t["torus"]["flux"]))
            if state["lo"] is not None and f_t > state["lo"] * (1 + 1e-12) and f_t not in S.size_of and \
                    f_t not in S.failed and _covering(S, f_t) is None:
                probe(f_t, "the flux of the torus traced beyond the chain")
        # between the largest verified flux and the smallest beyond it that failed or that a certificate rules out,
        # until they are within GAP of each other (the families of requests past a traced chain go on from its torus,
        # or walk back from it to their flux). (traced) When nothing above the largest verified flux is left
        # undecided, a request beyond everything seen gives the family past the chain to climb
        steps = 0
        while steps < 6:
            lo = state["lo"]
            if lo is None:
                break
            up = [f for f in S.failed if f > lo] + [
                c["no_surface_above_Wb"] for _, c, _ in S.certs if c.get("no_surface_above_Wb", 0) > lo]
            top = min(up) if up else None
            if top is None and crossed and not passed:
                passed = True
                seen = list(S.size_of) + list(S.failed) + cands
                if S.left() <= 4 * MIN_REQUEST_S:
                    raise _OutOfTime
                f_b = 2.0 * max(seen)
                r = S.ask(f_b, "beyond every flux seen, the family going on past the chain from the traced torus",
                          src=free, budget=max(S.default_budget(f_b), 0.5 * S.left()))
                if r is None:
                    raise _OutOfTime
                if _terminal(r):
                    raise _Terminal(r)
                more = _family(r.record or {})
                if f_b in S.size_of:
                    if free is src:
                        take(f_b)
                    else:
                        S.best.pop(f_b, None)
                        S.size_of.pop(f_b)
                        more.append(f_b)
                del r
                climb(sorted(set(more)))
                continue
            if top is None or (top - lo) <= GAP * lo:
                break
            f = 0.5 * (lo + top)
            if _covering(S, f) is not None:
                break
            if S.left() < 0.25 * S.default_budget(f):
                raise _OutOfTime       # (a request with a sliver of its budget decides nothing)
            probe(f, "between the largest verified flux and the least one beyond it that failed")
            steps += 1
    except _OutOfTime:
        state["stop"] = "budget"
    except _Terminal as t:
        return S.finish(t.r, [], brief=True)
    return _largest_answer(S, state["lo"], state.get("stop"))


class _OutOfTime(Exception):
    pass


class _Terminal(Exception):
    def __init__(self, r):
        super().__init__(r.status)
        self.r = r


def _largest_answer(S, lo, stop="budget", stages=True):
    """The answer to the largest: ``lo`` the largest verified flux (None: none), ``stop`` why the search asked no
    more: 'budget', or words, or None when it ended with time to spare in the ordinary way. ``stages``: the chains
    the answer's family stepped across and, where asked, the closing in on the chains run after it. When the closing
    in verified a surface beyond the answer's, that request's surface is the answer, with these words again for it."""
    from .api import Result
    words = []
    fields = dict(flux=lo, beyond=None, outward=[])
    if lo is None and stop is None and S.left() > MIN_REQUEST_S:
        stop = "no request verified a surface, and the search had no flux left to ask"
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
        e = getattr(r, "_ask_entry", {})
        for t in e.get("traced") or []:
            if t.get("outcome") == "torus" and t.get("flux_after") is not None:
                words.append(f"crossed: the chain p/q = {t['p']}/{t['q']}, the family going on from the torus at "
                             f"{_wb(abs(float(t['flux_after'])))} traced beyond it, where the jump found none "
                             f"(record['jumps'])")
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
    budget_out = stop == "budget" or S.left() <= MIN_REQUEST_S
    fields["beyond"] = items[0][1] if items else dict(kind="budget") if budget_out else dict(
        kind="not asked", why=stop or "the search ended with nothing beyond the answer asked")
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
    if not items and budget_out:
        words.append("beyond it: nothing decided, the budget ran out first")
    elif not items:
        words.append(f"beyond it: nothing decided, with {max(0.0, S.left()):.0f} s of the budget left: "
                     f"{fields['beyond']['why']}")
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
    fields["stop"] = "budget" if budget_out else stop
    if S.traced:
        fields["traced"] = S.traced
        for t in S.traced:
            got = (f"a torus at {_wb(abs(t['torus']['flux']))}" if t.get("outcome") == "torus" and t.get("torus")
                   else t.get("outcome", "nothing"))
            words.append(f"traced: beyond the chain p/q = {t['p']}/{t['q']} where the families stalled (from "
                         f"{_wb(t['from_flux_Wb'])}): {got}, in {t.get('stage_s', 0):.0f} s "
                         "(answer.evidence['search']['traced'])")
    if ups:
        fields["no_surface_above_Wb"] = ups[0]
    if lo is not None and stages:
        _crossed_stage(S, r)
        if getattr(S, "widths_asked", None) is not None:
            _widths_stage(S, r)
            f = _widths_beyond(S, lo)
            if f is not None:
                return _largest_answer(S, f, stop, stages=False)
    if not stages and getattr(S, "widths_from", None) is not None:
        words.insert(1, f"reached by the closing in on the chains: the search itself had reached "
                        f"{_wb(S.widths_from)}, and a request of the closing in verified this surface beyond it, in "
                        f"the time of widths_s and none past it (answer.evidence['search']['widths'])")
    return S.finish(r, words, **fields)


def _widths_beyond(S, lo):
    """The flux asked of the closing in's request that verified the surface of largest flux, when that surface lies
    beyond the search's answer at ``lo``: entered as a request of the search, its surface verified as the search's
    own (the shipped solver's answer to a direct request at that flux, around the magnetic axis). None otherwise."""
    got = getattr(S, "widths_largest", None)
    if got is None:
        return None
    f, r, wall = got
    if r is None or not verified(r) or abs(float(r.surface.flux)) <= lo * (1 + 1e-12):
        return None
    f = float(f)
    sz = size(r.surface)
    entry = dict(flux=f, purpose="the closing in on the chains: the surface of largest flux it verified, beyond the "
                                 "search's", budget_s=round(float(r.budget_s), 1),
                 wall_s=None if wall is None else round(float(wall), 1), status=r.status, iota=float(r.iota), **sz)
    if not repeatable(r):
        entry["repeatable"] = False
        entry["path"] = (r.record or {}).get("path")
    entry["default_budget"] = bool(r.budget_s >= S.default_budget(f))
    entry["chains"] = [dict(c, request=len(S.log) + 1) for c in _chains(r)]
    S.log.append(entry)
    r._ask_entry = entry
    S.best[f] = r
    S.size_of[f] = sz
    S.widths_from = lo
    return f


def _widths_stage(S, r):
    """The answer's chains closed in on with verified surfaces (torux.widths), in seconds of their own."""
    from . import widths as W
    secs, jobs = S.widths_asked
    t = time.time()
    _say(f"closing in on the answer's chains with verified surfaces: up to {secs:.0f} s")
    try:
        S.widths = W.stage(S, r, secs, jobs)
    except Exception as ex:  # noqa: BLE001  (the closing in costs its brackets, never the answer)
        S.widths = ({}, dict(error=repr(ex), budget_s=secs, wall_s=round(time.time() - t, 1)))
        return
    s = S.widths[1]
    _say(f"closed in on {s['chains']} chain{'s' if s['chains'] != 1 else ''} by {s['requests']} requests in "
         f"{s['wall_s']:.0f} s")


def _crossed_stage(S, r):
    """The chains the answer's family stepped across (torux.crossed), into the answering request's chains, with the
    time the search left, or CROSSED_S (at most CROSSED_SHARE of the budget) from now when that is less, to find
    their lines, and as long again to refine the residues of those lines; the tori it verified place every chain of
    the search."""
    from . import crossed as C
    e = getattr(r, "_ask_entry", None)
    if e is None:
        return
    t = time.time()
    deadline = max(S.deadline, t + min(CROSSED_S, CROSSED_SHARE * S.budget))
    try:
        recs, vt, ax = C.stage(r, S.src, deadline, refine_s=min(CROSSED_S, CROSSED_SHARE * S.budget))
    except Exception as ex:  # noqa: BLE001  (the stage costs its chains, never the answer)
        e["crossed"] = dict(error=repr(ex), wall_s=round(time.time() - t, 1))
        S.after = max(0.0, time.time() - S.deadline)
        return
    if recs is None:
        return
    from .closure import measure_record
    # how well each of their closed lines closes (torux.closure), and where a line was moved to its closed line,
    # whether it still lies between the verified tori either side
    meas = dict(crossed=recs)
    measure_record(r._engine, S.src, meas)
    C.recheck(recs, vt)
    from .chainlist import of_record
    k = next((i + 1 for i, x in enumerate(S.log) if x is e), None)
    got = of_record(dict(nfp=S.src.nfp, crossed=recs))
    e["chains"] = list(e.get("chains") or []) + [dict(c, request=k) for c in got]
    S.after = max(0.0, time.time() - S.deadline)
    e["crossed"] = dict(chains=len(got), cut=sum(1 for c in got if c.get("cut")), qmax=C.QMAX,
                        verified_tori=len(vt), wall_s=round(time.time() - t, 1), after_budget_s=round(S.after, 1),
                        **({"closed_lines": meas["closed_lines"]} if meas.get("closed_lines") else {}))
    S.crossed = (vt, r._engine, ax)
    _say(f"the answer's family stepped across {len(got)} chain{'s' if len(got) != 1 else ''} of q <= {C.QMAX} in "
         f"{e['crossed']['wall_s']:.0f} s")


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
