"""The answer torux.solve returns: what it found, how that was checked, and the numbers a physicist asks for next.

    s = torux.solve("coils.json", flux=0.07)
    print(s)                                   # what it found, how it was checked, or why there is none
    s.iota, s.volume, s.minor_radius           # SI. With no verified surface these raise ValueError
    s.profile                                  # (flux, iota) of the surfaces the solve followed
    s.section(phi=0.0)                         # R, Z of the cross-section at a cylindrical angle
    F, J = s.system(rc, zs, iota)              # torux's equations at any surface, and their Jacobian
    s.vjp(rc, zs, iota, lam)                   # lam^T dF / d(coils, flux)
    s.system_general(rc, rs, zc, zs, iota)     # the same without stellarator symmetry
    s.vjp_general(rc, rs, zc, zs, iota, lam)
    s.save("input.mine")                       # VMEC input (VMEC, VMEC++, DESC, SIMSOPT). Also *.json, *.vtk
    torux.Answer.read("answer.json")           # an answer saved as JSON, back

An Answer is a torux.Result: status, surface, evidence, record, points and field keep their meaning.  Its words and
its numbers say no more than the status does: numbers exist only for a verified surface ('surface', 'range').
The surface's angles are theta, its straight-field-line angle, counter-clockwise in the (R, Z) plane, and phi, the
cylindrical angle: iota has VMEC's sign, the opposite of the one QUASR quotes.  The radii follow VMEC's
and SIMSOPT's definitions: minor radius sqrt(mean cross-sectional area / pi), major radius
volume / (2 pi mean cross-sectional area), the cross-sections taken at constant cylindrical angle.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import textwrap
from fractions import Fraction

import numpy as np

from .api import Request, Result, _plain
from .api import solve as _api_solve

VERIFIED = ("surface", "range")
BAR = 1e-14                      # the solver's acceptance of |B.n|/|B| (torux.analysis.BN_SURFACE)
CERTIFICATE_BUDGET_S = 3300.0    # what a certificate needs (docs/usage.md)
WIDTH = 120                      # columns of the printed answer
FIELD_SLAB = 4096           # points given to the field engine at once
POINT_SLAB = 2048           # points of the surface evaluated at once
VTK_GRID = (256, 1024)      # the most points the VTK grid has in theta and round the torus
AREA_SLAB = 1 << 20          # grid points held at once while the area is summed
LABELS = 23                      # the column of words every printed block of torux starts its values at


def solve(source, flux=None, *, minor_radius=None, volume=None, largest=False, budget_s=None, tol=3e-15, verify=True,
          trace_lines=0, gradients=False) -> Answer:
    """The flux surface of ``source`` asked for by one of ``flux`` (full toroidal flux, Wb), ``minor_radius`` (m),
    ``volume`` (m^3) or ``largest``, verified, or the certificate that none exists, or a truthful 'undetermined'.
    With none of the four: the source's own request, or, when it carries none, its largest surface.  ``source``:
    anything torux.load reads, a torux.Source or a torux.Request. ``budget_s``: seconds for the whole question."""
    from . import ask
    from .source import Source
    if isinstance(source, Request):
        if flux is not None or minor_radius is not None or volume is not None or largest:
            raise ValueError("a torux.Request carries its own flux: give flux, minor_radius, volume or largest with "
                             "the source instead")
        req = source
        if not isinstance(req.source, Source):
            req = dataclasses.replace(req, source=_load(req.source))
        res = _api_solve(req)
    else:
        ask.one_question(flux, minor_radius, volume, largest)
        src = source if isinstance(source, Source) else _load(source)
        req = Request(src, flux, budget_s, tol, verify, trace_lines, gradients)
        res = ask.solve(src, flux, minor_radius=minor_radius, volume=volume, largest=largest, budget_s=budget_s,
                        tol=tol, verify=verify, trace_lines=trace_lines, gradients=gradients)
    search = (res.evidence or {}).get("search")
    if search:              # asked by size or for the largest: no flux was asked (evidence['search'] has the fluxes)
        return Answer._of(res, None)
    asked = req.flux
    if asked is None and req.source.request:
        asked = req.source.request.get("flux_Wb")
    return Answer._of(res, None if asked is None else abs(float(asked)))


def _load(x):
    from .sources import load
    return load(x)


class Answer(Result):
    """What torux.solve returns: the surface asked for and how it was checked, or why there is none, with the
    numbers (SI) and files that follow.  A torux.Result: status, surface, evidence and record keep their meaning.
    ``requested_flux`` is the full toroidal flux asked for (Wb), None when a size, the largest or the range was."""

    requested_flux = None
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    @classmethod
    def _of(cls, res, requested_flux):
        a = cls(**{f.name: getattr(res, f.name) for f in dataclasses.fields(Result)})
        a.requested_flux = requested_flux
        return a

    # ------------------------------------------------------------------ the status
    def _need(self):
        if self.status in VERIFIED and self.surface is not None:
            return
        if self.status.endswith("_unverified"):
            raise ValueError(f"the status is {self.status!r}: the torus found was not verified, so it gives no numbers "
                             f"(answer.surface holds it)")
        raise ValueError(f"no surface: the status is {self.status!r} (print the answer for why, and what to do "
                         f"next)")

    @property
    def iota(self):
        """Rotational transform of the surface: poloidal turns per toroidal turn, signed as VMEC signs it (theta
        counter-clockwise in the (R, Z) plane, phi the cylindrical angle). QUASR quotes the opposite sign."""
        self._need()
        return self.__dict__["_iota"]

    @iota.setter
    def iota(self, value):
        self.__dict__["_iota"] = value

    @property
    def flux(self):
        """Full toroidal flux the surface encloses (Wb, positive)."""
        self._need()
        return abs(float(self.surface.flux))

    @property
    def nfp(self):
        """Field periods of the surface."""
        self._need()
        return _int_nfp(self.surface)

    @property
    def volume(self):
        """Volume the surface encloses (m^3)."""
        return self._geometry()["volume"]

    @property
    def area(self):
        """Area of the surface (m^2)."""
        self._need()
        a = self.__dict__.get("_area")
        if a is None:
            a = self.__dict__["_area"] = _area(self.surface)
        return a

    @property
    def minor_radius(self):
        """sqrt(mean cross-sectional area / pi) (m), as VMEC and SIMSOPT define it."""
        return math.sqrt(self._geometry()["mean_section"] / math.pi)

    @property
    def major_radius(self):
        """volume / (2 pi mean cross-sectional area) (m), as VMEC and SIMSOPT define it."""
        g = self._geometry()
        return g["volume"] / (2 * math.pi * g["mean_section"])

    @property
    def aspect_ratio(self):
        """major radius / minor radius."""
        return self.major_radius / self.minor_radius

    @property
    def profile(self):
        """The surfaces the solve followed, as rows (flux in Wb, rotational transform) in order of flux.  The answer's
        own surface is the verified one. The others passed the solver's continuation check only."""
        return _profile(self)

    def section(self, phi=0.0, n=256):
        """R, Z (m) of the surface's cross-section at the cylindrical angle phi (rad), at n poloidal angles."""
        self._need()
        th = 2 * np.pi * np.arange(int(n)) / int(n)
        _, R, Z = _points(self.surface, th, np.full(int(n), float(phi)))
        return R, Z

    def points(self, theta, phi):
        """Cartesian points (m) of the surface at theta, its straight-field-line angle, counter-clockwise in the
        (R, Z) plane, and phi, the cylindrical angle (rad)."""
        self._need()
        shape = np.broadcast(np.asarray(theta, float), np.asarray(phi, float)).shape
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        return _points(self.surface, theta, phi)[0].reshape(shape + (3,))

    def field(self, x):
        """B (T) at Cartesian points x (..., 3) in metres."""
        if self._engine is None:
            raise ValueError("the field needs the source, which this answer does not hold (it was read from a file "
                             "or settled by proof): solve the source again")
        x = np.asarray(x, float)
        flat = x.reshape(-1, 3)
        # the engine holds every point against every node of a coil at once: many points go through it in slabs
        out = [np.asarray(Result.field(self, flat[i:i + FIELD_SLAB])) for i in range(0, len(flat), FIELD_SLAB)]
        return np.concatenate(out).reshape(x.shape)

    def system(self, rc, zs, iota):
        """The residual F and Jacobian J of torux's equations for a flux surface of the source, at any surface.

        rc, zs (M+1, 2N+1), M >= 1, N >= 0: R = sum rc[m, n+N] cos(m theta - n nfp (phi - phi0)), Z the same with zs
        and sin, phi0 = answer.surface.phi0, and iota its rotational transform.  The unknowns u are a stellarator
        symmetric SurfaceRZFourier's dofs in SIMSOPT's order, then iota: rc[0, n >= 0], rc[m >= 1, every n] row by
        row, zs[0, n >= 1], zs[m >= 1, every n], iota.  rc[0, n < 0] and zs[0, n <= 0] are not unknowns and are
        ignored.

        F (len(u),): the projections of R_phi + iota R_theta - R B_R / B_phi on sin(m theta - n nfp (phi - phi0)),
        indexed like zs in u, then those of Z_phi + iota Z_theta - R B_Z / B_phi on cos(...), indexed like rc (twice
        the mean over a field period, the mean itself for m = n = 0), then the flux row
        (Psi(u) - Psi0) rc[0, N] / |Psi0|, Psi(u) the toroidal flux through the surface and Psi0 =
        answer.surface.flux.  F = 0 on a flux surface of that flux.

        J (len(u), len(u)), dense: dF/du as torux's Newton method takes it.  Its flux row holds the factor
        rc[0, N] / |Psi0| fixed and differentiates the flux as the integral of B_phi over the cross-section, so it
        differs from the exact derivative of F by F[-1] / rc[0, N] in J[-1, 0], zero on a solution, and in the rest of
        the row by the quadrature error of torux's grid: 4e-6 of the row at M = 5, N = 4, below the 1e-10 finite
        differences resolve at M = 12, N = 10.  The other rows match finite differences to their precision."""
        prob, u, flux, scale = self._equations(rc, zs, iota, "answer.system")
        F, J, _, _ = prob.system(u if prob.np_path else _jnp().asarray(u), flux, scale)
        return np.asarray(F, dtype=np.float64), np.asarray(J, dtype=np.float64)

    def vjp(self, rc, zs, iota, lam):
        """lam^T dF/dp for F = answer.system(rc, zs, iota)[0], p the coils and the flux, at fixed rc, zs and iota.

        Returns a dict.  coef (ncoil, 3, 2K+1) and current (ncoil,): the coils, in the order of the source (for a
        SIMSOPT BiotSavart, the order of bs.coils), the x, y and z coefficients of each in CurveXYZFourier's order
        [c0, s1, c1, ..., sK, cK].  d_coef and d_current, of the same shapes: lam^T dF/dcoef and lam^T dF/dcurrent.
        d_flux: lam^T dF/dPsi0, Psi0 = answer.surface.flux.  A coil of zero current is kept: its d_coef is zero, as
        its field is, and its d_current is the derivative of F in its current.  On a surface with F = 0 and
        J^T lam = df/du, the derivative of f(u) in the coils and the flux is minus these.  The source must be made of
        Fourier coils alone, of one order K."""
        prob, u, flux, scale = self._equations(rc, zs, iota, "answer.vjp")
        lam = np.asarray(lam, dtype=np.float64)
        if lam.shape != u.shape:
            raise ValueError(f"lam has shape {lam.shape}: it needs one entry per equation, {u.shape}")
        eng = self._engine
        parts = list(eng.source.parts)
        other = sorted({p.kind for p in parts if p.kind != "fourier"}) + (["analytic"] if eng.analytic else [])
        if other or not parts:
            raise ValueError(f"answer.vjp takes its derivatives in Fourier coils alone, and this source holds "
                             f"{', '.join(other) + ' parts' if other else 'no parts'}")
        if len({p.order for p in parts}) > 1:
            raise ValueError(f"the source holds Fourier coils of orders {sorted({p.order for p in parts})} in separate "
                             f"parts: give them as one part of one order")
        from .gradients import _vjp_source
        out = _vjp_source(eng, prob, u, flux, scale, lam)
        coef = np.concatenate([p.coef for p in parts])
        current = np.concatenate([p.current for p in parts])
        d_coef = -np.concatenate([g["d_coef"] for g in out["parts"]])
        d_current = -np.concatenate([g["d_current"] for g in out["parts"]])
        zero = np.flatnonzero(current == 0.0)
        if zero.size:
            d_current[zero] = _unit_current_vjp(eng, prob, u, flux, scale, lam, coef[zero])
        # the flux row (Psi(u) - Psi0) rc[0, N] / |Psi0| has d/dPsi0 = -rc[0, N] / |Psi0| - sign(Psi0) F[-1] / |Psi0|
        b = prob.basis
        rcu, zsu = (np.zeros(b.mask_c.shape) for _ in range(2))
        rcu.ravel()[b.ic], zsu.ravel()[b.is_] = u[:b.nc], u[b.nc:b.nc + b.ns]
        last = (prob.flux_of(rcu, zsu) - flux) * scale
        d_flux = float(lam[-1] * (-scale - math.copysign(1.0, flux) * last / max(abs(flux), 1e-300)))
        return dict(coef=coef, current=current, d_coef=d_coef, d_current=d_current, d_flux=d_flux)

    def system_general(self, rc, rs, zc, zs, iota):
        """answer.system for a surface without stellarator symmetry: the residual F and Jacobian J of torux's
        equations for a flux surface of the source, at any surface.

        rc, rs, zc, zs (M+1, 2N+1), M >= 1, N >= 0: R = sum rc[m, n+N] cos(m theta - n nfp phi) + rs[m, n+N]
        sin(m theta - n nfp phi), Z the same with zc and zs, phi the cylindrical angle, and iota its rotational
        transform.  The unknowns u are a SurfaceRZFourier's dofs without stellarator symmetry in SIMSOPT's order,
        then iota: rc[0, n >= 0], rc[m >= 1, every n] row by row, then rs, zc and zs the same way except that rs and
        zs start at n >= 1, then iota.  rc[0, n < 0], rs[0, n <= 0], zc[0, n < 0] and zs[0, n <= 0] are not unknowns
        and are ignored.  The answer's surface may have stellarator symmetry or not.

        F (len(u),): G - (Pnu / Pnu[p]) G[p], G the projections of R_phi + iota R_theta - R B_R / B_phi and of
        Z_phi + iota Z_theta - R B_Z / B_phi on cos and sin(m theta - n nfp phi), indexed like rc, rs, zc, zs in u
        (twice the mean over a field period, the mean itself for the cosine at m = n = 0), Pnu those of
        nu = (-Z_theta, R_theta) / (R_theta^2 + Z_theta^2) on the reference, p the pivot, the entry of the largest
        |Pnu|.  At p, which G - (Pnu / Pnu[p]) G[p] leaves zero, F holds the phase condition
        <X - X_ref, X_ref,theta> / |X_ref,theta|_rms (X = (R, Z), the mean over the grid), and last the flux row
        (Psi(u) - Psi0) rc[0, N] / |Psi0| of answer.system.  The reference X_ref is answer.surface cut or padded to
        M, N.  F = 0 on a flux surface of that flux whose theta has the reference's origin: G = lambda Pnu there, a
        multiple of Pnu that is zero in the continuum and the size of the truncation in the projections.

        J (len(u), len(u)), dense: dF/du.  Its flux row is answer.system's (rc[0, N] / |Psi0| held fixed, the flux
        differentiated as the integral of B_phi over the cross-section).  The other rows are the derivative of F."""
        eq, u, flux, scale = self._equations_general(rc, rs, zc, zs, iota, "answer.system_general")
        return eq.system(u, flux, scale)

    def vjp_general(self, rc, rs, zc, zs, iota, lam):
        """lam^T dF/dp for F = answer.system_general(rc, rs, zc, zs, iota)[0], p the coils and the flux, at fixed
        rc, rs, zc, zs and iota.  Returns the dict answer.vjp returns, with its conventions."""
        eq, u, flux, scale = self._equations_general(rc, rs, zc, zs, iota, "answer.vjp_general")
        lam = np.asarray(lam, dtype=np.float64)
        if lam.shape != u.shape:
            raise ValueError(f"lam has shape {lam.shape}: it needs one entry per equation, {u.shape}")
        parts = self._fourier_parts("answer.vjp_general")
        from .gradients import _vjp_source_general
        out = _vjp_source_general(self._engine, eq, u, flux, scale, lam)
        coef = np.concatenate([p.coef for p in parts])
        current = np.concatenate([p.current for p in parts])
        d_coef = -np.concatenate([g["d_coef"] for g in out["parts"]])
        d_current = -np.concatenate([g["d_current"] for g in out["parts"]])
        zero = np.flatnonzero(current == 0.0)
        if zero.size:
            d_current[zero] = _unit_current_vjp_general(self._engine, eq, u, flux, scale, lam, coef[zero])
        # the flux row (Psi(u) - Psi0) rc[0, N] / |Psi0| has d/dPsi0 = -rc[0, N] / |Psi0| - sign(Psi0) F[-1] / |Psi0|
        last = (eq.flux_of(u) - flux) * scale
        d_flux = float(lam[-1] * (-scale - math.copysign(1.0, flux) * last / max(abs(flux), 1e-300)))
        return dict(coef=coef, current=current, d_coef=d_coef, d_current=d_current, d_flux=d_flux)

    def plot(self, path=None, *, dark=False):
        """The surface inside its coils, coloured by |B| in tesla, beside three cross-sections in metres: the
        matplotlib figure, or with ``path`` (.png, .pdf, .svg) the file, whose path is returned. ``dark=True`` for
        a dark page. Green only for a surface the independent check verified: a torus found and not verified is
        drawn in greys, its sections dashed. A warning names any part of the source the picture leaves out."""
        from .plot import plot
        return plot(self, path, dark=dark)

    def boozer(self):
        """|B| on the surface in Boozer angles: its harmonics in tesla and its quasi-symmetry (a
        torux.boozer.Boozer)."""
        from .boozer import boozer
        return boozer(self)

    # ------------------------------------------------------------------ words
    def __str__(self):
        return _words(self)

    __repr__ = __str__

    # ------------------------------------------------------------------ files
    def to_dict(self):
        """Everything as plain Python (JSON): what Result.to_dict gives, the request, the record, the numbers."""
        base = Result(self.status, self.budget_s, self.wall_s, self.surface, self.__dict__.get("_iota", np.nan),
                      self.evidence, self.gradients, self.record).to_dict()
        base["requested_flux"] = self.requested_flux
        base["record"] = _plain(self.record)
        if self.status in VERIFIED and self.surface is not None and _int_nfp(self.surface, strict=False):
            base["numbers"] = dict(flux=self.flux, requested_flux=self.requested_flux, iota=self.iota, nfp=self.nfp,
                                   volume=self.volume, area=self.area, major_radius=self.major_radius,
                                   minor_radius=self.minor_radius, aspect_ratio=self.aspect_ratio,
                                   profile=self.profile.tolist())
            mod_b = self._mod_b(required=False)
            if mod_b is not None:
                base["mod_B_on_vtk_grid"] = mod_b.tolist()
        return _plain(base)          # strict JSON: a number that is not finite is written as null

    def save(self, path, mpol=None, ntor=None):
        """Write the answer. The format from the name: *.json (everything), input.<name> (VMEC input: every Fourier
        coefficient, or cut to mpol, ntor as VMEC counts them. Returns the largest distance in m the cut moved the
        surface), *.vtk (the surface with |B| for ParaView)."""
        fmt = save_format(path)
        if fmt != "vmec" and (mpol is not None or ntor is not None):
            raise ValueError("mpol and ntor cut a VMEC input file: give them only with a name input.<name>")
        if fmt == "json":
            with open(path, "w") as fh:
                json.dump(self.to_dict(), fh, allow_nan=False)
            return None
        self._need()
        _int_nfp(self.surface)
        if fmt == "vmec":
            text, moved = _vmec_input(self, mpol, ntor)
            with open(path, "w") as fh:
                fh.write(text)
            return moved
        with open(path, "w") as fh:
            fh.write(_vtk(self))
        return None

    @classmethod
    def read(cls, path):
        """An answer saved as JSON, read back: its print, its numbers, its points and its files. The field, and
        with it plot and boozer, need the source: solve it again for those."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{os.fspath(path)}: no such file")
        with open(path) as fh:
            try:
                d = json.load(fh)
            except ValueError:
                d = None
        if not isinstance(d, dict) or "status" not in d or "budget_s" not in d:
            raise ValueError(f"{path}: not an answer saved by torux (Answer.save or Result.save)")
        a = cls(d["status"], _float(d["budget_s"]), _float(d.get("wall_s")), _surface_from(d.get("surface")),
                np.nan if d.get("iota") is None else float(d["iota"]), d.get("evidence") or {}, d.get("gradients"),
                d.get("record"))
        a.requested_flux = d.get("requested_flux")
        if d.get("mod_B_on_vtk_grid") is not None:
            a.__dict__["_modB"] = np.asarray(d["mod_B_on_vtk_grid"], float)
        return a

    # ------------------------------------------------------------------ internals
    def _equations(self, rc, zs, iota, what):
        """(problem, u, Psi0, rc[0, N] / |Psi0|) of system and vjp at the surface rc, zs, iota."""
        if self._engine is None:
            raise ValueError(f"{what} needs the source, which this answer does not hold (it was read from a file or "
                             f"settled by proof): solve the source again")
        t = self.surface
        if t is None:
            raise ValueError(f"{what} needs a surface, and this answer holds none: the status is {self.status!r}")
        if hasattr(t, "rs"):
            raise ValueError(f"{what} takes a stellarator symmetric surface, and this answer's surface is not "
                             f"stellarator symmetric: {what}_general takes it")
        rc, zs = np.asarray(rc, dtype=np.float64), np.asarray(zs, dtype=np.float64)
        if rc.ndim != 2 or rc.shape != zs.shape or rc.shape[0] < 2 or rc.shape[1] % 2 != 1:
            raise ValueError(f"rc and zs need one shape (M+1, 2N+1) with M >= 1 and N >= 0: they have {rc.shape} and "
                             f"{zs.shape}")
        iota = float(iota)
        M, N = rc.shape[0] - 1, (rc.shape[1] - 1) // 2
        from .torus import TorusProblem
        prob = TorusProblem(self._engine, t.nfp, M, N, phi0=float(getattr(t, "phi0", 0.0)))
        b = prob.basis
        u = np.concatenate([rc.ravel()[b.ic], zs.ravel()[b.is_], [iota]])
        flux = float(t.flux)
        return prob, u, flux, float(rc[0, N]) / max(abs(flux), 1e-300)

    def _equations_general(self, rc, rs, zc, zs, iota, what):
        """(equations, u, Psi0, rc[0, N] / |Psi0|) of system_general and vjp_general at the surface rc, rs, zc, zs,
        iota (torux.torus_general.GeneralEquations, the reference answer.surface in the series of phi itself)."""
        if self._engine is None:
            raise ValueError(f"{what} needs the source, which this answer does not hold (it was read from a file or "
                             f"settled by proof): solve the source again")
        t = self.surface
        if t is None:
            raise ValueError(f"{what} needs a surface, and this answer holds none: the status is {self.status!r}")
        a = [np.asarray(x, dtype=np.float64) for x in (rc, rs, zc, zs)]
        if a[0].ndim != 2 or any(x.shape != a[0].shape for x in a) or a[0].shape[0] < 2 or a[0].shape[1] % 2 != 1:
            raise ValueError(f"rc, rs, zc and zs need one shape (M+1, 2N+1) with M >= 1 and N >= 0: they have "
                             f"{', '.join(str(x.shape) for x in a)}")
        M, N = a[0].shape[0] - 1, (a[0].shape[1] - 1) // 2
        from .torus_general import GeneralEquations, general_arrays
        eq = GeneralEquations(self._engine, t.nfp, M, N, general_arrays(t))
        u = eq.pack(*a, float(iota))
        flux = float(t.flux)
        return eq, u, flux, float(a[0][0, N]) / max(abs(flux), 1e-300)

    def _fourier_parts(self, what):
        """The source's parts, when they are Fourier coils of one order alone (what the product in the coils
        takes)."""
        parts = list(self._engine.source.parts)
        other = sorted({p.kind for p in parts if p.kind != "fourier"}) + (["analytic"] if self._engine.analytic else [])
        if other or not parts:
            raise ValueError(f"{what} takes its derivatives in Fourier coils alone, and this source holds "
                             f"{', '.join(other) + ' parts' if other else 'no parts'}")
        if len({p.order for p in parts}) > 1:
            raise ValueError(f"the source holds Fourier coils of orders {sorted({p.order for p in parts})} in separate "
                             f"parts: give them as one part of one order")
        return parts

    def _geometry(self):
        self._need()
        g = self.__dict__.get("_geo")
        if g is None:
            g = self.__dict__["_geo"] = _size(self.surface)
        return g

    def _mod_b(self, required=True):
        mb = self.__dict__.get("_modB")
        if mb is None and self._engine is not None:
            x = _vtk_points(self.surface)
            mb = self.__dict__["_modB"] = np.linalg.norm(np.asarray(self.field(x)), axis=-1)
        if mb is None and required:
            raise ValueError("|B| on the surface needs the source: solve the source again to write the VTK file")
        return mb


def _float(v):
    return float("nan") if v is None else float(v)


def _jnp():
    import jax.numpy as jnp
    return jnp


def _unit_current_vjp(engine, prob, u, flux, scale, lam, coefs):
    """lam^T dF/dI of coils of zero current, coefficients coefs (k, 3, 2K+1): F is linearised in the field at the
    points of its grid, and each coil's field there at unit current, built as the engine builds a coil, is the
    direction."""
    import jax
    from .field import FieldEngine
    from .gradients import _grid_x, _residual_from_field
    from .source import FourierCurves, Source
    jnp = _jnp()
    uj = jnp.asarray(u)
    x = np.asarray(_grid_x(prob.c, uj, prob.symmetric), dtype=np.float64)
    o = engine.evaluate(x, grad=False, A=True)
    _, pull = jax.vjp(lambda B, A: _residual_from_field(prob.c, uj, B, A, flux, scale, prob.symmetric),
                      jnp.asarray(o["B"]), jnp.asarray(o["A"]))
    gB, gA = (np.asarray(g) for g in pull(jnp.asarray(lam)))
    out = []
    for c in coefs:
        one = FieldEngine(Source("unit current", [FourierCurves(c[None], np.ones(1))]), dmin=engine.dmin)
        q = one.evaluate(x, grad=False, A=True)
        out.append(float(np.sum(gB * np.asarray(q["B"])) + np.sum(gA * np.asarray(q["A"]))))
    return np.array(out)


def _unit_current_vjp_general(engine, eq, u, flux, scale, lam, coefs):
    """_unit_current_vjp for the general equations (torux.torus_general.GeneralEquations)."""
    from .field import FieldEngine
    from .gradients import _field_cotangents_general
    from .source import FourierCurves, Source
    x = np.ascontiguousarray(eq.points_of(u), dtype=np.float64)
    o = engine.evaluate(x, grad=False, A=True)
    gB, gA, _ = _field_cotangents_general(eq, u, np.asarray(o["B"]), np.asarray(o["A"]), flux, scale,
                                          eq.cotangent(lam))
    out = []
    for c in coefs:
        one = FieldEngine(Source("unit current", [FourierCurves(c[None], np.ones(1))]), dmin=engine.dmin)
        q = one.evaluate(x, grad=False, A=True)
        out.append(float(np.sum(gB * np.asarray(q["B"])) + np.sum(gA * np.asarray(q["A"]))))
    return np.array(out)


def save_format(path):
    """'json', 'vmec' or 'vtk' from a file name. ValueError for any other."""
    name = os.path.basename(os.fspath(path))
    low = name.lower()
    if low.endswith(".json"):
        return "json"
    if low.endswith(".vtk"):
        return "vtk"
    if name.startswith("input.") and len(name) > len("input."):
        return "vmec"
    raise ValueError(f"{name}: the name gives no format: use *.json, input.<name> (VMEC) or *.vtk")


# ====================================================================== the surface
def _arrays(t):
    """rc, rs, zc, zs of R = sum rc cos(m th - n psi) + rs sin(...), Z = sum zc cos(...) + zs sin(...),
    psi = nfp (phi - phi0)."""
    rc, zs = np.asarray(t.rc, float), np.asarray(t.zs, float)
    rs = np.asarray(t.rs, float) if hasattr(t, "rs") else np.zeros_like(rc)
    zc = np.asarray(t.zc, float) if hasattr(t, "zc") else np.zeros_like(rc)
    return rc, rs, zc, zs


def _int_nfp(t, strict=True):
    v = float(t.nfp)
    if v.is_integer() and v >= 1:
        return int(v)
    if strict:
        raise ValueError(f"the surface has {v} field periods: it does not close once round the machine")
    return None


def _points(t, theta, phi):
    """Cartesian points, R and Z of the torus at theta and phi, in slabs: the solver's evaluation holds every point
    against every poloidal and every toroidal mode, which for a list of many points on a surface of high resolution
    is gigabytes."""
    if hasattr(t, "rs"):
        from .torus_general import torus_points
    else:
        from .verify import torus_points
    theta, phi = np.asarray(theta, float).ravel(), np.asarray(phi, float).ravel()
    if len(theta) <= POINT_SLAB:
        x, R, Z = torus_points(t, theta, phi)[:3]
        return np.asarray(x), np.asarray(R), np.asarray(Z)
    parts = [torus_points(t, theta[i:i + POINT_SLAB], phi[i:i + POINT_SLAB])[:3]
             for i in range(0, len(theta), POINT_SLAB)]
    return tuple(np.concatenate([np.asarray(q[k]) for q in parts]) for k in range(3))


def _synth(c, s, th, ps):
    """sum c cos(m th - n ps) + s sin(m th - n ps) on the tensor grid th (nt,) x ps (np,)."""
    M, N = c.shape[0] - 1, (c.shape[1] - 1) // 2
    at = np.outer(th, np.arange(M + 1))
    ap = np.outer(ps, np.arange(-N, N + 1))
    Ct, St, Cp, Sp = np.cos(at), np.sin(at), np.cos(ap), np.sin(ap)
    return Ct @ (c @ Cp.T - s @ Sp.T) + St @ (c @ Sp.T + s @ Cp.T)


def _d_theta(c, s):
    m = np.arange(c.shape[0])[:, None]
    return m * s, -m * c


def _d_psi(c, s):
    N = (c.shape[1] - 1) // 2
    n = np.arange(-N, N + 1)[None, :]
    return -n * s, n * c


def _grid(t, nth, nps):
    rc, rs, zc, zs = _arrays(t)
    th = 2 * np.pi * np.arange(nth) / nth
    ps = 2 * np.pi * np.arange(nps) / nps
    R, Z = _synth(rc, rs, th, ps), _synth(zc, zs, th, ps)
    Rt, Zt = _synth(*_d_theta(rc, rs), th, ps), _synth(*_d_theta(zc, zs), th, ps)
    Rp, Zp = _synth(*_d_psi(rc, rs), th, ps), _synth(*_d_psi(zc, zs), th, ps)
    return R, Z, Rt, Zt, Rp, Zp


def _size(t):
    """Volume and mean cross-sectional area at constant cylindrical angle (m^3, m^2): the integrands are
    trigonometric polynomials of degree 3M, 3N and the grid holds them exactly.  The one measure of a surface's size
    in torux: the Answer's numbers and the search by size (torux.ask) both read it."""
    M, N = int(t.M), int(t.N)
    R, Z, Rt, Zt, Rp, Zp = _grid(t, 4 * M + 4, 4 * N + 4)
    volume = abs(4 * np.pi ** 2 * np.mean(0.5 * R * R * Zt))
    section = abs(2 * np.pi * np.mean(R * Zt))
    return dict(volume=float(volume), mean_section=float(section))


def _area(t):
    """The area of the surface (m^2).  Its integrand is analytic, so the trapezoid rule converges geometrically: the
    sum on the grid that holds the volume exactly and the sum on one twice as fine in each angle agree to rounding
    for a resolved surface, and the grid is doubled again, twice at most, only while they do not.  Each grid is
    summed in slabs of theta, so that a surface of any resolution needs the memory of one slab."""
    rc, rs, zc, zs = _arrays(t)
    nfp = _int_nfp(t)
    M, N = int(t.M), int(t.N)

    def area(nth, nps):
        ps = 2 * np.pi * np.arange(nps) / nps
        rows = max(1, AREA_SLAB // nps)
        sums = []
        for i in range(0, nth, rows):
            th = 2 * np.pi * np.arange(i, min(nth, i + rows)) / nth
            R = _synth(rc, rs, th, ps)
            Rt, Zt = _synth(*_d_theta(rc, rs), th, ps), _synth(*_d_theta(zc, zs), th, ps)
            Rf, Zf = nfp * _synth(*_d_psi(rc, rs), th, ps), nfp * _synth(*_d_psi(zc, zs), th, ps)   # d/dphi
            sums.append(float(np.sum(np.sqrt(R * R * (Rt * Rt + Zt * Zt) + (Zt * Rf - Rt * Zf) ** 2))))
        return 4 * np.pi ** 2 * math.fsum(sums) / (nth * nps)
    nth, nps = 4 * M + 4, 4 * N + 4
    a = area(nth, nps)
    for _ in range(3):
        nth, nps = 2 * nth, 2 * nps
        a, before = area(nth, nps), a
        if abs(a - before) <= 1e-13 * a:
            break
    return float(a)


def _profile(a):
    rows = []
    for p in ((a.record or {}).get("profile") or []):
        if isinstance(p, dict) and p.get("converged") and p.get("flux") is not None and p.get("iota") is not None:
            rows.append((abs(float(p["flux"])), float(p["iota"])))
    if a.status in VERIFIED and a.surface is not None:
        f, i = abs(float(a.surface.flux)), float(a.__dict__["_iota"])
        rows = [r for r in rows if abs(r[0] - f) > 1e-12 * f] + [(f, i)]
    rows.sort(key=lambda r: r[0])
    return np.array(rows, float).reshape(-1, 2)


def _surface_from(d):
    if not d:
        return None
    kw = dict(phi0=float(d.get("phi0", 0.0)))
    args = (float(d["iota"]), float(d["flux_Wb"]), int(d["M"]), int(d["N"]), int(d["nfp"]))
    if "rs" in d or "zc" in d:
        from .torus_general import GeneralTorus
        return GeneralTorus(*(np.asarray(d[k], float) for k in ("rc", "rs", "zc", "zs")), *args, **kw)
    from .torus import TorusState
    return TorusState(np.asarray(d["rc"], float), np.asarray(d["zs"], float), *args, **kw)


# ====================================================================== VMEC input
def _vmec_basis(t):
    """The coefficients in VMEC's basis cos / sin(m theta - n nfp phi) (phi0 moved into them), m = 0 rows folded onto
    n >= 0: (rbc, rbs, zbc, zbs), each (M + 1, 2N + 1)."""
    rc, rs, zc, zs = (a.copy() for a in _arrays(t))
    N = int(t.N)
    nfp = _int_nfp(t)
    phi0 = float(getattr(t, "phi0", 0.0))
    if phi0 != 0.0:
        n = np.arange(-N, N + 1)[None, :]
        k = nfp * phi0 / np.pi
        if abs(k - round(k)) <= 1e-12 * max(1.0, abs(k)):
            # a symmetry plane at a multiple of pi / nfp: cos(n nfp phi0) = +-1, sin = 0 (exact sign changes)
            sg = np.where((n * int(round(k))) % 2 == 0, 1.0, -1.0)
            rc, rs, zc, zs = rc * sg, rs * sg, zc * sg, zs * sg
        else:
            # cos(A + d) = cos A cos d - sin A sin d, sin(A + d) = sin A cos d + cos A sin d, d = n nfp phi0
            cd, sd = np.cos(n * nfp * phi0), np.sin(n * nfp * phi0)
            rc, rs = rc * cd + rs * sd, rs * cd - rc * sd
            zc, zs = zc * cd + zs * sd, zs * cd - zc * sd
    # m = 0: cos(-n psi) = cos(n psi), sin(-n psi) = -sin(n psi)
    for j in range(N):
        jp = 2 * N - j
        if rc[0, j] or rs[0, j] or zc[0, j] or zs[0, j]:
            rc[0, jp] += rc[0, j]
            zc[0, jp] += zc[0, j]
            rs[0, jp] -= rs[0, j]
            zs[0, jp] -= zs[0, j]
            rc[0, j] = rs[0, j] = zc[0, j] = zs[0, j] = 0.0
    rs[0, N] = zs[0, N] = 0.0
    return rc, rs, zc, zs


def _max_displacement(dR_c, dR_s, dZ_c, dZ_s):
    """The largest |(dR, dZ)| over the torus of the dropped series: a grid four times finer than its modes, then
    a local maximisation from the largest samples."""
    if not (np.any(dR_c) or np.any(dR_s) or np.any(dZ_c) or np.any(dZ_s)):
        return 0.0
    from scipy.optimize import minimize
    M, N = dR_c.shape[0] - 1, (dR_c.shape[1] - 1) // 2
    nth, nps = max(64, 8 * M + 8), max(64, 8 * N + 8)
    th = 2 * np.pi * np.arange(nth) / nth
    ps = 2 * np.pi * np.arange(nps) / nps
    f = _synth(dR_c, dR_s, th, ps) ** 2 + _synth(dZ_c, dZ_s, th, ps) ** 2
    best = float(f.max())
    m = np.arange(M + 1)[:, None]
    n = np.arange(-N, N + 1)[None, :]

    def neg(x):
        A = m * x[0] - n * x[1]
        C, S = np.cos(A), np.sin(A)
        r = np.sum(dR_c * C + dR_s * S)
        z = np.sum(dZ_c * C + dZ_s * S)
        rA = -dR_c * S + dR_s * C              # d/dA
        zA = -dZ_c * S + dZ_s * C
        g = np.array([2 * (r * np.sum(m * rA) + z * np.sum(m * zA)), 2 * (r * np.sum(-n * rA) + z * np.sum(-n * zA))])
        return -(r * r + z * z), -g
    order = np.argsort(f, axis=None)[::-1]
    starts = []
    for idx in order:
        i, j = np.unravel_index(idx, f.shape)
        if all(abs(i - a) + abs(j - b) > 2 for a, b in starts):
            starts.append((i, j))
        if len(starts) == 12:
            break
    for i, j in starts:
        r = minimize(neg, np.array([th[i], ps[j]]), jac=True, method="BFGS", options=dict(gtol=1e-14, maxiter=200))
        best = max(best, -float(r.fun))
    return math.sqrt(best)


def _vmec_input(a, mpol=None, ntor=None):
    t = a.surface
    M, N = int(t.M), int(t.N)
    for name, v, lo in (("mpol", mpol, 1), ("ntor", ntor, 0)):
        if v is not None and (int(v) != v or v < lo):
            raise ValueError(f"{name} must be a whole number of at least {lo}")
    cut = mpol is not None or ntor is not None
    MP = M + 1 if mpol is None else int(mpol)
    NT = N if ntor is None else int(ntor)
    rbc, rbs, zbc, zbs = _vmec_basis(t)
    m = np.arange(M + 1)[:, None]
    n = np.arange(-N, N + 1)[None, :]
    keep = (m < MP) & (np.abs(n) <= NT)
    moved = _max_displacement(*(np.where(keep, 0.0, x) for x in (rbc, rbs, zbc, zbs))) if cut else 0.0
    lasym = bool(np.any(rbs[keep]) or np.any(zbc[keep]))
    if cut:
        head = f"! cut to MPOL = {MP}, NTOR = {NT}: no point of the surface moved by more than {moved:.3e} m"
    else:
        head = "! every Fourier coefficient of the surface: no point moved"
    lines = [head,
             f"! the flux surface at {a.flux!r} Wb, rotational transform {a.iota!r} (status {a.status!r}). Its theta "
             f"is the straight-field-line angle, counter-clockwise in the (R, Z) plane, and phi the cylindrical angle, "
             f"so the transform has VMEC's sign (QUASR quotes the opposite)",
             "&INDATA",
             "  LFREEB = F",
             f"  LASYM = {'T' if lasym else 'F'}",
             f"  NFP = {_int_nfp(t)}",
             f"  MPOL = {MP}",
             f"  NTOR = {NT}",
             "  NS_ARRAY = 16 51",
             "  NITER_ARRAY = 5000 20000",
             "  FTOL_ARRAY = 1.0e-11 1.0e-14",
             "  DELT = 0.9",
             "  NSTEP = 200",
             f"  PHIEDGE = {a.flux!r}",
             "  GAMMA = 0.0",
             "  PMASS_TYPE = 'power_series'",
             "  AM = 0.0",
             "  PRES_SCALE = 0.0",
             "  NCURR = 1",
             "  PCURR_TYPE = 'power_series'",
             "  AC = 0.0",
             "  CURTOR = 0.0"]
    for mm in range(min(M + 1, MP)):
        for nn in range(-min(N, NT), min(N, NT) + 1):
            j = nn + N
            vals = [("RBC", rbc[mm, j]), ("ZBS", zbs[mm, j])]
            if lasym:
                vals += [("RBS", rbs[mm, j]), ("ZBC", zbc[mm, j])]
            if not any(v for _, v in vals):
                continue
            lines.append("  " + "  ".join(f"{k}({nn},{mm}) = {float(v)!r}" for k, v in vals))
    lines.append("/")
    return "\n".join(lines) + "\n", moved


# ====================================================================== VTK
def _vtk_shape(t):
    """Points of the VTK grid in theta and round the whole torus: twice the surface's modes, VTK_GRID at most."""
    M, N = int(t.M), int(t.N)
    return min(max(64, 2 * M + 2), VTK_GRID[0]), min(_int_nfp(t) * max(32, 2 * N + 2), VTK_GRID[1])


def _vtk_points(t):
    """Cartesian points of the VTK grid, theta fastest: summed on the tensor grid of the two angles, so that a
    surface of any resolution takes the memory of the grid alone."""
    nth, nph = _vtk_shape(t)
    th = 2 * np.pi * np.arange(nth) / nth
    ph = 2 * np.pi * np.arange(nph) / nph
    rc, rs, zc, zs = _arrays(t)
    ps = _int_nfp(t) * (ph - float(getattr(t, "phi0", 0.0)))
    R, Z = _synth(rc, rs, th, ps), _synth(zc, zs, th, ps)                 # (nth, nph)
    x = np.stack([R * np.cos(ph)[None, :], R * np.sin(ph)[None, :], Z], -1)
    return np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(-1, 3)      # (nph, nth): theta fastest


def _vtk(a):
    t = a.surface
    nth, nph = _vtk_shape(t)
    x = _vtk_points(t)
    mod_b = a._mod_b()
    out = ["# vtk DataFile Version 4.2",
           f"flux surface at {a.flux:.12g} Wb, |B| in T",
           "ASCII", "DATASET UNSTRUCTURED_GRID", f"POINTS {len(x)} double"]
    out += [f"{p[0]!r} {p[1]!r} {p[2]!r}" for p in x.tolist()]
    nc = nth * nph
    out.append(f"CELLS {nc} {5 * nc}")
    for k in range(nph):
        k2 = (k + 1) % nph
        for j in range(nth):
            j2 = (j + 1) % nth
            out.append(f"4 {k * nth + j} {k * nth + j2} {k2 * nth + j2} {k2 * nth + j}")
    out.append(f"CELL_TYPES {nc}")
    out += ["9"] * nc
    out += [f"POINT_DATA {len(x)}", "SCALARS modB double 1", "LOOKUP_TABLE default"]
    out += [repr(v) for v in np.asarray(mod_b, float).tolist()]
    return "\n".join(out) + "\n"


# ====================================================================== words
def _words(a):
    ev = _plain(a.evidence or {}) or {}
    rec = _plain(a.record or {}) or {}
    st = a.status
    if st in VERIFIED and a.surface is not None:
        rows = _surface_rows(a, ev, rec)
    elif st == "no_surface":
        rows = _no_surface_rows(a, ev, rec)
    else:
        rows = _open_rows(a, ev, rec)
    if ev.get("search"):
        # what was asked and reached under the status. How the search went before the time
        from .ask import search_rows
        sr = search_rows(a)
        head = [r for r in sr if r[0] in ("asked", "reached")]
        rows = rows[:1] + head + rows[1:] + [r for r in sr if r not in head]
    rows.append(("time", _s(a.wall_s)))
    return _table(rows)


def _table(rows):
    """Rows (label, text) as torux prints them: the labels in one column, so that stacked blocks line up, and the
    text wrapped under itself."""
    w = max([LABELS] + [len(k) + 3 for k, _ in rows])
    out = []
    for k, v in rows:
        # a number and its unit stay on one line
        v = re.sub(r"(\d) (s|Wb|m|m\^2|m\^3|T)\b", "\\1\u00a0\\2", str(v))
        parts = textwrap.wrap(v, max(40, WIDTH - w), break_on_hyphens=False) or [""]
        out.append(f"{k:<{w}}{parts[0]}".replace("\u00a0", " "))
        out += [(" " * w + p).replace("\u00a0", " ") for p in parts[1:]]
    return "\n".join(out)


def _s(t):
    """Seconds as the print gives them."""
    return f"{t:.1f} s" if t < 100 else f"{t:.0f} s"


def _e(x, d=1):
    return "nan" if x is None else f"{x:.{d}e}"


def _surface_rows(a, ev, rec):
    ind = ev.get("independent") or {}
    sol = ev.get("solver") or {}
    checked = bool(ind.get("ok"))
    if a.status == "surface":
        head = ("verified: the solver's check and an independent evaluator agree" if checked else
                "accepted by the solver's own checks: the independent check was not run")
    else:
        head = (f"the outermost surface verified: these coils hold at least {a.flux:.6g} Wb" if checked else
                f"the outermost surface the solver accepted (the independent check was not run): these coils hold "
                f"at least {a.flux:.6g} Wb")
    s = a.surface
    rows = [(a.status, head),
            ("toroidal flux", f"{a.flux:.12g} Wb"),
            ("rotational transform", f"{a.iota:.12g}"),
            ("radii", f"major {a.major_radius:.9g} m, minor {a.minor_radius:.9g} m, aspect ratio "
                      f"{a.aspect_ratio:.6g}"),
            ("volume", f"{a.volume:.9g} m^3"),
            ("Fourier series", f"M, N = {s.M}, {s.N}, with {a.nfp} field period{'s' if a.nfp != 1 else ''}")]
    if sol.get("bn_off") is not None:
        rows.append(("field across it", f"{_e(sol['bn_off'])} of |B| (the solver's own check)"))
    chk = ind.get("bn_sup", ind.get("bn_max"))
    if chk is not None:
        rows.append(("independent check", f"{_e(chk)} of |B| across it, and the same flux to within "
                                          f"{_e(ind.get('flux_rel_err'), 0)}"))
    if ind.get("trace_turns"):
        d = ind.get("trace_dist_max_rel")
        rows.append(("field lines", f"traced for {ind['trace_turns']} turns, they stay within {_e(d)} of the minor "
                                    f"radius" if d is not None and math.isfinite(d) else
                     f"traced for {ind['trace_turns']} turns, a line left the surface"))
    g = a.gradients
    if g is not None:
        if isinstance(g, dict) and g.get("skipped"):
            rows.append(("gradients", f"skipped: {g['skipped']}"))
        elif isinstance(g, dict) and g.get("error"):
            rows.append(("gradients", f"failed: {g['error']}"))
        elif isinstance(g, dict) and g.get("per_coil"):
            rows.append(("gradients", f"of iota in the flux. Per coil {g['per_coil']}. The one-period solve's: "
                                      f"answer.gradients['one_period']"))
        else:
            rows.append(("gradients", "of iota in every source parameter and the flux: answer.gradients"))
    if a.status == "range":
        rows += _range_rows(a, rec)
    return rows


def _iota_frac(p, q, nfp):
    try:
        return str(Fraction(int(p) * int(nfp), int(q)))
    except (TypeError, ValueError, ZeroDivisionError):
        return "?"


def _range_rows(a, rec):
    rows = []
    prof = a.profile
    if len(prof):
        rows.append(("followed", f"out to {prof[-1, 0]:.3g} Wb, {len(prof)} surface{'s' if len(prof) != 1 else ''}, "
                                 f"iota {prof[0, 1]:.6g} to {prof[-1, 1]:.6g} (the others: the solver's check only)"))
    nfp = rec.get("nfp") or a.nfp
    why = []
    stop = rec.get("family_stop")
    fmax = float(prof[-1, 0]) if len(prof) else a.flux
    chains = rec.get("chains") or []
    failed = [j for j in (rec.get("jumps") or []) if not j.get("ok")]
    if stop == "stalled" and failed:
        j = failed[-1]
        near = next((c.get("flux_r") for c in chains[::-1] if c.get("q") == j.get("q")), None)
        why.append(f"stalled at {fmax:.3g} Wb, short of the iota = {_iota_frac(j.get('p'), j.get('q'), nfp)} island "
                   f"chain" + (f" (near {near:.3g} Wb)" if near is not None else "") + ", which it could not jump")
    elif stop == "stalled" and chains:
        c = chains[-1]
        why.append(f"stalled at {fmax:.3g} Wb, near the iota = {_iota_frac(c.get('p'), c.get('q'), nfp)} island "
                   f"chain")
    elif stop == "stalled":
        why.append(f"stalled at {fmax:.3g} Wb: no torus further out converged")
    elif stop == "deadline":
        why.append(f"the budget ran out at {fmax:.3g} Wb, while the family was still being followed")
    elif stop:
        why.append(f"the family stopped at {fmax:.3g} Wb ({stop})")
    for t in ((rec.get("range") or {}).get("tried") or []):
        if t.get("ok") or t.get("flux") is None or t["flux"] <= a.flux * (1 + 1e-12):
            continue
        nc = t.get("near_coil")
        if isinstance(nc, dict) and not nc.get("ok") and t.get("bn_off") is None:
            why.append(f"the torus at {t['flux']:.3g} Wb lies too near a coil to resolve")
        elif t.get("guard") is False:
            why.append(f"the torus at {t['flux']:.3g} Wb lies too near a resonance (the resonance guard refused it)")
        elif t.get("bn_off") is not None:
            why.append(f"the torus at {t['flux']:.3g} Wb refined only to {_e(t['bn_off'])} of |B| (M, N = "
                       f"{t.get('M')}, {t.get('N')})")
    if why:
        rows.append(("beyond", "not verified: " + ", and ".join(why[:3])))
    return rows


# ---------------------------------------------------------------------- no surface
ARGUMENTS = {
    "ANTISYM": ("the currents are antisymmetric under reflection in a meridional plane, so B has no toroidal part "
                "there and no field line goes round the machine", True),
    "PLANAR": ("every coil lies in one horizontal plane, so B is vertical on it and no current links a surface", True),
    "NOTOR": ("B has no toroidal part, so no field line goes round the machine", True),
    "AXI": ("axisymmetric vacuum: the flux function has no extremum away from the conductors, so every closed "
            "surface encloses a conductor", True),
    "DIPOLE": ("the closed field lines lie on nested tori round the floating coil, enclosing no toroidal flux", True),
    "PERTURB": ("the coils are axisymmetric up to a deviation too small to open a surface away from the conductors",
                True),
    "LINKBOX": ("every field line from the boxes round the coils reaches both signs of the toroidal field, which no "
                "line on a surface does", True),
    "SIGNTOP": ("the sign of the toroidal field, proved on a grid round the coils, leaves no loop for a surface to "
                "wind round", True),
    "SYM": ("two mirror planes close every field line after one turn: degenerate surfaces with iota = 0", True),
    "AMP": ("Ampere's law: a surface needs current linked through its hole", True),
    "SHADOW": ("no loop clear of the coils' shadows links any current", True),
    "INDEX": ("every closed field line of one turn found is a saddle, which no surface encloses (numerical search)",
              False),
    "EXC": ("no field line traced from a grid of the section completes one turn (numerical)", False),
    "FLUXBOUND": ("a certified bound on the flux any surface can enclose", True),
}
DECISIVE = ("LINKBOX", "SIGNTOP", "ANTISYM", "PLANAR", "AXI", "PERTURB", "NOTOR", "DIPOLE")
VERDICTS = {
    "NO_TOROIDAL_FLUX_SURFACE": "proved: these coils make no surface that encloses toroidal flux, at any flux",
    "NO_VACUUM_INTERIOR_SURFACE": "proved: no surface with vacuum inside encloses toroidal flux: the only ones "
                                  "enclose a conductor",
    "DIPOLE_CLOSED_FIELD_LINES": "proved: the field lines close round the ring, and no surface encloses toroidal "
                                 "flux",
}


def _phrases(tags):
    return ", and ".join(ARGUMENTS[t][0] if t in ARGUMENTS else t.lower().replace("_", " ") for t in tags)


def _asked(a, rows):
    if a.requested_flux is not None and not (a.evidence or {}).get("search"):
        rows.append(("requested flux", f"{a.requested_flux:.9g} Wb"))


def _no_surface_rows(a, ev, rec):
    c = ev.get("certificate") or {}
    p = ev.get("proof") or {}
    nfp = rec.get("nfp") or 1
    kind = str(c.get("kind", ""))
    rows = []
    if kind.startswith("island_band"):
        rows.append(("no surface", "certified: an island chain fills the flux asked for"))
        _asked(a, rows)
        lo, hi = (c.get("interval") or [None, None])[:2]
        if lo is not None:
            rows.append(("none between", f"{lo:.9g} and {hi:.9g} Wb"))
        ch = c.get("chain") or {}
        q, pp = ch.get("q", c.get("q")), ch.get("p", c.get("p"))
        if q:
            rows.append(("island chain", (f"at iota = {_iota_frac(pp, q, nfp)}, " if pp is not None else "")
                         + f"{q} islands in each cross-section"))
        rows.append(("rests on", "an exact theorem (stellarator symmetry) and curves traced with a measured error "
                                 "budget, not interval arithmetic"))
    elif kind.startswith("edge_barrier"):
        rows.append(("no surface", "certified: the flux asked for lies beyond the last surface these coils make"))
        _asked(a, rows)
        if c.get("F_up") is not None:
            rows.append(("upper bound", f"{c['F_up']:.9g} Wb: no surface encloses more (the largest may enclose "
                                        f"less: ask largest=True)"))
        q = (c.get("orbit") or {}).get("q")
        rows.append(("rests on", (f"curves no surface crosses, from an X line of period {q}, with " if q else "")
                     + "some error terms strict bounds and some validated models: the weakest kind of certificate "
                       "torux gives"))
    elif p:
        v = str(p.get("verdict", ""))
        rows.append(("no surface", VERDICTS.get(v, "proved (" + v.replace("_", " ").lower() + ")")))
        _asked(a, rows)
        args = [t for t in (p.get("arguments") or []) if isinstance(t, str)]
        if p.get("kind") == "axisymmetric":
            main, more = args, []
        else:
            main = [t for t in args if t in DECISIVE] or args
            more = [t for t in args if t not in main]
        if main:
            rows.append(("because", _phrases(main)))
        if more:
            rows.append(("also", _phrases(more)))
    else:
        rows.append(("no surface", "certified"))
        _asked(a, rows)
        if c.get("F_up") is not None:
            rows.append(("upper bound", f"{c['F_up']:.9g} Wb"))
        elif c.get("interval"):
            rows.append(("none between", f"{c['interval'][0]:.9g} and {c['interval'][1]:.9g} Wb"))
    return rows


# ---------------------------------------------------------------------- everything else
STAGES = {"proxy_band": "an island band round the chain at the request", "seeds": "other starting tori",
          "seeds_trace": "tori from traced field lines", "cont_probe": "following the surfaces out from the axis",
          "edge_triage": "a test for the edge of confinement"}


def _found(a, rec):
    nfp = rec.get("nfp") or 1
    out = []
    axes = [x for x in (rec.get("axes") or []) if isinstance(x, dict) and x.get("residue") is not None]
    ell = [x for x in axes if 0 < x["residue"] < 1]
    if ell:
        io = ell[0].get("iota")
        out.append("the magnetic axis" + (f" (iota {io:.6g})" if io is not None else ""))
    elif axes:
        out.append(f"{len(axes)} closed field line{'s' if len(axes) != 1 else ''} of one turn, none elliptic")
    prof = _profile(a)
    if len(prof):
        out.append(f"surfaces out to {prof[-1, 0]:.6g} Wb (the solver's check only)")
    for c in (rec.get("chains") or [])[-1:]:
        out.append(f"the iota = {_iota_frac(c.get('p'), c.get('q'), nfp)} island chain"
                   + (f" near {c['flux_r']:.3g} Wb" if c.get("flux_r") is not None else ""))
    return out


def _tried(rec):
    out = []
    if rec.get("resonant_path") == "design Newton failed":
        out.append("Newton from the file's own surface (it did not converge)")
    if any(not j.get("ok") for j in (rec.get("jumps") or [])):
        out.append("jumping the island chain (it failed)")
    for s in ((rec.get("policy") or {}).get("stages") or []):
        ph = STAGES.get(s.get("stage"))
        if ph and not s.get("skipped") and s.get("attempts", 1) and ph not in out:
            out.append(ph)
    b = rec.get("bracket") or {}
    if b.get("log") or b.get("T1") or b.get("T2"):
        out.append("tori either side of the request")
    if rec.get("edge_barrier") is not None or rec.get("bracket_edge_barrier") is not None:
        out.append("an edge barrier, which did not hold the request")
    if rec.get("certificate") is not None or rec.get("bracket_certificate") is not None:
        out.append("an island band, which did not hold the request")
    return out


def _budget_bound(a, ev, rec):
    reason = str(rec.get("deferred_reason") or "")
    return bool(a.status == "timeout" or rec.get("budget_exhausted") or reason.startswith("budget")
                or rec.get("family_stop") == "deadline" or ev.get("budget_exceeded_s"))


def _next_budget(a):
    question = ((a.evidence or {}).get("search") or {}).get("question")
    if question in ("minor_radius", "volume"):
        return f"ask for a smaller size, or give a larger budget_s than {a.budget_s:.0f}"
    if question == "largest":
        return f"give a larger budget_s than {a.budget_s:.0f}"
    if a.budget_s < CERTIFICATE_BUDGET_S:
        return f"a certificate needs minutes: run again with budget_s={CERTIFICATE_BUDGET_S:.0f}"
    return (f"a larger budget_s than {a.budget_s:.0f} gives every stage more time. Or ask for a smaller flux, or for "
            f"the largest surface (largest=True)")


def _open_rows(a, ev, rec):
    st = a.status
    rows = []
    found, tried, nxt = _found(a, rec), _tried(rec), None
    p = ev.get("proof") or {}
    if st in ("surface_unverified", "range_unverified"):
        rows.append((st.replace("_", " "), "a torus was found and not verified: it is not an answer"))
        _asked(a, rows)
        why = _unverified_why(ev, rec)
        rows.append(("why", why))
        if a.surface is not None:
            found = [f"a torus at {abs(float(a.surface.flux)):.9g} Wb (answer.surface), not verified"]
        if "no time" in why or rec.get("refine_stop") == "deadline":
            nxt = _next_budget(a)
        else:
            nxt = "a slightly smaller flux, away from what spoilt this torus"
    elif st == "no_axis":
        rows.append(("no axis", "no magnetic axis: the search for an elliptic closed field line found none"))
        _asked(a, rows)
        nxt = ("if these are a tokamak's coils or a dipole, say so with kind='tokamak' or kind='dipole': torux "
               "settles those by proof. Otherwise check the currents and the field periods")
    elif st == "error":
        rows.append(("error", str(ev.get("error") or rec.get("error") or "the solve failed")))
        nxt = "this is a fault in torux, not an answer about the source: report it with the source file"
    elif st == "timeout":
        over = ev.get("budget_exceeded_s")
        rows.append(("timeout", "the budget ran out before anything was settled"
                     + (f": the last step ended {_s(over)} past the {a.budget_s:.0f} s budget" if over else "")))
        _asked(a, rows)
        nxt = _next_budget(a)
    elif p:
        v = str(p.get("verdict", ""))
        if v == "NO_TOROIDAL_FLUX_SURFACE_NUMERICAL":
            head = "no proof: the evidence that no surface encloses toroidal flux is numerical only"
        else:
            head = "no proof either way" + (f" ({p['note']})" if p.get("note") else "")
        rows.append(("undetermined", head))
        _asked(a, rows)
        args = [t for t in (p.get("arguments") or []) if isinstance(t, str)]
        if args:
            rows.append(("evidence", _phrases(args)))
        found, tried = [], []
        nxt = "torux has no proof for these coils: answer.evidence['proof'] holds what it measured"
    else:
        early = np.isfinite(a.wall_s) and a.wall_s < 0.9 * a.budget_s
        head = (f"neither a surface nor a certificate: torux stopped after {a.wall_s:.0f} s of its {a.budget_s:.0f} s "
                f"budget" if early else f"neither a surface nor a certificate within the {a.budget_s:.0f} s budget")
        rows.append(("undetermined", head))
        _asked(a, rows)
        if str(rec.get("deferred_reason") or "").startswith("policy"):
            tried.append(str(rec["deferred_reason"]))
        nxt = _next_budget(a)
    if found:
        rows.append(("found", ", ".join(found)))
    if tried:
        rows.append(("tried", ", ".join(tried[:4])))
    rows.append(("next", nxt))
    return rows


def _unverified_why(ev, rec):
    u = ev.get("unverified")
    ind = ev.get("independent") or {}
    if ind.get("skipped"):
        return f"the independent check was skipped: {ind['skipped']}"
    if u and "independent" in str(u):
        bits = []
        chk = ind.get("bn_sup", ind.get("bn_max"))
        if chk is not None:
            bits.append(f"it reads {_e(chk)} of |B| across the torus")
        if ind.get("flux_rel_err") is not None and ind["flux_rel_err"] >= 1e-12:
            bits.append(f"the flux is off by {_e(ind['flux_rel_err'], 0)}")
        if ind.get("trace_dist_max_rel") not in (None, 0.0) and not (ind["trace_dist_max_rel"] < 1e-9):
            bits.append("a traced field line left the torus")
        return "the independent evaluator refused it" + (": " + ", ".join(bits) if bits else "")
    s = rec.get("surface") or (rec.get("range") or {}).get("outer_torus") or {}
    g = rec.get("guard") or (rec.get("range") or {}).get("guard") or {}
    bits = []
    if s.get("bn_off") is not None and s["bn_off"] > BAR:
        bits.append(f"its field across reached only {_e(s['bn_off'])} of |B|, short of {BAR:.0e}")
    if g and g.get("ok") is False:
        bits.append("it lies too near a resonance (the resonance guard)")
    if s.get("quad_err_half") is not None and s["quad_err_half"] > 1e-8:
        bits.append("it lies too near a coil for the field's quadrature")
    if (s.get("topology") or {}).get("ok") is False:
        bits.append("its cross-sections are not simple closed curves round the axis")
    if bits:
        return "the solver's own acceptance refused it: " + ", and ".join(bits)
    return str(u or "the solver's own acceptance refused it")
