"""How good a flux surface of given coils a surface from another code is.

    c = torux.check("coils.json", "wout_mine.nc")
    print(c)
    c.leak          the largest |B.n|/|B| at 4,096 random points of the whole torus: 0 on an exact flux surface
    c.flux          the toroidal flux it encloses (Wb)

The surface may be a torux answer or its saved JSON, "quasr:954" (the surfaces QUASR stores), a VMEC input file
(its boundary), a VMEC or VMEC++ wout file (its outermost surface), a SIMSOPT JSON file holding surfaces (a QUASR
download included: the outermost,
or which= another), a live SIMSOPT SurfaceRZFourier or SurfaceXYZTensorFourier, a live DESC
FourierRZToroidalSurface or Equilibrium (its boundary), a VMEC++ output object, or a dictionary in the
rz_fourier or xyz_fourier form of results/evaluator/forms.py.  The kind is read from the object or the file.

Every surface is written exactly as one complex Fourier series of u = x + i y and of z in the surface's own
angles (theta, phi), from which points and normals follow. The field and its vector potential are the
independent evaluator's (torux.reference), which shares no code with the solver.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

NPTS = 4096                 # random points of the whole torus
NSEC = 6                    # sections for the flux
SEED = 0
N_LOOP_MAX = 8192           # largest rule round a section

KINDS_READ = ("a torux answer or its saved JSON, a VMEC input file, a VMEC or VMEC++ wout file, a SIMSOPT JSON "
              "file holding surfaces, a SIMSOPT SurfaceRZFourier or SurfaceXYZTensorFourier, a DESC "
              "FourierRZToroidalSurface or Equilibrium, a VMEC++ output, or a dictionary in the rz_fourier or "
              "xyz_fourier form")


# --------------------------------------------------------------------------- the measurement
@dataclass
class Check:
    """What torux.check measured on a surface: the field across it and the flux inside it.

    A small leak at 4,096 points says the surface is close to a flux surface of these coils. It is not torux's
    'surface' verdict, which also needs the solver's acceptance and the resonance guard."""
    surface: str                 # what the surface is: where it came from, its Fourier series, its field periods
    nfp: int                     # the surface's field periods (0: axisymmetric), read from its own series
    source_nfp: int              # the coils' field periods
    leak: float                  # the largest |B.n|/|B| at the points
    leak_median: float
    at: tuple                    # (theta, phi) of the largest, rad, in the surface's own angles
    rounding: float              # the evaluator's own rounding of B there, a fraction of |B|
    flux: float                  # toroidal flux enclosed (Wb, positive): the mean over the sections
    flux_spread: float           # (largest - smallest) / mean over the sections
    flux_stated: float = None    # the flux given, or stated by the surface's file (Wb)
    flux_stated_by: str = None
    flux_error: float = None     # flux / flux_stated - 1
    npts: int = NPTS
    nsec: int = NSEC
    notes: list = field(default_factory=list)

    def __str__(self):
        per = "none: the surface is axisymmetric" if self.nfp == 0 else str(self.nfp)
        if self.nfp != self.source_nfp:
            per += f" (the coils: {self.source_nfp})"
        rows = [("surface", self.surface), ("field periods", per),
                ("field across it", f"{self.leak:.1e} of |B| at worst, {self.leak_median:.1e} in the median, at "
                                    f"{self.npts:,} points (the evaluator's rounding: {_e(self.rounding)})"),
                ("toroidal flux", f"{self.flux:.12g} Wb, the mean through {self.nsec} cross-sections, which differ "
                                  f"by {self.flux_spread:.1e} of it")]
        if self.flux_stated is not None:
            rows.append(("stated flux", f"{self.flux_stated:.12g} Wb ({self.flux_stated_by}), and the flux inside is "
                                        f"{'larger' if self.flux_error >= 0 else 'smaller'} by "
                                        f"{abs(self.flux_error):.1e} of it"))
        rows += [("note", n) for n in self.notes]
        from .answer import _table
        return _table(rows)

    __repr__ = __str__


def _e(v):
    return "0" if v == 0 else f"{v:.0e}"


def check(source, surface, flux=None, *, which=None) -> Check:
    """How close ``surface`` is to a flux surface of ``source``: |B.n|/|B| at 4,096 random points and the
    toroidal flux it encloses (Wb).  ``source``: anything torux.load reads, or a torux.Source.  ``flux``: a
    flux to compare with (Wb, by default the one the surface's file states).  ``which``: the surface to take,
    by name or index, from a file holding several (default: the outermost).  The surface of a SIMSOPT BoozerSurface
    or QFMSurface is measured only against the coils its file says it was computed for: a source whose field on it
    differs is refused (ValueError)."""
    if flux is not None and (isinstance(flux, bool) or not isinstance(flux, (int, float, np.integer, np.floating))
                             or not math.isfinite(flux) or flux == 0):
        raise ValueError(f"flux={flux!r}: give the flux to compare with in Wb, as a number other than zero")
    src = _source(source)
    S = read_surface(surface, which)
    rng = np.random.default_rng(SEED)
    theta = 2 * np.pi * rng.random(NPTS)
    phi = 2 * np.pi * rng.random(NPTS)
    sections = 2 * np.pi * rng.random(NSEC)
    return measure_at(src, S, theta, phi, sections, flux)


def measure_at(src, S, theta, phi, sections, flux=None) -> Check:
    """The measurement of check at given angles (rad): |B.n|/|B| at the points (theta, phi) and the flux on
    the sections at the toroidal angles ``sections``.  S: a surface from read_surface."""
    from .reference import fast64
    if not isinstance(S, Surface):
        S = read_surface(S)
    theta = np.asarray(theta, float).ravel()
    phi = np.asarray(phi, float).ravel()
    if theta.shape != phi.shape or theta.size == 0:
        raise ValueError("theta and phi must be arrays of the same, nonzero length")
    notes = []
    F = fast64.Field(src)
    x, xt, xp = S.eval(theta, phi)
    nv = np.cross(xt, xp)
    B = np.asarray(F.evaluate(x, grad=False)["B"], float)
    if S.own_coils is not None:
        notes += _against_own_coils(S, src, x, B)
    taken = ((getattr(src, "meta", None) or {}).get("currents") or {})
    if taken.get("differing"):
        notes.append(f"the source's file states two values for {', '.join(taken['differing'])}: the "
                     f"{taken['taken']!r} ones were taken, as asked")
    with np.errstate(invalid="ignore", divide="ignore"):
        bn = np.abs(np.sum(B * nv, -1)) / (np.linalg.norm(B, axis=-1) * np.linalg.norm(nv, axis=-1))
    ok = np.isfinite(bn)
    if not ok.any():
        raise ValueError("the field or the surface's normal is undefined at every point: the surface is degenerate")
    if not ok.all():
        notes.append(f"{int((~ok).sum())} of {bn.size:,} points have no |B.n|/|B| (a point on a coil, or a "
                     "vanishing normal) and are left out")
    k = int(np.argmax(np.where(ok, bn, -1.0)))
    leak, leak_med = float(bn[k]), float(np.median(bn[ok]))
    rounding = _rounding(src, x[k])
    fl, conv = [], True
    for p in np.asarray(sections, float).ravel():
        f, c = _section_flux(S, F, float(p))
        fl.append(f)
        conv &= c
    fl = np.array(fl)
    if not conv:
        notes.append(f"the flux round a section did not settle with {N_LOOP_MAX:,} points (the surface runs "
                     "through or next to a coil): the flux is not reliable")
    if np.any(fl > 0) and np.any(fl < 0):
        notes.append("the sections disagree on the sign of the flux: the surface does not enclose one flux")
    af = np.abs(fl)
    fmean = float(np.mean(af))
    spread = float((af.max() - af.min()) / fmean) if fmean > 0 else float("nan")
    stated, by = (abs(float(flux)), "given") if flux is not None else (S.flux, S.flux_by)
    err = (fmean / stated - 1.0) if stated else None
    return Check(surface=S.describe(), nfp=S.nfp, source_nfp=int(getattr(src, "nfp", 1) or 1), leak=leak,
                 leak_median=leak_med, at=(float(theta[k]), float(phi[k])), rounding=rounding, flux=fmean,
                 flux_spread=spread, flux_stated=stated if stated else None, flux_stated_by=by if stated else None,
                 flux_error=err, npts=int(theta.size), nsec=int(fl.size), notes=notes)


OWN_FIELD_TOL = 1e-9         # the same coils, however written, make the same field to rounding


def _against_own_coils(S, src, x, B):
    """A SIMSOPT BoozerSurface or QFMSurface was computed for the coils its own file holds with it: if the source
    given makes a different field on the surface, the two files do not belong together and a leak measured against
    the source says nothing about the surface. Refused. Returns notes when the comparison cannot be made."""
    from .io.simsopt_json import Simson, coils_to_source
    from .reference import fast64
    own = S.own_coils
    taken = ((getattr(src, "meta", None) or {}).get("currents") or {}).get("taken")
    try:
        sim = Simson(own["dict"], own["origin"], taken)
        mine = coils_to_source(sim, sim.coils_of(own["biotsavart"]), own["biotsavart"], check_symmetry=False)
    except (ValueError, NotImplementedError, KeyError, TypeError) as e:
        return [f"{own['boozer']} was computed for its file's {own['biotsavart']}, which torux cannot read to compare "
                f"with the source given ({e})"]
    Bo = np.asarray(fast64.Field(mine).evaluate(x, grad=False)["B"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.linalg.norm(B - Bo, axis=-1) / np.linalg.norm(Bo, axis=-1)
    gap = float(np.nanmax(gap)) if np.any(np.isfinite(gap)) else float("inf")
    if not gap <= OWN_FIELD_TOL:
        raise ValueError(
            f"{own['origin']}: {own['surface']} is the surface of {own['boozer']}, which its file says was computed "
            f"for the file's own coils, {own['biotsavart']}. The source given makes a different field on it, by up to "
            f"{gap:.1e} of |B|: the surface was not computed for these coils, and what crosses it measures the "
            f"mismatch of the two coil sets, not the surface. Check it against its own coils: {own['biotsavart']} of "
            f"{own['origin']}, as the source torux.load(path, which={own['biotsavart']!r}) or "
            f"`torux check {own['origin']} {own['origin']} --coils {own['biotsavart']}`")
    return []


def _rounding(src, x):
    """|B_fast64 - B_double-double| / |B| at one point: the evaluator's own rounding there."""
    from .reference import fast64, precise
    x = np.asarray(x, float).reshape(1, 3)
    b64 = np.asarray(fast64.Field(src).evaluate(x, grad=False)["B"], float)
    bdd = np.asarray(precise.Reference(src).evaluate(x, grad=False)["B"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.linalg.norm(b64 - bdd) / np.linalg.norm(bdd))


def _section_flux(S, F, phi):
    """Loop integral of A once round theta at the surface's own toroidal angle phi: the trapezoid rule (spectrally
    accurate on the closed loop), doubled until it changes by at most 1e-14 of itself."""
    n = 64
    while n < 4 * S.M_arr:
        n *= 2

    def part(th):
        x, xt, _ = S.eval(th, np.full(th.size, phi))
        A = np.asarray(F.evaluate(x, grad=False)["A"], float)
        return float(np.sum(A * xt))

    s = part(2 * np.pi * np.arange(n) / n)
    f = 2 * np.pi * s / n
    while n < N_LOOP_MAX:
        s += part(2 * np.pi * (np.arange(n) + 0.5) / n)
        n *= 2
        g = 2 * np.pi * s / n
        if abs(g - f) <= 1e-14 * abs(g):
            return g, True
        f = g
    return f, False


def _source(source):
    if hasattr(source, "parts"):
        return source
    from .sources import load
    return load(source)


# --------------------------------------------------------------------------- the surface
class Surface:
    """A surface as complex Fourier series in its own angles:  u = x + i y = sum U[m, k] exp(i (m theta + k phi)),
    z = sum Z[m, k] exp(i (m theta + k phi)) (Z Hermitian: z real), m = -M..M, k = -K..K."""

    own_coils = None             # a SIMSOPT BoozerSurface's or QFMSurface's: the coils its file says it was
                                 # computed for

    def __init__(self, terms_u, terms_z, origin, kind, size, flux=None, flux_by=None):
        mu, ku, cu = (np.asarray(a) for a in terms_u)
        mz, kz, cz = (np.asarray(a) for a in terms_z)
        mm = np.concatenate([mu, mz]).astype(int)
        kk = np.concatenate([ku, kz]).astype(int)
        self.M_arr = int(np.max(np.abs(mm))) if mm.size else 0
        self.K_arr = int(np.max(np.abs(kk))) if kk.size else 0
        shape = (2 * self.M_arr + 1, 2 * self.K_arr + 1)
        self.U = np.zeros(shape, complex)
        self.Z = np.zeros(shape, complex)
        np.add.at(self.U, (mu.astype(int) + self.M_arr, ku.astype(int) + self.K_arr), cu.astype(complex))
        np.add.at(self.Z, (mz.astype(int) + self.M_arr, kz.astype(int) + self.K_arr), cz.astype(complex))
        if not (np.all(np.isfinite(self.U)) and np.all(np.isfinite(self.Z))):
            raise ValueError(f"{origin}: the surface has coefficients that are not finite numbers")
        if not np.any(self.U):
            raise ValueError(f"{origin}: the surface's coefficients are all zero")
        self.origin, self.kind, self.size = origin, kind, size
        self.flux = abs(float(flux)) if flux is not None and np.isfinite(flux) and flux != 0 else None
        self.flux_by = flux_by if self.flux is not None else None
        self.nfp = self._periods()

    def _periods(self):
        """Field periods from the series itself: u = exp(i phi) times a function of period 2 pi / nfp, z of
        period 2 pi / nfp. 0 for an axisymmetric surface."""
        k = np.arange(-self.K_arr, self.K_arr + 1)
        g = 0
        for v in np.unique(np.abs(k[np.nonzero(np.any(self.U != 0, axis=0))[0]] - 1)):
            g = math.gcd(g, int(v))
        for v in np.unique(np.abs(k[np.nonzero(np.any(self.Z != 0, axis=0))[0]])):
            g = math.gcd(g, int(v))
        return g

    def describe(self):
        return (f"{self.origin}: " if self.origin else "") + f"{self.kind}, {self.size}"

    def eval(self, theta, phi):
        """Points x and the derivatives x_theta, x_phi, each (P, 3), at angles theta, phi (rad)."""
        theta = np.asarray(theta, float).ravel()
        phi = np.asarray(phi, float).ravel()
        m = np.arange(-self.M_arr, self.M_arr + 1)
        k = np.arange(-self.K_arr, self.K_arr + 1)
        P = theta.size
        out = np.empty((3, P, 3))
        step = max(1, 2_000_000 // ((2 * self.M_arr + 1) * (2 * self.K_arr + 1)))
        for a in range(0, P, step):
            th, ph = theta[a:a + step], phi[a:a + step]
            Et = np.exp(1j * np.outer(th, m))
            Ep = np.exp(1j * np.outer(ph, k))
            res = []
            for C in (self.U, self.Z):
                G = Ep @ C.T                              # (p, m): sum_k C[m, k] exp(i k phi)
                Gp = Ep @ (C * (1j * k)).T
                res.append((np.sum(Et * G, 1), np.sum(Et * (1j * m) * G, 1), np.sum(Et * Gp, 1)))
            for j in range(3):
                u, z = res[0][j], res[1][j]
                out[j, a:a + step] = np.stack([u.real, u.imag, z.real], -1)
        return out[0], out[1], out[2]

    def volume(self, n=64):
        th, ph = np.meshgrid(2 * np.pi * np.arange(n) / n, 2 * np.pi * np.arange(n) / n, indexing="ij")
        x, xt, xp = self.eval(th, ph)
        return float(abs(np.sum(x * np.cross(xt, xp))) * (2 * np.pi / n) ** 2 / 3)


# ---- exact conversions to (m, k, coefficient) terms
def _real_series(m, n, c, s, phase=None):
    """sum c cos(m theta - n phi + phase) + s sin(...) as complex terms of exp(i (m theta + k phi))."""
    m, n = np.asarray(m, int), np.asarray(n, int)
    c, s = np.asarray(c, float), np.asarray(s, float)
    plus = (c - 1j * s) / 2
    if phase is not None:
        plus = plus * np.exp(1j * phase)
    return (np.concatenate([m, -m]), np.concatenate([-n, n]), np.concatenate([plus, np.conj(plus)]))


def _shift(terms, dk):
    m, k, c = terms
    return m, np.asarray(k) + dk, c


def _rz(m, n, rc, rs, zc, zs, phase=None):
    """R, Z series in m theta - n phi, phi the cylindrical angle: u = R exp(i phi)."""
    z0 = np.zeros(len(np.asarray(m)))
    rs = z0 if rs is None else rs
    zc = z0 if zc is None else zc
    return _shift(_real_series(m, n, rc, rs, phase), 1), _real_series(m, n, zc, zs, phase)


def _join(*ts):
    return tuple(np.concatenate([t[i] for t in ts]) for i in range(3))


def _dense_terms(Fm, Fk, C):
    """Nonzero entries of a dense array C[i, j] at mode numbers Fm[i], Fk[j]."""
    i, j = np.nonzero(C)
    return Fm[i], Fk[j], C[i, j]


def _tensor_xyz(X, Y, Zc, mpol, ntor, nfp, clamped=(False, False, False)):
    """SIMSOPT's SurfaceXYZTensorFourier: xhat = sum X[i, j] w_i(theta) v_j(phi), w_i = cos(i theta) for i <=
    mpol, sin((i - mpol) theta) above, v_j = cos(nfp j phi) for j <= ntor, sin(nfp (j - ntor) phi) above. A
    clamped coordinate multiplies its block i <= mpol, j <= ntor by sin^2(nfp phi / 2) + sin^2(theta / 2) =
    1 - cos(theta) / 2 - cos(nfp phi) / 2, with x = xhat cos phi - yhat sin phi, y = xhat sin phi + yhat cos phi,
    z = zhat: u = exp(i phi) (xhat + i yhat)."""
    Wt = np.zeros((2 * mpol + 1, 2 * mpol + 1), complex)        # basis i -> coefficient of exp(i m theta)
    for i in range(2 * mpol + 1):
        q = i if i <= mpol else i - mpol
        a, b = (0.5, 0.5) if i <= mpol else (-0.5j, 0.5j)
        Wt[i, mpol + q] += a
        Wt[i, mpol - q] += b
    K = nfp * ntor
    Vp = np.zeros((2 * ntor + 1, 2 * K + 1), complex)          # basis j -> coefficient of exp(i k phi)
    for j in range(2 * ntor + 1):
        q = j if j <= ntor else j - ntor
        a, b = (0.5, 0.5) if j <= ntor else (-0.5j, 0.5j)
        Vp[j, K + nfp * q] += a
        Vp[j, K - nfp * q] += b
    Fm, Fk = np.arange(-mpol, mpol + 1), np.arange(-K, K + 1)
    out = []
    for d, A in enumerate((X, Y, Zc)):
        A = np.asarray(A, float)
        t = _dense_terms(Fm, Fk, Wt.T @ A @ Vp)
        if clamped[d]:
            Ac = np.zeros_like(A)
            Ac[:mpol + 1, :ntor + 1] = A[:mpol + 1, :ntor + 1]
            gm, gk, gc = _dense_terms(Fm, Fk, Wt.T @ Ac @ Vp)
            t = _join(t, (gm + 1, gk, -gc / 4), (gm - 1, gk, -gc / 4), (gm, gk + nfp, -gc / 4),
                      (gm, gk - nfp, -gc / 4))
        out.append(t)
    xm, xk, xc = out[0]
    ym, yk, yc = out[1]
    u = _join((xm, xk, xc.astype(complex)), (ym, yk, 1j * yc))
    return _shift(u, 1), out[2]


# --------------------------------------------------------------------------- reading a surface
def read_surface(obj, which=None) -> Surface:
    """The surface of any kind check reads, recognised from the object or the file itself."""
    if isinstance(obj, Surface):
        _no_which(which, "a surface")
        return obj
    if isinstance(obj, (str, os.PathLike)):
        return _read_file(os.fspath(obj), which)
    if isinstance(obj, dict):
        return _read_dict(obj, "", which)
    mod, name = type(obj).__module__ or "", type(obj).__name__
    if mod.startswith("simsopt") and name in ("SurfaceRZFourier", "SurfaceXYZTensorFourier"):
        _no_which(which, "a SIMSOPT surface")
        return _simsopt_live(obj)
    if mod.startswith("desc") and name == "FourierRZToroidalSurface":
        _no_which(which, "a DESC surface")
        return _desc_surface(obj, "", "DESC FourierRZToroidalSurface")
    if mod.startswith("desc") and hasattr(obj, "Psi") and getattr(obj, "surface", None) is not None:
        _no_which(which, "a DESC equilibrium")
        return _desc_surface(obj.surface, "", f"the boundary of a DESC {name}", flux=float(np.asarray(obj.Psi)),
                             flux_by="the equilibrium's Psi")
    if mod.startswith("vmecpp"):
        w = getattr(obj, "wout", obj)
        if hasattr(w, "rmnc") and hasattr(w, "xm"):
            return _vmecpp_wout(w, which)
    if all(hasattr(obj, a) for a in ("rc", "zs", "M", "N", "nfp")):
        _no_which(which, "a torus")
        return _torus(obj, "", "torux's Fourier series")
    if hasattr(obj, "surface"):
        status = getattr(obj, "status", None)
        if obj.surface is None:
            raise ValueError(f"this {name} holds no surface" + (f" (its status is {status!r})" if status else "")
                             + ": give an answer that has one")
        S = read_surface(obj.surface, which)
        if status is not None:
            S.origin = f"a torux answer ({status})"
        return S
    raise ValueError(f"{name}: not a surface torux.check reads. It reads {KINDS_READ}")


def _no_which(which, what):
    if which is not None:
        raise ValueError(f"which= picks one surface of a file holding several, and {what} is one surface")


def _torus(t, origin, kind):
    """A torux torus: R = sum rc[m, n + N] cos(m theta - n nfp (phi - phi0)) (+ rs sin), Z = sum zs sin (+ zc
    cos), theta the straight-field-line angle, phi the cylindrical angle."""
    g = (lambda k: getattr(t, k, None)) if not isinstance(t, dict) else t.get
    M, N, nfp = int(g("M")), int(g("N")), int(g("nfp"))
    phi0 = float(g("phi0") or 0.0)
    m = np.repeat(np.arange(M + 1), 2 * N + 1)
    n = np.tile(np.arange(-N, N + 1), M + 1) * nfp
    arr = {}
    for key in ("rc", "rs", "zc", "zs"):
        v = g(key)
        if v is not None:
            a = np.asarray(v, float)
            if a.shape != (M + 1, 2 * N + 1):
                raise ValueError(f"{origin or 'the torus'}: {key} has shape {a.shape}, not (M + 1, 2 N + 1) = "
                                 f"{(M + 1, 2 * N + 1)}")
            arr[key] = a.ravel()
    if "rc" not in arr or "zs" not in arr:
        raise ValueError(f"{origin or 'the torus'}: a torus needs rc and zs")
    u, z = _rz(m, n, arr["rc"], arr.get("rs"), arr.get("zc"), arr["zs"], phase=n * phi0 if phi0 else None)
    fl = g("flux") if g("flux") is not None else g("flux_Wb")
    return Surface(u, z, origin, kind, f"M, N = {M}, {N}", flux=fl, flux_by="the answer's")


def _form(d, origin):
    """A dictionary in the form of results/evaluator/forms.py."""
    m, n = np.asarray(d["m"], int), np.asarray(d["n"], int)
    J = m.size

    def c(k):
        v = d.get(k)
        a = np.zeros(J) if v is None else np.asarray(v, float)
        if a.shape != (J,):
            raise ValueError(f"{origin or 'the dictionary'}: {k} has {a.size} values for {J} modes")
        return a
    if n.shape != (J,):
        raise ValueError(f"{origin or 'the dictionary'}: n has {n.size} values for {J} modes")
    size = f"{J} terms, m up to {int(np.abs(m).max()) if J else 0}, n up to {int(np.abs(n).max()) if J else 0}"
    if d["form"] == "rz_fourier":
        u, z = _rz(m, n, c("rc"), c("rs"), c("zc"), c("zs"))
    else:
        xm, xk, xc = _real_series(m, n, c("xc"), c("xs"))
        ym, yk, yc = _real_series(m, n, c("yc"), c("ys"))
        u, z = _join((xm, xk, xc), (ym, yk, 1j * yc)), _real_series(m, n, c("zc"), c("zs"))
    return Surface(u, z, origin, d["form"] + " form", size)


def _read_dict(d, origin, which):
    form = d.get("form")
    if form in ("rz_fourier", "xyz_fourier"):
        _no_which(which, "a form dictionary")
        return _form(d, origin)
    if "simsopt_objs" in d:
        return _simson(d, origin, which)
    s = d.get("surface", "absent")
    if isinstance(s, dict) and s.get("form") in ("rz_fourier", "xyz_fourier"):
        _no_which(which, "an answer")
        return _form(s, origin)
    if isinstance(s, dict) and "rc" in s and "zs" in s:
        _no_which(which, "an answer")
        return _torus(s, origin, "torux surface" + (f" ({d['status']})" if d.get("status") else ""))
    if s is None:
        raise ValueError(f"{origin or 'the dictionary'}: holds no surface"
                         + (f" (its status is {d['status']!r})" if d.get("status") else ""))
    st = (d.get("start") or {}).get("surface") if isinstance(d.get("start"), dict) else None
    if isinstance(st, dict) and st.get("kind") == "SurfaceXYZTensorFourier":
        _no_which(which, "this file's start surface")
        from .io.surfaces import SurfaceXYZTensorFourier as XYZ
        t = XYZ(st["nfp"], st["stellsym"], st["mpol"], st["ntor"], np.asarray(st["x"], float),
                clamped_dims=tuple(st.get("clamped_dims", (False, False, False))))
        u, z = _tensor_xyz(t.x, t.y, t.z, t.mpol, t.ntor, t.nfp, t.clamped_dims)
        req = d.get("requested_toroidal_flux_Wb")
        return Surface(u, z, origin, "its start surface (SIMSOPT SurfaceXYZTensorFourier)",
                       f"M, N = {t.mpol}, {t.ntor}", flux=req, flux_by="the file's request")
    raise ValueError(f"{origin or 'the dictionary'}: holds no surface torux.check reads. It reads {KINDS_READ}")


def _read_file(path, which):
    origin = os.path.basename(path)
    if not os.path.exists(path):
        from .sources import quasr_file, quasr_number
        n = quasr_number(path)
        if n is None:
            raise FileNotFoundError(f"{path}: no such file")
        path, origin = quasr_file(n), f"QUASR design {n}"          # the surfaces QUASR stores for that design
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: not a file")
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:3] == b"CDF" or raw[:8] == b"\x89HDF\r\n\x1a\n":
        return _wout_file(path, origin, raw[:3] == b"CDF", which)
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        import json
        try:
            d = json.loads(text)
        except ValueError as e:
            raise ValueError(f"{path}: a JSON file that does not parse ({e})") from None
        return _read_dict(d, origin, which)
    import re
    if re.search(r"&\s*indata", text, re.I):
        _no_which(which, "a VMEC input file")
        from .io.surfaces import vmec_input_boundary
        b = vmec_input_boundary(path)
        if b.rc.size == 0:
            raise ValueError(f"{path}: a VMEC input file with no RBC, ZBS boundary coefficients")
        u, z = _rz(b.m, b.n * b.nfp, b.rc, b.rs, b.zc, b.zs)
        size = f"M, N = {int(b.m.max())}, {int(np.abs(b.n).max())}"
        return Surface(u, z, origin, "VMEC input boundary", size, flux=b.meta.get("PHIEDGE"),
                       flux_by="the file's PHIEDGE")
    raise ValueError(f"{path}: not a surface file torux.check reads. It reads {KINDS_READ}")


# ---- VMEC wout (NetCDF3 from VMEC2000, NetCDF4 from VMEC++)
def _wout_file(path, origin, netcdf3, which):
    need = ("nfp", "xm", "xn", "rmnc", "zmns")
    opt = ("rmns", "zmnc", "phi", "lasym__logical__")
    v = {}
    if netcdf3:
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as f:
            for k in need + opt:
                if k in f.variables:
                    v[k] = np.array(f.variables[k].data, copy=True)
    else:
        try:
            import netCDF4
        except ImportError:
            netCDF4 = None
        if netCDF4 is not None:
            with netCDF4.Dataset(path) as f:
                for k in need + opt:
                    if k in f.variables:
                        v[k] = np.array(np.ma.filled(f.variables[k][...], np.nan))
        else:
            try:
                import h5py
            except ImportError:
                raise ImportError(f'{path}: reading a NetCDF4 (HDF5) file needs h5py: pip install ".[hdf5]" in the '
                                  f'torux folder') from None
            with h5py.File(path, "r") as f:
                for k in need + opt:
                    if k in f:
                        v[k] = np.array(f[k][()])
    if not all(k in v for k in need):
        missing = ", ".join(k for k in need if k not in v)
        raise ValueError(f"{path}: a NetCDF or HDF5 file, but not a VMEC wout (no {missing}). "
                         f"torux.check reads {KINDS_READ}")
    lasym = bool(np.asarray(v.get("lasym__logical__", 0)).ravel()[0]) if "lasym__logical__" in v else \
        ("rmns" in v and "zmnc" in v)
    return _wout_arrays(v["xm"], v["xn"], v["nfp"], v["rmnc"], v["zmns"], v.get("rmns") if lasym else None,
                        v.get("zmnc") if lasym else None, v.get("phi"), origin, "VMEC wout", which,
                        rows_are_surfaces=True)


def _wout_arrays(xm, xn, nfp, rmnc, zmns, rmns, zmnc, phi, origin, kind, which, rows_are_surfaces):
    xm, xn = np.asarray(xm, float).ravel(), np.asarray(xn, float).ravel()
    nfp = int(np.asarray(nfp).ravel()[0])
    J = xm.size

    def surfaces_first(a):
        a = np.asarray(a, float)
        a = a[None] if a.ndim == 1 else (a if rows_are_surfaces else a.T)
        if a.shape[1] != J:
            raise ValueError(f"{origin or kind}: rmnc and xm disagree on the number of modes")
        return a
    R = surfaces_first(rmnc)
    ns = R.shape[0]
    js = -1 if which is None else which
    if isinstance(js, str):
        try:
            js = int(js)
        except ValueError:
            raise ValueError(f"which= for a wout is the index of a radial surface, 0 to {ns - 1}") from None
    if not -ns <= int(js) < ns:
        raise ValueError(f"which={which!r}: this wout has {ns} radial surfaces, 0 to {ns - 1}")
    js = int(js) % ns
    pick = lambda a: None if a is None else surfaces_first(a)[js]  # noqa: E731
    u, z = _rz(xm.astype(int), np.rint(xn).astype(int), R[js], pick(rmns), pick(zmnc), pick(zmns))
    fl = None
    if phi is not None:
        ph = np.asarray(phi, float).ravel()
        fl = float(ph[js]) if ph.size == ns else None
    where = "outermost" if js == ns - 1 else f"radial index {js}"
    size = f"M, N = {int(xm.max())}, {int(np.abs(xn).max()) // max(nfp, 1)}"
    return Surface(u, z, origin, f"{kind}, {where} of {ns} surfaces", size, flux=fl, flux_by="the file's phi")


def _vmecpp_wout(w, which):
    """A VMEC++ output (VmecOutput, or its wout): rmnc and the like are (mode, surface)."""
    lasym = bool(getattr(w, "lasym", False))
    return _wout_arrays(w.xm, w.xn, w.nfp, w.rmnc, w.zmns, getattr(w, "rmns", None) if lasym else None,
                        getattr(w, "zmnc", None) if lasym else None, getattr(w, "phi", None), "",
                        "VMEC++ output", which, rows_are_surfaces=False)


# ---- SIMSOPT
def _simsopt_live(s):
    name = type(s).__name__
    if name == "SurfaceRZFourier":
        mpol, ntor, nfp = int(s.mpol), int(s.ntor), int(s.nfp)
        m = np.repeat(np.arange(mpol + 1), 2 * ntor + 1)
        n = np.tile(np.arange(-ntor, ntor + 1), mpol + 1) * nfp
        get = lambda k: None if getattr(s, k, None) is None else np.asarray(getattr(s, k), float).ravel()  # noqa: E731
        sym = bool(s.stellsym)
        u, z = _rz(m, n, get("rc"), None if sym else get("rs"), None if sym else get("zc"), get("zs"))
        return Surface(u, z, "", "SIMSOPT SurfaceRZFourier", f"M, N = {mpol}, {ntor}")
    clamped = tuple(bool(c) for c in getattr(s, "clamped_dims", (False, False, False)))
    u, z = _tensor_xyz(s.xcs, s.ycs, s.zcs, int(s.mpol), int(s.ntor), int(s.nfp), clamped)
    return Surface(u, z, "", "SIMSOPT SurfaceXYZTensorFourier", f"M, N = {int(s.mpol)}, {int(s.ntor)}")


def _simson_one(sim, nm, origin):
    import re
    o = sim.objs[nm]
    cls = o["@class"]
    # the name that which= takes. SIMSOPT's own names begin with the class
    label = nm if nm.startswith(cls) else f"SIMSOPT {cls} {nm}"
    if cls == "SurfaceXYZTensorFourier":
        from .io.surfaces import SurfaceXYZTensorFourier as XYZ
        t = XYZ.from_gson(o, sim.dofs_x(o), nm)
        u, z = _tensor_xyz(t.x, t.y, t.z, t.mpol, t.ntor, t.nfp, t.clamped_dims)
        return _boozer_of(sim, nm, origin, Surface(u, z, origin, label, f"M, N = {t.mpol}, {t.ntor}"))
    if cls == "SurfaceRZFourier":
        mpol, ntor, nfp = int(o["mpol"]), int(o["ntor"]), int(o["nfp"])
        dofs = sim.obj(o["dofs"])
        x = np.asarray(sim.dofs_x(o), float)
        names = dofs.get("names")
        coef = {k: np.zeros((mpol + 1, 2 * ntor + 1)) for k in ("rc", "rs", "zc", "zs")}
        if names is None or len(names) != x.size:
            raise ValueError(f"{origin}: {nm} does not name its degrees of freedom")
        for nmk, val in zip(names, x):
            mt = re.fullmatch(r"(rc|rs|zc|zs)\((-?\d+),(-?\d+)\)", str(nmk).replace(" ", ""))
            if not mt:
                raise ValueError(f"{origin}: {nm} has a degree of freedom {nmk!r} that is not rc, rs, zc or zs(m,n)")
            coef[mt.group(1)][int(mt.group(2)), int(mt.group(3)) + ntor] = val
        m = np.repeat(np.arange(mpol + 1), 2 * ntor + 1)
        n = np.tile(np.arange(-ntor, ntor + 1), mpol + 1) * nfp
        u, z = _rz(m, n, *(coef[k].ravel() for k in ("rc", "rs", "zc", "zs")))
        return _boozer_of(sim, nm, origin, Surface(u, z, origin, label, f"M, N = {mpol}, {ntor}"))
    raise ValueError(f"{origin}: {nm} is a SIMSOPT {cls}. torux.check reads SurfaceRZFourier and "
                     "SurfaceXYZTensorFourier")


def _boozer_of(sim, nm, origin, S):
    """S, with the coils a BoozerSurface or QFMSurface of the file computed it for, when one holds it."""
    from .io.simsopt_json import boozer_pairs
    for b, (bs, surf) in boozer_pairs(sim).items():
        if surf == nm and bs in sim.objs and sim.objs[bs]["@class"] == "BiotSavart":
            S.own_coils = dict(dict=sim.raw, origin=origin, boozer=b, biotsavart=bs, surface=nm)
            break
    return S


def _simson(d, origin, which):
    from .io.simsopt_json import Simson
    sim = Simson(d, origin)
    names = sim.surfaces()
    if not names:
        raise ValueError(f"{origin or 'the SIMSOPT file'}: holds no surface")
    if which is not None:
        pick = [nm for nm in names if nm == str(which)]
        if not pick:
            try:
                pick = [names[int(which)]]
            except (ValueError, IndexError):
                raise ValueError(f"which={which!r} is none of the file's {len(names)} surfaces: "
                                 f"{', '.join(names)} (or their index, 0 to {len(names) - 1})") from None
        S = _simson_one(sim, pick[0], origin)
        if len(names) > 1:
            S.kind += f", number {names.index(pick[0])} of the file's {len(names)}"
        return S
    if len(names) == 1:
        return _simson_one(sim, names[0], origin)
    surfs = [_simson_one(sim, nm, origin) for nm in names]
    vols = [s.volume() for s in surfs]
    k = int(np.argmax(vols))
    surfs[k].kind += f", the outermost of {len(names)}"
    return surfs[k]


# ---- DESC
def _desc_surface(s, origin, kind, flux=None, flux_by=None):
    """DESC's double Fourier series: sum c G_m(theta) G_n(NFP zeta), G_m = cos(|m| x) for m >= 0, sin(|m| x) for
    m < 0, and zeta is the cylindrical angle."""
    nfp = int(s.NFP)

    def terms(modes, c):
        modes = np.asarray(modes, int)
        c = np.asarray(c, float)
        m, n = modes[:, 1], modes[:, 2]
        am, an = np.abs(m), np.abs(n) * nfp
        out = []
        for sm, fm in ((1, 0.5), (-1, 0.5)):
            for sn, fn in ((1, 0.5), (-1, 0.5)):
                wm = np.where(m >= 0, fm, -0.5j * sm)
                wn = np.where(n >= 0, fn, -0.5j * sn)
                out.append((sm * am, sn * an, c * wm * wn))
        return _join(*out)
    R = terms(s.R_basis.modes, s.R_lmn)
    Z = terms(s.Z_basis.modes, s.Z_lmn)
    mR = np.asarray(s.R_basis.modes, int)
    size = f"M, N = {int(np.abs(mR[:, 1]).max())}, {int(np.abs(mR[:, 2]).max())}"
    return Surface(_shift(R, 1), Z, origin, kind, size, flux=flux, flux_by=flux_by)


# --------------------------------------------------------------------------- the command line
def add_command(sub):
    """The parser of `torux check`."""
    c = sub.add_parser("check", help="how close a surface from another code is to a flux surface of the coils",
                       description="How close a surface is to a flux surface of the source: |B.n|/|B| at 4,096 "
                                   "random points of it and the toroidal flux it encloses.  It measures and does not "
                                   "verify.")
    c.add_argument("files", nargs="+", metavar="SOURCE", help="a coil or magnet file, or quasr:NUMBER. Several files "
                                                               "whose parts add up are one source")
    c.add_argument("surface", metavar="SURFACE", help="an answer's .json, a VMEC input.<name> or wout_<name>.nc, a "
                                                      "SIMSOPT JSON file holding surfaces, or quasr:NUMBER (the "
                                                      "surfaces QUASR stores)")
    c.add_argument("--flux", type=float, metavar="WB", help="a flux to compare with, in Wb (default: the one the "
                                                            "surface's file states)")
    c.add_argument("--nfp", type=int, metavar="N", help="field periods of the source, where the format does not "
                                                        "state them")
    c.add_argument("--which", metavar="NAME", help="the surface to take from a file holding several: its name or "
                                                   "index (default: the outermost)")
    add_source_choices(c)
    return c


def add_source_choices(c):
    """--coils and --currents: what torux.load's which= and currents= say, for a SIMSOPT JSON source."""
    c.add_argument("--coils", metavar="NAME", action="append",
                   help="the coil set to take from a SIMSOPT JSON file holding several: its BiotSavart's name or "
                        "index. With several source files, once per file in order, - for a file that needs none")
    c.add_argument("--currents", choices=("saved", "dofs"), help="for a SIMSOPT JSON file whose Current objects "
                                                                 "state two different values: the one each held "
                                                                 "when the file was saved, or the one SIMSOPT reads "
                                                                 "back from its degrees of freedom")


def source_choices(a):
    """torux.load's which= and currents= from --coils and --currents: --coils once is which= of the first file, more
    than once a list, one entry per file, - for None."""
    def one(w):
        if w == "-":
            return None
        return int(w) if w.lstrip("-").isdigit() else w
    coils = a.coils or []
    which = None if not coils else one(coils[0]) if len(coils) == 1 else [one(w) for w in coils]
    return dict(which=which, currents=a.currents)


def run_command(a):
    """`torux check`: prints the measurement. 0 when it measured, 1 on an error."""
    import sys
    try:
        from .sources import load
        src = load(*a.files, nfp=a.nfp, **source_choices(a))
        which = a.which
        if which is not None and which.lstrip("-").isdigit():
            which = int(which)
        c = check(src, a.surface, flux=a.flux, which=which)
    except Exception as e:  # noqa: BLE001
        print(f"error   {e}", file=sys.stderr)
        return 1
    print(c)
    return 0
