"""Magnetic structure and flux-surface verdicts for the sources that are not stellarator-type
confinement fields: tokamak coil sets (axisymmetric PF/CS/TF sets and 3D perturbation coils),
other concepts (levitated dipoles, HBT-EP), axisymmetric verification sets.

    from torux import nonstell
    rec = nonstell.classify(source)          # JSON-serialisable dict: verdict, evidence, argument

Every field value used for a claim comes from the independent reference evaluator
(torux.reference.fast64, and torux.reference.trace for one-turn maps); torux.field.FieldEngine is used
only as a second, independent check of the critical points.

Definitions (used by every verdict)
-----------------------------------
Vacuum region Omega: R^3 minus the current-carrying elements (filaments, polygon conductors,
the z axis when a ToroidalField part is present).
Toroidal (rotational) flux surface: a C^1 embedded torus T in Omega, tangent to B, B != 0 on T,
meeting every half plane phi = const in one Jordan curve C_phi that bounds the disc D_phi; the
closed solid torus V = U D_phi has B.grad(phi) > 0 on V (or < 0 on V): field lines inside the
surface progress toroidally, the period map P: D_0 -> D_0 (phi -> phi + 2 pi) is defined
(the frame of the solver's tori, torux.torus).  The enclosed toroidal flux is the flux of B
through D_phi.  "Vacuum interior": V contains no current element.

Theorems (proof sketches in THEOREMS; each record names the ones it uses)
-------------------------------------------------------------------------
AMP  (Ampere, any source):  a toroidal flux surface with vacuum interior needs
     I_link(V) := sum_k I_k Lk(core(V), coil_k) > 0 (sign of B_phi); quantitatively
     mu0 I_link = mean over toroidal turns of the line integral of B along a field line on T.
SHADOW (consequence of AMP): with U a connected component of the (R, Z) half plane minus the
     projections ("shadows") of all coils, every horizontal circle in U links the same current
     I_circ(U); a core in U x S^1 links n_phi I_circ(U) + sum_holes n_h I_hole(h) with
     I_hole(h) = sum over the coils whose shadow lies in the hole h of I_k w_z(k) (w_z = winding
     about the z axis).  I_circ(U) = 0 and I_hole = 0 for every hole => no toroidal flux surface
     with V in U x S^1.
AXI  (axisymmetric vacuum): psi = R A_phi, Delta* psi = -mu0 R J_phi; in Omega Delta* psi = 0,
     strong maximum principle: no interior extrema, every critical point is a saddle (Hessian
     traceless); every closed psi contour encloses current; a toroidal flux surface with
     vacuum interior is impossible (AMP + Barbalat + maximum principle, or an isochronous
     family -- excluded where iota varies).  Closed nested surfaces exist only around conductors.
INDEX (one-turn map): P is an orientation- and flux-preserving homeomorphism of D_0; Lefschetz:
     the fixed points of P in D_0 (period-1 closed field lines) have total index +1 (boundary
     saddles count 0); ordinary saddles (positive eigenvalues, residue < 0) have index -1.  If
     every period-1 line in a region is an ordinary saddle, no toroidal flux surface has its
     section in that region.  P without fixed points in a region: no surface there (Brouwer).
EXC  (excursion): every point of D_0 lies on a line that makes unboundedly many toroidal turns
     with B_phi of one sign.  If no line started from a grid of spacing h in the section completes
     one turn, D_0 contains no grid point: its inradius is < h / sqrt 2.  (Numerical.)
SYM  (planar TF symmetry): planar coils in meridional planes, N-fold rotation symmetric and
     mirror symmetric: B is normal to every coil plane and mid plane, so P = identity: every
     line through the bore is closed after one turn; every closed curve in the bore sweeps a
     (degenerate, iota = 0) flux surface.
"""
from __future__ import annotations

import os as _os_env

import math
import time

import numpy as np
from numba import njit, prange

from .source import MU0, Source, Rings, AxisymPolygons, ToroidalField
from .reference import fast64
from .reference.geometry import ear_clip, signed_area
from .reference.fast64 import eval_point_B, eval_point

TWO_PI = 2.0 * math.pi
VERSION = 4          # bumped when a change alters verdicts or evidence
AXI_VERSION = 4      # version of the axisymmetric branch


def current_version(kind):
    return AXI_VERSION if kind == "axisymmetric" else VERSION


def is_current(rec):
    """Is a stored record up to date?  (Axisymmetric records without a toroidal field are unchanged
    since version 3: the version-4 additions only concern sources with a ToroidalField part.)"""
    v = rec.get("version")
    if v is None:
        return False
    if rec.get("kind") == "axisymmetric" and not rec.get("axisym", {}).get("RB") and v >= 3:
        return True
    return v == current_version(rec.get("kind"))

# ============================================================================ scope
IN_SCOPE_KINDS = ("tokamak", "dipole")


def scope(source):
    """Is the source one this module settles?  Its kind is Source.meta['kind'] (torux.load sets it): 'tokamak' (a
    tokamak's own coils, perturbation coils) and 'dipole' are; a stellarator-type confinement field is not."""
    kind = (getattr(source, "meta", None) or {}).get("kind", "stellarator")
    if kind in IN_SCOPE_KINDS:
        return True, ""
    return False, "a stellarator-type confinement field: its surfaces are sought as tori (torux.analysis)"


# ============================================================================ geometry
def closed_curves(source, n_fourier=None):
    """Every closed filament of the source as a closed polygon (points (m, 3), no repeated end).

    Fourier curves are sampled at n_fourier points (default 64 (order+1), at least 512); the
    linking numbers below are topological, so the polygon is adequate wherever the test loops
    keep a distance larger than its chord error from the curve."""
    out = []
    for ip, p in enumerate(source.parts):
        if p.kind == "fourier":
            n = n_fourier or max(512, 64 * (p.order + 1))
            x, _ = p.points(np.arange(n) / n)
            for i in range(p.n):
                out.append(dict(pts=x[i], I=float(p.current[i]), part=ip, index=i, kind="fourier"))
        elif p.kind == "polyline":
            for i in range(p.n):
                out.append(dict(pts=p.loop(i).copy(), I=float(p.current[i]), part=ip, index=i, kind="polyline"))
        elif p.kind == "rings":
            n = 512
            t = TWO_PI * np.arange(n) / n
            for i in range(p.n):
                pts = np.stack([p.R[i] * np.cos(t), p.R[i] * np.sin(t), np.full(n, p.Z[i])], -1)
                out.append(dict(pts=pts, I=float(p.current[i]), part=ip, index=i, kind="ring"))
    return out


def toroidal_winding(pts):
    """Winding number of a closed polygon about the z axis (requires it to avoid the axis)."""
    ph = np.arctan2(pts[:, 1], pts[:, 0])
    d = np.diff(np.append(ph, ph[0]))
    d = (d + np.pi) % TWO_PI - np.pi
    return int(np.round(d.sum() / TWO_PI))


def circle_fit(pts):
    """Best plane and circle through a closed polygon: centre, unit normal, radius, max
    relative deviation from the circle, planarity (smallest / largest singular value)."""
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    nrm = vt[2]
    r = np.linalg.norm((pts - c) - np.outer((pts - c) @ nrm, nrm), axis=1)
    rad = float(r.mean())
    return dict(center=c, normal=nrm, radius=rad, dev=float(np.max(np.abs(r - rad)) / rad),
                planarity=float(s[2] / s[0]))


def axisym_view(source, tol=1e-12):
    """The source as rings / polygons / RB if every part is axisymmetric about z.

    Returns None for 3D sources.  Fourier and polyline loops are accepted only if they are
    horizontal circles centred on z to relative deviation tol (returned in 'deviation');
    with tol > 1e-12 the view is the axisymmetric projection of a near-axisymmetric source."""
    R, Z, I, polys, RB, dev = [], [], [], [], 0.0, 0.0
    for p in source.parts:
        if p.kind == "rings":
            R += list(p.R); Z += list(p.Z); I += list(p.current)
        elif p.kind == "axipoly":
            for i in range(p.n):
                polys.append((p.vertices[p.start[i]:p.start[i + 1]].copy(), float(p.current[i])))
        elif p.kind == "toroidal":
            RB += p.RB
        elif p.kind in ("fourier", "polyline"):
            for c in closed_curves(Source(id="_", parts=[p])):
                x = c["pts"]
                Rr = np.hypot(x[:, 0], x[:, 1])
                Rm, Zm = Rr.mean(), x[:, 2].mean()
                d = max(np.abs(Rr - Rm).max() / Rm, np.abs(x[:, 2] - Zm).max() / Rm,
                        np.linalg.norm(x[:, :2].mean(0)) / Rm)
                w = toroidal_winding(x)
                if p.kind == "polyline":
                    # a polygon is never a circle: accept only if the vertex count makes the
                    # chord error below tol (it never does for the data's loops)
                    d = max(d, 1 - math.cos(math.pi / len(x)))
                if d > tol or abs(w) != 1:
                    return None
                dev = max(dev, d)
                R.append(Rm); Z.append(Zm); I.append(c["I"] * w)
        elif p.kind in ("segments", "dipoles"):
            return None
    if source.meta.get("analytic_parts"):
        return None
    return dict(R=np.array(R, float), Z=np.array(Z, float), I=np.array(I, float), polys=polys, RB=float(RB),
                deviation=float(dev))


def triangle_rule(order):
    """Collapsed (Duffy) Gauss rule on the unit triangle: points (q, 2) barycentric-free
    (u, v) -> a + u (b - a) + u v (c - b), weights summing to 1 (fraction of the area)."""
    x, w = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (x + 1); w = 0.5 * w
    U, V = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w) * U * 2.0
    return U.ravel(), V.ravel(), W.ravel()


def polygons_to_rings(polys, h, order=4):
    """Rings approximating uniform-current polygons for grid evaluation: one ring at the
    centroid for polygons smaller than h, a triangle Gauss rule of the given order otherwise."""
    R, Z, I = [], [], []
    rules = {1: (np.array([2.0 / 3.0]), np.array([0.5]), np.array([1.0]))}    # centroid rule
    for V, cur in polys:
        ext = np.ptp(V, axis=0).max()
        if ext < h:
            A = signed_area(V)
            # area centroid of the polygon
            x, y = V[:, 0], V[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            Ac = 0.5 * cr.sum()
            cx = ((x + x1) * cr).sum() / (6 * Ac); cy = ((y + y1) * cr).sum() / (6 * Ac)
            R.append(cx); Z.append(cy); I.append(cur)
            continue
        Atot = abs(signed_area(V))
        if len(V) > 48:
            # many-vertex region (merged mesh regions): lattice of spacing h/3, cell centres inside,
            # weights rescaled to the exact area (accurate beyond ~h from the polygon)
            hl = h / 3.0
            r0, z0 = V.min(0); r1, z1 = V.max(0)
            gr = np.arange(r0 + 0.5 * hl, r1, hl); gz = np.arange(z0 + 0.5 * hl, z1, hl)
            GR, GZ = np.meshgrid(gr, gz, indexing="ij")
            ins = _points_in_polygon(V, GR, GZ)
            if ins.sum() >= 1:
                R += list(GR[ins]); Z += list(GZ[ins]); I += list(np.full(int(ins.sum()), cur / ins.sum()))
                continue
        tris = ear_clip(V)
        for (i0, i1, i2) in tris:
            a, b, c = V[i0], V[i1], V[i2]
            At = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            size = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
            # rule order by triangle size: the grid only needs accuracy beyond ~h from the conductor
            q = 1 if size < h else (2 if size < 3 * h else order)
            Uq, Vq, Wq = rules[q] if q in rules else rules.setdefault(q, triangle_rule(q))
            P = a[None, :] + Uq[:, None] * (b - a)[None, :] + (Uq * Vq)[:, None] * (c - b)[None, :]
            R += list(P[:, 0]); Z += list(P[:, 1]); I += list(cur * Wq * At / Atot)
    return np.array(R), np.array(Z), np.array(I)


# ============================================================================ linking numbers
def lk_disk_crossings(pts, R0, Z0):
    """Linking number of the closed polygon pts with the horizontal circle C(R0, Z0) (oriented
    along +phi): the signed number of crossings of the polygon through the flat disc
    {z = Z0, R < R0} (+1 for crossings in +z).  Exact (integer arithmetic on signs); a vertex
    lying on the plane is resolved by symbolic perturbation Z0 -> Z0 + 0."""
    z = pts[:, 2] - Z0
    z = np.where(z == 0.0, 1e-300, z)          # symbolic perturbation: plane slightly below
    z1 = np.roll(z, -1)
    cr = (z < 0) != (z1 < 0)
    if not cr.any():
        return 0
    a = pts[cr]; b = np.roll(pts, -1, axis=0)[cr]
    t = z[cr] / (z[cr] - z1[cr])
    p = a + t[:, None] * (b - a)
    inside = np.hypot(p[:, 0], p[:, 1]) < R0
    sgn = np.where(z1[cr] > z[cr], 1, -1)
    return int(sgn[inside].sum())


def gauss_linking(P, Q):
    """Gauss linking integral of two closed polygons, exact for polygons: for every pair of
    segments the double integral is the signed area of the spherical quadrilateral swept by
    the unit vector from one segment to the other (Klenin-Langowski), evaluated here as two
    Van Oosterom-Strackee triangles with atan2 (well conditioned); the total / 4 pi is an
    integer up to rounding."""
    a1, a2 = P, np.roll(P, -1, axis=0)
    b1, b2 = Q, np.roll(Q, -1, axis=0)
    p1 = a1[:, None, :]; p2 = a2[:, None, :]; p3 = b1[None, :, :]; p4 = b2[None, :, :]

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    u1, u2, u3, u4 = unit(p3 - p1), unit(p4 - p1), unit(p4 - p2), unit(p3 - p2)

    def vos(a, b, c):
        num = (a * np.cross(b, c)).sum(-1)
        den = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
        return 2.0 * np.arctan2(num, den)
    om = vos(u1, u2, u3) + vos(u1, u3, u4)
    return float(-om.sum() / (4 * math.pi))


def horizontal_circle(R0, Z0, n=256):
    t = TWO_PI * np.arange(n) / n
    return np.stack([R0 * np.cos(t), R0 * np.sin(t), np.full(n, Z0)], -1)


def ampere_circle(F, R0, Z0, n=256):
    """(1/mu0) times the circulation of the reference field around C(R0, Z0), by the periodic
    trapezoid rule with n and n/2 nodes (spectral; the difference is the error estimate).
    By Ampere's law it equals the current linking the circle."""
    def circ(m):
        t = TWO_PI * np.arange(m) / m
        x = np.stack([R0 * np.cos(t), R0 * np.sin(t), np.full(m, Z0)], -1)
        B = F.evaluate(x, grad=False)["B"]
        Bphi = -B[:, 0] * np.sin(t) + B[:, 1] * np.cos(t)
        return float(Bphi.sum() * R0 * TWO_PI / m / MU0)
    a, b = circ(n), circ(n // 2)
    return a, abs(a - b)


# ============================================================================ shadows
def coil_box(curves, view=None, margin=0.15):
    pts = []
    for c in curves:
        x = c["pts"]
        pts.append(np.stack([np.hypot(x[:, 0], x[:, 1]), x[:, 2]], -1))
    if view is not None:
        if len(view["R"]):
            pts.append(np.stack([view["R"], view["Z"]], -1))
        for V, _ in view["polys"]:
            pts.append(V)
    P = np.concatenate(pts)
    Rmax, Zmin, Zmax = P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    L = max(Rmax, Zmax - Zmin)
    return dict(R=(max(1e-3 * L, P[:, 0].min() - margin * L), Rmax + margin * L),
                Z=(Zmin - margin * L, Zmax + margin * L), L=float(L))


def shadow_components(curves, box, n=200, dilate=2):
    """Connected components of the (R, Z) box minus the dilated coil shadows.

    Returns dict(Rg, Zg, shadow mask, labels, comps=[{label, area_frac, rep (R, Z),
    lk per coil at rep, I_circ, checks at other points}], holes=[{coils, I_hole}])."""
    from scipy import ndimage
    Rg = np.linspace(box["R"][0], box["R"][1], n)
    Zg = np.linspace(box["Z"][0], box["Z"][1], n)
    hR, hZ = Rg[1] - Rg[0], Zg[1] - Zg[0]
    sh = np.zeros((n, n), bool)
    owner = [[] for _ in curves]
    for k, c in enumerate(curves):
        x = c["pts"]
        xx = np.concatenate([x, x[:1]])
        seg = np.linalg.norm(np.diff(xx, axis=0), axis=1)
        m = np.maximum(1, np.ceil(seg / (0.25 * min(hR, hZ)))).astype(int)
        P = np.concatenate([xx[i] + np.outer(np.arange(m[i]) / m[i], xx[i + 1] - xx[i]) for i in range(len(x))])
        Rs, Zs = np.hypot(P[:, 0], P[:, 1]), P[:, 2]
        i = np.clip(np.round((Rs - Rg[0]) / hR).astype(int), 0, n - 1)
        j = np.clip(np.round((Zs - Zg[0]) / hZ).astype(int), 0, n - 1)
        sh[i, j] = True
        owner[k] = (i, j)
    shd = ndimage.binary_dilation(sh, iterations=dilate)
    lab, nl = ndimage.label(~shd)
    slab, ns = ndimage.label(shd, structure=np.ones((3, 3)))
    dist = ndimage.distance_transform_edt(~shd, sampling=(hR, hZ))
    comps = []
    rng = np.random.default_rng(0)
    for L in range(1, nl + 1):
        msk = lab == L
        frac = float(msk.mean())
        if frac < 2e-4:
            continue
        d = np.where(msk, dist, -1)
        ij = np.unravel_index(np.argmax(d), d.shape)
        rep = (float(Rg[ij[0]]), float(Zg[ij[1]]))
        lks = [lk_disk_crossings(c["pts"], rep[0], rep[1]) for c in curves]
        Ic = float(sum(l * c["I"] for l, c in zip(lks, curves)))
        # the same current links every horizontal circle in U (SHADOW): check at other points
        idx = np.argwhere(msk & (dist > 0.5 * dist[ij]))
        others = []
        for q in idx[rng.choice(len(idx), size=min(4, len(idx)), replace=False)]:
            Rq, Zq = float(Rg[q[0]]), float(Zg[q[1]])
            others.append(float(sum(lk_disk_crossings(c["pts"], Rq, Zq) * c["I"] for c in curves)))
        touches = bool(msk[0, :].any() or msk[-1, :].any() or msk[:, 0].any() or msk[:, -1].any())
        comps.append(dict(label=int(L), area_frac=frac, rep=rep, inradius=float(dist[ij]),
                          lk=[int(v) for v in lks], I_circ=Ic, I_circ_other_points=others,
                          touches_box=touches))
    holes = []
    for L in range(1, ns + 1):
        msk = slab == L
        ks = [k for k, (i, j) in enumerate(owner) if msk[i, j].any()]
        Ih = float(sum(curves[k]["I"] * toroidal_winding(curves[k]["pts"]) for k in ks))
        holes.append(dict(coils=ks, I_hole=Ih))
    return dict(Rg=Rg, Zg=Zg, shadow=shd, labels=lab, comps=comps, holes=holes, dist=dist)


# ============================================================================ tracing (numba)
class ScanField:
    """Plain float64 Biot-Savart of straight segments (+ the analytic 1/R toroidal field) for the
    field-line scans: Fourier curves and rings are replaced by fine polygons (n_poly points per
    loop, chord error ~ (pi / n_poly)^2 / 2 of the loop size).  Used only to decide which lines
    complete a toroidal turn and to seed fixed points; every reported field value and every
    refined period-1 line comes from torux.reference."""

    def __init__(self, source, n_poly=None):
        A, Bv, I = [], [], []
        RB = 0.0
        if n_poly is None:
            ords = [p.order for p in source.parts if p.kind == "fourier"]
            n_poly = int(min(1024, max(256, 48 * (max(ords) + 1)))) if ords else 1024
        for c in closed_curves(source, n_fourier=n_poly):
            x = c["pts"]
            if c["kind"] == "ring":
                n = n_poly
                t = TWO_PI * np.arange(n) / n
                R0 = np.hypot(x[0, 0], x[0, 1]); Z0 = x[0, 2]
                # circumradius with the area (dipole moment) of the circle: polygon error O(n^-4)
                R0 = R0 * math.sqrt((TWO_PI / n) / math.sin(TWO_PI / n))
                x = np.stack([R0 * np.cos(t), R0 * np.sin(t), np.full(n, Z0)], -1)
            A.append(x); Bv.append(np.roll(x, -1, axis=0)); I.append(np.full(len(x), c["I"]))
        for p in source.parts:
            if p.kind == "segments":
                A.append(p.a); Bv.append(p.b); I.append(p.current)
            elif p.kind == "toroidal":
                RB += p.RB
            elif p.kind in ("dipoles", "axipoly"):
                raise ValueError("ScanField: part kind %s not supported" % p.kind)
        self.a = np.ascontiguousarray(np.concatenate(A)) if A else np.zeros((0, 3))
        self.b = np.ascontiguousarray(np.concatenate(Bv)) if Bv else np.zeros((0, 3))
        self.w = np.ascontiguousarray(np.concatenate(I) * 1e-7) if I else np.zeros(0)
        self.packed = (self.a, self.b, self.w, float(RB))


@njit(cache=True, error_model="numpy", fastmath=False)
def _scan_B(P, x0, x1, x2):
    a, b, w, RB = P
    bx = 0.0; by = 0.0; bz = 0.0
    for k in range(w.shape[0]):
        r1x = x0 - a[k, 0]; r1y = x1 - a[k, 1]; r1z = x2 - a[k, 2]
        r2x = x0 - b[k, 0]; r2y = x1 - b[k, 1]; r2z = x2 - b[k, 2]
        n1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
        n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
        den = n1 * n2 * (n1 * n2 + r1x * r2x + r1y * r2y + r1z * r2z)
        if den <= 0.0:
            continue
        f = w[k] * (n1 + n2) / den
        bx += f * (r1y * r2z - r1z * r2y)
        by += f * (r1z * r2x - r1x * r2z)
        bz += f * (r1x * r2y - r1y * r2x)
    if RB != 0.0:
        r2 = x0 * x0 + x1 * x1
        bx += -RB * x1 / r2
        by += RB * x0 / r2
    return bx, by, bz


@njit(cache=True, error_model="numpy")
def _unit_field(P, x0, x1, x2, sig):
    bx, by, bz = _scan_B(P, x0, x1, x2)
    b = math.sqrt(bx * bx + by * by + bz * bz)
    if b == 0.0 or b != b:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    R2 = x0 * x0 + x1 * x1
    bphi = (-x1 * bx + x0 * by) / math.sqrt(R2)
    return sig * bx / b, sig * by / b, sig * bz / b, sig * bphi / b, b


@njit(cache=True, error_model="numpy")
def _dp_step(P, x, k1, h, sig, y, k7):
    """One Dormand-Prince 5(4) step of size h along sig B/|B| from x (k1 = direction at x).
    Writes y (5th-order solution) and k7 (direction at y); returns (error, min over the stages
    of mon * B_phi / |B| is left to the caller via the returned stage values)."""
    k2 = np.empty(3); k3 = np.empty(3); k4 = np.empty(3); k5 = np.empty(3); k6 = np.empty(3); t = np.empty(3)
    for q in range(3):
        t[q] = x[q] + h * (1.0 / 5) * k1[q]
    k2[0], k2[1], k2[2], fp2, _ = _unit_field(P, t[0], t[1], t[2], sig)
    for q in range(3):
        t[q] = x[q] + h * (3.0 / 40 * k1[q] + 9.0 / 40 * k2[q])
    k3[0], k3[1], k3[2], fp3, _ = _unit_field(P, t[0], t[1], t[2], sig)
    for q in range(3):
        t[q] = x[q] + h * (44.0 / 45 * k1[q] - 56.0 / 15 * k2[q] + 32.0 / 9 * k3[q])
    k4[0], k4[1], k4[2], fp4, _ = _unit_field(P, t[0], t[1], t[2], sig)
    for q in range(3):
        t[q] = x[q] + h * (19372.0 / 6561 * k1[q] - 25360.0 / 2187 * k2[q] + 64448.0 / 6561 * k3[q] - 212.0 / 729 * k4[q])
    k5[0], k5[1], k5[2], fp5, _ = _unit_field(P, t[0], t[1], t[2], sig)
    for q in range(3):
        t[q] = x[q] + h * (9017.0 / 3168 * k1[q] - 355.0 / 33 * k2[q] + 46732.0 / 5247 * k3[q] + 49.0 / 176 * k4[q]
                           - 5103.0 / 18656 * k5[q])
    k6[0], k6[1], k6[2], fp6, _ = _unit_field(P, t[0], t[1], t[2], sig)
    for q in range(3):
        y[q] = x[q] + h * (35.0 / 384 * k1[q] + 500.0 / 1113 * k3[q] + 125.0 / 192 * k4[q] - 2187.0 / 6784 * k5[q]
                           + 11.0 / 84 * k6[q])
    k7[0], k7[1], k7[2], fp7, bb7 = _unit_field(P, y[0], y[1], y[2], sig)
    err = 0.0
    for q in range(3):
        e = h * (71.0 / 57600 * k1[q] - 71.0 / 16695 * k3[q] + 71.0 / 1920 * k4[q] - 17253.0 / 339200 * k5[q]
                 + 22.0 / 525 * k6[q] - 1.0 / 40 * k7[q])
        err = max(err, abs(e))
    return err, fp2, fp3, fp4, fp5, fp6, fp7, bb7


@njit(cache=True, error_model="numpy")
def _dphi(x, y):
    d = math.atan2(y[1], y[0]) - math.atan2(x[1], x[0])
    if d > math.pi:
        d -= TWO_PI
    elif d < -math.pi:
        d += TWO_PI
    return d


@njit(cache=True, error_model="numpy")
def _excursion_one(P, x0, x1, x2, sig, dphi_max, tol, hmax, nmax, Rlo, Rhi, Zlo, Zhi, out):
    """Arc-length Dormand-Prince 5(4) along sig * B / |B| from x; stops when the toroidal
    component of the direction changes sign (reversal), when |accumulated phi| reaches dphi_max
    (the last step is shortened by a secant iteration so that the line ends exactly on the
    section phi0 +- dphi_max), when the line leaves the box or after nmax steps.
    out = [dphi, code, x, y, z, steps, s]; code: 0 reversal, 1 completed, 2 left box,
    3 step limit, 4 null field."""
    x = np.array([x0, x1, x2])
    acc = 0.0
    h = hmax * 0.1
    s = 0.0
    k1 = np.empty(3); k7 = np.empty(3); y = np.empty(3); yb = np.empty(3); k7b = np.empty(3)
    f0, f1, f2, fp, bb = _unit_field(P, x[0], x[1], x[2], sig)
    if bb == 0.0:
        out[0] = 0.0; out[1] = 4.0; out[2] = x[0]; out[3] = x[1]; out[4] = x[2]; out[5] = 0; out[6] = 0
        return
    k1[0] = f0; k1[1] = f1; k1[2] = f2
    mon = 1.0 if fp >= 0.0 else -1.0          # sign of dphi/ds at the start: monitored for reversal
    code = 3
    nst = 0
    for it in range(nmax):
        nst = it
        err, fp2, fp3, fp4, fp5, fp6, fp7, bb7 = _dp_step(P, x, k1, h, sig, y, k7)
        if err <= tol or h < 1e-12 * hmax:
            mn = min(mon * fp2, mon * fp3, mon * fp4, mon * fp5, mon * fp6, mon * fp7)
            d = _dphi(x, y)
            if mn <= 0.0 or bb7 == 0.0:
                code = 0 if bb7 != 0.0 else 4
                acc += d
                for q in range(3):
                    x[q] = y[q]
                break
            if abs(acc + d) >= dphi_max:
                # land on the section: secant (Illinois) on the step length
                ha, ga = 0.0, abs(acc) - dphi_max
                hb, gb = h, abs(acc + d) - dphi_max
                hc = h
                for itr in range(60):
                    hc = hb - gb * (hb - ha) / (gb - ga)
                    _dp_step(P, x, k1, hc, sig, yb, k7b)
                    gc = abs(acc + _dphi(x, yb)) - dphi_max
                    if abs(gc) < 1e-15 * dphi_max:
                        break
                    if (gc > 0) == (gb > 0):
                        ga = 0.5 * ga
                    else:
                        ha, ga = hb, gb
                    hb, gb = hc, gc
                for q in range(3):
                    x[q] = yb[q]
                acc = math.copysign(dphi_max, acc + d)
                s += hc
                code = 1
                break
            acc += d
            s += h
            for q in range(3):
                x[q] = y[q]
                k1[q] = k7[q]
            Rn = math.sqrt(x[0] * x[0] + x[1] * x[1])
            if Rn < Rlo or Rn > Rhi or x[2] < Zlo or x[2] > Zhi:
                code = 2
                break
        fac = 0.9 * (tol / max(err, 1e-300)) ** 0.2
        fac = min(4.0, max(0.2, fac))
        h = min(hmax, h * fac)
    out[0] = acc; out[1] = code; out[2] = x[0]; out[3] = x[1]; out[4] = x[2]; out[5] = nst; out[6] = s


@njit(parallel=True, cache=True, error_model="numpy")
def _excursion_many(P, X, sig, dphi_max, tol, hmax, nmax, box):
    n = X.shape[0]
    out = np.zeros((n, 7))
    for i in prange(n):
        o = np.zeros(7)
        _excursion_one(P, X[i, 0], X[i, 1], X[i, 2], sig[i], dphi_max, tol, hmax, nmax,
                       box[0], box[1], box[2], box[3], o)
        for k in range(7):
            out[i, k] = o[k]
    return out


def excursion_scan(F, Rs, Zs, phi0=0.0, dphi_max=TWO_PI, tol=None, hmax=None, nmax=100000, box=None,
                   both=True):
    """Toroidal excursion of the field lines through (R, phi0, Z) (F: a ScanField): followed
    along +B and -B (both=True) until B_phi reverses.  Returns dict with dphi (n, 2) (signed accumulated phi in
    the B_phi-forward and B_phi-backward directions), code (n, 2), steps."""
    Rs = np.asarray(Rs, float).ravel(); Zs = np.asarray(Zs, float).ravel()
    X = np.stack([Rs * np.cos(phi0), Rs * np.sin(phi0), Zs], -1)
    B = np.array([_scan_B(F.packed, *x) for x in X]).reshape(-1, 3)
    bphi = -B[:, 0] * math.sin(phi0) + B[:, 1] * math.cos(phi0)
    sg = np.where(bphi >= 0, 1.0, -1.0)
    L = box["L"] if box else float(np.max(Rs))
    tol = tol or 3e-10 * L
    hmax = hmax or 0.02 * L
    bx = np.array([box["R"][0] * 0.5, box["R"][1] * 2.0, box["Z"][0] - L, box["Z"][1] + L]) if box else \
        np.array([0.0, 1e300, -1e300, 1e300])
    o1 = _excursion_many(F.packed, X, sg, dphi_max, tol, hmax, nmax, bx)
    res = dict(sign_Bphi=sg, fwd=o1)
    if both:
        o2 = _excursion_many(F.packed, X, -sg, dphi_max, tol, hmax, nmax, bx)
        res["bwd"] = o2
    return res


# ============================================================================ one-turn map
def one_turn(F, R, Z, phi0=0.0, h=None, nturns=1, doubling=True):
    """Reference period map (torux.reference.trace, Gauss-Legendre IRK in phi) of the lines
    through (R, phi0, Z): returns R1, Z1 (n,), err (step-doubling, NaN without doubling), lost."""
    from .reference import trace as RT
    R = np.atleast_1d(np.asarray(R, float)); Z = np.atleast_1d(np.asarray(Z, float))
    out = RT.poincare(F, R, Z, [phi0], nturns, phi0=phi0, h=h or TWO_PI / 96, stages=6, tol=1e-14,
                      doubling=doubling)
    err = out["err"][:, -1, 0] if doubling else np.full(len(R), np.nan)
    return out["R"][:, -1, 0], out["Z"][:, -1, 0], err, out["lost"] >= 0


# ============================================================================ axisymmetric analysis
def _points_in_polygon(V, R, Z):
    """Even-odd rule, vectorised over points."""
    inside = np.zeros(R.shape, bool)
    x, y = V[:, 0], V[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    for a, b, c, d in zip(x, y, x1, y1):
        cond = (b > Z) != (d > Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a + (Z - b) * (c - a) / (d - b)
        inside ^= cond & (R < xc)
    return inside


def grid_source(view, h):
    """Rings-only Source for grid evaluation (polygons smaller than h -> centroid ring)."""
    R, Z, I = [view["R"]], [view["Z"]], [view["I"]]
    if view["polys"]:
        r, z, i = polygons_to_rings(view["polys"], h)
        R.append(r); Z.append(z); I.append(i)
    parts = [Rings(np.concatenate(R), np.concatenate(Z), np.concatenate(I))]
    if view["RB"]:
        parts.append(ToroidalField(view["RB"]))
    return Source(id="_grid", parts=parts)


def exact_source(view):
    parts = []
    if len(view["R"]):
        parts.append(Rings(view["R"], view["Z"], view["I"]))
    if view["polys"]:
        V = np.concatenate([v for v, _ in view["polys"]])
        st = np.cumsum([0] + [len(v) for v, _ in view["polys"]])
        parts.append(AxisymPolygons(V, st, np.array([c for _, c in view["polys"]])))
    if view["RB"]:
        parts.append(ToroidalField(view["RB"]))
    return Source(id="_exact", parts=parts)


def psi_on(F, R, Z):
    """psi = R A_phi, B_R, B_Z at (R, phi = 0, Z) from a reference Field."""
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    pts = np.stack([R.ravel(), np.zeros(R.size), Z.ravel()], -1)
    r = F.evaluate(pts, grad=False)
    psi = (R.ravel() * r["A"][:, 1]).reshape(R.shape)
    return psi, r["B"][:, 0].reshape(R.shape), r["B"][:, 2].reshape(R.shape)


def conductor_ids(view, Rg, Zg, pad):
    """Cell labels: -1 vacuum, j >= 0 conductor j (rings first, then polygons), dilated by pad
    cells (filaments: the cells within pad of the ring point)."""
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    hR, hZ = Rg[1] - Rg[0], Zg[1] - Zg[0]
    lab = -np.ones(RR.shape, int)
    k = 0
    for R0, Z0 in zip(view["R"], view["Z"]):
        m = (np.abs(RR - R0) <= pad * hR + 1e-12) & (np.abs(ZZ - Z0) <= pad * hZ + 1e-12)
        lab[m & (lab < 0)] = k
        k += 1
    from scipy import ndimage
    for V, _ in view["polys"]:
        # only the grid nodes in the polygon's bounding box (+ pad) are tested
        i0 = max(0, int(np.floor((V[:, 0].min() - Rg[0]) / hR)) - pad - 1)
        i1 = min(len(Rg), int(np.ceil((V[:, 0].max() - Rg[0]) / hR)) + pad + 2)
        j0 = max(0, int(np.floor((V[:, 1].min() - Zg[0]) / hZ)) - pad - 1)
        j1 = min(len(Zg), int(np.ceil((V[:, 1].max() - Zg[0]) / hZ)) + pad + 2)
        m = np.zeros(RR.shape, bool)
        m[i0:i1, j0:j1] = _points_in_polygon(V, RR[i0:i1, j0:j1], ZZ[i0:i1, j0:j1])
        # cells crossed by the boundary
        Vc = np.concatenate([V, V[:1]])
        for a, b in zip(Vc[:-1], Vc[1:]):
            nseg = int(np.ceil(np.linalg.norm(b - a) / (0.25 * min(hR, hZ)))) + 1
            P = a + np.outer(np.linspace(0, 1, nseg), b - a)
            i = np.clip(np.round((P[:, 0] - Rg[0]) / hR).astype(int), 0, len(Rg) - 1)
            j = np.clip(np.round((P[:, 1] - Zg[0]) / hZ).astype(int), 0, len(Zg) - 1)
            m[i, j] = True
        if pad:
            sub = ndimage.binary_dilation(m[i0:i1, j0:j1], iterations=pad)
            m[i0:i1, j0:j1] = sub
        lab[m & (lab < 0)] = k
        k += 1
    return lab


def grid_extrema(psi, cond):
    """Strict local extrema of psi over the 8 neighbours on interior grid nodes; split into
    those on conductor cells and those in vacuum."""
    c = psi[1:-1, 1:-1]
    nb = [psi[1 + di:psi.shape[0] - 1 + di, 1 + dj:psi.shape[1] - 1 + dj]
          for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    mx = np.all([c > x for x in nb], axis=0)
    mn = np.all([c < x for x in nb], axis=0)
    out = []
    for kind, m in (("max", mx), ("min", mn)):
        for i, j in np.argwhere(m):
            out.append(dict(i=int(i + 1), j=int(j + 1), kind=kind, conductor=int(cond[i + 1, j + 1])))
    return out


def cell_index(BR, BZ):
    """Poincare index of the vector field (B_R, B_Z) around every grid cell (sum of the wrapped
    angle increments along the 4 edges / 2 pi): +1 around an O-point or a filament, -1 around
    an X-point, 0 elsewhere (exact as long as the field turns by < pi along each edge)."""
    a = np.arctan2(BZ, BR)
    c = [a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]]
    tot = 0.0
    for k in range(4):
        d = c[(k + 1) % 4] - c[k]
        tot = tot + (d + np.pi) % TWO_PI - np.pi
    return np.round(tot / TWO_PI).astype(int)


def refine_null(Fe, R0, Z0, scale, maxit=40, Fa=None, exact_its=3):
    """Newton on (B_R, B_Z)(R, Z) = 0 with the reference field and its gradient (first with the
    cheap grid approximation Fa if given, then exact_its iterations with the exact reference Fe).
    Returns dict(R, Z, |Bpol|/scale, converged, Hessian eigenvalues of psi, its trace/norm)."""
    x = np.array([R0, Z0], float)
    if Fa is not None:
        r0 = refine_null(Fa, R0, Z0, scale, maxit=maxit, Fa=None)
        if r0["converged"]:
            x = np.array([r0["R"], r0["Z"]])
        maxit = exact_its
    ok = False
    for it in range(maxit):
        r = Fe.evaluate(np.array([[x[0], 0.0, x[1]]]), grad=True)
        B, dB = r["B"][0], r["dB"][0]
        f = np.array([B[0], B[2]])
        J = np.array([[dB[0, 0], dB[0, 2]], [dB[2, 0], dB[2, 2]]])
        try:
            dx = -np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break
        lim = 0.05 * max(x[0], 1e-3)
        if np.linalg.norm(dx) > lim:
            dx *= lim / np.linalg.norm(dx)
        x = x + dx
        if np.linalg.norm(dx) < 1e-14 * max(1.0, abs(x[0])) or (Fa is not None and np.linalg.norm(f) < 1e-14 * scale):
            ok = True
            break
    r = Fe.evaluate(np.array([[x[0], 0.0, x[1]]]), grad=True)
    B, dB, A = r["B"][0], r["dB"][0], r["A"][0]
    Rr = x[0]
    H = np.array([[Rr * dB[2, 0], Rr * dB[2, 2]], [Rr * dB[2, 2], -Rr * dB[0, 2]]])
    Hs = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(Hs)
    return dict(R=float(x[0]), Z=float(x[1]), psi=float(Rr * A[1]), Bpol_rel=float(math.hypot(B[0], B[2]) / scale),
                Bphi=float(B[1]), converged=bool(ok), hess_eig=[float(v) for v in ev],
                hess_trace_rel=float(abs(np.trace(H)) / max(np.abs(ev).max(), 1e-300)),
                det_hess=float(np.linalg.det(Hs)))


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        r = a
        while p[r] != r:
            r = p[r]
        while p[a] != r:
            p[a], a = r, p[a]
        return r


def merge_families(psi, cond, Rg, Zg, RB, sign, n_fil=0):
    """Nested families of closed psi contours (merge tree of the super-level sets of sign*psi).

    psi -> 0 on the z axis and at infinity (psi = R A_phi, dipole decay), so every component of
    {sign*psi > c} with c > 0 is bounded: its contours are closed.  Cells are added in decreasing
    sign*psi; a component is born at a grid maximum (on a conductor, by the maximum principle);
    the family of contours around its conductors ends where it merges with another component:
    at a vacuum saddle of psi ('saddle': the separatrix level), on a conductor cell ('conductor':
    the contour touches another conductor), or at the level 0 ('psi0': the contours reach the
    axis / infinity).  Each record: conductors enclosed, inner and outer levels, cause, area and
    toroidal flux RB * int dR dZ / R of the interior of the outermost closed contour (within the
    box: a lower bound when 'touched_box')."""
    nR, nZ = psi.shape
    hR, hZ = Rg[1] - Rg[0], Zg[1] - Zg[0]
    v = (sign * psi).ravel()
    order = np.argsort(-v, kind="stable")
    uf = _UF(nR * nZ)
    added = np.zeros(nR * nZ, bool)
    info = {}
    fams = []
    cid = cond.ravel()
    Rc = np.repeat(Rg, nZ)
    bnd = np.zeros((nR, nZ), bool); bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
    bnd = bnd.ravel()

    def close(r, level, cause, cell):
        d = info[r]
        if not d["cond"]:
            return
        fams.append(dict(sign=int(sign), conductors=sorted(d["cond"]), level_inner=float(sign * d["top"]),
                         level_outer=float(sign * level), cause=cause,
                         cell=[float(Rg[cell // nZ]), float(Zg[cell % nZ])] if cell is not None else None,
                         area=float(d["area"]), tor_flux=float(RB * d["aor"]), touched_box=bool(d["touch"])))

    for c in order:
        lev = v[c]
        if lev <= 0.0:
            for r in list(info):
                if uf.find(r) == r:
                    close(r, 0.0, "psi0", None)
            break
        i, j = divmod(int(c), nZ)
        roots = set()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a_, b_ = i + di, j + dj
            if 0 <= a_ < nR and 0 <= b_ < nZ and added[a_ * nZ + b_]:
                roots.add(uf.find(a_ * nZ + b_))
        added[c] = True
        if not roots:
            info[c] = dict(top=lev, cond=set(), touch=False, area=0.0, aor=0.0)
            r = c
        elif len(roots) == 1:
            r = roots.pop()
            uf.p[c] = r
        else:
            rl = list(roots)
            # a merge on a conductor cell is a contour touching a conductor, not a vacuum saddle
            cause = "saddle" if cid[c] < 0 else ("conductor" if cid[c] >= n_fil else "saddle")
            for q in rl:
                close(q, lev, cause, c)
            r = max(rl, key=lambda q: info[q]["area"])
            d = info[r]
            for q in rl:
                if q == r:
                    continue
                uf.p[q] = r
                e = info.pop(q)
                d["cond"] |= e["cond"]; d["touch"] = d["touch"] or e["touch"]
                d["area"] += e["area"]; d["aor"] += e["aor"]; d["top"] = max(d["top"], e["top"])
            uf.p[c] = r
        d = info[r]
        d["area"] += hR * hZ
        d["aor"] += hR * hZ / Rc[c]
        if cid[c] >= 0:
            d["cond"].add(int(cid[c]))
        if bnd[c]:
            d["touch"] = True
    else:
        for r in list(info):
            if uf.find(r) == r:
                close(r, float(v[order[-1]]), "grid_min", None)
    return fams


def analyse_axisym(source, view, n=None, engine_check=True, verbose=False, margin=0.4):
    """Axisymmetric vacuum structure: psi on a grid (reference field), maximum-principle check,
    critical points (refined by Newton on the exact reference field, checked with the solver's
    engine), nested families of closed contours around conductors with their separatrices and
    toroidal fluxes."""
    t0 = time.time()
    ncond = len(view["R"]) + len(view["polys"])
    if n is None:
        n = 260 if ncond < 300 else (200 if ncond < 2500 else 150)
    box = coil_box([], view, margin=margin)
    Rg = np.linspace(box["R"][0], box["R"][1], n)
    Zg = np.linspace(box["Z"][0], box["Z"][1], n)
    h = max(Rg[1] - Rg[0], Zg[1] - Zg[0])
    Fg = fast64.Field(grid_source(view, h))
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    psi, BR, BZ = psi_on(Fg, RR, ZZ)
    t_grid = time.time() - t0
    cond = conductor_ids(view, Rg, Zg, pad=2 if any(len(V) > 48 for V, _ in view["polys"]) else 1)
    ext = grid_extrema(psi, cond)
    vac_ext = [e for e in ext if e["conductor"] < 0]
    idx = cell_index(BR, BZ)
    # a cell is in vacuum if none of its 4 corner nodes is a conductor node
    cc = np.maximum.reduce([cond[:-1, :-1], cond[1:, :-1], cond[1:, 1:], cond[:-1, 1:]])
    vac_cells = cc < 0
    plus_vac = np.argwhere((idx > 0) & vac_cells)
    minus_vac = np.argwhere((idx < 0) & vac_cells)
    Bscale = float(np.median(np.hypot(BR, BZ)))
    Fe = fast64.Field(exact_source(view))
    xpts, rejected = [], []
    heavy = sum(len(V) for V, _ in view["polys"]) > 2000      # exact polygon evaluations are expensive
    for (i, j) in minus_vac:
        R0 = 0.5 * (Rg[i] + Rg[i + 1]); Z0 = 0.5 * (Zg[j] + Zg[j + 1])
        x = refine_null(Fe, R0, Z0, Bscale, Fa=Fg if heavy else None)
        x["cell"] = [int(i), int(j)]; x["cell_index"] = int(idx[i, j])
        x["moved_cells"] = float(math.hypot((x["R"] - R0) / h, (x["Z"] - Z0) / h))
        x["in_vacuum"] = bool(not any(_points_in_polygon(V, np.array([x["R"]]), np.array([x["Z"]]))[0]
                                      for V, _ in view["polys"]) and
                              (len(view["R"]) == 0 or np.min(np.hypot(view["R"] - x["R"], view["Z"] - x["Z"])) > 1e-6 * box["L"]))
        if not (x["converged"] and x["in_vacuum"]):
            x["rejected"] = True
        if any(abs(x["R"] - y["R"]) < 1e-9 * box["L"] and abs(x["Z"] - y["Z"]) < 1e-9 * box["L"] for y in xpts + rejected):
            continue
        (rejected if x.get("rejected") else xpts).append(x)
    n_engine_nodes = sum(max(len(V) - 2, 1) * 64 for V, _ in view["polys"]) + len(view["R"])
    if engine_check and xpts and n_engine_nodes > 300000:
        engine_check = False                # the solver's engine would expand the polygons into > 3e5 rings
        for x in xpts:
            x["Bpol_rel_engine"] = None
    if engine_check and xpts:
        try:
            from .field import FieldEngine
            eng = FieldEngine(exact_source(view), dmin=h)
            P = np.array([[x["R"], 0.0, x["Z"]] for x in xpts])
            Be = np.asarray(eng.evaluate(P, grad=False, A=False)["B"])
            for x, b in zip(xpts, Be):
                x["Bpol_rel_engine"] = float(math.hypot(b[0], b[2]) / Bscale)
        except Exception as e:     # the engine check is optional evidence
            for x in xpts:
                x["Bpol_rel_engine"] = None
            xpts[0]["engine_error"] = repr(e)
    fams = []
    for sg in (+1, -1):
        fams += merge_families(psi, cond, Rg, Zg, view["RB"], sg, n_fil=len(view["R"]))
    # attach the refined X-point to each saddle-closed family
    for f in fams:
        if f["cause"] == "saddle" and xpts:
            d = [math.hypot(x["R"] - f["cell"][0], x["Z"] - f["cell"][1]) for x in xpts]
            k = int(np.argmin(d))
            if d[k] < 3 * h:
                f["xpoint"] = [xpts[k]["R"], xpts[k]["Z"]]
                f["psi_separatrix"] = xpts[k]["psi"]
        if f["cause"] == "saddle" and "xpoint" not in f and f["cell"] is not None:
            i = int(round((f["cell"][0] - Rg[0]) / (Rg[1] - Rg[0]))); j = int(round((f["cell"][1] - Zg[0]) / (Zg[1] - Zg[0])))
            if min(i, j, n - 1 - i, n - 1 - j) <= 2:
                f["cause"] = "box_merge"  # merged through the box-edge region: true outer end outside the box
            else:
                f["cause"] = "saddle_unresolved"   # saddle inside a filament pad (not refined)
    fams.sort(key=lambda f: -f["area"])
    res = dict(
        grid=dict(n=n, R=[float(Rg[0]), float(Rg[-1])], Z=[float(Zg[0]), float(Zg[-1])], h=float(h),
                  n_conductors=ncond, grid_rings=int(len(Fg.packed[15])), seconds=t_grid),
        RB=view["RB"], deviation_from_axisymmetry=view["deviation"],
        max_principle=dict(n_grid_extrema=len(ext), n_extrema_on_conductors=len(ext) - len(vac_ext),
                           n_extrema_in_vacuum=len(vac_ext), vacuum_extrema=vac_ext[:20],
                           n_index_plus_cells_in_vacuum=int(len(plus_vac)),
                           index_plus_cells_in_vacuum=[[float(Rg[i]), float(Zg[j])] for i, j in plus_vac[:20]],
                           n_index_minus_cells_in_vacuum=int(len(minus_vac))),
        xpoints=xpts, xpoints_rejected=rejected[:20], n_xpoints_rejected=len(rejected), B_scale=Bscale,
        families=dict(n=len(fams), n_saddle=sum(f["cause"] == "saddle" for f in fams),
                      n_psi0=sum(f["cause"] == "psi0" for f in fams), n_conductor=sum(f["cause"] == "conductor" for f in fams),
                      n_touched_box=sum(f["touched_box"] for f in fams), largest=fams[:12]),
        seconds=time.time() - t0)
    return res, dict(psi=psi, BR=BR, BZ=BZ, Rg=Rg, Zg=Zg, cond=cond, fams=fams, Fe=Fe, Fg=Fg)


def dipole_structure(view, res, aux, f_ids):
    """Levitated-dipole structure: the family of closed contours around the floating coil
    (conductor ids f_ids) up to its separatrix; psi at the coil surface; separatrix extent on
    the coil's mid plane; the poloidal flux 2 pi (psi_edge - psi_X) of the closed-line region."""
    fams = [f for f in aux["fams"] if set(f["conductors"]) == set(f_ids) and f["cause"] in ("saddle", "psi0")]
    if not fams:
        return dict(found=False)
    fam = max(fams, key=lambda f: (f["cause"] == "saddle", f["area"]))
    Fe = aux["Fe"]
    Vs = [view["polys"][k - len(view["R"])][0] for k in f_ids if k >= len(view["R"])]
    pts = []
    for V in Vs:
        Vc = np.concatenate([V, V[:1]])
        for a, b in zip(Vc[:-1], Vc[1:]):
            pts.append(a + np.outer(np.linspace(0, 1, 40, endpoint=False), b - a))
    pts = np.concatenate(pts)
    # offset the samples slightly outward from the coil's centroid to stay in vacuum
    cen = np.concatenate(Vs).mean(0)
    out = pts + 1e-6 * (pts - cen) / np.linalg.norm(pts - cen, axis=1, keepdims=True)
    inside_any = np.zeros(len(out), bool)
    for V in Vs:
        inside_any |= _points_in_polygon(V, out[:, 0], out[:, 1])
    out = out[~inside_any]
    psi_b, _, _ = psi_on(Fe, out[:, 0], out[:, 1])
    psi_edge = float(psi_b.min()) if fam["sign"] > 0 else float(psi_b.max())
    psiX = fam.get("psi_separatrix", fam["level_outer"])
    # separatrix crossing of the mid plane Z = Zc, inner and outer side
    from scipy.optimize import brentq
    Zc = float(cen[1]); Rlo = float(min(V[:, 0].min() for V in Vs)); Rhi = float(max(V[:, 0].max() for V in Vs))
    f = lambda R: psi_on(Fe, np.array([R]), np.array([Zc]))[0][0] - psiX
    Rs_out = np.geomspace(Rhi * 1.0001, 100.0 * aux["Rg"][-1], 600)    # psi -> 0 at infinity
    v_out = np.array([f(R) for R in Rs_out])
    Rsep_out = None
    k = np.where(np.sign(v_out[:-1]) != np.sign(v_out[1:]))[0]
    if len(k):
        Rsep_out = float(brentq(f, Rs_out[k[0]], Rs_out[k[0] + 1], xtol=1e-14))
    Rs_in = np.linspace(max(1e-6, aux["Rg"][0]), Rlo * 0.9999, 200)
    v_in = np.array([f(R) for R in Rs_in])
    Rsep_in = None
    k = np.where(np.sign(v_in[:-1]) != np.sign(v_in[1:]))[0]
    if len(k):
        Rsep_in = float(brentq(f, Rs_in[k[-1]], Rs_in[k[-1] + 1], xtol=1e-14))
    return dict(found=True, family=fam, psi_X=float(psiX), xpoint=fam.get("xpoint"), psi_coil_edge=psi_edge,
                psi_max_grid=fam["level_inner"], poloidal_flux_closed_region_Wb=float(TWO_PI * abs(psi_edge - psiX)),
                separatrix_midplane_R=[Rsep_in, Rsep_out], midplane_Z=Zc, toroidal_flux_Wb=0.0,
                n_edge_samples=int(len(out)))


# ============================================================================ 3D sources
def holes_of(mask_U, comp_cells_owner, curves):
    """Holes of the complement component U: bounded components of the rest of the box (the
    exterior counts as U if U touches the box edge); I_hole = sum I_k w_z(k) of the coils whose
    shadow lies in the hole."""
    from scipy import ndimage
    touches = mask_U[0, :].any() or mask_U[-1, :].any() or mask_U[:, 0].any() or mask_U[:, -1].any()
    M = np.pad(mask_U, 1, constant_values=bool(touches))
    lab, n = ndimage.label(~M, structure=np.ones((3, 3)))
    edge = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - {0}
    out = []
    for L in range(1, n + 1):
        if L in edge:
            continue
        ks = [k for k, (i, j) in enumerate(comp_cells_owner) if (lab[i + 1, j + 1] == L).any()]
        out.append(dict(coils=ks, I_hole=float(sum(curves[k]["I"] * toroidal_winding(curves[k]["pts"]) for k in ks))))
    return out


def fixed_point_newton(F, R0, Z0, phi0=0.0, scale=1.0, maxit=14, tol=1e-12, eps_rel=1e-5, h=None):
    """Period-1 closed line: Newton on d(x) = P(x) - x with a central-difference Jacobian
    J = M - I of the reference period map (step eps_rel * scale; the IRK map is accurate to
    ~1e-13 scale, so J is accurate to ~1e-8).  The index and Greene's residue come from det J
    directly (r = (2 - tr M)/4 = det(M - I)/4 for det M = 1), which stays accurate when M is close
    to the identity (near-identity maps of nearly symmetric TF sets)."""
    x = np.array([R0, Z0], float)
    ok = False
    eps = eps_rel * scale

    def jac_at(x):
        Rq = np.array([x[0], x[0] + eps, x[0] - eps, x[0], x[0]])
        Zq = np.array([x[1], x[1], x[1], x[1] + eps, x[1] - eps])
        R1, Z1, err, lost = one_turn(F, Rq, Zq, phi0, h=h, doubling=False)
        if lost.any() or not np.all(np.isfinite(R1)):
            return None, None
        d = np.array([R1[0] - x[0], Z1[0] - x[1]])
        # J from differences of displacements (avoids the cancellation of M - I)
        J = np.array([[((R1[1] - Rq[1]) - (R1[2] - Rq[2])) / (2 * eps), ((R1[3] - Rq[3]) - (R1[4] - Rq[4])) / (2 * eps)],
                      [((Z1[1] - Zq[1]) - (Z1[2] - Zq[2])) / (2 * eps), ((Z1[3] - Zq[3]) - (Z1[4] - Zq[4])) / (2 * eps)]])
        return d, J

    def disp_at(x):
        R1, Z1, err, lost = one_turn(F, np.array([x[0]]), np.array([x[1]]), phi0, h=h, doubling=False)
        if lost.any() or not np.isfinite(R1[0]):
            return None
        return np.array([R1[0] - x[0], Z1[0] - x[1]])

    d, J = jac_at(x)
    if d is None:
        return dict(R=float(x[0]), Z=float(x[1]), converged=False, lost=True)
    # Newton with Broyden updates between full central-difference Jacobians (one line per step)
    for it in range(3 * maxit):
        if np.linalg.norm(d) < tol * scale:
            ok = True
            break
        try:
            dx = -np.linalg.solve(J, d)
        except np.linalg.LinAlgError:
            break
        lim = 0.05 * scale
        if np.linalg.norm(dx) > lim:
            dx *= lim / np.linalg.norm(dx)
        xn = x + dx
        dn = disp_at(xn)
        if dn is None:
            return dict(R=float(xn[0]), Z=float(xn[1]), converged=False, lost=True)
        if np.linalg.norm(dn) > 0.9 * np.linalg.norm(d) and it % 3 == 2:
            d, J = jac_at(xn)                   # poor progress: fresh Jacobian
            if d is None:
                return dict(R=float(xn[0]), Z=float(xn[1]), converged=False, lost=True)
        else:
            J = J + np.outer(dn - d - J @ dx, dx) / (dx @ dx)
            d = dn
        x = xn
        if np.linalg.norm(dx) < 1e-15 * scale:
            ok = np.linalg.norm(d) < 1e-9 * scale
            break
    # monodromy at the converged point from a fresh central-difference Jacobian
    dj, Jj = jac_at(x)
    if dj is not None:
        d, J = dj, Jj
        ok = ok and np.linalg.norm(d) < max(tol * scale, 1e-9 * scale)
    M = J + np.eye(2)
    detJ = float(np.linalg.det(J))
    tr = float(np.trace(M))
    ev = np.linalg.eigvals(M)
    _, _, err, _ = one_turn(F, np.array([x[0]]), np.array([x[1]]), phi0, h=h, doubling=True)
    res = detJ / 4
    # reliability of the type: the error of J (~ err / eps) against the scale sqrt|det J|
    J_err = float(err[0]) / eps
    s_min = float(np.linalg.svd(J, compute_uv=False)[-1]) if np.all(np.isfinite(J)) else 0.0
    typ = "elliptic" if 0 < res < 1 else ("saddle" if res < 0 else ("reflection-saddle" if res > 1 else "parabolic"))
    return dict(R=float(x[0]), Z=float(x[1]), residual=float(np.linalg.norm(d)), converged=bool(ok),
                M=M.tolist(), det=float(np.linalg.det(M)), trace=tr, det_M_minus_I=detJ, residue=float(res),
                eig=[[float(e.real), float(e.imag)] for e in ev], type=typ,
                index=(+1 if detJ > 0 else -1), err_one_turn=float(err[0]), J_err=J_err,
                type_reliable=bool(J_err < 0.1 * math.sqrt(abs(detJ))), sigma_min_J=s_min,
                position_uncertainty=float(np.linalg.norm(d) / s_min) if s_min > 0 else None,
                rotation_per_turn=float(math.acos(max(-1.0, min(1.0, 1 - 2 * res))) / TWO_PI) if 0 < res < 1 else None)


def delta_index(dR, dZ, ok):
    """Poincare index of the displacement field P(x) - x around grid cells whose 4 corners
    all completed a turn (0 elsewhere)."""
    a = np.arctan2(dZ, dR)
    c = [a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]]
    okc = ok[:-1, :-1] & ok[1:, :-1] & ok[1:, 1:] & ok[:-1, 1:]
    tot = 0.0
    for k in range(4):
        d = c[(k + 1) % 4] - c[k]
        tot = tot + (d + np.pi) % TWO_PI - np.pi
    idx = np.round(tot / TWO_PI).astype(int)
    return np.where(okc, idx, 0), okc


def refine_cell_index(SF, box, R0, R1, Z0, Z1, phi0, m=12):
    """Poincare index of P(x) - x along the boundary of the cell [R0, R1] x [Z0, Z1] sampled with
    m points per edge (scan tracer, forward lines).  None if a boundary line does not complete."""
    t = np.arange(m) / m
    Rb = np.concatenate([R0 + (R1 - R0) * t, np.full(m, R1), R1 - (R1 - R0) * t, np.full(m, R0)])
    Zb = np.concatenate([np.full(m, Z0), Z0 + (Z1 - Z0) * t, np.full(m, Z1), Z1 - (Z1 - Z0) * t])
    ex = excursion_scan(SF, Rb, Zb, phi0=phi0, box=box, both=False)
    e = ex["fwd"]
    if not np.all(e[:, 1] == 1):
        return None
    dR = np.hypot(e[:, 2], e[:, 3]) - Rb; dZ = e[:, 4] - Zb
    a = np.arctan2(dZ, dR)
    d = np.diff(np.append(a, a[0]))
    d = (d + np.pi) % TWO_PI - np.pi
    if np.max(np.abs(d)) > 0.5 * np.pi:
        return None                       # still under-resolved
    return int(np.round(d.sum() / TWO_PI))


def analyse_3d(source, n_shadow=200, n_scan=None, phi0=0.0, verbose=False, budget_s=None, F=None,
               skip_fixed_points=False):
    t0 = time.time()
    curves = closed_curves(source)
    F = F if F is not None else fast64.Field(source)
    RB = float(sum(p.RB for p in source.parts if p.kind == "toroidal"))
    if RB != 0.0:
        # the z axis carries the TF current: a straight filament, linked by every horizontal circle
        pass
    box = coil_box(curves, None, margin=0.15)
    coils = []
    for c in curves:
        cf = circle_fit(c["pts"])
        ce = cf["center"]; ephi = np.array([-ce[1], ce[0], 0.0]); nphi = np.linalg.norm(ephi)
        coils.append(dict(I=c["I"], w_z=toroidal_winding(c["pts"]), planarity=cf["planarity"],
                          circle_dev=cf["dev"], n_dot_z=float(abs(cf["normal"][2])),
                          n_dot_phi=float(abs(cf["normal"] @ ephi) / nphi) if nphi > 0 else None,
                          R=[float(np.hypot(c["pts"][:, 0], c["pts"][:, 1]).min()), float(np.hypot(c["pts"][:, 0], c["pts"][:, 1]).max())],
                          Z=[float(c["pts"][:, 2].min()), float(c["pts"][:, 2].max())]))
    sc = shadow_components(curves, box, n=n_shadow)
    from scipy import ndimage
    owners = []
    Rg, Zg = sc["Rg"], sc["Zg"]
    hR, hZ = Rg[1] - Rg[0], Zg[1] - Zg[0]
    for c in curves:
        x = c["pts"]
        Rs, Zs = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
        owners.append((np.clip(np.round((Rs - Rg[0]) / hR).astype(int), 0, len(Rg) - 1),
                       np.clip(np.round((Zs - Zg[0]) / hZ).astype(int), 0, len(Zg) - 1)))
    Itot = sum(abs(c["I"]) for c in curves) + abs(RB) * TWO_PI / MU0
    for comp in sc["comps"]:
        if RB:
            comp["I_circ"] += RB * TWO_PI / MU0       # the z-axis filament of the ToroidalField part
            comp["I_circ_other_points"] = [v + RB * TWO_PI / MU0 for v in comp["I_circ_other_points"]]
        a, e = ampere_circle(F, comp["rep"][0], comp["rep"][1], 256)
        comp["ampere_I"] = a; comp["ampere_err"] = e
        comp["ampere_vs_count_rel"] = float(abs(a - comp["I_circ"]) / max(Itot, 1e-300))
        comp["holes"] = holes_of(sc["labels"] == comp["label"], owners, curves)
        comp["I_circ_zero"] = bool(abs(comp["I_circ"]) <= 1e-12 * Itot)
        comp["holes_zero"] = bool(all(abs(hh["I_hole"]) <= 1e-12 * Itot for hh in comp["holes"]))
        comp["excluded_by_SHADOW"] = comp["I_circ_zero"] and comp["holes_zero"]
    excluded_frac = float(sum(c["area_frac"] for c in sc["comps"] if c["excluded_by_SHADOW"]))
    shadow_frac = float(sc["shadow"].mean())
    # ---------------------------------------------------------------- excursion scan
    if n_scan is None:
        n_scan = 33
    Rs = np.linspace(box["R"][0], box["R"][1], n_scan + 2)[1:-1]
    Zs = np.linspace(box["Z"][0], box["Z"][1], n_scan + 2)[1:-1]
    RRs, ZZs = np.meshgrid(Rs, Zs, indexing="ij")
    ts = time.time()
    SF = ScanField(source)
    ex = excursion_scan(SF, RRs.ravel(), ZZs.ravel(), phi0=phi0, box=box)
    t_scan = time.time() - ts
    dfw, cfw = ex["fwd"][:, 0], ex["fwd"][:, 1]
    dbw, cbw = ex["bwd"][:, 0], ex["bwd"][:, 1]
    comp_fw = (cfw == 1).reshape(RRs.shape); comp_bw = (cbw == 1).reshape(RRs.shape)
    complete = comp_fw | comp_bw
    exc = dict(n=int(RRs.size), h=[float(Rs[1] - Rs[0]), float(Zs[1] - Zs[0])], phi0=phi0,
               n_complete_fwd=int(comp_fw.sum()), n_complete_bwd=int(comp_bw.sum()),
               n_complete_both=int((comp_fw & comp_bw).sum()),
               max_abs_dphi_incomplete=float(max([abs(v) for v, c in zip(np.concatenate([dfw, dbw]), np.concatenate([cfw, cbw])) if c != 1], default=0.0)),
               codes_fwd={int(k): int((cfw == k).sum()) for k in np.unique(cfw)},
               codes_bwd={int(k): int((cbw == k).sum()) for k in np.unique(cbw)},
               seconds=t_scan)
    # ---------------------------------------------------------------- one-turn map on completing lines
    fp = dict(n_points=0)
    if complete.any():
        # the one-turn map P from the scan itself: forward-completing lines end exactly on phi0 + 2 pi
        okm = comp_fw.copy()
        e = ex["fwd"]
        R1 = np.hypot(e[:, 2], e[:, 3]).reshape(RRs.shape); Z1 = e[:, 4].reshape(RRs.shape)
        dR = np.where(okm, R1 - RRs, 0.0); dZ = np.where(okm, Z1 - ZZs, 0.0)
        # where only the backward line completes use -(P^-1(x) - x) (same fixed points and indices)
        eb = ex["bwd"]
        Rb = np.hypot(eb[:, 2], eb[:, 3]).reshape(RRs.shape); Zb = eb[:, 4].reshape(RRs.shape)
        onlyb = comp_bw & ~comp_fw
        dR = np.where(onlyb, -(Rb - RRs), dR); dZ = np.where(onlyb, -(Zb - ZZs), dZ)
        okm = okm | onlyb
        dn = np.where(okm, np.hypot(dR, dZ), np.nan)
        idx, okc = delta_index(dR, dZ, okm)
        hR, hZ = Rs[1] - Rs[0], Zs[1] - Zs[0]
        # confirm every cell index with a finely sampled boundary (the coarse index assumes that
        # P(x) - x turns by less than pi between neighbouring nodes)
        conf = {}
        for i, j in np.argwhere(idx != 0)[:24]:
            conf[(int(i), int(j))] = refine_cell_index(SF, box, RRs[i, j], RRs[i, j] + hR, ZZs[i, j], ZZs[i, j] + hZ, phi0)
        idx_conf = np.zeros_like(idx)
        for (i, j), v in conf.items():
            if v is not None:
                idx_conf[i, j] = v
        cand = [(i, j, "cell") for (i, j), v in conf.items() if v is not None and v > 0][:8] + \
               [(i, j, "cell") for (i, j), v in conf.items() if v is not None and v < 0][:6] + \
               [(i, j, "cell") for (i, j), v in conf.items() if v is None][:4]    # index +1 cells first
        mins = []
        for (i, j) in np.argwhere(okm):
            nb = [dn[a, b] for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                  if 0 <= a < dn.shape[0] and 0 <= b < dn.shape[1] and okm[a, b]]
            if len(nb) == 4 and dn[i, j] < min(nb):
                mins.append((dn[i, j], int(i), int(j)))
        mins.sort()
        cand += [(i, j, "min") for _, i, j in mins[:2]]
        L = box["L"]
        fps = []
        for (i, j, how) in cand[:20]:
            if skip_fixed_points:
                break
            R0 = RRs[i, j] + (0.5 * hR if how == "cell" else 0.0)
            Z0 = ZZs[i, j] + (0.5 * hZ if how == "cell" else 0.0)
            if any(math.hypot(R0 - q["seed"][0], Z0 - q["seed"][1]) < 0.5 * hR for q in fps):
                continue
            r = fixed_point_newton(F, R0, Z0, phi0, scale=L)
            if r.get("converged") and not r["type_reliable"]:
                r2 = fixed_point_newton(F, r["R"], r["Z"], phi0, scale=L, h=TWO_PI / 512)
                if r2.get("converged"):
                    r2["refined_step"] = TWO_PI / 512
                    r = r2
            r["seed"] = [float(R0), float(Z0)]; r["seed_kind"] = how
            if not r.get("converged"):
                continue
            tolpos = max(1e-8 * L, 10 * (r.get("position_uncertainty") or 0.0))
            if any(math.hypot(r["R"] - q["R"], r["Z"] - q["Z"]) < max(tolpos, 10 * (q.get("position_uncertainty") or 0.0))
                   for q in fps):
                continue
            fps.append(r)
            if budget_s and time.time() - t0 > budget_s:
                break
        fp = dict(n_points=int(okm.sum()), max_delta=float(np.nanmax(dn)), min_delta=float(np.nanmin(dn)),
                  median_delta=float(np.nanmedian(dn)),
                  index_sum_cells=int(idx_conf.sum()), n_index_cells=int((idx_conf != 0).sum()),
                  n_index_cells_coarse=int((idx != 0).sum()),
                  n_index_cells_unresolved=int(sum(v is None for v in conf.values())),
                  index_cells=[[float(RRs[i, j] + 0.5 * hR), float(ZZs[i, j] + 0.5 * hZ), int(idx[i, j]), conf.get((int(i), int(j)), "unchecked")]
                               for i, j in np.argwhere(idx != 0)][:40],
                  n_candidates=len(cand), fixed_points=fps,
                  completing_box=[float(RRs[complete].min()), float(RRs[complete].max()),
                                  float(ZZs[complete].min()), float(ZZs[complete].max())])
        # toroidal flux through the completing region of the section (grid sum, resolution h)
        Xc = np.stack([RRs * math.cos(phi0), RRs * math.sin(phi0), ZZs], -1).reshape(-1, 3)
        Bc = F.evaluate(Xc, grad=False)["B"]
        Bphi = (-Bc[:, 0] * math.sin(phi0) + Bc[:, 1] * math.cos(phi0)).reshape(RRs.shape)
        fp["toroidal_flux_completing_Wb"] = float(np.sum(np.where(complete, Bphi, 0.0)) * hR * hZ)
    return dict(box=box, coils=coils if len(coils) <= 60 else coils[:60], n_coils=len(curves), RB=RB,
                shadow=dict(n=n_shadow, shadow_area_frac=shadow_frac, excluded_area_frac=excluded_frac,
                            comps=sc["comps"]),
                excursion=exc, one_turn=fp, seconds=time.time() - t0), dict(F=F, RRs=RRs, ZZs=ZZs, complete=complete,
                                                                         ex=ex, sc=sc)


# ============================================================================ mirror planes
def _seg_arrays_of(curves):
    a = np.concatenate([c["pts"] for c in curves])
    b = np.concatenate([np.roll(c["pts"], -1, axis=0) for c in curves])
    I = np.concatenate([np.full(len(c["pts"]), c["I"]) for c in curves])
    return a, b, I


def reflection(phi0):
    n = np.array([-math.sin(phi0), math.cos(phi0), 0.0])
    return np.eye(3) - 2 * np.outer(n, n)


def mirror_planes(source, curves, F, box, tol=1e-10, max_cand=96):
    """Meridional planes phi = phi0 under whose reflection S the current set is symmetric
    (J(Sx) = S J(x): B normal to the plane) or antisymmetric (J(Sx) = -S J(x): B_phi = 0 on the
    plane).  Element-set matching of the segments (orientation-free); for every accepted plane
    the field statement is checked with the reference field on a grid of the half plane."""
    from torux.io._stellcommon import seg_set_maps as _seg_set_maps
    if any(p.kind in ("rings", "axipoly", "toroidal") for p in source.parts):
        # rings / polygons / the TF are symmetric under every meridional reflection
        pass
    a, b, I = _seg_arrays_of(curves)
    for p in source.parts:
        if p.kind == "toroidal":
            pass
    cen = np.array([np.arctan2(c["pts"][:, 1].mean(), c["pts"][:, 0].mean()) for c in curves])
    cand = list(np.mod(cen, TWO_PI))
    srt = np.sort(np.unique(np.round(np.mod(cen, TWO_PI), 12)))
    if len(srt) > 1:
        mids = 0.5 * (srt + np.roll(srt, -1)); mids[-1] = 0.5 * (srt[-1] + srt[0] + TWO_PI)
        cand += list(np.mod(mids, TWO_PI))
    cand += [0.0, math.pi / 2]
    cand = np.unique(np.round(np.mod(np.array(cand), math.pi), 12))[:max_cand]   # plane phi0 == phi0 + pi
    has_tor = any(p.kind == "toroidal" for p in source.parts)
    has_axi = any(p.kind in ("rings", "axipoly") for p in source.parts)
    out = []
    Rg = np.linspace(box["R"][0], box["R"][1], 24)[1:-1]
    Zg = np.linspace(box["Z"][0], box["Z"][1], 24)[1:-1]
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    for ph in cand:
        S = reflection(ph)
        ds = _seg_set_maps(a, b, I, S, +1, tol) if len(a) else 0.0
        da = _seg_set_maps(a, b, I, S, -1, tol) if len(a) else 0.0
        # the toroidal field and the axisymmetric parts are symmetric (sign +1) under S
        if (has_tor or has_axi) and da <= tol:
            da = np.inf
        kind = "sym" if ds <= tol else ("anti" if da <= tol else None)
        if kind is None:
            continue
        rec = dict(phi0=float(ph), kind=kind, dev=float(ds if kind == "sym" else da))
        pts = []
        for ph2 in (ph, ph + math.pi):
            pts.append(np.stack([RR.ravel() * math.cos(ph2), RR.ravel() * math.sin(ph2), ZZ.ravel()], -1))
        pts = np.concatenate(pts)
        B = F.evaluate(pts, grad=False)["B"]
        phs = np.arctan2(pts[:, 1], pts[:, 0])
        Bphi = -B[:, 0] * np.sin(phs) + B[:, 1] * np.cos(phs)
        Bn = np.linalg.norm(B, axis=1)
        if kind == "anti":
            rec["max_abs_Bphi_over_B"] = float(np.max(np.abs(Bphi) / Bn))
        else:
            rec["max_Bpol_over_B"] = float(np.max(np.sqrt(np.maximum(Bn ** 2 - Bphi ** 2, 0)) / Bn))
        out.append(rec)
    return out


def identity_from_mirrors(planes):
    """Two symmetric planes phi_a < phi_b: the field-line map over 2 (phi_b - phi_a) is the
    identity.  Returns the smallest such period d with 2 pi / d an integer (P = identity), or None."""
    sy = sorted({round(p["phi0"] % math.pi, 10) for p in planes if p["kind"] == "sym"})
    best = None
    for i in range(len(sy)):
        for j in range(i + 1, len(sy)):
            d = 2 * (sy[j] - sy[i])
            for dd in (d, TWO_PI - d):
                m = TWO_PI / dd if dd > 0 else 0
                if dd > 0 and abs(m - round(m)) < 1e-9:
                    best = dd if best is None else min(best, dd)
    return best


# ============================================================================ theorems (proof sketches)
THEOREMS = {
    "DEF": (
        "Toroidal flux surface: a C^1 embedded torus T in the vacuum region (outside every current element, "
        "and off the z axis when a ToroidalField part is present), tangent to B with B != 0 on T, meeting every half "
        "plane phi = const in one Jordan curve C_phi bounding the disc D_phi, with B.grad(phi) > 0 (or < 0) on the "
        "solid torus V = union of the D_phi (no toroidal-field reversal inside the surface).  Enclosed toroidal "
        "flux: the flux of B through D_phi (phi-independent when V is current free).  The period map "
        "P: D_0 -> D_0, phi -> phi + 2 pi along field lines, is then an orientation- and flux-preserving "
        "homeomorphism (this is the setting of the solver's tori, torux.torus)."),
    "LEMMA_A": (
        "(torux/barrier.py) R B_phi = x B_y - y B_x is harmonic where curl B = 0 and div B = 0 (its Laplacian is "
        "2 (curl B)_z).  For a toroidal flux surface T with B.grad(phi) > 0 on T and no source in its solid torus V, "
        "the minimum principle gives B_phi > 0 on all of V-bar; the assumption 'B.grad(phi) > 0 on V' of DEF is "
        "therefore automatic once it holds on T.  Consequences: lines through V-bar never reach B_phi <= 0, and "
        "every section D_phi lies in {B_phi > 0}."),
    "FLUXBOUND": (
        "Certified flux bound (torux.nonstell_cert.flux_bound).  By LEMMA_A the section D_phi of a surface with "
        "B_phi > 0 lies in {B_phi > 0} of the half plane H_phi, so its toroidal flux is <= int_{H_phi} B_phi^+ dA, "
        "for every phi (and <= int B_phi^- for surfaces with B_phi < 0).  The integral is bounded from above on a "
        "polar grid about (R, Z) = (0, z_c), log-spaced from 1e-3 L to 1e4 L: each cell contributes "
        "area x max(0, B_phi(c) + rho Lip) with Lip = |DB| + rho |D^2 B| + (|B| + rho |DB|_max) / (R - rho) from the "
        "reference field and gradient, |D^2 B| from finite differences of the reference gradient with their error "
        "bound plus rho x an analytic bound on |D^3 B| (|D^k (r/|r|^3)| <= (k+1)! / |r|^(k+2), Legendre), or the "
        "analytic piece bound far from the coils; cells near a coil get a 1/d integral bound (crossing pieces "
        "(2 / sin theta) 2 pi (2 rho) (mu0 |I| / 4 pi), non-crossing ones 16 rho asinh(2 rho / h)); the disc "
        "rho < 1e-3 L gets area x sup |B|; beyond 1e4 L the exact dipole term (its B_phi on a half plane is "
        "-(mu0/4pi) m.e_phi / |u|^3) plus the Taylor remainder C / (|u| - a)^4, C = sum (mu0 |I| / 4 pi) 3 P a^2.  "
        "The minimum over the half planes tried is kept, per sign.  Result: every toroidal flux surface has "
        "|flux| <= Phi_max (a proof for every request above Phi_max)."),
    "CAGE": (
        "(torux.nonstell_cert, implemented, not applied) If every point of the boundary of a solid C containing the "
        "coils (cylinder, or annulus when every coil has w_z = 0) and of one section of C is LOST (its line reaches "
        "both signs of B_phi), no toroidal flux surface exists: V-bar must meet C (else I_link = 0, AMP), cannot "
        "cross dC (LEMMA_A), so lies inside C, and its section would be in the lost section.  Lost cells are "
        "certified by the variational equation along the centre's line plus a second-derivative (tube) bound.  "
        "Measured cost: near the coils and where B_phi / |B| ~ 1e-5..1e-2 (near-antisymmetric planes, far field) "
        "the certificates need cells of 1e-4..1e-3 L, i.e. 1e5..1e6 cells per source: not run by default."),
    "CAGE2": (
        "Orientation sigma.  C = {R_lo <= R <= R_hi, |Z| <= Z_h} holds every coil "
        "in its interior, every coil of toroidal winding 0.  A point is EXCLUDED if sigma u <= 0 there (u = R B_phi) or "
        "its line reaches an excluded point (LEMMA_A: no V-bar of orientation sigma contains one); the rotation by "
        "2 pi / nfp and the stellarator symmetry map excluded points to excluded points.  If every point of dC is "
        "excluded, V-bar (it meets C by AMP, is connected and misses dC) lies inside C, its section D in a symmetry "
        "half plane lies in the part K of the cage section not certified excluded, and |flux| <= F_up = int_K "
        "(sigma B_phi)^+ dA; K empty: no surface of orientation sigma.  Pieces (quadtree squares of dC over one half "
        "period and of the half planes phi = 0, pi / nfp) are certified by SIGN0 (sigma u < 0 on the piece), BASE (at "
        "one arc length the piece's image has sigma u < 0), RELAY (the piece's lines cross a symmetry half plane in "
        "pieces certified in earlier rounds), EXIT (section pieces, once dC is complete: the image leaves C).  Strict: "
        "the theorem, the Taylor sign and flux bounds (field derivatives to third order at the centre, analytic |D^4 B| "
        "piece bounds), the raster covering, the well-founded rounds.  Validated: the 9 + 4 stencil enclosures of the "
        "images (torux.edgebound's affine and curved models), the GBS trace error, the calibrated trapezoid field "
        "(against the reference evaluator in every run), the smoothness of the maps on a piece (coil guard, |f| >= 0.05 "
        "up to and at a relay crossing).  Every run re-traces sampled certified pieces from random interior points "
        "with DOP853 on the reference field.  CAGE2-H: a non-empty residual whose squares carry an "
        "expanding cone field of the half-period maps (DH maps each cone into the next, expanding by >= 1.02; DH^-1 "
        "expands off the cones) holds no C^1 invariant curve -- the sections of T would have length 0 -- so no "
        "surface of that orientation either.  ZERO: a line through u = 0 is excluded for both orientations (sigma u = "
        "0 <= 0), certified on a piece by the hitting map of the zero surface (as a relay's).  EXIT also by the "
        "hitting map of dC."),
    "NOTOR": (
        "B_phi = 0 identically (axisymmetric source without a ToroidalField part; or a meridional half plane on "
        "which B_phi = 0, see ANTISYM): a toroidal flux surface meets that half plane with B.grad(phi) != 0, "
        "impossible.  For axisymmetric sources without toroidal field every field line lies in a meridional half "
        "plane on a psi contour; the closed contours (around conductors, see AXI) are closed poloidal field lines "
        "and sweep tori around the conductors that enclose zero toroidal flux."),
    "AXI": (
        "Axisymmetric vacuum (Rings, AxisymPolygons, ToroidalField RB = F).  B = grad psi x grad phi + F grad phi, "
        "psi = R A_phi; (curl B)_phi = -(1/R) Delta* psi, Delta* = d_RR - (1/R) d_R + d_ZZ, so Delta* psi = "
        "-mu0 R J_phi and Delta* psi = 0 in vacuum.  Delta* is uniformly elliptic on R >= R0 > 0 with no zeroth "
        "order term: strong maximum principle (Hopf; Gilbarg-Trudinger Thm 3.5): psi has no local extremum in the "
        "vacuum region unless constant (then B_pol = 0 everywhere, excluded by any coil current).  At a critical "
        "point B_pol = 0 the Hessian of psi has trace Delta* psi + psi_R/R = 0: every critical point is a saddle "
        "(X-point, index -1) or a degenerate (higher-order) saddle.  (i) A closed psi contour Gamma bounds a disc "
        "containing current: if the disc were current free, psi = const on Gamma forces psi = const inside "
        "(maximum principle), contradiction.  (ii) No toroidal flux surface with current-free interior V exists "
        "(F != 0): field lines on T project to the orbits of the planar Hamiltonian flow u = (R^2/F) B_pol "
        "(Hamiltonian psi).  Case (a): the core of V does not wind poloidally around any conductor with net "
        "current.  Ampere: along a field line B.dl = (F + R^2 |B_pol|^2 / F) dphi, so a segment on T of N toroidal "
        "turns, closed by an arc of C_0, gives 2 pi F N + int R^2 |B_pol|^2 / F dphi + O(1) = mu0 N I_link = "
        "2 pi F N; hence int_0^inf R^2 |B_pol|^2 dphi < inf along every line on T, and as the integrand is "
        "uniformly continuous in phi (Barbalat) B_pol -> 0: every line on T accumulates only on critical points of "
        "psi.  These are isolated (Hartman-Wintner), finitely many in the compact projection of T, so psi "
        "(constant along lines, continuous) takes finitely many values on the connected C_0: psi is constant on "
        "the Jordan curve C_0, and (i) gives current in D_0: contradiction.  Case (b): the core winds n != 0 times "
        "around a conductor group.  The winding angle of every line through D_0 around it grows like n per "
        "toroidal turn (D_0 is a disc avoiding it), while the bounded orbits of u on regular levels (all but "
        "finitely many) are closed contours around the group: all contours through D_0 have period 2 pi / |n|, "
        "so P = identity on the open set D_0 and the analytic period function is constant on the whole family of "
        "contours.  It is not: at the outer end of every family the contour passes through a saddle of psi or "
        "reaches psi -> 0 (the axis / infinity), where the period diverges (iota -> 0); next to a filament "
        "iota -> infinity.  Families ending on another polygon conductor (finite period at both ends) are checked "
        "numerically: P = identity on an open set would show as |P(x) - x| = 0 at grid nodes; the minimum of "
        "|P(x) - x| over vacuum grid nodes away from X-points is recorded (iso_check).  Hence the only "
        "toroidal flux surfaces of an axisymmetric vacuum field are tori that enclose a conductor (the psi "
        "contours around coils, nested from the conductor out to the separatrix through an X-point)."),
    "AMP": (
        "Ampere bound (any source).  Let T be a toroidal flux surface with current-free V.  A field-line arc on T "
        "from x in C_0 through N toroidal turns back to C_0, closed by an arc of C_0, is a closed curve in V "
        "homologous in V to N times the core; Ampere (Biot-Savart) gives its circulation mu0 N I_link with "
        "I_link = sum_k I_k Lk(core, coil_k) (+ the z-axis current 2 pi RB / mu0 of a ToroidalField part).  The "
        "circulation along the field-line part is int |B| dl >= B_min(T) * 2 pi R_min(T) * N, the closing arc "
        "contributes at most B_max |C_0|.  N -> infinity: mu0 I_link >= 2 pi R_min B_min > 0 (sign of B_phi).  So "
        "a toroidal flux surface needs a positive linked current; coils that do not encircle the tube of V "
        "(Lk = 0) do not contribute."),
    "SHADOW": (
        "Consequence of AMP.  Shadow of a coil: its projection (R, Z) to the meridional half plane.  Let U be a "
        "connected component of the half plane minus the (dilated) shadows; U x S^1 avoids every coil.  "
        "H_1(U x S^1) is generated by a horizontal circle {p} x S^1 and loops m_h around the holes h of U at fixed "
        "phi.  Horizontal circles at p, q in U are homologous (annulus gamma x S^1, gamma a path p -> q), so "
        "I_circ(U) = sum_k I_k Lk(C(p), coil_k) is the same for every p in U (checked at several points).  "
        "Lk(m_h, coil_k) = net crossings of coil k through the half plane phi inside m_h = w_z(k) (its winding "
        "about z) if the shadow of k lies in h, else 0.  A rotational torus has core class 1*circle + sum n_h m_h, "
        "so I_link = I_circ(U) + sum_h n_h I_hole(h), I_hole(h) = sum_{k in h} I_k w_z(k).  If I_circ(U) = 0 and "
        "every I_hole(h) = 0, AMP excludes every toroidal flux surface whose solid torus V contains a closed loop of "
        "toroidal winding 1 inside U x S^1 (that loop is homologous in V to the core), in particular every surface "
        "with V inside U x S^1, for any field-line geometry and any coil currents with the same linking pattern.  "
        "Surfaces all of whose toroidal loops cross the (dilated) shadow strip are not covered (EXC/INDEX).  The linking numbers are computed exactly (signed crossings of the coil polygon through the "
        "disc spanned by the circle) and checked against the circulation of the reference field (periodic "
        "trapezoid rule, spectrally accurate) divided by mu0 and against the Gauss linking integral."),
    "ANTISYM": (
        "If the current set is antisymmetric under the reflection S through a meridional plane (J(Sx) = -S J(x), "
        "checked by matching the reflected segment set with the original one with reversed currents, to "
        "rounding), Biot-Savart gives B(Sx) = S B(x); on the plane Sx = x, so the normal component B_phi vanishes "
        "on both half planes phi0, phi0 + pi (checked with the reference field).  A toroidal flux surface crosses "
        "every half plane with B.grad(phi) != 0: none exists, anywhere, for this current pattern.  Where the reflected "
        "set matches only to a small relative deviation dev (coordinates rounded in the source files; dev <= 2.6e-12 "
        "here) the statement is exact for the symmetrised set, and for the data B_phi / |B| on the plane is bounded by "
        "the recorded value (<= 1e-9): a surface would have to cross the plane at that pitch."),
    "PLANAR": (
        "All coils lie in one horizontal plane Pi (z = Z0, to rounding).  The reflection S through Pi fixes every "
        "coil pointwise with its current, so B(Sx) = -S B(x) and B is normal (vertical) on Pi off the wires: "
        "B_phi = 0 on Pi.  A toroidal flux surface therefore avoids Pi and lies (with V) on one side of it; each "
        "coil bounds a flat Seifert surface inside Pi, disjoint from V, so Lk(core, coil) = 0 and I_link = 0: "
        "AMP excludes it.  Holds for every current."),
    "SYM": (
        "If the current set is symmetric under reflections S_a, S_b through two meridional planes (J(Sx) = S J(x)), "
        "B(Sx) = -S B(x): B is normal to both planes.  A field line from (R, Z) on plane a reaches plane b and, by "
        "the reflection S_b, continues as its own mirror image to plane 2b - a at the same (R, Z): the field-line "
        "map over 2(b - a) is the identity.  If 2 pi / (2 (b - a)) is an integer, P = identity: every field line "
        "that makes a toroidal turn is closed, every closed curve inside the region of such lines sweeps an "
        "invariant rotational torus (iota = 0, all lines closed: degenerate surfaces); their toroidal fluxes fill "
        "(0, flux of the region)."),
    "INDEX": (
        "Fixed points of P (period-1 closed field lines).  Lefschetz-Hopf on the closed disc D_0 (P(D_0) = D_0, "
        "isolated fixed points): the indices add up to 1.  An interior ordinary saddle (eigenvalues 0 < 1/l < 1 "
        "< l, residue (2 - tr M)/4 < 0) has index -1; a saddle on the boundary C_0 (which C_0 must then follow "
        "along its stable/unstable branches) has index 0 relative to D_0 (the displacement of the retracted map "
        "never points into the half plane opposite to its expanding direction); elliptic and reflection-"
        "hyperbolic points have index +1.  So every toroidal flux surface encloses a fixed point of index +1 "
        "(and, by Brouwer's translation theorem with flux preservation, a fixed point in the open disc).  If the "
        "only period-1 lines of a region are ordinary saddles, no toroidal flux surface has its section in that "
        "region; if P has no fixed point in a region, none either.  Numerical part: fixed points are located by "
        "the Poincare index of P(x) - x on the cells of a grid of completed lines and Newton-refined with the "
        "reference period map; completeness is a numerical statement (grid resolution)."),
    "SIGNTOP": (
        "Sign topology of u = R B_phi.  Orientation sigma: V-bar lies in {sigma u > 0} (LEMMA_A), is invariant, and its core gamma has toroidal winding 1 and sigma I_link(gamma) > 0 (AMP).  X = {R <= R_b, Z_a <= Z <= Z_b} contains the coils; E = its exterior (minus the z axis), whose loops link no coil.  X is cut into closed voxels (R, phi, Z boxes); a voxel is N if sigma u <= 0 is proved on it (exact third-order Taylor polynomial at its centre + a fourth-order remainder from per-segment derivative bounds + rounding margin).  A tube T (voxels within delta of the coils) whose components carry no loop of winding != 0 and whose boundary faces are covered by dyadic pieces each proved sigma u <= 0 or LOST (every field line through the piece reaches sigma u < 0: validated tracing with variational tube bounds) is missed by V (V is connected, not inside T, and misses dT).  So V lies in K u E, K = the voxels neither N nor in T.  Nerve theorem: K u E minus the axis is homotopy equivalent to the nerve of its voxel cover (closed, convex in (R, phi, Z), phi-width < pi); its 1-skeleton loops, realised through voxel centres and shared face/edge/vertex centres, carry exact (tw, I_link): tw from the phi increments, Lk with each coil as signed crossings with its centroid fan (a 2-chain with boundary the coil; floating-point orientation predicates with a static error filter, exact rational arithmetic where undecided).  If no component's cycle group contains (tw = 1, sigma I_link > 0) -- for the component of E: no cycle with I_link != 0 -- no surface of orientation sigma exists; both orientations: none at all (for the given currents).  Every claim (sign boxes, tube, faces, pieces, LOST traces, topology) is re-derived by the checker with the reference geometry and its own code (line-potential derivatives, own RK4 tracer)."),
    "SKEW-RING": (
        "An exactly circular coil has a field axisymmetric about its axis A' with no azimuthal part about A': every line off A' is a closed curve in a half plane bounded by A'.  If A' and the z axis are skew, the z axis pierces each plane through A' off A', and the closed lines enclosing the piercing point wind once about the z axis; where they are star-shaped about it (B_phi of one sign) a smooth disc of them sweeps a solid torus whose boundary is a toroidal flux surface (all lines closed, iota = 0).  If A' meets or is parallel to the z axis, no closed line winds and no solid torus with a winding-1 core is a union of lines: none exists.  Demonstrated on a model ring (tilt 0.05, offset 0.03 m: lines between A' and the z axis close with winding 1 and B_phi / |B| in [0.025, 0.075]); the single PF coils of the data are tilted by 1e-3 about axes meeting the z axis to < 1e-7 m, their vertices rounded to ~5e-5 m: whether winding-1 closed lines (passing within ~1e-5 m of A', reaching ~1e8 m) and surfaces around them exist is decided by the rounding, not resolved here."),
    "LINKBOX": (
        "(torux.nonstell_index, certify_source) Exact exclusion by closed-line linking.  Let T be a toroidal flux surface, "
        "oriented so that B_phi > 0 on T; by LEMMA_A u = R B_phi > 0 on the closed solid torus V.  (1) The period map P of "
        "the section D (a closed disc) has a fixed point (Brouwer); it lies on a closed field line g in V of toroidal winding "
        "1 with u > 0 along g.  (2) Ampere for g alone: oint_g B.dl = int_g |B| dl > 0 equals mu0 sum_k I_k Lk(g, C_k) "
        "(+ the z-axis current of a ToroidalField part, which this theorem excludes), so g links a coil C_j with I_j != 0.  "
        "(3) For every coil of toroidal winding 0 a compact contractible box W_j = {R1 <= R <= R2, Z1 <= Z <= Z2, phi in an "
        "arc of length < 2 pi}, R1 > 0, contains C_j (segment extents exact).  If g missed W_j, C_j would be null-homotopic "
        "in W_j, inside R^3 - g, and Lk(g, C_j) = 0; and g is not contained in W_j, because every closed curve in the "
        "contractible W_j (off the z axis) has toroidal winding 0.  So g meets the boundary of some W_j.  (4) Every point "
        "of every box boundary is certified LOST for both orientations: its field line reaches u < 0 and u > 0 (cells: "
        "planar or cylindrical patches of the six faces; per cell the centre line is followed along +B and -B with its "
        "variational matrix (Dormand-Prince 5(4), unnormalised field-line time, reference field torux.reference.fast64); the "
        "deviation of every line of the cell from the linear prediction is bounded by variation of constants with an "
        "analytic or central-difference bound of |D^2 B| on the tube (bootstrap of the tube radius), and u on the image of "
        "the cell by a second-order Taylor bound with an analytic third-derivative remainder; integration errors: 10 x the "
        "embedded estimates, propagated with the one-step propagators -- a validated model, as in torux/barrier.py), or "
        "by TRANSFER: all lines of the cell cross a box face inside cells already certified (lost-ness belongs to the "
        "whole line; crossing by a clean-side test of the tube balls against the face), or by covering the cell's image "
        "at some time by smaller pieces (linear image of the cell plus the tube bound) that are certified in turn; on phi "
        "faces also by the "
        "period map of the face's half plane (validated phi-integration with a tube bound): N, |P(z) - z| > 0 on the "
        "cell's disc, or S, a Krawczyk disc with exactly one fixed point whose monodromy trace exceeds 2 (det = 1 by flux "
        "conservation): a regular saddle, index -1 inside D and 0 or -1 on its boundary (local model of a hyperbolic "
        "point with an invariant C^1 curve through it), so with only such fixed points the Lefschetz-Hopf sum could "
        "not be +1.  A point of g is not lost (u > 0 along all of g): contradiction, so no toroidal flux surface with B_phi > 0 exists, "
        "and the same for B_phi < 0.  Rotation symmetry (verified exactly on the segment set) lets one box per orbit be "
        "certified.  Holds for the given currents (unit currents where the data give none).  Coils with toroidal winding "
        "!= 0 (PF rings), ToroidalField parts and coils through the z axis are not covered (their linking surfaces reach "
        "the z axis); where they occur the residual is stated."),
    "EXC": (
        "Numerical exclusion by toroidal excursion.  Every point of D_0 lies on a field line that keeps B_phi of one "
        "sign for infinitely many toroidal turns.  Field lines are followed in arc length (Dormand-Prince 5(4), "
        "tolerance 3e-10 L, on a plain float64 Biot-Savart of the coils: polylines exact, Fourier curves and rings "
        "as 256-2048-gons, relative field error <~ 1e-5; the TF analytic) from every node of a grid of spacing h in "
        "the section phi0, in both directions, "
        "until B_phi changes sign (lines leaving the enlarged box [R_lo/2, 2 R_hi] x [Z_lo - L, Z_hi + L] count as not "
        "completing).  If no line completes one toroidal turn, D_0 contains no grid node: every "
        "toroidal flux surface whose section at phi0 meets the scan box has inradius < h / sqrt 2 there (numerical: it "
        "relies on the tracer at the nodes, not on bounds between them)."),
    "PERTURB": (
        "Near-axisymmetric sources (loops that are horizontal circles to relative deviation delta < 1e-5, e.g. FOCUS "
        "Fourier circles with 1e-8 centre offsets): AXI applies to the axisymmetric projection, whose period map "
        "P0 has no fixed points away from the saddles of psi (|P0(x) - x| >= m > 0 on compact sets avoiding them) "
        "and only ordinary saddles at them.  The source's map P differs from P0 by O(delta) in C^1, so for delta "
        "small P has no fixed point where m > O(delta) and one ordinary saddle near each saddle of P0 (structural "
        "stability): INDEX then excludes toroidal flux surfaces whose sections avoid the conductor families.  "
        "Inside conductor families (closed psi contours around coils, iota varying) the perturbation can open "
        "resonant islands of width O(sqrt delta); these, and the conductor-enclosing tori, are the only possible "
        "surfaces.  The 3D scan of the actual source (excursion, period-1 lines) is recorded as corroboration."),
    "DIPOLE": (
        "Levitated dipole (axisymmetric, no toroidal field): B_phi = 0, every field line is a psi contour in a "
        "meridional plane (NOTOR).  The family of closed contours around the floating coil runs from the coil "
        "surface (psi_edge, the extreme of psi on the coil boundary) to the separatrix psi = psi_X through the "
        "X-point that bounds it in the merge tree of psi (AXI(i): the contours enclose the coil).  Inside: closed "
        "field lines on nested tori psi = const, enclosing zero toroidal flux; the flux of B between the coil "
        "surface and the separatrix (through the annulus of the mid plane) is 2 pi |psi_edge - psi_X|."),
}


# ============================================================================ classify
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def _fcoil_ids(source, view):
    groups = source.meta.get("groups") or []
    ids = []
    off = len(view["R"])
    for g in groups:
        nm = str(g.get("name", ""))
        if nm.upper().startswith("F"):
            ids += [off + k for k in range(int(g["start"]), int(g["stop"]))] if source.parts[g["part"]].kind == "axipoly" \
                else list(range(int(g["start"]), int(g["stop"])))
    return ids


def attach_flux_bound(rec, results_dir=None):
    """Copy the FLUXBOUND certificate kept in results_dir/_fluxbound/<id>.json, if any, into rec['flux_bound']."""
    import json as _json
    from pathlib import Path as _P
    if not results_dir:
        return rec
    d = _P(results_dir)
    p = d / "_fluxbound" / (rec["id"] + ".json")
    if p.exists():
        fb = _json.load(open(p))
        if fb.get("applicable") and fb.get("Phi_max") is not None:
            rec["flux_bound"] = {k: fb[k] for k in ("Phi_max", "Phi_plus", "phi_plus", "Phi_minus", "phi_minus",
                                                     "ref_int_absBphi", "ratio", "dipole_moment", "grid", "seconds")
                                 if k in fb}
            rec["flux_bound"]["file"] = str(p.relative_to(d.parents[1]))
    return rec


def attach_linkbox(rec, results_dir=None, source=None):
    """Attach the LINKBOX certificate kept in results_dir/_index/<id>.json (torux.nonstell_index) to a 3D record:
    rec['linkbox'] (summary, file); if it proves the claim, verdict NO_TOROIDAL_FLUX_SURFACE with argument
    LINKBOX; otherwise rec['open'] states exactly what remains open.  Without a certificate file the scope
    (what LINKBOX could cover) is recorded from the source geometry (source given)."""
    import json as _json
    from pathlib import Path as _P
    d = _P(results_dir) if results_dir else None
    cands = [d / "_index" / (rec["id"] + t + ".json") for t in ("", "_cpu", "_r2", "_r3")] if d else []
    cands = [q for q in cands if q.exists()]
    lb = None
    best = None
    for q in cands:
        c = _json.load(open(q))
        if c.get("theorem") != "LINKBOX" or "error" in c:
            continue
        key = (bool(c.get("proved")), (c.get("stats") or {}).get("n_certified") or 0)
        if best is None or key > best[0]:
            best = (key, q, c)
    if best is not None:
        p, c = best[1], best[2]
        if True:
            lb = {k: c.get(k) for k in ("theorem", "version", "applicable", "proved", "reason", "n_residual",
                                        "residual_summary", "seconds", "wall_seconds", "seconds_total", "host", "budget",
                                        "symmetry", "cone_coils", "kappa", "tol", "L", "box_params", "smax", "min_size")}
            st = c.get("stats") or {}
            if c.get("check"):
                ck = c["check"]
                lb["independent_check"] = {k: ck.get(k) for k in ("n", "confirmed", "max_arclength", "geometry_ok", "method")}
            lb["n_budget_left"] = st.get("n_budget_left", 0)
            lb["n_min_size"] = st.get("n_min_size", len([e for e in (c.get("residual") or []) if e.get("why") == "min_size"]))
            lb.update(n_cells_certified=st.get("n_certified"), n_cells_tested=st.get("n_tested"),
                      n_transfer=st.get("transfer"), n_section=st.get("section"),
                      n_presplit=st.get("n_presplit"), max_arclength=st.get("max_s"),
                      n_boxes=len(c.get("boxes") or []),
                      n_boxes_certified=len((c.get("symmetry") or {}).get("represent") or []),
                      file=str(p.relative_to(d.parents[1])), run=True)
            if c.get("boxes"):
                lb["margins_m"] = [min(b["margin"] for b in c["boxes"]), max(b["margin"] for b in c["boxes"])]
            if lb.get("proved") and source is not None:
                from . import nonstell_index as _nsi
                gok, probs = _nsi.verify_geometry(source, c)
                lb["geometry_verified"] = bool(gok)
                if not gok:
                    lb["proved"] = False
                    lb["open"] = ["geometric hypotheses not confirmed: " + "; ".join(probs)]
    if lb is None and source is not None:
        from . import nonstell_index as _nsi
        lb = _nsi.scope_record(source)
    if lb is None:
        return rec
    rec["linkbox"] = lb
    if lb.get("proved"):
        if rec.get("verdict") != "NO_TOROIDAL_FLUX_SURFACE":
            rec["verdict_before_linkbox"] = rec.get("verdict")
        rec["verdict"] = "NO_TOROIDAL_FLUX_SURFACE"
        args = ["LINKBOX", "LEMMA_A", "AMP"]
        rec["arguments"] = args + [a for a in rec.get("arguments", []) if a not in args and a not in ("EXC", "INDEX")]
        rec.pop("open", None)
    else:
        op = list(lb.get("open") or [])
        if lb.get("run") and lb.get("applicable"):
            rs = lb.get("residual_summary") or {}
            if rs.get("n") or lb.get("n_budget_left"):
                stop = ((c.get("stats") or {}).get("stop_reason") if best is not None else None) or \
                    ("cell cap" if (lb.get("wall_seconds") or 0) < 0.8 * (lb.get("budget") or 1e300) else "time budget")
                op.append("LINKBOX run (%.0f s on %s, stopped by the %s): %s box-boundary cells certified; not certified: %d "
                          "cells still queued (checkpoint kept, the run resumes) and %d cells at the minimum size; the "
                          "listed open cells lie %.3g..%.3g m from a coil, R %.3g..%.3g m, Z %.3g..%.3g m (kinds %s)"
                          % (lb.get("wall_seconds") or 0, lb.get("host"), stop, lb.get("n_cells_certified"),
                             lb.get("n_budget_left") or 0, lb.get("n_min_size") or 0,
                             rs.get("dist_to_coil_min", float("nan")), rs.get("dist_to_coil_max", float("nan")),
                             rs.get("R_range", [float("nan")] * 2)[0], rs.get("R_range", [float("nan")] * 2)[1],
                             rs.get("Z_range", [float("nan")] * 2)[0], rs.get("Z_range", [float("nan")] * 2)[1],
                             rs.get("kinds")))
        elif lb.get("run") and lb.get("applicable") is False:
            op.append("LINKBOX not applicable: %s" % lb.get("reason"))
        if not op:
            op.append("LINKBOX applies (every coil has toroidal winding 0 and a contractible box) but no certificate run "
                      "has finished for this source yet: every box-boundary point remains to be certified")
        rec["open"] = op
    return rec


SKEW_RING_NOTE = (
    "SKEW-RING: an exactly circular coil whose "
    "axis is skew to the z axis has toroidal flux surfaces of closed lines (winding 1, B_phi of one sign, iota = 0); "
    "for this source's PF-type coils (toroidal winding != 0) whether such surfaces exist is decided by geometry at "
    "or below the data's rounding (tilts ~1e-3 about axes meeting the z axis to ~1e-7 m, vertices ~5e-5 m off "
    "the fitted circles), which the field-line scan does not resolve: the numerical no-surface verdict is not safe")


def skew_ring_guard(rec):
    """A numerical no-surface verdict for a source with PF-type coils (toroidal winding w_z != 0) becomes UNDECIDED
    (SKEW-RING): the scan cannot exclude the closed-line surfaces of (near-)circular coils with (near-)skew axes.
    Proved verdicts are untouched."""
    if rec.get("verdict") != "NO_TOROIDAL_FLUX_SURFACE_NUMERICAL":
        return rec
    coils = (rec.get("3d") or {}).get("coils") or []
    if any(c.get("w_z", 0) != 0 for c in coils):
        rec["verdict_before_skew_ring"] = rec["verdict"]
        rec["verdict"] = "UNDECIDED"
        rec["note"] = (rec.get("note") + "; " if rec.get("note") else "") + SKEW_RING_NOTE
        if "SKEW-RING" not in rec.get("arguments", []):
            rec["arguments"] = list(rec.get("arguments", [])) + ["SKEW-RING"]
    return rec


def proved_part(rec):
    """What is proved (as opposed to numerical) for a 3D record, from its stored evidence."""
    res = rec["3d"]
    comps = res["shadow"]["comps"]
    if "PERTURB" in rec.get("arguments", []):
        return ("AXI on the axisymmetric projection (exact) + PERTURB for the deviation %.2g: no surface with a "
                "current-free interior off the conductor families; the 3D scan below is corroboration"
                % rec.get("axisym_projection", {}).get("deviation", float("nan")))
    if rec.get("verdict") == "NO_TOROIDAL_FLUX_SURFACE":
        txt = "exact (%s)" % ", ".join(a for a in rec.get("arguments", []) if a in ("LINKBOX", "SIGNTOP", "ANTISYM", "PLANAR", "AXI", "PERTURB"))
        stp = rec.get("signtop")
        if stp and stp.get("proved"):
            txt += (": sign topology of R B_phi on a voxel grid %s with tube radius %.3g m, both orientations; every "
                    "claim re-derived by an independent check; for the given currents" % (stp.get("shape"), stp.get("delta", 0)))
        lb = rec.get("linkbox")
        if lb and lb.get("proved"):
            txt += (": every point of the %d box boundaries (%d certified, the rest rotated copies) lies on a field line "
                    "reaching both signs of B_phi (%s cells, arc length <= %.3g m); for the given currents"
                    % (lb.get("n_boxes") or 0, lb.get("n_boxes_certified") or 0, lb.get("n_cells_certified"),
                       lb.get("max_arclength") or 0.0))
        return txt
    w = 2 * (res["box"]["R"][1] - res["box"]["R"][0]) / res["shadow"]["n"]
    fb = rec.get("flux_bound")
    fbtxt = ("; FLUXBOUND: every toroidal flux surface has |flux| <= %.3e Wb" % fb["Phi_max"]) if fb else ""
    if any(c["excluded_by_SHADOW"] for c in comps):
        return ("SHADOW: no toroidal flux surface containing a toroidal loop that avoids the dilated coil shadows "
                "(complement components with I_circ = 0 and I_hole = 0 cover %.4f of the box; shadow strip half-width "
                "%.3g m)%s; EXC/INDEX numerical elsewhere" % (res["shadow"]["excluded_area_frac"], w, fbtxt))
    if any(c["I_circ_zero"] for c in comps):
        return ("SHADOW: no toroidal flux surface containing a toroidal loop that avoids the dilated coil shadows and "
                "does not wind poloidally around the shadow of a coil encircling the z axis (I_circ = 0 in components "
                "covering %.4f of the box; tori winding around PF-type loops not covered)%s; EXC/INDEX numerical "
                "elsewhere" % (sum(c["area_frac"] for c in comps if c["I_circ_zero"]), fbtxt))
    return "none (every complement component links current): AMP gives no obstruction; numerical only" + fbtxt


def classify(source, n_scan=None, verbose=False):
    """Verdict and evidence for one source (see the module docstring and THEOREMS)."""
    t0 = time.time()
    rec = dict(id=source.id, version=VERSION, family=source.family, currents=source.meta.get("currents"),
               parts=[[p.kind, int(p.n)] for p in source.parts], nfp=int(source.nfp), stellsym=bool(source.stellsym),
               request=source.request)
    ok, why = scope(source)
    if not ok:
        rec.update(verdict="OUT_OF_SCOPE", reason=why)
        return _jsonable(rec)
    view = axisym_view(source)
    if view is not None:
        res, aux = analyse_axisym(source, view)
        rec["kind"] = "axisymmetric"
        rec["version"] = AXI_VERSION
        rec["axisym"] = res
        mp = res["max_principle"]
        theorem_check = dict(no_vacuum_extrema=mp["n_extrema_in_vacuum"] == 0,
                             no_index_plus_cells_in_vacuum=mp["n_index_plus_cells_in_vacuum"] == 0,
                             all_xpoints_saddles=all(x["det_hess"] < 0 for x in res["xpoints"]),
                             max_hessian_trace_rel=max([x["hess_trace_rel"] for x in res["xpoints"]], default=0.0),
                             max_Bpol_rel_at_xpoints=max([x["Bpol_rel"] for x in res["xpoints"]], default=0.0))
        rec["theorem_check"] = theorem_check
        is_dipole = source.meta.get("kind") == "dipole"
        if view["RB"] == 0.0:
            if is_dipole:
                fid = _fcoil_ids(source, view)
                rec["dipole"] = dipole_structure(view, res, aux, fid)
                rec["verdict"] = "DIPOLE_CLOSED_FIELD_LINES"
                rec["arguments"] = ["NOTOR", "AXI", "DIPOLE"]
            else:
                rec["verdict"] = "NO_TOROIDAL_FLUX_SURFACE"
                rec["arguments"] = ["NOTOR", "AXI"]
            rec["toroidal_flux_range"] = "none (every closed field line is poloidal; toroidal flux 0)"
        else:
            rec["verdict"] = "NO_VACUUM_INTERIOR_SURFACE"
            rec["arguments"] = ["AXI", "AMP", "INDEX"]
            fams = [f for f in aux["fams"] if f["cause"] in ("saddle", "psi0")]
            rec["conductor_enclosing_tori"] = dict(
                n_families=len(aux["fams"]),
                max_toroidal_flux_Wb=float(max([abs(f["tor_flux"]) for f in fams if f["cause"] == "saddle"], default=0.0)),
                n_unbounded_families=sum(f["cause"] == "psi0" for f in fams),
                largest=sorted(fams, key=lambda f: (f["cause"] != "saddle", -abs(f["tor_flux"])))[:6],
                n_families_ending_on_conductor=sum(f["cause"] == "conductor" for f in aux["fams"]))
            rec["toroidal_flux_range"] = ("none with a current-free interior; tori enclosing a conductor exist with "
                                          "toroidal flux up to the separatrix lobe of each conductor family")
            prof = []
            for f in sorted([f for f in fams if f.get("xpoint")], key=lambda f: -abs(f["tor_flux"]))[:2]:
                try:
                    prof.append(dict(conductors=f["conductors"][:20], n_conductors=len(f["conductors"]),
                                     tor_flux=f["tor_flux"], **family_iota_profile(aux["Fg"], view, f, f["xpoint"])))
                except Exception as e:
                    prof.append(dict(error=repr(e)))
            rec["conductor_family_iota"] = prof
            try:
                rec["iso_check"] = iso_check(res, aux)
            except Exception as e:
                rec["iso_check"] = dict(error=repr(e))
    else:
        rec["kind"] = "3D"
        near = axisym_view(source, tol=1e-5)
        if near is not None:
            r2, aux2 = analyse_axisym(source, near, engine_check=False)
            rec["axisym_projection"] = dict(deviation=near["deviation"], RB=near["RB"], xpoints=r2["xpoints"],
                                            max_principle=r2["max_principle"], families=r2["families"],
                                            grid=r2["grid"])
            if near["RB"]:
                fams = [f for f in aux2["fams"] if f["cause"] in ("saddle", "psi0")]
                rec["conductor_enclosing_tori"] = dict(
                    n_families=len(aux2["fams"]),
                    max_toroidal_flux_Wb=float(max([abs(f["tor_flux"]) for f in fams if f["cause"] == "saddle"], default=0.0)),
                n_unbounded_families=sum(f["cause"] == "psi0" for f in fams),
                    largest=sorted(fams, key=lambda f: (f["cause"] != "saddle", -abs(f["tor_flux"])))[:6],
                    n_families_ending_on_conductor=sum(f["cause"] == "conductor" for f in aux2["fams"]))
                prof = []
                for f in sorted([f for f in fams if f.get("xpoint")], key=lambda f: -abs(f["tor_flux"]))[:2]:
                    try:
                        prof.append(dict(conductors=f["conductors"][:20], n_conductors=len(f["conductors"]),
                                         tor_flux=f["tor_flux"], **family_iota_profile(aux2["Fg"], near, f, f["xpoint"])))
                    except Exception as e:
                        prof.append(dict(error=repr(e)))
                rec["conductor_family_iota"] = prof
                try:
                    rec["iso_check"] = iso_check(r2, aux2)
                except Exception as e:
                    rec["iso_check"] = dict(error=repr(e))
        F = fast64.Field(source)
        curves = closed_curves(source)
        planes = mirror_planes(source, curves, F, coil_box(curves, None, margin=0.15))
        rec["mirror_planes"] = planes
        anti = [p for p in planes if p["kind"] == "anti"]
        zs = np.concatenate([c["pts"][:, 2] for c in curves]) if curves else np.zeros(0)
        Lb = coil_box(curves, None, margin=0.15)["L"] if curves else 1.0
        planar_all = bool(curves) and not any(p.kind in ("toroidal", "axipoly") for p in source.parts) and \
            float(np.ptp(zs)) <= 1e-12 * Lb
        rec["coplanar_horizontal"] = dict(holds=planar_all, z_spread=float(np.ptp(zs)) if len(zs) else None)
        if n_scan is None:
            heavy = any(p.kind == "fourier" for p in source.parts)       # slow reference field: coarser scan
            n_scan = 15 if (anti or planar_all) else (25 if heavy else 33)  # exact argument: scan only corroborates
        ident0 = identity_from_mirrors(planes)
        res, aux = analyse_3d(source, n_scan=n_scan, phi0=0.1234567, F=F, skip_fixed_points=ident0 is not None)
        rec["3d"] = res
        ident = identity_from_mirrors(planes)
        rec["P_identity_period"] = ident
        exc = res["excursion"]
        fp = res["one_turn"]
        ncomp = exc["n_complete_fwd"] + exc["n_complete_bwd"]
        shadow_all = all(c["excluded_by_SHADOW"] for c in res["shadow"]["comps"])
        if planar_all:
            rec["verdict"] = "NO_TOROIDAL_FLUX_SURFACE"
            rec["arguments"] = ["PLANAR", "AMP", "SHADOW", "EXC"] + (["ANTISYM"] if anti else [])
        elif anti:
            rec["verdict"] = "NO_TOROIDAL_FLUX_SURFACE"
            rec["arguments"] = ["ANTISYM", "AMP", "SHADOW", "EXC"]
        elif ncomp == 0:
            rec["verdict"] = "NO_TOROIDAL_FLUX_SURFACE_NUMERICAL"
            rec["arguments"] = ["AMP", "SHADOW", "EXC"]
        elif ident is not None:
            rec["verdict"] = "DEGENERATE_CLOSED_LINE_SURFACES"
            rec["arguments"] = ["SYM", "AMP", "SHADOW", "EXC"]
        else:
            fps = fp.get("fixed_points", [])
            if fps:
                label_period1_lines(aux["F"], fps, closed_curves(source), 0.1234567)
                for f in fps:
                    if "I_link" in f:
                        f["I_link"] += res["RB"] * TWO_PI / MU0
            ell = [f for f in fps if f["index"] > 0 and f.get("type_reliable") and not f.get("winds_around_PF_coil")]
            ell_c = [f for f in fps if f["index"] > 0 and f.get("type_reliable") and f.get("winds_around_PF_coil")]
            unrel = [f for f in fps if not f.get("type_reliable")]
            near_id = fp.get("median_delta", np.inf) < 1e-4 * res["box"]["L"]
            rec["period1_summary"] = dict(n=len(fps), n_index_plus_plasma=len(ell), n_index_plus_around_PF=len(ell_c),
                                          n_saddles=sum(f["index"] < 0 for f in fps), n_type_unreliable=len(unrel),
                                          index_sum_cells=fp.get("index_sum_cells"),
                                          n_index_plus_cells=sum(1 for c in fp.get("index_cells", []) if isinstance(c[3], int) and c[3] > 0),
                                          n_index_cells_unresolved=fp.get("n_index_cells_unresolved"),
                                          near_identity_map=bool(near_id))
            if ell and not near_id:
                try:
                    rec["elliptic_regions"] = [dict(R=f["R"], Z=f["Z"], **elliptic_region(aux["F"], f, 0.1234567, res["box"],
                                                                                        nturns=150))
                                               for f in ell[:3]]
                except Exception as e:
                    rec["elliptic_regions_error"] = repr(e)
            if not ell:
                rec["verdict"] = "NO_TOROIDAL_FLUX_SURFACE_NUMERICAL"
                rec["arguments"] = ["AMP", "SHADOW", "INDEX", "EXC"]
                if ell_c:
                    rec["note"] = ("elliptic period-1 lines exist only winding around PF-type coils (resonant "
                                   "islands of the conductor families)")
            else:
                rec["verdict"] = "NEAR_IDENTITY_ELLIPTIC" if near_id else "ELLIPTIC_PERIOD1_LINE"
                rec["arguments"] = ["INDEX", "AMP", "SHADOW"]
            if not ell and not ell_c and rec["period1_summary"]["n_index_plus_cells"] > 0:
                rec["verdict"] = "UNDECIDED"
                rec["note"] = ("index +1 cells of P(x) - x exist but no reliable index +1 period-1 line was refined "
                               "(candidate budget or tracer accuracy)")
        skew_ring_guard(rec)
        rec["proved_part"] = proved_part(rec)
        annotate_unresolved(rec, source)
        if near is not None:
            rec["verdict_3d"] = rec.get("verdict")
            rec["verdict"] = "NO_VACUUM_INTERIOR_SURFACE" if near["RB"] else "NO_TOROIDAL_FLUX_SURFACE"
            rec["arguments"] = ["AXI", "PERTURB"] + [a for a in rec.get("arguments", []) if a not in ("AXI",)]
            rec["toroidal_flux_range"] = ("none with a current-free interior away from the conductors; tori enclosing a "
                                          "conductor (and resonant islands of the conductor families, width ~ "
                                          "sqrt(deviation)) near the coils")
    rec["seconds"] = time.time() - t0
    return _jsonable(rec)


def closed_line_polygon(F, R0, Z0, phi0=0.0, n=192):
    """The period-1 line through (R0, phi0, Z0) as a closed polygon (n points, reference tracer)."""
    from .reference import trace as RT
    phis = phi0 + TWO_PI * np.arange(1, n) / n
    out = RT.poincare(F, np.array([R0]), np.array([Z0]), phis, 1, phi0=phi0, h=TWO_PI / 192, stages=6,
                      tol=1e-14, doubling=False)
    R = np.concatenate([[R0], out["R"][0, 0, :]]); Z = np.concatenate([[Z0], out["Z"][0, 0, :]])
    ph = np.concatenate([[phi0], np.sort(np.mod(phis - phi0, TWO_PI)) + phi0])
    return np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)


def label_period1_lines(F, fps, curves, phi0):
    """Linking numbers of every period-1 line with every coil (Gauss integral of polygons) and
    the label 'winds_around_PF_coil' when it links a planar horizontal loop (a PF-type coil:
    linking such a loop means winding poloidally around its wire)."""
    pf = []
    for k, c in enumerate(curves):
        cf = circle_fit(c["pts"])
        if cf["planarity"] < 1e-6 and abs(cf["normal"][2]) > 1 - 1e-6 and toroidal_winding(c["pts"]) != 0:
            pf.append(k)
    for f in fps:
        try:
            Lp = closed_line_polygon(F, f["R"], f["Z"], phi0)
            lk = [gauss_linking(Lp, c["pts"]) for c in curves]
            f["lk_coils"] = [int(round(v)) for v in lk]
            f["lk_max_dev_from_integer"] = float(max(abs(v - round(v)) for v in lk)) if lk else 0.0
            f["I_link"] = float(sum(round(v) * c["I"] for v, c in zip(lk, curves)))
            f["winds_around_PF_coil"] = bool(any(round(lk[k]) != 0 for k in pf))
        except Exception as e:
            f["lk_error"] = repr(e)
    return pf


def elliptic_region(F, fp, phi0, box, nstart=12, nturns=150, rmax=None):
    """Field lines started on a ray from an elliptic period-1 point (outward in R) and followed
    nturns toroidal turns with the reference tracer: for each start the rotation number about
    the point (mean winding angle per turn), whether it stays (not lost, bounded by the box),
    and the toroidal flux inside its section orbit (polygon of the orbit points sorted by angle,
    flux = int B_phi dA by triangle quadrature).  The outermost start that stays and keeps a
    monotone rotation profile bounds the region of (numerically) closed surfaces."""
    from .reference import trace as RT
    R0, Z0 = fp["R"], fp["Z"]
    L = box["L"]
    rmax = rmax or 0.25 * L
    rs = rmax * (np.arange(1, nstart + 1) / nstart) ** 1.5
    out = RT.poincare(F, R0 + rs, np.full(nstart, Z0), [phi0], nturns, phi0=phi0, h=TWO_PI / 64, stages=6,
                      tol=1e-13, doubling=False, bounds=(box["R"][0], box["R"][1], max(abs(box["Z"][0]), abs(box["Z"][1]))))
    rows = []
    for k in range(nstart):
        R = out["R"][k, :, 0]; Z = out["Z"][k, :, 0]
        ok = np.isfinite(R)
        lost = bool(out["lost"][k] >= 0) or not ok.all()
        ang = np.unwrap(np.arctan2(Z[ok] - Z0, R[ok] - R0)) if ok.sum() > 2 else np.array([0.0])
        a0 = math.atan2(0.0, rs[k])
        ang = np.concatenate([[a0], ang])
        ang = np.unwrap(ang)
        rot = float((ang[-1] - ang[0]) / (TWO_PI * max(ok.sum(), 1)))
        flux = None
        if not lost:
            th = np.mod(np.arctan2(Z - Z0, R - R0), TWO_PI)
            o = np.argsort(th)
            Pr, Pz = R[o], Z[o]
            # flux through the star-shaped polygon (centre = the O-point), 3-point rule per triangle
            tot = 0.0
            for i in range(len(Pr)):
                j = (i + 1) % len(Pr)
                tri = np.array([[R0, Z0], [Pr[i], Pz[i]], [Pr[j], Pz[j]]])
                A = 0.5 * ((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1]))
                mids = np.array([(tri[0] + tri[1]) / 2, (tri[1] + tri[2]) / 2, (tri[2] + tri[0]) / 2])
                X = np.stack([mids[:, 0] * math.cos(phi0), mids[:, 0] * math.sin(phi0), mids[:, 1]], -1)
                B = F.evaluate(X, grad=False)["B"]
                Bp = -B[:, 0] * math.sin(phi0) + B[:, 1] * math.cos(phi0)
                tot += A * Bp.mean()
            flux = float(tot)
        rows.append(dict(r=float(rs[k]), lost=lost, rotation=rot, toroidal_flux=flux,
                         spread_R=float(np.nanmax(R) - np.nanmin(R)) if ok.any() else None))
    good = []
    for r in rows:
        if r["lost"]:
            break
        good.append(r)
    return dict(starts=rows, n_confined_from_axis=len(good),
                max_flux_confined=max([abs(r["toroidal_flux"]) for r in good if r["toroidal_flux"] is not None], default=0.0),
                nturns=nturns, note="numerical (tracing); confined = not lost within nturns; surfaces vs thin island "
                                    "chains not distinguished")


def family_iota_profile(Fe, view, fam, xpt, n=8, nturns=40):
    """Rotational transform of the nested tori psi = c around the conductor(s) of a family, with
    a toroidal field (Fe: the reference field of the source or of its ring approximation): lines started on the segment from the family's psi-extremum conductor point
    towards its separatrix X-point (fractions 0.1 .. 0.9) and followed nturns toroidal turns with
    the reference tracer; iota = (poloidal angle about the conductor) / (2 pi nturns).  iota -> 0
    at the separatrix (period diverges), -> infinity at a filament."""
    from .reference import trace as RT
    ids = fam["conductors"]
    nR = len(view["R"])
    pts = []
    for k in ids:
        if k < nR:
            pts.append((view["R"][k], view["Z"][k], abs(view["I"][k])))
        else:
            V, c = view["polys"][k - nR]
            pts.append((V[:, 0].mean(), V[:, 1].mean(), abs(c)))
    # centre: the family's conductor nearest to the X-point among the strongest half (every contour of the
    # family encloses it, so the poloidal angle about it counts the poloidal turns)
    w = np.array([p[2] for p in pts])
    strong = [p for p in pts if p[2] >= 0.5 * w.max()]
    Rc, Zc = min(((p[0], p[1]) for p in strong), key=lambda q: math.hypot(q[0] - xpt[0], q[1] - xpt[1]))
    Rc, Zc = float(Rc), float(Zc)
    fr = np.linspace(0.1, 0.9, n)
    R0 = Rc + fr * (xpt[0] - Rc); Z0 = Zc + fr * (xpt[1] - Zc)
    out = RT.poincare(Fe, R0, Z0, TWO_PI * np.arange(1, 33) / 32, nturns, phi0=0.0, h=TWO_PI / 256, stages=6,
                      tol=1e-13, doubling=False)
    rows = []
    for k in range(n):
        R = np.concatenate([[R0[k]], out["R"][k].ravel()]); Z = np.concatenate([[Z0[k]], out["Z"][k].ravel()])
        if not np.all(np.isfinite(R)):
            rows.append(dict(frac=float(fr[k]), R=float(R0[k]), Z=float(Z0[k]), iota=None, lost=True))
            continue
        th = np.unwrap(np.arctan2(Z - Zc, R - Rc))
        psi, _, _ = psi_on(Fe, np.array([R0[k]]), np.array([Z0[k]]))
        rows.append(dict(frac=float(fr[k]), R=float(R0[k]), Z=float(Z0[k]), psi=float(psi[0]),
                         iota=float((th[-1] - th[0]) / (TWO_PI * nturns)), lost=False))
    return dict(center=[Rc, Zc], xpoint=list(xpt), profile=rows, nturns=nturns)


def iso_check(res, aux, n=16):
    """|P(x) - x| at vacuum grid nodes (with a toroidal field): an isochronous family (P = identity on
    an open set, AXI case (b)) would make it vanish on a whole region.  Nodes within 2 grid cells of
    a conductor or of an X-point are skipped; the period map uses the grid field (rings)."""
    Rg, Zg, cond = aux["Rg"], aux["Zg"], aux["cond"]
    h = res["grid"]["h"]
    Rs = np.linspace(Rg[0], Rg[-1], n + 2)[1:-1]; Zs = np.linspace(Zg[0], Zg[-1], n + 2)[1:-1]
    RR, ZZ = np.meshgrid(Rs, Zs, indexing="ij")
    from scipy import ndimage
    cd = ndimage.binary_dilation(cond >= 0, iterations=2)
    i = np.clip(np.round((RR - Rg[0]) / (Rg[1] - Rg[0])).astype(int), 0, len(Rg) - 1)
    j = np.clip(np.round((ZZ - Zg[0]) / (Zg[1] - Zg[0])).astype(int), 0, len(Zg) - 1)
    ok = ~cd[i, j]
    for x in res["xpoints"]:
        ok &= np.hypot(RR - x["R"], ZZ - x["Z"]) > 2 * h
    R0, Z0 = RR[ok], ZZ[ok]
    R1, Z1, err, lost = one_turn(aux["Fg"], R0, Z0, 0.0, doubling=False)
    d = np.hypot(R1 - R0, Z1 - Z0)
    good = np.isfinite(d)
    if not good.any():
        return dict(n=int(ok.sum()), n_traced=0)
    k = int(np.argmin(np.where(good, d, np.inf)))
    return dict(n=int(ok.sum()), n_traced=int(good.sum()), min_delta=float(d[good].min()),
                median_delta=float(np.median(d[good])), at=[float(R0[k]), float(Z0[k])], h=float(h))


def annotate_unresolved(rec, source):
    """For 3D records: distance (in the meridional plane) from every index cell of P(x) - x whose
    index could not be confirmed to the nearest coil projection, in units of the scan spacing.
    Unresolved cells sit where neighbouring lines stop completing a turn or P(x) - x turns too fast,
    i.e. next to coils; a fixed point hidden there would be the axis of a surface within that
    distance of a coil."""
    ot = rec.get("3d", {}).get("one_turn", {})
    cells = [c for c in ot.get("index_cells", []) if not isinstance(c[3], int)]
    if not cells:
        rec["unresolved_cells"] = dict(n=0)
        return rec
    pts = []
    for c in closed_curves(source):
        x = c["pts"]
        pts.append(np.stack([np.hypot(x[:, 0], x[:, 1]), x[:, 2]], -1))
    P = np.concatenate(pts) if pts else np.zeros((0, 2))
    h = rec["3d"]["excursion"]["h"][0]
    d = [float(np.min(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]))) if len(P) else float("inf") for c in cells]
    rec["unresolved_cells"] = dict(n=len(cells), n_coarse_plus=sum(1 for c in cells if c[2] > 0),
                                   max_dist_to_coil_m=float(max(d)), max_dist_to_coil_in_h=float(max(d) / h),
                                   cells=[[c[0], c[1], c[2], dd] for c, dd in zip(cells, d)])
    return rec


def refresh_iota(rec, source):
    """Recompute the conductor-family iota profiles of a stored record (axisymmetric with a toroidal
    field, or near-axisymmetric) from its stored families, without redoing the rest."""
    near = rec.get("kind") == "3D"
    view = axisym_view(source, tol=1e-5 if near else 1e-12)
    if view is None or not view["RB"]:
        return rec
    h = (rec["axisym_projection"] if near else rec["axisym"])["grid"]["h"]
    Fg = fast64.Field(grid_source(view, h))
    fams = [f for f in rec["conductor_enclosing_tori"]["largest"] if f["cause"] == "saddle" and f.get("xpoint")]
    prof = []
    for f in sorted(fams, key=lambda f: -abs(f["tor_flux"]))[:2]:
        try:
            prof.append(dict(conductors=f["conductors"][:20], n_conductors=len(f["conductors"]),
                             tor_flux=f["tor_flux"], **family_iota_profile(Fg, view, f, f["xpoint"])))
        except Exception as e:
            prof.append(dict(error=repr(e)))
    rec["conductor_family_iota"] = prof
    return rec
