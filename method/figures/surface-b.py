"""Figure 1 of the method page: the surface is a function of two angles, and in them a field line is straight.

    python method/figures/surface-b.py              # draw from data/surface-b.npz (ten seconds)
    python method/figures/surface-b.py --compute    # solve and trace again (two minutes more), then draw

One field period of the flux surface torux verifies for HSX's coils at 0.0359221439332229 Wb, the request of
examples/any_source.py,

    torux.solve("examples/hsx.json", flux=0.0359221439332229, budget_s=900)

drawn in perspective as a solid cut by the planes phi = 0 and phi = pi / 2, seen from inside the torus (the only
side from which both cuts face the eye, so phi increases from right to left), hidden lines removed.

What is drawn, and from what:

- The tube: the series of answer.surface with every coefficient (M, N = 192, 96), R = sum rc[m, n + N]
  cos(m theta - n nfp phi) and Z the same with zs and sin, theta the straight-field-line angle and phi the
  cylindrical one. Its outline is the contour generator, where the tube is seen edge on, and the rims of the two
  cuts. The cut faces are hatched, as a section is.
- The net, fine: the curves theta = 2 pi k / 11 and phi = (pi / 2) j / 3 (j = 0 and 3 are the cuts). The steps
  are chosen so that a field line crosses iota (pi / 6) / (2 pi / 11) = 0.995 cells of theta for every cell of
  phi: from a corner of the net it runs corner to corner, missing each next corner by under 1.5 % of a cell.
- The field lines, heavy: theta = theta0 + iota phi from the corners theta0 = 2 pi k / 11, k = 3 and 4, of the
  net on the cut phi = 0, the series' own lines. --compute also traces a field line from every corner of the net
  on that cut with torux's independent tracer (torux.reference.trace, Gauss-Legendre collocation on its own field
  evaluator) and compares it with the series' line at 64 angles over the period: they agree to 2.0e-15 m, the
  tracer's own error being 3e-16 m.
- On each cut, theta at the net's steps, ticked inward, the tick of theta = 0 the longest. Stellarator symmetry
  puts theta = 0 on the outboard midplane of both cuts, and theta runs counter-clockwise in the (R, Z) plane: up
  the outboard side, on both cuts, though they are seen from opposite sides of their planes. Uneven ticks on an
  even angle: theta is neither arc length nor the geometric angle.
- Lettered: theta and its sense on each cut, R and Z in the plane of the cut phi = 0, phi and its sense on an arc
  of constant R and Z from cut to cut below the tube, and where the line from k = 4 enters (theta0) and leaves
  (theta0 + iota pi / 2), dotted.

Visibility is computed, not drawn by eye. The solid is a mesh of the series (520,000 triangles, within 3e-5 m of
it) with a fan across each cut, and rays from the eye are cast against it. Every curve is cut exactly where it
crosses the contour generator (a root of n . (X - E), by bisection) or passes behind an outline (the crossing of
their images). Each label stands 3 units clear of every line and every other label, off the solid and on the
page: the first such of the places given for it, or for one on a leader, the one whose leader crosses fewest
lines. If there is none the drawing stops. The numbers in the figure and those a caption needs are written to
surface-b.json beside this file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data", "surface-b.npz")
NUMBERS = os.path.join(HERE, "surface-b.json")
SOURCE = os.path.join(REPO, "examples", "hsx.json")
SOURCE_NAME = "examples/hsx.json"
FLUX = 0.0359221439332229       # Wb, the request of examples/any_source.py
NTH = 11                        # the net: theta in steps of 2 pi / 11 ...
NPH = 3                         # ... and phi in steps of (pi / 2) / 3
TRACE_SECTIONS = 64             # points per traced line over the period, where it is compared with the series
P0, P1 = 0.0, 0.5 * math.pi     # the cuts: one field period of the four
VIEW = dict(azimuth=45.0,       # of the line of sight, from the x axis: looking out from inside the torus
            elevation=8.0,      # degrees above the plane z = 0
            distance=6.0,       # m, from the eye to the target
            target=(0.72, 0.72, 0.0))
LINES = (3, 4)                  # field lines from the corners theta0 = 2 pi k / NTH of the net on the cut phi = 0
LETTERED = 4                    # the one whose ends are lettered
TICK = 0.03                     # m, the ticks of theta on the cut faces (theta = 0: 1.7 times)
THETA_ARC = (0.14, 0.95, 0.035)  # theta's arrow on each cut: from, to (rad), and its distance outside the rim (m)
RZ = (0.05, -0.30, 0.13)        # R and Z in the plane phi = 0: origin outboard of the cut and below it, length (m)
DIM = (1.45, -0.46)             # phi's arc: its R and Z (m)
MARGINS = (30.0, 30.0, 34.0, 38.0)   # page units left, right, above, below the drawing: room for the lettering
LEADER_SPANS = ((95.0, 165.0), (200.0, 330.0))   # where the entry's and the exit's labels may stand (degrees)


# ---------------------------------------------------------------------------------------------- the run

def compute():
    """Solve the request, keep the series, and trace a field line from every corner of the net on the section
    phi = 0 with torux's independent tracer, to compare with the straight lines of the series."""
    import torux
    from torux.reference import fast64, trace as reference_trace

    t0 = time.time()
    a = torux.solve(SOURCE, flux=FLUX, budget_s=900)
    wall = time.time() - t0
    if a.status != "surface":
        raise RuntimeError(f"the request did not give a verified surface: {a.status}")
    t = a.surface
    ev = a.evidence
    phi0 = float(getattr(t, "phi0", 0.0))
    # the lines: from theta = 2 pi k / NTH on phi = 0, to phi = pi / 2, by the reference tracer (Gauss-Legendre,
    # its own field evaluator) with steps of 2 pi / 1024, a quarter of the spacing of the points compared, so that
    # its step-doubling estimate means something
    th0 = 2 * np.pi * np.arange(NTH) / NTH
    x0 = a.points(th0, np.zeros(NTH))
    R0, Z0 = np.hypot(x0[:, 0], x0[:, 1]), x0[:, 2]
    phis = 0.5 * np.pi * np.arange(1, TRACE_SECTIONS + 1) / TRACE_SECTIONS
    F = fast64.Field(torux.load(SOURCE))
    t1 = time.time()
    res = reference_trace.poincare(F, R0, Z0, phis, 1, phi0=0.0, h=2 * np.pi / 1024)
    trace_s = time.time() - t1
    if (res["lost"] >= 0).any():
        raise RuntimeError("a traced line was lost")
    Rt, Zt = res["R"][:, 0, :], res["Z"][:, 0, :]
    TH = th0[:, None] + a.iota * phis[None, :]
    xs = a.points(TH, np.broadcast_to(phis, TH.shape))
    dist = np.hypot(Rt - np.hypot(xs[..., 0], xs[..., 1]), Zt - xs[..., 2])
    facts = {
        "torux": torux.__version__,
        "status": a.status,
        "solve_s": wall,
        "trace_s": trace_s,
        "iota": float(a.iota),
        "flux_Wb": float(a.flux),
        "major_radius_m": float(a.major_radius),
        "minor_radius_m": float(a.minor_radius),
        "volume_m3": float(a.volume),
        "bn_solver": float(ev["solver"]["bn_off"]),
        "bn_independent": float(ev["independent"]["bn_sup"]),
        "flux_rel_independent": float(ev["independent"]["flux_rel_err"]),
        "trace_err_estimate_m": float(np.nanmax(res["err"])),
        "trace_h": 2 * np.pi / 1024,
        "trace_evals_per_line": int(np.max(res["evals"])),
    }
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, rc=np.asarray(t.rc, float), zs=np.asarray(t.zs, float), iota=float(a.iota),
                        flux=float(a.flux), nfp=int(t.nfp), phi0=phi0, M=int(t.M), N=int(t.N), th0=th0, phis=phis,
                        dist=dist, err=res["err"][:, 0, :], facts=json.dumps(facts))
    print(f"solved in {wall:.1f} s, traced in {trace_s:.1f} s: largest distance traced against the series "
          f"{dist.max():.2e} m")


# ---------------------------------------------------------------------------------------------- the surface

class Series:
    """torux's series for the surface: R = sum rc[m, n + N] cos(m theta - n nfp (phi - phi0)), Z the same with zs and
    sin, theta the straight-field-line angle and phi the cylindrical angle (answer.surface, docs/usage.md)."""

    def __init__(self, d):
        self.rc, self.zs = np.asarray(d["rc"], float), np.asarray(d["zs"], float)
        self.iota, self.nfp, self.phi0 = float(d["iota"]), int(d["nfp"]), float(d["phi0"])
        self.m = np.arange(self.rc.shape[0])
        N = (self.rc.shape[1] - 1) // 2
        self.n = np.arange(-N, N + 1)
        self.nw = self.n * self.nfp

    def rz(self, theta, phi, deriv=False):
        """R, Z (and R_theta, Z_theta, R_phi, Z_phi) at points theta, phi of any one shape."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        th, ph = theta.ravel(), phi.ravel()
        out = [np.empty(th.size) for _ in range(6 if deriv else 2)]
        m = self.m
        for i in range(0, th.size, 2048):
            sl = slice(i, i + 2048)
            am, an = np.outer(th[sl], m), np.outer(self.nfp * (ph[sl] - self.phi0), self.n)
            Cm, Sm, Cn, Sn = np.cos(am), np.sin(am), np.cos(an), np.sin(an)
            rC, rS, zC, zS = Cn @ self.rc.T, Sn @ self.rc.T, Cn @ self.zs.T, Sn @ self.zs.T
            out[0][sl] = np.sum(Cm * rC + Sm * rS, 1)
            out[1][sl] = np.sum(Sm * zC - Cm * zS, 1)
            if deriv:
                out[2][sl] = np.sum(m * (Cm * rS - Sm * rC), 1)
                out[3][sl] = np.sum(m * (Cm * zC + Sm * zS), 1)
                rCn, rSn = Cn @ (self.rc * self.nw).T, Sn @ (self.rc * self.nw).T
                zCn, zSn = Cn @ (self.zs * self.nw).T, Sn @ (self.zs * self.nw).T
                out[4][sl] = np.sum(Sm * rCn - Cm * rSn, 1)
                out[5][sl] = -np.sum(Cm * zCn + Sm * zSn, 1)
        return [o.reshape(theta.shape) for o in out]

    def grid(self, th, ph, deriv=False):
        """The same on the tensor grid th (a,) by ph (b,), by separable sums."""
        th, ph = np.asarray(th, float), np.asarray(ph, float)
        m = self.m
        am, an = np.outer(th, m), np.outer(self.nfp * (ph - self.phi0), self.n)
        Cm, Sm, Cn, Sn = np.cos(am), np.sin(am), np.cos(an), np.sin(an)
        rc, zs = self.rc, self.zs
        out = [Cm @ rc @ Cn.T + Sm @ rc @ Sn.T, Sm @ zs @ Cn.T - Cm @ zs @ Sn.T]
        if deriv:
            out += [(Cm * m) @ rc @ Sn.T - (Sm * m) @ rc @ Cn.T, (Cm * m) @ zs @ Cn.T + (Sm * m) @ zs @ Sn.T,
                    Sm @ (rc * self.nw) @ Cn.T - Cm @ (rc * self.nw) @ Sn.T,
                    -(Cm @ (zs * self.nw) @ Cn.T + Sm @ (zs * self.nw) @ Sn.T)]
        return out

    @staticmethod
    def cartesian(phi, R, Z, Rt=None, Zt=None, Rp=None, Zp=None):
        c, s = np.cos(phi), np.sin(phi)
        X = np.stack([R * c, R * s, Z], -1)
        if Rt is None:
            return X
        Xt = np.stack([Rt * c, Rt * s, Zt], -1)
        Xp = np.stack([Rp * c - R * s, Rp * s + R * c, Zp], -1)
        return X, Xt, Xp

    def xyz(self, theta, phi, deriv=False):
        """Cartesian points, and with deriv the tangents X_theta and X_phi."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        return self.cartesian(phi, *self.rz(theta, phi, deriv))

    def xyz_grid(self, th, ph, deriv=False):
        P = np.broadcast_to(np.asarray(ph, float)[None, :], (len(th), len(ph)))
        return self.cartesian(P, *self.grid(th, ph, deriv))


def plane(phi, R, Z):
    """The point (R, Z) of the half plane at cylindrical angle phi, in Cartesian coordinates."""
    R, Z = np.broadcast_arrays(np.asarray(R, float), np.asarray(Z, float))
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], -1)


# ---------------------------------------------------------------------------------------------- the view

class Camera:
    """A perspective view from the eye E towards a target, z up: image coordinates (u, v), u to the right."""

    def __init__(self, eye, target, f=1.0):
        self.E = np.asarray(eye, float)
        d = np.asarray(target, float) - self.E
        self.d = d / np.linalg.norm(d)
        r = np.cross(self.d, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.d)
        self.f = float(f)

    def __call__(self, X):
        v = np.asarray(X, float) - self.E
        z = v @ self.d
        return np.stack([self.f * (v @ self.r) / z, self.f * (v @ self.u) / z], -1)


class Solid:
    """The period of the tube between its two cut faces as triangles, for casting rays: a mesh of the series (its
    chords lie within 3e-5 m of the surface) and a fan across each cut face (each section is star-shaped about
    its centroid, which is checked), in blocks, each with its bounding box."""

    def __init__(self, series, p0, p1, nth=720, nph=360, block=12):
        th = 2 * np.pi * np.arange(nth) / nth
        ph = np.linspace(p0, p1, nph + 1)
        X = series.xyz_grid(th, ph)
        I, J = np.meshgrid(np.arange(nth), np.arange(nph), indexing="ij")
        I2 = (I + 1) % nth
        quads = np.stack([np.stack([X[I, J], X[I2, J], X[I2, J + 1]], -2),
                          np.stack([X[I, J], X[I2, J + 1], X[I, J + 1]], -2)], 2)
        blocks = [quads[a:a + block, b:b + block].reshape(-1, 3, 3)
                  for a in range(0, nth, block) for b in range(0, nph, block)]
        for p in (p0, p1):
            C = series.xyz(th, np.full(nth, p))
            R, Z = np.hypot(C[:, 0], C[:, 1]), C[:, 2]
            a = np.unwrap(np.arctan2(Z - Z.mean(), R - R.mean()))
            if not np.all(np.diff(a) > 0):
                raise RuntimeError("a cut face is not star-shaped about its centroid: its fan would be wrong")
            k = np.arange(nth)
            fan = np.stack([np.repeat(C.mean(0)[None], nth, 0), C[k], C[(k + 1) % nth]], 1)
            blocks += [fan[a:a + 4 * block] for a in range(0, nth, 4 * block)]
        self.lo = np.array([b.reshape(-1, 3).min(0) for b in blocks])
        self.hi = np.array([b.reshape(-1, 3).max(0) for b in blocks])
        self.tris = np.concatenate(blocks)
        self.off = np.cumsum([0] + [len(b) for b in blocks])

    def hit(self, o, d, tmax):
        """Whether the ray o + t d (d a unit vector) meets a triangle for some t in (0, tmax)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (self.lo - o) / d, (self.hi - o) / d
            tn = np.nanmax(np.minimum(t1, t2), 1)
            tf = np.nanmin(np.maximum(t1, t2), 1)
        near = np.nonzero((tf >= tn) & (tf > 0) & (tn < tmax))[0]
        if len(near) == 0:
            return False
        T = np.concatenate([self.tris[self.off[k]:self.off[k + 1]] for k in near])
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        p = np.cross(d, e2)
        det = np.sum(e1 * p, 1)
        ok = np.abs(det) > 1e-18
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - T[:, 0]
        u = np.sum(s * p, 1) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = np.sum(e2 * q, 1) * inv
        return bool(np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < tmax)))


class HiddenLines:
    """What of the cut period of the tube the eye sees.  Every curve drawn on the tube is a straight line in
    (theta, phi), and its visibility changes only where it crosses the contour generator (where the tube is seen
    edge on: a root of n . (X - E), found by bisection) or where its image crosses the image of an outline (a
    silhouette or a cut face's rim: a crossing of two polylines).  Points between are classified by casting the
    ray from the eye against the solid."""

    def __init__(self, series, cam, p0, p1, grid=(1440, 720)):
        self.s, self.cam, self.p0, self.p1 = series, cam, p0, p1
        self.solid = Solid(series, p0, p1)
        th = 2 * np.pi * np.arange(grid[0]) / grid[0]
        ph = np.linspace(p0, p1, grid[1] + 1)
        X, Xt, Xp = series.xyz_grid(th, ph, deriv=True)
        n = np.cross(Xt, Xp)
        self.sign = float(np.sign(np.sum(n * (X - X.mean(0, keepdims=True)))))   # outward
        self._g = (th, ph, np.sum(n * self.sign * (X - cam.E), -1))

    def g(self, th, ph):
        """n . (X - E) / |X - E|, n the unit outward normal: negative where the tube faces the eye."""
        X, n = self.normal(th, ph)
        v = X - self.cam.E
        return np.sum(n * v, -1) / np.linalg.norm(v, axis=-1)

    def normal(self, th, ph):
        X, Xt, Xp = self.s.xyz(th, ph, deriv=True)
        n = np.cross(Xt, Xp) * self.sign
        return X, n / np.linalg.norm(n, axis=-1, keepdims=True)

    def occluded(self, X, eps=3e-3):
        """Whether the solid lies between the eye and each point X, nearer than eps (m) short of it."""
        X = np.atleast_2d(X)
        out = np.zeros(len(X), bool)
        for i, x in enumerate(X):
            d = x - self.cam.E
            L = np.linalg.norm(d)
            out[i] = self.solid.hit(self.cam.E, d / L, L - eps)
        return out

    def section(self, phi, n=2048):
        th = 2 * np.pi * np.arange(n) / n
        return self.s.xyz(th, np.full(n, phi))

    def contours(self):
        """The contour generator, as curves (theta, phi): the zero set of g on a fine grid, each point then moved
        onto it by Newton steps (along theta alone on the cut planes, so that a curve ends on the cut)."""
        import contourpy
        th, ph, G = self._g
        lines = contourpy.contour_generator(ph, np.append(th, 2 * np.pi), np.vstack([G, G[:1]])).lines(0.0)
        out, h = [], 1e-6
        for seg in lines:
            p, t = seg[:, 0].copy(), seg[:, 1].copy()
            for _ in range(3):
                pa, pb = np.clip(p - h, self.p0, self.p1), np.clip(p + h, self.p0, self.p1)
                g0 = self.g(t, p)
                gt = (self.g(t + h, p) - self.g(t - h, p)) / (2 * h)
                gp = (self.g(t, pb) - self.g(t, pa)) / (pb - pa)
                edge = (p <= self.p0 + 1e-12) | (p >= self.p1 - 1e-12)
                gp = np.where(edge, 0.0, gp)
                n2 = gt * gt + gp * gp
                t, p = t - g0 * gt / n2, np.clip(p - g0 * gp / n2, self.p0, self.p1)
            out.append((np.mod(t, 2 * np.pi), p))
        return out

    def contour_visible(self, t, p, delta=3e-4):
        """A contour point is seen when the ray to a point just outside the tube beside it meets nothing."""
        X, n = self.normal(t, p)
        return ~self.occluded(X + delta * n)

    def outlines(self):
        """The visible pieces of the contour generator as image polylines, and the two rims."""
        rims = [np.vstack([self.cam(self.section(p)), self.cam(self.section(p)[:1])]) for p in (self.p0, self.p1)]
        cons = self.contours()
        imgs = [self.cam(self.s.xyz(t, p)) for t, p in cons]
        pieces = []
        for ci, (t, p) in enumerate(cons):
            vis, uv = self.contour_visible(t, p), imgs[ci]
            others = [im for j, im in enumerate(imgs) if j != ci] + rims
            for a, b in _runs(vis):
                ends = []
                for i0, i1 in ((a - 1, a), (b - 1, b)):
                    if i0 < 0 or i1 >= len(t):
                        ends.append(None)
                        continue
                    r = _first_crossing(uv[i0], uv[i1], others)
                    if r is None:
                        r = _bisect(lambda q: bool(self.contour_visible(np.array([t[i0] + q * (t[i1] - t[i0])]),
                                                                        np.array([p[i0] + q * (p[i1] - p[i0])]))[0]),
                                    bool(vis[i0]))
                    ends.append((t[i0] + r * (t[i1] - t[i0]), p[i0] + r * (p[i1] - p[i0])))
                tt, pp = list(t[a:b]), list(p[a:b])
                if ends[0] is not None:
                    tt.insert(0, ends[0][0])
                    pp.insert(0, ends[0][1])
                if ends[1] is not None:
                    tt.append(ends[1][0])
                    pp.append(ends[1][1])
                if len(tt) >= 2:
                    pieces.append(self.cam(self.s.xyz(np.array(tt), np.array(pp))))
        return pieces, rims

    def line(self, ta, pa, tb, pb, outline, n=1600):
        """Visible pieces, as image polylines, of the straight line from (ta, pa) to (tb, pb) in (theta, phi)."""
        at = lambda q: (ta + q * (tb - ta), pa + q * (pb - pa))      # noqa: E731
        q = np.linspace(0, 1, n)
        X, nrm = self.normal(*at(q))
        g = np.sum(nrm * (X - self.cam.E), -1)
        vis = g < 0
        vis[vis] = ~self.occluded(X[vis])
        uv = self.cam(X)
        out = []
        for a, b in _runs(vis):
            ends = []
            for i0, i1 in ((a - 1, a), (b - 1, b)):
                if i0 < 0 or i1 >= n:
                    ends.append(None)
                    continue
                if np.sign(g[i0]) != np.sign(g[i1]):
                    r = _bisect(lambda s: self.g(*[np.array([w]) for w in at(q[i0] + s * (q[i1] - q[i0]))])[0] < 0,
                                bool(g[i0] < 0))
                else:
                    r = _first_crossing(uv[i0], uv[i1], outline)
                    if r is None:
                        def seen(s):
                            th, ph = at(q[i0] + s * (q[i1] - q[i0]))
                            return bool(self.g(np.array([th]), np.array([ph]))[0] < 0
                                        and not self.occluded(self.s.xyz(np.array([th]), np.array([ph])))[0])
                        r = _bisect(seen, bool(vis[i0]))
                ends.append(q[i0] + r * (q[i1] - q[i0]))
            qq = q[a:b]
            if ends[0] is not None:
                qq = np.concatenate([[ends[0]], qq])
            if ends[1] is not None:
                qq = np.concatenate([qq, [ends[1]]])
            if len(qq) >= 2:
                out.append(self.cam(self.s.xyz(*at(qq))))
        return out


def _runs(mask):
    """Index ranges [a, b) of the runs of True."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _bisect(f, f0, steps=40):
    """The parameter in [0, 1] at which f changes from f0 (its value at 0)."""
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid) == f0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _first_crossing(a, b, polylines):
    """The parameter in [0, 1] along the segment a b at which it crosses one of the polylines, nearest its middle,
    or None."""
    best = None
    for P in polylines:
        p, s, r = P[:-1], np.diff(P, axis=0), b - a
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p[:, 0] - a[0]) * s[:, 1] - (p[:, 1] - a[1]) * s[:, 0]) / den
            u = ((p[:, 0] - a[0]) * r[1] - (p[:, 1] - a[1]) * r[0]) / den
        ok = (np.abs(den) > 1e-15) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        for c in t[ok]:
            if best is None or abs(c - 0.5) < abs(best - 0.5):
                best = float(c)
    return best


# ---------------------------------------------------------------------------------------------- the drawing

def camera():
    e, a = math.radians(VIEW["elevation"]), math.radians(VIEW["azimuth"])
    target = np.asarray(VIEW["target"], float)
    back = np.array([-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), math.sin(e)])
    return Camera(target + VIEW["distance"] * back, target, f=VIEW["distance"])


def picture(d):
    """Everything the eye sees, as image polylines: the outline, the rims, the net, the field lines."""
    s = Series(d)
    cam = camera()
    hl = HiddenLines(s, cam, P0, P1)
    pieces, rims = hl.outlines()
    outline = pieces + rims
    th = 2 * np.pi * np.arange(NTH) / NTH
    out = dict(series=s, cam=cam, hl=hl, outline=pieces, rims=rims)
    out["theta"] = [hl.line(t, P0, t, P1, outline) for t in th]
    out["phi"] = [hl.line(0.0, p, 2 * np.pi, p, outline, n=2400) for p in P0 + (P1 - P0) * np.arange(1, NPH) / NPH]
    out["lines"] = [hl.line(th[k], P0, th[k] + s.iota * (P1 - P0), P1, outline) for k in LINES]
    return out


class Page:
    """The house style's figure, with what it draws kept in page units, so that lettering can be checked clear of
    every line (3 units) and of every other label."""

    def __init__(self, style, pic, extra=()):
        self.style = style
        allp = np.vstack(pic["rims"] + pic["outline"] + [pic["cam"](X) for X in extra])
        lo, hi = allp.min(0), allp.max(0)
        L, R, T, B = MARGINS
        self.scale = (style.WIDTH - L - R) / (hi[0] - lo[0])
        self.height = T + B + self.scale * (hi[1] - lo[1])
        self.fig = style.Figure(self.height)
        self.v = self.fig.view((L, T, style.WIDTH - L - R, self.scale * (hi[1] - lo[1])), (lo[0], hi[0]),
                               (lo[1], hi[1]), equal=True)
        self.cam, self.solid = pic["cam"], pic["hl"].solid
        self.drawn, self.boxes = [], []

    def covers(self, box):
        """Whether the solid's image covers any of 15 points of the box: lettering stays off the drawing."""
        (x0, x1, y0, y1), (left, top, width, height) = self.v.lim, self.v.box
        l, t, r, b = box
        for X in np.linspace(l, r, 5):
            for Y in np.linspace(t, b, 3):
                u = x0 + (X - left) / width * (x1 - x0)
                w = y1 - (Y - top) / height * (y1 - y0)
                d = self.cam.d + (u / self.cam.f) * self.cam.r + (w / self.cam.f) * self.cam.u
                if self.solid.hit(self.cam.E, d / np.linalg.norm(d), np.inf):
                    return True
        return False

    def line(self, uv, keep=True, **kw):
        uv = np.asarray(uv, float)
        self.v.line(uv[:, 0], uv[:, 1], **kw)
        if keep:
            self.drawn.append(np.column_stack(self.v.page(uv[:, 0], uv[:, 1])))

    def line3(self, X, **kw):
        self.line(self.cam(X), **kw)

    def at(self, X):
        return tuple(float(c) for c in self.v.page(*self.cam(np.asarray(X, float))))

    def box(self, x, y, s, anchor, size, gap):
        """The ink box the style gives the label s at (x, y) with this anchor and gap (style.Figure._letter)."""
        w, h = self.style.measure(s, size)
        left = x + gap if "w" in anchor else x - gap - w if "e" in anchor else x - w / 2
        top = y + gap if anchor[0] == "n" else y - gap - h if anchor[0] == "s" else y - h / 2
        return (left, top, left + w, top + h)

    def clear(self, box, margin=3.0):
        l, t, r, b = box
        for p in self.drawn:
            a, q = p[:-1], p[1:]
            near = np.nonzero((np.minimum(a[:, 0], q[:, 0]) < r + margin) & (np.maximum(a[:, 0], q[:, 0]) > l - margin)
                              & (np.minimum(a[:, 1], q[:, 1]) < b + margin)
                              & (np.maximum(a[:, 1], q[:, 1]) > t - margin))[0]
            for i in near:
                n = int(np.hypot(*(q[i] - a[i]))) + 2
                sp = a[i] + np.linspace(0, 1, n)[:, None] * (q[i] - a[i])
                if np.any((sp[:, 0] > l - margin) & (sp[:, 0] < r + margin) & (sp[:, 1] > t - margin)
                          & (sp[:, 1] < b + margin)):
                    return False
        return all(bx[2] + margin < l or r + margin < bx[0] or bx[3] + margin < t or b + margin < bx[1]
                   for bx in self.boxes)

    def inside(self, box, margin=4.0):
        return box[0] >= margin and box[1] >= margin and box[2] <= self.style.WIDTH - margin and \
            box[3] <= self.height - margin

    def text(self, X, s, places, size="normal"):
        """The label s beside the point X (Cartesian): the first of the places (anchor, gap) that keeps it 3 units
        clear of every line and label and on the page. None clear stops the drawing."""
        x, y = self.at(X)
        for anchor, gap in places:
            bx = self.box(x, y, s, anchor, size, gap)
            if self.clear(bx) and self.inside(bx) and not self.covers(bx):
                self.fig.text(x, y, s, anchor=anchor, size=size, gap=gap)
                self.boxes.append(bx)
                return anchor, gap
        raise RuntimeError(f"no clear place for the label {s!r} by the point {[round(x), round(y)]}")

    def crossings(self, a, b):
        """How many drawn lines the segment from a to b crosses."""
        n = 0
        for P in self.drawn:
            p, sg, r = P[:-1], np.diff(P, axis=0), b - a
            den = r[0] * sg[:, 1] - r[1] * sg[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((p[:, 0] - a[0]) * sg[:, 1] - (p[:, 1] - a[1]) * sg[:, 0]) / den
                u = ((p[:, 0] - a[0]) * r[1] - (p[:, 1] - a[1]) * r[0]) / den
            n += int(np.sum((np.abs(den) > 1e-12) & (t > 0.02) & (t < 0.98) & (u >= 0) & (u <= 1)))
        return n

    def leader(self, X, s, span, size="normal", radii=(22, 28, 36, 46, 58, 72)):
        """The label s joined to the point X by the style's fine leader: of the places round X at these radii, every
        10 degrees within span (degrees, counter-clockwise from the right), the one whose label is clear and whose
        leader crosses the fewest lines, and of those the nearest."""
        tx, ty = self.at(X)
        names = ["e", "ne", "n", "nw", "w", "sw", "s", "se"]
        best = None
        for rad in radii:
            for deg in np.arange(span[0], span[1] + 1e-9, 10.0):
                x, y = tx + rad * math.cos(math.radians(deg)), ty - rad * math.sin(math.radians(deg))
                to = math.degrees(math.atan2(-(ty - y), tx - x)) % 360
                bx = self.box(x, y, s, names[int(((to + 22.5) % 360) // 45)], size, 5.0)
                if not (self.clear(bx) and self.inside(bx)) or self.covers(bx):
                    continue
                score = 100 * self.crossings(np.array([x, y]), np.array([tx, ty])) + rad
                if best is None or score < best[0]:
                    best = (score, x, y, bx)
        if best is None:
            raise RuntimeError(f"no clear place for the label {s!r} round the point {[round(tx), round(ty)]}")
        _, x, y, bx = best
        crossed = self.crossings(np.array([x, y]), np.array([tx, ty]))
        self.fig.leader(x, y, s, tx, ty, size=size)
        self.boxes.append(bx)
        self.drawn.append(np.array([[x, y], [tx, ty]]))
        return round(x - tx, 1), round(y - ty, 1), crossed


def _seen(hl, X, nudge=1e-3):
    """Visibility of points off the tube (on a cut face, or in the air): nothing between them and the eye."""
    v = X - hl.cam.E
    return ~hl.occluded(X - nudge * v / np.linalg.norm(v, axis=-1, keepdims=True), eps=1e-6)


def draw(previews=None, stem=None):
    sys.path.insert(0, HERE)
    import style
    d = np.load(DATA)
    pic = picture(d)
    s, hl, cam = pic["series"], pic["hl"], pic["cam"]
    # what is drawn besides the tube, in space: built first, so that the page holds it all
    th = 2 * np.pi * np.arange(NTH) / NTH
    ticks = []                      # theta on each cut: inward in the plane of the cut, theta = 0 the longest
    for p in (P0, P1):
        R, Z, Rt, Zt, _, _ = s.rz(th, np.full(NTH, p), deriv=True)
        nn = np.hypot(Rt, Zt)
        for k in range(NTH):
            ln = TICK * (1.7 if k == 0 else 1.0)
            ticks.append(plane(p, [R[k], R[k] - ln * Zt[k] / nn[k]], [Z[k], Z[k] + ln * Rt[k] / nn[k]]))
    a0, a1, off = THETA_ARC
    arcs, zeros = [], []            # theta's sense on each cut: just outside the rim, from theta = 0 upward
    for p in (P0, P1):
        t = np.linspace(a0, a1, 80)
        R, Z, Rt, Zt, _, _ = s.rz(t, np.full_like(t, p), deriv=True)
        nn = np.hypot(Rt, Zt)
        arcs.append(plane(p, R + off * Zt / nn, Z - off * Rt / nn))
        R0, Z0 = s.rz(np.array([0.0]), np.array([p]))
        zeros.append(plane(p, R0[0], Z0[0]))
    Rc, _ = s.rz(2 * np.pi * np.arange(512) / 512, np.zeros(512))
    dR, z0, ln = RZ                 # R and Z in the plane of the cut phi = 0
    o = (float(Rc.max()) + dR, z0)
    axes = [plane(P0, np.linspace(o[0], o[0] + ln, 20), np.full(20, o[1])),
            plane(P0, np.full(20, o[0]), np.linspace(o[1], o[1] + ln, 20))]
    Rd, Zd = DIM                    # phi's sense: an arc of constant R and Z from cut to cut, a line in each cut
    ph = np.linspace(P0, P1, 400)
    dim = plane(ph, np.full_like(ph, Rd), np.full_like(ph, Zd))
    feet = []
    for p in (P0, P1):
        _, Zq = s.rz(2 * np.pi * np.arange(512) / 512, np.full(512, p))
        feet.append(plane(p, [Rd, Rd], [float(Zq.min()) - 0.03, Zd - 0.025]))
    for X in ticks + arcs + axes + [dim] + feet:
        if not _seen(hl, X).all():
            raise RuntimeError("something drawn beside the tube is hidden behind it: move it")
    pg = Page(style, pic, extra=ticks + arcs + axes + [dim] + feet)

    # the solid: the cut faces hatched as a section is, the net, the outline, the field lines
    for rim in pic["rims"]:
        pg.v.fill(rim[:, 0], rim[:, 1], tone="hatch")
    for pcs in pic["theta"] + pic["phi"]:
        for p in pcs:
            pg.line(p, weight="fine")
    for p in pic["outline"]:
        pg.line(p, weight="medium")
    for rim in pic["rims"]:
        pg.line(rim, weight="medium", closed=True)
    for X in ticks:
        pg.line3(X, weight="medium")
    for pcs in pic["lines"]:
        for p in pcs:
            pg.line(p, weight="heavy")
    # the angles, their senses, and the lettering
    for X in arcs + axes:
        pg.line3(X, weight="medium", arrow="end")
    pg.line3(dim, weight="fine", arrow="end")
    for X in feet:
        pg.line3(X, weight="fine")
    for arc, zero, side in zip(arcs, zeros, ("w", "e")):
        pg.text(arc[len(arc) // 2], r"$\theta$", [(side, 4.0), (side, 7.0), ("s" + side, 4.0), ("n" + side, 4.0)])
        pg.text(zero, "$0$", [(side, 5.0), (side, 8.0), ("n" + side, 5.0), ("s" + side, 5.0)], size="small")
    pg.text(axes[0][-1], "$R$", [("w", 3.0), ("sw", 3.0), ("nw", 3.0)])
    pg.text(axes[1][-1], "$Z$", [("s", 3.0), ("se", 3.0), ("sw", 3.0)])
    for X, name in zip(feet, (r"$\varphi=0$", r"$\varphi=\pi/2$")):
        pg.text(X[-1], name, [("n", 4.0), ("n", 7.0)])
    t0 = 2 * np.pi * LETTERED / NTH  # where the lettered line enters and where it leaves
    ends = [s.xyz(np.array([t0]), np.array([P0]))[0], s.xyz(np.array([t0 + s.iota * (P1 - P0)]), np.array([P1]))[0]]
    pg.v.mark(*np.array([cam(e) for e in ends]).T, kind="dot")
    placed = [pg.leader(e, name, span) for e, name, span in
              zip(ends, (r"$\theta_0$", r"$\theta_0+\iota\pi/2$"), LEADER_SPANS)]
    print("leaders (page units from the point, lines crossed):", placed)
    lettered = {"theta": "theta on each cut, with an arrow along the rim from theta = 0 upward",
                "zero": "0, at theta = 0 on each cut, its longest tick",
                "R, Z": "the directions R and Z in the plane of the cut phi = 0",
                "phi": "phi = 0 and phi = pi / 2 at the ends of an arc of constant R and Z from cut to cut, its "
                       "arrow the sense of phi",
                "theta0": "theta0 and theta0 + iota pi / 2: where the line from the corner theta0 = 2 pi k / "
                          f"{NTH}, k = {LETTERED}, enters and leaves (dotted), iota pi / 2 = "
                          f"{s.iota * (P1 - P0):.4f} rad"}
    numbers(d, pic, lettered)
    return pg.fig.save(stem or os.path.join(HERE, "surface-b"), previews=previews)


def numbers(d, pic, lettered):
    """Write surface-b.json: every number in the figure and every number its caption needs, each with what it is."""
    f = json.loads(str(d["facts"]))
    s, cam = pic["series"], pic["cam"]
    iota = float(d["iota"])
    dth, dph = 2 * math.pi / NTH, (P1 - P0) / NPH
    adv = iota * (P1 - P0)
    th = 2 * np.pi * np.arange(256) / 256
    R, Z = s.rz(th, np.zeros(256))
    dist = np.asarray(d["dist"])
    drawn = [int(k) for k in LINES]
    seen = []                       # the fraction of each drawn line in sight, at 400 points
    for k in drawn:
        q = np.linspace(P0, P1, 400)
        X, n = pic["hl"].normal(k * dth + iota * q, q)
        front = np.sum(n * (X - cam.E), -1) < 0
        front[front] = ~pic["hl"].occluded(X[front])
        seen.append(float(front.mean()))
    # the ticks on the cut: arc length between them, and their geometric angle about the cut's centroid
    tf = 2 * np.pi * np.arange(NTH * 2000 + 1) / (NTH * 2000)
    Rf, Zf = s.rz(tf, np.zeros_like(tf))
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(Rf), np.diff(Zf)))])
    arcs = np.diff(cum[::2000])
    cross = Rf[:-1] * Zf[1:] - Rf[1:] * Zf[:-1]
    area = 0.5 * np.sum(cross)
    Rg, Zg = np.sum((Rf[:-1] + Rf[1:]) * cross) / (6 * area), np.sum((Zf[:-1] + Zf[1:]) * cross) / (6 * area)
    geo = np.degrees(np.unwrap(np.arctan2(Zf[::2000] - Zg, Rf[::2000] - Rg)))
    # the mesh the rays are cast against: how far it strays from the series, at cell centres and edge midpoints
    nth, nph = 720, 360
    tg, pg_ = 2 * np.pi * np.arange(nth + 1) / nth, np.linspace(P0, P1, nph + 1)
    tc, pc = 0.5 * (tg[:-1] + tg[1:]), 0.5 * (pg_[:-1] + pg_[1:])
    Xg = s.xyz_grid(tg, pg_)
    stray = max(np.linalg.norm(0.25 * (Xg[:-1, :-1] + Xg[1:, :-1] + Xg[:-1, 1:] + Xg[1:, 1:]) - s.xyz_grid(tc, pc),
                               axis=-1).max(),
                np.linalg.norm(0.5 * (Xg[:-1] + Xg[1:]) - s.xyz_grid(tc, pg_), axis=-1).max(),
                np.linalg.norm(0.5 * (Xg[:, :-1] + Xg[:, 1:]) - s.xyz_grid(tg, pc), axis=-1).max())
    out = {
        "figure": "surface-b: figure 1 of method/README.md, one field period of HSX's surface in perspective, its "
                  "angle net and two field lines",
        "example": {"value": SOURCE_NAME,
                    "what": "HSX, the Helically Symmetric eXperiment: 48 modular coils, 4 field periods (simsopt "
                            "1.11's coil set). The file carries no surface, so torux starts from the magnetic axis"},
        "request": {"flux_Wb": FLUX, "call": f"torux.solve('{SOURCE_NAME}', flux={FLUX!r}, budget_s=900)",
                    "what": f"the request of examples/any_source.py, run with torux {f['torux']}"},
        "answer": {"status": f["status"], "iota": iota, "flux_Wb": float(d["flux"]), "M": int(d["M"]),
                   "N": int(d["N"]), "nfp": int(d["nfp"]), "phi0": float(d["phi0"]),
                   "major_radius_m": f["major_radius_m"], "minor_radius_m": f["minor_radius_m"],
                   "volume_m3": f["volume_m3"], "bn_solver": f["bn_solver"], "bn_independent": f["bn_independent"],
                   "flux_rel_err_independent": f["flux_rel_independent"], "solve_s": f["solve_s"],
                   "what": "the verified surface whose series is drawn, every coefficient of it: R = sum rc[m, n + N] "
                           "cos(m theta - n nfp phi), Z = sum zs[m, n + N] sin(m theta - n nfp phi). bn_solver is the "
                           "largest |B.n|/|B| on the grid shifted half a cell from the solve's, bn_independent the "
                           "same by torux's independent evaluator at 4,096 points, flux_rel_err_independent its "
                           "flux through six sections against the request. Radii by VMEC's and SIMSOPT's definitions",
                   "from": "torux.solve: answer.surface (rc, zs), answer.iota, answer.evidence['solver']['bn_off'], "
                           "answer.evidence['independent'] (bn_sup, flux_rel_err), answer.major_radius, "
                           "answer.minor_radius, answer.volume"},
        "period": {"phi_from": P0, "phi_to": P1,
                   "cut_R_m": [float(R.min()), float(R.max())], "cut_Z_m": [float(Z.min()), float(Z.max())],
                   "what": "the tube is cut by the planes phi = 0 and phi = pi / 2, one field period of four. The "
                           "two cuts are the same curve in their (R, Z) planes (field-period symmetry), R and Z its "
                           "extent there, from the series"},
        "net": {"n_theta": NTH, "n_phi": NPH, "dtheta_rad": dth, "dphi_rad": dph,
                "theta_cells_per_phi_cell": iota * dph / dth,
                "drift_over_period_cells": NPH * (iota * dph / dth) - NPH,
                "what": "the fine curves theta = 2 pi k / 11 and phi = (pi / 2) j / 3 on the tube (j = 0 and 3 are "
                        "the cuts). The steps are chosen so that a field line crosses iota dphi / dtheta = 0.995 "
                        "cells of theta for every cell of phi: from a corner of the net it passes the next corners "
                        "within this fraction of a cell, too little to see"},
        "field_lines": {"k": drawn, "theta0_rad": [float(k * dth) for k in drawn],
                        "theta0_deg": [float(math.degrees(k * dth)) for k in drawn],
                        "theta_exit_rad": [float(k * dth + adv) for k in drawn],
                        "theta_exit_deg": [float(math.degrees(k * dth + adv)) for k in drawn],
                        "advance_rad": adv, "advance_deg": math.degrees(adv), "advance_theta_steps": adv / dth,
                        "fraction_in_sight": seen, "lettered_k": LETTERED,
                        "what": "the heavy curves theta = theta0 + iota phi on the tube, from the corners theta0 = 2 "
                                f"pi k / {NTH} of the net on the cut phi = 0. The advance iota pi / 2 is lettered at "
                                f"the end of the line from k = {LETTERED}, whose ends are dotted"},
        "check": {"lines_traced": int(dist.shape[0]), "points_per_line": int(dist.shape[1]),
                  "largest_distance_m": float(dist.max()),
                  "largest_distance_drawn_m": float(dist[drawn].max()),
                  "largest_distance_over_minor_radius": float(dist.max() / f["minor_radius_m"]),
                  "tracer_error_estimate_m": f["trace_err_estimate_m"], "tracer_step_rad": f["trace_h"],
                  "what": "a field line traced from every corner of the net on the cut phi = 0 to phi = pi / 2 in "
                          "the coils' field, compared with the series' line theta0 + iota phi at the same phi, 64 "
                          "points a line: the largest distance between the two, all eleven lines and the two drawn. "
                          "The tracer is torux's independent one (Gauss-Legendre collocation of order 12 on its own "
                          "field evaluator, which shares no code with the solver), and its step-doubling estimate "
                          "of its own error",
                  "from": "torux.reference.trace.poincare(torux.reference.fast64.Field(torux.load(source)), R0, Z0, "
                          "phis, 1, h=2 pi / 1024), answer.points"},
        "ticks": {"arc_between_m": [float(a) for a in arcs], "largest_over_smallest": float(arcs.max() / arcs.min()),
                  "geometric_angle_deg": [float(g - geo[0]) for g in geo[:-1]],
                  "what": "the ticks of theta on a cut, at the net's steps 2 pi / 11: the arc length between "
                          "successive ticks, from theta = 0, and the geometric angle of each about the cut's "
                          "centroid. Even in theta, uneven in both: theta is neither"},
        "hidden_lines": {"mesh": [nth, nph], "triangles": int(len(pic["hl"].solid.tris)), "mesh_stray_m": float(stray),
                         "what": "the solid against which rays from the eye are cast: the series on a grid of 720 "
                                 "by 360 in (theta, phi), two triangles a cell, and a fan across each cut. "
                                 "mesh_stray_m is the largest distance from the series of a cell's centre or an "
                                 "edge's midpoint"},
        "view": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in VIEW.items()},
                 "eye_m": [float(c) for c in cam.E],
                 "what": "the eye, in perspective: azimuth and elevation of the line from the target to it reversed, "
                         "degrees, and its distance (m). From inside the torus, the only side from which both cuts "
                         "are seen: phi increases from right to left"},
        "lettered": lettered,
    }
    with open(NUMBERS, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compute", action="store_true", help="solve and trace again, and keep what was found")
    ap.add_argument("--no-draw", action="store_true", help="with --compute: only compute")
    ap.add_argument("--previews", metavar="DIR", help="also render PNG previews at 838 and 324 px into DIR")
    a = ap.parse_args(argv)
    if a.compute or not os.path.exists(DATA):
        compute()
    if not a.no_draw:
        draw(previews=a.previews)


if __name__ == "__main__":
    sys.exit(main())
