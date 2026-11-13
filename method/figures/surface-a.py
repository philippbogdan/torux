"""Figure 1 of the method page: the surface is a function of two angles, and on it a field line is straight.

    python method/figures/surface-a.py              # draw from data/surface-a.npz
    python method/figures/surface-a.py --compute    # solve and trace again, then draw

The request is the one examples/any_source.py makes for HSX:

    torux.solve("examples/hsx.json", flux=0.0359221439332229, budget_s=900)

Left, the surface torux verifies, drawn from its own series R(theta, phi), Z(theta, phi) in perspective with
hidden lines removed. Its net is the curves theta = const and phi = const every 2 pi / 12, which is the grid of
the square on the right, and the sector from phi = 11 pi / 6 to 2 pi is cut away so that the section at phi = 0
shows, hatched. The heavy line is the image under the series of the straight line theta = theta0 + iota phi of the
square, from the dot at phi = 0 to the far cut, and the square draws it as far: one column short of 2 pi.

With --compute the line is also traced in the coils' field from the same start, by torux.reference.trace, which
shares no code with the solver, and through the solver's own field (Answer.field) by scipy, and the largest
distance between each traced line and the drawn one goes into surface-a.json with the other numbers.

The drawing evaluates the series without its coefficients below 1e-12 m (their sum, a bound on how far any
point moves, is in surface-a.json), and hides lines with an item buffer: every curve is sampled, and kept where
the triangle nearest the eye at its pixel is its own patch of the surface, or where nothing is nearer.
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
NAME = "surface-a"
DATA = os.path.join(HERE, "data", f"{NAME}.npz")
NUMBERS = os.path.join(HERE, f"{NAME}.json")
SOURCE = os.path.join(REPO, "examples", "hsx.json")
FLUX = 0.0359221439332229          # Wb, the request of examples/any_source.py
BUDGET_S = 900
CUT = 1e-12                        # m: coefficients kept for the drawing

TWO_PI = 2 * math.pi
THETA0 = 3 * TWO_PI / 12           # where the line starts, at phi = 0: a node of the net, the top
PHI_END = 11 * TWO_PI / 12         # the far cut: the solid is phi in [0, PHI_END], one column short
K_THETA, K_PHI = 12, 12            # the net, and the square's grid


# ---------------------------------------------------------------------------------------------- the run

def compute():
    """Solve the request, keep the series, and trace the drawn line in the coils' field two ways."""
    import torux
    from torux.reference import trace
    from scipy.integrate import solve_ivp

    t0 = time.time()
    s = torux.solve(SOURCE, flux=FLUX, budget_s=BUDGET_S)
    wall = time.time() - t0
    if s.status != "surface":
        raise SystemExit(f"torux.solve returned {s.status!r}, not a verified surface:\n{s}")
    t = s.surface
    if float(getattr(t, "phi0", 0.0)) != 0.0:
        raise SystemExit("the surface's symmetry plane is not phi = 0")
    rc, zs = np.asarray(t.rc, float), np.asarray(t.zs, float)
    M, N = rc.shape[0] - 1, (rc.shape[1] - 1) // 2
    m = np.repeat(np.arange(M + 1)[:, None], 2 * N + 1, 1)
    n = np.repeat(np.arange(-N, N + 1)[None, :], M + 1, 0)
    keep = (np.abs(rc) >= CUT) | (np.abs(zs) >= CUT)
    dropped = float(np.abs(rc[~keep]).sum() + np.abs(zs[~keep]).sum())
    iota = float(s.iota)

    # the drawn line, by the series (torux's own evaluation of the whole series), over one turn: the solid
    # carries it to PHI_END, the square to 2 pi
    nsec = 2880
    phis = TWO_PI * np.arange(1, nsec + 1) / nsec
    x = s.points(THETA0 + iota * phis, phis)
    Rs, Zs = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
    x0 = s.points(np.array([THETA0]), np.array([0.0]))[0]
    R0, Z0 = float(np.hypot(x0[0], x0[1])), float(x0[2])

    # traced by torux.reference.trace (Gauss-Legendre, order 12, in the independent evaluator's field)
    src = torux.load(SOURCE)
    tt = time.time()
    tr = trace.poincare(src, [R0], [Z0], phis, 1, phi0=0.0, h=TWO_PI / 512)
    t_ref = time.time() - tt
    Rr, Zr = tr["R"][0, 0], tr["Z"][0, 0]
    d_ref = np.hypot(Rr - Rs, Zr - Zs)

    # traced in the solver's own field, Answer.field, by scipy's DOP853
    def rhs(ph, y):
        R, Z = y
        c, sn = math.cos(ph), math.sin(ph)
        B = s.field(np.array([[R * c, R * sn, Z]]))[0]
        BR, Bp, BZ = B[0] * c + B[1] * sn, -B[0] * sn + B[1] * c, B[2]
        return [R * BR / Bp, R * BZ / Bp]

    tt = time.time()
    sol = solve_ivp(rhs, (0.0, TWO_PI), [R0, Z0], method="DOP853", rtol=1e-13, atol=1e-15, t_eval=phis)
    t_own = time.time() - tt
    d_own = np.hypot(sol.y[0] - Rs, sol.y[1] - Zs)
    on_solid = phis <= PHI_END + 1e-12
    # the drawing's own evaluation, from the coefficients it keeps, against torux's of the whole series
    ser = Series(m[keep], n[keep], rc[keep], zs[keep], int(s.nfp), float(getattr(t, "phi0", 0.0)))
    Rd, Zd = ser.at(THETA0 + iota * phis, phis)
    d_draw = np.hypot(Rd - Rs, Zd - Zs)

    ev = s.evidence
    ind = ev.get("independent", {})
    sv = ev.get("solver", {})
    numbers = {
        "example": "examples/hsx.json: HSX, the Helically Symmetric eXperiment, its 48 modular coils",
        "call": f"torux.solve('examples/hsx.json', flux={FLUX!r}, budget_s={BUDGET_S}), as examples/any_source.py",
        "status": s.status,
        "flux_Wb": {"value": float(s.flux), "what": "toroidal flux enclosed by the surface (Answer.flux)"},
        "iota": {"value": iota, "what": "rotational transform of the surface, VMEC's sign (Answer.iota): the "
                                        "slope of the line in the square"},
        "M": {"value": int(t.M), "what": "poloidal modes of the series (Answer.surface.M)"},
        "N": {"value": int(t.N), "what": "toroidal modes of the series, per field period (Answer.surface.N)"},
        "nfp": {"value": int(s.nfp), "what": "field periods (Answer.nfp)"},
        "major_radius_m": {"value": float(s.major_radius), "what": "Answer.major_radius"},
        "minor_radius_m": {"value": float(s.minor_radius), "what": "Answer.minor_radius"},
        "volume_m3": {"value": float(s.volume), "what": "Answer.volume"},
        "bn_solver": {"value": float(sv.get("bn_off", float("nan"))),
                      "what": "largest |B.n|/|B| on the grid shifted half a cell, the solver's own check "
                              "(Answer.evidence['solver']['bn_off'])"},
        "bn_independent": {"value": float(ind.get("bn_sup", float("nan"))),
                           "what": "largest |B.n|/|B| the independent evaluator finds, climbing from the worst of "
                                   "4,096 random points to the local maximum (Answer.evidence['independent']"
                                   "['bn_sup'])"},
        "solve_s": {"value": round(wall, 1), "what": "wall time of torux.solve here (seconds)"},
        "series_kept": {"value": int(keep.sum()), "of": int(rc.size),
                        "what": f"coefficients of R and Z with |rc| or |zs| >= {CUT} m, which the drawing uses"},
        "series_dropped_sum_m": {"value": dropped,
                                 "what": "sum of |rc| + |zs| over the coefficients the drawing leaves out: a "
                                         "bound on how far any drawn point moves"},
        "line_vs_reference_trace_m": {
            "value": float(d_ref[on_solid].max()),
            "what": "largest distance, at equal phi over the length drawn on the solid (phi from 0 to phi_end), "
                    "between the drawn line (the series at theta0 + iota phi, by Answer.points) and the field line "
                    "from the same start traced by torux.reference.trace.poincare (order-12 Gauss-Legendre in the "
                    "independent evaluator's field, which shares no code with the solver)",
            "over_whole_turn": float(d_ref.max()),
            "tracer_step_doubling_m": float(np.max(tr["err"][0, 0])),
            "tracer_step_doubling_what": "the tracer's own estimate: the largest difference between its runs at "
                                         "steps h and h / 2", "points_per_turn": int(nsec),
            "trace_s": round(t_ref, 1)},
        "line_vs_own_field_m": {
            "value": float(d_own[on_solid].max()),
            "what": "the same distance to the line integrated in the solver's own field, Answer.field, by scipy's "
                    "DOP853 (rtol 1e-13, atol 1e-15 m): the integrator's tolerance, not the surface, sets it",
            "over_whole_turn": float(d_own.max()), "trace_s": round(t_own, 1)},
        "drawing_series_vs_torux_m": {
            "value": float(d_draw.max()),
            "what": "largest distance along the line between the drawing's evaluation of the kept coefficients "
                    "and torux's own of the whole series (Answer.points)"},
        "line_start": {"theta0": THETA0, "phi": 0.0, "R_m": R0, "Z_m": Z0},
        "line_phi_end": PHI_END,
    }
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, m=m[keep].astype(np.int16), n=n[keep].astype(np.int16), rc=rc[keep], zs=zs[keep],
                        nfp=int(s.nfp), phi0=float(getattr(t, "phi0", 0.0)), iota=iota,
                        numbers=json.dumps(numbers))
    print(s)
    print(f"line against torux.reference.trace: {d_ref[on_solid].max():.2e} m, against Answer.field: "
          f"{d_own[on_solid].max():.2e} m (to phi_end), {d_ref.max():.2e} and {d_own.max():.2e} m over the turn")


# ---------------------------------------------------------------------------------------------- the surface

class Series:
    """R = sum rc cos(m th - n nfp (ph - phi0)), Z = sum zs sin(m th - n nfp (ph - phi0)), from the kept modes."""

    def __init__(self, m, n, rc, zs, nfp, phi0):
        self.m, self.n = np.asarray(m, float), np.asarray(n, float)
        self.rc, self.zs = np.asarray(rc, float), np.asarray(zs, float)
        self.nfp, self.phi0 = int(nfp), float(phi0)
        M, N = int(self.m.max()), int(np.abs(self.n).max())
        self.RC = np.zeros((M + 1, 2 * N + 1))
        self.ZS = np.zeros((M + 1, 2 * N + 1))
        self.RC[self.m.astype(int), self.n.astype(int) + N] = self.rc
        self.ZS[self.m.astype(int), self.n.astype(int) + N] = self.zs
        self.M, self.N = M, N

    def at(self, th, ph, deriv=False):
        """R, Z (and R_th, Z_th, R_ph, Z_ph) at points th, ph of one shape."""
        th, ph = np.broadcast_arrays(np.asarray(th, float), np.asarray(ph, float))
        shp = th.shape
        th, ph = th.ravel(), ph.ravel()
        out = [np.empty_like(th) for _ in range(6 if deriv else 2)]
        k = self.nfp * self.n
        step = max(1, 2_000_000 // len(self.m))
        for i in range(0, len(th), step):
            a = np.outer(th[i:i + step], self.m) - np.outer(ph[i:i + step] - self.phi0, k)
            c, s = np.cos(a), np.sin(a)
            out[0][i:i + step] = c @ self.rc
            out[1][i:i + step] = s @ self.zs
            if deriv:
                out[2][i:i + step] = -(s @ (self.m * self.rc))
                out[3][i:i + step] = c @ (self.m * self.zs)
                out[4][i:i + step] = s @ (k * self.rc)
                out[5][i:i + step] = -(c @ (k * self.zs))
        return [o.reshape(shp) for o in out]

    def grid(self, th, ph, deriv=False):
        """The same on the tensor grid th (nt,) x ph (np,), by matrix products."""
        m = np.arange(self.M + 1)
        k = self.nfp * np.arange(-self.N, self.N + 1)
        at, ap = np.outer(th, m), np.outer(np.asarray(ph) - self.phi0, k)
        Ct, St, Cp, Sp = np.cos(at), np.sin(at), np.cos(ap), np.sin(ap)
        rc, zs = self.RC, self.ZS
        R = Ct @ rc @ Cp.T + St @ rc @ Sp.T
        Z = St @ zs @ Cp.T - Ct @ zs @ Sp.T
        if not deriv:
            return R, Z
        mr, mz, kr, kz = m[:, None] * rc, m[:, None] * zs, rc * k[None, :], zs * k[None, :]
        return (R, Z, -(St @ mr @ Cp.T - Ct @ mr @ Sp.T), Ct @ mz @ Cp.T + St @ mz @ Sp.T,
                St @ kr @ Cp.T - Ct @ kr @ Sp.T, -(Ct @ kz @ Cp.T + St @ kz @ Sp.T))

    @staticmethod
    def _xyz(r, ph):
        c, s = np.cos(ph), np.sin(ph)
        R, Z = r[0], r[1]
        X = np.stack([R * c, R * s, Z], -1)
        if len(r) == 2:
            return X
        Rt, Zt, Rp, Zp = r[2:]
        return X, np.stack([Rt * c, Rt * s, Zt], -1), np.stack([Rp * c - R * s, Rp * s + R * c, Zp], -1)

    def xyz(self, th, ph, deriv=False):
        ph = np.broadcast_arrays(np.asarray(th, float), np.asarray(ph, float))[1]
        return self._xyz(self.at(th, ph, deriv), ph)

    def xyz_grid(self, th, ph, deriv=False):
        r = self.grid(th, ph, deriv)
        return self._xyz(r, np.broadcast_to(np.asarray(ph, float)[None, :], r[0].shape))


class Camera:
    """A pinhole camera: eye, the point it looks at, and the vertical of the world kept vertical."""

    def __init__(self, eye, target, up=(0.0, 0.0, 1.0)):
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        self.f, self.r, self.u = f, r, np.cross(r, f)

    def project(self, X):
        """Screen coordinates (x right, y up) and depth of points X (..., 3)."""
        d = X - self.eye
        z = d @ self.f
        return np.stack([(d @ self.r) / z, (d @ self.u) / z], -1), z


def rasterize(sxy, z, tris, labels, W, H, x0, y0, scale):
    """Item buffer: for each pixel the label of the triangle nearest the eye (-1 for none) and its 1 / depth."""
    px = (sxy[:, 0] - x0) * scale - 0.5
    py = (y0 - sxy[:, 1]) * scale - 0.5
    iz = 1.0 / z                       # linear on the screen: larger is nearer
    depth = np.full(H * W, -np.inf)
    lab = np.full(H * W, -1, dtype=np.int64)
    A, B, C = tris[:, 0], tris[:, 1], tris[:, 2]
    xa, ya, xb, yb, xc, yc = px[A], py[A], px[B], py[B], px[C], py[C]
    xmin = np.floor(np.minimum(np.minimum(xa, xb), xc)).astype(int)
    xmax = np.ceil(np.maximum(np.maximum(xa, xb), xc)).astype(int)
    ymin = np.floor(np.minimum(np.minimum(ya, yb), yc)).astype(int)
    ymax = np.ceil(np.maximum(np.maximum(ya, yb), yc)).astype(int)
    size = np.maximum(xmax - xmin + 1, ymax - ymin + 1)
    area = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
    ok = np.abs(area) > 1e-12
    for k in np.unique(size[ok]):
        sel = np.nonzero(ok & (size == k))[0]
        oy, ox = np.mgrid[0:k, 0:k]
        ox, oy = ox.ravel(), oy.ravel()
        chunk = max(1, 2_000_000 // (k * k))
        for c0 in range(0, len(sel), chunk):
            t = sel[c0:c0 + chunk]
            X = xmin[t, None] + ox[None, :]
            Y = ymin[t, None] + oy[None, :]
            ar = area[t, None]
            w0 = ((xb[t, None] - X) * (yc[t, None] - Y) - (xc[t, None] - X) * (yb[t, None] - Y)) / ar
            w1 = ((xc[t, None] - X) * (ya[t, None] - Y) - (xa[t, None] - X) * (yc[t, None] - Y)) / ar
            w2 = 1.0 - w0 - w1
            e = -1e-6
            inside = (w0 >= e) & (w1 >= e) & (w2 >= e) & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
            if not inside.any():
                continue
            d = w0 * iz[A[t], None] + w1 * iz[B[t], None] + w2 * iz[C[t], None]
            pix, dv = (Y * W + X)[inside], d[inside]
            lv = np.broadcast_to(labels[t, None], X.shape)[inside]
            order = np.lexsort((-dv, pix))
            pix, dv, lv = pix[order], dv[order], lv[order]
            first = np.ones(len(pix), bool)
            first[1:] = pix[1:] != pix[:-1]
            pix, dv, lv = pix[first], dv[first], lv[first]
            better = dv > depth[pix]
            depth[pix[better]] = dv[better]
            lab[pix[better]] = lv[better]
    return lab.reshape(H, W), depth.reshape(H, W)


class Solid:
    """The surface over phi in [pa, pb], closed by its sections there, seen by a camera."""

    def __init__(self, ser, cam, pa, pb, nth=288, nph=1792, buf_w=3200):
        self.ser, self.cam, self.pa, self.pb = ser, cam, pa, pb
        th = TWO_PI * np.arange(nth + 1) / nth
        ph = pa + (pb - pa) * np.arange(nph + 1) / nph
        self.nth, self.nph = nth, nph
        X = ser.xyz_grid(th, ph)
        sxy, z = cam.project(X.reshape(-1, 3))
        I, J = np.meshgrid(np.arange(nth), np.arange(nph), indexing="ij")
        v = lambda i, j: i * (nph + 1) + j
        a, b, c, d = v(I, J), v(I + 1, J), v(I + 1, J + 1), v(I, J + 1)
        tris = [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
        labs = [(I * nph + J).ravel()] * 2
        verts, depths = [sxy], [z]
        base = len(sxy)
        self.face_label, self.section = {}, {}
        nrho = 64
        for key, phc in (("a", pa), ("b", pb)):
            R, Z = ser.at(th[:-1], np.full(nth, phc))
            cR, cZ = R.mean(), Z.mean()            # the sections are star-shaped about their centroid
            self.section[key] = (phc, R, Z)
            rho = np.arange(nrho + 1)[:, None] / nrho
            Rf, Zf = cR + rho * (R[None, :] - cR), cZ + rho * (Z[None, :] - cZ)
            P = np.stack([Rf * math.cos(phc), Rf * math.sin(phc), Zf], -1).reshape(-1, 3)
            s2, z2 = cam.project(P)
            ii, jj = np.meshgrid(np.arange(nrho), np.arange(nth), indexing="ij")
            w = lambda i, j: base + i * nth + (j % nth)
            t2 = np.concatenate([np.stack([w(ii, jj), w(ii + 1, jj), w(ii + 1, jj + 1)], -1).reshape(-1, 3),
                                 np.stack([w(ii, jj), w(ii + 1, jj + 1), w(ii, jj + 1)], -1).reshape(-1, 3)])
            self.face_label[key] = nth * nph + len(self.face_label)
            tris.append(t2)
            labs.append(np.full(len(t2), self.face_label[key]))
            verts.append(s2)
            depths.append(z2)
            base += len(P)
        allv = np.concatenate(verts)
        lo, hi = allv.min(0), allv.max(0)
        pad = 0.01 * (hi - lo).max()
        lo, hi = lo - pad, hi + pad
        self.box = (lo, hi)
        scale = buf_w / (hi[0] - lo[0])
        self.x0, self.y0, self.scale = lo[0], hi[1], scale
        self.W, self.H = buf_w, int(math.ceil((hi[1] - lo[1]) * scale))
        self.lab, self.near = rasterize(allv, np.concatenate(depths), np.concatenate(tris), np.concatenate(labs),
                                        self.W, self.H, lo[0], hi[1], scale)
        self.dth, self.dph = TWO_PI / nth, (pb - pa) / nph

    def _pixel(self, sxy):
        j = np.floor((sxy[:, 0] - self.x0) * self.scale).astype(int)
        i = np.floor((self.y0 - sxy[:, 1]) * self.scale).astype(int)
        return np.clip(i, 0, self.H - 1), np.clip(j, 0, self.W - 1)

    def seen(self, th, ph, rim=False):
        """Which surface points (th, ph) the eye sees, and their screen points. rim: points on the outline,
        where the surface turns away and the facing test says nothing."""
        X, Xt, Xp = self.ser.xyz(th, ph, deriv=True)
        sxy, z = self.cam.project(X)
        i, j = self._pixel(sxy)
        L = self.lab[i, j]
        nsurf = self.nth * self.nph
        ok = np.zeros(len(th), bool)
        s = (L >= 0) & (L < nsurf)
        cth = (L[s] // self.nph + 0.5) * self.dth
        cph = self.pa + (L[s] % self.nph + 0.5) * self.dph
        dth = np.angle(np.exp(1j * (th[s] - cth)))
        ok[s] = (dth / 0.16) ** 2 + ((ph[s] - cph) / 0.07) ** 2 < 1.0
        for key, phc in (("a", self.pa), ("b", self.pb)):
            ok |= (np.abs(ph - phc) < 1e-9) & (L == self.face_label[key])
        # nothing nearer at its pixel: on the outline the pixel may show what lies beyond it, or nothing
        ok |= self.near[i, j] <= (1.0 / z) * (1 + 2e-5)
        if not rim:
            facing = (np.cross(Xp, Xt) * (self.cam.eye - X)).sum(-1) > 0
            for key, phc in (("a", self.pa), ("b", self.pb)):
                if self.face_seen(key):
                    facing |= np.abs(ph - phc) < 1e-9
            ok &= facing
        return ok, sxy

    def face_seen(self, key):
        """Whether the section at the cut `key` faces the eye."""
        phc = self.section[key][0]
        n = np.array([-math.sin(phc), math.cos(phc), 0.0]) * (-1 if key == "a" else 1)   # out of the solid
        R, Z = self.section[key][1].mean(), self.section[key][2].mean()
        P = np.array([R * math.cos(phc), R * math.sin(phc), Z])
        return float(n @ (self.cam.eye - P)) > 0

    def rim(self, nth=900, per_rad=520):
        """The outline: the curves on the surface where it turns from the eye, as (th, ph) arrays."""
        import contourpy
        nph = int((self.pb - self.pa) * per_rad)
        th = TWO_PI * np.arange(nth + 1) / nth
        ph = self.pa + (self.pb - self.pa) * np.arange(nph + 1) / nph
        X, Xt, Xp = self.ser.xyz_grid(th, ph, deriv=True)
        g = (np.cross(Xp, Xt) * (self.cam.eye - X)).sum(-1)
        lines = contourpy.contour_generator(ph, th, g).lines(0.0)
        return [(L[:, 1], L[:, 0]) for L in lines]


def runs(ok):
    m = np.concatenate([[0], ok.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def visible(solid, th, ph, rim=False, min_pts=3):
    ok, sxy = solid.seen(th, ph, rim=rim)
    return [sxy[a:b] for a, b in runs(ok) if b - a >= min_pts]


# ---------------------------------------------------------------------------------------------- the drawing

DESIGN = dict(
    azimuth=-100.0, elevation=32.0, distance=8.0, target=(0.15, 0.0, 0.0),   # the eye: degrees, metres
    theta0=THETA0, phi_end=PHI_END, net=(K_THETA, K_PHI), halo=True,
    height=380, solid_box=(10, 12, 545, 356), square=(590, 96, 200),       # page units
    theta_arrow=(-40.0, 40.0, 0.05),            # on the section at phi = 0: from, to (degrees), offset (m)
    phi_arrow=(90.0, 14.0, 40.0, 0.06),         # along theta = const: theta, phi from, to (degrees), offset (m)
    slope=(1.2, 1.0),                           # the slope triangle in the square: phi where it starts, its run
    axis_arrow=20.0,                            # how far the arrows of the square's axes run past it
)


def square_segments(theta0, iota, phi0, phi1):
    """Pieces of theta = theta0 + iota phi for phi in [phi0, phi1], folded into [0, 2 pi) x [0, 2 pi)."""
    br = {phi0, phi1}
    k = math.floor((theta0 + iota * phi0) / TWO_PI) + 1
    while (TWO_PI * k - theta0) / iota < phi1:
        br.add((TWO_PI * k - theta0) / iota)
        k += 1
    k = math.floor(phi0 / TWO_PI) + 1
    while TWO_PI * k < phi1:
        br.add(TWO_PI * k)
        k += 1
    br = sorted(br)
    out = []
    for a, b in zip(br[:-1], br[1:]):
        if b - a < 1e-12:
            continue
        mid = 0.5 * (a + b)
        sp = math.floor(mid / TWO_PI) * TWO_PI
        st = math.floor((theta0 + iota * mid) / TWO_PI) * TWO_PI
        out.append(((a - sp, b - sp), (theta0 + iota * a - st, theta0 + iota * b - st)))
    return out


def geometry(D, ser, iota):
    """What the solid's half of the plate is made of, in the camera's screen coordinates: the visible pieces of
    every curve, the section at phi = 0, the start of the line and the two arrows of the angles."""
    theta0, phi_end = D["theta0"], D["phi_end"]
    kth, kph = D["net"]
    az, el, dist = math.radians(D["azimuth"]), math.radians(D["elevation"]), D["distance"]
    target = np.asarray(D["target"], float)
    eye = target + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    cam = Camera(eye, target)
    solid = Solid(ser, cam, 0.0, phi_end)
    G = {"box": solid.box, "eye": eye}
    phc, R, Z = solid.section["a"]
    G["section"] = cam.project(np.stack([R * math.cos(phc), R * math.sin(phc), Z], -1))[0] \
        if solid.face_seen("a") else None
    ph = np.linspace(0.0, phi_end, 9000)
    G["net"] = [p for k in range(kth) for p in visible(solid, np.full_like(ph, TWO_PI * k / kth), ph)]
    th = np.linspace(0.0, TWO_PI, 2400)
    for j in range(1, kph):
        phj = TWO_PI * j / kph
        if phj < phi_end - 1e-9:
            G["net"] += visible(solid, th, np.full_like(th, phj))
    ph = np.linspace(0.0, phi_end, 30000)
    G["line"] = visible(solid, theta0 + iota * ph, ph)
    G["outline"] = [p for phc in (0.0, phi_end) for p in visible(solid, th, np.full_like(th, phc))]
    G["outline"] += [p for pth, pph in solid.rim() for p in visible(solid, pth, pph, rim=True)]
    G["start"] = cam.project(ser.xyz(np.array([theta0]), np.array([0.0])))[0][0]
    # theta: round the section at phi = 0, counter-clockwise in (R, Z), just outside it
    ta, tb, off = D["theta_arrow"]
    t = np.radians(np.linspace(ta, tb, 80))
    R, Z, Rt, Zt = ser.at(t, np.zeros_like(t), deriv=True)[:4]
    nrm = np.hypot(Rt, Zt)
    G["theta_arrow"] = cam.project(np.stack([R + off * Zt / nrm, np.zeros_like(R), Z - off * Rt / nrm], -1))[0]
    # phi: along theta = const, lifted off the surface along its normal
    tc, fa, fb, off = D["phi_arrow"]
    f = np.radians(np.linspace(fa, fb, 80))
    X, Xt, Xp = ser.xyz(np.full_like(f, math.radians(tc)), f, deriv=True)
    nv = np.cross(Xp, Xt)
    G["phi_arrow"] = cam.project(X + off * nv / np.linalg.norm(nv, axis=-1, keepdims=True))[0]
    return G


def draw(previews=None, design=None, stem=None):
    """Draw the plate from data/surface-a.npz and write the two SVG files and surface-a.json. design: changes
    to DESIGN, to try a variant (with stem, a path to write it to instead, and no JSON)."""
    sys.path.insert(0, HERE)
    import style

    D = dict(DESIGN, **(design or {}))
    d = np.load(DATA)
    numbers = json.loads(str(d["numbers"]))
    ser = Series(d["m"], d["n"], d["rc"], d["zs"], int(d["nfp"]), float(d["phi0"]))
    iota = float(d["iota"])
    theta0, phi_end = D["theta0"], D["phi_end"]
    kth, kph = D["net"]
    G = geometry(D, ser, iota)

    fig = style.Figure(height=D["height"])
    lo, hi = G["box"]
    left = fig.view(D["solid_box"], (lo[0], hi[0]), (lo[1], hi[1]), equal=True)

    def poly(view, p, **kw):
        view.line(p[:, 0], p[:, 1], **kw)

    # ---- the solid: the section at phi = 0 hatched, as a cut through a solid, then the net, the line over it,
    # and the outline over the line where the line turns out of sight
    if G["section"] is not None:
        left.fill(G["section"][:, 0], G["section"][:, 1], tone="hatch")
    for p in G["net"]:
        poly(left, p, weight="fine")
    for p in G["line"]:
        poly(left, p, weight="heavy", halo=D["halo"])
    for p in G["outline"]:
        poly(left, p, weight="medium")
    left.mark(G["start"][0], G["start"][1], kind="dot")
    pa_ = G["theta_arrow"]
    left.line(pa_[:, 0], pa_[:, 1], weight="fine", arrow="end")
    mid = pa_[len(pa_) // 2]
    left.text(mid[0], mid[1], r"$\theta$", anchor="w", gap=6)
    pf = G["phi_arrow"]
    left.line(pf[:, 0], pf[:, 1], weight="fine", arrow="end")
    mid = pf[len(pf) // 2]
    left.text(mid[0], mid[1], r"$\varphi$", anchor="s", gap=6)

    # ---- the square of the two angles
    L, T, S = D["square"]
    sq = fig.view((L, T, S, S), (0, TWO_PI), (0, TWO_PI))
    # the grid is the net, and the line goes as far as the solid carries it: to the far cut, one column short of
    # 2 pi
    for k in range(1, kth):
        sq.line([0, TWO_PI], [TWO_PI * k / kth] * 2, weight="fine")
    for j in range(1, kph):
        sq.line([TWO_PI * j / kph] * 2, [0, TWO_PI], weight="fine")
    sq.line([0, TWO_PI, TWO_PI, 0], [0, 0, TWO_PI, TWO_PI], weight="medium", closed=True)
    for (pa, pb), (ta, tb) in square_segments(theta0, iota, 0.0, phi_end):
        sq.line([pa, pb], [ta, tb], weight="heavy")
    sq.mark([0.0], [theta0], kind="dot")
    # the slope: a run of 1 and a rise of iota
    f0, run = D["slope"]
    y0 = theta0 + iota * f0
    sq.line([f0, f0 + run, f0 + run], [y0, y0, y0 + iota * run], weight="medium")
    sq.text(f0 + 0.5 * run, y0, "$1$", anchor="n", size="small", gap=3)
    sq.text(f0 + run, y0 + 0.5 * iota * run, r"$\iota$", anchor="w", gap=4)
    # numbers and names
    fig.text(L, T + S, "$0$", anchor="ne", size="small", gap=4)
    fig.text(L + S, T + S, r"$2\pi$", anchor="n", size="small", gap=6)
    fig.text(L, T, r"$2\pi$", anchor="e", size="small", gap=7)
    y_t0 = float(sq.page(0.0, theta0)[1])
    fig.text(L, y_t0, r"$\theta_0$", anchor="e", gap=8)
    ext = D["axis_arrow"]
    fig.arrow(L + S, T + S, L + S + ext, T + S, weight="medium")
    fig.text(L + S + ext, T + S, r"$\varphi$", anchor="w", gap=5)
    fig.arrow(L, T, L, T - ext, weight="medium")
    fig.text(L, T - ext, r"$\theta$", anchor="s", gap=5)
    fig.text(L + 0.55 * S, T - 12, r"$\theta=\theta_0+\iota\varphi$", anchor="s", gap=0)

    fig.save(stem or os.path.join(HERE, NAME), previews=previews)
    if stem is None:
        out = dict(numbers)
        out["drawing"] = {
            "theta0_rad": {"value": theta0, "what": "where the line starts, at phi = 0: the node of the net at the "
                                                   "top of the section, lettered theta_0 in the square"},
            "phi_end_rad": {"value": phi_end, "what": "the far cut: the solid is drawn for phi from 0 to here and "
                                                     "the sector from here to 2 pi is cut away"},
            "line_end_theta_rad": {"value": (theta0 + iota * phi_end) % TWO_PI,
                                   "what": "theta where the line meets the far cut, in both halves"},
            "line_turns": {"toroidal": phi_end / TWO_PI, "poloidal": iota * phi_end / TWO_PI,
                           "what": "how far the drawn line goes round the machine and round the section"},
            "net": {"theta": kth, "phi": kph, "what": "curves of constant theta and of constant phi, every "
                                                      "2 pi / 12: the net on the solid and the grid of the square"},
            "slope_triangle_rad": {"run": D["slope"][1], "rise": iota * D["slope"][1],
                                   "what": "lettered 1 and iota in the square"},
            "lettered": "0 and 2 pi at the ends of both angles, theta_0 = pi / 2, the run 1 of the slope "
                        "triangle and its rise iota",
            "camera": {"azimuth_deg": D["azimuth"], "elevation_deg": D["elevation"], "distance_m": D["distance"],
                       "target_m": list(map(float, D["target"])),
                       "what": "a pinhole camera at that azimuth and elevation from the target, which it looks at"},
        }
        with open(NUMBERS, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compute", action="store_true", help="solve and trace again before drawing")
    ap.add_argument("--previews", help="also write PNG previews at 838 and 324 px into this folder")
    a = ap.parse_args()
    if a.compute or not os.path.exists(DATA):
        compute()
    draw(a.previews)
