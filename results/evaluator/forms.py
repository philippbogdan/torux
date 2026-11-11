"""The two surface forms every answer is written in, and their exact evaluation.

    rz_fourier   R(theta, phi) = sum_j rc_j cos(m_j theta - n_j phi) + rs_j sin(m_j theta - n_j phi)
                 Z(theta, phi) = sum_j zc_j cos(...)               + zs_j sin(...)
                 x = (R cos phi, R sin phi, Z): phi is the cylindrical angle
    xyz_fourier  x_a(theta, phi) = sum_j ac_j cos(m_j theta - n_j phi) + as_j sin(m_j theta - n_j phi),
                 a in {x, y, z}: phi is the surface's own toroidal parameter

theta and phi run over [0, 2 pi) once each (poloidally and toroidally once round the torus); m_j and n_j are
integers (n_j already holds any factor of the field periods).  A solver's native coefficients go into one of
these forms by exact coefficient arithmetic (`from_simsopt_xyz_tensor` below for SIMSOPT's
SurfaceXYZTensorFourier); every answer also carries points its own library evaluated, which `check_points`
compares against.

Everything here runs in the dtype it is given (numpy.longdouble for the evaluator's precise tier).
"""
import numpy as np

RZ_KEYS = ("rc", "rs", "zc", "zs")
XYZ_KEYS = ("xc", "xs", "yc", "ys", "zc", "zs")


def ld_pi(dtype):
    return dtype("3.141592653589793238462643383279502884197") if dtype is np.longdouble else dtype(np.pi)


class Surface:
    """A surface in one of the two forms, held in `dtype`; evaluates x, x_theta, x_phi at (theta, phi)."""

    def __init__(self, d, dtype=np.float64):
        self.form = d["form"]
        self.dtype = dtype
        self.m = np.asarray(d["m"], dtype=np.int64)
        self.n = np.asarray(d["n"], dtype=np.int64)
        keys = RZ_KEYS if self.form == "rz_fourier" else XYZ_KEYS
        if self.form not in ("rz_fourier", "xyz_fourier"):
            raise ValueError(f"unknown surface form {self.form!r}")
        self.c = {k: np.asarray([dtype(repr(float(v))) for v in d.get(k, [0.0] * len(self.m))], dtype=dtype)
                  for k in keys}
        J = len(self.m)
        assert len(self.n) == J and all(len(v) == J for v in self.c.values())
        self.mf = self.m.astype(dtype)
        self.nf = self.n.astype(dtype)

    def _series(self, cc, ss, C, S):
        """value, d/dtheta, d/dphi of sum cc cos(a) + ss sin(a), a = m theta - n phi; C, S are (P, J)."""
        v = C @ cc + S @ ss
        dth = S @ (-self.mf * cc) + C @ (self.mf * ss)
        dph = S @ (self.nf * cc) + C @ (-self.nf * ss)
        return v, dth, dph

    def eval(self, theta, phi, chunk=None):
        """theta, phi: (P,) arrays (converted to dtype).  Returns x, x_theta, x_phi, each (P, 3).
        Points go in chunks of at most about 2 million (point, mode) pairs, so memory stays bounded."""
        dt = self.dtype
        if chunk is None:
            chunk = max(1, min(256, 2_000_000 // max(1, len(self.m))))
        theta = np.asarray(theta).astype(dt)
        phi = np.asarray(phi).astype(dt)
        P = len(theta)
        X, XT, XP = (np.empty((P, 3), dtype=dt) for _ in range(3))
        for a in range(0, P, chunk):
            th, ph = theta[a:a + chunk], phi[a:a + chunk]
            ang = np.outer(th, self.mf) - np.outer(ph, self.nf)
            C, S = np.cos(ang), np.sin(ang)
            if self.form == "rz_fourier":
                R, Rt, Rp = self._series(self.c["rc"], self.c["rs"], C, S)
                Z, Zt, Zp = self._series(self.c["zc"], self.c["zs"], C, S)
                cp, sp = np.cos(ph), np.sin(ph)
                X[a:a + chunk] = np.stack([R * cp, R * sp, Z], -1)
                XT[a:a + chunk] = np.stack([Rt * cp, Rt * sp, Zt], -1)
                XP[a:a + chunk] = np.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp], -1)
            else:
                for j, ax in enumerate("xyz"):
                    v, t, p = self._series(self.c[ax + "c"], self.c[ax + "s"], C, S)
                    X[a:a + chunk, j], XT[a:a + chunk, j], XP[a:a + chunk, j] = v, t, p
        return X, XT, XP


def check_points(surface_dict, pts):
    """Largest |x(ours) - x(the solver's library)| over the answer's own check points, in metres."""
    s = Surface(surface_dict, np.float64)
    x, _, _ = s.eval(np.asarray(pts["theta"], float), np.asarray(pts["phi"], float))
    return float(np.max(np.abs(x - np.asarray(pts["xyz"], float))))


# ---------------------------------------------------------------------------------------------------------
# Exact conversions from native representations.

def _canonical(acc):
    """{(m, n): {key: value}} -> arrays, with m >= 0 and n >= 0 where m == 0 (cos even, sin odd)."""
    out = {}
    for (m, n), d in acc.items():
        sgn = 1.0
        if m < 0 or (m == 0 and n < 0):
            m, n, sgn = -m, -n, -1.0
        e = out.setdefault((m, n), {})
        for k, v in d.items():
            e[k] = e.get(k, 0.0) + (sgn * v if k.endswith("s") else v)
    keys = sorted(out)
    return keys, out


def from_simsopt_xyz_tensor(s):
    """SIMSOPT SurfaceXYZTensorFourier -> xyz_fourier, exactly (products of the tensor basis written as sums,
    then the rotation by the toroidal parameter).  SIMSOPT's definition, with theta = 2 pi u, phi = 2 pi v:
        xhat = sum_ij xcs[i, j] w_i(theta) v_j(phi), likewise yhat, zhat (z = zhat)
        w_i = cos(i theta) for i <= mpol, sin((i - mpol) theta) above; v_j = cos(nfp j phi) for j <= ntor,
        sin(nfp (j - ntor) phi) above
        x = xhat cos(phi) - yhat sin(phi), y = xhat sin(phi) + yhat cos(phi)."""
    mpol, ntor, nfp = s.mpol, s.ntor, s.nfp
    mats = {"x": np.asarray(s.xcs), "y": np.asarray(s.ycs), "z": np.asarray(s.zcs)}
    hat = {}   # (m, n) -> {ac/as: v} for xhat, yhat, zhat in cos/sin(m theta - n phi)

    def add(ax, m, n, kind, v):
        e = hat.setdefault((m, n), {})
        k = f"{ax}{kind}"
        e[k] = e.get(k, 0.0) + v

    for ax, A in mats.items():
        for i in range(2 * mpol + 1):
            m, wt = (i, "c") if i <= mpol else (i - mpol, "s")
            for j in range(2 * ntor + 1):
                v = float(A[i, j])
                if v == 0.0:
                    continue
                k, vt = (j, "c") if j <= ntor else (j - ntor, "s")
                n = nfp * k
                h = 0.5 * v
                # A = m theta, B = n phi; A - B -> mode (m, n); A + B -> mode (m, -n)
                if wt == "c" and vt == "c":      # cos A cos B = (cos(A-B) + cos(A+B)) / 2
                    add(ax, m, n, "c", h); add(ax, m, -n, "c", h)
                elif wt == "s" and vt == "s":    # sin A sin B = (cos(A-B) - cos(A+B)) / 2
                    add(ax, m, n, "c", h); add(ax, m, -n, "c", -h)
                elif wt == "s" and vt == "c":    # sin A cos B = (sin(A+B) + sin(A-B)) / 2
                    add(ax, m, n, "s", h); add(ax, m, -n, "s", h)
                else:                            # cos A sin B = (sin(A+B) - sin(A-B)) / 2
                    add(ax, m, -n, "s", h); add(ax, m, n, "s", -h)
    out = {}

    def put(m, n, k, v):
        e = out.setdefault((m, n), {})
        e[k] = e.get(k, 0.0) + v

    for (m, n), d in hat.items():
        xc, xs = d.get("xc", 0.0), d.get("xs", 0.0)
        yc, ys = d.get("yc", 0.0), d.get("ys", 0.0)
        # f(a) cos(phi), a = m theta - n phi:  cos a cos phi = (cos(a - phi) + cos(a + phi)) / 2
        #   a - phi -> (m, n + 1), a + phi -> (m, n - 1);  sin a cos phi = (sin(a - phi) + sin(a + phi)) / 2
        #   cos a sin phi = (sin(a + phi) - sin(a - phi)) / 2;  sin a sin phi = (cos(a - phi) - cos(a + phi)) / 2
        # x = xhat cos phi - yhat sin phi
        put(m, n + 1, "xc", 0.5 * xc); put(m, n - 1, "xc", 0.5 * xc)
        put(m, n + 1, "xs", 0.5 * xs); put(m, n - 1, "xs", 0.5 * xs)
        put(m, n - 1, "xs", -0.5 * yc); put(m, n + 1, "xs", 0.5 * yc)
        put(m, n + 1, "xc", -0.5 * ys); put(m, n - 1, "xc", 0.5 * ys)
        # y = xhat sin phi + yhat cos phi
        put(m, n - 1, "ys", 0.5 * xc); put(m, n + 1, "ys", -0.5 * xc)
        put(m, n + 1, "yc", 0.5 * xs); put(m, n - 1, "yc", -0.5 * xs)
        put(m, n + 1, "yc", 0.5 * yc); put(m, n - 1, "yc", 0.5 * yc)
        put(m, n + 1, "ys", 0.5 * ys); put(m, n - 1, "ys", 0.5 * ys)
        # z = zhat
        if "zc" in d:
            put(m, n, "zc", d["zc"])
        if "zs" in d:
            put(m, n, "zs", d["zs"])
    keys, can = _canonical(out)
    res = dict(form="xyz_fourier", m=[k[0] for k in keys], n=[k[1] for k in keys])
    for k in XYZ_KEYS:
        res[k] = [can[key].get(k, 0.0) for key in keys]
    return res


def simsopt_check_points(s, k=16, seed=7):
    """k points evaluated by SIMSOPT itself on a copy of s at random (u, v), as theta = 2 pi u, phi = 2 pi v."""
    from simsopt.geo import SurfaceXYZTensorFourier
    rng = np.random.default_rng(seed)
    u, v = rng.random(k), rng.random(k)
    out = dict(theta=[], phi=[], xyz=[])
    for a, b in zip(u, v):
        e = SurfaceXYZTensorFourier(nfp=s.nfp, stellsym=s.stellsym, mpol=s.mpol, ntor=s.ntor,
                                    quadpoints_phi=[b], quadpoints_theta=[a])
        e.x = s.x
        out["theta"].append(2 * np.pi * a)
        out["phi"].append(2 * np.pi * b)
        out["xyz"].append(e.gamma()[0, 0].tolist())
    return out
