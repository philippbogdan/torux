"""SIMSOPT serialisation (JSON, root class ``SIMSON``) -> Source objects.

Format: SIMSOPT's own (its json.py, coil.py, curve.py, curvexyzfourier.py, curveplanarfourier.py,
framedcurve.py, surfacexyztensorfourier.py/.h).

What is resolved
----------------
Currents (signed, A):
  Current        -> its DOFs x[0] (the ``current`` key is ignored when a DOFs object is
                    linked: where they differ, the DOFs value is the one the object uses)
  ScaledCurrent  -> scale * resolve(current_to_scale), recursively
  CurrentSum     -> resolve(current_a) + resolve(current_b)
Coils: Coil, RegularizedCoil, CircularRegularizedCoil, RectangularRegularizedCoil
  (``curve`` + ``current``; the regularisation only matters for self-forces).
Curves (each gives r(t), t in [0, 1)):
  CurveXYZFourier        exact: coefficients [c0, s1, c1, ...] per component (the canonical order)
  JaxCurveXYZFourier     same
  RotatedCurve           p' = F^flip R(phi) p (curve.py: gamma @ rotmat, rotmat = R(phi)^T [F]),
                         recursive; exact on coefficients
  CurvePlanarFourier     r(phi) polar Fourier series in a plane rotated by the normalised
                         quaternion and translated (curveplanarfourier.py, jaxplanarcurve_pure);
                         converted exactly to Cartesian Fourier order K+1 (trig products)
  FramedFilament (CSX csx_finite_build), CurveFilament (simsopt finitebuild):
                         gamma + shift_n * n_alpha + shift_b * b_alpha, with the rotated
                         centroid or Frenet frame of framedcurve.py (rotated_centroid_frame /
                         rotated_frenet_frame, alpha from FrameRotation / ZeroRotation).  The
                         centroid is the mean of gamma over the curve's stored quadpoints
                         (as simsopt computes it).  Not a trigonometric polynomial: converted
                         to Fourier by FFT with a measured truncation error (meta['fit']).
  CurveCWSFourier        curve on a SurfaceRZFourier winding surface (idofs): DOFs
                         [theta_l, theta_c(0..N), theta_s(1..N), phi_l, phi_c(0..N), phi_s(1..N)],
                         angles in radians, theta(t) = 2 pi theta_l t + ..., phi likewise;
                         r = (R cos phi, R sin phi, Z), R = sum rc cos(m theta - nfp n phi),
                         Z = sum zs sin(m theta - nfp n phi).  The DOF layout was established by a
                         B.n test on the curve's own surface.  FFT-converted with measured error.
  OrientedCurveXYZFourier  local zero-mean Fourier curve (DOFs x0 y0 z0 yaw pitch roll, then
                         xs(k) xc(k) ys(k) yc(k) zs(k) zc(k)) rotated and translated.  Only
                         roll != 0 is handled (roll = pi/2 in the CNT coil sets); the map
                         is r = Rx(-roll) r_local + x0 (equivalently r_local @ Rx(roll) as row
                         vectors), established by B.n on its own Boozer surface: 2.2e-3 max
                         (4.8e-4 mean) versus 0.39 with Rx(+roll).  Nonzero yaw/pitch raise.
  csx_windowpane_curve_constant_R  r = (z(t), Rv cos phi(t), Rv sin phi(t)); FFT-converted.
  CurveXYZFourierSymmetries, and CurveXYZHelical (its earlier name, simsopt.geo.curvexyzhelical at
                         SIMSOPT fddc130f, the same curve with ntor = 1): xhat, yhat, z Fourier series in
                         2 pi nfp m t (stellsym: xc(0..order), ys(1..order), zs(1..order), otherwise xc(0..),
                         xs(1..), yc(0..), ys(1..), zc(0..), zs(1..)), and x + i y = exp(2 pi i ntor t)
                         (xhat + i yhat).  A trigonometric polynomial of order nfp order + |ntor|: converted
                         exactly, the DOF names checked against this layout.
Field objects: BiotSavart (``coils``: the full list), BoozerSurface (``biotsavart``,
``surface``).  Every BiotSavart object of a file is one candidate field source; the
caller chooses among them.

Fourier conversion of non-polynomial curves: samples at N = 2^k uniform t (k = 8..14),
real FFT, truncation at the last harmonic above 2^-52 * max|r|; accepted when the fit
reproduces r(t) at the N off-grid midpoints to <= 1e-13 * max|r| (the measured error is
recorded).  Otherwise the curve is represented by the dense polygon of its samples
(PolylineCurves) and the polygon's field error is stated in meta.

Public API:
  read_simson(path) -> Simson
  Simson.biotsavarts(), .coils_of(name), .surfaces(), .boozer_surfaces()
  coils_to_source(sim, coil_names, source_id, ...) -> Source
  load_simson_sources(path, id_prefix, ...) -> list[Source]   (one per distinct BiotSavart)
  load_quasr_serial(path, source_id) -> Source  (fast path, graph [surfaces, coils])
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from torux.source import FLIP, Source, rotation_z
from torux.io._stellcommon import (apply_verified_symmetry, data_path, fourier_part, polyline_part,
                                 rel_path)

COIL_CLASSES = {"Coil", "RegularizedCoil", "CircularRegularizedCoil", "RectangularRegularizedCoil"}
EPS = 2.0 ** -52


# --------------------------------------------------------------------------- GSON decoding
def _arr(v):
    if isinstance(v, dict) and v.get("@module") == "numpy" and v.get("@class") == "array":
        return np.asarray(v["data"], dtype=v.get("dtype", None))
    return v


class Simson:
    def __init__(self, d, path=None):
        if d.get("@class") != "SIMSON":
            raise ValueError(f"{path}: not a SIMSON file")
        self.path = str(path) if path is not None else None
        self.graph = d["graph"]
        self.objs = d["simsopt_objs"]
        self.version = d.get("@version")
        self._cur = {}
        self.use_current_key = False   # True: Current value from its 'current' key (non-default)

    def obj(self, ref):
        if isinstance(ref, dict) and ref.get("$type") == "ref":
            return self.objs[ref["value"]]
        if isinstance(ref, str):
            return self.objs[ref]
        return ref

    def name(self, ref):
        return ref["value"] if isinstance(ref, dict) else ref

    def dofs_x(self, o):
        """Full DOF vector of an object with a linked DOFs object (``dofs`` key)."""
        d = self.obj(o["dofs"])
        return np.asarray(_arr(d["x"]), dtype=np.float64)

    def classes(self):
        from collections import Counter
        return Counter(v["@class"] for v in self.objs.values())

    # ---------------------------------------------------------------- currents
    def current(self, ref):
        nm = self.name(ref)
        if nm in self._cur:
            return self._cur[nm]
        o = self.obj(ref)
        c = o["@class"]
        if c == "Current":
            if self.use_current_key and "current" in o:
                v = float(o["current"])
            elif "dofs" in o and o["dofs"] is not None:
                v = float(self.dofs_x(o)[0])
            else:
                v = float(o["current"])
        elif c == "ScaledCurrent":
            v = float(o["scale"]) * self.current(o["current_to_scale"])
        elif c == "CurrentSum":
            v = self.current(o["current_a"]) + self.current(o["current_b"])
        else:
            raise NotImplementedError(f"current class {c}")
        self._cur[nm] = v
        return v

    def current_stale(self, ref):
        """Differences between stored ``current`` keys and DOFs values."""
        out = []
        for k, o in self.objs.items():
            if o["@class"] == "Current" and "dofs" in o and o["dofs"] is not None:
                xv = float(self.dofs_x(o)[0])
                if "current" in o and float(o["current"]) != xv:
                    out.append((k, float(o["current"]), xv))
        return out

    # ---------------------------------------------------------------- graph search
    def _walk(self, node, seen, out_cls, acc):
        if isinstance(node, list):
            for v in node:
                self._walk(v, seen, out_cls, acc)
        elif isinstance(node, dict):
            if node.get("$type") == "ref":
                nm = node["value"]
                if nm in seen:
                    return
                seen.add(nm)
                o = self.objs[nm]
                if o["@class"] in out_cls:
                    acc.append(nm)
                for v in o.values():
                    self._walk(v, seen, out_cls, acc)
            else:
                for v in node.values():
                    self._walk(v, seen, out_cls, acc)

    def find(self, classes, from_graph=True):
        """Names of objects of the given classes, in graph order (then any unreferenced ones)."""
        acc = []
        seen = set()
        if from_graph:
            self._walk(self.graph, seen, set(classes), acc)
        for k, o in self.objs.items():
            if o["@class"] in classes and k not in acc:
                acc.append(k)
        return acc

    def biotsavarts(self):
        return self.find({"BiotSavart"})

    def boozer_surfaces(self):
        return self.find({"BoozerSurface"})

    def surfaces(self):
        return self.find({"SurfaceXYZTensorFourier", "SurfaceRZFourier", "SurfaceXYZFourier"})

    def coils_of(self, bs_name):
        o = self.objs[bs_name]
        return [self.name(r) for r in o["coils"]]

    def graph_coil_list(self):
        """For QUASR-style graphs [surfaces, coils]: the coil names of graph[1]."""
        g = self.graph
        if isinstance(g, list) and len(g) == 2 and isinstance(g[1], list) and g[1] and \
                isinstance(g[1][0], dict) and self.objs[g[1][0]["value"]]["@class"] in COIL_CLASSES:
            return [r["value"] for r in g[1]]
        return None

    # ---------------------------------------------------------------- curves
    def curve(self, ref):
        return curve_from_obj(self, self.obj(ref))


def read_simson(path):
    p = data_path(path)
    with open(p, "rb") as fh:
        d = json.loads(fh.read())
    return Simson(d, p)


# --------------------------------------------------------------------------- curve representations
@dataclass
class CurveRep:
    kind: str
    func: Callable                      # t (m,) -> (m, 3)
    coef: Optional[np.ndarray] = None   # (3, 2K+1) when the curve is an exact trig polynomial
    quadpoints: Optional[np.ndarray] = None
    note: str = ""


def eval_coef(coef, t):
    t = np.asarray(t, dtype=np.float64)
    K = (coef.shape[1] - 1) // 2
    k = np.arange(1, K + 1)
    ang = 2 * np.pi * np.outer(t, k)
    return coef[:, 0][None, :] + np.cos(ang) @ coef[:, 2::2].T + np.sin(ang) @ coef[:, 1::2].T


def deriv_coef(coef):
    """Coefficients of dr/dt (same layout)."""
    K = (coef.shape[1] - 1) // 2
    out = np.zeros_like(coef)
    k = 2 * np.pi * np.arange(1, K + 1)
    s, c = coef[:, 1::2], coef[:, 2::2]
    out[:, 1::2] = -k * c          # d/dt (c cos) = -k c sin
    out[:, 2::2] = k * s           # d/dt (s sin) =  k s cos
    return out


def _coef_rep(coef, kind, quadpoints=None):
    coef = np.asarray(coef, dtype=np.float64)
    return CurveRep(kind, lambda t, c=coef: eval_coef(c, t), coef, quadpoints)


def _xyz_dofs_to_coef(x, order):
    x = np.asarray(x, dtype=np.float64)
    n = 2 * order + 1
    return x[:3 * n].reshape(3, n)


def curve_from_obj(sim, o):
    c = o["@class"]
    qp = _arr(o.get("quadpoints")) if "quadpoints" in o else None
    if c in ("CurveXYZFourier", "JaxCurveXYZFourier"):
        return _coef_rep(_xyz_dofs_to_coef(sim.dofs_x(o), int(o["order"])), c, qp)
    if c == "RotatedCurve":
        base = sim.curve(o["curve"])
        phi = float(o["phi"])
        M = rotation_z(phi)
        if o.get("flip"):
            M = FLIP @ M
        if base.coef is not None:
            return _coef_rep(M @ base.coef, f"RotatedCurve({base.kind})", base.quadpoints)
        return CurveRep(f"RotatedCurve({base.kind})", lambda t, f=base.func, M=M: f(t) @ M.T, None,
                        base.quadpoints, base.note)
    if c in ("CurvePlanarFourier", "JaxCurvePlanarFourier"):
        return _planar_rep(sim.dofs_x(o), int(o["order"]), qp)
    if c in ("FramedFilament", "CurveFilament"):
        return _filament_rep(sim, o)
    if c == "CurveCWSFourier":
        return _cws_rep(sim, o)
    if c == "OrientedCurveXYZFourier":
        return _oriented_rep(sim.dofs_x(o), int(o["order"]), qp)
    if c in ("CurveXYZFourierSymmetries", "CurveXYZHelical"):
        return _helical_rep(sim, o, qp)
    if c == "csx_windowpane_curve_constant_R":
        return _windowpane_R_rep(sim.dofs_x(o), int(o["order"]), float(o["Rvessel"]), qp)
    if c == "csx_windowpane_curve_constant_Z":
        return _windowpane_Z_rep(sim.dofs_x(o), int(o["order"]), float(o["Zvessel"]), qp)
    raise NotImplementedError(f"curve class {c}: torux does not read this SIMSOPT curve class")


# ---- CurveXYZFourierSymmetries and CurveXYZHelical (exact)
def helical_names(order, stellsym):
    """The DOF names of CurveXYZFourierSymmetries (and CurveXYZHelical) in SIMSOPT's order."""
    if stellsym:
        return ([f"xc({i})" for i in range(order + 1)] + [f"ys({i})" for i in range(1, order + 1)]
                + [f"zs({i})" for i in range(1, order + 1)])
    out = []
    for a in "xyz":
        out += [f"{a}c(0)"] + [f"{a}c({i})" for i in range(1, order + 1)] + [f"{a}s({i})" for i in range(1, order + 1)]
    return out


def helical_gamma(x, order, nfp, stellsym, ntor, t):
    """r(t) of a CurveXYZFourierSymmetries with DOFs x (SIMSOPT's jaxXYZFourierSymmetriescurve_pure)."""
    x = np.asarray(x, dtype=np.float64)
    w = 2 * np.pi * np.asarray(t, dtype=np.float64)
    m = np.arange(order + 1)
    C = np.cos(np.outer(w, nfp * m))
    S = np.sin(np.outer(w, nfp * m))[:, 1:]
    k = order + 1
    if stellsym:
        xh = C @ x[:k]
        yh = S @ x[k:k + order]
        z = S @ x[k + order:k + 2 * order]
    else:
        b = 2 * order + 1
        xh, yh, z = (C @ x[j * b:j * b + k] + S @ x[j * b + k:(j + 1) * b] for j in range(3))
    a = ntor * w
    return np.stack([np.cos(a) * xh - np.sin(a) * yh, np.sin(a) * xh + np.cos(a) * yh, z], axis=1)


def _helical_rep(sim, o, qp):
    order, nfp, stellsym = int(o["order"]), int(o["nfp"]), bool(o["stellsym"])
    ntor = int(o.get("ntor", 1))
    x = sim.dofs_x(o)
    names = sim.obj(o["dofs"]).get("names")
    want = helical_names(order, stellsym)
    if x.size != len(want) or (names is not None and list(names) != want):
        raise NotImplementedError(f"{o['@class']} {o.get('@name', '')}: its degrees of freedom are not the layout "
                                  f"torux reads ({len(want)} named {want[0]} ... {want[-1]})")
    K = nfp * order + abs(ntor)
    N = 4 * (K + 2)
    coef = coef_from_samples(helical_gamma(x, order, nfp, stellsym, ntor, np.arange(N) / N))
    tail = np.abs(coef[:, 2 * K + 1:]).max() if coef.shape[1] > 2 * K + 1 else 0.0
    scale = np.abs(coef).max()
    assert tail <= 1e-13 * scale, ("helical conversion not exact", tail, scale)
    rep = _coef_rep(coef[:, :2 * K + 1].copy(), o["@class"], qp)
    rep.note = f"exact conversion to Cartesian Fourier order {K}, discarded harmonics <= {tail:.1e} m"
    return rep


# ---- CurvePlanarFourier (exact)
def planar_gamma(dofs, order, t):
    dofs = np.asarray(dofs, dtype=np.float64)
    coeffs = dofs[:2 * order + 1]
    q = dofs[2 * order + 1:2 * order + 5]
    qn = q / np.linalg.norm(q)
    center = dofs[2 * order + 5:2 * order + 8]
    phi = 2 * np.pi * np.asarray(t)
    j = np.arange(1, order + 1)[:, None]
    r = coeffs[0] + np.sum(coeffs[1:order + 1, None] * np.cos(j * phi[None]) +
                           coeffs[order + 1:2 * order + 1, None] * np.sin(j * phi[None]), axis=0)
    xi, yi = r * np.cos(phi), r * np.sin(phi)
    q0, qi, qj, qk = qn
    gx = (1 - 2 * (qj * qj + qk * qk)) * xi + 2 * (qi * qj - qk * q0) * yi + center[0]
    gy = (1 - 2 * (qi * qi + qk * qk)) * yi + 2 * (q0 * qk + qi * qj) * xi + center[1]
    gz = 2 * (qi * qk - q0 * qj) * xi + 2 * (q0 * qi + qj * qk) * yi + center[2]
    return np.stack([gx, gy, gz], axis=1)


def coef_from_samples(X):
    """SIMSOPT-order coefficients (3, 2K+1), K = N/2 - 1, from N uniform samples (N, 3)."""
    N = X.shape[0]
    F = np.fft.rfft(X, axis=0) / N
    K = (N - 1) // 2 if N % 2 else N // 2 - 1
    out = np.zeros((3, 2 * K + 1))
    out[:, 0] = F[0].real
    out[:, 2::2] = 2 * F[1:K + 1].real.T
    out[:, 1::2] = -2 * F[1:K + 1].imag.T
    return out


def _planar_rep(dofs, order, qp):
    K = order + 1
    N = 4 * (K + 2)
    t = np.arange(N) / N
    coef = coef_from_samples(planar_gamma(dofs, order, t))
    tail = np.abs(coef[:, 2 * K + 1:]).max() if coef.shape[1] > 2 * K + 1 else 0.0
    scale = np.abs(coef).max()
    assert tail <= 1e-13 * scale, ("planar conversion not exact", tail, scale)
    coef = coef[:, :2 * K + 1].copy()
    rep = _coef_rep(coef, "CurvePlanarFourier", qp)
    rep.note = f"exact conversion to Cartesian Fourier order {K}; discarded harmonics <= {tail:.1e} m"
    return rep


# ---- framed filaments (CSX finite build)
def _alpha(sim, rot_ref, t):
    if rot_ref is None:
        return np.zeros_like(t)
    o = sim.obj(rot_ref)
    if o["@class"] == "ZeroRotation":
        return np.zeros_like(t)
    if o["@class"] != "FrameRotation":
        raise NotImplementedError(o["@class"])
    x = sim.dofs_x(o)
    order = int(o["order"])
    a = np.full_like(t, x[0], dtype=np.float64)
    for j in range(1, order + 1):
        a = a + x[2 * j - 1] * np.sin(2 * np.pi * j * t) + x[2 * j] * np.cos(2 * np.pi * j * t)
    return float(o.get("scale", 1.0)) * a


def _filament_rep(sim, o):
    fc_ref = o.get("framedcurve")
    fc = sim.obj(fc_ref)
    if o["@class"] == "FramedFilament":
        dn, db = float(o["shift_n"]), float(o["shift_b"])
    else:   # simsopt CurveFilament
        dn, db = float(o["dn"]), float(o["db"])
    base = sim.curve(fc["curve"])
    if base.coef is None:
        raise NotImplementedError("framed filament on a non-Fourier base curve")
    frame = fc["@class"]
    rot_ref = fc.get("rotation")
    coef = base.coef
    d1 = deriv_coef(coef)
    d2 = deriv_coef(d1)
    qp = base.quadpoints if base.quadpoints is not None else np.linspace(0, 1, 200, endpoint=False)
    centroid = eval_coef(coef, qp).mean(axis=0)

    def func(t, coef=coef, d1=d1, d2=d2, centroid=centroid):
        t = np.asarray(t, dtype=np.float64)
        g = eval_coef(coef, t)
        gd = eval_coef(d1, t)
        tt = gd / np.linalg.norm(gd, axis=1)[:, None]
        if frame == "FramedCurveCentroid":
            delta = g - centroid[None, :]
            n = delta - np.sum(delta * tt, axis=1)[:, None] * tt
            n = n / np.linalg.norm(n, axis=1)[:, None]
        elif frame == "FramedCurveFrenet":
            gdd = eval_coef(d2, t)
            ngd = np.linalg.norm(gd, axis=1)[:, None]
            tdash = (ngd * gdd - (np.sum(gd * gdd, axis=1)[:, None] / ngd) * gd) / ngd ** 2
            n = tdash / np.linalg.norm(tdash, axis=1)[:, None]
        else:
            raise NotImplementedError(frame)
        b = np.cross(tt, n)
        al = _alpha(sim, rot_ref, t)
        nn = np.cos(al)[:, None] * n - np.sin(al)[:, None] * b
        bb = np.sin(al)[:, None] * n + np.cos(al)[:, None] * b
        return g + dn * nn + db * bb

    return CurveRep(f"{o['@class']}[{frame}]", func, None, qp,
                    f"offset filament dn={dn}, db={db} in the rotated {frame} frame")


# ---- CurveCWSFourier
def _surface_rz_from_idofs(idofs, mpol, ntor, stellsym=True):
    """SurfaceRZFourier DOF layout (simsopt): rc(0, 0..ntor), rc(1..mpol, -ntor..ntor), then
    zs(0, 1..ntor), zs(1..mpol, -ntor..ntor) [stellsym]; non-stellsym adds rs, zc likewise."""
    idofs = np.asarray(idofs, dtype=np.float64)
    modes = [(0, n) for n in range(0, ntor + 1)] + [(m, n) for m in range(1, mpol + 1) for n in range(-ntor, ntor + 1)]
    modes_s = [(0, n) for n in range(1, ntor + 1)] + [(m, n) for m in range(1, mpol + 1) for n in range(-ntor, ntor + 1)]
    k = 0
    rc = {}
    for mn in modes:
        rc[mn] = idofs[k]; k += 1
    if not stellsym:
        rs = {}
        for mn in modes_s:
            rs[mn] = idofs[k]; k += 1
        zc = {}
        for mn in modes:
            zc[mn] = idofs[k]; k += 1
    zs = {}
    for mn in modes_s:
        zs[mn] = idofs[k]; k += 1
    assert k == len(idofs), (k, len(idofs))
    if stellsym:
        return rc, zs, {}, {}
    return rc, zs, rs, zc


CWS_LAYOUT = "theta_first_cos_then_sin"


def cws_angles(x, order, t, layout=None):
    layout = layout or CWS_LAYOUT
    x = np.asarray(x, dtype=np.float64)
    nb = 2 * order + 2
    blocks = [x[:nb], x[nb:2 * nb]]
    if layout.startswith("phi_first"):
        blocks = blocks[::-1]
    w = 2 * np.pi * np.asarray(t)
    out = []
    for b in blocks:
        lin, c0 = b[0], b[1]
        rest = b[2:]
        if layout.endswith("cos_then_sin"):
            cs, ss = rest[:order], rest[order:]
        else:   # interleaved s1, c1, s2, c2 ...
            ss, cs = rest[0::2], rest[1::2]
        a = lin * w + c0
        for j in range(1, order + 1):
            a = a + cs[j - 1] * np.cos(j * w) + ss[j - 1] * np.sin(j * w)
        out.append(a)
    return out[0], out[1]       # theta, phi (radians)


def _cws_rep(sim, o, layout=None):
    mpol, ntor = int(o["mpol"]), int(o["ntor"])
    nfp = int(o["nfp"])
    ss = bool(o.get("stellsym", True))
    rc, zs, rs, zc = _surface_rz_from_idofs(o["idofs"], mpol, ntor, ss)
    x = sim.dofs_x(o)
    order = int(o["order"])
    qp = _arr(o.get("quadpoints"))

    def func(t, x=x):
        th, ph = cws_angles(x, order, t, layout)
        R = np.zeros_like(th)
        Z = np.zeros_like(th)
        for (m, n), v in rc.items():
            R += v * np.cos(m * th - nfp * n * ph)
        for (m, n), v in zs.items():
            Z += v * np.sin(m * th - nfp * n * ph)
        for (m, n), v in rs.items():
            R += v * np.sin(m * th - nfp * n * ph)
        for (m, n), v in zc.items():
            Z += v * np.cos(m * th - nfp * n * ph)
        return np.stack([R * np.cos(ph), R * np.sin(ph), Z], axis=1)

    return CurveRep("CurveCWSFourier", func, None, qp, f"CWS curve (layout {layout or CWS_LAYOUT})")


# ---- OrientedCurveXYZFourier
ORIENTED_ROLL_SIGN = -1.0   # established by B.n on cnt32_opt_56 (see module docstring)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def oriented_matrix(yaw, pitch, roll, sign=None):
    sgn = ORIENTED_ROLL_SIGN if sign is None else sign
    return rotation_z(sgn * yaw) @ _rot_y(sgn * pitch) @ _rot_x(sgn * roll)


def _oriented_rep(x, order, qp, sign=None):
    x = np.asarray(x, dtype=np.float64)
    center = x[:3]
    yaw, pitch, roll = x[3:6]
    if yaw != 0.0 or pitch != 0.0:
        # only roll is handled; the yaw/pitch composition order is unverified
        raise NotImplementedError("OrientedCurveXYZFourier with nonzero yaw/pitch: convention unverified")
    loc = np.zeros((3, 2 * order + 1))
    for j in range(order):
        blk = x[6 + 6 * j:12 + 6 * j]   # xs, xc, ys, yc, zs, zc of harmonic j+1
        for a in range(3):
            loc[a, 2 * j + 1] = blk[2 * a]
            loc[a, 2 * j + 2] = blk[2 * a + 1]
    M = oriented_matrix(yaw, pitch, roll, sign)
    coef = M @ loc
    coef[:, 0] += center
    rep = _coef_rep(coef, "OrientedCurveXYZFourier", qp)
    rep.note = "Rz(yaw) Ry(pitch) Rx(roll) applied to the zero-mean local Fourier curve, then translated"
    return rep


# ---- CSX windowpane curves (csx_windowpane.py)
def _windowpane_series(x, order, t):
    c = x[0:order + 1]
    s = x[order + 1:2 * order + 1]
    w = 2 * np.pi * np.asarray(t)
    v = np.full_like(w, c[0])
    for j in range(1, order + 1):
        v = v + s[j - 1] * np.sin(j * w) + c[j] * np.cos(j * w)
    return v


def _windowpane_R_rep(x, order, Rv, qp):
    x = np.asarray(x, dtype=np.float64)

    def func(t):
        z = _windowpane_series(x[:2 * order + 1], order, t)
        ph = _windowpane_series(x[2 * order + 1:4 * order + 2], order, t)
        return np.stack([z, Rv * np.cos(ph), Rv * np.sin(ph)], axis=1)

    return CurveRep("csx_windowpane_curve_constant_R", func, None, qp,
                    f"windowpane on the cylinder y^2+z^2={Rv}^2 about the x axis")


def _windowpane_Z_rep(x, order, Zv, qp):
    x = np.asarray(x, dtype=np.float64)

    def func(t):
        a = _windowpane_series(x[:2 * order + 1], order, t)
        b = _windowpane_series(x[2 * order + 1:4 * order + 2], order, t)
        return np.stack([np.full_like(a, Zv), a, b], axis=1)

    return CurveRep("csx_windowpane_curve_constant_Z", func, None, qp, f"windowpane in the plane x={Zv}")


# --------------------------------------------------------------------------- Fourier fitting
def fit_fourier(func, nmin=256, nmax=16384, rtol=1e-13):
    """Fourier coefficients of a smooth closed curve given as a function, with measured error.

    Returns (coef (3, 2K+1), err_abs, scale, N) or (None, err_abs, scale, N) if not converged.
    """
    N = nmin
    last = None
    while N <= nmax:
        t = np.arange(N) / N
        X = func(t)
        coef = coef_from_samples(X)
        scale = float(np.abs(X).max())
        K = (coef.shape[1] - 1) // 2
        mags = np.abs(coef[:, 1:]).reshape(3, K, 2).max(axis=(0, 2))
        big = np.where(mags > EPS * scale)[0]
        Kc = int(big[-1]) + 1 if len(big) else 0
        tm = (np.arange(N) + 0.5) / N
        if Kc < 0.75 * K:
            c = coef[:, :2 * Kc + 1]
            err = float(np.abs(eval_coef(c, tm) - func(tm)).max())
            last = (c, err, scale, N)
            if err <= rtol * scale:
                return c.copy(), err, scale, N
        N *= 2
    if last is None:
        t = np.arange(nmax) / nmax
        X = func(t)
        last = (None, float("inf"), float(np.abs(X).max()), nmax)
    return None, last[1], last[2], last[3]


# --------------------------------------------------------------------------- coils -> Source
def resolve_coils(sim, coil_names, fit_rtol=1e-13):
    """[(coef or None, polygon or None, current, info)] for the listed Coil objects."""
    out = []
    for nm in coil_names:
        o = sim.objs[nm]
        if o["@class"] not in COIL_CLASSES:
            raise NotImplementedError(f"coil class {o['@class']}")
        rep = sim.curve(o["curve"])
        I = sim.current(o["current"])
        info = {"coil": nm, "curve_kind": rep.kind}
        if rep.note:
            info["note"] = rep.note
        if rep.coef is not None:
            out.append((rep.coef, None, I, info))
            continue
        coef, err, scale, N = fit_fourier(rep.func, rtol=fit_rtol)
        info["fit"] = {"max_abs_error_m": err, "scale_m": scale, "N_samples": N,
                       "order": None if coef is None else (coef.shape[1] - 1) // 2}
        if coef is not None:
            out.append((coef, None, I, info))
        else:
            n = 4096
            out.append((None, rep.func(np.arange(n) / n), I, info))
    return out


def coils_to_source(sim, coil_names, source_id, nfp_candidates=(1,), stellsym_claim=True,
                    check_symmetry=True, family="", extra_files=(), loader="torux.io.simsopt_json:load_simson_sources"):
    res = resolve_coils(sim, coil_names)
    four = [(c, I) for c, p, I, _ in res if c is not None]
    poly = [(p, I) for c, p, I, _ in res if p is not None]
    parts = []
    if four:
        parts.append(fourier_part([c for c, _ in four], [I for _, I in four]))
    if poly:
        parts.append(polyline_part([p for p, _ in poly], [I for _, I in poly]))
    kinds = sorted({info["curve_kind"] for *_, info in res})
    fits = [info["fit"] for *_, info in res if "fit" in info]
    meta = {"simsopt_version": sim.version, "curve_kinds": kinds, "n_coil_objects": len(coil_names)}
    if fits:
        meta["fit"] = {"max_abs_error_m": max(f["max_abs_error_m"] for f in fits),
                       "max_order": max((f["order"] or 0) for f in fits),
                       "n_fitted": len(fits), "n_polygon": len(poly)}
    notes = sorted({info["note"] for *_, info in res if "note" in info})
    if notes:
        meta["curve_notes"] = notes
    stale = sim.current_stale(None)
    if stale:
        meta["stale_current_keys"] = stale[:20]
    files = [rel_path(sim.path)] + [rel_path(f) for f in extra_files]
    src = Source(id=source_id, parts=parts, family=family,
                 provenance={"files": files, "loader": loader, "format": "SIMSOPT SIMSON JSON"}, meta=meta)
    if check_symmetry:
        apply_verified_symmetry(src, nfp_candidates, stellsym_claim)
    return src


def nfp_hints(sim):
    """nfp values suggested by the file (surfaces' nfp, rotation angles of RotatedCurves)."""
    c = set()
    for k, o in sim.objs.items():
        if o["@class"] in ("SurfaceXYZTensorFourier", "SurfaceRZFourier", "CurveCWSFourier",
                           "CurveXYZFourierSymmetries", "CurveXYZHelical"):
            c.add(int(o["nfp"]))
        if o["@class"] == "RotatedCurve":
            ph = float(o["phi"]) % (2 * np.pi)
            if ph > 1e-12:
                n = 2 * np.pi / ph
                if abs(n - round(n)) < 1e-9:
                    c.add(int(round(n)))
    return sorted(c)


def load_simson_sources(path, id_prefix, family="", check_symmetry=True, extra_nfp=()):
    """One Source per BiotSavart object of the file (or the graph's coil list), in graph order.

    Ids: id_prefix if there is one field object, else f"{id_prefix}#{BiotSavartName}".
    """
    sim = read_simson(path)
    groups = []
    gl = sim.graph_coil_list()
    if gl is not None:
        groups.append(("coils", gl))
    for bs in sim.biotsavarts():
        groups.append((bs, sim.coils_of(bs)))
    if not groups:
        coils = sim.find(COIL_CLASSES)
        if coils:
            groups.append(("coils", coils))
    cands = sorted(set(nfp_hints(sim)) | set(extra_nfp) | {1})
    out = []
    for name, coils in groups:
        sid = id_prefix if len(groups) == 1 else f"{id_prefix}#{name}"
        src = coils_to_source(sim, coils, sid, nfp_candidates=cands, check_symmetry=check_symmetry, family=family)
        src.meta["field_object"] = name
        out.append(src)
    return out


def load_quasr_serial(path, source_id, check_symmetry=True, family=""):
    """Fast path for QUASR serial files (graph = [surfaces, coils])."""
    sim = read_simson(path)
    coils = sim.graph_coil_list()
    if coils is None:
        raise ValueError(f"{path}: not a QUASR-style graph")
    surf = [sim.objs[r["value"]] for r in sim.graph[0]]
    nfp = int(surf[0]["nfp"]) if surf else 1
    ss = bool(surf[0]["stellsym"]) if surf else True
    src = coils_to_source(sim, coils, source_id, nfp_candidates=[nfp], stellsym_claim=ss,
                          check_symmetry=check_symmetry, family=family,
                          loader="torux.io.simsopt_json:load_quasr_serial")
    src.meta["n_surfaces"] = len(surf)
    return src, sim
