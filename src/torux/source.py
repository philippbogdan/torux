"""Canonical description of an external magnetic-field source.

Every loader in ``torux.io`` turns a file into one or more
``Source`` objects; the field engine (``torux.field``) and the independent
reference evaluator (``torux.reference``) consume nothing else.

Units and frames: SI throughout -- lengths in m, currents
in A (the current of the whole filament, i.e. ampere-turns of the represented
winding), dipole moments in A m^2, fields in T, fluxes in Wb (full toroidal
flux, not divided by 2 pi).  Cartesian frame with the torus axis along z;
cylindrical (R, phi, Z) right-handed.  Positive current flows along
increasing curve parameter / point order.

A Source is the superposition of its parts.  Part kinds:

  FourierCurves    closed filaments r(t) = c0 + sum_k [s_k sin 2 pi k t + c_k cos 2 pi k t],
                   t in [0, 1); coef[i, a, :] = [c0, s1, c1, s2, c2, ..., sK, cK]
                   (the SIMSOPT CurveXYZFourier order) for Cartesian component a.
  PolylineCurves   closed polygons: points[start[i]:start[i+1]] in order, the
                   closing edge last -> first implied (no repeated end point).
  Segments         independent straight segments a[j] -> b[j] with current I[j].
  Dipoles          point dipoles at pos[j] with moment m[j].
  Rings            circular filament loops coaxial with z: radius R[j], height Z[j],
                   current I[j] (positive: counter-clockwise seen from +z, i.e.
                   along +phi).
  AxisymPolygons   axisymmetric conductors of polygonal (R, Z) cross-section
                   carrying total toroidal current I[j] (along +phi) uniformly
                   distributed over the cross-section area.
  ToroidalField    analytic 1/R toroidal field B_phi = RB / R (RB in T m).

Symmetry: ``nfp`` (discrete rotation by 2 pi / nfp about z) and ``stellsym``
(rotation by pi about the x axis) are *claims checked by the loader* on the
parts (``Source.check_symmetry``); the solver may exploit only verified
symmetry.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

MU0 = 4e-7 * np.pi  # SI value used by every data source here (pre-2019 definition)


# --------------------------------------------------------------------------- parts
@dataclass
class FourierCurves:
    coef: np.ndarray            # (n, 3, 2K+1)
    current: np.ndarray         # (n,)
    kind: str = "fourier"

    def __post_init__(self):
        self.coef = np.ascontiguousarray(self.coef, dtype=np.float64)
        self.current = np.ascontiguousarray(self.current, dtype=np.float64).reshape(-1)
        assert self.coef.ndim == 3 and self.coef.shape[1] == 3 and self.coef.shape[2] % 2 == 1
        assert self.coef.shape[0] == self.current.shape[0]

    @property
    def n(self):
        return self.coef.shape[0]

    @property
    def order(self):
        return (self.coef.shape[2] - 1) // 2

    def points(self, t):
        """Positions (n, len(t), 3) and derivatives d/dt (n, len(t), 3)."""
        t = np.asarray(t, dtype=np.float64)
        K = self.order
        k = np.arange(1, K + 1)
        ang = 2 * np.pi * np.outer(t, k)              # (nt, K)
        c, s = np.cos(ang), np.sin(ang)
        cc = self.coef[:, :, 2::2]                     # (n, 3, K) cosine coefficients
        ss = self.coef[:, :, 1::2]                     # (n, 3, K) sine coefficients
        x = self.coef[:, :, 0][:, None, :] + np.einsum("tk,nak->nta", c, cc) + np.einsum("tk,nak->nta", s, ss)
        dx = np.einsum("tk,nak->nta", -2 * np.pi * k * s, cc) + np.einsum("tk,nak->nta", 2 * np.pi * k * c, ss)
        return x, dx


@dataclass
class PolylineCurves:
    points: np.ndarray          # (P, 3)
    start: np.ndarray           # (n+1,) int offsets
    current: np.ndarray         # (n,)
    kind: str = "polyline"

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.start = np.ascontiguousarray(self.start, dtype=np.int64)
        self.current = np.ascontiguousarray(self.current, dtype=np.float64).reshape(-1)
        assert self.start[0] == 0 and self.start[-1] == self.points.shape[0]
        assert len(self.start) == len(self.current) + 1

    @property
    def n(self):
        return len(self.current)

    def loop(self, i):
        return self.points[self.start[i]:self.start[i + 1]]

    def as_segments(self):
        a, b, I = [], [], []
        for i in range(self.n):
            p = self.loop(i)
            a.append(p)
            b.append(np.roll(p, -1, axis=0))
            I.append(np.full(len(p), self.current[i]))
        return Segments(np.concatenate(a), np.concatenate(b), np.concatenate(I))


@dataclass
class Segments:
    a: np.ndarray               # (S, 3)
    b: np.ndarray               # (S, 3)
    current: np.ndarray         # (S,)
    kind: str = "segments"

    def __post_init__(self):
        self.a = np.ascontiguousarray(self.a, dtype=np.float64)
        self.b = np.ascontiguousarray(self.b, dtype=np.float64)
        self.current = np.ascontiguousarray(self.current, dtype=np.float64).reshape(-1)

    @property
    def n(self):
        return len(self.current)


@dataclass
class Dipoles:
    pos: np.ndarray             # (D, 3)
    moment: np.ndarray          # (D, 3)
    kind: str = "dipoles"

    def __post_init__(self):
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        self.moment = np.ascontiguousarray(self.moment, dtype=np.float64)

    @property
    def n(self):
        return self.pos.shape[0]


@dataclass
class Rings:
    R: np.ndarray
    Z: np.ndarray
    current: np.ndarray
    kind: str = "rings"

    def __post_init__(self):
        self.R = np.ascontiguousarray(self.R, dtype=np.float64).reshape(-1)
        self.Z = np.ascontiguousarray(self.Z, dtype=np.float64).reshape(-1)
        self.current = np.ascontiguousarray(self.current, dtype=np.float64).reshape(-1)

    @property
    def n(self):
        return len(self.current)


@dataclass
class AxisymPolygons:
    vertices: np.ndarray        # (V, 2) (R, Z), polygon i = vertices[start[i]:start[i+1]], not repeated
    start: np.ndarray           # (n+1,)
    current: np.ndarray         # (n,) total toroidal ampere-turns of each polygon
    kind: str = "axipoly"

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.start = np.ascontiguousarray(self.start, dtype=np.int64)
        self.current = np.ascontiguousarray(self.current, dtype=np.float64).reshape(-1)

    @property
    def n(self):
        return len(self.current)


@dataclass
class ToroidalField:
    RB: float                   # B_phi * R in T m
    kind: str = "toroidal"

    @property
    def n(self):
        return 1


PART_TYPES = {c.kind: c for c in (FourierCurves, PolylineCurves, Segments, Dipoles, Rings, AxisymPolygons)}
PART_TYPES["toroidal"] = ToroidalField


# --------------------------------------------------------------------------- source
@dataclass
class Source:
    id: str                                  # unique, path-like: "quasr/0000952", "w7x/standard", ...
    parts: list
    nfp: int = 1
    stellsym: bool = False
    family: str = ""                         # e.g. "QA", "QH", "QI", "tokamak", "dipole", ...
    request: Optional[dict] = None           # {"flux_Wb": float, "provenance": str} if the source has one
    references: list = field(default_factory=list)   # [{"kind": "vmec_wout"|..., "path": ..., "note": ...}]
    provenance: dict = field(default_factory=dict)   # {"files": [...], "loader": "torux.io.x:f", ...}
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- persistence
    def to_dict(self):
        parts = []
        for p in self.parts:
            d = {"kind": p.kind}
            for f in dataclasses.fields(p):
                if f.name == "kind":
                    continue
                v = getattr(p, f.name)
                d[f.name] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
            parts.append(d)
        return dict(id=self.id, parts=parts, nfp=self.nfp, stellsym=self.stellsym, family=self.family,
                    request=self.request, references=self.references, provenance=self.provenance, meta=self.meta)

    @classmethod
    def from_dict(cls, d):
        parts = []
        for p in d["parts"]:
            p = dict(p)
            k = p.pop("kind")
            cl = PART_TYPES[k]
            kw = {}
            for f in dataclasses.fields(cl):
                if f.name in p:
                    v = p[f.name]
                    kw[f.name] = np.asarray(v) if isinstance(v, list) else v
            parts.append(cl(**kw))
        return cls(id=d["id"], parts=parts, nfp=d.get("nfp", 1), stellsym=d.get("stellsym", False),
                   family=d.get("family", ""), request=d.get("request"), references=d.get("references", []),
                   provenance=d.get("provenance", {}), meta=d.get("meta", {}))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ---------------------------------------------------------------- summaries
    def summary(self):
        return {"id": self.id, "nfp": self.nfp, "stellsym": self.stellsym, "family": self.family,
                "parts": [(p.kind, p.n) for p in self.parts],
                "request": self.request}

    # ---------------------------------------------------------------- symmetry
    def sample_points(self):
        """Representative points of every current element (for symmetry checks)."""
        pts = []
        for p in self.parts:
            if p.kind == "fourier":
                x, _ = p.points(np.linspace(0, 1, 64, endpoint=False))
                pts.append(x.reshape(-1, 3))
            elif p.kind == "polyline":
                pts.append(p.points)
            elif p.kind == "segments":
                pts.append(0.5 * (p.a + p.b))
            elif p.kind == "dipoles":
                pts.append(p.pos)
        return np.concatenate(pts) if pts else np.zeros((0, 3))


def rotation_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


FLIP = np.diag([1.0, -1.0, -1.0])     # stellarator symmetry: rotation by pi about x
