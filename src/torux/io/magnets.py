"""Permanent-magnet sources and the coil readers they need.

Readers here:
  * read_pm_csv            Hammond & Kaptanoglu (2024) ``fig*_moments.csv`` tables
                           (ox, oy, oz, mx, my, mz, mr, mp; one half period of NCSX,
                           zero rows = unused positions).
  * read_corners_csv       magnet block corners (8 vertices per magnet,
                           t1..t4, b1..b4; with or without a header row).
  * blocks_to_dipoles      the "block corners" variant: each magnet as a uniformly
                           magnetised hexahedron, represented exactly as the volume
                           integral of its dipole density and discretised by an
                           n x n x n Gauss-Legendre rule on the trilinear map of the
                           block (n = 1: one dipole at the centre).  The quadrature
                           error decays like (block size / distance)^(2n) outside the
                           block; the point-dipole tables are the n = 1 member.
  * load_gpmo_json         QUADCOIL ``PM/GPMO_data_muse.json`` (SIMSON: BiotSavart of
                           the 16 MUSE TF coils, m_history (75460, 3, 22), R2_history).
  * read_focus_coils       FOCUS coil files (private reader: torux.io.focus of the
                           stellarator loaders did not exist when this was written).
                           coil_type 1 = Fourier filament (coil_symm 0/1/2),
                           coil_type 3 = background field (central current I along
                           +z -> B_phi = mu0 I / (2 pi R), uniform Bz).
  * simson_coils           SIMSON (simsopt JSON) coil lists (private reader:
                           Coil, CurveXYZFourier, RotatedCurve, Current, ScaledCurrent,
                           BiotSavart).

All lengths m, currents A, moments A m^2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from torux.source import FourierCurves, Dipoles, ToroidalField, FLIP, MU0, rotation_z
from torux.io.focus_dipoles import expand_dipoles


# --------------------------------------------------------------------------- FOCUS coils
def _is_number(tok):
    try:
        float(tok.replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def read_focus_coils(path, nfp=1):
    """FOCUS coil file -> dict(curves=FourierCurves (full device), names, symm, RB, Bz, notes).

    Fourier: r_a(t) = sum_k [a_c(k) cos(k t) + a_s(k) sin(k t)], t in [0, 2 pi): with t = 2 pi s
    this is the SIMSOPT order [c0, s1, c1, ...] with c_k = a_c(k), s_k = a_s(k) (exact).
    coil_symm 1: copies rotated by 2 pi k / nfp; 2: plus stellarator copies (coefficients
    mapped by F = diag(1,-1,-1), current reversed -- simsopt coils_via_symmetries)."""
    with open(path) as fh:
        raw = fh.read().splitlines()
    # token stream of non-comment lines
    lines = [ln for ln in raw if ln.strip() and not ln.strip().startswith("#")]
    pos = 0

    def nxt():
        nonlocal pos
        ln = lines[pos]
        pos += 1
        return ln.split()

    ncoil = int(float(nxt()[0]))
    coefs, cur, names, symm = [], [], [], []
    RB, Bz = 0.0, 0.0
    notes = []
    for ic in range(ncoil):
        hdr = nxt()
        ctype = int(float(hdr[0]))
        if ctype == 1:
            # "coil_type coil_symm coil_name" or "coil_type coil_name"
            if len(hdr) >= 3 and _is_number(hdr[1]):
                csymm, cname = int(float(hdr[1])), " ".join(hdr[2:])
            else:
                csymm, cname = 0, " ".join(hdr[1:])
            row = nxt()
            I = float(row[1])
            NF = int(float(nxt()[0]))
            rows = []
            while len(rows) < 6 * (NF + 1):
                rows.extend(float(t.replace("D", "E")) for t in nxt())
            A = np.array(rows[:6 * (NF + 1)]).reshape(6, NF + 1)
            c = np.zeros((3, 2 * NF + 1))
            for a in range(3):
                cc, ss = A[2 * a], A[2 * a + 1]
                c[a, 0] = cc[0]
                c[a, 2::2] = cc[1:]
                c[a, 1::2] = ss[1:]
            coefs.append(c)
            cur.append(I)
            names.append(cname)
            symm.append(csymm)
        elif ctype == 3:
            cname = " ".join(hdr[1:])
            row = nxt()
            I3, B3 = float(row[1]), float(row[3])
            RB += MU0 * I3 / (2 * np.pi)
            Bz += B3
            notes.append(f"coil_type 3 '{cname}': central current {I3} A -> RB = mu0 I / 2pi; Bz = {B3} T")
        else:
            raise NotImplementedError(f"{path}: FOCUS coil_type {ctype}")
    # symmetry expansion
    K = max(c.shape[1] for c in coefs) if coefs else 1
    full_c, full_I, full_names = [], [], []
    for c, I, s, nm in zip(coefs, cur, symm, names):
        cpad = np.zeros((3, K))
        cpad[:, :c.shape[1]] = c
        if s == 0:
            full_c.append(cpad); full_I.append(I); full_names.append(nm)
            continue
        for k in range(nfp):
            Rk = rotation_z(2 * np.pi * k / nfp)
            full_c.append(Rk @ cpad); full_I.append(I); full_names.append(nm)
            if s == 2:
                full_c.append(Rk @ FLIP @ cpad); full_I.append(-I); full_names.append(nm + "*")
    curves = FourierCurves(np.array(full_c), np.array(full_I)) if full_c else None
    return {"curves": curves, "names": full_names, "symm": symm, "RB": RB, "Bz": Bz, "notes": notes,
            "n_listed": ncoil}


# --------------------------------------------------------------------------- SIMSON
def _simson_load(path):
    with open(path) as fh:
        return json.load(fh)


def _arr(v):
    if isinstance(v, dict) and v.get("@class") == "array":
        return np.asarray(v["data"], dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


class _Simson:
    def __init__(self, d):
        self.d = d
        self.objs = d["simsopt_objs"]

    def get(self, ref):
        if isinstance(ref, dict) and ref.get("$type") == "ref":
            return self.objs[ref["value"]]
        return ref

    def dofs(self, obj):
        return _arr(self.get(obj["dofs"])["x"])

    def current(self, ref):
        o = self.get(ref)
        cl = o["@class"]
        if cl == "Current":
            return float(self.dofs(o)[0])
        if cl == "ScaledCurrent":
            return float(o["scale"]) * self.current(o["current_to_scale"])
        raise NotImplementedError(f"current class {cl}")

    def curve(self, ref):
        """SIMSOPT-order coefficients (3, 2K+1) of a curve."""
        o = self.get(ref)
        cl = o["@class"]
        if cl == "CurveXYZFourier":
            K = int(o["order"])
            x = self.dofs(o)
            return x.reshape(3, 2 * K + 1)
        if cl == "RotatedCurve":
            c = self.curve(o["curve"])
            phi = float(o["phi"])
            M = rotation_z(phi)
            if o.get("flip", False):
                M = FLIP @ M
            return M @ c
        raise NotImplementedError(f"curve class {cl}")

    def coils(self, refs):
        coefs, cur = [], []
        for r in refs:
            o = self.get(r)
            assert o["@class"] == "Coil", o["@class"]
            coefs.append(self.curve(o["curve"]))
            cur.append(self.current(o["current"]))
        K = max(c.shape[1] for c in coefs)
        C = np.zeros((len(coefs), 3, K))
        for i, c in enumerate(coefs):
            C[i, :, :c.shape[1]] = c
        return FourierCurves(C, np.array(cur))


def simson_coils(path, key=None):
    """FourierCurves of a SIMSON file whose graph is a list of Coil refs, or a dict whose
    entry ``key`` (default: first BiotSavart found) references a BiotSavart."""
    d = _simson_load(path)
    S = _Simson(d)
    g = d["graph"]
    if isinstance(g, list):
        refs = g
        if refs and S.get(refs[0])["@class"] == "BiotSavart":
            refs = S.get(refs[0])["coils"]
    elif isinstance(g, dict):
        if key is None:
            key = next(k for k, v in g.items() if isinstance(v, dict) and v.get("$type") == "ref"
                       and S.get(v)["@class"] == "BiotSavart")
        bs = S.get(g[key])
        refs = bs["coils"]
    else:
        raise ValueError(f"{path}: unsupported graph")
    return S.coils(refs)


# --------------------------------------------------------------------------- PM tables
def read_pm_csv(path):
    """(pos (N,3), moment (N,3), cylindrical (mr, mp) (N,2)) of a fig*_moments.csv table."""
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0:3].copy(), a[:, 3:6].copy(), a[:, 6:8].copy()


def read_corners_csv(path):
    """(N, 8, 3) corners t1..t4, b1..b4; header optional."""
    with open(path) as fh:
        first = fh.readline()
    skip = 0 if _is_number(first.split(",")[0].strip()) else 1
    a = np.loadtxt(path, delimiter=",", skiprows=skip)
    return a.reshape(-1, 8, 3)


def blocks_to_dipoles(corners, moments, n=2):
    """Uniformly magnetised hexahedra (corners (N,8,3) in t1..t4,b1..b4 order, total moments
    (N,3)) -> n^3 sub-dipoles per block on the Gauss-Legendre nodes of the trilinear map.

    Sub-dipole moment = (m / V) |J| w, V = block volume from the same rule (exact for
    trilinear maps of degree <= 2n-1 per variable; exact for parallelepipeds at any n).
    Returns (Dipoles, record)."""
    C = np.asarray(corners, dtype=np.float64)
    m = np.asarray(moments, dtype=np.float64)
    t1, t2, t3, t4, b1, b2, b3, b4 = (C[:, i] for i in range(8))
    # trilinear map on (u,v,w) in [0,1]^3: bottom face b1(0,0) b2(1,0) b3(1,1) b4(0,1); top t_k above b_k
    X = np.stack([b1, b2, b3, b4, t1, t2, t3, t4], axis=1)          # (N, 8, 3)
    corner_uvw = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    g, wg = np.polynomial.legendre.leggauss(n)
    g = 0.5 * (g + 1)
    wg = 0.5 * wg
    U, Vv, W = np.meshgrid(g, g, g, indexing="ij")
    WW = (wg[:, None, None] * wg[None, :, None] * wg[None, None, :]).ravel()
    uvw = np.stack([U.ravel(), Vv.ravel(), W.ravel()], axis=1)       # (q, 3)

    def shape(uvw):
        u, v, w = uvw[:, 0:1], uvw[:, 1:2], uvw[:, 2:3]
        cu, cv, cw = corner_uvw[:, 0][None], corner_uvw[:, 1][None], corner_uvw[:, 2][None]
        fu = np.where(cu == 1, u, 1 - u); fv = np.where(cv == 1, v, 1 - v); fw = np.where(cw == 1, w, 1 - w)
        Nf = fu * fv * fw
        su = np.where(cu == 1, 1.0, -1.0); sv = np.where(cv == 1, 1.0, -1.0); sw = np.where(cw == 1, 1.0, -1.0)
        dNu = su * fv * fw; dNv = fu * sv * fw; dNw = fu * fv * sw
        return Nf, dNu, dNv, dNw

    Nf, dNu, dNv, dNw = shape(uvw)                                    # (q, 8)
    P = np.einsum("qk,nka->nqa", Nf, X)
    Ju = np.einsum("qk,nka->nqa", dNu, X)
    Jv = np.einsum("qk,nka->nqa", dNv, X)
    Jw = np.einsum("qk,nka->nqa", dNw, X)
    detJ = np.abs(np.einsum("nqa,nqa->nq", Ju, np.cross(Jv, Jw)))
    wq = detJ * WW[None, :]
    V = wq.sum(axis=1)
    msub = (m / V[:, None])[:, None, :] * wq[:, :, None]
    rec = {"blocks": int(len(C)), "nodes_per_block": int(n ** 3), "volume_min_m3": float(V.min()),
           "volume_max_m3": float(V.max())}
    return Dipoles(P.reshape(-1, 3), msub.reshape(-1, 3)), rec


def block_centroids(corners):
    return np.asarray(corners).mean(axis=1)


# --------------------------------------------------------------------------- GPMO (QUADCOIL)
def load_gpmo_json(path):
    """dict(tf=FourierCurves, m_history (n,3,K), R2_history, Bn_history)."""
    d = _simson_load(path)
    S = _Simson(d)
    g = d["graph"]
    tf = S.coils(S.get(g["bs"])["coils"])
    return {"tf": tf, "m_history": _arr(g["m_history"]), "R2_history": _arr(g["R2_history"]),
            "Bn_history": _arr(g["Bn_history"])}


def gpmo_choice(R2_history, fB_target=1.29e-8):
    """Index used by PM/GPMO_plotting.ipynb: argmin (R2_history - 1.29e-8)^2."""
    return int(np.argmin((np.asarray(R2_history) - fB_target) ** 2))


def expand_halfperiod_dipoles(pos, m, nfp):
    """Half-period dipoles with stellarator symmetry (FOCUS symmetry 2) -> full device."""
    return expand_dipoles(pos, m, np.full(len(pos), 2), nfp)


# --------------------------------------------------------------------------- DESC coil sets (HDF5)
def _h5val(g, k):
    v = g[k][()]
    return v.decode() if isinstance(v, bytes) else v


def _fourier_basis_eval(modes_n, coeff, s, NFP=1):
    """DESC FourierSeries: n >= 0 -> cos(n NFP s), n < 0 -> sin(|n| NFP s)."""
    out = np.zeros_like(s)
    for n, c in zip(modes_n, coeff):
        out = out + (c * np.cos(n * NFP * s) if n >= 0 else c * np.sin(-n * NFP * s))
    return out


def _rodrigues(axis, angle):
    axis = np.asarray(axis, float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-300 or abs(angle) < 1e-300:
        return np.eye(3)
    k = axis / nrm
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def desc_curve_points(g, s):
    """Positions (len(s), 3) of a DESC coil group g (FourierPlanarCoil / FourierRZCoil /
    FourierXYZCoil) at parameter s in [0, 2 pi), including _rotmat and _shift
    (x -> rotmat @ x + shift)."""
    cls = _h5val(g, "__class__")
    if cls.endswith("FourierPlanarCoil"):
        if _h5val(g, "_basis") != "xyz":
            raise NotImplementedError("planar coil in rpz basis")
        modes = g["_r_basis/_modes"][()][:, 2]
        r = _fourier_basis_eval(modes, g["_r_n"][()], s, 1)
        X = np.stack([r * np.cos(s), r * np.sin(s), np.zeros_like(s)], 1)
        nrm = np.asarray(g["_normal"][()], float)
        nrm = nrm / np.linalg.norm(nrm)
        Zax = np.array([0.0, 0.0, 1.0])
        A = _rodrigues(np.cross(Zax, nrm), np.arccos(np.clip(nrm @ Zax, -1, 1)))
        X = X @ A.T + np.asarray(g["_center"][()], float)
    elif cls.endswith("FourierRZCoil"):
        NFP = int(g["_NFP"][()])
        R = _fourier_basis_eval(g["_R_basis/_modes"][()][:, 2], g["_R_n"][()], s, NFP)
        Z = _fourier_basis_eval(g["_Z_basis/_modes"][()][:, 2], g["_Z_n"][()], s, NFP)
        X = np.stack([R * np.cos(s), R * np.sin(s), Z], 1)
    elif cls.endswith("FourierXYZCoil"):
        X = np.stack([_fourier_basis_eval(g[f"_{a}_basis/_modes"][()][:, 2], g[f"_{a}_n"][()], s, 1)
                      for a in ("X", "Y", "Z")], 1)
    else:
        raise NotImplementedError(cls)
    rot = np.asarray(g["_rotmat"][()], float).reshape(3, 3)
    sh = np.asarray(g["_shift"][()], float)
    return X @ rot.T + sh


def desc_curve_order(g):
    cls = _h5val(g, "__class__")
    if cls.endswith("FourierPlanarCoil"):
        return int(np.abs(g["_r_basis/_modes"][()][:, 2]).max()) + 1
    if cls.endswith("FourierRZCoil"):
        return int(np.abs(g["_R_basis/_modes"][()][:, 2]).max()) * int(g["_NFP"][()]) + 1
    return int(np.abs(g["_X_basis/_modes"][()][:, 2]).max())


def samples_to_coef(X):
    """SIMSOPT-order coefficients (3, 2K+1) of a trigonometric polynomial sampled at
    t_j = j / M (X (M, 3)), K = (M - 1) // 2 (exact when the degree is < M / 2)."""
    M = len(X)
    F = np.fft.rfft(X, axis=0) / M
    K = (M - 1) // 2
    c = np.zeros((3, 2 * K + 1))
    c[:, 0] = F[0].real
    c[:, 2::2] = 2 * F[1:K + 1].real.T
    c[:, 1::2] = -2 * F[1:K + 1].imag.T
    return c


def desc_coilset(path):
    """All coils of a DESC (Mixed)CoilSet HDF5 file as FourierCurves (full set: CoilSet NFP
    rotations and sym reflections applied, DESC _compute_position / from_symmetry), with a
    list of group labels."""
    import h5py
    f = h5py.File(path, "r")
    coefs, cur, labels = [], [], []

    def walk(g, label, nfp_parent=1, sym_parent=False):
        cls = _h5val(g, "__class__")
        if cls.endswith("CoilSet") or cls.endswith("MixedCoilSet"):
            NFP = int(g["_NFP"][()]) if "_NFP" in g else 1
            sym = bool(g["_sym"][()]) if "_sym" in g else False
            kids = g["_coils"]
            for k in sorted((k for k in kids.keys() if k.isdigit()), key=int):
                walk(kids[k], f"{label}/{k}", NFP, sym)
            return
        K = desc_curve_order(g)
        M = 2 * K + 8
        s = 2 * np.pi * np.arange(M) / M
        X = desc_curve_points(g, s)
        c = samples_to_coef(X)
        I = float(g["_current"][()])
        base = [(c, I)]
        if sym_parent:
            nvec = np.array([-np.sin(np.pi / nfp_parent), np.cos(np.pi / nfp_parent), 0.0])
            Rn = np.eye(3) - 2 * np.outer(nvec, nvec)
            Rz = np.diag([1.0, 1.0, -1.0])
            base.append((Rz @ Rn @ c, -I))
        for kk in range(nfp_parent):
            Rk = rotation_z(2 * np.pi * kk / nfp_parent)
            for cb, Ib in base:
                coefs.append(Rk @ cb); cur.append(Ib); labels.append(label)

    walk(f, "")
    Kmax = max(c.shape[1] for c in coefs)
    C = np.zeros((len(coefs), 3, Kmax))
    for i, c in enumerate(coefs):
        C[i, :, :c.shape[1]] = c
    return FourierCurves(C, np.array(cur)), labels
