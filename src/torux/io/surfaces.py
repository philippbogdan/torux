"""Evaluable plasma and optimisation surfaces.

SurfaceXYZTensorFourier: coefficient arrays x, y, z of shape (2 mpol + 1, 2 ntor + 1) with
    basis
        w_i(theta) = 1, cos(theta) .. cos(mpol theta), sin(theta) .. sin(mpol theta)   (i = 0..2mpol)
        v_j(phi)   = 1, cos(nfp phi) .. cos(ntor nfp phi), sin(nfp phi) .. sin(ntor nfp phi)
        xhat = sum x_ij w_i v_j,  yhat likewise,  z = sum z_ij w_i v_j
        x = xhat cos(phi) - yhat sin(phi),  y = xhat sin(phi) + yhat cos(phi)
    with phi = 2 pi quadpoint_phi, theta = 2 pi quadpoint_theta (fractions of a full turn).
    DOF order: x(m, n) for m = 0..2mpol, n = 0..2ntor (row-major), skipping excluded entries,
    then y, then z.  Under stellsym the excluded entries are (header ``skip``):
        x: (n <= ntor and m > mpol) or (n > ntor and m <= mpol)
        y, z: (n <= ntor and m <= mpol) or (n > ntor and m > mpol)
    ``clamped_dims[d]`` multiplies the basis functions with n <= ntor and m <= mpol of
    coordinate d by sin^2(nfp phi / 2) + sin^2(theta / 2) (header ``bc_enforcer``).
VMEC boundary (input namelist RBC/ZBS[/RBS/ZBC], wout rmnc/zmns[/rmns/zmnc] last surface):
        R = sum RBC(n, m) cos(m theta - n nfp phi) + RBS sin(...),  Z = sum ZBS sin(...) + ZBC cos(...)
FOCUS boundary (.boundary/.plasma): header N_modes N_fp N_bn, rows n m Rbc Rbs Zbc Zbs with
        R = sum Rbc cos(m theta - n N_fp phi) + Rbs sin(...), Z = sum Zbs sin(...) + Zbc cos(...)
All surfaces expose ``gamma(phi, theta)`` (arrays of fractions of a full turn, broadcast) and
``gamma_and_derivs`` returning r, dr/dphi, dr/dtheta (derivatives w.r.t. the fractions).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from torux.io._stellcommon import data_path, rel_path

TWO_PI = 2 * np.pi


# --------------------------------------------------------------------------- SurfaceXYZTensorFourier
class SurfaceXYZTensorFourier:
    def __init__(self, nfp, stellsym, mpol, ntor, dofs=None, clamped_dims=(False, False, False),
                 quadpoints_phi=None, quadpoints_theta=None, name=""):
        self.nfp, self.stellsym, self.mpol, self.ntor = int(nfp), bool(stellsym), int(mpol), int(ntor)
        self.clamped_dims = tuple(bool(c) for c in clamped_dims)
        self.quadpoints_phi = None if quadpoints_phi is None else np.asarray(quadpoints_phi, float)
        self.quadpoints_theta = None if quadpoints_theta is None else np.asarray(quadpoints_theta, float)
        self.name = name
        shape = (2 * self.mpol + 1, 2 * self.ntor + 1)
        self.x = np.zeros(shape)
        self.y = np.zeros(shape)
        self.z = np.zeros(shape)
        if dofs is not None:
            self.set_dofs(dofs)

    # ---- DOF layout (header set_dofs_impl / skip)
    def skip(self, dim, m, n):
        if not self.stellsym:
            return False
        if dim == 0:
            return (n <= self.ntor and m > self.mpol) or (n > self.ntor and m <= self.mpol)
        return (n <= self.ntor and m <= self.mpol) or (n > self.ntor and m > self.mpol)

    def num_dofs(self):
        if self.stellsym:
            mp, nt = self.mpol, self.ntor
            return (nt + 1) * (mp + 1) + nt * mp + 2 * (nt + 1) * mp + 2 * nt * (mp + 1)
        return 3 * (2 * self.mpol + 1) * (2 * self.ntor + 1)

    def set_dofs(self, dofs):
        dofs = np.asarray(dofs, dtype=np.float64)
        if len(dofs) != self.num_dofs():
            raise ValueError(f"expected {self.num_dofs()} dofs, got {len(dofs)}")
        k = 0
        for dim, arr in enumerate((self.x, self.y, self.z)):
            for m in range(2 * self.mpol + 1):
                for n in range(2 * self.ntor + 1):
                    if self.skip(dim, m, n):
                        continue
                    arr[m, n] = dofs[k]
                    k += 1

    def get_dofs(self):
        out = []
        for dim, arr in enumerate((self.x, self.y, self.z)):
            for m in range(2 * self.mpol + 1):
                for n in range(2 * self.ntor + 1):
                    if not self.skip(dim, m, n):
                        out.append(arr[m, n])
        return np.asarray(out)

    # ---- basis
    def _basis(self, phi, theta):
        """Basis values and derivatives, phi, theta in radians (flattened arrays)."""
        mp, nt, nfp = self.mpol, self.ntor, self.nfp
        n = np.arange(2 * nt + 1)
        m = np.arange(2 * mp + 1)
        nn = np.where(n <= nt, n, n - nt) * nfp
        mm = np.where(m <= mp, m, m - mp)
        cosn = n <= nt
        cosm = m <= mp
        ap = np.outer(phi, nn)
        at = np.outer(theta, mm)
        cp, sp = np.cos(ap), np.sin(ap)
        ct, st = np.cos(at), np.sin(at)
        v = np.where(cosn, cp, sp)
        dv = np.where(cosn, -nn * sp, nn * cp)
        w = np.where(cosm, ct, st)
        dw = np.where(cosm, -mm * st, mm * ct)
        return v, dv, w, dw

    def _field(self, coeff, dim, v, dv, w, dw, phi, theta):
        """sum_ij c_ij w_i v_j (with optional bc enforcer) and its phi / theta derivatives."""
        # (w C) row-dotted with v: a BLAS product, not einsum's triple loop
        wc = w @ coeff
        f = np.sum(wc * v, axis=1)
        fp = np.sum(wc * dv, axis=1)
        ft = np.sum((dw @ coeff) * v, axis=1)
        if self.clamped_dims[dim]:
            mp, nt = self.mpol, self.ntor
            cc = np.zeros_like(coeff)
            cc[:mp + 1, :nt + 1] = coeff[:mp + 1, :nt + 1]
            wcc = w @ cc
            g = np.sum(wcc * v, axis=1)
            gp = np.sum(wcc * dv, axis=1)
            gt = np.sum((dw @ cc) * v, axis=1)
            e = np.sin(self.nfp * phi / 2) ** 2 + np.sin(theta / 2) ** 2
            ep = self.nfp * np.cos(self.nfp * phi / 2) * np.sin(self.nfp * phi / 2)
            et = np.cos(theta / 2) * np.sin(theta / 2)
            f = f - g + g * e
            fp = fp - gp + gp * e + g * ep
            ft = ft - gt + gt * e + g * et
        return f, fp, ft

    def gamma_and_derivs(self, phi, theta):
        """r, dr/dphi, dr/dtheta (derivatives w.r.t. the fractions phi, theta in [0,1))."""
        phi, theta = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
        shp = phi.shape
        P = TWO_PI * phi.ravel()
        T = TWO_PI * theta.ravel()
        v, dv, w, dw = self._basis(P, T)
        xh, xhp, xht = self._field(self.x, 0, v, dv, w, dw, P, T)
        yh, yhp, yht = self._field(self.y, 1, v, dv, w, dw, P, T)
        z, zp, zt = self._field(self.z, 2, v, dv, w, dw, P, T)
        c, s = np.cos(P), np.sin(P)
        X = xh * c - yh * s
        Y = xh * s + yh * c
        Xp = xhp * c - yhp * s - xh * s - yh * c
        Yp = xhp * s + yhp * c + xh * c - yh * s
        Xt = xht * c - yht * s
        Yt = xht * s + yht * c
        r = np.stack([X, Y, z], axis=-1).reshape(shp + (3,))
        rp = TWO_PI * np.stack([Xp, Yp, zp], axis=-1).reshape(shp + (3,))
        rt = TWO_PI * np.stack([Xt, Yt, zt], axis=-1).reshape(shp + (3,))
        return r, rp, rt

    def gamma(self, phi, theta):
        return self.gamma_and_derivs(phi, theta)[0]

    @classmethod
    def from_gson(cls, obj, dofs, name=""):
        return cls(obj["nfp"], obj["stellsym"], obj["mpol"], obj["ntor"], dofs,
                   tuple(obj.get("clamped_dims", (False, False, False))),
                   _arr(obj.get("quadpoints_phi")), _arr(obj.get("quadpoints_theta")), name)


def _arr(v):
    if isinstance(v, dict) and v.get("@class") == "array":
        return np.asarray(v["data"])
    return v


# --------------------------------------------------------------------------- generic geometry
def surface_grid(surf, nphi=64, ntheta=64, full_torus=True):
    nfp = getattr(surf, "nfp", 1)
    span = 1.0 if full_torus else 1.0 / nfp
    ph = np.arange(nphi) / nphi * span
    th = np.arange(ntheta) / ntheta
    P, T = np.meshgrid(ph, th, indexing="ij")
    r, rp, rt = surf.gamma_and_derivs(P, T)
    return r, rp, rt, span


def surface_normals(surf, nphi=64, ntheta=64, full_torus=True):
    """Points (nphi, ntheta, 3) and unnormalised normals dr/dphi x dr/dtheta (SIMSOPT orientation)."""
    r, rp, rt, _ = surface_grid(surf, nphi, ntheta, full_torus)
    return r, np.cross(rp, rt)


def surface_properties(surf, nphi=128, ntheta=128):
    """Volume (divergence theorem), area, major radius (R averaged over area), minor radius
    sqrt(V / (2 pi^2 R)), and R, Z extents -- on the full torus."""
    r, rp, rt, span = surface_grid(surf, nphi, ntheta, True)
    N = np.cross(rp, rt)
    dA = 1.0 / (nphi * ntheta)
    area = float(np.sum(np.linalg.norm(N, axis=-1)) * dA)
    vol = float(abs(np.sum(np.einsum("...a,...a->...", r, N)) * dA / 3.0))
    R = np.hypot(r[..., 0], r[..., 1])
    Rmaj = float(np.sum(R * np.linalg.norm(N, axis=-1)) * dA / area)
    a = float(np.sqrt(vol / (2 * np.pi ** 2 * Rmaj)))
    return {"volume": vol, "area": area, "major_radius": Rmaj, "minor_radius": a,
            "aspect_ratio": Rmaj / a, "Rmin": float(R.min()), "Rmax": float(R.max()),
            "Zmin": float(r[..., 2].min()), "Zmax": float(r[..., 2].max())}


# --------------------------------------------------------------------------- R/Z Fourier boundaries
@dataclass
class RZBoundary:
    nfp: int
    m: np.ndarray            # (M,)
    n: np.ndarray            # (M,) toroidal mode numbers *without* the nfp factor
    rc: np.ndarray           # cos coefficient of R
    zs: np.ndarray           # sin coefficient of Z
    rs: np.ndarray = None
    zc: np.ndarray = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.m = np.asarray(self.m, float)
        self.n = np.asarray(self.n, float)
        self.rc = np.asarray(self.rc, float)
        self.zs = np.asarray(self.zs, float)
        self.rs = np.zeros_like(self.rc) if self.rs is None else np.asarray(self.rs, float)
        self.zc = np.zeros_like(self.rc) if self.zc is None else np.asarray(self.zc, float)

    @property
    def stellsym(self):
        return not (np.any(self.rs) or np.any(self.zc))

    def gamma_and_derivs(self, phi, theta):
        phi, theta = np.broadcast_arrays(np.asarray(phi, float), np.asarray(theta, float))
        P = TWO_PI * phi[..., None]
        T = TWO_PI * theta[..., None]
        ang = self.m * T - self.n * self.nfp * P
        c, s = np.cos(ang), np.sin(ang)
        R = np.sum(self.rc * c + self.rs * s, axis=-1)
        Z = np.sum(self.zs * s + self.zc * c, axis=-1)
        dR_da = -self.rc * s + self.rs * c
        dZ_da = self.zs * c - self.zc * s
        Rt = np.sum(dR_da * self.m, axis=-1) * TWO_PI
        Zt = np.sum(dZ_da * self.m, axis=-1) * TWO_PI
        Rp = np.sum(dR_da * (-self.n * self.nfp), axis=-1) * TWO_PI
        Zp = np.sum(dZ_da * (-self.n * self.nfp), axis=-1) * TWO_PI
        ph = TWO_PI * phi
        cp, sp = np.cos(ph), np.sin(ph)
        r = np.stack([R * cp, R * sp, Z], axis=-1)
        rp = np.stack([Rp * cp - R * sp * TWO_PI, Rp * sp + R * cp * TWO_PI, Zp], axis=-1)
        rt = np.stack([Rt * cp, Rt * sp, Zt], axis=-1)
        return r, rp, rt

    def gamma(self, phi, theta):
        return self.gamma_and_derivs(phi, theta)[0]


# --------------------------------------------------------------------------- Fortran namelists
_FNUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"


def _fnum(s):
    return float(s.replace("D", "E").replace("d", "e"))


def read_namelist_text(path):
    txt = open(data_path(path), errors="replace").read()
    # strip comments ('!' to end of line, outside quotes -- quotes are rare in the numeric part)
    lines = []
    for ln in txt.splitlines():
        q = False
        out = []
        for ch in ln:
            if ch in "'\"":
                q = not q
            if ch == "!" and not q:
                break
            out.append(ch)
        lines.append("".join(out))
    return "\n".join(lines)


def parse_vmec_input(path):
    """Scalars, arrays and boundary modes of a VMEC &INDATA namelist (tolerant regex parser)."""
    txt = read_namelist_text(path)
    m0 = re.search(r"&\s*indata", txt, re.I)
    body = txt[m0.end():] if m0 else txt
    m1 = re.search(r"^\s*/\s*$|&\s*end", body, re.M | re.I)
    if m1:
        body = body[:m1.start()]
    out = {"_modes": {}}
    # indexed boundary coefficients NAME(n, m) = v
    for mm in re.finditer(r"\b(RBC|ZBS|RBS|ZBC)\s*\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)\s*=\s*(" + _FNUM + ")",
                          body, re.I):
        key = mm.group(1).upper()
        out["_modes"].setdefault(key, {})[(int(mm.group(2)), int(mm.group(3)))] = _fnum(mm.group(4))
    # plain assignments NAME = values (up to the next NAME= or NAME(...)=)
    toks = list(re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*=", body))
    for i, t in enumerate(toks):
        name = t.group(1).upper()
        end = toks[i + 1].start() if i + 1 < len(toks) else len(body)
        val = body[t.end():end].strip().rstrip(",").strip()
        if name in ("RBC", "ZBS", "RBS", "ZBC"):
            continue
        idx = t.group(2)
        if re.fullmatch(r"'[^']*'|\"[^\"]*\"", val):
            v = val[1:-1]
        elif re.fullmatch(r"\.?[tTfF][a-zA-Z]*\.?", val):
            v = val.strip(".").upper().startswith("T")
        else:
            nums = []
            for tok in re.split(r"[\s,]+", val):
                if not tok:
                    continue
                rep = re.fullmatch(r"(\d+)\*(" + _FNUM + ")", tok)
                if rep:
                    nums += [_fnum(rep.group(2))] * int(rep.group(1))
                elif re.fullmatch(_FNUM, tok):
                    nums.append(_fnum(tok))
                else:
                    nums = None
                    break
            if nums is None:
                v = val
            elif len(nums) == 1 and idx is None:
                v = nums[0]
            else:
                v = nums
        if idx is not None:
            out.setdefault(name + "()", {})[idx.strip("() ")] = v
        else:
            out[name] = v
    return out


def vmec_input_boundary(path):
    nl = parse_vmec_input(path)
    nfp = int(nl.get("NFP", 1))
    modes = nl["_modes"]
    keys = sorted(set().union(*[set(d) for d in modes.values()])) if modes else []
    n = np.array([k[0] for k in keys])
    m = np.array([k[1] for k in keys])
    g = lambda nm: np.array([modes.get(nm, {}).get(k, 0.0) for k in keys])
    meta = {k: nl[k] for k in ("PHIEDGE", "LASYM", "LFREEB", "MGRID_FILE", "EXTCUR", "CURTOR", "NFP",
                               "PRES_SCALE", "AM", "AC", "PMASS_TYPE", "PCURR_TYPE", "NCURR") if k in nl}
    if "EXTCUR()" in nl:
        ex = nl["EXTCUR()"]
        nmax = max(int(k) for k in ex)
        meta["EXTCUR"] = [float(ex.get(str(i), 0.0)) if not isinstance(ex.get(str(i), 0.0), list)
                          else ex[str(i)][0] for i in range(1, nmax + 1)]
    return RZBoundary(nfp, m, n, g("RBC"), g("ZBS"), g("RBS"), g("ZBC"), source=rel_path(data_path(path)), meta=meta)


def wout_boundary(path, surface=-1):
    """Last (or given) radial surface of a VMEC wout NetCDF file."""
    import netCDF4
    with netCDF4.Dataset(str(data_path(path))) as ds:
        nfp = int(ds["nfp"][...])
        xm = np.asarray(ds["xm"][:], float)
        xn = np.asarray(ds["xn"][:], float)
        rmnc = np.asarray(ds["rmnc"][:], float)[surface]
        zmns = np.asarray(ds["zmns"][:], float)[surface]
        rmns = np.asarray(ds["rmns"][:], float)[surface] if "rmns" in ds.variables else None
        zmnc = np.asarray(ds["zmnc"][:], float)[surface] if "zmnc" in ds.variables else None
        meta = {}
        for k in ("phi", "iotaf", "ctor", "betatotal", "lfreeb__logical__", "ier_flag", "Rmajor_p", "Aminor_p",
                  "volume_p", "lasym__logical__"):
            if k in ds.variables:
                v = np.asarray(ds[k][...])
                meta[k] = float(v[-1]) if v.ndim and k in ("phi", "iotaf") else (float(v) if v.ndim == 0 else v.tolist())
        meta["phiedge"] = meta.get("phi")
    return RZBoundary(nfp, xm, xn / nfp, rmnc, zmns, rmns, zmnc, source=rel_path(data_path(path)), meta=meta)


def focus_boundary(path):
    lines = [ln.strip() for ln in open(data_path(path), errors="replace") if ln.strip() and not ln.strip().startswith("#")]
    h = lines[0].split()
    nmodes, nfp, nbn = int(float(h[0])), int(float(h[1])), int(float(h[2]))
    rows = np.array([[_fnum(v) for v in ln.split()[:6]] for ln in lines[1:1 + nmodes]])
    bn = None
    if nbn:
        bn = np.array([[_fnum(v) for v in ln.split()[:4]] for ln in lines[1 + nmodes:1 + nmodes + nbn]])
    b = RZBoundary(nfp, rows[:, 1], rows[:, 0], rows[:, 2], rows[:, 5], rows[:, 3], rows[:, 4],
                   source=rel_path(data_path(path)), meta={"n_bn": nbn})
    b.bn = bn
    return b


# --------------------------------------------------------------------------- QUASR / SIMSON surfaces
def simson_surfaces(sim_or_path, names=None):
    """SurfaceXYZTensorFourier objects of a SIMSON file (graph order)."""
    from torux.io.simsopt_json import Simson, read_simson
    sim = sim_or_path if isinstance(sim_or_path, Simson) else read_simson(sim_or_path)
    out = []
    names = names or [n for n in sim.surfaces() if sim.objs[n]["@class"] == "SurfaceXYZTensorFourier"]
    for nm in names:
        o = sim.objs[nm]
        out.append(SurfaceXYZTensorFourier.from_gson(o, sim.dofs_x(o), nm))
    return out


def quasr_surfaces(path):
    """The optimisation surfaces of a QUASR download (graph[0] order; the last is the outermost)."""
    from torux.io.simsopt_json import read_simson
    sim = read_simson(path)
    names = [r["value"] for r in sim.graph[0]]
    return simson_surfaces(sim, names)


def pysurfaceopt_surfaces(path, nfp, mpol=10, ntor=10, stellsym=True):
    """Rows of a pysurfaceopt surface_dofs_*.txt file as SurfaceXYZTensorFourier objects
    (helpers.py: SurfaceXYZTensorFourier(mpol=10, ntor=10, stellsym=True, nfp=nfp); one row per
    surface; nfp 2 for data_stageII, 3 for data_ncsx; matching iota/G rows in the iotaG_* file)."""
    d = np.atleast_2d(np.loadtxt(data_path(path)))
    return [SurfaceXYZTensorFourier(nfp, stellsym, mpol, ntor, row, name=f"row{i}") for i, row in enumerate(d)]
