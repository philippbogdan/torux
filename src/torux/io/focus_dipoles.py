"""FOCUS / FAMUS dipole tables (``*.focus`` with coiltype 2) -> point dipoles.

Format:

  line 1   free text ("# Total number of coils, momentq")
  line 2   N  [momentq]                (momentq q defaults to 1)
  line 3   free text (column names)
  then N rows (comma separated)
           coiltype, symmetry, coilname, ox, oy, oz, Ic, M_0, pho, Lc, mp, mt [, op]

Moment (A m^2):  m = M_0 * pho**q * (cos mp sin mt, sin mp sin mt, cos mt);
``mp`` azimuth, ``mt`` polar angle (rad); negative pho reverses the moment
(for odd q; for even q the files here have no negative pho -- checked, an
error is raised otherwise).  ``Ic`` is an enable flag (rows with Ic == 0 are
not installed, e.g. diagnostic ports; simsopt drops them the same way).
``Lc`` is an optimisation flag and is ignored.

Symmetry code per row: 0 as is; 1 repeated over the nfp periods (rotation by
2 pi k / nfp, positions and moments); 2 repeated over the periods and by
stellarator symmetry, the stellarator image of a dipole at the phi = 0 plane
being position (x, -y, -z), moment (-mx, my, mz).  The full list is
ordered period by period: [R_k p (all rows), R_k F p (rows with symm 2)] for
k = 0..nfp-1.

nfp is not stored in a dipole table; the loaders take it from the paired
plasma surface (FOCUS ``.plasma``/``.boundary`` header ``N_modes N_fp N_bn``),
see ``read_focus_plasma_nfp``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from torux.source import Dipoles, FLIP, rotation_z

STELL_MOMENT = np.diag([-1.0, 1.0, 1.0])   # moment parity of the stellarator image at phi = 0


@dataclass
class FocusDipoleTable:
    path: str
    n_header: int               # dipole count stated in the header (line 2)
    momentq: float
    coiltype: np.ndarray
    symm: np.ndarray
    name: list
    pos: np.ndarray             # (N, 3)
    Ic: np.ndarray
    M0: np.ndarray
    pho: np.ndarray
    Lc: np.ndarray
    mp: np.ndarray
    mt: np.ndarray
    op: np.ndarray = None
    notes: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.Ic)

    def unit(self):
        return np.stack([np.cos(self.mp) * np.sin(self.mt), np.sin(self.mp) * np.sin(self.mt), np.cos(self.mt)], axis=1)

    def moments(self):
        q = self.momentq
        if np.any(self.pho < 0) and float(q) % 2 == 0:
            raise ValueError(f"{self.path}: negative pho with even momentq={q}: sign convention undefined")
        amp = self.M0 * np.power(self.pho, q)
        return amp[:, None] * self.unit()


def _num(tok):
    return float(tok.strip().replace("D", "E").replace("d", "e"))


def read_focus_dipoles(path):
    """Parse a FOCUS/FAMUS dipole table (all rows, nothing dropped)."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    hdr = lines[1].replace(",", " ").split()
    n_header = int(float(hdr[0]))
    q = float(hdr[1]) if len(hdr) > 1 else 1.0
    rows = []
    for ln in lines[3:]:
        if not ln.strip():
            continue
        tok = [t for t in ln.split(",")]
        if tok and tok[-1].strip() == "":
            tok = tok[:-1]
        if len(tok) < 12:
            raise ValueError(f"{path}: short dipole row {ln!r}")
        rows.append(tok)
    N = len(rows)
    ct = np.array([int(_num(r[0])) for r in rows])
    sy = np.array([int(_num(r[1])) for r in rows])
    name = [r[2].strip() for r in rows]
    num = np.array([[_num(r[i]) for i in (3, 4, 5, 6, 7, 8, 9, 10, 11)] for r in rows], dtype=np.float64).reshape(N, 9)
    op = None
    if N and len(rows[0]) > 12:
        try:
            op = np.array([_num(r[12]) for r in rows])
        except ValueError:
            op = None
    t = FocusDipoleTable(path=str(path), n_header=n_header, momentq=q, coiltype=ct, symm=sy, name=name,
                         pos=num[:, 0:3].copy(), Ic=num[:, 3].copy(), M0=num[:, 4].copy(), pho=num[:, 5].copy(),
                         Lc=num[:, 6].copy(), mp=num[:, 7].copy(), mt=num[:, 8].copy(), op=op)
    if N != n_header:
        t.notes.append(f"header states {n_header} rows, file has {N}")
    return t


def expand_dipoles(pos, m, symm, nfp):
    """Full device dipole list from per-row symmetry codes 0 / 1 / 2 (see module doc)."""
    pos = np.asarray(pos, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    symm = np.asarray(symm, dtype=np.int64)
    s0 = symm == 0
    s12 = symm >= 1
    s2 = symm == 2
    P, M = [pos[s0]], [m[s0]]
    for k in range(nfp):
        Rk = rotation_z(2 * np.pi * k / nfp)
        P.append(pos[s12] @ Rk.T)
        M.append(m[s12] @ Rk.T)
        if s2.any():
            P.append((pos[s2] @ FLIP.T) @ Rk.T)
            M.append((m[s2] @ STELL_MOMENT.T) @ Rk.T)
    return np.concatenate(P), np.concatenate(M)


def table_to_dipoles(t: FocusDipoleTable, nfp, drop_zero=True, enabled_only=True):
    """Dipoles part (full device) and a count record.

    enabled_only: drop rows with Ic == 0 (not installed).  drop_zero: drop rows with zero
    moment (they produce no field; counts kept in the record)."""
    m = t.moments()
    keep = np.ones(t.n, bool)
    if enabled_only:
        keep &= t.Ic != 0
    nz = np.linalg.norm(m, axis=1) > 0
    rec = {"rows": int(t.n), "rows_header": int(t.n_header), "momentq": t.momentq,
           "Ic1_rows": int((t.Ic != 0).sum()), "nonzero_rows": int(nz.sum()),
           "nonzero_enabled_rows": int((nz & (t.Ic != 0)).sum()),
           "symmetry_codes": {str(int(k)): int(v) for k, v in zip(*np.unique(t.symm, return_counts=True))},
           "nfp_expansion": int(nfp), "notes": list(t.notes)}
    if drop_zero:
        keep &= nz
    P, M = expand_dipoles(t.pos[keep], m[keep], t.symm[keep], nfp)
    rec["dipoles_full"] = int(len(P))
    rec["total_abs_moment_Am2"] = float(np.linalg.norm(M, axis=1).sum())
    return Dipoles(P, M), rec


def read_focus_plasma_nfp(path):
    """N_fp from the header of a FOCUS plasma/boundary file (line 2: N_modes N_fp N_bn)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    for ln in lines[:6]:
        tok = ln.replace(",", " ").split()
        if len(tok) >= 2:
            try:
                vals = [int(float(x)) for x in tok[:3]]
            except ValueError:
                continue
            return int(vals[1])
    raise ValueError(f"{path}: no N_fp header found")


def check_dipole_symmetry(dip: Dipoles, nfp, stellsym, tol=1e-12):
    """Exact set check: the expanded dipole set maps onto itself under the rotation by
    2 pi / nfp and (if stellsym) under (p, m) -> (F p, -F m) (field symmetry
    B(Fx) = -F B(x)).  Returns (ok_rot, ok_flip, max_dev)."""
    from scipy.spatial import cKDTree
    P, M = dip.pos, dip.moment
    scale_p = max(float(np.abs(P).max()), 1e-300)
    scale_m = max(float(np.abs(M).max()), 1e-300)
    tree = cKDTree(P)

    def matches(P2, M2):
        d, idx = tree.query(P2)
        dm = np.abs(M[idx] - M2).max() / scale_m if len(P2) else 0.0
        return float(max(d.max() / scale_p if len(P2) else 0.0, dm)), len(np.unique(idx)) == len(P2)

    out = []
    dev = 0.0
    if nfp > 1:
        R = rotation_z(2 * np.pi / nfp)
        d, uniq = matches(P @ R.T, M @ R.T)
        out.append(d <= tol and uniq)
        dev = max(dev, d)
    else:
        out.append(True)
    if stellsym:
        d, uniq = matches(P @ FLIP.T, -(M @ FLIP.T))
        out.append(d <= tol and uniq)
        dev = max(dev, d)
    else:
        out.append(True)
    return out[0], out[1], dev
