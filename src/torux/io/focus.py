"""FOCUS coil files (``*.focus`` with Fourier coils, coil_type 1) -> Source(FourierCurves).

Format: header ``# Total number of coils`` and the
count, then per coil: ``coil_type coil_symm coil_name``, ``Nseg current Ifree Length
Lfree target_length`` (A, m), ``NFcoil``, then six rows of NFcoil+1 coefficients in
the order xc, xs, yc, ys, zc, zs (m), with

    r_a(s) = sum_{k=0}^{NF} [ac_k cos(k s) + as_k sin(k s)],  s in [0, 2 pi).

Conversion to the canonical SIMSOPT order with t = s / (2 pi) is exact:
c0 = ac_0, s_k = as_k, c_k = ac_k  (as_0 is ignored; it multiplies sin 0 = 0).

``coil_symm``: 0 as is; 1 repeated over the nfp periods; 2 repeated over the periods
and by stellarator symmetry.  The file does not contain nfp: FOCUS takes it from
the plasma boundary (``.boundary`` header ``Nfp``); pass ``nfp`` explicitly.
Expansion used here: for k = 0..nfp-1, copies R(2 pi k/nfp) r(t) with current I
and, for coil_symm 2, F R(2 pi k/nfp) r(t) with current -I, F = diag(1, -1, -1).
The stellarator-symmetric copy F r(t) with -I is the same field source as FOCUS's
reversed-parameter image F r(-t) with +I; the symmetry of the result is verified
numerically (torux.io._stellcommon.verify_symmetry) and against HSX's MAKEGRID file in
the tests.

coil_type 2 (dipoles) and 3 (background field) are not handled here; FOCUS dipole
tables are read by torux.io.focus_dipoles.

Public API:
  read_focus_coils(path) -> list of FocusCoil
  focus_to_source(coils, nfp, source_id, ...) -> Source
  load_focus(path, nfp, source_id=None, ...) -> Source
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torux.source import FLIP, Source, rotation_z
from torux.io._stellcommon import apply_verified_symmetry, data_path, fourier_part, rel_path


@dataclass
class FocusCoil:
    coil_type: int
    symm: int
    name: str
    nseg: int
    current: float
    ifree: int
    length: float
    lfree: int
    target_length: float
    nf: int
    xc: np.ndarray
    xs: np.ndarray
    yc: np.ndarray
    ys: np.ndarray
    zc: np.ndarray
    zs: np.ndarray

    def simsopt_coef(self):
        """(3, 2 NF + 1) coefficients [c0, s1, c1, ..., sNF, cNF] for t = s / 2 pi."""
        out = np.zeros((3, 2 * self.nf + 1))
        for a, (c, s) in enumerate(((self.xc, self.xs), (self.yc, self.ys), (self.zc, self.zs))):
            out[a, 0] = c[0]
            out[a, 1::2] = s[1:]
            out[a, 2::2] = c[1:]
        return out


def _data_lines(path):
    out = []
    with open(data_path(path), "r", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def read_focus_coils(path):
    L = _data_lines(path)
    ncoils = int(float(L[0].split()[0]))
    coils = []
    i = 1
    for _ in range(ncoils):
        t = L[i].split()
        ctype, symm = int(t[0]), int(t[1])
        name = " ".join(t[2:])
        i += 1
        if ctype != 1:
            raise NotImplementedError(f"{path}: coil_type {ctype} is not a Fourier filament coil")
        t = L[i].split()
        nseg, cur, ifree, length, lfree = int(float(t[0])), float(t[1]), int(float(t[2])), float(t[3]), int(float(t[4]))
        tl = float(t[5]) if len(t) > 5 else float("nan")
        i += 1
        nf = int(float(L[i].split()[0]))
        i += 1
        rows = []
        for _r in range(6):
            vals = [float(v) for v in L[i].split()]
            i += 1
            while len(vals) < nf + 1:          # tolerate wrapped rows
                vals += [float(v) for v in L[i].split()]
                i += 1
            rows.append(np.array(vals[:nf + 1]))
        coils.append(FocusCoil(ctype, symm, name, nseg, cur, ifree, length, lfree, tl, nf, *rows))
    return coils


def expand_focus(coils, nfp):
    """Full list of (coef (3, 2K+1), current) after coil_symm expansion (SIMSOPT ordering)."""
    coefs, currents, labels = [], [], []
    for ci, c in enumerate(coils):
        base = c.simsopt_coef()
        if c.symm == 0:
            coefs.append(base)
            currents.append(c.current)
            labels.append((ci, 0, False))
            continue
        flips = (False, True) if c.symm == 2 else (False,)
        for k in range(nfp):
            Rk = rotation_z(2 * np.pi * k / nfp)
            for fl in flips:
                M = FLIP @ Rk if fl else Rk
                coefs.append(M @ base)
                currents.append(-c.current if fl else c.current)
                labels.append((ci, k, fl))
    return coefs, currents, labels


def focus_to_source(coils, nfp, source_id, files=(), check_symmetry=True, family="", nfp_candidates=None):
    coefs, currents, labels = expand_focus(coils, nfp)
    part = fourier_part(coefs, currents)
    symms = sorted({c.symm for c in coils})
    src = Source(id=source_id, parts=[part], family=family,
                 provenance={"files": [rel_path(f) for f in files], "loader": "torux.io.focus:load_focus",
                             "format": "FOCUS coils", "nfp_used_for_expansion": int(nfp)},
                 meta={"focus_coils": [{"name": c.name, "symm": c.symm, "current": c.current, "NFcoil": c.nf,
                                        "Nseg": c.nseg} for c in coils],
                       "coil_symm_values": symms,
                       "expansion": "k=0..nfp-1: R_k r (I); symm 2 also F R_k r (-I)"})
    if check_symmetry:
        stell_claim = all(c.symm == 2 for c in coils)
        cands = nfp_candidates if nfp_candidates is not None else [nfp]
        apply_verified_symmetry(src, cands, stell_claim)
    return src


def load_focus(path, nfp, source_id=None, **kw):
    coils = read_focus_coils(path)
    if source_id is None:
        source_id = "focus/" + rel_path(data_path(path))
    return focus_to_source(coils, nfp, source_id, files=[data_path(path)], **kw)


def focus_boundary_nfp(path):
    """Nfp from a FOCUS .boundary/.plasma header (``N_modes N_fp N_bn``)."""
    L = _data_lines(path)
    return int(float(L[0].split()[1]))
