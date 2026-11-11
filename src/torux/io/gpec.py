"""GPEC coil geometry files (``*.dat`` of its coil library) -> PolylineCurves.

Format: header ``ncoil s nsec nw`` (coils in
the array, strands per coil, points per strand, winding multiplier), then
ncoil * s * nsec rows ``X Y Z`` (m), the array written out in full.  Each strand is the
closed polygon through its nsec points; when a strand's last point repeats its first
(|p_last - p_0| <= 1e-9 m, e.g. d3d_iu.dat) it is dropped (the closing edge is implied).
No currents are stored.

Current convention: every strand of a
coil carries nw * I_coil (the winding multiplier applied per conductor, its sign
reversing the direction), with I_coil = 1 A per coil ("unit" currents, one group per
coil).  For the files with s > 1 (kstar_ivcc*, nstx_tfef, d3d_tbm_*) the alternative
reading "nw * I_coil shared among the s strands" would scale the field by 1/s; this is
recorded in meta["current_convention"].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from torux.source import PolylineCurves


def read_gpec(path, close_tol=1e-9):
    lines = Path(path).read_text().splitlines()
    hdr = lines[0].split()
    ncoil, s, nsec = int(hdr[0]), int(hdr[1]), int(hdr[2])
    nw = float(hdr[3])
    X = np.array([[float(t) for t in ln.split()[:3]] for ln in lines[1:1 + ncoil * s * nsec]])
    assert X.shape == (ncoil * s * nsec, 3), (path, X.shape)
    X = X.reshape(ncoil, s, nsec, 3)
    loops, closed_dup = [], 0
    for i in range(ncoil):
        for j in range(s):
            P = X[i, j]
            if np.linalg.norm(P[-1] - P[0]) <= close_tol:
                P = P[:-1]
                closed_dup += 1
            loops.append(P)
    return {"ncoil": ncoil, "strands": s, "nsec": nsec, "nw": nw, "loops": loops,
            "repeated_end_point_strands": closed_dup}


def gpec_part(g, coil_currents=None):
    """PolylineCurves with strand current nw * I_coil (I_coil = 1 A unless given)."""
    loops, I, groups = [], [], []
    k = 0
    for i in range(g["ncoil"]):
        Ic = 1.0 if coil_currents is None else float(coil_currents[i])
        i0 = len(I)
        for j in range(g["strands"]):
            loops.append(g["loops"][k]); I.append(g["nw"] * Ic)
            k += 1
        groups.append({"name": f"coil{i + 1}", "part": 0, "start": i0, "stop": len(I)})
    start = np.zeros(len(loops) + 1, dtype=np.int64)
    start[1:] = np.cumsum([len(p) for p in loops])
    return PolylineCurves(np.concatenate(loops), start, np.array(I)), groups
