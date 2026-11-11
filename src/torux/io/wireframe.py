"""Wireframe current distributions (Hammond 2025, doi:10.34770/6pbm-9981).

CSV columns: p1x, p1y, p1z, p1phi, p1theta,
p2x, p2y, p2z, p2phi, p2theta, current, orientation, constrained.  Positive
current flows from p1 to p2 (m, A; angles in degrees).  ``*_fulltorus.csv``
lists every segment of the torus, ``*_halfperiod.csv`` one half period.

Loader choice: the Source is built from the full-torus table (complete as
published).  The half-period table is checked against it: expanding it by
the field period (nfp = 2, ``NFP`` of Supporting_Files/input.LandremanPaul2021_QA)
and stellarator symmetry (segment a->b, current I  ->  F a -> F b, current -I)
must give the full-torus set (``check_halfperiod``).  Segments with zero
current are dropped from the Source (they carry no field); their count is kept.

External TF coils: the multi-step GSCO solution (Scripts/gsco_multistep.py)
is computed with the wireframe *plus* 3 planar TF coils per half period that
are not part of the CSV: ``create_equally_spaced_curves(3, nfp=2, stellsym=True,
R0=1.0, R1=0.85)`` (order 6; formats/simsopt/curve.py) with currents
``-pol_cur / (2 * 3 * nfp)``, ``pol_cur = -2 pi R00 B_axis / mu0``,
R00 = RBC(0,0) of the QA input, B_axis = 1 T, expanded with
``coils_via_symmetries`` (rotation then flip, flipped copies with reversed
current).  ``external_tf_coils`` rebuilds them exactly (FourierCurves).
"""
from __future__ import annotations

import re

import numpy as np

from torux.source import Segments, FourierCurves, FLIP, MU0, rotation_z

COLS = ["p1x", "p1y", "p1z", "p1phi", "p1theta", "p2x", "p2y", "p2z", "p2phi", "p2theta",
        "current", "orientation", "constrained"]


def read_wireframe_csv(path):
    with open(path) as fh:
        head = fh.readline().strip().split(",")
    assert head[:13] == COLS, head
    a = np.loadtxt(path, delimiter=",", skiprows=1).reshape(-1, 13)
    return {"a": a[:, 0:3].copy(), "b": a[:, 5:8].copy(), "phi1_deg": a[:, 3], "theta1_deg": a[:, 4],
            "phi2_deg": a[:, 8], "theta2_deg": a[:, 9], "current": a[:, 10].copy(),
            "orientation": a[:, 11].astype(int), "constrained": a[:, 12].astype(int)}


def wireframe_segments(tab, drop_zero=True):
    I = tab["current"]
    keep = I != 0 if drop_zero else np.ones(len(I), bool)
    return Segments(tab["a"][keep], tab["b"][keep], I[keep]), {
        "segments_listed": int(len(I)), "segments_nonzero": int((I != 0).sum()),
        "constrained": int(tab["constrained"].sum()), "total_abs_I_A": float(np.abs(I).sum())}


def expand_halfperiod(tab, nfp):
    """Full torus segments (a, b, I) from a half-period table (rotation, then stellarator image
    F a -> F b with current -I)."""
    A, B, I = [], [], []
    for k in range(nfp):
        R = rotation_z(2 * np.pi * k / nfp)
        A.append(tab["a"] @ R.T); B.append(tab["b"] @ R.T); I.append(tab["current"])
        A.append(tab["a"] @ FLIP.T @ R.T); B.append(tab["b"] @ FLIP.T @ R.T); I.append(-tab["current"])
    return np.concatenate(A), np.concatenate(B), np.concatenate(I)


def _canon(a, b, I):
    """Orientation-free canonical form: (lexicographically smaller endpoint first, signed current)."""
    key_a = np.round(a, 9)
    key_b = np.round(b, 9)
    s = np.array([tuple(x) > tuple(y) for x, y in zip(key_a, key_b)])
    a2 = np.where(s[:, None], b, a)
    b2 = np.where(s[:, None], a, b)
    I2 = np.where(s, -I, I)
    return a2, b2, I2


def check_halfperiod(half, full, nfp, tol=1e-9):
    """Max deviation between the symmetry-expanded half period and the full-torus table, as
    unordered sets of directed segments (zero-current segments included).
    Returns dict(n_expanded, n_full, max_pos_dev, max_current_dev_rel, matched)."""
    from scipy.spatial import cKDTree
    A, B, I = expand_halfperiod(half, nfp)
    a1, b1, I1 = _canon(A, B, I)
    a2, b2, I2 = _canon(full["a"], full["b"], full["current"])
    k1 = np.concatenate([a1, b1], axis=1)
    k2 = np.concatenate([a2, b2], axis=1)
    tree = cKDTree(k2)
    d, idx = tree.query(k1)
    scale = max(np.abs(I2).max(), 1e-300)
    return {"n_expanded": int(len(A)), "n_full": int(len(full["a"])), "max_pos_dev_m": float(d.max()),
            "unique_matches": int(len(np.unique(idx))),
            "max_current_dev_rel": float(np.abs(I1 - I2[idx]).max() / scale),
            "matched": bool(d.max() < tol and len(np.unique(idx)) == len(A) == len(full["a"]))}


def read_vmec_rbc00(path):
    txt = open(path).read()
    m = re.search(r"RBC\(\s*0\s*,\s*0\s*\)\s*=\s*([-+0-9.eEdD]+)", txt, re.I)
    nfp = re.search(r"\bNFP\s*=\s*(\d+)", txt, re.I)
    return float(m.group(1).replace("D", "E").replace("d", "e")), int(nfp.group(1))


def external_tf_coils(R00, nfp=2, n_hp=3, R0=1.0, R1=0.85, order=6, field_on_axis=1.0):
    """The 3-per-half-period planar TF coils of the record's multi-step solution (see module doc)."""
    pol_cur = -2.0 * np.pi * R00 * field_on_axis / MU0
    I0 = -pol_cur / (2 * n_hp * nfp)
    base = []
    for i in range(n_hp):
        ang = (i + 0.5) * (2 * np.pi) / ((1 + 1) * nfp * n_hp)
        c = np.zeros((3, 2 * order + 1))
        c[0, 0] = np.cos(ang) * R0
        c[0, 2] = np.cos(ang) * R1        # xc(1)
        c[1, 0] = np.sin(ang) * R0
        c[1, 2] = np.sin(ang) * R1        # yc(1)
        c[2, 1] = -R1                     # zs(1)
        base.append(c)
    C, I = [], []
    for k in range(nfp):
        for flip in (False, True):
            M = rotation_z(2 * np.pi * k / nfp)
            if flip:
                M = FLIP @ M
            for c in base:
                C.append(M @ c)
                I.append(-I0 if flip else I0)
    return FourierCurves(np.array(C), np.array(I))
