"""Fourier-filament JSON -> Source(FourierCurves).

Format: ``nfp``, ``stellarator_symmetric``,
``coils`` = complete list (every period and symmetric copy) of
{xc, xs, yc, ys, zc, zs, current_A}: r_a(t) = ac[0] + sum_{k>=1} (ac[k] cos 2 pi k t +
as[k] sin 2 pi k t), t in [0, 1), m; current_A in A along increasing t.  Conversion to
the canonical [c0, s1, c1, ...] order is a reordering (as[0] is ignored; it is 0 in every
file -- checked by ``read_fourier_json(strict=True)``).  A file may also carry
``requested_toroidal_flux_Wb``, the request, and ``database_metadata``.
"""
from __future__ import annotations

import json

import numpy as np

from torux.source import Source
from torux.io._stellcommon import apply_verified_symmetry, data_path, fourier_part, rel_path


def coil_coef(c, strict=True):
    K = len(c["xc"]) - 1
    out = np.zeros((3, 2 * K + 1))
    for a, ax in enumerate("xyz"):
        cc = np.asarray(c[ax + "c"], dtype=np.float64)
        ss = np.asarray(c[ax + "s"], dtype=np.float64)
        if strict and len(ss) and ss[0] != 0.0:
            raise ValueError(f"{ax}s[0] = {ss[0]} != 0")
        if len(ss) != len(cc):
            raise ValueError("xs/xc length mismatch")
        out[a, 0] = cc[0]
        out[a, 2::2] = cc[1:]
        out[a, 1::2] = ss[1:]
    return out


def read_fourier_json(path, strict=True):
    with open(data_path(path)) as fh:
        d = json.load(fh)
    coefs = [coil_coef(c, strict) for c in d["coils"]]
    currents = [float(c["current_A"]) for c in d["coils"]]
    return d, coefs, currents


def load_fourier_json(path, source_id=None, check_symmetry=True, family="", request_from_file=False,
                      request_provenance=""):
    d, coefs, currents = read_fourier_json(path)
    if source_id is None:
        source_id = "fourier_json/" + rel_path(data_path(path))
    part = fourier_part(coefs, currents)
    meta = {k: d[k] for k in ("nfp", "stellarator_symmetric", "family", "source", "id", "database_metadata",
                             "source_coilset_id", "source_boundary_id", "conversion_check") if k in d}
    src = Source(id=source_id, parts=[part], family=family or d.get("family", ""),
                 provenance={"files": [rel_path(data_path(path))], "loader": "torux.io.fourier_json:load_fourier_json",
                             "format": "Fourier-filament JSON"}, meta=meta)
    if "requested_toroidal_flux_Wb" in d:
        src.meta["requested_toroidal_flux_Wb"] = float(d["requested_toroidal_flux_Wb"])
        if request_from_file:
            src.request = {"flux_Wb": float(d["requested_toroidal_flux_Wb"]),
                           "provenance": request_provenance or f"{rel_path(data_path(path))}: requested_toroidal_flux_Wb"}
    if check_symmetry:
        apply_verified_symmetry(src, [int(d.get("nfp", 1))], bool(d.get("stellarator_symmetric", False)))
    return src
