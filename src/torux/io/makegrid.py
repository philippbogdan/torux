"""MAKEGRID coil files (``coils.*``) -> Source with a PolylineCurves part.

Format: header lines ``periods N``,
``begin filament``, ``mirror NIL|NUL`` (in any order, before the first number),
then rows ``x y z I`` (m, m, m, A).  A row with more than four tokens
(``x y z I group_id [group_name]``, I = 0 by convention) is the last point of the
current loop; the next row starts a new loop.  ``end`` (or end of file) stops.
The file lists every loop of the device; no symmetry is applied to it.  Current is
per loop and flows along the point order.

MAKEGRID semantics: the current given on a row flows on the straight segment from
that point to the next point of the same loop; the closing row carries no segment.
A loop is therefore the polygon p_0 -> p_1 -> ... -> p_{n-1} (the closing row).
Canonical form (torux.source.PolylineCurves): when the closing row repeats the first
point (|p_{n-1} - p_0| <= close_tol * max|p|) it is dropped and the closing edge is
implied; otherwise the closing row is kept and the implied edge p_{n-1} -> p_0 is
*added* -- this is reported in ``meta['open_loops']`` (no file here needs it).
If the current changes along a loop the loop is returned as Segments instead.

A loop of a single row with nonzero current (``x y z I group name``) is the MAKEGRID /
LIBSTELL circular-coil convention: a circle about the z axis of radius sqrt(x^2+y^2)
at height z carrying I along +phi.  It is returned in a Rings part.

Some files carry unit currents with the physical currents in an ``extcur`` table;
pass ``extcur={group_id: factor}`` (or a list indexed by group_id-1) to multiply.

Public API (also used by the magnets/tokamak loaders):
  read_makegrid(path, start_marker=None) -> MakegridFile
  makegrid_to_source(mg, source_id, extcur=None, nfp_candidates=None, ...) -> Source
  load_makegrid(path, source_id=None, ...) -> Source
  read_extcur(path) -> list of float
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from torux.source import PolylineCurves, Rings, Segments, Source
from torux.io._stellcommon import apply_verified_symmetry, data_path, polyline_part, rel_path

_NUM = re.compile(r"[dD]")


def _f(tok):
    return float(_NUM.sub("e", tok))


@dataclass
class MakegridLoop:
    points: np.ndarray          # (n, 3) every row of the loop including the closing row
    currents: np.ndarray        # (n,) current column (closing row usually 0)
    group: int
    name: str


@dataclass
class MakegridFile:
    path: str
    periods: int
    mirror: str
    loops: list = field(default_factory=list)
    header: list = field(default_factory=list)
    trailing_points: int = 0    # rows after the last closing row (not a loop)


def read_makegrid(path, start_marker=None):
    """Parse a MAKEGRID file.  ``start_marker``: ignore everything up to and including the
    first line containing this text (e.g. ``coils_dot_starts_below`` in V3FIT coils.test)."""
    path = data_path(path)
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    i0 = 0
    if start_marker is not None:
        for k, ln in enumerate(lines):
            if start_marker in ln:
                i0 = k + 1
                break
        else:
            raise ValueError(f"start marker {start_marker!r} not found in {path}")
    periods, mirror = 1, ""
    header = []
    loops = []
    pts, cur = [], []
    for ln in lines[i0:]:
        s = ln.split("!")[0].strip()
        if not s:
            continue
        tok = s.split()
        t0 = tok[0].lower()
        if t0 == "periods":
            periods = int(_f(tok[1]))
            header.append(ln.strip())
            continue
        if t0 == "begin":
            header.append(ln.strip())
            continue
        if t0 == "mirror":
            mirror = tok[1] if len(tok) > 1 else ""
            header.append(ln.strip())
            continue
        if t0 == "end":
            break
        if len(tok) < 4:
            raise ValueError(f"{path}: unexpected line {ln!r}")
        x, y, z, I = (_f(t) for t in tok[:4])
        pts.append((x, y, z))
        cur.append(I)
        if len(tok) > 4:
            gid = int(_f(tok[4]))
            name = " ".join(tok[5:])
            loops.append(MakegridLoop(np.array(pts, dtype=np.float64), np.array(cur, dtype=np.float64), gid, name))
            pts, cur = [], []
    return MakegridFile(str(path), periods, mirror, loops, header, trailing_points=len(pts))


def read_extcur(path):
    """EXTCUR values from an extcur file / VMEC namelist fragment (EXTCUR(i) = v or EXTCUR = v1, v2, ...)."""
    txt = open(data_path(path)).read()
    vals = {}
    for m in re.finditer(r"extcur\s*\(\s*(\d+)\s*\)\s*=\s*([-+0-9.eEdD]+)", txt, re.I):
        vals[int(m.group(1))] = _f(m.group(2))
    if vals:
        n = max(vals)
        return [vals.get(i, 0.0) for i in range(1, n + 1)]
    m = re.search(r"extcur\s*=\s*([^/&]*)", txt, re.I)
    if m:
        return [_f(t) for t in re.split(r"[\s,]+", m.group(1).strip()) if t and re.match(r"[-+0-9.]", t)]
    return []


def makegrid_to_source(mg, source_id, extcur=None, nfp_candidates=None, stellsym_claim=True,
                       close_tol=1e-12, check_symmetry=True, family="", groups=None):
    """Convert a parsed MakegridFile to a Source.

    extcur: optional {group_id: multiplier} or sequence (index group_id-1) applied to the
    file currents (for unit-current files).  groups: optional set of group ids to keep.
    nfp_candidates: nfp values to verify (default: [periods]).
    """
    loops_pts, loops_I, seg_a, seg_b, seg_I = [], [], [], [], []
    meta_loops = []
    open_loops = []
    ring_R, ring_Z, ring_I = [], [], []
    for k, lp in enumerate(mg.loops):
        if groups is not None and lp.group not in groups:
            continue
        mult = 1.0
        if extcur is not None:
            if isinstance(extcur, dict):
                mult = float(extcur.get(lp.group, 0.0))
            else:
                mult = float(extcur[lp.group - 1])
        P = lp.points
        if len(P) == 1:
            ring_R.append(float(np.hypot(P[0, 0], P[0, 1])))
            ring_Z.append(float(P[0, 2]))
            ring_I.append(float(lp.currents[0]) * mult)
            meta_loops.append({"group": lp.group, "name": lp.name, "n_points": 1, "ring": True})
            continue
        Iseg = lp.currents[:-1] * mult          # current of segment p_j -> p_{j+1}
        scale = float(np.abs(P).max())
        gap = float(np.linalg.norm(P[-1] - P[0]))
        closed = gap <= close_tol * max(scale, 1e-300)
        if not np.all(Iseg == Iseg[0]):
            # current varies along the loop: exact representation as independent segments
            seg_a.append(P[:-1])
            seg_b.append(P[1:])
            seg_I.append(Iseg)
            meta_loops.append({"group": lp.group, "name": lp.name, "n_points": int(len(P)), "segments": True})
            continue
        V = P[:-1] if closed else P
        if not closed:
            open_loops.append({"loop": k, "gap_m": gap})
        loops_pts.append(V)
        loops_I.append(Iseg[0])
        meta_loops.append({"group": lp.group, "name": lp.name, "n_points": int(len(V))})
    parts = []
    if loops_pts:
        parts.append(polyline_part(loops_pts, loops_I))
    if seg_a:
        parts.append(Segments(np.concatenate(seg_a), np.concatenate(seg_b), np.concatenate(seg_I)))
    if ring_R:
        parts.append(Rings(np.array(ring_R), np.array(ring_Z), np.array(ring_I)))
    src = Source(id=source_id, parts=parts, family=family,
                 provenance={"files": [rel_path(mg.path)], "loader": "torux.io.makegrid:load_makegrid",
                             "format": "MAKEGRID"},
                 meta={"periods_header": mg.periods, "mirror": mg.mirror,
                       "groups": sorted({(m["group"], m["name"]) for m in meta_loops}),
                       "n_loops": len(meta_loops), "open_loops": open_loops,
                       "extcur_applied": extcur is not None})
    src.meta["groups"] = [list(g) for g in src.meta["groups"]]
    src.meta["loop_groups"] = [m["group"] for m in meta_loops]
    if check_symmetry and parts:
        cands = nfp_candidates if nfp_candidates is not None else [mg.periods]
        apply_verified_symmetry(src, cands, stellsym_claim, fast=False)
    return src


def load_makegrid(path, source_id=None, start_marker=None, **kw):
    mg = read_makegrid(path, start_marker=start_marker)
    if source_id is None:
        source_id = "makegrid/" + rel_path(mg.path)
    return makegrid_to_source(mg, source_id, **kw)
