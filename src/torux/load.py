"""Field sources from files.

    src = torux.load("quasr_0000954.json")                    # a QUASR download (SIMSOPT JSON)
    src = torux.load("coils.json")                            # Fourier-filament JSON
    src = torux.load("coils.w7x")                             # MAKEGRID
    src = torux.load("tf.coils", "magnets.focus", nfp=2)      # several files, one source: coils and magnets
    src = torux.load("d3d_iu.dat", kind="tokamak")            # GPEC coil geometry

The format is read from the file itself.  What each format gives:

  SIMSOPT JSON         every coil of a BiotSavart object, each curve class converted exactly to a Fourier
                       filament (framed and winding-surface curves by FFT with a measured error).  A file with
                       several BiotSavart objects needs which=.  A Current object whose two values differ (the
                       one it held when saved and its degrees of freedom) needs currents=.  A QUASR download
                       also carries its optimisation surfaces: they become the start of the solve, and the
                       flux QUASR's own definition assigns to the outermost one becomes the request
                       (Source.request), so torux.solve(path) needs nothing else.
  Fourier-filament JSON  {"nfp", "stellarator_symmetric", "coils": [{"xc", "xs", "yc", "ys", "zc", "zs",
                       "current_A"}, ...]}, r(t) = c[0] + sum_k c[k] cos 2 pi k t + s[k] sin 2 pi k t, every coil
                       of the device written out; optional "requested_toroidal_flux_Wb".
  MAKEGRID             coils.* files: closed polygons with their currents.
  FOCUS                coil tables (Fourier filaments; give nfp) and dipole tables (permanent magnets; give nfp).
  GPEC                 coil-geometry .dat files (closed polygons, 1 A a coil).
  wireframe CSV        straight segments with their currents.
  TokaMaker mesh       .h5: axisymmetric conductors by cross-section (needs h5py).
  DESC coil set        .h5: every coil of a CoilSet as a Fourier filament (needs h5py).
  torux JSON           a Source saved with Source.save.

A claimed symmetry (field periods, stellarator symmetry) is used only after it has been verified on the parts.

``kind`` says what sort of source it is: 'stellarator' (the default: its surfaces are sought as tori), or
'tokamak' / 'dipole' (a tokamak's own coils, perturbation coils, a levitated dipole: settled by proof, see
torux.nonstell).  GPEC and TokaMaker files default to 'tokamak'.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .source import Source

KINDS = ("stellarator", "tokamak", "dipole")


def _name(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def _sniff(path):
    p = str(path)
    with open(p, "rb") as fh:
        head = fh.read(4096)
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    text = head.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        return "json"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    low = text.lower()
    if first.lower().startswith("periods") or "begin filament" in low:
        return "makegrid"
    if first.split(",")[:2] == ["p1x", "p1y"]:
        return "wireframe"
    if first.startswith("#") and "total number of" in first.lower():
        # FOCUS tables: a dipole table's rows are comma separated and begin with coil type 2
        rows = [ln for ln in lines[1:] if not ln.startswith("#")][1:3]
        if rows and "," in rows[0] and rows[0].split(",")[0].strip() == "2":
            return "focus_dipoles"
        return "focus_coils"
    if first and len(first.split()) == 4 and all(_isnum(t) for t in first.split()):
        return "gpec"
    raise ValueError(f"{p}: format not recognised (see torux.load for the formats read)")


def _isnum(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def _coil_surface_distance(src, surf):
    """Least distance between the filaments (2,048 points each) and a surface (a 256 x 256 grid of the torus)."""
    from scipy.spatial import cKDTree
    from .io.surfaces import surface_grid
    r, _, _, _ = surface_grid(surf, nphi=256, ntheta=256, full_torus=True)
    tree = cKDTree(r.reshape(-1, 3))
    best = np.inf
    for p in src.parts:
        if p.kind != "fourier":
            continue
        x, _ = p.points(np.arange(2048) / 2048)
        d, _ = tree.query(np.asarray(x).reshape(-1, 3), k=1)
        best = min(best, float(d.min()))
    return best


def _design_request(src, surfs, provenance):
    """The quadrature floor from the coils' distance to the outermost design surface, and the flux QUASR's
    definition assigns to that surface: the loop integral of A round its own poloidal curve at phi = 0, by the
    trapezoid rule on the surface's own quadrature points."""
    from .design import quasr_design_flux
    from .field import FieldEngine
    outer = surfs[-1]
    dist = _coil_surface_distance(src, outer)
    if np.isfinite(dist):
        src.meta["dmin"] = 0.2 * dist
    eng = FieldEngine(src, dmin=src.meta.get("dmin", 0.05))
    qt = getattr(outer, "quadpoints_theta", None)
    nq = len(qt) if qt is not None and len(qt) else None
    flux = quasr_design_flux(outer, lambda x: np.asarray(eng.evaluate(x, grad=False, A=True)["A"]), nq=nq)
    src.request = {"flux_Wb": float(flux), "provenance": provenance}


def _load_json(path, name, nfp, which, currents=None, how=None):
    with open(path, "rb") as fh:
        d = json.loads(fh.read())
    if isinstance(d, dict) and "simsopt_objs" in d:
        from .io.simsopt_json import Simson, load_quasr_serial, load_simson_sources
        from .io.surfaces import simson_surfaces
        sim = Simson(d, str(path), currents)
        if sim.graph_coil_list() is not None:
            src, sim = load_quasr_serial(str(path), name, currents=currents)
            surfs = simson_surfaces(sim, [r["value"] for r in sim.graph[0]])
            if surfs:
                src._design_surfaces = surfs
                _design_request(src, surfs, "the flux enclosed by the file's outermost stored surface "
                                            "(loop integral of A at its own quadrature points)")
            return src
        from .io.simsopt_json import coil_sets, pick_coil_set
        sets = coil_sets(sim, name)
        if not sets:
            raise ValueError(f"{path}: no BiotSavart object with coils")
        sid = pick_coil_set(sim, sets, which, how)[0]
        return load_simson_sources(str(path), name, extra_nfp=(nfp,) if nfp else (), which=sid,
                                   currents=currents)[0]
    if isinstance(d, dict) and isinstance(d.get("coils"), list) and d["coils"] and "xc" in d["coils"][0]:
        from .io.fourier_json import load_fourier_json
        src = load_fourier_json(str(path), source_id=name, request_from_file=True,
                                request_provenance="the file's requested_toroidal_flux_Wb")
        start = (d.get("start") or {}).get("surface")
        if start:
            from .io.surfaces import SurfaceXYZTensorFourier
            surf = SurfaceXYZTensorFourier(start["nfp"], start["stellsym"], start["mpol"], start["ntor"],
                                           np.asarray(start["x"], float),
                                           quadpoints_phi=start.get("quadpoints_phi"),
                                           quadpoints_theta=start.get("quadpoints_theta"), name="start")
            src._design_surfaces = [surf]
            dist = _coil_surface_distance(src, surf)
            if np.isfinite(dist):
                src.meta["dmin"] = 0.2 * dist
        return src
    if isinstance(d, dict) and "parts" in d and "id" in d:
        return Source.from_dict(d)
    raise ValueError(f"{path}: a JSON file, but not SIMSOPT, Fourier-filament or torux JSON")


def _load_one(path, name, nfp, which, currents=None, how=None):
    fmt = _sniff(path)
    if fmt == "json":
        return _load_json(path, name, nfp, which, currents, how), None
    if fmt == "makegrid":
        from .io.makegrid import load_makegrid
        kw = dict(nfp_candidates=[nfp]) if nfp else {}
        return load_makegrid(str(path), source_id=name, **kw), None
    if fmt == "focus_coils":
        if not nfp:
            raise ValueError(f"{path}: a FOCUS coil table does not say its field periods: give nfp=")
        from .io.focus import load_focus
        return load_focus(str(path), nfp, source_id=name), None
    if fmt == "focus_dipoles":
        if not nfp:
            raise ValueError(f"{path}: a FOCUS dipole table does not say its field periods: give nfp=")
        from .io.focus_dipoles import read_focus_dipoles, table_to_dipoles
        dip, rec = table_to_dipoles(read_focus_dipoles(str(path)), int(nfp))
        return Source(id=name, parts=[dip], meta={"dipole_table": rec},
                      provenance={"files": [str(path)], "loader": "torux.io.focus_dipoles"}), None
    if fmt == "gpec":
        from .io.gpec import gpec_part, read_gpec
        g = read_gpec(str(path))
        part, groups = gpec_part(g)
        return Source(id=name, parts=[part], family="tokamak-3D",
                      meta={"groups": groups, "currents": "unit"},
                      provenance={"files": [str(path)], "loader": "torux.io.gpec"}), "tokamak"
    if fmt == "wireframe":
        from .io.wireframe import read_wireframe_csv, wireframe_segments
        seg, rec = wireframe_segments(read_wireframe_csv(str(path)))
        return Source(id=name, parts=[seg], meta={"wireframe": rec},
                      provenance={"files": [str(path)], "loader": "torux.io.wireframe"}), None
    if fmt == "hdf5":
        import h5py
        with h5py.File(str(path), "r") as fh:
            mesh = "mesh" in fh and "r" in fh["mesh"]
        if mesh:
            from .io.axisym import tokamaker_mesh
            parts, groups = tokamaker_mesh(str(path))
            return Source(id=name, parts=list(parts), family="tokamak", meta={"groups": groups, "currents": "unit"},
                          provenance={"files": [str(path)], "loader": "torux.io.axisym:tokamaker_mesh"}), "tokamak"
        from .io.magnets import desc_coilset
        part, labels = desc_coilset(str(path))
        return Source(id=name, parts=[part], meta={"desc_groups": labels},
                      provenance={"files": [str(path)], "loader": "torux.io.magnets:desc_coilset"}), None
    raise ValueError(f"{path}: format {fmt} not handled")


def per_file(value, n, what):
    """One entry a file from which= or currents=: a list or tuple is one entry per file, in order, and must have n.
    One value is which= of the first file (every other then needs none) and currents= of every file."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{what}= has {len(value)} entr{'y' if len(value) == 1 else 'ies'} for {n} source "
                             f"file{'s' if n != 1 else ''}: give one entry per file, in order, None for a file that "
                             f"needs none")
        return list(value)
    return [value] + [None] * (n - 1) if what == "which" else [value] * n


WHICH_OF_SEVERAL = ("which=, a list with one entry per source file in order, None for a file that needs none (on the "
                    "command line, --coils once per source file in order, - for a file that needs none)")


def load(path, *more, nfp=None, kind=None, flux=None, which=None, name=None, currents=None):
    """A Source from one file, or from several whose parts add up to one source.

    nfp       field periods, for formats that do not state them (FOCUS tables) and as the symmetry to verify
    kind      'stellarator' (default), 'tokamak' or 'dipole'
    flux      the requested full toroidal flux in Wb, kept with the source (Source.request)
    which     for a SIMSOPT JSON file holding several coil sets: the one to take (its name or index). With several
              files, one value is the first file's, and a list gives one entry per file, None where a file needs none
    name      the source's id (default: the first file's name)
    currents  for a SIMSOPT JSON file whose Current objects state two different values: 'saved' (the value the
              object held when the file was saved) or 'dofs' (the value SIMSOPT reads back). Without it such a
              file is refused. One value is every file's, a list one entry per file"""
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    name = name or _name(path)
    whichs = per_file(which, 1 + len(more), "which")
    currs = per_file(currents, 1 + len(more), "currents")
    how = WHICH_OF_SEVERAL if more else None
    src, default_kind = _load_one(path, name, nfp, whichs[0], currs[0], how)
    for p, w, c in zip(more, whichs[1:], currs[1:]):
        other, k2 = _load_one(p, _name(p), nfp, w, c, how)
        src.parts = list(src.parts) + list(other.parts)
        src.provenance.setdefault("files", []).extend(other.provenance.get("files", [str(p)]))
        for k, v in other.meta.items():
            src.meta.setdefault(k, v)
        default_kind = default_kind or k2
    if more or (nfp and "symmetry_check" not in src.meta):
        # the symmetry of the whole: only what every part shares, verified on the parts together
        from .io._stellcommon import apply_verified_symmetry
        src.nfp, src.stellsym = 1, False
        apply_verified_symmetry(src, [int(nfp)] if nfp else [1], True, fast=False)
    src.meta["kind"] = kind or default_kind or src.meta.get("kind") or "stellarator"
    if flux is not None:
        src.request = {"flux_Wb": float(flux), "provenance": "given to torux.load"}
    return src
