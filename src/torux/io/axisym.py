"""Axisymmetric coil sets: IMAS pf_active (JSON / NetCDF / HDF5), FreeGS and FreeGSNKE
derived tables, TokaMaker meshes, DipolEq inputs, the SPARC PRD coil table, STEP, ARC.

Canonical parts (torux/source.py): ``Rings`` (filaments), ``AxisymPolygons`` (uniform
current over a polygonal (R, Z) cross-section), ``ToroidalField`` (RB / R).
Element current = total toroidal ampere-turns of the element, positive along +phi.

Conventions per source:

FreeGS (a table of its machines' coils; freegs/coil.py, shaped_coil.py, filament_coil.py,
  machine.py): Coil -> ring with current*turns; ShapedCoil -> polygon with current*turns
  (uniformly distributed, as the quadrature of shaped_coil.py integrates it);
  FilamentCoil (MASTU) -> its sample rings, each current*turns/npoints (the TSV column
  ampere_turn_factor_per_sample); Solenoid -> Ns rings at linspace(Zsmin, Zsmax, Ns), each
  carrying the solenoid current (machine.py controlPsi).  Circuit members are scaled by
  their circuit multiplier.  Currents: defaults are zero, so unit current per circuit
  (1 A per turn), meta currents = "unit".
FreeGSNKE (active_coils.json; multi_coil.py, build_machine.py): every listed
  filament is a ring carrying circuit current * polarity * multiplier (no averaging over
  filaments).  dR, dZ are the plotting/inductance rectangle; the "rect" variant uses them
  as AxisymPolygons with the same element currents (a geometry variant, not the
  FreeGSNKE field model).
TokaMaker meshes (*_mesh.h5): coil region reg_id, signed nturns, coil_set.  Region
  current = I(coil_set, A per turn) * nturns, uniformly distributed over the region's
  triangles (each triangle: I * nturns * area / region area).  "triangles": one polygon
  per triangle (exact); "merged": the region's boundary polygon when the region is a
  single simply connected loop (exact too: uniform current over a union), else triangles.
IMAS pf_active (SPARC device description JSON, ARC device_V3A.json, STEP NetCDF/HDF5):
  with circuit currents (STEP): element ampere-turns = circuit current * coil sign in the
  circuit * turns_with_sign.  Coil sign in a circuit: the circuit is followed from supply
  side 1; a coil entered at side 1 (and left at side 2) carries +I, entered at side 2 carries
  -I (IMAS: coil current positive from side 1 to side 2).  All STEP and ARC coils come out +1
  (STEP confirmed against FIESTA's Psi_vac).  Without currents (SPARC, ARC device files):
  unit current per coil, element = turns_with_sign, wiring recorded but not applied.  SPARC
  VSC is wired VS1U side 2 -> VS1L side 2 (sign -1) while VS1L also has turns_with_sign = -1:
  the strict IMAS reading of the circuit makes the pair co-directional, the TokaMaker notebook
  reads the turns sign alone as the anti-symmetric VS drive (unresolved data inconsistency).
  Geometry: rectangle -> 4-vertex polygon; annulus (r, z, radius_inner, radius_outer) with
  radius_inner = 0 -> regular N-gon (default N = 32) whose area-normalised polar second
  moment equals the disk's (circumradius rho = a sqrt(3 / (2 + cos(2 pi / N))));
  the exact disk parameters are kept in meta["annuli"].  (A disk is not a polygon; an
  exact "AxisymAnnuli" part kind would remove this approximation -- see tests for the
  measured field error.)  radius_inner > 0 does not occur (checked) and raises.
DipolEq (*.in): K_Coil blocks: rectangle centre (X, Z), full widths dX, dZ, InitialCurrent
  = total A; Enabled != 1 coils are left out.
SPARC PRD (PrimaryReferenceDischarge/8 - coilData...dat): currents in MA per named coil.
  CS coils: N_filaments rings at linspace(zMin, zMax, N) at rCtr, the coil current
  divided equally among them (FreeGS FilamentCoil convention, the code that produced the
  paired equilibrium; verified against the GEQDSK flux in the tests).  PF/DIV/VS: one
  ring at (Rctr, Zctr).  EFCC coils have no current in the table -> left out.
STEP (eb-cc/freeboundary): pf_active circuits as IMAS above; plus the TF centre-rod
  current irod of _src/equilibrium.mat (57.6 MA) as ToroidalField(RB = mu0 irod / 2 pi);
  equals equilibrium vacuum_toroidal_field b0 * r0 of the NetCDF to 1e-9.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from torux.source import Rings, AxisymPolygons, ToroidalField, MU0


# --------------------------------------------------------------------------- polygons
def polys_part(polys, currents):
    polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys]
    start = np.zeros(len(polys) + 1, dtype=np.int64)
    start[1:] = np.cumsum([len(p) for p in polys])
    return AxisymPolygons(np.concatenate(polys) if polys else np.zeros((0, 2)), start,
                          np.asarray(currents, dtype=np.float64))


def poly_area(P):
    P = np.asarray(P)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ccw(P):
    P = np.asarray(P, dtype=np.float64)
    return P if poly_area(P) >= 0 else P[::-1].copy()


def rect_polygon(r, z, w, h, rot_deg=0.0):
    P = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    if rot_deg:
        c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
        P = P @ np.array([[c, s], [-s, c]])
    return P + np.array([r, z])


def disk_polygon(r, z, a, N=32):
    rho = a * np.sqrt(3.0 / (2.0 + np.cos(2 * np.pi / N)))
    t = 2 * np.pi * (np.arange(N) + 0.5) / N
    return np.stack([r + rho * np.cos(t), z + rho * np.sin(t)], axis=1)


# --------------------------------------------------------------------------- IMAS pf_active
def imas_geometry_polygon(geom, N=32):
    """Polygon of an IMAS pf_active element geometry dict; returns (poly, annulus_or_None)."""
    gt = int(geom.get("geometry_type", 0))
    if gt == 2 or ("rectangle" in geom and gt in (0, 2)):
        g = geom["rectangle"]
        return rect_polygon(g["r"], g["z"], g["width"], g["height"]), None
    if gt == 5:
        g = geom["annulus"]
        if float(g.get("radius_inner", 0.0)) != 0.0:
            raise NotImplementedError("annulus with radius_inner > 0")
        return disk_polygon(g["r"], g["z"], g["radius_outer"], N), (g["r"], g["z"], g["radius_outer"])
    if gt == 1:
        g = geom["outline"]
        return ccw(np.stack([g["r"], g["z"]], axis=1)), None
    raise NotImplementedError(f"IMAS geometry_type {gt}")


def circuit_coil_signs(connections, n_supplies, n_coils):
    """Series-chain walk of an IMAS circuit connection matrix (nodes x 2(n_supplies+n_coils)).

    Returns (supply_index, [(coil_index, sign), ...]); sign +1 if the walk that leaves
    supply side 1 enters the coil at side 1."""
    C = np.asarray(connections, dtype=int)
    nn, nt = C.shape
    assert nt == 2 * (n_supplies + n_coils), (C.shape, n_supplies, n_coils)
    terms_at_node = [list(np.nonzero(C[i])[0]) for i in range(nn)]
    node_of = {}
    for i, ts in enumerate(terms_at_node):
        for t in ts:
            node_of[t] = i
    sup = [j for j in range(n_supplies) if 2 * j in node_of or 2 * j + 1 in node_of]
    if len(sup) != 1:
        raise ValueError(f"circuit with {len(sup)} supplies")
    s = sup[0]
    out = []
    cur_term = 2 * s                      # leave supply from side 1
    seen = set()
    for _ in range(nn + 2):
        node = node_of[cur_term]
        nxt = [t for t in terms_at_node[node] if t != cur_term]
        if len(nxt) != 1:
            raise ValueError("not a simple series circuit")
        t = nxt[0]
        if t < 2 * n_supplies:
            assert t == 2 * s + 1, "chain must return to the same supply's side 2"
            return s, out
        ci, side = divmod(t - 2 * n_supplies, 2)
        if ci in seen:
            raise ValueError("coil visited twice")
        seen.add(ci)
        out.append((ci, +1 if side == 0 else -1))
        cur_term = 2 * n_supplies + 2 * ci + (1 - side)
    raise ValueError("circuit walk did not close")


def imas_pf_active_json(path, ids_key="pf_active"):
    """Coils (elements as polygons + turns) and circuits of an IMAS-like JSON."""
    d = json.load(open(path))
    pa = d[ids_key]
    coils = []
    for c in pa["coil"]:
        els = []
        for e in c["element"]:
            P, ann = imas_geometry_polygon(e["geometry"])
            els.append({"poly": P, "annulus": ann, "turns": float(e["turns_with_sign"]),
                        "area": float(e.get("area", abs(poly_area(P))))})
        coils.append({"name": c.get("name", c.get("identifier")), "elements": els})
    ns, nc = len(pa.get("supply", [])), len(coils)
    circuits = []
    for cc in pa.get("circuit", []):
        s, members = circuit_coil_signs(cc["connections"], ns, nc)
        cur = cc.get("current", {}).get("data") if isinstance(cc.get("current"), dict) else None
        circuits.append({"name": cc.get("name"), "supply": s, "coils": members, "current": cur})
    return coils, circuits


def imas_pf_active_nc(path):
    """STEP-style IMAS NetCDF (group pf_active/0, sparse element arrays with fill values)."""
    import netCDF4
    d = netCDF4.Dataset(path)
    g = d["pf_active"]["0"]
    names = [str(x) for x in g["coil.name"][...]]
    nel = np.asarray(g["coil.element:shape"][...]).reshape(-1)
    get = lambda k: np.ma.filled(np.ma.asarray(g[k][...]).astype(np.float64), np.nan)
    turns = get("coil.element.turns_with_sign")
    gt = get("coil.element.geometry.geometry_type")
    r = get("coil.element.geometry.rectangle.r"); z = get("coil.element.geometry.rectangle.z")
    w = get("coil.element.geometry.rectangle.width"); h = get("coil.element.geometry.rectangle.height")
    coils = []
    for i, nm in enumerate(names):
        els = []
        for j in range(int(nel[i])):
            assert int(gt[i, j]) == 2
            els.append({"poly": rect_polygon(r[i, j], z[i, j], w[i, j], h[i, j]), "annulus": None,
                        "turns": float(turns[i, j]), "area": float(w[i, j] * h[i, j])})
        coils.append({"name": nm, "elements": els})
    cnames = [str(x) for x in g["circuit.name"][...]]
    conn = np.asarray(g["circuit.connections"][...])
    ccur = np.asarray(g["circuit.current.data"][...])
    ns = len(g["supply.name"][...])
    circuits = []
    for k, nm in enumerate(cnames):
        Ck = conn[k]
        Ck = Ck[np.any(Ck != 0, axis=1)]
        s, members = circuit_coil_signs(Ck, ns, len(coils))
        circuits.append({"name": nm, "supply": s, "coils": members, "current": ccur[k].tolist()})
    return coils, circuits


def imas_pf_active_h5(path):
    """STEP imasdb/pf_active.h5 (IMAS HDF5 backend, flattened AOS names with '&')."""
    import h5py
    f = h5py.File(path, "r")
    g = f["pf_active"]
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in g["coil[]&name"][...]]
    nel = np.asarray(g["coil[]&element[]&AOS_SHAPE"][...]).reshape(-1)
    turns = g["coil[]&element[]&turns_with_sign"][...]
    r = g["coil[]&element[]&geometry&rectangle&r"][...]; z = g["coil[]&element[]&geometry&rectangle&z"][...]
    w = g["coil[]&element[]&geometry&rectangle&width"][...]; h = g["coil[]&element[]&geometry&rectangle&height"][...]
    gt = g["coil[]&element[]&geometry&geometry_type"][...]
    coils = []
    for i, nm in enumerate(names):
        els = []
        for j in range(int(nel[i])):
            assert int(gt[i, j]) == 2
            els.append({"poly": rect_polygon(r[i, j], z[i, j], w[i, j], h[i, j]), "annulus": None,
                        "turns": float(turns[i, j]), "area": float(w[i, j] * h[i, j])})
        coils.append({"name": nm, "elements": els})
    cnames = [x.decode() if isinstance(x, bytes) else str(x) for x in g["circuit[]&name"][...]]
    conn = np.asarray(g["circuit[]&connections"][...])            # (circuit, terminal, node)
    cshape = np.asarray(g["circuit[]&connections_SHAPE"][...])    # (nodes, terminals) stored per circuit
    ccur = np.asarray(g["circuit[]&current&data"][...])
    ns = len(g["supply[]&name"][...])
    circuits = []
    for k, nm in enumerate(cnames):
        Ck = conn[k].T                                             # -> (node, terminal)
        Ck = Ck[np.any(Ck != 0, axis=1)]
        s, members = circuit_coil_signs(Ck, ns, len(coils))
        circuits.append({"name": nm, "supply": s, "coils": members, "current": ccur[k].tolist()})
    return coils, circuits


def imas_to_parts(coils, circuits, circuit_currents=None):
    """AxisymPolygons from IMAS coils/circuits.

    circuit_currents given (A, one per circuit): element ampere-turns = I_circuit * coil sign
    in the circuit * turns_with_sign.  circuit_currents None: unit current per *coil*
    (1 A flowing side 1 -> side 2, element = turns_with_sign), the circuit wiring recorded
    in the groups ("sign_in_circuit") but not applied.  Returns (part, groups, annuli)."""
    polys, cur, groups, annuli = [], [], [], []
    in_circ = {}
    for k, c in enumerate(circuits):
        for ci, sg in c["coils"]:
            in_circ[ci] = (k, sg)
    for ci, c in enumerate(coils):
        k, sg = in_circ.get(ci, (None, 1))
        if circuit_currents is None or k is None:
            I = 1.0
        else:
            I = float(circuit_currents[k]) * sg
        i0 = len(cur)
        for e in c["elements"]:
            polys.append(ccw(e["poly"]))
            cur.append(I * e["turns"])
            if e["annulus"] is not None:
                annuli.append([float(x) for x in e["annulus"]])
        groups.append({"name": c["name"], "circuit": None if k is None else circuits[k]["name"],
                       "sign_in_circuit": sg, "part": 0, "start": i0, "stop": len(cur),
                       "turns_sum": float(sum(e["turns"] for e in c["elements"]))})
    return polys_part(polys, cur), groups, annuli


# --------------------------------------------------------------------------- FreeGS
def read_tsv(path):
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def freegs_machine(tsv, machine, circuit_currents=None):
    """Parts of one FreeGS machine (from the table of its coils), unit current per circuit
    (A per turn) unless circuit_currents {circuit: A} is given."""
    rows = [r for r in read_tsv(tsv) if r["machine"] == machine]
    if not rows:
        raise KeyError(machine)
    ring_R, ring_Z, ring_I, polys, poly_I = [], [], [], [], []
    groups = {}
    objs = {}
    for r in rows:
        key = (r["circuit"], r["coil"], r["constructor"], r["source_line"])
        objs.setdefault(key, []).append(r)
    for (circ, coil, ctor, _), rr in objs.items():
        Ic = 1.0 if circuit_currents is None else float(circuit_currents.get(circ, 0.0))
        mult = float(rr[0]["circuit_multiplier"] or 1.0)
        turns = float(rr[0]["turns"] or 1.0)
        g = groups.setdefault(circ, {"name": circ, "members": []})
        if ctor == "Coil":
            assert len(rr) == 1
            ring_R.append(float(rr[0]["R_m"])); ring_Z.append(float(rr[0]["Z_m"])); ring_I.append(Ic * mult * turns)
            g["members"].append({"coil": coil, "kind": "ring", "n": 1, "ampere_turns_per_A": mult * turns})
        elif ctor == "ShapedCoil":
            P = np.array([[float(x["R_m"]), float(x["Z_m"])] for x in sorted(rr, key=lambda x: int(x["index"]))])
            polys.append(ccw(P)); poly_I.append(Ic * mult * turns)
            g["members"].append({"coil": coil, "kind": "polygon", "n": 1, "ampere_turns_per_A": mult * turns})
        elif ctor == "FilamentCoil":
            for x in sorted(rr, key=lambda x: int(x["index"])):
                f = float(x["ampere_turn_factor_per_sample"])
                ring_R.append(float(x["R_m"])); ring_Z.append(float(x["Z_m"])); ring_I.append(Ic * mult * f)
            g["members"].append({"coil": coil, "kind": "filament_coil", "n": len(rr),
                                 "ampere_turns_per_A": mult * sum(float(x["ampere_turn_factor_per_sample"]) for x in rr)})
        elif ctor == "Solenoid":
            assert len(rr) == 1
            x = rr[0]
            Ns = int(float(x["turns"]))
            Zs = np.linspace(float(x["Zmin_m"]), float(x["Zmax_m"]), Ns)
            for zz in Zs:
                ring_R.append(float(x["R_m"])); ring_Z.append(float(zz)); ring_I.append(Ic * mult)
            g["members"].append({"coil": coil, "kind": "solenoid", "n": Ns, "ampere_turns_per_A": mult * Ns})
        else:
            raise NotImplementedError(ctor)
    parts = []
    if ring_R:
        parts.append(Rings(ring_R, ring_Z, ring_I))
    if polys:
        parts.append(polys_part(polys, poly_I))
    return parts, list(groups.values())


# --------------------------------------------------------------------------- FreeGSNKE
def freegsnke_active(path, rect=False, circuit_currents=None):
    """Rings (or dR x dZ rectangles if rect) of a FreeGSNKE derived active_coils.json."""
    d = json.load(open(path))
    R, Z, I, polys = [], [], [], []
    groups = []
    for circ, v in d.items():
        Ic = 1.0 if circuit_currents is None else float(circuit_currents.get(circ, 0.0))
        subs = [(None, v)] if "R" in v else list(v.items())
        i0 = len(I)
        for sub, s in subs:
            fac = float(s["polarity"]) * float(s["multiplier"])
            for k in range(len(s["R"])):
                R.append(float(s["R"][k])); Z.append(float(s["Z"][k])); I.append(Ic * fac)
                if rect:
                    dR = np.atleast_1d(np.asarray(s["dR"], float)); dZ = np.atleast_1d(np.asarray(s["dZ"], float))
                    polys.append(rect_polygon(float(s["R"][k]), float(s["Z"][k]), float(dR[k if dR.size > 1 else 0]),
                                              float(dZ[k if dZ.size > 1 else 0])))
        groups.append({"name": circ, "part": 0, "start": i0, "stop": len(I),
                       "subcoils": [sub for sub, _ in subs]})
    if rect:
        return [polys_part(polys, I)], groups
    return [Rings(R, Z, I)], groups


# --------------------------------------------------------------------------- TokaMaker
def tokamaker_mesh(path, merge=True, coilset_currents=None):
    """AxisymPolygons of the coil regions of a TokaMaker mesh (see module doc)."""
    import h5py
    f = h5py.File(path, "r")
    r = f["mesh/r"][...]
    lc = f["mesh/lc"][...].astype(np.int64)
    reg = f["mesh/reg"][...].astype(np.int64)
    cd = json.loads(f["mesh/coil_dict"][()])
    polys, cur, groups = [], [], []
    for name, c in sorted(cd.items(), key=lambda kv: kv[1]["coil_id"]):
        tri = lc[reg == int(c["reg_id"])]
        P = r[tri]                                                   # (nt, 3, 2)
        a = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
        A = float(np.abs(a).sum())
        Is = 1.0 if coilset_currents is None else float(coilset_currents.get(c["coil_set"], 0.0))
        It = Is * float(c["nturns"])
        i0 = len(cur)
        loop = _region_boundary(tri, a, r) if merge else None
        if loop is not None and abs(abs(poly_area(loop)) - A) <= 1e-12 * max(A, 1e-300) * len(tri):
            polys.append(ccw(loop)); cur.append(It)
            how = "merged"
        else:
            for k in range(len(tri)):
                polys.append(ccw(P[k])); cur.append(It * abs(a[k]) / A)
            how = "triangles"
        groups.append({"name": name, "coil_set": c["coil_set"], "nturns": float(c["nturns"]), "reg_id": int(c["reg_id"]),
                       "part": 0, "start": i0, "stop": len(cur), "triangles": int(len(tri)), "area_m2": A,
                       "representation": how})
    return [polys_part(polys, cur)], groups


def _region_boundary(tri, signed_area, r):
    """Single boundary loop of a triangulated region (None if the boundary is not one loop)."""
    T = tri.copy()
    neg = signed_area < 0
    T[neg] = T[neg][:, ::-1]
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    key = {}
    for a, b in edges:
        key[(a, b)] = key.get((a, b), 0) + 1
    bnd = [(a, b) for (a, b) in key if (b, a) not in key]
    if not bnd:
        return None
    nxt = {}
    for a, b in bnd:
        if a in nxt:
            return None
        nxt[a] = b
    start = bnd[0][0]
    loop = [start]
    v = nxt[start]
    while v != start:
        loop.append(v)
        v = nxt.get(v)
        if v is None or len(loop) > len(bnd):
            return None
    if len(loop) != len(bnd):
        return None
    return r[np.array(loop)]


# --------------------------------------------------------------------------- DipolEq
def read_dipoleq(path):
    """K_Coil blocks of a DipolEq input: list of dicts (Name, Enabled, InitialCurrent, X, dX, Z, dZ)."""
    coils, cur = [], None
    for ln in open(path).read().splitlines():
        s = ln.split("//")[0].strip()
        if not s:
            continue
        if s.startswith("K_"):
            if cur is not None:
                coils.append(cur)
            cur = {"_block": s} if s == "K_Coil" else None
            continue
        if cur is not None and "=" in s:
            k, v = [t.strip() for t in s.split("=", 1)]
            try:
                cur[k] = float(v)
            except ValueError:
                cur[k] = v
    if cur is not None:
        coils.append(cur)
    return [c for c in coils if c.get("_block") == "K_Coil"]


def dipoleq_parts(path):
    cs = read_dipoleq(path)
    polys, I, groups, off = [], [], [], []
    for c in cs:
        if int(c.get("Enabled", 1)) != 1:
            off.append(c["Name"])
            continue
        groups.append({"name": c["Name"], "part": 0, "start": len(I), "stop": len(I) + 1})
        polys.append(rect_polygon(c["X"], c["Z"], c["dX"], c["dZ"]))
        I.append(float(c["InitialCurrent"]))
    return [polys_part(polys, I)], groups, off


# --------------------------------------------------------------------------- SPARC PRD
def read_sparc_prd(path):
    txt = open(path).read()
    cur = {}
    for m in re.finditer(r"^\s*([A-Za-z0-9]+)\s*:\s*([-+0-9.eE]+)\s*$", txt, re.M):
        cur[m.group(1).lower()] = float(m.group(2)) * 1e6
    cs, rings = {}, {}
    sect = None
    for ln in txt.splitlines():
        s = ln.strip()
        if s.startswith("==="):
            sect = s.strip("= ").split()[0].upper()
            continue
        tok = [t.strip() for t in s.split(",")]
        if len(tok) < 3 or not _num(tok[1]):
            continue
        name = tok[0].lower()
        if sect == "CS":
            cs[name] = (float(tok[1]), float(tok[2]), float(tok[3]), int(float(tok[4])))
        else:
            rings[name] = (float(tok[1]), float(tok[2]), sect)
    return cur, cs, rings


def _num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def sparc_prd_parts(path, cs_split="equal"):
    cur, cs, rings = read_sparc_prd(path)
    R, Z, I, groups, missing = [], [], [], [], []
    for name, (rc, zmin, zmax, nf) in cs.items():
        Ic = cur[name]
        i0 = len(I)
        for zz in np.linspace(zmin, zmax, nf):
            R.append(rc); Z.append(float(zz)); I.append(Ic / nf if cs_split == "equal" else Ic)
        groups.append({"name": name, "part": 0, "start": i0, "stop": len(I), "current_A": Ic})
    for name, (rc, zc, sect) in rings.items():
        if name not in cur:
            missing.append(name)
            continue
        groups.append({"name": name, "part": 0, "start": len(I), "stop": len(I) + 1, "current_A": cur[name]})
        R.append(rc); Z.append(zc); I.append(cur[name])
    return [Rings(R, Z, I)], groups, missing


# --------------------------------------------------------------------------- STEP
def step_irod(mat_path):
    import scipy.io as sio
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return float(m["equil"].irod)


def step_psi_vac(mat_path):
    """FIESTA vacuum flux of the free-boundary equilibrium: (r (nr,), z (nz,), Psi_vac (nz, nr)),
    circuit labels and currents."""
    import scipy.io as sio
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    e = m["equil"]
    r = np.asarray(e.Psi_vac.grid.r, float)
    z = np.asarray(e.Psi_vac.grid.z, float)
    P = np.asarray(e.Psi_vac.data, float)
    return r, z, P, list(e.icoil.circuit_labels), np.asarray(e.icoil.currents, float)


# --------------------------------------------------------------------------- GEQDSK (references)
def read_geqdsk(path):
    """Minimal GEQDSK reader (for reference checks): dict with nw, nh, rdim, zdim, rleft, zmid,
    rmaxis, zmaxis, simag, sibry, bcentr, current, fpol, pres, ffprim, pprime, psirz (nh, nw),
    qpsi, rbbbs, zbbbs, rlim, zlim.  psi as stored (per radian or not: source dependent)."""
    txt = Path(path).read_text(errors="replace").splitlines()
    head = txt[0]
    ints = re.findall(r"\d+", head[48:]) if len(head) > 48 else re.findall(r"\d+", head)
    nw, nh = int(ints[-2]), int(ints[-1])
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", "\n".join(txt[1:]).replace("D", "E"))]
    it = iter(nums)
    nx = lambda n: np.array([next(it) for _ in range(n)])
    rdim, zdim, rcentr, rleft, zmid = nx(5)
    rmaxis, zmaxis, simag, sibry, bcentr = nx(5)
    current, _, _, _, _ = nx(5)
    _ = nx(5)
    g = dict(nw=nw, nh=nh, rdim=rdim, zdim=zdim, rcentr=rcentr, rleft=rleft, zmid=zmid, rmaxis=rmaxis,
             zmaxis=zmaxis, simag=simag, sibry=sibry, bcentr=bcentr, current=current)
    g["fpol"] = nx(nw); g["pres"] = nx(nw); g["ffprim"] = nx(nw); g["pprime"] = nx(nw)
    g["psirz"] = nx(nw * nh).reshape(nh, nw)
    g["qpsi"] = nx(nw)
    try:
        nb, nl = int(next(it)), int(next(it))
        bb = nx(2 * nb).reshape(nb, 2); ll = nx(2 * nl).reshape(nl, 2)
        g["rbbbs"], g["zbbbs"], g["rlim"], g["zlim"] = bb[:, 0], bb[:, 1], ll[:, 0], ll[:, 1]
    except StopIteration:
        pass
    g["R"] = rleft + np.linspace(0, rdim, nw)
    g["Z"] = zmid - zdim / 2 + np.linspace(0, zdim, nh)
    return g
