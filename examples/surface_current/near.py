"""How close to a SurfaceCurrent's boundary the field holds: torux's FieldEngine (coils and sheet) against the
independent evaluator's double-double tier (torux.reference.precise), at random points of the boundary moved out
along its normal by d, for d from 1e-1 to 1e-5 of its minor radius.

    python examples/surface_current/near.py examples/vmex_surface_current/current.json [POINTS]

Prints one line per distance: the largest |B - B_dd| / |B| over POINTS points (default 8), and the seconds per point
of each evaluator.  The double-double tier takes seconds per point."""
import json
import sys
import time

import numpy as np

import torux
from torux.field import FieldEngine
from torux.reference import precise


def boundary_points(part, theta, phi, d):
    """Points of the boundary at (theta, phi), moved out along the unit normal r_phi x r_theta by d (m)."""
    m, n = part.modes[:, 0].astype(float), part.modes[:, 1].astype(float) * part.nfp
    zero = np.zeros(len(m))
    rs = zero if part.rmns is None else part.rmns
    zc = zero if part.zmnc is None else part.zmnc
    ang = np.outer(theta, m) - np.outer(phi, n)
    c, s = np.cos(ang), np.sin(ang)
    R, Z = c @ part.rmnc + s @ rs, s @ part.zmns + c @ zc
    Rt, Zt = (-s * m) @ part.rmnc + (c * m) @ rs, (c * m) @ part.zmns + (-s * m) @ zc
    Rp, Zp = (s * n) @ part.rmnc - (c * n) @ rs, (-c * n) @ part.zmns + (s * n) @ zc
    cp, sp = np.cos(phi), np.sin(phi)
    rt = np.stack([Rt * cp, Rt * sp, Zt], -1)
    rp = np.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp], -1)
    normal = np.cross(rp, rt)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return np.stack([R * cp, R * sp, Z], -1) + d * normal


def main():
    src = torux.load(sys.argv[1])
    npts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    part = next(p for p in src.parts if isinstance(p, torux.SurfaceCurrent))
    engine = FieldEngine(src)
    a = engine.sheets[0].size["minor_radius"]
    dd = precise.Reference(src)
    rng = np.random.default_rng(3)
    for rel in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        x = boundary_points(part, rng.uniform(0, 2 * np.pi, npts), rng.uniform(0, 2 * np.pi, npts), rel * a)
        t0 = time.time()
        B = np.asarray(engine.evaluate(x, grad=False, A=False)["B"])
        t1 = time.time()
        p = dd.evaluate(x, grad=False)
        t2 = time.time()
        Bd = np.asarray(p["B"]) + np.asarray(p["B_lo"])
        err = np.linalg.norm(B - Bd, axis=1) / np.linalg.norm(Bd, axis=1)
        print(json.dumps(dict(d_over_minor_radius=rel, d_m=rel * a, max_rel_B=float(err.max()),
                              engine_s_per_point=(t1 - t0) / npts, dd_s_per_point=(t2 - t1) / npts)), flush=True)


if __name__ == "__main__":
    main()
