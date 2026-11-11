"""Small 2D geometry helpers for axisymmetric polygons (meridional (R, Z) plane)."""
from __future__ import annotations

import numpy as np


def signed_area(V):
    V = np.asarray(V, dtype=float)
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ear_clip(V):
    """Triangulate a simple polygon (any orientation); returns a list of index triples,
    each triangle counter-clockwise."""
    V = np.asarray(V, dtype=float)
    n = len(V)
    idx = list(range(n))
    if signed_area(V) < 0:
        idx = idx[::-1]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = V[i0], V[i1], V[i2]
            if cross(a, b, c) <= 0:
                continue
            if any(inside(V[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError("ear clipping failed (polygon not simple?)")
    if len(idx) == 3:
        a, b, c = V[idx[0]], V[idx[1]], V[idx[2]]
        if cross(a, b, c) > 0:
            tris.append(tuple(idx))
    return tris
