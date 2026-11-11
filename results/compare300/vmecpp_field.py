"""The coils' magnetic field for VMEC++ (run_vmecpp.py), computed from the input's Fourier coefficients.

Each coil is r(t) = c_0 + sum_k c_k cos 2 pi k t + s_k sin 2 pi k t, t in [0, 1).  Its Biot-Savart integral is
taken with the N-point trapezoid rule, which is spectrally accurate on a smooth closed curve; N is chosen per
design by doubling until the field no longer changes (see `converge_nodes`).

Two kernels compute the same sum:
- `CoilField.xyz`: SIMSOPT's C++ `biot_savart_B` (simsoptpp, installed with VMEC++ 0.9.3, OpenMP over the points),
  for the quadrature test before VMEC++ starts, from the main thread.
- `CoilField.cyl`: VMEC++'s MagneticFieldFunction, a numba kernel compiled `nogil` and single-threaded.  VMEC++'s
  4 vacuum threads each call it on their own slice of the boundary, at the same time, so the field is computed by
  VMEC++'s own 4 threads and no other thread runs (exactly 4 threads; an OpenMP kernel called from VMEC++'s
  threads would start a team of its own per calling thread, whose waiting threads compete for the cores).

    x = (R cos phi, R sin phi, Z),  B_R = B_x cos phi + B_y sin phi,  B_phi = -B_x sin phi + B_y cos phi
"""
import threading
import time

import numba
import numpy as np
import simsoptpp as sopp


def coil_nodes(coil, n):
    """(gamma, dgamma/dt) at t_j = j / n, each (n, 3), from the coil's Fourier coefficients."""
    k = np.arange(len(coil["xc"]))
    ang = 2 * np.pi * np.outer(np.arange(n), k) / n
    c, s = np.cos(ang), np.sin(ang)
    cos = np.array([coil[a + "c"] for a in "xyz"]).T          # (K+1, 3)
    sin = np.array([coil[a + "s"] for a in "xyz"]).T
    sin[0] = 0.0
    w = 2 * np.pi * k[:, None]
    g = c @ cos + s @ sin
    dg = s @ (-w * cos) + c @ (w * sin)
    return np.ascontiguousarray(g), np.ascontiguousarray(dg)


@numba.njit(nogil=True, cache=False, fastmath=False)
def _biot_savart_cyl(r, phi, z, g, dg, cur):
    """B_R, B_phi, B_Z at (r, phi, z) of coils with nodes g[c, j], tangents dg[c, j] (d/dt), currents cur[c]:
    1e-7 sum_c I_c (1/N) sum_j dg x (x - g) / |x - g|^3."""
    P = r.shape[0]
    C, N = g.shape[0], g.shape[1]
    br = np.empty(P)
    bp = np.empty(P)
    bz = np.empty(P)
    for p in range(P):
        cp, sp = np.cos(phi[p]), np.sin(phi[p])
        x0, x1, x2 = r[p] * cp, r[p] * sp, z[p]
        bx = 0.0
        by = 0.0
        bzz = 0.0
        for c in range(C):
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for j in range(N):
                rx = x0 - g[c, j, 0]
                ry = x1 - g[c, j, 1]
                rz = x2 - g[c, j, 2]
                r2 = rx * rx + ry * ry + rz * rz
                inv = 1.0 / (r2 * np.sqrt(r2))
                sx += (dg[c, j, 1] * rz - dg[c, j, 2] * ry) * inv
                sy += (dg[c, j, 2] * rx - dg[c, j, 0] * rz) * inv
                sz += (dg[c, j, 0] * ry - dg[c, j, 1] * rx) * inv
            bx += cur[c] * sx
            by += cur[c] * sy
            bzz += cur[c] * sz
        f = 1e-7 / N
        bx *= f
        by *= f
        bzz *= f
        br[p] = bx * cp + by * sp
        bp[p] = -bx * sp + by * cp
        bz[p] = bzz
    return br, bp, bz


class CoilField:
    """B of the full coil set at Cartesian or cylindrical points, with n_nodes per coil."""

    def __init__(self, coils, n_nodes):
        self.n_nodes = int(n_nodes)
        self.currents = [float(c["current_A"]) for c in coils]
        nodes = [coil_nodes(c, self.n_nodes) for c in coils]
        self.gammas = [g for g, _ in nodes]
        self.dgammas = [d for _, d in nodes]
        self._g = np.ascontiguousarray(np.stack(self.gammas))      # (C, N, 3)
        self._dg = np.ascontiguousarray(np.stack(self.dgammas))
        self._cur = np.asarray(self.currents, dtype=np.float64)
        self.calls = 0
        self.points = 0
        self.seconds = 0.0          # summed over the vacuum threads' calls (they overlap in time)
        self._lock = threading.Lock()

    def xyz(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        return np.asarray(sopp.biot_savart_B(x, self.gammas, self.dgammas, self.currents))

    def cyl(self, r, phi, z):
        """VMEC++'s MagneticFieldFunction: b_r, b_phi, b_z = f(r, phi, z)."""
        t = time.perf_counter()
        out = _biot_savart_cyl(np.ascontiguousarray(r, dtype=np.float64), np.ascontiguousarray(phi, dtype=np.float64),
                               np.ascontiguousarray(z, dtype=np.float64), self._g, self._dg, self._cur)
        with self._lock:
            self.seconds += time.perf_counter() - t
            self.calls += 1
            self.points += len(r)
        return out


def converge_nodes(coils, test_xyz, tol=1e-14, n0=128, n_max=1 << 15):
    """Smallest power-of-two node count (per coil) whose field at test_xyz agrees with the doubled rule to
    tol of |B| at every test point.  Returns (n, history)."""
    n = n0
    prev = CoilField(coils, n).xyz(test_xyz)
    hist = []
    while n < n_max:
        cur = CoilField(coils, 2 * n).xyz(test_xyz)
        err = float(np.max(np.linalg.norm(cur - prev, axis=1) / np.linalg.norm(cur, axis=1)))
        hist.append((n, err))
        if err < tol:
            return n, hist
        n, prev = 2 * n, cur
    return n, hist
