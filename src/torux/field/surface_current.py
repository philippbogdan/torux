"""Exterior Biot-Savart field of a toroidal current sheet."""

import numpy as np
from numba import njit

from ..source import MU0, rotation_z

TAU = 2 * np.pi
_MAX_CACHE_DEPTH = 3
_PANEL_CACHE_BYTES = 32 * 1024**2
_MAX_DEPTH = 16
_INTERP_WEIGHTS = np.array([-1., 7., -21., 35., -35., 21., -7., 1.])


class SurfaceCurrentDomainError(ValueError):
    """A target is outside the supported exterior domain."""

    def __init__(self, message, reason="boundary"):
        super().__init__(message)
        self.reason = reason


def _spectrum(part):
    """Complex coefficients of R, Z and the covariant current in exp(i*(m*theta - n*nfp*phi))."""
    theta, phi = part.modes[:, 0], -part.nfp * part.modes[:, 1]

    def series(c, s):
        return (0 if c is None else c) + (0j if s is None else -1j * s)

    r, z = series(part.rmnc, part.rmns), series(part.zmnc, part.zmns)
    if part.covariant is None:
        kappa = series(part.kappac, part.kappas)
        ct, cp = 1j * theta * kappa, 1j * phi * kappa
    else:
        ct, cp = (series(*v) for v in part.covariant)
    coeff = np.column_stack([r, z, 1j * theta * r, 1j * theta * z, 1j * phi * r, 1j * phi * z, ct, cp])
    return coeff


def _tabulate(part, coeff, shape):
    """Precompute smooth geometry and current one real transform at a time."""
    m, n = part.modes.T
    direct, conjugate = n <= 0, n >= 0
    indices = (m[direct] % shape[0], -n[direct])
    mirrors = (-m[conjugate] % shape[0], n[conjugate])

    def samples(i):
        # The omitted half of the toroidal spectrum follows by conjugation.
        # At n=0 both contributions lie in the stored half.
        amps = np.zeros((shape[0], shape[1]//2 + 1), complex)
        np.add.at(amps, indices, coeff[direct, i] / 2)
        np.add.at(amps, mirrors, coeff[conjugate, i].conj() / 2)
        values = np.fft.irfft2(amps, s=shape, axes=(0, 1))
        values *= shape[0] * shape[1]
        return values

    table = np.empty((*shape, 5))
    table[..., 0] = samples(0)
    table[..., 1] = samples(1)
    ct, cp = samples(6), samples(7)
    if part.covariant is None:
        ct += part.I / TAU
        cp += part.G / TAU
    # K dS = (kappa_phi r_theta - kappa_theta r_phi) dtheta dphi.
    # The integrand's current weight includes mu0 / (4 pi).
    scale = MU0 / (4 * np.pi)
    for component, theta, phi in ((2, 2, 4), (4, 3, 5)):
        values = samples(theta)
        values *= cp
        other = samples(phi)
        other *= ct
        values -= other
        table[..., component] = scale * values
    table[..., 3] = -scale * ct * table[..., 0]
    return table


def _cache_depth(roots, order):
    """Deepest complete level whose node and centre arrays fit the cap."""
    bytes_per_panel = 8 * (3 + 6 * order**2)
    depth = -1
    for level in range(_MAX_CACHE_DEPTH + 1):
        panels = roots * (4**(level + 1) - 1) // 3
        if panels * bytes_per_panel > _PANEL_CACHE_BYTES:
            break
        depth = level
    return depth


@njit(cache=True)
def _stencil(x):
    index = int(np.floor(x)) - 3
    offset = x - index
    weights = np.zeros(8)
    nearest = round(offset)
    if abs(offset - nearest) < 1e-13:
        weights[nearest] = 1.
    else:
        total = 0.
        for i in range(8):
            weights[i] = _INTERP_WEIGHTS[i] / (offset - i)
            total += weights[i]
        weights /= total
    return index, weights


@njit(cache=True)
def _nodes(tab, theta, phi, dt, dp, g, w, nfp):
    """Gauss nodes of one panel; interpolate source data, never the target field."""
    order = len(g)
    nt, np_, _ = tab.shape
    out = np.empty((order * order, 6))
    p_index = np.empty(order, np.int64)
    p_weights = np.empty((order, 8))
    pc, ps = np.cos(phi + dp * g), np.sin(phi + dp * g)
    for k in range(order):
        p_index[k], p_weights[k] = _stencil((phi + dp * g[k]) * (np_ * nfp / TAU))
    p_lo, p_hi = p_index[0], p_index[-1] + 8
    # Nearby panels have overlapping toroidal stencils. Reuse the poloidal
    # interpolation at each toroidal grid point needed by the panel.
    row = np.empty((p_hi - p_lo, 5))
    for j in range(order):
        it, wt = _stencil((theta + dt * g[j]) * nt / TAU)
        row.fill(0.)
        for a in range(8):
            ii = (it + a) % nt
            for b in range(p_hi - p_lo):
                v = tab[ii, (p_lo + b) % np_]
                for component in range(5):
                    row[b, component] += wt[a] * v[component]
        for k in range(order):
            ip = p_index[k] - p_lo
            r, z, wr, wp_, wz = 0., 0., 0., 0., 0.
            for b in range(8):
                h = p_weights[k, b]
                v = row[ip + b]
                r += h * v[0]
                z += h * v[1]
                wr += h * v[2]
                wp_ += h * v[3]
                wz += h * v[4]
            c, s = pc[k], ps[k]
            weight = dt * dp * w[j] * w[k]
            out[j * order + k] = (r * c, r * s, z, weight * (wr * c - wp_ * s),
                                  weight * (wr * s + wp_ * c), weight * wz)
    return out


@njit(cache=True)
def _centres(tab, g, w, ntheta, nphi, nfp, depth):
    rows = ntheta * nphi
    total = rows * (4 ** (depth + 1) - 1) // 3
    centres = np.empty((total, 3))
    nodes = np.empty((total, len(g)**2, 6))
    mid = np.array([.5])
    unit = np.array([1.])
    start = 0
    for level in range(depth + 1):
        a, b = ntheta * 2**level, nphi * 2**level
        dt, dp = TAU / a, TAU / (nfp * b)
        for i in range(a):
            for j in range(b):
                k = start + i * b + j
                theta, phi = i * dt, j * dp
                centres[k] = _nodes(tab, theta, phi, dt, dp, mid, unit, nfp)[0, :3]
                nodes[k] = _nodes(tab, theta, phi, dt, dp, g, w, nfp)
        start += a * b
    return centres, nodes


@njit(cache=True)
def _accumulate(out, target, nodes, grad, potential):
    for node in nodes:
        x, y, z = target[0] - node[0], target[1] - node[1], target[2] - node[2]
        r2 = x*x + y*y + z*z
        ri = 1 / np.sqrt(r2)
        ri3 = ri / r2
        wx, wy, wz = node[3], node[4], node[5]
        bx, by, bz = (wy*z - wz*y) * ri3, (wz*x - wx*z) * ri3, (wx*y - wy*x) * ri3
        out[0] += bx
        out[1] += by
        out[2] += bz
        if potential:
            out[3] += wx * ri
            out[4] += wy * ri
            out[5] += wz * ri
        if grad:
            a, b, c = -3*x / r2, -3*y / r2, -3*z / r2
            out[6] += a*bx
            out[7] += b*bx - wz*ri3
            out[8] += c*bx + wy*ri3
            out[9] += a*by + wz*ri3
            out[10] += b*by
            out[11] += c*by - wx*ri3
            out[12] += a*bz - wy*ri3
            out[13] += b*bz + wx*ri3
            out[14] += c*bz


@njit(cache=True)
def _quadrature(points, tab, centres, nodes, g, w, ntheta, nphi, nfp, radius, depth, grad, potential):
    """Integrate over one field period. Uncached panels use the same quadrature."""
    out = np.zeros((len(points), 15))
    failed = np.zeros(len(points), np.bool_)
    mid, unit = np.array([.5]), np.array([1.])
    nroot = ntheta * nphi
    stack = np.empty((1 + 3 * _MAX_DEPTH, 3), np.int64)
    for k, x in enumerate(points):
        for root in range(nroot):
            stack[0] = 0, root // nphi, root % nphi
            size = 1
            while size:
                size -= 1
                level, i, j = stack[size]
                scale = 2**level
                dt, dp = TAU / (ntheta * scale), TAU / (nfp * nphi * scale)
                theta, phi = i * dt, j * dp
                idx = nroot * (4**level - 1) // 3 + i * nphi * scale + j
                if level <= depth:
                    centre = centres[idx]
                else:
                    centre = _nodes(tab, theta, phi, dt, dp, mid, unit, nfp)[0, :3]
                delta = x - centre
                if np.dot(delta, delta) > (radius / scale)**2:
                    panel = nodes[idx] if level <= depth else _nodes(tab, theta, phi, dt, dp, g, w, nfp)
                    _accumulate(out[k], x, panel, grad, potential)
                elif level == _MAX_DEPTH:
                    failed[k] = True
                else:
                    for a in range(2):
                        for b in range(2):
                            stack[size] = level + 1, 2*i + a, 2*j + b
                            size += 1
    return out, failed


@njit(cache=True)
def _clearance(r, z, phi, boundary, nfp, margin):
    """Lower estimate of the exterior distance to the supplied cut, in metres."""
    nt, np_, _ = boundary.shape
    index = (phi * (np_ * nfp / TAU)) % np_
    j = min(int(index), np_ - 1)
    k = (j + 1) % np_
    f = index - j
    r0, z0 = (1-f) * boundary[-1, j] + f * boundary[-1, k]
    inside = False
    nearest = np.inf
    for i in range(nt):
        r1 = (1-f) * boundary[i, j, 0] + f * boundary[i, k, 0]
        z1 = (1-f) * boundary[i, j, 1] + f * boundary[i, k, 1]
        if (z0 > z) != (z1 > z) and r < r0 + (r1-r0) * (z-z0) / (z1-z0):
            inside = not inside
        dr, dz = r1-r0, z1-z0
        length2 = dr*dr + dz*dz
        u = ((r-r0)*dr + (z-z0)*dz) / length2 if length2 else 0.
        u = min(1., max(0., u))
        nearest = min(nearest, (r-r0-u*dr)**2 + (z-z0-u*dz)**2)
        r0, z0 = r1, z1
    return (-1 if inside else 1) * np.sqrt(nearest) - margin


class SheetField:
    def __init__(self, part, order, samples):
        if (not isinstance(order, (int, np.integer)) or order < 4
                or not isinstance(samples, (int, np.integer)) or samples < 32):
            raise ValueError("sheet_order and sheet_samples must be integers, at least 4 and 32")
        self.nfp = part.nfp
        m, n = np.abs(part.modes.T)
        self.ntheta = max(6, (int(m.max()) + 2) // 3)
        self.nphi = max(6, (int(n.max()) + 2) // 3, (16 + self.nfp - 1) // self.nfp)
        self.cache_depth = _cache_depth(self.ntheta * self.nphi, order)
        self.rotations = [rotation_z(TAU * j / self.nfp) for j in range(self.nfp)]
        coeff = _spectrum(part)
        shape = (max(samples, 12 * (int(m.max()) + 1)), max(samples, 12 * (int(n.max()) + 1)))
        self.table = _tabulate(part, coeff, shape)
        self.boundary = self.table[..., :2]
        r, z = self.boundary.transpose(2, 0, 1)
        area = np.sum(r * np.roll(z, -1, axis=0) - z * np.roll(r, -1, axis=0), axis=0)
        if np.any(area <= 0) or r.min() <= 0:
            raise ValueError("SurfaceCurrent requires positive R and counterclockwise (R, Z) cuts")
        amp = np.linalg.norm(np.abs(coeff[:, :2]), axis=1)
        # Linear interpolation error of each cut from the second-derivative bounds in R,Z.
        self.margin = float(((TAU / shape[0])**2 * np.dot(m*m, amp)
                             + (TAU / (self.nfp * shape[1]))**2 * np.dot((self.nfp*n)**2, amp)) / 8
                            + 32 * np.finfo(float).eps * r.max())
        speed_theta = np.dot(m, amp)
        speed_phi = np.dot(self.nfp * n, amp) + np.abs(coeff[:, 0]).sum()
        self.radius = 1.3 * np.pi * (speed_theta / self.ntheta + speed_phi / (self.nfp * self.nphi))
        x, w = np.polynomial.legendre.leggauss(order)
        self.g, self.w = (x + 1) / 2, w / 2
        self._panels = None

    def clearance(self, r, z, phi):
        """Conservative signed separation from the boundary in the toroidal cut."""
        return _clearance(r, z, phi, self.boundary, self.nfp, self.margin)

    def evaluate(self, points, grad, potential):
        for i, (x, y, z) in enumerate(points):
            if self.clearance(np.hypot(x, y), z, np.arctan2(y, x)) <= 0:
                raise SurfaceCurrentDomainError(f"target {i} is inside or too close to the surface current boundary")
        if self._panels is None:
            self._panels = _centres(self.table, self.g, self.w, self.ntheta, self.nphi, self.nfp, self.cache_depth)
        out = np.zeros((len(points), 15))
        for rotation in self.rotations:
            # Rotate each target into the cached period, then rotate the
            # vector and tensor integrals back to the original frame.
            local, failed = _quadrature(points @ rotation, self.table, *self._panels, self.g, self.w,
                                        self.ntheta, self.nphi, self.nfp, self.radius, self.cache_depth,
                                        grad, potential)
            if failed.any():
                raise SurfaceCurrentDomainError("target is too close for the surface current quadrature to resolve",
                                                "quadrature_limit")
            out[:, :3] += local[:, :3] @ rotation.T
            if potential:
                out[:, 3:6] += local[:, 3:6] @ rotation.T
            if grad:
                tensor = local[:, 6:].reshape(-1, 3, 3)
                out[:, 6:] += (rotation @ tensor @ rotation.T).reshape(-1, 9)
        return out
