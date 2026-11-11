"""The solver's field engine: B, grad B and A of a Source at arbitrary points (JAX, float64,
GPU when present).

Fourier filaments are integrated by the N-point trapezoid rule, which converges
geometrically for a periodic analytic integrand: error ~ exp(-2 pi N alpha), alpha the
half-width of the strip in which the integrand r'(t) x (x - r(t)) / |x - r(t)|^3 is
analytic, about 0.46 d / max|r'| for a target at distance d.  N is chosen per coil for a
stated minimum target distance ``dmin``
and checked a posteriori: the even nodes form the N/2 rule, and |B_N - B_{N/2}| <= 1e-8 |B|
implies |B_N - B| ~ 1e-16 |B|; ``evaluate(..., err=True)`` returns that estimate per point.
Polylines and segments use the exact straight-segment formulas, dipoles the exact dipole
field, rings the elliptic-integral field in a cancellation-free form, axisymmetric
polygons Gauss quadrature of ring fields over the cross-section.

Everything is a pure function of the arrays in ``self.params`` (node positions and weights
derive from coil coefficients and currents through ``fourier_nodes``), so JAX can
differentiate the field with respect to the source parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from ..source import MU0, Source
from .. import fast_family as _so2
from . import kernels as K

try:                                   # numba CPU kernels (torux.field.cpu); the JAX kernels serve GPUs,
    from . import cpu as _cpu          # traced evaluations (autodiff) and the parts they do not cover
except Exception:                      # pragma: no cover
    _cpu = None

jax.config.update("jax_enable_x64", True)

MU0_4PI = MU0 / (4 * np.pi)


# ----------------------------------------------------------------------------- nodes
def fourier_nodes(coef, current, N):
    """Quadrature nodes of closed Fourier filaments at t_j = j / N (exact angle reduction).

    coef (n, 3, 2K+1), current (n,) -> r (n*N, 3), w (n*N, 3) with
    w = mu0/(4 pi) I r'(t_j) / N, and wh (n*N,) the half-rule factors.
    """
    coef = jnp.asarray(coef)
    n, _, nk = coef.shape
    Kord = (nk - 1) // 2
    j = np.arange(N)
    k = np.arange(1, Kord + 1)
    idx = np.outer(j, k) % N                      # exact reduction of k j mod N
    ang = 2 * np.pi * idx / N
    c = jnp.asarray(np.cos(ang))                  # (N, K)
    s = jnp.asarray(np.sin(ang))
    cc = coef[:, :, 2::2]                         # (n, 3, K)
    ss = coef[:, :, 1::2]
    kk = jnp.asarray(2 * np.pi * k)
    r = coef[:, None, :, 0] + jnp.einsum("tk,nak->nta", c, cc) + jnp.einsum("tk,nak->nta", s, ss)
    dr = jnp.einsum("tk,nak->nta", -s * kk, cc) + jnp.einsum("tk,nak->nta", c * kk, ss)
    w = dr * (MU0_4PI * jnp.asarray(current)[:, None, None] / N)
    wh = jnp.asarray(np.tile(np.where(j % 2 == 0, 2.0, 0.0), n))
    return r.reshape(-1, 3), w.reshape(-1, 3), wh


def nodes_needed(speed_max, dmin, eps=1e-17, strip=0.8):
    """Trapezoid nodes for relative error eps at distance dmin (geometric convergence; the integrand's
    strip of analyticity taken as strip * dmin / speed_max).  strip = 0.8 calibrated on 5 Nov against
    the reference evaluator on QUASR 10614, 1635364 and 19493 (design-surface points): the field is
    unchanged against the reference (5e-16 - 1.2e-15) from 0.46 to 0.9, while the N/2 rule's error (the
    a posteriori estimate, err_half) stays below 1e-11 -- so the N-point rule is at rounding level."""
    alpha = strip * dmin / speed_max
    N = math.log(2.0 / eps) / (2 * math.pi * alpha)
    return int(math.ceil(N))


def model_nodes(source, dmin, region_pts=None, region_h=0.0, region_margin=0.5, strip=0.8, eps=1e-17,
                node_multiple=8):
    """The per-coil node counts FieldEngine(source, dmin, region_pts=..., ...) gives its Fourier coils (nonzero
    current, source order) by the strip model, without building the nodes (the same integer arithmetic)."""
    tree = None
    if region_pts is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(region_pts, float))
    out = []
    for p in source.parts:
        if p.kind != "fourier":
            continue
        xs, dxs = p.points(np.arange(4096) / 4096)
        vmax = np.linalg.norm(dxs, axis=-1).max(axis=1)
        for i in range(p.n):
            if p.current[i] == 0.0:
                continue
            d_i = float(dmin)
            if tree is not None:
                if _so2.on("kdfast"):
                    # (fast_family 'kdfast') the same minimum by the exact coarse-to-fine search, as FieldEngine
                    di = min_sample_distance(tree, xs[i], vmax[i]) - 0.5 * float(region_h)
                else:
                    di = float(np.min(tree.query(xs[i], k=1)[0])) - 0.5 * float(region_h)
                d_i = max(float(dmin), region_margin * di)
            N = max(nodes_needed(vmax[i], d_i, eps, strip), 4 * p.order + 8)
            out.append(_round_up(N, node_multiple))
    return out


def fourier_nodes_np(coef, current, N):
    """fourier_nodes for one coil in numpy (the same formulas; no JAX dispatch, for measurements: equal to
    rounding, not bit for bit)."""
    K_ = (coef.shape[1] - 1) // 2
    j = np.arange(N)
    k = np.arange(1, K_ + 1)
    ang = 2 * np.pi * (np.outer(j, k) % N) / N
    c, s = np.cos(ang), np.sin(ang)
    cc, ss = coef[:, 2::2], coef[:, 1::2]
    r = coef[:, 0][None, :] + c @ cc.T + s @ ss.T
    dr = (-s * (2 * np.pi * k)) @ cc.T + (c * (2 * np.pi * k)) @ ss.T
    w = dr * (MU0_4PI * float(current) / N)
    wh = np.where(j % 2 == 0, 2.0, 0.0)
    return np.ascontiguousarray(r), np.ascontiguousarray(w), wh


def measured_nodes(source, N_model, region_pts, region_h=0.0, eps=1e-18, kclose=64, rate_floor=1e-6):
    """Per Fourier coil (nonzero current, source order): the node count its own measured convergence asks for
    on a region (fast_design 'nodes'; the refinement's torus).  For coil c with the model's N_c nodes, the half
    rule's relative error e at the kclose region points closest to it, e = max |B_N - B_N/2| / |B_N| (its own field),
    measures the trapezoid rule's geometric rate there: e = A exp(-pi N alpha), the N'-rule's error
    A exp(-2 pi N' alpha) = A^(1 - 2N'/N) e^(2N'/N); the rate is scaled to the closest point a target between
    the samples may reach (distance less region_h / 2: alpha is proportional to the distance), and
    N' = N ln(eps) / (2 ln e) puts the N'-rule at eps relative to the coil's own field (its share of the total
    is smaller still).  e at rounding (the half rule already exact) overstates the error, so N' is then larger
    than needed, never smaller.  Returns (N', info) with N' <= N_c (never more nodes than the model's) and
    N_c kept where the half rule is not in its geometric regime (e > rate_floor)."""
    from scipy.spatial import cKDTree
    region_pts = np.asarray(region_pts, float)
    tree = cKDTree(region_pts)
    out, info = [], []
    k = 0
    for p in source.parts:
        if p.kind != "fourier":
            continue
        xs, _ = p.points(np.arange(4096) / 4096)          # all of this part's coils at once
        for i in range(p.n):
            if p.current[i] == 0.0:
                continue
            N = int(N_model[k]); k += 1
            # candidates: the region points nearest to some point of the coil (every 16th of 4096: spacing about
            # 1e-3 of the coil's length, far below its distance to the region); then their distance to the coil
            _, jj = tree.query(xs[i][::16], k=min(16, len(region_pts)))
            cand = np.unique(np.asarray(jj).ravel())
            d, _ = cKDTree(xs[i]).query(region_pts[cand], k=1)
            o_ = np.argsort(d)[:kclose]
            idx, d = cand[o_], d[o_]
            dc = float(d[0])
            x = region_pts[idx]
            r, w, wh = fourier_nodes_np(p.coef[i], p.current[i], N)
            if _cpu is not None and cpu_kernels_wanted():
                o = _cpu.filament(x, r, w, wh, False, False)
                B, Bh = o[:, 0:3], o[:, 15:18]
            else:
                B, _, _, Bh = (np.asarray(v) for v in K.filament_nodes(jnp.asarray(x), jnp.asarray(r), jnp.asarray(w),
                                                                       jnp.asarray(wh), False, False))
            e = float(np.max(np.linalg.norm(B - Bh, axis=1) / np.linalg.norm(B, axis=1)))
            e = min(max(e, 1e-300), 1.0)
            scale = max(dc - 0.5 * float(region_h), 0.0) / dc if dc > 0 else 0.0
            lnE = math.log(e) * scale
            if not (lnE < math.log(rate_floor)):
                Nn = N
            else:
                Nn = min(N, int(math.ceil(N * math.log(eps) / (2.0 * lnE))))
            out.append(Nn)
            info.append(dict(N_model=N, N=Nn, e_half=e, d=dc, scale=scale))
    return out, info


def coil_samples(source, n=4096):
    """[(part, i, xs (n, 3), dxs (n, 3))] for the Fourier coils with current, in source order (t_j = j / n)."""
    out = []
    for p in source.parts:
        if p.kind != "fourier":
            continue
        xs, dxs = p.points(np.arange(n) / n)
        for i in range(p.n):
            if p.current[i] != 0.0:
                out.append((p, i, xs[i], dxs[i]))
    return out


def family_nodes(source, N_model, region_pts, eps=5e-16, kclose=64, ratio=1.25, node_multiple=8, bnorm=None,
                 upper=True, samples=None):
    """Per Fourier coil (nonzero current, source order): the fewest trapezoid nodes at which the coil's field at the
    kclose region points closest to it is within eps of its field with the model's N_c nodes, pointwise relative to
    the whole source's |B| there (bnorm(x) -> |B| at points x; None: the coil's own largest |B| there) --
    torux.fast_newton 'fmeas': the family's stepping-stone tori lie inside the region, farther from every coil, where the
    trapezoid error is smaller still.  Candidates from max(4 order + 8, 16) up by factors of ``ratio`` (rounded up to
    node_multiple); a candidate counts only if the next one also meets eps (the trapezoid error decays geometrically
    once the rule resolves the integrand, not before); upper: the larger of the two (one step of margin: with
    ~10 nodes per coil harmonic the error is not yet monotone in the target's distance, CoilStellaration DKKU9Heg...);
    never more than N_c.  The errors are computed, not modelled: N_c is the strip model's count for this region (at
    rounding: its own half rule is) and the reference here.  These counts are a proposal: torux.family certifies the
    engine built from them on the whole region (Continuation._fmeas_certify).  Returns (counts, info)."""
    from scipy.spatial import cKDTree
    region_pts = np.asarray(region_pts, float)
    tree = cKDTree(region_pts)
    use_cpu = _cpu is not None and cpu_kernels_wanted()

    def bfield(x, r, w, wh):
        if use_cpu:
            return _cpu.filament(x, r, w, wh, False, False)[:, 0:3]
        return np.asarray(K.filament_nodes(jnp.asarray(x), jnp.asarray(r), jnp.asarray(w), jnp.asarray(wh), False,
                                           False)[0])
    out, info = [], []
    coils = []
    for p, i, xs_i, dxs_i in (samples if samples is not None else coil_samples(source)):
        _, jj = tree.query(xs_i[::16], k=min(16, len(region_pts)))
        cand = np.unique(np.asarray(jj).ravel())
        d, jc = cKDTree(xs_i).query(region_pts[cand], k=1)
        # the trapezoid error at a target follows the strip half-width d / |r'(t*)| (t* its nearest coil point),
        # not d alone: the kclose targets with the narrowest strips, and the kclose / 2 nearest
        alpha = d / np.maximum(np.linalg.norm(dxs_i[jc], axis=1), 1e-300)
        o_ = np.unique(np.concatenate([np.argsort(alpha)[:kclose], np.argsort(d)[:kclose // 2]]))
        coils.append((p, i, region_pts[cand[o_]], d[o_]))
    # the whole source's |B| at every coil's closest points: one evaluation
    if bnorm is not None and coils:
        allx = np.concatenate([c[2] for c in coils])
        nb = np.maximum(np.asarray(bnorm(allx), float).reshape(-1), 1e-300)
        offs = np.cumsum([0] + [len(c[2]) for c in coils])
    for k, (p, i, x, d) in enumerate(coils):
        Nc = int(N_model[k])
        ref = bfield(x, *fourier_nodes_np(p.coef[i], p.current[i], Nc))
        if bnorm is not None:
            scale = nb[offs[k]:offs[k + 1]]
        else:
            scale = max(float(np.max(np.linalg.norm(ref, axis=1))), 1e-300)
        seq = []
        N = _round_up(max(4 * p.order + 8, 16), node_multiple)
        while N < Nc:
            seq.append(N)
            N = _round_up(int(math.ceil(ratio * N)), node_multiple)
        seq.append(Nc)

        def err(N):
            if N >= Nc:
                return 0.0
            return float(np.max(np.linalg.norm(bfield(x, *fourier_nodes_np(p.coef[i], p.current[i], N)) - ref,
                                               axis=1) / scale))
        Nf, ef = Nc, 0.0
        e_next = None
        for j in range(len(seq) - 1):
            e_j = e_next if e_next is not None else err(seq[j])
            e_next = None
            if e_j <= eps:
                e_next = err(seq[j + 1])
                if e_next <= eps:
                    Nf, ef = (seq[j + 1], e_next) if upper else (seq[j], e_j)
                    break
        out.append(Nf)
        info.append(dict(N_model=Nc, N=Nf, e=ef, d=float(np.min(d))))
    return out, info


def min_sample_distance(tree, pts, vmax, stride=16):
    """min over the closed curve's samples pts (n, 3) at t_j = j / n of their distance to the tree's points, exactly
    as np.min(tree.query(pts)[0]) (fast_family 'kdfast'), with about n / stride + a few stride queries: consecutive
    samples are at most h = max|r'| / n apart (vmax: the samples' largest |r'|, taken 1.5x), so a sample within
    stride / 2 steps of coarse sample k is at least d_k - (stride / 2) h from the region; the minimiser therefore
    lies next to a coarse sample with d_k <= min_k d_k + (stride / 2) h, and only those neighbourhoods are queried."""
    n = len(pts)
    if n <= 4 * stride or not np.isfinite(vmax) or vmax <= 0:
        return float(np.min(tree.query(pts, k=1)[0]))
    kc = np.arange(0, n, stride)
    dc = tree.query(pts[kc], k=1)[0]
    h = 1.5 * float(vmax) / n
    near = kc[dc <= float(np.min(dc)) + 0.5 * stride * h * (1 + 1e-12) + 1e-300]
    off = np.arange(-(stride // 2), stride // 2 + 1)
    cand = np.unique((near[:, None] + off[None, :]) % n)
    return float(min(np.min(dc), np.min(tree.query(pts[cand], k=1)[0])))


def _round_up(n, m):
    return ((n + m - 1) // m) * m


def gauss_triangle(order):
    """Gauss quadrature on the reference triangle (0,0),(1,0),(0,1) by collapsed tensor
    Gauss-Legendre x Gauss-Jacobi(1,0) (Duffy): exact for polynomials of degree 2*order-1."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    # Gauss-Jacobi alpha=1 via Golub-Welsch
    from scipy.special import roots_jacobi
    xj, wj = roots_jacobi(order, 1.0, 0.0)
    u = 0.5 * (xg + 1.0)
    v = 0.5 * (xj + 1.0)          # weight (1 - xj) -> collapsed coordinate
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wg, wj) * 0.125
    # map: x = u (1 - v), y = v  with Jacobian (1 - v); Gauss-Jacobi(1,0) carries (1 - xj) = 2(1 - v)
    X = U * (1.0 - V)
    Y = V
    return np.stack([X.ravel(), Y.ravel()], -1), W.ravel()


def _seg_dist(Q, A, B):
    """Distances (k,) from points Q (k, 2) to the segment A-B."""
    d = B - A
    L2 = float(d @ d)
    t = np.clip(((Q - A) @ d) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(len(Q))
    return np.linalg.norm(Q - (A + t[:, None] * d), axis=1)


def _polygon_distance(P, Q):
    """Smallest distance from points Q (k, 2) to the boundary of polygon P (closed, (v, 2))."""
    Q = np.atleast_2d(np.asarray(Q, float))
    best = np.full(len(Q), np.inf)
    for j in range(len(P)):
        best = np.minimum(best, _seg_dist(Q, P[j], P[(j + 1) % len(P)]))
    return float(np.min(best))


def graded_triangles(P, D, kappa=0.5, max_level=40):
    """Triangles covering polygon P, split (midpoint 4-split) until each one's longest edge h is at most
    kappa (depth + D), depth a lower bound of its distance to the polygon boundary.  A target at least D
    outside the conductor is then at least depth + D from every triangle (the segment to it crosses the
    boundary), i.e. at least h / kappa: every triangle's Gauss rule sees its integrand analytic in a
    neighbourhood kappa^-1 times its own size, whatever the target.  kappa calibrated against the
    double-double reference."""
    P = np.asarray(P, float)
    out = []
    stack = [(P[t[0]], P[t[1]], P[t[2]], 0) for t in triangulate_polygon(P)]
    while stack:
        a, b, c, lev = stack.pop()
        h = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
        # the distance to the boundary is 1-Lipschitz: the vertices' smallest depth less h bounds it below
        depth = max(0.0, _polygon_distance(P, np.stack([a, b, c])) - h) if lev < max_level else np.inf
        if lev < max_level and h > kappa * (depth + D):
            ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
            stack += [(a, ab, ca, lev + 1), (ab, b, bc, lev + 1), (ca, bc, c, lev + 1), (ab, bc, ca, lev + 1)]
        else:
            out.append((a, b, c))
    return out


def triangulate_polygon(P):
    """Ear-clipping triangulation of a simple polygon P (k, 2); returns (m, 3) vertex indices."""
    P = np.asarray(P, float)
    idx = list(range(len(P)))
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if area < 0:
        idx = idx[::-1]
    tris = []

    def is_ear(i0, i1, i2, poly):
        a, b, c = P[i0], P[i1], P[i2]
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
            return False
        for j in poly:
            if j in (i0, i1, i2):
                continue
            p = P[j]
            d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
            d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
            if d1 >= 0 and d2 >= 0 and d3 >= 0:
                return False
        return True

    poly = idx[:]
    guard = 0
    while len(poly) > 3 and guard < 10000:
        guard += 1
        for k in range(len(poly)):
            i0, i1, i2 = poly[k - 1], poly[k], poly[(k + 1) % len(poly)]
            if is_ear(i0, i1, i2, poly):
                tris.append((i0, i1, i2))
                poly.pop(k)
                break
        else:
            raise ValueError("polygon triangulation failed (not simple?)")
    tris.append(tuple(poly))
    return np.array(tris)


# ----------------------------------------------------------------------------- shared core
# Element arrays are padded to bucket sizes with exactly-zero contributions, and the evaluation is a
# module-level function of (params, x) with only shapes and part kinds static, so one compiled
# program serves every source with the same buckets.
CHUNK = {"r": 1024, "sa": 256, "dp": 1024, "rR": 64}
_FAR = 1.0e3


def cpu_kernels_wanted():
    """The engine evaluates concrete points with the numba kernels when JAX runs on the CPU
    (env TORUX_CPU_KERNELS=0 switches this off)."""
    import os
    if _cpu is None or os.environ.get("TORUX_CPU_KERNELS", "1") == "0":
        return False
    if _so2.on("cudafil") and _cuda_kernel() is not None:
        # (fast_family 'cudafil') a card with the hand-written filament kernel: the host path (numba kernels for
        # the other parts, the torus Newton in numpy / scipy) with the filament sums on the card
        return True
    try:
        return jax.default_backend() == "cpu"
    except Exception:
        return False


def _cuda_kernel():
    """torux.field.cuda_fil.CudaFilament on this process's card, or None."""
    try:
        from .cuda_fil import CudaFilament
    except Exception:  # noqa: BLE001
        return None
    return CudaFilament.get()


_THREADS_SET = []


def _bucket(n, chunk):
    """Smallest chunk * 2^k >= n (n > 0); 0 stays 0."""
    if n == 0:
        return 0
    b = chunk
    while b < n:
        b *= 2
    return b


def _pad_rows(a, n, fill):
    a = np.asarray(a, dtype=np.float64)
    if a.shape[0] == n:
        return a
    pad = np.broadcast_to(np.asarray(fill, dtype=np.float64), (n - a.shape[0],) + a.shape[1:])
    return np.concatenate([a, pad], axis=0)


def _scan_sum(fn, x, arrays, chunk, nout):
    """Sum of fn(x, *chunk_arrays) over chunks of the element arrays (lax.scan: small graphs)."""
    n = arrays[0].shape[0]
    nch = n // chunk
    resh = [a.reshape((nch, chunk) + a.shape[1:]) for a in arrays]

    def body(carry, sl):
        out = fn(x, *sl)
        return tuple(c + o for c, o in zip(carry, out)), None
    init = tuple(jnp.zeros(sh, x.dtype) for sh in nout)
    res, _ = jax.lax.scan(body, init, tuple(resh))
    return res


@partial(jax.jit, static_argnames=("kinds", "need_grad", "need_A"))
def eval_core(params, x, kinds, need_grad=True, need_A=True):
    """B, A, G = dB_a/dx_b and the half-rule B of every non-analytic part, at x (T, 3).
    kinds = (fourier, segments, dipoles, rings, toroidal) presence flags."""
    has_f, has_s, has_d, has_r, has_t = kinds
    T = x.shape[0]
    B = jnp.zeros((T, 3)); A = jnp.zeros((T, 3)); G = jnp.zeros((T, 3, 3)); Bh = jnp.zeros((T, 3))
    if has_f:
        def fk(xx, r, w, wh):
            b, a, g, bh = K.filament_nodes(xx, r, w, wh, need_grad, need_A)
            return b, a, g, bh
        b, a, g, bh = _scan_sum(fk, x, (params["r"], params["w"], params["wh"]), CHUNK["r"],
                                ((T, 3), (T, 3), (T, 3, 3), (T, 3)))
        B, A, G, Bh = B + b, A + a, G + g, Bh + bh
    others = []
    if has_s:
        others.append(((params["sa"], params["sb"], params["sw"]), CHUNK["sa"],
                       lambda y, a_, b_, w_: K.segments(y, a_, b_, w_)))
    if has_d:
        others.append(((params["dp"], params["dm"]), CHUNK["dp"], lambda y, p_, m_: K.dipoles(y, p_, m_)))
    if has_r:
        others.append(((params["rR"], params["rZ"], params["rw"]), CHUNK["rR"],
                       lambda y, R_, Z_, w_: K.rings(y, R_, Z_, w_)))
    for arrays, ch, fn in others:
        def fk(xx, *sl, fn=fn):
            b, a = fn(xx, *sl)
            if need_grad:
                g = jax.vmap(jax.jacfwd(lambda y: fn(y[None, :], *sl)[0][0]))(xx)
            else:
                g = jnp.zeros((xx.shape[0], 3, 3))
            return b, a, g
        b, a, g = _scan_sum(fk, x, arrays, ch, ((T, 3), (T, 3), (T, 3, 3)))
        B, A, G, Bh = B + b, A + a, G + g, Bh + b
    if has_t:
        def tor(y):
            R2 = y[:, 0] ** 2 + y[:, 1] ** 2
            Bt = params["RB"] / R2
            Bv = jnp.stack([-y[:, 1] * Bt, y[:, 0] * Bt, jnp.zeros_like(Bt)], -1)
            Av = jnp.stack([jnp.zeros_like(Bt), jnp.zeros_like(Bt), -0.5 * params["RB"] * jnp.log(R2)], -1)
            return Bv, Av
        b, a = tor(x)
        B, A, Bh = B + b, A + a, Bh + b
        if need_grad:
            G = G + jax.vmap(jax.jacfwd(lambda y: tor(y[None, :])[0][0]))(x)
    if not has_f:
        Bh = B
    return B, A, G, Bh


# ----------------------------------------------------------------------------- engine
@dataclass
class EngineInfo:
    dmin: float
    N_per_coil: np.ndarray
    n_nodes: int
    n_segments: int
    n_dipoles: int
    n_rings: int


class FieldEngine:
    """Vacuum field of a Source.  ``evaluate(x)`` -> dict(B, A, G[, err_half]) at points x (T,3).

    dmin: smallest distance from any filament at which full accuracy is wanted (targets
    closer are still evaluated; ``err_half`` shows the loss).  Element arrays are padded to
    shared bucket sizes so that compiled programs are reused across sources.
    """

    def __init__(self, source: Source, dmin: float = 0.05, eps: float = 1e-17, node_multiple=None,
                 axipoly_order: int = 8, target_chunk: int = 4096, region_pts=None, region_margin=0.5, strip=0.8,
                 region_h=0.0, axipoly_kappa=0.5, nodes_per_coil=None, nodes_np=False):
        """region_pts: optional points (k, 3) of the region where the field will be needed (e.g. a design
        surface); each Fourier coil then gets the nodes its own distance to that region needs (times
        region_margin, the share of that distance targets may approach), never fewer than for dmin.
        region_h: the spacing of the region sample (the sampled distance less h/2 is used, so a coil
        approaching the region between samples is not credited with more distance than it has).
        nodes_per_coil: optional node counts of the Fourier coils (those with nonzero current, in source
        order) replacing the strip model's (measured_nodes; still >= 4 order + 8, rounded to node_multiple).
        nodes_np: those coils' nodes built by fourier_nodes_np (numpy; JAX's op-by-op fourier_nodes compiles
        once per new node count, seconds for a coil set whose counts are all new)."""
        self.source = source
        self.dmin = float(dmin)
        from .. import fast_design as _so
        nodes_np = nodes_np or _so.on("npnodes")       # (fast_design 'npnodes') numpy nodes for every coil
        if node_multiple is None:
            # per-coil node counts rounded up to a multiple (even: the half rule); 64 keeps the JAX
            # buckets few (compiled programs shared), the CPU kernels need no padding
            node_multiple = 8 if cpu_kernels_wanted() else 64
        self.region_pts = None if region_pts is None else np.asarray(region_pts, float)
        self.target_chunk = target_chunk
        rs, ws, whs, Ns = [], [], [], []
        seg_a, seg_b, seg_w = [], [], []
        dip_p, dip_m = [], []
        ring_R, ring_Z, ring_w = [], [], []
        self.RB = 0.0
        tree = None
        if self.region_pts is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.region_pts)
        kf = 0
        for p in source.parts:
            if p.kind == "fourier":
                xs, dxs = p.points(np.arange(4096) / 4096)
                vmax = np.linalg.norm(dxs, axis=-1).max(axis=1)
                for i in range(p.n):
                    if p.current[i] == 0.0:
                        continue
                    if nodes_per_coil is not None:
                        N = _round_up(max(int(nodes_per_coil[kf]), 4 * p.order + 8), node_multiple)
                        kf += 1
                        if nodes_np:
                            r, w, wh = fourier_nodes_np(p.coef[i], p.current[i], N)
                        else:
                            r, w, wh = fourier_nodes(p.coef[i:i + 1], p.current[i:i + 1], N)
                        rs.append(np.asarray(r)); ws.append(np.asarray(w)); whs.append(np.asarray(wh)); Ns.append(N)
                        continue
                    d_i = self.dmin
                    if tree is not None:
                        if _so2.on("kdfast"):
                            # (fast_family 'kdfast') the same minimum by a coarse-to-fine search
                            di = min_sample_distance(tree, xs[i], vmax[i]) - 0.5 * float(region_h)
                        else:
                            di = float(np.min(tree.query(xs[i], k=1)[0])) - 0.5 * float(region_h)
                        d_i = max(self.dmin, region_margin * di)
                    N = max(nodes_needed(vmax[i], d_i, eps, strip), 4 * p.order + 8)
                    N = _round_up(N, node_multiple)
                    if nodes_np:
                        r, w, wh = fourier_nodes_np(p.coef[i], p.current[i], N)
                    else:
                        r, w, wh = fourier_nodes(p.coef[i:i + 1], p.current[i:i + 1], N)
                    rs.append(np.asarray(r)); ws.append(np.asarray(w)); whs.append(np.asarray(wh)); Ns.append(N)
            elif p.kind == "polyline":
                sg = p.as_segments()
                seg_a.append(sg.a); seg_b.append(sg.b); seg_w.append(MU0_4PI * sg.current)
            elif p.kind == "segments":
                seg_a.append(p.a); seg_b.append(p.b); seg_w.append(MU0_4PI * p.current)
            elif p.kind == "dipoles":
                dip_p.append(p.pos); dip_m.append(MU0_4PI * p.moment)
            elif p.kind == "rings":
                ring_R.append(p.R); ring_Z.append(p.Z); ring_w.append(MU0_4PI * p.current)
            elif p.kind == "axipoly":
                ref, wref = gauss_triangle(axipoly_order)
                for i in range(p.n):
                    P = p.vertices[p.start[i]:p.start[i + 1]]
                    # targets are at least D from the conductor: dmin, or the region's own distance
                    D = self.dmin
                    if self.region_pts is not None:
                        Rr = np.hypot(self.region_pts[:, 0], self.region_pts[:, 1])
                        D = max(D, region_margin * (_polygon_distance(P, np.stack([Rr, self.region_pts[:, 2]], -1))
                                                    - 0.5 * float(region_h)))
                    pts, wts = [], []
                    for (a, b, c) in graded_triangles(P, D, kappa=axipoly_kappa):
                        J = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                        pts.append(a + ref[:, :1] * (b - a) + ref[:, 1:] * (c - a))
                        wts.append(wref * J)
                    pts = np.concatenate(pts); wts = np.concatenate(wts)
                    area = wts.sum()
                    ring_R.append(pts[:, 0]); ring_Z.append(pts[:, 1])
                    ring_w.append(MU0_4PI * p.current[i] * wts / area)
            elif p.kind == "toroidal":
                self.RB += p.RB
            else:
                raise ValueError(f"unknown part kind {p.kind}")
        cat = lambda L, shape: np.concatenate(L) if L else np.zeros(shape)
        r, w, wh = cat(rs, (0, 3)), cat(ws, (0, 3)), cat(whs, (0,))
        sa, sb, sw = cat(seg_a, (0, 3)), cat(seg_b, (0, 3)), cat(seg_w, (0,))
        dp, dm = cat(dip_p, (0, 3)), cat(dip_m, (0, 3))
        rR, rZ, rw = cat(ring_R, (0,)), cat(ring_Z, (0,)), cat(ring_w, (0,))
        self.counts = dict(r=len(r), sa=len(sa), dp=len(dp), rR=len(rR))
        nr, ns, nd, nq = (_bucket(len(r), CHUNK["r"]), _bucket(len(sa), CHUNK["sa"]),
                          _bucket(len(dp), CHUNK["dp"]), _bucket(len(rR), CHUNK["rR"]))
        far = np.array([_FAR, 0.0, 0.0])
        self.params = dict(
            r=jnp.asarray(_pad_rows(r, nr, far)), w=jnp.asarray(_pad_rows(w, nr, 0.0)),
            wh=jnp.asarray(_pad_rows(wh, nr, 0.0)),
            sa=jnp.asarray(_pad_rows(sa, ns, far)), sb=jnp.asarray(_pad_rows(sb, ns, far + np.array([0, 0, 1.0]))),
            sw=jnp.asarray(_pad_rows(sw, ns, 0.0)),
            dp=jnp.asarray(_pad_rows(dp, nd, far)), dm=jnp.asarray(_pad_rows(dm, nd, 0.0)),
            rR=jnp.asarray(_pad_rows(rR, nq, 1.0)), rZ=jnp.asarray(_pad_rows(rZ, nq, _FAR)),
            rw=jnp.asarray(_pad_rows(rw, nq, 0.0)),
            RB=jnp.asarray(self.RB),
        )
        self.kinds = (nr > 0, ns > 0, nd > 0, nq > 0, self.RB != 0.0)
        self.has_tor = self.RB != 0.0
        self.cpu_kernels = cpu_kernels_wanted()
        if self.cpu_kernels:
            if not _THREADS_SET:
                _THREADS_SET.append(_cpu.set_threads())
            self._np = dict(r=r, w=w, wh=wh, sa=sa, sb=sb, sw=sw, dp=dp, dm=dm)
            # what the numba kernels do not cover goes through the JAX core
            self._rest_kinds = (False, False, False, nq > 0, self.RB != 0.0)
        self.analytic = []
        for spec in source.meta.get("analytic_parts", []) or []:
            from .analytic import analytic_part
            Rc = float(source.meta.get("analytic_R_ref", 1.0))
            self.analytic.append(analytic_part(spec, R_c=Rc))
        self.info = EngineInfo(self.dmin, np.array(Ns), len(r), len(sa), len(dp), len(rR))

    # ------------------------------------------------------------------ evaluation
    def _core(self, params, x, need_grad, need_A, kinds=None):
        kinds = self.kinds if kinds is None else kinds
        if any(kinds):
            B, A, G, Bh = eval_core(params, x, kinds, need_grad, need_A)
        else:
            T = x.shape[0]
            B = jnp.zeros((T, 3)); A = jnp.zeros((T, 3)); G = jnp.zeros((T, 3, 3)); Bh = jnp.zeros((T, 3))
        for fa in self.analytic:
            b, a, g = fa(x, need_grad, need_A)
            B, A, G, Bh = B + b, A + a, G + g, Bh + b
        return B, A, G, Bh

    def evaluate(self, x, grad=True, A=True, err=False, chunk=None, params=None):
        """B, A and grad B at points x (T, 3) (numpy or jax arrays).  Returns jax arrays.

        Targets are processed in chunks of ``target_chunk`` points (the last chunk padded), so
        the compiled program depends on the chunk size only.  Concrete points on a CPU go to the
        numba kernels (torux.field.cpu); traced points (inside jit / autodiff) to the JAX kernels."""
        if getattr(self, "cpu_kernels", False) and params is None:
            try:
                xn = np.asarray(x, dtype=np.float64)
            except Exception:          # a tracer: the JAX path
                xn = None
            if xn is not None:
                return self._evaluate_cpu(xn, grad, A, err, chunk)
        return self._evaluate_jax(x, grad, A, err, chunk, params)

    def _evaluate_cpu(self, x, grad, A, err, chunk):
        x = x.reshape(-1, 3)
        T = x.shape[0]
        B = np.zeros((T, 3)); Av = np.zeros((T, 3)); G = np.zeros((T, 3, 3)); Bh = np.zeros((T, 3))
        P = self._np
        if len(P["r"]):
            ck = _cuda_kernel() if _so2.on("cudafil") else None
            h = None
            if ck is not None:
                h = self.__dict__.get("_cuda_h")
                if h is None:
                    import weakref
                    h = self._cuda_h = ck.upload(P["r"], P["w"], P["wh"]) or False
                    if h:
                        weakref.finalize(self, ck.release, h)
            o = ck.evaluate(h, x, grad, A) if h else _cpu.filament(x, P["r"], P["w"], P["wh"], grad, A)
            B += o[:, 0:3]; Bh += o[:, 15:18]
            if A:
                Av += o[:, 3:6]
            if grad:
                G += o[:, 6:15].reshape(T, 3, 3)
        for key, fn in (("sa", _cpu.segments), ("dp", _cpu.dipoles)):
            if len(P[key]):
                o = fn(x, P["sa"], P["sb"], P["sw"], grad, A) if key == "sa" else fn(x, P["dp"], P["dm"], grad, A)
                B += o[:, 0:3]; Bh += o[:, 0:3]
                if A:
                    Av += o[:, 3:6]
                if grad:
                    G += o[:, 6:15].reshape(T, 3, 3)
        if any(self._rest_kinds) or self.analytic:
            o = self._evaluate_jax(x, grad, A, False, chunk, None, kinds=self._rest_kinds)
            b = np.asarray(o["B"])
            B += b; Bh += b
            if A:
                Av += np.asarray(o["A"])
            if grad:
                G += np.asarray(o["G"])
        res = dict(B=jnp.asarray(B), A=jnp.asarray(Av), G=jnp.asarray(G))
        if err:
            res["err_half"] = jnp.asarray(np.linalg.norm(B - Bh, axis=-1) / np.linalg.norm(B, axis=-1))
        return res

    def _evaluate_jax(self, x, grad=True, A=True, err=False, chunk=None, params=None, kinds=None):
        x = jnp.asarray(x, dtype=jnp.float64)
        p = self.params if params is None else params
        T = x.shape[0]
        chunk = chunk or self.target_chunk
        if T <= chunk:
            # pad small target sets to a multiple of 256 (fewer distinct compiled shapes)
            Tp = ((T + 255) // 256) * 256
            xp = x if Tp == T else jnp.concatenate([x, jnp.broadcast_to(x[-1:], (Tp - T, 3))])
            B, Av, G, Bh = self._core(p, xp, grad, A, kinds)
            B, Av, G, Bh = B[:T], Av[:T], G[:T], Bh[:T]
        else:
            outs = []
            for lo in range(0, T, chunk):
                xc = x[lo:lo + chunk]
                n = xc.shape[0]
                if n < chunk:
                    xc = jnp.concatenate([xc, jnp.broadcast_to(xc[-1:], (chunk - n, 3))])
                o = self._core(p, xc, grad, A, kinds)
                outs.append(tuple(v[:n] for v in o))
            B = jnp.concatenate([o[0] for o in outs]); Av = jnp.concatenate([o[1] for o in outs])
            G = jnp.concatenate([o[2] for o in outs]); Bh = jnp.concatenate([o[3] for o in outs])
        res = dict(B=B, A=Av, G=G)
        if err:
            nb = jnp.linalg.norm(B, axis=-1)
            res["err_half"] = jnp.linalg.norm(B - Bh, axis=-1) / nb      # error of the N/2 rule
        return res

    def B(self, x):
        return self.evaluate(x, grad=False, A=False)["B"]


class EngineStub(FieldEngine):
    """Stand-in for a FieldEngine inside shared jitted functions: traced params, static kinds."""

    def __init__(self, params, kinds, target_chunk):
        self.params = params
        self.kinds = kinds
        self.target_chunk = target_chunk
        self.analytic = []
        self.cpu_kernels = False
