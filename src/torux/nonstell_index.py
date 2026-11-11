"""Exact exclusion of toroidal flux surfaces by closed-line linking (theorem LINKBOX of torux.nonstell), with
validated LOST certificates on box boundaries -- or the exact residual of that proof.

    from torux import nonstell_index as nsi
    rec = nsi.certify_source(source)                 # JSON-serialisable: proved, residual cells, timings
    rec = nsi.certify_source(source, backend="jax")  # the same certificates, batched in JAX (GPU)
    nsi.scope_record(source)                         # what the theorem can cover (no run)

THEOREM (LINKBOX).  Source: closed filament coils C_k with currents I_k, no ToroidalField part.  Let T be a
toroidal flux surface (torux.nonstell DEF), V its closed solid torus, D = V n {phi = 0}; orient so that
B_phi > 0 on T (the other orientation is the same with B -> -B).
 (1) LEMMA A (torux/barrier.py): u = R B_phi = x B_y - y B_x is harmonic in the vacuum, so u > 0 on V.  The
     period map P: D -> D is a homeomorphism of the closed disc: it has a fixed point (Brouwer); with
     isolated fixed points their indices (in D) add up to +1 (Lefschetz-Hopf).
 (2) A fixed point lies on a closed field line g inside V, of toroidal winding 1, u > 0 along g.  Ampere for g
     alone: oint_g B.dl = int_g |B| dl > 0 = mu0 sum_k I_k Lk(g, C_k).  So g links a coil C_j, I_j != 0.
 (3) BOXES.  For a coil C_j of toroidal winding 0 let W_j = {R1 <= R <= R2, Z1 <= Z <= Z2, phi in an arc of
     length < 2 pi} (R1 > 0) be a compact box containing C_j (segment extents computed exactly; several coils
     may share a box).  W_j is contractible: if g missed W_j, C_j would be null-homotopic in W_j inside
     R^3 - g, so Lk(g, C_j) = 0; and g is not contained in W_j, because every closed curve in the
     contractible W_j (off the z axis) has toroidal winding 0.  Hence g meets the boundary of some W_j.
 (4) Every point z of every box boundary is certified LOST: its field line reaches u < 0 (z lies on no
     closed line with u > 0) and u > 0 (orientation -).  Then D contains no fixed point of P: contradiction.
     NO toroidal flux surface exists, for these currents.
 (5) Exact nfp-fold rotation symmetry (checked on the segment set here, independently of the loaders) maps
     boxes of symmetric coils onto each other and preserves lost-ness: one box per orbit is certified.
Not covered: coils of toroidal winding != 0 (PF-type rings: every surface spanning them meets the z axis,
and closed winding-1 lines may lie next to their wires), coils through the z axis, ToroidalField parts,
Fourier-curve and ring parts (cells implemented for polylines).  Two more certificates for points whose
lines complete toroidal turns are implemented for half-plane discs (certify_box_section: 'N' |P(z) - z| > 0,
'S' Krawczyk box of a regular saddle, index <= 0 in D also on its boundary); they are tested on model
fields but not yet wired into the box driver.

LOST certificate of a cell Q = conv(c + E_1..E_4) + ball(sag) (cert_cell / jax_batch_certifier):
the centre line g(t) is followed along +B and -B in the unnormalised time t (dx/dt = B) with its variational
matrix M (Dormand-Prince 5(4); field: the closed form of torux.reference.fast64 for straight segments).  For z in Q,
Phi_t(z) - g_n = M_n (z - c) + nu, |nu| <= E_n = N_n + e_x + e_M rho_Q with
    N_(n+1) <= a N_n + h exp(h L) K R_t^2 / 2,   a = ||M_(n+1) M_n^-1||_2 (Gershgorin bound),
K >= |D^2 B| and L >= |DB| on the ball of radius R_t + step around the step (R_t: bootstrap bound of |w| in
the step; K = min(analytic piece bound, |D^2 B(g)| + rad x analytic |D^3 B| bound) with D^2 B(g) from
central differences of the reference gradient plus their error bound (numba) or forward-mode AD of the
closed form (JAX)).  With r_n = max_j |M_n E_j| + ||M_n|| sag + E_n, for every z in Q
    u(Phi(z)) <= u(g) + max_j grad u . M E_j + |grad u^T M| sag + |grad u| E + max(lmax, 0) r^2 / 2 + T r^3 / 6
(lmax: Gershgorin bound of the top eigenvalue of D^2 u(g) = D^2 B[.,.].Jx + 2 DB[.].J., Jx = (-y, x, 0);
T >= R |D^3 B| + 3 |D^2 B| on the ball; or the cruder (R + r) |D^2 B| + 2 |DB| bound alone).  If this is
< 0 at some step, no point of Q lies on a closed line with u > 0; the lower bound > 0 handles orientation -.
Analytic piece bounds: |D^k (r/|r|^3)| <= (k+1)!/|r|^(k+2) (Legendre), per straight piece of length L at
distance d (mu0 |I| / 4 pi) (k+1)! min(L / d^(k+2), c_k / d^(k+1)).

IMAGE pieces (re-seeding): when the linear image of a cell has grown (r > reseed x its radius) while the tube
bound is still small, the image conv(g + M E_j) + ball(||M|| sag + E) is split into pieces of about the
cell's size, which become cells of their own; the cell is certified when all pieces are (every line of the
cell passes through a piece).

TRANSFER certificate: "z lies on no closed line with u > 0" is a property of z's whole field line.  If every
line of a cell Q crosses a box face inside a union of face cells already certified for that orientation,
Q is certified too.  Crossing: the tube balls B(g_n, R_t + step) (R_t the bootstrap bound of the step) are
followed against the face's supporting surface (signed distance, 1-Lipschitz: phi faces the plane of the
half plane, R faces the cylinder, Z faces the plane); between a step with the whole ball on one side and a
later step with the whole ball on the other side every line crosses, at a point of the union of the balls
of these steps, whose parameter rectangle must lie in the face; the certified union is checked by exact
area coverage with disjoint certified rectangles (a hierarchical partition, each piece registered once per
orientation); rotated copies of faces share the certified cells of their representative.  Certificates are
only drawn from cells certified earlier, so the chain ends in direct LOST certificates.

Known obstacle (residual of unfinished runs): near-antisymmetric planes.  With equal ("unit") currents a
mirror-symmetric coil array is antisymmetric up to the rounding of the data (|B_phi| / |B| ~ 1e-4..1e-3 on
the coils' mid-planes, near_mirror_level): u ~ the symmetry breaking on a band around the perturbed
mid-plane, and the lines of the band follow a near-planar hyperbolic closed line (winding 0, u of both
signs along it, of size ~ the symmetry breaking) for a long time before u changes sign.  LOST cells there
must be ~ (symmetry breaking) x (local scale) small and followed far (codes 1/2 of the residual cells; the
measured residual cells of mastu_rmpu lie within 2e-4 rad of the mid-plane on the R faces).

What is strict and what is a validated model
--------------------------------------------
STRICT (mathematics, given the field values): (1)-(5); the variation-of-constants tube bound with rigorous
bounds on |D^2 B|, |DB| over the balls (bootstrap); the Taylor bounds of u with Gershgorin eigenvalue bounds
and analytic remainders; convexity of the cells; the FD error bound (delta^2/6 |D^4 B| + 4e-14 |DB|/delta);
the exact symmetry check.
VALIDATED MODEL (as in torux/barrier.py): the centre line and M carry 10 x the embedded DP5(4) local error
estimates, propagated with the computed one-step propagators; the field (the closed form of
torux.reference.fast64._segment summed in plain float64 -- numba eval_point here and the JAX field -- is
compared with the compensated reference torux.reference.fast64 on 64 sample points in every run, recorded as
lean_field_check_rel / jax_field_check_rel, ~1e-16..1e-15 relative) is taken as exact; floating-point
rounding is not interval-bounded (every margin is >> 1e-12 relative).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from .reference import fast64
from .nonstell_cert import bound_pieces, dkb_bound, min_dist, _seg_dist

TWO_PI = 2.0 * math.pi


# ============================================================================ lean segment field
def lean_pack(source):
    """(a, b, w) arrays of every straight current segment of a polyline/segment source, w = mu0 I / 4 pi."""
    from .source import MU0
    A_, B_, I_ = [], [], []
    for p in source.parts:
        if p.kind == "polyline":
            sg = p.as_segments()
            A_.append(sg.a); B_.append(sg.b); I_.append(sg.current)
        elif p.kind == "segments":
            A_.append(p.a); B_.append(p.b); I_.append(p.current)
        else:
            raise ValueError("lean_pack: polyline/segment sources only (part %s)" % p.kind)
    A0 = np.concatenate(A_); B0 = np.concatenate(B_); W0 = np.concatenate(I_) * (MU0 / (4 * math.pi))
    keep = (np.linalg.norm(B0 - A0, axis=1) > 0) & (W0 != 0)
    return (np.ascontiguousarray(A0[keep]), np.ascontiguousarray(B0[keep]), np.ascontiguousarray(W0[keep]))


@njit(cache=True, error_model="numpy")
def eval_point(SG, x0, x1, x2, grad, out):
    """B (out[0:3]) and dB_i/dx_j (out[6:15], row-major) of the straight segments SG = (a, b, w): the closed form
    of torux.reference.fast64._segment (same rounding-aware choice of the cross product), plain float64 sum.
    out[3:6] = 0 (A is not needed here)."""
    a, b, w = SG
    for k in range(15):
        out[k] = 0.0
    for s in range(w.shape[0]):
        Lx = b[s, 0] - a[s, 0]; Ly = b[s, 1] - a[s, 1]; Lz = b[s, 2] - a[s, 2]
        L2 = Lx * Lx + Ly * Ly + Lz * Lz
        L = math.sqrt(L2)
        Rix = x0 - a[s, 0]; Riy = x1 - a[s, 1]; Riz = x2 - a[s, 2]
        Rfx = x0 - b[s, 0]; Rfy = x1 - b[s, 1]; Rfz = x2 - b[s, 2]
        ri2 = Rix * Rix + Riy * Riy + Riz * Riz; rf2 = Rfx * Rfx + Rfy * Rfy + Rfz * Rfz
        ri = math.sqrt(ri2); rf = math.sqrt(rf2)
        dot = Rix * Rfx + Riy * Rfy + Riz * Rfz
        Pp = ri * rf
        if max(ri, rf) <= L:
            cx = Riy * Rfz - Riz * Rfy; cy = Riz * Rfx - Rix * Rfz; cz = Rix * Rfy - Riy * Rfx
        elif ri <= rf:
            cx = Ly * Riz - Lz * Riy; cy = Lz * Rix - Lx * Riz; cz = Lx * Riy - Ly * Rix
        else:
            cx = Ly * Rfz - Lz * Rfy; cy = Lz * Rfx - Lx * Rfz; cz = Lx * Rfy - Ly * Rfx
        if dot >= 0.0:
            den = Pp + dot
        else:
            den = (cx * cx + cy * cy + cz * cz) / (Pp - dot)
        if den <= 0.0:
            for k in range(15):
                out[k] = np.nan
            return 0.0
        Ss = ri + rf
        f = w[s] * Ss / (Pp * den)
        out[0] += cx * f; out[1] += cy * f; out[2] += cz * f
        if grad:
            gx = Rix / ri + Rfx / rf; gy = Riy / ri + Rfy / rf; gz = Riz / ri + Rfz / rf
            c1 = (Ss * Ss + L2) / (2.0 * Ss * den)
            d0 = -f * (gx * c1 + Rix / ri2 + Rfx / rf2)
            d1 = -f * (gy * c1 + Riy / ri2 + Rfy / rf2)
            d2 = -f * (gz * c1 + Riz / ri2 + Rfz / rf2)
            out[6] += cx * d0; out[7] += cx * d1 - Lz * f; out[8] += cx * d2 + Ly * f
            out[9] += cy * d0 + Lz * f; out[10] += cy * d1; out[11] += cy * d2 - Lx * f
            out[12] += cz * d0 - Ly * f; out[13] += cz * d1 + Lx * f; out[14] += cz * d2
    return 0.0


@njit(parallel=True, cache=True, error_model="numpy")
def eval_points(SG, pts, grad):
    n = pts.shape[0]
    out = np.zeros((n, 15))
    for i in prange(n):
        o = np.empty(15)
        eval_point(SG, pts[i, 0], pts[i, 1], pts[i, 2], grad, o)
        for k in range(15):
            out[i, k] = o[k]
    return out, np.zeros(n)


def lean_check(source, SG, points):
    """Max relative deviation of the lean field (B, DB) from torux.reference.fast64 at points."""
    ref, _ = fast64.eval_points(fast64.Field(source).packed, np.ascontiguousarray(points), True)
    got, _ = eval_points(SG, np.ascontiguousarray(points), True)
    eB = np.max(np.linalg.norm(got[:, :3] - ref[:, :3], axis=1) / np.linalg.norm(ref[:, :3], axis=1))
    eD = np.max(np.linalg.norm(got[:, 6:15] - ref[:, 6:15], axis=1) / np.linalg.norm(ref[:, 6:15], axis=1))
    return float(eB), float(eD)


# ============================================================================ small linear algebra
@njit(cache=True)
def _inv3(M, out):
    a, b, c, d, e, f, g, h, i = M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]
    A = e * i - f * h; B_ = -(d * i - f * g); C = d * h - e * g
    det = a * A + b * B_ + c * C
    if det == 0.0 or det != det:
        return False
    out[0] = A / det; out[1] = -(b * i - c * h) / det; out[2] = (b * f - c * e) / det
    out[3] = B_ / det; out[4] = (a * i - c * g) / det; out[5] = -(a * f - c * d) / det
    out[6] = C / det; out[7] = -(a * h - b * g) / det; out[8] = (a * e - b * d) / det
    return True


@njit(cache=True)
def _matmul3(A, B, out):
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[3 * i + k] * B[3 * k + j]
            out[3 * i + j] = s


@njit(cache=True)
def _spec_ub(A):
    """Upper bound on the spectral norm of a 3x3 matrix: sqrt(Gershgorin bound of A^T A),
    capped by the Frobenius norm."""
    G = np.empty(9)
    fro = 0.0
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[3 * k + i] * A[3 * k + j]
            G[3 * i + j] = s
    for k in range(9):
        fro += A[k] * A[k]
    gm = 0.0
    for i in range(3):
        r = abs(G[3 * i]) + abs(G[3 * i + 1]) + abs(G[3 * i + 2])
        if r > gm:
            gm = r
    v = min(gm, fro)
    return math.sqrt(v) * (1.0 + 1e-14)


@njit(cache=True)
def _frob(A):
    s = 0.0
    for k in range(A.shape[0]):
        s += A[k] * A[k]
    return math.sqrt(s)


# ============================================================================ field helpers
@njit(cache=True, error_model="numpy")
def _ugrad(x0, x1, buf, g):
    """u = x B_y - y B_x and its gradient from buf = (B, A, DB)."""
    u = x0 * buf[1] - x1 * buf[0]
    for j in range(3):
        g[j] = x0 * buf[9 + j] - x1 * buf[6 + j]
    g[0] += buf[1]
    g[1] -= buf[0]
    return u


@njit(cache=True, error_model="numpy")
def _hess_u_bounds(P, Q, x0, x1, x2, buf, ndb):
    """(lmin, lmax) bounds on the eigenvalues of D^2 u at x (Gershgorin of the central-difference
    Hessian, widened by the FD error bound).  Returns (nan, nan) on failure."""
    dm = min_dist(Q, x0, x1, x2)
    if not (dm > 0.0):
        return np.nan, np.nan
    de = 1e-3 * dm
    xp = np.array([x0, x1, x2])
    D2 = np.empty((3, 3, 3))      # D2[i, j, k] = d_j d_k B_i
    for k in range(3):
        xp[k] += de
        eval_point(P, xp[0], xp[1], xp[2], True, buf)
        g1 = buf[6:15].copy()
        xp[k] -= 2 * de
        eval_point(P, xp[0], xp[1], xp[2], True, buf)
        xp[k] += de
        for i in range(3):
            for j in range(3):
                D2[i, j, k] = (g1[3 * i + j] - buf[6 + 3 * i + j]) / (2 * de)
    d4 = dkb_bound(Q, x0, x1, x2, de, 4)
    if not (d4 < np.inf):
        return np.nan, np.nan
    e_ent = de * de / 6.0 * d4 + 4e-14 * ndb / de          # per-entry error of D2
    eval_point(P, x0, x1, x2, True, buf)
    H = np.empty((3, 3))
    for j in range(3):
        for k in range(3):
            v = x0 * 0.5 * (D2[1, j, k] + D2[1, k, j]) - x1 * 0.5 * (D2[0, j, k] + D2[0, k, j])
            # delta_j0 d_k B_y - delta_j1 d_k B_x + delta_k0 d_j B_y - delta_k1 d_j B_x
            if j == 0:
                v += buf[9 + k]
            if j == 1:
                v -= buf[6 + k]
            if k == 0:
                v += buf[9 + j]
            if k == 1:
                v -= buf[6 + j]
            H[j, k] = v
    R = math.sqrt(x0 * x0 + x1 * x1)
    eh = (abs(x0) + abs(x1)) * e_ent                       # per-entry error of H
    lmax = -np.inf; lmin = np.inf
    for i in range(3):
        off = 0.0
        for j in range(3):
            if j != i:
                off += abs(H[i, j]) + eh
        hi = H[i, i] + eh + off
        lo = H[i, i] - eh - off
        if hi > lmax:
            lmax = hi
        if lo < lmin:
            lmin = lo
    return lmin, lmax


@njit(cache=True, error_model="numpy")
def _rhs12(P, y, sig, dy, buf):
    """dy/dtau for y = (x, M row-major): x' = sig B, M' = sig DB M (unnormalised field-line time).
    Returns |B| (0 on failure)."""
    eval_point(P, y[0], y[1], y[2], True, buf)
    nb = math.sqrt(buf[0] * buf[0] + buf[1] * buf[1] + buf[2] * buf[2])
    if not (nb > 0.0) or nb != nb:
        return 0.0
    dy[0] = sig * buf[0]; dy[1] = sig * buf[1]; dy[2] = sig * buf[2]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += buf[6 + 3 * i + k] * y[3 + 3 * k + j]
            dy[3 + 3 * i + j] = sig * s
    return nb


@njit(cache=True, error_model="numpy")
def d2b_ball(P, Q, x0, x1, x2, rad, ndb, buf):
    """Bound on |D^2 B| (sup over unit w of |D^2 B[w, w]|) on the ball of radius rad around x:
    min(analytic piece bound, |central differences of the reference gradient|_F + their error bound
    (delta^2/6 |D^4 B| + 4e-14 |DB| / delta per entry, x 3 sqrt 3 for the 27 entries) + rad |D^3 B|_ball)."""
    crude = dkb_bound(Q, x0, x1, x2, rad, 2)
    if not (crude < np.inf):
        return np.inf
    dm = min_dist(Q, x0, x1, x2)
    de = 1e-3 * dm
    xp = np.array([x0, x1, x2])
    tot = 0.0
    for kk in range(3):
        xp[kk] += de
        eval_point(P, xp[0], xp[1], xp[2], True, buf)
        g1 = buf[6:15].copy()
        xp[kk] -= 2 * de
        eval_point(P, xp[0], xp[1], xp[2], True, buf)
        xp[kk] += de
        for q in range(9):
            v = (g1[q] - buf[6 + q]) / (2 * de)
            tot += v * v
    d4 = dkb_bound(Q, x0, x1, x2, de, 4)
    d3 = dkb_bound(Q, x0, x1, x2, rad, 3)
    if not (d4 < np.inf and d3 < np.inf):
        return crude
    sharp = math.sqrt(tot) + 3.0 * math.sqrt(3.0) * (de * de / 6.0 * d4 + 4e-14 * ndb / de) + rad * d3
    return min(crude, sharp)


@njit(cache=True)
def dkb2_dist(Q, x0, x1, x2, rad):
    """(crude bound of |D^2 B| on the ball of radius rad (as dkb_bound k = 2), distance from x to the coils)."""
    a, b, w, L, S = Q
    tot = 0.0
    dmin = np.inf
    for j in range(w.shape[0]):
        d0 = _seg_dist(x0, x1, x2, a[j], b[j]) - S[j]
        if d0 < dmin:
            dmin = d0
        d = d0 - rad
        if d <= 0.0:
            tot = np.inf
            continue
        dk1 = d * d * d
        v1 = L[j] / (dk1 * d)
        v2 = 0.5 * math.pi / dk1
        tot += w[j] * 6.0 * (v1 if v1 < v2 else v2)
    return tot, dmin


# ============================================================================ the cell test
@njit(cache=True, error_model="numpy")
def _cell_test(P, Q, x0, x1, x2, buf, M, E4, sag, Eb, need_neg, need_pos, rel_margin, fresh=True):
    """Test at the point g = (x0, x1, x2) of the centre line with variational matrix M (row-major 9),
    cell corner offsets E4 (4, 3), sag, and the tube error Eb.  Returns (neg_ok, pos_ok, r).
    fresh=False: buf already holds the reference field and gradient at g."""
    if fresh:
        eval_point(P, x0, x1, x2, True, buf)
    g = np.empty(3)
    u = _ugrad(x0, x1, buf, g)
    # gT M
    gm = np.empty(3)
    for j in range(3):
        gm[j] = g[0] * M[j] + g[1] * M[3 + j] + g[2] * M[6 + j]
    lmax = -np.inf; lmin = np.inf
    rM = 0.0
    for q in range(4):
        v = gm[0] * E4[q, 0] + gm[1] * E4[q, 1] + gm[2] * E4[q, 2]
        if v > lmax:
            lmax = v
        if v < lmin:
            lmin = v
        w0 = M[0] * E4[q, 0] + M[1] * E4[q, 1] + M[2] * E4[q, 2]
        w1 = M[3] * E4[q, 0] + M[4] * E4[q, 1] + M[5] * E4[q, 2]
        w2 = M[6] * E4[q, 0] + M[7] * E4[q, 1] + M[8] * E4[q, 2]
        nw = math.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
        if nw > rM:
            rM = nw
    ngm = math.sqrt(gm[0] ** 2 + gm[1] ** 2 + gm[2] ** 2)
    ng = math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    r = rM + _spec_ub(M) * sag + Eb
    lin_hi = u + lmax + ngm * sag + ng * Eb
    lin_lo = u + lmin - ngm * sag - ng * Eb
    marg = rel_margin * (abs(u) + ng * r) + 1e-300
    neg_ok = False; pos_ok = False
    try_neg = need_neg and lin_hi < -marg
    try_pos = need_pos and lin_lo > marg
    if not (try_neg or try_pos):
        return False, False, r
    if r == 0.0:
        return try_neg, try_pos, r
    ndb = 0.0
    for k in range(6, 15):
        ndb += buf[k] * buf[k]
    ndb = math.sqrt(ndb)
    R = math.sqrt(x0 * x0 + x1 * x1)
    # crude second-order bound on the ball: |D^2 u| <= (R + r) |D^2 B| + 2 |DB|_ball
    d2 = dkb_bound(Q, x0, x1, x2, r, 2)
    if not (d2 < np.inf):
        return False, False, r
    H2 = (R + r) * d2 + 2.0 * (ndb + r * d2)
    if try_neg:
        neg_ok = lin_hi + 0.5 * H2 * r * r < -marg
    if try_pos:
        pos_ok = lin_lo - 0.5 * H2 * r * r > marg
    if (neg_ok or not try_neg) and (pos_ok or not try_pos):
        return neg_ok, pos_ok, r
    # sharp: central-difference Hessian at g plus a third-order remainder
    hb = np.empty(15)
    hmin, hmax = _hess_u_bounds(P, Q, x0, x1, x2, hb, ndb)
    if hmin != hmin:
        return neg_ok, pos_ok, r
    d3 = dkb_bound(Q, x0, x1, x2, r, 3)
    if not (d3 < np.inf):
        return neg_ok, pos_ok, r
    T = (R + r) * d3 + 3.0 * d2
    if try_neg and not neg_ok:
        val = lin_hi + 0.5 * max(hmax, 0.0) * r * r + T * r * r * r / 6.0
        neg_ok = val < -marg
    if try_pos and not pos_ok:
        val = lin_lo + 0.5 * min(hmin, 0.0) * r * r - T * r * r * r / 6.0
        pos_ok = val > marg
    return neg_ok, pos_ok, r


# ============================================================================ tracer with tube bound
@njit(cache=True, error_model="numpy")
def _face_sigma_param(fc, x0, x1, x2, r):
    """Signed distance-like sigma of x from the supporting surface of face fc (1-Lipschitz) and the bounding
    rectangle in the face's normalised parameters (u, v) of the ball of radius r around x (umin, umax, vmin,
    vmax); valid flag False when the ball is not near the face's side (phi faces: wrong half plane)."""
    typ = int(fc[0]); val = fc[1]; R1 = fc[2]; R2 = fc[3]; Z1 = fc[4]; Z2 = fc[5]; p1 = fc[6]; pl = fc[7]
    R = math.sqrt(x0 * x0 + x1 * x1)
    if typ == 0:
        cphi = math.cos(val); sphi = math.sin(val)
        sig = -x0 * sphi + x1 * cphi
        valid = x0 * cphi + x1 * sphi > r
        u0 = (R - r - R1) / (R2 - R1); u1 = (R + r - R1) / (R2 - R1)
        v0 = (x2 - r - Z1) / (Z2 - Z1); v1 = (x2 + r - Z1) / (Z2 - Z1)
        return sig, valid, u0, u1, v0, v1
    ph = math.atan2(x1, x0)
    rel = (ph - p1) % TWO_PI
    if rel > pl + 0.5 * (TWO_PI - pl):
        rel -= TWO_PI
    dph = math.asin(min(1.0, r / max(R - r, 1e-300))) if R > r else math.pi
    valid = R > 2.0 * r
    if typ == 1:
        sig = R - val
        u0 = (rel - dph) / pl; u1 = (rel + dph) / pl
        v0 = (x2 - r - Z1) / (Z2 - Z1); v1 = (x2 + r - Z1) / (Z2 - Z1)
    else:
        sig = x2 - val
        u0 = (R - r - R1) / (R2 - R1); u1 = (R + r - R1) / (R2 - R1)
        v0 = (rel - dph) / pl; v1 = (rel + dph) / pl
    return sig, valid, u0, u1, v0, v1


@njit(cache=True, error_model="numpy")
def _face_update(faces, fstate, fbox, x0, x1, x2, r, ev, dirn, s):
    """Track clean sides of the tube ball w.r.t. every face; emit an event (face key, param rectangle of the
    crossing region, direction, arc length) when the tube passes from one clean side to the other.
    fstate[f]: last clean side (+1 / -1, 0 unknown); fbox[f]: accumulated rectangle since that side."""
    for f in range(faces.shape[0]):
        sig, valid, u0, u1, v0, v1 = _face_sigma_param(faces[f], x0, x1, x2, r)
        if not valid:
            fstate[f] = 0.0
            fbox[f, 0] = np.inf; fbox[f, 1] = -np.inf; fbox[f, 2] = np.inf; fbox[f, 3] = -np.inf
            continue
        side = 0.0
        if sig > r:
            side = 1.0
        elif sig < -r:
            side = -1.0
        if side == 0.0:
            if fstate[f] != 0.0:
                if u0 < fbox[f, 0]:
                    fbox[f, 0] = u0
                if u1 > fbox[f, 1]:
                    fbox[f, 1] = u1
                if v0 < fbox[f, 2]:
                    fbox[f, 2] = v0
                if v1 > fbox[f, 3]:
                    fbox[f, 3] = v1
            continue
        if fstate[f] != 0.0 and side != fstate[f]:
            # crossing between the last clean step and now: include this ball too
            a0 = min(fbox[f, 0], u0); a1 = max(fbox[f, 1], u1); b0 = min(fbox[f, 2], v0); b1 = max(fbox[f, 3], v1)
            if a0 >= 0.0 and a1 <= 1.0 and b0 >= 0.0 and b1 <= 1.0:
                for k in range(ev.shape[0]):
                    if ev[k, 0] < 0.0:
                        ev[k, 0] = faces[f, 8]; ev[k, 1] = a0; ev[k, 2] = a1; ev[k, 3] = b0; ev[k, 4] = b1
                        ev[k, 5] = dirn; ev[k, 6] = s
                        break
        fstate[f] = side
        fbox[f, 0] = u0; fbox[f, 1] = u1; fbox[f, 2] = v0; fbox[f, 3] = v1


@njit(cache=True, error_model="numpy")
def cert_cell(P, Q, c, E4, sag, need_neg, need_pos, tol, hmax, smax, nmax, Rfac, rel_margin, out, diag,
              faces, ev, reseed=0.0):
    """LOST certificate of one cell (module docstring).  out = [neg_ok, pos_ok, s_neg, s_pos, steps,
    code, r_last, dphi_fwd, dphi_bwd, s_fwd, s_bwd, E_last]; code: 0 ok, 1 tube bootstrap failed,
    2 smax, 3 field bound failure (near a coil / |B| -> 0), 4 step limit."""
    buf = np.empty(15)
    rho = 0.0
    for q in range(4):
        v = math.sqrt(E4[q, 0] ** 2 + E4[q, 1] ** 2 + E4[q, 2] ** 2)
        if v > rho:
            rho = v
    rho += sag
    Mi = np.zeros(9); Mi[0] = 1.0; Mi[4] = 1.0; Mi[8] = 1.0
    negok, posok, r0 = _cell_test(P, Q, c[0], c[1], c[2], buf, Mi, E4, sag, 0.0,
                                  need_neg, need_pos, rel_margin)
    negok = negok or not need_neg
    posok = posok or not need_pos
    for k in range(out.shape[0]):
        out[k] = 0.0
    out[2] = 0.0 if negok else -1.0
    out[3] = 0.0 if posok else -1.0
    steps = 0
    code = 0
    r_last = r0
    E_last = 0.0
    if negok and posok:
        out[0] = 1.0; out[1] = 1.0
        return
    best_s = -1.0
    k1 = np.empty(12); k2 = np.empty(12); k3 = np.empty(12); k4 = np.empty(12)
    k5 = np.empty(12); k6 = np.empty(12); k7 = np.empty(12); t = np.empty(12); yn = np.empty(12)
    y = np.zeros(12)
    Minv = np.empty(9); A = np.empty(9); fb = np.empty(15)
    idg = 0
    for d in range(2):
        sig = 1.0 if d == 0 else -1.0
        y[:] = 0.0
        y[0] = c[0]; y[1] = c[1]; y[2] = c[2]
        y[3] = 1.0; y[7] = 1.0; y[11] = 1.0
        if _rhs12(P, y, sig, k1, buf) == 0.0:
            code = 3
            break
        nb0 = math.sqrt(buf[0] ** 2 + buf[1] ** 2 + buf[2] ** 2)
        nd0 = 0.0
        for q in range(6, 15):
            nd0 += buf[q] * buf[q]
        nd0 = math.sqrt(nd0)
        dm0 = min_dist(Q, y[0], y[1], y[2])
        hs = min(hmax, 0.06 * dm0, 0.5 * rho + 1e-3 * hmax, 0.06 * nb0 / max(nd0, 1e-300))  # arc-length step
        h = hs / nb0                                                # tau step
        s = 0.0
        N = 0.0; ex = 0.0; eM = 0.0
        rcur = r0
        dphi = 0.0
        dcode = 2
        nfc = faces.shape[0]
        fstate = np.zeros(nfc); fbox = np.empty((nfc, 4))
        if nfc > 0:
            _face_update(faces, fstate, fbox, y[0], y[1], y[2], rcur, ev, float(d), 0.0)
        for it in range(nmax):
            steps += 1
            for q in range(12):
                t[q] = y[q] + h * (0.2 * k1[q])
            ok2 = _rhs12(P, t, sig, k2, buf)
            for q in range(12):
                t[q] = y[q] + h * (3.0 / 40 * k1[q] + 9.0 / 40 * k2[q])
            ok3 = _rhs12(P, t, sig, k3, buf)
            for q in range(12):
                t[q] = y[q] + h * (44.0 / 45 * k1[q] - 56.0 / 15 * k2[q] + 32.0 / 9 * k3[q])
            ok4 = _rhs12(P, t, sig, k4, buf)
            for q in range(12):
                t[q] = y[q] + h * (19372.0 / 6561 * k1[q] - 25360.0 / 2187 * k2[q] + 64448.0 / 6561 * k3[q]
                                   - 212.0 / 729 * k4[q])
            ok5 = _rhs12(P, t, sig, k5, buf)
            for q in range(12):
                t[q] = y[q] + h * (9017.0 / 3168 * k1[q] - 355.0 / 33 * k2[q] + 46732.0 / 5247 * k3[q]
                                   + 49.0 / 176 * k4[q] - 5103.0 / 18656 * k5[q])
            ok6 = _rhs12(P, t, sig, k6, buf)
            for q in range(12):
                yn[q] = y[q] + h * (35.0 / 384 * k1[q] + 500.0 / 1113 * k3[q] + 125.0 / 192 * k4[q]
                                    - 2187.0 / 6784 * k5[q] + 11.0 / 84 * k6[q])
            ok7 = _rhs12(P, yn, sig, k7, buf)
            if ok2 == 0.0 or ok3 == 0.0 or ok4 == 0.0 or ok5 == 0.0 or ok6 == 0.0 or ok7 == 0.0:
                dcode = 3
                break
            errx = 0.0; errM = 0.0
            for q in range(12):
                e = abs(h * (71.0 / 57600 * k1[q] - 71.0 / 16695 * k3[q] + 71.0 / 1920 * k4[q]
                             - 17253.0 / 339200 * k5[q] + 22.0 / 525 * k6[q] - 1.0 / 40 * k7[q]))
                if q < 3:
                    if e > errx:
                        errx = e
                else:
                    if e > errM:
                        errM = e
            errx *= math.sqrt(3.0); errM *= 3.0
            err = max(errx, errM * rho)
            step_len = math.sqrt((yn[0] - y[0]) ** 2 + (yn[1] - y[1]) ** 2 + (yn[2] - y[2]) ** 2)
            if (err > tol or step_len > 1.5 * hs) and h * nb0 > 1e-12 * hmax:
                h = h * max(0.2, min(0.9 * (tol / max(err, 1e-300)) ** 0.2, hs / max(step_len, 1e-300)))
                continue
            # field data at the new point (k7 evaluated it into buf)
            for q in range(15):
                fb[q] = buf[q]
            nb1 = math.sqrt(buf[0] ** 2 + buf[1] ** 2 + buf[2] ** 2)
            ndb1 = 0.0
            for q in range(6, 15):
                ndb1 += buf[q] * buf[q]
            ndb1 = math.sqrt(ndb1)
            # one-step propagator A = M_{n+1} M_n^{-1}
            if not _inv3(y[3:12], Minv):
                dcode = 1
                break
            _matmul3(yn[3:12], Minv, A)
            a = _spec_ub(A) * (1.0 + 1e-9)
            Mn1 = _spec_ub(yn[3:12])
            My = _spec_ub(y[3:12])
            xn0 = yn[0]; xn1 = yn[1]; xn2 = yn[2]
            # tube bootstrap: R_t bounds |w| during the step; the ball covers the step's centre arc
            Rt = Rfac * max(rcur, 1e-300)
            okstep = False
            Kb = 0.0; Lb = 0.0; d2 = np.inf
            Nn = 0.0; exn = 0.0; eMn = 0.0; rn = 0.0
            sharp = False
            dmn = -1.0
            for attempt in range(6):
                rad = Rt + 1.2 * step_len + ex + 10.0 * errx
                if sharp:
                    d2 = d2b_ball(P, Q, xn0, xn1, xn2, rad, ndb1, buf)
                else:
                    d2, dmn = dkb2_dist(Q, xn0, xn1, xn2, rad)
                    if d2 < np.inf and h * 0.5 * d2 * Rt * Rt > 0.002 * rcur:
                        sharp = True
                        d2 = d2b_ball(P, Q, xn0, xn1, xn2, rad, ndb1, buf)
                if not (d2 < np.inf):
                    break
                Kb = d2
                Lb = ndb1 + rad * d2
                eh = math.exp(h * Lb)
                if eh * (rcur + 0.5 * h * Kb * Rt * Rt) > Rt:
                    Rt = 2.0 * eh * (rcur + 0.5 * h * Kb * Rt * Rt)
                    continue
                Nn = a * N + h * eh * 0.5 * Kb * Rt * Rt
                exn = a * ex + 10.0 * errx
                eMn = a * eM + 10.0 * errM + h * eh * Kb * ex * My
                En = Nn + exn + eMn * rho
                rM = 0.0
                for q in range(4):
                    w0 = yn[3] * E4[q, 0] + yn[4] * E4[q, 1] + yn[5] * E4[q, 2]
                    w1 = yn[6] * E4[q, 0] + yn[7] * E4[q, 1] + yn[8] * E4[q, 2]
                    w2 = yn[9] * E4[q, 0] + yn[10] * E4[q, 1] + yn[11] * E4[q, 2]
                    nw = math.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
                    if nw > rM:
                        rM = nw
                rn = rM + Mn1 * sag + En
                if rn <= Rt:
                    okstep = True
                    break
                Rt = 2.0 * rn
            if idg < diag.shape[0]:
                diag[idg, 0] = s + step_len; diag[idg, 1] = Rt; diag[idg, 2] = Kb; diag[idg, 3] = Nn
                diag[idg, 4] = exn; diag[idg, 5] = eMn; diag[idg, 6] = rn; diag[idg, 7] = a; diag[idg, 8] = d2
                diag[idg, 9] = step_len
                idg += 1
            if not okstep:
                dcode = 1
                break
            N = Nn; ex = exn; eM = eMn
            rcur = rn
            p0 = math.atan2(y[1], y[0]); p1 = math.atan2(yn[1], yn[0])
            dp = p1 - p0
            if dp > math.pi:
                dp -= TWO_PI
            elif dp < -math.pi:
                dp += TWO_PI
            dphi += dp
            s += step_len
            for q in range(12):
                y[q] = yn[q]
                k1[q] = k7[q]
            nb0 = nb1
            r_last = rcur
            E_last = N + ex + eM * rho
            if reseed > 0.0 and out.shape[0] >= 35 and rcur > reseed * rho and E_last < 0.05 * rcur:
                # the image has grown: hand it back to the driver (re-seeding).  The image of the cell is
                # contained in conv(g + M E_j) + ball(||M|| sag + E_last); out[21:24] = g, out[24:33] = M,
                # out[33] = ||M|| sag + E_last, out[34] = direction
                for q in range(3):
                    out[21 + q] = y[q]
                for q in range(9):
                    out[24 + q] = y[3 + q]
                out[33] = _spec_ub(y[3:12]) * sag + E_last
                out[34] = sig
                dcode = 5
                break
            if nfc > 0:
                _face_update(faces, fstate, fbox, y[0], y[1], y[2], max(Rt, rcur) + step_len, ev, float(d), s)
            ng, pg, _r = _cell_test(P, Q, y[0], y[1], y[2], fb, y[3:12], E4, sag, E_last,
                                    not negok, not posok, rel_margin, False)
            if ng and not negok:
                negok = True; out[2] = sig * s
            if pg and not posok:
                posok = True; out[3] = sig * s
            if negok and posok:
                break
            if s > smax:
                dcode = 2
                break
            if dmn < 0.0:
                dmn = min_dist(Q, y[0], y[1], y[2])
            hs = min(hmax, 0.06 * dmn, 0.06 * nb1 / max(ndb1, 1e-300))
            fac = 0.9 * (tol / max(err, 1e-300)) ** 0.2
            h = min(hs / nb1, h * min(4.0, max(0.2, fac)))
        else:
            dcode = 4
        if d == 0:
            out[7] = dphi; out[9] = s
        else:
            out[8] = dphi; out[10] = s
        if s > best_s and out.shape[0] >= 21:
            best_s = s
            for q in range(9):
                out[12 + q] = y[3 + q]
        if negok and posok:
            break
        if dcode == 5:
            code = 5
            break
        if code == 0 or dcode == 3:
            code = dcode
    out[0] = 1.0 if negok else 0.0
    out[1] = 1.0 if posok else 0.0
    out[4] = steps
    out[5] = 0.0 if (negok and posok) else float(code if code else 2)
    out[6] = r_last
    out[11] = E_last


@njit(parallel=True, cache=True, error_model="numpy")
def cert_many(P, Q, C, E4s, sags, need, tol, hmax, smax, nmax, Rfac, rel_margin, faces, emax, reseed=0.0):
    n = C.shape[0]
    out = np.zeros((n, 35))
    evs = np.full((n, emax, 7), -1.0)
    for i in prange(n):
        o = np.zeros(35)
        ev = np.full((emax, 7), -1.0)
        cert_cell(P, Q, C[i], E4s[i], sags[i], need[i, 0] != 0, need[i, 1] != 0, tol, hmax[i], smax, nmax,
                  Rfac, rel_margin, o, np.zeros((0, 10)), faces, ev, reseed)
        for k in range(35):
            out[i, k] = o[k]
        for k in range(emax):
            for q in range(7):
                evs[i, k, q] = ev[k, q]
    return out, evs


# ============================================================================ geometry: loops, cones, tubes
def coil_loops(source):
    """Every coil of the source as a closed polygon with its current; kind 'polyline' loops are exact,
    other kinds are returned with exact=False (not handled by this module's cells)."""
    from .nonstell import toroidal_winding
    loops = []
    has_tor = False
    for ip, p in enumerate(source.parts):
        if p.kind == "polyline":
            for i in range(p.n):
                x = np.array(p.loop(i), float)
                # drop zero-length closing segments (repeated end point)
                keep = [0]
                for k in range(1, len(x)):
                    if np.linalg.norm(x[k] - x[keep[-1]]) > 0.0:
                        keep.append(k)
                x = x[keep]
                if np.linalg.norm(x[-1] - x[0]) == 0.0:
                    x = x[:-1]
                loops.append(dict(kind="polyline", pts=x, I=float(p.current[i]), part=ip, index=i, exact=True,
                                  w_z=toroidal_winding(x)))
        elif p.kind == "toroidal":
            has_tor = True
        else:
            from .nonstell import closed_curves
            loops.append(dict(kind=p.kind, part=ip, exact=False))
    return loops, has_tor


def _phi_gap(pts, rho, nbins=7200):
    """Largest toroidal-angle gap (rad) not covered by the polygon's rho-tube (each segment covers the
    short arc between its end angles, widened by asin(rho / its distance from the z axis))."""
    cov = np.zeros(nbins, bool)
    n = len(pts)
    w = TWO_PI / nbins
    for i in range(n):
        a = pts[i]; b = pts[(i + 1) % n]
        # distance of the segment from the z axis
        d = _seg_dist(0.0, 0.0, 0.0, np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0]))
        if rho >= d:
            return 0.0
        pa = math.atan2(a[1], a[0]); pb = math.atan2(b[1], b[0])
        dp = (pb - pa + math.pi) % TWO_PI - math.pi
        widen = math.asin(rho / d)
        lo = pa + min(0.0, dp) - widen - w; hi = pa + max(0.0, dp) + widen + w
        k0 = int(math.floor(lo / w)); k1 = int(math.ceil(hi / w))
        if k1 - k0 >= nbins:
            return 0.0
        cov[np.arange(k0, k1 + 1) % nbins] = True
    if cov.all():
        return 0.0
    # longest run of False (cyclic)
    f = np.concatenate([~cov, ~cov])
    best = run = 0
    for v in f:
        run = run + 1 if v else 0
        best = max(best, run)
    return min(best, nbins) * w


def seg_dist_np(x, a, b):
    ab = b - a
    t = np.clip(np.einsum("...k,k->...", x - a, ab) / max(np.dot(ab, ab), 1e-300), 0.0, 1.0)
    return np.linalg.norm(x - (a + t[..., None] * ab), axis=-1)


def loop_dist(x, pts):
    """Distance from points x (..., 3) to the closed polygon pts."""
    n = len(pts)
    d = np.full(np.shape(x)[:-1], np.inf)
    for i in range(n):
        d = np.minimum(d, seg_dist_np(x, pts[i], pts[(i + 1) % n]))
    return d


def tube_radius(loops, j, frac=0.4, cap=None):
    """Tube radius rho_j for coil j (w_z = 0): below frac x the distance to every other coil, 0.9 x the
    distance from the z axis, and such that the tube stays in a wedge of toroidal angle < 2 pi (every
    closed curve in it has toroidal winding 0)."""
    pj = loops[j]["pts"]
    dmin = np.inf
    for m, lm in enumerate(loops):
        if m == j or not lm.get("exact"):
            continue
        dmin = min(dmin, float(np.min(loop_dist(lm["pts"], pj))))
    Rmin = float(np.min(np.hypot(pj[:, 0], pj[:, 1])))
    ext = float(np.max(np.ptp(pj, axis=0)))
    rho = min(frac * dmin, 0.9 * Rmin, (cap if cap else 0.08 * ext))
    while rho > 1e-6 * ext and _phi_gap(pj, rho) < 1e-3:
        rho *= 0.5
    return rho


class Cell:
    __slots__ = ("kind", "coil", "piece", "box", "need", "depth", "c", "E4", "sag", "rad", "dist")

    def __init__(self, kind, coil, piece, box, need=(True, True), depth=0):
        self.kind = kind; self.coil = coil; self.piece = piece; self.box = tuple(box)
        self.need = tuple(need); self.depth = depth


def cell_geometry(cell, G):
    """Centre c, corner offsets E4 (4, 3), sag of a cell; G: geometry dict of the source."""
    k = cell.kind
    if k in ("bphi", "bR", "bZ"):
        return box_cell_geometry(cell, G["boxes"][cell.coil]["box"])
    if k == "cone":
        lp = G["loops"][cell.coil]
        apex = G["apex"][cell.coil]
        P = lp["pts"]; i = cell.piece
        a = P[i]; b = P[(i + 1) % len(P)]
        t0, t1, u0, u1 = cell.box

        def X(t, u):
            return apex + t * ((a - apex) + u * (b - a))
        c = X(0.5 * (t0 + t1), 0.5 * (u0 + u1))
        E4 = np.array([X(t0, u0), X(t1, u0), X(t0, u1), X(t1, u1)]) - c
        return c, E4, 0.0
    if k == "cyl":
        lp = G["loops"][cell.coil]
        rho = G["rho"][cell.coil]
        P = lp["pts"]; i = cell.piece
        a = P[i]; b = P[(i + 1) % len(P)]
        t = b - a; ell = np.linalg.norm(t); t = t / ell
        n1 = np.cross(t, [0.0, 0.0, 1.0])
        if np.linalg.norm(n1) < 0.1:
            n1 = np.cross(t, [1.0, 0.0, 0.0])
        n1 /= np.linalg.norm(n1); n2 = np.cross(t, n1)
        s0, s1, th0, th1 = cell.box

        def X(s, th):
            return a + s * ell * t + rho * (math.cos(th) * n1 + math.sin(th) * n2)
        corners = np.array([X(s0, th0), X(s1, th0), X(s0, th1), X(s1, th1)])
        c = corners.mean(0)
        sag = rho * (1.0 - math.cos(0.5 * (th1 - th0))) * 1.0001
        return c, corners - c, sag
    if k == "sph":
        lp = G["loops"][cell.coil]
        rho = G["rho"][cell.coil]
        v = lp["pts"][cell.piece]
        f, a0, a1, b0, b1 = cell.box
        e0, e1, e2 = _CUBE[int(f)]

        def D(a, b):
            d = e0 + a * e1 + b * e2
            return d / np.linalg.norm(d)
        dc = D(0.5 * (a0 + a1), 0.5 * (b0 + b1))
        ch = max(np.linalg.norm(D(a, b) - dc) for a in (a0, a1) for b in (b0, b1))
        c = v + rho * dc
        return c, np.zeros((4, 3)), rho * ch * 1.0001
    raise ValueError(k)


_CUBE = []
for _ax in range(3):
    for _sg in (1.0, -1.0):
        _e0 = np.zeros(3); _e0[_ax] = _sg
        _e1 = np.zeros(3); _e1[(_ax + 1) % 3] = 1.0
        _e2 = np.zeros(3); _e2[(_ax + 2) % 3] = 1.0
        _CUBE.append((_e0, _e1, _e2))


def split_cell(cell, G, M=None):
    """Split a cell in two along its longer side (four when both are comparable); with the variational
    matrix M of a failed trace, along the parameter whose image under M is longer."""
    c, E4, sag = cell.c, cell.E4, cell.sag
    k = cell.kind
    bx = cell.box
    if k == "img":
        l1 = max(np.linalg.norm(E4[1] - E4[0]), np.linalg.norm(E4[3] - E4[2]))
        l2 = max(np.linalg.norm(E4[2] - E4[0]), np.linalg.norm(E4[3] - E4[1]))
        if M is not None and np.all(np.isfinite(M)):
            Mm = np.asarray(M).reshape(3, 3)
            l1 = max(np.linalg.norm(Mm @ (E4[1] - E4[0])), np.linalg.norm(Mm @ (E4[3] - E4[2])))
            l2 = max(np.linalg.norm(Mm @ (E4[2] - E4[0])), np.linalg.norm(Mm @ (E4[3] - E4[1])))
        na, nb = (2, 1) if l1 > 1.6 * l2 else ((1, 2) if l2 > 1.6 * l1 else (2, 2))
        kids = []
        for i in range(na):
            for j in range(nb):
                def X(a, b):
                    return ((1 - a) * (1 - b)) * E4[0] + (a * (1 - b)) * E4[1] + ((1 - a) * b) * E4[2] + (a * b) * E4[3]
                a0, a1 = i / na, (i + 1) / na; b0, b1 = j / nb, (j + 1) / nb
                cs = np.array([X(a0, b0), X(a1, b0), X(a0, b1), X(a1, b1)])
                cc = cs.mean(0)
                kk = Cell("img", -1, -1, (0.0, 1.0, 0.0, 1.0), cell.need, cell.depth + 1)
                kk.c = c + cc; kk.E4 = cs - cc; kk.sag = sag
                kk.rad = float(np.max(np.linalg.norm(kk.E4, axis=1))) + sag
                kids.append(kk)
        return kids
    if k == "sph":
        f, a0, a1, b0, b1 = bx
        am = 0.5 * (a0 + a1); bm = 0.5 * (b0 + b1)
        boxes = [(f, a0, am, b0, bm), (f, am, a1, b0, bm), (f, a0, am, bm, b1), (f, am, a1, bm, b1)]
    else:
        x0, x1, y0, y1 = bx
        # physical lengths of the two parameter directions
        l1 = max(np.linalg.norm(E4[1] - E4[0]), np.linalg.norm(E4[3] - E4[2]))
        l2 = max(np.linalg.norm(E4[2] - E4[0]), np.linalg.norm(E4[3] - E4[1]))
        if k == "cyl":
            l2 = max(l2, 2 * G["rho"][cell.coil] * math.sin(0.5 * (y1 - y0)))
        if M is not None and np.all(np.isfinite(M)):
            Mm = np.asarray(M).reshape(3, 3)
            l1 = max(np.linalg.norm(Mm @ (E4[1] - E4[0])), np.linalg.norm(Mm @ (E4[3] - E4[2])))
            l2 = max(np.linalg.norm(Mm @ (E4[2] - E4[0])), np.linalg.norm(Mm @ (E4[3] - E4[1])))
        xm = 0.5 * (x0 + x1); ym = 0.5 * (y0 + y1)
        if l1 > 1.6 * l2:
            boxes = [(x0, xm, y0, y1), (xm, x1, y0, y1)]
        elif l2 > 1.6 * l1:
            boxes = [(x0, x1, y0, ym), (x0, x1, ym, y1)]
        else:
            boxes = [(x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)]
    return [Cell(k, cell.coil, cell.piece, b, cell.need, cell.depth + 1) for b in boxes]


def _prepare(cells, G):
    for cl in cells:
        if cl.kind == "img":
            cl.rad = float(np.max(np.linalg.norm(cl.E4, axis=1))) + cl.sag
            continue
        cl.c, cl.E4, cl.sag = cell_geometry(cl, G)
        cl.rad = float(np.max(np.linalg.norm(cl.E4, axis=1))) + cl.sag


def initial_cells(G, h0):
    """Cells covering S: the cones of every coil (outside the tubes) and the tube surfaces."""
    cells = []
    for k, lp in enumerate(G["loops"]):
        if lp["I"] == 0.0:
            continue
        P = lp["pts"]; n = len(P)
        apex = G["apex"][k]
        rho = G["rho"][k]
        for i in range(n):
            a = P[i]; b = P[(i + 1) % n]
            L_ab = np.linalg.norm(b - a)
            L_ca = max(np.linalg.norm(a - apex), np.linalg.norm(b - apex))
            nt = max(1, int(math.ceil(L_ca / h0)))
            for it in range(nt):
                t0, t1 = it / nt, (it + 1) / nt
                width = t1 * L_ab
                nu = max(1, int(math.ceil(width / h0)))
                for iu in range(nu):
                    cells.append(Cell("cone", k, i, (t0, t1, iu / nu, (iu + 1) / nu)))
            if rho is not None:
                ns_ = max(1, int(math.ceil(L_ab / h0)))
                nth = max(4, int(math.ceil(TWO_PI * rho / h0)))
                for is_ in range(ns_):
                    for ith in range(nth):
                        cells.append(Cell("cyl", k, i, (is_ / ns_, (is_ + 1) / ns_, TWO_PI * ith / nth,
                                                         TWO_PI * (ith + 1) / nth)))
        if rho is not None:
            m = max(1, int(math.ceil(1.5 * rho / h0)))
            for i in range(n):
                for f in range(6):
                    for ia in range(m):
                        for ib in range(m):
                            cells.append(Cell("sph", k, i, (f, -1 + 2 * ia / m, -1 + 2 * (ia + 1) / m,
                                                            -1 + 2 * ib / m, -1 + 2 * (ib + 1) / m)))
    return cells


def filter_cells(cells, G):
    """Drop cells that need no certificate: cone cells entirely inside one tube Theta_j; cylinder and
    sphere cells of coil j entirely inside the rho_j-capsule of another piece of coil j (not on the
    level set dist(., C_j) = rho_j).  Returns (kept, n_dropped)."""
    keep = []
    nd = 0
    for cl in cells:
        drop = False
        if cl.kind == "cone":
            for j in G["tube_coils"]:
                pj = G["loops"][j]["pts"]
                if float(loop_dist(cl.c[None, :], pj)[0]) + cl.rad < G["rho"][j]:
                    drop = True
                    break
        else:
            pj = G["loops"][cl.coil]["pts"]; n = len(pj); rho = G["rho"][cl.coil]
            for m in range(n):
                if cl.kind == "cyl" and m == cl.piece:
                    continue
                if seg_dist_np(cl.c[None, :], pj[m], pj[(m + 1) % n])[0] + cl.rad < rho:
                    drop = True
                    break
        if drop:
            nd += 1
        else:
            keep.append(cl)
    return keep, nd


def source_geometry(source, tube_frac=0.4, tube_cap=None):
    """Loops, apexes, tube radii; raises NotImplementedError for unsupported part kinds."""
    loops, has_tor = coil_loops(source)
    G = dict(loops=loops, has_tor=has_tor, apex=[], rho=[], tube_coils=[])
    allp = np.concatenate([lp["pts"] for lp in loops if lp.get("exact")]) if any(lp.get("exact") for lp in loops) \
        else np.zeros((0, 3))
    G["L"] = float(np.max(np.hypot(allp[:, 0], allp[:, 1]))) if len(allp) else 1.0
    for j, lp in enumerate(loops):
        if not lp.get("exact"):
            G["apex"].append(None); G["rho"].append(None)
            continue
        P = lp["pts"]
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        mid = 0.5 * (P + np.roll(P, -1, 0))
        G["apex"].append((mid * seg[:, None]).sum(0) / seg.sum())
        if lp["w_z"] == 0 and lp["I"] != 0.0:
            G["rho"].append(tube_radius(loops, j, tube_frac, tube_cap))
            G["tube_coils"].append(j)
        else:
            G["rho"].append(None)
    return G


# ============================================================================ driver
def linkbox_geometry(source, margin_cap_rel=0.25, clear_frac=0.45, merge_rel=0.03):
    loops, has_tor = coil_loops(source)
    G = dict(loops=loops, has_tor=has_tor, apex=[], rho=[None] * len(loops), tube_coils=[])
    allp = np.concatenate([lp["pts"] for lp in loops if lp.get("exact")]) if any(lp.get("exact") for lp in loops) \
        else np.zeros((0, 3))
    G["L"] = float(np.max(np.hypot(allp[:, 0], allp[:, 1]))) if len(allp) else 1.0
    for lp in loops:
        if not lp.get("exact"):
            G["apex"].append(None)
            continue
        P = lp["pts"]
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        mid = 0.5 * (P + np.roll(P, -1, 0))
        G["apex"].append((mid * seg[:, None]).sum(0) / seg.sum())
    G["boxes"] = link_boxes(loops, margin_cap_rel, clear_frac, merge_rel) if not has_tor else []
    G["box_params"] = dict(margin_cap_rel=margin_cap_rel, clear_frac=clear_frac, merge_rel=merge_rel)
    G["cone_coils"] = [j for j, lp in enumerate(loops) if lp.get("exact") and lp["I"] != 0.0 and lp["w_z"] != 0]
    G["axis_coils"] = [j for j, lp in enumerate(loops) if lp.get("exact") and lp["I"] != 0.0 and lp["w_z"] == 0
                       and coil_box(lp["pts"])[0] <= 0.0]
    G["symmetry"] = box_symmetry(source, G)
    return G


@njit(parallel=True, cache=True)
def _min_dists(Q, X):
    out = np.empty(X.shape[0])
    for i in prange(X.shape[0]):
        out[i] = min_dist(Q, X[i, 0], X[i, 1], X[i, 2])
    return out


def box_cost_estimate(G, P, Q, kappa=0.12, h0_rel=0.02):
    """Rough number of cells of the certified box boundaries: sum over coarse face cells of
    area / (kappa x local scale)^2, local scale = min(distance to the coils, |B| / |DB|)."""
    cells = []
    for bi in G["symmetry"]["represent"]:
        if G["boxes"][bi]["contractible"]:
            cells += box_face_cells(bi, G["boxes"][bi]["box"], h0_rel * G["L"])
    if not cells:
        return np.inf
    _prepare(cells, G)
    C = np.array([c.c for c in cells])
    A = np.array([np.linalg.norm(c.E4[1] - c.E4[0]) * np.linalg.norm(c.E4[2] - c.E4[0]) for c in cells])
    fv, _ = eval_points(P, C, True)
    d = _min_dists(Q, C)
    sc = np.minimum(d, np.linalg.norm(fv[:, :3], axis=1) / np.maximum(np.linalg.norm(fv[:, 6:15], axis=1), 1e-300))
    return float(np.sum(A / np.maximum(kappa * sc, 1e-9) ** 2))


def empirical_cost(G, P, Q, n=120, seed=2, smax_rel=60.0):
    """Sample estimate of the number of cells of the certified box boundaries: at n random boundary points,
    the largest dyadic cell (from 0.12 x local scale down) whose LOST certificate succeeds for both
    orientations; cost = area x mean(1 / cell area).  Unsuccessful samples count at 1/512 of the scale."""
    cells = []
    for bi in G["symmetry"]["represent"]:
        if G["boxes"][bi]["contractible"]:
            cells += box_face_cells(bi, G["boxes"][bi]["box"], 0.01 * G["L"])
    if not cells:
        return np.inf
    _prepare(cells, G)
    L = G["L"]
    rng = np.random.default_rng(seed)
    areas = np.array([np.linalg.norm(c.E4[1] - c.E4[0]) * np.linalg.norm(c.E4[2] - c.E4[0]) for c in cells])
    idx = rng.choice(len(cells), min(n, len(cells)), replace=False, p=areas / areas.sum())
    inv = []
    for i in idx:
        cl = cells[i]
        fv, _ = eval_points(P, cl.c[None, :], True)
        sc = min(min_dist(Q, cl.c[0], cl.c[1], cl.c[2]), np.linalg.norm(fv[0, :3]) / max(np.linalg.norm(fv[0, 6:15]), 1e-300))
        got = None
        for k in range(9):
            size = 0.12 * sc / 2 ** k
            c = cl
            while c.rad > size:
                c = split_cell(c, G)[int(rng.integers(0, 2))]
                _prepare([c], G)
            o = np.zeros(21)
            cert_cell(P, Q, c.c, c.E4, c.sag, True, True, 1e-9 * L, 0.03 * L, smax_rel * L, 200000, 1.3, 1e-9, o,
                      np.zeros((0, 10)), np.zeros((0, 9)), np.full((1, 7), -1.0))
            if o[0] == 1.0 and o[1] == 1.0:
                got = c.rad
                break
        r = got if got else 0.12 * sc / 2 ** 9
        inv.append(1.0 / (2.0 * r * r))
    return float(areas.sum() * np.mean(inv))


def choose_geometry(source, options=((0.03, 0.03), (0.03, 0.08), (0.03, 0.25), (1.0, 0.03), (1.0, 0.08),
                                     (1.0, 0.25))):
    """Box grouping (merge_rel) and margin cap with the smallest sampled cost (empirical_cost) among the
    options whose boxes are all contractible; identical geometries are evaluated once."""
    P = lean_pack(source)
    Q, _ = bound_pieces(source)
    best = None
    tried = []
    seen = {}
    for mr, mc in options:
        G = linkbox_geometry(source, margin_cap_rel=mc, merge_rel=mr)
        ok = all(b["contractible"] for b in G["boxes"]) and len(G["boxes"]) > 0
        key = tuple(tuple(np.round(b["box"], 9)) for b in G["boxes"])
        if not ok:
            cost = np.inf
        elif key in seen:
            cost = seen[key]
        else:
            cost = empirical_cost(G, P, Q)
            seen[key] = cost
        tried.append(dict(merge_rel=mr, margin_cap_rel=mc, n_boxes=len(G["boxes"]), contractible=ok, est_cells=cost))
        if best is None or cost < best[0]:
            best = (cost, G)
    G = best[1]
    G["geometry_choice"] = tried
    return G


def _rot(pts, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    return np.stack([c * pts[:, 0] - s_ * pts[:, 1], s_ * pts[:, 0] + c * pts[:, 1], pts[:, 2]], -1)


def source_rotation_symmetric(source, nfp, tol=1e-9):
    """Exact check that the current set is invariant under the rotation by 2 pi / nfp about z: every
    segment (a, b, I) rotated matches a segment of the set (end points within tol x size, same current)."""
    from scipy.spatial import cKDTree
    A, B_, I = [], [], []
    for p in source.parts:
        if p.kind == "polyline":
            sg = p.as_segments()
            A.append(sg.a); B_.append(sg.b); I.append(sg.current)
        elif p.kind == "segments":
            A.append(p.a); B_.append(p.b); I.append(p.current)
        else:
            return False, "part kind %s" % p.kind
    A = np.concatenate(A); B_ = np.concatenate(B_); I = np.concatenate(I)
    keep = np.linalg.norm(B_ - A, axis=1) > 0
    A, B_, I = A[keep], B_[keep], I[keep]
    size = float(np.max(np.abs(np.concatenate([A, B_]))))
    X = np.concatenate([A, B_], 1)
    tree = cKDTree(X)
    ang = TWO_PI / nfp
    Xr = np.concatenate([_rot(A, ang), _rot(B_, ang)], 1)
    d, idx = tree.query(Xr)
    ok = bool(np.all(d <= tol * size) and np.allclose(I[idx], I, rtol=1e-12, atol=0))
    return ok, float(np.max(d) / size)


def box_symmetry(source, G, tol=1e-9):
    """For an nfp-fold rotation-symmetric current set (checked here, independently of the loader): the
    boxes of coils that are rotated copies of a representative coil are replaced by the rotated
    representative box; their cells need no certificate of their own (lost-ness and u = R B_phi are
    invariant under the symmetry).  Returns dict(nfp, verified, represent: list of box indices to certify,
    images: {box index: (representative, k)})."""
    nfp = int(getattr(source, "nfp", 1) or 1)
    out = dict(nfp=nfp, verified=False, represent=list(range(len(G["boxes"]))), images={})
    if nfp <= 1 or not G["boxes"]:
        return out
    ok, dev = source_rotation_symmetric(source, nfp, tol)
    out["verified"] = ok; out["deviation"] = dev
    if not ok:
        return out
    loops = G["loops"]
    size = max(float(np.max(np.abs(lp["pts"]))) for lp in loops if lp.get("exact"))

    def coilset_key(js, ang):
        return [_rot(loops[j]["pts"], ang) for j in js]

    def same(P1, P2):
        if len(P1) != len(P2):
            return False
        used = set()
        for a in P1:
            hit = False
            for k, b in enumerate(P2):
                if k in used or a.shape != b.shape:
                    continue
                if np.max(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)) <= 1e-8 * size:
                    used.add(k); hit = True
                    break
            if not hit:
                return False
        return True
    rep = []
    for bi, b in enumerate(G["boxes"]):
        found = None
        for ri in rep:
            for k in range(1, nfp):
                if same(coilset_key(G["boxes"][ri]["coils"], TWO_PI * k / nfp), [loops[j]["pts"] for j in b["coils"]]):
                    found = (ri, k)
                    break
            if found:
                break
        if found:
            ri, k = found
            rb = list(G["boxes"][ri]["box"])
            rb[4] = (rb[4] + TWO_PI * k / nfp) % TWO_PI
            # the rotated representative box must contain this group's coils (exact segment extents)
            if all(box_contains(rb, coil_box(loops[j]["pts"])) for j in b["coils"]):
                b["box"] = rb
                b["image_of"] = [ri, k]
                out["images"][bi] = (ri, k)
                continue
        rep.append(bi)
    out["represent"] = rep
    return out


FACE_NUM = {("bphi", 0): 0, ("bphi", 1): 1, ("bR", 0): 2, ("bR", 1): 3, ("bZ", 0): 4, ("bZ", 1): 5}


def faces_table(G):
    """Every face of every box (images included) for the crossing events: rows [type (0 phi, 1 R, 2 Z), value,
    R1, R2, Z1, Z2, phi1, phi_len, key]; key = 6 x (certified box) + face number, the same for a face and its
    rotated copies (normalised face parameters are invariant under the symmetry)."""
    rows = []
    rep = set(G["symmetry"]["represent"])
    for bi, b in enumerate(G["boxes"]):
        rb = bi if bi in rep else (b.get("image_of") or [None])[0]
        if rb is None or not b["contractible"]:
            continue
        R1, R2, Z1, Z2, p1, pl = b["box"]
        for (kind, piece), fn in FACE_NUM.items():
            if kind == "bphi":
                typ, val = 0, (p1 if piece == 0 else p1 + pl)
            elif kind == "bR":
                typ, val = 1, (R1 if piece == 0 else R2)
            else:
                typ, val = 2, (Z1 if piece == 0 else Z2)
            rows.append([typ, val, R1, R2, Z1, Z2, p1, pl, 6 * rb + fn])
    return np.ascontiguousarray(np.array(rows, float).reshape(-1, 9))


class CertIndex:
    """Certified face cells per (face key, orientation): disjoint rectangles in normalised face parameters,
    bucketed on an nb x nb grid; covered(key, o, rect) tells whether the rectangle lies in their union."""

    def __init__(self, nb=32):
        self.nb = nb
        self.rects = {}
        self.buckets = {}

    def _bk(self, u0, u1, v0, v1):
        nb = self.nb
        i0 = max(0, min(nb - 1, int(u0 * nb))); i1 = max(0, min(nb - 1, int(u1 * nb)))
        j0 = max(0, min(nb - 1, int(v0 * nb))); j1 = max(0, min(nb - 1, int(v1 * nb)))
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def add(self, key, o, rect):
        lst = self.rects.setdefault((key, o), [])
        k = len(lst)
        lst.append(rect)
        bkt = self.buckets.setdefault((key, o), {})
        for b in self._bk(*rect):
            bkt.setdefault(b, []).append(k)

    def covered(self, key, o, q):
        lst = self.rects.get((key, o))
        if not lst:
            return False
        bkt = self.buckets[(key, o)]
        cand = set()
        for b in self._bk(*q):
            cand.update(bkt.get(b, ()))
        if not cand:
            return False
        R = np.array([lst[k] for k in cand])
        du = np.minimum(R[:, 1], q[1]) - np.maximum(R[:, 0], q[0])
        dv = np.minimum(R[:, 3], q[3]) - np.maximum(R[:, 2], q[2])
        inter = np.sum(np.clip(du, 0, None) * np.clip(dv, 0, None))
        area = (q[1] - q[0]) * (q[3] - q[2])
        return inter >= area * (1.0 - 1e-9) - 1e-300


def image_cells(cl, o, nsplit=None):
    """Pieces of the image of a cell handed back by the tracer (code 5): the linear image of the cell's quad
    conv(g + M E_j) split into nsplit x nsplit sub-quads (bilinear parameters), each widened by the sag
    ||M|| sag + E (bound of the deviation of the true image from the linear one).  Every line of the cell
    passes through one of the pieces, so the cell is certified when every piece is."""
    g = o[21:24]; M = o[24:33].reshape(3, 3); sg = float(o[33])
    W = cl.E4 @ M.T                                  # corners of the linear image relative to g
    # corner order (u0 v0), (u1 v0), (u0 v1), (u1 v1); pieces about the size of the parent along each side
    l1 = max(np.linalg.norm(W[1] - W[0]), np.linalg.norm(W[3] - W[2]))
    l2 = max(np.linalg.norm(W[2] - W[0]), np.linalg.norm(W[3] - W[1]))
    r0 = max(cl.rad, 1e-300)
    na = nsplit or max(1, min(16, int(math.ceil(l1 / (1.2 * r0)))))
    nb = nsplit or max(1, min(16, int(math.ceil(l2 / (1.2 * r0)))))
    kids = []
    for i in range(na):
        for j in range(nb):
            def X(a, b):
                return ((1 - a) * (1 - b)) * W[0] + (a * (1 - b)) * W[1] + ((1 - a) * b) * W[2] + (a * b) * W[3]
            a0, a1 = i / na, (i + 1) / na; b0, b1 = j / nb, (j + 1) / nb
            cs = np.array([X(a0, b0), X(a1, b0), X(a0, b1), X(a1, b1)])
            cc = cs.mean(0)
            k = Cell("img", -1, -1, (0.0, 1.0, 0.0, 1.0), cl.need, cl.depth + 1)
            k.c = g + cc; k.E4 = cs - cc; k.sag = sg
            k.rad = float(np.max(np.linalg.norm(k.E4, axis=1))) + sg
            kids.append(k)
    return kids


def _cell_key(cl, G):
    if cl.kind not in ("bphi", "bR", "bZ"):
        return None
    return 6 * cl.coil + FACE_NUM[(cl.kind, cl.piece)]


def _section_pass(P, Q, G, part, out, tol, L):
    """N / S certificates (period map of the half plane of the face) for the cells on phi faces whose LOST test
    failed for an orientation.  Returns {cell index: {0 or 1 (orientation +/-): 'N' or 'S'}}."""
    idx, phis, R0s, Z0s, rhos, sgns, which = [], [], [], [], [], [], []
    for ci, (cl, o) in enumerate(zip(part, out)):
        if cl.kind != "bphi":
            continue
        # only where the centre line went a long way around (|accumulated phi| > 1 rad in a direction):
        # lines that reverse quickly are left to the LOST certificate of smaller cells
        if max(abs(o[7]), abs(o[8])) < 1.0:
            continue
        R1, R2, Z1, Z2, p1, pl = G["boxes"][cl.coil]["box"]
        ph = p1 if cl.piece == 0 else p1 + pl
        Rc = math.hypot(cl.c[0], cl.c[1]); Zc = cl.c[2]
        rho = float(np.max(np.linalg.norm(cl.E4, axis=1)))
        for k_, sg in ((0, 1.0), (1, -1.0)):
            if cl.need[k_] and o[k_] != 1.0:
                idx.append(ci); phis.append(ph); R0s.append(Rc); Z0s.append(Zc); rhos.append(rho); sgns.append(sg)
                which.append(k_)
    res = {}
    if not idx:
        return res
    o2 = phimap_many(P, Q, np.array(phis), np.array(R0s), np.array(Z0s), np.array(rhos), np.array(sgns),
                     1e-3 * tol, 0.05, 20000, 1.3)
    for j in range(len(idx)):
        v, _info = section_verdict(o2[j], R0s[j], Z0s[j], rhos[j])
        if v is not None:
            res.setdefault(idx[j], {})[which[j]] = v
    return res


def certify_source(source, h0_rel=0.04, min_rel=2e-5, max_depth=40, max_cells=2_000_000, budget_s=None,
                   tol_rel=1e-9, hmax_rel=0.03, smax_rel=6.0, nmax=200000, Rfac=1.3, rel_margin=1e-9,
                   kappa=0.12, margin_cap_rel=0.25, clear_frac=0.45, batch=4096, verbose=False, log=None,
                   backend="numba", part=None, sections=True, transfer=True, geometry="auto", checkpoint=None,
                   reseed=0.0):
    """Run the LINKBOX certification on every cell of the box boundaries (and of the Seifert cones of
    coils with toroidal winding != 0).  Returns a JSON-able record: applicable, proved (bool), counts,
    residual cells (location, failure codes), timing."""
    t0 = time.time()
    rec = dict(theorem="LINKBOX", version=2)
    loops, has_tor = coil_loops(source)
    if has_tor:
        rec.update(applicable=False, proved=False,
                   reason="ToroidalField part: the z-axis current is linked by every toroidal loop (no linking "
                          "restriction of the closed lines)")
        return rec
    if not all(lp.get("exact") for lp in loops):
        rec.update(applicable=False, proved=False,
                   reason="part kinds other than polylines (Fourier curves, rings) are not handled by the cell "
                          "geometry of torux.nonstell_index")
        return rec
    G = choose_geometry(source) if geometry == "auto" else linkbox_geometry(source, margin_cap_rel, clear_frac)
    L = G["L"]
    P = lean_pack(source)
    Q, _ = bound_pieces(source)
    allp_ = np.concatenate([lp["pts"] for lp in loops if lp.get("exact")])
    rng0 = np.random.default_rng(1)
    probe0 = allp_[rng0.integers(0, len(allp_), 64)] + rng0.normal(size=(64, 3)) * 0.05 * G["L"]
    rec["lean_field_check_rel"] = lean_check(source, P, probe0)
    jrun = None
    if backend == "jax":
        jrun = jax_batch_certifier(source)
        allp = np.concatenate([lp["pts"] for lp in loops if lp.get("exact")])
        rng_ = np.random.default_rng(0)
        probe = allp[rng_.integers(0, len(allp), 64)] + rng_.normal(size=(64, 3)) * 0.05 * L
        rec["jax_field_check_rel"] = jrun.check(probe)
        try:
            import jax as _jax
            rec["jax_devices"] = [str(d) for d in _jax.devices()]
        except Exception:
            pass
    h0 = h0_rel * L
    queue = []
    for bi in G["symmetry"]["represent"]:
        queue += box_face_cells(bi, G["boxes"][bi]["box"], h0)
    for k in G["cone_coils"]:
        lp = G["loops"][k]; Pp = lp["pts"]; n = len(Pp); apex = G["apex"][k]
        for i in range(n):
            a = Pp[i]; b = Pp[(i + 1) % n]
            nt = max(1, int(math.ceil(max(np.linalg.norm(a - apex), np.linalg.norm(b - apex)) / h0)))
            for it in range(nt):
                nu = max(1, int(math.ceil((it + 1) / nt * np.linalg.norm(b - a) / h0)))
                for iu in range(nu):
                    queue.append(Cell("cone", k, i, (it / nt, (it + 1) / nt, iu / nu, (iu + 1) / nu)))
    n_all = len(queue)
    if part is not None:
        queue = [cl for i, cl in enumerate(queue) if i % part[1] == part[0]]
    _prepare(queue, G)
    stats = dict(n_initial=len(queue), n_initial_all=n_all, n_presplit=0, n_tested=0, n_certified=0, n_split=0,
                 max_s=0.0, steps=0, codes={}, by_kind={}, section={}, saddle_cells=[])
    residual = []
    level = 0
    resumed = None
    if checkpoint is not None:
        import os as _os
        import pickle as _pk
        if _os.path.exists(checkpoint):
            with open(checkpoint, "rb") as fh:
                ck = _pk.load(fh)
            if ck.get("box_params") == G.get("box_params") and ck.get("n_initial_all") == n_all:
                queue = []
                if "qarr" not in ck:                      # first checkpoint format: list of tuples
                    ck["qarr"] = dict(kind=np.array([0] * len(ck["queue"]), np.int8),
                                      kinds=sorted({t[0] for t in ck["queue"]}))
                    kinds_ = ck["qarr"]["kinds"]
                    ck["qarr"].update(
                        kind=np.array([kinds_.index(t[0]) for t in ck["queue"]], np.int8),
                        coil=np.array([t[1] for t in ck["queue"]]), piece=np.array([t[2] for t in ck["queue"]]),
                        box=np.array([tuple(t[3]) + (0.0,) * (5 - len(t[3])) for t in ck["queue"]], float),
                        nbox=np.array([len(t[3]) for t in ck["queue"]]), need=np.array([t[4] for t in ck["queue"]], bool),
                        depth=np.array([t[5] for t in ck["queue"]]))
                qa = ck["qarr"]
                for i in range(len(qa["kind"])):
                    nb_ = int(qa["nbox"][i])
                    cl = Cell(qa["kinds"][int(qa["kind"][i])], int(qa["coil"][i]), int(qa["piece"][i]),
                              tuple(float(v) for v in qa["box"][i][:nb_]),
                              (bool(qa["need"][i][0]), bool(qa["need"][i][1])), int(qa["depth"][i]))
                    if "c" in qa:
                        cl.c = np.array(qa["c"][i]); cl.E4 = np.array(qa["E4"][i]); cl.sag = float(qa["sag"][i])
                    queue.append(cl)
                _prepare(queue, G)
                stats = ck["stats"]
                residual = ck["residual"]
                level = ck["level"]
                resumed = dict(level=level, n_queue=len(queue), elapsed_before=ck.get("elapsed", 0.0))
    tol = tol_rel * L; hmax = hmax_rel * L; smax = smax_rel * L
    not_contractible = [b for b in G["boxes"] if not b["contractible"]]
    faces = faces_table(G) if transfer else np.zeros((0, 9))
    cidx = CertIndex()
    stats.setdefault("transfer", 0)
    if resumed is not None and ck.get("cidx") is not None:
        cidx = ck["cidx"]

    def register(cl, k_):
        key = _cell_key(cl, G)
        if key is not None:
            cidx.add(key, k_, cl.box[:4] if cl.kind != "sph" else None)
    while queue:
        if stats["n_tested"] > max_cells:
            budget_s = 1e-9                  # treated as an exhausted budget (checkpoint below)
            stats["stop_reason"] = "max_cells %d" % max_cells
        # pre-split cells that are large compared with their distance to the coils (no test)
        ready = []
        nxt = []
        pending = []
        Cq = np.array([cl.c for cl in queue])
        fv, _dm = eval_points(P, Cq, True)
        scale = np.linalg.norm(fv[:, :3], axis=1) / np.maximum(np.linalg.norm(fv[:, 6:15], axis=1), 1e-300)
        for qi, cl in enumerate(queue):
            d = min(float(min_dist(Q, cl.c[0], cl.c[1], cl.c[2])), float(scale[qi]))
            if cl.rad > kappa * d and cl.depth < max_depth and cl.rad >= min_rel * L:
                kids = split_cell(cl, G)
                _prepare(kids, G)
                nxt += kids
                stats["n_presplit"] += 1
            else:
                ready.append(cl)
        over = False
        for b0 in range(0, len(ready), batch):
            if budget_s and time.time() - t0 > budget_s:
                left = ready[b0:]
                over = True
                break
            bat = ready[b0:b0 + batch]
            C = np.array([cl.c for cl in bat]); E4s = np.array([cl.E4 for cl in bat])
            sags = np.array([cl.sag for cl in bat]); need = np.array([[cl.need[0], cl.need[1]] for cl in bat], np.int64)
            hm = np.full(len(bat), hmax)
            if jrun is not None:
                out = jrun(C, E4s, sags, need, tol, hmax, smax, min(nmax, 4000), Rfac, rel_margin, block=batch)
                evs = np.full((len(bat), 1, 7), -1.0)
            else:
                out, evs = cert_many(P, Q, C, E4s, sags, need, tol, hm, smax, nmax, Rfac, rel_margin, faces, 12, reseed)
            stats["n_tested"] += len(bat)
            stats["steps"] += int(out[:, 4].sum())
            if sections:
                sec = _section_pass(P, Q, G, bat, out, tol, L)
            else:
                sec = {}
            for ci, (cl, o) in enumerate(zip(bat, out)):
                if ci in sec:
                    for k_, kind_ in sec[ci].items():
                        o[k_] = 1.0
                        stats["section"][kind_] = stats["section"].get(kind_, 0) + 1
                        if kind_ == "S":
                            stats["saddle_cells"].append(dict(centre=[float(v) for v in cl.c], orientation="+" if k_ == 0 else "-"))
                nneg = cl.need[0] and o[0] != 1.0
                npos = cl.need[1] and o[1] != 1.0
                for k_, flag, sv in ((0, cl.need[0] and o[0] == 1.0, o[2]), (1, cl.need[1] and o[1] == 1.0, o[3])):
                    if flag:
                        stats["max_s"] = max(stats["max_s"], abs(sv))
                    if flag:
                        register(cl, k_)
                bk = stats["by_kind"].setdefault(cl.kind, [0, 0])
                if not (nneg or npos):
                    stats["n_certified"] += 1
                    bk[0] += 1
                    continue
                cl.need = (nneg, npos)
                pending.append((cl, o.copy(), evs[ci].copy()))
        # transfer sweeps: a failed cell whose tube crosses a box face inside cells certified for the same
        # orientation is certified (its lines pass through points whose lines are certified)
        changed = True
        while changed and transfer:
            changed = False
            rest = []
            for cl, o, ev in pending:
                nd = list(cl.need)
                for k_ in (0, 1):
                    if not nd[k_]:
                        continue
                    for e in ev:
                        if e[0] < 0:
                            break
                        if cidx.covered(int(e[0]), k_, (e[1], e[2], e[3], e[4])):
                            nd[k_] = False
                            stats["transfer"] += 1
                            register(cl, k_)
                            changed = True
                            break
                cl.need = (nd[0], nd[1])
                if nd[0] or nd[1]:
                    rest.append((cl, o, ev))
                else:
                    stats["n_certified"] += 1
                    stats["by_kind"].setdefault(cl.kind, [0, 0])[0] += 1
            pending = rest
        for cl, o, ev in pending:
            code = int(o[5])
            stats["codes"][code] = stats["codes"].get(code, 0) + 1
            if code == 5 and o.shape[0] >= 35:
                kids = image_cells(cl, o)
                stats["n_reseed"] = stats.get("n_reseed", 0) + 1
                nxt += kids
                continue
            if cl.depth >= max_depth or cl.rad < min_rel * L:
                residual.append(_res_entry(cl, G, Q, "min_size", o, cl.need))
                stats["n_min_size"] = stats.get("n_min_size", 0) + 1
                stats["by_kind"].setdefault(cl.kind, [0, 0])[1] += 1
                continue
            Mf = o[12:21] if (o[9] > 0 or o[10] > 0) else None
            kids = split_cell(cl, G, Mf)
            _prepare(kids, G)
            stats["n_split"] += 1
            nxt += kids
        msg = (f"level {level}: tested {stats['n_tested']}, certified {stats['n_certified']} (transfer "
               f"{stats['transfer']}), presplit "
               f"{stats['n_presplit']}, next {len(nxt)}, residual {len(residual)}, codes {stats['codes']}, "
               f"{time.time() - t0:.0f}s")
        if verbose:
            print(msg, flush=True)
        if log:
            try:
                with open(log, "a") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass
        queue = nxt
        level += 1
        if over:
            queue = left + queue
            if checkpoint is not None:
                import pickle as _pk
                kinds_ = sorted({cl.kind for cl in queue})
                qarr = dict(kind=np.array([kinds_.index(cl.kind) for cl in queue], np.int8),
                            coil=np.array([cl.coil for cl in queue], np.int32),
                            piece=np.array([cl.piece for cl in queue], np.int32),
                            box=np.array([tuple(cl.box) + (0.0,) * (5 - len(cl.box)) for cl in queue], float),
                            nbox=np.array([len(cl.box) for cl in queue], np.int8),
                            need=np.array([cl.need for cl in queue], bool),
                            depth=np.array([cl.depth for cl in queue], np.int16), kinds=kinds_,
                            c=np.array([cl.c for cl in queue]), E4=np.array([cl.E4 for cl in queue]),
                            sag=np.array([cl.sag for cl in queue]))
                ck = dict(box_params=G.get("box_params"), n_initial_all=n_all, level=level, stats=stats,
                          residual=residual, cidx=cidx, elapsed=time.time() - t0 + (resumed or {}).get("elapsed_before", 0.0),
                          qarr=qarr)
                import os as _os
                try:
                    with open(checkpoint + ".tmp", "wb") as fh:
                        _pk.dump(ck, fh, protocol=4)
                    _os.replace(checkpoint + ".tmp", checkpoint)
                except OSError:
                    pass
            n_left = len(queue)
            for cl in queue[:2000]:
                residual.append(_res_entry(cl, G, Q, "budget", None))
            stats["n_budget_left"] = n_left
            queue = []
    proved = len(residual) == 0 and not not_contractible and not G["axis_coils"] and not G["cone_coils"]
    rec.update(applicable=True, proved=proved, L=L, h0=h0, min_size=min_rel * L, tol=tol, hmax=hmax, backend=backend,
               part=list(part) if part is not None else None,
               smax=smax, Rfac=Rfac, rel_margin=rel_margin, kappa=kappa,
               boxes=[dict(coils=[int(j) for j in b["coils"]], box=[float(v) for v in b["box"]],
                           tight=[float(v) for v in b["tight"]], margin=float(b["margin"]),
                           contractible=b["contractible"], image_of=b.get("image_of")) for b in G["boxes"]],
               symmetry=dict(nfp=G["symmetry"]["nfp"], verified=G["symmetry"]["verified"],
                             deviation=G["symmetry"].get("deviation"),
                             represent=[int(v) for v in G["symmetry"]["represent"]]),
               cone_coils=[int(j) for j in G["cone_coils"]], geometry_choice=G.get("geometry_choice"),
               box_params=G.get("box_params"),
               stats=stats, n_residual=len([e for e in residual if e.get("why") != "budget"]) + stats.get("n_budget_left", 0),
               residual=residual[:400], resumed=resumed, checkpoint=checkpoint,
               seconds_total=time.time() - t0 + (resumed or {}).get("elapsed_before", 0.0),
               seconds=time.time() - t0)
    if residual:
        rec["residual_summary"] = residual_summary(residual)
    return rec


def _res_entry(cl, G, Q, why, o, needs=None):
    d = float(min_dist(Q, cl.c[0], cl.c[1], cl.c[2]))
    e = dict(kind=cl.kind, coil=int(cl.coil) if cl.kind in ("cone", "cyl", "sph") else None,
             box_index=int(cl.coil) if cl.kind.startswith("b") else None, piece=int(cl.piece), box=[float(v) for v in cl.box],
             centre=[float(v) for v in cl.c], R=float(math.hypot(cl.c[0], cl.c[1])), Z=float(cl.c[2]),
             phi=float(math.atan2(cl.c[1], cl.c[0])), rad=float(cl.rad), dist_to_coil=d, why=why)
    if o is not None:
        e.update(code=int(o[5]), r_last=float(o[6]), dphi=[float(o[7]), float(o[8])], s=[float(o[9]), float(o[10])])
    if needs is not None:
        e["open_orientations"] = ([] + (["+"] if needs[0] else []) + (["-"] if needs[1] else []))
    return e


def residual_summary(res):
    d = np.array([r["dist_to_coil"] for r in res])
    kinds = {}
    for r in res:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    codes = {}
    for r in res:
        codes[r.get("code", -1)] = codes.get(r.get("code", -1), 0) + 1
    return dict(n=len(res), kinds=kinds, codes=codes, dist_to_coil_min=float(d.min()), dist_to_coil_max=float(d.max()),
                R_range=[float(min(r["R"] for r in res)), float(max(r["R"] for r in res))],
                Z_range=[float(min(r["Z"] for r in res)), float(max(r["Z"] for r in res))])


# ============================================================================ LINKBOX geometry (boxes in cylindrical coordinates)
def _seg_extents(a, b):
    """Exact (Rmin, Rmax, Zmin, Zmax, phi arc (start, length)) of a straight segment off the z axis."""
    a2 = np.array([a[0], a[1], 0.0]); b2 = np.array([b[0], b[1], 0.0])
    Rmin = _seg_dist(0.0, 0.0, 0.0, a2, b2)
    Rmax = max(math.hypot(a[0], a[1]), math.hypot(b[0], b[1]))
    pa = math.atan2(a[1], a[0]); pb = math.atan2(b[1], b[0])
    dp = (pb - pa + math.pi) % TWO_PI - math.pi
    start = pa if dp >= 0 else pb
    return Rmin, Rmax, min(a[2], b[2]), max(a[2], b[2]), start, abs(dp)


def _arc_union(arcs, nbins=36000):
    """Smallest arc (start, length) containing every arc (start, length); length >= 2 pi if none."""
    w = TWO_PI / nbins
    cov = np.zeros(nbins, bool)
    for st, ln in arcs:
        k0 = int(math.floor(st / w)); k1 = int(math.ceil((st + ln) / w))
        if k1 - k0 >= nbins:
            return 0.0, 4 * math.pi
        cov[np.arange(k0, k1 + 1) % nbins] = True
    if cov.all():
        return 0.0, 4 * math.pi
    # largest cyclic gap of uncovered bins
    idx = np.flatnonzero(cov)
    gaps = np.diff(np.concatenate([idx, [idx[0] + nbins]]))
    g = int(np.argmax(gaps))
    start = (idx[(g + 1) % len(idx)]) * w
    length = nbins * w - (gaps[g] - 1) * w
    return start % TWO_PI, length


def coil_box(pts):
    """Tight cylindrical box (R1, R2, Z1, Z2, phi1, phi_len) of a closed polygon (exact for segments)."""
    n = len(pts)
    R1 = np.inf; R2 = 0.0; Z1 = np.inf; Z2 = -np.inf; arcs = []
    for i in range(n):
        r1, r2, z1, z2, st, ln = _seg_extents(pts[i], pts[(i + 1) % n])
        R1 = min(R1, r1); R2 = max(R2, r2); Z1 = min(Z1, z1); Z2 = max(Z2, z2); arcs.append((st, ln))
    st, ln = _arc_union(arcs)
    return [R1, R2, Z1, Z2, st, ln]


def box_contains(box, inner, tol=1e-12):
    """Does the cylindrical box contain the tight box inner (both (R1, R2, Z1, Z2, phi1, phi_len))?"""
    if not (box[0] <= inner[0] + tol and inner[1] <= box[1] + tol and box[2] <= inner[2] + tol and
            inner[3] <= box[3] + tol):
        return False
    rel = (inner[4] - box[4]) % TWO_PI
    return rel + inner[5] <= box[5] + tol


def verify_geometry(source, rec):
    """Independent re-check of the geometric hypotheses of a LINKBOX certificate record: every coil with
    I != 0 has toroidal winding 0, lies off the z axis and inside a box of the record (exact segment
    extents), every box is contractible (R1 > 0, arc < 2 pi), the boxes not certified themselves are
    rotated copies of certified ones under a verified symmetry.  Returns (ok, list of problems)."""
    from .nonstell import toroidal_winding
    loops, has_tor = coil_loops(source)
    probs = []
    if has_tor:
        probs.append("ToroidalField part")
    boxes = rec.get("boxes") or []
    for b in boxes:
        bx = b["box"]
        if not (bx[0] > 0 and bx[5] < TWO_PI):
            probs.append("box not contractible: %s" % bx)
    for j, lp in enumerate(loops):
        if not lp.get("exact"):
            probs.append("coil %d: part kind %s" % (j, lp["kind"]))
            continue
        if lp["I"] == 0.0:
            continue
        if toroidal_winding(lp["pts"]) != 0:
            probs.append("coil %d winds around the z axis" % j)
            continue
        t = coil_box(lp["pts"])
        if t[0] <= 0:
            probs.append("coil %d meets the z axis" % j)
            continue
        if not any(box_contains(b["box"], t) for b in boxes):
            probs.append("coil %d is in no box" % j)
    rep = set((rec.get("symmetry") or {}).get("represent") or range(len(boxes)))
    for bi, b in enumerate(boxes):
        if bi in rep:
            continue
        io = b.get("image_of")
        if not io or not (rec.get("symmetry") or {}).get("verified"):
            probs.append("box %d neither certified nor a verified rotated copy" % bi)
            continue
        rb = list(boxes[io[0]]["box"]); rb[4] = (rb[4] + TWO_PI * io[1] / rec["symmetry"]["nfp"]) % TWO_PI
        if max(abs(rb[k] - b["box"][k]) for k in range(4)) > 1e-12 or \
                abs(((rb[4] - b["box"][4] + math.pi) % TWO_PI) - math.pi) > 1e-12:
            probs.append("box %d is not the rotated copy of box %d" % (bi, io[0]))
    if (rec.get("symmetry") or {}).get("verified"):
        ok_, dev = source_rotation_symmetric(source, rec["symmetry"]["nfp"])
        if not ok_:
            probs.append("rotation symmetry not confirmed")
    return (not probs), probs


def _box_union(b1, b2):
    st, ln = _arc_union([(b1[4], b1[5]), (b2[4], b2[5])])
    return [min(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), max(b1[3], b2[3]), st, ln]


def _pt_box_dist(p, box):
    """Distance-like separation (>= 0 outside, < 0 inside) of points p (m, 3) from a cylindrical box:
    max over the three coordinate excesses (R, Z, arc length R dphi)."""
    R = np.hypot(p[:, 0], p[:, 1]); Z = p[:, 2]; ph = np.arctan2(p[:, 1], p[:, 0])
    dR = np.maximum(box[0] - R, R - box[1])
    dZ = np.maximum(box[2] - Z, Z - box[3])
    rel = (ph - box[4]) % TWO_PI                     # in [0, 2 pi)
    over = np.where(rel <= box[5], -np.minimum(rel, box[5] - rel), np.minimum(rel - box[5], TWO_PI - rel))
    dP = over * R
    return np.maximum(np.maximum(dR, dZ), dP)


def _densify(pts, h):
    n = len(pts); out = []
    for i in range(n):
        a = pts[i]; b = pts[(i + 1) % n]
        m = max(1, int(math.ceil(np.linalg.norm(b - a) / h)))
        out.append(a + np.outer(np.arange(m) / m, b - a))
    return np.concatenate(out)


def link_boxes(loops, margin_cap_rel=0.25, clear_frac=0.45, merge_rel=0.03, max_phi=1.8 * math.pi):
    """Groups of w_z = 0 coils with one contractible cylindrical box each (module docstring, LINKBOX).
    Every coil starts in its own group; two groups are merged only when their tight boxes come closer
    than merge_rel x the smaller extent and the merged box stays in a wedge of angle < max_phi.  The
    margin of a group's box is min(margin_cap_rel x extent, clear_frac x distance to the other coils,
    0.9 x R_min).  Returns list of dict(coils, tight, box (R1, R2, Z1, Z2, phi1, phi_len), margin,
    contractible)."""
    idx = [j for j, lp in enumerate(loops) if lp.get("exact") and lp["I"] != 0.0 and lp["w_z"] == 0
           and coil_box(lp["pts"])[0] > 0.0]
    allpts = {j: _densify(loops[j]["pts"], 0.01 * max(1e-3, float(np.max(np.ptp(loops[j]["pts"], 0)))))
              for j in range(len(loops)) if loops[j].get("exact")}

    def ext_of(t):
        return max(t[1] - t[0], t[3] - t[2], t[5] * t[1])
    groups = [dict(coils=[j], tight=coil_box(loops[j]["pts"])) for j in idx]
    changed = True
    while changed:
        changed = False
        for g in groups:
            for h in groups:
                if h is g:
                    continue
                d = min(float(np.min(_pt_box_dist(allpts[j], g["tight"]))) for j in h["coils"])
                if d < merge_rel * min(ext_of(g["tight"]), ext_of(h["tight"])):
                    u = g["tight"]
                    for j in h["coils"]:
                        u = _box_union(u, coil_box(loops[j]["pts"]))
                    if u[5] < max_phi:
                        g["coils"] += h["coils"]
                        g["tight"] = u
                        groups.remove(h)
                        changed = True
                        break
            if changed:
                break
    out = []
    for g in groups:
        t = g["tight"]
        ext = ext_of(t)
        others = [j for j in allpts if j not in g["coils"]]
        op = np.concatenate([allpts[j] for j in others]) if others else np.zeros((0, 3))
        dmin = float(np.min(_pt_box_dist(op, t))) if len(op) else np.inf
        m = min(margin_cap_rel * ext, clear_frac * dmin, 0.9 * t[0])
        # per-face margins: each face moves out as far as the other coils in front of it allow
        cap = min(margin_cap_rel * ext, 0.9 * t[0])
        if len(op) and m > 0:
            R = np.hypot(op[:, 0], op[:, 1]); Z = op[:, 2]
            ph = np.arctan2(op[:, 1], op[:, 0])
            rel = (ph - t[4]) % TWO_PI
            arc = np.where(rel <= t[5], 0.0, np.minimum(rel - t[5], TWO_PI - rel))
            ex_lo_R = t[0] - R; ex_hi_R = R - t[1]; ex_lo_Z = t[2] - Z; ex_hi_Z = Z - t[3]
            ex_p1 = np.where(rel > t[5], TWO_PI - rel, -1.0) * R          # beyond phi1 (arc length)
            ex_p2 = np.where(rel > t[5], rel - t[5], -1.0) * R            # beyond phi2
            exc = np.stack([ex_lo_R, ex_hi_R, ex_lo_Z, ex_hi_Z, np.minimum(ex_p1, ex_p2)], 1)
            margins = []
            for f in range(4):
                # points whose largest excess is in this face's direction limit this face
                lim = exc[:, f] >= np.max(exc, axis=1) - 1e-15
                dm = float(np.min(exc[lim, f])) if np.any(lim) else np.inf
                margins.append(min(cap, clear_frac * dm) if dm > 0 else m)
            mf = [max(m, v) for v in margins]
            box = [t[0] - mf[0], t[1] + mf[1], t[2] - mf[2], t[3] + mf[3], t[4] - m / t[0], t[5] + 2 * m / t[0]]
            if box[0] <= 0 or float(np.min(_pt_box_dist(op, box))) < 0.5 * m:
                box = [t[0] - m, t[1] + m, t[2] - m, t[3] + m, t[4] - m / t[0], t[5] + 2 * m / t[0]]
        else:
            box = [t[0] - m, t[1] + m, t[2] - m, t[3] + m, t[4] - m / t[0], t[5] + 2 * m / t[0]]
        ok = box[5] < TWO_PI - 1e-6 and box[0] > 0 and m > 0
        out.append(dict(coils=g["coils"], tight=t, box=box, margin=m, contractible=bool(ok)))
    return out


def box_face_cells(bi, box, h0):
    """Initial cells on the six faces of box bi."""
    R1, R2, Z1, Z2, p1, pl = box
    cells = []

    def grid(n1, n2):
        return [(i / n1, (i + 1) / n1, j / n2, (j + 1) / n2) for i in range(n1) for j in range(n2)]
    nR = max(1, int(math.ceil((R2 - R1) / h0))); nZ = max(1, int(math.ceil((Z2 - Z1) / h0)))
    nP = max(1, int(math.ceil(pl * R2 / h0)))
    for face in ("p1", "p2"):
        for u0, u1, v0, v1 in grid(nR, nZ):
            cells.append(Cell("bphi", bi, 0 if face == "p1" else 1, (u0, u1, v0, v1)))
    for face in ("R1", "R2"):
        for u0, u1, v0, v1 in grid(nP, nZ):
            cells.append(Cell("bR", bi, 0 if face == "R1" else 1, (u0, u1, v0, v1)))
    for face in ("Z1", "Z2"):
        for u0, u1, v0, v1 in grid(nR, nP):
            cells.append(Cell("bZ", bi, 0 if face == "Z1" else 1, (u0, u1, v0, v1)))
    return cells


def box_cell_geometry(cell, box):
    R1, R2, Z1, Z2, p1, pl = box
    u0, u1, v0, v1 = cell.box
    k = cell.kind
    if k == "bphi":
        ph = p1 if cell.piece == 0 else p1 + pl
        c_, s_ = math.cos(ph), math.sin(ph)
        Ra, Rb = R1 + u0 * (R2 - R1), R1 + u1 * (R2 - R1)
        Za, Zb = Z1 + v0 * (Z2 - Z1), Z1 + v1 * (Z2 - Z1)
        corners = np.array([[R * c_, R * s_, Z] for Z in (Za, Zb) for R in (Ra, Rb)])
        sag = 0.0
    elif k == "bR":
        R = R1 if cell.piece == 0 else R2
        pa, pb = p1 + u0 * pl, p1 + u1 * pl
        Za, Zb = Z1 + v0 * (Z2 - Z1), Z1 + v1 * (Z2 - Z1)
        corners = np.array([[R * math.cos(p), R * math.sin(p), Z] for Z in (Za, Zb) for p in (pa, pb)])
        sag = R * (1.0 - math.cos(0.5 * (pb - pa))) * 1.0001
    else:
        Z = Z1 if cell.piece == 0 else Z2
        Ra, Rb = R1 + u0 * (R2 - R1), R1 + u1 * (R2 - R1)
        pa, pb = p1 + v0 * pl, p1 + v1 * pl
        corners = np.array([[R * math.cos(p), R * math.sin(p), Z] for p in (pa, pb) for R in (Ra, Rb)])
        sag = Rb * (1.0 - math.cos(0.5 * (pb - pa))) * 1.0001
    c = corners.mean(0)
    return c, corners - c, sag


# ============================================================================ period map in phi with tube bound (N, S)
@njit(cache=True, error_model="numpy")
def _phirhs(P, phi, y, dy, buf, ex, sgn):
    """Field-line equations in phi for y = (R, Z, M11, M12, M21, M22): (R, Z)' = R (B_R, B_Z) / B_phi,
    M' = Df M.  ex receives (sgn B_phi, |B|, |DB|_F, R).  Returns False where sgn B_phi <= 0."""
    c = math.cos(phi); s = math.sin(phi)
    R = y[0]; Z = y[1]
    eval_point(P, R * c, R * s, Z, True, buf)
    Bx = buf[0]; By = buf[1]; Bz = buf[2]
    BR = Bx * c + By * s
    Bp = -Bx * s + By * c
    if not (sgn * Bp > 0.0):
        return False
    # DB columns along e_R = (c, s, 0) and e_z
    dRB0 = buf[6] * c + buf[7] * s; dRB1 = buf[9] * c + buf[10] * s; dRB2 = buf[12] * c + buf[13] * s
    dZB0 = buf[8]; dZB1 = buf[11]; dZB2 = buf[14]
    dR_BR = dRB0 * c + dRB1 * s; dR_Bp = -dRB0 * s + dRB1 * c; dR_BZ = dRB2
    dZ_BR = dZB0 * c + dZB1 * s; dZ_Bp = -dZB0 * s + dZB1 * c; dZ_BZ = dZB2
    f1 = R * BR / Bp; f2 = R * Bz / Bp
    a11 = BR / Bp + R * (dR_BR / Bp - BR * dR_Bp / (Bp * Bp))
    a12 = R * (dZ_BR / Bp - BR * dZ_Bp / (Bp * Bp))
    a21 = Bz / Bp + R * (dR_BZ / Bp - Bz * dR_Bp / (Bp * Bp))
    a22 = R * (dZ_BZ / Bp - Bz * dZ_Bp / (Bp * Bp))
    dy[0] = f1; dy[1] = f2
    dy[2] = a11 * y[2] + a12 * y[4]; dy[3] = a11 * y[3] + a12 * y[5]
    dy[4] = a21 * y[2] + a22 * y[4]; dy[5] = a21 * y[3] + a22 * y[5]
    nd = 0.0
    for k in range(6, 15):
        nd += buf[k] * buf[k]
    ex[0] = sgn * Bp; ex[1] = math.sqrt(Bx * Bx + By * By + Bz * Bz); ex[2] = math.sqrt(nd); ex[3] = R
    ex[4] = math.sqrt(BR * BR + Bz * Bz)
    return True


@njit(cache=True)
def _spec2(a, b, c, d):
    """Spectral norm of the 2x2 matrix [[a, b], [c, d]] (exact formula, slightly enlarged)."""
    s1 = a * a + b * b + c * c + d * d
    det = a * d - b * c
    disc = max(s1 * s1 - 4.0 * det * det, 0.0)
    return math.sqrt(0.5 * (s1 + math.sqrt(disc))) * (1.0 + 1e-13) + 1e-300


@njit(cache=True, error_model="numpy")
def phimap(P, Q, phi0, dphi_total, R0, Z0, rho, tol, hmax, nmax, Rfac, sgn, out, diag):
    """Validated map of the disc of radius rho around (R0, Z0) on the half plane phi0 to phi0 + dphi_total.
    out = [ok, R1, Z1, M11, M12, M21, M22, E (|P(z) - P(c) - M (z - c)| bound), dM (bound of
    ||DP(z) - M||), eP (error of P(c)), eM (error of M), bphi_min (lower bound of B_phi on the tube), steps, code]
    code: 0 ok, 1 tube bootstrap failed, 3 field bound failure / B_phi not certified > 0, 4 step limit."""
    buf = np.empty(15); ex = np.empty(5)
    y = np.zeros(6); y[0] = R0; y[1] = Z0; y[2] = 1.0; y[5] = 1.0
    k1 = np.empty(6); k2 = np.empty(6); k3 = np.empty(6); k4 = np.empty(6); k5 = np.empty(6); k6 = np.empty(6)
    k7 = np.empty(6); t = np.empty(6); yn = np.empty(6)
    for k in range(out.shape[0]):
        out[k] = 0.0
    phi = phi0
    if not _phirhs(P, phi, y, k1, buf, ex, sgn):
        out[13] = 3.0
        return
    h = min(hmax, 0.1 * dphi_total)
    done = 0.0
    idg = 0
    N = 0.0; eP = 0.0; eM = 0.0; dM = 0.0
    r = rho
    bmin_all = np.inf
    steps = 0
    code = 4
    for it in range(nmax):
        steps += 1
        last = False
        if done + h >= dphi_total:
            h = dphi_total - done
            last = True
        hs_ = sgn * h
        for q in range(6):
            t[q] = y[q] + hs_ * (0.2 * k1[q])
        o2 = _phirhs(P, phi + 0.2 * hs_, t, k2, buf, ex, sgn)
        for q in range(6):
            t[q] = y[q] + hs_ * (3.0 / 40 * k1[q] + 9.0 / 40 * k2[q])
        o3 = _phirhs(P, phi + 0.3 * hs_, t, k3, buf, ex, sgn)
        for q in range(6):
            t[q] = y[q] + hs_ * (44.0 / 45 * k1[q] - 56.0 / 15 * k2[q] + 32.0 / 9 * k3[q])
        o4 = _phirhs(P, phi + 0.8 * hs_, t, k4, buf, ex, sgn)
        for q in range(6):
            t[q] = y[q] + hs_ * (19372.0 / 6561 * k1[q] - 25360.0 / 2187 * k2[q] + 64448.0 / 6561 * k3[q]
                               - 212.0 / 729 * k4[q])
        o5 = _phirhs(P, phi + 8.0 / 9.0 * hs_, t, k5, buf, ex, sgn)
        for q in range(6):
            t[q] = y[q] + hs_ * (9017.0 / 3168 * k1[q] - 355.0 / 33 * k2[q] + 46732.0 / 5247 * k3[q]
                               + 49.0 / 176 * k4[q] - 5103.0 / 18656 * k5[q])
        o6 = _phirhs(P, phi + hs_, t, k6, buf, ex, sgn)
        for q in range(6):
            yn[q] = y[q] + hs_ * (35.0 / 384 * k1[q] + 500.0 / 1113 * k3[q] + 125.0 / 192 * k4[q]
                                - 2187.0 / 6784 * k5[q] + 11.0 / 84 * k6[q])
        o7 = _phirhs(P, phi + hs_, yn, k7, buf, ex, sgn)
        if not (o2 and o3 and o4 and o5 and o6 and o7):
            if h > 1e-9:
                h *= 0.25
                continue
            code = 3
            break
        # geometric step limit: the 3D chord of the step stays below 0.1 x the local length scale
        Rc = ex[3]
        lscale = min(min_dist(Q, Rc * math.cos(phi + hs_), Rc * math.sin(phi + hs_), yn[1]), ex[1] / max(ex[2], 1e-300))
        if Rc * h > 0.1 * lscale and h > 1e-9:
            h = 0.09 * lscale / Rc
            continue
        errx = 0.0; errM = 0.0
        for q in range(6):
            e = abs(h * (71.0 / 57600 * k1[q] - 71.0 / 16695 * k3[q] + 71.0 / 1920 * k4[q]
                         - 17253.0 / 339200 * k5[q] + 22.0 / 525 * k6[q] - 1.0 / 40 * k7[q]))
            if q < 2:
                errx = max(errx, e)
            else:
                errM = max(errM, e)
        errx *= math.sqrt(2.0); errM *= 2.0
        err = max(errx, errM * rho)
        if err > tol and h > 1e-9:
            h = h * max(0.2, 0.9 * (tol / err) ** 0.2)
            continue
        Bp1 = ex[0]; nb1 = ex[1]; nd1 = ex[2]; R1 = ex[3]
        # propagator norm and tube bootstrap
        det0 = y[2] * y[5] - y[3] * y[4]
        if det0 == 0.0:
            code = 1
            break
        i11 = y[5] / det0; i12 = -y[3] / det0; i21 = -y[4] / det0; i22 = y[2] / det0
        A11 = yn[2] * i11 + yn[3] * i21; A12 = yn[2] * i12 + yn[3] * i22
        A21 = yn[4] * i11 + yn[5] * i21; A22 = yn[4] * i12 + yn[5] * i22
        a = _spec2(A11, A12, A21, A22) * (1.0 + 1e-9)
        My = _spec2(y[2], y[3], y[4], y[5]); Mn = _spec2(yn[2], yn[3], yn[4], yn[5])
        step = math.sqrt((yn[0] - y[0]) ** 2 + (yn[1] - y[1]) ** 2)
        Rt = Rfac * r
        ok = False
        Kf = 0.0; Lf = 0.0; bmin = 0.0
        Nn = 0.0; ePn = 0.0; eMn = 0.0; dMn = 0.0; rn = 0.0
        for attempt in range(5):
            # ball in 3D around the step end containing the cross-sections of the tube during the step:
            # (R, Z) within Rt + step, phi within h (arc R h)
            rad = Rt + step + eP + 10.0 * errx + (R1 + Rt + step) * h
            xx = R1 * math.cos(phi + hs_); yy = R1 * math.sin(phi + hs_)
            d2 = d2b_ball(P, Q, xx, yy, yn[1], rad, nd1, buf)
            if not (d2 < np.inf):
                break
            D1 = nd1 + rad * d2
            Gmax = ex[4] + rad * D1                      # |(B_R, B_Z)| on the ball
            bmin = Bp1 - rad * D1
            if bmin <= 0.0:
                break
            Rmax = R1 + rad
            # h = (B_R, B_Z) / B_phi (2-vector), f = R h; derivatives in (R, Z) at fixed phi
            dh = D1 / bmin + Gmax * D1 / (bmin * bmin)
            d2h = d2 / bmin + 2 * D1 * D1 / (bmin * bmin) + Gmax * d2 / (bmin * bmin) + 2 * Gmax * D1 * D1 / bmin ** 3
            Kf = 2 * dh + Rmax * d2h
            Lf = Gmax / bmin + Rmax * dh
            eh = math.exp(h * Lf)
            if eh * (r + 0.5 * h * Kf * Rt * Rt) > Rt:
                Rt = 2.0 * eh * (r + 0.5 * h * Kf * Rt * Rt)
                continue
            Nn = a * N + h * eh * 0.5 * Kf * Rt * Rt
            ePn = a * eP + 10.0 * errx
            eMn = a * eM + 10.0 * errM + h * eh * Kf * eP * My
            dMn = a * dM + h * eh * Kf * Rt * (max(My, Mn) + dM)
            rn = Mn * rho + Nn + ePn + eMn * rho
            if rn <= Rt:
                ok = True
                break
            Rt = 2.0 * rn
        if not ok:
            code = 1 if bmin > 0.0 else 3
            break
        if idg < diag.shape[0]:
            diag[idg, 0] = phi + hs_; diag[idg, 1] = a; diag[idg, 2] = errx; diag[idg, 3] = errM
            diag[idg, 4] = Kf; diag[idg, 5] = Rt; diag[idg, 6] = Nn; diag[idg, 7] = ePn; diag[idg, 8] = dMn
            diag[idg, 9] = h
            idg += 1
        N = Nn; eP = ePn; eM = eMn; dM = dMn; r = rn
        bmin_all = min(bmin_all, bmin)
        phi += hs_
        done += h
        for q in range(6):
            y[q] = yn[q]; k1[q] = k7[q]
        if last:
            code = 0
            break
        fac = 0.9 * (tol / max(err, 1e-300)) ** 0.2
        h = min(hmax, h * min(4.0, max(0.2, fac)))
    out[0] = 1.0 if code == 0 else 0.0
    out[1] = y[0]; out[2] = y[1]; out[3] = y[2]; out[4] = y[3]; out[5] = y[4]; out[6] = y[5]
    out[7] = N + eM * rho; out[8] = dM + eM; out[9] = eP; out[10] = eM; out[11] = bmin_all
    out[12] = steps; out[13] = code


@njit(parallel=True, cache=True, error_model="numpy")
def phimap_many(P, Q, phi0s, R0s, Z0s, rhos, sgns, tol, hmax, nmax, Rfac):
    n = R0s.shape[0]
    out = np.zeros((n, 14))
    for i in prange(n):
        o = np.zeros(14)
        phimap(P, Q, phi0s[i], TWO_PI, R0s[i], Z0s[i], rhos[i], tol, hmax, nmax, Rfac, sgns[i], o, np.zeros((0, 10)))
        for k in range(14):
            out[i, k] = o[k]
    return out


def section_verdict(out, R0, Z0, rho):
    """'N' / 'S' / None from a phimap output for the disc of radius rho around (R0, Z0) (certify_box_section)."""
    if out[0] != 1.0:
        return None, {}
    M = np.array([[out[3], out[4]], [out[5], out[6]]])
    f0 = np.array([out[1] - R0, out[2] - Z0])
    E = out[7]; dM = out[8]; eP = out[9]
    ImM = M - np.eye(2)
    snorm = np.linalg.norm(ImM, 2)
    info = dict(disp=float(np.linalg.norm(f0)), E=float(E), dM=float(dM), eP=float(eP))
    if np.linalg.norm(f0) > snorm * rho * (1 + 1e-12) + E + eP:
        return "N", info
    if abs(np.linalg.det(ImM)) < 1e-300:
        return None, info
    Y = np.linalg.inv(ImM)
    nY = np.linalg.norm(Y, 2)
    Cm = np.eye(2) - Y @ ImM
    rad_k = np.linalg.norm(Y @ f0) + nY * (eP + E + dM * rho) + np.linalg.norm(Cm, 2) * rho
    tr = float(np.trace(M))
    info.update(krawczyk_radius=float(rad_k), trace=tr)
    if rad_k < rho and tr - 2.0 * (dM + out[10]) * math.sqrt(2.0) > 2.0:
        info["fixed_point_disc"] = [float(R0 - (Y @ f0)[0]), float(Z0 - (Y @ f0)[1]), float(rad_k)]
        return "S", info
    return None, info


def certify_box_section(P, Q, phi0, R0, Z0, rho, L, sgn=1.0, tol_rel=1e-12, hmax=0.05, nmax=200000, Rfac=1.3):
    """Certificates for the disc of radius rho around (R0, Z0) on the half plane phi0 (orientation +,
    lines completing one toroidal turn with B_phi > 0 on the whole tube):
      'N': no fixed point of P in the disc (|P(c) - c| > sup |(M - I)(z - c)| + E + eP);
      'S': exactly one fixed point (Krawczyk with Y = (M - I)^-1) and it is a regular saddle
           (trace bound > 2; det DP = 1 at a fixed point by flux conservation).
    Returns dict(kind in {'N', 'S', None}, data)."""
    out = np.zeros(14)
    phimap(P, Q, phi0, TWO_PI, R0, Z0, rho, tol_rel * L, hmax, nmax, Rfac, sgn, out, np.zeros((0, 10)))
    res = dict(ok_map=bool(out[0] == 1.0), code=int(out[13]), steps=int(out[12]))
    if out[0] != 1.0:
        res["kind"] = None
        return res
    M = np.array([[out[3], out[4]], [out[5], out[6]]])
    f0 = np.array([out[1] - R0, out[2] - Z0])
    E = out[7]; dM = out[8]; eP = out[9]
    ImM = M - np.eye(2)
    snorm = np.linalg.norm(ImM, 2)
    res.update(P=[float(out[1]), float(out[2])], M=M.tolist(), E=float(E), dM=float(dM), eP=float(eP),
               disp=float(np.linalg.norm(f0)))
    if np.linalg.norm(f0) > snorm * rho * (1 + 1e-12) + E + eP:
        res["kind"] = "N"
        return res
    # Krawczyk
    try:
        Y = np.linalg.inv(ImM)
    except np.linalg.LinAlgError:
        res["kind"] = None
        return res
    nY = np.linalg.norm(Y, 2)
    C = np.eye(2) - Y @ ImM
    rad_k = np.linalg.norm(Y @ f0) + nY * (eP + E + dM * rho) + np.linalg.norm(C, 2) * rho
    tr = float(np.trace(M))
    res.update(krawczyk_radius=float(rad_k), trace=tr)
    if rad_k < rho and tr - 2.0 * (dM + out[10]) * math.sqrt(2.0) > 2.0:
        res["kind"] = "S"
        res["fixed_point_disc"] = [float(R0 - (Y @ f0)[0]), float(Z0 - (Y @ f0)[1]), float(rad_k)]
    else:
        res["kind"] = None
    return res


# ============================================================================ model fields (tests)
def _poly_source(sid, loops, currents):
    from .source import Source, PolylineCurves
    pts = np.concatenate(loops)
    start = np.concatenate([[0], np.cumsum([len(l) for l in loops])])
    return Source(id=sid, parts=[PolylineCurves(pts, start, np.asarray(currents, float))], family="model")


def model_saddle(n_tf=12, R0=1.0, a=0.5, I_tf=-2.0e5, R_pf=1.0, Z_pf=0.25, I_pf=1.0e4, m=96):
    """TF coils (circles of radius a in the planes phi_k, w_z = 0) and two equal PF rings (w_z = 1) at
    (R_pf, +-Z_pf): the poloidal null between the rings makes a regular hyperbolic period-1 line near
    (R_pf, 0) (up-down and n_tf-fold symmetric)."""
    loops, cur = [], []
    th = TWO_PI * np.arange(m) / m
    for k in range(n_tf):
        ph = TWO_PI * k / n_tf
        R = R0 + a * np.cos(th); Z = a * np.sin(th)
        loops.append(np.stack([R * math.cos(ph), R * math.sin(ph), Z], -1)); cur.append(I_tf)
    t = TWO_PI * np.arange(4 * m) / (4 * m)
    for z in (Z_pf, -Z_pf):
        loops.append(np.stack([R_pf * np.cos(t), R_pf * np.sin(t), np.full_like(t, z)], -1)); cur.append(I_pf)
    return _poly_source("model/tf_pf_saddle", loops, cur)


def model_elliptic(n_c=16, R0=1.0, a=0.45, b=0.25, I=-2.0e5, m=96):
    """Rotating-ellipse modular coils (planar ellipses in the planes phi_k, rotated by theta_k = phi_k,
    w_z = 0): a two-period stellarator whose magnetic axis is an elliptic period-1 line."""
    loops, cur = [], []
    th = TWO_PI * np.arange(m) / m
    for k in range(n_c):
        ph = TWO_PI * k / n_c
        rot = ph
        u = a * np.cos(th); v = b * np.sin(th)
        dR = u * math.cos(rot) - v * math.sin(rot); dZ = u * math.sin(rot) + v * math.cos(rot)
        R = R0 + dR
        loops.append(np.stack([R * math.cos(ph), R * math.sin(ph), dZ], -1)); cur.append(I)
    return _poly_source("model/rotating_ellipse", loops, cur)


def find_period1(source, R0, Z0, phi0=0.0, sgn=1.0, its=12, tol=1e-13):
    """Newton refinement of a period-1 point on the half plane phi0 with the phi-map (centre values only)."""
    from .nonstell_cert import bound_pieces as _bp
    P = lean_pack(source); Q, _ = _bp(source)
    x = np.array([R0, Z0], float)
    out = np.zeros(14)
    for _ in range(its):
        phimap(P, Q, phi0, TWO_PI, x[0], x[1], 0.0, 1e-13, 0.05, 200000, 1.3, sgn, out, np.zeros((0, 10)))
        if out[0] != 1.0:
            return None
        f = np.array([out[1] - x[0], out[2] - x[1]])
        M = np.array([[out[3], out[4]], [out[5], out[6]]])
        dx = np.linalg.solve(M - np.eye(2), -f)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return dict(R=float(x[0]), Z=float(x[1]), trace=float(np.trace(M)), disp=float(np.linalg.norm(f)))


# ============================================================================ scope (no certification run)
def scope_record(source):
    """What LINKBOX can cover for this source without running it: applicability, boxes, and the parts of
    the linking argument that are open (ToroidalField part, non-polyline parts, coils winding around the
    z axis, coils through the z axis, groups without a contractible box)."""
    loops, has_tor = coil_loops(source)
    rec = dict(theorem="LINKBOX", run=False)
    open_ = []
    if has_tor:
        open_.append("ToroidalField part: every toroidal loop links the z-axis current (no linking restriction)")
    kinds = sorted({lp["kind"] for lp in loops if not lp.get("exact")})
    if kinds:
        open_.append("parts %s (only polyline coils are handled by the cells)" % ", ".join(kinds))
    if not has_tor and not kinds:
        G = linkbox_geometry(source)
        nc = [b for b in G["boxes"] if not b["contractible"]]
        rec.update(n_boxes=len(G["boxes"]), n_represent=len(G["symmetry"]["represent"]),
                   cone_coils=[int(j) for j in G["cone_coils"]], axis_coils=[int(j) for j in G["axis_coils"]],
                   noncontractible_groups=[[int(j) for j in b["coils"]] for b in nc])
        if G["cone_coils"]:
            open_.append("%d coil(s) winding around the z axis (w_z != 0, PF-type rings): no contractible box off the "
                         "axis contains them; closed lines linking them cross their Seifert discs, which reach the z axis "
                         "and the wire (near-axis loops and helical lines next to the wire are not certified)"
                         % len(G["cone_coils"]))
        if G["axis_coils"]:
            open_.append("%d coil(s) passing through the z axis (no box off the axis)" % len(G["axis_coils"]))
        if nc:
            open_.append("%d coil group(s) without a contractible box (coils interleaved around the torus)" % len(nc))
    rec["applicable"] = not open_
    rec["open"] = open_
    return rec


# ============================================================================ JAX batch certifier (GPU)
def jax_batch_certifier(source, chunk=128):
    """Batched LOST certificates for polyline sources in JAX (GPU when available; float64 throughout).
    Same mathematics as cert_cell (unnormalised field-line time, DP5(4) with the variational matrix,
    variation-of-constants tube bound with the analytic |D^2 B| piece bound, second-order Taylor test of
    u = R B_phi with the analytic bound), except that the sharp central-difference refinements of
    cert_cell are not used.  The field is a plain float64 sum of the closed-form straight-segment field;
    its agreement with torux.reference.fast64 is checked on sample points of every run (validated model).
    Returns run(C, E4s, sags, need, tol, hmax, smax, nmax, Rfac, rel_margin) -> out (n, 21) like cert_many."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    jax.config.update("jax_enable_x64", True)
    from .source import MU0
    A_, B_, I_ = [], [], []
    for p in source.parts:
        if p.kind == "polyline":
            sg = p.as_segments()
            A_.append(sg.a); B_.append(sg.b); I_.append(sg.current)
        elif p.kind == "segments":
            A_.append(p.a); B_.append(p.b); I_.append(p.current)
        else:
            raise ValueError("jax_batch_certifier: polyline/segment sources only")
    A0 = np.concatenate(A_); B0 = np.concatenate(B_); W0 = np.concatenate(I_) * (MU0 / (4 * math.pi))
    keep = (np.linalg.norm(B0 - A0, axis=1) > 0) & (W0 != 0)
    A0, B0, W0 = A0[keep], B0[keep], W0[keep]
    S = len(W0)
    pad = (-S) % chunk
    # padded segments: zero weight, far away and of zero length are skipped by the masks below
    Ap = np.concatenate([A0, np.full((pad, 3), 1e6)]); Bp_ = np.concatenate([B0, np.full((pad, 3), 1e6 + 1.0)])
    Wp = np.concatenate([W0, np.zeros(pad)])
    nch = (S + pad) // chunk
    SA = jnp.asarray(Ap.reshape(nch, chunk, 3)); SB = jnp.asarray(Bp_.reshape(nch, chunk, 3))
    SW = jnp.asarray(Wp.reshape(nch, chunk))
    SL = jnp.linalg.norm(SB - SA, axis=-1)
    absW = jnp.abs(SW)

    def field(X):
        """B (n,3), DB (n,3,3) [DB[:, i, j] = dB_i/dx_j] at points X (n,3)."""
        def body(carry, seg):
            Bc, Dc = carry
            a, b, w = seg
            Ri = X[:, None, :] - a[None]; Rf = X[:, None, :] - b[None]
            Lv = (b - a)[None]
            L2 = jnp.sum(Lv * Lv, -1)
            ri2 = jnp.sum(Ri * Ri, -1); rf2 = jnp.sum(Rf * Rf, -1)
            ri = jnp.sqrt(ri2); rf = jnp.sqrt(rf2)
            dot = jnp.sum(Ri * Rf, -1); Pp = ri * rf
            c = jnp.cross(Ri, Rf)
            c2 = jnp.sum(c * c, -1)
            den = jnp.where(dot >= 0, Pp + dot, c2 / jnp.maximum(Pp - dot, 1e-300))
            Ss = ri + rf
            f = w[None] * Ss / (Pp * den)
            f = jnp.where(w[None] != 0, f, 0.0)
            Bc = Bc + jnp.sum(c * f[..., None], 1)
            g = Ri / ri[..., None] + Rf / rf[..., None]
            c1 = (Ss * Ss + L2) / (2.0 * Ss * den)
            d = -f[..., None] * (g * c1[..., None] + Ri / ri2[..., None] + Rf / rf2[..., None])
            # D[i, j] = c_i d_j + f eps-term (Levi-Civita with L)
            Dseg = c[..., :, None] * d[..., None, :]
            Lx, Ly, Lz = Lv[..., 0], Lv[..., 1], Lv[..., 2]
            fL = jnp.stack([jnp.stack([0 * Lx, -Lz, Ly], -1), jnp.stack([Lz, 0 * Lx, -Lx], -1),
                            jnp.stack([-Ly, Lx, 0 * Lx], -1)], -2) * f[..., None, None]
            Dc = Dc + jnp.sum(Dseg + fL, 1)
            return (Bc, Dc), None
        n = X.shape[0]
        (Bv, Dv), _ = lax.scan(body, (jnp.zeros((n, 3)), jnp.zeros((n, 3, 3))), (SA, SB, SW))
        return Bv, Dv

    def crude(X, rad):
        """analytic bounds of |D^2 B| and |D^3 B| on the balls of radius rad around X, and the distance to
        the coils (one pass over the segment chunks)."""
        n = X.shape[0]

        def body(carry, seg):
            t2, t3, dmin, bad = carry
            a, b, w, Ls = seg
            ab = b - a
            t = jnp.clip(jnp.sum((X[:, None, :] - a[None]) * ab[None], -1) /
                         jnp.maximum(jnp.sum(ab * ab, -1), 1e-300)[None], 0.0, 1.0)
            dvec = X[:, None, :] - (a[None] + t[..., None] * ab[None])
            D = jnp.sqrt(jnp.sum(dvec * dvec, -1))
            msk = jnp.abs(w)[None] > 0
            dmin = jnp.minimum(dmin, jnp.min(jnp.where(msk, D, jnp.inf), -1))
            d = D - rad[:, None]
            dpos = jnp.where(d > 0, d, 1.0)
            d3_ = dpos ** 3; d4_ = d3_ * dpos
            aw = jnp.abs(w)[None]
            v2 = jnp.minimum(Ls[None] / d4_, 0.5 * math.pi / d3_) * aw * 6.0
            v3 = jnp.minimum(Ls[None] / (d4_ * dpos), (4.0 / 3.0) / d4_) * aw * 24.0
            t2 = t2 + jnp.sum(jnp.where(msk, v2, 0.0), -1); t3 = t3 + jnp.sum(jnp.where(msk, v3, 0.0), -1)
            bad = bad | jnp.any((d <= 0) & msk, -1)
            return (t2, t3, dmin, bad), None
        (t2, t3, dmin, bad), _ = lax.scan(body, (jnp.zeros(n), jnp.zeros(n), jnp.full(n, jnp.inf),
                                                 jnp.zeros(n, bool)), (SA, SB, SW, SL))
        return jnp.where(bad, jnp.inf, t2), jnp.where(bad, jnp.inf, t3), dmin

    def d2field(X):
        """exact D^2 B (forward-mode AD of the closed form): D2[:, i, j, k] = d_k dB_i/dx_j, (n,3,3,3)."""
        cols = []
        for k in range(3):
            tk = jnp.zeros_like(X).at[:, k].set(1.0)
            _, (dB, dD) = jax.jvp(field, (X,), (tk,))
            cols.append(dD)
        return jnp.stack(cols, -1)

    def spec_ub(M):            # (n, 3, 3) -> (n,)
        G = jnp.einsum("nki,nkj->nij", M, M)
        gm = jnp.max(jnp.sum(jnp.abs(G), -1), -1)
        fro = jnp.sum(M * M, axis=(1, 2))
        return jnp.sqrt(jnp.minimum(gm, fro)) * (1.0 + 1e-14)

    def ugrad(X, Bv, Dv):
        u = X[:, 0] * Bv[:, 1] - X[:, 1] * Bv[:, 0]
        g = X[:, 0:1] * Dv[:, 1, :] - X[:, 1:2] * Dv[:, 0, :]
        g = g.at[:, 0].add(Bv[:, 1]).at[:, 1].add(-Bv[:, 0])
        return u, g

    def cell_test(X, Bv, Dv, D2, M, E4, sag, Eb, need_neg, need_pos, rel_margin):
        u, g = ugrad(X, Bv, Dv)
        gm = jnp.einsum("ni,nij->nj", g, M)
        lin = jnp.einsum("nj,nqj->nq", gm, E4)
        lmax = jnp.max(lin, -1); lmin = jnp.min(lin, -1)
        W = jnp.einsum("nij,nqj->nqi", M, E4)
        rM = jnp.max(jnp.sqrt(jnp.sum(W * W, -1)), -1)
        ngm = jnp.sqrt(jnp.sum(gm * gm, -1)); ng = jnp.sqrt(jnp.sum(g * g, -1))
        r = rM + spec_ub(M) * sag + Eb
        lin_hi = u + lmax + ngm * sag + ng * Eb
        lin_lo = u + lmin - ngm * sag - ng * Eb
        marg = rel_margin * (jnp.abs(u) + ng * r) + 1e-300
        d2, d3, _ = crude(X, r)
        ndb = jnp.sqrt(jnp.sum(Dv * Dv, axis=(1, 2)))
        R = jnp.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        H2 = (R + r) * d2 + 2.0 * (ndb + r * d2)
        neg_ok = need_neg & (lin_hi + 0.5 * H2 * r * r < -marg)
        pos_ok = need_pos & (lin_lo - 0.5 * H2 * r * r > marg)
        # exact Hessian of u at the point (D^2 u[w,w] = D^2 B[w,w].Jx + 2 DB[w].Jw) + third-order remainder
        H = X[:, 0, None, None] * D2[:, 1] - X[:, 1, None, None] * D2[:, 0]
        e0 = jnp.array([1.0, 0.0, 0.0]); e1 = jnp.array([0.0, 1.0, 0.0])
        Q1 = e0[None, :, None] * Dv[:, 1, None, :] - e1[None, :, None] * Dv[:, 0, None, :]
        H = H + Q1 + jnp.swapaxes(Q1, 1, 2)
        H = 0.5 * (H + jnp.swapaxes(H, 1, 2))
        dg = jnp.diagonal(H, axis1=1, axis2=2)
        off = jnp.sum(jnp.abs(H), -1) - jnp.abs(dg)
        hmax = jnp.max(dg + off, -1) * (1 + 1e-12); hmin = jnp.min(dg - off, -1) * (1 + 1e-12)
        T = (R + r) * d3 + 3.0 * d2
        neg_ok = neg_ok | (need_neg & (lin_hi + 0.5 * jnp.maximum(hmax, 0.0) * r * r + T * r ** 3 / 6.0 < -marg))
        pos_ok = pos_ok | (need_pos & (lin_lo + 0.5 * jnp.minimum(hmin, 0.0) * r * r - T * r ** 3 / 6.0 > marg))
        return neg_ok, pos_ok, r

    def rhs(Y, sig, Bv, Dv):
        dx = sig[:, None] * Bv
        dM = sig[:, None, None] * jnp.einsum("nik,nkj->nij", Dv, Y[:, 3:].reshape(-1, 3, 3))
        return jnp.concatenate([dx, dM.reshape(-1, 9)], 1)

    def stage_field(Y):
        Bv, Dv = field(Y[:, :3])
        return Bv, Dv

    @jax.jit
    def run(C, E4s, sags, need, tol, hmax, smax, nmax, Rfac, rel_margin):
        n = C.shape[0]
        rho = jnp.max(jnp.sqrt(jnp.sum(E4s * E4s, -1)), -1) + sags
        I9 = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))
        B0, D0 = field(C)
        nneg = need[:, 0] != 0; npos = need[:, 1] != 0
        ng0, pg0, r0 = cell_test(C, B0, D0, d2field(C), I9, E4s, sags, jnp.zeros(n), nneg, npos, rel_margin)
        negok = ng0 | ~nneg; posok = pg0 | ~npos
        Y0 = jnp.concatenate([C, I9.reshape(n, 9)], 1)
        sig0 = jnp.ones(n)
        _, _, dm0 = crude(C, jnp.zeros(n))
        nb0 = jnp.sqrt(jnp.sum(B0 * B0, -1)); nd0 = jnp.sqrt(jnp.sum(D0 * D0, axis=(1, 2)))
        hs0 = jnp.minimum(jnp.minimum(hmax, 0.06 * dm0), jnp.minimum(0.5 * rho + 1e-3 * hmax, 0.06 * nb0 / jnp.maximum(nd0, 1e-300)))
        st = dict(Y=Y0, K1=rhs(Y0, sig0, B0, D0), h=hs0 / nb0, hs=hs0, sig=sig0, dirn=jnp.where(negok & posok, 2, 0),
                  s=jnp.zeros(n), N=jnp.zeros(n), ex=jnp.zeros(n), eM=jnp.zeros(n), r=r0, dphi=jnp.zeros(n),
                  negok=negok, posok=posok, sneg=jnp.where(negok, 0.0, -1.0), spos=jnp.where(posok, 0.0, -1.0),
                  code=jnp.zeros(n, jnp.int32), steps=jnp.zeros(n, jnp.int32), it=jnp.int32(0),
                  Mbest=I9.reshape(n, 9), sbest=-jnp.ones(n), sfwd=jnp.zeros(n), sbwd=jnp.zeros(n),
                  pfwd=jnp.zeros(n), pbwd=jnp.zeros(n))

        def restart(st, m):
            """direction switch for cells m: back to the centre, along -B."""
            Y = jnp.where(m[:, None], Y0, st["Y"])
            sig = jnp.where(m, -1.0, st["sig"])
            K1 = jnp.where(m[:, None], rhs(Y0, -jnp.ones(n), B0, D0), st["K1"])
            upd = dict(Y=Y, sig=sig, K1=K1, h=jnp.where(m, hs0 / nb0, st["h"]), hs=jnp.where(m, hs0, st["hs"]),
                       s=jnp.where(m, 0.0, st["s"]), N=jnp.where(m, 0.0, st["N"]), ex=jnp.where(m, 0.0, st["ex"]),
                       eM=jnp.where(m, 0.0, st["eM"]), r=jnp.where(m, r0, st["r"]), dphi=jnp.where(m, 0.0, st["dphi"]),
                       dirn=jnp.where(m, 1, st["dirn"]))
            return {**st, **upd}

        def cond(st):
            return (st["it"] < nmax) & jnp.any(st["dirn"] < 2)

        def body(st):
            act = st["dirn"] < 2
            Y = st["Y"]; h = st["h"]; sig = st["sig"]; K1 = st["K1"]
            hh = h[:, None]
            T = Y + hh * (0.2 * K1); Bv, Dv = stage_field(T); K2 = rhs(T, sig, Bv, Dv)
            T = Y + hh * (3.0 / 40 * K1 + 9.0 / 40 * K2); Bv, Dv = stage_field(T); K3 = rhs(T, sig, Bv, Dv)
            T = Y + hh * (44.0 / 45 * K1 - 56.0 / 15 * K2 + 32.0 / 9 * K3); Bv, Dv = stage_field(T); K4 = rhs(T, sig, Bv, Dv)
            T = Y + hh * (19372.0 / 6561 * K1 - 25360.0 / 2187 * K2 + 64448.0 / 6561 * K3 - 212.0 / 729 * K4)
            Bv, Dv = stage_field(T); K5 = rhs(T, sig, Bv, Dv)
            T = Y + hh * (9017.0 / 3168 * K1 - 355.0 / 33 * K2 + 46732.0 / 5247 * K3 + 49.0 / 176 * K4 - 5103.0 / 18656 * K5)
            Bv, Dv = stage_field(T); K6 = rhs(T, sig, Bv, Dv)
            Yn = Y + hh * (35.0 / 384 * K1 + 500.0 / 1113 * K3 + 125.0 / 192 * K4 - 2187.0 / 6784 * K5 + 11.0 / 84 * K6)
            B1, D1 = stage_field(Yn); K7 = rhs(Yn, sig, B1, D1)
            Er = jnp.abs(hh * (71.0 / 57600 * K1 - 71.0 / 16695 * K3 + 71.0 / 1920 * K4 - 17253.0 / 339200 * K5
                              + 22.0 / 525 * K6 - 1.0 / 40 * K7))
            errx = jnp.max(Er[:, :3], -1) * math.sqrt(3.0); errM = jnp.max(Er[:, 3:], -1) * 3.0
            err = jnp.maximum(errx, errM * rho)
            step = jnp.sqrt(jnp.sum((Yn[:, :3] - Y[:, :3]) ** 2, -1))
            nb1 = jnp.sqrt(jnp.sum(B1 * B1, -1)); nd1 = jnp.sqrt(jnp.sum(D1 * D1, axis=(1, 2)))
            finite = jnp.all(jnp.isfinite(Yn), -1) & (nb1 > 0)
            reject = act & finite & ((err > tol) | (step > 1.5 * st["hs"])) & (h * nb1 > 1e-12 * hmax)
            # one-step propagator and tube bootstrap (three attempts)
            Mo = Y[:, 3:].reshape(n, 3, 3); Mn = Yn[:, 3:].reshape(n, 3, 3)
            Ainv = jnp.linalg.inv(Mo)
            a = spec_ub(jnp.einsum("nik,nkj->nij", Mn, Ainv)) * (1.0 + 1e-9)
            Mn1 = spec_ub(Mn); My = spec_ub(Mo)
            Rt = Rfac * st["r"]
            ok = jnp.zeros(n, bool)
            res = dict(N=st["N"], ex=st["ex"], eM=st["eM"], r=st["r"], dmn=jnp.zeros(n))
            D2n = d2field(Yn[:, :3])
            nD2 = jnp.sqrt(jnp.sum(D2n * D2n, axis=(1, 2, 3)))
            for _att in range(3):
                rad = Rt + 1.2 * step + st["ex"] + 10.0 * errx
                c2, c3, dmn = crude(Yn[:, :3], rad)
                d2 = jnp.minimum(c2, nD2 + rad * c3)
                Lb = nd1 + rad * d2
                eh = jnp.exp(h * Lb)
                grow = eh * (st["r"] + 0.5 * h * d2 * Rt * Rt) > Rt
                Nn = a * st["N"] + h * eh * 0.5 * d2 * Rt * Rt
                exn = a * st["ex"] + 10.0 * errx
                eMn = a * st["eM"] + 10.0 * errM + h * eh * d2 * st["ex"] * My
                En = Nn + exn + eMn * rho
                Wq = jnp.einsum("nij,nqj->nqi", Mn, E4s)
                rM = jnp.max(jnp.sqrt(jnp.sum(Wq * Wq, -1)), -1)
                rn = rM + Mn1 * sags + En
                good = jnp.isfinite(d2) & ~grow & (rn <= Rt) & ~ok
                res = {k: jnp.where(good, v, res[k]) for k, v in dict(N=Nn, ex=exn, eM=eMn, r=rn, dmn=dmn).items()}
                ok = ok | good
                Rt = jnp.where(ok, Rt, jnp.where(grow, 2.0 * eh * (st["r"] + 0.5 * h * d2 * Rt * Rt), 2.0 * rn))
            accept = act & finite & ~reject & ok
            fail = act & (~finite | (~reject & ~ok))
            E_last = res["N"] + res["ex"] + res["eM"] * rho
            ngt, pgt, _ = cell_test(Yn[:, :3], B1, D1, D2n, Mn, E4s, sags, E_last, ~st["negok"], ~st["posok"], rel_margin)
            snew = st["s"] + step
            p0 = jnp.arctan2(Y[:, 1], Y[:, 0]); p1 = jnp.arctan2(Yn[:, 1], Yn[:, 0])
            dp = jnp.mod(p1 - p0 + jnp.pi, 2 * jnp.pi) - jnp.pi
            negok = st["negok"] | (accept & ngt); posok = st["posok"] | (accept & pgt)
            sneg = jnp.where(accept & ngt & ~st["negok"], sig * snew, st["sneg"])
            spos = jnp.where(accept & pgt & ~st["posok"], sig * snew, st["spos"])
            Ynew = jnp.where(accept[:, None], Yn, Y); K1n = jnp.where(accept[:, None], K7, K1)
            s_ = jnp.where(accept, snew, st["s"])
            fac = 0.9 * (tol / jnp.maximum(err, 1e-300)) ** 0.2
            hs_new = jnp.minimum(jnp.minimum(hmax, 0.06 * res["dmn"]), 0.06 * nb1 / jnp.maximum(nd1, 1e-300))
            h_acc = jnp.minimum(hs_new / jnp.maximum(nb1, 1e-300), h * jnp.minimum(4.0, jnp.maximum(0.2, fac)))
            h_rej = h * jnp.maximum(0.2, jnp.minimum(0.9 * (tol / jnp.maximum(err, 1e-300)) ** 0.2,
                                                     st["hs"] / jnp.maximum(step, 1e-300)))
            hnew = jnp.where(accept, h_acc, jnp.where(reject, h_rej, h))
            hsn = jnp.where(accept, hs_new, st["hs"])
            done_both = negok & posok
            end_dir = act & (fail | (accept & (s_ > smax)) | done_both)
            code = jnp.where(fail & ~done_both & (st["code"] == 0), jnp.where(finite, 1, 3), st["code"])
            code = jnp.where(accept & (s_ > smax) & ~done_both & (st["code"] == 0), 2, code)
            better = end_dir & (s_ > st["sbest"])
            Mbest = jnp.where(better[:, None], Ynew[:, 3:], st["Mbest"])
            sbest = jnp.where(better, s_, st["sbest"])
            sfwd = jnp.where(end_dir & (st["dirn"] == 0), s_, st["sfwd"])
            sbwd = jnp.where(end_dir & (st["dirn"] == 1), s_, st["sbwd"])
            dph_new = jnp.where(accept, st["dphi"] + dp, st["dphi"])
            pfwd = jnp.where(end_dir & (st["dirn"] == 0), dph_new, st["pfwd"])
            pbwd = jnp.where(end_dir & (st["dirn"] == 1), dph_new, st["pbwd"])
            nst = {**st, "Y": Ynew, "K1": K1n, "h": hnew, "hs": hsn, "s": s_,
                   "N": jnp.where(accept, res["N"], st["N"]), "ex": jnp.where(accept, res["ex"], st["ex"]),
                   "eM": jnp.where(accept, res["eM"], st["eM"]), "r": jnp.where(accept, res["r"], st["r"]),
                   "dphi": jnp.where(accept, st["dphi"] + dp, st["dphi"]), "negok": negok, "posok": posok,
                   "sneg": sneg, "spos": spos, "code": code, "steps": st["steps"] + act.astype(jnp.int32),
                   "it": st["it"] + 1, "Mbest": Mbest, "sbest": sbest, "sfwd": sfwd, "sbwd": sbwd,
                   "pfwd": pfwd, "pbwd": pbwd,
                   "dirn": jnp.where(end_dir & (done_both | (st["dirn"] == 1)), 2, st["dirn"])}
            sw = end_dir & ~done_both & (st["dirn"] == 0)
            return restart(nst, sw)

        st = lax.while_loop(cond, body, st)
        code = jnp.where(st["dirn"] < 2, 4, st["code"])
        both = st["negok"] & st["posok"]
        out = jnp.zeros((n, 21))
        out = out.at[:, 0].set(st["negok"].astype(jnp.float64)).at[:, 1].set(st["posok"].astype(jnp.float64))
        out = out.at[:, 2].set(st["sneg"]).at[:, 3].set(st["spos"]).at[:, 4].set(st["steps"].astype(jnp.float64))
        out = out.at[:, 5].set(jnp.where(both, 0, jnp.maximum(code, 1)).astype(jnp.float64))
        out = out.at[:, 6].set(st["r"]).at[:, 9].set(st["sfwd"]).at[:, 10].set(st["sbwd"])
        out = out.at[:, 7].set(st["pfwd"]).at[:, 8].set(st["pbwd"])
        out = out.at[:, 12:21].set(st["Mbest"])
        return out

    def run_np(C, E4s, sags, need, tol, hmax, smax, nmax, Rfac, rel_margin, block=4096):
        """Pads the batch to a multiple of block (padding cells need nothing: done at once), so that the
        jitted function is compiled for few shapes."""
        n = len(C)
        m = max(block, int(math.ceil(n / block)) * block)
        if m > n:
            C = np.concatenate([C, np.repeat(C[:1], m - n, 0)])
            E4s = np.concatenate([E4s, np.repeat(E4s[:1], m - n, 0)])
            sags = np.concatenate([sags, np.repeat(sags[:1], m - n, 0)])
            need = np.concatenate([need, np.zeros((m - n, 2), need.dtype)])
        o = run(jnp.asarray(C), jnp.asarray(E4s), jnp.asarray(sags), jnp.asarray(need), float(tol), float(hmax),
                float(smax), int(nmax), float(Rfac), float(rel_margin))
        return np.asarray(o)[:n]

    def check(points):
        """max relative deviation of the JAX field (B, DB) from torux.reference.fast64 at sample points."""
        Bv, Dv = field(jnp.asarray(points))
        ref, _ = fast64.eval_points(fast64.Field(source).packed, np.ascontiguousarray(points), True)
        eB = np.max(np.linalg.norm(np.asarray(Bv) - ref[:, :3], axis=1) / np.linalg.norm(ref[:, :3], axis=1))
        eD = np.max(np.linalg.norm(np.asarray(Dv).reshape(-1, 9) - ref[:, 6:15], axis=1) /
                    np.linalg.norm(ref[:, 6:15], axis=1))
        return float(eB), float(eD)
    run_np.check = check
    return run_np


def near_mirror_level(source, n=24):
    """Diagnostic: for each coil (polylines), the median of |B_phi| / |B| on the meridional half plane through
    the middle of its toroidal extent, compared with the half planes 0.02 rad aside.  Ratios << 1 mark
    near-antisymmetric planes (B_phi ~ 0 on them: lines near them have u = R B_phi of size ~ the symmetry
    breaking, which LOST certificates must resolve)."""
    loops, has_tor = coil_loops(source)
    P = lean_pack(source)
    out = []
    for j, lp in enumerate(loops):
        if not lp.get("exact") or lp["I"] == 0.0:
            continue
        b = coil_box(lp["pts"])
        if b[0] <= 0:
            continue
        vals = []
        for ph in (b[4] + 0.5 * b[5], b[4] + 0.5 * b[5] + 0.02, b[4] + 0.5 * b[5] - 0.02):
            Rs = np.linspace(b[0] - 0.1 * (b[1] - b[0] + 0.1), b[1] + 0.1 * (b[1] - b[0] + 0.1), n)
            Zs = np.linspace(b[2] - 0.1 * (b[3] - b[2] + 0.1), b[3] + 0.1 * (b[3] - b[2] + 0.1), n)
            RR, ZZ = np.meshgrid(Rs, Zs)
            X = np.stack([RR.ravel() * math.cos(ph), RR.ravel() * math.sin(ph), ZZ.ravel()], -1)
            f, _ = eval_points(P, np.ascontiguousarray(X), False)
            Bp = (-X[:, 1] * f[:, 0] + X[:, 0] * f[:, 1]) / np.hypot(X[:, 0], X[:, 1])
            vals.append(float(np.median(np.abs(Bp) / np.maximum(np.linalg.norm(f[:, :3], axis=1), 1e-300))))
        out.append(dict(coil=j, mid=vals[0], aside=0.5 * (vals[1] + vals[2]), ratio=vals[0] / max(1e-300, 0.5 * (vals[1] + vals[2]))))
    return out
