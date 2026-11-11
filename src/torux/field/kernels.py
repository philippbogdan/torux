"""Pairwise field kernels (JAX, float64).

Each kernel maps targets x (T, 3) and a batch of source elements to B (T, 3), the vector
potential A (T, 3) and where cheap the gradient dB[a, b] = dB_a/dx_b (T, 3, 3), summed over
the batch.  The constant mu0 / (4 pi) is folded into the source weights by the engine.
Sums over source elements are XLA reduce fusions (tree sums on GPU): rounding grows like
log S for S elements.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def cross(a, b):
    return jnp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                      a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def skew(v):
    """[v]_x as (..., 3, 3): [v]_x e = v x e."""
    z = jnp.zeros_like(v[..., 0])
    return jnp.stack([jnp.stack([z, -v[..., 2], v[..., 1]], -1),
                      jnp.stack([v[..., 2], z, -v[..., 0]], -1),
                      jnp.stack([-v[..., 1], v[..., 0], z], -1)], axis=-2)


# ----------------------------------------------------------------------------- filaments
def filament_nodes(x, r, w, wh, need_grad=True, need_A=True):
    """Trapezoid-rule Biot-Savart sum over quadrature nodes of closed filaments.

    x (T,3) targets; r (S,3) node positions; w (S,3) weighted tangents
    (mu0 I / 4 pi * r'(t_j) / N); wh (S,) half-rule factors (2 on the even nodes of each
    coil, 0 on the odd ones): Bh = sum wh * (w x d)/|d|^3 is the N/2-point rule, whose
    difference from B estimates the error of the N/2 rule; the N-point rule's error is
    about its square (geometric convergence).

    Returns B (T,3), A (T,3), G (T,3,3), Bh (T,3).
    """
    d = x[:, None, :] - r[None, :, :]                      # (T,S,3)
    r2 = jnp.sum(d * d, axis=-1)
    ir = jax.lax.rsqrt(r2)
    ir3 = ir * ir * ir
    c = cross(w[None, :, :], d)                             # w x d
    cB = c * ir3[..., None]
    B = jnp.sum(cB, axis=1)
    Bh = jnp.sum(cB * wh[None, :, None], axis=1)
    A = jnp.sum(w[None, :, :] * ir[..., None], axis=1) if need_A else jnp.zeros_like(B)
    if need_grad:
        ir5 = 3.0 * ir3 * ir * ir
        # dB_a/dx_b = sum_j [ (w_j x e_b)_a / r^3 - 3 (w_j x d)_a d_b / r^5 ]
        S3 = jnp.sum(w[None, :, :] * ir3[..., None], axis=1)
        # sum_j (w_j x d_j)_a (d_j)_b 3/r^5 as a batched (3 x S)(S x 3) product: no (T, S, 3, 3) temporary
        P = jnp.einsum("tsa,tsb->tab", c * ir5[..., None], d)
        G = skew(S3) - P
    else:
        G = jnp.zeros(B.shape + (3,), B.dtype)
    return B, A, G, Bh


# ----------------------------------------------------------------------------- segments
def segments(x, a, b, cw):
    """Exact field and vector potential of straight segments a -> b, weight cw = mu0 I / 4 pi.

    With R_i = x - a, R_f = x - b, r_i, r_f their lengths, u_i, u_f the unit vectors and
    q = (r_i + r_f)^2 - L^2 = r_i r_f |u_i + u_f|^2 (no cancellation of r_i r_f + R_i.R_f):
        B = cw * 2 (r_i + r_f) / (r_i r_f q) * (dl x R_i)
        A = cw * dl / L * ln((r_i + r_f + L)^2 / q)  (= 2 atanh(L / (r_i + r_f)) for small L)
    """
    Ri = x[:, None, :] - a[None, :, :]
    Rf = x[:, None, :] - b[None, :, :]
    ri = jnp.sqrt(jnp.sum(Ri * Ri, -1))
    rf = jnp.sqrt(jnp.sum(Rf * Rf, -1))
    dl = (b - a)[None, :, :]
    L = jnp.sqrt(jnp.sum(dl * dl, -1))
    s = Ri / ri[..., None] + Rf / rf[..., None]
    q = ri * rf * jnp.sum(s * s, -1)
    fac = cw[None, :] * 2.0 * (ri + rf) / (ri * rf * q)
    B = jnp.sum(cross(dl, Ri) * fac[..., None], axis=1)
    # ln((S + L)^2 / q) = 2 atanh(L / S), S = r_i + r_f: the atanh form for L / S < 1/2 (the log of
    # a number near 1 would lose digits), the log form near the segment
    y = L / (ri + rf)
    lg = jnp.where(y < 0.5, 2.0 * jnp.arctanh(jnp.minimum(y, 0.5)), jnp.log((ri + rf + L) ** 2 / q))
    A = jnp.sum(dl * (cw[None, :] * lg / L)[..., None], axis=1)
    return B, A


# ----------------------------------------------------------------------------- dipoles
def dipoles(x, p, m):
    """Point dipoles at p with moments m (already times mu0 / 4 pi):
    B = (3 (m.d) d / r^2 - m) / r^3,  A = m x d / r^3."""
    d = x[:, None, :] - p[None, :, :]
    r2 = jnp.sum(d * d, -1)
    ir = jax.lax.rsqrt(r2)
    ir2 = ir * ir
    ir3 = ir2 * ir
    md = jnp.sum(m[None] * d, -1)
    B = jnp.sum(((3.0 * md * ir2)[..., None] * d - m[None]) * ir3[..., None], axis=1)
    A = jnp.sum(cross(m[None], d) * ir3[..., None], axis=1)
    return B, A


# ----------------------------------------------------------------------------- rings
def agm_KE(k2, kp2, n_iter=10):
    """Complete elliptic integrals of the first and second kind for parameter m = k^2.

    Returns K, E, Dk = (K - E)/k^2 and Sg = sum_{n>=1} 2^(n-1) c_n^2 / k^2 (so that
    Dk = K (1/2 + Sg) and (1 - k^2/2) K - E = k^2 K Sg), by the arithmetic-geometric mean
    a0 = 1, b0 = k', c0^2 = k^2, with c_{n+1} = c_n^2 / (4 a_{n+1}): every term is formed
    without subtractive cancellation, for k -> 0 as for k -> 1.  Both k2 and kp2 = 1 - k2
    must be supplied accurately by the caller.  Eight AGM steps reach rounding level
    for k' >= 1e-150 (convergence is quadratic once a_n ~ b_n).
    """
    a = jnp.ones_like(kp2)
    b = jnp.sqrt(kp2)
    c2 = k2                              # c_n^2, starting at c_0^2 = k^2
    Sg = jnp.zeros_like(k2)
    pw = 1.0                             # 2^(n-1) for the term n = i + 1
    for i in range(n_iter):
        an = 0.5 * (a + b)               # a_{n+1}
        bn = jnp.sqrt(a * b)
        q = c2 / (16.0 * an * an)
        # gamma_{n+1} = c_{n+1}^2 / k^2 = gamma_n * c_n^2 / (16 a_{n+1}^2); gamma_1 = k^2/(16 a_1^2)
        gam = q if i == 0 else gam * q
        Sg = Sg + pw * gam
        pw = 2.0 * pw
        a, b, c2 = an, bn, c2 * q        # c_{n+1}^2 = c_n^4 / (16 a_{n+1}^2)
    K = jnp.pi / (2.0 * a)
    Dk = K * (0.5 + Sg)
    E = K - k2 * Dk
    return K, E, Dk, Sg


def rings(x, R0, Z0, cw):
    """Circular filament loops coaxial with z (radius a = R0, height Z0), weight cw = mu0 I/4pi.

    With D2 = (a - R)^2 + z^2, S2 = (a + R)^2 + z^2, k^2 = 4 a R / S2, k'^2 = D2 / S2:
        B_R   = (mu0 I/pi) a z / S * (E / D2 - 2 Dk / S2)                       (k^2 >= 1/2)
              = (mu0 I/pi) a z / S^3 * K [k^2/2 - (2 - k^2) Sg] / k'^2           (k^2 <  1/2)
        B_Z   = (mu0 I/2pi) / S * (k^2 Dk + 2 a (a - R) E / D2)
        A_phi = (mu0 I/pi) 2 a / S * K Sg
    (the textbook forms rewritten so that nothing divides by R and no term cancels near the
    axis or far away).  Returns B, A (T,3) Cartesian.
    """
    X, Y, Zt = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    R = jnp.sqrt(X * X + Y * Y)
    a = R0[None, :]
    z = Zt - Z0[None, :]
    D2 = (a - R) ** 2 + z * z
    S2 = (a + R) ** 2 + z * z
    S = jnp.sqrt(S2)
    k2 = 4.0 * a * R / S2
    kp2 = D2 / S2
    K, E, Dk, Sg = agm_KE(k2, kp2)
    mu_pi = 4.0 * cw[None, :]
    BR_near = mu_pi * a * z / S * (E / D2 - 2.0 * Dk / S2)
    BR_far = mu_pi * a * z / (S * S2) * K * (0.5 * k2 - (2.0 - k2) * Sg) / kp2
    BR = jnp.where(k2 >= 0.5, BR_near, BR_far)
    BZ = 0.5 * mu_pi / S * (k2 * Dk + E * 2.0 * a * (a - R) / D2)
    Aphi = mu_pi * 2.0 * a / S * K * Sg
    BRs = jnp.sum(BR, axis=1, keepdims=True)
    BZs = jnp.sum(BZ, axis=1, keepdims=True)
    Ap = jnp.sum(Aphi, axis=1, keepdims=True)
    on_axis = R == 0
    Rs = jnp.where(on_axis, 1.0, R)
    c = jnp.where(on_axis, 1.0, X / Rs)
    s = jnp.where(on_axis, 0.0, Y / Rs)
    B = jnp.concatenate([BRs * c, BRs * s, BZs], axis=1)
    A = jnp.concatenate([-Ap * s, Ap * c, jnp.zeros_like(Ap)], axis=1)
    return B, A
