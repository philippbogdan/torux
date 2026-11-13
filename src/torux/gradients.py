"""Gradients of flux-surface quantities with respect to the source parameters (adjoint method).

At a solution u* of the torus system F(u, p) = 0 (Galerkin residuals and the flux equation;
p = every Fourier coefficient and current of the source's coils, and the requested flux),
an objective f(u) has total derivative

    df/dp = -lambda^T dF/dp,     J^T lambda = df/du,     J = dF/du (the Newton Jacobian).

dF/dp enters only through the field (B, grad B, A) on the collocation grid, which JAX
differentiates through the engine's node construction (torux.field.engine.fourier_nodes) back to
the coil coefficients and currents: lambda^T dF/dp is ONE vector-Jacobian product of the
system with respect to the coil data.  The flux derivative is -lambda_flux * d(flux eq)/dflux.

Parameters: every part's (Fourier coefficients and currents; polyline points and currents; segment
end points and currents; dipole positions and moments; ring radii, heights and currents; the toroidal
field's R B) and the requested flux; result["parts"] lists one gradient dict per part.  Objectives
provided: iota of the surface, and any user function of (rc, zs, iota).  Checked against central
finite differences in the tests.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
import jax
import jax.numpy as jnp

from .field.engine import FieldEngine, fourier_nodes, MU0_4PI, _pad_rows, _bucket, CHUNK, _FAR
from .torus import TorusProblem, TorusState, _system_body, _CONSTS

jax.config.update("jax_enable_x64", True)


class _DiffEngine(FieldEngine):
    """A FieldEngine whose element arrays are built inside JAX from the source's parameters (a list with
    one dict per part, torux.gradients.source_params), so that the system is differentiable with respect
    to them.  Node counts per Fourier coil and the bucket padding are those of the template engine."""

    def __init__(self, template: FieldEngine, sparams, layout):
        self.kinds = template.kinds
        self.target_chunk = template.target_chunk
        self.analytic = template.analytic
        self.cpu_kernels = False
        tp = template.params
        far = jnp.asarray(np.array([_FAR, 0.0, 0.0]))
        rs, ws, whs, sa, sb, sw, dp, dm, rR, rZ, rw = ([] for _ in range(11))
        RB = jnp.asarray(0.0)
        Ns = list(template.info.N_per_coil)
        k = 0
        for (kind, meta), prm in zip(layout, sparams):
            if kind == "fourier":
                for i in meta["coils"]:
                    r, w, wh = fourier_nodes(prm["coef"][i:i + 1], prm["current"][i:i + 1], int(Ns[k]))
                    k += 1
                    rs.append(r); ws.append(w); whs.append(wh)
            elif kind == "polyline":
                pts = prm["points"]
                sa.append(pts[meta["ia"]]); sb.append(pts[meta["ib"]])
                sw.append(MU0_4PI * jnp.repeat(prm["current"], meta["counts"], total_repeat_length=len(meta["ia"])))
            elif kind == "segments":
                sa.append(prm["a"]); sb.append(prm["b"]); sw.append(MU0_4PI * prm["current"])
            elif kind == "dipoles":
                dp.append(prm["pos"]); dm.append(MU0_4PI * prm["moment"])
            elif kind == "rings":
                rR.append(prm["R"]); rZ.append(prm["Z"]); rw.append(MU0_4PI * prm["current"])
            elif kind == "toroidal":
                RB = RB + prm["RB"]

        def cat(L, n, fill, width=None):
            if not L:
                return tp_like(n, fill, width)
            a = jnp.concatenate(L)
            pad = n - a.shape[0]
            if pad > 0:
                f = jnp.broadcast_to(jnp.asarray(fill, dtype=a.dtype), (pad,) + a.shape[1:])
                a = jnp.concatenate([a, f])
            return a

        def tp_like(n, fill, width):
            shape = (n,) if width is None else (n, width)
            return jnp.broadcast_to(jnp.asarray(fill, dtype=jnp.float64), shape)
        nr, ns_, nd, nq = (tp["r"].shape[0], tp["sa"].shape[0], tp["dp"].shape[0], tp["rR"].shape[0])
        self.params = dict(
            r=cat(rs, nr, far, 3), w=cat(ws, nr, 0.0, 3), wh=cat(whs, nr, 0.0),
            sa=cat(sa, ns_, far, 3), sb=cat(sb, ns_, far + jnp.asarray([0.0, 0.0, 1.0]), 3), sw=cat(sw, ns_, 0.0),
            dp=cat(dp, nd, far, 3), dm=cat(dm, nd, 0.0, 3),
            rR=cat(rR, nq, 1.0), rZ=cat(rZ, nq, _FAR), rw=cat(rw, nq, 0.0),
            RB=RB)


def source_params(source):
    """(params, layout): params a list with one dict of arrays per part (the parameters gradients are
    taken with respect to: Fourier coefficients and currents, polyline points and currents, segment end
    points and currents, dipole positions and moments, ring radii, heights and currents, the toroidal
    field's R B), layout the static description the differentiable engine needs.  Polygon conductors
    are not supported (NotImplementedError)."""
    params, layout = [], []
    for p in source.parts:
        if p.kind == "fourier":
            coils = [i for i in range(p.n) if p.current[i] != 0.0]
            params.append(dict(coef=jnp.asarray(p.coef), current=jnp.asarray(p.current)))
            layout.append(("fourier", dict(coils=coils)))
        elif p.kind == "polyline":
            ia, ib, counts = [], [], []
            for i in range(p.n):
                lo, hi = int(p.start[i]), int(p.start[i + 1])
                idx = np.arange(lo, hi)
                ia.append(idx); ib.append(np.roll(idx, -1)); counts.append(hi - lo)
            params.append(dict(points=jnp.asarray(p.points), current=jnp.asarray(p.current)))
            layout.append(("polyline", dict(ia=np.concatenate(ia), ib=np.concatenate(ib), counts=np.array(counts))))
        elif p.kind == "segments":
            params.append(dict(a=jnp.asarray(p.a), b=jnp.asarray(p.b), current=jnp.asarray(p.current)))
            layout.append(("segments", {}))
        elif p.kind == "dipoles":
            params.append(dict(pos=jnp.asarray(p.pos), moment=jnp.asarray(p.moment)))
            layout.append(("dipoles", {}))
        elif p.kind == "rings":
            params.append(dict(R=jnp.asarray(p.R), Z=jnp.asarray(p.Z), current=jnp.asarray(p.current)))
            layout.append(("rings", {}))
        elif p.kind == "toroidal":
            params.append(dict(RB=jnp.asarray(float(p.RB))))
            layout.append(("toroidal", {}))
        else:
            raise NotImplementedError(f"gradients with respect to {p.kind} parts")
    return params, layout


def _part_gradients(layout, grads):
    """Per-part gradient dicts (numpy, sign: d objective / d parameter)."""
    out = []
    for (kind, _), g in zip(layout, grads):
        out.append(dict(kind=kind, **{"d_" + k: -np.asarray(v) for k, v in g.items()}))
    return out


def fourier_part_data(source):
    """(coef, current) of the source's Fourier coils with nonzero current, in engine order."""
    coefs, curs = [], []
    for p in source.parts:
        if p.kind == "fourier":
            for i in range(p.n):
                if p.current[i] != 0.0:
                    coefs.append(p.coef[i]); curs.append(p.current[i])
    return np.array(coefs), np.array(curs)


def _grid_x(c, u, symmetric):
    from .torus import grid_points
    b = c.basis
    rc, zs, _ = b.unpack(u)
    R, Z, *_ = b.geometry(rc, zs)
    x, _ = grid_points(b, R, Z, symmetric)
    return x


def _residual_from_field(c, u, Bx, Ax, flux_target, flux_scale, symmetric):
    """The Newton system's residual F from Cartesian B and A at the grid points (torux.torus._system_post without
    the field gradients and the Jacobian)."""
    from .torus import field_from_points
    b = c.basis
    rc, zs, iota = b.unpack(u)
    R, Z, Rt, Zt, Rp, Zp = b.geometry(rc, zs)
    g = field_from_points(b, Bx, Ax, None, symmetric, need_grad=False)
    ib = 1.0 / g["Bp"]
    FRg = Rp + iota * Rt - R * g["BR"] * ib
    FZg = Zp + iota * Zt - R * g["BZ"] * ib
    flux = -2 * np.pi * jnp.mean(g["AR"] * Rt + g["AZ"] * Zt)
    FR = b.proj_s(FRg).ravel()[b.is_]
    FZ = b.proj_c(FZg).ravel()[b.ic]
    return jnp.concatenate([FR, FZ, jnp.atleast_1d((flux - flux_target) * flux_scale)])


def _residual_only(c, de, u, flux_target, flux_scale, symmetric):
    """The residual alone through an engine: B and A at the grid points, no field gradients -- what
    lambda^T dF/dp needs.  Through _system_body the VJP also traced the field gradients and the dense Jacobian it
    then discarded (QUASR 954, M, N = 48, 32: 31 of the adjoint's 36 s)."""
    x = _grid_x(c, u, symmetric)
    o = de.evaluate(x, grad=False, A=True)
    return _residual_from_field(c, u, o["B"], o["A"], flux_target, flux_scale, symmetric)


@njit(parallel=True, cache=True, fastmath=False)
def _filament_vjp_kernel(x, r, w, gB, gA, gr, gw):
    """Cotangents of trapezoid Fourier-filament nodes (torux.field.cpu.filament: B = sum w x d / |d|^3,
    A = sum w / |d|, d = x - r) for target cotangents gB, gA:
        gw_j = sum_t (d x gB_t) / |d|^3 + gA_t / |d|,
        gr_j = -sum_t [(gB_t x w_j) / |d|^3 - 3 (d . (gB_t x w_j)) d / |d|^5 - (gA_t . w_j) d / |d|^3],
    each node's sums over the targets compensated (Neumaier)."""
    S = r.shape[0]
    T = x.shape[0]
    for j in prange(S):
        r0 = r[j, 0]; r1 = r[j, 1]; r2 = r[j, 2]
        w0 = w[j, 0]; w1 = w[j, 1]; w2 = w[j, 2]
        acc = np.zeros(6)
        cmp = np.zeros(6)
        val = np.empty(6)
        for t in range(T):
            d0 = x[t, 0] - r0; d1 = x[t, 1] - r1; d2 = x[t, 2] - r2
            q = d0 * d0 + d1 * d1 + d2 * d2
            ir = 1.0 / math.sqrt(q)
            ir3 = ir * ir * ir
            b0 = gB[t, 0]; b1 = gB[t, 1]; b2 = gB[t, 2]
            a0 = gA[t, 0]; a1 = gA[t, 1]; a2 = gA[t, 2]
            # d x gB
            val[0] = (d1 * b2 - d2 * b1) * ir3 + a0 * ir
            val[1] = (d2 * b0 - d0 * b2) * ir3 + a1 * ir
            val[2] = (d0 * b1 - d1 * b0) * ir3 + a2 * ir
            # gB x w
            c0 = b1 * w2 - b2 * w1; c1 = b2 * w0 - b0 * w2; c2 = b0 * w1 - b1 * w0
            dc = d0 * c0 + d1 * c1 + d2 * c2
            aw = a0 * w0 + a1 * w1 + a2 * w2
            f5 = 3.0 * dc * ir3 * ir * ir
            val[3] = -(c0 * ir3 - f5 * d0 - aw * d0 * ir3)
            val[4] = -(c1 * ir3 - f5 * d1 - aw * d1 * ir3)
            val[5] = -(c2 * ir3 - f5 * d2 - aw * d2 * ir3)
            for k in range(6):
                s_ = acc[k]; v = val[k]
                u_ = s_ + v
                if abs(s_) >= abs(v):
                    cmp[k] += (s_ - u_) + v
                else:
                    cmp[k] += (v - u_) + s_
                acc[k] = u_
        gw[j, 0] = acc[0] + cmp[0]; gw[j, 1] = acc[1] + cmp[1]; gw[j, 2] = acc[2] + cmp[2]
        gr[j, 0] = acc[3] + cmp[3]; gr[j, 1] = acc[4] + cmp[4]; gr[j, 2] = acc[5] + cmp[5]


def _fourier_only(engine):
    src = engine.source
    return (all(p.kind == "fourier" for p in src.parts) and not getattr(engine, "analytic", None)
            and bool(getattr(engine, "cpu_kernels", False)))


def _vjp_source_fourier(engine, prob, u, flux, flux_scale, lam):
    """lambda^T dF/dp for sources of Fourier filaments only, on the host: the field at the grid points and the
    nodes' cotangents by numba (the analytic derivatives of the trapezoid Biot-Savart sums), the residual's
    post-processing and the nodes' construction from the coefficients (torux.field.engine.fourier_nodes) through
    JAX.  Same numbers as the differentiable-engine VJP (to rounding), without tracing Biot-Savart in JAX."""
    from .field import cpu as _cpu
    from .field.engine import fourier_nodes
    sp, layout = source_params(engine.source)
    c = prob.c
    uj = jnp.asarray(u)
    x = np.asarray(_grid_x(c, uj, prob.symmetric), dtype=np.float64)
    Ns = [int(v) for v in engine.info.N_per_coil]

    def nodes_fn(sp_):
        rs, ws, k = [], [], 0
        for (kind, meta), prm in zip(layout, sp_):
            for i in meta["coils"]:
                r_, w_, _ = fourier_nodes(prm["coef"][i:i + 1], prm["current"][i:i + 1], Ns[k])
                k += 1
                rs.append(r_); ws.append(w_)
        return jnp.concatenate(rs), jnp.concatenate(ws)
    (r_n, w_n), nodes_vjp = jax.vjp(nodes_fn, sp)
    r_np = np.ascontiguousarray(np.asarray(r_n), dtype=np.float64)
    w_np = np.ascontiguousarray(np.asarray(w_n), dtype=np.float64)
    o = _cpu.filament(x, r_np, w_np, np.zeros(len(r_np)), need_grad=False, need_A=True)
    Bx, Ax = jnp.asarray(o[:, 0:3]), jnp.asarray(o[:, 3:6])
    _, post_vjp = jax.vjp(lambda B_, A_, f_: _residual_from_field(c, uj, B_, A_, f_, flux_scale, prob.symmetric),
                          Bx, Ax, jnp.asarray(float(flux)))
    gB, gA, d_flux = post_vjp(jnp.asarray(lam))
    gr = np.empty_like(r_np)
    gw = np.empty_like(w_np)
    _filament_vjp_kernel(x, r_np, w_np, np.ascontiguousarray(np.asarray(gB)), np.ascontiguousarray(np.asarray(gA)),
                         gr, gw)
    d_sp = nodes_vjp((jnp.asarray(gr), jnp.asarray(gw)))[0]
    return sp, layout, d_sp, d_flux


def _vjp_source(engine, prob, u, flux, flux_scale, lam, fast=True):
    """lambda^T dF/dp for every source parameter and the flux (Fourier-only sources on the host by the analytic
    filament VJP, everything else through the differentiable engine)."""
    if fast and _fourier_only(engine):
        sp, layout, d_sp, d_flux = _vjp_source_fourier(engine, prob, u, flux, flux_scale, lam)
    else:
        sp, layout = source_params(engine.source)
        c = prob.c

        def Fp(sp_, flux_t):
            de = _DiffEngine(engine, sp_, layout)
            return _residual_only(c, de, jnp.asarray(u), flux_t, flux_scale, prob.symmetric)
        _, vjp = jax.vjp(Fp, sp, jnp.asarray(float(flux)))
        d_sp, d_flux = vjp(jnp.asarray(lam))
    parts = _part_gradients(layout, d_sp)
    out = dict(parts=parts, d_flux=-float(d_flux))
    # the Fourier-only summary (coils with nonzero current, engine order) kept for the API
    fc = [(p, l) for p, l in zip(parts, layout) if l[0] == "fourier"]
    if fc:
        out["d_coef"] = np.concatenate([p["d_coef"][l[1]["coils"]] for p, l in fc])
        out["d_current"] = np.concatenate([p["d_current"][l[1]["coils"]] for p, l in fc])
    return out


def adjoint_gradient(engine: FieldEngine, torus: TorusState, objective="iota"):
    """d objective / d (coil coefficients, currents, flux) at a converged torus (dense adjoint).

    Returns dict(value, d_coef (ncoil, 3, 2K+1), d_current (ncoil,), d_flux)."""
    prob = TorusProblem(engine, torus.nfp, torus.M, torus.N, phi0=torus.phi0)
    b = prob.basis
    u = b.pack(torus.rc, torus.zs, torus.iota)
    R00 = float(torus.rc[0, b.N])
    flux_scale = R00 / max(abs(torus.flux), 1e-300)
    F, J, flux, res = prob.system(u, torus.flux, flux_scale)
    if objective == "iota":
        f = float(u[-1])
        g = jnp.zeros_like(u).at[-1].set(1.0)
    else:
        f, g = jax.value_and_grad(lambda uu: objective(*b.unpack(uu)))(u)
    lam = jnp.linalg.solve(J.T, g)
    out = _vjp_source(engine, prob, u, torus.flux, flux_scale, lam)
    out.update(value=f, residual=float(jnp.max(jnp.abs(F))))
    return out


def adjoint_gradient_mf(engine: FieldEngine, torus: TorusState, objective="iota", gtol=1e-12, restart=80,
                        maxiter=6):
    """Matrix-free adjoint for any resolution: J^T lambda = df/du by GMRES on the transposed
    Galerkin Jacobian (jax.linear_transpose of the Newton-Krylov J v), preconditioned by the
    transposed reducibility preconditioner; then one VJP of the system with respect to the coils."""
    from .torus import KrylovTorus
    prob = TorusProblem(engine, torus.nfp, torus.M, torus.N, phi0=torus.phi0)
    b = prob.basis
    u = b.pack(torus.rc, torus.zs, torus.iota)
    R00 = float(torus.rc[0, b.N])
    flux_scale = R00 / max(abs(torus.flux), 1e-300)
    kt = KrylovTorus(prob)
    G = kt._grid(u)
    Jv = lambda v: kt._Jv(G, v, flux_scale)
    Pv = lambda r: kt._precond(G, r, flux_scale)
    JT = jax.linear_transpose(Jv, u)
    PT = jax.linear_transpose(Pv, u)
    if objective == "iota":
        f = float(u[-1])
        g = jnp.zeros_like(u).at[-1].set(1.0)
    else:
        f, g = jax.value_and_grad(lambda uu: objective(*b.unpack(uu)))(u)
    lam, info = jax.scipy.sparse.linalg.gmres(lambda v: JT(v)[0], g, M=lambda r: PT(r)[0], tol=gtol, atol=0.0,
                                              restart=restart, maxiter=maxiter, solve_method="batched")
    lin_res = float(jnp.max(jnp.abs(JT(lam)[0] - g)) / max(float(jnp.max(jnp.abs(g))), 1e-300))
    out = _vjp_source(engine, prob, u, torus.flux, flux_scale, lam)
    out.update(value=f, adjoint_residual=lin_res)
    return out


def _field_cotangents_general(eq, u, Bx, Ax, flux, flux_scale, mu):
    """gB, gA (T, 3) and the flux's: the cotangents of Cartesian B and A at the grid points of u, and of the flux, for
    mu on [G, the flux row] of the general equations (torux.torus_general.GeneralEquations.post)."""
    _, pull = jax.vjp(lambda B_, A_, f_: eq.post(u, B_, A_, f_, flux_scale), jnp.asarray(Bx), jnp.asarray(Ax),
                      jnp.asarray(float(flux)))
    gB, gA, d_flux = pull(jnp.asarray(mu))
    return (np.ascontiguousarray(np.asarray(gB), dtype=np.float64),
            np.ascontiguousarray(np.asarray(gA), dtype=np.float64), float(d_flux))


ENGINE_SLAB_PAIRS = 2_000_000    # target points times field elements held at once by a VJP through the engine


def _engine_vjp_slabs(engine, sp, layout, x, gB, gA, slab=None):
    """sum over the points x of gB . dB/dp + gA . dA/dp, p the source's parameters, through the differentiable engine
    a slab of points at a time: JAX keeps what the backward pass needs for every point against every element, so a
    slab's memory is that of the slab, not of the grid."""
    i_ = engine.info
    elements = max(1, int(i_.n_nodes) + int(i_.n_segments) + int(i_.n_dipoles) + int(i_.n_rings))
    slab = int(slab or max(64, ENGINE_SLAB_PAIRS // elements))
    total = None
    for i in range(0, len(x), slab):
        xs = jnp.asarray(x[i:i + slab])

        def fld(sp_):
            o = _DiffEngine(engine, sp_, layout).evaluate(xs, grad=False, A=True)
            return o["B"], o["A"]
        _, pull = jax.vjp(fld, sp)
        d = pull((jnp.asarray(gB[i:i + slab]), jnp.asarray(gA[i:i + slab])))[0]
        total = d if total is None else jax.tree_util.tree_map(jnp.add, total, d)
    return total


def _vjp_source_general(engine, eq, u, flux, flux_scale, lam, fast=True, slab=None):
    """lambda^T dF/dp and lambda^T dF/dflux (at fixed flux_scale) for the general equations at any surface,
    F = eq.system(u, flux, flux_scale)[0] (torux.torus_general.GeneralEquations), p every source parameter.  F holds
    the source through B and A at the grid points alone, so the product is taken at the field: the cotangents of B
    and A there first (a VJP of eq.post), then those of the parameters, by the analytic filament kernel for sources of
    Fourier coils alone (the memory of the grid and of the nodes, never of the two together), otherwise through the
    differentiable engine a slab of points at a time (_engine_vjp_slabs).  Returns what _vjp_source returns."""
    from .field import cpu as _cpu
    mu = eq.cotangent(lam)
    x = np.ascontiguousarray(eq.points_of(u), dtype=np.float64)
    sp, layout = source_params(engine.source)
    if fast and _fourier_only(engine):
        Ns = [int(v) for v in engine.info.N_per_coil]

        def nodes_fn(sp_):
            rs, ws, k = [], [], 0
            for (kind, meta), prm in zip(layout, sp_):
                for i in meta["coils"]:
                    r_, w_, _ = fourier_nodes(prm["coef"][i:i + 1], prm["current"][i:i + 1], Ns[k])
                    k += 1
                    rs.append(r_)
                    ws.append(w_)
            return jnp.concatenate(rs), jnp.concatenate(ws)
        (r_n, w_n), nodes_vjp = jax.vjp(nodes_fn, sp)
        r_np = np.ascontiguousarray(np.asarray(r_n), dtype=np.float64)
        w_np = np.ascontiguousarray(np.asarray(w_n), dtype=np.float64)
        o = _cpu.filament(x, r_np, w_np, np.zeros(len(r_np)), need_grad=False, need_A=True)
        gB, gA, d_flux = _field_cotangents_general(eq, u, o[:, 0:3], o[:, 3:6], flux, flux_scale, mu)
        gr, gw = np.empty_like(r_np), np.empty_like(w_np)
        _filament_vjp_kernel(x, r_np, w_np, gB, gA, gr, gw)
        d_sp = nodes_vjp((jnp.asarray(gr), jnp.asarray(gw)))[0]
    else:
        o = engine.evaluate(x, grad=False, A=True)
        gB, gA, d_flux = _field_cotangents_general(eq, u, np.asarray(o["B"]), np.asarray(o["A"]), flux, flux_scale, mu)
        d_sp = _engine_vjp_slabs(engine, sp, layout, x, gB, gA, slab)
    parts = _part_gradients(layout, d_sp)
    out = dict(parts=parts, d_flux=-float(d_flux))
    fc = [(p, l) for p, l in zip(parts, layout) if l[0] == "fourier"]
    if fc:
        out["d_coef"] = np.concatenate([p["d_coef"][l[1]["coils"]] for p, l in fc])
        out["d_current"] = np.concatenate([p["d_current"][l[1]["coils"]] for p, l in fc])
    return out


def adjoint_gradient_general(engine: FieldEngine, torus, objective="iota"):
    """d iota / d (every source parameter, the flux) at a converged torus of the general (non-stellarator-symmetric)
    basis: the adjoint of the general equations at any surface (torux.torus_general.GeneralEquations, the reference
    the torus itself, phi the cylindrical angle), J^T lambda = e_iota by LAPACK on J's own storage (J^T is J's memory
    read in Fortran order), then one vector-Jacobian product in the source (_vjp_source_general).  The dense
    Jacobian is the one large array."""
    import scipy.linalg as sla
    from .torus_general import GeneralEquations, general_arrays
    if objective != "iota":
        raise NotImplementedError("general basis: objective 'iota' only")
    ref = general_arrays(torus)
    eq = GeneralEquations(engine, torus.nfp, torus.M, torus.N, ref)
    u = eq.pack(*ref, torus.iota)
    flux_scale = float(ref[0][0, torus.N]) / max(abs(float(torus.flux)), 1e-300)
    F, J = eq.system(u, float(torus.flux), flux_scale)
    residual = float(np.max(np.abs(F)))
    g = np.zeros(len(u))
    g[-1] = 1.0                                   # iota closes the unknowns
    lu = sla.lu_factor(J.T, overwrite_a=True, check_finite=False)
    del J
    lam = sla.lu_solve(lu, g, check_finite=False)
    del lu
    out = _vjp_source_general(engine, eq, u, float(torus.flux), flux_scale, lam)
    out.update(value=float(u[-1]), residual=residual)
    return out


# --------------------------------------------------------------------------- the derivatives in one coil alone
# A source with field periods or stellarator symmetry is solved on one period, and the adjoint above differentiates
# that solve: it is exact along every change that keeps the symmetry the solve assumed, and a coil's entry is its
# part of a change made alike to it and all its copies. iota at a fixed flux is invariant under that symmetry group
# G, so its true gradient is G-invariant, and with G acting orthogonally on the parameters the true gradient is the
# group average of the one-period one: (P g).v = g.(P v) = grad.(P v) = (P grad).v = grad.v for every direction v.
# The average needs G's action on the parameters, which coil each coil's copy is and how their coefficients relate,
# and it is found here by matching the parts against their own images.

class NoMap(Exception):
    """A source whose copies are not images of its own elements in a way that maps their parameters."""


def _group(torus, general):
    """(alpha, n, F): the rotation by alpha about z and its order n, and the flip F (None without stellarator
    symmetry), that the solve of the torus assumed: rotation by 2 pi / nfp of the torus (an island torus's nfp may
    be a fraction p / q, and its rotation by 2 pi q / p has order p), and the rotation by pi about the line at phi0
    in the plane Z = 0 on the stellarator-symmetric basis."""
    from fractions import Fraction
    from .source import FLIP, rotation_z
    v = float(torus.nfp)
    f = Fraction(v).limit_denominator(1000)
    if not (v > 0 and abs(float(f) - v) <= 1e-12 * v):
        raise NoMap(f"the torus has {v} field periods, which is no rotation of finite order")
    phi0 = float(getattr(torus, "phi0", 0.0))
    F = None if general else rotation_z(2 * phi0) @ FLIP
    return 2 * np.pi * f.denominator / f.numerator, int(f.numerator), F


def _fourier_match(D, C, tol):
    """(o, e) with C the curve D traversed backwards when o = -1, its parameter then shifted, the shift
    w_k -> w_k e^k on w_k = c_k + i s_k of harmonic k, or None when C is no reparametrisation of D of that kind."""
    if np.abs(D[:, 0] - C[:, 0]).max() > tol:
        return None
    K = (D.shape[1] - 1) // 2
    k = np.arange(1, K + 1)
    wC = C[:, 2::2] + 1j * C[:, 1::2]
    for o in (1, -1):
        Do = D.copy()
        if o < 0:
            Do[:, 1::2] *= -1
        wD = Do[:, 2::2] + 1j * Do[:, 1::2]
        if K == 0 or np.abs(wD - wC).max() <= tol:
            return o, 1.0 + 0.0j
        big = np.nonzero(np.abs(wD).max(axis=0) > 1e3 * tol)[0]
        if not len(big):
            continue
        k0 = int(big[0])
        r = np.vdot(wD[:, k0], wC[:, k0])
        if abs(r) == 0.0:
            continue
        a = np.angle(r)
        for m in range(k0 + 1):
            e = np.exp(1j * (a + 2 * np.pi * m) / (k0 + 1))
            if np.abs(wD * e ** k[None, :] - wC).max() <= tol:
                return o, e
    return None


def _apply_fourier(G, M, o, e):
    """The map of _fourier_match on one coil's coefficients (or their gradient: the map is orthogonal)."""
    D = M @ G
    if o < 0:
        D[:, 1::2] *= -1
    if e != 1.0:
        K = (D.shape[1] - 1) // 2
        w = (D[:, 2::2] + 1j * D[:, 1::2]) * e ** np.arange(1, K + 1)[None, :]
        D[:, 2::2], D[:, 1::2] = w.real, w.imag
    return D


def _cyclic_order(q, n):
    """+1 or -1 when the positions q (in a loop of n points) follow each other forwards or backwards, else 0."""
    if len(q) != n:
        return 0
    d = np.mod(np.diff(np.append(q, q[0])), n)
    if np.all(d == 1 % n):
        return 1
    if np.all(d == (n - 1) % n):
        return -1
    return 0


def _action(source, M, s, tol=1e-8):
    """The parameters' map of the symmetry x -> M x of the source (s = -1: with the currents reversed, the flip of
    stellarator symmetry), as a function on per-part dicts of arrays keyed as torux.gradients' parts are
    (d_coef, d_current, d_points, ...). It maps gradients as it maps parameters, being orthogonal. Raises NoMap when
    an element's image is no element of the source of the same kind with a parameter map of the kinds matched here:
    Fourier coils up to a shift and a reversal of their parameter, polygon loops up to their first point and
    direction, segments up to their direction, dipoles and rings as they are."""
    from scipy.spatial import cKDTree
    parts = source.parts
    turn = 1.0 if M[2, 2] > 0 else -1.0              # -1: the map turns the toroidal direction round
    steps = []                                       # functions (gin, gout) filling gout from gin
    by_kind = {}
    for ip, p in enumerate(parts):
        by_kind.setdefault(p.kind, []).append(ip)

    def cat(g, ips, key):
        return np.concatenate([np.asarray(g[ip][key], dtype=np.float64) for ip in ips])

    def put(gout, ips, key, arr):
        k = 0
        for ip in ips:
            n = len(gout[ip][key])
            gout[ip][key] = arr[k:k + n]
            k += n

    def unique(idx):
        return len(np.unique(idx)) == len(idx)

    def worst(x, ref):
        return float(np.abs(x).max(initial=0.0)) / max(float(np.abs(ref).max(initial=0.0)), 1e-300)

    for kind, ips in by_kind.items():
        P = [parts[ip] for ip in ips]
        if kind == "fourier":
            coils = [(k, i) for k in range(len(P)) for i in range(P[k].n)]
            scale = max(max(float(np.abs(p.coef).max()) for p in P), 1e-300)
            Iscale = max(max(float(np.abs(p.current).max(initial=0.0)) for p in P), 1e-300)
            used, maps = set(), []
            for k, i in coils:
                D = M @ P[k].coef[i]
                hit = None
                for kq, j in coils:
                    q = P[kq]
                    if (kq, j) in used or q.coef.shape[2] != P[k].coef.shape[2]:
                        continue
                    m = _fourier_match(D, q.coef[j], tol * scale)
                    if m is not None and abs(q.current[j] - s * m[0] * P[k].current[i]) <= tol * Iscale:
                        hit = (kq, j, m[0], m[1])
                        break
                if hit is None:
                    raise NoMap(f"the image of coil {i} of part {ips[k]} is none of the source's coils with its "
                                f"coefficients mapped")
                used.add(hit[:2])
                maps.append((ips[k], i, ips[hit[0]], hit[1], hit[2], hit[3]))

            def step(gin, gout, maps=maps):
                for ip, i, jp, j, o, e in maps:
                    gout[jp]["d_coef"][j] = _apply_fourier(np.asarray(gin[ip]["d_coef"][i], dtype=np.float64), M, o, e)
                    gout[jp]["d_current"][j] = s * o * gin[ip]["d_current"][i]
            steps.append(step)
        elif kind == "polyline":
            pts = np.concatenate([p.points for p in P])
            off = np.cumsum([0] + [len(p.points) for p in P])
            nloops = np.cumsum([0] + [p.n for p in P])
            loop_of = np.concatenate([np.repeat(np.arange(p.n), np.diff(p.start)) + nloops[k] for k, p in enumerate(P)])
            first = np.concatenate([off[k] + p.start[:-1] for k, p in enumerate(P)])
            size = np.concatenate([np.diff(p.start) for p in P])
            I = np.concatenate([p.current for p in P])
            d, idx = cKDTree(pts).query(pts @ M.T)
            if worst(d, pts) > tol or not unique(idx):
                raise NoMap("the points of the source's polygon loops are not their own images")
            tgt = np.empty(len(I), dtype=np.int64)
            o = np.empty(len(I))
            for L in range(len(I)):
                t = idx[first[L]:first[L] + size[L]]
                j = int(loop_of[t[0]])
                o[L] = _cyclic_order(t - first[j], int(size[j])) if np.all(loop_of[t] == j) else 0
                tgt[L] = j
            if np.any(o == 0) or not unique(tgt) or worst(I[tgt] - s * o * I, I) > tol:
                raise NoMap("the source's polygon loops are not their own images point for point")

            def step(gin, gout, ips=ips, idx=idx, tgt=tgt, o=o):
                gp, gI = cat(gin, ips, "d_points"), cat(gin, ips, "d_current")
                op, oI = np.empty_like(gp), np.empty_like(gI)
                op[idx] = gp @ M.T
                oI[tgt] = s * o * gI
                put(gout, ips, "d_points", op)
                put(gout, ips, "d_current", oI)
            steps.append(step)
        elif kind == "segments":
            a = np.concatenate([p.a for p in P])
            b = np.concatenate([p.b for p in P])
            I = np.concatenate([p.current for p in P])
            n = len(a)
            key = np.concatenate([np.concatenate([a, b], 1), np.concatenate([b, a], 1)])
            d, idx = cKDTree(key).query(np.concatenate([a @ M.T, b @ M.T], 1))
            o = np.where(idx < n, 1.0, -1.0)
            idx = np.mod(idx, n)
            if worst(d, key) > tol or not unique(idx) or worst(I[idx] - s * o * I, I) > tol:
                raise NoMap("the source's segments are not their own images")

            def step(gin, gout, ips=ips, idx=idx, o=o):
                ga, gb, gI = (cat(gin, ips, k) for k in ("d_a", "d_b", "d_current"))
                ma, mb = ga @ M.T, gb @ M.T
                fw = (o > 0)[:, None]
                oa, ob, oI = np.empty_like(ga), np.empty_like(gb), np.empty_like(gI)
                oa[idx], ob[idx] = np.where(fw, ma, mb), np.where(fw, mb, ma)
                oI[idx] = s * o * gI
                for k, arr in (("d_a", oa), ("d_b", ob), ("d_current", oI)):
                    put(gout, ips, k, arr)
            steps.append(step)
        elif kind == "dipoles":
            pos = np.concatenate([p.pos for p in P])
            mom = np.concatenate([p.moment for p in P])
            d, idx = cKDTree(pos).query(pos @ M.T)
            if worst(d, pos) > tol or not unique(idx) or worst(mom[idx] - s * mom @ M.T, mom) > tol:
                raise NoMap("the source's dipoles are not their own images")

            def step(gin, gout, ips=ips, idx=idx):
                gp, gm = cat(gin, ips, "d_pos"), cat(gin, ips, "d_moment")
                op, om = np.empty_like(gp), np.empty_like(gm)
                op[idx], om[idx] = gp @ M.T, s * (gm @ M.T)
                put(gout, ips, "d_pos", op)
                put(gout, ips, "d_moment", om)
            steps.append(step)
        elif kind == "rings":
            R = np.concatenate([p.R for p in P])
            Z = np.concatenate([p.Z for p in P])
            I = np.concatenate([p.current for p in P])
            RZ = np.stack([R, Z], 1)
            d, idx = cKDTree(RZ).query(np.stack([R, turn * Z], 1))
            if worst(d, RZ) > tol or not unique(idx) or worst(I[idx] - s * turn * I, I) > tol:
                raise NoMap("the source's rings are not their own images")

            def step(gin, gout, ips=ips, idx=idx):
                gR, gZ, gI = (cat(gin, ips, k) for k in ("d_R", "d_Z", "d_current"))
                oR, oZ, oI = np.empty_like(gR), np.empty_like(gZ), np.empty_like(gI)
                oR[idx], oZ[idx], oI[idx] = gR, turn * gZ, s * turn * gI
                for k, arr in (("d_R", oR), ("d_Z", oZ), ("d_current", oI)):
                    put(gout, ips, k, arr)
            steps.append(step)
        elif kind == "toroidal":
            def step(gin, gout, ips=ips):
                for ip in ips:
                    gout[ip]["d_RB"] = s * turn * np.asarray(gin[ip]["d_RB"], dtype=np.float64)
            steps.append(step)
        else:
            raise NoMap(f"no map for {kind} parts")

    def act(gin):
        gout = [{k: (np.array(v, dtype=np.float64) if k != "kind" else v) for k, v in g.items()} for g in gin]
        for st in steps:
            st(gin, gout)
        return gout
    return act


def symmetric_average(source, torus, general, parts):
    """The group average of per-part gradients (torux.gradients' parts) over the symmetry the solve of the torus
    assumed: for the one-period solve's gradient of a quantity that symmetry leaves unchanged, the true gradient,
    every element's entries those of changing that element alone. None when the group is the identity alone.
    Raises NoMap when the source's copies give no map between their parameters, or maps that do not close into
    the group (a rotation n times over, the flip twice, and flip, rotation, flip, rotation each the identity)."""
    alpha, n, F = _group(torus, general)
    if n == 1 and F is None:
        return None
    from .source import rotation_z

    def lin(a, b, wa, wb):
        return [{k: (wa * x[k] + wb * y[k] if k != "kind" else x[k]) for k in x} for x, y in zip(a, b)]
    g = [{k: (np.asarray(v, dtype=np.float64) if k != "kind" else v) for k, v in p.items()} for p in parts]
    rng = np.random.default_rng(0)
    probe = [{k: (rng.standard_normal(np.shape(v)) if k != "kind" else v) for k, v in p.items()} for p in g]
    flip = _action(source, F, -1.0) if F is not None else (lambda x: x)
    rot = _action(source, rotation_z(alpha), 1.0) if n > 1 else (lambda x: x)

    def times(f, k, x):
        for _ in range(k):
            x = f(x)
        return x
    closes = [("the rotation", times(rot, n, probe))]
    if F is not None:
        # with the flip, the group is dihedral: flip twice and (flip, rotation) twice are each the identity. Without
        # it the group is the rotations alone, closed by the first
        closes += [("the flip", flip(flip(probe))), ("flip and rotation", flip(rot(flip(rot(probe)))))]
    for name, back in closes:
        if max(float(np.abs(x[k] - y[k]).max(initial=0.0)) for x, y in zip(back, probe) for k in x if k != "kind") > 1e-9:
            raise NoMap(f"the maps found for the source's copies do not close: {name} is not the identity")
    if F is not None:
        g = lin(g, flip(g), 0.5, 0.5)
    if n > 1:
        acc, cur = g, g
        for _ in range(n - 1):
            cur = rot(cur)
            acc = lin(acc, cur, 1.0, 1.0)
        g = lin(acc, acc, 1.0 / n, 0.0)
    return g


ONE_PERIOD = ("the derivatives of the solve on one field period: exact for every change that keeps the source's "
              "symmetry, and a coil's entries are its part of a change made alike to it and all its copies, which "
              "only their sum over the copies means")


def iota_gradients(source, torus, general, g):
    """answer.gradients from an adjoint's result g (adjoint_gradient and its kin): d_iota_d_flux, parts and, for
    Fourier coils with current, d_iota_d_coef and d_iota_d_current, each element's entries the derivatives of iota in
    that element alone (symmetric_average). When the source's copies cannot be mapped those keys are left out:
    per_coil says why, and one_period holds the one-period solve's derivatives under the same names."""
    out = dict(objective="iota", d_iota_d_flux=g["d_flux"])
    try:
        parts = symmetric_average(source, torus, general, g["parts"])
    except NoMap as e:
        one = dict(meaning=ONE_PERIOD, parts=g["parts"])
        if "d_coef" in g:
            one.update(d_iota_d_coef=g["d_coef"], d_iota_d_current=g["d_current"])
        out.update(per_coil=f"not given: {e}, so the derivatives in one coil alone are not known", one_period=one)
        return out
    if parts is None:
        out["parts"] = g["parts"]
        if "d_coef" in g:
            out.update(d_iota_d_coef=g["d_coef"], d_iota_d_current=g["d_current"])
        return out
    out["parts"] = parts
    fc = [(p, sp) for p, sp in zip(parts, source.parts) if sp.kind == "fourier"]
    if fc:
        keep = [np.nonzero(sp.current != 0.0)[0] for _, sp in fc]
        out["d_iota_d_coef"] = np.concatenate([p["d_coef"][k] for (p, _), k in zip(fc, keep)])
        out["d_iota_d_current"] = np.concatenate([p["d_current"][k] for (p, _), k in zip(fc, keep)])
    return out
