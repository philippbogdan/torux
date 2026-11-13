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
    """d iota / d (every source parameter, the flux) at a converged torus of the general (non-stellarator-
    symmetric) basis (torux.torus_general): the same adjoint as adjoint_gradient on that system (Galerkin residuals
    of R and Z, the flux equation, the phase condition; unknowns the four coefficient arrays, iota and the
    unfolding lambda): J^T lambda = e_iota, then one vector-Jacobian product of the residual with respect to the
    source through the differentiable engine."""
    from .torus_general import GeneralTorusProblem, phase_row, _grid_quantities, _residual
    if objective != "iota":
        raise NotImplementedError("general basis: objective 'iota' only")
    prob = GeneralTorusProblem(engine, torus.nfp, torus.M, torus.N, phi0=torus.phi0)
    b = prob.basis
    u = jnp.asarray(prob.pack(torus))
    ph = phase_row(b, torus)
    R00 = float(torus.rc[0, b.N])
    flux_scale = R00 / max(abs(torus.flux), 1e-300)
    F, J, flux, res = prob.system(u, torus.flux, flux_scale, ph)
    k = int(u.shape[0]) - 2                       # iota (then lambda) closes the unknowns
    g = jnp.zeros_like(u).at[k].set(1.0)
    lam = jnp.linalg.solve(jnp.asarray(J).T, g)
    sp, layout = source_params(engine.source)

    def Fp(sp_, flux_t):
        de = _DiffEngine(engine, sp_, layout)
        rc, rs, zc, zs, _, _ = b.unpack(u)
        R, Z, *_ = b.geometry(rc, rs, zc, zs)
        x = b.points(R, Z)[0]
        o = de.evaluate(x, grad=True, A=True)
        G = _grid_quantities(b, u, o["B"], o["A"], o["G"])
        Fv, _, _ = _residual(b, G, u, flux_t, flux_scale, ph, 0.0, False)
        return Fv
    _, vjp = jax.vjp(Fp, sp, jnp.asarray(float(torus.flux)))
    d_sp, d_flux = vjp(lam)
    out = dict(parts=_part_gradients(layout, d_sp), d_flux=-float(d_flux), value=float(u[k]),
               residual=float(jnp.max(jnp.abs(F))))
    fc = [(p, l) for p, l in zip(out["parts"], layout) if l[0] == "fourier"]
    if fc:
        out["d_coef"] = np.concatenate([p["d_coef"][l[1]["coils"]] for p, l in fc])
        out["d_current"] = np.concatenate([p["d_current"][l[1]["coils"]] for p, l in fc])
    return out
