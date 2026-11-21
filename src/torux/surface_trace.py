"""Direct field-line tracing with an exterior-only current sheet."""

import time

import numpy as np
from scipy.integrate import DOP853

from .field.surface_current import SurfaceCurrentDomainError


def _xyz(r, z, c, s):
    return np.array([[r*c, r*s, z]])


def _clear(engine, phi, y):
    r, z = y
    return r > 0 and np.isfinite(y).all() and all(sheet.clearance(r, z, phi) > 0 for sheet in engine.sheets)


def _trace(engine, seed, phi0, targets, step):
    """Crossings in increasing phi, the last accepted state on exit and its reason."""
    r0, z0 = seed
    c, s = np.cos(phi0), np.sin(phi0)
    bx, by, _ = np.asarray(engine.B(_xyz(r0, z0, c, s)))[0]
    sign = np.sign(c*by - s*bx)

    def velocity(phi, y):
        r, z = y
        if r <= 0 or not np.isfinite(y).all():
            raise SurfaceCurrentDomainError("field line left its cylindrical coordinates", "coordinates")
        c, s = np.cos(phi), np.sin(phi)
        b = np.asarray(engine.B(_xyz(r, z, c, s)))[0]
        bphi = c*b[1] - s*b[0]
        if not np.isfinite(b).all():
            raise SurfaceCurrentDomainError("field line reached an undefined field", "undefined_field")
        if sign*bphi <= 1e-12 * np.max(np.abs(b)):
            raise SurfaceCurrentDomainError("field line lost its toroidal direction", "toroidal_direction")
        return (r / bphi) * np.array([c*b[0] + s*b[1], b[2]])

    end = targets[-1]
    min_step = 1e-7
    data = np.full((len(targets), 2), np.nan)
    index, phi, y = 0, float(phi0), seed
    solver = None
    while phi < end:
        # A rejected trial can have stages inside the sheet. Retry from the
        # last accepted point with smaller steps, without evaluating there.
        try:
            if solver is None:
                solver = DOP853(velocity, phi, y, end, max_step=step, first_step=min(step, end-phi),
                                rtol=2e-10, atol=2e-12)
            solver.step()
            if solver.status == "failed":
                raise SurfaceCurrentDomainError("field-line integration failed", "integration_limit")
            hi = int(np.searchsorted(targets, solver.t, side="right"))
            if hi > index:
                cut = solver.dense_output()(targets[index:hi]).T
                if not all(_clear(engine, angle, p) for angle, p in zip(targets[index:hi], cut)):
                    raise SurfaceCurrentDomainError("field line reached the surface current boundary")
                data[index:hi] = cut
                index = hi
            phi, y = solver.t, solver.y
        except SurfaceCurrentDomainError as e:
            step *= .5
            if step < min_step:
                return data, np.array([y[0], phi - phi0, y[1]]), e.reason
            solver = None
    return data, np.full(3, np.nan), None


def exterior_section(engine, planes, starts, nper, nfp, began):
    from .poincare_section import Poincare

    period = 2 * np.pi / nfp
    phi0 = planes[0]
    for seed in starts:
        if not _clear(engine, phi0, seed):
            raise SurfaceCurrentDomainError(f"start {tuple(seed)} is inside or too close to the surface current boundary")
    rel = np.mod(planes - phi0, period)
    rel[np.isclose(rel, period, rtol=0, atol=1e-12) | (rel == 0)] = period
    order = np.argsort(rel)
    targets = phi0 + (np.arange(nper)[:, None] * period + rel[order]).ravel()
    paths, left_at, reasons = zip(*(_trace(engine, p, phi0, targets, period/32) for p in starts))
    crosses = np.asarray(paths).reshape(-1, nper, len(planes), 2)[:, :, np.argsort(order)]
    R, Z = crosses.transpose(3, 0, 2, 1)
    followed = np.isfinite(R[:, 0]).sum(axis=1)
    left_at = np.asarray(left_at)
    keep = max(1, int(np.count_nonzero(np.isfinite(R).any(axis=(0, 1)))))
    n = len(starts)
    return Poincare(planes=planes, R=R[..., :keep], Z=Z[..., :keep], start=starts, left=np.isfinite(left_at[:, 1]),
                    left_at=left_at, stop_reason=reasons, iota=np.full(n, np.nan), axis=None, turns=nper/nfp, periods=nper,
                    followed=followed, nfp=nfp, open=np.zeros(n, bool), source_id=str(engine.source.id),
                    seconds=time.time() - began, steps=None, order=8, surface_line=None, held=None,
                    _answer=None, _source=engine.source, _spread=np.full(n, np.nan),
                    _timings={}, _nodes=engine.info.n_nodes)
