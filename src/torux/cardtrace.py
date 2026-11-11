"""Field-line period maps for large batches of points on a GPU (JAX, float64), for the period-map surrogates of
torux.islandband (one batch of accurate traces of 10^4 - 10^6 nodes: on 4 CPU cores 100 - 1500 s, the step that
ran the full path's island bands out of time).

FilamentField: B of the engine's Fourier-filament nodes (the CPU kernel's nodes and weights, torux.seeds.FastB's sum)
as a jitted JAX sum over node chunks; checked against the engine at construction.
GPUPeriodMap: the period map of torux.fasttrace.LineTracer (Gragg-Bulirsch-Stoer, modified midpoint with n_j = 2j
substeps, j = 1..k, Aitken-Neville in h^2, error estimate |T_kk - T_k,k-1| summed over the macro steps; lines stopped
where |B_phi| < 1e-3 |B|, B_phi takes the other sign, R <= 0 or the field is not finite, their last good state kept)
-- the same scheme step for step, so its results equal the CPU tracer's to rounding.

Nothing proved here depends on this module being right: the surrogates built from these traces are validated, as
before, against the CPU accurate tracer at random points of the annulus (torux.islandband: tau = 3 x the largest
difference + 2 x the tracer's model), and every run also compares a sample of these traces with the CPU tracer's
(torux.islandband records the difference and refuses the GPU traces above 1e-12 of the scale).
"""
from __future__ import annotations

import math
import os

import numpy as np

_JAX = {}


def _card_present():
    """A CUDA card on this machine, judged without initialising JAX (the driver's tool or the device files)."""
    import shutil
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"):
        return False
    return bool(shutil.which("nvidia-smi")) or os.path.exists("/dev/nvidia0")


def prefer_cpu_default():
    """Make the CPU JAX's default backend while keeping a card addressable (called before JAX initialises its
    backends): the solver runs as on a CPU node (numba kernels, host Newton), the period-map batches go to the card.
    Without a card nothing changes."""
    if os.environ.get("TORUX_CARD_TRACES", "1") == "0" or not _card_present():
        return False
    import jax
    try:
        jax.config.update("jax_platforms", "cpu,cuda")
        jax.devices()                       # initialise now: a card that does not come up must not stop the run
        return True
    except Exception:  # noqa: BLE001
        try:
            from jax._src import xla_bridge as _xb
            if hasattr(_xb, "_clear_backends"):
                _xb._clear_backends()
        except Exception:  # noqa: BLE001
            pass
        jax.config.update("jax_platforms", "cpu")
        return False


def _gpu_devices():
    import jax
    for name in ("gpu", "cuda"):
        try:
            d = jax.devices(name)
            if d:
                return d
        except Exception:  # noqa: BLE001
            pass
    return []


def gpu_available():
    """A GPU JAX device (TORUX_CARD_TRACES=0 switches the GPU path off)."""
    if os.environ.get("TORUX_CARD_TRACES", "1") == "0":
        return False
    return bool(_gpu_devices())


def _gpu_device():
    return _gpu_devices()[0]


class FilamentField:
    """B(x) = sum_j w_j x (x - r_j) / |x - r_j|^3 over the engine's filament nodes (Fourier coils only)."""

    def __init__(self, engine, node_chunk=2048, check=True):
        import jax
        import jax.numpy as jnp
        P = getattr(engine, "_np", None)
        kinds = getattr(engine, "kinds", (True, False, False, False, False))
        if any(kinds[1:]) or getattr(engine, "analytic", None):
            raise ValueError("FilamentField: engines with Fourier filaments only")
        if P is not None:
            r_, w_ = np.asarray(P["r"], float), np.asarray(P["w"], float)
        else:
            # (an engine on JAX kernels: its padded node arrays; padding rows have zero weight)
            r_, w_ = np.asarray(engine.params["r"], float), np.asarray(engine.params["w"], float)
            keep = np.any(w_ != 0.0, axis=1)
            r_, w_ = r_[keep], w_[keep]
        if not len(r_):
            raise ValueError("FilamentField: no filament nodes")
        self.r_np = np.ascontiguousarray(r_)
        self.w_np = np.ascontiguousarray(w_)
        n = len(self.r_np)
        nc = int(math.ceil(n / node_chunk))
        pad = nc * node_chunk - n
        far = 1e3 * (1.0 + float(np.max(np.abs(self.r_np))))
        r = np.concatenate([self.r_np, np.full((pad, 3), far)])
        w = np.concatenate([self.w_np, np.zeros((pad, 3))])
        dev = _gpu_device() if gpu_available() else jax.devices()[0]
        self.device = dev
        self.r = jax.device_put(jnp.asarray(r.reshape(nc, node_chunk, 3)), dev)
        self.w = jax.device_put(jnp.asarray(w.reshape(nc, node_chunk, 3)), dev)

        def B(x):
            def body(acc, rw):
                rr, ww = rw
                d = x[:, None, :] - rr[None, :, :]
                q = jnp.sum(d * d, -1)
                ir3 = q ** -1.5
                c = jnp.stack([ww[None, :, 1] * d[..., 2] - ww[None, :, 2] * d[..., 1],
                               ww[None, :, 2] * d[..., 0] - ww[None, :, 0] * d[..., 2],
                               ww[None, :, 0] * d[..., 1] - ww[None, :, 1] * d[..., 0]], -1)
                return acc + jnp.sum(c * ir3[..., None], axis=1), None
            acc, _ = jax.lax.scan(body, jnp.zeros_like(x), (self.r, self.w))
            return acc
        self._B = jax.jit(B)
        self.check_err = None
        if check:
            rng = np.random.default_rng(0)
            rad = float(np.median(np.hypot(self.r_np[:, 0], self.r_np[:, 1])))
            ph = rng.uniform(0, 2 * np.pi, 64)
            x = np.stack([rad * np.cos(ph), rad * np.sin(ph), 0.05 * rad * rng.standard_normal(64)], -1)
            Be = np.asarray(engine.evaluate(x, grad=False, A=False)["B"])
            Bg = self.B(x)
            self.check_err = float(np.max(np.linalg.norm(Bg - Be, axis=1) / np.linalg.norm(Be, axis=1)))
            if not self.check_err < 1e-13:
                raise ValueError(f"FilamentField differs from the engine by {self.check_err:.2e}")

    def B(self, x):
        import jax
        import jax.numpy as jnp
        x = jax.device_put(jnp.asarray(np.asarray(x, float).reshape(-1, 3)), self.device)
        return np.asarray(self._B(x))


class GPUPeriodMap:
    """Period maps from the plane phi0 for batches of (R, Z) points: (y, lost, err) as
    torux.fasttrace.LineTracer.period_map(y, 1, +1, phi0) at `sp` macro steps of order 2k."""

    def __init__(self, field: FilamentField, nfp, sign, k=6, bphi_frac=1e-3, batch=16384):
        import jax
        import jax.numpy as jnp
        self.field, self.nfp, self.sign, self.k, self.frac = field, int(nfp), float(sign), int(k), float(bphi_frac)
        self.batch = int(batch)
        self.period = 2 * np.pi / self.nfp
        self._fns = {}
        self.jnp = jnp
        self.jax = jax

    def _fn(self, sp):
        if sp in self._fns:
            return self._fns[sp]
        jax, jnp = self.jax, self.jnp
        k, frac, sign = self.k, self.frac, self.sign
        Bf = self.field._B
        H = self.period / sp
        ns = [2 * j for j in range(1, k + 1)]

        def rhs(phi, y):
            R, Z = y[:, 0], y[:, 1]
            c, s = jnp.cos(phi), jnp.sin(phi)
            x = jnp.stack([R * c, R * s, Z], -1)
            B = Bf(x)
            BR = B[:, 0] * c + B[:, 1] * s
            Bp = -B[:, 0] * s + B[:, 1] * c
            Bm = jnp.sqrt(jnp.sum(B * B, -1))
            ok = jnp.isfinite(Bm) & (R > 0) & (sign * Bp >= frac * Bm) & jnp.all(jnp.isfinite(y), -1)
            Bps = jnp.where(ok, Bp, 1.0)
            return jnp.stack([R * BR / Bps, R * B[:, 2] / Bps], -1), ok

        def step(phi, y):
            f0, ok = rhs(phi, y)
            f0 = jnp.where(ok[:, None], f0, 0.0)
            T = []
            for n in ns:
                h = H / n
                z0 = y
                z1 = y + h * f0
                for m in range(1, n):
                    fz, okz = rhs(phi + m * h, z1)
                    ok = ok & okz
                    fz = jnp.where(okz[:, None], fz, 0.0)
                    z2 = z0 + 2.0 * h * fz
                    z0, z1 = z1, z2
                T.append(z1)
            # Aitken-Neville in h^2 (torux.fasttrace._neville)
            tab = [T[0]]
            for j in range(1, k):
                row = [T[j]]
                for kk in range(1, j + 1):
                    r = (ns[j] / ns[j - kk]) ** 2 - 1.0
                    row.append(row[kk - 1] + (row[kk - 1] - tab[kk - 1]) / r)
                tab = row
            yn = tab[-1]
            err = jnp.max(jnp.abs(tab[-1] - tab[-2]), axis=-1)
            ok = ok & jnp.all(jnp.isfinite(yn), -1)
            return yn, err, ok

        def run(y0, phi0):
            def body(i, st):
                y, err, lost, phi = st
                yn, e, ok = step(phi, y)
                good = (~lost) & ok
                y = jnp.where(good[:, None], yn, y)
                err = jnp.where(good, err + e, err)
                lost = lost | ((~lost) & (~ok))
                return y, err, lost, phi + H
            L = y0.shape[0]
            lost0 = ~jnp.all(jnp.isfinite(y0), -1)
            y, err, lost, _ = jax.lax.fori_loop(0, sp, body, (y0, jnp.zeros(L), lost0, phi0))
            return y, lost, err
        fn = jax.jit(run)
        self._fns[sp] = fn
        return fn

    def period_map(self, y, sp, phi0=0.0, deadline=None, factor=1.25, what="nodes"):
        """y (n, 2) -> (y1, lost, err) after one field period from phi0 at sp macro steps; batches of self.batch
        (the last padded).  deadline: refuse (torux.islandband.SurrogateDeadline) when the rest, at the measured rate,
        would end beyond it (factor), as the CPU path does."""
        import time
        jax, jnp = self.jax, self.jnp
        y = np.asarray(y, float).reshape(-1, 2)
        n = len(y)
        out_y = np.empty_like(y)
        out_l = np.zeros(n, bool)
        out_e = np.zeros(n)
        fn = self._fn(sp)
        b = min(self.batch, max(256, 1 << int(math.ceil(math.log2(max(n, 1))))))
        t0 = time.time()
        for s in range(0, n, b):
            blk = y[s:s + b]
            m = len(blk)
            if m < b:
                blk = np.concatenate([blk, np.repeat(blk[-1:], b - m, 0)])
            while True:
                try:
                    yy, ll, ee = fn(jax.device_put(jnp.asarray(blk), self.field.device), jnp.asarray(float(phi0)))
                    break
                except Exception as e:  # noqa: BLE001  (card memory: halve the batch and go on)
                    if "RESOURCE_EXHAUSTED" not in repr(e) and "out of memory" not in repr(e).lower() or b <= 256:
                        raise
                    self.batch = b = b // 2
                    return self.period_map(y, sp, phi0=phi0, deadline=deadline, factor=factor, what=what)
            out_y[s:s + m] = np.asarray(yy)[:m]
            out_l[s:s + m] = np.asarray(ll)[:m]
            out_e[s:s + m] = np.asarray(ee)[:m]
            if deadline is not None and s + b < n:
                el = time.time() - t0
                rest = el / (s + b) * (n - s - b)
                if time.time() + factor * rest > deadline:
                    from .islandband import SurrogateDeadline
                    raise SurrogateDeadline(f"GPU {what}: {n - s - b} of {n} still to trace, about {rest:.0f} s at the "
                                            f"measured rate, {deadline - time.time():.0f} s to the deadline")
        return out_y, out_l, out_e
