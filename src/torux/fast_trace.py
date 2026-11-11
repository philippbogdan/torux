"""Work reductions of the tracers under the certificates (island bands, edge barriers): switches.

The recommended set (bonly: bit-identical) is on by default.  TORUX_FAST_TRACE overrides it per process: a
comma-separated list of names, "all" for the recommended set, "every" for every switch, "none" for none.  In code:
``set_opts(...)`` / ``opts(...)``.  Independent of torux.fast_design, fast_family and fast_newton.

  bonly   B-only field evaluations of the tracers (torux.fasttrace.LineTracer, torux.barrier.FieldSection,
          torux.seeds.FastB's large batches) by field_B below: on the host CPU path the engine's own numba kernels on
          the same arrays, summed in the same order (filaments, then segments, then dipoles, into a zero array),
          returned as numpy -- without FieldEngine.evaluate's A and grad B arrays, its three JAX array conversions
          and the caller's conversion back.  The same arithmetic on the same inputs: the same bits.  Engines on a
          card (fast_family 'cudafil'), with JAX-only kinds or analytic parts, and objects other than a FieldEngine
          go through evaluate.
  tlean   (experimental) the island band's node traces, validation traces and accurate-tracer growth on a lean engine
          (LeanTraceEngine below): per Fourier coil the fewest trapezoid nodes (multiples of 8; the next multiple
          meeting the bound too, the larger taken) whose field on a tube of radius rho around the coil is within
          TLEAN_EPS Bmin of a converged reference (twice the standard engine's nodes) (rho = TLEAN_MARGIN x the
          smallest distance from any coil to the outer torus of the band's annulus, Bmin = the smallest |B| on that
          torus), then certified: each Cartesian component of a trapezoid Biot-Savart field is a sum of derivatives of
          1 / |x - r_j| (r_j the nodes, on the coils), harmonic off the nodes, and decays at infinity (the trapezoid
          sum of r' over a closed Fourier coil is zero), so the difference of two engines' fields is harmonic in the
          domain D outside the tubes and its largest magnitude over D is on the tubes' surfaces (maximum principle;
          |dB| is subharmonic); checked on the tube samples (each coil's own tube, the parts outside the other tubes)
          and on 4x finer patches around the 16 largest differences: max |B_lean - B_std| <= TLEAN_CERT Bmin, else no
          lean engine (the standard tracing) -- a check on samples, not a bound between them, as fast_newton 'fmeas'
          (the difference varies on the scale of the tube radius and of the lean nodes' spacing; the samples lie at
          half the standard node spacing along each coil, at least 2.6 per lean spacing where a lean engine is kept,
          8 around).  At run time every B-only evaluation point closer than rho + h (h: the coil samples' spacing) to
          any coil sample is evaluated by the standard engine, so every lean value lies in D.  A, grad B and the
          fluxes always come from the standard engine.  The tracer's error model (2 x GBS + 2e-13 R per period) is
          unchanged; the field difference moves measured period maps by 2-4e-14 (0.1-0.2 of that model's 2e-13 R
          floor, which every run's reference evidence checks).
  snodes  (experimental) a thin annulus' band surrogate (torux.islandband.BandSurrogate: 12 x 8 nodes per patch by
          default, the annulus narrower than half a patch) with SNODES_NT x SNODES_ND = 9 x 5.  Every build validates
          itself (tau = 3 x its check difference + 2 x the tracer's model), so the error model is the standard; the
          surrogates and their tau differ, hence outcomes may.
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np

NAMES = ("bonly", "tlean", "snodes")
# recommended: bonly, bit-identical (same kernels and order)
EXPERIMENTAL = ("tlean", "snodes")
RECOMMENDED = tuple(n for n in NAMES if n not in EXPERIMENTAL)

TLEAN_MARGIN = 0.75     # 'tlean': the tubes' radius, a share of the smallest coil distance to the annulus' outer torus
TLEAN_EPS = 2.5e-15     # 'tlean': each coil's field on its tube within this x Bmin of a converged reference (the rounding floor of
                        # a coil's own field on its tube is ~1e-15 Bmin)
TLEAN_CERT = 5e-15      # 'tlean': the certified bound on |B_lean - B_rel| / Bmin outside the tubes
TLEAN_AROUND = 8        # 'tlean': tube samples around each coil point
TLEAN_MIN_GAIN = 0.85   # 'tlean': no lean engine unless it has at most this share of the standard engine's nodes
SNODES_NT = 9           # 'snodes': a thin annulus' band-surrogate patches get SNODES_NT x SNODES_ND Chebyshev-Lobatto nodes
SNODES_ND = 5           #           (release: 12 x 8)


def _params_from_env():
    """TORUX_FAST_TRACE_PARAMS='TLEAN_MARGIN=0.6,TLEAN_CERT=1e-15' overrides the constants above (comparisons)."""
    v = os.environ.get("TORUX_FAST_TRACE_PARAMS")
    if not v:
        return
    g = globals()
    for item in v.split(","):
        k, _, x = item.partition("=")
        k = k.strip()
        if k not in ("TLEAN_MARGIN", "TLEAN_EPS", "TLEAN_CERT", "TLEAN_AROUND", "TLEAN_MIN_GAIN", "SNODES_NT", "SNODES_ND"):
            raise ValueError(f"TORUX_FAST_TRACE_PARAMS: unknown parameter {k}")
        g[k] = float(x) if isinstance(g[k], float) else int(x)


_params_from_env()


def _from_env():
    v = os.environ.get("TORUX_FAST_TRACE")
    if v is None or not v.strip():
        return set(RECOMMENDED)
    v = v.strip()
    if v in ("none", "off", "base"):
        return set()
    if v == "all":
        return set(RECOMMENDED)
    if v == "every":
        return set(NAMES)
    out = set()
    for s in v.split(","):
        s = s.strip()
        if not s:
            continue
        out |= set(RECOMMENDED) if s == "all" else {s}
    bad = out - set(NAMES)
    if bad:
        raise ValueError(f"TORUX_FAST_TRACE: unknown switches {sorted(bad)} (known: {NAMES})")
    return out


_on = _from_env()


def on(name):
    return name in _on


def active():
    return sorted(_on)


def set_opts(names):
    global _on
    names = set(RECOMMENDED) if names == "all" else set(NAMES) if names == "every" else set(names)
    bad = names - set(NAMES)
    if bad:
        raise ValueError(f"unknown switches {sorted(bad)}")
    _on = names


@contextlib.contextmanager
def opts(names):
    global _on
    old = _on
    set_opts(names)
    try:
        yield
    finally:
        _on = old


# ------------------------------------------------------------------------------------------------------------ bonly
def _host_plain(engine):
    """True for a FieldEngine on the host CPU path whose B evaluate sums numba kernels only (no card, no JAX kinds)."""
    from .field.engine import FieldEngine
    if type(engine) is not FieldEngine or not getattr(engine, "cpu_kernels", False):
        return False
    from . import fast_family as _so2
    if _so2.on("cudafil"):
        return False
    if any(getattr(engine, "_rest_kinds", ()) or ()) or getattr(engine, "analytic", None):
        return False
    return getattr(engine, "_np", None) is not None


def field_B(engine, x):
    """B (n, 3) numpy at points x for a tracer: np.asarray(engine.evaluate(x, grad=False, A=False)["B"]), by the same
    kernels in the same order without evaluate's other arrays and JAX conversions where 'bonly' is on (see above)."""
    if isinstance(engine, LeanTraceEngine):
        return engine.B_np(x)
    if on("bonly") and _host_plain(engine):
        return _b_host(engine, x)
    return np.asarray(engine.evaluate(x, grad=False, A=False)["B"], dtype=float)


def _b_host(engine, x):
    from .field import cpu as _cpu
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    T = x.shape[0]
    B = np.zeros((T, 3))
    P = engine._np
    if len(P["r"]):
        from . import fast_kernels as _so5
        if _so5.on("bkern"):
            # (fast_kernels 'bkern') the kernel's B rows only: the same terms in the same order, the same bits
            B += _so5.fil_B(x, P["r"], P["w"])
        else:
            o = _cpu.filament(x, P["r"], P["w"], P["wh"], False, False)
            B += o[:, 0:3]
    if len(P["sa"]):
        o = _cpu.segments(x, P["sa"], P["sb"], P["sw"], False, False)
        B += o[:, 0:3]
    if len(P["dp"]):
        o = _cpu.dipoles(x, P["dp"], P["dm"], False, False)
        B += o[:, 0:3]
    return B


# ------------------------------------------------------------------------------------------------------------ tlean
class LeanTraceEngine:
    """'tlean': B-only evaluations by a certified lean engine outside the coils' tubes, the standard engine inside them
    and for everything else (module docstring).  Delegates every other attribute to the standard engine."""

    def __init__(self, rel, lean, tree, rho, h, info):
        self._rel = rel
        self._lean = lean
        self._tree = tree
        self._rho = float(rho)
        self._h = float(h)
        self.tlean_info = info
        self.n_lean = 0
        self.n_rel = 0

    def __getattr__(self, name):
        return getattr(self.__dict__["_rel"], name)

    def B_np(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        d, _ = self._tree.query(x, k=1, distance_upper_bound=self._rho + self._h)
        near = np.isfinite(d)
        if not near.any():
            self.n_lean += len(x)
            return _b_any(self._lean, x)
        B = np.empty((len(x), 3))
        far = ~near
        if far.any():
            B[far] = _b_any(self._lean, x[far])
        B[near] = _b_any(self._rel, x[near])
        self.n_lean += int(far.sum())
        self.n_rel += int(near.sum())
        return B

    def evaluate(self, x, grad=True, A=True, **kw):
        if grad or A or kw.get("err") or kw.get("params") is not None:
            return self._rel.evaluate(x, grad=grad, A=A, **kw)
        import jax.numpy as jnp
        B = self.B_np(x)
        return dict(B=jnp.asarray(B), A=jnp.zeros_like(jnp.asarray(B)), G=jnp.zeros((len(B), 3, 3)))

    def B(self, x):
        return self.B_np(x)


def _b_any(engine, x):
    if _host_plain(engine):
        return _b_host(engine, x)
    return np.asarray(engine.evaluate(x, grad=False, A=False)["B"], dtype=float)


def _frame(t):
    """Unit normals (n1, n2) orthogonal to the unit tangents t (n, 3)."""
    a = np.where(np.abs(t[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    n1 = np.cross(t, a)
    n1 /= np.linalg.norm(n1, axis=1)[:, None]
    n2 = np.cross(t, n1)
    return n1, n2


def _coil_field(x, p, i, N):
    from .field import cpu as _cpu
    from .field.engine import fourier_nodes_np
    r, w, wh = fourier_nodes_np(p.coef[i], p.current[i], N)
    return _cpu.filament(x, r, w, wh, False, False)[:, 0:3]


def lean_trace_engine(rel, source, region_pts, region_h=0.0):
    """'tlean': a LeanTraceEngine for B-only evaluations around the region (points (k, 3) of the annulus' outer torus,
    every field period), or (None, info) when the standard engine is kept (not a plain host engine, no gain, the
    certificate does not hold).  Returns (engine or None, info)."""
    from scipy.spatial import cKDTree
    from .field import FieldEngine
    from .field.engine import coil_samples, min_sample_distance, _round_up
    import time
    t0 = time.time()
    info = dict()
    if not _host_plain(rel):
        info["refused"] = "not a plain host engine"
        return None, info
    region_pts = np.asarray(region_pts, float)
    samples = coil_samples(source)
    if not samples:
        info["refused"] = "no Fourier coils"
        return None, info
    try:
        Nrel = [int(n) for n in rel.info.N_per_coil]
    except Exception:  # noqa: BLE001
        info["refused"] = "standard engine without per-coil counts"
        return None, info
    if len(Nrel) != len(samples):
        info["refused"] = "coil count mismatch"
        return None, info
    treg = cKDTree(region_pts)
    dist = []
    hs = []
    for p, i, xs, dxs in samples:
        vmax = float(np.max(np.linalg.norm(dxs, axis=1)))
        dist.append(min_sample_distance(treg, xs, vmax) - 0.5 * float(region_h))
        hs.append(1.5 * vmax / len(xs))
    dmin = float(min(dist))
    if not (dmin > 0):
        info["refused"] = "a coil touches the region"
        return None, info
    rho = TLEAN_MARGIN * dmin
    h = float(max(hs))
    Bmin = float(np.min(np.linalg.norm(_b_any(rel, region_pts), axis=1)))
    info.update(rho=rho, dmin=dmin, Bmin=Bmin)
    # tube samples: along each coil at the standard node spacing / 2, TLEAN_AROUND around; each coil's own tube
    allxs = np.concatenate([s[2] for s in samples])
    owner = np.concatenate([np.full(len(s[2]), k) for k, s in enumerate(samples)])
    tcoil = cKDTree(allxs)
    tubes = []
    ang = 2 * np.pi * np.arange(TLEAN_AROUND) / TLEAN_AROUND
    for k, (p, i, xs, dxs) in enumerate(samples):
        m = int(min(len(xs), max(64, 2 * Nrel[k])))
        idx = np.unique((np.arange(m) * len(xs)) // m)
        t = dxs[idx] / np.linalg.norm(dxs[idx], axis=1)[:, None]
        n1, n2 = _frame(t)
        P = (xs[idx][:, None, :] + rho * (np.cos(ang)[None, :, None] * n1[:, None, :] +
                                          np.sin(ang)[None, :, None] * n2[:, None, :])).reshape(-1, 3)
        # the parts of the tube outside the other tubes (the boundary of D)
        dd, jj = tcoil.query(P, k=1)
        keep = (owner[jj] == k) | (dd >= rho - h)
        tubes.append(P[keep])
    # per coil: the fewest nodes meeting TLEAN_EPS Bmin on its own tube
    Nl = []
    for k, (p, i, xs, dxs) in enumerate(samples):
        X = tubes[k]
        # against a converged reference (twice the standard nodes): the standard own error on the tube is ~1e-15 Bmin
        ref = _coil_field(X, p, i, 2 * Nrel[k])
        seq = []
        N = _round_up(max(4 * p.order + 8, 16), 8)
        while N < Nrel[k]:
            seq.append(N)
            N = _round_up(int(math.ceil(1.25 * N)), 8)
        seq.append(Nrel[k])
        Nf = Nrel[k]
        cache = {}

        def err(N):
            if N not in cache:
                cache[N] = float(np.max(np.linalg.norm(_coil_field(X, p, i, N) - ref, axis=1)))
            return cache[N]
        bar = TLEAN_EPS * Bmin
        # the first ladder candidate meeting the bar, then the smallest multiple of 8 above its predecessor whose next
        # multiple of 8 meets it too (the trapezoid error decays geometrically once resolved); the larger one is taken
        # (one step of margin, torux.field.engine.family_nodes' rule)
        lo = None
        for j, N in enumerate(seq):
            if err(N) <= bar:
                lo = seq[j - 1] if j > 0 else max(8, N - 8)
                break
        if lo is not None:
            for N in range(_round_up(lo, 8), Nrel[k] + 1, 8):
                if err(N) <= bar and err(N + 8) <= bar:
                    Nf = min(N + 8, Nrel[k])
                    break
        Nl.append(int(Nf))
    info.update(nodes_rel=int(sum(Nrel)), nodes_lean=int(sum(Nl)))
    if sum(Nl) > TLEAN_MIN_GAIN * sum(Nrel):
        info["refused"] = "no gain"
        info["s"] = time.time() - t0
        return None, info
    lean = FieldEngine(source, dmin=rel.dmin, node_multiple=8, nodes_per_coil=Nl, nodes_np=True)
    # the certificate on the tubes: coarse, then 4x finer patches around the 16 largest differences
    X = np.concatenate(tubes)
    dB = np.linalg.norm(_b_any(lean, X) - _b_any(rel, X), axis=1)
    top = np.argsort(dB)[::-1][:16]
    fine = []
    for j in top:
        k = int(np.searchsorted(np.cumsum([len(t) for t in tubes]), j, side="right"))
        p, i, xs, dxs = samples[k]
        _, jj = cKDTree(xs).query(X[j])
        sl = (jj + np.arange(-8, 9)) % len(xs)
        t = dxs[sl] / np.linalg.norm(dxs[sl], axis=1)[:, None]
        n1, n2 = _frame(t)
        a2 = 2 * np.pi * np.arange(4 * TLEAN_AROUND) / (4 * TLEAN_AROUND)
        P = (xs[sl][:, None, :] + rho * (np.cos(a2)[None, :, None] * n1[:, None, :] +
                                         np.sin(a2)[None, :, None] * n2[:, None, :])).reshape(-1, 3)
        dd, jo = tcoil.query(P, k=1)
        fine.append(P[(owner[jo] == k) | (dd >= rho - h)])
    if fine:
        Xf = np.concatenate(fine)
        dB2 = np.linalg.norm(_b_any(lean, Xf) - _b_any(rel, Xf), axis=1)
    else:
        dB2 = np.zeros(0)
    d = float(max(np.max(dB), np.max(dB2) if len(dB2) else 0.0))
    info.update(cert=d / Bmin, tube_pts=int(len(X) + (len(Xf) if fine else 0)), s=time.time() - t0)
    if not (np.isfinite(d) and d <= TLEAN_CERT * Bmin):
        info["refused"] = "certificate"
        return None, info
    return LeanTraceEngine(rel, lean, tcoil, rho, h, info), info
