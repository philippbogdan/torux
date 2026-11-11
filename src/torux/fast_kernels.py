"""Kernels for the tracers under the certificates: the edge barrier's period maps and the island band's surrogate
node traces.  Switches.

The recommended set (bkern, ftrace) is on by default.  TORUX_FAST_KERNELS overrides it per process: a comma-separated
list of names, "all" for the recommended set, "every" for every switch, "none" for none.  In code:
``set_opts(...)`` / ``opts(...)``.  Independent of torux.fast_design .. fast_trace.

bkern and ftrace are the standard arithmetic, operation for operation and in the same order: the same bits.

  bkern   the B-only field of Fourier-filament nodes (fast_trace.field_B: every tracer's and section's B-only
          evaluation on the host) by a kernel that computes only B: torux.field.cpu's filament kernel computes 18 rows
          (B, A, grad B, the half-rule B) and drops 15 for a B-only call, and the 32-source blocks' Neumaier pass runs
          over all 18.  The same loop structure (16 targets per block, the target loop innermost: the form the
          compiler vectorises), the same terms in the same order (sources in blocks of SB = 32 in order, Neumaier
          over the blocks), then 0 + B as field_B sums into a zero array.  Segments and dipoles: the standard kernels.
  ftrace  torux.fasttrace.LineTracer.trace (every period map of torux.barrier.FieldSection and
          torux.islandband.PlaneSection, the barrier's scans, profiles and closed-line searches, the surrogates' node
          traces) as ONE compiled call per batch of lines: the Gragg-Bulirsch-Stoer macro steps of every line, all
          their stages, the Biot-Savart sums of the engine's filament nodes inline (bkern's 16-target blocks; a stage
          of at most FTRACE_SCALAR_NT targets one target at a time, the same lane arithmetic without a padded block),
          parallel over groups of G lines integrated in lock step (G = min(16, ceil(L / threads)): a stage of a group
          is G x (active sequences) targets).  The standard tracer makes one engine call per integrator stage for all
          lines (k^2 + 1 = 17 points per line and macro step in 2k = 8 calls at k = 4), with numpy glue around each,
          and small batches run far below the kernel's large-batch rate.  The same bits:
            * the field at a stage point: bkern's sum, then 0 + B (fast_trace.field_B);
            * the right-hand side, the modified-midpoint stages, the Aitken-Neville table and its error estimate as
              torux.fasttrace.gbs_step / LineTracer.rhs / LineTracer.trace write them (the same products, quotients and
              sums in the same order; no fused multiply-add: numba without fastmath);
            * the stage angles' cos and sin by numpy itself on the standard tracer's own angle values (phi accumulated
              by repeated addition of H, stage angles t + m (H / n_j)), passed to the kernel as a table (where numpy's
              cos / sin give an angle the same value at any position of any array: checked once at run time, else
              the standard tracer);
            * |B| = sqrt of the three squares summed in the order numpy's np.sum(B * B, -1) uses on this machine
              (measured at first use on random data; the switch stays off where neither order matches).
          A line is stopped where the standard tracer stops it (a stage with |B_phi| < frac |B|, the wrong sign,
          R <= 0, a non-finite field or state); its remaining stages in that macro step, which the standard tracer
          computes and discards, are skipped.  Deadlines: FieldSection's hard deadline is checked before each chunk of
          macro steps (chunks of <= ~2e7 interactions, a few ms) while a line is active, not at every stage.  Only
          host engines whose B-only field is the filament kernel (no card, segments, dipoles, JAX-only kinds or
          analytic parts; FieldSection's _B_engine or the tracer's own engine, never a separate field function) take
          this path; everything else runs the standard tracer.

Not bit-identical (a different surrogate, validated as the standard one validates its own):

  pnd     (experimental) the band surrogate's nodes across the annulus per patch (torux.islandband.BandSurrogate): the
          standard build sets one count nd = clip(8 (1 + ceil((1 + 2 margin) w / patch_len - 0.5)), 8, 32) for every
          patch from the widest patch's width w, so a band whose width varies (the chain's islands bulge it) traces
          its thin patches with up to 32 nodes across where the same rule applied to the patch's own width gives 8;
          here each patch k gets the rule at its own width w_k (the nodes per metre across stay the standard).  The
          coefficients of a patch with fewer nodes are zero-padded to the widest's (the evaluation sums exact zeros).
          Everything else is the standard build: the traces, the retraces, lost patches, the validation at random
          annulus points against the accurate tracer and tau = 3 x the largest check difference + 2 x the tracer's
          model of the patch's nodes and checks, the refusal above the bar.
"""
from __future__ import annotations

import contextlib
import math
import os
import time

import numpy as np

NAMES = ("bkern", "ftrace", "pnd")
# recommended: bkern and ftrace, bit-identical; pnd is experimental
EXPERIMENTAL = ("pnd",)
RECOMMENDED = tuple(n for n in NAMES if n not in EXPERIMENTAL)

FTRACE_CHUNK = 2e7          # 'ftrace': interactions per compiled call between deadline checks
FTRACE_GROUP_MAX = 16       # 'ftrace': at most this many lines integrated in lock step by one task
FTRACE_MIN_LINES = 1        # 'ftrace': smallest batch taking the compiled path
FTRACE_SCALAR_NT = 6        # 'ftrace': a stage of at most this many targets is summed one target at a time (no padding)


def _params_from_env():
    v = os.environ.get("TORUX_FAST_KERNELS_PARAMS")
    if not v:
        return
    g = globals()
    for item in v.split(","):
        k, _, x = item.partition("=")
        k = k.strip()
        if k not in ("FTRACE_CHUNK", "FTRACE_GROUP_MAX", "FTRACE_MIN_LINES", "FTRACE_SCALAR_NT"):
            raise ValueError(f"TORUX_FAST_KERNELS_PARAMS: unknown parameter {k}")
        g[k] = float(x) if isinstance(g[k], float) else int(x)


_params_from_env()


def _from_env():
    v = os.environ.get("TORUX_FAST_KERNELS")
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
        raise ValueError(f"TORUX_FAST_KERNELS: unknown switches {sorted(bad)} (known: {NAMES})")
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


# ------------------------------------------------------------------------------------------------------------ kernels
STATS = dict(calls=0, fallback=0, lines=0, evals=0, seconds=0.0, bkern=0)
_SUM3 = [None]          # the order of numpy's np.sum(B * B, -1) over 3 components: 0 = (a + b) + c, 1 = a + (b + c)


def sum3_order():
    """0 if np.sum(v, -1) over (n, 3) float64 equals (v0 + v1) + v2 bit for bit, 1 if v0 + (v1 + v2), -1 if neither
    (then 'ftrace' is not used)."""
    if _SUM3[0] is None:
        rng = np.random.default_rng(12345)
        ok0 = ok1 = True
        for n in (1, 2, 3, 5, 8, 17, 33, 100, 1000, 4097):
            v = rng.standard_normal((n, 3)) * np.exp(rng.uniform(-30, 30, (n, 3)))
            s = np.sum(v, -1)
            ok0 = ok0 and bool(np.array_equal(s, (v[:, 0] + v[:, 1]) + v[:, 2]))
            ok1 = ok1 and bool(np.array_equal(s, v[:, 0] + (v[:, 1] + v[:, 2])))
        _SUM3[0] = 0 if ok0 else 1 if ok1 else -1
    return _SUM3[0]


TB = 16                 # targets per block (torux.field.cpu.TB: the vectorised form)
SB = 32                 # sources per in-order block (torux.field.cpu.SB)

# the kernels at module level (numba caches them on disk; nested definitions were recompiled in every process)
from numba import njit, prange  # noqa: E402


@njit(cache=True, fastmath=False, inline="always")
def bblock(p0, p1, p2, r, w, acc, tot, cmp, Bb):
    """B (3, TB) of the filament nodes (r, w) at the TB targets (p0, p1, p2): torux.field.cpu's filament kernel,
    B rows only, the same terms in the same order (blocks of SB sources, Neumaier over the blocks), then the
    compensated total tot + cmp added to 0 (fast_trace.field_B sums the kernels into a zero array)."""
    S = r.shape[0]
    tot[:, :] = 0.0
    cmp[:, :] = 0.0
    for j0 in range(0, S, SB):
        acc[:, :] = 0.0
        for j in range(j0, min(S, j0 + SB)):
            r0 = r[j, 0]; r1 = r[j, 1]; r2_ = r[j, 2]
            w0 = w[j, 0]; w1 = w[j, 1]; w2 = w[j, 2]
            for t in range(TB):
                d0 = p0[t] - r0; d1 = p1[t] - r1; d2 = p2[t] - r2_
                q = d0 * d0 + d1 * d1 + d2 * d2
                ir = 1.0 / math.sqrt(q)
                ir3 = ir * ir * ir
                c0 = w1 * d2 - w2 * d1
                c1 = w2 * d0 - w0 * d2
                c2 = w0 * d1 - w1 * d0
                acc[0, t] += c0 * ir3; acc[1, t] += c1 * ir3; acc[2, t] += c2 * ir3
        for kk in range(3):
            for t in range(TB):
                s = tot[kk, t]; v = acc[kk, t]
                u = s + v
                if abs(s) >= abs(v):
                    cmp[kk, t] += (s - u) + v
                else:
                    cmp[kk, t] += (v - u) + s
                tot[kk, t] = u
    for kk in range(3):
        for t in range(TB):
            Bb[kk, t] = 0.0 + (tot[kk, t] + cmp[kk, t])

@njit(parallel=True, cache=True, fastmath=False)
def _fil_B_nb(xs, r, w, out):
    """'bkern': B (T, 3) at xs (3, T), parallel over blocks of TB targets (torux.field.cpu's layout)."""
    T = xs.shape[1]
    nblk = (T + TB - 1) // TB
    for b in prange(nblk):
        i0 = b * TB
        p0 = np.empty(TB); p1 = np.empty(TB); p2 = np.empty(TB)
        for t in range(TB):
            ii = min(i0 + t, T - 1)
            p0[t] = xs[0, ii]; p1[t] = xs[1, ii]; p2[t] = xs[2, ii]
        acc = np.empty((3, TB)); tot = np.empty((3, TB)); cmp = np.empty((3, TB)); Bb = np.empty((3, TB))
        bblock(p0, p1, p2, r, w, acc, tot, cmp, Bb)
        for t in range(TB):
            ii = i0 + t
            if ii < T:
                out[ii, 0] = Bb[0, t]; out[ii, 1] = Bb[1, t]; out[ii, 2] = Bb[2, t]

@njit(cache=True, fastmath=False, inline="always")
def bscalar(nt, X, Y, Z, BX, BY, BZ, r, w):
    """B at nt targets one at a time: each target's terms, block sums and Neumaier steps are bblock's lane
    arithmetic in the same order (the same bits), without the padding of a 16-lane block (for a stage of a few
    targets the vector block computes 16 lanes; scalar code was measured at 0.3-0.6 of a lane's rate)."""
    S = r.shape[0]
    for t in range(nt):
        px = X[t]; py = Y[t]; pz = Z[t]
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        e0 = 0.0; e1 = 0.0; e2 = 0.0
        for j0 in range(0, S, SB):
            a0 = 0.0; a1 = 0.0; a2 = 0.0
            for j in range(j0, min(S, j0 + SB)):
                r0 = r[j, 0]; r1 = r[j, 1]; r2_ = r[j, 2]
                w0 = w[j, 0]; w1 = w[j, 1]; w2 = w[j, 2]
                d0 = px - r0; d1 = py - r1; d2 = pz - r2_
                q = d0 * d0 + d1 * d1 + d2 * d2
                ir = 1.0 / math.sqrt(q)
                ir3 = ir * ir * ir
                c0 = w1 * d2 - w2 * d1
                c1 = w2 * d0 - w0 * d2
                c2 = w0 * d1 - w1 * d0
                a0 += c0 * ir3; a1 += c1 * ir3; a2 += c2 * ir3
            u = s0 + a0
            if abs(s0) >= abs(a0):
                e0 += (s0 - u) + a0
            else:
                e0 += (a0 - u) + s0
            s0 = u
            u = s1 + a1
            if abs(s1) >= abs(a1):
                e1 += (s1 - u) + a1
            else:
                e1 += (a1 - u) + s1
            s1 = u
            u = s2 + a2
            if abs(s2) >= abs(a2):
                e2 += (s2 - u) + a2
            else:
                e2 += (a2 - u) + s2
            s2 = u
        BX[t] = 0.0 + (s0 + e0); BY[t] = 0.0 + (s1 + e1); BZ[t] = 0.0 + (s2 + e2)


@njit(cache=True, fastmath=False, inline="always")
def eval_targets(nt, X, Y, Z, BX, BY, BZ, r, w, p0, p1, p2, acc, tot, cmp, Bb, nsc):
    """B at nt targets: one at a time when nt <= nsc, else in blocks of TB (the last block padded with its last
    target, as torux.field.cpu pads)."""
    if nt <= nsc:
        bscalar(nt, X, Y, Z, BX, BY, BZ, r, w)
        return
    for b0 in range(0, nt, TB):
        for t in range(TB):
            ii = min(b0 + t, nt - 1)
            p0[t] = X[ii]; p1[t] = Y[ii]; p2[t] = Z[ii]
        bblock(p0, p1, p2, r, w, acc, tot, cmp, Bb)
        for t in range(TB):
            ii = b0 + t
            if ii < nt:
                BX[ii] = Bb[0, t]; BY[ii] = Bb[1, t]; BZ[ii] = Bb[2, t]

@njit(parallel=True, cache=True, fastmath=False)
def gbs(y, err, lost, steps, grp, CS, SN, H, i0, i1, k, nsf, rtab, sign, frac, r, w, rec, G, order, nev, nsc):
    """Macro steps i0 .. i1 - 1 of LineTracer.trace for every line not lost (gbs_step's scheme, its arithmetic).
    y (L, 2), err (L,), lost (L,) bool, steps (L,) int64: the state, updated in place.  grp (L,): row of the
    angle tables CS, SN (U, nsteps, k^2 + 1): cos / sin of [t_i, then t_i + m H / n_j for m = 1 .. n_k - 1,
    j = m // 2 .. k - 1] (numpy's values).  H (L,).  nsf (k,) = 2 j (float), rtab (k, k) the Neville
    divisors.  rec (nrec, L, 2): states after each step when nrec > 0.  G lines per task in lock step.
    nev (ntask,): evaluations made, per task.  nsc: a stage of at most nsc targets is summed one target at a time."""
    L = y.shape[0]
    ntask = (L + G - 1) // G
    nmax = G * k
    nrec = rec.shape[0]
    for g in prange(ntask):
        l0 = g * G
        l1 = min(L, l0 + G)
        ng = l1 - l0
        X = np.empty(nmax); Y = np.empty(nmax); Zt = np.empty(nmax)
        BX = np.empty(nmax); BY = np.empty(nmax); BZ = np.empty(nmax)
        p0 = np.empty(TB); p1 = np.empty(TB); p2 = np.empty(TB)
        acc = np.empty((3, TB)); tot = np.empty((3, TB)); cmp = np.empty((3, TB)); Bb = np.empty((3, TB))
        tl = np.empty(nmax, np.int64)          # target -> line slot
        tj = np.empty(nmax, np.int64)          # target -> sequence
        zpR = np.empty((G, k)); zpZ = np.empty((G, k))
        zcR = np.empty((G, k)); zcZ = np.empty((G, k))
        TR = np.empty((G, k)); TZ = np.empty((G, k))
        hs = np.empty((G, k))
        act = np.empty(G, np.bool_)
        rowR = np.empty(k); rowZ = np.empty(k); tbR = np.empty(k); tbZ = np.empty(k)
        ne = 0
        for i in range(i0, i1):
            nact = 0
            for a in range(ng):
                l = l0 + a
                act[a] = not lost[l]
                if act[a]:
                    nact += 1
            if nact == 0:
                if nrec > 0:
                    for a in range(ng):
                        rec[i, l0 + a, 0] = y[l0 + a, 0]; rec[i, l0 + a, 1] = y[l0 + a, 1]
                continue
            ne += nact * (k * k + 1)
            # stage 0: f0 at (t_i, y)
            nt = 0
            for a in range(ng):
                if act[a]:
                    l = l0 + a
                    u = grp[l]
                    c = CS[u, i, 0]; s = SN[u, i, 0]
                    R = y[l, 0]
                    X[nt] = R * c; Y[nt] = R * s; Zt[nt] = y[l, 1]
                    tl[nt] = a
                    nt += 1
            eval_targets(nt, X, Y, Zt, BX, BY, BZ, r, w, p0, p1, p2, acc, tot, cmp, Bb, nsc)
            for tt in range(nt):
                a = tl[tt]
                l = l0 + a
                u = grp[l]
                c = CS[u, i, 0]; s = SN[u, i, 0]
                R = y[l, 0]; Z = y[l, 1]
                b0 = BX[tt]; b1 = BY[tt]; b2 = BZ[tt]
                BR = b0 * c + b1 * s
                Bp = -b0 * s + b1 * c
                q0 = b0 * b0; q1 = b1 * b1; q2 = b2 * b2
                if order == 0:
                    Bm = math.sqrt((q0 + q1) + q2)
                else:
                    Bm = math.sqrt(q0 + (q1 + q2))
                ok = (math.isfinite(Bm) and R > 0 and sign * Bp >= frac * Bm and math.isfinite(R)
                      and math.isfinite(Z))
                if not ok:
                    act[a] = False
                    lost[l] = True
                    continue
                fR = R * BR / Bp
                fZ = R * b2 / Bp
                Hl = H[l]
                for j in range(k):
                    h = Hl / nsf[j]
                    hs[a, j] = h
                    zpR[a, j] = R; zpZ[a, j] = Z
                    zcR[a, j] = R + h * fR
                    zcZ[a, j] = Z + h * fZ
            # substeps m = 1 .. n_k - 1: sequences j with n_j > m; off = the table column of (m, jmin)
            nlast = 2 * k
            off = 1
            for m in range(1, nlast):
                jmin = m // 2
                if m > 1:
                    off += k - (m - 1) // 2
                nt = 0
                for a in range(ng):
                    if act[a]:
                        l = l0 + a
                        u = grp[l]
                        for j in range(jmin, k):
                            cc = CS[u, i, off + j - jmin]; ss = SN[u, i, off + j - jmin]
                            R = zcR[a, j]
                            X[nt] = R * cc; Y[nt] = R * ss; Zt[nt] = zcZ[a, j]
                            tl[nt] = a
                            tj[nt] = j
                            nt += 1
                if nt == 0:
                    break
                eval_targets(nt, X, Y, Zt, BX, BY, BZ, r, w, p0, p1, p2, acc, tot, cmp, Bb, nsc)
                for tt in range(nt):
                    a = tl[tt]; j = tj[tt]
                    if not act[a]:
                        continue
                    l = l0 + a
                    u = grp[l]
                    cc = CS[u, i, off + j - jmin]; ss = SN[u, i, off + j - jmin]
                    R = zcR[a, j]; Z = zcZ[a, j]
                    b0 = BX[tt]; b1 = BY[tt]; b2 = BZ[tt]
                    BR = b0 * cc + b1 * ss
                    Bp = -b0 * ss + b1 * cc
                    q0 = b0 * b0; q1 = b1 * b1; q2 = b2 * b2
                    if order == 0:
                        Bm = math.sqrt((q0 + q1) + q2)
                    else:
                        Bm = math.sqrt(q0 + (q1 + q2))
                    ok = (math.isfinite(Bm) and R > 0 and sign * Bp >= frac * Bm and math.isfinite(R)
                          and math.isfinite(Z))
                    if not ok:
                        act[a] = False
                        lost[l] = True
                        continue
                    fR = R * BR / Bp
                    fZ = R * b2 / Bp
                    h2 = 2.0 * hs[a, j]
                    nR = zpR[a, j] + h2 * fR
                    nZ = zpZ[a, j] + h2 * fZ
                    zpR[a, j] = R; zpZ[a, j] = Z
                    zcR[a, j] = nR; zcZ[a, j] = nZ
                    if m + 1 == 2 * (j + 1):
                        TR[a, j] = nR; TZ[a, j] = nZ
            # Aitken-Neville in h^2 (torux.fasttrace._neville) and the line's update (LineTracer.trace)
            for a in range(ng):
                if not act[a]:
                    continue
                l = l0 + a
                tbR[0] = TR[a, 0]; tbZ[0] = TZ[a, 0]
                for j in range(1, k):
                    rowR[0] = TR[a, j]; rowZ[0] = TZ[a, j]
                    for kk in range(1, j + 1):
                        rr = rtab[j, kk]
                        rowR[kk] = rowR[kk - 1] + (rowR[kk - 1] - tbR[kk - 1]) / rr
                        rowZ[kk] = rowZ[kk - 1] + (rowZ[kk - 1] - tbZ[kk - 1]) / rr
                    for kk in range(j + 1):
                        tbR[kk] = rowR[kk]; tbZ[kk] = rowZ[kk]
                yR = tbR[k - 1]; yZ = tbZ[k - 1]
                if k > 1:
                    eR = abs(tbR[k - 1] - tbR[k - 2])
                    eZ = abs(tbZ[k - 1] - tbZ[k - 2])
                    e = eR if eR >= eZ else eZ
                    if eR != eR or eZ != eZ:
                        e = np.nan
                else:
                    e = 0.0
                if not (math.isfinite(yR) and math.isfinite(yZ)):
                    lost[l] = True
                    continue
                y[l, 0] = yR; y[l, 1] = yZ
                err[l] = err[l] + e
                steps[l] += 1
            if nrec > 0:
                for a in range(ng):
                    rec[i, l0 + a, 0] = y[l0 + a, 0]; rec[i, l0 + a, 1] = y[l0 + a, 1]
        nev[g] = ne



_CHECKED = []


def _kernels():
    if not _CHECKED:
        from .field import cpu as _cpu
        assert _cpu.TB == TB and _cpu.SB == SB, "fast_kernels: torux.field.cpu's block sizes changed"
        _CHECKED.append(True)
    return dict(fil_B=_fil_B_nb, gbs=gbs)


def fil_B(x, r, w):
    """'bkern': B (T, 3) of the filament nodes (r, w) at x (T, 3)."""
    xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1, 3).T)
    out = np.empty((xs.shape[1], 3))
    if xs.shape[1]:
        _kernels()["fil_B"](xs, np.ascontiguousarray(r, dtype=np.float64), np.ascontiguousarray(w, dtype=np.float64), out)
    STATS["bkern"] += 1
    return out


# ------------------------------------------------------------------------------------------------------------ ftrace
def _eligible(tracer):
    """(engine, deadline check) when the tracer's field is a host FieldEngine's filament kernel, else None."""
    from . import fast_trace as _so4
    f = tracer._field
    check = None
    if f is None:
        eng = tracer.engine
    else:
        sec = getattr(f, "__self__", None)
        from .barrier import FieldSection
        if not isinstance(sec, FieldSection) or getattr(f, "__func__", None) is not FieldSection._B_engine:
            return None
        eng = sec.engine
        check = sec.check_deadline
    if isinstance(eng, _so4.LeanTraceEngine) or not _so4._host_plain(eng):
        return None
    P = eng._np
    if len(P["r"]) == 0 or len(P["sa"]) or len(P["dp"]):
        return None
    return eng, check


def _angle_tables(phis, Hs, nsteps, k):
    """cos / sin (U, nsteps, k^2 + 1) of the standard stage angles for each (phi0, H) pair, computed as
    LineTracer.trace / gbs_step compute them (phi accumulated by repeated addition; stage t + m * (H / n_j))."""
    U = len(phis)
    ns = [2 * j for j in range(1, k + 1)]
    T = np.empty((nsteps + 1, U))
    T[0] = phis
    for i in range(nsteps):
        T[i + 1] = T[i] + Hs
    hs = [Hs / n for n in ns]
    cols = [T[:nsteps]]
    for m in range(1, ns[-1]):
        for j in range(k):
            if ns[j] > m:
                cols.append(T[:nsteps] + m * hs[j])
    A = np.ascontiguousarray(np.stack(cols, -1).transpose(1, 0, 2))      # (U, nsteps, ncol)
    return np.cos(A), np.sin(A), T


_TRIG = [None]


def trig_position_free():
    """True if numpy's float64 cos and sin give an angle the same value wherever it stands in an array and whatever
    the array's length (ftrace computes the standard stage angles' cos / sin in a table of another layout; numpy's
    SIMD loops could treat a vector's tail otherwise).  Measured once; ftrace is not used where it fails."""
    if _TRIG[0] is None:
        rng = np.random.default_rng(7)
        ang = rng.uniform(-7.0, 7.0, 4099)
        ok = True
        for fn in (np.cos, np.sin):
            full = fn(ang)
            for off in (0, 1, 3, 5, 7, 13):
                for n in (1, 2, 3, 4, 7, 8, 9, 16, 17, 33, 100):
                    ok = ok and bool(np.array_equal(fn(ang[off:off + n]), full[off:off + n]))
            ok = ok and bool(np.array_equal(fn(ang[::-1])[::-1], full))
        _TRIG[0] = ok
    return _TRIG[0]


def fused_trace(tracer, y0, phi0, dphi, nsteps=None, record=None):
    """LineTracer.trace by the compiled kernel (module docstring), or None where the path does not apply."""
    if _SUM3[0] is None:
        sum3_order()
    if _SUM3[0] < 0 or not trig_position_free():
        return None
    el = _eligible(tracer)
    if el is None:
        STATS["fallback"] += 1
        return None
    eng, check = el
    t_start = time.perf_counter()
    y = np.array(y0, dtype=float).reshape(-1, 2)
    L = y.shape[0]
    if L < FTRACE_MIN_LINES:
        return None
    phi = np.broadcast_to(np.asarray(phi0, dtype=float), (L,)).copy()
    dphi = np.broadcast_to(np.asarray(dphi, dtype=float), (L,)).copy()
    if nsteps is None:
        nsteps = max(1, int(math.ceil(np.max(np.abs(dphi)) / tracer.period * tracer.sp - 1e-9)))
    nsteps = int(nsteps)
    H = dphi / nsteps
    k = int(tracer.k)
    err = np.zeros(L)
    lost = ~np.all(np.isfinite(y), -1)
    steps = np.zeros(L, np.int64)
    # one angle table per distinct (phi0, H)
    key = np.ascontiguousarray(np.stack([phi, H], -1))
    _, first, grp = np.unique(key.view(np.dtype((np.void, 16))).ravel(), return_index=True, return_inverse=True)
    uniq = key[first]
    grp = np.ascontiguousarray(grp.reshape(-1).astype(np.int64))
    CS, SN, T = _angle_tables(uniq[:, 0].copy(), uniq[:, 1].copy(), nsteps, k)
    ns = [2 * j for j in range(1, k + 1)]
    rtab = np.ones((k, k))
    for j in range(1, k):
        for kk in range(1, j + 1):
            rtab[j, kk] = (ns[j] / ns[j - kk]) ** 2 - 1.0
    nsf = np.array([float(n) for n in ns])
    P = eng._np
    r = np.ascontiguousarray(P["r"], dtype=np.float64)
    w = np.ascontiguousarray(P["w"], dtype=np.float64)
    import numba
    nth = max(1, numba.get_num_threads())
    G = int(max(1, min(FTRACE_GROUP_MAX, -(-L // nth))))
    ntask = (L + G - 1) // G
    rec = np.empty((nsteps if record else 0, L, 2))
    gbs = _kernels()["gbs"]
    per_step = max(1.0, float(L) * (k * k + 1) * len(r))
    chunk = int(max(1, min(nsteps, FTRACE_CHUNK // per_step)))
    nev = np.zeros(ntask, np.int64)
    ev = 0
    i = 0
    while i < nsteps:
        j = min(nsteps, i + chunk)
        if not np.all(lost):
            if check is not None:
                check()
            gbs(y, err, lost, steps, grp, CS, SN, H, i, j, k, nsf, rtab, float(tracer.sign), float(tracer.frac), r, w,
                rec, G, _SUM3[0], nev, FTRACE_SCALAR_NT)
            ev += int(nev.sum())
        elif record:
            rec[i:j] = y[None]
        i = j
    phi_out = T[steps, grp]
    tracer.n_calls += 1
    tracer.n_eval += ev
    STATS["calls"] += 1
    STATS["lines"] += L
    STATS["evals"] += ev
    STATS["seconds"] += time.perf_counter() - t_start
    out = dict(y=y, err=err, lost=lost, phi=phi_out, steps=steps.astype(int))
    if record:
        out["rec"] = [rec[i_].copy() for i_ in range(nsteps)]
    return out


# ------------------------------------------------------------------------------------------------------------ pnd
def band_nd_per_patch(sur, nd_rule):
    """'pnd': the standard across-count rule at each patch's own width (BandSurrogate.__init__'s ann_lo / ann_hi,
    neighbours included); patches without a width (no torus sample) get the rule's minimum."""
    w = np.asarray(sur.ann_hi, float) - np.asarray(sur.ann_lo, float)
    ratio = (1.0 + 2.0 * sur.margin) * w / sur.patch_len
    ndk = np.where(np.isfinite(ratio), np.clip(8 * (1 + np.ceil(np.maximum(ratio - 0.5, 0.0))), 8, 32), 8)
    ndk = np.minimum(ndk.astype(np.int64), int(nd_rule))
    return ndk


def band_build_pnd(sur, verbose=False, deadline=None):
    """BandSurrogate.build with sur.nd_k nodes across patch k (module docstring); the standard build otherwise."""
    from . import islandband as IB
    t0 = time.time()
    npch, nt = sur.np_, sur.nt
    ndk = np.asarray(sur.nd_k, np.int64)
    ndmax = int(sur.nd)
    ut = IB._cheb_nodes(nt)
    groups, Xl = [], []
    for g in sorted(set(int(v) for v in ndk)):
        idx = np.flatnonzero(ndk == g)
        vd = IB._cheb_nodes(g)
        TH = sur.tb[idx][:, None, None] + 0.5 * (ut[None, :, None] + 1.0) * (sur.tb[idx + 1] - sur.tb[idx])[:, None, None]
        DD = sur.d_lo[idx][:, None, None] + 0.5 * (vd[None, None, :] + 1.0) * (sur.d_hi[idx] - sur.d_lo[idx])[:, None, None]
        TH, DD = np.broadcast_arrays(TH, DD)
        Xl.append(sur.ref.point(TH.ravel(), DD.ravel()))
        groups.append((g, idx))
    X = np.concatenate(Xl)
    if verbose:
        print(f"   band surrogate (pnd): {npch} patches of {sur.patch_len:.2e} m x {np.median(sur.d_hi - sur.d_lo):.2e} m "
              f"(median), {len(X)} nodes (release {npch * nt * ndmax}) ({time.time() - t0:.0f}s)", flush=True)
    Y, lost, tau, retraced = sur._trace_nodes(X, deadline=deadline)
    off = 0
    bad = []
    spans = []
    for g, idx in groups:
        n = len(idx) * nt * g
        spans.append((g, idx, slice(off, off + n)))
        lo_g = lost[off:off + n].reshape(len(idx), nt * g).any(1)
        bad.extend(idx[lo_g].tolist())
        off += n
    bad = np.unique(np.asarray(bad, np.int64))
    if len(bad) > 0.5 * npch:
        raise RuntimeError(f"{len(bad)} of {npch} surrogate patches have lost nodes")
    sur.active[bad] = False
    Y[lost] = X[lost]
    tau[lost] = 0.0
    Bp = np.concatenate([sur.sign * sur._bphi(sur.B(sur.xyz(X[k:k + sur.chunk])))
                         for k in range(0, len(X), sur.chunk)])
    coef = np.zeros((3, npch, nt, ndmax))
    sur.tau_patch = np.zeros(npch)
    tail = 0.0
    for g, idx, sl in spans:
        V = np.stack([Y[sl, 0], Y[sl, 1], Bp[sl]], 0).reshape(3, len(idx), nt, g)
        C = IB._cheb_coef2r(V)
        coef[:, idx, :, :g] = C
        sur.tau_patch[idx] = np.max(tau[sl].reshape(len(idx), nt * g), axis=1)
        a_ = sur.active[idx]
        if np.any(a_):
            tail = max(tail, float(np.max(np.abs(C[:2, a_, -2:, :]))), float(np.max(np.abs(C[:2, a_, :, -2:]))))
    sur.coef = coef
    # validation at random points of the annulus (each patch's own normal range): the standard
    rng = np.random.default_rng(sur.seed)
    k = rng.integers(0, npch, size=sur.n_check)
    k = k[sur.active[k]]
    tq = sur.tb[k] + rng.uniform(0, 1, len(k)) * (sur.tb[k + 1] - sur.tb[k])
    dq = sur.ann_lo[k] + rng.uniform(0, 1, len(k)) * (sur.ann_hi[k] - sur.ann_lo[k])
    cand = sur.ref.point(tq, dq)
    Yc, lc, tauc, _ = sur._trace_nodes(cand, deadline=deadline, gpu=False)
    Ys, ls, _ = sur.step(cand, np.ones(len(cand)), _raw=True)
    dist = np.linalg.norm(Ys - Yc, axis=1)
    good = ~(lc | ls)
    err = float(np.max(dist[good])) if np.any(good) else np.inf
    if not np.any(good):
        raise RuntimeError("no surrogate check point could be evaluated")
    db = sur.density(cand[good]) / (sur.sign * sur._bphi(sur.B(sur.xyz(cand[good])))) - 1.0
    sur.density_rel_err = float(np.max(np.abs(db)))
    tp = sur.tau_patch.copy()
    kc = sur._locate(cand)[0]
    np.maximum.at(tp, kc[good], tauc[good])
    sur.tau_cell = 3.0 * err + 2.0 * tp
    sur.tau_surr = float(np.max(sur.tau_cell[sur.active]))
    sur.tau_annulus = sur.tau_surr
    sur.info = dict(kind="band", patches=int(npch), patch_len=sur.patch_len, nt=nt, nd=ndmax, nodes=int(len(X)),
                    m_ref=getattr(sur, "m_ref", None),
                    width_median=float(np.median(sur.ann_hi - sur.ann_lo)), retraced=int(retraced),
                    lost_patches=int(len(bad)), reach_ok=bool(sur.reach_ok), n_check=int(np.sum(good)),
                    lost_checks=int(np.sum(ls & ~lc)), max_check_diff=err, tau_samples=float(np.max(tau)),
                    tau=sur.tau_surr, tau_annulus=sur.tau_annulus, coef_tail=tail,
                    density_rel_err=sur.density_rel_err, build_time=time.time() - t0,
                    gpu=getattr(sur.exact, "gpu_stats", None), first_sp=getattr(sur, "first_sp", 24),
                    pnd=dict(nodes_standard=int(npch * nt * ndmax), nd_counts={int(g): int(len(i)) for g, i in groups}))
    if verbose:
        print(f"   band surrogate (pnd): check max diff {err:.2e} ({int(np.sum(good))} points), tau {sur.tau_surr:.2e}, "
              f"tail {tail:.1e}, B_phi rel {sur.density_rel_err:.1e}, lost patches {len(bad)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return sur.info
