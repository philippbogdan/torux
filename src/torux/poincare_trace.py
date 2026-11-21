"""The tracer of torux.poincare: many field lines at once through planes of constant toroidal angle.

The toroidal angle phi is the independent variable, dR/dphi = R B_R / B_phi and dZ/dphi = R B_Z / B_phi, integrated by
Gragg-Bulirsch-Stoer extrapolation over fixed steps that end on every plane: the modified midpoint rule with 2, 4, ...,
2k substeps and Aitken-Neville extrapolation in h^2 (order 2k), as torux.fasttrace writes it, with two changes that
keep a long trace at rounding.  The substeps and the extrapolation work on the step's increment, not on the position
(a number a few hundred times smaller, so its rounding is too), and the increments are added to the position with
Kahan's compensated sum, so the rounding of a thousand turns does not walk.

A line stops where it cannot be followed in phi (|B_phi| < 1e-3 |B| or B_phi of the other sign, where a line turns
back next to a coil), where it leaves the box round the source, or where it comes nearer a conductor than the
distance down to which the field is computed to rounding (FieldEngine's dmin).  Where it stopped is kept.

Along the way each line's angle about the magnetic axis is followed at every step (the axis given at every step's
end), so that its turns about the axis are counted however far it moves in one field period.

Coils, magnets and the 1/R field go through one compiled kernel on the engine's own nodes, segments and dipoles.  A
coil far from a line is summed on every 2nd to 16th of its nodes where that rule was measured, round the magnetic
axis, within STRIDE_TOL of the field's strength of the rule on all of them (_field, _reach): the lines' field is the
engine's to 1e-14.  Rings and axisymmetric conductors go through the same scheme in numpy on the engine's field.
The kernel is called for about CHUNK_S seconds of periods at a time, and Ctrl-C between two calls ends the trace in a
KeyboardInterrupt (torux.interrupt).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

TB = 16                     # targets summed together, one per lane
SB = 32                     # sources summed plainly before a compensated step (torux.field.cpu's blocks)
LEVELS = 5                  # a coil may be summed on every node, every 2nd, .. every 16th
NS = 24                     # points along each coil its distance from a target is bounded by
STRIDE_TOL = 1e-15          # a coil's coarser rule is taken where it was measured within this share of the field's
                            # strength of its rule on every node: the lines' field stays within 1e-14 of the engine's
                            # (tests/test_poincare.py measures it)

K = 5                       # sequences of the extrapolation: order 10 (more amplify the rounding more)
FRAC = 1e-3                 # a line stops where |B_phi| < FRAC |B| (torux.fasttrace's convention)
GROUP_MAX = 8               # lines integrated in lock step by one task of the kernel
CHUNK_S = 0.5               # seconds of the kernel per call, between which Ctrl-C is looked for (torux.interrupt)


def neville_divisors(k=K):
    ns = [2 * j for j in range(1, k + 1)]
    rtab = np.ones((k, k))
    for j in range(1, k):
        for kk in range(1, j + 1):
            rtab[j, kk] = (ns[j] / ns[j - kk]) ** 2 - 1.0
    return np.array(ns, float), rtab


class Schedule:
    """The steps of one field period from the first plane: each step's length and where it starts in the period,
    the plane each step ends on (-1 for none), and the magnetic axis (R, Z) at each step's end (NaN without one).

    planes: angles of the planes in [0, period) relative to the first (0 first). steps: at least this many steps per
    period, every stretch between two planes cut into equal steps."""

    def __init__(self, planes_rel, period, steps, axis=None, phi0=0.0):
        cuts = list(planes_rel) + [period]
        h, off, plane = [], [], []
        for j in range(len(planes_rel)):
            span = cuts[j + 1] - cuts[j]
            n = max(1, int(math.ceil(span / period * steps - 1e-9)))
            for i in range(n):
                h.append(span / n)
                off.append(cuts[j] + i * span / n)
                plane.append(j + 1 if i == n - 1 else -1)
        plane[-1] = 0                                   # the period ends on the first plane
        self.h = np.array(h)
        self.off = np.array(off)
        self.plane = np.array(plane, np.int64)
        self.period = float(period)
        self.n = len(h)
        ends = phi0 + self.off + self.h
        if axis is None:
            self.axR = np.full(self.n, np.nan)
            self.axZ = np.full(self.n, np.nan)
        else:
            self.axR, self.axZ = (np.asarray(v, float) for v in axis(ends))


@njit(cache=True, fastmath=False)
def _field(nt, X, Y, Z, BX, BY, BZ, r, w, lv, ci, smp, reach, p0, p1, p2, acc, tot, cmp, ext):
    """B at nt targets: for each, every coil's filament nodes TB at a time, one node a lane (so a stage of a few
    targets keeps every lane busy), the coils added with Neumaier steps; then the segments, the dipoles and the 1/R
    field of ext = (segment starts, ends, weights, dipole positions, moments, R B_phi).

    A coil far from a target is summed on every m-th node (m = 2^L) weighted m: the trapezoid rule on N / m nodes of
    the same closed curve, at the largest L whose reach[c, L] the target's distance from the coil (bounded below by
    the coil's bounding sphere, or by its NS points less half their spacing) is past: the distance beyond which that
    rule was measured within STRIDE_TOL of the field's strength of the rule on every node (_reach).  r, w (3, T): the
    nodes and weights of every coil at every stride, each run padded to a whole number of TB with nodes of no weight
    far away.  lv (LEVELS, C, 2): where coil c's run at stride 2^L starts and how many nodes it has (0: no such
    stride).  ci (C, 6): unused, the bounding sphere's centre (3) and radius, half the spacing of smp (C, NS, 3),
    points along each coil."""
    sa, sb, sw, dp, dm, RB = ext
    C = ci.shape[0]
    a0 = np.empty(TB); a1 = np.empty(TB); a2 = np.empty(TB)
    for ii in range(nt):
        px = X[ii]; py = Y[ii]; pz = Z[ii]
        t0 = 0.0; t1 = 0.0; t2 = 0.0
        c0 = 0.0; c1 = 0.0; c2 = 0.0
        for c in range(C):
            top = LEVELS - 1
            while top > 0 and lv[top, c, 1] == 0:
                top -= 1
            L = 0
            if top > 0:
                # a lower bound of the target's distance from the coil: its bounding sphere, then its points
                e0 = px - ci[c, 1]; e1 = py - ci[c, 2]; e2 = pz - ci[c, 3]
                d = math.sqrt(e0 * e0 + e1 * e1 + e2 * e2) - ci[c, 4]
                if d < reach[c, top]:
                    q = 1e300
                    for k in range(smp.shape[1]):
                        e0 = px - smp[c, k, 0]; e1 = py - smp[c, k, 1]; e2 = pz - smp[c, k, 2]
                        v = e0 * e0 + e1 * e1 + e2 * e2
                        if v < q:
                            q = v
                    v = math.sqrt(q) - ci[c, 5]
                    if v > d:
                        d = v
                L = top
                while L > 0 and d < reach[c, L]:
                    L -= 1
            j0 = lv[L, c, 0]
            n = lv[L, c, 1]
            for lane in range(TB):
                a0[lane] = 0.0; a1[lane] = 0.0; a2[lane] = 0.0
            for jj in range(j0, j0 + n, TB):
                for lane in range(TB):
                    j = jj + lane
                    d0 = px - r[0, j]; d1 = py - r[1, j]; d2 = pz - r[2, j]
                    ir = 1.0 / math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                    ir3 = ir * ir * ir
                    a0[lane] += (w[1, j] * d2 - w[2, j] * d1) * ir3
                    a1[lane] += (w[2, j] * d0 - w[0, j] * d2) * ir3
                    a2[lane] += (w[0, j] * d1 - w[1, j] * d0) * ir3
            v0 = 0.0; v1 = 0.0; v2 = 0.0
            for lane in range(TB):
                v0 += a0[lane]; v1 += a1[lane]; v2 += a2[lane]
            u = t0 + v0
            c0 += ((t0 - u) + v0) if abs(t0) >= abs(v0) else ((v0 - u) + t0)
            t0 = u
            u = t1 + v1
            c1 += ((t1 - u) + v1) if abs(t1) >= abs(v1) else ((v1 - u) + t1)
            t1 = u
            u = t2 + v2
            c2 += ((t2 - u) + v2) if abs(t2) >= abs(v2) else ((v2 - u) + t2)
            t2 = u
        BX[ii] = t0 + c0; BY[ii] = t1 + c1; BZ[ii] = t2 + c2
    if sa.shape[0] or dp.shape[0]:
        for b0 in range(0, nt, TB):
            for t in range(TB):
                k = min(b0 + t, nt - 1)
                p0[t] = X[k]; p1[t] = Y[k]; p2[t] = Z[k]
            tot[:, :] = 0.0
            cmp[:, :] = 0.0
            _segments(sa, sb, sw, p0, p1, p2, acc, tot, cmp)
            _dipoles(dp, dm, p0, p1, p2, acc, tot, cmp)
            for t in range(TB):
                k = b0 + t
                if k < nt:
                    BX[k] += tot[0, t] + cmp[0, t]; BY[k] += tot[1, t] + cmp[1, t]; BZ[k] += tot[2, t] + cmp[2, t]
    if RB != 0.0:
        for k in range(nt):
            q = RB / (X[k] * X[k] + Y[k] * Y[k])
            BX[k] += -Y[k] * q
            BY[k] += X[k] * q


@njit(cache=True, fastmath=False, inline="always")
def _neumaier(acc, tot, cmp):
    for kk in range(3):
        for t in range(TB):
            s0 = tot[kk, t]; v = acc[kk, t]
            u = s0 + v
            if abs(s0) >= abs(v):
                cmp[kk, t] += (s0 - u) + v
            else:
                cmp[kk, t] += (v - u) + s0
            tot[kk, t] = u


@njit(cache=True, fastmath=False)
def _segments(a, bb, cw, p0, p1, p2, acc, tot, cmp):
    """Straight segments a -> b, exactly, as torux.field.cpu writes them (its B rows), into tot / cmp."""
    D = a.shape[0]
    for j0 in range(0, D, SB):
        acc[:, :] = 0.0
        for j in range(j0, min(D, j0 + SB)):
            a0 = a[j, 0]; a1 = a[j, 1]; a2 = a[j, 2]
            l0 = bb[j, 0] - a0; l1 = bb[j, 1] - a1; l2 = bb[j, 2] - a2
            c = cw[j]
            for t in range(TB):
                i0_ = p0[t] - a0; i1_ = p1[t] - a1; i2_ = p2[t] - a2
                f0 = i0_ - l0; f1 = i1_ - l1; f2 = i2_ - l2
                ri = math.sqrt(i0_ * i0_ + i1_ * i1_ + i2_ * i2_)
                rf = math.sqrt(f0 * f0 + f1 * f1 + f2 * f2)
                iri = 1.0 / ri; irf = 1.0 / rf
                s0 = i0_ * iri + f0 * irf; s1 = i1_ * iri + f1 * irf; s2 = i2_ * iri + f2 * irf
                q = ri * rf * (s0 * s0 + s1 * s1 + s2 * s2)
                fac = c * 2.0 * (ri + rf) / (ri * rf * q)
                acc[0, t] += fac * (l1 * i2_ - l2 * i1_)
                acc[1, t] += fac * (l2 * i0_ - l0 * i2_)
                acc[2, t] += fac * (l0 * i1_ - l1 * i0_)
        _neumaier(acc, tot, cmp)


@njit(cache=True, fastmath=False)
def _dipoles(p, m, p0, p1, p2, acc, tot, cmp):
    """Point dipoles, as torux.field.cpu writes them (its B rows), into tot / cmp."""
    D = p.shape[0]
    for j0 in range(0, D, SB):
        acc[:, :] = 0.0
        for j in range(j0, min(D, j0 + SB)):
            q0 = p[j, 0]; q1 = p[j, 1]; q2 = p[j, 2]
            m0 = m[j, 0]; m1 = m[j, 1]; m2 = m[j, 2]
            for t in range(TB):
                d0 = p0[t] - q0; d1 = p1[t] - q1; d2 = p2[t] - q2
                ir = 1.0 / math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                ir2 = ir * ir
                ir3 = ir2 * ir
                tt = 3.0 * (m0 * d0 + m1 * d1 + m2 * d2) * ir2
                acc[0, t] += (tt * d0 - m0) * ir3
                acc[1, t] += (tt * d1 - m1) * ir3
                acc[2, t] += (tt * d2 - m2) * ir3
        _neumaier(acc, tot, cmp)


@njit(cache=True, fastmath=False, inline="always")
def _rhs(b0, b1, b2, c, s, R, sign, frac):
    """dR/dphi, dZ/dphi from B at (R, phi, Z), and whether the line can be followed there."""
    BR = b0 * c + b1 * s
    Bp = -b0 * s + b1 * c
    Bm = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
    ok = math.isfinite(Bm) and R > 0 and sign * Bp >= frac * Bm
    if not ok:
        return 0.0, 0.0, False
    return R * BR / Bp, R * b2 / Bp, True


@njit(cache=True, fastmath=False, inline="always")
def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


@njit(parallel=True, cache=True, fastmath=False)
def _kernel(y, comp, theta, lost, done, stopR, stopZ, stopphi, phi0, nper0, nper1, hs, off, plane, axR, axZ,
            period, k, nsf, rtab, sign, frac, box, dstop2, nodes, recR, recZ, wind, G, r, w, lv, ci, smp, reach, ext):
    """Periods nper0 .. nper1 - 1 of every line not stopped.  y, comp (L, 2): position and its Kahan compensation.
    theta (L,): the angle about the axis followed so far.  lost (L,) bool, done (L,) int64: periods completed.
    stopR, stopZ, stopphi (L,): where a line stopped.  recR, recZ (L, P, nper): where each line crosses each plane in
    each period.  wind (L, nper + 1): theta at the end of each period."""
    L = y.shape[0]
    ntask = (L + G - 1) // G
    nmax = G * k
    S = hs.shape[0]
    nr = nodes.shape[0]
    for g in prange(ntask):
        l0 = g * G
        l1 = min(L, l0 + G)
        ng = l1 - l0
        X = np.empty(nmax); Y = np.empty(nmax); Zt = np.empty(nmax)
        BX = np.empty(nmax); BY = np.empty(nmax); BZ = np.empty(nmax)
        CC = np.empty(nmax); SS = np.empty(nmax)
        p0 = np.empty(TB); p1 = np.empty(TB); p2 = np.empty(TB)
        acc = np.empty((3, TB)); tot = np.empty((3, TB)); cmp = np.empty((3, TB))
        tl = np.empty(nmax, np.int64)
        tj = np.empty(nmax, np.int64)
        zpR = np.empty((G, k)); zpZ = np.empty((G, k))
        zcR = np.empty((G, k)); zcZ = np.empty((G, k))
        TR = np.empty((G, k)); TZ = np.empty((G, k))
        act = np.empty(G, np.bool_)
        rowR = np.empty(k); rowZ = np.empty(k); tbR = np.empty(k); tbZ = np.empty(k)
        for n in range(nper0, nper1):
            for si in range(S):
                H = hs[si]
                t0 = phi0 + n * period + off[si]
                nact = 0
                for a in range(ng):
                    act[a] = not lost[l0 + a]
                    if act[a]:
                        nact += 1
                if nact == 0:
                    break
                # stage 0
                c0 = math.cos(t0); s0 = math.sin(t0)
                nt = 0
                for a in range(ng):
                    if act[a]:
                        l = l0 + a
                        X[nt] = y[l, 0] * c0; Y[nt] = y[l, 0] * s0; Zt[nt] = y[l, 1]
                        tl[nt] = a
                        nt += 1
                _field(nt, X, Y, Zt, BX, BY, BZ, r, w, lv, ci, smp, reach, p0, p1, p2, acc, tot, cmp, ext)
                for tt in range(nt):
                    a = tl[tt]
                    l = l0 + a
                    fR, fZ, ok = _rhs(BX[tt], BY[tt], BZ[tt], c0, s0, y[l, 0], sign, frac)
                    if not ok:
                        act[a] = False
                        lost[l] = True
                        stopR[l] = y[l, 0]; stopZ[l] = y[l, 1]; stopphi[l] = t0
                        continue
                    for j in range(k):
                        h = H / nsf[j]
                        zpR[a, j] = 0.0; zpZ[a, j] = 0.0
                        zcR[a, j] = h * fR; zcZ[a, j] = h * fZ
                # substeps m = 1 .. 2k - 1 of the sequences j with 2 (j + 1) > m, on increments
                for m in range(1, 2 * k):
                    jmin = m // 2
                    nt = 0
                    for a in range(ng):
                        if act[a]:
                            l = l0 + a
                            for j in range(jmin, k):
                                ph = t0 + m * (H / nsf[j])
                                cc = math.cos(ph); ss = math.sin(ph)
                                R = y[l, 0] + zcR[a, j]
                                X[nt] = R * cc; Y[nt] = R * ss; Zt[nt] = y[l, 1] + zcZ[a, j]
                                CC[nt] = cc; SS[nt] = ss
                                tl[nt] = a
                                tj[nt] = j
                                nt += 1
                    if nt == 0:
                        break
                    _field(nt, X, Y, Zt, BX, BY, BZ, r, w, lv, ci, smp, reach, p0, p1, p2, acc, tot, cmp, ext)
                    for tt in range(nt):
                        a = tl[tt]; j = tj[tt]
                        if not act[a]:
                            continue
                        l = l0 + a
                        R = y[l, 0] + zcR[a, j]
                        fR, fZ, ok = _rhs(BX[tt], BY[tt], BZ[tt], CC[tt], SS[tt], R, sign, frac)
                        if not ok:
                            act[a] = False
                            lost[l] = True
                            stopR[l] = y[l, 0]; stopZ[l] = y[l, 1]; stopphi[l] = t0
                            continue
                        h2 = 2.0 * (H / nsf[j])
                        nR = zpR[a, j] + h2 * fR
                        nZ = zpZ[a, j] + h2 * fZ
                        zpR[a, j] = zcR[a, j]; zpZ[a, j] = zcZ[a, j]
                        zcR[a, j] = nR; zcZ[a, j] = nZ
                        if m + 1 == 2 * (j + 1):
                            TR[a, j] = nR; TZ[a, j] = nZ
                # Aitken-Neville on the increments, the compensated update, then the checks at the step's end
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
                    dR = tbR[k - 1]; dZ = tbZ[k - 1]
                    if not (math.isfinite(dR) and math.isfinite(dZ)):
                        lost[l] = True
                        stopR[l] = y[l, 0]; stopZ[l] = y[l, 1]; stopphi[l] = t0
                        continue
                    vR = dR - comp[l, 0]
                    sR = y[l, 0] + vR
                    comp[l, 0] = (sR - y[l, 0]) - vR
                    vZ = dZ - comp[l, 1]
                    sZ = y[l, 1] + vZ
                    comp[l, 1] = (sZ - y[l, 1]) - vZ
                    y[l, 0] = sR; y[l, 1] = sZ
                    t1 = t0 + H
                    out = sR < box[0] or sR > box[1] or sZ < box[2] or sZ > box[3]
                    if not out and dstop2 > 0:
                        c1 = math.cos(t1); s1 = math.sin(t1)
                        xx = sR * c1; yy = sR * s1
                        for q in range(nr):
                            e0 = xx - nodes[q, 0]; e1 = yy - nodes[q, 1]; e2 = sZ - nodes[q, 2]
                            if e0 * e0 + e1 * e1 + e2 * e2 < dstop2:
                                out = True
                                break
                    if out:
                        lost[l] = True
                        stopR[l] = sR; stopZ[l] = sZ; stopphi[l] = t1
                        continue
                    if axR[si] == axR[si]:
                        ang = math.atan2(sZ - axZ[si], sR - axR[si])
                        theta[l] = theta[l] + _wrap(ang - _wrap(theta[l]))
                    pl = plane[si]
                    if pl >= 0:
                        recR[l, pl, n] = sR; recZ[l, pl, n] = sZ
                    if si == S - 1:
                        done[l] = n + 1
                        wind[l, n + 1] = theta[l]


def _gbs_numpy(B, y, t0, H, k, nsf, rtab, sign, frac):
    """One step of the same scheme for lines y (L, 2) from angle t0 (scalar) by H, with the field B(x (n, 3)) -> (n, 3):
    the increment (L, 2) and whether each line could be followed."""
    L = len(y)

    def f(ph, R, Z):
        c, s = np.cos(ph), np.sin(ph)
        Bv = np.asarray(B(np.stack([R * c, R * s, Z], -1)), float)
        BR = Bv[:, 0] * c + Bv[:, 1] * s
        Bp = -Bv[:, 0] * s + Bv[:, 1] * c
        Bm = np.sqrt(np.sum(Bv * Bv, -1))
        ok = np.isfinite(Bm) & (R > 0) & (sign * Bp >= frac * Bm)
        Bp = np.where(ok, Bp, 1.0)
        return R * BR / Bp, R * Bv[:, 2] / Bp, ok

    fR, fZ, ok = f(np.full(L, t0), y[:, 0], y[:, 1])
    zp = np.zeros((k, L, 2))
    hs = H / nsf
    zc = np.stack([np.stack([h * fR, h * fZ], -1) for h in hs])
    T = np.empty((k, L, 2))
    for m in range(1, 2 * k):
        J = [j for j in range(k) if 2 * (j + 1) > m]
        ph = np.concatenate([np.full(L, t0 + m * hs[j]) for j in J])
        R = np.concatenate([y[:, 0] + zc[j, :, 0] for j in J])
        Z = np.concatenate([y[:, 1] + zc[j, :, 1] for j in J])
        gR, gZ, gok = f(ph, R, Z)
        for a, j in enumerate(J):
            sl = slice(a * L, (a + 1) * L)
            ok &= gok[sl]
            new = zp[j] + 2.0 * hs[j] * np.stack([gR[sl], gZ[sl]], -1)
            zp[j] = zc[j]
            zc[j] = new
            if m + 1 == 2 * (j + 1):
                T[j] = new
    tab = [T[0]]
    for j in range(1, k):
        row = [T[j]]
        for kk in range(1, j + 1):
            row.append(row[kk - 1] + (row[kk - 1] - tab[kk - 1]) / rtab[j, kk])
        tab = row
    d = tab[-1]
    return d, ok & np.all(np.isfinite(d), -1)


def _reach(r, w, curve, top, tol, region, rng=None):
    """For strides 2, 4, .. 2^top of one coil's nodes r, w: the distance from the coil beyond which the rule on every
    m-th node is within tol (T) of the rule on every node.  Measured at the points of region (where the lines can go)
    and at 192 more round the coil (coil points moved along random directions, kept where the whole coil is that far)
    at 24 distances from 1e-3 to 1e3 times the coil's size: a tenth past the farthest point where a rule was not
    within tol.  Zero where it always was."""
    rng = rng or np.random.default_rng(0)
    size = float(np.max(np.ptp(curve, axis=0)))
    base = curve[rng.integers(0, len(curve), 192)]
    u = rng.normal(size=(192, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = [np.asarray(region, float).reshape(-1, 3)] + [base + D * u for D in size * np.logspace(-3, 3, 24)]
    out = np.zeros(top)
    for x in pts:
        if not len(x):
            continue
        dist = np.sqrt(np.min(np.sum((x[:, None, :] - curve[None, :, :]) ** 2, -1), axis=1))
        for x0 in range(0, len(x), 256):
            xs, ds = x[x0:x0 + 256], dist[x0:x0 + 256]
            d = xs[:, None, :] - r[None, :, :]
            terms = (np.sum(d * d, -1) ** -1.5)[:, :, None] * np.cross(w[None, :, :], d)     # (points, nodes, 3)
            full = terms.sum(axis=1)
            for L in range(1, top + 1):
                m = 2 ** L
                bad = np.linalg.norm(m * terms[:, ::m].sum(axis=1) - full, axis=1) >= tol
                if bad.any():
                    out[L - 1] = max(out[L - 1], 1.1 * float(ds[bad].max()))
    return np.maximum.accumulate(out)


def _plain(engine):
    """Whether the kernel computes this engine's whole field: a FieldEngine on the host whose parts are filaments,
    segments, dipoles and the 1/R field (rings, axisymmetric conductors and analytic parts are JAX's alone)."""
    from .field.engine import FieldEngine
    from . import fast_family
    if type(engine) is not FieldEngine or not getattr(engine, "cpu_kernels", False) or fast_family.on("cudafil"):
        return False
    if engine.sheets:
        return False
    rest = tuple(getattr(engine, "_rest_kinds", ()) or ())
    if any(rest[:4]) or getattr(engine, "analytic", None) or getattr(engine, "_np", None) is None:
        return False
    return True


class Tracer:
    """Field lines of an engine's field through the planes of a Schedule.

    engine: a torux FieldEngine.  box: (Rmin, Rmax, Zmin, Zmax) outside which a line has left.  dstop: a line nearer
    than this to a filament node of a coil has left (0: no such test).  sign: the sign of B_phi along the lines.
    bref: the field's strength where the lines are (T), the scale of the kernel's tolerance (_field).  region: points
    where the lines can go, at which the coarser rules of far coils are measured (_reach)."""

    def __init__(self, engine, box, dstop, sign, k=K, bref=1.0, region=()):
        self.engine = engine
        self.box = np.asarray(box, float)
        self.sign = float(sign)
        self.k = int(k)
        self.nsf, self.rtab = neville_divisors(self.k)
        self.fast = _plain(engine) and self._levels(engine, float(bref), region)
        self.nodes = self._nodes()
        self.dstop = float(dstop)

    def set_order(self, k):
        """Extrapolate over k sequences: order 2k."""
        self.k = int(k)
        self.nsf, self.rtab = neville_divisors(self.k)

    def _levels(self, engine, bref=1.0, region=()):
        """The kernel's arrays (_field): every Fourier coil's nodes at every stride its node count allows down to
        2 order + 2 nodes, each run padded to TB, with the coil's bounding sphere, points and model constant; the
        coils' nodes every fourth for the distance a line stops at; the other sources.  False where the engine's
        nodes are not the source's Fourier coils in order."""
        from .field.engine import nodes_needed
        P = engine._np
        z3 = np.zeros((0, 3))

        def arr(v):
            v = np.asarray(v, float)
            return np.ascontiguousarray(v.reshape(-1, 3) if v.size else z3)
        r, w = arr(P["r"]), arr(P["w"])
        self.near = np.ascontiguousarray(r[::4])
        self.ext = (arr(P["sa"]), arr(P["sb"]), np.ascontiguousarray(np.asarray(P["sw"], float).reshape(-1)),
                    arr(P["dp"]), arr(P["dm"]), float(getattr(engine, "RB", 0.0)))
        Ns = np.asarray(engine.info.N_per_coil, int).reshape(-1)
        info, smp, curves = [], [], []
        for part in engine.source.parts:
            if part.kind != "fourier":
                continue
            xs, dxs = part.points(np.arange(4096) / 4096)
            vmax = np.linalg.norm(dxs, axis=-1).max(axis=1)
            for i in range(part.n):
                if part.current[i] != 0.0:
                    c = xs[i].mean(axis=0)
                    info.append([float(nodes_needed(vmax[i], 1.0)), c[0], c[1], c[2],
                                 float(np.linalg.norm(xs[i] - c, axis=1).max()), 0.5 * vmax[i] / NS,
                                 2 * part.order + 2, 0.0, 0.0])
                    smp.append(xs[i][np.arange(NS) * (4096 // NS)])
                    curves.append(xs[i][::8])
        if len(info) != len(Ns) or int(Ns.sum()) != len(r):
            return False
        C = len(info)
        start = np.concatenate([[0], np.cumsum(Ns)]).astype(int)
        lv = np.zeros((LEVELS, C, 2), np.int64)
        R, W = [], []
        at = 0
        for c in range(C):
            for L in range(LEVELS):
                m = 2 ** L
                if Ns[c] % m or (L > 0 and Ns[c] // m < info[c][6]):
                    break
                rr = r[start[c]:start[c + 1]:m]
                ww = w[start[c]:start[c + 1]:m] * m
                pad = (-len(rr)) % TB
                R.append(np.concatenate([rr, np.full((pad, 3), 1e6)]))
                W.append(np.concatenate([ww, np.zeros((pad, 3))]))
                lv[L, c] = (at, len(rr) + pad)
                at += len(rr) + pad
        self.r = np.ascontiguousarray(np.concatenate(R).T) if R else np.zeros((3, 0))
        self.w = np.ascontiguousarray(np.concatenate(W).T) if W else np.zeros((3, 0))
        self.lv = lv
        self.ci = np.ascontiguousarray(np.array(info, float).reshape(-1, 9))
        self.smp = np.ascontiguousarray(np.array(smp, float).reshape(-1, NS, 3))
        self.reach = np.full((C, LEVELS), np.inf)
        self.reach[:, 0] = 0.0
        for c in range(C):
            top = int(np.max(np.flatnonzero(lv[:, c, 1] > 0)))
            if top:
                self.reach[c, 1:top + 1] = _reach(r[start[c]:start[c + 1]], w[start[c]:start[c + 1]], curves[c], top,
                                                  STRIDE_TOL * bref, region)
        return True

    def _nodes(self):
        """Points on the conductors whose distance stops a line: the filament nodes of the coils (every fourth where
        the kernel takes them), the segments' ends and middles."""
        P = getattr(self.engine, "_np", None) or {}
        pts = []
        if self.fast:
            pts.append(self.near)
        elif len(P.get("r", ())):
            pts.append(np.asarray(P["r"], float))
        if len(P.get("sa", ())):
            a, b = np.asarray(P["sa"], float), np.asarray(P["sb"], float)
            pts += [a, b, 0.5 * (a + b)]
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def B_kernel(self, x):
        """B at points x (n, 3) by the kernel's own sum (for checks)."""
        x = np.ascontiguousarray(np.asarray(x, float).reshape(-1, 3))
        out = np.empty((3, len(x)))
        _field(len(x), x[:, 0].copy(), x[:, 1].copy(), x[:, 2].copy(), out[0], out[1], out[2], self.r, self.w,
               self.lv, self.ci, self.smp, self.reach, np.empty(TB), np.empty(TB), np.empty(TB), np.empty((3, TB)),
               np.empty((3, TB)), np.empty((3, TB)), self.ext)
        return out.T

    def B(self, x):
        from . import fast_trace
        return np.asarray(fast_trace.field_B(self.engine, x), float)

    def run(self, y0, schedule, nper, phi0=0.0, theta0=None, progress=None):
        """Every line from (R, Z) y0 (L, 2) in the plane phi0 for nper periods (theta0: the angles about the axis it
        has turned so far, to go on from; default the angle of y0).  Returns dict: R, Z (L, P, nper)
        crossings (NaN once stopped), wind (L, nper + 1) the angle about the axis at each period's end (NaN once
        stopped), lost (L,), periods (L,) completed, stop (L, 3) R, phi, Z where a line stopped (NaN otherwise).
        progress(periods done, lost): called after each piece of the trace."""
        sc = schedule
        y = np.array(y0, float).reshape(-1, 2)
        L = len(y)
        P = int(sc.plane.max()) + 1
        comp = np.zeros((L, 2))
        theta = np.zeros(L) if theta0 is None else np.array(theta0, float)
        if np.isfinite(sc.axR[-1]) and theta0 is None:
            theta = np.arctan2(y[:, 1] - sc.axZ[-1], y[:, 0] - sc.axR[-1])
        lost = ~np.all(np.isfinite(y), -1)
        done = np.zeros(L, np.int64)
        stop = np.full((L, 3), np.nan)
        recR = np.full((L, P, nper), np.nan)
        recZ = np.full((L, P, nper), np.nan)
        wind = np.full((L, nper + 1), np.nan)
        wind[:, 0] = theta
        dstop2 = self.dstop ** 2 if len(self.nodes) else 0.0
        if self.fast and L:
            import numba
            from .interrupt import held
            G = int(max(1, min(GROUP_MAX, -(-L // max(1, numba.get_num_threads())))))
            sR, sZ, sp = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
            # in chunks of about CHUNK_S seconds, with Ctrl-C checked between them: each period goes on from the
            # state the last left, so the lines are the same to the bit however the periods are cut
            n0, size = 0, 1
            with held() as asked:
                while n0 < nper and not asked() and not lost.all():
                    n1 = min(int(nper), n0 + size)
                    t = time.perf_counter()
                    _kernel(y, comp, theta, lost, done, sR, sZ, sp, float(phi0), n0, n1, sc.h, sc.off, sc.plane,
                            sc.axR, sc.axZ, sc.period, self.k, self.nsf, self.rtab, self.sign, FRAC, self.box,
                            dstop2, self.nodes, recR, recZ, wind, G, self.r, self.w, self.lv, self.ci, self.smp,
                            self.reach, self.ext)
                    per = (time.perf_counter() - t) / (n1 - n0)
                    size = int(max(1, min(4 * size, CHUNK_S / max(per, 1e-9))))
                    n0 = n1
                    if progress is not None:
                        progress(n0, lost)
            stop = np.column_stack([sR, sp - phi0, sZ])
        elif L:
            self._run_numpy(y, comp, theta, lost, done, stop, sc, nper, phi0, dstop2, recR, recZ, wind, progress)
        for i in range(L):
            wind[i, done[i] + 1:] = np.nan
        stop[~lost] = np.nan
        return dict(R=recR, Z=recZ, wind=wind, lost=lost, periods=done, stop=stop, y=y)

    def _run_numpy(self, y, comp, theta, lost, done, stop, sc, nper, phi0, dstop2, recR, recZ, wind, progress=None):
        from scipy.spatial import cKDTree
        tree = cKDTree(self.nodes) if dstop2 > 0 else None
        for n in range(nper):
            if progress is not None and n:
                progress(n, lost)
            for si in range(sc.n):
                act = np.flatnonzero(~lost)
                if not len(act):
                    return
                t0 = phi0 + n * sc.period + sc.off[si]
                d, ok = _gbs_numpy(self.B, y[act], t0, sc.h[si], self.k, self.nsf, self.rtab, self.sign, FRAC)
                bad = act[~ok]
                lost[bad] = True
                stop[bad] = np.column_stack([y[bad, 0], np.full(len(bad), t0 - phi0), y[bad, 1]])
                act, d = act[ok], d[ok]
                v = d - comp[act]
                s = y[act] + v
                comp[act] = (s - y[act]) - v
                y[act] = s
                t1 = t0 + sc.h[si]
                R, Z = s[:, 0], s[:, 1]
                out = (R < self.box[0]) | (R > self.box[1]) | (Z < self.box[2]) | (Z > self.box[3])
                if tree is not None and np.any(~out):
                    x = np.column_stack([R * np.cos(t1), R * np.sin(t1), Z])
                    out |= tree.query(x)[0] ** 2 < dstop2
                o = act[out]
                lost[o] = True
                stop[o] = np.column_stack([y[o, 0], np.full(len(o), t1 - phi0), y[o, 1]])
                act = act[~out]
                if np.isfinite(sc.axR[si]):
                    ang = np.arctan2(y[act, 1] - sc.axZ[si], y[act, 0] - sc.axR[si])
                    theta[act] += np.mod(ang - theta[act] + np.pi, 2 * np.pi) - np.pi
                if sc.plane[si] >= 0:
                    recR[act, sc.plane[si], n] = y[act, 0]
                    recZ[act, sc.plane[si], n] = y[act, 1]
                if si == sc.n - 1:
                    done[act] = n + 1
                    wind[act, n + 1] = theta[act]
