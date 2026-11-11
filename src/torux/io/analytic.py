"""Analytic vacuum fields that are not current elements: Dommaschk potentials.

Dommaschk (Comput. Phys. Commun. 40, 203 (1986)); construction as in the appendix of
Hudson et al. 2020 (Plasma Phys. Control. Fusion 62, 084002, eqs. 31-37):

    B = B0R0 * grad Phi,   Phi = phi + sum_{m,l} Phi_{m,l}(R, phi, Z)   (R in units of 1 m)
    Phi_{m,l} = a D_{m,l} cos(m phi) + b N_{m,l-1} cos(m phi)
              + c D_{m,l} sin(m phi) + d N_{m,l-1} sin(m phi)
    D_{m,n}(R, Z) = sum_{k=0}^{n/2}       Z^{n-2k} / (n-2k)! C^D_{m,k}(R)
    N_{m,n}(R, Z) = sum_{k=0}^{(n-1)/2}   Z^{n-2k} / (n-2k)! C^N_{m,k}(R)
    L_m C_{m,k} = -C_{m,k-1},  L_m = d^2/dR^2 + (1/R) d/dR - m^2 / R^2  (Laplace's equation)
    normalisation at R = 1:  C^D_{m,0} = 1, C^D_{m,0}' = 0;  C^N_{m,0} = 0, C^N_{m,0}' = 1;
                             C_{m,k} = C_{m,k}' = 0 for k >= 1.
For k = 0, 1 this reproduces the paper's closed forms (35)-(37):
C^D_{m,0} = (R^m + R^-m)/2, C^N_{m,0} = (R^m - R^-m)/(2m),
C^D_{m,1} = [-m(m-1) R^{m+2} + (m+1)(m-2) R^m + m(m+1) R^{2-m} - (m+2)(m-1) R^{-m}] / [8 m (m^2-1)]
(the PDF text shows the denominator as "8m(m - 1)" with the exponent 2 detached; the
version with m^2 - 1 is the one that satisfies L_m C^D_{m,1} = -C^D_{m,0}, checked in the tests).

The C_{m,k} are built exactly as finite sums of c R^p (ln R)^q (resonant powers produce
logarithms), so the field is an exact closed-form function (no quadrature).

DOM25B (SPEC's vacuum verification case): m = 5, l = 2, Phi_{5,2} = A1 D_{5,2} sin 5phi
+ A2 N_{5,1} cos 5phi with A1 = A2 = -1.489 (paper, citing Dommaschk IPP 0/48 table IV),
B0R0 = 1 T m (SPEC input curpol = 2 pi).
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial

import numpy as np


# --------------------------------------------------------------------------- exact R-functions
def _Lm(term_dict, m):
    """Apply L_m to sum c R^p ln^q R (dict {(p,q): c})."""
    out = {}
    for (p, q), c in term_dict.items():
        for (pp, qq), cc in (((p - 2, q), c * (p * p - m * m)), ((p - 2, q - 1), c * 2 * p * q),
                             ((p - 2, q - 2), c * q * (q - 1))):
            if qq >= 0 and cc != 0:
                out[(pp, qq)] = out.get((pp, qq), 0) + cc
    return out


def _solve_Lm(src, m):
    """Particular solution X of L_m X = src (dict), polynomial-log ansatz, exact Fractions."""
    src = {k: Fraction(v) for k, v in src.items() if v != 0}
    X = {}
    # process highest log power first for each power p
    guard = 0
    while src and guard < 1000:
        guard += 1
        (s, q) = max(src, key=lambda k: (k[1], -abs(k[0])))
        c = src[(s, q)]
        p = s + 2
        if p * p != m * m:
            # L_m [R^p ln^q] = (p^2-m^2) R^s ln^q + lower log powers
            a = c / (p * p - m * m)
            term = {(p, q): a}
        else:
            # resonant: L_m [R^p ln^{q+1}] = 2p(q+1) R^s ln^q + (q+1)q R^s ln^{q-1}   (p^2 = m^2)
            if p == 0:
                # m = 0, p = 0: L_0 [ln^{q+2}] = (q+2)(q+1) ln^q / R^2
                a = c / ((q + 2) * (q + 1))
                term = {(0, q + 2): a}
            else:
                a = c / (2 * p * (q + 1))
                term = {(p, q + 1): a}
        for k, v in term.items():
            X[k] = X.get(k, 0) + v
        L = _Lm(term, m)
        for k, v in L.items():
            src[k] = src.get(k, 0) - v
            if src[k] == 0:
                del src[k]
    if src:
        raise RuntimeError("L_m solve did not terminate")
    return X


def _value(td, R=1):
    return sum(c for (p, q), c in td.items() if q == 0)  # at R = 1, ln R = 0


def _deriv1(td):
    """value of d/dR at R=1: sum c (p R^{p-1} ln^q + q R^{p-1} ln^{q-1}) at R = 1."""
    return sum(c * p for (p, q), c in td.items() if q == 0) + sum(c * q for (p, q), c in td.items() if q == 1)


def _homog(m):
    return ({(m, 0): Fraction(1)}, {(-m, 0): Fraction(1)}) if m > 0 else ({(0, 0): Fraction(1)}, {(0, 1): Fraction(1)})


def _normalise(td, m, v0, d0):
    """Add a h1 + b h2 so that value(1) = v0, derivative(1) = d0."""
    h1, h2 = _homog(m)
    v, d = _value(td), _deriv1(td)
    # [h1(1) h2(1); h1'(1) h2'(1)] [a b]^T = [v0 - v, d0 - d]
    A = [[_value(h1), _value(h2)], [_deriv1(h1), _deriv1(h2)]]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    rv, rd = Fraction(v0) - v, Fraction(d0) - d
    a = (rv * A[1][1] - A[0][1] * rd) / det
    b = (A[0][0] * rd - A[1][0] * rv) / det
    out = dict(td)
    for h, c in ((h1, a), (h2, b)):
        for k, val in h.items():
            out[k] = out.get(k, 0) + c * val
    return {k: v for k, v in out.items() if v != 0}


_CACHE = {}


def C_functions(m, kmax, kind):
    """[C_{m,0}, ..., C_{m,kmax}] as dicts {(p, q): Fraction} (kind 'D' or 'N')."""
    key = (m, kmax, kind)
    if key in _CACHE:
        return _CACHE[key]
    out = []
    C0 = _normalise({}, m, 1, 0) if kind == "D" else _normalise({}, m, 0, 1)
    out.append(C0)
    for k in range(1, kmax + 1):
        src = {kk: -v for kk, v in out[-1].items()}
        X = _solve_Lm(src, m)
        out.append(_normalise(X, m, 0, 0))
    _CACHE[key] = out
    return out


def _eval_td(td, R, lnR):
    """value, dR, dRR of sum c R^p ln^q at arrays R."""
    f = np.zeros_like(R); fr = np.zeros_like(R)
    for (p, q), c in td.items():
        c = float(c)
        Rp = R ** p
        L = lnR ** q if q else 1.0
        f = f + c * Rp * L
        # d/dR (R^p ln^q) = p R^{p-1} ln^q + q R^{p-1} ln^{q-1}
        t = p * R ** (p - 1) * L
        if q:
            t = t + q * R ** (p - 1) * (lnR ** (q - 1) if q > 1 else 1.0)
        fr = fr + c * t
    return f, fr


def DN(m, n, kind, R, Z):
    """D_{m,n} (kind 'D') or N_{m,n} (kind 'N') and its R and Z derivatives."""
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    lnR = np.log(R)
    kmax = n // 2 if kind == "D" else (n - 1) // 2
    if n < 0 or kmax < 0:
        z = np.zeros_like(R)
        return z, z, z
    Cs = C_functions(m, kmax, kind)
    f = np.zeros_like(R); fR = np.zeros_like(R); fZ = np.zeros_like(R)
    for k in range(kmax + 1):
        e = n - 2 * k
        c, cr = _eval_td(Cs[k], R, lnR)
        zf = Z ** e / factorial(e)
        dz = Z ** (e - 1) / factorial(e - 1) if e >= 1 else 0.0
        f = f + zf * c; fR = fR + zf * cr; fZ = fZ + dz * c
    return f, fR, fZ


class DommaschkField:
    """B = B0R0 grad (phi + sum terms).  terms: list of (m, l, a, b, c, d) with
    Phi_{m,l} = (a D_{m,l} + b N_{m,l-1}) cos(m phi) + (c D_{m,l} + d N_{m,l-1}) sin(m phi).
    include_toroidal=False leaves out the grad(phi) part (use a ToroidalField(B0R0) part)."""

    kind = "dommaschk"

    def __init__(self, terms, B0R0=1.0, include_toroidal=True, nfp=None):
        self.terms = [tuple(float(x) if i >= 2 else int(x) for i, x in enumerate(t)) for t in terms]
        self.B0R0 = float(B0R0)
        self.include_toroidal = include_toroidal
        self.nfp = nfp

    def to_dict(self):
        return {"class": "torux.io.analytic:DommaschkField", "terms": [list(t) for t in self.terms],
                "B0R0": self.B0R0, "include_toroidal": self.include_toroidal, "nfp": self.nfp}

    def potential_grad_cyl(self, R, phi, Z):
        dR = np.zeros_like(R); dphi = np.zeros_like(R); dZ = np.zeros_like(R); Phi = np.zeros_like(R)
        if self.include_toroidal:
            Phi = Phi + phi
            dphi = dphi + 1.0
        for (m, l, a, b, c, d) in self.terms:
            D, DR_, DZ_ = DN(m, l, "D", R, Z)
            N, NR_, NZ_ = DN(m, l - 1, "N", R, Z)
            cm, sm = np.cos(m * phi), np.sin(m * phi)
            A = a * D + b * N; AR = a * DR_ + b * NR_; AZ = a * DZ_ + b * NZ_
            S = c * D + d * N; SR = c * DR_ + d * NR_; SZ = c * DZ_ + d * NZ_
            Phi = Phi + A * cm + S * sm
            dR = dR + AR * cm + SR * sm
            dZ = dZ + AZ * cm + SZ * sm
            dphi = dphi + m * (-A * sm + S * cm)
        return Phi, dR, dphi, dZ

    def B_cyl(self, R, phi, Z):
        _, dR, dphi, dZ = self.potential_grad_cyl(np.asarray(R, float), np.asarray(phi, float), np.asarray(Z, float))
        return self.B0R0 * dR, self.B0R0 * dphi / R, self.B0R0 * dZ

    def B(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        R = np.hypot(x[:, 0], x[:, 1]); ph = np.arctan2(x[:, 1], x[:, 0])
        BR, BP, BZ = self.B_cyl(R, ph, x[:, 2])
        c, s = np.cos(ph), np.sin(ph)
        return np.stack([BR * c - BP * s, BR * s + BP * c, BZ], 1)


def dom25b(B0R0=1.0, include_toroidal=True):
    """The SPEC verification field DOM25B: m = 5, l = 2, A1 = A2 = -1.489."""
    return DommaschkField([(5, 2, 0.0, -1.489, -1.489, 0.0)], B0R0=B0R0, include_toroidal=include_toroidal, nfp=5)
