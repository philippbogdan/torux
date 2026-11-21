"""The independent reference's SurfaceCurrent field (torux.reference.sheet, sheet_dd) against truths that use no
sheet code of torux: superposed rings (axisymmetric), a uniform tensor trapezoid rule written here (shaped,
non-axisymmetric, four-period, both input forms), Ampere's law on loops round the boundary, the public VMEX
virtual-casing field, and the double-double tier."""
import json
from pathlib import Path

import numpy as np
import pytest

import torux
from torux.reference import fast64, precise, trace
from torux.reference.sheet import SheetField
from torux.source import MU0

EXAMPLES = Path(__file__).parents[1] / "examples"


# ----------------------------------------------------------------------------------------------- boundaries
def dshape(nfp=4, M=16):
    """An axisymmetric D: R = 1 + 0.3 cos(t + 0.4 sin t), Z = 0.48 sin t, as cosine / sine series to m = M."""
    t = 2 * np.pi * np.arange(256) / 256
    cR = np.fft.rfft(1 + 0.3 * np.cos(t + 0.4 * np.sin(t))) / 256
    cZ = np.fft.rfft(0.48 * np.sin(t)) / 256
    modes = [(m, 0) for m in range(M + 1)]
    rmnc = np.r_[cR[0].real, 2 * cR[1:M + 1].real]
    zmns = np.r_[0.0, -2 * cZ[1:M + 1].imag]
    return modes, rmnc, zmns


def stell1():
    """One period, no stellarator symmetry, current potential with G and I."""
    modes = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, -1), (2, 1), (3, 2), (0, 2)]
    rmnc = [1.0, 0.28, 0.03, 0.06, 0.07, 0.02, 0.012, 0.004, 0.01]
    rmns = [0.0, 0.0, 0.01, 0.02, -0.015, 0.0, 0.005, 0.0, 0.003]
    zmns = [0.0, 0.33, -0.02, 0.05, -0.06, 0.015, 0.01, -0.003, 0.004]
    zmnc = [0.02, 0.01, 0.0, -0.01, 0.012, 0.0, 0.0, 0.002, 0.0]
    r = np.random.default_rng(5)
    w = 3e3 / (1 + np.abs(np.array(modes)).sum(1)) ** 2
    kc, ks = r.normal(size=len(modes)) * w, r.normal(size=len(modes)) * w
    kc[0] = ks[0] = 0
    return torux.SurfaceCurrent(1, modes, rmnc, zmns, rmns=rmns, zmnc=zmnc, kappac=kc, kappas=ks, G=4e5, I=-1.5e5)


def qa4(form="covariant"):
    """Four periods, stellarator-symmetric shape; the covariant form with components that are not a gradient,
    or (form='kappa') a current potential with G and I."""
    modes = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, 2), (3, 1), (0, 2), (1, 2)]
    rmnc = [1.0, 0.18, 0.02, 0.025, 0.04, 0.01, 0.008, 0.003, 0.002, 0.004, 0.003]
    zmns = [0.0, 0.22, -0.015, 0.02, 0.045, -0.008, 0.006, 0.002, -0.0015, 0.003, -0.002]
    r = np.random.default_rng(7)
    w = 2e4 / (1 + np.abs(np.array(modes)).sum(1)) ** 2
    if form == "kappa":
        kc, ks = r.normal(size=len(modes)) * w, r.normal(size=len(modes)) * w
        return torux.SurfaceCurrent(4, modes, rmnc, zmns, kappac=kc, kappas=ks, G=3.7e5, I=6e4)
    cov = r.normal(size=(2, 2, len(modes))) * w
    cov[0, 0, 0], cov[1, 0, 0] = 1.1e4, 6e4
    cov[:, 1, 0] = 0
    return torux.SurfaceCurrent(4, modes, rmnc, zmns, covariant=cov)


def geometry(part, th, ph):
    """R, Z, their theta and phi derivatives, and (c_theta, c_phi), straight from the definitions."""
    m, n = part.modes[:, 0].astype(float), part.modes[:, 1].astype(float) * part.nfp
    z = np.zeros(len(m))
    g = lambda v: z if v is None else np.asarray(v, float)                   # noqa: E731
    psi = th[..., None] * m - ph[..., None] * n
    c, s = np.cos(psi), np.sin(psi)
    rc, rs, zc, zs = g(part.rmnc), g(part.rmns), g(part.zmnc), g(part.zmns)
    out = [c @ rc + s @ rs, c @ zc + s @ zs, (-s * m) @ rc + (c * m) @ rs, (-s * m) @ zc + (c * m) @ zs,
           (s * n) @ rc - (c * n) @ rs, (s * n) @ zc - (c * n) @ zs]
    if part.covariant is not None:
        cv = np.asarray(part.covariant, float)
        out += [c @ cv[0, 0] + s @ cv[0, 1], c @ cv[1, 0] + s @ cv[1, 1]]
    else:
        kc, ks = g(part.kappac), g(part.kappas)
        out += [(-s * m) @ kc + (c * m) @ ks + part.I / (2 * np.pi), (s * n) @ kc - (c * n) @ ks + part.G / (2 * np.pi)]
    return out


def frame(part, th, ph):
    R, Z, Rt, Zt, Rp, Zp, ct, cp = geometry(part, th, ph)
    co, si = np.cos(ph), np.sin(ph)
    y = np.stack([R * co, R * si, Z], -1)
    rt = np.stack([Rt * co, Rt * si, Zt], -1)
    rp = np.stack([Rp * co - R * si, Rp * si + R * co, Zp], -1)
    return y, rt, rp, ct, cp


def off(part, d, n, seed):
    """n points at distance d along the outward unit normal from random boundary points."""
    r = np.random.default_rng(seed)
    th, ph = r.uniform(0, 2 * np.pi, n), r.uniform(0, 2 * np.pi, n)
    y, rt, rp, _, _ = frame(part, th, ph)
    N = np.cross(rp, rt)
    return y + d * N / np.linalg.norm(N, axis=1)[:, None]


def trapezoid(part, X, nt, nph):
    """B, A, dB by the uniform tensor trapezoid rule in (theta, phi) over the whole torus, pairwise sums."""
    TH, PH = np.meshgrid(2 * np.pi * np.arange(nt) / nt, 2 * np.pi * np.arange(nph) / nph, indexing="ij")
    y, rt, rp, ct, cp = frame(part, TH.ravel(), PH.ravel())
    J = (cp[:, None] * rt - ct[:, None] * rp).T.copy()
    y = y.T.copy()
    w = 1e-7 * (2 * np.pi) ** 2 / (nt * nph)
    S = lambda v: np.ascontiguousarray(v).sum(-1)            # noqa: E731  pairwise summation (contiguous axis)
    cross = lambda u, v: np.stack([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],     # noqa: E731
                                   u[0] * v[1] - u[1] * v[0]])
    B, A, G = [], [], []
    for x in X:
        D = x[:, None] - y
        r = np.sqrt(D[0] ** 2 + D[1] ** 2 + D[2] ** 2)
        JxD = cross(J, D)
        B.append(w * S(JxD / r ** 3))
        A.append(w * S(J / r))
        G.append(np.stack([w * S(cross(J, np.eye(3)[j][:, None] + 0 * D) / r ** 3 - 3 * JxD * D[j] / r ** 5)
                           for j in range(3)], 1))
    return np.array(B), np.array(A), np.array(G)


def rel(a, b):
    a, b = np.asarray(a).reshape(len(a), -1), np.asarray(b).reshape(len(b), -1)
    return float(np.max(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))


# ----------------------------------------------------------------------------------------------- truths
def test_axisymmetric_sheet_is_its_rings():
    """Axisymmetric D (declared four-period), current potential of theta only with I and G: outside, the poloidal
    current makes no field and the toroidal current is the rings -c_theta dtheta at (R, Z)(theta), whose field is
    the reference's closed-form ring evaluator (precise tier) summed by the trapezoid rule in theta."""
    modes, rmnc, zmns = dshape()
    K = len(modes)
    kc, ks = np.zeros(K), np.zeros(K)
    kc[1:5], ks[1:5] = [2e4, -7e3, 3e3, 1e3], [-1.2e4, 5e3, -2e3, 8e2]
    sheet = torux.SurfaceCurrent(4, modes, rmnc, zmns, kappac=kc, kappas=ks, I=3e5, G=2e5)
    S = SheetField(sheet)
    a = S.minor_radius
    for frac, nring in ((0.5, 2048), (1e-2, 16384)):
        X = off(sheet, frac * a, 4, seed=1)
        t = 2 * np.pi * np.arange(nring) / nring
        R, Z, _, _, _, _, ct, _ = geometry(sheet, t, 0 * t)
        rings = precise.Reference(torux.Rings(R, Z, -ct * 2 * np.pi / nring)).evaluate(X)
        got = S.evaluate(X)
        assert rel(got["B"], rings["B"]) < 4e-15
        if frac > 0.1:
            # (the rings' dB converges only algebraically in their number next to the boundary)
            assert rel(got["dB"], rings["dB"]) < 1e-13
        np.testing.assert_allclose(got["dist"], frac * a, rtol=1e-9)
    # with G = 0 the current is the rings alone, so A is theirs too
    X = off(sheet, 0.1 * a, 4, seed=2)
    nring = 8192
    t = 2 * np.pi * np.arange(nring) / nring
    R, Z, _, _, _, _, ct, _ = geometry(sheet, t, 0 * t)
    rings = precise.Reference(torux.Rings(R, Z, -ct * 2 * np.pi / nring)).evaluate(X, grad=False)
    sheet0 = torux.SurfaceCurrent(4, modes, rmnc, zmns, kappac=kc, kappas=ks, I=3e5)
    assert rel(SheetField(sheet0).evaluate(X, grad=False)["A"], rings["A"]) < 3e-15


@pytest.mark.parametrize("make", [stell1, qa4, lambda: qa4("kappa")], ids=["stell1-kappa", "qa4-covariant",
                                                                             "qa4-kappa"])
def test_shaped_boundaries_against_the_trapezoid_rule(make):
    """Away from the boundary the uniform trapezoid rule converges geometrically: at 1 and 0.5 minor radii it is
    the truth to rounding (checked by halving it)."""
    sheet = make()
    S = SheetField(sheet)
    a = S.minor_radius
    for frac, (nt, nph) in ((1.0, (256, 1024)), (0.5, (512, 2048))):
        X = off(sheet, frac * a, 3, seed=4)
        B, A, G = trapezoid(sheet, X, nt, nph)
        B2, _, _ = trapezoid(sheet, X, nt // 2, nph // 2)
        assert rel(B2, B) < 2e-15                      # the truth has converged
        got = fast64.Field(torux.Source("s", [sheet], nfp=sheet.nfp)).evaluate(X)
        assert rel(got["B"], B) < 5e-15
        assert rel(got["A"], A) < 5e-15
        assert rel(got["dB"], G) < 3e-14


def test_ampere_loops_round_the_boundary():
    """Outside the boundary: round the tube (a poloidal loop, theta increasing, so encircling -phi) the circulation of
    B is mu0 I, I flowing along -phi with K = n x grad kappa; along the tube (a toroidal loop) it is zero.  Loops 0.05 minor radii out, the
    trapezoid rule with spectral differentiation of the loop."""
    for sheet in (stell1(), qa4("kappa")):
        S = SheetField(sheet)
        d = 0.05 * S.minor_radius
        nl = 1024
        s = 2 * np.pi * np.arange(nl) / nl
        k = np.fft.fftfreq(nl, 1.0 / nl)
        for th, ph, want in ((s, np.full(nl, 0.7), MU0 * sheet.I), (np.full(nl, 2.1), s, 0.0)):
            y, rt, rp, _, _ = frame(sheet, th, ph)
            N = np.cross(rp, rt)
            X = y + d * N / np.linalg.norm(N, axis=1)[:, None]
            dX = np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(X, axis=0), axis=0))
            circ = np.sum(S.evaluate(X, grad=False)["B"] * dX) * 2 * np.pi / nl
            assert abs(circ - want) < 2e-13 * MU0 * max(abs(sheet.I), abs(sheet.G))


def test_public_vmex_virtual_casing_field():
    """VMEX's direct virtual-casing plasma field at 20 points 1 to 50 mm outside the Landreman-Paul QA boundary
    (examples/vmex_surface_current/reference.json): agreement to the accuracy of that reference."""
    ref = json.loads((EXAMPLES / "vmex_surface_current/reference.json").read_text())
    for case in ("current", "beta1"):
        src = torux.load(EXAMPLES / f"vmex_surface_current/{case}.json")
        sheet = [p for p in src.parts if p.kind == "surface_current"][0]
        got = SheetField(sheet).evaluate(np.array(ref[case]["points"]), grad=False)["B"]
        assert np.abs(got - np.array(ref[case]["B"])).max() < 1e-12


# ----------------------------------------------------------------------------------------------- the tiers
@pytest.mark.parametrize("make", [stell1, qa4], ids=["stell1-kappa", "qa4-covariant"])
def test_double_double_tier(make):
    """The precise tier: its estimate of its own error far below float64, float64 within a few ulps of it from
    1 down to 1e-10 minor radii, and the precise tier of a source adds the sheet to the coils."""
    sheet = make()
    S = SheetField(sheet)
    a = S.minor_radius
    X = np.concatenate([off(sheet, f * a, 1, seed=6) for f in (1.0, 1e-2, 1e-5, 1e-10)])
    P = precise.Reference(torux.Source("s", [sheet]))
    p = P.evaluate(X)
    f = S.evaluate(X)
    nb = np.linalg.norm(p["B"], axis=1)
    assert not p["flag"].any()
    assert np.all(p["err_B"] <= 1e-20 * p["scale_B"])
    assert np.all(p["err_A"] <= 1e-20 * p["scale_A"])
    assert rel(f["B"], p["B"]) < 4e-15
    assert rel(f["A"], p["A"]) < 4e-15
    dB = np.linalg.norm((f["dB"] - p["dB"]).reshape(-1, 9), axis=1) / np.linalg.norm(p["dB"].reshape(-1, 9), axis=1)
    assert np.all(np.abs(f["dB"] - p["dB"]).max(axis=(1, 2)) <= f["err_dB"])     # dB within its stated bound
    np.testing.assert_allclose(p["dmin"], f["dist"], rtol=1e-12)
    ring = torux.Rings([1.0], [0.0], [1e5])
    both = precise.Reference(torux.Source("s+r", [sheet, ring])).evaluate(X[:2], grad=False)
    alone = precise.Reference(ring).evaluate(X[:2], grad=False)
    np.testing.assert_allclose(both["B"], p["B"][:2] + alone["B"], rtol=0, atol=1e-15 * nb.max())


def test_four_period_rotation_and_both_forms():
    """B(R x) = R B(x) for the rotation by one field period, and a current potential given as such or as its
    covariant components (c_theta, c_phi) makes the same field."""
    sheet = qa4("kappa")
    S = SheetField(sheet)
    X = off(sheet, 0.02 * S.minor_radius, 4, seed=8)
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    b, br = S.evaluate(X), S.evaluate(X @ Rz.T)
    assert rel(br["B"], b["B"] @ Rz.T) < 4e-15
    assert rel(br["A"], b["A"] @ Rz.T) < 4e-15
    m, n = sheet.modes[:, 0], sheet.modes[:, 1] * 4
    cov = np.zeros((2, 2, len(m)))
    cov[0, 0], cov[0, 1] = m * sheet.kappas, -m * sheet.kappac           # d/dtheta
    cov[1, 0], cov[1, 1] = -n * sheet.kappas, n * sheet.kappac           # d/dphi
    cov[0, 0, 0], cov[1, 0, 0] = sheet.I / (2 * np.pi), sheet.G / (2 * np.pi)
    raw = torux.SurfaceCurrent(4, sheet.modes, sheet.rmnc, sheet.zmns, covariant=cov)
    assert rel(SheetField(raw).evaluate(X)["B"], b["B"]) < 3e-15


def test_refuses_inside_and_too_close():
    sheet = qa4()
    src = torux.Source("s", [sheet, torux.ToroidalField(1.0)], nfp=4)
    F = fast64.Field(src)
    a = F.sheets[0].minor_radius
    with pytest.raises(ValueError, match="inside the boundary"):
        F.evaluate(off(sheet, -1e-3 * a, 1, seed=0))
    with pytest.raises(ValueError, match="inside the boundary"):
        F.evaluate([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="closer to the boundary than 1e-12"):
        F.evaluate(off(sheet, 1e-13 * a, 1, seed=0))
    with pytest.raises(ValueError, match="inside the boundary"):
        precise.Reference(src).evaluate(off(sheet, -1e-3 * a, 1, seed=0))
    ok = F.evaluate(off(sheet, 1e-11 * a, 1, seed=0))
    assert np.isfinite(ok["B"]).all() and ok["dist_sheet"][0] > 1e-12 * a
    # the compiled point evaluators and the tracer do not include the sheet: they refuse it
    with pytest.raises(ValueError, match="SurfaceCurrent"):
        F.packed
    with pytest.raises(ValueError, match="SurfaceCurrent"):
        trace.poincare(src, [1.3], [0.0], [0.0], 1)


# ----------------------------------------------------------------------------------------------- virtual casing
def test_virtual_casing_null_on_a_four_period_flux_surface():
    """A vacuum flux surface of the four-period QUASR device 1100073 (tests/data, solved by torux; B.n/|B| there
    below 2e-14 in the reference's own coil field, measured here).  K = n x B_coils / mu0 on it, the covariant
    coefficients by FFT of the reference coil field, in both input forms: by virtual casing the sheet's field is 0
    outside and -B_coils inside (inside evaluated with the sheet's check switched off).  Non-axisymmetric, four
    periods, a real coil set: holds to a few 1e-15 of |B_coils| from 0.5 down to 1e-6 minor radii."""
    d = json.loads((Path(__file__).parent / "data/quasr_1100073_flux0.005_boundary.json").read_text())
    coils = torux.load(EXAMPLES / "quasr_1100073.json")
    nfp, modes = d["nfp"], np.array(d["modes"])
    geo = torux.SurfaceCurrent(nfp, modes, d["rmnc"], d["zmns"], rmns=d["rmns"], zmnc=d["zmnc"],
                               covariant=np.zeros((2, 2, len(modes))))
    Mc, Nc = 48, 32
    nt, nw = 4 * Mc, 4 * Nc
    TH, PH = np.meshgrid(2 * np.pi * np.arange(nt) / nt, 2 * np.pi / nfp * np.arange(nw) / nw, indexing="ij")
    y, rt, rp, _, _ = frame(geo, TH.ravel(), PH.ravel())
    F = fast64.Field(coils)
    B = F.evaluate(y, grad=False)["B"]
    N = np.cross(rp, rt)
    assert np.max(np.abs(np.sum(B * N, 1)) / np.linalg.norm(B, axis=1) / np.linalg.norm(N, axis=1)) < 2e-14
    # coefficients of exp(i (m theta - n w)), w = nfp phi
    Ct = np.fft.ifft(np.fft.fft((B * rt).sum(1).reshape(nt, nw) / MU0, axis=0), axis=1) / nt
    Cp = np.fft.ifft(np.fft.fft((B * rp).sum(1).reshape(nt, nw) / MU0, axis=0), axis=1) / nt
    idx = {tuple(mm): i for i, mm in enumerate(map(tuple, modes))}
    cov = np.zeros((2, 2, len(modes)))
    for (m, n), i in idx.items():
        if abs(m) <= Mc and abs(n) <= Nc:
            f = 1.0 if m == n == 0 else 2.0
            cov[:, 0, i] = f * Ct[m, n].real, f * Cp[m, n].real
            cov[:, 1, i] = -f * Ct[m, n].imag, -f * Cp[m, n].imag
    dropped = max(abs(Ct[m, n]) for m in range(-Mc, Mc) for n in range(-Nc, Nc) if (m, n) not in idx
                  and (-m, -n) not in idx)
    assert dropped < 1e-16 * np.abs(cov).max()
    # the same current as a potential: c_theta = d kappa / d theta, c_phi = d kappa / d phi
    kc, ks = np.zeros(len(modes)), np.zeros(len(modes))
    for (m, n), i in idx.items():
        if m:
            kc[i], ks[i] = -cov[0, 1, i] / m, cov[0, 0, i] / m
        elif n:
            kc[i], ks[i] = cov[1, 1, i] / (n * nfp), -cov[1, 0, i] / (n * nfp)
    i0 = idx[(0, 0)]
    forms = [dict(covariant=cov), dict(kappac=kc, kappas=ks, I=2 * np.pi * cov[0, 0, i0], G=2 * np.pi * cov[1, 0, i0])]
    th, ph = np.random.default_rng(3).uniform(0, 2 * np.pi, (2, 4))
    y0, rt0, rp0, _, _ = frame(geo, th, ph)
    n0 = np.cross(rp0, rt0)
    n0 /= np.linalg.norm(n0, axis=1)[:, None]
    for kw in forms:
        S = SheetField(torux.SurfaceCurrent(nfp, modes, d["rmnc"], d["zmns"], rmns=d["rmns"], zmnc=d["zmnc"], **kw))
        for frac in (0.5, 1e-2, 1e-6):
            out, inn = y0 + frac * S.minor_radius * n0, y0 - frac * S.minor_radius * n0
            bc, bci = F.evaluate(out, grad=False)["B"], F.evaluate(inn, grad=False)["B"]
            assert np.max(np.linalg.norm(S.evaluate(out, grad=False)["B"], axis=1) / np.linalg.norm(bc, axis=1)) < 1e-14
            assert rel(S.evaluate(inn, grad=False, check=False)["B"], -bci) < 1.5e-14
