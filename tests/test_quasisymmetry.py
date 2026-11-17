"""The two-term and three-term quasisymmetry residuals, against calculations that share no formula with them.

An axisymmetric field with a toroidal current, on an exact flux surface written in its straight-field-line angle, has
|B| = B_00 - b cos theta_B, one harmonic: f_C for (1, 0) and f_T vanish, and f_C for (0, 1), where the current's I
enters, has a closed form in elementary integrals.
On solved surfaces the residuals are computed a second way, in Boozer angles from the harmonics of |B| (which
torux.boozer's tests check against SIMSOPT), with every field value from torux.reference: in a vacuum, Boozer angles
give f_C = -(N dB/dtheta_B + M dB/dzeta_B) / B and sqrt(g) = (G + iota I) / B^2.  A quasi-helical QUASR design, whose
NonQuasiSymmetricRatio is 5e-10, has a two-term residual for its own helicity ten thousand times smaller than for
any other.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import example

import torux
from torux import boozer as BZ
from torux.boozer import Helicity, boozer
from torux.quasisymmetry import helicity_from_spectrum, quasisymmetry
from torux.torus import TorusState

from test_boozer import inputs

QH = "2405145"           # quasi-helical, nfp = 3, QUASR's qs_error 4.8e-10 on its one stored surface


@pytest.fixture(scope="module")
def solved_qh():
    res = torux.solve(inputs(QH))
    assert res.status == "surface"
    return res


# ------------------------------------------------------------------------------------------------ exact

R0, A, B0, C, NFP = 1.0, 0.3, 1.0, 0.4, 3


def _theta(ts):
    """The geometric poloidal angle of the circle (R - R0)^2 + Z^2 = A^2 at its straight-field-line angle ts, for the
    field below: d theta* / d theta = sqrt(R0^2 - A^2) / R."""
    k = math.sqrt((R0 + A) / (R0 - A))
    return 2 * np.arctan2(k * np.sin(ts / 2), np.cos(ts / 2))


def _axisymmetric(flip=False, geometric=False, shift=0.0, scale=0.0):
    """B = B0 R0 grad phi + grad psi_p x grad phi with psi_p = C ((R - R0)^2 + Z^2) / 2: circular flux surfaces round
    a toroidal current, iota = C sqrt(R0^2 - A^2) / (B0 R0). The surface of minor radius A as a torux surface in its
    straight-field-line angle (counter-clockwise, or clockwise with flip), and the field.  To be refused: the circle
    in its geometric angle, which is not straight, a circle shifted by shift (m) off the flux surface, and the field
    multiplied by 1 + scale (R - R0) / A, which a current then crosses."""
    K, M = 4096, 40
    ts = 2 * np.pi * np.arange(K) / K
    th = ts if geometric else _theta(ts)
    R, Z = R0 + shift + A * np.cos(th), A * np.sin(th)
    rc, zs = np.zeros((M + 1, 5)), np.zeros((M + 1, 5))
    for m in range(M + 1):
        rc[m, 2] = (1 if m == 0 else 2) * np.mean(R * np.cos(m * ts))
        zs[m, 2] = 2 * np.mean(Z * np.sin(m * ts))
    iota = C * math.sqrt(R0 ** 2 - A ** 2) / (B0 * R0)
    if flip:
        zs, iota = -zs, -iota
    t = TorusState(rc, zs, iota, 2 * np.pi * B0 * R0 * (R0 - math.sqrt(R0 ** 2 - A ** 2)), M, 2, NFP)

    def field(x):
        x = np.asarray(x, float)
        Rx = np.hypot(x[:, 0], x[:, 1])
        cp, sp = x[:, 0] / Rx, x[:, 1] / Rx
        BR, BZ_, Bphi = -C * x[:, 2] / Rx, C * (Rx - R0) / Rx, B0 * R0 / Rx
        f = (1 + scale * (Rx - R0) / A)[:, None]      # along B: the surface and its angle stay, the curl does not
        return f * np.stack([BR * cp - Bphi * sp, BR * sp + Bphi * cp, BZ_], -1)
    return SimpleNamespace(status="surface", surface=t, field=field)


def _single_harmonic():
    """|B| = B_00 - b cos theta_B on the surface, and <f_C^2> for (0, 1) in closed form.  Here B_theta* = I is constant,
    so the straight-field-line angles are Boozer's, and |B| = K / R, K = sqrt(B0^2 R0^2 + C^2 A^2), where
    1 / R = (R0 - A cos theta*) / (R0^2 - A^2) (the eccentric anomaly's relation).  With f_C = -N (dB/dtheta_B) / B and
    sqrt(g) proportional to 1 / B^2, <f_C^2> = N^2 (B_00 I_3 - I_2) / (3 I_2), I_k the integral over a turn of
    (B_00 - b cos theta)^-k."""
    K = math.sqrt(B0 ** 2 * R0 ** 2 + C ** 2 * A ** 2)
    b00, b = K * R0 / (R0 ** 2 - A ** 2), K * A / (R0 ** 2 - A ** 2)
    I2 = 2 * np.pi * b00 / (b00 ** 2 - b ** 2) ** 1.5
    I3 = np.pi * (2 * b00 ** 2 + b ** 2) / (b00 ** 2 - b ** 2) ** 2.5
    return b00, b, NFP ** 2 * (b00 * I3 - I2) / (3 * I2)


@pytest.mark.parametrize("flip", [False, True])
def test_an_axisymmetric_field_with_a_current_is_quasi_axisymmetric(flip):
    """|B| is one harmonic in Boozer angles, read from the field as (1, 0). The (1, 0) residual cancels only with the
    right sign and factor between its two terms. The (0, 1) one holds N = n nfp and the current inside the surface
    (I = C A^2 / sqrt(R0^2 - A^2), not zero), against its closed form. And f_T vanishes."""
    res = _axisymmetric(flip)
    b00, b, exact = _single_harmonic()
    spectrum = boozer(res)
    B = {(int(r["m"]), int(r["n"])): float(r["B"]) for r in spectrum.modes}
    assert abs(B.pop((0, 0)) - b00) < 1e-14 and abs(B.pop((1, 0)) + b) < 1e-14
    assert max(abs(v) for v in B.values()) < 1e-14
    qa = quasisymmetry(res)
    qp = quasisymmetry(res, helicity=(0, 1))
    assert qa.helicity == Helicity(1, 0) and not qa.given
    assert abs(qp.two_term / exact - 1) < 1e-13, (qp.two_term, exact)
    assert qa.two_term < 1e-28 * exact, qa.two_term
    assert qa.checks["|I|/|G|"] == pytest.approx(C * A ** 2 / math.sqrt(R0 ** 2 - A ** 2) / (B0 * R0), rel=1e-13)
    # rounding: f_T is 3e-4 on the most quasi-helical design below, and 0.3 on a good one's boundary
    assert qa.three_term < 1e-12 and qp.three_term == pytest.approx(qa.three_term, abs=1e-14)
    assert BZ._orientation(res.surface)[1] == (-1 if flip else 1)


def test_what_it_rests_on_is_checked():
    """A surface in an angle that is not straight, one the field crosses, and a field a current crosses are refused,
    each by its own measure."""
    with pytest.raises(ValueError, match="not a straight-field-line angle"):
        quasisymmetry(_axisymmetric(geometric=True), helicity=(1, 0))
    with pytest.raises(ValueError, match="the field crosses the surface"):
        quasisymmetry(_axisymmetric(shift=1e-3), helicity=(1, 0))
    with pytest.raises(ValueError, match="a current crosses this surface"):
        quasisymmetry(_axisymmetric(scale=0.01), helicity=(1, 0))


# ------------------------------------------------------------------------------------------------ in Boozer angles

def _boozer_route(res, M, n, nth, nph):
    """f_C, f_T and their flux-surface averages at the points of the surface's (nth, nph) grid, from the harmonics of
    |B| in Boozer angles, every field value from torux.reference."""
    from torux.reference import fast64
    F = fast64.Field(res._engine.source)
    t = res.surface
    shift, sign = BZ._orientation(t)
    z = BZ._transform(t, lambda x: F.evaluate(np.asarray(x, float))["B"], nth, nph, shift, sign, half=False)
    (Bmn,), _ = BZ._harmonics(z["theta"], z["zeta"], (z["modB"] * z["J"])[:, None], t.nfp, t.M, t.N)
    Bmn[0, :t.N] = 0.0                                    # m = 0, n < 0 duplicates n > 0
    mm = np.repeat(np.arange(t.M + 1), 2 * t.N + 1).astype(float)
    kk = np.tile(np.arange(-t.N, t.N + 1), t.M + 1) * float(t.nfp)
    b = Bmn.ravel()
    th, ze = z["theta"], z["zeta"]
    out = {k: np.empty(len(th)) for k in ("B", "Bt", "Bz", "Btt", "Btz", "Bzz")}
    for a in range(0, len(th), 2048):
        sl = slice(a, a + 2048)
        ang = th[sl, None] * mm - ze[sl, None] * kk
        c, s = np.cos(ang), np.sin(ang)
        out["B"][sl], out["Bt"][sl], out["Bz"][sl] = c @ b, -(s * mm) @ b, (s * kk) @ b
        out["Btt"][sl], out["Btz"][sl], out["Bzz"][sl] = -(c * mm * mm) @ b, (c * mm * kk) @ b, -(c * kk * kk) @ b
    B, Bt, Bz, Btt, Btz, Bzz = (out[k] for k in ("B", "Bt", "Bz", "Btt", "Btz", "Bzz"))
    iota, Gp = z["iota"], z["G"] + z["iota"] * z["I"]
    N = n * t.nfp
    f_C = -(N * Bt + M * Bz) / B
    Jinv = B ** 2 / Gp                                    # 1 / sqrt(g) in Boozer angles
    Y = iota * Bt + Bz                                    # (B . grad B) / Jinv
    Yt = (2 * B * Bt * Y + B ** 2 * (iota * Btt + Btz)) / Gp
    Yz = (2 * B * Bz * Y + B ** 2 * (iota * Btz + Bzz)) / Gp
    R = np.hypot(z["native"]["x"][..., 0], z["native"]["x"][..., 1]).ravel()
    f_T = R ** 2 * Jinv * (Bt * Yz - Bz * Yt) / B ** 4
    w = z["J"] / B ** 2                                   # sqrt(g) d theta_B d zeta_B per d theta d phi, up to G
    return dict(f_C=f_C, f_T=f_T, sign=sign, x=z["native"]["x"], two_term=float(np.sum(f_C ** 2 * w) / np.sum(w)),
                three_term=float(np.sqrt(np.sum(f_T ** 2 * w) / np.sum(w))),
                scale_C=float(np.max((abs(N * Bt) + abs(M * Bz)) / B)),
                scale_T=float(np.max(R ** 2 * abs(Jinv) * (abs(Bt * Yz) + abs(Bz * Yt)) / B ** 4)))


def _cases(solved_954, solved_qh):
    return {"954": (solved_954, Helicity(1, 0)), QH: (solved_qh, boozer(solved_qh).helicity)}


@pytest.mark.parametrize("design", ["954", QH])
def test_agrees_with_boozer_angles_point_by_point(design, solved_954, solved_qh):
    """At every point of its grid, f_C and f_T as torux computes them in the surface's own angles, with the solver's
    field, are those from the harmonics of |B| in Boozer angles, with the independent evaluator's field. And so are
    their averages."""
    res, hel = _cases(solved_954, solved_qh)[design]
    q = quasisymmetry(res, helicity=hel)
    nth, nph = q.f_C.shape
    ref = _boozer_route(res, hel.m, hel.n, nth, nph)
    # the arrays are at answer.points(theta, phi), and their weighted means are the residuals
    j, k = nth // 3, nph // 5
    assert np.allclose(res.points(q.theta[j], q.phi[k]), ref["x"][j, k], rtol=0, atol=1e-14)
    assert q.two_term == pytest.approx(np.sum(q.f_C ** 2 * q.weight) / np.sum(q.weight), rel=1e-13)
    assert q.three_term == pytest.approx(np.sqrt(np.sum(q.f_T ** 2 * q.weight) / np.sum(q.weight)), rel=1e-13)
    err_C = float(np.max(np.abs(q.f_C.ravel() - ref["f_C"])))
    err_T = float(np.max(np.abs(q.f_T.ravel() - ref["sign"] * ref["f_T"])))
    print(f"\n{design} {hel}: f_C within {err_C:.1e} of {ref['scale_C']:.2e}, f_T within {err_T:.1e} of "
          f"{ref['scale_T']:.2e}, two-term {q.two_term:.12e} against {ref['two_term']:.12e}, three-term "
          f"{q.three_term:.12e} against {ref['three_term']:.12e}")
    # the harmonics' second derivatives, m^2 times their last digits, set the Boozer side's error
    assert err_C < 1e-10 * ref["scale_C"] and err_T < 1e-8 * ref["scale_T"]
    assert abs(q.two_term / ref["two_term"] - 1) < 1e-9 and abs(q.three_term / ref["three_term"] - 1) < 1e-9


# ------------------------------------------------------------------------------------------------ the numbers

def test_quasi_helical_design_is_quasisymmetric_for_its_own_helicity_only(solved_qh):
    """The helicity read from the field is the Boozer spectrum's, and for it the two-term residual of this design,
    whose NonQuasiSymmetricRatio is 5e-10, is 2.4e-7. For any other helicity, a sign of n wrong among them, it is ten
    thousand times larger."""
    b = boozer(solved_qh)
    q = quasisymmetry(solved_qh)
    assert not q.given and q.helicity == helicity_from_spectrum(b) == b.helicity
    assert q.helicity.m == 1 and abs(q.helicity.n) == 1
    assert q.two_term < 1e-6, q.two_term
    for other in [(1, -q.helicity.n), (1, 0), (0, 1)]:
        assert quasisymmetry(solved_qh, helicity=other).two_term > 1e4 * q.two_term, other


def test_values(solved_954, solved_qh):
    """What torux gives, confirmed above against the calculation in Boozer angles."""
    q954 = quasisymmetry(solved_954)
    qqh = quasisymmetry(solved_qh)
    assert q954.helicity == Helicity(1, 0)
    assert q954.two_term == pytest.approx(TWO_954, rel=1e-9) and q954.three_term == pytest.approx(THREE_954, rel=1e-9)
    assert qqh.two_term == pytest.approx(TWO_QH, rel=1e-6) and qqh.three_term == pytest.approx(THREE_QH, rel=1e-9)
    for q in (q954, qqh):
        assert q.accuracy[0] <= 1e-12 * q.two_term + 1e-20 and q.accuracy[1] <= 1e-12 * q.three_term


TWO_954, THREE_954 = 3.682061000537125e-03, 3.2254180269266505e-02
TWO_QH, THREE_QH = 2.40067254196396e-07, 2.759890064806298e-04


@pytest.mark.parametrize("change", ["clockwise", "from the inboard point"])
def test_the_surface_angle_convention_does_not_matter(solved_qh, change):
    """The same surface with its theta running clockwise, or starting from the inboard point, gives the same
    residuals for the same helicity: the helicity is turned into the surface's own angle directions."""
    t = solved_qh.surface
    rc, zs, iota = np.array(t.rc), np.array(t.zs), t.iota
    m = np.arange(t.M + 1)[:, None]
    if change == "clockwise":                   # theta -> -theta
        rc, zs, iota = rc[:, ::-1], -zs[:, ::-1], -iota
    else:                                       # theta -> theta + pi
        rc, zs = rc * (-1.0) ** m, zs * (-1.0) ** m
    u = TorusState(rc, zs, iota, t.flux, t.M, t.N, t.nfp, phi0=t.phi0)
    th = np.linspace(0, 2 * np.pi, 7)
    back = -th if change == "clockwise" else th - np.pi
    assert np.allclose(np.stack(u.geometry_on(back, 0.1 + th)), np.stack(t.geometry_on(th, 0.1 + th)), atol=1e-14)
    shift, sign = BZ._orientation(t)
    assert BZ._orientation(u) == ((shift, -sign) if change == "clockwise" else (np.pi - shift, sign))
    other = SimpleNamespace(status="surface", surface=u, field=solved_qh.field)
    for hel in [boozer(solved_qh).helicity, (1, 0)]:
        a, b = quasisymmetry(solved_qh, helicity=hel), quasisymmetry(other, helicity=hel)
        assert abs(b.two_term - a.two_term) <= 1e-11 * a.two_term + 1e-20, (hel, a.two_term, b.two_term)
        assert abs(b.three_term / a.three_term - 1) < 1e-11


# ------------------------------------------------------------------------------------------------ the interface

def test_print_and_command_line(solved_954, capsys):
    from torux.cli import main
    s = " ".join(str(solved_954.quasisymmetry()).split())
    for part in ("two-term residual", "<f_C^2>", "quasi-axisymmetric", "(M, n) = (1, +0)", "largest harmonic",
                 "three-term residual", "<f_T^2>^(1/2)", "flux-surface rms", "psi = Psi / 2 pi",
                 "helicity_m = 1, helicity_n = 0", "f_T_normalized for helicity (1, 0)", "rests on", "theta straight"):
        assert part in s, part
    assert len(str(solved_954.quasisymmetry()).splitlines()) <= 11
    capsys.readouterr()
    assert main(["solve", example("quasr_0000954.json"), "--helicity", "1,0"]) == 0
    out = capsys.readouterr().out
    assert "two-term residual" in out and "(M, n) = (1, +0): as given" in out
    assert main(["solve", example("quasr_0000954.json"), "--helicity", "1"]) == 1
    assert "--helicity 1: give two integers" in capsys.readouterr().err


def test_refusals(solved_954):
    from torux.api import Result
    with pytest.raises(ValueError, match="no verified surface"):
        quasisymmetry(Result("no_surface", 1.0, 0.0))

    class General:
        rs = None
    with pytest.raises(ValueError, match="not stellarator-symmetric"):
        quasisymmetry(Result("surface", 1.0, 0.0, surface=General()))
    for bad in [(0, 0), (-1, 1), (1.5, 0), "1,0", (1, 2, 3)]:
        with pytest.raises(ValueError, match="helicity"):
            quasisymmetry(solved_954, helicity=bad)
