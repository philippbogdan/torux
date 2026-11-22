"""Convert a VMEX wout and its ESSOS coils to a torux source."""

import argparse
from pathlib import Path

import numpy as np

from torux import FourierCurves, Source, SurfaceCurrent
from torux.source import MU0

VMEX = "757fe28348d959d5d4a50653d25fcca67c8fb941"
UPSTREAM = f"https://github.com/uwplasma/VMEX/tree/{VMEX}"


def from_boundary(data):
    """Convert one field period of equally spaced VMEX samples, starting at theta=phi=0."""
    gamma, field = (np.asarray(data[k]).transpose(2, 1, 0) for k in ("gamma", "B_total"))
    nt, np_, _ = gamma.shape
    nfp = int(data["nfp"])
    theta, phi = np.asarray(data["theta"]), np.asarray(data["phi"])
    tau = 2*np.pi
    np.testing.assert_allclose(theta, tau * np.arange(nt)/nt, atol=1e-14)
    np.testing.assert_allclose(phi, tau * np.arange(np_)/(np_*nfp), atol=1e-14)
    c, s = np.cos(phi), np.sin(phi)
    r, z = gamma[..., 0]*c + gamma[..., 1]*s, gamma[..., 2]
    area = np.sum(r*np.roll(z, -1, axis=0) - z*np.roll(r, -1, axis=0), axis=0)
    if np.all(area < 0):
        rows = (-np.arange(nt)) % nt
        r, z, field = r[rows], z[rows], field[rows]
    elif np.any(area <= 0):
        raise ValueError("VMEX boundary cuts have inconsistent orientation")
    rz = np.stack([r, z], axis=-1)
    geometry = np.fft.fft2(rz, axes=(0, 1))
    m = np.rint(np.fft.fftfreq(nt)*nt).astype(int)
    k = np.rint(np.fft.fftfreq(np_)*np_).astype(int)
    rt, zt = np.fft.ifft2(1j*m[:, None, None]*geometry, axes=(0, 1)).real.transpose(2, 0, 1)
    rp, zp = np.fft.ifft2(1j*nfp*k[None, :, None]*geometry, axes=(0, 1)).real.transpose(2, 0, 1)
    br, bp, bz = field[..., 0]*c + field[..., 1]*s, field[..., 1]*c - field[..., 0]*s, field[..., 2]
    qt = (br*rt + bz*zt)/MU0
    qp = (br*rp + bp*r + bz*zp)/MU0
    values = np.stack([r, z, qt, qp], axis=-1)
    spectra = np.fft.fft2(values, axes=(0, 1))/(nt*np_)
    # One representative per +/- pair in phase m*theta - n*nfp*phi.
    mm, kk = np.meshgrid(m, k, indexing="ij")
    keep = ((mm > 0) | ((mm == 0) & (kk >= 0))) & (np.abs(kk) < np_/2)
    keep &= np.any(np.abs(spectra) > [1e-14, 1e-14, 1e-7, 1e-7], axis=-1)
    modes = np.column_stack([mm[keep], -kk[keep]])
    coeff = 2*spectra[keep]
    coeff[np.all(modes == 0, axis=1)] *= .5
    ccoef, scoef = coeff.real.T, -coeff.imag.T
    return SurfaceCurrent(nfp, modes, ccoef[0], scoef[1], rmns=scoef[0], zmnc=ccoef[1],
                          covariant=np.stack([ccoef[2:], scoef[2:]], axis=1))


def from_coils(path):
    from essos.coils import Coils

    coils = Coils.from_json(str(path))
    samples = np.asarray(coils.gamma)
    t = np.arange(samples.shape[1])/samples.shape[1]
    order = int(coils.order)
    freq = np.fft.rfft(samples, axis=1).transpose(0, 2, 1) / len(t)
    coeff = np.zeros((len(samples), 3, 2*order+1))
    coeff[..., 0] = freq[..., 0].real
    coeff[..., 1::2] = -2*freq[..., 1:order+1].imag
    coeff[..., 2::2] = 2*freq[..., 1:order+1].real
    part = FourierCurves(coeff, np.asarray(coils.currents))
    points, derivatives = part.points(t)
    np.testing.assert_allclose(points, samples, atol=2e-13, rtol=0)
    np.testing.assert_allclose(derivatives, np.asarray(coils.gamma_dash), atol=2e-12, rtol=0)
    return part


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout", type=Path)
    parser.add_argument("coils", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="landreman-paul-qa")
    parser.add_argument("--plasma-current", type=float, help="net toroidal plasma current in A, for provenance")
    args = parser.parse_args()
    from vmex.core.virtual_casing import surface_field_data_from_wout
    from vmex.core.wout import read_wout

    equilibrium = read_wout(str(args.wout))
    vmex = surface_field_data_from_wout(equilibrium, nphi=256, ntheta=256)
    data = {k: np.asarray(getattr(vmex, k)) for k in ("nfp", "gamma", "B_total", "theta", "phi")}
    sheet = from_boundary(data)
    # the toroidal flux inside the boundary: VMEC's phiedge, the last value of phi (its magnitude: torux takes the sign
    # from the field)
    sheet.flux = abs(float(np.asarray(equilibrium.phi)[-1]))
    meta = {"beta": float(equilibrium.betatotal), "edge_iota": float(equilibrium.iotaf[-1])}
    if args.plasma_current is not None:
        meta["plasma_current_A"] = args.plasma_current
    # stellarator symmetric when the sheet's coefficients are (to 1e-9 of their scale) and the coils' field is
    src = Source(args.name, [from_coils(args.coils), sheet], nfp=sheet.nfp, provenance={"upstream": UPSTREAM},
                 meta=meta)
    from torux.io._stellcommon import verify_symmetry
    _, coils_symmetric, _ = verify_symmetry(src.parts[:1], [sheet.nfp], True)
    src.stellsym = bool(coils_symmetric and sheet.symmetric())
    src.save(args.output)


if __name__ == "__main__":
    main()
