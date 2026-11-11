"""The VMEC input an answer saves, in the hands of the codes it is written for.  VMEC++ reads it point for point and
runs a file cut to an ordinary resolution to the same volume and rotational transform. DESC reads it.  SIMSOPT's
reading is in test_answer.py.  Each test skips where its code is not installed (VMEC++ and DESC install on Linux)."""
import warnings

import numpy as np
import pytest

from conftest import example

import torux


@pytest.fixture(scope="module")
def answer():
    return torux.solve(example("quasr_0000954.json"))


def _angles(k=500, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, k), rng.uniform(0, 2 * np.pi, k)


def test_vmecpp_reads_the_file_and_runs_a_cut_one(answer, tmp_path):
    vmecpp = pytest.importorskip("vmecpp")
    full, cut = tmp_path / "input.q954", tmp_path / "input.q954cut"
    answer.save(str(full))
    moved = answer.save(str(cut), mpol=9, ntor=8)
    th, ph = _angles()
    want = np.asarray(answer.points(th, ph))
    # the whole file, as VMEC++ holds it: R = sum rbc[m, n + ntor] cos(m theta - n nfp phi), summed here by hand
    vi = vmecpp.VmecInput.from_file(str(full))
    rbc, zbs = np.asarray(vi.rbc), np.asarray(vi.zbs)
    m = np.arange(rbc.shape[0])[:, None]
    n = (np.arange(rbc.shape[1]) - int(vi.ntor))[None, :]
    ang = m[None] * th[:, None, None] - n[None] * int(vi.nfp) * ph[:, None, None]
    R, Z = np.sum(rbc[None] * np.cos(ang), axis=(1, 2)), np.sum(zbs[None] * np.sin(ang), axis=(1, 2))
    got = np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)
    far = float(np.max(np.abs(got - want)))
    assert far < 1e-13                                   # 1.1e-15 m measured
    assert abs(float(vi.phiedge)) == answer.flux and int(vi.nfp) == answer.nfp
    # the cut file, run: a vacuum equilibrium inside torux's surface has torux's volume and rotational transform
    w = vmecpp.run(vmecpp.VmecInput.from_file(str(cut))).wout
    iota_edge = float(np.asarray(w.iotaf)[-1])
    print(f"\nVMEC++ reads the whole file to {far:.1e} m. Run on the file cut to MPOL 9, NTOR 8 (no point moved by "
          f"more than {moved:.1e} m): volume {float(w.volume_p):.9f} m^3 against {answer.volume:.9f}, minor radius "
          f"{float(w.Aminor_p):.9f} m against {answer.minor_radius:.9f}, edge rotational transform {iota_edge:.6f} "
          f"against {answer.iota:.6f}")
    assert moved < 2e-5                                  # 8.9e-6 m measured
    assert abs(float(w.volume_p) / answer.volume - 1) < 1e-6          # 2.4e-9 measured
    assert abs(float(w.Aminor_p) / answer.minor_radius - 1) < 1e-6    # 1.5e-9 measured
    assert abs(iota_edge / answer.iota - 1) < 2e-3                    # 4.6e-4 measured, at VMEC++'s 51 surfaces


def test_desc_reads_the_file(answer, tmp_path):
    pytest.importorskip("desc")
    from desc.geometry import FourierRZToroidalSurface
    from desc.grid import Grid
    full = tmp_path / "input.q954"
    answer.save(str(full))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # DESC warns that it turns the poloidal angle round
        s = FourierRZToroidalSurface.from_input_file(str(full))
    th, ph = _angles()
    want = np.asarray(answer.points(th, ph))
    d = s.compute(["R", "Z", "phi"], grid=Grid(np.column_stack([np.ones_like(th), -th, ph]), sort=False))
    got = np.stack([d["R"] * np.cos(d["phi"]), d["R"] * np.sin(d["phi"]), d["Z"]], -1)
    far = float(np.max(np.abs(np.asarray(got) - want)))
    print(f"\nDESC reads the whole file to {far:.1e} m, at its own poloidal angle, which runs the other way")
    assert int(s.NFP) == answer.nfp
    assert far < 1e-8                                    # 9.3e-10 m measured: DESC's converter keeps eight decimals


def test_a_live_desc_coil_set_gives_the_surface_of_the_same_coils_from_their_file():
    pytest.importorskip("desc")
    from desc.coils import CoilSet, FourierXYZCoil
    file = torux.load(example("quasr_0000954.json"))
    part = file.parts[0]                                  # torux.FourierCurves: coef[i, a] = [c0, s1, c1, ..., sK, cK]
    K = (part.coef.shape[2] - 1) // 2
    coils = []
    for i in range(part.coef.shape[0]):
        xyz = [np.concatenate([part.coef[i, a, 1::2][::-1], part.coef[i, a, :1], part.coef[i, a, 2::2]])
               for a in range(3)]                         # DESC: sin(|n| s) for n < 0, then cos(n s) for n >= 0
        coils.append(FourierXYZCoil(current=float(part.current[i]), X_n=xyz[0], Y_n=xyz[1], Z_n=xyz[2],
                                    modes=np.arange(-K, K + 1)))
    live = torux.solve(torux.load(CoilSet(*coils), nfp=2), flux=0.05)
    same = torux.solve(example("quasr_0000954.json"), flux=0.05)
    print(f"\na live DESC CoilSet: {live.status}, iota {live.iota!r}, and the same coils from their file: "
          f"iota {same.iota!r}")
    assert live.status == "surface" and same.status == "surface"
    assert abs(live.iota / same.iota - 1) < 1e-9
