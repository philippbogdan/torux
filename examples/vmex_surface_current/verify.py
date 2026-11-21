"""Compare Torux to VMEX on a fixed public equilibrium and locate the edge island."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import root_scalar

import torux
from torux.field import FieldEngine
from torux.surface_trace import _trace

HERE = Path(__file__).resolve().parent
TAU = 2*np.pi


def crossings(engine, seed, turns, *, per_turn=1, steps=64):
    targets = TAU * np.arange(1, turns*per_turn + 1) / per_turn
    points, stop, reason = _trace(engine, np.asarray(seed), 0., targets, TAU/steps)
    if reason is not None:
        raise RuntimeError(f"line stopped at R, phi, Z = {stop}: {reason}")
    return points


def centre(engine, edge, *, steps):
    """Find the outboard symmetry-line orbit that hits the midplane after one toroidal turn."""
    def midplane_error(r):
        return crossings(engine, [r, 0.], 1, steps=steps)[0, 1]

    sol = root_scalar(midplane_error, x0=edge + .0048, x1=edge + .0052, xtol=1e-10)
    if not sol.converged:
        raise RuntimeError("the outboard orbit did not converge")
    return np.array([sol.root, 0.])


def verify():
    refs = json.loads((HERE/"reference.json").read_text())
    result = {}
    for case in ("current", "beta1"):
        src = torux.load(HERE/(case + ".json"))
        sheet = src.parts[1]
        edge = float(np.sum(sheet.rmnc))
        points, ref = (np.asarray(refs[case][k]) for k in ("points", "B"))
        source = torux.Source("sheet only", [sheet], nfp=src.nfp)
        field = np.asarray(FieldEngine(source).B(points))
        finer = np.asarray(FieldEngine(source, sheet_order=12, sheet_samples=512).B(points))
        error = np.linalg.norm(field - ref, axis=-1)
        assert error.max() < 2e-11, error
        case_result = result[case] = {
            "vmex_probe_max_abs_error_T": float(error.max()),
            "probe_refinement_change_T": float(np.linalg.norm(field - finer, axis=-1).max()),
            "boundary_R_m": edge,
            "beta": src.meta["beta"],
            "edge_iota": src.meta["edge_iota"],
        }
        engine = FieldEngine(src, nodes_np=True)
        if case == "current":
            root = centre(engine, edge, steps=64)
            refined = FieldEngine(src, nodes_np=True, sheet_order=12, sheet_samples=512)
            refined_root = centre(refined, edge, steps=128)
            orbit = crossings(engine, root, 2, per_turn=32)
            coords = np.vstack([root, orbit])
            theta = np.linspace(0., 2*TAU, len(coords))
            slices = engine.sheets[0].boundary.mean(axis=0)
            slice_centre = slices[np.rint(theta * src.nfp * len(slices) / TAU).astype(int) % len(slices)]
            angle = np.unwrap(np.arctan2(coords[:, 1] - slice_centre[:, 1], coords[:, 0] - slice_centre[:, 0]))
            winding = angle[-1] - angle[0]
            closure = np.linalg.norm(orbit[-1] - root)
            min_clearance = min(engine.sheets[0].clearance(r, z, phi) for (r, z), phi in zip(coords, theta))
            assert np.linalg.norm(orbit[31] - root) > 0.01
            assert abs(winding/TAU - 1) < 1e-5
            assert min_clearance > 0 and closure < 1e-7
            assert abs(root[0] - edge - .004975) < 3e-5
            assert abs(refined_root[0] - root[0]) < 1e-6
            h = 2e-6
            monodromy = np.column_stack([
                (crossings(engine, root + perturb, 2)[-1] - crossings(engine, root - perturb, 2)[-1])/(2*h)
                for perturb in h*np.eye(2)
            ])
            determinant = np.linalg.det(monodromy)
            trace = np.trace(monodromy)
            assert abs(determinant - 1.) < 3e-4 and 1.75 < trace < 1.90
            case_result.update(centre_offset_m=float(root[0]-edge), centre_R_m=float(root[0]),
                               refined_centre_offset_m=float(refined_root[0]-edge), closure_m=float(closure),
                               min_sampled_clearance_m=float(min_clearance), orbit_winding=float(winding/TAU),
                               map_trace=float(trace), map_det=float(determinant))
            wide_offset = .00935
            wide = torux.poincare(src, planes=[0.], starts=[(edge + wide_offset, 0.)], turns=15)
            assert wide.left[0]
            case_result.update(wider_launch_offset_m=wide_offset,
                               wider_launch_stop_turns=float(wide.left_at[0, 1]/TAU))
        else:
            start = np.array([edge + result["current"]["centre_offset_m"], 0.])
            delta = np.linalg.norm(crossings(engine, start, 2)[-1] - start)
            assert delta > 0.01
            case_result["two_turn_displacement_m"] = float(delta)
    return result


def plot_island(result, stem):
    import matplotlib.pyplot as plt

    src = torux.load(HERE / "current.json")
    engine = FieldEngine(src, nodes_np=True)
    edge = result["current"]["boundary_R_m"]
    axis = result["current"]["centre_R_m"]
    start = np.array([axis + .001, 0.])
    orbit = crossings(engine, start, 34, steps=48)
    island = np.vstack([start, orbit[1::2]])
    np.savetxt(stem.with_suffix(".csv"), island, delimiter=",", header="R_m,Z_m", comments="")

    sheet = src.parts[1]
    phase = np.linspace(-.2, .2, 401)[:, None] * sheet.modes[:, 0]
    c, s = np.cos(phase), np.sin(phase)
    section = np.column_stack([c @ sheet.rmnc + s @ sheet.rmns, c @ sheet.zmnc + s @ sheet.zmns])
    mask = (section[:, 0] > edge - .006) & (abs(section[:, 1]) < .05)
    cut = section[mask][np.argsort(section[mask, 1])]
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout="constrained")
    ax.plot(1e3*(cut[:, 0] - edge), 1e3*cut[:, 1], color="#72818b", lw=1.15, label="VMEX boundary")
    ax.scatter(1e3*(island[:, 0] - edge), 1e3*island[:, 1], s=21, color="#086a72", label="Torux, every 2 turns")
    ax.scatter(1e3*(axis-edge), 0, marker="+", s=64, color="#934320", label="Island centre")
    span = np.max(abs(island[:, 1]))*1e3
    ax.set(xlabel=r"$R - R_{\rm boundary}$ (mm)", ylabel="$Z$ (mm)", xlim=(-.45, 7.25),
           ylim=(-1.35*span, 1.35*span))
    ax.set_title(r"VMEX QA  ·  4 kA  ·  $\phi = 0$", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8, borderaxespad=.1)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save a JSON copy of the results")
    parser.add_argument("--plot", type=Path, help="save the outboard island section as PDF, PNG and CSV")
    args = parser.parse_args()
    result = verify()
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")
    if args.plot:
        plot_island(result, args.plot)
