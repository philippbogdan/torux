"""Compare plasma-only field vectors and quadrature refinement at exterior probes."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

import torux
from torux.field import FieldEngine


def compare(source, probes):
    parts = [p for p in source.parts if isinstance(p, torux.SurfaceCurrent)]
    if not parts:
        raise ValueError("the source must contain a SurfaceCurrent")
    sheet = torux.Source(source.id, parts, nfp=source.nfp)
    points = np.asarray(probes["points"], dtype=float)
    reference = np.asarray(probes["B"], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or reference.shape != points.shape or not np.isfinite(reference).all()):
        raise ValueError("points and B must each contain the same nonzero number of finite three-component vectors")
    field = np.asarray(FieldEngine(sheet).B(points))
    finer = np.asarray(FieldEngine(sheet, sheet_order=16, sheet_eta=3.0).B(points))
    return np.column_stack([points, reference, field, finer,
                            np.linalg.norm(field - reference, axis=1),
                            np.linalg.norm(finer - reference, axis=1),
                            np.linalg.norm(finer - field, axis=1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="a torux JSON source containing a sheet, optionally with coils")
    parser.add_argument("probes", type=Path, help="JSON arrays points (Cartesian m) and B (Cartesian T)")
    parser.add_argument("--case", help="key to select inside a file containing multiple probe sets")
    args = parser.parse_args()
    try:
        data = json.loads(args.probes.read_text())
        if args.case is not None:
            data = data[args.case]
        rows = compare(torux.load(args.source), data)
    except (KeyError, ValueError, OSError) as e:
        parser.error(str(e))
    writer = csv.writer(sys.stdout)
    writer.writerow(["x_m", "y_m", "z_m", "Bx_ref_T", "By_ref_T", "Bz_ref_T", "Bx_T", "By_T", "Bz_T",
                     "Bx_refined_T", "By_refined_T", "Bz_refined_T", "error_T", "refined_error_T", "change_T"])
    writer.writerows(rows)


if __name__ == "__main__":
    main()
