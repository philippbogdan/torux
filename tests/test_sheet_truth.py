"""Exact cases for a source with a current sheet (tests/data/sheet_truth, from the fixtures of
work/t040-truth): coils whose vacuum flux surfaces released torux 0.3.0 solved and verified, a truth surface at
psi0 taken as the boundary Gamma, and on it the sheet K = n x B / mu0.  Outside Gamma the coils and the sheet make
exactly the coils' field (virtual casing), so the sheet's own field vanishes there, and torux.solve at an outer flux
must return the vacuum truth surface."""
import json
import os

import numpy as np
import pytest

import torux
from torux.field import FieldEngine

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sheet_truth")
CASES = sorted(d for d in os.listdir(HERE)) if os.path.isdir(HERE) else []


def case(name, form="covariant"):
    root = os.path.join(HERE, name)
    meta = json.load(open(os.path.join(root, "case.json")))
    src = torux.load(os.path.join(root, f"B_{form}.json"))
    part = next(p for p in src.parts if p.kind == "surface_current")
    if part.flux is None:
        part.flux = meta["truth"]["psi0"]["flux_A_Wb"]
    truth = json.load(open(os.path.join(root, "truth_psi1.json")))
    return src, part, meta, truth


def surface_points(rc, zs, nfp, theta, phi):
    rc, zs = np.asarray(rc), np.asarray(zs)
    M, N = rc.shape[0] - 1, (rc.shape[1] - 1) // 2
    m = np.arange(M + 1)[:, None, None]
    n = np.arange(-N, N + 1)[None, :, None]
    ang = m * theta[None, None, :] - n * nfp * phi[None, None, :]
    R = np.sum(rc[:, :, None] * np.cos(ang), axis=(0, 1))
    Z = np.sum(zs[:, :, None] * np.sin(ang), axis=(0, 1))
    return R, Z


@pytest.mark.parametrize("name", CASES)
def test_the_sheet_has_no_field_outside_its_boundary(name):
    """Virtual casing without plasma current: outside Gamma the sheet's field is zero, to the truth surface's own
    B.n (about 1e-15) and the sheet's series."""
    src, part, meta, truth = case(name)
    if meta.get("filament_current_A", 0.0) != 0.0:
        pytest.skip("a case with plasma current: the sheet's field outside is the filament's")
    sheet = FieldEngine(torux.Source("sheet", [part], nfp=src.nfp))
    coils = FieldEngine(torux.Source("coils", [p for p in src.parts if p.kind != "surface_current"], nfp=src.nfp))
    rng = np.random.default_rng(0)
    th, ph = rng.uniform(0, 2 * np.pi, 24), rng.uniform(0, 2 * np.pi, 24)
    R, Z = surface_points(truth["rc"], truth["zs"], src.nfp, th, ph)
    x = np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)
    ratio = np.linalg.norm(np.asarray(sheet.B(x)), axis=1) / np.linalg.norm(np.asarray(coils.B(x)), axis=1)
    assert ratio.max() < 1e-14


@pytest.mark.slow
@pytest.mark.parametrize("name", CASES)
def test_solve_returns_the_vacuum_truth_surface(name):
    src, part, meta, truth = case(name)
    s = torux.solve(src, flux=truth["flux_A_Wb"], budget_s=3600)
    assert s.status == "surface", str(s)
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    ph = np.linspace(0, 2 * np.pi / src.nfp, 32, endpoint=False)
    TH, PH = (a.ravel() for a in np.meshgrid(th, ph, indexing="ij"))
    R1, Z1 = surface_points(s.surface.rc, s.surface.zs, src.nfp, TH, PH)
    R0, Z0 = surface_points(truth["rc"], truth["zs"], src.nfp, TH, PH)
    assert np.max(np.hypot(R1 - R0, Z1 - Z0)) < TRUTH_POSITION_M
    assert abs(s.iota - truth["iota"]) < TRUTH_IOTA
    assert abs(s.flux - truth["flux_A_Wb"]) < 1e-14 * truth["flux_A_Wb"]


TRUTH_POSITION_M = 1e-12
TRUTH_IOTA = 1e-12
