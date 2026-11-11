"""The one evaluator every solver's answer goes through.  It shares no code with any solver.

    python results/evaluator/evaluate.py ANSWER.json [...] --coils COILS_DIR --out DIR
    python results/evaluator/evaluate.py --answers DIR --coils COILS_DIR --out DIR

For each answer that returned a surface:

- |B.n| / |B| at 4,000 random points (theta, phi uniform on the whole torus, the same points for every answer),
  with B the Biot-Savart field of the design's coils (the coefficients every solver was given) and n the surface's
  normal from its own Fourier series.  The precise tier computes everything (coil nodes, surface points and
  derivatives, the field and its sums) in x87 extended precision (numpy.longdouble on x86-64, unit roundoff
  5.4e-20), with each coil's trapezoid rule doubled until the halved rule agrees to 1e-10 of |B| (the rule is
  spectrally accurate on a smooth closed curve, so the error of the kept rule is ~1e-20).  What it reports is the
  surface's own error to about 1e-18, far below double-precision rounding (1.1e-16).  The same computation in
  float64 is reported beside it (`f64_*`).
- The toroidal flux the surface encloses: the loop integral of A round theta at four random phi (float64, the
  theta rule doubled until it changes by under 1e-14), against the requested flux.
- The answer's own check points (its library's evaluation of the surface) against this evaluation of the form it
  was written in: a wrong conversion would show here.

An answer is a JSON file: `design`, `solver`, `status` ('surface' or 'none') and, for a surface, `surface` in one of
the two forms of forms.py.  COILS_DIR holds quasr_<design>.json: the coils as Fourier filaments (`coils`: a list of
{xc, xs, yc, ys, zc, zs, current_A}) and `requested_toroidal_flux_Wb`.  judge.py beside this file writes both from a
torux solve.

The precise tier needs a machine whose long double is extended precision (x86-64); elsewhere pass --f64-only.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forms import Surface, check_points, ld_pi  # noqa: E402

NPTS, SEED = 4000, 20261107
MU0_4PI = "1e-7"            # SIMSOPT's and QUASR's mu0 / 4 pi (exact)
QUAD_TOL = 1e-10            # a coil's rule is kept when its halved rule agrees to this fraction of |B|
N_START, N_MAX = 256, 1 << 17


def precise_dtype():
    eps = float(np.finfo(np.longdouble).eps)
    return np.longdouble if eps < 1e-18 else None


def points(npts=NPTS, seed=SEED):
    rng = np.random.default_rng(seed)
    return 2 * np.pi * rng.random(npts), 2 * np.pi * rng.random(npts)


class Coils:
    def __init__(self, coils, dtype):
        self.dt = dtype
        self.pi = ld_pi(dtype)
        self.mu = dtype(MU0_4PI)
        self.coils = []
        for c in coils:
            K = len(c["xc"]) - 1
            cos = np.array([[dtype(repr(float(c[a + "c"][k]))) for a in "xyz"] for k in range(K + 1)], dtype=dtype)
            sin = np.array([[dtype(repr(float(c[a + "s"][k]))) if k > 0 else dtype(0) for a in "xyz"]
                            for k in range(K + 1)], dtype=dtype)
            self.coils.append((K, cos, sin, dtype(repr(float(c["current_A"])))))
        self.n_used = [0] * len(self.coils)

    def nodes(self, i, N):
        K, cos, sin, I = self.coils[i]
        dt = self.dt
        k = np.arange(K + 1)
        r = (np.outer(np.arange(N), k) % N).astype(dt)            # 2 pi k j / N, reduced exactly
        ang = (2 * self.pi) * r / dt(N)
        C, S = np.cos(ang), np.sin(ang)
        w = (2 * self.pi) * k.astype(dt)
        g = C @ cos + S @ sin                                       # (N, 3)
        dg = S @ (-w[:, None] * cos) + C @ (w[:, None] * sin)       # d gamma / dt
        return g, dg, I

    def _terms(self, x, g, dg, want):
        """Per-point sums over the nodes (full rule and its even half) of B or A integrands, chunked."""
        dt = self.dt
        P, N = len(x), len(g)
        full = np.zeros((P, 3), dtype=dt)
        half = np.zeros((P, 3), dtype=dt)
        step = max(1, int(2_000_000 // N))
        gx, gy, gz = (np.ascontiguousarray(g[:, j]) for j in range(3))
        dx, dy, dz = (np.ascontiguousarray(dg[:, j]) for j in range(3))
        for a in range(0, P, step):
            xs = x[a:a + step]
            rx = xs[:, 0:1] - gx[None, :]
            ry = xs[:, 1:2] - gy[None, :]
            rz = xs[:, 2:3] - gz[None, :]
            r2 = rx * rx + ry * ry + rz * rz
            if want == "B":
                inv = 1 / (r2 * np.sqrt(r2))
                comps = ((dy * rz - dz * ry) * inv, (dz * rx - dx * rz) * inv, (dx * ry - dy * rx) * inv)
            else:
                inv = 1 / np.sqrt(r2)
                comps = (dx * inv, dy * inv, dz * inv)
            for j, t in enumerate(comps):
                full[a:a + step, j] = np.sum(t, axis=1)
                half[a:a + step, j] = np.sum(t[:, ::2], axis=1)
        return full / dt(N), half / dt(N // 2)

    def field(self, x, want="B", ref=None, tol=QUAD_TOL, fixed=None):
        """B (or A) at points x, each coil's rule doubled until its halved rule agrees to tol of |ref|
        (|B| at x from a first pass when not given).  fixed: use these node counts instead."""
        dt = self.dt
        x = np.asarray(x).astype(dt)
        if ref is None and fixed is None:
            ref = np.linalg.norm(self.field(x, want, fixed=[N_START] * len(self.coils)), axis=1)
        total = np.zeros((len(x), 3), dtype=dt)
        for i in range(len(self.coils)):
            N = fixed[i] if fixed is not None else N_START
            while True:
                g, dg, I = self.nodes(i, N)
                full, half = self._terms(x, g, dg, want)
                if fixed is not None:
                    break
                err = np.max(np.linalg.norm(full - half, axis=1) * abs(self.mu * I) / ref)
                if err < tol or N >= N_MAX:
                    break
                N *= 2
            self.n_used[i] = N
            total += (self.mu * I) * full
        return total


def judge_bn(coils_json, surface, dtype, th, ph):
    S = Surface(surface, dtype)
    x, xt, xp = S.eval(th, ph)
    nrm = np.cross(xt, xp)
    C = Coils(coils_json, dtype)
    B = C.field(x, "B")
    num = np.abs(np.sum(B * nrm, axis=1))
    den = np.linalg.norm(B, axis=1) * np.linalg.norm(nrm, axis=1)
    bn = (num / den).astype(np.float64)
    nn = np.linalg.norm(nrm, axis=1).astype(np.float64)
    return bn, C.n_used, float(nn.min() / np.median(nn))


def judge_flux(coils_json, surface, req, nloops=4, seed=SEED + 1):
    S = Surface(surface, np.float64)
    C = Coils(coils_json, np.float64)
    phis = 2 * np.pi * np.random.default_rng(seed).random(nloops)
    vals, ntheta = [], []
    for ph in phis:
        prev = None
        n = 128
        while True:
            th = 2 * np.pi * np.arange(n) / n
            x, xt, _ = S.eval(th, np.full(n, ph))
            A = C.field(x, "A", tol=1e-13)
            f = float(np.sum(A * xt) * (2 * np.pi / n))
            if prev is not None and abs(f - prev) <= 1e-14 * abs(f):
                break
            if n >= 8192:
                break
            prev, n = f, 2 * n
        vals.append(abs(f))
        ntheta.append(n)
    vals = np.array(vals)
    rel = vals / req - 1.0
    return dict(flux_Wb=float(vals.mean()), flux_rel_err=float(rel[np.argmax(np.abs(rel))]),
                flux_spread_rel=float((vals.max() - vals.min()) / req), flux_ntheta=ntheta,
                flux_phis=phis.tolist())


def evaluate(coils_rec, answer, precise=True):
    out = dict(design=answer["design"], solver=answer["solver"], status=answer["status"])
    if answer["status"] != "surface":
        return out
    surf = answer["surface"]
    t0 = time.time()
    th, ph = points()
    if answer.get("check_points"):
        out["check_points_max_m"] = check_points(surf, answer["check_points"])
    out["n_modes"] = len(surf["m"])
    bn64, n64, nmin = judge_bn(coils_rec["coils"], surf, np.float64, th, ph)
    out.update(f64_bn_max=float(bn64.max()), f64_bn_median=float(np.median(bn64)), normal_min_over_median=nmin)
    if precise:
        dt = precise_dtype()
        bn, nq, _ = judge_bn(coils_rec["coils"], surf, dt, th, ph)
        k = int(np.argmax(bn))
        out.update(bn_max=float(bn.max()), bn_median=float(np.median(bn)), bn_p90=float(np.quantile(bn, 0.9)),
                   bn_argmax_theta_phi=[float(th[k]), float(ph[k])], coil_nodes=nq,
                   precision_eps=float(np.finfo(dt).eps))
    out.update(judge_flux(coils_rec["coils"], surf, coils_rec["requested_toroidal_flux_Wb"]))
    out["eval_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--answers", help="a directory of answer JSON files")
    ap.add_argument("--coils", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--f64-only", action="store_true", help="skip the precise tier")
    a = ap.parse_args()
    if not a.f64_only and precise_dtype() is None:
        sys.exit(f"long double here has eps {np.finfo(np.longdouble).eps:.1e}: not extended precision; pass --f64-only")
    files = [Path(f) for f in a.files] or sorted(Path(a.answers).glob("*.json"))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = out_dir / f.name
        raw = f.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if dst.exists():   # kept only if it judged these exact answer bytes; a replaced answer is judged again
            try:
                if json.loads(dst.read_text()).get("answer_sha256") == sha:
                    continue
            except json.JSONDecodeError:
                pass
        ans = json.loads(raw)
        rec = json.loads((Path(a.coils) / f"quasr_{ans['design']}.json").read_text())
        try:
            res = evaluate(rec, ans, precise=not a.f64_only)
        except Exception as e:  # noqa: BLE001
            res = dict(design=ans["design"], solver=ans["solver"], status=ans["status"], eval_error=repr(e))
        res["answer_sha256"] = sha
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(res) + "\n")
        tmp.replace(dst)
        print(f.name, {k: res.get(k) for k in ("bn_max", "f64_bn_max", "flux_rel_err", "eval_s", "eval_error")},
              flush=True)


if __name__ == "__main__":
    main()
