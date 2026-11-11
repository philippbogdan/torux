"""The one picture: what it draws is right by code that shares none with torux.plot (SIMSOPT's surface, curves and
Biot-Savart), no coil behind the surface is drawn over it, and it behaves as a function should in a script, a
notebook and a file."""
import copy
import importlib.util
import json
import math
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import torux

from conftest import example

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None
HAVE_SIMSOPT = importlib.util.find_spec("simsopt") is not None
needs_mpl = pytest.mark.skipif(not HAVE_MPL, reason='drawing needs matplotlib: pip install ".[plot]"')
needs_simsopt = pytest.mark.skipif(not (HAVE_MPL and HAVE_SIMSOPT), reason="needs matplotlib and SIMSOPT 1.11.1")
if HAVE_MPL:
    import matplotlib
    matplotlib.use("Agg")                  # no window opens while testing

HSX = dict(file="hsx.json", flux=0.0359221439332229, budget_s=900)
W7X = dict(file="w7x.json", flux=2.217430801869203, budget_s=3600)


# ------------------------------------------------------------------------------------------------ SIMSOPT's side
def simsopt_field(path, nquad=1024):
    """SIMSOPT's BiotSavart on the coils of a file, read by SIMSOPT itself (a QUASR download) or built in SIMSOPT
    from the file's Fourier coefficients (a Fourier-filament JSON file), every curve on ``nquad`` points."""
    import simsopt
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.geo import CurveXYZFourier, RotatedCurve
    q = np.linspace(0.0, 1.0, nquad, endpoint=False)
    with open(path) as fh:
        d = json.load(fh)
    coils = []
    if "simsopt_objs" in d:
        for c in simsopt.load(path)[1]:
            rot, curve = (c.curve, c.curve.curve) if isinstance(c.curve, RotatedCurve) else (None, c.curve)
            base = CurveXYZFourier(q, curve.order)
            base.local_full_x = curve.local_full_x
            new = base if rot is None else RotatedCurve(base, rot._phi, bool(rot.rotmat[2, 2] < 0))
            assert np.allclose(new.gamma()[0], c.curve.gamma()[0], rtol=0, atol=1e-14)   # the same curve
            coils.append(Coil(new, c.current))
    else:
        for c in d["coils"]:
            K = len(c["xc"]) - 1
            base = CurveXYZFourier(q, K)
            base.local_full_x = np.array([v for a in "xyz" for v in
                                          [c[a + "c"][0]] + [w for k in range(1, K + 1) for w in (c[a + "s"][k],
                                                                                                   c[a + "c"][k])]])
            coils.append(Coil(base, Current(c["current_A"])))
    return BiotSavart(coils), coils


def simsopt_sections(surface, phis, thetas):
    """R, Z (len(phis), len(thetas)) of the surface by SIMSOPT's SurfaceRZFourier with its coefficients copied in."""
    from simsopt.geo import SurfaceRZFourier
    T = surface
    sym = not hasattr(T, "rs")
    s = SurfaceRZFourier(nfp=int(T.nfp), stellsym=sym, mpol=int(T.M), ntor=int(T.N),
                         quadpoints_phi=(np.asarray(phis) - float(getattr(T, "phi0", 0.0))) / (2 * np.pi),
                         quadpoints_theta=np.asarray(thetas) / (2 * np.pi))
    s.rc[:, :] = T.rc
    s.zs[:, :] = T.zs
    if not sym:
        s.rs[:, :] = T.rs
        s.zc[:, :] = T.zc
    s.local_full_x = s.get_dofs()
    g = s.gamma()
    return np.hypot(g[..., 0], g[..., 1]), g[..., 2]


def check_against_simsopt(answer, path, fig=None):
    """The sections, |B| on the surface, the coils and the magnetic axis the picture draws, each against SIMSOPT:
    the largest differences, in metres and relative to |B|."""
    from simsopt.geo import CurveRZFourier
    import torux.plot as P
    p = P._picture(answer)
    out = {}
    # the three sections, at the drawn angles. And the figure draws exactly these
    R, Z = simsopt_sections(answer.surface, p.section_phi, p.section_theta)
    out["sections_m"] = max(max(float(np.max(np.abs(p.sections[k][:, 0] - R[k]))),
                                float(np.max(np.abs(p.sections[k][:, 1] - Z[k])))) for k in range(3))
    if fig is not None:
        drawn = [ax.lines[0].get_xydata() for ax in fig.axes[1:4]]
        out["drawn_is_data"] = all(np.array_equal(a, b) for a, b in zip(drawn, p.sections))
    # |B| where the surface is painted
    bs, coils = simsopt_field(path)
    bs.set_points(np.ascontiguousarray(p.B_points))
    ref = np.linalg.norm(bs.B(), axis=1)
    out["field_rel"] = float(np.max(np.abs(p.B_values - ref) / ref))
    out["field_points"] = len(ref)
    # the coils drawn (those with current): SIMSOPT's curves sit on the parameters drawn, t = k / 1024
    live = [c for c in coils if c.current.get_value() != 0]
    out["coils_drawn"] = len(p.lines)
    out["coils_m"] = (float(max(np.max(np.abs(ln[:-1] - c.curve.gamma())) for ln, c in zip(p.lines, live)))
                      if len(live) == len(p.lines) else None)
    # the magnetic axis: the evidence's closed line by SIMSOPT's CurveRZFourier, along SIMSOPT's field
    out["axis_m"] = out["axis_along"] = None
    if p.axis is not None:
        for a in answer.evidence["solver"]["axes"]:
            rc, zs = np.asarray(a["rc"], float), np.asarray(a["zs"], float)
            c = CurveRZFourier(p.section_phi / (2 * np.pi), len(rc) - 1, int(answer.surface.nfp), True)
            c.local_full_x = np.concatenate([rc, zs[1:]])
            g = c.gamma()
            d = float(np.max(np.abs(np.column_stack([np.hypot(g[:, 0], g[:, 1]), g[:, 2]]) - p.axis)))
            if d < 1e-9:
                bs.set_points(np.ascontiguousarray(g))
                B, tng = bs.B(), c.gammadash()
                along = np.linalg.norm(np.cross(tng, B), axis=1) / (np.linalg.norm(tng, axis=1) *
                                                                    np.linalg.norm(B, axis=1))
                out["axis_m"], out["axis_along"] = d, float(along.max())
                break
    return out


# ------------------------------------------------------------------------------------------------ depth
def depth_check(answer, p):
    """Against a depth buffer built here from the surface sampled finer than a pixel (points splatted, not faces
    painted), with a projection written here: how many coil pieces the picture draws over the surface lie behind
    it within their stroke (must be none), how many it hides that lie in front (only where a stroke grazes an edge),
    and on how many pixels the painted surface shows a far side where its near side should be."""
    import torux.plot as P
    cam = p.camera

    def project(x):
        v = np.asarray(x, float) - cam.eye
        d = v @ cam.forward
        return np.stack([(v @ cam.right) / d, (v @ cam.up) / d], -1), d

    wpx, hpx = p.machine[2] * P.DEPTH_DPI, p.machine[3] * P.DEPTH_DPI
    nx, ny = int(wpx), int(hpx)              # Agg cuts its canvas to whole pixels and counts rows from the top
    (x0, x1), (y0, y1) = p.xlim, p.ylim

    def pixel(s):
        return (s[..., 0] - x0) / (x1 - x0) * wpx, ny - (s[..., 1] - y0) / (y1 - y0) * hpx

    # how fine the sample must be: the coarse grid's largest step on screen, cut to under half a pixel
    cx, cy = pixel(project(p.vertices)[0])
    step_t = float(np.max(np.hypot(np.diff(cx, axis=0), np.diff(cy, axis=0))))
    step_p = float(np.max(np.hypot(np.diff(cx, axis=1), np.diff(cy, axis=1))))
    nth = int(math.ceil((len(p.theta) - 1) * step_t / 0.45))
    nph = int(math.ceil((len(p.phi) - 1) * step_p / 0.45))
    z = np.full(nx * ny, np.inf)
    th = np.linspace(0.0, 2 * np.pi, nth, endpoint=False)
    for ph in np.array_split(np.linspace(0.0, 2 * np.pi, nph, endpoint=False), max(1, nph // 256)):
        Rg, Zg = answer.surface.geometry_grid(th, ph)
        X = np.stack([Rg * np.cos(ph)[None, :], Rg * np.sin(ph)[None, :], Zg], -1).reshape(-1, 3)
        s, d = project(X)
        u, v = pixel(s)
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        np.minimum.at(z, j[ok] * nx + i[ok], d[ok])
    z = z.reshape(ny, nx)
    # the coil pieces, rebuilt here in the picture's order, and their strokes as drawn
    a3 = np.concatenate([ln[:-1] for ln in p.lines])
    b3 = np.concatenate([ln[1:] for ln in p.lines])
    sa, da = project(a3)
    sb, db = project(b3)
    assert np.allclose(np.stack([sa, sb], 1), p.seg, rtol=0, atol=1e-12)      # drawn where they are
    near, far = p.depth_range
    width_pt = P.COIL_PT * 0.5 * (near + far) / np.maximum(da, db)
    radius = int(math.ceil(float(np.max(width_pt)) / 2 / 72 * P.DEPTH_DPI + 0.5))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    from scipy.ndimage import minimum_filter
    zdisc = minimum_filter(z, footprint=xx ** 2 + yy ** 2 <= radius ** 2, mode="constant", cval=np.inf)

    def nearest_surface(s):
        u, v = pixel(s)
        i, j = np.clip(np.floor(u).astype(int), 0, nx - 1), np.clip(np.floor(v).astype(int), 0, ny - 1)
        return zdisc[j, i]

    za, zb = nearest_surface(sa), nearest_surface(sb)
    hidden_by_surface = (za < da) | (zb < db)
    front = p.seg_front
    over = int(np.sum(front & hidden_by_surface))
    on_surface = np.isfinite(za) | np.isfinite(zb)
    hidden_wrongly = int(np.sum(~front & ~hidden_by_surface & on_surface))
    # the painted surface against the sampled one, away from the edges where one layer ends over another
    from scipy.ndimage import maximum_filter
    painted = P._depth_buffer(p.faces, p.face_depth, p.xlim, p.ylim, p.machine[2], p.machine[3])
    minor = min(float(np.max(np.hypot(*(s - s.mean(axis=0)).T))) for s in p.sections)
    with np.errstate(invalid="ignore"):
        jump = maximum_filter(z, size=5) - minimum_filter(z, size=5)  # inf next to the background, nan off it
    inner = np.isfinite(painted) & np.isfinite(z) & (jump < 0.25 * minor)
    far_over_near = int(np.sum(painted[inner] - z[inner] > 0.25 * minor))
    return dict(pieces=len(front), front=int(front.sum()), over=over, hidden_wrongly=hidden_wrongly,
                surface_pixels=int(inner.sum()), far_over_near=far_over_near, sample=(nth, nph), radius_px=radius)


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def picture_954(solved_954):
    import torux.plot as P
    return P._picture(solved_954)


def _variant(res, **kw):
    """The same answer with some of its fields replaced (the field engine shared)."""
    out = copy.copy(res)
    out.evidence = copy.deepcopy(res.evidence)
    for k, v in kw.items():
        setattr(out, k, v)
    return out


# ------------------------------------------------------------------------------------------------ what is drawn
@needs_simsopt
def test_quasr_954_is_drawn_where_simsopt_puts_it(solved_954):
    """Sections to 1e-12 m and the coils to 1e-12 m against SIMSOPT's own evaluation at the same angles. |B| on the
    surface against SIMSOPT's BiotSavart to rounding. The axis mark on the evidence's closed field line, which runs
    along SIMSOPT's field."""
    import torux.plot as P
    fig = P.plot(solved_954)
    r = check_against_simsopt(solved_954, example("quasr_0000954.json"), fig)
    assert r["drawn_is_data"]
    assert r["sections_m"] < 1e-12
    assert r["coils_m"] < 1e-12
    assert r["field_rel"] < 5e-14
    assert r["axis_m"] is not None and r["axis_m"] < 1e-12 and r["axis_along"] < 1e-8


@needs_mpl
def test_no_coil_behind_the_surface_is_drawn_over_it_quasr_954(solved_954, picture_954):
    r = depth_check(solved_954, picture_954)
    assert r["front"] > 0 and r["over"] == 0
    assert r["hidden_wrongly"] < 0.01 * r["pieces"]
    assert r["far_over_near"] < 1e-3 * r["surface_pixels"]


@needs_mpl
@pytest.mark.slow
@pytest.mark.parametrize("case", [HSX, W7X], ids=["hsx", "w7x"])
def test_hsx_and_wendelstein_7x(case, tmp_path):
    """Solved from their coils alone (HSX about a minute, Wendelstein 7-X about five), drawn on both pages:
    against SIMSOPT where it is installed, and the depth everywhere."""
    import time
    import torux.plot as P
    t0 = time.time()
    res = torux.solve(example(case["file"]), flux=case["flux"], budget_s=case["budget_s"])
    t1 = time.time()
    assert res.status == "surface"
    out = tmp_path / f"{case['file']}.png"
    assert P.plot(res, out) == out
    print(f"\n{case['file']}: solved in {t1 - t0:.0f} s, drawn to a file {time.time() - t1:.1f} s after")
    p = P._picture(res)
    r = depth_check(res, p)
    assert r["front"] > 0 and r["over"] == 0
    assert r["hidden_wrongly"] < 0.01 * r["pieces"]
    assert r["far_over_near"] < 1e-3 * r["surface_pixels"]
    out = tmp_path / f"{case['file']}-dark.png"
    assert P.plot(res, out, dark=True) == out and out.stat().st_size > 0
    if HAVE_SIMSOPT:
        s = check_against_simsopt(res, example(case["file"]))
        assert s["sections_m"] < 1e-12 and s["coils_m"] < 1e-12 and s["field_rel"] < 5e-14
        assert s["axis_m"] is not None and s["axis_m"] < 1e-12 and s["axis_along"] < 1e-8


# ------------------------------------------------------------------------------------------------ as a function
@needs_mpl
def test_without_a_path_the_figure_comes_back_and_matplotlib_is_as_it_was(solved_954):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import torux.plot as P
    before = (dict(matplotlib.rcParams), matplotlib.get_backend(), plt.get_fignums())
    fig = P.plot(solved_954)
    assert isinstance(fig, Figure) and fig.canvas.manager is None        # not held open by pyplot
    assert (dict(matplotlib.rcParams), matplotlib.get_backend(), plt.get_fignums()) == before
    assert len(fig.axes) == 5                                            # the machine, three sections, the scale


@needs_mpl
def test_a_window_only_where_there_is_one(monkeypatch):
    """A GUI backend gets a window that does not block. A notebook backend that shows what pyplot holds gets the
    figure through pyplot. Inline (the cell's value) and file backends get a figure pyplot does not hold."""
    import torux.plot as P
    for name, want in (("agg", None), ("pdf", None), ("module://matplotlib_inline.backend_inline", None),
                       ("macosx", "window"), ("QtAgg", "window"), ("TkAgg", "window"), ("nbAgg", "pyplot"),
                       ("module://ipympl.backend_nbagg", "pyplot"), ("widget", "pyplot")):
        monkeypatch.setattr(matplotlib, "get_backend", lambda *a, _name=name, **k: _name)
        assert P._shown_by() == want, name


@needs_mpl
@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg"])
def test_with_a_path_the_file_is_written_and_its_path_comes_back(solved_954, tmp_path, ext):
    import matplotlib.pyplot as plt
    import torux.plot as P
    before = (dict(matplotlib.rcParams), plt.get_fignums())
    out = tmp_path / f"surface{ext}"
    assert P.plot(solved_954, out) == out
    data = out.read_bytes()
    if ext == ".png":
        from PIL import Image
        with Image.open(out) as im:
            assert im.size[0] == 1676
            assert im.convert("RGB").getpixel((2, 2)) == (255, 255, 255)
    elif ext == ".pdf":
        assert data.startswith(b"%PDF")
    else:
        assert ET.fromstring(data).tag.endswith("svg")
    assert (dict(matplotlib.rcParams), plt.get_fignums()) == before


@needs_mpl
def test_the_dark_page(solved_954, tmp_path):
    from PIL import Image
    import torux.plot as P
    out = P.plot(solved_954, str(tmp_path / "dark.png"), dark=True)
    with Image.open(out) as im:
        assert im.convert("RGB").getpixel((2, 2)) == (0x0d, 0x11, 0x17)


@needs_mpl
def test_a_renderer_that_cannot_blend_colours_still_paints_the_surface(solved_954, tmp_path, monkeypatch):
    """Some window backends (those drawn with Cairo) have no Gouraud triangles: the surface is then painted flat."""
    from matplotlib.backends.backend_agg import RendererAgg
    from PIL import Image
    import torux.plot as P
    original = RendererAgg._update_methods

    def without_blending(self):
        original(self)

        def refuse(*args, **kwargs):
            raise NotImplementedError

        self.draw_gouraud_triangles = refuse

    monkeypatch.setattr(RendererAgg, "_update_methods", without_blending)
    out = P.plot(solved_954, tmp_path / "flat.png")
    with Image.open(out) as im:
        rgb = np.asarray(im.convert("RGB")).astype(int)
    assert np.sum(rgb[..., 1] - rgb[..., 0] > 25) > 5000           # the green surface is there


@needs_mpl
def test_an_answer_without_a_surface_says_its_status():
    import torux.plot as P
    res = torux.solve(example("d3d_icoils_upper.dat"))
    with pytest.raises(ValueError, match="no_surface"):
        P.plot(res)


@needs_mpl
def test_a_picture_is_png_pdf_or_svg(solved_954, tmp_path):
    import torux.plot as P
    with pytest.raises(ValueError, match=r"\.png, \.pdf or \.svg"):
        P.plot(solved_954, tmp_path / "surface.jpg")


def test_without_matplotlib_one_sentence_names_the_install(monkeypatch):
    import torux.plot as P
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match=r'pip install "\.\[plot\]"'):
        P.plot(types.SimpleNamespace(surface=object(), status="surface"))


@needs_mpl
def test_an_answer_read_back_without_its_source_says_so(solved_954):
    import torux.plot as P
    with pytest.raises(ValueError, match="source"):
        P.plot(_variant(solved_954, _engine=None))


# ------------------------------------------------------------------------------------------------ every source
@needs_mpl
def test_every_kind_of_part_is_drawn_or_named(solved_954):
    """Filaments, polygons, segments and rings are drawn as lines, magnets as marks. A conductor of finite section,
    the 1/R field and a coil without current are left out and named in the line of words."""
    import torux.plot as P
    src = solved_954._engine.source
    ring = np.zeros((1, 3, 3))
    ring[0, 0, 2], ring[0, 1, 1] = 3.0, 3.0
    parts = list(src.parts) + [
        torux.FourierCurves(ring, np.array([0.0])),
        torux.PolylineCurves(np.array([[2.5, 0, -1], [2.5, 0, 1], [2.9, 0, 1], [2.9, 0, -1.0]]), np.array([0, 4]),
                             np.array([1e3])),
        torux.Segments(np.array([[0.0, 0, -1.5]]), np.array([[0.0, 0, 1.5]]), np.array([1e3])),
        torux.Rings(np.array([2.0]), np.array([1.2]), np.array([1e3])),
        torux.Dipoles(np.array([[1.4, 0.0, 0.0], [-1.4, 0.0, 0.0], [0.0, 1.4, 0.3]]), np.ones((3, 3))),
        torux.AxisymPolygons(np.array([[2.0, -0.1], [2.1, -0.1], [2.1, 0.1], [2.0, 0.1]]), np.array([0, 4]),
                             np.array([1e3])),
        torux.ToroidalField(0.1),
    ]
    mixed = torux.Source(id="everything", parts=parts, nfp=src.nfp, stellsym=src.stellsym)
    p = P._picture(_variant(solved_954, source=mixed))
    assert len(p.lines) == 4 + 1 + 1 + 1 and len(p.marks) == 3
    assert p.left_out == ["1 coil without current", "1 axisymmetric conductor", "the 1/R toroidal field"]
    text = P._caption(p, P._font(), 7.88)[0]
    assert "everything" in text and "by the independent check" in text and "not drawn: 1 coil without current" in text
    p.left_out = ["the 1/R toroidal field"]
    assert "not drawn: the 1/R toroidal field" in P._caption(p, P._font(), 7.88)[0]
    assert len(P.plot(_variant(solved_954, source=mixed)).axes) == 5


@needs_mpl
@pytest.mark.slow
@pytest.mark.parametrize("name, share", [("quasr_0219457.json", 0.5), ("quasr_1100073.json", 0.25)])
def test_the_other_quasr_examples(name, share, tmp_path):
    """Their own requests hold no surface (certificates, minutes each).  Each has one at a smaller flux, and it
    draws: 0219457 at half its request (seconds), 1100073 at a quarter (half a minute. At half it comes back
    undetermined within the default budget)."""
    import torux.plot as P
    src = torux.load(example(name))
    res = torux.solve(src, share * src.request["flux_Wb"])
    assert res.status == "surface"
    out = tmp_path / "x.png"
    assert P.plot(res, out) == out


# ------------------------------------------------------------------------------------------------ words, scale, look
@needs_mpl
def test_the_line_of_words_always_fits(picture_954):
    """However long the name and the list of what is not drawn, the line stays one line inside the figure, at no
    less than 7 points, any shortening marked with an ellipsis."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import torux.plot as P
    for name, left in (("quasr_0000954", []), ("w7x", ["20 coils without current"]),
                       ("a" * 300, ["x" * 200, "y" * 200]), ("a" * 40, ["3 coils without current",
                                                                     "2 axisymmetric conductors",
                                                                     "the 1/R toroidal field"])):
        p = copy.copy(picture_954)
        p.name, p.left_out = name, left
        fig = P._figure(p, False)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        t = fig.texts[-1]
        box = t.get_window_extent(canvas.get_renderer())
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width and t.get_fontsize() >= P.CAPTION_MIN_PT
        assert "\n" not in t.get_text()
        assert "of |B| by the independent check" in t.get_text()                 # the claim is never cut
        if len(name) > 100:
            assert "…" in t.get_text() and name[:8] in t.get_text()


@needs_mpl
def test_colour_scale_ticks_never_crowd():
    """At most five ticks on any range, their labels apart by at least one and a half lines."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import torux.plot as P
    font = P._font()
    for lo, hi in ((2.47, 3.31), (0.857, 1.152), (1e-3, 1.7e-3), (-12.0, 30.0), (0.9999, 1.0001), (5.0, 5.3)):
        t, labels = P._ticks(lo, hi, 1.55, font, along=True)
        assert 1 <= len(t) <= 5 and np.all((t >= lo) & (t <= hi))
        fig = Figure(figsize=(3, 1), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes((0.1, 0.5, 1.55 / 3, 0.1))
        ax.set_xlim(lo, hi)
        ax.set_xticks(t, labels, fontsize=P.LABEL_PT, fontfamily=font)
        canvas.draw()
        boxes = [lab.get_window_extent(canvas.get_renderer()) for lab in ax.get_xticklabels()]
        gaps = [b.x0 - a.x1 for a, b in zip(boxes[:-1], boxes[1:])]
        assert all(g >= 1.5 * P.LABEL_PT * fig.dpi / 72 - 1 for g in gaps), (lo, hi, labels, gaps)


@needs_mpl
def test_an_answer_not_verified_is_never_shown_as_one(solved_954):
    import torux.plot as P
    ev = copy.deepcopy(solved_954.evidence)
    ev["independent"]["ok"] = False
    ev["unverified"] = "the independent verification refused the torus"
    p = P._picture(_variant(solved_954, status="surface_unverified", evidence=ev))
    text = P._caption(p, P._font(), 7.78)[0]
    assert "not verified: the independent verification refused the torus" in text
    assert "by the independent check" not in text
    ev = copy.deepcopy(solved_954.evidence)
    del ev["independent"]                                    # solved with verify=False
    text = P._caption(P._picture(_variant(solved_954, evidence=ev)), P._font(), 7.78)[0]
    assert "solver's own check, not checked independently" in text


@needs_mpl
def test_the_colour_scale_is_one_green_in_even_steps_on_both_pages():
    """Constant hue (the repository's green, OKLCH 173.5 degrees), lightness monotone, every step the same size in
    OKLab within 10 per cent, and both ends apart from the page."""
    import torux.plot as P

    def oklab(rgb):
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]])
        m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]])
        return np.cbrt(lin @ m1.T) @ m2.T

    for dark, page in ((False, 1.0), (True, 0.176)):
        lab = oklab(P._ramp(dark))
        hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
        assert np.all(np.abs(hue - 173.5) < 1.5)
        L = lab[:, 0]
        assert np.all(np.diff(L) > 0) if dark else np.all(np.diff(L) < 0)
        steps = np.linalg.norm(np.diff(lab, axis=0), axis=1)
        assert steps.min() > 0.9 * steps.mean() and steps.max() < 1.1 * steps.mean()
        assert min(abs(L[0] - page), abs(L[-1] - page)) > 0.05
