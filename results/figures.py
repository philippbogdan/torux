"""The results figures, drawn from compare300.csv beside this file.

    python results/figures.py            # writes visuals/accuracy-{light,dark}.svg
                                         # and, once several tools are timed on one processor, visuals/time-*.svg

accuracy   every coil set of the 300-coil comparison (compare300.csv) as one dot per tool: the largest |B.n|/|B| on the
           surface the tool returned, measured by the common evaluator.
time       every coil set of the comparison as one dot per tool: seconds until the tool had the surface it returned.
refinement how the error falls as a surface is refined, one line a coil set (refinement.csv).

A tool appears as soon as compare300.csv holds its 300 rows. Nothing else needs to change.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "visuals")

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", rule="#d1d9e0", torux="#12a184", torux_text="#0b7d66", other="#8c959f",
                  band="#f0f3f6", none="#1f2328", open="#d8dee4", page="#ffffff"),
    "dark": dict(ink="#f0f6fc", muted="#9198a1", rule="#3d444d", torux="#1fe0b4", torux_text="#3fe6c0", other="#7d8590",
                 band="#161b22", none="#f0f6fc", open="#30363d", page="#0d1117"),
}
LABELS = {"torux": "torux", "simsopt-boozer-exact": "SIMSOPT BoozerExact", "simsopt-boozer-ls": "SIMSOPT BoozerLS",
          "simsopt-qfm": "SIMSOPT QFM", "desc": "DESC", "vmecpp": "VMEC++", "quasr-stored": "QUASR\u2019s stored surface"}


def read_compare():
    with open(os.path.join(HERE, "compare300.csv")) as fh:
        rows = list(csv.DictReader(fh))
    tools = {}
    for r in rows:
        tools.setdefault(r["solver"], []).append(r)
    return tools


def median(v):
    v = sorted(v)
    n = len(v)
    return float("nan") if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def swarm(xs, r, half):
    """Vertical offsets for dots at pixel positions xs (radius r) so that they do not cover each other, within
    +-half. Where there is no room left the dot goes to the least crowded place."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    placed, ys = [], [0.0] * len(xs)
    step = 2 * r * 0.92
    cand = [0.0]
    k = 1
    while k * step <= half:
        cand += [k * step, -k * step]
        k += 1
    for i in order:
        best, best_d = 0.0, -1.0
        for c in cand:
            d = min((math.hypot(xs[i] - px, c - py) for px, py in placed if abs(xs[i] - px) < 2.2 * r), default=1e9)
            if d >= 2 * r * 0.9:
                best, best_d = c, d
                break
            if d > best_d:
                best, best_d = c, d
        ys[i] = best
        placed.append((xs[i], best))
    return ys


def fmt_exp(e):
    """10 to the power e, the exponent raised (dy: every browser draws it the same)."""
    if e == 0:
        return "1"
    return f'10<tspan dy="-5" font-size="10">{str(e).replace("-", chr(0x2212))}</tspan>'


def strip_figure(tools, key, lo, hi, ticks, axis_label, theme, order, note, right_text, floor=None, floor_label="",
                 tick_text=fmt_exp):
    """One row a tool, one dot a coil set, on a logarithmic axis from 10^lo to 10^hi."""
    t = THEMES[theme]
    W, left, right, top, row = 838, 198, 152, 36, 64
    x0, x1 = left, W - right
    H = top + row * len(order) + 50

    def X(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">']
    yb = top + row * len(order)
    if floor is not None:
        o.append(f'<rect x="{x0:.1f}" y="{top - 8}" width="{X(floor) - x0:.1f}" height="{yb - top + 8}" fill="{t["band"]}"/>')
        o.append(f'<text x="{X(floor) + 5:.1f}" y="{top - 14}" font-size="13" fill="{t["muted"]}">{floor_label}</text>')
        o.append(f'<line x1="{X(floor):.1f}" y1="{top - 22}" x2="{X(floor):.1f}" y2="{yb}" stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="2 3"/>')
    for e in ticks:
        x = X(10.0 ** e)
        o.append(f'<line x1="{x:.1f}" y1="{yb}" x2="{x:.1f}" y2="{yb + 5}" stroke="{t["muted"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{yb + 21}" font-size="13.5" fill="{t["muted"]}" text-anchor="middle">{tick_text(e)}</text>')
    o.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{t["rule"]}" stroke-width="1"/>')
    o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{yb + 42}" font-size="14" fill="{t["muted"]}" text-anchor="middle">{axis_label}</text>')
    for k, name in enumerate(order):
        yc = top + row * k + row / 2
        mine = name == "torux"
        col = t["torux"] if mine else t["other"]
        vals = [float(r[key]) for r in tools[name] if r["status"] == "surface" and r[key] not in ("", None)]
        vals = [min(max(v, 10.0 ** lo * 1.02), 10.0 ** hi) for v in vals]
        xs = [X(v) for v in vals]
        rad = 2.0
        ys = swarm(xs, rad, row / 2 - 6)
        if k:
            o.append(f'<line x1="{x0}" y1="{top + row * k}" x2="{x1}" y2="{top + row * k}" stroke="{t["rule"]}" stroke-width="0.6"/>')
        o.append(f'<text x="{left - 14}" y="{yc + 5:.1f}" font-size="16" font-weight="{700 if mine else 400}" '
                 f'fill="{t["torux_text"] if mine else t["ink"]}" text-anchor="end">{LABELS.get(name, name)}</text>')
        o.append(f'<g fill="{col}" fill-opacity="{0.8 if mine else 0.62}">' +
                 "".join(f'<circle cx="{x:.1f}" cy="{yc + y:.1f}" r="{rad}"/>' for x, y in zip(xs, ys)) + "</g>")
        txt = right_text(name, tools[name], vals)
        for j, line in enumerate(txt):
            o.append(f'<text x="{x1 + 14}" y="{yc + 5 + 17 * (j - (len(txt) - 1) / 2):.1f}" font-size="13.5" '
                     f'fill="{t["torux_text"] if mine and j == 0 else t["muted"]}" font-weight="{600 if mine and j == 0 else 400}">{line}</text>')
    if note:
        o.append(f'<text x="{x1}" y="{top - 14}" font-size="12" fill="{t["muted"]}" text-anchor="end">{note}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def tool_order(tools, key):
    rest = [n for n in tools if n not in ("torux", "quasr-stored")]
    rest.sort(key=lambda n: median([float(r[key]) for r in tools[n] if r["status"] == "surface" and r[key] != ""]))
    return ["torux"] + rest + (["quasr-stored"] if "quasr-stored" in tools else [])


def accuracy(tools):
    def right(name, rows, vals):
        n = len(vals)
        if name == "quasr-stored":
            return [f"{n} surfaces"]
        if name == "torux":
            none = sum(r["solver_status"] == "no_surface" for r in rows)
            weak = sum(r.get("certificate") == "edge_barrier" for r in rows)       # the weaker kind of certificate
            return [f"{n} surfaces", f"{none - weak} + {weak} certified none", f"{len(rows) - n - none} undetermined"]
        return [f"{n} surfaces", f"{len(rows) - n} none"] if len(rows) > n else [f"{n} surfaces"]
    for theme in THEMES:
        svg = strip_figure(tools, "bn_max", -16.4, 0.0, [-15, -12, -9, -6, -3, 0],
                           "field crossing the surface, largest |B·n| / |B|", theme, tool_order(tools, "bn_max"),
                           "", right, floor=4e-16, floor_label="the limit of double precision")
        open(os.path.join(OUT, f"accuracy-{theme}.svg"), "w").write(svg)


def times(tools):
    timed = {n: v for n, v in tools.items() if n != "quasr-stored"}
    cpus = {r["cpu_model"] for v in timed.values() for r in v}
    if len(timed) < 2 or len(cpus) != 1:
        return False            # a comparison of times needs several tools on one processor

    def right(name, rows, vals):
        m = median(vals)
        return [f"median {m:,.0f} s" if m >= 100 else f"median {m:.2g} s"]
    seconds = {-1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1,000"}
    for theme in THEMES:
        svg = strip_figure(timed, "answer_s", -1.0, math.log10(3600.0), [-1, 0, 1, 2, 3],
                           "seconds until the tool had the surface it returned, four threads", theme,
                           tool_order(timed, "answer_s"), "", right, tick_text=seconds.get)
        open(os.path.join(OUT, f"time-{theme}.svg"), "w").write(svg)
    return True


def refinement():
    """How the error falls as the surface is given more Fourier terms: one line a coil set (refinement.csv)."""
    with open(os.path.join(HERE, "refinement.csv")) as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        M, N = int(r["M"]), int(r["N"])
        by.setdefault(r["design"], []).append((2 * M * (2 * N + 1) + 2 * N + 2, float(r["bn"])))
    W, H, left, right, top, bottom = 838, 340, 84, 18, 16, 56
    x0, x1, y0, y1 = left, W - right, top, H - bottom
    lx0, lx1, ly0, ly1 = 3.0, 6.1, -4.5, -16.5          # log10 unknowns, log10 error (top to bottom)

    def X(v):
        return x0 + (math.log10(v) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(v):
        return y0 + (math.log10(v) - ly0) / (ly1 - ly0) * (y1 - y0)
    for theme in THEMES:
        t = THEMES[theme]
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">']
        o.append(f'<rect x="{x0}" y="{Y(4e-16):.1f}" width="{x1 - x0}" height="{y1 - Y(4e-16):.1f}" fill="{t["band"]}"/>')
        o.append(f'<line x1="{x0}" y1="{Y(4e-16):.1f}" x2="{x1}" y2="{Y(4e-16):.1f}" stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="2 3"/>')
        o.append(f'<text x="{x0 + 10}" y="{Y(4e-16) + 17:.1f}" font-size="13" fill="{t["muted"]}">the limit of double precision</text>')
        for e in (-6, -9, -12, -15):
            y = Y(10.0 ** e)
            o.append(f'<line x1="{x0 - 5}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="{t["muted"]}"/>')
            o.append(f'<text x="{x0 - 10}" y="{y + 4.5:.1f}" font-size="13.5" fill="{t["muted"]}" text-anchor="end">{fmt_exp(e)}</text>')
        for e, lab in ((3, "1,000"), (4, "10,000"), (5, "100,000"), (6, "1,000,000")):
            x = X(10.0 ** e)
            o.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 5}" stroke="{t["muted"]}"/>')
            o.append(f'<text x="{x:.1f}" y="{y1 + 21}" font-size="13.5" fill="{t["muted"]}" text-anchor="middle">{lab}</text>')
        o.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{t["rule"]}"/>')
        o.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{t["rule"]}"/>')
        o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 44}" font-size="14" fill="{t["muted"]}" text-anchor="middle">numbers describing the surface</text>')
        o.append(f'<text x="{x0 - 58}" y="{(y0 + y1) / 2:.1f}" font-size="14" fill="{t["muted"]}" text-anchor="middle" transform="rotate(-90 {x0 - 58} {(y0 + y1) / 2:.1f})">field crossing the surface</text>')
        o.append(f'<g fill="none" stroke="{t["torux"]}" stroke-width="1" stroke-opacity="0.38" stroke-linejoin="round">')
        for pts in by.values():
            pts = [(max(min(n, 10 ** lx1), 10 ** lx0), max(min(b, 10 ** ly0), 10 ** ly1)) for n, b in pts]
            if len(pts) > 1:
                o.append('<polyline points="' + " ".join(f"{X(n):.1f},{Y(b):.1f}" for n, b in pts) + '"/>')
        o.append("</g>")
        o.append(f'<g fill="{t["torux"]}" fill-opacity="0.85">' + "".join(
            f'<circle cx="{X(max(min(p[-1][0], 10 ** lx1), 10 ** lx0)):.1f}" cy="{Y(p[-1][1]):.1f}" r="2"/>' for p in by.values()) + "</g>")
        o.append(f'<text x="{x1 - 8}" y="{y0 + 22}" font-size="14" fill="{t["ink"]}" text-anchor="end">each line is one coil set,</text>')
        o.append(f'<text x="{x1 - 8}" y="{y0 + 41}" font-size="14" fill="{t["ink"]}" text-anchor="end">refined until the digits run out</text>')
        o.append("</svg>")
        open(os.path.join(OUT, f"refinement-{theme}.svg"), "w").write("\n".join(o) + "\n")
    return len(by)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tools = read_compare()
    accuracy(tools)
    print("accuracy:", ", ".join(tool_order(tools, "bn_max")))
    print("time figure:", "written" if times(tools) else "not yet (needs several tools timed on one processor)")
    print("refinement:", refinement(), "coil sets")
