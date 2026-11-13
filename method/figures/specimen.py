"""Specimen plate of the drawing style in style.py: every weight, dash, tone and size of
lettering, and two small real drawings that use them together.

    python specimen.py [preview-dir]

writes specimen-light.svg and specimen-dark.svg beside this file, and PNG previews into
preview-dir when it is given.
"""

import sys

import numpy as np

import style

fig = style.Figure(height=1040)

# ------------------------------------------------------------------ lines
cols = (170, 300, 430)
for c, w in zip(cols, style.WEIGHTS):
    fig.text(c + 50, 36, w, size="small", baseline=True)
rows = ["solid", "dashed", "dotted", "dash-dot", "dash-dot-dot", "arrow", "marks"]
for i, name in enumerate(rows):
    y = 66 + 30 * i
    fig.text(140, y, name, anchor="e", size="small", gap=0)
    for c, w in zip(cols, style.WEIGHTS):
        if i < 5:
            fig.line([c, c + 100], [y, y], weight=w, dash=style.DASHES[i])
        elif i == 5:
            fig.arrow(c, y, c + 100, y, weight=w)
    if i == 6:
        for c, m in zip(cols, style.MARKS):
            fig.mark(c + 50, y, kind=m)

# ------------------------------------------------------------------ tones
def cell(i, j, label):
    left, top = 582 + 82 * i, 46 + 108 * j
    fig.text(left + 30, top + 84, label, size="small", baseline=True)
    return left, top, [left, left + 60, left + 60, left], [top, top, top + 60, top + 60]


for i, (tone, label) in enumerate([("stipple", "stipple"), ("hatch", "hatch"), ("crosshatch", "cross-hatch")]):
    left, top, xs, ys = cell(i, 0, label)
    fig.fill(xs, ys, tone=tone)
    fig.line(xs, ys, closed=True)
left, top, xs, ys = cell(0, 1, "flipped")
fig.fill([left, left + 60, left], [top, top, top + 60], tone="hatch")
fig.fill([left + 60, left + 60, left], [top, top + 60, top + 60], tone="hatch", flip=True)
fig.line([left + 60, left], [top, top + 60], weight="fine")
fig.line(xs, ys, closed=True)
left, top, xs, ys = cell(1, 1, "paper, ink")
th = np.linspace(0, 2 * np.pi, 241)
fig.fill(xs, ys, tone="crosshatch")
fig.fill(left + 30 + 18 * np.cos(th), top + 30 + 18 * np.sin(th), tone="paper")
fig.fill(left + 30 + 8 * np.cos(th), top + 30 + 8 * np.sin(th), tone="ink")
fig.line(xs, ys, closed=True)
left, top, xs, ys = cell(2, 1, "halo")
fig.line([left + 12, left + 12, left + 48, left + 48], [top, top + 60, top + 60, top], weight="medium")
fig.line([left - 4, left + 64], [top + 44, top + 16], weight="heavy", halo=True)

# ------------------------------------------------------------------ lettering
fig.line([28, 810], [272, 272], weight="fine")


def row(y, labels, size="normal", space=24):
    x = 30
    for s in labels:
        x = fig.text(x, y, s, anchor="w", size=size, gap=0, baseline=True)[2] + space


row(312, [r"$\psi$", r"$\theta$", r"$\varphi$", r"$\iota$", r"$R$", r"$Z$", r"$\Psi_0$", r"$\mathbf{B}$",
          r"$B_\varphi$", r"$N_{\mathrm{fp}}$", r"$\theta^{*}$", r"$s=\sqrt{\Psi}$"])
row(348, ["magnetic axis", "last closed surface", "island chain", "X point", "field line"], space=28)
row(386, [r"$\theta=\theta_0+\iota\varphi$", r"$\mathbf{B}\cdot\mathbf{n}=0$",
          r"$(\partial_\varphi+\iota\,\partial_\theta)\,Z=RB_Z/B_\varphi$"], space=36)
row(420, ["$0$", "$10$", "$-0.5$", r"$10^{-16}$", r"$2\pi$", r"$m=5$", "the series is lengthened",
          r"$|\mathbf{B}\cdot\mathbf{n}|/|B|$"], size="small", space=22)
fig.line([28, 810], [444, 444], weight="fine")

# ------------------------------------------------------------------ a cross-section
th = np.linspace(0, 2 * np.pi, 1441)


def bean(rho, delta=0.42, kappa=1.75, shift=0.16):
    d = delta * rho
    R = shift * (1 - rho ** 2) + rho * (np.cos(th) + d * np.cos(2 * th)) - 0.5 * d * rho
    Z = kappa * rho * np.sin(th)
    return R, Z


def offset(R, Z, dist):
    tR, tZ = np.gradient(R), np.gradient(Z)
    n = np.hypot(tR, tZ)
    return R + dist * tZ / n, Z - dist * tR / n


v = fig.view(box=(14, 462, 420, 528), x=(-1.65, 1.75), y=(-2.2, 2.2), equal=True)
lcfs = bean(1.0)
inner = offset(*lcfs, 0.32)
outer = offset(*lcfs, 0.46)
v.fill([lcfs[0], inner[0]], [lcfs[1], inner[1]], tone="stipple")
v.fill([inner[0], outer[0]], [inner[1], outer[1]], tone="hatch")
v.line(*inner, closed=True)
v.line(*outer, closed=True)
for rho in (0.25, 0.5, 0.75):
    v.line(*bean(rho), closed=True, weight="fine")
v.line(*lcfs, closed=True, weight="heavy")
ends = outer[0][np.argmin(np.abs(outer[1]) + 9 * (outer[0] > 0))], outer[0][np.argmin(np.abs(outer[1]) + 9 * (outer[0] < 0))]
v.line([ends[0] - 0.12, ends[1] + 0.12], [0, 0], weight="fine", dash="dashdot")
axis = (0.16, 0.0)
v.mark(*axis)
v.leader(-0.02, -2.36, "magnetic axis", 0.15, -0.04)
v.leader(0.42, 2.12, "last closed surface", 0.2, 1.62)
v.leader(1.32, -1.98, "wall", 1.13, -1.36)


def crossing(curve, angle):
    """Where the ray from the axis at `angle` meets a curve around it."""
    a = np.arctan2(curve[1] - axis[1], curve[0] - axis[0])
    k = np.argmin(np.abs(np.angle(np.exp(1j * (a - angle)))))
    return np.array([curve[0][k], curve[1][k]])


ray = np.radians(50)
p_lcfs, p_wall = crossing(lcfs, ray), crossing(inner, ray)
start = np.array(axis) + 0.14 * np.array([np.cos(ray), np.sin(ray)])
v.arrow(*start, *p_lcfs)
v.text(*(0.5 * (p_lcfs + p_wall)), r"$\psi$")
v.text(ends[1] + 0.12, 0, r"$Z=0$", anchor="sw", gap=4)

# ------------------------------------------------------------------ a semi-log plot
ax = fig.axes(box=(548, 520, 262, 370), x=(0, 40), y=(1e-18, 10), ylog=True)
n = np.arange(0, 41)
rng = np.random.default_rng(3)
a = np.exp(-0.95 * n) * (1 + 0.35 * np.sin(1.7 * n))
a = np.maximum(a, 2.2e-16 * np.exp(rng.normal(0, 0.5, n.size)))
ax.line(n, a, weight="medium")
ax.mark(n, a, kind="dot")
ax.line([0, 40], [2.2e-16, 2.2e-16], weight="fine", dash="dashed")
ax.line([0, 26], [3, 3 * np.exp(-0.95 * 26)], weight="fine", dash="dotted")
ax.frame("box")
ax.xticks(major=[0, 10, 20, 30, 40], minor=[5, 15, 25, 35])
ax.yticks(major=[1, 1e-4, 1e-8, 1e-12, 1e-16], minor=[1e-2, 1e-6, 1e-10, 1e-14, 1e-18, 10])
ax.xlabel("$m$")
ax.ylabel(r"$|R_m|$")
ax.text(31, 2.2e-16, r"$2^{-52}$", anchor="s", gap=5)
ax.text(17, 3 * np.exp(-0.95 * 17), r"$e^{-0.95m}$", anchor="sw", gap=6)

fig.save("specimen", previews=sys.argv[1] if len(sys.argv) > 1 else None)
