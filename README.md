<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="visuals/torux-dark.png">
    <img src="visuals/torux-light.png" width="830" alt="TORUX">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="visuals/any-coil-dark.png">
    <img src="visuals/any-coil-light.png" width="838" alt="Twelve magnet systems for fusion stand in three rows of four, drawn in copper: stellarator coil sets of many shapes, a wireframe, a ring of permanent magnets inside upright coils, the coils of the ITER tokamak, six window-frame coils and a levitated ring. Inside nine of them, teal light woven from field lines fills the vacuum magnetic surface torux verified. The last three, ITER's coils, DIII-D's I-coils and the dipole, hold no light: thin grey field lines loop round their coils instead and never go round the machine.">
  </picture>
</p>

<h1 align="center">Arithmetic limit, extremely fast. Any coil works.</h1>

<p align="center"><sub>(The aspiration. torux is in development, and vacuum only for now: no finite pressure.)</sub></p>

<p align="center"><a href="method/README.md"><b>The method</b></a> · <a href="docs/usage.md">Usage</a> · <a href="docs/verification.md">What is checked</a> · <a href="results/README.md">Results</a></p>

**Coils in, vacuum flux surface out: max |B·n|/|B| of 10⁻¹⁵, in seconds, with nothing to tune.**

It has done that for 231,878 of the 371,701 coil sets in [QUASR](https://quasr.flatironinstitute.org). A second evaluator, sharing no code with the solver, checked every one. For 5,206 more it returned a certificate that the surface asked for cannot exist, and for another 846 a weaker one. It never guesses: the other 133,771 it calls undetermined, three in four of them after less than a minute's work.

Above: twelve kinds of magnet, one solver. Nine carry the surface torux verified for them, drawn as light. The other three (a tokamak's poloidal coils, its perturbation coils, a levitated dipole) carry none, and that is an answer too: their field lines, the thin grey ones, close round the coils and never go round the machine, so no surface can enclose toroidal flux, and torux proves it. (A tokamak's surfaces need the current in its plasma, and a dipole holds its plasma on the closed lines themselves.) [Each one, close up.](visuals/README.md)

## Fifteen digits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="visuals/accuracy-dark.svg">
  <img src="visuals/accuracy-light.svg" width="838" alt="One row of dots for each tool on a logarithmic axis of the field crossing a surface, from 10 to the minus 15 up to 1. torux: 201 dots between 3 times 10 to the minus 16 and 7 times 10 to the minus 15, median 1 times 10 to the minus 15. SIMSOPT BoozerLS: 283 dots between 2 times 10 to the minus 15 and 4 times 10 to the minus 2, median 2 times 10 to the minus 8. SIMSOPT BoozerExact: 288 dots between 5 times 10 to the minus 15 and 2 times 10 to the minus 1, median 3 times 10 to the minus 8. SIMSOPT QFM: 297 dots between 5 times 10 to the minus 14 and 1, median 3 times 10 to the minus 7. VMEC++: 252 dots between 2 times 10 to the minus 6 and 2 times 10 to the minus 1, median 9 times 10 to the minus 4. DESC: 199 dots between 3 times 10 to the minus 8 and 2 times 10 to the minus 1, median 3 times 10 to the minus 3. QUASR's stored surface: 300 dots between 8 times 10 to the minus 9 and 2 times 10 to the minus 1, median 9 times 10 to the minus 4.">
</picture>

Three hundred coil sets drawn at random from QUASR. Every tool gets the same coils, the same requested flux and an hour on four threads for each. The other five need settings, and we chose them: the route each tool's documentation recommends, pushed wherever that paid on six coil sets outside the 300. torux has no settings to choose. One judge: an evaluator that shares no code with any of them, working in extended precision at 4,000 random points of each surface. A dot is a coil set, placed at the largest |B·n|/|B| on its surface. [How each tool was run, and how to run a row again.](results/compare300/README.md)

torux's surfaces have a median max |B·n|/|B| of 1.1 × 10⁻¹⁵, within 2.9 times the least a surface written in double precision can have. The worst of its 201 is at 6.8 × 10⁻¹⁵. On the 199 coil sets both solved, the closest tool, SIMSOPT BoozerExact, is 700,000 times higher: 5 orders of magnitude. The surfaces the designs are stored with, where every solve starts, are at 8.9 × 10⁻⁴: they are coarse, a few hundred Fourier terms each, where torux's take thousands. [Every dot is a row of `results/compare300.csv`.](results/compare300.csv)

| | surfaces returned | median max \|B·n\|/\|B\| | within 10⁻¹² | median seconds |
|---|---:|---:|---:|---:|
| **torux** | 201 of 300 | 1.1 × 10⁻¹⁵ | 201 | 4.2 |
| SIMSOPT BoozerLS | 283 of 300 | 2.4 × 10⁻⁸ | 38 | 824 |
| SIMSOPT BoozerExact | 288 of 300 | 2.7 × 10⁻⁸ | 51 | 67 |
| SIMSOPT QFM | 297 of 300 | 3.3 × 10⁻⁷ | 14 | 1,770 |
| VMEC++ | 252 of 300 | 9.4 × 10⁻⁴ | 0 | 1,525 |
| DESC | 199 of 300 | 2.6 × 10⁻³ | 0 | 885 |
| QUASR's stored surface | 300 of 300 | 8.9 × 10⁻⁴ | 0 | |

Nobody needs fifteen digits of a flux surface. What you need is never to wonder whether the surface is wrong or the solver. And at this precision "there is no surface" stops being a solve that failed and becomes an answer torux can certify: 22 of these 300, and 16 more by its weaker kind of certificate. The other 61 were still undetermined after their hour.

## Seconds

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="visuals/time-dark.svg">
  <img src="visuals/time-light.svg" width="838" alt="One row of dots for each tool on a logarithmic axis of seconds, from a tenth of a second to an hour: the time at which each tool had the surface it returned. torux: median 4.2 seconds. SIMSOPT BoozerExact: median 67 seconds. SIMSOPT BoozerLS: median 824 seconds. DESC: median 885 seconds. VMEC++: median 1,525 seconds. SIMSOPT QFM: median 1,770 seconds.">
</picture>

The same 300 coil sets on one processor (AMD EPYC 7742), four threads each. A dot is a surface a tool returned, placed at the time at which the tool had it, the tool's own checks included. torux: 4.2 s in the median, and then it stops. The others were set to spend their hour on accuracy and go on working after the surface they return, VMEC++ and SIMSOPT QFM nearly always until the hour ends: that later time is not counted here.

- SIMSOPT BoozerExact: 67 s. On the 199 coil sets both solved it was faster on 12 and more accurate on 0.
- SIMSOPT BoozerLS: 824 s. On the 199 coil sets both solved it was faster on 2 and more accurate on 0.
- DESC: 885 s. On the 141 coil sets both solved it was faster on 0 and more accurate on 0.
- VMEC++: 1,525 s. On the 177 coil sets both solved it was faster on 1 and more accurate on 0.
- SIMSOPT QFM: 1,770 s. On the 199 coil sets both solved it was faster on 0 and more accurate on 0.

## The method

<p align="center">
  <a href="method/README.md">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="method/figures/surface-a-dark.svg">
    <img src="method/figures/surface-a-light.svg" width="838" alt="Left, an ink drawing of the flux surface of HSX in perspective: a twisted, flattened tube bent into a ring with four field periods, hidden lines removed, a twelfth of it cut away so that a hatched cross-section faces the reader. A net of thin curves of constant theta and constant phi covers it, and one heavy field line runs round it, starting from a dot on the cut. Arrows show phi increasing round the ring and theta round the cross-section. Right, the square of the two angles, phi across and theta up, each from 0 to 2 pi, ruled in a 12 by 12 grid: the same field line is a straight heavy line from theta 0 on the left edge, rising with slope iota, leaving through the top and re-entering at the bottom. A small triangle marks the slope: 1 across, iota up. Above the square: theta = theta 0 + iota phi.">
    </picture>
  </a>
</p>

The surface is a Fourier series: R and Z in cos and sin of mθ − nN<sub>fp</sub>φ, the series VMEC, DESC and SIMSOPT read. φ is the cylindrical angle and θ the poloidal angle in which field lines are straight. The coefficients and ι are the unknowns of one square system, solved by Newton's method on the coils' Biot-Savart field: no finite elements, and no fit to traced lines. Where a flux surface exists it is the quadratic-flux-minimising surface, with the angle pinned so that the solution is unique.

**[The method, in five figures](method/README.md)**: the equation, the start from the magnetic axis with no initial guess, why the answer reaches rounding error, and the proofs that there is no surface.

## Use it

```sh
pip install torux                    # or: conda install -c conda-forge torux
torux solve quasr:954

uvx torux solve quasr:954            # or run it once, with nothing installed
```

```
surface                verified: the solver's check and an independent evaluator agree
toroidal flux          0.069106905814 Wb
rotational transform   -0.0975229741144
radii                  major 0.999994679 m, minor 0.150000523 m, aspect ratio 6.66661
volume                 0.444132929 m^3
Fourier series         M, N = 48, 32, with 2 field periods
field across it        1.1e-15 of |B| (the solver's own check)
independent check      3.3e-15 of |B| across it, and the same flux to within 8e-16
time                   0.7 s
```

The first run compiles for up to a minute. After that, seconds.

Every answer is one of three. A surface, which a second program sharing no code with the solver has checked at 4,096 random points. No surface, with a certificate: the island chain or the edge of confinement that forbids it, and the flux it rules out. Or undetermined, never a guess.

```python
import torux

s = torux.solve("coils.json", minor_radius=0.15)   # or flux=0.07 (Wb), volume=0.44 (m³), largest=True
print(s)                                 # what it found and how it was checked, or why there is no surface
s.iota, s.volume, s.aspect_ratio         # the numbers
s.save("input.mine")                     # a VMEC input: VMEC, VMEC++, DESC and SIMSOPT read it
s.plot("mine.png")                       # the picture below
s.boozer().quasisymmetry                 # |B| in Boozer angles, and how far it is from quasi-symmetry

torux.check("coils.json", "wout_other.nc")    # a surface from another code: how much of the field crosses it
torux.many(files, table="answers.csv")        # a folder of designs, one row each
```

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="visuals/plot-dark.png">
    <img src="visuals/plot-light.png" width="760" alt="What s.plot draws for the surface above: on the left the verified flux surface as a green torus, shaded by the strength of the field on it, threaded through its four copper coils. On the right are its cross-sections at three toroidal angles, an ellipse lying flat, a rounded triangle and an ellipse standing upright, and the scale of the field strength, about 0.9 to 1.1 tesla.">
  </picture>
</p>

Nothing to tune: no resolution, no tolerance, no starting guess. The coils are a file (SIMSOPT and QUASR JSON, MAKEGRID, FOCUS, permanent magnets, wireframes, a tokamak's coils), a QUASR design by its number, or a SIMSOPT or DESC object you already hold. [Usage](docs/usage.md) and [formats](docs/formats.md). [The method](method/README.md) and [what is checked, and by what](docs/verification.md).

## Credit

Used torux? Say so, so the next person finds it:

> Flux surfaces computed with [torux](https://github.com/philippbogdan/torux).

```bibtex
@software{bogdan2026torux,
  author = {Bogdan, Philipp},
  title  = {torux: vacuum flux surfaces at the limit of double-precision arithmetic},
  year   = {2026},
  url    = {https://github.com/philippbogdan/torux}
}
```

Public domain, [CC0 1.0](LICENSE), all but the picture of the twelve machines: six of them are drawn from files under GPL, LGPL and CC BY licences, and those renderings stay under their sources' terms. The coil sets torux was run on are other people's work, under their own licences: [credits](NOTICE.md).
