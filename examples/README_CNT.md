# CNT: four circular coils, two of them interlocked

An example source for torux: the Columbia Non-neutral Torus (CNT), a two-period stellarator of
ultralow aspect ratio made of only four circular planar coils. It is a small, exact and
unusual test case:

- **Exact coils.** Every coil is a circle, so the Fourier-filament file is exact at order 1.
  There is no discretisation error in the source.
- **Strong shaping.** The aspect ratio is about 2, and the cross-section at φ = 0 is a narrow
  bean. In straight-field-line angles the poloidal spectrum decays slowly: at the rounding floor
  torux needs M = 112 poloidal harmonics, against N = 32 toroidal ones.
- **Islands on both sides.** The requested surface lies between the ι = 1/5 and ι = 2/9
  island chains. The last closed surface sits just inside the ι = 1/4 chain.

## Files

| file | what it is |
|---|---|
| `cnt.json` | the four coils, Fourier-filament JSON, with the requested flux |
| `make_cnt.py` | writes `cnt.json` from the design parameters (standard library only) |
| `cnt.py` | the example: solve for the surface and compare it with the independent result |

Drop the files into `examples/` and run:

```sh
python examples/cnt.py
torux solve examples/cnt.json --flux 0.011624274070939846
```

`python examples/cnt.py` (torux 0.3.0, 4 threads; the time includes the first compilation, a second run takes 3 s):

```
surface                verified: the solver's check and an independent evaluator agree
toroidal flux          0.0116242740709 Wb
rotational transform   0.207942704542
radii                  major 0.306750332 m, minor 0.136080864 m, aspect ratio 2.25418
volume                 0.112126664 m^3
Fourier series         M, N = 112, 32, with 2 field periods
field across it        1.1e-15 of |B| (the solver's own check)
independent check      6.8e-16 of |B| across it, and the same flux to within 7e-16
time                   39.7 s

rotational transform  0.2079427045422565, expected 0.2079427045422565
minor radius          0.136080864196 m, expected 0.136080864196 m
aspect ratio          2.25417683, expected 2.25417683
```

Given only the flux, torux finds the same surface as the independent solver: ι agrees to all 16 digits,
and the minor radius and aspect ratio to every digit printed.

## The coils

Design: Pedersen et al., Fusion Sci. Technol. 46, 200 (2004), configuration with 64° tilt.
Fourier parameterisation of the interlocked (IL) coils: Yu et al., Phys. Plasmas 28, 092501 (2021).

| coil | geometry | current |
|---|---|---|
| IL 1 | (0.313 + 0.405 cos t, −y_s1 sin t, −z_s1 sin t) m | 170 kA |
| IL 2 | IL 1 rotated by π about z: (x, y, z) → (−x, −y, z) | 170 kA |
| PF top / bottom | rings of radius 1.08 m at z = ±0.405 m, counter-clockwise seen from +z | 40.3 kA each |

The tilt is the angle between the coil **planes**, not between their normals:
y_s1 = R_c √((1 − cos 64°)/2) and z_s1 = R_c √((1 + cos 64°)/2), with R_c = 0.405 m.
Using the angle between the normals instead gives an entirely different ι. The orientation
(z = −z_s1 sin t) gives positive ι in VMEC's sign convention.

## The expected answer

These values come from a solver written independently of torux. It solves the same invariance
equation, (∂φ + ι∂θ)X = V(X), at fixed ι in straight-field-line angles, by Newton–Krylov. It uses
an analytic circular-loop field (elliptic integrals) instead of Biot–Savart quadrature. The value
of ι was chosen first, from field-line tracing, and the flux was then computed on the surface.

| quantity | value |
|---|---|
| toroidal flux (μ0 = 4π × 10⁻⁷) | 0.011624274070939846 Wb |
| rotational transform | 0.2079427045422565 = 2 / (9 + 1/φ_golden), the noble number between 1/5 and 2/9 |
| minor radius √(⟨A⟩/π) | 0.136080864196 m |
| major radius V / (2π² a²) | 0.306750331666 m |
| aspect ratio | 2.25417683 |
| volume | 0.112126663517 m³ |

## Context: the region of closed surfaces

Field-line tracing with weighted Birkhoff averages gives:

- **Magnetic axis:** ι = 0.15257.
- **Last closed surface:** just inside the ι = 1/4 island chain, at ι = 0.24573, enclosing
  0.02270 Wb. Its minor radius is a = 0.1968 m, against 0.196 m in Pedersen 2004, Table I.
- **Requested surface:** at ρ = r/a = 0.691 of that region, s = Φ/Φ_edge = 0.512.

Outside the last closed surface there is a thin layer of island chains (1/4, 14/55, 12/47,
8/31, 6/23) and chaos. In it, regular-looking lines carry fluxes that differ by 5 to 20 %
between cross-sections. Lines are lost from R = 0.604 m on, measured on the outboard midplane
at φ = 90°. That traced edge of about 0.0227 Wb is the value to compare with the result of
`torux solve examples/cnt.json --largest`.

The tracing used μ0 = 1.25663706212 × 10⁻⁶ (CODATA 2018). Fluxes computed with it are larger
by a factor of 1 + 5.4 × 10⁻¹⁰. The geometry does not depend on μ0.
