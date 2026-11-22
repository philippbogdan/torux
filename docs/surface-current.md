# Outside a plasma: coils and a current sheet

`torux.SurfaceCurrent`, added in 0.4.0, holds the plasma boundary Γ, its surface current and the toroidal flux inside it. The exterior field is the sum of the coil and sheet fields. For virtual casing, K = n × B / μ₀, where n is the outward unit normal and B is the total field on Γ.

On exact test cases, the field and the surfaces returned by `torux.solve` agree with independent calculations to rounding level. It is not yet a practical tool. One surface takes tens of minutes to hours. And where Γ is not itself a flux surface of coils plus sheet, as in the public VMEX equilibrium as converged, torux verifies no surface close to it.

```python
import torux

src = torux.load("examples/vmex_surface_current/current.json")   # coils, and Γ with its current: 0.025 Wb inside
s = torux.solve(src, flux=0.045)          # the full toroidal flux: 0.025 Wb inside Γ and 0.02 Wb between Γ and the surface
s = torux.solve(src, largest=True)        # the largest surface torux can verify outside Γ
print(torux.check(src, "surface.json"))   # a surface from another code, against the same source
```

## Input

- `modes` contains distinct integer `(m, n)` pairs. Every coefficient uses the phase `m*theta - n*nfp*phi`. Both angles are in radians and `phi` is the cylindrical angle. `(0, n)` and `(0, -n)` may both appear.
- `rmnc`, `rmns`, `zmnc`, `zmns` are the cosine and sine coefficients of Γ's `R` and `Z` in metres. `rmns` and `zmnc` may be omitted. Each constant-`phi` section must run counterclockwise in `(R, Z)` as `theta` increases. The outward unit normal n points along `r_phi × r_theta`.
- For a current potential, give `kappac` and `kappas` in amperes, with secular terms `G*phi/(2*pi) + I*theta/(2*pi)` and K = n × grad_s κ. `G` is the current through the hole along `+z` linked by a loop along `+phi` on Γ. `I` is minus the net toroidal current along `+phi` inside Γ. If your convention is K = grad_s Φ × n, give κ = −Φ, including both secular terms.
- For tangential field input, give `covariant` with shape `(2, 2, len(modes))`. The first index selects `B·r_theta / mu0` or `B·r_phi / mu0`, in that order. The second selects cosine or sine coefficients, in that order. Use the total B on Γ, including coils and plasma. Both components are retained, allowing currents that are not gradients.
- `flux` is the magnitude of the toroidal flux inside Γ in Wb, from VMEC's `phiedge` or DESC's `Psi`. The field sets its sign. Supply it for `torux.solve` and for full flux in `torux.check`. Field evaluation and Poincaré sections can omit it.

`Source.save` writes coils and sheet to one torux JSON file. For separate files, `torux.load("coils.json", "sheet.json", nfp=4)` verifies symmetry by probing the coil field and checking the sheet coefficients. `torux.solve` requires one sheet and verified stellarator symmetry. The sine coefficients of `R`, cosine coefficients of `Z`, and cosine coefficients of `kappa` or sine coefficients of `covariant` must be at most 10⁻⁹ of the rest.

## What a verified surface means

The solve starts just outside Γ and continues outward without a magnetic axis. `torux.check` measures a supplied surface against the same field. Verification uses only the exterior field and applies the vacuum acceptance criteria, including the independent field and flux checks. Each sampled point must be outside Γ and at least 10⁻⁵ of Γ's minor radius from it. Each section must go once around Γ's section. Refining the sheet quadrature must change its field by at most 2 × 10⁻¹⁵ of |B|. [Verification](verification.md)

A flux request is for the full toroidal flux. torux adds the supplied flux inside Γ to the measured flux between Γ and the surface. `answer.evidence['sheet']` reports `own_flux_Wb`, the source field's flux through Γ from outside, and `own_flux_spread_Wb`, its variation between two sections. `own_flux_Wb` is zero for exact virtual casing. `boundary_bn_max` and `boundary_bn_median` report the maximum and median exterior |B·n|/|B| on Γ.

A failed search returns `undetermined` and gives the reason. There are no certificates of non-existence for a source with a sheet.

## What refuses

The solver refuses sources outside the input requirements above, flux at or below the flux inside Γ, and `minor_radius` or `volume` at or below Γ's corresponding size. Also refused are `gradients=True`, `answer.vjp`, `trace_lines`, `widths_s`, `answer.render` and `torux render`. `torux.poincare` requires `starts=` (`--start` on the command line). A trace that reaches Γ or loses its toroidal direction reports `stop_reason` and `left_at`.

## How close to Γ

For the VMEX `current.json` and QUASR 954 null cases, the coil and sheet field agreed with the independent double-double evaluator to within 4.6 × 10⁻¹⁶ of |B| down to 10⁻⁵ of Γ's minor radius. Requests requiring a closer surface are `undetermined`. `FieldEngine(src)` refuses points inside Γ or nearer than `engine.sheets[0].hard_limit`, 1.2 × 10⁻⁷ of a root panel's half-diagonal.

## Measured results and cost

Against calculations sharing no code with torux, the QUASR 954 null case returns the known coil surfaces at 0.03 and 0.05 Wb to within 2 × 10⁻¹⁵ m. A sheet replacing a current ring in a `1/R` toroidal field returns the closed-form surface to within 1 × 10⁻¹⁴ m. In a QUASR quasi-axisymmetric case where Γ is a flux surface to 6 × 10⁻¹⁴, a surface at 1.1 times the flux inside Γ verifies.

In the public VMEX cases, the maximum |B·n|/|B| on Γ is 4.8 × 10⁻³ for `current.json` and 4.0 × 10⁻³ for `beta1.json`. The first requests that verify are at 0.045 Wb, 1.8 times the flux inside Γ. They took 1,640 s and 10,623 s respectively on shared cores. Both inputs come from fresh solves of the [public Landreman-Paul QA example](https://github.com/uwplasma/VMEX/blob/757fe28348d959d5d4a50653d25fcca67c8fb941/docs/_static/figures/sources/make_extender_islands_figure.py) at about 1 % volume-averaged beta; `examples/vmex_surface_current/prepare.py` converts a VMEX `wout` and its ESSOS coils.

Sheets above about 10⁴ modes are out of reach today. A W7-X case with about 8,800 modes and a QUASR quasi-helical case with 19,333 modes did not finish within their budgets.
