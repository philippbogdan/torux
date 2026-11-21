# Outside a plasma: coils and a current sheet

A finite-beta equilibrium's field outside its plasma is the coils' field plus the field of the plasma's currents, and outside the plasma's boundary Γ the second is exactly the field of a current sheet on Γ: K = n × B / μ₀, B the total field on Γ (virtual casing). `torux.SurfaceCurrent` holds Γ, that current and the toroidal flux inside Γ. With the coils it is a source like any other: `torux.solve` returns the flux surface of the total field outside Γ, verified as a vacuum surface is, or a truthful `undetermined`, and `torux.check` measures a surface against it.

```python
import torux

src = torux.load("examples/vmex_surface_current/current.json")   # coils, and Γ with its current: 0.025 Wb inside
s = torux.solve(src, flux=0.045)          # the full toroidal flux: 0.025 Wb inside Γ and 0.02 Wb between Γ and the surface
s = torux.solve(src, largest=True)        # the largest surface torux can verify outside Γ
print(torux.check(src, "surface.json"))   # a surface from another code, against the same source
```

## What a surface means here

The same as in vacuum, by the same rules ([what is checked](verification.md)): the solver's own check at most 10⁻¹⁴ of |B| off its grid, the resonance guard, the coils' quadrature check, simple sections, and the independent evaluator (`torux.reference`, which evaluates the sheet with a quadrature of its own and shares no code with the solver's) at most 10⁻¹⁴ with its flux within 10⁻¹². A sheet adds three checks of the solver's: every sampled point of the surface outside Γ and at least 10⁻⁵ of Γ's minor radius from it, every section of the surface once round Γ's section, and the sheet's field on the surface moving by at most 2 × 10⁻¹⁵ of |B| when its quadrature is refined (Gauss order 16 and panels used from 3 radii, against 12 and 2).

Inside Γ the source's field is not the plasma's: for K = n × B / μ₀ it is zero. So nothing inside Γ is evidence of anything. There is no magnetic axis to start from: the family of surfaces starts next to Γ, from Γ itself moved out along its normals (2 % of its minor radius, less for a request nearer Γ, and farther, up to 20 %, where no torus converges nearer) and carried into the field's straight-field-line angles, and is continued outward. No surface is accepted that comes within 10⁻⁵ of Γ's minor radius of Γ or passes inside it, no closed field line of a chain is listed that comes as near, and there are no certificates that a surface cannot exist: their theorems hold for tori with no current inside, and a plasma has current inside. Where torux finds no surface the answer is `undetermined` and says why.

## The flux

A request is the usual full toroidal flux. The part carries the flux inside Γ (`flux=`, the equilibrium's own: VMEC's `phiedge`, DESC's `Psi`), and the full flux of a surface outside is that plus the flux of the total field between Γ and the surface. torux measures the second as the difference of the loop integrals of A round the surface and round Γ, the second from outside: the flux of the source's own field through Γ's sections, which is zero for an exact virtual-casing current (and the flux inside Γ for a sheet that carries the plasma's whole current, with the vacuum field inside), and is reported (`answer.evidence['sheet']['own_flux_Wb']`, with how much it differs between two sections, `own_flux_spread_Wb`). The engine's vector potential carries the flux inside Γ as a flux tube on a curve inside Γ, so a loop integral of A outside Γ gives the full flux; B is unchanged.

How far Γ is from a flux surface of the field outside it is measured directly: |B·n|/|B| on Γ from outside at 768 points of a field period (extrapolated from 1, 2 and 4 × 10⁻⁶ of the minor radius out), its largest and median values in `evidence['sheet']['boundary_bn_max']` and `['boundary_bn_median']`. For an exact boundary and current it is at rounding level; where it is not, no surface hugs Γ, and an answer near Γ is `undetermined` with these numbers as its reason.

## How close to Γ

The sheet's field (`torux.field.surface_current`) sums the series exactly at every quadrature node and forms every separation of a target from a node relative to a panel corner kept in double-double, so its accuracy does not fall as a target nears Γ. Measured against the independent evaluator's double-double tier at 8 random points of Γ moved out along its normal by d, the largest |B − B_dd|/|B| of coils and sheet together (`python examples/surface_current/near.py SOURCE`):

| d / Γ's minor radius | 10⁻¹ | 10⁻² | 10⁻³ | 10⁻⁴ | 10⁻⁵ |
|---|---|---|---|---|---|
| VMEX `current.json` (581 modes) | 2.1 × 10⁻¹⁶ | 2.1 × 10⁻¹⁶ | 3.6 × 10⁻¹⁶ | 2.9 × 10⁻¹⁶ | 2.7 × 10⁻¹⁶ |
| QUASR 954 null case (3,870 modes, `tests/data/sheet_truth`) | 4.3 × 10⁻¹⁶ | 2.3 × 10⁻¹⁶ | 3.0 × 10⁻¹⁶ | 3.3 × 10⁻¹⁶ | 4.6 × 10⁻¹⁶ |

An ideal toroidal solenoid on a D-shaped boundary, whose field outside is exactly zero, reads at most 5.7 × 10⁻¹⁶ of its field inside at the same distances (`tests/test_sheet_solve.py`).

A verified surface keeps at least 10⁻⁵ of Γ's minor radius from Γ, the distance down to which these measurements reach; a request whose surface would come nearer is `undetermined` and says so.

## Input

- `modes`: integer `(m, n)` pairs, one each; every coefficient has the phase `m*theta - n*nfp*phi`, both angles in radians, `phi` the cylindrical angle. `(0, n)` and `(0, -n)` may both appear.
- `rmnc`, `rmns`, `zmnc`, `zmns`: Γ's `R` and `Z` in metres (`rmns`, `zmnc` may be omitted). Each constant-`phi` section must run counterclockwise in `(R, Z)` as `theta` increases. `n` is the outward unit normal, along `r_phi × r_theta`.
- The current, as a potential: `kappac`, `kappas` in amperes, with the secular terms `G*phi/(2*pi) + I*theta/(2*pi)`, and `K = n × grad_s kappa`. `G` is the current through the hole along `+z` linked by a loop along `+phi` on Γ; `I` is minus the net toroidal current along `+phi` inside Γ. For a convention `K = grad_s Phi × n`, give `kappa = -Phi`, both secular terms included.
- Or as the tangential field: `covariant`, shape `(2, 2, len(modes))`, the cos and sin coefficients of `B·r_theta / mu0` and of `B·r_phi / mu0`, `B` the total field on Γ (coils and plasma). Both components are kept, so a current that is not a gradient is represented too.
- `flux`: the toroidal flux inside Γ in Wb. Its magnitude: the sign is the field's. Needed by `torux.solve` and for the full flux of `torux.check`. Field evaluation and Poincaré sections need none.

`Source.save` writes the part with the coils in one torux JSON file. With the coils and the sheet in two files, `torux.load("coils.json", "sheet.json", nfp=4)` verifies the symmetry of the whole: the coils' field by probing, the sheet's from its coefficients. `torux.solve` takes a source with one `SurfaceCurrent`, with stellarator symmetry (the source's, verified, and the sheet's coefficients: sine parts of `R`, cosine parts of `Z` and of `kappa`, or the sine parts of `covariant`, at most 10⁻⁹ of the rest).

## What works, and what refuses

Work as in vacuum, each tried on the exact ring case (`tests/test_sheet_ring.py`): a flux; `largest=True` (5.76 Wb verified within a 90 minute budget, nothing decided at twice that flux); `torux.many` (a table row per source); `torux.check` (the full flux of the ring's contour to 1.7 × 10⁻¹⁵, its theta either way round); `answer.plot` (it says that the picture leaves out the sheet), `answer.boozer()`, `answer.system`, `answer.save`; `torux.poincare` with `starts=`; and on the command line `torux solve`, `torux check`, `torux many` and `torux poincare --start`. The chains a request meets are listed as in vacuum, without the lines that come near Γ: the 4 kA VMEX case lists ι = 4/9 near 0.062 Wb.

`minor_radius` and `volume` are searched as in vacuum, from Γ's own size, and are slow: each request there takes 20 to 30 minutes with its independent check on shared cores, and on the ring a 90 minute budget left both one request short (`undetermined`, after verifying surfaces of minor radius 0.1205 m and volume 0.285 m³ on the way to 0.12 m and 0.28 m³). Give them hours.

Refused, each in one sentence: a source with several `SurfaceCurrent`s, one without `flux`, a flux not more than the flux inside Γ, a size not larger than Γ's, a source without stellarator symmetry, `gradients=True` and `answer.vjp` (the field is not differentiated in the sheet), `trace_lines` (the reference tracer does not take a sheet), `widths_s` (island widths are closed in on from the axis), `answer.render` and `torux render` (they light the surfaces inside the answer's by field lines, through Γ), and `torux.poincare` without `starts=`.

## Field, potential and sections

`FieldEngine(src)` gives B, A and grad B at Cartesian points outside Γ; a point inside Γ, or nearer Γ than the quadrature resolves (1.2 × 10⁻⁷ of a root panel's half-diagonal, `engine.sheets[0].hard_limit`), is refused. `FieldEngine(src, sheet_order=16, sheet_eta=3.0)` refines the sheet's quadrature: the change measures its error. `torux.poincare(src, starts=[(R, Z)])` follows field lines outside Γ through planes of constant toroidal angle; a line stops at its last accepted point when it reaches Γ or loses its toroidal direction, and `stop_reason` says which, `left_at` where.

## The public VMEX case

Neither bundled boundary is a flux surface of its coils and sheet: the field crosses Γ by up to 4.8 × 10⁻³ of |B| (median 6.4 × 10⁻⁴) in `current.json` and 4.0 × 10⁻³ (6.0 × 10⁻⁴) in `beta1.json`, where the coils alone cross it by 1.6 and 1.2 × 10⁻². That is the free-boundary equilibrium as converged and converted, not the sheet's quadrature, which holds 3 × 10⁻¹⁶ of |B| there. So no surface hugs Γ. Field lines started on the outboard midplane 0.1 % and 0.3 % of the minor radius outside Γ reach it within half a turn in both; in the 4 kA case a line started 5 % out reaches it after 14 turns, while lines from 7.5 % to 20 % out stay outside for the 100 turns followed (`torux.poincare(src, starts=[(R, 0)], turns=100)`).

What torux answers, with the wall time on shared cores:

| request | `current.json` (4 kA) | `beta1.json` (no plasma current) |
|---|---|---|
| 0.026 Wb | `undetermined` in 490 s: no start torus converged 0.3 to 1.4 % of the minor radius out, and the leak above | `undetermined` in 421 s, the same |
| 0.030 Wb | `undetermined` in 457 s: none converged 1 to 7 % out | no answer in the 3 hours given: a torus refined to 1.5 × 10⁻¹⁵ by the solver's own check in 778 s (M, N = 192, 56), its independent check not finished |
| 0.045 Wb | a surface in 1,640 s, 10 % of the minor radius outside Γ at its nearest; the coils' own vacuum surface at that flux lies up to 4.5 cm from it, ι 0.054 lower | a surface in 10,623 s (found in 563 s; the independent check's double precision read 1.3 × 10⁻¹¹ where it climbed and its double-double tier 8.5 × 10⁻¹⁶, which decides); the coils' vacuum surface at that flux lies up to 3.5 cm from it, ι within 5 × 10⁻⁵ |
| `largest=True` | LARGESTC | LARGESTB |

The bundled JSON inputs come from fresh solves of the [public Landreman-Paul QA example](https://github.com/uwplasma/VMEX/blob/757fe28348d959d5d4a50653d25fcca67c8fb941/docs/_static/figures/sources/make_extender_islands_figure.py), with ESSOS 0.22.0 and virtual-casing-jax 0.0.17: `current.json` with 4 kA of toroidal plasma current and `beta1.json` with none, both at about 1 % volume-averaged beta, 0.025 Wb inside the boundary. `examples/vmex_surface_current/prepare.py` converts a VMEX `wout` and its ESSOS coils (it needs that VMEX revision and ESSOS):

```bash
python examples/vmex_surface_current/prepare.py wout_current.nc \
  ESSOS_biot_savart_LandremanPaulQA.json current.json --plasma-current 4000
```

`python examples/vmex_surface_current/verify.py` compares 20 plasma-field vectors with VMEX's own virtual-casing evaluation and follows the outboard island of the 4 kA case. `python examples/surface_current/compare.py sheet.json probes.json > comparison.csv` compares your own plasma-only field vectors (`points` in Cartesian metres, `B` in tesla) with the sheet's field and with its refinement.

## Cost

Measured on shared cores. The sheet's quadrature is built once per source, in seconds. Before anything else torux measures the flux of the source's own field through Γ: 43 s on the VMEX case, 2 to 3 minutes on the QUASR null case, 101 minutes on a 46,491-mode sheet (HSX with a current filament), and a 108,608-mode sheet had not finished it in 3 hours. Each start torus takes 1 to 4 minutes on the VMEX case. Found and refined, then independently checked: the exact ring (65 modes) in 3.5 to 9 minutes and then 18 to 33 more; the QUASR null case (3,870 modes) in 14 to 20 minutes and then 64 to 71 more; the VMEX case at 0.045 Wb in 9 to 18 minutes and then 10 minutes to 3 hours, the longest where the check's double precision reads too much and its double-double tier decides. Next to Γ the engine's field costs 11 to 70 ms a point. The default budget of a source with a sheet is 900 s and a quarter of a second per mode of the sheet, twice that without a request; the independent check runs past it.
