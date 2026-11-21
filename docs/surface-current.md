# Outside a plasma: coils and a current sheet

A finite-beta equilibrium's field outside its plasma is the coils' field plus the field of the plasma's currents, and outside the plasma's boundary Γ the second is exactly the field of a current sheet on Γ: K = n × B / μ₀, B the total field on Γ (virtual casing). `torux.SurfaceCurrent` holds Γ, that current and the toroidal flux inside Γ. With the coils it is a source like any other: `torux.solve` returns the flux surface of the total field outside Γ, verified as a vacuum surface is, or a truthful `undetermined`, and `torux.check` measures a surface against it.

```python
import torux

src = torux.load("examples/vmex_surface_current/current.json")   # coils, and Γ with its current: 0.025 Wb inside
s = torux.solve(src, flux=0.026)          # the full toroidal flux: 0.025 Wb inside Γ and 0.001 Wb between Γ and the surface
s = torux.solve(src, largest=True)        # the largest surface torux can verify outside Γ
print(torux.check(src, "surface.json"))   # a surface from another code, against the same source
```

## What a surface means here

The same as in vacuum, by the same rules ([what is checked](verification.md)): the solver's own check at most 10⁻¹⁴ of |B| off its grid, the resonance guard, the coils' quadrature check, simple sections, and the independent evaluator (`torux.reference`, which evaluates the sheet with a quadrature of its own and shares no code with the solver's) at most 10⁻¹⁴ with its flux within 10⁻¹². A sheet adds three checks of the solver's: every sampled point of the surface outside Γ and at least 10⁻⁵ of Γ's minor radius from it, every section of the surface once round Γ's section, and the sheet's field on the surface moving by at most 2 × 10⁻¹⁵ of |B| when its quadrature is refined (Gauss order 16 and panels used from 3 radii, against 12 and 2).

Inside Γ the source's field is not the plasma's: for K = n × B / μ₀ it is zero. So nothing inside Γ is evidence of anything. There is no magnetic axis to start from: the family of surfaces starts next to Γ, from Γ itself moved out a little and carried into the field's straight-field-line angles, and is continued outward. No surface is accepted that reaches Γ or passes inside it, no closed field line of a chain is listed that comes within that distance of Γ, and there are no certificates that a surface cannot exist: their theorems hold for tori with no current inside, and a plasma has current inside. Where torux finds no surface the answer is `undetermined` and says why.

## The flux

A request is the usual full toroidal flux. The part carries the flux inside Γ (`flux=`, the equilibrium's own: VMEC's `phiedge`, DESC's `Psi`), and the full flux of a surface outside is that plus the flux of the total field between Γ and the surface. torux measures the second as the difference of the loop integrals of A round the surface and round Γ, the second from outside: the flux of the source's own field through Γ's sections, which is zero for an exact virtual-casing current and is reported (`answer.evidence['sheet']['own_flux_Wb']`), with how much it differs between two sections (`own_flux_spread_Wb`), which is how far Γ is from a flux surface of the field outside it. The engine's vector potential carries the flux inside Γ as a flux tube on a curve inside Γ, so a loop integral of A outside Γ gives the full flux; B is unchanged.

## How close to Γ

The sheet's field (`torux.field.surface_current`) sums the series exactly at every quadrature node and forms every separation of a target from a node relative to a panel corner kept in double-double, so its accuracy does not fall as a target nears Γ. Measured:

MEASUREMENTS

A verified surface keeps at least 10⁻⁵ of Γ's minor radius from Γ, the distance down to which these measurements reach; a request whose surface would come nearer is `undetermined` and says so.

## Input

- `modes`: integer `(m, n)` pairs, one each; every coefficient has the phase `m*theta - n*nfp*phi`, both angles in radians, `phi` the cylindrical angle. `(0, n)` and `(0, -n)` may both appear.
- `rmnc`, `rmns`, `zmnc`, `zmns`: Γ's `R` and `Z` in metres (`rmns`, `zmnc` may be omitted). Each constant-`phi` section must run counterclockwise in `(R, Z)` as `theta` increases. `n` is the outward unit normal, along `r_phi × r_theta`.
- The current, as a potential: `kappac`, `kappas` in amperes, with the secular terms `G*phi/(2*pi) + I*theta/(2*pi)`, and `K = n × grad_s kappa`. `G` is the current through the hole along `+z` linked by a loop along `+phi` on Γ; `I` is minus the net toroidal current along `+phi` inside Γ. For a convention `K = grad_s Phi × n`, give `kappa = -Phi`, both secular terms included.
- Or as the tangential field: `covariant`, shape `(2, 2, len(modes))`, the cos and sin coefficients of `B·r_theta / mu0` and of `B·r_phi / mu0`, `B` the total field on Γ (coils and plasma). Both components are kept, so a current that is not a gradient is represented too.
- `flux`: the toroidal flux inside Γ in Wb. Its magnitude: the sign is the field's. Needed by `torux.solve` and for the full flux of `torux.check`. Field evaluation and Poincaré sections need none.

`Source.save` writes the part with the coils in one torux JSON file. With the coils and the sheet in two files, `torux.load("coils.json", "sheet.json", nfp=4)` verifies the symmetry of the whole: the coils' field by probing, the sheet's from its coefficients. `torux.solve` takes a source with one `SurfaceCurrent`, with stellarator symmetry (the source's, verified, and the sheet's coefficients: sine parts of `R`, cosine parts of `Z` and of `kappa`, or the sine parts of `covariant`, at most 10⁻⁹ of the rest).

## What works, and what refuses

Work as in vacuum: a flux, `minor_radius` and `volume` (searched from Γ's own size), `largest=True`, `torux.check`, `torux.many`, the CLI, `answer.plot`, `answer.boozer()`, `answer.system`, and `torux.poincare` with `starts=`. The chains a request meets are listed as in vacuum, without the lines that come near Γ, and their widths closed in on by verified surfaces as in vacuum (`widths_s`).

Refused, each in one sentence: a source with several `SurfaceCurrent`s, one without `flux`, a flux not more than the flux inside Γ, a size not larger than Γ's, a source without stellarator symmetry, `gradients=True` and `answer.vjp` (the field is not differentiated in the sheet), `trace_lines` (the reference tracer does not take a sheet), `answer.render` and `torux render` (they light the surfaces inside the answer's by field lines, through Γ), and `torux.poincare` without `starts=`.

## Field, potential and sections

`FieldEngine(src)` gives B, A and grad B at Cartesian points outside Γ; a point inside Γ, or nearer than the quadrature resolves (about 10⁻⁸ of the root panel size), is refused. `FieldEngine(src, sheet_order=16, sheet_eta=3.0)` refines the sheet's quadrature: the change measures its error. `torux.poincare(src, starts=[(R, Z)])` follows field lines outside Γ through planes of constant toroidal angle; a line stops at its last accepted point when it reaches Γ or loses its toroidal direction, and `stop_reason` says which, `left_at` where.

## The public VMEX case

VMEXRESULTS

The bundled JSON inputs come from fresh solves of the [public Landreman-Paul QA example](https://github.com/uwplasma/VMEX/blob/757fe28348d959d5d4a50653d25fcca67c8fb941/docs/_static/figures/sources/make_extender_islands_figure.py), with ESSOS 0.22.0 and virtual-casing-jax 0.0.17: `current.json` with 4 kA of toroidal plasma current and `beta1.json` with none, both at about 1 % volume-averaged beta, 0.025 Wb inside the boundary. `examples/vmex_surface_current/prepare.py` converts a VMEX `wout` and its ESSOS coils (it needs that VMEX revision and ESSOS):

```bash
python examples/vmex_surface_current/prepare.py wout_current.nc \
  ESSOS_biot_savart_LandremanPaulQA.json current.json --plasma-current 4000
```

`python examples/vmex_surface_current/verify.py` compares 20 plasma-field vectors with VMEX's own virtual-casing evaluation and follows the outboard island of the 4 kA case. `python examples/surface_current/compare.py sheet.json probes.json > comparison.csv` compares your own plasma-only field vectors (`points` in Cartesian metres, `B` in tesla) with the sheet's field and with its refinement.

## Cost

COST
