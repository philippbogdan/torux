# Exterior fields from surface current

`SurfaceCurrent` adds the plasma's equivalent surface current to the external coils of a supplied equilibrium. Use `FieldEngine` for the field, its Cartesian derivative and vector potential, or `torux.poincare` for an explicitly seeded exterior section.

## Try the example

```bash
python -m pip install 'torux==0.4.0'
curl -fL https://raw.githubusercontent.com/philippbogdan/torux/v0.4.0/examples/vmex_surface_current/current.json -o vmex-4ka.json
torux poincare vmex-4ka.json --planes 0 --start 1.31843703,0 --turns 2
```

The launch is near the outboard island centre. The command follows four field periods, which is two toroidal turns on this device. The Python API returns the section crossings in metres.

```python
import torux
from torux.field import FieldEngine

src = torux.load("vmex-4ka.json")
b = FieldEngine(src).B([[1.31843703, 0.0, 0.0]])  # Cartesian metres in, tesla out
p = torux.poincare(src, planes=[0.0], starts=[(1.31843703, 0.0)], turns=2)
print(p.R[0, 0], p.Z[0, 0])
```

## Input

Add a `torux.SurfaceCurrent` to `Source.parts` alongside the coil parts. `Source.save` writes a portable Torux JSON file. The source's `nfp` is the number of field periods shared by the complete field. The sheet's `nfp` sets its own Fourier harmonics.

- `modes` gives integer `(m, n)` pairs. All coefficients use the phase `m*theta - n*nfp*phi`, with both angles in radians.
- `rmnc`, `rmns`, `zmnc` and `zmns` give the cosine and sine coefficients of cylindrical `R` and `Z` in metres. Omitted `rmns` and `zmnc` are zero. Each constant-`phi` cut must be simple, have positive `R`, and run counterclockwise in `(R, Z)` as `theta` increases. The outward normal follows `r_phi × r_theta`.
- `kappac` and `kappas` give the periodic current potential in amperes. The secular terms are `G*phi/(2*pi) + I*theta/(2*pi)`. The current convention is `K = n × grad_s kappa`.
- For a supplied tangential boundary field, use `covariant` in place of the potential, with shape `(2, 2, len(modes))`. The first axis is the `theta, phi` order of `(B_total · r_theta, B_total · r_phi) / mu0`. The second is the cosine, sine order. Torux retains both angular components.

Put geometry and current coefficients on the union of their mode lists and use zeros for missing coefficients. For a convention using `K = grad_s Phi × n`, set `kappa = -Phi`, including both secular terms. To keep the coil and sheet files separate, save a `Source("sheet", [sheet], nfp=sheet.nfp)` to `sheet.json` and use `torux.load("coils.json", "sheet.json", nfp=4)` for a four-period system.

The evaluator uses adaptive Gaussian panels for the Biot-Savart integral. The geometry and current density at panel nodes come from high-order periodic interpolation of a spectral source grid. The target field is evaluated directly. Use `FieldEngine(src, sheet_order=12, sheet_samples=512)` to refine both parts of the discretization and compare at the distances you need. The defaults are 10 Gaussian nodes per panel dimension and at least 256 source samples per angular direction. The reported accuracy of the VMEX example applies to its sampled points.

The panel cache stores one field period and uses at most 32 MiB per sheet. When finer levels fall outside the cache, Torux generates each panel as needed with the same quadrature. The spectral source grid has a separate memory cost which grows with Fourier resolution.

The evaluator rejects targets inside the boundary or within a buffer given by the error bound for approximating a cut by a polygon. Exterior Poincaré sections use toroidal angle as the integration coordinate. A trajectory stops at its last accepted point when it reaches an exterior cutoff or loses its toroidal direction. The per-line `stop_reason` reports why, and `left_at` gives the last accepted `(R, phi, Z)`, with `phi` measured from the launch plane.

The equilibrium and its current sheet are inputs. The flux-surface solver and the independent flux-surface verifier support vacuum sources. For sources with a sheet, `FieldEngine` takes concrete point arrays and fixed source coefficients. Use discretization refinement to assess the sheet field error.

## Compare your own exterior probes

From a Torux source checkout, the [comparison script](../examples/surface_current/compare.py) takes a Torux source and a small probe file. The probe file holds `points` in Cartesian metres and `B` in Cartesian tesla. Supply the **plasma-only** reference vectors. The script extracts the sheets from the source, even when it also contains coils, and writes one CSV row per point with both field evaluations, their errors against the reference and the change under refinement.

```bash
python examples/surface_current/compare.py sheet.json probes.json > comparison.csv
python examples/surface_current/compare.py examples/vmex_surface_current/current.json \
  examples/vmex_surface_current/reference.json --case current
```

## Public VMEX check

```bash
python examples/vmex_surface_current/verify.py
```

The script compares 20 plasma-field vectors against VMEX's direct virtual-casing evaluation, solves for the 4 kA elliptic orbit and checks its exterior clearance and two-turn return map. It checks a wider launch that reaches the exterior cutoff and traces the corresponding centre launch in the 0 kA case. Add `--plot path/stem` to save the outboard section as PDF, PNG and CSV with Matplotlib.

The 4 kA orbit is centred 4.976 mm outside the outboard boundary at `phi = 0`. The maximum absolute field discrepancy over both cases was 8.9 × 10⁻¹³ T. The example's tabulated field, evaluated separately on the same equilibrium, puts the centre at 4.53 mm. The wider launch reaches the cutoff after about 14.4 turns and the full island width remains open.

The bundled JSON inputs come from fresh solves of the [public Landreman-Paul QA example](https://github.com/uwplasma/VMEX/blob/757fe28348d959d5d4a50653d25fcca67c8fb941/docs/_static/figures/sources/make_extender_islands_figure.py), with ESSOS 0.22.0 and virtual-casing-jax 0.0.17. `current.json` uses 4 kA of toroidal plasma current and `beta1.json` uses zero. Both are at about 1% volume-averaged beta. The reference plasma field uses 256 × 256 boundary samples and VMEX graded quadrature with 288 angular and 1920 radial nodes at outward offsets of 1, 4, 11, 25 and 50 mm from each of two angular sites. The sites are `(theta, phi) = (0, 0)` and `(pi/4, pi/4)`.

To convert a VMEX equilibrium with its ESSOS coil file, use `examples/vmex_surface_current/prepare.py` from an environment with that VMEX revision and ESSOS installed.

```bash
python examples/vmex_surface_current/prepare.py wout_current.nc \
  ESSOS_biot_savart_LandremanPaulQA.json current.json --plasma-current 4000
```
