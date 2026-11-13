# Formats

## Sources

`torux.load(path)` reads the format from the file itself. `torux.solve(path)` calls it for you.

| source | what it gives | notes |
|---|---|---|
| **QUASR download** (`serial*.json`) | coils as Fourier filaments, the design's stored surfaces, and the request | the flux of the outermost stored surface becomes the request, and nothing else is needed |
| **QUASR by number** (`"quasr:954"`) | the same, downloaded once | kept in torux's cache, and needs the network the first time only |
| **SIMSOPT JSON** | every coil of a `BiotSavart` object | `CurveXYZFourier`, `RotatedCurve`, `CurvePlanarFourier`, `CurveXYZFourierSymmetries` and its earlier name `CurveXYZHelical` converted exactly, framed and winding-surface curves by FFT with a measured error. Several coil sets in one file: `which=` picks one (`--coils` on the command line), and with several such files, a list of one entry per file (`--coils` once per file). A `Current` whose two values differ is refused until `currents=` chooses one ([below](#a-current-with-two-values)) |
| **SIMSOPT objects** | a live `BiotSavart`, `Coil` or list of `Coil`s | through SIMSOPT's own `save`, then as the JSON, each `Current` with the value it holds, `get_value()`, the one SIMSOPT computes with |
| **Fourier-filament JSON** | closed filaments | `{"nfp", "stellarator_symmetric", "coils": [{"xc", "xs", "yc", "ys", "zc", "zs", "current_A"}]}`, every coil of the device written out, and optionally `"requested_toroidal_flux_Wb"` |
| **MAKEGRID** (`coils.*`) | closed polygons | circular coils on the axis become rings |
| **FOCUS** coil tables | Fourier filaments | give `nfp=` |
| **FOCUS / FAMUS** dipole tables | point dipoles (permanent magnets) | give `nfp=`. Usually loaded together with the coils |
| **GPEC** coil geometry (`.dat`) | closed polygons, 1 A a coil | a tokamak source by default |
| **wireframe CSV** | straight segments with their currents | give `nfp=` to have the symmetry verified and used |
| **TokaMaker mesh** (`.h5`) | axisymmetric conductors by cross-section | a tokamak source by default. Needs `h5py` |
| **DESC coil set** (`.h5`) | every coil of a `CoilSet` as a Fourier filament | needs `h5py` |
| **DESC objects** | a live `CoilSet`, `MixedCoilSet` or coil | through DESC's own `save`, then as the `.h5`. Needs `h5py` |
| **torux JSON** | a `Source` saved with `Source.save` | |

Several files make one source: `torux.load("tf.coils", "magnets.focus", nfp=2)` is the coils and the magnets together.

`kind=` says what sort of source it is. `"stellarator"`, the default, means its surfaces are sought as tori. `"tokamak"` and `"dipole"` are settled by proof: a tokamak's own coils and a levitated dipole make no toroidal flux surface in vacuum, and torux says why.

A claimed symmetry is used only after it has been verified on the parts themselves.

## Surfaces `torux.check` reads

| surface | notes |
|---|---|
| **VMEC or VMEC++ `wout`** (`.nc`) | the boundary, or with `which=` a radial index. NetCDF3 needs nothing more, NetCDF4 needs `h5py` or `netCDF4` |
| **VMEC input** (`input.*`) | its boundary, with `PHIEDGE` as the flux it states |
| **SIMSOPT or QUASR JSON**, or `"quasr:954"` | the outermost surface in the file, or `which=` by name or index. The surface of a `BoozerSurface` or `QFMSurface` is checked only against the coils it was computed for ([below](#a-surface-and-the-coils-it-was-computed-for)) |
| **a saved torux answer** (`.json`) | |
| **SIMSOPT objects** | `SurfaceRZFourier`, `SurfaceXYZTensorFourier`, anything with a `.surface` such as a `BoozerSurface` |
| **DESC objects** | `FourierRZToroidalSurface`, or an `Equilibrium`, whose `Psi` is the flux it states |
| **VMEC++ objects** | the output of `vmecpp.run`, or its `wout` |
| **a torux answer** | |

## SIMSOPT files torux refuses rather than guess

### A `Current` with two values

SIMSOPT writes a `Current` object's value twice: as its `current`, the value it held when the file was saved, and in its degrees of freedom, the value SIMSOPT itself takes when it reads the file back. In a file written the ordinary way they are the same number. A value set past the degrees of freedom makes them differ, and then the file does not say which current its coils carried. torux refuses such a file and names the object and both values. `torux.load(path, currents="saved")` takes the value each object held when saved, `currents="dofs"` the one SIMSOPT reads back, and on the command line `--currents saved` or `--currents dofs`. A check of a surface against coils read either way says which was taken.

A live SIMSOPT object is not a file: SIMSOPT computes its field with the value each `Current` holds, and torux reads it with that value without asking.

### A surface and the coils it was computed for

A `BoozerSurface` or `QFMSurface` in a SIMSOPT file names the `BiotSavart` it was computed for. `torux.check` of its surface against other coils compares their field with that `BiotSavart`'s on the surface, and if they differ by more than 10⁻⁹ of |B| it refuses: what crosses the surface then measures the mismatch of the two coil sets, not the surface. Give the surface's file as the source too, with `which=` (`--coils`) naming its coils where the file holds several.

## What an answer writes

`answer.save(path)` chooses by the name.

| name | what is written |
|---|---|
| `input.<name>` | a VMEC input file: the boundary with every coefficient (or cut with `mpol=`, `ntor=`), the flux as `PHIEDGE`, zero pressure and current. VMEC, VMEC++, DESC and SIMSOPT read it |
| `*.json` | the whole answer, which `torux.Answer.read` reads back |
| `*.vtk` | the surface as a mesh with \|B\| on it |
