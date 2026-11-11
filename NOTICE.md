# Credits

torux's own code, results, text and figures are public domain ([CC0 1.0](LICENSE)), all but one picture: the twelve machines ([below](#behind-the-pictures)). What it was run on is other people's work, and keeps their licences.

## Bundled inputs (`examples/`)

| file | from | licence |
|---|---|---|
| `quasr_0000954.json`, `quasr_0219457.json`, `quasr_1100073.json` | QUASR v4, the QUAsisymmetric Stellarator Repository, Andrew Giuliani, [doi:10.5281/zenodo.13717741](https://doi.org/10.5281/zenodo.13717741) | MIT |
| `hsx.json`, `w7x.json` | the HSX and Wendelstein 7-X coil sets shipped with [simsopt](https://github.com/hiddenSymmetries/simsopt) 1.11, rewritten as Fourier-filament JSON | MIT |
| `d3d_icoils_upper.dat` | DIII-D's upper I-coils, from the coil library of [GPEC](https://github.com/OpenFUSIONToolkit/GPEC) | MIT |

`results/compare300/inputs/`: 306 coil sets of QUASR v4 (as above, MIT), each rewritten as Fourier-filament JSON with its outermost stored surface, its VMEC namelist and its row of QUASR's table.

The MIT licence texts are in [`examples/LICENCES.md`](examples/LICENCES.md).

## Behind the results

`results/quasr.parquet`, `results/compare300.csv` and `results/refinement.csv` are measurements on coil sets of QUASR (as above).

## Behind the pictures

The pictures are torux's own renderings. The coils and magnets drawn in them are other people's designs and data, under these licences. Checksums and origins of the files are in [`visuals/sources.json`](visuals/sources.json) and [`visuals/one-line.json`](visuals/one-line.json).

**The picture of the twelve machines is not CC0.** `visuals/any-coil-light.png` and `visuals/any-coil-dark.png` are excluded from the dedication. Three of the machines in them are drawn from files under copyleft licences (LHD: GPL-3.0, ITER and the dipole: LGPL-3.0) and three from CC BY 4.0 data (CNT, the wireframe, MUSE). Those six renderings stay under the terms of their sources, in that picture and where each machine is drawn alone in `visuals/sources/` (`heliotron`, `tokamak`, `dipole`, `interlinked`, `wireframe`, `magnets`).

| in `visuals/any-coil-*.png` and `visuals/sources/` | from | licence |
|---|---|---|
| QUASR 0124533 | QUASR v4, the QUAsisymmetric Stellarator Repository, by Andrew Giuliani, [doi:10.5281/zenodo.13717741](https://doi.org/10.5281/zenodo.13717741) | MIT |
| HSX | HSX coil set from simsopt 1.11 (hiddenSymmetries/simsopt), in torux's Fourier-filament form, [github.com/hiddenSymmetries/simsopt](https://github.com/hiddenSymmetries/simsopt) | MIT |
| Wendelstein 7-X | Wendelstein 7-X coil set from simsopt 1.11 (hiddenSymmetries/simsopt), in torux's Fourier-filament form, [github.com/hiddenSymmetries/simsopt](https://github.com/hiddenSymmetries/simsopt) | MIT |
| LHD | LHD coil model (Y. Suzuki, 2017) from the FOCUS code's examples (PrincetonUniversity/FOCUS), [github.com/PrincetonUniversity/FOCUS](https://github.com/PrincetonUniversity/FOCUS) | GPL-3.0 |
| CNT | CNT coil set from "Update on the design of the Columbia Stellarator eXperiment" by A. Baillod, R. Lopez, A. Veksler, M. Campagna, D. Schmeling, E. Paul and A. Knyazev, [doi:10.5281/zenodo.18115389](https://doi.org/10.5281/zenodo.18115389) | CC BY 4.0 |
| STAR_Lite-A | STAR_Lite-A coil set (Harrer, Giuliani, Padidar, Davies, Naik and Lowe, 2026) shipped in simsopt's configs, [github.com/hiddenSymmetries/simsopt](https://github.com/hiddenSymmetries/simsopt) | MIT |
| wireframe (RCLS) | wireframe from "Data and code for A framework for discrete optimization of stellarator coils" by Kenneth Hammond, [doi:10.34770/6pbm-9981](https://doi.org/10.34770/6pbm-9981) | CC BY 4.0 |
| MUSE | MUSE zot43 magnet and coil model from the "QUADCOIL Quasi-single-stage Dataset" by Lanke Fu, [doi:10.5281/zenodo.19923438](https://doi.org/10.5281/zenodo.19923438) | CC BY 4.0 |
| US131 (umbilic) | US131 coil set from "Omnigenous umbilic stellarators: data and analysis files" by Rahul Gaur, [doi:10.5281/zenodo.15355215](https://doi.org/10.5281/zenodo.15355215) | MIT |
| ITER | ITER coil geometry of the TokaMaker examples (OpenFUSIONToolkit), [github.com/OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) | LGPL-3.0 |
| DIII-D upper I-coils | DIII-D upper I-coil geometry of the GPEC coil library (OpenFUSIONToolkit/GPEC), [github.com/OpenFUSIONToolkit/GPEC](https://github.com/OpenFUSIONToolkit/GPEC) | MIT |
| LDX-like dipole | LDX-like dipole of the TokaMaker examples (OpenFUSIONToolkit), [github.com/OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) | LGPL-3.0 |

`visuals/one-line-*`: the coils of Wendelstein 7-X, from the coil set shipped with [simsopt](https://github.com/hiddenSymmetries/simsopt) 1.11 (MIT).

`visuals/plot-*`: the coils of QUASR design 954, from QUASR v4 (as above, MIT), as `torux.solve("examples/quasr_0000954.json").plot` draws them.

GPL-3.0: [gnu.org/licenses/gpl-3.0](https://www.gnu.org/licenses/gpl-3.0.html). LGPL-3.0: [gnu.org/licenses/lgpl-3.0](https://www.gnu.org/licenses/lgpl-3.0.html). CC BY 4.0: [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/). Every picture is a new rendering of the geometry in the files named. Of those files, only the ones listed under bundled inputs are redistributed here.
