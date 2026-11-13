# Where things are

`src/torux`, about 48,000 lines of Python. Each module says in its own docstring what it does and why it is right. This is the map.

| | |
|---|---|
| **The tool** | |
| `answer.py` | `solve` and its `Answer`: the words, the numbers, the files it writes, the equations at any surface |
| `ask.py` | a surface asked for by its size, and the largest the coils hold |
| `check.py` | `check`: a surface from another code, measured against the coils |
| `batch.py` | `many`: every design in a process of its own, one row each |
| `load.py`, `sources.py` | `load`: a source from files, a QUASR number, a live SIMSOPT or DESC object |
| `plot.py`, `boozer.py` | the picture, and the Boozer spectrum and quasi-symmetry |
| `cli.py` | the `torux` command |
| `api.py` | the solver's one call per request: a stated budget, a truthful outcome |
| `source.py` | what a source is: filaments, polygons, segments, dipoles, rings, conductors |
| `io/` | one reader per file format |
| **The field** | |
| `field/` | the solver's field engine: B, its gradient and A at any point |
| `reference/` | the independent evaluator and tracer, which import nothing from the rest |
| **Surfaces** | |
| `axis.py`, `orbits.py` | the magnetic axis and closed field lines of any period |
| `torus.py`, `torus_cpu.py` | the invariant torus: the parameterisation method and its Newton solvers |
| `family.py` | continuation of surfaces from the axis outward, and refinement to the rounding floor |
| `design.py`, `axisselect.py` | starting from surfaces that come with a design |
| `seeds.py`, `nearaxis.py`, `noaxis.py` | surfaces that neither start reaches |
| `guard.py` | the resonance guard |
| `analysis.py` | the whole path for one source, stage by stage, inside its budget |
| `verify.py` | the independent verification and the acceptance rule |
| `*_general.py` | the same without stellarator symmetry |
| **When there is no surface** | |
| `chains.py`, `chainfind.py`, `manifolds.py` | island chains and the manifolds of their X lines |
| `islandband.py`, `certify.py` | the island-band certificates |
| `barrier.py`, `edgebound.py`, `exitcurve*.py`, `escape.py` | the edge: bounds on the flux any surface can enclose |
| `nonstell*.py` | sources that are not stellarators: a tokamak's coils, perturbation coils, dipoles |
| **Around them** | |
| `gradients.py` | adjoint gradients in every source parameter |
| `fasttrace.py`, `trace.py`, `cardtrace.py` | field-line tracers |
| `fast_*.py` | switches for work reductions that leave the answer unchanged |
| `budget.py`, `envfail.py` | the time budget, and telling a failing machine from a failing solve |

Tests are in `tests/` and need `pip install ".[test]"`: `python -m pytest` runs the quick ones (minutes), `python -m pytest -m slow` the certificates and the searches for a largest surface (up to three quarters of an hour each). The tests that hold the interface to SIMSOPT, DESC and VMEC++ skip where those are not installed.
