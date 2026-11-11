# How each tool was run

[`../compare300.csv`](../compare300.csv) has one row for each of 300 QUASR coil sets and each of six tools. This folder is how those rows were made: the file every tool was given, the program that ran each tool, and two commands that make any row again.

If you wrote one of these tools, your section is short and your program is one file. A setting you would have chosen differently is worth an issue.

## The same for all six

- **The coil sets.** 300 of QUASR's 371,701, drawn by `numpy.random.default_rng(20261107).choice(371701, 300, replace=False)` over the ids in ascending order.
- **The problem.** [`inputs/quasr_<id>.json`](inputs) holds the coils as Fourier coefficients, every copy written out, and the toroidal flux asked for: the flux QUASR's outermost stored surface encloses. Every tool is given exactly these numbers. The file also holds that stored surface, as SIMSOPT's coefficients and as QUASR's VMEC namelist. A tool may start from it and from anything it computes from the coils.
- **The budget.** 3,600 seconds, 12 GiB of memory, four compute threads on eight cores of an AMD EPYC 7742 that the run had to itself. No graphics card.
- **The code.** Each tool as released, unmodified: SIMSOPT 1.11.1, DESC 0.17.3, VMEC++ 0.9.3, torux 0.1.0 (this repository), under Python 3.12.14. [`requirements/`](requirements) has what each environment was built from and, in the `.freeze.txt` beside it, every package it then held. The programs in this folder are the ones that made the rows, with what tied them to the machines they ran on taken out: each tool's calls and settings are the same, statement for statement.
- **The settings.** Five of the six tools need them, and the choice is left to whoever runs the tool. Ours: the route the tool's documentation and examples recommend for its most accurate answer, then pushed wherever that paid inside the budget: resolution, tolerances, the quadrature of the coil field, continuation, restarts. Every choice was made on six designs outside the 300 ([`inputs/pilot/`](inputs/pilot)), and each program says beside each setting what was measured there. They are our choices and not the authors': on single designs a different choice does better, and for DESC a different route exists (`SurfaceQuadraticFlux` with the coils fixed) that these rows do not use. torux is given the coils and the flux and nothing else.
- **The answer.** A surface counts only if the tool's own criterion says it converged. Among several converged candidates the tool's own diagnostics choose. The evaluator is never asked.
- **The time.** `wall_s` is the whole run: from reading the input to the end, the tool's own checks included, in a process that has already imported the tool and finished one untimed solve of a pilot design. `answer_s` is when, inside that run, the tool had the surface it returned. The two are the same for torux, which stops there. The other five were set to spend their hour on accuracy and can go on after the level they return, so only `answer_s` is their speed, and it is the time the README shows. A timed run counts only if its threads waited for a processor for at most 5% of the time they ran and other work used its cores for at most 5% of that ([`fair_cores.py`](fair_cores.py)). Otherwise the design was run again.
- **The judge.** [`../evaluator`](../evaluator), which shares no code with any tool: the largest |B·n|/|B| at 4,000 random points of each returned surface, in extended precision. It needs numpy and nothing else, and judged these rows with numpy 2.3.5 under Python 3.14.4.

The row `quasr-stored` is the stored surface itself, judged like an answer: where every tool starts.

## SIMSOPT BoozerExact

[`run_boozer_exact.py`](run_boozer_exact.py). `BoozerSurface.solve_residual_equation_exactly_newton` on a stellarator-symmetric `SurfaceXYZTensorFourier` with mpol = ntor = M, on SIMSOPT's half-period collocation grid.

| | |
|---|---|
| resolution | a ladder: QUASR's own M (10 or 4 on nearly every design), then 12, 14, ..., 28, 32, 36, 40. Each level starts from the one before, padded exactly with zeros |
| Newton | tolerance 10⁻¹⁴ per equation relative to G·\|B\|, with 30 iterations at the first level and 10 after. SIMSOPT's default, 10⁻¹³ absolute, sits under the rounding floor from M = 26 and called converging levels failed |
| start | G and ι by least squares on the stored surface. If the first level fails: QUASR's ι, then a BoozerLS pre-solve |
| coil field | `BiotSavart` at twice the first quadrature that agrees with its double to 10⁻¹² of \|B\|: 192 to 800 points a coil, where QUASR's files carry 160 |
| flux label | `ToroidalFlux` on max(256, 8M) points |
| the answer | the converged level with the smallest Boozer residual at (4M + 2)² points off the collocation grid |

Time never limits it: its slowest run took 431 s. Memory does, and part of that memory is this program's and not SIMSOPT's: the program keeps the level before alive during a level, and a finished solve stays in memory until Python's collector next runs. So of the numbers that follow, the stops at 12 GiB are the program's doing. 149 of its 288 surfaces are at M = 40, the top of the ladder. 50 runs passed 12 GiB inside a level and returned the best level before it. The 12 that returned nothing never converged at their first levels, and ended when the pre-solve of M = 12, which holds 21.6 GB by itself, passed 12 GiB.

What one more level adds: thirty of these designs were also run with 16 GiB and a ladder to M = 48, and on twelve of them M = 40 and M = 44 both converged. By SIMSOPT's own measure of |B·n|/|B| off the grid, M = 44 leaks a median 1.7 times less than M = 40: 20 times less at most, and on three of the twelve up to 1.13 times more. On six of the twelve the answer itself, the level with the smallest off-grid residual, moved to M = 44 ([`boozer_exact_m44.csv`](boozer_exact_m44.csv)).

## SIMSOPT BoozerLS

[`run_boozer_ls.py`](run_boozer_ls.py). `BoozerSurface` with a penalty on the flux label, as in SIMSOPT's `examples/2_Intermediate/boozer.py`: dense BFGS (`minimize_boozer_penalty_constraints_LBFGS`) at the first level, then SIMSOPT's Levenberg-Marquardt (`minimize_boozer_penalty_constraints_ls`, `method="manual"`) at every level.

| | |
|---|---|
| resolution | the same ladder and surface as BoozerExact, to M = 32, with the residual on 1.5 (2M + 1) points each way over a field period, weighted by 1/\|B\| (SIMSOPT's default) |
| flux label | penalty weight 1/target², so that the penalty is on the relative flux error. A fixed weight of 100 let a 0.002 Wb flux slip by 18% |
| Levenberg-Marquardt | 40 iterations a call, run to its rounding floor. A level has converged when \|Jᵀr\| ≤ 3 × 10⁻¹⁴ G² (M/10)² and the flux is within 10⁻⁴ of the request. With the example's fixed 10⁻¹² a level that already met it never moved |
| not `run_code` | its Newton needs the flux label's Hessian, which SIMSOPT 1.11.1 builds as four dense arrays: 25 GiB at M = 20 |
| start, coil field, the answer | as BoozerExact |

What limits it here is memory again: 209 of its 283 surfaces are at M = 32, and M = 36 needs about 14 GB. 12 runs were stopped by the hour.

## SIMSOPT QFM

[`run_qfm.py`](run_qfm.py). `QfmSurface.minimize_qfm_penalty_constraints_LBFGS` on a stellarator-symmetric `SurfaceRZFourier` with mpol = ntor = M, called again from where it stopped until SIMSOPT reports success.

| | |
|---|---|
| resolution | a ladder M = 4, 6, ..., 24, 28, 32, then three more passes at M = 32. Quadrature: 2 (2M + 1) points in θ and half as many on SIMSOPT's half period in φ |
| tolerance | at each level, 10⁻⁵ of the objective at its start (10⁻⁶, 10⁻⁷, 10⁻⁸ in the last three passes). The example's 10⁻¹² reports success at \|B·n\|/\|B\| near 10⁻⁵ |
| flux label | penalty weight 1/target². No SLSQP stage: SIMSOPT writes its exact constraint as a squared error, whose gradient vanishes at the solution, and it failed on every pilot |
| start | `to_RZFourier()` of the stored surface or QUASR's VMEC boundary, whichever has the smaller QFM residual, cut to M = 4 |
| the answer | the converged pass with the smallest QFM residual on a (4M + 2)² grid off the quadrature points |

Its `wall_s` is the budget by construction: the last passes spend whatever is left of the hour, and 238 of its 297 runs were stopped by the hour. One, design 0003051, was stopped at 12 GiB inside those passes. 290 surfaces are at M = 32, where the ladder ends by setting.

A second run is not the first again. Two runs of this program on one machine, the ladder cut at M = 10 to keep them short (`--set 'levels=[4,6,8,10]' --set 'polish_rels=[]'`), returned two surfaces for design 0003051: by SIMSOPT's own measure off the grid one leaks 2.4 × 10⁻⁴ of |B| and the other 1.8 × 10⁻⁴ ([`qfm_twice.csv`](qfm_twice.csv)). So a QFM row run again agrees with the table in size, not in digits: design 0003051 came back at 6.9 × 10⁻⁶ where the table has 5.8 × 10⁻⁶ ([`reruns.csv`](reruns.csv)).

## DESC

[`run_desc.py`](run_desc.py). DESC's documented free-boundary route for a vacuum stellarator: `VacuumBoundaryError(field_fixed=True)` minimised by `proximal-lsq-exact` under `ForceBalance`, with pressure and current fixed at zero and Ψ at the requested flux. The surface judged is the boundary of that equilibrium.

| | |
|---|---|
| resolution | a ladder M = N = 6, 8, 10 with L = 1.5 M and DESC's default grids. A level is begun only if its time and memory, estimated from the level before, fit what is left |
| coils | one `FourierXYZCoil` for each input coil, 512 nodes each, doubled until B settles to 10⁻¹⁴. DESC's default for these coils, 75 nodes, leaves B wrong by up to 4 × 10⁻⁴ |
| start | the stored surface, fitted by `FourierRZToroidalSurface.from_values`, and the interior by `Equilibrium.solve`, or DESC's automatic continuation where that is not nested |
| stopping | outer: ftol 10⁻³, xtol and gtol 10⁻⁸, 100 iterations, restarted twice from where it stops. Inner: ftol 10⁻⁶, xtol 10⁻¹⁰, gtol 10⁻¹², 50 iterations |
| the answer | the highest level where DESC reports success and `Equilibrium.is_nested()` holds. DESC's tutorial settings reported success on two pilots with a force error of order one |

Every one of its 199 surfaces is at the first level, M = N = 6: a boundary of 85 Fourier terms, where the stored surface it starts from has a median 661. M = N = 8 was never begun, on any design. On 133 of the 199 the program's estimate of the memory it would need (1.9 times the first level's peak) came to more than 12 GiB, and on 66 its estimate of the time (three times the first level's) came to more than was left. Neither estimate was put to the test, and the first looks wrong: in each of those processes the untimed warm-up had just run a step at M = N = 8 on a pilot design, and the process never held more than 9.6 GiB ([`desc_ladder.csv`](desc_ladder.csv)). So this row is DESC at its lowest resolution on four processor threads: less than DESC could have done inside this budget, and far below what it is used at on a graphics card. Of the 101 without a surface, 64 ended in DESC's "A bad approximation caused failure to predict improvement".

## VMEC++

[`run_vmecpp.py`](run_vmecpp.py). A free-boundary equilibrium with no pressure and no current at the requested flux. The surface judged is its boundary. VMEC++ is given the coils' own field as a `MagneticFieldFunction` (Biot-Savart converged to 10⁻¹⁴, no grid in between).

| | |
|---|---|
| resolution | eleven levels, each hot-restarted from the one before: (ns, mpol, ntor, ftol) = (16, 11, 10, 10⁻¹⁰), (51, 11, 10, 10⁻¹¹), (101, 13, 12, 10⁻¹¹), (201, 15, 14, 10⁻¹¹), then ftol 10⁻¹² down to 10⁻¹⁸ at (201, 15, 14). On the pilots more modes and more surfaces changed nothing, and tighter ftol did |
| vacuum | NESTOR, or `only_coils` where NESTOR does not converge |
| start | QUASR's namelist boundary. If VMEC++ cannot begin from it: a fixed-boundary equilibrium inside it, then inside smoothed and shrunk copies of it |
| the answer | the highest level that met VMEC++'s own test (`ier_flag` 0 and fsqr, fsqz, fsql under ftol) |

Its `wall_s` is the budget by construction too: the ladder tightens until the hour ends, as it did for 168 of its 252 surfaces. 57 went through the last level.

## torux

[`run_torux.py`](run_torux.py). `torux.api.solve(torux.Request(source, flux, budget_s=3300, tol=3e-15, verify=True))`: the solver's one call at its defaults, the call `torux.solve` answers a request for a flux through, with the hour less a margin for its last step. The source is the same coils with the stored surface as the start, and nothing else of QUASR's is used. A surface counts only when torux says `surface`: its own acceptance and its independent check.

Three designs (0042793, 0946659, 0954058) never pass the timing rule: in every one of their 95, 64 and 34 timed runs torux's own threads waited for a processor for 5.1 to 14.3% of the time they ran, with other work on the cores at most 1.3%, and the outcome was the same every time. Each is counted by its run of median time.

## Run a row again

```sh
pip install -r results/compare300/requirements/simsopt.txt       # or desc.txt, vmecpp.txt, torux.txt: one environment each
python results/compare300/run_boozer_exact.py 0003051            # the tool on QUASR 0003051, as above
python results/compare300/row.py simsopt-boozer-exact 0003051    # the judge on that answer, above the table's row
```

Give the run eight cores of its own and 12 GiB, on Linux and an x86-64 processor: the timing rule reads Linux's scheduler, and the judge's extended precision is that processor's 80-bit arithmetic. Elsewhere the programs run without the meter, and `row.py` gives the judge's reading in double precision and says so: on torux's surfaces that reading is the judge's own rounding, a median 1.6 × 10⁻¹⁴. Design 0003051, run again this way on another processor of the same family ([`reruns.csv`](reruns.csv)):

| | leak, run again | leak, the table | seconds, run again and in the table |
|---|---|---|---|
| torux | 7.2 × 10⁻¹⁶ | 8.4 × 10⁻¹⁶ | 4.1, 3.9 |
| SIMSOPT BoozerExact | 5.199098215 × 10⁻⁷ | 5.199098233 × 10⁻⁷ | 117, 116 |
| SIMSOPT BoozerLS | 6.669734076 × 10⁻⁷ | 6.669734075 × 10⁻⁷ | 973, 836 |
| VMEC++ | 1.4402278 × 10⁻³ | 1.4402276 × 10⁻³ | 1,686, 1,549 |
| SIMSOPT QFM | 6.9 × 10⁻⁶ | 5.8 × 10⁻⁶ | 3,600, 3,389 |

The accuracy comes back to the table's within the digits another processor's arithmetic changes, and torux's is rounding both times. QFM is the exception, as its section says: its runs do not repeat. The times are the machine's.
