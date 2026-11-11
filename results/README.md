# Results

Two tables, and the scripts that turn them into every number and figure in the README.

```sh
python results/claims.py     # every number the README states
python results/figures.py     # every figure in visuals/ that shows data
```

## `quasr.parquet`: all of QUASR

One row for each of the 371,701 coil sets of [QUASR](https://quasr.flatironinstitute.org) (Giuliani 2024, [doi:10.5281/zenodo.13717741](https://doi.org/10.5281/zenodo.13717741)). The request is always the same: the surface enclosing the toroidal flux that the design's own outermost stored surface encloses.

| column | meaning |
|---|---|
| `id` | the QUASR design number |
| `nfp` | field periods |
| `flux_Wb` | the requested toroidal flux |
| `outcome` | `surface`, `no_surface` or `undetermined` |
| `note` | for an undetermined design: `torus_not_verified` (a torus was found but did not pass the rule below), `no_axis` |
| `iota`, `M`, `N` | a surface's rotational transform and the size of its Fourier series |
| `bn_solver` | a surface's largest \|B·n\|/\|B\| on a grid the solve never used, by the solver's own field |
| `bn_check` | the same quantity by the independent evaluator: its estimate of the largest value over the whole surface |
| `flux_err` | relative error of the enclosed flux, measured by the independent evaluator on six sections |
| `solve_s` | seconds the solve took (the solve alone, without the check, on machines that varied) |
| `check_s` | seconds the independent check took |
| `certificate` | for `no_surface`: `island_band` or `edge_barrier` |
| `excluded_lo_Wb`, `excluded_hi_Wb` | island band: no surface encloses a flux between these |
| `bound_Wb` | edge barrier: no surface encloses more than this |

`torux many` writes a table of your own designs in these columns: first `source`, the design as you named it, then the columns above, then `minor_radius_m` and `volume_m3` of the surface, then `asked`, the question the table answers (`flux`, `minor radius 0.12 m`, `volume 0.3 m^3` or `largest`). `id` is filled for a design asked for by its QUASR number. An outcome a QUASR run cannot have is written there too: `error`, with the reason in `note`.

**What counts.** A design is a `surface` only if all of this holds. The solver's own off-grid \|B·n\|/\|B\| is at most 10⁻¹⁴. The independent evaluator's estimate of the largest \|B·n\|/\|B\| over the surface is below 10⁻¹⁴ in double precision with its own rounding small, or else below 2 × 10⁻¹⁴ in double-double arithmetic. The enclosed flux is within 10⁻¹² of the request. It is `no_surface` only with a certificate whose every check passed. Everything else is `undetermined`. No design has both a verified surface and a certificate. The rule is `torux.verify.acceptance`, and [docs/verification.md](../docs/verification.md) says what each check is.

**An outcome is not a constant.** It depends on the time a design is given. The table keeps the best answer found for each design, and three in four of its undetermined designs got under a minute. One call with the default budget can come back `undetermined` for a design the table answers, and the other way round. On the 300 designs of the comparison below, each given an hour: the table has a surface for 194 of them, and 192 of those are surfaces again, while 9 designs the table leaves undetermined come back as surfaces and 38 as certificates. No design is a surface in one table and a certified no-surface in the other.

**Nor is it the same on every machine.** Near the edge of what a coil set holds, an outcome can depend on the clock, because the solver shares its budget among its stages by the time each takes. [`edge_requests.csv`](edge_requests.csv) holds two measured cases. Asked twice for the surface of QUASR 954 that encloses six times its design flux, torux verified it in 198 s and came back undetermined after 263 s in the next process. Asked for 0.0695 Wb of QUASR 1100073, a hair more than the design's own 0.06949574 Wb, it gave up after 1,914 s, where the design's own request gets its certificate in 116 s. A surface that is returned is verified either way.

## `compare300.csv`: the same 300 coil sets through every tool

300 QUASR designs drawn at random (numpy `default_rng(20261107)`). Every tool gets the same coil coefficients and the same requested flux, and at most an hour on four threads. Every surface a tool returns is then measured by one evaluator, [`evaluator/`](evaluator), which shares no code with any tool: the largest \|B·n\|/\|B\| at 4,000 random points in x87 extended precision (`bn_max`), and the enclosed flux against the request (`flux_rel_err`). `n_modes` is the number of Fourier terms the surface is written with. `wall_s` is the time from reading the input to the finished answer, the tool's own checks included, in a process that has already compiled. `cpu_model` and `threads` say on what. `solver_status` and `reason` are each run's own words as it wrote them: a `(C4)` there is a run's name for the 12 GiB rule. `certificate` is the kind of a torux certificate, `island_band` or `edge_barrier`. `answer_s` is the time at which the tool had the surface it returned: `wall_s` for torux, which stops there, and earlier for the others, which were set to spend their hour on accuracy and can go on after the level they return (read from each run's log and level times, and where those leave a second in doubt, the earlier time is taken).

How each tool was run, the file each was given and the commands that make a row again: [`compare300/`](compare300).

Rows: `torux`, `simsopt-boozer-exact`, `simsopt-boozer-ls`, `simsopt-qfm`, `desc`, `vmecpp`, `quasr-stored` (`quasr-stored` is the surface stored with each design, where every solve starts).

The evaluator's floor: an exact invariant torus, written in double precision, reads 3.9 × 10⁻¹⁶ ([`evaluator/test_output.txt`](evaluator/test_output.txt)). No surface stored in double precision can read much less.

```sh
python results/evaluator/judge.py examples/quasr_0000954.json   # solve with torux, then let the evaluator judge it
```

## `refinement.csv`

For each of torux's 201 surfaces in the comparison: its off-grid \|B·n\|/\|B\| at every resolution `M`, `N` of its refinement. The figure in the README's "How it works".
