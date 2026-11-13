# What is checked, and by what

torux says `surface` only when two programs that share no code agree, and `no_surface` only with a certificate. Both words are about the vacuum field of the source: no plasma is modelled. This page is what stands behind each.

## A surface

Three measurements, each by different code.

1. **The solver's own check.** |B·n|/|B| on a grid shifted half a cell from the one the solve used, with the solver's field. At most 10⁻¹⁴, or the torus is not accepted. Also required: the resonance guard, a quadrature error estimate below 10⁻⁸, and that the torus's sections are simple closed curves around the axis.
2. **The independent evaluator** (`torux.reference`, used by `torux.verify`). Written apart from the solver's field code and sharing none of it: Gauss-Legendre panels where the solver uses the trapezoid rule, Carlson elliptic integrals where it uses the arithmetic-geometric mean, compensated sums throughout, and a double-double tier of about 32 digits. `tests/test_independence.py` checks that it imports nothing of the solver. On every surface it measures:
   - |B·n|/|B| at 4,096 random points, with the surface's points and normals in long double, and from the largest of them a local maximisation: an estimate of the largest value over the whole surface (where long double is no wider than double, as on Apple silicon, this reading carries more of the evaluator's own rounding, and the acceptance below is the same everywhere),
   - its own rounding, by repeating the eight largest and eight random points in double-double,
   - the enclosed flux on six random sections.
3. **The acceptance** (`torux.verify.acceptance`). The estimated maximum must be below 10⁻¹⁴ with the evaluator's own rounding small against the margin. Where rounding is not small (next to a coil, double precision alone reads up to 2 × 10⁻¹³ on a surface that is good to 10⁻¹⁵), the double-double values decide and must be below 2 × 10⁻¹⁴. The flux must be within 10⁻¹².

Across the 231,878 QUASR surfaces: the solver's check reads 1.7 × 10⁻¹⁵ in the median and at most 1.0 × 10⁻¹⁴, the evaluator's estimated maximum 1.4 × 10⁻¹⁵ in the median and at most 1.4 × 10⁻¹⁴, and the flux is off by 1.6 × 10⁻¹⁵ in the median ([`results/quasr.parquet`](../results/quasr.parquet)).

**A third program.** The 300-coil comparison is judged by [`results/evaluator`](../results/evaluator), which shares no code with torux or with its independent evaluator and works in x87 extended precision. On torux's 201 surfaces there it reads 1.1 × 10⁻¹⁵ in the median and 6.8 × 10⁻¹⁵ at worst. Try it on the bundled example: `python results/evaluator/judge.py examples/quasr_0000954.json` solves it with torux and hands the surface to this evaluator.

## No surface

A certificate is a theorem whose hypotheses are established numerically, with an error budget that is part of the record ([the method](../method/README.md#4-where-there-is-no-surface-a-proof) states the two theorems). What is and is not rigorous:

- **Island bands** (5,206 of QUASR's certificates). The theorem is exact: reversibility puts heteroclinic points on the symmetry plane. The curves are computed by a tracer whose error model is measured in the same call and checked against the independent tracer. The numerics are validated, not interval arithmetic.
- **Edge barriers** (846). Some of their error terms are strict bounds and some are validated models (`torux.barrier` lists which). This is the weakest class: if a certificate is ever wrong, look here first.

Every certificate is checked for consistency with the surfaces: no QUASR design carries both a verified surface and a certificate.

## Undetermined

Everything else. A torus that does not pass the acceptance, a certificate with one failed check, a solve that ran out of time: none of these is an answer, and torux does not present one as an answer. Nor is it a verdict that no answer exists: most of QUASR's undetermined designs were given seconds, and an hour settles more ([`results/compare300.csv`](../results/compare300.csv)).

## What the interface adds

`torux.solve` hands a request for a flux to the solver unchanged, and a surface asked for by its size, or as the largest, is the solver's answer to a request at its own flux: nothing is scaled or interpolated (`tests/test_answer.py`, `tests/test_ask.py`). Everything else the interface computes is held to code that shares none with it. The tests that need SIMSOPT, DESC or VMEC++ skip where those are not installed.

| what | held to | agreement |
|---|---|---|
| volume, area, radii, aspect ratio | SIMSOPT 1.11.1, reading the saved VMEC file | 5 × 10⁻¹⁵, relative |
| the saved VMEC input | SIMSOPT, VMEC++ 0.9.3 and DESC 0.17.3 reading it back | points to 6 × 10⁻¹⁵ m, 1 × 10⁻¹⁵ m and 9 × 10⁻¹⁰ m (DESC's converter keeps eight decimals) |
| the same file, cut to MPOL 9, NTOR 8 | a VMEC++ run inside it | volume to 2 × 10⁻⁹, edge rotational transform -0.09757 against -0.09752 |
| `torux.check` | the comparison's evaluator, on the surfaces QUASR stores for 24 of its 300 designs | the largest leak to 5 × 10⁻¹⁵ of \|B\| |
| the Boozer spectrum and quasi-symmetry | SIMSOPT's `BoozerSurface` and `NonQuasiSymmetricRatio`, four designs | every harmonic within SIMSOPT's own residual, 3 × 10⁻¹³ to 1 × 10⁻⁹ of the mean field |
| coils given as SIMSOPT or DESC objects | each code's own Biot-Savart field | 4 × 10⁻¹⁵ and 2 × 10⁻¹⁵ of \|B\|, DESC's value of μ₀ accounted for |
| `torux.many` | single solves of the same designs | bit for bit |
