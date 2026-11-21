# What is checked, and by what

torux says `surface` only when two field codes that share nothing agree, and `no_surface` only with a certificate. Both words are about the vacuum field of the source: no plasma is modelled, except as a current sheet on a plasma's boundary given with the coils ([below](#a-source-with-a-current-sheet)), outside which the field is a vacuum field too. This page is what stands behind each.

## A surface

Three measurements, each by different code.

1. **The solver's own check.** |B·n|/|B| on a grid shifted half a cell from the one the solve used, with the solver's field. At most 10⁻¹⁴, or the torus is not accepted. Also required, and checked by the solver alone: the resonance guard, a quadrature error estimate below 10⁻⁸, and that the torus's sections are simple closed curves around the axis.
2. **The independent evaluator** (`torux.reference`, used by `torux.verify`). Written apart from the solver's field code and sharing none of it: Gauss-Legendre panels where the solver uses the trapezoid rule, Carlson elliptic integrals where it uses the arithmetic-geometric mean, compensated sums throughout, and a double-double tier of about 32 digits. `tests/test_independence.py` checks that it imports nothing of the solver. What the two share is the loader that read the coils, and the surface. On every surface it measures:
   - |B·n|/|B| at 4,096 random points, 64 values of θ by 64 of φ, with the surface's points and normals in long double, and from the largest of them a local maximisation: an estimate of the largest value over the whole surface (where long double is no wider than double, as on Apple silicon, this reading carries more of the evaluator's own rounding, and the acceptance below is the same everywhere),
   - its own rounding, by repeating the eight largest and eight random points in double-double,
   - the enclosed flux on six random sections.
3. **The acceptance** (`torux.verify.acceptance`). The estimated maximum must be below 10⁻¹⁴ with the evaluator's own rounding small against the margin. Where rounding is not small (next to a coil, double precision alone reads up to 2 × 10⁻¹³ on a surface that is good to 10⁻¹⁵), the double-double values decide and must be below 2 × 10⁻¹⁴. The flux must be within 10⁻¹².

This is a measurement, not a proof that an invariant torus lies beside the computed one.

Across the 231,878 QUASR surfaces: the solver's check reads 1.7 × 10⁻¹⁵ in the median and at most 1.0 × 10⁻¹⁴, the evaluator's estimated maximum 1.4 × 10⁻¹⁵ in the median and at most 1.4 × 10⁻¹⁴, and the flux is off by 1.6 × 10⁻¹⁵ in the median ([`results/quasr.parquet`](../results/quasr.parquet)).

**A third program.** The 300-coil comparison is judged by [`results/evaluator`](../results/evaluator), which shares no code with torux or with its independent evaluator and works in x87 extended precision. On torux's 201 surfaces there it reads 1.1 × 10⁻¹⁵ in the median and 6.8 × 10⁻¹⁵ at worst. Try it on the bundled example: `python results/evaluator/judge.py examples/quasr_0000954.json` solves it with torux and hands the surface to this evaluator.

## No surface

A certificate is a theorem whose hypotheses are checked numerically, with error estimates that are part of the record ([the method](../method/README.md#4-where-there-is-no-surface-a-certificate) states the two theorems, and the docstrings of `torux.islandband` and `torux.barrier` hold their proofs). The theorems are about every invariant torus around the magnetic axis with no coil inside it and B_φ of one sign on it. The arithmetic is floating point, so no certificate is a computer-assisted proof. What is and is not rigorous:

- **Island bands** (5,206 of QUASR's certificates), by two constructions that the table does not tell apart. In the first the theorem is exact: reversibility puts a heteroclinic point on the symmetry plane. The curves through it are grown by a tracer, or by an interpolant of the period map fitted to traces, whose error estimate is measured in the same call and compared with the independent tracer at sampled steps: checked, not proved to enclose the error. In a chain of several X lines a surface can also be a curve made of arcs of both, and the fluxes of those are taken out of the band. The theorem does not cover a chain of one island, a reflection-hyperbolic X line, a source without stellarator symmetry, or a chain whose X lines all have residues under 2.5 × 10⁻¹⁰. The second is the fallback, and the one used after a continuation from the axis: a surface on each side of the chain and a torus inside one island leave no room for the flux asked for, if all three are invariant tori. That is measured, to 10⁻¹² of |B| for the two surfaces and 10⁻⁹ for the third, and not proved. The 22 island bands of the comparison are all of the first.
- **Edge barriers** (846). Some of their error terms are bounds given their inputs and some are estimates checked against a second computation (`torux.barrier` lists which). This is the weakest class: if a certificate is ever wrong, look here first.

On the 300 designs of the comparison, none is a surface in one table and a certificate in the other.

## Undetermined

Everything else. A torus that does not pass the acceptance, a certificate with one failed check, a solve that ran out of time: none of these is an answer, and torux does not present one as an answer. Nor is it a verdict that no answer exists: most of QUASR's undetermined designs were given seconds, and an hour settles more ([`results/compare300.csv`](../results/compare300.csv)).

## A source with a current sheet

A source with a `SurfaceCurrent` ([outside a plasma](surface-current.md)) is held to the same three measurements and the same acceptance, about the field outside the sheet's boundary Γ: coils and sheet. The independent evaluator takes the sheet with a quadrature of its own (graded rules about the boundary's nearest points, positions near the target formed as offsets from the nearest one, `torux.reference.sheet` and `sheet_dd`), sharing no code with the solver's panels (`tests/test_independence.py` checks that no run of three significant lines repeats between them), and measures the flux inside Γ itself, from its own field and its own geometry of Γ. Inside Γ the source's field is not the plasma's, so nothing there is used: the solver accepts no torus that comes within 10⁻⁵ of Γ's minor radius of Γ, or whose sections do not go once round Γ's, or where the sheet's field moves by more than 2 × 10⁻¹⁵ of |B| under a refined quadrature. There are no certificates for such a source: their theorems are about tori with no current inside, and a plasma's boundary has current inside.

Held to exact cases, in which the field outside Γ is known without torux:

- QUASR 954's coils, Γ one of their verified surfaces (0.01 Wb) and on it the sheet that cancels their field outside it: `torux.solve` at 0.03 and 0.05 Wb returns the coils' own verified surfaces there to 6.7 and 7.6 × 10⁻¹⁵ m, their rotational transform to 1.5 and 1.8 × 10⁻¹⁶; the solver's check reads 1.0 and 1.3 × 10⁻¹⁵, the independent evaluator 1.0 and 1.1 × 10⁻¹⁵, the flux agrees to 8 × 10⁻¹⁵. The sheet written as a potential instead gives the 0.03 Wb surface to 6.9 × 10⁻¹⁵ m (`tests/test_sheet_truth.py`, slow).
RINGRESULT

Next to Γ the solver's field is within 4.6 × 10⁻¹⁶ of |B| of the independent evaluator's double-double tier down to 10⁻⁵ of Γ's minor radius ([how close to Γ](surface-current.md#how-close-to-γ)).

What the measurements say is about the field of the coils and the sheet as given. Whether the sheet is the plasma's current, and Γ a flux surface of the field outside it, is the equilibrium's: every answer reports how far the field crosses Γ (`evidence['sheet']['boundary_bn_max']`), and the flux of the source's own field through Γ's sections, zero for an exact virtual-casing current.

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
