# The method

For people who know the field. Everything here is in `src/torux`. The module named in each heading holds the detail in its docstring, and [code.md](code.md) is the map.

## The field (`torux.field`)

A source is a sum of parts: closed filaments as Fourier curves or polygons, straight segments, point dipoles, circular rings, axisymmetric conductors of polygonal cross-section, an analytic toroidal field.

- **Fourier filaments.** The N-point trapezoid rule on the closed curve, which converges exponentially for a periodic analytic integrand. N is chosen per coil from that coil's own distance to the region where the field is needed. The even nodes form the N/2 rule, and its difference from the N rule is an error estimate carried with every accepted surface.
- **Segments, dipoles, rings.** Closed forms written to avoid cancellation (the Hanson-Hirshman segment formula with a cancellation-free denominator, and ring fields by the arithmetic-geometric mean, with no division by R).
- **Sums.** Sources are added in blocks with Neumaier compensation, so the sum itself costs no digits.

B, its gradient and a vector potential A come from one pass.

## The axis and closed lines (`torux.axis`, `torux.orbits`)

A closed field line of period q is a Fourier curve solved by Newton with the exact Jacobian. Its monodromy comes from the variational equation along the converged line, so residues and on-axis transforms need no tracing.

## The surface (`torux.torus`, `torux.family`)

An invariant torus X(θ, φ) = (R, Z) in straight-field-line angles satisfies

(∂φ + ι ∂θ) X = V(X),  V = (R B_R / B_φ, R B_Z / B_φ).

The unknowns are the Fourier coefficients of R and Z and the transform ι. The equation is Galerkin-projected on a grid oversampled 2.5 times and closed by the enclosed toroidal flux, the loop integral of A. This is the parameterisation method for invariant tori (Haro and de la Llave), solved by:

- matrix-free Newton-Krylov with an approximate-reducibility preconditioner (two cohomological solves by FFT),
- behind it, for small systems, Newton with the exact Jacobian,
- next to a resonance, a Tikhonov-filtered step.

The resolution is not guessed: the refinement predicts the (M, N) it needs from the spectral decay of the current torus and jumps there. It stops when |B·n|/|B| on a half-shifted grid, points the solve never used, stops falling. A torus that levels off above 10⁻¹⁴ is not called a surface: nearly always a thin island chain sits at the request.

A source that comes with design surfaces (a QUASR download) starts from them. Any other starts from its magnetic axis and continues outward in the square root of the flux.

## The resonance guard (`torux.guard`)

At every near-resonant harmonic (m, n) the forcing the torus leaves there and the local shear give the Newton-Kantorovich quantity of that mode. The torus is accepted only at least two half-widths outside every chain.

## When there is no surface (`torux.islandband`, `torux.barrier`)

- **Island band.** For a stellarator-symmetric source the period map P in a symmetry plane is reversible: S P S = P⁻¹ with S(R, Z) = (R, −Z). A point where an unstable branch of a chain's X line crosses Z = 0 is therefore an exact heteroclinic point, however thin the chain. Closed curves built from unstable arcs to these points and their mirrored stable arcs bound every surface around the axis: none encloses a flux between the two curves' fluxes. The arcs' error tube and the quadrature error enter the interval.
- **Edge barrier.** R B_φ is harmonic in vacuum, so a line that reaches B_φ < 0 lies outside every surface. A hyperbolic closed line beyond the last surface whose manifold branches all reach such lost points lies on no surface. Its manifold arcs close a curve around the axis, and every surface lies inside it. The flux of that curve, plus its error terms, is an upper bound.

Both are certificates: a theorem plus an error budget measured in the same call and checked against the independent evaluator. Neither is a machine-checked proof: [verification.md](verification.md) says what that means in practice.

Sources that are not stellarators (a tokamak's own coils, perturbation coils, a levitated dipole) are settled by classical arguments in `torux.nonstell`: with no toroidal field there is no toroidal flux to enclose, in an axisymmetric vacuum the maximum principle for Δ*ψ leaves no interior extremum, and Ampère's law along a closed line excludes surfaces that link no current.

## Gradients (`torux.gradients`)

The adjoint of the torus system gives the derivative of ι, or of any function of the surface, in every source parameter (coil coefficients and currents, segment ends, dipole positions and moments, ring radii, the flux) for the cost of one more linear solve. It agrees with central differences to 10⁻⁵ or better (`tests/test_gradients.py`).

## What is known already

The parameterisation method is Haro and de la Llave's. Other ways to compute invariant tori from the trajectories themselves: Ruth and Bindel ([2023](https://arxiv.org/abs/2312.00967)), Ruth, Kulik and Burby ([2025](https://arxiv.org/abs/2505.08715)), and, with the parameterisation method, Blessing and Mireles James ([2023](https://arxiv.org/abs/2306.16597)). SIMSOPT's BoozerExact and BoozerLS surfaces (Giuliani and others, 2022) solve a related problem in Boozer coordinates. Converse KAM results on non-existence: Kallinikos, MacKay and Martinez-del-Rio ([2023](https://arxiv.org/abs/2304.09613)). What torux adds is the combination: rounding-level surfaces with a check that shares no code, certificates of absence, and a refusal to answer otherwise, in one call that runs unattended over a whole database.
