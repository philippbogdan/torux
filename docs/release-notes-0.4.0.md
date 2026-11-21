torux 0.4.0

- Outside a finite-beta plasma. `torux.SurfaceCurrent` holds the plasma's boundary Γ, the current on it (as a potential, or as the tangential field, K = n × B / μ₀, a virtual casing) and the toroidal flux inside Γ. With the coils it is a source like any other: `torux.solve` returns the flux surface of the field outside Γ at a full toroidal flux, by size, or the largest it can verify, accepted and independently verified as a vacuum surface is, or `undetermined` with the reason. The family of surfaces starts next to Γ, not at an axis, and nothing that comes within 10⁻⁵ of Γ's minor radius of Γ, or passes inside it, counts as evidence. `torux.check` measures any surface against such a source. [Outside a plasma](surface-current.md).
- Exact. QUASR 954's coils with a sheet that cancels their field outside a truth surface: `torux.solve` returns the coils' own verified surfaces at 0.03 and 0.05 Wb to 7.6 × 10⁻¹⁵ m, their rotational transform to 2 × 10⁻¹⁶. RING_LINE
- The sheet's field holds rounding next to Γ: within 4.6 × 10⁻¹⁶ of |B| of the independent evaluator's double-double tier, from 10⁻¹ down to 10⁻⁵ of Γ's minor radius. `torux.reference` evaluates the sheet with a quadrature of its own and measures the flux inside Γ itself.
- VMEX_LINE
- Vacuum sources: unchanged, the same answers bit for bit.

Known limits

- LIMITS
