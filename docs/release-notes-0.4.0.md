torux 0.4.0

- A plasma's boundary and its current, with the coils, is a source torux solves. `torux.SurfaceCurrent` holds the boundary Γ as an R, Z Fourier series, the current on it as a potential or as the tangential field (K = n × B / μ₀ for a virtual casing), and the toroidal flux inside Γ. `torux.solve` returns the flux surface of the field outside Γ at a full toroidal flux, by size, or the largest it can verify, with the same acceptance as a vacuum surface and the independent evaluator's check of it, or `undetermined` with the reason. The family of surfaces starts next to Γ, from Γ itself, not at an axis. No surface comes within 10⁻⁵ of Γ's minor radius of Γ, none that does not go round Γ, and no closed field line of a chain is listed that comes as near; there are no certificates of absence for such a source. `torux.check` measures a surface against it. [Outside a plasma](surface-current.md).
- The sheet's field is new: the series summed exactly at every quadrature node, every separation formed relative to a panel corner kept in double-double. RESULT_LINE
- `torux.reference` evaluates the sheet with a quadrature of its own and measures the flux inside Γ itself.
- E2E_LINE
- The public VMEX Landreman-Paul example carries the flux inside its boundary, 0.025 Wb. VMEX_LINE

Known limits

- LIMITS
