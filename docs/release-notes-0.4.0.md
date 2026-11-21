torux 0.4.0

- Outside a finite-beta plasma. `torux.SurfaceCurrent` holds the plasma's boundary Γ, the current on it (as a potential, or as the tangential field, K = n × B / μ₀, a virtual casing) and the toroidal flux inside Γ. With the coils it is a source like any other: `torux.solve` returns the flux surface of the field outside Γ at a full toroidal flux, by size, or the largest it can verify, accepted and independently verified as a vacuum surface is, or `undetermined` with the reason. The family of surfaces starts next to Γ, not at an axis, and nothing that comes within 10⁻⁵ of Γ's minor radius of Γ, or passes inside it, counts as evidence. `torux.check` measures any surface against such a source. [Outside a plasma](surface-current.md).
- Exact. QUASR 954's coils with a sheet that cancels their field outside a truth surface: `torux.solve` returns the coils' own verified surfaces at 0.03 and 0.05 Wb to 7.6 × 10⁻¹⁵ m, their rotational transform to 2 × 10⁻¹⁶. A current ring inside a toroidal field, replaced by its sheet: the closed-form flux surface to 1.0 × 10⁻¹⁴ m.
- The sheet's field holds rounding next to Γ: within 4.6 × 10⁻¹⁶ of |B| of the independent evaluator's double-double tier, from 10⁻¹ down to 10⁻⁵ of Γ's minor radius. `torux.reference` evaluates the sheet with a quadrature of its own and measures the flux inside Γ itself.
- The public VMEX example carries the flux inside its boundary, 0.025 Wb. Its boundaries are 4.8 × 10⁻³ and 4.0 × 10⁻³ of |B| from flux surfaces of their coils and sheets, so a request next to them is `undetermined` with that reason; at 0.045 Wb the 4 kA case gives a verified surface in 27 minutes.
- Vacuum sources: unchanged, the same answers bit for bit.

Known limits

- There are no certificates that a surface cannot exist outside Γ: their theorems are about tori with no current inside.
- A boundary that is not a flux surface of its coils and sheet has no surfaces next to it, and torux says so with the measure of how far it is. The bundled VMEX boundaries are 4.8 × 10⁻³ and 4.0 × 10⁻³ of |B| from one.
- Sheets of many modes are slow: the independent check of the 3,870-mode QUASR case takes about an hour, the first start torus of a 19,333-mode sheet had not converged after 2.6 hours, and on sheets of 46,491 and 108,608 modes the first measurement, the flux through Γ, takes 101 minutes and more than 3 hours.
- `gradients=True`, `answer.vjp`, `trace_lines`, `widths_s`, `answer.render` and `torux render` refuse such a source, each in one sentence.
