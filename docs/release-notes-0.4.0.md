# Torux 0.4.0

- `SurfaceCurrent` adds the exterior field of a supplied Fourier current sheet to the coils. It accepts a periodic current potential with secular terms or the covariant tangential boundary field. It supports field evaluation, vector potential, Cartesian field derivatives and explicitly seeded Poincaré sections.
- Exterior sections report why a line stopped, including the boundary cutoff or loss of toroidal direction. The Fourier sheet uses adaptive Biot-Savart quadrature with a 32 MiB panel cache per sheet.
- The public VMEX Landreman-Paul example includes the coils, sheet and reference vectors. The replay recovers the outboard `iota = 1/2` elliptic orbit at about 4.98 mm outside the supplied boundary in the 4 kA case. A separate script compares your plasma-only reference vectors under quadrature refinement.

The equilibrium and sheet are inputs. Torux's flux-surface solver and independent flux-surface verifier support vacuum sources.

[Try the VMEX case and compare your own probes](https://github.com/philippbogdan/torux/blob/v0.4.0/docs/surface-current.md).
