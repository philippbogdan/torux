"""Upper bounds on the flux of every flux surface from hyperbolic closed field lines whose invariant
manifolds reach the lost region ("edge barrier"), and flux bands free of flux surfaces inside wide
chaotic layers ("manifold band").

Setting and the class of tori covered
-------------------------------------
Sigma is the half plane phi = 0 (a stellarator-symmetry plane), P the period map (one field period
2 pi / nfp along the field lines), defined at every point whose line can be followed for one period
with B_phi of the axis' sign.  s = sign of B_phi on the magnetic axis; "B_phi" below means s B_phi.
a is the magnetic axis' point in Sigma (the elliptic period-one line the request's surfaces enclose).

COVERED TORI.  Gamma is an invariant torus (a closed surface made of field lines) with
  (T1) its solid torus V contains no source (coil, magnet) -- the vacuum region of a flux surface;
  (T2) the field is transverse to the planes phi = const on Gamma: B_phi > 0 on Gamma (this is what
       makes Gamma a flux surface in the solver's sense: it is parameterised over phi by the line
       equations dR/dphi = R B_R / B_phi; every surface torux.torus returns satisfies it);
  (T3) V contains the magnetic axis and Gamma meets Sigma in one Jordan curve gamma around a; D is
       the interior of gamma, flux(Gamma) = int_D B_phi dR dZ.
Every flux surface a request can be answered with is such a torus, so a bound over covered tori is
a bound over the answers.  (edge_barrier refuses a T_last whose section at phi = 0 is not a simple
closed curve, and one whose interior does not contain the elliptic period-one line it finds.)  What is NOT covered: tori containing coils, tori on which B_phi changes
sign, tori not enclosing the axis.

LEMMA A (maximum principle).  R B_phi = x B_y - y B_x is harmonic where curl B = 0, div B = 0
(Delta(x B_y - y B_x) = x Delta B_y - y Delta B_x + 2 (d_x B_y - d_y B_x) = 2 (curl B)_z = 0).  By (T1)
it attains its minimum over the closed solid torus on Gamma, so by (T2) B_phi > 0 on all of V.
Hence P is defined on D-bar, maps gamma onto gamma and D onto D (P(D) is connected, contains a,
misses gamma; same for P^-1), and every line through V stays in V with B_phi > 0.

LOST POINTS.  A point z of Sigma is *lost* if its field line (in either direction) reaches a point
where B_phi < 0.  By Lemma A no point of V-bar is lost: lost points lie outside D-bar for every
covered torus.  (Tracing in phi stops where |B_phi| < 1e-3 |B|; the proof continues the line in arc
length through the turning point until B_phi <= -beta |B|, see below.)

Let X = {X_0..X_{q-1}} be a hyperbolic periodic orbit of P (a closed field line of q field periods;
residue < 0 or > 1), W^u, W^s its invariant manifolds.  At each X_k there are four branches (half
manifolds); all branches are images under P^j of the four branches at X_0, the "generating curves"
G = {u+, u-, s+, s-} (two, u and s, for a reflection-hyperbolic orbit, whose branches swap under P^q).
A branch is EXCLUDED if (i) it contains a lost point, or (ii) a certified crossing box joins it to a
branch of the other kind: a parallelogram Q in which a true sub-arc of one joins the left and right
sides inside the horizontal strip of Q and a true sub-arc of the other joins bottom and top inside
the vertical strip (the curves then meet in Q: parallelogram lemma).  Exclusion is a property of the
generating curve (P-images of a lost point are on its field line; P maps such a box configuration
to another one).

LEMMA B.  If at most one generating curve is not excluded (none, for a reflection-hyperbolic orbit),
no covered torus meets X.  Proof: let X_k lie on gamma; then all of X does (invariance).  P^q (P^2q
if reflection-hyperbolic) restricted to gamma is an orientation-preserving circle homeomorphism
with isolated fixed points X_j, so every orbit on gamma converges monotonically along gamma to some
X_j in forward and backward time.  Hence each of the two arcs of gamma at X_k is near X_k a local
half-branch, and by invariance gamma contains two whole branches b1 != b2 at X_k.  Neither is
excluded: (i) gamma lies in V-bar, so b contains no lost point; (ii) let a box join b to a branch c
of the other kind (base point X_j).  c meets b at some h (parallelogram lemma), so c meets gamma;
the iterates of h that converge to X_j along c (forward if c is stable) converge to X_j along gamma
too, hence eventually lie on a local half-arc of gamma at X_j, which is therefore the local part of
c, and by invariance c lies in gamma.  b and c are open arcs of the Jordan curve gamma with the
common point h, so they share an open arc; stable and unstable manifolds of the real-analytic
period map (analytic vacuum field) are analytic curves, so b = c as sets.  Then the two box arcs are
sub-arcs of one injective curve; the end points of each lie outside the strip containing the other,
so neither parameter interval contains an end point of the other: the intervals, hence the arcs,
are disjoint, contradicting the parallelogram lemma.  So gamma would contain branches of two distinct
non-excluded generating curves (for a reflection-hyperbolic orbit both branches of one kind belong
to one generating curve: then none may be non-excluded).

LEMMA C.  If no covered torus meets X, then no branch meets any covered torus (a point of a branch
on gamma would have its orbit converge to X inside the closed invariant gamma), all X_k lie on one
side of every covered gamma (P permutes X and preserves D), and every branch, with its base point,
lies on that side (a connected set missing gamma).

THEOREM (edge barrier).  Suppose (a) Lemma B's hypothesis holds; (b) some generating curve has a
lost point; (c) there are certified crossing boxes (points h_k in W^u(X_k) and W^s(X_k')) for
consecutive points X_k, X_k' of X in the angular order around a, so that the arcs X_k -> h_k (on W^u)
and h_k -> X_k' (on W^s) form a closed curve C; (d) C's computed polyline C_c is
simple, contains a in its interior at distance > eps from C_c, where every point of the true C lies
within eps of C_c (and conversely, along matching parameterisations); (e) B_phi >= 0 on int(C_c).
Then every covered torus has flux(Gamma) <= Phi(C_c) + Delta, Delta = int over the eps-tube of C_c of
B_phi^+, bounded by sum over segments (2 eps l + pi eps^2) max B_phi^+.
Proof: by (b) and Lemma C every X_k and every manifold arc lies outside D-bar, so D (connected,
containing a) lies in the component of Sigma minus C containing a.  For x outside the eps-tube the
straight-line homotopy between C_c and C misses x, so wind(C, x) = wind(C_c, x); hence the a-component
(winding number wind(C, a) = wind(C_c, a) = 1) is contained in int(C_c) union tube, and
flux(Gamma) = int_D B_phi <= int_{int C_c} B_phi + Delta = Phi(C_c) + Delta (Lemma A: B_phi > 0 on D;
(e)).  Phi(C_c) is the loop integral of A along C_c (Stokes), with its quadrature error added.

THEOREM (manifold band).  With (a), (c), (d), (e) for two closed curves C_1, C_2 of manifold arcs of
the same orbit (the two classes: W^u(X_k) toward X_k+1 with W^s(X_k+1), and W^u(X_k+1) toward X_k
with W^s(X_k)), every covered torus has flux <= min Phi(C_i) + Delta_i (X outside D) or
flux >= max Phi(C_i) - Delta_i (X inside D: D is a disk containing the arcs, hence int(C_i), and
int(C_i) contains int(C_c,i) minus the tube).  No covered torus has its flux in between.

THEOREM (lost ring).  Let D_1..D_n be closed disks in Sigma every point of which is lost, and Q a
closed polygon each of whose edges lies inside the union of two of the disks (so Q lies in the union
U), Q simple with winding number 1 around a, B_phi >= 0 inside Q.  Then every covered torus has
flux <= Phi(Q): D-bar contains no lost point (Lemma A), so D misses U and Q; D is connected and
contains a, so D lies inside Q.  No orbit and no manifold enter.  edge_barrier tries it first when
the escape-time map shows a loop of quickly lost cells around a whose flux estimate is below the
request; otherwise (or when it fails) the manifold barrier.

THEOREM (edge barrier, reversible closing).  Assume the source is stellarator symmetric (exactly: QUASR
coil sets are generated from their base coils by the symmetry group; the evaluated field's defect is measured
in every run, ~1e-15) and Sigma is a symmetry plane (phi = 0 or pi / nfp), so P is reversible: S P S = P^-1,
S(R, Z) = (R, -Z), and S W^u(Y) = W^s(S Y) for every periodic point Y.  Let X be a regular hyperbolic
symmetric orbit (S X = X), c an unstable generating curve with branches c(X_k) = P^k c(X_0) and Sc(X_k) :=
S c(S X_k) (stable branches; P Sc(X_k) = Sc(X_k+1) since P S = S P^-1).  Suppose
 (R1) h is a point of c(X_k) on the line Z = 0, where X_k and S X_k are neighbours along the chain (a
      certified crossing of the computed branch with the exactly known line: parallelogram lemma) -- then
      h = S h lies on Sc(S X_k): an exact heteroclinic point, however small the splitting angle;
 (R2) C_c is the closed curve made, for every j in a window of q consecutive integers, of the arc of
      c(X_k+j) from X_k+j to P^j h and the arc of Sc(S X_k+j) from P^j h to S X_k+j (P^j h is on both), with
      a simple computed polyline C_c,c within eps of it, winding once around a outside the tube, and B_phi >= 0
      inside (the conditions (d), (e) of the edge barrier).
Then (i) a covered torus Gamma that misses X lies inside C_c when some branch of X has a certified lost
point (Lemma C: X and every branch lie outside D-bar, so D lies in the a-component of the complement of
C_c), and (ii) a covered torus through X that contains a branch of c or of Sc at some point of X IS C_c:
the branch runs along gamma monotonically to the next fixed point of P^q on gamma; the heteroclinic point it
contains converges (forward on Sc, backward on c) to that branch's partner, so gamma contains the arcs of
C_c of that gap, by invariance all of C_c, and a closed curve winding once around a inside the Jordan curve
gamma is gamma.  So flux(Gamma) <= Phi(C_c,c) + Delta in both cases.
GENERALISED LEMMA B.  Call c (with Sc) a closed family when (R1), (R2) hold.  A covered torus through X
contains branches of two distinct non-excluded generating curves at X_0 (Lemma B's proof); if one of them
belongs to a closed family c, the torus is C_c.  In a closed family the exclusion of one member excludes
the other (the torus would be C_c, which contains branches of both).  Hence: if at most one non-excluded
generating curve lies outside the closed families (none for a reflection-hyperbolic orbit), and some branch
has a certified lost point, every covered torus has flux <= max(min over certified curves (box or
reversible) of Phi + Delta, max over closed families with a non-excluded member of Phi + Delta)
(lemma_B_general; with no closed family this is Lemma B).  Typical use: the inner separatrix family closed
by (R1)-(R2), a lost point on the outer family.  An orbit with no mirror pair of chain neighbours in phi =
0 (q even, two points on Z = 0) has such pairs in the plane phi = pi / nfp: it is carried there by half a
period and certified in that plane (candidate_in_other_plane).
The curves of the reversible closing come from the same grown and chord-validated manifolds as the box
curves; the heteroclinic images P^j h (j = -(q div 2) .. q - (q div 2) - 1, forward and backward maps)
carry 2D error bounds e_j = |DP^j| box / 2 + 2 sum |DP^j DP^(i)^-1| tau_i (finite-difference Jacobians,
first order; the orbit's single steps are checked against the reference tracer in every run), each gap's
arcs are cut at the computed branches' points nearest P^j h (refused when farther than 4 x (branch error +
e_j)), the junction errors added to the adjacent segments; the polygon is decimated with the dropped
vertices' distances added to the errors (torux/islandband.py's construction).

What is strict and what is validated
------------------------------------
STRICT (mathematics, given the numbers it is fed): Lemmas A-C and both theorems for the covered tori;
the parallelogram lemma (true curves within eps of the polylines cross in the box); the homotopy /
winding-number argument for the eps-tube; the tube and quadrature terms added to the flux; the
loop-cut construction; the disk argument of the lost-point proof (every line of the disk reaches
B_phi < 0 if f < 0 on the region covering the disk's image); for the lost ring, the coverage of every
edge of Q by two disks (exact segment/disk computation with the radii reduced by 1e-12 scale; loop
cuts of Q keep sub-segments of covered edges), the winding number and simplicity of Q, and the flux
bound Phi(Q) + quadrature error + B_phi^+ over the 1e-12-scale strip.  Phase 4: the reversible closing and the
generalised Lemma B (given the exact symmetry, an assumption); the crossing of a branch with the exactly known line
Z = 0 (parallelogram lemma); the transfer of a lost proof to the other symmetry plane (the same field lines: the
half-period map carries the branch, the lost line through its point is the same line).
VALIDATED (error models with stated assumptions, each checked against an independent computation in
every run, but not proven to enclose the error):
  * tracer error per period tau = 2 x GBS estimate + 2e-13 R (reference tracer on sampled vertices); for lines
    re-traced next to coils tau = 2 x max(GBS estimate, |y_2f - y_f|) + 2e-13 R (step doubling; reference tracer
    on sampled re-traced steps in every run);
  * first-order propagation of the normal error along the manifolds (linearisation at scale eps);
  * chord errors from the turning-angle model (true midpoints computed for every segment used);
  * start error delta0 from the measured invariance defect (assumes the defect decreases at least
    linearly towards X; factor 2 for the parameters between the 6 samples);
  * e_X = 2 ||(M - I)^-1|| (closure + tau_q) with M from central differences (reference closure);
  * lost-point disk: image = fitted ellipse (fit residual = nonlinearity), Lipschitz constant of
    B_phi/|B| = 2 x the largest sampled gradient; position error = max(integrator estimate, distance to
    the independent DOP853 re-trace on the reference field); for the lost-ring disks the position
    error is 2 x the summed GBS estimates + the reference floor: the largest distance between the
    engine trace and re-traces (GBS k = 6 on the reference field; DOP853 on one) of n_ref = 4 sampled
    centre lines (the three with the largest arc length x stretch and one at random) -- the other
    disks' lines are not re-traced;
  * B_phi > 0 inside the curve: grid minimum less 2 x the largest sampled |grad B| times the spacing;
  * the heteroclinic images' 2D errors (reversible closing; first order, reference single steps);
  * the exact stellarator symmetry of the source (reversible closing and mirrored stable curves; the field's
    measured defect, ~1e-15, is part of every run's evidence);
  * the field itself (engine quadrature, reference loop integral of A and reference closure agree);
  * floating-point rounding is not interval-bounded (all quantities are ~1e-15 relative or smaller
    than the margins by many orders).

Numerics (every error estimated a posteriori and added to eps; which estimates are strict: above)
------------------------------------------------------------------------------------------------
* Orbit: symmetric closed lines from a 1D shooting on the symmetry lines Z = 0 of phi = 0 and
  phi = pi / nfp (a line through two symmetry planes closes after q field periods); X_0 by
  Illinois regula falsi on a coarse map, then 2D Newton on the accurate P^q(x) = x; monodromy M by
  central differences; the true periodic point lies within
  e_X = 2 ||(M - I)^-1|| (|P^q X_0 - X_0| + tau_q) (Newton-Kantorovich, hyperbolic M).
* Manifolds: polylines grown from the fundamental domain of the quadratic local manifold
  p(s) = x* + s v + s^2 w, (Lambda^2 I - DF) w = D^2F[v, v] / 2 (F = P^q, P^2q if reflection-hyperbolic;
  second derivative by central differences), s in [s0, Lambda s0], one period map at a time (all q
  points' branches at once), refined where image spacing > dmax or turning angle > amax by inserting
  pre-images (cubic chord-length interpolation of the previous piece; its error = |cubic -
  quadratic| is carried).  Each vertex carries a bound on its distance to the true manifold (normal
  error), propagated to first order: e' = beta e + tau, beta = det(DP) / stretch along the curve,
  det = B_phi(z) / B_phi(P z) (flux conservation), stretch from the neighbouring images, tau = 2 x the
  tracer's GBS estimate + 2e-13 R;
  plus amp * (delta0 + e_X), amp the product of the betas.  delta0, the start error, is measured: the
  invariance defect d = dist(F(p(s)), p) at 6 parameters per side of the domain; normal deviations
  are multiplied by mu per application of F, so the domain lies within sum_k mu^(k-1) d(s
  Lambda^-k) <= max d / (Lambda - mu) of the manifold when d decreases at least linearly with s (it is
  O(s^3)); delta0 = 2 max d / (Lambda - mu) (the factor 2 covers the parameters between samples).
  For a stellarator-symmetric field and orbit W^s is the mirror image S W^u (P^-1 = S P S,
  S(R,Z) = (R,-Z)); the measured symmetry defect of the reference field and the orbit's mirror
  mismatch are added.
* Chord errors: on the arcs used, every segment's true midpoint (image of the pre-image midpoint) is
  computed and inserted until the measured deviations agree with the turning-angle model and the
  bound d theta_max / 4 is below the target; that bound (factor 2 over the sagitta d theta / 8) is
  the segment's chord error.  With a request, the target is set so the tube term stays below 2 % of
  the margin (between 2e-6 and 2e-5 scale); a curve whose uncertified flux already exceeds the request
  is not certified (it could not hold it).
* Crossings: the parallelogram lemma (a curve joining the left and right sides of a parallelogram
  and a curve joining its bottom and top intersect inside it): a box Q around the polyline crossing,
  sides parallel to the two segments, such that the computed W^u arc stays eps_u / sin(theta) inside
  the strip |y| < b and ends eps_u / sin(theta) beyond x = -a and x = +a, and likewise for W^s; then
  the true branches cross in Q, and diam(Q) is added to the error of the arcs next to h.
* Closed curve: X_k -> (W^u arc) -> h_k -> (W^s arc) -> X_k+1 for all gaps; if the polygon is not
  simple, loops are cut at its self-crossings, each certified by the same lemma (the true arcs meet
  in the box, so the shortcut curve still consists of true manifold arcs); the piece with winding
  number 1 around a is kept and must be simple.
* Lost points: candidates come from single orbits shot along each generating curve from its
  fundamental domain, detected with a coarse map (k = 4, 4 steps per period: detection only) and
  repeated from the same start with the accurate map and error model (normal error propagated with a
  renormalised tangent companion); the mirror image S z of a lost point of W^u, a point of W^s, gets
  its own proof (backward lines) when Lemma B needs it.  The proof for
  a candidate z (normal error e): the centre and 16 points on the circle of radius r = max(2 e, 1e-6
  scale) are followed in arc length from the section (adaptive GBS, through the turning point of phi)
  to the centre's first point with B_phi <= -beta |B| (beta = 0.05); the linear map fitted to the
  circle's images (residual = nonlinearity) gives the image ellipse of the disk; f = B_phi/|B| is
  sampled on it and max f + L (h + 2 nonlinearity + 2 dx) < 0 (L = 2 x the largest sampled |grad f|,
  h the sample spacing, dx the larger of the integrator's estimate and the distance to an
  independent re-trace: scipy DOP853 on the reference field) shows that every line of the disk, hence
  the true manifold point in it, reaches B_phi < 0.
* Flux: loop integral of A with Gauss-Legendre 3 and 5 points per edge (difference = quadrature
  error); B_phi >= 0 on int(C_c) by a grid with spacing h: min B_phi - |grad B_phi|_max h / sqrt 2 > 0.
* Lost ring: escape times (coarse map, forward and backward, up to ring_nper = 6 periods; detection
  only) on a polar grid around a (96 x 32) between T_last and 0.7 x the coil distance (cells within
  0.03 scale of coil crossings left out); the loop: Dijkstra on the unwrapped grid through cells lost
  within T periods (smallest T with a loop whose flux estimate -- B_phi integrated over the wedges from
  a, for the choice only -- is <= 0.85 x request), cost = length x (1 + (esc - 1) + 4 x fraction of
  unusable neighbours) + lam x wedge flux (lam raised until the estimate fits).  Along the loop,
  disks of radius 0.6 h at spacing h (the cell spacing).  Each centre line is traced in arc length
  (GBS k = 4, tol 1e-9) forward, else backward, to its first point with f = B_phi/|B| <= -0.05 and on to
  the minimum of f (at most 0.5 m further); a centre not lost within S_max = (T_max + 2) x 1.2 x the
  period arc length 2 pi R_a / nfp blocks its cell and the loop is re-routed (up to 6 loops).  The
  disk proof follows the centre and 16 circle points to the centre's arc length (as for lost points,
  all disks batched); a disk whose bound fails is replaced by n = ceil(2 (bound - f_c) / -f_c) (2..8)
  smaller disks covering the same stretch of the loop (the bound grows about linearly with the radius;
  up to 3 refinements; still failing: its cell is blocked like a centre that is not lost).
  Certified disks of all loops form a pool (stretches of a re-routed loop
  already inside the pool's union get no new disks); the ring is the shortest chain of
  pool disks winding once around a whose consecutive centre segments are covered (Dijkstra on the
  two-sheeted cover of the overlap graph cut along a ray from a).  Q: vertex i on the ray from c_i
  towards a, as far in as the exact coverage of both adjacent edges allows; self-crossings cut
  exactly, keeping the piece of winding number 1.  (The model maps' certify_disks samples 25 points per
  disk: model only, not a proof.)
* Evidence: single-step reference checks of the tracer on manifold vertices (fast map vs the
  reference tracer on the reference field), the reference loop integral of A, the reference
  closure of X and the reference re-trace of the lost line.
* Period maps (FieldSection.step): a line whose per-period estimate exceeds 1e-7 scale (next to a coil) is
  re-traced with 4 x / 8 x (then 16 x / 32 x) the calibrated steps instead of being stopped, its error the larger
  of the finer trace's GBS estimate and the two traces' difference (the GBS estimate alone underestimated by up
  to 2.6 x next to coils: the reference check of refined steps in every run caught it); only lines still above 1e-4
  scale are stopped.
* Orbit periods (edge_barrier): the first round scans the lowest-order rationals at the edge of the rotation
  profile (the last lines before the first lost one; T_last's own rotation number when every line is lost)
  with q up to 64 for low-transform devices, and the profile's first ones; later rounds the rest and growing
  windows.  Weakly hyperbolic symmetric orbits (|residue| >= 5e-5) are screened after the strong ones (only
  the reversible closing can certify them).
* Candidates (selection only, no effect on any proof): with a request, orbits of q >= 5 whose chain polygon (the
  X points in angular order) encloses more than 2.5 x the request are not screened; the detection shooting
  takes at most 40 % of what is left, a candidate that is not the last in line at most 60 % (at least 60 s);
  when a round certifies nothing, the branches not lost are shot 4 x longer (extend_shooting, 30 % of what is
  left) and the candidates that gained a lost orbit are tried.  Thin-chain orbits whose 2D Newton on the
  coarse map does not converge can be taken from the accurate symmetric 1D root (symmetric_roots_accurate;
  off by default: on the known-surface checks it cost ~50 s per q at q ~ 20, where T_last lies far inside).
  When no profile line is lost within 0.7 x the coil distance, the profile and the scan reach 0.92 x.
* Screening is lazy (screen_many(lazy=True)): the accurate orbits, the quadratic term w and a provisional start
  domain (3e-5 scale / Lambda) for the detection shooting of every candidate; the measured start domain (the
  defect ladder: the screening's largest cost) and the accurate re-shoots only for the candidates tried
  (finalize_screened), each accurate line started inside the measured domain on the detected line's manifold
  orbit (parameter s_d Lambda^-k), so delta0 bounds its start error as before.  For a symmetric orbit one lost
  point per unstable family is certified up front (the reversible closing of one family needs the other's).
* Budget: every field evaluation of the section (its tracers, arc-length traces, densities) raises
  BarrierDeadline past deadline + grace (max(10 s, 3 % of the budget)); the stages stop at the deadline.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .fasttrace import LineTracer, trace_arclength


# ============================================================================ geometry
# the hard deadline of the running edge_barrier (also checked by the pure-geometry loops: polygon simplicity, loop
# cuts and decimation on curves of 10^4 - 10^5 vertices can take tens of seconds)
_HARD = [None]


def _hard_check():
    if _HARD[0] is not None and time.time() > _HARD[0]:
        raise BarrierDeadline(f"hard deadline passed by {time.time() - _HARD[0]:.1f} s (geometry)")


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def chord_param(P):
    P = np.asarray(P)
    if len(P) < 2:
        return np.zeros(len(P))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def turning_angles(P):
    """|turning angle| at each vertex of an open polyline (0 at its ends)."""
    P = np.asarray(P)
    n = len(P)
    th = np.zeros(n)
    if n >= 3:
        t = np.diff(P, axis=0)
        th[1:-1] = np.abs(np.arctan2(_cross(t[:-1], t[1:]), np.sum(t[:-1] * t[1:], 1)))
    return th


def _lag(xn, yn, x):
    k, m = xn.shape
    out = np.zeros((k, yn.shape[-1]))
    for j in range(m):
        l = np.ones(k)
        for i in range(m):
            if i != j:
                l = l * (x - xn[:, i]) / (xn[:, j] - xn[:, i])
        out += l[:, None] * yn[:, j]
    return out


def interp_curve(P, c, u):
    """Cubic Lagrange interpolation of the polyline P (chord parameter c) at parameters u.
    Returns (points (k, 2), error estimate (k,) = max |cubic - quadratic|)."""
    P = np.asarray(P, float)
    u = np.atleast_1d(np.asarray(u, float))
    n = len(P)
    i = np.clip(np.searchsorted(c, u, side="right") - 1, 0, n - 2)
    dseg = np.maximum(c[i + 1] - c[i], 1e-300)
    lin = P[i] + ((u - c[i]) / dseg)[:, None] * (P[i + 1] - P[i])
    if n < 4:
        return lin, c[i + 1] - c[i]
    lo = np.clip(i - 1, 0, n - 4)
    idx = lo[:, None] + np.arange(4)[None, :]
    cn = c[idx]
    span = cn[:, -1] - cn[:, 0]
    bad = np.min(np.diff(cn, axis=1), axis=1) <= 1e-12 * np.maximum(span, 1e-300)
    cub = _lag(cn, P[idx], u)
    q1 = _lag(cn[:, :3], P[idx[:, :3]], u)
    q2 = _lag(cn[:, 1:], P[idx[:, 1:]], u)
    est = np.maximum(np.linalg.norm(cub - q1, axis=1), np.linalg.norm(cub - q2, axis=1))
    cub = np.where(bad[:, None], lin, cub)
    est = np.where(bad, c[i + 1] - c[i], est)
    return cub, est


def seg_pairs(P, Q, pad=0.0, chunk=256):
    """Index pairs (i, j) of segments P[i]P[i+1], Q[j]Q[j+1] whose bounding boxes (inflated by pad)
    overlap."""
    P = np.asarray(P); Q = np.asarray(Q)
    if len(P) < 2 or len(Q) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    a0 = np.minimum(P[:-1], P[1:]) - pad
    a1 = np.maximum(P[:-1], P[1:]) + pad
    b0 = np.minimum(Q[:-1], Q[1:])
    b1 = np.maximum(Q[:-1], Q[1:])
    I, J = [], []
    for lo in range(0, len(a0), chunk):
        if lo % (16 * chunk) == 0:
            _hard_check()
        sl = slice(lo, lo + chunk)
        ov = ((a0[sl, None, 0] <= b1[None, :, 0]) & (a1[sl, None, 0] >= b0[None, :, 0]) &
              (a0[sl, None, 1] <= b1[None, :, 1]) & (a1[sl, None, 1] >= b0[None, :, 1]))
        ii, jj = np.nonzero(ov)
        I.append(ii + lo)
        J.append(jj)
    return np.concatenate(I), np.concatenate(J)


def seg_intersections(P, Q):
    """All proper intersections of the open polylines P and Q: arrays (i, j, t, u) with
    P[i] + t (P[i+1] - P[i]) = Q[j] + u (Q[j+1] - Q[j]), t, u in [0, 1]."""
    I, J = seg_pairs(P, Q)
    if len(I) == 0:
        return I, J, np.zeros(0), np.zeros(0)
    P = np.asarray(P); Q = np.asarray(Q)
    a = P[I + 1] - P[I]
    b = Q[J + 1] - Q[J]
    w = Q[J] - P[I]
    den = _cross(a, b)
    ok0 = den != 0
    d = np.where(ok0, den, 1.0)
    t = _cross(w, b) / d
    u = _cross(w, a) / d
    ok = ok0 & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return I[ok], J[ok], t[ok], u[ok]


def _pt_seg_dist(p, a, b):
    ab = b - a
    L2 = np.sum(ab * ab, -1)
    t = np.clip(np.sum((p - a) * ab, -1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def polyline_distance(pts, P):
    """Distance of each point (k, 2) to the polyline P (n, 2) (chunked brute force)."""
    pts = np.atleast_2d(pts)
    P = np.asarray(P)
    out = np.full(len(pts), np.inf)
    a, b = P[:-1], P[1:]
    for lo in range(0, len(pts), 64):
        d = _pt_seg_dist(pts[lo:lo + 64, None, :], a[None], b[None])
        out[lo:lo + 64] = np.min(d, axis=1)
    return out


def closed_polygon_simple(C, tol):
    """True iff the closed polygon C (n, 2) (last vertex joined to the first) has no two
    non-adjacent edges closer than tol and no adjacent edges folding back onto each other."""
    C = np.asarray(C)
    n = len(C)
    Pc = np.concatenate([C, C[:1]])
    I, J = seg_pairs(Pc, Pc, pad=tol)
    keep = I < J
    I, J = I[keep], J[keep]
    adj = (J - I == 1) | ((I == 0) & (J == n - 1))
    # adjacent edges: reject only folding back (angle between them ~ pi)
    if np.any(adj):
        ia, ja = I[adj], J[adj]
        e1 = Pc[ia + 1] - Pc[ia]
        e2 = Pc[ja + 1] - Pc[ja]
        cosang = np.sum(e1 * e2, 1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-300)
        if np.any(cosang < -1 + 1e-12):
            return False
    I, J = I[~adj], J[~adj]
    if len(I) == 0:
        return True
    a0, a1, b0, b1 = Pc[I], Pc[I + 1], Pc[J], Pc[J + 1]
    # intersection test
    a = a1 - a0
    b = b1 - b0
    w = b0 - a0
    den = _cross(a, b)
    d = np.where(den != 0, den, 1.0)
    t = _cross(w, b) / d
    u = _cross(w, a) / d
    inter = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    if np.any(inter):
        return False
    dist = np.minimum(np.minimum(_pt_seg_dist(a0, b0, b1), _pt_seg_dist(a1, b0, b1)),
                      np.minimum(_pt_seg_dist(b0, a0, a1), _pt_seg_dist(b1, a0, a1)))
    return bool(np.all(dist > tol))


def _dedupe_closed(C, E, tol):
    """Drop vertices of a closed polygon closer than tol to their predecessor (the merged segment
    keeps the larger error)."""
    C = np.asarray(C)
    E = np.asarray(E, float).copy()
    keep = np.ones(len(C), bool)
    for k in range(1, len(C)):
        if np.linalg.norm(C[k] - C[k - 1]) <= tol:
            keep[k] = False
            E[k - 1] = max(E[k - 1], E[k])
    if np.linalg.norm(C[0] - C[-1]) <= tol and keep[-1]:
        keep[-1] = False
        E[-2] = max(E[-2], E[-1])
    return C[keep], E[keep]


def simplify_closed_curve(C, E, a, tol, max_cuts=200):
    """Remove the self-intersections of the closed polygon C (segment k: C[k] -> C[k+1 mod n], error
    E[k]) by cutting loops at certified crossings: at a crossing of segments i < j certified by the
    parallelogram lemma (the true arcs meet in the box), the curve splits into C[0..i] + p + C[j+1..]
    and p + C[i+1..j]; both consist of true arcs, and the one with winding number 1 around a is kept
    (errors next to p raised to the box diameter).  Returns (C, E, info) or (None, None, info)."""
    C = np.asarray(C, float)
    E = np.asarray(E, float)
    info = dict(cuts=0, boxes=[])
    for it in range(max_cuts):
        _hard_check()
        P = np.concatenate([C, C[:1]])
        n = len(C)
        I, J, T, U = seg_intersections(P, P)
        keep = (J > I + 1) & ~((I == 0) & (J == n - 1))
        I, J, T, U = I[keep], J[keep], T[keep], U[keep]
        if len(I) == 0:
            return C, E, info
        o = np.argsort(J - I)                       # smallest loop first
        i, j = int(I[o[0]]), int(J[o[0]])
        cc = certify_crossing(P, E, P, E, i, j)
        if not cc["ok"]:
            info["reason"] = f"self-intersection of segments {i}, {j} not certified ({cc.get('reason')})"
            return None, None, info
        p_ = np.array(cc["h"])
        bd = cc["box_diam"]
        Ca = np.concatenate([C[:i + 1], p_[None], C[j + 1:]])
        Ea = np.concatenate([E[:i], [max(E[i], bd)], [max(E[j], bd)], E[j + 1:]])
        Cb = np.concatenate([p_[None], C[i + 1:j + 1]])
        Eb = np.concatenate([[max(E[i], bd)], E[i + 1:j], [max(E[j], bd)]])
        Ca, Ea = _dedupe_closed(Ca, Ea, tol)
        Cb, Eb = _dedupe_closed(Cb, Eb, tol)
        wa, wb = winding_number(Ca, a), winding_number(Cb, a)
        info["cuts"] += 1
        info["boxes"].append(bd)
        if abs(wa) == 1:
            C, E = Ca, Ea
        elif abs(wb) == 1:
            C, E = Cb, Eb
        else:
            info["reason"] = f"loop cut gives winding numbers {wa}, {wb}"
            return None, None, info
    info["reason"] = "too many self-intersections"
    return None, None, info


def winding_number(C, p):
    C = np.asarray(C) - np.asarray(p)[None, :]
    ang = np.arctan2(C[:, 1], C[:, 0])
    d = np.diff(np.concatenate([ang, ang[:1]]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return int(round(np.sum(d) / (2 * np.pi)))


def inside_polygon(C, pts):
    """Even-odd point-in-polygon (pts (k, 2))."""
    C = np.asarray(C)
    pts = np.atleast_2d(pts)
    out = np.zeros(len(pts), bool)
    x0, y0 = C[:, 0], C[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for lo in range(0, len(pts), 256):
        x = pts[lo:lo + 256, 0][:, None]
        y = pts[lo:lo + 256, 1][:, None]
        cond = (y0[None] > y) != (y1[None] > y)
        xin = x0[None] + (y - y0[None]) * (x1 - x0)[None] / np.where(y1 != y0, y1 - y0, 1e-300)[None]
        out[lo:lo + 256] = (np.sum(cond & (x < xin), axis=1) % 2) == 1
    return out


def signed_area(C):
    C = np.asarray(C)
    D = np.roll(C, -1, axis=0)
    return 0.5 * float(np.sum(C[:, 0] * D[:, 1] - D[:, 0] * C[:, 1]))


# ============================================================================ section systems
class HenonSection:
    """Model system: Henon's area-preserving quadratic map
        (x, y) -> (x cos a - (y - x^2) sin a, x sin a + (y - x^2) cos a)
    with the elliptic fixed point at the origin (the 'axis'), density 1 (flux = area) and the lost
    region |(x, y)| > R_lost (orbits there escape to infinity).  Used by the unit tests: the whole
    barrier pipeline runs on it unchanged."""

    def __init__(self, alpha, R_lost=3.0):
        self.alpha = float(alpha)
        self.c, self.s = math.cos(alpha), math.sin(alpha)
        self.R_lost = float(R_lost)
        self.scale = 1.0
        self.axis = np.zeros(2)
        self.n_steps = 0

    def step(self, y, d):
        y = np.asarray(y, float).reshape(-1, 2)
        d = np.broadcast_to(np.asarray(d), (len(y),))
        x, w = y[:, 0], y[:, 1]
        c, s = self.c, self.s
        fx = x * c - (w - x * x) * s
        fy = x * s + (w - x * x) * c
        bx = x * c + w * s
        by = -x * s + w * c + bx * bx
        out = np.where((d > 0)[:, None], np.stack([fx, fy], -1), np.stack([bx, by], -1))
        lost = ~np.all(np.isfinite(out), -1) | (np.hypot(out[:, 0], out[:, 1]) > self.R_lost) | \
            (np.hypot(x, w) > self.R_lost)
        self.n_steps += len(y)
        tau = 4e-16 * (1.0 + np.sum(out * out, -1))
        return out, lost, tau

    def density(self, y):
        return np.ones(len(np.atleast_2d(y)))

    def flux(self, C):
        A = abs(signed_area(C))
        return A, 1e-15 * A

    def density_check(self, C, h=None):
        return dict(ok=True, min_lower_bound=1.0, method="density 1")

    def density_max(self, pts):
        return 1.0

    def certify_lost(self, z, e, d, nmax=400):
        r = max(2.0 * e, 1e-12)
        th = 2 * np.pi * np.arange(16) / 16
        pts = np.concatenate([np.atleast_2d(z), np.asarray(z)[None] + r * np.stack([np.cos(th), np.sin(th)], -1)])
        y = pts.copy()
        lost_at = np.full(len(pts), -1)
        for n in range(nmax):
            y, lost, _ = self.step(y, np.full(len(y), d))
            new = lost & (lost_at < 0)
            lost_at[new] = n + 1
            if np.all(lost_at >= 0):
                break
        ok = bool(np.all(lost_at >= 0))
        return dict(ok=ok, radius=r, lost_after=lost_at.tolist(), criterion=f"|(x,y)| > {self.R_lost}")

    def reference_steps(self, y, d):
        return None

    def lost_centres(self, Cz, nmax=200, **kw):
        """Model only: steps until |(x, y)| > R_lost (forward first, then backward); S = steps."""
        Cz = np.atleast_2d(np.asarray(Cz, float))
        n = len(Cz)
        lost = np.zeros(n, bool)
        dd = np.zeros(n, int)
        SS = np.full(n, np.nan)
        for d in (+1, -1):
            idx = np.flatnonzero(~lost)
            y = Cz[idx].copy()
            gone = np.zeros(len(idx), bool)
            for it in range(nmax):
                y, l, _ = self.step(y, d)
                new = l & ~gone
                SS[idx[new]] = it + 1
                gone |= l
                y[gone] = 0.0
                if gone.all():
                    break
            lost[idx[gone]] = True
            dd[idx[gone]] = d
        return dict(lost=lost, d=dd, S=SS, f=np.where(lost, -1.0, np.nan))

    def certify_disks(self, Cz, rr, d, S, nmax=200, **kw):
        """Model only (not a proof): a disk counts as lost when its centre, 16 boundary points and 8
        inner points all leave |(x, y)| < R_lost in direction d within nmax steps."""
        Cz = np.atleast_2d(np.asarray(Cz, float))
        rr = np.broadcast_to(np.asarray(rr, float), (len(Cz),))
        d = np.broadcast_to(np.asarray(d, int), (len(Cz),))
        th = 2 * np.pi * np.arange(16) / 16
        offs = np.concatenate([[[0.0, 0.0]], np.stack([np.cos(th), np.sin(th)], -1),
                               0.5 * np.stack([np.cos(th[::2]), np.sin(th[::2])], -1)])
        recs = []
        for c, r, dk, Sk in zip(Cz, rr, d, np.broadcast_to(S, (len(Cz),))):
            y = c[None] + r * offs
            gone = np.zeros(len(y), bool)
            for _ in range(nmax):
                y, l, _ = self.step(y, dk)
                gone |= l
                y[gone] = 0.0
                if gone.all():
                    break
            ok = bool(gone.all())
            recs.append(dict(ok0=ok, direction=int(dk), radius=float(r), arc_length=float(Sk), bound0=-1.0 if ok else 1.0,
                             f_centre=-1.0, lipschitz=0.0, stretch=1.0, x_start=c.tolist(), x_end=c.tolist(),
                             factor=1 if ok else 2, reason=None if ok else "not all samples lost (model)"))
        return recs

    def ring_reference(self, X0, d, S, X_end, **kw):
        return dict(diff=np.zeros(len(np.atleast_2d(X0))), f_ref=-np.ones(len(np.atleast_2d(X0))), dop=[])


class TwistKickSection(HenonSection):
    """Model system: a kick (x, y) -> (x, y + K x exp(-x^2 / w^2)) followed by the twist rotation by
    omega0 + omega2 (x^2 + y^2) (both area preserving).  The kick dies off at large |x|, so far from
    the origin the map is a pure twist: chaotic layers around resonances are confined between
    invariant circles (no lost region unless R_lost is set).  Used for manifold-band tests."""

    def __init__(self, K, w=0.6, omega0=0.4, omega2=1.0, R_lost=50.0):
        super().__init__(0.0, R_lost=R_lost)
        self.K, self.w, self.o0, self.o2 = float(K), float(w), float(omega0), float(omega2)

    def step(self, y, d):
        y = np.asarray(y, float).reshape(-1, 2)
        d = np.broadcast_to(np.asarray(d), (len(y),))
        x, v = y[:, 0], y[:, 1]
        # forward: kick then rotate
        vk = v + self.K * x * np.exp(-x * x / self.w ** 2)
        om = self.o0 + self.o2 * (x * x + vk * vk)
        fx = x * np.cos(om) - vk * np.sin(om)
        fy = x * np.sin(om) + vk * np.cos(om)
        # backward: rotate back then unkick
        omb = self.o0 + self.o2 * (x * x + v * v)
        bx = x * np.cos(omb) + v * np.sin(omb)
        by = -x * np.sin(omb) + v * np.cos(omb)
        by = by - self.K * bx * np.exp(-bx * bx / self.w ** 2)
        out = np.where((d > 0)[:, None], np.stack([fx, fy], -1), np.stack([bx, by], -1))
        lost = ~np.all(np.isfinite(out), -1) | (np.hypot(out[:, 0], out[:, 1]) > self.R_lost)
        self.n_steps += len(y)
        tau = 4e-15 * (1.0 + np.sum(out * out, -1)) * (1.0 + abs(self.o2))
        return out, lost, tau


class BarrierDeadline(Exception):
    """Raised by a section's field evaluations past its hard deadline (edge_barrier catches it)."""


class ReversibleEdgeSection(HenonSection):
    """Model system for the reversible closing: the symmetric splitting F = K_1/2 o R o K_1/2 of a kick
    K_t(x, y) = (x, y + t k f(x)), f(x) = (x + c2 x^2) exp(-x^2 / w^2), and a twist rotation R by omega0 +
    omega2 (x^2 + y^2) (the model of torux/islandband.py), area preserving and reversible with S(x, y) = (x, -y)
    (S F S = F^-1, like a stellarator-symmetric period map in a symmetry plane), with the lost region
    |(x, y)| > R_lost.  Covered circles are the invariant circles around the origin inside R_lost (a point
    inside one never leaves it, so lost points lie outside every one, as Lemma A gives for a field)."""

    def __init__(self, k, w=0.6, omega0=1.0, omega2=1.0, c2=0.7, R_lost=50.0):
        super().__init__(0.0, R_lost=R_lost)
        self.k, self.w, self.o0, self.o2, self.c2 = float(k), float(w), float(omega0), float(omega2), float(c2)
        self.nfp = 1

        class _Src:
            stellsym = True
        self.source = _Src()

    def symmetry_defect(self, pts):
        return 0.0                                       # exactly reversible

    def _kick(self, x, y, t):
        return y + t * self.k * (x + self.c2 * x * x) * np.exp(-x * x / self.w ** 2)

    def _rot(self, x, y, sgn):
        om = sgn * (self.o0 + self.o2 * (x * x + y * y))
        c, s = np.cos(om), np.sin(om)
        return x * c - y * s, x * s + y * c

    def step(self, y, d):
        y = np.asarray(y, float).reshape(-1, 2)
        d = np.broadcast_to(np.asarray(d), (len(y),))
        sg = np.where(d > 0, 1.0, -1.0)
        x0, y0 = y[:, 0], y[:, 1] * sg                   # backward: F^-1 = S F S
        y1 = self._kick(x0, y0, 0.5)
        x1, y1 = self._rot(x0, y1, +1)
        y1 = self._kick(x1, y1, 0.5)
        out = np.stack([x1, y1 * sg], -1)
        lost = ~np.all(np.isfinite(out), -1) | (np.hypot(out[:, 0], out[:, 1]) > self.R_lost) | \
            (np.hypot(y[:, 0], y[:, 1]) > self.R_lost)
        self.n_steps += len(y)
        tau = 4e-15 * (1.0 + np.sum(out * out, -1)) * (1.0 + abs(self.o2) + abs(self.k))
        return out, lost, tau

    def symmetric_orbits(self, q, r_lo, r_hi, n=200):
        """Symmetric period-q orbits with a point on y = 0 in (r_lo, r_hi) (either sign of x), by Newton from a
        grid; distinct ones."""
        out = []
        for x0 in np.concatenate([np.linspace(r_lo, r_hi, n), -np.linspace(r_lo, r_hi, n)]):
            orb = orbit_from_point(self, np.array([x0, 0.0]), q, polish=8)
            if orb is None or orb.info["closure"] > 1e-12:
                continue
            if abs(orb.pts[0, 1]) > 1e-9 or np.min(np.linalg.norm(orb.pts[1:] - orb.pts[0], axis=1)) < 1e-6:
                continue
            if any(np.min(np.linalg.norm(o.pts - orb.pts[0], axis=1)) < 1e-7 for o in out):
                continue
            out.append(orb)
        return out


class FieldSection:
    """The section phi = 0 of a source's field: period maps by torux.fasttrace (engine field), densities,
    fluxes (loop integrals of A), the B_phi > 0 check, lost-point proofs and reference checks.

    Period maps: k = 4 GBS at sp macro steps per period; a line whose per-period error estimate exceeds
    refine_tol x scale (1e-7; a passage next to a coil or a turning point of phi) is re-traced with 4 x and then
    16 x the steps (refine_factors); only a line still above 1e-4 scale (or physically stopped: B_phi
    turning, R <= 0) is stopped.
    hard_deadline: every field evaluation through the section (its tracers, arc-length traces, densities)
    raises BarrierDeadline past it, so no stage can overrun the caller's budget."""

    def __init__(self, engine, source, sign=1.0, k=4, steps_per_period=12, reference=True, scale=None, phi0=0.0):
        self.engine = engine
        self.source = source
        self.nfp = int(source.nfp)
        self.sign = float(sign)
        self.hard_deadline = None
        self.tracer = LineTracer(engine, self.nfp, k=k, steps_per_period=steps_per_period, sign=self.sign,
                                 field=self._B_engine)
        self.k, self.sp = k, steps_per_period
        self.use_reference = reference
        self._ref = None
        self.scale = 1.0 if scale is None else float(scale)
        self.axis = None
        self.n_steps = 0
        self.n_refined = 0
        self.stop_on_err = True
        self.refine_tol = 1e-7
        self.refine_factors = (4, 16)
        self.refined_samples = []
        self.phi0 = float(phi0)

    def check_deadline(self):
        if self.hard_deadline is not None and time.time() > self.hard_deadline:
            raise BarrierDeadline(f"hard deadline passed by {time.time() - self.hard_deadline:.1f} s")

    def _B_engine(self, x):
        self.check_deadline()
        from . import fast_trace as _so4
        if _so4.on("bonly") or isinstance(self.engine, _so4.LeanTraceEngine):
            # (fast_trace 'bonly') the same kernels and order without evaluate's other arrays and JAX conversions
            return _so4.field_B(self.engine, np.asarray(x, float))
        return np.asarray(self.engine.evaluate(np.asarray(x, float), grad=False, A=False)["B"])

    def line_tracer(self, k=4, steps_per_period=None):
        """A LineTracer on this section's field (deadline-checked)."""
        return LineTracer(self.engine, self.nfp, k=k, steps_per_period=steps_per_period or self.sp, sign=self.sign,
                          field=self._B_engine)

    def coarse(self):
        """The same section with sp // 3 macro steps per period (detection only: its errors are not used)."""
        c = FieldSection(self.engine, self.source, sign=self.sign, k=4, steps_per_period=max(4, self.sp // 3),
                         reference=False, scale=self.scale, phi0=self.phi0)
        c._ref = self._ref
        c.stop_on_err = False                # detection only: its error estimates are not used
        c.refine_factors = ()
        c.hard_deadline = self.hard_deadline
        return c

    # -- field
    def B(self, x):
        self.check_deadline()
        from . import fast_trace as _so4
        if _so4.on("bonly") or isinstance(self.engine, _so4.LeanTraceEngine):
            return _so4.field_B(self.engine, np.asarray(x, float))      # (fast_trace 'bonly', 'tlean')
        return np.asarray(self.engine.evaluate(np.asarray(x, float), grad=False, A=False)["B"])

    def ref_field(self):
        if self._ref is None:
            from .reference.fast64 import Field
            self._ref = Field(self.source)
        return self._ref

    def xyz(self, y):
        y = np.atleast_2d(y)
        if self.phi0 == 0.0:
            return np.stack([y[:, 0], np.zeros(len(y)), y[:, 1]], -1)
        return np.stack([y[:, 0] * math.cos(self.phi0), y[:, 0] * math.sin(self.phi0), y[:, 1]], -1)

    def _bphi_of(self, B):
        """B_phi in the plane phi0 from Cartesian B (n, 3)."""
        if self.phi0 == 0.0:
            return B[:, 1]
        return -B[:, 0] * math.sin(self.phi0) + B[:, 1] * math.cos(self.phi0)

    # -- maps
    def step(self, y, d):
        y = np.asarray(y, float).reshape(-1, 2)
        self.check_deadline()
        dd = np.broadcast_to(np.asarray(d, float), (len(y),))
        yy, lost, err = self.tracer.period_map(y, 1, direction=dd, phi0=self.phi0)
        self.n_steps += len(y)
        if self.refine_factors and len(y):
            # lines with a large per-period estimate (next to a coil, or the GBS stages straying into a
            # turning region) are re-traced with more steps rather than stopped; lines the tracer stopped
            # are re-traced once too (a stop may come from an inaccurate stage near a coil)
            tol_r = self.refine_tol * self.scale
            for r_, fac in enumerate(self.refine_factors):
                bad = (~(err <= tol_r) & ~lost) if r_ else (~(err <= tol_r) | lost)
                if not np.any(bad):
                    break
                idx = np.flatnonzero(bad)
                # traces at fac and 2 fac x the steps: the finer one is kept, and its error is taken as the larger of
                # its own GBS estimate and the difference of the two (step doubling -- an overestimate of the finer
                # trace's error).  Next to coils the GBS estimate alone is not in its asymptotic regime: on QUASR
                # 819202 refined steps differed from the reference tracer by 2.6 x their 2 x estimate.
                y1, l1, e1 = self.tracer.period_map(y[idx], 1, direction=dd[idx], phi0=self.phi0,
                                                    nsteps=self.sp * fac)
                y2, l2, e2 = self.tracer.period_map(y[idx], 1, direction=dd[idx], phi0=self.phi0,
                                                    nsteps=self.sp * fac * 2)
                dif = np.linalg.norm(y2 - y1, axis=1)
                e2 = np.where(l1 | l2, e2, np.maximum(e2, dif))
                yy[idx], lost[idx], err[idx] = y2, l2, e2       # the finer trace replaces the coarser one
                self.n_refined += len(idx)
                # a few refined steps are kept for the reference evidence (reference_evidence checks them)
                for i_ in idx[~l2][:2]:
                    if len(self.refined_samples) < 48:
                        self.refined_samples.append((y[i_].copy(), float(dd[i_])))
        # tau: the GBS estimate is the error of the order 2k-2 value; it bounds the order-2k value's
        # error only in the asymptotic regime (measured against the reference tracer on edge points:
        # k = 6 at 6 steps per period underestimates by up to 2.6x; k = 4 at 12 steps per period, the
        # setting used, has error <= 0.16 x (2 x estimate + 2e-13 R) on three devices, at 16 <= 0.07); used:
        # 2 x estimate + 2e-13 R (checked on samples by the reference evidence of every run)
        floor = 2e-13 * np.maximum(1.0, np.abs(yy[:, 0]))
        # a period map whose own error estimate is still large after the refinements is unreliable, so the
        # line is stopped like a lost one (stops only truncate curves and shooting lines; lost-ness is
        # proven separately by certify_lost)
        if self.stop_on_err:
            lost = lost | ~np.isfinite(err) | (err > 1e-4 * self.scale)
        return yy, lost, 2.0 * err + floor

    def density(self, y):
        return self.sign * self._bphi_of(self.B(self.xyz(y)))

    # -- flux of a closed polygon in the plane phi = phi0 (counter-clockwise in (R, Z) -> int B_phi)
    def flux(self, C, field=None):
        C = np.asarray(C, float)
        if signed_area(C) < 0:
            C = C[::-1]
        D = np.roll(C, -1, axis=0)
        res = []
        for ng in (3, 5):
            g, w = np.polynomial.legendre.leggauss(ng)
            pts = np.concatenate([0.5 * (C + D) + 0.5 * gi * (D - C) for gi in g])
            x = self.xyz(pts)
            if field is None:
                A = np.asarray(self.engine.evaluate(x, grad=False, A=True)["A"])
            else:
                A = field.evaluate(x, grad=False)["A"]
            n = len(C)
            dR = (D - C)[:, 0]
            dZ = (D - C)[:, 1]
            AR = A[:, 0] if self.phi0 == 0.0 else A[:, 0] * math.cos(self.phi0) + A[:, 1] * math.sin(self.phi0)
            tot = 0.0
            for j in range(ng):
                sl = slice(j * n, (j + 1) * n)
                tot += 0.5 * w[j] * np.sum(AR[sl] * dR + A[sl, 2] * dZ)
            res.append(-self.sign * tot)
        return float(res[1]), float(abs(res[1] - res[0]))

    def density_check(self, C, h=None, n_target=120):
        """Lower bound of B_phi on the polygon's interior: grid (spacing h) covering it, the minimum of
        B_phi at the grid nodes within one cell of the interior less max|grad B| h / sqrt 2 (x 2)."""
        C = np.asarray(C, float)
        lo, hi = C.min(0), C.max(0)
        if h is None:
            h = float(np.max(hi - lo)) / n_target
        gx = np.arange(lo[0] - h, hi[0] + 2 * h, h)
        gz = np.arange(lo[1] - h, hi[1] + 2 * h, h)
        GX, GZ = np.meshgrid(gx, gz, indexing="ij")
        pts = np.stack([GX.ravel(), GZ.ravel()], -1)
        ins = inside_polygon(C, pts)
        # grid points within 1.5 h of the boundary (a superset: vertex distance <= 1.5 h + half the
        # longest edge; extra points only enter the minimum)
        from scipy.spatial import cKDTree
        lmax = float(np.max(np.linalg.norm(np.diff(np.concatenate([C, C[:1]]), axis=0), axis=1)))
        dv, _ = cKDTree(C).query(pts[~ins], k=1)
        near = dv <= 1.5 * h + 0.5 * lmax
        sel = np.flatnonzero(ins)
        sel = np.concatenate([sel, np.flatnonzero(~ins)[near]])
        o = self.engine.evaluate(self.xyz(pts[sel]), grad=True, A=False)
        B = np.asarray(o["B"])
        G = np.asarray(o["G"])
        bphi = self.sign * self._bphi_of(B)
        gmax = float(np.max(np.sqrt(np.sum(G * G, axis=(1, 2)))))
        lb = float(np.min(bphi)) - 2.0 * gmax * h / math.sqrt(2)
        return dict(ok=bool(lb > 0), min_grid=float(np.min(bphi)), grad_max=gmax, h=h, n=int(len(sel)),
                    min_lower_bound=lb)

    def density_max(self, pts):
        return float(np.max(np.abs(self._bphi_of(self.B(self.xyz(pts)))))) * 1.05

    # -- lost points
    def _bphi_ratio(self, x, field=None):
        B = self.B(x) if field is None else field.B(np.asarray(x, float))
        ephi = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], -1) / np.hypot(x[:, 0], x[:, 1])[:, None]
        return self.sign * np.sum(B * ephi, -1) / np.linalg.norm(B, axis=1)

    def certify_lost(self, z, e, d, beta=0.05, S_max=16.0, verbose=False):
        """Proof that the field line through every point within r >= 2 e of z (plane phi = 0) reaches
        B_phi < 0 (module docstring).  The disk's centre and 16 boundary points are followed in arc
        length (dx/ds = sigma B/|B|, sigma = d s: phi increasing for d = +1) from the section, adaptive
        steps, the centre until its first point with s B_phi <= -beta |B| (arc length S), all to S.
        The image of the disk is a 2D patch: the linear map fitted to the boundary images gives an
        ellipse, the fit residual bounds the nonlinearity; f = s B_phi / |B| is sampled on the ellipse
        and every image point is shown to have f < 0 by max f + L (h + 2 nonlinearity + 2 dx) < 0,
        L = 2 x the largest |grad f| sampled, h the sample spacing, dx the larger of the integrator's
        estimate and the distance to an independent re-trace of the centre (scipy DOP853 on the
        reference field torux.reference.fast64)."""
        t0 = time.time()
        r = max(2.0 * float(e), 1e-6 * self.scale)
        th = 2 * np.pi * np.arange(16) / 16
        pts = np.concatenate([np.atleast_2d(z), np.asarray(z)[None] + r * np.stack([np.cos(th), np.sin(th)], -1)])
        out = dict(ok=False, radius=r, point=np.asarray(z).tolist(), direction=int(d), beta=beta)
        x0 = self.xyz(pts)
        sig = d * self.sign
        stop = lambda x: self._bphi_ratio(x) <= -beta
        a1 = trace_arclength(self.B, x0[:1], sig, stop, S_max=S_max, H0=1e-3, tol=1e-12)
        if not a1["stopped"][0]:
            out["reason"] = f"centre: B_phi did not reach -beta |B| within {S_max} m of arc length"
            out["end_ratio"] = float(self._bphi_ratio(a1["x"])[0])
            return out
        S = float(a1["s"][0])
        a2 = trace_arclength(self.B, x0, sig, lambda x: np.zeros(len(x), bool), S_max=S, H0=1e-3, tol=1e-12)
        xe = a2["x"]
        ierr = float(np.max(a2["err"]))
        # linear image of the disk: x_i - x_c = r (cos th_i a1 + sin th_i a2) + nonlinearity
        Amat = np.stack([np.cos(th), np.sin(th)], -1)
        coef, *_ = np.linalg.lstsq(Amat, (xe[1:] - xe[0]) / r, rcond=None)      # (2, 3)
        resid = float(np.max(np.linalg.norm(xe[1:] - xe[0] - r * (Amat @ coef), axis=1)))
        rr = np.linspace(0.0, 1.0, 9)
        tt = 2 * np.pi * np.arange(48) / 48
        RR, TT = np.meshgrid(rr, tt, indexing="ij")
        w = np.stack([(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()], -1)
        X = xe[0] + r * (w @ coef)
        f = self._bphi_ratio(X)
        Ax = r * np.linalg.norm(coef, axis=1)
        rho = float(np.max(np.linalg.norm(X - xe[0], axis=1)))
        h = float(np.sum(Ax) * max(1.0 / 8, 2 * np.pi / 48))
        idx = np.linspace(0, len(X) - 1, 9).astype(int)
        hh = 1e-5
        E3 = np.eye(3)
        gr = []
        for i in idx:
            g = [(self._bphi_ratio(X[i:i + 1] + hh * E3[k:k + 1])[0] - self._bphi_ratio(X[i:i + 1] - hh * E3[k:k + 1])[0])
                 / (2 * hh) for k in range(3)]
            gr.append(np.linalg.norm(g))
        L = 2.0 * float(np.max(gr))
        ref = None
        if self.use_reference:
            ref = self.ref_check_arclength(x0[0], sig, S, beta, xe[0])
            out["reference"] = ref
            ierr = max(ierr, float(ref.get("end_point_diff", np.inf)))
        bound = float(np.max(f)) + L * (h + 2.0 * resid + 2.0 * ierr)
        out.update(arc_length=S, end_point=xe[0].tolist(), end_ratio=float(f[0]), ratios_max=float(np.max(f)),
                   image_radius=rho, nonlinearity=resid, sample_spacing=h, lipschitz=L, integration_err=ierr,
                   upper_bound_f=bound)
        ok = bool(bound < 0.0) and (ref is None or ref["ok"])
        out["ok"] = ok
        out["time"] = time.time() - t0
        if not ok:
            out["reason"] = "disk test failed" + ("" if bound < 0 else f" (bound on f {bound:.3g})") + \
                ("" if ref is None or ref["ok"] else " (reference re-trace)")
        return out

    # -- lost-ring disks
    def lost_centres(self, Cz, beta=0.05, S_max=15.0, k=4, tol=1e-9, H_max=0.1, extra=0.5, rise=0.02):
        """Centre lines of ring disks (detection; the disk proof re-traces them): from each point of
        Sigma in arc length, forward first and backward for those not lost forward, to the first point
        with f = s B_phi/|B| <= -beta and on to the minimum of f (stopped once f exceeds its running
        minimum by `rise`, `extra` m after reaching -beta, or at S_max).  Returns dict of arrays
        lost, d (direction), S (arc length at the minimum of f), f (that minimum)."""
        Cz = np.atleast_2d(np.asarray(Cz, float))
        n = len(Cz)
        lost = np.zeros(n, bool)
        dd = np.zeros(n, int)
        SS = np.full(n, np.nan)
        ff = np.full(n, np.nan)
        for d in (+1, -1):
            idx = np.flatnonzero(~lost)
            if len(idx) == 0:
                break
            o = _arclength_to_min(self.B, self._bphi_ratio, self.xyz(Cz[idx]), d * self.sign, S_max, beta,
                                  extra=extra, rise=rise, k=k, tol=tol, H_max=H_max)
            m = o["lost"]
            lost[idx[m]] = True
            dd[idx[m]] = d
            SS[idx[m]] = o["S"][m]
            ff[idx[m]] = o["f"][m]
        return dict(lost=lost, d=dd, S=SS, f=ff)

    def certify_disks(self, Cz, rr, d, S, k=4, tol=1e-9, H_max=0.1):
        """Lost-disk proofs (certify_lost's argument) for many disks at once: disk i (centre Cz[i],
        radius rr[i]) is followed in direction d[i] over the arc length S[i] (lost_centres): centre and
        16 circle points traced in arc length (adaptive GBS, all disks batched); the linear map fitted
        to the circle's images gives the image ellipse (fit residual = nonlinearity); f = s B_phi/|B| on
        9 x 48 points of it; L = 2 x the largest |grad f| at 9 of them (central differences);
        bound0 = max f + L (h + 2 nonlinearity + 2 dx), h the sample spacing, dx = 2 x the summed GBS
        estimates (largest of the 17 lines).  lost_ring adds 2 L x the reference floor (ring_reference)
        and requires the result < 0.  Records: ok0 (bound0 < 0), bound0, lipschitz, stretch (largest
        singular value of the fitted map), f_centre, x_start / x_end of the centre line, factor (the
        refinement predicted from the linear growth of bound0 with the radius when bound0 >= 0)."""
        Cz = np.atleast_2d(np.asarray(Cz, float))
        n = len(Cz)
        rr = np.broadcast_to(np.asarray(rr, float), (n,))
        d = np.broadcast_to(np.asarray(d, int), (n,))
        S = np.broadcast_to(np.asarray(S, float), (n,))
        th = 2 * np.pi * np.arange(16) / 16
        Amat = np.stack([np.cos(th), np.sin(th)], -1)
        P17 = np.concatenate([Cz[:, None, :], Cz[:, None, :] + rr[:, None, None] * Amat[None]], axis=1)   # (n, 17, 2)
        X0 = self.xyz(P17.reshape(-1, 2))
        a2 = _arclength_lines(self.B, X0, np.repeat(d * self.sign, 17), np.repeat(S, 17), k=k, tol=tol, H_max=H_max)
        XE = a2["x"].reshape(n, 17, 3)
        reached = np.all((a2["s"] >= np.repeat(S, 17) * (1 - 1e-12)).reshape(n, 17), axis=1)
        ierr = 2.0 * np.max(a2["err"].reshape(n, 17), axis=1)
        coefs, resids = [], []
        for m_ in range(n):
            cf, *_ = np.linalg.lstsq(Amat, (XE[m_, 1:] - XE[m_, 0]) / rr[m_], rcond=None)
            coefs.append(cf)
            resids.append(float(np.max(np.linalg.norm(XE[m_, 1:] - XE[m_, 0] - rr[m_] * (Amat @ cf), axis=1))))
        rgrid = np.linspace(0.0, 1.0, 9)
        tgrid = 2 * np.pi * np.arange(48) / 48
        RR, TT = np.meshgrid(rgrid, tgrid, indexing="ij")
        wv = np.stack([(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()], -1)        # (432, 2)
        Xs = np.stack([XE[m_, 0][None] + rr[m_] * (wv @ coefs[m_]) for m_ in range(n)])   # (n, 432, 3)
        f = self._bphi_ratio(Xs.reshape(-1, 3)).reshape(n, -1)
        sel = np.linspace(0, Xs.shape[1] - 1, 9).astype(int)
        hh = 1e-5
        E6 = np.concatenate([np.eye(3), -np.eye(3)]) * hh
        fg = self._bphi_ratio((Xs[:, sel, None, :] + E6[None, None]).reshape(-1, 3)).reshape(n, len(sel), 6)
        Lip = 2.0 * np.max(np.linalg.norm((fg[..., :3] - fg[..., 3:]) / (2 * hh), axis=-1), axis=1)
        hsp = np.array([float(np.sum(rr[m_] * np.linalg.norm(coefs[m_], axis=1)) * max(1 / 8, 2 * np.pi / 48))
                        for m_ in range(n)])
        bound0 = np.max(f, axis=1) + Lip * (hsp + 2 * np.array(resids) + 2 * ierr)
        fc = f[:, 0]
        recs = []
        for m_ in range(n):
            ok0 = bool(reached[m_] and np.isfinite(bound0[m_]) and bound0[m_] < 0)
            if ok0:
                fac = 1
            elif fc[m_] < 0 and np.isfinite(bound0[m_]) and reached[m_]:
                fac = int(min(max(math.ceil(2.0 * (bound0[m_] - fc[m_]) / (-fc[m_])), 2), 8))
            else:
                fac = 0
            recs.append(dict(ok0=ok0, direction=int(d[m_]), radius=float(rr[m_]), arc_length=float(S[m_]),
                             bound0=float(bound0[m_]), f_centre=float(fc[m_]), lipschitz=float(Lip[m_]),
                             nonlinearity=resids[m_], integration_err=float(ierr[m_]),
                             stretch=float(np.linalg.svd(coefs[m_], compute_uv=False)[0]),
                             x_start=X0[17 * m_].tolist(), x_end=XE[m_, 0].tolist(), factor=fac,
                             reason=None if ok0 else ("line not completed" if not reached[m_] else "bound on f >= 0")))
        return recs

    def ring_reference(self, X0, d, S, X_end, beta=0.05, n_dop=1, k=6, tol=1e-11, H_max=0.05):
        """Independent re-traces of sampled ring-disk centre lines: GBS (k = 6, tol 1e-11) on the
        reference field torux.reference.fast64 for all given lines, scipy DOP853 for the first n_dop.
        Returns dict(diff (largest of the two re-traces per line), f_ref (f at the reference end), dop)."""
        X0 = np.atleast_2d(np.asarray(X0, float))
        X_end = np.atleast_2d(np.asarray(X_end, float))
        d = np.broadcast_to(np.asarray(d, int), (len(X0),))
        S = np.broadcast_to(np.asarray(S, float), (len(X0),))
        F = self.ref_field()
        ar = _arclength_lines(lambda x: F.B(np.asarray(x, float)), X0, d * self.sign, S, k=k, tol=tol, H_max=H_max)
        diff = np.linalg.norm(ar["x"] - X_end, axis=1)
        diff[~(ar["s"] >= S * (1 - 1e-12))] = np.inf
        f_ref = self._bphi_ratio(ar["x"], field=F)
        dops = []
        for m_ in range(min(n_dop, len(X0))):
            rd = self.ref_check_arclength(X0[m_], d[m_] * self.sign, S[m_], beta, X_end[m_])
            dops.append(rd)
            diff[m_] = max(diff[m_], float(rd.get("end_point_diff", np.inf)))
        return dict(diff=diff, f_ref=f_ref, dop=dops)

    def ref_check_arclength(self, x_start, sig, S, beta, x_end):
        """Independent re-trace of a line in arc length: scipy's DOP853 (rtol 1e-12) on the reference
        field (torux.reference.fast64), from x_start over arc length S; the end point must lie within
        1e-5 of x_end and have s B_phi <= -beta/2 |B| in the reference field."""
        from scipy.integrate import solve_ivp
        F = self.ref_field()

        hd = self.hard_deadline

        def rhs(_, x):
            if hd is not None and time.time() > hd:
                raise BarrierDeadline("hard deadline during the reference re-trace")
            B = F.B(np.asarray(x, float)[None])[0]
            return sig * B / np.linalg.norm(B)
        try:
            sol = solve_ivp(rhs, (0.0, S), np.asarray(x_start, float), method="DOP853", rtol=1e-12, atol=1e-14)
            xr = sol.y[:, -1]
            fr = float(self._bphi_ratio(xr[None], field=F)[0])
            dx = float(np.linalg.norm(xr - np.asarray(x_end)))
            return dict(ok=bool(sol.success and dx < 1e-5 and fr <= -beta / 2), end_point_diff=dx,
                        end_ratio_ref=fr, nfev=int(sol.nfev))
        except BarrierDeadline:
            raise
        except Exception as ex:  # noqa: BLE001
            return dict(ok=False, error=repr(ex), end_point_diff=np.inf)

    # -- reference single steps (evidence for the tracer error model)
    def reference_steps(self, y, d):
        from .reference import trace as RT
        F = self.ref_field()
        y = np.atleast_2d(np.asarray(y, float)).copy()
        if d < 0:
            y[:, 1] *= -1
        P = 2 * np.pi / self.nfp
        # (backward maps by the stellarator symmetry about the plane phi0, a symmetry plane: Z -> -Z)
        o = RT.poincare(F, y[:, 0], y[:, 1], [self.phi0 + P], 1, phi0=self.phi0, h=2 * np.pi / 512)
        out = np.stack([o["R"][:, 0, 0], o["Z"][:, 0, 0]], -1)
        if d < 0:
            out[:, 1] *= -1
        return out, o["err"][:, 0, 0]

    def symmetry_defect(self, pts):
        """max |B(S x) - S* B(x)| / |B| for the stellarator symmetry S (R, phi, Z) -> (R, -phi, -Z) at points
        (x, y, z) (k, 3), on the reference field: the backward reference checks rely on it."""
        F = self.ref_field()
        x = np.asarray(pts, float)
        xs = x * np.array([1.0, -1.0, -1.0])
        B = F.B(x)
        Bs = F.B(xs)
        # B_x(Sx) = -B_x(x)?  S maps (x, y, z) -> (x, -y, -z); the field transforms as B(Sx) = -S B(x)
        pred = -B * np.array([1.0, -1.0, -1.0])
        return float(np.max(np.linalg.norm(Bs - pred, axis=1) / np.linalg.norm(B, axis=1)))


# ============================================================================ periodic orbits
@dataclass
class Orbit:
    q: int
    pts: np.ndarray                  # (q, 2): pts[j] = P^j(pts[0])
    M: np.ndarray                    # monodromy of P^q at pts[0]
    e_X: float = 0.0                 # bound on the distance to the true periodic orbit
    p: int = None
    info: dict = field(default_factory=dict)

    @property
    def residue(self):
        return float((2.0 - np.trace(self.M)) / 4.0)

    def eig(self):
        w, V = np.linalg.eig(self.M)
        w = np.real(w)
        V = np.real(V)
        iu = int(np.argmax(np.abs(w)))
        is_ = 1 - iu
        vu = V[:, iu] / np.linalg.norm(V[:, iu])
        vs = V[:, is_] / np.linalg.norm(V[:, is_])
        return dict(lam=float(w[iu]), mu=float(w[is_]), vu=vu, vs=vs, reflection=bool(w[iu] < 0))


def orbit_from_point(sec, x0, q, h_fd=None, d=+1, polish=2):
    """Orbit through x0 (an approximate period-q point): Newton polish x0 <- x0 - (M - I)^-1 (P^q x0 - x0)
    (polish iterations, kept only while the closure decreases), points, closure, monodromy (central
    differences), and e_X."""
    best = None
    x = np.asarray(x0, float)
    for it in range(polish + 1):
        o = _orbit_once(sec, x, q, h_fd)
        if o is None:
            break
        if best is not None and o.info["closure"] >= best.info["closure"]:
            break
        best = o
        if it == polish or o.info["closure"] < 1e-14 * sec.scale:
            break
        try:
            x = x - np.linalg.solve(o.M - np.eye(2), o.info["F"])
        except np.linalg.LinAlgError:
            break
    if best is not None:
        best.info.pop("F", None)
    return best


def orbits_from_points(sec, X0s, q, h_fd=None, polish=2):
    """orbit_from_point for many approximate period-q points at once (one tracer call per period)."""
    X = np.atleast_2d(np.asarray(X0s, float)).copy()
    n = len(X)
    h = h_fd or 1e-7 * sec.scale
    E = np.array([[0.0, 0.0], [h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    best = [None] * n
    alive = np.ones(n, bool)
    for it in range(polish + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        y = (X[idx][:, None, :] + E[None]).reshape(-1, 2)
        pts = [X[idx].copy()]
        tau = np.zeros(len(idx))
        lost = np.zeros(len(idx), bool)
        for j in range(q):
            y, l, t = sec.step(y, np.ones(len(y)))
            l = l.reshape(-1, 5).any(1)
            lost |= l
            tau += t.reshape(-1, 5)[:, 0]
            if j < q - 1:
                pts.append(y.reshape(-1, 5, 2)[:, 0].copy())
        Y = y.reshape(-1, 5, 2)
        for m, i in enumerate(idx):
            if lost[m]:
                alive[i] = False
                continue
            M = np.stack([(Y[m, 1] - Y[m, 2]) / (2 * h), (Y[m, 3] - Y[m, 4]) / (2 * h)], -1)
            F = Y[m, 0] - X[i]
            clos = float(np.linalg.norm(F))
            if best[i] is not None and clos >= best[i].info["closure"]:
                alive[i] = False
                continue
            try:
                nrm = float(np.linalg.norm(np.linalg.inv(M - np.eye(2)), 2))
            except np.linalg.LinAlgError:
                nrm = np.inf
            P_ = np.array([p_[m] for p_ in pts])
            best[i] = Orbit(q, P_, M, e_X=2.0 * nrm * (clos + tau[m]),
                            info=dict(closure=clos, tau_q=float(tau[m]), inv_norm=nrm, det=float(np.linalg.det(M))))
            if it == polish or clos < 1e-14 * sec.scale or not np.isfinite(nrm):
                alive[i] = False
                continue
            X[i] = X[i] - np.linalg.solve(M - np.eye(2), F)
    return best


def rotation_numbers(sec, X0s, a0, q, sub=2, tracer=None):
    """rotation_number for many points at once (any tracer: only the winding is needed)."""
    X0s = np.atleast_2d(X0s)
    tr = tracer or sec.tracer
    y = np.concatenate([X0s, np.atleast_2d(a0)])
    o = tr.trace(y, 0.0, q * 2 * np.pi / sec.nfp, nsteps=q * 4 * sub, record=True)
    st = np.array([y] + o["rec"])
    d = st[:, :-1, :] - st[:, -1:, :]
    ang = np.unwrap(np.arctan2(d[:, :, 1], d[:, :, 0]), axis=0)
    p = np.round((ang[-1] - ang[0]) / (2 * np.pi)).astype(int)
    return [None if (o["lost"][i] or o["lost"][-1]) else int(p[i]) for i in range(len(X0s))]


def _orbit_once(sec, x0, q, h_fd=None):
    x0 = np.asarray(x0, float)
    h = h_fd or 1e-7 * sec.scale
    E = np.array([[h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    y = np.concatenate([x0[None], x0[None] + E])
    pts = [x0.copy()]
    tau = 0.0
    lost_any = False
    for j in range(q):
        y, lost, t = sec.step(y, np.ones(len(y)))
        tau += float(t[0])
        lost_any |= bool(np.any(lost))
        if j < q - 1:
            pts.append(y[0].copy())
    if lost_any:
        return None
    M = np.stack([(y[1] - y[2]) / (2 * h), (y[3] - y[4]) / (2 * h)], -1)
    clos = float(np.linalg.norm(y[0] - x0))
    try:
        nrm = float(np.linalg.norm(np.linalg.inv(M - np.eye(2)), 2))
    except np.linalg.LinAlgError:
        nrm = np.inf
    eX = 2.0 * nrm * (clos + tau)
    return Orbit(q, np.array(pts), M, e_X=eX, info=dict(closure=clos, tau_q=tau, inv_norm=nrm,
                                                        det=float(np.linalg.det(M)), F=y[0] - x0))


# ============================================================================ manifold growth
@dataclass
class Piece:
    pts: np.ndarray
    err: np.ndarray                  # propagated normal error bound (tracing + interpolation) per vertex
    amp: np.ndarray                  # product of the normal factors since the fundamental domain
    beta: np.ndarray                 # normal factor of the last step per vertex
    level: int
    owner: int                       # iteration index: pts[0] is near orbit.pts[owner]
    upre: np.ndarray                 # parameter of each vertex in the pre-image snapshot (s for level 0)
    snap: tuple = None               # (pts, chord param, err, amp) of the previous piece when this one was made
    truncated: bool = False          # the piece ends before the image of the whole previous piece
    validated: np.ndarray = None     # per segment: midpoint checked


class GenCurve:
    """A generating curve: one branch at X_0 (kind 'u' grown forward, 's' backward) and all its
    images under P^j (the branches at X_{j mod q})."""

    def __init__(self, name, kind, x0, v, sign, lam_abs, reflection, s0, n0, q, w=None, mult=None):
        self.name = name
        self.kind = kind
        self.d = +1 if kind == "u" else -1
        self.q = q
        self.reflection = reflection
        self.x0 = np.asarray(x0, float)
        self.v = np.asarray(v, float) * sign
        self.w = np.zeros(2) if w is None else np.asarray(w, float)      # quadratic term (both signs)
        mult = mult if mult is not None else lam_abs ** (2 if reflection else 1)
        s = s0 * mult ** (np.arange(n0 + 1) / n0)
        pts = self.x0[None] + s[:, None] * self.v[None] + (s ** 2)[:, None] * self.w[None]
        n = len(pts)
        self.pieces = [Piece(pts, np.zeros(n), np.ones(n), np.ones(n), 0, 0, s)]
        self.s0, self.mult = s0, mult
        self.base = 0                     # iteration index of the base point (piece j belongs to X_{base + d j})
        self.extra_err = 0.0              # per-level error added to every vertex (mirrored curves)
        self.active = True
        self.lost = None
        self.stop_reason = None

    def level(self):
        return self.pieces[-1].level


def multi_period(sec, Y, D, nper):
    """nper[i] period maps (direction D[i]) for every line i, all lines advanced together.  Returns
    (images, lost, summed tau)."""
    Y = np.array(Y, float)
    D = np.asarray(D, float)
    nper = np.asarray(nper, int)
    lost = np.zeros(len(Y), bool)
    tau = np.zeros(len(Y))
    for k in range(int(np.max(nper)) if len(nper) else 0):
        act = np.flatnonzero((nper > k) & ~lost)
        if len(act) == 0:
            break
        y, l, t = sec.step(Y[act], D[act])
        Y[act] = y
        lost[act] |= l
        tau[act] += t
    return Y, lost, tau


def _closest_on_parabola(x0, v, w, sgn, y, t0):
    """Distance from y to the curve t -> x0 + sgn t v + t^2 w near t0 (Newton on the parameter)."""
    t = float(t0)
    for _ in range(8):
        p = x0 + sgn * t * v + t * t * w
        dp = sgn * v + 2 * t * w
        g1 = 2 * np.dot(p - y, dp)
        g2 = 2 * (np.dot(dp, dp) + np.dot(p - y, 2 * w))
        if g2 <= 0:
            break
        t -= g1 / g2
    p = x0 + sgn * t * v + t * t * w
    return float(np.linalg.norm(p - y))


def local_manifold(sec, orb, kind, n0=16, s_start=None, target=None, deadline=None, n_def=6):
    """Quadratic parameterisation of the local manifold of kind 'u' (map F = P^nper forward) or 's'
    (F^-1) at X_0: p(s) = X_0 + s v + s^2 w with (Lambda^2 I - DF) w = D^2F[v, v] / 2 (second derivative
    by central differences), and the measured start error: the invariance defect d_i = distance of
    F(p(s_i)) from the curve p for the fundamental domain s_i in [s0, Lambda s0] (both signs).  A
    normal deviation at parameter s maps to mu times it at Lambda s, so the distance of the domain
    from the true manifold is at most sum_k mu^(k-1) d(s Lambda^-k) <= max d / (Lambda - mu) when the
    defect decreases at least linearly with s (it is O(s^3)); delta0 = 2 max d / (Lambda - mu).
    s0 is reduced until delta0 <= target.  Returns dict."""
    eig = orb.eig()
    refl = eig["reflection"]
    nper = orb.q * (2 if refl else 1)
    M = orb.M @ orb.M if refl else orb.M
    lu, ls = abs(eig["lam"]) ** (2 if refl else 1), abs(eig["mu"]) ** (2 if refl else 1)
    if kind == "u":
        Lam, mu, v, d, DF = lu, ls, eig["vu"], +1, M
    else:
        Lam, mu, v, d, DF = 1.0 / ls, 1.0 / lu, eig["vs"], -1, np.linalg.inv(M)
    scale = sec.scale
    target = target if target is not None else 1e-8 * scale
    X = orb.pts[0]
    h = 3e-4 * scale
    img, lost, _ = multi_period(sec, np.stack([X + h * v, X - h * v, X]), np.full(3, d), np.full(3, nper))
    if np.any(lost):
        w = np.zeros(2)
    else:
        D2 = (img[0] + img[1] - 2 * img[2]) / h ** 2
        w = np.linalg.solve(Lam ** 2 * np.eye(2) - DF, 0.5 * D2)
    s0 = s_start if s_start is not None else 3e-4 * scale / Lam
    out = dict(kind=kind, v=v, w=w, Lambda=Lam, mu=mu, nper=nper, d=d)
    for tries in range(6):
        # defect at n_def parameters spread over the fundamental domain [s0, Lambda s0], both signs
        si = s0 * Lam ** (np.arange(n_def) / (n_def - 1))
        pts = np.concatenate([X[None] + sg * si[:, None] * v[None] + (si ** 2)[:, None] * w[None] for sg in (+1, -1)])
        img, lost, tau = multi_period(sec, pts, np.full(len(pts), d), np.full(len(pts), nper))
        if np.any(lost):
            dmax = np.inf
        else:
            dd = []
            for k in range(len(pts)):
                sg = +1 if k < n_def else -1
                dd.append(_closest_on_parabola(X, v, w, sg, img[k], Lam * si[k % n_def]))
            dmax = float(np.max(dd))
        delta0 = 2.0 * dmax / max(Lam - mu, 1e-12)
        out.update(s0=s0, defect_max=dmax, delta0=delta0, tries=tries + 1)
        if delta0 <= target or (deadline is not None and time.time() > deadline):
            break
        # the defect is O(s^3) (plus a floor of tracing errors): aim at target / 3
        fac = (target / 3.0 / delta0) ** (1.0 / 3.0) if np.isfinite(delta0) and delta0 > 0 else 0.1
        s0 *= float(np.clip(fac, 0.05, 0.7))
    out["ok"] = bool(out["delta0"] <= target)
    return out


def mirror_curve(cu: GenCurve, name, base, defect_step):
    """The stable curve S(W^u) of a stellarator-symmetric field (P^-1 = S P S, S(R, Z) = (R, -Z)):
    every piece reflected; the base point is S(X_0) = X_base.  defect_step bounds |S P S - P^-1| per
    period map (from the field's measured symmetry defect) and is added per level to the errors."""
    S = np.array([1.0, -1.0])
    cs = GenCurve.__new__(GenCurve)
    cs.name, cs.kind, cs.d, cs.q, cs.reflection = name, "s", -1, cu.q, cu.reflection
    cs.x0, cs.v, cs.w = cu.x0 * S, cu.v * S, cu.w * S
    cs.s0, cs.mult, cs.active, cs.lost, cs.stop_reason = cu.s0, cu.mult, False, None, "mirror"
    cs.base = base
    cs.extra_err = defect_step
    cs.pieces = []
    for p in cu.pieces:
        snap = None if p.snap is None else (p.snap[0] * S, p.snap[1], p.snap[2], p.snap[3])
        cs.pieces.append(Piece(p.pts * S, p.err.copy(), p.amp.copy(), p.beta.copy(), p.level,
                               (base - p.level) % cu.q, p.upre.copy(), snap=snap, truncated=p.truncated))
    return cs


def _merge(order_key, *arrays):
    o = np.argsort(order_key, kind="stable")
    return [a[o] for a in arrays], o


def grow_step(sec, curves, dmax, amax, max_refine=14, max_pts=6000, min_seg=None, cut=None):
    """Advance every active curve by one period map with refinement (one batched tracer call per
    refinement round).  cut = (a, R): image points farther than R from a end the piece like lost
    points (the branch stops there; arcs used by closed curves are never extended through them)."""
    act = [c for c in curves if c.active]
    if not act:
        return
    min_seg = min_seg if min_seg is not None else 1e-12 * sec.scale
    st = []
    for c in act:
        pc = c.pieces[-1]
        cp = chord_param(pc.pts) if pc.level > 0 else pc.upre
        st.append(dict(c=c, pre=pc.pts.copy(), u=cp.copy(), perr=pc.err.copy(), pamp=pc.amp.copy(),
                       snap=(pc.pts, cp, pc.err, pc.amp), base=pc))
    Y = np.concatenate([s["pre"] for s in st])
    D = np.concatenate([np.full(len(s["pre"]), s["c"].d) for s in st])
    img, lost, tau = sec.step(Y, D)
    if cut is not None:
        lost = lost | (np.linalg.norm(img - np.asarray(cut[0])[None], axis=1) > cut[1])
    o = 0
    for s in st:
        n = len(s["pre"])
        s["img"], s["lost"], s["tau"] = img[o:o + n], lost[o:o + n], tau[o:o + n]
        o += n
    for rnd in range(max_refine):
        newY, newD, owners = [], [], []
        for s in st:
            I = s["img"]
            lostv = s["lost"]
            f = np.flatnonzero(lostv)
            nk = f[0] if len(f) else len(I)
            if nk < 2:
                s["new"] = None
                continue
            Ik = I[:nk]
            dseg = np.linalg.norm(np.diff(Ik, axis=0), axis=1)
            bad = dseg > dmax
            th = turning_angles(Ik)
            sharp = th > amax
            bad[:-1] |= sharp[1:-1]
            bad[1:] |= sharp[1:-1]
            pre_d = np.abs(np.diff(s["u"][:nk]))
            bad &= pre_d > min_seg
            if len(f):
                # the segment towards the first lost point: refine to localise the lost boundary a little
                pass
            idx = np.flatnonzero(bad)
            if len(idx) == 0 or nk > max_pts:
                s["new"] = None
                continue
            # several pre-images per bad segment: as many as the spacing / turning excess suggests
            thseg = np.maximum(th[idx], th[np.minimum(idx + 1, nk - 1)])
            m = np.clip(np.ceil(np.maximum(dseg[idx] / dmax, thseg / amax)), 2, 8).astype(int)
            rep_idx = np.repeat(idx, m - 1)
            frac = np.concatenate([np.arange(1, mm) / mm for mm in m])
            um = s["u"][rep_idx] + frac * (s["u"][rep_idx + 1] - s["u"][rep_idx])
            idx = rep_idx
            c = s["c"]
            pc = s["base"]
            if pc.level == 0:
                pm = c.x0[None] + um[:, None] * c.v[None] + (um ** 2)[:, None] * c.w[None]
                ins = np.zeros(len(um))
                em = np.maximum(s["perr"][idx], s["perr"][idx + 1])
            else:
                snap_pts, snap_c, snap_err, snap_amp = s["snap"]
                pm, ins = interp_curve(snap_pts, snap_c, um)
                em = np.maximum(s["perr"][idx], s["perr"][idx + 1]) + ins
            am = np.maximum(s["pamp"][idx], s["pamp"][idx + 1])
            s["new"] = (um, pm, em, am)
            newY.append(pm)
            newD.append(np.full(len(pm), c.d))
        if not newY:
            break
        img2, lost2, tau2 = sec.step(np.concatenate(newY), np.concatenate(newD))
        if cut is not None:
            lost2 = lost2 | (np.linalg.norm(img2 - np.asarray(cut[0])[None], axis=1) > cut[1])
        o = 0
        for s in st:
            if s.get("new") is None:
                continue
            um, pm, em, am = s["new"]
            n = len(um)
            key = np.concatenate([s["u"], um])
            (s["u"], s["pre"], s["perr"], s["pamp"], s["img"], s["lost"], s["tau"]), _ = _merge(
                key, key, np.concatenate([s["pre"], pm]), np.concatenate([s["perr"], em]),
                np.concatenate([s["pamp"], am]), np.concatenate([s["img"], img2[o:o + n]]),
                np.concatenate([s["lost"], lost2[o:o + n]]), np.concatenate([s["tau"], tau2[o:o + n]]))
            o += n
            s["new"] = None
    # finalise
    dens_jobs = []
    for s in st:
        c = s["c"]
        f = np.flatnonzero(s["lost"])
        nk = f[0] if len(f) else len(s["img"])
        if len(f) and c.lost is None:
            i = f[0]
            c.lost = dict(level=s["base"].level, point=s["pre"][i].copy(), err=float(s["perr"][i]),
                          amp=float(s["pamp"][i]), owner=s["base"].owner)
        if nk < 2:
            c.active = False
            c.stop_reason = "lost"
            s["skip"] = True
            continue
        s["skip"] = False
        s["nk"] = nk
        s["trunc"] = bool(len(f)) or s["base"].truncated
        dens_jobs.append(s)
    if not dens_jobs:
        return
    allp = np.concatenate([np.concatenate([s["pre"][:s["nk"]], s["img"][:s["nk"]]]) for s in dens_jobs])
    dens = sec.density(allp)
    o = 0
    for s in dens_jobs:
        c = s["c"]
        nk = s["nk"]
        dp, di = dens[o:o + nk], dens[o + nk:o + 2 * nk]
        o += 2 * nk
        pre, I = s["pre"][:nk], s["img"][:nk]
        # stretch along the curve: central chord ratio
        ip = np.minimum(np.arange(nk) + 1, nk - 1)
        im = np.maximum(np.arange(nk) - 1, 0)
        num = np.linalg.norm(I[ip] - I[im], axis=1)
        den = np.linalg.norm(pre[ip] - pre[im], axis=1)
        stretch = num / np.maximum(den, 1e-300)
        beta = (np.abs(dp / di)) / np.maximum(stretch, 1e-300)
        err = beta * s["perr"][:nk] + s["tau"][:nk]
        amp = beta * s["pamp"][:nk]
        lev = s["base"].level + 1
        piece = Piece(I.copy(), err, amp, beta, lev, (c.base + c.d * lev) % c.q, s["u"][:nk].copy(), snap=s["snap"],
                      truncated=s["trunc"])
        c.pieces.append(piece)
        if nk >= max_pts:
            c.active = False
            c.stop_reason = "max_pts"


def branch_polyline(orb, c: GenCurve, owner, parity=0):
    """The continuous computed branch of generating curve c at X_owner (for reflection-hyperbolic
    orbits: parity 0/1 selects the side).  Returns dict(pts (n, 2) starting with X_owner, refs list of
    (piece index, vertex index) (None for X), stop: True if the continuous part ended (truncation))."""
    q = c.q
    period = 2 * q if c.reflection else q
    first = (c.d * (owner - c.base)) % q + (q if (c.reflection and parity) else 0)   # P^-j: X_b -> X_{b-j}
    lv = list(range(first, c.level() + 1, period))
    pts = [orb.pts[owner][None]]
    refs = [(None, None)]
    ended = False
    for L in lv:
        pc = c.pieces[L]
        P = pc.pts
        # junction: a later piece's first vertex is the image of the fundamental domain's start, the
        # same curve point as the previous piece's last vertex (the domain's end) -- dropped; the
        # junction segment is then segment 0 of this piece for the midpoint validation
        i0 = 1 if len(pts) > 1 else 0
        pts.append(P[i0:])
        refs.extend((L, i) for i in range(i0, len(P)))
        if pc.truncated:
            ended = True
            break
    return dict(pts=np.concatenate(pts), refs=refs, ended=ended, levels=lv)


def branch_segments(br, v0, v1):
    """(piece level, segment index) of the branch polyline's segments v0..v1-1 (segment v joins
    vertices v and v+1); a junction segment maps to segment 0 of the later piece."""
    out = []
    refs = br["refs"]
    for v in range(v0, min(v1, len(refs) - 1)):
        (La, ia), (Lb, ib) = refs[v], refs[v + 1]
        if La is None or Lb is None:
            continue
        if La == Lb and ib == ia + 1:
            out.append((La, ia))
        elif La != Lb and ib >= 1:
            out.append((Lb, ib - 1))
    return [(L, i) for (L, i) in out if L > 0]


def _branch_dir(orb, a, br):
    """+1 if the branch leaves X counter-clockwise around a, -1 clockwise (first displaced vertex)."""
    X = br["pts"][0]
    P = br["pts"][1:]
    if len(P) == 0:
        return 0
    dv = P[0] - X
    return int(np.sign(_cross(X - a, dv)))


def vertex_errors(orb, c: GenCurve, refs, delta0):
    """Normal error bound of each branch vertex: propagated err + amp (delta0 + e_X); X itself e_X."""
    E = np.empty(len(refs))
    for i, (L, j) in enumerate(refs):
        if L is None:
            E[i] = orb.e_X + c.extra_err
        else:
            pc = c.pieces[L]
            E[i] = pc.err[j] + pc.amp[j] * (delta0 + orb.e_X) + c.extra_err * (L + 1)
    return E


# ============================================================================ chord validation
def refine_arcs(sec, jobs, target, rounds=7):
    """Midpoint validation/refinement of the segments of several curves at once: jobs = list of
    (curve, segs), segs = (level, i) segments (vertex i, i+1 of piece level).  For each segment the
    true curve point at the pre-image midpoint is computed (one period map from the cubic pre-image;
    all curves' midpoints in one batched tracer call per round) and inserted; repeated on the halves
    while the chord bound d theta_max / 4 exceeds target or the midpoint deviation exceeds the
    turning-angle model.  Returns statistics (converged = no segment left open)."""
    stats = dict(inserted=0, rounds=0, max_dev=0.0, inconsistent=0, converged=False)
    curves = []
    todo = {}
    for c, segs in jobs:
        if c not in curves:
            curves.append(c)
        ci = curves.index(c)
        for (L, i) in segs:
            todo.setdefault((ci, L), set()).add(i)
    for rnd in range(rounds):
        work = []
        for (ci, L), idxs in todo.items():
            c = curves[ci]
            if L == 0 or not idxs:
                continue
            pc = c.pieces[L]
            idx = np.array(sorted(i for i in idxs if 0 <= i < len(pc.pts) - 1), int)
            if len(idx) == 0:
                continue
            um = 0.5 * (pc.upre[idx] + pc.upre[idx + 1])
            snap_pts, snap_c, snap_err, snap_amp = pc.snap
            if L - 1 == 0:
                pm = c.x0[None] + um[:, None] * c.v[None] + (um ** 2)[:, None] * c.w[None]
                ins = np.zeros(len(um))
            else:
                pm, ins = interp_curve(snap_pts, snap_c, um)
            jj = np.clip(np.searchsorted(snap_c, um) - 1, 0, len(snap_c) - 2)
            em = np.maximum(snap_err[jj], snap_err[jj + 1]) + ins
            am = np.maximum(snap_amp[jj], snap_amp[jj + 1])
            work.append((ci, L, idx, um, pm, em, am))
        if not work:
            stats["converged"] = True
            break
        stats["rounds"] += 1
        Y = np.concatenate([w_[4] for w_ in work])
        D = np.concatenate([np.full(len(w_[2]), curves[w_[0]].d) for w_ in work])
        img, lost, tau = sec.step(Y, D)
        o = 0
        new_todo = {}
        for (ci, L, idx, um, pm, em, am) in work:
            n = len(idx)
            I, lo_, ta = img[o:o + n], lost[o:o + n], tau[o:o + n]
            o += n
            pc = curves[ci].pieces[L]
            P = pc.pts
            a_, b_ = P[idx], P[idx + 1]
            dev = _pt_seg_dist(I, a_, b_)
            th = turning_angles(P)
            pred = np.linalg.norm(b_ - a_, axis=1) * np.maximum(th[idx], th[idx + 1]) / 8.0
            beta_m = 0.5 * (pc.beta[idx] + pc.beta[idx + 1])
            err_m = beta_m * em + ta
            okm = ~lo_
            incons = dev > 2.0 * pred + 4.0 * (pc.err[idx] + pc.err[idx + 1] + err_m)
            stats["inconsistent"] += int(np.sum(incons & okm))
            stats["max_dev"] = max(stats["max_dev"], float(np.max(dev[okm])) if np.any(okm) else 0.0)
            keep = okm
            key = np.concatenate([pc.upre, um[keep]])
            (pc.upre, pc.pts, pc.err, pc.amp, pc.beta), order = _merge(
                key, key, np.concatenate([P, I[keep]]), np.concatenate([pc.err, err_m[keep]]),
                np.concatenate([pc.amp, beta_m[keep] * am[keep]]), np.concatenate([pc.beta, beta_m[keep]]))
            stats["inserted"] += int(np.sum(keep))
            inv = np.empty(len(order), int)
            inv[order] = np.arange(len(order))
            newpos = inv[len(P):]
            th2 = turning_angles(pc.pts)
            dl = np.linalg.norm(np.diff(pc.pts, axis=0), axis=1)
            chord = dl * np.maximum(th2[:-1], th2[1:]) / 4.0
            nxt = set()
            for k_, p_ in zip(np.flatnonzero(keep), newpos):
                for sidx in (p_ - 1, p_):
                    if 0 <= sidx < len(pc.pts) - 1 and (chord[sidx] > target or incons[k_]):
                        nxt.add(int(sidx))
            new_todo[(ci, L)] = nxt
        todo = new_todo
    if not any(todo.values()):
        stats["converged"] = True
    return stats


def refine_arc_segments(sec, c: GenCurve, segs, target, rounds=7):
    """refine_arcs for one curve."""
    return refine_arcs(sec, [(c, segs)], target, rounds)


def segment_chord_errors(P):
    th = turning_angles(P)
    dl = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return dl * np.maximum(th[:-1], th[1:]) / 4.0


# ============================================================================ crossings
def certify_crossing(U, eU, S, eS, i, j, ladder=24):
    """Certify that the true curves near the polylines U (vertex errors eU per segment) and S (eS)
    cross near the intersection of segments U[i]U[i+1] and S[j]S[j+1] (parallelogram lemma).
    eU, eS: per-segment error bounds (len n-1).  Returns dict(ok, h, box_diam, angle, a, b)."""
    U = np.asarray(U); S = np.asarray(S)
    du = U[i + 1] - U[i]
    ds = S[j + 1] - S[j]
    den = _cross(du, ds)
    if den == 0:
        return dict(ok=False, reason="parallel")
    w = S[j] - U[i]
    t = _cross(w, ds) / den
    h = U[i] + t * du
    eu = du / np.linalg.norm(du)
    es = ds / np.linalg.norm(ds)
    sth = abs(_cross(eu, es))
    Minv = np.linalg.inv(np.stack([eu, es], -1))

    def obl(P):
        return (Minv @ (np.asarray(P) - h).T).T          # (x along eu, y along es)
    xu = obl(U)
    xs = obl(S)
    eU = np.asarray(eU, float)
    eS = np.asarray(eS, float)

    def walk(X, E, k, along, across, dirn, reach, strip):
        """From segment k walk in direction dirn (+1: vertices k+1, k+2, ..; -1: k, k-1, ..) to the first
        vertex with |along| >= reach + eps/sth, every vertex on the way (and the crossing segment's
        own vertices) having |across| <= strip - eps/sth, eps = the largest segment error walked.
        Returns (ok, sign of along at the exit vertex, eps used)."""
        n = len(X)
        idx = k + 1 if dirn > 0 else k
        eps = float(E[k])
        while 0 <= idx < n:
            sidx = idx - 1 if dirn > 0 else idx            # the segment just walked (towards idx)
            if 0 <= sidx < len(E):
                eps = max(eps, float(E[sidx]))
            mar = eps / sth
            if abs(X[idx, across]) > strip - mar:
                return False, 0.0, eps
            if abs(X[idx, along]) >= reach + mar:
                return True, float(np.sign(X[idx, along])), eps
            idx += dirn
        return False, 0.0, eps

    r0 = 2.0 * (float(eU[i]) + float(eS[j])) / sth + 1e-300
    last = None
    for m in range(ladder):
        r = r0 * (2.0 ** m)
        a = b = r
        u1 = walk(xu, eU, i, 0, 1, +1, a, b)
        u2 = walk(xu, eU, i, 0, 1, -1, a, b)
        s1 = walk(xs, eS, j, 1, 0, +1, b, a)
        s2 = walk(xs, eS, j, 1, 0, -1, b, a)
        last = (u1, u2, s1, s2)
        if u1[0] and u2[0] and s1[0] and s2[0] and u1[1] * u2[1] < 0 and s1[1] * s2[1] < 0:
            corners = np.array([[a, b], [a, -b], [-a, b], [-a, -b]])
            cp = h + corners[:, :1] * eu + corners[:, 1:] * es
            diam = float(max(np.linalg.norm(cp[0] - cp[3]), np.linalg.norm(cp[1] - cp[2])))
            return dict(ok=True, h=h.tolist(), box_diam=diam, angle=float(math.asin(min(1.0, sth))), a=a, b=b,
                        eps_u=max(u1[2], u2[2]), eps_s=max(s1[2], s2[2]), t=float(t))
    return dict(ok=False, reason="no box size satisfies the parallelogram conditions",
                angle=float(math.asin(min(1.0, sth))), eps_u=float(eU[i]), eps_s=float(eS[j]), h=h.tolist())


# ============================================================================ the closed curves
def _order_ccw(orb, a):
    ang = np.arctan2(orb.pts[:, 1] - a[1], orb.pts[:, 0] - a[0])
    return list(np.argsort(ang))


def _gap_branches(orb, a, curves):
    """For each X_k: {('u'|'s', dir): (curve, parity)} with dir = +1 ccw, -1 cw."""
    table = {k: {} for k in range(orb.q)}
    for c in curves:
        for k in range(orb.q):
            for par in ((0, 1) if c.reflection else (0,)):
                if c.level() < ((c.d * (k - c.base)) % orb.q + (orb.q if par else 0)):
                    continue
                br = branch_polyline(orb, c, k, par)
                dr = _branch_dir(orb, a, br)
                table[k].setdefault((c.kind, dr), (c, par))
    return table


def find_gap_crossings(orb, a, curves, cls, cache=None):
    """Polyline crossings for every gap of class cls ('A': u of X_k ccw with s of X_k+1 cw; 'B': u of
    X_k+1 cw with s of X_k ccw).  Returns list per gap of dict or None.  cache: dict reused between
    calls; a gap is recomputed only when one of its two branch polylines changed."""
    order = _order_ccw(orb, a)
    table = _gap_branches(orb, a, curves)
    res = []
    gseq = range(orb.q) if cls == "A" else range(orb.q - 1, -1, -1)
    for g in gseq:
        k, k1 = order[g], order[(g + 1) % orb.q]
        if cls == "A":
            ku, uk, ks, sk = k, ("u", +1), k1, ("s", -1)
        else:
            ku, uk, ks, sk = k1, ("u", -1), k, ("s", +1)
        if uk not in table[ku] or sk not in table[ks]:
            res.append(None)
            continue
        cu, pu = table[ku][uk]
        cs, ps = table[ks][sk]
        bu = branch_polyline(orb, cu, ku, pu)
        bs = branch_polyline(orb, cs, ks, ps)
        key = (cls, g, cu.name, pu, len(bu["pts"]), float(bu["pts"][-1, 0]), cs.name, ps, len(bs["pts"]),
               float(bs["pts"][-1, 0]))
        if cache is not None and key in cache:
            I, J, T, Uu = cache[key]
        else:
            I, J, T, Uu = seg_intersections(bu["pts"], bs["pts"])
            if cache is not None:
                cache[key] = (I, J, T, Uu)
        # skip the trivial touch at a shared X (q = 1: both start at X_0)
        if ku == ks:
            keep = ~((I == 0) & (J == 0))
            I, J, T, Uu = I[keep], J[keep], T[keep], Uu[keep]
        if len(I) == 0:
            res.append(dict(found=False, ku=ku, ks=ks, cu=cu, cs=cs, pu=pu, ps=ps))
            continue
        o = np.lexsort((T, I))
        res.append(dict(found=True, ku=ku, ks=ks, cu=cu, cs=cs, pu=pu, ps=ps, I=I[o], J=J[o], T=T[o], U=Uu[o]))
    return res


def assemble_curve(sec, orb, a, curves, cls, delta0, target, verbose=False, max_try=3, request=None):
    """Closed curve of class cls from certified crossings, refined arcs, and its certification.
    Returns dict(ok, F_up, flux, ...) (JSON-friendly)."""
    gaps = find_gap_crossings(orb, a, curves, cls)
    out = dict(cls=cls, ok=False, delta0_used=float(delta0))
    if any(g is None or not g["found"] for g in gaps):
        out["reason"] = "no polyline crossing for some gap"
        out["gaps_found"] = [bool(g and g["found"]) for g in gaps]
        return out
    # uncertified estimate of the curve's flux (polylines to their first crossings): with a request, a
    # curve that already encloses more is not certified, and the chord target follows the margin
    rough = []
    for g in gaps:
        bu = branch_polyline(orb, g["cu"], g["ku"], g["pu"])
        bs = branch_polyline(orb, g["cs"], g["ks"], g["ps"])
        i, j = int(g["I"][0]), int(g["J"][0])
        rough.append(bu["pts"][:i + 1])
        rough.append(bs["pts"][:j + 1][::-1][:-1])
    Cr = np.concatenate(rough)
    try:
        F_rough, _ = sec.flux(Cr)
        L_rough = float(np.sum(np.linalg.norm(np.diff(np.concatenate([Cr, Cr[:1]]), axis=0), axis=1)))
        B_rough = sec.density_max(Cr[::max(1, len(Cr) // 200)])
    except Exception:  # noqa: BLE001
        F_rough, L_rough, B_rough = np.nan, np.nan, np.nan
    out["flux_estimate"] = float(F_rough)
    if request is not None and np.isfinite(F_rough):
        if F_rough >= abs(request):
            out["reason"] = f"closed curve encloses more than the request (estimate {F_rough:.6g})"
            return out
        # tube inflation at most 2 % of the margin: eps <= 0.02 margin / (2 L B); never above 10 x default
        target = float(np.clip(0.02 * (abs(request) - F_rough) / (2 * L_rough * B_rough), target, 10 * target))
    out["chord_target"] = target
    # refine the arcs up to the first crossing (all branches together), then re-find and certify
    jobs = []
    for g in gaps:
        for (c, par, k, idx_v) in ((g["cu"], g["pu"], g["ku"], int(g["I"][0])),
                                   (g["cs"], g["ps"], g["ks"], int(g["J"][0]))):
            br = branch_polyline(orb, c, k, par)
            jobs.append((c, branch_segments(br, 1, min(idx_v + 8, len(br["refs"]) - 1))))
    st = refine_arcs(sec, jobs, target)
    out["chord_validation"] = st
    if verbose:
        print(f"     refine {cls}: {st}", flush=True)
    if not st["converged"]:
        out["reason"] = "chord validation did not converge (midpoints inconsistent with the polyline)"
        return out
    gaps = find_gap_crossings(orb, a, curves, cls)
    if any(g is None or not g["found"] for g in gaps):
        out["reason"] = "crossing lost after refinement"
        return out
    poly, segerr, cert = [], [], []
    for g in gaps:
        bu = branch_polyline(orb, g["cu"], g["ku"], g["pu"])
        bs = branch_polyline(orb, g["cs"], g["ks"], g["ps"])
        Eu = vertex_errors(orb, g["cu"], bu["refs"], delta0)
        Es = vertex_errors(orb, g["cs"], bs["refs"], delta0)
        su = np.maximum(Eu[:-1], Eu[1:]) + np.concatenate([[0.0], segment_chord_errors(bu["pts"][1:])])
        ss = np.maximum(Es[:-1], Es[1:]) + np.concatenate([[0.0], segment_chord_errors(bs["pts"][1:])])
        best = None
        for t_ in range(min(max_try, len(g["I"]))):
            i, j = int(g["I"][t_]), int(g["J"][t_])
            cc = certify_crossing(bu["pts"], su, bs["pts"], ss, i, j)
            if cc["ok"]:
                best = (i, j, cc)
                break
        if best is None:
            out["reason"] = f"crossing not certified ({cc.get('reason')}, angle {cc.get('angle'):.3g})"
            out["crossing_attempt"] = {k: v for k, v in cc.items() if k != 'h'}
            return out
        i, j, cc = best
        h = np.array(cc["h"])
        # arc: X_ku -> ... -> U[i] -> h -> S[j] ... -> X_ks (reversed s branch)
        arc_u = np.concatenate([bu["pts"][:i + 1], h[None]])
        eu = np.concatenate([su[:i], [max(su[i], cc["box_diam"])]])
        arc_s = np.concatenate([h[None], bs["pts"][:j + 1][::-1]])
        es = np.concatenate([[max(ss[j], cc["box_diam"])], ss[:j][::-1]])
        poly.append(arc_u[:-1])
        poly.append(arc_s[:-1])
        segerr.append(eu)
        segerr.append(es)
        cert.append(dict(gap=(int(g["ku"]), int(g["ks"])), h=cc["h"], angle=cc["angle"], box_diam=cc["box_diam"],
                         eps_u=cc["eps_u"], eps_s=cc["eps_s"], n_u=int(i + 1), n_s=int(j + 1),
                         u_curve=g["cu"].name, s_curve=g["cs"].name))
    C = np.concatenate(poly)
    E = np.concatenate(segerr)                   # error of segment k: C[k] -> C[k+1]
    out["crossings"] = cert
    out["n_vertices"] = int(len(C))
    eps_max = float(np.max(E))
    out["eps_max"] = eps_max
    out["eps_median"] = float(np.median(E))
    # orientation: counter-clockwise around a
    w = winding_number(C, a)
    out["winding"] = w
    if w == -1:
        C = C[::-1]
        E = np.concatenate([E[:-1][::-1], E[-1:]]) if len(E) == len(C) else E[::-1]
        w = 1
    if w != 1:
        out["reason"] = f"winding number around the axis point {w}"
        return out
    dmin_a = float(polyline_distance(np.atleast_2d(a), np.concatenate([C, C[:1]]))[0])
    out["dist_axis"] = dmin_a
    if dmin_a <= eps_max:
        out["reason"] = "axis point within the curve's error"
        return out
    simple = closed_polygon_simple(C, tol=1e-12 * sec.scale)
    out["simple_as_assembled"] = simple
    if not simple:
        C2, E2, sinfo = simplify_closed_curve(C, E, a, 1e-12 * sec.scale)
        out["loop_cuts"] = {k: v for k, v in sinfo.items() if k != "boxes"}
        out["loop_cut_box_max"] = float(max(sinfo["boxes"])) if sinfo["boxes"] else 0.0
        if C2 is None:
            out["reason"] = "computed closed curve not simple: " + sinfo.get("reason", "")
            return out
        if winding_number(C2, a) == -1:
            C2 = C2[::-1]
            E2 = np.concatenate([E2[:-1][::-1], E2[-1:]])
        if not closed_polygon_simple(C2, tol=1e-12 * sec.scale) or winding_number(C2, a) != 1:
            out["reason"] = "closed curve not simple after loop cuts"
            return out
        C, E = C2, E2
        out["eps_max"] = float(np.max(E))
        dmin_a = float(polyline_distance(np.atleast_2d(a), np.concatenate([C, C[:1]]))[0])
        out["dist_axis"] = dmin_a
        if dmin_a <= out["eps_max"]:
            out["reason"] = "axis point within the curve's error"
            return out
    out["simple"] = True
    F, qerr = sec.flux(C)
    dens = sec.density_check(C)
    out["density_check"] = dens
    if not dens["ok"]:
        out["reason"] = "B_phi > 0 not established inside the curve"
        return out
    L = np.linalg.norm(np.diff(np.concatenate([C, C[:1]]), axis=0), axis=1)
    Bmax = sec.density_max(C)
    tube = float(np.sum((2.0 * E * L + np.pi * E ** 2)) * Bmax)
    out.update(flux=float(F), quad_err=float(qerr), tube=tube, perimeter=float(np.sum(L)), Bphi_max=Bmax,
               F_up=float(F + qerr + tube), F_low=float(F - qerr - tube), ok=True)
    out["polygon"] = C
    return out


# ============================================================================ reversible closing (edge)
# For a stellarator-symmetric source the period map in a symmetry plane is reversible, S P S = P^-1 with
# S(R, Z) = (R, -Z); a point h of an unstable branch u_A(X_k) of a symmetric orbit on the line Z = 0 is then
# an exact heteroclinic point (h = S h lies on S u_A(X_k), a stable branch of S X_k), however small the angle
# between W^u and W^s.  One such point at a gap whose ends are mirror images (X_k, S X_k neighbours along the
# chain), and its images h_j = P^j h, close a curve C_A of manifold arcs around the axis (the construction of
# torux/islandband.py, from which these helpers are adapted).  THEOREM (edge barrier, reversible closing) in the
# module docstring.

def chain_order(orb, a):
    """Cyclic order of the orbit's points along the chain: k p mod q when the rotation number p/q is known
    (torus-angle order), else the polar angle about a.  Returns (order, succ, pred)."""
    q = orb.q
    if orb.p is not None and math.gcd(int(orb.p), q) == 1:
        order = sorted(range(q), key=lambda k: (k * int(orb.p)) % q)
    else:
        order = list(np.argsort(np.arctan2(orb.pts[:, 1] - a[1], orb.pts[:, 0] - a[0])))
    pos = {k: i for i, k in enumerate(order)}
    succ = {k: order[(pos[k] + 1) % q] for k in range(q)}
    pred = {k: order[(pos[k] - 1) % q] for k in range(q)}
    return order, succ, pred


def _rev_seg_errors(orb, c, br, delta0):
    """Per-segment error bounds of a branch polyline (vertex errors + chord errors)."""
    E = vertex_errors(orb, c, br["refs"], delta0) + getattr(c, "const_err", 0.0)
    ch = np.concatenate([[0.0], segment_chord_errors(br["pts"][1:])]) if len(br["pts"]) > 2 else np.zeros(1)
    return np.maximum(E[:-1], E[1:]) + ch[:len(E) - 1]


def _rev_upto_level(refs, Lmax):
    n = 0
    for (L, _) in refs:
        if L is not None and L > Lmax:
            break
        n += 1
    return max(n, 2)


def _rev_nearest_on_polyline(P, x):
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.sum(ab * ab, 1)
    t = np.clip(np.sum((x[None] - a) * ab, 1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - x[None], axis=1)
    i = int(np.argmin(d))
    return i, float(t[i]), float(d[i])


def rev_find_crossing(sec, orb, a, c, sigma, succ, pred, delta0, first_owner=None, want_dir=None, max_level=None):
    """First certified crossing of an unstable branch of curve c with the exact line Z = 0 (Fix S), at an X_k
    whose mirror partner S X_k is a neighbour along the chain (dir +1: the successor, -1: the predecessor).
    The crossing is certified by the parallelogram lemma against the exactly known line; up to four sign
    changes are tried along the branch.  Returns dict(owner, partner, dir, seg, h, box_diam, level, cert) of
    the lowest level, or None."""
    q = orb.q
    owners = range(q) if first_owner is None else [first_owner]
    best = None
    for k in owners:
        if c.level() < (c.d * (k - c.base)) % q:
            continue
        nb = sigma[k]
        if nb == k:
            continue
        br = branch_polyline(orb, c, k)
        P = br["pts"]
        if len(P) < 3:
            continue
        if succ[k] == pred[k]:
            d = int(_branch_dir(orb, a, br)) if nb == succ[k] else 0
        else:
            d = +1 if nb == succ[k] else (-1 if nb == pred[k] else 0)
        if d == 0 or (want_dir is not None and d != want_dir):
            continue
        s_ = np.sign(P[:, 1])
        idx = np.flatnonzero(s_[1:-1] * s_[2:] < 0) + 1
        su = None
        tries = 0
        for i in idx:
            L_, j_ = br["refs"][i + 1]
            if max_level is not None and L_ is not None and L_ > max_level:
                break
            t = P[i, 1] / (P[i, 1] - P[i + 1, 1])
            Rc = P[i, 0] + t * (P[i + 1, 0] - P[i, 0])
            if su is None:
                su = _rev_seg_errors(orb, c, br, delta0)
            Lw = 2.0 * sec.scale
            line = np.array([[Rc - Lw, 0.0], [Rc + Lw, 0.0]])
            cc = certify_crossing(P, su, line, np.zeros(1), int(i), 0)
            rec = dict(owner=int(k), partner=int(nb), dir=int(d), seg=int(i), h=np.array(cc.get("h", [Rc, 0.0])),
                       box_diam=float(cc.get("box_diam", np.inf)), level=int(L_ if L_ is not None else 0),
                       cert={kk: v for kk, v in cc.items() if kk != "h"}, ok=bool(cc["ok"]))
            if cc["ok"]:
                if best is None or rec["level"] < best["level"]:
                    best = rec
                break
            tries += 1
            if tries >= 4:
                break
    return best


def rev_js(q):
    """The image indices j of the q gaps: -(q // 2) .. q - q // 2 - 1 (forward and backward images of h, so
    that the arcs reach at most q / 2 levels beyond the heteroclinic point's)."""
    return list(range(-(q // 2), q - q // 2))


def rev_images(sec, h, box_diam, js, hfd):
    """h_j = P^j(h) for the indices js (negative: backward maps), with 2D error bounds
    e_j = |DP^j| box/2 + 2 sum_i |DP^(j) DP^(i)^-1| tau_i (first order; finite-difference Jacobians of the
    accurate map, i over the steps from h to h_j).  Returns dict j -> (point, error), the per-step taus,
    lost, and the Jacobian norms."""
    E = np.array([[0.0, 0.0], [hfd, 0.0], [-hfd, 0.0], [0.0, hfd], [0.0, -hfd]])
    pts, errs, taus_all, Jn = {0: h.copy()}, {0: 0.5 * box_diam}, {}, {0: 1.0}
    lost = False
    for sgn in (+1, -1):
        jmax = max([sgn * j for j in js] + [0])
        y = h[None] + E
        Js = [np.eye(2)]
        taus = [0.0]
        for n in range(1, jmax + 1):
            y, l, t = sec.step(y, np.full(len(y), float(sgn)))
            lost |= bool(np.any(l))
            Js.append(np.stack([(y[1] - y[2]) / (2 * hfd), (y[3] - y[4]) / (2 * hfd)], -1))
            taus.append(float(t[0]))
            j = sgn * n
            e = float(np.linalg.norm(Js[n], 2)) * 0.5 * box_diam
            e_tr = 0.0
            for i in range(1, n + 1):
                e_tr += float(np.linalg.norm(Js[n] @ np.linalg.inv(Js[i]), 2)) * taus[i]
            pts[j], errs[j], taus_all[j], Jn[j] = y[0].copy(), e + 2.0 * e_tr, float(t[0]), float(np.linalg.norm(Js[n], 2))
    return pts, errs, taus_all, lost, Jn


def rev_assemble(sec, orb, cu, cs, rec, hj, ej, succ, pred, delta0):
    """The closed curve from the unstable curve cu, its mirror cs (stable) and the heteroclinic orbit hj
    (dict j -> point, errors ej) starting in the gap of rec: for each j the arc of cu at X_(owner+j) up to the
    computed point nearest h_j, then the arc of cs at X_(partner+j) from h_j back to that X point.  Returns
    dict(ok, C, E (per-segment errors), junctions) -- or the reason."""
    q = orb.q
    cdir = rec["dir"]
    kA, sA = rec["owner"], rec["partner"]
    junction, gaps = [], []
    for j in rev_js(q):
        uo, so = (kA + j) % q, (sA + j) % q
        if (succ if cdir > 0 else pred)[uo] != so:
            return dict(ok=False, reason=f"image gap {j} ({uo} -> {so}) not between chain neighbours")
        bu = branch_polyline(orb, cu, uo)
        bs = branch_polyline(orb, cs, so)
        if len(bu["pts"]) < 2 or len(bs["pts"]) < 2:
            return dict(ok=False, reason=f"image gap {j}: branch not grown")
        eu = _rev_seg_errors(orb, cu, bu, delta0)
        es = _rev_seg_errors(orb, cs, bs, delta0)
        # search only up to the levels where the images lie (u: level L_h + j; s: the mirror of level L_h - j)
        nu_ = _rev_upto_level(bu["refs"], rec["level"] + j + 1)
        ns_ = _rev_upto_level(bs["refs"], max(rec["level"] - j + 1, 1))
        iu, tu, du = _rev_nearest_on_polyline(bu["pts"][:nu_], hj[j])
        is_, ts, ds = _rev_nearest_on_polyline(bs["pts"][:ns_], hj[j])
        lim_u = eu[iu] + ej[j] + 1e-12 * sec.scale
        lim_s = es[is_] + ej[j] + 1e-12 * sec.scale
        if du > 4.0 * lim_u or ds > 4.0 * lim_s:
            return dict(ok=False, reason=f"heteroclinic image {j} off the computed branches "
                                         f"(d_u {du:.2e} vs {lim_u:.2e}, d_s {ds:.2e} vs {lim_s:.2e})")
        Ju = ej[j] + du + eu[iu]
        Js = ej[j] + ds + es[is_]
        upts = bu["pts"][:iu + 1]
        uerr = np.concatenate([eu[:iu], [max(eu[iu], Ju)]])
        spts = bs["pts"][1:is_ + 1][::-1]
        serr = np.concatenate([[max(es[is_], Js)], es[:is_][::-1]])
        if cdir > 0:
            gaps.append(dict(start=uo, end=so, pts=np.concatenate([upts, hj[j][None], spts]),
                             err=np.concatenate([uerr, serr])))
        else:
            s_fwd = bs["pts"][:is_ + 1]
            s_err = np.concatenate([es[:is_], [max(es[is_], Js)]])
            u_rev = bu["pts"][1:iu + 1][::-1]
            u_err = np.concatenate([[max(eu[iu], Ju)], eu[:iu][::-1]])
            gaps.append(dict(start=so, end=uo, pts=np.concatenate([s_fwd, hj[j][None], u_rev]),
                             err=np.concatenate([s_err, u_err])))
        junction.append(dict(gap=(int(uo), int(so)), j=int(j), e_image=float(ej[j]), d_u=du, d_s=ds, J_u=Ju, J_s=Js))
    by_start = {g["start"]: g for g in gaps}
    if len(by_start) != q:
        return dict(ok=False, reason="gaps do not start at distinct X points")
    C, E = [], []
    k0 = gaps[0]["start"]
    k = k0
    for _ in range(q):
        g = by_start[k]
        C.append(g["pts"])
        E.append(g["err"])
        k = g["end"]
    if k != k0:
        return dict(ok=False, reason="gaps do not close up")
    C = np.concatenate(C)
    E = np.concatenate(E)
    return dict(ok=True, C=C, E=E, junctions=junction)


def decimate_closed(C, E, rel=0.25, abs_tol=None, corners=None):
    """Drop vertices of the closed polygon C (segment k: C[k] -> C[k+1 mod n], error E[k]) while every dropped
    vertex lies within rel x (the smallest error of the merged segments) of the new chord; the merged segment's
    error is max(merged errors) + max(vertex distance to the chord) (the old polyline lies within that distance
    of the chord, so the true curve stays within the new error along matching parameterisations).  Vertices
    within abs_tol (default a quarter of the 90th percentile of the errors) are always dropped, except near
    `corners` (the X points), where the tolerance is at most a tenth of the distance to the nearest corner.
    (torux/islandband.py.)"""
    C = np.asarray(C, float)
    E = np.asarray(E, float)
    n = len(C)
    if abs_tol is None:
        abs_tol = 0.25 * float(np.quantile(E, 0.9))
    capX = np.full(n, np.inf)
    if corners is not None and len(corners):
        K = np.asarray(corners, float)
        for lo_ in range(0, n, 4096):
            capX[lo_:lo_ + 4096] = 0.1 * np.min(np.linalg.norm(C[lo_:lo_ + 4096, None, :] - K[None], axis=-1), axis=1)
    capX = np.concatenate([capX, capX[:1]])
    P = np.concatenate([C, C[:1]])
    keep = [0]
    newE = []
    i = 0
    while i < n:
        if len(keep) % 256 == 0:
            _hard_check()

        def ok(j):
            if j - i < 2:
                return True, 0.0
            d = _pt_seg_dist(P[i + 1:j], P[i][None], P[j][None])
            m = float(np.max(d))
            return m <= max(rel * float(np.min(E[i:j])), min(abs_tol, float(np.min(capX[i + 1:j])))), m
        step = 2
        best_j, best_d = i + 1, 0.0
        while i + step <= n:
            good, m = ok(i + step)
            if not good:
                break
            best_j, best_d = i + step, m
            step *= 2
        lo, hi = best_j, min(i + step, n + 1)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            good, m = ok(mid)
            if good:
                lo, best_d = mid, m
            else:
                hi = mid
        j = lo
        newE.append(float(np.max(E[i:j])) + best_d)
        if j < n:
            keep.append(j)
        i = j
    return C[keep], np.array(newE)


def certify_closed_curve(sec, C, E, a, a_err=0.0, corners=None):
    """Simplicity (loop cuts certified), winding 1 around a beyond the tube, flux (loop integral of A with its
    quadrature error), B_phi > 0 inside, the tube term.  Returns (record, polygon or None)."""
    out = dict(ok=False)
    tol = 1e-12 * sec.scale
    n0 = len(C)
    C, E = decimate_closed(C, E, corners=corners)
    C, E = _dedupe_closed(C, E, tol)
    out["vertices_before_decimation"] = int(n0)
    w = winding_number(C, a)
    out["winding"] = w
    if w == -1:
        C = C[::-1]
        E = np.concatenate([E[:-1][::-1], E[-1:]])
        w = 1
    if w != 1:
        out["reason"] = f"winding number around the axis point {w}"
        return out, None
    simple = closed_polygon_simple(C, tol=tol)
    out["simple_as_assembled"] = bool(simple)
    if not simple:
        C2, E2, info = simplify_closed_curve(C, E, a, tol)
        out["loop_cuts"] = {k: v for k, v in info.items() if k != "boxes"}
        if C2 is None:
            out["reason"] = "closed curve not simple: " + info.get("reason", "")
            return out, None
        if winding_number(C2, a) == -1:
            C2 = C2[::-1]
            E2 = np.concatenate([E2[:-1][::-1], E2[-1:]])
        if not closed_polygon_simple(C2, tol=tol) or winding_number(C2, a) != 1:
            out["reason"] = "closed curve not simple after loop cuts"
            return out, None
        C, E = C2, E2
    out["eps_max"] = float(np.max(E))
    out["eps_median"] = float(np.median(E))
    dmin_a = float(polyline_distance(np.atleast_2d(a), np.concatenate([C, C[:1]]))[0])
    out["dist_axis"] = dmin_a
    if dmin_a <= out["eps_max"] + a_err:
        out["reason"] = "axis point within the curve's error"
        return out, None
    out["simple"] = True
    F, qerr = sec.flux(C)
    dens = sec.density_check(C)
    out["density_check"] = dens
    if not dens["ok"]:
        out["reason"] = "B_phi > 0 not established inside the curve"
        return out, None
    L = np.linalg.norm(np.diff(np.concatenate([C, C[:1]]), axis=0), axis=1)
    Bmax = sec.density_max(C)
    tube = float(np.sum((2.0 * E * L + np.pi * E ** 2)) * Bmax)
    out.update(flux=float(F), quad_err=float(qerr), tube=tube, perimeter=float(np.sum(L)), Bphi_max=Bmax,
               F_up=float(F + qerr + tube), F_low=float(F - qerr - tube), n_vertices=int(len(C)), ok=True)
    return out, C


def rev_close(sec, orb, a, c, rec, sigma, succ, pred, delta0, mirror_base, defect_step, target, request=None,
              a_err=0.0, verbose=False):
    """The reversible closed curve of unstable generating curve c from its symmetric heteroclinic point rec
    (rev_find_crossing): heteroclinic images, assembly, the uncertified flux (refused when it already holds
    more than the request), chord validation of every arc used, the crossing and images again, the
    certification (certify_closed_curve) and the reference re-trace of the heteroclinic orbit (single steps).
    Returns a JSON-friendly record (ok, F_up, ..., polygon as an array)."""
    q = orb.q
    hfd = 1e-7 * sec.scale
    out = dict(curve=c.name, kind="reversible", ok=False, crossing=dict(owner=rec["owner"], partner=rec["partner"],
                                                                       level=rec["level"], dir=rec["dir"]))

    def build(rec_):
        hj, ej, taus, lost, Jn = rev_images(sec, rec_["h"], rec_["box_diam"], rev_js(q), hfd)
        if lost:
            return None, "heteroclinic orbit lost", None
        cs = mirror_curve(c, "S" + c.name, mirror_base, 0.0)
        cs.const_err = defect_step
        asm = rev_assemble(sec, orb, c, cs, rec_, hj, ej, succ, pred, delta0)
        if not asm["ok"]:
            return None, asm["reason"], None
        return asm, None, (hj, ej, taus, Jn)

    asm, why, img = build(rec)
    if asm is None:
        out["reason"] = why
        return out
    try:
        F_rough, _ = sec.flux(asm["C"])
        L_rough = float(np.sum(np.linalg.norm(np.diff(np.concatenate([asm["C"], asm["C"][:1]]), axis=0), axis=1)))
        B_rough = sec.density_max(asm["C"][::max(1, len(asm["C"]) // 200)])
    except BarrierDeadline:
        raise
    except Exception:  # noqa: BLE001
        F_rough, L_rough, B_rough = np.nan, np.nan, np.nan
    out["flux_estimate"] = float(F_rough)
    if request is not None and np.isfinite(F_rough):
        if abs(F_rough) >= abs(request):
            out["reason"] = f"closed curve encloses more than the request (estimate {F_rough:.6g})"
            return out
        target = float(np.clip(0.02 * (abs(request) - abs(F_rough)) / (2 * L_rough * B_rough), target, 10 * target))
    out["chord_target"] = target
    # chord validation of the arcs used: every owner's branch up to its image point, and the mirrored arcs
    # (the branch at sigma(partner + j) up to the mirror of h_j)
    hj = img[0]
    segs = []
    for j in rev_js(q):
        uo = (rec["owner"] + j) % q
        br = branch_polyline(orb, c, uo)
        iu, _, _ = _rev_nearest_on_polyline(br["pts"], hj[j])
        segs += branch_segments(br, 1, min(iu + 4, len(br["refs"]) - 1))
        so = (rec["partner"] + j) % q
        br2 = branch_polyline(orb, c, sigma[so])
        is_, _, _ = _rev_nearest_on_polyline(br2["pts"], hj[j] * np.array([1.0, -1.0]))
        segs += branch_segments(br2, 1, min(is_ + 4, len(br2["refs"]) - 1))
    st = refine_arc_segments(sec, c, sorted(set(segs)), target)
    out["chord_validation"] = st
    if not st["converged"]:
        out["reason"] = "chord validation did not converge"
        return out
    rec2 = rev_find_crossing(sec, orb, a, c, sigma, succ, pred, delta0, first_owner=rec["owner"], want_dir=rec["dir"])
    if rec2 is None:
        out["reason"] = "symmetric crossing lost after chord validation"
        return out
    asm, why, img = build(rec2)
    if asm is None:
        out["reason"] = why
        return out
    hj, ej, taus, Jn = img
    cert, Cfin = certify_closed_curve(sec, asm["C"], asm["E"], a, a_err, corners=orb.pts)
    out.update({k: v for k, v in cert.items()})
    out["crossing"] = dict(owner=rec2["owner"], partner=rec2["partner"], level=rec2["level"], dir=rec2["dir"],
                           h=rec2["h"].tolist(), box_diam=rec2["box_diam"], angle=rec2["cert"].get("angle"),
                           eps_u=rec2["cert"].get("eps_u"))
    out["image_errors"] = {int(j): float(v) for j, v in ej.items()}
    out["junction_err_max"] = float(max(max(jn["J_u"], jn["J_s"]) for jn in asm["junctions"]))
    if not cert["ok"]:
        return out
    # the heteroclinic orbit's single steps against the reference tracer (the per-step tau of its error model)
    if hasattr(sec, "reference_steps") and getattr(sec, "use_reference", False) and q > 1:
        try:
            jf = sorted(j for j in hj if j > 0)
            jb = sorted((j for j in hj if j < 0), reverse=True)
            okh, dmax_, tmax_, emax_ = True, 0.0, 0.0, 0.0
            for seq, dd in ((jf, +1), (jb, -1)):
                if not seq:
                    continue
                src = np.array([hj[j - dd] for j in seq])
                r = sec.reference_steps(src, dd)
                if r is None:
                    continue
                yr, er = r
                d = np.linalg.norm(yr - np.array([hj[j] for j in seq]), axis=1)
                tt = np.array([taus[j] for j in seq])
                okh = okh and bool(np.all(d <= tt + 10 * er + 1e-13 * sec.scale))
                dmax_, tmax_, emax_ = max(dmax_, float(np.max(d))), max(tmax_, float(np.max(tt))), max(emax_, float(np.max(er)))
            if True:
                out["heteroclinic_reference"] = dict(ok=okh, max_diff=dmax_, max_tau=tmax_, max_ref_err=emax_)
                if not okh:
                    out["ok"] = False
                    out["reason"] = "heteroclinic orbit: reference single steps beyond the error model"
        except BarrierDeadline:
            raise
        except Exception as e:  # noqa: BLE001
            out["heteroclinic_reference"] = dict(ok=False, error=repr(e))
            out["ok"] = False
            out["reason"] = "heteroclinic orbit reference check failed"
    out["polygon"] = Cfin
    return out


def lemma_B_general(names, excluded, closed, reflection):
    """Generalised Lemma B (module docstring, THEOREM (edge barrier, reversible closing)): `names` the generating
    curves (u+, u-, Su+, Su- or u+, u-, s+, s-), `excluded` those with a certified lost point or a certified
    box crossing with a branch of the other kind, `closed` {family: F} the unstable curves c whose reversible
    closed curve C_c is certified (family of c: c and its mirror Sc).  A covered torus through X contains
    branches of two distinct non-excluded generating curves; if one of them belongs to a closed family c, the
    torus is C_c.  Returns (holds, nonexcluded, free, F_ii): holds iff at most one non-excluded generating
    curve (none if reflection-hyperbolic) lies outside the closed families; F_ii the flux bound such tori need
    (max over closed families with a non-excluded member; -inf when no torus can pass through X).  Within a
    closed family the exclusion of one member excludes the other (a torus containing a branch of c or of S c is
    C_c, which contains branches of both)."""
    fam = (lambda g: g[1:] if g.startswith("S") else g)
    # in a closed family the exclusion of one member excludes the other: a torus containing a branch of c or of
    # S c is C_c and so contains branches of both
    excluded = set(excluded) | {g for g in names if fam(g) in closed and any(fam(e) == fam(g) for e in excluded)}
    nonex = [g for g in names if g not in excluded]
    free = [g for g in nonex if fam(g) not in closed]
    holds = len(free) <= (0 if reflection else 1)
    F_ii = -np.inf
    if len(nonex) >= (1 if reflection else 2):
        vals = [closed[fam(g)] for g in nonex if fam(g) in closed]
        if vals:
            F_ii = float(max(vals))
    return holds, nonex, free, F_ii


# ============================================================================ lost shooting
def reshoot(sec, cand, nmax, h_t, deadline=None):
    """Repeat a detected lost orbit with the accurate map (and its error model) from the same start
    point, until lost (the orbit may slide along the manifold relative to the detection run)."""
    z, t = np.array(cand["start"][0]), np.array(cand["start"][1])
    d = cand["d"]
    e, amp = 0.0, 1.0
    for n in range(nmax):
        if deadline is not None and time.time() > deadline:
            return None
        Y = np.stack([z, t])
        img, lost, tau = sec.step(Y, np.full(2, d))
        if lost[0]:
            return dict(point=z.copy(), e=float(e), steps=n, d=d, amp=float(amp), start=cand["start"])
        dens = sec.density(np.stack([z, img[0]]))
        if lost[1]:
            t = img[0] + 0.1 * h_t * (t - z) / max(np.linalg.norm(t - z), 1e-300)
            z = img[0]
            e = e + tau[0]
            continue
        dv = img[1] - img[0]
        stretch = np.linalg.norm(dv) / max(np.linalg.norm(t - z), 1e-300)
        bt = abs(dens[0] / dens[1]) / max(stretch, 1e-300)
        e = bt * e + tau[0]
        amp = bt * amp
        z = img[0]
        t = img[0] + h_t * dv / max(np.linalg.norm(dv), 1e-300)
    return None


def shoot_lost_many(sec, jobs, s_frac=(0.3, 0.8), deadline=None, h_t=None, track=True, return_lines=False):
    """shoot_lost for several orbits in one batch (one tracer call per period for all lines).
    jobs: list of dict(orb, specs, s0, w, mult, nmax).  Returns a list of {name: candidate}."""
    h_t = h_t or 1e-7 * sec.scale
    lines = []
    for jid, jb in enumerate(jobs):
        x0 = jb["orb"].pts[0]
        for (name, kind, v, sgn) in jb["specs"]:
            ww = np.asarray(jb["w"][kind])
            for f in s_frac:
                s_ = jb["s0"][kind] * jb["mult"][kind] ** f
                z = x0 + sgn * s_ * v + s_ * s_ * ww
                tg = sgn * v + 2 * s_ * ww
                tg = tg / np.linalg.norm(tg)
                lines.append(dict(job=jid, name=name, d=+1 if kind == "u" else -1, z=z, t=z + h_t * tg, e=0.0,
                                  amp=1.0, n=0, done=False, lost=None, nmax=jb["nmax"],
                                  start=(z.copy(), (z + h_t * tg).copy())))
    _shoot_lines(sec, lines, h_t, deadline, track=track)
    res = [dict() for _ in jobs]
    for l in lines:
        if l["lost"] is None:
            continue
        b = res[l["job"]].get(l["name"])
        if b is None or l["lost"]["steps"] < b["steps"]:
            res[l["job"]][l["name"]] = l["lost"]
    return (res, lines) if return_lines else res


def _shoot_lines(sec, lines, h_t, deadline, track=True, first_only=True):
    """Advance shooting lines one period at a time until lost (or nmax).  track: follow the tangent
    companion and the normal error (the accurate pass); detection passes skip it."""
    n = 0
    while True:
        if deadline is not None and time.time() > deadline:
            break
        act = [l for l in lines if not l["done"] and n < l.get("nmax", 10 ** 9)]
        if not act:
            break
        if not track:
            Y = np.stack([l["z"] for l in act])
            img, lost, _ = sec.step(Y, np.array([l["d"] for l in act], float))
            for i, l in enumerate(act):
                if lost[i]:
                    l["done"] = True
                    l["lost"] = dict(point=l["z"].copy(), e=np.nan, steps=l.get("nd", 0), d=l["d"], amp=np.nan,
                                     start=[l["start"][0].tolist(), l["start"][1].tolist()])
                else:
                    l["z"] = img[i]
                    l["nd"] = l.get("nd", 0) + 1            # periods completed (detection lines)
            if first_only:
                # detection: the first lost line of a branch is the candidate (fewest periods); its siblings stop
                gone = {(l.get("job"), l.get("name")) for l in act if l["lost"] is not None}
                for l in act:
                    if not l["done"] and (l.get("job"), l.get("name")) in gone:
                        l["done"] = True
            n += 1
            continue
        Y = np.concatenate([np.stack([l["z"], l["t"]]) for l in act])
        D = np.repeat([l["d"] for l in act], 2)
        img, lost, tau = sec.step(Y, D)
        dens = sec.density(np.concatenate([Y[0::2], img[0::2]]))
        m = len(act)
        for i, l in enumerate(act):
            if lost[2 * i]:
                l["done"] = True
                l["lost"] = dict(point=l["z"].copy(), e=float(l["e"]), steps=n, d=l["d"], amp=float(l["amp"]),
                                 start=[l["start"][0].tolist(), l["start"][1].tolist()])
                continue
            if lost[2 * i + 1]:
                l["t"] = img[2 * i] + 0.1 * h_t * (Y[2 * i + 1] - Y[2 * i]) / max(np.linalg.norm(Y[2 * i + 1] - Y[2 * i]), 1e-300)
                l["z"] = img[2 * i]
                l["e"] = l["e"] + tau[2 * i]
                continue
            dv = img[2 * i + 1] - img[2 * i]
            stretch = np.linalg.norm(dv) / max(np.linalg.norm(Y[2 * i + 1] - Y[2 * i]), 1e-300)
            det = abs(dens[i] / dens[m + i])
            bt = det / max(stretch, 1e-300)
            l["e"] = bt * l["e"] + tau[2 * i]
            l["amp"] = bt * l["amp"]
            l["z"] = img[2 * i]
            l["t"] = img[2 * i] + h_t * dv / max(np.linalg.norm(dv), 1e-300)
            l["n"] = n + 1
        if first_only and any("job" in l for l in act):
            gone = {(l.get("job"), l.get("name")) for l in act if l["lost"] is not None and "job" in l}
            for l in act:
                if not l["done"] and "job" in l and (l.get("job"), l.get("name")) in gone:
                    l["done"] = True
        n += 1


def shoot_lost(sec, orb, gens_spec, s0, delta0=0.0, s_frac=(0.13, 0.41, 0.77), nmax=600, deadline=None, h_t=None,
               w=None, mult=None):
    """Orbits from the fundamental domains along each generating curve, followed until lost, with
    the normal error along the orbit (renormalised tangent companion).  gens_spec: list of
    (name, kind, v, sign).  Returns {name: candidate dict or None}."""
    eig = orb.eig()
    lam = abs(eig["lam"])
    mult = mult if mult is not None else lam ** (2 if eig["reflection"] else 1)
    x0 = orb.pts[0]
    h_t = h_t or 1e-7 * sec.scale
    lines = []
    for (name, kind, v, sgn) in gens_spec:
        ww = np.zeros(2) if w is None else np.asarray(w.get(kind, np.zeros(2)) if isinstance(w, dict) else w)
        s0k = s0[kind] if isinstance(s0, dict) else s0
        mk = mult[kind] if isinstance(mult, dict) else mult
        for f in s_frac:
            s = s0k * mk ** f
            z = x0 + sgn * s * v + s * s * ww
            tg = sgn * v + 2 * s * ww
            tg = tg / np.linalg.norm(tg)
            lines.append(dict(name=name, d=+1 if kind == "u" else -1, z=z, t=z + h_t * tg, e=0.0, amp=1.0,
                              n=0, done=False, lost=None, start=(z.copy(), (z + h_t * tg).copy())))
    for n in range(nmax):
        if deadline is not None and time.time() > deadline:
            break
        act = [l for l in lines if not l["done"]]
        if not act:
            break
        Y = np.concatenate([np.stack([l["z"], l["t"]]) for l in act])
        D = np.repeat([l["d"] for l in act], 2)
        img, lost, tau = sec.step(Y, D)
        dens = sec.density(np.concatenate([Y[0::2], img[0::2]]))
        m = len(act)
        for i, l in enumerate(act):
            if lost[2 * i]:
                l["done"] = True
                l["lost"] = dict(point=l["z"].copy(), e=float(l["e"]), steps=n, d=l["d"], amp=float(l["amp"]),
                                 start=[l["start"][0].tolist(), l["start"][1].tolist()])
                continue
            if lost[2 * i + 1]:
                # companion lost first: shrink it
                l["t"] = img[2 * i] + 0.1 * h_t * (Y[2 * i + 1] - Y[2 * i]) / max(np.linalg.norm(Y[2 * i + 1] - Y[2 * i]), 1e-300)
                l["z"] = img[2 * i]
                l["e"] = l["e"] + tau[2 * i]
                continue
            dv = img[2 * i + 1] - img[2 * i]
            stretch = np.linalg.norm(dv) / max(np.linalg.norm(Y[2 * i + 1] - Y[2 * i]), 1e-300)
            det = abs(dens[i] / dens[m + i])
            bt = det / max(stretch, 1e-300)
            l["e"] = bt * l["e"] + tau[2 * i]
            l["amp"] = bt * l["amp"]
            l["z"] = img[2 * i]
            l["t"] = img[2 * i] + h_t * dv / max(np.linalg.norm(dv), 1e-300)
            l["n"] = n + 1
    best = {}
    for l in lines:
        if l["lost"] is None:
            continue
        b = best.get(l["name"])
        if b is None or l["lost"]["steps"] < b["steps"]:
            best[l["name"]] = l["lost"]
    return best


# ============================================================================ the core
def mirror_info(sec, orb, scale, use_symmetry=True):
    """Stellarator symmetry: W^s(S X) = S W^u(X) (P^-1 = S P S) when the field and the orbit are
    symmetric.  Returns the base index of S X_0, the measured symmetry defect of the reference field and
    the per-period error added to mirrored curves, or None."""
    if not (use_symmetry and hasattr(sec, "symmetry_defect") and getattr(sec.source, "stellsym", False)):
        return None
    Sx = orb.pts * np.array([1.0, -1.0])
    dmat = np.linalg.norm(Sx[:, None, :] - orb.pts[None, :, :], axis=-1)
    if not np.all(np.min(dmat, axis=1) < 10.0 * orb.e_X + 1e-9 * scale):   # symmetric within its error bound
        return None
    b = int(np.argmin(dmat[0]))
    x3 = np.stack([orb.pts[:, 0] * np.cos(0.4), orb.pts[:, 0] * np.sin(0.4), orb.pts[:, 1]], -1)
    dfx = sec.symmetry_defect(x3)
    Lper = float(np.max(orb.pts[:, 0])) * 2 * np.pi / sec.nfp
    mism = float(np.max(np.min(dmat, axis=1)))       # |S X_k - X_sigma(k)|: orbit symmetric to this
    return dict(base=b, defect=dfx, mismatch=mism, step=10.0 * dfx * Lper + 1e-15 * scale + mism)


def screen_orbit(sec, orb, deadline=None, n0=16, shoot_extra=30, shoot_cap=3000, use_symmetry=True,
                 s_frac=(0.3, 0.8), detect=None):
    """The cheap part of a candidate: accurate orbit (e_X), local manifolds with measured start errors,
    the mirror decision, and lost-point candidates (detected with the coarse map, the accurate map and
    error model re-run from the same starts).  Returns dict (or dict(ok=False, reason))."""
    t0 = time.time()
    orb2 = orbit_from_point(sec, orb.pts[0], orb.q)
    if orb2 is None:
        return dict(ok=False, reason="orbit lost")
    orb2.p = orb.p
    orb2.info.update({k: v for k, v in orb.info.items() if k not in orb2.info})
    orb = orb2
    eig = orb.eig()
    lam = abs(eig["lam"])
    if not (lam > 1.0 + 1e-9 and abs(eig["mu"]) < 1.0):
        return dict(ok=False, reason="orbit not hyperbolic", orb=orb)
    scale = sec.scale
    mirror = mirror_info(sec, orb, scale, use_symmetry)
    loc = {"u": local_manifold(sec, orb, "u", n0=n0, deadline=deadline)}
    if not mirror:
        loc["s"] = local_manifold(sec, orb, "s", n0=n0, deadline=deadline)
    specs = [("u+", "u", eig["vu"], +1)]
    if not eig["reflection"]:
        specs += [("u-", "u", eig["vu"], -1)]
    if not mirror:
        specs += [("s+", "s", eig["vs"], +1)] + ([] if eig["reflection"] else [("s-", "s", eig["vs"], -1)])
    nshoot = int(min(shoot_cap, orb.q * (2 if eig["reflection"] else 1) *
                     (math.log(scale / loc["u"]["s0"]) / math.log(loc["u"]["Lambda"]) + shoot_extra)))
    det = detect if detect is not None else (sec.coarse() if hasattr(sec, "coarse") else sec)
    s0d = {k: v["s0"] for k, v in loc.items()}
    wd = {k: v["w"] for k, v in loc.items()}
    md = {k: v["Lambda"] for k, v in loc.items()}
    lost_c = shoot_lost(det, orb, specs, s0d, nmax=nshoot, deadline=deadline, w=wd, mult=md, s_frac=s_frac)
    if det is not sec:                              # certified errors come from the accurate map only
        acc = {}
        for name, cand in lost_c.items():
            rc = reshoot(sec, cand, cand["steps"] + 40 * orb.q, 1e-7 * sec.scale, deadline=deadline)
            if rc is not None:
                acc[name] = rc
        lost_c = acc
    return dict(ok=True, orb=orb, eig=eig, mirror=mirror, loc=loc, specs=specs, nshoot=nshoot, lost_c=lost_c,
                time=time.time() - t0)


def orbits_many(sec, orbs, polish=3, h_fd=None):
    """orbit_from_point for several orbits of any periods at once (one tracer call per period for all
    lines): Newton polish on P^q, points, closure, monodromy (central differences), e_X."""
    h = h_fd or 1e-7 * sec.scale
    E = np.array([[0.0, 0.0], [h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    X = np.array([o.pts[0] for o in orbs], float)
    qs = np.array([o.q for o in orbs], int)
    best = [None] * len(orbs)
    alive = np.ones(len(orbs), bool)
    for it in range(polish + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        Y = (X[idx][:, None, :] + E[None]).reshape(-1, 2)
        nper = np.repeat(qs[idx], 5)
        pts = [[X[i].copy()] for i in idx]
        tau = np.zeros(len(idx))
        lost = np.zeros(len(idx), bool)
        for k in range(int(np.max(nper))):
            act = np.flatnonzero(nper > k)
            y, l, t = sec.step(Y[act], np.ones(len(act)))
            Y[act] = y
            lo = np.zeros(len(Y), bool)
            lo[act] = l
            lost |= lo.reshape(-1, 5).any(1)
            tt = np.zeros(len(Y))
            tt[act] = t
            tau += tt.reshape(-1, 5)[:, 0]
            for m, i in enumerate(idx):
                if k < qs[i] - 1:
                    pts[m].append(Y[5 * m].copy())
        Yr = Y.reshape(-1, 5, 2)
        for m, i in enumerate(idx):
            if lost[m]:
                alive[i] = False
                continue
            M = np.stack([(Yr[m, 1] - Yr[m, 2]) / (2 * h), (Yr[m, 3] - Yr[m, 4]) / (2 * h)], -1)
            F = Yr[m, 0] - X[i]
            clos = float(np.linalg.norm(F))
            if best[i] is not None and clos >= best[i].info["closure"]:
                alive[i] = False
                continue
            try:
                nrm = float(np.linalg.norm(np.linalg.inv(M - np.eye(2)), 2))
            except np.linalg.LinAlgError:
                nrm = np.inf
            info = dict(orbs[i].info)
            info.update(closure=clos, tau_q=float(tau[m]), inv_norm=nrm, det=float(np.linalg.det(M)))
            best[i] = Orbit(int(qs[i]), np.array(pts[m]), M, e_X=2.0 * nrm * (clos + tau[m]), p=orbs[i].p, info=info)
            if it == polish or clos < 1e-14 * sec.scale or not np.isfinite(nrm):
                alive[i] = False
                continue
            X[i] = X[i] - np.linalg.solve(M - np.eye(2), F)
    return best


def local_manifolds_many(sec, entries, n0=16, target=None, n_def=6, max_tries=3, loose=1e-6, provisional=False):
    """local_manifold for several (orbit, kind) entries at once (one tracer call per period).  provisional=True:
    only the quadratic term w (3 lines per entry) and the provisional domain s0 = 3e-5 scale / Lambda, no measured
    start error (ok False, provisional True): enough to detect lost manifold orbits (screen_many); the measured
    domain is computed for the candidates actually tried (finalize_screened)."""
    scale = sec.scale
    target = target if target is not None else 1e-8 * scale
    res = []
    for orb, kind in entries:
        eig = orb.eig()
        refl = eig["reflection"]
        nper = orb.q * (2 if refl else 1)
        M = orb.M @ orb.M if refl else orb.M
        lu, ls = abs(eig["lam"]) ** (2 if refl else 1), abs(eig["mu"]) ** (2 if refl else 1)
        if kind == "u":
            Lam, mu, v, d, DF = lu, ls, eig["vu"], +1, M
        else:
            Lam, mu, v, d, DF = 1.0 / ls, 1.0 / lu, eig["vs"], -1, np.linalg.inv(M)
        res.append(dict(kind=kind, v=v, Lambda=Lam, mu=mu, nper=nper, d=d, DF=DF, X=orb.pts[0], s0=3e-4 * scale / Lam,
                        ok=False))
    h = 3e-4 * scale
    Y = np.concatenate([np.stack([r["X"] + h * r["v"], r["X"] - h * r["v"], r["X"]]) for r in res])
    img, lost, _ = multi_period(sec, Y, np.repeat([r["d"] for r in res], 3), np.repeat([r["nper"] for r in res], 3))
    for k, r in enumerate(res):
        if np.any(lost[3 * k:3 * k + 3]):
            r["w"] = np.zeros(2)
        else:
            D2 = (img[3 * k] + img[3 * k + 1] - 2 * img[3 * k + 2]) / h ** 2
            r["w"] = np.linalg.solve(r["Lambda"] ** 2 * np.eye(2) - r["DF"], 0.5 * D2)
    if provisional:
        for r in res:
            r.update(s0=0.1 * r["s0"], provisional=True, delta0=np.nan, defect_max=np.nan, tries=0)
            r.pop("DF", None)
            r.pop("X", None)
        return res
    # one pass: the defect at n_def parameters per side for several trial s0 (geometric ladder); the
    # largest s0 meeting the target is kept
    # (three rungs per pass: the local-manifold stage was the screening's largest cost -- 422 of 686 s on QUASR
    # 23786 with five -- its lines follow the orbits for q periods, re-traced next to coils)
    ladder = np.array([1.0, 0.1, 0.01])
    for tries in range(max_tries):
        todo = [r for r in res if not r["ok"]]
        if not todo:
            break
        Ys, Ds, Ns, meta = [], [], [], []
        for r in todo:
            for f in ladder:
                si = r["s0"] * f * r["Lambda"] ** (np.arange(n_def) / (n_def - 1))
                for sg in (+1, -1):
                    Ys.append(r["X"][None] + sg * si[:, None] * r["v"][None] + (si ** 2)[:, None] * r["w"][None])
                    Ds.append(np.full(n_def, r["d"]))
                    Ns.append(np.full(n_def, r["nper"]))
                meta.append((r, f, si))
        img, lost, _ = multi_period(sec, np.concatenate(Ys), np.concatenate(Ds), np.concatenate(Ns))
        o = 0
        trial = {}
        for (r, f, si) in meta:
            n = 2 * n_def
            im, lo = img[o:o + n], lost[o:o + n]
            o += n
            if np.any(lo):
                dmax = np.inf
            else:
                dd = [_closest_on_parabola(r["X"], r["v"], r["w"], (+1 if k < n_def else -1), im[k],
                                           r["Lambda"] * si[k % n_def]) for k in range(n)]
                dmax = float(np.max(dd))
            d0_ = 2.0 * dmax / max(r["Lambda"] - r["mu"], 1e-12)
            trial.setdefault(id(r), []).append((f, dmax, d0_))
            r.setdefault("_trials", []).append((f, dmax, d0_, d0_, r["s0"] * f))
        for r in todo:
            tr_ = trial[id(r)]
            good = [t for t in tr_ if t[2] <= target]
            if good:
                f, dmax, delta0 = max(good, key=lambda t: t[0])
                r.update(s0=r["s0"] * f, defect_max=dmax, delta0=delta0, tries=tries + 1, ok=True)
            else:
                f, dmax, delta0 = tr_[-1]
                r.update(defect_max=dmax, delta0=delta0, tries=tries + 1)
                r["s0"] *= float(ladder[-1]) * 0.1
    for r in res:
        if not r["ok"] and loose is not None:
            # the target was not met (the defect's floor: tracer noise amplified along the orbit): the smallest
            # measured start error is still a bound, carried into every vertex error (amp x delta0, amp shrinking
            # along an unstable branch) -- accepted up to loose x scale
            tr_ = r.get("_trials") or []
            good = [t for t in tr_ if np.isfinite(t[3]) and t[3] <= loose * scale]
            if good:
                f, dmax, delta0, _, s0_ = max(good, key=lambda t: t[4])
                r.update(s0=s0_, defect_max=dmax, delta0=delta0, ok=True, loose=True)
        r.pop("_trials", None)
        r.pop("DF", None)
        r.pop("X", None)
    return res


def screen_many(sec, orbs, deadline=None, n0=16, shoot_extra=10, shoot_cap=2000, use_symmetry=True,
                s_frac=(0.3, 0.8), verbose=False, detect_share=0.4, lazy=True):
    """screen_orbit for several candidate orbits: accurate orbits and local manifolds one by one, the
    shooting (coarse detection) for all candidates in one batch, accurate re-shoots of the lost ones.  The
    detection lines are kept (p_["_det"]) so that extend_shooting can continue the branches not lost."""
    pre = []
    t0 = time.time()
    tm = {}
    accs = orbits_many(sec, list(orbs), polish=2) if orbs else []
    tm["orbits"] = time.time() - t0
    staged = []
    for orb, orb2 in zip(orbs, accs):
        if orb2 is None:
            pre.append(dict(ok=False, reason="orbit lost"))
            continue
        orb2.p = orb.p
        orb2.info.update({k: v for k, v in orb.info.items() if k not in orb2.info})
        eig = orb2.eig()
        if not (abs(eig["lam"]) > 1.0 + 1e-9 and abs(eig["mu"]) < 1.0):
            pre.append(dict(ok=False, reason="orbit not hyperbolic", orb=orb2))
            continue
        mirror = mirror_info(sec, orb2, sec.scale, use_symmetry)
        staged.append((orb2, eig, mirror))
    entries = []
    for (orb2, eig, mirror) in staged:
        entries.append((orb2, "u"))
        if not mirror:
            entries.append((orb2, "s"))
    lms = local_manifolds_many(sec, entries, n0=n0, provisional=lazy) if entries else []
    tm["local_manifolds"] = time.time() - t0 - tm["orbits"]
    k_ = 0
    for (orb2, eig, mirror) in staged:
        loc = {"u": lms[k_]}
        k_ += 1
        if not mirror:
            loc["s"] = lms[k_]
            k_ += 1
        specs = [("u+", "u", eig["vu"], +1)] + ([] if eig["reflection"] else [("u-", "u", eig["vu"], -1)])
        if not mirror:
            specs += [("s+", "s", eig["vs"], +1)] + ([] if eig["reflection"] else [("s-", "s", eig["vs"], -1)])
        nshoot = int(min(shoot_cap, orb2.q * (2 if eig["reflection"] else 1) *
                         (math.log(sec.scale / loc["u"]["s0"]) / math.log(loc["u"]["Lambda"]) + shoot_extra)))
        pre.append(dict(ok=True, orb=orb2, eig=eig, mirror=mirror, loc=loc, specs=specs, nshoot=nshoot,
                        time=(time.time() - t0) / max(len(staged), 1)))
    oks = [p_ for p_ in pre if p_["ok"]]
    jobs = [dict(orb=p_["orb"], specs=p_["specs"], s0={k: v["s0"] for k, v in p_["loc"].items()},
                 w={k: v["w"] for k, v in p_["loc"].items()}, mult={k: v["Lambda"] for k, v in p_["loc"].items()},
                 nmax=p_["nshoot"]) for p_ in oks]
    t1 = time.time()
    det = sec.coarse() if hasattr(sec, "coarse") else sec
    # the detection may take at most detect_share of what is left (branches that are never lost run to their
    # limit; extend_shooting continues them later if nothing certifies)
    dl_d = deadline if deadline is None else time.time() + detect_share * max(deadline - time.time(), 0.0)
    found, lines = shoot_lost_many(det, jobs, s_frac=s_frac, deadline=dl_d, track=(det is sec), return_lines=True)
    tm["detect"] = time.time() - t1
    for jid, p_ in enumerate(oks):
        p_["_det"] = [l for l in lines if l["job"] == jid]
        p_["lost_c"] = found[jid]
    t2 = time.time()
    if lazy:
        # the measured start domain and the accurate re-shoots: for the candidates actually tried
        for p_ in oks:
            p_["lazy"] = True
            p_["detected"] = dict(p_["lost_c"])
    else:
        _reshoot_accurate(sec, oks, det, deadline)
    tm["reshoot"] = time.time() - t2
    t_shoot = time.time() - t1
    for p_ in oks:
        p_["time"] += t_shoot / max(len(jobs), 1)
        p_["screen_timings"] = tm
        if verbose:
            print(f"  screened q {p_['orb'].q} residue {p_['orb'].residue:.4g}: lost {sorted(p_['lost_c'])} "
                  f"s0 {p_['loc']['u']['s0']:.2e} delta0 {p_['loc']['u']['delta0']:.1e}", flush=True)
    if verbose:
        print(f"  screening timings {dict((k, round(v, 1)) for k, v in tm.items())}", flush=True)
    return pre


def _reshoot_accurate(sec, oks, det, deadline):
    """Accurate re-shoots (the accurate map and error model) of the detected lost orbits of the screened
    candidates oks, all together; p_["lost_c"] is replaced by the accurate candidates."""
    if det is sec:
        return
    rlines = []
    for p_ in oks:
        for name, cand in p_["lost_c"].items():
            z, t = np.array(cand["start"][0]), np.array(cand["start"][1])
            rlines.append(dict(owner=p_, name=name, d=cand["d"], z=z, t=t, e=0.0, amp=1.0, n=0, done=False,
                               lost=None, nmax=cand["steps"] + 5 * p_["orb"].q, start=(z.copy(), t.copy())))
        p_["lost_c"] = {}
    if rlines:
        _shoot_lines(sec, rlines, 1e-7 * sec.scale, deadline)
        for l in rlines:
            if l["lost"] is not None:
                l["owner"]["lost_c"][l["name"]] = l["lost"]


def finalize_screened(sec, p_, n0=16, deadline=None, verbose=False):
    """A lazily screened candidate (screen_many(lazy=True)) before it is tried: the measured local manifolds
    (local_manifolds_many: the start domain [s0, Lambda s0] and its error delta0) and the accurate re-shoots of
    the detected lost orbits, each started at the point of the measured domain on the same manifold orbit
    (parameter s_d Lambda^-k in [s0, Lambda s0) for the detection's s_d: k cycles more, or fewer for k < 0),
    with the accurate map and its error model.  p_ is updated in place (loc, lost_c, nshoot)."""
    if not p_.get("lazy"):
        return p_
    orb, eig, mirror = p_["orb"], p_["eig"], p_["mirror"]
    if p_.get("loc_measured") is None:
        entries = [(orb, "u")] + ([] if mirror else [(orb, "s")])
        lms = local_manifolds_many(sec, entries, n0=n0)
        loc = {"u": lms[0]}
        if not mirror:
            loc["s"] = lms[1]
        p_["loc_provisional"] = p_["loc"]
        p_["loc_measured"] = loc
    loc = p_["loc_measured"]
    old_loc = p_["loc_provisional"]
    p_["loc"] = loc
    p_["lazy"] = False
    nper = orb.q * (2 if eig["reflection"] else 1)
    h_t = 1e-7 * sec.scale
    rlines = []
    for name, cand in p_.get("detected", {}).items():
        kind = "u" if cand["d"] > 0 else "s"
        lo, lp = loc[kind], old_loc[kind]
        z0 = np.array(cand["start"][0])
        # the detection line's parameter on the (same) parabola: s_d = (z0 - X) . v  (first order is enough: it only
        # chooses the start of the accurate line, whose own error is measured)
        sgn = next((sg for (nm, kd, v_, sg) in p_["specs"] if nm == name), +1)
        v = np.asarray(lo["v"])
        s_d = abs(float(np.dot(z0 - orb.pts[0], sgn * v)))
        Lam = float(lo["Lambda"])
        if not (s_d > 0 and Lam > 1):
            continue
        k = int(math.floor(math.log(s_d / lo["s0"]) / math.log(Lam)))
        s1 = s_d * Lam ** (-k)
        w = np.asarray(lo["w"])
        # the detection's manifold orbit carried into the measured domain, and four more starts spread over it: the
        # accurate orbit slides along the branch relative to the coarse one (QUASR 234994: the carried start alone
        # was not lost within its limit), so the first of these lines that is lost is the candidate
        n_l = max(1, int(1.5 * (cand["steps"] + k * nper) + 10 * orb.q))
        for s_ in [s1] + [lo["s0"] * Lam ** f for f in (0.15, 0.4, 0.65, 0.9)]:
            z = orb.pts[0] + sgn * s_ * v + s_ * s_ * w
            tg = sgn * v + 2 * s_ * w
            tg = tg / np.linalg.norm(tg)
            rlines.append(dict(owner=p_, job=id(p_), name=name, d=cand["d"], z=z, t=z + h_t * tg, e=0.0, amp=1.0, n=0,
                               done=False, lost=None, nmax=n_l, start=(z.copy(), (z + h_t * tg).copy())))
    p_["lost_c"] = {}
    if rlines:
        _shoot_lines(sec, rlines, h_t, deadline, first_only=True)
        for l in rlines:
            if l["lost"] is not None:
                b_ = p_["lost_c"].get(l["name"])
                if b_ is None or l["lost"]["steps"] < b_["steps"]:
                    p_["lost_c"][l["name"]] = l["lost"]
    p_["nshoot"] = int(max([p_["nshoot"]] + [l["nmax"] for l in rlines]))
    if verbose:
        print(f"  finalized q {orb.q} residue {orb.residue:.4g}: s0 {loc['u']['s0']:.2e} delta0 {loc['u']['delta0']:.1e} "
              f"ok {all(v_['ok'] for v_ in loc.values())} accurate lost {sorted(p_['lost_c'])} of "
              f"{sorted(p_.get('detected', {}))}", flush=True)
    return p_


def extend_shooting(sec, pre, factor=4.0, cap=8000, deadline=None, deadline_detect=None, verbose=False):
    """Continue the detection lines of screened candidates whose branches were not lost (screen_many keeps
    them) up to factor x their first limit (at most cap periods), then the accurate re-shoots of the new lost
    orbits.  Returns the candidates that gained a lost manifold orbit."""
    todo = [p_ for p_ in pre if p_.get("ok") and p_.get("_det") and not p_.get("lost_c")]
    if not todo:
        return []
    lines = []
    for p_ in todo:
        nmax_new = int(min(cap, factor * p_["nshoot"]))
        for l in p_["_det"]:
            if l["lost"] is None and np.all(np.isfinite(l["z"])):
                l["nmax"] = max(0, nmax_new - int(l.get("nd", 0)))     # the periods still to go
                l["done"] = l["nmax"] <= 0
                lines.append(l)
        p_["nshoot_extended"] = nmax_new
    t0 = time.time()
    det = sec.coarse() if hasattr(sec, "coarse") else sec
    _shoot_lines(det, lines, 1e-7 * sec.scale, deadline_detect or deadline, track=(det is sec))
    for p_ in todo:
        best = {}
        for l in p_["_det"]:
            if l["lost"] is None:
                continue
            b = best.get(l["name"])
            if b is None or l["lost"]["steps"] < b["steps"]:
                best[l["name"]] = l["lost"]
        p_["lost_c"] = best
        if p_.get("lazy") or p_.get("loc_measured") is not None:
            # lazily screened: the accurate re-shoots start in the measured domain (finalize_screened)
            p_["detected"] = dict(best)
            p_["lazy"] = True
    _reshoot_accurate(sec, [p_ for p_ in todo if not p_.get("lazy")], det, deadline)
    gained = [p_ for p_ in todo if p_["lost_c"]]
    if verbose:
        print(f"  extended shooting ({time.time() - t0:.0f}s): {len(gained)} of {len(todo)} candidates now have a lost "
              f"manifold orbit", flush=True)
    return gained


def barrier_from_orbit(sec, orb, a, want="edge", deadline=None, verbose=False, s0=None, n0=16, dmax=None,
                       amax=0.25, max_cycles=None, extra_cycles=10, max_pts=4000, target=None, nshoot=None,
                       shoot_extra=30, shoot_cap=3000, use_symmetry=True, screened=None, request=None,
                       cut_factor=2.0, reversible=True, a_err=0.0, lost_deadline=None, rev_lost_all=True,
                       pre_lost=None):
    """Edge barrier / manifold band from one hyperbolic orbit (module docstring).  Returns a
    JSON-friendly dict; ok=True only when every step is certified.  screened: the output of
    screen_orbit (else computed here).  request: in edge mode, closed curves whose uncertified flux
    already exceeds it are not certified, and the chord-error target is set from the margin.
    reversible: in edge mode, for a symmetric orbit of a stellarator-symmetric source, also the closed
    curves of the reversible closing (exact heteroclinic points on Z = 0; module docstring) and the
    generalised Lemma B (lemma_B_general).  a_err: error of the axis point a."""
    t0 = time.time()
    if screened is None:
        screened = screen_orbit(sec, orb, deadline=deadline, n0=n0, shoot_extra=shoot_extra, shoot_cap=shoot_cap,
                                use_symmetry=use_symmetry)
    kind_ = "edge_barrier" if want == "edge" else "manifold_band"
    if not screened.get("ok"):
        return dict(kind=kind_, ok=False, reason=screened.get("reason"))
    orb, eig, mirror, loc = screened["orb"], screened["eig"], screened["mirror"], screened["loc"]
    lost_c = screened["lost_c"]
    lam = abs(eig["lam"])
    out = dict(kind=kind_, ok=False,
               orbit=dict(q=int(orb.q), p=orb.p, residue=orb.residue, lam=eig["lam"], e_X=orb.e_X,
                          points=orb.pts.tolist(), info={k: v for k, v in orb.info.items() if np.isscalar(v)}))
    scale = sec.scale
    dmax = dmax if dmax is not None else 1e-2 * scale     # spacing; the arcs used are validated/refined later
    target = target if target is not None else 2e-6 * scale
    out["nshoot"] = screened["nshoot"]
    out["timings"] = dict(screen=screened.get("time", 0.0))
    out["start"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                    for k, v in loc.items()}
    out["lost_candidates"] = {k: dict(steps=v["steps"], e=v["e"], amp=v["amp"], d=v["d"]) for k, v in lost_c.items()}
    pre_ok = {k: v for k, v in (pre_lost or {}).items() if v.get("ok")}
    if want == "edge" and not lost_c and not pre_ok:
        out["reason"] = f"no manifold orbit lost within {screened['nshoot']} periods"
        return out
    if not all(v["ok"] for v in loc.values()):
        out["reason"] = "start error target not reached by the local manifold"
        return out
    # start-segment error: measured invariance defect of the quadratic local manifold (local_manifold)
    delta0 = max(v["delta0"] for v in loc.values())
    # lost points proven elsewhere for these branches (the same field lines: candidate_in_other_plane)
    lost_cert = {k: dict(v) for k, v in pre_ok.items()}

    dl_lost = lost_deadline if lost_deadline is not None else deadline

    def certify_more(stop_when_one=True, mirrored=False):
        for name, cand in sorted(lost_c.items(), key=lambda kv: (kv[1]["d"] < 0, kv[1]["steps"])):
            if dl_lost is not None and time.time() > dl_lost:
                continue
            if stop_when_one and any(v["ok"] for v in lost_cert.values()):
                break
            err = cand["e"] + cand["amp"] * (orb.e_X + delta0)
            if name not in lost_cert:
                cert = sec.certify_lost(cand["point"], err, cand["d"])
                cert.update(curve=name, steps_from_X=cand["steps"], err=err)
                lost_cert[name] = cert
                if verbose:
                    print(f"   lost {name}: steps {cand['steps']} err {err:.2e} -> {cert['ok']} {cert.get('reason', '')}",
                          flush=True)
            if mirror and mirrored and ("S" + name) not in lost_cert:
                # the mirror image S z lies within err + (field asymmetry along the orbit) of S W^u = W^s:
                # its own proof (backward lines)
                zS = np.asarray(cand["point"]) * np.array([1.0, -1.0])
                errS = err + mirror["step"] * (cand["steps"] + 1)
                certS = sec.certify_lost(zS, errS, -cand["d"])
                certS.update(curve="S" + name, steps_from_X=cand["steps"], err=errS, mirror_of=name)
                lost_cert["S" + name] = certS
                if verbose:
                    print(f"   lost S{name}: err {errS:.2e} -> {certS['ok']} {certS.get('reason', '')}", flush=True)
    if want == "edge" and rev_lost_all and mirror and not eig["reflection"] and reversible:
        # one lost point per unstable family up front: the reversible closing of one family needs the other's
        certify_more(stop_when_one=False)
    else:
        certify_more()
    out["timings"]["lost"] = time.time() - t0
    if want == "edge" and not any(v["ok"] for v in lost_cert.values()):
        out["lost_attempts"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("point", "end_point")}
                                for k, v in lost_cert.items()}
        out["reason"] = "no certified lost point (X outside every covered torus not established)"
        return out
    # -- 2. grow the generating curves until the closed curves needed exist
    # stellarator symmetry: W^s(S X) = S W^u(X); used when the orbit is symmetric and the field's
    # symmetry defect is measured (the defect's effect on a period map is added to the errors)
    out["symmetry"] = mirror
    gspecs = [("u+", "u", eig["vu"], +1)] + ([] if eig["reflection"] else [("u-", "u", eig["vu"], -1)])
    if not mirror:
        gspecs += [("s+", "s", eig["vs"], +1)] + ([] if eig["reflection"] else [("s-", "s", eig["vs"], -1)])
    curves = [GenCurve(name, kind, orb.pts[0], v, sgn, lam, eig["reflection"], loc[kind]["s0"], n0, orb.q,
                       w=loc[kind]["w"], mult=loc[kind]["Lambda"])
              for (name, kind, v, sgn) in gspecs]
    names = [c.name for c in curves] + ([("S" + c.name) for c in curves] if mirror else [])

    def all_curves():
        if not mirror:
            return curves
        return curves + [mirror_curve(c, "S" + c.name, mirror["base"], mirror["step"]) for c in curves]

    # reversible closing (edge mode): symmetric orbit of a symmetric source, regular hyperbolic, with a mirror
    # pair of chain neighbours in this plane
    rev = None
    if want == "edge" and reversible and mirror and not eig["reflection"] and orb.q >= 2 \
            and getattr(getattr(sec, "source", None), "stellsym", True):
        order_, succ_, pred_ = chain_order(orb, a)
        sigma_ = [(mirror["base"] - k) % orb.q for k in range(orb.q)]
        if any(sigma_[k] != k and sigma_[k] in (succ_[k], pred_[k]) for k in range(orb.q)):
            rev = dict(order=order_, succ=succ_, pred=pred_, sigma=sigma_, found={}, res={})
        else:
            out["reversible"] = "no mirror pair of chain neighbours in this plane"

    def exclusion(results):
        ex = {k for k, v in lost_cert.items() if v["ok"]}
        for v in results.values():
            if v["ok"]:
                for cr in v["crossings"]:
                    ex.add(cr["u_curve"])
                    ex.add(cr["s_curve"])
        closed = {} if rev is None else {k: v["F_up"] for k, v in rev["res"].items() if v.get("ok")}
        holds, nonex, free, F_ii = lemma_B_general(names, ex, closed, eig["reflection"])
        return ex, nonex, holds, free, F_ii, closed

    def any_curve():
        return any(v["ok"] for v in results.values()) or (rev is not None and any(v.get("ok") for v in rev["res"].values()))

    def try_reversible():
        for c in curves:
            if c.kind != "u" or c.name in rev["res"]:
                continue
            rec = rev["found"].get(c.name)
            if rec is None:
                rec = rev_find_crossing(sec, orb, a, c, rev["sigma"], rev["succ"], rev["pred"], delta0)
                if rec is None:
                    continue
                rev["found"][c.name] = rec
                if verbose:
                    print(f"   reversible {c.name}: heteroclinic point on Z = 0 at level {rec['level']} owner "
                          f"{rec['owner']} -> {rec['partner']} R {rec['h'][0]:.8f} box {rec['box_diam']:.1e}", flush=True)
            if c.level() >= rec["level"] + max(rev_js(orb.q)) + 1 or not c.active:
                rr = rev_close(sec, orb, a, c, rec, rev["sigma"], rev["succ"], rev["pred"], delta0, mirror["base"],
                               mirror["step"], target, request=request, a_err=a_err, verbose=verbose)
                rev["res"][c.name] = rr
                if rr.get("ok"):
                    c.active = False
                    c.stop_reason = "closed (reversible)"
                if verbose:
                    print(f"   reversible {c.name}: ok {rr.get('ok')} {rr.get('reason', '')} F_up {rr.get('F_up')} "
                          f"eps_max {rr.get('eps_max')}", flush=True)

    period = orb.q * (2 if eig["reflection"] else 1)
    if max_cycles is None:
        max_cycles = int(math.ceil(math.log(scale / loc["u"]["s0"]) / math.log(loc["u"]["Lambda"]))) + extra_cycles
    results = {}
    xcache = {}
    # manifold pieces beyond cut_factor x the orbit's distance from a are cut off (they head for the
    # lost region; the closed curves lie near the orbit)
    cut = None if cut_factor is None else (np.asarray(a), cut_factor * float(np.max(np.linalg.norm(orb.pts - a, axis=1))))
    for istep in range(max_cycles * orb.q):
        if deadline is not None and time.time() > deadline:
            out["reason"] = "deadline during manifold growth"
            break
        grow_step(sec, curves, dmax, amax, max_pts=max_pts, cut=cut)
        cyc = istep // orb.q
        if istep + 1 < period:
            continue
        # crossings checked after every period map once all branches exist (cached per gap): the
        # growth stops in the middle of a cycle as soon as a closed curve is possible
        found = {}
        cur = all_curves()
        for cls in ("A", "B"):
            g = find_gap_crossings(orb, a, cur, cls, cache=xcache)
            found[cls] = all(x is not None and x["found"] for x in g)
        if verbose and ((istep + 1) % orb.q == 0 or found["A"] or found["B"]):
            print(f"   step {istep + 1} (cycle {cyc + 1}): levels {[c.level() for c in curves]} "
                  f"pts {[len(c.pieces[-1].pts) for c in curves]} crossings {found} ({time.time() - t0:.0f}s)",
                  flush=True)
        if want == "edge":
            # assemble a class as soon as its crossings exist; done when a curve is certified and the
            # exclusions satisfy Lemma B (else: more lost points, or keep growing for the other class)
            for cls in ("A", "B"):
                if found[cls] and cls not in results:
                    r = assemble_curve(sec, orb, a, all_curves(), cls, delta0, target, verbose=verbose,
                                       request=request)
                    results[cls] = r
                    if verbose:
                        print(f"   class {cls} (cycle {cyc + 1}): ok {r['ok']} {r.get('reason', '')} F_up {r.get('F_up')}",
                              flush=True)
            if rev is not None:
                try_reversible()
            if any_curve():
                if not exclusion(results)[2]:
                    certify_more(stop_when_one=False, mirrored=True)
                if exclusion(results)[2]:
                    break
        elif found["A"] and found["B"]:
            break
        if not any(c.active for c in curves):
            break
    out["timings"]["grow"] = time.time() - t0
    curves_all = all_curves()
    out["growth"] = {c.name: dict(levels=c.level(), points=int(sum(len(p.pts) for p in c.pieces)),
                                  stop=c.stop_reason) for c in curves}
    out["start_error"] = {k: dict(s0=v["s0"], delta0=v["delta0"], defect_max=v["defect_max"]) for k, v in loc.items()}
    for cls in ("A", "B"):
        if cls in results or (deadline is not None and time.time() > deadline):
            continue
        r = assemble_curve(sec, orb, a, curves_all, cls, delta0, target, verbose=verbose,
                           request=request if want == "edge" else None)
        results[cls] = r
        if verbose:
            print(f"   class {cls}: ok {r['ok']} {r.get('reason', '')} F_up {r.get('F_up')}", flush=True)
    if rev is not None and (deadline is None or time.time() < deadline):
        # the curves whose heteroclinic point appeared late (or that stopped growing): close them now
        try_reversible()
    out["timings"]["curves"] = time.time() - t0
    out["curves"] = {k: {kk: vv for kk, vv in v.items() if kk != "polygon"} for k, v in results.items()}
    if rev is not None:
        out["curves"].update({"rev:" + k: {kk: vv for kk, vv in v.items() if kk != "polygon"}
                              for k, v in rev["res"].items()})
        out["chain_order"] = [int(k) for k in rev["order"]]
    good = {k: v for k, v in results.items() if v["ok"]}
    if rev is not None:
        good.update({"rev:" + k: dict(v, cls="rev:" + k) for k, v in rev["res"].items() if v.get("ok")})
    lost_ok = {k for k, v in lost_cert.items() if v["ok"]}
    out["lost_point"] = next((v for v in lost_cert.values() if v["ok"]), None)
    out["lost_attempts"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("point", "end_point")}
                            for k, v in lost_cert.items()}
    # -- 3. exclusion of the generating curves (Lemma B; generalised with the reversible closed families)
    excluded, nonex, lemmaB, free, F_ii, closed = exclusion(results)
    out["excluded"] = sorted(excluded)
    out["lemma_B"] = bool(lemmaB)
    out["closed_families"] = closed
    # -- 4. reference evidence: single steps of the tracer on manifold vertices
    out["evidence"] = reference_evidence(sec, orb, curves_all, results)
    if want == "edge":
        if not lost_ok:
            out["reason"] = "no certified lost point (X outside every covered torus not established)"
            return out
        if not good:
            out["reason"] = out.get("reason") or "no certified closed curve"
            return out
        if not lemmaB:
            out["reason"] = f"generating curves not excluded: {free}"
            return out
        # every covered torus misses X (case i: below the smallest certified curve) or passes through X (case ii:
        # it is the reversible curve of a closed family with a non-excluded member, F_ii)
        best = min(good.values(), key=lambda v: v["F_up"])
        F_up = float(max(best["F_up"], F_ii))
        bind = best if F_up == best["F_up"] else next(v for v in good.values() if v.get("cls", "").startswith("rev:")
                                                      and v["F_up"] == F_up)
        out.update(ok=bool(out["evidence"].get("ok", True)), F_up=F_up, curve_class=bind["cls"],
                   F_case_i=float(best["F_up"]), F_case_ii=(None if not np.isfinite(F_ii) else float(F_ii)),
                   crossings=bind.get("crossings") if "crossings" in bind else [bind.get("crossing")],
                   polygon=np.asarray(bind["polygon"]).tolist(),
                   error_budget=dict(eps_max=bind["eps_max"], eps_median=bind["eps_median"], tube=bind["tube"],
                                     quad_err=bind["quad_err"], e_X=orb.e_X, delta0=delta0,
                                     density_lower_bound=bind["density_check"]["min_lower_bound"],
                                     junction_err_max=bind.get("junction_err_max")))
        if not out["ok"]:
            out["reason"] = "reference evidence failed"
    else:
        if not (len(good) == 2 and lemmaB):
            out["reason"] = out.get("reason") or "band needs both closed curves certified"
            return out
        lo = min(v["F_up"] for v in good.values())
        hi = max(v["F_low"] for v in good.values())
        out.update(interval=(float(lo), float(hi)), ok=bool(lo < hi and out["evidence"].get("ok", True)),
                   polygons={k: v["polygon"].tolist() for k, v in good.items()})
        if lo >= hi:
            out["reason"] = "the two closed curves enclose fluxes closer than their errors"
        elif not out["ok"]:
            out["reason"] = "reference evidence failed"
    out["timings"]["total"] = time.time() - t0
    return out


def reference_evidence(sec, orb, curves, results, nsample=24, seed=1):
    """Fast-map steps vs the reference tracer on vertices of the grown pieces (both directions):
    |P_fast(z) - P_ref(z)| must be below the error estimate tau used for that step."""
    if not hasattr(sec, "reference_steps") or not getattr(sec, "use_reference", False):
        return dict(ok=True, note="no reference for this system")
    rng = np.random.default_rng(seed)
    out = dict(ok=True, samples=[])
    if any(c.d < 0 for c in curves) and hasattr(sec, "symmetry_defect"):
        pts = np.concatenate([p.pts for c in curves for p in c.pieces[:3]])[:64]
        x3 = np.stack([pts[:, 0] * np.cos(0.3), pts[:, 0] * np.sin(0.3), pts[:, 1]], -1)
        sd = sec.symmetry_defect(x3)
        out["symmetry_defect"] = sd
        if not sd < 1e-12:
            out["ok"] = False
    for c in curves:
        lv = [p for p in c.pieces[:-1] if len(p.pts) > 2]
        if not lv:
            continue
        pick = rng.choice(len(lv), size=min(len(lv), max(1, nsample // len(curves))), replace=False)
        Y = np.array([lv[i].pts[rng.integers(len(lv[i].pts))] for i in pick])
        yf, lost, tau = sec.step(Y, np.full(len(Y), c.d))
        r = sec.reference_steps(Y, c.d)
        if r is None:
            continue
        yr, er = r
        diff = np.linalg.norm(yf - yr, axis=1)
        ok = (~lost) & np.isfinite(diff)
        out["samples"].append(dict(curve=c.name, n=int(np.sum(ok)), max_diff=float(np.max(diff[ok])) if np.any(ok) else None,
                                   max_tau=float(np.max(tau[ok])) if np.any(ok) else None,
                                   max_ref_err=float(np.max(er[ok])) if np.any(ok) else None))
        if np.any(ok) and np.any(diff[ok] > tau[ok] + 10 * er[ok] + 1e-13 * sec.scale):
            out["ok"] = False
    # steps the section re-traced with more steps (lines next to coils): their error model checked too
    rs = getattr(sec, "refined_samples", None) or []
    if rs:
        pick = rng.choice(len(rs), size=min(len(rs), 8), replace=False)
        for dsign in (+1.0, -1.0):
            Y = np.array([rs[i][0] for i in pick if rs[i][1] == dsign])
            if len(Y) == 0:
                continue
            yf, lost, tau = sec.step(Y, np.full(len(Y), dsign))
            r = sec.reference_steps(Y, dsign)
            if r is None:
                continue
            yr, er = r
            diff = np.linalg.norm(yf - yr, axis=1)
            ok = (~lost) & np.isfinite(diff)
            out["samples"].append(dict(curve=f"refined steps ({'+' if dsign > 0 else '-'})", n=int(np.sum(ok)),
                                       max_diff=float(np.max(diff[ok])) if np.any(ok) else None,
                                       max_tau=float(np.max(tau[ok])) if np.any(ok) else None,
                                       max_ref_err=float(np.max(er[ok])) if np.any(ok) else None))
            if np.any(ok) and np.any(diff[ok] > tau[ok] + 10 * er[ok] + 1e-13 * sec.scale):
                out["ok"] = False
    return out


# ============================================================================ field: axis, orbits, candidates
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def _refine_roots(fun, a, b, fa, fb, tol, maxit=60):
    """Illinois (modified regula falsi) for many brackets at once; fun(x array) -> values (NaN = failed).
    Returns roots (NaN where a bracket failed) and final |f|."""
    a, b, fa, fb = (np.array(v, float) for v in (a, b, fa, fb))
    side = np.zeros(len(a), int)
    x = 0.5 * (a + b)
    fx = np.full(len(a), np.nan)
    alive = np.isfinite(fa) & np.isfinite(fb) & (fa * fb < 0)
    for it in range(maxit):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        xc = (a[idx] * fb[idx] - b[idx] * fa[idx]) / (fb[idx] - fa[idx])
        bad = ~np.isfinite(xc) | (xc <= np.minimum(a[idx], b[idx])) | (xc >= np.maximum(a[idx], b[idx]))
        xc = np.where(bad, 0.5 * (a[idx] + b[idx]), xc)
        fc = fun(xc)
        x[idx], fx[idx] = xc, fc
        dead = ~np.isfinite(fc)
        alive[idx[dead]] = False
        for n, i in enumerate(idx):
            if dead[n]:
                continue
            if fc[n] * fb[i] < 0:
                a[i], fa[i] = b[i], fb[i]
                b[i], fb[i] = xc[n], fc[n]
                side[i] = 0
            else:
                b[i], fb[i] = xc[n], fc[n]
                fa[i] *= 0.5 if side[i] == 1 else 1.0
                side[i] = 1
            if abs(fc[n]) <= tol or abs(b[i] - a[i]) <= 4e-16 * max(1.0, abs(b[i])):
                alive[i] = False
    x[~np.isfinite(fx)] = np.nan
    return x, fx


def symmetric_roots(sec, q, plane, R_vals, coarse=None, tol=1e-9):
    """R of the stellarator-symmetric closed lines of q field periods crossing Z = 0 at
    phi = plane * pi / nfp, from sign changes of Z after q half periods along the scanned R values
    (a line through two symmetry planes is invariant under their composition, a shift by q field
    periods); refined with the accurate tracer."""
    nfp = sec.nfp
    phi_s = plane * np.pi / nfp
    tr = coarse or sec.tracer
    R_vals = np.asarray(R_vals, float)
    o = tr.trace(np.stack([R_vals, np.zeros_like(R_vals)], -1), phi_s, q * np.pi / nfp)
    Z = np.where(o["lost"], np.nan, o["y"][:, 1])
    i = np.flatnonzero(np.isfinite(Z[:-1]) & np.isfinite(Z[1:]) & (Z[:-1] * Z[1:] < 0))
    if len(i) == 0:
        return np.zeros(0)

    def fun(R):        # the coarse map: the roots are polished afterwards by Newton with the accurate map
        oo = tr.trace(np.stack([R, np.zeros_like(R)], -1), phi_s, q * np.pi / nfp)
        return np.where(oo["lost"], np.nan, oo["y"][:, 1])
    a, b = R_vals[i], R_vals[i + 1]
    fa, fb = Z[i], Z[i + 1]
    x, fx = _refine_roots(fun, a, b, fa, fb, tol * sec.scale, maxit=40)
    return x[np.isfinite(x)]


def symmetric_roots_rays(sec, q, rays, tracer, tol=1e-7, maxit=25):
    """symmetric_roots for several rays at once: rays = list of (plane, side, R_vals).  One coarse
    trace of every scan line, then Illinois on all brackets together (per-line start angle).
    Returns a list of (plane, side, R).  Only the coarse map is used: the roots are polished by Newton
    on P^q afterwards."""
    nfp = sec.nfp
    R_all, phi_all, meta = [], [], []
    for (plane, side, R_vals) in rays:
        R_all.append(R_vals)
        phi_all.append(np.full(len(R_vals), plane * np.pi / nfp))
        meta.append((plane, side, len(R_vals)))
    R_all = np.concatenate(R_all)
    phi_all = np.concatenate(phi_all)
    o = tracer.trace(np.stack([R_all, np.zeros_like(R_all)], -1), phi_all, q * np.pi / nfp)
    Z = np.where(o["lost"], np.nan, o["y"][:, 1])
    ia, pa, sa = [], [], []
    off = 0
    for (plane, side, n) in meta:
        Zr = Z[off:off + n]
        i = np.flatnonzero(np.isfinite(Zr[:-1]) & np.isfinite(Zr[1:]) & (Zr[:-1] * Zr[1:] < 0))
        ia.append(off + i)
        pa += [plane] * len(i)
        sa += [side] * len(i)
        off += n
    ia = np.concatenate(ia) if ia else np.zeros(0, int)
    if len(ia) == 0:
        return []
    phis = phi_all[ia]

    a, b, fa, fb = R_all[ia], R_all[ia + 1], Z[ia], Z[ia + 1]
    x, fx = _refine_roots_phi(tracer, q, nfp, a, b, fa, fb, phis, tol * sec.scale, maxit)
    return [(pa[k], sa[k], float(x[k])) for k in range(len(x)) if np.isfinite(x[k])]


def _refine_roots_phi(tracer, q, nfp, a, b, fa, fb, phis, tol, maxit):
    """Illinois for brackets with per-bracket start angles (Z after q half periods)."""
    a, b, fa, fb = (np.array(v, float) for v in (a, b, fa, fb))
    side = np.zeros(len(a), int)
    x = 0.5 * (a + b)
    fx = np.full(len(a), np.nan)
    alive = np.isfinite(fa) & np.isfinite(fb) & (fa * fb < 0)
    for it in range(maxit):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        xc = (a[idx] * fb[idx] - b[idx] * fa[idx]) / (fb[idx] - fa[idx])
        bad = ~np.isfinite(xc) | (xc <= np.minimum(a[idx], b[idx])) | (xc >= np.maximum(a[idx], b[idx]))
        xc = np.where(bad, 0.5 * (a[idx] + b[idx]), xc)
        oo = tracer.trace(np.stack([xc, np.zeros_like(xc)], -1), phis[idx], q * np.pi / nfp)
        fc = np.where(oo["lost"], np.nan, oo["y"][:, 1])
        x[idx], fx[idx] = xc, fc
        dead = ~np.isfinite(fc)
        alive[idx[dead]] = False
        for n, i in enumerate(idx):
            if dead[n]:
                continue
            if fc[n] * fb[i] < 0:
                a[i], fa[i] = b[i], fb[i]
                b[i], fb[i] = xc[n], fc[n]
                side[i] = 0
            else:
                b[i], fb[i] = xc[n], fc[n]
                fa[i] *= 0.5 if side[i] == 1 else 1.0
                side[i] = 1
            if abs(fc[n]) <= tol or abs(b[i] - a[i]) <= 4e-16 * max(1.0, abs(b[i])):
                alive[i] = False
    x[~np.isfinite(fx)] = np.nan
    return x, fx


def section_point_at_zero(sec, R, plane):
    """The point at phi = 0 of the line through (R, phi = plane pi/nfp, Z = 0)."""
    if plane == 0:
        return np.array([R, 0.0])
    o = sec.tracer.trace(np.array([[R, 0.0]]), np.pi / sec.nfp, -np.pi / sec.nfp)
    if o["lost"][0]:
        return None
    return o["y"][0].copy()


def rotation_number(sec, X0, a0, q):
    """Poloidal turns p of the closed line through X0 around the axis line through a0 over q periods."""
    tr = sec.tracer
    o = tr.trace(np.stack([X0, a0]), 0.0, q * 2 * np.pi / sec.nfp, nsteps=q * sec.sp * 4, record=True)
    if np.any(o["lost"]):
        return None
    st = np.array([np.asarray(X0) - np.asarray(a0)] + [r[0] - r[1] for r in o["rec"]])
    ang = np.unwrap(np.arctan2(st[:, 1], st[:, 0]))
    return int(round((ang[-1] - ang[0]) / (2 * np.pi)))


def _sym_crossings(T, phi):
    """(R_inboard, R_outboard) where the torus meets Z = 0 in the symmetry plane phi (theta = 0, pi)."""
    R, _ = T.geometry_on(np.array([0.0, np.pi]), np.array([phi, phi]))
    return float(min(R)), float(max(R))


def rotation_profile(sec, T_last, a0, n_lines=24, nper=64, reach=0.7):
    """Rotation numbers per field period (|poloidal angle advance| / 2 pi per period, around the axis
    line) of lines started on the outboard ray at phi = 0 beyond T_last, up to reach x the distance
    to the nearest coil point.  Lost lines get NaN.  Returns (R values, rho)."""
    cp = sec.source.sample_points()
    R0 = _sym_crossings(T_last, 0.0)[1]
    dc = float(np.min(np.linalg.norm(cp - np.array([R0, 0.0, 0.0]), axis=1)))
    R = R0 + np.linspace(0.0, reach * dc, n_lines + 1)[1:]
    y = np.concatenate([np.stack([R, np.zeros_like(R)], -1), np.atleast_2d(a0)])
    m = max(8, int(getattr(sec, "sp", 12)) * 2 // 3)       # (8 at the 12 steps of QUASR fields)
    tr = sec.line_tracer(k=4, steps_per_period=m) if hasattr(sec, "line_tracer") else \
        LineTracer(sec.engine, sec.nfp, k=4, steps_per_period=m, sign=sec.sign)
    o = tr.trace(y, 0.0, nper * 2 * np.pi / sec.nfp, nsteps=nper * m, record=True)
    st = np.array([y] + o["rec"])                         # (n+1, L+1, 2)
    d = st[:, :-1, :] - st[:, -1:, :]
    ang = np.unwrap(np.arctan2(d[:, :, 1], d[:, :, 0]), axis=0)
    # weighted Birkhoff average of the per-period advance (bump weights: fast convergence on tori)
    adv = ang[m::m] - ang[:-m:m]                          # (nper, L)
    t = (np.arange(nper) + 0.5) / nper
    w = np.exp(-1.0 / (t * (1 - t)))
    rho = np.abs(np.sum(w[:, None] * adv, 0) / np.sum(w)) / (2 * np.pi)
    rho[o["lost"][:-1]] = np.nan
    if o["lost"][-1]:
        rho[:] = np.nan
    return R, rho


def qs_from_profile(rho_last, rho, q_max=24, n_q=3, window=0.06):
    """Periods q of the lowest-order rationals between rho_last and the rotation numbers measured
    beyond T_last (in the order in which the profile meets them going outward); if fewer than n_q, the
    lowest-order rationals within window x rho_last in the direction of the profile's trend (the
    chaotic sea beyond the last measured line)."""
    from .chains import rationals_between
    fin = [r for r in rho if np.isfinite(r)]
    trend = np.sign(fin[-1] - rho_last) if fin and fin[-1] != rho_last else 1.0
    # extra rationals: both sides of rho_last within the window (the profile's direction first at equal
    # q; a resonance just beyond the last torus may lie on either side of the measured lines)
    ex = rationals_between(rho_last * (1 - window), rho_last * (1 + window), q_max)
    extra = sorted(ex, key=lambda t: (t[1], -trend * np.sign(t[0] / t[1] - rho_last), abs(t[0] / t[1] - rho_last)))
    out = []
    prev = rho_last
    for r in rho:
        if not np.isfinite(r):
            break
        for (p, q) in sorted(rationals_between(prev, r, q_max), key=lambda t: t[1]):
            if q not in out:
                out.append(q)
        prev = r
    out.sort(key=lambda q: q)            # low order first (the widest chains)
    out = out[:n_q]
    for (p, q) in extra:
        if len(out) >= n_q:
            break
        if q not in out:
            out.append(q)
    return out


def qs_edge_window(rho, rho_last=None, q_max=24, n_q=3, q_cap=64, n_edge=4, widen=0.03):
    """Periods q of the lowest-order rationals at the edge of the rotation profile: between the rotation
    numbers of the last n_edge lines before the first lost one (and of lines beyond it that survived),
    widened by `widen`.  The orbits whose manifolds certify edge requests sit there; chains between T_last and
    the edge are confined by tori.  When the
    first line beyond T_last is already lost the edge is T_last itself (rho_last).  q runs up to
    max(q_max, min(q_cap, 1.25 / rho_edge)): low-transform devices (rho ~ 0.02-0.04 per period) have their
    edge chains at q ~ 25-60."""
    from .chains import rationals_between
    r = np.array([np.nan if v is None else v for v in rho], float)
    fin = np.isfinite(r)
    i0 = int(np.argmax(~fin)) if np.any(~fin) else len(r)
    if i0 == 0:
        if rho_last is None or not np.isfinite(rho_last):
            return []
        vals = [float(rho_last)] + list(r[fin])
        ref_ = float(rho_last)
    else:
        vals = list(r[max(0, i0 - n_edge):i0]) + list(r[i0:][fin[i0:]])
        ref_ = float(r[i0 - 1])
    lo, hi = min(vals) * (1 - widen), max(vals) * (1 + widen)
    qc = int(max(q_max, min(q_cap, math.ceil(1.25 / max(lo, 1e-3)))))
    rats = sorted(rationals_between(lo, hi, qc), key=lambda t: (t[1], abs(t[0] / t[1] - ref_)))
    out = []
    for (p, q) in rats:
        if q not in out:
            out.append(q)
        if len(out) >= n_q:
            break
    return out


def qs_extend(qs, rho_ref, n_want, q_max=24, windows=(0.06, 0.12, 0.25)):
    """qs extended (to n_want values) by the lowest-order rationals within growing windows around rho_ref (the
    rotation number at the edge), each window's rationals by q then distance."""
    from .chains import rationals_between
    out = list(dict.fromkeys(int(q) for q in qs))
    if rho_ref is None or not np.isfinite(rho_ref) or rho_ref <= 0:
        return out
    qc = int(max(q_max, min(64, math.ceil(1.25 / max(rho_ref * (1 - windows[-1]), 1e-3)))))
    for w in windows:
        if len(out) >= n_want:
            break
        rats = sorted(rationals_between(rho_ref * (1 - w), rho_ref * (1 + w), qc),
                      key=lambda t: (t[1], abs(t[0] / t[1] - rho_ref)))
        for (p, q) in rats:
            if q not in out:
                out.append(q)
            if len(out) >= n_want:
                break
    return out


def find_axis_point(sec, R_lo, R_hi, n=33):
    """The magnetic axis' point at phi = 0 (elliptic symmetric period-one line between R_lo, R_hi)."""
    roots = symmetric_roots(sec, 1, 0, np.linspace(R_lo, R_hi, n))
    best = None
    for R in roots:
        orb = orbit_from_point(sec, np.array([R, 0.0]), 1)
        if orb is None:
            continue
        if 0 < orb.residue < 1 and (best is None or orb.e_X < best.e_X):
            best = orb
    return best


def ray_ratio(T_poly, a, pts):
    """|x - a| / (distance from a to the polygon T_poly along the ray through x), per point."""
    out = []
    Pc = np.concatenate([T_poly, T_poly[:1]])
    for x in np.atleast_2d(pts):
        d = x - a
        L = np.linalg.norm(d)
        far = a + d / L * 10.0 * (np.max(np.linalg.norm(T_poly - a, axis=1)) + 1.0)
        I, J, t, u = seg_intersections(np.stack([a, far]), Pc)
        if len(I) == 0:
            out.append(np.inf)
            continue
        out.append(L / (float(np.max(t)) * np.linalg.norm(far - a)))
    return np.array(out)


def symmetric_roots_accurate(sec, q, roots, rel=1e-4, tol=1e-13, maxit=30):
    """Accurate symmetric periodic orbits from coarse roots (plane, side, R) of Z after q half periods on a
    symmetry line, all roots together: brackets [R -+ w scale] (w = rel, 8 rel, 64 rel until the accurate
    tracer's Z changes sign), Illinois with the accurate tracer (batched) to |Z| <= tol scale, plane-1 points
    carried back to phi = 0, then the orbits on the accurate map without Newton (closure, finite-difference
    monodromy, e_X).  Returns a list of Orbit or None."""
    nfp = sec.nfp
    tr = sec.tracer
    n = len(roots)
    out = [None] * n
    if n == 0:
        return out
    pl = np.array([r[0] for r in roots], int)
    R = np.array([r[2] for r in roots], float)
    phis = pl * np.pi / nfp
    a_ = np.full(n, np.nan)
    b_ = np.full(n, np.nan)
    fa = np.full(n, np.nan)
    fb = np.full(n, np.nan)
    todo = np.arange(n)
    for w in (rel, 8 * rel, 64 * rel):
        if len(todo) == 0:
            break
        lo, hi = R[todo] - w * sec.scale, R[todo] + w * sec.scale
        o = tr.trace(np.stack([np.concatenate([lo, hi]), np.zeros(2 * len(todo))], -1),
                     np.concatenate([phis[todo], phis[todo]]), q * np.pi / nfp)
        Z = np.where(o["lost"], np.nan, o["y"][:, 1])
        za, zb = Z[:len(todo)], Z[len(todo):]
        okb = np.isfinite(za) & np.isfinite(zb) & (za * zb < 0)
        a_[todo[okb]], b_[todo[okb]], fa[todo[okb]], fb[todo[okb]] = lo[okb], hi[okb], za[okb], zb[okb]
        todo = todo[~okb & np.isfinite(za) & np.isfinite(zb)]
    have = np.flatnonzero(np.isfinite(a_))
    if len(have) == 0:
        return out
    x, fx = _refine_roots_phi(tr, q, nfp, a_[have], b_[have], fa[have], fb[have], phis[have], tol * sec.scale, maxit)
    X0 = np.stack([x, np.zeros_like(x)], -1)
    p1 = pl[have] == 1
    if np.any(p1):
        ob = tr.trace(X0[p1], np.pi / nfp, -np.pi / nfp)
        X0[p1] = np.where(ob["lost"][:, None], np.nan, ob["y"])
    good = np.all(np.isfinite(X0), 1)
    if not np.any(good):
        return out
    orbs = orbits_from_points(sec, X0[good], q, h_fd=1e-7 * sec.scale, polish=0)
    for k, orb in zip(have[good], orbs):
        if orb is not None:
            orb.info["symmetric_root"] = dict(plane=int(pl[k]), R=float(x[list(have).index(k)]))
        out[k] = orb
    return out


def orbit_candidates(sec, T_last, a0, qs, n_scan=160, reach=0.7, deadline=None, verbose=False, accurate_roots=False,
                     max_ratio=None):
    """Hyperbolic symmetric periodic orbits beyond T_last (all section points outside its section
    curve), from scans of the four symmetry rays (inboard/outboard at phi = 0 and pi/nfp) between
    T_last and reach x the distance to the nearest coil point.  Sorted innermost first."""
    nfp = sec.nfp
    src = sec.source
    cp = src.sample_points()
    th = 2 * np.pi * np.arange(2048) / 2048
    Tpoly0 = np.stack(T_last.geometry_on(th, np.zeros(2048)), -1)
    cands = []
    coarse = sec.line_tracer(k=4, steps_per_period=max(4, int(getattr(sec, "sp", 12)) // 3))
    det_sec = sec.coarse() if hasattr(sec, "coarse") else sec
    a_planes = None
    if max_ratio is not None:
        # the axis' R on the symmetry line of each plane (phi = 0: a0; phi = pi / nfp: a0 carried half a period)
        oa = coarse.trace(np.atleast_2d(a0), 0.0, np.pi / nfp)
        a_planes = {0: float(a0[0]), 1: (float(oa["y"][0, 0]) if not oa["lost"][0] else None)}
        if a_planes[1] is None:
            a_planes = None
    t0 = time.time()
    for q in qs:
        if deadline is not None and time.time() > deadline:
            break
        found = []
        if verbose:
            print(f"  scanning q = {q} ({time.time() - t0:.0f}s)", flush=True)
        rays = []
        for plane in (0, 1):
            phi_s = plane * np.pi / nfp
            for side in (+1, -1):                          # +1 outboard, -1 inboard
                Rin, Rout = _sym_crossings(T_last, phi_s)
                R0 = Rout if side > 0 else Rin
                x0 = np.array([R0 * math.cos(phi_s), R0 * math.sin(phi_s), 0.0])
                dc = float(np.min(np.linalg.norm(cp - x0, axis=1)))
                span = reach * dc
                if max_ratio is not None and a_planes is not None:
                    # with a request: orbits farther than max_ratio x T_last's distance from the axis (along the
                    # ray) enclose far more than it -- the scan stops there
                    da = abs(R0 - a_planes[plane])
                    span = min(span, max(0.0, (max_ratio - 1.0) * da)) if da > 0 else span
                # quadratic spacing: dense next to T_last, where the chaotic sea may be thin
                rays.append((plane, side, R0 + side * span * np.linspace(0.0, 1.0, n_scan)[1:] ** 2))
        found = symmetric_roots_rays(sec, q, rays, coarse)
        if not found:
            continue
        # points at phi = 0 (plane-1 roots traced back half a period, all at once)
        X0 = np.array([[R, 0.0] for (pl, sd, R) in found])
        p1 = np.array([pl == 1 for (pl, sd, R) in found])
        if np.any(p1):
            o = coarse.trace(X0[p1], np.pi / nfp, -np.pi / nfp)
            X0[p1] = np.where(o["lost"][:, None], np.nan, o["y"])
        ok = np.all(np.isfinite(X0), 1)
        # one orbit per root, classified with the detection map (the accurate orbit, monodromy and e_X
        # are recomputed for the screened candidates); duplicates removed after
        orbs = orbits_from_points(det_sec, X0[ok], q, h_fd=1e-5 * sec.scale, polish=3)
        meta = [f for f, k in zip(found, ok) if k]
        # weakly hyperbolic / elliptic symmetric orbits of thin chains: M - I is nearly singular, so the 2D Newton
        # on the coarse map (whose own error per period may be ~1e-5) does not converge; the symmetric 1D root
        # is well posed -- refined with the accurate tracer, the orbit is then measured on the accurate map
        # (only orbits whose coarse Newton ended near the root -- thin chains -- not those that diverged)
        bad = [m for m, orb in enumerate(orbs) if orb is not None and orb.info["closure"] > 1e-6 * sec.scale
               and np.linalg.norm(orb.pts[0] - X0[ok][m]) < 1e-3 * sec.scale and abs(orb.residue) < 0.05]
        if bad and accurate_roots:
            fixed = symmetric_roots_accurate(sec, q, [meta[m] for m in bad])
            for m, orb2 in zip(bad, fixed):
                if orb2 is not None:
                    orbs[m] = orb2
        new = []
        for (pl, sd, R), orb in zip(meta, orbs):
            if orb is None or orb.info["closure"] > 1e-6 * sec.scale:
                continue
            if q > 1 and np.min(np.linalg.norm(orb.pts[1:] - orb.pts[0], axis=1)) < 1e-6 * sec.scale:
                continue                                       # a lower period
            if any(np.min(np.linalg.norm(o_.pts - orb.pts[0], axis=1)) < 1e-5 * sec.scale for o_ in cands + new):
                continue
            orb.info.update(plane=pl, side=sd, R_sym=float(R))
            new.append(orb)
        if new:
            ps = rotation_numbers(sec, np.array([o_.pts[0] for o_ in new]), a0, q, tracer=coarse)
            for orb, p_ in zip(new, ps):
                rat = ray_ratio(Tpoly0, a0, orb.pts)
                orb.info.update(ratio_min=float(np.min(rat)), ratio_mean=float(np.mean(rat)))
                orb.p = p_
                if verbose:
                    print(f"  q {q} plane {orb.info['plane']} side {orb.info['side']:+d} R {orb.info['R_sym']:.6f}: "
                          f"residue {orb.residue:.4g} p {orb.p} ratio {np.min(rat):.4f}-{np.max(rat):.4f} "
                          f"closure {orb.info['closure']:.1e} e_X {orb.e_X:.1e}", flush=True)
            cands.extend(new)
    hyp = [o for o in cands if (o.residue < -1e-6 or o.residue > 1 + 1e-6) and o.info["ratio_min"] > 1.0]
    hyp.sort(key=lambda o: o.info["ratio_mean"])
    return hyp, cands


SECTION_STEPS = None        # macro steps per field period of FieldSection's accurate map: None = calibrated
STEP_LADDER = (12, 16, 24, 32, 48, 64, 96, 128, 192)
STEP_TOL = 1e-9


def calibrate_steps(engine, nfp, sign, pts, ladder=STEP_LADDER, tol=STEP_TOL):
    """The smallest macro-step count per field period on the ladder (k = 4) whose one-period map differs from
    the map at twice the count by <= tol x R at the sample points pts (k, 2) (the difference ~ the coarser
    map's error).  12 on QUASR 69266 (difference 5.7e-11 R, the setting the tracer error model was validated
    at); other QUASR fields need 16-64 (difference at 12 steps up to 8.8e-4 R, QUASR 1862319: 64 steps); the rippled
    CoilStellaration fields need 24-64 (DDMaikF39a8: 5.9e-5 R at 12, 3.5e-10 R at 48 steps; at 12 steps the
    per-period error estimate exceeds the 1e-4 scale stop on every line, so no closed line -- not even the axis
    -- is found).  Returns (steps, record)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    Rs = float(np.max(np.abs(pts[:, 0])))
    cache = {}

    def pmap(sp):
        if sp not in cache:
            tr = LineTracer(engine, nfp, k=4, steps_per_period=sp, sign=sign)
            y, lost, err = tr.period_map(pts, 1)
            cache[sp] = (y, lost)
        return cache[sp]
    hist = []
    for sp in ladder:
        y1, l1 = pmap(sp)
        y2, l2 = pmap(2 * sp)
        ok = ~(l1 | l2)
        d = float(np.max(np.linalg.norm(y1[ok] - y2[ok], axis=1))) / Rs if np.any(ok) else np.inf
        hist.append((sp, d, int(np.sum(ok))))
        if np.any(ok) and d <= tol:
            return sp, dict(steps=sp, history=hist)
    return ladder[-1], dict(steps=ladder[-1], history=hist, reached=False)


def _setup_section(engine, source, T_last, sign=None):
    th = np.array([0.0, np.pi])
    R2, _ = T_last.geometry_on(th, np.zeros(2))
    Rc = 0.5 * (R2[0] + R2[1])
    if sign is None:
        B = np.asarray(engine.evaluate(np.array([[Rc, 0.0, 0.0]]), grad=False, A=False)["B"])
        sign = 1.0 if B[0, 1] > 0 else -1.0
    scale = float(abs(R2[0] - R2[1]))
    if SECTION_STEPS is None:
        pts = np.array([[R2[0] + 0.05 * (Rc - R2[0]), 0.0], [Rc, 0.0], [R2[1] + 0.05 * (Rc - R2[1]), 0.0]])
        sp, cal = calibrate_steps(engine, int(source.nfp), sign, pts)
    else:
        sp, cal = int(SECTION_STEPS), dict(steps=int(SECTION_STEPS), fixed=True)
    sec = FieldSection(engine, source, sign=sign, scale=scale, steps_per_period=sp)
    sec.calibration = cal
    return sec, R2


def half_period_section(sec):
    """The same field's section in the other symmetry plane phi = pi / nfp (same steps, scale, deadline)."""
    s2 = FieldSection(sec.engine, sec.source, sign=sec.sign, k=sec.k, steps_per_period=sec.sp,
                      reference=sec.use_reference, scale=sec.scale, phi0=float(np.pi / sec.nfp) + sec.phi0)
    s2._ref = sec._ref
    s2.hard_deadline = sec.hard_deadline
    s2.calibration = getattr(sec, "calibration", None)
    return s2


def transfer_lost_names(sec, orb, s2, orb2, names, hp):
    """Plane-0 unstable generating curves with a certified lost point -> the names of the corresponding plane-2
    curves: the half-period map psi carries W^u(X_0) onto W^u(psi X_0) and the lost line is the same field line, so
    the plane-0 proof (every line of its disk reaches B_phi < 0; the true manifold point lies in the disk) makes
    the image point of the plane-2 branch lost.  The branch is identified by the sign of D psi v_u against the
    plane-2 unstable eigenvector (central differences over half a period).  Only u+ / u- are transferred."""
    v0 = orb.eig()["vu"]
    v2 = orb2.eig()["vu"]
    h = 1e-6 * sec.scale
    o = sec.tracer.trace(np.stack([orb.pts[0] + h * v0, orb.pts[0] - h * v0]), sec.phi0, hp)
    if np.any(o["lost"]):
        return {}
    dv = (o["y"][0] - o["y"][1]) / (2 * h)
    # psi(X_0) must be orb2's base point (the plane-2 orbit is the half-period image, re-polished)
    sgn = 1.0 if float(np.dot(dv, v2)) > 0 else -1.0
    out = {}
    for nm in names:
        if nm in ("u+", "u-"):
            plus = (nm == "u+") == (sgn > 0)
            out[nm] = "u+" if plus else "u-"
    return out


def candidate_in_other_plane(sec, orb, a0, deadline=None, verbose=False, request=None, reversible=True,
                             lost_names=(), lost_records=None, **kw):
    """An orbit with no mirror pair of chain neighbours in phi = 0 (q even, two points on Z = 0) has its pairs
    straddling the line Z = 0 of the other symmetry plane: the orbit and the axis point are carried there by
    half a period, polished on that plane's map, screened (lost manifold orbits) and given to
    barrier_from_orbit (the theorems hold in either symmetry plane: a covered torus meets every plane
    phi = const in one Jordan curve around the axis, with the same flux).  Returns (record, section)."""
    s2 = half_period_section(sec)
    hp = np.pi / sec.nfp
    o = sec.tracer.trace(np.concatenate([orb.pts, np.atleast_2d(a0)]), sec.phi0, hp)
    if np.any(o["lost"]):
        return dict(ok=False, reason="orbit or axis lost on the half period"), s2
    pts2, a2 = o["y"][:-1], o["y"][-1]
    ax2 = orbit_from_point(s2, a2, 1)
    if ax2 is None or not (0 < ax2.residue < 1):
        return dict(ok=False, reason="axis point not found in the other plane"), s2
    orb2 = Orbit(orb.q, pts2, orb.M, e_X=orb.e_X, p=orb.p, info=dict(orb.info))
    pre = screen_many(s2, [orb2], deadline=deadline, verbose=verbose)
    p_ = pre[0] if pre else dict(ok=False, reason="no screening")
    if not p_.get("ok"):
        return dict(ok=False, reason=f"other plane: {p_.get('reason')}"), s2
    pre_lost = {}
    if lost_names:
        # the plane-0 lost points carry over (the same field lines)
        np_ = np.linalg.norm(p_["orb"].pts[0] - pts2[0])
        if np_ < 1e-6 * sec.scale:
            tmap = transfer_lost_names(sec, orb, s2, p_["orb"], lost_names, hp)
            for nm0, nm2 in tmap.items():
                rec0 = dict((lost_records or {}).get(nm0) or {})
                pre_lost[nm2] = dict(ok=True, transferred_from=f"phi = {sec.phi0:.6f} branch {nm0}",
                                     proof={k: v for k, v in rec0.items() if k not in ("point", "end_point")})
    if not p_.get("lost_c") and not pre_lost:
        return dict(ok=False, reason="other plane: no lost manifold orbit"), s2
    finalize_screened(s2, p_, deadline=deadline, verbose=verbose)
    r = barrier_from_orbit(s2, p_["orb"], ax2.pts[0], want="edge", deadline=deadline, verbose=verbose, screened=p_,
                           request=request, reversible=reversible, a_err=float(ax2.e_X), pre_lost=pre_lost, **kw)
    r["transferred_lost"] = sorted(pre_lost)
    r["plane_phi"] = float(s2.phi0)
    return r, s2


def edge_barrier(engine, source, T_last, request=None, deadline=None, verbose=False, qs=None, q_max=24, extra_qs=(),
                 window=0.12, n_q=4, n_scan=96, max_candidates=6, max_screen=6, extra_after_success=0, orbits=None,
                 on_orbits=None, min_abs_residue=0.05, max_rounds=3, try_ring=True, ring_T_quick=4,
                 ring_share=0.5, ring_nper=6, ring_flux_share=0.85, reversible=True, min_rev_residue=5e-5,
                 edge_first=True, grace=None, other_plane=True, extend_shoot=True, cand_share=0.6, cand_min=60.0,
                 skip_factor=2.5, scan_ratio=4.0, **kw):
    """Upper bound F_up on the flux of every covered torus (module docstring) from hyperbolic
    symmetric closed lines beyond the outermost verified torus T_last whose manifolds reach the
    lost region.  Returns a JSON-serialisable dict:
      ok, kind="edge_barrier", F_up, F_last, request, holds_request (F_up < request),
      orbit (q, p, residue, section points, e_X), lost_point (the proof record), crossings,
      error_budget, evidence (reference checks), candidates tried, reason if not ok;
      certificate ("lost_ring" or "manifold"), escape_map (T_min_ring: smallest T with a loop of cells
      lost within T periods whose flux estimate is <= ring_flux_share x request, None if none; time),
      ring (the lost_ring record when it was tried).  try_ring=False skips the escape map.
    Also: curve_class ("A" / "B": box crossings; "rev:u+" / "rev:u-": reversible closing), F_case_i / F_case_ii
    and closed_families (generalised Lemma B), plane_phi (the symmetry plane of the certificate), q_edge,
    deadline_hit (the stage the hard deadline stopped), timings.
    Options of this phase: reversible (the reversible closing and generalised Lemma B), min_rev_residue (weak
    symmetric orbits screened after the strong ones), edge_first (the profile edge's q first), other_plane
    (orbits without a mirror pair of neighbours at phi = 0 tried at phi = pi / nfp, lost proofs carried over),
    extend_shoot, cand_share / cand_min (per-candidate time), skip_factor (orbits whose chain polygon encloses
    more than skip_factor x the request are not screened), scan_ratio (with a request the scan stops at
    sqrt(scan_ratio x request / F_last) x T_last's distance from the axis), grace (hard deadline = deadline +
    grace, default max(10 s, 3 % of the budget))."""
    t0 = time.time()
    out = dict(kind="edge_barrier", ok=False, F_last=float(abs(T_last.flux)), request=request,
               iota_last=float(T_last.iota))
    try:
        # T_last must be a covered torus: its section a simple closed curve (T3).  (CoilStellaration
        # DDMaikF39a8 after a 1/3 jump: theta = 0 and pi at the same point, the curve winds -2 times
        # around the axis -- the scale and the axis search below would be meaningless.)
        th_ = 2 * np.pi * np.arange(1024) / 1024
        T_poly = np.stack(T_last.geometry_on(th_, np.zeros(1024)), -1)
        R2c, _ = T_last.geometry_on(np.array([0.0, np.pi]), np.zeros(2))
        if not (closed_polygon_simple(T_poly, 1e-12 * float(np.max(np.abs(T_poly[:, 0]))))
                and abs(R2c[0] - R2c[1]) > 1e-9 * abs(R2c[0])):
            out["reason"] = "T_last's section at phi = 0 is not a simple closed curve (not a covered torus)"
            return _jsonable(out)
        sec, R2 = _setup_section(engine, source, T_last)
        if deadline is not None:
            # every field evaluation of the section raises past the hard deadline (the stages' own soft checks
            # stop at `deadline`; the grace covers the final reference checks of a certificate)
            g_ = grace if grace is not None else max(10.0, 0.03 * max(deadline - t0, 0.0))
            sec.hard_deadline = deadline + g_
            _HARD[0] = sec.hard_deadline
        out["tracer_steps"] = getattr(sec, "calibration", None)
        ax = find_axis_point(sec, R2[1] + 0.1 * (R2[0] - R2[1]), R2[0] - 0.1 * (R2[0] - R2[1]))
        if ax is None:
            out["reason"] = "magnetic axis point not found inside T_last"
            return _jsonable(out)
        a0 = ax.pts[0]
        out["axis_point"] = a0.tolist()
        out["axis_residue"] = ax.residue
        th_ = 2 * np.pi * np.arange(1024) / 1024
        if not inside_polygon(np.stack(T_last.geometry_on(th_, np.zeros(1024)), -1), a0[None])[0]:
            out["reason"] = "the elliptic period-one line found is not inside T_last"
            return _jsonable(out)
        nfp = source.nfp
        out["certificate"] = None
        if try_ring:
            # the lost ring first when the escape-time map shows a loop of quickly lost cells around
            # the axis (no orbit search needed); otherwise / if it fails: the manifold barrier
            tr0 = time.time()
            emap = escape_map(sec, a0, T_last, n_theta=96, n_r=32, nper=ring_nper,
                              exclude=(coil_crossings(source), 0.03 * sec.scale))
            Wr = ring_wedge_flux(sec, emap)
            bud = None if request is None else ring_flux_share * abs(request)
            Tmin = next((T_ for T_ in range(1, emap["nper"] + 1)
                         if ring_cells(emap, T_, W=Wr, budget=bud)[0] is not None), None)
            out["escape_map"] = dict(T_min_ring=Tmin, quick_fraction=float(np.mean(emap["esc"] <= ring_T_quick)),
                                     time=time.time() - tr0)
            if Tmin is not None and Tmin <= ring_T_quick:
                dl_ring = None if deadline is None else time.time() + ring_share * max(deadline - time.time(), 0.0)
                rr = lost_ring(sec, a0, T_last, request=request, emap=emap, T_quick=Tmin, T_max=ring_T_quick,
                               deadline=dl_ring, verbose=verbose, flux_share=ring_flux_share)
                out["ring"] = {k: v for k, v in rr.items() if k not in ("polygon", "disks")}
                if verbose:
                    print(f"  lost ring: ok {rr['ok']} F_up {rr.get('F_up')} {rr.get('reason', '')} "
                          f"({time.time() - tr0:.0f}s)", flush=True)
                if rr["ok"]:
                    fr = reference_flux(sec, np.array(rr["polygon"]))
                    if fr.get("ok") and rr["F_up"] > abs(T_last.flux):
                        out.update(ok=True, certificate="lost_ring", F_up=rr["F_up"], polygon=rr["polygon"],
                                   disks=rr["disks"], evidence=dict(flux_reference=fr,
                                                                    disks=rr.get("disk_evidence")))
                        out["holds_request"] = bool(request is not None and rr["F_up"] < abs(request))
                        if request is not None:
                            out["margin_rel"] = float((abs(request) - rr["F_up"]) / abs(request))
                        out.setdefault("timings", {})["total"] = time.time() - t0
                        return _jsonable(out)
                    out["ring"]["rejected"] = "reference flux or F_up <= F_last"
        reach_scan = 0.7
        if qs is None:
            Rp, rho = rotation_profile(sec, T_last, a0)
            if np.all(np.isfinite(rho)):
                # no line lost within 0.7 x the coil distance: the edge lies further out (e.g. QUASR 1682551, a
                # request 14 x F_last): the profile and the orbit scan reach 0.92 x the coil distance
                Rp, rho = rotation_profile(sec, T_last, a0, reach=0.92)
                reach_scan = 0.92
            out["reach"] = reach_scan
            out["rotation_profile"] = dict(R=Rp.tolist(), rho=[None if not np.isfinite(v) else float(v) for v in rho])
            q_all = qs_from_profile(abs(T_last.iota) / nfp, rho, q_max=q_max, n_q=n_q * max_rounds)
            if not q_all:
                from .chains import rationals_between
                r0 = abs(T_last.iota) / nfp
                rats = sorted(rationals_between(r0 * (1 - window), r0 * (1 + window), q_max), key=lambda t: t[1])
                q_all = list(dict.fromkeys(q for (p, q) in rats))[:n_q * max_rounds]
            # the chains at the profile's edge first, then the profile's rationals from T_last outward
            rho_last = abs(T_last.iota) / nfp
            q_edge = qs_edge_window(rho, rho_last=rho_last, q_max=q_max, n_q=n_q) if edge_first else []
            out["q_edge"] = q_edge
            q_prof = list(q_all)
            if len(q_prof) < n_q * max_rounds:
                # short lists (e.g. every profile line lost at once: the edge is T_last): the lowest-order
                # rationals in growing windows around the edge's rotation number
                i_l = next((i for i, v in enumerate(rho) if not np.isfinite(v)), len(rho))
                rho_ref = float(rho[i_l - 1]) if i_l > 0 else rho_last
                q_prof = qs_extend(q_prof, rho_ref, n_q * max_rounds, q_max=q_max)
            # the first round: the edge window's chains and the profile's first ones (the orbit search is cheap
            # next to the candidates; the candidates are then tried innermost first whatever their q)
            r1 = list(dict.fromkeys(list(q_edge[:3]) + q_prof[:2]))
            rest = [q for q in q_prof if q not in r1]
            rounds = [r1] + [rest[i:i + n_q] for i in range(0, len(rest), n_q)]
            if verbose:
                print(f"  rotation profile beyond T_last: {np.round(rho, 5)} -> rounds {rounds} (edge {q_edge})",
                      flush=True)
            eq = [int(q) for q in (extra_qs or ()) if int(q) >= 1]
            if eq:
                # the chain the family could not pass (the caller knows it; its q may lie above q_max or outside
                # the window of the rotation profile): scanned first, then the profile's own rounds
                rounds = [r for r in [eq] + [[q for q in r_ if q not in eq] for r_ in rounds] if r]
        else:
            rounds = [list(qs)]
        if orbits is not None:
            rounds = rounds[:1]
        out["q_scanned"] = []
        out["orbits_found"] = []
        out["screened"] = []
        out["timings"] = {}
        tried = []
        best = None
        sym_ok = bool(getattr(source, "stellsym", False)) and reversible
        for rnd, qs_r in enumerate(rounds[:max_rounds]):
            # iterative deepening: the next q values only if this round certified nothing that holds
            if deadline is not None and time.time() > deadline:
                break
            out["q_scanned"] += list(qs_r)
            try:
                if orbits is None:
                    mr_ = None if (request is None or not scan_ratio) else \
                        math.sqrt(scan_ratio * abs(request) / max(abs(T_last.flux), 1e-300))
                    hyp, allc = orbit_candidates(sec, T_last, a0, qs_r, n_scan=n_scan, deadline=deadline, reach=reach_scan,
                                                 max_ratio=mr_,
                                                 verbose=verbose)
                else:                                        # orbits from an earlier search (records as below)
                    allc = []
                    for d in orbits:
                        o = Orbit(int(d["q"]), np.array(d["points"]), np.array(d["M"]), e_X=float(d["e_X"]), p=d["p"],
                                  info=dict(d["info"]))
                        allc.append(o)
                    hyp = [o for o in allc if (o.residue < -1e-6 or o.residue > 1 + 1e-6) and o.info["ratio_min"] > 1.0]
                    hyp.sort(key=lambda o: o.info["ratio_mean"])
            except BarrierDeadline:
                out["deadline_hit"] = "orbit search"
                break
            # weakly hyperbolic X points belong to thin (near-integrable) chains: their manifolds split by tiny
            # angles, so box crossings cannot be certified -- the reversible closing needs none (exact
            # heteroclinic points on Z = 0), so for a symmetric source they are kept down to min_rev_residue,
            # after the strongly hyperbolic orbits of the chaotic edge
            strong = [o for o in hyp if abs(o.residue) >= min_abs_residue and abs(1 - o.residue) >= min_abs_residue]
            sid_ = {id(o) for o in strong}
            weak = [o for o in hyp if id(o) not in sid_ and sym_ok and o.residue < 0
                    and abs(o.residue) >= min_rev_residue]
            out["orbits_found"] += [dict(q=o.q, p=o.p, residue=o.residue, e_X=o.e_X, points=o.pts.tolist(),
                                         M=o.M.tolist(), info={k: v for k, v in o.info.items() if np.isscalar(v)})
                                    for o in allc]
            if on_orbits is not None:
                on_orbits(_jsonable(out["orbits_found"]))
            out["timings"][f"search{rnd}"] = time.time() - t0
            # orbits whose chain polygon (the X points in angular order) already encloses far more than the request
            # cannot give a curve holding it (the closed curves run next to the X points): not screened
            if request is not None and skip_factor:
                keep_s, keep_w = [], []
                for lst, keep in ((strong, keep_s), (weak, keep_w)):
                    for o in lst:
                        Fp = np.nan
                        if o.q >= 5:
                            try:
                                ang = np.arctan2(o.pts[:, 1] - a0[1], o.pts[:, 0] - a0[0])
                                Fp = abs(sec.flux(o.pts[np.argsort(ang)])[0])
                            except BarrierDeadline:
                                raise
                            except Exception:  # noqa: BLE001
                                Fp = np.nan
                        o.info["polygon_flux"] = float(Fp)
                        if np.isfinite(Fp) and Fp > skip_factor * abs(request):
                            out.setdefault("skipped_far", []).append(dict(q=o.q, residue=o.residue, polygon_flux=Fp))
                            continue
                        keep.append(o)
                strong, weak = keep_s, keep_w
            # screening: accurate orbits, local manifolds, one batched shooting pass for all candidates;
            # candidates with a lost manifold orbit are tried innermost first (smallest closed curves), the weak
            # ones after the strong
            cand_orbs = strong[:max_screen] + weak[:max(0, max_screen - len(strong[:max_screen]))]
            try:
                pre = screen_many(sec, cand_orbs, deadline=deadline, verbose=verbose)
            except BarrierDeadline:
                out["deadline_hit"] = "screening"
                break
            out["timings"][f"screen{rnd}"] = time.time() - t0
            ranked = [p_ for p_ in pre if p_.get("ok") and p_.get("lost_c")]
            ranked.sort(key=lambda p_: (abs(p_["orb"].residue) < min_abs_residue, p_["orb"].info.get("ratio_mean", np.inf)))
            out["screened"] += [dict(q=p_["orb"].q, residue=p_["orb"].residue, lost=sorted(p_.get("lost_c", {})),
                                     ratio=p_["orb"].info.get("ratio_mean"),
                                     start_ok={k: bool(v.get("ok")) for k, v in p_.get("loc", {}).items()},
                                     delta0={k: v.get("delta0") for k, v in p_.get("loc", {}).items()},
                                     e_X=p_["orb"].e_X) if p_.get("ok") else
                                dict(reason=p_.get("reason")) for p_ in pre]
            n_after = 0
            hit = False
            phases = [ranked[:max_candidates]]
            ph = 0
            while ph < len(phases):
                queue = phases[ph]
                ph += 1
                for i_c, p_ in enumerate(queue):
                    orb = p_["orb"]
                    if deadline is not None and time.time() > deadline:
                        break
                    # a candidate that is not the last one in line gets at most cand_share of what is left (at least
                    # cand_min s): one candidate growing manifolds to the deadline left none for the next (the
                    # certified orbit of 234994 was the fifth tried)
                    dl_c = deadline
                    if deadline is not None and i_c < len(queue) - 1:
                        left_ = deadline - time.time()
                        dl_c = min(deadline, time.time() + max(cand_min, cand_share * left_))
                    if dl_c is not None:
                        # the final reference closure of a certified orbit traces q periods with the reference
                        # tracer (~0.5 s a period: QUASR 660970's q = 60 certificate ended 27 s past the budget)
                        dl_c = dl_c - 0.5 * orb.q
                    if verbose:
                        print(f" candidate q {orb.q} p {orb.p} residue {orb.residue:.4g} ratio {orb.info['ratio_mean']:.4f}",
                              flush=True)
                    sec_r = sec
                    try:
                        finalize_screened(sec, p_, deadline=dl_c, verbose=verbose)
                        orb = p_["orb"]
                        r = barrier_from_orbit(sec, orb, a0, want="edge", deadline=dl_c, verbose=verbose, screened=p_,
                                               request=request, reversible=reversible, a_err=float(ax.e_X),
                                               lost_deadline=deadline, **kw)
                        if (not r["ok"] and sym_ok and other_plane and r.get("reversible")
                                and r.get("lost_point") is not None and (deadline is None or time.time() < deadline)):
                            if verbose:
                                print(f"  candidate q {orb.q}: no mirror pair of neighbours at phi = 0 -> phi = pi / nfp",
                                      flush=True)
                            la_ = r.get("lost_attempts") or {}
                            r2, s2 = candidate_in_other_plane(sec, orb, a0, deadline=dl_c, verbose=verbose,
                                                              request=request, reversible=reversible,
                                                              lost_names=[k_ for k_, v_ in la_.items() if v_.get("ok")],
                                                              lost_records=la_, **kw)
                            r["other_plane"] = {k_: v_ for k_, v_ in r2.items() if k_ in ("ok", "reason", "F_up", "curve_class",
                                                                                       "closed_families", "excluded")}
                            if r2.get("ok"):
                                r, sec_r = r2, s2
                    except BarrierDeadline:
                        r = dict(ok=False, reason="hard deadline during the candidate", timings={})
                        hit = True
                    r["_sec"] = sec_r
                    tried.append(dict(q=orb.q, p=orb.p, residue=orb.residue, ok=r["ok"], reason=r.get("reason"),
                                      plane_phi=r.get("plane_phi", 0.0), other_plane=r.get("other_plane"),
                                      F_up=r.get("F_up"), time=r.get("timings", {}).get("total"),
                                      timings=r.get("timings"), curve_class=r.get("curve_class"),
                                      lost_attempts=r.get("lost_attempts"), excluded=r.get("excluded"),
                                      lemma_B=r.get("lemma_B"), evidence=r.get("evidence"),
                                      closed_families=r.get("closed_families"), reversible=r.get("reversible"),
                                      growth=r.get("growth"),
                                      curves={k: {kk: vv for kk, vv in v.items() if kk in ("ok", "reason", "F_up", "eps_max",
                                                                                           "loop_cuts", "flux_estimate",
                                                                                           "crossing", "junction_err_max")}
                                              for k, v in (r.get("curves") or {}).items()}))
                    if best is not None:
                        n_after += 1                         # candidates tried after the first success
                    if r["ok"] and (best is None or r["F_up"] < best["F_up"]):
                        best = r
                    holds = best is not None and (request is None or best["F_up"] < abs(request))
                    if hit or (holds and n_after >= extra_after_success):
                        break
                holds = best is not None and (request is None or best["F_up"] < abs(request))
                if (not holds and not hit and ph == len(phases) and ph < 2 and extend_shoot
                        and (deadline is None or time.time() < deadline - 0.2 * max(deadline - t0, 0.0))):
                    # shoot the branches that were not lost (screen_many's detection lines) longer: candidates whose
                    # manifolds reach the lost region only after a long passage through the chaotic layer
                    try:
                        dl_x = None if deadline is None else time.time() + 0.3 * (deadline - time.time())
                        gained = extend_shooting(sec, pre, deadline=deadline, deadline_detect=dl_x, verbose=verbose)
                    except BarrierDeadline:
                        gained = []
                        hit = True
                    out.setdefault("extended_shooting", []).append(dict(round=rnd, gained=len(gained)))
                    if gained:
                        gained.sort(key=lambda p_: (abs(p_["orb"].residue) < min_abs_residue,
                                                    p_["orb"].info.get("ratio_mean", np.inf)))
                        out["screened"] += [dict(q=p_["orb"].q, residue=p_["orb"].residue, lost=sorted(p_["lost_c"]),
                                                 ratio=p_["orb"].info.get("ratio_mean"), extended=True) for p_ in gained]
                        phases.append(gained[:max_candidates])
            if hit:
                out["deadline_hit"] = "candidate"
                break
            if best is not None and (request is None or best["F_up"] < abs(request)):
                break
        out["candidates"] = tried
        if best is None:
            any_hyp = any(sc.get("q") is not None for sc in out.get("screened", []))
            out["reason"] = ("no candidate orbit gave a certified barrier" if tried else
                             "no screened orbit has a lost manifold orbit" if any_hyp else
                             f"deadline during the {out['deadline_hit']}" if out.get("deadline_hit") else
                             "no strongly hyperbolic orbit found beyond T_last")
        else:
            for k in ("F_up", "orbit", "lost_point", "crossings", "error_budget", "evidence", "excluded", "lemma_B",
                      "curve_class", "start_error", "polygon", "curves", "growth", "F_case_i", "F_case_ii",
                      "closed_families"):
                out[k] = best.get(k)
            out["ok"] = True
            out["certificate"] = "manifold"
            out["holds_request"] = bool(request is not None and best["F_up"] < abs(request))
            if request is not None:
                out["margin_rel"] = float((abs(request) - best["F_up"]) / abs(request))
            sec_b = best.pop("_sec", sec)
            out["plane_phi"] = float(sec_b.phi0)
            out["evidence"] = dict(best.get("evidence", {}))
            out["evidence"]["closure_reference"] = reference_closure(sec_b, best["orbit"])
            out["evidence"]["flux_reference"] = reference_flux(sec_b, np.array(best["polygon"]))
            okref = out["evidence"]["closure_reference"].get("ok", True) and out["evidence"]["flux_reference"].get("ok", True)
            if not okref:
                out["ok"] = False
                out["reason"] = "reference checks failed"
            elif best["F_up"] <= abs(T_last.flux):
                # T_last is a covered torus: a bound below its flux is a contradiction (bad input or bug)
                out["ok"] = False
                out["holds_request"] = False
                out["reason"] = "inconsistent: F_up below the flux of T_last"
    except BarrierDeadline as e:
        out["reason"] = f"hard deadline: {e}"
        out["deadline_hit"] = out.get("deadline_hit", "setup")
    except Exception as e:  # noqa: BLE001
        from .envfail import environment_failure
        if environment_failure(e):
            _HARD[0] = None
            raise
        import traceback
        out["reason"] = f"error: {e!r}"
        out["traceback"] = traceback.format_exc()
    finally:
        _HARD[0] = None
    out.setdefault("timings", {})["total"] = time.time() - t0
    return _jsonable(out)


def reference_closure(sec, orbit_rec):
    """|P_ref^q(X_0) - X_0| on the independent reference field/tracer (forward; one line)."""
    try:
        from .reference import trace as RT
        X0 = np.array(orbit_rec["points"][0])
        q = int(orbit_rec["q"])
        P = 2 * np.pi / sec.nfp
        ph = q * P
        nt = int(math.floor(ph / (2 * np.pi) - 1e-12)) + 1
        rel = ph - 2 * np.pi * (nt - 1)
        ph0 = float(getattr(sec, "phi0", 0.0))
        o = RT.poincare(sec.ref_field(), [X0[0]], [X0[1]], [ph0 + rel], nt, phi0=ph0, h=2 * np.pi / 512)
        y = np.array([o["R"][0, -1, 0], o["Z"][0, -1, 0]])
        d = float(np.linalg.norm(y - X0))
        # distance from X_0 to the reference field's periodic orbit (Newton-Kantorovich, hyperbolic M)
        inv = float(orbit_rec.get("info", {}).get("inv_norm", np.inf))
        e_ref = 2.0 * inv * (d + float(o["err"][0, -1, 0]))
        tol = max(10.0 * float(orbit_rec.get("e_X", 0.0)), 1e-12 * sec.scale)
        return dict(ok=bool(np.isfinite(e_ref) and e_ref <= tol), closure=d, e_X_reference=e_ref, tolerance=tol,
                    ref_err=float(o["err"][0, -1, 0]))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=repr(e))


def reference_flux(sec, C):
    """Loop integral of the reference field's A along the polygon vs the engine's."""
    try:
        F1, q1 = sec.flux(C)
        F2, q2 = sec.flux(C, field=sec.ref_field())
        rel = abs(F1 - F2) / max(abs(F1), 1e-300)
        return dict(ok=bool(rel < 1e-10), flux_engine=F1, flux_reference=F2, rel_diff=rel)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=repr(e))


def band_barrier(engine, source, X_line, T_inside=None, axis_point=None, request=None, deadline=None,
                 verbose=False, **kw):
    """Flux band free of covered tori from both closed curves of manifold arcs of a hyperbolic orbit
    (module docstring, manifold band).  X_line: a torux.orbits.ClosedLine, or (q, X_0) with X_0 the
    orbit's point in phi = 0.  The axis point comes from axis_point, else from a torus T_inside (any
    torus around the axis, e.g. the last one inside the layer).  Returns a JSON-serialisable dict:
    ok, kind="manifold_band", interval=(F_in, F_out), holds_request, orbit, curves, evidence."""
    t0 = time.time()
    out = dict(kind="manifold_band", ok=False, request=request)
    try:
        if hasattr(X_line, "section_points"):
            q = int(X_line.q)
            X0 = X_line.section_points(0.0)[0]
        else:
            q, X0 = int(X_line[0]), np.asarray(X_line[1], float)
        if T_inside is not None:
            sec, R2 = _setup_section(engine, source, T_inside)
            if axis_point is None:
                ax = find_axis_point(sec, R2[1] + 0.1 * (R2[0] - R2[1]), R2[0] - 0.1 * (R2[0] - R2[1]))
                axis_point = None if ax is None else ax.pts[0]
        else:
            B = np.asarray(engine.evaluate(np.array([[axis_point[0], 0.0, axis_point[1]]]), grad=False, A=False)["B"])
            sec = FieldSection(engine, source, sign=1.0 if B[0, 1] > 0 else -1.0,
                               scale=2.0 * float(np.linalg.norm(np.asarray(X0) - np.asarray(axis_point))))
        if axis_point is None:
            out["reason"] = "no axis point"
            return _jsonable(out)
        a0 = np.asarray(axis_point, float)
        orb = orbit_from_point(sec, X0, q)
        if orb is None:
            out["reason"] = "orbit lost"
            return _jsonable(out)
        orb.p = rotation_number(sec, orb.pts[0], a0, q)
        r = barrier_from_orbit(sec, orb, a0, want="band", deadline=deadline, verbose=verbose, **kw)
        out.update({k: v for k, v in r.items() if k not in ("kind",)})
        if r["ok"]:
            out["evidence"] = dict(r.get("evidence", {}))
            out["evidence"]["closure_reference"] = reference_closure(sec, r["orbit"])
            fr = {k: reference_flux(sec, np.array(v)) for k, v in r["polygons"].items()}
            out["evidence"]["flux_reference"] = fr
            if request is not None:
                out["holds_request"] = bool(r["interval"][0] < abs(request) < r["interval"][1])
            if not (out["evidence"]["closure_reference"].get("ok") and all(v.get("ok") for v in fr.values())):
                out["ok"] = False
                out["reason"] = "reference checks failed"
    except Exception as e:  # noqa: BLE001
        import traceback
        out["reason"] = f"error: {e!r}"
        out["traceback"] = traceback.format_exc()
    out.setdefault("timings", {})["total_band"] = time.time() - t0
    return _jsonable(out)


# ============================================================================ lost ring
# THEOREM (lost ring).  Let D_1..D_n be closed disks in Sigma, every point of each lost (its field line
# reaches B_phi < 0), and a closed polygon Q in their union U (certified exactly: each edge of Q is
# covered by two of the disks), Q simple with winding number 1 around the axis point a.  For a covered torus, D-bar contains no lost
# point (Lemma A), so D misses U, hence Q; D is connected and contains a, so D lies inside Q:
# flux(Gamma) <= Phi(Q) when B_phi >= 0 inside Q.  No manifold, no orbit: only the disk proofs.
def _arclength_lines(Bfun, x0, sig, S_max, stop=None, H0=1e-3, tol=1e-12, k=6, H_max=0.02, max_steps=20000):
    """Field lines in arc length with a per-line end S_max (and an optional stop condition); returns
    dict(x, s, stopped, err)."""
    from .fasttrace import gbs_step
    x = np.array(x0, dtype=float).reshape(-1, 3)
    L = len(x)
    sig = np.broadcast_to(np.asarray(sig, float), (L,)).copy()
    S_max = np.broadcast_to(np.asarray(S_max, float), (L,)).copy()
    s = np.zeros(L)
    H = np.full(L, float(H0))
    err = np.zeros(L)
    stopped = np.zeros(L, bool)
    done = S_max <= 0
    it = 0
    while not np.all(done) and it < max_steps:
        it += 1
        act = np.flatnonzero(~done)
        Ha = np.minimum(H[act], S_max[act] - s[act])

        def fa(t, xx, _sg=sig[act]):
            n = len(xx) // len(act)
            B = np.asarray(Bfun(xx), float)
            Bm = np.sqrt(np.sum(B * B, -1))
            ok = np.isfinite(Bm) & (Bm > 0)
            return np.tile(_sg, n)[:, None] * B / np.where(ok, Bm, 1.0)[:, None], ok
        xn, e, ok = gbs_step(fa, s[act], x[act], Ha, k)
        acc = ok & (e <= tol) & np.all(np.isfinite(xn), -1)
        ia = act[acc]
        x[ia] = xn[acc]
        s[ia] += Ha[acc]
        err[ia] += e[acc]
        fac = np.where(ok & (e > 0), 0.9 * (tol / np.maximum(e, 1e-300)) ** (1.0 / (2 * k - 1)), 3.0)
        H[act] = np.minimum(Ha * np.where(ok, np.clip(fac, 0.2, 3.0), 0.25), H_max)
        if stop is not None and len(ia):
            st = np.asarray(stop(x[ia]), bool)
            stopped[ia[st]] = True
            done[ia[st]] = True
        done |= s >= S_max * (1 - 1e-12)
        done |= H < 1e-14
    return dict(x=x, s=s, stopped=stopped, err=err)


def _arclength_to_min(Bfun, ffun, x0, sig, S_max, beta, extra=0.5, rise=0.02, H0=1e-3, tol=1e-9, k=4, H_max=0.1,
                      max_steps=50000):
    """Lines in arc length (adaptive GBS) until f = ffun(x) <= -beta, then on to the minimum of f
    (stopped once f exceeds its running minimum by rise, extra after first reaching -beta, or at
    S_max).  Returns dict(lost (reached -beta), S and f at the minimum, x there)."""
    from .fasttrace import gbs_step
    x = np.array(x0, dtype=float).reshape(-1, 3)
    L = len(x)
    sig = np.broadcast_to(np.asarray(sig, float), (L,)).copy()
    S_max = np.broadcast_to(np.asarray(S_max, float), (L,)).copy()
    s = np.zeros(L)
    H = np.full(L, float(H0))
    reached = np.zeros(L, bool)
    s_beta = np.full(L, np.nan)
    fmin = np.full(L, np.inf)
    smin = np.full(L, np.nan)
    xmin = np.full((L, 3), np.nan)
    done = S_max <= 0
    it = 0
    while not np.all(done) and it < max_steps:
        it += 1
        act = np.flatnonzero(~done)
        Ha = np.minimum(H[act], S_max[act] - s[act])

        def fa(t, xx, _sg=sig[act]):
            n = len(xx) // len(act)
            B = np.asarray(Bfun(xx), float)
            Bm = np.sqrt(np.sum(B * B, -1))
            ok = np.isfinite(Bm) & (Bm > 0)
            return np.tile(_sg, n)[:, None] * B / np.where(ok, Bm, 1.0)[:, None], ok
        xn, e, ok = gbs_step(fa, s[act], x[act], Ha, k)
        acc = ok & (e <= tol) & np.all(np.isfinite(xn), -1)
        ia = act[acc]
        x[ia] = xn[acc]
        s[ia] += Ha[acc]
        fac = np.where(ok & (e > 0), 0.9 * (tol / np.maximum(e, 1e-300)) ** (1.0 / (2 * k - 1)), 3.0)
        H[act] = np.minimum(Ha * np.where(ok, np.clip(fac, 0.2, 3.0), 0.25), H_max)
        if len(ia):
            f = np.asarray(ffun(x[ia]), float)
            nw = (f <= -beta) & ~reached[ia]
            reached[ia[nw]] = True
            s_beta[ia[nw]] = s[ia[nw]]
            up = reached[ia] & (f < fmin[ia])
            fmin[ia[up]] = f[up]
            smin[ia[up]] = s[ia[up]]
            xmin[ia[up]] = x[ia[up]]
            st = reached[ia] & ((f > fmin[ia] + rise) | (s[ia] > s_beta[ia] + extra))
            done[ia[st]] = True
        done |= s >= S_max * (1 - 1e-12)
        done |= H < 1e-14
    return dict(lost=reached, S=smin, f=np.where(reached, fmin, np.nan), x=xmin, s_beta=s_beta)


def coil_crossings(source, n=1024):
    """(R, Z) of the points where Fourier coils cross the half plane phi = 0 (y = 0, x > 0)."""
    out = []
    for p in source.parts:
        if p.kind != "fourier":
            continue
        xs, _ = p.points(np.arange(n) / n)
        for c in xs:
            y = c[:, 1]
            yn = np.roll(y, -1)
            i = np.flatnonzero((y * yn <= 0) & (y != yn))
            for k in i:
                t = y[k] / (y[k] - yn[k])
                q = c[k] + t * (np.roll(c, -1, axis=0)[k] - c[k])
                if q[0] > 0:
                    out.append([q[0], q[2]])
    return np.array(out).reshape(-1, 2)


def escape_map(sec, a, T_last, n_theta=128, n_r=40, reach=0.7, nper=4, exclude=None):
    """Escape times (field periods until the coarse map stops the line, forward or backward; inf if
    not within nper) on a polar grid around a, from T_last's section outward by reach x the distance to
    the nearest coil point (quadratic radial spacing: dense next to T_last).  Detection only (the
    coarse map also stops lines whose large steps leave the field's domain).  exclude: (points (k, 2),
    radius) of cells left out (coil crossings)."""
    a = np.asarray(a, float)
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    T_poly = np.stack(T_last.geometry_on(2 * np.pi * np.arange(1024) / 1024, np.zeros(1024)), -1)
    rin = ray_ratio_dist(T_poly, a, th)
    cp = sec.source.sample_points() if hasattr(sec, "source") else None
    pts = np.zeros((n_theta, n_r, 2))
    for i, t in enumerate(th):
        u = np.array([math.cos(t), math.sin(t)])
        x_in = a + rin[i] * u
        if cp is not None and len(cp):
            dc = float(np.min(np.linalg.norm(cp - np.array([x_in[0], 0.0, x_in[1]]), axis=1)))
        else:
            dc = getattr(sec, "R_lost", 3.0) - rin[i]
        rr = rin[i] * 1.0 + reach * dc * np.linspace(0.0, 1.0, n_r + 1)[1:] ** 2
        pts[i] = a + rr[:, None] * u
    Y = pts.reshape(-1, 2)
    det = sec.coarse() if hasattr(sec, "coarse") else sec
    esc = np.full(len(Y), np.inf)
    for d in (+1, -1):
        y = Y.copy()
        alive = np.ones(len(Y), bool)
        for n in range(nper):
            idx = np.flatnonzero(alive)
            if len(idx) == 0:
                break
            yy, lost, _ = det.step(y[idx], np.full(len(idx), d))
            y[idx] = yy
            newly = idx[lost]
            esc[newly] = np.minimum(esc[newly], n + 1)
            alive[newly] = False
    excl = np.zeros(len(Y), bool)
    if exclude is not None and len(exclude[0]):
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(exclude[0]).query(Y, k=1)
        excl = dd < exclude[1]
    return dict(theta=th, pts=pts, esc=esc.reshape(n_theta, n_r), excluded=excl.reshape(n_theta, n_r), nper=nper,
                a=a, rin=rin)


def ray_ratio_dist(T_poly, a, th):
    """Distance from a to the polygon along each ray angle (largest crossing)."""
    Pc = np.concatenate([T_poly, T_poly[:1]])
    far = 10.0 * (np.max(np.linalg.norm(T_poly - a, axis=1)) + 1.0)
    out = np.zeros(len(th))
    for i, t in enumerate(th):
        e = a + far * np.array([math.cos(t), math.sin(t)])
        I, J, tt, u = seg_intersections(np.stack([a, e]), Pc)
        out[i] = far * float(np.max(tt)) if len(tt) else 0.0
    return out


def ring_wedge_flux(sec, emap, n_q=48):
    """W[i, j]: flux estimate of the wedge of angular width 2 pi / n_theta from the axis point a to the
    grid point (i, j) of the escape map (density B_phi on n_q radial Gauss points per cell ray; for the
    choice of the ring path only)."""
    a = np.asarray(emap["a"])
    pts = emap["pts"]
    nt, nr = pts.shape[:2]
    rad = np.linalg.norm(pts - a[None, None], axis=-1)
    u = (pts[:, 0] - a[None]) / rad[:, :1]
    g, w = np.polynomial.legendre.leggauss(n_q)
    W = np.zeros((nt, nr))
    dth = 2 * np.pi / nt
    edges = np.concatenate([np.zeros((nt, 1)), rad], axis=1)
    for j in range(nr):
        lo, hi = edges[:, j], edges[:, j + 1]
        rr = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * g[None]
        xy = a[None, None] + rr[..., None] * u[:, None, :]
        dens = np.asarray(sec.density(xy.reshape(-1, 2)), float).reshape(nt, n_q)
        W[:, j] = (W[:, j - 1] if j else 0.0) + dth * 0.5 * (hi - lo) * np.sum(w[None] * dens * rr, axis=1)
    return W


def ring_cells(emap, T, blocked=None, W=None, budget=None, lams=(0.0, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6)):
    """A short robust closed loop around the axis through grid cells with escape time <= T (not
    excluded, not blocked): Dijkstra on the unwrapped polar grid (steps to the next angle with radial
    index change <= 1, and radial steps); cost of a step = its length x (1 + (esc - 1) + 4 x the
    fraction of the 8 neighbours that are not usable), plus lam x the wedge flux W of the cell entered
    by a step to the next angle (the sum of these is the loop's flux estimate).  With a budget, lam
    increases through lams until the loop's flux estimate is <= budget.  A loop from column 0, row j
    back to column n_theta, row j.  Returns (list of (i, j), flux estimate) or (None, None)."""
    import heapq
    esc, excl, pts = emap["esc"], emap["excluded"], emap["pts"]
    nt, nr = esc.shape
    ok = (esc <= T) & ~excl
    if blocked is not None:
        ok &= ~blocked
    okp = np.pad(ok.astype(float), ((0, 0), (1, 1)))
    nb = np.zeros((nt, nr))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += np.roll(okp, di, axis=0)[:, 1 + dj:1 + dj + nr]
    pen = 1.0 + np.where(np.isfinite(esc), esc - 1.0, 0.0) + 4.0 * (1.0 - nb / 8.0)
    Wz = np.zeros((nt, nr)) if W is None else np.asarray(W)

    # step costs as nested lists (pure-Python Dijkstra)
    okl = ok.tolist()
    Lr = np.linalg.norm(np.diff(pts, axis=1), axis=-1)                       # (nt, nr-1): (i, j) -> (i, j+1)
    Lc = np.full((nt, nr, 3), np.inf)                                        # (i, j) -> (i+1, j+dj)
    for q_, dj in enumerate((-1, 0, 1)):
        jj = np.arange(nr) + dj
        v = (jj >= 0) & (jj < nr)
        Lc[:, v, q_] = np.linalg.norm(np.roll(pts, -1, axis=0)[:, jj[v]] - pts[:, v], axis=-1)
    cr = (Lr * np.roll(pen, 0, axis=0)[:, 1:]).tolist()                      # cost of entering (i, j+1)
    cl_ = (Lr * pen[:, :-1]).tolist()                                        # cost of entering (i, j) from j+1
    penr = np.roll(pen, -1, axis=0)
    Wr = np.roll(Wz, -1, axis=0)

    def loop(lam):
        cc = []
        for q_, dj in enumerate((-1, 0, 1)):
            jj = np.clip(np.arange(nr) + dj, 0, nr - 1)
            cc.append((Lc[:, :, q_] * penr[:, jj] + lam * Wr[:, jj]).tolist())
        best = None
        for j0 in range(nr):
            if not okl[0][j0]:
                continue
            dist = {(0, j0): 0.0}
            prev = {}
            hq = [(0.0, 0, j0)]
            found = None
            while hq:
                dcur, i, j = heapq.heappop(hq)
                if dcur > dist.get((i, j), math.inf):
                    continue
                if i == nt and j == j0:
                    found = dcur
                    break
                if best is not None and dcur >= best[0]:
                    break
                im = i % nt
                cand = []
                if j > 0:
                    cand.append((i, j - 1, cl_[im][j - 1]))
                if j < nr - 1:
                    cand.append((i, j + 1, cr[im][j]))
                if i < nt:
                    for q_, dj in enumerate((-1, 0, 1)):
                        jj = j + dj
                        if 0 <= jj < nr:
                            cand.append((i + 1, jj, cc[q_][im][j]))
                for (ii, jj, c_) in cand:
                    if not okl[ii % nt][jj]:
                        continue
                    nd = dcur + c_
                    if nd < dist.get((ii, jj), math.inf):
                        dist[(ii, jj)] = nd
                        prev[(ii, jj)] = (i, j)
                        heapq.heappush(hq, (nd, ii, jj))
            if found is not None and (best is None or found < best[0]):
                path = [(nt, j0)]
                while path[-1] != (0, j0):
                    path.append(prev[path[-1]])
                path = path[::-1]
                fl = float(sum(Wz[ii % nt, jj] for (ii, jj), (i_, _) in zip(path[1:], path[:-1]) if ii != i_))
                best = (found, [(i % nt, j) for (i, j) in path[:-1]], fl)
        return best
    if budget is None or W is None:
        b = loop(0.0)
        return (None, None) if b is None else (b[1], b[2])
    for lam in lams:
        b = loop(lam)
        if b is None:
            return None, None
        if b[2] <= budget:
            return b[1], b[2]
    return None, b[2]


def _segment_covered(p, q, c1, r1, c2, r2):
    """True if the segment p q lies in disk(c1, r1) union disk(c2, r2), p in the first, q in the second."""
    if np.linalg.norm(p - c1) > r1 or np.linalg.norm(q - c2) > r2:
        return False
    dv = q - p
    A = float(dv @ dv)
    if A == 0:
        return True
    B = float(2 * dv @ (p - c1))
    Cc = float((p - c1) @ (p - c1) - r1 * r1)
    disc = B * B - 4 * A * Cc
    if disc < 0:
        return False
    tb = (-B + math.sqrt(disc)) / (2 * A)
    if tb >= 1:
        return True
    x = p + tb * dv
    return bool(np.linalg.norm(x - c2) <= r2)


def ring_cycle(C, R, a, delta):
    """Shortest closed chain of disks winding once around a, consecutive disks joined by a centre
    segment inside their union (exact check, radii reduced by delta): Dijkstra on the two-sheeted
    cover of the overlap graph cut along a ray from a.  Disks containing a are not used.  Returns
    the list of disk indices (in order) or None."""
    import heapq
    from scipy.spatial import cKDTree
    C = np.asarray(C, float)
    R = np.asarray(R, float)
    n = len(C)
    if n < 3:
        return None
    Rs = R - delta
    usable = np.linalg.norm(C - a[None], axis=1) > R
    th = np.arctan2(C[:, 1] - a[1], C[:, 0] - a[0])
    ths = np.sort(th)
    gaps = np.diff(np.concatenate([ths, ths[:1] + 2 * np.pi]))
    g = int(np.argmax(gaps))
    thc = ths[g] + 0.5 * gaps[g]                                     # cut ray in the widest angular gap
    adj = [[] for _ in range(n)]
    for i, j in cKDTree(C).query_pairs(2.0 * float(np.max(Rs))):
        if not (usable[i] and usable[j]):
            continue
        if not _segment_covered(C[i], C[j], C[i], Rs[i], C[j], Rs[j]):
            continue
        dth = (th[j] - th[i] + np.pi) % (2 * np.pi) - np.pi
        u = (thc - th[i] + np.pi) % (2 * np.pi) - np.pi
        cr = 1 if (dth > 0 and 0 < u < dth) else (-1 if (dth < 0 and dth < u < 0) else 0)
        ln = float(np.linalg.norm(C[i] - C[j]))
        adj[i].append((j, cr, ln))
        adj[j].append((i, -cr, ln))
    starts = sorted({i for i in range(n) for (j, cr, _) in adj[i] if cr != 0})
    best = None
    for s0 in starts:
        dist = {(s0, 0): 0.0}
        prev = {}
        hq = [(0.0, s0, 0)]
        found = None
        while hq:
            dcur, i, sh = heapq.heappop(hq)
            if dcur > dist.get((i, sh), np.inf):
                continue
            if best is not None and dcur >= best[0]:
                break
            if i == s0 and sh == 1:
                found = dcur
                break
            for (j, cr, ln) in adj[i]:
                sj = sh + cr
                if sj < 0 or sj > 1:
                    continue
                nd = dcur + ln
                if nd < dist.get((j, sj), np.inf):
                    dist[(j, sj)] = nd
                    prev[(j, sj)] = (i, sh)
                    heapq.heappush(hq, (nd, j, sj))
        if found is not None and (best is None or found < best[0]):
            cyc = [(s0, 1)]
            while cyc[-1] != (s0, 0):
                cyc.append(prev[cyc[-1]])
            best = (found, [i for (i, _) in cyc[::-1][:-1]])
    return None if best is None else best[1]


def ring_inner_polygon(C, R, a, delta):
    """Closed polygon Q in the union of the disks (centres C, radii R shrunk by delta): vertex i on the
    ray from c_i towards a, as far in as the coverage of both adjacent segments allows (exact check)."""
    n = len(C)
    Rs = R - delta
    u = (a[None, :] - C) / np.linalg.norm(a[None, :] - C, axis=1)[:, None]
    t = 0.9 * Rs.copy()
    for _ in range(40):
        P = C + t[:, None] * u
        bad = np.zeros(n, bool)
        for i in range(n):
            k = (i + 1) % n
            if not _segment_covered(P[i], P[k], C[i], Rs[i], C[k], Rs[k]):
                bad[i] = bad[k] = True
        if not np.any(bad):
            return P, t
        t[bad] *= 0.5
        t[bad & (t < 1e-3 * Rs)] = 0.0
    P = C + t[:, None] * u
    ok = all(_segment_covered(P[i], P[(i + 1) % n], C[i], Rs[i], C[(i + 1) % n], Rs[(i + 1) % n]) for i in range(n))
    return (P, t) if ok else (None, t)


def cut_loops_exact(Q, a, max_cuts=500):
    """Remove the self-intersections of a closed polygon whose edges are exact (no error): at a crossing
    of edges i < j the polygon splits into Q[0..i] + p + Q[j+1..] and p + Q[i+1..j] (p the crossing,
    on both edges); the piece with winding number +-1 around a is kept.  Every edge of the result is a
    sub-segment of an edge of Q.  Returns the polygon or None."""
    Q = np.asarray(Q, float)
    for _ in range(max_cuts):
        P = np.concatenate([Q, Q[:1]])
        n = len(Q)
        I, J, T, U = seg_intersections(P, P)
        keep = (J > I + 1) & ~((I == 0) & (J == n - 1))
        I, J, T = I[keep], J[keep], T[keep]
        if len(I) == 0:
            return Q
        o = int(np.argmin(J - I))
        i, j, t = int(I[o]), int(J[o]), float(T[o])
        p_ = P[i] + t * (P[i + 1] - P[i])
        Qa = np.concatenate([Q[:i + 1], p_[None], Q[j + 1:]])
        Qb = np.concatenate([p_[None], Q[i + 1:j + 1]])
        if abs(winding_number(Qa, a)) == 1:
            Q = Qa
        elif abs(winding_number(Qb, a)) == 1:
            Q = Qb
        else:
            return None
    return None


def lost_ring(sec, a, T_last, request=None, emap=None, T_quick=2, T_max=None, h0=None, max_depth=3, deadline=None,
              verbose=False, coil_margin=None, path=None, max_rounds=6, n_ref=4, flux_share=0.85, S_max=None,
              chunk=256):
    """Upper bound on the flux of every covered torus from a ring of certified-lost disks around the
    axis point a (theorem above).  The ring is chosen on the escape-time map (ring_cells: a short
    loop through cells lost within T_quick periods, inside the flux estimate flux_share x request),
    sampled with spacing h0 (default: the cell spacing), disks of radius 0.6 h.  Per disk: the
    centre's line (lost_centres; not lost -> its cell is blocked and the loop re-routed, up to
    max_rounds loops), then the disk proof (certify_disks); a disk whose bound fails is replaced by
    `factor` smaller disks covering the same stretch (up to max_depth refinements).  Every certified
    disk goes into a pool; the ring is the shortest chain of overlapping pool disks winding once around
    a (ring_cycle).  Then: reference re-traces of n_ref sampled centre lines (the largest arc length x
    stretch; DOP853 on one) give the reference floor added to every disk's position error; Q = the
    inner polygon of the chain (loops cut exactly), simple, winding 1 around a, B_phi > 0 inside.
    A proposed `path` (closed polygon) is used as is: any disk not certified refuses it.  Refuses
    when the ring cannot be closed, or F_up >= request.  Returns dict."""
    from scipy.spatial import cKDTree
    t0 = time.time()
    a = np.asarray(a, float)
    out = dict(kind="lost_ring", ok=False)
    scale = sec.scale
    delta = 1e-12 * scale
    T_max = T_quick if T_max is None else max(T_max, T_quick)
    if S_max is None:
        S_max = (T_max + 2) * 2 * np.pi * float(a[0]) / getattr(sec, "nfp", 1) * 1.2
    stats = dict(rounds=0, centres=0, not_lost=0, disks=0, refined=0, depth=0, T=T_quick)
    pool_C, pool_R, pool_rec = [], [], []
    blocked = None
    if path is None:
        if emap is None:
            excl = None
            if hasattr(sec, "source"):
                excl = (coil_crossings(sec.source), coil_margin if coil_margin is not None else 0.03 * scale)
            emap = escape_map(sec, a, T_last, exclude=excl)
        out["escape_quick_fraction"] = float(np.mean(emap["esc"] <= T_quick))
        blocked = np.zeros(emap["esc"].shape, bool)
        W = ring_wedge_flux(sec, emap)
        budget = None if request is None else flux_share * abs(request)
        G = emap["pts"].reshape(-1, 2)
    cyc = None
    T_cur = T_quick
    for rnd in range(1 if path is not None else max_rounds):
        if deadline is not None and time.time() > deadline:
            out["reason"] = "deadline"
            break
        if path is not None:
            P = np.asarray(path, float)
        else:
            cells, fl_est = ring_cells(emap, T_cur, blocked=blocked, W=W, budget=budget)
            while cells is None and T_cur < T_max:
                T_cur += 1
                cells, fl_est = ring_cells(emap, T_cur, blocked=blocked, W=W, budget=budget)
            if cells is None:
                out["reason"] = (f"no loop of usable cells lost within {T_cur} periods around the axis (round {rnd})"
                                 + ("" if fl_est is None else f"; least flux estimate {fl_est:.4g} > {budget:.4g}"))
                break
            stats["T"] = T_cur
            stats["flux_estimate"] = fl_est
            P = np.array([emap["pts"][i, j] for (i, j) in cells])
        stats["rounds"] += 1
        Pc = np.concatenate([P, P[:1]])
        cl = chord_param(Pc)
        L_path = float(cl[-1])
        hP = h0 if h0 is not None else min(float(np.median(np.diff(cl))), 0.4 * scale)
        m = max(int(math.ceil(L_path / hP)), 8)
        hs = L_path / m

        def at(par):
            par = np.asarray(par, float) % L_path
            sg = np.clip(np.searchsorted(cl, par, side="right") - 1, 0, len(Pc) - 2)
            w_ = (par - cl[sg]) / np.maximum(cl[sg + 1] - cl[sg], 1e-300)
            return Pc[sg] + w_[:, None] * (Pc[sg + 1] - Pc[sg])
        par0 = hs * np.arange(m)
        if pool_C:
            # stretches [par - h/2, par + h/2] of the new loop already inside the pool's union are skipped
            # (a new disk next to such a stretch overlaps the pool disk covering its end: the overlap
            # graph joins them)
            PCa, PRa = np.array(pool_C), np.array(pool_R)
            tr_ = cKDTree(PCa)
            offs = np.linspace(-0.5, 0.5, 9) * hs
            Xq = at((par0[:, None] + offs[None]).ravel())
            nbr = tr_.query_ball_point(Xq, float(PRa.max()))
            inside = np.array([any(np.linalg.norm(Xq[q_] - PCa[k_]) <= 0.98 * PRa[k_] for k_ in nb_)
                               for q_, nb_ in enumerate(nbr)]).reshape(len(par0), len(offs))
            par0 = par0[~np.all(inside, axis=1)]
        todo = [(p_, hs, 0) for p_ in par0]
        bad_pts = []
        while todo:
            if deadline is not None and time.time() > deadline:
                out["reason"] = "deadline during the disk proofs"
                todo = []
                break
            batch, todo = todo[:chunk], todo[chunk:]          # chunks: the deadline is checked between them
            par = np.array([t_[0] for t_ in batch])
            hh = np.array([t_[1] for t_ in batch])
            dep = np.array([t_[2] for t_ in batch])
            Cz = at(par)
            cen = sec.lost_centres(Cz, S_max=S_max)
            stats["centres"] += len(Cz)
            nl = np.flatnonzero(~cen["lost"])
            stats["not_lost"] += len(nl)
            bad_pts += [Cz[k] for k in nl]
            li = np.flatnonzero(cen["lost"])
            nxt = []
            if len(li):
                recs = sec.certify_disks(Cz[li], 0.6 * hh[li], cen["d"][li], cen["S"][li])
                stats["disks"] += len(li)
                for q_, k in enumerate(li):
                    rc = recs[q_]
                    if rc["ok0"]:
                        pool_C.append(Cz[k])
                        pool_R.append(0.6 * hh[k])
                        pool_rec.append(rc)
                    elif dep[k] < max_depth and rc["factor"] >= 2:
                        nf = int(rc["factor"])
                        hn = hh[k] / nf
                        nxt += [(par[k] - hh[k] / 2 + (q2 + 0.5) * hn, hn, dep[k] + 1) for q2 in range(nf)]
                        stats["refined"] += 1
                        stats["depth"] = int(max(stats["depth"], dep[k] + 1))
                    else:
                        bad_pts.append(Cz[k])
            todo = todo + nxt
            if verbose:
                print(f"   ring round {rnd}: {len(pool_C)} disks certified, {len(nl)} centres not lost, "
                      f"{len(todo)} to do ({time.time() - t0:.0f}s)", flush=True)
        if out.get("reason", "").startswith("deadline"):
            break
        if path is not None and bad_pts:
            out["reason"] = f"proposed ring: {len(bad_pts)} disks not certified (centre not lost or bound failed)"
            break
        if pool_C:
            cyc = ring_cycle(np.array(pool_C), np.array(pool_R), a, delta)
        if cyc is not None:
            break
        if path is not None:
            out["reason"] = "proposed ring: the certified disks do not close around the axis"
            break
        # block the cells of the failed points and re-route
        if bad_pts:
            _, gi = cKDTree(G).query(np.array(bad_pts), k=1)
            blocked.reshape(-1)[gi] = True
        else:
            out["reason"] = "the certified disks do not close around the axis"
            break
    out["stats"] = stats
    if cyc is None:
        out.setdefault("reason", "no ring of certified disks")
        out["time"] = time.time() - t0
        return out
    C = np.array([pool_C[i] for i in cyc])
    R = np.array([pool_R[i] for i in cyc])
    recs = [pool_rec[i] for i in cyc]
    out["n_disks"] = int(len(C))
    out["n_pool"] = int(len(pool_C))
    # reference re-traces of sampled centre lines; their largest difference is added to every disk's
    # position error (2 L x floor in the bound)
    key = np.array([rc["arc_length"] * max(rc["stretch"], 1.0) for rc in recs])
    smp = list(np.argsort(-key)[:max(n_ref - 1, 1)])
    rng = np.random.default_rng(len(C))
    rest = [i for i in range(len(C)) if i not in smp]
    if rest:
        smp.append(int(rng.choice(rest)))
    if getattr(sec, "use_reference", True):
        ref = sec.ring_reference(np.array([recs[i]["x_start"] for i in smp]), np.array([recs[i]["direction"] for i in smp]),
                                 np.array([recs[i]["arc_length"] for i in smp]), np.array([recs[i]["x_end"] for i in smp]))
        floor = float(np.max(ref["diff"]))
        f_ref_max = float(np.max(ref["f_ref"]))
    else:
        ref, floor, f_ref_max = None, 0.0, -1.0
    bounds = np.array([rc["bound0"] + 2.0 * rc["lipschitz"] * floor for rc in recs])
    out["disk_evidence"] = dict(reference_floor=floor, reference_f_max=f_ref_max, n_reference=len(smp),
                                dop853=None if ref is None else [{k: v for k, v in d_.items()} for d_ in ref["dop"]],
                                max_bound_f=float(np.max(bounds)), max_arc_length=float(max(rc["arc_length"] for rc in recs)),
                                max_stretch=float(max(rc["stretch"] for rc in recs)),
                                min_radius=float(R.min()), max_radius=float(R.max()))
    if not (np.isfinite(floor) and floor < 1e-5 and f_ref_max < 0):
        out["reason"] = f"reference re-trace of sampled disk centres disagrees (floor {floor:.3g}, f_ref {f_ref_max:.3g})"
        return out
    if not np.all(bounds < 0):
        out["reason"] = "a disk bound fails with the reference floor"
        return out
    Q, tt = ring_inner_polygon(C, R, a, delta)
    if Q is None:
        out["reason"] = "consecutive disks do not cover the ring polygon"
        return out
    if not closed_polygon_simple(Q, 1e-13 * scale):
        Q = cut_loops_exact(Q, a)
    if Q is None or not closed_polygon_simple(Q, 1e-13 * scale) or abs(winding_number(Q, a)) != 1:
        out["reason"] = "ring polygon not simple or not around the axis"
        return out
    F, qerr = sec.flux(Q)
    dens = sec.density_check(Q)
    out["density_check"] = dens
    if not dens["ok"]:
        out["reason"] = "B_phi > 0 not established inside the ring polygon"
        return out
    Lq = float(np.sum(np.linalg.norm(np.diff(np.concatenate([Q, Q[:1]]), axis=0), axis=1)))
    strip = 2 * delta * Lq * sec.density_max(Q)
    F_up = float(abs(F) + qerr + strip)
    out.update(flux=float(abs(F)), quad_err=float(qerr), strip=float(strip), F_up=F_up, polygon=Q.tolist(),
               disks=dict(centres=C.tolist(), radii=R.tolist(), directions=[rc["direction"] for rc in recs],
                          arc_lengths=[rc["arc_length"] for rc in recs], bounds=bounds.tolist()))
    out["time"] = time.time() - t0
    if request is not None and F_up >= abs(request):
        out["reason"] = f"ring encloses more than the request (F_up {F_up:.6g})"
        return out
    out["ok"] = True
    return out
