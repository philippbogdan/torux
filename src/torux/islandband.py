"""Island-band certificate from the invariant manifolds of a chain's X line and the stellarator
symmetry ("manifold island band"): no flux surface around the magnetic axis encloses a flux in
(F_lo, F_hi), for chains of any width, including the thin chains of rational-target designs whose
separatrices split by far less than any error bound.

Setting and the class of tori covered
-------------------------------------
Sigma is the half plane phi = phi_s, a symmetry plane of the stellarator symmetry (phi_s = 0 or
pi / nfp), P the period map from Sigma to itself (one field period along the lines), S(R, Z) =
(R, -Z) the reflection of Sigma.  s = sign of B_phi on the magnetic axis; B_phi below means s B_phi.
a is the magnetic axis' point in Sigma.  COVERED TORI are those of torux/barrier.py: an invariant torus
Gamma with (T1) no source in its solid torus V, (T2) B_phi > 0 on Gamma, (T3) V contains the magnetic
axis and Gamma meets Sigma in one Jordan curve gamma around a (D its interior, flux(Gamma) =
int_D B_phi dR dZ).  Every flux surface a request can be answered with is covered.  (T3) holds in
every plane phi = const once it holds in one (phi has no critical point on Gamma by (T2)), and the
flux is the same through every plane, so the plane phi_s may be either symmetry plane.

ASSUMPTION (exact symmetry).  The source is stellarator symmetric: its field satisfies
B(R, -phi, -Z) = (B_R, -B_phi, -B_Z)(R, phi, Z) exactly (QUASR coil sets are generated from their
base coils by the symmetry group; the loader checks the claim and every run measures the defect of
the evaluated field, ~1e-15).  Then P is reversible: S P S = P^-1, so for a periodic orbit X,
S(W^u(X_k)) = W^s(S X_k), and a point h of an unstable branch of X_k on the line Fix(S) = {Z = 0}
lies on W^s(S X_k): an exact heteroclinic point, however small the angle between W^u and W^s.

The chain.  X = {X_0..X_{q-1}}, q >= 2, a hyperbolic (not reflection-hyperbolic) periodic orbit of
P, X_j = P^j X_0.  At each X_k the
four branches (half manifolds) of W^u, W^s.

THEOREM (manifold island band).  Let c_A, c_B be the two unstable branches of X at X_0 and u_c(X_k) =
P^k(c) the branch of c at X_k (so P maps u_c(X_k) onto u_c(X_k+1)); s_c(X_k) = S u_c(S X_k) the mirrored
stable branches (P maps them equivariantly too, as P S = S P^-1).  Suppose there are two closed curves
C_A, C_B such that
 (1) C_A = union over the q gaps (X_k, X_k') of consecutive points along the chain of an
     arc of u_A(X_k) from X_k to a point h_k and an arc of s_A(X_k') from h_k to X_k', with h_k in both
     branches (a heteroclinic point); C_B likewise with u_B(X_k') from X_k' to h'_k and s_B(X_k) from
     h'_k to X_k (the B arcs run the other way round);
 (2) each C_i has a computed polyline C_c,i, simple, with every point of the true C_i within eps of it
     along matching parameterisations, and a outside the eps-tube of C_c,i with winding 1;
 (3) B_phi >= 0 inside C_c,i.
Then no covered torus has its flux in (F_lo, F_hi), F_lo = min_i (Phi(C_c,i) + Delta_i), F_hi =
max_i (Phi(C_c,i) - Delta_i), Delta_i the flux of B_phi^+ over the eps-tube plus the quadrature error.

Proof.  Let gamma be the section of a covered torus.  By Lemma A of torux/barrier.py (R B_phi is
harmonic), B_phi > 0 on D-bar and P maps gamma onto gamma and D onto D.
(i) gamma contains no point of X.  Then no branch meets gamma (a point of W^u(X_k) on gamma has its
backward orbit converging to X_k inside the closed invariant gamma; same forward for W^s), so every
branch with its base point lies on one side of gamma, and all of X on one side (P permutes X and
preserves D).  If X lies in D, then C_A, C_B lie in D and every point with nonzero winding number of
C_i is in D (the exterior of gamma is connected, unbounded and misses C_i); outside the eps-tube the
straight-line homotopy C_i -> C_c,i keeps the winding numbers, so int(C_c,i) minus the tube lies in D
and flux(gamma) >= Phi(C_c,i) - Delta_i for both i.  If X lies outside D-bar, D is connected, misses
C_i and contains a, so every x in D has wind(C_i, x) = wind(C_i, a) = wind(C_c,i, a) = 1, hence D lies
in int(C_c,i) union the tube, and by (3) flux(gamma) <= Phi(C_c,i) + Delta_i for both i.
(ii) gamma contains a point of X, hence all of it.  X_k is an isolated fixed point of P^q and P^q
preserves gamma's orientation, so on a short arc of gamma from X_k it moves points monotonically: the
arc is a local unstable or stable branch, and by invariance gamma contains the whole branch, which runs
along gamma monotonically to the next fixed point of P^q on gamma.  Every branch at X_k contains a
heteroclinic point of (1) (u_A(X_k): h_k; s_A(X_k): h_(k-1); u_B, s_B: the h'), whose forward (backward)
orbit converges to the partner of (1), so the branch is the arc of gamma from X_k to that partner.  If
both arcs of gamma at X_0 were stable, by the P-equivariance every X_k would be so, but the arc from X_0
(s_A or s_B, containing a heteroclinic point) is also an unstable branch of its partner: so gamma
contains u_A(X_0) or u_B(X_0), say u_A(X_0), hence its images u_A(X_k) for all k, i.e. the closed curve
C_A (each u_A(X_k) = s_A(X_k') is the arc through h_k).  A closed curve inside the Jordan curve gamma is
gamma: gamma = C_A, a Jordan curve within the eps-tube of the simple C_c,A, so |flux(gamma) -
Phi(C_c,A)| <= Delta_A; likewise for B.  In all cases flux(gamma) <= F_lo or >= F_hi.  (Loop cuts: if
the computed C_c,i is not simple and is cut at certified crossings of its arcs, case (ii) is impossible
for that class -- C_i would not be a Jordan curve -- and the cut curve, still made of manifold arcs,
serves case (i).)

What the certificate computes and checks.  The plane phi_s is a symmetry plane in which Z = 0 straddles a
gap of the X orbit (two mirror-image X points X_k, S X_k are chain neighbours), checked on the X line's
points: first the plane of the chain's other symmetric line (an O line, or a reflection-hyperbolic centre --
neither is used by the theorem), else either plane.  The X orbit is
the X line's point Newton-polished on the section map with the line's spectral monodromy M (the
variational equation along the closed line; a finite-difference monodromy is far too noisy for the
eigenvectors of a weak X line, multiplier 1.0001-1.01).  The two unstable branches are grown with all
their images (torux.barrier.GenCurve: one period map at a time, refined by pre-image insertion, normal
error per vertex propagated); the stable branches are their mirror images.  The order along the chain
is that of the torus angle (k p mod q; the polar angle about the axis is not monotonic along a chain
in a bean-shaped section).  For each unstable generating curve one heteroclinic point is a certified
crossing of its branch at some X_k with the exact line Z = 0 (parallelogram lemma, the line known
exactly), where S X_k is X_k's successor (class A) or predecessor (class B) along the chain; the other
gaps' points are its images P^j h (a heteroclinic orbit), traced, with a 2D error bound; each gap's
arcs are cut at the computed branches' points nearest P^j h (searched only up to the image's level,
and refused when farther than the branch error + the image error).  The arcs used are chord-validated,
the closed polygons decimated with the dropped vertices' distances added to the errors, checked simple
(or cut at certified self-crossings), with winding 1 around the axis point outside the error tube; then
the loop integrals of A, the tube flux and the B_phi > 0 grid check.  Finally, when T1, T2 are the tori
between which the chain was found, the band must lie inside their bracket (F1, F2) (invariant tori
enclose the separatrices or lie outside them; a band reaching beyond them is refused as inconsistent);
the record states whether they meet the surface bar torux.analysis.BN_SURFACE (off-grid |B.n|/|B| <=
1e-14), i.e. whether the band's edges are verified surfaces -- the band itself does not depend on it.
T1, T2 otherwise only start the closed-line search and bound the surrogate's annulus: the theorem does
not use them (bracket_check=False for proxy surfaces: proxy_annulus offsets a design surface along its
normals by a smooth fitted offset that follows the chain's closed lines, widened by a margin).

Error terms (Delta_i).  STRICT given the inputs below: the parallelogram lemma (crossing with Z = 0),
the winding/tube argument, the polygon simplicity test and decimation bookkeeping, the Jordan-curve
topology, the reversibility argument, Lemma A.  VALIDATED MODELS (estimates checked against
independent computations, as in torux/barrier.py): the period map's error per step tau -- for the
accurate tracer (24 steps per period when it carries the manifolds) 2 x the GBS estimate + 2e-13 R; for the surrogates (BandSurrogate: Chebyshev
interpolants of the period map on patches aligned with the annulus between T1 and T2, in normal
coordinates of a smooth reference curve; PatchSurrogate: square cells; one batch of accurate traces)
per patch 3 x the largest difference from the accurate tracer at random points of the annulus + 2 x
the tracer's model + 2e-13 R -- checked against the independent reference tracer on
sampled manifold vertices in every run; first-order propagation of normal errors along the manifolds
(e' = beta e + tau, beta = det / stretch); e_X = 2 |(M - I)^-1| (closure + q tau) (Newton-Kantorovich;
M the X line's spectral monodromy, accuracy from two step counts, or -- when the closed line is off the
orbit -- the product of the surrogate's analytic Jacobians along the orbit, accuracy from the accurate
tracer's Richardson-extrapolated central differences); the start segment's error
0.5 kappa (lam s0)^2 (a-priori curvature bound kappa, checked against the measured curvature) +
lam s0 x the eigenvector angle error (first-order perturbation bound from the monodromy's accuracy);
chord errors by midpoint validation; the heteroclinic images' error |DP^j| box/2 + 2 sum |DP^(j-i)|
tau_i (finite-difference Jacobians, first order; checked against the reference tracer); the flux
quadrature error (difference of 3- and 5-point Gauss rules per edge); B_phi > 0 inside by a grid with
a sampled-gradient Lipschitz bound; the field itself (engine vs reference evaluator at rounding).
Growth that cannot finish: when the levels still needed (q ln(reach / s0) / ln lam, reach = half the
distance between a mirror pair of neighbours), at the measured time per level, would end beyond the deadline
by more than a factor 1.5, the attempt is refused early with that estimate (a cheap failure instead of
the whole budget).  EVIDENCE in every run: the reference closure of the X orbit, reference re-traces of sampled manifold
vertices (forward and, by the symmetry, backward) and of the heteroclinic orbit, the reference field's
loop integral of A along both polygons, the reference field's symmetry defect.

Chains with several X orbits, off-symmetry X lines.  The proof never assumes that X is the chain's only
hyperbolic orbit: in (ii) the arc of gamma from X_k ends at the first fixed point of P^q along gamma,
which the heteroclinic point forces to be X_k's partner, whatever other periodic orbits the chain has;
so a certificate built from one symmetric X orbit is valid in a chain with further X orbits (its curves
then pass the others without touching them, and fail the simplicity check when they cannot).  An X orbit
that is not mapped to itself by S is refused by the single-orbit certificate (mirror mismatch), and
torux.chains.chain_lines finds only symmetric lines; such chains go to island_band_certificate_pairs
(directly, or from island_band_certificate_manifold with pairs_fallback=True):

THEOREM' (several X orbits).  Let XX = X^1 u .. u X^r be hyperbolic (not reflection-hyperbolic) period-q
orbits, closed under S, ordered along the chain (|XX| >= 3); gaps = consecutive pairs (x, x+); P permutes
the gaps, with T gap types (P-orbits of gaps).  For each gap let a_A be an arc of an unstable branch of x
to a heteroclinic point h and on to x+ along a stable branch of x+, and a_B one of an unstable branch of x+
to h' and on to x along a stable branch of x, such that every one of the four branches at every point of
XX carries one of these heteroclinic points, and the branches used in a gap and in its P-image are
P-images of each other.  For sigma in {A, B}^T let M_sigma be the closed curve of the arcs a_sigma(type)
of all gaps; C_A, C_B the uniform ones, satisfying (2) and (3).  Then no covered torus has its flux in
(F_lo, F_hi) minus the closed intervals [Phi_sigma - Delta_sigma, Phi_sigma + Delta_sigma] of the mixed
sigma, Phi_sigma the loop integral of A along M_sigma's computed polyline oriented by its winding about a
(winding 0: omitted), Delta_sigma its tube bound and quadrature error.
Proof.  (i) as before: C_A joins all of XX, so all of it lies on one side of gamma.  (ii) gamma containing
a point of XX contains all of XX (the branch connections join it to the rest); each arc of gamma at x is a
whole branch running to the partner of its heteroclinic point, a neighbour of x; gamma visits XX in the
chain order (|XX| >= 3: a step forward followed by one back would close gamma early), one arc per gap,
that gap's A or B connection; P maps gamma onto itself and A connections to A connections, so the choice
is constant on gap types: gamma = M_sigma.  For a uniform sigma as before; otherwise flux(gamma) is the loop
integral of A along the Jordan curve M_sigma, within Delta_sigma of the computed polyline's (the
straight-line homotopy sweeps only the tube).
Implemented when every gap type has a gap straddled by Z = 0 in one of the two symmetry planes (its ends
a mirror pair there).  In the certificate's plane Sigma (that of an O line) such a gap takes the
heteroclinic points of both classes from Fix(S), the other gaps of its type by P^j images.  A gap type
straddled only in the other plane Sigma' (phi_s + pi / nfp; its reflection S' reverses P' = H P H^-1, H
the half-period map Sigma -> Sigma') takes them from Sigma': the X orbits there are the H-images of
Sigma's (Newton on a second BandSurrogate, of Sigma', from H(X^i_0), so that the indices correspond;
checked: within 1e-6 scale of H(X^i_k), equal multipliers, the same order along the chain), their
unstable branches are grown in Sigma' (start segments s0 |DH v_u|, so that the levels match Sigma's to
first order), and a certified crossing with Z = 0 there is an exact heteroclinic point h' of P' (on u(H x)
and on s(S' H x)); h = H^-1(h') is then one of P on u(x) and s(x+), x+ = H^-1 S' H x the neighbour, and
serves as the gap's heteroclinic point in Sigma exactly as an image P^j h does (the branches searched up to
the level of h' + 3, refused when farther than the branch error + h's error).  h's error: |DH^-1| box'/2
+ the half map's tau (VALIDATED MODEL: the accurate tracer at 12 and 24 steps per half period, tau = 2 x
the finer one's GBS estimate + 2e-13 R + their difference; DH^-1 by central differences, first order),
with the other plane's reference evidence (sampled vertex re-traces, X-orbit closures, the reference
tracer's half-period image of h returning to h' within |DH| tau + 10 x its own estimate) required too.  The
hyperbolic orbits are found by Newton on the surrogate's P^q over the annulus, monodromy from its
analytic Jacobians (checked against the accurate tracer).  Three QUASR requests have all X
points off the symmetry lines (mirror pairs X, X' = S X): QUASR 1639254 (nfp 4, chain -2/3: symmetric
lines = two O orbits, residues 0.035 and 0.173, both on Z = 0 at phi = 0) has every gap type straddled
at phi = 0; 1976964 (nfp 6, -1/2) and 136101 (nfp 1, -1/2) have their two O orbits one in each symmetry
plane, so each gap type is straddled in one plane only and the other plane's heteroclinic points are
needed.  Also not covered:
q = 1 chains (homoclinic: no partner across Fix(S)), reflection-hyperbolic X lines, non-symmetric sources,
and chains too weak for the manifold growth: every X residue (pairs: every hyperbolic period-q orbit's
residue; and when the chain has no symmetric X line, its O residues) below MIN_RESIDUE = 2.5e-10 in
modulus, i.e. lambda - 1 = 2 |r|^(1/2) + O(|r|) < 3.2e-5 (the manifolds would need ~ 1 / (lambda - 1) > 3e4
iterates of P^q per e-fold to reach Fix(S); where P^q is that close to the identity its fixed points are
not isolated numerically either).  These are refused with that reason (QUASR designs whose transform sits
on a low-order rational: e.g. 0196784, 0019628 -1/20, X residues -1.8e-13, -2.4e-12; 0180117 -3/20, O
residues 1.7e-12).

Third-pass options (all off by default: the full path's behaviour; torux.analysis.POLICY switches them on, read at
call time).  proxy_annulus_sym: the proxy annulus fitted with the symmetry (an even offset function, interpolating
the chain's points).  axis_point(inverse=True): an inverse-hyperbolic period-one line as the axis point a when the
design's line is one (the theorem uses a only as a point inside every covered torus's disk).  POLICY
'surrogate_steps': node traces start at the steps per period a 64-node sample needs, and builds the sample projects far
beyond the deadline are refused at once.  POLICY 'card_traces': node traces on a card (torux.cardtrace, the same GBS
scheme); the surrogates' validation points stay on the CPU accurate tracer, so the validated error model is unchanged,
and every card batch is re-traced in part on the CPU and refused when a difference exceeds both error estimates.
None of them changes what a certificate proves.

Axis identity (every axis kind, both paths).  The covered tori are those around the magnetic axis -- the analysis's
period-one line D, the line its answers' tori must enclose (torux.analysis.torus_topology).  The proof uses the axis
point a only as a point inside every covered torus's disk.  axis_point finds a by itself (a symmetric period-one root
on T_inner's segment of Z = 0); on the proxy path T_inner is not invariant, so nothing but a's identity with D puts a
inside the tori around D (a torus around D need not enclose another line inside T_inner); on the bracket path an
invariant T1 enclosing a and D makes it automatic, and the same check applies.  axis_line=D (torux.axis.Axis, or any
object with .at(phi) -> (R, Z)) makes axis_point refuse unless the root it takes is D's point within
SAME_LINE_REL x scale + 10 e_a (e_a its Newton error bound; the scale term covers D's own representation error: the
design path's 16-mode Fourier line was seen up to 4.6e-4 m (0.56 of this tolerance) from the polished point on a
strongly shaped nfp = 8 axis whose last modes are 2e-4) and no other symmetric period-one root lies on the scanned
segment (any index, or a root whose orbit did not polish; distinct from the pick by more than ten times their Newton
bounds).  The certificate functions take axis_line and record the check (axis_identity); torux.analysis passes D on
every path.  axis_line=None keeps the earlier behaviour (no identity check: direct callers with their own geometry,
the tests).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import barrier as BR
from .envfail import environment_failure

# chains with every hyperbolic residue below this in modulus (lambda - 1 < 3.2e-5) are refused (docstring)
MIN_RESIDUE = 2.5e-10


# ============================================================================ section at phi_s
class PlaneSection(BR.FieldSection):
    """torux.barrier.FieldSection for the symmetry plane phi = phi_s (0 or pi / nfp)."""

    def __init__(self, engine, source, phi_s=0.0, sign=1.0, k=6, steps_per_period=12, reference=True,
                 scale=None):
        super().__init__(engine, source, sign=sign, k=k, steps_per_period=steps_per_period, reference=reference,
                         scale=scale)
        self.phi_s = float(phi_s)
        self.c, self.s = math.cos(self.phi_s), math.sin(self.phi_s)

    def coarse(self):
        c = PlaneSection(self.engine, self.source, phi_s=self.phi_s, sign=self.sign, k=self.k, steps_per_period=6,
                         reference=False, scale=self.scale)
        c._ref = self._ref
        return c

    def xyz(self, y):
        y = np.atleast_2d(y)
        return np.stack([y[:, 0] * self.c, y[:, 0] * self.s, y[:, 1]], -1)

    def _bphi(self, B):
        return -B[:, 0] * self.s + B[:, 1] * self.c

    def step(self, y, d):
        y = np.asarray(y, float).reshape(-1, 2)
        yy, lost, err = self.tracer.period_map(y, 1, direction=d, phi0=self.phi_s)
        self.n_steps += len(y)
        floor = 2e-13 * np.maximum(1.0, np.abs(yy[:, 0]))
        return yy, lost, 2.0 * err + floor

    def density(self, y):
        return self.sign * self._bphi(self.B(self.xyz(y)))

    def flux(self, C, field=None):
        C = np.asarray(C, float)
        if BR.signed_area(C) < 0:
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
            AR = A[:, 0] * self.c + A[:, 1] * self.s
            n = len(C)
            dR = (D - C)[:, 0]
            dZ = (D - C)[:, 1]
            tot = 0.0
            for j in range(ng):
                sl = slice(j * n, (j + 1) * n)
                tot += 0.5 * w[j] * np.sum(AR[sl] * dR + A[sl, 2] * dZ)
            res.append(-self.sign * tot)
        return float(res[1]), float(abs(res[1] - res[0]))

    def density_check(self, C, h=None, n_target=120):
        C = np.asarray(C, float)
        lo, hi = C.min(0), C.max(0)
        if h is None:
            h = float(np.max(hi - lo)) / n_target
        gx = np.arange(lo[0] - h, hi[0] + 2 * h, h)
        gz = np.arange(lo[1] - h, hi[1] + 2 * h, h)
        GX, GZ = np.meshgrid(gx, gz, indexing="ij")
        pts = np.stack([GX.ravel(), GZ.ravel()], -1)
        ins = BR.inside_polygon(C, pts)
        near = BR.polyline_distance(pts[~ins], np.concatenate([C, C[:1]])) <= 1.5 * h
        sel = np.concatenate([np.flatnonzero(ins), np.flatnonzero(~ins)[near]])
        o = self.engine.evaluate(self.xyz(pts[sel]), grad=True, A=False)
        B = np.asarray(o["B"])
        G = np.asarray(o["G"])
        bphi = self.sign * self._bphi(B)
        gmax = float(np.max(np.sqrt(np.sum(G * G, axis=(1, 2)))))
        lb = float(np.min(bphi)) - 2.0 * gmax * h / math.sqrt(2)
        return dict(ok=bool(lb > 0), min_grid=float(np.min(bphi)), grad_max=gmax, h=h, n=int(len(sel)),
                    min_lower_bound=lb)

    def density_max(self, pts):
        return float(np.max(np.abs(self._bphi(self.B(self.xyz(pts)))))) * 1.05

    def reference_steps(self, y, d):
        from .reference import trace as RT
        F = self.ref_field()
        y = np.atleast_2d(np.asarray(y, float)).copy()
        if d < 0:
            y[:, 1] *= -1                         # P^-1 = S P S
        P = 2 * np.pi / self.nfp
        o = RT.poincare(F, y[:, 0], y[:, 1], [self.phi_s + P], 1, phi0=self.phi_s, h=2 * np.pi / 512)
        out = np.stack([o["R"][:, 0, 0], o["Z"][:, 0, 0]], -1)
        if d < 0:
            out[:, 1] *= -1
        return out, o["err"][:, 0, 0]

    def reference_map(self, y, nper):
        """Reference tracer: the points after 1..nper periods (nper, n, 2) and the error estimates."""
        from .reference import trace as RT
        y = np.atleast_2d(np.asarray(y, float))
        P = 2 * np.pi / self.nfp
        # sections at phi_s + j P (j = 1..nper) as repetitions of whole turns where possible
        phis = [self.phi_s + j * P for j in range(1, self.nfp + 1)]
        nt = int(math.ceil(nper / self.nfp))
        o = RT.poincare(self.ref_field(), y[:, 0], y[:, 1], phis, nt, phi0=self.phi_s, h=2 * np.pi / 512)
        R = o["R"].reshape(len(y), -1)[:, :nper]
        Z = o["Z"].reshape(len(y), -1)[:, :nper]
        E = o["err"].reshape(len(y), -1)[:, :nper]
        return np.stack([R, Z], -1).transpose(1, 0, 2), E.T


def _bphi_sign(engine, R, phi_s):
    x = np.array([[R * math.cos(phi_s), R * math.sin(phi_s), 0.0]])
    B = np.asarray(engine.evaluate(x, grad=False, A=False)["B"])[0]
    return 1.0 if (-B[0] * math.sin(phi_s) + B[1] * math.cos(phi_s)) > 0 else -1.0



# ============================================================================ model system (tests)
class ReversibleTwistKick(BR.HenonSection):
    """Model system: the symmetric splitting F = K_1/2 o R o K_1/2 of a kick K_t(x, y) = (x, y + t k f(x)),
    f(x) = (x + c2 x^2) exp(-x^2 / w^2) (c2 != 0: no extra
    symmetry (x, y) -> (-x, -y)), and a twist rotation R by omega0 + omega2 (x^2 + y^2).  Area preserving
    (density 1, flux = area) and reversible with S(x, y) = (x, -y): S K S = K^-1, S R S = R^-1, hence
    S F S = F^-1, like the period map of a stellarator-symmetric field in a symmetry plane.  The origin
    is the elliptic 'axis'; chains p/q sit where omega(r) ~ 2 pi p / q (thin for small k)."""

    def __init__(self, k, w=0.6, omega0=1.0, omega2=1.0, c2=0.7, R_lost=50.0):
        super().__init__(0.0, R_lost=R_lost)
        self.k, self.w, self.o0, self.o2, self.c2 = float(k), float(w), float(omega0), float(omega2), float(c2)
        self.nfp = 1

    def _kick(self, x, y, t):
        return y + t * self.k * (x + self.c2 * x * x) * np.exp(-x * x / self.w ** 2)

    def _rot_half(self, x, y, sgn):
        om = 0.5 * sgn * (self.o0 + self.o2 * (x * x + y * y))
        c, s = np.cos(om), np.sin(om)
        return x * c - y * s, x * s + y * c

    def half_map(self, Y, forward=True):
        """The model's 'half period' H = R_1/2 o K_1/2 (F = S H^-1 S o H: the other plane's map is
        F' = H F H^-1 = R_1/2 o K o R_1/2, reversible with the same S), or H^-1.  Returns (Y, lost, tau)."""
        Y = np.asarray(Y, float).reshape(-1, 2)
        x, y = Y[:, 0], Y[:, 1]
        if forward:
            y = self._kick(x, y, 0.5)
            x, y = self._rot_half(x, y, +1)
        else:
            x, y = self._rot_half(x, y, -1)
            y = self._kick(x, y, -0.5)
        out = np.stack([x, y], -1)
        return out, ~np.all(np.isfinite(out), -1), 4e-15 * (1.0 + np.sum(out * out, -1))

    def other_plane(self):
        """The model's other 'symmetry plane': the map F' = R_1/2 o K o R_1/2."""
        import copy
        o = copy.copy(self)
        o.forward = o._forward_other
        return o

    def _forward_other(self, x, y):
        x, y = self._rot_half(x, y, +1)
        y = self._kick(x, y, 1.0)
        return self._rot_half(x, y, +1)

    def _rot(self, x, y, sgn):
        om = sgn * (self.o0 + self.o2 * (x * x + y * y))
        c, s = np.cos(om), np.sin(om)
        return x * c - y * s, x * s + y * c

    def forward(self, x, y):
        y = self._kick(x, y, 0.5)
        x, y = self._rot(x, y, +1)
        return x, self._kick(x, y, 0.5)

    def step(self, y, d):
        y = np.asarray(y, float).reshape(-1, 2)
        d = np.broadcast_to(np.asarray(d), (len(y),))
        sg = np.where(d > 0, 1.0, -1.0)
        x0, y0 = y[:, 0], y[:, 1] * sg                   # backward: F^-1 = S F S
        x1, y1 = self.forward(x0, y0)
        out = np.stack([x1, y1 * sg], -1)
        lost = ~np.all(np.isfinite(out), -1) | (np.hypot(out[:, 0], out[:, 1]) > self.R_lost)
        self.n_steps += len(y)
        tau = 4e-15 * (1.0 + np.sum(out * out, -1)) * (1.0 + abs(self.o2) + abs(self.k))
        return out, lost, tau

    def symmetric_orbits(self, q, r_lo, r_hi, n=400):
        """Symmetric period-q orbits with a point on y = 0 in (r_lo, r_hi) (either sign of x): roots of
        y(F^m z) on the symmetry lines (a reversible orbit meeting Fix(S) twice is periodic): scanned
        by Newton from a grid.  Returns torux.barrier.Orbit list (distinct)."""
        out = []
        for x0 in np.concatenate([np.linspace(r_lo, r_hi, n), -np.linspace(r_lo, r_hi, n)]):
            orb = BR.orbit_from_point(self, np.array([x0, 0.0]), q, polish=8)
            if orb is None or orb.info["closure"] > 1e-12:
                continue
            if abs(orb.pts[0, 1]) > 1e-9 or np.min(np.linalg.norm(orb.pts[1:] - orb.pts[0], axis=1)) < 1e-6:
                continue
            if any(np.min(np.linalg.norm(o.pts - orb.pts[0], axis=1)) < 1e-7 for o in out):
                continue
            out.append(orb)
        return out

    def circle_area(self, z, n=4000):
        """Area enclosed by the orbit of z (an invariant circle around the origin), from its points
        ordered by polar angle; with the largest angular gap (for tests: the area of invariant circles)."""
        pts = [np.asarray(z, float)]
        y = np.asarray(z, float)[None]
        for _ in range(n):
            y, _, _ = self.step(y, 1)
            pts.append(y[0])
        P = np.array(pts)
        th = np.arctan2(P[:, 1], P[:, 0])
        o = np.argsort(th)
        gap = float(np.max(np.diff(np.concatenate([th[o], th[o][:1] + 2 * np.pi]))))
        return abs(BR.signed_area(P[o])), gap

# ============================================================================ helpers
def torus_section(T, phi, n=1024):
    th = 2 * np.pi * np.arange(n) / n
    R, Z = T.geometry_on(th, np.full(n, phi))
    return np.stack([R, Z], -1)


SAME_LINE_REL = 1e-2       # the axis point is D's when within this fraction of the section's scale (+ 10 e_a)


def axis_point(sec, T_inner, plane, inverse=False, axis_line=None, info=None):
    """The magnetic axis' point in the section (elliptic symmetric period-one line on Z = 0 inside
    T_inner's section).

    inverse=True: the design's period-one line is inverse hyperbolic (QUASR designs with the axis transform
    at a half-integer per period; torux.analysis's design path then takes that line as the one every surface encloses,
    analysis 'axis_kind'), and where no elliptic symmetric line lies inside T_inner the inverse-hyperbolic one is the
    axis point a.  The theorem uses a only as a point inside every covered torus's disk (T3: the class of tori around
    the axis the request's surfaces enclose), never its stability.

    axis_line (the analysis's line D, module docstring 'Axis identity'): None is returned unless the point taken is
    D's (within SAME_LINE_REL x sec.scale + 10 e_a) and the only symmetric period-one root on the scanned segment;
    info (a dict) receives the check and, when refused, the reason."""
    # T_inner meets Z = 0 at theta = 0, pi (symmetric torus)
    R2, _ = T_inner.geometry_on(np.array([0.0, np.pi]), np.array([sec.phi_s, sec.phi_s]))
    lo, hi = float(min(R2)), float(max(R2))
    w = hi - lo
    roots = BR.symmetric_roots(sec, 1, plane, np.linspace(lo + 0.05 * w, hi - 0.05 * w, 33))
    best = None
    inv = None
    found = []
    for R in roots:
        orb = BR.orbit_from_point(sec, np.array([R, 0.0]), 1)
        found.append((float(R), orb))
        if orb is None:
            continue
        if 0 < orb.residue < 1 and (best is None or orb.e_X < best.e_X):
            best = orb
        if inverse and orb.residue > 1 and (inv is None or orb.e_X < inv.e_X):
            inv = orb
    if best is None and inv is not None:
        inv.info["axis_kind"] = "inverse_hyperbolic"
        best = inv
    if axis_line is None or best is None:
        return best
    # axis identity: the point taken must be D's, and the only period-one root on the segment
    rec = info if info is not None else {}
    Ra, Za = axis_line.at(np.array([sec.phi_s]))
    D = np.array([float(np.ravel(Ra)[0]), float(np.ravel(Za)[0])])
    tol = SAME_LINE_REL * float(sec.scale) + 10.0 * float(best.e_X)
    d = float(np.hypot(*(best.pts[0] - D)))
    rec.update(D=D.tolist(), dist=d, tol=tol, e_a=float(best.e_X), residue=float(best.residue), n_roots=len(found),
               segment=[lo + 0.05 * w, hi - 0.05 * w])
    # another root is another line unless it is the pick's point within the two Newton bounds (D's tolerance above
    # also covers D's own representation error; two lines of the field are told apart by their error bounds)
    others = []
    floor = 1e-9 * float(sec.scale)
    for R, orb in found:
        if orb is best:
            continue
        if orb is None:
            if abs(R - float(best.pts[0][0])) > max(10.0 * float(best.e_X), 1e3 * floor):
                others.append(dict(R=R, polished=False))
            continue
        if float(np.hypot(*(orb.pts[0] - best.pts[0]))) > max(10.0 * (float(best.e_X) + float(orb.e_X)), floor):
            others.append(dict(R=R, residue=float(orb.residue), e_X=float(orb.e_X)))
    rec["others"] = others
    if not np.isfinite(d) or d > tol:
        rec["ok"] = False
        rec["reason"] = (f"axis point is not the analysis's period-one line: ({best.pts[0][0]:.9f}, "
                         f"{best.pts[0][1]:.2e}) vs D ({D[0]:.9f}, {D[1]:.2e}) at phi_s {sec.phi_s:.4f}, distance "
                         f"{d:.3e} > {tol:.3e}")
        return None
    if others:
        rec["ok"] = False
        rec["reason"] = (f"axis point not unique: {len(others)} other symmetric period-one root(s) inside T1's segment "
                         f"of Z = 0 (R {[round(o['R'], 9) for o in others]})")
        return None
    rec["ok"] = True
    return best


def line_monodromy_plane(engine, line, phi_s, nsteps=None):
    """Monodromy of the q-period map at the closed line's point in the plane phi_s, from the variational
    equation dY/dphi = DV(X(phi)) Y along the line (torux.orbits.spectral_monodromy, started at phi_s: one
    field + gradient evaluation on a phi grid, trigonometric interpolation of DV, Gauss-Legendre
    collocation of order 12).  Returns (M, dM): dM = max |M(n) - M(2n)| (two step counts)."""
    from .orbits import _cyl
    q, nfp = line.q, line.nfp
    L = 2 * np.pi * q / nfp
    K = line.K
    n_grid = max(64, 4 * K + 8)
    n_grid += n_grid % 2
    tt = np.arange(n_grid) / n_grid
    phi = phi_s + L * tt
    R, Z = line.at(phi)
    _, _, aRR, aRZ, aZR, aZZ = _cyl(engine, np.asarray(R), np.asarray(Z), np.asarray(phi))
    Avals = np.stack([np.asarray(aRR), np.asarray(aRZ), np.asarray(aZR), np.asarray(aZZ)], -1)
    F = np.fft.rfft(Avals, axis=0) / n_grid
    F[1:] *= 2.0
    if n_grid % 2 == 0:
        F[-1] *= 0.5
    kk = np.arange(F.shape[0])

    def A_at(p_):
        t = (np.atleast_1d(p_) - phi_s) / L
        return np.real(np.exp(2j * np.pi * np.outer(t, kk)) @ F).reshape(-1, 2, 2)

    s_ = 6
    xg, wg = np.polynomial.legendre.leggauss(s_)
    cg = 0.5 * (xg + 1)
    bg = 0.5 * wg
    V = np.vander(cg, s_, increasing=True)
    Ab = np.zeros((s_, s_))
    for i in range(s_):
        Ab[i] = np.linalg.solve(V.T, cg[i] ** (np.arange(1, s_ + 1)) / np.arange(1, s_ + 1))

    def integrate(ns):
        h = L / ns
        Y = np.eye(2)
        I2s = np.eye(2 * s_)
        for i in range(ns):
            Ast = A_at(phi_s + i * h + cg * h)
            Mbig = np.zeros((2 * s_, 2 * s_))
            for a_ in range(s_):
                for b_ in range(s_):
                    Mbig[2 * a_:2 * a_ + 2, 2 * b_:2 * b_ + 2] = h * Ab[a_, b_] * Ast[a_]
            rhs = np.concatenate([Ast[a_] @ Y for a_ in range(s_)], axis=0)
            Kst = np.linalg.solve(I2s - Mbig, rhs)
            Y = Y + h * sum(bg[a_] * Kst[2 * a_:2 * a_ + 2] for a_ in range(s_))
        return Y

    ns = nsteps or max(64, 2 * K)
    M1 = integrate(ns)
    M2 = integrate(2 * ns)
    return M2, float(np.max(np.abs(M1 - M2)))


class _few_threads:
    """Run a block with one numba thread: tracing a handful of points (monodromy stencils, single orbits)
    makes thousands of engine calls whose cost on several threads is the threads' synchronisation."""

    def __init__(self, n=1):
        self.n = n
        self.old = None

    def __enter__(self):
        try:
            import numba
            self.old = numba.get_num_threads()
            numba.set_num_threads(min(self.n, self.old))
        except Exception:  # noqa: BLE001
            self.old = None
        return self

    def __exit__(self, *a):
        if self.old is not None:
            import numba
            numba.set_num_threads(self.old)
        return False


def fd_monodromy(sec, x, q, h):
    """Monodromy of the section's P^q at x by central differences (step h) and the image of x."""
    E = np.array([[0.0, 0.0], [h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    y = np.asarray(x, float)[None] + E
    pts = [y[0].copy()]
    tau = 0.0
    lost = False
    with _few_threads():
        return _fd_steps(sec, y, q, h, pts, tau, lost)


def _fd_steps(sec, y, q, h, pts, tau, lost):
    for j in range(q):
        y, l, t = sec.step(y, np.ones(len(y)))
        tau += float(t[0])
        lost |= bool(np.any(l))
        if j < q - 1:
            pts.append(y[0].copy())
    M = np.stack([(y[1] - y[2]) / (2 * h), (y[3] - y[4]) / (2 * h)], -1)
    return M, y[0], np.array(pts), tau, lost


def x_orbit_from_line(sec, engine, line, phi_s, polish=8):
    """The X orbit in the section: Newton on P^q(x) = x from the closed line's point, e_X = 2 |(M - I)^-1|
    (closure + tau_q) (Newton-Kantorovich).  The monodromy M (and its error dM, which bounds the unstable
    eigenvector's direction):
    * the line's spectral monodromy (variational equation along the closed line; dM from two step
      counts) when the polished point is the line's own point to within the Newton-Kantorovich radius --
      a finite-difference monodromy of the accurate tracer is far too noisy for a weak X line;
    * otherwise (the line is only as accurate as its Fourier truncation, ~1e-5 m when started from an
      unconverged surface, and the spectral M along it is then not the orbit's), with a surrogate map: the
      product of its analytic Jacobians along the polished orbit, dM from the accurate tracer's (48 steps
      per period) central differences at steps h, h/2, h/4 (h = 3e-5 scale) Richardson-extrapolated
      (_richardson_check);
    * with the accurate map only, when Newton with the spectral M does not converge: Newton with the
      accurate tracer's central differences (48 steps per period), dM from two step sizes and the noise.
    Returns (torux.barrier.Orbit, info) or (None, info)."""
    q = line.q
    M_spec, dM = line_monodromy_plane(engine, line, phi_s)
    info = dict(dM=dM, residue_spectral=float((2 - np.trace(M_spec)) / 4), det=float(np.linalg.det(M_spec)))
    x_line = line.section_points(phi_s)[0].copy()
    x = x_line.copy()
    exact = getattr(sec, "exact", None)
    smooth = exact is not None                       # a surrogate: its finite differences are clean
    h = 1e-6 * sec.scale
    M = M_spec
    best = None
    for it in range(polish + 1):
        if smooth:
            M, y0, pts, tau, lost = section_monodromy(sec, x, q, h)
        else:
            y = x[None].copy()
            pts = [x.copy()]
            tau = 0.0
            lost = False
            for j in range(q):
                y, l, t = sec.step(y, np.ones(1))
                tau += float(t[0])
                lost |= bool(l[0])
                if j < q - 1:
                    pts.append(y[0].copy())
            y0 = y[0]
        if lost:
            info.setdefault("lost_at_iteration", it)
            break
        Fv = y0 - x
        clos = float(np.linalg.norm(Fv))
        info.setdefault("closures", []).append(clos)
        if best is None or clos < best[1]:
            best = (np.array(pts), clos, tau, M.copy())
        else:
            break
        if clos < 1e-15 * sec.scale:
            break
        x = x - np.linalg.solve(M - np.eye(2), Fv)
    if not smooth and (best is None or best[1] > 1e-11 * sec.scale):
        # Newton with the (fixed) spectral M did not converge (the line is off the orbit and the orbit weakly
        # hyperbolic): Newton with the finite-difference monodromy of a fine accurate section instead
        fine = PlaneSection(sec.engine, sec.source, phi_s=sec.phi_s, sign=sec.sign, scale=sec.scale, reference=False,
                            steps_per_period=48)
        x = x_line.copy()
        h_ex = 3e-5 * sec.scale
        for it in range(polish + 1):
            M2, y0, pts2, tau2, lost2 = fd_monodromy(fine, x, q, h_ex)
            if lost2:
                break
            Fv = y0 - x
            clos2 = float(np.linalg.norm(Fv))
            info.setdefault("closures_fd", []).append(clos2)
            if best is None or clos2 < best[1]:
                best = (np.array(pts2), clos2, tau2, M2.copy())
                info["monodromy_fd_newton"] = True
            elif it > 0:
                break
            if clos2 < 1e-15 * sec.scale:
                break
            x = x - np.linalg.solve(M2 - np.eye(2), Fv)
    if best is None:
        return None, info
    pts, clos, tau, M = best
    dev = float(np.linalg.norm(pts[0] - x_line))
    r_spec = 2.0 * float(np.linalg.norm(np.linalg.inv(M_spec - np.eye(2)), 2)) * (clos + tau)
    info["dev_line_point"] = dev
    if info.get("monodromy_fd_newton") and dev > max(r_spec, 1e-12 * sec.scale):
        # the accurate tracer's finite-difference monodromy at the polished point, its error from the
        # Richardson-extrapolated differences at three step sizes
        fine = PlaneSection(sec.engine, sec.source, phi_s=sec.phi_s, sign=sec.sign, scale=sec.scale, reference=False,
                            steps_per_period=48)
        dM_use = _richardson_check(fine, pts[0], q, 1e-4 * sec.scale, M)
        info["monodromy"] = "finite differences of the accurate tracer (48 steps per period)"
    elif not smooth or dev <= max(r_spec, 1e-12 * sec.scale):
        M, dM_use = M_spec, dM
        info["monodromy"] = "spectral"
    else:
        # checked against the accurate tracer's Richardson-extrapolated central differences
        h_ex = 3e-5 * sec.scale
        fine = PlaneSection(exact.engine, exact.source, phi_s=exact.phi_s, sign=exact.sign, scale=exact.scale,
                            reference=False, steps_per_period=48)
        dM_use = _richardson_check(fine, pts[0], q, h_ex, M)
        info.update(monodromy="the surrogate's analytic Jacobians (checked against the accurate tracer)"
                    if hasattr(sec, "jacobian") else "finite differences of the surrogate (checked against the "
                    "accurate tracer)", dM_fd_vs_exact=dM_use)
    inv = np.linalg.inv(M - np.eye(2))
    nrm = float(np.linalg.norm(inv, 2))
    e_X = 2.0 * nrm * (clos + tau)
    orb = BR.Orbit(q, pts, M, e_X=e_X, info=dict(closure=clos, tau_q=tau, inv_norm=nrm, det=float(np.linalg.det(M)),
                                                 monodromy=info["monodromy"], dM=dM_use))
    return orb, info


def _richardson_check(fine, x, q, h, M):
    """Entrywise error bound of a monodromy M at x against the accurate tracer: central differences at
    steps h, h/2, h/4 (one batched trace of the three stencils, one thread), Richardson-extrapolated twice
    (O(h^4)); dM = |M - R(h/2, h/4)| + |R(h, h/2) - R(h/2, h/4)| (the extrapolation's own error) +
    2 tau_q / (h / 4) (the tracer's noise)."""
    x = np.asarray(x, float)
    E = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    y = np.concatenate([x[None] + f_ * h * E for f_ in (1.0, 0.5, 0.25)])
    tau = np.zeros(len(y))
    with _few_threads():
        for _ in range(q):
            y, lost, t = fine.step(y, np.ones(len(y)))
            if np.any(lost):
                return np.inf
            tau += t
    Ms = []
    for b, f_ in enumerate((1.0, 0.5, 0.25)):
        Y = y[5 * b:5 * b + 5]
        hh = f_ * h
        Ms.append(np.stack([(Y[1] - Y[2]) / (2 * hh), (Y[3] - Y[4]) / (2 * hh)], -1))
    R1 = (4.0 * Ms[1] - Ms[0]) / 3.0
    R2 = (4.0 * Ms[2] - Ms[1]) / 3.0
    return float(np.max(np.abs(M - R2)) + np.max(np.abs(R1 - R2))) + 2.0 * float(np.max(tau)) / (0.25 * h)

def eigen_error(M, dM):
    """First-order bound on the angle error of M's eigenvectors from an entrywise error dM:
    kappa(V) |E| / |lambda - mu| (V the eigenvector matrix), with |E| <= 2 dM."""
    w, V = np.linalg.eig(M)
    w = np.real(w)
    V = np.real(V)
    kap = float(np.linalg.cond(V))
    return kap * 2.0 * 2.0 * dM / max(abs(w[0] - w[1]), 1e-300)


def _angle(pts, a):
    pts = np.atleast_2d(pts)
    return np.arctan2(pts[:, 1] - a[1], pts[:, 0] - a[0])


def _first_axis_crossing(P, start=1):
    """Indices i (>= start) of the segments P[i] -> P[i+1] crossing Z = 0 (sign change), in order."""
    Z = P[:, 1]
    s = np.sign(Z)
    idx = np.flatnonzero((s[start:-1] * s[start + 1:] < 0)) + start
    return idx


def _nearest_on_polyline(P, x):
    """(segment index i, parameter t, distance) of the point of the polyline P nearest to x."""
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.sum(ab * ab, 1)
    t = np.clip(np.sum((x[None] - a) * ab, 1) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    d = np.linalg.norm(a + t[:, None] * ab - x[None], axis=1)
    i = int(np.argmin(d))
    return i, float(t[i]), float(d[i])


# ============================================================================ the certificate
def _straddled_plane(Xl, others, p, q, nfp):
    """(the other symmetric line used or None, plane, X lines straddled there) for the first symmetry plane
    (0: phi = 0, 1: phi = pi / nfp) in which some X line's points contain a mirror pair X_k, S X_k of chain
    neighbours (order along the chain: k p mod q); the planes of the other symmetric lines (O, reflection-
    hyperbolic) first.  (None, None, []) when neither plane qualifies."""
    cand = []
    for l in others:
        pl = int(round((float(l.phi0) % (2 * np.pi / nfp)) / (np.pi / nfp))) % 2
        if pl not in [c[1] for c in cand]:
            cand.append((l, pl))
    for pl in (0, 1):
        if pl not in [c[1] for c in cand]:
            cand.append((None, pl))
    order = sorted(range(q), key=lambda k: (k * int(p)) % q) if math.gcd(int(p), q) == 1 else list(range(q))
    pos = {k: i for i, k in enumerate(order)}
    for (l, pl) in cand:
        phi_s = pl * np.pi / nfp
        good = []
        for X in Xl:
            P = X.section_points(phi_s)
            if len(P) != q:
                continue
            d = np.linalg.norm(P[:, None, :] - (P * np.array([1.0, -1.0]))[None, :, :], axis=-1)
            m = np.argmin(d, axis=1)            # m[k]: the index of S P_k
            if any(m[k] != k and (pos[m[k]] - pos[k]) % q in (1, q - 1) for k in range(q)):
                good.append(X)
        if good:
            return l, pl, good
    return None, None, []


def island_band_certificate_manifold(engine, source, T1, T2, p, q, request, deadline=None, verbose=False,
                                     lines=None, s0=None, n0=None, dmax=None, amax=0.2, target=None,
                                     max_levels=None, max_pts=6000, reference=True, section=None,
                                     surrogate="auto", surrogate_opts=None, bracket_check=True, keep_polygons=False,
                                     pairs_fallback=False, axis_inverse=False, axis_line=None):
    """The manifold island-band certificate (module docstring) for the chain p/q (per field period)
    between the verified tori T1 (inside, |flux| F1) and T2 (beyond, F2).  Returns a JSON-serialisable
    dict: ok, kind="island_band_manifold", interval=(F_lo, F_hi), holds_request, chain (p, q, residues
    of its X and O lines), X lines used, error_budget, evidence (reference checks), timings, reason
    when not ok.

    T1, T2 serve as the start of the closed-line search, the annulus of the period-map surrogate and a
    consistency check (the band must lie inside (F1, F2) when they are verified tori); the theorem does
    not use them.  bracket_check=False: T1, T2 are only nested surfaces around the chain (proxies), the
    consistency check is skipped (recorded).

    lines: optional precomputed closed lines of the chain (torux.orbits.ClosedLine list); else they are
    found by torux.chains.chain_lines from the torus nearer the resonance.  section: optional section
    object (a PlaneSection or a validated surrogate with the same interface) used for the manifolds;
    surrogate="auto" (default) builds a BandSurrogate (else a PatchSurrogate; surrogate_opts["kinds"]) on the
    annulus between T1 and T2 and uses it when its validated error is below 1e-9 x scale.  keep_polygons:
    include the certified polygons (vertex lists) in the result.  pairs_fallback: a chain without a usable
    symmetric X line goes to island_band_certificate_pairs (hyperbolic orbits found on the surrogate, both
    symmetry planes).  axis_line: the analysis's period-one line D; the axis point must be D's (module docstring,
    'Axis identity'; refused otherwise, the check recorded as axis_identity)."""
    t0 = time.time()
    req = None if request is None else abs(float(request))
    F1, F2 = abs(float(T1.flux)), abs(float(T2.flux))
    out = dict(kind="island_band_manifold", ok=False, request=req, F1=F1, F2=F2, holds_request=False,
               chain=dict(p=int(p), q=int(q)), timings={})

    def done(reason=None):
        if reason:
            out["reason"] = reason
        out["timings"]["total"] = time.time() - t0
        if verbose and reason:
            print(f"  manifold band: {reason}", flush=True)
        return BR._jsonable(out)

    try:
        nfp = int(source.nfp)
        if not getattr(source, "stellsym", False):
            return done("source not stellarator symmetric (the certificate needs the exact symmetry)")
        out["source_symmetry"] = (getattr(source, "meta", None) or {}).get("symmetry_check")
        if q < 2:
            return done("q = 1 chains are not covered (homoclinic structure)")
        if bracket_check and not (F1 < F2):
            return done("bracket tori not ordered (F1 >= F2)")
        # ---------------------------------------------------------------- the chain's closed lines
        if lines is None:
            from .chains import chain_lines
            iota_r = p / q * nfp
            base = T1 if abs(T1.iota - iota_r) <= abs(T2.iota - iota_r) else T2
            lines = chain_lines(engine, base, p, q, verbose=verbose)
        out["timings"]["lines"] = time.time() - t0
        Xl = [l for l in lines if l.residue < 0]
        Ol = [l for l in lines if 0 < l.residue < 1]
        out["chain"].update(X_residues=[float(l.residue) for l in Xl], O_residues=[float(l.residue) for l in Ol],
                            reflection_residues=[float(l.residue) for l in lines if l.residue >= 1])
        if verbose:
            print(f"  chain {p}/{q}: X residues {[f'{l.residue:.3e}' for l in Xl]} O residues "
                  f"{[f'{l.residue:.3e}' for l in Ol]}", flush=True)
        if Xl and all(abs(l.residue) < MIN_RESIDUE for l in Xl):
            return done(f"chain too weak for the manifold method: X residues "
                        f"{[f'{l.residue:.2e}' for l in Xl]} below {MIN_RESIDUE:.1e} in modulus (lambda - 1 < 3.2e-5)")
        Xl = [l for l in Xl if abs(l.residue) >= MIN_RESIDUE]
        if not Xl:
            if Ol and all(abs(l.residue) < MIN_RESIDUE for l in Ol):
                return done(f"chain too weak for the manifold method: no symmetric X line, O residues "
                            f"{[f'{l.residue:.2e}' for l in Ol]} below {MIN_RESIDUE:.1e}")
            if pairs_fallback:
                r = island_band_certificate_pairs(engine, source, T1, T2, p, q, request, deadline=deadline,
                                                  verbose=verbose, lines=lines, reference=reference,
                                                  bracket_check=bracket_check, axis_inverse=axis_inverse,
                                                  axis_line=axis_line)
                r["single_orbit_reason"] = "no hyperbolic symmetric line of the chain"
                return r
            return done("no hyperbolic symmetric line of the chain")
        # the plane: one in which Fix(S) = {Z = 0} straddles a gap of the X orbit (a mirror pair X_k, S X_k of
        # chain neighbours) -- first the symmetry plane of the chain's other symmetric line (O, or a
        # reflection-hyperbolic centre: the theorem does not use it), checked on the X lines' points
        Rl = [l for l in lines if l.residue >= 1]
        O, plane, Xl = _straddled_plane(Xl, Ol + Rl, p, q, nfp)
        if plane is None:
            return done("no symmetry plane in which Z = 0 straddles a gap of the symmetric X orbit")
        phi_s = plane * np.pi / nfp
        out["phi_s"] = phi_s
        # ---------------------------------------------------------------- section, axis, X orbit
        P1, P2 = torus_section(T1, phi_s), torus_section(T2, phi_s)
        # the chain's lines must lie between T1 and T2 (a chain outside the bracket is not this request's)
        if bracket_check:
            Xl = [l for l in Xl if not np.any(BR.inside_polygon(P1, l.section_points(phi_s)))
                  and np.all(BR.inside_polygon(P2, l.section_points(phi_s)))]
            if not Xl or (O is not None and not (np.all(BR.inside_polygon(P2, O.section_points(phi_s)))
                                                  and not np.any(BR.inside_polygon(P1, O.section_points(phi_s))))):
                return done("the chain's lines do not lie between T1 and T2")
        scale = float(np.max(P2[:, 0]) - np.min(P2[:, 0]))
        Rmid = float(0.5 * (np.max(P1[:, 0]) + np.min(P1[:, 0])))
        sign = _bphi_sign(engine, Rmid, phi_s)
        sec_exact = PlaneSection(engine, source, phi_s=phi_s, sign=sign, scale=scale, reference=reference)
        # the axis point first (cheap; a refusal here must not cost a surrogate)
        ax_id = {}
        with _few_threads():
            ax = axis_point(sec_exact, T1, plane, inverse=axis_inverse, axis_line=axis_line, info=ax_id)
        if axis_line is not None:
            out["axis_identity"] = ax_id
        if ax is None:
            return done(ax_id.get("reason") or "magnetic axis point not found inside T1")
        out["axis_kind"] = ax.info.get("axis_kind", "elliptic")
        a = ax.pts[0]
        out["axis_point"] = a.tolist()
        out["axis_e"] = float(ax.e_X)
        if not BR.inside_polygon(P1, a[None])[0]:
            return done("axis point not inside T1's section")
        sec = section
        if sec is None and surrogate:
            # the period map on the annulus between T1 and T2 as a validated piecewise-Chebyshev surrogate (one
            # batch of accurate traces instead of thousands of small ones): patches aligned with the annulus
            # (BandSurrogate), else square cells (PatchSurrogate); refused if the validation error is large
            so = dict(surrogate_opts or {})
            max_tau = so.pop("max_tau", 1e-9 * scale)
            kinds = so.pop("kinds", ("band", "cells"))
            out["surrogate"] = []
            eng_tr = _tlean(engine, source, T2, out)        # (fast_trace 'tlean') else the engine itself
            for kind in kinds:
                try:
                    cls_ = BandSurrogate if kind == "band" else PatchSurrogate
                    sur = cls_(eng_tr, source, T1, T2, phi_s=phi_s, sign=sign, scale=scale, reference=reference, **so)
                    info = sur.build(verbose=verbose, deadline=deadline)
                    out["surrogate"].append(info)
                    if info["tau_annulus"] < max_tau and info["density_rel_err"] < 1e-6:
                        sec = sur
                        break
                except SurrogateDeadline as e:
                    # the other kind and the accurate tracer cost more: no time for this band
                    out["surrogate"].append(dict(kind=kind, error=repr(e)))
                    return done(f"surrogate build refused: {e}")
                except Exception as e:  # noqa: BLE001
                    if environment_failure(e):
                        raise
                    out["surrogate"].append(dict(kind=kind, error=repr(e)))
            if sec is None and surrogate is True:
                return done(f"surrogate map not accurate enough or failed ({out['surrogate']})")
            out["timings"]["surrogate"] = time.time() - t0
        if sec is None:
            # no surrogate: the accurate tracer at 24 steps per period (at 12, the GBS estimate was seen to fall
            # below the reference tracer's difference on strongly shaped fields: refused by the evidence)
            sec = PlaneSection(_tlean(engine, source, T2, out), source, phi_s=phi_s, sign=sign, scale=scale,
                               reference=reference, steps_per_period=24)
        out["section"] = type(sec).__name__
        best = None
        out["x_orbit_attempts"] = []
        for Xline in Xl:
            X0 = Xline.section_points(phi_s)
            with _few_threads():
                orb, minfo = x_orbit_from_line(sec, engine, Xline, phi_s)
            if orb is not None:
                minfo["dev_from_line"] = float(np.max(np.linalg.norm(orb.pts - X0, axis=1)))
                minfo["e_X"] = float(orb.e_X)
            out["x_orbit_attempts"].append(minfo)
            if orb is None:
                if verbose:
                    print(f"  X orbit from the line (residue {Xline.residue:.3e}) failed: {minfo}", flush=True)
                continue
            if verbose:
                print(f"  X orbit from the line (residue {Xline.residue:.3e}): {minfo}", flush=True)
            # the polished orbit must be the line's (not another chain's): within a small fraction of the
            # chain's scale, or of its own error bound
            if minfo["dev_from_line"] > max(1e-4 * scale, 10.0 * orb.e_X):
                continue
            if best is None or orb.e_X < best[0].e_X:
                best = (orb, Xline)
        if best is not None and best[0].e_X > 1e-3 * scale:
            return done(f"X orbit not determined (e_X {best[0].e_X:.2e}: Newton from the closed line did not converge)")
        if best is None:
            return done("X orbit not reproduced by the section's map")
        orb, Xline = best
        orb.p = int(p)
        eig = orb.eig()
        out["X_line"] = dict(residue_line=float(Xline.residue), residue=orb.residue, lam=eig["lam"], e_X=orb.e_X,
                             points=orb.pts.tolist(), closure=orb.info["closure"], phi0_line=float(Xline.phi0),
                             monodromy_err=orb.info.get("dM"), eigvec_angle_err=eigen_error(orb.M, orb.info["dM"]))
        out["O_line"] = None if O is None else dict(residue=float(O.residue), points=O.section_points(phi_s).tolist())
        if verbose:
            print(f"  plane phi_s {phi_s:.4f}: X orbit residue {orb.residue:.4e} lambda {eig['lam']:.6f} "
                  f"e_X {orb.e_X:.2e}; axis ({a[0]:.6f}, {a[1]:.2e}) ({time.time() - t0:.0f}s)", flush=True)
        # orbit inside T2 and outside T1 (the chain lies between the bracket tori)
        if bracket_check and (np.any(BR.inside_polygon(P1, orb.pts)) or not np.all(BR.inside_polygon(P2, orb.pts))):
            return done("X orbit not between T1 and T2 in the section")
        res = band_from_orbit(sec, orb, a, axis_e=float(ax.e_X), scale=scale, deadline=deadline, verbose=verbose,
                              t0=t0, s0=s0, n0=n0, dmax=dmax, amax=amax, target=target, max_levels=max_levels,
                              max_pts=max_pts)
        out.update({k: v for k, v in res.items() if k not in ("ok", "reason", "polygons", "timings")})
        out["timings"].update(res.get("timings", {}))
        if not res["ok"]:
            return done(res.get("reason"))
        lo, hi = res["interval"]
        tol_b = 1e-12 * F2
        out["inside_bracket"] = bool(F1 - tol_b <= lo and hi <= F2 + tol_b)
        out["bracket_check"] = bool(bracket_check)
        out["bracket_tori"] = _bracket_tori_record(T1, T2) if bracket_check else None
        if bracket_check and not out["inside_bracket"]:
            return done(f"band ({lo:.10e}, {hi:.10e}) not inside the bracket ({F1:.10e}, {F2:.10e}): inconsistent")
        # ---------------------------------------------------------------- independent evidence
        ev = dict(res.get("evidence", {}))
        if reference:
            ev["closure_reference"] = reference_closure_plane(sec_exact, orb)
            ev["flux_reference"] = {k: BR.reference_flux(sec, np.array(v)) for k, v in res["polygons"].items()}
            ev["heteroclinic_reference"] = res.get("heteroclinic_reference")
        out["evidence"] = ev
        ok_ev = bool(ev.get("ok", True)) and (not reference or (
            ev["closure_reference"].get("ok", False) and all(v.get("ok", False) for v in ev["flux_reference"].values())
            and (ev.get("heteroclinic_reference") or {}).get("ok", False)))
        out["evidence_ok"] = ok_ev
        if keep_polygons:
            out["polygons"] = res["polygons"]
        if not ok_ev:
            return done("reference checks failed")
        out["ok"] = True
        out["rigorous"] = True          # in torux/barrier.py's sense: strict lemmas + validated error models
        if req is not None:
            out["holds_request"] = bool(lo < req < hi)
            out["margin_rel"] = float(min(req - lo, hi - req) / req)
        return done()
    except Exception as e:  # noqa: BLE001
        import traceback
        out["traceback"] = traceback.format_exc()
        return done(f"error: {e!r}")



class OffsetSurface:
    """The surface base offset along the in-plane normal of its sections: x = base(theta, phi) + (d +
    d_theta(theta, phi)) n(theta, phi) (n the outward unit normal of the section curve phi = const; d < 0
    inwards; d_theta an optional smooth variable offset, defined in the two symmetry planes).  Duck-types
    the TorusState methods the certificate uses for T1, T2 (geometry_on, flux, iota, nfp)."""

    def __init__(self, base, d, dvar=None):
        self.base, self.d = base, float(d)
        self.dvar = dvar                                  # {plane (0 or 1): Fourier coefficients (2, K+1)}
        self.nfp, self.iota, self.flux, self.phi0 = base.nfp, base.iota, base.flux, base.phi0
        self.M, self.N = base.M, base.N
        th = 2 * np.pi * np.arange(256) / 256
        R, Z = base.geometry_on(th, np.zeros(256))
        self.orient = 1.0 if BR.signed_area(np.stack([R, Z], -1)) > 0 else -1.0

    def offset(self, theta, phi):
        theta = np.asarray(theta, float)
        if self.dvar is None:
            return np.full(theta.shape, self.d)
        per = 2 * np.pi / self.nfp
        ph = np.mod(np.asarray(phi, float) - self.phi0, per)
        plane = np.where(np.abs(ph - 0.5 * per) < 1e-9, 1, 0)
        if np.any((np.abs(ph) > 1e-9) & (np.abs(ph - per) > 1e-9) & (plane == 0)):
            raise ValueError("variable offsets are defined in the symmetry planes only")
        out = np.full(theta.shape, self.d)
        for pl, cf in self.dvar.items():
            sel = plane == pl
            if np.any(sel):
                m = np.arange(cf.shape[1])
                a = np.multiply.outer(theta[sel], m)
                out[sel] += np.cos(a) @ cf[0] + np.sin(a) @ cf[1]
        return out

    def geometry_on(self, theta, phi):
        b = self.base
        theta = np.asarray(theta, float)
        phi = np.broadcast_to(np.asarray(phi, float), theta.shape)
        m = np.arange(b.M + 1)[:, None]
        n = np.arange(-b.N, b.N + 1)[None, :]
        ang = m * theta[..., None, None] - n * b.nfp * (phi[..., None, None] - b.phi0)
        rc, zs = np.asarray(b.rc), np.asarray(b.zs)
        R = np.sum(rc * np.cos(ang), axis=(-2, -1))
        Z = np.sum(zs * np.sin(ang), axis=(-2, -1))
        Rt = np.sum(-m * rc * np.sin(ang), axis=(-2, -1))
        Zt = np.sum(m * zs * np.cos(ang), axis=(-2, -1))
        nr = np.hypot(Rt, Zt)
        # outward normal of a counter-clockwise curve: (Z_t, -R_t) / |.|
        nR, nZ = self.orient * Zt / nr, -self.orient * Rt / nr
        dd = self.offset(theta, phi)
        return R + dd * nR, Z + dd * nZ


_TLEAN = {}


def _tlean_region(T2, nth=256, nph=64):
    """(fast_trace 'tlean') Points of the annulus' outer torus over the whole torus and their spacing: T2 itself, or for
    an offset proxy its base offset by the largest outward offset of its two symmetry planes everywhere (a variable
    offset is defined in those planes only).  Separable synthesis on the tensor grid (memory O(nth M + nph N), any
    resolution: the pointwise geometry_on held nth nph (M + 1) (2 N + 1) numbers -- 12 GB for a refined bracket torus)."""
    nfp = int(T2.nfp)
    per = 2 * np.pi / nfp
    base, dmax = T2, 0.0
    if isinstance(T2, OffsetSurface):
        th = 2 * np.pi * np.arange(512) / 512
        dmax = max(float(np.max(T2.offset(th, np.full(512, T2.phi0 + pl * 0.5 * per)))) for pl in (0, 1))
        base = T2.base
    th = 2 * np.pi * np.arange(nth) / nth
    ph = T2.phi0 + 2 * np.pi * np.arange(nph * nfp) / (nph * nfp)
    rc, zs = np.asarray(base.rc, float), np.asarray(base.zs, float)
    m = np.arange(base.M + 1)
    n = np.arange(-base.N, base.N + 1)
    at = np.outer(th, m)
    ap = np.outer(base.nfp * (ph - base.phi0), n)
    Ct, St, Cp, Sp = np.cos(at), np.sin(at), np.cos(ap), np.sin(ap)
    R = Ct @ rc @ Cp.T + St @ rc @ Sp.T
    Z = St @ zs @ Cp.T - Ct @ zs @ Sp.T
    if dmax != 0.0:
        mrc, mzs = m[:, None] * rc, m[:, None] * zs
        Rt = -(St @ mrc @ Cp.T - Ct @ mrc @ Sp.T)
        Zt = Ct @ mzs @ Cp.T + St @ mzs @ Sp.T
        nr = np.hypot(Rt, Zt)
        R = R + dmax * T2.orient * Zt / nr
        Z = Z - dmax * T2.orient * Rt / nr
    PH = np.broadcast_to(ph[None, :], R.shape)
    X = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1)
    h = math.hypot(float(np.max(np.linalg.norm(np.roll(X, -1, 0) - X, axis=-1))),
                   float(np.max(np.linalg.norm(np.roll(X, -1, 1) - X, axis=-1))))
    return X.reshape(-1, 3), h


def _tlean(engine, source, T2, out=None):
    """(fast_trace 'tlean') The certified lean engine for the B-only tracing around the annulus whose outer torus is T2
    (torux.fast_trace.lean_trace_engine; built once per engine and T2), else the engine itself."""
    from . import fast_trace as _so4
    if not _so4.on("tlean"):
        return engine
    key = (id(engine), id(T2))
    if key not in _TLEAN:
        try:
            X, h = _tlean_region(T2)
            le, info = _so4.lean_trace_engine(engine, source, X, region_h=h)
        except Exception as e:  # noqa: BLE001
            if environment_failure(e):
                raise
            le, info = None, dict(refused=repr(e)[:200])
        _TLEAN.clear()
        _TLEAN[key] = (le, info, engine, T2)          # the objects kept: their ids stay unique while cached
    le, info = _TLEAN[key][:2]
    if out is not None:
        out.setdefault("tlean", []).append(dict(info))
    return le if le is not None else engine


def chain_flux_estimate(engine, source, lines, p, q, phi_s=0.0, n=2048):
    """A cheap estimate of the flux where the chain p/q sits (not a bound): the toroidal flux through the
    smooth closed curve through its closed lines' points in the plane phi_s, in the order along the chain
    (torus angle k p mod q; trigonometric interpolation of degree (q - 1) / 2: a q-gon's chords would cut
    off ~ (2 pi / q)^2 / 6 of the flux).  The X lines' points when there are some, else the others'.  On
    certified bands with q >= 10 the estimate lies inside the band; for q < 8 it can be off by ~10 %.
    For ordering candidate chains by the distance of a request from them.  Returns dict(flux, points)."""
    Xl = [l for l in lines if l.residue < 0] or list(lines)
    P = np.asarray(Xl[0].section_points(phi_s), float)
    order = sorted(range(q), key=lambda k: (k * int(p)) % q) if math.gcd(int(p), q) == 1 else list(range(q))
    C = P[order]
    nh = (q - 1) // 2
    m = np.arange(nh + 1)
    t = 2 * np.pi * np.arange(n) / n
    w = np.where(m == 0, 1.0, 2.0)
    E = np.exp(1j * np.outer(t, m))
    Cs = np.stack([np.real(E @ (w * (np.fft.fft(C[:, j]) / q)[:nh + 1])) for j in (0, 1)], -1)
    Rmid = float(0.5 * (Cs[:, 0].max() + Cs[:, 0].min()))
    sec = PlaneSection(engine, source, phi_s=phi_s, sign=_bphi_sign(engine, Rmid, phi_s),
                       scale=float(np.ptp(Cs[:, 0])), reference=False)
    F, _ = sec.flux(Cs)
    return dict(flux=float(abs(F)), points="X" if Xl[0].residue < 0 else "other", q=int(q))


def proxy_annulus(base, lines, margin=3.0, n=1024, kmax=8):
    """Two surfaces around a chain from a surface `base` near it (a TorusState, e.g. a design surface's
    start), in the two symmetry planes: base offset along its sections' normals by d0(theta) + [d_in, d_out],
    d0 a smooth Fourier fit (<= kmax harmonics and <= (points - 1) / 3, by least squares; fewer if the fit
    leaves the points' offset range widened by half of it anywhere on the curve) of the normal offsets of the
    chain's closed-line points from base, so that the band follows the chain even where base is not parallel to it;
    d_in, d_out = the residuals' range widened by margin x max(the residuals' spread, a twentieth of the
    mean spacing of the chain points along the curve) on each side (the separatrices reach beyond the lines
    by about the island width).  The surrogate's patches follow the normal segments (the offset curves need
    not be simple).  For the surrogate's annulus only; not invariant tori (use with bracket_check=False)."""
    th = 2 * np.pi * np.arange(n) / n
    dvar = {}
    res_all = []
    spacing = []
    off_all = []
    for plane, phi_s in ((0, 0.0), (1, np.pi / base.nfp)):
        ref = SectionCurve(section_coeffs(base, phi_s))
        P = np.concatenate([ln.section_points(phi_s) for ln in lines])
        tt, dd, ok = ref.coords(P)
        if np.any(ok):
            off_all.extend(np.asarray(dd)[ok].tolist())
        if not np.all(ok):
            # fall back to the polygon distance (beyond the reach of the base's normal coordinates)
            R, Z = base.geometry_on(th, np.full(n, phi_s))
            C = np.stack([R, Z], -1)
            d_ = BR.polyline_distance(P, np.concatenate([C, C[:1]]))
            dd = np.where(BR.inside_polygon(C, P), -d_, d_)
            dvar = None
            res_all.extend(dd.tolist())
            continue
        # the largest K <= kmax whose fit does not overshoot between the points (a fit with nearly as many
        # coefficients as points can swing far from the data between them): its range over the curve within
        # the data's range widened by half of it
        span_d = float(dd.max() - dd.min())
        for K in range(int(min(kmax, (len(P) - 1) // 3)), -1, -1):
            m = np.arange(K + 1)
            A = np.concatenate([np.cos(np.outer(tt, m)), np.sin(np.outer(tt, m[1:]))], 1)
            coef, *_ = np.linalg.lstsq(A, dd, rcond=None)
            Af = np.concatenate([np.cos(np.outer(th, m)), np.sin(np.outer(th, m[1:]))], 1)
            fit = Af @ coef
            if fit.min() >= dd.min() - 0.5 * span_d - 1e-15 and fit.max() <= dd.max() + 0.5 * span_d + 1e-15:
                break
        cf = np.zeros((2, K + 1))
        cf[0] = coef[:K + 1]
        cf[1, 1:] = coef[K + 1:]
        if dvar is not None:
            dvar[plane] = cf
        res_all.extend((dd - A @ coef).tolist())
        x, x1, _ = ref.eval(np.sort(tt))
        ds = np.linalg.norm(np.diff(np.concatenate([x, x[:1]]), axis=0), axis=1)
        spacing.append(float(np.mean(ds)))
    res = np.array(res_all)
    w = max(float(res.max() - res.min()), 0.05 * (min(spacing) if spacing else 0.0), 1e-6)
    d_in, d_out = float(res.min() - margin * w), float(res.max() + margin * w)
    Tin, Tout = OffsetSurface(base, d_in, dvar), OffsetSurface(base, d_out, dvar)
    info = dict(d_in=d_in, d_out=d_out, width_unit=w, variable=dvar is not None,
                residual_range=(float(res.min()), float(res.max())), mean_spacing=spacing,
                offset_range=(float(min(off_all)), float(max(off_all))) if off_all else None)
    return Tin, Tout, info

def proxy_annulus_sym(base, lines, margin=3.0, n=1024, kmax=16):
    """proxy_annulus with the symmetry used: the chain's points are S-symmetric and the base's parameter is
    symmetric in both symmetry planes (theta -> -theta is Z -> -Z), so their normal offsets are an even function of
    theta -- a cosine series, fitted with up to (distinct |theta| values) - 1 harmonics (proxy_annulus fitted cosines
    and sines with at most (points - 1) / 3 harmonics: for the low-order chains of failed design tori the residuals
    were +-1-2 cm, and the inner offset surface reached the axis -- 'magnetic axis point not found inside T1').  The
    same overshoot test, widths and margins as proxy_annulus.  Not invariant tori (bracket_check=False)."""
    th = 2 * np.pi * np.arange(n) / n
    dvar = {}
    res_all = []
    spacing = []
    off_all = []
    fits = {}
    for plane, phi_s in ((0, 0.0), (1, np.pi / base.nfp)):
        ref = SectionCurve(section_coeffs(base, phi_s))
        P = np.concatenate([ln.section_points(phi_s) for ln in lines])
        tt, dd, ok = ref.coords(P)
        if not np.all(ok):
            return proxy_annulus(base, lines, margin=margin, n=n)
        off_all.extend(np.asarray(dd).tolist())
        # distinct |theta| (mirror pairs share theta up to sign)
        a = np.abs(np.angle(np.exp(1j * tt)))
        uniq = np.unique(np.round(a / 1e-9)).size
        span_d = float(dd.max() - dd.min())
        K = int(min(kmax, max(uniq - 1, 0)))
        while True:
            m = np.arange(K + 1)
            A = np.cos(np.outer(tt, m))
            coef, *_ = np.linalg.lstsq(A, dd, rcond=None)
            fit = np.cos(np.outer(th, m)) @ coef
            if K == 0 or (fit.min() >= dd.min() - 0.5 * span_d - 1e-15 and fit.max() <= dd.max() + 0.5 * span_d + 1e-15):
                break
            K -= 1
        cf = np.zeros((2, K + 1))
        cf[0] = coef
        dvar[plane] = cf
        fits[plane] = K
        res_all.extend((dd - A @ coef).tolist())
        x, x1, _ = ref.eval(np.sort(tt))
        ds = np.linalg.norm(np.diff(np.concatenate([x, x[:1]]), axis=0), axis=1)
        spacing.append(float(np.mean(ds)))
    res = np.array(res_all)
    w = max(float(res.max() - res.min()), 0.05 * (min(spacing) if spacing else 0.0), 1e-6)
    d_in, d_out = float(res.min() - margin * w), float(res.max() + margin * w)
    Tin, Tout = OffsetSurface(base, d_in, dvar), OffsetSurface(base, d_out, dvar)
    info = dict(d_in=d_in, d_out=d_out, width_unit=w, variable=True, symmetric_fit=fits,
                residual_range=(float(res.min()), float(res.max())), mean_spacing=spacing,
                offset_range=(float(min(off_all)), float(max(off_all))) if off_all else None)
    return Tin, Tout, info


def _bracket_tori_record(T1, T2):
    """The bracket tori's off-grid |B.n|/|B| and whether both meet the surface acceptance bar
    (torux.analysis.BN_SURFACE): only then are they verified surfaces at the band's edges; the bracket
    consistency check refuses a band reaching beyond them either way (conservative)."""
    try:
        from .analysis import BN_SURFACE
    except Exception:  # noqa: BLE001
        BN_SURFACE = 1e-14
    bn = [float(getattr(T, "bn_off", np.nan)) for T in (T1, T2)]
    return dict(bn_off=bn, bar=float(BN_SURFACE),
                verified=bool(all(np.isfinite(b) and b <= BN_SURFACE for b in bn)))


def band_from_orbit(sec, orb, a, axis_e=0.0, scale=None, deadline=None, verbose=False, t0=None, **kw):
    """The manifold island band (module docstring) from a symmetric hyperbolic periodic orbit `orb`
    (torux.barrier.Orbit, pts[j] = P^j pts[0]) of the reversible section map `sec` (S(R, Z) = (R, -Z)),
    around the point a (axis, error axis_e).  Returns dict(ok, interval, curves, ...)."""
    t0 = t0 or time.time()
    scale = scale or sec.scale
    eig = orb.eig()
    lam = abs(eig["lam"])
    out = dict(ok=False, timings={})
    if eig["reflection"] or not (lam > 1.0 + 1e-12 and abs(eig["mu"]) < 1.0):
        out["reason"] = "X orbit not (regular) hyperbolic"
        return out
    if orb.q < 2:
        out["reason"] = "q = 1 chains are not covered"
        return out
    # symmetric orbit: S X_0 = X_b
    Sx = orb.pts * np.array([1.0, -1.0])
    dmat = np.linalg.norm(Sx[:, None, :] - orb.pts[None, :, :], axis=-1)
    mism = float(np.max(np.min(dmat, axis=1)))
    if mism > 10.0 * orb.e_X + 1e-9 * scale:
        out["reason"] = f"X orbit not symmetric (mirror mismatch {mism:.2e})"
        return out
    bS = int(np.argmin(dmat[0]))
    # the cyclic order along the chain: X_k sits at torus angle theta_0 + 2 pi k p / q, so the order is that
    # of k p mod q (the polar angle about the axis is not monotonic along a chain in a bean-shaped section);
    # without p (model maps), the polar angle
    if orb.p is not None and math.gcd(int(orb.p), orb.q) == 1:
        order = sorted(range(orb.q), key=lambda k: (k * int(orb.p)) % orb.q)
    else:
        order = list(np.argsort(_angle(orb.pts, a)))
    pos = {k: i for i, k in enumerate(order)}
    q = orb.q
    succ = {k: order[(pos[k] + 1) % q] for k in range(q)}
    pred = {k: order[(pos[k] - 1) % q] for k in range(q)}
    out["chain_order"] = [int(k) for k in order]
    res = _grow_and_close(sec, orb, a, eig, lam, bS, succ, pred, mism, scale, deadline=deadline, verbose=verbose,
                          t0=t0, axis_e=axis_e, **kw)
    out.update(res)
    if not res["ok"]:
        return out
    cA, cB = res["curves"]["A"], res["curves"]["B"]
    lo = min(cA["F_up"], cB["F_up"])
    hi = max(cA["F_low"], cB["F_low"])
    out["interval"] = (float(lo), float(hi))
    out["error_budget"] = dict(
        e_X=orb.e_X, axis_e=float(axis_e), delta0=res["start_error"]["delta0"],
        A={k: cA[k] for k in ("eps_max", "eps_median", "tube", "quad_err", "junction_err_max")},
        B={k: cB[k] for k in ("eps_max", "eps_median", "tube", "quad_err", "junction_err_max")})
    if not lo < hi:
        out["ok"] = False
        out["reason"] = "the two separatrix curves enclose fluxes closer than their errors"
    return out

def reference_closure_plane(sec, orb):
    """|P_ref^q(X_0) - X_0| with the independent reference tracer, and the reference orbit's distance
    bound (Newton-Kantorovich with the section's monodromy)."""
    try:
        Y, E = sec.reference_map(orb.pts[0], orb.q)
        y = Y[-1, 0]
        d = float(np.linalg.norm(y - orb.pts[0]))
        inv = float(orb.info.get("inv_norm", np.inf))
        e_ref = 2.0 * inv * (d + float(E[-1, 0]))
        tol = max(10.0 * float(orb.e_X), 1e-12 * sec.scale)
        return dict(ok=bool(np.isfinite(e_ref) and e_ref <= tol), closure=d, e_X_reference=e_ref, tolerance=tol,
                    ref_err=float(E[-1, 0]))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=repr(e))


# ============================================================================ growth and closing
def _seg_errors(orb, c, br, delta0):
    """Per-segment error bounds of a branch polyline (vertex errors + chord errors)."""
    E = BR.vertex_errors(orb, c, br["refs"], delta0) + getattr(c, "const_err", 0.0)
    ch = np.concatenate([[0.0], BR.segment_chord_errors(br["pts"][1:])]) if len(br["pts"]) > 2 else np.zeros(1)
    return np.maximum(E[:-1], E[1:]) + ch[:len(E) - 1]


def _upto_level(refs, Lmax):
    """Number of leading vertices of a branch polyline whose piece level is <= Lmax (X counts)."""
    n = 0
    for (L, _) in refs:
        if L is not None and L > Lmax:
            break
        n += 1
    return max(n, 2)


def _new_z_crossing(c):
    """True if a piece grown since the last call (or the junction to its predecessor of the same owner)
    changes the sign of Z: a branch can only have gained a crossing with Z = 0 there.  (The full search
    rebuilds every branch polyline; this keeps the growth loop's checks O(new pieces).)"""
    last = getattr(c, "_zchecked", 0)
    q = c.q
    hit = False
    for L in range(max(last, 1), c.level() + 1):
        z = c.pieces[L].pts[:, 1]
        if len(z) and (np.any(z[:-1] * z[1:] <= 0) or (L - q >= 0 and len(c.pieces[L - q].pts)
                                                         and c.pieces[L - q].pts[-1, 1] * z[0] <= 0)):
            hit = True
    c._zchecked = c.level() + 1
    if hit:
        c._zhit = True
    return bool(getattr(c, "_zhit", False))


def _find_symmetric_crossing(sec, orb, a, c, sigma, succ, pred, delta0, first_owner=None, want_dir=None):
    """First certified crossing of an unstable branch of curve c with the line Z = 0 (Fix S), at an X_k
    whose mirror partner S X_k is a neighbour along the chain (succ: +1, pred: -1; want_dir restricts the
    sense).  Any crossing gives a heteroclinic point (h = S h is on W^s(S X_k)); up to four are tried in
    order along the branch.  Returns dict(owner, partner, dir, seg, h, box_diam, level, cert) or None."""
    q = orb.q
    owners = range(q) if first_owner is None else [first_owner]
    best = None
    for k in owners:
        if c.level() < (c.d * (k - c.base)) % q:
            continue
        nb = sigma[k]
        if nb == k:
            continue
        br = BR.branch_polyline(orb, c, k)
        P = br["pts"]
        if len(P) < 3:
            continue
        if succ[k] == pred[k]:
            # q = 2: both neighbours are the other point; the sense is the branch's polar sense about a
            d = int(BR._branch_dir(orb, a, br)) if nb == succ[k] else 0
        else:
            d = +1 if nb == succ[k] else (-1 if nb == pred[k] else 0)
        if d == 0 or (want_dir is not None and d != want_dir):
            continue
        idx = _first_axis_crossing(P, start=1)
        su = None
        tries = 0
        for i in idx:
            t = P[i, 1] / (P[i, 1] - P[i + 1, 1])
            Rc = P[i, 0] + t * (P[i + 1, 0] - P[i, 0])
            if su is None:
                su = _seg_errors(orb, c, br, delta0)
            L = 2.0 * sec.scale
            line = np.array([[Rc - L, 0.0], [Rc + L, 0.0]])
            cc = BR.certify_crossing(P, su, line, np.zeros(1), int(i), 0)
            L_, j_ = br["refs"][i + 1]
            rec = dict(owner=int(k), partner=int(nb), dir=int(d), seg=int(i), h=np.array(cc.get("h", [Rc, 0.0])),
                       box_diam=float(cc.get("box_diam", np.inf)), level=int(L_ if L_ is not None else 0),
                       cert={kk: v for kk, v in cc.items() if kk != "h"}, ok=bool(cc["ok"]))
            if cc["ok"]:
                if best is None or rec["level"] < best["level"]:
                    best = rec
                break                                       # the first certified crossing along the branch
            tries += 1
            if tries >= 4:
                break
    return best


def _heteroclinic_images(sec, h, box_diam, q, hfd):
    """h_j = P^j(h) for j = 0..q-1 with 2D error bounds: e_j = |DP^j| box/2 + sum_i |DP^(j-i)| tau_i
    (first order, finite-difference Jacobians of the accurate map)."""
    E = np.array([[0.0, 0.0], [hfd, 0.0], [-hfd, 0.0], [0.0, hfd], [0.0, -hfd]])
    y = h[None] + E
    pts = [h.copy()]
    Js = [np.eye(2)]
    taus = [0.0]
    lost = False
    for j in range(1, q):
        y, l, t = sec.step(y, np.ones(len(y)))
        lost |= bool(np.any(l))
        pts.append(y[0].copy())
        Js.append(np.stack([(y[1] - y[2]) / (2 * hfd), (y[3] - y[4]) / (2 * hfd)], -1))
        taus.append(float(t[0]))
    errs = []
    for j in range(q):
        e = float(np.linalg.norm(Js[j], 2)) * 0.5 * box_diam
        e_tr = 0.0
        for i in range(1, j + 1):
            Jji = Js[j] @ np.linalg.inv(Js[i])
            e_tr += float(np.linalg.norm(Jji, 2)) * taus[i]
        errs.append(e + 2.0 * e_tr)
    return np.array(pts), np.array(errs), np.array(taus), lost, [float(np.linalg.norm(J, 2)) for J in Js]


def _assemble(sec, orb, a, cls, cu, cs, rec, hj, ej, sigma, succ, pred, delta0, verbose=False):
    """Closed curve of class cls from the unstable curve cu, its mirror cs, the heteroclinic orbit hj
    (with errors ej) starting in the gap of rec.  Returns dict with polygon, per-segment errors, etc."""
    q = orb.q
    cdir = +1 if cls == "A" else -1
    kA, sA = rec["owner"], rec["partner"]
    pieces = []
    junction = []
    gaps = []
    for j in range(q):
        uo = (kA + j) % q
        so = (sA + j) % q
        if (succ if cdir > 0 else pred)[uo] != so:
            return dict(ok=False, reason=f"class {cls}: image gap {j} ({uo} -> {so}) not between chain neighbours")
        bu = BR.branch_polyline(orb, cu, uo)
        bs = BR.branch_polyline(orb, cs, so)
        eu = _seg_errors(orb, cu, bu, delta0)
        es = _seg_errors(orb, cs, bs, delta0)
        # search only up to the levels where the images lie (the u side: level L_h + j; the s side: the
        # mirror of level L_h - j), so that a later passage of a branch near h_j cannot be taken
        nu_ = _upto_level(bu["refs"], rec["level"] + j + 1)
        ns_ = _upto_level(bs["refs"], max(rec["level"] - j + 1, 1))
        iu, tu, du = _nearest_on_polyline(bu["pts"][:nu_], hj[j])
        is_, ts, ds = _nearest_on_polyline(bs["pts"][:ns_], hj[j])
        # consistency: the true branches pass through the true h_j, so the computed ones pass within
        # their error + e_j of the computed h_j
        lim_u = eu[iu] + ej[j] + 1e-12 * sec.scale
        lim_s = es[is_] + ej[j] + 1e-12 * sec.scale
        if du > 4.0 * lim_u or ds > 4.0 * lim_s:
            return dict(ok=False, reason=f"class {cls}: heteroclinic image {j} off the computed branches "
                                         f"(d_u {du:.2e} vs {lim_u:.2e}, d_s {ds:.2e} vs {lim_s:.2e})")
        Ju = ej[j] + du + eu[iu]
        Js = ej[j] + ds + es[is_]
        # u arc: X_uo, vertices 1..iu, then h_j;  s arc: h_j, vertices is_..1 of the s branch, (X_so next)
        upts = bu["pts"][:iu + 1]
        uerr = np.concatenate([eu[:iu], [max(eu[iu], Ju)]])          # segment errors along upts + h_j
        spts = bs["pts"][1:is_ + 1][::-1]
        serr = np.concatenate([[max(es[is_], Js)], es[:is_][::-1]]) if is_ >= 0 else np.zeros(0)
        # serr[k]: segment from (h_j, spts...) k -> k+1; the last one ends at X_so
        if cdir > 0:
            gpts = np.concatenate([upts, hj[j][None], spts])
            gerr = np.concatenate([uerr, serr])
            gaps.append(dict(start=uo, end=so, pts=gpts, err=gerr))
        else:
            # traversed counter-clockwise: X_so -> s branch -> h_j -> u branch reversed -> X_uo
            s_fwd = bs["pts"][:is_ + 1]
            s_err = np.concatenate([es[:is_], [max(es[is_], Js)]])
            u_rev = bu["pts"][1:iu + 1][::-1]
            u_err = np.concatenate([[max(eu[iu], Ju)], eu[:iu][::-1]])
            gpts = np.concatenate([s_fwd, hj[j][None], u_rev])
            gerr = np.concatenate([s_err, u_err])
            gaps.append(dict(start=so, end=uo, pts=gpts, err=gerr))
        junction.append(dict(gap=(int(uo), int(so)), e_image=float(ej[j]), d_u=du, d_s=ds, J_u=Ju, J_s=Js,
                             u_vertex=int(iu), s_vertex=int(is_), u_level=bu["refs"][min(iu + 1, len(bu["refs"]) - 1)][0],
                             s_level=bs["refs"][min(is_ + 1, len(bs["refs"]) - 1)][0]))
        pieces.append((bu, bs, iu, is_))
    # chain the gaps counter-clockwise starting from the gap that starts at order[0]
    by_start = {g["start"]: g for g in gaps}
    if len(by_start) != q:
        return dict(ok=False, reason=f"class {cls}: gaps do not start at distinct X points")
    C, E = [], []
    k0 = gaps[0]["start"]
    k = k0
    for _ in range(q):
        g = by_start[k]
        C.append(g["pts"])
        E.append(g["err"])
        k = g["end"]
    if k != k0:
        return dict(ok=False, reason=f"class {cls}: gaps do not close up")
    C = np.concatenate(C)
    E = np.concatenate(E)
    assert len(C) == len(E), (len(C), len(E))
    return dict(ok=True, C=C, E=E, junctions=junction, gaps=gaps)


def decimate_closed(C, E, rel=0.25, abs_tol=None, corners=None):
    """Drop vertices of the closed polygon C (segment k: C[k] -> C[k+1 mod n], error E[k]) while every
    dropped vertex lies within rel x (the smallest error of the merged segments) of the new chord.  The
    merged segment's error is max(merged errors) + max(vertex distance to the chord): the old polyline
    lies within that distance of the chord (the stadium around a segment is convex), so the true curve
    stays within the new error along matching parameterisations.  Vertices within abs_tol (default a
    quarter of the 90th percentile of the errors) are always dropped: the errors near the X points, far below the
    curve's typical error, may grow to that size -- except near `corners` (the X points, where an incoming
    and an outgoing arc meet at a small angle): there the tolerance is at most a tenth of the distance to
    the nearest corner, so that the decimation cannot make the polygon cross itself."""
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
    if _njit is not None:
        keep, newE = _decimate_nb(np.ascontiguousarray(P), np.ascontiguousarray(E), np.ascontiguousarray(capX),
                                  float(rel), float(abs_tol))
        return C[keep], newE
    keep = [0]
    newE = []
    i = 0
    while i < n:
        def ok(j):
            if j - i < 2:
                return True, 0.0
            a, b = P[i], P[j]
            d = BR._pt_seg_dist(P[i + 1:j], a[None], b[None])
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


def closed_polygon_simple_fast(C, tol):
    """torux.barrier.closed_polygon_simple (no two non-adjacent edges closer than tol, no adjacent edges
    folding back), with a uniform-grid broad phase (cells of the longest edge's size) instead of all
    bounding-box pairs: the same predicate on the same candidate set, O(n) pairs for a curve."""
    C = np.asarray(C, float)
    n = len(C)
    Pc = np.concatenate([C, C[:1]])
    lo = np.minimum(Pc[:-1], Pc[1:]) - tol
    hi = np.maximum(Pc[:-1], Pc[1:]) + tol
    h = float(np.max(hi - lo)) + 1e-300
    org = lo.min(0)
    i0 = np.floor((lo - org) / h).astype(np.int64)
    i1 = np.floor((hi - org) / h).astype(np.int64)
    ny = int(i1[:, 1].max()) + 2
    cells, edges = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            cx, cy = i0[:, 0] + dx, i0[:, 1] + dy
            sel = (cx <= i1[:, 0]) & (cy <= i1[:, 1])
            cells.append(cx[sel] * ny + cy[sel])
            edges.append(np.flatnonzero(sel))
    cells = np.concatenate(cells)
    edges = np.concatenate(edges)
    o = np.argsort(cells, kind="stable")
    cells, edges = cells[o], edges[o]
    starts = np.flatnonzero(np.concatenate([[True], cells[1:] != cells[:-1]]))
    ends = np.concatenate([starts[1:], [len(cells)]])
    I, J = [], []
    for a_, b_ in zip(starts, ends):
        m = b_ - a_
        if m < 2:
            continue
        e = edges[a_:b_]
        ii, jj = np.triu_indices(m, 1)
        I.append(e[ii])
        J.append(e[jj])
    if not I:
        return True
    I = np.concatenate(I)
    J = np.concatenate(J)
    I, J = np.minimum(I, J), np.maximum(I, J)
    key = np.unique(I * n + J)
    I, J = key // n, key % n
    ov = np.all(lo[I] <= hi[J], 1) & np.all(lo[J] <= hi[I], 1) & (I < J)
    I, J = I[ov], J[ov]
    adj = (J - I == 1) | ((I == 0) & (J == n - 1))
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
    a = a1 - a0
    b = b1 - b0
    w = b0 - a0
    den = BR._cross(a, b)
    d = np.where(den != 0, den, 1.0)
    t = BR._cross(w, b) / d
    u = BR._cross(w, a) / d
    if np.any((den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
        return False
    dist = np.minimum(np.minimum(BR._pt_seg_dist(a0, b0, b1), BR._pt_seg_dist(a1, b0, b1)),
                      np.minimum(BR._pt_seg_dist(b0, a0, a1), BR._pt_seg_dist(b1, a0, a1)))
    return bool(np.all(dist > tol))


def _certify_curve(sec, C, E, a, a_err, corners=None):
    """Simplicity (loop cuts certified), winding 1 around a beyond the tube, flux, density, tube."""
    out = dict(ok=False)
    tol = 1e-12 * sec.scale
    n0 = len(C)
    E = np.asarray(E, float)
    km = int(np.argmax(E))
    out["E_before_decimation"] = dict(max=float(E[km]), p50=float(np.median(E)), p90=float(np.quantile(E, 0.9)),
                                      argmax_frac=float(km / max(len(E), 1)), argmax_point=np.asarray(C[km]).tolist())
    C, E = decimate_closed(C, E, corners=corners)
    C, E = BR._dedupe_closed(C, E, tol)
    out["vertices_before_decimation"] = int(n0)
    w = BR.winding_number(C, a)
    out["winding"] = w
    if w == -1:
        C = C[::-1]
        E = np.concatenate([E[:-1][::-1], E[-1:]])
        w = 1
    if w != 1:
        out["reason"] = f"winding number around the axis point {w}"
        return out, None
    simple = closed_polygon_simple_fast(C, tol=tol)
    out["simple_as_assembled"] = bool(simple)
    if not simple:
        C2, E2, info = BR.simplify_closed_curve(C, E, a, tol)
        out["loop_cuts"] = {k: v for k, v in info.items() if k != "boxes"}
        if C2 is None:
            out["reason"] = "closed curve not simple: " + info.get("reason", "")
            return out, None
        if BR.winding_number(C2, a) == -1:
            C2 = C2[::-1]
            E2 = np.concatenate([E2[:-1][::-1], E2[-1:]])
        if not closed_polygon_simple_fast(C2, tol=tol) or BR.winding_number(C2, a) != 1:
            out["reason"] = "closed curve not simple after loop cuts"
            return out, None
        C, E = C2, E2
    eps_max = float(np.max(E))
    out["eps_max"] = eps_max
    out["eps_median"] = float(np.median(E))
    dmin_a = float(BR.polyline_distance(np.atleast_2d(a), np.concatenate([C, C[:1]]))[0])
    out["dist_axis"] = dmin_a
    if dmin_a <= eps_max + a_err:
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


def _start_length(e_X, lam, scale, reach=None):
    """The start segment's length s0.  Its curvature error 0.5 kappa_a (lam s0)^2 (kappa_a = 1e3 / scale,
    checked against the measured curvature) is kept at the size of e_X, which every vertex near X carries
    anyway: s0 = max(1e-6 scale / lam, (2 e_X / kappa_a)^(1/2) / lam), at most 1e-3 scale.  (A longer start
    for weak chains -- its error is contracted along the branch -- saves levels, which are cheap for weak
    chains, but widens the error tube next to every X point as s0^3: not used; reach is unused.)"""
    kappa_a = 1e3 / scale
    e = max(e_X, 1e-12 * scale)
    s_e = math.sqrt(2.0 * e / kappa_a) / lam
    return float(min(max(1e-6 * scale / lam, s_e), 1e-3 * scale))


def predicted_levels(q, lam, s0, reach):
    """Levels (period maps) the unstable branches need to reach `reach` from the start segment s0:
    q ln(reach / s0) / ln(lam) (each level of P^q stretches by lam near X; measured on the certified
    runs: 0.55 - 0.95 of q ln(1e6) / ln(lam))."""
    return float(q * math.log(max(reach / s0, math.e)) / math.log(max(lam, 1.0 + 1e-15)))


def _growth_too_slow(t_start, levels_done, levels_needed, deadline, factor=1.5):
    """Refusal text when the levels still needed, at the measured time per level, end beyond the deadline
    by more than `factor` (cheap failure instead of a deadline after the whole budget), else None."""
    if deadline is None or levels_done < 1:
        return None
    per = (time.time() - t_start) / levels_done
    rest = max(levels_needed - levels_done, 0.0) * per
    if time.time() + rest / factor > deadline:
        return (f"growth would need ~{levels_needed:.0f} levels ({per * 1e3:.1f} ms per level: ~{rest:.0f} s more), "
                f"beyond the deadline")
    return None


def _grow_and_close(sec, orb, a, eig, lam, bS, succ, pred, mism, scale, deadline=None, verbose=False, s0=None,
                    n0=None, dmax=None, amax=0.2, target=None, max_levels=None, max_pts=6000, t0=None, axis_e=0.0):
    t0 = t0 or time.time()
    q = orb.q
    out = dict(ok=False, timings={})
    # half the distance between a mirror pair of neighbours (Z = 0 lies between them): the branches' reach
    reach = 0.5 * float(np.min([np.linalg.norm(orb.pts[k] - orb.pts[(bS - k) % q]) for k in range(q)
                                if (bS - k) % q != k] or [scale]))
    if s0 is None:
        s0 = _start_length(orb.e_X, lam, scale, reach)
    dmax = dmax if dmax is not None else 1e-2 * scale
    target = target if target is not None else 2e-6 * scale
    sigma = [(bS - k) % q for k in range(q)]
    # the field's symmetry defect (reference evaluator) -> per-level error of the mirrored curves
    dfx = 0.0
    if hasattr(sec, "symmetry_defect") and getattr(sec, "use_reference", False):
        x3 = np.stack([orb.pts[:, 0] * np.cos(0.4), orb.pts[:, 0] * np.sin(0.4), orb.pts[:, 1]], -1)
        dfx = sec.symmetry_defect(x3)
    Lper = float(np.max(np.abs(orb.pts[:, 0]))) * 2 * np.pi / getattr(sec, "nfp", 1)
    # for the exactly symmetric field S(W^u(X)) = W^s(S X): the mirrored polylines carry the u curves' errors;
    # the computed orbit's mirror mismatch and the evaluated field's measured defect are added once (the
    # evaluated field's asymmetry is part of the tracer's error model per step)
    defect_step = 10.0 * dfx * Lper + 1e-15 * scale + mism
    out["symmetry"] = dict(base=int(bS), defect=float(dfx), mismatch=float(mism), step=float(defect_step))
    specs = [("u+", "u", eig["vu"], +1), ("u-", "u", eig["vu"], -1)]
    if n0 is None:
        # weak X lines: thousands of fundamental-domain images, each short and straight next to X
        n0 = 16 if lam > 1.1 else (6 if lam > 1.01 else 3)
    curves = [BR.GenCurve(name, kind, orb.pts[0], v, sgn, lam, False, s0, n0, q) for (name, kind, v, sgn) in specs]
    kappa_a = 1e3 / scale
    # start segment: linearisation (curvature) + eigenvector direction error (monodromy accuracy: the
    # spectral monodromy's two-resolution difference, or the finite-difference noise 2 tau_q / h)
    dM = orb.info.get("dM")
    if dM is None:
        dM = 2.0 * orb.info.get("tau_q", 0.0) / (1e-7 * scale) + 1e-14
    th_err = eigen_error(orb.M, dM)
    out["eigvec_angle_err"] = th_err
    if not th_err < 0.05:
        out["reason"] = f"unstable eigenvector not determined (angle error bound {th_err:.2e})"
        return out
    delta0 = 0.5 * kappa_a * (lam * s0) ** 2 + lam * s0 * th_err
    if max_levels is None:
        max_levels = int(min(400000, 3 * q * math.ceil(math.log(scale / s0) / math.log(lam)) + 6 * q))
    # the levels needed to reach the symmetric heteroclinic points
    lev_pred = predicted_levels(q, lam, s0, reach)
    out["growth_plan"] = dict(s0=s0, n0=n0, dmax=dmax, max_levels=max_levels, lam=lam, predicted_levels=lev_pred)
    cls_curve = {}
    found = {}
    lev = 0
    status = None
    t_grow = time.time()
    while True:
        if deadline is not None and time.time() > deadline:
            status = "deadline during manifold growth"
            break
        for _ in range(q):
            BR.grow_step(sec, curves, dmax, amax, max_pts=max_pts)
        lev = min(c.level() for c in curves)
        if not found and lev >= 4 * q and lev % (4 * q) < q:
            slow = _growth_too_slow(t_grow, lev, lev_pred, deadline)
            if slow:
                status = slow
                break
        if lev >= q - 1:
            # each unstable generating curve is classified by the sense along the chain of its first
            # certified symmetric heteroclinic connection: A towards the successor, B the predecessor
            for c in curves:
                if any(cc is c for cc in cls_curve.values()):
                    continue
                if not _new_z_crossing(c):
                    continue                     # no sign change of Z on the pieces grown since the last look
                taken = [r["dir"] for r in found.values()]
                rec = _find_symmetric_crossing(sec, orb, a, c, sigma, succ, pred, delta0,
                                               want_dir=None if not taken else -taken[0])
                if rec is not None:
                    cls = "A" if rec["dir"] > 0 else "B"
                    found[cls] = rec
                    cls_curve[cls] = c
                    if verbose:
                        print(f"   class {cls} ({c.name}): symmetric heteroclinic point at level {rec['level']} owner "
                              f"{rec['owner']} -> {rec['partner']}: R {rec['h'][0]:.10f} box {rec['box_diam']:.2e} "
                              f"({time.time() - t0:.0f}s)", flush=True)
        # a class's curve stops growing once it reaches 2q levels beyond its heteroclinic point (the images
        # need q - 1 more): its folds would only add points
        for cls, rec in found.items():
            c = cls_curve[cls]
            if c.active and c.level() >= rec["level"] + 2 * q:
                c.active = False
                c.stop_reason = "done"
        if len(found) == 2 and all(cls_curve[k].level() >= found[k]["level"] + 2 * q for k in found):
            break
        lev = max(c.level() for c in curves)
        if verbose and (lev // q) % max(1, 200 // q) == 0:
            print(f"   level {lev}: points {[len(c.pieces[-1].pts) for c in curves]} found {sorted(found)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if lev >= max_levels:
            status = f"no symmetric heteroclinic point for classes {sorted(set('AB') - set(found))} within {lev} levels"
            break
        stopped = [c for c in curves if not c.active and c.stop_reason != "done"]
        if stopped:
            status = "growth stopped: " + ", ".join(f"{c.name}: {c.stop_reason}" for c in stopped)
            break
    out["timings"]["grow"] = time.time() - t0
    out["growth"] = {c.name: dict(levels=c.level(), points=int(sum(len(p.pts) for p in c.pieces)), stop=c.stop_reason,
                                  cls=next((k for k, v in cls_curve.items() if v is c), None)) for c in curves}
    if status is not None:
        out["reason"] = status
        return out
    # measured curvature near X against the a-priori bound used for the start segment: from the deviation
    # of the computed branch at X_0 from its start line X_0 + t v (dev ~ kappa t^2 / 2), over t in
    # [lam^2 s0, 1e-2 scale] (turning angles of the tiny segments next to X are rounding noise)
    kap = 0.0
    for c in curves:
        for pc in c.pieces[1:]:
            if pc.owner != 0:
                continue
            d_ = pc.pts - orb.pts[0]
            t_ = np.abs(d_ @ c.v)
            dev = np.abs(d_[:, 0] * c.v[1] - d_[:, 1] * c.v[0])
            sel = (t_ >= lam ** 2 * s0) & (t_ <= 1e-2 * scale)
            if np.any(sel):
                k_ = 2.0 * np.maximum(dev[sel] - pc.err[sel] - th_err * t_[sel], 0.0) / t_[sel] ** 2
                kap = max(kap, float(np.max(k_)))
    if kap > kappa_a:
        delta0 = 0.5 * 2.0 * kap * (lam * s0) ** 2 + lam * s0 * th_err
    out["start_error"] = dict(s0=s0, kappa_measured=kap, kappa_bound=kappa_a, delta0=delta0)
    hfd = 1e-7 * scale
    # --- chord validation of the arcs used, then the final crossings and heteroclinic orbits
    curves_out = {}
    polys = {}
    het_ref = dict(ok=True, samples=[])
    for cls in ("A", "B"):
        c = cls_curve[cls]
        rec = found[cls]
        hj, ej, taus, lost, Jn = _heteroclinic_images(sec, rec["h"], rec["box_diam"], q, hfd)
        if lost:
            out["reason"] = f"class {cls}: heteroclinic orbit lost"
            return out
        # segments to validate: every owner's branch up to its image point (and the preimages' points)
        segs = []
        for j in range(q):
            uo = (rec["owner"] + j) % q
            br = BR.branch_polyline(orb, c, uo)
            iu, _, _ = _nearest_on_polyline(br["pts"], hj[j])
            segs += BR.branch_segments(br, 1, min(iu + 4, len(br["refs"]) - 1))
            # the mirrored arcs: S(h_j) on the branch at owner sigma(partner + j)
            so = (rec["partner"] + j) % q
            br2 = BR.branch_polyline(orb, c, sigma[so])
            is_, _, _ = _nearest_on_polyline(br2["pts"], hj[j] * np.array([1.0, -1.0]))
            segs += BR.branch_segments(br2, 1, min(is_ + 4, len(br2["refs"]) - 1))
        segs = sorted(set(segs))
        st = BR.refine_arc_segments(sec, c, segs, target)
        out.setdefault("chord_validation", {})[cls] = st
        if verbose:
            print(f"   class {cls}: chord validation {st} ({time.time() - t0:.0f}s)", flush=True)
        if not st["converged"]:
            out["reason"] = f"class {cls}: chord validation did not converge"
            return out
        rec2 = _find_symmetric_crossing(sec, orb, a, c, sigma, succ, pred, delta0, first_owner=rec["owner"],
                                        want_dir=rec["dir"])
        if rec2 is None:
            out["reason"] = f"class {cls}: symmetric crossing lost after chord validation"
            return out
        rec = rec2
        found[cls] = rec
        hj, ej, taus, lost, Jn = _heteroclinic_images(sec, rec["h"], rec["box_diam"], q, hfd)
        if lost:
            out["reason"] = f"class {cls}: heteroclinic orbit lost"
            return out
        cs = BR.mirror_curve(c, "S" + c.name, bS, 0.0)
        cs.const_err = defect_step
        asm = _assemble(sec, orb, a, cls, c, cs, rec, hj, ej, sigma, succ, pred, delta0, verbose=verbose)
        if not asm["ok"]:
            out["reason"] = asm["reason"]
            return out
        cert, Cfin = _certify_curve(sec, asm["C"], asm["E"], a, axis_e, corners=orb.pts)
        cert.update(crossing=dict(owner=rec["owner"], partner=rec["partner"], level=rec["level"], h=rec["h"].tolist(),
                                  box_diam=rec["box_diam"], angle=rec["cert"].get("angle"),
                                  eps_u=rec["cert"].get("eps_u")),
                    image_errors=ej.tolist(), image_jacobian_norms=Jn, curve=c.name,
                    junctions=asm["junctions"],
                    junction_err_max=float(max(max(jn["J_u"], jn["J_s"]) for jn in asm["junctions"])))
        curves_out[cls] = cert
        if verbose:
            print(f"   class {cls}: ok {cert['ok']} {cert.get('reason', '')} flux {cert.get('flux')} tube "
                  f"{cert.get('tube')} eps_max {cert.get('eps_max')} ({time.time() - t0:.0f}s)", flush=True)
        if not cert["ok"]:
            out["curves"] = curves_out
            out["reason"] = f"class {cls}: " + cert.get("reason", "")
            return out
        polys[cls] = Cfin
        curves_out[cls]["_gaps"] = asm["gaps"]
        # reference re-trace of the heteroclinic orbit
        if hasattr(sec, "reference_map") and getattr(sec, "use_reference", False):
            Y, Er = sec.reference_map(rec["h"], q - 1)
            d = np.linalg.norm(Y[:, 0, :] - hj[1:], axis=1)
            tr_err = np.array([ej[j] - Jn[j] * 0.5 * rec["box_diam"] for j in range(1, q)])
            okh = bool(np.all(d <= np.maximum(tr_err, 0) + 10 * Er[:, 0] + 1e-12 * scale))
            het_ref["samples"].append(dict(cls=cls, max_diff=float(np.max(d)), max_model=float(np.max(tr_err)),
                                           max_ref_err=float(np.max(Er[:, 0])), ok=okh))
            het_ref["ok"] = bool(het_ref["ok"] and okh)
    for v in curves_out.values():
        v.pop("_gaps", None)
    out["curves"] = curves_out
    out["heteroclinic_reference"] = het_ref
    out["evidence"] = BR.reference_evidence(sec, orb, curves + [BR.mirror_curve(c, "S" + c.name, bS, 0.0)
                                                                 for c in curves], {})
    out["polygons"] = {k: v.tolist() for k, v in polys.items()}
    out["timings"]["curves"] = time.time() - t0
    out["ok"] = True
    return out



# ============================================================================ validated surrogate of P
def _cheb_nodes(n):
    return np.cos(np.pi * np.arange(n) / (n - 1))           # Chebyshev-Lobatto on [-1, 1] (descending)


def _cheb_coef2(V):
    """2D Chebyshev coefficients from values on the Lobatto tensor grid (..., n, n)."""
    from scipy.fft import dct
    n = V.shape[-1]
    C = dct(dct(V, type=1, axis=-1), type=1, axis=-2) / (n - 1) ** 2
    C[..., 0, :] *= 0.5
    C[..., -1, :] *= 0.5
    C[..., :, 0] *= 0.5
    C[..., :, -1] *= 0.5
    return C


def _cheb_vander(u, n):
    T = np.empty((len(u), n))
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = u
    for k in range(2, n):
        T[:, k] = 2 * u * T[:, k - 1] - T[:, k - 2]
    return T


def _cheb_vander_d(u, n):
    """T_k'(u) (k < n) via T_k' = k U_(k-1)."""
    U = np.empty((len(u), max(n, 1)))
    U[:, 0] = 1.0
    if n > 1:
        U[:, 1] = 2 * u
    for k in range(2, n):
        U[:, k] = 2 * u * U[:, k - 1] - U[:, k - 2]
    D = np.zeros((len(u), n))
    for k in range(1, n):
        D[:, k] = k * U[:, k - 1]
    return D


def section_monodromy(sec, x, q, h):
    """Monodromy of the section's P^q at x and the orbit: the product of the surrogate's analytic Jacobians
    along the orbit when it has them (piecewise interpolants: finite differences would see the patches'
    jumps, ~tau / h), else central differences with step h.  Returns (M, image, points, tau, lost)."""
    if not hasattr(sec, "jacobian"):
        return fd_monodromy(sec, x, q, h)
    y = np.asarray(x, float)[None]
    pts = [y[0].copy()]
    M = np.eye(2)
    tau = 0.0
    lost = False
    for j in range(q):
        J = sec.jacobian(y)
        y, l, t = sec.step(y, np.ones(1))
        tau += float(t[0])
        lost |= bool(l[0]) or not np.all(np.isfinite(J))
        M = J[0] @ M
        if j < q - 1:
            pts.append(y[0].copy())
    return M, y[0], np.array(pts), tau, lost


class SurrogateDeadline(RuntimeError):
    """A surrogate build refused: at its measured tracing rate it would end beyond the deadline."""


# the smallest chunk of a surrogate build with a deadline (_trace_batched)
PROBE_NODES = 256
# the surrogates' retrace bar: nodes whose error estimate exceeds RETRACE_REL x max(1, |R|) are traced again at 48
# (then 96) steps per period
RETRACE_REL = 5e-12

def _p3(key):
    """A third-pass switch of torux.analysis.POLICY, read at call time (default off: the full path's behaviour)."""
    try:
        from .analysis import POLICY
        return bool(POLICY.get(key))
    except Exception:  # noqa: BLE001
        return False


# card node traces: the GPUPeriodMap per (engine, sign), and the CPU cross-check
# sample per batch
GPU_CHECK_NODES = 48
_GPU_MAPS = {}


FIRST_SP_SAMPLE = 64
FIRST_SP_SHARE = 0.3


def _first_steps(sur, X, gpu, deadline=None):
    """The steps per period the node traces start at: 24, or 48 (96) when a uniform sample of
    FIRST_SP_SAMPLE nodes shows more than FIRST_SP_SHARE of them above the retrace bar at 24 (48) steps: tracing
    every node at 24 and again at 48 those above the bar can cost 2.2x the work of tracing all at 48 (QUASR 0017820:
    13205 of 21312 nodes).  Each node's value and error estimate still come from its own trace.
    With a deadline, a build that the sample's rate (halved: a sample of 64 pays more per node than the build's
    chunks) projects beyond it by more than 1.25x is refused at once (SurrogateDeadline), before a sixteenth of
    the nodes is traced (50-300 s for the 300k-600k-node annuli of low-order chains)."""
    idx = np.linspace(0, len(X) - 1, FIRST_SP_SAMPLE).astype(int)
    chosen, rate, big = 96, None, 0.0
    on_card = False
    for sp in (24, 48):
        sec = PlaneSection(sur.engine, sur.source, phi_s=sur.phi_s, sign=sur.sign, scale=sur.scale, reference=False,
                           steps_per_period=sp)
        g = _gpu_step(sec, sp) if gpu else None
        on_card = on_card or g is not None
        st = g or sec.step
        t0 = time.time()
        y, l, t = st(X[idx], np.ones(len(idx)))
        # (on a card the sample's time is the kernels' compilation and launch, not the build's rate: no early
        # refusal there -- the chunked build's own projection after its first chunk decides)
        rate = None if on_card else (time.time() - t0) / len(idx)
        big = float(np.mean(~l & (t > RETRACE_REL * np.maximum(1.0, np.abs(y[:, 0])))))
        if big <= FIRST_SP_SHARE:
            chosen = sp
            break
    if chosen == 96 and rate is not None:
        rate = 2.0 * rate
    if deadline is not None and rate is not None:
        rest = 0.5 * rate * len(X) * (1.0 + 2.0 * big)
        left = deadline - time.time()
        if rest > 1.25 * left:
            raise SurrogateDeadline(f"surrogate nodes: {len(X)} to trace at {chosen} steps, about {rest:.0f} s at the "
                                    f"sample's rate, {left:.0f} s to the deadline")
    return chosen


def _gpu_step(sec, sp):
    """(POLICY 'card_traces') step(X, ones) -> (Y, lost, tau) for the section's plane on the GPU (torux.cardtrace: the
    period map of
    PlaneSection(steps_per_period=sp).step to rounding, tau = 2 x the GBS estimate + 2e-13 max(1, |R|)), or None
    without a usable card.  Every call re-traces GPU_CHECK_NODES of its points on the CPU tracer and raises when
    |Y_gpu - Y_cpu| exceeds tau_gpu + tau_cpu at one of them (both are the same scheme: they agree to rounding)."""
    if not _p3("card_traces"):
        return None
    try:
        from . import cardtrace
        if not cardtrace.gpu_available():
            return None
        key = (id(sec.engine), float(sec.sign), int(sec.nfp))
        if key not in _GPU_MAPS:
            _GPU_MAPS[key] = cardtrace.GPUPeriodMap(cardtrace.FilamentField(sec.engine), sec.nfp, sec.sign)
        gm = _GPU_MAPS[key]
    except Exception as e:  # noqa: BLE001
        if environment_failure(e):
            raise
        return None
    cpu = PlaneSection(sec.engine, sec.source, phi_s=sec.phi_s, sign=sec.sign, scale=sec.scale, reference=False,
                       steps_per_period=sp)
    stats = getattr(sec, "gpu_stats", None)
    if stats is None:
        stats = sec.gpu_stats = dict(nodes=0, checked=0, max_diff=0.0, max_diff_over_tau=0.0, wall=0.0)

    def step(X, d):
        t0 = time.time()
        X = np.asarray(X, float).reshape(-1, 2)
        y, lost, err = gm.period_map(X, sp, phi0=sec.phi_s)
        tau = 2.0 * err + 2e-13 * np.maximum(1.0, np.abs(y[:, 0]))
        rng = np.random.default_rng(len(X))
        idx = rng.choice(len(X), size=min(GPU_CHECK_NODES, len(X)), replace=False)
        yc, lc, tc = cpu.step(X[idx], np.ones(len(idx)))
        both = ~(lc | lost[idx])
        if np.any(lc != lost[idx]):
            raise RuntimeError(f"GPU node traces: {int(np.sum(lc != lost[idx]))} of {len(idx)} checked nodes lost on "
                               f"one tracer only")
        if np.any(both):
            dd = np.linalg.norm(y[idx][both] - yc[both], axis=1)
            ratio = dd / (tau[idx][both] + tc[both])
            stats["max_diff"] = max(stats["max_diff"], float(np.max(dd)))
            stats["max_diff_over_tau"] = max(stats["max_diff_over_tau"], float(np.max(ratio)))
            if np.any(ratio > 1.0):
                raise RuntimeError(f"GPU node traces differ from the CPU tracer by {np.max(dd):.2e} (> both error "
                                   f"estimates) at {int(np.sum(ratio > 1))} of {int(both.sum())} checked nodes")
        stats["nodes"] += len(X)
        stats["checked"] += len(idx)
        stats["wall"] += time.time() - t0
        return y, lost, tau
    step.on_card = True
    return step


def _trace_batched(step, X, chunk, deadline=None, factor=1.25, what="surrogate nodes", retrace=False):
    """(Y, lost, tau) of step(X[sl], ones) over X in chunks.  With a deadline the chunks are short and after each
    the rest is projected at the measured rate and the build refused (SurrogateDeadline) when it would end beyond
    the deadline by more than `factor` x the projection's margin, or once the deadline has passed -- a cheap
    refusal instead of a build that runs past the budget (the 47-period band of QUASR 2439098: 24192 nodes traced
    in 1740 s on a loaded host, then a second plane's 32256 started after its 1500 s deadline).

    With a deadline the chunks are a sixteenth of X (at least 256 points) and
    interleaved: chunk c holds the nodes c, c + S, c + 2 S, ... (S chunks), so the first is a uniform sample of
    the whole annulus -- its rate and its share of nodes above the retrace bar are the build's, where a block of
    the first patches was not (a small first probe projects 1.1-9 x the actual build times; equal chunk sizes
    also keep one array shape).  retrace (the builds' first round at 24 steps): the
    projection includes the retracing at 48 steps (twice the cost) of the nodes whose error estimate exceeds the
    builds' retrace bar (RETRACE_REL x max(1, |R|)), from the share measured so far.  The values
    do not depend on the order (each node is traced on its own)."""
    n = len(X)
    Y = np.empty_like(X)
    lost = np.zeros(n, bool)
    tau = np.zeros(n)
    if getattr(step, "on_card", False):
        # (card traces) a card's first chunks carry the kernels' compilation: a lenient projection there
        factor = max(factor, 4.0)
    if deadline is None:
        for k in range(0, n, int(chunk)):
            sl = slice(k, min(n, k + int(chunk)))
            Y[sl], lost[sl], tau[sl] = step(X[sl], np.ones(sl.stop - sl.start))
        return Y, lost, tau
    chunk = min(int(chunk), max(256, -(-n // 16)))
    S = max(1, -(-n // chunk))
    order = np.concatenate([np.arange(c, n, S) for c in range(S)])
    k, t0 = 0, time.time()
    for c in range(S):
        if time.time() > deadline:
            raise SurrogateDeadline(f"deadline while tracing {what} ({k} of {n} done)")
        idx = np.arange(c, n, S)
        Y[idx], lost[idx], tau[idx] = step(X[idx], np.ones(len(idx)))
        k += len(idx)
        if k < n:
            done = order[:k]
            per = (time.time() - t0) / k
            big = 0.0
            if retrace:
                big = float(np.mean(~lost[done] & (tau[done] > RETRACE_REL * np.maximum(1.0, np.abs(Y[done, 0])))))
            rest = per * ((n - k) + 2.0 * big * n)
            extra = f" (with {100 * big:.0f} % of the nodes retraced at 48 steps)" if big > 0 else ""
            if time.time() + rest / factor > deadline:
                raise SurrogateDeadline(f"{what}: {n - k} of {n} still to trace, about {rest:.0f} s at the measured "
                                        f"rate{extra}, {deadline - time.time():.0f} s to the deadline")
    return Y, lost, tau


class PatchSurrogate(PlaneSection):
    """PlaneSection whose period map is a validated piecewise-Chebyshev surrogate on the annulus between
    the sections of two tori T1 (inside) and T2 (outside).

    The plane is covered by a uniform grid of square cells (side c); the cells meeting the closed annulus
    (dilated by `margin`) carry n x n tensor Chebyshev interpolants of the period map's image (R', Z') and
    of B_phi, from one batch of accurate traces at the Lobatto nodes (the map is analytic: the
    interpolation error falls geometrically with n at the cell size).  The backward map is S F S
    (S(R, Z) = (R, -Z), the stellarator symmetry, exact for the symmetric field; the cells are mirrored
    when the annulus is).  Points in cells without an interpolant are lost (no extrapolation): the chain's
    manifolds live between the invariant tori T1, T2, so a lost point means the curve left the annulus
    and the certificate fails.  Error per step: tau = 3 x the largest difference between the surrogate
    and the accurate tracer at n_check random points of the annulus + 2 x the tracer's own error model
    (2 x GBS estimate + 2e-13 R) of the samples, plus the 2e-13 R floor (a validated model; every run
    also compares sampled surrogate steps with the independent reference tracer)."""

    def __init__(self, engine, source, T1, T2, phi_s=0.0, sign=1.0, scale=None, cell=None, n=12, margin=None,
                 n_check=300, reference=True, seed=0, chunk=20000):
        super().__init__(engine, source, phi_s=phi_s, sign=sign, scale=scale, reference=reference)
        self.P1 = torus_section(T1, phi_s, 2048)
        self.P2 = torus_section(T2, phi_s, 2048)
        diam = float(np.max(np.ptp(self.P2, axis=0)))
        self.cell = float(cell if cell is not None else 0.04 * diam)
        self.n = int(n)
        self.margin = float(margin if margin is not None else 2e-3 * self.scale)
        self.n_check = int(n_check)
        self.seed = seed
        self.chunk = int(chunk)
        self.info = {}
        # the nodes are traced with 24 macro steps per period (the GBS estimate, which enters tau, falls by
        # ~2^10 against the 12 steps of the growth tracer; the samples are traced only once)
        self.exact = PlaneSection(engine, source, phi_s=phi_s, sign=sign, scale=scale, reference=False,
                                  steps_per_period=24)

    def _cells_of(self, y):
        i = np.floor((y[:, 0] - self.R0) / self.cell).astype(np.int64)
        j = np.floor((y[:, 1] - self.Z0) / self.cell).astype(np.int64)
        ok = (i >= 0) & (i < self.ni) & (j >= 0) & (j < self.nj)
        idx = np.full(len(y), -1, np.int64)
        idx[ok] = self.cell_id[i[ok], j[ok]]
        return idx, i, j

    def in_annulus(self, y, margin=0.0):
        y = np.atleast_2d(y)
        ins2 = BR.inside_polygon(self.P2, y)
        out1 = ~BR.inside_polygon(self.P1, y)
        ok = ins2 & out1
        if margin > 0:
            near = (BR.polyline_distance(y, np.concatenate([self.P1, self.P1[:1]])) <= margin) | \
                (BR.polyline_distance(y, np.concatenate([self.P2, self.P2[:1]])) <= margin)
            ok |= near
        return ok

    def build(self, verbose=False, deadline=None):
        t0 = time.time()
        c = self.cell
        lo = np.minimum(self.P2.min(0), (self.P2 * np.array([1.0, -1.0])).min(0)) - 2 * c
        hi = np.maximum(self.P2.max(0), (self.P2 * np.array([1.0, -1.0])).max(0)) + 2 * c
        self.R0, self.Z0 = float(lo[0]), float(lo[1])
        self.ni = int(np.ceil((hi[0] - lo[0]) / c))
        self.nj = int(np.ceil((hi[1] - lo[1]) / c))
        # cells meeting the annulus: the segments P1(theta) -> P2(theta) sweep it (a homotopy between the two
        # boundary curves covers the region between them), sampled finer than a cell; plus the boundary curves
        # offset outwards by the margin along their normals
        L = np.linalg.norm(self.P2 - self.P1, axis=1)
        m = int(min(400, max(8, np.ceil(3.0 * np.max(L) / c) + 2)))
        t = np.linspace(0.0, 1.0, m)
        S = (self.P1[:, None, :] + t[None, :, None] * (self.P2 - self.P1)[:, None, :]).reshape(-1, 2)
        offs = []
        for P, sgn in ((self.P1, -1.0), (self.P2, 1.0)):
            tng = np.roll(P, -1, 0) - np.roll(P, 1, 0)
            nrm = np.stack([tng[:, 1], -tng[:, 0]], -1) / np.linalg.norm(tng, axis=1)[:, None]
            if BR.signed_area(P) < 0:
                nrm = -nrm                                    # outward for a counter-clockwise curve
            for f in np.linspace(0.0, 1.0, int(np.ceil(self.margin / (c / 3))) + 1):
                offs.append(P + sgn * f * self.margin * nrm)
        S = np.concatenate([S] + offs)
        S = np.concatenate([S, S * np.array([1.0, -1.0])])     # mirror-symmetric cover (backward map)
        act = np.zeros((self.ni, self.nj), bool)
        ii = np.clip(np.floor((S[:, 0] - self.R0) / c).astype(int), 0, self.ni - 1)
        jj = np.clip(np.floor((S[:, 1] - self.Z0) / c).astype(int), 0, self.nj - 1)
        act[ii, jj] = True
        self.cell_id = np.full((self.ni, self.nj), -1, np.int64)
        ci, cj = np.nonzero(act)
        self.cell_id[ci, cj] = np.arange(len(ci))
        self.cells = np.stack([ci, cj], -1)
        nc = len(ci)
        u = _cheb_nodes(self.n)
        U, V = np.meshgrid(u, u, indexing="ij")
        Rn = self.R0 + (ci[:, None, None] + 0.5 + 0.5 * U[None]) * c
        Zn = self.Z0 + (cj[:, None, None] + 0.5 + 0.5 * V[None]) * c
        X = np.stack([Rn.ravel(), Zn.ravel()], -1)
        if verbose:
            print(f"   surrogate: {nc} cells of {c:.2e} m, {len(X)} nodes ({time.time() - t0:.0f}s)", flush=True)
        sp0 = (_first_steps(self, X, True, deadline=deadline)
               if (_p3("surrogate_steps") and len(X) > 4 * FIRST_SP_SAMPLE) else 24)
        self.first_sp = sp0
        if sp0 == 24:
            st0 = _gpu_step(self.exact, 24) or self.exact.step
        else:
            ex0 = PlaneSection(self.engine, self.source, phi_s=self.phi_s, sign=self.sign, scale=self.scale,
                               reference=False, steps_per_period=sp0)
            st0 = _gpu_step(ex0, sp0) or ex0.step
        Y, lost, tau = _trace_batched(st0, X, self.chunk, deadline=deadline, retrace=True)
        if np.any(lost):
            # cells with a node whose line cannot be followed for a period (beyond the confinement region,
            # B_phi sign change) carry no interpolant: points there are lost
            bad_cells = np.unique(np.flatnonzero(lost) // (self.n * self.n))
            if len(bad_cells) > 0.5 * nc:
                raise RuntimeError(f"{len(bad_cells)} of {nc} surrogate cells have lost nodes")
            self.cell_id[ci[bad_cells], cj[bad_cells]] = -1
            Y[lost] = X[lost]
            tau[lost] = 0.0
            self.lost_cells = int(len(bad_cells))
        # nodes whose error model is large (strongly varying field): traced again with 2x, 4x the steps
        for sp in (48, 96):
            if sp <= sp0:
                continue
            big = np.flatnonzero(tau > 5e-12 * np.maximum(1.0, np.abs(Y[:, 0])))
            if len(big) == 0:
                break
            fine = PlaneSection(self.engine, self.source, phi_s=self.phi_s, sign=self.sign, scale=self.scale,
                                reference=False, steps_per_period=sp)
            y2, l2, t2 = _trace_batched(_gpu_step(fine, sp) or fine.step, X[big], self.chunk, deadline=deadline,
                                        what=f"nodes at {sp} steps")
            better = (~l2) & (t2 < tau[big])
            Y[big[better]], tau[big[better]] = y2[better], t2[better]
            self.retraced = int(len(big))
        Bp = np.concatenate([self.sign * self._bphi(self.B(self.xyz(X[k:k + self.chunk])))
                             for k in range(0, len(X), self.chunk)])
        Vals = np.stack([Y[:, 0], Y[:, 1], Bp], 0).reshape(3, nc, self.n, self.n)
        self.coef = _cheb_coef2(Vals)                          # (3, nc, n, n)
        self.tau_samples = float(np.max(tau))
        self.tau_cell = np.max(tau.reshape(nc, self.n * self.n), axis=1)   # the tracer's model per cell
        tail = float(np.max(np.abs(self.coef[:2, :, -2:, :])) + np.max(np.abs(self.coef[:2, :, :, -2:])))
        # validation at random points of the annulus
        rng = np.random.default_rng(self.seed)
        blo, bhi = self.P2.min(0), self.P2.max(0)
        cand = np.zeros((0, 2))
        for _ in range(20):
            c_ = rng.uniform(blo, bhi, size=(20000, 2))
            cand = np.concatenate([cand, c_[self.in_annulus(c_)]])
            if len(cand) >= self.n_check:
                break
        cand = cand[:self.n_check]
        Yc, lc, tauc = self.exact.step(cand, np.ones(len(cand)))
        for sp in (48, 96):
            big = np.flatnonzero(tauc > 5e-12 * np.maximum(1.0, np.abs(Yc[:, 0])))
            if len(big) == 0:
                break
            fine = PlaneSection(self.engine, self.source, phi_s=self.phi_s, sign=self.sign, scale=self.scale,
                                reference=False, steps_per_period=sp)
            y2, l2, t2 = fine.step(cand[big], np.ones(len(big)))
            better = (~l2) & (t2 < tauc[big])
            Yc[big[better]], tauc[big[better]] = y2[better], t2[better]
        Ys, ls, _ = self.step(cand, np.ones(len(cand)), _raw=True)
        d = np.linalg.norm(Ys - Yc, axis=1)
        good = ~(lc | ls)
        err = float(np.max(d[good])) if np.any(good) else np.inf
        db = self.density(cand) / (self.sign * self._bphi(self.B(self.xyz(cand)))) - 1.0
        self.density_rel_err = float(np.max(np.abs(db)))
        # per cell: 3 x the validation difference + 2 x the tracer's model of the cell's nodes (and of the
        # check points falling in it); the error of a step from a point is its cell's
        tc = self.tau_cell.copy()
        ic = self._cells_of(cand)[0]
        okc = ic >= 0
        if np.any(okc):
            np.maximum.at(tc, ic[okc], tauc[okc])
        self.tau_cell = 3.0 * err + 2.0 * tc
        self.tau_surr = float(np.max(self.tau_cell))
        ins = self.in_annulus(np.stack([self.R0 + (self.cells[:, 0] + 0.5) * c, self.Z0 + (self.cells[:, 1] + 0.5) * c], -1),
                              margin=c)
        self.tau_annulus = float(np.max(self.tau_cell[ins])) if np.any(ins) else self.tau_surr
        self.info = dict(cells=int(nc), cell=c, n=self.n, nodes=int(len(X)), retraced=getattr(self, "retraced", 0),
                         lost_cells=getattr(self, "lost_cells", 0), n_check=int(np.sum(good)),
                         lost_checks=int(np.sum(ls & ~lc)), max_check_diff=err, tau_samples=self.tau_samples,
                         tau=self.tau_surr, tau_annulus=self.tau_annulus, coef_tail=tail, density_rel_err=self.density_rel_err,
                         build_time=time.time() - t0)
        if verbose:
            print(f"   surrogate: check max diff {err:.2e} ({int(np.sum(good))} points), tau {self.tau_annulus:.2e} "
                  f"(max over cells {self.tau_surr:.2e}), "
                  f"tail {tail:.1e}, B_phi rel {self.density_rel_err:.1e} ({time.time() - t0:.0f}s)", flush=True)
        return self.info

    def _eval(self, y, which):
        idx, i, j = self._cells_of(y)
        ok = idx >= 0
        out = np.full((len(which), len(y)), np.nan)
        if np.any(ok):
            k = idx[ok]
            u = 2.0 * ((y[ok, 0] - self.R0) / self.cell - i[ok]) - 1.0
            v = 2.0 * ((y[ok, 1] - self.Z0) / self.cell - j[ok]) - 1.0
            Tu = _cheb_vander(u, self.n)
            Tv = _cheb_vander(v, self.n)
            for a, w in enumerate(which):
                out[a, ok] = np.einsum("ki,kij,kj->k", Tu, self.coef[w][k], Tv)
        return out, ok

    def density(self, y):
        y = np.atleast_2d(np.asarray(y, float))
        o, ok = self._eval(y, (2,))
        return o[0]

    def jacobian(self, y):
        """d(R', Z') / d(R, Z) of the forward surrogate map (the cell's interpolant differentiated)."""
        y = np.atleast_2d(np.asarray(y, float))
        idx, i, j = self._cells_of(y)
        J = np.full((len(y), 2, 2), np.nan)
        ok = idx >= 0
        if np.any(ok):
            k = idx[ok]
            u = 2.0 * ((y[ok, 0] - self.R0) / self.cell - i[ok]) - 1.0
            v = 2.0 * ((y[ok, 1] - self.Z0) / self.cell - j[ok]) - 1.0
            Tu, Tv = _cheb_vander(u, self.n), _cheb_vander(v, self.n)
            Du, Dv = _cheb_vander_d(u, self.n), _cheb_vander_d(v, self.n)
            for a in range(2):
                J[ok, a, 0] = np.einsum("ki,kij,kj->k", Du, self.coef[a][k], Tv) * 2.0 / self.cell
                J[ok, a, 1] = np.einsum("ki,kij,kj->k", Tu, self.coef[a][k], Dv) * 2.0 / self.cell
        return J

    def step(self, y, d, _raw=False):
        y = np.asarray(y, float).reshape(-1, 2)
        d = np.broadcast_to(np.asarray(d), (len(y),))
        sg = np.where(d > 0, 1.0, -1.0)
        flip = np.stack([np.ones(len(y)), sg], -1)
        o, ok = self._eval(y * flip, (0, 1))
        out = np.stack([o[0], o[1]], -1) * flip
        lost = ~(ok & np.all(np.isfinite(out), -1))
        if np.any(~lost):
            ok2 = self._cells_of(out * flip)[0] >= 0              # the image must stay where the map is known
            lost |= ~ok2
        self.n_steps += len(y)
        if _raw:
            return out, lost, np.zeros(len(y))
        idx = self._cells_of(y * flip)[0]
        tc = np.where(idx >= 0, self.tau_cell[np.maximum(idx, 0)], self.tau_surr)
        tau = tc + 2e-13 * np.maximum(1.0, np.abs(out[:, 0]))
        return out, lost, tau


# ============================================================================ surrogate for thin annuli
def section_coeffs(T, phi):
    """Fourier coefficients in theta of the section at phi of a TorusState: (4, M+1) = R cos, R sin,
    Z cos, Z sin."""
    n = np.arange(-T.N, T.N + 1)
    psi = n * T.nfp * (phi - T.phi0)
    c, s = np.cos(psi), np.sin(psi)
    rc, zs = np.asarray(T.rc), np.asarray(T.zs)
    return np.stack([rc @ c, rc @ s, -(zs @ s), zs @ c], 0)


class SectionCurve:
    """A closed curve x(theta) = sum_m (Rc, Zc) cos m theta + (Rs, Zs) sin m theta with derivatives, the
    outward unit normal, and the normal coordinates (theta, d) of points: x = c(theta) + d n(theta), theta
    the foot of the perpendicular (Newton from the nearest sample)."""

    def __init__(self, coef):
        self.f = np.asarray(coef, float)
        self.m = np.arange(self.f.shape[1])
        th = 2 * np.pi * np.arange(4096) / 4096
        x, _, _ = self.eval(th)
        self.orient = 1.0 if BR.signed_area(x) > 0 else -1.0
        from scipy.spatial import cKDTree
        self._th = th
        self._tree = cKDTree(x)

    def eval(self, th):
        th = np.atleast_1d(np.asarray(th, float))
        a = np.outer(th, self.m)
        C, S = np.cos(a), np.sin(a)
        f, m = self.f, self.m
        x = np.stack([C @ f[0] + S @ f[1], C @ f[2] + S @ f[3]], -1)
        x1 = np.stack([-S @ (m * f[0]) + C @ (m * f[1]), -S @ (m * f[2]) + C @ (m * f[3])], -1)
        x2 = np.stack([-C @ (m * m * f[0]) - S @ (m * m * f[1]), -C @ (m * m * f[2]) - S @ (m * m * f[3])], -1)
        return x, x1, x2

    def arclength(self, n=1024):
        """The same curve re-parameterised (to interpolation accuracy) by arc length: the trigonometric
        interpolant of n points of the curve at equal arc length.  Any smooth curve serves as reference;
        uniform speed keeps the patches' functions free of the parameterisation's harmonics."""
        th = 2 * np.pi * np.arange(16 * n) / (16 * n)
        x, x1, _ = self.eval(th)
        sp = np.linalg.norm(x1, axis=1)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (sp + np.roll(sp, -1)) * (2 * np.pi / (16 * n)))])
        tq = np.interp(np.linspace(0, s[-1], n + 1)[:-1], s, np.concatenate([th, [2 * np.pi]]))
        P, _, _ = self.eval(tq)
        F = np.fft.rfft(P, axis=0) / n
        F[1:] *= 2.0
        K = n // 2 - 1
        coef = np.stack([F[:K + 1, 0].real, -F[:K + 1, 0].imag, F[:K + 1, 1].real, -F[:K + 1, 1].imag], 0)
        return SectionCurve(coef)

    def normal(self, x1):
        nr = np.linalg.norm(x1, axis=1)
        return self.orient * np.stack([x1[:, 1], -x1[:, 0]], -1) / nr[:, None]

    def point(self, th, d):
        x, x1, _ = self.eval(th)
        return x + np.asarray(d, float)[:, None] * self.normal(x1)

    def coords(self, y, it=8):
        """(theta in [0, 2 pi), d, ok): ok False where the foot is not a local minimum of the distance
        (beyond the curve's reach) or Newton did not settle."""
        y = np.atleast_2d(np.asarray(y, float))
        _, idx = self._tree.query(y)
        th = self._th[idx].copy()
        for _ in range(it):
            x, x1, x2 = self.eval(th)
            r = y - x
            g = np.sum(r * x1, 1)
            gp = -np.sum(x1 * x1, 1) + np.sum(r * x2, 1)
            step = -g / np.where(gp != 0, gp, -1.0)
            th = th + step
            if len(step) == 0 or np.max(np.abs(step)) < 1e-15:
                break
        x, x1, x2 = self.eval(th)
        r = y - x
        n = self.normal(x1)
        d = np.sum(r * n, 1)
        gp = -np.sum(x1 * x1, 1) + np.sum(r * x2, 1)
        resid = np.linalg.norm(r - d[:, None] * n, axis=1)
        ok = (gp < 0) & (resid < 1e-11 * max(1.0, float(np.max(np.abs(y)))))
        return np.mod(th, 2 * np.pi), d, ok


def _cheb_coef2r(V):
    """2D Chebyshev coefficients from values on a Lobatto tensor grid (..., nt, nd)."""
    from scipy.fft import dct
    nt, nd = V.shape[-2], V.shape[-1]
    C = dct(dct(V, type=1, axis=-1), type=1, axis=-2) / ((nt - 1) * (nd - 1))
    C[..., 0, :] *= 0.5
    C[..., -1, :] *= 0.5
    C[..., :, 0] *= 0.5
    C[..., :, -1] *= 0.5
    return C


class BandSurrogate(PlaneSection):
    """PlaneSection whose period map is a validated piecewise-Chebyshev surrogate on patches aligned with
    the annulus between the sections of T1 and T2: normal coordinates (theta, d) of a smooth reference
    curve c (the base surface of offset proxies, else T1's section low-pass filtered to m <= 8 (12, 16, 6,
    24, all: the first whose normal coordinates cover the band one to one)), x = c(theta) + d n(theta);
    patches at most `patch_len` long along c and turning by at most 0.3 rad (short patches at the tips of
    bean-shaped sections, where the normal coordinates bend), and across exactly the annulus' local width
    plus `margin` x that width on each side; nt x nd Chebyshev-Lobatto nodes per patch (nd = 8, up to 32
    when the annulus is as wide as the patches are long) for the image (R', Z') and B_phi, from one batch
    of accurate traces (24 macro steps per period, nodes with large error estimates traced again with 48
    and 96); analytic Jacobians from the interpolants (for monodromies: finite differences would see the
    patches' jumps); the evaluation compiled (numba) when available.  A thin annulus (0.1 mm across) thus gets thin patches: no node lies far from
    it (square cells of a centimetre reached the coils' neighbourhood).  Points outside the patches, at a
    foot beyond the curve's reach, or in a patch with a lost node are lost.  Error per step: per patch,
    3 x the largest difference from the accurate tracer at random annulus points + 2 x the tracer's
    model of the patch's nodes and check points, plus 2e-13 R (a validated model); every run also
    compares sampled steps with the independent reference tracer.  The backward map is S F S."""

    def __init__(self, engine, source, T1, T2, phi_s=0.0, sign=1.0, scale=None, patch_len=None, nt=12, nd=None,
                 margin=0.25, n_check=400, reference=True, seed=0, chunk=20000):
        super().__init__(engine, source, phi_s=phi_s, sign=sign, scale=scale, reference=reference)
        self.margin = float(margin)
        offs = isinstance(T1, OffsetSurface) and isinstance(T2, OffsetSurface) and T1.base is T2.base
        if offs:
            self.ref = SectionCurve(section_coeffs(T1.base, phi_s))
        else:
            # a smooth reference curve: T1's section low-pass filtered (a torus next to a chain carries slowly
            # decaying harmonics of q in its straight-field-line parameterisation, which the patches would have
            # to resolve); the first filter whose normal coordinates cover T1 and T2 within the curve's reach
            cf = section_coeffs(T1, phi_s)
            ref, m_ref = None, None
            for m_try in (8, 12, 16, 6, 24, cf.shape[1] - 1):
                if m_try > cf.shape[1] - 1:
                    continue
                cand = SectionCurve(cf[:, :m_try + 1])
                if self._roundtrip_ok(cand, (T1, T2), phi_s):
                    ref, m_ref = cand, m_try
                    break
            if ref is None:
                raise RuntimeError("no reference curve whose normal coordinates cover the annulus one-to-one")
            self.ref = ref
            self.m_ref = m_ref
        self.nt = int(nt)
        self.margin = float(margin)
        self.n_check, self.seed, self.chunk = int(n_check), seed, int(chunk)
        th = 2 * np.pi * np.arange(8192) / 8192
        x, x1, x2 = self.ref.eval(th)
        sp = np.linalg.norm(x1, axis=1)
        kap = np.abs(x1[:, 0] * x2[:, 1] - x1[:, 1] * x2[:, 0]) / sp ** 3
        ds = sp * (2 * np.pi / 8192)
        self.perimeter = float(np.sum(ds))
        diam = float(np.max(np.ptp(x, axis=0)))
        self.patch_len = float(patch_len if patch_len is not None else 0.02 * diam)
        # patch boundaries at equal increments of int max(1 / patch_len, kappa / 0.3) ds: a patch is at most
        # patch_len long and turns by at most 0.3 rad (the normal coordinates bend with the curve; a sharp tip of
        # a bean-shaped section gets short patches)
        dens = np.maximum(1.0 / self.patch_len, np.maximum(kap, np.roll(kap, 1)) / 0.3) * ds
        cost = np.concatenate([[0.0], np.cumsum(dens)])
        npch = max(16, int(np.ceil(cost[-1])))
        self.tb = np.interp(np.linspace(0, cost[-1], npch + 1), cost, np.concatenate([th, [2 * np.pi]]))
        self.tb[0], self.tb[-1] = 0.0, 2 * np.pi
        self.np_ = npch
        # the annulus' normal range per patch: T1 and T2 sections in the reference's normal coordinates
        if offs:
            # the offsets (constant or following the chain) per patch, neighbours included
            tt = 2 * np.pi * (np.arange(8192) + 0.5) / 8192
            o1, o2 = T1.offset(tt, np.full(8192, phi_s)), T2.offset(tt, np.full(8192, phi_s))
            kk = np.clip(np.searchsorted(self.tb, tt, side="right") - 1, 0, npch - 1)
            lo_t, hi_t = np.full(npch, np.inf), np.full(npch, -np.inf)
            np.minimum.at(lo_t, kk, np.minimum(o1, o2))
            np.maximum.at(hi_t, kk, np.maximum(o1, o2))
            lo_t = np.minimum(lo_t, np.minimum(np.roll(lo_t, 1), np.roll(lo_t, -1)))
            hi_t = np.maximum(hi_t, np.maximum(np.roll(hi_t, 1), np.roll(hi_t, -1)))
            self.reach_ok = True
        else:
            lo_t, hi_t = np.full(npch, np.inf), np.full(npch, -np.inf)
            ok_all = True
            for T in (T1, T2):
                P = torus_section(T, phi_s, 8192)
                tt, dd, ok = self.ref.coords(P)
                ok_all &= bool(np.all(ok))
                k = np.clip(np.searchsorted(self.tb, tt, side="right") - 1, 0, npch - 1)
                np.minimum.at(lo_t, k, dd)
                np.maximum.at(hi_t, k, dd)
            # neighbouring patches' ranges too (the boundary curves are sampled, not continuous in the patch)
            lo_t = np.minimum(lo_t, np.minimum(np.roll(lo_t, 1), np.roll(lo_t, -1)))
            hi_t = np.maximum(hi_t, np.maximum(np.roll(hi_t, 1), np.roll(hi_t, -1)))
            self.reach_ok = ok_all
        w = hi_t - lo_t
        # across: 8 nodes for a thin annulus, more when it is as wide as the patches are long
        ratio = (1.0 + 2.0 * self.margin) * float(np.max(w)) / self.patch_len
        self.nd = int(nd) if nd is not None else int(np.clip(8 * (1 + np.ceil(max(ratio - 0.5, 0.0))), 8, 32))
        from . import fast_trace as _so4
        if _so4.on("snodes") and nd is None and nt == 12 and self.nd == 8:
            # (fast_trace 'snodes') a thin annulus' patches with SNODES_NT x SNODES_ND nodes instead of 12 x 8 (the
            # truncated interpolants of the profile's accepted builds met their own check differences at 9 x 5 or
            # fewer); the build validates itself as before (tau from its own check points)
            self.nt, self.nd = int(_so4.SNODES_NT), int(_so4.SNODES_ND)
        self.ann_lo, self.ann_hi = lo_t, hi_t
        from . import fast_kernels as _so5
        if _so5.on("pnd") and nd is None and self.nt == 12 and self.nd > 8:
            # (fast_kernels 'pnd') the across-count rule at each patch's own width (the widest's count above)
            self.nd_k = _so5.band_nd_per_patch(self, self.nd)
            self.nd = int(np.max(self.nd_k))
        self.d_lo = lo_t - self.margin * w - 1e-9 * self.scale
        self.d_hi = hi_t + self.margin * w + 1e-9 * self.scale
        self.active = np.ones(npch, bool)
        self.info = {}
        self.exact = PlaneSection(engine, source, phi_s=phi_s, sign=sign, scale=scale, reference=False,
                                  steps_per_period=24)

    def _roundtrip_ok(self, ref, tori, phi_s, n_th=1024, n_d=9):
        """The normal coordinates of ref cover the band of T1, T2 (widened by the patches' margins) one to
        one: every grid point (theta, d) maps back to itself (the truncation of a straight-field-line
        section can make cusps whose normals cross)."""
        dd = []
        for T in tori:
            t_, d_, ok = ref.coords(torus_section(T, phi_s, 2048))
            if not np.all(ok):
                return False
            dd.append(d_)
        dd = np.concatenate(dd)
        w = float(dd.max() - dd.min())
        D = np.linspace(dd.min() - (self.margin + 0.1) * w, dd.max() + (self.margin + 0.1) * w, n_d)
        th = 2 * np.pi * np.arange(n_th) / n_th
        TT, DD = np.meshgrid(th, D, indexing="ij")
        tt, d2, ok = ref.coords(ref.point(TT.ravel(), DD.ravel()))
        dth = np.abs(np.angle(np.exp(1j * (tt - TT.ravel()))))
        return bool(np.all(ok) and np.max(dth) < 1e-9 and np.max(np.abs(d2 - DD.ravel())) < 1e-12 * max(self.scale, 1.0))

    def _locate(self, y):
        y = np.ascontiguousarray(np.atleast_2d(np.asarray(y, float)))
        if _njit is not None:
            if not hasattr(self, "_nb"):
                th_s = 2 * np.pi * np.arange(512) / 512
                xs, _, _ = self.ref.eval(th_s)
                self._nb = (np.ascontiguousarray(self.ref.f), float(self.ref.orient), th_s, np.ascontiguousarray(xs))
            f, o_, th_s, xs = self._nb
            out = np.empty((len(y), 6))
            _band_locate_nb(f, o_, th_s, xs, self.tb, self.d_lo, self.d_hi, self.active, y, out)
            return out[:, 0].astype(np.int64), out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5] > 0.5
        th, d, ok = self.ref.coords(y)
        k = np.clip(np.searchsorted(self.tb, th, side="right") - 1, 0, self.np_ - 1)
        u = 2.0 * (th - self.tb[k]) / (self.tb[k + 1] - self.tb[k]) - 1.0
        v = 2.0 * (d - self.d_lo[k]) / (self.d_hi[k] - self.d_lo[k]) - 1.0
        ok &= (np.abs(v) <= 1.0 + 1e-12) & self.active[k]
        return k, u, v, th, d, ok

    def in_annulus(self, y, margin=0.0):
        k, u, v, th, d, ok = self._locate(np.atleast_2d(y))
        return ok & (d >= self.ann_lo[k] - margin) & (d <= self.ann_hi[k] + margin)

    def _trace_nodes(self, X, deadline=None, gpu=True):
        """gpu: the node traces on the card when one is present and POLICY 'card_traces' is on (torux.cardtrace); the
        validation points are traced
        with gpu=False (the CPU accurate tracer: the validated error model does not rest on the
        GPU traces)."""
        sp0 = (_first_steps(self, X, gpu, deadline=deadline)
               if (_p3("surrogate_steps") and len(X) > 4 * FIRST_SP_SAMPLE) else 24)
        self.first_sp = sp0
        if sp0 == 24:
            st0 = (_gpu_step(self.exact, 24) if gpu else None) or self.exact.step
        else:
            ex0 = PlaneSection(self.engine, self.source, phi_s=self.phi_s, sign=self.sign, scale=self.scale,
                               reference=False, steps_per_period=sp0)
            st0 = (_gpu_step(ex0, sp0) if gpu else None) or ex0.step
        Y, lost, tau = _trace_batched(st0, X, self.chunk, deadline=deadline, retrace=True)
        retraced = 0
        for sp in (48, 96):
            if sp <= sp0:
                continue
            big = np.flatnonzero(~lost & (tau > 5e-12 * np.maximum(1.0, np.abs(Y[:, 0]))))
            if len(big) == 0:
                break
            retraced = max(retraced, len(big))
            fine = PlaneSection(self.engine, self.source, phi_s=self.phi_s, sign=self.sign, scale=self.scale,
                                reference=False, steps_per_period=sp)
            y2, l2, t2 = _trace_batched((_gpu_step(fine, sp) if gpu else None) or fine.step, X[big], self.chunk,
                                        deadline=deadline, what=f"nodes at {sp} steps")
            better = (~l2) & (t2 < tau[big])
            Y[big[better]], tau[big[better]] = y2[better], t2[better]
        return Y, lost, tau, retraced

    def build(self, verbose=False, deadline=None):
        if getattr(self, "nd_k", None) is not None:
            from . import fast_kernels as _so5
            return _so5.band_build_pnd(self, verbose=verbose, deadline=deadline)    # (fast_kernels 'pnd')
        t0 = time.time()
        npch, nt, nd = self.np_, self.nt, self.nd
        ut = _cheb_nodes(nt)
        vd = _cheb_nodes(nd)
        TH = self.tb[:-1, None, None] + 0.5 * (ut[None, :, None] + 1.0) * np.diff(self.tb)[:, None, None]
        DD = self.d_lo[:, None, None] + 0.5 * (vd[None, None, :] + 1.0) * (self.d_hi - self.d_lo)[:, None, None]
        TH, DD = np.broadcast_arrays(TH, DD)
        X = self.ref.point(TH.ravel(), DD.ravel())
        if verbose:
            print(f"   band surrogate: {npch} patches of {self.patch_len:.2e} m x {np.median(self.d_hi - self.d_lo):.2e} m "
                  f"(median), {len(X)} nodes ({time.time() - t0:.0f}s)", flush=True)
        Y, lost, tau, retraced = self._trace_nodes(X, deadline=deadline)
        per = nt * nd
        bad = np.unique(np.flatnonzero(lost) // per)
        if len(bad) > 0.5 * npch:
            raise RuntimeError(f"{len(bad)} of {npch} surrogate patches have lost nodes")
        self.active[bad] = False
        Y[lost] = X[lost]
        tau[lost] = 0.0
        Bp = np.concatenate([self.sign * self._bphi(self.B(self.xyz(X[k:k + self.chunk])))
                             for k in range(0, len(X), self.chunk)])
        Vals = np.stack([Y[:, 0], Y[:, 1], Bp], 0).reshape(3, npch, nt, nd)
        self.coef = _cheb_coef2r(Vals)
        self.tau_patch = np.max(tau.reshape(npch, per), axis=1)
        tail = float(max(np.max(np.abs(self.coef[:2, self.active, -2:, :])),
                         np.max(np.abs(self.coef[:2, self.active, :, -2:]))))
        # validation at random points of the annulus (each patch's own normal range)
        rng = np.random.default_rng(self.seed)
        k = rng.integers(0, npch, size=self.n_check)
        k = k[self.active[k]]
        tq = self.tb[k] + rng.uniform(0, 1, len(k)) * (self.tb[k + 1] - self.tb[k])
        dq = self.ann_lo[k] + rng.uniform(0, 1, len(k)) * (self.ann_hi[k] - self.ann_lo[k])
        cand = self.ref.point(tq, dq)
        Yc, lc, tauc, _ = self._trace_nodes(cand, deadline=deadline, gpu=False)
        Ys, ls, _ = self.step(cand, np.ones(len(cand)), _raw=True)
        dist = np.linalg.norm(Ys - Yc, axis=1)
        good = ~(lc | ls)
        err = float(np.max(dist[good])) if np.any(good) else np.inf
        if not np.any(good):
            raise RuntimeError("no surrogate check point could be evaluated")
        db = self.density(cand[good]) / (self.sign * self._bphi(self.B(self.xyz(cand[good])))) - 1.0
        self.density_rel_err = float(np.max(np.abs(db)))
        tp = self.tau_patch.copy()
        kc = self._locate(cand)[0]
        np.maximum.at(tp, kc[good], tauc[good])
        self.tau_cell = 3.0 * err + 2.0 * tp
        self.tau_surr = float(np.max(self.tau_cell[self.active]))
        self.tau_annulus = self.tau_surr
        self.info = dict(kind="band", patches=int(npch), patch_len=self.patch_len, nt=nt, nd=nd, nodes=int(len(X)),
                         m_ref=getattr(self, "m_ref", None),
                         width_median=float(np.median(self.ann_hi - self.ann_lo)), retraced=int(retraced),
                         lost_patches=int(len(bad)), reach_ok=bool(self.reach_ok), n_check=int(np.sum(good)),
                         lost_checks=int(np.sum(ls & ~lc)), max_check_diff=err, tau_samples=float(np.max(tau)),
                         tau=self.tau_surr, tau_annulus=self.tau_annulus, coef_tail=tail,
                         density_rel_err=self.density_rel_err, build_time=time.time() - t0,
                         gpu=getattr(self.exact, "gpu_stats", None), first_sp=getattr(self, "first_sp", 24))
        if verbose:
            print(f"   band surrogate: check max diff {err:.2e} ({int(np.sum(good))} points), tau {self.tau_surr:.2e}, "
                  f"tail {tail:.1e}, B_phi rel {self.density_rel_err:.1e}, lost patches {len(bad)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return self.info

    def _eval(self, y, which):
        k, u, v, th, d, ok = self._locate(y)
        out = np.full((len(which), len(y)), np.nan)
        if _njit is not None:
            if not hasattr(self, "_coef_c"):
                self._coef_c = np.ascontiguousarray(self.coef)
            _band_values_nb(self._coef_c, np.asarray(which, np.int64), k, u, v, ok, out)
            return out, ok, k
        if np.any(ok):
            Tu = _cheb_vander(np.clip(u[ok], -1, 1), self.nt)
            Tv = _cheb_vander(np.clip(v[ok], -1, 1), self.nd)
            kk = k[ok]
            for a, w in enumerate(which):
                out[a, ok] = np.einsum("ki,kij,kj->k", Tu, self.coef[w][kk], Tv)
        return out, ok, k

    def density(self, y):
        o, ok, _ = self._eval(np.atleast_2d(np.asarray(y, float)), (2,))
        return o[0]

    def jacobian(self, y):
        """d(R', Z') / d(R, Z) of the forward surrogate map: the patch interpolant's derivative in (theta, d)
        times the inverse of d x / d(theta, d) = [c' + d n', n]."""
        y = np.atleast_2d(np.asarray(y, float))
        k, u, v, th, d, ok = self._locate(y)
        J = np.full((len(y), 2, 2), np.nan)
        if np.any(ok):
            kk = k[ok]
            uu, vv = np.clip(u[ok], -1, 1), np.clip(v[ok], -1, 1)
            Tu, Tv = _cheb_vander(uu, self.nt), _cheb_vander(vv, self.nd)
            Du, Dv = _cheb_vander_d(uu, self.nt), _cheb_vander_d(vv, self.nd)
            du = 2.0 / (self.tb[kk + 1] - self.tb[kk])
            dv = 2.0 / (self.d_hi[kk] - self.d_lo[kk])
            G = np.empty((len(kk), 2, 2))
            for a in range(2):
                G[:, a, 0] = np.einsum("ki,kij,kj->k", Du, self.coef[a][kk], Tv) * du
                G[:, a, 1] = np.einsum("ki,kij,kj->k", Tu, self.coef[a][kk], Dv) * dv
            x, x1, x2 = self.ref.eval(th[ok])
            sp = np.linalg.norm(x1, axis=1)
            n = self.ref.normal(x1)
            n1 = self.ref.orient * np.stack([x2[:, 1], -x2[:, 0]], -1) / sp[:, None] \
                - n * (np.sum(x1 * x2, 1) / sp ** 2)[:, None]
            A = np.stack([x1 + d[ok][:, None] * n1, n], -1)          # columns d x / d theta, d x / d d
            J[ok] = G @ np.linalg.inv(A)
        return J

    def step(self, y, d, _raw=False):
        y = np.asarray(y, float).reshape(-1, 2)
        dd = np.broadcast_to(np.asarray(d), (len(y),))
        sg = np.where(dd > 0, 1.0, -1.0)
        flip = np.stack([np.ones(len(y)), sg], -1)
        o, ok, k = self._eval(y * flip, (0, 1))
        out = np.stack([o[0], o[1]], -1) * flip
        lost = ~(ok & np.all(np.isfinite(out), -1))
        if np.any(~lost):
            ok2 = self._locate(np.where(lost[:, None], y, out) * flip)[5]
            lost |= ~ok2
        self.n_steps += len(y)
        if _raw:
            return out, lost, np.zeros(len(y))
        tc = np.where(ok, self.tau_cell[k], self.tau_surr)
        return out, lost, tc + 2e-13 * np.maximum(1.0, np.abs(out[:, 0]))


# ============================================================================ compiled band-surrogate kernels
try:
    from numba import njit as _njit
except Exception:  # noqa: BLE001
    _njit = None

if _njit is not None:
    @_njit(cache=True)
    def _curve_eval_nb(f, th):
        M = f.shape[1]
        x0 = 0.0
        y0 = 0.0
        xd = 0.0
        yd = 0.0
        xdd = 0.0
        ydd = 0.0
        for m in range(M):
            c = np.cos(m * th)
            s = np.sin(m * th)
            x0 += f[0, m] * c + f[1, m] * s
            y0 += f[2, m] * c + f[3, m] * s
            xd += m * (-f[0, m] * s + f[1, m] * c)
            yd += m * (-f[2, m] * s + f[3, m] * c)
            xdd += -m * m * (f[0, m] * c + f[1, m] * s)
            ydd += -m * m * (f[2, m] * c + f[3, m] * s)
        return x0, y0, xd, yd, xdd, ydd

    @_njit(cache=True)
    def _band_locate_nb(f, orient, samp_th, samp_x, tb, d_lo, d_hi, active, Y, out):
        """out[i] = (k, u, v, theta, d, ok) for the points Y (n, 2)."""
        n = Y.shape[0]
        ns = samp_th.shape[0]
        npch = tb.shape[0] - 1
        twopi = 2.0 * np.pi
        for i in range(n):
            yx = Y[i, 0]
            yy = Y[i, 1]
            best = 1e300
            th = 0.0
            for j in range(ns):
                dx = samp_x[j, 0] - yx
                dy = samp_x[j, 1] - yy
                dd = dx * dx + dy * dy
                if dd < best:
                    best = dd
                    th = samp_th[j]
            for it in range(12):
                x0, y0, xd, yd, xdd, ydd = _curve_eval_nb(f, th)
                rx = yx - x0
                ry = yy - y0
                g = rx * xd + ry * yd
                gp = -(xd * xd + yd * yd) + rx * xdd + ry * ydd
                if gp == 0.0:
                    break
                st = -g / gp
                th += st
                if abs(st) < 1e-15:
                    break
            x0, y0, xd, yd, xdd, ydd = _curve_eval_nb(f, th)
            rx = yx - x0
            ry = yy - y0
            sp = np.sqrt(xd * xd + yd * yd)
            nx = orient * yd / sp
            ny = -orient * xd / sp
            d = rx * nx + ry * ny
            gp = -(xd * xd + yd * yd) + rx * xdd + ry * ydd
            res = np.sqrt((rx - d * nx) ** 2 + (ry - d * ny) ** 2)
            th = th - twopi * np.floor(th / twopi)
            # patch by binary search
            lo = 0
            hi = npch
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if tb[mid] <= th:
                    lo = mid
                else:
                    hi = mid
            k = lo
            u = 2.0 * (th - tb[k]) / (tb[k + 1] - tb[k]) - 1.0
            v = 2.0 * (d - d_lo[k]) / (d_hi[k] - d_lo[k]) - 1.0
            scl = max(1.0, abs(yx), abs(yy))
            ok = (gp < 0.0) and (res < 1e-11 * scl) and (abs(v) <= 1.0 + 1e-12) and active[k]
            out[i, 0] = k
            out[i, 1] = u
            out[i, 2] = v
            out[i, 3] = th
            out[i, 4] = d
            out[i, 5] = 1.0 if ok else 0.0

    @_njit(cache=True)
    def _cheb_sum_nb(C, u, v, nt, nd):
        Tu = np.empty(nt)
        Tv = np.empty(nd)
        Tu[0] = 1.0
        Tv[0] = 1.0
        if nt > 1:
            Tu[1] = u
        if nd > 1:
            Tv[1] = v
        for a in range(2, nt):
            Tu[a] = 2 * u * Tu[a - 1] - Tu[a - 2]
        for b in range(2, nd):
            Tv[b] = 2 * v * Tv[b - 1] - Tv[b - 2]
        s = 0.0
        for a in range(nt):
            ra = 0.0
            for b in range(nd):
                ra += C[a, b] * Tv[b]
            s += Tu[a] * ra
        return s


    @_njit(cache=True)
    def _band_values_nb(coef, which, k, u, v, ok, out):
        nt = coef.shape[2]
        nd = coef.shape[3]
        for i in range(k.shape[0]):
            if not ok[i]:
                continue
            uu = min(1.0, max(-1.0, u[i]))
            vv = min(1.0, max(-1.0, v[i]))
            for a in range(which.shape[0]):
                out[a, i] = _cheb_sum_nb(coef[which[a], k[i]], uu, vv, nt, nd)


# ============================================================================ chains of several X orbits
def find_period_orbits(sec, q, n_theta=720, n_d=7, max_cand=60, scale=None):
    """Period-q orbits of the section map in a BandSurrogate's annulus: |P^q x - x| on a (theta, d) grid
    of the annulus, its local minima along theta as starts, Newton with the product of the surrogate's
    Jacobians.  Returns a list of dict(pts (q, 2), M, closure, tau_q, residue)."""
    scale = scale or sec.scale
    th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    k = np.clip(np.searchsorted(sec.tb, th, side="right") - 1, 0, sec.np_ - 1)
    D = sec.ann_lo[k][:, None] + np.linspace(0.0, 1.0, n_d)[None, :] * (sec.ann_hi[k] - sec.ann_lo[k])[:, None]
    TH = np.repeat(th[:, None], n_d, 1)
    X = sec.ref.point(TH.ravel(), D.ravel())
    Y = X.copy()
    lost = np.zeros(len(X), bool)
    for _ in range(q):
        Y, l, _ = sec.step(Y, np.ones(len(Y)))
        lost |= l
        Y[lost] = X[lost]
    F = np.where(lost, np.inf, np.linalg.norm(Y - X, axis=1)).reshape(n_theta, n_d)
    j = np.argmin(F, axis=1)
    f = F[np.arange(n_theta), j]
    loc = np.flatnonzero((f <= np.roll(f, 1)) & (f <= np.roll(f, -1)) & np.isfinite(f))
    loc = loc[np.argsort(f[loc])][:max_cand]
    found = []
    h = 1e-6 * scale
    for i in loc:
        x = X.reshape(n_theta, n_d, 2)[i, j[i]].copy()
        best = None
        for it in range(15):
            M, y0, pts, tau, ls = section_monodromy(sec, x, q, h)
            if ls:
                break
            Fv = y0 - x
            clos = float(np.linalg.norm(Fv))
            if best is None or clos < best["closure"]:
                best = dict(pts=pts, M=M, closure=clos, tau_q=tau)
            if clos < 1e-14 * scale:
                break
            try:
                x = x - np.linalg.solve(M - np.eye(2), Fv)
            except np.linalg.LinAlgError:
                break
        if best is None or best["closure"] > 1e-11 * scale:
            continue
        P0 = best["pts"]
        if q > 1 and np.min(np.linalg.norm(P0[1:] - P0[0], axis=1)) < 1e-6 * scale:
            continue                                            # a lower period
        if any(np.min(np.linalg.norm(o["pts"] - P0[0], axis=1)) < 1e-7 * scale for o in found):
            continue
        best["residue"] = float((2 - np.trace(best["M"])) / 4)
        found.append(best)
    return found


def _orbit_from_points(sec, pts, M, clos, tau, q):
    """torux.barrier.Orbit with e_X = 2 |(M - I)^-1| (closure + tau_q) and dM checked against the accurate
    tracer's Richardson-extrapolated central differences (as in x_orbit_from_line)."""
    exact = getattr(sec, "exact", sec)
    h_ex = 3e-5 * sec.scale
    fine = PlaneSection(exact.engine, exact.source, phi_s=exact.phi_s, sign=exact.sign, scale=exact.scale,
                        reference=False, steps_per_period=48)
    dM = _richardson_check(fine, pts[0], q, h_ex, M)
    inv = np.linalg.inv(M - np.eye(2))
    nrm = float(np.linalg.norm(inv, 2))
    return BR.Orbit(q, np.asarray(pts), M, e_X=2.0 * nrm * (clos + tau),
                    info=dict(closure=clos, tau_q=tau, inv_norm=nrm, det=float(np.linalg.det(M)), dM=dM,
                              monodromy="the surrogate's analytic Jacobians (checked against the accurate tracer)"))


def _crossings_multi(sec, orbs, i, c, mirror, succ, pred, delta0, a, want_dir=None, first_owner=None):
    """For the unstable curve c of orbit i: per owner X^i_k whose mirror partner is a chain neighbour, the
    first certified crossing of the branch with Z = 0 (up to four tried).  Returns records (dicts)."""
    q = orbs[i].q
    recs = []
    for k in (range(q) if first_owner is None else [first_owner]):
        if c.level() < (c.d * (k - c.base)) % q:
            continue
        node = (i, k)
        part = mirror[node]
        if part == node:
            continue
        br = BR.branch_polyline(orbs[i], c, k)
        P = br["pts"]
        if len(P) < 3:
            continue
        if succ[node] == pred[node]:
            d = int(BR._branch_dir(orbs[i], a, br)) if part == succ[node] else 0
        else:
            d = +1 if part == succ[node] else (-1 if part == pred[node] else 0)
        if d == 0 or (want_dir is not None and d != want_dir):
            continue
        idx = _first_axis_crossing(P, start=1)
        su = None
        for t_, ii in enumerate(idx[:4]):
            tt = P[ii, 1] / (P[ii, 1] - P[ii + 1, 1])
            Rc = P[ii, 0] + tt * (P[ii + 1, 0] - P[ii, 0])
            if su is None:
                su = _seg_errors(orbs[i], c, br, delta0)
            line = np.array([[Rc - 2.0 * sec.scale, 0.0], [Rc + 2.0 * sec.scale, 0.0]])
            cc = BR.certify_crossing(P, su, line, np.zeros(1), int(ii), 0)
            if not cc["ok"]:
                continue
            L_, _ = br["refs"][ii + 1]
            left, right = (node, part) if d > 0 else (part, node)
            tkey = (left[0], right[0], (right[1] - left[1]) % q)
            recs.append(dict(orbit=i, owner=k, p_orbit=part[0], partner=part[1], dir=d, h=np.array(cc["h"]),
                             box_diam=float(cc["box_diam"]), level=int(L_ if L_ is not None else 0), type=tkey,
                             cls="A" if d > 0 else "B", cert={kk: v for kk, v in cc.items() if kk != "h"}))
            break
    return recs


def _gap_arcs(sec, orb_u, orb_s, cls, cu, cs, rec, hj, ej, delta0_u, delta0_s, slack=1):
    """The q gaps of one type and class (images of rec's gap): arcs from the left to the right X point
    along the chain (the right one excluded), per-segment errors, junction data.  The branches are searched
    for the images h_j up to their levels + slack (h_j's level on the unstable branch: rec's + j; on the
    stable one: rec's - j; slack 1, or 3 for a point from the other plane: its level is that of the other
    plane's curve, +-1 here, and S x+ = P^(+-1) x)."""
    q = orb_u.q
    gaps = []
    for j in range(q):
        uo = (rec["owner"] + j) % q
        so = (rec["partner"] + j) % q
        bu = BR.branch_polyline(orb_u, cu, uo)
        bs = BR.branch_polyline(orb_s, cs, so)
        eu = _seg_errors(orb_u, cu, bu, delta0_u)
        es = _seg_errors(orb_s, cs, bs, delta0_s)
        nu_ = _upto_level(bu["refs"], rec["level"] + j + slack)
        ns_ = _upto_level(bs["refs"], max(rec["level"] - j + slack, 1))
        iu, _, du = _nearest_on_polyline(bu["pts"][:nu_], hj[j])
        is_, _, ds = _nearest_on_polyline(bs["pts"][:ns_], hj[j])
        lim_u = eu[iu] + ej[j] + 1e-12 * sec.scale
        lim_s = es[is_] + ej[j] + 1e-12 * sec.scale
        if du > 4.0 * lim_u or ds > 4.0 * lim_s:
            return None, f"heteroclinic image {j} off the computed branches (d_u {du:.2e}, d_s {ds:.2e})"
        Ju = ej[j] + du + eu[iu]
        Js = ej[j] + ds + es[is_]
        nu = (rec["orbit"], uo)
        ns = (rec["p_orbit"], so)
        if cls == "A":
            upts = bu["pts"][:iu + 1]
            uerr = np.concatenate([eu[:iu], [max(eu[iu], Ju)]])
            spts = bs["pts"][1:is_ + 1][::-1]
            serr = np.concatenate([[max(es[is_], Js)], es[:is_][::-1]])
            gaps.append(dict(start=nu, end=ns, pts=np.concatenate([upts, hj[j][None], spts]),
                             err=np.concatenate([uerr, serr]), J=max(Ju, Js)))
        else:
            s_fwd = bs["pts"][:is_ + 1]
            s_err = np.concatenate([es[:is_], [max(es[is_], Js)]])
            u_rev = bu["pts"][1:iu + 1][::-1]
            u_err = np.concatenate([[max(eu[iu], Ju)], eu[:iu][::-1]])
            gaps.append(dict(start=ns, end=nu, pts=np.concatenate([s_fwd, hj[j][None], u_rev]),
                             err=np.concatenate([s_err, u_err]), J=max(Ju, Js)))
    return gaps, None


def _chain_gaps(gap_lists, n_nodes):
    """Concatenate gaps (each from its start node to its end node) into one closed polygon."""
    by_start = {}
    for gl in gap_lists:
        for g in gl:
            if g["start"] in by_start:
                return None, None
            by_start[g["start"]] = g
    if len(by_start) != n_nodes:
        return None, None
    k0 = next(iter(by_start))
    k = k0
    C, E = [], []
    for _ in range(n_nodes):
        g = by_start[k]
        C.append(g["pts"])
        E.append(g["err"])
        k = g["end"]
    if k != k0:
        return None, None
    return np.concatenate(C), np.concatenate(E)


def _loop_flux(sec, C, E, a, corners):
    """Flux of a closed polygon made of manifold arcs that may be gamma itself (a mixed curve of case (ii)):
    the loop integral of A, oriented counter-clockwise around a (winding 0: it cannot be a curve around the
    axis), with the tube bound Delta.  Returns dict or None for winding 0."""
    C, E = decimate_closed(C, E, corners=corners)
    w = BR.winding_number(C, a)
    if w == 0:
        return None
    if w < 0:
        C = C[::-1]
        E = np.concatenate([E[:-1][::-1], E[-1:]])
    D = np.roll(C, -1, axis=0)
    if not hasattr(sec, "engine"):
        # model sections (density 1): the loop integral of A = (-y, x) / 2 is the shoelace sum
        F = float(BR.signed_area(C))
        L = np.linalg.norm(D - C, axis=1)
        tube = float(np.sum(2.0 * E * L + np.pi * E ** 2)) * sec.density_max(C)
        qerr = 1e-15 * abs(F)
        return dict(flux=F, quad_err=qerr, tube=tube, winding=int(w), lo=F - qerr - tube, hi=F + qerr + tube,
                    n_vertices=int(len(C)))
    res = []
    for ng in (3, 5):
        g, wq = np.polynomial.legendre.leggauss(ng)
        pts = np.concatenate([0.5 * (C + D) + 0.5 * gi * (D - C) for gi in g])
        A = np.asarray(sec.engine.evaluate(sec.xyz(pts), grad=False, A=True)["A"])
        AR = A[:, 0] * sec.c + A[:, 1] * sec.s
        n = len(C)
        tot = 0.0
        for j in range(ng):
            sl = slice(j * n, (j + 1) * n)
            tot += 0.5 * wq[j] * np.sum(AR[sl] * (D - C)[:, 0] + A[sl, 2] * (D - C)[:, 1])
        res.append(-sec.sign * tot)
    L = np.linalg.norm(D - C, axis=1)
    tube = float(np.sum(2.0 * E * L + np.pi * E ** 2)) * sec.density_max(C)
    F, qerr = float(res[1]), float(abs(res[1] - res[0]))
    return dict(flux=F, quad_err=qerr, tube=tube, winding=int(w), lo=F - qerr - tube, hi=F + qerr + tube,
                n_vertices=int(len(C)))


# ---------------------------------------------------------------------------- the other symmetry plane
def _half_map(sec, Y, forward=True):
    """The half-period map H from sec's plane phi_s to phi_s + pi / nfp (forward) or H^-1 back, with the
    accurate tracer (sec.exact for surrogates) at its steps per period and at twice that: returns the finer
    result, lost, and tau = 2 x its GBS estimate + 2e-13 R + |fine - coarse| per point."""
    if hasattr(sec, "half_map"):                    # model sections: their own half map
        return sec.half_map(Y, forward)
    ex = getattr(sec, "exact", sec)
    nfp = int(ex.source.nfp)
    half = np.pi / nfp
    phi0 = ex.phi_s if forward else ex.phi_s + half
    dphi = half if forward else -half
    Y = np.asarray(Y, float).reshape(-1, 2)
    outs = []
    for sp in (24, 48):
        tr = PlaneSection(ex.engine, ex.source, phi_s=ex.phi_s, sign=ex.sign, scale=ex.scale, reference=False,
                          steps_per_period=sp).tracer
        outs.append(tr.trace(Y, phi0, np.full(len(Y), dphi), nsteps=sp // 2))
    y, lost = outs[1]["y"], outs[1]["lost"] | outs[0]["lost"]
    tau = 2.0 * outs[1]["err"] + 2e-13 * np.maximum(1.0, np.abs(y[:, 0])) + np.linalg.norm(y - outs[0]["y"], axis=1)
    return y, lost, tau


def _reference_half(sec, x):
    """The independent reference tracer's half-period image of x (sec's plane -> the other symmetry plane):
    (point, error estimate) or None."""
    if hasattr(sec, "half_map"):                    # model sections: no independent reference
        y, lost, tau = sec.half_map(np.asarray(x, float)[None], True)
        return y[0], float(tau[0])
    try:
        from .reference import trace as RT
        ex = getattr(sec, "exact", sec)
        o = RT.poincare(ex.ref_field(), np.array([x[0]]), np.array([x[1]]), [ex.phi_s + np.pi / int(ex.source.nfp)], 1,
                        phi0=ex.phi_s, h=2 * np.pi / 512)
        return np.array([float(o["R"].ravel()[0]), float(o["Z"].ravel()[0])]), float(o["err"].ravel()[0])
    except Exception:  # noqa: BLE001
        return None


def _half_jacobian(sec, x, forward=True, hfd=None):
    """Central-difference Jacobian of the half-period map at x (first order, as the heteroclinic images')."""
    hfd = hfd or 1e-7 * sec.scale
    E = np.array([[0.0, 0.0], [hfd, 0.0], [-hfd, 0.0], [0.0, hfd], [0.0, -hfd]])
    y, lost, tau = _half_map(sec, np.asarray(x, float)[None] + E, forward)
    J = np.stack([(y[1] - y[2]) / (2 * hfd), (y[3] - y[4]) / (2 * hfd)], -1)
    return J, y[0], bool(np.any(lost)), float(tau[0])


def _newton_period_orbit(sec, x, q, scale, its=15):
    """Newton on the section's P^q from x with its monodromy (section_monodromy).  Returns the best
    dict(pts, M, closure, tau_q) or None."""
    h = 1e-6 * scale
    best = None
    x = np.asarray(x, float).copy()
    for it in range(its):
        M, y0, pts, tau, ls = section_monodromy(sec, x, q, h)
        if ls:
            break
        Fv = y0 - x
        clos = float(np.linalg.norm(Fv))
        if best is None or clos < best["closure"]:
            best = dict(pts=np.asarray(pts), M=M, closure=clos, tau_q=tau)
        if clos < 1e-14 * scale:
            break
        try:
            x = x - np.linalg.solve(M - np.eye(2), Fv)
        except np.linalg.LinAlgError:
            break
    return best


def _setup_transfer(sec, orbs, nodes, succ, other, q, scale):
    """The chain in the other symmetry plane: other() -> (sec1, a1, a1_e) (a BandSurrogate of the plane
    phi_s + pi / nfp and its axis point) or a reason; the X orbits there as H-images of orbs (Newton on
    sec1 from H(X^i_0), so that X1^i_k = H(X^i_k) index by index, checked), the S mirror table there and the
    chain order (must equal this plane's: H preserves the order along the chain).  Returns dict or a
    reason string."""
    got = other()
    if isinstance(got, str):
        return got
    sec1, a1, a1_e = got
    orbs1, Hs = [], []
    for i, o in enumerate(orbs):
        Hx, lost, tau = _half_map(sec, o.pts, True)
        if np.any(lost):
            return f"orbit {i}: half-period image lost"
        field = hasattr(getattr(sec1, "exact", sec1), "engine")
        if field:
            b = _newton_period_orbit(sec1, Hx[0], q, scale)
            if b is None or b["closure"] > 1e-11 * scale:
                return f"orbit {i}: no period-{q} orbit of the other plane's map near its half-period image"
            pts_b = b["pts"]
        else:                                          # model sections: Newton with central differences
            ob = BR.orbit_from_point(sec1, Hx[0], q, polish=8)
            if ob is None or ob.info["closure"] > 1e-11 * scale:
                return f"orbit {i}: no period-{q} orbit of the other plane's map near its half-period image"
            pts_b = ob.pts
        dev = float(np.max(np.linalg.norm(pts_b - Hx, axis=1)))
        if dev > 1e-6 * scale:
            return f"orbit {i}: the other plane's orbit is not its half-period image (deviation {dev:.2e})"
        o1 = _orbit_from_points(sec1, b["pts"], b["M"], b["closure"], b["tau_q"], q) if field else ob
        o1.p = o.p
        J, _, lj, _ = _half_jacobian(sec, o.pts[0], True)
        if lj:
            return f"orbit {i}: half-period Jacobian lost"
        e0, e1 = o.eig(), o1.eig()
        tol_l = max(1e-6, 20.0 * (float(o.info.get("dM") or 0.0) + float(o1.info.get("dM") or 0.0))) * abs(e0["lam"])
        if (e0["reflection"] != e1["reflection"]) or abs(abs(e0["lam"]) - abs(e1["lam"])) > tol_l:
            return f"orbit {i}: multipliers differ between the planes ({e0['lam']:.12f}, {e1['lam']:.12f})"
        Jv0 = J @ e0["vu"]
        orbs1.append(o1)
        Hs.append(dict(dev=dev, tau=float(np.max(tau)), stretch=float(np.linalg.norm(Jv0)),
                       sign=float(np.sign(np.dot(Jv0, e1["vu"])))))
    pts1 = np.array([orbs1[i].pts[k] for (i, k) in nodes])
    tol = 10.0 * max(o.e_X for o in orbs1) + 1e-9 * scale
    mirror1 = {}
    for n_, x in zip(nodes, pts1):
        dd = np.linalg.norm(pts1 - x * np.array([1.0, -1.0]), axis=1)
        m = int(np.argmin(dd))
        if dd[m] > tol:
            return f"the other plane's X orbits are not closed under its symmetry (mismatch {dd[m]:.2e})"
        mirror1[n_] = nodes[m]
    if hasattr(sec1, "ref"):
        th1, _, ok1 = sec1.ref.coords(pts1)
        if not np.all(ok1):
            return "the other plane's X points are outside its reference curve's normal coordinates"
    else:
        th1 = _angle(pts1, a1)
    order1 = [nodes[m] for m in np.argsort(th1)]
    pos1 = {n_: s_ for s_, n_ in enumerate(order1)}
    N = len(order1)
    succ1 = {n_: order1[(pos1[n_] + 1) % N] for n_ in nodes}
    if any(succ1[n_] != succ[n_] for n_ in nodes):
        return "the chain order differs between the symmetry planes"
    return dict(sec=sec1, a=a1, a_e=a1_e, orbs=orbs1, mirror=mirror1, half=Hs)


def band_from_orbits(sec, orbs, a, axis_e=0.0, deadline=None, verbose=False, t0=None, dmax=None, amax=0.2,
                     target=None, max_levels=None, max_pts=6000, other=None):
    """The manifold island band of a chain with several hyperbolic period-q orbits orbs (a list of
    torux.barrier.Orbit, closed under S; module docstring, 'several X orbits'): C_A, C_B fully certified,
    the mixed curves' fluxes removed from (F_lo, F_hi).  Needs sec.ref (a BandSurrogate) for the order along
    the chain.  Gap types with no gap straddled by Z = 0 here take their heteroclinic points from the other
    symmetry plane when other() gives its section (see _setup_transfer): branches grown there, certified
    crossings with its Z = 0, mapped back by the half-period map.  Returns dict(ok, interval, excluded (list
    of open intervals), mixed, curves, ...)."""
    t0 = t0 or time.time()
    out = dict(ok=False, timings={}, n_orbits=len(orbs))
    q = orbs[0].q
    scale = sec.scale
    nodes = [(i, k) for i in range(len(orbs)) for k in range(q)]
    pts = np.array([orbs[i].pts[k] for (i, k) in nodes])
    # mirror table (S maps the set of orbits onto itself)
    mirror = {}
    tol = 10.0 * max(o.e_X for o in orbs) + 1e-9 * scale
    for n_, x in zip(nodes, pts):
        dd = np.linalg.norm(pts - x * np.array([1.0, -1.0]), axis=1)
        m = int(np.argmin(dd))
        if dd[m] > tol:
            out["reason"] = f"the X orbits are not closed under the symmetry (mismatch {dd[m]:.2e})"
            return out
        mirror[n_] = nodes[m]
    # order along the chain: the reference curve's parameter of each X point (polar angle about a for
    # sections without one, e.g. the model maps)
    if hasattr(sec, "ref"):
        th, _, okc = sec.ref.coords(pts)
        if not np.all(okc):
            out["reason"] = "X points outside the reference curve's normal coordinates"
            return out
    else:
        th = _angle(pts, a)
    order = [nodes[m] for m in np.argsort(th)]
    pos = {n_: s_ for s_, n_ in enumerate(order)}
    N = len(order)
    succ = {n_: order[(pos[n_] + 1) % N] for n_ in nodes}
    pred = {n_: order[(pos[n_] - 1) % N] for n_ in nodes}
    types = sorted({(n_[0], succ[n_][0], (succ[n_][1] - n_[1]) % q) for n_ in nodes})
    out["chain_order"] = [list(n_) for n_ in order]
    out["gap_types"] = [list(t_) for t_ in types]
    needed = {(t_, c_) for t_ in types for c_ in "AB"}
    # a gap type's heteroclinic points come from Fix(S) = {Z = 0} of this plane only if some gap of that
    # type is straddled by it (its ends a mirror pair); the other reversors' fixed sets are not used
    straddled = {(n_[0], succ[n_][0], (succ[n_][1] - n_[1]) % q) for n_ in nodes if mirror[n_] == succ[n_]}
    missing = set(types) - straddled
    tr = None
    if missing:
        if other is None:
            out["reason"] = (f"gap types {sorted(missing)} have no gap straddled by Z = 0 in this plane (their "
                             f"heteroclinic points need the other symmetry plane)")
            return out
        tr = _setup_transfer(sec, orbs, nodes, succ, other, q, scale)
        if isinstance(tr, str):
            out["reason"] = f"gap types {sorted(missing)} need the other symmetry plane: {tr}"
            return out
        straddled1 = {(n_[0], succ[n_][0], (succ[n_][1] - n_[1]) % q) for n_ in nodes if tr["mirror"][n_] == succ[n_]}
        if missing - straddled1:
            out["reason"] = f"gap types {sorted(missing - straddled1)} are straddled by Z = 0 in neither symmetry plane"
            return out
        out["other_plane"] = dict(phi_s=float(getattr(tr["sec"], "phi_s", np.nan)), half_map=tr["half"],
                                  X_orbits=[dict(residue=o.residue, e_X=o.e_X, dM=o.info.get("dM")) for o in tr["orbs"]])
    needed1 = {(t_, c_) for t_ in missing for c_ in "AB"}
    # per orbit: eigen-directions, start segments, curves
    per = []
    curves = []
    kappa_a = 1e3 / scale
    dmax = dmax if dmax is not None else 1e-2 * scale
    target = target if target is not None else 2e-6 * scale
    for i, orb in enumerate(orbs):
        eig = orb.eig()
        lam = abs(eig["lam"])
        if eig["reflection"] or not lam > 1.0 + 1e-12:
            out["reason"] = f"orbit {i} not (regular) hyperbolic"
            return out
        th_err = eigen_error(orb.M, orb.info.get("dM", 1e-14))
        if not th_err < 0.05:
            out["reason"] = f"orbit {i}: unstable eigenvector not determined (angle error bound {th_err:.2e})"
            return out
        s0 = _start_length(orb.e_X, lam, scale,
                           0.25 * float(np.min([np.linalg.norm(pts[u] - pts[v]) for u in range(len(pts))
                                                for v in range(u)] or [scale])))
        n0 = 16 if lam > 1.1 else (6 if lam > 1.01 else 3)
        delta0 = 0.5 * kappa_a * (lam * s0) ** 2 + lam * s0 * th_err
        cs_ = [BR.GenCurve(f"{i}u{'+' if sg > 0 else '-'}", "u", orb.pts[0], eig["vu"], sg, lam, False, s0, n0, q)
               for sg in (+1, -1)]
        for c in cs_:
            c.orbit = i
        curves += cs_
        per.append(dict(lam=lam, s0=s0, delta0=delta0, th_err=th_err))
    # the other plane: curves of the orbits owning a transferred gap type, start segments the H-images of
    # this plane's (s0 x |DH v_u|), so that their levels match this plane's to first order
    curves1, per1 = [], {}
    if tr is not None:
        own = {t_[0] for (t_, c_) in needed1} | {t_[1] for (t_, c_) in needed1}
        for i in sorted(own):
            o1 = tr["orbs"][i]
            eig1 = o1.eig()
            lam1 = abs(eig1["lam"])
            th1 = eigen_error(o1.M, o1.info.get("dM", 1e-14))
            if not th1 < 0.05:
                out["reason"] = f"orbit {i} (other plane): unstable eigenvector not determined ({th1:.2e})"
                return out
            s01 = float(per[i]["s0"] * tr["half"][i]["stretch"])
            n0 = 16 if lam1 > 1.1 else (6 if lam1 > 1.01 else 3)
            d01 = 0.5 * kappa_a * (lam1 * s01) ** 2 + lam1 * s01 * th1
            for sg in (+1, -1):
                c1 = BR.GenCurve(f"{i}u{'+' if sg > 0 else '-'}@1", "u", o1.pts[0], eig1["vu"], sg, lam1, False,
                                 s01, n0, q)
                c1.orbit = i
                # the curve of this plane it is the H-image of
                c1.twin = next(c for c in curves if c.orbit == i and c.name.endswith("+" if sg * tr["half"][i]["sign"] > 0
                                                                                    else "-"))
                curves1.append(c1)
            per1[i] = dict(lam=lam1, s0=s01, delta0=d01, th_err=th1)
    if max_levels is None:
        lam_min = min(p_["lam"] for p_ in per)
        max_levels = int(min(400000, 3 * q * math.ceil(math.log(scale / min(p_["s0"] for p_ in per))
                                                        / math.log(lam_min)) + 6 * q))
    found = {}
    curve_of = {}
    found1, curve1_of = {}, {}
    status = None
    reach = 0.25 * float(np.min([np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i)]
                                or [scale]))
    lev_pred = max(predicted_levels(q, p_["lam"], p_["s0"], reach) for p_ in per)
    out["growth_plan"] = dict(predicted_levels=lev_pred, max_levels=max_levels)
    t_grow = time.time()
    while True:
        if deadline is not None and time.time() > deadline:
            status = "deadline during manifold growth"
            break
        lv_ = min(c.level() for c in curves)
        if len(found) < len(needed) and lv_ >= 4 * q and lv_ % (4 * q) < q:
            slow = _growth_too_slow(t_grow, lv_, lev_pred, deadline)
            if slow:
                status = slow
                break
        for _ in range(q):
            BR.grow_step(sec, curves, dmax, amax, max_pts=max_pts)
            if curves1:
                BR.grow_step(tr["sec"], curves1, dmax, amax, max_pts=max_pts)
        if min(c.level() for c in curves) >= q - 1:
            for c in curves:
                i = c.orbit
                for r in _crossings_multi(sec, orbs, i, c, mirror, succ, pred, per[i]["delta0"], a):
                    key = (r["type"], r["cls"])
                    if key in needed and key not in needed1 and key not in found:
                        found[key] = r
                        curve_of[key] = c
                        if verbose:
                            print(f"   {r['cls']} gap type {r['type']} ({c.name}): heteroclinic point at level "
                                  f"{r['level']} ({time.time() - t0:.0f}s)", flush=True)
            for c1 in curves1:
                i = c1.orbit
                if not c1.active:
                    continue
                for r in _crossings_multi(tr["sec"], tr["orbs"], i, c1, tr["mirror"], succ, pred, per1[i]["delta0"],
                                          tr["a"]):
                    key = (r["type"], r["cls"])
                    if key in needed1 and key not in found:
                        found1[key] = r
                        curve1_of[key] = c1
                        # this plane's record: the level (start segments matched), the curve (the twin)
                        found[key] = dict(r, transferred=True)
                        curve_of[key] = c1.twin
                        if verbose:
                            print(f"   {r['cls']} gap type {r['type']} ({c1.name}, other plane): heteroclinic point "
                                  f"at level {r['level']} ({time.time() - t0:.0f}s)", flush=True)
            for c1 in curves1:
                keys = [k_ for k_, cc in curve1_of.items() if cc is c1]
                if c1.active and keys and c1.level() >= max(found1[k_]["level"] for k_ in keys) + 2 * q:
                    d = found1[keys[0]]["dir"]
                    serve = {(t_, "A" if d > 0 else "B") for t_ in missing if (t_[0] if d > 0 else t_[1]) == c1.orbit}
                    if serve <= set(found1):
                        c1.active = False
                        c1.stop_reason = "done"
        for c in curves:
            keys = [k_ for k_, cc in curve_of.items() if cc is c]
            if keys and c.active and c.level() >= max(found[k_]["level"] for k_ in keys) + 2 * q + 3:
                # all the (type, class) pairs this curve can serve found?
                d = found[keys[0]]["dir"]
                serve = {(t_, "A" if d > 0 else "B") for t_ in types if (t_[0] if d > 0 else t_[1]) == c.orbit}
                if serve <= set(found):
                    c.active = False
                    c.stop_reason = "done"
        if needed <= set(found) and all(cc.level() >= found[k_]["level"] + 2 * q + 3 for k_, cc in curve_of.items()):
            break
        if max(c.level() for c in curves) >= max_levels:
            status = f"heteroclinic points missing for {sorted(needed - set(found))}"
            break
        stopped = [c for c in curves + curves1 if not c.active and c.stop_reason != "done"]
        if stopped:
            status = "growth stopped: " + ", ".join(f"{c.name}: {c.stop_reason}" for c in stopped)
            break
    out["timings"]["grow"] = time.time() - t0
    out["growth"] = {c.name: dict(levels=c.level(), points=int(sum(len(p_.pts) for p_ in c.pieces)),
                                  stop=c.stop_reason) for c in curves + curves1}
    if status is not None:
        out["reason"] = status
        return out
    hfd = 1e-7 * scale
    arcs = {}
    het_ref = dict(ok=True, samples=[])
    transfers = []
    for key in sorted(found1, key=str):
        # the other plane's crossing, chord-validated there and re-certified, then mapped back by H^-1:
        # h = H^-1(h1) with error |DH^-1| box1 / 2 + tau (first order), a heteroclinic point of this plane
        r1, c1 = found1[key], curve1_of[key]
        i1 = r1["orbit"]
        br1 = BR.branch_polyline(tr["orbs"][i1], c1, r1["owner"])
        iu1, _, _ = _nearest_on_polyline(br1["pts"], r1["h"])
        st1 = BR.refine_arc_segments(tr["sec"], c1, sorted(set(BR.branch_segments(br1, 1, min(iu1 + 4, len(br1["refs"]) - 1)))),
                                     target)
        if not st1["converged"]:
            out["reason"] = f"{key}: chord validation did not converge (other plane)"
            return out
        r2 = [r for r in _crossings_multi(tr["sec"], tr["orbs"], i1, c1, tr["mirror"], succ, pred, per1[i1]["delta0"],
                                          tr["a"], first_owner=r1["owner"]) if (r["type"], r["cls"]) == key]
        if not r2:
            out["reason"] = f"{key}: crossing lost after chord validation (other plane)"
            return out
        r1 = found1[key] = r2[0]
        Jb, hb, lb, tb = _half_jacobian(sec, r1["h"], False, hfd)
        if lb:
            out["reason"] = f"{key}: half-period image of the heteroclinic point lost"
            return out
        eb = float(np.linalg.norm(Jb, 2)) * 0.5 * r1["box_diam"] + tb
        # evidence: the reference tracer's half-period image of h must return to h1 (within the two maps'
        # errors: |DH| tau + the reference's own estimate)
        ref_half = _reference_half(sec, hb)
        dh = float(np.linalg.norm(ref_half[0] - r1["h"])) if ref_half is not None else np.inf
        lim_h = float(np.linalg.norm(np.linalg.inv(Jb), 2)) * tb + 10.0 * (ref_half[1] if ref_half is not None else 0.0) \
            + 1e-12 * scale
        transfers.append(dict(key=str(key), h_other=r1["h"].tolist(), box_other=float(r1["box_diam"]),
                              h=hb.tolist(), err=eb, tau_half=tb, DH_inv=float(np.linalg.norm(Jb, 2)),
                              level=int(r1["level"]), reference_diff=dh, reference_limit=lim_h,
                              reference_ok=bool(dh <= lim_h)))
        if not dh <= lim_h:
            out["reason"] = f"{key}: half-period map not confirmed by the reference tracer ({dh:.2e} > {lim_h:.2e})"
            out["other_plane"]["transfers"] = transfers
            return out
        found[key] = dict(r1, h=hb, box_diam=2.0 * eb, transferred=True)
    for key, rec in sorted(found.items(), key=lambda kv: str(kv[0])):
        c = curve_of[key]
        i, j = rec["orbit"], rec["p_orbit"]
        slack = 3 if rec.get("transferred") else 1
        # chord validation of c's branches up to the images (as in the single-orbit case)
        hj, ej, taus, lost, Jn = _heteroclinic_images(sec, rec["h"], rec["box_diam"], q, hfd)
        if lost:
            out["reason"] = f"{key}: heteroclinic orbit lost"
            return out
        segs = []
        for m in range(q):
            br = BR.branch_polyline(orbs[i], c, (rec["owner"] + m) % q)
            iu, _, _ = _nearest_on_polyline(br["pts"], hj[m])
            segs += BR.branch_segments(br, 1, min(iu + 4, len(br["refs"]) - 1))
            br2 = BR.branch_polyline(orbs[i], c, mirror[(j, (rec["partner"] + m) % q)][1])
            is_, _, _ = _nearest_on_polyline(br2["pts"], hj[m] * np.array([1.0, -1.0]))
            segs += BR.branch_segments(br2, 1, min(is_ + 4, len(br2["refs"]) - 1))
        st = BR.refine_arc_segments(sec, c, sorted(set(segs)), target)
        if not st["converged"]:
            out["reason"] = f"{key}: chord validation did not converge"
            return out
        if not rec.get("transferred"):
            r2 = [r for r in _crossings_multi(sec, orbs, i, c, mirror, succ, pred, per[i]["delta0"], a,
                                              first_owner=rec["owner"]) if (r["type"], r["cls"]) == key]
            if not r2:
                out["reason"] = f"{key}: crossing lost after chord validation"
                return out
            rec = found[key] = r2[0]
        hj, ej, taus, lost, Jn = _heteroclinic_images(sec, rec["h"], rec["box_diam"], q, hfd)
        if lost:
            out["reason"] = f"{key}: heteroclinic orbit lost"
            return out
        # the mirror of c: stable branches of orbit j (S maps X^i_0 to X^j_b)
        b = mirror[(i, 0)][1]
        cs = BR.mirror_curve(c, "S" + c.name, b, 0.0)
        cs.const_err = 1e-15 * scale + tol
        gaps, why = _gap_arcs(sec, orbs[i], orbs[j], rec["cls"], c, cs, rec, hj, ej, per[i]["delta0"],
                              per[i]["delta0"], slack=slack)
        if gaps is None:
            out["reason"] = f"{key}: {why}"
            return out
        arcs[key] = gaps
        if hasattr(sec, "reference_map") and getattr(sec, "use_reference", False):
            Y, Er = sec.reference_map(rec["h"], q - 1)
            dd = np.linalg.norm(Y[:, 0, :] - hj[1:], axis=1)
            trm = np.array([ej[m] - Jn[m] * 0.5 * rec["box_diam"] for m in range(1, q)])
            okh = bool(np.all(dd <= np.maximum(trm, 0) + 10 * Er[:, 0] + 1e-12 * scale))
            het_ref["samples"].append(dict(key=str(key), max_diff=float(np.max(dd)), ok=okh))
            het_ref["ok"] = bool(het_ref["ok"] and okh)
    corners = pts
    res_c = {}
    polys = {}
    for cl in "AB":
        C, E = _chain_gaps([arcs[(t_, cl)] for t_ in types], N)
        if C is None:
            out["reason"] = f"class {cl}: the gaps do not chain into one closed curve"
            return out
        cert, Cfin = _certify_curve(sec, C, E, a, axis_e, corners=corners)
        res_c[cl] = cert
        if not cert["ok"]:
            out["curves"] = res_c
            out["reason"] = f"class {cl}: " + cert.get("reason", "")
            return out
        polys[cl] = Cfin.tolist()
    lo = min(res_c["A"]["F_up"], res_c["B"]["F_up"])
    hi = max(res_c["A"]["F_low"], res_c["B"]["F_low"])
    mixed = []
    import itertools
    for sig in itertools.product("AB", repeat=len(types)):
        if len(set(sig)) == 1:
            continue
        C, E = _chain_gaps([arcs[(t_, s_)] for t_, s_ in zip(types, sig)], N)
        if C is None:
            out["reason"] = f"mixed curve {sig}: the gaps do not chain"
            return out
        mf = _loop_flux(sec, C, E, a, corners)
        mixed.append(dict(classes="".join(sig), **(mf or dict(winding=0))))
    out["curves"] = res_c
    out["mixed"] = mixed
    out["heteroclinic_reference"] = het_ref
    out["evidence"] = BR.reference_evidence(sec, orbs[0], curves, {})
    if tr is not None:
        out["other_plane"].update(transfers=transfers,
                                  evidence=BR.reference_evidence(tr["sec"], tr["orbs"][0], curves1, {}),
                                  closure_reference=[reference_closure_plane(getattr(tr["sec"], "exact", tr["sec"]), o)
                                                     for o in tr["orbs"]]
                                  if hasattr(getattr(tr["sec"], "exact", tr["sec"]), "reference_map") else [])
    out["polygons"] = polys
    out["timings"]["curves"] = time.time() - t0
    if not lo < hi:
        out["reason"] = "the separatrix curves enclose fluxes closer than their errors"
        return out
    # the excluded set: (lo, hi) minus the mixed curves' closed intervals
    cuts = sorted((m["lo"], m["hi"]) for m in mixed if "lo" in m)
    pieces = []
    cur = lo
    for (u_, v_) in cuts:
        if v_ <= cur:
            continue
        if u_ > cur:
            pieces.append((float(cur), float(min(u_, hi))))
        cur = max(cur, v_)
        if cur >= hi:
            break
    if cur < hi:
        pieces.append((float(cur), float(hi)))
    out["interval"] = (float(lo), float(hi))
    out["excluded"] = [p_ for p_ in pieces if p_[0] < p_[1]]
    out["ok"] = bool(out["excluded"])
    if not out["ok"]:
        out["reason"] = "the mixed curves' fluxes cover the band"
    return out


def island_band_certificate_pairs(engine, source, T1, T2, p, q, request, deadline=None, verbose=False, lines=None,
                                  reference=True, bracket_check=True, surrogate_opts=None, axis_inverse=False,
                                  axis_line=None):
    """The manifold island band of a chain whose X lines lie off the symmetry lines (module docstring,
    'several X orbits'): the hyperbolic period-q orbits in the annulus between T1 and T2 found by Newton on
    a BandSurrogate, closed under S; C_A, C_B certified; the mixed curves' fluxes removed.  Returns a
    JSON-serialisable dict like island_band_certificate_manifold's, kind="island_band_manifold_pairs",
    with "excluded" (the certified open intervals) and holds_request (the request in one of them).  axis_line: the
    analysis's period-one line D, as for island_band_certificate_manifold (checked in every plane used)."""
    t0 = time.time()
    req = None if request is None else abs(float(request))
    F1, F2 = abs(float(T1.flux)), abs(float(T2.flux))
    out = dict(kind="island_band_manifold_pairs", ok=False, request=req, F1=F1, F2=F2, holds_request=False,
               chain=dict(p=int(p), q=int(q)), timings={})

    def done(reason=None):
        if reason:
            out["reason"] = reason
        out["timings"]["total"] = time.time() - t0
        if verbose and reason:
            print(f"  manifold band (pairs): {reason}", flush=True)
        return BR._jsonable(out)

    try:
        nfp = int(source.nfp)
        if not getattr(source, "stellsym", False):
            return done("source not stellarator symmetric")
        if lines is None:
            from .chains import chain_lines
            iota_r = p / q * nfp
            base = T1 if abs(T1.iota - iota_r) <= abs(T2.iota - iota_r) else T2
            lines = chain_lines(engine, base, p, q, verbose=verbose)
        if lines and all(abs(l.residue) < MIN_RESIDUE for l in lines):
            return done(f"chain too weak for the manifold method: residues "
                        f"{[f'{l.residue:.2e}' for l in lines]} below {MIN_RESIDUE:.1e} in modulus")
        # the planes: those of the chain's non-hyperbolic symmetric lines (O, reflection-hyperbolic centres)
        # first, then the other (the theorem uses neither line: the planes only need Z = 0 to straddle gaps)
        Ol = [l for l in lines if l.residue > 0]
        planes = [int(round((float(l.phi0) % (2 * np.pi / nfp)) / (np.pi / nfp))) % 2 for l in Ol]
        planes = list(dict.fromkeys(planes + [0, 1]))
        out["planes_tried"] = []
        cache = {}

        def plane_setup(plane):
            """(sec, sec_exact, axis orbit, scale, info) of a symmetry plane, or a reason; built once."""
            if plane in cache:
                return cache[plane]
            phi_s = plane * np.pi / nfp
            P1, P2 = torus_section(T1, phi_s), torus_section(T2, phi_s)
            scale = float(np.max(P2[:, 0]) - np.min(P2[:, 0]))
            Rmid = float(0.5 * (np.max(P1[:, 0]) + np.min(P1[:, 0])))
            sign = _bphi_sign(engine, Rmid, phi_s)
            sec_exact = PlaneSection(engine, source, phi_s=phi_s, sign=sign, scale=scale, reference=reference)
            ax_id = {}
            with _few_threads():
                ax = axis_point(sec_exact, T1, plane, inverse=axis_inverse, axis_line=axis_line, info=ax_id)
            if axis_line is not None:
                out.setdefault("axis_identity", {})[str(plane)] = ax_id
            if ax is None:
                cache[plane] = ax_id.get("reason") or "axis point not found"
                return cache[plane]
            sec = BandSurrogate(_tlean(engine, source, T2, out), source, T1, T2, phi_s=phi_s, sign=sign, scale=scale,
                                reference=reference, **(surrogate_opts or {}))
            try:
                info = sec.build(verbose=verbose, deadline=deadline)
            except RuntimeError as e:
                cache[plane] = f"surrogate failed: {e}"
                return cache[plane]
            if not (info["tau_annulus"] < 1e-9 * scale and info["density_rel_err"] < 1e-6):
                cache[plane] = (f"surrogate not accurate enough (plane {plane}: tau {info['tau_annulus']:.2e}, check "
                                f"diff {info['max_check_diff']:.2e}, retraced {info.get('retraced')} of {info.get('nodes')} "
                                f"nodes, bar {1e-9 * scale:.1e})")
                return cache[plane]
            cache[plane] = (sec, sec_exact, ax, scale, info)
            return cache[plane]

        def other_of(plane):
            def other():
                got = plane_setup(1 - plane)
                if isinstance(got, str):
                    return got
                return got[0], got[2].pts[0], float(got[2].e_X)
            return other

        for plane in planes:
            phi_s = plane * np.pi / nfp
            rec_p = dict(phi_s=phi_s)
            out["planes_tried"].append(rec_p)
            got = plane_setup(plane)
            if isinstance(got, str):
                rec_p["reason"] = got
                continue
            sec, sec_exact, ax, scale, info = got
            rec_p["surrogate"] = info
            a = ax.pts[0]
            cand = find_period_orbits(sec, q, scale=scale)
            hyp = [o for o in cand if o["residue"] < -MIN_RESIDUE]
            rec_p["orbits"] = [dict(residue=o["residue"], point=o["pts"][0].tolist(), closure=o["closure"]) for o in cand]
            if verbose:
                print(f"  plane {phi_s:.4f}: period-{q} orbits residues {[round(o['residue'], 6) for o in cand]}",
                      flush=True)
            if not hyp:
                rec_p["reason"] = (f"no hyperbolic period-q orbit in the annulus with residue below -{MIN_RESIDUE:.1e}"
                                   f" ({len(cand)} period-q orbits found)")
                continue
            orbs = [_orbit_from_points(sec, o["pts"], o["M"], o["closure"], o["tau_q"], q) for o in hyp]
            for o in orbs:
                o.p = int(p)
            if any(o.e_X > 1e-3 * scale for o in orbs):
                rec_p["reason"] = "an X orbit not determined (e_X)"
                continue
            # every gap type must have a gap straddled by Z = 0 in this plane (a mirror pair of neighbours)
            res = band_from_orbits(sec, orbs, a, axis_e=float(ax.e_X), deadline=deadline, verbose=verbose, t0=t0,
                                   other=other_of(plane))
            rec_p.update({k: v for k, v in res.items() if k not in ("polygons",)})
            if not res["ok"]:
                continue
            lo, hi = res["interval"]
            out.update(phi_s=phi_s, axis_point=a.tolist(), interval=(lo, hi), excluded=res["excluded"],
                       mixed=res["mixed"], curves=res["curves"], X_orbits=[dict(residue=o.residue, e_X=o.e_X,
                       points=o.pts.tolist(), dM=o.info["dM"]) for o in orbs], chain_order=res.get("chain_order"),
                       gap_types=res.get("gap_types"), surrogate=info)
            tol_b = 1e-12 * F2
            out["bracket_check"] = bool(bracket_check)
            out["inside_bracket"] = bool(F1 - tol_b <= lo and hi <= F2 + tol_b)
            out["bracket_tori"] = _bracket_tori_record(T1, T2) if bracket_check else None
            if bracket_check and not out["inside_bracket"]:
                return done("band not inside the bracket: inconsistent")
            ev = dict(res.get("evidence", {}))
            ev["closure_reference"] = [reference_closure_plane(sec_exact, o) for o in orbs]
            ev["flux_reference"] = {k: BR.reference_flux(sec, np.array(v)) for k, v in res["polygons"].items()}
            ev["heteroclinic_reference"] = res.get("heteroclinic_reference")
            out["evidence"] = ev
            ok_ev = bool(ev.get("ok", True)) and all(c.get("ok", False) for c in ev["closure_reference"]) and \
                all(v.get("ok", False) for v in ev["flux_reference"].values()) and \
                (ev.get("heteroclinic_reference") or {}).get("ok", False)
            if res.get("other_plane"):
                op = res["other_plane"]
                out["other_plane"] = op
                ok_ev = ok_ev and bool((op.get("evidence") or {}).get("ok", False)) and \
                    all(c.get("ok", False) for c in op.get("closure_reference", []))
            out["evidence_ok"] = ok_ev
            if not ok_ev:
                return done("reference checks failed")
            out["ok"] = True
            out["rigorous"] = True
            if req is not None:
                out["holds_request"] = bool(any(u_ < req < v_ for (u_, v_) in res["excluded"]))
            return done()
        return done("no plane gave a certified band: " + "; ".join(str(r.get("reason")) for r in out["planes_tried"]))
    except Exception as e:  # noqa: BLE001
        import traceback
        out["traceback"] = traceback.format_exc()
        return done(f"error: {e!r}")


if _njit is not None:
    @_njit(cache=True)
    def _decimate_nb(P, E, capX, rel, abs_tol):
        """Greedy chord merging (decimate_closed): returns kept vertex indices (count k) and new errors."""
        n = P.shape[0] - 1
        keep = np.empty(n, np.int64)
        newE = np.empty(n)
        nk = 0
        keep[0] = 0
        nk = 1
        ne = 0
        i = 0

        def ok(i, j):
            if j - i < 2:
                return True, 0.0
            ax = P[i, 0]
            ay = P[i, 1]
            bx = P[j, 0] - ax
            by = P[j, 1] - ay
            L2 = bx * bx + by * by
            m = 0.0
            emin = 1e300
            cmin = 1e300
            for t in range(i + 1, j):
                px = P[t, 0] - ax
                py = P[t, 1] - ay
                s = (px * bx + py * by) / L2 if L2 > 0 else 0.0
                s = min(1.0, max(0.0, s))
                dx = px - s * bx
                dy = py - s * by
                d = np.sqrt(dx * dx + dy * dy)
                if d > m:
                    m = d
                if capX[t] < cmin:
                    cmin = capX[t]
            for t in range(i, j):
                if E[t] < emin:
                    emin = E[t]
            return m <= max(rel * emin, min(abs_tol, cmin)), m

        while i < n:
            step = 2
            best_j = i + 1
            best_d = 0.0
            while i + step <= n:
                g, m = ok(i, i + step)
                if not g:
                    break
                best_j = i + step
                best_d = m
                step *= 2
            lo = best_j
            hi = min(i + step, n + 1)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                g, m = ok(i, mid)
                if g:
                    lo = mid
                    best_d = m
                else:
                    hi = mid
            j = lo
            emax = 0.0
            for t in range(i, j):
                if E[t] > emax:
                    emax = E[t]
            newE[ne] = emax + best_d
            ne += 1
            if j < n:
                keep[nk] = j
                nk += 1
            i = j
        return keep[:nk], newE[:ne]
