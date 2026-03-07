# ExactPV2: Float-Free PV Pipeline via Resultants and Polynomial Remainder Sequences

## 1. Overview

ExactPV2 reformulates the combinatorial PV pipeline so that **all
topological decisions** use pure integer arithmetic on the
integer-coefficient polynomials P_k and Q.  No floating-point eigenvalue
approximations, no Sturm evaluation at float values, no Higham
certification, and no SoS delta-shift are needed.

The key tools are:
- **Sylvester resultant** Res(f, g): integer determinant, zero iff f and g
  share a root
- **Polynomial remainder sequence (PRS)**: pseudo-remainder chain that
  determines sign(g(alpha)) where f(alpha) = 0, in pure integer arithmetic
- **Polynomial GCD**: exact integer GCD via Euclidean algorithm on integer
  polynomials, for shared-root factoring

### Pipeline Summary

| Phase | ExactPV1 (current) | ExactPV2 (proposed) |
|-------|-------------------|---------------------|
| Per-triangle | Face poly + N_k (deg 4) + Sturm isolation + Higham + SoS | Face poly P_k (deg 3) only — just coefficients |
| Per-tet | Collect punctures; Q-interval via Sturm count at float lambda; sort by float lambda; pair | Collect P_k from faces; validity + ordering + pairing via PRS/resultant |
| Float usage | Root isolation, Sturm eval, lambda-sort | Spatial output coordinates only |
| Guarantee | Threshold-free | Float-free (for all topological decisions) |


## 2. Background: PRS and Root-Sign Determination

### 2.1 Pseudo-Remainder

For polynomials f (degree m) and g (degree n <= m) with integer
coefficients, the **pseudo-remainder** is:

    prem(f, g) = lc(g)^(m-n+1) * f  mod  g

where lc(g) is the leading coefficient of g.  The result has integer
coefficients (no division needed).  Key property:

    prem(f, g)(alpha) = lc(g)^(m-n+1) * f(alpha)    when g(alpha) = 0

so sign(f(alpha)) = sign(prem(f,g)(alpha)) * sign(lc(g)^(m-n+1)).

### 2.2 PRS for Sign Determination

To determine sign(g(alpha)) where f(alpha) = 0 (both integer cubics):

1. Compute r = prem(g, f)  — degree <= 2, integer coefficients
   (For equal-degree cubics: r = lc(f)*g - lc(g)*f, a quadratic with
   coefficients that are 2x2 cross-products of the original coefficients.)
2. sign(g(alpha)) = sign(r(alpha)) * sign(lc(f))
3. To determine sign(r(alpha)): compute s = prem(f, r) — degree <= 1
4. sign(r(alpha)) = sign(s(alpha)) * sign(lc(r)^...)
5. For linear s: sign(s(alpha)) determined by comparing alpha with -s[0]/s[1],
   which reduces to evaluating f(-s[0]/s[1]) — a rational evaluation, exact
   after clearing denominators.
6. For constant s: sign is immediate.

**Total: a finite chain of integer operations.  No floats anywhere.**

### 2.3 One-Root Comparison Formula

For two integer cubics f and g, each with exactly one real root (Delta < 0):

    alpha > beta  iff  sign(lc(f) * lc(g) * Res(f, g)) > 0

where Res(f, g) is the 6x6 Sylvester determinant.

**Proof:** Res(f,g) = lc(f)^3 * g(alpha) * g(alpha_2) * g(alpha_3).
Since alpha_2, alpha_3 are complex conjugates: g(alpha_2)*g(alpha_3) =
|g(alpha_2)|^2 > 0.  So sign(g(alpha)) = sign(Res(f,g)) * sign(lc(f)).
For a one-root cubic: alpha > beta iff sign(g(alpha)) = sign(lc(g)).
Combining: alpha > beta iff sign(lc(f) * lc(g) * Res(f,g)) > 0.


## 3. Per-Triangle: Face Polynomial Computation

Each triangular face has a characteristic polynomial P_k (integer cubic,
from the 3 vertex field values at the face opposite vertex k).  This is the
SAME polynomial as the tet-level P_k (paper line 376: "P_k reduces to the
Peikert-Roth characteristic polynomial D(lambda) on the opposite face").

**Per-triangle computation:**
- Quantize field values to integers (scale S = 2^20 or 2^16)
- Compute the cubic characteristic polynomial P_k in exact int128 arithmetic
- Compute discriminant sign -> root count (1 or 3)
- **Done.**  No root isolation, no N_k, no barycentric sign checks.

Each face polynomial is computed once and shared by two adjacent tets.


## 4. Per-Tet: Validity, Detection, and Pairing

### 4.1 Collect Polynomials

For each tet with 4 faces:
- Collect P_0, P_1, P_2, P_3 from the 4 face computations
- Compute Q = det(A - lambda*B) in int128, OR use Q = P_0 + P_1 + P_2 + P_3
  (from the identity sum(mu_k) = 1 => sum(P_k) = Q)

### 4.2 Validity Test for Each P_k Root

A root lambda* of P_k is a valid puncture on face k iff all other
barycentric coordinates are non-negative at lambda*:

    mu_j(lambda*) = P_j(lambda*) / Q(lambda*) >= 0   for all j != k

Since P_k(lambda*) = 0 and Q(lambda*) = sum_{j!=k} P_j(lambda*), this
requires all P_j(lambda*) (j != k) to have the same sign.

**Test via PRS:**

For each root lambda* of P_k (identified by index: 1st, 2nd, or 3rd
smallest root):
1. For each j != k: compute sign(P_j(lambda*)) using PRS of P_k and P_j
2. Valid puncture iff all three signs agree

**One-root case (Delta_{P_k} < 0):**
P_k has one real root alpha.
sign(P_j(alpha)) = sign(Res(P_k, P_j) * lc(P_k))
— a single 6x6 Sylvester determinant in int128.

**Three-root case (Delta_{P_k} > 0):**
P_k has three roots alpha_1 < alpha_2 < alpha_3.  For each root alpha_i,
sign(P_j(alpha_i)) is determined by the PRS chain:
1. r = prem(P_j, P_k) — quadratic, integer coefficients
2. The signs of r at the three roots of P_k are determined by:
   - If disc(r) < 0 (no real roots of r): r has constant sign, all three
     sign(r(alpha_i)) are the same = sign(lc(r))
   - If disc(r) >= 0: r has roots rho_1, rho_2.  Need to determine how
     P_k's roots interleave with r's roots.
3. Compute s = prem(P_k, r) — linear, integer coefficients
4. sign(r(alpha_i)) depends on position of alpha_i relative to r's roots,
   determined by sign(s(alpha_i)) and further integer tests.

Each step produces integer polynomials from the previous step's integer
coefficients.  The chain terminates in at most 3 steps for cubics.

### 4.3 Q-Interval Assignment

Instead of evaluating the Sturm sequence of Q at a float lambda, we
determine sign(Q(lambda*)) where P_k(lambda*) = 0 via PRS:

    sign(Q(lambda*)) = sign(Res(P_k, Q) * lc(P_k))   [one-root case]

For the three-root case with three Q-roots (Delta_Q > 0), we need the
full Sturm count of Q below lambda*.  This equals the number of Q-roots
less than lambda*, which is determined by the PRS of P_k and Q (giving
sign(Q(alpha_i)) for each root alpha_i of P_k) combined with the PRS of
P_k and Q' (giving sign(Q'(alpha_i))).

Alternatively: for each Q-root beta_j (root of Q, identified by index),
determine whether alpha_i < beta_j or alpha_i > beta_j via the PRS
comparison.  This gives the Q-interval of alpha_i directly.


## 5. Edge and Vertex Detection

### 5.1 Edge Punctures (D01)

A puncture lies on edge e (shared by faces i and j) iff P_i and P_j share
a root:

    D01 detected by:  Res(P_i, P_j) = 0

This is a single 6x6 Sylvester determinant — exact integer test.
Compare with ExactPV1: "check if N_k has a root in the isolating interval
via Sturm counting" (requires float interval endpoints + Sturm evaluation).

**Identifying the shared root:** If Res(P_i, P_j) = 0, the shared root
is a root of h = gcd(P_i, P_j), computable via the Euclidean algorithm
on integer polynomials.  deg(h) gives the number of shared roots.

### 5.2 Vertex Punctures (D00)

A puncture at vertex v_m iff P_i = P_j = P_l = 0 for {i,j,l} = {0,1,2,3}\{m}:

    D00 detected by:  Res(P_i, P_j) = 0  AND  Res(P_j, P_l) = 0  AND  Res(P_i, P_l) = 0

(Or: deg(gcd(P_i, P_j, P_l)) >= 1, where the three-way GCD is computed
by iterated pairwise GCD.)

### 5.3 Ownership

Same min-index rule as ExactPV1:
- Edge puncture: assigned to the face with smaller opposite-vertex index
- Vertex puncture: assigned to the face where v_m has the smallest global
  vertex index among the incident faces


## 6. Pass-Through Detection (Shared Roots of P_k and Q)

### 6.1 The Subtlety: Isolated vs Non-Isolated Shared Roots

This is the most delicate part.  A Q-root lambda* is a pass-through iff
the singularity mu_k = P_k/Q is removable.  This requires ALL FOUR P_k
to vanish at lambda* (paper Section 4):

- If only one P_k vanishes: the other P_j/Q still diverges -> genuine pole
- If three P_k vanish: the fourth is forced to zero (since sum(P_k) = Q = 0)
- So: pass-through iff at least 3 of the 4 P_k vanish at lambda*

### 6.2 Taxonomy of Shared Roots

**SR (isolated shared root):** deg(gcd(Q, P_k)) = 1 for some k.
Q and P_k share exactly one root.

**ISR (non-isolated shared root):** deg(gcd(Q, P_k)) >= 2 for some k.
Q and P_k share two or more roots.

### 6.3 Exact Detection via GCD

The GCD approach is MORE INFORMATIVE than the Sylvester resultant:

1. **Compute h = gcd(P_0, P_1, P_2, P_3)** via iterated Euclidean algorithm
   on integer polynomials.  This gives the polynomial whose roots are
   common to ALL four P_k.

2. **If deg(h) >= 1:** the roots of h are candidates for pass-throughs.
   Since sum(P_k) = Q, if all four P_k vanish at lambda*, then
   Q(lambda*) = 0 automatically.  So h divides Q as well.

3. **Factor out h:**
   - Q_red = Q / h  (reduced Q — genuine poles only)
   - P_k_red = P_k / h  (reduced numerators)
   - All divisions are exact (h divides each polynomial)

4. **After factoring:** the reduced rational functions P_k_red / Q_red have
   NO pass-throughs.  Every root of Q_red is a genuine pole.  The stitching
   uses Q_red for interval assignment.

### 6.4 Why GCD is Better Than Resultant

| Property | Res(P_k, Q) = 0 | gcd(P_0,...,P_3) |
|----------|-----------------|------------------|
| Tests | any P_k shares any root with Q | all P_k share a common root |
| Sufficiency | necessary, not sufficient | necessary AND sufficient |
| Information | boolean (yes/no) | the shared factor polynomial h |
| Localization | global (some root, don't know which) | exact (roots of h) |
| Action | conservative merge (all Q-intervals) | precise: factor out h, only h-roots are pass-throughs |

### 6.5 Handling Remaining Cases

After GCD factoring:
- **No common root (deg(h) = 0):** all Q-roots are genuine poles, no
  pass-throughs.  Use Q directly for interval assignment.
- **Common root exists (deg(h) >= 1):** factor out h.  Use Q_red = Q/h
  for interval assignment.  The h-roots are pass-throughs where the curve
  crosses smoothly.

**Edge case:** What if Res(P_k, Q) = 0 for some k, but gcd(P_0,...,P_3)
has degree 0?  This means P_k shares a root with Q, but NOT all four P_k
share that root.  At the shared root lambda*: P_k(lambda*) = 0 and
Q(lambda*) = 0, but some P_j(lambda*) != 0, so mu_j = P_j/Q -> infinity.
This is a GENUINE POLE, not a pass-through!

In ExactPV1, the conservative Res(P_k, Q) = 0 test would INCORRECTLY
classify this as a pass-through and merge intervals.  ExactPV2's GCD
approach correctly identifies it as a genuine pole.

**However:** The paper notes (line 2124) that this case may not occur in
practice: "if Q(lambda*) = 0 and any three P_k vanish at lambda*, the
fourth is forced to zero."  So the question is: can Res(P_k, Q) = 0
without all P_j vanishing?  Yes — P_k and Q can share a root where the
other P_j don't vanish.  The ΣP_k = Q identity only forces the SUM to
vanish, not individual terms.

**This is a theoretical improvement of ExactPV2 over ExactPV1.**


## 7. Puncture Ordering and Pairing

### 7.1 Comparing Roots Across Different P_k

To sort valid punctures by lambda, we need to compare roots of different
P_k polynomials.

**One-root case:** alpha (root of P_i) vs beta (root of P_j):

    alpha > beta  iff  sign(lc(P_i) * lc(P_j) * Res(P_i, P_j)) > 0

**Three-root case:** Use PRS chain as described in Section 2.2.

### 7.2 Pairing Algorithm

After sorting valid punctures by lambda and assigning Q-intervals (using
Q_red after GCD factoring):

1. Group punctures by Q_red-interval (same interval = same curve segment)
2. Within each group, pair consecutive punctures: (0,1), (2,3), ...
3. Each pair is an entry-exit through the tet

### 7.3 Alternative: Sign-Pattern Sweep

Instead of explicit sorting, sweep through all roots of P_0,...,P_3, Q_red
left-to-right, tracking the sign of each P_k:
- At lambda = -infinity: signs determined by leading coefficients
- At each P_k root: P_k flips sign
- At each Q_red root: genuine pole, curve exits tet
- "Inside" segments: all P_k have the same sign
- Valid punctures: P_k roots at inside/outside transitions
- Pairing: consecutive entry-exit pairs of each inside segment

This approach determines validity + ordering + pairing simultaneously.


## 8. Complete ExactPV2 Algorithm

```
Input: Tetrahedral mesh with PL fields v, w (integer or quantized)
Output: PV curves as ordered lists of puncture points

Phase 1: Per-Triangle (face polynomial computation)
  For each triangular face F_k:
    Quantize field values to int64 (scale S)
    Compute P_k(lambda) in int128 arithmetic  [cubic, 4 integer coefficients]
    Compute discriminant sign -> root count (1 or 3)

Phase 2: Per-Tet (validity, detection, pairing)
  For each tet T with faces F_0, F_1, F_2, F_3:
    Collect P_0, P_1, P_2, P_3 from Phase 1
    Compute Q = P_0 + P_1 + P_2 + P_3  [or via det(A - lambda*B)]

    // Pass-through factoring
    h = gcd(P_0, P_1, P_2, P_3)  [iterated Euclidean algorithm, int128]
    if deg(h) >= 1:
      Q_red = Q / h;  P_k_red = P_k / h  [exact division]
    else:
      Q_red = Q;  P_k_red = P_k

    // For each root of P_k: validity test
    For each face k with root(s) of P_k:
      For each root alpha_i of P_k (identified by index):
        For each j != k:
          sign_j = sign(P_j(alpha_i))  via PRS of P_k and P_j  [integer]
        Valid iff all sign_j agree AND sign_j = sign(Q(alpha_i))

    // Edge/vertex detection
    For each pair (i,j): if Res(P_i, P_j) = 0 -> edge puncture
    For triples: if all three Res = 0 -> vertex puncture
    Apply ownership rule

    // Ordering and pairing
    Sort valid punctures by lambda via PRS/resultant comparisons
    Assign Q_red-intervals via PRS of P_k and Q_red
    Pair consecutive punctures within each Q_red-interval

Phase 3: Curve Assembly
  Build adjacency graph from pairwise connections
  Trace curves (degree-1 nodes first for open, then degree-2 for closed)
```


## 9. Comparison: ExactPV1 vs ExactPV2

| Property | ExactPV1 | ExactPV2 |
|----------|----------|----------|
| Scope | per-face extraction + per-tet stitching | per-face polynomial + per-tet analysis |
| Face computation | P_k poly + N_k (deg 4) + Sturm + Higham | P_k poly only (deg 3) |
| Polynomial degrees | N_k, D: degree 4; Q, P_k: degree 3 | P_k, Q: degree 3 only |
| Float arithmetic | root isolation, Sturm eval, lambda-sort | spatial output only |
| Sign certification | Higham bounds + SoS delta-shift | PRS/resultant (exact integer) |
| Edge detection (D01) | N_k root count in isolating interval | Res(P_i, P_j) = 0 |
| Vertex detection (D00) | two N_k root counts + ownership rule | three pairwise Res = 0 |
| Pass-through (SR) | Res(P_k, Q) = 0 (necessary, not sufficient) | gcd(P_0,...,P_3) (necessary AND sufficient) |
| Pass-through action | conservative merge (all Q-intervals) | precise factoring (Q_red = Q/h) |
| Theoretical guarantee | threshold-free | float-free for all topological decisions |
| Practical speed | O(1) float eval per puncture | O(d^2) integer PRS per comparison |
| Background needed | Sturm sequences, Higham bounds, SoS | Resultants, PRS, polynomial GCD |


## 10. Open Questions and Risks

1. **PRS for three-root cubics:** The one-root case is clean (single
   resultant).  The three-root case requires a PRS chain with multiple
   integer operations.  Need to verify overflow bounds for int128 at each
   PRS step.

2. **GCD computation overflow:** The Euclidean algorithm on int128
   polynomials involves pseudo-remainders whose coefficients can grow.
   Content reduction (dividing by GCD of coefficients) at each step is
   needed, same as for Sturm sequences.

3. **Q = sum(P_k) vs Q = det(A - lambda*B):** Are these always identical?
   Mathematically yes (from mu_k = P_k/Q and sum(mu_k) = 1).  Need to
   verify that the int128 computation gives the same result (potential
   for different overflow behavior).

4. **Face polynomial = tet P_k?** Paper states P_k "reduces to" the face
   characteristic polynomial (line 376).  Need to verify this is exact
   equality (same integer coefficients), not just "same roots up to a
   scalar."  If they differ by a constant factor, the PRS/resultant
   approach still works (roots are the same).

5. **ISR handling precision:** The GCD approach correctly factors out the
   common factor h.  But if h has degree 2 (ISR with two shared roots),
   Q_red = Q/h is linear — need to handle degree reduction gracefully.

6. **Practical validation:** Need to verify that ExactPV2 produces the
   same topological output as ExactPV1 on all 6 test fields (F1-F6)
   before it can replace the current methodology.
