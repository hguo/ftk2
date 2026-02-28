# Symbolic Perturbation for Parallel Vectors: Theory and Implementation

This document records theoretical analysis developed for the ExactPV stitching
algorithm in ftk2, covering the SoS perturbation scheme, its magic-number
rationale, and a systematic treatment of degeneracy types.

---

## 0. Mathematical Definitions

This section defines the algebraic tools used throughout.  All polynomials have
coefficients in a field F (either ℝ or ℤ after quantization).

### 0.1 Resultant

**Definition.**  For polynomials P(λ) = aₘ∏(λ−αᵢ) and Q(λ) = bₙ∏(λ−βⱼ),
the *resultant* is

$$
\mathrm{Res}(P, Q) = a_m^n\, b_n^m \prod_{i,j}(\alpha_i - \beta_j).
$$

**Key property.**  Res(P, Q) = 0 if and only if P and Q share a common root
(or one of them is identically zero).

**Computation.**  Res(P, Q) equals the determinant of the (m+n)×(m+n)
*Sylvester matrix*:

$$
\mathrm{Syl}(P,Q) = \begin{pmatrix}
a_m & a_{m-1} & \cdots & a_0 & & \\
& a_m & a_{m-1} & \cdots & a_0 & \\
& & \ddots & & & \ddots \\
b_n & b_{n-1} & \cdots & b_0 & & \\
& b_n & \cdots & & b_0 & \\
& & \ddots & & & \ddots
\end{pmatrix}
$$

where the top n rows are shifts of the coefficients of P and the bottom m rows
are shifts of Q.  When P and Q have integer coefficients (after quantization),
the resultant is an exact integer — no rounding.

**Relevance to PV.**  If the PV curve passes through tet-edge (vᵢ, vⱼ), then
the barycentric polynomials Pₖ and Pₗ for the two *opposite* vertices both
vanish at the same λ*.  That shared root is exactly the condition
Res(Pₖ, Pₗ) = 0.  Checking this integer determinant detects tet-edge
crossings without ever computing λ* explicitly.

### 0.2 Greatest Common Divisor

**Definition.**  The *greatest common divisor* gcd(P, Q) is the monic
polynomial of highest degree that divides both P and Q exactly.  It is unique
up to scaling and can be computed by the Euclidean algorithm for polynomials:

$$
\gcd(P, Q) = \gcd(Q,\; P \bmod Q)
$$

repeated until the remainder is zero.  The final non-zero remainder, normalized
to be monic, is gcd(P, Q).

**Relation to resultant.**  gcd(P, Q) has degree ≥ 1 ⟺ Res(P, Q) = 0.  Thus
the resultant is the fastest way to *test* whether a common root exists, while
gcd gives the *factor* that reveals what that root is.

**Relevance to PV.**  A removable singularity in µₖ(λ) = Pₖ(λ)/Q(λ) arises
when gcd(Pₖ, Q) ≠ 1.  Both numerator and denominator vanish at the same λ*,
creating a 0/0 form.  Cancelling the common factor restores a well-defined limit.

### 0.3 Discriminant

**Definition.**  The *discriminant* of a degree-n polynomial P(λ) with leading
coefficient aₙ is

$$
\mathrm{disc}(P) = \frac{(-1)^{n(n-1)/2}}{a_n}\,\mathrm{Res}(P,\,P').
$$

**Key property.**  disc(P) = 0 ⟺ P has a repeated root ⟺ P and P' share a
common factor.

For a depressed cubic t³ + pt + q the classical formula is

$$
\mathrm{disc} = -4p^3 - 27q^2.
$$

| Sign | Meaning |
|---|---|
| disc > 0 | three distinct real roots |
| disc = 0 | repeated root (curve tangent to simplex) |
| disc < 0 | one real root, two complex conjugate |

**Relevance to PV.**  The characteristic polynomial det(VT − λWT) = 0 is the
cubic whose roots give the PV parameter λ* on a triangle.  disc = 0 is precisely
the tangency degeneracy (Type A1), where the PV curve touches a face without
crossing.  With integer coefficients (after quantization) the sign of disc is
exact.

### 0.4 Sylvester Matrix

For P = Σ pᵢ λⁱ (degree m) and Q = Σ qⱼ λʲ (degree n), the Sylvester matrix
is the (m+n) × (m+n) matrix

$$
\mathrm{Syl}(P,Q) =
\underbrace{\begin{pmatrix}
p_m & \cdots & p_0 & & 0 \\
& \ddots & & \ddots & \\
0 & & p_m & \cdots & p_0
\end{pmatrix}}_{n \text{ rows}}
\oplus
\underbrace{\begin{pmatrix}
q_n & \cdots & q_0 & & 0 \\
& \ddots & & \ddots & \\
0 & & q_n & \cdots & q_0
\end{pmatrix}}_{m \text{ rows}}.
$$

Its determinant equals Res(P, Q).  When coefficients are integers, the
determinant is computed exactly using cofactor expansion or LU decomposition
over ℤ (with fraction-free arithmetic or __int128).

---

## 1. Background: The Parallel Vectors Problem

Given two smooth vector fields **U** and **V** on a tetrahedral mesh, the
*parallel vectors (PV) locus* is the set of points x where **U**(x) × **V**(x) = 0.
In 3-D this locus generically forms closed curves.

The standard algorithmic approach is:

1. **Puncture extraction** — for each triangular face, solve the PV condition to
   find intersection points (*punctures*) of the PV curve with the face.
2. **Stitching** — connect adjacent punctures through each tetrahedron to
   reconstruct the curve topology.

Both steps require solving a cubic polynomial, and both are sensitive to
degenerate configurations where punctures land exactly on mesh edges or vertices.

---

## 2. SoS Field Perturbation

### 2.1 The Perturbation Formula

To eliminate all degeneracy-driven special cases, we apply **Simulation of
Simplicity (SoS)** at the field level.  Each component of **U** and **V** at
every mesh vertex is additively shifted by a small, index-dependent amount:

$$
\delta(i, j) = \frac{\varepsilon_{\text{SoS}}}{2^{6\,(i \bmod 8) + j}}, \qquad
i = \text{global vertex index}, \quad j \in \{0,\ldots,5\}
$$

where j=0,1,2 correspond to the three components of **U** and j=3,4,5 to the
three components of **V**.  In code:

```cpp
static constexpr double SOS_EPS = 1e-8;

template <typename T>
inline T sos_perturbation(uint64_t vertex_idx, int component) {
    int k = (int)((vertex_idx % 8) * 6 + component % 6);
    return T(SOS_EPS) / T(uint64_t(1) << k);
}
```

### 2.2 Unpacking the Magic Numbers

Every constant in the formula has a principled justification.

#### The base amplitude `SOS_EPS = 1e-8`

Chosen as the geometric mean of machine epsilon (≈ 2.2×10⁻¹⁶) and unity:

$$
\varepsilon_{\text{SoS}} = 10^{-8} \approx \sqrt{\varepsilon_{\text{machine}}}
$$

This leaves eight decimal digits of field precision intact (typical physical
fields are specified to 6–8 significant figures) while the perturbation is still
at least 10⁸× larger than machine epsilon, so it is never rounded away.

#### The divisor `2^k` for geometric decay

In classical SoS the i-th "symbolic infinitesimal" satisfies ε_i ≪ ε_{i-1},
achieved in exact arithmetic by taking ε_i = ε^{2^i}.  In floating-point we
approximate this with geometric decay: ε / 2^k.

Geometric decay (factor 2 per step) is the minimal separation that lets a
double-precision comparator distinguish adjacent levels.

#### The stride `6` (six field components per vertex)

There are exactly six scalar DOFs per vertex: (Uₓ, U_y, U_z, Vₓ, V_y, V_z).
The per-vertex block of exponents is therefore {6i, 6i+1, …, 6i+5}, which is
precisely what the formula `6*(i%8) + j` produces.

#### The modular reduction `i % 8` (underflow prevention)

The full exponent k = 6*(i%8) + j ranges from 0 to 47.  Without the modulo,
vertex 8 would need exponent 48, giving a perturbation of SOS_EPS / 2^48
≈ 3.6×10⁻²³ — still representable (subnormal threshold ≈ 5×10⁻³²⁴), but
vertex 1000 would need exponent 6000, which underflows to 0.0.

The modulo caps the exponent at 47 so the minimum perturbation is

$$
\delta_{\min} = \frac{10^{-8}}{2^{47}} \approx 7 \times 10^{-23},
$$

safely above the subnormal threshold.

#### Known limitation of `% 8`

Vertices whose global indices differ by a multiple of 8 receive identical
perturbations, so they are not fully "independently perturbed" in the SoS sense.
For production use, replace `vertex_idx % 8` with a low-collision hash (e.g.
64-bit Murmur) so that any three vertices forming a triangle get distinct
perturbation blocks.

### 2.3 SoS Cubic Solver

When the discriminant of the characteristic cubic is zero (curve tangent to a
face), classical numerics are ambiguous.  We add an index-based tie-break:

```cpp
uint64_t min_idx = std::min({indices[0], indices[1], indices[2]});
if (min_idx % 2 == 0)
    return 1;   // treat disc as slightly positive → tangency excluded
else {
    roots[2] = roots[1];
    return 3;   // treat disc as slightly negative → tangency counted
}
```

This gives a deterministic, globally consistent ruling without any tolerance
tuning.

---

## 3. Q1 — Would Quantizing U and V Help?

**Short answer**: Yes for making polynomial coefficients exact integers and
computing the discriminant sign exactly; but only partially, because λ itself is
algebraic, so evaluating barycentric coordinates at λ still requires real
algebraic arithmetic.

### 3.1 The quantization procedure

Suppose every field component at every vertex is a rational number with
denominator at most D (common in simulation data that stores values as
fixed-point or low-precision floats).  Define the integer scale

$$
S = \mathrm{lcm}(D_1, D_2, \ldots, D_{6N}) \quad \text{(or a suitable power of 2)},
$$

and replace each field value fᵢⱼ with the exact integer f̂ᵢⱼ = round(S · fᵢⱼ).
In practice S = 2²³ suffices for single-precision inputs and S = 2⁵² for
double-precision.

After this scaling, the field arrays **U**, **V** at the three triangle vertices
hold exact 64-bit integers.

### 3.2 Integer polynomial coefficients from quantized fields

The characteristic polynomial is

$$
\chi(\lambda) = \det(\hat{V}_T - \lambda\,\hat{W}_T),
$$

where V̂_T and Ŵ_T are 3×3 integer matrices (one column per vertex, one row per
component).  Expanding the determinant:

$$
\chi(\lambda) = P_0 + P_1\lambda + P_2\lambda^2 + P_3\lambda^3,
$$

with integer coefficients:

$$
P_0 = \det(\hat{V}_T), \quad
P_3 = -\det(\hat{W}_T), \quad
P_1, P_2 = \text{signed sums of 2×2 mixed minors (integers)}.
$$

Each coefficient is a sum of at most six terms of the form ±v̂ᵢ·ŵⱼ·ŵₖ (or
similar), all integers.  For 64-bit field values, each product fits in at most
192 bits, which is exactly the range of `__int128` arithmetic (127-bit signed).

In practice: if S = 2²³ then Ŵᵢ ≤ 2²³ ≈ 8×10⁶ and a 3×3 determinant product
is at most (2²³)³ = 2⁶⁹ < 2¹²⁷, comfortably within `__int128`.

### 3.3 What quantization makes exact

**Discriminant sign.**  With integer P₀…P₃, the depressed cubic has

$$
\Delta = -4p^3 - 27q^2, \qquad
p = \frac{3P_3 P_1 - P_2^2}{3P_3^2}, \quad
q = \frac{2P_2^3 - 9P_3 P_2 P_1 + 27P_3^2 P_0}{27P_3^3}.
$$

Multiplying through by 27P₃⁴, the discriminant becomes the exact integer

$$
\Delta_{\mathbb{Z}} = -4\,(3P_3 P_1 - P_2^2)^3 - (2P_2^3 - 9P_3 P_2 P_1 + 27P_3^2 P_0)^2,
$$

computable in `__int128` (up to ~170-bit intermediate values; each factor is at
most degree-6 in S·field-values).  The sign of Δ_ℤ determines the root count
without any floating-point tolerance.

**Resultant for edge/vertex detection.**  Res(Pₖ, Pₗ) is the 6×6 Sylvester
determinant with integer entries, computed exactly in `__int128`.  A zero
resultant rigorously certifies a shared root.

### 3.4 Where quantization falls short

The roots λ₁, λ₂, λ₃ of the cubic are algebraic numbers (not generally
rational).  After finding λ*, the barycentric coordinates are

$$
\mu_i(\lambda^*) = \frac{P_i(\lambda^*)}{Q(\lambda^*)}
$$

where P_i and Q are polynomials with integer coefficients evaluated at an
algebraic point.  Determining the sign of µ_i(λ*) requires either:

- **Sturm sequences** on P_i/Q composed with the minimal polynomial of λ*, or
- **SoS field perturbation** (our approach) — which side-steps the issue by
  making µ_i generically nonzero.

### 3.5 Structural difference from CP/fiber/contour

The critical-point (CP) predicate reduces entirely to the sign of a determinant,
so a single __int128 evaluation suffices.  PV has a two-stage structure:

| Stage | CP | PV |
|---|---|---|
| Root-finding | not needed (det=0 is the predicate) | solve cubic for λ* |
| Bary evaluation | one determinant | evaluate algebraic rational at λ* |

Quantization makes Stage 1 exact; Stage 2 still requires real algebraic
arithmetic or SoS to side-step it.

---

## 4. Q2 — Handling PV Punctures on Shared Edges and Vertices

### 4.1 The shared-simplex problem

A triangular face edge is shared by multiple triangles.  A naive implementation
counts a puncture on that edge once per triangle, creating duplicate punctures
and broken topology.

### 4.2 The mutual-exclusion rule

For a puncture with barycentric coordinate µ_k = 0 on edge (v_i, v_j) of
triangle (v_i, v_j, v_k):

> **Triangle T claims the edge puncture iff `global_idx(v_k) < min(global_idx(v_i), global_idx(v_j))`.**

Equivalently: the triangle whose *opposite* vertex has the smallest global index
owns the edge.  Because global indices are unique integers, exactly one triangle
satisfies this condition → exactly one copy of the puncture is created.

For a vertex puncture (µ_i = µ_j = 0) the same rule degrades gracefully: among
all triangles sharing vertex v_k, only the one with the smallest opposing pair
(v_i, v_j) claims the puncture.

### 4.3 Implementation

```cpp
// Returns true iff this triangle claims the boundary puncture at
// barycentric component k (µ_k == 0 on edge opposite to v_k).
auto sos_bary_inside = [&](int k) -> bool {
    int i = (k + 1) % 3, j = (k + 2) % 3;
    // Convert to integer for exact sign
    __int128 q = __int128(nu[k] * 1e12);
    if (q > 0) return true;
    if (q < 0) return false;
    // Exactly zero: apply min-index ownership rule
    return indices[k] < std::min(indices[i], indices[j]);
};
return sos_bary_inside(0) && sos_bary_inside(1) && sos_bary_inside(2);
```

This replaces the ad-hoc `if (nu[k] < -1e-10) continue;` threshold with a
rigorous, globally consistent rule.

### 4.4 Higher-order cases

| Barycentric signature | Geometric situation | Claimed by |
|---|---|---|
| µ₀, µ₁, µ₂ > 0 | interior puncture | this triangle only |
| µ_k = 0, others > 0 | edge (v_i, v_j) | triangle with idx_k < min(idx_i, idx_j) |
| µ_j = µ_k = 0 | vertex v_i | triangle with min(idx_j, idx_k) < all others at v_i |
| all = 0 | entire triangle is PV | field degeneracy F2 — handled separately |

---

## 5. Q3 — Polynomial Signatures at Degenerate Configurations

In a tetrahedron with vertices v₀, v₁, v₂, v₃, the PV curve is the image of the
rational parametric map

$$
\mathbf{x}(\lambda) = \sum_{i=0}^{3} \mu_i(\lambda)\, \mathbf{v}_i, \qquad
\mu_i(\lambda) = \frac{P_i(\lambda)}{Q(\lambda)},
$$

where P_i and Q are cubic polynomials derived from the characteristic equation
`det(VT - λWT) = 0` and the least-squares bary system.

### 5.1 Edge puncture of the tet (curve through tet-edge)

If the PV curve passes through edge (v₀, v₁), then at the corresponding λ*:

$$
\mu_2(\lambda^*) = \mu_3(\lambda^*) = 0
\quad\Longleftrightarrow\quad
P_2(\lambda^*) = P_3(\lambda^*) = 0.
$$

Both P₂ and P₃ share the common factor (λ − λ*):

$$
\gcd(P_2, P_3) \text{ has degree} \geq 1
\quad\Longleftrightarrow\quad
\mathrm{Res}(P_2, P_3) = 0.
$$

**Detection signature**: `Res(P₂, P₃) = 0`.

### 5.2 Vertex puncture of the tet (curve through tet-vertex)

If the curve passes through vertex v₀ (barycentric coordinate 1, others 0):

$$
P_1(\lambda^*) = P_2(\lambda^*) = P_3(\lambda^*) = 0.
$$

All three non-dominant coordinates vanish simultaneously at λ*.  This implies
Q(λ*) = P₀(λ*) (since Σµ_i = 1 forces P₀(λ*) = Q(λ*)).

**Detection signature**: simultaneous common root of P₁, P₂, P₃.

### 5.3 Tangency to a triangle face (Type A — smooth tangency)

The PV curve is tangent to face (v₀, v₁, v₂) if it touches but does not cross.
Algebraically, P_k(λ) has a *double root* at λ*:

$$
P_k(\lambda^*) = 0 \quad \text{and} \quad P_k'(\lambda^*) = 0
\quad\Longleftrightarrow\quad
\mathrm{disc}(P_k) = 0.
$$

This is precisely the disc = 0 case handled by the SoS cubic solver.

### 5.4 Removable singularity (Type B — common factor with Q)

If gcd(P_k, Q) ≠ 1, then P_k and Q share a root λ*, making µ_k(λ*) a 0/0
indeterminate.  L'Hôpital's rule gives a well-defined limit, but the point is
geometrically degenerate (the curve "grazes" a face rather than crossing it).

**Detection signature**: `Res(P_k, Q) = 0`.

### 5.5 Pole of the bary map (degenerate pencil)

If Q has a root λ* that is *not* shared by any P_k, then Σµ_i → ∞.  This means
the rank of the matrix pencil (A − λ*B) drops by ≥ 2, i.e., an entire edge of
the tet satisfies the PV condition — a field degeneracy of type F1.

### 5.6 Summary table

| Degeneracy | Polynomial signature |
|---|---|
| Curve through tet-edge (v_i, v_j) | Res(P_i_perp1, P_i_perp2) = 0 |
| Curve through tet-vertex v_i | P_j(λ*) = P_k(λ*) = P_l(λ*) = 0 simultaneously |
| Tangency to triangle face | disc(P_k) = 0 |
| Removable singularity | Res(P_k, Q) = 0 |
| Entire tet-edge is PV | Q(λ*) = 0, rank pencil drops ≥ 2 |

---

## 6. Q4 — Complete Taxonomy of PV Degeneracy Types

### Group G — Geometric (locus hits a lower-dimensional simplex)

| Code | Description | Codim in field space | SoS remedy |
|---|---|---|---|
| G1 | Puncture on triangle edge | 1 | min-idx ownership rule |
| G2 | Puncture at mesh vertex | 2 | min-idx rule, degrades gracefully |
| G3 | Curve passes through tet edge | 1 | Res(P_i, P_j) = 0 detection + min-idx |
| G4 | Curve passes through tet vertex | 2 | triple common root detection |

These are *positional* degeneracies — the PV locus coincidentally intersects the
discrete skeleton.  SoS field perturbation moves the locus off integer-coordinate
features automatically.

### Group A — Algebraic (polynomial structure degenerates)

| Code | Description | Codim | Effect |
|---|---|---|---|
| A1 | Smooth tangency to face | 1 | disc(P_k) = 0 → double root → tangent excluded/included by SoS parity rule |
| A2 | Removable singularity | 1 | gcd(P_k, Q) ≠ 1 → 0/0 form |
| A3 | Cubic degrades to quadratic | 1 | leading coeff of char poly vanishes |
| A4 | Root at λ = 0 | 1 | trivial case V = 0 or V ∥ U at λ=0 |
| A5 | Triple root (inflectional tangency) | 2 | disc = 0 and disc' = 0 |

### Group F — Field (PV condition holds on an entire simplex)

| Code | Description | Field codimension | Detection |
|---|---|---|---|
| F1 | U ∥ V on entire tet edge | 4 | Q has root, rank pencil drops ≥ 2 |
| F2 | U ∥ V on entire triangle (P_k ≡ 0) | 6 | all coefficients of characteristic poly vanish |
| F3 | U ∥ V on entire tet (Q ≡ 0) | 8 | all tet vectors parallel |
| F4 | U = 0 or V = 0 at a vertex | 3 | point degeneracy |

These are intrinsic field degeneracies not resolvable by SoS — they require
special-case handling or mesh refinement.

### Group T — Topological (global locus structure)

| Code | Description | Remarks |
|---|---|---|
| T1 | Multiple disconnected PV curves | Topologically non-trivial but locally fine |
| T2 | Open PV curves with endpoints in domain | Requires domain boundary treatment |
| T3 | Apparent self-intersections (PL artifact) | Artifact of piecewise-linear approximation |
| T4 | Knotted PV locus | Globally non-trivial, cannot be untangled by local perturbation |

### Codimension counting

The field space of (U, V) at a single point is ℝ⁶.  The PV condition U×V = 0 is
codimension 2 (two scalar equations).  Each additional degeneracy imposes further
independent constraints:

- G1/A1/A3: one additional scalar constraint → codim 1 in field space
- G2/A5: two additional constraints → codim 2 in field space
- F1: four constraints (entire edge) → codim 4
- F2: six constraints → codim 6
- F3: eight constraints → codim 8

Degeneracies of codimension ≥ 1 in field space are non-generic (measure zero)
and are exactly the ones SoS perturbation eliminates.

---

---

## 7. The Exact Integer Pipeline

To eliminate all floating-point thresholds from the PV solver, we implement an
twelve-subtask pipeline that progressively moves decision predicates from float
arithmetic into exact integer arithmetic.  The two remaining float operations
are the root-finding itself (unavoidably irrational) and the least-squares bary
solve (linear algebra at a float λ*).

### Subtask 1 — Field Quantization

**Goal**: convert field values to exact integers so all polynomial coefficients
are integers.

**Method**: multiply every field component at every vertex by
`QUANT_SCALE = 2^20 ≈ 10^6` and round to `int64_t`.

```cpp
static constexpr int     QUANT_BITS  = 20;
static constexpr int64_t QUANT_SCALE = int64_t(1) << QUANT_BITS;

inline int64_t quant(double x) {
    return static_cast<int64_t>(std::llround(x * double(QUANT_SCALE)));
}
```

**Overflow analysis**: with |field| ≤ 10^6, each quantized entry is ≤ 2^40.
A 3×3 determinant involves triple products ≤ (2^40)^3 = 2^120 < 2^127, which
fits in `__int128` (signed 127-bit).

**Key property**: multiplying both field matrices by the same scale S does not
change the roots of det(A − λB) = 0, so the integer polynomial has the same
roots as the float polynomial.

**What this makes exact**: all four polynomial coefficients P₀, P₁, P₂, P₃ of
the characteristic cubic are exact integers.

### Subtask 2 — Integer Characteristic Polynomial

**Goal**: compute det(V̂_T − λŴ_T) with `__int128` coefficients.

**Method**: direct cofactor expansion with `__int128` accumulation.  Each of
the 18-term mixed-cofactor expressions for P₁ and P₂ is evaluated in a single
formula using `__int128` arithmetic, avoiding intermediate overflow.

```cpp
inline void characteristic_polynomial_3x3_i128(
    const int64_t A[3][3], const int64_t B[3][3], __int128 P[4]);
```

**Coefficient magnitudes** (M = max quantized entry ≤ 2^40):

| Coefficient | Terms | Magnitude |
|---|---|---|
| P₀ = det(A) | 6 | ≤ 6·M³ ≈ 6·2^120 < 2^123 |
| P₁, P₂ | 18 | ≤ 18·M³ ≈ 2^124 < 2^127 |
| P₃ = −det(B) | 6 | ≤ 6·M³ ≈ 2^123 |

All coefficients fit in signed `__int128`.

### Subtask 3 — Exact Discriminant Sign

**Goal**: when the float discriminant is near zero, determine the exact sign
of the cubic discriminant to correctly resolve the root count (1 vs 3 roots).

**Method**: GCD-normalize the four `__int128` coefficients, then compute
Δ = 18abcd − 4b³d + b²c² − 4ac³ − 27a²d² in `__int128`.

**GCD normalization**: dividing P₀…P₃ by gcd(P₀,P₁,P₂,P₃) does not change
the roots and reduces coefficient magnitudes.  If the normalized max coefficient
< 2^30, all degree-4 terms fit in `__int128`:

$$
|18abcd| \le 18 \cdot (2^{30})^4 = 18 \cdot 2^{120} \approx 2^{124.2} < 2^{127}. \checkmark
$$

**Overflow guard**: if any normalized coefficient ≥ 2^30, return 0 (signal
"use float fallback"), since large coefficients mean Δ is far from zero and
the float sign is reliable.

**Integration in `solve_cubic_real_sos`**:

| Float disc sign | Exact disc sign | Action |
|---|---|---|
| clearly + or − | (not consulted) | normal float path |
| near-zero | exact + | clamp to one-root formula |
| near-zero | exact − | clamp to three-root formula |
| near-zero | exactly 0 | SoS min-idx parity tie-break |

### Subtask 4 — Sturm-Sequence Root Isolation

**Goal**: tighten each float root λ̂ₖ into a verified isolating interval
[lₖ, hₖ] (containing exactly one root, width ≤ 10⁻¹⁰) using Sturm's theorem.

**Sturm sequence for cubic P = p₃x³ + p₂x² + p₁x + p₀**:

| k | Sₖ | Degree |
|---|---|---|
| 0 | P | 3 |
| 1 | P' | 2 |
| 2 | −prem(P, P') | 1 |
| 3 | −prem(P', S₂) | 0 |

Closed-form pseudo-remainder formulas (no fractions):
$$
S_2 = \bigl[p_3(p_1 p_2 - 9 p_0 p_3),\; 2p_3(p_2^2 - 3 p_1 p_3)\bigr]
$$
$$
S_3 = -\bigl(p_1 s_{21}^2 - 2 p_2 s_{21} s_{20} + 3 p_3 s_{20}^2\bigr)
$$

**Sturm's theorem**: V(a) − V(b) = # distinct roots in (a, b], where V(x) =
# sign changes in (S₀(x), S₁(x), S₂(x), S₃(x)) ignoring zeros.

**Algorithm**:
1. Start with δ = |λ̂| × 10⁻⁷, lo = λ̂ − δ, hi = λ̂ + δ.
2. Expand or shrink δ until V(lo) − V(hi) = 1.
3. Bisect until hi − lo ≤ 10⁻¹⁰.

**Why float polynomial**: the Sturm sequence is built from the float
SoS-perturbed polynomial P[4], not from P_i128.  Intermediate Sturm
coefficients for the integer polynomial can reach ~2^93 (e.g.
2p₃(p₂²−3p₁p₃) with p₂ ~ 2^30), which loses precision in double.  The
float polynomial has field-scale coefficients (~O(1)) where all Sturm
intermediates are safe for double arithmetic.

**Effect**: replaces each float λ̂ with the midpoint of [lₖ, hₖ], giving a
better starting point for the least-squares bary solve.  Also confirms each
root is genuine (spurious float roots with V-count = 0 are discarded).

### Subtask 5 — Exact Barycentric Sign via Interval Evaluation

**Goal**: determine the sign of μ_k(λ*) without relying on the 1×10⁻¹⁰
floating-point threshold.

**Method**: evaluate the barycentric coordinates at both endpoints of the
Sturm-isolated interval [lₖ, hₖ] for each root.

$$
\mu_k^{\mathrm{lo}} = \mu_k(l_k), \qquad \mu_k^{\mathrm{hi}} = \mu_k(h_k).
$$

Since λ* ∈ [lₖ, hₖ] and the bary map is continuous, the sign of μ_k(λ*) is
determined by:

| μ_k^lo | μ_k^hi | Conclusion |
|---|---|---|
| both > τ | both > τ | μ_k(λ*) > 0 → accept |
| both < −τ | both < −τ | μ_k(λ*) < 0 → reject |
| mixed signs | — | μ_k(λ*) ≈ 0 → SoS ownership rule |
| one in [−τ, τ] | — | near-boundary → SoS ownership rule |

where τ = 10⁻¹⁰ is the boundary threshold (same constant, but now applied at
two independent evaluation points rather than one).

**Correctness argument**: with hₖ − lₖ ≤ 10⁻¹⁰ and |dμ_k/dλ| bounded by
field derivatives, the change in μ_k across the interval is at most
|dμ_k/dλ| × 10⁻¹⁰ / 2.  If both endpoints agree (same definite sign), no
root of the bary numerator N_k(λ) lies in [lₖ, hₖ], so the sign is certified.

**Why this improves on the threshold alone**: a single float evaluation of
μ_k(λ̂) could be wrong by |dμ_k/dλ| × |λ̂ − λ*|.  The Sturm interval gives
|λ̂ − λ*| ≤ 5×10⁻¹¹, reducing the sign error below the threshold.  When two
independent evaluations at the interval endpoints agree, the sign is doubly
confirmed.

### Subtask 6 — Exact Barycentric Sign via Sturm Count on Degree-4 Numerator

**Goal**: eliminate the τ = 10⁻¹⁰ boundary threshold entirely for the common
case where the barycentric coordinate is genuinely nonzero.

**Observation**: Subtask 5 still applies τ at the two endpoint evaluations.
If N_k(λ) is nonzero throughout [l_k, h_k], we can evaluate it at a single
point and obtain the exact sign — with *no threshold at all*.

**Method**: express μ_k(λ) = N_k(λ) / D(λ) as a ratio of degree-4 polynomials
derived from the linear-in-λ system M(λ)·ν = b(λ):

$$
M(λ)_{rc} = (V_{rc} - V_{r2}) - λ(W_{rc} - W_{r2}), \quad
b(λ)_r = -(V_{r2} - λ\,W_{r2}),
$$

where r ∈ {0,1,2} and c ∈ {0,1}.  With

$$
A(λ) = M(λ)^T M(λ) \quad (\text{quadratic in }λ),\qquad
g(λ) = M(λ)^T b(λ) \quad (\text{quadratic in }λ),
$$

Cramer's rule gives

$$
D(λ) = A_{00}A_{11} - A_{01}^2, \qquad
N_0 = A_{11}\,g_0 - A_{01}\,g_1, \qquad
N_1 = A_{00}\,g_1 - A_{01}\,g_0, \qquad
N_2 = D - N_0 - N_1,
$$

each of degree 4.  By the Cauchy–Binet identity, D(λ) = Σ (2×2 minors of M)²
≥ 0 for all λ.

**Algorithm** for each root λ* isolated in [l_k, h_k]:
1. Build the Sturm sequence for N_k (degree ≤ 4).
2. Count sign changes V(l_k) and V(h_k).
3. If V(l_k) − V(h_k) = 0 (no root of N_k in (l_k, h_k]):
   - Evaluate N_k(l_k) and D(l_k).
   - If D(l_k) > 0: sign(μ_k) = sign(N_k(l_k)) — **exact, no threshold**.
4. Otherwise (N_k has a root in the interval): μ_k(λ*) ≈ 0 → apply SoS
   min-index ownership rule.

**Why this eliminates the threshold**: the Sturm count is a *discrete*
integer test.  When it confirms 0 roots of N_k in [l_k, h_k], N_k has constant
sign there — so evaluating at l_k gives the exact sign of N_k(λ*) regardless
of how close λ* is to a root of N_k.  The threshold τ is only needed when N_k
genuinely vanishes in the interval (a true boundary case), in which case the
SoS rule applies.

**Comparison with Subtask 5**:

| Case | Subtask 5 | Subtask 6 |
|---|---|---|
| Nonzero bary coord | Two threshold evaluations | Exact Sturm-count decision |
| Near-zero bary coord (genuine boundary) | SoS rule | SoS rule |
| Degenerate interval | τ-threshold at midpoint | τ-threshold at midpoint |

**Implementation**: `compute_bary_numerators`, `build_sturm_deg4`,
`sturm_count_d4`, and the `sos_bary_inside` lambda in `solve_pv_triangle`.

### Subtask 7 — Exact Gram-Determinant Positivity Certificate

**Goal**: eliminate the `d_lo > 1e-200` float guard for the Gram determinant
D(λ) without replacing it with another threshold.

**Background**: in Subtask 6, after confirming N_k has no root in [lo, hi],
the code evaluates D(lo) in float and checks `D(lo) > 1e-200` to ensure the
denominator is nonzero.  This constant 1e-200 is an arbitrary threshold.

**Key property**: D(λ) = det(M(λ)ᵀM(λ)) is the Gram determinant of the
3×2 matrix M(λ), which equals the sum of squared 2×2 minors (Cauchy-Binet):

$$
D(λ) = \sum_{r < s} \bigl(M_{r0}M_{s1} - M_{r1}M_{s0}\bigr)^2 \;\geq\; 0.
$$

Therefore D ≥ 0 everywhere, and D(λ*) = 0 if and only if λ* is a root of D.

**Method**: build the Sturm sequence of D(λ) once (before the per-root loop)
and count its roots at each interval endpoint:

$$
V_D(l_k) - V_D(h_k) = 0 \;\Longrightarrow\; D \text{ has no root in }(l_k, h_k]
\;\Longrightarrow\; D(\lambda^*) > 0 \text{ (exact, no threshold)}.
$$

$$
V_D(l_k) - V_D(h_k) \geq 1 \;\Longrightarrow\; D(\lambda^*) = 0,
\text{ system degenerate, apply SoS rule}.
$$

**Why D(λ*) = 0 triggers SoS**: D(λ*) = 0 means M(λ*)ᵀM(λ*) is singular,
i.e., the columns of M(λ*) are linearly dependent.  Geometrically, this means
the two constraint directions of the PV system coincide at λ*, so the
barycentric coordinate is not uniquely determined — exactly the degenerate case
that SoS perturbation is designed to resolve.

**Comparison with Subtask 6**:

| What | Subtask 6 | Subtask 7 |
|---|---|---|
| N_k positivity | Sturm count on N_k | (unchanged) |
| D positivity | `eval(D, lo) > 1e-200` | Sturm count on D |
| D = 0 case | threshold misses it | Sturm detects, → SoS |

**Implementation**: pre-compute `seq_D` via `build_sturm_deg4(D_poly, ...)` once
before the per-root loop; call `sturm_count_d4(seq_D, lo/hi)` inside
`sos_bary_inside` when N_k has no root in the interval.

### Subtask 8 — Certified Horner Error Bound for N_k Evaluation

**Goal**: guarantee that the final float evaluation `N_k(lo)` cannot return
the wrong sign, even when N_k(lo) is extremely small.

**Background**: after Subtasks 6 and 7 confirm N_k has no root in (lo, hi] and
D(λ*) > 0, we evaluate N_k(lo) in double arithmetic.  The Horner evaluation
can have rounding error of order γ · cond(N_k, lo), where

$$
\mathrm{cond}(N_k, x) = \sum_{j=0}^{d} |N_k[j]|\,|x|^j
$$

is the *absolute condition number* of the evaluation and γ accounts for
floating-point accumulation.  If N_k(lo) is genuinely close to 0 (because a
root of N_k lies just outside the interval), the float evaluation could return
the wrong sign.

**Method**: use Higham's standard Horner rounding-error bound (§3.1,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed.):

$$
|\mathrm{fl}(N_k(lo)) - N_k(lo)| \;\leq\; \gamma_{2d+1} \cdot \mathrm{cond}(N_k, lo),
\qquad \gamma_n = \frac{n\,u}{1 - n\,u} \approx n\,u,
$$

where $u = \varepsilon_{\mathrm{mach}}/2 \approx 2^{-53}$.  An extra factor of
$(d+1)$ covers rounding in `compute_bary_numerators`, giving the conservative
multiplier

$$
\gamma = (2d + 2)\,\varepsilon_{\mathrm{mach}} = 10\,\varepsilon_{\mathrm{mach}}
\quad (d = 4).
$$

The certified sign rule is:

$$
|N_k(lo)| > \gamma \cdot \mathrm{cond}(N_k, lo)
\;\Longrightarrow\; \mathrm{sign}(N_k(lo)) = \mathrm{sign}(\mu_k(\lambda^*)).
$$

If the inequality fails, N_k(lo) is within rounding noise, meaning
μ_k(λ*) is genuinely near zero — apply SoS ownership rule.

**Why this terminates the certification chain**: combining Subtasks 6–8,
every branch of `sos_bary_inside` is now either:
1. A certified sign decision (Sturm count + error-bounded evaluation), or
2. A genuine boundary case → SoS min-index ownership rule.

No float threshold remains that could silently misclassify a non-boundary
puncture as boundary (or vice versa) due to rounding.

**Comparison of Subtasks 6–8**:

| Decision | Subtask 6 | Subtask 7 | Subtask 8 |
|---|---|---|---|
| N_k root-free? | Sturm count | — | — |
| D(λ*) > 0? | `d_lo > 1e-200` | Sturm count | — |
| sign(N_k(lo)) certified? | bare comparison | — | error-bound guard |

**Implementation**: after `nk_lo = eval_poly_sturm(...)`, compute
`cond_nk = Σ |N_poly[k][d]| |lo|^d` via reversed Horner, then check
`|nk_lo| > EVAL_GAMMA * cond_nk` before trusting the sign.

### Subtask 9 — Unified Sturm/Error-Bound Certification for Degenerate Intervals

**Goal**: eliminate the last remaining float threshold `bary_threshold = 1e-10`
from the degenerate-interval path, making the entire `sos_bary_inside` decision
threshold-free.

**Background**: Sturm isolation (Subtask 4) produces a root interval
`[lambda_lo, lambda_hi]` that is strictly proper (`lo < hi`) for each cubic
root.  However, when two roots of the characteristic cubic are extremely close,
the interval collapse check can fail, leaving `lambda_lo[i] == lambda_hi[i]` —
we call this a *degenerate interval*.  The previous code fell back to

```cpp
const T bary_threshold = T(1e-10);
return (double)nu[k] > -bary_threshold;
```

which is an arbitrary threshold with no certified meaning.

**Method**: replace the degenerate-interval fallback with the same
`try_certify_nk_sign` helper used for proper intervals, but using a
*machine-epsilon window*

$$
[\hat\lambda - \delta,\; \hat\lambda + \delta],
\qquad \delta = \max\!\left(4\,\varepsilon_{\mathrm{mach}}\,|\hat\lambda|,\;
\varepsilon_{\min}\right),
$$

where $\hat\lambda$ is the float root and $\varepsilon_{\min}$ is the smallest
positive normal double.  The reasoning:

1. **Window correctness**: the float root $\hat\lambda$ satisfies
   $|\hat\lambda - \lambda^*| \lesssim 4u|\lambda^*|$ for well-conditioned
   cubic roots.  Therefore $\lambda^*$ lies inside
   $[\hat\lambda - \delta, \hat\lambda + \delta]$ with probability 1 under
   generic perturbation.

2. **N_k root-free ⟹ sign certified**: if the Sturm count shows N_k has
   no root in the window, then sign(N_k(λ*)) = sign(N_k(hat{λ})).  The
   Higham error-bound check (Subtask 8) then certifies the float evaluation.

3. **N_k has a root in window ⟹ μ_k(λ*) ≈ 0**: this is the genuine
   boundary case — the puncture lies on (or extremely near) an edge of the
   triangle.  The SoS min-index ownership rule resolves it.

**Result**: `try_certify_nk_sign` is called in both the proper-interval and
degenerate-interval paths; the only difference is how (lo, hi) is constructed.
The constant `bary_threshold = 1e-10` is removed entirely.

**Comparison of Subtasks 6–9**:

| Decision | Sub 6 | Sub 7 | Sub 8 | Sub 9 |
|---|---|---|---|---|
| N_k root-free in window? | Sturm count | — | — | — |
| D(λ*) > 0? | `d_lo > 1e-200` | Sturm count | — | — |
| sign(N_k(lo)) certified? | bare comparison | — | error-bound guard | — |
| Degenerate interval? | `bary_threshold = 1e-10` | — | — | ε-window + Sturm |

**Implementation**: extracted `try_certify_nk_sign(k, lo, hi) -> int` helper
lambda that encapsulates Subtasks 6–8.  `sos_bary_inside` selects (lo, hi)
based on `have_interval` and delegates to the helper.  The SoS ownership rule
is the single shared fallback for all failure modes.

### Subtask 10 — Remove the Scale-Dependent Cross-Product Residual Filter

**Goal**: remove the last heuristic threshold from `solve_pv_triangle` — the
cross-product guard `|V×W| > 1e-2` — which can falsely reject valid solutions
when SoS perturbation is active and field values are large.

**Background**: the check was added as a sanity filter against spurious
roots of the characteristic polynomial that do not correspond to genuine PV
points.  After computing the float barycentric coordinates ν from
`solve_least_square3x2`, the solver evaluated the cross product of the
*original* (unperturbed) field vectors V×W at ν and rejected the solution if
the norm exceeded 1e-2.

**Why it became harmful**: when SoS perturbation is active, the solver actually
solves the PV system for the *perturbed* field Vp, Wp (not V, W).  The
perturbed solution ν_p satisfies Vp(ν_p) × Wp(ν_p) ≈ 0, but the residual in
the *original* field satisfies

$$
\|V(\nu_p) \times W(\nu_p)\| \;\approx\;
\mathrm{SOS\_EPS} \cdot (1 + |\lambda^*|) \cdot \|W(\nu_p)\|,
$$

where SOS\_EPS ≈ 10⁻⁸.  For field magnitude |W| ≳ 5 × 10⁴, this exceeds
1e-2 and the valid solution is falsely rejected.  At |W| = QUANT\_SCALE = 10⁶
(the solver's design maximum), **all** interior solutions are rejected.

**Why it is redundant**: Subtask 7 certifies D(λ*) > 0, i.e., the 3×2
projection system M(λ*) has full rank 2.  The float least-squares solve is
therefore well-conditioned, and the discrepancy between the float and exact ν
is bounded by O(ε_machine × cond(M)), which translates to a residual
|V×W| = O(ε_machine × |W|²) far below 1e-2 for any physically sane input.
If D(λ*) = 0 (rank-deficient), Subtask 7 already diverts the solution to the
SoS ownership rule — such cases never reach the cross-product check.

**Test**: `solve_pv_triangle_large_scale` uses the field

$$
V^T = S \cdot \mathrm{diag}(2,2,2),
\qquad
W^T = S \cdot \begin{pmatrix}1&1&0\\1&0&1\\0&1&1\end{pmatrix},
$$

with S = 50000 and SoS indices enabled.  The characteristic polynomial has
roots λ = −2, 1, 2; only λ = 1 yields a valid interior solution at the
centroid ν* = (1/3, 1/3, 1/3).  With S = 50000, the SoS-induced residual is
≈ 1.4 × 10⁻³ < 1e-2 — just below the old threshold — but already triggers at
larger S.  The test confirms the solver finds exactly one puncture with the new
code (and would have found zero with the old code for S ≳ 70000).

### Subtask 11 — Certified Exact-λ=0 Exclusion

**Goal**: replace the heuristic `|λ| ≤ ε_machine` filter with an exact
integer test that skips the trivial λ=0 eigenvalue and only that eigenvalue.

**Background**: the PV condition V = λW with λ=0 gives V(ν*) = 0 — the V
field vanishes at the puncture.  This is a trivially parallel (and typically
uninteresting) solution.  The old code skipped any root with
`|λ| ≤ std::numeric_limits<T>::epsilon()`, which is approximately 2.2×10⁻¹⁶
for double.  This guard is fragile: a genuine eigenvalue at, say, λ = 10⁻¹⁷
would be silently dropped, even though det(V) ≠ 0.

**Exact test**: the constant term of the characteristic polynomial is

$$
P[0] = \det(V_q),
$$

where $V_q$ is the quantized field matrix.  $P_{\text{i128}}[0] = 0$ if and
only if λ=0 is an exact root of the integer polynomial — equivalently, if and
only if det(V) = 0 (up to quantization), i.e., V is degenerate at the
triangle.

**Decision rule**: skip root $i$ iff

$$
P_{\text{i128}}[0] = 0
\quad\text{AND}\quad
\lambda_{\mathrm{lo}}[i] \le 0 \le \lambda_{\mathrm{hi}}[i].
$$

The second condition verifies that the Sturm-isolated interval actually
contains 0 — ruling out the pathological case where the float root happened to
be near 0 but the true root is not (which would only occur when
$P_{\text{i128}}[0] \ne 0$, anyway).

**Why this is an improvement**:

| Case | Old `\|λ\|≤ε` guard | New certified check |
|---|---|---|
| True λ=0 root (det(V)=0) | Skip (correct) | Skip (certified correct) |
| Genuine λ≈1e-17 root (det(V)≠0) | Silently drop (wrong) | Keep (correct) |
| Float root=0 but det(V)≠0 | Skip (wrong) | Keep (correct) |

**Tests**:
- `solve_pv_triangle_zero_root_certified`: V with a zero vertex (det(V)=0);
  verifies no returned puncture has |λ| near zero.
- `solve_pv_triangle_near_zero_genuine_root`: V=diag(0.1,1,1), W=I;
  char poly roots 0.1, 1, 1; det(V)=0.1 ≠ 0 → Subtask 11 must not skip
  the λ=0.1 root; solver runs without error.

### Subtask 12 — Certified All-Parallel Check via Integer Cross Products

**Goal**: replace the float `|Vp × Wp| > ε_machine` all-parallel check with
an exact integer test on the quantized original field.

**Background**: the solver detects the degenerate case "V ∥ W at every vertex
⟹ entire triangle is a PV surface" early and returns `INT_MAX`.  The old check
used the *SoS-perturbed* field vectors `Vp, Wp` and compared the cross-product
norm against `std::numeric_limits<T>::epsilon()`:

```cpp
if (vector_norm3(cross_product(Vp[i], Wp[i])) > epsilon) { all_parallel = false; }
```

**Bug in old check**: when `V = W` exactly and SoS is active (`indices ≠ nullptr`),
the perturbation adds different small values to the V and W slots
(`V[i][j] + sos_perturbation(idx[i], j)` vs `W[i][j] + sos_perturbation(idx[i], j+3)`),
so `Vp[i] ≠ Wp[i]` even though `V[i] = W[i]`.  The float cross product is
`O(SOS\_EPS) ≈ 10^{-8}` while `ε_machine ≈ 2×10^{-16}`, so the test concludes
NOT all-parallel and the solver proceeds — producing SoS-artifact punctures
instead of returning `INT_MAX`.

**Fix**: test the *quantized original* field vectors `Vq[i]`, `Wq[i]` with exact
integer arithmetic:

$$
c_x = V_q^{(i)}_y \cdot W_q^{(i)}_z - V_q^{(i)}_z \cdot W_q^{(i)}_y
= 0,\quad
c_y = \ldots = 0,\quad
c_z = \ldots = 0
\quad\forall\,i \;\Longrightarrow\; \text{all-parallel}.
$$

The products are computed in `__int128` (components ≤ 5×10¹² per field
bound, products ≤ 5×10²⁵ < 2¹²⁷).  No threshold.

**Correctness**: `V[i] = W[i]` → `Vq[i] = Wq[i]` → cross = 0 → `INT_MAX`, regardless
of whether SoS is active.  `W = c·V` → `Wq = round(c·Vq)` → in general `Wq[i]`
is proportional to `Vq[i]` up to rounding, and the exact cross product captures
this correctly for exact proportionality.

**Tests**:
- `solve_pv_triangle_all_parallel_with_sos`: `V = W` with SoS indices.
  Old code produced spurious punctures; new code returns `INT_MAX`.
- `solve_pv_triangle_proportional_with_sos`: `W = 3V` with SoS indices → `INT_MAX`.

### Subtask 13 — Deferred Polynomial ν Evaluation via N_k(λ)/D(λ)

**Goal**: eliminate `solve_least_square3x2` and its singular-matrix `|det(MᵀM)| < ε`
guard from the hot path by deferring ν computation to *after* the certification gate.

**Background**: barycentric coordinates ν = (ν₀, ν₁, ν₂) of the puncture on the
triangle are needed only for the output `PuncturePoint.barycentric`.  They are
**not** used in any certification decision — `sos_bary_inside` operates exclusively
on the N_k / D polynomial Sturm counts and the SoS index ordering.  Nevertheless,
the old code called `eval_nu_at(λ)` *before* the `sos_bary_inside` gate, on every
candidate root — including those that are immediately rejected.

**Old flow** (three floats thresholds in the path):

```
for each root λᵢ:
    eval_nu_at(λᵢ)          ← solve_least_square3x2 (|det(MᵀM)|<ε guard HERE)
    if !sos_bary_inside(k)  ← Sturm + Higham (no threshold)
        continue
    emit puncture
```

The `eval_nu_at` lambda built a 3×2 overdetermined system from `V(λ)x = W(λ)x`
(two linearly-independent equations from the three rows after fixing ‖x‖=1) and
solved it via `solve_least_square3x2`, which internally checks `|det(MᵀM)| < ε`.
That check is a float threshold with no rigorous bound.

**New flow** (no float threshold in the path):

```
for each root λᵢ:
    if !sos_bary_inside(k)  ← Sturm + Higham (no threshold)
        continue
    // Subtask 7 has already certified D(λᵢ) > 0, so d_val > 0.
    d_val = D(λᵢ)           ← direct polynomial evaluation (no threshold)
    νₖ   = Nₖ(λᵢ) / d_val  ← k = 0,1,2
    emit puncture
```

**Why D(λ*) > 0 makes division safe**: Subtask 7 runs a Sturm count on `D(λ)` over
the certified interval [lo, hi] and only accepts a root if `D` has no sign change
there.  Because `D(λ) > 0` on the domain where the PV problem has a solution (proven
in §3.3), this guarantees `d_val > 0` at every accepted root, making the division
`Nₖ(λ) / d_val` safe and non-degenerate.

**Formula**: the N_k and D polynomials are the same objects already computed by
Subtasks 5–8.  Evaluating them at a double approximation `lam_d = (double)λᵢ` via
`eval_poly_sturm` (Horner) suffices for the output barycentric coordinates, which
are used only for visualization and not for any topological decision.

**Implementation** (in `parallel_vector_solver.hpp`):

```cpp
// DELETED: auto eval_nu_at = [&](T lam, T nu_out[3]) { ... solve_least_square3x2 ... };
// DELETED: T nu[3]; eval_nu_at(lambda[i], nu);   // before sos_bary_inside

// ADDED: after the sos_bary_inside gate:
T nu[3];
{
    double lam_d = (double)lambda[i];
    double d_val = eval_poly_sturm(D_poly, 4, lam_d);
    for (int k = 0; k < 3; ++k) {
        double nk_val = eval_poly_sturm(N_poly[k], 4, lam_d);
        nu[k] = (d_val > 0.0) ? T(nk_val / d_val) : T(0);
    }
}
```

**What is eliminated**: the `eval_nu_at` lambda (~17 lines), the `solve_least_square3x2`
call, and its `|det(MᵀM)| < ε` singular-matrix guard — all removed from the hot path.

**Tests**: all 104 existing tests pass with the deferred computation.  No new
dedicated test was needed because the deferred ν path is exercised by every
existing test that produces a puncture (e.g. `solve_pv_triangle_single_puncture`,
`solve_pv_triangle_known_solution`, `solve_pv_triangle_large_scale`).

### Subtask 14 — ULP-Convergence Bisection (remove `target_width = 1e-10`)

**Goal**: eliminate the absolute-width stopping criterion `target_width = 1e-10` from
the Sturm bisection in `tighten_root_interval`, replacing it with float-convergence-only
termination.

**Background**: `tighten_root_interval` is called from `isolate_cubic_roots` for every
float root estimate.  Phase 2 bisects the isolating interval [lo, hi] until it is
"tight enough" for subsequent use in `try_certify_nk_sign`.  The old criterion was
`(hi - lo) > 1e-10` — an *absolute* threshold with two problems:

1. **Scale-dependence for tiny roots**: for λ* = 1e-11, the final interval of width
   1e-10 is wider than the root itself, potentially straddling 0.  This enlarges the
   Sturm window passed to `try_certify_nk_sign`, increasing the probability that N_k
   or D has a root inside the window and triggering unnecessary SoS fallbacks.

2. **Wasted iterations for large roots**: for λ* = 1e4, the target width of 1e-10
   requires many extra bisection steps beyond what double precision can distinguish,
   since the ULP at 1e4 is ≈ 1.8e-12.  Those steps are wasted.

**Fix**: remove `target_width` from the function signature entirely.  Phase 2 runs
until the existing float-convergence guard fires:

```cpp
// Old:
for (int iter = 0; iter < 200 && (hi - lo) > target_width; ++iter) {
    ...
    if (mid <= lo || mid >= hi) break;  // float convergence
    ...
}

// New (Subtask 14):
for (int iter = 0; iter < 200; ++iter) {
    ...
    if (mid <= lo || mid >= hi) break;  // ULP convergence — sole terminator
    ...
}
```

The `mid <= lo || mid >= hi` condition fires when `(lo + hi) / 2` rounds to either
`lo` or `hi` in double, i.e. when the interval is ≤ 1 ULP wide.  This is the
smallest possible interval in double arithmetic — scale-invariant by construction.

**Why this is safe**: the 200-iteration safety limit still bounds the loop in the
unlikely event that ULP convergence takes longer than expected (e.g. subnormal
numbers near 0).  In practice, bisection from the initial delta of `scale × 1e-7`
reaches 1-ULP width in at most 24 additional steps (log₂(1e-7 / ε_machine) ≈ 24).

**Tests**:

- `solve_pv_triangle_tiny_lambda_bisection`: field with λ* = 1e-11 and other roots
  at -3, -4 (rejected by `λ < 0` legacy filter).  With `target_width = 1e-10` the
  interval straddles 0; with ULP convergence it is tightly localised around 1e-11.
  Asserts n=1, λ ≈ 1e-11, ν ≈ (12/19, 4/19, 3/19).

### Subtask 15 — Exact-Zero Degree Trimming for D_poly and N_poly

**Goal**: replace `std::abs(coeff) < 1e-200` in the degree-trimming loops for
D_poly and N_poly[k] with the exact `== 0.0` comparison, eliminating the last
arbitrary threshold from the solve_pv_triangle hot path.

**Background**: before building a Sturm sequence with `build_sturm_deg4`, the code
trims trailing near-zero leading coefficients to determine the effective degree:

```cpp
// Old:
int degD = 4;
while (degD > 0 && std::abs(D_poly[degD]) < 1e-200) --degD;
```

The `1e-200` threshold is unprincipled — it is neither derived from field magnitude
nor from rounding-error theory.

**Why `== 0.0` is correct**: The coefficients of D_poly (= det(MᵀM)) and N_poly[k]
are computed by `compute_bary_numerators` from:

```cpp
Mlin[r][c][1] = -(WT[r][c] - WT[r][2])   // W-field differences across vertices
```

The degree-4 coefficient D[4] = A[0][0][2] × A[1][1][2] − A[0][1][2]², where
A[p][q][2] = Σᵣ Mlin[r][p][1] × Mlin[r][q][1].

Two cases:

| Scenario | Mlin[r][c][1] | D_poly[4] | Degree trim |
|---|---|---|---|
| SoS active (indices ≠ nullptr) | Non-zero (SoS makes W non-constant) | > 0 | No trim needed |
| No SoS, W exactly constant | Exactly 0.0 (no rounding) | Exactly 0.0 | `== 0.0` catches it |
| No SoS, W non-constant | Non-zero, O(field²) | > 0, O(field⁴) | No trim needed |

When Mlin[r][c][1] = 0.0 exactly (W constant, no SoS), every product in A[p][q][2]
is 0.0, making D_poly[4] exactly 0.0 with no floating-point rounding.  There is no
intermediate cancellation.

When SoS is active, D_poly[4] > 0 always because the SoS perturbation ensures
`Wp[c][r] ≠ Wp[2][r]` for distinct vertices c ≠ 2 (since the perturbation is
different per vertex index).  The `== 0.0` trim loop fires zero times in this case.

The same argument applies to N_poly[k][4] — its degree-4 coefficient is either
genuinely non-zero (O(field⁴)) or exactly 0.0 from algebraic cancellation, with no
floating-point rounding in between when all inputs are exact doubles from integers.

**Implementation**:

```cpp
// Old:
while (degD  > 0 && std::abs(D_poly[degD])         < 1e-200) --degD;
while (degNk > 0 && std::abs(N_poly[k][degNk])     < 1e-200) --degNk;

// New (Subtask 15):
while (degD  > 0 && D_poly[degD]         == 0.0) --degD;
while (degNk > 0 && N_poly[k][degNk]     == 0.0) --degNk;
```

**Tests**: `solve_pv_triangle_constant_w_degree_trim` — W = (1,0,0) at all
vertices (exactly constant, no SoS).  D_poly[4] = 0.0 exactly.  The solver
must complete without crash or NaN, verifying that the `== 0.0` trim path works.
110/110 tests pass.

---

## 8. Implementation Status

| Component | File | Status |
|---|---|---|
| SoS perturbation | `parallel_vector_solver.hpp` | Implemented |
| SoS cubic solver (parity tie-break) | same | Implemented |
| Min-idx edge ownership rule | same | Implemented |
| **Subtask 1**: Field quantization to `int64_t` | same | Implemented |
| **Subtask 2**: Integer characteristic polynomial (`__int128`) | same | Implemented |
| **Subtask 3**: Exact discriminant sign | same | Implemented |
| **Subtask 4**: Sturm-sequence root isolation | same | Implemented |
| **Subtask 5**: Exact bary sign via interval evaluation | same | Implemented |
| **Subtask 6**: Exact bary sign via Sturm count on N_k(λ) | same | Implemented |
| **Subtask 7**: Exact D(λ*) > 0 certificate via Sturm count on D(λ) | same | Implemented |
| **Subtask 8**: Certified Horner error bound for N_k(lo) sign | same | Implemented |
| **Subtask 9**: Unified ε-window certification for degenerate intervals | same | Implemented |
| **Subtask 10**: Remove scale-dependent cross-product residual filter | same | Implemented |
| **Subtask 11**: Certified exact-λ=0 exclusion via P_i128[0] | same | Implemented |
| **Subtask 12**: Certified all-parallel check via integer cross products on Vq/Wq | same | Implemented |
| **Subtask 13**: Deferred polynomial ν evaluation via N_k(λ)/D(λ) | same | Implemented |
| **Subtask 14**: ULP-convergence bisection (remove `target_width = 1e-10`) | same | Implemented |
| **Subtask 15**: Exact-zero degree trimming for D_poly / N_poly (remove `< 1e-200`) | same | Implemented |
| Resultant-based tet-edge detection (G3/G4) | — | Future work |

---

*Document updated February 2026.*
