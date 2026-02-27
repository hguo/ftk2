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

## 7. Implementation Status

| Component | File | Status |
|---|---|---|
| SoS perturbation | `include/ftk2/numeric/parallel_vector_solver.hpp` | Implemented |
| SoS cubic solver | same | Implemented (disc=0 parity tie-break) |
| Min-idx edge ownership | same | Implemented (`sos_bary_inside` lambda) |
| Predicate index passing | `include/ftk2/core/predicate.hpp` | Implemented |
| Test field (single circle) | `examples/exact_pv_stitching.cpp` | Implemented |
| Quantization (__int128) | — | Future work |
| Resultant-based edge detection | — | Future work |

---

*Document generated from session notes, February 2026.*
