# Exact Parallel Vector (ExactPV) Tracking - Research & Implementation Plan

## Overview

This document outlines the research and implementation plan for **Exact Parallel Vector (ExactPV) tracking** in FTK2, with special focus on **4D spacetime generalization**. This represents a significant research contribution that extends beyond the existing 3D spatial implementation.

**Goal**: Implement exact analytical tracking of parallel vectors (v × w = 0) with full 4D spacetime support, handling all corner cases rigorously.

**Expected Outcome**: Research paper + production-ready implementation

---

## Background

### Mathematical Foundation

**Parallel Vector Criterion**: Find locations where v(x) ∥ w(x), i.e., v × w = 0

In piecewise-linear (PL) fields on simplicial meshes:
- **3D spatial (tetrahedra)**: Parallel vectors form **cubic rational curves**
- **4D spacetime (4-simplices/pentatopes)**: Parallel vectors form **cubic rational surfaces** (2D manifolds)

**Key Papers**:
1. Hanqi Guo, et al. "Exact Analytical Parallel Vectors Operator" (2021): [arXiv:2107.02708](https://arxiv.org/abs/2107.02708)
2. Extends work on critical points and levelsets to parallel vectors

**Physical Applications**:
- Vortex core lines in 3D turbulence
- Vortex core surfaces in 4D spacetime
- Magnetic field topology
- Vector field singularities

---

## Current State (Legacy FTK)

### Implemented (3D Spatial)

**Files**:
- `/ftk/include/ftk/numeric/parallel_vector_solver3.hh` - 3D solver
- `/ftk/include/ftk/numeric/parallel_vector_solver2.hh` - 2D solver
- `/ftk/tests/test_parallel_vectors.cpp` - Unit tests

**Key Functions**:

1. **Triangle (2-simplex in 3D)**:
   ```cpp
   int solve_pv_s2v3(const T V[3][3], const T W[3][3],
                     T lambda[3], T mu[3][3]);
   ```
   - Solves cubic characteristic polynomial
   - Returns up to 3 puncture locations
   - Each puncture: (lambda, mu) where mu are barycentric coords

2. **Tetrahedron (3-simplex in 3D)**:
   ```cpp
   void characteristic_polynomials_pv_s3v3(const T V[4][3], const T W[4][3],
                                            T Q[4], T P[4][4]);
   disjoint_intervals<T> solve_pv_inequalities_s3v3(const T V[4][3], const T W[4][3]);
   ```
   - Computes characteristic polynomial Q(λ) and rational barycentric coords P_i(λ)/Q(λ)
   - Solves inequalities to find valid λ intervals where all mu_i ∈ [0,1]
   - Returns intervals representing curve segments

**Assumptions** (LIMITATIONS):
- ✅ Works for 3D spatial (static fields)
- ⚠️ Each 2-cell (triangle/face) **assumes ≤1 puncture**
- ❌ No 4D spacetime support
- ❌ No SoS (Simulation of Simplicity) for degenerate cases
- ❌ No theory for stitching multiple punctures in 4D

---

## Research Challenges for 4D Generalization

### Challenge 1: Multiple Punctures per 2-Cell

**Problem**:
- In 3D: Triangle can have **up to 3 punctures** (cubic has 3 real roots)
- Current tracing code assumes **≤1 puncture per 2-cell**
- This breaks manifold stitching logic

**Example**:
```
Triangle T has 3 punctures: p1, p2, p3
Current code: Only handles p1
Result: Broken/incomplete manifold
```

**Research Needed**:
1. Extend manifold data structure to handle multi-puncture faces
2. Develop stitching algorithm for multiple entry/exit points
3. Determine connectivity: Which punctures connect to which edges?
4. Handle topology: Does this create handles/holes in the surface?

**Implementation Approach**:
- Store **all punctures** per 2-cell, not just first
- Use graph-theoretic approach: punctures as nodes, edges as connections
- Apply topological consistency checks

### Challenge 2: Lack of SoS for Degenerate Cases

**Problem**:
- PV curve can intersect simplex **edges** or **vertices** (lower-dimensional simplices)
- Current solver uses floating-point checks (epsilon thresholds)
- No symbolic perturbation (SoS) to resolve degeneracies

**Degenerate Cases**:
1. **Curve passes through vertex**: All 3/4 vertices in a simplex
2. **Curve lies on edge**: Puncture exactly at barycentric boundary (mu_i = 0 or 1)
3. **Curve lies on face**: Entire face is part of PV surface

**Research Needed**:
- Extend SoS framework from critical points to parallel vectors
- Derive perturbation scheme for vector fields (not just scalar!)
- Ensure consistent orientation across simplices

**Reference**:
- FTK2 critical points use SoS (Edelsbrunner-Mücke)
- Need to generalize to 2D manifolds (PV surfaces)

### Challenge 3: 4D Spacetime - Stitching Complexes in 4-Cells

**Problem**:
- In 4D (4-simplex/pentatope), PV surface is a **cubic rational 2-manifold**
- Surface parametrization: (λ, t) → barycentric coords (mu_0, mu_1, mu_2, mu_3, mu_4)
- Must stitch 2D surface patches across 3D faces (tetrahedral boundaries)

**Key Questions**:
1. **How to extract surface patch** from a 4-simplex?
   - 3D: Extract curve from tetrahedron (1D manifold from 3D cell)
   - 4D: Extract surface from pentatope (2D manifold from 4D cell)

2. **How to match punctures** across 3D faces?
   - 3D: Match single punctures on triangular faces
   - 4D: Match **puncture curves** on tetrahedral faces
   - Need consistency: puncture curves must align at shared faces

3. **What is the topology** of PV surfaces in 4D?
   - Can surfaces have boundaries? (Yes, at domain boundaries)
   - Can surfaces be closed? (Yes, loops in spacetime)
   - Can surfaces have handles/genus > 0? (Open question)

**Mathematics**:
- PV surface in pentatope is defined by:
  ```
  v(λ, t) × w(λ, t) = 0
  ```
  where v, w are linearly interpolated in space-time barycentric coords

- Parametric form:
  ```
  λ(s, t) ∈ ℝ (scalar parameter)
  t(s, t) ∈ ℝ (time parameter)
  mu_i(λ, t) ∈ [0,1] (5 barycentric coords, sum = 1)
  ```

- **Cubic rational surface**: Degree 3 in (λ, t)

**Research Needed**:
1. Derive characteristic polynomials for 4D case
2. Develop inequality solver for 2D parameter space
3. Create surface extraction algorithm (marching-cubes-like for 4D?)
4. Prove stitching consistency at shared 3-cells

### Challenge 4: Computational Complexity

**Problem**:
- 3D solver: Cubic polynomial → O(1) roots per tetrahedron
- 4D solver: Cubic rational surface → Infinitely many points, need discretization

**Questions**:
1. How to represent cubic rational surfaces?
   - Implicit: Q(λ, t) · mu_i(λ, t) - P_i(λ, t) = 0
   - Parametric: Generate mesh on (λ, t) domain?
   - Adaptive refinement?

2. How to ensure completeness?
   - Marching cubes misses thin features
   - Need adaptive sampling based on curvature?

3. Performance targets:
   - 3D spatial: ~1000 tets/sec on CPU
   - 4D spacetime: ???

**Implementation Strategy**:
- Start with uniform sampling in (λ, t) parameter space
- Extract isosurface where constraints satisfied
- Later: Adaptive refinement based on surface complexity

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Set up ExactPV infrastructure in FTK2

**Tasks**:
1. Create `include/ftk2/numeric/parallel_vector_solver.hpp`
   - Port 3D solver from legacy FTK
   - Clean up API, modernize C++ (C++17)
   - Add comprehensive unit tests

2. Create `ExactPVPredicate<M, T>` (M = codimension)
   - `ExactPVPredicate<2, double>` for 3D spatial (curves)
   - Template for future 4D (surfaces)

3. Basic extraction without stitching:
   - Detect punctures on 2-simplices
   - No manifold construction yet (just point cloud)

**Deliverable**: Working 3D spatial ExactPV extraction (point cloud output)

---

### Phase 2: Multi-Puncture Handling (Weeks 3-4)

**Goal**: Handle multiple punctures per 2-cell

**Tasks**:
1. Extend `FeatureElement` to store multiple punctures:
   ```cpp
   struct MultiPunctureElement {
       Simplex simplex;  // 2-cell (triangle)
       std::vector<PuncturePoint> punctures;  // Up to 3
       // Each PuncturePoint: (lambda, barycentric_coords)
   };
   ```

2. Implement puncture sorting/ordering:
   - Order punctures by λ value? By spatial location?
   - Ensure consistent orientation

3. Update extraction logic:
   - `extract_simplex()` returns multiple elements per 2-cell
   - Track all punctures, not just first

4. Test with synthetic data:
   - Generate fields guaranteed to have 3 punctures per triangle
   - Verify all detected

**Deliverable**: Multi-puncture detection in 3D

---

### Phase 3: 3D Manifold Stitching (Weeks 5-7)

**Goal**: Reconstruct continuous curves from punctures

**Tasks**:
1. Design stitching algorithm:
   - Build graph: punctures as nodes, edges connect adjacent punctures
   - Use union-find or similar to build connected components
   - Handle ambiguity: multiple punctures → which connects to which?

2. Topology resolution:
   - Detect closed loops (periodic trajectories)
   - Detect bifurcations (where curves split/merge)
   - Handle boundaries properly

3. Output format:
   - Polylines (VTK VTP format)
   - Each curve as a separate trajectory

4. Validate against:
   - Synthetic vortex cores (known geometry)
   - Compare with ApproxPV results (should match closely)

**Deliverable**: Complete 3D spatial ExactPV with manifold stitching

---

### Phase 4: SoS for Degenerate Cases (Weeks 8-10)

**Goal**: Handle edge/vertex intersections rigorously

**Tasks**:
1. Research SoS for vector fields:
   - Review literature (Edelsbrunner, Harer, etc.)
   - Adapt to parallel vector problem
   - Derive perturbation scheme

2. Implement SoS predicates:
   - Replace floating-point comparisons with symbolic
   - Ensure consistent orientation

3. Test degenerate cases:
   - Generate fields where PV curves pass through vertices
   - Verify robustness

**Deliverable**: Robust 3D ExactPV with SoS

**Note**: This may extend into Phase 7 (research phase) if theory development is needed

---

### Phase 5: 4D Theory Development (Weeks 11-14) **[RESEARCH]**

**Goal**: Develop mathematical theory for 4D PV surfaces

**Tasks**:
1. Derive characteristic polynomials for 4D:
   - Extend from 3D tetrahedron to 4D pentatope
   - Verify with symbolic computation (Mathematica/Maple)

2. Inequality solver for 2D parameter space:
   - Input: Cubic rational constraints in (λ, t)
   - Output: Valid region in (λ, t) where mu_i ∈ [0,1]
   - Algorithm: Quantized intervals + adaptive refinement?

3. Surface parametrization:
   - Implicit: F(λ, t) = 0 (constraint satisfaction)
   - Parametric: (λ, t) → (mu_0, mu_1, mu_2, mu_3, mu_4)

4. Prove stitching consistency:
   - Show that surfaces match at shared 3-cells
   - Determine matching conditions for puncture curves

**Deliverable**: Theoretical framework + paper draft

---

### Phase 6: 4D Prototype Implementation (Weeks 15-18)

**Goal**: Implement basic 4D PV surface extraction

**Tasks**:
1. Implement 4D solver:
   ```cpp
   void characteristic_polynomials_pv_s4v3(const T V[5][3], const T W[5][3], ...);
   bool solve_pv_surface_s4v3(const T V[5][3], const T W[5][3], Surface& surface);
   ```

2. Surface extraction (initial approach):
   - Uniform grid in (λ, t) parameter space
   - Evaluate constraints at grid points
   - Extract isosurface (marching-squares-like in 2D)

3. Represent surface:
   - Triangle mesh in (λ, t) space
   - Map to 3D+time barycentric coords
   - Output as VTP (polygonal data)

4. Test with simple 4D fields:
   - Synthetic vortex tubes evolving in time
   - Known analytical solutions

**Deliverable**: Basic 4D PV surface extraction (prototype)

---

### Phase 7: 4D Stitching & Refinement (Weeks 19-22) **[RESEARCH]**

**Goal**: Stitch surface patches into continuous manifold

**Tasks**:
1. Implement face matching:
   - For each 3-cell (tetrahedron) boundary, extract puncture curves
   - Match curves across shared 3-cells
   - Build global surface

2. Adaptive refinement:
   - Detect high-curvature regions
   - Refine (λ, t) grid adaptively
   - Ensure topological consistency

3. Topology analysis:
   - Compute genus (if closed surface)
   - Detect boundaries
   - Handle multi-component surfaces

4. Validate:
   - Compare 3D spatial slices with Phase 3 results
   - Check temporal continuity

**Deliverable**: Complete 4D PV surface tracking with stitching

---

### Phase 8: Optimization & Production (Weeks 23-26)

**Goal**: Production-ready implementation with GPU support

**Tasks**:
1. Performance optimization:
   - Profile critical paths
   - Optimize cubic solver (vectorization, SIMD)
   - Reduce memory allocations

2. GPU acceleration:
   - CUDA kernels for puncture detection
   - Parallel stitching algorithm
   - Target: 10-100× speedup

3. Streaming mode:
   - Memory-efficient 4D processing
   - Process 2 timesteps at a time (like ApproxPV)

4. Documentation:
   - API reference
   - Theory documentation
   - Tutorial examples

**Deliverable**: Production ExactPV with GPU support

---

### Phase 9: Paper Writing (Weeks 24-28, overlapping)

**Goal**: Publish research paper on 4D ExactPV

**Structure**:
1. **Introduction**:
   - Motivation: Vortex dynamics in 4D spacetime
   - Challenges: Multiple punctures, stitching, degeneracies

2. **Background**:
   - Parallel vector criterion
   - Existing work (3D spatial)
   - Limitation to 3D

3. **Theory** (main contribution):
   - Characteristic polynomials for 4D
   - Cubic rational surfaces in pentatopes
   - Stitching consistency theorem
   - SoS extension to vector fields

4. **Implementation**:
   - Surface extraction algorithm
   - Adaptive refinement
   - GPU parallelization

5. **Results**:
   - Validation on synthetic data
   - Real-world applications (turbulence, MHD)
   - Performance benchmarks

6. **Discussion**:
   - Topology of PV surfaces
   - Comparison with approximate methods
   - Future work (higher dimensions?)

**Target Venue**: IEEE VIS, IEEE TVCG, ACM TOG, or Journal of Computational Physics

---

## Technical Specifications

### Data Structures

```cpp
namespace ftk2 {

// Puncture point on a 2-cell (triangle)
struct PuncturePoint {
    double lambda;                    // Scalar parameter
    std::array<double, 3> barycentric; // mu on triangle
    std::array<double, 3> coords;     // 3D spatial coordinates
    int edge_connectivity[2];         // Which edges does this connect?
};

// Multi-puncture element
struct MultiPunctureElement : public FeatureElement {
    std::vector<PuncturePoint> punctures;
};

// PV surface patch (4D)
struct PVSurfacePatch {
    Simplex simplex;                  // 4-simplex (pentatope)
    std::vector<Triangle> triangles;  // Surface triangulation
    // Each triangle: vertices in (lambda, t) parameter space
};

// Predicate for exact parallel vectors
template <int M, typename T>
struct ExactPVPredicate : public Predicate<M, T> {
    static constexpr int codimension = M;

    std::string vector_u_name = "u";  // First vector field
    std::string vector_v_name = "v";  // Second vector field

    // For 3D spatial (M=2):
    bool extract_it(const Simplex& s,
                   const T V[3][3], const T W[3][3],
                   std::vector<FeatureElement>& elements) const;

    // For 4D spacetime (M=2, but in 4D ambient):
    bool extract_surface(const Simplex& s,
                        const T V[5][3], const T W[5][3],
                        PVSurfacePatch& patch) const;
};

} // namespace ftk2
```

### Solver API

```cpp
namespace ftk2 {

// 3D spatial: Solve on triangle (returns multiple punctures)
template <typename T>
int solve_pv_triangle(const T V[3][3], const T W[3][3],
                     std::vector<PuncturePoint>& punctures,
                     T epsilon = std::numeric_limits<T>::epsilon());

// 3D spatial: Solve on tetrahedron (returns curve segment)
template <typename T>
bool solve_pv_tetrahedron(const T V[4][3], const T W[4][3],
                         CurveSegment& segment,
                         T epsilon = std::numeric_limits<T>::epsilon());

// 4D spacetime: Solve on pentatope (returns surface patch)
template <typename T>
bool solve_pv_pentatope(const T V[5][3], const T W[5][3],
                       PVSurfacePatch& patch,
                       int resolution = 16,  // Grid resolution in (λ, t)
                       T epsilon = std::numeric_limits<T>::epsilon());

} // namespace ftk2
```

---

## Success Metrics

### Implementation Milestones

| Phase | Milestone | Success Criteria |
|-------|-----------|------------------|
| 1 | 3D Foundation | Unit tests pass, extracts punctures |
| 2 | Multi-Puncture | Detects all 3 punctures per triangle |
| 3 | 3D Stitching | Reconstructs continuous curves |
| 4 | SoS | Handles degenerate cases correctly |
| 5 | 4D Theory | Paper draft, proofs verified |
| 6 | 4D Prototype | Extracts surfaces from pentatopes |
| 7 | 4D Stitching | Global surface manifold |
| 8 | Production | GPU 10-100× speedup, < 1GB memory |
| 9 | Paper | Accepted to major venue |

### Research Questions to Answer

1. ✅ or ❌ **Can we rigorously handle multiple punctures per face?**
   - Algorithm design
   - Proof of correctness

2. ✅ or ❌ **Can we extend SoS to vector fields?**
   - Theoretical framework
   - Practical implementation

3. ✅ or ❌ **Can we extract cubic rational surfaces from 4D cells?**
   - Parametrization
   - Completeness guarantee

4. ✅ or ❌ **Can we stitch 4D surfaces consistently?**
   - Matching conditions
   - Topological correctness

5. ✅ or ❌ **Is 4D ExactPV computationally feasible?**
   - Performance benchmarks
   - Memory requirements

---

## Comparison: ApproxPV vs ExactPV

| Aspect | ApproxPV (Fiber-based) | ExactPV (Cubic rational) |
|--------|----------------------|--------------------------|
| **Method** | W = U × V, track W₀=W₁=0 | Solve v × w = 0 directly |
| **Accuracy** | Approximate (filter by \|W₂\|) | Exact (analytical solution) |
| **Speed** | 10-100× faster | Baseline |
| **Complexity** | Simple (fiber tracking) | Complex (cubic solver) |
| **Multiple punctures** | N/A | ✅ Supported (research) |
| **4D support** | ✅ Works (fiber in 4D) | 🔬 Research needed |
| **Use case** | Interactive exploration | Final publication-quality results |

**Complementary**: ApproxPV for fast initial exploration, ExactPV for precise analysis

---

## Timeline Summary

| Weeks | Phase | Focus | Output |
|-------|-------|-------|--------|
| 1-2 | Foundation | Port 3D solver | Working prototype |
| 3-4 | Multi-Puncture | Handle multiple roots | Robust extraction |
| 5-7 | 3D Stitching | Manifold reconstruction | Continuous curves |
| 8-10 | SoS (partial) | Degenerate cases | Robust code |
| 11-14 | 4D Theory | Research | Paper draft |
| 15-18 | 4D Prototype | Implementation | Surface extraction |
| 19-22 | 4D Stitching | Research + impl | Complete 4D |
| 23-26 | Production | GPU, optimization | Release-ready |
| 24-28 | Paper | Writing | Submission |

**Total**: ~6-7 months for complete implementation + paper

**Minimum Viable Product (MVP)**: Phases 1-3 (3D spatial ExactPV) = 7 weeks

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| Multi-puncture stitching fails | Medium | High | Fall back to single-puncture approximation |
| 4D theory intractable | Low | High | Focus on 3D, publish 4D as future work |
| Performance too slow | Medium | Medium | GPU optimization, adaptive sampling |
| SoS extension impossible | Low | Medium | Use epsilon-based tolerance (less rigorous) |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| Insufficient novelty for publication | Low | High | Emphasize 4D contribution, thorough validation |
| Someone publishes similar work first | Very Low | High | Monitor arxiv, accelerate timeline if needed |
| Reviewers request extensive revisions | Medium | Medium | Build in 2-month buffer for revisions |

---

## Next Steps

1. **Immediate** (This week):
   - Read legacy FTK solver code thoroughly
   - Create skeleton `ExactPVPredicate` class
   - Set up unit test framework

2. **Week 1-2**:
   - Port 3D solver to FTK2
   - Implement basic extraction (single puncture)
   - Validate against legacy FTK

3. **Research Preparation**:
   - Review SoS literature (Edelsbrunner, Mücke)
   - Study cubic rational surfaces (algebraic geometry)
   - Identify potential advisors/collaborators for paper

4. **Collaboration**:
   - Consider involving topology experts for SoS
   - Reach out to application scientists (CFD, MHD) for validation data

---

## References

### Papers

1. **Guo et al.** "Exact Analytical Parallel Vectors Operator" (2021): [arXiv:2107.02708](https://arxiv.org/abs/2107.02708)
2. **Edelsbrunner & Mücke** "Simulation of Simplicity" (1990): ACM TOG
3. **Peikert & Roth** "The Parallel Vectors Operator" (1999): IEEE Vis
4. **Sujudi & Haimes** "Identification of Swirling Flow in 3D Vector Fields" (1995): AIAA

### Code References

- Legacy FTK: `/ftk/include/ftk/numeric/parallel_vector_solver3.hh`
- FTK2 Critical Points (SoS reference): `include/ftk2/core/predicate.hpp`
- FTK2 ApproxPV: `include/ftk2/numeric/cross_product.hpp`

### Algebraic Geometry

- **Cox, Little, O'Shea**: "Ideals, Varieties, and Algorithms" (Chapter on elimination theory)
- **Hartshorne**: "Algebraic Geometry" (Cubic surfaces)

---

## Conclusion

Exact Parallel Vector tracking with 4D generalization represents a significant research challenge worthy of publication. The combination of:

1. ✅ Practical importance (vortex dynamics)
2. ✅ Theoretical novelty (4D cubic rational surfaces)
3. ✅ Implementation complexity (multiple punctures, stitching, SoS)
4. ✅ Performance considerations (GPU, streaming)

makes this an ideal Ph.D.-level project that advances both theory and practice in scientific visualization and computational topology.

**This is not just an implementation task – it's a research project!**
