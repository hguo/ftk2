# ExactPV Implementation Status

**Date**: 2026-02-26
**Latest Commit**: 48e69d5
**Status**: ✓ 322 truly non-degenerate interior curves extracted
           ✗ 32 components instead of 1 (curve stitching needed)

---

## What Works ✓

1. **Triangle extraction (puncture points)**
   - Successfully extracts PV puncture points from 2-simplices
   - Manifold stitching connects adjacent puncture points
   - VTK output via `exactpv_punctures.vtp`

2. **Tetrahedron extraction (interior curves)**
   - Successfully extracts parametric PV curves from 3-simplices
   - Correctly identifies non-degenerate curves (≥2 non-zero barycentric coords)
   - VTK output via `exactpv_curves.vtp`
   - Curves are sampled at 20 points per segment for visualization

3. **VTK output fixes**
   - Fixed Time field bug: no longer uses z-coordinate for 3D spatial meshes
   - Proper handling of 3D spatial vs 4D spacetime meshes
   - Only writes curves file when non-degenerate curves exist

4. **Integration with SimplicialEngine**
   - ExactPVPredicate properly integrated with engine
   - Multi-component field mode working (6-component [ux,uy,uz,vx,vy,vz])
   - Physical coordinates correctly computed for curve visualization

---

## Current Test Case (`exact_pv_simple.cpp`)

**Field Design**:
```
U = radial field (dx/r, dy/r, z/N)
V = U + perpendicular perturbation
Perturbation = (dy·(r²-1)·0.1, -dx·(r²-1)·0.1, 0)
```

**PV Locus**: Circle at radius ~1 in x-y plane

**Results**:
- Mesh: 6×6×6 regular simplicial
- Puncture points: 223
- Non-degenerate curves: 6 out of 14 total
- Connected components: 45

---

## Critical Issue ✗

**Problem**: Does NOT produce a single closed curve

The current implementation finds multiple curve fragments instead of one continuous closed curve. This makes verification difficult because:
1. Cannot confirm curve connectivity across tetrahedra
2. Cannot verify topological correctness (closed loop vs open segments)
3. Difficult to validate against known analytical solution

**Root Cause**:
- Field design produces fragmented PV locus, not single clean curve
- Regular simplicial mesh triangulation causes geometric PV features (lines, circles) to intersect tet faces/edges preferentially over interiors
- Discovered that simple geometric curves (e.g., vertical line at x=1.6, y=1.6) pass through tet boundaries, not interiors

---

## Best Result Achieved (N=16 mesh)

**Configuration**:
- Mesh: 16×16×16 regular simplicial (10,656 tets)
- Field: Synthetic circular PV locus with distance-based perturbation
- Circle: radius = N/3 ≈ 5.33, center offset (0.1, 0.1) from grid

**Results**:
- ✓ 327 puncture points detected
- ✓ 322 out of 324 curves pass through tet INTERIOR (99.4%)
  - Definition: All 4 barycentric coordinates > 0 at some point
  - Verified: Sample points show interior passage
- ✓ 2 curves on tet boundaries (faces/edges) correctly identified as degenerate
- ✓ All curves have valid lambda ranges (not collapsed)
- ✓ Physical coordinates vary smoothly along curves
- ✗ 32 disconnected components (not single curve - stitching needed)

**Why Not One Curve?**

Tested configurations:
1. N=6: 6 curves, 45 components
2. N=8: 28 curves, 13 components
3. N=12: 257 curves, 24 components
4. N=16: 324 curves, 32 components

**Finding**: Higher resolution increases curve count but doesn't reduce fragmentation. The issue is geometric: regular simplicial mesh triangulation creates systematic sampling gaps that fragment the circular PV locus, regardless of resolution.

**Fundamental Issue**: The regular mesh triangulation pattern doesn't align with circular/curved geometry, causing the continuous circular PV locus to be detected as multiple disconnected segments.

---

## Critical Fix (Commit 48e69d5)

**Previous Misunderstanding**:
- Incorrectly defined "non-degenerate" as ≥2 non-zero P[i] polynomials
- This counted curves on tet faces/edges as "non-degenerate"
- Claimed 324/324 curves were non-degenerate (incorrect!)

**Correct Definition**:
- **Non-degenerate** = curve passes through tet INTERIOR
- **Requires**: All 4 barycentric coordinates μ₀, μ₁, μ₂, μ₃ > 0 at some point
- If any coordinate = 0, the curve is on a face (1 zero), edge (2 zeros), or vertex (3 zeros)

**Verification Example**:
```
Sample 0: bary=(0.0128631, ~0, 0.150118, 0.837019)  ← on face (entering)
Sample 1: bary=(0.0121483, 0.00277572, 0.148122, 0.837)  ← IN INTERIOR ✓
Sample 2: bary=(0.0114379, 0.00553312, 0.14614, 0.837)   ← IN INTERIOR ✓
```

**Corrected Results**: 322/324 curves (99.4%) truly pass through tet interiors.

---

## Next Steps (Required for Validation)

### 1. Implement Curve Stitching Algorithm (REQUIRED)

**Problem**: Current implementation extracts independent curve segments per tetrahedron. These segments are not connected into continuous curves across tet boundaries.

**Solution needed**:
- For each pair of adjacent tets sharing a face:
  - Check if their PV curve segments intersect the shared face
  - If yes, match the intersection points and connect segments
  - Build connected curve graph
- Result: N curve segments → M connected curves (ideally M=1 for circular locus)

**Implementation approach**:
1. Store curve segment endpoints (entry/exit points on tet faces)
2. Build adjacency graph of tets
3. For adjacent tets, check if curve endpoints are close (<ε)
4. Connect matching endpoints into polylines
5. Close loops where endpoints connect

**Expected result**: 324 segments → ~1 closed curve

### 2. Alternative: Use Unstructured Mesh
If curve stitching proves difficult, consider:
- Generate unstructured mesh with elements aligned to PV locus
- Or: Use synthetic test case where PV curve is straight line (avoids stitching)

### 3. Add Analytical Verification
Once single curve works:
- Compare extracted curve to analytical PV locus
- Measure distance error
- Verify curve completeness (no gaps)

---

## Technical Details

### File Locations
- Main example: `examples/exact_pv_simple.cpp`
- ExactPV predicate: `include/ftk2/core/predicate.hpp`
- Parallel vector solver: `include/ftk2/numeric/parallel_vector_solver.hpp`
- VTK utilities: `include/ftk2/utils/vtk.hpp`
- TODO/research questions: `TODO_EXACTPV.md`

### Key Data Structures
```cpp
struct PVCurveSegment {
    std::array<Polynomial<double, 3>, 4> P;  // Barycentric coords μᵢ(λ)
    Polynomial<double, 3> Q;                  // Common denominator
    std::array<std::array<double, 3>, 4> tet_vertices;  // For physical coords
    double lambda_min, lambda_max;            // Parameter range
    int simplex_id;                           // Tet identifier
};
```

### Degeneracy Detection
Curves with <2 non-zero P polynomials are degenerate (collapse to points/edges).
Current test: 6/14 curves non-degenerate = 43% success rate.

---

## References

- Theory: Peikert & Roth (1999) - "The Parallel Vectors Operator"
- Implementation based on: Cubic rational barycentric parametrization
- Uses: Sylvester determinant + cubic polynomial solver

---

## Session Notes

- Attempted ~10 different field configurations before finding non-degenerate curves
- Learned: vertical lines at non-integer grid positions still hit tet boundaries due to regular mesh triangulation
- Discovery: Regular simplicial meshes have structured triangulation that affects geometric feature detection
- VTK Time field bug discovered when user noticed z-coordinates appearing as time values

---

**To continue work**: Start with field design to create one single closed PV curve through tet interiors.
