# ExactPV Implementation Status

**Date**: 2026-02-26
**Commit**: eaa57e5
**Status**: Work in Progress - Non-degenerate curves found, but not single closed curve

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

## Next Steps (Required for Validation)

### 1. Design Single-Curve Field
Create a vector field with ONE clean closed PV curve that:
- Passes through multiple tet interiors (not just boundaries)
- Forms a topologically closed loop
- Has known analytical form for verification

**Candidate approaches**:
- Perfect circular helix with carefully chosen radius/pitch
- Use irrational offsets to avoid mesh alignment
- Consider larger mesh (8×8×8 or 10×10×10) for better interior coverage
- Or: Use unstructured mesh to avoid regular triangulation artifacts

### 2. Implement Curve Stitching
Currently each tet produces an independent curve segment. Need to:
- Match curve endpoints at shared tet faces
- Connect segments into continuous curves
- Validate topological consistency

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
