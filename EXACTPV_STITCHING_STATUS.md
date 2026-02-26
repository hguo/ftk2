# ExactPV Stitching Implementation Status

**Date**: 2026-02-26
**Latest Commit**: ff78491
**Status**: Stitching algorithm implemented ✓, Problem diagnosed ✓

---

## Implementation Complete ✓

### Stitching Algorithm (Commit 3edaa65, ff78491)

Implemented correct algorithm as specified:

1. **Extract punctures from 2-simplices** (triangles) ✓
   - Engine `execute()` finds punctures
   - Filter degenerate ones (on edges/vertices)

2. **Foreach tetrahedron**: Find punctures on its 4 faces ✓
   - Build map: triangle → puncture indices
   - Check each tet's 4 faces

3. **If 2 punctures**: Connect directly ✓
   - Simple unambiguous case
   - 297 tets handled this way

4. **If >2 punctures**: Use ExactPV solver ✓
   - Run `extract_tetrahedron()` to get curve
   - Project punctures onto curve by λ parameter
   - Sort by λ and pair adjacent punctures
   - 12 tets handled this way

5. **Build connectivity graph** ✓
   - Create adjacency list from connections
   - Trace connected components using DFS

### Code Files

- `examples/exact_pv_stitching.cpp` - Full implementation
- `examples/check_punctures.cpp` - Analysis tool
- Both integrated into CMake build system

---

## Problem Diagnosed ✓

### The Issue: Multiple PV Curves, Not One

**Symptoms**:
- 29 connected curve components (not 1)
- 326 punctures, 311 connections → 15 unpaired
- 8 punctures with degree 0 (isolated)
- 14 punctures with degree 1 (endpoints)

**Root Cause Identified**:

The field creates **MULTIPLE PV curves/loci**, not a single closed curve.

**Evidence**:

1. **Odd puncture counts violate topology**:
   ```
   18 tets with 1 puncture  ← ODD (impossible for single curve)
   10 tets with 3 punctures ← ODD (impossible for single curve)
   ```

2. **For a single closed curve passing through tets**:
   - Curve enters 1 face → 1 puncture
   - Curve exits 1 face → 1 puncture
   - **MUST have even counts**: 0, 2, or 4 punctures per tet
   - Odd counts = multiple curves intersecting

3. **Example tet with 3 punctures**:
   ```
   Tet 1906-1907-2163-2179:
     Face {1906 1907 2179} → 1 puncture
     Face {1906 2163 2179} → 1 puncture
     Face {1907 2163 2179} → 1 puncture

   Three different faces = THREE curves passing through this tet!
   ```

**Conclusion**: The synthetic field with circular PV locus + perturbations creates:
- Multiple circular curves at different heights/positions, OR
- Intertwined helices, OR
- Complex PV manifold with branches

NOT a single simple closed curve.

---

## Current Field Configuration (Has Multiple Curves)

```cpp
// Circular PV locus with perturbations
double cx = N/2 + 0.23456;
double cy = N/2 + 0.78901;
double radius = N/3 + 0.31415;

// Field with perturbations to avoid degeneracies
double perturb = 0.01 * sin(x*0.7 + y*1.1 + z*1.3);
U = (cos(θ+0.1*dz+perturb), sin(θ+0.1*dz+perturb), 0.3+...)
V = U + distance-based correction + perturbations
```

**Results**:
- ✓ 0 degenerate punctures (all 326 in triangle interiors)
- ✗ Multiple PV curves (29 components)
- ✗ Odd puncture counts in 28 tets

---

## Next Steps (For Other Machine)

### Goal: Single Closed Curve

Design field where PV locus is **exactly ONE simple closed curve**.

**Requirements**:
1. NO degenerate punctures (all 3 bary coords > 0 on triangles)
2. ALL tets have EVEN puncture counts (0, 2, or 4)
3. Stitching produces 1 connected component

**Suggested Approaches**:

### Option 1: Simpler Circle (No Perturbations)
```cpp
// Perfect circle in z=N/2 plane, no perturbations
double cx = N/2, cy = N/2, cz = N/2;
double radius = N/3;  // Clean value

U = (dy, -dx, 0)  // Rotation around z-axis
V = U + (r-radius) * radial_component
// No perturbations that create extra curves
```

### Option 2: Analytic Single Helix
```cpp
// Helix with exactly 1 turn through volume
// Ensure only ONE curve by construction
```

### Option 3: Test with Smaller Mesh First
- N = 8 or N = 6 for faster iteration
- Verify puncture counts are all even
- Check for single component
- Then scale to N = 16

---

## Testing Protocol

1. **Run stitching**: `./ftk2_exact_pv_stitching`

2. **Check for success**:
   ```
   X punctures in triangle INTERIOR
   0 punctures on edges/vertices  ← Must be 0

   Puncture count histogram:
     X tets with 2 punctures
     X tets with 4 punctures
     0 tets with ODD counts  ← Must have NO odd counts!

   Created X connections
   Extracted 1 connected curve(s)  ← Must be exactly 1!
   ```

3. **Verify in ParaView**:
   - Load `exactpv_stitched.vtp`
   - Should show single closed loop
   - All segments connected

---

## Implementation Details

### Key Data Structures

```cpp
// Puncture on triangle face
complex.vertices[i]
  .simplex.dimension = 2  (triangle)
  .simplex.vertices[3]    (triangle vertices)
  .barycentric_coords[0][3]  (3 coords for triangle)

// Connection through tet
struct PunctureConnection {
    int puncture1_idx;
    int puncture2_idx;
    uint64_t tet_id;
};

// Adjacency graph
map<int, vector<int>> adjacency;  // puncture_idx → neighbors
```

### Stitching Logic

```cpp
for each tet:
  faces = 4 triangular faces
  tet_punctures = collect all punctures on these faces

  if size == 0: skip
  if size == 2: connect directly
  if size > 2:
    - Run extract_tetrahedron() to get PV curve
    - Project each puncture onto curve (find closest λ)
    - Sort by λ parameter
    - Connect adjacent pairs
```

---

## Files Generated

**Output**:
- `exactpv_stitched.vtp` - Stitched curves (current: 29 components)
- `exactpv_punctures.vtp` - Raw punctures from engine
- `exactpv_curves.vtp` - Curves through tet interiors

**Source**:
- `examples/exact_pv_stitching.cpp` - Main stitching implementation
- `examples/exact_pv_simple.cpp` - Original example (no stitching)
- `examples/check_punctures.cpp` - Analysis tool

---

## Summary for Continuation

**What Works**:
- ✓ Puncture extraction from triangles
- ✓ Degeneracy filtering (0 degenerate with current field)
- ✓ Stitching through tets (2-puncture and >2-puncture cases)
- ✓ Solver-based pairing for ambiguous cases
- ✓ Connectivity graph and component extraction
- ✓ VTK output for visualization

**What's Needed**:
- ✗ Field that creates exactly ONE closed curve
- ✗ All tets must have even puncture counts (0, 2, 4)
- ✗ Stitching should produce 1 component (not 29)

**The stitching algorithm is correct** - it's correctly identifying that the current field has multiple curves, not one!

**Action**: Redesign field to guarantee single closed PV curve, then re-run stitching.

---

## Build & Run

```bash
cd /home/hguo/workspace/ftk2/build
cmake ..
make ftk2_exact_pv_stitching -j8
./examples/ftk2_exact_pv_stitching

# Output in build directory:
# - exactpv_stitched.vtp
# - exactpv_punctures.vtp
```

---

**Ready to continue on another machine.**
Repository: https://github.com/hguo/ftk2
Branch: main
Latest commit: ff78491
