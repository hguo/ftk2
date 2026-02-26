# FTK2 Feature Gap Analysis

## Overview

This document compares FTK (legacy) and FTK2 to identify missing features that need to be implemented in FTK2.

**Last updated**: February 2026

---

## Feature Comparison Matrix

| Feature Type | FTK Status | FTK2 Status | Priority | Notes |
|-------------|-----------|-------------|----------|-------|
| **Critical Points** | ✅ 2D/3D Regular/Unstructured | ✅ 2D/3D Regular/Unstructured/Extruded | ✅ **DONE** | Core feature, fully implemented with SoS |
| **Levelsets/Contours** | ✅ 2D/3D Regular | ✅ 2D/3D via ContourPredicate | ✅ **DONE** | Isosurface tracking |
| **Fibers (Isosurface Intersections)** | ✅ Implicit | ✅ FiberPredicate (m=2) | ✅ **DONE** | Two scalar field intersection |
| **Critical Lines** | ✅ 3D Regular/Unstructured | ❌ **MISSING** | 🎯 **HIGH** | Ridge/valley lines in 3D scalar fields |
| **Parallel Vectors (ExactPV)** | ✅ 2D/3D with cubic solver | ❌ **MISSING** | 🎯 **HIGH** | v × w = 0, vortex cores |
| **Magnetic Flux Vortices (TDGL)** | ✅ 3D Regular | ❌ **MISSING** | 🎯 **MEDIUM-HIGH** | Superconductor vortices, phase-based |
| **Sujudi-Haimes Vortices** | ✅ 3D Regular | ❌ **MISSING** | 🎯 **MEDIUM** | Vortex cores via eigenanalysis |
| **Levy-Degani-Seginer Vortices** | ✅ 3D Regular | ❌ **MISSING** | 🎯 **MEDIUM** | Alternative vortex detection |
| **Ridge/Valley Lines** | ✅ 3D Regular | ❌ **MISSING** | 🎯 **MEDIUM** | Extremal curves in scalar fields |
| **Lagrangian Particle Tracing** | ✅ Regular/MPAS | ❌ **MISSING** | 🎯 **MEDIUM-HIGH** | Pathline integration |
| **Connected Components** | ✅ Generic tracker | ✅ Implicit (union-find) | ✅ **DONE** | Via union-find in engine |
| **Threshold Tracker** | ✅ Generic | ⚠️ **PARTIAL** | 🎯 **LOW** | Threshold-based features |
| **Feature Flow Fields (FFF)** | ✅ Stable FFF | ❌ **MISSING** | 🎯 **LOW-MEDIUM** | Streamline-based tracking |
| **XGC Blob Filaments** | ✅ 2D/3D | ❌ **MISSING** | 🎯 **LOW** | Application-specific (fusion) |
| **MPAS Ocean** | ✅ Particle tracer | ❌ **MISSING** | 🎯 **LOW** | Application-specific (climate) |

---

## Detailed Feature Descriptions

### ✅ **Already in FTK2**

#### 1. Critical Points
- **Status**: Fully implemented with multi-level SoS
- **Location**: `include/ftk2/core/predicate.hpp` → `CriticalPointPredicate`
- **Features**:
  - Gradient fields (Hessian): min/max/1-saddle/2-saddle
  - Non-gradient fields (Jacobian): source/sink/saddle/spiral/center
  - 2D/3D spatial, 3D/4D spacetime
  - CPU and GPU (CUDA)
  - Regular, unstructured, extruded meshes

#### 2. Levelsets/Contours
- **Status**: Implemented
- **Location**: `include/ftk2/core/predicate.hpp` → `ContourPredicate`
- **Features**:
  - Isosurface tracking (codimension m=1)
  - 2D/3D spatial, 3D/4D spacetime

#### 3. Fibers (Isosurface Intersections)
- **Status**: Implemented
- **Location**: `include/ftk2/core/predicate.hpp` → `FiberPredicate`
- **Features**:
  - Intersection of two scalar fields (codimension m=2)
  - 3D/4D spacetime

---

### ❌ **Missing from FTK2** (Priority Order)

#### 1. **Parallel Vectors (ExactPV)** 🎯 **HIGH PRIORITY**

**Description**: Tracks locations where two vector fields v and w are parallel (v × w = 0).

**Applications**:
- Vortex core lines in 3D flows
- Magnetic field topology
- Vector field singularities

**Mathematical Foundation**:
- **Paper**: [arXiv:2107.02708](https://arxiv.org/abs/2107.02708) - Exact Analytical Parallel Vectors
- In piecewise-linear fields, parallel vectors form **cubic rational curves** within tetrahedra
- A single tet can have **multiple curve segments** (manifold generator)
- Requires analytical cubic rational solver

**Implementation Requirements**:
```cpp
// Predicate for parallel vectors
template <int M, typename T>
struct ParallelVectorPredicate : public Predicate<2, T> {  // codimension m=2
    static constexpr int codimension = 2;

    // Check if v × w = 0 on simplex
    bool check(const Simplex& s,
               const std::vector<const ftk::ndarray<T>*>& arrays,
               FeatureElement& element) const override;

private:
    // Solve cubic rational curve: v(x,y,z,t) × w(x,y,z,t) = 0
    // Returns multiple segments per tet
    std::vector<CurveSegment> solve_cubic_rational(const T V[][3], const T W[][3]);
};
```

**Challenges**:
- Cubic solver is complex (see `ftk/numeric/parallel_vector_solver3.hh`)
- Multiple solutions per simplex (need manifold stitching)
- Requires robust numerical handling

**References in FTK**:
- `include/ftk/numeric/parallel_vector_solver2.hh`
- `include/ftk/numeric/parallel_vector_solver3.hh`
- `tests/test_parallel_vectors.cpp`

**Estimated effort**: 4-6 weeks

---

#### 2. **Critical Lines (Ridge/Valley Lines)** 🎯 **HIGH PRIORITY**

**Description**: Tracks ridges (local maxima along a curve) and valleys (local minima along a curve) in 3D scalar fields.

**Applications**:
- Feature extraction in medical imaging
- Terrain analysis
- Turbulence structures

**Mathematical Foundation**:
- Critical lines are defined as intersections of two isosurfaces in 3D
- Alternatively: extremal curves of scalar field (∇s perpendicular to curve tangent)
- Codimension m=2 in 3D spatial (→ curves in 3D)
- Codimension m=2 in 4D spacetime (→ surfaces in 4D)

**Implementation Requirements**:
```cpp
// Predicate for critical lines (ridge/valley)
template <typename T>
struct CriticalLinePredicate : public Predicate<2, T> {  // codimension m=2
    static constexpr int codimension = 2;

    // Check simplex for ridge/valley
    bool check(const Simplex& s,
               const std::vector<const ftk::ndarray<T>*>& arrays,
               FeatureElement& element) const override;

private:
    // Classify as ridge vs valley
    enum class LineType { Ridge, Valley };
    LineType classify(const T hessian[3][3]);
};
```

**References in FTK**:
- `include/ftk/filters/critical_line_tracker.hh`
- `include/ftk/filters/critical_line_tracker_3d_regular.hh`
- `include/ftk/filters/ridge_valley_tracker_3d_regular.hh`

**Estimated effort**: 3-4 weeks

---

#### 3. **Magnetic Flux Vortices (TDGL)** 🎯 **MEDIUM-HIGH PRIORITY**

**Description**: Tracks magnetic flux vortices in superconductors simulated with Time-Dependent Ginzburg-Landau (TDGL) equations.

**Applications**:
- Superconductor simulations
- Bose-Einstein condensates
- Complex-valued field topology

**Mathematical Foundation**:
- **Paper**: [Guo et al. 2017, IEEE TVCG](https://ieeexplore.ieee.org/document/8031581)
- Based on complex order parameter ψ = ρe^(iφ)
- Vortices occur where phase φ has 2π winding around a point
- Extraction on 2-simplices (triangles/tets) via phase analysis

**Input Data**:
- Real part (re)
- Imaginary part (im)
- Or: magnitude (ρ) and phase (φ)

**Implementation Requirements**:
```cpp
// Predicate for TDGL vortices
template <typename T>
struct TDGLVortexPredicate : public Predicate<2, T> {  // codimension m=2 (?)
    static constexpr int codimension = 2;

    bool check(const Simplex& s,
               const std::vector<const ftk::ndarray<T>*>& arrays,
               FeatureElement& element) const override;

private:
    // Compute winding number around edge
    int compute_winding_number(const T re[], const T im[]);

    // Phase unwrapping
    T unwrap_phase(T phi0, T phi1);
};
```

**Challenges**:
- Phase unwrapping and branch cuts
- Winding number computation
- Application-specific (limited adoption)

**References in FTK**:
- `include/ftk/filters/tdgl_vortex_tracker.hh`
- `include/ftk/filters/tdgl_vortex_tracker_3d_regular.hh`
- `include/ftk/io/tdgl.hh`

**Estimated effort**: 3-4 weeks

---

#### 4. **Lagrangian Particle Tracing** 🎯 **MEDIUM-HIGH PRIORITY**

**Description**: Integrates pathlines of particles in time-varying vector fields.

**Applications**:
- Ocean/atmosphere particle tracking
- Fluid transport analysis
- Pollutant dispersion

**Mathematical Foundation**:
- Solve ODE: dx/dt = v(x, t)
- Runge-Kutta integration (RK1, RK4)
- Interpolation across simplicial mesh

**Implementation Requirements**:
```cpp
// Particle tracer
template <typename T>
class ParticleTracer {
public:
    // Seed particles
    void seed_particles(const std::vector<std::array<T, 3>>& positions);

    // Integrate forward in time
    void integrate(const std::map<std::string, ftk::ndarray<T>>& velocity_field,
                   int num_steps, T dt);

    // Get trajectories
    std::vector<Trajectory> get_trajectories();

private:
    enum Integrator { RK1, RK4 };
    Integrator integrator_ = RK4;

    // Interpolate velocity at arbitrary point
    bool eval_velocity(const std::array<T, 3>& x, T t, std::array<T, 3>& v);
};
```

**Challenges**:
- Interpolation across unstructured meshes
- Particle seeding strategies
- Checkpointing for long integrations

**References in FTK**:
- `include/ftk/filters/particle_tracer.hh`
- `include/ftk/filters/particle_tracer_regular.hh`
- `include/ftk/filters/particle_tracer_mpas_ocean.hh`

**Estimated effort**: 4-5 weeks

---

#### 5. **Sujudi-Haimes Vortex Cores** 🎯 **MEDIUM PRIORITY**

**Description**: Identifies vortex core lines in 3D vector fields using the Sujudi-Haimes criterion.

**Applications**:
- Vortex identification in CFD
- Turbulence analysis

**Mathematical Foundation**:
- Vortex cores where: v · (∇ × v) = 0 and complex eigenvalues of Jacobian
- Combines parallel vectors (v ⊥ ω) with eigenvalue analysis
- Builds on critical line tracking

**Implementation**:
- Extends `ParallelVectorPredicate` with eigenvalue check
- Requires Jacobian computation

**References in FTK**:
- `include/ftk/filters/sujudi_haimes_tracker_3d_regular.hh`

**Estimated effort**: 2-3 weeks (after parallel vectors implemented)

---

#### 6. **Levy-Degani-Seginer Vortex Cores** 🎯 **MEDIUM PRIORITY**

**Description**: Alternative vortex core definition using eigenvectors.

**Applications**:
- Vortex identification (alternative to Sujudi-Haimes)

**Mathematical Foundation**:
- Different eigenvalue-based criterion
- Similar to Sujudi-Haimes but different classification

**References in FTK**:
- `include/ftk/filters/levy_degani_seginer_tracker_3d_regular.hh`

**Estimated effort**: 2-3 weeks

---

#### 7. **Feature Flow Fields (Stable FFF)** 🎯 **LOW-MEDIUM PRIORITY**

**Description**: Constructs a vector field whose streamlines are feature trajectories.

**Applications**:
- Robust tracking in noisy data
- Smooth trajectory interpolation

**Mathematical Foundation**:
- **Paper**: [Theisel et al. 2010](https://ieeexplore.ieee.org/document/5487517)
- Creates "stable" vector field that converges to true features
- Alternative to zero-crossing methods

**References**:
- Mentioned in `docs/RESEARCH_NOTES.md`
- No explicit FTK implementation

**Estimated effort**: 5-6 weeks (research-heavy)

---

#### 8. **Application-Specific Features** 🎯 **LOW PRIORITY**

##### XGC Blob Filaments
- Fusion plasma simulation (XGC code)
- Blob tracking in toroidal geometry
- **FTK**: `include/ftk/filters/xgc_blob_filament_tracker.hh`

##### MPAS Ocean
- Ocean circulation (MPAS-Ocean model)
- Particle tracing on spherical/unstructured meshes
- **FTK**: `include/ftk/filters/mpas_ocean_tracker.hh`

**Decision**: Low priority unless specific user demand. Focus on general features first.

**Estimated effort**: 2-3 weeks each (if needed)

---

## Implementation Roadmap

### Phase 1: Core Feature Gaps (Next 3-6 months)

**Priority 1: Parallel Vectors (ExactPV)**
- Timeline: 4-6 weeks
- Dependency: None
- Impact: HIGH (vortex cores, many applications)
- Implementation:
  1. Port cubic rational solver from FTK
  2. Create `ParallelVectorPredicate`
  3. Manifold stitching for multiple segments
  4. Tests with synthetic data
  5. Examples (tornado, vortex ring)

**Priority 2: Critical Lines**
- Timeline: 3-4 weeks
- Dependency: None
- Impact: HIGH (medical imaging, terrain analysis)
- Implementation:
  1. Create `CriticalLinePredicate`
  2. Ridge vs valley classification
  3. Tests with synthetic mountains/valleys
  4. Integration with existing engine

**Priority 3: Lagrangian Particles**
- Timeline: 4-5 weeks
- Dependency: None (but benefits from stream integration)
- Impact: MEDIUM-HIGH (ocean/climate applications)
- Implementation:
  1. RK1/RK4 integrators
  2. Velocity interpolation (simplicial)
  3. Particle seeding strategies
  4. Checkpointing for long runs
  5. MPAS mesh support

**Priority 4: Magnetic Flux Vortices (TDGL)**
- Timeline: 3-4 weeks
- Dependency: None
- Impact: MEDIUM-HIGH (superconductor community)
- Implementation:
  1. Phase winding number computation
  2. Create `TDGLVortexPredicate`
  3. Tests with TDGL simulation data
  4. Documentation for physics users

### Phase 2: Vortex Analysis (Months 4-6)

**Priority 5: Sujudi-Haimes Vortices**
- Timeline: 2-3 weeks
- Dependency: Parallel vectors
- Implementation: Extend PV with eigenvalue analysis

**Priority 6: Levy-Degani-Seginer Vortices**
- Timeline: 2-3 weeks
- Dependency: Parallel vectors

### Phase 3: Advanced Features (Months 6-12)

**Priority 7: Feature Flow Fields**
- Timeline: 5-6 weeks
- Research-intensive

**Priority 8: Application-Specific**
- As needed based on user demand

---

## Integration with High-Level API

All new features should be exposed via the high-level API (see `HIGH_LEVEL_API_DESIGN.md`).

### YAML Configuration Examples

**Parallel Vectors**:
```yaml
tracking:
  feature: parallel_vectors
  dimension: 3
  input:
    type: vector_pair
    variables:
      v: [u, v, w]
      w: [bx, by, bz]
  execution:
    backend: cuda
```

**Critical Lines**:
```yaml
tracking:
  feature: critical_lines
  dimension: 3
  input:
    type: scalar
    variables: [elevation]
  options:
    line_type: ridge  # or valley
```

**Particle Tracing**:
```yaml
tracking:
  feature: particles
  dimension: 3
  input:
    type: vector
    variables: [u, v, w]
  options:
    integrator: rk4
    num_steps: 1000
    dt: 0.01
    seeding:
      type: grid
      stride: [4, 4, 4]
```

**TDGL Vortices**:
```yaml
tracking:
  feature: tdgl_vortex
  dimension: 3
  input:
    type: complex
    variables:
      real: psi_re
      imag: psi_im
```

---

## Testing Strategy

For each new feature:

1. **Unit tests**: Synthetic data with known ground truth
2. **Integration tests**: Real scientific data
3. **Performance tests**: Benchmark against FTK (if applicable)
4. **Regression tests**: Ensure correctness over time

### Synthetic Test Cases to Create

- **Parallel Vectors**: ABC flow, vortex ring, tornado
- **Critical Lines**: Gaussian mountain, saddle surface
- **Particles**: Steady vortex, time-varying flow
- **TDGL**: Analytical vortex lattice

---

## Documentation Requirements

For each new feature:

- [ ] Mathematical foundation (papers, equations)
- [ ] Implementation notes (algorithms, edge cases)
- [ ] YAML configuration reference
- [ ] C++ API usage examples
- [ ] Python API usage examples
- [ ] Performance characteristics
- [ ] Known limitations

---

## References

### FTK Feature Implementations
- Critical Points: `ftk/filters/critical_point_tracker*.hh`
- Contours: `ftk/filters/contour_tracker*.hh`
- Critical Lines: `ftk/filters/critical_line_tracker*.hh`
- Parallel Vectors: `ftk/numeric/parallel_vector_solver*.hh`
- TDGL: `ftk/filters/tdgl_vortex_tracker*.hh`
- Sujudi-Haimes: `ftk/filters/sujudi_haimes_tracker_3d_regular.hh`
- Particles: `ftk/filters/particle_tracer*.hh`

### Research Papers
- ExactPV: [arXiv:2107.02708](https://arxiv.org/abs/2107.02708)
- TDGL: [Guo et al. 2017, IEEE TVCG](https://ieeexplore.ieee.org/document/8031581)
- Stable FFF: [Theisel et al. 2010](https://ieeexplore.ieee.org/document/5487517)
- Simplicial Spacetime: [arXiv:2011.08697](https://arxiv.org/abs/2011.08697)

---

## Next Steps

1. **Prioritize**: Get user feedback on priority order
2. **Validate**: Ensure mathematical foundations are clear
3. **Implement**: Start with Parallel Vectors (highest ROI)
4. **Document**: Create detailed design docs for each feature
5. **Test**: Comprehensive test suite for each

**Question for discussion**: Should we implement all missing features, or focus on the most commonly used ones (CP, PV, particles)?
