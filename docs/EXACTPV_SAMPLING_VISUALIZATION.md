# ExactPV Sampling and Visualization Strategy

## Overview

This document addresses the critical requirements for **sampling and visualizing** exact parallel vector (ExactPV) curves and surfaces, which are **cubic rational** and require nonlinear adaptive sampling.

**Key Requirements**:
1. Extract and visualize interior points inside tetrahedra (3D) and 4-cells/pentatopes (4D)
2. Identify critical points on PV curves/surfaces (extrema, saddles of vector fields)
3. Adaptive nonlinear sampling to capture cubic rational geometry
4. Efficient storage: parametric representation + deferred sampling

---

## Problem Statement

### Challenge: Cubic Rational Geometry

**3D Spatial (Tetrahedron)**:
- PV curve parametrized by λ ∈ [λ_min, λ_max]
- Barycentric coords: mu_i(λ) = P_i(λ) / Q(λ) (rational functions of degree 3)
- Curve is NOT piecewise-linear → uniform sampling misses features

**4D Spacetime (Pentatope)**:
- PV surface parametrized by (λ, t) ∈ [λ_min, λ_max] × [t_min, t_max]
- Barycentric coords: mu_i(λ, t) = P_i(λ, t) / Q(λ, t) (rational surfaces of degree 3)
- Surface is NOT planar → uniform grid sampling misses high-curvature regions

**Visualization Requirements**:
1. **Accurate geometry**: Must capture cubic curvature
2. **Critical features**: Must sample at extrema, saddles, inflection points
3. **Efficiency**: Minimize samples while maintaining quality
4. **Flexibility**: Support both direct rendering and offline visualization

---

## Two-Stage Approach

### Stage 1: Parametric Storage (Extraction Time)

**Goal**: Store analytical representation, defer sampling to visualization time

**Storage Format**:

```cpp
// 3D Curve Segment (in tetrahedron)
struct PVCurveSegment {
    Simplex simplex;              // Tetrahedron

    // Parametric representation: lambda ∈ [lambda_min, lambda_max]
    double lambda_min, lambda_max;

    // Rational barycentric coordinates: mu_i(λ) = P_i(λ) / Q(λ)
    Polynomial<double, 3> P[4];   // Degree 3 numerators (4 barycentric coords)
    Polynomial<double, 3> Q;      // Degree 3 denominator (shared)

    // Critical points on curve (computed during extraction)
    std::vector<CriticalPoint1D> critical_points;
};

// 4D Surface Patch (in pentatope)
struct PVSurfacePatch {
    Simplex simplex;              // Pentatope

    // Parametric representation: (lambda, t) ∈ domain
    double lambda_min, lambda_max;
    double t_min, t_max;

    // Rational barycentric coordinates: mu_i(λ, t) = P_i(λ, t) / Q(λ, t)
    BivarPolynomial<double, 3> P[5];  // Degree 3 in (λ, t)
    BivarPolynomial<double, 3> Q;     // Shared denominator

    // Critical points on surface (computed during extraction)
    std::vector<CriticalPoint2D> critical_points;
};
```

**Advantages**:
- ✅ Exact representation (no sampling error)
- ✅ Compact storage (O(1) per segment/patch)
- ✅ Flexible: Resample at any resolution later
- ✅ Enables analytical queries (intersection, distance, etc.)

### Stage 2: Adaptive Sampling (Visualization Time)

**Goal**: Generate high-quality polyline/mesh for rendering

**Approach**: Adaptive refinement based on curvature

---

## Adaptive Sampling Algorithms

### 3D Curves: Curvature-Adaptive Polylines

**Algorithm**:

```python
def adaptive_sample_curve(segment: PVCurveSegment, epsilon: float) -> List[Point3D]:
    """
    Adaptively sample cubic rational curve.

    Args:
        segment: Parametric curve representation
        epsilon: Maximum allowed deviation from true curve

    Returns:
        List of 3D points forming polyline
    """
    samples = []

    # Start with critical points (extrema, inflections)
    lambda_samples = [segment.lambda_min, segment.lambda_max]
    for cp in segment.critical_points:
        lambda_samples.append(cp.lambda)
    lambda_samples.sort()

    # Refine each sub-interval
    for i in range(len(lambda_samples) - 1):
        lambda_a = lambda_samples[i]
        lambda_b = lambda_samples[i + 1]

        samples.extend(refine_interval(segment, lambda_a, lambda_b, epsilon))

    return samples

def refine_interval(segment, lambda_a, lambda_b, epsilon):
    """Recursively refine interval [lambda_a, lambda_b]."""
    # Evaluate curve at endpoints and midpoint
    p_a = evaluate_curve(segment, lambda_a)
    p_b = evaluate_curve(segment, lambda_b)
    lambda_mid = (lambda_a + lambda_b) / 2
    p_mid = evaluate_curve(segment, lambda_mid)

    # Linear approximation error: distance from midpoint to line segment
    p_mid_linear = (p_a + p_b) / 2
    error = ||p_mid - p_mid_linear||

    if error < epsilon:
        # Accept linear approximation
        return [p_a, p_b]
    else:
        # Refine recursively
        left = refine_interval(segment, lambda_a, lambda_mid, epsilon)
        right = refine_interval(segment, lambda_mid, lambda_b, epsilon)
        return left + right[1:]  # Avoid duplicating midpoint

def evaluate_curve(segment, lambda):
    """Evaluate curve at parameter lambda."""
    # Compute barycentric coords: mu_i = P_i(lambda) / Q(lambda)
    Q_val = segment.Q.evaluate(lambda)
    mu = [segment.P[i].evaluate(lambda) / Q_val for i in range(4)]

    # Interpolate 3D position
    vertices = segment.simplex.vertices  # 4 vertices of tetrahedron
    p = sum(mu[i] * vertices[i] for i in range(4))
    return p
```

**Curvature-Based Refinement**:

```cpp
template <typename T>
double compute_curvature(const PVCurveSegment& segment, T lambda) {
    // First derivative: tangent vector
    auto T_vec = differentiate_curve(segment, lambda);

    // Second derivative: curvature vector
    auto K_vec = differentiate_curve_twice(segment, lambda);

    // Curvature: ||T × K|| / ||T||³
    auto cross = cross_product(T_vec, K_vec);
    double kappa = norm(cross) / pow(norm(T_vec), 3);

    return kappa;
}

// Adaptive sampling prioritizes high-curvature regions
std::vector<double> adaptive_lambda_samples(const PVCurveSegment& segment, int max_samples) {
    std::vector<double> samples;

    // Start with uniform sampling
    for (int i = 0; i <= max_samples; ++i) {
        double lambda = lerp(segment.lambda_min, segment.lambda_max, i / double(max_samples));
        samples.push_back(lambda);
    }

    // Refine based on curvature
    for (int iter = 0; iter < 5; ++iter) {
        std::vector<double> refined;
        for (int i = 0; i < samples.size() - 1; ++i) {
            double lambda_a = samples[i];
            double lambda_b = samples[i + 1];
            double lambda_mid = (lambda_a + lambda_b) / 2;

            double kappa = compute_curvature(segment, lambda_mid);

            refined.push_back(lambda_a);
            if (kappa > threshold) {
                refined.push_back(lambda_mid);  // Add midpoint in high-curvature region
            }
        }
        refined.push_back(samples.back());
        samples = refined;
    }

    return samples;
}
```

---

### 4D Surfaces: Adaptive Triangulation

**Algorithm**:

```python
def adaptive_sample_surface(patch: PVSurfacePatch, epsilon: float) -> TriangleMesh:
    """
    Adaptively sample cubic rational surface.

    Args:
        patch: Parametric surface representation
        epsilon: Maximum allowed deviation

    Returns:
        Triangle mesh in 3D+time
    """
    # Start with coarse grid in (λ, t) parameter space
    grid = UniformGrid2D(patch.lambda_min, patch.lambda_max,
                         patch.t_min, patch.t_max,
                         resolution=16)

    # Evaluate surface at grid points
    vertices = []
    for (lambda_i, t_j) in grid.points():
        p = evaluate_surface(patch, lambda_i, t_j)
        vertices.append(p)

    # Build initial triangulation (Delaunay in parameter space)
    triangles = delaunay_triangulation(grid.points())

    # Refine triangles based on curvature
    for iter in range(max_iterations):
        triangles_to_refine = []
        for tri in triangles:
            curvature = estimate_surface_curvature(patch, tri)
            if curvature > threshold:
                triangles_to_refine.append(tri)

        if not triangles_to_refine:
            break

        # Subdivide high-curvature triangles
        triangles = subdivide_triangles(triangles_to_refine)

    return TriangleMesh(vertices, triangles)

def evaluate_surface(patch, lambda_val, t_val):
    """Evaluate surface at parameter (lambda, t)."""
    Q_val = patch.Q.evaluate(lambda_val, t_val)
    mu = [patch.P[i].evaluate(lambda_val, t_val) / Q_val for i in range(5)]

    # Interpolate in 5-vertex pentatope
    vertices = patch.simplex.vertices
    p = sum(mu[i] * vertices[i] for i in range(5))
    return p  # 4D point (x, y, z, t)
```

**Curvature Estimation for Surfaces**:

```cpp
template <typename T>
double estimate_surface_curvature(const PVSurfacePatch& patch,
                                  double lambda, double t) {
    // Compute principal curvatures using Hessian
    auto H = compute_hessian(patch, lambda, t);

    // Principal curvatures: eigenvalues of H
    auto [kappa1, kappa2] = eigenvalues_2x2(H);

    // Gaussian curvature: K = kappa1 * kappa2
    double K = kappa1 * kappa2;

    // Mean curvature: H = (kappa1 + kappa2) / 2
    double H_mean = (kappa1 + kappa2) / 2;

    // Use max curvature for refinement criterion
    return std::max(std::abs(kappa1), std::abs(kappa2));
}
```

---

## Critical Point Detection

### Why Detect Critical Points?

On PV curves/surfaces, critical points of the vector fields (v or w) reveal important flow structures:
- **Maxima/minima**: Peaks of velocity/vorticity along vortex core
- **Saddles**: Bifurcation points where flow changes character
- **Inflection points**: Transitions in curvature

### 3D Curves: 1D Critical Points

**Definition**: Points where dv/dλ or dw/dλ is parallel to v or w

```cpp
struct CriticalPoint1D {
    double lambda;              // Parameter value
    std::array<double, 3> pos;  // 3D spatial position

    enum Type {
        VelocityMax,     // ||v|| maximum
        VelocityMin,     // ||v|| minimum
        VorticityMax,    // ||w|| maximum
        VorticityMin,    // ||w|| minimum
        Inflection       // Curvature extremum
    } type;

    double value;  // Magnitude at critical point
};

template <typename T>
std::vector<CriticalPoint1D> find_critical_points_1d(const PVCurveSegment& segment) {
    std::vector<CriticalPoint1D> cps;

    // Find extrema of ||v(λ)||
    // ||v(λ)||² = v(λ) · v(λ)
    // d/dλ ||v||² = 2 v · (dv/dλ) = 0
    // → v ⊥ dv/dλ

    auto v_poly = interpolate_vector(segment.V, segment.P, segment.Q);  // v(λ)
    auto dv_poly = differentiate_polynomial(v_poly);                     // dv/dλ

    // Solve: v(λ) · dv/dλ(λ) = 0
    auto dot_poly = dot_product_polynomial(v_poly, dv_poly);
    auto roots = solve_polynomial(dot_poly);

    for (double lambda : roots) {
        if (lambda >= segment.lambda_min && lambda <= segment.lambda_max) {
            CriticalPoint1D cp;
            cp.lambda = lambda;
            cp.pos = evaluate_curve(segment, lambda);

            // Classify: check second derivative
            auto v_val = v_poly.evaluate(lambda);
            auto d2v_val = differentiate_polynomial_twice(v_poly).evaluate(lambda);
            double d2_norm = dot(v_val, d2v_val);

            if (d2_norm > 0) {
                cp.type = CriticalPoint1D::VelocityMin;
            } else {
                cp.type = CriticalPoint1D::VelocityMax;
            }

            cp.value = norm(v_val);
            cps.push_back(cp);
        }
    }

    // Similarly for ||w(λ)||
    // ...

    return cps;
}
```

### 4D Surfaces: 2D Critical Points

**Definition**: Points where ∇v or ∇w has rank < 2 (critical points of vector fields on surface)

```cpp
struct CriticalPoint2D {
    double lambda, t;            // Parameter values
    std::array<double, 4> pos;   // 4D spacetime position

    enum Type {
        Maximum,      // Local maximum of ||v|| or ||w||
        Minimum,      // Local minimum
        Saddle,       // Saddle point
        DegenMax,     // Degenerate maximum (high curvature)
        DegenMin      // Degenerate minimum
    } type;

    double value;  // Magnitude
};

template <typename T>
std::vector<CriticalPoint2D> find_critical_points_2d(const PVSurfacePatch& patch) {
    std::vector<CriticalPoint2D> cps;

    // Solve: ∇_surface ||v(λ, t)|| = 0
    // This is a system of 2 polynomial equations in (λ, t)

    auto v_poly = interpolate_vector_bivar(patch.V, patch.P, patch.Q);
    auto grad_lambda = differentiate_polynomial_lambda(v_poly);
    auto grad_t = differentiate_polynomial_t(v_poly);

    // Solve:
    // v · (∂v/∂λ) = 0
    // v · (∂v/∂t) = 0
    auto eq1 = dot_product_bivar(v_poly, grad_lambda);
    auto eq2 = dot_product_bivar(v_poly, grad_t);

    // Resultant method or numerical solver for bivariate polynomial system
    auto solutions = solve_bivariate_system(eq1, eq2,
                                           patch.lambda_min, patch.lambda_max,
                                           patch.t_min, patch.t_max);

    for (auto [lambda, t] : solutions) {
        CriticalPoint2D cp;
        cp.lambda = lambda;
        cp.t = t;
        cp.pos = evaluate_surface(patch, lambda, t);

        // Classify using Hessian
        auto H = compute_hessian_surface(v_poly, lambda, t);
        auto [eig1, eig2] = eigenvalues_2x2(H);

        if (eig1 > 0 && eig2 > 0) {
            cp.type = CriticalPoint2D::Minimum;
        } else if (eig1 < 0 && eig2 < 0) {
            cp.type = CriticalPoint2D::Maximum;
        } else {
            cp.type = CriticalPoint2D::Saddle;
        }

        cp.value = norm(v_poly.evaluate(lambda, t));
        cps.push_back(cp);
    }

    return cps;
}
```

---

## Storage Format Design

### Hierarchical Storage

```cpp
namespace ftk2 {

// Level 1: Parametric (exact, compact)
struct ExactPVComplex {
    // 3D spatial
    std::vector<PVCurveSegment> curve_segments;

    // 4D spacetime
    std::vector<PVSurfacePatch> surface_patches;

    // Metadata
    int spatial_dim;        // 3 for 3D spatial, 4 for spacetime
    int feature_dim;        // 1 for curves, 2 for surfaces

    // Save to file (compact representation)
    void save(const std::string& filename) const;
    static ExactPVComplex load(const std::string& filename);
};

// Level 2: Sampled (for visualization)
struct SampledPVComplex {
    // 3D curves → polylines
    std::vector<Polyline> polylines;

    // 4D surfaces → triangle meshes
    std::vector<TriangleMesh> meshes;

    // Generate from parametric representation
    static SampledPVComplex from_exact(const ExactPVComplex& exact,
                                       double tolerance = 1e-3);

    // Export to visualization formats
    void write_vtp(const std::string& filename) const;  // VTK PolyData
    void write_vtu(const std::string& filename) const;  // VTK Unstructured
};

} // namespace ftk2
```

### File Format (HDF5)

```
exactpv.h5
├── /curve_segments
│   ├── segment_0000
│   │   ├── simplex: [v0, v1, v2, v3] (vertex IDs)
│   │   ├── lambda_range: [lambda_min, lambda_max]
│   │   ├── P_coefficients: [4 × 4 array] (P[0..3], each degree 3)
│   │   ├── Q_coefficients: [4 array] (degree 3)
│   │   └── critical_points: [...] (optional)
│   ├── segment_0001
│   └── ...
├── /surface_patches
│   ├── patch_0000
│   │   ├── simplex: [v0, v1, v2, v3, v4] (5 vertices)
│   │   ├── lambda_range: [lambda_min, lambda_max]
│   │   ├── t_range: [t_min, t_max]
│   │   ├── P_coefficients: [5 × 4 × 4 array] (bivariate, degree 3 in λ,t)
│   │   ├── Q_coefficients: [4 × 4 array]
│   │   └── critical_points_2d: [...]
│   └── ...
└── /metadata
    ├── spatial_dim: 3 or 4
    ├── feature_dim: 1 or 2
    ├── num_segments: N
    └── num_patches: M
```

---

## Visualization Workflow

### Option 1: Direct Rendering (Real-time)

```cpp
// In FTK2 viewer or ParaView plugin
void render_exactpv(const ExactPVComplex& complex) {
    for (const auto& segment : complex.curve_segments) {
        // Adaptive sampling on-the-fly
        auto samples = adaptive_sample_curve(segment, tolerance);

        // Render as polyline
        glBegin(GL_LINE_STRIP);
        for (const auto& p : samples) {
            glVertex3d(p.x, p.y, p.z);
        }
        glEnd();
    }
}
```

### Option 2: Offline Export (Publication-quality)

```cpp
// Export to VTK for ParaView/VisIt
void export_to_vtk(const ExactPVComplex& complex,
                  const std::string& filename,
                  double tolerance = 1e-4) {
    auto sampled = SampledPVComplex::from_exact(complex, tolerance);
    sampled.write_vtp(filename);
}

// Usage:
ExactPVComplex complex = engine.get_exact_complex();
export_to_vtk(complex, "vortex_cores.vtp", 1e-5);  // High-quality
```

### Option 3: Interactive Refinement

```python
# Python API for interactive exploration
import ftk2

complex = ftk2.ExactPVComplex.load("exactpv.h5")

# Coarse preview
sampled_coarse = complex.sample(tolerance=1e-2)
viewer.show(sampled_coarse)

# User zooms in → refine locally
region = viewer.get_selected_region()
sampled_fine = complex.sample_region(region, tolerance=1e-5)
viewer.update(sampled_fine)
```

---

## Implementation Strategy

### Phase 1: Parametric Storage (Weeks 1-3)

1. **Define data structures**:
   - `PVCurveSegment`, `PVSurfacePatch`
   - `ExactPVComplex` container

2. **Implement extraction**:
   - Store cubic rational representation
   - No sampling yet (just parameters)

3. **File I/O**:
   - HDF5 serialization
   - Load/save parametric representation

**Deliverable**: Parametric extraction + storage

### Phase 2: Critical Point Detection (Weeks 4-5)

1. **1D critical points** (on curves):
   - Extrema of ||v||, ||w||
   - Store with curve segments

2. **2D critical points** (on surfaces):
   - Solve bivariate polynomial systems
   - Classify using Hessian

3. **Unit tests**:
   - Synthetic fields with known critical points
   - Verify detection accuracy

**Deliverable**: Critical point detection

### Phase 3: Adaptive Sampling (Weeks 6-8)

1. **1D sampling** (curves):
   - Curvature-based refinement
   - Error-controlled polyline generation

2. **2D sampling** (surfaces):
   - Adaptive triangulation
   - Curvature-based subdivision

3. **Optimization**:
   - Cache polynomial evaluations
   - Parallel sampling (OpenMP)

**Deliverable**: High-quality sampling algorithms

### Phase 4: Visualization Integration (Weeks 9-10)

1. **VTK export**:
   - Polylines (curves)
   - Triangle meshes (surfaces)

2. **ParaView plugin** (optional):
   - Load .h5 files directly
   - Interactive refinement

3. **Examples**:
   - Vortex core visualization
   - Comparison with ApproxPV

**Deliverable**: End-to-end visualization pipeline

---

## Performance Targets

| Task | Target | Notes |
|------|--------|-------|
| **Parametric extraction** | 1000 tets/sec | Fast (no sampling) |
| **Critical point detection** | 500 curves/sec | Polynomial solving |
| **Adaptive sampling (curve)** | 100 curves/sec | Depends on tolerance |
| **Adaptive sampling (surface)** | 10 patches/sec | 2D more expensive |
| **VTK export** | <1 sec for 10k curves | I/O bound |

---

## Advanced Features (Future)

### 1. Analytical Queries

Store parametric representation enables:
- **Distance queries**: min distance from point to PV curve/surface
- **Intersection**: PV curve ∩ plane (analytical)
- **Integration**: ∫ f(x) dx along PV curve (Gaussian quadrature on cubic)

### 2. Temporal Interpolation

For 4D surfaces, interpolate in time:
```cpp
PVCurveSegment get_curve_at_time(const PVSurfacePatch& patch, double t_query) {
    // Fix t = t_query, extract 1D curve in λ
    PVCurveSegment curve;
    curve.lambda_min = patch.lambda_min;
    curve.lambda_max = patch.lambda_max;

    // Evaluate bivariate polynomials at t = t_query
    for (int i = 0; i < 5; ++i) {
        curve.P[i] = patch.P[i].evaluate_t(t_query);  // → univariate polynomial in λ
    }
    curve.Q = patch.Q.evaluate_t(t_query);

    return curve;
}
```

### 3. Level-of-Detail (LOD)

Generate multiple resolutions:
```cpp
struct MultiResolutionPV {
    SampledPVComplex lod_coarse;   // 100 samples/curve
    SampledPVComplex lod_medium;   // 1000 samples/curve
    SampledPVComplex lod_fine;     // 10000 samples/curve

    ExactPVComplex parametric;     // Full precision
};
```

---

## Summary

**Key Decisions**:

1. ✅ **Two-stage approach**: Parametric storage + deferred sampling
   - Exact representation
   - Flexible resolution

2. ✅ **Adaptive sampling**: Curvature-based refinement
   - High quality
   - Efficient (fewer samples)

3. ✅ **Critical point detection**: Analytical polynomial solving
   - Important flow features
   - Guide sampling strategy

4. ✅ **HDF5 storage**: Compact, portable
   - Industry standard
   - Supports large data

**Benefits**:
- 📊 Exact geometry (cubic rational)
- 🎯 Accurate critical features
- 💾 Compact storage (parametric)
- 🔍 Interactive refinement
- 🖼️ Publication-quality output

This strategy provides a solid foundation for ExactPV visualization that respects the mathematical nature of cubic rational geometry while remaining practical for large-scale scientific datasets.
