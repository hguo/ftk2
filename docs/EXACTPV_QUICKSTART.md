# ExactPV Quick Start Guide

## What We're Building

**Exact Parallel Vector (ExactPV) Tracking**: Analytical detection of locations where two vector fields v and w are parallel (v × w = 0), forming **cubic rational curves** (3D) and **surfaces** (4D).

**Key Innovation**: First implementation with full 4D spacetime support + proper handling of multiple punctures + adaptive nonlinear sampling.

---

## Why This Matters

1. **Research Contribution**: Novel 4D theory → publishable paper
2. **Practical Impact**: Gold-standard vortex core detection (complements ApproxPV)
3. **Technical Challenge**: Cubic rational geometry + topological stitching + degenerate cases

---

## Documentation Structure

| Document | Purpose | Status |
|----------|---------|--------|
| `EXACTPV_RESEARCH_PLAN.md` | Overall research & implementation roadmap (6-7 months) | ✅ Complete |
| `EXACTPV_SAMPLING_VISUALIZATION.md` | Sampling strategy & visualization pipeline | ✅ Complete |
| `EXACTPV_QUICKSTART.md` | This file - immediate action plan | ✅ Complete |

---

## Immediate Next Steps (Week 1)

### Day 1-2: Deep Dive into Legacy Code

**Goal**: Fully understand existing 3D solver

**Tasks**:
1. Read and annotate:
   - `/ftk/include/ftk/numeric/parallel_vector_solver3.hh` (350 lines)
   - `/ftk/include/ftk/numeric/parallel_vector_solver2.hh` (for reference)
   - `/ftk/tests/test_parallel_vectors.cpp` (unit tests)

2. Key functions to understand:
   ```cpp
   // Triangle: up to 3 punctures
   int solve_pv_s2v3(const T V[3][3], const T W[3][3],
                     T lambda[3], T mu[3][3]);

   // Tetrahedron: curve segment
   void characteristic_polynomials_pv_s3v3(const T V[4][3], const T W[4][3],
                                            T Q[4], T P[4][4]);
   disjoint_intervals<T> solve_pv_inequalities_s3v3(...);
   ```

3. Run tests:
   ```bash
   cd /home/hguo/workspace/ftk/build
   ./tests/test_parallel_vectors
   ```

4. Create notes document:
   ```
   docs/EXACTPV_LEGACY_ANALYSIS.md
   ```
   Document:
   - Algorithm flow
   - Data structures
   - Edge cases handled (or not handled)
   - Performance characteristics

**Deliverable**: Detailed understanding + notes

---

### Day 3-4: FTK2 Infrastructure Setup

**Goal**: Create skeleton for ExactPV in FTK2

**Tasks**:

1. **Create solver header**:
   ```bash
   touch include/ftk2/numeric/parallel_vector_solver.hpp
   ```

   Basic structure:
   ```cpp
   #pragma once

   #include <array>
   #include <vector>
   #include <cmath>

   namespace ftk2 {

   // Polynomial representation (degree D)
   template <typename T, int D>
   struct Polynomial {
       std::array<T, D+1> coeffs;  // coeffs[0] + coeffs[1]*x + ... + coeffs[D]*x^D

       T evaluate(T x) const;
       Polynomial<T, D-1> differentiate() const;
   };

   // Bivariate polynomial (degree D in both variables)
   template <typename T, int D>
   struct BivarPolynomial {
       std::array<std::array<T, D+1>, D+1> coeffs;  // coeffs[i][j] * lambda^i * t^j

       T evaluate(T lambda, T t) const;
       BivarPolynomial<T, D-1> diff_lambda() const;
       BivarPolynomial<T, D-1> diff_t() const;
   };

   // Puncture point on 2-cell
   struct PuncturePoint {
       double lambda;
       std::array<double, 3> barycentric;  // mu on triangle
       std::array<double, 3> coords_3d;    // Spatial position
   };

   // Curve segment (3D spatial, in tetrahedron)
   struct PVCurveSegment {
       int simplex_id;
       double lambda_min, lambda_max;
       std::array<Polynomial<double, 3>, 4> P;  // 4 barycentric coords
       Polynomial<double, 3> Q;                 // Shared denominator
   };

   // Surface patch (4D spacetime, in pentatope)
   struct PVSurfacePatch {
       int simplex_id;
       double lambda_min, lambda_max;
       double t_min, t_max;
       std::array<BivarPolynomial<double, 3>, 5> P;  // 5 barycentric coords
       BivarPolynomial<double, 3> Q;
   };

   // 3D solver: Triangle → up to 3 punctures
   template <typename T>
   int solve_pv_triangle(const T V[3][3], const T W[3][3],
                        std::vector<PuncturePoint>& punctures,
                        T epsilon = std::numeric_limits<T>::epsilon());

   // 3D solver: Tetrahedron → curve segment
   template <typename T>
   bool solve_pv_tetrahedron(const T V[4][3], const T W[4][3],
                            PVCurveSegment& segment,
                            T epsilon = std::numeric_limits<T>::epsilon());

   // 4D solver: Pentatope → surface patch (TBD - research phase)
   template <typename T>
   bool solve_pv_pentatope(const T V[5][3], const T W[5][3],
                          PVSurfacePatch& patch,
                          int resolution = 16,
                          T epsilon = std::numeric_limits<T>::epsilon());

   } // namespace ftk2
   ```

2. **Create predicate**:
   ```bash
   # Add to include/ftk2/core/predicate.hpp
   ```

   ```cpp
   // Exact Parallel Vectors (codimension M=2)
   template <typename T = double>
   struct ExactPVPredicate : public Predicate<2, T> {
       static constexpr int codimension = 2;

       std::string vector_u_name = "u";
       std::string vector_v_name = "v";

       // For 3D spatial: extract punctures on triangles
       bool extract_simplex(const Simplex& s,
                           const std::map<std::string, ftk::ndarray<T>>& data,
                           std::vector<FeatureElement>& elements) const override;

       // Storage for parametric curves (3D)
       mutable std::vector<PVCurveSegment> curve_segments;

       // Storage for parametric surfaces (4D)
       mutable std::vector<PVSurfacePatch> surface_patches;
   };
   ```

3. **Create test file**:
   ```bash
   touch tests/test_exactpv.cpp
   ```

   Basic unit test:
   ```cpp
   #include <catch.hpp>
   #include <ftk2/numeric/parallel_vector_solver.hpp>

   TEST_CASE("polynomial_evaluation") {
       ftk2::Polynomial<double, 3> p;
       p.coeffs = {1.0, 2.0, 3.0, 4.0};  // 1 + 2x + 3x² + 4x³

       REQUIRE(p.evaluate(0.0) == 1.0);
       REQUIRE(p.evaluate(1.0) == 10.0);  // 1+2+3+4
       REQUIRE(p.evaluate(2.0) == 49.0);  // 1+4+12+32
   }

   TEST_CASE("solve_pv_triangle_simple") {
       // Simple test case: parallel vectors everywhere
       double V[3][3] = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
       double W[3][3] = {{2, 0, 0}, {2, 0, 0}, {2, 0, 0}};

       std::vector<ftk2::PuncturePoint> punctures;
       int n = ftk2::solve_pv_triangle(V, W, punctures);

       // Should detect degenerate case (entire triangle is PV surface)
       REQUIRE(n == std::numeric_limits<int>::max());
   }
   ```

4. **Update CMake**:
   ```cmake
   # In tests/CMakeLists.txt
   add_executable(ftk2_test_exactpv test_exactpv.cpp)
   target_link_libraries(ftk2_test_exactpv PRIVATE ftk2 Catch2::Catch2)
   ```

**Deliverable**: Basic infrastructure + failing tests (implementation comes next)

---

### Day 5: First Implementation - Polynomial Utils

**Goal**: Implement polynomial evaluation and differentiation

**Tasks**:

```cpp
// In include/ftk2/numeric/parallel_vector_solver.hpp

template <typename T, int D>
T Polynomial<T, D>::evaluate(T x) const {
    T result = coeffs[D];
    for (int i = D - 1; i >= 0; --i) {
        result = result * x + coeffs[i];  // Horner's method
    }
    return result;
}

template <typename T, int D>
Polynomial<T, D-1> Polynomial<T, D>::differentiate() const {
    Polynomial<T, D-1> result;
    for (int i = 0; i < D; ++i) {
        result.coeffs[i] = coeffs[i + 1] * (i + 1);
    }
    return result;
}

// Add polynomial operations
template <typename T, int D>
Polynomial<T, D> operator+(const Polynomial<T, D>& a, const Polynomial<T, D>& b) {
    Polynomial<T, D> result;
    for (int i = 0; i <= D; ++i) {
        result.coeffs[i] = a.coeffs[i] + b.coeffs[i];
    }
    return result;
}

template <typename T, int D>
Polynomial<T, D> operator*(const Polynomial<T, D>& a, T scalar) {
    Polynomial<T, D> result;
    for (int i = 0; i <= D; ++i) {
        result.coeffs[i] = a.coeffs[i] * scalar;
    }
    return result;
}

// Polynomial multiplication (degree increases)
template <typename T, int D1, int D2>
Polynomial<T, D1 + D2> multiply(const Polynomial<T, D1>& a, const Polynomial<T, D2>& b) {
    Polynomial<T, D1 + D2> result;
    result.coeffs.fill(0);

    for (int i = 0; i <= D1; ++i) {
        for (int j = 0; j <= D2; ++j) {
            result.coeffs[i + j] += a.coeffs[i] * b.coeffs[j];
        }
    }
    return result;
}
```

**Test**:
```cpp
TEST_CASE("polynomial_differentiation") {
    ftk2::Polynomial<double, 3> p;
    p.coeffs = {1, 2, 3, 4};  // 1 + 2x + 3x² + 4x³

    auto dp = p.differentiate();  // 2 + 6x + 12x²

    REQUIRE(dp.evaluate(0.0) == 2.0);
    REQUIRE(dp.evaluate(1.0) == 20.0);  // 2+6+12
}

TEST_CASE("polynomial_multiplication") {
    ftk2::Polynomial<double, 1> p;
    p.coeffs = {1, 2};  // 1 + 2x

    ftk2::Polynomial<double, 1> q;
    q.coeffs = {3, 4};  // 3 + 4x

    auto r = ftk2::multiply(p, q);  // (1+2x)(3+4x) = 3 + 10x + 8x²

    REQUIRE(r.coeffs[0] == 3);
    REQUIRE(r.coeffs[1] == 10);
    REQUIRE(r.coeffs[2] == 8);
}
```

**Deliverable**: Working polynomial utilities

---

## Week 1 Goals Summary

By end of Week 1, you should have:

✅ **Understanding**:
- Deep knowledge of legacy 3D solver
- Notes on algorithm & data structures

✅ **Infrastructure**:
- `include/ftk2/numeric/parallel_vector_solver.hpp` (skeleton)
- `ExactPVPredicate` in `predicate.hpp`
- Test framework (`test_exactpv.cpp`)

✅ **Basic Implementation**:
- Polynomial class with evaluate/differentiate
- Unit tests passing

✅ **Momentum**:
- Clear path forward for Week 2 (port cubic solver)

---

## Week 2 Preview: Port 3D Solver

**Tasks**:
1. Port `solve_pv_s2v3` (triangle → 3 punctures)
2. Port `characteristic_polynomials_pv_s3v3` (tetrahedron)
3. Port cubic solver (roots of degree-3 polynomial)
4. Implement inequality solver (λ intervals)
5. Unit tests comparing with legacy FTK

**Deliverable**: Working 3D spatial ExactPV (feature parity with legacy)

---

## Development Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/exactpv-foundation

# Regular commits
git commit -m "Add polynomial utilities"
git commit -m "Port solve_pv_s2v3 from legacy"
git commit -m "Add unit tests for triangle solver"

# Merge when phase complete
git checkout main
git merge feature/exactpv-foundation
```

### Testing

```bash
# Build and run tests
cd build
make ftk2_test_exactpv -j8
./tests/ftk2_test_exactpv

# Run specific test
./tests/ftk2_test_exactpv "polynomial_evaluation"
```

### Documentation

Keep notes in:
- `docs/EXACTPV_PROGRESS.md` (weekly progress log)
- `docs/EXACTPV_DECISIONS.md` (design decisions & rationale)

---

## Success Metrics (Week 1)

| Metric | Target | Actual |
|--------|--------|--------|
| Legacy code understood | 100% | __ % |
| FTK2 skeleton created | 100% | __ % |
| Polynomial tests passing | 5+ tests | __ tests |
| Build time | < 10 sec | __ sec |
| Motivation level | 🔥🔥🔥 | __ |

---

## Resources

### Legacy FTK Code

- **3D Solver**: `/home/hguo/workspace/ftk/include/ftk/numeric/parallel_vector_solver3.hh`
- **2D Solver**: `/home/hguo/workspace/ftk/include/ftk/numeric/parallel_vector_solver2.hh`
- **Tests**: `/home/hguo/workspace/ftk/tests/test_parallel_vectors.cpp`

### Papers

1. **Your paper**: [arXiv:2107.02708](https://arxiv.org/abs/2107.02708) - Exact Analytical Parallel Vectors
2. **Peikert & Roth** (1999): "The Parallel Vectors Operator" - IEEE Vis
3. **Edelsbrunner & Mücke** (1990): "Simulation of Simplicity" - ACM TOG

### Math References

- **Cubic equation solver**: Wikipedia "Cubic function"
- **Rational functions**: Any real analysis textbook
- **Polynomial interpolation**: Numerical recipes

---

## Questions to Resolve (During Week 1)

1. **Q**: Should we use `std::array` or `std::vector` for polynomial coefficients?
   **A**: Use `std::array<T, D+1>` (fixed size, stack allocation, faster)

2. **Q**: How to handle degenerate cases in tests?
   **A**: Return `std::numeric_limits<int>::max()` for infinite solutions

3. **Q**: What epsilon value for floating-point comparisons?
   **A**: Start with `std::numeric_limits<T>::epsilon()`, tune later

4. **Q**: Should polynomial class be in separate file?
   **A**: Start in parallel_vector_solver.hpp, refactor if reused elsewhere

---

## Getting Started Right Now

```bash
# 1. Navigate to FTK2
cd /home/hguo/workspace/ftk2

# 2. Create feature branch
git checkout -b feature/exactpv-week1

# 3. Create skeleton files
touch include/ftk2/numeric/parallel_vector_solver.hpp
touch tests/test_exactpv.cpp

# 4. Open in editor
code include/ftk2/numeric/parallel_vector_solver.hpp

# 5. Start with polynomial class (copy skeleton from Day 5 above)
```

---

## Motivation

🎯 **This is not just coding - it's research!**

By the end of this project, you will have:
- ✅ Solved an open problem (4D PV surfaces)
- ✅ Published a paper in a major venue
- ✅ Built a production tool used by scientists worldwide
- ✅ Advanced the state-of-the-art in flow visualization

**Let's build something amazing!** 🚀
