#pragma once

#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

namespace ftk2 {

// ============================================================================
// Polynomial Representation (degree D)
// ============================================================================

template <typename T, int D>
struct Polynomial {
    std::array<T, D+1> coeffs;  // coeffs[0] + coeffs[1]*x + ... + coeffs[D]*x^D

    Polynomial() {
        coeffs.fill(T(0));
    }

    // Evaluate polynomial at x using Horner's method
    T evaluate(T x) const {
        T result = coeffs[D];
        for (int i = D - 1; i >= 0; --i) {
            result = result * x + coeffs[i];
        }
        return result;
    }

    // Differentiate: returns polynomial of degree D-1
    Polynomial<T, D-1> differentiate() const {
        static_assert(D > 0, "Cannot differentiate constant polynomial");
        Polynomial<T, D-1> result;
        for (int i = 0; i < D; ++i) {
            result.coeffs[i] = coeffs[i + 1] * T(i + 1);
        }
        return result;
    }

    // Polynomial addition
    Polynomial<T, D> operator+(const Polynomial<T, D>& other) const {
        Polynomial<T, D> result;
        for (int i = 0; i <= D; ++i) {
            result.coeffs[i] = coeffs[i] + other.coeffs[i];
        }
        return result;
    }

    // Polynomial subtraction
    Polynomial<T, D> operator-(const Polynomial<T, D>& other) const {
        Polynomial<T, D> result;
        for (int i = 0; i <= D; ++i) {
            result.coeffs[i] = coeffs[i] - other.coeffs[i];
        }
        return result;
    }

    // Scalar multiplication
    Polynomial<T, D> operator*(T scalar) const {
        Polynomial<T, D> result;
        for (int i = 0; i <= D; ++i) {
            result.coeffs[i] = coeffs[i] * scalar;
        }
        return result;
    }
};

// Polynomial multiplication (degree increases to D1 + D2)
template <typename T, int D1, int D2>
Polynomial<T, D1 + D2> multiply(const Polynomial<T, D1>& a, const Polynomial<T, D2>& b) {
    Polynomial<T, D1 + D2> result;
    result.coeffs.fill(T(0));

    for (int i = 0; i <= D1; ++i) {
        for (int j = 0; j <= D2; ++j) {
            result.coeffs[i + j] += a.coeffs[i] * b.coeffs[j];
        }
    }
    return result;
}

// ============================================================================
// Bivariate Polynomial (degree D in both variables)
// ============================================================================

template <typename T, int D>
struct BivarPolynomial {
    std::array<std::array<T, D+1>, D+1> coeffs;  // coeffs[i][j] * lambda^i * t^j

    BivarPolynomial() {
        for (auto& row : coeffs) {
            row.fill(T(0));
        }
    }

    // Evaluate at (lambda, t)
    T evaluate(T lambda, T t) const {
        T result = T(0);
        T lambda_pow = T(1);
        for (int i = 0; i <= D; ++i) {
            T t_pow = T(1);
            for (int j = 0; j <= D; ++j) {
                result += coeffs[i][j] * lambda_pow * t_pow;
                t_pow *= t;
            }
            lambda_pow *= lambda;
        }
        return result;
    }

    // Partial derivative with respect to lambda
    BivarPolynomial<T, D-1> diff_lambda() const {
        static_assert(D > 0, "Cannot differentiate constant");
        BivarPolynomial<T, D-1> result;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j <= D; ++j) {
                if (j < D) {
                    result.coeffs[i][j] = coeffs[i + 1][j] * T(i + 1);
                }
            }
        }
        return result;
    }

    // Partial derivative with respect to t
    BivarPolynomial<T, D-1> diff_t() const {
        static_assert(D > 0, "Cannot differentiate constant");
        BivarPolynomial<T, D-1> result;
        for (int i = 0; i <= D; ++i) {
            for (int j = 0; j < D; ++j) {
                if (i < D) {
                    result.coeffs[i][j] = coeffs[i][j + 1] * T(j + 1);
                }
            }
        }
        return result;
    }

    // Evaluate at fixed t, returns univariate polynomial in lambda
    Polynomial<T, D> evaluate_t(T t_val) const {
        Polynomial<T, D> result;
        for (int i = 0; i <= D; ++i) {
            T t_pow = T(1);
            result.coeffs[i] = T(0);
            for (int j = 0; j <= D; ++j) {
                result.coeffs[i] += coeffs[i][j] * t_pow;
                t_pow *= t_val;
            }
        }
        return result;
    }
};

// ============================================================================
// Puncture Point (on 2-cell/triangle)
// ============================================================================

struct PuncturePoint {
    double lambda;                         // Scalar parameter
    std::array<double, 3> barycentric;     // mu on triangle (3 coords)
    std::array<double, 3> coords_3d;       // 3D spatial position

    PuncturePoint() : lambda(0.0) {
        barycentric.fill(0.0);
        coords_3d.fill(0.0);
    }
};

// ============================================================================
// Critical Point on Curve (1D)
// ============================================================================

struct CriticalPoint1D {
    enum Type {
        VelocityMax,     // ||v|| maximum
        VelocityMin,     // ||v|| minimum
        VorticityMax,    // ||w|| maximum
        VorticityMin,    // ||w|| minimum
        Inflection       // Curvature extremum
    };

    double lambda;                    // Parameter value
    std::array<double, 3> pos;        // 3D spatial position
    Type type;
    double value;                     // Magnitude at critical point

    CriticalPoint1D() : lambda(0.0), type(Inflection), value(0.0) {
        pos.fill(0.0);
    }
};

// ============================================================================
// Critical Point on Surface (2D)
// ============================================================================

struct CriticalPoint2D {
    enum Type {
        Maximum,      // Local maximum of ||v|| or ||w||
        Minimum,      // Local minimum
        Saddle,       // Saddle point
        DegenMax,     // Degenerate maximum
        DegenMin      // Degenerate minimum
    };

    double lambda, t;                 // Parameter values
    std::array<double, 4> pos;        // 4D spacetime position
    Type type;
    double value;                     // Magnitude

    CriticalPoint2D() : lambda(0.0), t(0.0), type(Minimum), value(0.0) {
        pos.fill(0.0);
    }
};

// ============================================================================
// PV Curve Segment (3D spatial, in tetrahedron)
// ============================================================================

struct PVCurveSegment {
    int simplex_id;                              // Tetrahedron ID
    double lambda_min, lambda_max;               // Parameter range

    std::array<Polynomial<double, 3>, 4> P;      // 4 barycentric coords (numerators)
    Polynomial<double, 3> Q;                     // Shared denominator

    std::vector<CriticalPoint1D> critical_points;  // Critical points on curve

    PVCurveSegment() : simplex_id(-1), lambda_min(0.0), lambda_max(1.0) {}

    // Evaluate curve at parameter lambda
    std::array<double, 4> get_barycentric(double lambda) const {
        double Q_val = Q.evaluate(lambda);
        if (std::abs(Q_val) < std::numeric_limits<double>::epsilon()) {
            // Degenerate case
            return {0.25, 0.25, 0.25, 0.25};
        }

        std::array<double, 4> mu;
        for (int i = 0; i < 4; ++i) {
            mu[i] = P[i].evaluate(lambda) / Q_val;
        }
        return mu;
    }
};

// ============================================================================
// PV Surface Patch (4D spacetime, in pentatope)
// ============================================================================

struct PVSurfacePatch {
    int simplex_id;                              // Pentatope ID
    double lambda_min, lambda_max;               // Lambda parameter range
    double t_min, t_max;                         // Time parameter range

    std::array<BivarPolynomial<double, 3>, 5> P; // 5 barycentric coords (numerators)
    BivarPolynomial<double, 3> Q;                // Shared denominator

    std::vector<CriticalPoint2D> critical_points;  // Critical points on surface

    PVSurfacePatch() : simplex_id(-1),
                       lambda_min(0.0), lambda_max(1.0),
                       t_min(0.0), t_max(1.0) {}

    // Evaluate surface at parameter (lambda, t)
    std::array<double, 5> get_barycentric(double lambda, double t) const {
        double Q_val = Q.evaluate(lambda, t);
        if (std::abs(Q_val) < std::numeric_limits<double>::epsilon()) {
            return {0.2, 0.2, 0.2, 0.2, 0.2};
        }

        std::array<double, 5> mu;
        for (int i = 0; i < 5; ++i) {
            mu[i] = P[i].evaluate(lambda, t) / Q_val;
        }
        return mu;
    }
};

// ============================================================================
// Solver Functions (3D Spatial)
// ============================================================================

/**
 * @brief Solve for parallel vectors on a triangle (2-simplex)
 *
 * Finds up to 3 puncture points where v × w = 0.
 *
 * @param V Vectors at 3 triangle vertices [3][3]
 * @param W Vectors at 3 triangle vertices [3][3]
 * @param punctures Output vector of puncture points
 * @param epsilon Tolerance for numerical comparisons
 * @return Number of punctures found (0-3), or INT_MAX if degenerate (entire triangle is PV)
 */
template <typename T>
int solve_pv_triangle(const T V[3][3], const T W[3][3],
                     std::vector<PuncturePoint>& punctures,
                     T epsilon = std::numeric_limits<T>::epsilon());

/**
 * @brief Solve for parallel vector curve in a tetrahedron (3-simplex)
 *
 * Finds the cubic rational curve segment where v × w = 0.
 *
 * @param V Vectors at 4 tetrahedron vertices [4][3]
 * @param W Vectors at 4 tetrahedron vertices [4][3]
 * @param segment Output curve segment (parametric representation)
 * @param epsilon Tolerance
 * @return true if curve exists in tetrahedron, false otherwise
 */
template <typename T>
bool solve_pv_tetrahedron(const T V[4][3], const T W[4][3],
                         PVCurveSegment& segment,
                         T epsilon = std::numeric_limits<T>::epsilon());

// ============================================================================
// Solver Functions (4D Spacetime) - Research Phase
// ============================================================================

/**
 * @brief Solve for parallel vector surface in a pentatope (4-simplex)
 *
 * Finds the cubic rational surface patch where v × w = 0.
 *
 * @param V Vectors at 5 pentatope vertices [5][3]
 * @param W Vectors at 5 pentatope vertices [5][3]
 * @param patch Output surface patch (parametric representation)
 * @param resolution Grid resolution for surface extraction
 * @param epsilon Tolerance
 * @return true if surface exists in pentatope, false otherwise
 */
template <typename T>
bool solve_pv_pentatope(const T V[5][3], const T W[5][3],
                       PVSurfacePatch& patch,
                       int resolution = 16,
                       T epsilon = std::numeric_limits<T>::epsilon());

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Verify that v and w are parallel at given barycentric coordinates
 */
template <typename T>
bool verify_parallel(const T V[3][3], const T W[3][3],
                    const T mu[3],
                    T epsilon = std::numeric_limits<T>::epsilon()) {
    // Interpolate v and w
    T v[3] = {0, 0, 0};
    T w[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            v[j] += mu[i] * V[i][j];
            w[j] += mu[i] * W[i][j];
        }
    }

    // Compute cross product
    T cross[3];
    cross[0] = v[1] * w[2] - v[2] * w[1];
    cross[1] = v[2] * w[0] - v[0] * w[2];
    cross[2] = v[0] * w[1] - v[1] * w[0];

    // Check if cross product is near zero
    T norm = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    return norm <= epsilon;
}

// ============================================================================
// Implementation (Stub for Week 1, will be filled in Week 2)
// ============================================================================

template <typename T>
int solve_pv_triangle(const T V[3][3], const T W[3][3],
                     std::vector<PuncturePoint>& punctures,
                     T epsilon) {
    // Check for degenerate case: all vectors parallel
    bool all_parallel = true;
    for (int i = 0; i < 3; ++i) {
        T cross[3];
        cross[0] = V[i][1] * W[i][2] - V[i][2] * W[i][1];
        cross[1] = V[i][2] * W[i][0] - V[i][0] * W[i][2];
        cross[2] = V[i][0] * W[i][1] - V[i][1] * W[i][0];

        T norm = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
        if (norm > epsilon) {
            all_parallel = false;
            break;
        }
    }

    if (all_parallel) {
        return std::numeric_limits<int>::max();  // Degenerate case
    }

    // TODO (Week 2): Implement actual cubic solver
    // For now, return 0 (no punctures found)
    punctures.clear();
    return 0;
}

template <typename T>
bool solve_pv_tetrahedron(const T V[4][3], const T W[4][3],
                         PVCurveSegment& segment,
                         T epsilon) {
    // TODO (Week 2): Implement tetrahedron solver
    // This will compute characteristic polynomials and solve inequalities
    return false;
}

template <typename T>
bool solve_pv_pentatope(const T V[5][3], const T W[5][3],
                       PVSurfacePatch& patch,
                       int resolution,
                       T epsilon) {
    // TODO (Phase 6): Implement 4D solver (research phase)
    return false;
}

} // namespace ftk2
