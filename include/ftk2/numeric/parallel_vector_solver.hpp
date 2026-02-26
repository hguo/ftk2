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
 * @brief Compute 2x2 determinant
 */
template <typename T>
inline T det2(const T a[2][2]) {
    return a[0][0] * a[1][1] - a[1][0] * a[0][1];
}

/**
 * @brief Compute 3x3 determinant
 */
template <typename T>
inline T det3(const T a[3][3]) {
    return a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
         - a[0][1] * (a[1][0] * a[2][2] - a[2][0] * a[1][2])
         + a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
}

/**
 * @brief Compute 3x3 matrix trace
 */
template <typename T>
inline T trace3(const T a[3][3]) {
    return a[0][0] + a[1][1] + a[2][2];
}

/**
 * @brief Compute characteristic polynomial for generalized eigenvalue problem (3x3)
 * det(A - λB) = P[0] + P[1]λ + P[2]λ² + P[3]λ³
 */
template <typename T>
void characteristic_polynomial_3x3(const T A[3][3], const T B[3][3], T P[4]) {
    P[0] = det3(A);
    P[1] = A[1][2] * A[2][1] * B[0][0] - A[1][1] * A[2][2] * B[0][0] - A[1][2] * A[2][0] * B[0][1]
          +A[1][0] * A[2][2] * B[0][1] + A[1][1] * A[2][0] * B[0][2] - A[1][0] * A[2][1] * B[0][2]
          -A[0][2] * A[2][1] * B[1][0] + A[0][1] * A[2][2] * B[1][0] + A[0][2] * A[2][0] * B[1][1]
          -A[0][0] * A[2][2] * B[1][1] - A[0][1] * A[2][0] * B[1][2] + A[0][0] * A[2][1] * B[1][2]
          +A[0][2] * A[1][1] * B[2][0] - A[0][1] * A[1][2] * B[2][0] - A[0][2] * A[1][0] * B[2][1]
          +A[0][0] * A[1][2] * B[2][1] + A[0][1] * A[1][0] * B[2][2] - A[0][0] * A[1][1] * B[2][2];
    P[2] =-A[2][2] * B[0][1] * B[1][0] + A[2][1] * B[0][2] * B[1][0] + A[2][2] * B[0][0] * B[1][1]
          -A[2][0] * B[0][2] * B[1][1] - A[2][1] * B[0][0] * B[1][2] + A[2][0] * B[0][1] * B[1][2]
          +A[1][2] * B[0][1] * B[2][0] - A[1][1] * B[0][2] * B[2][0] - A[0][2] * B[1][1] * B[2][0]
          +A[0][1] * B[1][2] * B[2][0] - A[1][2] * B[0][0] * B[2][1] + A[1][0] * B[0][2] * B[2][1]
          +A[0][2] * B[1][0] * B[2][1] - A[0][0] * B[1][2] * B[2][1] + A[1][1] * B[0][0] * B[2][2]
          -A[1][0] * B[0][1] * B[2][2] - A[0][1] * B[1][0] * B[2][2] + A[0][0] * B[1][1] * B[2][2];
    P[3] = -det3(B);
}

/**
 * @brief Solve cubic equation: P[0] + P[1]x + P[2]x² + P[3]x³ = 0
 * Returns number of real roots
 */
template <typename T>
int solve_cubic_real(const T P[4], T roots[3], T epsilon = std::numeric_limits<T>::epsilon()) {
    if (std::abs(P[3]) < epsilon) {
        // Degenerate to quadratic
        if (std::abs(P[2]) < epsilon) {
            // Linear equation
            if (std::abs(P[1]) < epsilon) return 0;
            roots[0] = -P[0] / P[1];
            return 1;
        }
        // Quadratic: P[2]x² + P[1]x + P[0] = 0
        T disc = P[1] * P[1] - 4 * P[2] * P[0];
        if (disc < 0) return 0;
        if (std::abs(disc) < epsilon) {
            roots[0] = -P[1] / (2 * P[2]);
            return 1;
        }
        T sqrt_disc = std::sqrt(disc);
        roots[0] = (-P[1] + sqrt_disc) / (2 * P[2]);
        roots[1] = (-P[1] - sqrt_disc) / (2 * P[2]);
        return 2;
    }

    // Normalize to x³ + bx² + cx + d = 0
    T b = P[2] / P[3];
    T c = P[1] / P[3];
    T d = P[0] / P[3];

    T q = (3 * c - b * b) / 9;
    T r = (-(27 * d) + b * (9 * c - 2 * b * b)) / 54;
    T disc = q * q * q + r * r;
    T term1 = b / 3;

    if (disc > 0) {
        // One real root
        T s = r + std::sqrt(disc);
        s = (s < 0) ? -std::pow(-s, 1.0/3.0) : std::pow(s, 1.0/3.0);
        T t = r - std::sqrt(disc);
        t = (t < 0) ? -std::pow(-t, 1.0/3.0) : std::pow(t, 1.0/3.0);
        roots[0] = -term1 + s + t;
        return 1;
    } else if (std::abs(disc) < epsilon) {
        // Two or three equal roots
        T r13 = (r < 0) ? -std::pow(-r, 1.0/3.0) : std::pow(r, 1.0/3.0);
        roots[0] = -term1 + 2 * r13;
        roots[1] = -(r13 + term1);
        if (std::abs(roots[0] - roots[1]) < epsilon) return 1;
        return 2;
    } else {
        // Three distinct real roots
        q = -q;
        T dum1 = std::acos(r / std::sqrt(q * q * q));
        T r13 = 2 * std::sqrt(q);
        roots[0] = -term1 + r13 * std::cos(dum1 / 3);
        roots[1] = -term1 + r13 * std::cos((dum1 + 2 * M_PI) / 3);
        roots[2] = -term1 + r13 * std::cos((dum1 + 4 * M_PI) / 3);
        return 3;
    }
}

/**
 * @brief Solve 3x2 least squares system Mx = b
 * Returns condition number estimate
 */
template <typename T>
T solve_least_square3x2(const T M[3][2], const T b[3], T x[2], T epsilon = std::numeric_limits<T>::epsilon()) {
    // M^T M x = M^T b
    T MTM[2][2] = {
        {M[0][0]*M[0][0] + M[1][0]*M[1][0] + M[2][0]*M[2][0],
         M[0][0]*M[0][1] + M[1][0]*M[1][1] + M[2][0]*M[2][1]},
        {M[0][1]*M[0][0] + M[1][1]*M[1][0] + M[2][1]*M[2][0],
         M[0][1]*M[0][1] + M[1][1]*M[1][1] + M[2][1]*M[2][1]}
    };
    T MTb[2] = {
        M[0][0]*b[0] + M[1][0]*b[1] + M[2][0]*b[2],
        M[0][1]*b[0] + M[1][1]*b[1] + M[2][1]*b[2]
    };

    T det = MTM[0][0] * MTM[1][1] - MTM[0][1] * MTM[1][0];
    if (std::abs(det) < epsilon) {
        x[0] = x[1] = 0;
        return std::numeric_limits<T>::infinity();
    }

    x[0] = (MTM[1][1] * MTb[0] - MTM[0][1] * MTb[1]) / det;
    x[1] = (MTM[0][0] * MTb[1] - MTM[1][0] * MTb[0]) / det;

    // Estimate condition number (very rough)
    T trace = MTM[0][0] + MTM[1][1];
    return trace / std::abs(det);
}

/**
 * @brief Linear interpolation on triangle (3 vertices, 3D vectors)
 */
template <typename T>
void lerp_s2v3(const T V[3][3], const T mu[3], T result[3]) {
    result[0] = mu[0] * V[0][0] + mu[1] * V[1][0] + mu[2] * V[2][0];
    result[1] = mu[0] * V[0][1] + mu[1] * V[1][1] + mu[2] * V[2][1];
    result[2] = mu[0] * V[0][2] + mu[1] * V[1][2] + mu[2] * V[2][2];
}

/**
 * @brief Compute 3D cross product
 */
template <typename T>
void cross_product3(const T a[3], const T b[3], T result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * @brief Compute 3D vector norm
 */
template <typename T>
T vector_norm3(const T v[3]) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/**
 * @brief Verify that v and w are parallel at given barycentric coordinates
 */
template <typename T>
bool verify_parallel(const T V[3][3], const T W[3][3],
                    const T mu[3],
                    T epsilon = std::numeric_limits<T>::epsilon()) {
    T v[3], w[3], cross[3];
    lerp_s2v3(V, mu, v);
    lerp_s2v3(W, mu, w);
    cross_product3(v, w, cross);
    return vector_norm3(cross) <= epsilon;
}

// ============================================================================
// Implementation (Stub for Week 1, will be filled in Week 2)
// ============================================================================

template <typename T>
int solve_pv_triangle(const T V[3][3], const T W[3][3],
                     std::vector<PuncturePoint>& punctures,
                     T epsilon) {
    punctures.clear();

    // Check for degenerate case: all vectors parallel at vertices
    bool all_parallel = true;
    for (int i = 0; i < 3; ++i) {
        T cross[3];
        cross_product3(V[i], W[i], cross);
        if (vector_norm3(cross) > epsilon) {
            all_parallel = false;
            break;
        }
    }

    if (all_parallel) {
        return std::numeric_limits<int>::max();  // Entire triangle is PV surface
    }

    // Transpose matrices for characteristic polynomial computation
    T VT[3][3], WT[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            VT[i][j] = V[j][i];
            WT[i][j] = W[j][i];
        }
    }

    // Compute characteristic polynomial: det(V - λW) = 0
    T P[4];
    characteristic_polynomial_3x3(VT, WT, P);

    // Solve cubic equation for λ values
    T lambda[3];
    int n_roots = solve_cubic_real(P, lambda, epsilon);

    // For each root, compute barycentric coordinates and verify
    for (int i = 0; i < n_roots; ++i) {
        if (std::abs(lambda[i]) <= epsilon) continue;  // Skip λ ≈ 0

        // Set up least squares system: (V - λW)μ = 0
        // We solve for first two barycentric coords; third is 1 - μ₀ - μ₁
        const T M[3][2] = {
            {(VT[0][0] - VT[0][2]) - lambda[i] * (WT[0][0] - WT[0][2]),
             (VT[0][1] - VT[0][2]) - lambda[i] * (WT[0][1] - WT[0][2])},
            {(VT[1][0] - VT[1][2]) - lambda[i] * (WT[1][0] - WT[1][2]),
             (VT[1][1] - VT[1][2]) - lambda[i] * (WT[1][1] - WT[1][2])},
            {(VT[2][0] - VT[2][2]) - lambda[i] * (WT[2][0] - WT[2][2]),
             (VT[2][1] - VT[2][2]) - lambda[i] * (WT[2][1] - WT[2][2])}
        };
        const T b[3] = {
            -(VT[0][2] - lambda[i] * WT[0][2]),
            -(VT[1][2] - lambda[i] * WT[1][2]),
            -(VT[2][2] - lambda[i] * WT[2][2])
        };

        T nu[3];
        T cond = solve_least_square3x2(M, b, nu, epsilon);
        nu[2] = T(1) - nu[0] - nu[1];

        // Check if barycentric coordinates are valid (within [0,1])
        if (nu[0] >= -epsilon && nu[0] <= 1 + epsilon &&
            nu[1] >= -epsilon && nu[1] <= 1 + epsilon &&
            nu[2] >= -epsilon && nu[2] <= 1 + epsilon)
        {
            // Verify solution by checking cross product
            T v[3], w[3], cross[3];
            lerp_s2v3(V, nu, v);
            lerp_s2v3(W, nu, w);
            cross_product3(v, w, cross);
            T norm = vector_norm3(cross);

            if (norm > 1e-2) {
                // Reject due to large residual
                continue;
            }

            // Valid puncture point found
            PuncturePoint p;
            p.lambda = lambda[i];
            p.barycentric[0] = nu[0];
            p.barycentric[1] = nu[1];
            p.barycentric[2] = nu[2];
            // Note: 3D coords would be computed from mesh coordinates
            p.coords_3d[0] = p.coords_3d[1] = p.coords_3d[2] = 0;

            punctures.push_back(p);
        }
    }

    return punctures.size();
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
