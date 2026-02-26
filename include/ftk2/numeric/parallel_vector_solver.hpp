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
 * @brief Compute characteristic polynomial for generalized eigenvalue problem (2x2)
 * det(A - λB) = P[0] + P[1]λ + P[2]λ²
 */
template <typename T>
void characteristic_polynomial_2x2(T a00, T a01, T a10, T a11, T b00, T b01, T b10, T b11, T P[3]) {
    P[2] = b00 * b11 - b10 * b01;
    P[1] = -(a00 * b11 - a10 * b01 + b00 * a11 - b10 * a01);
    P[0] = a00 * a11 - a10 * a01;
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
 * @brief Multiply two polynomials: R = P * Q
 * P has degree m, Q has degree n, R has degree m+n
 */
template <typename T>
void polynomial_multiply(const T P[], int m, const T Q[], int n, T R[]) {
    for (int i = 0; i <= m + n; ++i) R[i] = T(0);
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            R[i + j] += P[i] * Q[j];
}

/**
 * @brief Add polynomial Q to P in place
 */
template <typename T>
void polynomial_add_inplace(T P[], int m, const T Q[], int n) {
    for (int i = 0; i <= n && i <= m; ++i)
        P[i] += Q[i];
}

/**
 * @brief Subtract polynomial Q from P: R = P - Q
 */
template <typename T>
void polynomial_subtract(const T P[], int m, const T Q[], int n, T R[]) {
    int maxd = std::max(m, n);
    for (int i = 0; i <= maxd; ++i) {
        T p_val = (i <= m) ? P[i] : T(0);
        T q_val = (i <= n) ? Q[i] : T(0);
        R[i] = p_val - q_val;
    }
}

/**
 * @brief Multiply polynomial by scalar
 */
template <typename T>
void polynomial_scalar_multiply(T P[], int m, T scalar) {
    for (int i = 0; i <= m; ++i)
        P[i] *= scalar;
}

// Forward declaration
template <typename T>
int solve_cubic_real(const T P[4], T roots[3], T epsilon = std::numeric_limits<T>::epsilon());

/**
 * @brief Solve cubic rational inequality P(λ)/Q(λ) ≥ 0
 *
 * Returns intervals where the rational function is non-negative.
 * The rational function changes sign at:
 * 1. Roots of P (numerator zeros)
 * 2. Roots of Q (denominator poles)
 *
 * Algorithm:
 * 1. Find all roots of P and Q
 * 2. Sort roots to create test intervals
 * 3. Test sign in each interval
 * 4. Return intervals where sign is non-negative
 */
template <typename T>
struct Interval {
    T min, max;
    bool min_inclusive, max_inclusive;

    Interval() : min(0), max(0), min_inclusive(false), max_inclusive(false) {}
    Interval(T a, T b, bool inc_a = true, bool inc_b = true)
        : min(a), max(b), min_inclusive(inc_a), max_inclusive(inc_b) {}

    bool contains(T x, T eps = std::numeric_limits<T>::epsilon()) const {
        if (x < min - eps || x > max + eps) return false;
        if (std::abs(x - min) < eps) return min_inclusive;
        if (std::abs(x - max) < eps) return max_inclusive;
        return true;
    }

    T length() const { return max - min; }
};

/**
 * @brief Find valid parameter range for barycentric coordinates
 *
 * Computes intervals where all Pᵢ(λ)/Q(λ) ≥ 0.
 * Note: We don't need to check ≤ 1 because barycentric coords sum to 1!
 */
template <typename T>
std::vector<Interval<T>> solve_barycentric_inequalities(
    const T P[4][4], const T Q[4],
    T epsilon = std::numeric_limits<T>::epsilon())
{
    std::vector<T> critical_points;

    // Find roots of Q (denominator) - these are poles
    T q_roots[3];
    int n_q_roots = solve_cubic_real(Q, q_roots, epsilon);
    for (int i = 0; i < n_q_roots; ++i) {
        critical_points.push_back(q_roots[i]);
    }

    // Find roots of each Pᵢ (numerators) - these are zeros
    for (int i = 0; i < 4; ++i) {
        T p_roots[3];
        int n_p_roots = solve_cubic_real(P[i], p_roots, epsilon);
        for (int j = 0; j < n_p_roots; ++j) {
            critical_points.push_back(p_roots[j]);
        }
    }

    // Sort critical points
    std::sort(critical_points.begin(), critical_points.end());

    // Remove duplicates
    critical_points.erase(
        std::unique(critical_points.begin(), critical_points.end(),
                   [epsilon](T a, T b) { return std::abs(a - b) < epsilon; }),
        critical_points.end()
    );

    // Test intervals between critical points
    std::vector<Interval<T>> valid_intervals;

    // Add boundary points for testing
    std::vector<T> test_points = {-1e10}; // Start with -infinity
    test_points.insert(test_points.end(), critical_points.begin(), critical_points.end());
    test_points.push_back(1e10); // End with +infinity

    for (size_t i = 0; i + 1 < test_points.size(); ++i) {
        T left = test_points[i];
        T right = test_points[i + 1];
        T mid = (left + right) * 0.5;

        // Evaluate P[i]/Q at midpoint
        T q_val = Q[0] + Q[1]*mid + Q[2]*mid*mid + Q[3]*mid*mid*mid;

        // Skip if Q is zero (pole)
        if (std::abs(q_val) < epsilon) continue;

        // Check if all barycentric coordinates are non-negative at midpoint
        bool all_nonneg = true;
        for (int j = 0; j < 4; ++j) {
            T p_val = P[j][0] + P[j][1]*mid + P[j][2]*mid*mid + P[j][3]*mid*mid*mid;
            T ratio = p_val / q_val;

            if (ratio < -epsilon) {
                all_nonneg = false;
                break;
            }
        }

        if (all_nonneg) {
            // Determine if endpoints are inclusive
            bool left_inc = false, right_inc = false;

            // Check left endpoint
            if (i > 0) {
                T q_left = Q[0] + Q[1]*left + Q[2]*left*left + Q[3]*left*left*left;
                if (std::abs(q_left) > epsilon) {
                    bool left_valid = true;
                    for (int j = 0; j < 4; ++j) {
                        T p_left = P[j][0] + P[j][1]*left + P[j][2]*left*left + P[j][3]*left*left*left;
                        if (p_left / q_left < -epsilon) {
                            left_valid = false;
                            break;
                        }
                    }
                    left_inc = left_valid;
                }
            }

            // Check right endpoint
            if (i + 2 < test_points.size()) {
                T q_right = Q[0] + Q[1]*right + Q[2]*right*right + Q[3]*right*right*right;
                if (std::abs(q_right) > epsilon) {
                    bool right_valid = true;
                    for (int j = 0; j < 4; ++j) {
                        T p_right = P[j][0] + P[j][1]*right + P[j][2]*right*right + P[j][3]*right*right*right;
                        if (p_right / q_right < -epsilon) {
                            right_valid = false;
                            break;
                        }
                    }
                    right_inc = right_valid;
                }
            }

            valid_intervals.push_back(Interval<T>(left, right, left_inc, right_inc));
        }
    }

    return valid_intervals;
}

/**
 * @brief Solve cubic equation: P[0] + P[1]x + P[2]x² + P[3]x³ = 0
 * Returns number of real roots
 */
template <typename T>
int solve_cubic_real(const T P[4], T roots[3], T epsilon) {
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
 * @brief Verify that v and w are parallel at given barycentric coordinates (triangle)
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

/**
 * @brief Compute characteristic polynomials for tetrahedron PV problem
 *
 * For a tetrahedron with vectors V and W at 4 vertices, computes:
 * - Q[4]: denominator polynomial (degree 3)
 * - P[4][4]: numerator polynomials for 4 barycentric coordinates (each degree 3)
 *
 * Barycentric coordinates are: μᵢ(λ) = Pᵢ(λ) / Q(λ)
 */
template <typename T>
void characteristic_polynomials_pv_tetrahedron(const T V[4][3], const T W[4][3], T Q[4], T P[4][4]) {
    // Linear transformation: map first 3 vertices to axes, 4th vertex to origin
    T A[3][3] = {
        {V[0][0] - V[3][0], V[1][0] - V[3][0], V[2][0] - V[3][0]},
        {V[0][1] - V[3][1], V[1][1] - V[3][1], V[2][1] - V[3][1]},
        {V[0][2] - V[3][2], V[1][2] - V[3][2], V[2][2] - V[3][2]}
    };
    T B[3][3] = {
        {W[0][0] - W[3][0], W[1][0] - W[3][0], W[2][0] - W[3][0]},
        {W[0][1] - W[3][1], W[1][1] - W[3][1], W[2][1] - W[3][1]},
        {W[0][2] - W[3][2], W[1][2] - W[3][2], W[2][2] - W[3][2]}
    };
    T a[3] = {V[3][0], V[3][1], V[3][2]};
    T b[3] = {W[3][0], W[3][1], W[3][2]};

    // Right-hand side polynomials: rhs[i] = -a[i] + b[i]*λ
    T rhs[3][2] = {
        {-a[0], b[0]},
        {-a[1], b[1]},
        {-a[2], b[2]}
    };

    // Compute denominator Q: det(A - λB)
    characteristic_polynomial_3x3(A, B, Q);

    // Build adjugate matrix of (A - λB)
    // Each entry is a polynomial of degree 2
    T adj[3][3][3];
    characteristic_polynomial_2x2(A[1][1], A[1][2], A[2][1], A[2][2], B[1][1], B[1][2], B[2][1], B[2][2], adj[0][0]);
    characteristic_polynomial_2x2(A[1][0], A[1][2], A[2][0], A[2][2], B[1][0], B[1][2], B[2][0], B[2][2], adj[1][0]);
    characteristic_polynomial_2x2(A[1][0], A[1][1], A[2][0], A[2][1], B[1][0], B[1][1], B[2][0], B[2][1], adj[2][0]);
    characteristic_polynomial_2x2(A[0][1], A[0][2], A[2][1], A[2][2], B[0][1], B[0][2], B[2][1], B[2][2], adj[0][1]);
    characteristic_polynomial_2x2(A[0][0], A[0][2], A[2][0], A[2][2], B[0][0], B[0][2], B[2][0], B[2][2], adj[1][1]);
    characteristic_polynomial_2x2(A[0][0], A[0][1], A[2][0], A[2][1], B[0][0], B[0][1], B[2][0], B[2][1], adj[2][1]);
    characteristic_polynomial_2x2(A[0][1], A[0][2], A[1][1], A[1][2], B[0][1], B[0][2], B[1][1], B[1][2], adj[0][2]);
    characteristic_polynomial_2x2(A[0][0], A[0][2], A[1][0], A[1][2], B[0][0], B[0][2], B[1][0], B[1][2], adj[1][2]);
    characteristic_polynomial_2x2(A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1], adj[2][2]);

    // Fix signs for adjugate matrix
    polynomial_scalar_multiply(adj[0][1], 2, T(-1));
    polynomial_scalar_multiply(adj[1][0], 2, T(-1));
    polynomial_scalar_multiply(adj[1][2], 2, T(-1));
    polynomial_scalar_multiply(adj[2][1], 2, T(-1));

    // Compute P[i] = adj[i]^T * rhs
    // P[i] = sum_j adj[i][j] * rhs[j]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) P[i][j] = T(0);

        for (int j = 0; j < 3; ++j) {
            T poly[4] = {0};
            polynomial_multiply(adj[i][j], 2, rhs[j], 1, poly);
            polynomial_add_inplace(P[i], 3, poly, 3);
        }
    }

    // Fourth coordinate: P[3] = Q - P[0] - P[1] - P[2]
    P[3][0] = Q[0] - P[0][0] - P[1][0] - P[2][0];
    P[3][1] = Q[1] - P[0][1] - P[1][1] - P[2][1];
    P[3][2] = Q[2] - P[0][2] - P[1][2] - P[2][2];
    P[3][3] = Q[3] - P[0][3] - P[1][3] - P[2][3];
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
    // Check for degenerate case: all vectors parallel at vertices
    bool all_parallel = true;
    for (int i = 0; i < 4; ++i) {
        T cross[3];
        cross_product3(V[i], W[i], cross);
        if (vector_norm3(cross) > epsilon) {
            all_parallel = false;
            break;
        }
    }

    if (all_parallel) {
        // Entire tetrahedron is a PV region - this is degenerate
        return false;
    }

    // Compute characteristic polynomials
    T Q[4], P[4][4];
    characteristic_polynomials_pv_tetrahedron(V, W, Q, P);

    // Check if Q is zero (degenerate)
    bool q_zero = true;
    for (int i = 0; i <= 3; ++i) {
        if (std::abs(Q[i]) > epsilon) {
            q_zero = false;
            break;
        }
    }
    if (q_zero) return false;

    // Store polynomials in segment
    segment.Q.coeffs[0] = Q[0];
    segment.Q.coeffs[1] = Q[1];
    segment.Q.coeffs[2] = Q[2];
    segment.Q.coeffs[3] = Q[3];

    for (int i = 0; i < 4; ++i) {
        segment.P[i].coeffs[0] = P[i][0];
        segment.P[i].coeffs[1] = P[i][1];
        segment.P[i].coeffs[2] = P[i][2];
        segment.P[i].coeffs[3] = P[i][3];
    }

    // Find valid parameter ranges where all barycentric coordinates are non-negative
    // Note: We don't need to check ≤ 1 because they sum to 1 by construction!
    auto intervals = solve_barycentric_inequalities(P, Q, epsilon);

    if (intervals.empty()) {
        return false; // No valid parameter range
    }

    // Use the first (typically largest) valid interval
    // In practice, there's usually only one interval for well-behaved cases
    segment.lambda_min = intervals[0].min;
    segment.lambda_max = intervals[0].max;

    // Clamp to reasonable bounds if intervals extend to infinity
    if (segment.lambda_min < -100) segment.lambda_min = -100;
    if (segment.lambda_max > 100) segment.lambda_max = 100;

    return true;
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
