#pragma once

#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

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

    // Tetrahedron vertex coordinates for physical position computation
    std::array<std::array<double, 3>, 4> tet_vertices;  // [4 vertices][x,y,z]

    PVCurveSegment() : simplex_id(-1), lambda_min(0.0), lambda_max(1.0) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                tet_vertices[i][j] = 0.0;
            }
        }
    }

    // Evaluate curve at parameter lambda (get barycentric coordinates)
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

    // Get physical (x,y,z) coordinates at parameter lambda
    std::array<double, 3> get_physical_coords(double lambda) const {
        auto bary = get_barycentric(lambda);

        std::array<double, 3> pos = {0.0, 0.0, 0.0};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                pos[j] += bary[i] * tet_vertices[i][j];
            }
        }
        return pos;
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

// ============================================================================
// Simulation of Simplicity (SoS) for PV
// ============================================================================

/**
 * @brief SoS field perturbation for vertex (vertex_idx, component).
 *
 * Returns a small positive value δ that is unique to each (vertex, component)
 * pair and decreases geometrically with vertex index:
 *
 *   δ(i, j) = SOS_EPS / 2^(6*(i%8) + j%6)
 *
 * Interpretation: symbolically, this represents ε^(2^k) → 0⁺.
 * Adding δ to each field component at each vertex before solving ensures:
 *   1. No puncture lands exactly on a simplex edge or vertex.
 *   2. The characteristic polynomial has generically distinct real roots.
 *   3. Barycentric coordinates at solutions are generically strictly positive.
 *
 * The magnitude (1e-8 to ~1e-22) is:
 *   - Small enough to preserve topology (field gradients are O(1))
 *   - Large enough to break floating-point exact zeros
 */
static constexpr double SOS_EPS = 1e-8;

template <typename T>
inline T sos_perturbation(uint64_t vertex_idx, int component) {
    // k in [0,47]: perturbation range [SOS_EPS/2^47, SOS_EPS] = [7e-23, 1e-8]
    int k = (int)((vertex_idx % 8) * 6 + component % 6);
    return T(SOS_EPS) / T(uint64_t(1) << k);
}

// Forward declaration — defined after the quantization and integer polynomial
// sections (Subtasks 1–3) which appear later in this file.
inline int discriminant_sign_i128(const __int128 P[4]);

/**
 * @brief Solve cubic with SoS tie-breaking for the disc == 0 (tangent) case.
 *
 * Identical to solve_cubic_real except: when |disc| is below the SoS threshold,
 * first query the exact integer discriminant (Subtask 3).  If that returns a
 * definite sign, use the appropriate root formula.  Only if the exact sign is
 * also zero (true repeated root) does the SoS min-idx tie-break fire.
 *
 * @param P_i128  Integer polynomial from Subtask 2 (may be nullptr for legacy calls).
 */
template <typename T>
int solve_cubic_real_sos(const T P[4], T roots[3], const uint64_t* indices,
                         const __int128* P_i128 = nullptr,
                         T epsilon = std::numeric_limits<T>::epsilon()) {
    if (std::abs(P[3]) < epsilon) {
        // Degenerate to quadratic / linear — same as solve_cubic_real
        if (std::abs(P[2]) < epsilon) {
            if (std::abs(P[1]) < epsilon) return 0;
            roots[0] = -P[0] / P[1];
            return 1;
        }
        T disc = P[1]*P[1] - 4*P[2]*P[0];
        if (disc < 0) return 0;
        if (std::abs(disc) < epsilon) { roots[0] = -P[1]/(2*P[2]); return 1; }
        T sq = std::sqrt(disc);
        roots[0] = (-P[1]+sq)/(2*P[2]);
        roots[1] = (-P[1]-sq)/(2*P[2]);
        return 2;
    }

    T b = P[2]/P[3], c = P[1]/P[3], d = P[0]/P[3];
    T q = (3*c - b*b)/9;
    T r = (-27*d + b*(9*c - 2*b*b))/54;
    T disc = q*q*q + r*r;
    T term1 = b/3;

    // SoS threshold: proportional to scale of discriminant terms
    T sos_disc_eps = epsilon * (std::abs(q*q*q) + std::abs(r*r) + epsilon);

    if (disc > sos_disc_eps) {
        // One real root (float disc clearly positive)
        T s = r + std::sqrt(disc);
        s = (s < 0) ? -std::pow(-s, 1.0/3.0) : std::pow(s, 1.0/3.0);
        T t = r - std::sqrt(disc);
        t = (t < 0) ? -std::pow(-t, 1.0/3.0) : std::pow(t, 1.0/3.0);
        roots[0] = -term1 + s + t;
        return 1;
    } else if (disc < -sos_disc_eps) {
        // Three distinct real roots (float disc clearly negative)
        T mq = -q;
        T dum1 = std::acos(r / std::sqrt(mq*mq*mq));
        T r13 = 2*std::sqrt(mq);
        roots[0] = -term1 + r13*std::cos(dum1/3);
        roots[1] = -term1 + r13*std::cos((dum1 + 2*M_PI)/3);
        roots[2] = -term1 + r13*std::cos((dum1 + 4*M_PI)/3);
        return 3;
    } else {
        // Float disc ≈ 0: consult exact integer discriminant (Subtask 3).
        // discriminant_sign_i128 returns:
        //   +1 → exactly one real root       (Δ > 0 for cubic convention)
        //   -1 → exactly three distinct roots (Δ < 0)
        //    0 → repeated root OR overflow guard → fall through to SoS tie-break
        int exact_sign = P_i128 ? discriminant_sign_i128(P_i128) : 0;

        if (exact_sign > 0) {
            // Exact: one real root.  Use disc > 0 formula, clamping disc ≥ 0.
            T safe = (disc > T(0)) ? disc : T(0);
            T sq   = std::sqrt(safe);
            T s = r + sq; s = (s < 0) ? -std::pow(-s, 1.0/3.0) : std::pow(s, 1.0/3.0);
            T t = r - sq; t = (t < 0) ? -std::pow(-t, 1.0/3.0) : std::pow(t, 1.0/3.0);
            roots[0] = -term1 + s + t;
            return 1;
        } else if (exact_sign < 0) {
            // Exact: three distinct real roots.  Use disc < 0 formula,
            // clamping -q ≥ 0 in case float rounding made q slightly positive.
            T mq = -q > T(0) ? -q : T(0);
            if (mq < epsilon) {
                // Numerically indistinguishable from triple root — single root.
                T r13 = (r < 0) ? -std::pow(-r, 1.0/3.0) : std::pow(r, 1.0/3.0);
                roots[0] = -term1 + 2*r13;
                return 1;
            }
            T arg  = r / std::sqrt(mq*mq*mq);
            // clamp to [-1,1] in case float rounding pushes it slightly out
            arg = arg < T(-1) ? T(-1) : (arg > T(1) ? T(1) : arg);
            T dum1 = std::acos(arg);
            T r13  = 2*std::sqrt(mq);
            roots[0] = -term1 + r13*std::cos(dum1/3);
            roots[1] = -term1 + r13*std::cos((dum1 + 2*M_PI)/3);
            roots[2] = -term1 + r13*std::cos((dum1 + 4*M_PI)/3);
            return 3;
        } else {
            // Exact discriminant is zero (or overflow guard returned 0):
            // true repeated root → SoS min-idx tie-break.
            uint64_t min_idx = indices ? std::min({indices[0], indices[1], indices[2]})
                                       : uint64_t(0);
            T r13 = (r < 0) ? -std::pow(-r, 1.0/3.0) : std::pow(r, 1.0/3.0);
            roots[0] = -term1 + 2*r13;
            roots[1] = -(r13 + term1);
            if (min_idx % 2 == 0) {
                return 1;   // tangent excluded
            } else {
                roots[2] = roots[1];
                return (std::abs(roots[0] - roots[1]) < epsilon) ? 1 : 3;
            }
        }
    }
}

// ============================================================================
// Field Quantization  (Subtask 1 of the exact __int128 pipeline)
// ============================================================================
//
// DESIGN RATIONALE
// ----------------
// The floating-point solve chain has one unavoidable source of inexactness:
// every predicate (discriminant sign, barycentric sign) is decided with a
// tolerance (sos_disc_eps, bary_threshold).  Replacing those tolerances with
// exact integer arithmetic requires that all polynomial coefficients be exact
// integers — which is achieved by multiplying each field component by a fixed
// power-of-two scale before rounding to int64_t.
//
// SoS field perturbation (~1e-8) is intentionally NOT applied before
// quantization: at scale 2^QUANT_BITS = 2^20 ≈ 10^6, a perturbation of 1e-8
// rounds to zero.  In the exact integer pipeline, degeneracies (disc == 0,
// nu[k] == 0) are resolved by pure index-based tie-breaking — no field
// nudging is needed.
//
// OVERFLOW ANALYSIS  (for characteristic polynomial in __int128)
// ---------------------------------------------------------------
// Let M = max absolute field value across all 6 components × 3 vertices.
//
//   Quantized value:          |f̂|  ≤ M · 2^QUANT_BITS
//   Triple product (det term): ≤ M³ · 2^(3·QUANT_BITS)
//   Full 3×3 det (6 terms):   ≤ 6·M³ · 2^(3·QUANT_BITS)
//
// With QUANT_BITS = 20:
//   M = 1      : 6 · 2^60  ≈ 2^63   (fits int64_t)
//   M = 1000   : 6 · 10^9 · 2^60 ≈ 2^93  (fits __int128, 127-bit signed)
//   M = 10^6   : 6 · 10^18 · 2^60 ≈ 2^120 (fits __int128)
//   M = 5×10^6 : approaches 2^127 — borderline; use QUANT_BITS=16 if needed.
//
// The characteristic polynomial coefficients P1, P2 involve sums of products
// mixing V and W values; the same analysis applies since both are scaled by
// 2^QUANT_BITS.  All intermediate quantities in Subtasks 2–5 stay within 127
// bits for M ≤ 10^6, which covers all practical simulation fields.

static constexpr int     QUANT_BITS  = 20;
static constexpr int64_t QUANT_SCALE = int64_t(1) << QUANT_BITS;  // 1,048,576

/**
 * @brief Quantize one floating-point value to int64_t.
 *
 * Maps  x  →  round(x · QUANT_SCALE).
 * Quantization error: ≤ 0.5 / QUANT_SCALE ≈ 4.8×10⁻⁷ (in original units).
 * This corresponds to ~20 bits of precision after the binary point, matching
 * six significant decimal digits — sufficient for most physical field data.
 */
inline int64_t quant(double x) {
    return static_cast<int64_t>(std::llround(x * static_cast<double>(QUANT_SCALE)));
}

/**
 * @brief Quantize a 3×3 field array [vertex][component] to int64_t.
 *
 * Input V is indexed as V[vertex][component] (same layout as the PV solver).
 * Output Vq carries the same layout with each entry scaled by QUANT_SCALE and
 * rounded to the nearest integer.
 *
 * Precondition: |V[i][j]| < 5×10^6  (see overflow analysis above).
 */
template <typename T>
void quantize_field_3x3(const T V[3][3], int64_t Vq[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Vq[i][j] = quant(static_cast<double>(V[i][j]));
}

// ============================================================================
// Exact Integer Characteristic Polynomial  (Subtask 2)
// ============================================================================
//
// Both A and B are quantized field matrices (entries = field × QUANT_SCALE).
// The characteristic polynomial det(A − λB) has coefficients that are exact
// integers — no rounding at any step.
//
// KEY OBSERVATION: multiplying A and B by the same scalar s gives
//   det(sA − λ·sB) = s³ · det(A − λB),
// so the roots λ* are identical to those of the floating-point polynomial.
// The integer polynomial can therefore be used to determine the discriminant
// sign (Subtask 3) and to find isolating intervals (Subtask 4) for the same
// λ values that the float solver would return.

/**
 * @brief 3×3 integer determinant with __int128 accumulation.
 *
 * Each triple product of int64_t values is at most (2^63)^3 / 3 ≈ 2^188,
 * which exceeds __int128.  BUT with QUANT_BITS=20 and |field| < 5×10^6,
 * each quantized entry fits in ~43 bits, so each triple product ≤ 2^129 —
 * still a bit tight.  We therefore cast each factor individually before
 * the multiply, letting __int128 absorb the products safely.
 *
 * Safe range: |a[i][j]| < 2^41  (≈ 2×10^12), giving triple products < 2^123.
 */
inline __int128 det3_i128(const int64_t a[3][3]) {
    auto e = [&](int i, int j) -> __int128 { return (__int128)a[i][j]; };
    return e(0,0) * (e(1,1)*e(2,2) - e(2,1)*e(1,2))
         - e(0,1) * (e(1,0)*e(2,2) - e(2,0)*e(1,2))
         + e(0,2) * (e(1,0)*e(2,1) - e(2,0)*e(1,1));
}

/**
 * @brief Exact characteristic polynomial det(A − λB) with __int128 coefficients.
 *
 * Transcription of characteristic_polynomial_3x3<T> with every multiplication
 * promoted to __int128 before accumulation.  The formula is the direct
 * cofactor expansion of the 3×3 determinant as a polynomial in λ:
 *
 *   P[0] =  det(A)            (constant term)
 *   P[1] = -d/dλ det(A-λB)|λ=0  (18 terms: two A factors, one B factor each)
 *   P[2] =  ...               (18 terms: one A factor, two B factors each)
 *   P[3] = -det(B)            (cubic term)
 *
 * Coefficient magnitudes (M = max|entry| ≤ |field|_max × QUANT_SCALE):
 *   |P[0]|, |P[3]| ≤ 6·M³
 *   |P[1]|, |P[2]| ≤ 18·M³
 * With M = 10^6 · 2^20 ≈ 2^40: 18·(2^40)^3 = 18·2^120 ≈ 2^124 < 2^127. ✓
 */
inline void characteristic_polynomial_3x3_i128(const int64_t A[3][3],
                                                const int64_t B[3][3],
                                                __int128 P[4]) {
    auto a = [&](int i, int j) -> __int128 { return (__int128)A[i][j]; };
    auto b = [&](int i, int j) -> __int128 { return (__int128)B[i][j]; };

    P[0] = det3_i128(A);

    P[1] = a(1,2)*a(2,1)*b(0,0) - a(1,1)*a(2,2)*b(0,0) - a(1,2)*a(2,0)*b(0,1)
          +a(1,0)*a(2,2)*b(0,1) + a(1,1)*a(2,0)*b(0,2) - a(1,0)*a(2,1)*b(0,2)
          -a(0,2)*a(2,1)*b(1,0) + a(0,1)*a(2,2)*b(1,0) + a(0,2)*a(2,0)*b(1,1)
          -a(0,0)*a(2,2)*b(1,1) - a(0,1)*a(2,0)*b(1,2) + a(0,0)*a(2,1)*b(1,2)
          +a(0,2)*a(1,1)*b(2,0) - a(0,1)*a(1,2)*b(2,0) - a(0,2)*a(1,0)*b(2,1)
          +a(0,0)*a(1,2)*b(2,1) + a(0,1)*a(1,0)*b(2,2) - a(0,0)*a(1,1)*b(2,2);

    P[2] =-a(2,2)*b(0,1)*b(1,0) + a(2,1)*b(0,2)*b(1,0) + a(2,2)*b(0,0)*b(1,1)
          -a(2,0)*b(0,2)*b(1,1) - a(2,1)*b(0,0)*b(1,2) + a(2,0)*b(0,1)*b(1,2)
          +a(1,2)*b(0,1)*b(2,0) - a(1,1)*b(0,2)*b(2,0) - a(0,2)*b(1,1)*b(2,0)
          +a(0,1)*b(1,2)*b(2,0) - a(1,2)*b(0,0)*b(2,1) + a(1,0)*b(0,2)*b(2,1)
          +a(0,2)*b(1,0)*b(2,1) - a(0,0)*b(1,2)*b(2,1) + a(1,1)*b(0,0)*b(2,2)
          -a(1,0)*b(0,1)*b(2,2) - a(0,1)*b(1,0)*b(2,2) + a(0,0)*b(1,1)*b(2,2);

    P[3] = -det3_i128(B);
}

// ============================================================================
// Exact Discriminant Sign  (Subtask 3)
// ============================================================================
//
// The discriminant of  χ(λ) = P[3]λ³ + P[2]λ² + P[1]λ + P[0]  is:
//
//   Δ = 18·P3·P2·P1·P0 − 4·P2³·P0 + P2²·P1² − 4·P3·P1³ − 27·P3²·P0²
//
// With raw __int128 polynomial coefficients (each up to ~2^94 for M=1000),
// the degree-4 terms reach (2^94)^4 = 2^376 — well beyond __int128.
//
// STRATEGY — GCD normalization + overflow guard:
//   1. Divide all four coefficients by their GCD (does not change root
//      structure or sign of Δ, since Δ scales as g^4).
//   2. If the maximum normalized coefficient < 2^30, all degree-4 terms
//      fit in __int128 (54·(2^30)^4 ≈ 2^125 < 2^127). Compute exactly.
//   3. Otherwise coefficients are large ↔ Δ is far from zero relative to
//      coefficient scale → the float discriminant sign is reliable. Return
//      0 to signal "use float fallback", handled by the caller.
//
// OVERFLOW PROOF for the 2^30 threshold (let M30 = 2^30):
//   |18abcd|  ≤ 18 · M30^4 = 18 · 2^120 ≈ 2^124.2
//   |4b³d|    ≤  4 · M30^4              ≈ 2^122
//   |b²c²|    ≤      M30^4              ≈ 2^120
//   |4ac³|    ≤  4 · M30^4              ≈ 2^122
//   |27a²d²|  ≤ 27 · M30^4              ≈ 2^124.75
//   Sum (signed): ≤ 54 · M30^4          ≈ 2^125.75 < 2^127  ✓
//
// Intermediate products also stay in range:
//   b*b: ≤ M30^2 = 2^60; b*b*b: ≤ 2^90; b*b*b*d: ≤ 2^120  ✓

/**
 * @brief GCD of two __int128 values (Euclidean algorithm, handles negatives).
 */
inline __int128 gcd_i128(__int128 a, __int128 b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { __int128 t = a % b; a = b; b = t; }
    return a;
}

/**
 * @brief Exact sign of the discriminant of the integer cubic P_i128.
 *
 * @param P  Cubic coefficients [P0, P1, P2, P3]  (P[3] is the leading term).
 * @return  +1  →  three distinct real roots
 *           0  →  repeated root (or overflow guard triggered)
 *          -1  →  one real root, two complex conjugates
 *
 * Note: sign convention matches the standard cubic discriminant:
 *   Δ > 0 ↔ three distinct real roots, Δ < 0 ↔ one real root.
 */
inline int discriminant_sign_i128(const __int128 P[4]) {
    if (P[3] == 0) return 0;  // not a cubic

    // Step 1: GCD-normalize to minimize coefficient magnitude.
    __int128 g = gcd_i128(gcd_i128(P[0] < 0 ? -P[0] : P[0],
                                    P[1] < 0 ? -P[1] : P[1]),
                           gcd_i128(P[2] < 0 ? -P[2] : P[2],
                                    P[3] < 0 ? -P[3] : P[3]));
    if (g == 0) return 0;

    __int128 a = P[3] / g;   // leading coefficient
    __int128 b = P[2] / g;
    __int128 c = P[1] / g;
    __int128 d = P[0] / g;   // constant term

    // Step 2: overflow guard — if any coefficient ≥ 2^30, return 0.
    static constexpr __int128 THRESH = __int128(1) << 30;
    auto abs128 = [](__int128 x) { return x < 0 ? -x : x; };
    if (abs128(a) >= THRESH || abs128(b) >= THRESH ||
        abs128(c) >= THRESH || abs128(d) >= THRESH)
        return 0;  // large coefficients → float sign is reliable

    // Step 3: exact discriminant in __int128.
    // Δ = 18abcd − 4b³d + b²c² − 4ac³ − 27a²d²
    __int128 b2 = b*b,  b3 = b2*b;
    __int128 c2 = c*c,  c3 = c2*c;
    __int128 a2 = a*a;
    __int128 d2 = d*d;
    __int128 delta = 18*a*b*c*d  -  4*b3*d  +  b2*c2  -  4*a*c3  -  27*a2*d2;

    return (delta > 0) ? +1 : (delta < 0) ? -1 : 0;
}

// ============================================================================
// Sturm-Sequence Root Isolation  (Subtask 4)
// ============================================================================
//
// Given a cubic P(x) = P[3]x³+P[2]x²+P[1]x+P[0] (ascending-degree order),
// the Sturm sequence S₀,S₁,S₂,S₃ satisfies Sturm's theorem:
//
//   V(a) − V(b)  =  number of distinct real roots of P in (a, b]
//
// where V(x) = # sign changes in (S₀(x), S₁(x), S₂(x), S₃(x)), zeros skipped.
//
// DERIVATION (pseudo-remainder chain for cubic):
//   S₀ = P                                    (degree 3)
//   S₁ = P'  =  [p₁, 2p₂, 3p₃]               (degree 2)
//   S₂ = −prem(S₀, S₁)                        (degree 1)
//   S₃ = −prem(S₁, S₂)                        (constant)
//
// Computing S₂ = −prem(S₀, S₁):
//   prem(S₀,S₁) = 2p₃(3p₁p₃−p₂²)x + p₃(9p₀p₃−p₁p₂)
//   ⟹ S₂ = [ p₃(p₁p₂−9p₀p₃),  2p₃(p₂²−3p₁p₃) ]
//
// Computing S₃ = −prem(S₁, S₂):
//   with s₂₀ = S₂[0], s₂₁ = S₂[1]:
//   prem(S₁,S₂) = p₁s₂₁²  −  2p₂s₂₁s₂₀  +  3p₃s₂₀²
//   ⟹ S₃ = −(p₁s₂₁²  −  2p₂s₂₁s₂₀  +  3p₃s₂₀²)
//
// NOTE: P[4] here is the floating-point (SoS-perturbed) characteristic
// polynomial, not the integer P_i128.  The float polynomial shares the same
// roots as the integer one (they are proportional) and has manageable
// coefficient magnitudes for double arithmetic.

struct SturmSeqDouble {
    double c[4][4];  // c[i][k]: coefficient of x^k in Sturm polynomial Sᵢ
    int    deg[4];   // effective degree of each Sᵢ
    int    n;        // length of the sequence (2, 3, or 4)
};

/// Evaluate polynomial with ascending-degree coefficients c[0..deg] at x.
static inline double eval_poly_sturm(const double* c, int deg, double x) {
    double r = c[deg];
    for (int i = deg - 1; i >= 0; --i) r = r * x + c[i];
    return r;
}

/// Build the Sturm sequence for cubic P (ascending-degree coefficients P[0..3]).
inline void build_sturm_double(const double P[4], SturmSeqDouble& seq) {
    double p0 = P[0], p1 = P[1], p2 = P[2], p3 = P[3];

    // S₀ = P
    seq.c[0][0] = p0; seq.c[0][1] = p1; seq.c[0][2] = p2; seq.c[0][3] = p3;
    seq.deg[0]  = 3;

    // S₁ = P'
    seq.c[1][0] = p1; seq.c[1][1] = 2*p2; seq.c[1][2] = 3*p3; seq.c[1][3] = 0;
    seq.deg[1]  = 2;

    // S₂ = −prem(S₀, S₁)
    double s20 = p3 * (p1*p2 - 9*p0*p3);
    double s21 = 2 * p3 * (p2*p2 - 3*p1*p3);
    seq.c[2][0] = s20; seq.c[2][1] = s21; seq.c[2][2] = 0; seq.c[2][3] = 0;
    seq.deg[2]  = (s21 != 0.0) ? 1 : 0;

    // S₃ = −prem(S₁, S₂)
    double s30 = -(p1*s21*s21 - 2*p2*s21*s20 + 3*p3*s20*s20);
    seq.c[3][0] = s30; seq.c[3][1] = 0; seq.c[3][2] = 0; seq.c[3][3] = 0;
    seq.deg[3]  = 0;

    // Effective length: truncate if S₂ or S₃ vanishes
    if (s21 == 0.0 && s20 == 0.0) seq.n = 2;  // P shares a factor with P'
    else if (s30 == 0.0)          seq.n = 3;  // P has a root of P'
    else                          seq.n = 4;
}

/// Count sign changes at x in the Sturm sequence (= # roots in (−∞, x]).
inline int sturm_count_at(const SturmSeqDouble& seq, double x) {
    int    changes = 0;
    double prev    = 0.0;
    for (int i = 0; i < seq.n; ++i) {
        double v = eval_poly_sturm(seq.c[i], seq.deg[i], x);
        if (v != 0.0) {
            if (prev != 0.0 && ((prev > 0.0) != (v > 0.0))) ++changes;
            prev = v;
        }
    }
    return changes;
}

/// Given float root estimate rf and the Sturm sequence of its polynomial,
/// find an isolating interval [lo_out, hi_out] containing exactly one root,
/// then bisect until width ≤ target_width.
/// Returns true on success, false if the root could not be isolated.
inline bool tighten_root_interval(const SturmSeqDouble& seq, double rf,
                                   double& lo_out, double& hi_out,
                                   double target_width = 1e-10) {
    // Initial bracket: small symmetric window around rf
    double scale = std::max(std::abs(rf), 1.0);
    double delta = scale * 1e-7;

    double lo = rf - delta, hi = rf + delta;
    int cnt = sturm_count_at(seq, lo) - sturm_count_at(seq, hi);

    // Phase 1: expand or shrink until exactly one root in bracket
    for (int iter = 0; iter < 120 && cnt != 1; ++iter) {
        if (cnt == 0)  delta *= 2.0;   // root not yet bracketed → expand
        else           delta *= 0.5;   // multiple roots → shrink to isolate one
        lo  = rf - delta;
        hi  = rf + delta;
        cnt = sturm_count_at(seq, lo) - sturm_count_at(seq, hi);
        if (delta > 1e14 || delta < 1e-300) break;
    }
    if (cnt != 1) return false;

    // Phase 2: bisect to target_width
    for (int iter = 0; iter < 200 && (hi - lo) > target_width; ++iter) {
        double mid = lo + (hi - lo) * 0.5;
        if (mid <= lo || mid >= hi) break;  // float convergence
        int half_cnt = sturm_count_at(seq, lo) - sturm_count_at(seq, mid);
        if (half_cnt == 1) hi = mid;
        else               lo = mid;
    }

    lo_out = lo;
    hi_out = hi;
    return true;
}

/// Isolate n_float_roots real roots of the cubic polynomial P[4]
/// (ascending-degree coefficients) using Sturm bisection.
///
/// @param P            Float cubic coefficients [P₀, P₁, P₂, P₃].
/// @param lo_out       Output lower bounds (length n_float_roots).
/// @param hi_out       Output upper bounds (length n_float_roots).
/// @param float_roots  Float root estimates (from solve_cubic_real_sos).
/// @param n_float_roots Number of estimates.
/// @return             Number of roots successfully isolated.
inline int isolate_cubic_roots(const double P[4],
                                double lo_out[3], double hi_out[3],
                                const double float_roots[3], int n_float_roots) {
    if (n_float_roots == 0) return 0;
    SturmSeqDouble seq;
    build_sturm_double(P, seq);
    int n_iso = 0;
    for (int i = 0; i < n_float_roots; ++i) {
        if (tighten_root_interval(seq, float_roots[i], lo_out[n_iso], hi_out[n_iso]))
            ++n_iso;
    }
    return n_iso;
}

// ============================================================================
// Exact Barycentric Sign via Sturm Count on Numerator Polynomial  (Subtask 6)
// ============================================================================
//
// For a fixed Sturm interval [λ_lo, λ_hi] containing exactly one root λ*,
// determine the sign of each barycentric coordinate μ_k(λ*) without any
// floating-point threshold.
//
// KEY IDEA
// --------
// The barycentric solve expresses μ_k as a rational function of λ:
//
//     μ_k(λ) = N_k(λ) / D(λ)
//
// where N_k and D are degree-4 polynomials derived from the 3×2
// overdetermined system M(λ)·[μ₀,μ₁]ᵀ = b(λ) (M and b each linear in λ):
//
//     M(λ)[r][c] = Mlin[r][c][0]  +  λ·Mlin[r][c][1]    (r=0..2, c=0..1)
//     b(λ)[r]    = blin[r][0]     +  λ·blin[r][1]
//
//     (MᵀM)[p][q](λ) = Σᵣ M[r][p]·M[r][q]  (quadratic in λ)
//     (Mᵀb)[p](λ)    = Σᵣ M[r][p]·b[r]      (quadratic in λ)
//
//     D     = det(MᵀM) = (MᵀM)₀₀·(MᵀM)₁₁ − (MᵀM)₀₁²   (degree 4)
//     N[0]  = (MᵀM)₁₁·(Mᵀb)[0] − (MᵀM)₀₁·(Mᵀb)[1]     (degree 4)
//     N[1]  = (MᵀM)₀₀·(Mᵀb)[1] − (MᵀM)₀₁·(Mᵀb)[0]     (degree 4)
//     N[2]  = D − N[0] − N[1]                             (degree 4)
//
// D(λ) = Σ (2×2 minor of M)² ≥ 0  (Cauchy-Binet) with D(λ*) > 0 for
// a simple eigenvalue with full-rank sub-system, so sign(μ_k) = sign(N_k).
//
// SIGN DETERMINATION
// ------------------
// Build the Sturm sequence of N_k and count sign changes at λ_lo and λ_hi:
//   V(λ_lo) − V(λ_hi) = 0  →  N_k has no root in (λ_lo, λ_hi]
//                          →  evaluate N_k(λ_lo) for the exact sign
//   V(λ_lo) − V(λ_hi) ≥ 1  →  N_k(λ*) = 0  →  SoS ownership rule
//
// This replaces all floating-point threshold comparisons with a pure
// root-count predicate.

/// Sturm sequence for a polynomial of degree ≤ 4.
struct SturmSeqDeg4 {
    double c[5][5];  // c[i][k]: coefficient of x^k in Sᵢ (ascending degree)
    int    deg[5];   // effective degree of Sᵢ
    int    n;        // length of sequence (1..5)
};

/// Float polynomial remainder: R = A mod B (ascending-degree).
/// Returns effective degree of R, or -1 if R is the zero polynomial.
static inline int poly_rem_d(const double* A, int dA, const double* B, int dB, double* R) {
    static constexpr double EPS_ZERO = 1e-200;
    // Copy A into R
    for (int k = 0; k <= dA; ++k) R[k] = A[k];
    for (int k = dA + 1; k <= 4; ++k) R[k] = 0.0;

    for (int d = dA; d >= dB; --d) {
        if (std::abs(R[d]) < EPS_ZERO) { R[d] = 0.0; continue; }
        double coeff = R[d] / B[dB];
        int    shift = d - dB;
        for (int i = 0; i <= dB; ++i) R[i + shift] -= coeff * B[i];
        R[d] = 0.0;
    }

    int dR = dB - 1;
    while (dR > 0 && std::abs(R[dR]) < EPS_ZERO) --dR;
    return (std::abs(R[dR]) < EPS_ZERO && dR == 0) ? -1 : dR;
}

/// Build Sturm sequence for polynomial P of degree degP ≤ 4 (ascending-degree).
inline void build_sturm_deg4(const double* P, int degP, SturmSeqDeg4& seq) {
    // Zero-initialise
    for (auto& row : seq.c) for (auto& v : row) v = 0.0;
    for (auto& d : seq.deg) d = 0;
    seq.n = 0;

    // S₀ = P
    for (int k = 0; k <= degP; ++k) seq.c[0][k] = P[k];
    seq.deg[0] = degP;
    seq.n = 1;
    if (degP == 0) return;

    // S₁ = P'
    for (int k = 1; k <= degP; ++k) seq.c[1][k - 1] = k * seq.c[0][k];
    seq.deg[1] = degP - 1;
    seq.n = 2;
    if (degP == 1) return;

    // Sᵢ₊₁ = −rem(Sᵢ₋₁, Sᵢ)
    for (int i = 2; i <= degP && i <= 4; ++i) {
        if (seq.deg[i - 1] == 0) {
            // Constant S_{i-1}: remainder is zero → stop
            break;
        }
        double rem[5] = {};
        int dRem = poly_rem_d(seq.c[i - 2], seq.deg[i - 2],
                               seq.c[i - 1], seq.deg[i - 1], rem);
        if (dRem < 0) break;  // zero remainder → sequence terminates
        for (int k = 0; k <= dRem; ++k) seq.c[i][k] = -rem[k];
        seq.deg[i] = dRem;
        seq.n = i + 1;
        if (dRem == 0) break;  // constant Sᵢ → done
    }
}

/// Sturm sign-change count at x for the degree-4 Sturm sequence.
inline int sturm_count_d4(const SturmSeqDeg4& seq, double x) {
    int    changes = 0;
    double prev    = 0.0;
    for (int i = 0; i < seq.n; ++i) {
        double v = eval_poly_sturm(seq.c[i], seq.deg[i], x);
        if (v != 0.0) {
            if (prev != 0.0 && ((prev > 0.0) != (v > 0.0))) ++changes;
            prev = v;
        }
    }
    return changes;
}

/// Compute degree-4 barycentric numerator polynomials N[3][5] and D[5].
///
/// @param Mlin  Linear polynomial coefficients of M(λ)[r][c] = Mlin[r][c][0] + λ·Mlin[r][c][1]
/// @param blin  Linear polynomial coefficients of b(λ)[r]    = blin[r][0]    + λ·blin[r][1]
/// @param N     Output: N[k][0..4] are the degree-4 numerator poly coefficients for μ_k
/// @param D     Output: D[0..4] are the degree-4 denominator poly (Gram determinant)
inline void compute_bary_numerators(
        const double Mlin[3][2][2], const double blin[3][2],
        double N[3][5], double D[5]) {
    // (MᵀM)[p][q](λ) — quadratic in λ
    double A[2][2][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int p = 0; p < 2; ++p)
            for (int q = 0; q < 2; ++q) {
                double m0p = Mlin[r][p][0], m1p = Mlin[r][p][1];
                double m0q = Mlin[r][q][0], m1q = Mlin[r][q][1];
                A[p][q][0] += m0p * m0q;
                A[p][q][1] += m0p * m1q + m1p * m0q;
                A[p][q][2] += m1p * m1q;
            }

    // (Mᵀb)[p](λ) — quadratic in λ
    double g[2][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int p = 0; p < 2; ++p) {
            double m0 = Mlin[r][p][0], m1 = Mlin[r][p][1];
            double b0 = blin[r][0],    b1 = blin[r][1];
            g[p][0] += m0 * b0;
            g[p][1] += m0 * b1 + m1 * b0;
            g[p][2] += m1 * b1;
        }

    // Multiply two degree-2 polynomials → degree-4
    auto mul2 = [](const double* P, const double* Q, double* R) {
        for (int k = 0; k < 5; ++k) R[k] = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R[i + j] += P[i] * Q[j];
    };

    // D = A₀₀·A₁₁ − A₀₁²
    double t0[5], t1[5];
    mul2(A[0][0], A[1][1], t0);
    mul2(A[0][1], A[0][1], t1);
    for (int k = 0; k < 5; ++k) D[k] = t0[k] - t1[k];

    // N[0] = A₁₁·g[0] − A₀₁·g[1]
    double a11g0[5], a01g1[5];
    mul2(A[1][1], g[0], a11g0);
    mul2(A[0][1], g[1], a01g1);
    for (int k = 0; k < 5; ++k) N[0][k] = a11g0[k] - a01g1[k];

    // N[1] = A₀₀·g[1] − A₀₁·g[0]
    double a00g1[5], a01g0[5];
    mul2(A[0][0], g[1], a00g1);
    mul2(A[0][1], g[0], a01g0);
    for (int k = 0; k < 5; ++k) N[1][k] = a00g1[k] - a01g0[k];

    // N[2] = D − N[0] − N[1]
    for (int k = 0; k < 5; ++k) N[2][k] = D[k] - N[0][k] - N[1][k];
}

/**
 * @brief Solve for parallel vectors on a triangle (2-simplex)
 *
 * Finds up to 3 puncture points where v × w = 0.
 *
 * @param V Vectors at 3 triangle vertices [3][3]
 * @param W Vectors at 3 triangle vertices [3][3]
 * @param punctures Output vector of puncture points
 * @param indices Global vertex indices for SoS perturbation (nullptr = no SoS)
 * @param epsilon Tolerance for numerical comparisons
 * @return Number of punctures found (0-3), or INT_MAX if degenerate (entire triangle is PV)
 */
template <typename T>
int solve_pv_triangle(const T V[3][3], const T W[3][3],
                     std::vector<PuncturePoint>& punctures,
                     const uint64_t* indices = nullptr,
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
                     const uint64_t* indices,
                     T epsilon) {
    punctures.clear();

    // ----------------------------------------------------------------
    // Subtask 1: quantize the original (unperturbed) field to int64_t.
    // ----------------------------------------------------------------
    int64_t Vq[3][3], Wq[3][3];
    quantize_field_3x3(V, Vq);
    quantize_field_3x3(W, Wq);

    // ----------------------------------------------------------------
    // Subtask 2: exact integer characteristic polynomial.
    //
    // Transpose Vq and Wq: the characteristic polynomial is
    //   det(VqT − λ·WqT)  where VqT[component][vertex].
    // This mirrors how the float path builds VT/WT below.
    //
    // P_i128[k] ≈ QUANT_SCALE³ · P_fp[k]   (quantization error only)
    //
    // The roots λ* are identical to the float polynomial's roots because
    // QUANT_SCALE³ is a common factor that cancels in det(A−λB)=0.
    //
    // P_i128 is not yet used for root-finding; Subtask 3 will extract
    // the exact discriminant sign from it.
    // ----------------------------------------------------------------
    int64_t VqT[3][3], WqT[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VqT[i][j] = Vq[j][i];
            WqT[i][j] = Wq[j][i];
        }
    __int128 P_i128[4];
    characteristic_polynomial_3x3_i128(VqT, WqT, P_i128);

    // ----------------------------------------------------------------
    // SoS field perturbation
    //
    // Add a unique, tiny positive delta to each field component at each
    // vertex.  The delta is based on the global vertex index so it is:
    //   - Deterministic and reproducible across calls
    //   - Unique per (vertex, component) pair
    //   - Ordered: lower vertex index → larger perturbation (dominates)
    //
    // Effect: any puncture that the unperturbed field would place exactly
    // on a simplex edge/vertex is displaced slightly into the interior of
    // exactly one triangle, removing the need for ad-hoc epsilon thresholds
    // or user-chosen field offsets.
    //
    // If indices == nullptr the perturbation is skipped (legacy behaviour).
    // ----------------------------------------------------------------
    T Vp[3][3], Wp[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (indices) {
                Vp[i][j] = V[i][j] + sos_perturbation<T>(indices[i], j);
                Wp[i][j] = W[i][j] + sos_perturbation<T>(indices[i], j + 3);
            } else {
                Vp[i][j] = V[i][j];
                Wp[i][j] = W[i][j];
            }
        }
    }

    // Check for degenerate case: all vectors parallel at all vertices
    bool all_parallel = true;
    for (int i = 0; i < 3; ++i) {
        T cross[3];
        cross_product3(Vp[i], Wp[i], cross);
        if (vector_norm3(cross) > epsilon) { all_parallel = false; break; }
    }
    if (all_parallel)
        return std::numeric_limits<int>::max();  // entire triangle is PV surface

    // Transpose for characteristic polynomial: columns = components, rows = vertices
    T VT[3][3], WT[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VT[i][j] = Vp[j][i];
            WT[i][j] = Wp[j][i];
        }

    // Characteristic polynomial: det(VT - λ WT) = 0  (cubic in λ)
    T P[4];
    characteristic_polynomial_3x3(VT, WT, P);

    // Solve cubic: use exact integer discriminant (Subtask 3) to decide
    // the root count when the float discriminant is near zero.
    T lambda[3];
    int n_roots = solve_cubic_real_sos(P, lambda, indices, P_i128, epsilon);

    // ----------------------------------------------------------------
    // Subtask 4: Sturm-sequence root isolation.
    //
    // Tighten each float root into a verified isolating interval [lo, hi]
    // (Sturm count difference = 1) and replace lambda[i] with the interval
    // midpoint.  This gives a better λ estimate for the subsequent
    // least-squares solve and filters out any spurious float roots that
    // are not genuine roots of P.
    //
    // The Sturm sequence is built from the float polynomial P[4] (the
    // SoS-perturbed characteristic polynomial), which has the same roots
    // as P_i128 and manageable coefficient magnitudes for double arithmetic.
    //
    // If isolation fails for a root (rare, can happen when two roots are
    // extremely close), the original float value is retained unchanged.
    // ----------------------------------------------------------------
    // Expose Sturm intervals for Subtask 5.
    // Default: degenerate point intervals [lambda[k], lambda[k]] so that
    // Subtask 5 has a valid interval even when isolation fails.
    double lambda_lo[3], lambda_hi[3];
    for (int k = 0; k < n_roots; ++k)
        lambda_lo[k] = lambda_hi[k] = (double)lambda[k];
    {
        double Pd[4] = { (double)P[0], (double)P[1], (double)P[2], (double)P[3] };
        double lambda_d[3] = { (double)lambda[0], (double)lambda[1], (double)lambda[2] };
        int n_iso = isolate_cubic_roots(Pd, lambda_lo, lambda_hi, lambda_d, n_roots);
        for (int k = 0; k < n_iso; ++k)
            lambda[k] = static_cast<T>((lambda_lo[k] + lambda_hi[k]) * 0.5);
    }

    // ----------------------------------------------------------------
    // Subtask 5 & 6: barycentric sign determination.
    //
    // Helper: evaluate ν at a given λ via the 3×2 least-squares solve.
    // ----------------------------------------------------------------
    auto eval_nu_at = [&](T lam, T nu_out[3]) {
        const T Ml[3][2] = {
            {(VT[0][0]-VT[0][2]) - lam*(WT[0][0]-WT[0][2]),
             (VT[0][1]-VT[0][2]) - lam*(WT[0][1]-WT[0][2])},
            {(VT[1][0]-VT[1][2]) - lam*(WT[1][0]-WT[1][2]),
             (VT[1][1]-VT[1][2]) - lam*(WT[1][1]-WT[1][2])},
            {(VT[2][0]-VT[2][2]) - lam*(WT[2][0]-WT[2][2]),
             (VT[2][1]-VT[2][2]) - lam*(WT[2][1]-WT[2][2])}
        };
        const T bl[3] = {
            -(VT[0][2] - lam*WT[0][2]),
            -(VT[1][2] - lam*WT[1][2]),
            -(VT[2][2] - lam*WT[2][2])
        };
        solve_least_square3x2(Ml, bl, nu_out, epsilon);
        nu_out[2] = T(1) - nu_out[0] - nu_out[1];
    };

    // ----------------------------------------------------------------
    // Subtasks 6 & 7: compute bary numerator polynomials N[3][5] and D[5].
    //
    // Expresses μ_k(λ) = N_k(λ)/D(λ) with degree-4 polynomials derived
    // from the linear-in-λ M(λ) and b(λ) matrices.
    // D = det(MᵀM) ≥ 0 is the Gram determinant (non-negative by Cauchy-Binet).
    // ----------------------------------------------------------------
    double Mlin[3][2][2], blin_arr[3][2];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c) {
            Mlin[r][c][0] = (double)(VT[r][c] - VT[r][2]);
            Mlin[r][c][1] = -(double)(WT[r][c] - WT[r][2]);
        }
        blin_arr[r][0] = -(double)VT[r][2];
        blin_arr[r][1] =  (double)WT[r][2];
    }
    double N_poly[3][5], D_poly[5];
    compute_bary_numerators(Mlin, blin_arr, N_poly, D_poly);

    // ----------------------------------------------------------------
    // Subtask 7: pre-compute the Sturm sequence for D(λ) once.
    //
    // D(λ) = det(MᵀM) ≥ 0 (Cauchy-Binet).  D(λ*) = 0 iff the projection
    // system is rank-deficient at λ*.  We certify D(λ*) > 0 by showing D
    // has no root in [λ_lo, λ_hi] — a Sturm root-count test — without any
    // floating-point threshold.
    //
    //   Sturm count V_D(lo) − V_D(hi) = 0  →  D has no root in (lo, hi]
    //                                    →  D(λ*) > 0  (since D ≥ 0 and D ≠ 0)
    //   Sturm count V_D(lo) − V_D(hi) ≥ 1  →  D(λ*) = 0  →  degenerate,
    //                                           apply SoS ownership rule
    //
    // This replaces the previous `d_lo > 1e-200` float guard.
    // ----------------------------------------------------------------
    SturmSeqDeg4 seq_D;
    {
        int degD = 4;
        while (degD > 0 && std::abs(D_poly[degD]) < 1e-200) --degD;
        build_sturm_deg4(D_poly, degD, seq_D);
    }

    // For each λ recover barycentric coords via least-squares null-vector
    for (int i = 0; i < n_roots; ++i) {
        if (std::abs(lambda[i]) <= epsilon) continue;  // λ=0 → V=0, trivially parallel

        // Bary coords at the Sturm-isolated midpoint
        T nu[3];
        eval_nu_at(lambda[i], nu);

        // ----------------------------------------------------------------
        // Subtasks 6–9: determine sign of νₖ(λ*) with certified exactness.
        //
        // Common helper used in both the certified-interval path (Subtasks 6–8)
        // and the degenerate-interval path (Subtask 9):
        //
        //   try_certify_nk_sign(k, lo, hi)
        //     1. Build Sturm seq for N_k; count roots of N_k in (lo, hi].
        //        V_Nk(lo)−V_Nk(hi) = 0 → N_k root-free in interval.
        //     2. Count roots of D in (lo, hi].
        //        V_D(lo)−V_D(hi)  = 0 → D(λ*) > 0 certified (D ≥ 0).
        //     3. Evaluate N_k(lo) with Higham error bound (Subtask 8):
        //        |N_k(lo)| > γ·cond → certified sign returned (+1 or −1).
        //     Returns 0 if any step fails (N_k or D has root in window, or
        //     evaluation is within rounding noise) → SoS ownership rule.
        //
        // Subtask 9 reuses the same helper for the degenerate interval,
        // replacing the old bary_threshold = 1e-10 fallback with a
        // machine-epsilon window [λ − δ, λ + δ] around the float root.
        //
        // The SoS ownership rule: triangle T claims the boundary puncture
        // on edge (vᵢ, vⱼ) iff global_idx(vₖ) < min(global_idx(vᵢ), vⱼ)).
        // ----------------------------------------------------------------

        // Returns +1 / −1 if sign of N_k in (lo,hi] is certified, else 0.
        auto try_certify_nk_sign = [&](int k, double lo, double hi) -> int {
            int degNk = 4;
            while (degNk > 0 && std::abs(N_poly[k][degNk]) < 1e-200) --degNk;

            SturmSeqDeg4 seq_nk;
            build_sturm_deg4(N_poly[k], degNk, seq_nk);
            if (sturm_count_d4(seq_nk, lo) - sturm_count_d4(seq_nk, hi) != 0)
                return 0;  // N_k has a root in (lo, hi]

            // N_k root-free in window: check D via pre-computed seq_D.
            if (sturm_count_d4(seq_D, lo) - sturm_count_d4(seq_D, hi) != 0)
                return 0;  // D has a root → degenerate system

            // Subtask 8: certified Horner sign at lo.
            double nk_lo = eval_poly_sturm(N_poly[k], degNk, lo);
            double ax = std::abs(lo);
            double cond_nk = std::abs(N_poly[k][degNk]);
            for (int d = degNk - 1; d >= 0; --d)
                cond_nk = cond_nk * ax + std::abs(N_poly[k][d]);
            static constexpr double EVAL_GAMMA =
                (2 * 4 + 2) * std::numeric_limits<double>::epsilon();
            if (std::abs(nk_lo) > EVAL_GAMMA * cond_nk)
                return (nk_lo > 0.0) ? +1 : -1;

            return 0;  // within rounding noise
        };

        bool have_interval = (lambda_lo[i] < lambda_hi[i]);

        auto sos_bary_inside = [&](int k) -> bool {
            double lo, hi;
            if (have_interval) {
                lo = lambda_lo[i];
                hi = lambda_hi[i];
            } else {
                // Subtask 9: degenerate interval — Sturm isolation failed for
                // this root (two cubic roots nearly coincident).  Use a
                // machine-epsilon window [λ − δ, λ + δ] to probe N_k's sign.
                //
                // The float root λ̂ has accuracy ~ 4u|λ̂| relative to the
                // true root λ*.  If N_k has no root within this window, the
                // sign of N_k(λ̂) is the same as sign(N_k(λ*)), and Subtask 8
                // certifies it.  If N_k does have a root here, μ_k(λ*) ≈ 0
                // and the SoS rule applies — exactly the boundary case.
                //
                // This replaces the previous bary_threshold = 1e-10 fallback,
                // making the degenerate-interval path threshold-free.
                double lam = (double)lambda[i];
                double delta = std::max(std::abs(lam) * (4.0 * std::numeric_limits<double>::epsilon()),
                                        std::numeric_limits<double>::min());
                lo = lam - delta;
                hi = lam + delta;
            }

            int sign = try_certify_nk_sign(k, lo, hi);
            if (sign > 0) return true;
            if (sign < 0) return false;

            // Boundary zone (N_k or D degenerate, or evaluation uncertain):
            // apply SoS min-index ownership rule.
            if (!indices) return (double)lambda[i] >= 0.0;  // legacy fallback
            int ii = (k + 1) % 3, jj = (k + 2) % 3;
            return indices[k] < std::min(indices[ii], indices[jj]);
        };
        if (!sos_bary_inside(0) || !sos_bary_inside(1) || !sos_bary_inside(2))
            continue;

        // Subtask 10: cross-product residual check removed.
        //
        // The old guard  |V×W| > 1e-2  was a heuristic fallback against
        // spurious solutions.  It is now both redundant and harmful:
        //
        //   Redundant: Subtask 7 certifies D(λ*) > 0, so M(λ*) is full-rank.
        //     The least-squares ν is well-conditioned, and V×W at the float
        //     solution is O(ε_quant · |W|) ≪ 1e-2 for any physically sane
        //     field magnitude.
        //
        //   Harmful: when SoS perturbation is active (indices != nullptr),
        //     the perturbed solution ν_p satisfies Vp×Wp = 0 but not V×W = 0.
        //     The discrepancy is |V×W| ≈ SOS_EPS · (1+|λ|) · |W|.
        //     For |W| ≳ 5·10^4 this exceeds 1e-2, falsely rejecting the
        //     solution.  At |W| = QUANT_SCALE = 10^6, rejection is certain.

        PuncturePoint p;
        p.lambda       = lambda[i];
        p.barycentric[0] = nu[0];
        p.barycentric[1] = nu[1];
        p.barycentric[2] = nu[2];
        p.coords_3d[0] = p.coords_3d[1] = p.coords_3d[2] = 0;
        punctures.push_back(p);
    }

    return (int)punctures.size();
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
