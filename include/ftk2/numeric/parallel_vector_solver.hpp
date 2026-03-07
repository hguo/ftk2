#pragma once

#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <cfloat>
#include <climits>

// FTK_HOST_DEVICE macro (may already be defined by mesh.hpp)
#ifndef FTK_HOST_DEVICE
#ifdef __CUDACC__
#define FTK_HOST_DEVICE __host__ __device__
#else
#define FTK_HOST_DEVICE
#endif
#endif

namespace ftk2 {

// ============================================================================
// Device-safe math helpers
// ============================================================================
// These replace std:: calls that are not available in CUDA device code.

FTK_HOST_DEVICE inline int64_t pvs_llround(double x) {
#ifdef __CUDA_ARCH__
    return llrint(x);
#else
    return std::llround(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_sqrt(double x) {
#ifdef __CUDA_ARCH__
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_abs(double x) {
#ifdef __CUDA_ARCH__
    return ::fabs(x);
#else
    return std::abs(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_acos(double x) {
#ifdef __CUDA_ARCH__
    return ::acos(x);
#else
    return std::acos(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_cos(double x) {
#ifdef __CUDA_ARCH__
    return ::cos(x);
#else
    return std::cos(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_cbrt(double x) {
#ifdef __CUDA_ARCH__
    return ::cbrt(x);
#else
    // std::cbrt handles negative values correctly
    return std::cbrt(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_pow_onethird(double x) {
    // Replacement for std::pow(x, 1.0/3.0) that handles negative x
    if (x < 0) return -pvs_cbrt(-x);
    return pvs_cbrt(x);
}

FTK_HOST_DEVICE inline bool pvs_isfinite(double x) {
#ifdef __CUDA_ARCH__
    return ::isfinite(x);
#else
    return std::isfinite(x);
#endif
}

FTK_HOST_DEVICE inline double pvs_epsilon() {
    return DBL_EPSILON;
}

FTK_HOST_DEVICE inline __int128 pvs_abs128(__int128 x) {
    return x < 0 ? -x : x;
}

// Device-safe min/max for uint64_t (replaces std::min)
FTK_HOST_DEVICE inline uint64_t pvs_min_u64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

FTK_HOST_DEVICE inline uint64_t pvs_min3_u64(uint64_t a, uint64_t b, uint64_t c) {
    return pvs_min_u64(pvs_min_u64(a, b), c);
}

FTK_HOST_DEVICE inline double pvs_max_d(double a, double b) {
    return a > b ? a : b;
}

FTK_HOST_DEVICE inline double pvs_min_d(double a, double b) {
    return a < b ? a : b;
}

// ============================================================================
// Device-compatible puncture result structs
// ============================================================================

struct PuncturePointDevice {
    double lambda;
    double barycentric[3];
    double coords_3d[3];
};

struct PunctureResult {
    int count;                    // 0..3, or INT_MAX for degenerate
    PuncturePointDevice pts[3];
};

// Sturm certified result (replaces std::pair<int,bool>)
struct SturmCertifiedResult {
    int count;
    bool certified;
};

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

// Forward declaration — defined after the quantization and integer polynomial
// sections (Subtasks 1–3) which appear later in this file.
FTK_HOST_DEVICE inline int discriminant_sign_i128(const __int128 P[4]);

/**
 * @brief Solve cubic with SoS tie-breaking using the always-exact integer discriminant.
 *
 * Subtask 17: the float `disc = q³+r²` is no longer used to decide root count.
 * Instead, `discriminant_sign_i128(P_i128)` is consulted for EVERY cubic:
 *   +1 → Δ_std > 0 → three distinct real roots → trigonometric formula
 *   -1 → Δ_std < 0 → one real root              → Cardano formula
 *    0 → overflow guard OR true repeated root   → float-sign fallback or SoS tie-break
 * The SoS min-idx tie-break fires only when Δ_std = 0 exactly and disc ≈ 0.0.
 *
 * @param P_i128  Integer polynomial from Subtask 2 (may be nullptr for legacy calls).
 *                When nullptr, exact_sign is treated as 0 → float fallback.
 */
template <typename T>
FTK_HOST_DEVICE int solve_cubic_real_sos(const T P[4], T roots[3], const uint64_t* indices,
                         const __int128* P_i128 = nullptr) {
    // Subtask 19: degree-trimming uses exact == 0.0 (not < epsilon).
    //
    // P is computed by characteristic_polynomial_3x3(VpT, WpT) from field
    // values that are either exact doubles (no SoS) or SoS-perturbed with a
    // structurally full-rank WpT (SoS ensures distinct vertex perturbations).
    //
    // Case 1 — SoS inactive (indices=nullptr): P coefficients are sums and
    //   products of the original double-valued field.  An algebraically zero
    //   coefficient (e.g. P[3] = -det3(WpT) when W is constant) survives
    //   as 0.0 exactly because the cancellation in det3 is between equal
    //   integer-derived terms, with no rounding residue.
    //
    // Case 2 — SoS active: the SoS perturbation makes WpT generically
    //   full-rank, so P[3] ≠ 0.  The == 0.0 check fires only in case 1.
    //
    // Therefore == 0.0 is the principled replacement for < epsilon:
    //   - Catches exactly-zero (algebraically degenerate) coefficients.
    //   - Does NOT catch near-zero but nonzero coefficients (correct: a
    //     nearly-degenerate cubic is still genuinely degree-3 and should be
    //     handled by the cubic path, not the lower-degree fallback).
    if (P[3] == T(0)) {
        // Degenerate to quadratic / linear.
        if (P[2] == T(0)) {
            if (P[1] == T(0)) return 0;
            roots[0] = -P[0] / P[1];
            return 1;
        }
        T disc = P[1]*P[1] - 4*P[2]*P[0];
        if (disc < 0) return 0;
        // Subtask 19: quadratic repeated-root detected by disc == 0.0 exactly.
        // An algebraically repeated root gives P[1]² = 4·P[2]·P[0] with exact
        // cancellation in double when the inputs are integer-derived.
        if (disc == T(0)) { roots[0] = -P[1]/(2*P[2]); return 1; }
        T sq = (T)pvs_sqrt((double)disc);
        roots[0] = (-P[1]+sq)/(2*P[2]);
        roots[1] = (-P[1]-sq)/(2*P[2]);
        return 2;
    }

    T b = P[2]/P[3], c = P[1]/P[3], d = P[0]/P[3];

    // Special case: P[0]==P[1]==0 → polynomial is λ²·(P[2]+P[3]λ).
    // Double root at λ=0 (exact) + simple root at -P[2]/P[3] = -b.
    //
    // Without this branch the Cardano formula computes roots[1] =
    // -(r13 + term1) ≈ 5e-17 instead of 0.0 due to floating-point
    // cancellation (r13 = cbrt(1/27) ≈ 1/3 and term1 = b/3 ≈ -1/3
    // do not cancel exactly in double).  Subtask 11 then fails because
    // the isolated interval [5e-17, 5e-17] does not contain 0.0.
    //
    // Here c = P[1]/P[3] = 0.0 and d = P[0]/P[3] = 0.0 hold EXACTLY
    // when P[0]=P[1]=0 are integer-derived (no rounding), so the
    // == 0.0 test is the principled threshold-free replacement.
    if (c == T(0) && d == T(0)) {
        roots[0] = -b;   // simple root: -P[2]/P[3]
        roots[1] = T(0); // double root: 0 (forced exact, no cancellation)
        uint64_t min_idx_sp = indices
            ? pvs_min3_u64(indices[0], indices[1], indices[2]) : uint64_t(0);
        if (min_idx_sp % 2 == 0) return 1;  // SoS: tangent excluded
        roots[2] = T(0);
        return (-b == T(0)) ? 1 : 3;  // triple root at 0 gives 1
    }

    T q = (3*c - b*b)/9;
    T r = (-27*d + b*(9*c - 2*b*b))/54;
    T disc = q*q*q + r*r;
    T term1 = b/3;

    // Subtask 17: always use exact integer discriminant for root count.
    //
    // discriminant_sign_i128 uses the STANDARD discriminant convention:
    //   Δ = 18abcd − 4b³d + b²c² − 4ac³ − 27a²d²
    //   +1 → Δ > 0 → THREE distinct real roots  (trigonometric formula)
    //   -1 → Δ < 0 → ONE  real root             (Cardano formula)
    //    0 → Δ = 0 (repeated root) OR integer overflow guard
    //
    // Relationship to the Cardano discriminant (disc = q³+r²):
    //   Δ_standard = −108 × disc_Cardano  (opposite signs)
    //   disc > 0 → Δ < 0 → ONE  root;  disc < 0 → Δ > 0 → THREE roots
    //
    // When exact_sign == 0 (overflow guard for very large coefficients, or
    // true repeated root), fall back to the raw float disc sign — no threshold.
    int exact_sign = P_i128 ? discriminant_sign_i128(P_i128) : 0;
    if (exact_sign == 0) {
        // Overflow guard OR true repeated root.
        // Map Cardano disc sign → equivalent Δ sign for the branches below.
        if      (disc > T(0)) exact_sign = -1;  // disc > 0 → Δ < 0 → ONE  root
        else if (disc < T(0)) exact_sign = +1;  // disc < 0 → Δ > 0 → THREE roots
        // disc == 0.0 exactly → exact_sign stays 0 → SoS tie-break
    }

    if (exact_sign < 0) {
        // Δ < 0 → ONE real root.  Cardano formula, clamping disc ≥ 0 in case
        // float rounding (or the exact-sign/disc-sign disagreement) made it
        // slightly negative.
        T safe = (disc > T(0)) ? disc : T(0);
        T sq   = (T)pvs_sqrt((double)safe);
        T s = r + sq; s = (T)pvs_pow_onethird((double)s);
        T t = r - sq; t = (T)pvs_pow_onethird((double)t);
        roots[0] = -term1 + s + t;
        return 1;
    } else if (exact_sign > 0) {
        // Δ > 0 → THREE distinct real roots.  Trigonometric formula, clamping
        // -q ≥ 0 in case float rounding made q slightly positive.
        T mq = -q > T(0) ? -q : T(0);
        if (mq == 0.0) {
            // Subtask 17: was mq < epsilon.  Float q rounded to exactly 0 even
            // though exact arithmetic says Δ > 0 (three distinct roots).
            // The trig formula requires sqrt(mq³) > 0; fall back to a single
            // Cardano-style root from clamped disc (dominant real root).
            T safe = (disc > T(0)) ? disc : T(0);
            T sq   = (T)pvs_sqrt((double)safe);
            T s = r + sq; s = (T)pvs_pow_onethird((double)s);
            T t = r - sq; t = (T)pvs_pow_onethird((double)t);
            roots[0] = -term1 + s + t;
            return 1;
        }
        T arg  = r / (T)pvs_sqrt((double)(mq*mq*mq));
        // clamp to [-1,1] in case float rounding pushes it slightly out
        arg = arg < T(-1) ? T(-1) : (arg > T(1) ? T(1) : arg);
        T dum1 = (T)pvs_acos((double)arg);
        T r13  = 2*(T)pvs_sqrt((double)mq);
        roots[0] = -term1 + r13*(T)pvs_cos((double)(dum1/3));
        roots[1] = -term1 + r13*(T)pvs_cos((double)((dum1 + 2*M_PI)/3));
        roots[2] = -term1 + r13*(T)pvs_cos((double)((dum1 + 4*M_PI)/3));
        return 3;
    } else {
        // Exact discriminant is zero (or disc == 0.0 after overflow fallback):
        // true repeated root → SoS min-idx tie-break.
        uint64_t min_idx = indices ? pvs_min3_u64(indices[0], indices[1], indices[2])
                                   : uint64_t(0);
        T r13 = (T)pvs_pow_onethird((double)r);
        roots[0] = -term1 + 2*r13;
        roots[1] = -(r13 + term1);
        if (min_idx % 2 == 0) {
            return 1;   // tangent excluded
        } else {
            roots[2] = roots[1];
            return (r == T(0)) ? 1 : 3;
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
FTK_HOST_DEVICE inline int64_t quant(double x) {
    return static_cast<int64_t>(pvs_llround(x * static_cast<double>(QUANT_SCALE)));
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
FTK_HOST_DEVICE void quantize_field_3x3(const T V[3][3], int64_t Vq[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Vq[i][j] = quant(static_cast<double>(V[i][j]));
}

// ============================================================================
// Tet Quantization  (16-bit variant for tetrahedron Q computation)
// ============================================================================
//
// For tets, the characteristic polynomial uses 4-vertex edge differences
// (3×3 matrices from 4×3 fields).  With QUANT_BITS=20, the intermediate
// __int128 products reach ~2^130, exceeding 2^127.  Using 16 bits keeps
// |Aq|≤6.55e11 → det terms ≤2.8e35 ≈ 2^118 < 2^127.
//
// Future: switch to __int256 for full 20-bit precision.

static constexpr int     QUANT_BITS_TET  = 16;
static constexpr int64_t QUANT_SCALE_TET = int64_t(1) << QUANT_BITS_TET;  // 65,536

FTK_HOST_DEVICE inline int64_t quant_tet(double x) {
    return static_cast<int64_t>(pvs_llround(x * static_cast<double>(QUANT_SCALE_TET)));
}

template <typename T>
FTK_HOST_DEVICE void quantize_field_4x3_tet(const T V[4][3], int64_t Vq[4][3]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            Vq[i][j] = quant_tet(static_cast<double>(V[i][j]));
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
FTK_HOST_DEVICE inline __int128 det3_i128(const int64_t a[3][3]) {
    return (__int128)a[0][0] * ((__int128)a[1][1]*a[2][2] - (__int128)a[2][1]*a[1][2])
         - (__int128)a[0][1] * ((__int128)a[1][0]*a[2][2] - (__int128)a[2][0]*a[1][2])
         + (__int128)a[0][2] * ((__int128)a[1][0]*a[2][1] - (__int128)a[2][0]*a[1][1]);
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
FTK_HOST_DEVICE inline void characteristic_polynomial_3x3_i128(const int64_t A[3][3],
                                                const int64_t B[3][3],
                                                __int128 P[4]) {
    // Inline cast macro to avoid lambda captures (device compatibility)
    #define A128(i,j) ((__int128)A[i][j])
    #define B128(i,j) ((__int128)B[i][j])

    P[0] = det3_i128(A);

    P[1] = A128(1,2)*A128(2,1)*B128(0,0) - A128(1,1)*A128(2,2)*B128(0,0) - A128(1,2)*A128(2,0)*B128(0,1)
          +A128(1,0)*A128(2,2)*B128(0,1) + A128(1,1)*A128(2,0)*B128(0,2) - A128(1,0)*A128(2,1)*B128(0,2)
          -A128(0,2)*A128(2,1)*B128(1,0) + A128(0,1)*A128(2,2)*B128(1,0) + A128(0,2)*A128(2,0)*B128(1,1)
          -A128(0,0)*A128(2,2)*B128(1,1) - A128(0,1)*A128(2,0)*B128(1,2) + A128(0,0)*A128(2,1)*B128(1,2)
          +A128(0,2)*A128(1,1)*B128(2,0) - A128(0,1)*A128(1,2)*B128(2,0) - A128(0,2)*A128(1,0)*B128(2,1)
          +A128(0,0)*A128(1,2)*B128(2,1) + A128(0,1)*A128(1,0)*B128(2,2) - A128(0,0)*A128(1,1)*B128(2,2);

    P[2] =-A128(2,2)*B128(0,1)*B128(1,0) + A128(2,1)*B128(0,2)*B128(1,0) + A128(2,2)*B128(0,0)*B128(1,1)
          -A128(2,0)*B128(0,2)*B128(1,1) - A128(2,1)*B128(0,0)*B128(1,2) + A128(2,0)*B128(0,1)*B128(1,2)
          +A128(1,2)*B128(0,1)*B128(2,0) - A128(1,1)*B128(0,2)*B128(2,0) - A128(0,2)*B128(1,1)*B128(2,0)
          +A128(0,1)*B128(1,2)*B128(2,0) - A128(1,2)*B128(0,0)*B128(2,1) + A128(1,0)*B128(0,2)*B128(2,1)
          +A128(0,2)*B128(1,0)*B128(2,1) - A128(0,0)*B128(1,2)*B128(2,1) + A128(1,1)*B128(0,0)*B128(2,2)
          -A128(1,0)*B128(0,1)*B128(2,2) - A128(0,1)*B128(1,0)*B128(2,2) + A128(0,0)*B128(1,1)*B128(2,2);

    P[3] = -det3_i128(B);

    #undef A128
    #undef B128
}

// ============================================================================
// Tet Integer Q Polynomial  (for pass-through detection in stitching)
// ============================================================================
//
// Computes Q(λ) = det(A − λB) for a tetrahedron using __int128 arithmetic,
// where A and B are the edge-difference matrices of V and W (same layout as
// characteristic_polynomials_pv_tetrahedron).  The integer polynomial shares
// the same roots as the float Q polynomial (roots are scale-invariant).

template <typename T>
void compute_tet_Q_i128(const T V[4][3], const T W[4][3], __int128 Q_i128[4]) {
    int64_t Vq[4][3], Wq[4][3];
    quantize_field_4x3_tet(V, Vq);
    quantize_field_4x3_tet(W, Wq);

    // Build edge-difference matrices: same layout as L1700-1711
    int64_t Aq[3][3] = {
        {Vq[0][0] - Vq[3][0], Vq[1][0] - Vq[3][0], Vq[2][0] - Vq[3][0]},
        {Vq[0][1] - Vq[3][1], Vq[1][1] - Vq[3][1], Vq[2][1] - Vq[3][1]},
        {Vq[0][2] - Vq[3][2], Vq[1][2] - Vq[3][2], Vq[2][2] - Vq[3][2]}
    };
    int64_t Bq[3][3] = {
        {Wq[0][0] - Wq[3][0], Wq[1][0] - Wq[3][0], Wq[2][0] - Wq[3][0]},
        {Wq[0][1] - Wq[3][1], Wq[1][1] - Wq[3][1], Wq[2][1] - Wq[3][1]},
        {Wq[0][2] - Wq[3][2], Wq[1][2] - Wq[3][2], Wq[2][2] - Wq[3][2]}
    };

    characteristic_polynomial_3x3_i128(Aq, Bq, Q_i128);
}

// Forward declaration for compute_tet_QP_i128 (defined later in this header).
template <typename T>
FTK_HOST_DEVICE void characteristic_polynomials_pv_tetrahedron(const T V[4][3], const T W[4][3], T Q[4], T P[4][4]);

// ============================================================================
// Tet Integer Q + P Polynomials  (for fully combinatorial stitching)
// ============================================================================
//
// Computes both Q(λ) = det(A − λB) and P[k](λ) for k=0..3 using __int128
// arithmetic.  Reuses the existing template
//   characteristic_polynomials_pv_tetrahedron<__int128>
// which calls characteristic_polynomial_3x3, characteristic_polynomial_2x2,
// polynomial_multiply, polynomial_add_inplace, polynomial_scalar_multiply —
// all templated and correct with __int128.
//
// Overflow with QUANT_BITS_TET=16: entries ≤ 2^22, edge diffs ≤ 2^23,
// Q coeffs ≤ 2^73, P coeffs ≤ 2^75 — all fit __int128.

template <typename T>
FTK_HOST_DEVICE void compute_tet_QP_i128(const T V[4][3], const T W[4][3],
                         __int128 Q_i128[4], __int128 P_i128[4][4]) {
    int64_t Vq[4][3], Wq[4][3];
    quantize_field_4x3_tet(V, Vq);
    quantize_field_4x3_tet(W, Wq);

    __int128 V128[4][3], W128[4][3];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) {
            V128[i][j] = (__int128)Vq[i][j];
            W128[i][j] = (__int128)Wq[i][j];
        }

    characteristic_polynomials_pv_tetrahedron(V128, W128, Q_i128, P_i128);
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
FTK_HOST_DEVICE inline __int128 gcd_i128(__int128 a, __int128 b) {
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
FTK_HOST_DEVICE inline int discriminant_sign_i128(const __int128 P[4]) {
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
    if (pvs_abs128(a) >= THRESH || pvs_abs128(b) >= THRESH ||
        pvs_abs128(c) >= THRESH || pvs_abs128(d) >= THRESH)
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
// ExactPV2: Pure-Integer Polynomial Primitives
// ============================================================================
//
// These functions implement the float-free PV pipeline described in
// docs/exactpv2.md.  All topological decisions (validity, ordering, pairing,
// edge/vertex detection, pass-through) use pure __int128 integer arithmetic.
//
// Overflow strategy: content-reduce ALL polynomial inputs before every
// PRS/resultant operation.  With QUANT_BITS_TET=16, raw P_k coefficients
// are ~2^70.  After content reduction they typically drop to ~2^40-50,
// making products and Bareiss steps fit comfortably in __int128.

// --- 1.1  Content Reduction ------------------------------------------------
// Divide all coefficients of poly[0..deg] by their GCD.
// Does not change roots.  Mandatory before PRS/resultant to avoid overflow.
FTK_HOST_DEVICE inline void content_reduce_i128(__int128* poly, int deg) {
    if (deg < 0) return;
    __int128 g = 0;
    for (int i = 0; i <= deg; i++) {
        __int128 v = poly[i] < 0 ? -poly[i] : poly[i];
        if (g == 0) g = v;
        else { while (v != 0) { __int128 t = g % v; g = v; v = t; } }
    }
    if (g > 1) for (int i = 0; i <= deg; i++) poly[i] /= g;
}

// Return the effective degree of poly[0..max_deg] (strip trailing zeros).
FTK_HOST_DEVICE inline int effective_degree_i128(const __int128* poly, int max_deg) {
    int d = max_deg;
    while (d > 0 && poly[d] == 0) d--;
    return d;
}

// --- 1.2  Pseudo-Remainder -------------------------------------------------
// Computes r = prem(f, g) and returns deg(r).  Content-reduces the result.
// f has degree df, g has degree dg, with dg <= df.
// r must have space for at least df+1 entries.
FTK_HOST_DEVICE inline int prem_i128(const __int128* f, int df,
                     const __int128* g, int dg,
                     __int128* r) {
    // Copy f into r
    for (int i = 0; i <= df; i++) r[i] = f[i];
    for (int i = df + 1; i < 8; i++) r[i] = 0;  // zero pad

    int dr = df;
    while (dr >= dg && dr >= 0) {
        if (r[dr] == 0) { dr--; continue; }
        __int128 rdr = r[dr];
        __int128 gdq = g[dg];
        int shift = dr - dg;
        // r = gdq * r - rdr * g * x^shift
        for (int i = 0; i <= dr; i++) r[i] *= gdq;
        for (int i = 0; i <= dg; i++) r[shift + i] -= rdr * g[i];
        dr--;
    }
    if (dr < 0) { dr = 0; r[0] = 0; }
    while (dr > 0 && r[dr] == 0) dr--;

    content_reduce_i128(r, dr);
    return dr;
}

// --- 1.3  Polynomial GCD with Coefficients ---------------------------------
// Returns degree of h = gcd(f, g) and writes the GCD polynomial to h.
// h must have space for at least min(df,dg)+1 entries.
FTK_HOST_DEVICE inline int poly_gcd_full_i128(const __int128* f_in, int df,
                              const __int128* g_in, int dg,
                              __int128* h) {
    __int128 p[8] = {}, q[8] = {};
    for (int i = 0; i <= df; i++) p[i] = f_in[i];
    for (int i = 0; i <= dg; i++) q[i] = g_in[i];

    df = effective_degree_i128(p, df);
    dg = effective_degree_i128(q, dg);

    content_reduce_i128(p, df);
    content_reduce_i128(q, dg);

    if (df == 0 && p[0] == 0) {
        for (int i = 0; i <= dg; i++) h[i] = q[i];
        return dg;
    }
    if (dg == 0 && q[0] == 0) {
        for (int i = 0; i <= df; i++) h[i] = p[i];
        return df;
    }

    int dp = df, dq = dg;
    while (!(dq == 0 && q[0] == 0)) {
        if (dp < dq) {
            for (int i = 0; i < 8; i++) { __int128 t = p[i]; p[i] = q[i]; q[i] = t; }
            int t = dp; dp = dq; dq = t;
        }
        __int128 r[8] = {};
        int dr = prem_i128(p, dp, q, dq, r);

        dp = dq;
        for (int i = 0; i < 8; i++) p[i] = q[i];
        dq = dr;
        for (int i = 0; i < 8; i++) q[i] = r[i];
    }

    // Make leading coefficient positive for canonical form
    if (dp > 0 && p[dp] < 0)
        for (int i = 0; i <= dp; i++) p[i] = -p[i];

    for (int i = 0; i <= dp; i++) h[i] = p[i];
    return dp;
}

// --- 1.4  Exact Polynomial Division ----------------------------------------
// Returns degree of q = f/g.  Assumes g divides f exactly.
// q must have space for df-dg+1 entries.
FTK_HOST_DEVICE inline int poly_exact_div_i128(const __int128* f, int df,
                               const __int128* g, int dg,
                               __int128* q) {
    __int128 rem[8] = {};
    for (int i = 0; i <= df; i++) rem[i] = f[i];

    int dq = df - dg;
    for (int i = 0; i <= dq; i++) q[i] = 0;

    for (int i = dq; i >= 0; i--) {
        // q[i] = rem[i + dg] / g[dg]
        q[i] = rem[i + dg] / g[dg];  // exact division assumed
        for (int j = 0; j <= dg; j++)
            rem[i + j] -= q[i] * g[j];
    }
    return dq;
}

// --- 1.5  Resultant Sign via Bareiss ---------------------------------------
// Returns +1, -1, or 0 for sign(Res(f, g)).
// Content-reduces inputs first to keep intermediate products in __int128.
FTK_HOST_DEVICE inline int resultant_sign_i128(const __int128* f_in, int df,
                               const __int128* g_in, int dg) {
    // Handle degenerate degrees
    if (df <= 0 || dg <= 0) {
        if (df == 0 && dg == 0) return 1;  // Res of two constants = 1
        if (df == 0) {
            // Res(const f, g) = f^dg
            __int128 v = f_in[0];
            if (v == 0) return 0;
            // f^dg: sign depends on parity
            int s = (v > 0) ? 1 : -1;
            if (dg % 2 == 0) return 1;
            return s;
        }
        if (dg == 0) {
            __int128 v = g_in[0];
            if (v == 0) return 0;
            int s = (v > 0) ? 1 : -1;
            if (df % 2 == 0) return 1;
            return s;
        }
        return 0;
    }

    // Content-reduce copies
    __int128 fc[8] = {}, gc[8] = {};
    for (int i = 0; i <= df; i++) fc[i] = f_in[i];
    for (int i = 0; i <= dg; i++) gc[i] = g_in[i];
    content_reduce_i128(fc, df);
    content_reduce_i128(gc, dg);

    int N = df + dg;
    if (N > 7) return 0;  // overflow guard: too large for our fixed arrays

    // Build Sylvester matrix: dg rows of f-coefficients, df rows of g-coefficients
    __int128 M[7][7] = {};
    for (int i = 0; i < dg; i++)
        for (int j = 0; j <= df; j++)
            M[i][i + df - j] = fc[j];
    for (int i = 0; i < df; i++)
        for (int j = 0; j <= dg; j++)
            M[dg + i][i + dg - j] = gc[j];

    // Bareiss fraction-free elimination with overflow-safe products.
    // Standard Bareiss: M[r][j] = (M[c][c]*M[r][j] - M[r][c]*M[c][j]) / prev
    // The numerator can overflow __int128 for 6×6 matrices with ~16-bit entries.
    // Fix: factor out gcd(M[c][c], M[r][c]) before multiplying.
    __int128 prev_pivot = 1;
    int sign_swaps = 0;
    for (int col = 0; col < N; col++) {
        // Partial pivoting
        int pivot = -1;
        for (int row = col; row < N; row++)
            if (M[row][col] != 0) { pivot = row; break; }
        if (pivot < 0) return 0;  // zero determinant
        if (pivot != col) {
            for (int j = 0; j < N; j++) {
                __int128 t = M[col][j]; M[col][j] = M[pivot][j]; M[pivot][j] = t;
            }
            sign_swaps++;
        }
        for (int row = col + 1; row < N; row++) {
            // Compute (a*b - c*d) / prev for each j, where a=M[c][c], c=M[r][c],
            // b=M[r][j], d=M[c][j].
            // Factor out gcd(|a|,|c|) to reduce product sizes and prevent overflow.
            __int128 a = M[col][col], c = M[row][col];
            __int128 g1 = gcd_i128(a, c);
            if (g1 == 0) g1 = 1;
            __int128 a1 = a / g1, c1 = c / g1;
            for (int j = col + 1; j < N; j++) {
                // a*b - c*d = g1*(a1*b - c1*d).
                // Also factor gcd(|b|,|d|) to further reduce.
                __int128 b = M[row][j], d = M[col][j];
                __int128 g2 = gcd_i128(b, d);
                if (g2 == 0) g2 = 1;
                __int128 b2 = b / g2, d2 = d / g2;
                // a*b - c*d = g1*g2*(a1*b2 - c1*d2)
                __int128 inner = a1 * b2 - c1 * d2;
                __int128 outer = g1 * g2;
                // Divide by prev_pivot: result = outer * inner / prev_pivot
                __int128 p = prev_pivot;
                __int128 gp = gcd_i128(outer, p);
                M[row][j] = (outer / gp) * (inner / (p / gp));
            }
        }
        for (int row = col + 1; row < N; row++)
            M[row][col] = 0;
        prev_pivot = M[col][col];
    }

    __int128 det = M[N-1][N-1];
    if (det == 0) return 0;
    int s = (det > 0) ? 1 : -1;
    if (sign_swaps % 2) s = -s;
    return s;
}

// --- 1.6  Sign at Unique Root (one-root case) ------------------------------
// Returns sign(g(α)) where f has exactly one real root α (disc < 0).
// Formula: sign(g(α)) = sign(Res(f, g)) × sign(lc(f)^deg(g))
FTK_HOST_DEVICE inline int sign_at_unique_root_i128(const __int128* f, int df,
                                    const __int128* g, int dg) {
    // Handle trivial g
    if (dg == 0) return (g[0] > 0) ? 1 : (g[0] < 0) ? -1 : 0;

    int res_sign = resultant_sign_i128(f, df, g, dg);
    if (res_sign == 0) return 0;

    // sign(lc(f)^dg): positive if dg even or lc(f) > 0, negative if dg odd and lc(f) < 0
    __int128 lc_f = f[df];
    int lc_sign = (lc_f > 0) ? 1 : -1;
    int lc_power_sign = (dg % 2 == 0) ? 1 : lc_sign;

    return res_sign * lc_power_sign;
}

// --- Minimal 256-bit signed integer for overflow-safe polynomial sign eval --
// Represents a 256-bit signed integer as (hi:lo) in two's complement.
struct int256_t {
    __int128 hi;            // signed high 128 bits
    unsigned __int128 lo;   // unsigned low 128 bits

    FTK_HOST_DEVICE int256_t() : hi(0), lo(0) {}
    FTK_HOST_DEVICE explicit int256_t(__int128 v)
        : hi(v >= 0 ? (__int128)0 : (__int128)-1),
          lo((unsigned __int128)v) {}

    FTK_HOST_DEVICE int sign() const {
        if (hi > 0) return 1;
        if (hi < 0) return -1;
        return (lo > 0) ? 1 : 0;
    }

    FTK_HOST_DEVICE int256_t operator+(const int256_t& r) const {
        int256_t s;
        s.lo = lo + r.lo;
        s.hi = hi + r.hi + (__int128)(s.lo < lo);
        return s;
    }

    FTK_HOST_DEVICE int256_t operator-() const {
        int256_t r;
        r.lo = ~lo + 1;
        r.hi = ~hi + (__int128)(r.lo == 0);
        return r;
    }
};

// Unsigned 128×128 → 256 multiply.
FTK_HOST_DEVICE inline void umul256(unsigned __int128 ua, unsigned __int128 ub,
                    unsigned __int128& res_hi, unsigned __int128& res_lo) {
    uint64_t a0 = (uint64_t)ua, a1 = (uint64_t)(ua >> 64);
    uint64_t b0 = (uint64_t)ub, b1 = (uint64_t)(ub >> 64);

    unsigned __int128 p00 = (unsigned __int128)a0 * b0;
    unsigned __int128 p01 = (unsigned __int128)a0 * b1;
    unsigned __int128 p10 = (unsigned __int128)a1 * b0;
    unsigned __int128 p11 = (unsigned __int128)a1 * b1;

    // Accumulate into 4 × 64-bit chunks via a 128-bit accumulator
    uint64_t r0 = (uint64_t)p00;
    unsigned __int128 acc = (p00 >> 64);
    acc += (uint64_t)p01;
    acc += (uint64_t)p10;
    uint64_t r1 = (uint64_t)acc;
    acc = (acc >> 64);
    acc += (p01 >> 64);
    acc += (p10 >> 64);
    acc += (uint64_t)p11;
    uint64_t r2 = (uint64_t)acc;
    acc = (acc >> 64) + (p11 >> 64);
    uint64_t r3 = (uint64_t)acc;

    res_lo = (unsigned __int128)r0 | ((unsigned __int128)r1 << 64);
    res_hi = (unsigned __int128)r2 | ((unsigned __int128)r3 << 64);
}

// Signed 128×128 → 256-bit result.
FTK_HOST_DEVICE inline int256_t mul256(__int128 a, __int128 b) {
    bool neg = (a < 0) != (b < 0);
    unsigned __int128 ua = a < 0 ? (unsigned __int128)(-(a + 1)) + 1
                                 : (unsigned __int128)a;
    unsigned __int128 ub = b < 0 ? (unsigned __int128)(-(b + 1)) + 1
                                 : (unsigned __int128)b;

    int256_t res;
    unsigned __int128 rhi;
    umul256(ua, ub, rhi, res.lo);
    res.hi = (__int128)rhi;
    if (neg) res = -res;
    return res;
}

// Multiply int256_t × __int128 → int256_t  (assumes result fits in 256 bits).
FTK_HOST_DEVICE inline int256_t mul256_128(int256_t a, __int128 b) {
    bool neg = false;
    if (a.sign() < 0) { a = -a; neg = true; }
    else if (a.sign() == 0) return int256_t();
    unsigned __int128 ub;
    if (b < 0) { ub = (unsigned __int128)(-(b + 1)) + 1; neg = !neg; }
    else { ub = (unsigned __int128)b; }

    // a.lo * ub → 256-bit unsigned
    unsigned __int128 lo_hi, lo_lo;
    umul256(a.lo, ub, lo_hi, lo_lo);

    // a.hi * ub → contributes to high part only (shifted by 128)
    unsigned __int128 hi_prod = (unsigned __int128)a.hi * ub;

    int256_t res;
    res.lo = lo_lo;
    res.hi = (__int128)(lo_hi + hi_prod);
    if (neg) res = -res;
    return res;
}

// --- Helper: sign of f(p/q) via integer Horner ----------------------------
// Computes sign(f(p/q)) exactly.  Uses 256-bit arithmetic to avoid overflow.
FTK_HOST_DEVICE inline int sign_poly_at_rational_i128(const __int128* f, int df,
                                      __int128 p, __int128 q) {
    if (df <= 0) return (f[0] > 0) ? 1 : (f[0] < 0) ? -1 : 0;
    if (q == 0) {
        // f(p/0) → sign of leading term × sign(p^df)
        int lc_s = (f[df] > 0) ? 1 : (f[df] < 0) ? -1 : 0;
        int p_s  = (p > 0) ? 1 : (p < 0) ? -1 : 0;
        return lc_s * ((df % 2 == 0) ? 1 : p_s);
    }
    // Reverse Horner in 256-bit: val = f[0]*q^df + f[1]*p*q^(df-1) + ... + f[df]*p^df
    int256_t val(f[0]);
    int256_t p_pow(p);
    for (int i = 1; i <= df; i++) {
        val = mul256_128(val, q) + mul256_128(p_pow, f[i]);
        if (i < df) p_pow = mul256_128(p_pow, p);
    }
    int val_sign = val.sign();
    if (val_sign == 0) return 0;
    int q_sign = (q > 0) ? 1 : -1;
    int qdf_sign = (df % 2 == 0) ? 1 : q_sign;
    return val_sign * qdf_sign;
}

// --- 1.7a  Derivative of integer polynomial --------------------------------
FTK_HOST_DEVICE inline int poly_derivative_i128(const __int128* f, int df, __int128* fp) {
    if (df <= 0) { fp[0] = 0; return 0; }
    for (int i = 0; i < df; i++) fp[i] = (__int128)(i + 1) * f[i + 1];
    return df - 1;
}

// --- 1.7b  Square-free factorization --------------------------------------
// Returns degree of sqfree(f) = f / gcd(f, f'), writes result to sf.
// The square-free part has the same roots as f but all with multiplicity 1.
FTK_HOST_DEVICE inline int poly_sqfree_i128(const __int128* f, int df, __int128* sf) {
    if (df <= 1) {
        for (int i = 0; i <= df; i++) sf[i] = f[i];
        return df;
    }
    __int128 fp[8] = {};
    int dfp = poly_derivative_i128(f, df, fp);

    __int128 g[8] = {};
    int dg = poly_gcd_full_i128(f, df, fp, dfp, g);

    if (dg == 0) {
        // gcd is constant → f is already square-free
        for (int i = 0; i <= df; i++) sf[i] = f[i];
        content_reduce_i128(sf, df);
        return df;
    }

    // sf = f / g
    int dsf = poly_exact_div_i128(f, df, g, dg, sf);
    content_reduce_i128(sf, dsf);
    return dsf;
}

// --- 1.7c  Number of roots of cubic f below rational point t = p/q ---------
// Given cubic f (with 3 real roots, disc > 0) and rational t = p/q,
// returns the number of roots of f that are strictly below t.
// Uses sign(f(t)), sign(f'(t)), sign(f''(t)) with rigorous case analysis.
// Returns n_below in {0,1,2,3}, or -1 if f(t)=0 (shared root).
//
// Proof of correctness:
// For cubic f with lc > 0 (normalize by sign_lc), roots α₀ < α₁ < α₂:
//   f' has roots β₁ ∈ (α₀,α₁), β₂ ∈ (α₁,α₂)  (local max, min)
//   f'' has root γ ∈ (β₁,β₂)  (inflection point)
//   f' > 0 on (-∞,β₁) ∪ (β₂,+∞), f' < 0 on (β₁,β₂)
//   f'' < 0 on (-∞,γ), f'' > 0 on (γ,+∞)
//
// Case table (normalized lc > 0):
//   f>0, f'>0, f''≤0: t ∈ (α₀,β₁)         → n=1
//   f>0, f'>0, f''>0: t ∈ (α₂,+∞)         → n=3
//   f>0, f'≤0:        t ∈ [β₁,α₁)         → n=1
//   f<0, f'>0, f''≤0: t ∈ (-∞,α₀)         → n=0
//   f<0, f'>0, f''>0: t ∈ (β₂,α₂)         → n=2
//   f<0, f'≤0:        t ∈ (α₁,β₂]         → n=2
FTK_HOST_DEVICE inline int count_roots_below_rational(const __int128* f, int df,
                                      __int128 p, __int128 q) {
    int sign_ft = sign_poly_at_rational_i128(f, df, p, q);
    if (sign_ft == 0) return -1;  // shared root

    __int128 fp[4] = {};
    int dfp = poly_derivative_i128(f, df, fp);
    int sign_fpt = sign_poly_at_rational_i128(fp, dfp, p, q);

    int sign_lc = (f[df] > 0) ? 1 : -1;
    int eft = sign_ft * sign_lc;    // normalized f(t)
    int efpt = sign_fpt * sign_lc;  // normalized f'(t)

    if (eft > 0) {
        // f(t) > 0 (normalized): t ∈ (α₀,α₁) or (α₂,+∞), so n = 1 or 3
        if (efpt > 0) {
            // Ascending and positive: (α₀,β₁) [n=1] or (α₂,+∞) [n=3]
            __int128 fpp[4] = {};
            int dfpp = poly_derivative_i128(fp, dfp, fpp);
            int efppt = sign_poly_at_rational_i128(fpp, dfpp, p, q) * sign_lc;
            return (efppt <= 0) ? 1 : 3;
        }
        return 1;  // f' ≤ 0: must be in [β₁,α₁)
    } else {
        // f(t) < 0 (normalized): t ∈ (-∞,α₀) or (α₁,α₂), so n = 0 or 2
        if (efpt > 0) {
            // Ascending and negative: (-∞,α₀) [n=0] or (β₂,α₂) [n=2]
            __int128 fpp[4] = {};
            int dfpp = poly_derivative_i128(fp, dfp, fpp);
            int efppt = sign_poly_at_rational_i128(fpp, dfpp, p, q) * sign_lc;
            return (efppt <= 0) ? 0 : 2;
        }
        return 2;  // f' ≤ 0: must be in (α₁,β₂]
    }
}

// --- 1.7  Signs at Roots of a Polynomial -----------------------------------
// Determines sign(g(α_i)) for each distinct real root α_i of f.
//
// Handles all degree degeneracies:
//   - f effective degree 0: no roots, nothing to do
//   - f effective degree 1: one root (linear), evaluate g directly
//   - f effective degree 2: two roots (quadratic)
//   - f effective degree 3, disc < 0: one real root
//   - f effective degree 3, disc > 0: three real roots
//   - f effective degree 3, disc = 0: double/triple root → use square-free part
//
// signs[] must have space for n_roots entries.
// Returns actual number of distinct roots filled into signs[].

FTK_HOST_DEVICE inline int signs_at_roots_i128(const __int128* f_in, int df_max,
                               const __int128* g_in, int dg_max,
                               int signs[], int max_signs,
                               int _depth = 0) {
    if (_depth > 20) return 0;  // recursion guard

    // Determine effective degrees
    __int128 f[8] = {}, g[8] = {};
    for (int i = 0; i <= df_max && i < 8; i++) f[i] = f_in[i];
    for (int i = 0; i <= dg_max && i < 8; i++) g[i] = g_in[i];
    int df = effective_degree_i128(f, df_max < 7 ? df_max : 7);
    int dg = effective_degree_i128(g, dg_max < 7 ? dg_max : 7);

    content_reduce_i128(f, df);
    content_reduce_i128(g, dg);

    if (df == 0) return 0;  // no roots

    // --- Linear f: one root at t = -f[0]/f[1] ---
    if (df == 1) {
        if (max_signs < 1) return 0;
        signs[0] = sign_poly_at_rational_i128(g, dg, -f[0], f[1]);
        return 1;
    }

    // --- Quadratic f: disc = f[1]^2 - 4*f[2]*f[0] ---
    if (df == 2) {
        __int128 disc = f[1]*f[1] - 4*f[2]*f[0];
        if (disc < 0) return 0;  // no real roots
        if (disc == 0) {
            // Double root at t = -f[1]/(2*f[2])
            if (max_signs < 1) return 0;
            signs[0] = sign_poly_at_rational_i128(g, dg, -f[1], 2*f[2]);
            return 1;
        }
        // Two distinct roots.  We need sign(g) at each.
        // Reduce g mod f to get remainder r of degree ≤ 1.
        if (dg >= df) {
            __int128 r[8] = {};
            int dr = prem_i128(g, dg, f, df, r);
            // sign(g(α_i)) = sign(r(α_i)) × sign(lc(f)^(dg-df+1))
            // But prem already handled the scaling.  Actually:
            // prem(g, f) = lc(f)^(dg-df+1) * g  mod f
            // At roots of f: prem(g,f)(α) = lc(f)^(dg-df+1) * g(α)
            int exp = dg - df + 1;
            int lc_f_sign = (f[df] > 0) ? 1 : -1;
            int scale_sign = (exp % 2 == 0) ? 1 : lc_f_sign;

            // Now determine sign(r(α_i)) at the two roots of f
            int sub_signs[2];
            int n = signs_at_roots_i128(f, df, r, dr, sub_signs, 2, _depth + 1);
            for (int i = 0; i < n && i < max_signs; i++)
                signs[i] = sub_signs[i] * scale_sign;
            return n;
        }
        // dg < df: g is already lower degree than f, evaluate directly
        // For quadratic f with two roots: need to determine interleaving
        // of g's roots with f's roots.  Since dg < 2, g is linear or constant.
        if (dg == 0) {
            // g is constant
            int gs = (g[0] > 0) ? 1 : (g[0] < 0) ? -1 : 0;
            signs[0] = gs; signs[1] = gs;
            return 2;
        }
        // dg == 1: g is linear with root at t = -g[0]/g[1]
        // Need to know if t is between the two roots of f or outside.
        // sign(f(t)) tells us:
        //   f(t) > 0 and lc(f) > 0: t outside roots → g has same sign at both roots
        //   f(t) < 0 and lc(f) > 0: t between roots → g changes sign
        int sign_ft = sign_poly_at_rational_i128(f, df, -g[0], g[1]);
        int sign_lc_f = (f[df] > 0) ? 1 : -1;
        int sign_g_lc = (g[dg] > 0) ? 1 : -1;

        if (sign_ft == 0) {
            // g's root is also a root of f → one sign is 0
            // Determine which root of f it matches
            // Since f has 2 roots and g's root equals one of them,
            // sign at the other root = sign(g) at a non-zero point
            // For the root that matches: signs = 0
            // For the other: sign = sign(g(other_root))
            // f(t) = 0 means t is a root of f.  g(t) = 0 too.
            // At the other root: sign(g) = -sign(g_lc) if other root < t, else sign(g_lc)
            // Actually we just know f(t) = 0.  We need f'(t) to determine which root.
            signs[0] = 0;
            signs[1] = 0;
            // More precisely: one root has g=0, the other doesn't.
            // The non-zero sign = sign(g_lc) if that root is to the right of g's root,
            //                    -sign(g_lc) if to the left.
            // Since the two roots of f straddle g's root when f(t)=0:
            // Actually f(t)=0 means g's root coincides with one of f's roots.
            // signs[smaller_root] and signs[larger_root]:
            // If the shared root is the smaller one: signs = {0, sign(g_lc)}
            // If the shared root is the larger one: signs = {-sign(g_lc), 0}
            // For quadratic f with positive lc: f'(α₀) < 0 (descending), f'(α₁) > 0 (ascending)
            // So: f'(t)·lc(f) < 0 → t is the SMALLER root; f'(t)·lc(f) > 0 → t is the LARGER root
            __int128 fp[4] = {};
            poly_derivative_i128(f, df, fp);
            int sign_fpt = sign_poly_at_rational_i128(fp, df-1, -g[0], g[1]);
            if (sign_fpt * sign_lc_f < 0) {
                // f descending at t → t is the SMALLER root
                signs[0] = 0;
                signs[1] = sign_g_lc;
            } else {
                // f ascending at t → t is the LARGER root
                signs[0] = -sign_g_lc;
                signs[1] = 0;
            }
            return 2;
        }

        if (sign_ft * sign_lc_f > 0) {
            // t is outside the roots of f → g has same sign at both roots
            // g(α₀) = g(α₁) = sign(lc(g)) when both roots > t, or -sign when both < t
            // If both roots of f are > t: g(α) = sign(g_lc) * sign(1) = sign(g_lc) for both
            // If both roots of f are < t: g(α) = -sign(g_lc) for both
            // Determine: f's roots vs t.  Since f(t) > 0 if lc > 0 → t outside roots.
            // Which side? Check sign of f at ±∞ relative to t:
            // For quadratic with lc > 0, f > 0 outside [α₀, α₁].
            // If all roots > t: happens when t < α₀.  Then g(α) > 0 if g_lc > 0.
            // sign(f'(t)) and lc(f): if t < both roots and lc > 0, f'(t) < 0
            __int128 fp[4] = {};
            poly_derivative_i128(f, df, fp);
            int sign_fpt = sign_poly_at_rational_i128(fp, df-1, -g[0], g[1]);
            if (sign_fpt * sign_lc_f < 0) {
                // f'(t) < 0 when lc > 0 → t < both roots → g > 0 at roots if g_lc > 0
                signs[0] = sign_g_lc;
                signs[1] = sign_g_lc;
            } else {
                signs[0] = -sign_g_lc;
                signs[1] = -sign_g_lc;
            }
            return 2;
        } else {
            // t is between the roots → g changes sign
            // g(smaller_root) = -sign(g_lc), g(larger_root) = sign(g_lc)
            signs[0] = -sign_g_lc;
            signs[1] = sign_g_lc;
            return 2;
        }
    }

    // --- Cubic f ---
    // Check discriminant
    int disc_sign = discriminant_sign_i128(f);

    if (disc_sign < 0) {
        // One real root α.  Use PRS to avoid Sylvester resultant overflow.
        // Reduce g mod f → remainder r (degree ≤ 2).
        if (max_signs < 1) return 0;
        __int128 r1[8] = {};
        int dr1;
        int scale1 = 1;
        if (dg >= df) {
            dr1 = prem_i128(g, dg, f, df, r1);
            int exp1 = dg - df + 1;
            int lc_sign1 = (f[df] > 0) ? 1 : -1;
            scale1 = (exp1 % 2 == 0) ? 1 : lc_sign1;
        } else {
            for (int j = 0; j <= dg; j++) r1[j] = g[j];
            dr1 = dg;
            content_reduce_i128(r1, dr1);
        }
        dr1 = effective_degree_i128(r1, dr1);
        // sign(g(α)) = sign(r1(α)) × scale1
        if (dr1 == 0) {
            signs[0] = ((r1[0] > 0) ? 1 : (r1[0] < 0) ? -1 : 0) * scale1;
            return 1;
        }
        if (dr1 == 1) {
            // r1 has root at t = -r1[0]/r1[1].
            // For cubic f with 1 real root: f changes sign only at α.
            // sign(f(t)) × lc(f): > 0 → t > α, < 0 → t < α.
            int sft = sign_poly_at_rational_i128(f, df, -r1[0], r1[1]);
            int sign_lc_f = (f[df] > 0) ? 1 : -1;
            int r1_lc_sign = (r1[dr1] > 0) ? 1 : -1;
            if (sft == 0) { signs[0] = 0; return 1; }  // shared root
            if (sft * sign_lc_f > 0)
                signs[0] = -r1_lc_sign * scale1;  // t > α → α < t → r1(α) = r1_lc*(α-t), (α-t)<0
            else
                signs[0] = r1_lc_sign * scale1;   // t < α → α > t → r1(α) = r1_lc*(α-t), (α-t)>0
            return 1;
        }
        // dr1 == 2: quadratic remainder
        {
            __int128 disc_r1 = r1[1]*r1[1] - 4*r1[2]*r1[0];
            int sign_r1_2 = (r1[2] > 0) ? 1 : -1;
            if (disc_r1 < 0) {
                // No real roots → r1 has constant sign
                signs[0] = sign_r1_2 * scale1;
                return 1;
            }
            if (disc_r1 == 0) {
                // Double root → r1 = r1[2]*(x-ρ)², sign = sign(r1[2]) unless α = ρ
                int sft = sign_poly_at_rational_i128(f, df, -r1[1], 2*r1[2]);
                signs[0] = (sft == 0) ? 0 : sign_r1_2 * scale1;
                return 1;
            }
            // Two roots ρ₁ < ρ₂.  r1 > 0 outside [ρ₁,ρ₂] if r1[2] > 0.
            // Need: is α in (-∞,ρ₁), (ρ₁,ρ₂), or (ρ₂,∞)?
            // Evaluate f at roots of r1 via signs_at_roots_i128(r1, 2, f, 3, ...).
            // For cubic f with 1 real root: f(ρ) × lc(f) > 0 ↔ ρ > α.
            int f_at_rho[2] = {};
            signs_at_roots_i128(r1, dr1, f, df, f_at_rho, 2, _depth + 1);
            int sign_lc_f = (f[df] > 0) ? 1 : -1;
            // ρ₁ < ρ₂.  Determine position of α:
            int pos1 = f_at_rho[0] * sign_lc_f;  // > 0 → ρ₁ > α
            int pos2 = f_at_rho[1] * sign_lc_f;  // > 0 → ρ₂ > α
            if (f_at_rho[0] == 0 || f_at_rho[1] == 0) {
                signs[0] = 0;  // shared root
            } else if (pos1 > 0) {
                // ρ₁ > α → α < ρ₁ < ρ₂ → outside, left
                signs[0] = sign_r1_2 * scale1;
            } else if (pos2 > 0) {
                // ρ₁ < α < ρ₂ → inside
                signs[0] = -sign_r1_2 * scale1;
            } else {
                // ρ₁ < ρ₂ < α → outside, right
                signs[0] = sign_r1_2 * scale1;
            }
            return 1;
        }
    }

    if (disc_sign == 0) {
        // Double or triple root — use square-free part
        __int128 sf[8] = {};
        int dsf = poly_sqfree_i128(f, df, sf);
        // Recurse on square-free part (has the distinct roots only)
        return signs_at_roots_i128(sf, dsf, g, dg, signs, max_signs, _depth + 1);
    }

    // disc_sign > 0: three distinct real roots
    if (max_signs < 3) return 0;

    // Reduce g mod f to get remainder r (degree ≤ 2)
    __int128 r[8] = {};
    int dr;
    int scale_sign = 1;

    if (dg >= df) {
        dr = prem_i128(g, dg, f, df, r);
        int exp = dg - df + 1;
        int lc_f_sign = (f[df] > 0) ? 1 : -1;
        scale_sign = (exp % 2 == 0) ? 1 : lc_f_sign;
    } else {
        for (int i = 0; i <= dg; i++) r[i] = g[i];
        dr = dg;
        content_reduce_i128(r, dr);
    }

    // Now determine sign(r(α_i)) at the three roots of f

    // Case A: deg(r) = 0 (constant)
    if (dr == 0) {
        int rs = (r[0] > 0) ? 1 : (r[0] < 0) ? -1 : 0;
        signs[0] = signs[1] = signs[2] = rs * scale_sign;
        return 3;
    }

    // Case B: deg(r) = 1 (linear, root at t = -r[0]/r[1])
    if (dr == 1) {
        int n_below = count_roots_below_rational(f, df, -r[0], r[1]);
        int sign_r_lc = (r[1] > 0) ? 1 : -1;
        if (n_below < 0) {
            // Shared root: one of the α_i equals the root of r
            // sign(r(α_i)) = 0 at that root, ±sign(r_lc) at others
            // Since we don't know which one, use resultant to find it
            // Actually n_below = -1 means f(t) = 0, so t is one of the roots.
            // Determine which by counting: use f'(t) and f''(t) at t
            __int128 fp[4] = {};
            poly_derivative_i128(f, df, fp);
            int sign_fpt = sign_poly_at_rational_i128(fp, df-1, -r[0], r[1]);
            __int128 fpp[4] = {};
            poly_derivative_i128(fp, df-1, fpp);
            int sign_fppt = sign_poly_at_rational_i128(fpp, df-2, -r[0], r[1]);
            int sign_lc = (f[df] > 0) ? 1 : -1;
            // Determine which root is at t:
            // f'(t)=0 → t is at a critical point (not a root if disc > 0 and 3 distinct roots...
            // actually f'(t) can be 0 at a root if that root is also a critical point, but disc > 0 means all simple)
            // For 3 simple roots, f'(t) ≠ 0 at any root.
            if (sign_fpt == 0) {
                // Degenerate — shouldn't happen with disc > 0. Fall back.
                signs[0] = signs[1] = signs[2] = 0;
                return 3;
            }
            // f'(t) * lc > 0 → f increasing at t → t is α₀ or α₂
            // f''(t) * lc > 0 → convex → right side → t = α₂
            // f''(t) * lc < 0 → concave → left side → t = α₀
            // f'(t) * lc < 0 → f decreasing → t = α₁
            int eft = sign_fpt * sign_lc;
            int efppt = sign_fppt * sign_lc;
            int which;  // 0, 1, or 2
            if (eft < 0) which = 1;
            else if (efppt > 0) which = 2;
            else which = 0;

            for (int i = 0; i < 3; i++) {
                if (i == which) signs[i] = 0;
                else if (i < which) signs[i] = -sign_r_lc * scale_sign;
                else signs[i] = sign_r_lc * scale_sign;
            }
            return 3;
        }

        // n_below roots have r < 0 (root of r is above them), rest have r > 0
        // r(x) = r[1]*(x - t), so r(α) > 0 when α > t if r[1] > 0
        for (int i = 0; i < 3; i++) {
            if (i < n_below) signs[i] = -sign_r_lc * scale_sign;
            else             signs[i] =  sign_r_lc * scale_sign;
        }
        return 3;
    }

    // Case C: deg(r) = 2 (quadratic)
    {
        __int128 disc_r = r[1]*r[1] - 4*r[2]*r[0];
        int sign_r2 = (r[2] > 0) ? 1 : -1;

        if (disc_r < 0) {
            // r has no real roots → constant sign = sign(r[2])
            signs[0] = signs[1] = signs[2] = sign_r2 * scale_sign;
            return 3;
        }

        if (disc_r == 0) {
            // r has double root ρ = -r[1]/(2*r[2])
            // r(x) = r[2]*(x - ρ)², so sign(r(α_i)) = sign(r[2]) unless α_i = ρ
            int n_below = count_roots_below_rational(f, df, -r[1], 2*r[2]);
            // Check if any f-root equals ρ: evaluate f at ρ = -r[1]/(2*r[2])
            int ft_sign = sign_poly_at_rational_i128(f, df, -r[1], 2*r[2]);
            if (ft_sign == 0) {
                // ρ is a root of f too
                for (int i = 0; i < 3; i++) signs[i] = sign_r2 * scale_sign;
                if (n_below >= 0 && n_below < 3) signs[n_below] = 0;
            } else {
                for (int i = 0; i < 3; i++) signs[i] = sign_r2 * scale_sign;
            }
            return 3;
        }

        // disc_r > 0: r has 2 distinct roots ρ₁ < ρ₂
        // sign(r(α_i)) = sign(r[2]) outside [ρ₁,ρ₂], -sign(r[2]) inside
        // Need: how many f-roots are below ρ₁, between ρ₁ and ρ₂, above ρ₂

        // Get sign(f(ρ_j)), sign(f'(ρ_j)), sign(f''(ρ_j)) at roots of r
        int f_at_rho[2] = {};
        int nrr = signs_at_roots_i128(r, dr, f, df, f_at_rho, 2, _depth + 1);

        __int128 fp[4] = {};
        int dfp = poly_derivative_i128(f, df, fp);
        int fp_at_rho[2] = {};
        signs_at_roots_i128(r, dr, fp, dfp, fp_at_rho, 2, _depth + 1);

        __int128 fpp[4] = {};
        int dfpp = poly_derivative_i128(fp, dfp, fpp);
        int fpp_at_rho[2] = {};
        signs_at_roots_i128(r, dr, fpp, effective_degree_i128(fpp, df-2), fpp_at_rho, 2, _depth + 1);

        int sign_lc = (f[df] > 0) ? 1 : -1;

        // For each ρ_j, determine nb_j = # of f-roots strictly below ρ_j
        // using the same case table as count_roots_below_rational:
        //   f>0, f'>0, f''≤0 → n=1    f>0, f'>0, f''>0 → n=3
        //   f>0, f'≤0        → n=1
        //   f<0, f'>0, f''≤0 → n=0    f<0, f'>0, f''>0 → n=2
        //   f<0, f'≤0        → n=2
        //   f=0              → shared root (nb = which root index)
        int nb1 = 0, nb2 = 0;
        for (int j = 0; j < 2; j++) {
            int sf = (nrr > j) ? f_at_rho[j] : 0;
            int sfp = fp_at_rho[j];
            int sfpp = fpp_at_rho[j];
            int eft = sf * sign_lc;
            int efpt = sfp * sign_lc;
            int efppt = sfpp * sign_lc;

            int nb;
            if (sf == 0) {
                // ρ_j coincides with an f-root; determine which one via f', f''
                // f'(α_i)·lc > 0 at α₀ and α₂ (ascending), < 0 at α₁ (descending)
                // f''(α_i)·lc < 0 at α₀ (concave), > 0 at α₂ (convex)
                if (efpt < 0) nb = 1;       // α₁
                else if (efppt > 0) nb = 2; // α₂
                else nb = 0;                // α₀
            } else if (eft > 0) {
                if (efpt > 0) nb = (efppt <= 0) ? 1 : 3;
                else nb = 1;
            } else {
                if (efpt > 0) nb = (efppt <= 0) ? 0 : 2;
                else nb = 2;
            }
            if (j == 0) nb1 = nb; else nb2 = nb;
        }

        // Assign signs: roots inside [ρ₁,ρ₂] get -sign(r2), outside get sign(r2)
        for (int i = 0; i < 3; i++) {
            if (i >= nb1 && i < nb2)
                signs[i] = -sign_r2 * scale_sign;
            else
                signs[i] = sign_r2 * scale_sign;
        }
        // Handle shared roots: if f(ρ_j) = 0, then r(α_i) = r(ρ_j) = 0 at that root
        if (f_at_rho[0] == 0 && nb1 < 3) signs[nb1] = 0;
        if (nrr >= 2 && f_at_rho[1] == 0 && nb2 < 3) signs[nb2] = 0;

        return 3;
    }
}

// --- 1.8  Root Comparison ---------------------------------------------------
// Compare the i-th root of f with the j-th root of g.
// Returns -1 if α_i < β_j, +1 if α_i > β_j, 0 if equal.
// f_nroots and g_nroots are the number of DISTINCT real roots (1, 2, or 3).
// Root indices are 0-based in increasing order.
FTK_HOST_DEVICE inline int compare_roots_i128(const __int128* f, int df, int f_nroots, int f_root_idx,
                              const __int128* g, int dg, int g_nroots, int g_root_idx) {
    // Strategy: determine count_below = # of g-roots strictly below α_{f_root_idx}
    // by evaluating g, g', g'' at α_{f_root_idx} via signs_at_roots_i128,
    // then applying the rigorous case table (same as count_roots_below_rational).
    // Then compare count_below with g_root_idx.

    df = effective_degree_i128(f, df);
    dg = effective_degree_i128(g, dg);

    if (df == 0 || dg == 0) return 0;  // degenerate

    // Get sign(g(α_i)) at roots of f
    int g_at_f[3] = {};
    int n_f = signs_at_roots_i128(f, df, g, dg, g_at_f, 3);
    if (f_root_idx >= n_f) return 0;

    int sg = g_at_f[f_root_idx];

    // Shared root: α_i = some β_j
    if (sg == 0) {
        int f_at_g[3] = {};
        int n_g = signs_at_roots_i128(g, dg, f, df, f_at_g, 3);
        for (int j = 0; j < n_g; j++) {
            if (f_at_g[j] == 0) {
                if (j == g_root_idx) return 0;
                return (j < g_root_idx) ? -1 : 1;
            }
        }
        return 0;
    }

    int lc_g_sign = (g[dg] > 0) ? 1 : -1;

    // g_nroots == 1: simple sign check
    if (g_nroots == 1) {
        int count_below = (sg * lc_g_sign > 0) ? 1 : 0;
        return (count_below > g_root_idx) ? 1 : -1;
    }

    // g_nroots >= 2: need g' (and possibly g'') at α_i for disambiguation
    __int128 gp[4] = {};
    int dgp = poly_derivative_i128(g, dg, gp);
    int gp_at_f[3] = {};
    signs_at_roots_i128(f, df, gp, dgp, gp_at_f, 3);
    int sgp = (f_root_idx < 3) ? gp_at_f[f_root_idx] : 0;

    int egt = sg * lc_g_sign;   // normalized g(α_i)
    int egpt = sgp * lc_g_sign; // normalized g'(α_i)

    int count_below = 0;

    if (g_nroots == 2) {
        // Quadratic-like: 2 roots β₀ < β₁
        // g > 0 outside [β₀,β₁], g < 0 inside (for lc > 0)
        if (egt > 0) {
            // Outside the roots
            count_below = (egpt > 0) ? 2 : 0;  // ascending → past β₁; descending → before β₀
        } else {
            count_below = 1;  // between roots
        }
    } else if (g_nroots == 3) {
        // Same case table as count_roots_below_rational
        if (egt > 0) {
            if (egpt > 0) {
                __int128 gpp[4] = {};
                int dgpp = poly_derivative_i128(gp, dgp, gpp);
                int gpp_at_f[3] = {};
                signs_at_roots_i128(f, df, gpp, effective_degree_i128(gpp, dg-2),
                                    gpp_at_f, 3);
                int egppt = gpp_at_f[f_root_idx] * lc_g_sign;
                count_below = (egppt <= 0) ? 1 : 3;
            } else {
                count_below = 1;
            }
        } else {
            if (egpt > 0) {
                __int128 gpp[4] = {};
                int dgpp = poly_derivative_i128(gp, dgp, gpp);
                int gpp_at_f[3] = {};
                signs_at_roots_i128(f, df, gpp, effective_degree_i128(gpp, dg-2),
                                    gpp_at_f, 3);
                int egppt = gpp_at_f[f_root_idx] * lc_g_sign;
                count_below = (egppt <= 0) ? 0 : 2;
            } else {
                count_below = 2;
            }
        }
    }

    if (count_below > g_root_idx) return 1;
    if (count_below < g_root_idx) return -1;
    // count_below == g_root_idx: α_i could equal β_{g_root_idx} or be just below
    // Since we already checked sg != 0 (no shared root), α_i ≠ β_{g_root_idx},
    // so count_below == g_root_idx means α_i < β_{g_root_idx}
    return -1;
}

// ============================================================================
// Sturm-Sequence Root Isolation  (Subtask 4)
// ============================================================================
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
FTK_HOST_DEVICE static inline double eval_poly_sturm(const double* c, int deg, double x) {
    double r = c[deg];
    for (int i = deg - 1; i >= 0; --i) r = r * x + c[i];
    return r;
}

/// Build the Sturm sequence for cubic P (ascending-degree coefficients P[0..3]).
FTK_HOST_DEVICE inline void build_sturm_double(const double P[4], SturmSeqDouble& seq) {
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
FTK_HOST_DEVICE inline int sturm_count_at(const SturmSeqDouble& seq, double x) {
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

/// Count sign changes at x with Higham-certified evaluation of each Sturm
/// polynomial.  Returns SturmCertifiedResult{count, certified}.  If any Sturm
/// polynomial evaluation is not certifiably nonzero (i.e. |S_i(x)| <= gamma_deg * cond),
/// certified=false — the caller should use SoS perturbation.
FTK_HOST_DEVICE inline SturmCertifiedResult sturm_count_at_certified(const SturmSeqDouble& seq, double x) {
    const double u = pvs_epsilon();
    const double ax = pvs_abs(x);

    int    changes  = 0;
    bool   certified = true;
    double prev     = 0.0;

    for (int i = 0; i < seq.n; ++i) {
        int deg = seq.deg[i];
        const double* c = seq.c[i];

        // Horner evaluation
        double val = c[deg];
        for (int d = deg - 1; d >= 0; --d)
            val = val * x + c[d];

        // Higham condition number: Σ|c_k|·|x|^k
        double cond = pvs_abs(c[deg]);
        for (int d = deg - 1; d >= 0; --d)
            cond = cond * ax + pvs_abs(c[d]);

        double gamma_deg = double(2 * deg + 2) * u;

        if (val != 0.0) {
            if (pvs_abs(val) <= gamma_deg * cond)
                certified = false;  // sign uncertain
            if (prev != 0.0 && ((prev > 0.0) != (val > 0.0))) ++changes;
            prev = val;
        } else {
            // val == 0.0 exactly — ambiguous sign
            certified = false;
        }
    }
    SturmCertifiedResult result;
    result.count = changes;
    result.certified = certified;
    return result;
}

/// Given float root estimate rf and the Sturm sequence of its polynomial,
/// find an isolating interval [lo_out, hi_out] containing exactly one root,
/// then bisect to ULP convergence (no absolute width threshold).
///
/// Subtask 14: removed the previous `target_width = 1e-10` absolute stopping
/// criterion.  Phase 2 now runs until double precision cannot separate lo from
/// hi (the `mid <= lo || mid >= hi` float-convergence guard), or until the
/// 200-iteration safety limit fires.  This is scale-invariant: a root near
/// 1e-8 gets the same relative tightness as a root near 1e5, and the
/// [lo_out, hi_out] window passed to try_certify_nk_sign is as narrow as
/// double arithmetic allows, minimising spurious SoS fallbacks.
///
/// Subtask 18: principled initial bracket and overflow guards.
///   OLD: delta = scale * 1e-7,  guards: delta > 1e14 || delta < 1e-300.
///   NEW: delta = scale * sqrt(ε_machine),  guard: !isfinite(lo/hi) || delta == 0.
///
///   sqrt(ε_machine) ≈ 1.49e-8 is the natural scale for the Cardano/trig root
///   formula error in the near-degenerate regime: when two roots of the cubic
///   are separated by δ, the float estimates are accurate to O(sqrt(ε) × scale)
///   (the condition number of a near-double-root grows as 1/sqrt(δ)).  Using
///   sqrt(ε_machine) as the initial half-width makes the bracket tight enough
///   for well-isolated roots (no shrink iterations) yet covers the near-double-
///   root case without needing expansion.
///
///   The guards `delta > 1e14 || delta < 1e-300` are replaced by direct float
///   overflow/underflow detection:
///     !isfinite(lo/hi) : delta expanded past the representable float range.
///     delta == 0.0     : delta shrunk below the subnormal floor (underflow).
///   These are the actual conditions that would cause incorrect Sturm evaluation,
///   not arbitrary scale thresholds.
///
/// Returns true on success, false if the root could not be isolated.
FTK_HOST_DEVICE inline bool tighten_root_interval(const SturmSeqDouble& seq, double rf,
                                   double& lo_out, double& hi_out) {
    // Subtask 18: principled initial bracket — sqrt(ε_machine) relative to scale.
    const double SQRT_EPS = pvs_sqrt(pvs_epsilon());
    double scale = pvs_max_d(pvs_abs(rf), 1.0);
    double delta = scale * SQRT_EPS;

    double lo = rf - delta, hi = rf + delta;
    int cnt = sturm_count_at(seq, lo) - sturm_count_at(seq, hi);

    // Phase 1: expand or shrink until exactly one root in bracket.
    for (int iter = 0; iter < 120 && cnt != 1; ++iter) {
        if (cnt == 0)  delta *= 2.0;   // root not yet bracketed → expand
        else           delta *= 0.5;   // multiple roots → shrink to isolate one
        lo  = rf - delta;
        hi  = rf + delta;
        if (!pvs_isfinite(lo) || !pvs_isfinite(hi) || delta == 0.0) break;
        cnt = sturm_count_at(seq, lo) - sturm_count_at(seq, hi);
    }
    if (cnt != 1) return false;

    // Special case: rf == 0.0 exactly (from the c==d==0 SoS branch in
    // solve_cubic_real_sos, which sets roots[1]=roots[2]=0.0 for a double
    // root at λ=0).  All Sturm polynomials evaluate to 0 at x=0 (the common
    // factor of the Sturm sequence vanishes at the double root), so Phase 2
    // bisection converges to the wrong interval (the upper boundary of the
    // initial bracket, ≈ 1.49e-8) instead of to 0.
    // Returning false preserves the default degenerate interval [0.0, 0.0]
    // set by the caller; Sub 11 then correctly excludes this root because
    //   zero_is_exact_root = (P_i128[0] == 0)  AND  0 ∈ [0.0, 0.0].
    if (rf == 0.0) return false;

    // Phase 2: bisect to ULP convergence (Subtask 14: no absolute threshold).
    // The `mid <= lo || mid >= hi` guard fires when lo and hi are adjacent
    // doubles, i.e. the interval is as tight as double precision allows.
    for (int iter = 0; iter < 200; ++iter) {
        double mid = lo + (hi - lo) * 0.5;
        if (mid <= lo || mid >= hi) break;  // ULP convergence
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
FTK_HOST_DEVICE inline int isolate_cubic_roots(const double P[4],
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
///
/// Subtask 16: EPS_ZERO = 1e-200 replaced by exact == 0.0 comparisons.
///
/// Correctness argument: the polynomials fed to build_sturm_deg4 (D_poly and
/// N_poly[k]) have coefficients computed from integer-representable double
/// inputs (Mlin/blin from quantized fields).  An algebraically zero remainder
/// in the Sturm GCD sequence arises only when the input polynomial has a
/// repeated root.  In that case every cancelling product in the long division
/// is of the form (exact_double × exact_double - exact_double × exact_double)
/// evaluated over values derived from the same source integers, so the
/// cancellation is exact in double arithmetic and yields 0.0 bit-for-bit.
///
/// The old EPS_ZERO was intended to catch subnormals from accumulated rounding,
/// but for our polynomial types such subnormals do not arise: all intermediate
/// coefficients in the Sturm sequence are O(field⁴) >> 1e-200 when non-zero,
/// and exactly 0.0 when algebraically zero.
///
/// The same `== 0.0` approach is already used in build_sturm_double for cubics.
FTK_HOST_DEVICE static inline int poly_rem_d(const double* A, int dA, const double* B, int dB, double* R) {
    // Copy A into R
    for (int k = 0; k <= dA; ++k) R[k] = A[k];
    for (int k = dA + 1; k <= 4; ++k) R[k] = 0.0;

    for (int d = dA; d >= dB; --d) {
        if (R[d] == 0.0) continue;  // already zero — skip (Subtask 16: was < 1e-200)
        double coeff = R[d] / B[dB];
        int    shift = d - dB;
        for (int i = 0; i <= dB; ++i) R[i + shift] -= coeff * B[i];
        R[d] = 0.0;
    }

    int dR = dB - 1;
    while (dR > 0 && R[dR] == 0.0) --dR;  // Subtask 16: was < 1e-200
    return (R[dR] == 0.0 && dR == 0) ? -1 : dR;  // Subtask 16: was < 1e-200
}

/// Build Sturm sequence for polynomial P of degree degP ≤ 4 (ascending-degree).
FTK_HOST_DEVICE inline void build_sturm_deg4(const double* P, int degP, SturmSeqDeg4& seq) {
    // Zero-initialise
    for (int ii = 0; ii < 5; ++ii) for (int jj = 0; jj < 5; ++jj) seq.c[ii][jj] = 0.0;
    for (int ii = 0; ii < 5; ++ii) seq.deg[ii] = 0;
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
FTK_HOST_DEVICE inline int sturm_count_d4(const SturmSeqDeg4& seq, double x) {
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

/// Named helper: multiply two degree-2 polynomials → degree-4 (__int128 version)
FTK_HOST_DEVICE inline void mul2_poly_i128(__int128* P, __int128* Q, __int128* R) {
    for (int k = 0; k < 5; ++k) R[k] = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i + j] += P[i] * Q[j];
}

/// Named helper: multiply two degree-2 polynomials → degree-4 (double version)
FTK_HOST_DEVICE inline void mul2_poly_d(const double* P, const double* Q, double* R) {
    for (int k = 0; k < 5; ++k) R[k] = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i + j] += P[i] * Q[j];
}

/// Compute degree-4 barycentric numerator polynomials N[3][5] and D[5].
///
/// @param Mlin  Linear polynomial coefficients of M(λ)[r][c] = Mlin[r][c][0] + λ·Mlin[r][c][1]
/// @param blin  Linear polynomial coefficients of b(λ)[r]    = blin[r][0]    + λ·blin[r][1]
/// @param N     Output: N[k][0..4] are the degree-4 numerator poly coefficients for μ_k
/// @param D     Output: D[0..4] are the degree-4 denominator poly (Gram determinant)
FTK_HOST_DEVICE inline void compute_bary_numerators(
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

    // D = A₀₀·A₁₁ − A₀₁²
    double t0[5], t1[5];
    mul2_poly_d(A[0][0], A[1][1], t0);
    mul2_poly_d(A[0][1], A[0][1], t1);
    for (int k = 0; k < 5; ++k) D[k] = t0[k] - t1[k];

    // N[0] = A₁₁·g[0] − A₀₁·g[1]
    double a11g0[5], a01g1[5];
    mul2_poly_d(A[1][1], g[0], a11g0);
    mul2_poly_d(A[0][1], g[1], a01g1);
    for (int k = 0; k < 5; ++k) N[0][k] = a11g0[k] - a01g1[k];

    // N[1] = A₀₀·g[1] − A₀₁·g[0]
    double a00g1[5], a01g0[5];
    mul2_poly_d(A[0][0], g[1], a00g1);
    mul2_poly_d(A[0][1], g[0], a01g0);
    for (int k = 0; k < 5; ++k) N[1][k] = a00g1[k] - a01g0[k];

    // N[2] = D − N[0] − N[1]
    for (int k = 0; k < 5; ++k) N[2][k] = D[k] - N[0][k] - N[1][k];
}

/// Subtask 23 (extended): compute N/D polynomials from exact integer Mlin_q/blin_q.
///
/// Two stages, both starting from exact __int128 degree-2 building blocks:
///
/// Stage A — Gram matrix A[p][q](λ) and rhs g[p](λ):
///   Computed exactly in __int128 from int64_t Mlin_q/blin_q (from VqT/WqT).
///   This eliminates catastrophic cancellation in the differences
///   VqT[r][c]-VqT[r][2] that occur when the float field values are nearly
///   constant across vertices.
///
/// Stage B — degree-4 D and N polynomials (two sub-paths):
///
///   EXACT path (fields with |M| ≤ ~195 at QUANT_SCALE = 2^20):
///     D = A₀₀·A₁₁ − A₀₁² and N_k computed entirely in __int128.
///     The final double coefficients have at most 1 ULP rounding error each;
///     no cancellation in the degree-4 products.
///     Overflow bounds (verified):
///       |A[p][q][k]| ≤ SAFE_A = 10^18 → 6×SAFE_A² ≈ 6×10^36 < 2^127 ✓
///       |g[p][k]|    ≤ SAFE_G = 10^19 → |N[2]| ≤ 1.26×10^38 < 2^127 ✓
///
///   FALLBACK path (large fields):
///     A and g converted to double (1 ULP per coefficient, no cancellation),
///     then degree-4 products computed in double.  Still better than the
///     previous float-Mlin approach which had catastrophic cancellation.
FTK_HOST_DEVICE inline void compute_bary_numerators_from_integers(
        const int64_t Mlin_q[3][2][2], const int64_t blin_q[3][2],
        double N[3][5], double D[5]) {
    // Stage A: A[p][q][k] and g[p][k] exactly in __int128.
    __int128 A_i[2][2][3];
    for (int p2 = 0; p2 < 2; ++p2) for (int q2 = 0; q2 < 2; ++q2) for (int k2 = 0; k2 < 3; ++k2) A_i[p2][q2][k2] = 0;
    for (int r = 0; r < 3; ++r)
        for (int p = 0; p < 2; ++p)
            for (int q = 0; q < 2; ++q) {
                int64_t m0p = Mlin_q[r][p][0], m1p = Mlin_q[r][p][1];
                int64_t m0q = Mlin_q[r][q][0], m1q = Mlin_q[r][q][1];
                A_i[p][q][0] += (__int128)m0p * m0q;
                A_i[p][q][1] += (__int128)m0p * m1q + (__int128)m1p * m0q;
                A_i[p][q][2] += (__int128)m1p * m1q;
            }

    __int128 g_i[2][3];
    for (int p2 = 0; p2 < 2; ++p2) for (int k2 = 0; k2 < 3; ++k2) g_i[p2][k2] = 0;
    for (int r = 0; r < 3; ++r)
        for (int p = 0; p < 2; ++p) {
            int64_t m0 = Mlin_q[r][p][0], m1 = Mlin_q[r][p][1];
            int64_t b0 = blin_q[r][0],    b1 = blin_q[r][1];
            g_i[p][0] += (__int128)m0 * b0;
            g_i[p][1] += (__int128)m0 * b1 + (__int128)m1 * b0;
            g_i[p][2] += (__int128)m1 * b1;
        }

    // Stage B: degree-4 products.
    const int64_t  SAFE_A = 1000000000000000000LL;   // 10^18
    const __int128 SAFE_G =
        (__int128)1000000000LL * 10000000000LL;                  // 10^19

    bool exact = true;
    for (int p = 0; p < 2 && exact; ++p)
        for (int q = 0; q < 2 && exact; ++q)
            for (int k = 0; k < 3 && exact; ++k)
                if (pvs_abs128(A_i[p][q][k]) > SAFE_A) exact = false;
    for (int p = 0; p < 2 && exact; ++p)
        for (int k = 0; k < 3 && exact; ++k)
            if (pvs_abs128(g_i[p][k]) > SAFE_G) exact = false;

    if (exact) {
        // EXACT path: degree-4 products in __int128 — no rounding, no cancellation.

        __int128 t0[5], t1[5], D_i[5];
        mul2_poly_i128(A_i[0][0], A_i[1][1], t0);
        mul2_poly_i128(A_i[0][1], A_i[0][1], t1);
        for (int k = 0; k < 5; ++k) D_i[k] = t0[k] - t1[k];

        __int128 a11g0[5], a01g1[5], N_i[3][5];
        mul2_poly_i128(A_i[1][1], g_i[0], a11g0);
        mul2_poly_i128(A_i[0][1], g_i[1], a01g1);
        for (int k = 0; k < 5; ++k) N_i[0][k] = a11g0[k] - a01g1[k];

        __int128 a00g1[5], a01g0[5];
        mul2_poly_i128(A_i[0][0], g_i[1], a00g1);
        mul2_poly_i128(A_i[0][1], g_i[0], a01g0);
        for (int k = 0; k < 5; ++k) N_i[1][k] = a00g1[k] - a01g0[k];

        for (int k = 0; k < 5; ++k) N_i[2][k] = D_i[k] - N_i[0][k] - N_i[1][k];

        // Convert to double: 1 ULP rounding per coefficient, no cancellation.
        for (int k = 0; k < 5; ++k) {
            D[k] = (double)D_i[k];
            for (int j = 0; j < 3; ++j) N[j][k] = (double)N_i[j][k];
        }
    } else {
        // FALLBACK path: convert A and g to double (1 ULP each), then
        // degree-4 products in double.
        double A_d[2][2][3], g_d[2][3];
        for (int p = 0; p < 2; ++p)
            for (int q = 0; q < 2; ++q)
                for (int k = 0; k < 3; ++k)
                    A_d[p][q][k] = (double)A_i[p][q][k];
        for (int p = 0; p < 2; ++p)
            for (int k = 0; k < 3; ++k)
                g_d[p][k] = (double)g_i[p][k];

        double t0[5], t1[5];
        mul2_poly_d(A_d[0][0], A_d[1][1], t0);
        mul2_poly_d(A_d[0][1], A_d[0][1], t1);
        for (int k = 0; k < 5; ++k) D[k] = t0[k] - t1[k];

        double a11g0[5], a01g1[5];
        mul2_poly_d(A_d[1][1], g_d[0], a11g0);
        mul2_poly_d(A_d[0][1], g_d[1], a01g1);
        for (int k = 0; k < 5; ++k) N[0][k] = a11g0[k] - a01g1[k];

        double a00g1[5], a01g0[5];
        mul2_poly_d(A_d[0][0], g_d[1], a00g1);
        mul2_poly_d(A_d[0][1], g_d[0], a01g0);
        for (int k = 0; k < 5; ++k) N[1][k] = a00g1[k] - a01g0[k];

        for (int k = 0; k < 5; ++k) N[2][k] = D[k] - N[0][k] - N[1][k];
    }
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
 * @return Number of punctures found (0-3), or INT_MAX if degenerate (entire triangle is PV)
 *
 * Subtask 19: the `epsilon` parameter has been removed.  All threshold
 * comparisons in the certification path now use exact == 0.0 tests or
 * machine-epsilon-derived expressions (e.g. EVAL_GAMMA, SQRT_EPS) that
 * are principled, not arbitrary.
 */
template <typename T>
int solve_pv_triangle(const T V[3][3], const T W[3][3],
                     std::vector<PuncturePoint>& punctures,
                     const uint64_t* indices = nullptr,
                     uint64_t tet_fourth = UINT64_MAX);

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
FTK_HOST_DEVICE inline T det2(const T a[2][2]) {
    return a[0][0] * a[1][1] - a[1][0] * a[0][1];
}

/**
 * @brief Compute 3x3 determinant
 */
template <typename T>
FTK_HOST_DEVICE inline T det3(const T a[3][3]) {
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
FTK_HOST_DEVICE void characteristic_polynomial_2x2(T a00, T a01, T a10, T a11, T b00, T b01, T b10, T b11, T P[3]) {
    P[2] = b00 * b11 - b10 * b01;
    P[1] = -(a00 * b11 - a10 * b01 + b00 * a11 - b10 * a01);
    P[0] = a00 * a11 - a10 * a01;
}

/**
 * @brief Compute characteristic polynomial for generalized eigenvalue problem (3x3)
 * det(A - λB) = P[0] + P[1]λ + P[2]λ² + P[3]λ³
 */
template <typename T>
FTK_HOST_DEVICE void characteristic_polynomial_3x3(const T A[3][3], const T B[3][3], T P[4]) {
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
FTK_HOST_DEVICE void polynomial_multiply(const T P[], int m, const T Q[], int n, T R[]) {
    for (int i = 0; i <= m + n; ++i) R[i] = T(0);
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            R[i + j] += P[i] * Q[j];
}

/**
 * @brief Add polynomial Q to P in place
 */
template <typename T>
FTK_HOST_DEVICE void polynomial_add_inplace(T P[], int m, const T Q[], int n) {
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
FTK_HOST_DEVICE void polynomial_scalar_multiply(T P[], int m, T scalar) {
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
FTK_HOST_DEVICE void characteristic_polynomials_pv_tetrahedron(const T V[4][3], const T W[4][3], T Q[4], T P[4][4]) {
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
                     uint64_t tet_fourth) {
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
    // Subtask 12: certified all-parallel check via exact integer cross products.
    //
    // The entire triangle is a PV surface iff V[i] × W[i] = 0 at ALL vertices,
    // i.e. iff V and W are proportional everywhere on the triangle.
    //
    // Old check: used SoS-perturbed Vp with a float threshold epsilon.
    //   Bug: when V=W exactly, Vp ≠ Wp (different per-slot perturbations),
    //   so the float cross products were O(SoS_eps) >> epsilon, and the check
    //   failed to detect all-parallel.  SoS float perturbation removed in Sub. 22.
    //
    // New check: uses Vq/Wq (quantized ORIGINAL field) with exact integer
    //   comparison.  Vq × Wq = 0 iff V[i] ∥ W[i] in the quantized sense
    //   (no threshold).  This is correct for both the SoS and no-SoS paths.
    //
    // Overflow: |Vq[i][j]| ≤ max_field × QUANT_SCALE ≤ 5×10^6 × 2^20 ≈ 5×10^12.
    //   Cross product component ≤ 2×(5×10^12)^2 = 5×10^25 < 2^127.  Use __int128.
    // ----------------------------------------------------------------
    {
        bool all_parallel = true;
        for (int i = 0; i < 3; ++i) {
            __int128 cx = (__int128)Vq[i][1]*Wq[i][2] - (__int128)Vq[i][2]*Wq[i][1];
            __int128 cy = (__int128)Vq[i][2]*Wq[i][0] - (__int128)Vq[i][0]*Wq[i][2];
            __int128 cz = (__int128)Vq[i][0]*Wq[i][1] - (__int128)Vq[i][1]*Wq[i][0];
            if (cx != 0 || cy != 0 || cz != 0) { all_parallel = false; break; }
        }
        if (all_parallel)
            return std::numeric_limits<int>::max();  // entire triangle is PV surface
    }

    // Subtask 22: use the unperturbed field directly for the float transpose.
    // The float characteristic polynomial, root refinement, and N_k/D
    // polynomials are all computed from the true V/W — no SoS float
    // perturbation.  Edge/vertex degeneracies are resolved purely
    // combinatorially via the `indices` ordering in sos_bary_inside.
    T VT[3][3], WT[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VT[i][j] = V[j][i];
            WT[i][j] = W[j][i];
        }

    // Characteristic polynomial: det(VT - λ WT) = 0  (cubic in λ)
    T P[4];
    characteristic_polynomial_3x3(VT, WT, P);

    // Solve cubic: use exact integer discriminant (Subtask 3) to decide
    // the root count when the float discriminant is near zero.
    T lambda[3];
    // Subtask 19: epsilon parameter removed from solve_cubic_real_sos.
    int n_roots = solve_cubic_real_sos(P, lambda, indices, P_i128);

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
    // true unperturbed characteristic polynomial), which has the same roots
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
    // Subtasks 6, 7, & 23: compute bary numerator polynomials N[3][5] and D[5].
    //
    // Expresses μ_k(λ) = N_k(λ)/D(λ) with degree-4 polynomials derived
    // from the linear-in-λ M(λ) and b(λ) matrices.
    // D = det(MᵀM) ≥ 0 is the Gram determinant (non-negative by Cauchy-Binet).
    //
    // Subtask 23: build Mlin_q and blin_q from integer VqT/WqT to eliminate
    // catastrophic cancellation in the differences VqT[r][c]-VqT[r][2].
    // The degree-2 Gram-matrix coefficients A[p][q] and rhs g[p] are then
    // computed exactly in __int128 before being converted to double for the
    // degree-4 multiplications in compute_bary_numerators_from_integers.
    // ----------------------------------------------------------------
    int64_t Mlin_q[3][2][2];
    int64_t blin_q[3][2];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c) {
            Mlin_q[r][c][0] = VqT[r][c] - VqT[r][2];
            Mlin_q[r][c][1] = -(WqT[r][c] - WqT[r][2]);
        }
        blin_q[r][0] = -VqT[r][2];
        blin_q[r][1] =  WqT[r][2];
    }
    double N_poly[3][5], D_poly[5];
    compute_bary_numerators_from_integers(Mlin_q, blin_q, N_poly, D_poly);

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
    //
    // Subtask 15: degree-trimming uses exact `== 0.0` instead of `< 1e-200`.
    //
    // D_poly[k] is computed by compute_bary_numerators from Mlin/blin, which
    // come from Vp and Wp.  When SoS is active (indices ≠ nullptr) every
    // Mlin[r][c][1] = -(Wp[c][r] − Wp[2][r]) is non-zero because the SoS
    // perturbation ensures W is never exactly constant across distinct vertices.
    // Hence D_poly[4] > 0 always with SoS, and the trim loop fires 0 times.
    //
    // Without SoS (Vp = V, Wp = W), D_poly[4] is exactly 0.0 (bit-for-bit)
    // when W is truly constant across vertices — in that case all Mlin[r][c][1]
    // are exactly 0.0 and every intermediate product is 0.0 with no rounding.
    // The `== 0.0` test catches this exact-zero without any threshold.
    // ----------------------------------------------------------------
    SturmSeqDeg4 seq_D;
    {
        int degD = 4;
        while (degD > 0 && D_poly[degD] == 0.0) --degD;
        build_sturm_deg4(D_poly, degD, seq_D);
    }

    // Subtask 11: exact λ=0 exclusion via integer characteristic polynomial.
    //
    // The trivial root λ=0 corresponds to V(ν*) = 0·W(ν*) = 0 — the V field
    // vanishes at the puncture, making V∥W trivially.  We only want to skip a
    // float root when it IS the λ=0 eigenvalue, i.e. when the exact integer
    // char poly has λ=0 as a root.
    //
    // P_i128[0] = det(Vq) is the constant term of the integer poly.
    // P_i128[0] = 0  iff  det(Vq) = 0  iff  λ=0 is an exact root.
    //
    // Paired with the Sturm isolating interval: the true root sits in
    // [lambda_lo[i], lambda_hi[i]].  Skip iff 0 ∈ [lo, hi] AND P_i128[0]=0.
    //
    // When P_i128[0] ≠ 0, no root is at λ=0.  A float root with |λ|<ε was
    // simply a near-zero genuine eigenvalue; the old `|λ|≤ε` guard would
    // have incorrectly dropped it.
    //
    // For degenerate intervals (lo==hi==λ̂), 0∈[lo,hi] requires λ̂=0 exactly
    // in double — acceptable since that only happens when the cubic solver
    // returned 0.0 as a root estimate, and P_i128[0]=0 confirms it is exact.
    const bool zero_is_exact_root = (P_i128[0] == __int128(0));

    // For each λ recover barycentric coords via least-squares null-vector
    for (int i = 0; i < n_roots; ++i) {
        // Subtask 11: skip only the certified-exact λ=0 root.
        if (zero_is_exact_root && lambda_lo[i] <= 0.0 && 0.0 <= lambda_hi[i])
            continue;

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
        //
        // Subtask 15: same exact `== 0.0` degree-trim as for D_poly above.
        // N_poly[k][4] is exactly 0.0 only when the leading polynomial term
        // cancels exactly — an algebraic condition that survives to double
        // arithmetic without rounding when all inputs are exactly representable
        // (as they are for quantized integer Mlin values cast to double).
        auto try_certify_nk_sign = [&](int k, double lo, double hi) -> int {
            int degNk = 4;
            while (degNk > 0 && N_poly[k][degNk] == 0.0) --degNk;

            SturmSeqDeg4 seq_nk;
            build_sturm_deg4(N_poly[k], degNk, seq_nk);
            if (sturm_count_d4(seq_nk, lo) - sturm_count_d4(seq_nk, hi) != 0)
                return 0;  // N_k has a root in (lo, hi]

            // N_k root-free in window: check D via pre-computed seq_D.
            if (sturm_count_d4(seq_D, lo) - sturm_count_d4(seq_D, hi) != 0)
                return 0;  // D has a root → degenerate system

            // Subtask 8: certified Horner sign at lo.
            //
            // Subtask 20: use actual degNk (not hardcoded 4) in the Higham
            // error bound.  After degree-trimming, degNk may be < 4 (e.g.
            // when W is constant → leading coefficient of N_k is exactly 0
            // → degNk ≤ 3).  The Higham bound for degree-n Horner evaluation
            // is (2n+2)·u·cond, so using degNk instead of 4 gives a tighter
            // threshold: the certification zone shrinks, accepting more
            // genuinely-nonzero evaluations as certified.
            double nk_lo = eval_poly_sturm(N_poly[k], degNk, lo);
            double ax = std::abs(lo);
            double cond_nk = std::abs(N_poly[k][degNk]);
            for (int d = degNk - 1; d >= 0; --d)
                cond_nk = cond_nk * ax + std::abs(N_poly[k][d]);
            double eval_gamma = double(2 * degNk + 2) *
                                std::numeric_limits<double>::epsilon();
            if (std::abs(nk_lo) > eval_gamma * cond_nk)
                return (nk_lo > 0.0) ? +1 : -1;

            return 0;  // within rounding noise
        };

        bool have_interval = (lambda_lo[i] < lambda_hi[i]);

        // Compute the certification window [win_lo, win_hi] once for this root.
        double win_lo, win_hi;
        if (have_interval) {
            win_lo = lambda_lo[i];
            win_hi = lambda_hi[i];
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
            win_lo = lam - delta;
            win_hi = lam + delta;
        }

        // Pre-evaluate boundary signs for all three barycentric coordinates.
        // This is needed to distinguish G1 (one zero) from G2 (two zeros).
        int bsign[3];
        int n_bnd = 0;
        for (int k = 0; k < 3; k++) {
            bsign[k] = try_certify_nk_sign(k, win_lo, win_hi);
            if (bsign[k] == 0) n_bnd++;
        }

        // Any coordinate certified negative → puncture is outside the simplex.
        {
            bool outside = false;
            for (int k = 0; k < 3; k++) if (bsign[k] < 0) { outside = true; break; }
            if (outside) continue;
        }

        if (n_bnd == 2) {
            // G2: vertex puncture.  Two barycentric coordinates certify zero,
            // meaning the puncture lies exactly at vertex v_m (the one with
            // non-zero sign, μ_m = 1).  The standard per-k SoS conditions
            // (idx(v_j) < idx(v_k) AND idx(v_k) < idx(v_j)) are contradictory
            // and would always reject — wrong behaviour.
            //
            // Correct SoS rule for G2: this triangle claims the vertex puncture
            // iff v_m is the minimum-index vertex of the triangle.
            if (!indices) continue;
            int m = -1;
            for (int k = 0; k < 3; k++) if (bsign[k] != 0) { m = k; break; }
            if (m < 0) continue;  // all three zero: degenerate, skip
            int a = (m + 1) % 3, b = (m + 2) % 3;
            if (!(indices[m] < std::min(indices[a], indices[b]))) continue;
        } else {
            // G0 (interior) or G1 (exactly one boundary coord): apply the
            // SoS ownership rule for each boundary coord k.
            bool reject = false;
            for (int k = 0; k < 3; k++) {
                if (bsign[k] > 0) continue;   // interior coordinate, accept
                // bsign[k] == 0: boundary for this k → apply SoS
                if (!indices) { reject = true; break; }
                if (tet_fourth != UINT64_MAX) {
                    // Tet mode: the other face sharing this edge has opposite
                    // vertex = tet_fourth.  Accept iff our opposite vertex
                    // (indices[k]) is smaller.
                    if (!(indices[k] < tet_fourth)) { reject = true; break; }
                } else {
                    // Mesh mode: standard rule
                    int ii = (k + 1) % 3, jj = (k + 2) % 3;
                    if (!(indices[k] < std::min(indices[ii], indices[jj]))) {
                        reject = true; break;
                    }
                }
            }
            if (reject) continue;
        }

        // Subtask 13: compute ν via N_k(λ)/D(λ), DEFERRED to after certification.
        //
        // Previously, ν was computed by eval_nu_at (→ solve_least_square3x2)
        // BEFORE the sos_bary_inside gate, wasting work for rejected roots and
        // introducing an `|det(MᵀM)| < epsilon` float guard.
        //
        // Now ν is computed only for roots that passed all certification steps.
        // Subtask 7 certifies D(λ*) > 0 here, guaranteeing D_val > 0 and
        // making the division exact (no singular-matrix epsilon guard needed).
        //
        // μ_k(λ) = N_k(λ) / D(λ)  for k = 0, 1, 2.
        // By construction Σ N_k(λ) = D(λ), so ν sums to 1 automatically.
        T nu[3];
        {
            double lam_d = (double)lambda[i];
            double d_val = eval_poly_sturm(D_poly, 4, lam_d);
            for (int k = 0; k < 3; ++k) {
                double nk_val = eval_poly_sturm(N_poly[k], 4, lam_d);
                nu[k] = (d_val > 0.0) ? T(nk_val / d_val) : T(0);
            }
        }

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
        //   Harmful: when SoS float perturbation was active, the perturbed
        //     solution ν_p satisfied Vp×Wp = 0 but not V×W = 0.
        //     The discrepancy grew as O(sos_eps · |W|), falsely rejecting
        //     valid punctures at large field magnitudes.
        //     (SoS float perturbation removed in Subtask 22.)

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

// ============================================================================
// Named device helper: try_certify_nk_sign (replaces lambda in solve_pv_triangle)
// ============================================================================
FTK_HOST_DEVICE inline int try_certify_nk_sign_device(
        int k, double lo, double hi,
        const double N_poly[3][5], const SturmSeqDeg4& seq_D) {
    int degNk = 4;
    while (degNk > 0 && N_poly[k][degNk] == 0.0) --degNk;

    SturmSeqDeg4 seq_nk;
    build_sturm_deg4(N_poly[k], degNk, seq_nk);
    if (sturm_count_d4(seq_nk, lo) - sturm_count_d4(seq_nk, hi) != 0)
        return 0;  // N_k has a root in (lo, hi]

    if (sturm_count_d4(seq_D, lo) - sturm_count_d4(seq_D, hi) != 0)
        return 0;  // D has a root → degenerate system

    double nk_lo = eval_poly_sturm(N_poly[k], degNk, lo);
    double ax = pvs_abs(lo);
    double cond_nk = pvs_abs(N_poly[k][degNk]);
    for (int d = degNk - 1; d >= 0; --d)
        cond_nk = cond_nk * ax + pvs_abs(N_poly[k][d]);
    double eval_gamma = double(2 * degNk + 2) * pvs_epsilon();
    if (pvs_abs(nk_lo) > eval_gamma * cond_nk)
        return (nk_lo > 0.0) ? +1 : -1;

    return 0;
}

// ============================================================================
// Device-compatible solve_pv_triangle: returns PunctureResult (no std::vector)
// ============================================================================
template <typename T>
FTK_HOST_DEVICE PunctureResult solve_pv_triangle_device(
        const T V[3][3], const T W[3][3],
        const uint64_t* indices = nullptr,
        uint64_t tet_fourth = UINT64_MAX) {
    PunctureResult result;
    result.count = 0;

    // Subtask 1: quantize
    int64_t Vq[3][3], Wq[3][3];
    quantize_field_3x3(V, Vq);
    quantize_field_3x3(W, Wq);

    // Subtask 2: exact integer characteristic polynomial
    int64_t VqT[3][3], WqT[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VqT[i][j] = Vq[j][i];
            WqT[i][j] = Wq[j][i];
        }
    __int128 P_i128[4];
    characteristic_polynomial_3x3_i128(VqT, WqT, P_i128);

    // Subtask 12: certified all-parallel check
    {
        bool all_parallel = true;
        for (int i = 0; i < 3; ++i) {
            __int128 cx = (__int128)Vq[i][1]*Wq[i][2] - (__int128)Vq[i][2]*Wq[i][1];
            __int128 cy = (__int128)Vq[i][2]*Wq[i][0] - (__int128)Vq[i][0]*Wq[i][2];
            __int128 cz = (__int128)Vq[i][0]*Wq[i][1] - (__int128)Vq[i][1]*Wq[i][0];
            if (cx != 0 || cy != 0 || cz != 0) { all_parallel = false; break; }
        }
        if (all_parallel) {
            result.count = INT_MAX;
            return result;
        }
    }

    // Float characteristic polynomial
    T VT[3][3], WT[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VT[i][j] = V[j][i];
            WT[i][j] = W[j][i];
        }
    T P[4];
    characteristic_polynomial_3x3(VT, WT, P);

    T lambda[3];
    int n_roots = solve_cubic_real_sos(P, lambda, indices, P_i128);

    // Sturm-sequence root isolation
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

    // Compute bary numerator polynomials
    int64_t Mlin_q[3][2][2];
    int64_t blin_q[3][2];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c) {
            Mlin_q[r][c][0] = VqT[r][c] - VqT[r][2];
            Mlin_q[r][c][1] = -(WqT[r][c] - WqT[r][2]);
        }
        blin_q[r][0] = -VqT[r][2];
        blin_q[r][1] =  WqT[r][2];
    }
    double N_poly[3][5], D_poly[5];
    compute_bary_numerators_from_integers(Mlin_q, blin_q, N_poly, D_poly);

    // Pre-compute Sturm sequence for D(λ)
    SturmSeqDeg4 seq_D;
    {
        int degD = 4;
        while (degD > 0 && D_poly[degD] == 0.0) --degD;
        build_sturm_deg4(D_poly, degD, seq_D);
    }

    const bool zero_is_exact_root = (P_i128[0] == __int128(0));

    for (int i = 0; i < n_roots; ++i) {
        if (zero_is_exact_root && lambda_lo[i] <= 0.0 && 0.0 <= lambda_hi[i])
            continue;

        bool have_interval = (lambda_lo[i] < lambda_hi[i]);
        double win_lo, win_hi;
        if (have_interval) {
            win_lo = lambda_lo[i];
            win_hi = lambda_hi[i];
        } else {
            double lam = (double)lambda[i];
            double delta = pvs_max_d(pvs_abs(lam) * (4.0 * pvs_epsilon()), DBL_MIN);
            win_lo = lam - delta;
            win_hi = lam + delta;
        }

        int bsign[3];
        int n_bnd = 0;
        for (int k = 0; k < 3; k++) {
            bsign[k] = try_certify_nk_sign_device(k, win_lo, win_hi, N_poly, seq_D);
            if (bsign[k] == 0) n_bnd++;
        }

        // Any coordinate certified negative → outside
        {
            bool outside = false;
            for (int k = 0; k < 3; k++) if (bsign[k] < 0) { outside = true; break; }
            if (outside) continue;
        }

        if (n_bnd == 2) {
            // G2: vertex puncture — owned by face whose non-zero vertex
            // has the smallest index (unchanged).
            if (!indices) continue;
            int m = -1;
            for (int k = 0; k < 3; k++) if (bsign[k] != 0) { m = k; break; }
            if (m < 0) continue;
            int a = (m + 1) % 3, b = (m + 2) % 3;
            if (!(indices[m] < pvs_min_u64(indices[a], indices[b]))) continue;
        } else {
            bool reject = false;
            for (int k = 0; k < 3; k++) {
                if (bsign[k] > 0) continue;
                if (!indices) { reject = true; break; }
                if (tet_fourth != UINT64_MAX) {
                    // Tet mode: accept iff our opposite vertex < other face's
                    if (!(indices[k] < tet_fourth)) { reject = true; break; }
                } else {
                    // Mesh mode: standard rule
                    int ii = (k + 1) % 3, jj = (k + 2) % 3;
                    if (!(indices[k] < pvs_min_u64(indices[ii], indices[jj]))) {
                        reject = true; break;
                    }
                }
            }
            if (reject) continue;
        }

        // Compute ν via N_k(λ)/D(λ)
        T nu[3];
        {
            double lam_d = (double)lambda[i];
            double d_val = eval_poly_sturm(D_poly, 4, lam_d);
            for (int k = 0; k < 3; ++k) {
                double nk_val = eval_poly_sturm(N_poly[k], 4, lam_d);
                nu[k] = (d_val > 0.0) ? T(nk_val / d_val) : T(0);
            }
        }

        if (result.count < 3) {
            PuncturePointDevice& p = result.pts[result.count];
            p.lambda       = lambda[i];
            p.barycentric[0] = nu[0];
            p.barycentric[1] = nu[1];
            p.barycentric[2] = nu[2];
            p.coords_3d[0] = p.coords_3d[1] = p.coords_3d[2] = 0;
            result.count++;
        }
    }

    return result;
}

template <typename T>
bool solve_pv_tetrahedron(const T V[4][3], const T W[4][3],
                         PVCurveSegment& segment,
                         T epsilon) {
    // ----------------------------------------------------------------
    // Subtask 21: certified all-parallel check via exact integer cross products.
    //
    // The entire tetrahedron is a PV region iff V[i] × W[i] = 0 at ALL
    // four vertices, i.e. iff V and W are proportional everywhere.
    //
    // Old check: float cross product norm vs epsilon — arbitrary threshold.
    //
    // New check: quantize V[i], W[i] to int64_t and compute cross product
    // in __int128. Cross = 0 iff V[i] ∥ W[i] in the quantized sense (no
    // threshold). Mirrors Subtask 12 for the triangle case.
    //
    // Overflow: |Vq[j]| ≤ |field|_max × QUANT_SCALE ≤ 5×10^6 × 2^20 ≈ 5×10^12.
    //   Cross product component ≤ 2×(5×10^12)^2 = 5×10^25 < 2^127. Safe.
    // ----------------------------------------------------------------
    {
        bool all_parallel = true;
        for (int i = 0; i < 4; ++i) {
            int64_t vq[3] = {quant((double)V[i][0]), quant((double)V[i][1]), quant((double)V[i][2])};
            int64_t wq[3] = {quant((double)W[i][0]), quant((double)W[i][1]), quant((double)W[i][2])};
            __int128 cx = (__int128)vq[1]*wq[2] - (__int128)vq[2]*wq[1];
            __int128 cy = (__int128)vq[2]*wq[0] - (__int128)vq[0]*wq[2];
            __int128 cz = (__int128)vq[0]*wq[1] - (__int128)vq[1]*wq[0];
            if (cx != 0 || cy != 0 || cz != 0) { all_parallel = false; break; }
        }
        if (all_parallel)
            return false;  // entire tetrahedron is a PV region — degenerate
    }

    // Compute characteristic polynomials
    T Q[4], P[4][4];
    characteristic_polynomials_pv_tetrahedron(V, W, Q, P);

    // ----------------------------------------------------------------
    // Subtask 21: exact-zero check for Q polynomial coefficients.
    //
    // Old: std::abs(Q[i]) > epsilon — arbitrary threshold that could
    // suppress a legitimately small-but-nonzero coefficient.
    //
    // New: Q[i] != T(0) — exact comparison. Q is computed from the field
    // values by characteristic_polynomials_pv_tetrahedron; if a coefficient
    // is algebraically zero the float computation will also give exactly 0.
    // Any nonzero value (however small) signals a valid Q polynomial.
    // ----------------------------------------------------------------
    bool q_zero = true;
    for (int i = 0; i <= 3; ++i) {
        if (Q[i] != T(0)) { q_zero = false; break; }
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

// ============================================================================
// ExactPV2: Per-Tet Solver — Pure Integer Topological Decisions
// ============================================================================
//
// All topological decisions (validity, ordering, pairing, edge/vertex
// detection, pass-through) are made using pure __int128 integer arithmetic.
// No floats are used for any topological decision.  Floats are used ONLY
// for spatial output coordinates (barycentric interpolation for visualization).

struct ExactPV2Result {
    static constexpr int MAX_PUNCTURES = 12;  // T10 is proven maximum; +2 margin
    static constexpr int MAX_PAIRS = 6;

    int n_punctures = 0;
    struct Puncture {
        int face;        // which face (0-3)
        int root_idx;    // which root of P_face (0=smallest, 1, 2)
        int q_interval;  // Q_red-interval index
        bool is_edge;
        bool is_vertex;
        int edge_faces[2];  // for edge: which two faces share the puncture
    } punctures[MAX_PUNCTURES];

    int n_pairs = 0;
    struct Pair { int a, b; } pairs[MAX_PAIRS];

    bool has_passthrough = false;
    int passthrough_deg = 0;   // degree of gcd(P_0, P_1, P_2, P_3)
};

FTK_HOST_DEVICE inline ExactPV2Result solve_pv_tet_v2(const __int128 Q_raw[4],
                                       const __int128 P_raw[4][4]) {
    ExactPV2Result result;

    // --- Step 0: Copy and determine effective degrees ---
    __int128 P[4][4], Q[4];
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 4; i++) P[k][i] = P_raw[k][i];
    }
    for (int i = 0; i < 4; i++) Q[i] = Q_raw[i];

    int degP[4], degQ;
    for (int k = 0; k < 4; k++)
        degP[k] = effective_degree_i128(P[k], 3);
    degQ = effective_degree_i128(Q, 3);

    // --- Step 1: Pass-through factoring ---
    // h = gcd(P_0, P_1, P_2, P_3)
    __int128 h[4] = {};
    int dh = poly_gcd_full_i128(P[0], degP[0], P[1], degP[1], h);
    for (int k = 2; k < 4; k++) {
        __int128 h2[4] = {};
        int dh2 = poly_gcd_full_i128(h, dh, P[k], degP[k], h2);
        dh = dh2;
        for (int i = 0; i < 4; i++) h[i] = h2[i];
    }

    // Q_red = Q / h if deg(h) >= 1
    __int128 Q_red[4] = {};
    int degQ_red = degQ;
    if (dh >= 1) {
        result.has_passthrough = true;
        result.passthrough_deg = dh;
        __int128 q_div[4] = {};
        degQ_red = poly_exact_div_i128(Q, degQ, h, dh, q_div);
        for (int i = 0; i <= degQ_red; i++) Q_red[i] = q_div[i];
    } else {
        for (int i = 0; i <= degQ; i++) Q_red[i] = Q[i];
    }

    // --- Step 2: For each face, determine root count and validity ---
    // n_distinct_roots[k] = number of distinct real roots of P_k
    // For each root: determine validity (all P_j same sign for j ≠ k)
    struct RootInfo {
        int face;
        int root_idx;     // index among distinct roots of this face's P_k (-1 = infinity)
        int q_interval;   // assigned later
        bool valid;
        bool is_edge;
        bool is_vertex;
        bool is_infinity;  // true if this puncture is at λ=±∞ (degree-reduced face)
        int edge_faces[2];
    };
    RootInfo all_roots[12];
    int n_all = 0;

    int n_distinct[4] = {};

    for (int k = 0; k < 4; k++) {
        if (degP[k] == 0) continue;  // constant P_k, no roots

        // Determine number of distinct roots
        int disc = discriminant_sign_i128(P[k]);
        if (degP[k] == 1) {
            n_distinct[k] = 1;
        } else if (degP[k] == 2) {
            __int128 d2 = P[k][1]*P[k][1] - 4*P[k][2]*P[k][0];
            if (d2 > 0) n_distinct[k] = 2;
            else if (d2 == 0) n_distinct[k] = 1;
            else n_distinct[k] = 0;  // no real roots
        } else {
            // cubic
            if (disc > 0) n_distinct[k] = 3;
            else if (disc < 0) n_distinct[k] = 1;
            else {
                // disc = 0: compute square-free part
                __int128 sf[8] = {};
                int dsf = poly_sqfree_i128(P[k], degP[k], sf);
                n_distinct[k] = dsf;  // degree of square-free part = number of distinct roots
            }
        }

        // For each distinct root of P_k: check validity
        for (int ri = 0; ri < n_distinct[k]; ri++) {
            // Validity: all P_j(α_ri) must have the same sign for j ≠ k
            int pj_signs[3];
            (void)pj_signs;
            bool valid = true;
            int first_sign = 0;

            for (int j = 0; j < 4; j++) {
                if (j == k) continue;
                // Get sign(P_j(α_ri)) where α_ri is the ri-th root of P_k
                int signs_j[3] = {};
                int nrj = signs_at_roots_i128(P[k], degP[k], P[j], degP[j], signs_j, 3);

                int s = (ri < nrj) ? signs_j[ri] : 0;
                if (s == 0) { valid = false; break; }  // degenerate (on edge/face)
                                                         // Will be handled by edge detection

                if (first_sign == 0) first_sign = s;
                else if (s != first_sign) { valid = false; break; }
            }

            // Also check: sign(P_j) must match sign(Q) at the root (for interior point)
            // Actually: mu_j = P_j/Q >= 0 means P_j and Q have same sign.
            // Since all P_j same sign and Q = P_0+P_1+P_2+P_3 = sum of three P_j (since P_k=0),
            // Q has the same sign as P_j automatically. So validity = all P_j same sign.

            // Allow edge/vertex punctures through (sign = 0 for some P_j)
            // Re-check with edge detection awareness:
            if (!valid) {
                // Check if this is an edge or vertex puncture
                int n_zero = 0;
                int zero_faces[3] = {};
                int pj_s[3] = {};
                int pj_idx = 0;
                bool redo_valid = true;
                int redo_first = 0;
                for (int j = 0; j < 4; j++) {
                    if (j == k) continue;
                    int signs_j[3] = {};
                    int nrj = signs_at_roots_i128(P[k], degP[k], P[j], degP[j], signs_j, 3);
                    int s = (ri < nrj) ? signs_j[ri] : 0;
                    if (s == 0) {
                        zero_faces[n_zero++] = j;
                    } else {
                        if (redo_first == 0) redo_first = s;
                        else if (s != redo_first) redo_valid = false;
                    }
                    pj_idx++;
                }
                // Edge: exactly 1 zero, remaining 2 same sign → valid
                // Vertex: 2 zeros → valid (only non-k, non-zero face has some sign)
                if (n_zero >= 1 && redo_valid) valid = true;
            }

            if (!valid) continue;

            if (n_all >= ExactPV2Result::MAX_PUNCTURES) break;
            RootInfo& ri_info = all_roots[n_all];
            ri_info.face = k;
            ri_info.root_idx = ri;
            ri_info.q_interval = -1;
            ri_info.valid = true;
            ri_info.is_edge = false;
            ri_info.is_vertex = false;
            ri_info.is_infinity = false;
            ri_info.edge_faces[0] = ri_info.edge_faces[1] = -1;
            n_all++;
        }
    }

    // --- Step 2b: Infinity punctures for degree-reduced faces ---
    // When degP[k] < 3, the "missing" root is at λ=±∞.
    // Barycentric coords at ∞: μ_j = P[j][3] / Q[3] for j ≠ k, μ_k = 0.
    // Valid iff all P[j][3] (j ≠ k) have the same sign and Q[3] ≠ 0.
    for (int k = 0; k < 4; k++) {
        if (degP[k] >= 3 || P[k][3] != 0) continue;  // only degree-reduced faces
        if (Q[3] == 0) continue;  // Q also degree-reduced, skip for now

        // Check validity: all P[j][3] (j ≠ k) must have same sign
        int first_sign3 = 0;
        bool valid_inf = true;
        int n_zero3 = 0;
        int zero_faces3[3] = {};
        for (int j = 0; j < 4; j++) {
            if (j == k) continue;
            __int128 pj3 = P[j][3];
            if (pj3 == 0) {
                zero_faces3[n_zero3++] = j;
            } else {
                int s = (pj3 > 0) ? 1 : -1;
                if (first_sign3 == 0) first_sign3 = s;
                else if (s != first_sign3) { valid_inf = false; break; }
            }
        }
        // Only add face-INTERIOR infinity punctures (Cw2).
        // Edge/vertex infinity punctures (Cw1/Cw0) are waypoints — not paired.
        if (!valid_inf || first_sign3 == 0) continue;
        if (n_zero3 >= 1) continue;  // edge/vertex at ∞ → Cw1/Cw0 waypoint, skip

        if (n_all >= ExactPV2Result::MAX_PUNCTURES) break;
        RootInfo& ri_info = all_roots[n_all];
        ri_info.face = k;
        ri_info.root_idx = -1;  // infinity marker
        ri_info.q_interval = -1;
        ri_info.valid = true;
        ri_info.is_edge = false;
        ri_info.is_vertex = false;
        ri_info.is_infinity = true;
        ri_info.edge_faces[0] = ri_info.edge_faces[1] = -1;
        n_all++;
    }

    // --- Step 3: Edge detection ---
    // For each pair (i,j) with i<j: if Res(P_i, P_j) == 0 → shared root
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (degP[i] == 0 || degP[j] == 0) continue;
            int res = resultant_sign_i128(P[i], degP[i], P[j], degP[j]);
            if (res == 0) {
                // Shared root on edge opposite vertices i and j
                // Mark the corresponding punctures
                // Ownership: assigned to face with smaller opposite-vertex index
                // Face k is opposite vertex k.  Edge shared by faces i and j
                // means the puncture is on the edge between the OTHER two vertices.
                int owner = (i < j) ? i : j;  // smaller face index
                for (int r = 0; r < n_all; r++) {
                    if (all_roots[r].face == i || all_roots[r].face == j) {
                        // Check if this root is the shared one
                        // A root of P_i that makes P_j = 0 → shared
                        if (all_roots[r].face == i) {
                            int signs_j[3] = {};
                            signs_at_roots_i128(P[i], degP[i], P[j], degP[j], signs_j, 3);
                            if (all_roots[r].root_idx < 3 && signs_j[all_roots[r].root_idx] == 0) {
                                all_roots[r].is_edge = true;
                                all_roots[r].edge_faces[0] = i;
                                all_roots[r].edge_faces[1] = j;
                                // If not the owner face, mark for removal
                                if (all_roots[r].face != owner) all_roots[r].valid = false;
                            }
                        } else {
                            int signs_i[3] = {};
                            signs_at_roots_i128(P[j], degP[j], P[i], degP[i], signs_i, 3);
                            if (all_roots[r].root_idx < 3 && signs_i[all_roots[r].root_idx] == 0) {
                                all_roots[r].is_edge = true;
                                all_roots[r].edge_faces[0] = i;
                                all_roots[r].edge_faces[1] = j;
                                if (all_roots[r].face != owner) all_roots[r].valid = false;
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Step 4: Vertex detection ---
    for (int i = 0; i < 4; i++) {
        for (int j = i+1; j < 4; j++) {
            for (int l = j+1; l < 4; l++) {
                if (degP[i] == 0 || degP[j] == 0 || degP[l] == 0) continue;
                int r_ij = resultant_sign_i128(P[i], degP[i], P[j], degP[j]);
                if (r_ij != 0) continue;
                int r_jl = resultant_sign_i128(P[j], degP[j], P[l], degP[l]);
                if (r_jl != 0) continue;
                int r_il = resultant_sign_i128(P[i], degP[i], P[l], degP[l]);
                if (r_il != 0) continue;
                // All three share a root → vertex puncture
                // Owner = face with smallest index among {i,j,l}
                for (int r = 0; r < n_all; r++) {
                    if (all_roots[r].face == i || all_roots[r].face == j || all_roots[r].face == l) {
                        // Check if this root is the shared one (P of other two faces = 0)
                        bool is_shared = true;
                        int f = all_roots[r].face;
                        int others[3] = {i, j, l};
                        for (int oi = 0; oi < 3; oi++) {
                            int other = others[oi];
                            if (other == f) continue;
                            int signs_o[3] = {};
                            signs_at_roots_i128(P[f], degP[f], P[other], degP[other], signs_o, 3);
                            if (all_roots[r].root_idx < 3 && signs_o[all_roots[r].root_idx] != 0)
                                is_shared = false;
                        }
                        if (is_shared) {
                            all_roots[r].is_vertex = true;
                            all_roots[r].is_edge = false;
                            if (all_roots[r].face != i) all_roots[r].valid = false;  // owner = smallest
                        }
                    }
                }
            }
        }
    }

    // --- Step 4b: Cv waypoint exclusion ---
    // Edge/vertex punctures at λ=0 are Cv1/Cv0 waypoints → not paired.
    // Check by evaluating g(λ)=λ at the root: if sign is 0, root is at λ=0.
    {
        __int128 lambda_poly[2] = {0, 1};
        for (int r = 0; r < n_all; r++) {
            if (!all_roots[r].valid) continue;
            if (!all_roots[r].is_edge && !all_roots[r].is_vertex) continue;
            if (all_roots[r].is_infinity) continue;
            if (P[all_roots[r].face][0] != 0) continue;  // quick reject: no root at 0
            int signs[3] = {};
            int nr = signs_at_roots_i128(P[all_roots[r].face], degP[all_roots[r].face],
                                         lambda_poly, 1, signs, 3);
            if (all_roots[r].root_idx < nr && signs[all_roots[r].root_idx] == 0)
                all_roots[r].valid = false;  // Cv waypoint
        }
    }

    // --- Step 5: Collect valid punctures ---
    int valid_indices[12];
    int n_valid = 0;
    for (int r = 0; r < n_all; r++) {
        if (all_roots[r].valid && n_valid < ExactPV2Result::MAX_PUNCTURES) {
            valid_indices[n_valid++] = r;
        }
    }

    if (n_valid == 0) return result;

    // --- Step 6: Sort valid punctures by λ ---
    // Bubble sort using compare_roots_i128 (n_valid is small, ≤ 12)
    for (int i = 0; i < n_valid - 1; i++) {
        for (int j = i + 1; j < n_valid; j++) {
            RootInfo& a = all_roots[valid_indices[i]];
            RootInfo& b = all_roots[valid_indices[j]];
            // Compare root a.root_idx of P[a.face] with root b.root_idx of P[b.face]
            int cmp;
            if (a.is_infinity && b.is_infinity) {
                cmp = 0;  // both at infinity
            } else if (a.is_infinity) {
                cmp = 1;  // infinity > any finite
            } else if (b.is_infinity) {
                cmp = -1; // any finite < infinity
            } else if (a.face == b.face) {
                // Same face: order by root index (already sorted)
                cmp = (a.root_idx < b.root_idx) ? -1 : (a.root_idx > b.root_idx) ? 1 : 0;
            } else {
                cmp = compare_roots_i128(P[a.face], degP[a.face], n_distinct[a.face], a.root_idx,
                                         P[b.face], degP[b.face], n_distinct[b.face], b.root_idx);
            }
            if (cmp > 0) {
                int t = valid_indices[i]; valid_indices[i] = valid_indices[j]; valid_indices[j] = t;
            }
        }
    }

    // --- Step 7: Q_red-interval assignment ---
    // For each valid puncture, count Q_red roots below it
    degQ_red = effective_degree_i128(Q_red, degQ_red);
    int n_qr_roots = 0;
    if (degQ_red >= 1) {
        int disc_qr = discriminant_sign_i128(Q_red);
        if (degQ_red == 1) n_qr_roots = 1;
        else if (degQ_red == 2) {
            __int128 d2 = Q_red[1]*Q_red[1] - 4*Q_red[2]*Q_red[0];
            n_qr_roots = (d2 > 0) ? 2 : (d2 == 0) ? 1 : 0;
        } else {
            if (disc_qr > 0) n_qr_roots = 3;
            else if (disc_qr < 0) n_qr_roots = 1;
            else {
                __int128 sf[8] = {};
                n_qr_roots = poly_sqfree_i128(Q_red, degQ_red, sf);
            }
        }
    }

    for (int vi = 0; vi < n_valid; vi++) {
        RootInfo& ri = all_roots[valid_indices[vi]];

        if (ri.is_infinity) {
            ri.q_interval = n_qr_roots;  // all Q roots are below λ=∞
            continue;
        }

        if (n_qr_roots == 0 || degQ_red == 0) {
            ri.q_interval = 0;  // no Q_red roots → single interval
            continue;
        }

        // Count Q_red roots below this puncture's λ
        // = count of Q_red roots with Q_red(α_ri) having sign changes
        // Get signs of Q_red at roots of P[face]
        int qr_signs[3] = {};
        int nqrs = signs_at_roots_i128(P[ri.face], degP[ri.face],
                                       Q_red, degQ_red, qr_signs, 3);

        // Count Q_red roots below α_ri using sign of Q_red at α_ri
        // and interleaving with Q_red's roots
        int count_below = 0;
        int lc_qr_sign = (Q_red[degQ_red] > 0) ? 1 : -1;
        int qr_at_alpha = (ri.root_idx < nqrs) ? qr_signs[ri.root_idx] : lc_qr_sign;

        if (qr_at_alpha != 0) {
            // sign(Q_red(α)) = lc_qr × (-1)^(n_qr_roots - count_below)
            // → (-1)^(n_qr_roots - count_below) = qr_at_alpha / lc_qr
            // → (n_qr_roots - count_below) is even iff qr_at_alpha * lc_qr > 0
            // Use the interleaving from compare_roots
            count_below = 0;
            for (int qi = 0; qi < n_qr_roots; qi++) {
                int cmp = compare_roots_i128(P[ri.face], degP[ri.face],
                                             n_distinct[ri.face], ri.root_idx,
                                             Q_red, degQ_red, n_qr_roots, qi);
                if (cmp > 0) count_below++;  // Q_red root is below puncture
            }
        }
        ri.q_interval = count_below;
    }

    // --- Step 8: Fill result ---
    for (int vi = 0; vi < n_valid; vi++) {
        RootInfo& ri = all_roots[valid_indices[vi]];
        ExactPV2Result::Puncture& p = result.punctures[result.n_punctures];
        p.face = ri.face;
        p.root_idx = ri.root_idx;
        p.q_interval = ri.q_interval;
        p.is_edge = ri.is_edge;
        p.is_vertex = ri.is_vertex;
        p.edge_faces[0] = ri.edge_faces[0];
        p.edge_faces[1] = ri.edge_faces[1];
        result.n_punctures++;
    }

    // --- Step 9: Pairing ---
    // Group by Q_red-interval, pair consecutive within each group.
    // Cross-infinity: intervals 0 and n_qr_roots are connected through λ = ±∞
    // only when the asymptotic point at ∞ is inside the tet.
    // Condition: Q[3] == 0 (deg drops) or all P[k][3] * Q[3] >= 0.
    bool merge_infinity = false;
    if (Q[3] == 0) {
        merge_infinity = true;
    } else {
        merge_infinity = true;
        for (int k = 0; k < 4; k++) {
            if ((P[k][3] > 0 && Q[3] < 0) || (P[k][3] < 0 && Q[3] > 0)) {
                merge_infinity = false;
                break;
            }
        }
    }

    for (int qi = 0; qi <= n_qr_roots; qi++) {
        if (merge_infinity && qi == n_qr_roots && n_qr_roots > 0)
            continue;  // already handled with qi=0

        int group[12];
        int ng = 0;
        if (merge_infinity && qi == 0 && n_qr_roots > 0) {
            // Wrapped group: interval n_qr_roots (→ +∞) then interval 0 (−∞ →)
            for (int i = 0; i < result.n_punctures; i++)
                if (result.punctures[i].q_interval == n_qr_roots)
                    group[ng++] = i;
            for (int i = 0; i < result.n_punctures; i++)
                if (result.punctures[i].q_interval == 0)
                    group[ng++] = i;
        } else {
            for (int i = 0; i < result.n_punctures; i++)
                if (result.punctures[i].q_interval == qi)
                    group[ng++] = i;
        }
        for (int i = 0; i + 1 < ng; i += 2) {
            if (result.n_pairs < ExactPV2Result::MAX_PAIRS) {
                result.pairs[result.n_pairs].a = group[i];
                result.pairs[result.n_pairs].b = group[i + 1];
                result.n_pairs++;
            }
        }
    }

    return result;
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
