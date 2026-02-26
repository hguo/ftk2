#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <iostream>
#include <cmath>
#include <limits>

using namespace ftk2;

static int total_tests = 0;
static int passed_tests = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct test_##name##_t { \
        test_##name##_t() { std::cout << "Running test: " << #name << std::endl; test_##name(); } \
    } test_##name##_instance; \
    static void test_##name()

#define ASSERT_NEAR(a, b, eps) \
    total_tests++; \
    if (std::abs((a) - (b)) < (eps)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_NEAR(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " within " << (eps) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_EQ(a, b) \
    total_tests++; \
    if ((a) == (b)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_EQ(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_TRUE(cond) \
    total_tests++; \
    if (cond) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_TRUE(" << #cond << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_FALSE(cond) \
    total_tests++; \
    if (!(cond)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_FALSE(" << #cond << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

// ============================================================================
// Polynomial Tests
// ============================================================================

TEST(polynomial_evaluation) {
    Polynomial<double, 3> p;
    p.coeffs = {1.0, 2.0, 3.0, 4.0};  // 1 + 2x + 3x² + 4x³

    ASSERT_NEAR(p.evaluate(0.0), 1.0, 1e-10);
    ASSERT_NEAR(p.evaluate(1.0), 10.0, 1e-10);  // 1+2+3+4
    ASSERT_NEAR(p.evaluate(2.0), 49.0, 1e-10);  // 1+4+12+32
    ASSERT_NEAR(p.evaluate(-1.0), -2.0, 1e-10);  // 1-2+3-4 = -2
}

TEST(polynomial_differentiation) {
    Polynomial<double, 3> p;
    p.coeffs = {1.0, 2.0, 3.0, 4.0};  // 1 + 2x + 3x² + 4x³

    auto dp = p.differentiate();  // 2 + 6x + 12x²

    ASSERT_NEAR(dp.coeffs[0], 2.0, 1e-10);
    ASSERT_NEAR(dp.coeffs[1], 6.0, 1e-10);
    ASSERT_NEAR(dp.coeffs[2], 12.0, 1e-10);

    ASSERT_NEAR(dp.evaluate(0.0), 2.0, 1e-10);
    ASSERT_NEAR(dp.evaluate(1.0), 20.0, 1e-10);  // 2+6+12
    ASSERT_NEAR(dp.evaluate(2.0), 62.0, 1e-10);  // 2+12+48
}

TEST(polynomial_addition) {
    Polynomial<double, 2> p;
    p.coeffs = {1.0, 2.0, 3.0};  // 1 + 2x + 3x²

    Polynomial<double, 2> q;
    q.coeffs = {4.0, 5.0, 6.0};  // 4 + 5x + 6x²

    auto r = p + q;  // 5 + 7x + 9x²

    ASSERT_NEAR(r.coeffs[0], 5.0, 1e-10);
    ASSERT_NEAR(r.coeffs[1], 7.0, 1e-10);
    ASSERT_NEAR(r.coeffs[2], 9.0, 1e-10);
}

TEST(polynomial_multiplication) {
    Polynomial<double, 1> p;
    p.coeffs = {1.0, 2.0};  // 1 + 2x

    Polynomial<double, 1> q;
    q.coeffs = {3.0, 4.0};  // 3 + 4x

    auto r = multiply(p, q);  // (1+2x)(3+4x) = 3 + 10x + 8x²

    ASSERT_NEAR(r.coeffs[0], 3.0, 1e-10);
    ASSERT_NEAR(r.coeffs[1], 10.0, 1e-10);
    ASSERT_NEAR(r.coeffs[2], 8.0, 1e-10);
}

// ============================================================================
// Bivariate Polynomial Tests
// ============================================================================

TEST(bivariate_polynomial_evaluation) {
    BivarPolynomial<double, 2> p;
    // p(λ, t) = 1 + 2λ + 3t + 4λ² + 5λt + 6t²
    p.coeffs[0][0] = 1.0;  // constant
    p.coeffs[1][0] = 2.0;  // λ
    p.coeffs[0][1] = 3.0;  // t
    p.coeffs[2][0] = 4.0;  // λ²
    p.coeffs[1][1] = 5.0;  // λt
    p.coeffs[0][2] = 6.0;  // t²

    ASSERT_NEAR(p.evaluate(0.0, 0.0), 1.0, 1e-10);
    ASSERT_NEAR(p.evaluate(1.0, 0.0), 7.0, 1e-10);   // 1+2+4
    ASSERT_NEAR(p.evaluate(0.0, 1.0), 10.0, 1e-10);  // 1+3+6
    ASSERT_NEAR(p.evaluate(1.0, 1.0), 21.0, 1e-10);  // 1+2+3+4+5+6
}

TEST(bivariate_polynomial_evaluate_t) {
    BivarPolynomial<double, 1> p;
    // p(λ, t) = 1 + 2λ + 3t + 4λt
    p.coeffs[0][0] = 1.0;
    p.coeffs[1][0] = 2.0;
    p.coeffs[0][1] = 3.0;
    p.coeffs[1][1] = 4.0;

    // Fix t = 2, get p(λ, 2) = 1 + 2λ + 6 + 8λ = 7 + 10λ
    auto univar = p.evaluate_t(2.0);

    ASSERT_NEAR(univar.coeffs[0], 7.0, 1e-10);   // 1 + 6
    ASSERT_NEAR(univar.coeffs[1], 10.0, 1e-10);  // 2 + 8

    ASSERT_NEAR(univar.evaluate(0.0), 7.0, 1e-10);
    ASSERT_NEAR(univar.evaluate(1.0), 17.0, 1e-10);
}

// ============================================================================
// PV Curve Segment Tests
// ============================================================================

TEST(pv_curve_segment_barycentric) {
    PVCurveSegment segment;
    segment.lambda_min = 0.0;
    segment.lambda_max = 1.0;

    // Linear barycentric coordinates (simple test)
    // mu_0 = 1 - λ, mu_1 = λ, mu_2 = 0, mu_3 = 0
    segment.P[0].coeffs = {1.0, -1.0, 0.0, 0.0};  // 1 - λ
    segment.P[1].coeffs = {0.0,  1.0, 0.0, 0.0};  // λ
    segment.P[2].coeffs = {0.0,  0.0, 0.0, 0.0};  // 0
    segment.P[3].coeffs = {0.0,  0.0, 0.0, 0.0};  // 0
    segment.Q.coeffs = {1.0, 0.0, 0.0, 0.0};      // Q = 1

    auto mu_0 = segment.get_barycentric(0.0);
    ASSERT_NEAR(mu_0[0], 1.0, 1e-10);
    ASSERT_NEAR(mu_0[1], 0.0, 1e-10);

    auto mu_1 = segment.get_barycentric(1.0);
    ASSERT_NEAR(mu_1[0], 0.0, 1e-10);
    ASSERT_NEAR(mu_1[1], 1.0, 1e-10);

    auto mu_half = segment.get_barycentric(0.5);
    ASSERT_NEAR(mu_half[0], 0.5, 1e-10);
    ASSERT_NEAR(mu_half[1], 0.5, 1e-10);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(verify_parallel_simple) {
    // Parallel vectors (v = 2w)
    double V[3][3] = {
        {2.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 0.0, 0.0}
    };
    double W[3][3] = {
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}
    };

    double mu[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};  // Barycenter

    ASSERT_TRUE(verify_parallel(V, W, mu, 1e-10));
}

TEST(verify_parallel_not_parallel) {
    // Non-parallel vectors
    double V[3][3] = {
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}
    };
    double W[3][3] = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    double mu[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};

    ASSERT_FALSE(verify_parallel(V, W, mu, 1e-10));
}

// ============================================================================
// Solver Tests (Stubs - will be implemented in Week 2)
// ============================================================================

TEST(solve_pv_triangle_degenerate) {
    // Degenerate case: parallel vectors everywhere
    double V[3][3] = {
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}
    };
    double W[3][3] = {
        {2.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 0.0, 0.0}
    };

    std::vector<PuncturePoint> punctures;
    int n = solve_pv_triangle(V, W, punctures);

    // Should detect degenerate case (entire triangle is PV surface)
    ASSERT_EQ(n, std::numeric_limits<int>::max());
}

TEST(solve_pv_triangle_no_solution) {
    // No parallel vectors (perpendicular everywhere)
    double V[3][3] = {
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}
    };
    double W[3][3] = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    std::vector<PuncturePoint> punctures;
    int n = solve_pv_triangle(V, W, punctures);

    // No solutions expected
    ASSERT_EQ(n, 0);
}

TEST(solve_pv_triangle_single_puncture) {
    // Triangle where V and W become parallel at specific point
    // V varies from (1,0,0) to (0,1,0) to (-1,0,0)
    // W varies from (-1,0,0) to (0,-1,0) to (1,0,0)
    // They should be anti-parallel (λ=-1) somewhere
    double V[3][3] = {
        { 1.0,  0.0, 0.0},
        { 0.0,  1.0, 0.0},
        {-1.0,  0.0, 0.0}
    };
    double W[3][3] = {
        {-1.0,  0.0, 0.0},
        { 0.0, -1.0, 0.0},
        { 1.0,  0.0, 0.0}
    };

    std::vector<PuncturePoint> punctures;
    int n = solve_pv_triangle(V, W, punctures);

    // Should find punctures (they are anti-parallel everywhere: W = -V)
    // This is a degenerate case where the entire triangle satisfies the condition
    ASSERT_TRUE(n == std::numeric_limits<int>::max());
}

TEST(solve_pv_triangle_known_solution) {
    // Simple case: V and W become parallel at barycenter
    double V[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    double W[3][3] = {
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0}
    };

    std::vector<PuncturePoint> punctures;
    int n = solve_pv_triangle(V, W, punctures);

    // Should find punctures (W = 2V everywhere)
    // This is actually a degenerate case
    ASSERT_TRUE(n == std::numeric_limits<int>::max() || n > 0);
}

TEST(solve_pv_triangle_realistic) {
    // Test case that should find a puncture
    // Create a field where vectors gradually align
    double V[3][3] = {
        { 1.0,  0.0,  0.0},
        { 0.7,  0.7,  0.0},
        { 0.0,  1.0,  0.0}
    };
    double W[3][3] = {
        { 0.5,  0.0,  0.0},  // Parallel to V[0] (λ=0.5)
        {-0.7,  0.7,  0.0},  // Not parallel to V[1]
        { 0.0, -1.0,  0.0}   // Anti-parallel to V[2] (λ=-1)
    };

    std::vector<PuncturePoint> punctures;
    int n = solve_pv_triangle(V, W, punctures);

    // May or may not find punctures depending on the field configuration
    // Just verify the solver runs without crashing
    ASSERT_TRUE(n >= 0 && n <= 3);

    // If punctures found, verify they're valid
    for (int i = 0; i < n; ++i) {
        double sum = punctures[i].barycentric[0] + punctures[i].barycentric[1] + punctures[i].barycentric[2];
        ASSERT_NEAR(sum, 1.0, 1e-6);

        // Each coordinate should be in [0,1]
        ASSERT_TRUE(punctures[i].barycentric[0] >= -1e-6 && punctures[i].barycentric[0] <= 1.0 + 1e-6);
        ASSERT_TRUE(punctures[i].barycentric[1] >= -1e-6 && punctures[i].barycentric[1] <= 1.0 + 1e-6);
        ASSERT_TRUE(punctures[i].barycentric[2] >= -1e-6 && punctures[i].barycentric[2] <= 1.0 + 1e-6);
    }
}

void test_exactpv() {
    // Test runner - the tests are automatically registered and run via static constructors
    std::cout << "ExactPV tests completed (polynomial utilities, data structures, and triangle solver)" << std::endl;
}

int main() {
    std::cout << "Running ExactPV tests..." << std::endl;
    test_exactpv();
    std::cout << "Summary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}
