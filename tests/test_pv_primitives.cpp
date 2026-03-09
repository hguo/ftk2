#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <iostream>
#include <cmath>

using namespace ftk2;

static int total_tests = 0;
static int passed_tests = 0;
static int failed_tests = 0;

#define ASSERT_EQ(a, b) \
    total_tests++; \
    if ((a) == (b)) { \
        passed_tests++; \
    } else { \
        failed_tests++; \
        std::cerr << "FAILED: " << #a << " == " << #b \
                  << " got " << (long long)(a) << ", expected " << (long long)(b) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_TRUE(cond) \
    total_tests++; \
    if (cond) { \
        passed_tests++; \
    } else { \
        failed_tests++; \
        std::cerr << "FAILED: " << #cond \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

// ============================================================================
// prem_i128 tests
// ============================================================================

void test_prem_basic() {
    std::cout << "  prem_basic" << std::endl;
    // f = x^2 - 1, g = x - 2
    // prem(f, g) = f(2) = 3, but content_reduce divides by gcd → 3/3=1? No, gcd(3)=3, r=1
    // Actually gcd of a single element is |element|, so content_reduce(3) → 1 if it divides by 3
    // Let me just check what we get.
    __int128 f[8] = {-1, 0, 1};
    __int128 g[8] = {-2, 1};
    __int128 r[8] = {};
    int actual_exp;
    int dr = prem_i128(f, 2, g, 1, r, &actual_exp);
    ASSERT_EQ(dr, 0);
    ASSERT_TRUE(r[0] > 0);  // positive (f(2) = 3 > 0)
    ASSERT_EQ(actual_exp, 2);
}

void test_prem_zero_skip() {
    std::cout << "  prem_zero_skip (actual_exp tracking)" << std::endl;

    // Case where zero-skip causes actual_exp < delta:
    // f = x^2 + 1, g = x
    // prem(f, g): delta = 2-1+1 = 2
    // r = [1, 0, 1], dr=2
    // Iter 1: r[2]=1 ≠ 0, n_muls=1
    //   r = g[1]*[1,0,1] - 1*[0,0,1] = [1,0,1] - [0,0,1] = [1,0,0]
    //   dr-- → dr=1, r[1]=0 → dr-- → dr=0
    //   stop (dr=0 < dg=1)
    // actual_exp = 1 < delta = 2
    __int128 f2[8] = {1, 0, 1};
    __int128 g2[8] = {0, 1};
    __int128 r2[8] = {};
    int exp2;
    int dr2 = prem_i128(f2, 2, g2, 1, r2, &exp2);
    ASSERT_EQ(dr2, 0);
    ASSERT_EQ(r2[0], (__int128)1);
    ASSERT_EQ(exp2, 1);  // actual_exp=1 < delta=2!
}

void test_prem_seed2520() {
    std::cout << "  prem_seed2520 (original bug case)" << std::endl;
    // P0 at root of P1 should be positive.
    // signs_at_roots_i128 uses prem internally.
    __int128 P1[8] = {-1, -1, -3, 2};
    __int128 P0[8] = {-2, -2, -3, 3};
    int signs[3] = {};
    int nr = signs_at_roots_i128(P1, 3, P0, 3, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 1);  // P0 > 0 at the unique real root of P1
}

// ============================================================================
// signs_at_roots_i128 tests
// ============================================================================

void test_signs_linear() {
    std::cout << "  signs_linear" << std::endl;
    // f = x - 2 (root at x=2), g = x + 1 → g(2) = 3 > 0
    __int128 f[8] = {-2, 1};
    __int128 g[8] = {1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 1, g, 1, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 1);

    // g = -x + 1 → g(2) = -1 < 0
    __int128 g2[8] = {1, -1};
    int signs2[3] = {};
    nr = signs_at_roots_i128(f, 1, g2, 1, signs2, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs2[0], -1);

    // g = x - 2 → g(2) = 0
    __int128 g3[8] = {-2, 1};
    int signs3[3] = {};
    nr = signs_at_roots_i128(f, 1, g3, 1, signs3, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs3[0], 0);
}

void test_signs_quadratic() {
    std::cout << "  signs_quadratic" << std::endl;
    // f = (x-2)(x-3) = x^2-5x+6, roots at 2, 3
    __int128 f[8] = {6, -5, 1};

    // g = x - 1 → g(2)=1>0, g(3)=2>0
    __int128 g[8] = {-1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 2, g, 1, signs, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs[0], 1);
    ASSERT_EQ(signs[1], 1);

    // g = 2x-5 → g(2)=-1<0, g(3)=1>0
    __int128 g2[8] = {-5, 2};
    int signs2[3] = {};
    nr = signs_at_roots_i128(f, 2, g2, 1, signs2, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs2[0], -1);
    ASSERT_EQ(signs2[1], 1);

    // g = f → g(2)=g(3)=0
    int signs3[3] = {};
    nr = signs_at_roots_i128(f, 2, f, 2, signs3, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs3[0], 0);
    ASSERT_EQ(signs3[1], 0);
}

void test_signs_quadratic_no_real_roots() {
    std::cout << "  signs_quadratic_no_real_roots" << std::endl;
    __int128 f[8] = {1, 0, 1};  // x^2+1
    __int128 g[8] = {1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 2, g, 1, signs, 3);
    ASSERT_EQ(nr, 0);
}

void test_signs_quadratic_double_root() {
    std::cout << "  signs_quadratic_double_root" << std::endl;
    // f = (x-3)^2 = x^2-6x+9, double root at 3
    // g = x-1 → g(3) = 2 > 0
    __int128 f[8] = {9, -6, 1};
    __int128 g[8] = {-1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 2, g, 1, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 1);
}

void test_signs_cubic_3_roots() {
    std::cout << "  signs_cubic_3_roots (disc>0)" << std::endl;
    // f = (x-1)(x-2)(x-3) = x^3-6x^2+11x-6
    __int128 f[8] = {-6, 11, -6, 1};

    // g = x → g(1)=1, g(2)=2, g(3)=3
    __int128 g[8] = {0, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 1, signs, 3);
    ASSERT_EQ(nr, 3);
    ASSERT_EQ(signs[0], 1);
    ASSERT_EQ(signs[1], 1);
    ASSERT_EQ(signs[2], 1);

    // g = 2x-5 → g(1)=-3, g(2)=-1, g(3)=1
    __int128 g2[8] = {-5, 2};
    int signs2[3] = {};
    nr = signs_at_roots_i128(f, 3, g2, 1, signs2, 3);
    ASSERT_EQ(nr, 3);
    ASSERT_EQ(signs2[0], -1);
    ASSERT_EQ(signs2[1], -1);
    ASSERT_EQ(signs2[2], 1);

    // g = x-2 → g(1)=-1, g(2)=0, g(3)=1
    __int128 g3[8] = {-2, 1};
    int signs3[3] = {};
    nr = signs_at_roots_i128(f, 3, g3, 1, signs3, 3);
    ASSERT_EQ(nr, 3);
    ASSERT_EQ(signs3[0], -1);
    ASSERT_EQ(signs3[1], 0);
    ASSERT_EQ(signs3[2], 1);
}

void test_signs_cubic_1_root() {
    std::cout << "  signs_cubic_1_root (disc<0)" << std::endl;
    // f = x^3+x+2 = (x+1)(x^2-x+2), disc<0, 1 real root at -1
    __int128 f[8] = {2, 1, 0, 1};

    // g = x+3 → g(-1) = 2 > 0
    __int128 g[8] = {3, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 1, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 1);

    // g = x → g(-1) = -1 < 0
    __int128 g2[8] = {0, 1};
    int signs2[3] = {};
    nr = signs_at_roots_i128(f, 3, g2, 1, signs2, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs2[0], -1);
}

void test_signs_cubic_shared_root() {
    std::cout << "  signs_cubic_shared_root" << std::endl;
    // f = (x+1)(x^2+1), disc<0, 1 real root at -1
    // g = (x+1)(x-2) = x^2-x-2, g(-1) = 0
    __int128 f[8] = {1, 1, 1, 1};
    __int128 g[8] = {-2, -1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 2, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 0);
}

void test_signs_cubic_disc0() {
    std::cout << "  signs_cubic_disc0 (double root)" << std::endl;
    // f = (x-1)^2(x+2) = x^3-3x+2, disc=0
    // 2 distinct roots: -2, 1
    __int128 f[8] = {2, -3, 0, 1};

    // g = x → g(-2)=-2, g(1)=1
    __int128 g[8] = {0, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 1, signs, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs[0], -1);
    ASSERT_EQ(signs[1], 1);
}

void test_signs_constant_g() {
    std::cout << "  signs_constant_g" << std::endl;
    __int128 f[8] = {-1, 1};  // x-1

    __int128 g1[8] = {5};  // g=5 > 0
    int s1[3] = {};
    int nr = signs_at_roots_i128(f, 1, g1, 0, s1, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(s1[0], 1);

    __int128 g2[8] = {-3};  // g=-3 < 0
    int s2[3] = {};
    nr = signs_at_roots_i128(f, 1, g2, 0, s2, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(s2[0], -1);

    __int128 g3[8] = {0};  // g=0
    int s3[3] = {};
    nr = signs_at_roots_i128(f, 1, g3, 0, s3, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(s3[0], 0);
}

void test_signs_g_higher_degree() {
    std::cout << "  signs_g_higher_degree_than_f" << std::endl;
    // f = x-1, g = x^2-4 → g(1) = -3
    __int128 f[8] = {-1, 1};
    __int128 g[8] = {-4, 0, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 1, g, 2, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], -1);
}

void test_signs_negative_leading_coeff() {
    std::cout << "  signs_negative_leading_coeff" << std::endl;
    // f = -2x+4 (root at 2, lc=-2), g = x-1 → g(2)=1>0
    __int128 f[8] = {4, -2};
    __int128 g[8] = {-1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 1, g, 1, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 1);

    // g = x-3 → g(2) = -1 < 0
    __int128 g2[8] = {-3, 1};
    int signs2[3] = {};
    nr = signs_at_roots_i128(f, 1, g2, 1, signs2, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs2[0], -1);
}

void test_signs_seed91502_edge_product() {
    std::cout << "  signs_seed91502_edge_product" << std::endl;
    // From seed 91502: P[1] = [15,9,15,9] (cubic, disc<0, 1 real root at -5/3)
    // prod = P'[1]·P'[3] = [189, 522, -36, -1134, -729]
    // prod(-5/3) = -1156 < 0
    __int128 P1[8] = {15, 9, 15, 9};
    __int128 prod[8] = {189, 522, -36, -1134, -729};
    int signs[3] = {};
    int nr = signs_at_roots_i128(P1, 3, prod, 4, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], -1);  // prod(-5/3) = -1156 < 0
}

// ============================================================================
// resultant_sign_i128 tests
// ============================================================================

void test_resultant_shared_root() {
    std::cout << "  resultant_shared_root" << std::endl;
    // f = (x-1)(x+2) = x^2+x-2, g = (x-1)(x-3) = x^2-4x+3
    __int128 f[8] = {-2, 1, 1};
    __int128 g[8] = {3, -4, 1};
    int res = resultant_sign_i128(f, 2, g, 2);
    ASSERT_EQ(res, 0);
}

void test_resultant_no_shared_root() {
    std::cout << "  resultant_no_shared_root" << std::endl;
    __int128 f[8] = {-1, 1};  // x-1
    __int128 g[8] = {-2, 1};  // x-2
    int res = resultant_sign_i128(f, 1, g, 1);
    ASSERT_TRUE(res != 0);
}

void test_resultant_cubic_shared() {
    std::cout << "  resultant_cubic_shared (seed 91502)" << std::endl;
    // P[1] = 3(3λ+5)(λ^2+1) = [15,9,15,9]
    // P[3] = -(3λ+5)(3λ^2-3λ-2) = [10,21,-6,-9]
    // Share root at -5/3. Resultant = 0.
    __int128 P1[8] = {15, 9, 15, 9};
    __int128 P3[8] = {10, 21, -6, -9};
    int res = resultant_sign_i128(P1, 3, P3, 3);
    ASSERT_EQ(res, 0);
}

void test_resultant_mixed_degree() {
    std::cout << "  resultant_mixed_degree" << std::endl;
    // cubic sharing root with linear
    __int128 f[8] = {-6, 11, -6, 1};  // (x-1)(x-2)(x-3)
    __int128 g[8] = {-2, 1};          // x-2
    int res = resultant_sign_i128(f, 3, g, 1);
    ASSERT_EQ(res, 0);

    __int128 g2[8] = {-4, 1};  // x-4, no shared root
    res = resultant_sign_i128(f, 3, g2, 1);
    ASSERT_TRUE(res != 0);
}

// ============================================================================
// poly_gcd_full_i128 tests
// ============================================================================

void test_gcd_shared_factor() {
    std::cout << "  gcd_shared_factor" << std::endl;
    __int128 f[8] = {-2, 1, 1};  // (x-1)(x+2)
    __int128 g[8] = {3, -4, 1};  // (x-1)(x-3)
    __int128 h[8] = {};
    int dh = poly_gcd_full_i128(f, 2, g, 2, h);
    ASSERT_EQ(dh, 1);
    ASSERT_TRUE(h[1] != 0);
    ASSERT_EQ(h[0] + h[1], (__int128)0);  // h(1) = 0
}

void test_gcd_no_shared_factor() {
    std::cout << "  gcd_no_shared_factor" << std::endl;
    __int128 f[8] = {1, 0, 1};  // x^2+1
    __int128 g[8] = {-1, 1};    // x-1
    __int128 h[8] = {};
    int dh = poly_gcd_full_i128(f, 2, g, 1, h);
    ASSERT_EQ(dh, 0);
}

void test_gcd_four_polys() {
    std::cout << "  gcd_four_polys" << std::endl;
    // 4 polys sharing factor (x-1)
    __int128 P[4][8] = {
        {-1, 1, -1, 1, 0,0,0,0},   // (x-1)(x^2+1)
        {-2, 1, 1, 0, 0,0,0,0},    // (x-1)(x+2)
        {1, -3, 2, 0, 0,0,0,0},    // (x-1)(2x-1)
        {-3, 2, 1, 0, 0,0,0,0}     // (x-1)(x+3)
    };
    int degP[4] = {3, 2, 2, 2};

    __int128 h[8] = {};
    int dh = poly_gcd_full_i128(P[0], degP[0], P[1], degP[1], h);
    for (int k = 2; k < 4; k++) {
        __int128 h2[8] = {};
        int dh2 = poly_gcd_full_i128(h, dh, P[k], degP[k], h2);
        dh = dh2;
        for (int i = 0; i < 8; i++) h[i] = h2[i];
    }

    ASSERT_EQ(dh, 1);
    ASSERT_TRUE(h[1] != 0);
    ASSERT_EQ(h[0] + h[1], (__int128)0);
}

// ============================================================================
// solve_pv_tet_v2 regression tests
// ============================================================================

void test_solve_v2_seed91502() {
    std::cout << "  solve_v2_seed91502 (bare T1 regression)" << std::endl;
    // P[1] and P[3] share root at -5/3. Pass-through (P'·P'<0) → exclude.
    // Expected: T0 (0 valid punctures)
    __int128 Q[4], P[4][4];
    int V[4][3] = {{-2,3,0},{-2,1,-2},{-1,-1,0},{-3,0,3}};
    int W[4][3] = {{0,0,-3},{1,3,2},{1,0,1},{-3,3,0}};
    compute_tet_QP_i128(V, W, Q, P);

    // Verify P[1] and P[3] share a root
    int degP1 = effective_degree_i128(P[1], 3);
    int degP3 = effective_degree_i128(P[3], 3);
    int res13 = resultant_sign_i128(P[1], degP1, P[3], degP3);
    ASSERT_EQ(res13, 0);

    ExactPV2Result result = solve_pv_tet_v2(Q, P);
    // Edge puncture should be excluded (pass-through). T=0.
    ASSERT_EQ(result.n_punctures, 0);
}

void test_solve_v2_seed4984() {
    std::cout << "  solve_v2_seed4984 (Cw degree regression)" << std::endl;
    // Q degree 2 (Q[3]=0). 1 puncture, Cw1 waypoint.
    __int128 Q[4], P[4][4];
    int V[4][3] = {{-3,0,3},{-2,3,3},{-3,0,2},{3,-2,-2}};
    int W[4][3] = {{0,-1,-1},{0,0,2},{0,1,0},{0,-1,0}};
    compute_tet_QP_i128(V, W, Q, P);

    ASSERT_EQ(Q[3], (__int128)0);

    ExactPV2Result result = solve_pv_tet_v2(Q, P);
    // 1 face puncture + 1 Cw1 edge at ∞ (now included), paired cross-infinity
    ASSERT_EQ(result.n_punctures, 2);
    ASSERT_EQ(result.n_pairs, 1);
}

void test_solve_v2_seed6247() {
    std::cout << "  solve_v2_seed6247 (edge + tangency pass-through)" << std::endl;
    // Edge puncture at λ=1 where P[0]∩P[2]=0 and P'[2](1)=0 (tangency).
    // P''[2]·Q < 0 → isolated tangency → exclude.
    // Without fix: T=3 (bare odd). With fix: T=2 (even).
    __int128 Q[4], P[4][4];
    int V[4][3] = {{2,1,2},{3,0,-1},{-2,3,2},{-3,3,1}};
    int W[4][3] = {{3,-2,3},{3,0,2},{3,3,3},{-3,3,-1}};
    compute_tet_QP_i128(V, W, Q, P);

    ExactPV2Result result = solve_pv_tet_v2(Q, P);
    // Edge puncture excluded → T=2 (even)
    ASSERT_TRUE(result.n_punctures % 2 == 0);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== prem_i128 ===" << std::endl;
    test_prem_basic();
    test_prem_zero_skip();
    test_prem_seed2520();

    std::cout << "\n=== signs_at_roots_i128 ===" << std::endl;
    test_signs_linear();
    test_signs_quadratic();
    test_signs_quadratic_no_real_roots();
    test_signs_quadratic_double_root();
    test_signs_cubic_3_roots();
    test_signs_cubic_1_root();
    test_signs_cubic_shared_root();
    test_signs_cubic_disc0();
    test_signs_constant_g();
    test_signs_g_higher_degree();
    test_signs_negative_leading_coeff();
    test_signs_seed91502_edge_product();

    std::cout << "\n=== resultant_sign_i128 ===" << std::endl;
    test_resultant_shared_root();
    test_resultant_no_shared_root();
    test_resultant_cubic_shared();
    test_resultant_mixed_degree();

    std::cout << "\n=== poly_gcd_full_i128 ===" << std::endl;
    test_gcd_shared_factor();
    test_gcd_no_shared_factor();
    test_gcd_four_polys();

    std::cout << "\n=== solve_pv_tet_v2 regressions ===" << std::endl;
    test_solve_v2_seed91502();
    test_solve_v2_seed4984();
    test_solve_v2_seed6247();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total: " << total_tests << ", Passed: " << passed_tests
              << ", Failed: " << failed_tests << std::endl;
    return failed_tests > 0 ? 1 : 0;
}
