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
// Paper cases (figures_v15): 30 approved cases with hardcoded V/W
// ============================================================================

void test_paper_cases() {
    std::cout << "  30 approved paper cases (figures_v15)" << std::endl;
    // Macro: compute Q/P, run v2 solver, check n_punctures and n_pairs
    #define RUN(V, W, seed, exp_np, exp_npairs) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << seed \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
    } while(0)

    // T4_(1,3)_Q3+_Cw
    {  int V[4][3] = {{-1,3,-17},{-6,11,-3},{3,6,-14},{11,7,-1}};
       int W[4][3] = {{18,16,7},{3,-9,12},{-4,0,-8},{4,14,0}};
       RUN(V, W, 414, 4, 2); }
    // T6_(2,2,2)_Q3+
    {  int V[4][3] = {{-3,5,-9},{-18,6,14},{4,0,-4},{11,0,1}};
       int W[4][3] = {{7,-11,0},{16,-4,6},{-3,-17,-15},{-18,17,19}};
       RUN(V, W, 515, 6, 3); }
    // T2_Q2
    {  int V[4][3] = {{12,-3,-8},{18,-20,-19},{6,4,2},{8,20,7}};
       int W[4][3] = {{-17,19,-14},{-17,13,9},{-17,-13,10},{-17,1,8}};
       RUN(V, W, 898, 2, 1); }
    // T6_(2,2,2)_Q3+
    {  int V[4][3] = {{-1,-19,-10},{15,20,10},{-8,-13,-3},{15,-3,-6}};
       int W[4][3] = {{16,-20,13},{-18,5,-4},{5,7,-7},{-16,2,-12}};
       RUN(V, W, 1570, 6, 3); }
    // T2_Q3+
    {  int V[4][3] = {{15,8,20},{3,12,3},{12,-20,-15},{-16,-5,-18}};
       int W[4][3] = {{-15,13,2},{8,13,20},{7,18,8},{-16,13,-3}};
       RUN(V, W, 2098, 2, 1); }
    // T2_(1,1)_Q3+_Cw
    {  int V[4][3] = {{-13,8,-13},{-16,-2,20},{-13,8,-13},{2,-8,8}};
       int W[4][3] = {{-10,9,7},{-4,4,-3},{-14,2,6},{12,-5,-4}};
       RUN(V, W, 3392, 2, 1); }
    // T6_(3,3)_Q3+_Cw
    {  int V[4][3] = {{14,12,-15},{14,-7,1},{-8,-15,-7},{1,5,-19}};
       int W[4][3] = {{3,-1,-4},{-11,4,20},{9,14,-9},{-10,-15,-18}};
       RUN(V, W, 4170, 6, 3); }
    // T2_Q3-_Cv1
    {  int V[4][3] = {{-14,-4,6},{16,-3,7},{3,0,20},{-3,0,-20}};
       int W[4][3] = {{-9,7,6},{2,-14,1},{12,6,17},{1,-7,5}};
       RUN(V, W, 4423, 2, 1); }
    // T6_(2,4)_Q3+
    {  int V[4][3] = {{-5,5,-15},{-1,8,5},{-1,8,-1},{-1,-2,6}};
       int W[4][3] = {{-1,-17,11},{13,-20,20},{12,3,12},{-17,16,-19}};
       RUN(V, W, 4988, 6, 3); }
    // T6_Q3+
    {  int V[4][3] = {{4,-20,14},{-1,2,-17},{-5,-18,-20},{12,18,-13}};
       int W[4][3] = {{3,11,-18},{-9,9,-19},{4,-16,3},{19,0,4}};
       RUN(V, W, 5629, 6, 3); }
    // T4_(2,2)_Q3+
    {  int V[4][3] = {{-19,-2,13},{16,13,-14},{-8,5,18},{-14,-6,-14}};
       int W[4][3] = {{-13,12,10},{18,2,17},{-14,-8,3},{-17,-9,-5}};
       RUN(V, W, 6737, 4, 2); }
    // T2_Q3-_Cv
    {  int V[4][3] = {{-15,17,3},{7,-13,17},{3,-14,-2},{13,10,-16}};
       int W[4][3] = {{-19,-17,-8},{11,19,7},{13,7,0},{4,14,0}};
       RUN(V, W, 8482, 2, 1); }
    // T4_(2,2)_Q3+
    {  int V[4][3] = {{-11,9,-5},{-8,4,-15},{-8,4,-15},{-9,-18,11}};
       int W[4][3] = {{6,1,8},{-5,15,18},{-8,-11,-20},{-11,-16,-19}};
       RUN(V, W, 10183, 4, 2); }
    // T2_Q3+_Cv
    {  int V[4][3] = {{-19,-2,13},{2,-13,20},{-17,-10,-13},{12,10,-6}};
       int W[4][3] = {{19,4,-8},{17,1,19},{4,19,-11},{-9,1,-3}};
       RUN(V, W, 10307, 2, 1); }
    // T2_Q3+
    {  int V[4][3] = {{-6,-20,7},{6,-1,0},{-9,-16,-10},{5,-13,2}};
       int W[4][3] = {{-5,-13,7},{0,7,-18},{11,-3,12},{-17,-14,-3}};
       RUN(V, W, 10310, 2, 1); }
    // T2_Q3-
    {  int V[4][3] = {{4,-20,-3},{-17,-7,8},{-15,-7,-20},{1,-13,-18}};
       int W[4][3] = {{12,-11,13},{6,17,-15},{4,2,13},{7,-16,19}};
       RUN(V, W, 10312, 2, 1); }
    // T6_Q3-
    {  int V[4][3] = {{5,-6,-9},{18,-2,6},{-14,5,2},{14,14,-19}};
       int W[4][3] = {{14,1,-15},{6,-6,-10},{-2,-5,10},{0,4,1}};
       RUN(V, W, 10322, 6, 3); }
    // T4_(1,1,2)_Q3+_Cw
    {  int V[4][3] = {{-17,6,-7},{-9,2,-19},{11,20,7},{9,-10,1}};
       int W[4][3] = {{11,-19,17},{16,-5,0},{-13,-6,-12},{-19,14,0}};
       RUN(V, W, 10417, 4, 2); }
    // T4_Q3-
    {  int V[4][3] = {{-4,-12,-9},{-5,14,6},{-18,14,10},{2,8,9}};
       int W[4][3] = {{-17,5,-13},{-1,1,4},{-6,17,7},{18,-1,0}};
       RUN(V, W, 10420, 4, 2); }
    // T8_(2,2,4)_Q3+
    {  int V[4][3] = {{-6,-10,-5},{-17,9,19},{20,8,20},{2,5,11}};
       int W[4][3] = {{13,19,-20},{8,6,-2},{-5,-4,0},{-4,2,19}};
       RUN(V, W, 10553, 8, 4); }
    // T4_(2,2)_Q2
    {  int V[4][3] = {{-2,-15,-4},{-11,5,8},{13,-8,5},{-9,7,-3}};
       int W[4][3] = {{-7,7,3},{4,-4,-8},{3,3,17},{-10,11,10}};
       RUN(V, W, 11834, 4, 2); }
    // T4_Q3+
    {  int V[4][3] = {{7,-2,-8},{-1,19,10},{6,-10,5},{-12,20,13}};
       int W[4][3] = {{20,-12,-20},{-8,-14,12},{13,14,12},{14,12,-7}};
       RUN(V, W, 12369, 4, 2); }
    // T4_Q3-
    {  int V[4][3] = {{14,-14,19},{8,-4,3},{-18,20,4},{-5,-13,9}};
       int W[4][3] = {{-17,-19,5},{-16,0,-7},{19,2,-17},{-18,19,18}};
       RUN(V, W, 14299, 4, 2); }
    // T4_Q3-_Cw2
    {  int V[4][3] = {{-12,9,13},{0,-4,-6},{14,-14,-3},{9,-17,12}};
       int W[4][3] = {{-6,-5,7},{18,0,-12},{19,13,18},{-3,5,-1}};
       RUN(V, W, 15587, 4, 2); }
    // T8_(2,6)_Q3-
    {  int V[4][3] = {{-10,8,7},{-8,9,19},{13,-8,-12},{-14,15,14}};
       int W[4][3] = {{17,12,-15},{1,-14,-8},{-2,0,-15},{-7,-10,9}};
       RUN(V, W, 25710, 8, 4); }
    // T8_Q3-
    {  int V[4][3] = {{-20,-9,6},{18,19,15},{5,-5,-17},{-2,13,-3}};
       int W[4][3] = {{19,1,-3},{-7,12,16},{10,17,19},{-12,-4,-1}};
       RUN(V, W, 30810, 8, 4); }
    // T4_Q2-
    {  int V[4][3] = {{1,-15,-2},{13,11,9},{12,16,-9},{-7,9,13}};
       int W[4][3] = {{9,0,19},{13,12,8},{8,2,13},{-3,-8,3}};
       RUN(V, W, 58615, 4, 2); }
    // T2_Q3+_D00
    {  int V[4][3] = {{-14,2,-6},{-12,19,-18},{-20,17,-4},{11,-9,-9}};
       int W[4][3] = {{-8,1,18},{19,-6,4},{-15,-6,-11},{-11,9,9}};
       RUN(V, W, 65683, 2, 1); }
    // T10_Q3-
    {  int V[4][3] = {{6,-8,-6},{3,17,10},{17,-10,17},{-6,-18,-5}};
       int W[4][3] = {{-8,10,-5},{-9,-18,15},{18,-15,9},{-9,-7,5}};
       RUN(V, W, 1783954, 10, 5); }
    // T2_(1,1)_Q3+_Cw
    {  int V[4][3] = {{-9,7,-18},{-2,19,13},{20,3,7},{-11,5,1}};
       int W[4][3] = {{-3,9,-12},{-16,7,-8},{9,-9,9},{4,13,-7}};
       RUN(V, W, 136281707, 2, 1); }

    #undef RUN
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

    std::cout << "\n=== paper cases (figures_v15) ===" << std::endl;
    test_paper_cases();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total: " << total_tests << ", Passed: " << passed_tests
              << ", Failed: " << failed_tests << std::endl;
    return failed_tests > 0 ? 1 : 0;
}
