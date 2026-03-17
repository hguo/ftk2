#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <ftk2/numeric/pv_tet_classify.hpp>
#include <ftk2/numeric/pv_tri_classify_2d.hpp>
#include <iostream>
#include <cmath>
#include <cstring>

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

#define ASSERT_EQ_STR(a, b) \
    total_tests++; \
    if ((a) == std::string(b)) { \
        passed_tests++; \
    } else { \
        failed_tests++; \
        std::cerr << "FAILED: " << #a << " == \"" << (b) << "\"" \
                  << " got \"" << (a) << "\"" \
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
    #define RUNTAG(V, W, sd, exp_np, exp_npairs, exp_cat) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << sd \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
        { TetCaseV2GPU tv2; \
          memset(&tv2, 0, sizeof(tv2)); \
          for (int i=0;i<4;i++) for (int j=0;j<3;j++) { tv2.V[i][j]=V[i][j]; tv2.W[i][j]=W[i][j]; } \
          tv2.v2 = result; tv2.seed = sd; \
          for (int k=0;k<4;k++) tv2.disc_sign[k] = discriminant_sign_i128(P[k]); \
          ClassifiedCase cc = classify_case_v2(tv2); \
          total_tests++; \
          if (cc.category != std::string(exp_cat)) { \
              failed_tests++; \
              std::cerr << "FAILED: seed " << sd \
                        << " category=\"" << cc.category << "\" (exp \"" << (exp_cat) << "\")" \
                        << std::endl; \
          } else { passed_tests++; } \
        } \
    } while(0)

    {  int V[4][3] = {{-1,3,-17},{-6,11,-3},{3,6,-14},{11,7,-1}};
       int W[4][3] = {{18,16,7},{3,-9,12},{-4,0,-8},{4,14,0}};
       RUNTAG(V, W, 414, 4, 2, "T4_(1,3)_Q3+_Cw"); }
    {  int V[4][3] = {{-3,5,-9},{-18,6,14},{4,0,-4},{11,0,1}};
       int W[4][3] = {{7,-11,0},{16,-4,6},{-3,-17,-15},{-18,17,19}};
       RUNTAG(V, W, 515, 6, 3, "T6_(2,2,2)_Q3+"); }
    {  int V[4][3] = {{12,-3,-8},{18,-20,-19},{6,4,2},{8,20,7}};
       int W[4][3] = {{-17,19,-14},{-17,13,9},{-17,-13,10},{-17,1,8}};
       RUNTAG(V, W, 898, 2, 1, "T2_Q2"); }
    {  int V[4][3] = {{-1,-19,-10},{15,20,10},{-8,-13,-3},{15,-3,-6}};
       int W[4][3] = {{16,-20,13},{-18,5,-4},{5,7,-7},{-16,2,-12}};
       RUNTAG(V, W, 1570, 6, 3, "T6_(2,2,2)_Q3+"); }
    {  int V[4][3] = {{15,8,20},{3,12,3},{12,-20,-15},{-16,-5,-18}};
       int W[4][3] = {{-15,13,2},{8,13,20},{7,18,8},{-16,13,-3}};
       RUNTAG(V, W, 2098, 2, 1, "T2_Q3+"); }
    {  int V[4][3] = {{-13,8,-13},{-16,-2,20},{-13,8,-13},{2,-8,8}};
       int W[4][3] = {{-10,9,7},{-4,4,-3},{-14,2,6},{12,-5,-4}};
       RUNTAG(V, W, 3392, 2, 1, "T2_(1,1)_Q3+_Cw"); }
    {  int V[4][3] = {{14,12,-15},{14,-7,1},{-8,-15,-7},{1,5,-19}};
       int W[4][3] = {{3,-1,-4},{-11,4,20},{9,14,-9},{-10,-15,-18}};
       RUNTAG(V, W, 4170, 6, 3, "T6_(3,3)_Q3+_Cw"); }
    {  int V[4][3] = {{-14,-4,6},{16,-3,7},{3,0,20},{-3,0,-20}};
       int W[4][3] = {{-9,7,6},{2,-14,1},{12,6,17},{1,-7,5}};
       RUNTAG(V, W, 4423, 2, 1, "T2_Q3-_Cv1"); }
    {  int V[4][3] = {{-5,5,-15},{-1,8,5},{-1,8,-1},{-1,-2,6}};
       int W[4][3] = {{-1,-17,11},{13,-20,20},{12,3,12},{-17,16,-19}};
       RUNTAG(V, W, 4988, 6, 3, "T6_(2,4)_Q3+"); }
    {  int V[4][3] = {{4,-20,14},{-1,2,-17},{-5,-18,-20},{12,18,-13}};
       int W[4][3] = {{3,11,-18},{-9,9,-19},{4,-16,3},{19,0,4}};
       RUNTAG(V, W, 5629, 6, 3, "T6_Q3+"); }
    {  int V[4][3] = {{-19,-2,13},{16,13,-14},{-8,5,18},{-14,-6,-14}};
       int W[4][3] = {{-13,12,10},{18,2,17},{-14,-8,3},{-17,-9,-5}};
       RUNTAG(V, W, 6737, 4, 2, "T4_(2,2)_Q3+"); }
    {  int V[4][3] = {{-15,17,3},{7,-13,17},{3,-14,-2},{13,10,-16}};
       int W[4][3] = {{-19,-17,-8},{11,19,7},{13,7,0},{4,14,0}};
       RUNTAG(V, W, 8482, 2, 1, "T2_Q3-_Cv"); }
    {  int V[4][3] = {{-11,9,-5},{-8,4,-15},{-8,4,-15},{-9,-18,11}};
       int W[4][3] = {{6,1,8},{-5,15,18},{-8,-11,-20},{-11,-16,-19}};
       RUNTAG(V, W, 10183, 4, 2, "T4_(2,2)_Q3+"); }
    {  int V[4][3] = {{-19,-2,13},{2,-13,20},{-17,-10,-13},{12,10,-6}};
       int W[4][3] = {{19,4,-8},{17,1,19},{4,19,-11},{-9,1,-3}};
       RUNTAG(V, W, 10307, 2, 1, "T2_Q3+_Cv"); }
    {  int V[4][3] = {{-6,-20,7},{6,-1,0},{-9,-16,-10},{5,-13,2}};
       int W[4][3] = {{-5,-13,7},{0,7,-18},{11,-3,12},{-17,-14,-3}};
       RUNTAG(V, W, 10310, 2, 1, "T2_Q3+"); }
    {  int V[4][3] = {{4,-20,-3},{-17,-7,8},{-15,-7,-20},{1,-13,-18}};
       int W[4][3] = {{12,-11,13},{6,17,-15},{4,2,13},{7,-16,19}};
       RUNTAG(V, W, 10312, 2, 1, "T2_Q3-"); }
    {  int V[4][3] = {{5,-6,-9},{18,-2,6},{-14,5,2},{14,14,-19}};
       int W[4][3] = {{14,1,-15},{6,-6,-10},{-2,-5,10},{0,4,1}};
       RUNTAG(V, W, 10322, 6, 3, "T6_Q3-"); }
    {  int V[4][3] = {{-17,6,-7},{-9,2,-19},{11,20,7},{9,-10,1}};
       int W[4][3] = {{11,-19,17},{16,-5,0},{-13,-6,-12},{-19,14,0}};
       RUNTAG(V, W, 10417, 4, 2, "T4_(1,1,2)_Q3+_Cw"); }
    {  int V[4][3] = {{-4,-12,-9},{-5,14,6},{-18,14,10},{2,8,9}};
       int W[4][3] = {{-17,5,-13},{-1,1,4},{-6,17,7},{18,-1,0}};
       RUNTAG(V, W, 10420, 4, 2, "T4_Q3-"); }
    {  int V[4][3] = {{-6,-10,-5},{-17,9,19},{20,8,20},{2,5,11}};
       int W[4][3] = {{13,19,-20},{8,6,-2},{-5,-4,0},{-4,2,19}};
       RUNTAG(V, W, 10553, 8, 4, "T8_(2,2,4)_Q3+"); }
    {  int V[4][3] = {{-2,-15,-4},{-11,5,8},{13,-8,5},{-9,7,-3}};
       int W[4][3] = {{-7,7,3},{4,-4,-8},{3,3,17},{-10,11,10}};
       RUNTAG(V, W, 11834, 4, 2, "T4_(2,2)_Q2"); }
    {  int V[4][3] = {{7,-2,-8},{-1,19,10},{6,-10,5},{-12,20,13}};
       int W[4][3] = {{20,-12,-20},{-8,-14,12},{13,14,12},{14,12,-7}};
       RUNTAG(V, W, 12369, 4, 2, "T4_Q3+"); }
    {  int V[4][3] = {{14,-14,19},{8,-4,3},{-18,20,4},{-5,-13,9}};
       int W[4][3] = {{-17,-19,5},{-16,0,-7},{19,2,-17},{-18,19,18}};
       RUNTAG(V, W, 14299, 4, 2, "T4_Q3-"); }
    {  int V[4][3] = {{-12,9,13},{0,-4,-6},{14,-14,-3},{9,-17,12}};
       int W[4][3] = {{-6,-5,7},{18,0,-12},{19,13,18},{-3,5,-1}};
       RUNTAG(V, W, 15587, 4, 2, "T4_Q3-_Cw2"); }
    {  int V[4][3] = {{-10,8,7},{-8,9,19},{13,-8,-12},{-14,15,14}};
       int W[4][3] = {{17,12,-15},{1,-14,-8},{-2,0,-15},{-7,-10,9}};
       RUNTAG(V, W, 25710, 8, 4, "T8_(2,6)_Q3-"); }
    {  int V[4][3] = {{-20,-9,6},{18,19,15},{5,-5,-17},{-2,13,-3}};
       int W[4][3] = {{19,1,-3},{-7,12,16},{10,17,19},{-12,-4,-1}};
       RUNTAG(V, W, 30810, 8, 4, "T8_Q3-"); }
    {  int V[4][3] = {{1,-15,-2},{13,11,9},{12,16,-9},{-7,9,13}};
       int W[4][3] = {{9,0,19},{13,12,8},{8,2,13},{-3,-8,3}};
       RUNTAG(V, W, 58615, 4, 2, "T4_Q2-"); }
    {  int V[4][3] = {{-14,2,-6},{-12,19,-18},{-20,17,-4},{11,-9,-9}};
       int W[4][3] = {{-8,1,18},{19,-6,4},{-15,-6,-11},{-11,9,9}};
       RUNTAG(V, W, 65683, 2, 1, "T2_Q3+_D00"); }
    {  int V[4][3] = {{6,-8,-6},{3,17,10},{17,-10,17},{-6,-18,-5}};
       int W[4][3] = {{-8,10,-5},{-9,-18,15},{18,-15,9},{-9,-7,5}};
       RUNTAG(V, W, 1783954, 10, 5, "T10_Q3-"); }
    {  int V[4][3] = {{-9,7,-18},{-2,19,13},{20,3,7},{-11,5,1}};
       int W[4][3] = {{-3,9,-12},{-16,7,-8},{9,-9,9},{4,13,-7}};
       RUNTAG(V, W, 136281707, 2, 1, "T2_(1,1)_Q3+_Cw"); }

    #undef RUNTAG
}

// ============================================================================
// Paper cases (figures_v18): 38 additional approved cases
// ============================================================================

void test_paper_cases_v18() {
    std::cout << "  38 approved paper cases (figures_v18)" << std::endl;
    #define RUNTAG(V, W, sd, exp_np, exp_npairs, exp_cat) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << sd \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
        { TetCaseV2GPU tv2; \
          memset(&tv2, 0, sizeof(tv2)); \
          for (int i=0;i<4;i++) for (int j=0;j<3;j++) { tv2.V[i][j]=V[i][j]; tv2.W[i][j]=W[i][j]; } \
          tv2.v2 = result; tv2.seed = sd; \
          for (int k=0;k<4;k++) tv2.disc_sign[k] = discriminant_sign_i128(P[k]); \
          ClassifiedCase cc = classify_case_v2(tv2); \
          total_tests++; \
          if (cc.category != std::string(exp_cat)) { \
              failed_tests++; \
              std::cerr << "FAILED: seed " << sd \
                        << " category=\"" << cc.category << "\" (exp \"" << (exp_cat) << "\")" \
                        << std::endl; \
          } else { passed_tests++; } \
        } \
    } while(0)

    {  int V[4][3] = {{-11,-10,-11},{-2,17,13},{-8,5,12},{-17,-3,1}};
       int W[4][3] = {{6,10,17},{-19,-19,5},{-14,-19,15},{4,19,18}};
       RUNTAG(V, W, 3618, 0, 0, "T0_Q3+"); }
    {  int V[4][3] = {{-2,-9,-4},{18,7,4},{15,4,-12},{12,-8,0}};
       int W[4][3] = {{-14,14,-5},{18,-14,-20},{-6,-1,-8},{-3,-8,-15}};
       RUNTAG(V, W, 3639, 0, 0, "T0_Q3-"); }
    {  int V[4][3] = {{7,4,5},{17,15,-16},{-14,6,19},{-10,-17,13}};
       int W[4][3] = {{13,-16,-4},{-8,10,-2},{-10,2,-5},{-12,-5,-8}};
       RUNTAG(V, W, 3621, 2, 1, "T2_Q3+"); }
    {  int V[4][3] = {{-19,-5,13},{13,-5,20},{7,18,13},{8,-8,5}};
       int W[4][3] = {{-17,18,7},{-11,-2,-10},{17,1,-6},{4,-12,-8}};
       RUNTAG(V, W, 3619, 2, 1, "T2_Q3-"); }
    {  int V[4][3] = {{13,18,19},{13,-4,-16},{5,-3,-19},{-20,-3,1}};
       int W[4][3] = {{-10,20,-4},{6,-12,5},{6,-18,13},{10,7,11}};
       RUNTAG(V, W, 1611, 4, 2, "T4_Q3+"); }
    {  int V[4][3] = {{2,13,-17},{-2,-12,-3},{0,-1,15},{17,15,5}};
       int W[4][3] = {{19,-7,-9},{5,-9,5},{-7,0,-7},{-4,4,18}};
       RUNTAG(V, W, 3617, 4, 2, "T4_Q3-"); }
    {  int V[4][3] = {{-17,15,13},{-20,-10,16},{1,4,-15},{20,8,17}};
       int W[4][3] = {{-19,5,16},{-7,-6,17},{15,5,-4},{-8,13,-9}};
       RUNTAG(V, W, 2397, 6, 3, "T6_Q3+"); }
    {  int V[4][3] = {{20,-14,-1},{-20,-18,5},{-6,0,-19},{-17,1,-20}};
       int W[4][3] = {{-15,8,11},{16,16,-2},{-3,-20,14},{-16,-20,-4}};
       RUNTAG(V, W, 3640, 2, 1, "T2_(1,1)_Q3+_Cw"); }
    {  int V[4][3] = {{12,13,14},{16,-18,5},{-6,4,5},{13,15,-11}};
       int W[4][3] = {{-5,-9,-3},{11,-3,12},{-18,5,-6},{20,2,-1}};
       RUNTAG(V, W, 3627, 2, 1, "T2_(1,1)_Q3-_Cw"); }
    {  int V[4][3] = {{-11,-2,14},{12,-4,-6},{19,-8,-16},{7,-2,-16}};
       int W[4][3] = {{-18,-19,19},{9,-13,-9},{12,-15,7},{-13,14,7}};
       RUNTAG(V, W, 9579, 4, 2, "T4_(2,2)_Q3+"); }
    {  int V[4][3] = {{18,-7,12},{8,-11,-6},{-12,-6,19},{2,-18,5}};
       int W[4][3] = {{5,9,-19},{11,-16,-8},{5,20,-3},{-10,11,-10}};
       RUNTAG(V, W, 13109, 4, 2, "T4_(2,2)_Q3-"); }
    {  int V[4][3] = {{-14,-9,10},{-16,18,8},{20,1,-13},{-2,13,-8}};
       int W[4][3] = {{-6,6,11},{-4,-8,-14},{7,-18,-8},{11,13,-14}};
       RUNTAG(V, W, 9476, 4, 2, "T4_(1,3)_Q3-_Cw"); }
    {  int V[4][3] = {{16,15,4},{-10,8,17},{20,-1,8},{-14,-9,-4}};
       int W[4][3] = {{-7,4,-3},{-6,16,-7},{14,-9,14},{10,8,-11}};
       RUNTAG(V, W, 11021, 6, 3, "T6_(2,2,2)_Q3+"); }
    {  int V[4][3] = {{9,9,-8},{-11,20,10},{-19,-8,20},{16,-8,-12}};
       int W[4][3] = {{-11,-18,12},{-9,-15,-15},{-12,8,-1},{12,6,16}};
       RUNTAG(V, W, 12630, 6, 3, "T6_(2,4)_Q3-"); }
    {  int V[4][3] = {{-12,-6,-5},{9,2,15},{9,10,20},{-4,7,0}};
       int W[4][3] = {{-17,-8,11},{3,13,1},{-12,18,-6},{10,-9,-1}};
       RUNTAG(V, W, 5690, 6, 3, "T6_(3,3)_Q3-_Cw"); }
    {  int V[4][3] = {{0,7,2},{16,-17,-9},{20,7,-1},{-17,9,10}};
       int W[4][3] = {{18,-17,4},{20,13,-7},{18,-20,17},{-15,9,-18}};
       RUNTAG(V, W, 57668, 8, 4, "T8_(4,4)_Q3-"); }
    {  int V[4][3] = {{19,9,7},{7,0,18},{8,-11,-10},{-20,19,4}};
       int W[4][3] = {{-19,-11,20},{-7,16,-1},{15,-2,-5},{-7,8,11}};
       RUNTAG(V, W, 3637, 2, 1, "T2_Q3+_Cv"); }
    {  int V[4][3] = {{10,4,0},{-17,10,-14},{-19,-14,14},{-8,-17,5}};
       int W[4][3] = {{1,-15,1},{-3,15,20},{7,-20,14},{0,-6,3}};
       RUNTAG(V, W, 3616, 2, 1, "T2_Q3-_Cv"); }
    {  int V[4][3] = {{-10,13,-3},{5,-1,1},{5,-4,14},{3,-5,1}};
       int W[4][3] = {{-10,-15,7},{-17,-7,11},{6,-13,-7},{10,-16,-18}};
       RUNTAG(V, W, 9974, 2, 1, "T2_Q3+_Cv2"); }
    {  int V[4][3] = {{1,19,0},{-19,19,5},{3,-19,-1},{-11,17,20}};
       int W[4][3] = {{11,-12,-19},{-8,-17,-2},{-6,-14,-16},{8,10,-16}};
       RUNTAG(V, W, 2173, 2, 1, "T2_Q3-_Cv2"); }
    {  int V[4][3] = {{-13,-2,-15},{-10,0,1},{13,8,-16},{6,-9,20}};
       int W[4][3] = {{0,7,10},{1,-12,13},{-11,-10,-18},{3,11,20}};
       RUNTAG(V, W, 8056, 4, 2, "T4_(2,2)_Q3+_Cv"); }
    {  int V[4][3] = {{-17,-9,2},{-16,-16,-17},{2,17,6},{15,-2,7}};
       int W[4][3] = {{6,16,-14},{5,-6,7},{8,8,6},{-9,8,-8}};
       RUNTAG(V, W, 14150, 6, 3, "T6_(2,2,2)_Q3+_Cv"); }
    {  int V[4][3] = {{-8,-20,12},{17,-11,16},{20,13,17},{-9,9,4}};
       int W[4][3] = {{18,-9,3},{-6,3,-1},{16,-12,4},{16,-9,8}};
       RUNTAG(V, W, 11272, 2, 1, "T2_Q3+_Cw1"); }
    {  int V[4][3] = {{18,17,19},{10,-4,-5},{-8,4,-9},{-6,20,-4}};
       int W[4][3] = {{-9,7,1},{-9,-9,-7},{-8,0,9},{9,-7,-1}};
       RUNTAG(V, W, 12810, 2, 1, "T2_(1,1)_Q3-_Cw1"); }
    {  int V[4][3] = {{-9,-6,-3},{14,-15,1},{1,11,-2},{-8,-6,-7}};
       int W[4][3] = {{2,7,-7},{1,-15,-15},{-9,4,-18},{9,-4,18}};
       RUNTAG(V, W, 6782, 4, 2, "T4_(1,1,2)_Q3+_Cw1"); }
    {  int V[4][3] = {{-11,-2,-16},{7,14,16},{10,-16,-11},{-14,-15,8}};
       int W[4][3] = {{-5,-12,20},{14,-8,1},{8,20,-18},{-14,8,-1}};
       RUNTAG(V, W, 46038, 6, 3, "T6_(2,2,2)_Q3+_Cv_Cw1"); }
    {  int V[4][3] = {{19,2,-3},{5,-12,1},{13,4,13},{-3,-18,9}};
       int W[4][3] = {{-17,20,18},{-15,11,-7},{-1,18,-2},{1,6,-3}};
       RUNTAG(V, W, 65893, 2, 1, "T2_Q3+_D00"); }
    {  int V[4][3] = {{18,-4,1},{17,19,17},{-20,5,0},{-10,-3,-15}};
       int W[4][3] = {{-18,-11,-1},{2,17,16},{-12,3,0},{-18,8,-15}};
       RUNTAG(V, W, 23330, 6, 3, "T6_(2,2,2)_Q3+_D00"); }
    {  int V[4][3] = {{8,-5,-7},{0,0,0},{13,16,18},{2,0,-4}};
       int W[4][3] = {{-20,-10,16},{-11,-6,-9},{6,4,-18},{12,15,-3}};
       RUNTAG(V, W, 36554, 2, 1, "T2_Q3+_Cv0_D00"); }
    {  int V[4][3] = {{9,12,1},{1,15,-9},{11,-8,5},{-1,-9,-10}};
       int W[4][3] = {{9,0,17},{8,-12,0},{10,18,-17},{0,0,0}};
       RUNTAG(V, W, 77331, 2, 1, "T2_Q3+_Cw0_D00"); }
    {  int V[4][3] = {{3,5,4},{11,19,-9},{-7,18,-7},{5,-19,-14}};
       int W[4][3] = {{-12,-8,14},{0,-17,0},{-18,-5,15},{-6,-14,1}};
       RUNTAG(V, W, 2360, 2, 1, "T2_Q2"); }
    {  int V[4][3] = {{2,-2,-1},{-15,-1,20},{-16,17,-7},{17,10,12}};
       int W[4][3] = {{4,1,-18},{-9,-2,15},{8,-19,-20},{-3,4,18}};
       RUNTAG(V, W, 4367, 4, 2, "T4_(1,1,2)_Q3+_Cv_Cw"); }
    {  int V[4][3] = {{20,14,5},{-8,4,6},{13,-13,-7},{16,-7,-18}};
       int W[4][3] = {{18,-4,20},{-10,2,-13},{-19,-2,-6},{4,17,-20}};
       RUNTAG(V, W, 8206, 6, 3, "T6_(1,5)_Q3-_Cv_Cw"); }
    {  int V[4][3] = {{-4,-7,6},{6,-20,3},{20,8,-13},{19,15,3}};
       int W[4][3] = {{14,-8,16},{9,-8,-1},{-16,-3,8},{-7,6,-1}};
       RUNTAG(V, W, 25274, 8, 4, "T8_(1,7)_Q3-_Cw"); }
    {  int V[4][3] = {{-11,14,8},{3,-8,-4},{-15,-8,14},{-3,8,4}};
       int W[4][3] = {{-6,-13,9},{-14,5,-12},{-3,-5,-1},{-9,-20,-15}};
       RUNTAG(V, W, 8139, 0, 0, "T0_Q3+_Cv1"); }
    {  int V[4][3] = {{-8,16,-14},{17,-13,-2},{-17,2,-3},{8,9,-1}};
       int W[4][3] = {{-11,1,9},{-3,10,-10},{11,-1,-9},{-16,1,18}};
       RUNTAG(V, W, 52788, 0, 0, "T0_Q3-_Cw1"); }
    {  int V[4][3] = {{0,0,0},{-7,-12,-5},{9,-20,-16},{11,11,11}};
       int W[4][3] = {{8,-18,-18},{9,-10,9},{20,4,-14},{8,13,-20}};
       RUNTAG(V, W, 54581, 0, 0, "T0_Q3-_Cv0_D00"); }
    {  int V[4][3] = {{7,16,14},{1,11,-10},{-4,-14,-16},{11,-5,17}};
       int W[4][3] = {{-8,-20,3},{0,0,0},{15,-18,-8},{-1,1,-20}};
       RUNTAG(V, W, 1704, 4, 2, "T4_Q3-_Cw0_D00"); }

    #undef RUNTAG
}

// ============================================================================
// Constructed degenerate cases (figures_v19): B, SR, ISR, TN, D01, D11, D12, D22
// ============================================================================

void test_constructed_cases() {
    std::cout << "  10 constructed degenerate cases (figures_v19)" << std::endl;
    #define RUNTAG(V, W, sd, exp_np, exp_npairs, exp_cat) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << sd \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
        { TetCaseV2GPU tv2; \
          memset(&tv2, 0, sizeof(tv2)); \
          for (int i=0;i<4;i++) for (int j=0;j<3;j++) { tv2.V[i][j]=V[i][j]; tv2.W[i][j]=W[i][j]; } \
          tv2.v2 = result; tv2.seed = sd; \
          for (int k=0;k<4;k++) tv2.disc_sign[k] = discriminant_sign_i128(P[k]); \
          ClassifiedCase cc = classify_case_v2(tv2); \
          total_tests++; \
          if (cc.category != std::string(exp_cat)) { \
              failed_tests++; \
              std::cerr << "FAILED: seed " << sd \
                        << " category=\"" << cc.category << "\" (exp \"" << (exp_cat) << "\")" \
                        << std::endl; \
          } else { passed_tests++; } \
        } \
    } while(0)

    {  int V[4][3] = {{2,-1,3},{-1,2,-2},{-2,2,0},{-1,1,-3}};
       int W[4][3] = {{0,2,2},{-2,1,-1},{3,2,-2},{-3,-1,2}};
       RUNTAG(V, W, 1, 4, 2, "T4_Q3-_TN"); }
    {  int V[4][3] = {{2,0,0},{0,3,0},{0,0,5},{7,-8,4}};
       int W[4][3] = {{-2,0,0},{0,-3,0},{0,0,-5},{1,2,-3}};
       RUNTAG(V, W, 2, 0, 0, "T0_Q3o_D22"); }
    {  int V[4][3] = {{1,2,1},{-2,-2,-2},{1,1,1},{3,-2,3}};
       int W[4][3] = {{2,2,-3},{3,-3,0},{-2,-3,3},{-2,-2,-3}};
       RUNTAG(V, W, 3, 2, 1, "T2_(1,1)_Q3+_SR"); }
    {  int V[4][3] = {{0,-1,0},{2,-2,0},{0,0,-3},{0,2,-1}};
       int W[4][3] = {{0,0,-3},{2,1,-1},{0,-1,-3},{2,-2,-3}};
       RUNTAG(V, W, 4, 2, 1, "T2_(1,1)_Q3+_ISR"); }
    {  int V[4][3] = {{-1,-2,1},{-3,2,-1},{3,1,0},{-1,-3,0}};
       int W[4][3] = {{2,-1,1},{-1,-1,1},{0,2,-2},{-2,-1,1}};
       RUNTAG(V, W, 5, 0, 0, "T0_Q2-_Cv_B"); }
    {  int V[4][3] = {{-3,-1,2},{1,1,1},{-2,-2,0},{2,-2,0}};
       int W[4][3] = {{1,1,-2},{1,-1,-1},{0,2,-3},{0,1,-3}};
       RUNTAG(V, W, 6, 1, 0, "T0_Q3-_D01"); }
    {  int V[4][3] = {{-3,-3,2},{1,-1,1},{-2,2,-1},{3,3,-1}};
       int W[4][3] = {{0,-2,2},{-3,1,-1},{2,-2,1},{-3,-3,1}};
       RUNTAG(V, W, 7, 2, 1, "T2_Q3+_D11"); }
    {  int V[4][3] = {{-2,-2,2},{-3,0,3},{1,2,-1},{3,-3,-2}};
       int W[4][3] = {{2,0,-2},{0,-2,0},{0,-1,0},{1,1,1}};
       RUNTAG(V, W, 8, 1, 0, "T2_Q3-_D12"); }
    {  int V[4][3] = {{2,2,-2},{0,0,3},{2,-1,2},{1,0,1}};
       int W[4][3] = {{2,1,-2},{-3,1,-3},{-2,1,-2},{3,-2,3}};
       RUNTAG(V, W, 9, 2, 1, "T2_Q3+_TN_D00"); }
    {  int V[4][3] = {{12,15,-6},{-16,-9,-3},{8,-16,-20},{1,-1,-3}};
       int W[4][3] = {{12,8,-19},{-16,-9,-10},{8,16,11},{1,3,11}};
       RUNTAG(V, W, 101980, 4, 2, "T4_(1,1,2)_Q3+_SR"); }

    #undef RUNTAG
}

// ============================================================================
// New structural cases (figures_v20): T10, Cw2, missing T-distributions
// ============================================================================

void test_structural_cases_v20() {
    std::cout << "  11 new structural cases (figures_v20)" << std::endl;
    #define RUNTAG(V, W, sd, exp_np, exp_npairs, exp_cat) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << sd \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
        { TetCaseV2GPU tv2; \
          memset(&tv2, 0, sizeof(tv2)); \
          for (int i=0;i<4;i++) for (int j=0;j<3;j++) { tv2.V[i][j]=V[i][j]; tv2.W[i][j]=W[i][j]; } \
          tv2.v2 = result; tv2.seed = sd; \
          for (int k=0;k<4;k++) tv2.disc_sign[k] = discriminant_sign_i128(P[k]); \
          ClassifiedCase cc = classify_case_v2(tv2); \
          total_tests++; \
          if (cc.category != std::string(exp_cat)) { \
              failed_tests++; \
              std::cerr << "FAILED: seed " << sd \
                        << " category=\"" << cc.category << "\" (exp \"" << (exp_cat) << "\")" \
                        << std::endl; \
          } else { passed_tests++; } \
        } \
    } while(0)

    {  int V[4][3] = {{-25,11,-6},{-44,24,16},{-17,-29,6},{41,3,3}};
       int W[4][3] = {{-40,-34,31},{10,-2,1},{-13,8,-4},{33,-40,-43}};
       RUNTAG(V, W, 394695, 10, 5, "T10_(3,7)_Q3-_Cv_Cw"); }
    {  int V[4][3] = {{-6,-10,19},{-19,18,10},{-18,-7,10},{10,-9,-6}};
       int W[4][3] = {{-14,1,-8},{-17,11,0},{18,-13,20},{17,17,-9}};
       RUNTAG(V, W, 8090, 8, 4, "T8_(3,5)_Q3-_Cw"); }
    {  int V[4][3] = {{-1,-15,-11},{2,16,17},{16,20,-10},{-7,13,1}};
       int W[4][3] = {{-10,19,16},{18,-19,-14},{-16,18,12},{-19,6,2}};
       RUNTAG(V, W, 11073, 6, 3, "T6_(1,1,2,2)_Q3+_Cv_Cw"); }
    {  int V[4][3] = {{-17,-5,-9},{4,-19,6},{18,11,9},{-5,-9,-2}};
       int W[4][3] = {{-16,10,19},{9,-17,2},{4,14,14},{-4,-8,-14}};
       RUNTAG(V, W, 13599, 6, 3, "T6_(1,2,3)_Q3+_Cv_Cw"); }
    {  int V[4][3] = {{-6,-20,16},{-13,13,-16},{10,0,20},{-1,5,-12}};
       int W[4][3] = {{-3,-19,7},{-19,14,6},{-5,3,-13},{17,20,-8}};
       RUNTAG(V, W, 21181, 6, 3, "T6_(1,1,4)_Q3+_Cv_Cw"); }
    {  int V[4][3] = {{-7,6,15},{-8,20,13},{-12,17,-11},{2,7,-11}};
       int W[4][3] = {{18,0,-6},{10,4,9},{8,-8,8},{-13,1,3}};
       RUNTAG(V, W, 52919, 2, 1, "T2_Q3+_Cw2"); }
    {  int V[4][3] = {{-3,-6,-9},{6,-3,9},{-9,6,-3},{3,3,3}};
       int W[4][3] = {{1,2,3},{-2,1,-3},{3,-2,1},{-1,-1,-1}};
       RUNTAG(V, W, 997, 0, 0, "T0_Q3o_Cv_Cw_D33"); }
    {  int V[4][3] = {{1,2,0},{-1,3,0},{2,-1,0},{-2,1,0}};
       int W[4][3] = {{3,-1,0},{1,2,0},{-1,3,0},{2,-2,0}};
       RUNTAG(V, W, 990, 0, 0, "T0_Qz_Cv1_D23"); }
    // D23 + Cv + Cw: origin strictly inside both V and W hulls
    {  int V[4][3] = {{3,1,0},{-1,3,0},{-2,-1,0},{0,-3,0}};
       int W[4][3] = {{2,3,0},{-3,1,0},{-1,-2,0},{1,-1,0}};
       RUNTAG(V, W, 991, 0, 0, "T0_Qz_Cv_Cw_D23"); }
    // D23 + Cv only: origin inside V hull, W all first quadrant
    {  int V[4][3] = {{3,1,0},{-1,3,0},{-2,-1,0},{0,-3,0}};
       int W[4][3] = {{1,1,0},{2,1,0},{1,2,0},{3,2,0}};
       RUNTAG(V, W, 992, 0, 0, "T0_Qz_Cv_D23"); }
    // D23 + Cw only: V all first quadrant, origin inside W hull
    {  int V[4][3] = {{1,1,0},{2,1,0},{1,2,0},{3,2,0}};
       int W[4][3] = {{2,3,0},{-3,1,0},{-1,-2,0},{1,-1,0}};
       RUNTAG(V, W, 993, 0, 0, "T0_Qz_Cw_D23"); }

    #undef RUNTAG
}

// ============================================================================
// content_reduce_i128 tests
// ============================================================================

void test_content_reduce_basic() {
    std::cout << "  content_reduce_basic" << std::endl;
    __int128 p[4] = {6, 10, 4};
    content_reduce_i128(p, 2);
    ASSERT_EQ(p[0], (__int128)3);
    ASSERT_EQ(p[1], (__int128)5);
    ASSERT_EQ(p[2], (__int128)2);
}

void test_content_reduce_already_primitive() {
    std::cout << "  content_reduce_already_primitive" << std::endl;
    __int128 p[4] = {3, 5, 7};
    content_reduce_i128(p, 2);
    ASSERT_EQ(p[0], (__int128)3);
    ASSERT_EQ(p[1], (__int128)5);
    ASSERT_EQ(p[2], (__int128)7);
}

void test_content_reduce_with_zero() {
    std::cout << "  content_reduce_with_zero" << std::endl;
    __int128 p[4] = {0, 6, 9};
    content_reduce_i128(p, 2);
    ASSERT_EQ(p[0], (__int128)0);
    ASSERT_EQ(p[1], (__int128)2);
    ASSERT_EQ(p[2], (__int128)3);
}

void test_content_reduce_negative() {
    std::cout << "  content_reduce_negative" << std::endl;
    __int128 p[4] = {-12, 18, -6};
    content_reduce_i128(p, 2);
    ASSERT_EQ(p[0], (__int128)-2);
    ASSERT_EQ(p[1], (__int128)3);
    ASSERT_EQ(p[2], (__int128)-1);
}

void test_content_reduce_degree_zero() {
    std::cout << "  content_reduce_degree_zero" << std::endl;
    __int128 p[1] = {42};
    content_reduce_i128(p, 0);
    ASSERT_EQ(p[0], (__int128)1);
}

void test_content_reduce_neg_degree() {
    std::cout << "  content_reduce_neg_degree" << std::endl;
    __int128 p[1] = {99};
    content_reduce_i128(p, -1);
    ASSERT_EQ(p[0], (__int128)99);
}

// ============================================================================
// effective_degree_i128 tests
// ============================================================================

void test_effective_degree() {
    std::cout << "  effective_degree" << std::endl;
    __int128 p1[4] = {1, 2, 3, 0};
    ASSERT_EQ(effective_degree_i128(p1, 3), 2);

    __int128 p2[4] = {5, 0, 0, 0};
    ASSERT_EQ(effective_degree_i128(p2, 3), 0);

    __int128 p3[4] = {0, 0, 0, 0};
    ASSERT_EQ(effective_degree_i128(p3, 3), 0);

    __int128 p4[4] = {1, 2, 3, 4};
    ASSERT_EQ(effective_degree_i128(p4, 3), 3);
}

// ============================================================================
// discriminant_sign_i128 tests
// ============================================================================

void test_discriminant_sign_three_roots() {
    std::cout << "  discriminant_sign_three_roots" << std::endl;
    __int128 P[4] = {-6, 11, -6, 1};  // (x-1)(x-2)(x-3), disc > 0
    ASSERT_EQ(discriminant_sign_i128(P), 1);
}

void test_discriminant_sign_one_root() {
    std::cout << "  discriminant_sign_one_root" << std::endl;
    __int128 P[4] = {2, 1, 0, 1};  // x^3+x+2, disc < 0
    ASSERT_EQ(discriminant_sign_i128(P), -1);
}

void test_discriminant_sign_repeated() {
    std::cout << "  discriminant_sign_repeated" << std::endl;
    __int128 P[4] = {2, -3, 0, 1};  // (x-1)^2(x+2), disc = 0
    ASSERT_EQ(discriminant_sign_i128(P), 0);
}

void test_discriminant_sign_triple_root() {
    std::cout << "  discriminant_sign_triple_root" << std::endl;
    __int128 P[4] = {-8, 12, -6, 1};  // (x-2)^3, disc = 0
    ASSERT_EQ(discriminant_sign_i128(P), 0);
}

void test_discriminant_sign_not_cubic() {
    std::cout << "  discriminant_sign_not_cubic" << std::endl;
    __int128 P[4] = {1, 2, 3, 0};
    ASSERT_EQ(discriminant_sign_i128(P), 0);
}

void test_discriminant_sign_neg_leading() {
    std::cout << "  discriminant_sign_neg_leading" << std::endl;
    __int128 P[4] = {6, -11, 6, -1};  // -(x-1)(x-2)(x-3), disc > 0
    ASSERT_EQ(discriminant_sign_i128(P), 1);
}

// ============================================================================
// poly_exact_div_i128 tests
// ============================================================================

void test_poly_exact_div_linear() {
    std::cout << "  poly_exact_div_linear" << std::endl;
    // (x-1)(x-2)(x-3) / (x-1) = (x-2)(x-3) = x^2-5x+6
    __int128 f[8] = {-6, 11, -6, 1};
    __int128 g[8] = {-1, 1};
    __int128 q[8] = {};
    int dq = poly_exact_div_i128(f, 3, g, 1, q);
    ASSERT_EQ(dq, 2);
    ASSERT_EQ(q[0], (__int128)6);
    ASSERT_EQ(q[1], (__int128)-5);
    ASSERT_EQ(q[2], (__int128)1);
}

void test_poly_exact_div_quadratic() {
    std::cout << "  poly_exact_div_quadratic" << std::endl;
    // (x^2-1)(x-3) = x^3-3x^2-x+3, divided by x^2-1 = x-3
    __int128 f[8] = {3, -1, -3, 1};
    __int128 g[8] = {-1, 0, 1};
    __int128 q[8] = {};
    int dq = poly_exact_div_i128(f, 3, g, 2, q);
    ASSERT_EQ(dq, 1);
    ASSERT_EQ(q[0], (__int128)-3);
    ASSERT_EQ(q[1], (__int128)1);
}

void test_poly_exact_div_equal_degree() {
    std::cout << "  poly_exact_div_equal_degree" << std::endl;
    // 2x^2+4x+2 / (x^2+2x+1) = 2
    __int128 f[8] = {2, 4, 2};
    __int128 g[8] = {1, 2, 1};
    __int128 q[8] = {};
    int dq = poly_exact_div_i128(f, 2, g, 2, q);
    ASSERT_EQ(dq, 0);
    ASSERT_EQ(q[0], (__int128)2);
}

void test_poly_exact_div_neg_leading() {
    std::cout << "  poly_exact_div_neg_leading" << std::endl;
    // (x+1)(-2x+6) = -2x^2+4x+6, divided by (-2x+6) = x+1
    __int128 f[8] = {6, 4, -2};
    __int128 g[8] = {6, -2};
    __int128 q[8] = {};
    int dq = poly_exact_div_i128(f, 2, g, 1, q);
    ASSERT_EQ(dq, 1);
    ASSERT_EQ(q[0], (__int128)1);
    ASSERT_EQ(q[1], (__int128)1);
}

// ============================================================================
// poly_sqfree_i128 tests
// ============================================================================

void test_sqfree_already() {
    std::cout << "  sqfree_already_squarefree" << std::endl;
    __int128 f[8] = {-6, 11, -6, 1};  // (x-1)(x-2)(x-3)
    __int128 sf[8] = {};
    int dsf = poly_sqfree_i128(f, 3, sf);
    ASSERT_EQ(dsf, 3);
}

void test_sqfree_double_root() {
    std::cout << "  sqfree_double_root" << std::endl;
    // (x-1)^2(x+2) = x^3-3x+2
    __int128 f[8] = {2, -3, 0, 1};
    __int128 sf[8] = {};
    int dsf = poly_sqfree_i128(f, 3, sf);
    ASSERT_EQ(dsf, 2);
    // sf should vanish at x=1: sf[0]+sf[1]+sf[2] = 0
    ASSERT_EQ(sf[0]+sf[1]+sf[2], (__int128)0);
}

void test_sqfree_triple_root() {
    std::cout << "  sqfree_triple_root" << std::endl;
    // (x-2)^3 = x^3-6x^2+12x-8
    __int128 f[8] = {-8, 12, -6, 1};
    __int128 sf[8] = {};
    int dsf = poly_sqfree_i128(f, 3, sf);
    ASSERT_EQ(dsf, 1);
}

void test_sqfree_quadratic_double() {
    std::cout << "  sqfree_quadratic_double" << std::endl;
    // (x-3)^2 = x^2-6x+9
    __int128 f[8] = {9, -6, 1};
    __int128 sf[8] = {};
    int dsf = poly_sqfree_i128(f, 2, sf);
    ASSERT_EQ(dsf, 1);
}

void test_sqfree_linear() {
    std::cout << "  sqfree_linear" << std::endl;
    __int128 f[8] = {-5, 1};
    __int128 sf[8] = {};
    int dsf = poly_sqfree_i128(f, 1, sf);
    ASSERT_EQ(dsf, 1);
    ASSERT_EQ(sf[0], (__int128)-5);
    ASSERT_EQ(sf[1], (__int128)1);
}

// ============================================================================
// sign_at_unique_root_i128 tests
// ============================================================================

void test_sign_unique_root_positive() {
    std::cout << "  sign_unique_root_positive" << std::endl;
    // f = x^3+x+2 (1 real root at -1), g = x+3 => g(-1) = 2 > 0
    __int128 f[8] = {2, 1, 0, 1};
    __int128 g[8] = {3, 1};
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g, 1), 1);
}

void test_sign_unique_root_negative() {
    std::cout << "  sign_unique_root_negative" << std::endl;
    __int128 f[8] = {2, 1, 0, 1};
    __int128 g[8] = {0, 1};  // g = x, g(-1) = -1
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g, 1), -1);
}

void test_sign_unique_root_shared() {
    std::cout << "  sign_unique_root_shared" << std::endl;
    __int128 f[8] = {1, 1, 1, 1};  // (x+1)(x^2+1)
    __int128 g[8] = {1, 1};        // x+1, g(-1) = 0
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g, 1), 0);
}

void test_sign_unique_root_constant_g() {
    std::cout << "  sign_unique_root_constant_g" << std::endl;
    __int128 f[8] = {2, 1, 0, 1};
    __int128 g_pos[8] = {5};
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g_pos, 0), 1);
    __int128 g_neg[8] = {-3};
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g_neg, 0), -1);
    __int128 g_zero[8] = {0};
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g_zero, 0), 0);
}

void test_sign_unique_root_neg_lc() {
    std::cout << "  sign_unique_root_neg_lc" << std::endl;
    // f = -x^3-x-2 (1 real root at -1), g = x+3 => g(-1)=2>0
    __int128 f[8] = {-2, -1, 0, -1};
    __int128 g[8] = {3, 1};
    ASSERT_EQ(sign_at_unique_root_i128(f, 3, g, 1), 1);
}

// ============================================================================
// compare_roots_i128 tests
// ============================================================================

void test_compare_roots_disjoint() {
    std::cout << "  compare_roots_disjoint" << std::endl;
    __int128 f[8] = {-1, 1};  // root at 1
    __int128 g[8] = {-3, 1};  // root at 3
    ASSERT_EQ(compare_roots_i128(f, 1, 1, 0, g, 1, 1, 0), -1);  // 1 < 3
    ASSERT_EQ(compare_roots_i128(g, 1, 1, 0, f, 1, 1, 0), 1);   // 3 > 1
}

void test_compare_roots_equal() {
    std::cout << "  compare_roots_equal" << std::endl;
    __int128 f[8] = {-2, 1};           // root at 2
    __int128 g[8] = {10, -7, 1};       // (x-2)(x-5), roots at 2, 5
    ASSERT_EQ(compare_roots_i128(f, 1, 1, 0, g, 2, 2, 0), 0);  // both at 2
}

void test_compare_roots_3root_cubic() {
    std::cout << "  compare_roots_3root_cubic" << std::endl;
    __int128 f[8] = {-15, 23, -9, 1};   // (x-1)(x-3)(x-5)
    __int128 g[8] = {-48, 44, -12, 1};  // (x-2)(x-4)(x-6)
    ASSERT_EQ(compare_roots_i128(f, 3, 3, 0, g, 3, 3, 0), -1);  // 1 < 2
    ASSERT_EQ(compare_roots_i128(f, 3, 3, 1, g, 3, 3, 1), -1);  // 3 < 4
    ASSERT_EQ(compare_roots_i128(f, 3, 3, 2, g, 3, 3, 1), 1);   // 5 > 4
}

void test_compare_roots_degenerate() {
    std::cout << "  compare_roots_degenerate" << std::endl;
    __int128 f[8] = {5};  // constant, no roots
    __int128 g[8] = {-1, 1};
    ASSERT_EQ(compare_roots_i128(f, 0, 0, 0, g, 1, 1, 0), 0);
}

// ============================================================================
// compute_tet_QP_i128 tests
// ============================================================================

void test_compute_tet_QP_identity() {
    std::cout << "  compute_tet_QP_identity" << std::endl;
    // Q = P[0] + P[1] + P[2] + P[3] must hold exactly
    int V[4][3] = {{1,2,3},{-1,0,2},{3,-1,1},{0,1,-1}};
    int W[4][3] = {{0,1,-1},{2,-1,0},{-1,2,1},{1,0,2}};
    __int128 Q[4], P[4][4];
    compute_tet_QP_i128(V, W, Q, P);
    for (int i = 0; i < 4; i++) {
        __int128 sum = P[0][i] + P[1][i] + P[2][i] + P[3][i];
        ASSERT_EQ(sum, Q[i]);
    }
}

void test_compute_tet_QP_coplanar_zero() {
    std::cout << "  compute_tet_QP_coplanar_zero" << std::endl;
    // All V,W in z=0 plane => Q ≡ 0, P ≡ 0 (D23)
    int V[4][3] = {{1,0,0},{0,1,0},{-1,0,0},{0,-1,0}};
    int W[4][3] = {{0,1,0},{-1,0,0},{0,-1,0},{1,0,0}};
    __int128 Q[4], P[4][4];
    compute_tet_QP_i128(V, W, Q, P);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(Q[i], (__int128)0);
        for (int k = 0; k < 4; k++) {
            ASSERT_EQ(P[k][i], (__int128)0);
        }
    }
}

// ============================================================================
// check_field_zero_in_tet tests
// ============================================================================

void test_field_zero_inside() {
    std::cout << "  field_zero_inside" << std::endl;
    // Origin in convex hull of (1,0,0),(0,1,0),(0,0,1),(-1,-1,-1)
    int F[4][3] = {{1,0,0},{0,1,0},{0,0,1},{-1,-1,-1}};
    int64_t num[4]; int64_t den;
    ASSERT_TRUE(check_field_zero_in_tet(F, num, &den));
    ASSERT_TRUE(den != 0);
}

void test_field_zero_outside() {
    std::cout << "  field_zero_outside" << std::endl;
    // All same direction — origin not inside
    int F[4][3] = {{1,1,1},{2,2,2},{3,3,3},{4,4,4}};
    ASSERT_TRUE(!check_field_zero_in_tet(F));
}

void test_field_zero_at_vertex() {
    std::cout << "  field_zero_at_vertex" << std::endl;
    // F[3] = (0,0,0), so origin is at vertex 3
    int F[4][3] = {{1,0,0},{0,1,0},{0,0,1},{0,0,0}};
    ASSERT_TRUE(check_field_zero_in_tet(F));
}

void test_field_zero_coplanar_det0() {
    std::cout << "  field_zero_coplanar_det0" << std::endl;
    // Coplanar vectors: det = 0 → returns false
    int F[4][3] = {{1,0,0},{0,1,0},{-1,-1,0},{2,3,0}};
    ASSERT_TRUE(!check_field_zero_in_tet(F));
}

// ============================================================================
// check_field_zero_coplanar tests
// ============================================================================

void test_coplanar_inside() {
    std::cout << "  coplanar_inside" << std::endl;
    int F[4][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    ASSERT_TRUE(check_field_zero_coplanar(F));
}

void test_coplanar_outside() {
    std::cout << "  coplanar_outside" << std::endl;
    int F[4][3] = {{1,1,0},{2,1,0},{1,2,0},{2,2,0}};
    ASSERT_TRUE(!check_field_zero_coplanar(F));
}

void test_coplanar_all_zero() {
    std::cout << "  coplanar_all_zero" << std::endl;
    int F[4][3] = {{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
    ASSERT_TRUE(check_field_zero_coplanar(F));
}

void test_coplanar_1d_inside() {
    std::cout << "  coplanar_1d_inside" << std::endl;
    int F[4][3] = {{1,0,0},{2,0,0},{-1,0,0},{3,0,0}};
    ASSERT_TRUE(check_field_zero_coplanar(F));
}

void test_coplanar_1d_outside() {
    std::cout << "  coplanar_1d_outside" << std::endl;
    int F[4][3] = {{1,0,0},{2,0,0},{3,0,0},{4,0,0}};
    ASSERT_TRUE(!check_field_zero_coplanar(F));
}

void test_coplanar_3d_plane() {
    std::cout << "  coplanar_3d_plane (non-axis-aligned)" << std::endl;
    // Points in plane x+y+z=0: (1,-1,0),(0,1,-1),(-1,0,1),(1,0,-1)
    // Origin: does it lie in convex hull?
    // Triangle (1,-1,0),(0,1,-1),(-1,0,1): project dropping max |n|
    // n = (1,-1,0)×(0,1,-1) = (1,1,1), drop any coord, say z
    // 2D: (1,-1),(0,1),(-1,0)
    // s1 = 1*1-(-1)*0 = 1, s2 = 0*0-1*(-1) = 1, s3 = (-1)*(-1)-0*1 = 1
    // All positive → origin inside
    int F[4][3] = {{1,-1,0},{0,1,-1},{-1,0,1},{1,0,-1}};
    ASSERT_TRUE(check_field_zero_coplanar(F));
}

// ============================================================================
// Additional prem_i128 tests
// ============================================================================

void test_prem_equal_degree() {
    std::cout << "  prem_equal_degree" << std::endl;
    __int128 f[8] = {2, 0, 1};  // x^2+2
    __int128 g[8] = {1, 0, 1};  // x^2+1
    __int128 r[8] = {};
    int exp;
    int dr = prem_i128(f, 2, g, 2, r, &exp);
    ASSERT_EQ(dr, 0);
    ASSERT_TRUE(r[0] > 0);  // prem = g[2]^1*(f-g) = 1 > 0
}

void test_prem_exact_division() {
    std::cout << "  prem_exact_division" << std::endl;
    // f = x^2-1, g = x-1 (divides f), prem = 0
    __int128 f[8] = {-1, 0, 1};
    __int128 g[8] = {-1, 1};
    __int128 r[8] = {};
    int exp;
    int dr = prem_i128(f, 2, g, 1, r, &exp);
    // Content-reduced result should be 0
    ASSERT_EQ(r[0], (__int128)0);
}

void test_prem_neg_leading() {
    std::cout << "  prem_neg_leading" << std::endl;
    __int128 f[8] = {1, 0, 1};  // x^2+1
    __int128 g[8] = {3, -1};    // -x+3
    __int128 r[8] = {};
    int exp;
    int dr = prem_i128(f, 2, g, 1, r, &exp);
    ASSERT_EQ(dr, 0);
    ASSERT_EQ(exp, 2);
    // prem should evaluate to f(3) = 10, content-reduced
    ASSERT_TRUE(r[0] > 0);
}

// ============================================================================
// Additional signs_at_roots_i128 tests
// ============================================================================

void test_signs_quadratic_linear_shared() {
    std::cout << "  signs_quadratic_linear_shared" << std::endl;
    // f = (x-2)(x-5) = x^2-7x+10, g = x-2 => g(2)=0, g(5)=3
    __int128 f[8] = {10, -7, 1};
    __int128 g[8] = {-2, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 2, g, 1, signs, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs[0], 0);
    ASSERT_EQ(signs[1], 1);
}

void test_signs_quadratic_both_negative() {
    std::cout << "  signs_quadratic_both_negative" << std::endl;
    // f = (x-2)(x-5), g = x-10 => g(2)=-8, g(5)=-5
    __int128 f[8] = {10, -7, 1};
    __int128 g[8] = {-10, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 2, g, 1, signs, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs[0], -1);
    ASSERT_EQ(signs[1], -1);
}

void test_signs_cubic_3roots_shared_middle() {
    std::cout << "  signs_cubic_3roots_shared_middle" << std::endl;
    // f = (x-1)(x-3)(x-5), g = x-3 => g(1)=-2, g(3)=0, g(5)=2
    __int128 f[8] = {-15, 23, -9, 1};
    __int128 g[8] = {-3, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 1, signs, 3);
    ASSERT_EQ(nr, 3);
    ASSERT_EQ(signs[0], -1);
    ASSERT_EQ(signs[1], 0);
    ASSERT_EQ(signs[2], 1);
}

void test_signs_cubic_1root_quadratic_g() {
    std::cout << "  signs_cubic_1root_quadratic_g" << std::endl;
    // f = x^3+x+2 (root at -1), g = x^2+x+2 (no real roots)
    // g(-1) = 1-1+2 = 2 > 0
    __int128 f[8] = {2, 1, 0, 1};
    __int128 g[8] = {2, 1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 2, signs, 3);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(signs[0], 1);
}

void test_signs_quadratic_prem_reduction() {
    std::cout << "  signs_quadratic_prem_reduction" << std::endl;
    // f = (x-1)(x-4) = x^2-5x+4, g = (x-2)(x+3) = x^2+x-6
    // g(1) = -4, g(4) = 14
    __int128 f[8] = {4, -5, 1};
    __int128 g[8] = {-6, 1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 2, g, 2, signs, 3);
    ASSERT_EQ(nr, 2);
    ASSERT_EQ(signs[0], -1);
    ASSERT_EQ(signs[1], 1);
}

void test_signs_zero_polynomial() {
    std::cout << "  signs_zero_polynomial" << std::endl;
    __int128 f[8] = {0, 0, 0, 0};
    __int128 g[8] = {1, 1};
    int signs[3] = {};
    int nr = signs_at_roots_i128(f, 3, g, 1, signs, 3);
    ASSERT_EQ(nr, 0);
}

// ============================================================================
// Additional resultant_sign_i128 tests
// ============================================================================

void test_resultant_constant_f() {
    std::cout << "  resultant_constant_f" << std::endl;
    __int128 f[8] = {5};
    __int128 g[8] = {-3, 1};
    int res = resultant_sign_i128(f, 0, g, 1);
    ASSERT_EQ(res, 1);  // 5^1 = 5 > 0
}

void test_resultant_constant_f_neg() {
    std::cout << "  resultant_constant_f_neg" << std::endl;
    __int128 f[8] = {-5};
    __int128 g[8] = {-3, 1};
    int res = resultant_sign_i128(f, 0, g, 1);
    ASSERT_EQ(res, -1);  // (-5)^1 = -5 < 0
}

void test_resultant_constant_f_zero() {
    std::cout << "  resultant_constant_f_zero" << std::endl;
    __int128 f[8] = {0};
    __int128 g[8] = {-3, 1};
    int res = resultant_sign_i128(f, 0, g, 1);
    ASSERT_EQ(res, 0);
}

void test_resultant_both_constants() {
    std::cout << "  resultant_both_constants" << std::endl;
    __int128 f[8] = {3};
    __int128 g[8] = {7};
    int res = resultant_sign_i128(f, 0, g, 0);
    ASSERT_EQ(res, 1);
}

// ============================================================================
// Additional poly_gcd_full_i128 tests
// ============================================================================

void test_gcd_f_zero() {
    std::cout << "  gcd_f_zero" << std::endl;
    __int128 f[8] = {0, 0, 0};
    __int128 g[8] = {-2, 1};
    __int128 h[8] = {};
    int dh = poly_gcd_full_i128(f, 2, g, 1, h);
    ASSERT_EQ(dh, 1);
    ASSERT_TRUE(h[1] != 0);
}

void test_gcd_quadratic_factor() {
    std::cout << "  gcd_quadratic_factor" << std::endl;
    // (x^2+1)(x-1), (x^2+1)(x+2) → gcd = x^2+1
    __int128 f[8] = {-1, 1, -1, 1};
    __int128 g[8] = {2, 1, 2, 1};
    __int128 h[8] = {};
    int dh = poly_gcd_full_i128(f, 3, g, 3, h);
    ASSERT_EQ(dh, 2);
    ASSERT_EQ(h[1], (__int128)0);  // no linear term in x^2+1
}

void test_gcd_swap_degrees() {
    std::cout << "  gcd_swap_degrees" << std::endl;
    // f has lower degree → triggers swap
    __int128 f[8] = {-1, 1};       // x-1
    __int128 g[8] = {-2, 1, 1};    // (x-1)(x+2) = x^2+x-2
    __int128 h[8] = {};
    int dh = poly_gcd_full_i128(f, 1, g, 2, h);
    ASSERT_EQ(dh, 1);
    ASSERT_EQ(h[0] + h[1], (__int128)0);  // h(1) = 0
}

// ============================================================================
// Main
// ============================================================================

static void test_classify_2d_cases();  // forward declaration

// ============================================================================
// RP1 sign helper tests
// ============================================================================

void test_sign_at_plus_inf() {
    std::cout << "  sign_at_plus_inf (degree 1/2/3)" << std::endl;
    // Linear: p(x) = -3 + 2x, leading coeff +2 → +1 at +∞
    { __int128 p[] = {-3, 2}; ASSERT_EQ(sign_at_plus_inf_i128(p, 1), 1); }
    // Linear: p(x) = 5 - 7x, leading coeff -7 → -1 at +∞
    { __int128 p[] = {5, -7}; ASSERT_EQ(sign_at_plus_inf_i128(p, 1), -1); }
    // Quadratic: p(x) = 1 - 2x + 3x², leading coeff +3 → +1 at +∞
    { __int128 p[] = {1, -2, 3}; ASSERT_EQ(sign_at_plus_inf_i128(p, 2), 1); }
    // Quadratic: leading coeff -3 → -1 at +∞
    { __int128 p[] = {1, 2, -3}; ASSERT_EQ(sign_at_plus_inf_i128(p, 2), -1); }
    // Cubic: leading coeff +1 → +1
    { __int128 p[] = {-1, 0, 0, 1}; ASSERT_EQ(sign_at_plus_inf_i128(p, 3), 1); }
    // Constant: p(x) = -5
    { __int128 p[] = {-5}; ASSERT_EQ(sign_at_plus_inf_i128(p, 0), -1); }
    // Degree drops: p[2] = 0, effective linear
    { __int128 p[] = {1, 3, 0}; ASSERT_EQ(sign_at_plus_inf_i128(p, 2), 1); }
}

void test_sign_at_inf() {
    std::cout << "  sign_at_minus_inf (degree 1/2/3)" << std::endl;
    // Linear: p(x) = 2x, leading coeff +2, odd deg → -1 at -∞
    { __int128 p[] = {0, 2}; ASSERT_EQ(sign_at_inf_i128(p, 1), -1); }
    // Linear: p(x) = -7x, leading coeff -7, odd deg → +1 at -∞
    { __int128 p[] = {0, -7}; ASSERT_EQ(sign_at_inf_i128(p, 1), 1); }
    // Quadratic: p(x) = 3x², even deg → +1 at -∞ (same as +∞)
    { __int128 p[] = {0, 0, 3}; ASSERT_EQ(sign_at_inf_i128(p, 2), 1); }
    // Quadratic: leading coeff -3, even deg → -1 at -∞
    { __int128 p[] = {0, 0, -3}; ASSERT_EQ(sign_at_inf_i128(p, 2), -1); }
    // Cubic: leading coeff +1, odd deg → -1 at -∞
    { __int128 p[] = {0, 0, 0, 1}; ASSERT_EQ(sign_at_inf_i128(p, 3), -1); }
    // Cubic: leading coeff -2, odd deg → +1 at -∞
    { __int128 p[] = {0, 0, 0, -2}; ASSERT_EQ(sign_at_inf_i128(p, 3), 1); }
}

void test_sign_just_after_root() {
    std::cout << "  sign_just_after_root (degree 1/2/3)" << std::endl;
    // Linear: p(x) = x - 1, lc=+1, 1 root, root_idx=0 → exp = 0 → lc = +1
    { __int128 p[] = {-1, 1}; ASSERT_EQ(sign_just_after_root_i128(p, 1, 1, 0), 1); }
    // Linear: p(x) = -x + 1, lc=-1, 1 root, root_idx=0 → exp = 0 → -1
    { __int128 p[] = {1, -1}; ASSERT_EQ(sign_just_after_root_i128(p, 1, 1, 0), -1); }
    // Quadratic with 2 roots: p(x) = (x-1)(x-3) = x²-4x+3, lc=+1, 2 roots
    // Just after root 0 (smaller): exp = 2-1-0 = 1 → -lc = -1
    { __int128 p[] = {3, -4, 1}; ASSERT_EQ(sign_just_after_root_i128(p, 2, 2, 0), -1); }
    // Just after root 1 (larger): exp = 2-1-1 = 0 → lc = +1
    { __int128 p[] = {3, -4, 1}; ASSERT_EQ(sign_just_after_root_i128(p, 2, 2, 1), 1); }
    // Cubic with 3 roots: p(x) = (x-1)(x-2)(x-3) = x³-6x²+11x-6, lc=+1
    // After root 0: exp = 3-1-0 = 2 → lc = +1
    { __int128 p[] = {-6, 11, -6, 1}; ASSERT_EQ(sign_just_after_root_i128(p, 3, 3, 0), 1); }
    // After root 1: exp = 3-1-1 = 1 → -lc = -1
    { __int128 p[] = {-6, 11, -6, 1}; ASSERT_EQ(sign_just_after_root_i128(p, 3, 3, 1), -1); }
    // After root 2: exp = 3-1-2 = 0 → lc = +1
    { __int128 p[] = {-6, 11, -6, 1}; ASSERT_EQ(sign_just_after_root_i128(p, 3, 3, 2), 1); }
    // Negative leading coeff: p(x) = -(x-1)(x-3) = -x²+4x-3, lc=-1
    { __int128 p[] = {-3, 4, -1}; ASSERT_EQ(sign_just_after_root_i128(p, 2, 2, 0), 1); }
    { __int128 p[] = {-3, 4, -1}; ASSERT_EQ(sign_just_after_root_i128(p, 2, 2, 1), -1); }
}

void test_rp1_pairing_basic() {
    std::cout << "  pair_punctures_rp1: 2 punctures, 1 pair" << std::endl;
    // Simple 2D case: 2 punctures on different faces, same Q-interval
    // P_red[0] = x - 1 (root at 1), P_red[1] = x - 2 (root at 2), P_red[2] = 5 (no roots)
    // Q_red = 1 (constant positive)
    __int128 P_red[3][4] = {{-1, 1, 0, 0}, {-2, 1, 0, 0}, {5, 0, 0, 0}};
    int degP_red[] = {1, 1, 0};
    int n_distinct_red[] = {1, 1, 0};
    __int128 Q_red[] = {1, 0, 0, 0};
    int degQ_red = 0;

    int sorted[] = {0, 1};  // puncture 0 (face=0, root=0), puncture 1 (face=1, root=0)
    int p_face[] = {0, 1};
    int p_root_idx[] = {0, 0};
    int p_qi[] = {0, 0};

    RP1PairResult out[4];
    int n = pair_punctures_rp1(3, sorted, 2, 2, p_face, p_root_idx,
                                p_qi, 0, true,
                                P_red, degP_red, n_distinct_red,
                                Q_red, degQ_red, out, 4);
    // Should find 1 pair: between the two punctures (direct or complement)
    ASSERT_EQ(n, 1);
}

// ============================================================================
// Diverse coverage: 75 categories from comprehensive DB not yet in tests
// ============================================================================

void test_diverse_coverage() {
    std::cout << "  109 diverse coverage cases (75 comprehensive + 34 fresh scan)" << std::endl;
    #define RUNTAG(V, W, sd, exp_np, exp_npairs, exp_cat) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << sd \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
        { TetCaseV2GPU tv2; \
          memset(&tv2, 0, sizeof(tv2)); \
          for (int i=0;i<4;i++) for (int j=0;j<3;j++) { tv2.V[i][j]=V[i][j]; tv2.W[i][j]=W[i][j]; } \
          tv2.v2 = result; tv2.seed = sd; \
          for (int k=0;k<4;k++) tv2.disc_sign[k] = discriminant_sign_i128(P[k]); \
          ClassifiedCase cc = classify_case_v2(tv2); \
          total_tests++; \
          if (cc.category != std::string(exp_cat)) { \
              failed_tests++; \
              std::cerr << "FAILED: seed " << sd \
                        << " category=\"" << cc.category << "\" (exp \"" << (exp_cat) << "\")" \
                        << std::endl; \
          } else { passed_tests++; } \
        } \
    } while(0)

    // T0 cases
    {  int V[4][3] = {{-2,12,6},{2,-1,3},{8,-12,-4},{15,14,3}};
       int W[4][3] = {{19,-20,-16},{11,12,-16},{17,-12,-20},{16,-8,-13}};
       RUNTAG(V, W, 8052, 0, 0, "T0_Q2"); }
    {  int V[4][3] = {{-9,-17,-3},{-4,9,-18},{-16,-18,16},{0,5,-17}};
       int W[4][3] = {{-18,12,0},{-19,12,10},{20,12,4},{-11,12,-3}};
       RUNTAG(V, W, 36052, 0, 0, "T0_Q2-"); }
    {  int V[4][3] = {{-16,-18,-7},{-6,18,-15},{-19,19,-13},{-7,5,17}};
       int W[4][3] = {{10,0,4},{-10,0,-4},{16,-9,15},{-13,-20,17}};
       RUNTAG(V, W, 63168, 0, 0, "T0_Q3+_Cw1"); }
    {  int V[4][3] = {{-17,-2,0},{7,20,-17},{13,3,17},{-13,-4,6}};
       int W[4][3] = {{16,17,8},{16,17,-13},{6,-16,-4},{13,4,-6}};
       RUNTAG(V, W, 17233, 0, 0, "T0_Q3+_D00"); }
    {  int V[4][3] = {{4,1,14},{16,-5,2},{-16,5,-2},{-20,17,15}};
       int W[4][3] = {{3,14,1},{-3,11,-10},{-5,9,17},{9,17,-18}};
       RUNTAG(V, W, 8148, 0, 0, "T0_Q3-_Cv1"); }
    {  int V[4][3] = {{-19,17,16},{-4,-19,-5},{-19,-10,-12},{-6,-16,-19}};
       int W[4][3] = {{-19,17,16},{-3,12,1},{-5,-1,4},{8,20,-17}};
       RUNTAG(V, W, 4325, 0, 0, "T0_Q3-_D00"); }

    // T1 cases (odd T with waypoint tags — legitimate)
    {  int V[4][3] = {{-7,-5,1},{-20,-19,17},{13,5,7},{-10,16,-4}};
       int W[4][3] = {{11,9,8},{0,0,0},{-9,5,-8},{12,-9,-3}};
       RUNTAG(V, W, 72661, 1, 0, "T1_Q3+_Cw0_D00"); }
    {  int V[4][3] = {{7,-1,-7},{-7,1,7},{17,12,9},{6,7,-13}};
       int W[4][3] = {{-3,5,-15},{19,-3,-2},{-6,-15,18},{-7,14,1}};
       RUNTAG(V, W, 40278, 1, 0, "T1_Q3-_Cv1_Cw"); }

    // T2 cases
    {  int V[4][3] = {{8,-14,-11},{0,11,10},{-13,13,7},{-10,16,15}};
       int W[4][3] = {{7,-3,-19},{11,17,3},{-3,8,0},{-1,-7,10}};
       RUNTAG(V, W, 17657, 2, 1, "T2_(1,1)_Q3+_Cv_Cw"); }
    {  int V[4][3] = {{-8,12,13},{-19,-20,-17},{-10,7,16},{7,7,-7}};
       int W[4][3] = {{20,8,-4},{-1,-14,-10},{0,18,-16},{-20,-17,12}};
       RUNTAG(V, W, 206, 2, 1, "T2_(1,1)_Q3+_Cw2"); }
    {  int V[4][3] = {{6,-7,4},{1,-19,-8},{-6,-13,-17},{-6,7,-4}};
       int W[4][3] = {{-7,15,-7},{-11,-17,20},{17,-5,20},{11,-16,-3}};
       RUNTAG(V, W, 56346, 2, 1, "T2_(1,1)_Q3-_Cv1_Cw"); }
    {  int V[4][3] = {{-20,-7,7},{19,-16,0},{-13,19,14},{4,19,-7}};
       int W[4][3] = {{-7,10,-18},{12,-15,14},{-19,2,16},{2,10,1}};
       RUNTAG(V, W, 13104, 2, 1, "T2_(1,1)_Q3-_Cv_Cw"); }
    {  int V[4][3] = {{13,-19,-2},{11,9,20},{4,19,8},{-12,-13,-12}};
       int W[4][3] = {{4,2,4},{-19,-7,7},{-12,10,-4},{-10,-19,-17}};
       RUNTAG(V, W, 3220, 2, 1, "T2_(1,1)_Q3-_Cv_Cw2"); }
    {  int V[4][3] = {{11,-14,8},{8,14,-19},{7,-8,2},{-14,2,-3}};
       int W[4][3] = {{-6,9,5},{0,-17,-17},{9,0,6},{-2,-10,6}};
       RUNTAG(V, W, 5914, 2, 1, "T2_(1,1)_Q3-_Cw2"); }
    {  int V[4][3] = {{-7,10,11},{16,19,20},{10,10,10},{-11,14,2}};
       int W[4][3] = {{-19,-2,-7},{14,-15,-18},{3,3,3},{12,-13,7}};
       RUNTAG(V, W, 27798, 2, 1, "T2_(1,1)_Q3-_Cw_D00"); }
    {  int V[4][3] = {{2,0,-10},{1,-16,-5},{2,-20,5},{-14,12,19}};
       int W[4][3] = {{-17,-16,-10},{-7,-9,9},{-10,11,15},{-14,15,11}};
       RUNTAG(V, W, 38676, 2, 1, "T2_Q2-"); }
    {  int V[4][3] = {{-2,-12,19},{-10,-20,-1},{3,12,13},{3,10,-16}};
       int W[4][3] = {{12,10,-15},{-3,20,15},{14,9,0},{3,16,3}};
       RUNTAG(V, W, 20916, 2, 1, "T2_Q2-_Cv"); }
    {  int V[4][3] = {{16,11,-18},{-8,18,12},{-20,-17,7},{-15,-14,18}};
       int W[4][3] = {{-19,-14,17},{1,12,12},{-20,-18,17},{-1,4,12}};
       RUNTAG(V, W, 39074, 2, 1, "T2_Q2_Cv"); }
    {  int V[4][3] = {{16,20,17},{-16,4,18},{2,-12,-9},{-1,-12,-17}};
       int W[4][3] = {{-1,16,0},{-8,2,9},{2,14,-12},{-17,2,20}};
       RUNTAG(V, W, 47889, 2, 1, "T2_Q3+_Cv_D00"); }
    {  int V[4][3] = {{-7,9,-10},{-17,-17,-13},{-4,7,16},{0,0,0}};
       int W[4][3] = {{-10,-5,-20},{8,14,-6},{3,4,18},{12,11,-10}};
       RUNTAG(V, W, 6452, 2, 1, "T2_Q3-_Cv0_D00"); }
    {  int V[4][3] = {{9,-4,0},{0,7,9},{-15,-1,-2},{-5,12,-10}};
       int W[4][3] = {{-9,-14,-18},{-1,-7,6},{9,14,18},{8,-2,16}};
       RUNTAG(V, W, 24230, 2, 1, "T2_Q3-_Cv_Cw1"); }
    {  int V[4][3] = {{-7,-5,12},{0,-9,-14},{-1,7,-18},{-18,-16,11}};
       int W[4][3] = {{0,0,0},{-14,5,16},{-17,0,-14},{-5,-20,-10}};
       RUNTAG(V, W, 15765, 2, 1, "T2_Q3-_Cw0_D00"); }
    {  int V[4][3] = {{7,-10,19},{3,-9,3},{-2,-16,11},{6,-15,-20}};
       int W[4][3] = {{3,16,2},{5,-12,-11},{5,-6,18},{-3,-16,-2}};
       RUNTAG(V, W, 42369, 2, 1, "T2_Q3-_Cw1"); }
    {  int V[4][3] = {{18,-19,7},{-4,-14,15},{-3,-12,-8},{-8,6,17}};
       int W[4][3] = {{12,-3,-17},{-3,-4,-10},{12,16,13},{-9,-12,6}};
       RUNTAG(V, W, 22690, 2, 1, "T2_Q3-_Cw2"); }
    {  int V[4][3] = {{-9,-18,-17},{5,5,8},{15,-10,-20},{3,5,-19}};
       int W[4][3] = {{-1,10,10},{20,-19,3},{-12,8,16},{17,-15,15}};
       RUNTAG(V, W, 25538, 2, 1, "T2_Q3-_D00"); }

    // T4 cases
    {  int V[4][3] = {{-7,-9,-1},{10,-8,-18},{2,13,10},{-13,6,-20}};
       int W[4][3] = {{-19,-1,-13},{9,9,15},{14,-10,-14},{-2,2,20}};
       RUNTAG(V, W, 11671, 4, 2, "T4_(1,1,2)_Q3+_Cv2_Cw"); }
    {  int V[4][3] = {{3,1,6},{2,9,-18},{-18,17,10},{-20,-20,-7}};
       int W[4][3] = {{14,4,-1},{-9,-7,12},{1,3,-15},{-13,-7,19}};
       RUNTAG(V, W, 88479, 4, 2, "T4_(1,1,2)_Q3+_Cv_Cw2"); }
    {  int V[4][3] = {{12,-6,-5},{19,0,-18},{-18,12,10},{17,-14,16}};
       int W[4][3] = {{-10,-20,-5},{-14,4,-1},{-6,0,-20},{13,20,15}};
       RUNTAG(V, W, 32400, 4, 2, "T4_(1,1,2)_Q3+_Cw2"); }
    {  int V[4][3] = {{-20,-17,0},{-14,-20,-17},{20,8,13},{17,20,9}};
       int W[4][3] = {{15,-1,-9},{2,3,11},{-17,1,9},{16,-6,16}};
       RUNTAG(V, W, 45853, 4, 2, "T4_(1,3)_Q3+_Cv_Cw"); }
    {  int V[4][3] = {{-15,17,-1},{-19,-15,12},{-6,-17,11},{7,-12,-19}};
       int W[4][3] = {{19,-20,-9},{1,-5,-6},{-8,7,-2},{-7,15,14}};
       RUNTAG(V, W, 35760, 4, 2, "T4_(1,3)_Q3+_Cw2"); }
    {  int V[4][3] = {{-15,-14,5},{-13,4,-12},{9,12,15},{-10,-14,-20}};
       int W[4][3] = {{3,-19,12},{1,-3,-11},{-17,5,10},{17,8,19}};
       RUNTAG(V, W, 12004, 4, 2, "T4_(1,3)_Q3-_Cv2_Cw"); }
    {  int V[4][3] = {{-4,-5,0},{2,12,20},{14,6,-19},{-2,9,17}};
       int W[4][3] = {{15,8,17},{6,-2,-11},{-4,-13,-7},{-10,11,12}};
       RUNTAG(V, W, 9486, 4, 2, "T4_(1,3)_Q3-_Cv_Cw"); }
    {  int V[4][3] = {{-15,-19,9},{6,6,-4},{-6,-17,13},{6,20,-16}};
       int W[4][3] = {{15,-12,-6},{0,0,13},{5,-5,-1},{-18,18,-2}};
       RUNTAG(V, W, 9419, 4, 2, "T4_(1,3)_Q3-_Cw2"); }
    {  int V[4][3] = {{-9,1,-2},{10,-5,1},{-18,19,16},{5,-13,3}};
       int W[4][3] = {{18,-2,4},{6,-13,-2},{-14,-19,18},{-9,6,-5}};
       RUNTAG(V, W, 18846, 4, 2, "T4_(1,3)_Q3-_Cw_D00"); }
    {  int V[4][3] = {{3,10,-8},{16,3,10},{-15,-5,7},{-3,-10,8}};
       int W[4][3] = {{6,20,-5},{0,1,-12},{18,11,17},{6,-14,-15}};
       RUNTAG(V, W, 30822, 3, 1, "T4_(2,2)_Q3+_Cv1"); }
    {  int V[4][3] = {{-6,-18,-7},{4,-13,-15},{-6,9,14},{12,9,-7}};
       int W[4][3] = {{7,12,2},{-15,-17,0},{7,11,19},{2,9,20}};
       RUNTAG(V, W, 10588, 4, 2, "T4_(2,2)_Q3+_Cv2"); }
    {  int V[4][3] = {{-14,-2,-19},{-12,-6,16},{11,12,-7},{-3,-12,5}};
       int W[4][3] = {{9,-8,-9},{4,1,5},{-7,11,16},{0,0,0}};
       RUNTAG(V, W, 69263, 4, 2, "T4_(2,2)_Q3+_Cv_Cw0_D00"); }
    {  int V[4][3] = {{9,8,-16},{14,-14,-10},{16,-14,7},{-18,13,14}};
       int W[4][3] = {{12,-2,-16},{-14,-1,-4},{-6,18,14},{-9,-7,9}};
       RUNTAG(V, W, 30191, 4, 2, "T4_(2,2)_Q3+_Cv_Cw2"); }
    {  int V[4][3] = {{7,4,2},{19,-13,17},{1,-3,10},{10,7,-19}};
       int W[4][3] = {{-18,-16,-15},{0,0,0},{19,-9,-6},{-12,11,19}};
       RUNTAG(V, W, 57320, 3, 1, "T4_(2,2)_Q3+_Cw0_D00"); }
    {  int V[4][3] = {{12,-3,12},{-7,-18,-10},{18,-4,19},{0,-16,-6}};
       int W[4][3] = {{16,-18,8},{2,-13,-1},{-2,1,-5},{4,0,11}};
       RUNTAG(V, W, 12342, 4, 2, "T4_(2,2)_Q3+_Cw2"); }
    {  int V[4][3] = {{-7,19,-2},{-15,-20,-10},{19,-3,9},{-6,-13,0}};
       int W[4][3] = {{13,1,-16},{-6,17,14},{-17,-20,12},{9,15,-14}};
       RUNTAG(V, W, 7802, 4, 2, "T4_(2,2)_Q3-_Cv"); }
    {  int V[4][3] = {{19,18,16},{7,-3,-9},{-1,1,3},{-20,3,9}};
       int W[4][3] = {{-6,10,13},{20,13,2},{-13,4,-9},{-12,-17,15}};
       RUNTAG(V, W, 56067, 4, 2, "T4_(2,2)_Q3-_Cv2"); }
    {  int V[4][3] = {{-18,-12,10},{13,15,-15},{12,-9,-1},{15,10,3}};
       int W[4][3] = {{18,1,-5},{16,-1,-18},{12,-9,-1},{20,-16,12}};
       RUNTAG(V, W, 17231, 4, 2, "T4_(2,2)_Q3-_Cv_D00"); }
    {  int V[4][3] = {{10,-2,20},{15,10,3},{-9,7,3},{4,3,-11}};
       int W[4][3] = {{0,-14,-14},{2,-1,18},{-14,-2,20},{14,16,-6}};
       RUNTAG(V, W, 23771, 4, 2, "T4_(2,2)_Q3-_Cw2"); }
    {  int V[4][3] = {{-15,-13,18},{6,0,10},{2,-8,-1},{6,2,-15}};
       int W[4][3] = {{1,-19,9},{-6,0,-10},{-2,-17,-20},{-14,2,11}};
       RUNTAG(V, W, 23341, 4, 2, "T4_(2,2)_Q3-_D00"); }
    {  int V[4][3] = {{-6,-6,10},{-3,-7,-18},{6,-7,2},{-3,14,-10}};
       int W[4][3] = {{3,4,12},{10,-3,20},{-1,8,10},{-5,12,-20}};
       RUNTAG(V, W, 32605, 4, 2, "T4_Q2-_Cv"); }
    {  int V[4][3] = {{12,5,11},{-12,-6,-14},{-14,1,6},{-16,-12,9}};
       int W[4][3] = {{0,-3,-7},{-7,-3,-7},{-7,-11,-11},{-10,-1,-6}};
       RUNTAG(V, W, 62420, 4, 2, "T4_Q2_Cv"); }
    {  int V[4][3] = {{11,-10,-20},{-2,17,-15},{4,-18,19},{-15,-20,-9}};
       int W[4][3] = {{13,14,14},{18,20,-16},{19,10,6},{-5,13,17}};
       RUNTAG(V, W, 10008, 4, 2, "T4_Q3+_Cv"); }
    {  int V[4][3] = {{-4,-19,19},{-8,12,-12},{-13,18,2},{10,3,-3}};
       int W[4][3] = {{-8,-2,2},{-8,8,19},{-12,6,-16},{1,-13,-8}};
       RUNTAG(V, W, 6770, 4, 2, "T4_Q3+_Cv2"); }
    {  int V[4][3] = {{-5,-3,3},{9,14,-13},{-6,12,-4},{16,-13,8}};
       int W[4][3] = {{-6,-8,-13},{10,14,10},{-16,1,-11},{2,12,-17}};
       RUNTAG(V, W, 12885, 4, 2, "T4_Q3-_Cv"); }
    {  int V[4][3] = {{10,8,-13},{-10,-8,13},{-18,16,-18},{9,7,-19}};
       int W[4][3] = {{-5,0,12},{-15,-1,-5},{-7,-12,-9},{7,10,-12}};
       RUNTAG(V, W, 16927, 4, 2, "T4_Q3-_Cv1"); }
    {  int V[4][3] = {{6,0,3},{-14,-6,-2},{5,3,0},{4,-10,-10}};
       int W[4][3] = {{-17,-8,-17},{-1,-6,7},{-6,-6,9},{16,5,-3}};
       RUNTAG(V, W, 7908, 4, 2, "T4_Q3-_Cv2"); }
    {  int V[4][3] = {{8,-9,17},{19,14,-12},{-10,8,-17},{-14,20,7}};
       int W[4][3] = {{-10,15,-12},{1,14,-17},{-11,-16,7},{12,7,6}};
       RUNTAG(V, W, 54580, 4, 2, "T4_Q3-_Cv_Cw2"); }

    // T6 cases
    {  int V[4][3] = {{-8,-2,-16},{-15,0,19},{7,-6,9},{-10,8,-12}};
       int W[4][3] = {{-20,2,-8},{-6,1,3},{8,1,-2},{8,-8,17}};
       RUNTAG(V, W, 13886, 6, 3, "T6_(1,1,2,2)_Q3+_Cw"); }
    {  int V[4][3] = {{-11,18,-10},{-3,-6,-16},{-10,-20,17},{-10,9,10}};
       int W[4][3] = {{9,-8,-8},{-9,-16,-16},{-6,14,14},{-20,19,16}};
       RUNTAG(V, W, 13591, 6, 3, "T6_(1,1,2,2)_Q3+_Cw2"); }
    {  int V[4][3] = {{-14,2,-15},{-17,1,8},{17,4,16},{10,-6,15}};
       int W[4][3] = {{17,-19,16},{-6,-9,16},{18,12,18},{-16,14,-17}};
       RUNTAG(V, W, 11780, 6, 3, "T6_(1,1,4)_Q3+_Cw"); }
    {  int V[4][3] = {{8,9,-17},{17,-5,1},{11,-3,15},{-18,9,16}};
       int W[4][3] = {{8,16,1},{10,-16,2},{0,-18,-5},{-18,19,-3}};
       RUNTAG(V, W, 292, 6, 3, "T6_(1,2,3)_Q3+_Cw"); }
    {  int V[4][3] = {{-11,5,17},{-6,3,14},{-5,-13,-3},{12,-4,11}};
       int W[4][3] = {{-9,10,-1},{-9,-11,-10},{14,8,15},{-13,4,-19}};
       RUNTAG(V, W, 50279, 6, 3, "T6_(1,5)_Q3+_Cw"); }
    {  int V[4][3] = {{-7,-12,-11},{4,9,-1},{7,-5,-4},{-2,16,14}};
       int W[4][3] = {{14,12,-11},{3,-13,-18},{-12,-7,1},{-2,6,14}};
       RUNTAG(V, W, 60034, 6, 3, "T6_(1,5)_Q3-_Cv2_Cw"); }
    {  int V[4][3] = {{-8,-1,-5},{17,15,0},{19,-17,-9},{6,-7,4}};
       int W[4][3] = {{3,-19,11},{3,9,20},{-16,18,-5},{12,-5,-19}};
       RUNTAG(V, W, 11841, 6, 3, "T6_(1,5)_Q3-_Cw"); }
    {  int V[4][3] = {{3,20,8},{16,-1,4},{-1,7,11},{0,-10,-10}};
       int W[4][3] = {{-19,17,9},{-10,-7,17},{18,4,-11},{-11,9,12}};
       RUNTAG(V, W, 12514, 6, 3, "T6_(2,2,2)_Q3+_Cv2"); }
    {  int V[4][3] = {{15,-18,20},{-9,3,-15},{-1,-5,20},{17,5,7}};
       int W[4][3] = {{-8,9,-8},{0,-20,0},{11,10,6},{4,16,4}};
       RUNTAG(V, W, 4632, 6, 3, "T6_(2,2,2)_Q3+_Cv_Cw2"); }
    {  int V[4][3] = {{13,8,5},{-1,-3,14},{-14,-20,17},{6,6,-9}};
       int W[4][3] = {{8,-17,-12},{-3,0,-1},{13,-2,13},{8,5,-19}};
       RUNTAG(V, W, 49083, 6, 3, "T6_(2,2,2)_Q3+_Cw2"); }
    {  int V[4][3] = {{15,-5,-9},{-4,14,-5},{3,17,14},{-15,-10,-4}};
       int W[4][3] = {{-18,13,-1},{15,-17,2},{5,18,13},{-12,19,3}};
       RUNTAG(V, W, 9084, 6, 3, "T6_(2,4)_Q3+_Cv"); }
    {  int V[4][3] = {{4,-8,-8},{6,12,18},{20,-17,-12},{-18,-8,-19}};
       int W[4][3] = {{10,16,-11},{-1,-20,20},{4,10,-6},{-19,0,-2}};
       RUNTAG(V, W, 52738, 6, 3, "T6_(2,4)_Q3+_Cv2"); }
    {  int V[4][3] = {{-11,-1,-19},{16,-6,12},{12,-1,2},{-2,-9,-17}};
       int W[4][3] = {{6,2,20},{-10,16,15},{5,-9,-10},{-5,11,-20}};
       RUNTAG(V, W, 96613, 6, 3, "T6_(2,4)_Q3+_Cw2"); }
    {  int V[4][3] = {{12,-2,-5},{-16,18,19},{-5,-10,0},{-15,13,-4}};
       int W[4][3] = {{-15,15,17},{-11,-12,15},{7,1,6},{1,12,-20}};
       RUNTAG(V, W, 10861, 6, 3, "T6_(2,4)_Q3-_Cv"); }
    {  int V[4][3] = {{11,-4,-3},{-1,17,-10},{14,-5,-17},{-6,0,-19}};
       int W[4][3] = {{16,-5,-9},{-16,-1,12},{16,15,-19},{-15,-6,2}};
       RUNTAG(V, W, 44300, 6, 3, "T6_(2,4)_Q3-_Cw2"); }
    {  int V[4][3] = {{-2,-8,-11},{3,20,10},{-13,-19,14},{17,-18,-15}};
       int W[4][3] = {{-17,-2,-3},{-13,18,-7},{12,-6,4},{-6,-7,-20}};
       RUNTAG(V, W, 7610, 6, 3, "T6_(3,3)_Q3-_Cv_Cw"); }
    {  int V[4][3] = {{-19,3,9},{-7,7,19},{7,-7,16},{11,1,-18}};
       int W[4][3] = {{16,5,-13},{19,-5,-7},{9,2,18},{-19,-19,12}};
       RUNTAG(V, W, 1154, 6, 3, "T6_Q3-_Cv"); }
    {  int V[4][3] = {{16,-16,-8},{-6,-4,-12},{-1,8,11},{9,-16,17}};
       int W[4][3] = {{6,-13,-3},{5,18,7},{16,-20,-8},{-14,-3,13}};
       RUNTAG(V, W, 15318, 6, 3, "T6_Q3-_Cv2"); }

    // T8 cases
    {  int V[4][3] = {{-10,-6,-15},{-14,17,-2},{-18,0,5},{8,3,10}};
       int W[4][3] = {{4,16,7},{-11,13,4},{-4,-9,9},{8,1,-19}};
       RUNTAG(V, W, 40471, 8, 4, "T8_(1,7)_Q3-_Cv_Cw"); }
    {  int V[4][3] = {{6,-6,3},{-16,11,-1},{-6,17,-14},{5,-20,-5}};
       int W[4][3] = {{-19,-19,18},{-16,3,17},{17,-12,20},{13,-3,-17}};
       RUNTAG(V, W, 19653, 8, 4, "T8_(2,6)_Q3-_Cv"); }
    {  int V[4][3] = {{20,2,20},{-8,-1,-12},{-7,18,-17},{-12,-13,14}};
       int W[4][3] = {{7,15,7},{-7,16,-11},{-3,-5,3},{18,-1,3}};
       RUNTAG(V, W, 11734, 8, 4, "T8_Q3-_Cv"); }
    {  int V[4][3] = {{20,-8,-18},{-19,17,20},{14,-13,14},{-9,6,-1}};
       int W[4][3] = {{-11,-11,11},{17,16,-16},{-13,10,-11},{3,11,-11}};
       RUNTAG(V, W, 96555, 8, 4, "T8_Q3-_Cv_Cw2"); }

    // --- 34 NEW categories from fresh 500M scan (seed 12345, R=20) ---
    {  int V[4][3] = {{7,17,10},{0,0,0},{12,17,9},{-3,-3,19}};
       int W[4][3] = {{9,-9,-11},{7,6,-13},{-17,16,0},{1,-10,16}};
       RUNTAG(V, W, 106239, 2, 1, "T2_(1,1)_Q3+_Cv0_Cw_D00"); }
    {  int V[4][3] = {{20,13,-9},{9,18,6},{-9,-18,-6},{6,12,-2}};
       int W[4][3] = {{14,4,-20},{-7,-13,-3},{-6,4,-11},{3,-1,15}};
       RUNTAG(V, W, 100554, 2, 1, "T2_(1,1)_Q3+_Cv1_Cw"); }
    {  int V[4][3] = {{-10,20,14},{-7,-16,8},{-19,5,14},{15,2,-12}};
       int W[4][3] = {{-6,6,2},{19,19,-5},{8,-14,-1},{-18,-6,0}};
       RUNTAG(V, W, 58432, 2, 1, "T2_(1,1)_Q3-_Cv2_Cw"); }
    {  int V[4][3] = {{-4,6,4},{16,-6,2},{-14,-11,4},{12,-18,-12}};
       int W[4][3] = {{4,-1,9},{-12,11,-5},{2,17,3},{18,6,13}};
       RUNTAG(V, W, 4707, 1, 0, "T2_Q3+_Cv1"); }
    {  int V[4][3] = {{10,-2,3},{-6,18,-6},{20,1,-19},{-3,-11,4}};
       int W[4][3] = {{0,0,0},{4,5,5},{15,-14,15},{-2,-14,5}};
       RUNTAG(V, W, 4147, 2, 1, "T2_Q3+_Cv_Cw0_D00"); }
    {  int V[4][3] = {{-5,-5,-3},{-1,-7,8},{8,-14,-4},{8,20,-1}};
       int W[4][3] = {{10,10,6},{7,-10,19},{14,-13,-20},{19,-5,11}};
       RUNTAG(V, W, 101439, 2, 1, "T2_Q3-_Cv_D00"); }
    {  int V[4][3] = {{-11,13,14},{-1,4,-13},{0,-14,-7},{-1,-8,-12}};
       int W[4][3] = {{-20,17,-16},{-8,-3,-5},{0,-12,-6},{11,-5,10}};
       RUNTAG(V, W, 8208, 4, 2, "T4_(1,1,2)_Q3+_Cw_D00"); }
    {  int V[4][3] = {{7,7,-8},{-2,-20,7},{-15,17,-10},{0,-12,14}};
       int W[4][3] = {{-16,17,-2},{2,6,-13},{10,-20,20},{-10,20,-20}};
       RUNTAG(V, W, 29967, 4, 2, "T4_(1,3)_Q3-_Cv_Cw1"); }
    {  int V[4][3] = {{16,-7,-4},{-3,-2,-15},{19,-6,0},{-20,-7,14}};
       int W[4][3] = {{11,20,11},{15,-7,-18},{13,-13,0},{-1,1,0}};
       RUNTAG(V, W, 2702, 4, 2, "T4_(1,3)_Q3-_Cw1"); }
    {  int V[4][3] = {{8,-4,20},{-1,-2,-4},{-20,4,8},{4,18,-6}};
       int W[4][3] = {{15,15,-9},{7,-15,14},{-7,5,8},{-11,19,1}};
       RUNTAG(V, W, 92150, 4, 2, "T4_(2,2)_Q2_Cv"); }
    {  int V[4][3] = {{-7,-20,8},{-11,-1,-19},{13,-6,15},{-4,14,-8}};
       int W[4][3] = {{-19,-19,-18},{-4,6,10},{6,-9,-15},{20,-20,-2}};
       RUNTAG(V, W, 101349, 4, 2, "T4_(2,2)_Q3+_Cv_Cw1"); }
    {  int V[4][3] = {{1,-1,14},{-3,-12,13},{14,-10,-6},{-10,-8,-14}};
       int W[4][3] = {{0,-6,8},{-3,17,16},{0,6,-8},{10,17,-15}};
       RUNTAG(V, W, 90084, 4, 2, "T4_(2,2)_Q3+_Cw1"); }
    {  int V[4][3] = {{-15,18,2},{17,1,-20},{-14,-4,15},{-10,-17,-7}};
       int W[4][3] = {{15,-18,-2},{4,0,-8},{-16,-13,-19},{18,11,9}};
       RUNTAG(V, W, 10028, 4, 2, "T4_(2,2)_Q3+_D00"); }
    {  int V[4][3] = {{0,0,0},{-8,5,2},{3,0,3},{11,-5,-10}};
       int W[4][3] = {{-1,-19,2},{0,-8,-4},{8,15,12},{8,-1,17}};
       RUNTAG(V, W, 104980, 3, 1, "T4_(2,2)_Q3-_Cv0_D00"); }
    {  int V[4][3] = {{0,-13,10},{-11,-8,4},{-10,-10,2},{5,5,-1}};
       int W[4][3] = {{-6,13,12},{-16,6,-6},{13,-8,-6},{-2,-10,-13}};
       RUNTAG(V, W, 16852, 3, 1, "T4_(2,2)_Q3-_Cv1"); }
    {  int V[4][3] = {{-18,-20,9},{4,5,0},{19,1,-8},{-10,17,-3}};
       int W[4][3] = {{0,10,8},{0,-10,-8},{-5,12,-4},{-9,-1,20}};
       RUNTAG(V, W, 43370, 4, 2, "T4_(2,2)_Q3-_Cv_Cw1"); }
    {  int V[4][3] = {{-3,0,0},{1,-15,9},{-13,20,0},{0,18,-6}};
       int W[4][3] = {{-13,-17,17},{-15,18,-10},{13,17,-17},{15,-9,-5}};
       RUNTAG(V, W, 80416, 4, 2, "T4_(2,2)_Q3-_Cw1"); }
    {  int V[4][3] = {{20,6,-11},{-14,-18,10},{-20,-8,-17},{-7,14,-1}};
       int W[4][3] = {{4,3,-1},{-18,5,13},{-4,-13,-1},{-19,3,13}};
       RUNTAG(V, W, 99581, 4, 2, "T4_Q2"); }
    {  int V[4][3] = {{14,8,-2},{12,14,-14},{18,-1,2},{-7,-4,1}};
       int W[4][3] = {{15,-9,-4},{10,-20,20},{-12,10,7},{3,11,-2}};
       RUNTAG(V, W, 76373, 3, 1, "T4_Q3+_Cv1"); }
    {  int V[4][3] = {{1,-2,0},{-4,7,-9},{-5,-17,13},{15,12,-4}};
       int W[4][3] = {{8,-16,0},{-6,-11,7},{-14,-12,-10},{5,0,5}};
       RUNTAG(V, W, 92684, 4, 2, "T4_Q3+_D00"); }
    {  int V[4][3] = {{-9,7,-13},{-6,-3,17},{15,-18,-19},{4,-17,-10}};
       int W[4][3] = {{-5,-7,-8},{-18,-20,-2},{9,10,1},{-14,0,-19}};
       RUNTAG(V, W, 122002, 4, 2, "T4_Q3-_Cw1"); }
    {  int V[4][3] = {{13,-2,-11},{-14,-6,-6},{18,-17,1},{-13,8,13}};
       int W[4][3] = {{4,-5,16},{0,0,0},{20,-15,-19},{-13,16,-7}};
       RUNTAG(V, W, 64454, 5, 2, "T5_(1,2,2)_Q3+_Cv_Cw0_D00"); }
    {  int V[4][3] = {{3,0,-2},{11,17,19},{-4,4,-3},{8,-8,6}};
       int W[4][3] = {{-14,-10,-9},{2,4,14},{19,18,5},{16,9,3}};
       RUNTAG(V, W, 63678, 5, 2, "T6_(1,1,2,2)_Q3+_Cv1_Cw"); }
    {  int V[4][3] = {{14,13,-13},{-4,-8,8},{-16,-3,-9},{-3,14,-14}};
       int W[4][3] = {{-17,-9,12},{9,4,-6},{12,9,8},{-3,8,-18}};
       RUNTAG(V, W, 48981, 6, 3, "T6_(1,1,2,2)_Q3+_Cv2_Cw"); }
    {  int V[4][3] = {{-15,9,-2},{18,9,-6},{-5,1,-19},{6,-7,6}};
       int W[4][3] = {{-7,0,16},{-5,-8,-7},{-9,-10,-3},{17,16,-2}};
       RUNTAG(V, W, 98694, 6, 3, "T6_(1,1,2,2)_Q3+_Cv_Cw2"); }
    {  int V[4][3] = {{-8,-3,-9},{12,1,10},{11,11,-11},{-8,-3,-2}};
       int W[4][3] = {{-3,1,19},{10,18,18},{14,-7,-12},{-18,-7,-15}};
       RUNTAG(V, W, 11186, 6, 3, "T6_(1,1,4)_Q3+_Cv2_Cw"); }
    {  int V[4][3] = {{-10,-7,-16},{-8,0,5},{4,14,1},{12,0,-7}};
       int W[4][3] = {{0,-5,1},{2,-3,8},{13,17,-6},{-4,14,-2}};
       RUNTAG(V, W, 103653, 6, 3, "T6_(1,1,4)_Q3+_Cv_Cw2"); }
    {  int V[4][3] = {{-3,14,-7},{14,10,3},{-13,4,14},{-2,-3,-4}};
       int W[4][3] = {{16,-5,18},{-1,-14,-9},{-15,19,-9},{-6,-18,13}};
       RUNTAG(V, W, 128479, 6, 3, "T6_(1,5)_Q3-_Cw2"); }
    {  int V[4][3] = {{18,1,-12},{-6,-6,2},{18,20,-5},{-18,-19,1}};
       int W[4][3] = {{-7,6,-11},{-17,16,20},{9,-10,-8},{-5,4,-19}};
       RUNTAG(V, W, 99032, 6, 3, "T6_(2,2,2)_Q2_Cv"); }
    {  int V[4][3] = {{20,8,-17},{18,17,19},{-20,-14,-4},{-15,14,19}};
       int W[4][3] = {{17,8,-12},{6,1,14},{-8,15,-18},{7,-19,5}};
       RUNTAG(V, W, 40984, 6, 3, "T6_(2,4)_Q3-_Cv2"); }
    {  int V[4][3] = {{1,18,1},{2,-1,13},{-5,-17,-8},{6,18,-6}};
       int W[4][3] = {{2,-9,-8},{-2,12,10},{-6,3,8},{14,6,11}};
       RUNTAG(V, W, 96004, 6, 3, "T6_(2,4)_Q3-_Cv_Cw2"); }
    {  int V[4][3] = {{15,-20,9},{-4,9,-19},{-6,-6,-10},{11,16,16}};
       int W[4][3] = {{6,-4,11},{16,-6,-19},{16,-6,-19},{-9,2,-8}};
       RUNTAG(V, W, 22007, 6, 3, "T6_Q2-"); }
    {  int V[4][3] = {{10,-9,-13},{13,16,1},{-9,13,-17},{4,-8,14}};
       int W[4][3] = {{-14,-4,9},{15,3,14},{-14,11,3},{17,16,6}};
       RUNTAG(V, W, 5893, 6, 3, "T6_Q3+_Cv"); }
    {  int V[4][3] = {{-7,-4,5},{3,-1,10},{13,-14,-10},{-7,14,2}};
       int W[4][3] = {{-11,6,11},{2,1,-1},{-17,15,-5},{-4,-16,-4}};
       RUNTAG(V, W, 119472, 8, 4, "T8_(3,5)_Q3-_Cv_Cw"); }

    #undef RUNTAG
}

// ============================================================================
// Constructed D11/D12/D22 cases (high-dimensional degeneracies)
// ============================================================================

void test_d_cases_3d() {
    std::cout << "  13 constructed 3D D11/D12/D22/D33 cases" << std::endl;
    #define RUNTAG(V, W, sd, exp_np, exp_npairs, exp_cat) do { \
        __int128 Q[4], P[4][4]; \
        compute_tet_QP_i128(V, W, Q, P); \
        ExactPV2Result result = solve_pv_tet_v2(Q, P); \
        total_tests++; \
        if (result.n_punctures != (exp_np) || result.n_pairs != (exp_npairs)) { \
            failed_tests++; \
            std::cerr << "FAILED: seed " << sd \
                      << " n_punctures=" << result.n_punctures << " (exp " << (exp_np) << ")" \
                      << " n_pairs=" << result.n_pairs << " (exp " << (exp_npairs) << ")" \
                      << std::endl; \
        } else { passed_tests++; } \
        { TetCaseV2GPU tv2; \
          memset(&tv2, 0, sizeof(tv2)); \
          for (int i=0;i<4;i++) for (int j=0;j<3;j++) { tv2.V[i][j]=V[i][j]; tv2.W[i][j]=W[i][j]; } \
          tv2.v2 = result; tv2.seed = sd; \
          for (int k=0;k<4;k++) tv2.disc_sign[k] = discriminant_sign_i128(P[k]); \
          ClassifiedCase cc = classify_case_v2(tv2); \
          total_tests++; \
          if (cc.category != std::string(exp_cat)) { \
              failed_tests++; \
              std::cerr << "FAILED: seed " << sd \
                        << " category=\"" << cc.category << "\" (exp \"" << (exp_cat) << "\")" \
                        << std::endl; \
          } else { passed_tests++; } \
        } \
    } while(0)

    // 3D D11: two vertices PV at same λ
    { int V[4][3]={{3,-2,5},{-4,7,1},{6,-1,3},{-1,4,-2}};
      int W[4][3]={{-3,2,-5},{4,-7,-1},{2,5,-8},{7,-3,1}};
      RUNTAG(V, W, 100001, 0, 0, "T0_Q3+_D11"); }
    { int V[4][3]={{1,2,3},{-3,1,5},{4,6,-2},{8,-4,6}};
      int W[4][3]={{-5,4,7},{2,-6,3},{-2,-3,1},{-4,2,-3}};
      RUNTAG(V, W, 100002, 0, 0, "T0_Q3+_D11"); }
    { int V[4][3]={{5,-3,7},{2,4,-6},{-1,8,3},{-7,2,4}};
      int W[4][3]={{-5,3,-7},{3,-1,5},{4,-2,1},{7,-2,-4}};
      RUNTAG(V, W, 100003, 2, 1, "T2_Q3+_ISR_D11"); }
    { int V[4][3]={{4,-3,8},{-6,5,2},{7,-1,-3},{2,3,-5}};
      int W[4][3]={{1,4,-2},{6,-5,-2},{-7,1,3},{-3,6,4}};
      RUNTAG(V, W, 100004, 4, 2, "T4_(1,1,2)_Q3+_ISR_Cw_D11"); }
    { int V[4][3]={{6,-9,12},{1,3,-5},{-3,6,15},{4,-2,7}};
      int W[4][3]={{-2,3,-4},{5,-1,2},{1,-2,-5},{-3,5,1}};
      RUNTAG(V, W, 100005, 2, 1, "T2_Q3-_D11"); }
    { int V[4][3]={{3,-5,2},{-4,7,1},{6,-1,3},{2,-3,8}};
      int W[4][3]={{3,-5,2},{-4,7,1},{-2,4,-6},{1,5,-3}};
      RUNTAG(V, W, 100006, 2, 1, "T2_(1,1)_Q3-_SR_D11"); }

    // 3D D22: three vertices PV at same λ
    { int V[4][3]={{5,-3,2},{-1,7,-4},{3,2,-6},{4,-1,5}};
      int W[4][3]={{-5,3,-2},{1,-7,4},{-3,-2,6},{-2,3,7}};
      RUNTAG(V, W, 100007, 0, 0, "T0_Q3o_D22"); }
    { int V[4][3]={{3,1,-5},{6,-4,8},{-2,10,4},{8,-6,14}};
      int W[4][3]={{2,-4,7},{-3,2,-4},{1,-5,-2},{-4,3,-7}};
      RUNTAG(V, W, 100008, 0, 0, "T0_Q3o_D22"); }
    { int V[4][3]={{4,-7,2},{-3,5,1},{6,2,-8},{8,-4,3}};
      int W[4][3]={{-4,7,-2},{3,-5,-1},{1,-6,5},{-8,4,-3}};
      RUNTAG(V, W, 100009, 2, 1, "T2_Q3o_D22"); }
    { int V[4][3]={{2,-5,4},{3,1,-7},{-6,3,8},{7,-2,1}};
      int W[4][3]={{-2,5,-4},{4,-3,6},{6,-3,-8},{-7,2,-1}};
      RUNTAG(V, W, 100010, 2, 1, "T2_(1,1)_Q3o_Cw_D22"); }

    // 3D D33: all 4 vertices PV at same λ
    { int V[4][3]={{3,-2,5},{-1,4,-3},{7,1,-6},{-5,8,2}};
      int W[4][3]={{-3,2,-5},{1,-4,3},{-7,-1,6},{5,-8,-2}};
      RUNTAG(V, W, 100011, 0, 0, "T0_Q3o_D33"); }

    // 3D D12: P[k]≡0 (PV curve lies on face)
    { int V[4][3]={{-2,-2,2},{-3,0,3},{1,2,-1},{6,-5,-3}};
      int W[4][3]={{2,0,-2},{0,-2,0},{0,-1,0},{2,3,1}};
      RUNTAG(V, W, 100012, 1, 0, "T2_Q3-_D12"); }

    #undef RUNTAG
}

void test_d_cases_2d() {
    std::cout << "  7 constructed 2D D11/D22 cases" << std::endl;

    // 2D D11: two vertices PV at same λ
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200001;
      gpu.V[0][0]=3; gpu.V[0][1]=-5; gpu.W[0][0]=-3; gpu.W[0][1]=5;
      gpu.V[1][0]=-4; gpu.V[1][1]=7; gpu.W[1][0]=4; gpu.W[1][1]=-7;
      gpu.V[2][0]=2; gpu.V[2][1]=-3; gpu.W[2][0]=1; gpu.W[2][1]=6;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR_Cw_D00_D11"); }
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200002;
      gpu.V[0][0]=5; gpu.V[0][1]=-3; gpu.W[0][0]=2; gpu.W[0][1]=1;
      gpu.V[1][0]=4; gpu.V[1][1]=-6; gpu.W[1][0]=-2; gpu.W[1][1]=3;
      gpu.V[2][0]=-8; gpu.V[2][1]=10; gpu.W[2][0]=4; gpu.W[2][1]=-5;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_SR_Cv_D00_D11"); }
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200003;
      gpu.V[0][0]=7; gpu.V[0][1]=-2; gpu.W[0][0]=-7; gpu.W[0][1]=2;
      gpu.V[1][0]=1; gpu.V[1][1]=5; gpu.W[1][0]=3; gpu.W[1][1]=-8;
      gpu.V[2][0]=-4; gpu.V[2][1]=3; gpu.W[2][0]=4; gpu.W[2][1]=-3;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_SR_D00_D11"); }
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200004;
      gpu.V[0][0]=3; gpu.V[0][1]=5; gpu.W[0][0]=3; gpu.W[0][1]=5;
      gpu.V[1][0]=-7; gpu.V[1][1]=2; gpu.W[1][0]=-7; gpu.W[1][1]=2;
      gpu.V[2][0]=4; gpu.V[2][1]=-6; gpu.W[2][0]=-1; gpu.W[2][1]=8;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR_Cv_D00_D11"); }
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200005;
      gpu.V[0][0]=2; gpu.V[0][1]=-7; gpu.W[0][0]=5; gpu.W[0][1]=1;
      gpu.V[1][0]=6; gpu.V[1][1]=9; gpu.W[1][0]=-2; gpu.W[1][1]=-3;
      gpu.V[2][0]=-12; gpu.V[2][1]=3; gpu.W[2][0]=4; gpu.W[2][1]=-1;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv_D00_D11"); }

    // 2D D22: all 3 vertices PV at same λ
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200006;
      gpu.V[0][0]=3; gpu.V[0][1]=-5; gpu.W[0][0]=-3; gpu.W[0][1]=5;
      gpu.V[1][0]=-2; gpu.V[1][1]=4; gpu.W[1][0]=2; gpu.W[1][1]=-4;
      gpu.V[2][0]=7; gpu.V[2][1]=-1; gpu.W[2][0]=-7; gpu.W[2][1]=1;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2o_ISR_D00_D11_D22"); }
    { ftk2::TriCaseV2GPU gpu; gpu.seed = 200007;
      gpu.V[0][0]=4; gpu.V[0][1]=-6; gpu.W[0][0]=-2; gpu.W[0][1]=3;
      gpu.V[1][0]=-8; gpu.V[1][1]=2; gpu.W[1][0]=4; gpu.W[1][1]=-1;
      gpu.V[2][0]=10; gpu.V[2][1]=14; gpu.W[2][0]=-5; gpu.W[2][1]=-7;
      __int128 V128[3][2], W128[3][2];
      for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
      __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
      gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
      for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
          __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
          gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
      ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2o_ISR_Cv_Cw_D00_D11_D22"); }
}

int main() {
    std::cout << "=== content_reduce_i128 ===" << std::endl;
    test_content_reduce_basic();
    test_content_reduce_already_primitive();
    test_content_reduce_with_zero();
    test_content_reduce_negative();
    test_content_reduce_degree_zero();
    test_content_reduce_neg_degree();

    std::cout << "\n=== effective_degree_i128 ===" << std::endl;
    test_effective_degree();

    std::cout << "\n=== discriminant_sign_i128 ===" << std::endl;
    test_discriminant_sign_three_roots();
    test_discriminant_sign_one_root();
    test_discriminant_sign_repeated();
    test_discriminant_sign_triple_root();
    test_discriminant_sign_not_cubic();
    test_discriminant_sign_neg_leading();

    std::cout << "\n=== prem_i128 ===" << std::endl;
    test_prem_basic();
    test_prem_zero_skip();
    test_prem_seed2520();
    test_prem_equal_degree();
    test_prem_exact_division();
    test_prem_neg_leading();

    std::cout << "\n=== poly_exact_div_i128 ===" << std::endl;
    test_poly_exact_div_linear();
    test_poly_exact_div_quadratic();
    test_poly_exact_div_equal_degree();
    test_poly_exact_div_neg_leading();

    std::cout << "\n=== poly_sqfree_i128 ===" << std::endl;
    test_sqfree_already();
    test_sqfree_double_root();
    test_sqfree_triple_root();
    test_sqfree_quadratic_double();
    test_sqfree_linear();

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
    test_signs_quadratic_linear_shared();
    test_signs_quadratic_both_negative();
    test_signs_cubic_3roots_shared_middle();
    test_signs_cubic_1root_quadratic_g();
    test_signs_quadratic_prem_reduction();
    test_signs_zero_polynomial();

    std::cout << "\n=== sign_at_unique_root_i128 ===" << std::endl;
    test_sign_unique_root_positive();
    test_sign_unique_root_negative();
    test_sign_unique_root_shared();
    test_sign_unique_root_constant_g();
    test_sign_unique_root_neg_lc();

    std::cout << "\n=== resultant_sign_i128 ===" << std::endl;
    test_resultant_shared_root();
    test_resultant_no_shared_root();
    test_resultant_cubic_shared();
    test_resultant_mixed_degree();
    test_resultant_constant_f();
    test_resultant_constant_f_neg();
    test_resultant_constant_f_zero();
    test_resultant_both_constants();

    std::cout << "\n=== poly_gcd_full_i128 ===" << std::endl;
    test_gcd_shared_factor();
    test_gcd_no_shared_factor();
    test_gcd_four_polys();
    test_gcd_f_zero();
    test_gcd_quadratic_factor();
    test_gcd_swap_degrees();

    std::cout << "\n=== compare_roots_i128 ===" << std::endl;
    test_compare_roots_disjoint();
    test_compare_roots_equal();
    test_compare_roots_3root_cubic();
    test_compare_roots_degenerate();

    std::cout << "\n=== RP1 sign helpers ===" << std::endl;
    test_sign_at_plus_inf();
    test_sign_at_inf();
    test_sign_just_after_root();
    test_rp1_pairing_basic();

    std::cout << "\n=== compute_tet_QP_i128 ===" << std::endl;
    test_compute_tet_QP_identity();
    test_compute_tet_QP_coplanar_zero();

    std::cout << "\n=== check_field_zero_in_tet ===" << std::endl;
    test_field_zero_inside();
    test_field_zero_outside();
    test_field_zero_at_vertex();
    test_field_zero_coplanar_det0();

    std::cout << "\n=== check_field_zero_coplanar ===" << std::endl;
    test_coplanar_inside();
    test_coplanar_outside();
    test_coplanar_all_zero();
    test_coplanar_1d_inside();
    test_coplanar_1d_outside();
    test_coplanar_3d_plane();

    std::cout << "\n=== solve_pv_tet_v2 regressions ===" << std::endl;
    test_solve_v2_seed91502();
    test_solve_v2_seed4984();
    test_solve_v2_seed6247();

    std::cout << "\n=== paper cases (figures_v15) ===" << std::endl;
    test_paper_cases();

    std::cout << "\n=== paper cases (figures_v18) ===" << std::endl;
    test_paper_cases_v18();

    std::cout << "\n=== constructed degenerate cases (figures_v19) ===" << std::endl;
    test_constructed_cases();

    std::cout << "\n=== new structural cases (figures_v20) ===" << std::endl;
    test_structural_cases_v20();

    std::cout << "\n=== diverse coverage (109 categories) ===" << std::endl;
    test_diverse_coverage();

    std::cout << "\n=== constructed 3D D11/D12/D22/D33 ===" << std::endl;
    test_d_cases_3d();

    std::cout << "\n=== constructed 2D D11/D22 ===" << std::endl;
    test_d_cases_2d();

    std::cout << "\n=== 2D classification (curated + regression) ===" << std::endl;
    test_classify_2d_cases();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total: " << total_tests << ", Passed: " << passed_tests
              << ", Failed: " << failed_tests << std::endl;
    return failed_tests > 0 ? 1 : 0;
}
// 179 2D PV classification regression tests
// 172 2D PV classification regression tests
static void test_classify_2d_cases() {
    std::cout << "Testing 172 2D cases..." << std::endl;
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 78187;
        gpu.V[0][0]=6; gpu.V[0][1]=14; gpu.W[0][0]=0; gpu.W[0][1]=-12;
        gpu.V[1][0]=16; gpu.V[1][1]=3; gpu.W[1][0]=3; gpu.W[1][1]=-18;
        gpu.V[2][0]=14; gpu.V[2][1]=-8; gpu.W[2][0]=-2; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q0");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 4574;
        gpu.V[0][0]=-3; gpu.V[0][1]=9; gpu.W[0][0]=-20; gpu.W[0][1]=-12;
        gpu.V[1][0]=-7; gpu.V[1][1]=-12; gpu.W[1][0]=16; gpu.W[1][1]=-14;
        gpu.V[2][0]=-2; gpu.V[2][1]=0; gpu.W[2][0]=16; gpu.W[2][1]=-14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 64902;
        gpu.V[0][0]=11; gpu.V[0][1]=20; gpu.W[0][0]=0; gpu.W[0][1]=15;
        gpu.V[1][0]=19; gpu.V[1][1]=-10; gpu.W[1][0]=12; gpu.W[1][1]=15;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=-3; gpu.W[2][1]=15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q1_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6073;
        gpu.V[0][0]=4; gpu.V[0][1]=18; gpu.W[0][0]=1; gpu.W[0][1]=17;
        gpu.V[1][0]=18; gpu.V[1][1]=-18; gpu.W[1][0]=-11; gpu.W[1][1]=11;
        gpu.V[2][0]=3; gpu.V[2][1]=14; gpu.W[2][0]=-5; gpu.W[2][1]=14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8744;
        gpu.V[0][0]=10; gpu.V[0][1]=14; gpu.W[0][0]=-16; gpu.W[0][1]=-13;
        gpu.V[1][0]=-13; gpu.V[1][1]=19; gpu.W[1][0]=6; gpu.W[1][1]=-11;
        gpu.V[2][0]=-6; gpu.V[2][1]=9; gpu.W[2][0]=-4; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 18751;
        gpu.V[0][0]=14; gpu.V[0][1]=-20; gpu.W[0][0]=-16; gpu.W[0][1]=7;
        gpu.V[1][0]=14; gpu.V[1][1]=11; gpu.W[1][0]=11; gpu.W[1][1]=-13;
        gpu.V[2][0]=14; gpu.V[2][1]=9; gpu.W[2][0]=20; gpu.W[2][1]=-9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9366;
        gpu.V[0][0]=1; gpu.V[0][1]=16; gpu.W[0][0]=-14; gpu.W[0][1]=1;
        gpu.V[1][0]=0; gpu.V[1][1]=0; gpu.W[1][0]=-20; gpu.W[1][1]=-9;
        gpu.V[2][0]=-6; gpu.V[2][1]=16; gpu.W[2][0]=-7; gpu.W[2][1]=-2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 14496;
        gpu.V[0][0]=-1; gpu.V[0][1]=19; gpu.W[0][0]=19; gpu.W[0][1]=-8;
        gpu.V[1][0]=10; gpu.V[1][1]=-13; gpu.W[1][0]=9; gpu.W[1][1]=-12;
        gpu.V[2][0]=8; gpu.V[2][1]=12; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11810;
        gpu.V[0][0]=-12; gpu.V[0][1]=3; gpu.W[0][0]=4; gpu.W[0][1]=-18;
        gpu.V[1][0]=-3; gpu.V[1][1]=-4; gpu.W[1][0]=-3; gpu.W[1][1]=-4;
        gpu.V[2][0]=-8; gpu.V[2][1]=-7; gpu.W[2][0]=-11; gpu.W[2][1]=-19;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 25644;
        gpu.V[0][0]=-4; gpu.V[0][1]=6; gpu.W[0][0]=-4; gpu.W[0][1]=4;
        gpu.V[1][0]=14; gpu.V[1][1]=-16; gpu.W[1][0]=12; gpu.W[1][1]=6;
        gpu.V[2][0]=-20; gpu.V[2][1]=20; gpu.W[2][0]=-16; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_ISR_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 545;
        gpu.V[0][0]=9; gpu.V[0][1]=-12; gpu.W[0][0]=0; gpu.W[0][1]=-8;
        gpu.V[1][0]=6; gpu.V[1][1]=-12; gpu.W[1][0]=-15; gpu.W[1][1]=-8;
        gpu.V[2][0]=10; gpu.V[2][1]=14; gpu.W[2][0]=13; gpu.W[2][1]=-17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_SR");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 56002;
        gpu.V[0][0]=-12; gpu.V[0][1]=-19; gpu.W[0][0]=13; gpu.W[0][1]=0;
        gpu.V[1][0]=-9; gpu.V[1][1]=0; gpu.W[1][0]=0; gpu.W[1][1]=-3;
        gpu.V[2][0]=-12; gpu.V[2][1]=-19; gpu.W[2][0]=-4; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_SR_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 35894;
        gpu.V[0][0]=-14; gpu.V[0][1]=-3; gpu.W[0][0]=5; gpu.W[0][1]=7;
        gpu.V[1][0]=11; gpu.V[1][1]=0; gpu.W[1][0]=-7; gpu.W[1][1]=0;
        gpu.V[2][0]=-6; gpu.V[2][1]=-3; gpu.W[2][0]=3; gpu.W[2][1]=7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_SR_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 31432;
        gpu.V[0][0]=2; gpu.V[0][1]=7; gpu.W[0][0]=-2; gpu.W[0][1]=-1;
        gpu.V[1][0]=20; gpu.V[1][1]=2; gpu.W[1][0]=-10; gpu.W[1][1]=9;
        gpu.V[2][0]=6; gpu.V[2][1]=7; gpu.W[2][0]=-8; gpu.W[2][1]=-9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2+_TN");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8751;
        gpu.V[0][0]=-18; gpu.V[0][1]=0; gpu.W[0][0]=-4; gpu.W[0][1]=-19;
        gpu.V[1][0]=-19; gpu.V[1][1]=-18; gpu.W[1][0]=9; gpu.W[1][1]=-18;
        gpu.V[2][0]=2; gpu.V[2][1]=-11; gpu.W[2][0]=10; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2-");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 20538;
        gpu.V[0][0]=0; gpu.V[0][1]=0; gpu.W[0][0]=-6; gpu.W[0][1]=-3;
        gpu.V[1][0]=-1; gpu.V[1][1]=20; gpu.W[1][0]=-8; gpu.W[1][1]=4;
        gpu.V[2][0]=-5; gpu.V[2][1]=5; gpu.W[2][0]=-10; gpu.W[2][1]=-5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2-_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7000;
        gpu.V[0][0]=-7; gpu.V[0][1]=13; gpu.W[0][0]=6; gpu.W[0][1]=12;
        gpu.V[1][0]=-1; gpu.V[1][1]=-17; gpu.W[1][0]=-18; gpu.W[1][1]=14;
        gpu.V[2][0]=8; gpu.V[2][1]=-2; gpu.W[2][0]=3; gpu.W[2][1]=-18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2-_Cv_Cw_B");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9631;
        gpu.V[0][0]=-3; gpu.V[0][1]=4; gpu.W[0][0]=-7; gpu.W[0][1]=-15;
        gpu.V[1][0]=-3; gpu.V[1][1]=-3; gpu.W[1][0]=-13; gpu.W[1][1]=-13;
        gpu.V[2][0]=-18; gpu.V[2][1]=1; gpu.W[2][0]=-12; gpu.W[2][1]=-2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Q2-_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 49064;
        gpu.V[0][0]=-11; gpu.V[0][1]=12; gpu.W[0][0]=20; gpu.W[0][1]=6;
        gpu.V[1][0]=-14; gpu.V[1][1]=-13; gpu.W[1][0]=2; gpu.W[1][1]=4;
        gpu.V[2][0]=-14; gpu.V[2][1]=-13; gpu.W[2][0]=2; gpu.W[2][1]=4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T0_Qz_Cv1_D11");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 53518;
        gpu.V[0][0]=12; gpu.V[0][1]=1; gpu.W[0][0]=0; gpu.W[0][1]=-11;
        gpu.V[1][0]=19; gpu.V[1][1]=-1; gpu.W[1][0]=0; gpu.W[1][1]=4;
        gpu.V[2][0]=-14; gpu.V[2][1]=0; gpu.W[2][0]=0; gpu.W[2][1]=7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q1_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9828;
        gpu.V[0][0]=2; gpu.V[0][1]=-18; gpu.W[0][0]=-7; gpu.W[0][1]=7;
        gpu.V[1][0]=-6; gpu.V[1][1]=14; gpu.W[1][0]=7; gpu.W[1][1]=7;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=-7; gpu.W[2][1]=-17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv0_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9790;
        gpu.V[0][0]=9; gpu.V[0][1]=12; gpu.W[0][0]=3; gpu.W[0][1]=-15;
        gpu.V[1][0]=19; gpu.V[1][1]=-16; gpu.W[1][0]=-6; gpu.W[1][1]=4;
        gpu.V[2][0]=-6; gpu.V[2][1]=-8; gpu.W[2][0]=12; gpu.W[2][1]=-2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 90742;
        gpu.V[0][0]=-8; gpu.V[0][1]=-8; gpu.W[0][0]=-18; gpu.W[0][1]=-9;
        gpu.V[1][0]=1; gpu.V[1][1]=1; gpu.W[1][0]=3; gpu.W[1][1]=3;
        gpu.V[2][0]=-9; gpu.V[2][1]=-16; gpu.W[2][0]=-10; gpu.W[2][1]=-15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv1_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11899;
        gpu.V[0][0]=5; gpu.V[0][1]=14; gpu.W[0][0]=1; gpu.W[0][1]=19;
        gpu.V[1][0]=-13; gpu.V[1][1]=-3; gpu.W[1][0]=-16; gpu.W[1][1]=5;
        gpu.V[2][0]=5; gpu.V[2][1]=-13; gpu.W[2][0]=15; gpu.W[2][1]=-6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 86787;
        gpu.V[0][0]=-8; gpu.V[0][1]=-9; gpu.W[0][0]=10; gpu.W[0][1]=14;
        gpu.V[1][0]=8; gpu.V[1][1]=16; gpu.W[1][0]=18; gpu.W[1][1]=-6;
        gpu.V[2][0]=19; gpu.V[2][1]=19; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 14219;
        gpu.V[0][0]=20; gpu.V[0][1]=6; gpu.W[0][0]=-9; gpu.W[0][1]=-18;
        gpu.V[1][0]=13; gpu.V[1][1]=-16; gpu.W[1][0]=10; gpu.W[1][1]=20;
        gpu.V[2][0]=-15; gpu.V[2][1]=-3; gpu.W[2][0]=20; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 27428;
        gpu.V[0][0]=-18; gpu.V[0][1]=-9; gpu.W[0][0]=4; gpu.W[0][1]=2;
        gpu.V[1][0]=-5; gpu.V[1][1]=-19; gpu.W[1][0]=-20; gpu.W[1][1]=3;
        gpu.V[2][0]=4; gpu.V[2][1]=8; gpu.W[2][0]=-16; gpu.W[2][1]=-13;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cv_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13894;
        gpu.V[0][0]=14; gpu.V[0][1]=-15; gpu.W[0][0]=1; gpu.W[0][1]=-13;
        gpu.V[1][0]=-15; gpu.V[1][1]=9; gpu.W[1][0]=-8; gpu.W[1][1]=12;
        gpu.V[2][0]=2; gpu.V[2][1]=-10; gpu.W[2][0]=17; gpu.W[2][1]=-9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 303;
        gpu.V[0][0]=10; gpu.V[0][1]=18; gpu.W[0][0]=-17; gpu.W[0][1]=15;
        gpu.V[1][0]=1; gpu.V[1][1]=5; gpu.W[1][0]=10; gpu.W[1][1]=3;
        gpu.V[2][0]=4; gpu.V[2][1]=16; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6923;
        gpu.V[0][0]=11; gpu.V[0][1]=20; gpu.W[0][0]=-4; gpu.W[0][1]=2;
        gpu.V[1][0]=10; gpu.V[1][1]=-15; gpu.W[1][0]=-9; gpu.W[1][1]=14;
        gpu.V[2][0]=19; gpu.V[2][1]=-15; gpu.W[2][0]=4; gpu.W[2][1]=-2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7637;
        gpu.V[0][0]=13; gpu.V[0][1]=-2; gpu.W[0][0]=-3; gpu.W[0][1]=3;
        gpu.V[1][0]=12; gpu.V[1][1]=-5; gpu.W[1][0]=12; gpu.W[1][1]=-5;
        gpu.V[2][0]=1; gpu.V[2][1]=19; gpu.W[2][0]=9; gpu.W[2][1]=-9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3801;
        gpu.V[0][0]=-8; gpu.V[0][1]=4; gpu.W[0][0]=-5; gpu.W[0][1]=3;
        gpu.V[1][0]=2; gpu.V[1][1]=2; gpu.W[1][0]=3; gpu.W[1][1]=3;
        gpu.V[2][0]=19; gpu.V[2][1]=20; gpu.W[2][0]=8; gpu.W[2][1]=-7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11620;
        gpu.V[0][0]=-18; gpu.V[0][1]=-15; gpu.W[0][0]=-8; gpu.W[0][1]=-5;
        gpu.V[1][0]=17; gpu.V[1][1]=18; gpu.W[1][0]=8; gpu.W[1][1]=9;
        gpu.V[2][0]=11; gpu.V[2][1]=10; gpu.W[2][0]=16; gpu.W[2][1]=15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 88499;
        gpu.V[0][0]=-5; gpu.V[0][1]=2; gpu.W[0][0]=12; gpu.W[0][1]=-10;
        gpu.V[1][0]=-9; gpu.V[1][1]=-3; gpu.W[1][0]=1; gpu.W[1][1]=15;
        gpu.V[2][0]=10; gpu.V[2][1]=-4; gpu.W[2][0]=16; gpu.W[2][1]=20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 32116;
        gpu.V[0][0]=16; gpu.V[0][1]=10; gpu.W[0][0]=5; gpu.W[0][1]=-18;
        gpu.V[1][0]=0; gpu.V[1][1]=1; gpu.W[1][0]=9; gpu.W[1][1]=15;
        gpu.V[2][0]=-11; gpu.V[2][1]=-10; gpu.W[2][0]=-13; gpu.W[2][1]=-7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 25679;
        gpu.V[0][0]=-5; gpu.V[0][1]=7; gpu.W[0][0]=-16; gpu.W[0][1]=-2;
        gpu.V[1][0]=-10; gpu.V[1][1]=-10; gpu.W[1][0]=-2; gpu.W[1][1]=-2;
        gpu.V[2][0]=17; gpu.V[2][1]=-1; gpu.W[2][0]=20; gpu.W[2][1]=-1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 14669;
        gpu.V[0][0]=-10; gpu.V[0][1]=13; gpu.W[0][0]=-5; gpu.W[0][1]=7;
        gpu.V[1][0]=-20; gpu.V[1][1]=-11; gpu.W[1][0]=14; gpu.W[1][1]=-4;
        gpu.V[2][0]=-14; gpu.V[2][1]=-11; gpu.W[2][0]=1; gpu.W[2][1]=-4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_(1,1)_Q2+_SR_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 1892;
        gpu.V[0][0]=13; gpu.V[0][1]=12; gpu.W[0][0]=-18; gpu.W[0][1]=19;
        gpu.V[1][0]=3; gpu.V[1][1]=-5; gpu.W[1][0]=-5; gpu.W[1][1]=-7;
        gpu.V[2][0]=11; gpu.V[2][1]=17; gpu.W[2][0]=-18; gpu.W[2][1]=19;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5222;
        gpu.V[0][0]=19; gpu.V[0][1]=11; gpu.W[0][0]=-6; gpu.W[0][1]=2;
        gpu.V[1][0]=-8; gpu.V[1][1]=-8; gpu.W[1][0]=7; gpu.W[1][1]=2;
        gpu.V[2][0]=-12; gpu.V[2][1]=10; gpu.W[2][0]=16; gpu.W[2][1]=2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q1_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 22092;
        gpu.V[0][0]=-13; gpu.V[0][1]=-3; gpu.W[0][0]=-3; gpu.W[0][1]=17;
        gpu.V[1][0]=1; gpu.V[1][1]=-1; gpu.W[1][0]=12; gpu.W[1][1]=2;
        gpu.V[2][0]=-6; gpu.V[2][1]=6; gpu.W[2][0]=14; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q1_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 67958;
        gpu.V[0][0]=12; gpu.V[0][1]=-11; gpu.W[0][0]=-18; gpu.W[0][1]=7;
        gpu.V[1][0]=0; gpu.V[1][1]=11; gpu.W[1][0]=0; gpu.W[1][1]=-1;
        gpu.V[2][0]=-1; gpu.V[2][1]=-20; gpu.W[2][0]=0; gpu.W[2][1]=-1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q1_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 34189;
        gpu.V[0][0]=3; gpu.V[0][1]=-18; gpu.W[0][0]=0; gpu.W[0][1]=0;
        gpu.V[1][0]=-11; gpu.V[1][1]=-14; gpu.W[1][0]=18; gpu.W[1][1]=12;
        gpu.V[2][0]=-11; gpu.V[2][1]=-5; gpu.W[2][0]=-12; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 64566;
        gpu.V[0][0]=7; gpu.V[0][1]=-7; gpu.W[0][0]=8; gpu.W[0][1]=-12;
        gpu.V[1][0]=-4; gpu.V[1][1]=-9; gpu.W[1][0]=8; gpu.W[1][1]=-8;
        gpu.V[2][0]=7; gpu.V[2][1]=1; gpu.W[2][0]=8; gpu.W[2][1]=15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q1_SR");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13891;
        gpu.V[0][0]=1; gpu.V[0][1]=3; gpu.W[0][0]=-7; gpu.W[0][1]=-7;
        gpu.V[1][0]=-9; gpu.V[1][1]=5; gpu.W[1][0]=-11; gpu.W[1][1]=19;
        gpu.V[2][0]=-10; gpu.V[2][1]=10; gpu.W[2][0]=3; gpu.W[2][1]=-17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13908;
        gpu.V[0][0]=10; gpu.V[0][1]=8; gpu.W[0][0]=16; gpu.W[0][1]=-8;
        gpu.V[1][0]=-2; gpu.V[1][1]=-16; gpu.W[1][0]=6; gpu.W[1][1]=7;
        gpu.V[2][0]=-20; gpu.V[2][1]=-3; gpu.W[2][0]=-6; gpu.W[2][1]=-18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 60087;
        gpu.V[0][0]=-11; gpu.V[0][1]=3; gpu.W[0][0]=-4; gpu.W[0][1]=1;
        gpu.V[1][0]=0; gpu.V[1][1]=0; gpu.W[1][0]=4; gpu.W[1][1]=-1;
        gpu.V[2][0]=10; gpu.V[2][1]=13; gpu.W[2][0]=-3; gpu.W[2][1]=-9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv0_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10514;
        gpu.V[0][0]=0; gpu.V[0][1]=-20; gpu.W[0][0]=18; gpu.W[0][1]=-20;
        gpu.V[1][0]=0; gpu.V[1][1]=0; gpu.W[1][0]=-3; gpu.W[1][1]=-9;
        gpu.V[2][0]=-1; gpu.V[2][1]=-14; gpu.W[2][0]=15; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8817;
        gpu.V[0][0]=-2; gpu.V[0][1]=-11; gpu.W[0][0]=7; gpu.W[0][1]=7;
        gpu.V[1][0]=20; gpu.V[1][1]=10; gpu.W[1][0]=5; gpu.W[1][1]=-16;
        gpu.V[2][0]=-8; gpu.V[2][1]=-4; gpu.W[2][0]=13; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5119;
        gpu.V[0][0]=0; gpu.V[0][1]=-5; gpu.W[0][0]=0; gpu.W[0][1]=-18;
        gpu.V[1][0]=14; gpu.V[1][1]=15; gpu.W[1][0]=-9; gpu.W[1][1]=-7;
        gpu.V[2][0]=0; gpu.V[2][1]=20; gpu.W[2][0]=-4; gpu.W[2][1]=-4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8347;
        gpu.V[0][0]=16; gpu.V[0][1]=-9; gpu.W[0][0]=6; gpu.W[0][1]=0;
        gpu.V[1][0]=-20; gpu.V[1][1]=13; gpu.W[1][0]=-1; gpu.W[1][1]=-13;
        gpu.V[2][0]=-12; gpu.V[2][1]=0; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12639;
        gpu.V[0][0]=-5; gpu.V[0][1]=-5; gpu.W[0][0]=-13; gpu.W[0][1]=-9;
        gpu.V[1][0]=17; gpu.V[1][1]=11; gpu.W[1][0]=13; gpu.W[1][1]=9;
        gpu.V[2][0]=-19; gpu.V[2][1]=2; gpu.W[2][0]=10; gpu.W[2][1]=17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9021;
        gpu.V[0][0]=-12; gpu.V[0][1]=-16; gpu.W[0][0]=12; gpu.W[0][1]=16;
        gpu.V[1][0]=11; gpu.V[1][1]=18; gpu.W[1][0]=0; gpu.W[1][1]=-20;
        gpu.V[2][0]=7; gpu.V[2][1]=-11; gpu.W[2][0]=9; gpu.W[2][1]=-1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 45;
        gpu.V[0][0]=-20; gpu.V[0][1]=-11; gpu.W[0][0]=16; gpu.W[0][1]=11;
        gpu.V[1][0]=-5; gpu.V[1][1]=19; gpu.W[1][0]=-2; gpu.W[1][1]=9;
        gpu.V[2][0]=11; gpu.V[2][1]=-4; gpu.W[2][0]=5; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cv_TN");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3170;
        gpu.V[0][0]=-16; gpu.V[0][1]=-2; gpu.W[0][0]=0; gpu.W[0][1]=0;
        gpu.V[1][0]=4; gpu.V[1][1]=-4; gpu.W[1][0]=-1; gpu.W[1][1]=3;
        gpu.V[2][0]=-11; gpu.V[2][1]=-5; gpu.W[2][0]=15; gpu.W[2][1]=11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 147;
        gpu.V[0][0]=-18; gpu.V[0][1]=-3; gpu.W[0][0]=7; gpu.W[0][1]=-4;
        gpu.V[1][0]=16; gpu.V[1][1]=8; gpu.W[1][0]=18; gpu.W[1][1]=6;
        gpu.V[2][0]=-15; gpu.V[2][1]=-2; gpu.W[2][0]=-15; gpu.W[2][1]=-5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 25292;
        gpu.V[0][0]=-3; gpu.V[0][1]=3; gpu.W[0][0]=-20; gpu.W[0][1]=20;
        gpu.V[1][0]=-4; gpu.V[1][1]=0; gpu.W[1][0]=9; gpu.W[1][1]=-9;
        gpu.V[2][0]=-11; gpu.V[2][1]=1; gpu.W[2][0]=1; gpu.W[2][1]=6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5198;
        gpu.V[0][0]=-19; gpu.V[0][1]=19; gpu.W[0][0]=19; gpu.W[0][1]=-19;
        gpu.V[1][0]=-15; gpu.V[1][1]=-18; gpu.W[1][0]=8; gpu.W[1][1]=-6;
        gpu.V[2][0]=1; gpu.V[2][1]=5; gpu.W[2][0]=12; gpu.W[2][1]=8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11624;
        gpu.V[0][0]=12; gpu.V[0][1]=12; gpu.W[0][0]=-7; gpu.W[0][1]=19;
        gpu.V[1][0]=-4; gpu.V[1][1]=6; gpu.W[1][0]=-4; gpu.W[1][1]=-4;
        gpu.V[2][0]=12; gpu.V[2][1]=8; gpu.W[2][0]=-7; gpu.W[2][1]=-9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_SR");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 15423;
        gpu.V[0][0]=-13; gpu.V[0][1]=4; gpu.W[0][0]=20; gpu.W[0][1]=6;
        gpu.V[1][0]=18; gpu.V[1][1]=-16; gpu.W[1][0]=-3; gpu.W[1][1]=14;
        gpu.V[2][0]=3; gpu.V[2][1]=6; gpu.W[2][0]=4; gpu.W[2][1]=4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_SR_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 30118;
        gpu.V[0][0]=8; gpu.V[0][1]=2; gpu.W[0][0]=0; gpu.W[0][1]=14;
        gpu.V[1][0]=-4; gpu.V[1][1]=-1; gpu.W[1][0]=-1; gpu.W[1][1]=0;
        gpu.V[2][0]=-1; gpu.V[2][1]=-11; gpu.W[2][0]=-18; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_SR_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 2309;
        gpu.V[0][0]=3; gpu.V[0][1]=13; gpu.W[0][0]=19; gpu.W[0][1]=17;
        gpu.V[1][0]=6; gpu.V[1][1]=-10; gpu.W[1][0]=3; gpu.W[1][1]=-5;
        gpu.V[2][0]=-4; gpu.V[2][1]=-16; gpu.W[2][0]=18; gpu.W[2][1]=4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_SR_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13003;
        gpu.V[0][0]=-19; gpu.V[0][1]=4; gpu.W[0][0]=0; gpu.W[0][1]=18;
        gpu.V[1][0]=-19; gpu.V[1][1]=12; gpu.W[1][0]=0; gpu.W[1][1]=-14;
        gpu.V[2][0]=3; gpu.V[2][1]=-4; gpu.W[2][0]=17; gpu.W[2][1]=-13;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_SR_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 80957;
        gpu.V[0][0]=12; gpu.V[0][1]=-14; gpu.W[0][0]=-10; gpu.W[0][1]=8;
        gpu.V[1][0]=7; gpu.V[1][1]=-6; gpu.W[1][0]=18; gpu.W[1][1]=-17;
        gpu.V[2][0]=-11; gpu.V[2][1]=4; gpu.W[2][0]=-12; gpu.W[2][1]=4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2+_TN");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3149;
        gpu.V[0][0]=5; gpu.V[0][1]=-17; gpu.W[0][0]=19; gpu.W[0][1]=-16;
        gpu.V[1][0]=17; gpu.V[1][1]=5; gpu.W[1][0]=11; gpu.W[1][1]=6;
        gpu.V[2][0]=4; gpu.V[2][1]=6; gpu.W[2][0]=-7; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3142;
        gpu.V[0][0]=-2; gpu.V[0][1]=20; gpu.W[0][0]=20; gpu.W[0][1]=0;
        gpu.V[1][0]=-19; gpu.V[1][1]=-2; gpu.W[1][0]=16; gpu.W[1][1]=4;
        gpu.V[2][0]=5; gpu.V[2][1]=0; gpu.W[2][0]=-7; gpu.W[2][1]=-4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 18722;
        gpu.V[0][0]=15; gpu.V[0][1]=-2; gpu.W[0][0]=11; gpu.W[0][1]=-19;
        gpu.V[1][0]=6; gpu.V[1][1]=-16; gpu.W[1][0]=-2; gpu.W[1][1]=-6;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=2; gpu.W[2][1]=8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv0_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6621;
        gpu.V[0][0]=0; gpu.V[0][1]=0; gpu.W[0][0]=-16; gpu.W[0][1]=-4;
        gpu.V[1][0]=-14; gpu.V[1][1]=12; gpu.W[1][0]=-7; gpu.W[1][1]=-16;
        gpu.V[2][0]=7; gpu.V[2][1]=-13; gpu.W[2][0]=-7; gpu.W[2][1]=-3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 2142;
        gpu.V[0][0]=1; gpu.V[0][1]=-14; gpu.W[0][0]=-12; gpu.W[0][1]=-20;
        gpu.V[1][0]=2; gpu.V[1][1]=1; gpu.W[1][0]=9; gpu.W[1][1]=-17;
        gpu.V[2][0]=-4; gpu.V[2][1]=-2; gpu.W[2][0]=13; gpu.W[2][1]=-7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 554;
        gpu.V[0][0]=-14; gpu.V[0][1]=-17; gpu.W[0][0]=20; gpu.W[0][1]=-16;
        gpu.V[1][0]=9; gpu.V[1][1]=-9; gpu.W[1][0]=19; gpu.W[1][1]=8;
        gpu.V[2][0]=-15; gpu.V[2][1]=15; gpu.W[2][0]=-19; gpu.W[2][1]=4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11852;
        gpu.V[0][0]=-5; gpu.V[0][1]=-1; gpu.W[0][0]=14; gpu.W[0][1]=8;
        gpu.V[1][0]=-10; gpu.V[1][1]=-17; gpu.W[1][0]=-14; gpu.W[1][1]=4;
        gpu.V[2][0]=10; gpu.V[2][1]=2; gpu.W[2][0]=-7; gpu.W[2][1]=-4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv1_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 36158;
        gpu.V[0][0]=-10; gpu.V[0][1]=-20; gpu.W[0][0]=-7; gpu.W[0][1]=-14;
        gpu.V[1][0]=-16; gpu.V[1][1]=-18; gpu.W[1][0]=-1; gpu.W[1][1]=12;
        gpu.V[2][0]=8; gpu.V[2][1]=16; gpu.W[2][0]=7; gpu.W[2][1]=7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv1_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12649;
        gpu.V[0][0]=-14; gpu.V[0][1]=-14; gpu.W[0][0]=16; gpu.W[0][1]=-11;
        gpu.V[1][0]=8; gpu.V[1][1]=8; gpu.W[1][0]=-14; gpu.W[1][1]=-14;
        gpu.V[2][0]=18; gpu.V[2][1]=-19; gpu.W[2][0]=19; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13901;
        gpu.V[0][0]=17; gpu.V[0][1]=-11; gpu.W[0][0]=0; gpu.W[0][1]=12;
        gpu.V[1][0]=-20; gpu.V[1][1]=16; gpu.W[1][0]=-13; gpu.W[1][1]=10;
        gpu.V[2][0]=-3; gpu.V[2][1]=-4; gpu.W[2][0]=17; gpu.W[2][1]=-17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10623;
        gpu.V[0][0]=-7; gpu.V[0][1]=-17; gpu.W[0][0]=-18; gpu.W[0][1]=3;
        gpu.V[1][0]=14; gpu.V[1][1]=10; gpu.W[1][0]=0; gpu.W[1][1]=0;
        gpu.V[2][0]=-17; gpu.V[2][1]=5; gpu.W[2][0]=4; gpu.W[2][1]=8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11089;
        gpu.V[0][0]=11; gpu.V[0][1]=-10; gpu.W[0][0]=-3; gpu.W[0][1]=3;
        gpu.V[1][0]=-9; gpu.V[1][1]=-8; gpu.W[1][0]=2; gpu.W[1][1]=20;
        gpu.V[2][0]=5; gpu.V[2][1]=10; gpu.W[2][0]=8; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 32083;
        gpu.V[0][0]=-9; gpu.V[0][1]=0; gpu.W[0][0]=-3; gpu.W[0][1]=0;
        gpu.V[1][0]=-9; gpu.V[1][1]=-9; gpu.W[1][0]=17; gpu.W[1][1]=0;
        gpu.V[2][0]=18; gpu.V[2][1]=10; gpu.W[2][0]=-3; gpu.W[2][1]=3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12450;
        gpu.V[0][0]=-12; gpu.V[0][1]=0; gpu.W[0][0]=-11; gpu.W[0][1]=-7;
        gpu.V[1][0]=4; gpu.V[1][1]=19; gpu.W[1][0]=-8; gpu.W[1][1]=11;
        gpu.V[2][0]=18; gpu.V[2][1]=-9; gpu.W[2][0]=14; gpu.W[2][1]=-7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 49876;
        gpu.V[0][0]=20; gpu.V[0][1]=0; gpu.W[0][0]=10; gpu.W[0][1]=-15;
        gpu.V[1][0]=-16; gpu.V[1][1]=14; gpu.W[1][0]=-8; gpu.W[1][1]=19;
        gpu.V[2][0]=-10; gpu.V[2][1]=-18; gpu.W[2][0]=-18; gpu.W[2][1]=-14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_Cw_TN");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3321;
        gpu.V[0][0]=0; gpu.V[0][1]=2; gpu.W[0][0]=0; gpu.W[0][1]=-5;
        gpu.V[1][0]=13; gpu.V[1][1]=-13; gpu.W[1][0]=3; gpu.W[1][1]=6;
        gpu.V[2][0]=-20; gpu.V[2][1]=16; gpu.W[2][0]=-1; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3146;
        gpu.V[0][0]=-13; gpu.V[0][1]=6; gpu.W[0][0]=-16; gpu.W[0][1]=-17;
        gpu.V[1][0]=1; gpu.V[1][1]=-14; gpu.W[1][0]=4; gpu.W[1][1]=-5;
        gpu.V[2][0]=6; gpu.V[2][1]=-6; gpu.W[2][0]=11; gpu.W[2][1]=17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 23264;
        gpu.V[0][0]=0; gpu.V[0][1]=-18; gpu.W[0][0]=2; gpu.W[0][1]=-2;
        gpu.V[1][0]=15; gpu.V[1][1]=18; gpu.W[1][0]=0; gpu.W[1][1]=0;
        gpu.V[2][0]=14; gpu.V[2][1]=-9; gpu.W[2][0]=4; gpu.W[2][1]=5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7505;
        gpu.V[0][0]=15; gpu.V[0][1]=12; gpu.W[0][0]=-9; gpu.W[0][1]=9;
        gpu.V[1][0]=19; gpu.V[1][1]=-14; gpu.W[1][0]=15; gpu.W[1][1]=-1;
        gpu.V[2][0]=8; gpu.V[2][1]=10; gpu.W[2][0]=11; gpu.W[2][1]=-11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 67364;
        gpu.V[0][0]=-2; gpu.V[0][1]=-3; gpu.W[0][0]=-10; gpu.W[0][1]=-15;
        gpu.V[1][0]=18; gpu.V[1][1]=19; gpu.W[1][0]=-5; gpu.W[1][1]=9;
        gpu.V[2][0]=19; gpu.V[2][1]=-12; gpu.W[2][0]=12; gpu.W[2][1]=18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 707;
        gpu.V[0][0]=9; gpu.V[0][1]=20; gpu.W[0][0]=-7; gpu.W[0][1]=1;
        gpu.V[1][0]=11; gpu.V[1][1]=10; gpu.W[1][0]=8; gpu.W[1][1]=15;
        gpu.V[2][0]=-6; gpu.V[2][1]=8; gpu.W[2][0]=6; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12226;
        gpu.V[0][0]=-18; gpu.V[0][1]=18; gpu.W[0][0]=-13; gpu.W[0][1]=2;
        gpu.V[1][0]=-15; gpu.V[1][1]=-18; gpu.W[1][0]=-5; gpu.W[1][1]=-6;
        gpu.V[2][0]=-4; gpu.V[2][1]=-6; gpu.W[2][0]=-14; gpu.W[2][1]=6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2-_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 4675;
        gpu.V[0][0]=-5; gpu.V[0][1]=11; gpu.W[0][0]=18; gpu.W[0][1]=-18;
        gpu.V[1][0]=1; gpu.V[1][1]=17; gpu.W[1][0]=-11; gpu.W[1][1]=-14;
        gpu.V[2][0]=-3; gpu.V[2][1]=13; gpu.W[2][0]=13; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2o");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13200;
        gpu.V[0][0]=-14; gpu.V[0][1]=-18; gpu.W[0][0]=0; gpu.W[0][1]=-5;
        gpu.V[1][0]=19; gpu.V[1][1]=-8; gpu.W[1][0]=20; gpu.W[1][1]=6;
        gpu.V[2][0]=-20; gpu.V[2][1]=-5; gpu.W[2][0]=-19; gpu.W[2][1]=9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T2_Q2o_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 16968;
        gpu.V[0][0]=0; gpu.V[0][1]=0; gpu.W[0][0]=3; gpu.W[0][1]=12;
        gpu.V[1][0]=-1; gpu.V[1][1]=-7; gpu.W[1][0]=-18; gpu.W[1][1]=19;
        gpu.V[2][0]=-20; gpu.V[2][1]=12; gpu.W[2][0]=15; gpu.W[2][1]=-17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv0_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 1585;
        gpu.V[0][0]=-13; gpu.V[0][1]=13; gpu.W[0][0]=-15; gpu.W[0][1]=5;
        gpu.V[1][0]=-14; gpu.V[1][1]=9; gpu.W[1][0]=16; gpu.W[1][1]=-2;
        gpu.V[2][0]=17; gpu.V[2][1]=-17; gpu.W[2][0]=6; gpu.W[2][1]=-11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 85172;
        gpu.V[0][0]=2; gpu.V[0][1]=-15; gpu.W[0][0]=12; gpu.W[0][1]=1;
        gpu.V[1][0]=0; gpu.V[1][1]=4; gpu.W[1][0]=-8; gpu.W[1][1]=-1;
        gpu.V[2][0]=0; gpu.V[2][1]=-17; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv1_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10857;
        gpu.V[0][0]=7; gpu.V[0][1]=7; gpu.W[0][0]=9; gpu.W[0][1]=-13;
        gpu.V[1][0]=18; gpu.V[1][1]=10; gpu.W[1][0]=-4; gpu.W[1][1]=8;
        gpu.V[2][0]=-9; gpu.V[2][1]=-9; gpu.W[2][0]=4; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv1_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 96328;
        gpu.V[0][0]=-11; gpu.V[0][1]=5; gpu.W[0][0]=-6; gpu.W[0][1]=18;
        gpu.V[1][0]=-3; gpu.V[1][1]=-3; gpu.W[1][0]=-8; gpu.W[1][1]=-15;
        gpu.V[2][0]=20; gpu.V[2][1]=20; gpu.W[2][0]=6; gpu.W[2][1]=6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv1_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7288;
        gpu.V[0][0]=-6; gpu.V[0][1]=-16; gpu.W[0][0]=10; gpu.W[0][1]=4;
        gpu.V[1][0]=20; gpu.V[1][1]=20; gpu.W[1][0]=-17; gpu.W[1][1]=-7;
        gpu.V[2][0]=5; gpu.V[2][1]=18; gpu.W[2][0]=11; gpu.W[2][1]=17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 48779;
        gpu.V[0][0]=0; gpu.V[0][1]=-13; gpu.W[0][0]=-3; gpu.W[0][1]=1;
        gpu.V[1][0]=-18; gpu.V[1][1]=17; gpu.W[1][0]=0; gpu.W[1][1]=0;
        gpu.V[2][0]=15; gpu.V[2][1]=1; gpu.W[2][0]=3; gpu.W[2][1]=10;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 547;
        gpu.V[0][0]=-16; gpu.V[0][1]=15; gpu.W[0][0]=7; gpu.W[0][1]=7;
        gpu.V[1][0]=-5; gpu.V[1][1]=-1; gpu.W[1][0]=-17; gpu.W[1][1]=-17;
        gpu.V[2][0]=17; gpu.V[2][1]=0; gpu.W[2][0]=-10; gpu.W[2][1]=20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 32951;
        gpu.V[0][0]=15; gpu.V[0][1]=-20; gpu.W[0][0]=3; gpu.W[0][1]=-4;
        gpu.V[1][0]=-12; gpu.V[1][1]=-5; gpu.W[1][0]=-2; gpu.W[1][1]=19;
        gpu.V[2][0]=1; gpu.V[2][1]=6; gpu.W[2][0]=-15; gpu.W[2][1]=20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11193;
        gpu.V[0][0]=11; gpu.V[0][1]=-3; gpu.W[0][0]=1; gpu.W[0][1]=10;
        gpu.V[1][0]=0; gpu.V[1][1]=9; gpu.W[1][0]=0; gpu.W[1][1]=-10;
        gpu.V[2][0]=-16; gpu.V[2][1]=-10; gpu.W[2][0]=-7; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cv_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7277;
        gpu.V[0][0]=12; gpu.V[0][1]=20; gpu.W[0][0]=-11; gpu.W[0][1]=0;
        gpu.V[1][0]=-18; gpu.V[1][1]=-18; gpu.W[1][0]=13; gpu.W[1][1]=5;
        gpu.V[2][0]=-20; gpu.V[2][1]=19; gpu.W[2][0]=16; gpu.W[2][1]=-3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8785;
        gpu.V[0][0]=11; gpu.V[0][1]=20; gpu.W[0][0]=0; gpu.W[0][1]=0;
        gpu.V[1][0]=16; gpu.V[1][1]=-7; gpu.W[1][0]=18; gpu.W[1][1]=8;
        gpu.V[2][0]=-19; gpu.V[2][1]=10; gpu.W[2][0]=-14; gpu.W[2][1]=10;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10947;
        gpu.V[0][0]=-2; gpu.V[0][1]=1; gpu.W[0][0]=1; gpu.W[0][1]=-8;
        gpu.V[1][0]=18; gpu.V[1][1]=-13; gpu.W[1][0]=8; gpu.W[1][1]=-4;
        gpu.V[2][0]=7; gpu.V[2][1]=-9; gpu.W[2][0]=-2; gpu.W[2][1]=16;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 40554;
        gpu.V[0][0]=-19; gpu.V[0][1]=-4; gpu.W[0][0]=4; gpu.W[0][1]=4;
        gpu.V[1][0]=-13; gpu.V[1][1]=-13; gpu.W[1][0]=-14; gpu.W[1][1]=-14;
        gpu.V[2][0]=14; gpu.V[2][1]=-9; gpu.W[2][0]=-7; gpu.W[2][1]=11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 4961;
        gpu.V[0][0]=17; gpu.V[0][1]=0; gpu.W[0][0]=19; gpu.W[0][1]=0;
        gpu.V[1][0]=-12; gpu.V[1][1]=2; gpu.W[1][0]=-8; gpu.W[1][1]=15;
        gpu.V[2][0]=-13; gpu.V[2][1]=6; gpu.W[2][0]=3; gpu.W[2][1]=-14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 60547;
        gpu.V[0][0]=5; gpu.V[0][1]=5; gpu.W[0][0]=-17; gpu.W[0][1]=-18;
        gpu.V[1][0]=6; gpu.V[1][1]=14; gpu.W[1][0]=-16; gpu.W[1][1]=-12;
        gpu.V[2][0]=-4; gpu.V[2][1]=-4; gpu.W[2][0]=6; gpu.W[2][1]=5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_SR_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11413;
        gpu.V[0][0]=4; gpu.V[0][1]=3; gpu.W[0][0]=-6; gpu.W[0][1]=9;
        gpu.V[1][0]=-7; gpu.V[1][1]=-17; gpu.W[1][0]=-2; gpu.W[1][1]=-11;
        gpu.V[2][0]=-1; gpu.V[2][1]=3; gpu.W[2][0]=11; gpu.W[2][1]=9;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_SR_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 39988;
        gpu.V[0][0]=-14; gpu.V[0][1]=1; gpu.W[0][0]=6; gpu.W[0][1]=1;
        gpu.V[1][0]=-17; gpu.V[1][1]=-14; gpu.W[1][0]=6; gpu.W[1][1]=-5;
        gpu.V[2][0]=-3; gpu.V[2][1]=1; gpu.W[2][0]=-19; gpu.W[2][1]=1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_SR_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6945;
        gpu.V[0][0]=-12; gpu.V[0][1]=-11; gpu.W[0][0]=5; gpu.W[0][1]=4;
        gpu.V[1][0]=-17; gpu.V[1][1]=-16; gpu.W[1][0]=-12; gpu.W[1][1]=-10;
        gpu.V[2][0]=-1; gpu.V[2][1]=0; gpu.W[2][0]=-10; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,1,2)_Q2+_SR_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 61833;
        gpu.V[0][0]=16; gpu.V[0][1]=8; gpu.W[0][0]=13; gpu.W[0][1]=-19;
        gpu.V[1][0]=15; gpu.V[1][1]=13; gpu.W[1][0]=18; gpu.W[1][1]=15;
        gpu.V[2][0]=-10; gpu.V[2][1]=-5; gpu.W[2][0]=-14; gpu.W[2][1]=18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3696;
        gpu.V[0][0]=4; gpu.V[0][1]=-12; gpu.W[0][0]=17; gpu.W[0][1]=-17;
        gpu.V[1][0]=6; gpu.V[1][1]=11; gpu.W[1][0]=4; gpu.W[1][1]=8;
        gpu.V[2][0]=-20; gpu.V[2][1]=-8; gpu.W[2][0]=-9; gpu.W[2][1]=-11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 18513;
        gpu.V[0][0]=17; gpu.V[0][1]=-7; gpu.W[0][0]=10; gpu.W[0][1]=17;
        gpu.V[1][0]=-7; gpu.V[1][1]=16; gpu.W[1][0]=-10; gpu.W[1][1]=-17;
        gpu.V[2][0]=-19; gpu.V[2][1]=3; gpu.W[2][0]=20; gpu.W[2][1]=6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12807;
        gpu.V[0][0]=-20; gpu.V[0][1]=-1; gpu.W[0][0]=6; gpu.W[0][1]=-7;
        gpu.V[1][0]=-18; gpu.V[1][1]=-4; gpu.W[1][0]=0; gpu.W[1][1]=-13;
        gpu.V[2][0]=14; gpu.V[2][1]=8; gpu.W[2][0]=-3; gpu.W[2][1]=20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 3363;
        gpu.V[0][0]=8; gpu.V[0][1]=-7; gpu.W[0][0]=6; gpu.W[0][1]=-12;
        gpu.V[1][0]=0; gpu.V[1][1]=-7; gpu.W[1][0]=-1; gpu.W[1][1]=1;
        gpu.V[2][0]=9; gpu.V[2][1]=-14; gpu.W[2][0]=11; gpu.W[2][1]=-11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 14346;
        gpu.V[0][0]=-12; gpu.V[0][1]=14; gpu.W[0][0]=-10; gpu.W[0][1]=3;
        gpu.V[1][0]=-16; gpu.V[1][1]=-2; gpu.W[1][0]=19; gpu.W[1][1]=-7;
        gpu.V[2][0]=-20; gpu.V[2][1]=12; gpu.W[2][0]=-20; gpu.W[2][1]=12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 83146;
        gpu.V[0][0]=-4; gpu.V[0][1]=-6; gpu.W[0][0]=5; gpu.W[0][1]=9;
        gpu.V[1][0]=12; gpu.V[1][1]=10; gpu.W[1][0]=20; gpu.W[1][1]=3;
        gpu.V[2][0]=-4; gpu.V[2][1]=-6; gpu.W[2][0]=-18; gpu.W[2][1]=-3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(1,3)_Q2+_SR_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6283;
        gpu.V[0][0]=11; gpu.V[0][1]=5; gpu.W[0][0]=13; gpu.W[0][1]=-15;
        gpu.V[1][0]=16; gpu.V[1][1]=5; gpu.W[1][0]=-17; gpu.W[1][1]=20;
        gpu.V[2][0]=-10; gpu.V[2][1]=-17; gpu.W[2][0]=13; gpu.W[2][1]=-15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7267;
        gpu.V[0][0]=-4; gpu.V[0][1]=-7; gpu.W[0][0]=-18; gpu.W[0][1]=-19;
        gpu.V[1][0]=-7; gpu.V[1][1]=12; gpu.W[1][0]=15; gpu.W[1][1]=14;
        gpu.V[2][0]=14; gpu.V[2][1]=-8; gpu.W[2][0]=2; gpu.W[2][1]=1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q1_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 99270;
        gpu.V[0][0]=-5; gpu.V[0][1]=-5; gpu.W[0][0]=-14; gpu.W[0][1]=-5;
        gpu.V[1][0]=-9; gpu.V[1][1]=2; gpu.W[1][0]=-8; gpu.W[1][1]=-2;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=8; gpu.W[2][1]=6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q1_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 15056;
        gpu.V[0][0]=-15; gpu.V[0][1]=5; gpu.W[0][0]=9; gpu.W[0][1]=-3;
        gpu.V[1][0]=5; gpu.V[1][1]=-11; gpu.W[1][0]=-5; gpu.W[1][1]=-3;
        gpu.V[2][0]=17; gpu.V[2][1]=18; gpu.W[2][0]=-20; gpu.W[2][1]=-3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q1_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 95974;
        gpu.V[0][0]=-4; gpu.V[0][1]=-8; gpu.W[0][0]=-12; gpu.W[0][1]=-3;
        gpu.V[1][0]=-5; gpu.V[1][1]=-8; gpu.W[1][0]=10; gpu.W[1][1]=-3;
        gpu.V[2][0]=0; gpu.V[2][1]=20; gpu.W[2][0]=19; gpu.W[2][1]=-3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q1_SR");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8801;
        gpu.V[0][0]=-14; gpu.V[0][1]=-16; gpu.W[0][0]=-19; gpu.W[0][1]=-17;
        gpu.V[1][0]=18; gpu.V[1][1]=-17; gpu.W[1][0]=-7; gpu.W[1][1]=-2;
        gpu.V[2][0]=7; gpu.V[2][1]=3; gpu.W[2][0]=18; gpu.W[2][1]=1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7292;
        gpu.V[0][0]=-5; gpu.V[0][1]=16; gpu.W[0][0]=-4; gpu.W[0][1]=11;
        gpu.V[1][0]=1; gpu.V[1][1]=-2; gpu.W[1][0]=-13; gpu.W[1][1]=11;
        gpu.V[2][0]=-9; gpu.V[2][1]=5; gpu.W[2][0]=20; gpu.W[2][1]=-16;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10567;
        gpu.V[0][0]=-4; gpu.V[0][1]=-9; gpu.W[0][0]=7; gpu.W[0][1]=-14;
        gpu.V[1][0]=7; gpu.V[1][1]=-4; gpu.W[1][0]=5; gpu.W[1][1]=-8;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=0; gpu.W[2][1]=18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12616;
        gpu.V[0][0]=7; gpu.V[0][1]=8; gpu.W[0][0]=-2; gpu.W[0][1]=14;
        gpu.V[1][0]=17; gpu.V[1][1]=17; gpu.W[1][0]=-6; gpu.W[1][1]=-11;
        gpu.V[2][0]=-12; gpu.V[2][1]=-12; gpu.W[2][0]=-3; gpu.W[2][1]=-19;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 18333;
        gpu.V[0][0]=0; gpu.V[0][1]=2; gpu.W[0][0]=0; gpu.W[0][1]=3;
        gpu.V[1][0]=5; gpu.V[1][1]=16; gpu.W[1][0]=6; gpu.W[1][1]=19;
        gpu.V[2][0]=0; gpu.V[2][1]=-19; gpu.W[2][0]=2; gpu.W[2][1]=10;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 4181;
        gpu.V[0][0]=13; gpu.V[0][1]=13; gpu.W[0][0]=-20; gpu.W[0][1]=20;
        gpu.V[1][0]=2; gpu.V[1][1]=16; gpu.W[1][0]=20; gpu.W[1][1]=-11;
        gpu.V[2][0]=-11; gpu.V[2][1]=-20; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5231;
        gpu.V[0][0]=1; gpu.V[0][1]=-3; gpu.W[0][0]=0; gpu.W[0][1]=1;
        gpu.V[1][0]=17; gpu.V[1][1]=14; gpu.W[1][0]=0; gpu.W[1][1]=-9;
        gpu.V[2][0]=-10; gpu.V[2][1]=-2; gpu.W[2][0]=-11; gpu.W[2][1]=-14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 22117;
        gpu.V[0][0]=13; gpu.V[0][1]=-6; gpu.W[0][0]=-13; gpu.W[0][1]=6;
        gpu.V[1][0]=-9; gpu.V[1][1]=-13; gpu.W[1][0]=8; gpu.W[1][1]=-6;
        gpu.V[2][0]=-4; gpu.V[2][1]=14; gpu.W[2][0]=13; gpu.W[2][1]=-6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 763;
        gpu.V[0][0]=6; gpu.V[0][1]=12; gpu.W[0][0]=9; gpu.W[0][1]=18;
        gpu.V[1][0]=14; gpu.V[1][1]=-17; gpu.W[1][0]=3; gpu.W[1][1]=-18;
        gpu.V[2][0]=-19; gpu.V[2][1]=9; gpu.W[2][0]=10; gpu.W[2][1]=-18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10354;
        gpu.V[0][0]=-8; gpu.V[0][1]=4; gpu.W[0][0]=-7; gpu.W[0][1]=12;
        gpu.V[1][0]=-9; gpu.V[1][1]=-6; gpu.W[1][0]=-11; gpu.W[1][1]=-10;
        gpu.V[2][0]=9; gpu.V[2][1]=-8; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6485;
        gpu.V[0][0]=13; gpu.V[0][1]=-17; gpu.W[0][0]=18; gpu.W[0][1]=-18;
        gpu.V[1][0]=18; gpu.V[1][1]=15; gpu.W[1][0]=8; gpu.W[1][1]=1;
        gpu.V[2][0]=13; gpu.V[2][1]=-20; gpu.W[2][0]=-19; gpu.W[2][1]=19;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8316;
        gpu.V[0][0]=-19; gpu.V[0][1]=-19; gpu.W[0][0]=4; gpu.W[0][1]=13;
        gpu.V[1][0]=-15; gpu.V[1][1]=-15; gpu.W[1][0]=13; gpu.W[1][1]=13;
        gpu.V[2][0]=19; gpu.V[2][1]=12; gpu.W[2][0]=14; gpu.W[2][1]=-7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8705;
        gpu.V[0][0]=-20; gpu.V[0][1]=-5; gpu.W[0][0]=9; gpu.W[0][1]=3;
        gpu.V[1][0]=10; gpu.V[1][1]=10; gpu.W[1][0]=-13; gpu.W[1][1]=-8;
        gpu.V[2][0]=11; gpu.V[2][1]=4; gpu.W[2][0]=15; gpu.W[2][1]=-15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_SR");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 1632;
        gpu.V[0][0]=-2; gpu.V[0][1]=19; gpu.W[0][0]=-5; gpu.W[0][1]=-14;
        gpu.V[1][0]=9; gpu.V[1][1]=1; gpu.W[1][0]=18; gpu.W[1][1]=13;
        gpu.V[2][0]=-2; gpu.V[2][1]=-8; gpu.W[2][0]=-4; gpu.W[2][1]=-5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_(2,2)_Q2+_SR_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7134;
        gpu.V[0][0]=11; gpu.V[0][1]=-15; gpu.W[0][0]=19; gpu.W[0][1]=-20;
        gpu.V[1][0]=-19; gpu.V[1][1]=3; gpu.W[1][0]=18; gpu.W[1][1]=1;
        gpu.V[2][0]=8; gpu.V[2][1]=-7; gpu.W[2][0]=19; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 364;
        gpu.V[0][0]=-20; gpu.V[0][1]=6; gpu.W[0][0]=8; gpu.W[0][1]=-3;
        gpu.V[1][0]=-8; gpu.V[1][1]=17; gpu.W[1][0]=-15; gpu.W[1][1]=17;
        gpu.V[2][0]=4; gpu.V[2][1]=-5; gpu.W[2][0]=8; gpu.W[2][1]=-3;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q1_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 57354;
        gpu.V[0][0]=17; gpu.V[0][1]=-6; gpu.W[0][0]=-20; gpu.W[0][1]=6;
        gpu.V[1][0]=-10; gpu.V[1][1]=2; gpu.W[1][0]=-7; gpu.W[1][1]=10;
        gpu.V[2][0]=-17; gpu.V[2][1]=6; gpu.W[2][0]=6; gpu.W[2][1]=14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q1_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5172;
        gpu.V[0][0]=-4; gpu.V[0][1]=14; gpu.W[0][0]=-1; gpu.W[0][1]=-7;
        gpu.V[1][0]=-3; gpu.V[1][1]=15; gpu.W[1][0]=6; gpu.W[1][1]=8;
        gpu.V[2][0]=-19; gpu.V[2][1]=4; gpu.W[2][0]=17; gpu.W[2][1]=-4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12801;
        gpu.V[0][0]=0; gpu.V[0][1]=-19; gpu.W[0][0]=13; gpu.W[0][1]=12;
        gpu.V[1][0]=12; gpu.V[1][1]=7; gpu.W[1][0]=2; gpu.W[1][1]=-18;
        gpu.V[2][0]=-20; gpu.V[2][1]=-6; gpu.W[2][0]=11; gpu.W[2][1]=16;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 46380;
        gpu.V[0][0]=-13; gpu.V[0][1]=-13; gpu.W[0][0]=-12; gpu.W[0][1]=-3;
        gpu.V[1][0]=8; gpu.V[1][1]=-3; gpu.W[1][0]=15; gpu.W[1][1]=-12;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=19; gpu.W[2][1]=-17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 10306;
        gpu.V[0][0]=-11; gpu.V[0][1]=-11; gpu.W[0][0]=5; gpu.W[0][1]=-19;
        gpu.V[1][0]=-11; gpu.V[1][1]=-13; gpu.W[1][0]=-14; gpu.W[1][1]=-13;
        gpu.V[2][0]=13; gpu.V[2][1]=13; gpu.W[2][0]=19; gpu.W[2][1]=-6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 19463;
        gpu.V[0][0]=14; gpu.V[0][1]=2; gpu.W[0][0]=12; gpu.W[0][1]=-6;
        gpu.V[1][0]=17; gpu.V[1][1]=12; gpu.W[1][0]=17; gpu.W[1][1]=-8;
        gpu.V[2][0]=-15; gpu.V[2][1]=-9; gpu.W[2][0]=-8; gpu.W[2][1]=4;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 35739;
        gpu.V[0][0]=0; gpu.V[0][1]=16; gpu.W[0][0]=0; gpu.W[0][1]=-6;
        gpu.V[1][0]=9; gpu.V[1][1]=-3; gpu.W[1][0]=15; gpu.W[1][1]=15;
        gpu.V[2][0]=-17; gpu.V[2][1]=4; gpu.W[2][0]=-7; gpu.W[2][1]=-13;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5918;
        gpu.V[0][0]=12; gpu.V[0][1]=18; gpu.W[0][0]=-7; gpu.W[0][1]=-5;
        gpu.V[1][0]=-11; gpu.V[1][1]=-14; gpu.W[1][0]=6; gpu.W[1][1]=4;
        gpu.V[2][0]=0; gpu.V[2][1]=13; gpu.W[2][0]=7; gpu.W[2][1]=5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 51067;
        gpu.V[0][0]=-13; gpu.V[0][1]=10; gpu.W[0][0]=-16; gpu.W[0][1]=-2;
        gpu.V[1][0]=4; gpu.V[1][1]=-17; gpu.W[1][0]=2; gpu.W[1][1]=2;
        gpu.V[2][0]=-16; gpu.V[2][1]=-16; gpu.W[2][0]=-19; gpu.W[2][1]=-19;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9290;
        gpu.V[0][0]=0; gpu.V[0][1]=18; gpu.W[0][0]=0; gpu.W[0][1]=-13;
        gpu.V[1][0]=10; gpu.V[1][1]=1; gpu.W[1][0]=19; gpu.W[1][1]=3;
        gpu.V[2][0]=-16; gpu.V[2][1]=8; gpu.W[2][0]=-1; gpu.W[2][1]=-10;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 43156;
        gpu.V[0][0]=9; gpu.V[0][1]=1; gpu.W[0][0]=-18; gpu.W[0][1]=-3;
        gpu.V[1][0]=19; gpu.V[1][1]=7; gpu.W[1][0]=12; gpu.W[1][1]=12;
        gpu.V[2][0]=18; gpu.V[2][1]=10; gpu.W[2][0]=-18; gpu.W[2][1]=-12;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2+_TN");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 6996;
        gpu.V[0][0]=4; gpu.V[0][1]=-14; gpu.W[0][0]=1; gpu.W[0][1]=-12;
        gpu.V[1][0]=-17; gpu.V[1][1]=-5; gpu.W[1][0]=-6; gpu.W[1][1]=-14;
        gpu.V[2][0]=7; gpu.V[2][1]=0; gpu.W[2][0]=-11; gpu.W[2][1]=6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8055;
        gpu.V[0][0]=6; gpu.V[0][1]=3; gpu.W[0][0]=-7; gpu.W[0][1]=16;
        gpu.V[1][0]=-18; gpu.V[1][1]=-13; gpu.W[1][0]=-5; gpu.W[1][1]=-10;
        gpu.V[2][0]=-16; gpu.V[2][1]=-4; gpu.W[2][0]=-18; gpu.W[2][1]=-1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7208;
        gpu.V[0][0]=0; gpu.V[0][1]=0; gpu.W[0][0]=19; gpu.W[0][1]=7;
        gpu.V[1][0]=10; gpu.V[1][1]=-4; gpu.W[1][0]=-7; gpu.W[1][1]=18;
        gpu.V[2][0]=-6; gpu.V[2][1]=12; gpu.W[2][0]=3; gpu.W[2][1]=-8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv0_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 2851;
        gpu.V[0][0]=-12; gpu.V[0][1]=-15; gpu.W[0][0]=4; gpu.W[0][1]=14;
        gpu.V[1][0]=7; gpu.V[1][1]=-2; gpu.W[1][0]=12; gpu.W[1][1]=-5;
        gpu.V[2][0]=0; gpu.V[2][1]=0; gpu.W[2][0]=12; gpu.W[2][1]=-2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 848;
        gpu.V[0][0]=-11; gpu.V[0][1]=0; gpu.W[0][0]=11; gpu.W[0][1]=-1;
        gpu.V[1][0]=1; gpu.V[1][1]=0; gpu.W[1][0]=1; gpu.W[1][1]=14;
        gpu.V[2][0]=18; gpu.V[2][1]=20; gpu.W[2][0]=-20; gpu.W[2][1]=5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12850;
        gpu.V[0][0]=-10; gpu.V[0][1]=14; gpu.W[0][0]=-7; gpu.W[0][1]=8;
        gpu.V[1][0]=-17; gpu.V[1][1]=10; gpu.W[1][0]=-2; gpu.W[1][1]=-12;
        gpu.V[2][0]=5; gpu.V[2][1]=-7; gpu.W[2][0]=15; gpu.W[2][1]=18;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 1710;
        gpu.V[0][0]=-9; gpu.V[0][1]=-11; gpu.W[0][0]=-11; gpu.W[0][1]=8;
        gpu.V[1][0]=-16; gpu.V[1][1]=-10; gpu.W[1][0]=-14; gpu.W[1][1]=16;
        gpu.V[2][0]=4; gpu.V[2][1]=4; gpu.W[2][0]=18; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 35120;
        gpu.V[0][0]=9; gpu.V[0][1]=10; gpu.W[0][0]=13; gpu.W[0][1]=-19;
        gpu.V[1][0]=-6; gpu.V[1][1]=-4; gpu.W[1][0]=-10; gpu.W[1][1]=19;
        gpu.V[2][0]=6; gpu.V[2][1]=0; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 11838;
        gpu.V[0][0]=-1; gpu.V[0][1]=17; gpu.W[0][0]=-1; gpu.W[0][1]=-1;
        gpu.V[1][0]=18; gpu.V[1][1]=8; gpu.W[1][0]=9; gpu.W[1][1]=9;
        gpu.V[2][0]=-9; gpu.V[2][1]=-5; gpu.W[2][0]=16; gpu.W[2][1]=1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8410;
        gpu.V[0][0]=2; gpu.V[0][1]=2; gpu.W[0][0]=1; gpu.W[0][1]=1;
        gpu.V[1][0]=5; gpu.V[1][1]=-9; gpu.W[1][0]=13; gpu.W[1][1]=-8;
        gpu.V[2][0]=-5; gpu.V[2][1]=6; gpu.W[2][0]=-19; gpu.W[2][1]=-7;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7576;
        gpu.V[0][0]=-5; gpu.V[0][1]=0; gpu.W[0][0]=14; gpu.W[0][1]=0;
        gpu.V[1][0]=-4; gpu.V[1][1]=16; gpu.W[1][0]=-16; gpu.W[1][1]=13;
        gpu.V[2][0]=2; gpu.V[2][1]=-3; gpu.W[2][0]=19; gpu.W[2][1]=11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cv_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 7004;
        gpu.V[0][0]=-2; gpu.V[0][1]=4; gpu.W[0][0]=-12; gpu.W[0][1]=15;
        gpu.V[1][0]=-20; gpu.V[1][1]=2; gpu.W[1][0]=-14; gpu.W[1][1]=-9;
        gpu.V[2][0]=-6; gpu.V[2][1]=-15; gpu.W[2][0]=12; gpu.W[2][1]=-10;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 29203;
        gpu.V[0][0]=-7; gpu.V[0][1]=2; gpu.W[0][0]=-12; gpu.W[0][1]=5;
        gpu.V[1][0]=19; gpu.V[1][1]=4; gpu.W[1][0]=-14; gpu.W[1][1]=-11;
        gpu.V[2][0]=0; gpu.V[2][1]=7; gpu.W[2][0]=0; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cw0_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 12439;
        gpu.V[0][0]=6; gpu.V[0][1]=-9; gpu.W[0][0]=9; gpu.W[0][1]=-15;
        gpu.V[1][0]=7; gpu.V[1][1]=-19; gpu.W[1][0]=-19; gpu.W[1][1]=-14;
        gpu.V[2][0]=-7; gpu.V[2][1]=-4; gpu.W[2][0]=19; gpu.W[2][1]=14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 46794;
        gpu.V[0][0]=-4; gpu.V[0][1]=-7; gpu.W[0][0]=-8; gpu.W[0][1]=-12;
        gpu.V[1][0]=8; gpu.V[1][1]=16; gpu.W[1][0]=3; gpu.W[1][1]=0;
        gpu.V[2][0]=-16; gpu.V[2][1]=0; gpu.W[2][0]=-16; gpu.W[2][1]=0;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cw1_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 9911;
        gpu.V[0][0]=4; gpu.V[0][1]=-2; gpu.W[0][0]=10; gpu.W[0][1]=-5;
        gpu.V[1][0]=17; gpu.V[1][1]=-14; gpu.W[1][0]=-4; gpu.W[1][1]=-2;
        gpu.V[2][0]=-8; gpu.V[2][1]=-1; gpu.W[2][0]=10; gpu.W[2][1]=8;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_Cw_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 5966;
        gpu.V[0][0]=15; gpu.V[0][1]=-8; gpu.W[0][0]=18; gpu.W[0][1]=-15;
        gpu.V[1][0]=-9; gpu.V[1][1]=-7; gpu.W[1][0]=9; gpu.W[1][1]=7;
        gpu.V[2][0]=0; gpu.V[2][1]=-12; gpu.W[2][0]=-12; gpu.W[2][1]=11;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2-_D00");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 88475;
        gpu.V[0][0]=-2; gpu.V[0][1]=10; gpu.W[0][0]=-4; gpu.W[0][1]=14;
        gpu.V[1][0]=-12; gpu.V[1][1]=-13; gpu.W[1][0]=6; gpu.W[1][1]=7;
        gpu.V[2][0]=-18; gpu.V[2][1]=-9; gpu.W[2][0]=12; gpu.W[2][1]=-15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T4_Q2o");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 73915;
        gpu.V[0][0]=7; gpu.V[0][1]=6; gpu.W[0][0]=-14; gpu.W[0][1]=-13;
        gpu.V[1][0]=7; gpu.V[1][1]=10; gpu.W[1][0]=5; gpu.W[1][1]=5;
        gpu.V[2][0]=-8; gpu.V[2][1]=11; gpu.W[2][0]=-8; gpu.W[2][1]=15;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_(2,4)_Q2+");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 13061;
        gpu.V[0][0]=15; gpu.V[0][1]=19; gpu.W[0][0]=0; gpu.W[0][1]=4;
        gpu.V[1][0]=10; gpu.V[1][1]=1; gpu.W[1][0]=16; gpu.W[1][1]=3;
        gpu.V[2][0]=-11; gpu.V[2][1]=15; gpu.W[2][0]=1; gpu.W[2][1]=-5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 1297;
        gpu.V[0][0]=-14; gpu.V[0][1]=-3; gpu.W[0][0]=-19; gpu.W[0][1]=13;
        gpu.V[1][0]=10; gpu.V[1][1]=13; gpu.W[1][0]=15; gpu.W[1][1]=19;
        gpu.V[2][0]=20; gpu.V[2][1]=4; gpu.W[2][0]=-2; gpu.W[2][1]=-1;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cv");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 26102;
        gpu.V[0][0]=19; gpu.V[0][1]=0; gpu.W[0][0]=-13; gpu.W[0][1]=2;
        gpu.V[1][0]=0; gpu.V[1][1]=-15; gpu.W[1][0]=15; gpu.W[1][1]=-11;
        gpu.V[2][0]=0; gpu.V[2][1]=5; gpu.W[2][0]=-6; gpu.W[2][1]=-14;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cv1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 39802;
        gpu.V[0][0]=0; gpu.V[0][1]=-13; gpu.W[0][0]=-14; gpu.W[0][1]=-17;
        gpu.V[1][0]=-17; gpu.V[1][1]=3; gpu.W[1][0]=-3; gpu.W[1][1]=4;
        gpu.V[2][0]=0; gpu.V[2][1]=1; gpu.W[2][0]=8; gpu.W[2][1]=-5;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cv1_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 16434;
        gpu.V[0][0]=-14; gpu.V[0][1]=-8; gpu.W[0][0]=-1; gpu.W[0][1]=-2;
        gpu.V[1][0]=-4; gpu.V[1][1]=16; gpu.W[1][0]=-15; gpu.W[1][1]=9;
        gpu.V[2][0]=2; gpu.V[2][1]=-7; gpu.W[2][0]=12; gpu.W[2][1]=17;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cv_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 87904;
        gpu.V[0][0]=-7; gpu.V[0][1]=5; gpu.W[0][0]=1; gpu.W[0][1]=0;
        gpu.V[1][0]=-19; gpu.V[1][1]=-10; gpu.W[1][0]=-1; gpu.W[1][1]=0;
        gpu.V[2][0]=11; gpu.V[2][1]=-3; gpu.W[2][0]=-4; gpu.W[2][1]=-6;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cv_Cw1");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 8564;
        gpu.V[0][0]=1; gpu.V[0][1]=6; gpu.W[0][0]=14; gpu.W[0][1]=12;
        gpu.V[1][0]=1; gpu.V[1][1]=-2; gpu.W[1][0]=-6; gpu.W[1][1]=11;
        gpu.V[2][0]=18; gpu.V[2][1]=-17; gpu.W[2][0]=9; gpu.W[2][1]=-20;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cw");
    }
    {
        ftk2::TriCaseV2GPU gpu; gpu.seed = 2458;
        gpu.V[0][0]=-16; gpu.V[0][1]=17; gpu.W[0][0]=17; gpu.W[0][1]=17;
        gpu.V[1][0]=1; gpu.V[1][1]=-1; gpu.W[1][0]=-3; gpu.W[1][1]=-2;
        gpu.V[2][0]=18; gpu.V[2][1]=17; gpu.W[2][0]=-2; gpu.W[2][1]=-2;
        __int128 V128[3][2], W128[3][2];
        for(int i=0;i<3;i++) for(int j=0;j<2;j++){V128[i][j]=gpu.V[i][j]; W128[i][j]=gpu.W[i][j];}
        __int128 Q[3], P[3][3]; ftk2::compute_tri_QP_2d(V128, W128, Q, P);
        gpu.v2 = ftk2::solve_pv_tri_2d(Q, P);
        for(int k=0;k<3;k++){int dk=ftk2::effective_degree_i128(P[k],2);
            __int128 disc=(dk==2)?P[k][1]*P[k][1]-(__int128)4*P[k][0]*P[k][2]:0;
            gpu.disc_sign[k]=(disc>0)?1:(disc<0)?-1:0;}
        ASSERT_EQ_STR(ftk2::classify_case_v2_2d(gpu).category, "T6_Q2-_Cw1");
    }
}
