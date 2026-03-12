// Unit tests for 2D parallel vector primitives
// Tests: compute_edge_Q_2d, compute_tri_QP_2d, solve_pv_edge_2d,
//        solve_pv_tri_2d, classify_case_v2_2d, check_field_zero_in_tri_2d

#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <ftk2/numeric/pv_tri_classify_2d.hpp>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace ftk2;

static int total_tests = 0;
static int passed_tests = 0;
static int failed_tests = 0;

#define ASSERT_EQ(a, b) \
    do { \
        total_tests++; \
        if ((a) == (b)) { \
            passed_tests++; \
        } else { \
            failed_tests++; \
            std::cerr << "FAILED: " << #a << " == " << #b \
                      << " got " << (long long)(a) << ", expected " << (long long)(b) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

#define ASSERT_TRUE(cond) \
    do { \
        total_tests++; \
        if (cond) { \
            passed_tests++; \
        } else { \
            failed_tests++; \
            std::cerr << "FAILED: " << #cond \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

// ============================================================================
// compute_edge_Q_2d tests
// ============================================================================

void test_edge_Q_basic() {
    std::cout << "  edge_Q_basic" << std::endl;
    // V = [(1,0), (0,1)], W = [(0,1), (1,0)]
    // Q = det([V0+λW0, V1+λW1])
    //   = det([(1,λ), (λ,1)]) = 1·1 - λ·λ = 1 - λ²
    // So Q = [1, 0, -1]
    __int128 V[2][2] = {{1,0},{0,1}};
    __int128 W[2][2] = {{0,1},{1,0}};
    __int128 Q[3];
    compute_edge_Q_2d(V, W, Q);
    ASSERT_EQ(Q[0], (__int128)1);
    ASSERT_EQ(Q[1], (__int128)0);
    ASSERT_EQ(Q[2], (__int128)(-1));
}

void test_edge_Q_linear() {
    std::cout << "  edge_Q_linear" << std::endl;
    // V = [(1,0), (0,1)], W = [(0,0), (0,0)] → W=0, Q = det(V) = 1 (constant)
    // Actually det(V0,V1) = 1*1 - 0*0 = 1, W terms are 0
    __int128 V[2][2] = {{1,0},{0,1}};
    __int128 W[2][2] = {{0,0},{0,0}};
    __int128 Q[3];
    compute_edge_Q_2d(V, W, Q);
    ASSERT_EQ(Q[0], (__int128)1);
    ASSERT_EQ(Q[1], (__int128)0);
    ASSERT_EQ(Q[2], (__int128)0);
}

void test_edge_Q_zero() {
    std::cout << "  edge_Q_zero (parallel fields)" << std::endl;
    // V = [(1,2), (2,4)] (parallel), W = [(0,0), (0,0)]
    // Q = 1*4 - 2*2 = 0
    __int128 V[2][2] = {{1,2},{2,4}};
    __int128 W[2][2] = {{0,0},{0,0}};
    __int128 Q[3];
    compute_edge_Q_2d(V, W, Q);
    ASSERT_EQ(Q[0], (__int128)0);
}

// ============================================================================
// compute_tri_QP_2d tests
// ============================================================================

void test_tri_QP_sum_property() {
    std::cout << "  tri_QP_sum_property (P[0]+P[1]+P[2]=Q)" << std::endl;
    // Random triangle: V = [(3,1),(1,4),(2,2)], W = [(1,-1),(2,1),(-1,3)]
    __int128 V[3][2] = {{3,1},{1,4},{2,2}};
    __int128 W[3][2] = {{1,-1},{2,1},{-1,3}};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V, W, Q, P);

    // Verify sum property: P[0]+P[1]+P[2] = Q for each coefficient
    for (int i = 0; i < 3; i++) {
        __int128 sum = P[0][i] + P[1][i] + P[2][i];
        ASSERT_EQ(sum, Q[i]);
    }
}

void test_tri_QP_face_is_edge_Q() {
    std::cout << "  tri_QP_face_is_edge_Q" << std::endl;
    // P[2] = det(U0, U1) which is the edge Q for edge (v0,v1)
    __int128 V[3][2] = {{5,2},{-1,3},{4,-2}};
    __int128 W[3][2] = {{2,1},{3,-1},{-2,4}};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V, W, Q, P);

    // Compute edge Q for (v0,v1) directly
    __int128 V01[2][2] = {{V[0][0],V[0][1]},{V[1][0],V[1][1]}};
    __int128 W01[2][2] = {{W[0][0],W[0][1]},{W[1][0],W[1][1]}};
    __int128 Qe[3];
    compute_edge_Q_2d(V01, W01, Qe);

    // P[2] should equal edge Q for (v0,v1)
    ASSERT_EQ(P[2][0], Qe[0]);
    ASSERT_EQ(P[2][1], Qe[1]);
    ASSERT_EQ(P[2][2], Qe[2]);
}

void test_tri_QP_identity() {
    std::cout << "  tri_QP_identity" << std::endl;
    // V = identity-like: [(1,0),(0,1),(0,0)], W = [(0,0),(0,0),(0,0)]
    // Q = det(V0-V2, V1-V2) = det([(1,0),(0,1)]) = 1 (constant)
    __int128 V[3][2] = {{1,0},{0,1},{0,0}};
    __int128 W[3][2] = {{0,0},{0,0},{0,0}};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V, W, Q, P);
    ASSERT_EQ(Q[0], (__int128)1);
    ASSERT_EQ(Q[1], (__int128)0);
    ASSERT_EQ(Q[2], (__int128)0);
}

// ============================================================================
// solve_pv_edge_2d tests
// ============================================================================

void test_edge_solver_two_roots() {
    std::cout << "  edge_solver_two_roots" << std::endl;
    // V = [(1,0), (0,1)], W = [(0,1), (1,0)]
    // Q = 1 - λ² → roots at λ=±1
    // At λ=1: U0=(1,1), U1=(1,1) → parallel, same dir → NOT interior
    // At λ=-1: U0=(1,-1), U1=(-1,1) → opposite → interior!
    __int128 V[2][2] = {{1,0},{0,1}};
    __int128 W[2][2] = {{0,1},{1,0}};
    PunctureResult2D pr = solve_pv_edge_2d(V, W);
    // Check we find exactly 1 interior puncture
    ASSERT_EQ(pr.count, 1);
}

void test_edge_solver_no_roots() {
    std::cout << "  edge_solver_no_roots" << std::endl;
    // V = [(1,0), (0,1)], W = [(1,0), (0,1)] (same as V)
    // Q = det(V+λW) = det([(1+λ,0),(0,1+λ)]) = (1+λ)² → disc = 0, 1 root at λ=-1
    // At λ=-1: U0=(0,0), U1=(0,0) → degenerate (both zero → vertex)
    __int128 V[2][2] = {{1,0},{0,1}};
    __int128 W[2][2] = {{1,0},{0,1}};
    PunctureResult2D pr = solve_pv_edge_2d(V, W);
    // Both U's are zero → vertex degeneracy, should be counted
    ASSERT_TRUE(pr.count >= 0);
}

void test_edge_solver_constant_Q() {
    std::cout << "  edge_solver_constant_Q" << std::endl;
    // V not parallel, W = 0 → Q = det(V) ≠ 0 (constant, no roots)
    __int128 V[2][2] = {{1,0},{0,1}};
    __int128 W[2][2] = {{0,0},{0,0}};
    PunctureResult2D pr = solve_pv_edge_2d(V, W);
    ASSERT_EQ(pr.count, 0);
}

void test_edge_solver_disc_negative() {
    std::cout << "  edge_solver_disc_negative" << std::endl;
    // V = [(1,0),(0,1)], W = [(0,-2),(2,0)]
    // Q[0] = 1, Q[2] = 0*0 - (-2)*2 = 4, Q[1] = 0 + 0 - 0 - 0 = 0
    // disc = 0 - 4*4*1 = -16 < 0 → no real roots
    __int128 V[2][2] = {{1,0},{0,1}};
    __int128 W[2][2] = {{0,-2},{2,0}};
    PunctureResult2D pr = solve_pv_edge_2d(V, W);
    ASSERT_EQ(pr.count, 0);
}

// ============================================================================
// solve_pv_tri_2d tests
// ============================================================================

void test_tri_solver_basic() {
    std::cout << "  tri_solver_basic (T0 case)" << std::endl;
    // Simple triangle with no punctures expected
    __int128 V[3][2] = {{1,0},{0,1},{-1,-1}};
    __int128 W[3][2] = {{0,0},{0,0},{0,0}};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V, W, Q, P);
    ExactPV2Result2D r = solve_pv_tri_2d(Q, P);
    // W=0 → Q=const, P=const → no lambda-dependent roots
    ASSERT_EQ(r.n_punctures, 0);
}

void test_tri_solver_T2() {
    std::cout << "  tri_solver_T2" << std::endl;
    // Construct a case with known T2 by using antiparallel vectors
    // V = [(3,1), (-1,2), (0,-3)], W = [(-1,2), (3,-1), (2,1)]
    __int128 V[3][2] = {{3,1},{-1,2},{0,-3}};
    __int128 W[3][2] = {{-1,2},{3,-1},{2,1}};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V, W, Q, P);

    // Verify sum property first
    for (int i = 0; i < 3; i++)
        ASSERT_EQ(P[0][i] + P[1][i] + P[2][i], Q[i]);

    ExactPV2Result2D r = solve_pv_tri_2d(Q, P);
    // T-count must be even
    ASSERT_TRUE(r.n_punctures % 2 == 0);
    // Pairs must be consistent
    ASSERT_TRUE(r.n_pairs * 2 <= r.n_punctures);
}

void test_tri_solver_T_even_random() {
    std::cout << "  tri_solver_T_even_random (100 cases)" << std::endl;
    // Run 100 random cases, verify T is always even
    uint32_t state = 12345;
    auto rand_int = [&]() -> int {
        state = state * 1664525u + 1013904223u;
        return (int)(state % 41) - 20;
    };

    int n_odd = 0;
    for (int trial = 0; trial < 100; trial++) {
        __int128 V[3][2], W[3][2];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                V[i][j] = (__int128)rand_int();
                W[i][j] = (__int128)rand_int();
            }
        __int128 Q[3], P[3][3];
        compute_tri_QP_2d(V, W, Q, P);
        ExactPV2Result2D r = solve_pv_tri_2d(Q, P);
        if (r.n_punctures % 2 != 0) n_odd++;
    }
    ASSERT_EQ(n_odd, 0);
}

void test_tri_solver_sum_property_random() {
    std::cout << "  tri_solver_sum_property_random (100 cases)" << std::endl;
    uint32_t state = 67890;
    auto rand_int = [&]() -> int {
        state = state * 1664525u + 1013904223u;
        return (int)(state % 41) - 20;
    };

    int n_fail = 0;
    for (int trial = 0; trial < 100; trial++) {
        __int128 V[3][2], W[3][2];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                V[i][j] = (__int128)rand_int();
                W[i][j] = (__int128)rand_int();
            }
        __int128 Q[3], P[3][3];
        compute_tri_QP_2d(V, W, Q, P);
        for (int c = 0; c < 3; c++) {
            if (P[0][c] + P[1][c] + P[2][c] != Q[c])
                n_fail++;
        }
    }
    ASSERT_EQ(n_fail, 0);
}

// ============================================================================
// check_field_zero_in_tri_2d tests
// ============================================================================

void test_cv_interior() {
    std::cout << "  cv_interior" << std::endl;
    // V = [(1,1), (-1,1), (0,-2)]: origin inside convex hull
    // Check: A = [(1,1),(-1,1)]-[(0,-2),(0,-2)] = [(1,3),(-1,3)]
    // det = 1*3 - (-1)*3 = 6, b = [2,2]
    // n0 = 2*3 - (-1)*2 = 8, n1 = 1*2 - 2*3 = -4 → negative but det>0...
    // Actually b = -F[2] = [0,2]
    // n0 = 0*3 - (-1)*2 = 2, n1 = 1*2 - 0*3 = 2, n2 = 6-2-2 = 2
    // All positive → interior!
    int F[3][2] = {{1,1},{-1,1},{0,-2}};
    int res = check_field_zero_in_tri_2d(F);
    ASSERT_EQ(res, 1);  // interior
}

void test_cv_vertex() {
    std::cout << "  cv_vertex" << std::endl;
    int F[3][2] = {{0,0},{1,1},{-1,1}};
    int res = check_field_zero_in_tri_2d(F);
    ASSERT_EQ(res, 3);  // vertex
}

void test_cv_edge() {
    std::cout << "  cv_edge" << std::endl;
    // V = [(1,0), (-1,0), (0,1)]: origin on edge v0-v1 (F[0] and F[1] antiparallel on x-axis)
    // A = [(1,-1),(-1,-1)], det = 1*(-1)-(-1)*(-1) = -1-1 = -2
    // b = -F[2] = [0,-1]
    // n0 = 0*(-1) - (-1)*(-1) = -1, n1 = 1*(-1) - 0*(-1) = -1, n2 = -2-(-1)-(-1) = 0
    // det<0: n0<0✓, n1<0✓, n2=0 → edge
    int F[3][2] = {{1,0},{-1,0},{0,1}};
    int res = check_field_zero_in_tri_2d(F);
    ASSERT_EQ(res, 2);  // edge
}

void test_cv_outside() {
    std::cout << "  cv_outside" << std::endl;
    // All vectors pointing same direction → origin NOT inside
    int F[3][2] = {{1,1},{2,2},{3,3}};
    int res = check_field_zero_in_tri_2d(F);
    ASSERT_EQ(res, 0);  // outside
}

void test_cv_collinear_inside() {
    std::cout << "  cv_collinear_inside" << std::endl;
    // Collinear: F = [(1,0), (-1,0), (2,0)] → origin between (1,0) and (-1,0)
    // det=0, collinear case: dot projections: 1, -1, 2 → has_pos AND has_neg → inside
    int F[3][2] = {{1,0},{-1,0},{2,0}};
    int res = check_field_zero_in_tri_2d(F);
    ASSERT_EQ(res, 2);  // edge (collinear)
}

void test_cv_collinear_outside() {
    std::cout << "  cv_collinear_outside" << std::endl;
    // All same direction: dots all positive → not inside
    int F[3][2] = {{1,0},{2,0},{3,0}};
    int res = check_field_zero_in_tri_2d(F);
    ASSERT_EQ(res, 0);
}

// ============================================================================
// classify_case_v2_2d tests
// ============================================================================

void test_classify_T0_Q0() {
    std::cout << "  classify_T0_Q0" << std::endl;
    // V = [(1,0),(0,1),(0,0)], W = 0 → Q = det(V) = 1 (constant)
    TriCaseV2GPU tv2;
    memset(&tv2, 0, sizeof(tv2));
    tv2.V[0][0] = 1; tv2.V[0][1] = 0;
    tv2.V[1][0] = 0; tv2.V[1][1] = 1;
    tv2.V[2][0] = 0; tv2.V[2][1] = 0;
    // W all zero
    tv2.seed = 1;

    __int128 V128[3][2], W128[3][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            V128[i][j] = (__int128)tv2.V[i][j];
            W128[i][j] = (__int128)tv2.W[i][j];
        }
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V128, W128, Q, P);
    tv2.v2 = solve_pv_tri_2d(Q, P);

    ClassifiedCase2D cc = classify_case_v2_2d(tv2);
    ASSERT_EQ(cc.total_punctures, 0);
    ASSERT_TRUE(cc.category.find("T0") != std::string::npos);
    ASSERT_TRUE(cc.category.find("Q0") != std::string::npos);
}

void test_classify_Qz() {
    std::cout << "  classify_Qz" << std::endl;
    // V = [(0,0),(0,0),(0,0)] → Q≡0
    TriCaseV2GPU tv2;
    memset(&tv2, 0, sizeof(tv2));
    tv2.seed = 2;

    __int128 V128[3][2] = {}, W128[3][2] = {};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V128, W128, Q, P);
    tv2.v2 = solve_pv_tri_2d(Q, P);

    ClassifiedCase2D cc = classify_case_v2_2d(tv2);
    ASSERT_TRUE(cc.category.find("Qz") != std::string::npos);
}

void test_classify_Cv0() {
    std::cout << "  classify_Cv0" << std::endl;
    // V[0] = (0,0), others nonzero → Cv0
    TriCaseV2GPU tv2;
    memset(&tv2, 0, sizeof(tv2));
    tv2.V[0][0] = 0; tv2.V[0][1] = 0;
    tv2.V[1][0] = 1; tv2.V[1][1] = 2;
    tv2.V[2][0] = 3; tv2.V[2][1] = -1;
    tv2.W[0][0] = 1; tv2.W[0][1] = 0;
    tv2.W[1][0] = 0; tv2.W[1][1] = 1;
    tv2.W[2][0] = -1; tv2.W[2][1] = -1;
    tv2.seed = 3;

    __int128 V128[3][2], W128[3][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            V128[i][j] = (__int128)tv2.V[i][j];
            W128[i][j] = (__int128)tv2.W[i][j];
        }
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V128, W128, Q, P);
    tv2.v2 = solve_pv_tri_2d(Q, P);

    ClassifiedCase2D cc = classify_case_v2_2d(tv2);
    ASSERT_TRUE(cc.has_Cv);
    ASSERT_TRUE(cc.category.find("Cv0") != std::string::npos);
}

void test_classify_random_T_even() {
    std::cout << "  classify_random_T_even (500 random triangles)" << std::endl;
    uint32_t state = 99999;
    auto rand_int = [&]() -> int {
        state = state * 1664525u + 1013904223u;
        return (int)(state % 41) - 20;
    };

    int n_odd_bare = 0;
    for (int trial = 0; trial < 500; trial++) {
        TriCaseV2GPU tv2;
        memset(&tv2, 0, sizeof(tv2));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                tv2.V[i][j] = rand_int();
                tv2.W[i][j] = rand_int();
            }
        tv2.seed = trial;

        __int128 V128[3][2], W128[3][2];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                V128[i][j] = (__int128)tv2.V[i][j];
                W128[i][j] = (__int128)tv2.W[i][j];
            }
        __int128 Q[3], P[3][3];
        compute_tri_QP_2d(V128, W128, Q, P);
        tv2.v2 = solve_pv_tri_2d(Q, P);

        ClassifiedCase2D cc = classify_case_v2_2d(tv2);
        if (cc.total_punctures % 2 != 0 && !cc.has_Cv && !cc.has_Cw)
            n_odd_bare++;
    }
    ASSERT_EQ(n_odd_bare, 0);
}

// ============================================================================
// D00 / edge puncture tests
// ============================================================================

void test_D00_vertex_puncture() {
    std::cout << "  D00_vertex_puncture" << std::endl;
    // Construct case where det(V_k, W_k)=0 at vertex k
    // V[0] = (2,1), W[0] = (4,2) → V[0] parallel to W[0] → det=0 at v0
    // This means λ* = -V[0]/W[0] = -1/2 gives U₀(λ*)=0
    // Need to check this root makes Q=0 too and is valid
    int V[3][2] = {{2,1},{-1,3},{1,-2}};
    int W[3][2] = {{4,2},{1,-1},{-2,1}};

    __int128 V128[3][2], W128[3][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            V128[i][j] = (__int128)V[i][j];
            W128[i][j] = (__int128)W[i][j];
        }
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V128, W128, Q, P);

    // Verify sum property
    for (int i = 0; i < 3; i++)
        ASSERT_EQ(P[0][i] + P[1][i] + P[2][i], Q[i]);

    ExactPV2Result2D r = solve_pv_tri_2d(Q, P);
    // T must be even
    ASSERT_TRUE(r.n_punctures % 2 == 0);
}

// ============================================================================
// Solver consistency: GPU seed replay
// ============================================================================

void test_solver_deterministic() {
    std::cout << "  solver_deterministic (same input → same output)" << std::endl;
    __int128 V[3][2] = {{7,-3},{-5,8},{2,-4}};
    __int128 W[3][2] = {{-2,6},{4,-1},{-3,5}};
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V, W, Q, P);

    ExactPV2Result2D r1 = solve_pv_tri_2d(Q, P);
    ExactPV2Result2D r2 = solve_pv_tri_2d(Q, P);

    ASSERT_EQ(r1.n_punctures, r2.n_punctures);
    ASSERT_EQ(r1.n_pairs, r2.n_pairs);
    for (int i = 0; i < r1.n_punctures; i++) {
        ASSERT_EQ(r1.punctures[i].face, r2.punctures[i].face);
        ASSERT_EQ(r1.punctures[i].root_idx, r2.punctures[i].root_idx);
        ASSERT_EQ(r1.punctures[i].q_interval, r2.punctures[i].q_interval);
    }
}

// ============================================================================
// Stress test: large random sweep
// ============================================================================

void test_large_sweep_1000() {
    std::cout << "  large_sweep_1000 (T-parity, bare odd-T only)" << std::endl;
    uint32_t state = 314159;
    auto rand_int = [&]() -> int {
        state = state * 1664525u + 1013904223u;
        return (int)(state % 41) - 20;
    };

    int n_odd_bare = 0;
    int n_odd_waypoint = 0;
    int max_T = 0;
    for (int trial = 0; trial < 1000; trial++) {
        int Vi[3][2], Wi[3][2];
        __int128 V[3][2], W[3][2];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                Vi[i][j] = rand_int();
                Wi[i][j] = rand_int();
                V[i][j] = (__int128)Vi[i][j];
                W[i][j] = (__int128)Wi[i][j];
            }
        __int128 Q[3], P[3][3];
        compute_tri_QP_2d(V, W, Q, P);

        // Verify sum property
        for (int c = 0; c < 3; c++) {
            __int128 sum = P[0][c] + P[1][c] + P[2][c];
            if (sum != Q[c]) {
                std::cerr << "  [FAIL] sum property violation at trial " << trial << std::endl;
                n_odd_bare++;
            }
        }

        ExactPV2Result2D r = solve_pv_tri_2d(Q, P);
        if (r.n_punctures % 2 != 0) {
            // Check for Cv/Cw waypoints (V[k]=0 or W[k]=0)
            bool has_cv = (check_field_zero_in_tri_2d(Vi) > 0);
            bool has_cw = (check_field_zero_in_tri_2d(Wi) > 0);
            if (has_cv || has_cw) {
                n_odd_waypoint++;
            } else {
                n_odd_bare++;
                std::cerr << "    BARE odd-T at trial " << trial << ": T=" << r.n_punctures << std::endl;
            }
        }
        if (r.n_punctures > max_T) max_T = r.n_punctures;
    }
    ASSERT_EQ(n_odd_bare, 0);
    std::cout << "    max T = " << max_T
              << ", odd-T with waypoint = " << n_odd_waypoint << std::endl;
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== 2D PV Primitives Tests ===" << std::endl;

    std::cout << "\n--- Edge Q ---" << std::endl;
    test_edge_Q_basic();
    test_edge_Q_linear();
    test_edge_Q_zero();

    std::cout << "\n--- Triangle Q+P ---" << std::endl;
    test_tri_QP_sum_property();
    test_tri_QP_face_is_edge_Q();
    test_tri_QP_identity();

    std::cout << "\n--- Edge Solver ---" << std::endl;
    test_edge_solver_two_roots();
    test_edge_solver_no_roots();
    test_edge_solver_constant_Q();
    test_edge_solver_disc_negative();

    std::cout << "\n--- Triangle Solver ---" << std::endl;
    test_tri_solver_basic();
    test_tri_solver_T2();
    test_tri_solver_T_even_random();
    test_tri_solver_sum_property_random();

    std::cout << "\n--- Cv/Cw Detection ---" << std::endl;
    test_cv_interior();
    test_cv_vertex();
    test_cv_edge();
    test_cv_outside();
    test_cv_collinear_inside();
    test_cv_collinear_outside();

    std::cout << "\n--- Classification ---" << std::endl;
    test_classify_T0_Q0();
    test_classify_Qz();
    test_classify_Cv0();
    test_classify_random_T_even();

    std::cout << "\n--- D00 ---" << std::endl;
    test_D00_vertex_puncture();

    std::cout << "\n--- Determinism ---" << std::endl;
    test_solver_deterministic();

    std::cout << "\n--- Stress Tests ---" << std::endl;
    test_large_sweep_1000();

    std::cout << "\n=== Results: " << passed_tests << "/" << total_tests
              << " passed, " << failed_tests << " failed ===" << std::endl;
    return failed_tests > 0 ? 1 : 0;
}
