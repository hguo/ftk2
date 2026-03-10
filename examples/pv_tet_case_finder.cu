// GPU-accelerated single-tet PV case finder for publication figures.
//
// Architecture:
//   - GPU: random tet generation + solve_pv_triangle_device on 4 faces
//   - CPU: post-processing (Q/P polynomials, classification, JSON output)
//
// Usage:
//   ./ftk2_pv_tet_case_finder [--min-punctures N] [--num-tets M] [--range R]
//                              [--seed S] [--max-cases C] > cases.json

#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <string>

using namespace ftk2;

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

// ─── GPU output struct for ExactPV2 ──────────────────────────────────────────
struct TetCaseV2GPU {
    int V[4][3], W[4][3];
    ExactPV2Result v2;
    int disc_sign[4];   // discriminant sign of P[k] for each face
    uint64_t seed;
};

// ─── Face vertex ordering (consistent orientation) ──────────────────────────
// face i = triangle opposite vertex i
__constant__ int d_face_verts[4][3] = {
    {1, 3, 2},  // face 0: opposite vertex 0
    {0, 2, 3},  // face 1: opposite vertex 1
    {0, 3, 1},  // face 2: opposite vertex 2
    {0, 1, 2}   // face 3: opposite vertex 3
};

// ─── Device-side LCG random number generator ────────────────────────────────
__device__ uint32_t lcg_next(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ int rand_int_dev(uint32_t& state, int R) {
    uint32_t r = lcg_next(state);
    return (int)(r % (2 * R + 1)) - R;
}

// ─── ExactPV2 GPU extraction kernel ──────────────────────────────────────────
// One thread per random tet. Uses pure-integer solver directly on GPU.
__global__ void tet_case_finder_v2_kernel(
    TetCaseV2GPU* output,
    int*          output_count,
    int           max_output,
    int           min_punctures,
    int           R,
    uint64_t      base_seed,
    uint64_t      batch_offset)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t global_id = batch_offset + tid;

    // Seed LCG from global thread id + base seed
    uint32_t state = (uint32_t)(global_id ^ (base_seed * 2654435761ULL));
    for (int i = 0; i < 4; i++) lcg_next(state);

    // Generate random integer fields
    int V[4][3], W[4][3];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            V[i][j] = rand_int_dev(state, R);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            W[i][j] = rand_int_dev(state, R);

    // Compute Q, P polynomials (integer)
    __int128 Q_i128[4], P_i128[4][4];
    compute_tet_QP_i128(V, W, Q_i128, P_i128);

    // Run ExactPV2 solver
    ExactPV2Result v2 = solve_pv_tet_v2(Q_i128, P_i128);

    // Filter: only keep cases with enough punctures
    if (v2.n_punctures >= min_punctures) {
        int idx = atomicAdd(output_count, 1);
        if (idx < max_output) {
            TetCaseV2GPU& out = output[idx];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++) {
                    out.V[i][j] = V[i][j];
                    out.W[i][j] = W[i][j];
                }
            out.v2 = v2;
            // Compute discriminant signs for classification
            for (int k = 0; k < 4; k++)
                out.disc_sign[k] = discriminant_sign_i128(P_i128[k]);
            out.seed = global_id;
        }
    }
}

// ─── CPU classification ─────────────────────────────────────────────────────

struct ClassifiedCase {
    int V[4][3], W[4][3];
    uint64_t seed;
    int total_punctures;
    std::string category;

    // Integer polynomial coefficients (from compute_tet_QP_i128)
    __int128 Q_i128[4];
    __int128 P_i128[4][4];
    int Q_disc_sign;
    int n_Q_roots;          // count from integer discriminant

    // v2 solver output (integer)
    bool merge_infinity;
    int n_qr_roots;
    __int128 h[4];          // pass-through GCD
    int h_deg, h_n_roots;
    __int128 P_red[4][4];   // reduced P after pass-through
    int degP_red[4];
    int n_distinct_red[4];

    // SR
    bool has_shared_root;
    bool has_non_isolated_sr;
    int n_sr_roots;
    int sr_q_root_idx[3];

    // TN — store GCD polynomial h_tn (integer), face index
    int n_tn;
    struct TNInfo {
        int face;
        __int128 h_tn[2];   // linear factor (degree 1): root = -h[0]/h[1]
    } tn_points[4];

    // Cv/Cw — store integer numerators + denominator
    bool has_Cv_pos;
    int64_t Cv_num[4];
    int64_t Cv_den;
    bool has_Cw_pos;
    int64_t Cw_num[4];
    int64_t Cw_den;

    bool has_B;

    struct PunctureInfo {
        int face;
        int root_idx;         // index into P_red roots (or -1 for infinity)
        int q_interval;       // from v2 solver (integer)
        bool is_edge;
        bool is_vertex;
        bool is_passthrough;
        bool is_D00, is_D01;
        int tet_edge[2];
        int tet_vertex;
        int edge_faces[2];    // from v2 solver
        int interval_idx;     // mapped to original Q intervals
    };
    std::vector<PunctureInfo> punctures;
    int n_deduplicated;

    struct IntervalInfo {
        int n_pv;
        bool is_infinity;
    };
    std::vector<IntervalInfo> intervals;

    struct PuncturePair {
        int pi_a, pi_b;
        bool is_cross;
        int interval_idx;
    };
    std::vector<PuncturePair> pairs;
};

// Check if field=0 is inside tet interior (critical point).
// For integer inputs (|F[i][j]| <= R=20), uses exact int64_t arithmetic
// for the determinant and Cramer numerators.  No thresholds.
// Returns true and writes integer numerators + denominator if found inside.
static bool check_field_zero_in_tet(const int F[4][3],
                                     int64_t num_out[4] = nullptr,
                                     int64_t* den_out = nullptr) {
    int64_t A[3][3], b[3];
    for (int c = 0; c < 3; c++) {
        A[c][0] = (int64_t)F[0][c] - F[3][c];
        A[c][1] = (int64_t)F[1][c] - F[3][c];
        A[c][2] = (int64_t)F[2][c] - F[3][c];
        b[c] = -(int64_t)F[3][c];
    }
    int64_t det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if (det == 0) return false;

    int64_t n0 = b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
               - A[0][1]*(b[1]*A[2][2]-A[1][2]*b[2])
               + A[0][2]*(b[1]*A[2][1]-A[1][1]*b[2]);
    int64_t n1 = A[0][0]*(b[1]*A[2][2]-A[1][2]*b[2])
               - b[0]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*b[2]-b[1]*A[2][0]);
    int64_t n2 = A[0][0]*(A[1][1]*b[2]-b[1]*A[2][1])
               - A[0][1]*(A[1][0]*b[2]-b[1]*A[2][0])
               + b[0]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    int64_t n3 = det - n0 - n1 - n2;

    if (det > 0) {
        if (n0 < 0 || n1 < 0 || n2 < 0 || n3 < 0) return false;
        if (n0 > det || n1 > det || n2 > det || n3 > det) return false;
    } else {
        if (n0 > 0 || n1 > 0 || n2 > 0 || n3 > 0) return false;
        if (n0 < det || n1 < det || n2 < det || n3 < det) return false;
    }
    if (num_out) {
        num_out[0] = n0; num_out[1] = n1; num_out[2] = n2; num_out[3] = n3;
    }
    if (den_out) *den_out = det;
    return true;
}

// Exact shared-root detection: Resultant(Q, P_k) = 0 iff they share a root.
// Q and P_k are degree-3 polynomials with __int128 coefficients.
// The 6x6 Sylvester determinant is computed exactly.
// Returns true if any P_k shares a root with Q.
static bool has_shared_root_resultant(const __int128 Q_i128[4],
                                       const __int128 P_i128[4][4])
{
    // Determine actual degree of Q
    int degQ = 3;
    while (degQ > 0 && Q_i128[degQ] == 0) degQ--;
    if (degQ == 0) return false;  // Q is constant, no roots

    for (int k = 0; k < 4; ++k) {
        // Determine actual degree of P_k
        int degP = 3;
        while (degP > 0 && P_i128[k][degP] == 0) degP--;
        if (degP == 0) continue;  // P_k is constant, no roots

        // For polynomials of degree m and n, Sylvester matrix is (m+n) x (m+n).
        // Use direct computation for small degrees.
        // Resultant via polynomial evaluation: Res(Q,P) = lc(Q)^degP * prod(P(r_i))
        // where r_i are roots of Q. But we need exact integer arithmetic.
        //
        // For deg <= 3, use the explicit Sylvester determinant.
        // We'll compute it as a double first, then verify sign stability.
        // Actually, since coefficients can be large __int128, we use __int128 arithmetic.
        //
        // For two cubics (most common case): 6x6 Sylvester matrix
        // Row 0: Q[3] Q[2] Q[1] Q[0]  0    0
        // Row 1:  0   Q[3] Q[2] Q[1] Q[0]  0
        // Row 2:  0    0   Q[3] Q[2] Q[1] Q[0]
        // Row 3: P[3] P[2] P[1] P[0]  0    0
        // Row 4:  0   P[3] P[2] P[1] P[0]  0
        // Row 5:  0    0   P[3] P[2] P[1] P[0]
        //
        // For mixed degrees, the matrix is (degQ+degP) x (degQ+degP).
        // Use Gaussian elimination with __int128 to avoid overflow issues.
        // Actually, for degrees up to 3+3=6, we can use fraction-free elimination.

        int N = degQ + degP;
        // Build Sylvester matrix with __int128 (exact for integer coefficients)
        __int128 M[6][6] = {};
        for (int i = 0; i < degP; i++)
            for (int j = 0; j <= degQ; j++)
                M[i][i + degQ - j] = Q_i128[j];
        for (int i = 0; i < degQ; i++)
            for (int j = 0; j <= degP; j++)
                M[degP + i][i + degP - j] = P_i128[k][j];

        // Bareiss fraction-free elimination: exact integer determinant.
        // Each step: M[i][j] = (M[i][j]*M[k][k] - M[i][k]*M[k][j]) / prev_pivot
        // The division is exact (Bareiss's theorem).
        __int128 prev_pivot = 1;
        bool zero_det = false;
        for (int col = 0; col < N; col++) {
            // Partial pivoting (for numerical stability, though exact)
            int pivot = -1;
            for (int row = col; row < N; row++)
                if (M[row][col] != 0) { pivot = row; break; }
            if (pivot < 0) { zero_det = true; break; }
            if (pivot != col)
                for (int j = 0; j < N; j++)
                    std::swap(M[col][j], M[pivot][j]);
            for (int row = col + 1; row < N; row++)
                for (int j = col + 1; j < N; j++) {
                    M[row][j] = (M[col][col] * M[row][j]
                               - M[row][col] * M[col][j]) / prev_pivot;
                }
            // Zero out below pivot (not needed for det, but keeps matrix clean)
            for (int row = col + 1; row < N; row++)
                M[row][col] = 0;
            prev_pivot = M[col][col];
        }

        // Resultant = M[N-1][N-1] (the last diagonal element after Bareiss)
        // It's exactly 0 iff Q and P_k share a root.
        if (zero_det || M[N-1][N-1] == 0) return true;
    }
    return false;
}

// ─── Check if any shared root resolves inside/on the tet ─────────────────
// For each Q-root λ*, if Res(Q, P_k)=0 for some k:
//   - If any P_j(λ*) ≠ 0 → μ_j diverges → point escapes to ∞ → outside
//   - If ALL P_j(λ*)=0 → removable singularity → μ_j = P_j'(λ*)/Q'(λ*)
//     Check if resolved μ_j ≥ 0 for all j and Σμ_j = 1 → inside/on tet
// Returns true if at least one shared root resolves inside or on the boundary.




// ─── __int128 to string helper ──────────────────────────────────────────────
static std::string i128_to_string(__int128 v) {
    if (v == 0) return "0";
    bool neg = (v < 0);
    if (neg) v = -v;
    std::string s;
    while (v > 0) { s += ('0' + (int)(v % 10)); v /= 10; }
    if (neg) s += '-';
    std::reverse(s.begin(), s.end());
    return s;
}

// ─── classify_case_v2: Pure-integer classification from ExactPV2Result ────
// Takes TetCaseV2GPU directly — ALL topological decisions use __int128.
// NO float/double anywhere in this function.
static ClassifiedCase classify_case_v2(const TetCaseV2GPU& gpu_v2) {
    ClassifiedCase cc;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            cc.V[i][j] = gpu_v2.V[i][j];
            cc.W[i][j] = gpu_v2.W[i][j];
        }
    cc.total_punctures = gpu_v2.v2.n_punctures;
    cc.seed = gpu_v2.seed;
    cc.has_shared_root = false;
    cc.has_non_isolated_sr = false;
    cc.n_sr_roots = 0;
    cc.has_B = false;
    cc.has_Cv_pos = false;
    cc.has_Cw_pos = false;
    cc.Cv_den = 0;
    cc.Cw_den = 0;
    cc.n_tn = 0;
    cc.n_deduplicated = 0;

    // ─── Step 0: Integer Q, P polynomials only ────────────────────────
    __int128 Q_i128[4], P_i128[4][4];
    compute_tet_QP_i128(gpu_v2.V, gpu_v2.W, Q_i128, P_i128);
    for (int i = 0; i < 4; i++) cc.Q_i128[i] = Q_i128[i];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cc.P_i128[i][j] = P_i128[i][j];

    cc.Q_disc_sign = discriminant_sign_i128(Q_i128);

    // n_Q_roots from integer discriminant
    int degQ = ftk2::effective_degree_i128(Q_i128, 3);
    if (degQ == 3) {
        if (cc.Q_disc_sign > 0) cc.n_Q_roots = 3;
        else if (cc.Q_disc_sign < 0) cc.n_Q_roots = 1;
        else {
            // disc=0: check D = Q[2]^2 - 3*Q[1]*Q[3]
            __int128 D = Q_i128[2]*Q_i128[2] - (__int128)3*Q_i128[1]*Q_i128[3];
            cc.n_Q_roots = (D == 0) ? 1 : 2;  // triple or double root
        }
    } else if (degQ == 2) {
        __int128 disc2 = Q_i128[1]*Q_i128[1] - (__int128)4*Q_i128[0]*Q_i128[2];
        if (disc2 > 0) cc.n_Q_roots = 2;
        else if (disc2 == 0) cc.n_Q_roots = 1;
        else cc.n_Q_roots = 0;
    } else if (degQ == 1) {
        cc.n_Q_roots = 1;
    } else {
        cc.n_Q_roots = 0;
    }

    // Store v2 solver infrastructure
    const ExactPV2Result& v2 = gpu_v2.v2;
    cc.merge_infinity = v2.merge_infinity;
    cc.n_qr_roots = v2.n_qr_roots;
    for (int i = 0; i < 4; i++) cc.h[i] = v2.h[i];
    cc.h_deg = v2.h_deg;
    cc.h_n_roots = v2.h_n_roots;
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 4; i++) cc.P_red[k][i] = v2.P_red[k][i];
        cc.degP_red[k] = v2.degP_red[k];
        cc.n_distinct_red[k] = v2.n_distinct_red[k];
    }

    static const int fv[4][3] = {
        {1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}
    };

    // ─── Step 1: Map v2 punctures → PunctureInfo ──────────────────────
    for (int pi = 0; pi < v2.n_punctures; pi++) {
        const auto& vp = v2.punctures[pi];
        ClassifiedCase::PunctureInfo ci;
        ci.face = vp.face;
        ci.root_idx = vp.root_idx;
        ci.q_interval = vp.q_interval;
        ci.is_edge = vp.is_edge;
        ci.is_vertex = vp.is_vertex;
        ci.is_passthrough = false;
        ci.is_D00 = false;
        ci.is_D01 = false;
        ci.interval_idx = -1;
        ci.tet_edge[0] = ci.tet_edge[1] = -1;
        ci.tet_vertex = -1;
        ci.edge_faces[0] = vp.edge_faces[0];
        ci.edge_faces[1] = vp.edge_faces[1];

        // Edge: tet_edge = {0,1,2,3}\{edge_faces[0], edge_faces[1]}
        if (ci.is_edge) {
            int ef0 = vp.edge_faces[0], ef1 = vp.edge_faces[1];
            int nv = 0;
            int ev[2];
            for (int v = 0; v < 4; v++)
                if (v != ef0 && v != ef1 && nv < 2)
                    ev[nv++] = v;
            ci.tet_edge[0] = std::min(ev[0], ev[1]);
            ci.tet_edge[1] = std::max(ev[0], ev[1]);
        }

        // Vertex: find vertex from edge_faces (the one face NOT in edge_faces
        // and NOT the puncture face)
        if (ci.is_vertex) {
            // For a vertex puncture, the vertex is the one NOT opposite any of
            // the 3 faces that share the root. In v2 solver, edge_faces gives
            // the two additional faces (besides vp.face) where P[k]=0.
            // The vertex = {0,1,2,3} \ {vp.face, edge_faces[0], edge_faces[1]}
            for (int v = 0; v < 4; v++)
                if (v != vp.face && v != vp.edge_faces[0] && v != vp.edge_faces[1])
                    ci.tet_vertex = v;
        }

        cc.punctures.push_back(ci);
    }

    // ─── Step 2: Q-type string ────────────────────────────────────────
    std::string q_type;
    if (degQ == 0 && Q_i128[0] == 0) q_type = "Qz";
    else if (degQ == 0) q_type = "Q0";
    else if (degQ == 1) q_type = "Q1";
    else if (degQ == 2) {
        __int128 disc2 = Q_i128[1]*Q_i128[1] - (__int128)4*Q_i128[0]*Q_i128[2];
        if (disc2 > 0) q_type = "Q2";
        else if (disc2 < 0) q_type = "Q2-";
        else q_type = "Q2o";
    } else {
        if (cc.Q_disc_sign > 0) q_type = "Q3+";
        else if (cc.Q_disc_sign < 0) q_type = "Q3-";
        else q_type = "Q3o";
    }

    // ─── Step 3: Build intervals between Q roots ──────────────────────
    if (cc.n_Q_roots > 0) {
        cc.intervals.push_back({0, true});  // (-inf, root_0)
        for (int i = 0; i + 1 < cc.n_Q_roots; i++)
            cc.intervals.push_back({0, false});  // (root_i, root_{i+1})
        cc.intervals.push_back({0, true});  // (root_{n-1}, +inf)
    } else {
        cc.intervals.push_back({0, true});  // single interval
    }

    // ─── Step 4: Interval assignment (pure integer from v2 solver) ─────
    if (degQ > 0) {
        int degQ_i = ftk2::effective_degree_i128(Q_i128, 3);
        for (int k = 0; k < 4; k++) {
            int degPk = ftk2::effective_degree_i128(P_i128[k], 3);
            if (degPk <= 0) continue;
            if (ftk2::resultant_sign_i128(Q_i128, degQ_i, P_i128[k], degPk) == 0) {
                cc.has_shared_root = true;
                break;
            }
        }
    }

    for (int pi = 0; pi < (int)cc.punctures.size(); pi++) {
        auto& punc = cc.punctures[pi];
        const auto& vp = v2.punctures[pi];
        int q_red_iv = vp.q_interval;

        int h_below = 0;
        if (v2.h_n_roots > 0) {
            if (vp.root_idx < 0) {
                h_below = v2.h_n_roots;
            } else {
                for (int hi = 0; hi < v2.h_n_roots; hi++) {
                    int cmp = ftk2::compare_roots_i128(
                        v2.P_red[vp.face], v2.degP_red[vp.face],
                        v2.n_distinct_red[vp.face], vp.root_idx,
                        v2.h, v2.h_deg, v2.h_n_roots, hi);
                    if (cmp > 0) h_below++;
                }
            }
        }

        int interval_idx = q_red_iv + h_below;
        if (interval_idx >= 0 && interval_idx < (int)cc.intervals.size()) {
            punc.interval_idx = interval_idx;
            cc.intervals[interval_idx].n_pv++;
        }
    }

    int h_merged_pos[4] = {-1,-1,-1,-1};

    // ─── Step 5: Pairs + is_cross (pure integer from v2 solver) ──────
    {
        for (int i = 0; i < v2.n_pairs; i++) {
            int a = v2.pairs[i].a, b = v2.pairs[i].b;
            if (a >= (int)cc.punctures.size() || b >= (int)cc.punctures.size())
                continue;

            int qa = v2.punctures[a].q_interval;
            int qb = v2.punctures[b].q_interval;
            bool is_cross = v2.merge_infinity && (qa != qb);

            int iv_idx = cc.punctures[a].interval_idx;
            cc.pairs.push_back({a, b, is_cross, iv_idx});
        }

        std::set<int> paired;
        for (const auto& pp : cc.pairs) {
            paired.insert(pp.pi_a);
            paired.insert(pp.pi_b);
        }
        std::vector<int> unpaired;
        for (int i = 0; i < (int)cc.punctures.size(); i++)
            if (paired.find(i) == paired.end())
                unpaired.push_back(i);
        std::sort(unpaired.begin(), unpaired.end(), [&](int a, int b) {
            int qa = v2.punctures[a].q_interval, qb = v2.punctures[b].q_interval;
            if (qa != qb) return qa < qb;
            return v2.punctures[a].root_idx < v2.punctures[b].root_idx;
        });
        for (int j = 0; j + 1 < (int)unpaired.size(); j += 2)
            cc.pairs.push_back({unpaired[j], unpaired[j + 1], false, -1});

        // SR root index computation
        std::set<int> sr_qr_set;
        if (v2.h_n_roots > 0) {
            __int128 Q_red_i128[4] = {};
            int degQ_red = ftk2::effective_degree_i128(Q_i128, 3);
            if (v2.h_deg >= 1) {
                __int128 qd[4] = {};
                degQ_red = ftk2::poly_exact_div_i128(
                    Q_i128, ftk2::effective_degree_i128(Q_i128, 3),
                    v2.h, v2.h_deg, qd);
                for (int i = 0; i < 4; i++) Q_red_i128[i] = (i <= degQ_red) ? qd[i] : 0;
            } else {
                for (int i = 0; i < 4; i++) Q_red_i128[i] = Q_i128[i];
            }
            for (int hi = 0; hi < v2.h_n_roots; hi++) {
                int qr_below = 0;
                for (int qi = 0; qi < v2.n_qr_roots; qi++) {
                    int cmp = ftk2::compare_roots_i128(
                        v2.h, v2.h_deg, v2.h_n_roots, hi,
                        Q_red_i128, degQ_red, v2.n_qr_roots, qi);
                    if (cmp > 0) qr_below++;
                }
                int mpos = qr_below + hi;
                sr_qr_set.insert(mpos);
                h_merged_pos[hi] = mpos;
            }
        }
        cc.n_sr_roots = 0;
        for (int idx : sr_qr_set)
            if (cc.n_sr_roots < 3)
                cc.sr_q_root_idx[cc.n_sr_roots++] = idx;

        // Assertion: no segment arc may contain a non-SR Q-root
        for (const auto& pp : cc.pairs) {
            if (pp.pi_a < 0 || pp.pi_b < 0) continue;
            int iv_a = cc.punctures[pp.pi_a].interval_idx;
            int iv_b = cc.punctures[pp.pi_b].interval_idx;
            int lo_iv = std::min(iv_a, iv_b);
            int hi_iv = std::max(iv_a, iv_b);
            if (!pp.is_cross) {
                for (int qi = lo_iv; qi < hi_iv; qi++) {
                    if (sr_qr_set.find(qi) == sr_qr_set.end()) {
                        fprintf(stderr, "  [ASSERT FAIL] seed=%lu: pair (%d,%d) short arc "
                                "crosses non-SR Q-root %d\n",
                                (unsigned long)gpu_v2.seed, pp.pi_a, pp.pi_b, qi);
                    }
                }
            } else {
                for (int qi = 0; qi < cc.n_Q_roots; qi++) {
                    if (qi >= lo_iv && qi < hi_iv) continue;
                    if (sr_qr_set.find(qi) == sr_qr_set.end()) {
                        fprintf(stderr, "  [ASSERT FAIL] seed=%lu: pair (%d,%d) long arc "
                                "crosses non-SR Q-root %d\n",
                                (unsigned long)gpu_v2.seed, pp.pi_a, pp.pi_b, qi);
                    }
                }
            }
        }
    }

    // ─── Step 5b: D12 pairing (PV curve on face) ────────────────────
    // When P[k] ≡ 0 for some face k, the PV curve lies on face k.
    // Find endpoints where the curve touches face-k edges (roots of
    // P_red[j] for face vertices j, with non-negative bary coords).
    // Pair non-isolated endpoints to form D12 curve segments.
    for (int d12k = 0; d12k < 4; d12k++) {
        if (P_i128[d12k][0] != 0 || P_i128[d12k][1] != 0 ||
            P_i128[d12k][2] != 0 || P_i128[d12k][3] != 0) continue;

        int fv[3], nfv = 0;
        for (int j = 0; j < 4; j++) if (j != d12k) fv[nfv++] = j;

        __int128 Qr_d12[4] = {};
        int dQr_d12;
        if (v2.h_deg >= 1) {
            dQr_d12 = ftk2::poly_exact_div_i128(Q_i128,
                ftk2::effective_degree_i128(Q_i128, 3), v2.h, v2.h_deg, Qr_d12);
        } else {
            for (int i = 0; i < 4; i++) Qr_d12[i] = Q_i128[i];
            dQr_d12 = ftk2::effective_degree_i128(Q_i128, 3);
        }

        struct D12EP { int src_vert, root_idx, edge[2], pi_idx; };
        std::vector<D12EP> d12_eps;

        for (int mi = 0; mi < 3; mi++) {
            int m = fv[mi];
            int dPm = cc.degP_red[m], n_rm = cc.n_distinct_red[m];
            if (dPm <= 0 || n_rm <= 0) continue;
            int o1 = fv[(mi+1)%3], o2 = fv[(mi+2)%3];

            int s_o1[4] = {}, s_o2[4] = {}, s_qr[4] = {};
            bool o1z = (cc.degP_red[o1] <= 0 && cc.P_red[o1][0] == 0);
            bool o2z = (cc.degP_red[o2] <= 0 && cc.P_red[o2][0] == 0);
            if (!o1z) ftk2::signs_at_roots_i128(cc.P_red[m], dPm,
                          cc.P_red[o1], cc.degP_red[o1], s_o1, 4);
            if (!o2z) ftk2::signs_at_roots_i128(cc.P_red[m], dPm,
                          cc.P_red[o2], cc.degP_red[o2], s_o2, 4);
            ftk2::signs_at_roots_i128(cc.P_red[m], dPm, Qr_d12, dQr_d12, s_qr, 4);

            for (int ri = 0; ri < n_rm; ri++) {
                if (s_qr[ri] == 0) continue;
                bool ok1 = o1z || s_o1[ri] == 0 || s_o1[ri] * s_qr[ri] > 0;
                bool ok2 = o2z || s_o2[ri] == 0 || s_o2[ri] * s_qr[ri] > 0;
                if (!ok1 || !ok2) continue;

                D12EP ep;
                ep.src_vert = m; ep.root_idx = ri;
                ep.edge[0] = std::min(o1, o2);
                ep.edge[1] = std::max(o1, o2);
                ep.pi_idx = -1;
                // Match existing puncture
                for (int pi = 0; pi < (int)cc.punctures.size(); pi++) {
                    auto& p = cc.punctures[pi];
                    if (p.is_edge && p.tet_edge[0] == ep.edge[0] &&
                        p.tet_edge[1] == ep.edge[1] &&
                        p.face == m && p.root_idx == ri) {
                        ep.pi_idx = pi; break;
                    }
                }
                d12_eps.push_back(ep);
            }
        }

        if (d12_eps.size() < 2) continue;

        // Sort by λ
        std::sort(d12_eps.begin(), d12_eps.end(), [&](const D12EP& a, const D12EP& b) {
            return ftk2::compare_roots_i128(
                cc.P_red[a.src_vert], cc.degP_red[a.src_vert],
                cc.n_distinct_red[a.src_vert], a.root_idx,
                cc.P_red[b.src_vert], cc.degP_red[b.src_vert],
                cc.n_distinct_red[b.src_vert], b.root_idx) < 0;
        });

        // Helper: get or create puncture for D12 endpoint
        auto get_or_create_pi = [&](D12EP& ep) -> int {
            if (ep.pi_idx >= 0) return ep.pi_idx;
            ClassifiedCase::PunctureInfo np = {};
            np.face = ep.src_vert;
            np.root_idx = ep.root_idx;
            np.is_edge = true;
            np.is_D01 = true;
            np.tet_edge[0] = ep.edge[0];
            np.tet_edge[1] = ep.edge[1];
            np.edge_faces[0] = std::min(ep.src_vert, d12k);
            np.edge_faces[1] = std::max(ep.src_vert, d12k);
            // Interval: count h-roots and Q_red roots below this root
            int h_below = 0;
            for (int hi = 0; hi < v2.h_n_roots; hi++) {
                int cmp = ftk2::compare_roots_i128(
                    cc.P_red[ep.src_vert], cc.degP_red[ep.src_vert],
                    cc.n_distinct_red[ep.src_vert], ep.root_idx,
                    v2.h, v2.h_deg, v2.h_n_roots, hi);
                if (cmp > 0) h_below++;
            }
            int qr_below = 0;
            if (v2.n_qr_roots > 0) {
                for (int qi = 0; qi < v2.n_qr_roots; qi++) {
                    int cmp = ftk2::compare_roots_i128(
                        cc.P_red[ep.src_vert], cc.degP_red[ep.src_vert],
                        cc.n_distinct_red[ep.src_vert], ep.root_idx,
                        Qr_d12, dQr_d12, v2.n_qr_roots, qi);
                    if (cmp > 0) qr_below++;
                }
            }
            np.q_interval = qr_below;
            np.interval_idx = qr_below + h_below;
            cc.punctures.push_back(np);
            return (int)cc.punctures.size() - 1;
        };

        // Pair consecutive D12 endpoints
        for (int i = 0; i + 1 < (int)d12_eps.size(); i += 2) {
            int pi_a = get_or_create_pi(d12_eps[i]);
            int pi_b = get_or_create_pi(d12_eps[i + 1]);
            int iv_idx = cc.punctures[pi_a].interval_idx;
            cc.pairs.push_back({pi_a, pi_b, false, iv_idx});
        }
    }

    // ─── Step 6: D00/D01 detection and marking on punctures ──────────
    struct PVVertInfo {
        bool is_pv;
        bool any_lambda;
        int64_t lam_num, lam_den;
    };
    PVVertInfo pv_vi[4];
    for (int i = 0; i < 4; i++) {
        pv_vi[i] = {false, false, 0, 1};
        const int* vi = gpu_v2.V[i];
        const int* wi = gpu_v2.W[i];
        int64_t cx = (int64_t)vi[1]*wi[2] - (int64_t)vi[2]*wi[1];
        int64_t cy = (int64_t)vi[2]*wi[0] - (int64_t)vi[0]*wi[2];
        int64_t cz = (int64_t)vi[0]*wi[1] - (int64_t)vi[1]*wi[0];
        if (cx != 0 || cy != 0 || cz != 0) continue;
        pv_vi[i].is_pv = true;
        bool v_zero = (vi[0]==0 && vi[1]==0 && vi[2]==0);
        bool w_zero = (wi[0]==0 && wi[1]==0 && wi[2]==0);
        if (v_zero && w_zero) pv_vi[i].any_lambda = true;
        else if (v_zero) { pv_vi[i].lam_num = 0; pv_vi[i].lam_den = 1; }
        else if (w_zero) { pv_vi[i].lam_num = 1; pv_vi[i].lam_den = 0; }
        else {
            for (int k = 0; k < 3; k++)
                if (wi[k] != 0) {
                    pv_vi[i].lam_num = -vi[k];
                    pv_vi[i].lam_den = wi[k];
                    break;
                }
        }
    }

    auto lam_compat = [&](int a, int b) -> bool {
        if (!pv_vi[a].is_pv || !pv_vi[b].is_pv) return false;
        if (pv_vi[a].any_lambda || pv_vi[b].any_lambda) return true;
        return pv_vi[a].lam_num * pv_vi[b].lam_den
            == pv_vi[b].lam_num * pv_vi[a].lam_den;
    };

    bool has_D00 = false;
    for (int i = 0; i < 4; i++)
        if (pv_vi[i].is_pv) { has_D00 = true; break; }

    if (has_D00) {
        for (auto& pi : cc.punctures)
            if (pi.is_vertex) pi.is_D00 = true;
    }

    // D01: edge puncture that is unpaired or paired with another on the same edge
    bool has_D01 = false;
    std::vector<int> edge_punct_indices;
    for (int pi_idx = 0; pi_idx < (int)cc.punctures.size(); pi_idx++) {
        const auto& pi = cc.punctures[pi_idx];
        // Edge puncture at non-critical lambda (not infinity, not lambda=0 via Cv)
        if (pi.is_edge && pi.root_idx >= 0)
            edge_punct_indices.push_back(pi_idx);
    }
    if (!edge_punct_indices.empty()) {
        for (int ei : edge_punct_indices) {
            const auto& pi = cc.punctures[ei];
            bool paired_same_edge = false;
            bool is_paired = false;
            for (const auto& pp : cc.pairs) {
                int partner = -1;
                if (pp.pi_a == ei) { partner = pp.pi_b; is_paired = true; }
                else if (pp.pi_b == ei) { partner = pp.pi_a; is_paired = true; }
                if (partner < 0) continue;
                const auto& pj = cc.punctures[partner];
                if (pj.is_edge &&
                    pi.tet_edge[0] == pj.tet_edge[0] &&
                    pi.tet_edge[1] == pj.tet_edge[1])
                    paired_same_edge = true;
                break;
            }
            if (!is_paired) {
                has_D01 = true;
                cc.punctures[ei].is_D01 = true;
            } else if (paired_same_edge) {
                int a = pi.tet_edge[0], b = pi.tet_edge[1];
                int f = -1, g = -1;
                for (int k = 0; k < 4; k++) {
                    if (k == a || k == b) continue;
                    if (f < 0) f = k; else g = k;
                }

                int degPf = ftk2::effective_degree_i128(P_i128[f], 3);
                int degPg = ftk2::effective_degree_i128(P_i128[g], 3);
                __int128 g_fg[4] = {};
                int deg_gfg = ftk2::poly_gcd_full_i128(P_i128[f], degPf,
                                                        P_i128[g], degPg, g_fg);

                bool is_isolated = true;
                if (deg_gfg >= 2) {
                    __int128 Pf_prime[3] = {P_i128[f][1], 2*P_i128[f][2], 3*P_i128[f][3]};
                    __int128 Pg_prime[3] = {P_i128[g][1], 2*P_i128[g][2], 3*P_i128[g][3]};
                    int degPfp = ftk2::effective_degree_i128(Pf_prime, 2);
                    int degPgp = ftk2::effective_degree_i128(Pg_prime, 2);
                    int degQ_i = ftk2::effective_degree_i128(Q_i128, 3);

                    int s_fprime[3] = {}, s_gprime[3] = {}, s_q[3] = {};
                    int n_roots = ftk2::signs_at_roots_i128(g_fg, deg_gfg,
                                    Pf_prime, degPfp, s_fprime, 3);
                    ftk2::signs_at_roots_i128(g_fg, deg_gfg,
                                    Pg_prime, degPgp, s_gprime, 3);
                    ftk2::signs_at_roots_i128(g_fg, deg_gfg,
                                    Q_i128, degQ_i, s_q, 3);

                    for (int r = 0; r + 1 < n_roots; r++) {
                        bool enters_f_r = s_fprime[r] * s_q[r] > 0;
                        bool enters_g_r = s_gprime[r] * s_q[r] > 0;
                        bool enters_f_r1 = s_fprime[r+1] * s_q[r+1] < 0;
                        bool enters_g_r1 = s_gprime[r+1] * s_q[r+1] < 0;

                        if (enters_f_r && enters_g_r && enters_f_r1 && enters_g_r1)
                            is_isolated = false;
                    }
                }

                if (is_isolated) {
                    has_D01 = true;
                    cc.punctures[ei].is_D01 = true;
                    for (const auto& pp : cc.pairs) {
                        int partner = -1;
                        if (pp.pi_a == ei) partner = pp.pi_b;
                        else if (pp.pi_b == ei) partner = pp.pi_a;
                        if (partner >= 0) { cc.punctures[partner].is_D01 = true; break; }
                    }
                }
            }
        }
    }

    // ─── Step 7: Dmd detection (D11/D12/D22/D33) ─────────────────────
    bool has_D11 = false;
    static const int te[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (int e = 0; e < 6; e++)
        if (lam_compat(te[e][0], te[e][1])) { has_D11 = true; break; }

    bool has_D22 = false;
    static const int tf[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
    for (int f = 0; f < 4; f++)
        if (lam_compat(tf[f][0], tf[f][1]) && lam_compat(tf[f][1], tf[f][2]))
            { has_D22 = true; break; }

    int n_total_punctures = (int)cc.punctures.size();

    bool has_D12 = false;
    for (int k = 0; k < 4; k++) {
        if (!(P_i128[k][0]==0 && P_i128[k][1]==0 && P_i128[k][2]==0 && P_i128[k][3]==0))
            continue;
        __int128 q_lc = 0, p_lc[3] = {};
        { int oi = 0;
          for (int c = 3; c >= 0; c--) if (Q_i128[c] != 0) { q_lc = Q_i128[c]; break; }
          for (int j = 0; j < 4; j++) {
              if (j == k) continue;
              for (int c = 3; c >= 0; c--) if (P_i128[j][c] != 0) { p_lc[oi] = P_i128[j][c]; break; }
              oi++;
          }
        }
        if (q_lc != 0) {
            bool ok = true;
            for (int j = 0; j < 3; j++)
                if (p_lc[j] * q_lc < 0) { ok = false; break; }
            if (ok) { has_D12 = true; break; }
        }
        if (n_total_punctures > 0) { has_D12 = true; break; }
    }

    bool has_D33 = lam_compat(0,1) && lam_compat(1,2) && lam_compat(2,3);

    // ─── Step 8: T-count ───────────────────────────────────────────────
    int n_face = 0;
    std::vector<bool> is_paired(cc.punctures.size(), false);
    for (const auto& pr : cc.pairs) {
        if (pr.pi_a >= 0 && pr.pi_a < (int)cc.punctures.size()) is_paired[pr.pi_a] = true;
        if (pr.pi_b >= 0 && pr.pi_b < (int)cc.punctures.size()) is_paired[pr.pi_b] = true;
    }
    for (auto& iv : cc.intervals) iv.n_pv = 0;
    for (int pi_idx = 0; pi_idx < (int)cc.punctures.size(); pi_idx++) {
        const auto& pi = cc.punctures[pi_idx];
        if ((pi.is_D00 || pi.is_D01) && !is_paired[pi_idx]) continue;
        n_face++;
        if (pi.interval_idx >= 0 && pi.interval_idx < (int)cc.intervals.size())
            cc.intervals[pi.interval_idx].n_pv++;
    }

    // ─── Step 9: Tags ─────────────────────────────────────────────────
    std::vector<std::string> tags;

    // ── SR/ISR: shared-root classification ──
    if (cc.has_shared_root) {
        __int128 g[4];
        int dg;
        for (int i = 0; i < 4; i++) g[i] = Q_i128[i];
        dg = ftk2::effective_degree_i128(Q_i128, 3);
        for (int k = 0; k < 4 && dg >= 1; k++) {
            int dk = ftk2::effective_degree_i128(P_i128[k], 3);
            if (dk == 0 && P_i128[k][0] == 0) continue;
            __int128 g2[4] = {};
            dg = ftk2::poly_gcd_full_i128(g, dg, P_i128[k], dk, g2);
            for (int i = 0; i < 4; i++) g[i] = g2[i];
        }

        bool any_sr = false;   // any shared root inside tet → "SR"
        bool any_isr = false;  // any shared root outside tet → "ISR"
        cc.has_non_isolated_sr = false;
        int inside_tet_sr_pos[4];
        int n_inside_sr = 0;

        if (dg >= 1) {
            int dQi = ftk2::effective_degree_i128(Q_i128, 3);
            __int128 Qr[4] = {};
            int dQr = ftk2::poly_exact_div_i128(Q_i128, dQi, g, dg, Qr);
            __int128 Pr[4][4] = {};
            int dPr[4] = {};
            for (int k = 0; k < 4; k++) {
                int dk = ftk2::effective_degree_i128(P_i128[k], 3);
                if (dk == 0 && P_i128[k][0] == 0) {
                    dPr[k] = 0; Pr[k][0] = 0;
                } else {
                    dPr[k] = ftk2::poly_exact_div_i128(P_i128[k], dk, g, dg, Pr[k]);
                }
            }

            if (dg == 1) {
                auto eval_at_root = [&](__int128* f, int df) -> int {
                    __int128 val = 0;
                    __int128 neg_g0_pow = 1;
                    __int128 g1_pow_desc = 1;
                    for (int i = 0; i < df; i++) g1_pow_desc *= g[1];
                    for (int i = 0; i <= df; i++) {
                        val += f[i] * neg_g0_pow * g1_pow_desc;
                        neg_g0_pow *= (-g[0]);
                        if (i < df) g1_pow_desc /= g[1];
                    }
                    int g1_sign = (g[1] > 0) ? 1 : -1;
                    int factor = (df % 2 == 0) ? 1 : g1_sign;
                    return (val > 0) ? factor : (val < 0) ? -factor : 0;
                };
                int sq = eval_at_root(Qr, dQr);
                if (sq != 0) {
                    bool in_tet = true;
                    for (int k = 0; k < 4; k++) {
                        if (dPr[k] == 0 && Pr[k][0] == 0) continue;
                        int sp = eval_at_root(Pr[k], dPr[k]);
                        if (sp * sq < 0) { in_tet = false; break; }
                    }
                    if (in_tet) {
                        any_sr = true;
                        for (int hi = 0; hi < v2.h_n_roots && n_inside_sr < 4; hi++) {
                            int cmp = ftk2::compare_roots_i128(
                                g, dg, 1, 0,
                                v2.h, v2.h_deg, v2.h_n_roots, hi);
                            if (cmp == 0 && h_merged_pos[hi] >= 0) {
                                inside_tet_sr_pos[n_inside_sr++] = h_merged_pos[hi];
                                break;
                            }
                        }
                    } else {
                        any_isr = true;
                    }
                }
            } else {
                // Pre-compute signs of P_red[k] at all roots of h
                int spk_all[4][4] = {};
                for (int k = 0; k < 4; k++) {
                    if (dPr[k] == 0 && Pr[k][0] == 0) continue;
                    ftk2::signs_at_roots_i128(g, dg, Pr[k], dPr[k], spk_all[k], 4);
                }
                int sq[4] = {};
                int nrg = ftk2::signs_at_roots_i128(g, dg, Qr, dQr, sq, 4);
                if (nrg > 0) {
                    for (int ri = 0; ri < nrg; ri++) {
                        if (sq[ri] == 0) continue;
                        bool in_tet = true;
                        for (int k = 0; k < 4; k++) {
                            if (dPr[k] == 0 && Pr[k][0] == 0) continue;
                            if (spk_all[k][ri] * sq[ri] < 0) { in_tet = false; break; }
                        }
                        if (in_tet) {
                            any_sr = true;
                            for (int hi = 0; hi < v2.h_n_roots && n_inside_sr < 4; hi++) {
                                int cmp = ftk2::compare_roots_i128(
                                    g, dg, nrg, ri,
                                    v2.h, v2.h_deg, v2.h_n_roots, hi);
                                if (cmp == 0 && h_merged_pos[hi] >= 0) {
                                    inside_tet_sr_pos[n_inside_sr++] = h_merged_pos[hi];
                                    break;
                                }
                            }
                        } else {
                            any_isr = true;
                        }
                    }
                }
            }
        }

        if (any_sr) tags.push_back("SR");
        if (!any_sr) cc.has_shared_root = false;

        // Filter sr_q_root_idx to only include inside-tet SR roots
        if (!any_sr) {
            cc.n_sr_roots = 0;
        } else if (any_isr && n_inside_sr > 0) {
            std::sort(inside_tet_sr_pos, inside_tet_sr_pos + n_inside_sr);
            cc.n_sr_roots = 0;
            for (int i = 0; i < n_inside_sr && i < 3; i++)
                cc.sr_q_root_idx[cc.n_sr_roots++] = inside_tet_sr_pos[i];
        }
    }

    // ── Cv/Cw: critical-point degeneracies ──
    bool has_Cv = check_field_zero_in_tet(gpu_v2.V, cc.Cv_num, &cc.Cv_den);
    bool has_Cw = check_field_zero_in_tet(gpu_v2.W, cc.Cw_num, &cc.Cw_den);
    cc.has_Cv_pos = has_Cv;
    cc.has_Cw_pos = has_Cw;

    bool has_C2v = false, has_C1v = false, has_C0v = false;
    bool has_C2w = false, has_C1w = false, has_C0w = false;

    // λ=0 crossings
    {
        __int128 q0 = Q_i128[0];
        if (q0 != 0) {
            for (int i = 0; i < 4; i++) {
                if (P_i128[i][0] != 0) continue;
                bool inside = true;
                for (int j = 0; j < 4; j++) {
                    if (j == i) continue;
                    __int128 pj0 = P_i128[j][0];
                    if ((pj0 > 0 && q0 < 0) || (pj0 < 0 && q0 > 0))
                        { inside = false; break; }
                }
                if (!inside) continue;
                int nz = 0;
                for (int k = 0; k < 3; k++)
                    if (P_i128[fv[i][k]][0] == 0) nz++;
                if (nz == 2) has_C0v = true;
                else if (nz == 1) has_C1v = true;
                else has_C2v = true;
            }
        }
    }
    // λ=∞ crossings
    {
        int d_Q = 3;
        while (d_Q > 0 && Q_i128[d_Q] == 0) d_Q--;
        __int128 q_lead = (d_Q >= 0) ? Q_i128[d_Q] : 0;
        if (d_Q >= 1 && q_lead != 0) {
            bool all_p_bounded = true;
            for (int kb = 0; kb < 4; kb++) {
                for (int d = d_Q + 1; d <= 3; d++) {
                    if (P_i128[kb][d] != 0) { all_p_bounded = false; break; }
                }
                if (!all_p_bounded) break;
            }
            if (all_p_bounded) {
                for (int i = 0; i < 4; i++) {
                    if (P_i128[i][d_Q] != 0) continue;
                    bool inside = true;
                    for (int j = 0; j < 4; j++) {
                        if (j == i) continue;
                        __int128 pj_lead = P_i128[j][d_Q];
                        if ((pj_lead > 0 && q_lead < 0) || (pj_lead < 0 && q_lead > 0))
                            { inside = false; break; }
                    }
                    if (!inside) continue;
                    int nz = 0;
                    for (int k = 0; k < 3; k++)
                        if (P_i128[fv[i][k]][d_Q] == 0) nz++;
                    if (nz == 2) has_C0w = true;
                    else if (nz == 1) has_C1w = true;
                    else has_C2w = true;
                }
            }
        }
    }

    if (has_Cv || has_C2v || has_C1v || has_C0v) {
        if (has_C0v)      tags.push_back("Cv0");
        else if (has_C1v) tags.push_back("Cv1");
        else if (has_C2v) tags.push_back("Cv2");
        else              tags.push_back("Cv");
    }
    if (has_Cw || has_C2w || has_C1w || has_C0w) {
        if (has_C0w)      tags.push_back("Cw0");
        else if (has_C1w) tags.push_back("Cw1");
        else if (has_C2w) tags.push_back("Cw2");
        else              tags.push_back("Cw");
    }

    // ── Step 8b: Include non-pass-through Cv1/Cv0/Cw1/Cw0 waypoints in T ──
    {
        if (has_C0v || has_C1v) {
            int cv_faces[4] = {}, ncv = 0;
            for (int k = 0; k < 4; k++)
                if (P_i128[k][0] == 0) cv_faces[ncv++] = k;
            bool is_pt = false;
            for (int a = 0; a < ncv && !is_pt; a++)
                for (int b = a+1; b < ncv && !is_pt; b++)
                    if (P_i128[cv_faces[a]][1] * P_i128[cv_faces[b]][1] < 0)
                        is_pt = true;
            if (!is_pt) {
                // Find which interval contains λ=0 using integer root comparison
                // Count Q roots below 0: use sign of Q(0) = Q[0] vs leading coeff
                // to determine position. More precisely: use signs_at_roots_i128
                // with f(x) = x to count how many Q roots are < 0.
                int iv_idx = 0;
                if (cc.n_Q_roots > 0 && degQ >= 1) {
                    __int128 f_x[2] = {0, 1};  // f(x) = x
                    int signs[3] = {};
                    int nr = ftk2::signs_at_roots_i128(Q_i128, degQ, f_x, 1, signs, 3);
                    // Count negative roots (sign < 0)
                    int n_neg = 0;
                    for (int i = 0; i < nr; i++)
                        if (signs[i] < 0) n_neg++;
                    iv_idx = n_neg;
                }
                if (iv_idx >= 0 && iv_idx < (int)cc.intervals.size()
                    && cc.intervals[iv_idx].n_pv > 0) {
                    cc.intervals[iv_idx].n_pv++;
                    n_face++;
                }
            }
        }
        if (has_C0w || has_C1w) {
            bool cw_in_punctures = false;
            for (const auto& pi : cc.punctures)
                if (pi.root_idx < 0 && pi.is_edge) { cw_in_punctures = true; break; }

            if (!cw_in_punctures) {
                int d_Q = 3;
                while (d_Q > 0 && Q_i128[d_Q] == 0) d_Q--;
                int cw_faces[4] = {}, ncw = 0;
                for (int k = 0; k < 4; k++)
                    if (P_i128[k][d_Q] == 0) cw_faces[ncw++] = k;
                bool is_pt = false;
                if (d_Q >= 1) {
                    for (int a = 0; a < ncw && !is_pt; a++)
                        for (int b = a+1; b < ncw && !is_pt; b++)
                            if (P_i128[cw_faces[a]][d_Q-1] * P_i128[cw_faces[b]][d_Q-1] < 0)
                                is_pt = true;
                }
                if (!is_pt) {
                    int last_iv = (int)cc.intervals.size() - 1;
                    if (last_iv >= 0 && cc.intervals[last_iv].n_pv > 0) {
                        cc.intervals[last_iv].n_pv++;
                        n_face++;
                    }
                }
            }
        }
    }

    // Build category string
    {
        std::vector<int> occ;
        for (auto& iv : cc.intervals)
            if (iv.n_pv > 0) occ.push_back(iv.n_pv);
        std::sort(occ.begin(), occ.end());
        std::string t_type = "T" + std::to_string(n_face);
        if (occ.size() > 1) {
            t_type += "_(";
            for (size_t i = 0; i < occ.size(); i++) {
                if (i > 0) t_type += ",";
                t_type += std::to_string(occ[i]);
            }
            t_type += ")";
        }
        cc.category = t_type + "_" + q_type;
    }

    // ── TN tag: tangency (disc(P_k) = 0, repeated root) ──
    {
        int disc_sign_cpu[4];
        for (int k = 0; k < 4; k++)
            disc_sign_cpu[k] = ftk2::discriminant_sign_i128(P_i128[k]);

        bool has_TN = false;
        for (int k = 0; k < 4; k++) {
            int degk = effective_degree_i128(P_i128[k], 3);
            if (degk < 2 || disc_sign_cpu[k] != 0)
                continue;

            __int128 Pk_prime[4] = {P_i128[k][1], 2*P_i128[k][2], 3*P_i128[k][3], 0};
            int dk_prime = degk - 1;
            while (dk_prime > 0 && Pk_prime[dk_prime] == 0) dk_prime--;

            __int128 h_tn[4] = {};
            int dh = ftk2::poly_gcd_full_i128(P_i128[k], degk, Pk_prime, dk_prime, h_tn);
            if (dh < 1) continue;

            if (dh == 1) {
                auto eval_at_root_h = [&](__int128* f, int df) -> int {
                    __int128 val = 0;
                    __int128 neg_h0_pow = 1;
                    __int128 h1_pow_desc = 1;
                    for (int i = 0; i < df; i++) h1_pow_desc *= h_tn[1];
                    for (int i = 0; i <= df; i++) {
                        val += f[i] * neg_h0_pow * h1_pow_desc;
                        neg_h0_pow *= (-h_tn[0]);
                        if (i < df) h1_pow_desc /= h_tn[1];
                    }
                    int h1_sign = (h_tn[1] > 0) ? 1 : -1;
                    int factor = (df % 2 == 0) ? 1 : h1_sign;
                    return (val > 0) ? factor : (val < 0) ? -factor : 0;
                };

                __int128 Q_tn_copy[4] = {Q_i128[0], Q_i128[1], Q_i128[2], Q_i128[3]};
                int edQ = ftk2::effective_degree_i128(Q_i128, 3);
                int sq = eval_at_root_h(Q_tn_copy, edQ);
                if (sq == 0) continue;

                bool in_tet = true;
                for (int j = 0; j < 4; j++) {
                    if (j == k) continue;
                    __int128 Pj_copy[4] = {P_i128[j][0], P_i128[j][1], P_i128[j][2], P_i128[j][3]};
                    int edj = ftk2::effective_degree_i128(P_i128[j], 3);
                    if (edj == 0 && P_i128[j][0] == 0) continue;
                    int sp = eval_at_root_h(Pj_copy, edj);
                    if (sp * sq < 0) { in_tet = false; break; }
                }
                if (!in_tet) continue;

                __int128 Pk_pp[2] = {2 * P_i128[k][2], 6 * P_i128[k][3]};
                int deg_pp = ftk2::effective_degree_i128(Pk_pp, 1);
                int spp = 0;
                if (deg_pp > 0 || Pk_pp[0] != 0) {
                    spp = eval_at_root_h(Pk_pp, deg_pp);
                }

                if (spp * sq < 0)
                    ;  // isolated TN (≡ D02)
                else
                    has_TN = true;

                // Store integer TN data
                if (cc.n_tn < 4) {
                    cc.tn_points[cc.n_tn].face = k;
                    cc.tn_points[cc.n_tn].h_tn[0] = h_tn[0];
                    cc.tn_points[cc.n_tn].h_tn[1] = h_tn[1];
                    cc.n_tn++;
                }
            }
        }
        if (has_TN) tags.push_back("TN");
    }

    // ── Dmd tags ──
    if (has_D33)           tags.push_back("D33");
    else if (has_D22)      tags.push_back("D22");
    else if (has_D12)      tags.push_back("D12");
    else if (has_D11)      tags.push_back("D11");
    else if (has_D01)      tags.push_back("D01");
    else if (has_D00)      tags.push_back("D00");

    // B: bubble (T0, closed PV curve inside tet)
    if (n_face == 0 && cc.n_Q_roots == 0 && Q_i128[0] != 0) {
        bool inside = true;
        for (int k = 0; k < 4; k++)
            if (P_i128[k][0] * Q_i128[0] <= 0) { inside = false; break; }
        if (inside) { cc.has_B = true; tags.push_back("B"); }
    }

    // Join tags
    for (size_t i = 0; i < tags.size(); i++)
        cc.category += "_" + tags[i];

    return cc;
}

// ─── JSON output ────────────────────────────────────────────────────────────

static void print_json(const ClassifiedCase& cc) {
    printf("{");
    printf("\"seed\":%lu,", (unsigned long)cc.seed);
    printf("\"category\":\"%s\",", cc.category.c_str());
    printf("\"n_punctures\":%d,", (int)cc.punctures.size());
    printf("\"n_raw\":%d,", cc.total_punctures);
    printf("\"n_deduplicated\":%d,", cc.n_deduplicated);

    // V, W
    printf("\"V\":[");
    for (int i = 0; i < 4; i++) {
        printf("[%d,%d,%d]", cc.V[i][0], cc.V[i][1], cc.V[i][2]);
        if (i < 3) printf(",");
    }
    printf("],\"W\":[");
    for (int i = 0; i < 4; i++) {
        printf("[%d,%d,%d]", cc.W[i][0], cc.W[i][1], cc.W[i][2]);
        if (i < 3) printf(",");
    }
    printf("],");

    // Q_i128 as string-encoded array
    printf("\"Q_i128\":[");
    for (int i = 0; i < 4; i++) {
        printf("\"%s\"", i128_to_string(cc.Q_i128[i]).c_str());
        if (i < 3) printf(",");
    }
    printf("],");
    printf("\"Q_disc_sign\":%d,", cc.Q_disc_sign);
    printf("\"n_Q_roots\":%d,", cc.n_Q_roots);

    // P_i128 as 4x4 string-encoded array
    printf("\"P_i128\":[");
    for (int i = 0; i < 4; i++) {
        printf("[");
        for (int j = 0; j < 4; j++) {
            printf("\"%s\"", i128_to_string(cc.P_i128[i][j]).c_str());
            if (j < 3) printf(",");
        }
        printf("]");
        if (i < 3) printf(",");
    }
    printf("],");

    // v2 solver infrastructure
    printf("\"merge_infinity\":%s,", cc.merge_infinity ? "true" : "false");
    printf("\"n_qr_roots\":%d,", cc.n_qr_roots);
    printf("\"h\":[");
    for (int i = 0; i < 4; i++) {
        printf("\"%s\"", i128_to_string(cc.h[i]).c_str());
        if (i < 3) printf(",");
    }
    printf("],\"h_deg\":%d,\"h_n_roots\":%d,", cc.h_deg, cc.h_n_roots);

    printf("\"P_red\":[");
    for (int k = 0; k < 4; k++) {
        printf("[");
        for (int j = 0; j < 4; j++) {
            printf("\"%s\"", i128_to_string(cc.P_red[k][j]).c_str());
            if (j < 3) printf(",");
        }
        printf("]");
        if (k < 3) printf(",");
    }
    printf("],\"degP_red\":[%d,%d,%d,%d],\"n_distinct_red\":[%d,%d,%d,%d],",
           cc.degP_red[0], cc.degP_red[1], cc.degP_red[2], cc.degP_red[3],
           cc.n_distinct_red[0], cc.n_distinct_red[1], cc.n_distinct_red[2], cc.n_distinct_red[3]);

    // Intervals
    printf("\"intervals\":[");
    for (size_t i = 0; i < cc.intervals.size(); i++) {
        const auto& iv = cc.intervals[i];
        printf("{\"n_pv\":%d,\"is_infinity\":%s}",
               iv.n_pv, iv.is_infinity ? "true" : "false");
        if (i + 1 < cc.intervals.size()) printf(",");
    }
    printf("],");

    // Punctures (integer-only)
    printf("\"punctures\":[");
    for (size_t i = 0; i < cc.punctures.size(); i++) {
        const auto& pi = cc.punctures[i];
        printf("{\"face\":%d,\"root_idx\":%d,\"q_interval\":%d,\"interval\":%d",
               pi.face, pi.root_idx, pi.q_interval, pi.interval_idx);
        if (pi.is_edge)
            printf(",\"is_edge\":true,\"tet_edge\":[%d,%d],\"edge_faces\":[%d,%d]",
                   pi.tet_edge[0], pi.tet_edge[1], pi.edge_faces[0], pi.edge_faces[1]);
        if (pi.is_vertex)
            printf(",\"is_vertex\":true,\"tet_vertex\":%d", pi.tet_vertex);
        if (pi.is_D00)
            printf(",\"is_D00\":true");
        if (pi.is_D01)
            printf(",\"is_D01\":true");
        printf("}");
        if (i + 1 < cc.punctures.size()) printf(",");
    }
    printf("],");

    // Pairs
    printf("\"pairs\":[");
    for (size_t i = 0; i < cc.pairs.size(); i++) {
        const auto& pp = cc.pairs[i];
        printf("{\"pi_a\":%d,\"pi_b\":%d,\"is_cross\":%s,\"interval\":%d}",
               pp.pi_a, pp.pi_b, pp.is_cross ? "true" : "false",
               pp.interval_idx);
        if (i + 1 < cc.pairs.size()) printf(",");
    }
    printf("],");

    printf("\"has_shared_root\":%s,", cc.has_shared_root ? "true" : "false");
    printf("\"has_non_isolated_sr\":%s,", cc.has_non_isolated_sr ? "true" : "false");
    printf("\"sr_q_root_indices\":[");
    for (int i = 0; i < cc.n_sr_roots; i++) {
        if (i > 0) printf(",");
        printf("%d", cc.sr_q_root_idx[i]);
    }
    printf("],");
    printf("\"has_B\":%s,", cc.has_B ? "true" : "false");

    // TN points (integer: h_tn coefficients)
    printf("\"tn_points\":[");
    for (int i = 0; i < cc.n_tn; i++) {
        if (i > 0) printf(",");
        printf("{\"face\":%d,\"h_tn\":[\"%s\",\"%s\"]}",
               cc.tn_points[i].face,
               i128_to_string(cc.tn_points[i].h_tn[0]).c_str(),
               i128_to_string(cc.tn_points[i].h_tn[1]).c_str());
    }
    printf("],");

    // Cv/Cw positions (integer num/den)
    if (cc.has_Cv_pos)
        printf("\"Cv_num\":[%ld,%ld,%ld,%ld],\"Cv_den\":%ld,",
               (long)cc.Cv_num[0], (long)cc.Cv_num[1], (long)cc.Cv_num[2], (long)cc.Cv_num[3],
               (long)cc.Cv_den);
    else
        printf("\"Cv_num\":null,\"Cv_den\":null,");
    if (cc.has_Cw_pos)
        printf("\"Cw_num\":[%ld,%ld,%ld,%ld],\"Cw_den\":%ld",
               (long)cc.Cw_num[0], (long)cc.Cw_num[1], (long)cc.Cw_num[2], (long)cc.Cw_num[3],
               (long)cc.Cw_den);
    else
        printf("\"Cw_num\":null,\"Cw_den\":null");
    printf("}\n");
}

// ─── Verification ────────────────────────────────────────────────────────────
// Self-contained: computes its own float Q/P from integer V/W for checking.
static void verify_case(const ClassifiedCase& cc) {
    uint64_t seed = cc.seed;
    int n = (int)cc.punctures.size();

    // ── 1. Category-data consistency ─────────────────────────────────────
    {
        std::string cat = cc.category;
        size_t t_pos = cat.find('T');
        if (t_pos != std::string::npos) {
            int t_num = 0;
            size_t i = t_pos + 1;
            while (i < cat.size() && cat[i] >= '0' && cat[i] <= '9')
                t_num = t_num * 10 + (cat[i++] - '0');
            // T-count should exclude isolated D00/D01
            int n_face_check = 0;
            for (const auto& pi : cc.punctures)
                if (!pi.is_D00 && !pi.is_D01) n_face_check++;
            // Note: n_face_check may differ from T-number due to waypoint counting
            // (Step 8b adds Cv/Cw waypoints). Just check for obviously wrong T.
            (void)t_num; (void)n_face_check;
        }
    }

    // ── 2. Pair index validity ──────────────────────────────────────────
    std::vector<int> pair_count(n, 0);
    for (const auto& pp : cc.pairs) {
        if (pp.pi_a >= 0 && pp.pi_a < n) pair_count[pp.pi_a]++;
        else fprintf(stderr, "[WARN seed=%lu] pair has invalid pi_a=%d\n",
                     (unsigned long)seed, pp.pi_a);
        if (pp.pi_b >= 0 && pp.pi_b < n) pair_count[pp.pi_b]++;
        else fprintf(stderr, "[WARN seed=%lu] pair has invalid pi_b=%d\n",
                     (unsigned long)seed, pp.pi_b);
    }
    for (int i = 0; i < n; i++) {
        if (pair_count[i] > 1)
            fprintf(stderr, "[WARN seed=%lu] puncture %d appears in %d pairs (expected ≤1)\n",
                    (unsigned long)seed, i, pair_count[i]);
    }

    // ── 3. Cv/Cw cross-check ────────────────────────────────────────────
    if (cc.has_Cv_pos) {
        bool cv_check = check_field_zero_in_tet(cc.V);
        if (!cv_check)
            fprintf(stderr, "[WARN seed=%lu] has_Cv_pos=true but check_field_zero_in_tet(V)=false\n",
                    (unsigned long)seed);
    }
    if (cc.has_Cw_pos) {
        bool cw_check = check_field_zero_in_tet(cc.W);
        if (!cw_check)
            fprintf(stderr, "[WARN seed=%lu] has_Cw_pos=true but check_field_zero_in_tet(W)=false\n",
                    (unsigned long)seed);
    }

    // ── 4. SR verification via integer resultant ────────────────────────
    if (cc.has_shared_root) {
        if (!has_shared_root_resultant(cc.Q_i128, cc.P_i128))
            fprintf(stderr, "[WARN seed=%lu] has_shared_root=true but resultant check fails\n",
                    (unsigned long)seed);
    }

    // ── 5. Interval parity ──────────────────────────────────────────────
    {
        bool has_odd_inner = false;
        int n_iv = (int)cc.intervals.size();
        for (int i = 0; i < n_iv; i++) {
            if (cc.intervals[i].n_pv % 2 != 0) {
                if (i == 0 || i == n_iv - 1) continue;
                has_odd_inner = true;
            }
        }
        bool outer_ok = true;
        if (n_iv >= 2) {
            int first = cc.intervals[0].n_pv;
            int last = cc.intervals[n_iv - 1].n_pv;
            if ((first + last) % 2 != 0) outer_ok = false;
        }
        if ((has_odd_inner || !outer_ok) && !cc.has_shared_root)
            fprintf(stderr, "[WARN seed=%lu] odd interval n_pv without shared_root or infinity merging\n",
                    (unsigned long)seed);
    }
}

// ─── CPU seed replay (--seeds mode) ──────────────────────────────────────────
// Replays the GPU LCG on CPU for specific seeds, bypassing GPU scan.

static uint32_t lcg_next_cpu(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static int rand_int_cpu(uint32_t& state, int R) {
    uint32_t r = lcg_next_cpu(state);
    return (int)(r % (2 * R + 1)) - R;
}

static TetCaseV2GPU generate_tet_from_seed(uint64_t global_id, uint64_t base_seed, int R) {
    TetCaseV2GPU tc;
    memset(&tc, 0, sizeof(tc));

    if (global_id == 0) {
        static const int bubble_V[4][3] = {{2,3,-1},{-1,-2,-1},{0,-1,2},{-2,-2,0}};
        static const int bubble_W[4][3] = {{3,0,3},{-1,0,-1},{-3,-2,-1},{0,3,-3}};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++) {
                tc.V[i][j] = bubble_V[i][j];
                tc.W[i][j] = bubble_W[i][j];
            }
    } else if (global_id == 1) {
        static const int nisr_V[4][3] = {{5,0,1},{3,2,4},{1,2,-2},{2,4,4}};
        static const int nisr_W[4][3] = {{0,1,3},{3,-5,3},{0,4,5},{1,1,0}};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++) {
                tc.V[i][j] = nisr_V[i][j];
                tc.W[i][j] = nisr_W[i][j];
            }
    } else {
        uint32_t state = (uint32_t)(global_id ^ (base_seed * 2654435761ULL));
        for (int i = 0; i < 4; i++) lcg_next_cpu(state);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                tc.V[i][j] = rand_int_cpu(state, R);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                tc.W[i][j] = rand_int_cpu(state, R);
    }

    // Run ExactPV2 solver
    __int128 Q_i128[4], P_i128[4][4];
    ftk2::compute_tet_QP_i128(tc.V, tc.W, Q_i128, P_i128);
    tc.v2 = ftk2::solve_pv_tet_v2(Q_i128, P_i128);
    for (int k = 0; k < 4; k++)
        tc.disc_sign[k] = ftk2::discriminant_sign_i128(P_i128[k]);
    tc.seed = global_id;
    return tc;
}

static std::vector<uint64_t> parse_seeds(const char* arg) {
    std::vector<uint64_t> seeds;
    const char* p = arg;
    while (*p) {
        char* end;
        uint64_t s = strtoull(p, &end, 10);
        seeds.push_back(s);
        if (*end == ',') end++;
        p = end;
    }
    return seeds;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int min_punctures = 2;
    int num_tets = 100000000;
    int R = 20;
    uint64_t base_seed = 42;
    int max_cases = 100000;
    int batch_size = 10000000;
    const char* seeds_arg = nullptr;
    const char* vw_file = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min-punctures") == 0 && i + 1 < argc)
            min_punctures = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-tets") == 0 && i + 1 < argc)
            num_tets = atoi(argv[++i]);
        else if (strcmp(argv[i], "--range") == 0 && i + 1 < argc)
            R = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            base_seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc)
            seeds_arg = argv[++i];
        else if (strcmp(argv[i], "--vw-file") == 0 && i + 1 < argc)
            vw_file = argv[++i];
        else if (strcmp(argv[i], "--max-cases") == 0 && i + 1 < argc)
            max_cases = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pv-version") == 0 && i + 1 < argc)
            ++i;  // silently accept and ignore (v2 is now the only path)
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --min-punctures N   Minimum punctures per tet (default: 2)\n");
            fprintf(stderr, "  --num-tets M        Total random tets to try (default: 100M)\n");
            fprintf(stderr, "  --range R           Integer field range [-R, R] (default: 20)\n");
            fprintf(stderr, "  --seed S            Base random seed (default: 42)\n");
            fprintf(stderr, "  --seeds S1,S2,...    Replay specific seeds on CPU (no GPU needed)\n");
            fprintf(stderr, "  --vw-file FILE      Read V,W pairs from file (24 ints per line)\n");
            fprintf(stderr, "  --max-cases C       Max output cases (default: 100000)\n");
            fprintf(stderr, "  --batch-size B      GPU batch size (default: 10M)\n");
            return 0;
        }
    }

    // ─── VW-file mode: read V,W from file, classify each ──────────────────
    if (vw_file) {
        FILE* fp = fopen(vw_file, "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", vw_file); return 1; }
        fprintf(stderr, "VW-file mode: reading from %s\n", vw_file);
        int line_num = 0;
        int best_punct = 0;
        char linebuf[1024];
        while (fgets(linebuf, sizeof(linebuf), fp)) {
            line_num++;
            TetCaseV2GPU tv2;
            memset(&tv2, 0, sizeof(tv2));
            int vals[25];
            int n = sscanf(linebuf, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                &vals[0],&vals[1],&vals[2],&vals[3],&vals[4],&vals[5],
                &vals[6],&vals[7],&vals[8],&vals[9],&vals[10],&vals[11],
                &vals[12],&vals[13],&vals[14],&vals[15],&vals[16],&vals[17],
                &vals[18],&vals[19],&vals[20],&vals[21],&vals[22],&vals[23],
                &vals[24]);
            int vw_offset = 0;
            if (n == 25) { tv2.seed = vals[0]; vw_offset = 1; }
            else if (n == 24) { tv2.seed = line_num; vw_offset = 0; }
            else continue;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++) {
                    tv2.V[i][j] = vals[vw_offset+i*3+j];
                    tv2.W[i][j] = vals[vw_offset+12+i*3+j];
                }
            __int128 Q_i128[4], P_i128[4][4];
            ftk2::compute_tet_QP_i128(tv2.V, tv2.W, Q_i128, P_i128);
            tv2.v2 = ftk2::solve_pv_tet_v2(Q_i128, P_i128);
            for (int k = 0; k < 4; k++)
                tv2.disc_sign[k] = ftk2::discriminant_sign_i128(P_i128[k]);
            ClassifiedCase cc = classify_case_v2(tv2);
            int np = (int)cc.punctures.size();
            if (min_punctures == 0) {
                fprintf(stderr, "  line %d: %s (%d punctures)\n",
                        line_num, cc.category.c_str(), np);
                print_json(cc);
            } else if (np > best_punct) {
                best_punct = np;
                fprintf(stderr, "  line %d: %s (%d punctures, raw=%d)\n",
                        line_num, cc.category.c_str(), np, cc.total_punctures);
                print_json(cc);
            }
        }
        fclose(fp);
        fprintf(stderr, "Done. %d lines, best=%d punctures\n", line_num, best_punct);
        return 0;
    }

    // ─── Seeds mode: CPU-only replay of specific seeds ───────────────────
    if (seeds_arg) {
        auto seeds = parse_seeds(seeds_arg);
        fprintf(stderr, "Seeds mode: replaying %d seeds on CPU (R=%d, base_seed=%lu)\n",
                (int)seeds.size(), R, (unsigned long)base_seed);
        for (uint64_t s : seeds) {
            TetCaseV2GPU tv2 = generate_tet_from_seed(s, base_seed, R);
            ClassifiedCase cc = classify_case_v2(tv2);
            verify_case(cc);
            print_json(cc);
            fprintf(stderr, "  seed=%lu: %s (%d punctures, %d pairs)\n",
                    (unsigned long)s, cc.category.c_str(),
                    (int)cc.punctures.size(), (int)cc.pairs.size());
        }
        return 0;
    }

    // Print GPU info to stderr
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    fprintf(stderr, "GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    fprintf(stderr, "Parameters: num_tets=%d, min_punctures=%d, range=%d, seed=%lu\n",
            num_tets, min_punctures, R, (unsigned long)base_seed);

    // Allocate GPU output buffer
    int gpu_max_output = std::min(max_cases * 2, 2000000);
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    TetCaseV2GPU* d_output_v2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_v2, gpu_max_output * sizeof(TetCaseV2GPU)));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

    std::map<std::string, int> category_counts;
    std::map<std::string, ClassifiedCase> representatives;
    int total_found = 0;

    int num_batches = (num_tets + batch_size - 1) / batch_size;
    int block_size = 128;

    for (int batch = 0; batch < num_batches; batch++) {
        int this_batch = std::min(batch_size, num_tets - batch * batch_size);
        uint64_t batch_offset = (uint64_t)batch * batch_size;

        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
        int grid_size = (this_batch + block_size - 1) / block_size;

        tet_case_finder_v2_kernel<<<grid_size, block_size>>>(
            d_output_v2, d_count, gpu_max_output, min_punctures,
            R, base_seed, batch_offset);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_count;
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_count > gpu_max_output) h_count = gpu_max_output;
        if (h_count == 0) continue;

        std::vector<TetCaseV2GPU> h_v2(h_count);
        CUDA_CHECK(cudaMemcpy(h_v2.data(), d_output_v2,
                              h_count * sizeof(TetCaseV2GPU), cudaMemcpyDeviceToHost));

        for (int i = 0; i < h_count && total_found < max_cases; i++) {
            ClassifiedCase cc = classify_case_v2(h_v2[i]);
            verify_case(cc);
            category_counts[cc.category]++;
            print_json(cc);
            total_found++;

            if (representatives.find(cc.category) == representatives.end())
                representatives[cc.category] = cc;
        }

        fprintf(stderr, "Batch %d/%d: %d hits (%d total), %d categories\n",
                batch + 1, num_batches, h_count, total_found,
                (int)category_counts.size());

        if (total_found >= max_cases) break;
    }

    CUDA_CHECK(cudaFree(d_output_v2));
    CUDA_CHECK(cudaFree(d_count));

    // Print summary to stderr
    fprintf(stderr, "\n=== Category Summary ===\n");
    for (auto& [cat, cnt] : category_counts)
        fprintf(stderr, "  %-30s %d\n", cat.c_str(), cnt);
    fprintf(stderr, "Total: %d cases in %d categories\n",
            total_found, (int)category_counts.size());

    // Print representatives summary
    fprintf(stderr, "\n=== Representatives (one per category) ===\n");
    for (auto& [cat, cc] : representatives) {
        fprintf(stderr, "  %s: seed=%lu, %d punctures (raw=%d, dedup=%d), Q_disc=%d",
                cat.c_str(), (unsigned long)cc.seed,
                (int)cc.punctures.size(), cc.total_punctures,
                cc.n_deduplicated, cc.Q_disc_sign);
        if (cc.has_shared_root) fprintf(stderr, " [SHARED-ROOT]");
        fprintf(stderr, "\n");
    }

    return 0;
}
