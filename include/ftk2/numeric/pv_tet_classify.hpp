// Pure-integer PV tet classifier — extracted from pv_tet_case_finder.cu
// All topological decisions use __int128.  NO float/double anywhere.
#ifndef FTK2_PV_TET_CLASSIFY_HPP
#define FTK2_PV_TET_CLASSIFY_HPP

#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <cstdio>
#include <cstdint>

namespace ftk2 {

// ─── GPU output struct for ExactPV2 ──────────────────────────────────────────
struct TetCaseV2GPU {
    int V[4][3], W[4][3];
    ExactPV2Result v2;
    int disc_sign[4];   // discriminant sign of P[k] for each face
    uint64_t seed;
};

// ─── Classified case ────────────────────────────────────────────────────────
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

    int d22_face;   // face index with D22 degeneracy (-1 if none)

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
inline bool check_field_zero_in_tet(const int F[4][3],
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

// Check if field=0 is inside tet for D23 (coplanar) case.
// Projects 4 coplanar 3D integer points to 2D and checks if origin
// is in convex hull via triangle-based point-in-polygon.
// Pure integer, no floats.
inline bool check_field_zero_coplanar(const int F[4][3]) {
    // Find two non-parallel vectors for normal computation
    int64_t nx = 0, ny = 0, nz = 0;
    bool found_pair = false;
    for (int i = 0; i < 4 && !found_pair; i++)
        for (int j = i + 1; j < 4 && !found_pair; j++) {
            nx = (int64_t)F[i][1]*F[j][2] - (int64_t)F[i][2]*F[j][1];
            ny = (int64_t)F[i][2]*F[j][0] - (int64_t)F[i][0]*F[j][2];
            nz = (int64_t)F[i][0]*F[j][1] - (int64_t)F[i][1]*F[j][0];
            if (nx != 0 || ny != 0 || nz != 0) found_pair = true;
        }

    if (!found_pair) {
        // All vectors parallel or zero: 1D case
        int ui = -1;
        for (int i = 0; i < 4; i++)
            if (F[i][0] != 0 || F[i][1] != 0 || F[i][2] != 0)
                { ui = i; break; }
        if (ui < 0) return true; // all zero → origin is in hull

        // Origin in hull iff some projections ≤ 0 and some ≥ 0
        bool has_nonpos = false, has_nonneg = false;
        for (int i = 0; i < 4; i++) {
            int64_t dot = (int64_t)F[i][0]*F[ui][0]
                        + (int64_t)F[i][1]*F[ui][1]
                        + (int64_t)F[i][2]*F[ui][2];
            if (dot <= 0) has_nonpos = true;
            if (dot >= 0) has_nonneg = true;
        }
        return has_nonpos && has_nonneg;
    }

    // Project to 2D: drop coordinate with largest |normal component|
    int64_t anx = nx < 0 ? -nx : nx;
    int64_t any = ny < 0 ? -ny : ny;
    int64_t anz = nz < 0 ? -nz : nz;
    int c0, c1;
    if (anx >= any && anx >= anz) { c0 = 1; c1 = 2; }
    else if (any >= anz) { c0 = 0; c1 = 2; }
    else { c0 = 0; c1 = 1; }

    int64_t px[4], py[4];
    for (int i = 0; i < 4; i++) {
        px[i] = F[i][c0];
        py[i] = F[i][c1];
    }

    // Check if origin is inside any triangle formed by 3 of 4 points
    // (Carathéodory: convex hull = union of all such triangles)
    static const int tri[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    for (int t = 0; t < 4; t++) {
        int a = tri[t][0], b = tri[t][1], c = tri[t][2];
        // 2D cross products: s1 = A×B, s2 = B×C, s3 = C×A
        int64_t s1 = px[a]*py[b] - py[a]*px[b];
        int64_t s2 = px[b]*py[c] - py[b]*px[c];
        int64_t s3 = px[c]*py[a] - py[c]*px[a];
        // Skip degenerate triangle (collinear)
        if (s1 == 0 && s2 == 0 && s3 == 0) continue;
        // Origin inside (including boundary) if all signs agree
        if (s1 >= 0 && s2 >= 0 && s3 >= 0) return true;
        if (s1 <= 0 && s2 <= 0 && s3 <= 0) return true;
    }
    return false;
}

// Exact shared-root detection: Resultant(Q, P_k) = 0 iff they share a root.
// Q and P_k are degree-3 polynomials with __int128 coefficients.
// The 6x6 Sylvester determinant is computed exactly.
// Returns true if any P_k shares a root with Q.
inline bool has_shared_root_resultant(const __int128 Q_i128[4],
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
        __int128 prev_pivot = 1;
        bool zero_det = false;
        for (int col = 0; col < N; col++) {
            // Partial pivoting
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
            for (int row = col + 1; row < N; row++)
                M[row][col] = 0;
            prev_pivot = M[col][col];
        }

        if (zero_det || M[N-1][N-1] == 0) return true;
    }
    return false;
}

// __int128 to string helper
inline std::string i128_to_string(__int128 v) {
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
inline ClassifiedCase classify_case_v2(const TetCaseV2GPU& gpu_v2) {
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
    cc.d22_face = -1;
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
            __int128 D = Q_i128[2]*Q_i128[2] - (__int128)3*Q_i128[1]*Q_i128[3];
            cc.n_Q_roots = (D == 0) ? 1 : 2;
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

        // Vertex: find vertex from edge_faces
        if (ci.is_vertex) {
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
        cc.intervals.push_back({0, true});
        for (int i = 0; i + 1 < cc.n_Q_roots; i++)
            cc.intervals.push_back({0, false});
        cc.intervals.push_back({0, true});
    } else {
        cc.intervals.push_back({0, true});
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
    for (int d12k = 0; d12k < 4; d12k++) {
        if (P_i128[d12k][0] != 0 || P_i128[d12k][1] != 0 ||
            P_i128[d12k][2] != 0 || P_i128[d12k][3] != 0) continue;

        int fv_d12[3], nfv = 0;
        for (int j = 0; j < 4; j++) if (j != d12k) fv_d12[nfv++] = j;

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
            int m = fv_d12[mi];
            int dPm = cc.degP_red[m], n_rm = cc.n_distinct_red[m];
            if (dPm <= 0 || n_rm <= 0) continue;
            int o1 = fv_d12[(mi+1)%3], o2 = fv_d12[(mi+2)%3];

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
            { has_D22 = true; cc.d22_face = f; break; }

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

    // D23: PV surface in tet interior — all V_i,W_i coplanar (rank ≤ 2)
    bool has_D23 = false;
    {
        int64_t vecs[8][3];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                vecs[i][j] = gpu_v2.V[i][j];
                vecs[i+4][j] = gpu_v2.W[i][j];
            }
        }
        int ui = -1, vi_idx = -1;
        for (int i = 0; i < 8 && ui < 0; i++)
            if (vecs[i][0] != 0 || vecs[i][1] != 0 || vecs[i][2] != 0)
                ui = i;
        if (ui >= 0) {
            for (int i = ui + 1; i < 8 && vi_idx < 0; i++) {
                int64_t cx = vecs[ui][1]*vecs[i][2] - vecs[ui][2]*vecs[i][1];
                int64_t cy = vecs[ui][2]*vecs[i][0] - vecs[ui][0]*vecs[i][2];
                int64_t cz = vecs[ui][0]*vecs[i][1] - vecs[ui][1]*vecs[i][0];
                if (cx != 0 || cy != 0 || cz != 0) vi_idx = i;
            }
        }
        if (ui >= 0 && vi_idx >= 0) {
            has_D23 = true;
            int64_t ux = vecs[ui][0], uy = vecs[ui][1], uz = vecs[ui][2];
            int64_t vx = vecs[vi_idx][0], vy = vecs[vi_idx][1], vz = vecs[vi_idx][2];
            for (int i = 0; i < 8; i++) {
                if (i == ui || i == vi_idx) continue;
                int64_t wx = vecs[i][0], wy = vecs[i][1], wz = vecs[i][2];
                int64_t det = ux*(vy*wz - vz*wy)
                            - uy*(vx*wz - vz*wx)
                            + uz*(vx*wy - vy*wx);
                if (det != 0) { has_D23 = false; break; }
            }
        } else {
            has_D23 = (ui >= 0);
        }
    }

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

        bool any_sr = false;
        bool any_isr = false;
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

        // Filter sr_q_root_idx to only include inside-tet roots
        if (!any_sr) {
            cc.n_sr_roots = 0;
        } else if (any_isr && n_inside_sr > 0) {
            std::sort(inside_tet_sr_pos, inside_tet_sr_pos + n_inside_sr);
            cc.n_sr_roots = 0;
            for (int i = 0; i < n_inside_sr && i < 3; i++)
                cc.sr_q_root_idx[cc.n_sr_roots++] = inside_tet_sr_pos[i];
        }

        // Classify inside-tet roots: SR (arc crosses through) vs ISR (isolated)
        bool has_connecting_sr = false, has_isolated_sr = false;
        for (int i = 0; i < n_inside_sr; i++) {
            int pos = inside_tet_sr_pos[i];
            bool crossed = false;
            for (size_t pi = 0; pi < cc.pairs.size() && !crossed; pi++) {
                const auto& pp = cc.pairs[pi];
                if (pp.pi_a < 0 || pp.pi_b < 0) continue;
                int iv_a = cc.punctures[pp.pi_a].interval_idx;
                int iv_b = cc.punctures[pp.pi_b].interval_idx;
                int lo = std::min(iv_a, iv_b);
                int hi = std::max(iv_a, iv_b);
                if (!pp.is_cross) {
                    if (pos >= lo && pos < hi) crossed = true;
                } else {
                    if (pos < lo || pos >= hi) crossed = true;
                }
            }
            if (crossed) has_connecting_sr = true;
            else has_isolated_sr = true;
        }

        if (has_connecting_sr) tags.push_back("SR");
        if (has_isolated_sr) { tags.push_back("ISR"); cc.has_non_isolated_sr = true; }
        if (!has_connecting_sr && !has_isolated_sr) cc.has_shared_root = false;
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

    // D23 fallback for Cv/Cw: coplanar vectors make det=0, so
    // check_field_zero_in_tet fails. Use 2D projection instead.
    if (has_D23) {
        if (!has_Cv) {
            has_Cv = check_field_zero_coplanar(gpu_v2.V);
            cc.has_Cv_pos = has_Cv;
            if (has_Cv) {
                // Cv0: any V[i] = (0,0,0)
                for (int i = 0; i < 4; i++)
                    if (gpu_v2.V[i][0]==0 && gpu_v2.V[i][1]==0 && gpu_v2.V[i][2]==0)
                        { has_C0v = true; break; }
                // Cv1: origin on edge (antiparallel pair)
                if (!has_C0v) {
                    for (int i = 0; i < 4 && !has_C1v; i++)
                        for (int j = i+1; j < 4 && !has_C1v; j++) {
                            int64_t cx = (int64_t)gpu_v2.V[i][1]*gpu_v2.V[j][2]
                                       - (int64_t)gpu_v2.V[i][2]*gpu_v2.V[j][1];
                            int64_t cy = (int64_t)gpu_v2.V[i][2]*gpu_v2.V[j][0]
                                       - (int64_t)gpu_v2.V[i][0]*gpu_v2.V[j][2];
                            int64_t cz = (int64_t)gpu_v2.V[i][0]*gpu_v2.V[j][1]
                                       - (int64_t)gpu_v2.V[i][1]*gpu_v2.V[j][0];
                            if (cx==0 && cy==0 && cz==0) {
                                int64_t dot = (int64_t)gpu_v2.V[i][0]*gpu_v2.V[j][0]
                                            + (int64_t)gpu_v2.V[i][1]*gpu_v2.V[j][1]
                                            + (int64_t)gpu_v2.V[i][2]*gpu_v2.V[j][2];
                                if (dot < 0) has_C1v = true;
                            }
                        }
                }
            }
        }
        if (!has_Cw) {
            has_Cw = check_field_zero_coplanar(gpu_v2.W);
            cc.has_Cw_pos = has_Cw;
            if (has_Cw) {
                for (int i = 0; i < 4; i++)
                    if (gpu_v2.W[i][0]==0 && gpu_v2.W[i][1]==0 && gpu_v2.W[i][2]==0)
                        { has_C0w = true; break; }
                if (!has_C0w) {
                    for (int i = 0; i < 4 && !has_C1w; i++)
                        for (int j = i+1; j < 4 && !has_C1w; j++) {
                            int64_t cx = (int64_t)gpu_v2.W[i][1]*gpu_v2.W[j][2]
                                       - (int64_t)gpu_v2.W[i][2]*gpu_v2.W[j][1];
                            int64_t cy = (int64_t)gpu_v2.W[i][2]*gpu_v2.W[j][0]
                                       - (int64_t)gpu_v2.W[i][0]*gpu_v2.W[j][2];
                            int64_t cz = (int64_t)gpu_v2.W[i][0]*gpu_v2.W[j][1]
                                       - (int64_t)gpu_v2.W[i][1]*gpu_v2.W[j][0];
                            if (cx==0 && cy==0 && cz==0) {
                                int64_t dot = (int64_t)gpu_v2.W[i][0]*gpu_v2.W[j][0]
                                            + (int64_t)gpu_v2.W[i][1]*gpu_v2.W[j][1]
                                            + (int64_t)gpu_v2.W[i][2]*gpu_v2.W[j][2];
                                if (dot < 0) has_C1w = true;
                            }
                        }
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
                int iv_idx = 0;
                if (cc.n_Q_roots > 0 && degQ >= 1) {
                    __int128 f_x[2] = {0, 1};
                    int signs[3] = {};
                    int nr = ftk2::signs_at_roots_i128(Q_i128, degQ, f_x, 1, signs, 3);
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
    else if (has_D23)      tags.push_back("D23");
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

} // namespace ftk2

#endif // FTK2_PV_TET_CLASSIFY_HPP
