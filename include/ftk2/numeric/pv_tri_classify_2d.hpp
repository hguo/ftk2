// Pure-integer PV triangle classifier for 2D vector fields
// All topological decisions use __int128. NO float/double anywhere.
#ifndef FTK2_PV_TRI_CLASSIFY_2D_HPP
#define FTK2_PV_TRI_CLASSIFY_2D_HPP

#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <cstdio>
#include <cstdint>

namespace ftk2 {

// ─── GPU output struct for 2D ExactPV2 ──────────────────────────────────────
struct TriCaseV2GPU {
    int V[3][2], W[3][2];
    ExactPV2Result2D v2;
    int disc_sign[3];   // discriminant sign of P[k] for each edge
    uint64_t seed;
};

// ─── Classified case (2D) ──────────────────────────────────────────────────
struct ClassifiedCase2D {
    int V[3][2], W[3][2];
    uint64_t seed;
    int total_punctures;
    std::string category;

    // Integer polynomial coefficients
    __int128 Q_i128[3];
    __int128 P_i128[3][3];
    int Q_disc_sign;
    int n_Q_roots;

    // v2 solver output
    bool merge_infinity;
    int n_qr_roots;
    __int128 h[3];
    int h_deg, h_n_roots;
    __int128 P_red[3][3];
    int degP_red[3];
    int n_distinct_red[3];

    // SR
    bool has_shared_root;
    bool has_non_isolated_sr;
    int n_sr_roots;
    int sr_q_root_idx[2];

    // TN
    int n_tn;
    struct TNInfo {
        int face;
        __int128 h_tn[2];   // linear factor
    } tn_points[3];

    // Cv/Cw — integer check for field=0 in triangle
    bool has_Cv;
    bool has_Cw;
    int Cv_type;   // 0=vertex, 1=edge, 2=interior
    int Cw_type;

    bool has_B;
    bool has_vertex_d00;  // det(V[i],W[i])=0 at any vertex
    int d11_edge;   // edge with D11 degeneracy (-1 if none)

    struct PunctureInfo {
        int face;
        int root_idx;
        int q_interval;
        bool is_edge;
        bool is_vertex;
        bool is_passthrough;
        bool is_D00, is_D01;
        int tri_vertex;
        int edge_faces[2];
        int interval_idx;
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

// Check if field=0 is inside triangle interior (Cv/Cw in 2D).
// For 2D integer field F[3][2] on a triangle with vertices 0,1,2:
//   F(x) = μ₀·F[0] + μ₁·F[1] + μ₂·F[2] = 0
//   With μ₀+μ₁+μ₂=1, solve 2×2 system:
//     (F[0]-F[2])·μ₀ + (F[1]-F[2])·μ₁ = -F[2]
// Uses exact Cramer's rule (int64_t sufficient for small inputs).
// Returns 0=not found, 1=interior, 2=edge(Cv1), 3=vertex(Cv0).
inline int check_field_zero_in_tri_2d(const int F[3][2]) {
    // Check for vertex zeros first (Cv0/Cw0)
    for (int i = 0; i < 3; i++)
        if (F[i][0] == 0 && F[i][1] == 0)
            return 3;  // vertex

    // 2×2 system: A·[μ₀,μ₁]ᵀ = b
    int64_t a00 = (int64_t)F[0][0] - F[2][0];
    int64_t a01 = (int64_t)F[1][0] - F[2][0];
    int64_t a10 = (int64_t)F[0][1] - F[2][1];
    int64_t a11 = (int64_t)F[1][1] - F[2][1];
    int64_t b0 = -(int64_t)F[2][0];
    int64_t b1 = -(int64_t)F[2][1];

    int64_t det = a00*a11 - a01*a10;
    if (det == 0) {
        // Collinear field vectors — check if origin in convex hull (1D)
        // Project all F[i] onto any non-zero F[j]
        int ui = -1;
        for (int i = 0; i < 3; i++)
            if (F[i][0] != 0 || F[i][1] != 0) { ui = i; break; }
        if (ui < 0) return 1;  // all zero → origin trivially inside

        bool has_pos = false, has_neg = false;
        for (int i = 0; i < 3; i++) {
            int64_t dot = (int64_t)F[i][0]*F[ui][0] + (int64_t)F[i][1]*F[ui][1];
            if (dot > 0) has_pos = true;
            if (dot < 0) has_neg = true;
        }
        if (has_pos && has_neg) return 2;  // on edge (antiparallel pair)
        return 0;  // not inside
    }

    int64_t n0 = b0*a11 - a01*b1;
    int64_t n1 = a00*b1 - b0*a10;
    int64_t n2 = det - n0 - n1;

    // Interior: all bary coords strictly positive (same sign as det)
    if (det > 0) {
        if (n0 > 0 && n1 > 0 && n2 > 0) return 1;  // interior
        if (n0 >= 0 && n1 >= 0 && n2 >= 0) {
            // On boundary
            int n_zero = (n0 == 0 ? 1 : 0) + (n1 == 0 ? 1 : 0) + (n2 == 0 ? 1 : 0);
            if (n_zero >= 2) return 3;  // vertex
            if (n_zero == 1) return 2;  // edge
        }
    } else {
        if (n0 < 0 && n1 < 0 && n2 < 0) return 1;
        if (n0 <= 0 && n1 <= 0 && n2 <= 0) {
            int n_zero = (n0 == 0 ? 1 : 0) + (n1 == 0 ? 1 : 0) + (n2 == 0 ? 1 : 0);
            if (n_zero >= 2) return 3;
            if (n_zero == 1) return 2;
        }
    }
    return 0;
}

// ─── classify_case_v2_2d: Pure-integer classification from ExactPV2Result2D ──
inline ClassifiedCase2D classify_case_v2_2d(const TriCaseV2GPU& gpu_v2) {
    ClassifiedCase2D cc;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            cc.V[i][j] = gpu_v2.V[i][j];
            cc.W[i][j] = gpu_v2.W[i][j];
        }
    cc.total_punctures = gpu_v2.v2.n_punctures;
    cc.seed = gpu_v2.seed;
    cc.has_shared_root = false;
    cc.has_non_isolated_sr = false;
    cc.n_sr_roots = 0;
    cc.has_B = false;
    cc.has_vertex_d00 = false;
    cc.d11_edge = -1;
    cc.has_Cv = false;
    cc.has_Cw = false;
    cc.Cv_type = -1;
    cc.Cw_type = -1;
    cc.n_tn = 0;
    cc.n_deduplicated = 0;

    // ─── Step 0: Integer Q, P polynomials ────────────────────────────
    __int128 V128[3][2], W128[3][2];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) {
            V128[i][j] = (__int128)gpu_v2.V[i][j];
            W128[i][j] = (__int128)gpu_v2.W[i][j];
        }
    compute_tri_QP_2d(V128, W128, cc.Q_i128, cc.P_i128);

    int degQ = effective_degree_i128(cc.Q_i128, 2);

    // Q discriminant (degree 2 only: b²-4ac)
    if (degQ == 2) {
        __int128 disc2 = cc.Q_i128[1]*cc.Q_i128[1]
                       - (__int128)4*cc.Q_i128[0]*cc.Q_i128[2];
        cc.Q_disc_sign = (disc2 > 0) ? 1 : (disc2 < 0) ? -1 : 0;
        if (disc2 > 0) cc.n_Q_roots = 2;
        else if (disc2 == 0) cc.n_Q_roots = 1;
        else cc.n_Q_roots = 0;
    } else if (degQ == 1) {
        cc.Q_disc_sign = 0;
        cc.n_Q_roots = 1;
    } else {
        cc.Q_disc_sign = 0;
        cc.n_Q_roots = 0;
    }

    // Store v2 solver infrastructure
    const ExactPV2Result2D& v2 = gpu_v2.v2;
    cc.merge_infinity = v2.merge_infinity;
    cc.n_qr_roots = v2.n_qr_roots;
    for (int i = 0; i < 3; i++) cc.h[i] = v2.h[i];
    cc.h_deg = v2.h_deg;
    cc.h_n_roots = v2.h_n_roots;
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < 3; i++) cc.P_red[k][i] = v2.P_red[k][i];
        cc.degP_red[k] = v2.degP_red[k];
        cc.n_distinct_red[k] = v2.n_distinct_red[k];
    }

    // ─── Step 1: Map v2 punctures → PunctureInfo ─────────────────────
    for (int pi = 0; pi < v2.n_punctures; pi++) {
        const auto& vp = v2.punctures[pi];
        ClassifiedCase2D::PunctureInfo ci;
        ci.face = vp.face;
        ci.root_idx = vp.root_idx;
        ci.q_interval = vp.q_interval;
        ci.is_edge = vp.is_edge;
        ci.is_vertex = vp.is_vertex;
        ci.is_passthrough = false;
        ci.is_D00 = false;
        ci.is_D01 = false;
        ci.interval_idx = -1;
        ci.tri_vertex = -1;
        ci.edge_faces[0] = vp.edge_faces[0];
        ci.edge_faces[1] = vp.edge_faces[1];

        // In 2D: edge puncture (shared by 2 faces) = mesh vertex
        // Vertex opposite face k is vertex k.  Edge shared by faces i,j
        // means mesh vertex = {0,1,2}\{i,j}
        if (ci.is_edge || ci.is_vertex) {
            int ef0 = vp.edge_faces[0], ef1 = vp.edge_faces[1];
            for (int v = 0; v < 3; v++)
                if (v != ef0 && v != ef1)
                    ci.tri_vertex = v;
            ci.is_D00 = true;
        }

        cc.punctures.push_back(ci);
    }

    // ─── Step 2: Q-type string ───────────────────────────────────────
    std::string q_type;
    if (degQ == 0 && cc.Q_i128[0] == 0) q_type = "Qz";
    else if (degQ == 0) q_type = "Q0";
    else if (degQ == 1) q_type = "Q1";
    else {
        __int128 disc2 = cc.Q_i128[1]*cc.Q_i128[1]
                       - (__int128)4*cc.Q_i128[0]*cc.Q_i128[2];
        if (disc2 > 0) q_type = "Q2+";
        else if (disc2 < 0) q_type = "Q2-";
        else q_type = "Q2o";
    }

    // ─── Step 3: Build intervals ─────────────────────────────────────
    if (cc.n_Q_roots > 0) {
        cc.intervals.push_back({0, true});
        for (int i = 0; i + 1 < cc.n_Q_roots; i++)
            cc.intervals.push_back({0, false});
        cc.intervals.push_back({0, true});
    } else {
        cc.intervals.push_back({0, true});
    }

    // ─── Step 4: SR detection + interval assignment ──────────────────
    // Detect SR, then suppress if L'Hôpital bary is outside triangle.
    if (degQ > 0) {
        int degQ_i = effective_degree_i128(cc.Q_i128, 2);
        for (int k = 0; k < 3; k++) {
            int degPk = effective_degree_i128(cc.P_i128[k], 2);
            if (degPk <= 0) continue;
            if (resultant_sign_i128(cc.Q_i128, degQ_i, cc.P_i128[k], degPk) == 0) {
                cc.has_shared_root = true;
                break;
            }
        }
    }
    // Suppress SR if ALL shared roots have L'Hôpital bary outside triangle.
    // For each face k with resultant(Q,P[k])=0, compute g=gcd(Q,P[k]).
    // If g is linear, check bary at g's root. Pure integer.
    if (cc.has_shared_root) {
        int degQ_i = effective_degree_i128(cc.Q_i128, 2);
        bool any_inside = false;
        for (int k = 0; k < 3; k++) {
            int degPk = effective_degree_i128(cc.P_i128[k], 2);
            if (degPk <= 0) continue;
            if (resultant_sign_i128(cc.Q_i128, degQ_i, cc.P_i128[k], degPk) != 0)
                continue;
            // Shared root on face k. Compute gcd(Q, P[k]).
            __int128 g[3] = {};
            int dg = poly_gcd_full_i128(cc.Q_i128, degQ_i, cc.P_i128[k], degPk, g);
            if (dg == 1) {
                __int128 g0 = g[0], g1 = g[1];
                // L'Hôpital at root -g0/g1:
                // denom = g1·Q[1] - 2·Q[2]·g0
                __int128 denom = g1 * cc.Q_i128[1] - (__int128)2 * cc.Q_i128[2] * g0;
                if (denom != 0) {
                    bool inside = true;
                    for (int j = 0; j < 3; j++) {
                        __int128 mu_j = g1 * cc.P_i128[j][1]
                                      - (__int128)2 * cc.P_i128[j][2] * g0;
                        if (mu_j * denom < 0) { inside = false; break; }
                    }
                    if (inside) any_inside = true;
                }
            } else if (dg >= 2) {
                any_inside = true; // conservatively keep SR for higher-degree GCD
            }
        }
        // Also check h roots (gcd of all P's)
        if (v2.h_deg == 1) {
            __int128 h0 = v2.h[0], h1 = v2.h[1];
            __int128 denom = h1 * cc.Q_i128[1] - (__int128)2 * cc.Q_i128[2] * h0;
            if (denom != 0) {
                bool inside = true;
                for (int j = 0; j < 3; j++) {
                    __int128 mu_j = h1 * cc.P_i128[j][1]
                                  - (__int128)2 * cc.P_i128[j][2] * h0;
                    if (mu_j * denom < 0) { inside = false; break; }
                }
                if (inside) any_inside = true;
            }
        } else if (v2.h_deg >= 2) {
            any_inside = true;
        }
        if (!any_inside) cc.has_shared_root = false;
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
                    int cmp = compare_roots_i128(
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

    // ─── Step 5: Pairs ──────────────────────────────────────────────
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

        // SR root indices
        std::set<int> sr_qr_set;
        if (v2.h_n_roots > 0) {
            __int128 Q_red_i128[3] = {};
            int degQ_red = effective_degree_i128(cc.Q_i128, 2);
            if (v2.h_deg >= 1) {
                __int128 qd[3] = {};
                degQ_red = poly_exact_div_i128(
                    cc.Q_i128, effective_degree_i128(cc.Q_i128, 2),
                    v2.h, v2.h_deg, qd);
                for (int i = 0; i < 3; i++) Q_red_i128[i] = (i <= degQ_red) ? qd[i] : 0;
            } else {
                for (int i = 0; i < 3; i++) Q_red_i128[i] = cc.Q_i128[i];
            }
            for (int hi = 0; hi < v2.h_n_roots; hi++) {
                int qr_below = 0;
                for (int qi = 0; qi < v2.n_qr_roots; qi++) {
                    int cmp = compare_roots_i128(
                        v2.h, v2.h_deg, v2.h_n_roots, hi,
                        Q_red_i128, degQ_red, v2.n_qr_roots, qi);
                    if (cmp > 0) qr_below++;
                }
                sr_qr_set.insert(qr_below + hi);
            }
        }
        cc.n_sr_roots = 0;
        for (int idx : sr_qr_set)
            if (cc.n_sr_roots < 2)
                cc.sr_q_root_idx[cc.n_sr_roots++] = idx;
    }

    // ─── Step 6: D11 detection (PV curve on edge) ────────────────────
    for (int k = 0; k < 3; k++) {
        if (cc.P_i128[k][0] == 0 && cc.P_i128[k][1] == 0 && cc.P_i128[k][2] == 0) {
            cc.d11_edge = k;
            break;
        }
    }

    // ─── Step 7: D00 detection (vertex puncture + vertex det=0) ──────
    for (int pi = 0; pi < (int)cc.punctures.size(); pi++) {
        auto& punc = cc.punctures[pi];
        if (punc.is_edge) {
            punc.is_D00 = true;
        }
    }
    // Also detect D00 from vertex field degeneracy: det(V[i], W[i]) = 0
    cc.has_vertex_d00 = false;
    for (int i = 0; i < 3; i++) {
        int64_t det = (int64_t)gpu_v2.V[i][0] * gpu_v2.W[i][1]
                    - (int64_t)gpu_v2.V[i][1] * gpu_v2.W[i][0];
        if (det == 0) cc.has_vertex_d00 = true;
    }

    // ─── Step 8: TN detection (with inside-triangle check) ──────────
    for (int k = 0; k < 3; k++) {
        int degPk = effective_degree_i128(cc.P_i128[k], 2);
        if (degPk < 2) continue;
        __int128 a = cc.P_i128[k][2], b = cc.P_i128[k][1];
        __int128 disc_pk = b*b - (__int128)4*cc.P_i128[k][0]*a;
        if (disc_pk != 0) continue;
        // Double root → TN candidate at λ = -b/(2a)
        // Check bary coords at tangency point: mu_j = P[j](-b/(2a)) / Q(-b/(2a))
        // Multiply through by (2a)²: q4a2 = 4a²Q[0] - 2abQ[1] + b²Q[2]
        //                             mu_j_num = 4a²P[j][0] - 2abP[j][1] + b²P[j][2]
        __int128 a2_4 = (__int128)4*a*a;
        __int128 ab_2 = (__int128)2*a*b;
        __int128 b2 = b*b;
        __int128 q4a2 = a2_4*cc.Q_i128[0] - ab_2*cc.Q_i128[1] + b2*cc.Q_i128[2];
        if (q4a2 == 0) {
            // SR+TN: Q also zero at tangency → skip (handled by SR logic)
            continue;
        }
        bool inside = true;
        int n_zero = 0;
        for (int j = 0; j < 3; j++) {
            if (j == k) continue;  // P[k] is zero at its own double root
            __int128 mu_j = a2_4*cc.P_i128[j][0] - ab_2*cc.P_i128[j][1]
                          + b2*cc.P_i128[j][2];
            __int128 prod = mu_j * q4a2;
            if (prod < 0) { inside = false; break; }
            if (mu_j == 0) n_zero++;
        }
        if (!inside) continue;
        // mu[k] is always 0 (P[k] at its own double root), so total zeros = n_zero + 1
        if (n_zero + 1 >= 2) continue;  // at vertex → D00, skip TN

        if (cc.n_tn < 3) {
            cc.tn_points[cc.n_tn].face = k;
            cc.tn_points[cc.n_tn].h_tn[0] = -b;
            cc.tn_points[cc.n_tn].h_tn[1] = (__int128)2*a;
            cc.n_tn++;
        }
    }

    // ─── Step 9: Cv/Cw detection (PV-curve limit) ──────────────────
    // Cv: bary = P[k](0)/Q(0).  L'Hôpital when Q(0)=0.
    // Only tag if bary is inside triangle (all same sign as denom).
    {
        __int128 denom = cc.Q_i128[0];
        __int128 mu[3];
        if (denom == 0) {
            denom = cc.Q_i128[1];
            for (int k = 0; k < 3; k++) mu[k] = cc.P_i128[k][1];
        } else {
            for (int k = 0; k < 3; k++) mu[k] = cc.P_i128[k][0];
        }
        if (denom != 0) {
            bool inside = true;
            int n_zero = 0;
            for (int k = 0; k < 3; k++) {
                if (mu[k] * denom < 0) { inside = false; break; }
                if (mu[k] == 0) n_zero++;
            }
            if (inside) {
                cc.has_Cv = true;
                if (n_zero >= 2) cc.Cv_type = 3;
                else if (n_zero == 1) cc.Cv_type = 2;
                else cc.Cv_type = 1;
            }
        } else {
            // Q≡0 (Qz): fall back to geometric check
            int cv_res = check_field_zero_in_tri_2d(gpu_v2.V);
            if (cv_res > 0) { cc.has_Cv = true; cc.Cv_type = cv_res; }
        }
    }
    // Cw: PV-curve limit — bary = P[k][d]/Q[d] where d=deg(Q).
    // Geometric check gives false positives (W=0 but PV curve diverges).
    {
        int dQ = effective_degree_i128(cc.Q_i128, 2);
        bool cw_valid = (dQ > 0);
        if (cw_valid) {
            for (int k = 0; k < 3; k++) {
                int dPk = effective_degree_i128(cc.P_i128[k], 2);
                if (dPk > dQ) { cw_valid = false; break; }
            }
        }
        if (cw_valid) {
            __int128 denom = cc.Q_i128[dQ];
            bool inside = true;
            int n_zero = 0;
            for (int k = 0; k < 3; k++) {
                __int128 mu_k = (effective_degree_i128(cc.P_i128[k], 2) == dQ)
                                ? cc.P_i128[k][dQ] : (__int128)0;
                __int128 prod = mu_k * denom;
                if (prod < 0) { inside = false; break; }
                if (mu_k == 0) n_zero++;
            }
            if (inside) {
                cc.has_Cw = true;
                if (n_zero >= 2) cc.Cw_type = 3;       // vertex (Cw0)
                else if (n_zero == 1) cc.Cw_type = 2;   // edge (Cw1)
                else cc.Cw_type = 1;                     // interior (Cw)
            }
        }
    }

    // ─── Step 10: Bubble detection (B) ───────────────────────────────
    // T0, no Q roots, degQ>0, all bary coords positive everywhere → closed curve
    if (v2.n_punctures == 0 && cc.n_Q_roots == 0 && degQ > 0) {
        bool bounded = true;
        for (int k = 0; k < 3; k++) {
            if (effective_degree_i128(cc.P_i128[k], 2) > degQ) {
                bounded = false;
                break;
            }
        }
        if (bounded) {
            // Check at λ=0: P[k][0] * Q[0] > 0 for all k
            bool at_zero = true;
            for (int k = 0; k < 3; k++) {
                if (cc.P_i128[k][0] * cc.Q_i128[0] <= 0) {
                    at_zero = false;
                    break;
                }
            }
            // Check leading: P[k][degQ] * Q[degQ] > 0 for all k
            bool at_inf = true;
            for (int k = 0; k < 3; k++) {
                __int128 pk_lead = (effective_degree_i128(cc.P_i128[k], 2) == degQ)
                                   ? cc.P_i128[k][degQ] : (__int128)0;
                if (pk_lead * cc.Q_i128[degQ] <= 0) {
                    at_inf = false;
                    break;
                }
            }
            cc.has_B = at_zero && at_inf;
        }
    }

    // ─── Step 11: ISR detection ──────────────────────────────────────
    if (cc.has_shared_root && v2.passthrough_deg >= 2) {
        cc.has_non_isolated_sr = true;
    }

    // ─── Build category string ───────────────────────────────────────
    int T_count = v2.n_punctures;
    cc.n_deduplicated = T_count;

    // Build interval tuple for multi-interval cases
    std::vector<int> interval_counts;
    for (const auto& iv : cc.intervals)
        if (iv.n_pv > 0) interval_counts.push_back(iv.n_pv);
    std::sort(interval_counts.begin(), interval_counts.end());

    std::string cat = "T" + std::to_string(T_count);

    // Add interval tuple if multi-interval
    if (interval_counts.size() > 1) {
        cat += "_(";
        for (int i = 0; i < (int)interval_counts.size(); i++) {
            if (i > 0) cat += ",";
            cat += std::to_string(interval_counts[i]);
        }
        cat += ")";
    }

    cat += "_" + q_type;

    // Degeneracy tags (ordered: SR, ISR, Cv, Cw, Dmd, B, TN)
    if (cc.has_shared_root && !cc.has_non_isolated_sr) cat += "_SR";
    if (cc.has_non_isolated_sr) cat += "_ISR";

    if (cc.has_Cv) {
        if (cc.Cv_type == 3) cat += "_Cv0";
        else if (cc.Cv_type == 2) cat += "_Cv1";
        else cat += "_Cv";
    }
    if (cc.has_Cw) {
        if (cc.Cw_type == 3) cat += "_Cw0";
        else if (cc.Cw_type == 2) cat += "_Cw1";
        else cat += "_Cw";
    }

    // D00 (vertex puncture OR vertex det(V,W)=0)
    bool has_d00 = cc.has_vertex_d00;
    for (const auto& p : cc.punctures)
        if (p.is_D00) has_d00 = true;
    if (has_d00) cat += "_D00";

    // D11 (PV curve on edge)
    if (cc.d11_edge >= 0) cat += "_D11";

    // D22 (entire triangle is PV: Q≡0 and all P≡0)
    if (q_type == "Qz") {
        bool all_p_zero = true;
        for (int k = 0; k < 3; k++)
            if (cc.P_i128[k][0] != 0 || cc.P_i128[k][1] != 0 || cc.P_i128[k][2] != 0)
                all_p_zero = false;
        if (all_p_zero) cat += "_D22";
    }

    if (cc.has_B) cat += "_B";
    if (cc.n_tn > 0) cat += "_TN";

    cc.category = cat;
    return cc;
}

// i128_to_string is defined in pv_tet_classify.hpp; provide inline version here
// in case the 2D classifier is used standalone.
#ifndef FTK2_PV_TET_CLASSIFY_HPP
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
#endif

// JSON output helper for 2D classified case
inline void print_json_2d(FILE* f, const ClassifiedCase2D& cc) {
    fprintf(f, "{\"seed\":%lu,\"category\":\"%s\",\"T\":%d",
            (unsigned long)cc.seed, cc.category.c_str(), cc.total_punctures);

    fprintf(f, ",\"V\":[[%d,%d],[%d,%d],[%d,%d]]",
            cc.V[0][0],cc.V[0][1], cc.V[1][0],cc.V[1][1], cc.V[2][0],cc.V[2][1]);
    fprintf(f, ",\"W\":[[%d,%d],[%d,%d],[%d,%d]]",
            cc.W[0][0],cc.W[0][1], cc.W[1][0],cc.W[1][1], cc.W[2][0],cc.W[2][1]);

    // Q polynomial
    fprintf(f, ",\"Q\":[");
    for (int i = 0; i < 3; i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "\"%s\"", i128_to_string(cc.Q_i128[i]).c_str());
    }
    fprintf(f, "]");

    // P polynomials
    fprintf(f, ",\"P\":[");
    for (int k = 0; k < 3; k++) {
        if (k > 0) fprintf(f, ",");
        fprintf(f, "[");
        for (int i = 0; i < 3; i++) {
            if (i > 0) fprintf(f, ",");
            fprintf(f, "\"%s\"", i128_to_string(cc.P_i128[k][i]).c_str());
        }
        fprintf(f, "]");
    }
    fprintf(f, "]");

    fprintf(f, ",\"n_pairs\":%d", (int)cc.pairs.size());
    fprintf(f, ",\"n_Q_roots\":%d", cc.n_Q_roots);
    fprintf(f, ",\"merge_infinity\":%s", cc.merge_infinity ? "true" : "false");

    if (cc.has_shared_root) fprintf(f, ",\"SR\":true");
    if (cc.has_non_isolated_sr) fprintf(f, ",\"ISR\":true");
    if (cc.has_Cv) fprintf(f, ",\"Cv\":%d", cc.Cv_type);
    if (cc.has_Cw) fprintf(f, ",\"Cw\":%d", cc.Cw_type);
    if (cc.n_tn > 0) fprintf(f, ",\"TN\":%d", cc.n_tn);
    if (cc.has_B) fprintf(f, ",\"B\":true");
    if (cc.d11_edge >= 0) fprintf(f, ",\"D11\":%d", cc.d11_edge);

    // Puncture details
    fprintf(f, ",\"punctures\":[");
    for (int i = 0; i < (int)cc.punctures.size(); i++) {
        if (i > 0) fprintf(f, ",");
        const auto& p = cc.punctures[i];
        fprintf(f, "{\"face\":%d,\"root_idx\":%d,\"q_interval\":%d",
                p.face, p.root_idx, p.q_interval);
        if (p.is_edge) fprintf(f, ",\"is_edge\":true");
        if (p.is_D00) fprintf(f, ",\"is_D00\":true");
        fprintf(f, "}");
    }
    fprintf(f, "]");

    // Pairs
    fprintf(f, ",\"pairs\":[");
    for (int i = 0; i < (int)cc.pairs.size(); i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "[%d,%d]", cc.pairs[i].pi_a, cc.pairs[i].pi_b);
    }
    fprintf(f, "]");

    fprintf(f, "}\n");
}

} // namespace ftk2

#endif // FTK2_PV_TRI_CLASSIFY_2D_HPP
