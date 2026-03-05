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

// ─── GPU output struct ──────────────────────────────────────────────────────
struct TetCaseGPU {
    int V[4][3], W[4][3];
    PunctureResult face[4];
    int total_punctures;
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

// ─── Extraction kernel ──────────────────────────────────────────────────────
// One thread per random tet. Each thread:
//   1. Generate random V[4][3], W[4][3] from LCG
//   2. Solve PV on 4 faces
//   3. If total punctures >= min_punctures, write to output
__global__ void tet_case_finder_kernel(
    TetCaseGPU* output,
    int*        output_count,
    int         max_output,
    int         min_punctures,
    int         R,
    uint64_t    base_seed,
    uint64_t    batch_offset)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t global_id = batch_offset + tid;

    // Seed LCG from global thread id + base seed
    uint32_t state = (uint32_t)(global_id ^ (base_seed * 2654435761ULL));
    // Warm up LCG
    for (int i = 0; i < 4; i++) lcg_next(state);

    // Generate random integer fields
    int V[4][3], W[4][3];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            V[i][j] = rand_int_dev(state, R);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            W[i][j] = rand_int_dev(state, R);

    // Solve PV on each face
    PunctureResult faces[4];
    int total = 0;

    for (int fi = 0; fi < 4; fi++) {
        double Vf[3][3], Wf[3][3];
        for (int vi = 0; vi < 3; vi++) {
            int src = d_face_verts[fi][vi];
            for (int c = 0; c < 3; c++) {
                Vf[vi][c] = (double)V[src][c];
                Wf[vi][c] = (double)W[src][c];
            }
        }

        // Use dummy indices for SoS: face vertex global IDs
        // (we use small unique values since these are standalone tets)
        uint64_t indices[3];
        for (int vi = 0; vi < 3; vi++)
            indices[vi] = (uint64_t)d_face_verts[fi][vi];

        faces[fi] = solve_pv_triangle_device(Vf, Wf, indices);

        if (faces[fi].count > 0 && faces[fi].count < INT_MAX)
            total += faces[fi].count;
    }

    // Filter: only keep cases with enough punctures
    if (total >= min_punctures) {
        int idx = atomicAdd(output_count, 1);
        if (idx < max_output) {
            TetCaseGPU& out = output[idx];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++) {
                    out.V[i][j] = V[i][j];
                    out.W[i][j] = W[i][j];
                }
            for (int fi = 0; fi < 4; fi++)
                out.face[fi] = faces[fi];
            out.total_punctures = total;
            out.seed = global_id;
        }
    }
}

// ─── CPU classification ─────────────────────────────────────────────────────

struct ClassifiedCase {
    TetCaseGPU gpu;
    std::string category;      // e.g. "T4a_Q3+"
    double Q_coeffs[4];
    int Q_disc_sign;
    int n_Q_roots;
    double Q_roots[3];
    double P_coeffs[4][4];
    bool has_shared_root;
    bool has_B;                // bubble (closed loop inside tet, T0 only)

    struct PunctureInfo {
        int face;
        double lambda;
        double bary[3];
        bool is_edge;          // on a tet edge (1 face-bary ≈ 0)
        bool is_vertex;        // on a tet vertex (2 face-bary ≈ 0)
        int tet_edge[2];       // shared tet edge vertices (if is_edge)
        int tet_vertex;        // shared tet vertex (if is_vertex)
        int interval_idx;      // Q-interval this puncture belongs to (-1 if unassigned)
    };
    std::vector<PunctureInfo> punctures;
    int n_deduplicated;        // number removed by edge/vertex dedup

    struct IntervalInfo {
        double lb, ub;
        int n_pv;
        bool is_infinity;      // spans lambda -> ±∞
    };
    std::vector<IntervalInfo> intervals;

    struct PuncturePair {
        int pi_a, pi_b;       // indices into punctures[]
        bool is_cross;         // true if pair spans through infinity (Cw)
        int interval_idx;      // Q-interval this pair belongs to (-1 for SR pass-through)
    };
    std::vector<PuncturePair> pairs;
};

// Check if field=0 is inside tet interior (critical point)
static bool check_field_zero_in_tet(const int F[4][3]) {
    // Solve: sum_i mu_i * F_i = 0, sum mu_i = 1
    // => (F_0-F_3)*mu_0 + (F_1-F_3)*mu_1 + (F_2-F_3)*mu_2 = -F_3
    double A[3][3], b[3];
    for (int c = 0; c < 3; c++) {
        A[c][0] = F[0][c] - F[3][c];
        A[c][1] = F[1][c] - F[3][c];
        A[c][2] = F[2][c] - F[3][c];
        b[c] = -F[3][c];
    }
    double det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
               - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if (std::abs(det) < 1e-15) return false;
    double inv = 1.0 / det;
    double mu[4];
    mu[0] = inv * (b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                - A[0][1]*(b[1]*A[2][2]-A[1][2]*b[2])
                + A[0][2]*(b[1]*A[2][1]-A[1][1]*b[2]));
    mu[1] = inv * (A[0][0]*(b[1]*A[2][2]-A[1][2]*b[2])
                - b[0]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                + A[0][2]*(A[1][0]*b[2]-b[1]*A[2][0]));
    mu[2] = inv * (A[0][0]*(A[1][1]*b[2]-b[1]*A[2][1])
                - A[0][1]*(A[1][0]*b[2]-b[1]*A[2][0])
                + b[0]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]));
    mu[3] = 1.0 - mu[0] - mu[1] - mu[2];
    const double eps = 1e-6;
    for (int i = 0; i < 4; i++)
        if (mu[i] < -eps || mu[i] > 1.0 + eps) return false;
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
        // Build Sylvester matrix as double (coefficients fit in double for R<=20)
        double S[6][6] = {};
        for (int i = 0; i < degP; i++)
            for (int j = 0; j <= degQ; j++)
                S[i][i + degQ - j] = (double)Q_i128[j];
        for (int i = 0; i < degQ; i++)
            for (int j = 0; j <= degP; j++)
                S[degP + i][i + degP - j] = (double)P_i128[k][j];

        // Compute determinant via LU decomposition
        // For N<=6, use cofactor expansion or direct Gaussian elimination
        double mat[6][6];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                mat[i][j] = S[i][j];

        double det = 1.0;
        for (int col = 0; col < N; col++) {
            // Partial pivoting
            int pivot = col;
            for (int row = col + 1; row < N; row++)
                if (std::abs(mat[row][col]) > std::abs(mat[pivot][col]))
                    pivot = row;
            if (pivot != col) {
                for (int j = 0; j < N; j++)
                    std::swap(mat[col][j], mat[pivot][j]);
                det = -det;
            }
            if (std::abs(mat[col][col]) < 1e-30) { det = 0; break; }
            det *= mat[col][col];
            double inv = 1.0 / mat[col][col];
            for (int row = col + 1; row < N; row++) {
                double f = mat[row][col] * inv;
                for (int j = col; j < N; j++)
                    mat[row][j] -= f * mat[col][j];
            }
        }

        // The resultant is an integer (Q, P have integer coeffs).
        // If |det| < 0.5, it's exactly 0.
        if (std::abs(det) < 0.5) return true;
    }
    return false;
}

static ClassifiedCase classify_case(const TetCaseGPU& gpu) {
    ClassifiedCase cc;
    cc.gpu = gpu;
    cc.has_shared_root = false;
    cc.has_B = false;

    // Convert to double arrays
    double Vd[4][3], Wd[4][3];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            Vd[i][j] = (double)gpu.V[i][j];
            Wd[i][j] = (double)gpu.W[i][j];
        }

    // Compute Q, P polynomials (float)
    double Q[4], P[4][4];
    characteristic_polynomials_pv_tetrahedron(Vd, Wd, Q, P);
    for (int i = 0; i < 4; i++) cc.Q_coeffs[i] = Q[i];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cc.P_coeffs[i][j] = P[i][j];

    // Compute exact Q, P via __int128
    __int128 Q_i128[4], P_i128[4][4];
    compute_tet_QP_i128(Vd, Wd, Q_i128, P_i128);

    // Discriminant sign
    cc.Q_disc_sign = discriminant_sign_i128(Q_i128);

    // Find Q roots
    cc.n_Q_roots = solve_cubic_real(Q, cc.Q_roots);

    // Sort Q roots
    std::sort(cc.Q_roots, cc.Q_roots + cc.n_Q_roots);

    // Face vertex ordering (CPU side): face i = triangle opposite vertex i
    static const int fv[4][3] = {
        {1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}
    };

    // Re-solve each face on CPU using the exact solver with SoS.
    // The SoS ownership rule (based on global vertex indices) ensures that
    // edge/vertex punctures are assigned to exactly one face — no threshold-
    // based deduplication needed.
    //
    // The `indices` array passed to solve_pv_triangle contains the global
    // (tet-level) vertex indices for this face.  The SoS rule for edge k:
    //   "this triangle claims the boundary puncture iff
    //    indices[k] < min(indices[(k+1)%3], indices[(k+2)%3])"
    // For vertex punctures (2 bary coords ≈ 0):
    //   "this triangle claims it iff the non-zero vertex has the smallest index"
    //
    // Since the tet vertices are {0,1,2,3}, these rules deterministically
    // assign each shared-edge puncture to exactly one of the two adjacent faces.
    int gpu_raw_total = 0;
    for (int fi = 0; fi < 4; fi++) {
        if (gpu.face[fi].count > 0 && gpu.face[fi].count < INT_MAX)
            gpu_raw_total += gpu.face[fi].count;
    }

    for (int fi = 0; fi < 4; fi++) {
        // Build face V, W arrays from tet vertices
        double Vf[3][3], Wf[3][3];
        for (int vi = 0; vi < 3; vi++)
            for (int j = 0; j < 3; j++) {
                Vf[vi][j] = Vd[fv[fi][vi]][j];
                Wf[vi][j] = Wd[fv[fi][vi]][j];
            }

        // Global vertex indices for SoS tie-breaking
        uint64_t indices[3] = {
            (uint64_t)fv[fi][0], (uint64_t)fv[fi][1], (uint64_t)fv[fi][2]
        };

        // Exact CPU solver with SoS
        std::vector<PuncturePoint> results;
        solve_pv_triangle(Vf, Wf, results, indices);

        for (const auto& r : results) {
            ClassifiedCase::PunctureInfo pi;
            pi.face = fi;
            pi.lambda = r.lambda;
            for (int j = 0; j < 3; j++)
                pi.bary[j] = r.barycentric[j];
            pi.is_edge = false;
            pi.is_vertex = false;
            pi.tet_edge[0] = pi.tet_edge[1] = -1;
            pi.tet_vertex = -1;
            pi.interval_idx = -1;

            // Detect edge/vertex from the exact solver's SoS-accepted bary coords.
            // After SoS, accepted boundary punctures have bary ≈ 0 for the
            // opposite vertex coordinate — the SoS already decided this face
            // owns the puncture, so we just read off which coords are near zero.
            // Use a generous threshold only for D01/D00 LABELING (not for dedup).
            int big_idx[3];
            int ns = 0, nb = 0;
            for (int j = 0; j < 3; j++) {
                if (std::abs(pi.bary[j]) < 1e-4)
                    ns++;
                else
                    big_idx[nb++] = j;
            }

            if (ns == 2 && nb == 1) {
                pi.is_vertex = true;
                pi.tet_vertex = fv[fi][big_idx[0]];
            } else if (ns == 1 && nb == 2) {
                pi.is_edge = true;
                int e0 = fv[fi][big_idx[0]], e1 = fv[fi][big_idx[1]];
                pi.tet_edge[0] = std::min(e0, e1);
                pi.tet_edge[1] = std::max(e0, e1);
            }

            cc.punctures.push_back(pi);
        }
    }

    // No threshold-based deduplication: the SoS ownership rule in
    // solve_pv_triangle ensures each edge/vertex puncture is reported
    // by exactly one face.
    cc.n_deduplicated = gpu_raw_total - (int)cc.punctures.size();

    // ─── Infinity punctures ───────────────────────────────────────────────
    // When P_i[3]/Q[3] = 0 (i.e., P_i[3] = 0, Q[3] != 0), the PV curve
    // at λ→∞ has μ_i = 0, meaning it asymptotically meets face i.
    // This is a real face crossing at λ=∞ (w=0 critical point on a face).
    // Similarly P_i[0] = 0 with Q[0] != 0 means μ_i(0) = 0: crossing at λ=0.
    if (Q[3] != 0.0) {
        for (int i = 0; i < 4; i++) {
            if (P[i][3] == 0.0) {
                // Check if the limit point is inside the opposite face
                // μ_j(∞) = P_j[3]/Q[3] for j != i; need all >= 0
                bool inside = true;
                for (int j = 0; j < 4; j++) {
                    if (j == i) continue;
                    if (P[j][3] / Q[3] < -1e-10) { inside = false; break; }
                }
                if (inside) {
                    ClassifiedCase::PunctureInfo pi;
                    pi.face = i;
                    pi.lambda = INFINITY;  // λ→∞ (Cw critical point)
                    // Barycentric coords on face i from the limit point
                    double sum = 0;
                    int vi = 0;
                    static const int fv[4][3] = {
                        {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
                    };
                    for (int k = 0; k < 3; k++) {
                        pi.bary[k] = P[fv[i][k]][3] / Q[3];
                        sum += pi.bary[k];
                    }
                    if (sum > 1e-10) {
                        for (int k = 0; k < 3; k++) pi.bary[k] /= sum;
                    }
                    pi.is_edge = false;
                    pi.is_vertex = false;
                    pi.tet_edge[0] = pi.tet_edge[1] = -1;
                    pi.tet_vertex = -1;
                    pi.interval_idx = -1;
                    cc.punctures.push_back(pi);
                }
            }
        }
    }
    // λ=0 punctures: P_i[0]=0 with Q[0]!=0
    if (Q[0] != 0.0) {
        for (int i = 0; i < 4; i++) {
            if (P[i][0] == 0.0) {
                bool inside = true;
                for (int j = 0; j < 4; j++) {
                    if (j == i) continue;
                    if (P[j][0] / Q[0] < -1e-10) { inside = false; break; }
                }
                if (inside) {
                    ClassifiedCase::PunctureInfo pi;
                    pi.face = i;
                    pi.lambda = 0.0;
                    static const int fv[4][3] = {
                        {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
                    };
                    double sum = 0;
                    for (int k = 0; k < 3; k++) {
                        pi.bary[k] = P[fv[i][k]][0] / Q[0];
                        sum += pi.bary[k];
                    }
                    if (sum > 1e-10) {
                        for (int k = 0; k < 3; k++) pi.bary[k] /= sum;
                    }
                    pi.is_edge = false;
                    pi.is_vertex = false;
                    pi.tet_edge[0] = pi.tet_edge[1] = -1;
                    pi.tet_vertex = -1;
                    pi.interval_idx = -1;
                    // Check this wasn't already found by the triangle solver
                    bool already = false;
                    for (auto& ep : cc.punctures)
                        if (std::abs(ep.lambda) < 1e-10 && ep.face == i) already = true;
                    if (!already) cc.punctures.push_back(pi);
                }
            }
        }
    }

    // ─── Deduplicate edge/vertex punctures ──────────────────────────────
    // The triangle solver reports punctures on shared edges/vertices from
    // EACH adjacent face.  Infinity/zero punctures may also land on edges.
    // Detect edge/vertex status for infinity/zero punctures first, then
    // dedup: for each unique (tet_edge) or (tet_vertex), keep the one from
    // the face with the smallest index.
    {
        static const int fv_dedup[4][3] = {
            {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
        };
        for (auto& pi : cc.punctures) {
            if (pi.is_edge || pi.is_vertex) continue;
            // Detect edge/vertex for infinity/zero punctures
            int ns = 0, nb = 0;
            int big_idx[3];
            for (int j = 0; j < 3; j++) {
                if (std::abs(pi.bary[j]) < 1e-4) ns++;
                else big_idx[nb++] = j;
            }
            if (ns == 2 && nb == 1) {
                pi.is_vertex = true;
                pi.tet_vertex = fv_dedup[pi.face][big_idx[0]];
            } else if (ns == 1 && nb == 2) {
                pi.is_edge = true;
                int e0 = fv_dedup[pi.face][big_idx[0]];
                int e1 = fv_dedup[pi.face][big_idx[1]];
                pi.tet_edge[0] = std::min(e0, e1);
                pi.tet_edge[1] = std::max(e0, e1);
            }
        }
        // Dedup: keep first occurrence of each edge/vertex
        std::vector<ClassifiedCase::PunctureInfo> deduped;
        for (auto& pi : cc.punctures) {
            bool dup = false;
            if (pi.is_vertex) {
                for (auto& ep : deduped)
                    if (ep.is_vertex && ep.tet_vertex == pi.tet_vertex) { dup = true; break; }
            } else if (pi.is_edge) {
                for (auto& ep : deduped)
                    if (ep.is_edge && ep.tet_edge[0] == pi.tet_edge[0]
                        && ep.tet_edge[1] == pi.tet_edge[1]) { dup = true; break; }
            }
            if (!dup) deduped.push_back(pi);
        }
        cc.n_deduplicated = (int)cc.punctures.size() - (int)deduped.size();
        cc.punctures = std::move(deduped);
    }

    // Sort punctures by lambda for interval assignment
    std::sort(cc.punctures.begin(), cc.punctures.end(),
        [](const ClassifiedCase::PunctureInfo& a, const ClassifiedCase::PunctureInfo& b) {
            return a.lambda < b.lambda;
        });

    // Determine Q structure
    int degQ = 3;
    while (degQ > 0 && Q[degQ] == 0.0) degQ--;

    std::string q_type;
    if (degQ == 0 && Q[0] == 0.0) q_type = "Qz";  // Q ≡ 0 (data degeneracy D)
    else if (degQ == 0) q_type = "Q0";              // Q = const ≠ 0
    else if (degQ == 1) q_type = "Q1";
    else if (degQ == 2) q_type = "Q2";
    else {
        if (cc.Q_disc_sign > 0) q_type = "Q3+";
        else if (cc.Q_disc_sign < 0) q_type = "Q3-";
        else q_type = "Q3o";
    }

    // Build intervals between Q roots
    // Augmented interval scheme: intervals between consecutive Q roots,
    // plus infinity-spanning intervals
    if (cc.n_Q_roots > 0) {
        // Interval before first root (includes -inf)
        cc.intervals.push_back({-INFINITY, cc.Q_roots[0], 0, true});

        // Intervals between consecutive roots
        for (int i = 0; i + 1 < cc.n_Q_roots; i++)
            cc.intervals.push_back({cc.Q_roots[i], cc.Q_roots[i+1], 0, false});

        // Interval after last root (includes +inf)
        cc.intervals.push_back({cc.Q_roots[cc.n_Q_roots-1], INFINITY, 0, true});
    } else {
        // No Q roots — single interval spanning everything
        cc.intervals.push_back({-INFINITY, INFINITY, 0, true});
    }

    // Assign punctures to intervals.
    //
    // Use the FLOAT Q coefficients (not the quantized Q_i128) for interval
    // assignment.  For integer-valued inputs, Q coefficients are small exact
    // integers (|Q[i]| < 2^19 for R=20), so all double arithmetic is exact.
    // The quantized Q_i128 coefficients are scaled by 2^48, which causes
    // __int128 overflow in the Sturm S₂ computation (products reach 2^189).
    //
    // Strategy:
    //   Q3- (1 root, 2 intervals): sign(Q(λ)) vs sign(q₃) determines interval.
    //   Q3+ (3 roots, 4 intervals): build Sturm chain with S₂ computed in
    //     __int128 (from the small float Q coefficients) and S₃ sign derived
    //     from the exact discriminant.
    double Q_exact[4];
    for (int i = 0; i < 4; i++) Q_exact[i] = Q[i];  // small exact integers

    if (degQ > 0 && cc.n_Q_roots == 1) {
        // --- Q3- (1 root): sign-of-Q method ---
        // For a cubic with positive leading coeff, Q(x) > 0 iff x > root.
        // So: sign(Q(λ)) == sign(q₃) ⟹ λ > root ⟹ interval 1.
        for (auto& pi : cc.punctures) {
            double Qval = eval_poly_sturm(Q_exact, 3, pi.lambda);
            // Higham certification
            double ax = std::abs(pi.lambda);
            double cond = std::abs(Q_exact[3]);
            for (int d = 2; d >= 0; --d) cond = cond * ax + std::abs(Q_exact[d]);
            double gamma = (double)(2 * 3 + 2) * DBL_EPSILON;
            bool cert = std::abs(Qval) > gamma * cond;

            if (!cert) {
                double delta = 4.0 * DBL_EPSILON * std::max(1.0, ax);
                Qval = eval_poly_sturm(Q_exact, 3, pi.lambda + delta);
            }
            int interval_idx = ((Qval > 0) == (Q_exact[3] > 0)) ? 1 : 0;
            pi.interval_idx = interval_idx;
            cc.intervals[interval_idx].n_pv++;
        }

        // Shared root: exact resultant check (replaces Sturm-based pass-through)
        // Use float Q/P (exact integers for integer input) converted to __int128
        {
            __int128 Qi[4], Pi[4][4];
            for (int j = 0; j < 4; j++) Qi[j] = (__int128)llround(Q[j]);
            for (int a = 0; a < 4; a++)
                for (int j = 0; j < 4; j++) Pi[a][j] = (__int128)llround(P[a][j]);
            cc.has_shared_root = has_shared_root_resultant(Qi, Pi);
        }

    } else if (degQ > 0 && cc.n_Q_roots > 1) {
        // --- Q3+ / Q2 (≥2 roots): Sturm chain ---
        // Use the small float Q coefficients (exact integers for integer input),
        // NOT the quantized Q_i128 (scaled by 2^48, overflows __int128 in S₂).
        __int128 p0 = (__int128)llround(Q[0]), p1 = (__int128)llround(Q[1]),
                 p2 = (__int128)llround(Q[2]), p3 = (__int128)llround(Q[3]);
        SturmSeqDouble Q_seq;

        // S₀ = Q
        for (int i = 0; i < 4; i++) Q_seq.c[0][i] = Q_exact[i];
        Q_seq.deg[0] = degQ;

        // S₁ = Q'
        Q_seq.c[1][0] = Q_exact[1];
        Q_seq.c[1][1] = 2.0 * Q_exact[2];
        Q_seq.c[1][2] = 3.0 * Q_exact[3];
        Q_seq.c[1][3] = 0;
        Q_seq.deg[1] = degQ - 1;

        if (degQ == 2) {
            // --- Quadratic Q (2 roots): Sturm chain S₀, S₁, S₂ ---
            // S₂ = q₂·(q₁² - 4·q₀·q₂)  (constant, proportional to discriminant)
            __int128 s2_const = p2 * (p1 * p1 - (__int128)4 * p0 * p2);
            Q_seq.c[2][0] = (s2_const > 0) ? 1.0 : ((s2_const < 0) ? -1.0 : 0.0);
            Q_seq.c[2][1] = 0; Q_seq.c[2][2] = 0; Q_seq.c[2][3] = 0;
            Q_seq.deg[2] = 0;
            Q_seq.n = (s2_const != 0) ? 3 : 2;
        } else {
            // --- Cubic Q (3 roots): S₂ from pseudo-remainder, S₃ from discriminant ---
            // For R=20, Q coeffs < 2^19, products reach ~2^95 — fits __int128.
            __int128 s20_i = p3 * (p1 * p2 - (__int128)9 * p0 * p3);
            __int128 s21_i = (__int128)2 * p3 * (p2 * p2 - (__int128)3 * p1 * p3);
            __int128 abs20 = (s20_i >= 0) ? s20_i : -s20_i;
            __int128 abs21 = (s21_i >= 0) ? s21_i : -s21_i;
            __int128 s2max = (abs20 > abs21) ? abs20 : abs21;
            if (s2max > 0) {
                Q_seq.c[2][0] = (double)s20_i / (double)s2max;
                Q_seq.c[2][1] = (double)s21_i / (double)s2max;
            } else {
                Q_seq.c[2][0] = 0; Q_seq.c[2][1] = 0;
            }
            Q_seq.c[2][2] = 0; Q_seq.c[2][3] = 0;
            Q_seq.deg[2] = (s21_i != 0) ? 1 : 0;

            // S₃: sign derived from exact discriminant.
            // sign(s30) = sign(q₃) × sign(Δ_Q).
            if (s21_i == 0 && s20_i == 0) {
                Q_seq.c[3][0] = 0; Q_seq.n = 2;
            } else if (cc.Q_disc_sign == 0) {
                Q_seq.c[3][0] = 0; Q_seq.n = 3;
            } else {
                int sign_lc = (p3 > 0) ? 1 : -1;
                Q_seq.c[3][0] = (double)(sign_lc * cc.Q_disc_sign);
                Q_seq.deg[3] = 0;
                Q_seq.n = 4;
            }
        }

        for (auto& pi : cc.punctures) {
            auto [count, cert] = sturm_count_at_certified(Q_seq, pi.lambda);
            if (!cert) {
                double delta = 4.0 * DBL_EPSILON * std::max(1.0, std::abs(pi.lambda));
                count = sturm_count_at(Q_seq, pi.lambda + delta);
            }
            int interval_idx = cc.n_Q_roots - count;
            if (interval_idx >= 0 && interval_idx < (int)cc.intervals.size()) {
                pi.interval_idx = interval_idx;
                cc.intervals[interval_idx].n_pv++;
                // Sanity check: puncture lambda should be inside the assigned interval
                double lb = cc.intervals[interval_idx].lb;
                double ub = cc.intervals[interval_idx].ub;
                if (pi.lambda < lb || pi.lambda > ub)
                    fprintf(stderr, "WARNING: seed=%lu lambda=%.6f assigned to interval %d [%.6f, %.6f] (count=%d cert=%d)\n",
                            (unsigned long)cc.gpu.seed, pi.lambda, interval_idx, lb, ub, count, cert);
            }
        }

        // Shared root: exact resultant check
        {
            __int128 Qi[4], Pi[4][4];
            for (int j = 0; j < 4; j++) Qi[j] = (__int128)llround(Q[j]);
            for (int a = 0; a < 4; a++)
                for (int j = 0; j < 4; j++) Pi[a][j] = (__int128)llround(P[a][j]);
            cc.has_shared_root = has_shared_root_resultant(Qi, Pi);
        }
    } else {
        // degQ == 0 or no Q roots: all punctures in single interval
        for (auto& pi : cc.punctures)
            pi.interval_idx = 0;
        cc.intervals[0].n_pv = (int)cc.punctures.size();
    }

    // Use deduplicated puncture count for classification
    int n = (int)cc.punctures.size();

    // Build sorted interval-occupancy tuple: collect non-zero n_pv counts
    std::vector<int> occ;
    for (auto& iv : cc.intervals)
        if (iv.n_pv > 0) occ.push_back(iv.n_pv);
    std::sort(occ.begin(), occ.end());

    // T-category: T{n} + sorted tuple suffix (e.g. T4_(2,2), T8_(4,4))
    std::string t_type = "T" + std::to_string(n);
    if (occ.size() > 1) {
        // Multiple occupied intervals: append _(n1,n2,...)
        t_type += "_(";
        for (size_t i = 0; i < occ.size(); i++) {
            if (i > 0) t_type += ",";
            t_type += std::to_string(occ[i]);
        }
        t_type += ")";
    }

    cc.category = t_type + "_" + q_type;

    // Collect degeneracy tags (joined without leading underscore after Q-type)
    std::vector<std::string> tags;
    if (cc.has_shared_root) tags.push_back("SR");

    // Critical-point degeneracies: C[d]{v|w}
    //   d = dimension of simplex element (omitted for tet interior = nondegenerate)
    //   v = lambda=0 (v-field zero), w = lambda->inf (w-field zero)
    //
    //   Cv/Cw   : field zero in tet interior (nondegenerate, d=3 omitted)
    //   C2v/C2w : PV curve crosses face at lambda=0 / lambda->inf (d=2)
    //   C1v/C1w : same on edge (d=1)
    //   C0v/C0w : same at vertex (d=0)

    // Tet interior: v=0 or w=0 anywhere inside
    // Q[0] = det(A) = 0 means λ=0 is a Q-root → v=0 at the PV point (Cv).
    // Q[3] = det(B) = 0 means λ→∞ is a Q-root → w=0 (Cw).
    // check_field_zero_in_tet fails when det=0 (singular system), so we
    // also detect Cv/Cw via Q coefficients (exact for integer-derived doubles).
    bool has_Cv = check_field_zero_in_tet(gpu.V) || (Q[0] == 0.0);
    bool has_Cw = check_field_zero_in_tet(gpu.W) || (Q[3] == 0.0);

    // Face/edge/vertex crossings at special lambda values
    const double lam_zero_eps = 1e-6;
    // Use std::isinf() to detect λ=∞ punctures (stored as INFINITY)

    bool has_C2v = false, has_C2w = false;
    bool has_C1v = false, has_C1w = false;
    bool has_C0v = false, has_C0w = false;
    bool has_D01 = false;  // point puncture on edge (any λ)

    for (const auto& pi : cc.punctures) {
        bool is_lam0 = (std::abs(pi.lambda) < lam_zero_eps);
        bool is_laminf = std::isinf(pi.lambda);

        // D01: point puncture on a 1-cell (edge)
        if (pi.is_edge) has_D01 = true;

        // Critical-point degeneracies at special λ
        if (is_lam0) {
            if (pi.is_vertex) has_C0v = true;
            else if (pi.is_edge) has_C1v = true;
            else has_C2v = true;
        }
        if (is_laminf) {
            if (pi.is_vertex) has_C0w = true;
            else if (pi.is_edge) has_C1w = true;
            else has_C2w = true;
        }
    }

    if (has_Cv)  tags.push_back("Cv");
    if (has_Cw)  tags.push_back("Cw");
    if (has_C2v) tags.push_back("C2v");
    if (has_C2w) tags.push_back("C2w");
    if (has_C1v) tags.push_back("C1v");
    if (has_C1w) tags.push_back("C1w");
    if (has_C0v) tags.push_back("C0v");
    if (has_C0w) tags.push_back("C0w");

    // ─── Dmd tags: PV m-manifold on d-cell ───────────────────────────────
    // D00: point on vertex (V∥W at vertex)
    // D01: point on edge (puncture lands on tet edge)
    // D02: point on face — nondegenerate, no tag
    // D11: curve on edge (entire edge is PV, same λ)
    // D12: curve on face (PV curve lies on a face)
    // D13: curve in tet — nondegenerate, no tag
    // D22: surface on face (entire face is PV)
    // D23: surface in tet (PV surface in tet interior)
    // D33: entire tet is PV
    // Report only the highest-dimensional (max m) tag.

    struct PVVertInfo {
        bool is_pv;
        bool any_lambda;    // V_i=0 and W_i=0: compatible with any λ
        int64_t lam_num, lam_den;  // λ = -lam_num/lam_den (exact rational)
    };
    PVVertInfo pv_vi[4];

    for (int i = 0; i < 4; i++) {
        pv_vi[i] = {false, false, 0, 1};
        const int* vi = gpu.V[i];
        const int* wi = gpu.W[i];

        // Cross product V_i × W_i (exact integer)
        int64_t cx = (int64_t)vi[1]*wi[2] - (int64_t)vi[2]*wi[1];
        int64_t cy = (int64_t)vi[2]*wi[0] - (int64_t)vi[0]*wi[2];
        int64_t cz = (int64_t)vi[0]*wi[1] - (int64_t)vi[1]*wi[0];
        if (cx != 0 || cy != 0 || cz != 0) continue;  // not parallel

        pv_vi[i].is_pv = true;
        bool v_zero = (vi[0]==0 && vi[1]==0 && vi[2]==0);
        bool w_zero = (wi[0]==0 && wi[1]==0 && wi[2]==0);

        if (v_zero && w_zero) {
            pv_vi[i].any_lambda = true;
        } else if (v_zero) {
            pv_vi[i].lam_num = 0; pv_vi[i].lam_den = 1;  // λ=0
        } else if (w_zero) {
            pv_vi[i].lam_num = 1; pv_vi[i].lam_den = 0;  // λ→∞
        } else {
            // V_i + λW_i = 0 → λ = -V_i[k]/W_i[k]
            for (int k = 0; k < 3; k++)
                if (wi[k] != 0) {
                    pv_vi[i].lam_num = -vi[k];
                    pv_vi[i].lam_den = wi[k];
                    break;
                }
        }
    }

    // Two PV vertices have compatible λ?
    auto lam_compat = [&](int a, int b) -> bool {
        if (!pv_vi[a].is_pv || !pv_vi[b].is_pv) return false;
        if (pv_vi[a].any_lambda || pv_vi[b].any_lambda) return true;
        return pv_vi[a].lam_num * pv_vi[b].lam_den
            == pv_vi[b].lam_num * pv_vi[a].lam_den;
    };

    bool has_D00 = false;  // point on vertex (V∥W at vertex)
    for (int i = 0; i < 4; i++)
        if (pv_vi[i].is_pv) { has_D00 = true; break; }

    bool has_D11 = false;  // curve on edge (both endpoints PV, same λ)
    static const int te[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (int e = 0; e < 6; e++)
        if (lam_compat(te[e][0], te[e][1])) { has_D11 = true; break; }

    bool has_D22 = false;  // surface on face (all 3 vertices PV, same λ)
    static const int tf[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
    for (int f = 0; f < 4; f++)
        if (lam_compat(tf[f][0], tf[f][1]) && lam_compat(tf[f][1], tf[f][2]))
            { has_D22 = true; break; }

    bool has_D33 = lam_compat(0,1) && lam_compat(1,2) && lam_compat(2,3);

    // Append highest-dimensional Dmd tag
    if (has_D33)      tags.push_back("D33");
    else if (has_D22) tags.push_back("D22");
    else if (has_D11) tags.push_back("D11");
    else if (has_D01) tags.push_back("D01");
    else if (has_D00) tags.push_back("D00");

    // Internal loop: T0 case with closed PV curve entirely inside tet.
    // Requires Q with no real roots and all μ_k(0) = P_k(0)/Q(0) > 0.
    if (n == 0 && cc.n_Q_roots == 0 && Q[0] != 0.0) {
        bool inside = true;
        for (int k = 0; k < 4; k++) {
            if (P[k][0] / Q[0] <= 0) { inside = false; break; }
        }
        if (inside) {
            cc.has_B = true;
            tags.push_back("B");
        }
    }

    // Join tags with underscore: T{n}_Q{type}_{tag1}_{tag2}_...
    for (size_t i = 0; i < tags.size(); i++) {
        cc.category += "_" + tags[i];
    }

    // ─── Build puncture pairs ─────────────────────────────────────────────
    // Port of Python _build_pairs(): assign punctures to intervals, sort by
    // lambda within each interval, handle Cw merging, pair consecutively,
    // handle SR pass-through unpaired remainders.
    {
        // Group puncture indices by interval
        std::map<int, std::vector<int>> iv_puncs;
        for (int i = 0; i < (int)cc.punctures.size(); i++) {
            int iv = cc.punctures[i].interval_idx;
            if (iv >= 0)
                iv_puncs[iv].push_back(i);
        }

        // Sort within each interval by lambda (infinity sorts to end)
        for (auto& [iv_idx, pis] : iv_puncs) {
            std::sort(pis.begin(), pis.end(), [&](int a, int b) {
                double la = cc.punctures[a].lambda;
                double lb = cc.punctures[b].lambda;
                if (std::isinf(la) && std::isinf(lb)) return false;
                if (std::isinf(la)) return false;  // inf sorts last
                if (std::isinf(lb)) return true;
                return la < lb;
            });
        }

        // Cw merging: merge leftmost and rightmost intervals (they connect
        // through infinity on the projective line)
        bool has_cw_tag = (cc.category.find("Cw") != std::string::npos);
        std::set<int> right_pis_set, left_pis_set;

        if (has_cw_tag && iv_puncs.size() >= 2) {
            int left_iv = iv_puncs.begin()->first;
            int right_iv = iv_puncs.rbegin()->first;
            if (left_iv != right_iv &&
                (!iv_puncs[left_iv].empty() || !iv_puncs[right_iv].empty())) {
                for (int pi : iv_puncs[right_iv])
                    right_pis_set.insert(pi);
                for (int pi : iv_puncs[left_iv])
                    left_pis_set.insert(pi);
                // Merge: right then left (projective order through infinity)
                auto& merged = iv_puncs[right_iv];
                merged.insert(merged.end(),
                              iv_puncs[left_iv].begin(),
                              iv_puncs[left_iv].end());
                iv_puncs.erase(left_iv);
            }
        }

        // Pair within intervals; collect unpaired
        std::vector<std::pair<int, int>> unpaired;  // (iv_idx, pi)
        for (auto& [iv_idx, pis] : iv_puncs) {
            for (int j = 0; j + 1 < (int)pis.size(); j += 2) {
                int pi_a = pis[j], pi_b = pis[j + 1];
                bool is_cross = has_cw_tag &&
                    ((right_pis_set.count(pi_a) && left_pis_set.count(pi_b)) ||
                     (left_pis_set.count(pi_a) && right_pis_set.count(pi_b)));
                cc.pairs.push_back({pi_a, pi_b, is_cross, iv_idx});
            }
            if ((int)pis.size() % 2 == 1)
                unpaired.push_back({iv_idx, pis.back()});
        }

        // Remaining unpaired: pair by lambda proximity (SR pass-through)
        std::vector<int> up;
        for (auto& [iv, pi] : unpaired)
            up.push_back(pi);
        std::sort(up.begin(), up.end(), [&](int a, int b) {
            double la = cc.punctures[a].lambda;
            double lb = cc.punctures[b].lambda;
            if (std::isinf(la) && std::isinf(lb)) return false;
            if (std::isinf(la)) return false;
            if (std::isinf(lb)) return true;
            return la < lb;
        });
        for (int j = 0; j + 1 < (int)up.size(); j += 2)
            cc.pairs.push_back({up[j], up[j + 1], false, -1});
    }

    return cc;
}

// ─── JSON output ────────────────────────────────────────────────────────────

static void print_json(const ClassifiedCase& cc) {
    printf("{");
    printf("\"seed\":%lu,", (unsigned long)cc.gpu.seed);
    printf("\"category\":\"%s\",", cc.category.c_str());
    printf("\"n_punctures\":%d,", (int)cc.punctures.size());
    printf("\"n_raw\":%d,", cc.gpu.total_punctures);
    printf("\"n_deduplicated\":%d,", cc.n_deduplicated);

    // V
    printf("\"V\":[");
    for (int i = 0; i < 4; i++) {
        printf("[%d,%d,%d]", cc.gpu.V[i][0], cc.gpu.V[i][1], cc.gpu.V[i][2]);
        if (i < 3) printf(",");
    }
    printf("],");

    // W
    printf("\"W\":[");
    for (int i = 0; i < 4; i++) {
        printf("[%d,%d,%d]", cc.gpu.W[i][0], cc.gpu.W[i][1], cc.gpu.W[i][2]);
        if (i < 3) printf(",");
    }
    printf("],");

    // Q coefficients
    printf("\"Q_coeffs\":[%.15g,%.15g,%.15g,%.15g],",
           cc.Q_coeffs[0], cc.Q_coeffs[1], cc.Q_coeffs[2], cc.Q_coeffs[3]);
    printf("\"Q_disc_sign\":%d,", cc.Q_disc_sign);

    // Q roots
    printf("\"Q_roots\":[");
    for (int i = 0; i < cc.n_Q_roots; i++) {
        printf("%.15g", cc.Q_roots[i]);
        if (i + 1 < cc.n_Q_roots) printf(",");
    }
    printf("],");

    // P coefficients
    printf("\"P_coeffs\":[");
    for (int i = 0; i < 4; i++) {
        printf("[%.15g,%.15g,%.15g,%.15g]",
               cc.P_coeffs[i][0], cc.P_coeffs[i][1],
               cc.P_coeffs[i][2], cc.P_coeffs[i][3]);
        if (i < 3) printf(",");
    }
    printf("],");

    // Intervals
    printf("\"intervals\":[");
    for (size_t i = 0; i < cc.intervals.size(); i++) {
        const auto& iv = cc.intervals[i];
        // Use null for ±infinity bounds (valid JSON); is_infinity flag marks them.
        auto print_bound = [](double v) {
            if (std::isinf(v)) printf("null");
            else printf("%.15g", v);
        };
        printf("{\"lb\":"); print_bound(iv.lb);
        printf(",\"ub\":"); print_bound(iv.ub);
        printf(",\"n_pv\":%d,\"is_infinity\":%s}",
               iv.n_pv, iv.is_infinity ? "true" : "false");
        if (i + 1 < cc.intervals.size()) printf(",");
    }
    printf("],");

    // Punctures (with interval assignment)
    printf("\"punctures\":[");
    for (size_t i = 0; i < cc.punctures.size(); i++) {
        const auto& pi = cc.punctures[i];
        printf("{\"face\":%d,\"lambda\":", pi.face);
        if (std::isinf(pi.lambda)) printf("null");
        else printf("%.15g", pi.lambda);
        printf(",\"bary\":[%.15g,%.15g,%.15g],\"interval\":%d}",
               pi.bary[0], pi.bary[1], pi.bary[2], pi.interval_idx);
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
    printf("\"has_B\":%s", cc.has_B ? "true" : "false");
    printf("}\n");
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

static TetCaseGPU generate_tet_from_seed(uint64_t global_id, uint64_t base_seed, int R) {
    uint32_t state = (uint32_t)(global_id ^ (base_seed * 2654435761ULL));
    for (int i = 0; i < 4; i++) lcg_next_cpu(state);

    TetCaseGPU tc;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            tc.V[i][j] = rand_int_cpu(state, R);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            tc.W[i][j] = rand_int_cpu(state, R);

    // Solve PV on CPU (face vertex ordering matches GPU)
    static const int fv[4][3] = {
        {1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}
    };
    tc.total_punctures = 0;
    for (int fi = 0; fi < 4; fi++) {
        double Vf[3][3], Wf[3][3];
        for (int vi = 0; vi < 3; vi++)
            for (int c = 0; c < 3; c++) {
                Vf[vi][c] = (double)tc.V[fv[fi][vi]][c];
                Wf[vi][c] = (double)tc.W[fv[fi][vi]][c];
            }
        uint64_t indices[3] = {
            (uint64_t)fv[fi][0], (uint64_t)fv[fi][1], (uint64_t)fv[fi][2]
        };
        tc.face[fi] = solve_pv_triangle_device(Vf, Wf, indices);
        if (tc.face[fi].count > 0 && tc.face[fi].count < INT_MAX)
            tc.total_punctures += tc.face[fi].count;
    }
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
    // Default parameters
    int min_punctures = 2;
    int num_tets = 100000000;  // 100M
    int R = 20;
    uint64_t base_seed = 42;
    int max_cases = 100000;
    int batch_size = 10000000;  // 10M per batch
    const char* seeds_arg = nullptr;

    // Parse command-line arguments
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
        else if (strcmp(argv[i], "--max-cases") == 0 && i + 1 < argc)
            max_cases = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --min-punctures N   Minimum punctures per tet (default: 2)\n");
            fprintf(stderr, "  --num-tets M        Total random tets to try (default: 100M)\n");
            fprintf(stderr, "  --range R           Integer field range [-R, R] (default: 20)\n");
            fprintf(stderr, "  --seed S            Base random seed (default: 42)\n");
            fprintf(stderr, "  --seeds S1,S2,...    Replay specific seeds on CPU (no GPU needed)\n");
            fprintf(stderr, "  --max-cases C       Max output cases (default: 100000)\n");
            fprintf(stderr, "  --batch-size B      GPU batch size (default: 10M)\n");
            return 0;
        }
    }

    // ─── Seeds mode: CPU-only replay of specific seeds ───────────────────
    if (seeds_arg) {
        auto seeds = parse_seeds(seeds_arg);
        fprintf(stderr, "Seeds mode: replaying %d seeds on CPU (R=%d, base_seed=%lu)\n",
                (int)seeds.size(), R, (unsigned long)base_seed);
        for (uint64_t s : seeds) {
            TetCaseGPU tc = generate_tet_from_seed(s, base_seed, R);
            ClassifiedCase cc = classify_case(tc);
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
    int gpu_max_output = std::min(max_cases * 2, 2000000);  // 2M max
    TetCaseGPU* d_output;
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_output, gpu_max_output * sizeof(TetCaseGPU)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    // Category histogram
    std::map<std::string, int> category_counts;
    // Store one representative per category
    std::map<std::string, ClassifiedCase> representatives;
    int total_found = 0;

    int num_batches = (num_tets + batch_size - 1) / batch_size;
    int block_size = 256;

    for (int batch = 0; batch < num_batches; batch++) {
        int this_batch = std::min(batch_size, num_tets - batch * batch_size);
        uint64_t batch_offset = (uint64_t)batch * batch_size;

        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

        int grid_size = (this_batch + block_size - 1) / block_size;
        tet_case_finder_kernel<<<grid_size, block_size>>>(
            d_output, d_count, gpu_max_output, min_punctures,
            R, base_seed, batch_offset);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download count
        int h_count;
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_count > gpu_max_output) h_count = gpu_max_output;

        if (h_count == 0) continue;

        // Download results
        std::vector<TetCaseGPU> h_cases(h_count);
        CUDA_CHECK(cudaMemcpy(h_cases.data(), d_output,
                              h_count * sizeof(TetCaseGPU), cudaMemcpyDeviceToHost));

        // CPU classification
        for (int i = 0; i < h_count && total_found < max_cases; i++) {
            ClassifiedCase cc = classify_case(h_cases[i]);
            category_counts[cc.category]++;

            // Print JSON to stdout
            print_json(cc);
            total_found++;

            // Store first representative of each category
            if (representatives.find(cc.category) == representatives.end())
                representatives[cc.category] = cc;
        }

        fprintf(stderr, "Batch %d/%d: %d hits (%d total), %d categories\n",
                batch + 1, num_batches, h_count, total_found,
                (int)category_counts.size());

        if (total_found >= max_cases) break;
    }

    CUDA_CHECK(cudaFree(d_output));
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
                cat.c_str(), (unsigned long)cc.gpu.seed,
                (int)cc.punctures.size(), cc.gpu.total_punctures,
                cc.n_deduplicated, cc.Q_disc_sign);
        if (cc.has_shared_root) fprintf(stderr, " [SHARED-ROOT]");
        fprintf(stderr, "\n");
    }

    return 0;
}
