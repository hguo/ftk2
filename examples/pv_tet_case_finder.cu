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

        uint64_t tet_fourth = (uint64_t)fi;  // face fi is opposite vertex fi
        faces[fi] = solve_pv_triangle_device(Vf, Wf, indices, tet_fourth);

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
    TetCaseGPU gpu;
    std::string category;      // e.g. "T4a_Q3+"
    double Q_coeffs[4];
    int Q_disc_sign;
    int n_Q_roots;
    double Q_roots[3];
    double P_coeffs[4][4];
    bool has_shared_root;
    bool has_non_isolated_sr;  // gcd(Q, P_k) degree ≥ 2 for some k
    int n_sr_roots;            // number of shared-root Q-root indices
    int sr_q_root_idx[3];      // which Q roots are shared roots (indices into Q_roots[])
    bool has_B;                // bubble (closed loop inside tet, T0 only)

    // TN: tangency points (disc(P_k) = 0, repeated root)
    int n_tn;                  // number of tangency faces
    struct TNInfo {
        int face;              // which face has the tangency
        double lambda;         // λ at the double root
        double bary[3];        // face bary coords at the tangency point
    } tn_points[4];

    // Cv/Cw positions (tet barycentric coords, valid if has_Cv/has_Cw)
    bool has_Cv_pos;
    double Cv_mu[4];          // tet bary coords where v=0
    bool has_Cw_pos;
    double Cw_mu[4];          // tet bary coords where w=0

    struct PunctureInfo {
        int face;
        double lambda;
        double bary[3];
        bool is_edge;          // on a tet edge (1 face-bary ≈ 0)
        bool is_vertex;        // on a tet vertex (2 face-bary ≈ 0)
        bool is_passthrough;   // edge pass-through: curve stays outside tet on both sides
        bool is_D00;           // isolated PV point at vertex (0-manifold), excluded from T-count
        bool is_D01;           // isolated PV point on edge (0-manifold), excluded from T-count
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

// Check if field=0 is inside tet interior (critical point).
// For integer inputs (|F[i][j]| <= R=20), uses exact int64_t arithmetic
// for the determinant and Cramer numerators.  No thresholds.
// Returns true and writes barycentric coords to mu_out if found inside.
static bool check_field_zero_in_tet(const int F[4][3], double mu_out[4] = nullptr) {
    // Solve: sum_i mu_i * F_i = 0, sum mu_i = 1
    // => (F_0-F_3)*mu_0 + (F_1-F_3)*mu_1 + (F_2-F_3)*mu_2 = -F_3
    int64_t A[3][3], b[3];
    for (int c = 0; c < 3; c++) {
        A[c][0] = (int64_t)F[0][c] - F[3][c];
        A[c][1] = (int64_t)F[1][c] - F[3][c];
        A[c][2] = (int64_t)F[2][c] - F[3][c];
        b[c] = -(int64_t)F[3][c];
    }
    // det is exact integer; |A[i][j]| <= 2R=40, so |det| < 6*40^3 = 384000
    int64_t det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if (det == 0) return false;  // singular: degenerate (D11/D22/D33)

    // Cramer numerators (exact integers)
    int64_t n0 = b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
               - A[0][1]*(b[1]*A[2][2]-A[1][2]*b[2])
               + A[0][2]*(b[1]*A[2][1]-A[1][1]*b[2]);
    int64_t n1 = A[0][0]*(b[1]*A[2][2]-A[1][2]*b[2])
               - b[0]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*b[2]-b[1]*A[2][0]);
    int64_t n2 = A[0][0]*(A[1][1]*b[2]-b[1]*A[2][1])
               - A[0][1]*(A[1][0]*b[2]-b[1]*A[2][0])
               + b[0]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    int64_t n3 = det - n0 - n1 - n2;  // mu_3 = 1 - mu_0 - mu_1 - mu_2

    // Inside tet: all mu_k = n_k/det in [0, 1], i.e. n_k and det have same sign
    // and |n_k| <= |det|.
    if (det > 0) {
        if (n0 < 0 || n1 < 0 || n2 < 0 || n3 < 0) return false;
        if (n0 > det || n1 > det || n2 > det || n3 > det) return false;
    } else {
        if (n0 > 0 || n1 > 0 || n2 > 0 || n3 > 0) return false;
        if (n0 < det || n1 < det || n2 < det || n3 < det) return false;
    }
    if (mu_out) {
        double inv = 1.0 / (double)det;
        mu_out[0] = (double)n0 * inv;
        mu_out[1] = (double)n1 * inv;
        mu_out[2] = (double)n2 * inv;
        mu_out[3] = (double)n3 * inv;
    }
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
// ─── Check if a shared root is relevant to the PV curve in this tet ──────
// After interval assignment, check if the shared Q-root sits between two
// intervals that both contain punctures.  If not, the SR is outside the
// curve's λ-range and doesn't affect topology inside the tet.
//
// For infinity-spanning intervals: the two outermost intervals wrap around
// through ±∞, so an SR at the boundary between them IS relevant if both
// have punctures (the curve passes through infinity).
static bool is_shared_root_relevant(
    const __int128 Q_i128[4], const __int128 P_i128[4][4],
    const double Q_roots[], int n_Q_roots,
    const std::vector<ClassifiedCase::IntervalInfo>& intervals)
{
    if (n_Q_roots == 0) return false;
    int n_intervals = (int)intervals.size();
    if (n_intervals < 2) return false;

    // SR is relevant if punctures are distributed across 2+ intervals.
    // This means the curve passes through at least one Q-interval boundary,
    // which requires an SR pass-through.  If all punctures are in a single
    // interval, the SR is outside the curve's range and irrelevant.
    int n_occupied = 0;
    for (const auto& iv : intervals)
        if (iv.n_pv > 0) n_occupied++;
    return n_occupied >= 2;
}

// ─── Exact polynomial GCD degree via pseudo-remainder ────────────────────
// Returns the degree of gcd(a, b) where a, b have integer coefficients
// (stored as double, degree ≤ 3).  Uses __int128 to avoid overflow during
// the pseudo-remainder steps.  Content reduction after each step keeps
// coefficients bounded.
static int poly_gcd_degree_i128(const double* a_in, int da, const double* b_in, int db) {
    __int128 p[4] = {}, q[4] = {};
    for (int i = 0; i <= std::min(da, 3); i++) p[i] = llround(a_in[i]);
    for (int i = 0; i <= std::min(db, 3); i++) q[i] = llround(b_in[i]);

    while (da > 0 && p[da] == 0) da--;
    while (db > 0 && q[db] == 0) db--;
    if (da == 0 && p[0] == 0) return db;
    if (db == 0 && q[0] == 0) return da;

    int dp = da, dq = db;

    auto content_reduce = [](__int128* poly, int d) {
        if (d < 0) return;
        __int128 g = poly[0] < 0 ? -poly[0] : poly[0];
        for (int i = 1; i <= d; i++) {
            __int128 v = poly[i] < 0 ? -poly[i] : poly[i];
            while (v != 0) { __int128 t = g % v; g = v; v = t; }
        }
        if (g > 1) for (int i = 0; i <= d; i++) poly[i] /= g;
    };

    while (!(dq == 0 && q[0] == 0)) {
        if (dp < dq) {
            std::swap(dp, dq);
            for (int i = 0; i < 4; i++) std::swap(p[i], q[i]);
        }
        // Pseudo-remainder: r = lc(q) * p - (p[dp]/...) * q * x^shift
        __int128 r[4] = {};
        int dr = dp;
        for (int i = 0; i <= dp; i++) r[i] = p[i];

        while (dr >= dq && dr >= 0) {
            if (r[dr] == 0) { dr--; continue; }
            __int128 rdr = r[dr];
            __int128 qdq = q[dq];
            int shift = dr - dq;
            for (int i = 0; i <= dr; i++) r[i] *= qdq;
            for (int i = 0; i <= dq; i++) r[shift + i] -= rdr * q[i];
            dr--;
        }
        if (dr < 0) { dr = 0; r[0] = 0; }
        while (dr > 0 && r[dr] == 0) dr--;

        content_reduce(r, dr);

        dp = dq;
        for (int i = 0; i < 4; i++) p[i] = q[i];
        dq = dr;
        for (int i = 0; i < 4; i++) q[i] = r[i];
    }
    return dp;
}

// ─── Exact edge/vertex detection via Sturm counting on N_k ──────────────
// For each regular puncture on a face, determines which face-barycentric
// coordinates are exactly zero by checking if the bary-numerator polynomial
// N_k(λ) has a root at the puncture's λ.
//
// Algorithm:
//   1. Compute face char poly P_face and integer P_i128 from integer V, W
//   2. Solve cubic + isolate roots → root intervals [lo, hi]
//   3. Compute N_k from integer Mlin/blin (no quantization)
//   4. For each puncture: match to root interval, Sturm-count N_k in [lo, hi]
//   5. If N_k has a root in [lo, hi] → bary k is zero → on edge/vertex
//
// No thresholds: the Sturm count is exact for integer-derived polynomials.
static void detect_edge_vertex_exact(
    const int V_face[3][3], const int W_face[3][3],
    const int fv[3],  // tet vertex indices for this face's 3 vertices
    std::vector<ClassifiedCase::PunctureInfo>& punctures,
    int first_pi, int num_pi)
{
    if (num_pi == 0) return;

    // Step 1: Compute face char poly from integer V, W (exact for R≤20)
    int64_t VqT[3][3], WqT[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VqT[i][j] = (int64_t)V_face[j][i];
            WqT[i][j] = (int64_t)W_face[j][i];
        }

    // Float char poly (coefficients are exact integers in double for R≤20)
    double VTd[3][3], WTd[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            VTd[i][j] = (double)VqT[i][j];
            WTd[i][j] = (double)WqT[i][j];
        }
    double P_face[4];
    characteristic_polynomial_3x3(VTd, WTd, P_face);

    // Integer char poly for exact discriminant
    __int128 P_i128[4];
    characteristic_polynomial_3x3_i128(VqT, WqT, P_i128);

    // Step 2: Solve cubic and find isolation intervals.
    // Use SQRT_EPS-width intervals (not ULP-tight) for reliable N_k Sturm
    // evaluation.  ULP-tight intervals cause catastrophic cancellation when
    // N_k has a root at exactly the same λ as P_face — Horner evaluation of
    // a degree-4 polynomial at a point 1 ULP from a root has O(ε_machine)
    // relative error, which can flip the sign.
    //
    // The SQRT_EPS width matches the solver's tighten_root_interval Phase 1
    // bracket — the natural scale for degree-4 polynomial evaluation.
    double roots[3];
    int n_roots = solve_cubic_real_sos(P_face, roots,
                                       (const uint64_t*)nullptr, P_i128);

    // Build Sturm sequence for P_face to verify isolation
    SturmSeqDouble P_seq;
    build_sturm_double(P_face, P_seq);

    const double SQRT_EPS = std::sqrt(std::numeric_limits<double>::epsilon());
    double lo[3], hi[3];
    for (int i = 0; i < n_roots; ++i) {
        double scale = std::max(1.0, std::abs(roots[i]));
        double half_w = scale * SQRT_EPS;
        lo[i] = roots[i] - half_w;
        hi[i] = roots[i] + half_w;
        // Verify exactly one P_face root in the interval
        int cnt = sturm_count_at(P_seq, lo[i]) - sturm_count_at(P_seq, hi[i]);
        // Expand if root not yet bracketed, shrink if multiple roots
        for (int iter = 0; iter < 60 && cnt != 1; ++iter) {
            if (cnt == 0) half_w *= 2.0;
            else          half_w *= 0.5;
            lo[i] = roots[i] - half_w;
            hi[i] = roots[i] + half_w;
            if (!std::isfinite(lo[i]) || !std::isfinite(hi[i])) break;
            cnt = sturm_count_at(P_seq, lo[i]) - sturm_count_at(P_seq, hi[i]);
        }
        if (cnt != 1) {
            // Fallback: degenerate interval (won't detect N_k roots)
            lo[i] = hi[i] = roots[i];
        }
    }

    // Step 3: Compute N_k, D from integer V, W (no quantization)
    // For R=20: Mlin entries ≤ 40, blin entries ≤ 20, N_k coeffs ≤ ~7e7
    int64_t Mlin[3][2][2], blin[3][2];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c) {
            Mlin[r][c][0] = VqT[r][c] - VqT[r][2];
            Mlin[r][c][1] = -(WqT[r][c] - WqT[r][2]);
        }
        blin[r][0] = -VqT[r][2];
        blin[r][1] =  WqT[r][2];
    }
    double N[3][5], D[5];
    compute_bary_numerators_from_integers(Mlin, blin, N, D);

    // Step 4: Build Sturm sequences for N_k
    SturmSeqDeg4 seq_nk[3];
    for (int k = 0; k < 3; k++) {
        int degNk = 4;
        while (degNk > 0 && N[k][degNk] == 0.0) --degNk;
        build_sturm_deg4(N[k], degNk, seq_nk[k]);
    }

    // Step 5: For each puncture, match to root interval and check N_k
    for (int pi = first_pi; pi < first_pi + num_pi; pi++) {
        auto& punct = punctures[pi];

        // Find the matching root interval by lambda proximity
        int best_root = -1;
        double best_dist = INFINITY;
        for (int r = 0; r < n_roots; r++) {
            double mid = 0.5 * (lo[r] + hi[r]);
            double dist = std::abs(punct.lambda - mid);
            if (dist < best_dist) { best_dist = dist; best_root = r; }
        }

        if (best_root < 0) continue;

        // Check each bary coord via Sturm counting on N_k
        int n_zero = 0, n_nonzero = 0;
        int big_idx[3];
        for (int k = 0; k < 3; k++) {
            int nk_roots = sturm_count_d4(seq_nk[k], lo[best_root])
                         - sturm_count_d4(seq_nk[k], hi[best_root]);
            if (nk_roots > 0)
                n_zero++;
            else
                big_idx[n_nonzero++] = k;
        }

        if (n_zero == 2 && n_nonzero == 1) {
            punct.is_vertex = true;
            punct.tet_vertex = fv[big_idx[0]];
        } else if (n_zero == 1 && n_nonzero == 2) {
            punct.is_edge = true;
            int e0 = fv[big_idx[0]], e1 = fv[big_idx[1]];
            punct.tet_edge[0] = std::min(e0, e1);
            punct.tet_edge[1] = std::max(e0, e1);
        }
    }
}

static ClassifiedCase classify_case(const TetCaseGPU& gpu) {
    ClassifiedCase cc;
    cc.gpu = gpu;
    cc.has_shared_root = false;
    cc.has_non_isolated_sr = false;
    cc.n_sr_roots = 0;
    cc.has_B = false;
    cc.has_Cv_pos = false;
    cc.has_Cw_pos = false;

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

        // Exact CPU solver with SoS (tet mode: pass 4th vertex index)
        uint64_t tet_fourth = (uint64_t)fi;  // face fi is opposite vertex fi
        std::vector<PuncturePoint> results;
        solve_pv_triangle(Vf, Wf, results, indices, tet_fourth);

        int face_start = (int)cc.punctures.size();
        for (const auto& r : results) {
            ClassifiedCase::PunctureInfo pi;
            pi.face = fi;
            pi.lambda = r.lambda;
            for (int j = 0; j < 3; j++)
                pi.bary[j] = r.barycentric[j];
            pi.is_edge = false;
            pi.is_vertex = false;
            pi.is_passthrough = false;
            pi.tet_edge[0] = pi.tet_edge[1] = -1;
            pi.tet_vertex = -1;
            pi.interval_idx = -1;
            cc.punctures.push_back(pi);
        }
        int face_count = (int)cc.punctures.size() - face_start;

        // Exact edge/vertex detection: Sturm counting on N_k polynomials
        // determines which face-barycentric coordinates are exactly zero.
        // No thresholds — replaces abs(bary[j]) < 1e-4.
        int V_face[3][3], W_face[3][3];
        for (int vi = 0; vi < 3; vi++)
            for (int j = 0; j < 3; j++) {
                V_face[vi][j] = gpu.V[fv[fi][vi]][j];
                W_face[vi][j] = gpu.W[fv[fi][vi]][j];
            }
        detect_edge_vertex_exact(V_face, W_face, fv[fi],
                                 cc.punctures, face_start, face_count);
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
        int64_t q3 = llround(Q[3]);
        for (int i = 0; i < 4; i++) {
            if (P[i][3] == 0.0) {
                // Check if the limit point is inside the opposite face
                // μ_j(∞) = P_j[3]/Q[3] for j != i; need all >= 0
                // Exact int64 sign check: P_j[3] and Q[3] must have same sign
                bool inside = true;
                for (int j = 0; j < 4; j++) {
                    if (j == i) continue;
                    int64_t pj3 = llround(P[j][3]);
                    if ((pj3 > 0 && q3 < 0) || (pj3 < 0 && q3 > 0))
                        { inside = false; break; }
                }
                if (inside) {
                    ClassifiedCase::PunctureInfo pi;
                    pi.face = i;
                    pi.lambda = INFINITY;
                    static const int fv_inf[4][3] = {
                        {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
                    };
                    double sum = 0;
                    for (int k = 0; k < 3; k++) {
                        pi.bary[k] = P[fv_inf[i][k]][3] / Q[3];
                        sum += pi.bary[k];
                    }
                    if (sum != 0.0) {
                        for (int k = 0; k < 3; k++) pi.bary[k] /= sum;
                    }
                    // Exact edge/vertex detection for infinity puncture:
                    // bary k is zero iff P[fv[i][k]][3] == 0 (exact integer)
                    pi.is_edge = false;
                    pi.is_vertex = false;
                    pi.is_passthrough = false;
                    pi.tet_edge[0] = pi.tet_edge[1] = -1;
                    pi.tet_vertex = -1;
                    int nz = 0, nnz = 0;
                    int big[3];
                    for (int k = 0; k < 3; k++) {
                        if (P[fv_inf[i][k]][3] == 0.0) nz++;
                        else big[nnz++] = k;
                    }
                    if (nz == 2 && nnz == 1) {
                        pi.is_vertex = true;
                        pi.tet_vertex = fv_inf[i][big[0]];
                    } else if (nz == 1 && nnz == 2) {
                        pi.is_edge = true;
                        int e0 = fv_inf[i][big[0]], e1 = fv_inf[i][big[1]];
                        pi.tet_edge[0] = std::min(e0, e1);
                        pi.tet_edge[1] = std::max(e0, e1);
                    }
                    pi.interval_idx = -1;
                    cc.punctures.push_back(pi);
                }
            }
        }
    }
    // λ=0 punctures: P_i[0]=0 with Q[0]!=0
    if (Q[0] != 0.0) {
        int64_t q0 = llround(Q[0]);
        for (int i = 0; i < 4; i++) {
            if (P[i][0] == 0.0) {
                // Exact int64 sign check for inside test
                bool inside = true;
                for (int j = 0; j < 4; j++) {
                    if (j == i) continue;
                    int64_t pj0 = llround(P[j][0]);
                    if ((pj0 > 0 && q0 < 0) || (pj0 < 0 && q0 > 0))
                        { inside = false; break; }
                }
                if (inside) {
                    ClassifiedCase::PunctureInfo pi;
                    pi.face = i;
                    pi.lambda = 0.0;
                    static const int fv_zero[4][3] = {
                        {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
                    };
                    double sum = 0;
                    for (int k = 0; k < 3; k++) {
                        pi.bary[k] = P[fv_zero[i][k]][0] / Q[0];
                        sum += pi.bary[k];
                    }
                    if (sum != 0.0) {
                        for (int k = 0; k < 3; k++) pi.bary[k] /= sum;
                    }
                    // Exact edge/vertex detection: bary k = 0 iff P[fv[i][k]][0] == 0
                    pi.is_edge = false;
                    pi.is_vertex = false;
                    pi.is_passthrough = false;
                    pi.tet_edge[0] = pi.tet_edge[1] = -1;
                    pi.tet_vertex = -1;
                    int nz = 0, nnz = 0;
                    int big[3];
                    for (int k = 0; k < 3; k++) {
                        if (P[fv_zero[i][k]][0] == 0.0) nz++;
                        else big[nnz++] = k;
                    }
                    if (nz == 2 && nnz == 1) {
                        pi.is_vertex = true;
                        pi.tet_vertex = fv_zero[i][big[0]];
                    } else if (nz == 1 && nnz == 2) {
                        pi.is_edge = true;
                        int e0 = fv_zero[i][big[0]], e1 = fv_zero[i][big[1]];
                        pi.tet_edge[0] = std::min(e0, e1);
                        pi.tet_edge[1] = std::max(e0, e1);
                    }
                    pi.interval_idx = -1;
                    // Check this wasn't already found by the triangle solver
                    // Exact: ep.lambda == 0.0 (integer inputs produce exact 0.0)
                    bool already = false;
                    for (auto& ep : cc.punctures)
                        if (ep.lambda == 0.0 && ep.face == i) already = true;
                    if (!already) cc.punctures.push_back(pi);
                }
            }
        }
    }

    // ─── Deduplicate edge/vertex punctures ──────────────────────────────
    // Edge/vertex detection is now done exactly:
    //   - Regular punctures: Sturm counting on N_k (detect_edge_vertex_exact)
    //   - Infinity/zero punctures: P[k][3]==0 / P[k][0]==0 (exact integer)
    // No threshold-based detection needed here.
    {
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

    // ─── Filter ISR punctures at Q roots ────────────────────────────────
    // At a Q root α, bary coords μ_j = P[j](α)/Q(α).  Since Q(α)=0:
    //   - If any P[j](α) ≠ 0 → μ_j → ∞ → point NOT in tet → spurious.
    //   - Only if ALL P[j](α) = 0 (SR for all faces) can the point be in the tet
    //     (via L'Hôpital limits).
    // We detect this exactly: for each face k, if gcd(Q, P[k]) has degree ≥ 1,
    // check whether ALL other P[j] also share that root.  If not, filter out
    // the puncture.
    //
    // Exact check: h = gcd(Q, P[k]) has the shared roots.  For each j ≠ k,
    // resultant(h, P[j]) = 0 iff P[j] shares at least one root with h.
    // If resultant(h, P[j]) ≠ 0, then P[j] is nonzero at ALL roots of h,
    // so μ_j → ∞ at those roots → puncture invalid.
    {
        __int128 Qi[4], Pi[4][4];
        for (int j = 0; j < 4; j++) Qi[j] = (__int128)llround(Q[j]);
        for (int a = 0; a < 4; a++)
            for (int j = 0; j < 4; j++) Pi[a][j] = (__int128)llround(P[a][j]);

        // For each face, compute h = gcd(Q, P[k])
        bool face_has_isr[4] = {};
        bool face_isr_valid[4] = {};  // true if all other P[j] also vanish at h's roots
        for (int k = 0; k < 4; k++) {
            __int128 h[4] = {};
            int dh = ftk2::poly_gcd_full_i128(Qi, 3, Pi[k], 3, h);
            if (dh < 1) continue;
            face_has_isr[k] = true;

            // Check if all other P[j] vanish at ALL roots of h.
            // Must use deg(gcd(h, P[j])) == deg(h), not resultant == 0.
            bool all_vanish = true;
            for (int j = 0; j < 4; j++) {
                if (j == k) continue;
                int edj = 3;
                while (edj > 0 && Pi[j][edj] == 0) edj--;
                if (edj == 0 && Pi[j][0] == 0) continue;  // P[j] ≡ 0
                __int128 g[4] = {};
                int dg = ftk2::poly_gcd_full_i128(h, dh, Pi[j], edj, g);
                if (dg < dh) { all_vanish = false; break; }
            }
            face_isr_valid[k] = all_vanish;
        }

        // Filter: remove punctures on face k at Q roots when face shares
        // roots with Q.  At Q roots, bary = P(λ)/Q(λ) = 0/0 — the v1
        // per-triangle solver's bary coords are unreliable.  The SR/ISR
        // tagging (below) uses L'Hôpital limits for actual in-tet check.
        std::vector<ClassifiedCase::PunctureInfo> filtered;
        for (auto& pi : cc.punctures) {
            if (face_has_isr[pi.face]) {
                // Check if this puncture's λ is at a Q root
                bool at_q_root = false;
                // Exact: check if P[face](λ) ≈ 0 AND Q(λ) ≈ 0
                // For integer inputs, the shared roots are algebraic;
                // float λ will be very close.  Use bary=[0,0,0] as indicator.
                double bsum = std::abs(pi.bary[0]) + std::abs(pi.bary[1]) + std::abs(pi.bary[2]);
                if (bsum < 1e-10) at_q_root = true;
                // Also check proximity to Q roots
                for (int qi = 0; qi < cc.n_Q_roots; qi++) {
                    double dist = std::abs(pi.lambda - cc.Q_roots[qi]);
                    if (dist < 1e-6 * std::max(1.0, std::abs(pi.lambda)))
                        at_q_root = true;
                }
                if (at_q_root) continue;  // skip this puncture
            }
            filtered.push_back(pi);
        }
        if (filtered.size() != cc.punctures.size()) {
            fprintf(stderr, "  ISR filter: removed %d spurious punctures at Q roots\n",
                    (int)(cc.punctures.size() - filtered.size()));
            cc.punctures = std::move(filtered);
        }
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
    else if (degQ == 2) {
        // Quadratic discriminant: b²-4ac where Q = aλ² + bλ + c
        double disc2 = Q[1]*Q[1] - 4*Q[2]*Q[0];
        if (disc2 > 0) q_type = "Q2";          // 2 real roots
        else if (disc2 < 0) q_type = "Q2-";    // 0 real roots
        else q_type = "Q2o";                    // 1 repeated root
    }
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

        // Shared root: purely algebraic (resultant = 0)
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

        // Shared root: purely algebraic (resultant = 0)
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

    // Use deduplicated puncture count for classification.
    // Edge/vertex punctures at critical λ (λ=0 for Cv1/Cv0, λ=∞ for
    // Cw1/Cw0) are waypoints: the PV curve passes through the sub-simplex
    // smoothly at the field-zero critical point.  These are excluded from
    // T-count, interval occupancy, and pairing.
    //
    // Edge/vertex punctures at generic λ (D01/D00) may be genuine curve
    // endpoints or pass-through waypoints.
    //
    // At an edge puncture, two P[k] are simultaneously zero.  If their
    // derivatives P'[k1] · P'[k2] < 0 at that λ, the violated bary coord
    // swaps without the curve entering the tet — a "pass-through".
    //
    // At a vertex puncture, three P[k] are simultaneously zero.  If the
    // three derivatives don't all have the same sign (relative to Q), the
    // curve can't be inside the tet on either side — also a pass-through.
    //
    // ALL sign tests use exact __int128 polynomial arithmetic (resultant/GCD).
    // Pass-throughs are excluded from T-count and pairing.
    {
        // Compute __int128 P derivatives: dP[k] = P'[k](λ) = P[k][1] + 2P[k][2]λ + 3P[k][3]λ²
        __int128 dP_i128[4][3];  // degree-2 polynomials
        for (int k = 0; k < 4; k++) {
            dP_i128[k][0] = P_i128[k][1];
            dP_i128[k][1] = 2 * P_i128[k][2];
            dP_i128[k][2] = 3 * P_i128[k][3];
        }

        for (auto& pi : cc.punctures) {
            if (pi.lambda == 0.0 || std::isinf(pi.lambda))
                continue;
            if (pi.is_edge) {
                // Edge: tet_edge=[va,vb] → zero P's are k∉{va,vb}
                int va = pi.tet_edge[0], vb = pi.tet_edge[1];
                int k1 = -1, k2 = -1;
                for (int k = 0; k < 4; k++) {
                    if (k == va || k == vb) continue;
                    if (k1 < 0) k1 = k; else k2 = k;
                }
                // g = gcd(P[k1], P[k2]): the common root polynomial
                __int128 g[4];
                int dg = poly_gcd_full_i128(P_i128[k1], effective_degree_i128(P_i128[k1], 3),
                                            P_i128[k2], effective_degree_i128(P_i128[k2], 3), g);
                if (dg < 1) continue;  // no common root (shouldn't happen)
                // Product of derivatives: f = P'[k1] · P'[k2] (degree ≤ 4)
                __int128 prod[5] = {};
                for (int i = 0; i <= 2; i++)
                    for (int j = 0; j <= 2; j++)
                        prod[i + j] += dP_i128[k1][i] * dP_i128[k2][j];
                int dp = 4;
                while (dp > 0 && prod[dp] == 0) dp--;
                // Sign of prod at root(s) of g: pass-through iff negative
                int sign = resultant_sign_i128(g, dg, prod, dp);
                if (sign < 0)
                    pi.is_passthrough = true;
            } else if (pi.is_vertex) {
                // Vertex: tet_vertex=v → zero P's are k≠v
                int v = pi.tet_vertex;
                int ks[3], nk = 0;
                for (int k = 0; k < 4; k++)
                    if (k != v) ks[nk++] = k;
                // g = gcd(P[ks[0]], gcd(P[ks[1]], P[ks[2]])): common root polynomial
                __int128 g12[4];
                int dg12 = poly_gcd_full_i128(P_i128[ks[1]], effective_degree_i128(P_i128[ks[1]], 3),
                                              P_i128[ks[2]], effective_degree_i128(P_i128[ks[2]], 3), g12);
                __int128 g[4];
                int dg = poly_gcd_full_i128(P_i128[ks[0]], effective_degree_i128(P_i128[ks[0]], 3),
                                            g12, dg12, g);
                if (dg < 1) continue;
                // Check sign of P'[ks[0]] · P'[ks[1]] at root(s) of g
                __int128 prod01[5] = {};
                for (int i = 0; i <= 2; i++)
                    for (int j = 0; j <= 2; j++)
                        prod01[i + j] += dP_i128[ks[0]][i] * dP_i128[ks[1]][j];
                int dp01 = 4;
                while (dp01 > 0 && prod01[dp01] == 0) dp01--;
                int sign01 = resultant_sign_i128(g, dg, prod01, dp01);
                // Check sign of P'[ks[0]] · P'[ks[2]] at root(s) of g
                __int128 prod02[5] = {};
                for (int i = 0; i <= 2; i++)
                    for (int j = 0; j <= 2; j++)
                        prod02[i + j] += dP_i128[ks[0]][i] * dP_i128[ks[2]][j];
                int dp02 = 4;
                while (dp02 > 0 && prod02[dp02] == 0) dp02--;
                int sign02 = resultant_sign_i128(g, dg, prod02, dp02);
                // Pass-through if any pair of derivatives has opposite sign
                if (sign01 < 0 || sign02 < 0)
                    pi.is_passthrough = true;
            }
        }
    }
    int n = (int)cc.punctures.size();
    auto is_waypoint = [](const ClassifiedCase::PunctureInfo& pi) {
        if (pi.is_passthrough) return true;
        return (pi.is_edge || pi.is_vertex) &&
               (pi.lambda == 0.0 || std::isinf(pi.lambda));
    };
    int n_face = 0;  // face-interior punctures only
    for (const auto& pi : cc.punctures)
        if (!is_waypoint(pi)) n_face++;

    // Recount interval occupancy excluding waypoints
    for (auto& iv : cc.intervals) iv.n_pv = 0;
    for (const auto& pi : cc.punctures)
        if (!is_waypoint(pi) && pi.interval_idx >= 0)
            cc.intervals[pi.interval_idx].n_pv++;

    // Build sorted interval-occupancy tuple: collect non-zero n_pv counts
    std::vector<int> occ;
    for (auto& iv : cc.intervals)
        if (iv.n_pv > 0) occ.push_back(iv.n_pv);
    std::sort(occ.begin(), occ.end());

    // T-category: T{n_face} + sorted tuple suffix (e.g. T4_(2,2), T8_(4,4))
    std::string t_type = "T" + std::to_string(n_face);
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

    // Shared-root classification: SR (isolated, gcd=1) vs ISR (non-isolated, gcd≥2)
    // Only tag SR/ISR if the shared root is in/on the tet.
    //
    // Algorithm (pure integer):
    // 1. Compute g = gcd(Q, P[0], ..., P[3]) — polynomial whose roots are
    //    where ALL five polynomials vanish (necessary for finite bary coords).
    // 2. If deg(g) ≥ 1: compute Q_red = Q/g, P_red[k] = P[k]/g.
    //    At each root α of g, bary coord = P_red[k](α) / Q_red(α).
    //    Check sign(P_red[k](α)) × sign(Q_red(α)) ≥ 0 for all k
    //    using integer sign-at-root primitives.
    if (cc.has_shared_root) {
        __int128 Qi[4], Pi[4][4];
        for (int j = 0; j < 4; j++) Qi[j] = (__int128)llround(Q[j]);
        for (int a = 0; a < 4; a++)
            for (int j = 0; j < 4; j++) Pi[a][j] = (__int128)llround(P[a][j]);

        // Step 1: g = gcd(Q, P[0], ..., P[3])
        __int128 g[4];
        int dg;
        // Start with Q
        for (int i = 0; i < 4; i++) g[i] = Qi[i];
        dg = 3;
        while (dg > 0 && g[dg] == 0) dg--;
        // Iterate GCD with each P[k]
        for (int k = 0; k < 4 && dg >= 1; k++) {
            int dk = 3;
            while (dk > 0 && Pi[k][dk] == 0) dk--;
            if (dk == 0 && Pi[k][0] == 0) continue;  // P[k] ≡ 0, gcd unchanged
            __int128 g2[4] = {};
            dg = ftk2::poly_gcd_full_i128(g, dg, Pi[k], dk, g2);
            for (int i = 0; i < 4; i++) g[i] = g2[i];
        }

        bool any_in_tet = false;
        cc.has_non_isolated_sr = false;

        if (dg >= 1) {
            // Step 2: Q_red = Q/g, P_red[k] = P[k]/g
            __int128 Qr[4] = {};
            int dQr = ftk2::poly_exact_div_i128(Qi, 3, g, dg, Qr);
            __int128 Pr[4][4] = {};
            int dPr[4] = {};
            for (int k = 0; k < 4; k++) {
                int dk = 3;
                while (dk > 0 && Pi[k][dk] == 0) dk--;
                if (dk == 0 && Pi[k][0] == 0) {
                    dPr[k] = 0; Pr[k][0] = 0;  // P[k] ≡ 0 → P_red ≡ 0 → μ_k = 0 (on face)
                } else {
                    dPr[k] = ftk2::poly_exact_div_i128(Pi[k], dk, g, dg, Pr[k]);
                }
            }

            // Check in-tet: at each root α of g, sign(P_red[k](α)) must agree
            // with sign(Q_red(α)) for all k (so μ_k = P_red/Q_red ≥ 0).
            // For each root of g, compute sign(Q_red(α)) and sign(P_red[k](α))
            // using integer Sturm/PRS primitives.
            //
            // For deg(g) = 1: α = -g[0]/g[1] (rational root)
            //   sign(f(α)) = sign(f(-g[0]/g[1])) = sign(g[1]^deg(f)) × sign(f_eval)
            //   where f_eval = sum_i f[i]·(-g[0])^i·g[1]^(deg(f)-i)
            if (dg == 1) {
                // α = -g[0]/g[1], evaluate exactly
                auto eval_at_root = [&](__int128* f, int df) -> int {
                    // Compute f(-g[0]/g[1]) × g[1]^df
                    __int128 val = 0;
                    __int128 neg_g0_pow = 1;  // (-g[0])^i
                    __int128 g1_pow_desc = 1; // g[1]^(df-i) precomputed
                    // Precompute g[1]^df
                    for (int i = 0; i < df; i++) g1_pow_desc *= g[1];
                    for (int i = 0; i <= df; i++) {
                        val += f[i] * neg_g0_pow * g1_pow_desc;
                        neg_g0_pow *= (-g[0]);
                        if (i < df) g1_pow_desc /= g[1];
                    }
                    // sign(f(α)) = sign(val) × sign(g[1]^df)  ... but val already has g[1]^df
                    // Actually val = g[1]^df × f(α), so sign(f(α)) = sign(val) if g[1]^df > 0
                    // g[1]^df sign: if df even, always +; if df odd, sign(g[1])
                    int g1_sign = (g[1] > 0) ? 1 : -1;
                    int factor = (df % 2 == 0) ? 1 : g1_sign;
                    return (val > 0) ? factor : (val < 0) ? -factor : 0;
                };

                int sq = eval_at_root(Qr, dQr);
                if (sq != 0) {
                    bool in_tet = true;
                    for (int k = 0; k < 4; k++) {
                        if (dPr[k] == 0 && Pr[k][0] == 0) continue;  // μ_k = 0 → on face, ok
                        int sp = eval_at_root(Pr[k], dPr[k]);
                        if (sp * sq < 0) { in_tet = false; break; }  // μ_k < 0 → outside
                    }
                    if (in_tet) any_in_tet = true;
                }
            } else {
                // deg(g) ≥ 2: check each root of g via signs_at_roots_i128.
                // Q_red and P_red[k] have degree ≤ 1 (originals ≤ 3, g ≥ 2).
                int sq[4] = {};
                int nrg = ftk2::signs_at_roots_i128(g, dg, Qr, dQr, sq, 4);
                if (nrg > 0) {
                    // For each root of g, check all P_red[k]
                    for (int ri = 0; ri < nrg; ri++) {
                        if (sq[ri] == 0) continue;  // Q_red = 0 at root: degenerate, skip
                        bool in_tet = true;
                        int sp[4] = {};
                        for (int k = 0; k < 4; k++) {
                            if (dPr[k] == 0 && Pr[k][0] == 0) continue;  // μ_k = 0 → on face
                            int spk[4] = {};
                            ftk2::signs_at_roots_i128(g, dg, Pr[k], dPr[k], spk, 4);
                            if (spk[ri] * sq[ri] < 0) { in_tet = false; break; }
                        }
                        if (in_tet) { any_in_tet = true; break; }
                    }
                }
            }

            if (dg >= 2) cc.has_non_isolated_sr = true;
        }

        if (any_in_tet) {
            tags.push_back(cc.has_non_isolated_sr ? "ISR" : "SR");
            // Record which Q-root indices are SR roots: evaluate g at each Q_root.
            // g divides Q, so g's roots are a subset of Q's roots.
            cc.n_sr_roots = 0;
            for (int qi = 0; qi < cc.n_Q_roots && cc.n_sr_roots < 3; qi++) {
                double r = cc.Q_roots[qi];
                double gval = 0, gscale = 0;
                for (int j = 0; j <= dg; j++) {
                    double gj = (double)g[j];
                    double rpow = 1.0;
                    for (int p = 0; p < j; p++) rpow *= r;
                    gval += gj * rpow;
                    gscale += std::abs(gj * rpow);
                }
                bool is_root = (std::abs(gval) < 1e-10) ||
                               (gscale > 0 && std::abs(gval) / gscale < 1e-6);
                if (is_root)
                    cc.sr_q_root_idx[cc.n_sr_roots++] = qi;
            }
        } else {
            cc.has_shared_root = false;
        }
    }

    // Critical-point degeneracies: Cv{d} / Cw{d}
    //   d = dimension of simplex element where the critical point lies
    //       (omitted when the zero is in the tet interior)
    //   v = lambda=0 (v-field zero), w = lambda->inf (w-field zero)
    //
    //   Cv/Cw   : field zero in tet interior (d=3 omitted)
    //   Cv2/Cw2 : field zero on face (d=2)
    //   Cv1/Cw1 : field zero on edge (d=1)
    //   Cv0/Cw0 : field zero at vertex (d=0)
    //
    // Only the most specific (lowest-dimension) tag is emitted;
    // e.g. Cv on edge → Cv1 (not Cv + Cv1).

    // Tet interior: v=0 or w=0 anywhere inside.
    // check_field_zero_in_tet solves V*mu=0 (or W*mu=0) via exact integer
    // Cramer's rule.  Returns false when det=0 (singular: D11/D22/D33
    // degeneracy) or when the solution is outside the tet.
    // No Q[0]/Q[3] fallback: Q(0)=0 means det(A)=0 (singular), which is
    // necessary but NOT sufficient for v=0 inside the tet.
    bool has_Cv = check_field_zero_in_tet(gpu.V, cc.Cv_mu);
    bool has_Cw = check_field_zero_in_tet(gpu.W, cc.Cw_mu);
    cc.has_Cv_pos = has_Cv;
    cc.has_Cw_pos = has_Cw;

    // Face/edge/vertex crossings at special lambda values.
    // For exact integer inputs, λ=0 from the solver is exactly 0.0;
    // λ=∞ punctures are stored as INFINITY.  No thresholds needed.

    bool has_C2v = false, has_C2w = false;
    bool has_C1v = false, has_C1w = false;
    bool has_C0v = false, has_C0w = false;
    bool has_D01 = false;  // isolated PV point on edge (deferred until after pairing)
    // Collect indices of edge punctures at generic λ for post-pairing D01 check
    std::vector<int> edge_punct_indices;

    for (int pi_idx = 0; pi_idx < (int)cc.punctures.size(); pi_idx++) {
        const auto& pi = cc.punctures[pi_idx];
        bool is_lam0 = (pi.lambda == 0.0);
        bool is_laminf = std::isinf(pi.lambda);

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

        // Candidate D01: edge puncture at generic λ (deferred check)
        if (pi.is_edge && !is_lam0 && !is_laminf)
            edge_punct_indices.push_back(pi_idx);
    }

    // Emit a single Cv{d}/Cw{d} tag using the most specific (lowest) d.
    // When Cv and a face/edge/vertex tag both hold for the same critical point,
    // only the most specific tag is kept (e.g. Cv + C1v → Cv1, not Cv_C1v).
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

    // ─── TN tag: tangency (repeated root of P_k) ────────────────────────
    // When disc(P_k) = 0 for some face k, P_k has a repeated root.
    // The PV curve touches face k tangentially at the double root.
    // Only tag TN if the tangency point is inside the tet (exact integer check).
    //
    // Algorithm:
    //   1. disc(P_k) = 0 → P_k has a repeated root α
    //   2. h = gcd(P_k, P_k') → h is the repeated-root factor (linear for double root)
    //   3. At α: μ_k = 0 (on face k). Check μ_j = P_j(α)/Q(α) ≥ 0 for j ≠ k
    //      via sign(P_j(α)) × sign(Q(α)) ≥ 0 using exact integer eval_at_root.
    cc.n_tn = 0;
    {
        __int128 Qi_tn[4];
        for (int j = 0; j < 4; j++) Qi_tn[j] = (__int128)llround(Q[j]);

        bool has_TN = false;
        for (int k = 0; k < 4; k++) {
            int degk = effective_degree_i128(P_i128[k], 3);
            if (degk < 2 || discriminant_sign_i128(P_i128[k]) != 0)
                continue;

            // Compute h = gcd(P_k, P_k') to find the repeated-root factor
            __int128 Pk_prime[4] = {P_i128[k][1], 2*P_i128[k][2], 3*P_i128[k][3], 0};
            int dk_prime = degk - 1;
            while (dk_prime > 0 && Pk_prime[dk_prime] == 0) dk_prime--;

            __int128 h[4] = {};
            int dh = ftk2::poly_gcd_full_i128(P_i128[k], degk, Pk_prime, dk_prime, h);
            if (dh < 1) continue;

            // For linear h: double root at α = -h[0]/h[1]
            // Check in-tet via exact integer sign evaluation
            if (dh == 1) {
                auto eval_at_root_h = [&](__int128* f, int df) -> int {
                    __int128 val = 0;
                    __int128 neg_h0_pow = 1;
                    __int128 h1_pow_desc = 1;
                    for (int i = 0; i < df; i++) h1_pow_desc *= h[1];
                    for (int i = 0; i <= df; i++) {
                        val += f[i] * neg_h0_pow * h1_pow_desc;
                        neg_h0_pow *= (-h[0]);
                        if (i < df) h1_pow_desc /= h[1];
                    }
                    int h1_sign = (h[1] > 0) ? 1 : -1;
                    int factor = (df % 2 == 0) ? 1 : h1_sign;
                    return (val > 0) ? factor : (val < 0) ? -factor : 0;
                };

                int edQ = 3;
                while (edQ > 0 && Qi_tn[edQ] == 0) edQ--;
                int sq = eval_at_root_h(Qi_tn, edQ);
                if (sq == 0) continue;  // Q vanishes at double root → SR, not TN

                bool in_tet = true;
                for (int j = 0; j < 4; j++) {
                    if (j == k) continue;  // μ_k = 0 (on face k)
                    int edj = 3;
                    while (edj > 0 && P_i128[j][edj] == 0) edj--;
                    if (edj == 0 && P_i128[j][0] == 0) continue;  // P_j ≡ 0 → μ_j = 0
                    int sp = eval_at_root_h(P_i128[j], edj);
                    if (sp * sq < 0) { in_tet = false; break; }
                }
                if (!in_tet) continue;

                has_TN = true;
                // Float λ and bary for visualization only
                double lam_tn = -(double)h[0] / (double)h[1];
                double bary[3] = {};
                double Qval = Q[0] + Q[1]*lam_tn + Q[2]*lam_tn*lam_tn + Q[3]*lam_tn*lam_tn*lam_tn;
                if (std::abs(Qval) > 1e-20) {
                    static const int face_verts[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
                    for (int fi = 0; fi < 3; fi++) {
                        int j = face_verts[k][fi];
                        double Pj = P[j][0] + P[j][1]*lam_tn + P[j][2]*lam_tn*lam_tn + P[j][3]*lam_tn*lam_tn*lam_tn;
                        bary[fi] = Pj / Qval;
                    }
                }
                if (cc.n_tn < 4) {
                    cc.tn_points[cc.n_tn].face = k;
                    cc.tn_points[cc.n_tn].lambda = lam_tn;
                    for (int fi = 0; fi < 3; fi++)
                        cc.tn_points[cc.n_tn].bary[fi] = bary[fi];
                    cc.n_tn++;
                }
            }
            // dh >= 2: triple root — extremely rare, skip for now
        }
        if (has_TN) tags.push_back("TN");
    }

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

    bool has_D12 = false;  // curve on face (P[k] ≡ 0 AND curve passes through face triangle)
    for (int k = 0; k < 4; k++) {
        if (!(P[k][0]==0.0 && P[k][1]==0.0 && P[k][2]==0.0 && P[k][3]==0.0))
            continue;
        // P[k] ≡ 0: curve lies on face k's plane (μ_k = 0 for all λ).
        // Only tag D12 if the curve actually enters the face triangle,
        // i.e., sign(P[j](λ)) = sign(Q(λ)) for all j ≠ k at some λ.
        // Quick check: at λ → +∞, sign = sign(leading coeff).
        // If all P[j] leading coeffs agree with Q's, curve is inside at large λ.
        __int128 q_lc = 0, p_lc[3] = {};
        { int oi = 0;
          for (int c = 3; c >= 0; c--) if ((__int128)llround(Q[c]) != 0) { q_lc = (__int128)llround(Q[c]); break; }
          for (int j = 0; j < 4; j++) {
              if (j == k) continue;
              for (int c = 3; c >= 0; c--) if ((__int128)llround(P[j][c]) != 0) { p_lc[oi] = (__int128)llround(P[j][c]); break; }
              oi++;
          }
        }
        if (q_lc != 0) {
            bool ok = true;
            for (int j = 0; j < 3; j++)
                if (p_lc[j] * q_lc < 0) { ok = false; break; }
            if (ok) { has_D12 = true; break; }
        }
        // Also check λ → -∞: sign may differ for odd-degree leading terms.
        // Negate leading coeffs of odd-degree polynomials.
        // Actually, just check if n_punctures > 0 (curve enters tet).
        if (n > 0) { has_D12 = true; break; }
    }

    bool has_D33 = lam_compat(0,1) && lam_compat(1,2) && lam_compat(2,3);

    // Append highest-dimensional Dmd tag
    // Note: D01 is deferred until after pairing (checked below).
    // D00 is tentatively added here; removed later if D01 is confirmed.
    bool has_D00_tentative = false;
    if (has_D33)      tags.push_back("D33");
    else if (has_D22) tags.push_back("D22");
    else if (has_D12) tags.push_back("D12");
    else if (has_D11) tags.push_back("D11");
    else if (has_D00) { tags.push_back("D00"); has_D00_tentative = true; }

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
        // Group non-waypoint puncture indices by interval.
        // Only critical waypoints (Cv1/Cv0/Cw1/Cw0 at λ=0/∞) are excluded.
        std::map<int, std::vector<int>> iv_puncs;
        for (int i = 0; i < (int)cc.punctures.size(); i++) {
            if (is_waypoint(cc.punctures[i]))
                continue;  // skip critical waypoints
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

        // Infinity merging: the two infinity-extending half-intervals connect
        // through infinity when the PV curve at λ→±∞ is inside the tet.
        //
        // Case 1: Q[3]=0 → deg(Q)<3, the curve reaches λ=∞ (Cw-type).
        // Case 2: Q[3]≠0 → asymptotic point μ_k(∞) = P_k[3]/Q[3].
        //         If all P_k[3] have the same sign as Q[3] (exact integer check),
        //         the asymptotic point is inside the tet and the curve wraps
        //         through infinity without Cw.
        //
        // For integer inputs, P_k[3] and Q[3] are exact integers in double.
        bool merge_infinity;
        if (Q[3] == 0.0) {
            merge_infinity = true;
        } else {
            // Exact integer sign check: P_k[3] * Q[3] >= 0 for all k
            int64_t q3 = llround(Q[3]);
            merge_infinity = true;
            for (int k = 0; k < 4; k++) {
                int64_t pk3 = llround(P[k][3]);
                if ((pk3 > 0 && q3 < 0) || (pk3 < 0 && q3 > 0)) {
                    merge_infinity = false;
                    break;
                }
            }
        }
        std::set<int> right_pis_set, left_pis_set;

        if (merge_infinity && iv_puncs.size() >= 2) {
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
                bool is_cross = merge_infinity &&
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

    // ─── Deferred D01 check ──────────────────────────────────────────────
    // D01 = isolated PV point on edge.  An edge puncture that is paired with
    // a puncture on a *different* face or edge is NOT isolated — it's a regular
    // curve crossing at an edge position.  Only flag D01 when both punctures
    // in a pair are edge punctures on the SAME tet edge (degenerate tangency)
    // or when an edge puncture is unpaired.
    if (!edge_punct_indices.empty()) {
        // Build set of edge puncture indices for quick lookup
        std::set<int> edge_set(edge_punct_indices.begin(), edge_punct_indices.end());

        // Check each edge puncture: is it paired with the same tet edge?
        for (int ei : edge_punct_indices) {
            const auto& pi = cc.punctures[ei];
            bool paired_same_edge = false;
            bool is_paired = false;

            for (const auto& pp : cc.pairs) {
                int partner = -1;
                if (pp.pi_a == ei) { partner = pp.pi_b; is_paired = true; }
                else if (pp.pi_b == ei) { partner = pp.pi_a; is_paired = true; }
                if (partner < 0) continue;

                // Check if partner is also on the same tet edge
                const auto& pj = cc.punctures[partner];
                if (pj.is_edge &&
                    pi.tet_edge[0] == pj.tet_edge[0] &&
                    pi.tet_edge[1] == pj.tet_edge[1]) {
                    paired_same_edge = true;
                }
                break;
            }

            // D01 only if unpaired or paired on same edge (degenerate tangency)
            if (!is_paired || paired_same_edge) {
                has_D01 = true;
                break;
            }
        }

        // Append D01 to category if confirmed (wasn't added during initial tagging)
        if (has_D01) {
            if (has_D00_tentative) {
                // Replace D00 with D01 (D01 is higher-dimensional)
                auto pos = cc.category.find("_D00");
                if (pos != std::string::npos)
                    cc.category.replace(pos, 4, "_D01");
            } else if (cc.category.find("D01") == std::string::npos &&
                       cc.category.find("D11") == std::string::npos &&
                       cc.category.find("D22") == std::string::npos &&
                       cc.category.find("D33") == std::string::npos) {
                cc.category += "_D01";
            }
        }
    }

    return cc;
}

// ─── classify_case_v2: Pure-integer classification from ExactPV2Result ────
// Takes TetCaseV2GPU directly — all topological decisions use __int128.
// Floats computed only for visualization (lambda, bary for JSON/plotting).
static ClassifiedCase classify_case_v2(const TetCaseV2GPU& gpu_v2) {
    ClassifiedCase cc;
    memset(&cc.gpu, 0, sizeof(cc.gpu));
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            cc.gpu.V[i][j] = gpu_v2.V[i][j];
            cc.gpu.W[i][j] = gpu_v2.W[i][j];
        }
    cc.gpu.total_punctures = gpu_v2.v2.n_punctures;
    cc.gpu.seed = gpu_v2.seed;
    cc.has_shared_root = false;
    cc.has_non_isolated_sr = false;
    cc.n_sr_roots = 0;
    cc.has_B = false;
    cc.has_Cv_pos = false;
    cc.has_Cw_pos = false;
    cc.n_tn = 0;
    cc.n_deduplicated = 0;

    // ─── Step 0: Float + integer Q, P polynomials ─────────────────────
    double Vd[4][3], Wd[4][3];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            Vd[i][j] = (double)gpu_v2.V[i][j];
            Wd[i][j] = (double)gpu_v2.W[i][j];
        }

    double Q[4], P[4][4];
    characteristic_polynomials_pv_tetrahedron(Vd, Wd, Q, P);
    for (int i = 0; i < 4; i++) cc.Q_coeffs[i] = Q[i];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cc.P_coeffs[i][j] = P[i][j];

    __int128 Q_i128[4], P_i128[4][4];
    compute_tet_QP_i128(gpu_v2.V, gpu_v2.W, Q_i128, P_i128);

    cc.Q_disc_sign = discriminant_sign_i128(Q_i128);
    cc.n_Q_roots = solve_cubic_real(Q, cc.Q_roots);
    std::sort(cc.Q_roots, cc.Q_roots + cc.n_Q_roots);

    static const int fv[4][3] = {
        {1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}
    };

    // Solve float face cubics for lambda recovery
    double face_roots[4][3];
    int face_n_distinct[4];
    for (int fi = 0; fi < 4; fi++) {
        double raw_roots[3];
        int nr = solve_cubic_real(P[fi], raw_roots);
        std::sort(raw_roots, raw_roots + nr);
        // Deduplicate (merge roots within relative epsilon)
        face_n_distinct[fi] = 0;
        for (int r = 0; r < nr; r++) {
            if (face_n_distinct[fi] == 0 ||
                std::abs(raw_roots[r] - face_roots[fi][face_n_distinct[fi]-1]) >
                1e-8 * std::max(1.0, std::abs(raw_roots[r]))) {
                face_roots[fi][face_n_distinct[fi]++] = raw_roots[r];
            }
        }
    }

    // ─── Step 1: Map v2 punctures → PunctureInfo ──────────────────────
    const ExactPV2Result& v2 = gpu_v2.v2;
    for (int pi = 0; pi < v2.n_punctures; pi++) {
        const auto& vp = v2.punctures[pi];
        ClassifiedCase::PunctureInfo ci;
        ci.face = vp.face;
        ci.is_edge = vp.is_edge;
        ci.is_vertex = vp.is_vertex;
        ci.is_passthrough = false;  // v2 already excluded pass-throughs
        ci.is_D00 = false;
        ci.is_D01 = false;
        ci.interval_idx = -1;
        ci.tet_edge[0] = ci.tet_edge[1] = -1;
        ci.tet_vertex = -1;

        // Float lambda from root_idx
        if (vp.root_idx < 0) {
            ci.lambda = INFINITY;  // infinity puncture (Cw2)
            // Guard: v2 solver may spuriously set edge/vertex flags on infinity
            // punctures (root_idx=-1 causes signs_at_roots_i128[-1] UB in step 3).
            // Cw2 face-interior infinity punctures must NOT be edge/vertex.
            ci.is_edge = false;
            ci.is_vertex = false;
        } else if (vp.root_idx < face_n_distinct[vp.face]) {
            ci.lambda = face_roots[vp.face][vp.root_idx];
        } else {
            // Fallback: shouldn't happen for generic cases
            ci.lambda = face_roots[vp.face][face_n_distinct[vp.face] - 1];
            fprintf(stderr, "  [v2] seed=%lu: root_idx=%d >= n_distinct=%d for face %d\n",
                    (unsigned long)gpu_v2.seed, vp.root_idx, face_n_distinct[vp.face], vp.face);
        }

        // Float bary for visualization
        if (std::isinf(ci.lambda)) {
            double sum = 0;
            for (int k = 0; k < 3; k++) {
                ci.bary[k] = P[fv[vp.face][k]][3] / Q[3];
                sum += ci.bary[k];
            }
            if (sum != 0.0)
                for (int k = 0; k < 3; k++) ci.bary[k] /= sum;
        } else {
            double Qval = Q[0] + ci.lambda*(Q[1] + ci.lambda*(Q[2] + ci.lambda*Q[3]));
            double sum = 0;
            for (int k = 0; k < 3; k++) {
                int j = fv[vp.face][k];
                double Pj = P[j][0] + ci.lambda*(P[j][1] + ci.lambda*(P[j][2] + ci.lambda*P[j][3]));
                ci.bary[k] = (std::abs(Qval) > 1e-20) ? Pj / Qval : 0.0;
                sum += ci.bary[k];
            }
            if (sum != 0.0)
                for (int k = 0; k < 3; k++) ci.bary[k] /= sum;
        }

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

        // Vertex: find k with max |P[k](lambda)| (the non-zero face = tet vertex)
        if (ci.is_vertex) {
            double best = -1;
            for (int k = 0; k < 4; k++) {
                double pk_abs;
                if (std::isinf(ci.lambda))
                    pk_abs = std::abs(P[k][3]);
                else
                    pk_abs = std::abs(P[k][0] + ci.lambda*(P[k][1] + ci.lambda*(P[k][2] + ci.lambda*P[k][3])));
                if (pk_abs > best) { best = pk_abs; ci.tet_vertex = k; }
            }
        }

        cc.punctures.push_back(ci);
    }
    // v2 punctures are already sorted by lambda (solver step 6) and deduplicated (SoS ownership)

    // ─── Step 2: Q-type string ────────────────────────────────────────
    int degQ = 3;
    while (degQ > 0 && Q[degQ] == 0.0) degQ--;

    std::string q_type;
    if (degQ == 0 && Q[0] == 0.0) q_type = "Qz";
    else if (degQ == 0) q_type = "Q0";
    else if (degQ == 1) q_type = "Q1";
    else if (degQ == 2) {
        double disc2 = Q[1]*Q[1] - 4*Q[2]*Q[0];
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
        cc.intervals.push_back({-INFINITY, cc.Q_roots[0], 0, true});
        for (int i = 0; i + 1 < cc.n_Q_roots; i++)
            cc.intervals.push_back({cc.Q_roots[i], cc.Q_roots[i+1], 0, false});
        cc.intervals.push_back({cc.Q_roots[cc.n_Q_roots-1], INFINITY, 0, true});
    } else {
        cc.intervals.push_back({-INFINITY, INFINITY, 0, true});
    }

    // ─── Step 4: Interval assignment (Sturm-based on float lambda) ────
    // Uses original Q (not Q_red) for consistency with v1 taxonomy.
    double Q_exact[4];
    for (int i = 0; i < 4; i++) Q_exact[i] = Q[i];

    // Shared root detection (needed for SR tag and verify_case)
    // Use Q_i128/P_i128 from compute_tet_QP_i128 (line 1921) with
    // resultant_sign_i128 (GCD-factored Bareiss, no overflow).
    if (degQ > 0) {
        int degQ_i = ftk2::effective_degree_i128(Q_i128, 3);
        bool sr = false;
        for (int k = 0; k < 4 && !sr; k++) {
            int degPk = ftk2::effective_degree_i128(P_i128[k], 3);
            if (degPk <= 0) continue;
            if (ftk2::resultant_sign_i128(Q_i128, degQ_i, P_i128[k], degPk) == 0)
                sr = true;
        }
        cc.has_shared_root = sr;
    }

    if (degQ > 0 && cc.n_Q_roots == 1) {
        // Sign-of-Q method (Q3-, Q1)
        for (auto& pi : cc.punctures) {
            if (std::isinf(pi.lambda)) {
                pi.interval_idx = 1;  // infinity > the single root
                cc.intervals[1].n_pv++;
                continue;
            }
            double Qval = eval_poly_sturm(Q_exact, 3, pi.lambda);
            double ax = std::abs(pi.lambda);
            double cond = std::abs(Q_exact[3]);
            for (int d = 2; d >= 0; --d) cond = cond * ax + std::abs(Q_exact[d]);
            double gamma = (double)(2 * 3 + 2) * DBL_EPSILON;
            bool cert = std::abs(Qval) > gamma * cond;
            if (!cert) {
                double delta = 4.0 * DBL_EPSILON * std::max(1.0, ax);
                Qval = eval_poly_sturm(Q_exact, 3, pi.lambda + delta);
            }
            // For any degree with 1 root: above root iff sign(Q(λ)) == sign(lc)
            double lc = Q_exact[degQ];
            int interval_idx = ((Qval > 0) == (lc > 0)) ? 1 : 0;
            pi.interval_idx = interval_idx;
            cc.intervals[interval_idx].n_pv++;
        }
    } else if (degQ > 0 && cc.n_Q_roots > 1) {
        // Sturm chain (Q3+, Q2)
        __int128 p0 = (__int128)llround(Q[0]), p1 = (__int128)llround(Q[1]),
                 p2 = (__int128)llround(Q[2]), p3 = (__int128)llround(Q[3]);
        SturmSeqDouble Q_seq;

        for (int i = 0; i < 4; i++) Q_seq.c[0][i] = Q_exact[i];
        Q_seq.deg[0] = degQ;
        Q_seq.c[1][0] = Q_exact[1];
        Q_seq.c[1][1] = 2.0 * Q_exact[2];
        Q_seq.c[1][2] = 3.0 * Q_exact[3];
        Q_seq.c[1][3] = 0;
        Q_seq.deg[1] = degQ - 1;

        if (degQ == 2) {
            __int128 s2_const = p2 * (p1 * p1 - (__int128)4 * p0 * p2);
            Q_seq.c[2][0] = (s2_const > 0) ? 1.0 : ((s2_const < 0) ? -1.0 : 0.0);
            Q_seq.c[2][1] = 0; Q_seq.c[2][2] = 0; Q_seq.c[2][3] = 0;
            Q_seq.deg[2] = 0;
            Q_seq.n = (s2_const != 0) ? 3 : 2;
        } else {
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
            if (std::isinf(pi.lambda)) {
                pi.interval_idx = cc.n_Q_roots;
                cc.intervals[cc.n_Q_roots].n_pv++;
                continue;
            }
            auto [count, cert] = sturm_count_at_certified(Q_seq, pi.lambda);
            if (!cert) {
                double delta = 4.0 * DBL_EPSILON * std::max(1.0, std::abs(pi.lambda));
                count = sturm_count_at(Q_seq, pi.lambda + delta);
            }
            int interval_idx = cc.n_Q_roots - count;
            if (interval_idx >= 0 && interval_idx < (int)cc.intervals.size()) {
                pi.interval_idx = interval_idx;
                cc.intervals[interval_idx].n_pv++;
            }
        }
    } else {
        for (auto& pi : cc.punctures)
            pi.interval_idx = 0;
        cc.intervals[0].n_pv = (int)cc.punctures.size();
    }

    // ─── Step 5: Pairs from ExactPV2Result (moved before T-count) ─────
    {
        // Infinity merging (same logic as v1)
        bool merge_infinity;
        if (Q[3] == 0.0) {
            merge_infinity = true;
        } else {
            int64_t q3 = llround(Q[3]);
            merge_infinity = true;
            for (int k = 0; k < 4; k++) {
                int64_t pk3 = llround(P[k][3]);
                if ((pk3 > 0 && q3 < 0) || (pk3 < 0 && q3 > 0)) {
                    merge_infinity = false;
                    break;
                }
            }
        }

        // Determine left/right interval sets for cross detection
        int n_iv = (int)cc.intervals.size();
        std::set<int> right_pis_set, left_pis_set;
        if (merge_infinity && n_iv >= 2) {
            for (int i = 0; i < (int)cc.punctures.size(); i++) {
                if (cc.punctures[i].interval_idx == n_iv - 1)
                    right_pis_set.insert(i);
                if (cc.punctures[i].interval_idx == 0)
                    left_pis_set.insert(i);
            }
        }

        for (int i = 0; i < v2.n_pairs; i++) {
            int a = v2.pairs[i].a, b = v2.pairs[i].b;
            if (a >= (int)cc.punctures.size() || b >= (int)cc.punctures.size())
                continue;  // safety
            bool is_cross = merge_infinity &&
                ((right_pis_set.count(a) && left_pis_set.count(b)) ||
                 (left_pis_set.count(a) && right_pis_set.count(b)));
            int iv_idx = cc.punctures[a].interval_idx;
            cc.pairs.push_back({a, b, is_cross, iv_idx});
        }

        // Handle unpaired remainder (SR pass-through)
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
            double la = cc.punctures[a].lambda;
            double lb = cc.punctures[b].lambda;
            if (std::isinf(la) && std::isinf(lb)) return false;
            if (std::isinf(la)) return false;
            if (std::isinf(lb)) return true;
            return la < lb;
        });
        for (int j = 0; j + 1 < (int)unpaired.size(); j += 2)
            cc.pairs.push_back({unpaired[j], unpaired[j + 1], false, -1});
    }

    // ─── Step 6: D00/D01 detection and marking on punctures ──────────
    // Detect D00 (PV vertex) and D01 (isolated edge puncture) BEFORE T-count
    // so they can be excluded from T (which counts 1-manifold curve crossings).

    // D00: vertex i where V[i] × W[i] = 0
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

    // Mark vertex punctures as D00 (0-manifold isolated point at vertex)
    if (has_D00) {
        for (auto& pi : cc.punctures)
            if (pi.is_vertex) pi.is_D00 = true;
    }

    // D01: edge puncture that is unpaired or paired with another on the same edge
    // For same-edge pairs, use derivative entry test to check if curve enters tet
    bool has_D01 = false;
    std::vector<int> edge_punct_indices;
    for (int pi_idx = 0; pi_idx < (int)cc.punctures.size(); pi_idx++) {
        const auto& pi = cc.punctures[pi_idx];
        if (pi.is_edge && !std::isinf(pi.lambda) && pi.lambda != 0.0)
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
                // Unpaired edge puncture → always D01
                has_D01 = true;
                cc.punctures[ei].is_D01 = true;
            } else if (paired_same_edge) {
                // Same-edge pair: derivative entry test to distinguish
                // isolated D01 vs connected U-turn curve through tet interior
                int a = pi.tet_edge[0], b = pi.tet_edge[1];
                // The two faces sharing edge [a,b] are {0,1,2,3} \ {a, b}
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

                bool is_isolated = true;  // conservative default
                if (deg_gfg >= 2) {
                    // Derivatives of P[f] and P[g] (integer coefficients)
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

                    // Check consecutive root pairs for entry at both endpoints
                    for (int r = 0; r + 1 < n_roots; r++) {
                        // At root r (smaller), moving toward root r+1:
                        // curve enters iff P[f]'/Q > 0 and P[g]'/Q > 0
                        bool enters_f_r = s_fprime[r] * s_q[r] > 0;
                        bool enters_g_r = s_gprime[r] * s_q[r] > 0;
                        // At root r+1 (larger), moving toward root r:
                        // curve enters iff P[f]'/Q < 0 and P[g]'/Q < 0
                        bool enters_f_r1 = s_fprime[r+1] * s_q[r+1] < 0;
                        bool enters_g_r1 = s_gprime[r+1] * s_q[r+1] < 0;

                        if (enters_f_r && enters_g_r && enters_f_r1 && enters_g_r1)
                            is_isolated = false;  // curve enters tet at both endpoints → connected
                    }
                }
                // deg_gfg < 2: fewer than 2 shared roots → conservative D01
                // Any s_q[r]==0 or s_fprime[r]==0: SR/TN case → conservative D01

                if (is_isolated) {
                    has_D01 = true;
                    cc.punctures[ei].is_D01 = true;
                    // Mark partner
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
        if (!(P[k][0]==0.0 && P[k][1]==0.0 && P[k][2]==0.0 && P[k][3]==0.0))
            continue;
        __int128 q_lc = 0, p_lc[3] = {};
        { int oi = 0;
          for (int c = 3; c >= 0; c--) if ((__int128)llround(Q[c]) != 0) { q_lc = (__int128)llround(Q[c]); break; }
          for (int j = 0; j < 4; j++) {
              if (j == k) continue;
              for (int c = 3; c >= 0; c--) if ((__int128)llround(P[j][c]) != 0) { p_lc[oi] = (__int128)llround(P[j][c]); break; }
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
    // T counts punctures that participate in PV curve segments (1-manifold).
    // Excluded: isolated D00/D01 (unpaired 0-manifold point at vertex/edge).
    // Included: all other punctures (face-interior, Cw2, non-isolated edge/vertex).
    // Non-isolated D00/D01 (paired with face-interior puncture) ARE counted.
    int n_face = 0;
    // Build set of paired puncture indices
    std::vector<bool> is_paired(cc.punctures.size(), false);
    for (const auto& pr : cc.pairs) {
        if (pr.pi_a >= 0 && pr.pi_a < (int)cc.punctures.size()) is_paired[pr.pi_a] = true;
        if (pr.pi_b >= 0 && pr.pi_b < (int)cc.punctures.size()) is_paired[pr.pi_b] = true;
    }
    // Recompute interval occupancy: exclude only ISOLATED D00/D01 (unpaired)
    for (auto& iv : cc.intervals) iv.n_pv = 0;
    for (int pi_idx = 0; pi_idx < (int)cc.punctures.size(); pi_idx++) {
        const auto& pi = cc.punctures[pi_idx];
        if ((pi.is_D00 || pi.is_D01) && !is_paired[pi_idx]) continue;
        n_face++;
        if (pi.interval_idx >= 0 && pi.interval_idx < (int)cc.intervals.size())
            cc.intervals[pi.interval_idx].n_pv++;
    }

    // Category string built after Cv/Cw detection (Step 8b) to include waypoints

    // ─── Step 9: Tags ─────────────────────────────────────────────────
    std::vector<std::string> tags;

    // ── SR/ISR: shared-root classification ──
    if (cc.has_shared_root) {
        // Use Q_i128/P_i128 directly (from compute_tet_QP_i128, no llround overflow)
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

        bool any_in_tet = false;
        cc.has_non_isolated_sr = false;

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
                    if (in_tet) any_in_tet = true;
                }
            } else {
                int sq[4] = {};
                int nrg = ftk2::signs_at_roots_i128(g, dg, Qr, dQr, sq, 4);
                if (nrg > 0) {
                    for (int ri = 0; ri < nrg; ri++) {
                        if (sq[ri] == 0) continue;
                        bool in_tet = true;
                        for (int k = 0; k < 4; k++) {
                            if (dPr[k] == 0 && Pr[k][0] == 0) continue;
                            int spk[4] = {};
                            ftk2::signs_at_roots_i128(g, dg, Pr[k], dPr[k], spk, 4);
                            if (spk[ri] * sq[ri] < 0) { in_tet = false; break; }
                        }
                        if (in_tet) { any_in_tet = true; break; }
                    }
                }
            }
            if (dg >= 2) cc.has_non_isolated_sr = true;
        }

        if (any_in_tet) {
            tags.push_back(cc.has_non_isolated_sr ? "ISR" : "SR");
            cc.n_sr_roots = 0;
            for (int qi = 0; qi < cc.n_Q_roots && cc.n_sr_roots < 3; qi++) {
                double r = cc.Q_roots[qi];
                double gval = 0, gscale = 0;
                for (int j = 0; j <= dg; j++) {
                    double gj = (double)g[j];
                    double rpow = 1.0;
                    for (int p = 0; p < j; p++) rpow *= r;
                    gval += gj * rpow;
                    gscale += std::abs(gj * rpow);
                }
                bool is_root = (std::abs(gval) < 1e-10) ||
                               (gscale > 0 && std::abs(gval) / gscale < 1e-6);
                if (is_root)
                    cc.sr_q_root_idx[cc.n_sr_roots++] = qi;
            }
        } else {
            cc.has_shared_root = false;
        }
    }

    // ── Cv/Cw: critical-point degeneracies ──
    // Interior v=0 / w=0 (independent of punctures)
    bool has_Cv = check_field_zero_in_tet(gpu_v2.V, cc.Cv_mu);
    bool has_Cw = check_field_zero_in_tet(gpu_v2.W, cc.Cw_mu);
    cc.has_Cv_pos = has_Cv;
    cc.has_Cw_pos = has_Cw;

    // Face/edge/vertex crossings at λ=0 and λ=∞ (from integer P coefficients).
    // V2 excludes Cv1/Cv0/Cw1/Cw0 punctures — detect from polynomials.
    bool has_C2v = false, has_C1v = false, has_C0v = false;
    bool has_C2w = false, has_C1w = false, has_C0w = false;

    // λ=0 crossings: face i when P[i][0]=0 with Q[0]≠0
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
    // λ=∞ crossings: generalized for degree-reduced Q.
    // When deg(Q) < 3, check at actual degree d_Q instead of 3.
    {
        int d_Q = 3;
        while (d_Q > 0 && Q_i128[d_Q] == 0) d_Q--;
        __int128 q_lead = (d_Q >= 0) ? Q_i128[d_Q] : 0;
        if (d_Q >= 1 && q_lead != 0) {
            // Guard: all P[k] must have degree ≤ d_Q
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

    // Emit Cv{d}/Cw{d} tag (most specific wins)
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
    // Only genuine crossings (not pass-through) contribute to T-count.
    // Pass-through test: derivative product < 0 at the edge/vertex.
    {
        // Cv at λ=0: P'[k](0) = P[k][1]
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
                for (int i = 0; i < cc.n_Q_roots; i++) {
                    if (cc.Q_roots[i] > 0.0) break;
                    iv_idx = i + 1;
                }
                if (iv_idx >= 0 && iv_idx < (int)cc.intervals.size()) {
                    cc.intervals[iv_idx].n_pv++;
                    n_face++;
                }
            }
        }
        // Cw at λ→∞: "derivative" analog is P[k][d_Q-1]
        if (has_C0w || has_C1w) {
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
                if (last_iv >= 0) {
                    cc.intervals[last_iv].n_pv++;
                    n_face++;
                }
            }
        }
    }

    // Build category string: T{n}[_(tuple)]_Q{type}
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
        // Compute disc_sign on CPU from unquantized P coefficients (max ~50000
        // at R=20, fits __int128 easily).  GPU disc_sign overflows at R≥20
        // because quantized P_i128 coefficients are ~10^18.
        int disc_sign_cpu[4];
        for (int k = 0; k < 4; k++) {
            __int128 Pk_cpu[4];
            for (int j = 0; j < 4; j++) Pk_cpu[j] = (__int128)llround(P[k][j]);
            disc_sign_cpu[k] = ftk2::discriminant_sign_i128(Pk_cpu);
        }

        bool has_TN = false;   // non-isolated TN (touch from inside, 2 punctures)
        for (int k = 0; k < 4; k++) {
            int degk = effective_degree_i128(P_i128[k], 3);
            if (degk < 2 || disc_sign_cpu[k] != 0)
                continue;

            __int128 Pk_prime[4] = {P_i128[k][1], 2*P_i128[k][2], 3*P_i128[k][3], 0};
            int dk_prime = degk - 1;
            while (dk_prime > 0 && Pk_prime[dk_prime] == 0) dk_prime--;

            __int128 h[4] = {};
            int dh = ftk2::poly_gcd_full_i128(P_i128[k], degk, Pk_prime, dk_prime, h);
            if (dh < 1) continue;

            if (dh == 1) {
                auto eval_at_root_h = [&](__int128* f, int df) -> int {
                    __int128 val = 0;
                    __int128 neg_h0_pow = 1;
                    __int128 h1_pow_desc = 1;
                    for (int i = 0; i < df; i++) h1_pow_desc *= h[1];
                    for (int i = 0; i <= df; i++) {
                        val += f[i] * neg_h0_pow * h1_pow_desc;
                        neg_h0_pow *= (-h[0]);
                        if (i < df) h1_pow_desc /= h[1];
                    }
                    int h1_sign = (h[1] > 0) ? 1 : -1;
                    int factor = (df % 2 == 0) ? 1 : h1_sign;
                    return (val > 0) ? factor : (val < 0) ? -factor : 0;
                };

                // Use Q_i128 directly (from compute_tet_QP_i128)
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

                // Determine isolated vs non-isolated TN via P[k]''(α) · Q(α) sign
                __int128 Pk_pp[2] = {2 * P_i128[k][2], 6 * P_i128[k][3]};
                int deg_pp = ftk2::effective_degree_i128(Pk_pp, 1);
                int spp = 0;
                if (deg_pp > 0 || Pk_pp[0] != 0) {
                    spp = eval_at_root_h(Pk_pp, deg_pp);
                }

                if (spp * sq < 0)
                    ;  // isolated TN (≡ D02): curve touches from outside, already excluded
                else
                    has_TN = true;    // non-isolated: curve touches from inside

                double lam_tn = -(double)h[0] / (double)h[1];
                double bary[3] = {};
                double Qval = Q[0] + Q[1]*lam_tn + Q[2]*lam_tn*lam_tn + Q[3]*lam_tn*lam_tn*lam_tn;
                if (std::abs(Qval) > 1e-20) {
                    static const int face_verts[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
                    for (int fi = 0; fi < 3; fi++) {
                        int j = face_verts[k][fi];
                        double Pj = P[j][0] + P[j][1]*lam_tn + P[j][2]*lam_tn*lam_tn + P[j][3]*lam_tn*lam_tn*lam_tn;
                        bary[fi] = Pj / Qval;
                    }
                }
                if (cc.n_tn < 4) {
                    cc.tn_points[cc.n_tn].face = k;
                    cc.tn_points[cc.n_tn].lambda = lam_tn;
                    for (int fi = 0; fi < 3; fi++)
                        cc.tn_points[cc.n_tn].bary[fi] = bary[fi];
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
    if (n_face == 0 && cc.n_Q_roots == 0 && Q[0] != 0.0) {
        bool inside = true;
        for (int k = 0; k < 4; k++)
            if (P[k][0] / Q[0] <= 0) { inside = false; break; }
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
    printf("\"seed\":%lu,", (unsigned long)cc.gpu.seed);
    printf("\"category\":\"%s\",", cc.category.c_str());
    auto is_waypoint_out = [](const ClassifiedCase::PunctureInfo& pi) {
        if (pi.is_passthrough) return true;
        return (pi.is_edge || pi.is_vertex) &&
               (pi.lambda == 0.0 || std::isinf(pi.lambda));
    };
    int n_face_out = 0;
    for (const auto& pi : cc.punctures)
        if (!is_waypoint_out(pi)) n_face_out++;
    printf("\"n_punctures\":%d,", n_face_out);
    printf("\"n_total\":%d,", (int)cc.punctures.size());
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
        printf(",\"bary\":[%.15g,%.15g,%.15g],\"interval\":%d",
               pi.bary[0], pi.bary[1], pi.bary[2], pi.interval_idx);
        if (pi.is_edge) {
            printf(",\"is_edge\":true,\"tet_edge\":[%d,%d]", pi.tet_edge[0], pi.tet_edge[1]);
            if (pi.is_passthrough)
                printf(",\"is_passthrough\":true");
        }
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

    // TN points
    printf("\"tn_points\":[");
    for (int i = 0; i < cc.n_tn; i++) {
        if (i > 0) printf(",");
        printf("{\"face\":%d,\"lambda\":%.15g,\"bary\":[%.15g,%.15g,%.15g]}",
               cc.tn_points[i].face, cc.tn_points[i].lambda,
               cc.tn_points[i].bary[0], cc.tn_points[i].bary[1], cc.tn_points[i].bary[2]);
    }
    printf("],");

    // Cv/Cw positions (tet barycentric)
    if (cc.has_Cv_pos)
        printf("\"Cv_mu\":[%.15g,%.15g,%.15g,%.15g],",
               cc.Cv_mu[0], cc.Cv_mu[1], cc.Cv_mu[2], cc.Cv_mu[3]);
    else
        printf("\"Cv_mu\":null,");
    if (cc.has_Cw_pos)
        printf("\"Cw_mu\":[%.15g,%.15g,%.15g,%.15g]",
               cc.Cw_mu[0], cc.Cw_mu[1], cc.Cw_mu[2], cc.Cw_mu[3]);
    else
        printf("\"Cw_mu\":null");
    printf("}\n");
}

// ─── Verification ────────────────────────────────────────────────────────────
// Checks invariants of a classified case using exact arithmetic where possible.
// Prints [WARN seed=X] messages for any violations.

static void verify_case(const ClassifiedCase& cc, const double Q[4], const double P[4][4]) {
    uint64_t seed = cc.gpu.seed;
    int n = (int)cc.punctures.size();

    // ── 1. Puncture-interval consistency ──────────────────────────────────
    // Waypoints = edge/vertex at critical λ (0 or ∞), or pass-through edge.
    auto is_waypoint = [](const ClassifiedCase::PunctureInfo& pi) {
        if (pi.is_passthrough) return true;
        return (pi.is_edge || pi.is_vertex) &&
               (pi.lambda == 0.0 || std::isinf(pi.lambda));
    };
    int sum_iv_npv = 0;
    for (const auto& iv : cc.intervals) sum_iv_npv += iv.n_pv;
    // n_face_check: T-count (excludes isolated D00/D01)
    int n_face_check = 0;
    for (const auto& pi : cc.punctures)
        if (!pi.is_D00 && !pi.is_D01) n_face_check++;
    if (sum_iv_npv != n_face_check)
        fprintf(stderr, "[WARN seed=%lu] Σ interval.n_pv=%d ≠ n_face=%d\n",
                (unsigned long)seed, sum_iv_npv, n_face_check);

    for (int i = 0; i < n; i++) {
        const auto& pi = cc.punctures[i];
        int iv = pi.interval_idx;
        if (iv < 0 || iv >= (int)cc.intervals.size()) {
            fprintf(stderr, "[WARN seed=%lu] puncture %d has invalid interval_idx=%d\n",
                    (unsigned long)seed, i, iv);
            continue;
        }
        double lb = cc.intervals[iv].lb, ub = cc.intervals[iv].ub;
        if (!std::isinf(pi.lambda)) {
            if (pi.lambda < lb - 1e-6 || pi.lambda > ub + 1e-6)
                fprintf(stderr, "[WARN seed=%lu] puncture %d λ=%.6g outside interval %d [%.6g, %.6g]\n",
                        (unsigned long)seed, i, pi.lambda, iv, lb, ub);
        }
    }

    // ── 2. Pair completeness ─────────────────────────────────────────────
    // Waypoints (Cv1/Cv0/Cw1/Cw0 at λ=0/∞) are not paired.
    // All other punctures (including D01/D00 at generic λ) must be paired.
    //
    // With SR: each Q-root shared with some P_k allows one unpaired
    // puncture (the curve enters/exits via SR passage instead of face
    // crossing).  The discrepancy (n_face - 2×pairs) must be ≤ the
    // number of Q-roots that are shared with at least one P_k.
    // n_face counts ALL non-waypoint punctures (for pair completeness check)
    int n_face = 0;
    for (const auto& pi : cc.punctures)
        if (!is_waypoint(pi)) n_face++;

    int n_sr_roots = 0;
    if (cc.has_shared_root) {
        // Count Q-roots where at least one P_k vanishes
        for (int ri = 0; ri < cc.n_Q_roots; ri++) {
            double r = cc.Q_roots[ri];
            for (int k = 0; k < 4; k++) {
                double pk_val = P[k][0] + P[k][1]*r + P[k][2]*r*r + P[k][3]*r*r*r;
                double pk_scale = std::max({std::abs(P[k][0]), std::abs(P[k][1]),
                                            std::abs(P[k][2]), std::abs(P[k][3]), 1.0});
                if (std::abs(pk_val) < pk_scale * 1e-6) {
                    n_sr_roots++;
                    break;
                }
            }
        }
    }

    int discrepancy = n_face - 2 * (int)cc.pairs.size();
    if (discrepancy != 0 && discrepancy > n_sr_roots)
        fprintf(stderr, "[WARN seed=%lu] 2×pairs=%d ≠ n_face=%d (n_total=%d, sr_roots=%d)\n",
                (unsigned long)seed, 2*(int)cc.pairs.size(), n_face, n, n_sr_roots);

    // Check each non-waypoint puncture appears in at most one pair
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
        if (is_waypoint(cc.punctures[i])) {
            if (pair_count[i] != 0)
                fprintf(stderr, "[WARN seed=%lu] waypoint puncture %d appears in %d pairs\n",
                        (unsigned long)seed, i, pair_count[i]);
        } else if (pair_count[i] > 1) {
            fprintf(stderr, "[WARN seed=%lu] puncture %d appears in %d pairs (expected ≤1)\n",
                    (unsigned long)seed, i, pair_count[i]);
        }
    }

    // ── 3. Category-data consistency ─────────────────────────────────────
    // T-number should match total puncture count (all interval endpoints)
    {
        std::string cat = cc.category;
        // Extract T number from "T{n}_..." or "T{n}_(..."
        size_t t_pos = cat.find('T');
        if (t_pos != std::string::npos) {
            int t_num = 0;
            size_t i = t_pos + 1;
            while (i < cat.size() && cat[i] >= '0' && cat[i] <= '9')
                t_num = t_num * 10 + (cat[i++] - '0');
            if (t_num != n_face_check)
                fprintf(stderr, "[WARN seed=%lu] T-number=%d but n_face=%d (n_total=%d)\n",
                        (unsigned long)seed, t_num, n_face_check, n);
        }
    }

    // ── 4. Q polynomial roots ────────────────────────────────────────────
    for (int i = 0; i < cc.n_Q_roots; i++) {
        double r = cc.Q_roots[i];
        // For roots near 0, use exact integer evaluation: Q(0) = Q[0]
        if (std::abs(r) < 1e-10) {
            int64_t Q0_int = llround(Q[0]);
            if (Q0_int != 0)
                fprintf(stderr, "[WARN seed=%lu] Q(root[%d]=%.6g): Q[0]=%ld ≠ 0\n",
                        (unsigned long)seed, i, r, (long)Q0_int);
            continue;
        }
        double Qval = Q[0] + r*(Q[1] + r*(Q[2] + r*Q[3]));
        double cond = std::abs(Q[3]);
        double ax = std::abs(r);
        for (int d = 2; d >= 0; --d) cond = cond * ax + std::abs(Q[d]);
        if (std::abs(Qval) > 1e-6 * cond)
            fprintf(stderr, "[WARN seed=%lu] Q(root[%d]=%.6g)=%.6g (cond=%.6g)\n",
                    (unsigned long)seed, i, r, Qval, cond);
    }
    // Roots should be sorted
    for (int i = 0; i + 1 < cc.n_Q_roots; i++)
        if (cc.Q_roots[i] > cc.Q_roots[i+1] + 1e-15)
            fprintf(stderr, "[WARN seed=%lu] Q roots not sorted: root[%d]=%.15g > root[%d]=%.15g\n",
                    (unsigned long)seed, i, cc.Q_roots[i], i+1, cc.Q_roots[i+1]);

    // ── 5. Cv/Cw position validity ──────────────────────────────────────
    if (cc.has_Cv_pos) {
        double sum_cv = 0;
        bool bad = false;
        for (int k = 0; k < 4; k++) {
            if (cc.Cv_mu[k] < -1e-10 || cc.Cv_mu[k] > 1.0 + 1e-10) bad = true;
            sum_cv += cc.Cv_mu[k];
        }
        if (bad || std::abs(sum_cv - 1.0) > 1e-10)
            fprintf(stderr, "[WARN seed=%lu] Cv_mu invalid: [%.6g,%.6g,%.6g,%.6g] sum=%.6g\n",
                    (unsigned long)seed, cc.Cv_mu[0], cc.Cv_mu[1], cc.Cv_mu[2], cc.Cv_mu[3], sum_cv);
        // Cross-check with check_field_zero_in_tet
        bool cv_check = check_field_zero_in_tet(cc.gpu.V);
        if (!cv_check)
            fprintf(stderr, "[WARN seed=%lu] has_Cv_pos=true but check_field_zero_in_tet(V)=false\n",
                    (unsigned long)seed);
    }
    if (cc.has_Cw_pos) {
        double sum_cw = 0;
        bool bad = false;
        for (int k = 0; k < 4; k++) {
            if (cc.Cw_mu[k] < -1e-10 || cc.Cw_mu[k] > 1.0 + 1e-10) bad = true;
            sum_cw += cc.Cw_mu[k];
        }
        if (bad || std::abs(sum_cw - 1.0) > 1e-10)
            fprintf(stderr, "[WARN seed=%lu] Cw_mu invalid: [%.6g,%.6g,%.6g,%.6g] sum=%.6g\n",
                    (unsigned long)seed, cc.Cw_mu[0], cc.Cw_mu[1], cc.Cw_mu[2], cc.Cw_mu[3], sum_cw);
        bool cw_check = check_field_zero_in_tet(cc.gpu.W);
        if (!cw_check)
            fprintf(stderr, "[WARN seed=%lu] has_Cw_pos=true but check_field_zero_in_tet(W)=false\n",
                    (unsigned long)seed);
    }

    // ── 6. SR/ISR verification ─────────────────────────────────────────
    if (cc.has_shared_root) {
        // Verify: resultant(Q, P_k) = 0 for at least one k
        __int128 Qi[4], Pi[4][4];
        for (int j = 0; j < 4; j++) Qi[j] = (__int128)llround(Q[j]);
        for (int a = 0; a < 4; a++)
            for (int j = 0; j < 4; j++) Pi[a][j] = (__int128)llround(P[a][j]);
        if (!has_shared_root_resultant(Qi, Pi))
            fprintf(stderr, "[WARN seed=%lu] has_shared_root=true but resultant check fails\n",
                    (unsigned long)seed);

        // Cross-check ISR flag: gcd(Q, P_k) ≥ 2 for some k
        bool verified_isr = false;
        for (int k = 0; k < 4; k++) {
            int gd = poly_gcd_degree_i128(Q, 3, P[k], 3);
            if (gd >= 2) { verified_isr = true; break; }
        }
        if (cc.has_non_isolated_sr && !verified_isr)
            fprintf(stderr, "[WARN seed=%lu] ISR tag but no gcd degree ≥ 2\n",
                    (unsigned long)seed);
        if (!cc.has_non_isolated_sr && verified_isr)
            fprintf(stderr, "[WARN seed=%lu] gcd degree ≥ 2 found but ISR tag missing\n",
                    (unsigned long)seed);
    }

    // ── 7. Edge/vertex tag hierarchy ────────────────────────────────────
    for (int i = 0; i < n; i++) {
        const auto& pi = cc.punctures[i];
        bool is_lam0 = (pi.lambda == 0.0);
        bool is_laminf = std::isinf(pi.lambda);
        // D01 at λ=0 should be Cv1, not D01
        if (pi.is_edge && is_lam0) {
            if (cc.category.find("D01") != std::string::npos &&
                cc.category.find("Cv1") == std::string::npos)
                fprintf(stderr, "[WARN seed=%lu] edge puncture at λ=0 tagged D01 without Cv1\n",
                        (unsigned long)seed);
        }
        // D00 at λ=0 should be Cv0, not D00
        if (pi.is_vertex && is_lam0) {
            if (cc.category.find("D00") != std::string::npos &&
                cc.category.find("Cv0") == std::string::npos)
                fprintf(stderr, "[WARN seed=%lu] vertex puncture at λ=0 tagged D00 without Cv0\n",
                        (unsigned long)seed);
        }
        // Same for λ=∞
        if (pi.is_edge && is_laminf) {
            if (cc.category.find("D01") != std::string::npos &&
                cc.category.find("Cw1") == std::string::npos)
                fprintf(stderr, "[WARN seed=%lu] edge puncture at λ=∞ tagged D01 without Cw1\n",
                        (unsigned long)seed);
        }
        if (pi.is_vertex && is_laminf) {
            if (cc.category.find("D00") != std::string::npos &&
                cc.category.find("Cw0") == std::string::npos)
                fprintf(stderr, "[WARN seed=%lu] vertex puncture at λ=∞ tagged D00 without Cw0\n",
                        (unsigned long)seed);
        }
    }

    // ── 8. Interval parity ──────────────────────────────────────────────
    // Each interval's n_pv (face-interior punctures only) should be even
    // after accounting for infinity merging and SR pass-through.
    {
        bool has_odd_inner = false;
        int n_iv = (int)cc.intervals.size();
        for (int i = 0; i < n_iv; i++) {
            if (cc.intervals[i].n_pv % 2 != 0) {
                // Outer intervals (first/last) may merge through infinity
                if (i == 0 || i == n_iv - 1) continue;
                has_odd_inner = true;
            }
        }
        // Check if outer intervals have matching odd parity (they merge)
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

static TetCaseGPU generate_tet_from_seed(uint64_t global_id, uint64_t base_seed, int R) {
    TetCaseGPU tc;

    if (global_id == 0) {
        // Seed 0: analytically constructed bubble case (T0_Q2-_Cv_B)
        // V∥W never holds inside tet, but PV curve forms a closed loop.
        static const int bubble_V[4][3] = {{2,3,-1},{-1,-2,-1},{0,-1,2},{-2,-2,0}};
        static const int bubble_W[4][3] = {{3,0,3},{-1,0,-1},{-3,-2,-1},{0,3,-3}};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++) {
                tc.V[i][j] = bubble_V[i][j];
                tc.W[i][j] = bubble_W[i][j];
            }
    } else if (global_id == 1) {
        // Seed 1: non-isolated SR case (Q3+ with gcd(Q,P_2) deg 2).
        // Found by random search at R=5.
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
        uint64_t tet_fourth = (uint64_t)fi;
        tc.face[fi] = solve_pv_triangle_device(Vf, Wf, indices, tet_fourth);
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
    const char* vw_file = nullptr;
    int pv_version = 1;  // 1=v1 (default), 2=v2, 3=both

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
        else if (strcmp(argv[i], "--vw-file") == 0 && i + 1 < argc)
            vw_file = argv[++i];
        else if (strcmp(argv[i], "--max-cases") == 0 && i + 1 < argc)
            max_cases = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pv-version") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if (strcmp(v, "1") == 0) pv_version = 1;
            else if (strcmp(v, "2") == 0) pv_version = 2;
            else if (strcmp(v, "both") == 0) pv_version = 3;
            else { fprintf(stderr, "Unknown --pv-version: %s\n", v); return 1; }
        }
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
            fprintf(stderr, "  --pv-version 1|2|both  PV algorithm version (default: 1)\n");
            return 0;
        }
    }
    fprintf(stderr, "PV version: %s\n", pv_version == 1 ? "v1" : pv_version == 2 ? "v2" : "both");

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
            TetCaseGPU tc;
            memset(&tc, 0, sizeof(tc));
            int vals[25];
            int n = sscanf(linebuf, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                &vals[0],&vals[1],&vals[2],&vals[3],&vals[4],&vals[5],
                &vals[6],&vals[7],&vals[8],&vals[9],&vals[10],&vals[11],
                &vals[12],&vals[13],&vals[14],&vals[15],&vals[16],&vals[17],
                &vals[18],&vals[19],&vals[20],&vals[21],&vals[22],&vals[23],
                &vals[24]);
            int vw_offset = 0;
            if (n == 25) {
                // First value is seed, remaining 24 are V,W
                tc.seed = vals[0];
                vw_offset = 1;
            } else if (n == 24) {
                tc.seed = line_num;
                vw_offset = 0;
            } else {
                continue;
            }
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++) {
                    tc.V[i][j] = vals[vw_offset+i*3+j];
                    tc.W[i][j] = vals[vw_offset+12+i*3+j];
                }
            // Solve PV on 4 faces
            static const int fv[4][3] = {{1,3,2},{0,2,3},{0,3,1},{0,1,2}};
            tc.total_punctures = 0;
            for (int fi = 0; fi < 4; fi++) {
                int Vf[3][3], Wf[3][3];
                for (int vi = 0; vi < 3; vi++)
                    for (int ci = 0; ci < 3; ci++) {
                        Vf[vi][ci] = tc.V[fv[fi][vi]][ci];
                        Wf[vi][ci] = tc.W[fv[fi][vi]][ci];
                    }
                uint64_t indices[3] = {
                    (uint64_t)fv[fi][0], (uint64_t)fv[fi][1], (uint64_t)fv[fi][2]
                };
                uint64_t tet_fourth = (uint64_t)fi;
                tc.face[fi] = solve_pv_triangle_device(Vf, Wf, indices, tet_fourth);
                if (tc.face[fi].count > 0 && tc.face[fi].count < INT_MAX)
                    tc.total_punctures += tc.face[fi].count;
            }
            ClassifiedCase cc;
            if (pv_version == 2 || pv_version == 3) {
                // ExactPV2 path: pure-integer solver + classifier
                __int128 Q_i128[4], P_i128[4][4];
                ftk2::compute_tet_QP_i128(tc.V, tc.W, Q_i128, P_i128);
                ftk2::ExactPV2Result v2 = ftk2::solve_pv_tet_v2(Q_i128, P_i128);
                TetCaseV2GPU tv2;
                memcpy(tv2.V, tc.V, sizeof(tc.V));
                memcpy(tv2.W, tc.W, sizeof(tc.W));
                tv2.v2 = v2;
                for (int k = 0; k < 4; k++)
                    tv2.disc_sign[k] = ftk2::discriminant_sign_i128(P_i128[k]);
                tv2.seed = tc.seed;
                cc = classify_case_v2(tv2);
            } else {
                cc = classify_case(tc);
            }
            int np = (int)cc.punctures.size();
            if (min_punctures == 0) {
                // Print ALL cases when --min-punctures 0
                fprintf(stderr, "  line %d: %s (%d punctures)\n",
                        line_num, cc.category.c_str(), np);
                print_json(cc);
            } else if (np > best_punct) {
                best_punct = np;
                fprintf(stderr, "  line %d: %s (%d punctures, raw=%d)\n",
                        line_num, cc.category.c_str(), np,
                        cc.gpu.total_punctures);
                print_json(cc);
            }
            if (np >= 12) {
                fprintf(stderr, "*** T12 FOUND at line %d! ***\n", line_num);
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
            TetCaseGPU tc = generate_tet_from_seed(s, base_seed, R);
            ClassifiedCase cc = classify_case(tc);
            verify_case(cc, cc.Q_coeffs, cc.P_coeffs);
            print_json(cc);
            fprintf(stderr, "  seed=%lu: %s (%d punctures, %d pairs)\n",
                    (unsigned long)s, cc.category.c_str(),
                    (int)cc.punctures.size(), (int)cc.pairs.size());

            // ExactPV2 comparison
            if (pv_version == 2 || pv_version == 3) {
                __int128 Q_i128[4], P_i128[4][4];
                ftk2::compute_tet_QP_i128(tc.V, tc.W, Q_i128, P_i128);
                ftk2::ExactPV2Result v2 = ftk2::solve_pv_tet_v2(Q_i128, P_i128);
                fprintf(stderr, "    [v2] %d punctures, %d pairs, passthrough=%d (deg %d)\n",
                        v2.n_punctures, v2.n_pairs, v2.has_passthrough, v2.passthrough_deg);

                if (pv_version == 3) {
                    // Compare: v1 pairs vs v2 pairs count
                    int v1_pairs = (int)cc.pairs.size();
                    if (v1_pairs != v2.n_pairs) {
                        fprintf(stderr, "    [MISMATCH] v1 pairs=%d, v2 pairs=%d\n",
                                v1_pairs, v2.n_pairs);
                        for (int pi = 0; pi < v2.n_punctures; pi++) {
                            auto& p = v2.punctures[pi];
                            fprintf(stderr, "      v2 punct[%d]: face=%d root_idx=%d q_int=%d edge=%d\n",
                                    pi, p.face, p.root_idx, p.q_interval, p.is_edge);
                        }
                        for (int pi = 0; pi < v2.n_pairs; pi++) {
                            fprintf(stderr, "      v2 pair[%d]: %d-%d\n",
                                    pi, v2.pairs[pi].a, v2.pairs[pi].b);
                        }
                    } else
                        fprintf(stderr, "    [MATCH] %d pairs\n", v1_pairs);
                }
            }
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
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    // V1 GPU buffer
    TetCaseGPU* d_output_v1 = nullptr;
    // V2 GPU buffer
    TetCaseV2GPU* d_output_v2 = nullptr;

    if (pv_version == 2) {
        CUDA_CHECK(cudaMalloc(&d_output_v2, gpu_max_output * sizeof(TetCaseV2GPU)));
        // Set stack size for signs_at_roots_i128 recursion
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
    } else {
        CUDA_CHECK(cudaMalloc(&d_output_v1, gpu_max_output * sizeof(TetCaseGPU)));
    }

    // Category histogram
    std::map<std::string, int> category_counts;
    // Store one representative per category
    std::map<std::string, ClassifiedCase> representatives;
    int total_found = 0;

    int num_batches = (num_tets + batch_size - 1) / batch_size;
    int block_size = (pv_version == 2) ? 128 : 256;  // lower block size for v2 (high register pressure)

    for (int batch = 0; batch < num_batches; batch++) {
        int this_batch = std::min(batch_size, num_tets - batch * batch_size);
        uint64_t batch_offset = (uint64_t)batch * batch_size;

        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

        int grid_size = (this_batch + block_size - 1) / block_size;

        if (pv_version == 2) {
            tet_case_finder_v2_kernel<<<grid_size, block_size>>>(
                d_output_v2, d_count, gpu_max_output, min_punctures,
                R, base_seed, batch_offset);
        } else {
            tet_case_finder_kernel<<<grid_size, block_size>>>(
                d_output_v1, d_count, gpu_max_output, min_punctures,
                R, base_seed, batch_offset);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download count
        int h_count;
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_count > gpu_max_output) h_count = gpu_max_output;

        if (h_count == 0) continue;

        if (pv_version == 2) {
            // Download v2 results
            std::vector<TetCaseV2GPU> h_v2(h_count);
            CUDA_CHECK(cudaMemcpy(h_v2.data(), d_output_v2,
                                  h_count * sizeof(TetCaseV2GPU), cudaMemcpyDeviceToHost));

            // Pure-integer classification from ExactPV2Result (no re-solve)
            for (int i = 0; i < h_count && total_found < max_cases; i++) {
                ClassifiedCase cc = classify_case_v2(h_v2[i]);
                verify_case(cc, cc.Q_coeffs, cc.P_coeffs);
                category_counts[cc.category]++;
                print_json(cc);
                total_found++;

                if (representatives.find(cc.category) == representatives.end())
                    representatives[cc.category] = cc;
            }
        } else {
            // Download v1 results
            std::vector<TetCaseGPU> h_cases(h_count);
            CUDA_CHECK(cudaMemcpy(h_cases.data(), d_output_v1,
                                  h_count * sizeof(TetCaseGPU), cudaMemcpyDeviceToHost));

            // CPU classification
            for (int i = 0; i < h_count && total_found < max_cases; i++) {
                ClassifiedCase cc = classify_case(h_cases[i]);
                verify_case(cc, cc.Q_coeffs, cc.P_coeffs);
                category_counts[cc.category]++;

                // Print JSON to stdout
                print_json(cc);
                total_found++;

                // Store first representative of each category
                if (representatives.find(cc.category) == representatives.end())
                    representatives[cc.category] = cc;

                // ExactPV2 comparison
                if (pv_version == 3) {
                    __int128 Q_i128[4], P_i128[4][4];
                    ftk2::compute_tet_QP_i128(h_cases[i].V, h_cases[i].W, Q_i128, P_i128);
                    ftk2::ExactPV2Result v2 = ftk2::solve_pv_tet_v2(Q_i128, P_i128);

                    int v1_pairs = (int)cc.pairs.size();
                    if (v1_pairs != v2.n_pairs)
                        fprintf(stderr, "    [MISMATCH] seed=%lu v1=%d v2=%d (%s)\n",
                                (unsigned long)h_cases[i].seed, v1_pairs, v2.n_pairs,
                                cc.category.c_str());
                }
            }
        }

        fprintf(stderr, "Batch %d/%d: %d hits (%d total), %d categories\n",
                batch + 1, num_batches, h_count, total_found,
                (int)category_counts.size());

        if (total_found >= max_cases) break;
    }

    if (d_output_v1) CUDA_CHECK(cudaFree(d_output_v1));
    if (d_output_v2) CUDA_CHECK(cudaFree(d_output_v2));
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
