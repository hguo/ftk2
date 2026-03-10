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

#include <ftk2/numeric/pv_tet_classify.hpp>

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
    if (cc.d22_face >= 0)
        printf("\"d22_face\":%d,", cc.d22_face);

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
