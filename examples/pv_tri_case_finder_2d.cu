// GPU-accelerated single-triangle PV case finder for 2D vector fields.
//
// Architecture (mirrors 3D tet case finder):
//   - GPU: random triangle generation + compute_tri_QP_2d + solve_pv_tri_2d
//          All pure-integer __int128 arithmetic, NO floats, NO Sturm sequences.
//   - CPU: classification + JSON output
//
// Usage:
//   ./ftk2_pv_tri_case_finder_2d [--min-punctures N] [--num-tris M] [--range R]
//                                 [--seed S] [--max-cases C] > cases.jsonl

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

#include <ftk2/numeric/pv_tri_classify_2d.hpp>

// ─── Device-side LCG random number generator ────────────────────────────────
__device__ uint32_t lcg_next_2d(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ int rand_int_dev_2d(uint32_t& state, int R) {
    uint32_t r = lcg_next_2d(state);
    return (int)(r % (2 * R + 1)) - R;
}

// ─── Pure-integer GPU extraction kernel ─────────────────────────────────────
// One thread per random triangle.  Uses pure-integer solve_pv_tri_2d on GPU.
// Same pattern as 3D: tet_case_finder_v2_kernel calls solve_pv_tet_v2 on GPU.
__global__ void tri_case_finder_2d_kernel(
    TriCaseV2GPU* output,
    int*          output_count,
    int           max_output,
    int           min_punctures,
    int           R,
    uint64_t      base_seed,
    uint64_t      batch_offset)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t global_id = batch_offset + tid;

    uint32_t state = (uint32_t)(global_id ^ (base_seed * 2654435761ULL));
    for (int i = 0; i < 4; i++) lcg_next_2d(state);

    // Generate random 2D integer fields: V[3][2], W[3][2]
    int V[3][2], W[3][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            V[i][j] = rand_int_dev_2d(state, R);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            W[i][j] = rand_int_dev_2d(state, R);

    // Compute Q, P polynomials (pure integer, degree ≤ 2)
    __int128 V128[3][2], W128[3][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            V128[i][j] = (__int128)V[i][j];
            W128[i][j] = (__int128)W[i][j];
        }
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V128, W128, Q, P);

    // Run pure-integer solver (same approach as solve_pv_tet_v2 in 3D)
    ExactPV2Result2D v2 = solve_pv_tri_2d(Q, P);

    // Filter: only keep cases with enough punctures
    if (v2.n_punctures >= min_punctures) {
        int idx = atomicAdd(output_count, 1);
        if (idx < max_output) {
            TriCaseV2GPU& out = output[idx];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++) {
                    out.V[i][j] = V[i][j];
                    out.W[i][j] = W[i][j];
                }
            out.v2 = v2;
            // Discriminant signs of P[k]
            for (int k = 0; k < 3; k++) {
                int degPk = effective_degree_i128(P[k], 2);
                if (degPk < 2) {
                    out.disc_sign[k] = 0;
                } else {
                    __int128 d = P[k][1]*P[k][1] - 4*P[k][2]*P[k][0];
                    out.disc_sign[k] = (d > 0) ? 1 : (d < 0) ? -1 : 0;
                }
            }
            out.seed = global_id;
        }
    }
}

// ─── CPU-only seed replay ───────────────────────────────────────────────────
static TriCaseV2GPU generate_tri_from_seed(uint64_t seed, uint64_t base_seed, int R) {
    TriCaseV2GPU tv2;
    memset(&tv2, 0, sizeof(tv2));
    uint32_t state = (uint32_t)(seed ^ (base_seed * 2654435761ULL));
    for (int i = 0; i < 4; i++)
        state = state * 1664525u + 1013904223u;
    auto rand_int = [&]() -> int {
        state = state * 1664525u + 1013904223u;
        return (int)(state % (2 * R + 1)) - R;
    };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            tv2.V[i][j] = rand_int();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            tv2.W[i][j] = rand_int();

    __int128 V128[3][2], W128[3][2];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            V128[i][j] = (__int128)tv2.V[i][j];
            W128[i][j] = (__int128)tv2.W[i][j];
        }
    __int128 Q[3], P[3][3];
    compute_tri_QP_2d(V128, W128, Q, P);
    tv2.v2 = solve_pv_tri_2d(Q, P);
    for (int k = 0; k < 3; k++) {
        int degPk = effective_degree_i128(P[k], 2);
        if (degPk < 2)
            tv2.disc_sign[k] = 0;
        else {
            __int128 d = P[k][1]*P[k][1] - 4*P[k][2]*P[k][0];
            tv2.disc_sign[k] = (d > 0) ? 1 : (d < 0) ? -1 : 0;
        }
    }
    tv2.seed = seed;
    return tv2;
}

static std::vector<uint64_t> parse_seeds(const char* s) {
    std::vector<uint64_t> seeds;
    const char* p = s;
    while (*p) {
        char* end;
        uint64_t v = strtoull(p, &end, 10);
        seeds.push_back(v);
        if (*end == ',') end++;
        p = end;
        if (p == s) break;
        s = p;
    }
    return seeds;
}

static void verify_case(const ClassifiedCase2D& cc) {
    if (cc.total_punctures % 2 != 0) {
        bool has_waypoint = cc.has_Cv || cc.has_Cw;
        if (!has_waypoint) {
            fprintf(stderr, "  [BUG] seed=%lu: bare odd T-count %d! category=%s\n",
                    (unsigned long)cc.seed, cc.total_punctures, cc.category.c_str());
        }
    }
}

int main(int argc, char** argv)
{
    int min_punctures = 0;
    int num_tris = 100000000;
    int R = 20;
    uint64_t base_seed = 42;
    int max_cases = 100000;
    int batch_size = 10000000;
    const char* seeds_arg = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min-punctures") == 0 && i + 1 < argc)
            min_punctures = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-tris") == 0 && i + 1 < argc)
            num_tris = atoi(argv[++i]);
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
            fprintf(stderr, "  --min-punctures N   Minimum punctures per triangle (default: 0)\n");
            fprintf(stderr, "  --num-tris M        Total random triangles to try (default: 100M)\n");
            fprintf(stderr, "  --range R           Integer field range [-R, R] (default: 20)\n");
            fprintf(stderr, "  --seed S            Base random seed (default: 42)\n");
            fprintf(stderr, "  --seeds S1,S2,...   Replay specific seeds on CPU\n");
            fprintf(stderr, "  --max-cases C       Max output cases (default: 100000)\n");
            fprintf(stderr, "  --batch-size B      GPU batch size (default: 10M)\n");
            return 0;
        }
    }

    // ─── Seeds mode ─────────────────────────────────────────────────────
    if (seeds_arg) {
        auto seeds = parse_seeds(seeds_arg);
        fprintf(stderr, "Seeds mode: replaying %d seeds on CPU (R=%d, base_seed=%lu)\n",
                (int)seeds.size(), R, (unsigned long)base_seed);
        for (uint64_t s : seeds) {
            TriCaseV2GPU tv2 = generate_tri_from_seed(s, base_seed, R);
            ClassifiedCase2D cc = classify_case_v2_2d(tv2);
            verify_case(cc);
            print_json_2d(stdout, cc);
            fprintf(stderr, "  seed=%lu: %s (%d punctures, %d pairs)\n",
                    (unsigned long)s, cc.category.c_str(),
                    (int)cc.punctures.size(), (int)cc.pairs.size());
        }
        return 0;
    }

    // ─── GPU mode ───────────────────────────────────────────────────────
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    fprintf(stderr, "GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    fprintf(stderr, "Parameters: num_tris=%d, min_punctures=%d, range=%d, seed=%lu\n",
            num_tris, min_punctures, R, (unsigned long)base_seed);

    int gpu_max_output = std::min(max_cases * 2, 2000000);
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    TriCaseV2GPU* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, gpu_max_output * sizeof(TriCaseV2GPU)));
    // signs_at_roots_i128 recurses with __int128 arrays on stack;
    // 64KB per thread handles depth-20 guard comfortably.
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 65536));

    std::map<std::string, int> category_counts;
    std::map<std::string, ClassifiedCase2D> representatives;
    int total_found = 0;
    int total_odd_T = 0;

    int num_batches = (num_tris + batch_size - 1) / batch_size;
    int block_size = 128;

    for (int batch = 0; batch < num_batches; batch++) {
        int this_batch = std::min(batch_size, num_tris - batch * batch_size);
        uint64_t batch_offset = (uint64_t)batch * batch_size;

        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
        int grid_size = (this_batch + block_size - 1) / block_size;

        tri_case_finder_2d_kernel<<<grid_size, block_size>>>(
            d_output, d_count, gpu_max_output, min_punctures,
            R, base_seed, batch_offset);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_count;
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_count > gpu_max_output) h_count = gpu_max_output;
        if (h_count == 0) continue;

        std::vector<TriCaseV2GPU> h_results(h_count);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_output,
                              h_count * sizeof(TriCaseV2GPU), cudaMemcpyDeviceToHost));

        for (int i = 0; i < h_count && total_found < max_cases; i++) {
            ClassifiedCase2D cc = classify_case_v2_2d(h_results[i]);
            verify_case(cc);
            if (cc.total_punctures % 2 != 0 && !cc.has_Cv && !cc.has_Cw)
                total_odd_T++;

            category_counts[cc.category]++;
            print_json_2d(stdout, cc);
            total_found++;

            if (representatives.find(cc.category) == representatives.end())
                representatives[cc.category] = cc;
        }

        fprintf(stderr, "Batch %d/%d: %d hits (%d total), %d categories, %d odd-T\n",
                batch + 1, num_batches, h_count, total_found,
                (int)category_counts.size(), total_odd_T);

        if (total_found >= max_cases) break;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_count));

    // Summary
    fprintf(stderr, "\n=== Category Summary ===\n");
    for (auto& [cat, cnt] : category_counts)
        fprintf(stderr, "  %-35s %d\n", cat.c_str(), cnt);
    fprintf(stderr, "Total: %d cases in %d categories, %d bare odd-T\n",
            total_found, (int)category_counts.size(), total_odd_T);

    return 0;
}
