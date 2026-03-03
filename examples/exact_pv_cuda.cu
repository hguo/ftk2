// Standalone CUDA kernel for exact PV extraction + CPU stitching.
//
// Architecture: GPU extracts punctures (up to 3 per triangle),
// CPU performs stitching and curve tracing (graph-based, not GPU-suitable).

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/device_mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <ftk2/core/cuda_engine.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <chrono>

using namespace ftk2;

// ─── CUDA Error Check ─────────────────────────────────────────────────────────
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                      << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

// ─── Device-side puncture output element ──────────────────────────────────────
struct DevicePuncture {
    Simplex simplex;
    float barycentric[3];
    float scalar;  // lambda
};

// ─── 12 unique triangle types per 3D hypercube (Freudenthal triangulation) ──
// Each triangle includes the hypercube base vertex {0,0,0} as one of its
// three vertices.  Every triangle in the mesh is owned by exactly one
// hypercube, so iterating (hypercube × 12 types) visits each triangle once.
__constant__ int d_unit_triangles[12][3][3] = {
    {{0,0,0},{0,0,1},{0,1,1}},    // type  0
    {{0,0,0},{0,0,1},{1,0,1}},    // type  1
    {{0,0,0},{0,0,1},{1,1,1}},    // type  2
    {{0,0,0},{0,1,0},{0,1,1}},    // type  3
    {{0,0,0},{0,1,0},{1,1,0}},    // type  4
    {{0,0,0},{0,1,0},{1,1,1}},    // type  5
    {{0,0,0},{0,1,1},{1,1,1}},    // type  6
    {{0,0,0},{1,0,0},{1,0,1}},    // type  7
    {{0,0,0},{1,0,0},{1,1,0}},    // type  8
    {{0,0,0},{1,0,0},{1,1,1}},    // type  9
    {{0,0,0},{1,0,1},{1,1,1}},    // type 10
    {{0,0,0},{1,1,0},{1,1,1}}     // type 11
};

// ─── Extraction kernel ───────────────────────────────────────────────────────
// One thread per (hypercube, triangle_type) pair.  Total threads =
// n_hypercubes × 12.  Each triangle is processed exactly once.
__global__ void pv_extraction_kernel(
    RegularSimplicialMeshDevice mesh,
    CudaDataView<double> data_view,  // shape [6, nx, ny, nz]
    DevicePuncture* output,
    int* output_count,
    int max_output)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_hc = mesh.get_num_hypercubes();
    if (tid >= n_hc * 12) return;

    uint64_t hc_idx = tid / 12;
    int type = (int)(tid % 12);

    // Decode hypercube index to base coordinates
    uint64_t base[3];
    uint64_t temp = hc_idx;
    for (int i = 0; i < 3; ++i) {
        base[i] = temp % (mesh.local_dims[i] - 1) + mesh.offset[i];
        temp /= (mesh.local_dims[i] - 1);
    }

    // Build triangle vertices from lookup table
    Simplex tri;
    tri.dimension = 2;
    uint64_t v_coords[3][4];
    for (int vi = 0; vi < 3; ++vi) {
        v_coords[vi][0] = base[0] + d_unit_triangles[type][vi][0];
        v_coords[vi][1] = base[1] + d_unit_triangles[type][vi][1];
        v_coords[vi][2] = base[2] + d_unit_triangles[type][vi][2];
        v_coords[vi][3] = 0;
        tri.vertices[vi] = mesh.coords_to_id(v_coords[vi]);
    }
    tri.sort_vertices();

    // Load field values at triangle vertices
    double V[3][3], W[3][3];
    for (int vi = 0; vi < 3; ++vi) {
        for (int c = 0; c < 3; ++c) {
            V[vi][c] = data_view.f(c, v_coords[vi][0], v_coords[vi][1], v_coords[vi][2]);
            W[vi][c] = data_view.f(c + 3, v_coords[vi][0], v_coords[vi][1], v_coords[vi][2]);
        }
    }

    // Solve for punctures
    PunctureResult pr = solve_pv_triangle_device(V, W, tri.vertices);

    // Store results
    if (pr.count > 0 && pr.count < INT_MAX) {
        for (int k = 0; k < pr.count; ++k) {
            int idx = atomicAdd(output_count, 1);
            if (idx < max_output) {
                output[idx].simplex = tri;
                output[idx].barycentric[0] = (float)pr.pts[k].barycentric[0];
                output[idx].barycentric[1] = (float)pr.pts[k].barycentric[1];
                output[idx].barycentric[2] = (float)pr.pts[k].barycentric[2];
                output[idx].scalar = (float)pr.pts[k].lambda;
            }
        }
    }
}

// ─── Host driver ──────────────────────────────────────────────────────────────
struct PunctureConnection {
    int puncture1_idx, puncture2_idx;
    uint64_t tet_id;
};

// Sturm-based pass-through detection (same as CPU version)
static bool is_passthrough_sturm(const __int128 P_i128[4][4],
                                 double lo, double hi, int n_q_roots)
{
    for (int k = 0; k < 4; ++k) {
        double P_d[4];
        for (int j = 0; j <= 3; ++j) P_d[j] = (double)P_i128[k][j];
        SturmSeqDouble seq;
        build_sturm_double(P_d, seq);
        int pk_count = sturm_count_at(seq, lo) - sturm_count_at(seq, hi);
        if (pk_count >= n_q_roots) return true;
    }
    return false;
}

// Combinatorial stitching (same as CPU version)
static void stitch_ambiguous_tet(
    const std::vector<int>&    tet_punctures,
    const std::vector<DevicePuncture>& punctures,
    const __int128             Q_i128[4],
    const __int128             P_i128[4][4],
    uint64_t                   tet_id,
    std::vector<PunctureConnection>& connections)
{
    int n = (int)tet_punctures.size();

    std::vector<std::pair<double,int>> lam_idx;
    lam_idx.reserve(n);
    for (int p : tet_punctures)
        lam_idx.push_back({(double)punctures[p].scalar, p});
    std::sort(lam_idx.begin(), lam_idx.end());

    double Q_d[4];
    for (int k = 0; k <= 3; ++k) Q_d[k] = (double)Q_i128[k];
    int degQ = 3;
    while (degQ > 0 && Q_d[degQ] == 0.0) --degQ;

    if (degQ == 0) {
        for (int j = 0; j + 1 < n; j += 2)
            connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
        return;
    }

    SturmSeqDouble Q_seq;
    build_sturm_double(Q_d, Q_seq);

    std::vector<int> qi(n);
    for (int i = 0; i < n; ++i) {
        double lam = lam_idx[i].first;
        auto [count, cert] = sturm_count_at_certified(Q_seq, lam);
        if (!cert) {
            double delta = 4.0 * std::numeric_limits<double>::epsilon()
                         * std::max(1.0, std::abs(lam));
            count = sturm_count_at(Q_seq, lam + delta);
        }
        qi[i] = count;
    }

    std::vector<int> qi_eff(n);
    qi_eff[0] = 0;
    for (int i = 1; i < n; ++i) {
        if (qi[i] == qi[i-1]) {
            qi_eff[i] = qi_eff[i-1];
        } else {
            int n_q_roots = std::abs(qi[i-1] - qi[i]);
            if (is_passthrough_sturm(P_i128, lam_idx[i-1].first,
                                     lam_idx[i].first, n_q_roots)) {
                qi_eff[i] = qi_eff[i-1];
            } else {
                qi_eff[i] = qi_eff[i-1] + 1;
            }
        }
    }

    int group_start = 0;
    for (int i = 1; i <= n; ++i) {
        if (i == n || qi_eff[i] != qi_eff[i-1]) {
            for (int j = group_start; j + 1 < i; j += 2)
                connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
            group_start = i;
        }
    }
}

// ─── Field evaluation type ────────────────────────────────────────────────────
using FieldEval = std::function<std::array<double,6>(double,double,double)>;

struct TestCase {
    std::string name;
    std::string description;
    FieldEval   eval;
};

// ─── Run one test case ────────────────────────────────────────────────────────
static void run_test_case(const TestCase& tc, int N) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Build ndarray from field eval
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)N, (uint64_t)N, (uint64_t)N});
    ftk::ndarray<double> uv({6, (size_t)N, (size_t)N, (size_t)N});
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                auto fv = tc.eval((double)x, (double)y, (double)z);
                for (int c = 0; c < 6; ++c)
                    uv.f(c, x, y, z) = fv[c];
            }

    // === GPU Extraction ===
    auto t_gpu_start = std::chrono::high_resolution_clock::now();

    // Setup device mesh
    RegularSimplicialMeshDevice d_mesh;
    d_mesh.ndims = 3;
    auto l_dims = mesh->get_local_dims();
    auto off = mesh->get_offset();
    auto g_dims = mesh->get_global_dims();
    for (int i = 0; i < 4; ++i) {
        d_mesh.local_dims[i] = (i < 3) ? l_dims[i] : 1;
        d_mesh.offset[i] = (i < 3) ? off[i] : 0;
        d_mesh.global_dims[i] = (i < 3) ? g_dims[i] : 1;
    }

    // Upload field data
    uv.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
    CudaDataView<double> h_view;
    h_view.data = static_cast<double*>(uv.get_devptr());
    h_view.ndims = 4;
    auto lattice = uv.get_lattice();
    for (int i = 0; i < 4; ++i) {
        h_view.dims[i] = (i < uv.nd()) ? uv.dimf(i) : 1;
        h_view.s[i] = (i < (int)uv.nd()) ? lattice.prod_[uv.nd() - 1 - i] : 0;
    }

    // Allocate output buffer
    int max_output = 1 << 20;  // 1M elements
    DevicePuncture* d_output;
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_output, max_output * sizeof(DevicePuncture)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    // Launch kernel: one thread per (hypercube, triangle_type), 12 types per hypercube
    uint64_t n_hc = 1;
    for (int i = 0; i < 3; ++i) n_hc *= (d_mesh.local_dims[i] - 1);
    uint64_t total_threads = n_hc * 12;
    int block_size = 256;
    int grid_size = ((int)total_threads + block_size - 1) / block_size;
    pv_extraction_kernel<<<grid_size, block_size>>>(d_mesh, h_view, d_output, d_count, max_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    int h_count;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_count > max_output) {
        std::cerr << "WARNING: output buffer overflow (" << h_count << " > " << max_output << ")\n";
        h_count = max_output;
    }

    std::vector<DevicePuncture> h_punctures(h_count);
    if (h_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_punctures.data(), d_output,
                              h_count * sizeof(DevicePuncture), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_count));

    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();

    std::cout << "  GPU extraction: " << h_count << " punctures in " << gpu_ms << " ms\n";

    // === G2 global deduplication ===
    {
        std::map<uint64_t, std::vector<int>> vtx_g2;
        const float bary_tol = 1e-5f;
        for (int p = 0; p < h_count; ++p) {
            const float* bc = h_punctures[p].barycentric;
            int n_zero = 0, m = -1;
            for (int k = 0; k < 3; k++) {
                if (std::abs(bc[k]) < bary_tol) n_zero++;
                else m = k;
            }
            if (n_zero == 2 && m >= 0)
                vtx_g2[h_punctures[p].simplex.vertices[m]].push_back(p);
        }
        std::set<int> to_remove;
        for (auto& [vid, puncs] : vtx_g2) {
            if (puncs.size() <= 1) continue;
            int canonical = puncs[0];
            std::vector<uint64_t> can_f(h_punctures[canonical].simplex.vertices,
                                        h_punctures[canonical].simplex.vertices + 3);
            std::sort(can_f.begin(), can_f.end());
            for (int p : puncs) {
                std::vector<uint64_t> f(h_punctures[p].simplex.vertices,
                                        h_punctures[p].simplex.vertices + 3);
                std::sort(f.begin(), f.end());
                if (f < can_f) { canonical = p; can_f = f; }
            }
            for (int p : puncs) if (p != canonical) to_remove.insert(p);
        }
        if (!to_remove.empty()) {
            // Compact punctures
            std::vector<DevicePuncture> compacted;
            compacted.reserve(h_count - to_remove.size());
            for (int i = 0; i < h_count; ++i)
                if (!to_remove.count(i)) compacted.push_back(h_punctures[i]);
            h_punctures = std::move(compacted);
            h_count = (int)h_punctures.size();
            std::cout << "  G2 global dedup: removed " << to_remove.size()
                      << " duplicates, " << h_count << " remaining\n";
        }
    }

    // === CPU Stitching ===
    auto t_stitch_start = std::chrono::high_resolution_clock::now();

    // Build face → puncture list
    std::map<std::set<uint64_t>, std::vector<int>> face_to_punc;
    for (int i = 0; i < h_count; ++i) {
        const auto& s = h_punctures[i].simplex;
        face_to_punc[{s.vertices[0], s.vertices[1], s.vertices[2]}].push_back(i);
    }

    // Per-triangle histogram
    {
        std::map<int,int> hist;
        for (auto& [k,v] : face_to_punc) hist[v.size()]++;
        std::cout << "  Per-triangle puncture histogram:\n";
        for (auto& [n,cnt] : hist)
            std::cout << "    " << cnt << " triangles with " << n << " puncture(s)\n";
    }

    // Stitch through tetrahedra
    std::vector<PunctureConnection> connections;
    int n2 = 0, n_more = 0;
    std::map<int,int> tet_hist;

    mesh->iterate_simplices(3, [&](const Simplex& s) {
        std::vector<std::set<uint64_t>> faces = {
            {s.vertices[0], s.vertices[1], s.vertices[2]},
            {s.vertices[0], s.vertices[1], s.vertices[3]},
            {s.vertices[0], s.vertices[2], s.vertices[3]},
            {s.vertices[1], s.vertices[2], s.vertices[3]}
        };
        std::vector<int> tp;
        for (auto& f : faces)
            if (face_to_punc.count(f))
                for (int p : face_to_punc[f]) tp.push_back(p);
        if (tp.empty()) return true;

        tet_hist[tp.size()]++;
        uint64_t tet_id = *std::min_element(s.vertices, s.vertices + 4);

        if (tp.size() == 2) {
            n2++;
            connections.push_back({tp[0], tp[1], tet_id});
        } else if (tp.size() > 2) {
            n_more++;
            double V_arr[4][3], W_arr[4][3];
            for (int i = 0; i < 4; ++i) {
                auto c = mesh->get_vertex_coordinates(s.vertices[i]);
                auto fv = tc.eval(c[0], c[1], c[2]);
                for (int j = 0; j < 3; ++j) V_arr[i][j] = fv[j];
                for (int j = 0; j < 3; ++j) W_arr[i][j] = fv[3+j];
            }
            __int128 Q_i128[4], P_i128[4][4];
            compute_tet_QP_i128(V_arr, W_arr, Q_i128, P_i128);

            bool q_zero = (Q_i128[0] == 0 && Q_i128[1] == 0 &&
                           Q_i128[2] == 0 && Q_i128[3] == 0);

            if (!q_zero) {
                stitch_ambiguous_tet(tp, h_punctures, Q_i128, P_i128, tet_id, connections);
            } else {
                std::vector<std::pair<double,int>> lam_idx;
                for (int p : tp)
                    lam_idx.push_back({(double)h_punctures[p].scalar, p});
                std::sort(lam_idx.begin(), lam_idx.end());
                for (size_t j = 0; j + 1 < lam_idx.size(); j += 2)
                    connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
            }
        }
        return true;
    });

    std::cout << "  Tet histogram: ";
    for (auto& [n,c] : tet_hist) std::cout << c << "x" << n << "  ";
    std::cout << "\n";
    std::cout << "  " << n2 << " tets with 2, " << n_more << " tets with >2 punctures\n";
    std::cout << "  " << connections.size() << " connections (before dedup)\n";

    // Deduplicate connections
    {
        std::set<std::pair<int,int>> seen;
        std::vector<PunctureConnection> unique;
        for (auto& c : connections) {
            auto key = std::make_pair(std::min(c.puncture1_idx, c.puncture2_idx),
                                      std::max(c.puncture1_idx, c.puncture2_idx));
            if (seen.insert(key).second)
                unique.push_back(c);
        }
        connections = std::move(unique);
    }
    std::cout << "  " << connections.size() << " connections (after dedup)\n";

    // Build adjacency graph
    std::map<int,std::vector<int>> adj;
    for (auto& c : connections) {
        adj[c.puncture1_idx].push_back(c.puncture2_idx);
        adj[c.puncture2_idx].push_back(c.puncture1_idx);
    }

    // Trace curves
    std::set<int> visited;
    std::vector<std::vector<int>> curves;
    std::vector<bool> curve_closed;

    std::vector<int> starts;
    for (int i = 0; i < h_count; ++i) if (adj[i].size() == 1) starts.push_back(i);
    for (int i = 0; i < h_count; ++i) if (adj[i].size() == 2) starts.push_back(i);

    for (int start : starts) {
        if (visited.count(start) || adj[start].empty()) continue;
        std::vector<int> path;
        int curr = start, prev = -1;
        bool closed = false;
        while (true) {
            if (visited.count(curr)) { closed = (curr == start); break; }
            visited.insert(curr);
            path.push_back(curr);
            int next = -1;
            for (int nb : adj[curr]) if (nb != prev) { next = nb; break; }
            if (next == -1) break;
            prev = curr; curr = next;
        }
        if (path.size() > 1) {
            curves.push_back(std::move(path));
            curve_closed.push_back(closed);
        }
    }

    auto t_stitch_end = std::chrono::high_resolution_clock::now();
    double stitch_ms = std::chrono::duration<double, std::milli>(t_stitch_end - t_stitch_start).count();

    std::cout << "  CPU stitching: " << stitch_ms << " ms\n";
    std::cout << "  " << curves.size() << " curve(s):";
    for (size_t i = 0; i < curves.size(); ++i)
        std::cout << " [" << curves[i].size() << "pts,"
                  << (curve_closed[i] ? "closed" : "open") << "]";
    std::cout << "\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "  Total: " << total_ms << " ms\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main()
{
    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (" << prop.multiProcessorCount << " SMs)\n";

    const int N = 16;
    const double cx = N/2.0, cy = N/2.0;
    const double z0 = N/2.0 + 0.5;
    const double R  = N/3.0;
    const double R1 = N/4.5;
    const double R2 = N/3.0;
    const double z1 = N/3.0 + 0.5;
    const double z2 = 2.0*N/3.0 + 0.5;

    std::vector<TestCase> cases = {
        {
            "field1_one_circle",
            "Q=0 | 1 circle | U=(1,0,0), V=(1,z-z0,r^2-R^2)",
            [=](double x, double y, double z) -> std::array<double,6> {
                double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                return {1,0,0, 1,z-z0,r2-R*R};
            }
        },
        {
            "field2_two_stacked_circles",
            "Q=0 | 2 circles (stacked)",
            [=](double x, double y, double z) -> std::array<double,6> {
                double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                return {1,0,0, 1,(z-z1)*(z-z2),r2-R*R};
            }
        },
        {
            "field3_two_concentric_circles",
            "Q=0 | 2 concentric circles",
            [=](double x, double y, double z) -> std::array<double,6> {
                double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                return {1,0,0, 1,z-z0,(r2-R1*R1)*(r2-R2*R2)};
            }
        },
        {
            "field4_v_constant",
            "Q=det(A)!=0 (constant) | open PV line",
            [=](double x, double y, double z) -> std::array<double,6> {
                return {x-cx, y-cy, z-z0, 1,0,0};
            }
        },
        {
            "field5_cyclic_q_cubic",
            "Q cubic | 1 open PV line (dir (1,1,1))",
            [=](double x, double y, double z) -> std::array<double,6> {
                const double r = 8.37, p = 8.13, q = 8.51;
                return {y-p, z-q, x-r,   z-q, x-r, y-p};
            }
        },
        {
            "field6_three_pv_lines",
            "Q cubic | 3 open PV lines (distinct lambda=1,2,3)",
            [](double x, double y, double z) -> std::array<double,6> {
                const double cx6 = 8.37, cy6 = 8.13, z06 = 8.51;
                double xp = x - cx6, yp = y - cy6, zp = z - z06;
                double ux = (11.0/6)*xp + (-1.0/6)*yp + (-2.0/3)*zp;
                double uy = (-1.0/6)*xp + (11.0/6)*yp + (-2.0/3)*zp;
                double uz = (-2.0/3)*xp + (-2.0/3)*yp + (7.0/3)*zp;
                return {ux, uy, uz, xp, yp, zp};
            }
        },
    };

    // Also run CPU baseline for comparison
    std::cout << "\n=== CPU Baseline ===\n";
    for (const auto& tc : cases) {
        std::cout << "\n" << std::string(60,'=') << "\n";
        std::cout << "CASE: " << tc.name << "\n";
        std::cout << "DESC: " << tc.description << "\n";
        std::cout << std::string(60,'=') << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        auto mesh = std::make_shared<RegularSimplicialMesh>(
            std::vector<uint64_t>{(uint64_t)N,(uint64_t)N,(uint64_t)N});
        ftk::ndarray<double> uv({6, (size_t)N, (size_t)N, (size_t)N});
        for (int z = 0; z < N; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    auto fv = tc.eval((double)x, (double)y, (double)z);
                    for (int c = 0; c < 6; ++c) uv.f(c, x, y, z) = fv[c];
                }
        std::map<std::string, ftk::ndarray<double>> data;
        data["uv"] = uv;
        ExactPVPredicate<double> pred;
        pred.vector_var_name = "uv";
        SimplicialEngine<double, ExactPVPredicate<double>> engine(mesh, pred);
        engine.execute(data, {"uv"});
        auto complex = engine.get_complex();
        int cpu_count = 0;
        for (size_t i = 0; i < complex.vertices.size(); ++i)
            if (complex.vertices[i].simplex.dimension == 2) cpu_count++;
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  CPU: " << cpu_count << " face punctures in " << cpu_ms << " ms\n";
    }

    std::cout << "\n\n=== GPU Extraction + CPU Stitching ===\n";
    for (const auto& tc : cases) {
        std::cout << "\n" << std::string(60,'=') << "\n";
        std::cout << "CASE: " << tc.name << "\n";
        std::cout << "DESC: " << tc.description << "\n";
        std::cout << std::string(60,'=') << "\n";
        run_test_case(tc, N);
    }

    return 0;
}
