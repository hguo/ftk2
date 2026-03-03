#pragma once

#include <ftk2/core/parallel.hpp>
#include <ftk2/core/coface_lut.hpp>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <atomic>

#if FTK_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifndef FTK_HOST_DEVICE
#ifdef __CUDACC__
#define FTK_HOST_DEVICE __host__ __device__
#else
#define FTK_HOST_DEVICE
#endif
#endif

namespace ftk2 {

/**
 * @brief Represents a simplex in the mesh.
 */
struct Simplex {
    int dimension;
    uint64_t vertices[5]; // Max dimension supported is 4 (pentatope)

    FTK_HOST_DEVICE
    bool operator<(const Simplex& other) const {
        if (dimension != other.dimension) return dimension < other.dimension;
        for (int i = 0; i <= dimension; ++i) {
            if (vertices[i] != other.vertices[i]) return vertices[i] < other.vertices[i];
        }
        return false;
    }

    FTK_HOST_DEVICE
    bool operator==(const Simplex& other) const {
        if (dimension != other.dimension) return false;
        for (int i = 0; i <= dimension; ++i) {
            if (vertices[i] != other.vertices[i]) return false;
        }
        return true;
    }

    FTK_HOST_DEVICE
    void sort_vertices() {
        for (int i = 0; i <= dimension; ++i) {
            for (int j = i + 1; j <= dimension; ++j) {
                if (vertices[i] > vertices[j]) {
                    uint64_t temp = vertices[i];
                    vertices[i] = vertices[j];
                    vertices[j] = temp;
                }
            }
        }
    }
};

struct SimplexHash {
    size_t operator()(const Simplex& s) const {
        size_t h = s.dimension;
        for (int i = 0; i <= s.dimension; ++i) {
            h ^= std::hash<uint64_t>{}(s.vertices[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

/**
 * @brief Abstract base class for all mesh types.
 */
class Mesh {
public:
    virtual ~Mesh() = default;
    virtual int get_spatial_dimension() const = 0;
    virtual int get_total_dimension() const = 0;
    virtual uint64_t get_num_vertices() const = 0;
    virtual void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const = 0;
    virtual void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const = 0;
    virtual std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const = 0;

    void faces(const Simplex& s, std::function<void(const Simplex&)> callback) const {
        if (s.dimension == 0) return;
        for (int i = 0; i <= s.dimension; ++i) {
            Simplex f;
            f.dimension = s.dimension - 1;
            int next = 0;
            for (int j = 0; j <= s.dimension; ++j) {
                if (i == j) continue;
                f.vertices[next++] = s.vertices[j];
            }
            callback(f);
        }
    }
};

// ---- CPU Freudenthal Lookup Tables ----
// Each table lists the unique m-simplices per d-hypercube.
// cpu_lut_D_M[type][vertex][coord]: vertex offset from hypercube base.
// Prefixed cpu_ to avoid symbol conflicts with CUDA __constant__ tables.

// d=2
inline constexpr int cpu_lut_2_1[3][2][2] = {
    {{0,0},{0,1}},  {{0,0},{1,0}},  {{0,0},{1,1}}
};
inline constexpr int cpu_lut_2_2[2][3][2] = {
    {{0,0},{1,0},{1,1}},  {{0,0},{0,1},{1,1}}
};

// d=3
inline constexpr int cpu_lut_3_1[7][2][3] = {
    {{0,0,0},{0,0,1}},  {{0,0,0},{0,1,0}},  {{0,0,0},{0,1,1}},  {{0,0,0},{1,0,0}},
    {{0,0,0},{1,0,1}},  {{0,0,0},{1,1,0}},  {{0,0,0},{1,1,1}}
};
inline constexpr int cpu_lut_3_2[12][3][3] = {
    {{0,0,0},{0,0,1},{0,1,1}},  {{0,0,0},{0,0,1},{1,0,1}},
    {{0,0,0},{0,0,1},{1,1,1}},  {{0,0,0},{0,1,0},{0,1,1}},
    {{0,0,0},{0,1,0},{1,1,0}},  {{0,0,0},{0,1,0},{1,1,1}},
    {{0,0,0},{0,1,1},{1,1,1}},  {{0,0,0},{1,0,0},{1,0,1}},
    {{0,0,0},{1,0,0},{1,1,0}},  {{0,0,0},{1,0,0},{1,1,1}},
    {{0,0,0},{1,0,1},{1,1,1}},  {{0,0,0},{1,1,0},{1,1,1}}
};
inline constexpr int cpu_lut_3_3[6][4][3] = {
    {{0,0,0},{1,0,0},{1,1,0},{1,1,1}},  {{0,0,0},{1,0,0},{1,0,1},{1,1,1}},
    {{0,0,0},{0,1,0},{1,1,0},{1,1,1}},  {{0,0,0},{0,1,0},{0,1,1},{1,1,1}},
    {{0,0,0},{0,0,1},{1,0,1},{1,1,1}},  {{0,0,0},{0,0,1},{0,1,1},{1,1,1}}
};

// d=4
inline constexpr int cpu_lut_4_1[15][2][4] = {
    {{0,0,0,0},{0,0,0,1}},  {{0,0,0,0},{0,0,1,0}},  {{0,0,0,0},{0,0,1,1}},
    {{0,0,0,0},{0,1,0,0}},  {{0,0,0,0},{0,1,0,1}},  {{0,0,0,0},{0,1,1,0}},
    {{0,0,0,0},{0,1,1,1}},  {{0,0,0,0},{1,0,0,0}},  {{0,0,0,0},{1,0,0,1}},
    {{0,0,0,0},{1,0,1,0}},  {{0,0,0,0},{1,0,1,1}},  {{0,0,0,0},{1,1,0,0}},
    {{0,0,0,0},{1,1,0,1}},  {{0,0,0,0},{1,1,1,0}},  {{0,0,0,0},{1,1,1,1}}
};
inline constexpr int cpu_lut_4_2[50][3][4] = {
    {{0,0,0,0},{0,0,0,1},{0,0,1,1}},  {{0,0,0,0},{0,0,0,1},{0,1,0,1}},
    {{0,0,0,0},{0,0,0,1},{0,1,1,1}},  {{0,0,0,0},{0,0,0,1},{1,0,0,1}},
    {{0,0,0,0},{0,0,0,1},{1,0,1,1}},  {{0,0,0,0},{0,0,0,1},{1,1,0,1}},
    {{0,0,0,0},{0,0,0,1},{1,1,1,1}},  {{0,0,0,0},{0,0,1,0},{0,0,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,1,1,0}},  {{0,0,0,0},{0,0,1,0},{0,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{1,0,1,0}},  {{0,0,0,0},{0,0,1,0},{1,0,1,1}},
    {{0,0,0,0},{0,0,1,0},{1,1,1,0}},  {{0,0,0,0},{0,0,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,1},{0,1,1,1}},  {{0,0,0,0},{0,0,1,1},{1,0,1,1}},
    {{0,0,0,0},{0,0,1,1},{1,1,1,1}},  {{0,0,0,0},{0,1,0,0},{0,1,0,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,1,0}},  {{0,0,0,0},{0,1,0,0},{0,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{1,1,0,0}},  {{0,0,0,0},{0,1,0,0},{1,1,0,1}},
    {{0,0,0,0},{0,1,0,0},{1,1,1,0}},  {{0,0,0,0},{0,1,0,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,1},{0,1,1,1}},  {{0,0,0,0},{0,1,0,1},{1,1,0,1}},
    {{0,0,0,0},{0,1,0,1},{1,1,1,1}},  {{0,0,0,0},{0,1,1,0},{0,1,1,1}},
    {{0,0,0,0},{0,1,1,0},{1,1,1,0}},  {{0,0,0,0},{0,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{1,0,0,0},{1,0,0,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,1,0}},  {{0,0,0,0},{1,0,0,0},{1,0,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,1,0,0}},  {{0,0,0,0},{1,0,0,0},{1,1,0,1}},
    {{0,0,0,0},{1,0,0,0},{1,1,1,0}},  {{0,0,0,0},{1,0,0,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,1},{1,0,1,1}},  {{0,0,0,0},{1,0,0,1},{1,1,0,1}},
    {{0,0,0,0},{1,0,0,1},{1,1,1,1}},  {{0,0,0,0},{1,0,1,0},{1,0,1,1}},
    {{0,0,0,0},{1,0,1,0},{1,1,1,0}},  {{0,0,0,0},{1,0,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,1,1},{1,1,1,1}},  {{0,0,0,0},{1,1,0,0},{1,1,0,1}},
    {{0,0,0,0},{1,1,0,0},{1,1,1,0}},  {{0,0,0,0},{1,1,0,0},{1,1,1,1}},
    {{0,0,0,0},{1,1,0,1},{1,1,1,1}},  {{0,0,0,0},{1,1,1,0},{1,1,1,1}}
};
inline constexpr int cpu_lut_4_3[60][4][4] = {
    {{0,0,0,0},{0,0,0,1},{0,0,1,1},{0,1,1,1}},  {{0,0,0,0},{0,0,0,1},{0,0,1,1},{1,0,1,1}},
    {{0,0,0,0},{0,0,0,1},{0,0,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,0,1},{0,1,0,1},{0,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{0,1,0,1},{1,1,0,1}},  {{0,0,0,0},{0,0,0,1},{0,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,0,1},{1,0,0,1},{1,0,1,1}},
    {{0,0,0,0},{0,0,0,1},{1,0,0,1},{1,1,0,1}},  {{0,0,0,0},{0,0,0,1},{1,0,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{1,0,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,0,1,1},{0,1,1,1}},  {{0,0,0,0},{0,0,1,0},{0,0,1,1},{1,0,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,0,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,1,0},{0,1,1,0},{0,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,1,1,0},{1,1,1,0}},  {{0,0,0,0},{0,0,1,0},{0,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,1,0},{1,0,1,0},{1,0,1,1}},
    {{0,0,0,0},{0,0,1,0},{1,0,1,0},{1,1,1,0}},  {{0,0,0,0},{0,0,1,0},{1,0,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{1,0,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,1},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{0,0,1,1},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,0,1},{0,1,1,1}},  {{0,0,0,0},{0,1,0,0},{0,1,0,1},{1,1,0,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,0,1},{1,1,1,1}},  {{0,0,0,0},{0,1,0,0},{0,1,1,0},{0,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,1,0},{1,1,1,0}},  {{0,0,0,0},{0,1,0,0},{0,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{0,1,0,0},{1,1,0,0},{1,1,0,1}},
    {{0,0,0,0},{0,1,0,0},{1,1,0,0},{1,1,1,0}},  {{0,0,0,0},{0,1,0,0},{1,1,0,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{1,1,0,1},{1,1,1,1}},  {{0,0,0,0},{0,1,0,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,1},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{0,1,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,1,1,0},{0,1,1,1},{1,1,1,1}},  {{0,0,0,0},{0,1,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,0,1},{1,0,1,1}},  {{0,0,0,0},{1,0,0,0},{1,0,0,1},{1,1,0,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,0,1},{1,1,1,1}},  {{0,0,0,0},{1,0,0,0},{1,0,1,0},{1,0,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,1,0},{1,1,1,0}},  {{0,0,0,0},{1,0,0,0},{1,0,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,1,1},{1,1,1,1}},  {{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,0,1}},
    {{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,1,0}},  {{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,1,0,1},{1,1,1,1}},  {{0,0,0,0},{1,0,0,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,1},{1,0,1,1},{1,1,1,1}},  {{0,0,0,0},{1,0,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{1,0,1,0},{1,0,1,1},{1,1,1,1}},  {{0,0,0,0},{1,0,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,1,0,0},{1,1,0,1},{1,1,1,1}},  {{0,0,0,0},{1,1,0,0},{1,1,1,0},{1,1,1,1}}
};
inline constexpr int cpu_lut_4_4[24][5][4] = {
    {{0,0,0,0},{0,0,0,1},{0,0,1,1},{0,1,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{0,0,1,1},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{0,1,0,1},{0,1,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{0,1,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{1,0,0,1},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,0,1},{1,0,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,0,1,1},{0,1,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,0,1,1},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,1,1,0},{0,1,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{0,1,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{1,0,1,0},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,0,1,0},{1,0,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,0,1},{0,1,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,1,0},{0,1,1,1},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{0,1,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{1,1,0,0},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{0,1,0,0},{1,1,0,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,0,1},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,0,1},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,1,0},{1,0,1,1},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,0,1,0},{1,1,1,0},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,0,1},{1,1,1,1}},
    {{0,0,0,0},{1,0,0,0},{1,1,0,0},{1,1,1,0},{1,1,1,1}}
};

// Returns the number of unique m-simplex types per d-hypercube, or 0 if unsupported.
inline int cpu_get_num_simplex_types(int d, int m) {
    if (d == 2) { if (m == 1) return 3; if (m == 2) return 2; }
    if (d == 3) { if (m == 1) return 7; if (m == 2) return 12; if (m == 3) return 6; }
    if (d == 4) { if (m == 1) return 15; if (m == 2) return 50; if (m == 3) return 60; if (m == 4) return 24; }
    return 0;
}

// Dispatch to the correct LUT: returns vertex offset for (d, m, type, vi, j).
inline int cpu_lut_dispatch(int d, int m, int type, int vi, int j) {
    if (d == 2 && m == 1) return cpu_lut_2_1[type][vi][j];
    if (d == 2 && m == 2) return cpu_lut_2_2[type][vi][j];
    if (d == 3 && m == 1) return cpu_lut_3_1[type][vi][j];
    if (d == 3 && m == 2) return cpu_lut_3_2[type][vi][j];
    if (d == 3 && m == 3) return cpu_lut_3_3[type][vi][j];
    if (d == 4 && m == 1) return cpu_lut_4_1[type][vi][j];
    if (d == 4 && m == 2) return cpu_lut_4_2[type][vi][j];
    if (d == 4 && m == 3) return cpu_lut_4_3[type][vi][j];
    if (d == 4 && m == 4) return cpu_lut_4_4[type][vi][j];
    return 0;
}

/**
 * @brief Represents a regular simplicial mesh.
 */
class RegularSimplicialMesh : public Mesh {
public:
    RegularSimplicialMesh(const std::vector<uint64_t>& dims, 
                          const std::vector<uint64_t>& offset = {},
                          const std::vector<uint64_t>& global_dims = {}) 
        : dims_(dims), offset_(offset), global_dims_(global_dims) 
    {
        if (offset_.empty()) offset_.assign(dims_.size(), 0);
        if (global_dims_.empty()) global_dims_ = dims_;
    }

    int get_spatial_dimension() const override { return dims_.size(); }
    int get_total_dimension() const override { return dims_.size(); }

    void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const override {
        int d = get_total_dimension();
        int n_types = cpu_get_num_simplex_types(d, k);
        if (n_types > 0) {
            // LUT path: iterate (hypercube x type), each simplex visited exactly once
            std::vector<uint64_t> coords(d);
            iterate_hypercubes(0, coords, [&](const std::vector<uint64_t>& base_coords) {
                for (int type = 0; type < n_types; ++type) {
                    Simplex s; s.dimension = k;
                    for (int vi = 0; vi <= k; ++vi) {
                        uint64_t g[4] = {0};
                        for (int j = 0; j < d; ++j)
                            g[j] = base_coords[j] + offset_[j] + cpu_lut_dispatch(d, k, type, vi, j);
                        s.vertices[vi] = grid_index_to_id(g);
                    }
                    s.sort_vertices();
                    callback(s);
                }
            });
        } else {
            // Fallback for unsupported (d,k): original permutation + find_k_faces
            std::vector<uint64_t> coords(dims_.size());
            iterate_hypercubes(0, coords, [&](const std::vector<uint64_t>& base_coords) {
                uint64_t hc_idx = hypercube_coords_to_idx(base_coords);
                int n_p = 1; for(int i=1; i<=d; ++i) n_p *= i;
                for (int p_idx = 0; p_idx < n_p; ++p_idx) {
                    Simplex cell; get_d_simplex(hc_idx, p_idx, cell);
                    if (k == d) callback(cell);
                    else find_k_faces(cell, k, callback);
                }
            });
        }
    }

    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {
        int k = s.dimension;
        int d = get_total_dimension();
        if (k >= d) return;

        // Check if coface LUT is available for this (d, k)
        int n_cofaces_check = cpu_coface_count_dispatch(d, k, 0);
        if (n_cofaces_check == 0) {
            // Fallback: brute-force for unsupported (d,k)
            int target_dim = k + 1;
            iterate_simplices(target_dim, [&](const Simplex& candidate) {
                bool contains_all = true;
                for (int i = 0; i <= k; ++i) {
                    bool found = false;
                    for (int j = 0; j <= target_dim; ++j) {
                        if (candidate.vertices[j] == s.vertices[i]) { found = true; break; }
                    }
                    if (!found) { contains_all = false; break; }
                }
                if (contains_all) callback(candidate);
            });
            return;
        }

        // Recover corner and type from the simplex
        int corner[4] = {0, 0, 0, 0};
        int simplex_type = 0;

        if (k == 0) {
            // Vertex: corner = vertex grid coordinates, no type needed
            std::vector<uint64_t> gc = id_to_grid_index(s.vertices[0]);
            for (int j = 0; j < d; ++j) corner[j] = (int)gc[j];
        } else {
            simplex_type = recover_simplex_type(s, d, k, corner);
            if (simplex_type < 0) return; // should not happen
        }

        int n_cofaces = cpu_coface_count_dispatch(d, k, simplex_type);
        for (int ci = 0; ci < n_cofaces; ++ci) {
            int coface_type = cpu_coface_lut_dispatch(d, k, simplex_type, ci, 0);
            int coface_corner[4] = {0, 0, 0, 0};
            for (int j = 0; j < d; ++j)
                coface_corner[j] = corner[j] + cpu_coface_lut_dispatch(d, k, simplex_type, ci, 1 + j);

            // Boundary check: coface corner must be a valid hypercube base
            // Local base = coface_corner[j] - offset_[j], must be in [0, dims_[j]-2]
            bool valid = true;
            for (int j = 0; j < d; ++j) {
                int local_base = coface_corner[j] - (int)offset_[j];
                if (local_base < 0 || local_base > (int)dims_[j] - 2) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            // Build coface simplex from LUT
            Simplex cf;
            cf.dimension = k + 1;
            for (int vi = 0; vi <= k + 1; ++vi) {
                uint64_t g[4] = {0};
                for (int j = 0; j < d; ++j)
                    g[j] = (uint64_t)(coface_corner[j] + cpu_lut_dispatch(d, k + 1, coface_type, vi, j));
                cf.vertices[vi] = grid_index_to_id(g);
            }
            cf.sort_vertices();
            callback(cf);
        }
    }

    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override {
        std::vector<uint64_t> grid_idx = id_to_grid_index(vertex_id);
        std::vector<double> coords(grid_idx.begin(), grid_idx.end());
        return coords;
    }

    uint64_t get_num_vertices() const override {
        uint64_t n = 1; for (auto d : dims_) n *= d; return n;
    }

    std::vector<uint64_t> get_vertex_coords_local(uint64_t index) const {
        std::vector<uint64_t> coords(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) { coords[i] = index % dims_[i]; index /= dims_[i]; }
        return coords;
    }

    bool is_hypercube_base(const std::vector<uint64_t>& coords) const {
        for (size_t i = 0; i < coords.size(); ++i) if (coords[i] >= dims_[i] - 1) return false;
        return true;
    }

    uint64_t hypercube_coords_to_idx(const std::vector<uint64_t>& coords) const {
        uint64_t idx = 0; uint64_t multiplier = 1;
        for (size_t i = 0; i < dims_.size(); ++i) {
            idx += coords[i] * multiplier; multiplier *= (dims_[i] - 1);
        }
        return idx;
    }

    void get_d_simplex(uint64_t hc_idx, int p_idx, Simplex& s) const {
        int d = get_total_dimension(); s.dimension = d;
        std::vector<uint64_t> base_coords(d); uint64_t temp_idx = hc_idx;
        for (int i = 0; i < d; ++i) { base_coords[i] = temp_idx % (dims_[i] - 1) + offset_[i]; temp_idx /= (dims_[i] - 1); }
        std::vector<int> p(d); std::iota(p.begin(), p.end(), 0);
        int temp_p_idx = p_idx;
        for (int i = 0; i < d; ++i) { int j = temp_p_idx % (d - i); std::swap(p[i], p[i + j]); temp_p_idx /= (d - i); }
        std::vector<uint64_t> current_coords = base_coords;
        uint64_t g[4] = {0}; for(int i=0; i<d; ++i) g[i] = current_coords[i]; s.vertices[0] = grid_index_to_id(g);
        for (int i = 0; i < d; ++i) { current_coords[p[i]]++; uint64_t gi[4] = {0}; for(int j=0; j<d; ++j) gi[j] = current_coords[j]; s.vertices[i + 1] = grid_index_to_id(gi); }
        s.sort_vertices();
    }

    const std::vector<uint64_t>& get_local_dims() const { return dims_; }
    const std::vector<uint64_t>& get_offset() const { return offset_; }
    const std::vector<uint64_t>& get_global_dims() const { return global_dims_; }

    std::vector<uint64_t> id_to_grid_index(uint64_t id) const {
        std::vector<uint64_t> coords(global_dims_.size());
        for (size_t i = 0; i < global_dims_.size(); ++i) { coords[i] = id % global_dims_[i]; id /= global_dims_[i]; }
        return coords;
    }

    FTK_HOST_DEVICE void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        for (int i = 0; i < 4; ++i) coords[i] = 0;
#ifndef __CUDA_ARCH__
        uint64_t temp_id = id;
        for (size_t i = 0; i < global_dims_.size(); ++i) {
            coords[i] = temp_id % global_dims_[i];
            temp_id /= global_dims_[i];
        }
#else
        (void)id; // Suppress unused warning on device
#endif
    }

    FTK_HOST_DEVICE uint64_t grid_index_to_id(const uint64_t g[4]) const {
        uint64_t id = 0;
#ifndef __CUDA_ARCH__
        uint64_t multiplier = 1;
        for (size_t i = 0; i < global_dims_.size(); ++i) {
            id += g[i] * multiplier;
            multiplier *= global_dims_[i];
        }
#else
        (void)g; // Suppress unused warning on device
#endif
        return id;
    }

protected:

private:
    // Recover (corner, simplex_type) from a Simplex's vertex IDs.
    // Returns the matching LUT type index, sets corner[] to the global corner coords.
    // Returns -1 if no match found (should not happen for valid simplices).
    int recover_simplex_type(const Simplex& s, int d, int k, int corner[4]) const {
        // Get grid coordinates for each vertex
        int coords[5][4];
        for (int vi = 0; vi <= k; ++vi) {
            std::vector<uint64_t> gc = id_to_grid_index(s.vertices[vi]);
            for (int j = 0; j < d; ++j) coords[vi][j] = (int)gc[j];
            for (int j = d; j < 4; ++j) coords[vi][j] = 0;
        }

        // Corner = min coordinate in each dimension
        for (int j = 0; j < d; ++j) {
            corner[j] = coords[0][j];
            for (int vi = 1; vi <= k; ++vi)
                if (coords[vi][j] < corner[j]) corner[j] = coords[vi][j];
        }

        // Compute relative offsets and sort them lexicographically
        int offsets[5][4];
        for (int vi = 0; vi <= k; ++vi)
            for (int j = 0; j < d; ++j)
                offsets[vi][j] = coords[vi][j] - corner[j];

        // Sort offsets lexicographically (bubble sort, k+1 <= 5 elements)
        for (int i = 0; i <= k; ++i) {
            for (int j = i + 1; j <= k; ++j) {
                bool swap = false;
                for (int c = 0; c < d; ++c) {
                    if (offsets[i][c] < offsets[j][c]) break;
                    if (offsets[i][c] > offsets[j][c]) { swap = true; break; }
                }
                if (swap) {
                    for (int c = 0; c < d; ++c)
                        std::swap(offsets[i][c], offsets[j][c]);
                }
            }
        }

        // Linear scan through LUT types to find a match
        int n_types = cpu_get_num_simplex_types(d, k);
        for (int t = 0; t < n_types; ++t) {
            bool match = true;
            for (int vi = 0; vi <= k && match; ++vi)
                for (int j = 0; j < d && match; ++j)
                    if (offsets[vi][j] != cpu_lut_dispatch(d, k, t, vi, j))
                        match = false;
            if (match) return t;
        }
        return -1;
    }

    void iterate_hypercubes(int dim, std::vector<uint64_t>& coords, std::function<void(const std::vector<uint64_t>&)> callback) const {
        if (dim == dims_.size()) {
            callback(coords); return;
        }
        for (uint64_t i = 0; i < dims_[dim] - 1; ++i) {
            coords[dim] = i;
            iterate_hypercubes(dim + 1, coords, callback);
        }
    }

    void find_k_faces(const Simplex& s, int k, std::function<void(const Simplex&)> callback) const {
        int n = s.dimension + 1;
        int r = k + 1;
        std::vector<int> p(r);
        std::iota(p.begin(), p.end(), 0);
        while (p[0] <= n - r) {
            Simplex f; f.dimension = k;
            for (int i = 0; i < r; ++i) f.vertices[i] = s.vertices[p[i]];
            callback(f);
            int i = r - 1;
            while (i >= 0 && p[i] == n - r + i) i--;
            if (i < 0) break;
            p[i]++;
            for (int j = i + 1; j < r; j++) p[j] = p[i] + j - i;
        }
    }

    std::vector<uint64_t> dims_, offset_, global_dims_;
};

/**
 * @brief Represents a mesh extruded in time.
 */
class ExtrudedSimplicialMesh : public Mesh {
public:
    ExtrudedSimplicialMesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers = 0) 
        : base_mesh_(base_mesh), n_layers_(n_layers) 
    {
        std::atomic<uint64_t> max_v(0);
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(base_mesh_);
        if (reg_mesh) {
            uint64_t g[4] = {0};
            auto g_dims = reg_mesh->get_global_dims();
            for(int i=0; i<4 && i<g_dims.size(); ++i) g[i] = g_dims[i];
            max_v = reg_mesh->grid_index_to_id(g);
        } else {
            base_mesh_->iterate_simplices(0, [&](const Simplex& s) {
                uint64_t v = s.vertices[0] + 1;
                uint64_t current = max_v.load();
                while (v > current && !max_v.compare_exchange_weak(current, v));
            });
        }
        n_spatial_verts_ = max_v.load();
    }

    int get_spatial_dimension() const override { return base_mesh_->get_spatial_dimension(); }
    int get_total_dimension() const override { return base_mesh_->get_total_dimension() + 1; }
    uint64_t get_num_vertices() const override { return n_spatial_verts_ * (n_layers_ + 1); }

    void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const override {
        int d_spatial = base_mesh_->get_total_dimension();
        if (k < 0 || k > d_spatial + 1) return;

        // 1. Purely spatial simplices at each timestep
        for (uint64_t t = 0; t <= n_layers_; ++t) {
            base_mesh_->iterate_simplices(k, [&](const Simplex& s_spatial) {
                Simplex s = s_spatial;
                for (int i = 0; i <= k; ++i) s.vertices[i] += t * n_spatial_verts_;
                callback(s);
            });
        }

        // 2. Spacetime simplices connecting t and t+1
        if (k > 0) {
            for (uint64_t t = 0; t < n_layers_; ++t) {
                base_mesh_->iterate_simplices(k - 1, [&](const Simplex& s_spatial) {
                    // Kuhn subdivision of (k-1)-simplex * [t, t+1] gives k simplices of dim k
                    // Spatial vertices are sorted: v0 < v1 < ... < vk-1
                    for (int j = 0; j < k; ++j) {
                        Simplex s; s.dimension = k;
                        // Vertices at time t: v0, ..., vj
                        for (int l = 0; l <= j; ++l) s.vertices[l] = s_spatial.vertices[l] + t * n_spatial_verts_;
                        // Vertices at time t+1: vj, ..., vk-1
                        for (int l = j; l < k; ++l) s.vertices[l + 1] = s_spatial.vertices[l] + (t + 1) * n_spatial_verts_;
                        s.sort_vertices();
                        callback(s);
                    }
                });
            }
        }
    }

    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {
        // Find all (d+1)-simplices that contain simplex s
        if (s.dimension >= get_total_dimension()) return;  // No cofaces for top-dimensional simplices

        int target_dim = s.dimension + 1;

        // Iterate over all simplices of dimension target_dim
        iterate_simplices(target_dim, [&](const Simplex& candidate) {
            // Check if candidate contains all vertices of s
            bool contains_all = true;
            for (int i = 0; i <= s.dimension; ++i) {
                bool found = false;
                for (int j = 0; j <= target_dim; ++j) {
                    if (candidate.vertices[j] == s.vertices[i]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    contains_all = false;
                    break;
                }
            }

            if (contains_all) {
                callback(candidate);
            }
        });
    }

    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override {
        uint64_t t = vertex_id / n_spatial_verts_;
        uint64_t v = vertex_id % n_spatial_verts_;
        std::vector<double> coords = base_mesh_->get_vertex_coordinates(v);
        coords.push_back(static_cast<double>(t));
        return coords;
    }

    std::shared_ptr<Mesh> get_base_mesh() const { return base_mesh_; }
    uint64_t get_n_layers() const { return n_layers_; }
    uint64_t get_n_spatial_verts() const { return n_spatial_verts_; }

private:
    std::shared_ptr<Mesh> base_mesh_;
    uint64_t n_layers_;
    uint64_t n_spatial_verts_;
};

/**
 * @brief Helper to encode a simplex into a unique ID based on a regular grid.
 * @note Host-only due to std::vector usage.
 */
inline uint64_t encode_simplex_id(const Simplex& s, const RegularSimplicialMesh& mesh) {
    uint64_t v_min = s.vertices[0];
    for (int i = 1; i <= s.dimension; ++i) if (s.vertices[i] < v_min) v_min = s.vertices[i];
    std::vector<uint64_t> c_min = mesh.id_to_grid_index(v_min);
    uint64_t combined_mask = 0;
    for (int i = 0; i <= s.dimension; ++i) {
        if (s.vertices[i] == v_min) continue;
        std::vector<uint64_t> ci = mesh.id_to_grid_index(s.vertices[i]);
        uint64_t mask = 0;
        for (int k = 0; k < 4 && k < ci.size(); ++k) if (ci[k] > c_min[k]) mask |= (1 << k);
        combined_mask = (combined_mask << 4) | (mask & 0xF);
    }
    return (v_min << 24) | (combined_mask & 0xFFFFFF);
}

/**
 * @brief Factory class for creating different mesh types.
 */
class MeshFactory {
public:
    static std::unique_ptr<Mesh> create_regular_mesh(const std::vector<uint64_t>& local_dims, 
                                                   const std::vector<uint64_t>& offset = {},
                                                   const std::vector<uint64_t>& global_dims = {});
    static std::unique_ptr<Mesh> create_unstructured_mesh(int spatial_dim, int cell_dim,
                                                        const std::vector<double>& coords,
                                                        const std::vector<uint64_t>& cells);
    static std::unique_ptr<Mesh> create_extruded_mesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers);
};

} // namespace ftk2
