#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <memory>
#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_set>
#include <iostream>
#include <ftk2/core/parallel.hpp>

#ifdef __CUDACC__
#define FTK_HOST_DEVICE __host__ __device__
#else
#define FTK_HOST_DEVICE
#endif

namespace ftk2 {

/**
 * @brief Represents a K-dimensional simplex within a mesh.
 */
struct Simplex {
    int dimension; // K
    uint64_t vertices[16]; // K+1 vertex IDs

    FTK_HOST_DEVICE
    Simplex() : dimension(-1) {
        for (int i = 0; i < 16; ++i) vertices[i] = 0;
    }

    FTK_HOST_DEVICE int num_vertices() const { return dimension + 1; }
    
    FTK_HOST_DEVICE void sort_vertices() {
        if (dimension < 0) return;
        for (int i = 0; i <= dimension; ++i) {
            for (int j = i + 1; j <= dimension; ++j) {
                if (vertices[i] > vertices[j]) {
                    uint64_t tmp = vertices[i];
                    vertices[i] = vertices[j];
                    vertices[j] = tmp;
                }
            }
        }
        for (int i = dimension + 1; i < 16; ++i) vertices[i] = 0;
    }

    FTK_HOST_DEVICE uint64_t hash() const {
        uint64_t h = 0;
        for (int i = 0; i <= dimension; ++i) {
            h ^= vertices[i] + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    FTK_HOST_DEVICE bool operator==(const Simplex& other) const {
        if (dimension != other.dimension) return false;
        for (int i = 0; i <= dimension; ++i) {
            if (vertices[i] != other.vertices[i]) return false;
        }
        return true;
    }
    
    FTK_HOST_DEVICE bool operator<(const Simplex& other) const {
        if (dimension != other.dimension) return dimension < other.dimension;
        for (int i = 0; i <= dimension; ++i) {
            if (vertices[i] != other.vertices[i]) return vertices[i] < other.vertices[i];
        }
        return false;
    }
};

struct SimplexHash {
    std::size_t operator()(const Simplex& s) const {
        return static_cast<std::size_t>(s.hash());
    }
};

/**
 * @brief A unified interface for simplicial meshes of any dimension.
 */
class Mesh : public std::enable_shared_from_this<Mesh> {
public:
    virtual ~Mesh() = default;

    virtual int get_spatial_dimension() const = 0;
    virtual int get_total_dimension() const = 0;
    virtual bool is_simplicial() const { return true; }

    virtual void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const = 0;

    virtual void faces(const Simplex& s, std::function<void(const Simplex&)> callback) const {
        int n = s.dimension + 1;
        int target_k = s.dimension - 1;
        std::vector<int> p(target_k + 1);
        std::iota(p.begin(), p.end(), 0);
        while (p[0] <= n - (target_k + 1)) {
            Simplex f; f.dimension = target_k;
            for (int i = 0; i <= target_k; ++i) f.vertices[i] = s.vertices[p[i]];
            callback(f);
            int i = target_k;
            while (i >= 0 && p[i] == n - (target_k + 1) + i) i--;
            if (i < 0) break;
            p[i]++;
            for (int j = i + 1; j <= target_k; j++) p[j] = p[i] + j - i;
        }
    }

    virtual void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const = 0;
    virtual std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const = 0;

    std::shared_ptr<Mesh> extrude(uint64_t n_layers = 0);
};

/**
 * @brief Regular simplicial mesh.
 */
class RegularSimplicialMesh : public Mesh {
public:
    RegularSimplicialMesh(const std::vector<uint64_t>& local_dims, 
                         const std::vector<uint64_t>& offset = {},
                         const std::vector<uint64_t>& global_dims = {}) 
        : dims_(local_dims), offset_(offset), global_dims_(global_dims) 
    {
        if (offset_.empty()) offset_.assign(dims_.size(), 0);
        if (global_dims_.empty()) global_dims_ = dims_;
    }

    int get_spatial_dimension() const override { return dims_.size() - 1; }
    int get_total_dimension() const override { return dims_.size(); }

    void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const override {
        int d = get_total_dimension();
        if (k > d) return;

        uint64_t n_v = get_num_vertices();
        
        ftk2::parallel_for(uint64_t(0), n_v, [&](uint64_t v_idx) {
            auto l0 = get_vertex_coords_local(v_idx);
            std::vector<uint64_t> g0 = l0; for(int i=0; i<d; ++i) g0[i] += offset_[i];
            uint64_t v0 = grid_index_to_id(g0);

            // Generate all chains 0 < m1 < m2 < ... < mk < (1 << d)
            std::function<void(int, int, std::vector<int>&)> gen = [&](int last_m, int depth, std::vector<int>& chain) {
                if (depth == k) {
                    Simplex s; s.dimension = k; s.vertices[0] = v0;
                    for (int i = 0; i < k; ++i) {
                        std::vector<uint64_t> gi = g0; bool in = true;
                        for (int j = 0; j < d; ++j) { gi[j] += (chain[i] >> j) & 1; if (gi[j] >= global_dims_[j]) in = false; }
                        if (!in) return;
                        s.vertices[i + 1] = grid_index_to_id(gi);
                    }
                    s.sort_vertices();
                    if (s.vertices[0] == v0) callback(s); // Canonical representation: v0 is the minimum
                    return;
                }
                for (int m = last_m + 1; m < (1 << d); ++m) {
                    if ((last_m & m) == last_m) {
                        chain.push_back(m);
                        gen(m, depth + 1, chain);
                        chain.pop_back();
                    }
                }
            };
            std::vector<int> chain;
            gen(0, 0, chain);
        });
    }

    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {}

    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override {
        std::vector<uint64_t> g_coords = id_to_grid_index(vertex_id);
        std::vector<double> phys_coords(g_coords.size());
        for (size_t i = 0; i < g_coords.size(); ++i) phys_coords[i] = static_cast<double>(g_coords[i]);
        return phys_coords;
    }

    std::vector<uint64_t> get_local_dims() const { return dims_; }
    std::vector<uint64_t> get_offset() const { return offset_; }
    std::vector<uint64_t> get_global_dims() const { return global_dims_; }

    uint64_t get_num_vertices() const {
        uint64_t n = 1;
        for (auto d : dims_) n *= d;
        return n;
    }

    uint64_t get_num_hypercubes() const {
        uint64_t n = 1;
        for (auto d : dims_) n *= (d - 1);
        return n;
    }

    bool is_hypercube_base(const std::vector<uint64_t>& coords) const {
        for (int i = 0; i < dims_.size(); ++i) if (coords[i] >= dims_[i] - 1) return false;
        return true;
    }

    uint64_t hypercube_coords_to_idx(const std::vector<uint64_t>& coords) const {
        uint64_t idx = 0;
        uint64_t multiplier = 1;
        for (int i = 0; i < dims_.size(); ++i) {
            idx += coords[i] * multiplier;
            multiplier *= (dims_[i] - 1);
        }
        return idx;
    }

    void get_d_simplex(uint64_t hc_index, int p_index, Simplex& s) const {
        int d = get_total_dimension();
        s.dimension = d;
        std::vector<uint64_t> base_coords(d);
        uint64_t tmp_idx = hc_index;
        for (int i = 0; i < d; ++i) {
            base_coords[i] = tmp_idx % (dims_[i] - 1);
            tmp_idx /= (dims_[i] - 1);
        }

        std::vector<uint64_t> global_base = base_coords;
        for (int i = 0; i < d; ++i) global_base[i] += offset_[i];

        std::vector<int> p(d);
        std::iota(p.begin(), p.end(), 0);
        for (int i = 0; i < p_index; ++i) std::next_permutation(p.begin(), p.end());

        std::vector<uint64_t> curr = global_base;
        s.vertices[0] = grid_index_to_id(curr);
        for (int i = 0; i < d; ++i) {
            curr[p[i]]++;
            s.vertices[i + 1] = grid_index_to_id(curr);
        }
        s.sort_vertices();
    }

    std::vector<uint64_t> get_vertex_coords_local(uint64_t index) const {
        std::vector<uint64_t> coords(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            coords[i] = index % dims_[i];
            index /= dims_[i];
        }
        return coords;
    }

    uint64_t grid_index_to_id(const std::vector<uint64_t>& g) const {
        uint64_t id = 0;
        uint64_t multiplier = 1;
        for (size_t i = 0; i < global_dims_.size(); ++i) {
            uint64_t c = (i < g.size()) ? g[i] : 0;
            id += c * multiplier;
            multiplier *= global_dims_[i];
        }
        return id;
    }

    std::vector<uint64_t> id_to_grid_index(uint64_t global_id) const {
        std::vector<uint64_t> coords(global_dims_.size(), 0);
        for (size_t i = 0; i < global_dims_.size(); ++i) {
            coords[i] = global_id % global_dims_[i];
            global_id /= global_dims_[i];
        }
        return coords;
    }

    void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        std::vector<uint64_t> g = id_to_grid_index(id);
        for(int i=0; i<4; ++i) coords[i] = (i < g.size()) ? g[i] : 0;
    }

protected:

private:
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

class ExtrudedSimplicialMesh : public Mesh {
public:
    ExtrudedSimplicialMesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers = 0) 
        : base_mesh_(base_mesh), n_layers_(n_layers) {}
    int get_spatial_dimension() const override { return base_mesh_->get_spatial_dimension(); }
    int get_total_dimension() const override { return base_mesh_->get_total_dimension() + 1; }
    void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const override {}
    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {}
    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override { return {}; }
private:
    std::shared_ptr<Mesh> base_mesh_;
    uint64_t n_layers_;
};

inline std::shared_ptr<Mesh> Mesh::extrude(uint64_t n_layers) {
    return std::make_shared<ExtrudedSimplicialMesh>(shared_from_this(), n_layers);
}

class DeformedExtrudedSimplicialMesh : public Mesh {
public:
    DeformedExtrudedSimplicialMesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers = 0)
        : base_mesh_(base_mesh), n_layers_(n_layers) {}
    int get_spatial_dimension() const override { return base_mesh_->get_spatial_dimension(); }
    int get_total_dimension() const override { return base_mesh_->get_total_dimension() + 1; }
    void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const override {}
    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {}
    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override { return {}; }
private:
    std::shared_ptr<Mesh> base_mesh_;
    uint64_t n_layers_;
};

class MeshFactory {
public:
    static std::unique_ptr<Mesh> create_regular_mesh(const std::vector<uint64_t>& local_dims, 
                                                   const std::vector<uint64_t>& offset = {},
                                                   const std::vector<uint64_t>& global_dims = {});
    static std::unique_ptr<Mesh> create_extruded_mesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers = 0);
};

/**
 * @brief Unified encoding for simplices in regular meshes.
 */
template <typename DeviceMesh>
FTK_HOST_DEVICE inline uint64_t encode_simplex_id(const Simplex& s, const DeviceMesh& mesh) {
    uint64_t v_min = s.vertices[0];
    uint64_t c_min[4];
    mesh.id_to_coords(v_min, c_min);
    
    uint64_t combined_mask = 0;
    for (int i = 1; i <= s.dimension; ++i) {
        uint64_t ci[4]; mesh.id_to_coords(s.vertices[i], ci);
        uint64_t mask = 0;
        for (int k = 0; k < 4; ++k) if (ci[k] > c_min[k]) mask |= (1ULL << k);
        combined_mask |= (mask << ((i - 1) * 4));
    }
    return (v_min << 24) | (combined_mask & 0xFFFFFF);
}

} // namespace ftk2
