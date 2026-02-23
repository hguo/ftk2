#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <memory>
#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_set>

#ifdef __CUDACC__
#define FTK_HOST_DEVICE __host__ __device__
#else
#define FTK_HOST_DEVICE
#endif

namespace ftk2 {

/**
 * @brief Represents a K-dimensional simplex within a mesh.
 * Vertices are ALWAYS stored in increasing order of their IDs.
 */
struct Simplex {
    int dimension; // K
    uint64_t vertices[16]; // K+1 vertex IDs

    FTK_HOST_DEVICE int num_vertices() const { return dimension + 1; }
    
    /**
     * @brief Ensure vertices are sorted. Must be called after manual vertex assignment.
     */
    void sort_vertices() {
        std::sort(vertices, vertices + dimension + 1);
    }

    bool operator==(const Simplex& other) const {
        if (dimension != other.dimension) return false;
        for (int i = 0; i <= dimension; ++i) {
            if (vertices[i] != other.vertices[i]) return false;
        }
        return true;
    }
    
    bool operator<(const Simplex& other) const {
        if (dimension != other.dimension) return dimension < other.dimension;
        for (int i = 0; i <= dimension; ++i) {
            if (vertices[i] != other.vertices[i]) return vertices[i] < other.vertices[i];
        }
        return false;
    }
};

struct SimplexHash {
    std::size_t operator()(const Simplex& s) const {
        std::size_t h = 0;
        for (int i = 0; i <= s.dimension; ++i) {
            h ^= std::hash<uint64_t>{}(s.vertices[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
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
            // f.vertices are already sorted if s.vertices is sorted and p is increasing
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
 * @brief Regular simplicial mesh using Kuhn's triangulation.
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

        std::unordered_set<Simplex, SimplexHash> visited;
        std::vector<uint64_t> current_coords(d, 0);
        
        iterate_hypercubes(0, current_coords, [&](const std::vector<uint64_t>& base_coords) {
            std::vector<int> p(d);
            std::iota(p.begin(), p.end(), 0);
            do {
                Simplex top; top.dimension = d;
                std::vector<uint64_t> v_coords = base_coords;
                top.vertices[0] = coords_to_id(v_coords);
                for (int i = 0; i < d; ++i) {
                    v_coords[p[i]] += 1;
                    top.vertices[i + 1] = coords_to_id(v_coords);
                }
                top.sort_vertices(); // IMPORTANT: Ensure top-level simplex is sorted
                
                find_k_faces(top, k, [&](const Simplex& f) {
                    if (visited.find(f) == visited.end()) {
                        visited.insert(f);
                        callback(f);
                    }
                });
            } while (std::next_permutation(p.begin(), p.end()));
        });
    }

    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {}

    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override {
        std::vector<uint64_t> g_coords = id_to_coords(vertex_id);
        std::vector<double> phys_coords(g_coords.size());
        for (size_t i = 0; i < g_coords.size(); ++i) phys_coords[i] = static_cast<double>(g_coords[i]);
        return phys_coords;
    }

protected:
    uint64_t coords_to_id(const std::vector<uint64_t>& local_coords) const {
        uint64_t id = 0;
        uint64_t multiplier = 1;
        for (size_t i = 0; i < dims_.size(); ++i) {
            id += (local_coords[i] + offset_[i]) * multiplier;
            multiplier *= global_dims_[i];
        }
        return id;
    }

    std::vector<uint64_t> id_to_coords(uint64_t global_id) const {
        std::vector<uint64_t> coords(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            coords[i] = global_id % global_dims_[i];
            global_id /= global_dims_[i];
        }
        return coords;
    }

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
            // f.vertices are sorted because s.vertices is sorted and p is increasing
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

} // namespace ftk2
