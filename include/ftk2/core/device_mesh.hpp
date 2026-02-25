#pragma once

#include <ftk2/core/mesh.hpp>

namespace ftk2 {

/**
 * @brief Lightweight, POD-compatible regular mesh for CUDA kernels.
 * 
 * Implements implicit indexing and simplicial subdivision on the device.
 */
struct RegularSimplicialMeshDevice {
    uint64_t local_dims[4];
    uint64_t offset[4];
    uint64_t global_dims[4];
    int ndims;

    /**
     * @brief Map global vertex ID back to global coordinates.
     */
    FTK_HOST_DEVICE 
    void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        coords[0] = 0; coords[1] = 0; coords[2] = 0; coords[3] = 0;
        for (int i = 0; i < ndims; ++i) {
            coords[i] = id % global_dims[i];
            id /= global_dims[i];
        }
    }

    /**
     * @brief Map global coordinates to a unique vertex ID.
     */
    FTK_HOST_DEVICE
    uint64_t coords_to_id(const uint64_t coords[4]) const {
        uint64_t id = 0;
        uint64_t multiplier = 1;
        for (int i = 0; i < ndims; ++i) {
            id += coords[i] * multiplier;
            multiplier *= global_dims[i];
        }
        return id;
    }

    /**
     * @brief Total number of vertices in the local mesh.
     */
    FTK_HOST_DEVICE
    uint64_t get_num_vertices() const {
        uint64_t n = 1;
        for (int i = 0; i < ndims; ++i) n *= local_dims[i];
        return n;
    }

    /**
     * @brief Get local coordinates of a vertex from its linear index.
     */
    FTK_HOST_DEVICE
    void get_vertex_coords_local(uint64_t index, uint64_t coords[4]) const {
        for (int i = 0; i < ndims; ++i) {
            coords[i] = index % local_dims[i];
            index /= local_dims[i];
        }
    }

    /**
     * @brief Total number of hypercubes in the local mesh.
     */
    FTK_HOST_DEVICE
    uint64_t get_num_hypercubes() const {
        uint64_t n = 1;
        for (int i = 0; i < ndims; ++i) n *= (local_dims[i] - 1);
        return n;
    }

    /**
     * @brief Get the base coordinates of a hypercube from its linear index.
     */
    FTK_HOST_DEVICE
    void get_hypercube_coords(uint64_t index, uint64_t coords[4]) const {
        for (int i = 0; i < ndims; ++i) {
            uint64_t size = local_dims[i] - 1;
            coords[i] = index % size;
            index /= size;
        }
    }

    FTK_HOST_DEVICE
    bool is_hypercube_base(const uint64_t coords[4]) const {
        for (int i = 0; i < ndims; ++i) if (coords[i] >= local_dims[i] - 1) return false;
        return true;
    }

    FTK_HOST_DEVICE
    uint64_t hypercube_coords_to_idx(const uint64_t coords[4]) const {
        uint64_t idx = 0;
        uint64_t multiplier = 1;
        for (int i = 0; i < ndims; ++i) {
            idx += coords[i] * multiplier;
            multiplier *= (local_dims[i] - 1);
        }
        return idx;
    }

    struct HypercubeCoords { uint64_t coords[4]; };
    FTK_HOST_DEVICE
    HypercubeCoords get_hypercube_coords_internal(uint64_t index) const {
        HypercubeCoords res = {{0, 0, 0, 0}};
        for (int i = 0; i < ndims; ++i) {
            uint64_t size = local_dims[i] - 1;
            res.coords[i] = index % size;
            index /= size;
        }
        return res;
    }

    /**
     * @brief Get a specific top-level d-simplex from a hypercube.
     * 
     * @param hc_index Index of the hypercube.
     * @param p_index Index of the permutation (0 to d!-1).
     * @param s Output simplex.
     */
    FTK_HOST_DEVICE
    void get_d_simplex(uint64_t hc_index, int p_index, Simplex& s) const {
        s.dimension = ndims;
        uint64_t base_coords[4];
        get_hypercube_coords(hc_index, base_coords);

        // Map relative base_coords to global coords using offset
        uint64_t global_base[4];
        for (int i = 0; i < ndims; ++i) global_base[i] = base_coords[i] + offset[i];

        // Generate permutation p from p_index
        int p[4];
        get_permutation(ndims, p_index, p);

        uint64_t curr[4];
        for (int i = 0; i < ndims; ++i) curr[i] = global_base[i];
        
        s.vertices[0] = coords_to_id(curr);
        for (int i = 0; i < ndims; ++i) {
            curr[p[i]] += 1;
            s.vertices[i + 1] = coords_to_id(curr);
        }
        
        // Simplicial tracking requires sorted vertices for uniqueness
        for (int i = 0; i <= ndims; ++i) {
            for (int j = i + 1; j <= ndims; ++j) {
                if (s.vertices[i] > s.vertices[j]) {
                    uint64_t tmp = s.vertices[i];
                    s.vertices[i] = s.vertices[j];
                    s.vertices[j] = tmp;
                }
            }
        }
    }

private:
    /**
     * @brief Compute the n-th permutation of [0...d-1].
     */
    FTK_HOST_DEVICE
    void get_permutation(int d, int n, int p[4]) const {
        int available[4] = {0, 1, 2, 3};
        int fact[5] = {1, 1, 2, 6, 24};
        for (int i = 0; i < d; ++i) {
            int idx = n / fact[d - 1 - i];
            p[i] = available[idx];
            for (int j = idx; j < 3; ++j) available[j] = available[j + 1];
            n %= fact[d - 1 - i];
        }
    }
};

/**
 * @brief Lightweight, POD-compatible unstructured mesh for CUDA kernels.
 */
struct UnstructuredSimplicialMeshDevice {
    const Simplex* simplices[4]; // Pointer to arrays of unique simplices for each dim 0..3
    uint64_t n_simplices[4];
    int spatial_dim;
    int cell_dim;

    FTK_HOST_DEVICE uint64_t get_num_vertices() const { return n_simplices[0]; }
    FTK_HOST_DEVICE uint64_t get_num_simplices(int k) const { return (k >= 0 && k <= cell_dim) ? n_simplices[k] : 0; }
    FTK_HOST_DEVICE const Simplex& get_simplex(int k, uint64_t idx) const { return simplices[k][idx]; }
    FTK_HOST_DEVICE int get_total_dimension() const { return cell_dim; }

    FTK_HOST_DEVICE void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        coords[0] = id; coords[1] = 0; coords[2] = 0; coords[3] = 0;
    }
    FTK_HOST_DEVICE uint64_t coords_to_id(const uint64_t coords[4]) const {
        return coords[0];
    }
};

/**
 * @brief Lightweight representation of an extruded mesh on device.
 */
struct ExtrudedSimplicialMeshDevice {
    UnstructuredSimplicialMeshDevice base_mesh;
    uint64_t n_layers;
    uint64_t n_spatial_verts;

    FTK_HOST_DEVICE uint64_t get_num_vertices() const { return n_spatial_verts * (n_layers + 1); }
    FTK_HOST_DEVICE void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        coords[0] = id; coords[1] = 0; coords[2] = 0; coords[3] = 0;
    }
    FTK_HOST_DEVICE uint64_t coords_to_id(const uint64_t coords[4]) const {
        return coords[0];
    }
};

} // namespace ftk2
