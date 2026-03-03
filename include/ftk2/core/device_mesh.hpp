#pragma once

#include <ftk2/core/mesh.hpp>

namespace ftk2 {

/**
 * @brief Lightweight, POD-compatible regular mesh for CUDA kernels.
 */
struct RegularSimplicialMeshDevice {
    uint64_t local_dims[4];
    uint64_t offset[4];
    uint64_t global_dims[4];
    int ndims;

    FTK_HOST_DEVICE void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        coords[0] = 0; coords[1] = 0; coords[2] = 0; coords[3] = 0;
        for (int i = 0; i < ndims; ++i) { coords[i] = id % global_dims[i]; id /= global_dims[i]; }
    }

    FTK_HOST_DEVICE uint64_t coords_to_id(const uint64_t coords[4]) const {
        uint64_t id = 0; uint64_t multiplier = 1;
        for (int i = 0; i < ndims; ++i) { id += coords[i] * multiplier; multiplier *= global_dims[i]; }
        return id;
    }

    FTK_HOST_DEVICE uint64_t get_num_vertices() const {
        uint64_t n = 1; for (int i = 0; i < ndims; ++i) n *= local_dims[i]; return n;
    }

    FTK_HOST_DEVICE void get_vertex_coords_local(uint64_t index, uint64_t coords[4]) const {
        for (int i = 0; i < ndims; ++i) { coords[i] = index % local_dims[i]; index /= local_dims[i]; }
    }

    FTK_HOST_DEVICE uint64_t get_num_hypercubes() const {
        uint64_t n = 1;
        for (int i = 0; i < ndims; ++i) n *= (local_dims[i] - 1);
        return n;
    }

    FTK_HOST_DEVICE bool is_hypercube_base(const uint64_t coords[4]) const {
        for (int i = 0; i < ndims; ++i) if (coords[i] >= local_dims[i] - 1) return false;
        return true;
    }

    FTK_HOST_DEVICE uint64_t hypercube_coords_to_idx(const uint64_t coords[4]) const {
        uint64_t idx = 0; uint64_t multiplier = 1;
        for (int i = 0; i < ndims; ++i) { idx += coords[i] * multiplier; multiplier *= (local_dims[i] - 1); }
        return idx;
    }

    FTK_HOST_DEVICE void get_d_simplex(uint64_t hc_index, int p_index, Simplex& s) const {
        s.dimension = ndims;
        uint64_t base[4]; uint64_t temp_idx = hc_index;
        for (int i = 0; i < ndims; ++i) { base[i] = temp_idx % (local_dims[i] - 1) + offset[i]; temp_idx /= (local_dims[i] - 1); }
        
        int p[4]; int available[4] = {0, 1, 2, 3}; int fact[5] = {1, 1, 2, 6, 24};
        int n = p_index;
        for (int i = 0; i < ndims; ++i) { int idx = n / fact[ndims - 1 - i]; p[i] = available[idx]; for (int j = idx; j < 3; ++j) available[j] = available[j + 1]; n %= fact[ndims - 1 - i]; }

        uint64_t curr[4]; for (int i = 0; i < ndims; ++i) curr[i] = base[i];
        s.vertices[0] = coords_to_id(curr);
        for (int i = 0; i < ndims; ++i) { curr[p[i]]++; s.vertices[i + 1] = coords_to_id(curr); }
        s.sort_vertices();
    }
};

/**
 * @brief Lightweight, POD-compatible unstructured mesh for CUDA kernels.
 */
struct UnstructuredSimplicialMeshDevice {
    const Simplex* simplices[4]; 
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
    FTK_HOST_DEVICE int get_total_dimension() const { return base_mesh.cell_dim + 1; }
    FTK_HOST_DEVICE void id_to_coords(uint64_t id, uint64_t coords[4]) const {
        coords[0] = id; coords[1] = 0; coords[2] = 0; coords[3] = 0;
    }
    FTK_HOST_DEVICE uint64_t coords_to_id(const uint64_t coords[4]) const {
        return coords[0];
    }
};

/**
 * @brief Helper to encode a simplex into a unique ID on device.
 */
template <typename DeviceMesh>
__device__
inline uint64_t encode_simplex_id(const Simplex& s, const DeviceMesh& mesh) {
    uint64_t v_min = s.vertices[0];
    for (int i = 1; i <= s.dimension; ++i) if (s.vertices[i] < v_min) v_min = s.vertices[i];
    
    uint64_t c_min[4] = {0}; mesh.id_to_coords(v_min, c_min);
    uint64_t combined_mask = 0;
    for (int i = 0; i <= s.dimension; ++i) {
        if (s.vertices[i] == v_min) continue;
        uint64_t ci[4] = {0}; mesh.id_to_coords(s.vertices[i], ci);
        uint64_t mask = 0;
        // For regular meshes, bitmask relative to c_min. 
        // For unstructured, we'll just use vertex IDs which are unique.
        // Actually, Simplex already has sorted vertices. 
        // We need a stable ID for Simplex.
        for (int k = 0; k < 4; ++k) if (ci[k] > c_min[k]) mask |= (1 << k);
        combined_mask = (combined_mask << 4) | (mask & 0xF);
    }
    return (v_min << 24) | (combined_mask & 0xFFFFFF);
}

} // namespace ftk2
