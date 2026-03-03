#pragma once

#ifdef __CUDACC__

#include <ftk2/core/device_mesh.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/feature.hpp>

namespace ftk2 {

// ─── Lookup tables: unique m-simplex types per d-hypercube (Freudenthal) ─────
// Each entry lists vertex offsets from the hypercube base vertex.
// The base vertex {0,...,0} is always included, guaranteeing unique ownership.

// d=2, m=1: 3 edges (2 vertices × 2 coords)
__constant__ int lut_2_1[3][2][2] = {
    {{0,0},{0,1}},  {{0,0},{1,0}},  {{0,0},{1,1}}
};

// d=2, m=2: 2 triangles (3 vertices × 2 coords)
__constant__ int lut_2_2[2][3][2] = {
    {{0,0},{1,0},{1,1}},  {{0,0},{0,1},{1,1}}
};

// d=3, m=1: 7 edges (2 vertices × 3 coords)
__constant__ int lut_3_1[7][2][3] = {
    {{0,0,0},{0,0,1}},  {{0,0,0},{0,1,0}},  {{0,0,0},{0,1,1}},  {{0,0,0},{1,0,0}},
    {{0,0,0},{1,0,1}},  {{0,0,0},{1,1,0}},  {{0,0,0},{1,1,1}}
};

// d=3, m=2: 12 triangles (3 vertices × 3 coords)
__constant__ int lut_3_2[12][3][3] = {
    {{0,0,0},{0,0,1},{0,1,1}},  {{0,0,0},{0,0,1},{1,0,1}},
    {{0,0,0},{0,0,1},{1,1,1}},  {{0,0,0},{0,1,0},{0,1,1}},
    {{0,0,0},{0,1,0},{1,1,0}},  {{0,0,0},{0,1,0},{1,1,1}},
    {{0,0,0},{0,1,1},{1,1,1}},  {{0,0,0},{1,0,0},{1,0,1}},
    {{0,0,0},{1,0,0},{1,1,0}},  {{0,0,0},{1,0,0},{1,1,1}},
    {{0,0,0},{1,0,1},{1,1,1}},  {{0,0,0},{1,1,0},{1,1,1}}
};

// d=3, m=3: 6 tetrahedra (4 vertices × 3 coords)
__constant__ int lut_3_3[6][4][3] = {
    {{0,0,0},{1,0,0},{1,1,0},{1,1,1}},  {{0,0,0},{1,0,0},{1,0,1},{1,1,1}},
    {{0,0,0},{0,1,0},{1,1,0},{1,1,1}},  {{0,0,0},{0,1,0},{0,1,1},{1,1,1}},
    {{0,0,0},{0,0,1},{1,0,1},{1,1,1}},  {{0,0,0},{0,0,1},{0,1,1},{1,1,1}}
};

// Returns the number of unique m-simplex types per d-hypercube, or 0 if unsupported.
FTK_HOST_DEVICE
inline int get_num_simplex_types(int d, int m) {
    if (d == 2) { if (m == 1) return 3; if (m == 2) return 2; }
    if (d == 3) { if (m == 1) return 7; if (m == 2) return 12; if (m == 3) return 6; }
    return 0;
}

/**
 * @brief Represents a raw data view for CUDA kernels.
 */
template <typename T>
struct CudaDataView {
    const T* data;
    uint64_t dims[4];
    uint64_t s[4]; // strides
    int ndims;

    FTK_HOST_DEVICE
    T f(uint64_t i0, uint64_t i1 = 0, uint64_t i2 = 0, uint64_t i3 = 0) const {
        return data[i0*s[0] + i1*s[1] + i2*s[2] + i3*s[3]];
    }
};

template <typename IDType, int Dim>
struct DeviceManifoldSimplex {
    IDType nodes[Dim + 1];
};

template <typename IDType>
struct CudaExtractionResult {
    FeatureElement* nodes;
    int* node_count;
    DeviceManifoldSimplex<IDType, 1>* edges;
    int* edge_count;
    DeviceManifoldSimplex<IDType, 2>* faces;
    int* face_count;
    DeviceManifoldSimplex<IDType, 3>* volumes;
    int* volume_count;
    int max_nodes;
    int max_conn;
};

/**
 * @brief CUDA implementation of Marching Tetrahedra logic.
 */
template <typename T, typename DeviceMesh, typename PredicateDevice, typename IDType>
__device__
void marching_tetrahedron_device(
    const Simplex& cell, 
    const DeviceMesh& mesh, 
    const PredicateDevice& pred, 
    const CudaDataView<T>* data,
    CudaExtractionResult<IDType>& res) 
{
    T vals[4];
    int A[4], B[4];
    int nA = 0, nB = 0;

    for (int i = 0; i < 4; ++i) {
        uint64_t coords[4];
        mesh.id_to_coords(cell.vertices[i], coords);
        vals[i] = data[0].f(coords[0], coords[1], coords[2], coords[3]) - pred.threshold;
        if (sos::sign(vals[i], cell.vertices[i], pred.sos_q) > 0) A[nA++] = i;
        else B[nB++] = i;
    }

    auto make_edge_simplex = [&](int i, int j) -> Simplex {
        Simplex s; s.dimension = 1;
        uint64_t v0 = cell.vertices[i], v1 = cell.vertices[j];
        if (v0 < v1) { s.vertices[0] = v0; s.vertices[1] = v1; }
        else { s.vertices[0] = v1; s.vertices[1] = v0; }
        return s;
    };

    if (nA == 1 || nB == 1) {
        int single = (nA == 1) ? A[0] : B[0];
        int others[3];
        if (nA == 1) { for(int i=0; i<3; ++i) others[i] = B[i]; }
        else { for(int i=0; i<3; ++i) others[i] = A[i]; }

        int idx = atomicAdd(res.face_count, 1);
        if (idx < res.max_conn) {
            for (int i = 0; i < 3; ++i) {
                res.faces[idx].nodes[i] = encode_simplex_id(make_edge_simplex(single, others[i]), mesh);
            }
        }
    } else if (nA == 2 && nB == 2) {
        int idx = atomicAdd(res.face_count, 2);
        if (idx + 1 < res.max_conn) {
            uint64_t nodes[4];
            int ni = 0;
            for (int i = 0; i < nA; ++i) {
                for (int j = 0; j < nB; ++j) {
                    nodes[ni++] = encode_simplex_id(make_edge_simplex(A[i], B[j]), mesh);
                }
            }
            // Sort nodes for canonical triangulation
            for (int i=0; i<4; ++i) {
                for (int j=i+1; j<4; ++j) {
                    if (nodes[i] > nodes[j]) {
                        uint64_t tmp = nodes[i]; nodes[i] = nodes[j]; nodes[j] = tmp;
                    }
                }
            }
            res.faces[idx].nodes[0] = nodes[0]; res.faces[idx].nodes[1] = nodes[1]; res.faces[idx].nodes[2] = nodes[2];
            res.faces[idx+1].nodes[0] = nodes[0]; res.faces[idx+1].nodes[1] = nodes[2]; res.faces[idx+1].nodes[2] = nodes[3];
        }
    }
}

/**
 * @brief CUDA implementation of Marching Pentatopes logic.
 */
template <typename T, typename DeviceMesh, typename PredicateDevice, typename IDType>
__device__
void marching_pentatope_device(
    const Simplex& cell, 
    const DeviceMesh& mesh, 
    const PredicateDevice& pred, 
    const CudaDataView<T>* data,
    CudaExtractionResult<IDType>& res) 
{
    T vals[5];
    int A[5], B[5];
    int nA = 0, nB = 0;

    for (int i = 0; i < 5; ++i) {
        uint64_t coords[4];
        mesh.id_to_coords(cell.vertices[i], coords);
        vals[i] = data[0].f(coords[0], coords[1], coords[2], coords[3]) - pred.threshold;
        if (sos::sign(vals[i], cell.vertices[i], pred.sos_q) > 0) A[nA++] = i;
        else B[nB++] = i;
    }

    auto make_edge_simplex = [&](int i, int j) -> Simplex {
        Simplex s; s.dimension = 1;
        uint64_t v0 = cell.vertices[i], v1 = cell.vertices[j];
        if (v0 < v1) { s.vertices[0] = v0; s.vertices[1] = v1; }
        else { s.vertices[0] = v1; s.vertices[1] = v0; }
        return s;
    };

    if (nA == 1 || nB == 1) {
        int single = (nA == 1) ? A[0] : B[0];
        int others[4];
        if (nA == 1) { for(int i=0; i<4; ++i) others[i] = B[i]; }
        else { for(int i=0; i<4; ++i) others[i] = A[i]; }

        int idx = atomicAdd(res.volume_count, 1);
        if (idx < res.max_conn) {
            for (int i = 0; i < 4; ++i) {
                res.volumes[idx].nodes[i] = encode_simplex_id(make_edge_simplex(single, others[i]), mesh);
            }
        }
    } else if (nA == 2 || nB == 2) {
        const int* two = (nA == 2) ? A : B;
        const int* three = (nA == 2) ? B : A;
        int idx = atomicAdd(res.volume_count, 3);
        if (idx + 2 < res.max_conn) {
            uint64_t T0[3], T1[3];
            for (int i = 0; i < 3; ++i) {
                T0[i] = encode_simplex_id(make_edge_simplex(two[0], three[i]), mesh);
                T1[i] = encode_simplex_id(make_edge_simplex(two[1], three[i]), mesh);
            }
            res.volumes[idx].nodes[0] = T0[0]; res.volumes[idx].nodes[1] = T0[1];
            res.volumes[idx].nodes[2] = T0[2]; res.volumes[idx].nodes[3] = T1[2];
            res.volumes[idx+1].nodes[0] = T0[0]; res.volumes[idx+1].nodes[1] = T0[1];
            res.volumes[idx+1].nodes[2] = T1[1]; res.volumes[idx+1].nodes[3] = T1[2];
            res.volumes[idx+2].nodes[0] = T0[0]; res.volumes[idx+2].nodes[1] = T1[0];
            res.volumes[idx+2].nodes[2] = T1[1]; res.volumes[idx+2].nodes[3] = T1[2];
        }
    }
}

/**
 * @brief Build an m-simplex from lookup table entry.
 * Reads vertex offsets from the appropriate __constant__ LUT and adds them
 * to the hypercube base coordinates to produce global vertex IDs.
 */
template <typename DeviceMesh>
__device__
inline bool build_simplex_from_lut(
    const DeviceMesh& mesh, int d, int m, int type,
    const uint64_t base[4], Simplex& s)
{
    s.dimension = m;
    uint64_t coords[4];
    for (int vi = 0; vi <= m; ++vi) {
        for (int k = 0; k < d; ++k) {
            int off;
            if (d == 2 && m == 1) off = lut_2_1[type][vi][k];
            else if (d == 2 && m == 2) off = lut_2_2[type][vi][k];
            else if (d == 3 && m == 1) off = lut_3_1[type][vi][k];
            else if (d == 3 && m == 2) off = lut_3_2[type][vi][k];
            else if (d == 3 && m == 3) off = lut_3_3[type][vi][k];
            else return false;
            coords[k] = base[k] + off;
        }
        for (int k = d; k < 4; ++k) coords[k] = 0;
        s.vertices[vi] = mesh.coords_to_id(coords);
    }
    s.sort_vertices();
    return true;
}

/**
 * @brief CUDA kernel for parallel feature extraction and manifold patching.
 *
 * One thread per (hypercube, simplex_type) pair.  Lookup tables guarantee
 * each m-simplex is processed exactly once (no ownership check needed).
 * Phase 2 (manifold patching) runs once per hypercube, guarded by type==0.
 */
template <typename DeviceMesh, typename PredicateDevice, typename T, typename IDType>
__global__ void extraction_kernel(
    DeviceMesh mesh,
    PredicateDevice pred,
    CudaDataView<T>* data_views,
    int n_vars,
    CudaExtractionResult<IDType> results)
{
    int d = mesh.ndims;
    constexpr int m = PredicateDevice::codimension;
    int n_types = get_num_simplex_types(d, m);

    // Fallback: unsupported (d,m) — should not happen for d=2,3
    if (n_types == 0) return;

    uint64_t n_hc = mesh.get_num_hypercubes();
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_hc * (uint64_t)n_types) return;

    uint64_t hc_idx = tid / (uint64_t)n_types;
    int type = (int)(tid % (uint64_t)n_types);

    // Decode hypercube index to base coordinates
    uint64_t base[4] = {0};
    uint64_t temp = hc_idx;
    for (int k = 0; k < d; ++k) {
        base[k] = temp % (mesh.local_dims[k] - 1) + mesh.offset[k];
        temp /= (mesh.local_dims[k] - 1);
    }

    // 1. Extract nodes on m-simplices via lookup table
    Simplex s;
    if (build_simplex_from_lut(mesh, d, m, type, base, s)) {
        FeatureElement el;
        if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
            int idx = atomicAdd(results.node_count, 1);
            if (idx < results.max_nodes) results.nodes[idx] = el;
        }
    }

    // 2. Perform top-level manifold patching (once per hypercube)
    if (type == 0) {
        int n_p = 1; for(int i=1; i<=d; ++i) n_p *= i;

        for (int p_idx = 0; p_idx < n_p; ++p_idx) {
            Simplex cell;
            mesh.get_d_simplex(hc_idx, p_idx, cell);

            if constexpr (PredicateDevice::codimension == 1) {
                if (d == 4) marching_pentatope_device<T>(cell, mesh, pred, data_views, results);
                else if (d == 3) marching_tetrahedron_device<T>(cell, mesh, pred, data_views, results);
            } else {
                uint64_t node_ids[32]; int node_masks[32]; int node_count = 0;
                int k = d - m; // resulting manifold dimension

                // Combination generator for (d+1) choose (m+1)
                int p[8]; for (int i = 0; i <= m; ++i) p[i] = i;
                while (p[0] <= d - m) {
                    Simplex face; face.dimension = m;
                    for (int i = 0; i <= m; ++i) face.vertices[i] = cell.vertices[p[i]];
                    face.sort_vertices();

                    FeatureElement el;
                    if (pred.extract_device(face, data_views, n_vars, mesh, el)) {
                        int mask = 0;
                        for (int i = 0; i <= m; ++i) {
                            for (int j = 0; j <= d; ++j) if (face.vertices[i] == cell.vertices[j]) mask |= (1 << j);
                        }
                        node_masks[node_count] = mask;
                        node_ids[node_count++] = encode_simplex_id(face, mesh);
                    }

                    int i = m;
                    while (i >= 0 && p[i] == d - m + i) i--;
                    if (i < 0) break;
                    p[i]++;
                    for (int j = i + 1; j <= m; j++) p[j] = p[i] + j - i;
                }

                if (node_count >= 2) {
                    // Sort node_ids for deterministic triangulation
                    for (int i=0; i<node_count; ++i) {
                        for (int j=i+1; j<node_count; ++j) {
                            if (node_ids[i] > node_ids[j]) {
                                uint64_t tmp = node_ids[i]; node_ids[i] = node_ids[j]; node_ids[j] = tmp;
                                int tmp_m = node_masks[i]; node_masks[i] = node_masks[j]; node_masks[j] = tmp_m;
                            }
                        }
                    }
                    uint64_t h_S = node_ids[0];

                    if (k == 1) {
                        for (int i = 1; i < node_count; ++i) {
                            int idx = atomicAdd(results.edge_count, 1);
                            if (idx < results.max_conn) {
                                results.edges[idx].nodes[0] = h_S;
                                results.edges[idx].nodes[1] = node_ids[i];
                            }
                        }
                    } else if (k == 2) {
                        // Watertight star-fan for k=2
                        for (int i = 0; i <= d; ++i) {
                            int tet_mask = ((1 << (d + 1)) - 1) ^ (1 << i);
                            uint64_t nodes_T[32]; int count_T = 0;
                            for (int j = 0; j < node_count; ++j) {
                                if ((node_masks[j] & tet_mask) == node_masks[j]) nodes_T[count_T++] = node_ids[j];
                            }
                            if (count_T < 2) continue;

                            uint64_t h_T = nodes_T[0];
                            for (int j = 1; j < count_T; ++j) {
                                uint64_t n_id = nodes_T[j];
                                if (h_S != h_T && h_S != n_id) {
                                    int idx = atomicAdd(results.face_count, 1);
                                    if (idx < results.max_conn) {
                                        results.faces[idx].nodes[0] = h_S;
                                        results.faces[idx].nodes[1] = h_T;
                                        results.faces[idx].nodes[2] = n_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief CUDA kernel for parallel feature extraction on unstructured meshes.
 */
template <typename PredicateDevice, typename T, typename IDType>
__global__ void extraction_kernel_unstructured(
    UnstructuredSimplicialMeshDevice mesh, 
    PredicateDevice pred, 
    CudaDataView<T>* data_views,
    int n_vars,
    CudaExtractionResult<IDType> results) 
{
    int m = PredicateDevice::codimension;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Extract nodes
    if (idx < mesh.get_num_simplices(m)) {
        const Simplex& s = mesh.get_simplex(m, idx);
        FeatureElement el;
        if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
            int n_idx = atomicAdd(results.node_count, 1);
            if (n_idx < results.max_nodes) results.nodes[n_idx] = el;
        }
    }
}

/**
 * @brief CUDA kernel for parallel feature extraction on extruded meshes.
 */
template <typename PredicateDevice, typename T, typename IDType>
__global__ void extraction_kernel_extruded(
    ExtrudedSimplicialMeshDevice mesh, 
    PredicateDevice pred, 
    CudaDataView<T>* data_views,
    int n_vars,
    CudaExtractionResult<IDType> results) 
{
    int m = PredicateDevice::codimension;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // A thread handles one spatial simplex and one layer
    uint64_t n_spatial_m = mesh.base_mesh.get_num_simplices(m);
    uint64_t n_spatial_m_minus_1 = (m > 0) ? mesh.base_mesh.get_num_simplices(m - 1) : 0;
    
    // 1. Discover nodes on spatial m-simplices at each timestep
    uint64_t n_total_spatial = n_spatial_m * (mesh.n_layers + 1);
    if (idx < n_total_spatial) {
        uint64_t s_idx = idx % n_spatial_m;
        uint64_t t = idx / n_spatial_m;
        Simplex s = mesh.base_mesh.get_simplex(m, s_idx);
        for (int i = 0; i <= m; ++i) s.vertices[i] += t * mesh.n_spatial_verts;
        
        FeatureElement el;
        if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
            int n_idx = atomicAdd(results.node_count, 1);
            if (n_idx < results.max_nodes) results.nodes[n_idx] = el;
        }
    }

    // 2. Discover nodes on spacetime m-simplices connecting t and t+1
    if (m > 0) {
        uint64_t n_total_spacetime = n_spatial_m_minus_1 * mesh.n_layers * m;
        if (idx >= n_total_spatial && idx < n_total_spatial + n_total_spacetime) {
            uint64_t i = idx - n_total_spatial;
            uint64_t s_idx = i % n_spatial_m_minus_1;
            uint64_t rem = i / n_spatial_m_minus_1;
            uint64_t t = rem / m;
            int j = rem % m; // j-th spacetime simplex from Kuhn subdivision

            Simplex base_s = mesh.base_mesh.get_simplex(m - 1, s_idx);
            Simplex s; s.dimension = m;
            for (int l = 0; l <= j; ++l) s.vertices[l] = base_s.vertices[l] + t * mesh.n_spatial_verts;
            for (int l = j; l < m; ++l) s.vertices[l + 1] = base_s.vertices[l] + (t + 1) * mesh.n_spatial_verts;
            s.sort_vertices();

            FeatureElement el;
            if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
                int n_idx = atomicAdd(results.node_count, 1);
                if (n_idx < results.max_nodes) results.nodes[n_idx] = el;
            }
        }
    }
}

} // namespace ftk2

#endif
