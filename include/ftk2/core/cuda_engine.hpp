#pragma once

#ifdef __CUDACC__

#include <ftk2/core/device_mesh.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/feature.hpp>

namespace ftk2 {

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
 * @brief Unified encoding for simplices in regular meshes.
 */
template <typename DeviceMesh>
__device__ __host__ inline uint64_t encode_simplex_id(const Simplex& s, const DeviceMesh& mesh) {
    uint64_t v0 = s.vertices[0];
    uint64_t c0[4], ci[4];
    mesh.id_to_coords(v0, c0);
    uint64_t combined_mask = 0;
    for (int i = 1; i <= s.dimension; ++i) {
        mesh.id_to_coords(s.vertices[i], ci);
        uint64_t mask = 0;
        for (int k = 0; k < 4; ++k) if (ci[k] > c0[k]) mask |= (1ULL << k);
        combined_mask |= (mask << ((i - 1) * 4));
    }
    return (v0 << 24) | (combined_mask & 0xFFFFFF);
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
        if (sos::sign(vals[i], cell.vertices[i]) > 0) A[nA++] = i;
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
 * @brief CUDA kernel for parallel feature extraction and manifold patching.
 */
template <typename DeviceMesh, typename PredicateDevice, typename T, typename IDType>
__global__ void extraction_kernel(
    DeviceMesh mesh, 
    PredicateDevice pred, 
    CudaDataView<T>* data_views,
    int n_vars,
    CudaExtractionResult<IDType> results) 
{
    uint64_t v_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= mesh.get_num_vertices()) return;

    int d = mesh.ndims;
    int m = PredicateDevice::codimension;

    uint64_t local_coords[4], global_coords[4];
    mesh.get_vertex_coords_local(v_idx, local_coords);
    for (int k = 0; k < 4; ++k) global_coords[k] = local_coords[k] + mesh.offset[k];
    uint64_t v0 = mesh.coords_to_id(global_coords);

    // 1. Extract nodes on m-simplices
    if (m == 1) {
        for (int mask = 1; mask < (1 << d); ++mask) {
            uint64_t v1_c[4]; bool in = true;
            for (int k = 0; k < 4; ++k) {
                v1_c[k] = global_coords[k] + ((mask >> k) & 1);
                if (v1_c[k] >= mesh.global_dims[k]) in = false;
            }
            if (in) {
                Simplex s; s.dimension = 1;
                s.vertices[0] = v0; s.vertices[1] = mesh.coords_to_id(v1_c);
                s.sort_vertices();
                FeatureElement el;
                if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
                    int idx = atomicAdd(results.node_count, 1);
                    if (idx < results.max_nodes) results.nodes[idx] = el;
                }
            }
        }
    } else if (m == 2) {
        for (int m1 = 1; m1 < (1 << d); ++m1) {
            for (int m2 = 1; m2 < (1 << d); ++m2) {
                if ((m1 & m2) == m1 && m1 != m2) {
                    uint64_t v1_c[4], v2_c[4]; bool in = true;
                    for (int k = 0; k < 4; ++k) {
                        v1_c[k] = global_coords[k] + ((m1 >> k) & 1);
                        v2_c[k] = global_coords[k] + ((m2 >> k) & 1);
                        if (v1_c[k] >= mesh.global_dims[k] || v2_c[k] >= mesh.global_dims[k]) in = false;
                    }
                    if (in) {
                        Simplex s; s.dimension = 2;
                        s.vertices[0] = v0;
                        s.vertices[1] = mesh.coords_to_id(v1_c);
                        s.vertices[2] = mesh.coords_to_id(v2_c);
                        s.sort_vertices();
                        FeatureElement el;
                        if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
                            int idx = atomicAdd(results.node_count, 1);
                            if (idx < results.max_nodes) results.nodes[idx] = el;
                        }
                    }
                }
            }
        }
    } else if (m == 3) {
        for (int m1 = 1; m1 < (1 << d); ++m1) {
            for (int m2 = 1; m2 < (1 << d); ++m2) {
                if ((m1 & m2) == m1 && m1 != m2) {
                    for (int m3 = 1; m3 < (1 << d); ++m3) {
                        if ((m2 & m3) == m2 && m2 != m3) {
                            uint64_t v1_c[4], v2_c[4], v3_c[4]; bool in = true;
                            for (int k = 0; k < 4; ++k) {
                                v1_c[k] = global_coords[k] + ((m1 >> k) & 1);
                                v2_c[k] = global_coords[k] + ((m2 >> k) & 1);
                                v3_c[k] = global_coords[k] + ((m3 >> k) & 1);
                                if (v1_c[k] >= mesh.global_dims[k] || v2_c[k] >= mesh.global_dims[k] || v3_c[k] >= mesh.global_dims[k]) in = false;
                            }
                            if (in) {
                                Simplex s; s.dimension = 3;
                                s.vertices[0] = v0;
                                s.vertices[1] = mesh.coords_to_id(v1_c);
                                s.vertices[2] = mesh.coords_to_id(v2_c);
                                s.vertices[3] = mesh.coords_to_id(v3_c);
                                s.sort_vertices();
                                FeatureElement el;
                                if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
                                    int idx = atomicAdd(results.node_count, 1);
                                    if (idx < results.max_nodes) results.nodes[idx] = el;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Perform top-level manifold patching
    if (mesh.is_hypercube_base(local_coords)) {
        uint64_t hc_idx = mesh.hypercube_coords_to_idx(local_coords);
        int n_p = 1; for(int i=1; i<=d; ++i) n_p *= i;

        for (int p_idx = 0; p_idx < n_p; ++p_idx) {
            Simplex cell;
            mesh.get_d_simplex(hc_idx, p_idx, cell);
            
            if constexpr (PredicateDevice::codimension == 1) {
                if (d == 4) marching_pentatope_device<T>(cell, mesh, pred, data_views, results);
            } else {
                uint64_t node_ids[32]; int node_count = 0;
                if (m == 3 && d == 4) {
                    for (int i=0; i<5; ++i) {
                        Simplex face; face.dimension = 3;
                        int fj = 0;
                        for (int j=0; j<5; ++j) if (i != j) face.vertices[fj++] = cell.vertices[j];
                        face.sort_vertices();
                        FeatureElement el;
                        if (pred.extract_device(face, data_views, n_vars, mesh, el)) {
                            node_ids[node_count++] = encode_simplex_id(face, mesh);
                        }
                    }
                    if (node_count >= 2) {
                        // Canonical sort node_ids
                        for (int i=0; i<node_count; ++i) {
                            for (int j=i+1; j<node_count; ++j) {
                                if (node_ids[i] > node_ids[j]) {
                                    uint64_t tmp = node_ids[i]; node_ids[i] = node_ids[j]; node_ids[j] = tmp;
                                }
                            }
                        }
                        for (int i=1; i<node_count; ++i) {
                            int idx = atomicAdd(results.edge_count, 1);
                            if (idx < results.max_conn) {
                                results.edges[idx].nodes[0] = node_ids[0];
                                results.edges[idx].nodes[1] = node_ids[i];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace ftk2

#endif
