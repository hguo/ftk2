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

    uint64_t local_coords[4] = {0}, global_coords[4] = {0};
    mesh.get_vertex_coords_local(v_idx, local_coords);
    for (int k = 0; k < d; ++k) global_coords[k] = local_coords[k] + mesh.offset[k];
    uint64_t v0 = mesh.coords_to_id(global_coords);

    // 1. Extract nodes on m-simplices
    if (m == 1) {
        for (int m1 = 1; m1 < (1 << d); ++m1) {
            uint64_t g1[4]; bool in = true;
            for (int j = 0; j < d; ++j) { 
                g1[j] = global_coords[j] + ((m1 >> j) & 1); 
                if (g1[j] >= mesh.offset[j] + mesh.local_dims[j]) in = false; 
                if (g1[j] < mesh.offset[j]) in = false;
            }
            if (in) {
                Simplex s; s.dimension = 1; s.vertices[0] = v0; s.vertices[1] = mesh.coords_to_id(g1);
                s.sort_vertices();
                if (s.vertices[0] == v0) {
                    FeatureElement el;
                    if (pred.extract_device(s, data_views, n_vars, mesh, el)) {
                        int idx = atomicAdd(results.node_count, 1);
                        if (idx < results.max_nodes) results.nodes[idx] = el;
                    }
                }
            }
        }
    } else if (m == 2) {
        for (int m1 = 1; m1 < (1 << d); ++m1) {
            for (int m2 = m1 + 1; m2 < (1 << d); ++m2) {
                if ((m1 & m2) == m1) {
                    uint64_t g1[4], g2[4]; bool in = true;
                    for (int j = 0; j < d; ++j) {
                        g1[j] = global_coords[j] + ((m1 >> j) & 1);
                        g2[j] = global_coords[j] + ((m2 >> j) & 1);
                        if (g1[j] >= mesh.offset[j] + mesh.local_dims[j] || g2[j] >= mesh.offset[j] + mesh.local_dims[j]) in = false;
                        if (g1[j] < mesh.offset[j] || g2[j] < mesh.offset[j]) in = false;
                    }
                    if (in) {
                        Simplex s; s.dimension = 2; s.vertices[0] = v0;
                        s.vertices[1] = mesh.coords_to_id(g1);
                        s.vertices[2] = mesh.coords_to_id(g2);
                        s.sort_vertices();
                        if (s.vertices[0] == v0) {
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
    } else if (m == 3) {
        for (int m1 = 1; m1 < (1 << d); ++m1) {
            for (int m2 = m1 + 1; m2 < (1 << d); ++m2) {
                if ((m1 & m2) == m1) {
                    for (int m3 = m2 + 1; m3 < (1 << d); ++m3) {
                        if ((m2 & m3) == m2) {
                            uint64_t g1[4], g2[4], g3[4]; bool in = true;
                            for (int j = 0; j < d; ++j) {
                                g1[j] = global_coords[j] + ((m1 >> j) & 1);
                                g2[j] = global_coords[j] + ((m2 >> j) & 1);
                                g3[j] = global_coords[j] + ((m3 >> j) & 1);
                                if (g1[j] >= mesh.offset[j] + mesh.local_dims[j] || 
                                    g2[j] >= mesh.offset[j] + mesh.local_dims[j] || 
                                    g3[j] >= mesh.offset[j] + mesh.local_dims[j]) in = false;
                                if (g1[j] < mesh.offset[j] || g2[j] < mesh.offset[j] || g3[j] < mesh.offset[j]) in = false;
                            }
                            if (in) {
                                Simplex s; s.dimension = 3; s.vertices[0] = v0;
                                s.vertices[1] = mesh.coords_to_id(g1);
                                s.vertices[2] = mesh.coords_to_id(g2);
                                s.vertices[3] = mesh.coords_to_id(g3);
                                s.sort_vertices();
                                if (s.vertices[0] == v0) {
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

} // namespace ftk2

#endif
