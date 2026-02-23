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
 * @brief CUDA implementation of Marching Pentatopes logic.
 */
template <typename T, typename DeviceMesh, typename DevicePredicate, typename IDType>
__device__
void marching_pentatope_device(
    const Simplex& cell, 
    const DeviceMesh& mesh, 
    const DevicePredicate& pred, 
    const CudaDataView<T>* data,
    CudaExtractionResult<IDType>& res) 
{
    // 1. Collect values and determine signs
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

    auto encode_edge = [&](int i, int j) -> uint64_t {
        uint64_t v0 = cell.vertices[i], v1 = cell.vertices[j];
        if (v0 > v1) { uint64_t tmp = v0; v0 = v1; v1 = tmp; }
        
        uint64_t c0[4], c1[4];
        mesh.id_to_coords(v0, c0);
        mesh.id_to_coords(v1, c1);
        
        uint64_t mask = 0;
        for (int k = 0; k < 4; ++k) if (c1[k] > c0[k]) mask |= (1 << k);
        
        return (v0 << 4) | (mask & 0xF);
    };

    // 2. Discover nodes on edges and emit manifold simplices
    if (nA == 1 || nB == 1) {
        int single = (nA == 1) ? A[0] : B[0];
        int others[4];
        if (nA == 1) { for(int i=0; i<4; ++i) others[i] = B[i]; }
        else { for(int i=0; i<4; ++i) others[i] = A[i]; }

        // Case 1: 4 nodes form a tetrahedron
        int idx = atomicAdd(res.volume_count, 1);
        if (idx < res.max_conn) {
            for (int i = 0; i < 4; ++i) {
                res.volumes[idx].nodes[i] = encode_edge(single, others[i]);
            }
        }
    } else if (nA == 2 || nB == 2) {
        // Case 2: 6 nodes form a triangular prism (split into 3 tets)
        const int* two = (nA == 2) ? A : B;
        const int* three = (nA == 2) ? B : A;

        int idx = atomicAdd(res.volume_count, 3);
        if (idx + 2 < res.max_conn) {
            uint64_t T0[3], T1[3];
            for (int i = 0; i < 3; ++i) {
                T0[i] = encode_edge(two[0], three[i]);
                T1[i] = encode_edge(two[1], three[i]);
            }
            // Tet 1
            res.volumes[idx].nodes[0] = T0[0]; res.volumes[idx].nodes[1] = T0[1];
            res.volumes[idx].nodes[2] = T0[2]; res.volumes[idx].nodes[3] = T1[2];
            // Tet 2
            res.volumes[idx+1].nodes[0] = T0[0]; res.volumes[idx+1].nodes[1] = T0[1];
            res.volumes[idx+1].nodes[2] = T1[1]; res.volumes[idx+1].nodes[3] = T1[2];
            // Tet 3
            res.volumes[idx+2].nodes[0] = T0[0]; res.volumes[idx+2].nodes[1] = T1[0];
            res.volumes[idx+2].nodes[2] = T1[1]; res.volumes[idx+2].nodes[3] = T1[2];
        }
    }
}

/**
 * @brief CUDA kernel for parallel feature extraction and manifold patching.
 */
template <typename DeviceMesh, typename DevicePredicate, typename T, typename IDType>
__global__ void extraction_kernel(
    DeviceMesh mesh, 
    DevicePredicate pred, 
    CudaDataView<T>* data_views,
    int n_vars,
    CudaExtractionResult<IDType> results) 
{
    uint64_t v_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= mesh.get_num_vertices()) return;

    int d = mesh.ndims;
    int m = DevicePredicate::codimension;

    uint64_t local_coords[4], global_coords[4];
    mesh.get_vertex_coords_local(v_idx, local_coords);
    for (int k = 0; k < 4; ++k) global_coords[k] = local_coords[k] + mesh.offset[k];
    uint64_t v0 = mesh.coords_to_id(global_coords);

    // 1. Extract nodes on m-simplices (e.g. edges if m=1)
    if (m == 1) {
        for (int mask = 1; mask < (1 << d); ++mask) {
            uint64_t v1_coords[4];
            bool in_bounds = true;
            for (int k = 0; k < 4; ++k) {
                v1_coords[k] = global_coords[k] + ((mask >> k) & 1);
                if (v1_coords[k] >= mesh.global_dims[k]) in_bounds = false;
            }

            if (in_bounds) {
                uint64_t v1 = mesh.coords_to_id(v1_coords);
                Simplex edge; edge.dimension = 1;
                edge.vertices[0] = v0; edge.vertices[1] = v1;

                FeatureElement el;
                if (pred.extract_device(edge, data_views, n_vars, mesh, el)) {
                    int idx = atomicAdd(results.node_count, 1);
                    if (idx < results.max_nodes) results.nodes[idx] = el;
                }
            }
        }
    }

    // 2. Perform top-level manifold patching
    if (mesh.is_hypercube_base(local_coords)) {
        uint64_t hc_idx = mesh.hypercube_coords_to_idx(local_coords);
        int n_simplices_per_hc = 1;
        for(int i=1; i<=d; ++i) n_simplices_per_hc *= i;

        for (int p_idx = 0; p_idx < n_simplices_per_hc; ++p_idx) {
            Simplex cell;
            mesh.get_d_simplex(hc_idx, p_idx, cell);
            
            if constexpr (DevicePredicate::codimension == 1 && sizeof(IDType) >= 8) {
                if (d == 4) marching_pentatope_device<T>(cell, mesh, pred, data_views, results);
            }
        }
    }
}

} // namespace ftk2

#endif
