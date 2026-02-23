#pragma once

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
        return data[i0 + i1*s[1] + i2*s[2] + i3*s[3]];
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
    int signs[5];
    int pos_count = 0;
    for (int i = 0; i < 5; ++i) {
        uint64_t coords[4];
        mesh.id_to_coords(cell.vertices[i], coords);
        vals[i] = data[0].f(coords[0], coords[1], coords[2], coords[3]); // Assuming m=1, first var
        signs[i] = (vals[i] > 0) ? 1 : -1;
        if (signs[i] > 0) pos_count++;
    }

    // 2. Discover nodes on edges
    IDType node_ids[10]; // Map from edge (i,j) to global node ID
    int edge_node_count = 0;
    // ... (Extraction logic for nodes)

    // 3. Tetrahedralize based on split
    if (pos_count == 1 || pos_count == 4) {
        // Case 1: 4 nodes form a tetrahedron
        // Emit 1 tet to res.volumes
    } else if (pos_count == 2 || pos_count == 3) {
        // Case 2: 6 nodes form a triangular prism
        // Align bases and emit 3 tets
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
    uint64_t hc_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (hc_idx >= mesh.get_num_hypercubes()) return;

    int d = mesh.ndims;
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

} // namespace ftk2
