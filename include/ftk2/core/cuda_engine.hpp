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

/**
 * @brief Generic extraction kernel for CUDA.
 */
template <typename DeviceMesh, typename DevicePredicate, typename T>
__global__ void extraction_kernel(
    DeviceMesh mesh, 
    DevicePredicate pred, 
    CudaDataView<T> data_views[4], // Fixed-size for simplicity
    int n_vars,
    FeatureElement* output,
    int* count,
    int max_output) 
{
    uint64_t hc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hc_idx >= mesh.get_num_hypercubes()) return;

    int d = mesh.ndims;
    int n_simplices_per_hc = 1; // For d=2, it's 2; for d=3, it's 6...
    for(int i=1; i<=d; ++i) n_simplices_per_hc *= i;

    for (int p_idx = 0; p_idx < n_simplices_per_hc; ++p_idx) {
        Simplex s;
        mesh.get_d_simplex(hc_idx, p_idx, s);

        // For Milestone 2, we assume DevicePredicate handles its own data access 
        // using the data_views provided.
        auto elements = pred.extract_device(s, data_views, n_vars, mesh);
        
        for (const auto& el : elements) {
            int idx = atomicAdd(count, 1);
            if (idx < max_output) {
                output[idx] = el;
            }
        }
    }
}

} // namespace ftk2
