#pragma once

#include <ndarray/ndarray.hh>
#include <cmath>

namespace ftk2 {

/**
 * @brief Compute cross product of two vector fields: W = U × V
 *
 * For 3D vectors: W = (u_y*v_z - u_z*v_y, u_z*v_x - u_x*v_z, u_x*v_y - u_y*v_x)
 * For 2D vectors: W_z = u_x*v_y - u_y*v_x (scalar result)
 *
 * @param u First vector field [M, spatial..., time]
 * @param v Second vector field [M, spatial..., time]
 * @param w Output cross product [M, spatial..., time] or [spatial..., time] for 2D
 */
template <typename T>
void cross_product_3d(const ftk::ndarray<T>& u, const ftk::ndarray<T>& v, ftk::ndarray<T>& w)
{
    // Assume shape: [3, spatial..., time]
    if (u.nd() < 2 || v.nd() < 2) {
        throw std::invalid_argument("cross_product_3d: arrays must have at least 2 dimensions [ncomp, ...]");
    }

    if (u.dimf(0) != 3 || v.dimf(0) != 3) {
        throw std::invalid_argument("cross_product_3d: first dimension must be 3 (vector components)");
    }

    // Output has same shape as input
    std::vector<size_t> shape;
    for (int d = 0; d < u.nd(); ++d) {
        shape.push_back(u.dimf(d));
    }
    w.reshapef(shape);

    // Compute cross product for each spatial-temporal point
    size_t n_points = u.nelem() / 3;  // Total number of spatial-temporal points

    for (size_t idx = 0; idx < n_points; ++idx) {
        // Get components
        T ux = u[idx * 3 + 0];
        T uy = u[idx * 3 + 1];
        T uz = u[idx * 3 + 2];

        T vx = v[idx * 3 + 0];
        T vy = v[idx * 3 + 1];
        T vz = v[idx * 3 + 2];

        // Cross product: W = U × V
        w[idx * 3 + 0] = uy * vz - uz * vy;  // w_x
        w[idx * 3 + 1] = uz * vx - ux * vz;  // w_y
        w[idx * 3 + 2] = ux * vy - uy * vx;  // w_z
    }
}

/**
 * @brief Compute cross product for 2D vector fields
 *
 * For 2D vectors u=(u_x, u_y) and v=(v_x, v_y), the cross product is:
 * w = u_x*v_y - u_y*v_x (scalar, perpendicular to plane)
 *
 * @param u First vector field [2, spatial..., time]
 * @param v Second vector field [2, spatial..., time]
 * @param w Output scalar field [spatial..., time]
 */
template <typename T>
void cross_product_2d(const ftk::ndarray<T>& u, const ftk::ndarray<T>& v, ftk::ndarray<T>& w)
{
    if (u.nd() < 2 || v.nd() < 2) {
        throw std::invalid_argument("cross_product_2d: arrays must have at least 2 dimensions [ncomp, ...]");
    }

    if (u.dimf(0) != 2 || v.dimf(0) != 2) {
        throw std::invalid_argument("cross_product_2d: first dimension must be 2 (vector components)");
    }

    // Output has one less dimension (no component dimension)
    std::vector<size_t> shape;
    for (int d = 1; d < u.nd(); ++d) {
        shape.push_back(u.dimf(d));
    }
    w.reshapef(shape);

    size_t n_points = u.nelem() / 2;

    for (size_t idx = 0; idx < n_points; ++idx) {
        T ux = u[idx * 2 + 0];
        T uy = u[idx * 2 + 1];

        T vx = v[idx * 2 + 0];
        T vy = v[idx * 2 + 1];

        // Cross product magnitude (scalar)
        w[idx] = ux * vy - uy * vx;
    }
}

/**
 * @brief Decompose multi-component array into separate components
 *
 * Input: [ncomp, spatial..., time]
 * Output: vector of ncomp arrays, each [spatial..., time]
 */
template <typename T>
std::vector<ftk::ndarray<T>> decompose_components(const ftk::ndarray<T>& multi)
{
    if (multi.nd() < 2) {
        throw std::invalid_argument("decompose_components: array must have at least 2 dimensions");
    }

    int ncomp = multi.dimf(0);
    std::vector<ftk::ndarray<T>> components(ncomp);

    // Shape without first dimension
    std::vector<size_t> shape;
    for (int d = 1; d < multi.nd(); ++d) {
        shape.push_back(multi.dimf(d));
    }

    size_t n_points = multi.nelem() / ncomp;

    for (int c = 0; c < ncomp; ++c) {
        components[c].reshapef(shape);

        for (size_t idx = 0; idx < n_points; ++idx) {
            components[c][idx] = multi[idx * ncomp + c];
        }
    }

    return components;
}

/**
 * @brief Compute magnitude of vector field
 *
 * @param vec Vector field [ncomp, spatial..., time]
 * @param mag Output magnitude [spatial..., time]
 */
template <typename T>
void compute_magnitude(const ftk::ndarray<T>& vec, ftk::ndarray<T>& mag)
{
    if (vec.nd() < 2) {
        throw std::invalid_argument("compute_magnitude: array must have at least 2 dimensions");
    }

    int ncomp = vec.dimf(0);

    std::vector<size_t> shape;
    for (int d = 1; d < vec.nd(); ++d) {
        shape.push_back(vec.dimf(d));
    }
    mag.reshapef(shape);

    size_t n_points = vec.nelem() / ncomp;

    for (size_t idx = 0; idx < n_points; ++idx) {
        T sum_sq = 0;
        for (int c = 0; c < ncomp; ++c) {
            T val = vec[idx * ncomp + c];
            sum_sq += val * val;
        }
        mag[idx] = std::sqrt(sum_sq);
    }
}

} // namespace ftk2
