#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/zero_crossing.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/feature.hpp>
#include <ndarray/ndarray.hh>
#include <vector>
#include <string>

#ifdef __CUDACC__
#include <ftk2/core/device_mesh.hpp>
#endif

namespace ftk2 {

/**
 * @brief Base class for feature predicates.
 */
template <int M, typename T = double>
struct Predicate {
    static constexpr int codimension = M;
    double sos_q = 1000000.0;
};

// Forward declaration for device view
template <typename T> struct CudaDataView;

/**
 * @brief Attribute configuration for recording at feature locations
 */
struct AttributeSpec {
    std::string name;           // Attribute name
    std::string source;         // Source data array name
    std::string type = "scalar"; // scalar, magnitude, component_X
    int component = -1;         // For multi-component: which component
    int slot = -1;              // Which attributes[] slot to use (0-15)
};

/**
 * @brief Predicate for finding critical points (zeros of a vector field).
 *
 * Supports two data formats:
 * 1. Multi-component array: Single array with shape [M, spatial..., time]
 * 2. Separate arrays: M separate arrays (legacy, deprecated)
 */
template <int M, typename T = double>
struct CriticalPointPredicate : public Predicate<M, T> {
    // Multi-component mode (preferred)
    std::string vector_var_name;  // Name of multi-component vector array

    // Legacy mode (deprecated - separate scalar arrays)
    std::string var_names[M];

    // Optional scalar field for classification
    std::string scalar_var_name;

    // Data format flag
    bool use_multicomponent = true;  // Default to multi-component

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[M+1][M], FeatureElement& el, 
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const 
    {
        uint64_t indices[M + 1];
        for(int i=0; i<=M; ++i) indices[i] = s.vertices[i];
        if (!sos::origin_inside<M, T>::check(values, indices, this->sos_q)) return false;
        
        T lambda[M + 1];
        if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) return false; 
        
        el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];

        el.type = 0; el.scalar = 0.0f;
        if (!scalar_var_name.empty() && arrays.size() > M && mesh) {
            T s_val = 0;
            for (int i = 0; i <= M; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                const auto& arr = *arrays[M];
                if (coords.size() == 1) s_val += lambda[i] * arr.f(coords[0]);
                else if (coords.size() == 2) s_val += lambda[i] * arr.f(coords[0], coords[1]);
                else if (coords.size() == 3) s_val += lambda[i] * arr.f(coords[0], coords[1], coords[2]);
                else if (coords.size() == 4) s_val += lambda[i] * arr.f(coords[0], coords[1], coords[2], coords[3]);
            }
            el.scalar = (float)s_val;
        }

        // Initialize attributes to zero
        // (Actual attribute interpolation is handled by the engine after extract_it succeeds)
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;

        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = M;
        bool has_scalar;
        double sos_q;
        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[M + 1][M]; uint64_t indices[M + 1];
            for (int i = 0; i <= M; ++i) {
                indices[i] = s.vertices[i]; uint64_t coords[4] = {0}; mesh.id_to_coords(indices[i], coords);
                for (int j = 0; j < M && j < n_vars; ++j) values[i][j] = data[j].f(coords[0], coords[1], coords[2], coords[3]);
            }
            if (!sos::origin_inside<M, T>::check(values, indices, sos_q)) return false;
            T lambda[M + 1]; if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) { for (int i = 0; i <= M; ++i) lambda[i] = (T)1.0 / (M + 1); }
            el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
            el.type = 0; el.scalar = 0.0f;
            if (has_scalar && n_vars > M) {
                T s_val = 0; for (int i = 0; i <= M; ++i) { uint64_t coords[4] = {0}; mesh.id_to_coords(s.vertices[i], coords); s_val += lambda[i] * data[M].f(coords[0], coords[1], coords[2], coords[3]); }
                el.scalar = (float)s_val;
            }
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true; 
        }
    };
    Device get_device() const { return {!scalar_var_name.empty(), this->sos_q}; }
#endif
};

/**
 * @brief Predicate for finding contours (isosurfaces of a scalar field).
 */
template <typename T = double>
struct ContourPredicate : public Predicate<1, T> {
    std::string var_name;
    T threshold = (T)0.0;

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[2][1], FeatureElement& el,
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const 
    {
        uint64_t indices[2] = {s.vertices[0], s.vertices[1]};
        T adjusted_values[2][1] = {{values[0][0] - threshold}, {values[1][0] - threshold}};
        if (!sos::origin_inside<1, T>::check(adjusted_values, indices, this->sos_q)) return false;
        T lambda[2]; if (!ZeroCrossingSolver<1, T>::solve(adjusted_values, lambda)) { for (int i = 0; i <= 1; ++i) lambda[i] = (T)0.5; }
        el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 1; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        el.scalar = (float)threshold;
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = 1;
        T threshold;
        double sos_q;
        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[2][1]; uint64_t indices[2];
            for (int i = 0; i <= 1; ++i) {
                indices[i] = s.vertices[i]; uint64_t coords[4] = {0}; mesh.id_to_coords(indices[i], coords);
                values[i][0] = data[0].f(coords[0], coords[1], coords[2], coords[3]) - threshold;
            }
            if (!sos::origin_inside<1, T>::check(values, indices, sos_q)) return false;
            T lambda[2]; if (!ZeroCrossingSolver<1, T>::solve(values, lambda)) { for (int i = 0; i <= 1; ++i) lambda[i] = (T)0.5; }
            el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= 1; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
            el.type = 0; el.scalar = (float)threshold;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true;
        }
    };
    Device get_device() const { return {threshold, this->sos_q}; }
#endif
};

/**
 * @brief Predicate for finding fibers (intersections of two isosurfaces from two scalar fields).
 */
template <typename T = double>
struct FiberPredicate : public Predicate<2, T> {
    std::string var_names[2];
    T thresholds[2] = {(T)0.0, (T)0.0};

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[3][2], FeatureElement& el,
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const 
    {
        uint64_t indices[3] = {s.vertices[0], s.vertices[1], s.vertices[2]};
        T adj[3][2]; for(int i=0; i<3; ++i) for(int j=0; j<2; ++j) adj[i][j] = values[i][j] - thresholds[j];
        if (!sos::origin_inside<2, T>::check(adj, indices, this->sos_q)) return false;
        T lambda[3]; if (!ZeroCrossingSolver<2, T>::solve(adj, lambda)) { for (int i = 0; i <= 2; ++i) lambda[i] = (T)1.0 / 3.0; }
        
        el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        el.scalar = 0.0f;
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = 2;
        T thresholds[2];
        double sos_q;
        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[3][2]; uint64_t indices[3];
            for (int i = 0; i <= 2; ++i) {
                indices[i] = s.vertices[i]; uint64_t coords[4] = {0}; mesh.id_to_coords(indices[i], coords);
                for (int j = 0; j < 2 && j < n_vars; ++j) values[i][j] = data[j].f(coords[0], coords[1], coords[2], coords[3]) - thresholds[j];
            }
            if (!sos::origin_inside<2, T>::check(values, indices, sos_q)) return false;
            T lambda[3]; if (!ZeroCrossingSolver<2, T>::solve(values, lambda)) { for (int i = 0; i <= 2; ++i) lambda[i] = (T)1.0 / 3.0; }
            el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
            el.type = 0; el.scalar = 0.0f;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true;
        }
    };
    Device get_device() const { return {{thresholds[0], thresholds[1]}, this->sos_q}; }
#endif
};

} // namespace ftk2
