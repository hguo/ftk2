#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/zero_crossing.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/feature.hpp>
#include <ndarray/ndarray.hh>
#include <map>
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
};

// Forward declaration for device view
template <typename T> struct CudaDataView;

/**
 * @brief Predicate for finding critical points (zeros of a vector field).
 */
template <int M, typename T = double>
struct CriticalPointPredicate : public Predicate<M, T> {
    char var_names[M][32]; 
    char scalar_var_name[32]; 

    CriticalPointPredicate() {
        for(int i=0; i<M; ++i) var_names[i][0] = '\0';
        scalar_var_name[0] = '\0';
    }

    /**
     * @brief Extract feature elements from a simplex (CPU).
     */
    std::vector<FeatureElement> extract(const Simplex& s, 
                                       const std::map<std::string, ftk::ndarray<T>>& data,
                                       const Mesh& mesh) const 
    {
        T values[M + 1][M];
        uint64_t indices[M + 1];
        
        for (int i = 0; i <= M; ++i) {
            indices[i] = s.vertices[i];
            auto coords = mesh.get_vertex_coordinates(s.vertices[i]);
            for (int j = 0; j < M; ++j) {
                auto it = data.find(std::string(var_names[j]));
                if (it == data.end()) return {};

                if (coords.size() == 1) values[i][j] = it->second.f(coords[0]);
                else if (coords.size() == 2) values[i][j] = it->second.f(coords[0], coords[1]);
                else if (coords.size() == 3) values[i][j] = it->second.f(coords[0], coords[1], coords[2]);
                else if (coords.size() == 4) values[i][j] = it->second.f(coords[0], coords[1], coords[2], coords[3]);
            }
        }

        if (!sos::origin_inside<M, T>::check(values, indices)) return {};

        T lambda[M + 1];
        if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) {
            for (int i = 0; i <= M; ++i) lambda[i] = 1.0 / (M + 1);
        }

        FeatureElement el; el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];

        if (scalar_var_name[0] != '\0') {
            auto it = data.find(std::string(scalar_var_name));
            if (it != data.end()) {
                T s_val = 0;
                for (int i = 0; i <= M; ++i) {
                    auto coords = mesh.get_vertex_coordinates(s.vertices[i]);
                    T v = 0;
                    if (coords.size() == 2) v = it->second.f(coords[0], coords[1]);
                    else if (coords.size() == 3) v = it->second.f(coords[0], coords[1], coords[2]);
                    else if (coords.size() == 4) v = it->second.f(coords[0], coords[1], coords[2], coords[3]);
                    s_val += lambda[i] * v;
                }
                el.scalar = (float)s_val;
            }
        }
        return {el};
    }

#ifdef __CUDACC__
    /**
     * @brief Extract feature elements from a simplex (GPU).
     */
    __device__
    bool extract_device(const Simplex& s, 
                       const CudaDataView<T>* data,
                       int n_vars,
                       const RegularSimplicialMeshDevice& mesh,
                       FeatureElement& el) const 
    {
        T values[M + 1][M];
        uint64_t indices[M + 1];
        
        for (int i = 0; i <= M; ++i) {
            indices[i] = s.vertices[i];
            uint64_t coords[4] = {0};
            mesh.id_to_coords(indices[i], coords);
            for (int j = 0; j < M && j < n_vars; ++j) {
                values[i][j] = data[j].f(coords[0], coords[1], coords[2], coords[3]);
            }
        }

        if (!sos::origin_inside<M, T>::check(values, indices)) return false;

        T lambda[M + 1];
        if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) {
            for (int i = 0; i <= M; ++i) lambda[i] = 1.0 / (M + 1);
        }

        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        
        el.type = 0;
        el.scalar = 0.0f;
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
        
        // Scalar interpolation can be added here too
        
        return true; 
    }
#endif
};

/**
 * @brief Predicate for finding contours (isosurfaces of a scalar field).
 */
template <typename T = double>
struct ContourPredicate : public Predicate<1, T> {
    char var_name[32];
    T threshold = 0.0;

    ContourPredicate() { var_name[0] = '\0'; }

    std::vector<FeatureElement> extract(const Simplex& s, 
                                       const std::map<std::string, ftk::ndarray<T>>& data,
                                       const Mesh& mesh) const 
    {
        auto it = data.find(std::string(var_name));
        if (it == data.end()) return {};

        T values[2][1];
        uint64_t indices[2];
        for (int i = 0; i <= 1; ++i) {
            indices[i] = s.vertices[i];
            auto coords = mesh.get_vertex_coordinates(s.vertices[i]);
            if (coords.size() == 2) values[i][0] = it->second.f(coords[0], coords[1]) - threshold;
            else if (coords.size() == 3) values[i][0] = it->second.f(coords[0], coords[1], coords[2]) - threshold;
            else if (coords.size() == 4) values[i][0] = it->second.f(coords[0], coords[1], coords[2], coords[3]) - threshold;
        }

        if (!sos::origin_inside<1, T>::check(values, indices)) return {};

        T lambda[2];
        if (!ZeroCrossingSolver<1, T>::solve(values, lambda)) {
            for (int i = 0; i <= 1; ++i) lambda[i] = 0.5;
        }

        FeatureElement el;
        el.simplex = s;
        el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 1; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        return {el};
    }

#ifdef __CUDACC__
    __device__
    bool extract_device(const Simplex& s, 
                       const CudaDataView<T>* data,
                       int n_vars,
                       const RegularSimplicialMeshDevice& mesh,
                       FeatureElement& el) const 
    {
        T values[2][1];
        uint64_t indices[2];
        for (int i = 0; i <= 1; ++i) {
            indices[i] = s.vertices[i];
            uint64_t coords[4] = {0};
            mesh.id_to_coords(indices[i], coords);
            values[i][0] = data[0].f(coords[0], coords[1], coords[2], coords[3]) - threshold;
        }

        if (!sos::origin_inside<1, T>::check(values, indices)) return false;

        T lambda[2];
        if (!ZeroCrossingSolver<1, T>::solve(values, lambda)) {
            for (int i = 0; i <= 1; ++i) lambda[i] = 0.5;
        }

        el.simplex = s;
        el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 1; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        
        el.type = 0; // Default type
        el.scalar = 0.0f; // Could interpolate from data[0] if needed
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
        
        return true;
    }
#endif
};

/**
 * @brief Predicate for finding the intersection of two isosurfaces.
 */
template <typename T = double>
struct IsosurfaceIntersectionPredicate : public Predicate<2, T> {
    char var_names[2][32];
    T thresholds[2] = {0.0, 0.0};

    IsosurfaceIntersectionPredicate() {
        var_names[0][0] = '\0'; var_names[1][0] = '\0';
    }

    std::vector<FeatureElement> extract(const Simplex& s, 
                                       const std::map<std::string, ftk::ndarray<T>>& data,
                                       const Mesh& mesh) const 
    {
        T values[3][2];
        uint64_t indices[3];
        for (int i = 0; i <= 2; ++i) {
            indices[i] = s.vertices[i];
            auto coords = mesh.get_vertex_coordinates(s.vertices[i]);
            for (int j = 0; j < 2; ++j) {
                auto it = data.find(std::string(var_names[j]));
                if (it == data.end()) return {};
                if (coords.size() == 2) values[i][j] = it->second.f(coords[0], coords[1]) - thresholds[j];
                else if (coords.size() == 3) values[i][j] = it->second.f(coords[0], coords[1], coords[2]) - thresholds[j];
                else if (coords.size() == 4) values[i][j] = it->second.f(coords[0], coords[1], coords[2], coords[3]) - thresholds[j];
            }
        }

        if (!sos::origin_inside<2, T>::check(values, indices)) return {};

        T lambda[3];
        if (!ZeroCrossingSolver<2, T>::solve(values, lambda)) {
            for (int i = 0; i <= 2; ++i) lambda[i] = 1.0 / 3.0;
        }

        FeatureElement el;
        el.simplex = s;
        el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        return {el};
    }
};

} // namespace ftk2
