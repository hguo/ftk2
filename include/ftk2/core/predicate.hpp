#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/zero_crossing.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/feature.hpp>
#include <ndarray/ndarray.hh>
#include <map>
#include <string>

namespace ftk2 {

/**
 * @brief Base class for feature predicates.
 */
template <int M, typename T = double>
struct Predicate {
    static constexpr int codimension = M;
};

/**
 * @brief Predicate for finding critical points (zeros of a vector field).
 */
template <int M, typename T = double>
struct CriticalPointPredicate : public Predicate<M, T> {
    std::string var_names[M]; 
    std::string scalar_var_name; 

    /**
     * @brief Extract feature elements from a simplex.
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
                auto it = data.find(var_names[j]);
                if (it == data.end()) return {};

                if (coords.size() == 1) values[i][j] = it->second.f(coords[0]);
                else if (coords.size() == 2) values[i][j] = it->second.f(coords[0], coords[1]);
                else if (coords.size() == 3) values[i][j] = it->second.f(coords[0], coords[1], coords[2]);
                else if (coords.size() == 4) values[i][j] = it->second.f(coords[0], coords[1], coords[2], coords[3]);
            }
        }

        // 2. Perform Robust SoS check
        if (!sos::origin_inside<M, T>::check(values, indices)) {
            return {};
        }

        // 3. Solve for exact location
        T lambda[M + 1];
        if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) {
            // Topological fallback: Centroid if Solver fails but SoS says origin is inside
            for (int i = 0; i <= M; ++i) lambda[i] = 1.0 / (M + 1);
        }

        FeatureElement el;
        el.simplex = s;
        el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];

        // 4. Optional: Interpolate scalar field
        if (!scalar_var_name.empty()) {
            auto it = data.find(scalar_var_name);
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
};

/**
 * @brief Predicate for finding contours (isosurfaces of a scalar field).
 */
template <typename T = double>
struct ContourPredicate : public Predicate<1, T> {
    std::string var_name;
    T threshold = 0.0;

    std::vector<FeatureElement> extract(const Simplex& s, 
                                       const std::map<std::string, ftk::ndarray<T>>& data,
                                       const Mesh& mesh) const 
    {
        auto it = data.find(var_name);
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
};

/**
 * @brief Predicate for finding the intersection of two isosurfaces.
 */
template <typename T = double>
struct IsosurfaceIntersectionPredicate : public Predicate<2, T> {
    std::string var_names[2];
    T thresholds[2] = {0.0, 0.0};

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
                auto it = data.find(var_names[j]);
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
