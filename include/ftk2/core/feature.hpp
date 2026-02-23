#pragma once

#include <ftk2/core/mesh.hpp>
#include <vector>
#include <array>
#include <ndarray/ndarray.hh>

namespace ftk2 {

/**
 * @brief Represents a geometric primitive within a simplex.
 */
enum class FeatureGeometryType {
    Point,      // k=0 manifold (e.g., Critical Point)
    Segment,    // k=1 manifold (e.g., PV segment, Isosurface edge)
    Polygon     // k=2 manifold (e.g., Isosurface face)
};

/**
 * @brief A unified, POD-compatible feature element.
 */
struct FeatureElement {
    Simplex simplex; 
    FeatureGeometryType geometry_type;

    float barycentric_coords[3][16]; 

    uint64_t track_id; 
    uint32_t type;     
    float scalar;   
    
    float attributes[16]; 
};

/**
 * @brief Represents a manifold trajectory across spacetime.
 */
struct Track {
    int id;
    int dimension; // k = (d+1) - m
    std::vector<FeatureElement> elements;

    /**
     * @brief Intersect the track manifold with a specific time step.
     */
    std::vector<FeatureElement> slice(double t, const Mesh& mesh) const {
        return {};
    }
};

} // namespace ftk2
