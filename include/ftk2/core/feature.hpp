#pragma once

#include <vector>
#include <string>

namespace ftk2 {

/**
 * @brief Represents a generic feature (e.g., a critical point, a vortex core).
 */
struct Feature {
    int id;
    int timestep;
    std::vector<double> position;
    double scalar_value; // e.g., velocity magnitude at this point

    Feature() : id(-1), timestep(-1), scalar_value(0.0) {}
};

/**
 * @brief Represents a manifold trajectory across spacetime.
 * 
 * A track is a collection of simplices (of dimension k) that 
 * form a connected manifold in the spacetime mesh.
 */
struct Track {
    int id;
    int dimension; // k = (d+1) - m
    std::vector<Simplex> simplices;

    /**
     * @brief Intersect the track manifold with a specific time step.
     * 
     * @param t The time step to slice at.
     * @return A collection of (k-1)-dimensional features at that time step.
     */
    std::vector<Feature> slice(double t) const {
        // Find all simplices in the track that span across time step t.
        // For each simplex, calculate the intersection point/segment/polygon.
        return {};
    }
};

} // namespace ftk2
