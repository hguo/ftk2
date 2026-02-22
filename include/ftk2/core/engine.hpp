#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <map>
#include <vector>

namespace ftk2 {

/**
 * @brief The Unified Simplicial Engine for FTK2.
 * 
 * This engine unifies extraction and tracking into a single 
 * simplicial topological labeling process.
 */
template <typename Mesh, typename Predicate>
class SimplicialEngine {
public:
    SimplicialEngine(std::shared_ptr<Mesh> mesh) : mesh_(mesh) {}

    /**
     * @brief Execute the extraction and tracking pipeline.
     */
    void execute() {
        int m = Predicate::codimension;

        // 1. Extraction (Map Phase)
        // Find all m-simplices that contain feature elements.
        mesh_->iterate_simplices(m, [&](const Simplex& s) {
            auto elements = predicate_.extract(s); 
            if (!elements.empty()) {
                active_simplices_[s] = elements;
            }
        });

        // 2. Tracking (Merge Phase)
        // Stitch feature elements across (m+1)-dimensional cofaces.
        mesh_->iterate_simplices(m + 1, [&](const Simplex& coface) {
            std::vector<Simplex> feature_faces;
            mesh_->faces(coface, [&](const Simplex& f) {
                if (active_simplices_.count(f)) {
                    feature_faces.push_back(f);
                }
            });

            // Connect feature elements within this coface
            // This is the core "Tracking" logic.
            for (size_t i = 1; i < feature_faces.size(); ++i) {
                stitch(feature_faces[0], feature_faces[i]);
            }
        });
    }

private:
    /**
     * @brief Stitch two feature-containing simplices together.
     * This updates a Union-Find structure to group them into a track.
     */
    void stitch(const Simplex& s1, const Simplex& s2) {
        // Union-Find merge(s1, s2)
    }

    std::shared_ptr<Mesh> mesh_;
    Predicate predicate_;
    std::map<Simplex, std::vector<Feature>> active_simplices_;
};

} // namespace ftk2
