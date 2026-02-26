#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <cmath>

using namespace ftk2;

/**
 * @example unstructured_3d_manifold_debug
 *
 * Debug manifold stitching: trace which pentatopes should connect CPs
 */

int main(int argc, char** argv) {
    const int DW = 6, DH = 6, DD = 6, DT = 3;  // Small mesh

    std::cout << "=== MANIFOLD STITCHING DEBUG ===\n\n";

    // VERY slow trajectory to ensure overlap
    std::vector<std::array<double, 3>> trajectory(DT);
    std::cout << "[1/3] Trajectory (super slow motion):" << std::endl;
    for (int t = 0; t < DT; ++t) {
        trajectory[t][0] = 3.0 + t * 0.05;  // Move 0.05 units per timestep
        trajectory[t][1] = 3.0 + t * 0.05;
        trajectory[t][2] = 3.0 + t * 0.05;
        std::cout << "  t=" << t << ": (" << trajectory[t][0] << ", "
                  << trajectory[t][1] << ", " << trajectory[t][2] << ")" << std::endl;
    }

    // Create ExtrudedSimplicialMesh
    std::cout << "\n[2/3] Creating extruded spacetime mesh..." << std::endl;
    auto spatial_mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD}
    );
    auto spacetime_mesh = std::make_shared<ExtrudedSimplicialMesh>(spatial_mesh, DT - 1);

    // Generate field
    uint64_t n_vertices = spacetime_mesh->get_num_vertices();
    ftk::ndarray<double> u, v, w;
    u.reshapef({(size_t)n_vertices});
    v.reshapef({(size_t)n_vertices});
    w.reshapef({(size_t)n_vertices});

    std::cout << "[3/3] Assigning field..." << std::endl;
    for (uint64_t i = 0; i < n_vertices; ++i) {
        auto coords = spacetime_mesh->get_vertex_coordinates(i);
        double x = coords[0];
        double y = coords[1];
        double z = coords[2];
        double t = coords[3];

        // Linearly interpolate the center position based on time
        double cx, cy, cz;
        if (t <= 0) {
            cx = trajectory[0][0];
            cy = trajectory[0][1];
            cz = trajectory[0][2];
        } else if (t >= DT - 1) {
            cx = trajectory[DT-1][0];
            cy = trajectory[DT-1][1];
            cz = trajectory[DT-1][2];
        } else {
            int t0 = (int)t;
            int t1 = t0 + 1;
            double alpha = t - t0;  // interpolation weight
            cx = trajectory[t0][0] * (1 - alpha) + trajectory[t1][0] * alpha;
            cy = trajectory[t0][1] * (1 - alpha) + trajectory[t1][1] * alpha;
            cz = trajectory[t0][2] * (1 - alpha) + trajectory[t1][2] * alpha;
        }

        u[i] = x - cx;
        v[i] = y - cy;
        w[i] = z - cz;
    }

    std::map<std::string, ftk::ndarray<double>> data_map = {{"U", u}, {"V", v}, {"W", w}};

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";

    std::cout << "\nTracking critical points..." << std::endl;

    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(spacetime_mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    // Count unique tracks
    std::set<int> unique_tracks;
    for (const auto& vertex : cp_complex.vertices) {
        unique_tracks.insert(vertex.track_id);
    }

    // Results
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Detected nodes: " << cp_complex.vertices.size() << std::endl;
    std::cout << "Unique track IDs: " << unique_tracks.size() << std::endl;

    std::cout << "\nTrack IDs: {";
    for (int tid : unique_tracks) std::cout << tid << ", ";
    std::cout << "}" << std::endl;

    // Check connectivity
    int n_edges = 0;
    for (const auto& conn : cp_complex.connectivity) {
        if (conn.dimension == 1) {
            n_edges = conn.indices.size() / 2;
            std::cout << "\nTrajectory edges: " << n_edges << std::endl;
            if (n_edges <= 20) {  // Print if small enough
                std::cout << "Edges:" << std::endl;
                for (size_t i = 0; i < conn.indices.size(); i += 2) {
                    int v0 = conn.indices[i];
                    int v1 = conn.indices[i + 1];
                    const auto& n0 = cp_complex.vertices[v0];
                    const auto& n1 = cp_complex.vertices[v1];
                    std::cout << "  [" << v0 << "]<->[" << v1 << "]  "
                              << "track_id: " << n0.track_id << "<->" << n1.track_id << std::endl;
                }
            }
            break;
        }
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;

    if (unique_tracks.size() == 1) {
        std::cout << "✅ SUCCESS: ONE single trajectory!" << std::endl;
    } else {
        std::cout << "❌ FAILED: " << unique_tracks.size() << " disconnected trajectories" << std::endl;
        std::cout << "\nDEBUG INSIGHT:" << std::endl;
        std::cout << "Each timestep has detections, but they're not connecting." << std::endl;
        std::cout << "This suggests manifold stitching is not finding CPs in shared pentatopes." << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    write_complex_to_vtu(cp_complex, *spacetime_mesh, "manifold_debug_cp.vtu", 1);
    write_complex_to_vtp(cp_complex, *spacetime_mesh, "manifold_debug_cp.vtp");

    return (unique_tracks.size() == 1) ? 0 : 1;
}
