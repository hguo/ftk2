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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ftk2;

/**
 * @example unstructured_3d_extruded_test
 *
 * Use ExtrudedSimplicialMesh to explicitly connect timesteps
 * (like old FTK's simplicial_unstructured_extruded_3d_mesh)
 */

int main(int argc, char** argv) {
    const int DW = 12, DH = 12, DD = 12, DT = 5;

    std::cout << "=== EXTRUDED MESH TEST ===" << std::endl;
    std::cout << "Using ExtrudedSimplicialMesh to connect timesteps\n" << std::endl;

    // Define trajectory
    std::vector<std::array<double, 3>> trajectory(DT);
    std::cout << "[1/4] Trajectory:" << std::endl;
    for (int t = 0; t < DT; ++t) {
        double theta = 2.0 * M_PI * t / DT;
        trajectory[t][0] = DW / 2.0 + 2.5 * std::cos(theta);
        trajectory[t][1] = DH / 2.0 + 2.5 * std::sin(theta);
        trajectory[t][2] = DD / 2.0 + 2.0 * std::sin(2.0 * theta);
        std::cout << "  t=" << t << ": (" << trajectory[t][0] << ", "
                  << trajectory[t][1] << ", " << trajectory[t][2] << ")" << std::endl;
    }

    // Create SPATIAL mesh (3D only)
    std::cout << "\n[2/4] Creating spatial mesh..." << std::endl;
    auto spatial_mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD}
    );
    std::cout << "  Spatial dimensions: " << DW << "×" << DH << "×" << DD << std::endl;

    // Extrude in time
    std::cout << "\n[3/4] Extruding mesh in time..." << std::endl;
    auto spacetime_mesh = std::make_shared<ExtrudedSimplicialMesh>(spatial_mesh, DT - 1);
    std::cout << "  Total dimensions: " << spacetime_mesh->get_total_dimension() << "D" << std::endl;
    std::cout << "  Total vertices: " << spacetime_mesh->get_num_vertices() << std::endl;

    // Generate analytical field
    uint64_t n_vertices = spacetime_mesh->get_num_vertices();
    ftk::ndarray<double> u, v, w;
    u.reshapef({(size_t)n_vertices});
    v.reshapef({(size_t)n_vertices});
    w.reshapef({(size_t)n_vertices});

    std::cout << "\n[4/4] Assigning analytical field values..." << std::endl;
    for (uint64_t i = 0; i < n_vertices; ++i) {
        auto coords = spacetime_mesh->get_vertex_coordinates(i);
        double x = coords[0];
        double y = coords[1];
        double z = coords[2];
        double t = coords[3]; // Time is last coordinate

        // Find timestep
        int t_idx = (int)(t + 0.5);
        if (t_idx < 0) t_idx = 0;
        if (t_idx >= DT) t_idx = DT - 1;

        double cx = trajectory[t_idx][0];
        double cy = trajectory[t_idx][1];
        double cz = trajectory[t_idx][2];

        u[i] = x - cx;
        v[i] = y - cy;
        w[i] = z - cz;
    }

    std::map<std::string, ftk::ndarray<double>> data_map = {
        {"U", u},
        {"V", v},
        {"W", w}
    };

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";

    std::cout << "\nTracking critical points..." << std::endl;
    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(spacetime_mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    // Count trajectory segments and unique tracks
    int n_trajectory_segments = 0;
    for (const auto& conn : cp_complex.connectivity) {
        if (conn.dimension == 1) {
            n_trajectory_segments = conn.indices.size() / 2;
            break;
        }
    }

    std::set<int> unique_tracks;
    for (const auto& vertex : cp_complex.vertices) {
        unique_tracks.insert(vertex.track_id);
    }

    write_complex_to_vtu(cp_complex, *spacetime_mesh, "extruded_test_3d_cp.vtu", 1);
    write_complex_to_vtp(cp_complex, *spacetime_mesh, "extruded_test_3d_cp.vtp");

    // Results
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "                       RESULTS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nDetected nodes:          " << cp_complex.vertices.size() << std::endl;
    std::cout << "Trajectory segments:     " << n_trajectory_segments << std::endl;
    std::cout << "Unique track IDs:        " << unique_tracks.size() << std::endl;

    std::cout << "\n" << std::string(70, '-') << std::endl;
    if (unique_tracks.size() == 1) {
        std::cout << "✅ SUCCESS: ONE single trajectory!" << std::endl;
        std::cout << "   All " << cp_complex.vertices.size() << " nodes connected" << std::endl;
    } else {
        std::cout << "❌ FAILED: " << unique_tracks.size() << " disconnected trajectories" << std::endl;
        std::cout << "   Expected: 1 trajectory" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nKEY INSIGHT:" << std::endl;
    std::cout << "  ExtrudedSimplicialMesh explicitly creates spacetime simplices" << std::endl;
    std::cout << "  connecting spatial simplices at t and t+1 (Kuhn subdivision)" << std::endl;

    return (unique_tracks.size() == 1) ? 0 : 1;
}
