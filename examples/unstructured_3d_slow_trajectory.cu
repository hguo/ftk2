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
 * @example unstructured_3d_slow_trajectory
 *
 * SLOW trajectory: moves only 0.1 grid units per timestep
 * Detections should definitely overlap!
 */

int main(int argc, char** argv) {
    const int DW = 12, DH = 12, DD = 12, DT = 5;

    std::cout << "=== SLOW TRAJECTORY TEST ===" << std::endl;
    std::cout << "Trajectory moves SLOWLY (0.1 units/timestep) to ensure overlap\n" << std::endl;

    // SLOW trajectory: just move linearly
    std::vector<std::array<double, 3>> trajectory(DT);
    std::cout << "[1/3] Trajectory (very slow motion):" << std::endl;
    for (int t = 0; t < DT; ++t) {
        trajectory[t][0] = 6.0 + t * 0.1;  // Move slowly in X
        trajectory[t][1] = 6.0 + t * 0.1;  // Move slowly in Y
        trajectory[t][2] = 6.0 + t * 0.1;  // Move slowly in Z
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

    std::cout << "[3/3] Assigning field and tracking..." << std::endl;
    for (uint64_t i = 0; i < n_vertices; ++i) {
        auto coords = spacetime_mesh->get_vertex_coordinates(i);
        double x = coords[0];
        double y = coords[1];
        double z = coords[2];
        double t = coords[3];

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

    std::map<std::string, ftk::ndarray<double>> data_map = {{"U", u}, {"V", v}, {"W", w}};

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";

    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(spacetime_mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    // Count unique tracks
    std::set<int> unique_tracks;
    for (const auto& vertex : cp_complex.vertices) {
        unique_tracks.insert(vertex.track_id);
    }

    write_complex_to_vtu(cp_complex, *spacetime_mesh, "slow_trajectory_cp.vtu", 1);
    write_complex_to_vtp(cp_complex, *spacetime_mesh, "slow_trajectory_cp.vtp");

    // Results
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Detected nodes:      " << cp_complex.vertices.size() << std::endl;
    std::cout << "Unique track IDs:    " << unique_tracks.size() << std::endl;

    if (unique_tracks.size() == 1) {
        std::cout << "\n✅ SUCCESS: ONE single trajectory!" << std::endl;
    } else {
        std::cout << "\n❌ FAILED: " << unique_tracks.size() << " disconnected trajectories" << std::endl;
        std::cout << "\nThis means even with SLOW motion (0.1 units/step)," << std::endl;
        std::cout << "detections are not connecting across timesteps." << std::endl;
        std::cout << "→ Bug in manifold stitching or Union-Find connectivity!" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    return (unique_tracks.size() == 1) ? 0 : 1;
}
