#include <ftk2/core/mesh.hpp>
#include <iostream>
#include <set>
#include <map>

using namespace ftk2;

/**
 * @example unstructured_3d_pentatope_debug
 *
 * Debug tool to verify ExtrudedSimplicialMesh creates proper pentatopes
 * connecting spatial tets at t and t+1.
 */

int main(int argc, char** argv) {
    const int DW = 3, DH = 3, DD = 3, DT = 2;  // Small mesh for manual verification

    std::cout << "=== PENTATOPE DEBUG TEST ===\n";
    std::cout << "Spatial: " << DW << "×" << DH << "×" << DD << ", Timesteps: " << DT << "\n\n";

    // Create spatial mesh
    auto spatial_mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD}
    );

    std::cout << "[1/4] Spatial mesh:\n";
    std::cout << "  Vertices: " << spatial_mesh->get_num_vertices() << "\n";
    std::cout << "  3-simplices (tets): ";
    int n_spatial_tets = 0;
    spatial_mesh->iterate_simplices(3, [&](const Simplex& s) { n_spatial_tets++; });
    std::cout << n_spatial_tets << "\n";

    // Extrude in time
    auto spacetime_mesh = std::make_shared<ExtrudedSimplicialMesh>(spatial_mesh, DT - 1);

    std::cout << "\n[2/4] Spacetime mesh:\n";
    std::cout << "  Total dimension: " << spacetime_mesh->get_total_dimension() << "D\n";
    std::cout << "  Total vertices: " << spacetime_mesh->get_num_vertices() << "\n";

    // Count spacetime tets and pentatopes
    int n_tets_t0 = 0, n_tets_t1 = 0, n_tets_spacetime = 0;
    spacetime_mesh->iterate_simplices(3, [&](const Simplex& tet) {
        n_tets_spacetime++;
        auto c0 = spacetime_mesh->get_vertex_coordinates(tet.vertices[0]);
        auto c1 = spacetime_mesh->get_vertex_coordinates(tet.vertices[1]);
        auto c2 = spacetime_mesh->get_vertex_coordinates(tet.vertices[2]);
        auto c3 = spacetime_mesh->get_vertex_coordinates(tet.vertices[3]);

        // Check timestep
        double t0 = c0[3], t1 = c1[3], t2 = c2[3], t3 = c3[3];
        double max_t = std::max({t0, t1, t2, t3});
        double min_t = std::min({t0, t1, t2, t3});

        if (max_t == min_t) {
            // Pure spatial tet at single timestep
            if (max_t == 0) n_tets_t0++;
            else if (max_t == 1) n_tets_t1++;
        }
    });

    int n_pentatopes = 0;
    std::map<std::string, int> pent_type_counts;

    spacetime_mesh->iterate_simplices(4, [&](const Simplex& pent) {
        n_pentatopes++;

        // Get timesteps of vertices
        std::vector<double> times(5);
        for (int i = 0; i < 5; ++i) {
            auto coords = spacetime_mesh->get_vertex_coordinates(pent.vertices[i]);
            times[i] = coords[3];
        }

        // Count vertices at each timestep
        int n_t0 = 0, n_t1 = 0;
        for (double t : times) {
            if (t == 0) n_t0++;
            else if (t == 1) n_t1++;
        }

        std::string type = "t0=" + std::to_string(n_t0) + ",t1=" + std::to_string(n_t1);
        pent_type_counts[type]++;
    });

    std::cout << "  3-simplices (tets): " << n_tets_spacetime << "\n";
    std::cout << "    - At t=0: " << n_tets_t0 << "\n";
    std::cout << "    - At t=1: " << n_tets_t1 << "\n";
    std::cout << "    - Spacetime (mixed t): " << (n_tets_spacetime - n_tets_t0 - n_tets_t1) << "\n";
    std::cout << "  4-simplices (pentatopes): " << n_pentatopes << "\n";

    std::cout << "\n[3/4] Pentatope types:\n";
    for (const auto& [type, count] : pent_type_counts) {
        std::cout << "  " << type << ": " << count << "\n";
    }

    // Expected: Each spatial tet creates 4 spacetime pentatopes
    int expected_pentatopes = 4 * n_spatial_tets;
    std::cout << "\n[4/4] Verification:\n";
    std::cout << "  Expected pentatopes: " << expected_pentatopes << " (4 × " << n_spatial_tets << " spatial tets)\n";
    std::cout << "  Actual pentatopes:   " << n_pentatopes << "\n";

    if (n_pentatopes == expected_pentatopes) {
        std::cout << "  ✅ PASS: Correct number of pentatopes\n";
    } else {
        std::cout << "  ❌ FAIL: Wrong number of pentatopes\n";
    }

    // Check that we have pentatopes connecting both timesteps
    bool has_connecting_pents = false;
    for (const auto& [type, count] : pent_type_counts) {
        if (type.find("t0=") != std::string::npos && type.find("t1=") != std::string::npos) {
            // Extract counts
            size_t pos0 = type.find("t0=") + 3;
            size_t pos1 = type.find(",t1=") + 4;
            int n_t0 = std::stoi(type.substr(pos0, type.find(",") - pos0));
            int n_t1 = std::stoi(type.substr(pos1));

            if (n_t0 > 0 && n_t1 > 0) {
                has_connecting_pents = true;
                break;
            }
        }
    }

    if (has_connecting_pents) {
        std::cout << "  ✅ PASS: Pentatopes connect t=0 and t=1\n";
    } else {
        std::cout << "  ❌ FAIL: No pentatopes connect timesteps!\n";
        std::cout << "  → This would explain disconnected trajectories\n";
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "KEY INSIGHT:\n";
    std::cout << "For trajectory tracking to work, we need pentatopes that contain\n";
    std::cout << "both vertices at t=0 and vertices at t=1. These spacetime pentatopes\n";
    std::cout << "enable manifold stitching to connect critical points across timesteps.\n";

    return 0;
}
