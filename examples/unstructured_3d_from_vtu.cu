#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
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
 * @example unstructured_3d_from_vtu
 *
 * TRUE UNSTRUCTURED TEST: Loads a VTU mesh and assigns analytical values.
 * No regular grid - uses actual unstructured mesh geometry.
 */

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vtu_file> [n_timesteps]" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " mesh.vtu 5" << std::endl;
        return 1;
    }

    const std::string vtu_path = argv[1];
    const int DT = (argc >= 3) ? std::atoi(argv[2]) : 5;

    std::cout << "=== TRUE UNSTRUCTURED 3D TEST ===" << std::endl;
    std::cout << "Loading mesh from: " << vtu_path << std::endl;
    std::cout << "Timesteps: " << DT << std::endl;
    std::cout << "\nThis test loads an ACTUAL unstructured mesh (not a regular grid)" << std::endl;
    std::cout << "and assigns analytical vector field values to its vertices.\n" << std::endl;

    // Load the spatial mesh
    std::cout << "[1/5] Loading unstructured mesh..." << std::endl;
    auto base_mesh = read_vtu(vtu_path);
    if (!base_mesh) {
        std::cerr << "ERROR: Failed to load mesh from " << vtu_path << std::endl;
        return 1;
    }

    uint64_t n_verts = base_mesh->get_num_vertices();
    std::cout << "  ✓ Loaded mesh with " << n_verts << " vertices" << std::endl;

    // Get mesh bounds
    double min_x = 1e10, max_x = -1e10;
    double min_y = 1e10, max_y = -1e10;
    double min_z = 1e10, max_z = -1e10;

    for (uint64_t i = 0; i < n_verts; ++i) {
        auto coords = base_mesh->get_vertex_coordinates(i);
        min_x = std::min(min_x, coords[0]);
        max_x = std::max(max_x, coords[0]);
        min_y = std::min(min_y, coords[1]);
        max_y = std::max(max_y, coords[1]);
        min_z = std::min(min_z, coords[2]);
        max_z = std::max(max_z, coords[2]);
    }

    std::cout << "  Mesh bounds: " << std::endl;
    std::cout << "    X: [" << min_x << ", " << max_x << "]" << std::endl;
    std::cout << "    Y: [" << min_y << ", " << max_y << "]" << std::endl;
    std::cout << "    Z: [" << min_z << ", " << max_z << "]" << std::endl;

    double cx_center = (min_x + max_x) / 2.0;
    double cy_center = (min_y + max_y) / 2.0;
    double cz_center = (min_z + max_z) / 2.0;
    double radius = std::min({max_x - min_x, max_y - min_y, max_z - min_z}) * 0.3;

    // Define trajectory within mesh bounds
    std::cout << "\n[2/5] Defining trajectory (within mesh bounds)..." << std::endl;
    std::vector<std::array<double, 3>> trajectory(DT);
    for (int t = 0; t < DT; ++t) {
        double theta = 2.0 * M_PI * t / DT;
        trajectory[t][0] = cx_center + radius * std::cos(theta);
        trajectory[t][1] = cy_center + radius * std::sin(theta);
        trajectory[t][2] = cz_center + radius * 0.8 * std::sin(2.0 * theta);

        std::cout << "  t=" << t << ": ("
                  << trajectory[t][0] << ", "
                  << trajectory[t][1] << ", "
                  << trajectory[t][2] << ")" << std::endl;
    }

    // Extrude mesh in time
    std::cout << "\n[3/5] Creating spacetime mesh (extruding in time)..." << std::endl;
    auto spacetime_mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, DT - 1);

    uint64_t n_spacetime_verts = n_verts * DT;
    std::cout << "  ✓ Spacetime mesh has " << n_spacetime_verts << " vertices" << std::endl;

    // Assign analytical vector field values
    std::cout << "\n[4/5] Assigning analytical vector field values to vertices..." << std::endl;

    ftk::ndarray<double> u, v, w;
    u.reshapef({(size_t)n_spacetime_verts});
    v.reshapef({(size_t)n_spacetime_verts});
    w.reshapef({(size_t)n_spacetime_verts});

    for (uint64_t i = 0; i < n_spacetime_verts; ++i) {
        // Get vertex coordinates (includes time dimension)
        auto coords = spacetime_mesh->get_vertex_coordinates(i);
        double x = coords[0];
        double y = coords[1];
        double z = coords[2];
        double t = coords[3];  // Time is the last dimension

        // Find which timestep (discretize time)
        int t_idx = (int)(t + 0.5);  // Round to nearest integer
        if (t_idx < 0) t_idx = 0;
        if (t_idx >= DT) t_idx = DT - 1;

        double cx = trajectory[t_idx][0];
        double cy = trajectory[t_idx][1];
        double cz = trajectory[t_idx][2];

        // Analytical vector field: (u,v,w) = (x-cx, y-cy, z-cz)
        // Zero at exactly (cx, cy, cz)
        u[i] = x - cx;
        v[i] = y - cy;
        w[i] = z - cz;
    }

    std::cout << "  ✓ Assigned field values to " << n_spacetime_verts << " vertices" << std::endl;

    // Track critical points
    std::cout << "\n[5/5] Tracking critical points on unstructured spacetime mesh..." << std::endl;

    std::map<std::string, ftk::ndarray<double>> data_map = {
        {"U", u},
        {"V", v},
        {"W", w}
    };

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";

    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(spacetime_mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    // Write output
    write_complex_to_vtu(cp_complex, *spacetime_mesh, "unstructured_vtu_cp.vtu", 1);
    write_complex_to_vtp(cp_complex, *spacetime_mesh, "unstructured_vtu_cp.vtp");

    // Count trajectory segments
    int n_trajectory_segments = 0;
    for (const auto& conn : cp_complex.connectivity) {
        if (conn.dimension == 1) {
            n_trajectory_segments = conn.indices.size() / 2;
            break;
        }
    }

    // Results
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "                       RESULTS SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nMesh type:              TRUE UNSTRUCTURED (from VTU file)" << std::endl;
    std::cout << "Base mesh vertices:     " << n_verts << std::endl;
    std::cout << "Spacetime vertices:     " << n_spacetime_verts << std::endl;
    std::cout << "\nExpected CPs:           " << DT << " (one per timestep)" << std::endl;
    std::cout << "Detected nodes:         " << cp_complex.vertices.size() << std::endl;

    if (n_trajectory_segments > 0) {
        std::cout << "Trajectory segments:    " << n_trajectory_segments << std::endl;
    }

    if (cp_complex.vertices.size() > 0) {
        double ratio = (double)cp_complex.vertices.size() / DT;
        std::cout << "\nNodes per CP:           ~" << (int)ratio << std::endl;
    }

    std::cout << "\n" << std::string(70, '-') << std::endl;
    if (cp_complex.vertices.size() == 0) {
        std::cout << "STATUS: ❌ FAILED - No critical points detected" << std::endl;
    } else {
        std::cout << "STATUS: ✅ SUCCESS - Critical points detected on unstructured mesh!" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  • unstructured_vtu_cp.vtu  (nodes)" << std::endl;
    std::cout << "  • unstructured_vtu_cp.vtp  (trajectories)" << std::endl;

    std::cout << "\nKEY POINT:" << std::endl;
    std::cout << "  This test uses a TRUE unstructured mesh (not a regular grid)." << std::endl;
    std::cout << "  The mesh geometry comes from the input VTU file." << std::endl;
    std::cout << "  Analytical field values are assigned to the ACTUAL mesh vertices." << std::endl;

    return 0;
}
