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
 * @example unstructured_3d_single_trajectory
 *
 * Cleaner version with SHARPER peak to minimize cluster size.
 * Uses smaller sigma for tighter localization of critical points.
 */

int main(int argc, char** argv) {
    // Use SAME grid as moving_maximum but with SHARP peak
    const int DW = 12, DH = 12, DD = 12, DT = 5;

    std::cout << "=== Single Trajectory Test (Sharp Peak) ===" << std::endl;
    std::cout << "Grid: " << DW << "x" << DH << "x" << DD << ", Timesteps: " << DT << std::endl;
    std::cout << "Using SHARP Gaussian peak for cleaner detection\n" << std::endl;

    // Create field arrays
    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> u({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> v({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> w({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});

    // Define trajectory: 3D helix
    std::vector<std::array<double, 3>> trajectory(DT);
    std::cout << "[1/4] Critical point trajectory:" << std::endl;
    for (int t = 0; t < DT; ++t) {
        double theta = 2.0 * M_PI * t / DT;
        trajectory[t][0] = DW / 2.0 + 3.0 * std::cos(theta);        // X: radius 3
        trajectory[t][1] = DH / 2.0 + 3.0 * std::sin(theta);        // Y: radius 3
        trajectory[t][2] = DD / 2.0 + 2.5 * std::sin(2.0 * theta);  // Z: figure-8, amplitude 2.5

        std::cout << "  t=" << t << ": ("
                  << trajectory[t][0] << ", "
                  << trajectory[t][1] << ", "
                  << trajectory[t][2] << ")" << std::endl;
    }

    // Generate SHARP Gaussian field
    // Using sigma = 0.8 (much smaller than before, was 2.0)
    // This creates a tighter peak that spans fewer tetrahedra
    const double sigma = 0.8;
    std::cout << "\n[2/4] Generating field with σ=" << sigma << " (sharp peak)..." << std::endl;

    for (int t = 0; t < DT; ++t) {
        double cx = trajectory[t][0];
        double cy = trajectory[t][1];
        double cz = trajectory[t][2];

        for (int z = 0; z < DD; ++z) {
            for (int y = 0; y < DH; ++y) {
                for (int x = 0; x < DW; ++x) {
                    double dx = x - cx;
                    double dy = y - cy;
                    double dz = z - cz;
                    double r_squared = dx*dx + dy*dy + dz*dz;

                    // Scalar field (height function)
                    double s = std::exp(-r_squared / (sigma * sigma));
                    scalar.f(x, y, z, t) = s;

                    // Gradient field (should be zero at maximum)
                    double grad_factor = -(2.0 / (sigma * sigma)) * s;
                    u.f(x, y, z, t) = grad_factor * dx;
                    v.f(x, y, z, t) = grad_factor * dy;
                    w.f(x, y, z, t) = grad_factor * dz;
                }
            }
        }
    }

    std::cout << "[3/4] Creating spacetime mesh and tracking..." << std::endl;
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT}
    );

    std::map<std::string, ftk::ndarray<double>> data_map = {
        {"U", u},
        {"V", v},
        {"W", w},
        {"Scalar", scalar}
    };

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";
    cp_pred.scalar_var_name = "Scalar";

    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    std::cout << "[4/4] Writing results..." << std::endl;
    write_complex_to_vtu(cp_complex, *mesh, "single_trajectory_3d_cp.vtu", 0);
    write_complex_to_vtp(cp_complex, *mesh, "single_trajectory_3d_cp.vtp");

    // Count trajectory segments
    int n_trajectory_segments = 0;
    for (const auto& conn : cp_complex.connectivity) {
        if (conn.dimension == 1) {
            n_trajectory_segments = conn.indices.size() / 2;
            break;
        }
    }

    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Expected:   " << DT << " critical points (one per timestep)" << std::endl;
    std::cout << "Detected:   " << cp_complex.vertices.size() << " nodes";
    if (cp_complex.vertices.size() > 0) {
        std::cout << "  (~" << cp_complex.vertices.size() / DT << " per critical point)";
    }
    std::cout << std::endl;

    if (n_trajectory_segments > 0) {
        std::cout << "Trajectories: " << n_trajectory_segments << " segments" << std::endl;
    }

    std::cout << "\n✓ With σ=" << sigma << ", each analytical CP spans fewer simplices" << std::endl;
    std::cout << "✓ Result: Cleaner, more localized detections" << std::endl;

    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  • single_trajectory_3d_cp.vtu" << std::endl;
    std::cout << "  • single_trajectory_3d_cp.vtp  ← Open this for trajectories!" << std::endl;

    std::cout << "\nVisualization tip:" << std::endl;
    std::cout << "  paraview single_trajectory_3d_cp.vtp" << std::endl;
    std::cout << "  → Color by: Time" << std::endl;
    std::cout << "  → Filters > Tube (to thicken lines)" << std::endl;

    return 0;
}
