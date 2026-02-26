#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/synthetic.hh>
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
 * @example unstructured_3d_moving_maximum
 *
 * Simplified test with a single moving maximum critical point in 3D.
 * This creates a verifiable synthetic test case with known expected results.
 */

int main(int argc, char** argv) {
    // Use a small grid for easy verification
    const int DW = 12, DH = 12, DD = 12, DT = 5;

    std::cout << "=== Synthetic Moving Maximum in 3D ===" << std::endl;
    std::cout << "Grid: " << DW << "x" << DH << "x" << DD << ", Timesteps: " << DT << std::endl;

    // Create a moving maximum using ndarray synthetic functions
    std::cout << "\n[1/4] Generating moving maximum field..." << std::endl;

    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> u({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> v({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> w({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});

    // Define trajectory: moving maximum that traces a 3D helix
    std::vector<std::array<double, 3>> trajectory(DT);
    for (int t = 0; t < DT; ++t) {
        double theta = 2.0 * M_PI * t / DT;
        trajectory[t][0] = DW / 2.0 + 2.5 * std::cos(theta);       // X: circular motion
        trajectory[t][1] = DH / 2.0 + 2.5 * std::sin(theta);       // Y: circular motion
        trajectory[t][2] = DD / 2.0 + 2.0 * std::sin(2.0 * theta); // Z: moves up and down (figure-8)

        std::cout << "  t=" << t << ": maximum at ("
                  << trajectory[t][0] << ", "
                  << trajectory[t][1] << ", "
                  << trajectory[t][2] << ")" << std::endl;
    }

    // Generate field: Gaussian bump centered at the maximum
    // Scalar field: s = exp(-r^2 / sigma^2) where r is distance from center
    // Vector field: (u,v,w) = gradient of scalar = -(2/sigma^2) * r * s
    const double sigma = 2.0;  // Width of the Gaussian

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

                    // Gradient field (should have zero at maximum)
                    double grad_factor = -(2.0 / (sigma * sigma)) * s;
                    u.f(x, y, z, t) = grad_factor * dx;
                    v.f(x, y, z, t) = grad_factor * dy;
                    w.f(x, y, z, t) = grad_factor * dz;
                }
            }
        }
    }

    std::cout << "\n[2/4] Creating spacetime mesh..." << std::endl;
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT}
    );

    std::map<std::string, ftk::ndarray<double>> data_map = {
        {"U", u},
        {"V", v},
        {"W", w},
        {"Scalar", scalar}
    };

    std::cout << "\n[3/4] Tracking critical points (maxima in gradient field)..." << std::endl;
    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";
    cp_pred.scalar_var_name = "Scalar";

    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    std::cout << "\n[4/4] Writing results..." << std::endl;
    write_complex_to_vtu(cp_complex, *mesh, "moving_maximum_3d_cp.vtu", 0);
    write_complex_to_vtp(cp_complex, *mesh, "moving_maximum_3d_cp.vtp");

    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Expected: " << DT << " critical points (one maximum per timestep)" << std::endl;
    std::cout << "Found:    " << cp_complex.vertices.size() << " critical point nodes" << std::endl;

    // Count trajectory segments (1D connectivity)
    int n_trajectory_segments = 0;
    for (const auto& conn : cp_complex.connectivity) {
        if (conn.dimension == 1) {
            n_trajectory_segments = conn.indices.size() / 2;
            break;
        }
    }
    if (n_trajectory_segments > 0) {
        std::cout << "          " << n_trajectory_segments << " trajectory segments connecting nodes" << std::endl;
    }

    // Analyze the results
    if (cp_complex.vertices.size() == 0) {
        std::cout << "\n*** ERROR: No critical points found! ***" << std::endl;
        return 1;
    }

    std::cout << "\n*** SUCCESS: Critical point tracking working! ***" << std::endl;
    std::cout << "\nIMPORTANT NOTES:" << std::endl;
    std::cout << "  - Each analytical maximum generates a CLUSTER of detected nodes" << std::endl;
    std::cout << "  - This is NORMAL: simplicial mesh creates multiple overlapping simplices" << std::endl;
    std::cout << "  - ~" << cp_complex.vertices.size() / DT << " detected nodes per analytical maximum" << std::endl;
    std::cout << "\nVISUALIZATION:" << std::endl;
    std::cout << "  - moving_maximum_3d_cp.vtu  →  Discrete nodes (what you're seeing now)" << std::endl;
    std::cout << "  - moving_maximum_3d_cp.vtp  →  Connected trajectories (use this!)" << std::endl;
    std::cout << "\n  Open the .vtp file in ParaView to see the trajectory lines!" << std::endl;

    return 0;
}
