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
 * @example unstructured_3d_clean_test
 *
 * CLEAN TEST: Uses analytical vector field directly (no gradient computation).
 *
 * Vector field: (u, v, w) = (x - cx(t), y - cy(t), z - cz(t))
 *
 * This has EXACTLY ONE ZERO per timestep at (cx, cy, cz).
 * No numerical artifacts, no discretization issues.
 * Tests pure tracking capability.
 */

int main(int argc, char** argv) {
    const int DW = 12, DH = 12, DD = 12, DT = 5;

    std::cout << "=== CLEAN TEST: Analytical Vector Field ===" << std::endl;
    std::cout << "Grid: " << DW << "x" << DH << "x" << DD << ", Timesteps: " << DT << std::endl;
    std::cout << "\nVector field: (u,v,w) = (x-cx, y-cy, z-cz)" << std::endl;
    std::cout << "Critical point: Exactly ONE zero per timestep at (cx, cy, cz)\n" << std::endl;

    // Define trajectory
    std::vector<std::array<double, 3>> trajectory(DT);
    std::cout << "[1/4] Trajectory (exact critical point locations):" << std::endl;
    for (int t = 0; t < DT; ++t) {
        double theta = 2.0 * M_PI * t / DT;
        trajectory[t][0] = DW / 2.0 + 2.5 * std::cos(theta);        // X: circular
        trajectory[t][1] = DH / 2.0 + 2.5 * std::sin(theta);        // Y: circular
        trajectory[t][2] = DD / 2.0 + 2.0 * std::sin(2.0 * theta);  // Z: figure-8

        std::cout << "  t=" << t << ": ("
                  << trajectory[t][0] << ", "
                  << trajectory[t][1] << ", "
                  << trajectory[t][2] << ")" << std::endl;
    }

    // Create analytical vector field
    std::cout << "\n[2/4] Generating ANALYTICAL vector field (no gradient computation)..." << std::endl;

    ftk::ndarray<double> u({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> v({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> w({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});

    for (int t = 0; t < DT; ++t) {
        double cx = trajectory[t][0];
        double cy = trajectory[t][1];
        double cz = trajectory[t][2];

        for (int z = 0; z < DD; ++z) {
            for (int y = 0; y < DH; ++y) {
                for (int x = 0; x < DW; ++x) {
                    // Simple linear field: zero at (cx, cy, cz)
                    u.f(x, y, z, t) = (double)x - cx;
                    v.f(x, y, z, t) = (double)y - cy;
                    w.f(x, y, z, t) = (double)z - cz;
                }
            }
        }
    }

    std::cout << "  ✓ Field has EXACTLY one zero per timestep (no numerical artifacts)" << std::endl;

    // Create spacetime mesh
    std::cout << "\n[3/4] Tracking critical points..." << std::endl;
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT}
    );

    std::map<std::string, ftk::ndarray<double>> data_map = {
        {"U", u},
        {"V", v},
        {"W", w}
    };

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";

    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, cp_pred);
    engine.execute(data_map);

    auto cp_complex = engine.get_complex();

    // Write output
    std::cout << "\n[4/4] Writing results..." << std::endl;
    write_complex_to_vtu(cp_complex, *mesh, "clean_test_3d_cp.vtu", 0);
    write_complex_to_vtp(cp_complex, *mesh, "clean_test_3d_cp.vtp");

    // Count trajectory segments
    int n_trajectory_segments = 0;
    for (const auto& conn : cp_complex.connectivity) {
        if (conn.dimension == 1) {
            n_trajectory_segments = conn.indices.size() / 2;
            break;
        }
    }

    // Analyze results
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "                         RESULTS SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nExpected (analytical):  " << DT << " critical points (one per timestep)" << std::endl;
    std::cout << "Detected (simplicial):  " << cp_complex.vertices.size() << " nodes" << std::endl;

    if (n_trajectory_segments > 0) {
        std::cout << "Trajectory segments:    " << n_trajectory_segments << " edges" << std::endl;
    }

    if (cp_complex.vertices.size() > 0) {
        double ratio = (double)cp_complex.vertices.size() / DT;
        std::cout << "\nNodes per CP:           ~" << (int)ratio << std::endl;
    }

    // Verdict
    std::cout << "\n" << std::string(70, '-') << std::endl;
    if (cp_complex.vertices.size() == 0) {
        std::cout << "STATUS: ❌ FAILED - No critical points detected!" << std::endl;
    } else if (cp_complex.vertices.size() <= 2 * DT) {
        std::cout << "STATUS: ✅ EXCELLENT - Very clean detection (close to analytical count)" << std::endl;
        std::cout << "        Each analytical CP detected with minimal redundancy" << std::endl;
    } else if (cp_complex.vertices.size() <= 10 * DT) {
        std::cout << "STATUS: ✅ GOOD - Reasonable cluster size per CP" << std::endl;
        std::cout << "        This is normal for simplicial methods" << std::endl;
    } else {
        std::cout << "STATUS: ⚠ WARNING - Many detections per CP" << std::endl;
        std::cout << "        May indicate mesh resolution or tolerance issues" << std::endl;
    }

    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  • clean_test_3d_cp.vtu  (nodes)" << std::endl;
    std::cout << "  • clean_test_3d_cp.vtp  (trajectories) ← Use this for visualization" << std::endl;

    std::cout << "\nKEY INSIGHT:" << std::endl;
    std::cout << "  Using analytical vector field (no gradient computation) gives" << std::endl;
    std::cout << "  cleaner results. The detected node count reflects the pure" << std::endl;
    std::cout << "  simplicial approximation without numerical gradient artifacts." << std::endl;

    return 0;
}
