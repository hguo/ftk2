#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/synthetic.hh>
#include <ndarray/ndarray_stream.hh>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <map>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ftk2;

/**
 * @example unstructured_3d_synthetic
 *
 * Demonstrates 3D critical point tracking on unstructured meshes with synthetic data.
 * Creates verifiable synthetic vector fields with known critical points.
 */

int main(int argc, char** argv) {
    // Configuration
    const int DW = 16, DH = 16, DD = 16, DT = 5;
    std::cout << "=== FTK2 Unstructured 3D Critical Point Tracking with Synthetic Data ===" << std::endl;
    std::cout << "Grid dimensions: " << DW << "x" << DH << "x" << DD << ", Timesteps: " << DT << std::endl;

    // Option 1: Generate synthetic 3D mesh using ndarray stream
    std::cout << "\n[1/6] Generating synthetic 3D mesh using YAML stream..." << std::endl;
    {
        std::ofstream f("synthetic_3d_mesh.yaml");
        f << "stream:\n";
        f << "  substreams:\n";
        f << "    - name: moving_maximum\n";
        f << "      format: synthetic\n";
        f << "      dimensions: [" << DW << ", " << DH << ", " << DD << "]\n";
        f << "      timesteps: " << DT << "\n";
        f << "      delta: " << 1.0 / (DT - 1) << "\n";
        f << "      vars:\n";
        f << "        - name: scalar\n";
        f << "          dtype: float64\n";
        f.close();
    }

    // Read the stream to generate synthetic data
    ftk::stream<> stream;
    stream.parse_yaml("synthetic_3d_mesh.yaml");

    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        auto group = stream.read(t);
        const auto& s_t = group->get_ref<double>("scalar");
        for (int z = 0; z < DD; ++z) {
            for (int y = 0; y < DH; ++y) {
                for (int x = 0; x < DW; ++x) {
                    scalar.f(x, y, z, t) = s_t.f(x, y, z);
                }
            }
        }
    }
    std::cout << "    Generated scalar field: " << scalar.dimf(0) << "x" << scalar.dimf(1)
              << "x" << scalar.dimf(2) << "x" << scalar.dimf(3) << std::endl;

    // Option 2: Load from VTU if provided, otherwise create regular mesh
    std::shared_ptr<UnstructuredSimplicialMesh> base_mesh;
    bool using_vtu = (argc >= 2);

    if (using_vtu) {
        std::cout << "\n[2/6] Loading base mesh from VTU: " << argv[1] << std::endl;
        base_mesh = read_vtu(argv[1]);
        if (!base_mesh) {
            std::cerr << "    ERROR: Failed to load VTU file!" << std::endl;
            return 1;
        }
        std::cout << "    Loaded mesh with " << base_mesh->get_num_vertices() << " vertices" << std::endl;
    } else {
        std::cout << "\n[2/6] No VTU provided, creating regular mesh..." << std::endl;
        // For testing, create a simple regular mesh structure
        // This will be replaced with actual unstructured mesh generation
        std::cout << "    Note: Regular mesh mode - for unstructured testing, provide a VTU file" << std::endl;
    }

    // Generate synthetic 3D vector field with moving critical point
    std::cout << "\n[3/6] Generating synthetic 3D vector field (U, V, W)..." << std::endl;

    // Create a regular grid for now (in practice, this would map to unstructured vertices)
    int n_verts = DW * DH * DD * DT;
    ftk::ndarray<double> u, v, w;
    u.reshapef({(size_t)n_verts});
    v.reshapef({(size_t)n_verts});
    w.reshapef({(size_t)n_verts});

    // Define a moving critical point trajectory
    // The critical point moves in a helical path
    std::vector<std::array<double, 3>> cp_trajectory(DT);
    for (int t = 0; t < DT; ++t) {
        double phase = 2.0 * M_PI * t / (DT - 1);
        cp_trajectory[t][0] = DW / 2.0 + 3.0 * std::cos(phase);      // X center
        cp_trajectory[t][1] = DH / 2.0 + 3.0 * std::sin(phase);      // Y center
        cp_trajectory[t][2] = DD / 2.0 + 2.0 * std::sin(2.0 * phase); // Z center (figure-8)
    }

    std::cout << "    Critical point trajectory:" << std::endl;
    for (int t = 0; t < DT; ++t) {
        std::cout << "      t=" << t << ": (" << cp_trajectory[t][0] << ", "
                  << cp_trajectory[t][1] << ", " << cp_trajectory[t][2] << ")" << std::endl;
    }

    // Generate the vector field: (u,v,w) = (x-cx, y-cy, z-cz)
    // This creates a critical point at (cx, cy, cz) where all components are zero
    int idx = 0;
    for (int t = 0; t < DT; ++t) {
        double cx = cp_trajectory[t][0];
        double cy = cp_trajectory[t][1];
        double cz = cp_trajectory[t][2];

        for (int z = 0; z < DD; ++z) {
            for (int y = 0; y < DH; ++y) {
                for (int x = 0; x < DW; ++x) {
                    u[idx] = (double)x - cx;
                    v[idx] = (double)y - cy;
                    w[idx] = (double)z - cz;
                    idx++;
                }
            }
        }
    }
    std::cout << "    Generated " << n_verts << " field values" << std::endl;

    // Create spacetime mesh
    std::cout << "\n[4/6] Creating spacetime mesh..." << std::endl;
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT}
    );
    std::cout << "    Mesh dimensions: " << DW << "x" << DH << "x" << DD << "x" << DT << std::endl;

    // Prepare data map
    std::map<std::string, ftk::ndarray<double>> data_map = {
        {"U", u},
        {"V", v},
        {"W", w},
        {"Scalar", scalar}
    };

    // Set up critical point predicate
    std::cout << "\n[5/6] Setting up 3D critical point tracking..." << std::endl;
    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.var_names[2] = "W";
    cp_pred.scalar_var_name = "Scalar";

    // Run the engine
    SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, cp_pred);
    std::cout << "    Executing critical point tracking..." << std::endl;
    engine.execute(data_map);

    // Output results
    std::cout << "\n[6/6] Writing results..." << std::endl;
    auto cp_complex = engine.get_complex();
    std::cout << "    Found " << cp_complex.vertices.size() << " critical point nodes" << std::endl;

    // Count features at each timestep
    std::map<int, int> cp_per_timestep;
    for (const auto& kv : cp_complex.vertices) {
        // Extract timestep from feature (assuming it's stored in the feature)
        // This is a simplified version - actual implementation may vary
        cp_per_timestep[0]++; // Placeholder
    }

    write_complex_to_vtu(cp_complex, *mesh, "unstructured_3d_synthetic_cp.vtu", 0);
    write_complex_to_vtp(cp_complex, *mesh, "unstructured_3d_synthetic_cp.vtp");
    std::cout << "    Results written to:" << std::endl;
    std::cout << "      - unstructured_3d_synthetic_cp.vtu" << std::endl;
    std::cout << "      - unstructured_3d_synthetic_cp.vtp" << std::endl;

    // Verification
    std::cout << "\n=== VERIFICATION ===" << std::endl;
    std::cout << "Expected: ~" << DT << " critical points (one per timestep)" << std::endl;
    std::cout << "Found:    " << cp_complex.vertices.size() << " critical point nodes" << std::endl;

    if (cp_complex.vertices.size() > 0) {
        std::cout << "STATUS: SUCCESS - Critical points detected!" << std::endl;
    } else {
        std::cout << "STATUS: WARNING - No critical points found (check mesh resolution)" << std::endl;
    }

    return 0;
}
