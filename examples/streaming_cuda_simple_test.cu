#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>

using namespace ftk2;

int main() {
    std::cout << "=== FTK2 CUDA Streaming Test (Manual Data) ===" << std::endl;

    // Dimensions
    const int nx = 16, ny = 16, nt = 5;

    // Create test data: simple moving gradient
    std::map<std::string, ftk::ndarray<double>> timestep_data[nt];

    for (int t = 0; t < nt; ++t) {
        ftk::ndarray<double> u, v;
        u.reshapef({nx, ny});
        v.reshapef({nx, ny});

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                // Moving ramp pattern
                double x = (double)i / nx;
                double y = (double)j / ny;
                double offset = (double)t / nt;

                u.f(i, j) = x - 0.5 + offset;
                v.f(i, j) = y - 0.5 + offset;
            }
        }

        timestep_data[t]["u"] = u;
        timestep_data[t]["v"] = v;
    }

    // Create spacetime mesh (includes time dimension)
    std::vector<uint64_t> spacetime_dims = {nx, ny, nt};
    auto mesh = std::make_shared<RegularSimplicialMesh>(spacetime_dims);

    // Combine all timesteps into single arrays with time dimension
    ftk::ndarray<double> u_all, v_all;
    u_all.reshapef({nx, ny, nt});
    v_all.reshapef({nx, ny, nt});

    for (int t = 0; t < nt; ++t) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                u_all.f(i, j, t) = timestep_data[t]["u"].f(i, j);
                v_all.f(i, j, t) = timestep_data[t]["v"].f(i, j);
            }
        }
    }

    std::map<std::string, ftk::ndarray<double>> data;
    data["u"] = u_all;
    data["v"] = v_all;

    // Create predicate (legacy mode for simplicity)
    CriticalPointPredicate<2, double> predicate;
    predicate.use_multicomponent = false;
    predicate.var_names[0] = "u";
    predicate.var_names[1] = "v";

    // Create engine
    SimplicialEngine<double, CriticalPointPredicate<2>> engine(mesh, predicate);

    // Execute with CUDA
    std::cout << "\nExecuting CUDA tracking..." << std::endl;
    std::cout << "Data shape: [" << nx << ", " << ny << ", " << nt << "]" << std::endl;
    std::cout << "Total memory: ~" << (nx * ny * nt * 2 * sizeof(double)) << " bytes" << std::endl;

    engine.execute_cuda(data);

    // Get results
    auto complex = engine.get_complex();
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Features found: " << complex.vertices.size() << std::endl;

    std::cout << "\nNote: Full CUDA streaming (2-timestep memory) requires ftk::stream integration" << std::endl;
    std::cout << "This example demonstrates the CUDA kernel - streaming mode coming soon!" << std::endl;

    return 0;
}
