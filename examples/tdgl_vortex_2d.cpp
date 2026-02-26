#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>

using namespace ftk2;

// Generate synthetic TDGL data with vortices
void generate_tdgl_vortex_field(int nx, int ny, int nt,
                                ftk::ndarray<double>& re,
                                ftk::ndarray<double>& im)
{
    re.reshapef({(size_t)nx, (size_t)ny, (size_t)nt});
    im.reshapef({(size_t)nx, (size_t)ny, (size_t)nt});

    // Create a vortex at center (nx/2, ny/2) with winding number +1
    double vortex_x = nx / 2.0;
    double vortex_y = ny / 2.0;

    for (int t = 0; t < nt; ++t) {
        // Vortex moves slightly over time
        double offset = 0.2 * t;

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double x = i - vortex_x - offset;
                double y = j - vortex_y;

                // Distance from vortex core
                double r = std::sqrt(x * x + y * y);

                // Phase: atan2(y, x) gives winding number +1
                double theta = std::atan2(y, x);

                // Amplitude: rho = tanh(r) (drops to zero at core)
                double rho = std::tanh(r / 3.0);

                // Complex field: psi = rho * exp(i*theta)
                re.f(i, j, t) = rho * std::cos(theta);
                im.f(i, j, t) = rho * std::sin(theta);
            }
        }
    }
}

int main() {
    std::cout << "=== FTK2 TDGL Vortex Tracking ===" << std::endl;

    // Parameters
    const int nx = 32, ny = 32, nt = 10;

    std::cout << "Generating synthetic TDGL field with moving vortex..." << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << " spatial, " << nt << " timesteps" << std::endl;

    // Generate data
    ftk::ndarray<double> re, im;
    generate_tdgl_vortex_field(nx, ny, nt, re, im);

    std::map<std::string, ftk::ndarray<double>> data;
    data["re"] = re;
    data["im"] = im;

    std::cout << "  Complex field generated" << std::endl;

    // Create spacetime mesh
    std::vector<uint64_t> dims = {nx, ny, nt};
    auto mesh = std::make_shared<RegularSimplicialMesh>(dims);

    std::cout << "\nMesh: " << nx << "x" << ny << "x" << nt
              << " (" << mesh->get_num_vertices() << " vertices)" << std::endl;

    // Create TDGL vortex predicate
    TDGLVortexPredicate<double> predicate;
    predicate.re_name = "re";
    predicate.im_name = "im";
    predicate.min_winding = 1;  // Detect vortices with |winding| >= 1

    std::cout << "\nPredicate: TDGL vortex (min_winding=" << predicate.min_winding << ")" << std::endl;

    // Create engine and execute
    SimplicialEngine<double, TDGLVortexPredicate<double>> engine(mesh, predicate);

    std::cout << "\nExecuting vortex tracking..." << std::endl;
    engine.execute(data);

    // Get results
    auto complex = engine.get_complex();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Vortex detections: " << complex.vertices.size() << std::endl;

    // Analyze vortices
    std::map<int, int> winding_counts;
    for (const auto& v : complex.vertices) {
        winding_counts[v.type]++;
    }

    std::cout << "\nWinding number distribution:" << std::endl;
    for (const auto& [winding, count] : winding_counts) {
        std::cout << "  w=" << winding << ": " << count << " detections" << std::endl;
    }

    // Print sample vortex trajectories
    std::map<uint64_t, std::vector<int>> track_timesteps;
    for (const auto& v : complex.vertices) {
        auto coords = mesh->get_vertex_coordinates(v.simplex.vertices[0]);
        int t = (coords.size() > 2) ? (int)coords[2] : 0;
        track_timesteps[v.track_id].push_back(t);
    }

    std::cout << "\nVortex trajectories:" << std::endl;
    int count = 0;
    for (const auto& [track_id, timesteps] : track_timesteps) {
        std::cout << "  Track " << track_id << ": " << timesteps.size()
                  << " detections across " << (timesteps.back() - timesteps.front() + 1)
                  << " timesteps" << std::endl;
        if (++count >= 5) break;
    }

    // Write output
    std::cout << "\nWriting results to tdgl_vortex_2d.vtu..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "tdgl_vortex_2d.vtu");

    std::cout << "\n=== Complete ===" << std::endl;
    std::cout << "Expected: 1 vortex trajectory (winding +1) across " << nt << " timesteps" << std::endl;

    return 0;
}
