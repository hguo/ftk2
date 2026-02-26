#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_stream.hh>
#include <iostream>
#include <fstream>

using namespace ftk2;

int main() {
    std::cout << "=== FTK2 2D CP Tracking - Streaming CUDA ===" << std::endl;

    // Create spatial mesh (2D, without time)
    std::vector<uint64_t> spatial_dims = {32, 32};
    auto spatial_mesh = std::make_shared<RegularSimplicialMesh>(spatial_dims);

    std::cout << "Spatial mesh: " << spatial_dims[0] << "x" << spatial_dims[1] << std::endl;

    // Configure synthetic stream (moving ramps, 10 timesteps)
    std::cout << "Setting up synthetic data stream..." << std::endl;

    // Write stream config to temporary file
    std::string temp_yaml = "/tmp/ftk2_streaming_test.yaml";
    std::ofstream fout(temp_yaml);
    fout << "filetype: synthetic\n";
    fout << "name: moving_ramps\n";
    fout << "dimensions: [32, 32]\n";
    fout << "n_timesteps: 10\n";
    fout << "vars:\n";
    fout << "  - name: velocity\n";
    fout << "    type: float\n";
    fout << "    components: [u, v]\n";
    fout << "    synthetic:\n";
    fout << "      generator: moving_ramps\n";
    fout << "      ramp_velocity: 1.0\n";
    fout.close();

    // Create stream from YAML
    ftk::stream<ftk::native_storage> stream;
    stream.parse_yaml(temp_yaml);

    std::cout << "Stream configured: " << stream.total_timesteps() << " timesteps" << std::endl;
    std::cout << "Memory usage: 2 consecutive timesteps only" << std::endl;

    // Create predicate
    CriticalPointPredicate<2, double> predicate;
    predicate.use_multicomponent = true;
    predicate.vector_var_name = "velocity";

    // Create engine
    SimplicialEngine<double, CriticalPointPredicate<2>> engine(spatial_mesh, predicate);

    // Execute with streaming (GPU memory limited to 2 timesteps)
    std::cout << "\nExecuting streaming CUDA tracking..." << std::endl;
    engine.execute_cuda_streaming(stream, spatial_mesh);

    // Get results
    auto complex = engine.get_complex();
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Features found: " << complex.vertices.size() << std::endl;

    // Print sample features
    int count = 0;
    for (const auto& feature : complex.vertices) {
        std::cout << "  Feature " << count << ": track_id=" << feature.track_id << std::endl;
        if (++count >= 5) break;
    }

    std::cout << "\n=== Memory Efficiency ===" << std::endl;
    std::cout << "Traditional: ~" << (32*32*10*2*8) << " bytes (all timesteps)" << std::endl;
    std::cout << "Streaming:   ~" << (32*32*2*2*8) << " bytes (2 timesteps)" << std::endl;
    std::cout << "Reduction:   5x" << std::endl;

    return 0;
}
