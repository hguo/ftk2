#include <ftk2/high_level/feature_tracker.hpp>
#include <iostream>

using namespace ftk2;

/**
 * @brief High-level API example for Sujudi-Haimes vortex core tracking
 *
 * This demonstrates how simple ApproxPV tracking is with the high-level API:
 * - No manual cross product computation
 * - No manual fiber predicate configuration
 * - Automatic vorticity computation
 * - Built-in filtering
 */
int main(int argc, char* argv[]) {
    std::cout << "=== FTK2 Sujudi-Haimes High-Level API Example ===" << std::endl;

    // Method 1: Load configuration from YAML file
    if (argc > 1) {
        std::string yaml_path = argv[1];
        std::cout << "\nMethod 1: Loading from YAML: " << yaml_path << std::endl;

        auto tracker = FeatureTracker::from_yaml(yaml_path);
        auto results = tracker->execute();

        std::cout << "\nResults:" << std::endl;
        std::cout << "  Vortex cores detected: " << results.num_vertices() << std::endl;

        return 0;
    }

    // Method 2: Programmatic configuration
    std::cout << "\nMethod 2: Programmatic configuration" << std::endl;

    TrackingConfig config;

    // Feature: Sujudi-Haimes vortex cores (velocity × vorticity)
    config.feature = FeatureType::SujadiHaimes;
    config.dimension = 3;

    // Input: velocity field
    config.input.type = InputType::Vector;
    config.input.variables = {"u", "v", "w"};
    config.input.field_type = "flow";

    // Data: synthetic tornado (for demonstration)
    config.data.source = DataSourceType::Synthetic;
    config.data.generator = "tornado";
    config.data.generator_params["nx"] = 32;
    config.data.generator_params["ny"] = 32;
    config.data.generator_params["nz"] = 32;
    config.data.generator_params["nt"] = 10;

    // Options: automatic vorticity computation and filtering
    config.options.auto_compute_vorticity = true;
    config.options.w2_threshold = 0.1;
    config.options.filter_mode = "absolute";

    // Output: VTP format with attributes
    config.output.filename = "vortex_cores.vtp";

    AttributeConfig w2_attr;
    w2_attr.name = "w2";
    w2_attr.source = "w2";
    w2_attr.type = "scalar";
    config.output.attributes.push_back(w2_attr);

    AttributeConfig vel_mag;
    vel_mag.name = "velocity_magnitude";
    vel_mag.source = "velocity";
    vel_mag.type = "magnitude";
    config.output.attributes.push_back(vel_mag);

    AttributeConfig vort_mag;
    vort_mag.name = "vorticity_magnitude";
    vort_mag.source = "vorticity";
    vort_mag.type = "magnitude";
    config.output.attributes.push_back(vort_mag);

    // Execution: CPU backend, double precision
    config.execution.backend = Backend::CPU;
    config.execution.precision = Precision::Double;

    // Validate configuration
    try {
        config.validate();
    } catch (const std::exception& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    }

    // Create tracker and execute
    try {
        auto tracker = FeatureTracker::create(config);
        auto results = tracker->execute();

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Vortex cores detected: " << results.num_vertices() << std::endl;
        std::cout << "Output written to: " << config.output.filename << std::endl;

        // Analyze trajectory lengths
        const auto& complex = results.get_complex();
        std::map<uint64_t, int> track_lengths;
        for (const auto& v : complex.vertices) {
            track_lengths[v.track_id]++;
        }

        std::cout << "\nTrajectory statistics:" << std::endl;
        std::cout << "  Number of trajectories: " << track_lengths.size() << std::endl;

        int max_length = 0;
        int min_length = 1000000;
        for (const auto& [id, length] : track_lengths) {
            max_length = std::max(max_length, length);
            min_length = std::min(min_length, length);
        }

        std::cout << "  Trajectory length range: [" << min_length << ", " << max_length << "]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Execution error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n=== Complete ===" << std::endl;
    std::cout << "\nComparison with low-level API:" << std::endl;
    std::cout << "  High-level: ~15 lines of configuration" << std::endl;
    std::cout << "  Low-level: ~100 lines (manual cross product, predicate setup, filtering)" << std::endl;
    std::cout << "  Speedup: 6-7× less code!" << std::endl;

    return 0;
}
