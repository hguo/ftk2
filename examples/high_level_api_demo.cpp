/**
 * @file high_level_api_demo.cpp
 * @brief Demonstration of FTK2 High-Level API (Layer 2)
 *
 * This example shows how to use the configuration-driven high-level API.
 * For now, it demonstrates config loading and validation. Full execution
 * requires data source implementation (Phase 2).
 */

#include <ftk2/high_level/feature_tracker.hpp>
#include <iostream>
#include <exception>

int main(int argc, char** argv) {
    try {
        std::string config_path = "high_level_example_config.yaml";

        if (argc >= 2) {
            config_path = argv[1];
        }

        std::cout << "=== FTK2 High-Level API Demo ===" << std::endl;
        std::cout << "Loading configuration from: " << config_path << std::endl;
        std::cout << std::endl;

        // Load and validate configuration
        auto config = ftk2::TrackingConfig::from_yaml(config_path);

        std::cout << "Configuration loaded successfully!" << std::endl;
        std::cout << std::endl;

        std::cout << "Configuration summary:" << std::endl;
        std::cout << "  Feature: " << ftk2::TrackingConfig::feature_type_to_string(config.feature) << std::endl;
        std::cout << "  Dimension: " << config.dimension << "D" << std::endl;
        std::cout << "  Input type: " << ftk2::TrackingConfig::input_type_to_string(config.input.type) << std::endl;
        std::cout << "  Field type: " << config.input.field_type << std::endl;
        std::cout << "  Data source: " << ftk2::TrackingConfig::data_source_type_to_string(config.data.source) << std::endl;
        std::cout << "  Mesh type: " << ftk2::TrackingConfig::mesh_type_to_string(config.mesh.type) << std::endl;

        if (!config.mesh.dimensions.empty()) {
            std::cout << "  Mesh dimensions: [";
            for (size_t i = 0; i < config.mesh.dimensions.size(); ++i) {
                std::cout << config.mesh.dimensions[i];
                if (i < config.mesh.dimensions.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "  Backend: " << ftk2::TrackingConfig::backend_to_string(config.execution.backend) << std::endl;
        std::cout << "  Precision: " << ftk2::TrackingConfig::precision_to_string(config.execution.precision) << std::endl;
        std::cout << "  Output: " << config.output.trajectories << std::endl;
        std::cout << std::endl;

        // Test round-trip: save to another YAML
        std::string output_config = "config_roundtrip.yaml";
        config.to_yaml(output_config);
        std::cout << "Configuration saved to: " << output_config << std::endl;
        std::cout << std::endl;

        // Create tracker (will dispatch based on precision)
        std::cout << "Creating feature tracker..." << std::endl;
        auto tracker = ftk2::FeatureTracker::create(config);
        std::cout << "Tracker created successfully!" << std::endl;
        std::cout << std::endl;

        // Note: Execution requires data source implementation
        std::cout << "=== Note ===" << std::endl;
        std::cout << "Full tracking execution requires:" << std::endl;
        std::cout << "  1. Data source implementation (stream, VTU, synthetic)" << std::endl;
        std::cout << "  2. Gradient computation (for scalar inputs)" << std::endl;
        std::cout << "  3. Output writers (VTP, HDF5)" << std::endl;
        std::cout << std::endl;
        std::cout << "See docs/USER_INTERFACE_LAYERS.md for implementation plan." << std::endl;
        std::cout << std::endl;

        // Uncomment when data sources are implemented:
        // auto results = tracker->execute();
        // std::cout << "Found " << results.num_trajectories() << " trajectories" << std::endl;

        std::cout << "=== Configuration Validation Complete ===" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
