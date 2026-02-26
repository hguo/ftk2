/**
 * @file high_level_tracking_test.cpp
 * @brief Full end-to-end test of high-level API with stream data
 */

#include <ftk2/high_level/feature_tracker.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return 1;
    }

    try {
        // Create tracker from YAML config
        auto tracker = ftk2::FeatureTracker::from_yaml(argv[1]);

        // Execute tracking
        auto results = tracker->execute();

        std::cout << "\n=== Tracking Complete ===\n";
        std::cout << "Found " << results.num_vertices() << " feature vertices\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
