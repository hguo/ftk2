#include <ftk2/core/source.hpp>
#include <ftk2/core/extractor.hpp>
#include <ftk2/core/tracker.hpp>
#include <ftk2/core/feature.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <map>

namespace ftk2 {

/**
 * @brief A simple dummy extractor that returns a single feature at the center.
 */
template <typename T>
class DummyExtractor : public Extractor<T, Feature> {
public:
    void set_input(const std::map<std::string, ftk::ndarray<T>>& data) override {
        // Just store the data for now
        data_ = data;
    }

    void execute() override {
        // Extract a dummy feature
        Feature f;
        f.id = 0;
        f.timestep = 0; // Should be set by the tracker/pipeline
        f.position = {0.5, 0.5, 0.5}; // Center of a unit cube
        f.scalar_value = 1.0;
        features_ = {f};
    }

    std::vector<Feature> get_output() const override {
        return features_;
    }

private:
    std::map<std::string, ftk::ndarray<T>> data_;
    std::vector<Feature> features_;
};

/**
 * @brief A simple dummy tracker that collects all features.
 */
class DummyTracker : public Tracker<Feature, Track> {
public:
    void push_features(const std::vector<Feature>& features, int time_step) override {
        for (auto f : features) {
            f.timestep = time_step;
            all_features_.push_back(f);
        }
    }

    void finalize() override {
        // For simplicity, just create one track with all features
        Track t;
        t.id = 0;
        t.features = all_features_;
        tracks_ = {t};
    }

    std::vector<Track> get_tracks() const override {
        return tracks_;
    }

private:
    std::vector<Feature> all_features_;
    std::vector<Track> tracks_;
};

} // namespace ftk2

int main(int argc, char** argv) {
    // 1. Initialize components
    auto source = std::make_unique<ftk2::NdarrayStreamSource<float>>();
    auto extractor = std::make_unique<ftk2::DummyExtractor<float>>();
    auto tracker = std::make_unique<ftk2::DummyTracker>();

    // 2. Configure source (e.g., set data path)
    std::map<std::string, std::string> config = {{"path", "data.nc"}};
    source->configure(config);

    // 3. Process time steps
    while (source->next_timestep()) {
        int t = source->get_current_timestep();
        std::cout << "Processing time step " << t << "..." << std::endl;

        // Extract features
        extractor->set_input(source->get_current_data());
        extractor->execute();

        // Push features to tracker
        tracker->push_features(extractor->get_output(), t);
    }

    // 4. Finalize tracking and output results
    tracker->finalize();
    auto tracks = tracker->get_tracks();

    std::cout << "Extracted " << tracks.size() << " tracks." << std::endl;

    return 0;
}
