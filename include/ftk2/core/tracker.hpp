#pragma once

#include <ftk2/core/extractor.hpp>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>

namespace ftk2 {

/**
 * @brief Base class for temporal tracking of features.
 * 
 * @tparam FeatureType The type of features to be tracked (e.g., CriticalPoint).
 * @tparam TrackType The type of tracks to be produced (e.g., CriticalPointTrajectory).
 */
template <typename FeatureType, typename TrackType>
class Tracker {
public:
    virtual ~Tracker() = default;

    /**
     * @brief Process a new set of features for the current time step.
     * 
     * @param features Extracted features from a single time step.
     * @param time_step The current time step index.
     */
    virtual void push_features(const std::vector<FeatureType>& features, int time_step) = 0;

    /**
     * @brief Finalize tracking after all features have been pushed.
     */
    virtual void finalize() = 0;

    /**
     * @brief Get the tracked features.
     * 
     * @return A vector of tracks.
     */
    virtual std::vector<TrackType> get_tracks() const = 0;
};

/**
 * @brief Factory for creating trackers.
 */
template <typename FeatureType, typename TrackType>
class TrackerFactory {
public:
    using Creator = std::function<std::unique_ptr<Tracker<FeatureType, TrackType>>()>;

    static void register_tracker(const std::string& name, Creator creator) {
        get_registry()[name] = creator;
    }

    static std::unique_ptr<Tracker<FeatureType, TrackType>> create(const std::string& name) {
        auto it = get_registry().find(name);
        if (it != get_registry().end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    static std::map<std::string, Creator>& get_registry() {
        static std::map<std::string, Creator> registry;
        return registry;
    }
};

} // namespace ftk2
