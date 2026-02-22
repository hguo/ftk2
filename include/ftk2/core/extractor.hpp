#pragma once

#include <ftk/ndarray.hh>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace ftk2 {

/**
 * @brief Base class for feature extraction from a single time step.
 * 
 * @tparam T The data type (e.g., float, double).
 * @tparam FeatureType The type of features to be extracted (e.g., CriticalPoint).
 */
template <typename T, typename FeatureType>
class Extractor {
public:
    virtual ~Extractor() = default;

    /**
     * @brief Set the input data for the current time step.
     * 
     * @param data A map of variable names to ndarray objects.
     */
    virtual void set_input(const std::map<std::string, ftk::ndarray<T>>& data) = 0;

    /**
     * @brief Perform the feature extraction.
     */
    virtual void execute() = 0;

    /**
     * @brief Get the extracted features.
     * 
     * @return A vector of extracted features.
     */
    virtual std::vector<FeatureType> get_output() const = 0;
};

/**
 * @brief Factory for creating extractors.
 */
template <typename T, typename FeatureType>
class ExtractorFactory {
public:
    using Creator = std::function<std::unique_ptr<Extractor<T, FeatureType>>()>;

    static void register_extractor(const std::string& name, Creator creator) {
        get_registry()[name] = creator;
    }

    static std::unique_ptr<Extractor<T, FeatureType>> create(const std::string& name) {
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
