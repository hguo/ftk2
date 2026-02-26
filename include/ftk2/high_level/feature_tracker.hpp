#pragma once

#include <ftk2/high_level/tracking_config.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/complex.hpp>
#include <ndarray/ndarray.hh>
#include <memory>
#include <string>

namespace ftk2 {

/**
 * @brief Results from feature tracking
 *
 * For now, this is just a wrapper around FeatureComplex.
 * In the future, we may add trajectory extraction and analysis methods.
 */
class TrackingResults {
public:
    TrackingResults() = default;

    explicit TrackingResults(const FeatureComplex& complex)
        : complex_(complex) {}

    /**
     * @brief Get underlying feature complex
     */
    const FeatureComplex& get_complex() const {
        return complex_;
    }

    /**
     * @brief Number of feature elements (vertices) found
     */
    size_t num_vertices() const {
        return complex_.vertices.size();
    }

private:
    FeatureComplex complex_;
};

/**
 * @brief High-level feature tracker interface
 *
 * This is the main user-facing API for FTK2. It handles:
 * - Configuration parsing
 * - Type dispatch (precision, backend)
 * - Data preprocessing (gradient computation, etc.)
 * - Engine execution
 * - Output writing
 */
class FeatureTracker {
public:
    /**
     * @brief Create tracker from YAML configuration file
     */
    static std::unique_ptr<FeatureTracker> from_yaml(const std::string& yaml_path);

    /**
     * @brief Create tracker from configuration object
     */
    static std::unique_ptr<FeatureTracker> create(const TrackingConfig& config);

    virtual ~FeatureTracker() = default;

    /**
     * @brief Execute tracking
     */
    virtual TrackingResults execute() = 0;

    /**
     * @brief Get configuration
     */
    virtual const TrackingConfig& get_config() const = 0;

    /**
     * @brief Get output path
     */
    virtual std::string get_output_path() const = 0;

protected:
    FeatureTracker() = default;
};

/**
 * @brief Typed feature tracker implementation
 *
 * This is the concrete implementation that dispatches based on precision.
 */
template <typename T>
class FeatureTrackerImpl : public FeatureTracker {
public:
    explicit FeatureTrackerImpl(const TrackingConfig& config)
        : config_(config) {}

    TrackingResults execute() override;

    const TrackingConfig& get_config() const override {
        return config_;
    }

    std::string get_output_path() const override {
        return config_.output.trajectories;
    }

private:
    TrackingConfig config_;

    // Create mesh from config
    std::shared_ptr<Mesh> create_mesh();

    // Create data map (load/generate data)
    std::map<std::string, ftk::ndarray<T>> create_data();

    // Preprocessing: compute gradient if needed
    std::map<std::string, ftk::ndarray<T>> preprocess_data(
        const std::map<std::string, ftk::ndarray<T>>& raw_data);

    // Execute for specific feature type
    template <typename PredicateType>
    TrackingResults execute_with_predicate(
        std::shared_ptr<Mesh> mesh,
        const std::map<std::string, ftk::ndarray<T>>& data);

    // Streaming execution - process timestep pairs incrementally
    // Only holds 2 timesteps in memory at once
    template <typename PredicateType>
    TrackingResults execute_streaming(
        std::shared_ptr<Mesh> spatial_mesh,
        const DataConfig& data_config);

    // Auto-derive dimension from data shape
    int infer_dimension(const std::map<std::string, ftk::ndarray<T>>& data);

    // Write output files
    void write_results(const TrackingResults& results);
};

} // namespace ftk2
