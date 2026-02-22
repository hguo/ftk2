#pragma once

#include <ftk/ndarray.hh>
#include <ftk/ndarray/stream.hh>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace ftk2 {

/**
 * @brief Abstract base class for data sources.
 * 
 * Provides a stream of time steps, where each time step
 * consists of a set of named variables as ndarrays.
 */
template <typename T>
class Source {
public:
    virtual ~Source() = default;

    /**
     * @brief Set the configuration for the source.
     */
    virtual void configure(const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Advance to the next time step.
     */
    virtual bool next_timestep() = 0;

    /**
     * @brief Get the current time step index.
     */
    virtual int get_current_timestep() const = 0;

    /**
     * @brief Get the data for the current time step.
     */
    virtual std::map<std::string, ftk::ndarray<T>> get_current_data() const = 0;
};

/**
 * @brief A Source implementation that wraps ftk::ndarray_stream.
 */
template <typename T>
class NdarrayStreamSource : public Source<T> {
public:
    NdarrayStreamSource() {}

    void configure(const std::map<std::string, std::string>& config) override {
        if (config.count("input")) {
            stream_.set_input_source(config.at("input"));
        }
        // In ndarray, stream configuration is often handled via YAML or direct calls
        // to set_input_source, set_path_pattern, etc.
    }

    bool next_timestep() override {
        return stream_.advance();
    }

    int get_current_timestep() const override {
        return stream_.get_current_timestep();
    }

    std::map<std::string, ftk::ndarray<T>> get_current_data() const override {
        std::map<std::string, ftk::ndarray<T>> data;
        for (const auto& var : stream_.get_variable_names()) {
            data[var] = stream_.get(var);
        }
        return data;
    }

    /**
     * @brief Access the underlying ftk::ndarray_stream directly if needed.
     */
    ftk::ndarray_stream<T>& get_stream() { return stream_; }

private:
    ftk::ndarray_stream<T> stream_;
};

} // namespace ftk2

} // namespace ftk2
