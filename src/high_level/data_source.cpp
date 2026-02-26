#include <ftk2/high_level/tracking_config.hpp>
#include <ndarray/ndarray.hh>

#if NDARRAY_HAVE_YAML
#include <ndarray/ndarray_stream.hh>
#endif

#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <unistd.h>

namespace ftk2 {

/**
 * @brief Streaming data loader - yields consecutive timestep pairs
 * Only holds 2 timesteps in memory at once for memory efficiency
 */
template <typename T>
class StreamingDataLoader {
public:
    using stream_type = ftk::stream<ftk::native_storage>;

    StreamingDataLoader(stream_type& stream) : stream_(stream), current_t_(0) {
        total_timesteps_ = stream_.total_timesteps();

        // Get variable names from first timestep
        if (total_timesteps_ > 0) {
            auto group = stream_.read(0);
            for (const auto& kv : *group) {
                var_names_.push_back(kv.first);
            }
        }
    }

    // Get next pair of consecutive timesteps
    bool next_pair(std::map<std::string, ftk::ndarray<T>>& data_t0,
                   std::map<std::string, ftk::ndarray<T>>& data_t1,
                   int& t0, int& t1) {
        if (current_t_ >= total_timesteps_ - 1) {
            return false;  // No more pairs
        }

        t0 = current_t_;
        t1 = current_t_ + 1;

        // Load t0 (use cached if available from previous iteration)
        if (current_t_ == 0 || cached_next_.empty()) {
            data_t0 = load_timestep(t0);
        } else {
            data_t0 = std::move(cached_next_);
        }

        // Load t1
        data_t1 = load_timestep(t1);

        // Cache t1 for next iteration (t1 becomes t0 in next pair)
        cached_next_ = data_t1;

        current_t_++;
        return true;
    }

    int total_timesteps() const { return total_timesteps_; }
    int current_timestep() const { return current_t_; }
    const std::vector<std::string>& var_names() const { return var_names_; }

private:
    std::map<std::string, ftk::ndarray<T>> load_timestep(int t) {
        auto group = stream_.read(t);
        std::map<std::string, ftk::ndarray<T>> data;

        // Get all variables from group
        for (const auto& name : var_names_) {
            if (group->has(name)) {
                const auto& arr = group->get_ref<T>(name);
                data[name] = arr;  // Copy timestep data
            }
        }

        return data;
    }

    stream_type& stream_;
    int current_t_;
    int total_timesteps_;
    std::vector<std::string> var_names_;
    std::map<std::string, ftk::ndarray<T>> cached_next_;
};

/**
 * @brief Load data from ndarray stream (YAML configuration)
 *
 * ndarray stream already handles all formats: NetCDF, HDF5, ADIOS2, VTK, synthetic.
 * No need for separate loaders - stream is the universal interface.
 *
 * NOTE: This loads ALL timesteps into memory - for large datasets, use streaming execution instead.
 */
template <typename T>
std::map<std::string, ftk::ndarray<T>> load_stream_data(const DataConfig& config) {
#if NDARRAY_HAVE_YAML
    std::map<std::string, ftk::ndarray<T>> data;

    // Create stream from ndarray library
    ftk::stream stream;

    // Parse stream configuration (inline or external file)
    if (config.inline_stream) {
        // Inline stream config - write to temporary file
        // The parse_yaml method expects a file with a "stream:" root node
        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "stream" << YAML::Value << config.inline_stream;
        emitter << YAML::EndMap;

        std::string temp_file = "/tmp/ftk2_stream_" + std::to_string(getpid()) + ".yaml";
        std::ofstream fout(temp_file);
        fout << emitter.c_str();
        fout.close();

        stream.parse_yaml(temp_file);

        // Clean up temp file
        std::remove(temp_file.c_str());
    } else if (!config.stream_yaml.empty()) {
        // External stream YAML file
        stream.parse_yaml(config.stream_yaml);
    } else {
        throw std::runtime_error("Stream source requires either inline_stream or stream_yaml");
    }

    // Get total number of timesteps
    int n_timesteps = stream.total_timesteps();

    if (n_timesteps == 0) {
        throw std::runtime_error("Stream has no timesteps");
    }

    std::cout << "Stream info:" << std::endl;
    std::cout << "  Timesteps: " << n_timesteps << std::endl;
    std::cout << "  Variables: ";

    // Load all timesteps into arrays
    // For now, we load all data into memory. Later, we can implement streaming execution.
    bool first = true;
    std::vector<std::string> var_names;

    for (int t = 0; t < n_timesteps; ++t) {
        auto group = stream.read(t);

        if (first) {
            // Get variable names from first group (ndarray_group is a std::map)
            for (const auto& kv : *group) {
                var_names.push_back(kv.first);
                std::cout << kv.first << " ";
            }
            std::cout << std::endl;
            first = false;
        }

        // Copy data from group to our data map
        for (const auto& name : var_names) {
            if (group->has(name)) {
                // Get zero-copy reference to array from group
                const auto& arr = group->get_ref<T>(name);

                if (t == 0) {
                    // First timestep: determine dimensions and allocate
                    std::vector<size_t> dims;
                    for (int d = 0; d < arr.nd(); ++d) {
                        dims.push_back(arr.dimf(d));
                    }

                    // Add time dimension
                    dims.push_back(n_timesteps);

                    // Allocate
                    data[name].reshapef(dims);
                }

                // Copy this timestep's data
                // TODO: Optimize with zero-copy or smart slicing
                size_t spatial_size = arr.size();
                const T* src = arr.data();
                T* dst = data[name].data() + t * spatial_size;

                std::copy(src, src + spatial_size, dst);
            }
        }
    }

    std::cout << "  Loaded " << data.size() << " variables" << std::endl;
    std::cout << "  Variable names: ";
    for (const auto& kv : data) {
        std::cout << kv.first << " (shape: ";
        const auto& arr = kv.second;
        for (int d = 0; d < arr.nd(); ++d) {
            std::cout << arr.dimf(d);
            if (d < arr.nd() - 1) std::cout << "x";
        }
        std::cout << ") ";
    }
    std::cout << std::endl;

    // Handle multi-component arrays: decompose into separate component arrays
    // Example: "velocity" with shape [3, nx, ny, nz, nt] -> u, v, w with shape [nx, ny, nz, nt]
    // Also ensure all data has a time dimension (add singleton if needed)
    std::map<std::string, ftk::ndarray<T>> decomposed_data;
    for (const auto& kv : data) {
        const auto& name = kv.first;
        auto arr = kv.second;  // Non-const copy for potential modification

        // First, ensure data has time dimension
        // For 2D/3D data without time: add singleton time dimension
        bool needs_time_dim = false;
        if (arr.nd() == 2 || arr.nd() == 3) {
            // Spatial-only data (2D or 3D), no time dimension
            needs_time_dim = true;
        } else if (arr.nd() == 3 && arr.dimf(0) == 2) {
            // 2D vector field without time: [2, nx, ny]
            needs_time_dim = true;
        } else if (arr.nd() == 4 && arr.dimf(0) == 3) {
            // 3D vector field without time: [3, nx, ny, nz]
            needs_time_dim = true;
        }

        if (needs_time_dim) {
            std::cout << "  Adding singleton time dimension to " << name << std::endl;
            std::vector<size_t> new_dims;
            for (int d = 0; d < arr.nd(); ++d) {
                new_dims.push_back(arr.dimf(d));
            }
            new_dims.push_back(1);  // Add time dimension

            ftk::ndarray<T> arr_with_time;
            arr_with_time.reshapef(new_dims);
            std::copy(arr.data(), arr.data() + arr.size(), arr_with_time.data());
            arr = std::move(arr_with_time);
        }

        // Now check if first dimension is 2 or 3 (vector components)
        if (arr.nd() >= 4 && (arr.dimf(0) == 2 || arr.dimf(0) == 3)) {
            size_t num_components = arr.dimf(0);
            std::vector<size_t> spatial_dims;
            for (int d = 1; d < arr.nd(); ++d) {
                spatial_dims.push_back(arr.dimf(d));
            }

            // Standard component names
            const char* component_names[] = {"u", "v", "w"};

            std::cout << "  Decomposing " << name << " into " << num_components << " components: ";

            // Create separate arrays for each component
            for (size_t c = 0; c < num_components; ++c) {
                ftk::ndarray<T> component_arr;
                component_arr.reshapef(spatial_dims);

                // Copy component data (assumes C-order: component is the outermost dimension)
                size_t spatial_size = component_arr.size();
                const T* src = arr.data() + c * spatial_size;
                T* dst = component_arr.data();
                std::copy(src, src + spatial_size, dst);

                std::string component_name = component_names[c];
                decomposed_data[component_name] = std::move(component_arr);
                std::cout << component_name << " ";
            }
            std::cout << std::endl;
        } else {
            // Not a multi-component array, keep as-is
            decomposed_data[name] = arr;
        }
    }

    return decomposed_data;

#else
    throw std::runtime_error("ndarray stream support requires NDARRAY_HAVE_YAML");
#endif
}

// Explicit template instantiations
template std::map<std::string, ftk::ndarray<float>> load_stream_data<float>(const DataConfig&);
template std::map<std::string, ftk::ndarray<double>> load_stream_data<double>(const DataConfig&);

} // namespace ftk2
