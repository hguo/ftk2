#include <ftk2/high_level/feature_tracker.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/utils/vtk.hpp>
#include <iostream>
#include <fstream>

namespace ftk2 {

// Forward declaration of data loading functions
template <typename T>
std::map<std::string, ftk::ndarray<T>> load_stream_data(const DataConfig& config);

// Forward declaration of streaming loader
template <typename T>
class StreamingDataLoader;

// ============================================================================
// Factory Methods
// ============================================================================

std::unique_ptr<FeatureTracker> FeatureTracker::from_yaml(const std::string& yaml_path) {
    auto config = TrackingConfig::from_yaml(yaml_path);
    return create(config);
}

std::unique_ptr<FeatureTracker> FeatureTracker::create(const TrackingConfig& config) {
    // Dispatch based on precision
    if (config.execution.precision == Precision::Double) {
        return std::make_unique<FeatureTrackerImpl<double>>(config);
    } else {
        return std::make_unique<FeatureTrackerImpl<float>>(config);
    }
}

// ============================================================================
// FeatureTrackerImpl: Mesh Creation
// ============================================================================

template <typename T>
std::shared_ptr<Mesh> FeatureTrackerImpl<T>::create_mesh() {
    if (config_.mesh.type == MeshType::Regular) {
        // Regular simplicial mesh
        return std::make_shared<RegularSimplicialMesh>(config_.mesh.dimensions);

    } else if (config_.mesh.type == MeshType::Unstructured) {
        // Load from VTU file
        if (config_.mesh.mesh_file.empty()) {
            throw std::runtime_error("mesh_file required for unstructured mesh");
        }

        auto spatial_mesh = read_vtu(config_.mesh.mesh_file);
        if (!spatial_mesh) {
            throw std::runtime_error("Failed to load mesh from " + config_.mesh.mesh_file);
        }

        return spatial_mesh;

    } else if (config_.mesh.type == MeshType::Extruded) {
        // Not directly creatable from config (need timestep info from data)
        throw std::runtime_error("Extruded mesh requires timestep information from data");
    }

    throw std::runtime_error("Unknown mesh type");
}

// ============================================================================
// FeatureTrackerImpl: Data Loading
// ============================================================================

template <typename T>
std::map<std::string, ftk::ndarray<T>> FeatureTrackerImpl<T>::create_data() {
    std::map<std::string, ftk::ndarray<T>> data;

    if (config_.data.source == DataSourceType::Arrays) {
        // Data will be provided by user via execute_with_data() (future)
        throw std::runtime_error("Arrays data source requires execute_with_data() API. "
                                "Use 'source: stream' with synthetic data or actual files.");

    } else if (config_.data.source == DataSourceType::Stream) {
        // Load from ndarray stream (YAML config)
        data = load_stream_data<T>(config_.data);

    } else if (config_.data.source == DataSourceType::VTU) {
        // TODO: Load VTU time series
        throw std::runtime_error("VTU data source not yet implemented");

    } else if (config_.data.source == DataSourceType::Synthetic) {
        // TODO: Synthetic data generation
        throw std::runtime_error("Synthetic data source not yet implemented. "
                                "Use 'source: stream' with synthetic stream instead.");
    }

    return data;
}

// ============================================================================
// FeatureTrackerImpl: Preprocessing
// ============================================================================

template <typename T>
std::map<std::string, ftk::ndarray<T>> FeatureTrackerImpl<T>::preprocess_data(
    const std::map<std::string, ftk::ndarray<T>>& raw_data)
{
    if (config_.input.type == InputType::Scalar) {
        // Compute gradient from scalar field
        // The gradient field becomes the vector field for CriticalPointPredicate
        // TODO: Implement gradient computation using finite differences
        //       - For regular meshes: use central differences
        //       - For unstructured meshes: use mesh connectivity
        //       Output: multi-component array [dim, spatial..., time]
        throw std::runtime_error(
            "Gradient computation from scalar field not yet implemented.\n"
            "Note: CriticalPointPredicate only handles vector fields.\n"
            "Gradient derivation should happen here in preprocessing.");

    } else if (config_.input.type == InputType::Vector ||
               config_.input.type == InputType::GradientVector) {
        // Vector field provided directly - this is what CriticalPointPredicate expects
        // Data should be multi-component: [dim, spatial..., time]
        return raw_data;

    } else if (config_.input.type == InputType::MultiScalar) {
        // Multiple scalar fields (e.g., for fiber tracking)
        // TODO: Handle multiple scalar fields
        throw std::runtime_error("MultiScalar input not yet implemented");

    } else if (config_.input.type == InputType::Complex) {
        // Complex fields for TDGL vortex tracking
        // TODO: Handle complex fields (TDGL)
        throw std::runtime_error("Complex input not yet implemented");
    }

    return raw_data;
}

// ============================================================================
// FeatureTrackerImpl: Dimension Inference
// ============================================================================

template <typename T>
int FeatureTrackerImpl<T>::infer_dimension(const std::map<std::string, ftk::ndarray<T>>& data) {
    if (data.empty()) {
        throw std::runtime_error("Cannot infer dimension from empty data");
    }

    const auto& first_var = data.begin()->second;
    int nd = first_var.nd();

    // Analyze shape to determine spatial dimension
    // Formats:
    //   Scalar (old):     [nx, ny, nt] -> 2D  or  [nx, ny, nz, nt] -> 3D
    //   Scalar (multi):   [1, nx, ny, nt] -> 2D  or  [1, nx, ny, nz, nt] -> 3D
    //   Vector:           [2, nx, ny, nt] -> 2D  or  [3, nx, ny, nz, nt] -> 3D

    // Check if first dimension is component count (1, 2, or 3)
    bool has_components = (nd >= 3 && first_var.dimf(0) >= 1 && first_var.dimf(0) <= 3);

    int spatial_dims;
    if (has_components) {
        // Vector field: [ncomp, spatial..., time]
        spatial_dims = nd - 2;  // Subtract component and time dimensions
    } else {
        // Scalar field: [spatial..., time]
        spatial_dims = nd - 1;  // Subtract time dimension
    }

    if (spatial_dims < 2 || spatial_dims > 3) {
        throw std::runtime_error("Inferred spatial dimension must be 2D or 3D, got " +
                                std::to_string(spatial_dims) + "D");
    }

    return spatial_dims;
}

// ============================================================================
// FeatureTrackerImpl: Streaming Execution
// ============================================================================

template <typename T>
template <typename PredicateType>
TrackingResults FeatureTrackerImpl<T>::execute_streaming(
    std::shared_ptr<Mesh> spatial_mesh,
    const DataConfig& data_config)
{
    // This will be implemented in Phase 2
    // For now, throw not implemented
    throw std::runtime_error("Streaming execution not yet fully implemented. "
                            "Use non-streaming mode for now (loads all data into memory).");
}

// ============================================================================
// FeatureTrackerImpl: Execution
// ============================================================================

template <typename T>
template <typename PredicateType>
TrackingResults FeatureTrackerImpl<T>::execute_with_predicate(
    std::shared_ptr<Mesh> mesh,
    const std::map<std::string, ftk::ndarray<T>>& data)
{
    // Create predicate
    PredicateType predicate;

    // Initialize predicate for CriticalPointPredicate specifically
    if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<2, T>> ||
                  std::is_same_v<PredicateType, CriticalPointPredicate<3, T>>) {
        // CriticalPointPredicate - use multi-component arrays
        predicate.use_multicomponent = true;

        // Try to find a multi-component vector array
        // Look for arrays with first dimension matching codimension
        bool found_multicomp = false;
        for (const auto& kv : data) {
            const auto& arr = kv.second;
            if (arr.nd() >= 3 && arr.dimf(0) == PredicateType::codimension) {
                // Found multi-component array
                predicate.vector_var_name = kv.first;
                found_multicomp = true;
                std::cout << "  Using multi-component array: " << kv.first
                          << " [" << arr.dimf(0) << " components]" << std::endl;
                break;
            }
        }

        // Fallback to legacy mode if no multi-component array found
        if (!found_multicomp) {
            std::cout << "  Warning: No multi-component array found, using legacy mode" << std::endl;
            predicate.use_multicomponent = false;
            if (config_.input.variables.size() >= PredicateType::codimension) {
                for (int i = 0; i < PredicateType::codimension; ++i) {
                    predicate.var_names[i] = config_.input.variables[i];
                }
            }
        }
    }

    // Configure attributes to record (all predicate types)
    if (!config_.output.attributes.empty()) {
        std::cout << "  Configuring " << config_.output.attributes.size() << " attributes to record:" << std::endl;
        for (const auto& attr_cfg : config_.output.attributes) {
            // Validate that the source array exists in data
            if (data.find(attr_cfg.source) == data.end()) {
                std::cout << "    Warning: Attribute source '" << attr_cfg.source << "' not found in data, skipping" << std::endl;
                continue;
            }

            AttributeSpec spec;
            spec.name = attr_cfg.name;
            spec.source = attr_cfg.source;
            spec.type = attr_cfg.type;
            spec.component = attr_cfg.component;
            spec.slot = predicate.attributes.size();  // Auto-assign slot (0-15)

            if (spec.slot >= 16) {
                std::cout << "    Warning: Maximum 16 attributes supported, skipping '" << attr_cfg.name << "'" << std::endl;
                break;
            }

            predicate.attributes.push_back(spec);
            std::cout << "    [" << spec.slot << "] " << spec.name
                      << " <- " << spec.source << " (" << spec.type << ")" << std::endl;
        }
    }

    // Create engine
    SimplicialEngine<T, PredicateType> engine(mesh, predicate);

    // Execute
    std::cout << "Executing tracking with " << mesh->get_num_vertices()
              << " vertices..." << std::endl;

    engine.execute(data);

    std::cout << "Tracking complete. Building trajectories..." << std::endl;

    // Get results
    auto complex = engine.get_complex();

    std::cout << "Found " << complex.vertices.size()
              << " feature vertices" << std::endl;

    return TrackingResults(complex);
}

template <typename T>
TrackingResults FeatureTrackerImpl<T>::execute() {
    std::cout << "=== FTK2 Feature Tracker ===" << std::endl;

    // 1. Load data first (needed for auto-deriving mesh dimensions and feature dimension)
    std::cout << "[1/5] Loading data..." << std::endl;
    auto raw_data = create_data();

    // Auto-derive dimension if not specified
    if (config_.dimension == 0) {
        config_.dimension = infer_dimension(raw_data);
        std::cout << "  Auto-derived dimension: " << config_.dimension << "D" << std::endl;
    }

    // Now print configuration (after auto-derivation)
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Feature: " << TrackingConfig::feature_type_to_string(config_.feature) << std::endl;
    std::cout << "  Dimension: " << config_.dimension << "D" << std::endl;
    std::cout << "  Precision: " << (std::is_same<T, double>::value ? "double" : "float") << std::endl;
    std::cout << "  Backend: " << TrackingConfig::backend_to_string(config_.execution.backend) << std::endl;
    std::cout << std::endl;

    // 2. Create mesh (with auto-derived dimensions if needed)
    std::cout << "[2/5] Creating mesh..." << std::endl;
    std::shared_ptr<Mesh> spatial_mesh;

    // Auto-derive mesh dimensions from data if not specified
    if (config_.mesh.type == MeshType::Regular &&
        config_.mesh.dimensions.empty() &&
        !raw_data.empty()) {
        const auto& first_var = raw_data.begin()->second;
        std::cout << "  Auto-deriving mesh dimensions from data..." << std::endl;

        // Assume last dimension is time, first might be component
        int nd = first_var.nd();
        std::vector<uint64_t> inferred_dims;

        // Check if first dimension is component count (1, 2, or 3)
        int start_dim = 0;
        if (first_var.dimf(0) >= 1 && first_var.dimf(0) <= 3) {
            // Likely a component dimension
            start_dim = 1;
        }

        // Last dimension is time if there are enough dimensions
        int end_dim = nd;
        if (nd - start_dim >= 3) {
            // Has spatial + time
            end_dim = nd - 1;
        }

        // Extract spatial dimensions
        for (int d = start_dim; d < end_dim; ++d) {
            inferred_dims.push_back(first_var.dimf(d));
        }

        std::cout << "  Inferred dimensions: [";
        for (size_t i = 0; i < inferred_dims.size(); ++i) {
            std::cout << inferred_dims[i];
            if (i < inferred_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Create new spatial mesh with inferred dimensions
        spatial_mesh = std::make_shared<RegularSimplicialMesh>(inferred_dims);
    } else {
        // Dimensions provided in config or not a regular mesh
        spatial_mesh = create_mesh();
    }

    // Infer timesteps from data
    int num_timesteps = 1;
    if (!raw_data.empty()) {
        const auto& first_var = raw_data.begin()->second;
        if (first_var.nd() >= 4) {
            num_timesteps = first_var.dimf(first_var.nd() - 1);
        }
    }

    std::cout << "  Data has " << num_timesteps << " timestep(s)" << std::endl;

    // Always create spacetime mesh (even for static data with 1 timestep)
    // This simplifies the core - it always handles time-varying data
    std::shared_ptr<Mesh> mesh;
    if (num_timesteps > 1) {
        std::cout << "  Creating spacetime mesh (" << num_timesteps << " timesteps)..." << std::endl;
    } else {
        std::cout << "  Creating spacetime mesh (static data, 1 timestep)..." << std::endl;
    }
    mesh = std::make_shared<ExtrudedSimplicialMesh>(spatial_mesh, num_timesteps - 1);

    // 3. Preprocess data
    std::cout << "[3/5] Preprocessing data..." << std::endl;
    auto data = preprocess_data(raw_data);

    // 4. Execute tracking (dispatch by feature type)
    std::cout << "[4/5] Executing tracking..." << std::endl;

    TrackingResults results;

    if (config_.feature == FeatureType::CriticalPoints) {
        if (config_.dimension == 2) {
            results = execute_with_predicate<CriticalPointPredicate<2, T>>(mesh, data);
        } else if (config_.dimension == 3) {
            results = execute_with_predicate<CriticalPointPredicate<3, T>>(mesh, data);
        } else {
            throw std::runtime_error("Invalid dimension for critical points");
        }

    } else if (config_.feature == FeatureType::Levelsets) {
        results = execute_with_predicate<ContourPredicate<T>>(mesh, data);

    } else if (config_.feature == FeatureType::Fibers) {
        results = execute_with_predicate<FiberPredicate<T>>(mesh, data);

    } else {
        throw std::runtime_error("Unsupported feature type");
    }

    // 5. Write output
    std::cout << "[5/5] Writing output..." << std::endl;
    write_results(results);

    std::cout << std::endl;
    std::cout << "=== Tracking Complete ===" << std::endl;
    std::cout << "Results written to: " << config_.output.trajectories << std::endl;

    return results;
}

// ============================================================================
// FeatureTrackerImpl: Output Writing
// ============================================================================

template <typename T>
void FeatureTrackerImpl<T>::write_results(const TrackingResults& results) {
    const auto& complex = results.get_complex();

    // Write feature complex (text format for now)
    if (!config_.output.trajectories.empty()) {
        // TODO: Implement proper VTP writer
        // For now, just write a simple text format with feature elements
        std::ofstream fout(config_.output.trajectories);
        if (!fout) {
            throw std::runtime_error("Cannot open output file: " + config_.output.trajectories);
        }

        fout << "# FTK2 Feature Complex\n";
        fout << "# Feature: " << TrackingConfig::feature_type_to_string(config_.feature) << "\n";
        fout << "# Num vertices: " << complex.vertices.size() << "\n";

        // Write attribute header
        if (!config_.output.attributes.empty()) {
            fout << "# Attributes: ";
            for (size_t i = 0; i < config_.output.attributes.size(); ++i) {
                fout << config_.output.attributes[i].name;
                if (i < config_.output.attributes.size() - 1) fout << ", ";
            }
            fout << "\n";
        }
        fout << "\n";

        // Write feature elements
        fout << "# Feature Elements:\n";
        for (size_t i = 0; i < complex.vertices.size(); ++i) {
            const auto& elem = complex.vertices[i];
            fout << i << ": track_id=" << elem.track_id
                 << " type=" << elem.type
                 << " scalar=" << elem.scalar;

            // Write attributes if any were configured
            if (!config_.output.attributes.empty()) {
                fout << " attrs=[";
                for (size_t j = 0; j < config_.output.attributes.size(); ++j) {
                    if (j < 16) {  // Max 16 attributes
                        fout << elem.attributes[j];
                        if (j < config_.output.attributes.size() - 1) fout << ", ";
                    }
                }
                fout << "]";
            }
            fout << "\n";
        }

        std::cout << "  Wrote feature complex to: " << config_.output.trajectories << std::endl;
    }

    // Write statistics (JSON format)
    if (!config_.output.statistics.empty()) {
        std::ofstream fout(config_.output.statistics);
        if (!fout) {
            throw std::runtime_error("Cannot open statistics file: " + config_.output.statistics);
        }

        fout << "{\n";
        fout << "  \"feature\": \"" << TrackingConfig::feature_type_to_string(config_.feature) << "\",\n";
        fout << "  \"dimension\": " << config_.dimension << ",\n";
        fout << "  \"num_vertices\": " << complex.vertices.size() << ",\n";
        fout << "  \"precision\": \"" << (std::is_same<T, double>::value ? "double" : "float") << "\",\n";
        fout << "  \"backend\": \"" << TrackingConfig::backend_to_string(config_.execution.backend) << "\"\n";
        fout << "}\n";

        std::cout << "  Wrote statistics to: " << config_.output.statistics << std::endl;
    }
}

// Explicit template instantiations
template class FeatureTrackerImpl<float>;
template class FeatureTrackerImpl<double>;

} // namespace ftk2
