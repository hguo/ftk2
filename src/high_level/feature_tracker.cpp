#include <ftk2/high_level/feature_tracker.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/utils/vtk.hpp>
#include <iostream>
#include <fstream>

namespace ftk2 {

// Forward declaration of data loading functions
template <typename T>
std::map<std::string, ftk::ndarray<T>> load_stream_data(const DataConfig& config);

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
        // Need to compute gradient
        // TODO: Implement gradient computation
        throw std::runtime_error("Gradient computation from scalar field not yet implemented. "
                                "For now, please provide vector field directly.");

    } else if (config_.input.type == InputType::Vector ||
               config_.input.type == InputType::GradientVector) {
        // Use data as-is
        return raw_data;

    } else if (config_.input.type == InputType::MultiScalar) {
        // TODO: Handle multiple scalar fields
        throw std::runtime_error("MultiScalar input not yet implemented");

    } else if (config_.input.type == InputType::Complex) {
        // TODO: Handle complex fields (TDGL)
        throw std::runtime_error("Complex input not yet implemented");
    }

    return raw_data;
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

    // Initialize predicate variable names for CriticalPointPredicate
    if constexpr (PredicateType::codimension == 2 || PredicateType::codimension == 3) {
        // This is a CriticalPointPredicate - set var_names
        if (config_.input.variables.size() >= PredicateType::codimension) {
            for (int i = 0; i < PredicateType::codimension; ++i) {
                predicate.var_names[i] = config_.input.variables[i];
            }
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
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Feature: " << TrackingConfig::feature_type_to_string(config_.feature) << std::endl;
    std::cout << "  Dimension: " << config_.dimension << "D" << std::endl;
    std::cout << "  Precision: " << (std::is_same<T, double>::value ? "double" : "float") << std::endl;
    std::cout << "  Backend: " << TrackingConfig::backend_to_string(config_.execution.backend) << std::endl;
    std::cout << std::endl;

    // 1. Create mesh
    std::cout << "[1/5] Creating mesh..." << std::endl;
    auto spatial_mesh = create_mesh();

    // 2. Load data
    std::cout << "[2/5] Loading data..." << std::endl;
    auto raw_data = create_data();

    // Auto-derive mesh dimensions from data if not specified
    if (config_.mesh.type == MeshType::Regular &&
        config_.mesh.dimensions.empty() &&
        !raw_data.empty()) {
        const auto& first_var = raw_data.begin()->second;
        std::cout << "  Auto-deriving mesh dimensions from data..." << std::endl;

        // Assume last dimension is time, others are spatial
        int nd = first_var.nd();
        std::vector<uint64_t> inferred_dims;

        // For static data: all dimensions are spatial
        // For temporal data: last dimension is time
        int spatial_dims = (nd >= 4) ? (nd - 1) : nd;

        for (int d = 0; d < spatial_dims; ++d) {
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
        fout << "\n";

        // Write feature elements
        fout << "# Feature Elements:\n";
        for (size_t i = 0; i < complex.vertices.size(); ++i) {
            const auto& elem = complex.vertices[i];
            fout << i << ": track_id=" << elem.track_id
                 << " type=" << elem.type
                 << " scalar=" << elem.scalar << "\n";
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
