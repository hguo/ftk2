#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace ftk2 {

// Forward declarations
class TrackingConfig;

/**
 * @brief Feature types supported by FTK2
 */
enum class FeatureType {
    CriticalPoints,
    Levelsets,
    Fibers,
    // Future features (see FEATURE_GAP_ANALYSIS.md):
    // ParallelVectors,
    // CriticalLines,
    // Particles,
    // TDGLVortex
};

/**
 * @brief Input data types
 */
enum class InputType {
    Scalar,           // Single scalar field
    Vector,           // Vector field (u, v, w)
    GradientVector,   // Pre-computed gradient
    MultiScalar,      // Multiple scalar fields (e.g., for fibers)
    Complex           // Complex field (re, im) for TDGL
};

/**
 * @brief Data source types
 */
enum class DataSourceType {
    Stream,      // ndarray YAML stream
    Arrays,      // Direct ndarray arrays (for testing)
    VTU,         // VTU file series
    Synthetic    // Synthetic data generator
};

/**
 * @brief Execution backend
 */
enum class Backend {
    CPU,
    CUDA,
    MPI
};

/**
 * @brief Numerical precision
 */
enum class Precision {
    Float,
    Double
};

/**
 * @brief Mesh types
 */
enum class MeshType {
    Regular,
    Unstructured,
    Extruded
};

/**
 * @brief Input configuration
 */
struct InputConfig {
    InputType type = InputType::Scalar;
    std::vector<std::string> variables;

    // Metadata: gradient vs non-gradient vector fields
    // "gradient" -> Hessian classification (min/max/saddle)
    // "flow" -> Jacobian classification (source/sink/spiral)
    std::string field_type = "gradient";

    // For GradientVector: optional scalar field for Hessian
    std::string scalar_field;

    // For MultiScalar: map of field names
    std::map<std::string, std::string> field_map;

    void validate() const {
        if (variables.empty()) {
            throw std::invalid_argument("Input variables cannot be empty");
        }

        if (field_type != "gradient" && field_type != "flow") {
            throw std::invalid_argument("field_type must be 'gradient' or 'flow'");
        }
    }
};

/**
 * @brief Data source configuration
 */
struct DataConfig {
    DataSourceType source = DataSourceType::Arrays;

    // For Stream source - can be external file or inline config
    std::string stream_yaml;        // Path to external stream YAML
    YAML::Node inline_stream;       // Inline stream configuration

    // For VTU source
    std::string vtu_pattern;  // e.g., "output_*.vtu"

    // For Arrays source (testing)
    // Actual arrays passed separately in execute()

    // For Synthetic source
    std::string generator;  // e.g., "moving_maximum", "tornado"
    std::map<std::string, double> generator_params;

    void validate() const {
        if (source == DataSourceType::Stream) {
            if (stream_yaml.empty() && !inline_stream) {
                throw std::invalid_argument("stream_yaml or inline stream config required for Stream source");
            }
        }
        if (source == DataSourceType::VTU && vtu_pattern.empty()) {
            throw std::invalid_argument("vtu_pattern required for VTU source");
        }
        if (source == DataSourceType::Synthetic && generator.empty()) {
            throw std::invalid_argument("generator required for Synthetic source");
        }
    }
};

/**
 * @brief Mesh configuration
 */
struct MeshConfig {
    MeshType type = MeshType::Regular;

    // For Regular mesh
    std::vector<uint64_t> dimensions;  // [nx, ny, nz]
    std::vector<double> spacing = {1.0, 1.0, 1.0};
    std::vector<double> origin = {0.0, 0.0, 0.0};

    // For Unstructured mesh
    std::string mesh_file;  // VTU file

    void validate() const {
        if (type == MeshType::Regular) {
            // Dimensions can be empty - will be auto-derived from data
            if (!dimensions.empty() && (dimensions.size() < 2 || dimensions.size() > 3)) {
                throw std::invalid_argument("dimensions must be 2D or 3D");
            }
        }
        if (type == MeshType::Unstructured && mesh_file.empty()) {
            throw std::invalid_argument("mesh_file required for Unstructured mesh");
        }
    }
};

/**
 * @brief Execution configuration
 */
struct ExecutionConfig {
    Backend backend = Backend::CPU;
    Precision precision = Precision::Double;
    int num_threads = -1;  // -1 = auto

    void validate() const {
        // Backend validation happens at runtime (check CUDA availability)
    }
};

/**
 * @brief Output configuration
 */
struct OutputConfig {
    std::string trajectories;  // VTP file path
    std::string statistics;    // JSON file path
    std::string format = "vtp";  // vtp, json, hdf5

    void validate() const {
        if (trajectories.empty()) {
            throw std::invalid_argument("trajectories output path required");
        }
    }
};

/**
 * @brief Feature-specific options
 */
struct FeatureOptions {
    // For Critical Lines
    std::string line_type = "ridge";  // ridge or valley

    // For Particles
    std::string integrator = "rk4";  // rk1, rk4
    int num_steps = 1000;
    double dt = 0.01;
    std::map<std::string, std::string> seeding;

    // For TDGL
    int min_winding = 1;

    // For Levelsets
    double threshold = 0.0;
};

/**
 * @brief Complete tracking configuration
 */
class TrackingConfig {
public:
    // Core settings
    FeatureType feature = FeatureType::CriticalPoints;
    int dimension = 3;  // Spatial dimension (2 or 3)

    // Sub-configurations
    InputConfig input;
    DataConfig data;
    MeshConfig mesh;
    ExecutionConfig execution;
    OutputConfig output;
    FeatureOptions options;

    /**
     * @brief Validate entire configuration
     */
    void validate() const {
        if (dimension < 2 || dimension > 3) {
            throw std::invalid_argument("dimension must be 2 or 3");
        }

        input.validate();
        data.validate();
        mesh.validate();
        execution.validate();
        output.validate();
    }

    /**
     * @brief Load configuration from YAML file
     */
    static TrackingConfig from_yaml(const std::string& path);

    /**
     * @brief Save configuration to YAML file
     */
    void to_yaml(const std::string& path) const;

    /**
     * @brief Load from YAML node (for nested configs)
     */
    static TrackingConfig from_yaml_node(const YAML::Node& node);

    /**
     * @brief Convert to YAML node
     */
    YAML::Node to_yaml_node() const;

    // Helper for enum conversions (public for use in implementation)
    static FeatureType parse_feature_type(const std::string& str);
    static std::string feature_type_to_string(FeatureType type);

    static InputType parse_input_type(const std::string& str);
    static std::string input_type_to_string(InputType type);

    static DataSourceType parse_data_source_type(const std::string& str);
    static std::string data_source_type_to_string(DataSourceType type);

    static Backend parse_backend(const std::string& str);
    static std::string backend_to_string(Backend backend);

    static Precision parse_precision(const std::string& str);
    static std::string precision_to_string(Precision precision);

    static MeshType parse_mesh_type(const std::string& str);
    static std::string mesh_type_to_string(MeshType type);
};

} // namespace ftk2
