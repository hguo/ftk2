#include <ftk2/high_level/tracking_config.hpp>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace ftk2 {

// Helper: lowercase string
static std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return str;
}

// ============================================================================
// Enum Parsers
// ============================================================================

FeatureType TrackingConfig::parse_feature_type(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "critical_points" || lower == "cp") return FeatureType::CriticalPoints;
    if (lower == "levelsets" || lower == "contour" || lower == "iso") return FeatureType::Levelsets;
    if (lower == "fibers") return FeatureType::Fibers;
    throw std::invalid_argument("Unknown feature type: " + str);
}

std::string TrackingConfig::feature_type_to_string(FeatureType type) {
    switch (type) {
        case FeatureType::CriticalPoints: return "critical_points";
        case FeatureType::Levelsets: return "levelsets";
        case FeatureType::Fibers: return "fibers";
        default: return "unknown";
    }
}

InputType TrackingConfig::parse_input_type(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "scalar") return InputType::Scalar;
    if (lower == "vector") return InputType::Vector;
    if (lower == "gradient_vector") return InputType::GradientVector;
    if (lower == "multi_scalar") return InputType::MultiScalar;
    if (lower == "complex") return InputType::Complex;
    throw std::invalid_argument("Unknown input type: " + str);
}

std::string TrackingConfig::input_type_to_string(InputType type) {
    switch (type) {
        case InputType::Scalar: return "scalar";
        case InputType::Vector: return "vector";
        case InputType::GradientVector: return "gradient_vector";
        case InputType::MultiScalar: return "multi_scalar";
        case InputType::Complex: return "complex";
        default: return "unknown";
    }
}

DataSourceType TrackingConfig::parse_data_source_type(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "stream") return DataSourceType::Stream;
    if (lower == "arrays") return DataSourceType::Arrays;
    if (lower == "vtu") return DataSourceType::VTU;
    if (lower == "synthetic") return DataSourceType::Synthetic;
    throw std::invalid_argument("Unknown data source type: " + str);
}

std::string TrackingConfig::data_source_type_to_string(DataSourceType type) {
    switch (type) {
        case DataSourceType::Stream: return "stream";
        case DataSourceType::Arrays: return "arrays";
        case DataSourceType::VTU: return "vtu";
        case DataSourceType::Synthetic: return "synthetic";
        default: return "unknown";
    }
}

Backend TrackingConfig::parse_backend(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "cpu") return Backend::CPU;
    if (lower == "cuda" || lower == "gpu") return Backend::CUDA;
    if (lower == "mpi") return Backend::MPI;
    throw std::invalid_argument("Unknown backend: " + str);
}

std::string TrackingConfig::backend_to_string(Backend backend) {
    switch (backend) {
        case Backend::CPU: return "cpu";
        case Backend::CUDA: return "cuda";
        case Backend::MPI: return "mpi";
        default: return "unknown";
    }
}

Precision TrackingConfig::parse_precision(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "float" || lower == "single") return Precision::Float;
    if (lower == "double") return Precision::Double;
    throw std::invalid_argument("Unknown precision: " + str);
}

std::string TrackingConfig::precision_to_string(Precision precision) {
    switch (precision) {
        case Precision::Float: return "float";
        case Precision::Double: return "double";
        default: return "unknown";
    }
}

MeshType TrackingConfig::parse_mesh_type(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "regular") return MeshType::Regular;
    if (lower == "unstructured") return MeshType::Unstructured;
    if (lower == "extruded") return MeshType::Extruded;
    throw std::invalid_argument("Unknown mesh type: " + str);
}

std::string TrackingConfig::mesh_type_to_string(MeshType type) {
    switch (type) {
        case MeshType::Regular: return "regular";
        case MeshType::Unstructured: return "unstructured";
        case MeshType::Extruded: return "extruded";
        default: return "unknown";
    }
}

OutputType TrackingConfig::parse_output_type(const std::string& str) {
    std::string lower = to_lower(str);
    if (lower == "discrete") return OutputType::Discrete;
    if (lower == "traced") return OutputType::Traced;
    if (lower == "sliced") return OutputType::Sliced;
    if (lower == "intercepted") return OutputType::Intercepted;
    throw std::invalid_argument("Unknown output type: " + str);
}

std::string TrackingConfig::output_type_to_string(OutputType type) {
    switch (type) {
        case OutputType::Discrete: return "discrete";
        case OutputType::Traced: return "traced";
        case OutputType::Sliced: return "sliced";
        case OutputType::Intercepted: return "intercepted";
        default: return "unknown";
    }
}

// ============================================================================
// YAML Serialization
// ============================================================================

TrackingConfig TrackingConfig::from_yaml(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    if (!root["tracking"]) {
        throw std::invalid_argument("YAML file must contain 'tracking' key");
    }

    return from_yaml_node(root["tracking"]);
}

TrackingConfig TrackingConfig::from_yaml_node(const YAML::Node& node) {
    TrackingConfig config;

    // Feature type
    if (node["feature"]) {
        config.feature = parse_feature_type(node["feature"].as<std::string>());
    }

    // Dimension
    if (node["dimension"]) {
        config.dimension = node["dimension"].as<int>();
    }

    // Input configuration
    if (node["input"]) {
        auto input_node = node["input"];

        if (input_node["type"]) {
            config.input.type = parse_input_type(input_node["type"].as<std::string>());
        }

        if (input_node["variables"]) {
            config.input.variables = input_node["variables"].as<std::vector<std::string>>();
        }

        if (input_node["field_type"]) {
            config.input.field_type = input_node["field_type"].as<std::string>();
        }

        if (input_node["scalar_field"]) {
            config.input.scalar_field = input_node["scalar_field"].as<std::string>();
        }

        if (input_node["field_map"]) {
            config.input.field_map = input_node["field_map"].as<std::map<std::string, std::string>>();
        }
    }

    // Data configuration
    if (node["data"]) {
        auto data_node = node["data"];

        if (data_node["source"]) {
            config.data.source = parse_data_source_type(data_node["source"].as<std::string>());
        }

        // Stream config: can be external file path or inline
        if (data_node["stream_yaml"] || data_node["stream_config"]) {
            config.data.stream_yaml = data_node["stream_yaml"] ?
                data_node["stream_yaml"].as<std::string>() :
                data_node["stream_config"].as<std::string>();
        }

        // Inline stream configuration (user puts stream config in same YAML)
        if (data_node["stream"]) {
            config.data.inline_stream = data_node["stream"];
        }

        if (data_node["vtu_pattern"]) {
            config.data.vtu_pattern = data_node["vtu_pattern"].as<std::string>();
        }

        if (data_node["generator"]) {
            config.data.generator = data_node["generator"].as<std::string>();
        }

        if (data_node["generator_params"]) {
            config.data.generator_params = data_node["generator_params"].as<std::map<std::string, double>>();
        }
    }

    // Mesh configuration
    if (node["mesh"]) {
        auto mesh_node = node["mesh"];

        if (mesh_node["type"]) {
            config.mesh.type = parse_mesh_type(mesh_node["type"].as<std::string>());
        }

        if (mesh_node["dimensions"]) {
            config.mesh.dimensions = mesh_node["dimensions"].as<std::vector<uint64_t>>();
        }

        if (mesh_node["spacing"]) {
            config.mesh.spacing = mesh_node["spacing"].as<std::vector<double>>();
        }

        if (mesh_node["origin"]) {
            config.mesh.origin = mesh_node["origin"].as<std::vector<double>>();
        }

        if (mesh_node["mesh_file"]) {
            config.mesh.mesh_file = mesh_node["mesh_file"].as<std::string>();
        }
    }

    // Execution configuration
    if (node["execution"]) {
        auto exec_node = node["execution"];

        if (exec_node["backend"]) {
            config.execution.backend = parse_backend(exec_node["backend"].as<std::string>());
        }

        if (exec_node["precision"]) {
            config.execution.precision = parse_precision(exec_node["precision"].as<std::string>());
        }

        if (exec_node["threads"]) {
            config.execution.num_threads = exec_node["threads"].as<int>();
        }
    }

    // Output configuration
    if (node["output"]) {
        auto output_node = node["output"];

        if (output_node["filename"]) {
            config.output.filename = output_node["filename"].as<std::string>();
        }

        if (output_node["trajectories"]) {
            config.output.trajectories = output_node["trajectories"].as<std::string>();
        }

        if (output_node["statistics"]) {
            config.output.statistics = output_node["statistics"].as<std::string>();
        }

        if (output_node["format"]) {
            config.output.format = output_node["format"].as<std::string>();
        }

        if (output_node["type"]) {
            config.output.type = parse_output_type(output_node["type"].as<std::string>());
        }

        // Parse attributes to record
        if (output_node["attributes"]) {
            auto attrs_node = output_node["attributes"];
            if (attrs_node.IsSequence()) {
                for (const auto& attr_node : attrs_node) {
                    AttributeConfig attr;

                    if (attr_node.IsScalar()) {
                        // Simple form: just the source name
                        attr.name = attr_node.as<std::string>();
                        attr.source = attr.name;
                        attr.type = "scalar";
                    } else if (attr_node.IsMap()) {
                        // Full form with all fields
                        if (attr_node["name"]) {
                            attr.name = attr_node["name"].as<std::string>();
                        }
                        if (attr_node["source"]) {
                            attr.source = attr_node["source"].as<std::string>();
                        } else if (!attr.name.empty()) {
                            attr.source = attr.name;
                        }
                        if (attr_node["type"]) {
                            attr.type = attr_node["type"].as<std::string>();
                        }
                        if (attr_node["component"]) {
                            attr.component = attr_node["component"].as<int>();
                        }
                    }

                    config.output.attributes.push_back(attr);
                }
            }
        }
    }

    // Feature options
    if (node["options"]) {
        auto opts = node["options"];

        if (opts["line_type"]) {
            config.options.line_type = opts["line_type"].as<std::string>();
        }

        if (opts["integrator"]) {
            config.options.integrator = opts["integrator"].as<std::string>();
        }

        if (opts["num_steps"]) {
            config.options.num_steps = opts["num_steps"].as<int>();
        }

        if (opts["dt"]) {
            config.options.dt = opts["dt"].as<double>();
        }

        if (opts["min_winding"]) {
            config.options.min_winding = opts["min_winding"].as<int>();
        }

        if (opts["threshold"]) {
            config.options.threshold = opts["threshold"].as<double>();
        }

        if (opts["seeding"]) {
            config.options.seeding = opts["seeding"].as<std::map<std::string, std::string>>();
        }
    }

    // Validate
    config.validate();

    return config;
}

void TrackingConfig::to_yaml(const std::string& path) const {
    YAML::Node root;
    root["tracking"] = to_yaml_node();

    std::ofstream fout(path);
    if (!fout) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    fout << root;
}

YAML::Node TrackingConfig::to_yaml_node() const {
    YAML::Node node;

    // Core settings
    node["feature"] = feature_type_to_string(feature);
    node["dimension"] = dimension;

    // Input
    YAML::Node input_node;
    input_node["type"] = input_type_to_string(input.type);
    input_node["variables"] = input.variables;
    input_node["field_type"] = input.field_type;
    if (!input.scalar_field.empty()) {
        input_node["scalar_field"] = input.scalar_field;
    }
    if (!input.field_map.empty()) {
        input_node["field_map"] = input.field_map;
    }
    node["input"] = input_node;

    // Data
    YAML::Node data_node;
    data_node["source"] = data_source_type_to_string(data.source);
    if (!data.stream_yaml.empty()) {
        data_node["stream_config"] = data.stream_yaml;
    }
    if (data.inline_stream) {
        data_node["stream"] = data.inline_stream;
    }
    if (!data.vtu_pattern.empty()) {
        data_node["vtu_pattern"] = data.vtu_pattern;
    }
    if (!data.generator.empty()) {
        data_node["generator"] = data.generator;
    }
    if (!data.generator_params.empty()) {
        data_node["generator_params"] = data.generator_params;
    }
    node["data"] = data_node;

    // Mesh
    YAML::Node mesh_node;
    mesh_node["type"] = mesh_type_to_string(mesh.type);
    if (!mesh.dimensions.empty()) {
        mesh_node["dimensions"] = mesh.dimensions;
    }
    mesh_node["spacing"] = mesh.spacing;
    mesh_node["origin"] = mesh.origin;
    if (!mesh.mesh_file.empty()) {
        mesh_node["mesh_file"] = mesh.mesh_file;
    }
    node["mesh"] = mesh_node;

    // Execution
    YAML::Node exec_node;
    exec_node["backend"] = backend_to_string(execution.backend);
    exec_node["precision"] = precision_to_string(execution.precision);
    if (execution.num_threads != -1) {
        exec_node["threads"] = execution.num_threads;
    }
    node["execution"] = exec_node;

    // Output
    YAML::Node output_node;
    output_node["trajectories"] = output.trajectories;
    if (!output.statistics.empty()) {
        output_node["statistics"] = output.statistics;
    }
    output_node["format"] = output.format;
    node["output"] = output_node;

    return node;
}

} // namespace ftk2
