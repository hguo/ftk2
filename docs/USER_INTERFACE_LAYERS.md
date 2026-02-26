# FTK2 User Interface Layers: Comprehensive Plan

## Overview

FTK2 needs **5 distinct user interfaces** to serve different user communities and use cases:

1. **Low-Level API** (C++) - For library developers, maximum control
2. **High-Level API** (C++) - For C++ users, configuration-driven
3. **JSON API + Executable** - For command-line users, no coding
4. **Python High-Level API** - For data scientists, Jupyter notebooks
5. **VTK/ParaView Plugins** - For visualization users, interactive

Each layer builds on the previous ones, creating a **cohesive architecture**.

---

## Architecture: Layered Design

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: ParaView Plugin                                   │  ← Interactive GUI
│  (VTK filter + property panel)                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Python High-Level API                             │  ← Jupyter, scripts
│  (pybind11 wrapper of Layer 2)                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: JSON API + Executable                             │  ← Command-line
│  (Wraps Layer 2 with JSON/YAML config)                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: High-Level API (C++)                              │  ← Configuration
│  (TrackingConfig, Factory, Adapters)                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Low-Level API (C++)                               │  ← Library core
│  (SimplicialEngine, Predicate, Mesh)                        │
└─────────────────────────────────────────────────────────────┘
```

**Key principle**: Each layer is **thin** and delegates to lower layers. No duplication of logic.

---

## Layer 1: Low-Level API (C++)

### Purpose
- **Who**: Library developers, researchers extending FTK2
- **When**: Maximum control, custom predicates, research prototypes
- **Complexity**: High (requires understanding of simplicial topology)

### Current Status
✅ **IMPLEMENTED** - Core engine is working

### API Surface

```cpp
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/mesh.hpp>
#include <ndarray/ndarray.hh>

// 1. Create mesh
auto spatial_mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{64, 64, 64});
auto spacetime_mesh = std::make_shared<ExtrudedSimplicialMesh>(spatial_mesh, num_timesteps - 1);

// 2. Define predicate
CriticalPointPredicate<3, double> predicate;

// 3. Create engine
SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(spacetime_mesh, predicate);

// 4. Prepare data
ftk::ndarray<double> u, v, w;
// ... fill arrays ...
std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"W", w}};

// 5. Execute
engine.execute(data);

// 6. Get results
auto complex = engine.get_complex();
auto trajectories = complex.get_trajectories();
```

### Documentation Needed
- [ ] API reference (Doxygen)
- [ ] Developer guide (how to create custom predicates)
- [ ] Mesh topology primer
- [ ] Examples for each predicate type

### Estimated Effort
- Documentation: 2-3 weeks
- Examples: 1 week
- **Total**: 3-4 weeks

---

## Layer 2: High-Level API (C++)

### Purpose
- **Who**: C++ application developers
- **When**: Production code, configurable tracking
- **Complexity**: Low (configuration-driven)

### Current Status
❌ **NOT IMPLEMENTED** - Design complete (see HIGH_LEVEL_API_DESIGN.md)

### API Surface

```cpp
#include <ftk2/high_level/feature_tracker.hpp>

// Option A: From YAML
auto tracker = ftk2::FeatureTracker::from_yaml("tracking_config.yaml");
auto results = tracker->execute();

// Option B: Programmatic configuration
ftk2::TrackingConfig config;
config.feature = ftk2::FeatureType::CriticalPoints;
config.input.type = ftk2::InputType::Scalar;
config.input.variables = {"temperature"};
config.input.field_type = "gradient";  // Metadata for CP classification
config.data.source = ftk2::DataSourceType::Stream;
config.data.stream_yaml = "data_stream.yaml";
config.execution.backend = ftk2::Backend::CUDA;
config.execution.precision = ftk2::Precision::Double;

auto tracker = ftk2::FeatureTracker::create(config);
auto results = tracker->execute();

// Access results
for (const auto& traj : results.get_trajectories()) {
    std::cout << "Trajectory " << traj.id << ": " << traj.type_string() << "\n";
}
```

### YAML Configuration

```yaml
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: scalar
    variables: [temperature]
    field_type: gradient  # Hessian vs Jacobian classification

  data:
    source: stream
    stream_config: data_stream.yaml

  mesh:
    type: regular
    dimensions: [512, 512, 512]
    spacing: [1.0, 1.0, 1.0]

  execution:
    backend: cuda
    precision: double
    threads: 8

  output:
    trajectories: results.vtp
    statistics: stats.json
```

### Implementation Tasks
- [ ] `TrackingConfig` struct with validation
- [ ] YAML serialization/deserialization (yaml-cpp)
- [ ] Factory pattern (type dispatch)
- [ ] Input adapters (scalar→gradient, preprocessing)
- [ ] Data source abstraction (stream, arrays, VTU)
- [ ] Output writers (VTP, JSON, HDF5)
- [ ] Integration with Layer 1 (SimplicialEngine)
- [ ] Error handling and diagnostics

### Documentation Needed
- [ ] YAML configuration reference
- [ ] C++ API guide
- [ ] Example configs for each feature type
- [ ] Migration guide from low-level API

### Estimated Effort
- Core implementation: 4-5 weeks
- Testing and examples: 2 weeks
- Documentation: 1 week
- **Total**: 7-8 weeks

---

## Layer 3: JSON API + Executable

### Purpose
- **Who**: Command-line users, HPC workflows
- **When**: Batch processing, no coding required
- **Complexity**: Very low (just config files)

### Current Status
❌ **NOT IMPLEMENTED** - Legacy FTK had json_interface.hh

### Executable Usage

```bash
# Run tracking from config file
ftk2 tracking_config.yaml

# Or with JSON
ftk2 tracking_config.json

# Specify output format override
ftk2 tracking_config.yaml --output trajectories.vtp

# Multiple configs in parallel
mpirun -n 16 ftk2 tracking_config.yaml
```

### JSON/YAML Format
Same as Layer 2 YAML configuration. No difference in structure.

### Implementation

**File**: `src/ftk2_main.cpp`
```cpp
#include <ftk2/high_level/feature_tracker.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ftk2 <config.yaml|config.json>\n";
        return 1;
    }

    std::string config_path = argv[1];

    try {
        // Use Layer 2 API
        auto tracker = ftk2::FeatureTracker::from_yaml(config_path);

        std::cout << "Executing tracking...\n";
        auto results = tracker->execute();

        std::cout << "Found " << results.get_trajectories().size()
                  << " trajectories\n";

        // Output written automatically based on config
        std::cout << "Results written to " << tracker->get_output_path() << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
```

**CMake**:
```cmake
add_executable(ftk2 src/ftk2_main.cpp)
target_link_libraries(ftk2 PRIVATE ftk2_high_level ndarray::ndarray)
install(TARGETS ftk2 DESTINATION bin)
```

### Command-Line Options (Extended)

```bash
# Override config options
ftk2 config.yaml \
  --backend=cuda \
  --precision=float \
  --threads=16 \
  --output=custom_output.vtp

# Validate config without running
ftk2 config.yaml --validate

# Show estimated resource usage
ftk2 config.yaml --dry-run

# Verbose output
ftk2 config.yaml --verbose
```

### Implementation Tasks
- [ ] Main executable (`src/ftk2_main.cpp`)
- [ ] Command-line argument parsing (CLI11 or similar)
- [ ] Config validation and error messages
- [ ] Progress reporting (stdout)
- [ ] Return codes (success/failure)
- [ ] Man page / help text
- [ ] Installation / packaging

### Documentation Needed
- [ ] Command-line reference
- [ ] Example workflows
- [ ] HPC integration guide (job scripts)
- [ ] Troubleshooting

### Estimated Effort
- Implementation: 1-2 weeks
- Testing: 1 week
- Documentation: 1 week
- **Total**: 3-4 weeks

**Dependency**: Layer 2 must be complete

---

## Layer 4: Python High-Level API

### Purpose
- **Who**: Data scientists, Python developers
- **When**: Jupyter notebooks, analysis pipelines
- **Complexity**: Very low (Pythonic interface)

### Current Status
❌ **NOT IMPLEMENTED** - Design in PYTHON_BINDINGS.md

### API Surface

```python
import pyftk2

# Option A: From YAML (same as C++)
tracker = pyftk2.FeatureTracker.from_yaml("tracking_config.yaml")
results = tracker.execute()

# Option B: Programmatic (Pythonic)
config = pyftk2.TrackingConfig(
    feature="critical_points",
    dimension=3,
    input=pyftk2.InputConfig(
        type="scalar",
        variables=["temperature"],
        field_type="gradient"
    ),
    data=pyftk2.DataConfig(
        source="stream",
        stream_yaml="data_stream.yaml"
    ),
    execution=pyftk2.ExecutionConfig(
        backend="cuda",
        precision="double"
    )
)

tracker = pyftk2.FeatureTracker(config)
results = tracker.execute()

# Access results (Pythonic)
print(f"Found {len(results.trajectories)} trajectories")

for traj in results.trajectories:
    print(f"Trajectory {traj.id}: {traj.type}")
    print(f"  Length: {len(traj.positions)}")
    print(f"  Positions: {traj.positions.shape}")  # NumPy array

# Visualize
import matplotlib.pyplot as plt
for traj in results.trajectories:
    plt.plot(traj.positions[:, 0], traj.positions[:, 1])
plt.show()
```

### NumPy Integration

```python
# Direct array input (no stream)
import numpy as np

# Create synthetic data
dims = (64, 64, 64, 10)
u = np.random.randn(*dims)
v = np.random.randn(*dims)
w = np.random.randn(*dims)

# Track
tracker = pyftk2.CriticalPointTracker(
    mesh=(64, 64, 64),
    timesteps=10,
    backend="cuda"
)
tracker.set_fields(u=u, v=v, w=w)
results = tracker.execute()
```

### Implementation (pybind11)

**File**: `python/pyftk2.cpp`
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ftk2/high_level/feature_tracker.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pyftk2, m) {
    m.doc() = "Python bindings for FTK2 feature tracking";

    // TrackingConfig
    py::class_<ftk2::TrackingConfig>(m, "TrackingConfig")
        .def(py::init<>())
        .def_readwrite("feature", &ftk2::TrackingConfig::feature)
        .def_readwrite("dimension", &ftk2::TrackingConfig::dimension)
        // ... more fields
        .def("to_yaml", &ftk2::TrackingConfig::to_yaml)
        .def_static("from_yaml", &ftk2::TrackingConfig::from_yaml);

    // FeatureTracker
    py::class_<ftk2::FeatureTracker>(m, "FeatureTracker")
        .def_static("from_yaml", &ftk2::FeatureTracker::from_yaml)
        .def("execute", &ftk2::FeatureTracker::execute);

    // Results
    py::class_<ftk2::TrackingResults>(m, "TrackingResults")
        .def("get_trajectories", &ftk2::TrackingResults::get_trajectories);

    // Trajectory (with NumPy array for positions)
    py::class_<ftk2::Trajectory>(m, "Trajectory")
        .def_readonly("id", &ftk2::Trajectory::id)
        .def_property_readonly("positions", [](const ftk2::Trajectory& t) {
            // Convert to NumPy array (zero-copy if possible)
            return py::array_t<double>(...);
        });
}
```

### Implementation Tasks
- [ ] pybind11 bindings for TrackingConfig
- [ ] pybind11 bindings for FeatureTracker
- [ ] NumPy array conversions (zero-copy)
- [ ] Trajectory and results objects
- [ ] Exception handling (C++ → Python)
- [ ] setup.py / pyproject.toml
- [ ] Wheel building (PyPI)

### Documentation Needed
- [ ] Python API reference (Sphinx)
- [ ] Jupyter notebook tutorials
- [ ] Integration with pandas, xarray
- [ ] Gallery of examples

### Estimated Effort
- Core bindings: 3-4 weeks
- NumPy integration: 1 week
- Testing and examples: 2 weeks
- Documentation: 2 weeks
- **Total**: 8-9 weeks

**Dependency**: Layer 2 must be complete

---

## Layer 5: VTK/ParaView Plugin

### Purpose
- **Who**: Visualization users, domain scientists
- **When**: Interactive exploration in ParaView
- **Complexity**: Very low (GUI-driven)

### Current Status
❌ **NOT IMPLEMENTED** - Design in PARAVIEW_INTEGRATION.md

### ParaView Usage

**GUI Workflow**:
1. Load data in ParaView (VTU, NetCDF, etc.)
2. Apply filter: `Filters → FTK2 → Feature Tracker`
3. Configure in property panel:
   - Feature type: Critical Points ▼
   - Field: temperature ▼
   - Field type: gradient ▼
   - Backend: CUDA ☑
4. Click "Apply"
5. Visualize trajectories (tubes, glyphs, etc.)

### Plugin Architecture

```cpp
// VTK filter wrapper
class vtkFTK2FeatureTracker : public vtkPolyDataAlgorithm {
public:
    // Parameters (shown in property panel)
    vtkSetMacro(FeatureType, int);
    vtkSetStringMacro(FieldName);
    vtkSetStringMacro(FieldType);  // gradient vs flow
    vtkSetMacro(UseGPU, bool);

protected:
    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override {
        // Get input
        auto input = vtkUnstructuredGrid::GetData(inputVector[0]);

        // Convert to FTK2 mesh and data
        auto mesh = ConvertVTKToFTK2(input);
        auto data = ExtractFieldData(input, FieldName);

        // Use Layer 2 API
        ftk2::TrackingConfig config;
        config.feature = static_cast<ftk2::FeatureType>(FeatureType);
        config.input.type = ftk2::InputType::Scalar;
        config.input.variables = {FieldName};
        config.input.field_type = FieldType;
        config.execution.backend = UseGPU ? ftk2::Backend::CUDA : ftk2::Backend::CPU;

        auto tracker = ftk2::FeatureTracker::create(config);
        auto results = tracker->execute();

        // Convert results to vtkPolyData
        auto output = ConvertFTK2ToVTK(results);
        outputVector[0]->SetData(0, output);
        return 1;
    }

private:
    int FeatureType;
    char* FieldName;
    char* FieldType;
    bool UseGPU;
};
```

### ParaView XML

```xml
<ServerManagerConfiguration>
  <ProxyGroup name="filters">
    <SourceProxy name="FTK2FeatureTracker" class="vtkFTK2FeatureTracker">
      <Documentation>
        Track features in time-varying data using FTK2.
      </Documentation>

      <IntVectorProperty name="FeatureType"
                        command="SetFeatureType"
                        default_values="0">
        <EnumerationDomain name="enum">
          <Entry value="0" text="Critical Points"/>
          <Entry value="1" text="Levelsets"/>
          <Entry value="2" text="Parallel Vectors"/>
        </EnumerationDomain>
      </IntVectorProperty>

      <StringVectorProperty name="FieldName"
                           command="SetFieldName">
        <ArrayListDomain name="array_list" attribute_type="Scalars">
          <RequiredProperties>
            <Property name="Input" function="Input"/>
          </RequiredProperties>
        </ArrayListDomain>
      </StringVectorProperty>

      <StringVectorProperty name="FieldType"
                           command="SetFieldType"
                           default_values="gradient">
        <EnumerationDomain name="enum">
          <Entry value="gradient" text="Gradient (Hessian)"/>
          <Entry value="flow" text="Flow (Jacobian)"/>
        </EnumerationDomain>
      </StringVectorProperty>

      <IntVectorProperty name="UseGPU"
                        command="SetUseGPU"
                        default_values="0">
        <BooleanDomain name="bool"/>
      </IntVectorProperty>
    </SourceProxy>
  </ProxyGroup>
</ServerManagerConfiguration>
```

### Implementation Tasks
- [ ] VTK filter base (`vtkFTK2FeatureTracker`)
- [ ] Data conversion (VTK ↔ FTK2)
- [ ] ParaView XML property panel
- [ ] CMake integration with ParaView
- [ ] Plugin packaging (shared library)
- [ ] Testing with ParaView datasets

### Documentation Needed
- [ ] ParaView user guide
- [ ] Video tutorial
- [ ] Example datasets
- [ ] Troubleshooting

### Estimated Effort
- VTK filter implementation: 4-5 weeks
- ParaView XML and GUI: 2 weeks
- Testing and examples: 2 weeks
- Documentation: 1 week
- **Total**: 9-10 weeks

**Dependency**: Layer 2 must be complete

---

## Implementation Timeline

### Overall Dependency Graph

```
Layer 1 (Low-level)     [DONE]
     ↓
Layer 2 (High-level)    [4-6 weeks] ← Start here
     ↓
     ├─→ Layer 3 (Executable)    [3-4 weeks, parallel]
     ├─→ Layer 4 (Python)        [8-9 weeks, parallel]
     └─→ Layer 5 (ParaView)      [9-10 weeks, parallel]
```

### Phase 1: Foundation (Weeks 1-8)
**Goal**: Layer 2 (High-Level API) complete

**Tasks**:
- Week 1-2: TrackingConfig + YAML parsing
- Week 3-4: Factory and type dispatch
- Week 5-6: Data source abstraction (streams, arrays)
- Week 7-8: Integration, testing, documentation

**Deliverable**: Working high-level C++ API with YAML configs

---

### Phase 2: User Interfaces (Weeks 9-20, Parallel)

**Team A: Executable (Weeks 9-12)**
- Week 9-10: ftk2 executable + CLI parsing
- Week 11: Testing and examples
- Week 12: Documentation

**Team B: Python (Weeks 9-17)**
- Week 9-12: pybind11 bindings
- Week 13-14: NumPy integration and testing
- Week 15-16: Jupyter notebooks
- Week 17: Sphinx documentation

**Team C: ParaView (Weeks 9-18)**
- Week 9-13: VTK filter implementation
- Week 14-15: ParaView XML and property panel
- Week 16-17: Testing with datasets
- Week 18: Video tutorials

**By Week 18**: All 5 layers complete!

---

## Unified Testing Strategy

### Layer 1 (Low-level)
- Unit tests for each predicate
- Mesh topology tests
- SoS robustness tests

### Layer 2 (High-level)
- Config validation tests
- Type dispatch tests
- Integration tests (end-to-end)

### Layer 3 (Executable)
- CLI argument parsing tests
- Return code tests
- Example config tests

### Layer 4 (Python)
- Python unit tests (pytest)
- NumPy integration tests
- Jupyter notebook tests (nbval)

### Layer 5 (ParaView)
- VTK filter tests
- ParaView integration tests
- GUI workflow tests (manual)

### Cross-Layer Integration Tests
- Same config used in C++, CLI, Python, ParaView
- Verify identical results across all interfaces
- Performance comparison

---

## Documentation Structure

```
docs/
├── user_guide/
│   ├── low_level_api.md       # Layer 1
│   ├── high_level_api.md      # Layer 2
│   ├── command_line.md        # Layer 3
│   ├── python_api.md          # Layer 4
│   └── paraview_plugin.md     # Layer 5
├── tutorials/
│   ├── critical_points.md     # Using all 5 interfaces
│   ├── parallel_vectors.md
│   └── particles.md
├── api_reference/
│   ├── cpp/                   # Doxygen
│   └── python/                # Sphinx
└── examples/
    ├── configs/               # YAML/JSON examples
    ├── cpp/                   # C++ examples
    ├── python/                # Python scripts
    └── notebooks/             # Jupyter notebooks
```

---

## Resource Requirements

### Development Team
- **Core developer** (Layer 1-2): Full-time, 8 weeks
- **Python developer** (Layer 4): Part-time, 8 weeks
- **VTK developer** (Layer 5): Part-time, 10 weeks
- **Technical writer**: Part-time, ongoing

### Infrastructure
- CI/CD for all layers (GitHub Actions)
- Build matrix (Linux, macOS, Windows)
- Test datasets (synthetic + real)
- Documentation hosting (Read the Docs, GitHub Pages)

---

## Success Metrics

### Technical
- [ ] All 5 layers functional
- [ ] Identical results across layers
- [ ] 90%+ test coverage
- [ ] <5 seconds startup time (executable)

### Usability
- [ ] Scientists can use without C++ knowledge (Python, ParaView)
- [ ] HPC users can run from command line (executable)
- [ ] Clear error messages at all layers
- [ ] Comprehensive documentation and examples

### Adoption
- [ ] 10+ users per layer
- [ ] Community contributions (GitHub PRs)
- [ ] ParaView plugin in official distribution
- [ ] PyPI package (pyftk2)

---

## Risk Mitigation

### Risk 1: Layer 2 takes longer than expected
**Mitigation**: Layer 2 is foundation. Extend timeline if needed. Do NOT compromise quality.

### Risk 2: Python bindings are complex
**Mitigation**: Start with simple wrappers. Enhance iteratively based on user feedback.

### Risk 3: ParaView integration breaks
**Mitigation**: Maintain backward compatibility. Test against multiple ParaView versions.

### Risk 4: Documentation lags behind code
**Mitigation**: Documentation is part of "done". No feature complete without docs.

---

## Next Steps

1. **Validate priorities**: Which layer is most urgent for users?
2. **Team formation**: Who can work on each layer?
3. **Start Phase 1**: Implement Layer 2 (high-level API)
4. **Continuous feedback**: User testing at each milestone

**Question for discussion**: Should we implement all layers, or prioritize based on target audience (e.g., Python + ParaView first for scientists)?
