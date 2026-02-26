# High-Level API Design for FTK2

## Problem Statement

FTK2 needs a high-level abstraction to handle the combinatorial complexity of:
- **Feature types**: Critical points, levelsets, fibers, parallel vectors, magnetic flux, etc.
- **Input types**: Scalar (→gradient), vector, multi-field, phase/complex
- **Precisions**: float, double
- **Input sources**: ndarray streams (YAML), direct arrays, VTU files, synthetic
- **Execution backends**: CPU, CUDA, MPI
- **Mesh types**: Regular, unstructured, extruded

**Current problem**: The low-level `execute(data_map)` API requires users to:
- Manually compute gradients
- Handle type conversions
- Know internal data layout
- Route to correct backend

**Goal**: High-level API where users specify **WHAT** (not HOW):
```cpp
"Track critical points in this scalar field using GPU"
```

The system should automatically:
- Compute gradients
- Dispatch to correct types
- Handle preprocessing
- Route to appropriate backend

---

## Design Options

### Option 1: Fluent Builder Pattern

**API**:
```cpp
auto tracker = ftk2::FeatureTracker::create()
    .feature("critical_points")
    .dimension(3)
    .input_scalar("temperature")  // Auto-compute gradient
    .from_stream("config.yaml")
    .use_gpu(true)
    .precision<double>()
    .build();

auto results = tracker.execute();
```

**Advantages**:
- Fluent, readable API
- Type safety at compile time
- Easy validation before execution

**Disadvantages**:
- Verbose for simple cases
- Harder to configure from files (YAML)
- C++ API only (hard to expose to Python)

**Implementation sketch**:
```cpp
class FeatureTrackerBuilder {
    std::string feature_type_;
    int dimension_;
    std::string input_type_;  // "scalar", "vector", "multi_scalar"
    std::vector<std::string> variables_;
    std::shared_ptr<ftk::stream> stream_;
    bool use_gpu_ = false;
    std::string precision_ = "double";

public:
    FeatureTrackerBuilder& feature(const std::string& type) {
        feature_type_ = type;
        return *this;
    }

    FeatureTrackerBuilder& input_scalar(const std::string& var) {
        input_type_ = "scalar";
        variables_ = {var};
        return *this;
    }

    FeatureTrackerBuilder& from_stream(const std::string& yaml_path) {
        stream_ = std::make_shared<ftk::stream>();
        stream_->parse_yaml(yaml_path);
        return *this;
    }

    std::unique_ptr<FeatureTracker> build() {
        // Validate configuration
        if (feature_type_.empty()) throw std::runtime_error("Feature type not set");

        // Dispatch to correct implementation
        if (feature_type_ == "critical_points") {
            if (input_type_ == "scalar") {
                // Will need gradient computation
                if (precision_ == "double") {
                    if (use_gpu_) {
                        return std::make_unique<CPTrackerScalarGPU<double>>(...);
                    } else {
                        return std::make_unique<CPTrackerScalarCPU<double>>(...);
                    }
                } else {
                    // float version
                }
            } else if (input_type_ == "vector") {
                // Direct vector field
            }
        }
        // ... other feature types
    }
};
```

---

### Option 2: Configuration Object + Factory

**API**:
```cpp
ftk2::TrackingConfig config;
config.feature = ftk2::FeatureType::CriticalPoints;
config.dimension = 3;
config.input.type = ftk2::InputType::Scalar;
config.input.variables = {"temperature"};
config.source.stream_yaml = "config.yaml";
config.execution.backend = ftk2::Backend::CUDA;
config.execution.precision = ftk2::Precision::Double;

auto tracker = ftk2::FeatureTracker::create(config);
auto results = tracker->execute();
```

**Advantages**:
- Configuration is data (can serialize/deserialize)
- Easy to expose to Python
- Can be constructed from YAML
- Clear separation of configuration and execution

**Disadvantages**:
- More boilerplate
- Runtime type checking (unless using std::variant)

**Implementation sketch**:
```cpp
enum class FeatureType { CriticalPoints, Levelsets, Fibers, ParallelVectors, MagneticFlux };
enum class InputType { Scalar, Vector, MultiScalar, Complex };
enum class Backend { CPU, CUDA, MPI };
enum class Precision { Float, Double };

struct InputConfig {
    InputType type;
    std::vector<std::string> variables;
    // For scalar: compute gradient
    // For vector: use directly
    // For multi_scalar: multiple inputs
};

struct SourceConfig {
    std::string stream_yaml;  // If using stream
    std::map<std::string, ftk::ndarray<double>> arrays;  // If direct arrays
    std::string vtu_path;  // If VTU file
};

struct ExecutionConfig {
    Backend backend = Backend::CPU;
    Precision precision = Precision::Double;
    int num_threads = -1;  // -1 = auto
};

struct TrackingConfig {
    FeatureType feature;
    int dimension;
    InputConfig input;
    SourceConfig source;
    ExecutionConfig execution;

    // Validation
    void validate() const {
        if (dimension < 2 || dimension > 3) throw std::invalid_argument("Invalid dimension");
        // ... more checks
    }

    // Serialization
    void to_yaml(const std::string& path) const;
    static TrackingConfig from_yaml(const std::string& path);
};

class FeatureTracker {
public:
    static std::unique_ptr<FeatureTracker> create(const TrackingConfig& config) {
        config.validate();

        // Type dispatch
        if (config.execution.precision == Precision::Double) {
            return create_typed<double>(config);
        } else {
            return create_typed<float>(config);
        }
    }

    virtual FeatureComplex execute() = 0;

private:
    template <typename T>
    static std::unique_ptr<FeatureTracker> create_typed(const TrackingConfig& config) {
        // Feature type dispatch
        if (config.feature == FeatureType::CriticalPoints) {
            return create_cp_tracker<T>(config);
        }
        // ... other features
    }

    template <typename T>
    static std::unique_ptr<FeatureTracker> create_cp_tracker(const TrackingConfig& config) {
        // Backend dispatch
        if (config.execution.backend == Backend::CUDA) {
            return std::make_unique<CPTrackerGPU<T>>(config);
        } else {
            return std::make_unique<CPTrackerCPU<T>>(config);
        }
    }
};
```

---

### Option 3: Unified YAML Configuration (Highest Level)

**API**:
```cpp
// C++ usage
auto tracker = ftk2::FeatureTracker::from_yaml("tracking_config.yaml");
auto results = tracker->execute();
```

**YAML config**:
```yaml
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: scalar
    variables:
      - temperature
    # Auto-compute gradient from scalar field

  source:
    type: stream
    config: data_stream.yaml
    # OR
    # type: vtu
    # path: mesh.vtu
    # OR
    # type: synthetic
    # generator: moving_maximum

  execution:
    backend: cuda
    precision: double
    threads: 8

  output:
    trajectories: results.vtp
    statistics: stats.json
```

**Data stream YAML** (`data_stream.yaml`):
```yaml
stream:
  path_prefix: /data/simulation
  substreams:
    - name: input
      format: netcdf
      filenames: "timestep_*.nc"
      vars:
        - name: temperature
          possible_names: [temperature, temp, T]
```

**Python usage**:
```python
import pyftk2

# Everything from YAML
tracker = pyftk2.FeatureTracker.from_yaml("tracking_config.yaml")
results = tracker.execute()

# Or hybrid (YAML + overrides)
config = pyftk2.TrackingConfig.from_yaml("tracking_config.yaml")
config.execution.use_gpu = True  # Override
tracker = pyftk2.FeatureTracker(config)
```

**Advantages**:
- Most user-friendly (scientists configure via YAML)
- Configuration is data (version control, reproducibility)
- Works identically in C++ and Python
- Easy to generate configs programmatically

**Disadvantages**:
- Requires YAML parsing
- Runtime configuration means later error detection
- Need good error messages

---

## Preprocessing and Type Dispatch

### Critical Distinction: Gradient vs. Non-Gradient Vector Fields

**IMPORTANT**: For critical point tracking, there are **two fundamentally different cases**:

1. **Gradient vector fields** (v = ∇s, derived from scalar field s):
   - CP classification uses **Hessian** of s
   - Types: minimum, maximum, 1-saddle, 2-saddle, degenerate
   - Physics: Potential fields, temperature gradients, pressure gradients

2. **Non-gradient vector fields** (general flow fields):
   - CP classification uses **Jacobian** of v
   - Types: source, sink, saddle, spiral (focus), center, etc.
   - Physics: Fluid flow, magnetic fields, general dynamical systems

**Users MUST specify which case applies**, as the classification methods are completely different.

### Input Type Taxonomy

```yaml
input:
  type: scalar  # Will compute gradient → gradient vector field (Hessian classification)
  variables: [temperature]

# OR

input:
  type: gradient_vector  # Already a gradient field (Hessian classification)
  variables: [grad_x, grad_y, grad_z]
  scalar_field: temperature  # Optional: if available for Hessian computation

# OR

input:
  type: vector  # Non-gradient vector field (Jacobian classification)
  variables: [u, v, w]
```

### Input Adapter Layer

```cpp
enum class VectorFieldType {
    Gradient,      // From scalar or marked as gradient (use Hessian)
    NonGradient    // General flow field (use Jacobian)
};

class InputAdapter {
public:
    virtual ~InputAdapter() = default;

    // Returns vector field (either direct or computed)
    virtual std::map<std::string, ftk::ndarray<T>> get_vector_field() = 0;

    // Returns classification type
    virtual VectorFieldType get_field_type() const = 0;

    // For gradient fields: return scalar for Hessian computation
    virtual const ftk::ndarray<T>* get_scalar_field() const { return nullptr; }
};

class ScalarGradientAdapter : public InputAdapter {
    ftk::ndarray<T> scalar_;
    mutable std::map<std::string, ftk::ndarray<T>> gradient_cache_;

public:
    std::map<std::string, ftk::ndarray<T>> get_vector_field() override {
        if (gradient_cache_.empty()) {
            auto grad = compute_gradient(scalar_);
            gradient_cache_ = {
                {"U", grad[0]},
                {"V", grad[1]},
                {"W", grad[2]}
            };
        }
        return gradient_cache_;
    }

    VectorFieldType get_field_type() const override {
        return VectorFieldType::Gradient;  // Will use Hessian
    }

    const ftk::ndarray<T>* get_scalar_field() const override {
        return &scalar_;  // Provide for Hessian computation
    }

private:
    std::array<ftk::ndarray<T>, 3> compute_gradient(const ftk::ndarray<T>& scalar) {
        // Finite difference gradient computation
        // Uses mesh spacing from config
    }
};

class GradientVectorAdapter : public InputAdapter {
    // User provides gradient components directly
    std::map<std::string, ftk::ndarray<T>> gradient_;
    ftk::ndarray<T> scalar_;  // Optional
    bool has_scalar_;

public:
    std::map<std::string, ftk::ndarray<T>> get_vector_field() override {
        return gradient_;
    }

    VectorFieldType get_field_type() const override {
        return VectorFieldType::Gradient;  // Will use Hessian
    }

    const ftk::ndarray<T>* get_scalar_field() const override {
        return has_scalar_ ? &scalar_ : nullptr;
    }
};

class NonGradientVectorAdapter : public InputAdapter {
    // General vector field (flow, magnetic field, etc.)
    std::map<std::string, ftk::ndarray<T>> vector_;

public:
    std::map<std::string, ftk::ndarray<T>> get_vector_field() override {
        return vector_;
    }

    VectorFieldType get_field_type() const override {
        return VectorFieldType::NonGradient;  // Will use Jacobian
    }
};
```

### Type Dispatch Hierarchy

```cpp
// Level 1: Feature type dispatch
template <typename T>
std::unique_ptr<FeatureTracker> create_tracker(const TrackingConfig& config) {
    switch (config.feature) {
        case FeatureType::CriticalPoints:
            return create_cp_tracker<T>(config);
        case FeatureType::Levelsets:
            return create_levelset_tracker<T>(config);
        // ... other features
    }
}

// Level 2: Input type dispatch
template <typename T>
std::unique_ptr<FeatureTracker> create_cp_tracker(const TrackingConfig& config) {
    std::unique_ptr<InputAdapter> adapter;

    switch (config.input.type) {
        case InputType::Scalar:
            adapter = std::make_unique<ScalarGradientAdapter<T>>(config);
            break;
        case InputType::Vector:
            adapter = std::make_unique<DirectVectorAdapter<T>>(config);
            break;
    }

    return create_cp_backend<T>(config, std::move(adapter));
}

// Level 3: Backend dispatch
template <typename T>
std::unique_ptr<FeatureTracker> create_cp_backend(
    const TrackingConfig& config,
    std::unique_ptr<InputAdapter> adapter)
{
    switch (config.execution.backend) {
        case Backend::CPU:
            return std::make_unique<CPTrackerCPU<T>>(config, std::move(adapter));
        case Backend::CUDA:
            return std::make_unique<CPTrackerGPU<T>>(config, std::move(adapter));
        case Backend::MPI:
            return std::make_unique<CPTrackerMPI<T>>(config, std::move(adapter));
    }
}
```

---

## Handling Streams

### Stream Abstraction

```cpp
class DataSource {
public:
    virtual ~DataSource() = default;

    virtual int get_num_timesteps() const = 0;
    virtual std::shared_ptr<ftk::ndarray_group> read_timestep(int t) = 0;
    virtual std::vector<std::string> get_variable_names() const = 0;
};

class StreamDataSource : public DataSource {
    ftk::stream stream_;

public:
    StreamDataSource(const std::string& yaml_path) {
        stream_.parse_yaml(yaml_path);
    }

    int get_num_timesteps() const override {
        return stream_.total_timesteps();
    }

    std::shared_ptr<ftk::ndarray_group> read_timestep(int t) override {
        return stream_.read(t);
    }
};

class DirectArrayDataSource : public DataSource {
    std::map<std::string, ftk::ndarray<T>> arrays_;
    int num_timesteps_;

public:
    DirectArrayDataSource(const std::map<std::string, ftk::ndarray<T>>& arrays, int nt)
        : arrays_(arrays), num_timesteps_(nt) {}

    int get_num_timesteps() const override {
        return num_timesteps_;
    }

    std::shared_ptr<ftk::ndarray_group> read_timestep(int t) override {
        // Extract slice at timestep t
        auto group = std::make_shared<ftk::ndarray_group>();
        for (const auto& [name, arr] : arrays_) {
            auto slice = arr.slice_time(t);
            group->set(name, slice);
        }
        return group;
    }
};
```

---

## Complete Examples: Different Input Types

### Example 1: Scalar Field → Gradient Vector Field (Hessian Classification)

**Use case**: Tracking temperature extrema (min/max) and saddle points.

**tracking_config.yaml**:
```yaml
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: scalar
    variables:
      - temperature

  source:
    type: stream
    config: climate_data.yaml

  mesh:
    type: regular
    dimensions: [512, 512, 512]
    spacing: [1.0, 1.0, 1.0]

  execution:
    backend: cuda
    precision: double

  output:
    trajectories: cp_trajectories.vtp
    statistics: cp_stats.json
```

**climate_data.yaml**:
```yaml
stream:
  path_prefix: /data/cesm
  substreams:
    - name: atmosphere
      format: netcdf
      filenames: "atm_*.nc"
      vars:
        - name: temperature
          possible_names: [T, temp, temperature]
```

**C++ Usage**:
```cpp
#include <ftk2/high_level/feature_tracker.hpp>

int main() {
    auto tracker = ftk2::FeatureTracker::from_yaml("tracking_config.yaml");
    auto results = tracker->execute();

    // CPs classified as min/max/1-saddle/2-saddle (Hessian)
    for (const auto& traj : results.get_trajectories()) {
        std::cout << "Trajectory " << traj.id << ": " << traj.type_string() << "\n";
        // type_string() returns "minimum", "maximum", "1-saddle", "2-saddle"
    }
    return 0;
}
```

---

### Example 2: Non-Gradient Vector Field (Jacobian Classification)

**Use case**: Tracking critical points in fluid flow (sources, sinks, saddles, spirals).

**tracking_config.yaml**:
```yaml
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: vector  # Direct vector field
    variables: [u, v, w]
    field_type: flow  # IMPORTANT: non-gradient → Jacobian classification

  data:
    source: stream
    stream_config: cfd_flow.yaml

  mesh:
    type: regular
    dimensions: [128, 128, 64]

  execution:
    backend: cuda
    precision: float  # Lower precision for CFD

  output:
    trajectories: flow_cp.vtp
```

**C++ Usage**:
```cpp
auto tracker = ftk2::FeatureTracker::from_yaml("tracking_config.yaml");
auto results = tracker->execute();

// CPs classified using Jacobian (source/sink/saddle/spiral/center)
for (const auto& traj : results.get_trajectories()) {
    std::cout << "Flow CP " << traj.id << ": " << traj.type_string() << "\n";
    // type_string() returns "source", "sink", "saddle", "spiral", "center"
}
```

---

### Example 3: Gradient Vector Field (Pre-computed)

**Use case**: User already computed ∇T externally, provides gradient components.

**tracking_config.yaml**:
```yaml
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: gradient_vector  # Already a gradient
    variables: [grad_x, grad_y, grad_z]
    field_type: gradient  # Use Hessian
    scalar_field: temperature  # Optional: for Hessian if available

  data:
    source: stream
    stream_config: precomputed_gradient.yaml

  execution:
    backend: cpu

  output:
    trajectories: gradient_cp.vtp
```

**Key difference**: No gradient computation needed, but still use Hessian classification.

---

### Summary: field_type is Metadata

**Important insight** (per user feedback): `field_type: gradient` vs `field_type: flow` is just **metadata** that affects:
- CP classification method (Hessian vs Jacobian)
- CP type labeling (min/max/saddle vs source/sink/spiral)

**It does NOT fundamentally change**:
- Tracking algorithm (still simplicial spacetime)
- Data pipeline (still uses ndarray streams)
- GPU dispatch (same CUDA kernels)
- MPI parallelism (same distributed union-find)

It's a **configuration parameter**, not an architectural concern.

### C++ Usage (Original)

### Python Usage

```python
import pyftk2

# Load configuration
tracker = pyftk2.FeatureTracker.from_yaml("tracking_config.yaml")

# Execute
results = tracker.execute()

# Analysis
print(f"Found {len(results.trajectories)} trajectories")

# Visualization
import matplotlib.pyplot as plt
for traj in results.trajectories:
    plt.plot(traj.positions[:, 0], traj.positions[:, 1])
plt.show()
```

### What Happens Under the Hood

1. **Parse YAML** → `TrackingConfig` object
2. **Validate config** (feature type valid, variables exist, etc.)
3. **Dispatch by precision** → `create_tracker<double>(config)`
4. **Dispatch by feature** → `create_cp_tracker<double>(config)`
5. **Create input adapter** → `ScalarGradientAdapter` (will compute ∇T)
6. **Dispatch by backend** → `CPTrackerGPU<double>`
7. **Create data source** → `StreamDataSource` (wraps ndarray stream)
8. **Execute**:
   - For each timestep pair (t, t+1):
     - Read data via stream
     - Compute gradient (adapter)
     - Transfer to GPU
     - Extract features
     - Stitch manifolds
   - Build trajectories (union-find)
9. **Write output** (VTP, JSON)

---

## Feature Type Support

Based on [FEATURE_GAP_ANALYSIS.md](FEATURE_GAP_ANALYSIS.md), the high-level API should support:

### Currently Supported (FTK2)
- `critical_points` - Critical points in 2D/3D
- `levelsets` - Isosurface tracking
- `fibers` - Isosurface intersections

### Planned Features (Priority Order)
- `parallel_vectors` - ExactPV vortex cores (HIGH)
- `critical_lines` - Ridge/valley lines (HIGH)
- `particles` - Lagrangian particle tracing (MEDIUM-HIGH)
- `tdgl_vortex` - Magnetic flux vortices (MEDIUM-HIGH)
- `sujudi_haimes` - Vortex cores (MEDIUM)
- `levy_degani_seginer` - Alternative vortex cores (MEDIUM)
- `feature_flow` - Stable feature flow fields (LOW-MEDIUM)

### YAML Feature Configuration

```yaml
tracking:
  feature: <feature_type>  # See list above

  # Feature-specific options
  options:
    # For critical_lines
    line_type: ridge  # or valley

    # For particles
    integrator: rk4  # or rk1
    num_steps: 1000
    dt: 0.01
    seeding:
      type: grid  # or random, file
      stride: [4, 4, 4]

    # For tdgl_vortex
    min_winding: 1  # Minimum winding number
```

---

## Implementation Strategy

### Phase 1: Configuration Infrastructure
**Goal**: Define config objects and YAML parsing

**Tasks**:
- [ ] Define `TrackingConfig` struct with all options
- [ ] YAML serialization/deserialization
- [ ] Config validation
- [ ] Error messages

### Phase 2: Data Source Abstraction
**Goal**: Unified interface for different input sources

**Tasks**:
- [ ] `DataSource` base class
- [ ] `StreamDataSource` (wraps ndarray stream)
- [ ] `DirectArrayDataSource` (for testing)
- [ ] `VTUDataSource` (read VTU series)
- [ ] `SyntheticDataSource` (for testing)

### Phase 3: Input Adapters
**Goal**: Handle preprocessing (gradient computation, etc.)

**Tasks**:
- [ ] `InputAdapter` base class
- [ ] `DirectVectorAdapter` (no preprocessing)
- [ ] `ScalarGradientAdapter` (compute gradient)
- [ ] `MultiScalarAdapter` (for fiber tracking)
- [ ] Finite difference gradient implementation

### Phase 4: Factory and Dispatch
**Goal**: Route to correct implementation based on config

**Tasks**:
- [ ] `FeatureTracker::create()` factory
- [ ] Type dispatch (float/double)
- [ ] Feature dispatch (CP/levelset/fiber)
- [ ] Backend dispatch (CPU/CUDA/MPI)

### Phase 5: Integration
**Goal**: Connect high-level API to existing engine

**Tasks**:
- [ ] Adapt `SimplicialEngine` to work with new API
- [ ] Streaming execution loop
- [ ] Output writers (VTP, JSON, HDF5)

### Phase 6: Python Bindings
**Goal**: Expose high-level API to Python

**Tasks**:
- [ ] pybind11 wrappers for `TrackingConfig`
- [ ] Python `FeatureTracker` class
- [ ] Result objects (trajectories, statistics)
- [ ] Examples and notebooks

---

## Open Questions

1. **Mesh specification**:
   - Regular: dimensions + spacing
   - Unstructured: extract from VTU?
   - Should mesh be part of config or inferred from data source?

2. **Multi-timestep data**:
   - Some formats store all timesteps in one file
   - Others use one file per timestep
   - How to handle both uniformly?

3. **GPU memory management**:
   - Stream data to GPU (only active slab)
   - Or load all in GPU memory if fits?
   - User-configurable threshold?

4. **MPI decomposition**:
   - Automatic based on available ranks?
   - Or user-specified decomposition strategy?

5. **Incremental execution**:
   - Should we support checkpointing?
   - Resume from partial results?

---

## Cleaner Design: Separation of Concerns

### Learning from FTK's json_interface.hh

The old FTK had `json_interface.hh` which was a **monolithic class** mixing:
- Configuration parsing
- Type dispatch
- Execution
- Output writing

**Problems**:
- Hard to test individual components
- Difficult to extend with new features
- Configuration and execution tightly coupled
- Hard to use programmatically (non-YAML use cases)

### New FTK2 Design: Modular Layers

```
┌─────────────────────────────────────────┐
│  Configuration Layer (YAML/JSON)        │  ← User-facing
├─────────────────────────────────────────┤
│  Validation Layer                       │  ← Check config validity
├─────────────────────────────────────────┤
│  Factory Layer (Type Dispatch)          │  ← Route to implementations
├─────────────────────────────────────────┤
│  Adapter Layer (Preprocessing)          │  ← Gradient, type conversion
├─────────────────────────────────────────┤
│  Data Source Layer (I/O Abstraction)    │  ← ndarray streams, VTU, etc.
├─────────────────────────────────────────┤
│  Execution Layer (SimplicialEngine)     │  ← Core tracking algorithms
├─────────────────────────────────────────┤
│  Output Layer (Writers)                 │  ← VTP, JSON, HDF5, etc.
└─────────────────────────────────────────┘
```

Each layer is **independent, testable, and extensible**.

### Clean Integration with ndarray YAML Streams

**Key insight**: ndarray already has YAML streams. FTK2 should **extend** ndarray YAML with tracking-specific options, not reinvent it.

**Two YAML files approach**:

1. **Data stream** (ndarray format):
```yaml
# data_stream.yaml - Pure ndarray config
stream:
  path_prefix: /data/simulation
  substreams:
    - name: input
      format: netcdf
      filenames: "timestep_*.nc"
      vars:
        - name: temperature
          possible_names: [T, temp, temperature]
```

2. **Tracking config** (FTK2-specific):
```yaml
# tracking_config.yaml - FTK2 config referencing data stream
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: scalar  # Will compute gradient
    variables: [temperature]
    field_type: gradient  # IMPORTANT: gradient vs non-gradient
    # field_type: gradient → use Hessian (min/max/saddle)
    # field_type: flow → use Jacobian (source/sink/spiral)

  data:
    source: stream
    stream_config: data_stream.yaml  # Reference ndarray YAML

  mesh:
    type: regular
    dimensions: [512, 512, 512]
    spacing: [1.0, 1.0, 1.0]

  execution:
    backend: cuda
    precision: double

  output:
    trajectories: results.vtp
```

**Advantage**: Separation of data I/O (ndarray) and tracking (FTK2).

### Alternative: Merged YAML (Single File)

```yaml
# merged_config.yaml - Everything in one file
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: scalar
    variables: [temperature]
    field_type: gradient  # Use Hessian classification

  data:
    source: stream
    stream:  # Inline ndarray stream config
      path_prefix: /data/simulation
      substreams:
        - name: input
          format: netcdf
          filenames: "timestep_*.nc"
          vars:
            - name: temperature

  mesh:
    type: regular
    dimensions: [512, 512, 512]

  execution:
    backend: cuda

  output:
    trajectories: results.vtp
```

**Both approaches supported** - user chooses.

## Recommendation

I recommend **Option 3 (Unified YAML Configuration)** with the **modular layer design** because:

1. **Most user-friendly**: Scientists configure via YAML, no code
2. **Reproducible**: Configuration is version-controlled
3. **Language-agnostic**: Same config works in C++ and Python
4. **Flexible**: Can override in code if needed
5. **Industry standard**: YAML configs are ubiquitous

Implementation approach:
1. Start with **Option 2 (Configuration Object)** infrastructure
2. Add YAML serialization to get **Option 3**
3. Optionally add **Option 1 (Builder)** for programmatic construction

This provides maximum flexibility while keeping the most common use case (YAML-driven) extremely simple.

---

## Next Steps

**For discussion**:
1. Does the config structure make sense for your use cases?
2. What other feature types need to be supported?
3. Should we prioritize Python or C++ API first?
4. Any critical preprocessing steps I'm missing?

**For implementation** (if design approved):
1. Create `TrackingConfig` struct
2. Implement YAML parsing (using yaml-cpp)
3. Create simple factory for critical points + scalar input + CPU backend
4. Integrate with existing `SimplicialEngine`
5. Add tests and examples

What aspects of this design should we refine first?
