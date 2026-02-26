# Approximate Parallel Vector (ApproxPV) High-Level API

## Overview

The **Approximate Parallel Vector (ApproxPV)** tracking provides a fiber-based approach to detect where two vector fields U and V are approximately parallel. This avoids expensive cubic root finding required by traditional parallel vector methods.

## Method

Given two vector fields **U** and **V**:

1. **Compute cross product**: W = U × V
2. **Track fiber surface**: W₀ = W₁ = 0 (first two components zero)
3. **Record attribute**: W₂ (third component) for filtering
4. **Filter results**: Keep features where |W₂| < threshold

When U ∥ V (parallel), the cross product W ≈ 0, making this an effective approximate detector.

## Derived Descriptors

### 1. Sujudi-Haimes Vortex Cores

**Physical meaning**: Vortex cores where velocity is parallel to vorticity.

**Fields**:
- U = velocity
- V = vorticity = ∇ × velocity

**Application**: Turbulent flow vortex identification

### 2. Levy-Degani-Seginer Criterion

**Physical meaning**: Alternative vortex core definition.

**Fields**:
- U = velocity
- V = ∇ × velocity - (velocity · ∇)velocity

**Application**: Compressible flow vortex identification

### 3. General ApproxPV

**Physical meaning**: Any two vector fields with parallel regions.

**Fields**: User-defined U and V

**Application**: Custom feature detection

---

## YAML Configuration

### Example 1: Sujudi-Haimes Vortex Cores

```yaml
tracking:
  feature: sujudi_haimes
  dimension: 3  # 3D spatial (4D spacetime)

  input:
    type: vector
    variables: [u, v, w]
    field_type: flow

  data:
    source: stream
    stream_yaml: velocity_data.yaml

  options:
    auto_compute_vorticity: true  # Automatically compute ∇ × velocity
    w2_threshold: 0.1             # Filter by |W_2| < 0.1
    filter_mode: absolute

  output:
    filename: vortex_cores.vtp
    attributes:
      - name: w2_magnitude
        source: w2
        type: scalar
      - name: velocity_magnitude
        source: velocity
        type: magnitude
      - name: vorticity_magnitude
        source: vorticity
        type: magnitude

  execution:
    backend: cuda
    precision: double
```

**What happens**:
1. FTK2 loads velocity field (u, v, w)
2. Automatically computes vorticity: ω = ∇ × velocity
3. Computes W = velocity × vorticity
4. Tracks fiber W₀ = W₁ = 0
5. Records W₂, velocity magnitude, vorticity magnitude as attributes
6. Filters results by |W₂| < 0.1
7. Outputs vortex core trajectories

---

### Example 2: General ApproxPV (Custom Vector Pairs)

```yaml
tracking:
  feature: approx_parallel_vectors
  dimension: 3

  input:
    type: paired_vectors
    vector_u: velocity
    vector_v: magnetic_field

  data:
    source: stream
    stream_yaml: mhd_data.yaml

  options:
    w2_threshold: 0.05
    filter_mode: percentile
    filter_percentile: 0.1  # Top 10% most parallel

  output:
    filename: parallel_features.vtp
    attributes:
      - name: w2
        source: w2
        type: scalar
      - name: parallelism
        source: w2
        type: abs_reciprocal  # 1/|W_2| measures parallelism

  execution:
    backend: cpu
```

**What happens**:
1. Loads pre-existing velocity and magnetic_field arrays
2. Computes W = velocity × magnetic_field
3. Tracks fiber W₀ = W₁ = 0
4. Filters top 10% most parallel features (smallest |W₂|)
5. Outputs results with parallelism metric

---

### Example 3: Levy-Degani-Seginer

```yaml
tracking:
  feature: levy_degani_seginer
  dimension: 3

  input:
    type: vector
    variables: [u, v, w]
    field_type: flow

  data:
    source: vtu
    vtu_pattern: "compressible_flow_*.vtu"

  options:
    w2_threshold: 0.15
    filter_mode: absolute

  output:
    filename: lds_vortices.vtp

  execution:
    backend: cuda
    precision: float
```

**What happens**:
1. Loads velocity from VTU series
2. Computes V = ∇ × velocity - (velocity · ∇)velocity
3. Computes W = velocity × V
4. Tracks and filters
5. Outputs vortex lines

---

### Example 4: Multi-Resolution Filtering

```yaml
tracking:
  feature: sujudi_haimes
  dimension: 3

  input:
    type: vector
    variables: [u, v, w]

  data:
    source: stream
    stream_yaml: turbulence.yaml

  options:
    auto_compute_vorticity: true
    # No threshold - output all detections for post-processing
    w2_threshold: -1  # Negative = no filtering

  output:
    filename: all_vortex_candidates.vtp
    attributes:
      - name: w2
        source: w2
        type: scalar
      - name: w2_abs
        source: w2
        type: abs
      - name: velocity_mag
        source: velocity
        type: magnitude
      - name: vorticity_mag
        source: vorticity
        type: magnitude
      - name: q_criterion
        source: q_criterion
        type: scalar

  execution:
    backend: cuda
    precision: double
```

**Use case**: Export all candidates with rich attributes for external filtering/analysis.

---

## Programming API

### Low-Level (Direct)

```cpp
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/numeric/cross_product.hpp>

// 1. Load or generate data
ftk::ndarray<double> velocity, vorticity;
// ... load data ...

// 2. Compute cross product
ftk::ndarray<double> w;
cross_product_3d(velocity, vorticity, w);

// 3. Decompose into components
auto w_components = decompose_components(w);

// 4. Create mesh
auto mesh = std::make_shared<RegularSimplicialMesh>(
    std::vector<uint64_t>{nx, ny, nz, nt});

// 5. Configure fiber predicate
FiberPredicate<double> predicate;
predicate.var_names[0] = "w0";
predicate.var_names[1] = "w1";
predicate.thresholds[0] = 0.0;
predicate.thresholds[1] = 0.0;

// 6. Add W₂ as attribute
AttributeSpec w2_attr;
w2_attr.name = "w2";
w2_attr.source = "w2";
w2_attr.type = "scalar";
w2_attr.slot = 0;
predicate.attributes.push_back(w2_attr);

// 7. Prepare data map
std::map<std::string, ftk::ndarray<double>> data;
data["w0"] = w_components[0];
data["w1"] = w_components[1];
data["w2"] = w_components[2];

// 8. Execute tracking
SimplicialEngine<double, FiberPredicate<double>> engine(mesh, predicate);
engine.execute(data);

// 9. Get and filter results
auto complex = engine.get_complex();
double w2_threshold = 0.1;

for (const auto& v : complex.vertices) {
    double w2_val = v.attributes[0];
    if (std::abs(w2_val) < w2_threshold) {
        // Process parallel vector feature
        uint64_t track_id = v.track_id;
        // ...
    }
}

// 10. Write output
write_complex_to_vtu(complex, *mesh, "output.vtu");
```

---

### High-Level (Configuration-Based)

```cpp
#include <ftk2/high_level/tracking.hpp>

// 1. Load configuration from YAML
auto config = TrackingConfig::from_yaml("sujudi_haimes.yaml");

// 2. Create tracker
Tracker tracker(config);

// 3. Execute (data loaded automatically from config)
tracker.execute();

// 4. Get results
auto complex = tracker.get_complex();

// 5. Analyze
std::cout << "Vortex cores detected: " << complex.vertices.size() << "\n";

// Optional: Access filtered results
auto filtered = tracker.get_filtered_complex(
    [](const FeatureElement& el) {
        return std::abs(el.attributes[0]) < 0.1;  // W₂ filter
    }
);
```

---

### High-Level (Programmatic Configuration)

```cpp
#include <ftk2/high_level/tracking.hpp>

// 1. Build configuration programmatically
TrackingConfig config;

config.feature = FeatureType::SujadiHaimes;
config.dimension = 3;

config.input.type = InputType::Vector;
config.input.variables = {"u", "v", "w"};
config.input.field_type = "flow";

config.data.source = DataSourceType::Stream;
config.data.stream_yaml = "velocity_data.yaml";

config.options.auto_compute_vorticity = true;
config.options.w2_threshold = 0.1;
config.options.filter_mode = "absolute";

config.output.filename = "vortex_cores.vtp";

AttributeConfig w2_attr;
w2_attr.name = "w2_magnitude";
w2_attr.source = "w2";
w2_attr.type = "scalar";
config.output.attributes.push_back(w2_attr);

config.execution.backend = Backend::CUDA;
config.execution.precision = Precision::Double;

// 2. Validate
config.validate();

// 3. Execute
Tracker tracker(config);
tracker.execute();

// 4. Results
auto complex = tracker.get_complex();
```

---

### Streaming Mode (Large Datasets)

```cpp
#include <ftk2/high_level/tracking.hpp>

TrackingConfig config;
config.feature = FeatureType::SujadiHaimes;
config.dimension = 3;

config.input.type = InputType::Vector;
config.input.variables = {"u", "v", "w"};

config.data.source = DataSourceType::Stream;
config.data.stream_yaml = "large_dataset.yaml";

config.execution.backend = Backend::CUDA;

config.output.filename = "vortices_streaming.vtp";

// Enable streaming mode (2 timesteps in GPU memory)
Tracker tracker(config);
tracker.set_streaming_mode(true);
tracker.execute();

auto complex = tracker.get_complex();
```

---

## Filter Modes

### 1. Absolute Threshold

```yaml
options:
  filter_mode: absolute
  w2_threshold: 0.1  # Keep |W₂| < 0.1
```

**Use**: When you know the physical scale of parallelism.

### 2. Relative Threshold

```yaml
options:
  filter_mode: relative
  w2_threshold: 0.05  # Keep |W₂| < 0.05 * max(|W₂|)
```

**Use**: Adaptive filtering based on data range.

### 3. Percentile Threshold

```yaml
options:
  filter_mode: percentile
  filter_percentile: 0.1  # Keep top 10% most parallel
```

**Use**: When you want a fixed number of features regardless of absolute values.

### 4. No Filtering (Export All)

```yaml
options:
  w2_threshold: -1  # Negative = no filtering
```

**Use**: Export all candidates for post-processing.

---

## Attributes

### Standard Attributes

```yaml
output:
  attributes:
    # W₂ component (primary parallelism metric)
    - name: w2
      source: w2
      type: scalar

    # Absolute W₂ (always positive)
    - name: w2_abs
      source: w2
      type: abs

    # Original vector magnitudes
    - name: velocity_magnitude
      source: velocity
      type: magnitude

    - name: vorticity_magnitude
      source: vorticity
      type: magnitude

    # Individual vector components
    - name: velocity_x
      source: velocity
      type: component_0

    - name: velocity_y
      source: velocity
      type: component_1

    - name: velocity_z
      source: velocity
      type: component_2
```

### Derived Attributes (Computed)

```yaml
output:
  attributes:
    # Parallelism score: 1 / |W₂| (larger = more parallel)
    - name: parallelism
      source: w2
      type: abs_reciprocal

    # Normalized parallelism: |W₂| / (|U| * |V|)
    - name: normalized_w2
      source: w2
      type: normalized_cross_product
      sources: [velocity, vorticity]
```

---

## Performance Tuning

### CPU Execution

```yaml
execution:
  backend: cpu
  num_threads: -1  # Use all cores
  precision: float  # Faster than double
```

**Speedup**: ~4-8× with float vs double on modern CPUs.

### GPU Execution

```yaml
execution:
  backend: cuda
  precision: float
```

**Speedup**: 10-100× vs CPU for large 3D datasets.

### GPU Streaming (Memory-Constrained)

```yaml
execution:
  backend: cuda
  precision: float

# Streaming automatically enabled for datasets > GPU memory
```

**Memory**: O(1) with timesteps, constant GPU usage.

---

## Complete Examples

### Minimal Sujudi-Haimes

```yaml
tracking:
  feature: sujudi_haimes
  dimension: 3

  input:
    type: vector
    variables: [u, v, w]

  data:
    source: stream
    stream_yaml: data.yaml

  options:
    auto_compute_vorticity: true
    w2_threshold: 0.1

  output:
    filename: vortices.vtp
```

### Full-Featured ApproxPV

```yaml
tracking:
  feature: approx_parallel_vectors
  dimension: 3

  input:
    type: paired_vectors
    vector_u: electric_field
    vector_v: magnetic_field

  data:
    source: stream
    stream_yaml: em_data.yaml

  mesh:
    type: regular
    dimensions: [128, 128, 128]
    spacing: [0.1, 0.1, 0.1]

  options:
    w2_threshold: 0.05
    filter_mode: percentile
    filter_percentile: 0.05  # Top 5%

  output:
    type: traced
    filename: parallel_em.vtp
    format: vtp
    statistics: stats.json
    attributes:
      - name: w2
        source: w2
        type: scalar
      - name: e_magnitude
        source: electric_field
        type: magnitude
      - name: b_magnitude
        source: magnetic_field
        type: magnitude
      - name: poynting_vector
        source: poynting
        type: magnitude

  execution:
    backend: cuda
    precision: double
    num_threads: 8
```

---

## Advanced Usage

### Custom Post-Processing

```cpp
#include <ftk2/high_level/tracking.hpp>

auto config = TrackingConfig::from_yaml("config.yaml");
Tracker tracker(config);
tracker.execute();

auto complex = tracker.get_complex();

// Custom filtering: Keep only long trajectories with low W₂
std::map<uint64_t, std::vector<FeatureElement>> tracks;
for (const auto& v : complex.vertices) {
    tracks[v.track_id].push_back(v);
}

for (const auto& [track_id, elements] : tracks) {
    if (elements.size() < 10) continue;  // Skip short tracks

    // Compute mean |W₂| for this track
    double mean_w2 = 0;
    for (const auto& el : elements) {
        mean_w2 += std::abs(el.attributes[0]);
    }
    mean_w2 /= elements.size();

    if (mean_w2 < 0.05) {
        std::cout << "Persistent vortex core: track " << track_id
                  << ", length=" << elements.size()
                  << ", mean_w2=" << mean_w2 << "\n";
    }
}
```

### Multi-Criteria Filtering

```cpp
auto config = TrackingConfig::from_yaml("sujudi_haimes.yaml");
config.options.w2_threshold = -1;  // No filtering during tracking

Tracker tracker(config);
tracker.execute();
auto complex = tracker.get_complex();

// Filter by multiple criteria
auto filtered = tracker.get_filtered_complex(
    [](const FeatureElement& el) {
        double w2 = std::abs(el.attributes[0]);      // W₂
        double vel_mag = el.attributes[1];           // Velocity magnitude
        double vort_mag = el.attributes[2];          // Vorticity magnitude

        return (w2 < 0.1) &&                         // Good parallelism
               (vel_mag > 1.0) &&                    // Strong velocity
               (vort_mag > 5.0) &&                   // Strong rotation
               (w2 / (vel_mag * vort_mag) < 0.01);   // Normalized threshold
    }
);

write_complex_to_vtu(filtered, *mesh, "filtered_vortices.vtu");
```

---

## Migration from Legacy FTK

### Legacy FTK (ParallelVector Filter)

```cpp
// Old: Requires cubic root finding
ftk::parallel_vector_tracker_3d_regular tracker;
tracker.set_input_array_names({"u", "v", "w"}, {"omega_x", "omega_y", "omega_z"});
tracker.set_tolerance(1e-3);
tracker.run();
```

### FTK2 (ApproxPV)

```yaml
# New: Fiber-based approximation (no cubic solver)
tracking:
  feature: sujudi_haimes
  dimension: 3
  input:
    type: vector
    variables: [u, v, w]
  options:
    auto_compute_vorticity: true
    w2_threshold: 0.1  # Approximately equivalent to tolerance
  output:
    filename: vortices.vtp
```

**Advantages**:
- ✅ 10-100× faster (no cubic solver)
- ✅ GPU acceleration
- ✅ Memory-efficient streaming
- ✅ Unified API with other features
- ⚠️ Approximate (may miss some features with |W₂| near threshold)

---

## Troubleshooting

### No Features Detected

**Problem**: `complex.vertices.size() == 0`

**Solutions**:
1. **Increase threshold**: Try `w2_threshold: 1.0` or `-1` (no filtering)
2. **Check input data**: Verify U and V are not zero
3. **Verify dimensions**: Ensure mesh dimensions match data
4. **Check for NaNs**: Invalid data produces no features

### Too Many Features

**Problem**: Millions of detections

**Solutions**:
1. **Decrease threshold**: Try `w2_threshold: 0.01`
2. **Use percentile mode**: `filter_mode: percentile, filter_percentile: 0.01`
3. **Add attribute filters**: Filter by velocity/vorticity magnitude
4. **Increase grid resolution**: Coarse grids produce more noise

### High Memory Usage

**Problem**: Out of memory on GPU

**Solutions**:
1. **Enable streaming**: Automatically used for large datasets
2. **Use float precision**: `precision: float` (2× memory reduction)
3. **Process smaller domains**: Split dataset temporally
4. **Use CPU backend**: More memory available

### Performance Issues

**Problem**: Tracking too slow

**Solutions**:
1. **Use GPU**: `backend: cuda` (10-100× speedup)
2. **Use float**: `precision: float` (2-4× speedup)
3. **Reduce timesteps**: Process subset temporally
4. **Simplify attributes**: Fewer attributes = faster execution

---

## See Also

- **Low-level API**: `examples/parallel_vector_approximate.cpp`
- **Cross product utilities**: `include/ftk2/numeric/cross_product.hpp`
- **Fiber predicate**: `include/ftk2/core/predicate.hpp`
- **Configuration system**: `include/ftk2/high_level/tracking_config.hpp`
- **Legacy comparison**: `docs/FEATURE_GAP_ANALYSIS.md`
