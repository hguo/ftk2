# Attribute Recording in FTK2

## Overview

FTK2 supports flexible attribute recording at feature locations. Users can specify which fields to sample and record at critical points, contours, or other feature types. Attributes are interpolated using barycentric coordinates for accurate feature characterization.

## Motivation

When tracking features, you often want to record additional physical quantities at feature locations:

- **Critical Points**: Original scalar value of gradient field, velocity magnitude, temperature, pressure
- **Contours**: Gradient magnitude, tangent vectors, curvature
- **Fibers**: Additional scalar fields, derivatives, physical properties
- **TDGL Vortices**: Phase, amplitude, winding number

## Configuration

Attributes are specified in the `output.attributes` section of the tracking configuration YAML:

```yaml
tracking:
  output:
    trajectories: output.txt
    attributes:
      # Simple form: just source name
      - temperature
      - pressure

      # Full form: specify all details
      - name: vel_magnitude
        source: velocity
        type: magnitude

      - name: vel_u
        source: velocity
        type: scalar
        component: 0

      - name: vel_v
        source: velocity
        type: scalar
        component: 1
```

## Attribute Specification

### Simple Form

```yaml
attributes:
  - temperature  # Attribute name = source name = "temperature"
  - pressure     # Attribute name = source name = "pressure"
```

### Full Form

```yaml
attributes:
  - name: temp            # Output attribute name
    source: temperature   # Source data array name
    type: scalar          # How to extract value
    component: -1         # Component index (for multi-component arrays)
```

### Fields

- **name** (required in full form): Name in output files
- **source** (required): Source data array name from stream
- **type** (optional, default="scalar"): Extraction method
  - `scalar`: Single value (default for single-component arrays)
  - `magnitude`: Vector magnitude √(u² + v² + w²)
  - `component_X`: Specific component (alternative syntax)
- **component** (optional, default=-1): Component index for multi-component arrays
  - `-1`: Use entire array (for scalar fields)
  - `0, 1, 2, ...`: Specific component index

## Attribute Types

### 1. Scalar Fields

Record scalar values at feature locations:

```yaml
attributes:
  - temperature  # [1, nx, ny, nz, nt] array
  - pressure     # [1, nx, ny, nz, nt] array
```

**Use Case**: Recording temperature at critical points of a velocity field.

### 2. Vector Components

Record individual components of vector fields:

```yaml
attributes:
  - name: vel_u
    source: velocity
    type: scalar
    component: 0

  - name: vel_v
    source: velocity
    type: scalar
    component: 1

  - name: vel_w
    source: velocity
    type: scalar
    component: 2
```

**Use Case**: Recording velocity components (u, v, w) at critical points.

### 3. Vector Magnitude

Compute and record vector magnitude:

```yaml
attributes:
  - name: vel_magnitude
    source: velocity
    type: magnitude
```

**Formula**: √(u² + v² + w²)

**Use Case**: Recording flow speed at critical points.

### 4. Gradient Field Values

Record the original scalar value when tracking gradient critical points:

```yaml
tracking:
  feature: critical_points
  input:
    type: scalar        # Will compute gradient
    variables: [temperature]
    scalar_field: temperature  # For CP classification

  output:
    attributes:
      - temperature     # Record original scalar at CP location
```

**Use Case**: Finding critical points of ∇T and recording the temperature value.

## Complete Example

```yaml
tracking:
  feature: critical_points
  dimension: 3

  input:
    type: vector
    variables: [velocity]
    field_type: flow

  data:
    source: stream
    stream:
      filetype: synthetic
      name: tornado
      dimensions: [32, 32, 32]
      n_timesteps: 10

      vars:
        # Vector field for critical point tracking
        - name: velocity
          type: float
          components: [u, v, w]
          synthetic:
            generator: tornado

        # Additional scalar fields for attributes
        - name: temperature
          type: float
          components: [T]
          synthetic:
            generator: sine_wave
            amplitude: 100.0

        - name: pressure
          type: float
          components: [p]
          synthetic:
            generator: gaussian
            amplitude: 1000.0

  mesh:
    type: regular

  execution:
    backend: cpu
    precision: double

  output:
    trajectories: cp_output.txt
    attributes:
      # Physical properties at CP locations
      - temperature
      - pressure

      # Velocity characteristics
      - name: vel_magnitude
        source: velocity
        type: magnitude

      # Velocity components
      - name: u
        source: velocity
        component: 0

      - name: v
        source: velocity
        component: 1

      - name: w
        source: velocity
        component: 2
```

## Implementation Details

### Interpolation

Attributes are interpolated using **barycentric coordinates** computed during feature extraction:

```cpp
// For each attribute at feature location
for (int i = 0; i < num_vertices; ++i) {
    // Get value at simplex vertex i
    T vertex_value = sample_attribute(source, vertex_coords[i]);

    // Weight by barycentric coordinate
    attr_value += barycentric_coords[i] * vertex_value;
}
```

This ensures attributes are accurately sampled at the precise feature location (not just nearest vertex).

### Multi-Component Array Handling

The system automatically detects multi-component arrays by checking if the first dimension is 1-16:

```cpp
// Check if array is multi-component: [ncomp, spatial..., time]
bool is_multicomp = (array.nd() >= 2 &&
                     array.dimf(0) >= 1 &&
                     array.dimf(0) <= 16);
```

For multi-component arrays:
- `component: 0` → Extract first component
- `component: 1` → Extract second component
- `type: magnitude` → Compute √(c₀² + c₁² + ...)

### Storage

Feature elements have a fixed-size attribute array:

```cpp
struct FeatureElement {
    // ...
    float attributes[16];  // Up to 16 user-defined attributes
};
```

**Limitation**: Maximum 16 attributes per feature element.

## Output Format

### Text Format

```
# FTK2 Feature Complex
# Feature: critical_points
# Num vertices: 42
# Attributes: temperature, pressure, vel_magnitude, u, v, w

# Feature Elements:
0: track_id=1 type=0 scalar=0.0 attrs=[293.5, 1013.2, 5.43, 1.2, -3.4, 4.1]
1: track_id=1 type=0 scalar=0.0 attrs=[294.1, 1012.8, 5.67, 1.3, -3.5, 4.3]
...
```

### Future Formats

- **VTP**: Attributes as point data arrays
- **JSON**: Structured attribute objects
- **HDF5**: Efficient storage for large datasets
- **Binary**: Native FTK2 format

## Use Cases

### 1. Vortex Core Characterization

```yaml
attributes:
  - name: vorticity_magnitude
    source: vorticity
    type: magnitude
  - pressure
  - helicity
```

### 2. Temperature Extrema Analysis

```yaml
tracking:
  feature: critical_points
  input:
    type: scalar
    variables: [temperature]

output:
  attributes:
    - temperature         # Value at extremum
    - name: grad_magnitude
      source: gradient
      type: magnitude     # How steep is the extremum?
```

### 3. Isosurface Properties

```yaml
tracking:
  feature: levelsets
  options:
    threshold: 273.15  # 0°C isotherm

output:
  attributes:
    - name: grad_temp
      source: temperature_gradient
      type: magnitude   # Temperature gradient at isosurface
    - pressure          # Pressure at isosurface
```

### 4. Multi-Field Feature Detection

```yaml
tracking:
  feature: fibers
  # Track intersection of two isosurfaces

output:
  attributes:
    - field1
    - field2
    - velocity_magnitude  # Flow speed at fiber
```

## Design Rationale

### Why Barycentric Interpolation?

Features often lie **between vertices** in the mesh. Simple nearest-neighbor sampling would be inaccurate. Barycentric interpolation provides:

- **Exact feature location**: Respects the computed barycentric coordinates
- **Smooth values**: Linear interpolation within simplices
- **Consistent with FTK**: Same interpolation used for feature extraction

### Why Fixed-Size Array (16 slots)?

- **GPU compatibility**: Dynamic memory in device code is complex
- **Cache efficiency**: Fixed layout improves memory access patterns
- **Sufficient for most use cases**: 16 attributes covers typical scenarios

Future versions may support dynamic sizing for CPU-only workflows.

### Why Separate from Predicates?

Attribute recording is a **post-processing step** after feature extraction:

1. Predicate extracts feature (computes barycentric coords)
2. Engine interpolates attributes (using those coords)

This separation keeps predicates focused on feature detection logic.

## Performance Considerations

### Memory Overhead

Each feature element stores 16 floats (64 bytes) for attributes:

- **Small overhead**: Compared to trajectory data
- **Fixed size**: No dynamic allocation during tracking
- **Optional**: Only used if user specifies attributes

### Computation Cost

Attribute interpolation adds minimal overhead:

- **Linear time**: O(num_vertices) per feature
- **Simple operations**: Weighted sums
- **Cache friendly**: Sequential vertex access

**Benchmark**: <1% overhead for typical attribute counts (1-5).

## Future Extensions

### Attribute Derivation

Support computed attributes:

```yaml
attributes:
  - name: vorticity
    type: curl
    source: velocity

  - name: divergence
    type: div
    source: velocity
```

### Time Derivatives

Track rate of change:

```yaml
attributes:
  - name: temp_rate
    type: time_derivative
    source: temperature
```

### Statistics

Record local statistics:

```yaml
attributes:
  - name: temp_variance
    type: local_variance
    source: temperature
    radius: 3  # 3-cell neighborhood
```

## See Also

- `include/ftk2/core/predicate.hpp` - AttributeSpec definition
- `include/ftk2/core/engine.hpp` - Interpolation implementation
- `src/high_level/feature_tracker.cpp` - Attribute configuration
- `docs/MULTICOMPONENT_ARRAYS.md` - Data format details
- `examples/cp_with_attributes.yaml` - Complete example
