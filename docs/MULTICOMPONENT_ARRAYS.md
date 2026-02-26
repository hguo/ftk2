# Multi-Component Array Architecture

## Overview

FTK2 now uses **multi-component arrays** as the standard data format throughout the core. This provides:
- **Memory efficiency**: Contiguous storage for vector components
- **Better cache locality**: Components stored together
- **Cleaner API**: Single array instead of M separate arrays
- **Flexibility**: Can store additional attributes beyond vector components

## Design Principles

### 1. All Fields are Multi-Component

**Even scalar fields** are represented with a component dimension:
```
Scalar: [1, nx, ny, nz, nt]
Vector: [M, nx, ny, nz, nt]
```

**Rationale**: Uniform interface, allows adding attributes to any field.

### 2. All Fields are Time-Varying

**Even static fields** have a time dimension:
```
Static:  [M, nx, ny, nz, 1]
Dynamic: [M, nx, ny, nz, nt]
```

**Rationale**: Simplifies core - single code path for temporal tracking.

### 3. Component Dimension is First

**Array layout**: `[ncomponents, spatial_dims..., time]`

```
2D vector field:  [2, nx, ny, nt]
3D vector field:  [3, nx, ny, nz, nt]
3D scalar field:  [1, nx, ny, nz, nt]
```

**Rationale**: Standard ndarray convention, efficient for per-component operations.

## Data Flow

### High-Level API → Core

```
Stream Data → Multi-Component → Predicates
              (no decomposition)
```

**Old way (deprecated)**:
```yaml
velocity [3, nx, ny, nz, nt]
  ↓ decompose
u [nx, ny, nz, nt]
v [nx, ny, nz, nt]
w [nx, ny, nz, nt]
  ↓
Predicate (3 separate arrays)
```

**New way (preferred)**:
```yaml
velocity [3, nx, ny, nz, nt]
  ↓ keep intact
Predicate (1 multi-component array)
```

### Predicate Data Access

**Multi-component mode**:
```cpp
// Predicate configuration
predicate.use_multicomponent = true;
predicate.vector_var_name = "velocity";

// Engine extracts values:
for (int i = 0; i <= M; ++i) {
    for (int j = 0; j < M; ++j) {
        // Access component j at vertex i
        values[i][j] = vec_array.f(j, x, y, z, t);
    }
}
```

**Legacy mode (deprecated)**:
```cpp
// Predicate configuration
predicate.use_multicomponent = false;
predicate.var_names[0] = "u";
predicate.var_names[1] = "v";
predicate.var_names[2] = "w";

// Engine extracts values from separate arrays
for (int i = 0; i <= M; ++i) {
    for (int j = 0; j < M; ++j) {
        values[i][j] = data.at(var_names[j]).f(x, y, z, t);
    }
}
```

## Predicate Roles

### Critical Point Predicate

**Role**: Find zeros of vector fields (critical points)

**Input**: Vector field only (not scalar fields)

```cpp
// For gradient-based critical points:
// 1. User provides scalar field
// 2. High-level API computes gradient → vector field
// 3. CriticalPointPredicate finds zeros of gradient

// For flow-based critical points:
// 1. User provides velocity field directly
// 2. CriticalPointPredicate finds zeros of velocity
```

**Data format**: Multi-component vector array
```
shape: [M, spatial..., time]
M = 2 for 2D, M = 3 for 3D
```

### Other Predicates

**ContourPredicate**: Scalar fields only
```
shape: [1, spatial..., time]
```

**FiberPredicate**: Multiple scalar fields
```
Multiple arrays: [1, spatial..., time] each
```

## Gradient Computation

**Location**: Preprocessing step in high-level API, NOT in predicates

```cpp
// Preprocessing (high-level/feature_tracker.cpp)
if (input.type == Scalar) {
    // Compute gradient: scalar → vector
    // Output: [M, spatial..., time]
}

// Tracking (core/predicate.hpp)
CriticalPointPredicate {
    // Only accepts vector fields
    // Does NOT compute gradients
}
```

**Methods**:
- **Regular mesh**: Finite differences (central, forward, backward)
- **Unstructured mesh**: Mesh connectivity-based gradients

## Implementation Status

### ✅ Completed

- Multi-component support in `CriticalPointPredicate`
- Engine extracts from multi-component arrays (CPU)
- High-level API keeps arrays intact (no decomposition)
- Auto-detection of multi-component vs legacy mode
- Documentation

### 🚧 In Progress

- Gradient computation (scalar → vector preprocessing)
- CUDA kernel updates for multi-component
- ContourPredicate and FiberPredicate updates

### ⏳ TODO

- Remove legacy mode (deprecate separate arrays)
- Performance benchmarks (multi-comp vs separate)
- Other predicates (parallel vectors, critical lines, etc.)

## Examples

### Example 1: Tornado Flow Field

```yaml
# Stream provides multi-component array
data:
  stream:
    vars:
      - name: velocity
        components: [u, v, w]

# Auto-loaded as [3, nx, ny, nz, nt]
# Passed to CriticalPointPredicate intact
```

### Example 2: Gradient of Scalar Field

```yaml
input:
  type: scalar
  variables: [temperature]

# High-level API preprocessing:
# 1. Load temperature: [1, nx, ny, nz, nt]
# 2. Compute gradient: [3, nx, ny, nz, nt]
# 3. Pass gradient to CriticalPointPredicate
```

### Example 3: Direct Vector Field

```yaml
input:
  type: vector
  variables: [velocity]  # Multi-component array name

# Stream provides velocity: [3, nx, ny, nz, nt]
# Passed directly to CriticalPointPredicate
# No preprocessing needed
```

## Migration Guide

### Old Code (Deprecated)

```cpp
// Old: Separate arrays
ftk::ndarray<double> u, v, w;
data["u"] = u;
data["v"] = v;
data["w"] = w;

predicate.var_names[0] = "u";
predicate.var_names[1] = "v";
predicate.var_names[2] = "w";
```

### New Code (Recommended)

```cpp
// New: Multi-component array
ftk::ndarray<double> velocity;
velocity.reshapef({3, nx, ny, nz, nt});
// Fill velocity data: velocity(component, x, y, z, t)
data["velocity"] = velocity;

predicate.use_multicomponent = true;
predicate.vector_var_name = "velocity";
```

## Performance Considerations

### Memory Layout

**Multi-component** (contiguous):
```
[u0, u1, u2, ..., v0, v1, v2, ..., w0, w1, w2, ...]
```

**Separate arrays** (scattered):
```
u: [u0, u1, u2, ...]
v: [v0, v1, v2, ...]
w: [w0, w1, w2, ...]
```

### Cache Performance

Multi-component provides better cache locality when:
- Accessing all components at one point
- Processing vertices in simplices (typical use case)

### GPU Performance

Multi-component is more efficient for:
- Coalesced memory access
- Reduced kernel launches
- Better occupancy

## See Also

- `include/ftk2/core/predicate.hpp` - Predicate definitions
- `include/ftk2/core/engine.hpp` - Engine data access
- `src/high_level/feature_tracker.cpp` - Predicate initialization
- `src/high_level/data_source.cpp` - Data loading
- `HIGH_LEVEL_API_PROGRESS.md` - Overall progress
