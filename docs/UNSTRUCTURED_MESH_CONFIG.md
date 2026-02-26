# Unstructured Mesh Configuration Guide

This document explains how to configure FTK2 for tracking features on unstructured meshes.

## Overview

Unstructured mesh tracking in FTK2 follows this workflow:
1. **Load spatial mesh** from VTU file
2. **Extrude in time** to create spacetime mesh
3. **Load time-varying data** for all spacetime vertices
4. **Track features** using simplicial decomposition

## Basic Configuration

```yaml
tracking:
  feature: critical_points
  # dimension: auto-derived from data

  input:
    type: vector
    variables: [u, v, w]
    field_type: flow

  mesh:
    type: unstructured
    mesh_file: path/to/mesh.vtu

  data:
    source: stream
    # ... data configuration

  output:
    type: traced
    filename: tracks.vtp
```

## Mesh Configuration

### 2D Unstructured Mesh

```yaml
mesh:
  type: unstructured
  mesh_file: ../tests/data/1x1.vtu  # Base 2D triangular mesh
  # Automatically extruded in time
```

**Example meshes:**
- `1x1.vtu` - 2D unit square triangulation (562 vertices, 1042 triangles)
- `2x1.vtu` - 2D rectangular domain

### 3D Unstructured Mesh

```yaml
mesh:
  type: unstructured
  mesh_file: ../tests/data/3d.vtu  # Base 3D tetrahedral mesh
  # Automatically extruded in time
```

**Example meshes:**
- `3d.vtu` - 3D tetrahedral mesh

### Mesh Requirements

- **Format**: VTK Unstructured Grid (`.vtu`)
- **Cell types**: Triangles (2D) or Tetrahedra (3D)
- **Coordinates**: Must be in PointData
- **Connectivity**: Standard VTK format

## Data Sources

### 1. Synthetic Data

Generate synthetic vector fields on mesh vertices:

```yaml
data:
  source: stream
  stream:
    substreams:
      - name: synthetic
        format: synthetic
        synthetic:
          name: moving_maximum  # or tornado, woven, etc.
          n_timesteps: 10
        vars:
          - name: velocity
            components: [u, v, w]
```

**Available synthetic patterns:**
- `moving_maximum` - Gradient field with moving maximum
- `moving_minimum` - Gradient field with moving minimum
- `moving_ramp` - Linear ramp field
- `tornado` - Vortex flow field
- `double_gyre` - Double gyre flow
- `merger` - Merging critical points

### 2. VTU Time Series

Read mesh and data from VTU files:

```yaml
data:
  source: stream
  stream:
    substreams:
      - name: vtu_series
        format: vtu
        filenames:
          - sim_t000.vtu
          - sim_t001.vtu
          - sim_t002.vtu
        # Or use pattern:
        # filename_pattern: "sim_t%03d.vtu"
        # n_timesteps: 10
        vars:
          - name: velocity
            components: [vx, vy, vz]
            vtu_name: Velocity  # PointData variable name
```

**Data location:**
- **PointData**: Preferred (values at mesh vertices)
- **CellData**: Will be interpolated to vertices if needed

### 3. NetCDF/HDF5 Data

Mesh from VTU, time-varying data from NetCDF/HDF5:

```yaml
mesh:
  type: unstructured
  mesh_file: mesh.vtu

data:
  source: stream
  stream:
    substreams:
      # NetCDF
      - name: netcdf_data
        format: netcdf
        filenames: [data.nc]
        vars:
          - name: velocity
            components: [vx, vy, vz]
            nc_name: velocity

      # HDF5
      - name: hdf5_data
        format: hdf5
        filename_pattern: "data_%04d.h5"
        n_timesteps: 10
        vars:
          - name: velocity
            components: [vx, vy, vz]
            h5_name: /fields/velocity

      # ADIOS2
      - name: adios2_data
        format: adios2
        filename: data.bp
        adios2_engine: BPFile
        vars:
          - name: velocity
            components: [vx, vy, vz]
```

**Important:** Data arrays must have the same vertex ordering as the VTU mesh.

## Complete Examples

### Example 1: 2D Synthetic Data

```yaml
# File: unstructured_2d_synthetic.yaml
tracking:
  feature: critical_points

  input:
    type: vector
    variables: [u, v]
    field_type: gradient

  data:
    source: stream
    stream:
      substreams:
        - format: synthetic
          synthetic:
            name: moving_maximum
            n_timesteps: 10
          vars:
            - name: velocity
              components: [u, v]

  mesh:
    type: unstructured
    mesh_file: ../tests/data/1x1.vtu

  execution:
    backend: cpu
    precision: double

  output:
    type: traced
    filename: 2d_tracks.vtp
```

### Example 2: 3D with Real Data

```yaml
# File: unstructured_3d_netcdf.yaml
tracking:
  feature: critical_points

  input:
    type: vector
    variables: [vx, vy, vz]
    field_type: flow

  mesh:
    type: unstructured
    mesh_file: simulation/mesh.vtu

  data:
    source: stream
    stream:
      substreams:
        - format: netcdf
          filenames:
            - simulation/velocity_t000.nc
            - simulation/velocity_t001.nc
            - simulation/velocity_t002.nc
          vars:
            - name: velocity
              components: [vx, vy, vz]
              nc_name: velocity

  execution:
    backend: cuda  # GPU acceleration
    precision: double

  output:
    type: traced
    format: vtp
    filename: 3d_tracks.vtp
```

### Example 3: Sliced Output for Real-Time Visualization

```yaml
tracking:
  feature: critical_points

  mesh:
    type: unstructured
    mesh_file: ../tests/data/3d.vtu

  data:
    source: stream
    stream:
      substreams:
        - format: synthetic
          synthetic:
            name: tornado
            n_timesteps: 20
          vars:
            - name: velocity
              components: [u, v, w]

  input:
    type: vector
    variables: [u, v, w]
    field_type: flow

  execution:
    backend: cpu
    precision: double

  output:
    type: sliced  # Output per-timestep
    format: vtp
    filename: tornado_t%04d.vtp
```

## Output Formats

### Traced Output (Trajectories)

Full feature trajectories across time:

```yaml
output:
  type: traced
  format: vtp  # or pvtp, json, binary, hdf5
  filename: tracks.vtp
```

### Sliced Output (Per-Timestep)

Features extracted at each individual timestep:

```yaml
output:
  type: sliced
  format: vtp
  filename: "slice-%04d.vtp"  # Pattern with timestep
```

**Use cases:**
- Real-time visualization during long runs
- Streaming execution with limited memory
- Per-timestep analysis

### Discrete Output (Untracked)

Raw feature points without trajectory linking:

```yaml
output:
  type: discrete
  format: vtp
  filename: features.vtp
```

## Execution Options

### CPU Backend

```yaml
execution:
  backend: cpu
  precision: double  # or float
  threads: 8  # Number of OpenMP threads
```

### GPU Backend (CUDA)

```yaml
execution:
  backend: cuda
  precision: double
  device: 0  # GPU device ID
```

**Requirements:**
- CUDA-capable GPU
- FTK2 compiled with CUDA support

### Streaming Execution (Memory-Efficient)

For large datasets, use streaming mode:

```yaml
execution:
  backend: cpu
  precision: double
  streaming: true  # Only 2 timesteps in memory
  threads: 8
```

## Advanced Options

### Field Type Classification

**Gradient fields** (Hessian classification):
- Critical point types: minimum, maximum, saddle

```yaml
input:
  field_type: gradient
```

**Flow fields** (Jacobian classification):
- Critical point types: source, sink, spiral, etc.

```yaml
input:
  field_type: flow
```

### Feature Options

```yaml
options:
  # Critical points
  sos_q: 1000000.0  # SoS perturbation

  # Levelsets
  threshold: 0.5

  # Fibers
  fiber_threshold: 0.1
```

### Post-Processing

```yaml
output:
  type: traced
  filename: tracks.vtp
  statistics: stats.json  # JSON statistics file

  # Optional post-processing
  filter:
    min_duration: 3  # Filter short trajectories
    min_length: 0.1  # Filter short spatial extent
```

## Workflow Summary

1. **Prepare mesh**: VTU file with triangles/tetrahedra
2. **Prepare data**: Time series in VTU/NetCDF/HDF5/synthetic
3. **Create config**: YAML file with mesh and data sources
4. **Run tracking**: `ftk2_track config.yaml`
5. **Visualize**: Load VTP output in ParaView

## Common Issues

### Issue: "Mesh and data vertex count mismatch"

**Solution:** Ensure data arrays have the same number of vertices as the VTU mesh.

### Issue: "Synthetic data on unstructured mesh"

**Solution:** FTK2 will generate synthetic data for all mesh vertices automatically.

### Issue: "Memory errors with large meshes"

**Solution:** Enable streaming execution:
```yaml
execution:
  streaming: true
```

## See Also

- `examples/unstructured_2d_synthetic.yaml` - 2D example
- `examples/unstructured_3d_synthetic.yaml` - 3D example
- `examples/unstructured_vtu_series.yaml` - VTU time series
- `examples/unstructured_netcdf_data.yaml` - NetCDF data source
- `HIGH_LEVEL_API_PROGRESS.md` - Implementation status
- `STREAMING_EXECUTION.md` - Streaming architecture
