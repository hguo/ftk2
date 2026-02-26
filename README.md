# FTK2: Feature Tracking Kit 2

FTK2 is a redesigned and modernized feature tracking toolkit for scientific data analysis. It extracts, tracks, and visualizes critical points, contours, fibers, and other topological features in scalar and vector fields.

## Overview

FTK2 is a complete rewrite of [FTK](https://github.com/hguo/ftk) with:

- **Modern C++17/20** design with clean separation of concerns
- **High-level configuration-driven API** for ease of use
- **Multi-component array support** for efficient data handling
- **GPU acceleration** via CUDA for large-scale datasets
- **Flexible attribute recording** to characterize features
- **Multiple data sources**: ndarray streams (NetCDF, HDF5, ADIOS2), VTU, synthetic
- **5-layer architecture**: Low-level C++ → High-level C++ → JSON API → Python → ParaView

## Features

### Supported Feature Types

- **Critical Points**: Zeros of vector fields (gradient or flow)
  - Gradient fields: Minima, maxima, saddles (Hessian classification)
  - Flow fields: Sources, sinks, spirals (Jacobian classification)
- **Contours**: Isosurfaces of scalar fields
- **Fibers**: Intersections of multiple isosurfaces
- **Future**: Parallel vectors, critical lines, TDGL vortices, particles

### Key Capabilities

- **Spacetime tracking**: Automatic temporal tracking of features across timesteps
- **Unstructured meshes**: Support for VTU tetrahedral and triangle meshes
- **Attribute recording**: Sample and record arbitrary fields at feature locations
- **Multi-level SoS**: Simulation of Simplicity for robust feature detection
- **Memory-efficient streaming**: Process large datasets with minimal memory footprint
- **GPU acceleration**: CUDA kernels for regular and unstructured meshes

## Quick Start

### Installation

```bash
# Prerequisites
# - CMake 3.12+
# - C++17 compiler
# - CUDA (optional, for GPU support)
# - ndarray library
# - VTK 9.x (optional, for VTU support)

# Clone repository
git clone https://github.com/hguo/ftk2.git
cd ftk2

# Build
mkdir build && cd build
cmake .. \
  -Dndarray_DIR=/path/to/ndarray \
  -DVTK_DIR=/path/to/vtk \
  -DFTK_BUILD_CUDA=ON
make -j8

# Run tests
./tests/ftk2_tests
```

### Basic Usage (High-Level API)

Create a YAML configuration file:

```yaml
# tracking_config.yaml
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
      filetype: netcdf
      filename: simulation_*.nc

  mesh:
    type: regular

  execution:
    backend: cuda
    precision: double

  output:
    trajectories: critical_points.vtp
    format: vtp
    attributes:
      - temperature
      - pressure
```

Run tracking:

```bash
./bin/ftk2_track tracking_config.yaml
```

### Programmatic Usage (Low-Level API)

```cpp
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>

// Create mesh
auto mesh = std::make_shared<RegularSimplicialMesh>(
    std::vector<uint64_t>{32, 32, 32, 10});  // nx, ny, nz, nt

// Define feature predicate
CriticalPointPredicate<3, double> predicate;
predicate.use_multicomponent = true;
predicate.vector_var_name = "velocity";

// Create engine and execute
SimplicialEngine<double, CriticalPointPredicate<3>> engine(mesh, predicate);
engine.execute(data);

// Get results
auto complex = engine.get_complex();
for (const auto& feature : complex.vertices) {
    std::cout << "Feature at " << feature.simplex
              << " type=" << feature.type << std::endl;
}
```

## Examples

See `examples/` directory:

- `examples/critical_point_2d.cpp` - 2D critical point tracking
- `examples/critical_point_3d.cpp` - 3D critical point tracking
- `examples/levelset_2d.cpp` - 2D contour tracking
- `examples/unstructured_3d_synthetic.yaml` - Unstructured mesh tracking
- `examples/cp_with_attributes.yaml` - Attribute recording example

## Documentation

- **[STRATEGIC_ROADMAP.md](docs/STRATEGIC_ROADMAP.md)** - Project phases and timeline
- **[HIGH_LEVEL_API_DESIGN.md](docs/HIGH_LEVEL_API_DESIGN.md)** - High-level API architecture
- **[MULTICOMPONENT_ARRAYS.md](docs/MULTICOMPONENT_ARRAYS.md)** - Data format and design
- **[ATTRIBUTE_RECORDING.md](docs/ATTRIBUTE_RECORDING.md)** - Recording attributes at features
- **[UNSTRUCTURED_MESH_CONFIG.md](docs/UNSTRUCTURED_MESH_CONFIG.md)** - Unstructured mesh workflows
- **[USER_INTERFACE_LAYERS.md](docs/USER_INTERFACE_LAYERS.md)** - 5-layer architecture

## Architecture

### Core Components

```
ftk2/
├── core/                    # Core tracking algorithms
│   ├── mesh.hpp            # Mesh abstractions (regular, unstructured, extruded)
│   ├── predicate.hpp       # Feature predicates (CP, contour, fiber)
│   ├── engine.hpp          # Tracking engine (CPU/CUDA)
│   ├── zero_crossing.hpp   # Zero crossing solver
│   ├── sos.hpp             # Simulation of Simplicity
│   └── feature.hpp         # Feature element structures
├── high_level/             # High-level API
│   ├── tracking_config.hpp # YAML configuration
│   ├── feature_tracker.hpp # Simplified interface
│   └── data_source.cpp     # Data loading
├── utils/                  # Utilities
│   └── vtk.hpp            # VTK I/O
└── examples/               # Usage examples
```

### Data Flow

```
Stream Data (NetCDF/HDF5/VTU)
    ↓
Multi-Component Arrays [ncomp, spatial..., time]
    ↓
Simplicial Mesh (regular/unstructured/extruded)
    ↓
Predicates (CP/Contour/Fiber)
    ↓
Engine (CPU/CUDA)
    ↓
Feature Complex (vertices + connectivity)
    ↓
Output (VTP/JSON/HDF5)
```

## Multi-Component Array Format

FTK2 uses a uniform data format throughout:

```
Format: [ncomponents, spatial_dims..., time]

Examples:
  2D scalar:  [1, nx, ny, nt]
  2D vector:  [2, nx, ny, nt]
  3D vector:  [3, nx, ny, nz, nt]
  Static:     [M, nx, ny, nz, 1]
```

**Benefits**: Memory efficiency, cache locality, cleaner API

## GPU Acceleration

FTK2 provides CUDA kernels for:

- Regular meshes: Efficient parallel extraction
- Unstructured meshes: Device-side mesh representation
- Automatic memory management and overflow handling

**Performance**: 10-100× speedup for large 3D datasets

```yaml
execution:
  backend: cuda    # Use GPU acceleration
  precision: float # Single precision for better GPU performance
```

## Attribute Recording

Record additional fields at feature locations:

```yaml
output:
  attributes:
    - temperature              # Scalar field
    - pressure                 # Another scalar
    - name: vel_magnitude     # Vector magnitude
      source: velocity
      type: magnitude
    - name: u                  # Vector component
      source: velocity
      component: 0
```

Attributes are interpolated using barycentric coordinates for accuracy.

## Integration with ndarray

FTK2 uses the [ndarray](https://github.com/hguo/ndarray) library for universal I/O:

- **NetCDF**: Climate and atmospheric data
- **HDF5**: Large-scale simulations
- **ADIOS2**: Exascale I/O
- **VTK**: Visualization data
- **Synthetic**: Built-in data generators

**No format-specific code needed** - ndarray handles all I/O.

## Comparison with FTK

| Feature | FTK (Legacy) | FTK2 |
|---------|-------------|------|
| API | Low-level only | 5 layers (low to high) |
| Configuration | JSON fragments | Complete YAML |
| Data format | Separate arrays | Multi-component |
| Mesh types | Regular, Cartesian | Regular, unstructured, extruded |
| GPU support | Limited | Full CUDA integration |
| Attribute recording | Basic | Flexible and user-defined |
| Memory efficiency | Loads all data | Streaming support |
| Code quality | Research prototype | Production-ready |

## Development Status

**Current Phase**: High-Level API Implementation

✅ **Completed**:
- Core tracking algorithms (CPU/CUDA)
- Multi-component array support
- Unstructured mesh support
- GPU acceleration
- Attribute recording system
- YAML configuration

🚧 **In Progress**:
- Streaming execution (hold only 2 timesteps)
- Gradient computation (scalar → vector preprocessing)
- Output writers (VTP, JSON, HDF5)

⏳ **Planned**:
- Python bindings
- ParaView plugin
- Additional feature types (parallel vectors, critical lines)
- Distributed computing (MPI)

## Contributing

Contributions are welcome! Please see [ROADMAP.md](docs/ROADMAP.md) for planned features.

## Citation

If you use FTK2 in your research, please cite:

```bibtex
@article{ftk2,
  title={FTK2: A Modern Feature Tracking Toolkit},
  author={Guo, Hanqi and others},
  journal={TBD},
  year={2025}
}
```

## Related Projects

- **[FTK (Legacy)](https://github.com/hguo/ftk)** - Original feature tracking toolkit
- **[ndarray](https://github.com/hguo/ndarray)** - Universal array I/O library
- **[VTK](https://vtk.org/)** - Visualization toolkit

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contact

- **Author**: Hanqi Guo
- **Repository**: https://github.com/hguo/ftk2
- **Documentation**: https://github.com/hguo/ftk2/tree/main/docs

## Acknowledgments

FTK2 is supported by:
- U.S. Department of Energy
- Argonne National Laboratory
