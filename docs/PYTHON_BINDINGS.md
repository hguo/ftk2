# PyFTK2: Python Bindings Strategy

## Vision
Make feature tracking accessible to data scientists and domain experts who prefer Python.

## Architecture: pybind11 Approach

### Why pybind11?
- Modern C++17 compatibility
- Automatic NumPy integration
- Clean, readable binding code
- Excellent documentation

### Module Structure
```python
import pyftk2

# Core mesh types
mesh = pyftk2.RegularMesh(dims=[128, 128, 128])
mesh = pyftk2.UnstructuredMesh.from_vtu("data.vtu")

# Field data (NumPy arrays)
vector_field = np.array(...)  # Shape: (n_vertices, 3)

# Feature tracking
tracker = pyftk2.CriticalPointTracker(mesh)
tracker.set_field(vector_field)
tracker.execute()

# Results as Python objects
trajectories = tracker.get_trajectories()
for traj in trajectories:
    print(f"Track {traj.id}: {len(traj.points)} points")
    positions = traj.positions  # NumPy array (n, 4) [x, y, z, t]
    types = traj.types          # CP types at each point
```

## Implementation Plan

### Phase 1: Core Bindings

**File: `python/pyftk2.cpp`**
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>

namespace py = pybind11;

// Wrap Mesh types
void init_mesh(py::module& m) {
    py::class_<ftk2::Mesh, std::shared_ptr<ftk2::Mesh>>(m, "Mesh")
        .def("get_num_vertices", &ftk2::Mesh::get_num_vertices)
        .def("get_spatial_dimension", &ftk2::Mesh::get_spatial_dimension);

    py::class_<ftk2::RegularSimplicialMesh, ftk2::Mesh,
               std::shared_ptr<ftk2::RegularSimplicialMesh>>(m, "RegularMesh")
        .def(py::init<const std::vector<uint64_t>&>());
}

// Wrap Engine with NumPy interface
void init_engine(py::module& m) {
    py::class_<ftk2::CriticalPointTracker>(m, "CriticalPointTracker")
        .def(py::init<std::shared_ptr<ftk2::Mesh>>())
        .def("set_field", [](ftk2::CriticalPointTracker& self,
                            py::array_t<double> u,
                            py::array_t<double> v,
                            py::array_t<double> w) {
            // Convert NumPy arrays to ndarray
            auto u_info = u.request();
            ftk::ndarray<double> u_nd;
            u_nd.from_numpy(u_info.ptr, {(size_t)u_info.shape[0]});
            // ... set fields
        })
        .def("execute", &ftk2::CriticalPointTracker::execute)
        .def("get_trajectories", [](ftk2::CriticalPointTracker& self) {
            auto complex = self.get_complex();
            py::list trajectories;
            // Convert to Python list of dicts
            return trajectories;
        });
}

PYBIND11_MODULE(pyftk2, m) {
    m.doc() = "Python bindings for FTK2 feature tracking";
    init_mesh(m);
    init_engine(m);
}
```

### Phase 2: High-Level Python API

**File: `python/ftk2/__init__.py`**
```python
"""FTK2: Feature Tracking Kit 2

High-level Python interface for topological feature tracking.
"""

from .core import RegularMesh, UnstructuredMesh
from .trackers import (
    CriticalPointTracker,
    LevelsetTracker,
    FiberTracker
)
from .io import read_vtu, write_trajectories

__version__ = "2.0.0"

def track_critical_points(mesh, vector_field, **kwargs):
    """
    Convenience function for critical point tracking.

    Parameters
    ----------
    mesh : Mesh or tuple
        Mesh object or dimensions for regular mesh
    vector_field : ndarray
        Vector field array, shape (n_vertices, 3)
    use_gpu : bool, optional
        Enable GPU acceleration (default: False)

    Returns
    -------
    trajectories : list of Trajectory
        Detected critical point trajectories

    Examples
    --------
    >>> import pyftk2
    >>> import numpy as np
    >>>
    >>> # Create regular mesh
    >>> dims = (64, 64, 64)
    >>>
    >>> # Generate synthetic field
    >>> field = generate_field(dims)
    >>>
    >>> # Track critical points
    >>> trajs = pyftk2.track_critical_points(dims, field)
    >>> print(f"Found {len(trajs)} trajectories")
    """
    if isinstance(mesh, tuple):
        mesh = RegularMesh(mesh)

    tracker = CriticalPointTracker(mesh, **kwargs)
    tracker.set_field(vector_field)
    tracker.execute()
    return tracker.get_trajectories()
```

### Phase 3: Jupyter Integration

**Notebook Examples:**
```python
# notebooks/01_getting_started.ipynb

import pyftk2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create synthetic moving vortex
dims = (32, 32, 32, 10)  # 32^3 spatial, 10 timesteps
field = pyftk2.synthetic.moving_vortex(dims, center=[16, 16, 16],
                                       velocity=[0.5, 0.5, 0.5])

# Track critical points
mesh = pyftk2.RegularMesh(dims[:3])
trajectories = pyftk2.track_critical_points(mesh, field)

# Visualize
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for traj in trajectories:
    pos = traj.positions
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=2)
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], s=100, c='green')
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], s=100, c='red')

plt.title('Critical Point Trajectories')
plt.show()
```

### Phase 4: SciPy Ecosystem Integration

```python
# Integration with xarray for labeled arrays
import xarray as xr

def track_from_xarray(dataset, vector_vars=['u', 'v', 'w'], **kwargs):
    """Track features from xarray Dataset."""
    # Extract coordinates
    mesh = create_mesh_from_coords(dataset.coords)

    # Stack vector components
    field = np.stack([dataset[v].values for v in vector_vars], axis=-1)

    # Track
    trajectories = track_critical_points(mesh, field, **kwargs)

    # Return as xarray Dataset
    return trajectories_to_xarray(trajectories)
```

## Build System

**File: `python/CMakeLists.txt`**
```cmake
find_package(pybind11 REQUIRED)

pybind11_add_module(pyftk2
    pyftk2.cpp
    mesh_bindings.cpp
    engine_bindings.cpp
    io_bindings.cpp
)

target_link_libraries(pyftk2 PRIVATE ftk2)

# Install
install(TARGETS pyftk2 DESTINATION python/ftk2)
install(DIRECTORY ftk2/ DESTINATION python/ftk2 FILES_MATCHING PATTERN "*.py")
```

**File: `python/setup.py`**
```python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(['cmake', '--build', 'build',
                              '--target', 'pyftk2'])
        super().run()

setup(
    name='pyftk2',
    version='2.0.0',
    author='Hanqi Guo',
    description='Python bindings for FTK2',
    packages=['ftk2'],
    ext_modules=[Extension('pyftk2', sources=[])],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=['numpy>=1.20', 'scipy>=1.7'],
    python_requires='>=3.8',
)
```

## Testing

**File: `python/tests/test_critical_points.py`**
```python
import pytest
import pyftk2
import numpy as np

def test_regular_mesh():
    mesh = pyftk2.RegularMesh([10, 10, 10])
    assert mesh.get_num_vertices() == 1000
    assert mesh.get_spatial_dimension() == 3

def test_synthetic_tracking():
    # Create moving maximum
    dims = (16, 16, 16)
    center = [8, 8, 8]

    # Generate field with one CP at center
    field = generate_radial_field(dims, center)

    # Track
    trajs = pyftk2.track_critical_points(dims, field)

    # Verify
    assert len(trajs) == 1
    assert trajs[0].type == 'maximum'
    np.testing.assert_allclose(trajs[0].positions[0, :3], center, atol=0.5)
```

## Documentation

- Sphinx documentation with autodoc
- Gallery of examples (sphinx-gallery)
- Jupyter notebook tutorials
- API reference
- Comparison with FTK Python bindings

## Timeline
- Week 1-2: Core bindings (mesh, engine)
- Week 3-4: High-level API and convenience functions
- Week 5-6: Jupyter integration and examples
- Week 7-8: Documentation and testing
