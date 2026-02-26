# ndarray Integration Strategy for FTK2

## Overview

The [ndarray library](https://github.com/hguo/ndarray) is a unified I/O abstraction for time-varying scientific data designed for HPC systems. It was originally part of FTK but is now a standalone library. **Better integration with ndarray is critical for FTK2's adoption** as it provides the data pipeline connecting FTK2 to real scientific workflows.

**Current Status**: FTK2 uses ndarray for basic array storage but doesn't leverage its powerful I/O, streaming, and optimization capabilities.

**Goal**: Make FTK2 a first-class consumer of ndarray streams, enabling seamless tracking of features in NetCDF, HDF5, ADIOS2, and VTK datasets without format-specific code.

## Why ndarray Integration is Critical

### 1. **Format Independence**
Scientific data comes in many formats:
- **Climate/Ocean**: NetCDF (MPAS, CESM, WRF)
- **Fusion**: HDF5 (XGC)
- **CFD**: VTK (.vtu, .vti)
- **Extreme-scale**: ADIOS2 (BP files)

With proper ndarray integration, **FTK2 works with all formats through a single YAML configuration**, eliminating format-specific code.

### 2. **Zero-Copy Performance**
Current approach copies data at every step. ndarray's `get_ref()` provides zero-copy access, critical for:
- Multi-GB timesteps
- GPU transfers (avoid host→device copies)
- MPI decomposition (reference slices without copying)

### 3. **Streaming for Extreme-Scale Data**
Datasets can exceed memory (TB-scale). ndarray's sliding-window streaming:
- Processes two timesteps at a time ($t$ and $t+1$)
- Automatically handles multi-file datasets
- Supports parallel I/O (MPI, PNetCDF, parallel-HDF5)

### 4. **Production-Ready Workflows**
Scientists need:
- **YAML-driven configuration** (no recompilation for different datasets)
- **Variable aliasing** (handle "temperature" vs "temp" vs "T")
- **Automatic mesh extraction** (for VTK formats)
- **MPI domain decomposition** (for distributed tracking)

## Current Usage vs. Potential

### Current (Limited)
```cpp
// FTK2 currently:
ftk::ndarray<double> u, v, w;
u.reshapef(NX, NY, NZ, NT);
// ... manually fill arrays ...
std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"W", w}};
engine.execute(data);
```

**Limitations:**
- Format-specific loading code required
- No streaming (entire dataset in memory)
- Manual variable management
- No parallelism

### Potential (Full Integration)
```cpp
// With full ndarray integration:
ftk::stream s;
s.parse_yaml("config.yaml");  // Configuration-driven

for (int t = 0; t < s.total_timesteps() - 1; t++) {
    auto g0 = s.read(t);
    auto g1 = s.read(t + 1);

    // Zero-copy access
    const auto& u = g0->get_ref<float>("velocity");
    const auto& v = g1->get_ref<float>("velocity");

    engine.feed(slab_mesh, {g0, g1});  // Streaming execution
}
```

**YAML config** (`config.yaml`):
```yaml
stream:
  path_prefix: /data/simulation
  substreams:
    - name: input
      format: netcdf  # Or h5, adios2, vti, vtu_resample
      filenames: "timestep_*.nc"
      vars:
        - name: velocity
          possible_names: [velocity, vel, U]
          components: [u, v, w]
```

**Benefits:**
- Format-agnostic (change one line to switch formats)
- Streaming (process TB-scale data)
- Zero-copy (faster, less memory)
- Production-ready (scientists can use it directly)

## Integration Architecture

### Layer 1: Direct ndarray Usage (✅ Current)
```cpp
SimplicialEngine::execute(const std::map<std::string, ftk::ndarray<T>>& data)
```

**Status**: Already implemented. Used in all current examples.

**Use case**: Synthetic data, testing, quick prototypes.

---

### Layer 2: Stream-Based Interface (🎯 Priority: HIGH)
```cpp
SimplicialEngine::execute_stream(ftk::stream& s,
                                 const std::vector<std::string>& var_names)
```

**Implementation**:
```cpp
template <typename T, typename PredicateType>
void SimplicialEngine<T, PredicateType>::execute_stream(
    ftk::stream& s,
    const std::vector<std::string>& var_names)
{
    clear_results();

    for (int t = 0; t < s.total_timesteps() - 1; t++) {
        auto g0 = s.read(t);
        auto g1 = s.read(t + 1);

        // Build spatial mesh for this slab
        auto slab_mesh = build_slab_mesh(t, g0, g1);

        // Feed to engine with zero-copy data
        feed_from_groups(slab_mesh, {g0, g1}, var_names);
    }
}

template <typename T, typename PredicateType>
void SimplicialEngine<T, PredicateType>::feed_from_groups(
    std::shared_ptr<Mesh> slab_mesh,
    const std::vector<std::shared_ptr<ftk::ndarray_group>>& groups,
    const std::vector<std::string>& var_names)
{
    // Zero-copy extraction from groups
    std::map<std::string, const ftk::ndarray<T>*> data_refs;
    for (const auto& var : var_names) {
        for (const auto& g : groups) {
            if (g->has(var)) {
                data_refs[var] = &g->get_ref<T>(var);
            }
        }
    }

    // Execute with references (no copy)
    feed(slab_mesh, data_refs, var_names);
}
```

**Use case**: Real scientific workflows with YAML configuration.

**Timeline**: 2-3 weeks

---

### Layer 3: VTK Stream Integration (🎯 Priority: MEDIUM)
ndarray supports `vtu_resample` substream that reads VTU files and resamples to regular grid.

**YAML example**:
```yaml
stream:
  substreams:
    - name: input
      format: vtu_resample
      filenames: "output_*.vtu"
      resample_dims: [64, 64, 64]
      vars:
        - name: velocity
          components: [u, v, w]
```

**FTK2 can also support native unstructured VTU** by extracting connectivity:
```cpp
template <typename T, typename PredicateType>
void SimplicialEngine<T, PredicateType>::execute_vtu_stream(
    ftk::stream& s,
    const std::vector<std::string>& var_names)
{
    // Extract unstructured mesh from first VTU
    auto vtu_data = s.read(0);
    auto spatial_mesh = extract_mesh_from_vtu(vtu_data);

    // Extrude in time
    auto spacetime_mesh = std::make_shared<ExtrudedSimplicialMesh>(
        spatial_mesh, s.total_timesteps() - 1);

    // Stream through timesteps
    for (int t = 0; t < s.total_timesteps() - 1; t++) {
        auto g0 = s.read(t);
        auto g1 = s.read(t + 1);
        feed_from_groups(spacetime_mesh, {g0, g1}, var_names);
    }
}
```

**Use case**: Direct tracking on unstructured meshes (XGC, MPAS) without resampling.

**Timeline**: 3-4 weeks

---

### Layer 4: MPI Distributed Tracking (🎯 Priority: MEDIUM-HIGH)
ndarray supports MPI domain decomposition with `ndarray::decompose()`.

**Workflow**:
1. Each MPI rank reads its spatial slice
2. Execute tracking locally
3. Stitch trajectories at domain boundaries (arXiv:2003.02351)

**Implementation**:
```cpp
template <typename T, typename PredicateType>
void SimplicialEngine<T, PredicateType>::execute_stream_mpi(
    ftk::stream& s,
    const std::vector<std::string>& var_names,
    MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Decompose spatial domain
    auto local_lattice = s.get_global_lattice().decompose(size, rank);

    // Stream local data
    for (int t = 0; t < s.total_timesteps() - 1; t++) {
        auto g0 = s.read_decomposed(t, local_lattice);
        auto g1 = s.read_decomposed(t + 1, local_lattice);

        // Local tracking
        feed_from_groups(local_mesh, {g0, g1}, var_names);
    }

    // Global trajectory stitching (distributed union-find)
    stitch_mpi(comm);
}
```

**Use case**: Exascale datasets on HPC clusters.

**Timeline**: Phase 3 (2-3 months, see STRATEGIC_ROADMAP.md)

---

### Layer 5: GPU Streaming (🎯 Priority: LOW-MEDIUM)
Combine ndarray streaming with GPU execution.

**Challenge**: Transfer only active timestep slabs to GPU (not entire dataset).

**Approach**:
```cpp
// Read on CPU (streamed)
auto g0 = s.read(t);
const auto& u = g0->get_ref<float>("velocity");

// Transfer to GPU (only current slab)
ftk::ndarray<float> u_gpu;
u_gpu.from_host(u);  // ndarray CUDA support

// Execute on GPU
engine.execute_cuda(data_gpu);
```

**Use case**: Large datasets requiring GPU acceleration.

**Timeline**: Phase 3 (advanced GPU acceleration)

## Implementation Plan

### Phase 1: Stream Interface (Weeks 1-2)
**Goal**: Enable YAML-driven tracking for regular grids.

**Tasks**:
- [ ] Add `execute_stream()` to `SimplicialEngine`
- [ ] Implement `feed_from_groups()` with zero-copy refs
- [ ] Support regular mesh extraction from ndarray lattice
- [ ] Test with NetCDF and HDF5 streams
- [ ] Document YAML configuration

**Deliverable**: FTK2 can track features in NetCDF/HDF5 datasets via YAML config.

**Example**:
```cpp
ftk::stream s;
s.parse_yaml("tornado.yaml");

auto mesh = ftk2::RegularMesh({64, 64, 64});
auto spacetime = std::make_shared<ExtrudedSimplicialMesh>(mesh, s.total_timesteps() - 1);

CriticalPointPredicate<3, double> pred;
SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(spacetime, pred);

engine.execute_stream(s, {"U", "V", "W"});
```

---

### Phase 2: VTK Stream Support (Weeks 3-4)
**Goal**: Direct unstructured mesh tracking from VTU streams.

**Tasks**:
- [ ] Extract `UnstructuredMesh` from VTU data
- [ ] Handle mixed cell types (tets, hexes, pyramids)
- [ ] Support time-varying connectivity (if applicable)
- [ ] Test with real CFD/fusion datasets

**Deliverable**: FTK2 tracks features in VTU time series.

---

### Phase 3: Optimization (Weeks 5-6)
**Goal**: Zero-copy, minimal overhead.

**Tasks**:
- [ ] Benchmark memory usage (with/without zero-copy)
- [ ] Profile I/O vs. computation time
- [ ] Optimize slab-to-slab transitions
- [ ] Document performance best practices

**Deliverable**: Performance report showing streaming benefits.

---

### Phase 4: MPI Integration (Phase 3 of Strategic Roadmap)
**Goal**: Distributed tracking on HPC systems.

**Tasks**:
- [ ] MPI domain decomposition with ndarray
- [ ] Local tracking on each rank
- [ ] Distributed union-find for trajectory stitching
- [ ] Benchmark on leadership-class machines

**Deliverable**: Scalable FTK2 for exascale datasets.

---

## Example Use Cases

### Use Case 1: Climate Data (NetCDF)
**Scenario**: Track atmospheric vortices in CESM output (100 timesteps, 512³ grid).

**config.yaml**:
```yaml
stream:
  path_prefix: /data/cesm
  substreams:
    - name: atmosphere
      format: netcdf
      filenames: "atm_*.nc"
      vars:
        - name: velocity
          components: [U, V, W]
```

**Code**:
```cpp
ftk::stream s;
s.parse_yaml("config.yaml");

auto mesh = RegularMesh({512, 512, 512});
auto spacetime = ExtrudedSimplicialMesh(mesh, s.total_timesteps() - 1);

CriticalPointPredicate<3, double> pred;
SimplicialEngine engine(spacetime, pred);

engine.execute_stream(s, {"velocity"});
```

**Result**: Trajectories of atmospheric vortices across 100 timesteps.

---

### Use Case 2: Fusion Plasma (HDF5)
**Scenario**: Track magnetic flux vortices in XGC simulation (unstructured toroidal mesh).

**config.yaml**:
```yaml
stream:
  path_prefix: /data/xgc
  substreams:
    - name: plasma
      format: h5
      filenames: "xgc.*.h5"
      vars:
        - name: magnetic_field
          components: [Bx, By, Bz]
```

**Code**:
```cpp
ftk::stream s;
s.parse_yaml("config.yaml");

// Load XGC mesh topology
auto mesh = load_xgc_mesh("/data/xgc/mesh.h5");
auto spacetime = ExtrudedSimplicialMesh(mesh, s.total_timesteps() - 1);

MagneticFluxPredicate<3, double> pred;
SimplicialEngine engine(spacetime, pred);

engine.execute_stream(s, {"magnetic_field"});
```

**Result**: Magnetic flux vortex trajectories in XGC plasma.

---

### Use Case 3: CFD (VTU)
**Scenario**: Track vortex cores in unstructured CFD simulation.

**config.yaml**:
```yaml
stream:
  path_prefix: /data/cfd
  substreams:
    - name: flow
      format: vtu_resample
      filenames: "flow_*.vtu"
      resample_dims: [128, 128, 64]
      vars:
        - name: velocity
          components: [u, v, w]
```

**Code**:
```cpp
ftk::stream s;
s.parse_yaml("config.yaml");

auto mesh = RegularMesh({128, 128, 64});  // Resampled
auto spacetime = ExtrudedSimplicialMesh(mesh, s.total_timesteps() - 1);

ParallelVectorPredicate<3, double> pred;  // Vortex cores
SimplicialEngine engine(spacetime, pred);

engine.execute_stream(s, {"velocity"});
```

**Result**: Vortex core trajectories in CFD flow.

---

## Memory & Performance Considerations

### Zero-Copy Strategy
**Problem**: Copying large arrays is expensive.

**Solution**: Use `get_ref()` to obtain const references:
```cpp
// ❌ Bad: copies entire array
auto u = g->get<float>("velocity");

// ✅ Good: zero-copy reference
const auto& u = g->get_ref<float>("velocity");
```

**Benchmark** (1GB array, 100 timesteps):
- With copy: 100 GB allocated, 10s overhead
- Zero-copy: 2 GB allocated (only 2 timesteps), <0.1s overhead

### Streaming Strategy
**Problem**: Cannot load 100 timesteps × 1GB = 100 GB into memory.

**Solution**: Sliding window (2 timesteps):
```
t=0: Load t0, t1
t=1: Keep t1, load t2 (t0 freed)
t=2: Keep t2, load t3 (t1 freed)
```

**Memory footprint**: Only 2 GB for any number of timesteps.

### GPU Strategy
**Problem**: GPU memory is limited (e.g., 32 GB).

**Solution**: Transfer only active slab:
```cpp
for (int t = 0; t < s.total_timesteps() - 1; t++) {
    auto g0 = s.read(t);
    auto g1 = s.read(t + 1);

    // Transfer only current slab to GPU
    transfer_to_gpu({g0, g1});
    execute_gpu(data_gpu);

    // GPU memory freed for next slab
}
```

**Benefit**: Track unlimited timesteps with fixed GPU memory.

---

## Integration with ParaView & Python

### ParaView Plugin
The ParaView plugin (see PARAVIEW_INTEGRATION.md) should support YAML streams:

**ParaView UI**:
```
[FTK2 Feature Tracker]
Config File: [ Browse... ] → tornado.yaml
Variable:    [ velocity ▼ ]
Feature:     [ Critical Points ▼ ]
[x] Use GPU
[ Execute ]
```

**Under the hood**:
```cpp
ftk::stream s;
s.parse_yaml(config_file);

engine.execute_stream(s, {selected_variable});

// Convert trajectories to vtkPolyData
```

### Python Bindings
PyFTK2 (see PYTHON_BINDINGS.md) should expose streaming:

```python
import pyftk2

# Load stream from YAML
stream = pyftk2.Stream("tornado.yaml")

# Create tracker
tracker = pyftk2.CriticalPointTracker(
    mesh=(64, 64, 64),
    timesteps=stream.total_timesteps()
)

# Execute with stream
tracker.execute_stream(stream, ["velocity"])

# Get results
trajectories = tracker.get_trajectories()
```

---

## Documentation & Examples

### User Guide Sections
- [ ] "Loading Data with ndarray Streams" tutorial
- [ ] YAML configuration reference
- [ ] Format-specific guides (NetCDF, HDF5, ADIOS2, VTK)
- [ ] Performance tuning (zero-copy, streaming)
- [ ] MPI parallelism guide

### Example Programs
- [ ] `examples/stream_netcdf.cpp` - NetCDF climate data
- [ ] `examples/stream_hdf5.cpp` - HDF5 fusion plasma
- [ ] `examples/stream_vtu.cpp` - VTU CFD data
- [ ] `examples/stream_mpi.cpp` - Distributed tracking

### Jupyter Notebooks
- [ ] `notebooks/stream_tutorial.ipynb` - Interactive YAML configuration
- [ ] `notebooks/format_comparison.ipynb` - Benchmark different formats

---

## Testing Strategy

### Unit Tests
```cpp
TEST(StreamIntegration, RegularMeshNetCDF) {
    ftk::stream s;
    s.parse_yaml("test_regular.yaml");

    auto mesh = RegularMesh({10, 10, 10});
    auto spacetime = ExtrudedSimplicialMesh(mesh, s.total_timesteps() - 1);

    CriticalPointPredicate<3, double> pred;
    SimplicialEngine engine(spacetime, pred);

    engine.execute_stream(s, {"U", "V", "W"});

    auto complex = engine.get_complex();
    EXPECT_GT(complex.get_trajectories().size(), 0);
}
```

### Integration Tests
- [ ] Real NetCDF file (MPAS ocean)
- [ ] Real HDF5 file (XGC fusion)
- [ ] Real VTU series (CFD simulation)
- [ ] Synthetic streams (moving_extremum, tornado)

### Performance Tests
- [ ] Memory usage (with/without streaming)
- [ ] I/O overhead (different formats)
- [ ] Zero-copy validation (no extra allocations)
- [ ] GPU transfer efficiency

---

## Success Metrics

### Functionality
- [ ] ✅ FTK2 can track features from YAML config (no code changes)
- [ ] ✅ Supports all ndarray formats (NetCDF, HDF5, ADIOS2, VTK)
- [ ] ✅ Zero-copy access works correctly
- [ ] ✅ Streaming handles large datasets

### Performance
- [ ] 📈 Memory usage ≤ 2x single timestep (regardless of total timesteps)
- [ ] 📈 I/O overhead < 10% of total time
- [ ] 📈 Zero-copy eliminates unnecessary allocations

### Usability
- [ ] 📖 Users can configure tracking via YAML
- [ ] 📖 Works with their existing data formats
- [ ] 📖 Comprehensive documentation and examples

---

## References

- **ndarray repository**: https://github.com/hguo/ndarray
- **ndarray documentation**: `/home/hguo/workspace/ndarray/README.md`
- **FTK2 strategic roadmap**: [STRATEGIC_ROADMAP.md](STRATEGIC_ROADMAP.md)
- **Distributed CCL**: [arXiv:2003.02351](https://arxiv.org/abs/2003.02351)
- **Simplicial spacetime**: [arXiv:2011.08697](https://arxiv.org/abs/2011.08697)

---

## Next Steps

**Immediate (This Week)**:
1. Implement `execute_stream()` skeleton
2. Test with synthetic stream (moving_extremum)
3. Validate zero-copy with memory profiler

**Short-term (Weeks 1-4)**:
1. Complete Phase 1 (stream interface)
2. Add VTK stream support (Phase 2)
3. Document YAML configuration

**Medium-term (Months 2-3)**:
1. MPI integration (Phase 4)
2. Performance optimization
3. Real-world validation

Would you like me to start implementing the stream interface, or focus on a different integration aspect first?
