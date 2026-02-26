# FTK2 Strategic Roadmap: Establishing the Next Generation

## Vision
**FTK2 is a complete modernization of FTK**, designed for:
- **Performance:** GPU acceleration, distributed computing
- **Usability:** Python bindings, ParaView integration, clean APIs
- **Robustness:** Multi-level SoS, rigorous topology, comprehensive testing
- **Extensibility:** Modular architecture, easy to add new features

## Technical Foundation

### Simplicial Spacetime Framework
FTK2 is built on the **simplicial spacetime meshing framework** (Guo et al., TVCG 2021, [arXiv:2011.08697](https://arxiv.org/abs/2011.08697)), which treats time as a $(d+1)$-th dimension and subdivides the spacetime domain into simplices. This provides:
- **Combinatorial guarantees** for feature continuity
- **Elimination of heuristics** in frame-by-frame tracking
- **Unified handling** of all feature types through codimension ($m$)

### Codimension Framework
FTK2 uses **codimension ($m$)** to unify feature tracking:
- **$m=1$**: Isosurfaces/levelsets (3D volume tracks in 4D spacetime)
- **$m=2$**: Magnetic flux vortices, 3D parallel vectors, isosurface intersections (2D surface tracks in 4D)
- **$m=3$**: 3D critical points (1D curve tracks in 4D spacetime)

The transformation is: **Spacetime Mesh ($M$) → Feature Complex ($C$)**, where features on $m$-simplices are connected via $(m+1)$-simplices.

### Robustness via Simulation of Simplicity (SoS)
Multi-level symbolic perturbation (Edelsbrunner & Mücke, 1990) ensures:
- Definitive tie-breaking in degenerate cases
- Exactly one simplex claims each feature point
- No "junk" triangles or topological holes
- Quantized `__int128` arithmetic with $10^6$ factor

**Future:** Transition to formal `ftk2::FixedPoint` data type for unified robust predicates across CPU/GPU.

### Research Foundation
Key publications informing FTK2 architecture:
- **Core Framework**: [Guo et al. 2021, arXiv:2011.08697](https://arxiv.org/abs/2011.08697)
- **Distributed CCL**: [Guo et al. 2020, arXiv:2003.02351](https://arxiv.org/abs/2003.02351)
- **ExactPV**: [Guo et al. 2021, arXiv:2107.02708](https://arxiv.org/abs/2107.02708)
- **Magnetic Flux**: [Guo et al. 2017, IEEE TVCG](https://ieeexplore.ieee.org/document/8031581)
- **Deforming Meshes**: [Student 2023, arXiv:2309.02677](https://arxiv.org/abs/2309.02677)
- **SoS**: [Edelsbrunner & Mücke 1990](https://arxiv.org/abs/math/9410209)

See [REFERENCES.md](REFERENCES.md) for complete bibliography.

## Current State (February 2025)

### ✅ Completed (Milestone 1: Core Engine)
- [x] Unified Mesh abstraction (Regular, Extruded, Unstructured)
- [x] SimplicialEngine for feature extraction and tracking
- [x] Critical point, levelset, and fiber tracking
- [x] Multi-level SoS for robust degeneracy handling
- [x] GPU acceleration basics (CUDA kernels)
- [x] Manifold stitching and connectivity (cofaces/faces)
- [x] Clean C++17 architecture
- [x] Test suite with synthetic examples

### 🎯 Strategic Priorities

## Phase 1: VALIDATION & REAL-WORLD TESTING (Priority: CRITICAL)
**Timeline: 2-3 weeks**
**Goal: Prove FTK2 matches or exceeds FTK on real scientific data**

### 1.1 Real Data Testing
- [ ] Port FTK test cases to FTK2
- [ ] Test with real VTU files from applications:
  - [ ] CFD simulations
  - [ ] Climate/ocean data (MPAS)
  - [ ] Fusion plasma (XGC)
- [ ] Create validation benchmark suite
- [ ] Document any numerical differences

### 1.2 Performance Benchmarking
- [ ] CPU performance comparison with FTK
- [ ] GPU speedup measurements
- [ ] Memory usage profiling
- [ ] Scalability tests (mesh size, timesteps)
- [ ] Identify and optimize bottlenecks

### 1.3 Correctness Verification
- [ ] Topology validation (connected component counts)
- [ ] Numerical accuracy (CP positions)
- [ ] Edge cases (degeneracies, boundaries)
- [ ] Regression test suite

**Deliverables:**
- Validation report comparing FTK vs FTK2
- Performance benchmark results
- Test suite with 20+ real-world cases

---

## Phase 2: ECOSYSTEM INTEGRATION (Priority: HIGH)
**Timeline: 2-3 months**
**Goal: Make FTK2 accessible to the broader scientific community**

### 2.1 ParaView Plugin (HIGHEST PRIORITY)
**Why:** ParaView is the standard tool for visualization in scientific computing. This is critical for adoption.

**Tasks:**
- [ ] VTK filter wrapper for FTK2 engine
- [ ] Data conversion (VTK ↔ FTK2)
- [ ] ParaView XML for property panel
- [ ] CMake integration with ParaView
- [ ] User documentation and tutorials
- [ ] Video demonstrations

**Timeline:** 6-8 weeks
**See:** [PARAVIEW_INTEGRATION.md](PARAVIEW_INTEGRATION.md)

### 2.2 Python Bindings (HIGH PRIORITY)
**Why:** Python is the lingua franca of data science. Python bindings make FTK2 accessible to a much wider audience.

**Tasks:**
- [ ] pybind11 bindings for core types
- [ ] NumPy integration for field data
- [ ] High-level Python API
- [ ] Jupyter notebook examples
- [ ] Integration with SciPy ecosystem (xarray)
- [ ] Sphinx documentation
- [ ] PyPI package

**Timeline:** 6-8 weeks
**See:** [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md)

### 2.3 ndarray Stream Integration (HIGH PRIORITY)
**Why:** ndarray is FTK2's data pipeline to real scientific workflows. Proper integration makes FTK2 format-agnostic and production-ready.

**Tasks:**
- [ ] Implement `execute_stream()` interface
- [ ] Zero-copy data access with `get_ref()`
- [ ] YAML-driven configuration (no recompilation)
- [ ] Support NetCDF, HDF5, ADIOS2, VTK streams
- [ ] Sliding-window streaming for TB-scale datasets
- [ ] VTU stream integration (unstructured meshes)
- [ ] Performance benchmarks (memory, I/O overhead)

**Timeline:** 4-6 weeks
**See:** [NDARRAY_INTEGRATION.md](NDARRAY_INTEGRATION.md)

### 2.4 Documentation & Examples
- [ ] API documentation (Doxygen)
- [ ] User guide
- [ ] Developer guide
- [ ] Tutorial series (including YAML stream configuration)
- [ ] Gallery of examples
- [ ] Publication-quality figures

**Deliverables:**
- ParaView plugin (installable)
- PyFTK2 package on PyPI
- Stream-based FTK2 for production workflows
- Comprehensive documentation website

---

## Phase 3: DISTRIBUTED & SCALABLE TRACKING (Priority: MEDIUM-HIGH)
**Timeline: 2-3 months**
**Goal: Scale to exascale datasets with distributed-memory parallelism**

### 3.1 MPI Integration (arXiv:2003.02351)
**Distributed-Parallel Connected Component Labeling:**
- [ ] Asynchronous label propagation (no global bottleneck)
- [ ] Cross-node trajectory stitching at domain boundaries
- [ ] Merging of local track segments across MPI ranks
- [ ] Load balancing strategies for irregular feature distributions
- [ ] Ghost cell handling for boundary simplices

**Technical Approach:** Extend `SimplicialEngine` union-find to work across distributed memory, with local CCL on each rank followed by boundary reconciliation.

### 3.2 Streaming & Out-of-Core Processing
**Sliding-Window Temporal Processing:**
- [ ] Two-timestep window ($t$ and $t+1$)
- [ ] Data flow: $t+1 \rightarrow t$, load $t+2 \rightarrow$ former $t$ slot
- [ ] Integration with `ftk::ndarray_stream` for zero-copy IO (see [NDARRAY_INTEGRATION.md](NDARRAY_INTEGRATION.md))
- [ ] Support for NetCDF, HDF5, ADIOS2 backends
- [ ] Progress checkpointing for resumable tracking

**Chunked Spatial Processing:**
- [ ] Spatial domain decomposition with overlap
- [ ] Chunk-local feature extraction
- [ ] Cross-chunk trajectory stitching
- [ ] Memory-efficient data structures (minimal footprint)

**Hybrid Unstructured Processing:**
- [ ] Explicit spatial topology (connectivity tables)
- [ ] Implicit temporal extrusion (on-the-fly in kernels)
- [ ] GPU strategy: 1 thread per spatial cell, extrude to spacetime
- [ ] Minimal GPU memory footprint (data, not connectivity)

### 3.3 Advanced GPU Acceleration
- [ ] Multi-GPU support (CUDA streams, concurrent kernels)
- [ ] GPU-GPU communication (NCCL for inter-GPU transfers)
- [ ] Optimized kernels for unstructured meshes (coalesced memory access)
- [ ] Unified memory management (CUDA managed memory)
- [ ] Dynamic kernel selection (regular vs unstructured paths)

**Deliverables:**
- MPI-enabled FTK2 for clusters
- Streaming support for extreme-scale datasets
- Benchmarks on leadership-class machines (Summit, Frontier)
- Documentation for HPC users

---

## Phase 4: FEATURE COMPLETENESS (Priority: MEDIUM)

**See also**: [FEATURE_GAP_ANALYSIS.md](FEATURE_GAP_ANALYSIS.md) for detailed missing feature inventory.


**Timeline: 3-6 months**
**Goal: Match FTK's feature coverage**

### 4.1 Core Feature Types

#### Exact Analytical Parallel Vectors (ExactPV)
**Source:** [arXiv:2107.02708](https://arxiv.org/abs/2107.02708)
- [ ] Analytical cubic rational solver for PV curves
- [ ] "Manifold Generator" returning multiple `FeatureElement` segments per simplex
- [ ] Stitching logic to match segments across faces
- [ ] Topologically exact trajectories in piecewise-linear fields
- [ ] Application: Vortex detection in CFD (Sujudi-Haimes, Levy-Degani-Seginer)

#### Magnetic Flux Vortices
**Source:** [Guo et al. 2017, IEEE TVCG](https://ieeexplore.ieee.org/document/8031581)
- [ ] Phase-angle based extraction on 2-simplices
- [ ] Winding number calculations for periodic/phase data
- [ ] Robust handling of branch cuts and degeneracies
- [ ] Application: TDGL superconductor simulations, Bose-Einstein condensates

#### Stable Feature Flow Fields (Stable FFF)
**Source:** [Theisel et al. 2010](https://ieeexplore.ieee.org/document/5487517)
- [ ] Vector field whose streamlines are feature trajectories
- [ ] Numerical integration converges toward true feature even if perturbed
- [ ] `FlowPredicate` for simplicial integration
- [ ] "Snap" features to correct topological paths in noisy data
- [ ] Robust alternative to zero-crossing tracking

#### Lagrangian Particle Tracing
- [ ] Simplicial pathline integration (unified with FFF)
- [ ] Support for MPAS-Ocean and other unstructured ocean/climate models
- [ ] Deforming grid support (following moving meshes)
- [ ] Stream integration for large-scale particle sets (see [NDARRAY_INTEGRATION.md](NDARRAY_INTEGRATION.md))

#### Critical Lines
- [ ] Ridge/valley line extraction in 3D scalar fields
- [ ] Codimension $m=2$ tracking in 4D spacetime

### 4.2 Advanced Mesh Support

#### Explicit Unstructured Meshes
- [ ] CPU/GPU support for arbitrary connectivity (VTK, MPAS, XGC)
- [ ] Reading from `.vtu`, `.nc` (MPAS), XGC HDF5 formats
- [ ] Hybrid approach: explicit spatial + implicit temporal extrusion
- [ ] Efficient GPU kernels (1 thread per cell, on-the-fly extrusion)

#### Deforming Spacetime Meshes
**Source:** [arXiv:2309.02677](https://arxiv.org/abs/2309.02677)
- [ ] Non-linear extrusion for evolving/moving grids
- [ ] `DeformedExtrudedSimplicialMesh` decorator
- [ ] Decouple topology (simplicial) from geometry (evolving)
- [ ] Application: XGC blob filaments, moving climate fronts

#### Recursive Extrusion
- [ ] 2D XGC mesh → 3D toroidal volume → 4D spacetime
- [ ] Multi-stage extrusion for complex geometries
- [ ] Feature tracking through entire extrusion chain

#### Arbitrary Polyhedral Cells & AMR
- [ ] Subdivision of arbitrary polyhedra into simplices
- [ ] Adaptive Mesh Refinement (AMR) with level transitions
- [ ] Consistent orientation across refinement boundaries

### 4.3 Quality of Life
- [ ] Configuration file support (YAML/JSON)
- [ ] Progress reporting (% complete, estimated time)
- [ ] Comprehensive error handling and diagnostics
- [ ] Plugin architecture for custom feature predicates
- [ ] Unified `ftk2::FixedPoint` type (replace hard-coded SoS quantization)
- [ ] Consistent manifold orientation (global invariant based on vertex ID parity)

**Deliverables:**
- Feature parity with FTK
- Application-specific examples (CFD, climate, fusion, superconductors)
- Comprehensive test suite for each feature type

---

## Phase 5: COMMUNITY & SUSTAINABILITY (Priority: ONGOING)
**Timeline: Continuous**
**Goal: Build a sustainable open-source project**

### 5.1 Community Building
- [ ] GitHub Discussions for Q&A
- [ ] Contributing guide
- [ ] Code of conduct
- [ ] Issue templates
- [ ] Regular releases (semantic versioning)

### 5.2 Publications & Outreach
- [ ] FTK2 methodology paper (e.g., TVCG, SC)
- [ ] Tutorial at conferences (IEEE VIS, SC)
- [ ] Blog posts and use cases
- [ ] Social media presence

### 5.3 Collaboration
- [ ] Integration with other tools (VisIt, Ensight)
- [ ] Partnerships with application scientists
- [ ] Academic/industry collaborations

---

## Key Success Metrics

### Technical Metrics
- [ ] **Performance:** 10x GPU speedup over CPU
- [ ] **Scalability:** Linear weak scaling to 1000+ nodes
- [ ] **Correctness:** 100% agreement with FTK on test suite
- [ ] **Coverage:** 90% code coverage in tests

### Adoption Metrics
- [ ] **Users:** 50+ active users in first year
- [ ] **Citations:** 10+ citations in first year
- [ ] **Downloads:** 1000+ PyPI downloads/month
- [ ] **ParaView:** Plugin included in official distribution

### Community Metrics
- [ ] **Contributors:** 5+ external contributors
- [ ] **Issues:** Average response time < 3 days
- [ ] **Documentation:** 90%+ positive feedback

---

## Comparison: FTK vs FTK2

| Aspect | FTK | FTK2 |
|--------|-----|------|
| **Architecture** | C++11, template-heavy | C++17, clean modern design |
| **GPU** | Limited CUDA support | First-class GPU acceleration |
| **Python** | Basic bindings | Full-featured PyFTK2 |
| **ParaView** | Basic plugin | Polished, well-integrated |
| **Distributed** | Basic MPI | Advanced distributed CCL |
| **Documentation** | Sparse | Comprehensive |
| **Testing** | Limited | Extensive test suite |
| **Usability** | Expert-friendly | Accessible to all |

---

## Key Planning Documents

- **[FEATURE_GAP_ANALYSIS.md](FEATURE_GAP_ANALYSIS.md)** - Missing features from FTK (parallel vectors, particles, etc.)
- **[NDARRAY_INTEGRATION.md](NDARRAY_INTEGRATION.md)** - Data I/O and streaming integration
- **[HIGH_LEVEL_API_DESIGN.md](HIGH_LEVEL_API_DESIGN.md)** - Configuration-driven API design
- **[USER_INTERFACE_LAYERS.md](USER_INTERFACE_LAYERS.md)** - 5-layer user interface plan
- **[PARAVIEW_INTEGRATION.md](PARAVIEW_INTEGRATION.md)** - ParaView plugin architecture
- **[PYTHON_BINDINGS.md](PYTHON_BINDINGS.md)** - Python API design

## Next Immediate Actions (This Week)

1. **ndarray stream integration (HIGH PRIORITY):**
   - Implement `execute_stream()` skeleton
   - Test with synthetic stream (moving_extremum, tornado)
   - Validate zero-copy with memory profiler
   - See [NDARRAY_INTEGRATION.md](NDARRAY_INTEGRATION.md)

2. **Test with real VTU file:**
   - Find or create a real unstructured mesh example
   - Verify `unstructured_3d_from_vtu.cu` works correctly
   - Compare results with FTK if possible

3. **Create validation suite:**
   - Port 3-5 FTK test cases
   - Document expected results
   - Test with real datasets (NetCDF, HDF5)

4. **Start ParaView integration:**
   - Set up VTK filter skeleton
   - Test data conversion VTK ↔ FTK2
   - Integrate with ndarray streams

5. **Update documentation:**
   - Add INSTALL.md with dependencies
   - Create CONTRIBUTING.md
   - Update README.md with vision
   - Document YAML stream configuration

6. **Prepare for publication:**
   - Draft outline for FTK2 paper
   - Collect benchmark data (including stream performance)
   - Generate figures

---

## Long-Term Vision (3-5 years)

**FTK2 becomes the standard for topological feature tracking:**
- Default tool in ParaView/VisIt
- Used in production at DOE labs
- 1000+ users worldwide
- 100+ citations
- Active community of contributors
- Integration into major simulation codes

**FTK2 enables new science:**
- Real-time feature tracking during simulation
- Interactive exploration of large datasets
- Automated feature detection in data pipelines
- New topological analysis methods
