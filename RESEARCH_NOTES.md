# FTK2: Research & Design Synthesis

This document summarizes the research insights and technical discoveries that have shaped the FTK2 architecture.

## 1. Simplicial Spacetime Framework (Guo et al. 2021)
*   **Source**: [arXiv:2011.08697](https://arxiv.org/abs/2011.08697)
*   **Key Insight**: Treat time as a $(d+1)$-th dimension and subdivide the spacetime domain into simplices.
*   **FTK2 Implementation**: The `SimplicialEngine` uses this framework to provide combinatorial guarantees for feature continuity. By finding features on $m$-simplices and connecting them via $(m+1)$-simplices, we eliminate the heuristics and ambiguities of frame-by-frame tracking.

## 2. Exact Analytical Parallel Vectors (ExactPV)
*   **Source**: [arXiv:2107.02708](https://arxiv.org/abs/2107.02708)
*   **Key Insight**: In Piecewise Linear (PL) fields, Parallel Vector (PV) features are defined by cubic rational curves. A single tetrahedron can be intersected multiple times by the same curve.
*   **FTK2 Implementation**: The `ExactPV` predicate is designed as a "Manifold Generator." It returns multiple `FeatureElement` segments per simplex. The engine's "Stitching" logic matches these segments across faces to ensure topologically exact trajectories.

## 3. Stable Feature Flow Fields (Stable FFF)
*   **Source**: [Theisel et al. 2010 (5487517)](https://ieeexplore.ieee.org/document/5487517)
*   **Key Insight**: Constructing a vector field whose streamlines are feature trajectories. "Stable" FFFs ensure that numerical integration converges toward the true feature line even if perturbed.
*   **FTK2 Implementation**: Planned as a robust alternative to zero-crossing tracking. The engine will support a `FlowPredicate` that performs simplicial integration to "snap" features to their correct topological paths in noisy data.

## 4. Distributed-Parallel CCL
*   **Source**: [arXiv:2003.02351](https://arxiv.org/abs/2003.02351)
*   **Key Insight**: Asynchronous label propagation allows for Connected Component Labeling across distributed memory boundaries without a global bottleneck.
*   **FTK2 Implementation**: High-priority roadmap item for Milestone 3. The `SimplicialEngine` will use this algorithm to perform tracking across MPI ranks, merging local track segments at domain boundaries.

## 5. Deforming Spacetime Meshes
*   **Source**: [Student's Paper 2023 (2309.02677)](https://arxiv.org/abs/2309.02677)
*   **Key Insight**: Extending the spacetime framework to evolving or moving simulation grids via non-linear extrusion.
*   **FTK2 Implementation**: Implemented via the `DeformedExtrudedSimplicialMesh` decorator. This mesh type decouples the topology (which remains simplicial) from the geometry (which evolves per layer), allowing for complex tracking scenarios like XGC blob filaments.

## 6. Recursive Extrusion & Codimension
*   **Insight**: Tracking is a transformation: **Spacetime Mesh ($M$) $\rightarrow$ Feature Complex ($C$)**.
*   **Implementation**: FTK2 uses the concept of **Codimension ($m$)** to unify all features.
    *   $m=1$: Isosurfaces (3D volume tracks in 4D).
    *   $m=2$: Magnetic Flux Vortices, 3D Parallel Vectors, **Isosurface Intersections** (2D surface tracks in 4D).
    *   $m=3$: 3D Critical Points (1D curve tracks in 4D).
*   **Recursive Extrusion**: A 2D XGC mesh can be extruded into a 3D toroidal volume and then into 4D spacetime, with the engine tracking features through the entire chain.

## 7. Unified Data IO (`ndarray`)
*   **Insight**: Leveraging a standalone, high-performance library (`hguo/ndarray`) for all scientific IO.
*   **Implementation**: FTK2's `Source` layer wraps `ftk::ndarray_stream`, ensuring zero-copy data flow and native support for NetCDF, HDF5, and ADIOS2.

## 8. Lagrangian Particle Tracing & Feature Flow Fields
*   **Insight**: Lagrangian particle tracers (as implemented for MPAS-Ocean) and Feature Flow Fields (FFF) share a common integration-based foundation.
*   **Implementation**: FTK2 will extend its simplicial engine to support **Simplicial Pathline Integration**. This allows for tracking not just zero-crossings, but also features that follow a flow field. This unified integration engine will support various data formats and mesh types, including MPAS and deforming grids.

## 9. Magnetic Flux Vortices in Simplicial Settings
*   **Insight**: Magnetic flux vortices (as described in the 2017/2015 papers) are extracted based on phase angles winding around 2-simplices, rather than simple zero-crossings.
*   **Implementation**: Planned for Milestone 4. FTK2 will extend the predicate system to handle periodic/phase data and winding number calculations.

## 10. Robustness via Quantization (SoS)
*   **Insight**: Floating-point noise near zero can cause inconsistent topological decisions, leading to "junk" triangles and holes.
*   **Implementation**: Currently using quantized `__int128` arithmetic with a $10^6$ factor for `det2` and `det3`. This ensures definitive tie-breaking in degenerate cases.
*   **Future Plan**: Transition to a formal `ftk2::FixedPoint` data type that provides unified, efficient robust predicates across CPU and GPU architectures.

## 11. The Orientation Challenge in High-D Manifolds
*   **Insight**: Inconsistent vertex ordering during the subdivision of high-dimensional prisms (e.g., in Marching Pentatopes) leads to "messy" normals in ParaView when slicing spacetime volumes.
*   **Future Fix**: Implement a global orientation invariant based on vertex ID parity or specific combinatorial rules to ensure every simplex in the feature complex has a consistent winding order relative to the manifold's normal.
