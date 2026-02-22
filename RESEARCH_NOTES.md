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
