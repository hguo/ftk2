# FTK2 Development Roadmap

This document outlines the planned milestones and future features for the FTK2 project.

## Milestone 1: Core Simplicial Engine (Current Focus)
*   [ ] Implementation of `Mesh` abstractions (Regular, Extruded).
*   [ ] Unified `SimplicialEngine` for extraction and tracking.
*   [ ] **Zero-Crossing Solver (Universal)**: Handle Critical Points, Isosurfaces, Magnetic Flux Vortices, and **Isosurface Intersections** ($m=1, 2, 3$).

## Milestone 2: GPU Acceleration
*   [ ] CUDA/HIP kernels for mesh traversal.
*   [ ] Parallel Union-Find on the GPU for local CCL.
*   [ ] Unified device-compatible feature structures.

## Milestone 3: Distributed-Parallel Tracking (High Priority)
Integrate the distributed-parallel CCL algorithm from arXiv:2003.02351.
*   [ ] Implement the asynchronous label propagation and merging logic.
*   [ ] Integrate with MPI for cross-node tracking.

## Milestone 4: Advanced Mesh & Feature Types
*   [ ] **ExactPV Predicate**: Analytical cubic rational solver (arXiv:2107.02708).
*   [ ] **Stable Feature Flow Fields (Stable FFF)**: High-precision tracking via stable streamline integration (IEEE TVCG 2010).
*   [ ] **Deforming Spacetime Mesh**: Non-linear extrusion (arXiv:2309.02677).

## Milestone 5: ParaView & Ecosystem
*   [ ] **ParaView Plugins (High Priority)**: Integration with ParaView/VTK for interactive analysis.
*   [ ] **Python Bindings (PyFTK2)**: High-level Python interface for data science workflows.
*   [ ] **Comprehensive Benchmarking**: Performance validation and optimization.
