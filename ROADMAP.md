# FTK2 Development Roadmap

This document outlines the planned milestones and future features for the FTK2 project.

## Milestone 1: Core Simplicial Engine (Current Focus)
*   [ ] Implementation of `Mesh` abstractions (Regular, Extruded).
*   [ ] Unified `SimplicialEngine` for extraction and tracking.
*   [ ] `ZeroCrossingSolver` for basic feature types.
*   [ ] Basic `ExactPV` predicate integration.

## Milestone 2: GPU Acceleration
*   [ ] CUDA/HIP kernels for mesh traversal.
*   [ ] Parallel Union-Find on the GPU for local CCL.
*   [ ] Unified device-compatible feature structures.

## Milestone 3: Distributed-Parallel Tracking (High Priority)
Integrate the distributed-parallel CCL algorithm from arXiv:2003.02351.
*   [ ] Implement the asynchronous label propagation and merging logic.
*   [ ] Integrate with MPI for cross-node tracking.
*   [ ] Ensure scalability for extreme-scale spacetime meshes.
*   *Note: This algorithm will allow for seamless tracking across domain boundaries in large-scale simulations.*

## Milestone 4: Advanced Mesh & Feature Types
*   [ ] **Deforming Spacetime Mesh**: Implement the non-linear extrusion logic from arXiv:2309.02677 to support evolving vertex positions.
*   [ ] **Isosurface Intersections**: Add a zero-crossing predicate for finding the intersection of two 3D scalar fields ($m=2$).
*   [ ] **Support for Higher-Order Elements**: Extend simplicial elements to support quadratic and higher-order interpolation.

## Milestone 5: ParaView & Ecosystem
*   [ ] **ParaView Plugins (High Priority)**: Integration with ParaView/VTK for interactive analysis.
*   [ ] **Python Bindings (PyFTK2)**: High-level Python interface for data science workflows.
*   [ ] **Comprehensive Benchmarking**: Performance validation and optimization.
